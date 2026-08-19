"""
Probabilistic window-level augmentation for a frozen-encoder + MLP-head setup.

Key idea
--------
Re-extracting embeddings for the whole training set on every fold/trial is expensive.
Instead, each WINDOW (5s sub-clip) independently gets a fresh coin flip every epoch:

    - with probability `aug_prob`   -> load raw audio, augment, run through
                                        the encoder on the fly
    - with probability 1-aug_prob   -> reuse the pre-extracted CLEAN
                                        window-level embedding (no encoder call)

Windows belonging to the same clip (mix of fresh + cached) are then mean-pooled
into the clip-level embedding that actually feeds the MLP, resulting in a
two-stage pool_features pipeline (frame->window, window->clip).

Because Dataset.__getitem__ is called fresh every epoch, the augmented subset
and the augmentations themselves change every epoch automatically -- no
epoch-tracking logic required.

Setting aug_prob=0.0 reduces this exactly to the "no augmentation" baseline
(no raw audio is ever loaded, no encoder call is ever made).

Built around PipistrellePreprocessingPipeline (load/bandpass/time-expand/
window) and AudioCompose (per-window, per-transform probabilistic
augmentation). File loading only happens for a clip if at least one of its
windows got flagged this epoch -- I/O and filtering are paid once per file
regardless of window count, so there's nothing to gain from finer-grained
loading.
"""

import random
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from torch.utils.data import Dataset, DataLoader
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import average_precision_score
from avex import load_model

# Uses your existing PipistrellePreprocessingPipeline and AudioCompose classes.


# ---------------------------------------------------------------------------
# 1. MLP head - plain nn.Module, trained directly (no sklearn wrapper).
#    BatchNorm1d as the first layer normalizes using the actual batch
#    statistics during training (clean+augmented mix, whatever that epoch's
#    batch happens to contain) but tracks running mean/var for a FIXED,
#    consistent transform at eval/inference time -- no separate scaler to
#    fit, and no train/eval mismatch to keep in sync by hand.
# ---------------------------------------------------------------------------
class MLPHead(nn.Module):
    def __init__(self, input_dim=1536, hidden_dim1=128, hidden_dim2=64, num_classes=5, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# 3. Dataset. One item = one clip = several windows. Each window independently
#    gets flagged for augmentation this epoch. Loading only happens at all if
#    at least one window in the clip got flagged -- file I/O, bandpass, and
#    time-expansion in your pipeline are paid once per FILE regardless of how
#    many windows come out of it, so there's no benefit to per-window loading.
# ---------------------------------------------------------------------------
class WindowMixDataset(Dataset):
    def __init__(self, clip_df, clean_window_embeddings, labels, root_dir,
                 pipeline, augment_pipeline, aug_prob=0.2,
                 path_col="relative_path", clip_id_col="relative_path"):
        """
        clip_df:      one row per clip (the fold's training rows), with a
                       `path_col` giving the audio file's relative path
        clean_window_embeddings: {clip_id: np.ndarray[n_windows, input_dim]}
                       pre-extracted WITHOUT augmentation, one row per window,
                       in the same window order the pipeline produces
        labels:       array [n_clips, n_labels], aligned to clip_df
        root_dir:     base path for raw audio
        pipeline:     a PipistrellePreprocessingPipeline instance matching
                       the encoder (same target_sr / window_sec / overlap
                       used when the clean embeddings were extracted)
        augment_pipeline: an AudioCompose instance -- applied only to the
                       flagged windows, not the whole clip
        aug_prob:     probability that any given window is re-encoded fresh
        clip_id_col:  column used to key into clean_window_embeddings
        """
        self.clip_df = clip_df.reset_index(drop=True)
        self.clean_window_embeddings = clean_window_embeddings
        self.labels = labels
        self.root_dir = root_dir
        self.pipeline = pipeline
        self.augment_pipeline = augment_pipeline
        self.aug_prob = aug_prob
        self.path_col = path_col
        self.clip_id_col = clip_id_col

    def __len__(self):
        return len(self.clip_df)

    def __getitem__(self, idx):
        row = self.clip_df.iloc[idx]
        clip_id = row[self.clip_id_col]
        clean_embs = torch.as_tensor(self.clean_window_embeddings[clip_id], dtype=torch.float32)
        n_windows = clean_embs.shape[0]
        label = torch.as_tensor(self.labels[idx], dtype=torch.float32)

        flags = [random.random() < self.aug_prob for _ in range(n_windows)]

        if not any(flags):
            # No window flagged this epoch -> never touch raw audio.
            return {
                "clean_embeddings": clean_embs,
                "flagged_idx": [],
                "aug_windows": None,
                "label": label,
            }

        file_path = Path(self.root_dir) / Path(str(row[self.path_col]).replace("\\", "/"))
        windows = self.pipeline.forward(file_path)  # [n_windows, window_samples], raw & unaugmented

        # windows produced fresh this epoch may not exactly match n_windows
        # from the clean extraction (e.g. edge padding); guard against drift.
        n = min(n_windows, windows.shape[0])
        flagged_idx = [i for i in range(n) if flags[i]]

        flagged_raw = windows[flagged_idx]  # [K, window_samples]
        aug_windows = self.augment_pipeline(flagged_raw, target_sr=self.pipeline.target_sr)

        return {
            "clean_embeddings": clean_embs,
            "flagged_idx": flagged_idx,
            "aug_windows": aug_windows,
            "label": label,
        }


def collate_clips(batch):
    """Keeps each clip's clean embeddings intact (for the padded grid) while
    flattening all flagged raw windows across the WHOLE batch into one list,
    so they go through the encoder in a single batched call regardless of
    which clip they came from."""
    labels = torch.stack([b["label"] for b in batch])
    n_windows = [b["clean_embeddings"].shape[0] for b in batch]

    flat_raw = []  # (clip_pos, slot, waveform)
    for clip_pos, b in enumerate(batch):
        if b["aug_windows"] is not None:
            for i, slot in enumerate(b["flagged_idx"]):
                flat_raw.append((clip_pos, slot, b["aug_windows"][i]))

    return {
        "labels": labels,
        "n_windows": n_windows,
        "clean_embeddings": [b["clean_embeddings"] for b in batch],
        "flat_raw": flat_raw,
    }


# ---------------------------------------------------------------------------
# 4. Turn a collated batch into clip-level embeddings: start from each clip's
#    cached clean window embeddings, overwrite the flagged slots with a
#    freshly-encoded batch, then mean-pool across each clip's real windows.
#    Raw embeddings go in as-is -- normalization now happens inside MLPHead's
#    BatchNorm1d, not here.
# ---------------------------------------------------------------------------
def build_clip_embeddings(batch, encoder, encoder_name, device, input_dim=1536):
    n_clips = len(batch["n_windows"])
    max_windows = max(batch["n_windows"])
    window_grid = torch.zeros(n_clips, max_windows, input_dim, device=device)
    #windows are padded to the max window length within a batch [n_clips, max_windows, input_dim]
    window_mask = torch.zeros(n_clips, max_windows, device=device)
    #is 1if window real, 0 if padded [n_clips, max_windows]

    # -- start from cached clean embeddings for every clip --
    for clip_pos, embs in enumerate(batch["clean_embeddings"]):
        k = embs.shape[0]
        window_grid[clip_pos, :k] = embs.to(device)
        window_mask[clip_pos, :k] = 1

    # -- overwrite flagged slots with ONE batched augment-encode call --
    if batch["flat_raw"]:
        # Left on CPU: extract_feature does its own numpy conversion and
        # (for perch2) its own ONNX device placement via the `device` arg below;
        # forcing this onto the GPU first would just add a pointless round-trip.
        raw_batch = torch.stack([w for _, _, w in batch["flat_raw"]])

        with torch.no_grad():
            fresh_embs = embed_and_pool_window(encoder, raw_batch, encoder_name, device=device)

        for i, (clip_pos, slot, _) in enumerate(batch["flat_raw"]):
            #in the window grid, replace the extracted embedding at exact clip and window index
            window_grid[clip_pos, slot] = fresh_embs[i] 
            # mask already 1 here since it overwrites a real (clean) slot

    #how many real windows per clip
    counts = window_mask.sum(dim=1, keepdim=True).clamp(min=1)
    #mean pool embeddings across windows, ignoring padded slots (mask=0)
    clip_embeddings = (window_grid * window_mask.unsqueeze(-1)).sum(dim=1) / counts
    return clip_embeddings


# ---------------------------------------------------------------------------
# 4b. Encoder loading + window-level feature extraction, dispatched across
#     perch2 (ONNX Runtime), effnetb0, and NLM_BEATs (torch). One `encoder`
#     object and `encoder_name` string flow through the whole pipeline, so
#     switching encoders is just changing what you pass into run_cv_trials.
# ---------------------------------------------------------------------------
def load_encoder(model_name, device="cpu"):
    """Loads the specified encoder, ready for extract_feature."""
    if model_name == "perch2":
        model_path = hf_hub_download(repo_id="justinchuby/Perch-onnx", filename="perch_v2.onnx")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 8
        opts.inter_op_num_threads = 8
        providers = (
            ["CPUExecutionProvider"] if device == "cpu"
            else ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        return ort.InferenceSession(model_path, sess_options=opts, providers=providers)
    elif model_name in ("effnetb0", "NLM_BEATs"):
        # TODO: point this at your actual model-loading utility (`load_model`
        # in your snippet) -- kept as a placeholder import here.
        model_key = "esp_aves2_effnetb0_all" if model_name == "effnetb0" else "esp_aves2_naturelm_audio_v1_beats"
        encoder = load_model(model_key, device=device, return_features_only=True)
        encoder.to(device)
        return encoder
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")


def extract_feature(window, encoder, model_name, device="cpu"):
    """
    window: [K, window_samples] numpy array or torch tensor.
    Returns a torch tensor of UNPOOLED spatial/temporal features
    (i.e. before the ax=(1,2)/(1,)/(2,3) collapse below) -- same contract
    as your original extract_feature.
    """
    if isinstance(window, torch.Tensor):
        window_np = window.cpu().numpy()
    else:
        window_np = window
    window_np = window_np.astype(np.float32)

    if model_name == "perch2":
        # ONNX Runtime: plain numpy in, numpy out -- session already pinned
        # to CPU/CUDA execution provider at load_encoder time.
        out = encoder.run(["spatial_embedding"], {"inputs": window_np})[0]
        return torch.from_numpy(out)
    else:
        encoder.eval()
        with torch.no_grad():
            t_window = torch.from_numpy(window_np).to(device)
            feats = encoder(t_window)
            if isinstance(feats, dict):
                feats = feats["x"]
            return feats.cpu()


_SPATIAL_POOL_AXES = {
    "effnetb0": (2, 3),
    "NLM_BEATs": (1,),
    "perch2": (1, 2),
}


def embed_and_pool_window(encoder, raw_batch, encoder_name, device="cpu"):
    """
    raw_batch: torch tensor [K, window_samples] of augmented windows.
    Returns:   torch tensor [K, input_dim] on `device`, spatially pooled
               exactly like pool_features' non-windows branch -- i.e. this
               produces per-WINDOW embeddings, at the same level your cached
               clean_window_embeddings live at. Pooling across windows to get
               a clip-level embedding happens separately, in build_clip_embeddings.

    NOTE: the perch2 ax=(1,2) mapping assumes the ONNX export's
    "spatial_embedding" output keeps the same [K, spatial_h, spatial_w, C]
    layout as the TF Hub version. Worth a quick shape check the first time
    you run this against the ONNX model.
    """
    feats = extract_feature(raw_batch, encoder, encoder_name, device=device)
    ax = _SPATIAL_POOL_AXES[encoder_name]
    pooled = feats.mean(dim=ax)
    return pooled.to(device)
# ---------------------------------------------------------------------------
# 5. Training loop for one fold. Encoder frozen; only the MLP is trained.
# ---------------------------------------------------------------------------
def train_mlp_fold(train_clip_df, clean_window_embeddings, y_train,
                    root_dir, encoder, encoder_name, pipeline, augment_pipeline,
                    input_dim=1536, num_classes=5, aug_prob=0.2,
                    epochs=30, batch_size=16, lr=1e-3, device="cuda"):

    # Perch2 (loaded via load_encoder) is an ONNX Runtime session, not an
    # nn.Module -- only freeze/move torch-based encoders (effnetb0/NLM_BEATs).
    if isinstance(encoder, nn.Module):
        encoder.to(device).eval()
        for p in encoder.parameters():
            p.requires_grad_(False)

    mlp = MLPHead(input_dim=input_dim, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    dataset = WindowMixDataset(train_clip_df, clean_window_embeddings, y_train, root_dir,
                                pipeline=pipeline, augment_pipeline=augment_pipeline,
                                aug_prob=aug_prob)
    # drop_last=True: BatchNorm1d can't compute a variance from a batch of 1,
    # which a trailing partial batch could produce.
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         collate_fn=collate_clips, drop_last=True)

    for epoch in range(epochs):
        mlp.train()
        running_loss = 0.0
        for batch in loader:
            labels = batch["labels"].to(device)
            clip_embs = build_clip_embeddings(
                batch, encoder, encoder_name, device, input_dim=input_dim
            )
            logits = mlp(clip_embs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)

        print(f"    epoch {epoch + 1}/{epochs} - loss {running_loss / len(dataset):.4f}")

    return mlp


@torch.no_grad()
def predict_mlp(mlp, X_clip_level, device="cuda"):
    mlp.eval()
    X = torch.tensor(X_clip_level, dtype=torch.float32).to(device)
    return torch.sigmoid(mlp(X)).cpu().numpy()


# ---------------------------------------------------------------------------
# 6. Outer trial x fold loop, same shape as your original function. Run this
#    twice (aug_prob=0.2 vs aug_prob=0.0) to compare with/without augmentation.
#    aug_prob=0.0 never loads raw audio or calls the encoder, so it's exactly
#    your no-augmentation baseline, just expressed the same way.
# ---------------------------------------------------------------------------
def run_cv_trials(clip_df, clean_window_embeddings, X, y, root_dir,
                   encoder, encoder_name, pipeline, augment_pipeline,
                   label_cols, n_splits=5, num_trials=5, aug_prob=0.2,
                   random_state=42, device="cuda", **train_kwargs):

    all_results = []

    for trial in range(num_trials):
        seed = random_state + trial
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        print(f"Trial {trial + 1}/{num_trials} (aug_prob={aug_prob}, seed={seed})")

        cv = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold_scores = []
        oof_true, oof_pred, oof_idx = [], [], []

        for fold, (train_idx, test_idx) in enumerate(cv.split(clip_df, y)):
            print(f"  fold {fold + 1}/{n_splits}")
            train_df = clip_df.iloc[train_idx]
            y_train = y[train_idx]

            mlp = train_mlp_fold(
                train_df, clean_window_embeddings, y_train,
                root_dir, encoder, encoder_name, pipeline, augment_pipeline,
                aug_prob=aug_prob, device=device, **train_kwargs
            )

            y_pred = predict_mlp(mlp, X[test_idx], device=device)
            score = average_precision_score(y[test_idx], y_pred, average="macro")
            fold_scores.append(score)

            oof_true.append(y[test_idx])
            oof_pred.append(y_pred)
            oof_idx.append(test_idx)

        all_results.append({
            "trial": trial,
            "aug_prob": aug_prob,
            "mean_AP": np.mean(fold_scores),
            "std_AP": np.std(fold_scores, ddof=1),
            "oof_y_true": np.concatenate(oof_true),
            "oof_y_pred_proba": np.concatenate(oof_pred),
            "oof_indices": np.concatenate(oof_idx),
        })

    return all_results


# ---------------------------------------------------------------------------
# Usage sketch:
#   file_path = Path("/idiap/temp/adeych/data/processed_features/my_features.pkl")
#   dir = Path("/idiap/temp/adeych/data")
#   root_dir = dir / "xenocanto-dataset"
#   X_dir = dir / "feature_banks"
#   encoder_name = "perch2"   # or "effnetb0" / "NLM_BEATs" -- everything else
#                              # (pipeline, dataset, pooling, training loop)
#                              # is encoder-agnostic and just follows this string
#
#   encoder = load_encoder(encoder_name, device="cuda")
#
#   if encoder_name == "perch2":
#       target_sr = 32000 
#       pipeline = PipistrellePreprocessingPipeline(target_sr=target_sr, expansion_factor=5,window_sec =5,overlap=0.5)
#       X_bags = 
#   elif encoder_name == "effnetb0":
#       target_sr = 16000
#       pipeline = PipistrellePreprocessingPipeline(target_sr=target_sr, expansion_factor=10,window_sec =10,overlap=0.5)
#   elif encoder_name == "NLM_BEATs":
#       target_sr = 16000
#       pipeline = PipistrellePreprocessingPipeline(target_sr=target_sr, expansion_factor=10,window_sec =10,overlap=0.5)
#   else:
#       raise ValueError(f"Unsupported encoder : {encoder_name}. Please choose from 'perch2', 'effnetb0', or 'NLM_BEATs'.")
#   
#   augment_pipeline = AudioCompose([
#       (TimeMasking, 0.3),
#       (FrequencyMasking(), 0.3),
#       (TimeStretch(target_sr=target_sr), 0.3),
#       (WaveformMixup(), 0.3),
#   ])
#   clean_window_embeddings = pool_features(X_bags, windows=False,window_pooled=False,encoder = encoder_name)
#   X = pool_features(clean_window_embeddings,windows=True,encoder = encoder_name)
#   
#   label_cols = ['type_a', 'type_b', 'type_c', 'type_d', 'echo']
#   clip_df = pd.read_csv(dir / "bat_metadata.csv")  # one row per clip, with relative_path column
#
#   results_aug = run_cv_trials(
#       clip_df, clean_window_embeddings, X, y, root_dir,
#       encoder, encoder_name, pipeline, augment_pipeline,
#       label_cols, aug_prob=0.2, input_dim=X.shape[1],
#   )
#   results_noaug = run_cv_trials(
#       clip_df, clean_window_embeddings, X, y, root_dir,
#       encoder, encoder_name, pipeline, augment_pipeline,
#       label_cols, aug_prob=0.0, input_dim=X.shape[1],   # never loads audio or calls the encoder
#   )
#
#   then compare mean_AP / std_AP across trials as you already do.
# ---------------------------------------------------------------------------
