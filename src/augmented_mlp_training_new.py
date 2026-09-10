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
from src.preprocessing import iterative_oversample

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
                 mixup_prob=0.2, mixup_alpha=0.2, # <-- ADDED MIXUP PARAMS
                 path_col="relative_path", clip_id_col="relative_path"):
        
        self.clip_df = clip_df.reset_index(drop=True)
        self.clean_window_embeddings = clean_window_embeddings
        self.labels = labels
        self.root_dir = root_dir
        self.pipeline = pipeline
        self.augment_pipeline = augment_pipeline
        self.aug_prob = aug_prob
        self.mixup_prob = mixup_prob
        self.mixup_alpha = mixup_alpha
        self.path_col = path_col
        self.clip_id_col = clip_id_col

    def __len__(self):
        return len(self.clip_df)

    def _load_raw_windows(self, idx):
        """Helper to get raw numpy windows for a given index."""
        row = self.clip_df.iloc[idx]
        file_path = Path(self.root_dir) / Path(str(row[self.path_col]).replace("\\", "/"))
        return self.pipeline.forward(file_path)

    def __getitem__(self, idx):
        row = self.clip_df.iloc[idx]
        clip_id = row[self.clip_id_col]
        clean_embs = torch.as_tensor(self.clean_window_embeddings[clip_id], dtype=torch.float32)
        n_windows = clean_embs.shape[0]
        label1 = torch.as_tensor(self.labels[idx], dtype=torch.float32)

        # ---------------------------------------------------------
        # WAVEFORM MIXUP BRANCH (Fixed for PyTorch Tensors)
        # ---------------------------------------------------------
        if self.mixup_prob > 0 and random.random() < self.mixup_prob and random.random() < self.aug_prob :
            # 1. Pick a random second clip
            idx2 = random.randint(0, len(self.clip_df) - 1)
            label2 = torch.as_tensor(self.labels[idx2], dtype=torch.float32)
            
            # 2. Load BOTH raw waveforms and process into windows (PyTorch tensors)
            win1 = self._load_raw_windows(idx)
            win2 = self._load_raw_windows(idx2)
            
            # Ensure they are torch tensors
            if not isinstance(win1, torch.Tensor):
                win1 = torch.from_numpy(win1)
            if not isinstance(win2, torch.Tensor):
                win2 = torch.from_numpy(win2)

            # 3. Find max length and pad the shorter tensor with zeros
            max_len = max(win1.shape[0], win2.shape[0])
            
            if win1.shape[0] < max_len:
                pad = torch.zeros((max_len - win1.shape[0], win1.shape[1]), dtype=win1.dtype)
                win1 = torch.cat([win1, pad], dim=0)
            if win2.shape[0] < max_len:
                pad = torch.zeros((max_len - win2.shape[0], win2.shape[1]), dtype=win2.dtype)
                win2 = torch.cat([win2, pad], dim=0)
                
            # 4. Mix waveforms and labels using PyTorch
            lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
            mixed_windows = lam * win1 + (1 - lam) * win2
            mixed_label = lam * label1 + (1 - lam) * label2
            
            # 5. Optionally apply single-sample augments
            mixed_windows = self.augment_pipeline(mixed_windows, target_sr=self.pipeline.target_sr)
            
            # 6. Return ALL windows as flagged, and EMPTY clean embeddings
            return {
                "clean_embeddings": torch.empty((0, clean_embs.shape[1]), dtype=torch.float32), 
                "flagged_idx": list(range(max_len)),
                "aug_windows": mixed_windows,
                "label": mixed_label,
            }
        
        # ---------------------------------------------------------
        # PARTIAL WINDOW AUGMENTATION BRANCH
        # ---------------------------------------------------------
        flags = [random.random() < self.aug_prob for _ in range(n_windows)]

        if not any(flags):
            return {
                "clean_embeddings": clean_embs,
                "flagged_idx": [],
                "aug_windows": None,
                "label": label1,
            }

        windows = self._load_raw_windows(idx)
        n = min(n_windows, windows.shape[0])
        flagged_idx = [i for i in range(n) if flags[i]]
        
        flagged_raw = windows[flagged_idx]
        aug_windows = self.augment_pipeline(flagged_raw, target_sr=self.pipeline.target_sr)

        return {
            "clean_embeddings": clean_embs,
            "flagged_idx": flagged_idx,
            "aug_windows": aug_windows,
            "label": label1,
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

    # --- DEBUG: COLLATE CHECK ---
    # Optional: only print if flat_raw isn't empty, to see when augments actually happen
    if len(flat_raw) > 0 and random.random() < 0.05: # print ~5% of the time
        print(f"[DEBUG Collate] Batch size: {len(batch)} clips. Total freshly augmented windows: {len(flat_raw)}")
    # ----------------------------

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

    # Clean cached window counts per clip
    max_clean = max(batch["n_windows"]) if batch["n_windows"] else 0
    
    # Highest raw slot index from flagged/mixup windows (+1 to get count)
    max_raw_slot = max([slot for _, slot, _ in batch["flat_raw"]]) + 1 if batch["flat_raw"] else 0
    
    # Grid size must accommodate both
    max_windows = max(max_clean, max_raw_slot)
    
    window_grid = torch.zeros(n_clips, max_windows, input_dim, device=device)
    window_mask = torch.zeros(n_clips, max_windows, device=device)

    # -- start from cached clean embeddings (Mixup clips will have shape 0 here) --
    for clip_pos, embs in enumerate(batch["clean_embeddings"]):
        k = embs.shape[0]
        if k > 0:
            window_grid[clip_pos, :k] = embs.to(device)
            window_mask[clip_pos, :k] = 1

    # -- overwrite flagged slots with ONE batched augment-encode call --
    if batch["flat_raw"]:
        raw_batch = torch.stack([w for _, _, w in batch["flat_raw"]])

        with torch.no_grad():
            fresh_embs = embed_and_pool_window(encoder, raw_batch, encoder_name, device=device)

            # --- DEBUG: EMBEDDING REPLACEMENT ---
            if not hasattr(build_clip_embeddings, "debug_done"):
                print(f"[DEBUG Build] window_grid shape: {window_grid.shape}")
                print(f"[DEBUG Build] raw_batch shape: {raw_batch.shape} -> fresh_embs shape: {fresh_embs.shape}")
                build_clip_embeddings.debug_done = True
            # ------------------------------------

        for i, (clip_pos, slot, _) in enumerate(batch["flat_raw"]):
            window_grid[clip_pos, slot] = fresh_embs[i]
            
            # Explicitly set mask to 1. 
            # (Crucial for Mixup clips where the mask started as 0)
            window_mask[clip_pos, slot] = 1 

    counts = window_mask.sum(dim=1, keepdim=True).clamp(min=1)
    clip_embeddings = (window_grid * window_mask.unsqueeze(-1)).sum(dim=1) / counts
    return clip_embeddings


# ---------------------------------------------------------------------------
# 4b. Encoder loading + window-level feature extraction, dispatched across
#     perch2 (ONNX Runtime), effnetb0, and NLM_BEATs (torch). One `encoder`
#     object and `encoder_name` string flow through the whole pipeline, so
#     switching encoders is just changing what you pass into run_cv_trials.
# ---------------------------------------------------------------------------
def load_encoder(model_name, device="cuda"):
    """Loads the specified encoder, ready for extract_feature."""
    if model_name == "perch2":
        model_path = hf_hub_download(repo_id="justinchuby/Perch-onnx", filename="perch_v2.onnx")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 8
        opts.inter_op_num_threads = 8
        cuda_options = {
            'device_id': 0,
            'gpu_mem_limit': 12 * 1024 * 1024 * 1024,  # Cap at 12 GB VRAM
            'arena_extend_strategy': 'kSameAsRequested',  # Allocates memory strictly on demand
        } 
        providers = (
            ["CPUExecutionProvider"] if device == "cpu"
            else [('CUDAExecutionProvider', cuda_options), 'CPUExecutionProvider']
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


def extract_feature(window, encoder, model_name, device="cuda",max_batch_size = 60):
    """
    window: [K, window_samples] numpy array or torch tensor.
    Returns a torch tensor of UNPOOLED spatial/temporal features
    (i.e. before the ax=(1,2)/(1,)/(2,3) collapse below) -- same contract
    as your original extract_feature.
    """
    print(f"[DEBUG] extract_feature called with model_name='{model_name}', window type={type(window)}")
    if isinstance(window, torch.Tensor):
        window_np = window.cpu().numpy()
    else:
        window_np = window
    window_np = window_np.astype(np.float32)

    total_windows = window_np.shape[0]
    if total_windows == 0:
        return torch.empty(0)

    print(f"[DEBUG] total_windows = {total_windows}")

    outputs = []

    if model_name == "perch2":
        # ONNX Runtime: plain numpy in, numpy out -- session already pinned
        # to CPU/CUDA execution provider at load_encoder time.
        # ONNX Runtime: slice and process sub-batches
        for start_idx in range(0, total_windows, max_batch_size):
            chunk = window_np[start_idx : start_idx + max_batch_size]
            out = encoder.run(["spatial_embedding"], {"inputs": chunk})[0]
            outputs.append(torch.from_numpy(out))
    else:
        encoder.eval()
        with torch.no_grad():
            for start_idx in range(0, total_windows, max_batch_size):
                chunk = window_np[start_idx : start_idx + max_batch_size]
                t_window = torch.from_numpy(chunk).to(device)
                feats = encoder(t_window)
                if isinstance(feats, dict):
                    feats = feats["x"]
                outputs.append(feats.cpu())

    return torch.cat(outputs, dim=0)


_SPATIAL_POOL_AXES = {
    "effnetb0": (2, 3),
    "NLM_BEATs": (1,),
    "perch2": (1, 2),
}


def embed_and_pool_window(encoder, raw_batch, encoder_name, device="cuda"):
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
                    mixup_prob=0.2, mixup_alpha=0.2,
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
                                aug_prob=aug_prob,mixup_prob=mixup_prob, mixup_alpha=mixup_alpha)
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

            # --- DEBUG: MLP INPUT ---
            if epoch == 0 and running_loss == 0.0: # Only print on the very first batch
                print(f"[DEBUG MLP] clip_embs shape: {clip_embs.shape} (Expected: [{labels.size(0)}, {input_dim}])")
                print(f"[DEBUG MLP] labels shape: {labels.shape}")
                if torch.isnan(clip_embs).any():
                    print("[WARNING] NaNs detected in clip_embs!")
            # ------------------------


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
                   use_oversampling = True, ovs_percentage = 0.5,
                   mixup_prob=0.2, mixup_alpha=0.2,epochs = 30,
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
        best_models = []
        oof_true, oof_pred, oof_idx = [], [], []
        train_histories, val_histories = [], []

        for fold, (train_idx, test_idx) in enumerate(cv.split(clip_df, y)):
            print(f"  fold {fold + 1}/{n_splits}")

            y_train = y[train_idx]
            train_idx_copy = train_idx

            if use_oversampling:
                # Pass integer positions 0..N-1 as dummy X
                dummy_positions = np.arange(len(train_idx))
                
                resampled_positions, y_train = iterative_oversample(
                    X=dummy_positions,
                    y=y_train,
                    target_percentage=ovs_percentage,
                    random_state=seed + fold
                )
                # Map back to real clip_df indices (with duplicates included)
                train_idx = train_idx[resampled_positions]
          
            train_df = clip_df.iloc[train_idx].reset_index(drop=True)

            # --- DEBUG: OVERSAMPLING CHECK ---
            print(f"[DEBUG Fold {fold+1}] Original train size: {len(y[train_idx_copy]) if 'train_idx_copy' in locals() else 'N/A'}, Oversampled size: {len(train_df)}")
            print(f"[DEBUG Fold {fold+1}] train_df shape: {train_df.shape}, y_train shape: {y_train.shape}")
            print(f"[DEBUG Fold {fold+1}] y_train sums: {y_train.sum(axis=0)}")
            # ---------------------------------
            

            mlp = train_mlp_fold(
                train_df, clean_window_embeddings, y_train,
                root_dir, encoder, encoder_name, pipeline, augment_pipeline,
                mixup_prob=mixup_prob, mixup_alpha=mixup_alpha,
                input_dim=X.shape[1],epochs = epochs,
                aug_prob=aug_prob, device=device, **train_kwargs
            )

            y_pred = predict_mlp(mlp, X[test_idx], device=device)
            score = average_precision_score(y[test_idx], y_pred, average="macro")
            fold_scores.append(score)

            # Save fold-level predictions and models
            best_models.append(mlp)
            oof_true.append(y[test_idx])
            oof_pred.append(y_pred)
            oof_idx.append(test_idx)

            # Extract loss history if available on the mlp instance
            train_histories.append(getattr(mlp, "train_history", []))
            val_histories.append(getattr(mlp, "val_history", []))

        all_results.append({
            "trial": trial,
            "model": f"MLP_{encoder_name}",
            "aug_prob": aug_prob,
            "best_models": best_models,

            "mean_AP": np.mean(fold_scores),
            "std_AP": np.std(fold_scores, ddof=1),

            "y_true_cv": oof_true,                 # Fold-by-fold unconcatenated list
            "y_pred_proba_cv": oof_pred,           # Fold-by-fold unconcatenated list
            "oof_y_true": np.concatenate(oof_true), # Flattened Out-Of-Fold targets
            "oof_y_pred_proba": np.concatenate(oof_pred), # Flattened Out-Of-Fold probabilities
            "oof_indices": np.concatenate(oof_idx),

            "train_histories": train_histories,
            "val_histories": val_histories
        })

    return all_results

import argparse
import os
import sys
import argparse
import os
import sys
from huggingface_hub import hf_hub_download
from src.feature_generation import extract_feature, extract_encoder
from src.preprocessing import BioacousticDataset
import time
import numpy as np
from pathlib import Path
from src.augmented_mlp_training_new import load_encoder,run_cv_trials
from src.feature_generation import pool_features
from src.preprocessing import PipistrellePreprocessingPipeline,AudioCompose,TimeMasking,FrequencyMasking,TimeStretch,WaveformMixup
import pandas as pd
from pathlib import Path
import pickle


def augmentation_experiment(encoder_name : str,augmentation,epochs = 20) :
    
    dir = Path("/idiap/temp/adeych/data")

    encoder = load_encoder(encoder_name, device="cuda")

    if encoder_name == "perch2":
       target_sr = 32000 
       pipeline = PipistrellePreprocessingPipeline(target_sr=target_sr, expansion_factor=5,window_sec =5,overlap=0.5)
       X_bags_dir = "perch2-bags.pkl"
    elif encoder_name == "effnetb0":
        target_sr = 16000
        pipeline = PipistrellePreprocessingPipeline(target_sr=target_sr, expansion_factor=10,window_sec =10,overlap=0.5)
        X_bags_dir = "effnetb0-bags.pkl"
    elif encoder_name == "NLM_BEATs":
        target_sr = 16000
        pipeline = PipistrellePreprocessingPipeline(target_sr=target_sr, expansion_factor=10,window_sec =10,overlap=0.5)
        X_bags_dir = "NLM-bags.pkl"
    else:
        raise ValueError(f"Unsupported encoder : {encoder_name}. Please choose from 'perch2', 'effnetb0', or 'NLM_BEATs'.")

    augment_pipeline = AudioCompose([
        (TimeMasking(), 0.3),
        (FrequencyMasking(), 0.3),
        (TimeStretch(target_sr=target_sr), 0.3),
    ])

    label_cols = ['type_a', 'type_b', 'type_c', 'type_d', 'echo']
    clip_df = pd.read_csv(dir / "bat_metadata.csv")  # one row per clip, with relative_path column
    root_dir = dir / "xenocanto-dataset"

    print("Started loading features")
    with open(dir / "feature_banks" / X_bags_dir, "rb") as f:
        X_bags = pickle.load(f)
    print("Finished loading features")
    window_level_list = pool_features(X_bags, windows=False,window_pooled=False,encoder = encoder_name)
    clean_window_embeddings = dict(zip(clip_df['relative_path'], window_level_list))
    X = pool_features(window_level_list,windows=True,encoder = encoder_name)

    y = np.load(dir / "feature_banks" / "labels.npy")

    all_results = run_cv_trials(clip_df, clean_window_embeddings, X, y, root_dir,
                   encoder, encoder_name, pipeline, augment_pipeline,
                   label_cols, n_splits=5, num_trials=2, aug_prob=augmentation,
                   use_oversampling = True, ovs_percentage = 0.2,
                   mixup_prob=0.3, mixup_alpha=0.2,epochs =epochs,
                   random_state=42, device="cuda")

    return all_results

# ---------------------------------------------------------------------------
# Usage sketch:
#
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
