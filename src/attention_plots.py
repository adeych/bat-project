"""
Attention-weight visualization for fitted ABMIL/LSTM models, using the same
`all_results` schema produced by abmil_classifier_tuned_optuna /
abmil_classifier_tuned / abmil_classifier_quick (see mil_multilabel.py /
analysis_utils.py).

For a chosen (variant, trial, fold), this module:
  1. Recovers that outer fold's held-out TEST bags -- and their already-
     computed true labels / predicted probabilities -- directly from the
     stored result dict. No re-running of the outer CV split, and no
     recomputation of predict_proba.
  2. Samples one test bag, uniformly at random, from among those the fold's
     model both (a) has a true positive for the requested label, and (b)
     correctly predicts positive for it (predicted probability > threshold).
     Re-samples on every call.
  3. Runs the fold's fitted main ABMIL/LSTM model on that ONE bag's
     already-computed embedding (X_bags[bag_idx] -- no re-encoding of audio)
     to pull out that label's attention weights, one value per instance/
     window, via predict_abmil's return_attention machinery (which
     ABMILSklearnWrapper.predict_proba normally throws away).
  4. Re-runs ONLY the (bandpass -> time-expansion) steps of
     PipistrellePreprocessingPipeline on the corresponding audio file --
     reusing the pipeline's own methods, not a reimplementation of them --
     to get the continuous preprocessed waveform for a spectrogram panel,
     time-aligned against the same window boundaries the attention weights
     came from.
  5. Plots a two-panel figure: a linear-frequency, log-power (dB)
     spectrogram of the full recording on top, and each window's attention
     weight on the bottom, sharing a time axis. Windows overlap in time (50%
     at overlap=0.5), so most timestamps fall under TWO windows, each with
     its own independently-computed attention weight -- the bottom panel
     draws each window as a semi-transparent bar spanning its true [start,
     end) extent (so overlapping bars visibly blend) rather than a single
     interpolated line, which would silently hide that ambiguity. A marker
     at each window's center still gives the exact per-window value to read
     off precisely.

IMPORTANT CAVEATS
-----------------
- Only works for variants with pooling='attention' (ABMIL, ABMIL_LSTM,
  ABMIL_LSTM_residual*, ABMIL_no_proj, ABMIL_LSTM_no_proj*). Mean/last
  pooling variants (LSTM_only*, LSTM_residual_proj, LSTM_last, LSTM_no_proj*)
  have no attention mechanism at all -- calling this on one of those raises
  a clear ValueError rather than silently returning nothing meaningful.
- When ensemble=True was used, attention only exists for the main model's
  "complex" labels -- the ensemble's "simple" labels (e.g. type_a) are
  predicted by a plain LinearProbe with no attention mechanism. Requesting
  one of those also raises a clear ValueError.
- This assumes the SAME per-window embedding order was used to build both
  X_bags and the audio file's windows (i.e. X_bags[bag_idx][i] corresponds
  to window i of that recording) -- true as long as the encoder was run
  window-by-window over PipistrellePreprocessingPipeline's own output in
  order, which is how X_bags was originally built.
- Attention is a per-WINDOW quantity, not a per-instant one -- with
  overlapping windows, no plot can resolve which exact moment inside a
  window "caused" its weight without further assumptions. The bottom panel
  is honest about this: it shows each window's weight across its full
  extent (see point 5 above), not a false sub-window resolution.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchaudio

from src.mil_multilabel import predict_abmil


# ============================================================================
# INTERNAL HELPERS
# ============================================================================

def _find_result(all_results, variant, trial):
    """Locates the single result dict for (variant, trial) in all_results."""
    for res in all_results:
        if res['model'] == variant and res['trial'] == trial:
            return res
    raise ValueError(f"No result found for variant={variant!r}, trial={trial!r} in all_results.")


def _fold_test_data(res, fold):
    """
    Recovers, for one outer fold, the ORIGINAL bag indices (into X_bags/y)
    that were held out as that fold's test set, plus that fold's y_true and
    y_pred_proba arrays (row-aligned to those bag indices) -- all read
    straight out of the stored result dict. No re-running of the outer CV
    split, no recomputed predictions.

    Relies on 'oof_indices' being the concatenation of each fold's test_idx
    in fold order (see abmil_classifier_tuned_optuna / abmil_classifier_tuned
    / abmil_classifier_quick), so fold f's slice is recoverable purely from
    the per-fold lengths already in 'y_true_cv'.
    """
    y_true_cv = res['y_true_cv']
    if fold >= len(y_true_cv):
        raise ValueError(f"fold={fold} out of range -- this result has {len(y_true_cv)} folds.")

    start = sum(len(y_true_cv[f]) for f in range(fold))
    length = len(y_true_cv[fold])
    oof_indices = np.asarray(res['oof_indices'])

    bag_indices = oof_indices[start:start + length]
    y_true = np.asarray(res['y_true_cv'][fold])
    y_pred = np.asarray(res['y_pred_proba_cv'][fold])
    return bag_indices, y_true, y_pred


def _sample_correct_positive_bag(res, fold, label_cols, label_name, threshold=0.5):
    """
    Among the given fold's test bags, samples one bag index (into X_bags)
    uniformly at random from those where label_name is a true positive AND
    the fold's fitted model correctly predicts it positive (predicted
    probability > threshold). Re-samples on every call (no fixed seed) --
    it's fine for the same bag to be reused across different labels' plots
    if it happens to be positive for more than one.

    Returns
    -------
    bag_idx : int
        Index into X_bags / y for the sampled recording.
    pred_prob : float
        That fold's predicted probability for label_name on this bag,
        already computed during the original CV run (read from
        y_pred_proba_cv, not recomputed here).
    """
    bag_indices, y_true, y_pred = _fold_test_data(res, fold)
    label_idx = label_cols.index(label_name)

    is_positive = y_true[:, label_idx] == 1
    is_correct = y_pred[:, label_idx] > threshold
    candidates = np.where(is_positive & is_correct)[0]

    if len(candidates) == 0:
        raise ValueError(
            f"No correctly-predicted positive test bags found for label={label_name!r} "
            f"in {res['model']} (trial {res['trial']}, fold {fold}) at threshold={threshold} -- "
            f"try a different fold/trial, or lower the threshold."
        )

    chosen = np.random.choice(candidates)
    return int(bag_indices[chosen]), float(y_pred[chosen, label_idx])


def _get_attention_weights(model, X_bags, bag_idx, label_cols, label_name):
    """
    Runs the fold's fitted main ABMIL/LSTM model (never the ensemble linear
    probe, which has no attention) on a single bag's ALREADY-COMPUTED
    embedding (X_bags[bag_idx] -- no re-encoding of audio happens here) and
    returns that label's attention weights, one value per instance/window,
    in the same order as X_bags[bag_idx]'s rows.

    Handles both ensemble=True (where the main model was trained only on a
    restricted "complex" label subset, so its internal label positions don't
    match label_cols' full ordering) and ensemble=False.

    Raises
    ------
    ValueError
        If model.pooling != 'attention' (mean/last pooling variants have no
        attention weights), or if label_name is one of the ensemble's
        "simple" labels (handled by the linear probe, which has no
        attention mechanism at all).
    """
    if model.pooling != 'attention':
        raise ValueError(
            f"This model uses pooling={model.pooling!r}, which has no attention weights -- "
            f"only pooling='attention' variants (ABMIL, ABMIL_LSTM*, etc.) produce them."
        )

    label_idx = label_cols.index(label_name)

    if getattr(model, 'ensemble', False) and hasattr(model, '_complex_idx'):
        if label_idx not in model._complex_idx:
            raise ValueError(
                f"label={label_name!r} is one of this model's ensemble 'simple' labels -- "
                f"it's predicted by the linear probe, which has no attention mechanism. "
                f"Attention only exists for the complex labels: "
                f"{[label_cols[i] for i in model._complex_idx]}."
            )
        local_label_idx = model._complex_idx.index(label_idx)
    else:
        local_label_idx = label_idx

    _, attention_dict = predict_abmil(
        model.model_, [X_bags[bag_idx]], model.scaler_, device=model.device
    )
    return attention_dict[0][local_label_idx]


def _preprocessed_audio_and_windows(file_path, pipeline):
    """
    Runs the SAME steps as PipistrellePreprocessingPipeline.forward() (load
    -> bandpass -> time expansion -> windowing), reusing the pipeline
    instance's own methods -- this is exactly the same filtering/resampling
    that produced the embeddings in X_bags, not a reimplementation of it.
    Also returns the CONTINUOUS (pre-windowing) audio, since forward() only
    returns the windowed tensor and the spectrogram panel needs the
    continuous signal to plot smoothly across the whole recording.

    Returns
    -------
    audio : torch.Tensor, shape (1, T)
        Continuous preprocessed (bandpassed, time-expanded) mono audio at
        pipeline.target_sr.
    windows : torch.Tensor, shape (n_windows, win_samples)
        Same output forward() would return -- used only to sanity-check
        n_windows against the attention weight count.
    """
    audio, orig_sr = pipeline.load(file_path)
    audio = pipeline.apply_bandpass(audio, orig_sr)
    audio = pipeline.apply_time_expansion(audio, orig_sr)
    windows = pipeline.window_audio(audio)
    return audio, windows


# ============================================================================
# PUBLIC PLOTTING FUNCTIONS
# ============================================================================

def plot_attention_for_label(
    all_results, X_bags, metadata_df, label_name,
    variant, trial, fold,dataset_path,
    pipeline=None, preprocessing_cls=None,
    target_sr=32000, expansion_factor=5, window_sec=5, overlap=0.5,
    label_cols=('type_a', 'type_b', 'type_c', 'type_d', 'echo'),
    file_path_col='file_path',
    threshold=0.5,
    n_fft=1024, hop_length=256,
    figsize=(10, 6),
):
    """
    Samples one correctly-predicted, truly-positive test bag for label_name
    from (variant, trial, fold), and plots its attention weights against a
    spectrogram of the underlying audio.

    Parameters
    ----------
    all_results : list of dict
        Output of abmil_classifier_tuned_optuna / abmil_classifier_tuned /
        abmil_classifier_quick.
    X_bags : list of (n_windows, n_features) arrays
        The SAME bags array passed into the classifier function that
        produced all_results -- bag order/indices must match, since sampled
        indices are read straight from oof_indices.
    metadata_df : pd.DataFrame
        Row-aligned with X_bags/y (same order, same indices) -- must have a
        column (see file_path_col) giving each bag's source audio file path.
    label_name : str
        Which label to sample/plot attention for (must be one of label_cols,
        and must be a "complex" label if ensemble=True was used for this
        model -- see module docstring caveats).
    variant, trial, fold : the (variant, trial, outer fold) identifying
        which fitted model in all_results to use.
    pipeline : PipistrellePreprocessingPipeline instance or None
        A ready-to-use preprocessing pipeline. If given, target_sr/
        expansion_factor/window_sec/overlap below are ignored (the pipeline
        instance's own settings are used). Provide this OR preprocessing_cls.
    preprocessing_cls : type or None
        The PipistrellePreprocessingPipeline class itself (imported by the
        caller -- this module doesn't know its import path). If pipeline is
        None, one is instantiated as
        preprocessing_cls(target_sr=target_sr, expansion_factor=expansion_factor,
        window_sec=window_sec, overlap=overlap).
    target_sr, expansion_factor, window_sec, overlap : the exact settings
        used to generate the embeddings in X_bags (defaults match what was
        confirmed: 32000 / 5 / 5 / 0.5) -- only used when instantiating from
        preprocessing_cls; ignored if `pipeline` is passed directly.
    label_cols : sequence of str
        All label names, in the same column order as y / X_bags' labels.
    file_path_col : str
        Column in metadata_df holding each bag's audio file path.
    dataset_path : str or Path
        Prepended to the file_path_col value to get the full path to the
        audio file.
    threshold : float
        Predicted-probability cutoff for "correctly predicted positive".
    n_fft, hop_length : int
        torchaudio.transforms.Spectrogram parameters for the audio panel
        (linear frequency, log-power/dB scale).
    figsize : tuple
        Passed to plt.subplots.

    Returns
    -------
    dict with keys 'bag_idx', 'file_path', 'pred_prob', 'attention',
    'centers_sec' -- the sampled bag's identity, the fold model's predicted
    probability for label_name on it, the attention weight per window, and
    each window's center time in seconds (same length/order as 'attention').
    """
    label_cols = list(label_cols)
    res = _find_result(all_results, variant, trial)
    model = res['best_models'][fold]

    bag_idx, pred_prob = _sample_correct_positive_bag(res, fold, label_cols, label_name, threshold)
    attn = np.asarray(_get_attention_weights(model, X_bags, bag_idx, label_cols, label_name))

    file_path = Path(dataset_path) / Path(metadata_df.iloc[bag_idx][file_path_col].replace("\\", "/"))
    file_path = str(file_path)

    if pipeline is None:
        if preprocessing_cls is None:
            raise ValueError(
                "Pass either a ready `pipeline` instance or a `preprocessing_cls` to "
                "instantiate one from (target_sr/expansion_factor/window_sec/overlap)."
            )
        pipeline = preprocessing_cls(
            target_sr=target_sr, expansion_factor=expansion_factor,
            window_sec=window_sec, overlap=overlap,
        )

    audio, windows = _preprocessed_audio_and_windows(file_path, pipeline)

    n_windows_embed = X_bags[bag_idx].shape[0]
    if windows.shape[0] != n_windows_embed:
        print(
            f"[Warning] recomputed window count ({windows.shape[0]}) != embedding window "
            f"count ({n_windows_embed}) for {file_path} -- truncating to the shorter of the "
            f"two. This can happen if the file changed on disk since embeddings were "
            f"generated, or if target_sr/expansion_factor/window_sec/overlap here don't "
            f"match what actually produced X_bags."
        )
    n_windows = min(windows.shape[0], n_windows_embed, len(attn))
    attn = attn[:n_windows]

    win_samples = pipeline.win_samples
    hop_samples = pipeline.hop_samples
    sr = pipeline.target_sr
    centers_sec = (np.arange(n_windows) * hop_samples + win_samples / 2) / sr
    starts_sec = (np.arange(n_windows) * hop_samples) / sr

    spec_transform = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=2.0)
    spec = spec_transform(audio)  # (1, freq_bins, time_frames)
    spec_db = 10 * torch.log10(spec.clamp_min(1e-10)).squeeze(0).numpy()
    duration_sec = audio.shape[-1] / sr

    fig, (ax_spec, ax_attn) = plt.subplots(
        2, 1, figsize=figsize, sharex=True,
        gridspec_kw={'height_ratios': [2, 1]},
    )

    ax_spec.imshow(
        spec_db, origin='lower', aspect='auto', cmap='magma',
        extent=[0, duration_sec, 0, sr / 2],
    )
    ax_spec.set_ylabel("Frequency (Hz)")
    ax_spec.set_title(
        f"{variant} (trial {trial}, fold {fold}) -- attention for '{label_name}'\n"
        f"{Path(str(file_path)).name}  (predicted prob={pred_prob:.3f})"
    )

    # Bars show each window's TRUE [start, end) extent at its own attention
    # height. Since windows overlap by (1 - hop_samples/win_samples) here
    # (50% at overlap=0.5), consecutive bars overlap in x-range too -- the
    # semi-transparent fill lets that overlap show through directly, rather
    # than hiding it behind a single interpolated line, which would silently
    # imply one attention value per instant when most instants actually fall
    # under TWO windows, each with its own independently-computed weight.
    win_dur_sec = win_samples / sr
    ax_attn.bar(starts_sec, attn, width=win_dur_sec, align='edge',
                color='seagreen', alpha=0.35, edgecolor='none', zorder=1)
    # Markers at each window's center give the exact per-window value to
    # read off precisely; the connecting line is a visual aid between them,
    # not a claim about attention at in-between instants -- the bars behind
    # it are what actually represent the window's extent/overlap.
    ax_attn.plot(centers_sec, attn, marker='o', color='seagreen', zorder=2)
    ax_attn.set_ylabel("Attention weight")
    ax_attn.set_xlabel("Time (s)")

    for s in starts_sec:
        ax_spec.axvline(s, color='white', alpha=0.15, linewidth=0.6)

    plt.tight_layout()
    plt.show()

    return dict(
        bag_idx=bag_idx, file_path=file_path, pred_prob=pred_prob,
        attention=attn, centers_sec=centers_sec,
    )


def plot_attention_grid(
    all_results, X_bags, metadata_df,
    variant, trial, fold,
    labels=('type_b', 'type_c', 'type_d', 'echo'),
    **kwargs,
):
    """
    Convenience wrapper: calls plot_attention_for_label once per label in
    `labels` (default: the four ensemble-eligible complex labels), producing
    one two-panel figure per label. A label with no correctly-predicted
    positive test bags in this fold is skipped with a printed message rather
    than raising, so one missing label doesn't stop the rest of the grid.

    All other parameters (pipeline / preprocessing_cls, target_sr/
    expansion_factor/window_sec/overlap, label_cols, file_path_col,
    threshold, n_fft, hop_length, figsize) are forwarded to
    plot_attention_for_label -- see that function's docstring for details.

    Returns
    -------
    dict[str, dict] : label_name -> the dict plot_attention_for_label
        returned for that label (only present for labels that succeeded).
    """
    results = {}
    for label_name in labels:
        try:
            results[label_name] = plot_attention_for_label(
                all_results, X_bags, metadata_df, label_name,
                variant, trial, fold, **kwargs,
            )
        except ValueError as e:
            print(f"[Skipping {label_name!r}] {e}")
    return results


# ============================================================================
# USAGE EXAMPLE
# ============================================================================
#
#   from src.preprocessing import PipistrellePreprocessingPipeline  # your actual import path
#
#   # One label at a time:
#   info = plot_attention_for_label(
#       all_results, X_bags, metadata_df, label_name='type_b',
#       variant='ABMIL_LSTM', trial=0, fold=1,
#       preprocessing_cls=PipistrellePreprocessingPipeline,
#   )
#
#   # All four complex labels at once:
#   grid = plot_attention_grid(
#       all_results, X_bags, metadata_df,
#       variant='ABMIL_LSTM', trial=0, fold=1,
#       preprocessing_cls=PipistrellePreprocessingPipeline,
#   )
