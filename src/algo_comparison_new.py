"""
Two LaTeX comparison tables for ABMIL / ABMIL+LSTM / Mean-only / LSTM-only /
LinearProbe, built from separate `all_results` lists per model (same schema
as abmil_classifier_tuned_optuna / linear_probe_tuned_optuna: one dict per
trial, each with 'mean_AP', 'y_true_cv', 'y_pred_proba_cv', 'best_models', ...).

Table 1 ("ensemble comparison"): ABMIL / ABMIL+LSTM / Mean-only / LSTM-only
(all run with ensemble=True) plus the standalone LinearProbe baseline,
across ALL 5 labels plus a final cmAP column -- same shape as Table 2. The
one difference: since ensemble=True routes Type A through the SAME
independently-tuned linear probe regardless of which main architecture is
being compared (see ABMILSklearnWrapper.fit -- the ensemble Optuna study and
final refit are both explicitly re-seeded from self.random_state, so they
don't depend on how much RNG state the main model consumed beforehand), the
four ensembled architectures' Type A column is expected to be numerically
identical, while Type B/C/D/echo still come from each architecture's own
attention/pooling mechanism and are never merged. This is CHECKED, not
assumed -- if the four Type A values don't actually match (mismatched run
settings, or GPU/cudnn non-determinism even with matching seeds), all four
are shown separately in that column with a printed warning instead of being
silently merged. When they do match, the shared Type A value is shown ONCE
via \\multirow spanning the four ensembled rows, rather than repeated four
times; every other column (Type B/C/D/echo/cmAP) always shows each row's
own value.

Table 2 ("architecture x label comparison"): ABMIL / ABMIL+LSTM / Mean-only
/ LSTM-only, run WITHOUT ensembling (so Type A is predicted natively by
each architecture's own attention/pooling mechanism, not a shared linear
probe -- these four legitimately differ and are never merged), across all
5 labels plus a final cmAP column.

Every cell is 'mean $\\pm$ std' computed ACROSS TRIALS: for the overall
cmAP columns, that's mean/std of each trial's own `mean_AP` field (the
existing fold-averaged summary already stored per trial -- not recomputed).
For per-label columns (Type A alone in Table 1, all 5 in Table 2), there is
no such field already stored, so each trial's per-label value is recomputed
as the mean AP across that trial's OWN outer folds (from y_true_cv /
y_pred_proba_cv) -- mirroring exactly how mean_AP itself was derived
(fold-level AP, then averaged over folds), just restricted to one label
instead of macro-averaged over all of them. Mean/std are then taken across
trials, matching the ddof=1 sample-std convention used for std_AP
everywhere else in this project. A trial contributes NaN for a given label
if a fold-level AP was undefined for it (a fold with only one class present
for that label -- possible with rare labels in small folds even with
stratified CV); NaN trial contributions are dropped rather than treated as
a real value.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score


# ============================================================================
# INTERNAL HELPERS
# ============================================================================

def _escape_latex(text):
    """Escapes LaTeX special characters in a plain-text string."""
    escapes = {
        '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_',
        '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}', '\\': r'\textbackslash{}',
    }
    return ''.join(escapes.get(ch, ch) for ch in str(text))


def _format_mean_std(mean, std, decimals=3):
    """'mean $\\pm$ std', or just 'mean' if std is NaN/unavailable (e.g. a
    single trial), or '--' if mean itself is NaN/undefined."""
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return '--'
    mean_str = f"{mean:.{decimals}f}"
    if std is None or (isinstance(std, float) and np.isnan(std)):
        return mean_str
    return f"{mean_str} $\\pm$ {std:.{decimals}f}"


def _trial_overall_ap(results_list):
    """One value per trial dict: its already-stored `mean_AP` (fold-averaged
    macro-AP across whatever labels that run covers -- all 5 if
    ensemble=True, since predict_proba reassembles both sub-models; all 5
    natively if ensemble=False too)."""
    return np.array([res['mean_AP'] for res in results_list], dtype=float)


def _trial_label_ap(results_list, label_idx):
    """
    One value per trial dict: the mean, across that trial's own outer
    folds, of the AP for a single label column -- recomputed from
    y_true_cv/y_pred_proba_cv since no per-label field is stored directly.
    A fold contributes NaN (and is excluded from that trial's mean) if the
    label has only one class present in that fold's y_true.
    """
    trial_values = []
    for res in results_list:
        fold_aps = []
        for y_true_fold, y_pred_fold in zip(res['y_true_cv'], res['y_pred_proba_cv']):
            y_true_fold = np.asarray(y_true_fold)
            y_pred_fold = np.asarray(y_pred_fold)
            y_t = y_true_fold[:, label_idx]
            if y_t.sum() == 0 or y_t.sum() == len(y_t):
                continue  # undefined for this fold -- skip, don't count
            fold_aps.append(average_precision_score(y_t, y_pred_fold[:, label_idx]))
        trial_values.append(np.mean(fold_aps) if fold_aps else np.nan)
    return np.array(trial_values, dtype=float)


def _mean_std_across_trials(trial_values):
    """Mean/std (ddof=1, matching std_AP's convention elsewhere in this
    project) across trials, ignoring NaN trials. std is NaN if fewer than 2
    valid trials remain."""
    valid = trial_values[~np.isnan(trial_values)]
    if len(valid) == 0:
        return np.nan, np.nan
    mean = float(np.mean(valid))
    std = float(np.std(valid, ddof=1)) if len(valid) > 1 else np.nan
    return mean, std


def _check_ensemble_flag(results_list, expected, model_label):
    """Best-effort sanity check that a results list actually came from an
    ensemble=True/False run, via the fitted wrapper's own .ensemble
    attribute. Prints a warning rather than raising -- this is meant to
    catch an accidentally-swapped argument, not to be a hard requirement
    (e.g. LinearProbe's wrapper has no .ensemble attribute at all, and that's
    fine -- it's skipped)."""
    try:
        actual = results_list[0]['best_models'][0].ensemble
    except (KeyError, IndexError, AttributeError):
        return
    if actual != expected:
        print(f"[Warning] {model_label}: expected ensemble={expected} but the fitted "
              f"model reports ensemble={actual} -- check you passed the right results list.")


def _check_shared_type_a(values_by_model, label_idx_name='Type A', tolerance=1e-4):
    """
    values_by_model: dict {display_name: (mean, std)} for the 3 ensembled
    architectures' Type A AP. Returns (merged, shared_value_or_None).
    Merges (returns True) only if every model's mean AND std agree within
    `tolerance`; otherwise prints a warning explaining why they were kept
    separate and returns False.
    """
    names = list(values_by_model.keys())
    means = [values_by_model[n][0] for n in names]
    stds = [values_by_model[n][1] for n in names]

    def _close(a, b):
        if np.isnan(a) and np.isnan(b):
            return True
        if np.isnan(a) or np.isnan(b):
            return False
        return abs(a - b) < tolerance

    means_match = all(_close(m, means[0]) for m in means)
    stds_match = all(_close(s, stds[0]) for s in stds)

    if means_match and stds_match:
        return True, (means[0], stds[0])

    print(
        f"[Warning] {label_idx_name} AP differs across the 3 ensembled architectures "
        f"({dict(zip(names, zip(means, stds)))}) -- expected identical, since ensemble=True "
        f"routes {label_idx_name} through the same independently re-seeded linear probe "
        f"regardless of the main model's architecture. Showing separately per row instead of "
        f"merging. This can happen from mismatched random_state/num_trials/"
        f"ensemble_n_optuna_trials/ensemble_n_split_in/ensemble_n_epochs_max across the runs, "
        f"or GPU/cudnn non-determinism even with matching seeds."
    )
    return False, None


# ============================================================================
# TABLE 1: ENSEMBLE COMPARISON (ABMIL / ABMIL_LSTM / LSTM_only + LinearProbe)
# ============================================================================

def _build_ensemble_comparison_table(
    abmil_results_ensemble, abmil_lstm_results_ensemble,
    mean_only_results_ensemble, lstm_results_ensemble,
    linear_probe_results,
    label_cols, label_display_names, decimals, type_a_tolerance, resize_to_fit,
    caption, table_label,
):
    type_a_idx = list(label_cols).index('type_a')

    _check_ensemble_flag(abmil_results_ensemble, True, 'ABMIL (ensemble)')
    _check_ensemble_flag(abmil_lstm_results_ensemble, True, 'ABMIL+LSTM (ensemble)')
    _check_ensemble_flag(mean_only_results_ensemble, True, 'Mean-only (ensemble)')
    _check_ensemble_flag(lstm_results_ensemble, True, 'LSTM-only (ensemble)')

    rows = [
        ('ABMIL', abmil_results_ensemble),
        ('ABMIL+LSTM', abmil_lstm_results_ensemble),
        ('Mean-only', mean_only_results_ensemble),
        ('LSTM-only', lstm_results_ensemble),
    ]
    all_rows = rows + [('Linear Probe', linear_probe_results)]

    # Per-label mean/std (all 5 labels) + overall cmAP, for all 4 rows.
    per_label_by_model = {}
    overall_by_model = {}
    for name, results_list in all_rows:
        per_label_by_model[name] = {
            label: _mean_std_across_trials(_trial_label_ap(results_list, li))
            for li, label in enumerate(label_cols)
        }
        overall_by_model[name] = _mean_std_across_trials(_trial_overall_ap(results_list))

    # Only Type A is a candidate for merging -- it's the one label ensembling
    # routes through a shared linear probe regardless of main architecture.
    # Type B/C/D/echo still come from each architecture's OWN attention
    # model even when ensemble=True, so they're never merged.
    type_a_ensembled = {name: per_label_by_model[name]['type_a'] for name, _ in rows}
    merged, shared_type_a = _check_shared_type_a(type_a_ensembled, tolerance=type_a_tolerance)

    header = ['Model'] + [label_display_names.get(l, l) for l in label_cols] + ['cmAP']
    col_spec = 'l ' + 'c' * len(label_cols) + ' c'

    lines = []
    lines.append('% Requires \\usepackage{booktabs} and \\usepackage{multirow} in your preamble.')
    if resize_to_fit:
        lines.append('% Requires \\usepackage{adjustbox} in your preamble (for the shrink-only box).')
    lines.append('\\begin{table}[htbp]')
    lines.append('\\centering')
    lines.append(f'\\caption{{{caption}}}')
    lines.append(f'\\label{{{table_label}}}')
    if resize_to_fit:
        lines.append('\\begin{adjustbox}{max width=\\textwidth}')
    lines.append(f'\\begin{{tabular}}{{{col_spec}}}')
    lines.append('\\toprule')
    lines.append(' & '.join(header) + r' \\')
    lines.append('\\midrule')

    for row_i, (name, _) in enumerate(rows):
        cells = [name]
        for label in label_cols:
            if label == 'type_a' and merged:
                if row_i == 0:
                    shared_cell = _format_mean_std(*shared_type_a, decimals=decimals)
                    cells.append(f'\\multirow{{{len(rows)}}}{{*}}{{{shared_cell}}}')
                else:
                    cells.append('')
            else:
                cells.append(_format_mean_std(*per_label_by_model[name][label], decimals=decimals))
        cells.append(_format_mean_std(*overall_by_model[name], decimals=decimals))
        lines.append(' & '.join(cells) + r' \\')

    lines.append('\\midrule')
    lp_cells = ['Linear Probe']
    for label in label_cols:
        lp_cells.append(_format_mean_std(*per_label_by_model['Linear Probe'][label], decimals=decimals))
    lp_cells.append(_format_mean_std(*overall_by_model['Linear Probe'], decimals=decimals))
    lines.append(' & '.join(lp_cells) + r' \\')

    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    if resize_to_fit:
        lines.append('\\end{adjustbox}')
    lines.append('\\end{table}')

    return '\n'.join(lines)


# ============================================================================
# TABLE 2: ARCHITECTURE x LABEL COMPARISON (non-ensembled)
# ============================================================================

def _build_architecture_label_table(
    abmil_results, abmil_lstm_results, mean_only_results, lstm_results,
    label_cols, label_display_names, decimals, resize_to_fit,
    caption, table_label,
):
    _check_ensemble_flag(abmil_results, False, 'ABMIL (non-ensemble)')
    _check_ensemble_flag(abmil_lstm_results, False, 'ABMIL+LSTM (non-ensemble)')
    _check_ensemble_flag(mean_only_results, False, 'Mean-only (non-ensemble)')
    _check_ensemble_flag(lstm_results, False, 'LSTM-only (non-ensemble)')

    rows = [
        ('ABMIL', abmil_results),
        ('ABMIL+LSTM', abmil_lstm_results),
        ('Mean-only', mean_only_results),
        ('LSTM-only', lstm_results),
    ]

    header = ['Model'] + [label_display_names.get(l, l) for l in label_cols] + ['cmAP']
    col_spec = 'l ' + 'c' * len(label_cols) + ' c'

    lines = []
    lines.append('% Requires \\usepackage{booktabs} in your preamble.')
    if resize_to_fit:
        lines.append('% Requires \\usepackage{adjustbox} in your preamble (for the shrink-only box).')
    lines.append('\\begin{table}[htbp]')
    lines.append('\\centering')
    lines.append(f'\\caption{{{caption}}}')
    lines.append(f'\\label{{{table_label}}}')
    if resize_to_fit:
        lines.append('\\begin{adjustbox}{max width=\\textwidth}')
    lines.append(f'\\begin{{tabular}}{{{col_spec}}}')
    lines.append('\\toprule')
    lines.append(' & '.join(header) + r' \\')
    lines.append('\\midrule')

    for name, results_list in rows:
        cells = [name]
        for label_idx in range(len(label_cols)):
            m, s = _mean_std_across_trials(_trial_label_ap(results_list, label_idx))
            cells.append(_format_mean_std(m, s, decimals=decimals))
        m, s = _mean_std_across_trials(_trial_overall_ap(results_list))
        cells.append(_format_mean_std(m, s, decimals=decimals))
        lines.append(' & '.join(cells) + r' \\')

    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    if resize_to_fit:
        lines.append('\\end{adjustbox}')
    lines.append('\\end{table}')

    return '\n'.join(lines)


# ============================================================================
# PUBLIC ENTRY POINT
# ============================================================================

def architecture_comparison_latex_tables(
    abmil_results_ensemble, abmil_lstm_results_ensemble,
    mean_only_results_ensemble, lstm_results_ensemble,
    linear_probe_results,
    abmil_results, abmil_lstm_results, mean_only_results, lstm_results,
    label_cols=('type_a', 'type_b', 'type_c', 'type_d', 'echo'),
    label_display_names=None,
    decimals=3,
    type_a_tolerance=1e-4,
    resize_to_fit=True,
    table1_caption='Ensembled ABMIL / ABMIL+LSTM / Mean-only / LSTM-only against the standalone '
                    'linear-probe baseline, across all call types.',
    table1_label='tab:ensemble_comparison',
    table2_caption='ABMIL / ABMIL+LSTM / Mean-only / LSTM-only compared across all call types, '
                    'without ensembling.',
    table2_label='tab:architecture_label_comparison',
    output_path_table1='ensemble_comparison_table.tex',
    output_path_table2='architecture_label_comparison_table.tex',
):
    """
    Builds both comparison tables. See module docstring for full methodology.

    Parameters
    ----------
    abmil_results_ensemble, abmil_lstm_results_ensemble, mean_only_results_ensemble,
    lstm_results_ensemble : list of dict
        Results for 'ABMIL', 'ABMIL_LSTM', 'Mean_only', 'LSTM_only' respectively,
        each run with ensemble=True (e.g. via abmil_classifier_tuned_optuna(...,
        variants=('ABMIL',), ensemble=True, ...)). Used only for Table 1.
    linear_probe_results : list of dict
        Output of linear_probe_tuned_optuna. Used only for Table 1.
    abmil_results, abmil_lstm_results, mean_only_results, lstm_results : list of dict
        Same four architectures, this time run WITHOUT ensembling
        (ensemble=False, or omitted). Used only for Table 2.
    label_cols : sequence of str
        Must match the column order of y in every results list.
    label_display_names : dict or None
        Header text per label for Table 2. Default:
        {'type_a': 'Type A', ..., 'echo': 'Echolocation'}.
    decimals : int
        Decimal places for every mean/std.
    type_a_tolerance : float
        Absolute tolerance for deciding whether the 4 ensembled
        architectures' Type A AP counts as "the same" (see module
        docstring) -- below this, merged into one \\multirow cell; above
        it, shown separately with a printed warning.
    resize_to_fit : bool
        Wrap each tabular in an adjustbox with max width=\\textwidth so it
        can't overflow the page width -- unlike \\resizebox, this only
        SHRINKS an oversized table; a naturally-narrow table is left at its
        normal size instead of being stretched to fill the full page width
        (confirmed by actually rendering both tables -- \\resizebox
        stretched a small table's font to an oversized, unusual-looking
        size before this fix).
    table1_caption, table1_label, table2_caption, table2_label : str
        \\caption{}/\\label{} contents for each table.
    output_path_table1, output_path_table2 : str or None
        Where to write each .tex file (snippets meant to be \\input{}'d into
        a larger document). Pass None to skip writing and just get the
        strings back.

    Returns
    -------
    (latex_table1, latex_table2) : tuple of str
    """
    label_cols = list(label_cols)
    if label_display_names is None:
        label_display_names = {
            'type_a': 'Type A', 'type_b': 'Type B', 'type_c': 'Type C',
            'type_d': 'Type D', 'echo': 'Echolocation',
        }

    latex_table1 = _build_ensemble_comparison_table(
        abmil_results_ensemble, abmil_lstm_results_ensemble,
        mean_only_results_ensemble, lstm_results_ensemble,
        linear_probe_results,
        label_cols=label_cols, label_display_names=label_display_names,
        decimals=decimals, type_a_tolerance=type_a_tolerance,
        resize_to_fit=resize_to_fit,
        caption=table1_caption, table_label=table1_label,
    )
    latex_table2 = _build_architecture_label_table(
        abmil_results, abmil_lstm_results, mean_only_results, lstm_results,
        label_cols=label_cols, label_display_names=label_display_names,
        decimals=decimals, resize_to_fit=resize_to_fit,
        caption=table2_caption, table_label=table2_label,
    )

    if output_path_table1 is not None:
        with open(output_path_table1, 'w') as f:
            f.write(latex_table1)
    if output_path_table2 is not None:
        with open(output_path_table2, 'w') as f:
            f.write(latex_table2)

    return latex_table1, latex_table2


# ============================================================================
# ENSEMBLED vs NON-ENSEMBLED STATISTICAL COMPARISON
# ============================================================================
#
# Paired Wilcoxon signed-rank test (primary) + paired t-test (secondary),
# comparing the ensembled and non-ensembled runs of ONE architecture at a
# time. Paired because, as long as both runs used the same random_state /
# num_trials / n_split_out, the outer CV split depends only on
# (X_bags, y, trial_seed) -- not on ensemble/variants -- so trial i's folds
# are the SAME held-out data in both runs, just scored by two different
# models. This is checked (via each fold's fitted wrapper's own
# .random_state), not assumed.
#
# Default pairing is at the TRIAL level (one paired difference per trial,
# using each trial's already-aggregated mean AP) rather than the FOLD level
# (one per (trial, fold)). Fold-level pairing gives more nominal samples,
# but folds within a trial share most of their training data, so treating
# them as independent inflates the false-positive rate -- a well-documented
# pitfall in the CV-comparison literature (Dietterich 1998; Nadeau & Bengio
# 2003). Trial-level avoids this since different trials use genuinely
# different splits. Fold-level pairing is available via level='fold' for
# more power at the cost of that caveat -- use with that in mind, not as a
# free upgrade.
#
# POWER WARNING: Wilcoxon's minimum achievable two-sided p-value is set by
# sample size alone -- with n=5 trials (a common default in this project) it
# cannot go below ~0.0625, so it can NEVER reach the conventional 0.05
# threshold regardless of true effect size. n_trials >= 6 is the bare
# minimum for Wilcoxon to be able to detect anything at that threshold; more
# is better.


def _fold_level_values(results_list, label_idx):
    """Flat array of per-(trial, fold) AP values, in trial-then-fold order.
    label_idx=None computes macro-AP across all labels for that fold
    (mirrors how mean_AP's own per-fold scores were computed); a specific
    label_idx computes that label's AP alone, with NaN for a fold where the
    label has only one class present (undefined AP -- not silently 0)."""
    values = []
    for res in results_list:
        for y_true_fold, y_pred_fold in zip(res['y_true_cv'], res['y_pred_proba_cv']):
            y_true_fold = np.asarray(y_true_fold)
            y_pred_fold = np.asarray(y_pred_fold)
            if label_idx is None:
                values.append(average_precision_score(y_true_fold, y_pred_fold, average='macro'))
            else:
                y_t = y_true_fold[:, label_idx]
                if y_t.sum() == 0 or y_t.sum() == len(y_t):
                    values.append(np.nan)
                else:
                    values.append(average_precision_score(y_t, y_pred_fold[:, label_idx]))
    return np.array(values, dtype=float)


def _fold_label_level_values(results_list, label_cols):
    """
    Flat array of per-(trial, fold, label) AP values, in
    trial-then-fold-then-label order -- every individual label's AP for
    every fold of every trial, NOT averaged across labels or folds the way
    _fold_level_values(..., label_idx=None) is. This is what lets a pooled
    comparison reach n = num_trials * n_split_out * len(label_cols) (e.g.
    5 trials x 5 folds x 5 labels = 125) -- see
    compare_ensembled_vs_plain_pooled for why this pooling conflates
    several different things and should be read as a supplementary check,
    not the primary claim. NaN for a (fold, label) where that label has
    only one class present in that fold (undefined AP -- not silently 0)."""
    values = []
    for res in results_list:
        for y_true_fold, y_pred_fold in zip(res['y_true_cv'], res['y_pred_proba_cv']):
            y_true_fold = np.asarray(y_true_fold)
            y_pred_fold = np.asarray(y_pred_fold)
            for label_idx in range(len(label_cols)):
                y_t = y_true_fold[:, label_idx]
                if y_t.sum() == 0 or y_t.sum() == len(y_t):
                    values.append(np.nan)
                else:
                    values.append(average_precision_score(y_t, y_pred_fold[:, label_idx]))
    return np.array(values, dtype=float)


def _paired_values(results_a, results_b, label_idx, level):
    if level == 'trial':
        if label_idx is None:
            return _trial_overall_ap(results_a), _trial_overall_ap(results_b)
        return _trial_label_ap(results_a, label_idx), _trial_label_ap(results_b, label_idx)
    elif level == 'fold':
        return (_fold_level_values(results_a, label_idx), _fold_level_values(results_b, label_idx))
    else:
        raise ValueError(f"level must be 'trial' or 'fold', got {level!r}")


def _check_paired_seeds(results_a, results_b, name_a, name_b):
    """Best-effort check that both results lists used matching per-trial
    random_state values, via each trial's fitted wrapper's own
    .random_state attribute -- paired tests assume the SAME outer-CV folds
    were used for both conditions. Prints a warning rather than raising."""
    try:
        seeds_a = [res['best_models'][0].random_state for res in results_a]
        seeds_b = [res['best_models'][0].random_state for res in results_b]
    except (KeyError, IndexError, AttributeError):
        return
    if seeds_a != seeds_b:
        print(
            f"[Warning] {name_a} vs {name_b}: per-trial random_state sequences differ "
            f"({seeds_a} vs {seeds_b}) -- paired tests assume the SAME outer-CV folds were "
            f"used for both conditions. Results may not be validly paired if these actually "
            f"used different splits."
        )


def compare_ensembled_vs_plain(
    results_ensemble, results_plain, model_name,
    label_cols=('type_a', 'type_b', 'type_c', 'type_d', 'echo'),
    include_overall=True,
    alternative='two-sided',
    level='trial',
):
    """
    Paired Wilcoxon signed-rank test comparing the ensembled and
    non-ensembled runs of ONE architecture, per label (plus overall cmAP if
    include_overall). See the section header above for the full
    methodology and caveats (pairing rationale, trial vs. fold level, the
    small-n power limitation).

    Parameters
    ----------
    results_ensemble, results_plain : list of dict
        Same architecture, ensemble=True and ensemble=False/omitted
        respectively. Must have the same number of trials (paired test
        requirement).
    model_name : str
        Label for the 'model' column in the returned DataFrame (e.g. 'ABMIL').
    label_cols : sequence of str
    include_overall : bool
        Whether to also test overall cmAP (label='cmAP', using each trial's
        stored mean_AP rather than a recomputed per-label value).
    alternative : {'two-sided', 'less', 'greater'}
        Passed to scipy.stats.wilcoxon. 'greater'/'less' test whether
        results_ensemble is greater/less than results_plain specifically --
        only use a one-sided alternative if you had that direction as a
        prior hypothesis, not chosen after looking at the data.
    level : {'trial', 'fold'}
        Pairing granularity -- see section header above. 'trial' (default)
        is the statistically safer choice.

    Returns
    -------
    pd.DataFrame, one row per (model_name, label) with: n_pairs,
    mean_ensemble, mean_plain, mean_diff, wilcoxon_stat, wilcoxon_p, note.
    A row is all-NaN with a note if fewer than 2 valid paired differences
    remain (e.g. too few trials, or every fold degenerate for that label).
    """
    label_cols = list(label_cols)
    _check_paired_seeds(results_ensemble, results_plain,
                         f'{model_name} (ensemble)', f'{model_name} (plain)')

    if level == 'trial' and len(results_ensemble) != len(results_plain):
        raise ValueError(
            f"{model_name}: ensembled run has {len(results_ensemble)} trials, non-ensembled "
            f"has {len(results_plain)} -- paired tests require the same number of trials in both."
        )

    comparisons = [(label, i) for i, label in enumerate(label_cols)]
    if include_overall:
        comparisons.append(('cmAP', None))

    rows = []
    for display_label, label_idx in comparisons:
        va, vb = _paired_values(results_ensemble, results_plain, label_idx, level)
        if len(va) != len(vb):
            raise ValueError(
                f"{model_name}/{display_label}: {len(va)} vs {len(vb)} paired values at "
                f"level={level!r} -- results lists must have matching trial/fold structure."
            )
        diff = va - vb
        valid = ~(np.isnan(va) | np.isnan(vb))
        va_v, vb_v, diff_v = va[valid], vb[valid], diff[valid]
        n = len(diff_v)

        row = {
            'model': model_name, 'label': display_label, 'n_pairs': n,
            'mean_ensemble': np.mean(va_v) if n else np.nan,
            'mean_plain': np.mean(vb_v) if n else np.nan,
            'mean_diff': np.mean(diff_v) if n else np.nan,
        }

        if n < 2 or np.allclose(diff_v, diff_v[0]):
            # Too few valid pairs, or every difference identical (e.g. all
            # zero) -- scipy.stats.wilcoxon is undefined/degenerate here.
            row.update(wilcoxon_stat=np.nan, wilcoxon_p=np.nan,
                       note='fewer than 2 valid pairs, or no variation in the differences')
        else:
            try:
                w_stat, w_p = stats.wilcoxon(diff_v, alternative=alternative)
            except ValueError:
                w_stat, w_p = np.nan, np.nan
            row.update(wilcoxon_stat=w_stat, wilcoxon_p=w_p, note='')

        rows.append(row)

    return pd.DataFrame(rows)


def compare_ensembled_vs_plain_pooled(
    results_ensemble, results_plain, model_name,
    label_cols=('type_a', 'type_b', 'type_c', 'type_d', 'echo'),
    alternative='two-sided',
):
    """
    Paired Wilcoxon signed-rank test pooling ALL labels together with
    fold-level pairing, giving n_pairs = num_trials * n_split_out *
    len(label_cols) (e.g. 5 trials x 5 folds x 5 labels = 125) -- one
    single p-value per architecture, instead of compare_ensembled_vs_plain's
    6-row per-label breakdown.

    IMPORTANT -- this pooling has two real validity costs relative to the
    per-label, trial-level test, neither resolved by having more samples:

    1. Fold-level pairing (needed to reach n=125 at all -- trial-level
       pooling across labels alone only gives num_trials * len(label_cols),
       e.g. 25) reintroduces the fold-correlation problem: folds within one
       CV repeat share most of their training data, so treating them as
       independent samples understates the true variance (Dietterich 1998;
       Nadeau & Bengio 2003).
    2. Pooling across labels conflates whatever ensembling does to each
       label into ONE number. Ensembling routes Type A through a dedicated
       linear probe while leaving Type B/C/D/echo to the main model
       (trained on one fewer label) -- mechanistically two different
       interventions wearing one name. A significant pooled result could be
       almost entirely a large, real Type A effect, spread evenly across
       all five labels, or anything in between -- this test can't
       distinguish those; a null result could equally be hiding a real
       effect in one label cancelled out by four flat ones.

    Use this as a supplementary robustness check alongside
    compare_ensembled_vs_plain's per-label breakdown -- not as a
    replacement for it, since it can't tell you WHICH label is driving
    whatever it finds.

    Parameters
    ----------
    results_ensemble, results_plain : list of dict
        Same architecture, ensemble=True and ensemble=False/omitted
        respectively. Must have matching trial/fold structure.
    model_name : str
        Label for the 'model' column in the returned DataFrame (e.g. 'ABMIL').
    label_cols : sequence of str
        Labels pooled together.
    alternative : {'two-sided', 'less', 'greater'}
        Passed to scipy.stats.wilcoxon -- see compare_ensembled_vs_plain's
        docstring for the same caveat about only using a one-sided
        alternative if it was a prior hypothesis.

    Returns
    -------
    pd.DataFrame with exactly one row: model, label ('pooled (all types)'),
    n_pairs, mean_ensemble, mean_plain, mean_diff, wilcoxon_stat,
    wilcoxon_p, note. The 'label' column is a constant placeholder (not a
    real label) so this row can be concatenated directly with
    compare_ensembled_vs_plain's output if you want one combined table.
    """
    label_cols = list(label_cols)
    _check_paired_seeds(results_ensemble, results_plain,
                         f'{model_name} (ensemble)', f'{model_name} (plain)')

    va = _fold_label_level_values(results_ensemble, label_cols)
    vb = _fold_label_level_values(results_plain, label_cols)
    if len(va) != len(vb):
        raise ValueError(
            f"{model_name}: {len(va)} vs {len(vb)} pooled (trial, fold, label) values -- "
            f"results lists must have matching trial/fold/label structure."
        )

    diff = va - vb
    valid = ~(np.isnan(va) | np.isnan(vb))
    va_v, vb_v, diff_v = va[valid], vb[valid], diff[valid]
    n = len(diff_v)

    row = {
        'model': model_name, 'label': 'pooled (all types)', 'n_pairs': n,
        'mean_ensemble': np.mean(va_v) if n else np.nan,
        'mean_plain': np.mean(vb_v) if n else np.nan,
        'mean_diff': np.mean(diff_v) if n else np.nan,
    }

    if n < 2 or np.allclose(diff_v, diff_v[0]):
        row.update(wilcoxon_stat=np.nan, wilcoxon_p=np.nan,
                   note='fewer than 2 valid pairs, or no variation in the differences')
    else:
        try:
            w_stat, w_p = stats.wilcoxon(diff_v, alternative=alternative)
        except ValueError:
            w_stat, w_p = np.nan, np.nan
        row.update(wilcoxon_stat=w_stat, wilcoxon_p=w_p, note='')

    return pd.DataFrame([row])


def compare_all_ensembled_vs_plain_pooled(
    abmil_results_ensemble, abmil_results,
    abmil_lstm_results_ensemble, abmil_lstm_results,
    lstm_results_ensemble, lstm_results,
    label_cols=('type_a', 'type_b', 'type_c', 'type_d', 'echo'),
    alternative='two-sided',
    correction=None,
):
    """
    Runs compare_ensembled_vs_plain_pooled for all 3 architectures and
    concatenates into one 3-row DataFrame (one pooled test per
    architecture). See that function's docstring for the pooling caveats --
    this is a supplementary check, not a replacement for
    compare_all_ensembled_vs_plain's per-label breakdown.

    Parameters
    ----------
    abmil_results_ensemble, abmil_results, abmil_lstm_results_ensemble,
    abmil_lstm_results, lstm_results_ensemble, lstm_results : list of dict
        Ensembled/non-ensembled result pairs for each of the 3 architectures.
    label_cols, alternative : see compare_ensembled_vs_plain_pooled.
    correction : {None, 'fdr_bh', 'bonferroni'}
        If given, adds a 'wilcoxon_p_corrected' column across these 3 rows
        -- a much milder multiple-comparisons concern than the per-label
        version's up to 18 rows, but still 3 simultaneous tests.

    Returns
    -------
    pd.DataFrame -- see compare_ensembled_vs_plain_pooled for the column
    schema (plus 'wilcoxon_p_corrected' when `correction` is set).
    """
    dfs = [
        compare_ensembled_vs_plain_pooled(
            abmil_results_ensemble, abmil_results, 'ABMIL',
            label_cols=label_cols, alternative=alternative,
        ),
        compare_ensembled_vs_plain_pooled(
            abmil_lstm_results_ensemble, abmil_lstm_results, 'ABMIL+LSTM',
            label_cols=label_cols, alternative=alternative,
        ),
        compare_ensembled_vs_plain_pooled(
            lstm_results_ensemble, lstm_results, 'LSTM',
            label_cols=label_cols, alternative=alternative,
        ),
    ]
    result = pd.concat(dfs, ignore_index=True)

    if correction is not None:
        corr_fn = {'fdr_bh': _bh_correction, 'bonferroni': _bonferroni_correction}.get(correction)
        if corr_fn is None:
            raise ValueError(f"correction must be None, 'fdr_bh', or 'bonferroni', got {correction!r}")
        result['wilcoxon_p_corrected'] = corr_fn(result['wilcoxon_p'].to_numpy())

    return result


def _bh_correction(pvals):
    """Benjamini-Hochberg FDR correction. NaN inputs stay NaN and are
    excluded from the correction (don't count toward the total number of
    tests)."""
    pvals = np.asarray(pvals, dtype=float)
    valid_mask = ~np.isnan(pvals)
    valid_p = pvals[valid_mask]
    n = len(valid_p)
    out = pvals.copy()
    if n == 0:
        return out
    order = np.argsort(valid_p)
    ranked = valid_p[order]
    corrected = ranked * n / np.arange(1, n + 1)
    corrected = np.minimum.accumulate(corrected[::-1])[::-1]  # BH step-up monotonicity
    corrected = np.clip(corrected, 0, 1)
    out_valid = np.empty(n)
    out_valid[order] = corrected
    out[valid_mask] = out_valid
    return out


def _bonferroni_correction(pvals):
    pvals = np.asarray(pvals, dtype=float)
    valid_mask = ~np.isnan(pvals)
    n = int(valid_mask.sum())
    out = pvals.copy()
    out[valid_mask] = np.clip(pvals[valid_mask] * n, 0, 1)
    return out


def compare_all_ensembled_vs_plain(
    abmil_results_ensemble, abmil_results,
    abmil_lstm_results_ensemble, abmil_lstm_results,
    lstm_results_ensemble, lstm_results,
    label_cols=('type_a', 'type_b', 'type_c', 'type_d', 'echo'),
    include_overall=True,
    alternative='two-sided',
    level='trial',
    correction=None,
):
    """
    Runs compare_ensembled_vs_plain for all 3 architectures and concatenates
    into one tidy DataFrame -- up to 3 x 6 = 18 rows (5 labels + cmAP, per
    architecture) if include_overall=True.

    Parameters
    ----------
    abmil_results_ensemble, abmil_results, abmil_lstm_results_ensemble,
    abmil_lstm_results, lstm_results_ensemble, lstm_results : list of dict
        Ensembled/non-ensembled result pairs for each of the 3 architectures.
    label_cols, include_overall, alternative, level : see compare_ensembled_vs_plain.
    correction : {None, 'fdr_bh', 'bonferroni'}
        If given, adds a 'wilcoxon_p_corrected' column, correcting across
        ALL rows in the returned table (i.e. across all 3 architectures'
        tests jointly, not each architecture separately -- appropriate if
        you're going to look at/report on all of them together, which is
        the usual case). None (default) returns only the raw p-values --
        with up to 18 simultaneous tests, worth considering deliberately
        rather than leaving uncorrected by default without thinking about it.

    Returns
    -------
    pd.DataFrame -- see compare_ensembled_vs_plain for the column schema
    (plus 'wilcoxon_p_corrected' when `correction` is set).
    """
    dfs = [
        compare_ensembled_vs_plain(
            abmil_results_ensemble, abmil_results, 'ABMIL',
            label_cols=label_cols, include_overall=include_overall,
            alternative=alternative, level=level,
        ),
        compare_ensembled_vs_plain(
            abmil_lstm_results_ensemble, abmil_lstm_results, 'ABMIL+LSTM',
            label_cols=label_cols, include_overall=include_overall,
            alternative=alternative, level=level,
        ),
        compare_ensembled_vs_plain(
            lstm_results_ensemble, lstm_results, 'LSTM',
            label_cols=label_cols, include_overall=include_overall,
            alternative=alternative, level=level,
        ),
    ]
    result = pd.concat(dfs, ignore_index=True)

    if correction is not None:
        corr_fn = {'fdr_bh': _bh_correction, 'bonferroni': _bonferroni_correction}.get(correction)
        if corr_fn is None:
            raise ValueError(f"correction must be None, 'fdr_bh', or 'bonferroni', got {correction!r}")
        result['wilcoxon_p_corrected'] = corr_fn(result['wilcoxon_p'].to_numpy())

    return result


# ============================================================================
# USAGE EXAMPLE
# ============================================================================
#
#   abmil_ens       = abmil_classifier_tuned_optuna(X_bags, y, variants=('ABMIL',),
#                                                     ensemble=True, ensemble_labels=(0,))
#   abmil_lstm_ens  = abmil_classifier_tuned_optuna(X_bags, y, variants=('ABMIL_LSTM',),
#                                                     ensemble=True, ensemble_labels=(0,))
#   mean_only_ens   = abmil_classifier_tuned_optuna(X_bags, y, variants=('Mean_only',),
#                                                     ensemble=True, ensemble_labels=(0,))
#   lstm_ens        = abmil_classifier_tuned_optuna(X_bags, y, variants=('LSTM_only',),
#                                                     ensemble=True, ensemble_labels=(0,))
#   linear_probe    = linear_probe_tuned_optuna(X_pooled, y)
#
#   abmil_plain      = abmil_classifier_tuned_optuna(X_bags, y, variants=('ABMIL',))
#   abmil_lstm_plain = abmil_classifier_tuned_optuna(X_bags, y, variants=('ABMIL_LSTM',))
#   mean_only_plain  = abmil_classifier_tuned_optuna(X_bags, y, variants=('Mean_only',))
#   lstm_plain       = abmil_classifier_tuned_optuna(X_bags, y, variants=('LSTM_only',))
#
#   table1, table2 = architecture_comparison_latex_tables(
#       abmil_ens, abmil_lstm_ens, mean_only_ens, lstm_ens, linear_probe,
#       abmil_plain, abmil_lstm_plain, mean_only_plain, lstm_plain,
#   )
#
#   stats_df = compare_all_ensembled_vs_plain(
#       abmil_ens, abmil_plain, abmil_lstm_ens, abmil_lstm_plain, lstm_ens, lstm_plain,
#       correction='fdr_bh',
#   )
#   print(stats_df[['model', 'label', 'n_pairs', 'mean_diff', 'wilcoxon_p', 'wilcoxon_p_corrected']])
#
#   # Supplementary pooled-across-labels check (n=125 if 5 trials x 5 folds x
#   # 5 labels) -- read alongside stats_df above, not instead of it; see
#   # compare_ensembled_vs_plain_pooled's docstring for what this can't tell you.
#   stats_df_pooled = compare_all_ensembled_vs_plain_pooled(
#       abmil_ens, abmil_plain, abmil_lstm_ens, abmil_lstm_plain, lstm_ens, lstm_plain,
#       correction='fdr_bh',
#   )
#   print(stats_df_pooled)
