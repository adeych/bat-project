"""
Two LaTeX comparison tables for ABMIL / ABMIL_LSTM / LSTM_only / LinearProbe,
built from separate `all_results` lists per model (same schema as
abmil_classifier_tuned_optuna / linear_probe_tuned_optuna: one dict per
trial, each with 'mean_AP', 'y_true_cv', 'y_pred_proba_cv', 'best_models', ...).

Table 1 ("ensemble comparison"): ABMIL / ABMIL_LSTM / LSTM_only (all run
with ensemble=True) plus the standalone LinearProbe baseline, across ALL 5
labels plus a final cmAP column -- same shape as Table 2. The one difference:
since ensemble=True routes Type A through the SAME independently-tuned
linear probe regardless of which main architecture is being compared (see
ABMILSklearnWrapper.fit -- the ensemble Optuna study and final refit are
both explicitly re-seeded from self.random_state, so they don't depend on
how much RNG state the main model consumed beforehand), the three ensembled
architectures' Type A column is expected to be numerically identical, while
Type B/C/D/echo still come from each architecture's own attention model and
are never merged. This is CHECKED, not assumed -- if the three Type A values
don't actually match (mismatched run settings, or GPU/cudnn
non-determinism even with matching seeds), all three are shown separately
in that column with a printed warning instead of being silently merged.
When they do match, the shared Type A value is shown ONCE via \\multirow
spanning the three ensembled rows, rather than repeated three times; every
other column (Type B/C/D/echo/cmAP) always shows each row's own value.

Table 2 ("architecture x label comparison"): ABMIL / ABMIL_LSTM / LSTM_only,
run WITHOUT ensembling (so Type A is predicted natively by each
architecture's own attention/pooling mechanism, not a shared linear probe --
these three legitimately differ and are never merged), across all 5 labels
plus a final cmAP column.

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
    abmil_results_ensemble, abmil_lstm_results_ensemble, lstm_results_ensemble,
    linear_probe_results,
    label_cols, label_display_names, decimals, type_a_tolerance, resize_to_fit,
    caption, table_label,
):
    type_a_idx = list(label_cols).index('type_a')

    _check_ensemble_flag(abmil_results_ensemble, True, 'ABMIL (ensemble)')
    _check_ensemble_flag(abmil_lstm_results_ensemble, True, 'ABMIL+LSTM (ensemble)')
    _check_ensemble_flag(lstm_results_ensemble, True, 'LSTM (ensemble)')

    rows = [
        ('ABMIL', abmil_results_ensemble),
        ('ABMIL+LSTM', abmil_lstm_results_ensemble),
        ('LSTM', lstm_results_ensemble),
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
                    cells.append(f'\\multirow{{3}}{{*}}{{{shared_cell}}}')
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
    abmil_results, abmil_lstm_results, lstm_results,
    label_cols, label_display_names, decimals, resize_to_fit,
    caption, table_label,
):
    _check_ensemble_flag(abmil_results, False, 'ABMIL (non-ensemble)')
    _check_ensemble_flag(abmil_lstm_results, False, 'ABMIL+LSTM (non-ensemble)')
    _check_ensemble_flag(lstm_results, False, 'LSTM (non-ensemble)')

    rows = [
        ('ABMIL', abmil_results),
        ('ABMIL+LSTM', abmil_lstm_results),
        ('LSTM', lstm_results),
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
    abmil_results_ensemble, abmil_lstm_results_ensemble, lstm_results_ensemble,
    linear_probe_results,
    abmil_results, abmil_lstm_results, lstm_results,
    label_cols=('type_a', 'type_b', 'type_c', 'type_d', 'echo'),
    label_display_names=None,
    decimals=3,
    type_a_tolerance=1e-4,
    resize_to_fit=True,
    table1_caption='Ensembled ABMIL / ABMIL+LSTM / LSTM against the standalone '
                    'linear-probe baseline, across all call types.',
    table1_label='tab:ensemble_comparison',
    table2_caption='ABMIL / ABMIL+LSTM / LSTM compared across all call types, without ensembling.',
    table2_label='tab:architecture_label_comparison',
    output_path_table1='ensemble_comparison_table.tex',
    output_path_table2='architecture_label_comparison_table.tex',
):
    """
    Builds both comparison tables. See module docstring for full methodology.

    Parameters
    ----------
    abmil_results_ensemble, abmil_lstm_results_ensemble, lstm_results_ensemble : list of dict
        Results for 'ABMIL', 'ABMIL_LSTM', 'LSTM_only' respectively, each run
        with ensemble=True (e.g. via abmil_classifier_tuned_optuna(...,
        variants=('ABMIL',), ensemble=True, ...)). Used only for Table 1.
    linear_probe_results : list of dict
        Output of linear_probe_tuned_optuna. Used only for Table 1.
    abmil_results, abmil_lstm_results, lstm_results : list of dict
        Same three architectures, this time run WITHOUT ensembling
        (ensemble=False, or omitted). Used only for Table 2.
    label_cols : sequence of str
        Must match the column order of y in every results list.
    label_display_names : dict or None
        Header text per label for Table 2. Default:
        {'type_a': 'Type A', ..., 'echo': 'Echolocation'}.
    decimals : int
        Decimal places for every mean/std.
    type_a_tolerance : float
        Absolute tolerance for deciding whether the 3 ensembled
        architectures' Type A AP counts as "the same" (see module
        docstring) -- below this, merged into one \\multirow cell; above
        it, shown separately with a printed warning.
    resize_to_fit : bool
        Wrap each tabular in an adjustbox with max width=\\textwidth so it
        can't overflow the page width -- unlike \\resizebox, this only
        SHRINKS an oversized table; a naturally-narrow table (like Table 1)
        is left at its normal size instead of being stretched to fill the
        full page width (confirmed by actually rendering both tables --
        \\resizebox stretched Table 1's small 3-column layout to an
        oversized, unusual-looking font before this fix).
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
        abmil_results_ensemble, abmil_lstm_results_ensemble, lstm_results_ensemble,
        linear_probe_results,
        label_cols=label_cols, label_display_names=label_display_names,
        decimals=decimals, type_a_tolerance=type_a_tolerance,
        resize_to_fit=resize_to_fit,
        caption=table1_caption, table_label=table1_label,
    )
    latex_table2 = _build_architecture_label_table(
        abmil_results, abmil_lstm_results, lstm_results,
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
# USAGE EXAMPLE
# ============================================================================
#
#   abmil_ens       = abmil_classifier_tuned_optuna(X_bags, y, variants=('ABMIL',),
#                                                     ensemble=True, ensemble_labels=(0,))
#   abmil_lstm_ens  = abmil_classifier_tuned_optuna(X_bags, y, variants=('ABMIL_LSTM',),
#                                                     ensemble=True, ensemble_labels=(0,))
#   lstm_ens        = abmil_classifier_tuned_optuna(X_bags, y, variants=('LSTM_only',),
#                                                     ensemble=True, ensemble_labels=(0,))
#   linear_probe    = linear_probe_tuned_optuna(X_pooled, y)
#
#   abmil_plain      = abmil_classifier_tuned_optuna(X_bags, y, variants=('ABMIL',))
#   abmil_lstm_plain = abmil_classifier_tuned_optuna(X_bags, y, variants=('ABMIL_LSTM',))
#   lstm_plain       = abmil_classifier_tuned_optuna(X_bags, y, variants=('LSTM_only',))
#
#   table1, table2 = architecture_comparison_latex_tables(
#       abmil_ens, abmil_lstm_ens, lstm_ens, linear_probe,
#       abmil_plain, abmil_lstm_plain, lstm_plain,
#   )
