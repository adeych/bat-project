"""
Boxplot comparison of temporal MIL variants (ABMIL, ABMIL+LSTM residual,
LSTM-only residual, LSTM-last) across multiple audio feature encoders.

Reuses the exact visual style of plot_comprehensive_boxplots (2x3 grid: one
subplot per call type + cmAP, pastel boxplots grouped on the x-axis with a
hued sub-group per encoder, shared bottom legend) but swaps the x-axis
grouping from {lr, svm, rf, mlp} to the four
temporal-MIL variants, with encoder as the hue/grouping dimension (3 boxes
side by side per variant).

Expected input format for each of the four variant arguments:
    {encoder_name: results_list}
where results_list is exactly what abmil_classifier_quick (or
abmil_classifier_tuned) returns for that encoder+variant: a list of trial
dicts, each with 'y_true_cv' and 'y_pred_proba_cv' (lists of per-fold arrays).
The same set of encoder_name keys is expected across all four variant dicts.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import average_precision_score


# Display order and two-line labels for the four variants, in plot order.
VARIANT_ORDER = ['ABMIL', 'ABMIL_LSTM_residual', 'LSTM_only_residual', 'LSTM_last']
VARIANT_DISPLAY_LABELS = {
    'ABMIL': 'ABMIL',
    'ABMIL_LSTM_residual': 'ABMIL +\nLSTM (resid.)',
    'LSTM_only_residual': 'LSTM only\n(resid.)',
    'LSTM_last': 'LSTM\n(last state)',
}


def _rows_from_variant_results(variant_name, encoder_results, label_names):
    """
    Flattens {encoder_name: results_list} for ONE variant into per-fold rows.
    """
    rows = []

    for encoder_name, trials_list in encoder_results.items():
        for trial in trials_list:
            y_true_folds = trial['y_true_cv']
            y_pred_proba_folds = trial['y_pred_proba_cv']

            for fold_idx in range(len(y_true_folds)):
                y_true = y_true_folds[fold_idx]
                y_pred_proba = y_pred_proba_folds[fold_idx]

                fold_cmap = average_precision_score(y_true, y_pred_proba, average='macro')

                row = {
                    'Encoder': encoder_name,
                    'Model': variant_name,
                    'cmAP': fold_cmap,
                }
                for idx, name in enumerate(label_names):
                    row[f'AP_{name}'] = average_precision_score(y_true[:, idx], y_pred_proba[:, idx])
                rows.append(row)

    return rows


def plot_variant_comparison_boxplots(results_abmil, results_abmil_lstm_residual,
                                      results_lstm_residual, results_lstm_last,
                                      label_names=None):
    """
    Plots a 2x3 grid of subplots with layout:
    Row 1: Type A, Type B, Type C
    Row 2: Type D, Echo, cmAP
    Each subplot shows boxplots of AP for one call type (or cmAP), with the
    four temporal-MIL variants along the x-axis and one box per encoder
    (hue) within each variant group -- 3 boxes side by side per variant, 4
    variant groups per subplot, matching the visual style of
    plot_comprehensive_boxplots.

    Parameters
    ----------
    results_abmil, results_abmil_lstm_residual, results_lstm_residual,
    results_lstm_last : dict {encoder_name: results_list}
        results_list is the output of abmil_classifier_quick (or
        abmil_classifier_tuned) for that variant, run separately per encoder.
        All four dicts should share the same encoder_name keys.
    label_names : list of str, optional
        Class names corresponding to the columns of y_true / y_pred_proba.
        Defaults to ['Type A', 'Type B', 'Type C', 'Type D', 'Echo'].
    """
    if label_names is None:
        label_names = ['Type A', 'Type B', 'Type C', 'Type D', 'Echo']

    variant_results = {
        'ABMIL': results_abmil,
        'ABMIL_LSTM_residual': results_abmil_lstm_residual,
        'LSTM_only_residual': results_lstm_residual,
        'LSTM_last': results_lstm_last,
    }

    encoder_key_sets = [set(d.keys()) for d in variant_results.values()]
    if len(set(map(frozenset, encoder_key_sets))) > 1:
        print("Warning: the four variant result dicts don't all have the same "
              "encoder keys -- plot will only use encoders common to all four "
              "where present, but double check your inputs.")

    # --- Step 1: Parse Raw Fold Arrays into a Structured DataFrame ---
    all_rows = []

    for variant_name in VARIANT_ORDER:
        encoder_results = variant_results[variant_name]
        rows = _rows_from_variant_results(variant_name, encoder_results, label_names)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    # --- Step 2: Establish Layout Structure (2 Rows x 3 Columns) ---
    sns.set_context("paper", font_scale=1.1)
    sns.set_style("whitegrid", {"grid.linestyle": "--", "grid.alpha": 0.5})

    pastel_palette = sns.color_palette("Pastel1", n_colors=3)
    encoders_list = sorted(df['Encoder'].unique())
    encoder_colors = {encoders_list[i]: pastel_palette[i] for i in range(len(encoders_list))}

    row1_targets = [(f'AP_{label_names[0]}', label_names[0]),
                    (f'AP_{label_names[1]}', label_names[1]),
                    (f'AP_{label_names[2]}', label_names[2])]

    row2_targets = [(f'AP_{label_names[3]}', label_names[3]),
                    (f'AP_{label_names[4]}', label_names[4]),
                    ('cmAP', r'$cmAP$')]

    grid_targets = [row1_targets, row2_targets]

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    # --- Step 3: Run Visualization Iteration Loop ---
    for r_idx in range(2):
        for c_idx in range(3):
            ax = axes[r_idx, c_idx]
            col_name, display_title = grid_targets[r_idx][c_idx]

            sns.boxplot(
                data=df, x='Model', y=col_name, hue='Encoder',
                order=VARIANT_ORDER, palette=encoder_colors,
                ax=ax, width=0.6, showfliers=True,
                flierprops=dict(marker='o', markerfacecolor='gray', markersize=4, markeredgecolor='none', alpha=0.6),
                boxprops=dict(edgecolor='#4d4d4d', linewidth=1.2),
                whiskerprops=dict(color='#4d4d4d', linewidth=1.1),
                capprops=dict(color='#4d4d4d', linewidth=1.1),
                medianprops=dict(color='#2c3e50', linewidth=1.5)
            )

            # --- Step 4: Refine Independent Scales & Clean multi-line Labels ---
            ax.set_title(display_title, fontsize=18, fontweight='bold', pad=12)

            ax.set_xlabel("", fontsize=13)
            ax.set_ylabel("Average Precision (AP)" if c_idx == 0 else "", fontsize=14)

            display_labels = [VARIANT_DISPLAY_LABELS[v] for v in VARIANT_ORDER]
            ax.set_xticklabels(display_labels, rotation=0, ha='center', fontsize=13)
            ax.tick_params(axis='y', labelsize=14)

            current_data = df[col_name].dropna()
            if not current_data.empty:
                ymin, ymax = current_data.min(), current_data.max()
                yrange = ymax - ymin if ymax != ymin else 0.1
                ax.set_ylim(max(0, ymin - 0.05 * yrange), min(1.02, ymax + 0.05 * yrange))

            ax.get_legend().remove()

    # --- Step 5: Generate a Unified Global External Legend Below Layout ---
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        title="Audio Feature Encoders",
        loc='lower center',
        ncol=3,
        fontsize=13,
        title_fontsize=11,
        frameon=True,
        facecolor='white',
        edgecolor='#e0e0e0',
        bbox_to_anchor=(0.5, 0.00)
    )

    plt.tight_layout(rect=[0, 0.12, 1, 0.95])
    plt.subplots_adjust(hspace=0.4, wspace=0.22)
    plt.show()


# ============================================================================
# USAGE EXAMPLE
# ============================================================================
#
# Run abmil_classifier_quick separately per encoder, per variant, e.g.:
#
#   results_abmil = {
#       'perch2': abmil_classifier_quick(X_bags_perch2, y, variants=('ABMIL',)),
#       'encoder_b': abmil_classifier_quick(X_bags_b, y, variants=('ABMIL',)),
#       'encoder_c': abmil_classifier_quick(X_bags_c, y, variants=('ABMIL',)),
#   }
#   results_abmil_lstm_residual = {
#       'perch2': abmil_classifier_quick(X_bags_perch2, y, variants=('ABMIL_LSTM_residual',)),
#       ... same for the other encoders
#   }
#   # ... and similarly for results_lstm_residual (variant='LSTM_only_residual')
#   # and results_lstm_last (variant='LSTM_last')
#
#   plot_variant_comparison_boxplots(
#       results_abmil, results_abmil_lstm_residual,
#       results_lstm_residual, results_lstm_last,
#   )
