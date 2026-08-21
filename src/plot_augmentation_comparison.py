"""
Boxplot comparison of data-augmentation on vs. off, across multiple audio
feature encoders.

Same visual style as plot_variant_comparison_boxplots (2x3 grid: one subplot
per call type + cmAP, pastel boxplots grouped on the x-axis with a hued
sub-group per encoder, shared bottom legend), but the x-axis groups are now
the two augmentation conditions ("No Augmentation" / "With Augmentation")
instead of MIL variants, with encoder as the hue/grouping dimension.

Expected input format for each of the two arguments:
    {encoder_name: results_list}
where results_list is a list of trial dicts like:
    {
        "trial": ..., "model": f"MLP_{encoder_name}", "aug_prob": ...,
        "best_models": ..., "mean_AP": ..., "std_AP": ...,
        "y_true_cv": [...],        # fold-by-fold list of (n, n_labels) arrays
        "y_pred_proba_cv": [...],  # fold-by-fold list of (n, n_labels) arrays
        "oof_y_true": ..., "oof_y_pred_proba": ..., "oof_indices": ...,
        "train_histories": ..., "val_histories": ...,
    }
Only 'y_true_cv' and 'y_pred_proba_cv' are used here; which of the two
arguments a result came from (no-aug vs. aug) determines its x-axis group --
the 'model'/'aug_prob' fields inside each trial dict aren't parsed, so they
can hold whatever encoder/probability bookkeeping you already use elsewhere.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import average_precision_score


GROUP_ORDER = ['No Augmentation', 'With Augmentation']


def _rows_from_group_results(group_name, encoder_results, label_names):
    """Flattens {encoder_name: results_list} for ONE group into per-fold rows."""
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
                    'Group': group_name,
                    'cmAP': fold_cmap,
                }
                for idx, name in enumerate(label_names):
                    row[f'AP_{name}'] = average_precision_score(y_true[:, idx], y_pred_proba[:, idx])
                rows.append(row)

    return rows


def plot_augmentation_comparison_boxplots(results_no_aug, results_aug, label_names=None):
    """
    Plots a 2x3 grid of subplots with layout:
    Row 1: Type A, Type B, Type C
    Row 2: Type D, Echo, cmAP
    Each subplot shows boxplots of AP for one call type (or cmAP), with
    "No Augmentation" / "With Augmentation" along the x-axis and one box per
    encoder (hue) within each group.

    Parameters
    ----------
    results_no_aug, results_aug : dict {encoder_name: results_list}
        results_list is your MLP training output for that encoder, run
        separately with augmentation off / on. Both dicts should share the
        same encoder_name keys.
    label_names : list of str, optional
        Class names corresponding to the columns of y_true / y_pred_proba.
        Defaults to ['Type A', 'Type B', 'Type C', 'Type D', 'Echo'].
    """
    if label_names is None:
        label_names = ['Type A', 'Type B', 'Type C', 'Type D', 'Echo']

    group_results = {
        'No Augmentation': results_no_aug,
        'With Augmentation': results_aug,
    }

    encoder_key_sets = [set(d.keys()) for d in group_results.values()]
    if len(set(map(frozenset, encoder_key_sets))) > 1:
        print("Warning: results_no_aug and results_aug don't have the same "
              "encoder keys -- double check your inputs.")

    # --- Step 1: Parse Raw Fold Arrays into a Structured DataFrame ---
    all_rows = []
    for group_name in GROUP_ORDER:
        encoder_results = group_results[group_name]
        all_rows.extend(_rows_from_group_results(group_name, encoder_results, label_names))

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

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # --- Step 3: Run Visualization Iteration Loop ---
    for r_idx in range(2):
        for c_idx in range(3):
            ax = axes[r_idx, c_idx]
            col_name, display_title = grid_targets[r_idx][c_idx]

            sns.boxplot(
                data=df, x='Group', y=col_name, hue='Encoder',
                order=GROUP_ORDER, palette=encoder_colors,
                ax=ax, width=0.5, showfliers=True,
                flierprops=dict(marker='o', markerfacecolor='gray', markersize=4, markeredgecolor='none', alpha=0.6),
                boxprops=dict(edgecolor='#4d4d4d', linewidth=1.2),
                whiskerprops=dict(color='#4d4d4d', linewidth=1.1),
                capprops=dict(color='#4d4d4d', linewidth=1.1),
                medianprops=dict(color='#2c3e50', linewidth=1.5)
            )

            # --- Step 4: Refine Independent Scales & Clean Labels ---
            ax.set_title(display_title, fontsize=18, fontweight='bold', pad=12)

            ax.set_xlabel("", fontsize=13)
            ax.set_ylabel("Average Precision (AP)" if c_idx == 0 else "", fontsize=14)

            ax.set_xticklabels(GROUP_ORDER, rotation=0, ha='center', fontsize=13)
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
#   results_no_aug = {
#       'perch2': all_results_no_aug_perch2,     # your existing all_results list, aug_prob=0
#       'encoder_b': all_results_no_aug_b,
#       'encoder_c': all_results_no_aug_c,
#   }
#   results_aug = {
#       'perch2': all_results_aug_perch2,        # same encoders, aug_prob > 0
#       'encoder_b': all_results_aug_b,
#       'encoder_c': all_results_aug_c,
#   }
#
#   plot_augmentation_comparison_boxplots(results_no_aug, results_aug)
