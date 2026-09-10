"""
Statistical comparison functions for the augmentation experiment results, and
for the MIL temporal-variant comparison (ABMIL / ABMIL_LSTM_residual /
LSTM_only_residual / LSTM_last).

Four functions, two per comparison:

1. Which encoder performs best overall (pooling augmentation-on and
   augmentation-off runs together)?
   -> compare_encoders_overall(): Friedman test (global "do encoders differ?")
      followed by Nemenyi post-hoc pairwise comparisons, same approach as
      perform_encoder_statistical_analysis / perform_and_plot_nemenyi, with
      blocks now matched across (model, augmentation condition, trial, fold,
      class) so both conditions' runs contribute.

2. For a given encoder, does augmentation help or hurt?
   -> compare_augmentation_per_encoder(): a paired Wilcoxon signed-rank test
      per encoder, NOT Friedman+Nemenyi. Friedman/Nemenyi are rank-based
      tests designed for 3+ matched conditions; with exactly two paired
      conditions (aug vs. no-aug), the standard non-parametric choice is the
      Wilcoxon signed-rank test, which uses the *magnitude* of each matched
      difference rather than collapsing each comparison to a rank/sign the
      way Friedman effectively does at k=2. Since a separate test is run per
      encoder, both raw and Holm-Bonferroni-corrected p-values are reported
      (see that function's docstring for when to use which).

3. Which encoder performs best overall, across the four MIL variants?
   -> compare_encoders_overall_variants(): same principle as (1), pooling all
      four MIL variants together as matched blocks.

4. For a given encoder, which MIL variant works best?
   -> compare_variants_per_encoder(): Friedman + Nemenyi per encoder, across
      five variants (ABMIL, ABMIL_LSTM_residual, LSTM_only_residual,
      LSTM_last, Mean_only). Unlike (2), Friedman+Nemenyi IS the right test
      here -- there are 5 matched conditions per encoder, not 2, which is
      exactly what Friedman/Nemenyi are built for.

All functions expect matched CV structure: for a given encoder, the same
(trial_idx, fold_idx) should refer to the same underlying data split across
whichever conditions/variants are being compared (i.e. run with the same CV
seeds), same assumption your existing encoder-comparison code already relies
on. The Nemenyi critical value is computed exactly for the number of items
being compared via the studentized range distribution, so it's correct
whether comparing 3 encoders or 4 MIL variants.

Also included: plot_critical_difference_diagram / plot_critical_difference_diagrams,
rendering the classic Demsar (2006) rank-axis diagram with Nemenyi clique
bars from any of the above functions' matched-block DataFrames (most
directly, compare_variants_per_encoder's per-encoder output). Building
this surfaced a real inconsistency in _friedman_nemenyi_analysis's CD
value (it was sqrt(2) too large relative to its own pairwise p_matrix,
confirmed numerically -- a rank gap equal to the old CD had a pairwise
p-value of ~0.001, not >= alpha) -- fixed there, so the printed CD, the
pairwise p_matrix, and the new diagram's clique bars are now all mutually
consistent.
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FixedLocator
from sklearn.metrics import average_precision_score


def _clean_model_name(model_name):
    model_name_lower = model_name.lower()
    if any(x in model_name_lower for x in ['prevalence', 'dummy', 'guesser']):
        return None
    if any(x in model_name_lower for x in ['logistic', 'regressor', 'regression']):
        return 'Logistic Regression'
    if 'svm' in model_name_lower:
        return 'SVM'
    if any(x in model_name_lower for x in ['forest', 'rf']):
        return 'Random Forest'
    if 'mlp' in model_name_lower:
        return 'MLP'
    return model_name


def _holm_bonferroni(pvalues):
    """
    Holm-Bonferroni step-down correction. Returns adjusted p-values in the
    SAME order as the input (not sorted).
    """
    pvalues = np.asarray(pvalues, dtype=float)
    n = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        corrected = (n - rank) * pvalues[idx]
        running_max = max(running_max, corrected)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


# ============================================================================
# 1. WHICH ENCODER IS BEST OVERALL (pooled across augmentation conditions)
# ============================================================================

def _extract_matched_encoder_blocks(results_no_aug, results_aug, label_names):
    """
    Builds a DataFrame of matched per-block AP scores: one row per block
    (model, aug_condition, trial_idx, fold_idx, class_name), one column per
    encoder. Only blocks with a score from every encoder are kept (matched
    design required for Friedman/Nemenyi).
    """
    structured_data = {}

    for aug_label, results_dict in (('No Augmentation', results_no_aug),
                                     ('With Augmentation', results_aug)):
        for encoder_name, trials_list in results_dict.items():
            active_trial_idx = 0
            for trial in trials_list:
                clean_model = _clean_model_name(trial.get('model', 'model'))
                if clean_model is None:
                    continue

                y_true_folds = trial['y_true_cv']
                y_pred_proba_folds = trial['y_pred_proba_cv']

                for fold_idx in range(len(y_true_folds)):
                    y_true = y_true_folds[fold_idx]
                    y_pred_proba = y_pred_proba_folds[fold_idx]

                    for c_idx, class_name in enumerate(label_names):
                        ap_score = average_precision_score(y_true[:, c_idx], y_pred_proba[:, c_idx])
                        block_id = (clean_model, aug_label, active_trial_idx, fold_idx, class_name)
                        structured_data.setdefault(block_id, {})[encoder_name] = ap_score

                active_trial_idx += 1

    return pd.DataFrame.from_dict(structured_data, orient='index').dropna()


def _plot_nemenyi_heatmap(p_matrix, encoders):
    """Same heatmap style as perform_and_plot_nemenyi, taking a precomputed p-value matrix."""
    k = len(encoders)
    bin_matrix = pd.DataFrame(0, index=encoders, columns=encoders, dtype=float)
    annot_labels = np.empty((k, k), dtype=object)

    for i in range(k):
        for j in range(k):
            if i == j:
                bin_matrix.iloc[i, j] = np.nan
                annot_labels[i, j] = ""
                continue
            pval = p_matrix.iloc[i, j]
            if pval < 0.001:
                bin_matrix.iloc[i, j] = 3
                annot_labels[i, j] = f"{pval:.2e}"
            elif pval < 0.01:
                bin_matrix.iloc[i, j] = 2
                annot_labels[i, j] = f"{pval:.3f}"
            elif pval < 0.05:
                bin_matrix.iloc[i, j] = 1
                annot_labels[i, j] = f"{pval:.3f}"
            else:
                bin_matrix.iloc[i, j] = 0
                annot_labels[i, j] = f"{pval:.3f}"

    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    colors = ['#FCD7D7', "#CBE6CF", "#A9D1B6", "#7A9B85"]
    custom_cmap = ListedColormap(colors)

    sns.heatmap(
        bin_matrix,
        cmap=custom_cmap,
        vmin=0, vmax=3,
        annot=annot_labels,
        fmt="",
        annot_kws={"fontsize": 10, "fontweight": "bold", "color": "#1c1c1c"},
        linewidths=1.2,
        linecolor='#9e9e9e',
        cbar=False,
        square=True,
        ax=ax
    )

    ax.set_xticklabels(encoders, rotation=45, fontsize=11, fontweight='bold')
    ax.set_yticklabels(encoders, rotation=0, fontsize=11, fontweight='bold')

    cax = fig.add_axes([0.86, 0.28, 0.035, 0.44])
    norm = Normalize(vmin=0, vmax=4)
    mappable = ScalarMappable(norm=norm, cmap=custom_cmap)

    cb = fig.colorbar(mappable, cax=cax, boundaries=[0, 1, 2, 3, 4], orientation='vertical')
    cb.locator = FixedLocator([0.5, 1.5, 2.5, 3.5])
    cb.update_ticks()
    cb.ax.set_yticklabels(['NS', r'$p < 0.05$', r'$p < 0.01$', r'$p < 0.001$'], fontsize=11)
    cb.ax.tick_params(size=0)
    cb.outline.set_edgecolor('#9e9e9e')
    cb.outline.set_linewidth(1.2)

    plt.subplots_adjust(left=0.25, right=0.84, top=0.88, bottom=0.2)
    plt.show()


def _friedman_nemenyi_analysis(df_stat, alpha=0.05, plot=True, item_label="items", group_label=""):
    """
    Shared Friedman + Nemenyi core: given a matched-block DataFrame (rows =
    blocks, columns = the things being compared -- encoders, methods, etc.),
    runs the global Friedman test, and if significant, computes mean ranks
    and the full Nemenyi pairwise p-value matrix, printing results and
    optionally plotting the heatmap. The Nemenyi critical value is computed
    exactly for the actual number of columns (k) via the studentized range
    distribution, so this works correctly for any k (not just k=3).

    Returns
    -------
    dict with keys: 'friedman_stat', 'p_value', 'mean_ranks' (Series or None),
    'p_matrix' (DataFrame or None), 'best' (str or None), 'cd_value' (float
    or None) -- the Nemenyi critical difference, consistent with p_matrix
    (a rank gap of exactly cd_value corresponds to a pairwise p-value of
    exactly alpha; see the pairwise loop below for why this needs the
    sqrt(2) factor that q_alpha alone doesn't include).
    """
    items = list(df_stat.columns)
    k = len(items)
    N = len(df_stat)

    prefix = f"[{group_label}] " if group_label else ""
    print(f"{prefix}Number of matched blocks (N): {N}")
    print(f"{prefix}{item_label.capitalize()} compared: {items}")

    item_vectors = [df_stat[col].values for col in items]
    friedman_stat, p_value = stats.friedmanchisquare(*item_vectors)

    print(f"{prefix}Friedman chi^2 statistic: {friedman_stat:.4f} | p-value: {p_value:.4e}")

    result = {'friedman_stat': friedman_stat, 'p_value': p_value,
              'mean_ranks': None, 'p_matrix': None, 'best': None, 'cd_value': None}

    if p_value >= alpha:
        print(f"{prefix}Result: No significant difference among {item_label} detected (p >= {alpha}).\n")
        return result

    print(f"{prefix}Result: Significant difference detected (p < {alpha}). Running Nemenyi post-hoc test.")

    ranks = df_stat.rank(axis=1, ascending=False)
    mean_ranks = ranks.mean()
    print(f"{prefix}Mean ranks (rank 1 = best): " +
          ", ".join(f"{name}={mean_ranks[name]:.3f}" for name in items))
    best = mean_ranks.idxmin()
    print(f"{prefix}Best-ranked {item_label[:-1] if item_label.endswith('s') else item_label}: {best}")

    # Exact studentized-range critical value for this k, rather than a
    # hardcoded table entry -- valid for any number of items being compared.
    #
    # FIX: this used to be `cd_value = q_alpha * sei`, which is sqrt(2) too
    # large -- it disagreed with the pairwise p_matrix computed just below,
    # which correctly multiplies rank_diff/sei by sqrt(2) before evaluating
    # the studentized-range survival function (Demsar 2006's Nemenyi CD is
    # defined using q_alpha already divided by sqrt(2); scipy's
    # studentized_range.ppf returns the UNDIVIDED critical value). Verified
    # numerically: at rank_diff == the old cd_value, the pairwise p-value
    # below comes out to ~0.001, not >= alpha -- i.e. the old CD would have
    # visually grouped together items the p_matrix itself calls
    # significantly different. Dividing by sqrt(2) here makes "rank_diff <=
    # cd_value" and "p_pairwise >= alpha" agree exactly, which matters
    # directly for plot_critical_difference_diagram's clique bars below.
    q_alpha = stats.studentized_range.ppf(1 - alpha, k, np.inf)
    sei = np.sqrt((k * (k + 1)) / (6 * N))
    cd_value = (q_alpha / np.sqrt(2)) * sei
    print(f"{prefix}Nemenyi Critical Difference (CD): {cd_value:.4f}")

    p_matrix = pd.DataFrame(1.0, index=items, columns=items)
    for i in range(k):
        for j in range(i + 1, k):
            name1, name2 = items[i], items[j]
            rank_diff = abs(mean_ranks[name1] - mean_ranks[name2])
            q_value = (rank_diff / sei) * np.sqrt(2)
            p_pairwise = stats.studentized_range.sf(q_value, k, np.inf)
            p_matrix.loc[name1, name2] = p_pairwise
            p_matrix.loc[name2, name1] = p_pairwise

    print(f"{prefix}Nemenyi pairwise p-value matrix:")
    print(p_matrix.to_string(float_format=lambda x: f"{x:.4e}" if x < 0.001 else f"{x:.4f}"))
    print()

    if plot:
        _plot_nemenyi_heatmap(p_matrix, items)

    result.update({'mean_ranks': mean_ranks, 'p_matrix': p_matrix, 'best': best, 'cd_value': cd_value})
    return result


# ============================================================================
# DIAGNOSTIC: WHY MEAN RANK CAN DISAGREE WITH MEAN/AGGREGATE SCORE (e.g. cmAP)
# ============================================================================

def diagnose_rank_vs_score_disagreement(df_stat):
    """
    Breaks down every item's mean rank AND mean raw score, PER LABEL
    (extracted from df_stat's own block index -- each row's index is a
    (trial_idx, fold_idx, class_name) tuple, so class_name is just the last
    element), to help localize cases where Friedman/Nemenyi's mean-rank
    ordering disagrees with a simple average-score ordering (e.g. cmAP).

    WHY THIS HAPPENS (not a bug): mean rank averages each item's ORDINAL
    position within every block (best=1, worst=k) -- magnitude-blind, it
    only cares who beat whom, not by how much. An aggregate score like
    cmAP averages the RAW values directly -- magnitude-sensitive, blind to
    how consistently something actually wins. These are different
    statistics and can disagree whenever performance is uneven across
    blocks: since blocks here are (trial, fold, label) and different
    labels sit at very different baseline AP scales, an item that wins BIG
    on one label (e.g. a rare/hard class where going from 0.30 to 0.50 AP
    is a large numeric jump) but is narrowly, consistently behind on
    OTHER labels (e.g. easy classes where everyone scores 0.85-0.95 and
    losing by 0.01 still costs the rank in every one of those blocks) can
    end up with a HIGHER average score overall while having a WORSE
    average rank -- rank doesn't care the margin was tiny, and the
    average score doesn't care how often it actually won.

    Use this to check that story directly: look for an item with a good
    mean_score but bad mean_rank on one label (the label it's "winning big"
    on) and a merely-slightly-worse mean_score but consistently bad
    mean_rank on the others (the labels it's "narrowly losing" on).

    Parameters
    ----------
    df_stat : pd.DataFrame
        Any matched-block DataFrame from this module whose row index is
        (trial_idx, fold_idx, class_name) tuples -- i.e. what
        compare_variants_per_encoder / compare_variants_per_encoder1 /
        compare_encoders_overall_variants build internally (accessible via
        compare_variants_per_encoder's returned per-encoder dict directly).

    Returns
    -------
    pd.DataFrame with one row per (label, item): mean_rank and mean_score
    computed within that label's blocks only, plus n_blocks (how many
    blocks contributed -- small n per label means noisier per-label numbers).
    """
    ranks = df_stat.rank(axis=1, ascending=False)
    labels = [idx[-1] for idx in df_stat.index]

    rows = []
    for label in sorted(set(labels)):
        mask = [l == label for l in labels]
        sub_ranks = ranks.loc[mask]
        sub_scores = df_stat.loc[mask]
        for item in df_stat.columns:
            rows.append({
                'label': label, 'item': item,
                'mean_rank': sub_ranks[item].mean(),
                'mean_score': sub_scores[item].mean(),
                'n_blocks': len(sub_ranks),
            })
    return pd.DataFrame(rows)




def _find_cliques(sorted_vals, cd_value):
    """
    Maximal runs of consecutive (rank-sorted) items whose full span is
    within cd_value -- the standard clique-finding rule for CD diagrams: if
    the whole run's rank spread fits within CD, every pair inside it is
    statistically indistinguishable too (a run's endpoints being within CD
    implies every internal pair is also within CD, since ranks are sorted).
    Returns a list of (start_idx, end_idx) index pairs into sorted_vals,
    each spanning >= 2 items, with any clique fully contained in another
    dropped (only maximal cliques are kept -- drawing a contained clique's
    bar too would just be visual clutter, not new information).
    """
    k = len(sorted_vals)
    raw_cliques = []
    for i in range(k):
        j = i
        while j + 1 < k and sorted_vals[j + 1] - sorted_vals[i] <= cd_value:
            j += 1
        if j > i:
            raw_cliques.append((i, j))
    return [c for c in raw_cliques
            if not any(c != c2 and c2[0] <= c[0] and c[1] <= c2[1] for c2 in raw_cliques)]


def plot_critical_difference_diagram(df_stat, alpha=0.05, title=None, group_label="",
                                      rename=None, label_side='top', ax=None):
    """
    Renders a Demsar (2006)-style critical-difference diagram for one
    matched-block DataFrame (rows = blocks, columns = the items being
    compared -- e.g. one encoder's df_stat from compare_variants_per_encoder
    / compare_variants_per_encoder1, or the df_stat compare_encoders_overall
    / compare_encoders_overall_variants return).

    Internally runs the SAME Friedman + Nemenyi analysis
    _friedman_nemenyi_analysis uses (so the numbers match what that
    function prints, including its significance gate): if the global
    Friedman test isn't significant at this alpha, NOTHING is drawn -- per
    Demsar's own recommended workflow, a CD diagram (built from Nemenyi's
    pairwise comparisons) is only meaningful as a post-hoc step after a
    significant omnibus result, the same discipline
    compare_variants_per_encoder / compare_encoders_overall already follow
    (a printed message explains why nothing was drawn).

    Items are placed on a rank axis at their mean rank (rank 1 = best,
    leftmost); a thick horizontal bar ("clique") connects any run of items
    whose mean-rank spread is within the Nemenyi critical difference (CD)
    at this alpha -- items sharing a bar are NOT statistically
    distinguishable from each other; items that don't share any bar ARE.

    Parameters
    ----------
    df_stat : pd.DataFrame
        Rows = matched blocks, columns = items being compared.
    alpha : float
    title : str or None
        Plot title. Falls back to a generic title if None.
    group_label : str
        Forwarded to _friedman_nemenyi_analysis's printed output only (e.g.
        an encoder name), so console messages read consistently with
        compare_variants_per_encoder's own output.
    rename : dict or None
        Maps df_stat's ORIGINAL column names to display labels (e.g.
        {'ABMIL_LSTM_residual': 'ABMIL-LSTM'}) -- applied only to what's
        drawn/returned, never to the underlying Friedman/Nemenyi
        computation (which only ever depends on df_stat's values, not its
        column names). Columns not present in `rename` keep their original
        name. Get this dict wrong and you mislabel which bar is which
        architecture, so build it explicitly from df_stat.columns rather
        than guessing a positional order.
    label_side : {'top', 'bottom'}
        Which side of the rank axis every item's stem + label is drawn on
        (all items on the SAME side, stacked by increasing distance from
        the axis in rank order so none overlap regardless of how close
        together their ranks are). Clique bars are drawn right at the
        rank axis, on this same side, touching the top of the stems --
        not offset onto the opposite side, which would leave them floating
        with nothing connecting to them.
    ax : matplotlib.axes.Axes or None
        Draw into an existing axes (e.g. one panel of a multi-encoder grid
        via plot_critical_difference_diagrams) instead of creating a new
        figure.

    Returns
    -------
    dict with 'mean_ranks' (Series, using the renamed labels if `rename`
    was given) and 'cd_value' (float) if the Friedman test was significant
    and a diagram was drawn; both None otherwise.
    """
    if label_side not in ('top', 'bottom'):
        raise ValueError(f"label_side must be 'top' or 'bottom', got {label_side!r}")

    analysis = _friedman_nemenyi_analysis(df_stat, alpha=alpha, plot=False,
                                           item_label="items", group_label=group_label)
    mean_ranks, cd_value = analysis['mean_ranks'], analysis['cd_value']
    if mean_ranks is None:
        print(f"No critical-difference diagram drawn -- Friedman test was not significant "
              f"at alpha={alpha}, so a post-hoc pairwise comparison isn't warranted.")
        return {'mean_ranks': None, 'cd_value': None}

    mean_ranks = mean_ranks.sort_values()
    if rename:
        mean_ranks = mean_ranks.rename(index=rename)
    sorted_items = mean_ranks.index.tolist()
    sorted_vals = mean_ranks.values
    k = len(sorted_items)
    cliques = _find_cliques(sorted_vals, cd_value)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7, 1.2 + 0.5 * k))

    lo, hi = 1, k
    axis_y = 0.0
    stem_step = 0.35
    sign = 1 if label_side == 'top' else -1
    label_extent = 0.6 + stem_step * k

    ax.set_xlim(lo - 0.5, hi + 0.5)
    if label_side == 'top':
        ax.set_ylim(-0.5, label_extent + 0.6)
    else:
        ax.set_ylim(-label_extent - 0.6, 0.5)

    va_near, va_far = ('bottom', 'top') if label_side == 'top' else ('top', 'bottom')

    # Rank axis line + integer tick marks/labels.
    ax.plot([lo, hi], [axis_y, axis_y], color='black', linewidth=1.2)
    for r in range(lo, hi + 1):
        ax.plot([r, r], [axis_y - 0.04, axis_y + 0.04], color='black', linewidth=1.2)
        ax.text(r, axis_y + 0.12 * sign, str(r), ha='center', va=va_near, fontsize=9)

    # CD reference scale bar, on the same side as the labels, past their
    # furthest extent so it never overlaps them.
    cd_y = sign * (label_extent + 0.35)
    ax.plot([lo, lo + cd_value], [cd_y, cd_y], color='black', linewidth=1.6)
    for xe in (lo, lo + cd_value):
        ax.plot([xe, xe], [cd_y - 0.06 * sign, cd_y + 0.06 * sign], color='black', linewidth=1.6)
    ax.text(lo + cd_value / 2, cd_y + 0.12 * sign, f"CD = {cd_value:.3f}",
            ha='center', va=va_near, fontsize=9)

    # Per-item stems + labels, ALL on the same side (label_side), stacked
    # by increasing height in rank order -- guarantees no two labels
    # overlap regardless of how close together their ranks are.
    for idx, (item, r) in enumerate(zip(sorted_items, sorted_vals)):
        y_label = sign * (0.2 + stem_step * (idx + 1))
        ax.plot([r, r], [axis_y, y_label], color='gray', linewidth=1)
        ax.text(r, y_label + 0.02 * sign, item, ha='center', va=va_near, fontsize=10)

    # Clique bars, drawn right at the rank axis -- the same height every
    # stem starts from, on the SAME side as the labels (a small offset from
    # axis_y, not opposite/floating in empty space) -- so each bar visually
    # touches the top of the stems it's grouping, matching the standard
    # Demsar-style rendering. Stacked further out on that same side only
    # if cliques overlap in x-range (rare) and need separating.
    clique_base = axis_y + sign * 0.06
    for level, (i, j) in enumerate(cliques):
        y = clique_base + sign * level * 0.10
        ax.plot([sorted_vals[i], sorted_vals[j]], [y, y],
                color='black', linewidth=3, solid_capstyle='butt')

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ('top', 'right', 'left', 'bottom'):
        ax.spines[spine].set_visible(False)
    ax.set_title(title if title else "Critical difference diagram", fontsize=11)

    if own_fig:
        plt.tight_layout()
        plt.show()

    return {'mean_ranks': mean_ranks, 'cd_value': cd_value}


def plot_critical_difference_diagrams(per_encoder_results, alpha=0.05, ncols=2,
                                       rename=None, label_side='top'):
    """
    Convenience wrapper: renders one critical-difference diagram per
    encoder from compare_variants_per_encoder's (or
    compare_variants_per_encoder1's) returned dict, arranged in a grid.

    Encoders with fewer than 2 matched variants are skipped outright
    (nothing to compare); encoders whose Friedman test isn't significant at
    this alpha are also skipped from drawing (per
    plot_critical_difference_diagram's own gate), with that panel left
    blank and labeled rather than showing a meaningless diagram.

    Parameters
    ----------
    per_encoder_results : dict {encoder_name: df_stat}
        Output of compare_variants_per_encoder / compare_variants_per_encoder1.
    alpha : float
    ncols : int
        Number of subplot columns in the grid.
    rename, label_side : forwarded to plot_critical_difference_diagram for
        every encoder -- see that function's docstring.

    Returns
    -------
    dict {encoder_name: {'mean_ranks': ..., 'cd_value': ...}} -- only for
    encoders where a diagram was actually drawn (Friedman significant).
    """
    candidates = {enc: df for enc, df in per_encoder_results.items()
                  if not df.empty and len(df.columns) >= 2}
    too_few = set(per_encoder_results) - set(candidates)
    if too_few:
        print(f"Skipping {too_few}: fewer than 2 matched variants to compare.")
    if not candidates:
        print("Nothing to plot.")
        return {}

    n = len(candidates)
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 3.5 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    results = {}
    for ax, (encoder_name, df_stat) in zip(axes_flat, candidates.items()):
        print(f"\n=== Encoder: {encoder_name} ===")
        out = plot_critical_difference_diagram(
            df_stat, alpha=alpha, title=f"Encoder: {encoder_name}",
            group_label=encoder_name, rename=rename, label_side=label_side, ax=ax,
        )
        if out['mean_ranks'] is not None:
            results[encoder_name] = out
        else:
            ax.axis('off')
            ax.set_title(f"Encoder: {encoder_name} (not significant)", fontsize=10)

    for ax in axes_flat[len(candidates):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    return results


def compare_encoders_overall(results_no_aug, results_aug, label_names=None, alpha=0.05, plot=True):
    """
    Friedman test + Nemenyi post-hoc to determine which encoder performs best
    overall, pooling augmentation-on and augmentation-off runs together as
    matched blocks (i.e. this answers "regardless of augmentation, which
    encoder tends to score higher").

    Parameters
    ----------
    results_no_aug, results_aug : dict {encoder_name: results_list}
        Same format as your other plotting/analysis functions.
    label_names : list of str, optional
    alpha : float
        Significance threshold for the global Friedman test.
    plot : bool
        If True (default) and the Friedman test is significant, also renders
        the Nemenyi pairwise significance heatmap.

    Returns
    -------
    df_stat : pandas.DataFrame
        The matched per-block AP scores (index = block, columns = encoders)
        used for the test, in case you want to inspect or re-analyze it.
    """
    if label_names is None:
        label_names = ['Type A', 'Type B', 'Type C', 'Type D', 'Echo']

    df_stat = _extract_matched_encoder_blocks(results_no_aug, results_aug, label_names)
    print("=== Which encoder is best overall? (Friedman + Nemenyi) ===")
    _friedman_nemenyi_analysis(df_stat, alpha=alpha, plot=plot, item_label="encoders")
    return df_stat


# ============================================================================
# 2. FOR EACH ENCODER, DOES AUGMENTATION HELP? (paired Wilcoxon signed-rank)
# ============================================================================

def compare_augmentation_per_encoder(results_no_aug, results_aug, label_names=None,
                                      alpha=0.05, correct=True):
    """
    For each encoder, tests whether augmentation changes AP performance using
    a paired Wilcoxon signed-rank test on matched (trial, fold, class)
    blocks -- the appropriate non-parametric test for exactly two paired
    conditions (see module docstring for why this replaces Friedman/Nemenyi
    here).

    Both the raw and Holm-Bonferroni-corrected p-values are always computed
    and returned/printed side by side, so you can compare them directly.
    `correct` only controls which one drives the Verdict column:

    - correct=True  (default): Holm-corrected p-values decide the verdict.
      Use this if you'd report/interpret the per-encoder results together as
      a joint claim (e.g. "augmentation helps across our pipeline") -- with
      3 independent tests at alpha=0.05, treated uncorrected, there's
      roughly a 1-(1-0.05)^3 ~= 14% chance at least one comes back
      "significant" by chance alone, and correction controls that.
    - correct=False: raw p-values decide the verdict. Use this if each
      encoder's result is a genuinely separate, standalone decision you'd
      act on in isolation regardless of the other encoders' outcomes (e.g.
      "should THIS encoder use augmentation, independent of what we do for
      the others") -- correction isn't protecting against an error you're
      not making, and only costs you power.

    Parameters
    ----------
    results_no_aug, results_aug : dict {encoder_name: results_list}
    label_names : list of str, optional
    alpha : float
        Significance threshold applied to whichever p-value drives the verdict.
    correct : bool
        Whether the Verdict column (and the printed significance calls) use
        the Holm-corrected p-value (True, default) or the raw p-value (False).

    Returns
    -------
    pandas.DataFrame with one row per encoder: number of matched blocks,
    median AP under each condition, the Wilcoxon statistic, raw p-value,
    Holm-corrected p-value, and a plain-language verdict.
    """
    if label_names is None:
        label_names = ['Type A', 'Type B', 'Type C', 'Type D', 'Echo']

    encoders = sorted(set(results_no_aug.keys()) & set(results_aug.keys()))
    missing = set(results_no_aug.keys()) ^ set(results_aug.keys())
    if missing:
        print(f"Warning: encoder(s) {missing} present in only one of the two inputs -- skipping them.")

    rows = []
    for encoder_name in encoders:
        block_scores = {'no_aug': {}, 'aug': {}}

        for key, results_dict in (('no_aug', results_no_aug), ('aug', results_aug)):
            trials_list = results_dict[encoder_name]
            active_trial_idx = 0
            for trial in trials_list:
                y_true_folds = trial['y_true_cv']
                y_pred_proba_folds = trial['y_pred_proba_cv']

                for fold_idx in range(len(y_true_folds)):
                    y_true = y_true_folds[fold_idx]
                    y_pred_proba = y_pred_proba_folds[fold_idx]

                    for c_idx, class_name in enumerate(label_names):
                        ap = average_precision_score(y_true[:, c_idx], y_pred_proba[:, c_idx])
                        block_id = (active_trial_idx, fold_idx, class_name)
                        block_scores[key][block_id] = ap

                active_trial_idx += 1

        matched_ids = sorted(set(block_scores['no_aug']) & set(block_scores['aug']))
        n = len(matched_ids)
        if n < 1:
            print(f"Skipping {encoder_name}: no matched blocks between the two conditions.")
            continue

        no_aug_vals = np.array([block_scores['no_aug'][b] for b in matched_ids])
        aug_vals = np.array([block_scores['aug'][b] for b in matched_ids])

        if np.all(aug_vals == no_aug_vals):
            # scipy.stats.wilcoxon raises when every paired difference is exactly zero
            stat, p = np.nan, 1.0
        else:
            stat, p = stats.wilcoxon(aug_vals, no_aug_vals, zero_method='wilcox', alternative='two-sided')

        rows.append({
            'Encoder': encoder_name,
            'N_blocks': n,
            'Median_AP_no_aug': np.median(no_aug_vals),
            'Median_AP_aug': np.median(aug_vals),
            'Wilcoxon_stat': stat,
            'p_value': p,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("No encoders could be compared.")
        return df

    df['p_value_holm'] = _holm_bonferroni(df['p_value'].values)

    decision_col = 'p_value_holm' if correct else 'p_value'

    def _verdict(row):
        if row[decision_col] >= alpha:
            return 'No significant difference'
        return 'Augmentation better' if row['Median_AP_aug'] > row['Median_AP_no_aug'] else 'No-augmentation better'

    df['Verdict'] = df.apply(_verdict, axis=1)

    which = 'Holm-corrected' if correct else 'raw (uncorrected)'
    print(f"=== Augmentation vs. No-Augmentation per encoder (Wilcoxon signed-rank) ===")
    print(f"Verdict column driven by {which} p-values at alpha={alpha}. "
          f"Both p_value and p_value_holm are shown below regardless.")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    return df


# ============================================================================
# 3. WHICH ENCODER IS BEST OVERALL, ACROSS THE 4 MIL VARIANTS
#    (ABMIL, ABMIL_LSTM_residual, LSTM_only_residual, LSTM_last)
# ============================================================================

VARIANT_ORDER = ['ABMIL', 'ABMIL_LSTM_residual', 'LSTM_only_residual', 'LSTM_last']


def _extract_matched_encoder_blocks_variants(variant_results, label_names):
    """
    Builds a DataFrame of matched per-block AP scores: one row per block
    (variant, trial_idx, fold_idx, class_name), one column per encoder.
    variant_results : dict {variant_name: {encoder_name: results_list}}
    """
    structured_data = {}

    for variant_name in VARIANT_ORDER:
        encoder_results = variant_results[variant_name]
        for encoder_name, trials_list in encoder_results.items():
            active_trial_idx = 0
            for trial in trials_list:
                y_true_folds = trial['y_true_cv']
                y_pred_proba_folds = trial['y_pred_proba_cv']

                for fold_idx in range(len(y_true_folds)):
                    y_true = y_true_folds[fold_idx]
                    y_pred_proba = y_pred_proba_folds[fold_idx]

                    for c_idx, class_name in enumerate(label_names):
                        ap_score = average_precision_score(y_true[:, c_idx], y_pred_proba[:, c_idx])
                        block_id = (variant_name, active_trial_idx, fold_idx, class_name)
                        structured_data.setdefault(block_id, {})[encoder_name] = ap_score

                active_trial_idx += 1

    return pd.DataFrame.from_dict(structured_data, orient='index').dropna()


def compare_encoders_overall_variants(results_abmil, results_abmil_lstm_residual,
                                       results_lstm_residual, results_lstm_last,
                                       label_names=None, alpha=0.05, plot=True):
    """
    Friedman test + Nemenyi post-hoc to determine which encoder performs best
    overall, pooling all four MIL variants together as matched blocks (i.e.
    this answers "regardless of which temporal variant is used, which
    encoder tends to score higher"). Same principle as compare_encoders_overall,
    just with the four MIL variants standing in for the two augmentation
    conditions.

    Parameters
    ----------
    results_abmil, results_abmil_lstm_residual, results_lstm_residual,
    results_lstm_last : dict {encoder_name: results_list}
        Same format as plot_variant_comparison_boxplots's inputs.
    label_names : list of str, optional
    alpha : float
    plot : bool
        If True (default) and the Friedman test is significant, also renders
        the Nemenyi pairwise significance heatmap.

    Returns
    -------
    df_stat : pandas.DataFrame
        The matched per-block AP scores (index = block, columns = encoders).
    """
    if label_names is None:
        label_names = ['Type A', 'Type B', 'Type C', 'Type D', 'Echo']

    variant_results = {
        'ABMIL': results_abmil,
        'ABMIL_LSTM_residual': results_abmil_lstm_residual,
        'LSTM_only_residual': results_lstm_residual,
        'LSTM_last': results_lstm_last,
    }

    df_stat = _extract_matched_encoder_blocks_variants(variant_results, label_names)
    print("=== Which encoder is best overall? (Friedman + Nemenyi, pooled across MIL variants) ===")
    _friedman_nemenyi_analysis(df_stat, alpha=alpha, plot=plot, item_label="encoders")
    return df_stat


# ============================================================================
# 4. FOR EACH ENCODER, WHICH MIL VARIANT WORKS BEST?
#    (Friedman + Nemenyi -- 4 matched conditions, so this IS the right test)
# ============================================================================

def compare_variants_per_encoder1(results_abmil, results_abmil_lstm,
                                  results_lstm_only,
                                  label_names=None, alpha=0.05, plot=True):
    """
    For each encoder separately, runs a Friedman test across the three MIL
    variants (ABMIL, ABMIL_LSTM, LSTM_only) on matched (trial, fold, class)
    blocks, followed by Nemenyi post-hoc pairwise comparisons if significant.

    Parameters
    ----------
    results_abmil, results_abmil_lstm, results_lstm_only : dict {encoder_name: results_list}
    label_names : list of str, optional
    alpha : float
    plot : bool
        If True (default), renders a Nemenyi heatmap for each encoder where
        the Friedman test comes back significant.

    Returns
    -------
    dict {encoder_name: df_stat}
        Per-encoder matched-block DataFrames (index = block, columns =
        variants), in case you want to inspect or re-analyze any of them.
    """
    if label_names is None:
        label_names = ['Type A', 'Type B', 'Type C', 'Type D', 'Echo']

    variant_results = {
        'ABMIL': results_abmil,
        'ABMIL_LSTM': results_abmil_lstm,
        'LSTM_only': results_lstm_only,
    }
    variant_order = ['ABMIL', 'ABMIL_LSTM', 'LSTM_only']

    encoder_key_sets = [set(d.keys()) for d in variant_results.values()]
    encoders = sorted(set.intersection(*encoder_key_sets))
    all_seen = set.union(*encoder_key_sets)
    missing = all_seen - set(encoders)
    if missing:
        print(f"Warning: encoder(s) {missing} not present in all three variant inputs -- skipping them.\n")

    per_encoder_results = {}

    for encoder_name in encoders:
        print(f"\n=== Encoder: {encoder_name} -- which MIL variant works best? (Friedman + Nemenyi) ===")

        structured_data = {}
        for variant_name in variant_order:
            trials_list = variant_results[variant_name][encoder_name]
            active_trial_idx = 0
            for trial in trials_list:
                y_true_folds = trial['y_true_cv']
                y_pred_proba_folds = trial['y_pred_proba_cv']

                for fold_idx in range(len(y_true_folds)):
                    y_true = y_true_folds[fold_idx]
                    y_pred_proba = y_pred_proba_folds[fold_idx]

                    for c_idx, class_name in enumerate(label_names):
                        ap_score = average_precision_score(y_true[:, c_idx], y_pred_proba[:, c_idx])
                        block_id = (active_trial_idx, fold_idx, class_name)
                        structured_data.setdefault(block_id, {})[variant_name] = ap_score

                active_trial_idx += 1

        df_stat = pd.DataFrame.from_dict(structured_data, orient='index').dropna()
        per_encoder_results[encoder_name] = df_stat

        if df_stat.empty or len(df_stat.columns) < 2:
            print(f"Not enough matched blocks/variants to test for {encoder_name}, skipping.")
            continue

        _friedman_nemenyi_analysis(df_stat, alpha=alpha, plot=plot, item_label="variants",
                                    group_label=encoder_name)

    return per_encoder_results

def compare_variants_per_encoder(results_abmil, results_abmil_lstm_residual,
                                  results_lstm_residual, results_lstm_last,
                                  results_mean_only,
                                  label_names=None, alpha=0.05, plot=True):
    """
    For each encoder separately, runs a Friedman test across the five MIL
    variants (ABMIL, ABMIL_LSTM_residual, LSTM_only_residual, LSTM_last,
    Mean_only) on matched (trial, fold, class) blocks, followed by Nemenyi
    post-hoc pairwise comparisons if significant. Unlike the augmentation
    comparison, Friedman+Nemenyi IS the right test here since there are 5
    matched conditions per encoder, not 2.

    Uses its OWN local variant order/dict rather than the module-level
    VARIANT_ORDER, which stays at its original 4 entries -- VARIANT_ORDER is
    also relied on by compare_encoders_overall_variants /
    _extract_matched_encoder_blocks_variants, and those weren't asked to
    change; adding Mean_only to the shared global would have silently broken
    them (compare_encoders_overall_variants's own variant_results dict only
    has 4 keys, so it would KeyError on a 5th name pulled from the global).
    This mirrors the pattern already used by the deprecated
    compare_variants_per_encoder1 above, which has its own local
    variant_order for the same reason.

    Parameters
    ----------
    results_abmil, results_abmil_lstm_residual, results_lstm_residual,
    results_lstm_last, results_mean_only : dict {encoder_name: results_list}
    label_names : list of str, optional
    alpha : float
    plot : bool
        If True (default), renders a Nemenyi heatmap for each encoder where
        the Friedman test comes back significant.

    Returns
    -------
    dict {encoder_name: df_stat}
        Per-encoder matched-block DataFrames (index = block, columns =
        variants), in case you want to inspect or re-analyze any of them.
    """
    if label_names is None:
        label_names = ['Type A', 'Type B', 'Type C', 'Type D', 'Echo']

    variant_results = {
        'ABMIL': results_abmil,
        'ABMIL_LSTM_residual': results_abmil_lstm_residual,
        'LSTM_only_residual': results_lstm_residual,
        'LSTM_last': results_lstm_last,
        'Mean_only': results_mean_only,
    }
    variant_order = ['ABMIL', 'ABMIL_LSTM_residual', 'LSTM_only_residual', 'LSTM_last', 'Mean_only']

    encoder_key_sets = [set(d.keys()) for d in variant_results.values()]
    encoders = sorted(set.intersection(*encoder_key_sets))
    all_seen = set.union(*encoder_key_sets)
    missing = all_seen - set(encoders)
    if missing:
        print(f"Warning: encoder(s) {missing} not present in all five variant inputs -- skipping them.\n")

    per_encoder_results = {}

    for encoder_name in encoders:
        print(f"\n=== Encoder: {encoder_name} -- which MIL variant works best? (Friedman + Nemenyi) ===")

        structured_data = {}
        for variant_name in variant_order:
            trials_list = variant_results[variant_name][encoder_name]
            active_trial_idx = 0
            for trial in trials_list:
                y_true_folds = trial['y_true_cv']
                y_pred_proba_folds = trial['y_pred_proba_cv']

                for fold_idx in range(len(y_true_folds)):
                    y_true = y_true_folds[fold_idx]
                    y_pred_proba = y_pred_proba_folds[fold_idx]

                    for c_idx, class_name in enumerate(label_names):
                        ap_score = average_precision_score(y_true[:, c_idx], y_pred_proba[:, c_idx])
                        block_id = (active_trial_idx, fold_idx, class_name)
                        structured_data.setdefault(block_id, {})[variant_name] = ap_score

                active_trial_idx += 1

        df_stat = pd.DataFrame.from_dict(structured_data, orient='index').dropna()
        per_encoder_results[encoder_name] = df_stat

        if df_stat.empty or len(df_stat.columns) < 2:
            print(f"Not enough matched blocks/variants to test for {encoder_name}, skipping.")
            continue

        _friedman_nemenyi_analysis(df_stat, alpha=alpha, plot=plot, item_label="variants",
                                    group_label=encoder_name)

    return per_encoder_results


# ============================================================================
# USAGE EXAMPLE
# ============================================================================
#
#   results_no_aug = {'perch2': ..., 'encoder_b': ..., 'encoder_c': ...}
#   results_aug    = {'perch2': ..., 'encoder_b': ..., 'encoder_c': ...}
#
#   # Which encoder is best overall (pooling both augmentation conditions)?
#   df_blocks = compare_encoders_overall(results_no_aug, results_aug)
#
#   # For each encoder, does augmentation help? (defaults to Holm-corrected verdicts)
#   df_aug = compare_augmentation_per_encoder(results_no_aug, results_aug)
#
#   # If each encoder is a standalone decision rather than a joint claim,
#   # use the raw per-test p-values instead:
#   df_aug_raw = compare_augmentation_per_encoder(results_no_aug, results_aug, correct=False)
#
#   # --- MIL variant comparison (ABMIL / ABMIL_LSTM_residual / LSTM_only_residual / LSTM_last) ---
#   results_abmil = {'perch2': ..., 'encoder_b': ..., 'encoder_c': ...}
#   results_abmil_lstm_residual = {...}
#   results_lstm_residual = {...}
#   results_lstm_last = {...}
#
#   # Which encoder is best overall (pooling all four variants)?
#   df_blocks_variants = compare_encoders_overall_variants(
#       results_abmil, results_abmil_lstm_residual, results_lstm_residual, results_lstm_last
#   )
#
#   # For each encoder, which variant works best? (now includes Mean_only)
#   results_mean_only = {...}
#   per_encoder = compare_variants_per_encoder(
#       results_abmil, results_abmil_lstm_residual, results_lstm_residual, results_lstm_last,
#       results_mean_only,
#   )
#
#   # Critical-difference diagram(s) from that same per_encoder output --
#   # one figure per encoder in a grid; encoders with a non-significant
#   # Friedman test are skipped (left blank) rather than drawing a
#   # meaningless diagram for them.
#   cd_results = plot_critical_difference_diagrams(per_encoder)
#
#   # Or just one encoder's diagram on its own:
#   plot_critical_difference_diagram(per_encoder['perch2'], title="Encoder: perch2")
