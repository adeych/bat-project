"""
Out-of-distribution (OOD) evaluation for a single ABMILSklearnWrapper trained
on the FULL training set (e.g. via train_abmil_all_data), evaluated against a
separate OOD dataset with its own metadata (file paths, multi-label targets,
species).

Two kinds of results, deliberately treated differently:

1. PER-SPECIES AP, with bootstrap percentile confidence intervals.
   Bootstrapping resamples RECORDINGS WITHIN each species (with replacement,
   same size as that species' own count) -- not the whole OOD set pooled
   together -- so each species' CI reflects only its own sampling
   uncertainty. Species with fewer than `min_n` recordings are skipped
   entirely (returned separately, not silently dropped) since a bootstrap CI
   on a handful of examples isn't informative. The row with no species
   assigned is excluded from this breakdown (there's nothing to group it
   into) but is NOT excluded from the overall metrics below, since it still
   has valid labels and the overall metrics don't need a species at all.

2. OVERALL per-label AP + macro-averaged cmAP, as plain POINT ESTIMATES
   across the full OOD set (species-agnostic, no bootstrapping) -- per the
   explicit choice to keep this one un-bootstrapped.

Model predictions are computed ONCE for the whole OOD set via
model.predict_proba(X_bags_ood) -- NOT evaluate_abmil_ood, which only reads
fitted_wrapper.model_ directly and so silently drops any ensemble "simple"
labels (e.g. type_a) whenever ensemble=True. predict_proba correctly
reassembles the ensemble's complex/simple predictions into the full label
ordering. Bootstrapping only ever resamples ROWS from these fixed
(label, prediction) pairs -- it never re-runs the model, which is both the
correct way to bootstrap a metric and the only way 1000 iterations stays
cheap.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def bootstrap_ap_per_species(
    model, X_bags_ood, metadata_df,
    label_cols=('type_a', 'type_b', 'type_c', 'type_d', 'echo'),
    species_col='species_latin',
    min_n=3,
    n_bootstrap=1000,
    ci=0.95,
    random_state=42,
):
    """
    Per-(species, label) AP with a within-species bootstrap percentile CI,
    plus a per-species cmAP (macro-average over whichever labels are
    actually defined for that species) with its own bootstrap CI, appended
    into the SAME long-format table as a pseudo-label row ('cmAP').

    Each bootstrap iteration draws ONE shared resample of a species'
    recordings (not a separate resample per label) -- every defined label's
    AP, and that iteration's cmAP, are computed off that same resample. This
    is what makes the cmAP CI a real macro-average of correlated draws
    rather than a combination of five independently-resampled distributions,
    and it's also why results are NOT bit-identical to a version that
    resampled each label independently, even at the same random_state.

    "Defined" labels for a species are those with both classes present at
    the species level (n_positive > 0 and < n) -- cmAP is only ever averaged
    over these; there's no minimum count of defined labels required, so a
    species with only one defined label gets a cmAP numerically equal to
    that single AP (see the `n_labels_in_cmap` column, which makes this
    visible rather than silently implying every cmAP rests on the same
    number of labels).

    Parameters
    ----------
    model : ABMILSklearnWrapper
        A single fitted wrapper (e.g. from train_abmil_all_data) -- its
        predict_proba is called ONCE for the whole OOD set up front.
    X_bags_ood : list of (n_windows, n_features) arrays
        OOD bag embeddings, in the SAME row order as metadata_df.
    metadata_df : pd.DataFrame
        Row-aligned with X_bags_ood. Must contain label_cols and species_col.
    label_cols : sequence of str
        Multi-label target columns, in the same order the model was trained
        on.
    species_col : str
        Column in metadata_df to group by (default 'species_latin').
    min_n : int
        Minimum recordings a species needs (after dropping rows with no
        species_col value) to get a per-species AP/CI/cmAP at all. Species
        below this are reported in `excluded_df`, not silently dropped.
    n_bootstrap : int
        Bootstrap iterations per species (shared across its labels and cmAP).
    ci : float
        Confidence level for the percentile interval (e.g. 0.95 -> 2.5th/
        97.5th percentiles).
    random_state : int
        Base seed. Each species gets its own independent, reproducible RNG
        (seeded as random_state + species rank), so results are stable
        across reruns and don't depend on species iteration order.

    Returns
    -------
    per_species_df : pd.DataFrame
        One row per (species, label) PLUS one row per species with
        label='cmAP', columns: species, label, n, n_positive, AP, ci_low,
        ci_high, n_valid_bootstrap, n_labels_in_cmap, note.
        n_positive/n_labels_in_cmap are NaN on ordinary label rows (not
        applicable); n_labels_in_cmap is NaN on ordinary label rows and set
        on cmAP rows. AP/ci_low/ci_high are NaN when a (species, label) has
        only one class present (AP undefined), or when a species has NO
        defined labels at all (cmAP undefined). If enough individual
        bootstrap resamples were ALSO degenerate (fewer than 50 valid
        resamples survived for that label, or fewer than 50 iterations had
        at least one defined label survive for cmAP), ci_low/ci_high are
        NaN with a note even though the point estimate itself is fine.
    excluded_df : pd.DataFrame
        One row per species that was skipped for having fewer than min_n
        recordings, or the (up to one) row with no species_col value at all.
    y_pred_ood : np.ndarray, shape (len(metadata_df), len(label_cols))
        The full OOD predicted-probability array, computed once -- handed
        back so overall_ap_summary (or anything else) doesn't need to
        recompute it.
    """
    label_cols = list(label_cols)
    y_ood = metadata_df[label_cols].to_numpy()
    y_pred_ood = model.predict_proba(X_bags_ood)

    has_species = metadata_df[species_col].notna().to_numpy()
    n_missing = int((~has_species).sum())

    rows = []
    excluded = []
    if n_missing > 0:
        excluded.append({
            'species': None, 'n': n_missing,
            'reason': 'no species_col value -- excluded from per-species breakdown '
                      '(still included in overall metrics)',
        })

    species_values = sorted(metadata_df.loc[has_species, species_col].unique())
    for s_i, species in enumerate(species_values):
        species_mask = has_species & (metadata_df[species_col] == species).to_numpy()
        idx = np.where(species_mask)[0]
        n = len(idx)

        if n < min_n:
            excluded.append({'species': species, 'n': n, 'reason': f'n < min_n ({min_n})'})
            continue

        y_true_sp = y_ood[idx]
        y_pred_sp = y_pred_ood[idx]
        # Independent, reproducible RNG per species -- doesn't depend on
        # iteration order and won't shift if species are added/removed.
        rng = np.random.RandomState(random_state + s_i)

        # --- Point estimates + which labels are "defined" for this species ---
        point_ap = {}
        n_pos_by_label = {}
        for label_idx, label in enumerate(label_cols):
            y_t = y_true_sp[:, label_idx]
            n_pos = int(y_t.sum())
            n_pos_by_label[label] = n_pos
            if n_pos == 0 or n_pos == n:
                point_ap[label] = np.nan
            else:
                point_ap[label] = average_precision_score(y_t, y_pred_sp[:, label_idx])
        defined_labels = [l for l in label_cols if not np.isnan(point_ap[l])]

        # --- Shared-resample bootstrap: one resample per iteration, reused
        #     across every defined label AND that iteration's cmAP. ---
        boot_aps_by_label = {label: [] for label in label_cols}
        boot_cmaps = []
        for _ in range(n_bootstrap):
            resample_idx = rng.randint(0, n, size=n)
            y_t_b_all = y_true_sp[resample_idx]
            y_p_b_all = y_pred_sp[resample_idx]

            iter_aps = []
            for label_idx, label in enumerate(label_cols):
                if label not in defined_labels:
                    continue
                y_t_b = y_t_b_all[:, label_idx]
                # A resample can land on only one class even when the full
                # species-label group has both -- skip just THIS label for
                # just THIS iteration, rather than reporting a misleading AP.
                if y_t_b.sum() == 0 or y_t_b.sum() == n:
                    continue
                y_p_b = y_p_b_all[:, label_idx]
                ap_b = average_precision_score(y_t_b, y_p_b)
                boot_aps_by_label[label].append(ap_b)
                iter_aps.append(ap_b)

            if iter_aps:  # at least one defined label survived this resample
                boot_cmaps.append(float(np.mean(iter_aps)))
            # else: every defined label degenerated in this resample (rare,
            # only likely for very small n) -- this iteration contributes to
            # neither any label's CI nor cmAP's CI.

        # --- Per-label rows ---
        for label in label_cols:
            n_pos = n_pos_by_label[label]
            if label not in defined_labels:
                rows.append({
                    'species': species, 'label': label, 'n': n, 'n_positive': n_pos,
                    'AP': np.nan, 'ci_low': np.nan, 'ci_high': np.nan,
                    'n_valid_bootstrap': 0, 'n_labels_in_cmap': np.nan,
                    'note': 'undefined -- only one class present',
                })
                continue

            boot_aps = boot_aps_by_label[label]
            n_valid = len(boot_aps)
            if n_valid < 50:
                ci_low, ci_high = np.nan, np.nan
                note = f'only {n_valid}/{n_bootstrap} valid resamples -- CI unreliable'
            else:
                alpha = (1 - ci) / 2
                ci_low, ci_high = np.percentile(boot_aps, [100 * alpha, 100 * (1 - alpha)])
                note = ''

            rows.append({
                'species': species, 'label': label, 'n': n, 'n_positive': n_pos,
                'AP': point_ap[label], 'ci_low': ci_low, 'ci_high': ci_high,
                'n_valid_bootstrap': n_valid, 'n_labels_in_cmap': np.nan, 'note': note,
            })

        # --- cmAP row (pseudo-label) ---
        if defined_labels:
            point_cmap = float(np.mean([point_ap[l] for l in defined_labels]))
            n_valid_cmap = len(boot_cmaps)
            if n_valid_cmap < 50:
                ci_low_c, ci_high_c = np.nan, np.nan
                note_c = f'only {n_valid_cmap}/{n_bootstrap} valid resamples -- CI unreliable'
            else:
                alpha = (1 - ci) / 2
                ci_low_c, ci_high_c = np.percentile(boot_cmaps, [100 * alpha, 100 * (1 - alpha)])
                note_c = ''
            rows.append({
                'species': species, 'label': 'cmAP', 'n': n, 'n_positive': np.nan,
                'AP': point_cmap, 'ci_low': ci_low_c, 'ci_high': ci_high_c,
                'n_valid_bootstrap': n_valid_cmap, 'n_labels_in_cmap': len(defined_labels),
                'note': note_c,
            })
        else:
            rows.append({
                'species': species, 'label': 'cmAP', 'n': n, 'n_positive': np.nan,
                'AP': np.nan, 'ci_low': np.nan, 'ci_high': np.nan,
                'n_valid_bootstrap': 0, 'n_labels_in_cmap': 0,
                'note': 'no labels with both classes present for this species',
            })

    per_species_df = pd.DataFrame(rows)
    excluded_df = pd.DataFrame(excluded)
    return per_species_df, excluded_df, y_pred_ood


def overall_ap_summary(
    metadata_df, y_pred_ood,
    label_cols=('type_a', 'type_b', 'type_c', 'type_d', 'echo'),
):
    """
    Per-label AP + macro-averaged cmAP across the FULL OOD set, as plain
    point estimates -- no bootstrapping, no species grouping. Includes every
    row in metadata_df (including the one with no species assigned, if any
    -- this metric doesn't need a species).

    Parameters
    ----------
    metadata_df : pd.DataFrame
        Must contain label_cols. Row order doesn't need to match anything
        else as long as y_pred_ood is row-aligned to it.
    y_pred_ood : np.ndarray, shape (len(metadata_df), len(label_cols))
        As returned by bootstrap_ap_per_species (or model.predict_proba(X_bags_ood)
        directly).
    label_cols : sequence of str

    Returns
    -------
    pd.DataFrame with columns ['label', 'AP'] -- one row per label plus a
    final 'cmAP' row (macro-average across labels).
    """
    label_cols = list(label_cols)
    y_ood = metadata_df[label_cols].to_numpy()

    per_label_rows = [
        {'label': label, 'AP': average_precision_score(y_ood[:, i], y_pred_ood[:, i])}
        for i, label in enumerate(label_cols)
    ]
    cmap = average_precision_score(y_ood, y_pred_ood, average='macro')
    per_label_rows.append({'label': 'cmAP', 'AP': cmap})

    return pd.DataFrame(per_label_rows)


def evaluate_ood(
    model, X_bags_ood, metadata_df,
    label_cols=('type_a', 'type_b', 'type_c', 'type_d', 'echo'),
    species_col='species_latin',
    min_n=3,
    n_bootstrap=1000,
    ci=0.95,
    random_state=42,
):
    """
    Full OOD evaluation in one call: per-species AP with within-species
    bootstrap percentile CIs (see bootstrap_ap_per_species) plus overall
    per-label AP and macro-cmAP as plain point estimates across the whole
    OOD set (see overall_ap_summary). Model predictions are computed once
    and shared between both.

    Parameters mirror bootstrap_ap_per_species -- see its docstring for
    details on each.

    Returns
    -------
    per_species_df, excluded_df, overall_df : the three DataFrames described
        in bootstrap_ap_per_species and overall_ap_summary.
    """
    per_species_df, excluded_df, y_pred_ood = bootstrap_ap_per_species(
        model, X_bags_ood, metadata_df,
        label_cols=label_cols, species_col=species_col, min_n=min_n,
        n_bootstrap=n_bootstrap, ci=ci, random_state=random_state,
    )
    overall_df = overall_ap_summary(metadata_df, y_pred_ood, label_cols=label_cols)
    return per_species_df, excluded_df, overall_df


# ============================================================================
# LATEX TABLE EXPORT
# ============================================================================

_LATEX_ESCAPES = {
    '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_',
    '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
    '^': r'\textasciicircum{}', '\\': r'\textbackslash{}',
}


def _escape_latex(text):
    """Escapes LaTeX special characters in a plain-text string."""
    return ''.join(_LATEX_ESCAPES.get(ch, ch) for ch in str(text))


def _format_cell(ap, ci_low=None, ci_high=None, decimals=3):
    """
    Formats one table cell: '--' for NaN/undefined AP, 'AP [ci_low, ci_high]'
    when both CI bounds are given and finite, or just 'AP' (no brackets)
    when ci_low/ci_high are None or NaN -- used for the point-estimate-only
    Overall row, which has no bootstrap CI by design.
    """
    if pd.isna(ap):
        return '--'
    ap_str = f"{ap:.{decimals}f}"
    if ci_low is None or ci_high is None or pd.isna(ci_low) or pd.isna(ci_high):
        return ap_str
    return f"{ap_str} [{ci_low:.{decimals}f}, {ci_high:.{decimals}f}]"


def per_species_latex_table(
    per_species_df, overall_df, excluded_df=None,
    label_cols=('type_a', 'type_b', 'type_c', 'type_d', 'echo'),
    label_display_names=None,
    decimals=3,
    caption="Out-of-distribution average precision (AP) by species and call type, "
            "with 95\\% bootstrap percentile confidence intervals.",
    table_label='tab:ood_ap_species',
    resize_to_fit=True,
    output_path='ood_ap_table.tex',
):
    """
    Builds a wide LaTeX table (booktabs style) from the outputs of
    evaluate_ood / bootstrap_ap_per_species / overall_ap_summary: one row
    per species, one column per label, a final cmAP column filled from each
    species' own 'cmAP' pseudo-label row (with its own bootstrap CI -- see
    bootstrap_ap_per_species), and a final bold 'Overall' row (from
    overall_df).

    Each species-row cell is 'AP [ci_low, ci_high]' to `decimals` places, or
    '--' where AP is NaN (only one class present for that species/label --
    see bootstrap_ap_per_species). A species' cmAP cell gets a
    '$\\dagger$' marker (with a footnote explaining it) whenever it's
    averaged over fewer than the full label_cols set -- there's no minimum
    number of defined labels required to report a per-species cmAP, so a
    species with only one defined label gets a cmAP numerically equal to
    that single AP; the marker makes that visible rather than implying every
    species' cmAP rests on the same footing. The Overall row has no
    bootstrap CI by design (overall_ap_summary is point-estimates-only), so
    its cells show just the AP value, no brackets. Species already excluded
    from per_species_df (below min_n, or missing a species assignment) are
    NOT shown as rows -- if `excluded_df` is passed, a footnote below the
    table reports how many were left out and why, and the Overall row's `n`
    is reconstructed as the full OOD count (shown species' n + excluded n);
    without excluded_df, the Overall row's `n` cell is left as '--' to
    avoid implying a total that wasn't actually verified.

    Parameters
    ----------
    per_species_df, overall_df, excluded_df : as returned by evaluate_ood
        (excluded_df is optional -- omit for no footnote / no Overall `n`).
    label_cols : sequence of str
        Column order for the table (must match the labels present in
        per_species_df/overall_df).
    label_display_names : dict or None
        Maps each label_cols entry to its column header text. Default:
        {'type_a': 'Type A', 'type_b': 'Type B', 'type_c': 'Type C',
         'type_d': 'Type D', 'echo': 'Echolocation'} -- override any/all of
        these if you want different header text.
    decimals : int
        Decimal places for AP and CI bounds.
    caption, table_label : str
        LaTeX \\caption{} and \\label{} contents.
    output_path : str or None
        Where to write the .tex file (a table snippet meant to be
        \\input{}'d into a larger document -- not a standalone compilable
        document). Pass None to skip writing and just get the string back.
    resize_to_fit : bool
        With CIs crammed into every cell, this table is genuinely wide (9
        columns including bracketed ranges) -- left unscaled it overflows a
        standard \\textwidth (confirmed by actually compiling it). When True
        (default), wraps the tabular in \\resizebox{\\textwidth}{!}{...} so
        it's guaranteed to fit the page width, at the cost of a smaller
        (but LaTeX-scaled, so still crisp) font. Requires \\usepackage{graphicx}.
        Set False if you'd rather handle sizing yourself (e.g. a landscape
        page, or a smaller custom font).

    Returns
    -------
    str : the full LaTeX snippet (also written to output_path unless None).
    """
    label_cols = list(label_cols)
    if label_display_names is None:
        label_display_names = {
            'type_a': 'Type A', 'type_b': 'Type B', 'type_c': 'Type C',
            'type_d': 'Type D', 'echo': 'Echolocation',
        }

    species_order = per_species_df['species'].drop_duplicates().tolist()
    n_by_species = per_species_df.groupby('species')['n'].first()
    n_total_labels = len(label_cols)
    any_partial_cmap = False  # tracks whether the footnote about partial cmAP is needed

    body_rows = []
    for species in species_order:
        sp_df = per_species_df[per_species_df['species'] == species].set_index('label')
        cells = [_escape_latex(species), str(int(n_by_species[species]))]
        for label in label_cols:
            if label in sp_df.index:
                r = sp_df.loc[label]
                cells.append(_format_cell(r['AP'], r['ci_low'], r['ci_high'], decimals))
            else:
                cells.append('--')
        if 'cmAP' in sp_df.index:
            r = sp_df.loc['cmAP']
            cmap_cell = _format_cell(r['AP'], r['ci_low'], r['ci_high'], decimals)
            n_used = r['n_labels_in_cmap']
            # Flag when a species' cmAP isn't averaged over the full label
            # set -- e.g. Pipistrellus pygmaeus has only one defined label,
            # so its "average" is that single AP value, not a real average.
            if pd.notna(n_used) and int(n_used) < n_total_labels and cmap_cell != '--':
                cmap_cell += r'$^\dagger$'
                any_partial_cmap = True
            cells.append(cmap_cell)
        else:
            cells.append('--')
        body_rows.append(cells)

    overall_lookup = overall_df.set_index('label')['AP']
    if excluded_df is not None and not excluded_df.empty:
        total_n = int(n_by_species.sum() + excluded_df['n'].sum())
        overall_n_str = str(total_n)
    else:
        overall_n_str = '--'
    overall_cells = [r'\textbf{Overall}', overall_n_str]
    for label in label_cols:
        overall_cells.append(_format_cell(overall_lookup.get(label, float('nan')), decimals=decimals))
    overall_cells.append(_format_cell(overall_lookup.get('cmAP', float('nan')), decimals=decimals))

    header = ['Species', '$n$'] + [label_display_names.get(l, l) for l in label_cols] + ['cmAP']
    col_spec = 'l r ' + 'c' * len(label_cols) + ' c'

    lines = []
    lines.append('% Requires \\usepackage{booktabs} in your preamble.')
    if resize_to_fit:
        lines.append('% Requires \\usepackage{graphicx} in your preamble (for \\resizebox).')
    lines.append('\\begin{table}[htbp]')
    lines.append('\\centering')
    lines.append(f'\\caption{{{caption}}}')
    lines.append(f'\\label{{{table_label}}}')
    if resize_to_fit:
        lines.append('\\resizebox{\\textwidth}{!}{%')
    lines.append(f'\\begin{{tabular}}{{{col_spec}}}')
    lines.append('\\toprule')
    lines.append(' & '.join(header) + r' \\')
    lines.append('\\midrule')
    for cells in body_rows:
        lines.append(' & '.join(cells) + r' \\')
    lines.append('\\midrule')
    lines.append(' & '.join(overall_cells) + r' \\')
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    if resize_to_fit:
        lines.append('}')

    note_parts = []
    if excluded_df is not None and not excluded_df.empty:
        n_species_excluded = int(excluded_df['species'].notna().sum())
        n_unassigned = int(excluded_df['species'].isna().sum())
        if n_species_excluded > 0:
            note_parts.append(f"{n_species_excluded} species excluded for having too few recordings")
        if n_unassigned > 0:
            note_parts.append(
                f"{n_unassigned} recording(s) with no species assignment excluded from "
                f"the per-species rows (still counted in Overall)"
            )
    if any_partial_cmap:
        note_parts.append(
            r'$\dagger$ cmAP averaged over fewer than the full set of '
            f'{n_total_labels} call types for that species (only those with both '
            r'classes present -- see \texttt{n\_labels\_in\_cmap})'
        )
    if note_parts:
        lines.append('')
        lines.append(r'\vspace{2pt}')
        lines.append(r'{\footnotesize Note: ' + '; '.join(note_parts) + '.}')

    lines.append('\\end{table}')
    latex = '\n'.join(lines)

    if output_path is not None:
        with open(output_path, 'w') as f:
            f.write(latex)

    return latex


# ============================================================================
# USAGE EXAMPLE
# ============================================================================
#
#   final_model = train_abmil_all_data(X_bags, y, variant='ABMIL_LSTM_residual_proj', ...)
#
#   per_species_df, excluded_df, overall_df = evaluate_ood(
#       final_model, X_bags_ood, metadata_df,
#   )
#
#   print(overall_df)                                  # type_a..echo + cmAP, point estimates
#   print(per_species_df.sort_values('AP'))             # per (species, label) + 95% CI
#   print(excluded_df)                                  # species skipped (n < min_n) or unassigned
#
#   latex = per_species_latex_table(
#       per_species_df, overall_df, excluded_df,
#       output_path='ood_ap_table.tex',
#   )

