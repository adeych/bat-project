"""
Analysis utilities for the output of abmil_classifier_tuned_optuna /
abmil_classifier_tuned / abmil_classifier_quick (same result schema):

    all_results = [
        {
            'trial': i, 'model': variant,
            'best_models': [...],            # one fitted ABMILSklearnWrapper per outer fold
            'mean_AP': ..., 'std_AP': ...,
            'y_true_cv': [...], 'y_pred_proba_cv': [...],   # per-fold arrays
            'oof_y_true': ..., 'oof_y_pred_proba': ..., 'oof_indices': ...,
            'train_histories': [...], 'val_histories': [...],  # per-fold curves
            'best_params': [...],
        },
        ...
    ]

Three things worth knowing before reading the code below:

1. `train_histories` / `val_histories` are the FINAL REFIT's curves (the model
   trained on the full outer-train fold at the winning hyperparameters), not
   the inner-CV search curves. No validation split is passed at that stage, so
   `val_histories` is all NaN by design -- only train_loss is meaningful there.
   `plot_loss_curves` below plots exactly these (train-only).

2. The TRUE validation curves used for hyperparameter/epoch selection --
   mean train-loss, mean val-loss, and mean val-AP averaged across each outer
   fold's inner-CV splits -- ARE threaded out into `all_results`, but only
   when it comes from `abmil_classifier_tuned_optuna` (not
   `abmil_classifier_tuned` / `abmil_classifier_quick`, which don't run that
   inner Optuna search): `res['inner_cv_train_loss_curves']`,
   `res['inner_cv_val_loss_curves']`, `res['inner_cv_val_ap_curves']`, each a
   list with one curve per outer fold, taken from the winning trial's
   `study.best_trial.user_attrs`. `plot_inner_cv_curves` and
   `inner_cv_best_epoch_summary` below read these.

3. When ensemble=True, `best_models[fold]` is still the single
   ABMILSklearnWrapper for that fold -- it just has two sub-models: the main
   ABMIL/LSTM model (trained on the complex labels) and `.simple_model_` (a
   LinearProbeSklearnWrapper trained on the simple labels). Both are fully
   fitted sklearn-style estimators, so `.get_params()` on each gives you back
   the actual hyperparameters that were used, and `.simple_model_.history_`
   gives you the linear probe's own FINAL REFIT training curve (train-only,
   same caveat as point 1). The linear probe's own INNER-CV curves (train
   loss / val loss / val AP, mean across its inner folds, same caveat as
   point 2) live on `model.simple_model_inner_cv_history_` -- a dict with
   keys `'mean_train_loss_curve'`, `'mean_val_loss_curve'`,
   `'mean_val_ap_curve'` -- populated automatically by `ABMILSklearnWrapper.fit`
   whenever ensemble=True, no extra bookkeeping needed on the training side.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

# Constructor args that are pipeline bookkeeping, not tunable hyperparameters --
# dropped from the summary tables so groupby-by-hyperparameter isn't cluttered.
_MAIN_BOOKKEEPING_COLS = (
    'n_labels', 'device', 'random_state',
    'ensemble', 'ensemble_labels',
    'ensemble_n_optuna_trials', 'ensemble_n_split_in', 'ensemble_n_epochs_max',
)
_SIMPLE_BOOKKEEPING_COLS = ('n_labels', 'device', 'random_state')


def collect_results(all_results, variant=None):
    """
    Walks all_results and builds two tidy DataFrames -- one row per
    (variant, trial, fold) -- with the actually-deployed hyperparameters and
    that fold's held-out macro AP, for the main model and (where ensemble=True
    was used) the linear probe separately.

    Parameters
    ----------
    all_results : list of dict
        Output of abmil_classifier_tuned_optuna / abmil_classifier_tuned /
        abmil_classifier_quick.
    variant : str or None
        Restrict to results where result['model'] == variant. None = all
        variants present in all_results.

    Returns
    -------
    main_params_df : pd.DataFrame
        Columns: variant, trial, fold, fold_AP, <main model hyperparameters>.
        fold_AP is computed on the complex labels only when ensemble=True for
        that fold, else on all labels.
    simple_params_df : pd.DataFrame
        Same shape, for the linear probe, restricted to folds where
        ensemble=True. Empty (with no rows) if ensemble was never used in
        all_results.
    """
    main_rows, simple_rows = [], []

    for res in all_results:
        if variant is not None and res['model'] != variant:
            continue

        model_name = res['model']
        trial = res['trial']
        best_models = res['best_models']
        y_true_folds = res['y_true_cv']
        y_pred_folds = res['y_pred_proba_cv']

        for fold_idx, model in enumerate(best_models):
            y_true = np.asarray(y_true_folds[fold_idx])
            y_pred = np.asarray(y_pred_folds[fold_idx])

            is_ensemble = getattr(model, 'ensemble', False) and hasattr(model, '_complex_idx')

            if is_ensemble:
                complex_idx, simple_idx = model._complex_idx, model._simple_idx
                main_ap = average_precision_score(
                    y_true[:, complex_idx], y_pred[:, complex_idx], average='macro')
            else:
                main_ap = average_precision_score(y_true, y_pred, average='macro')

            main_params = model.get_params(deep=False)
            for k in _MAIN_BOOKKEEPING_COLS:
                main_params.pop(k, None)

            main_rows.append({
                'variant': model_name, 'trial': trial, 'fold': fold_idx,
                'fold_AP': main_ap, **main_params,
            })

            if is_ensemble and hasattr(model, 'simple_model_'):
                simple_ap = average_precision_score(
                    y_true[:, simple_idx], y_pred[:, simple_idx], average='macro')
                simple_params = model.simple_model_.get_params(deep=False)
                for k in _SIMPLE_BOOKKEEPING_COLS:
                    simple_params.pop(k, None)

                simple_rows.append({
                    'variant': model_name, 'trial': trial, 'fold': fold_idx,
                    'fold_AP': simple_ap, **simple_params,
                })

    return pd.DataFrame(main_rows), pd.DataFrame(simple_rows)


def plot_loss_curves(all_results, variant=None, max_folds=5):
    """
    Plots the final-refit train-loss curve per fold for the main model
    (one figure per (trial, variant)), and, where ensemble=True, a separate
    figure per fold for the linear probe's own train-loss curve.

    Both curves are TRAIN loss only -- see the module docstring for why
    val_loss isn't meaningful at this stage.
    """
    for res in all_results:
        if variant is not None and res['model'] != variant:
            continue

        model_name, trial = res['model'], res['trial']

        fig, ax = plt.subplots(figsize=(6, 4))
        for fold_idx, curve in enumerate(res['train_histories'][:max_folds]):
            ax.plot(curve, label=f"fold {fold_idx}")
        ax.set_title(f"{model_name} (trial {trial}) -- main model train loss")
        ax.set_xlabel("epoch")
        ax.set_ylabel("train loss")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.show()

        for fold_idx, model in enumerate(res['best_models'][:max_folds]):
            simple_model = getattr(model, 'simple_model_', None)
            if simple_model is None or not hasattr(simple_model, 'history_'):
                continue
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(simple_model.history_['train_loss'])
            ax.set_title(f"{model_name} (trial {trial}, fold {fold_idx}) -- linear probe train loss")
            ax.set_xlabel("epoch")
            ax.set_ylabel("train loss")
            plt.tight_layout()
            plt.show()


def plot_inner_cv_curves(all_results, variant=None, max_folds=5, include_ensemble=True):
    """
    Plots the INNER-CV curves actually used for hyperparameter/epoch
    selection -- distinct from plot_loss_curves, which plots the final
    refit's train-only loss (no validation split at that stage, see module
    docstring point 1). For each (trial, variant, outer fold), draws one
    figure with two panels: mean train-loss vs. mean val-loss (left) and mean
    val-AP (right), each averaged across that fold's inner-CV splits.

    Only populated for output from abmil_classifier_tuned_optuna -- results
    from abmil_classifier_tuned / abmil_classifier_quick don't run the inner
    Optuna search these curves come from, so results without
    'inner_cv_train_loss_curves' are skipped with a printed note.

    When ensemble=True was used for a given fold, also plots a second figure
    for that fold with the linear probe's own inner-CV curves (read from
    model.simple_model_inner_cv_history_ -- see module docstring point 3).

    Parameters
    ----------
    all_results : list of dict
        Output of abmil_classifier_tuned_optuna.
    variant : str or None
        Restrict to this variant. None = all variants present.
    max_folds : int
        Cap on number of outer folds plotted per (trial, variant).
    include_ensemble : bool
        Whether to also plot the ensemble linear probe's inner-CV curves.
        Skipped automatically for folds/models where ensemble wasn't used or
        no inner-CV history was recorded.
    """
    for res in all_results:
        if variant is not None and res['model'] != variant:
            continue
        if 'inner_cv_train_loss_curves' not in res:
            print(f"Skipping {res['model']} (trial {res['trial']}): no inner_cv_*_curves "
                  f"in this result -- only abmil_classifier_tuned_optuna populates these.")
            continue

        model_name, trial = res['model'], res['trial']
        train_curves = res['inner_cv_train_loss_curves']
        val_curves = res['inner_cv_val_loss_curves']
        ap_curves = res['inner_cv_val_ap_curves']
        n_folds = min(max_folds, len(train_curves))

        for fold_idx in range(n_folds):
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))

            axes[0].plot(train_curves[fold_idx], label='train loss')
            axes[0].plot(val_curves[fold_idx], label='val loss')
            axes[0].set_title(f"{model_name} (trial {trial}, fold {fold_idx})\nmain model inner-CV loss")
            axes[0].set_xlabel("epoch")
            axes[0].set_ylabel("loss")
            axes[0].legend(fontsize=8)

            axes[1].plot(ap_curves[fold_idx], color='seagreen')
            axes[1].set_title("main model inner-CV val AP")
            axes[1].set_xlabel("epoch")
            axes[1].set_ylabel("val AP (macro)")

            plt.tight_layout()
            plt.show()

        if include_ensemble:
            for fold_idx, model in enumerate(res['best_models'][:max_folds]):
                history = getattr(model, 'simple_model_inner_cv_history_', None)
                if not history or history.get('mean_train_loss_curve') is None:
                    continue

                fig, axes = plt.subplots(1, 2, figsize=(10, 4))

                axes[0].plot(history['mean_train_loss_curve'], label='train loss')
                axes[0].plot(history['mean_val_loss_curve'], label='val loss')
                axes[0].set_title(f"{model_name} (trial {trial}, fold {fold_idx})\n"
                                   f"ensemble linear probe inner-CV loss")
                axes[0].set_xlabel("epoch")
                axes[0].set_ylabel("loss")
                axes[0].legend(fontsize=8)

                axes[1].plot(history['mean_val_ap_curve'], color='seagreen')
                axes[1].set_title("linear probe inner-CV val AP")
                axes[1].set_xlabel("epoch")
                axes[1].set_ylabel("val AP (macro)")

                plt.tight_layout()
                plt.show()


def inner_cv_best_epoch_summary(all_results, variant=None):
    """
    Tabular counterpart to plot_inner_cv_curves: for each (variant, trial,
    outer fold), reports the epoch and value of the peak inner-CV val-AP
    curve for the main model -- i.e. the epoch that was actually selected as
    "best" during that fold's hyperparameter search, and how good it looked
    on the held-out inner folds -- plus the same for the ensemble linear
    probe where available.

    Only populated for output from abmil_classifier_tuned_optuna (see module
    docstring point 2); results without 'inner_cv_val_ap_curves' contribute
    no rows.

    Parameters
    ----------
    all_results : list of dict
        Output of abmil_classifier_tuned_optuna.
    variant : str or None
        Restrict to this variant. None = all variants present.

    Returns
    -------
    pd.DataFrame with columns: variant, trial, fold, best_epoch, best_val_AP,
    simple_best_epoch, simple_best_val_AP. The last two are NaN for folds
    where ensemble wasn't used or no linear-probe inner-CV history was found.
    """
    rows = []
    for res in all_results:
        if variant is not None and res['model'] != variant:
            continue
        if 'inner_cv_val_ap_curves' not in res:
            continue

        model_name, trial = res['model'], res['trial']
        ap_curves = res['inner_cv_val_ap_curves']
        best_models = res['best_models']

        for fold_idx, ap_curve in enumerate(ap_curves):
            ap_curve = np.asarray(ap_curve)
            best_epoch = int(np.argmax(ap_curve)) + 1
            best_val_ap = float(ap_curve[best_epoch - 1])

            simple_best_epoch = np.nan
            simple_best_val_ap = np.nan
            if fold_idx < len(best_models):
                history = getattr(best_models[fold_idx], 'simple_model_inner_cv_history_', None)
                if history and history.get('mean_val_ap_curve') is not None:
                    simple_ap_curve = np.asarray(history['mean_val_ap_curve'])
                    simple_best_epoch = int(np.argmax(simple_ap_curve)) + 1
                    simple_best_val_ap = float(simple_ap_curve[simple_best_epoch - 1])

            rows.append({
                'variant': model_name, 'trial': trial, 'fold': fold_idx,
                'best_epoch': best_epoch, 'best_val_AP': best_val_ap,
                'simple_best_epoch': simple_best_epoch, 'simple_best_val_AP': simple_best_val_ap,
            })

    return pd.DataFrame(rows)


def hyperparam_performance_summary(params_df, metric_col='fold_AP',
                                    id_cols=('variant', 'trial', 'fold')):
    """
    For each hyperparameter column in params_df (as returned by
    collect_results), groups by its EXACT value and reports mean/std/count of
    metric_col -- appropriate for hyperparameters drawn from a small discrete
    set (hidden_dim, attention_dim, batch_size, lstm_hidden_dim, pooling, ...).
    For hyperparameters sampled from a continuous range (dropout,
    learning_rate, weight_decay), exact-value grouping is close to useless
    since almost every row has a distinct value -- use
    hyperparam_range_summary for those instead. Columns that never vary (e.g.
    n_epochs, if it was fixed) are skipped since there's nothing to compare.

    Returns
    -------
    dict[str, pd.DataFrame] : one entry per hyperparameter column, each a
        DataFrame indexed by that hyperparameter's values, sorted by mean
        metric_col descending.
    """
    if params_df.empty:
        return {}

    param_cols = [c for c in params_df.columns if c not in id_cols and c != metric_col]
    summaries = {}
    for col in param_cols:
        if params_df[col].nunique(dropna=False) <= 1:
            continue
        summaries[col] = (
            params_df.groupby(col)[metric_col]
            .agg(['mean', 'std', 'count'])
            .sort_values('mean', ascending=False)
        )
    return summaries


def hyperparam_range_summary(params_df, columns=None, metric_col='fold_AP',
                              bins=5, scale='auto',
                              id_cols=('variant', 'trial', 'fold')):
    """
    Bins continuous hyperparameter columns into ranges and reports
    count/mean/std of metric_col per bin -- e.g. for dropout: how many rows
    fall in [0.1, 0.2), [0.2, 0.3), etc., and what their average fold_AP was.
    Complements hyperparam_performance_summary, which groups by exact value
    and is more suited to small discrete search spaces (hidden_dim,
    attention_dim, batch_size, ...).

    Parameters
    ----------
    params_df : pd.DataFrame
        As returned by collect_results (main_params_df or simple_params_df).
    columns : list of str or None
        Which columns to bin. None = auto-detect: every numeric column not in
        id_cols/metric_col whose number of distinct values exceeds `bins`
        (columns with fewer distinct values than bins are better read with
        hyperparam_performance_summary instead, so they're skipped here).
        Pass explicitly (e.g. ['dropout', 'learning_rate', 'weight_decay']) to
        control exactly which columns get binned.
    metric_col : str
        Column to aggregate within each bin.
    bins : int
        Number of bins per column.
    scale : {'auto', 'linear', 'log'}
        Bin edges spacing. 'auto' picks 'log' (edges spaced as powers of 10)
        for columns that are strictly positive and span at least one order of
        magnitude (typical for learning_rate/weight_decay, which are sampled
        log-uniformly) -- equal-width linear bins on such a column would dump
        almost everything into one bin near the small end. Everything else
        (e.g. dropout) gets 'linear' (equal-width bins).

    Returns
    -------
    dict[str, pd.DataFrame] : one entry per binned column, each a DataFrame
        indexed by the bin interval (e.g. "(0.10, 0.20]"), with columns
        ['count', 'mean', 'std'], ordered low-to-high by bin.
    """
    if params_df.empty:
        return {}

    if columns is None:
        candidate_cols = [c for c in params_df.columns if c not in id_cols and c != metric_col]
        columns = [
            c for c in candidate_cols
            if pd.api.types.is_numeric_dtype(params_df[c])
            and params_df[c].nunique(dropna=False) > bins
        ]

    summaries = {}
    for col in columns:
        series = params_df[col].dropna()
        if series.nunique() <= 1:
            continue

        col_scale = scale
        if col_scale == 'auto':
            positive = (series > 0).all()
            spans_order_of_magnitude = positive and (series.max() / series.min() >= 10)
            col_scale = 'log' if spans_order_of_magnitude else 'linear'

        if col_scale == 'log':
            edges = np.logspace(np.log10(series.min()), np.log10(series.max()), bins + 1)
        else:
            edges = np.linspace(series.min(), series.max(), bins + 1)
        edges = np.unique(edges)  # guard against degenerate/duplicate edges
        if len(edges) < 2:
            continue

        # precision=10: pandas' default precision=3 mis-rounds the
        # include_lowest adjustment on very small/log-spaced edges (e.g.
        # learning_rate, weight_decay), producing a nonsensical negative
        # left edge on the first bin. A higher precision avoids that; we
        # reformat the labels ourselves afterward to keep them readable.
        binned = pd.cut(params_df[col], bins=edges, include_lowest=True, precision=10)
        summary = (
            params_df.groupby(binned, observed=True)[metric_col]
            .agg(['count', 'mean', 'std'])
        )
        summary.index = [f"({iv.left:.4g}, {iv.right:.4g}]" for iv in summary.index]
        summaries[col] = summary

    return summaries


def plot_hyperparameter_pairplot(params_df, metric_col='fold_AP', columns=None,
                                  variant=None, cmap='viridis', figsize=None,
                                  diag='hist'):
    """
    Plots every pairwise combination of hyperparameter columns in params_df
    as a grid of scatter plots colored by metric_col, with each
    hyperparameter's own marginal distribution on the diagonal -- e.g. 5
    varying hyperparameters (as with the LSTM_only variant: hidden_dim,
    dropout, learning_rate, weight_decay, batch_size) gives a 5x5 = 25-cell
    grid. This surfaces interactions between two hyperparameters at once
    (e.g. "high AP only when learning_rate is low AND dropout is high") that
    hyperparam_performance_summary / hyperparam_range_summary can't show,
    since those look at one hyperparameter at a time.

    Only columns with more than one distinct value are included (constant
    columns -- e.g. structural flags fixed by the variant -- carry no
    information in a pairplot and are dropped automatically).

    Parameters
    ----------
    params_df : pd.DataFrame
        As returned by collect_results (main_params_df or simple_params_df).
    metric_col : str
        Column used to color each scatter point (default 'fold_AP').
    columns : list of str or None
        Which hyperparameter columns to include as grid axes. None =
        auto-detect: every column that isn't 'variant'/'trial'/'fold'/
        metric_col and has more than one distinct value. Pass explicitly to
        control the grid size / which hyperparameters appear.
    variant : str or None
        If params_df spans multiple variants, restrict the plot to this one
        -- mixed variants may not share the same hyperparameters (e.g.
        lstm_hidden_dim only exists for LSTM variants), which would leave
        gaps or force an inconsistent grid otherwise.
    cmap : str
        Matplotlib colormap name for the metric_col color scale.
    figsize : tuple or None
        Passed to plt.subplots; defaults to roughly 2.2 inches per cell.
    diag : {'hist', 'none'}
        What to draw on the diagonal cells: each hyperparameter's own value
        distribution, or nothing.
    """
    if params_df.empty:
        print("params_df is empty -- nothing to plot.")
        return

    df = params_df if variant is None else params_df[params_df['variant'] == variant]
    if df.empty:
        print(f"No rows found for variant={variant!r}.")
        return

    id_cols = ('variant', 'trial', 'fold')
    if columns is None:
        candidate_cols = [c for c in df.columns if c not in id_cols and c != metric_col]
        columns = [c for c in candidate_cols if df[c].nunique(dropna=False) > 1]

    if not columns:
        print("No varying hyperparameter columns found to plot.")
        return

    n = len(columns)
    if figsize is None:
        figsize = (2.2 * n, 2.2 * n)

    fig, axes = plt.subplots(n, n, figsize=figsize, squeeze=False)

    metric_vals = df[metric_col].astype(float).values
    vmin, vmax = np.nanmin(metric_vals), np.nanmax(metric_vals)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    colormap = plt.get_cmap(cmap)

    def _numeric_values(col):
        """Returns (plot_values, category_labels_or_None) for a column."""
        series = df[col]
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            return series.astype(float).values, None
        categories = sorted(series.astype(str).unique())
        cat_to_pos = {c: i for i, c in enumerate(categories)}
        return series.astype(str).map(cat_to_pos).values.astype(float), categories

    col_values = {col: _numeric_values(col) for col in columns}

    for i, row_col in enumerate(columns):
        y_vals, y_cats = col_values[row_col]
        for j, col_col in enumerate(columns):
            ax = axes[i][j]
            x_vals, x_cats = col_values[col_col]

            if i == j:
                if diag == 'hist':
                    n_bins = min(10, max(2, df[col_col].nunique()))
                    ax.hist(x_vals, bins=n_bins, color='steelblue', alpha=0.85)
                ax.set_yticks([])
            else:
                ax.scatter(x_vals, y_vals, c=metric_vals, cmap=colormap, norm=norm,
                           s=25, alpha=0.85, edgecolors='none')

            if x_cats is not None:
                ax.set_xticks(range(len(x_cats)))
                ax.set_xticklabels(x_cats, fontsize=6, rotation=45, ha='right')
            else:
                ax.tick_params(labelsize=6)

            if i != j and y_cats is not None:
                ax.set_yticks(range(len(y_cats)))
                ax.set_yticklabels(y_cats, fontsize=6)
            elif i != j:
                ax.tick_params(labelsize=6)

            ax.set_ylabel(row_col if j == 0 else "", fontsize=8)
            ax.set_xlabel(col_col if i == n - 1 else "", fontsize=8)

    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02, label=metric_col)

    title = f"Hyperparameter pairwise combinations ({len(df)} configs, {n}x{n} grid)"
    if variant is not None:
        title += f" -- {variant}"
    fig.suptitle(title, y=1.02)

    plt.show()


def combination_frequency_summary(params_df, columns=('attention_dim', 'hidden_dim', 'lstm_hidden_dim'),
                                   metric_col='fold_AP'):
    """
    Groups params_df by the joint combination of `columns` (e.g.
    attention_dim + hidden_dim + lstm_hidden_dim together) and reports how
    often each exact combination was the winning config plus its mean/std
    fold_AP -- answers "which combinations of these hyperparameters get
    picked together most often, and how well do they do", as opposed to
    hyperparam_performance_summary (one hyperparameter at a time) or
    plot_hyperparameter_pairplot (every pairwise relationship at once).

    Parameters
    ----------
    params_df : pd.DataFrame
        As returned by collect_results (main_params_df or simple_params_df).
        If it spans multiple variants, filter to one first
        (params_df[params_df['variant'] == 'LSTM_only']) -- a variant that
        doesn't use one of `columns` (e.g. attention_dim for a 'mean'-pooling
        variant, or lstm_hidden_dim for a non-LSTM variant) will just be
        missing that column entirely, which is handled below, but mixing
        variants that DO and DON'T have a column together will silently
        merge NaN-column rows into one group -- best kept to a single variant.
    columns : tuple of str
        Which columns to combine. Columns not present in params_df (e.g.
        lstm_hidden_dim for a non-LSTM variant) are dropped with a printed
        note rather than raising, so the same call works across variants.
    metric_col : str
        Column to aggregate within each combination.

    Returns
    -------
    pd.DataFrame with one row per distinct combination of `columns` seen in
    params_df: the combination's own columns, plus `count`,
    `mean_<metric_col>`, `std_<metric_col>` -- sorted by count descending
    (most common combination first).
    """
    if params_df.empty:
        print("params_df is empty -- nothing to summarize.")
        return pd.DataFrame()

    present = [c for c in columns if c in params_df.columns]
    missing = [c for c in columns if c not in params_df.columns]
    if missing:
        print(f"Skipping columns not present in params_df: {missing}")
    if not present:
        print("None of the requested columns are present in params_df.")
        return pd.DataFrame()

    summary = (
        params_df.groupby(list(present), dropna=False)[metric_col]
        .agg(['count', 'mean', 'std'])
        .reset_index()
        .rename(columns={'mean': f'mean_{metric_col}', 'std': f'std_{metric_col}'})
        .sort_values('count', ascending=False)
        .reset_index(drop=True)
    )
    return summary