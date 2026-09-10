"""
Linear probe baseline: training loop + Optuna-tuned nested-CV evaluation,
mirroring train_abmil / abmil_classifier_tuned_optuna's structure and output
schema, but for the LinearProbe (nn.Dropout -> nn.Linear) on ALREADY
MEAN-POOLED, fixed-length inputs.

Key structural difference from the ABMIL/LSTM versions: since pooling has
already happened (X is (n_bags, n_features), not variable-length bags), there
is no need for the per-bag training loop or the custom collate_fn train_abmil
needs for variable-length MIL bags. Training here uses standard vectorized
minibatches via a plain DataLoader, so it will be noticeably faster per epoch
than the MIL versions -- this isn't a shortcut, it's a genuine consequence of
the inputs already being fixed-size vectors.

Only one "model" is being tuned here (no variant selection), but the output
list still tags each result 'model': 'LinearProbe' so it drops straight into
your existing plotting/statistical-comparison functions
(plot_variant_comparison_boxplots, compare_variants_per_encoder, etc.)
alongside ABMIL / ABMIL_LSTM_residual / LSTM_only_residual / LSTM_last.
"""

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import average_precision_score, roc_auc_score
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import optuna


# ============================================================================
# DATASET
# ============================================================================

class LinearProbeDataset(Dataset):
    """
    Dataset for the linear probe. Unlike MILDataset, every item is already a
    fixed-length (n_features,) vector (one already-pooled embedding per
    recording), so no custom collate_fn is needed -- the default PyTorch
    collate handles regular batching fine.
    """

    def __init__(self, X, y, scaler=None, fit_scaler=False):
        X = np.asarray(X)
        self.y = torch.FloatTensor(np.asarray(y))
        self.scaler = scaler

        if fit_scaler and scaler is not None:
            scaler.fit(X)

        X_scaled = scaler.transform(X) if scaler is not None else X
        self.X = torch.FloatTensor(X_scaled)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================================
# MODEL
# ============================================================================

class LinearProbe(nn.Module):
    """
    nn.Dropout(input) -> nn.Linear(n_features, n_labels). No hidden layers,
    no attention, no temporal context -- the "no learned architecture" floor
    for comparison against ABMIL / ABMIL_LSTM_residual / LSTM_only_residual /
    LSTM_last. Dropout is applied to the input embedding itself (there's
    nothing else to apply it to with a single linear layer).
    """

    def __init__(self, n_features, n_labels, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(n_features, n_labels)

    def forward(self, x):
        return self.linear(self.dropout(x))


# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

def train_linear_probe(
    X_train, y_train,
    X_val=None, y_val=None,
    n_labels=5,
    n_epochs=50,
    batch_size=16,
    learning_rate=1e-3,
    weight_decay=1e-5,
    dropout=0.0,
    device='cpu',
    verbose=True,
    random_state=42,
    early_stopping=True,
):
    """
    Same role/output shape as train_abmil, for the LinearProbe on already
    mean-pooled (n_samples, n_features) inputs.

    Parameters mirror train_abmil where they overlap (n_labels, n_epochs,
    batch_size, learning_rate, weight_decay, device, verbose, random_state).
    early_stopping=False (used by the Optuna inner-loop objective, same
    pattern as your ABMIL version) runs the full n_epochs_max every time so a
    best-epoch can be selected post-hoc from the validation-AP curve.

    Returns
    -------
    model : LinearProbe
    history : dict with 'train_loss', 'val_loss', 'val_ap', 'val_auc'
    scaler : StandardScaler
    """
    device = torch.device(device)
    scaler = StandardScaler()

    g = torch.Generator()
    g.manual_seed(random_state)

    train_dataset = LinearProbeDataset(X_train, y_train, scaler=scaler, fit_scaler=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)

    if X_val is not None:
        val_dataset = LinearProbeDataset(X_val, y_val, scaler=scaler, fit_scaler=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, generator=g)
    else:
        val_loader = None

    n_features = train_dataset.X.shape[1]

    model = LinearProbe(n_features=n_features, n_labels=n_labels, dropout=dropout).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if early_stopping:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    history = {'train_loss': [], 'val_loss': [], 'val_ap': [], 'val_auc': []}

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = model.state_dict().copy()

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            all_y_true, all_y_pred = [], []

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    logits = model(X_batch)
                    loss = criterion(logits, y_batch)
                    val_loss += loss.item()

                    all_y_true.append(y_batch.cpu().numpy())
                    all_y_pred.append(logits.detach().cpu().numpy())

            val_loss /= len(val_loader)
            history['val_loss'].append(val_loss)

            all_y_true = np.concatenate(all_y_true, axis=0)
            all_y_pred = np.concatenate(all_y_pred, axis=0)

            val_ap = average_precision_score(all_y_true, all_y_pred, average='macro')
            val_auc = roc_auc_score(all_y_true, all_y_pred, average='macro')
            history['val_ap'].append(val_ap)
            history['val_auc'].append(val_auc)

            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val AP: {val_ap:.4f} | Val AUC: {val_auc:.4f}")

            if early_stopping:
                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1

                if patience_counter >= 10:
                    print(f"Early stopping at epoch {epoch+1}")
                    model.load_state_dict(best_model_state)
                    break
        else:
            history['val_loss'].append(np.nan)
            history['val_ap'].append(np.nan)
            history['val_auc'].append(np.nan)

    return model, history, scaler


def predict_linear_probe(model, X, scaler, device='cpu'):
    """Get predictions from a trained LinearProbe. Mirrors predict_abmil's role."""
    device = torch.device(device)
    model.to(device)
    model.eval()

    X_scaled = scaler.transform(np.asarray(X))
    X_tensor = torch.FloatTensor(X_scaled).to(device)

    with torch.no_grad():
        logits = model(X_tensor)
        y_pred = torch.sigmoid(logits).detach().cpu().numpy()

    return y_pred


class LinearProbeSklearnWrapper(BaseEstimator, ClassifierMixin):
    """
    SKLearn-style wrapper for LinearProbe, mirroring ABMILSklearnWrapper's
    role (fit/predict_proba/predict) so it plugs into the same final-refit
    step as the ABMIL variants. No ensemble/ultra-minority fallback here --
    that logic is specific to ABMIL and wasn't requested for this baseline.
    """
    def __init__(self, n_labels=5, n_epochs=20, batch_size=16, dropout=0.0,
                 learning_rate=1e-3, weight_decay=1e-5, device='cpu',
                 random_state=42, early_stopping=True):
        self.n_labels = n_labels
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = device
        self.random_state = random_state
        self.early_stopping = early_stopping

    def fit(self, X, y):
        self.model_, self.history_, self.scaler_ = train_linear_probe(
            X_train=X, y_train=y,
            n_labels=self.n_labels, n_epochs=self.n_epochs, batch_size=self.batch_size,
            dropout=self.dropout, learning_rate=self.learning_rate, weight_decay=self.weight_decay,
            device=self.device, verbose=False, random_state=self.random_state,
            early_stopping=self.early_stopping,
        )
        return self

    def predict_proba(self, X):
        return predict_linear_probe(self.model_, X, self.scaler_, device=self.device)

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)

    @property
    def classes_(self):
        return [np.array([0, 1]) for _ in range(self.n_labels)]


# ============================================================================
# OPTUNA SEARCH SPACE
# ============================================================================

def _suggest_linear_probe_params(trial):
    """
    Search space for the linear probe: no hidden_dim/attention_dim (there's
    no hidden layer), no use_lstm/pooling choice (pooling is fixed to mean
    and already applied before this data reaches these functions). Batch size
    can afford to go higher than the MIL versions' range since this trains on
    real vectorized minibatches rather than one bag at a time -- adjust the
    list if you want it to match your MIL batch_size range exactly instead.
    """
    return dict(
        dropout = 0.0,
        learning_rate=trial.suggest_categorical("learning_rate",[5e-5,1e-4,5e-4]),
        weight_decay=trial.suggest_categorical("weight_decay",[1e-5]),
        batch_size=trial.suggest_categorical("batch_size", [4]),
    )


def _make_linear_probe_optuna_objective(X_train, y_train, n_split_in, n_epochs_max, seed):
    """
    In addition to the mean validation-AP curve (used to pick the best
    epoch/config), this also tracks the mean training-loss and
    validation-loss curves across the inner CV folds, stored as trial user
    attrs alongside mean_val_ap_curve -- see linear_probe_tuned_optuna, which
    pulls the winning trial's curves into 'inner_cv_train_loss_curves' /
    'inner_cv_val_loss_curves' / 'inner_cv_val_ap_curves' in its results.
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    inner_cv = MultilabelStratifiedKFold(n_splits=n_split_in, shuffle=True, random_state=seed)

    def objective(trial):
        params = _suggest_linear_probe_params(trial)

        fold_val_ap_curves = []
        fold_train_loss_curves = []
        fold_val_loss_curves = []
        for tr_idx, val_idx in inner_cv.split(X_train, y_train):
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]

            _, history, _ = train_linear_probe(
                X_train=X_tr, y_train=y_tr,
                X_val=X_val, y_val=y_val,
                n_labels=y_train.shape[1], n_epochs=n_epochs_max,
                batch_size=params["batch_size"], verbose=False, random_state=seed,
                early_stopping=False,
                dropout=params["dropout"],
                learning_rate=params["learning_rate"], weight_decay=params["weight_decay"],
            )
            fold_val_ap_curves.append(history["val_ap"])
            fold_train_loss_curves.append(history["train_loss"])
            fold_val_loss_curves.append(history["val_loss"])

        mean_ap_curve = np.mean(fold_val_ap_curves, axis=0)
        mean_train_loss_curve = np.mean(fold_train_loss_curves, axis=0)
        mean_val_loss_curve = np.mean(fold_val_loss_curves, axis=0)
        best_epoch = int(np.argmax(mean_ap_curve)) + 1
        print(best_epoch)
        print(mean_ap_curve[best_epoch-1])

        trial.set_user_attr("mean_val_ap_curve", mean_ap_curve.tolist())
        trial.set_user_attr("mean_train_loss_curve", mean_train_loss_curve.tolist())
        trial.set_user_attr("mean_val_loss_curve", mean_val_loss_curve.tolist())

        return float(mean_ap_curve[n_epochs_max - 1])

    return objective


# ============================================================================
# NESTED-CV TUNED EVALUATION
# ============================================================================

def linear_probe_tuned_optuna(X, y, n_split_out=5, n_split_in=5, num_trials=5,
                               random_state=42, n_optuna_trials=20, n_epochs_max=40):
    """
    Same role and output schema as abmil_classifier_tuned_optuna, but for the
    LinearProbe baseline on already mean-pooled (n_samples, n_features)
    inputs -- pool your windows down to one vector per recording BEFORE
    calling this (no bag-of-windows handling happens in here).

    There's no `variants` loop since there's only one model here; each result
    dict is tagged 'model': 'LinearProbe' so it drops straight into your
    existing plotting/statistical-comparison functions alongside the ABMIL /
    LSTM variants.

    Parameters
    ----------
    X : array-like, shape (n_bags, n_features)
        Already mean-pooled per-recording embeddings.
    y : array-like, shape (n_bags, n_labels)
    n_split_out, n_split_in : outer/inner CV splits
    num_trials : int
        Repeated CV runs with different seeds, same as the other tuned functions.
    random_state : int
    n_optuna_trials : int
        Number of Optuna trials per (outer fold) -- same role as n_optuna_trials
        in abmil_classifier_tuned_optuna.
    n_epochs_max : int
        Fixed epoch budget for every inner-loop trial (no early stopping during
        the search itself); the best epoch is picked post-hoc from the mean
        validation-AP curve, same pattern as the ABMIL/LSTM version.

    Returns
    -------
    list of dict, one per trial, same schema as abmil_classifier_tuned_optuna's
    output (with 'model' always 'LinearProbe'), plus
    'inner_cv_train_loss_curves' / 'inner_cv_val_loss_curves' /
    'inner_cv_val_ap_curves' -- one list entry per outer fold, each the
    winning trial's mean curve across that fold's inner CV (distinct from
    'train_histories'/'val_histories', which are the FINAL REFIT's curves on
    the full outer-train fold with no validation split, so val is NaN there).
    """
    X = np.asarray(X)
    y = np.array(y)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = []

    for i in range(num_trials):
        trial_seed = random_state + i
        print(f"Starting Trial {i+1}/{num_trials} with random_state={trial_seed}...")

        np.random.seed(trial_seed)
        random.seed(trial_seed)
        torch.manual_seed(trial_seed)
        torch.cuda.manual_seed_all(trial_seed)

        outer_cv = MultilabelStratifiedKFold(n_splits=n_split_out, shuffle=True, random_state=trial_seed)

        print("  Tuning and evaluating model: LinearProbe")
        all_y_true, all_y_pred_proba, all_test_indices, best_models = [], [], [], []
        outer_scores = []
        fold_train_histories, fold_val_histories = [], []
        all_best_params = []
        # Inner-CV (mean-across-inner-folds) curves for the winning trial in
        # each outer fold -- distinct from fold_train_histories/
        # fold_val_histories above, which are the FINAL REFIT's curves on the
        # full outer-train fold (no validation split, so val is NaN there).
        fold_inner_train_loss_curves, fold_inner_val_loss_curves = [], []
        fold_inner_val_ap_curves = []

        for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
            print(f"    Evaluating fold {fold+1}/{n_split_out}")
            X_train_fold, X_test_fold = X[train_idx], X[test_idx]
            y_train_fold, y_test_fold = y[train_idx], y[test_idx]

            # --- Optuna study: tunes hyperparams + picks epoch count on inner CV ---
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=trial_seed * 100 + fold, multivariate=True),
                study_name=f"LinearProbe_trial{i}_fold{fold}",
            )
            study.optimize(
                _make_linear_probe_optuna_objective(X_train_fold, y_train_fold,
                                                      n_split_in, n_epochs_max, trial_seed),
                n_trials=n_optuna_trials,
            )

            best_params = study.best_params
            best_trial_attrs = study.best_trial.user_attrs
            fold_inner_train_loss_curves.append(best_trial_attrs.get("mean_train_loss_curve"))
            fold_inner_val_loss_curves.append(best_trial_attrs.get("mean_val_loss_curve"))
            fold_inner_val_ap_curves.append(best_trial_attrs.get("mean_val_ap_curve"))

            # --- Refit on the FULL outer-train fold at best hyperparams/epoch ---
            final_model = LinearProbeSklearnWrapper(
                n_labels=y.shape[1], n_epochs=n_epochs_max, batch_size=best_params["batch_size"],
                dropout=0.0, learning_rate=best_params["learning_rate"],
                weight_decay=best_params["weight_decay"], device=device,
                random_state=trial_seed,
            )
            final_model.fit(X_train_fold, y_train_fold)

            fold_train_histories.append(final_model.history_['train_loss'])
            fold_val_histories.append(final_model.history_['val_loss'])  # NaNs (no val set passed at this stage)

            y_pred_proba = final_model.predict_proba(X_test_fold)

            fold_score = average_precision_score(y_test_fold, y_pred_proba, average='macro')
            outer_scores.append(fold_score)

            all_y_true.append(y_test_fold)
            all_y_pred_proba.append(y_pred_proba)
            all_test_indices.append(test_idx)
            best_models.append(final_model)
            all_best_params.append(best_params)

        y_true_cv, y_pred_proba_cv = all_y_true, all_y_pred_proba
        all_y_true = np.concatenate(all_y_true, axis=0)
        all_y_pred_proba = np.concatenate(all_y_pred_proba, axis=0)
        all_test_indices = np.concatenate(all_test_indices, axis=0)

        all_results.append({
            'trial': i,
            'model': 'LinearProbe',
            'best_models': best_models,
            'mean_AP': np.mean(outer_scores),
            'std_AP': np.std(outer_scores, ddof=1),
            'y_true_cv': y_true_cv,
            'y_pred_proba_cv': y_pred_proba_cv,
            'oof_y_true': all_y_true,
            'oof_y_pred_proba': all_y_pred_proba,
            'oof_indices': all_test_indices,
            'train_histories': fold_train_histories,
            'val_histories': fold_val_histories,
            'best_params': all_best_params,
            'inner_cv_train_loss_curves': fold_inner_train_loss_curves,
            'inner_cv_val_loss_curves': fold_inner_val_loss_curves,
            'inner_cv_val_ap_curves': fold_inner_val_ap_curves,
        })

    return all_results


# ============================================================================
# USAGE EXAMPLE
# ============================================================================
#
#   # X_pooled: (n_bags, n_features) -- your windows already mean-pooled per recording
#   results_linear_probe = linear_probe_tuned_optuna(
#       X_pooled, y, n_split_out=5, n_split_in=5, num_trials=5,
#       n_optuna_trials=20, n_epochs_max=50,
#   )
#
#   # Drops straight into your existing per-encoder result dicts alongside the
#   # ABMIL/LSTM variants, e.g.:
#   #   results_linear_probe_per_encoder = {'perch2': results_linear_probe, ...}
