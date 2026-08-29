"""
Multiple Instance Learning (MIL) for Multi-Label Bioacoustic Classification.

This module implements an Attention-Based Multiple Instance Learning (ABMIL) model
for multi-label classification of bioacoustic recordings, with an OPTIONAL temporal
context layer (a bidirectional LSTM) inserted before pooling.

Architectural variants are supported via `use_lstm`, `lstm_residual`,
`lstm_residual_proj`, and `pooling`:

  1. 'ABMIL'                    : use_lstm=False, pooling='attention'
                                  -> your original ABMIL, unchanged.
  2. 'ABMIL_LSTM'               : use_lstm=True, lstm_residual=False, pooling='attention'
                                  -> recurrent-context ABMIL. Instance embeddings are passed
                                  through a BiLSTM to inject temporal context, THEN pooled with
                                  the same gated attention as ABMIL. The LSTM output fully
                                  replaces the pre-LSTM representation.
  3. 'ABMIL_LSTM_residual'      : use_lstm=True, lstm_residual=True, pooling='attention'
                                  -> H_out = H + LSTM(H) (a skip connection around the LSTM).
                                  Requires lstm_hidden_dim * num_directions == hidden_dim (an
                                  exact-match residual add -- see 'ABMIL_LSTM_residual_proj'
                                  below if you want lstm_hidden_dim to be independently tunable).
  4. 'ABMIL_LSTM_residual_proj' : use_lstm=True, lstm_residual_proj=True, pooling='attention'
                                  -> H_out = Proj(H) + LSTM(H), where Proj is a single learned
                                  linear shortcut (hidden_dim -> lstm_hidden_dim*num_directions).
                                  This is the standard ResNet-style shortcut-projection trick,
                                  and it decouples lstm_hidden_dim from hidden_dim entirely, so
                                  lstm_hidden_dim can be tuned freely (see LSTM_PROJ_EXTRA_PARAM_GRID
                                  / _suggest_variant_params). Everything downstream (attention,
                                  classifier) then operates in the LSTM's output dimension rather
                                  than hidden_dim for this variant.
  5. 'LSTM_only'                : use_lstm=True, pooling='mean' -> no attention at all.
                                  Isolates whether temporal context alone (without learned
                                  instance weighting) explains any gain, as a control arm.
  6. 'LSTM_only_residual'       : same, with lstm_residual=True (exact-match skip connection).
  7. 'LSTM_residual_proj'       : use_lstm=True, lstm_residual_proj=True, pooling='mean'
                                  -> mean-pooled counterpart of ABMIL_LSTM_residual_proj: same
                                  projected-shortcut residual LSTM, no attention.
  8. 'LSTM_last'                : use_lstm=True, pooling='last' -> classic RNN-sequence-
                                  classifier style baseline using the LSTM's final state.
                                  Incompatible with lstm_residual and lstm_residual_proj (there
                                  is no per-timestep sequence of outputs left to add a shortcut
                                  to -- only the final hidden state survives).

All variants share the same feature_fc -> [optional (residual) LSTM] -> pooling ->
classifier skeleton, so they are directly comparable and swappable via constructor
arguments. Default behavior (use_lstm=False) is IDENTICAL to the original ABMIL
implementation -- nothing changes unless you opt in.

LABEL-GROUPED ENSEMBLE (ensemble=True / ensemble_labels=...):
Instead of the original sklearn-LogisticRegression fallback, `ensemble_labels`
now names the "simple" label indices (e.g. type_a) that get pulled OUT of the
main ABMIL/LSTM model entirely -- the main model is trained with
n_labels = (total labels - len(ensemble_labels)), i.e. it never sees or builds
attention branches for the simple labels at all. The simple labels are instead
predicted by a separate LinearProbe (Dropout -> Linear) trained on mean-pooled
raw instance embeddings. The two models are NOT trained jointly: the LinearProbe
gets its own independent nested-CV Optuna hyperparameter search every time
ABMILSklearnWrapper.fit() is called with ensemble=True, using the exact search
space / objective pattern from the linear-probe module (see
`_make_linear_probe_optuna_objective`). predict_proba() reassembles both
models' outputs into a single (n_samples, n_labels) array in the original
column order.
"""

import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.stats import loguniform
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    make_scorer,
    roc_auc_score,
)
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import optuna

# LinearProbe backend for the "simple label" arm of the ensemble -- separately
# trained/tuned, not part of the ABMIL/LSTM computation graph. Assumed to live
# alongside this module as src/linear_probe.py (mirrors the existing
# `from src.feature_generation import pool_features` import convention used
# further down in this file). Adjust the import path if your module lives
# elsewhere.
#
# NOTE: we deliberately do NOT import _make_linear_probe_optuna_objective from
# that module -- it hardcodes MultilabelStratifiedKFold, which raises on a
# single-column target (sklearn reports single-column 0/1 data as type
# 'binary', not 'multilabel-indicator', and iterstrat requires the latter).
# Since ensemble_labels is very often a single label (e.g. just type_a), we
# use our own copy below that falls back to plain StratifiedKFold in that case
# and MultilabelStratifiedKFold when ensemble_labels has 2+ entries.
from src.linear_probe import LinearProbeSklearnWrapper, train_linear_probe


def _suggest_ensemble_linear_probe_params(trial):
    """Same search space as the standalone linear-probe module's
    _suggest_linear_probe_params (dropout fixed at 0.0, since there's no
    hidden layer to regularize besides the input itself)."""
    return dict(
        dropout=0.0,
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        batch_size=trial.suggest_categorical("batch_size", [4, 8, 16]),
    )


def _make_ensemble_linear_probe_objective(X_train, y_train, n_split_in, n_epochs_max, seed):
    """
    Local counterpart to the linear-probe module's
    _make_linear_probe_optuna_objective, used for the ensemble's simple-label
    arm. Picks the CV splitter based on how many simple labels there are:
    StratifiedKFold for a single label (MultilabelStratifiedKFold rejects
    single-column targets), MultilabelStratifiedKFold for 2+.
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    single_label = y_train.shape[1] == 1

    if single_label:
        splitter = StratifiedKFold(n_splits=n_split_in, shuffle=True, random_state=seed)
        def split(X, y):
            return splitter.split(X, y.ravel())
    else:
        splitter = MultilabelStratifiedKFold(n_splits=n_split_in, shuffle=True, random_state=seed)
        def split(X, y):
            return splitter.split(X, y)

    def objective(trial):
        params = _suggest_ensemble_linear_probe_params(trial)

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        fold_val_ap_curves = []
        for tr_idx, val_idx in split(X_train, y_train):
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

        mean_ap_curve = np.mean(fold_val_ap_curves, axis=0)
        trial.set_user_attr("mean_val_ap_curve", mean_ap_curve.tolist())
        return float(mean_ap_curve[n_epochs_max - 1])

    return objective

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.DEBUG)


# ============================================================================
# DATASET CLASS  (unchanged)
# ============================================================================

class MILDataset(Dataset):
    """Dataset for Multiple Instance Learning."""

    def __init__(self, X_bags, y_labels, scaler=None, fit_scaler=False):
        """
        Parameters
        ----------
        X_bags : list of (n_windows, n_features) arrays
            One bag (recording) per element. IMPORTANT: bags can have different lengths!
            Bags are assumed to be ordered in time (window 0 first, window K-1 last) --
            this matters for the LSTM variants below, which are order-sensitive.
        y_labels : (n_bags, n_labels) array
            Multi-label targets
        scaler : StandardScaler or None
        fit_scaler : bool
            If True, fit scaler on this data
        """
        self.X_bags = X_bags
        self.y_labels = torch.FloatTensor(y_labels)
        self.scaler = scaler

        if fit_scaler and scaler is not None:
            all_instances = np.vstack(X_bags)
            scaler.fit(all_instances)

        self.X_bags_scaled = []
        for bag in X_bags:
            if scaler is not None:
                bag_scaled = scaler.transform(bag)
            else:
                bag_scaled = bag
            self.X_bags_scaled.append(torch.FloatTensor(bag_scaled))

    def __len__(self):
        return len(self.X_bags_scaled)

    def __getitem__(self, idx):
        return self.X_bags_scaled[idx], self.y_labels[idx]


def collate_mil(batch):
    """Custom collate function for MIL batches with variable-length bags."""
    bags = [item[0] for item in batch]
    labels = torch.stack([item[1] for item in batch])
    return bags, labels


# ============================================================================
# ABMIL MODEL  (extended with optional LSTM temporal encoder + proj residual)
# ============================================================================

class ABMIL(nn.Module):
    """
    Attention-Based Multiple Instance Learning, with an optional temporal
    context layer.

    Architecture:
    1. Feature extraction : Maps instance features through FC layer (-> hidden_dim)
    2. [Optional] Temporal encoder: BiLSTM over the ordered instance sequence,
       injecting neighboring-window context into each instance's representation
       before pooling (inspired by PS-DeVCEM's residual-temporal-feature attention).
       The output dimension of this stage is `self.post_lstm_dim`:
         - hidden_dim, if use_lstm=False
         - lstm_hidden_dim * num_directions, if use_lstm=True (whether plain,
           exact-match residual, or projected residual -- see lstm_residual /
           lstm_residual_proj below)
    3. Pooling: 'attention' (ABMIL gated attention + weighted sum),
                'mean' (uniform mean pooling -- no attention, isolates the LSTM's
                        contribution as a standalone baseline),
                'last' (use the LSTM's final hidden state as the bag representation --
                        also attention-free, and only valid when use_lstm=True)
       All pooling modes operate on / classify from `post_lstm_dim`.
    4. Classification: Multi-label head (sigmoid), input dim = post_lstm_dim

    For multi-label + pooling='attention': trains one attention module per label
    (label-specific attention), same as before. 'mean'/'last' pooling produce a
    single bag representation shared across all labels (no per-label attention,
    since there is no attention at all in those modes).
    """

    def __init__(self, n_features, n_labels, hidden_dim=256, dropout=0.2,
                 attention_dim=128, label_specific_attention=True,
                 use_lstm=False, lstm_hidden_dim=None, lstm_bidirectional=True,
                 lstm_num_layers=1, lstm_residual=False, lstm_residual_proj=False,
                 pooling='attention'):
        """
        Parameters
        ----------
        n_features : int
            Dimension of instance features
        n_labels : int
            Number of labels
        hidden_dim : int
            Hidden dimension for feature processing (output of feature_fc).
            When use_lstm=False, this is also the dimension the classifier and
            attention modules operate on. When use_lstm=True, the *effective*
            dimension used by attention/classifier is `post_lstm_dim` (see class
            docstring) -- which equals hidden_dim only when lstm_residual=True
            (exact-match) or when lstm_hidden_dim is left at its auto-derived
            default; it can differ from hidden_dim when lstm_residual_proj=True
            with a freely-chosen lstm_hidden_dim.
        dropout : float
            Dropout probability (also used as inter-layer LSTM dropout when
            lstm_num_layers > 1)
        attention_dim : int
            Dimension of attention mechanism (only used when pooling='attention')
        label_specific_attention : bool
            If True, use separate attention for each label (only used when
            pooling='attention')
        use_lstm : bool
            If True, instance embeddings are passed through a (bi)LSTM to inject
            temporal context before pooling. Default False = identical to the
            original ABMIL.
        lstm_hidden_dim : int or None
            Hidden size per LSTM direction. Defaults to hidden_dim // num_directions
            if not given. With lstm_residual_proj=True this can be set to any
            value -- it's a genuinely free hyperparameter in that mode.
        lstm_bidirectional : bool
            Whether the LSTM reads the bag in both directions. Recommended True
            since you have the full recording available (not a streaming/causal
            setting).
        lstm_num_layers : int
            Number of stacked LSTM layers. Keep at 1 given your dataset size
            unless you have evidence more depth helps.
        lstm_residual : bool
            If True, wrap the LSTM in an EXACT-MATCH skip connection:
            H_out = H + LSTM(H). Requires lstm_hidden_dim * num_directions ==
            hidden_dim (raises otherwise -- use lstm_residual_proj instead if you
            want lstm_hidden_dim to be independently sized). Mutually exclusive
            with lstm_residual_proj.
        lstm_residual_proj : bool
            If True, wrap the LSTM in a PROJECTED skip connection:
            H_out = Proj(H) + LSTM(H), where Proj is a single learned
            nn.Linear(hidden_dim, lstm_hidden_dim * num_directions) shortcut (no
            activation/dropout -- a standard identity-style residual-shortcut
            projection). This lets lstm_hidden_dim be tuned completely
            independently of hidden_dim. Requires use_lstm=True; incompatible
            with lstm_residual and with pooling='last'.
        pooling : {'attention', 'mean', 'last'}
            'attention' -> standard ABMIL gated-attention pooling (default,
                            matches original behavior when use_lstm=False)
            'mean'      -> simple mean pooling over (optionally LSTM-contextualized)
                            instances, no attention. Use with use_lstm=True to get
                            a "temporal context, no learned weighting" ablation.
            'last'      -> use the LSTM's final hidden state as the bag
                            representation. Requires use_lstm=True, and is
                            incompatible with lstm_residual / lstm_residual_proj
                            (there's no sequence of per-instance outputs left to
                            add a skip to -- only the final hidden state survives).
        """
        super().__init__()

        self.n_features = n_features
        self.n_labels = n_labels
        self.label_specific_attention = label_specific_attention
        self.use_lstm = use_lstm
        self.lstm_bidirectional = lstm_bidirectional
        self.lstm_residual = lstm_residual
        self.lstm_residual_proj = lstm_residual_proj
        self.pooling = pooling

        if pooling not in ('attention', 'mean', 'last'):
            raise ValueError(f"pooling must be one of 'attention', 'mean', 'last', got {pooling!r}")
        if pooling == 'last' and not use_lstm:
            raise ValueError("pooling='last' requires use_lstm=True")
        if pooling == 'last' and lstm_residual:
            raise ValueError("lstm_residual is not applicable to pooling='last' "
                              "(it applies to per-timestep outputs, not the final state)")
        if pooling == 'last' and lstm_residual_proj:
            raise ValueError("lstm_residual_proj is not applicable to pooling='last' "
                              "(no per-timestep sequence of outputs to project a shortcut onto -- "
                              "only the LSTM's final hidden state survives in that mode)")
        if lstm_residual and not use_lstm:
            raise ValueError("lstm_residual=True requires use_lstm=True")
        if lstm_residual_proj and not use_lstm:
            raise ValueError("lstm_residual_proj=True requires use_lstm=True")
        if lstm_residual and lstm_residual_proj:
            raise ValueError(
                "lstm_residual and lstm_residual_proj are mutually exclusive -- use "
                "lstm_residual for the exact-dimension-match residual (lstm_hidden_dim "
                "tied to hidden_dim), or lstm_residual_proj to allow an independently "
                "tuned lstm_hidden_dim via a learned shortcut projection."
            )

        # Feature processing
        self.feature_fc = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Optional temporal encoder
        if use_lstm:
            num_directions = 2 if lstm_bidirectional else 1
            if lstm_hidden_dim is None:
                assert hidden_dim % num_directions == 0, (
                    "hidden_dim must be divisible by 2 when lstm_bidirectional=True "
                    "and lstm_hidden_dim is not given explicitly"
                )
                lstm_hidden_dim = hidden_dim // num_directions
            lstm_output_dim = lstm_hidden_dim * num_directions

            if lstm_residual and lstm_output_dim != hidden_dim:
                raise ValueError(
                    f"lstm_residual=True (exact-match) requires lstm_hidden_dim * "
                    f"num_directions ({lstm_output_dim}) == hidden_dim ({hidden_dim}). "
                    f"Leave lstm_hidden_dim=None to get this automatically, or use "
                    f"lstm_residual_proj=True instead to allow an independently-sized "
                    f"lstm_hidden_dim via a learned shortcut projection."
                )

            self.lstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=lstm_hidden_dim,
                num_layers=lstm_num_layers,
                batch_first=True,
                bidirectional=lstm_bidirectional,
                dropout=dropout if lstm_num_layers > 1 else 0.0,
            )

            if lstm_residual_proj:
                # Standard residual-shortcut projection (ResNet-style): a single
                # learned linear map from the pre-LSTM representation to the
                # LSTM's output dimension, so H_out = Proj(H) + LSTM(H) even when
                # lstm_hidden_dim is tuned independently of hidden_dim. Kept as a
                # plain identity-style shortcut -- no activation, no dropout.
                self.lstm_shortcut = nn.Linear(hidden_dim, lstm_output_dim)

            post_lstm_dim = lstm_output_dim
        else:
            post_lstm_dim = hidden_dim

        self.post_lstm_dim = post_lstm_dim

        # Attention modules (only needed for pooling='attention')
        if pooling == 'attention':
            if label_specific_attention:
                self.attention_modules = nn.ModuleList([
                    self._build_attention(post_lstm_dim, attention_dim)
                    for _ in range(n_labels)
                ])
            else:
                self.attention = self._build_attention(post_lstm_dim, attention_dim)

        self.classifier = nn.Linear(post_lstm_dim, n_labels)

    @staticmethod
    def _build_attention(post_lstm_dim, attention_dim):
        """Build attention module."""
        return nn.Sequential(
            nn.Linear(post_lstm_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )

    def _apply_temporal_encoder(self, H):
        """
        Applies the optional LSTM over the ordered instance sequence.

        Parameters
        ----------
        H : (n_instances, hidden_dim)

        Returns
        -------
        H_out : (n_instances, post_lstm_dim)
            LSTM outputs at every timestep (equals H unchanged if use_lstm=False).
            If lstm_residual=True: H + LSTM(H) (post_lstm_dim == hidden_dim).
            If lstm_residual_proj=True: Proj(H) + LSTM(H) (post_lstm_dim == the
            LSTM's own output dim, independent of hidden_dim).
        h_n : final hidden state tensor or None
            Only populated when use_lstm=True; needed for pooling='last'. NOTE:
            when lstm_residual or lstm_residual_proj is True, h_n is still the
            raw LSTM final state (the residual/shortcut add only applies to the
            per-timestep outputs H_out) -- pooling='last' is disallowed together
            with either residual mode in __init__ to avoid ambiguity here.
        """
        if not self.use_lstm:
            return H, None
        H_seq = H.unsqueeze(0)  # (1, n_instances, hidden_dim) -- batch size 1 per bag
        lstm_out, (h_n, c_n) = self.lstm(H_seq)
        lstm_out = lstm_out.squeeze(0)  # (n_instances, post_lstm_dim)
        if self.lstm_residual_proj:
            H_out = self.lstm_shortcut(H) + lstm_out
        elif self.lstm_residual:
            H_out = H + lstm_out
        else:
            H_out = lstm_out
        return H_out, h_n

    def forward(self, x_bag, return_attention=False):
        """
        Forward pass.

        Parameters
        ----------
        x_bag : (n_instances, n_features)
            Instances from one bag, in temporal order
        return_attention : bool
            If True, return attention weights (only meaningful for pooling='attention')

        Returns
        -------
        logits : (n_labels,)
            Prediction logits
        attention_weights : list of (n_instances,), (n_instances,), or None
            Attention weights per label (pooling='attention' only); None for
            'mean'/'last' pooling since there is no attention to report.
        """
        # Feature processing
        H = self.feature_fc(x_bag)  # (n_instances, hidden_dim)

        # Optional temporal context
        H, h_n = self._apply_temporal_encoder(H)  # (n_instances, post_lstm_dim)

        # --- No-attention pooling modes (used for the LSTM-only ablation) ---
        if self.pooling == 'mean':
            M = torch.mean(H, dim=0, keepdim=True)  # (1, post_lstm_dim)
            logits = self.classifier(M).squeeze(0)  # (n_labels,)
            return logits, None

        if self.pooling == 'last':
            # h_n: (num_layers * num_directions, batch=1, lstm_hidden_dim)
            if self.lstm_bidirectional:
                last = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (1, post_lstm_dim)
            else:
                last = h_n[-1]  # (1, post_lstm_dim)
            logits = self.classifier(last).squeeze(0)  # (n_labels,)
            return logits, None

        # --- Attention pooling (ABMIL / ABMIL+LSTM) ---
        if self.label_specific_attention:
            logits_list = []
            attention_weights_list = []

            for label_idx in range(self.n_labels):
                A = self.attention_modules[label_idx](H)  # (n_instances, 1)
                A = torch.softmax(A, dim=0)  # (n_instances, 1)
                A_squeezed = A.squeeze(1)  # (n_instances,)

                M = torch.sum(A * H, dim=0, keepdim=True)  # (1, post_lstm_dim)

                logit = self.classifier(M)[0, label_idx]
                logits_list.append(logit)
                attention_weights_list.append(A_squeezed)

            logits = torch.stack(logits_list)
            attention_weights = attention_weights_list if return_attention else None
        else:
            A = self.attention(H)  # (n_instances, 1)
            A = torch.softmax(A, dim=0)  # (n_instances, 1)

            M = torch.sum(A * H, dim=0, keepdim=True)  # (1, post_lstm_dim)

            logits = self.classifier(M).squeeze(0)  # (n_labels,)
            attention_weights = A.squeeze(1) if return_attention else None

        return logits, attention_weights


# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

def train_abmil(
    X_bags_train,
    y_train,
    X_bags_val=None,
    y_val=None,
    n_labels=5,
    n_epochs=50,
    batch_size=16,
    learning_rate=1e-3,
    weight_decay=1e-5,
    hidden_dim=256,
    attention_dim=128,
    dropout=0.2,
    label_specific_attention=True,
    device='cuda',
    verbose=True,
    random_state=42,
    early_stopping=False,
    # --- new, all default to original ABMIL behavior ---
    use_lstm=False,
    lstm_hidden_dim=None,
    lstm_bidirectional=True,
    lstm_num_layers=1,
    lstm_residual=False,
    lstm_residual_proj=False,
    pooling='attention',
):
    """
    Train ABMIL model (optionally with an LSTM temporal-context layer, plain or
    residual (exact-match or projected), and/or attention-free pooling for the
    LSTM-only ablation).

    New parameters
    --------------
    use_lstm : bool
        Insert a BiLSTM before pooling. Default False = original ABMIL.
    lstm_hidden_dim, lstm_bidirectional, lstm_num_layers, lstm_residual,
    lstm_residual_proj : see ABMIL docstring.
    pooling : {'attention', 'mean', 'last'}
        See ABMIL docstring. Default 'attention' = original ABMIL.

    Returns
    -------
    model : ABMIL
    history : dict
    scaler : StandardScaler
    """
    device = torch.device(device)
    scaler = StandardScaler()

    g = torch.Generator()
    g.manual_seed(random_state)

    train_dataset = MILDataset(X_bags_train, y_train, scaler=scaler, fit_scaler=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                               collate_fn=collate_mil,
                               generator=g
                               )

    if X_bags_val is not None:
        val_dataset = MILDataset(X_bags_val, y_val, scaler=scaler, fit_scaler=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                 collate_fn=collate_mil,
                                 generator=g
                                 )
    else:
        val_loader = None

    n_features = train_dataset.X_bags_scaled[0].shape[1]

    model = ABMIL(
        n_features=n_features,
        n_labels=n_labels,
        hidden_dim=hidden_dim,
        attention_dim=attention_dim,
        dropout=dropout,
        label_specific_attention=label_specific_attention,
        use_lstm=use_lstm,
        lstm_hidden_dim=lstm_hidden_dim,
        lstm_bidirectional=lstm_bidirectional,
        lstm_num_layers=lstm_num_layers,
        lstm_residual=lstm_residual,
        lstm_residual_proj=lstm_residual_proj,
        pooling=pooling,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if early_stopping:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_ap': [],
        'val_auc': [],
    }

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = model.state_dict().copy()

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()

            batch_loss = 0.0
            for bag_idx in range(len(X_batch)):
                X_bag = X_batch[bag_idx].to(device)
                y_bag = y_batch[bag_idx].to(device)

                logits, _ = model(X_bag)
                loss = criterion(logits, y_bag)
                batch_loss += loss

            batch_loss = batch_loss / len(X_batch)
            batch_loss.backward()
            optimizer.step()
            train_loss += batch_loss.item()

        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            all_y_true = []
            all_y_pred = []

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    batch_loss = 0.0

                    for bag_idx in range(len(X_batch)):
                        X_bag = X_batch[bag_idx].to(device)
                        y_bag = y_batch[bag_idx].to(device)

                        logits, _ = model(X_bag)
                        loss = criterion(logits, y_bag)
                        batch_loss += loss

                        all_y_true.append(y_bag.cpu().numpy())
                        all_y_pred.append(logits.detach().cpu().numpy())

                    batch_loss = batch_loss / len(X_batch)
                    val_loss += batch_loss.item()

            val_loss /= len(val_loader)
            history['val_loss'].append(val_loss)

            all_y_true = np.array(all_y_true)
            all_y_pred = np.array(all_y_pred)

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


def predict_abmil(model, X_bags, scaler, device='cuda'):
    """Get predictions from trained ABMIL model (unchanged; works for all variants)."""
    device = torch.device(device)
    model.to(device)
    model.eval()

    predictions = []
    attention_dict = {}

    with torch.no_grad():
        for bag_idx, X_bag in enumerate(X_bags):
            X_bag_scaled = scaler.transform(X_bag)
            X_bag_tensor = torch.FloatTensor(X_bag_scaled).to(device)

            logits, attention_weights = model(X_bag_tensor, return_attention=True)
            y_pred = torch.sigmoid(logits).detach().cpu().numpy()
            predictions.append(y_pred)

            if attention_weights is not None:
                attention_dict[bag_idx] = [aw.detach().cpu().numpy() for aw in attention_weights]

    return np.array(predictions), attention_dict


class ABMILSklearnWrapper(BaseEstimator, ClassifierMixin):
    """
    SKLearn wrapper for ABMIL to enable use in GridSearchCV and RandomizedSearchCV.
    Exposes use_lstm / lstm_* / pooling so the temporal variants (including the
    projected-residual ones) can be tuned and compared through the exact same
    nested-CV machinery.

    LABEL-GROUPED ENSEMBLE (ensemble=True)
    ---------------------------------------
    When ensemble=True, `ensemble_labels` names the "simple" label indices that
    are pulled out of the main ABMIL/LSTM model. The main model is then trained
    with n_labels = total_labels - len(ensemble_labels) -- it never builds or
    trains attention branches for the simple labels at all. The simple labels
    are predicted separately by a LinearProbe (Dropout -> Linear) on mean-pooled
    raw instance embeddings. The two are trained independently, NOT jointly:
    every call to fit() runs its own nested-CV Optuna search to tune the
    LinearProbe (same search space / objective as the standalone linear-probe
    module), then refits it on the full data passed to fit(). predict_proba()
    reassembles both models' predictions into the original label column order.
    """
    def __init__(self, hidden_dim=256, attention_dim=128, dropout=0.2,
                 learning_rate=1e-3, weight_decay=1e-5, batch_size=4,
                 n_epochs=20, n_labels=5, device='cuda', random_state=42,
                 ensemble=False, ensemble_labels=None,
                 ensemble_n_optuna_trials=15, ensemble_n_split_in=3,
                 ensemble_n_epochs_max=40,
                 early_stopping=False,
                 use_lstm=False, lstm_hidden_dim=None, lstm_bidirectional=True,
                 lstm_num_layers=1, lstm_residual=False, lstm_residual_proj=False,
                 pooling='attention'):
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.n_labels = n_labels
        self.device = device
        self.random_state = random_state
        self.ensemble = ensemble
        self.ensemble_labels = ensemble_labels
        self.ensemble_n_optuna_trials = ensemble_n_optuna_trials
        self.ensemble_n_split_in = ensemble_n_split_in
        self.ensemble_n_epochs_max = ensemble_n_epochs_max
        self.early_stopping = early_stopping
        self.use_lstm = use_lstm
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_bidirectional = lstm_bidirectional
        self.lstm_num_layers = lstm_num_layers
        self.lstm_residual = lstm_residual
        self.lstm_residual_proj = lstm_residual_proj
        self.pooling = pooling

    def _get_ensemble_labels(self):
        simple = set(self.ensemble_labels) if self.ensemble_labels is not None else {0}
        return sorted(simple)

    def _meanpool(self, X):
        return np.array([bag.mean(axis=0) for bag in X])

    def fit(self, X, y):
        y = np.asarray(y)

        if self.ensemble:
            simple_idx = self._get_ensemble_labels()
            complex_idx = [i for i in range(y.shape[1]) if i not in simple_idx]
            self._complex_idx = complex_idx
            self._simple_idx = simple_idx

            # --- Complex labels: main ABMIL/LSTM model, restricted to complex_idx ---
            y_complex = y[:, complex_idx]
            self.model_, self.history_, self.scaler_ = train_abmil(
                X_bags_train=X, y_train=y_complex,
                n_labels=len(complex_idx), n_epochs=self.n_epochs, batch_size=self.batch_size,
                learning_rate=self.learning_rate, weight_decay=self.weight_decay,
                hidden_dim=self.hidden_dim, attention_dim=self.attention_dim, dropout=self.dropout,
                label_specific_attention=True,
                device=self.device, verbose=False, random_state=self.random_state,
                early_stopping=self.early_stopping,
                use_lstm=self.use_lstm, lstm_hidden_dim=self.lstm_hidden_dim,
                lstm_bidirectional=self.lstm_bidirectional, lstm_num_layers=self.lstm_num_layers,
                lstm_residual=self.lstm_residual, lstm_residual_proj=self.lstm_residual_proj,
                pooling=self.pooling,
            )

            # --- Simple labels: independently-tuned LinearProbe on mean-pooled
            #     raw embeddings. Not joint with the main model -- its own
            #     nested-CV Optuna search every fit() call, same pattern as
            #     linear_probe_tuned_optuna's per-fold tuning. ---
            y_simple = y[:, simple_idx]
            X_pool = self._meanpool(X)

            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=self.random_state, multivariate=True),
                study_name=f"ensemble_linear_probe_seed{self.random_state}",
            )
            study.optimize(
                _make_ensemble_linear_probe_objective(
                    X_pool, y_simple, self.ensemble_n_split_in,
                    self.ensemble_n_epochs_max, self.random_state,
                ),
                n_trials=self.ensemble_n_optuna_trials,
            )
            best_params = study.best_params

            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
            random.seed(self.random_state)

            self.simple_model_ = LinearProbeSklearnWrapper(
                n_labels=len(simple_idx), n_epochs=self.ensemble_n_epochs_max,
                batch_size=best_params["batch_size"], dropout=0.0,
                learning_rate=best_params["learning_rate"], weight_decay=best_params["weight_decay"],
                device=self.device, random_state=self.random_state, early_stopping=False,
            )
            self.simple_model_.fit(X_pool, y_simple)

        else:
            self.model_, self.history_, self.scaler_ = train_abmil(
                X_bags_train=X, y_train=y,
                n_labels=self.n_labels, n_epochs=self.n_epochs, batch_size=self.batch_size,
                learning_rate=self.learning_rate, weight_decay=self.weight_decay,
                hidden_dim=self.hidden_dim, attention_dim=self.attention_dim, dropout=self.dropout,
                label_specific_attention=True,
                device=self.device, verbose=False, random_state=self.random_state,
                early_stopping=self.early_stopping,
                use_lstm=self.use_lstm, lstm_hidden_dim=self.lstm_hidden_dim,
                lstm_bidirectional=self.lstm_bidirectional, lstm_num_layers=self.lstm_num_layers,
                lstm_residual=self.lstm_residual, lstm_residual_proj=self.lstm_residual_proj,
                pooling=self.pooling,
            )

        return self

    def predict_proba(self, X):
        if not self.ensemble:
            preds, _ = predict_abmil(self.model_, X, self.scaler_, device=self.device)
            return preds

        complex_preds, _ = predict_abmil(self.model_, X, self.scaler_, device=self.device)
        X_pool = self._meanpool(X)
        simple_preds = self.simple_model_.predict_proba(X_pool)

        preds = np.zeros((len(X), self.n_labels))
        preds[:, self._complex_idx] = complex_preds
        preds[:, self._simple_idx] = simple_preds
        return preds

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)

    @property
    def classes_(self):
        return [np.array([0, 1]) for _ in range(self.n_labels)]


# ============================================================================
# ARCHITECTURE VARIANT CONFIGS
# ============================================================================

# Names you can pass around:
#   'ABMIL'                    -- original ABMIL, no LSTM
#   'ABMIL_LSTM'                -- recurrent-context ABMIL, LSTM output replaces H entirely
#   'ABMIL_LSTM_residual'       -- H_out = H + LSTM(H), exact-match dims required
#   'ABMIL_LSTM_residual_proj'  -- H_out = Proj(H) + LSTM(H), lstm_hidden_dim freely tunable
#   'LSTM_only'                 -- LSTM + mean pooling, no attention (ablation arm)
#   'LSTM_only_residual'        -- same, with the exact-match residual connection
#   'LSTM_residual_proj'        -- same, with the projected-shortcut residual connection
#   'LSTM_last'                 -- LSTM final hidden state + no attention (alternate
#                                   attention-free baseline; incompatible with any residual mode)
VARIANT_WRAPPER_KWARGS = {
    'ABMIL':                    dict(use_lstm=False, pooling='attention'),
    'ABMIL_LSTM':                dict(use_lstm=True,  pooling='attention', lstm_residual=False,
                                       lstm_residual_proj=False,
                                       lstm_bidirectional=True, lstm_num_layers=1),
    'ABMIL_LSTM_residual':       dict(use_lstm=True,  pooling='attention', lstm_residual=True,
                                       lstm_residual_proj=False,
                                       lstm_bidirectional=True, lstm_num_layers=1),
    'ABMIL_LSTM_residual_proj':  dict(use_lstm=True,  pooling='attention', lstm_residual=False,
                                       lstm_residual_proj=True,
                                       lstm_bidirectional=True, lstm_num_layers=1),
    'LSTM_only':                 dict(use_lstm=True,  pooling='mean',      lstm_residual=False,
                                       lstm_residual_proj=False,
                                       lstm_bidirectional=True, lstm_num_layers=1),
    'LSTM_only_residual':        dict(use_lstm=True,  pooling='mean',      lstm_residual=True,
                                       lstm_residual_proj=False,
                                       lstm_bidirectional=True, lstm_num_layers=1),
    'LSTM_residual_proj':        dict(use_lstm=True,  pooling='mean',      lstm_residual=False,
                                       lstm_residual_proj=True,
                                       lstm_bidirectional=True, lstm_num_layers=1),
    'LSTM_last':                 dict(use_lstm=True,  pooling='last',      lstm_residual=False,
                                       lstm_residual_proj=False,
                                       lstm_bidirectional=True, lstm_num_layers=1),
}

# Base hyperparameter search space, shared by all variants.
BASE_PARAM_GRID = {
    'hidden_dim': [64, 128, 256],
    'attention_dim': [32, 64, 128],  # ignored by LSTM_only / LSTM_last, harmless to search
    'dropout': [0.1, 0.3, 0.5],
    'learning_rate': loguniform(1e-4, 5e-3),
    'weight_decay': loguniform(1e-6, 1e-2),
}

# Only added on top of BASE_PARAM_GRID for the projected-residual variants --
# NOT added to every variant, because for 'ABMIL_LSTM_residual' /
# 'LSTM_only_residual' (exact-match residual) an lstm_hidden_dim that doesn't
# equal hidden_dim // num_directions would raise inside ABMIL.__init__ and
# crash the search.
LSTM_PROJ_EXTRA_PARAM_GRID = {
    'lstm_hidden_dim': [32, 64, 128, 256],
}

_PROJ_VARIANTS = ('ABMIL_LSTM_residual_proj', 'LSTM_residual_proj')


def _build_variant_wrapper_and_grid(variant, n_labels, device, trial_seed,
                                     ensemble=True, ensemble_labels=(0,), n_epochs=20, batch_size=4,
                                     ensemble_n_optuna_trials=15, ensemble_n_split_in=3,
                                     ensemble_n_epochs_max=40):
    """Builds the ABMILSklearnWrapper + param grid for a given architecture variant name."""
    if variant not in VARIANT_WRAPPER_KWARGS:
        raise ValueError(f"Unknown variant {variant!r}. Choose from {list(VARIANT_WRAPPER_KWARGS)}")

    wrapper = ABMILSklearnWrapper(
        n_labels=n_labels, n_epochs=n_epochs, batch_size=batch_size,
        device=device, ensemble=ensemble, ensemble_labels=list(ensemble_labels),
        ensemble_n_optuna_trials=ensemble_n_optuna_trials,
        ensemble_n_split_in=ensemble_n_split_in,
        ensemble_n_epochs_max=ensemble_n_epochs_max,
        random_state=trial_seed,
        **VARIANT_WRAPPER_KWARGS[variant],
    )
    params = dict(BASE_PARAM_GRID)
    if variant in _PROJ_VARIANTS:
        params.update(LSTM_PROJ_EXTRA_PARAM_GRID)
    return wrapper, params


# ============================================================================
# TUNING WITH OPTUNA
# ============================================================================

def _suggest_variant_params_old(trial, variant):
    common = dict(
        hidden_dim=trial.suggest_categorical("hidden_dim", [32, 64, 128, 256]),
        dropout=trial.suggest_float("dropout", 0.1, 0.5),
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        batch_size=trial.suggest_categorical("batch_size", [4, 8, 16]),
    )
    if variant == "ABMIL":
        extra = dict(attention_dim=trial.suggest_categorical("attention_dim", [16, 32, 64, 128]),
                     use_lstm=False, pooling="attention")
    elif variant == "ABMIL_LSTM":
        extra = dict(attention_dim=trial.suggest_categorical("attention_dim", [16, 32, 64, 128]),
                     use_lstm=True, pooling="attention", lstm_residual=True,
                     lstm_num_layers=1,
                     lstm_bidirectional=True)
    elif variant == "ABMIL_LSTM_residual_proj":
        extra = dict(attention_dim=trial.suggest_categorical("attention_dim", [16, 32, 64, 128]),
                     use_lstm=True, pooling="attention", lstm_residual=False,
                     lstm_residual_proj=True,
                     lstm_num_layers=1,
                     lstm_bidirectional=True,
                     lstm_hidden_dim=trial.suggest_categorical("lstm_hidden_dim", [16, 32, 64, 128]))
    elif variant == "LSTM_only":
        extra = dict(use_lstm=True, pooling="mean", lstm_residual=True,
                     lstm_num_layers=1,
                     lstm_bidirectional=True)
    elif variant == "LSTM_residual_proj":
        extra = dict(use_lstm=True, pooling="mean", lstm_residual=False,
                     lstm_residual_proj=True,
                     lstm_num_layers=1,
                     lstm_bidirectional=True,
                     lstm_hidden_dim=trial.suggest_categorical("lstm_hidden_dim", [16, 32, 64, 128]))
    else:
        raise ValueError(variant)

    return common, extra


def _suggest_variant_params(trial, variant):
    common = dict(
        hidden_dim=trial.suggest_categorical("hidden_dim", [128]),
        dropout=trial.suggest_categorical("dropout", [0.3]),
        learning_rate=trial.suggest_categorical("learning_rate",[1e-5,5e-5,1e-4,5e-4,1e-3]),
        weight_decay=trial.suggest_categorical("weight_decay",[1e-5]),
        batch_size=trial.suggest_categorical("batch_size", [4]),
    )
    if variant == "ABMIL":
        extra = dict(attention_dim=trial.suggest_categorical("attention_dim", [ 64]),
                     use_lstm=False, pooling="attention")
    elif variant == "ABMIL_LSTM":
        extra = dict(attention_dim=trial.suggest_categorical("attention_dim", [ 64]),
                     use_lstm=True, pooling="attention", lstm_residual=True,
                     lstm_num_layers=1,
                     lstm_bidirectional=True)
    elif variant == "ABMIL_LSTM_residual_proj":
        extra = dict(attention_dim=trial.suggest_categorical("attention_dim", [ 64]),
                     use_lstm=True, pooling="attention", lstm_residual=False,
                     lstm_residual_proj=True,
                     lstm_num_layers=1,
                     lstm_bidirectional=True,
                     lstm_hidden_dim=trial.suggest_categorical("lstm_hidden_dim", [128]))
    elif variant == "LSTM_only":
        extra = dict(use_lstm=True, pooling="mean", lstm_residual=True,
                     lstm_num_layers=1,
                     lstm_bidirectional=True)
    elif variant == "LSTM_residual_proj":
        extra = dict(use_lstm=True, pooling="mean", lstm_residual=False,
                     lstm_residual_proj=True,
                     lstm_num_layers=1,
                     lstm_bidirectional=True,
                     lstm_hidden_dim=trial.suggest_categorical("lstm_hidden_dim", [128]))
    else:
        raise ValueError(variant)

    return common, extra


def _make_optuna_objective(variant, X_bags_train, y_train, n_split_in, n_epochs_max,
                            seed, label_indices=None):
    """
    label_indices : list of int or None
        If given, restricts y_train to these label columns before splitting/
        training/scoring -- used when ensemble=True so the main model's
        hyperparameter search is optimized for the labels it will actually be
        deployed on (the complex labels), not the full label set.
    """
    y_train = np.asarray(y_train)
    if label_indices is not None:
        y_train = y_train[:, label_indices]
    inner_cv = MultilabelStratifiedKFold(n_splits=n_split_in, shuffle=True, random_state=seed)

    def objective(trial):
        common, extra = _suggest_variant_params(trial, variant)

        fold_val_ap_curves = []
        #DEBUG
        print("Start inner loop)")
        constant = 0
        for tr_idx, val_idx in inner_cv.split(X_bags_train, y_train):
            constant = constant + 1
            print(f"Inner loop number {constant}")
            X_tr = [X_bags_train[i] for i in tr_idx]
            X_val = [X_bags_train[i] for i in val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]

            _, history, _ = train_abmil(
                X_bags_train=X_tr, y_train=y_tr,
                X_bags_val=X_val, y_val=y_val,
                n_labels=y_train.shape[1], n_epochs=n_epochs_max,
                batch_size=common["batch_size"], verbose=False, random_state=seed,
                early_stopping= False,
                hidden_dim=common["hidden_dim"], dropout=common["dropout"],
                learning_rate=common["learning_rate"], weight_decay=common["weight_decay"],
                attention_dim=extra.get("attention_dim", 128),
                use_lstm=extra["use_lstm"], pooling=extra["pooling"],
                lstm_residual=extra.get("lstm_residual", False),
                lstm_residual_proj=extra.get("lstm_residual_proj", False),
                lstm_hidden_dim=extra.get("lstm_hidden_dim", None),
                lstm_num_layers=extra.get("lstm_num_layers", 1),
                lstm_bidirectional=extra.get("lstm_bidirectional", False),
            )
            fold_val_ap_curves.append(history["val_ap"])
        print("End inner loop)")
        mean_ap_curve = np.mean(fold_val_ap_curves, axis=0)
        best_epoch = int(np.argmax(mean_ap_curve)) + 1
        print(best_epoch)
        print(mean_ap_curve[best_epoch-1])

        trial.set_user_attr("mean_val_ap_curve", mean_ap_curve.tolist())

        return float(mean_ap_curve[n_epochs_max - 1])

    return objective


def abmil_classifier_tuned_optuna(X_bags, y, n_split_out=5, n_split_in=5, num_trials=5,
                                   random_state=42, n_optuna_trials=20, n_epochs_max=40,
                                   variants=('ABMIL',),
                                   ensemble=False, ensemble_labels=None,
                                   ensemble_n_optuna_trials=15, ensemble_n_split_in=3,
                                   ensemble_n_epochs_max=40):
    """
    Same role as abmil_classifier_tuned, but the inner-loop search is an Optuna
    study (TPE) per (variant, outer fold) instead of RandomizedSearchCV, and each
    study also selects a per-fold epoch count from validation-AP curves (no
    early stopping during training itself -- every inner fit runs the full
    n_epochs_max).

    ensemble / ensemble_labels : see ABMILSklearnWrapper. When ensemble=True,
    the Optuna search for each variant's main-model hyperparameters is run
    against the complex labels only (total labels minus ensemble_labels) --
    matching what the final wrapper will actually train on -- while the simple
    labels are handled by their own independently-tuned LinearProbe inside the
    final wrapper's fit().
    """
    y = np.array(y)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = []

    complex_idx = None
    if ensemble:
        simple_idx = sorted(set(ensemble_labels) if ensemble_labels is not None else {0})
        complex_idx = [i for i in range(y.shape[1]) if i not in simple_idx]

    for i in range(num_trials):
        trial_seed = random_state + i
        print(f"Starting Trial {i+1}/{num_trials} with random_state={trial_seed}...")

        np.random.seed(trial_seed)
        random.seed(trial_seed)
        torch.manual_seed(trial_seed)
        torch.cuda.manual_seed_all(trial_seed)

        outer_cv = MultilabelStratifiedKFold(n_splits=n_split_out, shuffle=True, random_state=trial_seed)

        for variant in variants:
            print(f"  Tuning and evaluating model: {variant}")
            all_y_true, all_y_pred_proba, all_test_indices, best_models = [], [], [], []
            outer_scores = []
            fold_train_histories, fold_val_histories = [], []
            all_best_params = []

            for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_bags, y)):
                print(f"    Evaluating fold {fold+1}/{n_split_out}")
                X_bags_train = [X_bags[i] for i in train_idx]
                X_bags_test = [X_bags[i] for i in test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # --- Optuna study: tunes hyperparams + picks epoch count on inner CV ---
                study = optuna.create_study(
                    direction="maximize",
                    sampler=optuna.samplers.TPESampler(seed=trial_seed * 100 + fold, multivariate=True),
                    study_name=f"{variant}_trial{i}_fold{fold}",
                )
                study.optimize(
                    _make_optuna_objective(variant, X_bags_train, y_train,
                                            n_split_in, n_epochs_max, trial_seed,
                                            label_indices=complex_idx),
                    n_trials=n_optuna_trials,
                )

                best_params = study.best_params

                # --- Refit on the FULL outer-train fold at best hyperparams/epoch,
                #     via the wrapper so the ensemble LinearProbe blend (if any)
                #     is included ---
                _, extra = _suggest_variant_params(optuna.trial.FixedTrial(best_params), variant)

                final_model = ABMILSklearnWrapper(
                    n_labels=y.shape[1], n_epochs=n_epochs_max, batch_size=best_params["batch_size"],
                    device=device,
                    random_state=trial_seed,
                    ensemble=ensemble, ensemble_labels=ensemble_labels,
                    ensemble_n_optuna_trials=ensemble_n_optuna_trials,
                    ensemble_n_split_in=ensemble_n_split_in,
                    ensemble_n_epochs_max=ensemble_n_epochs_max,
                    hidden_dim=best_params["hidden_dim"], dropout=best_params["dropout"],
                    learning_rate=best_params["learning_rate"], weight_decay=best_params["weight_decay"],
                    attention_dim=best_params.get("attention_dim", 128),
                    use_lstm=extra["use_lstm"], pooling=extra["pooling"],
                    lstm_residual=extra.get("lstm_residual", False),
                    lstm_residual_proj=extra.get("lstm_residual_proj", False),
                    lstm_hidden_dim=extra.get("lstm_hidden_dim", None),
                    lstm_num_layers=extra.get("lstm_num_layers", 1),
                    lstm_bidirectional=extra.get("lstm_bidirectional", True),
                )
                final_model.fit(X_bags_train, y_train)

                fold_train_histories.append(final_model.history_['train_loss'])
                fold_val_histories.append(final_model.history_['val_loss'])  # NaNs (no val set passed at this stage)

                y_pred_proba = final_model.predict_proba(X_bags_test)
                if isinstance(y_pred_proba, list):
                    y_pred_proba = np.column_stack([prob[:, 1] for prob in y_pred_proba])
                elif isinstance(y_pred_proba, np.ndarray) and y_pred_proba.ndim == 3:
                    y_pred_proba = y_pred_proba[:, :, 1].T

                fold_score = average_precision_score(y_test, y_pred_proba, average='macro')
                outer_scores.append(fold_score)

                all_y_true.append(y_test)
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
                'model': variant,
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
                'best_params' : all_best_params,
            })

    return all_results


def abmil_classifier_tuned(X_bags, y, n_split_out=5, n_split_in=5, num_trials=5, random_state=42,
                            n_iter_search=4, variants=('ABMIL',),
                            ensemble_n_optuna_trials=15, ensemble_n_split_in=3,
                            ensemble_n_epochs_max=40):
    """
    Trains and evaluates one or more architecture variants with nested cross-validation
    and hyperparameter tuning via RandomizedSearchCV.

    NOTE: ensemble=True is hardcoded per-variant below (ensemble_labels=(0,)),
    same as the original behavior. Every RandomizedSearchCV candidate fit now
    also runs its own nested Optuna search to tune the ensemble's LinearProbe
    (see ABMILSklearnWrapper.fit), so this path is noticeably more expensive
    per candidate than before -- reduce n_iter_search / ensemble_n_optuna_trials
    if this gets too slow.

    X_bags: list of arrays, each array is (n_windows, n_features) for one recording
    y: (n_bags, n_labels) array of multi-label targets
    n_iter_search: int, number of parameter settings sampled in RandomizedSearchCV per fold
    n_split_out, n_split_in: outer/inner CV splits
    num_trials: int, number of repeated nested CV runs with different seeds
    random_state: int, base random seed
    variants: tuple of str
        Which architecture(s) to run in this sweep, drawn from
        VARIANT_WRAPPER_KWARGS: 'ABMIL' (default -- identical to original behavior),
        'ABMIL_LSTM', 'ABMIL_LSTM_residual', 'ABMIL_LSTM_residual_proj', 'LSTM_only',
        'LSTM_only_residual', 'LSTM_residual_proj', 'LSTM_last'.

    Returns a list of results dictionaries (one per trial x variant) containing OOF
    predictions, best models, and performance metrics.
    """
    y = np.array(y)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scorer = make_scorer(average_precision_score, average='macro', response_method='predict_proba')
    all_results = []

    for i in range(num_trials):
        trial_seed = random_state + i
        print(f"Starting Trial {i+1}/{num_trials} with random_state={trial_seed}...")

        np.random.seed(trial_seed)
        random.seed(trial_seed)
        torch.manual_seed(trial_seed)
        torch.cuda.manual_seed_all(trial_seed)

        model_params = {}
        for variant in variants:
            wrapper, params = _build_variant_wrapper_and_grid(
                variant, n_labels=y.shape[1], device=device, trial_seed=trial_seed,
                ensemble=True, ensemble_labels=(0,),
                ensemble_n_optuna_trials=ensemble_n_optuna_trials,
                ensemble_n_split_in=ensemble_n_split_in,
                ensemble_n_epochs_max=ensemble_n_epochs_max,
            )
            model_params[variant] = {'model': wrapper, 'params': params}

        inner_cv = MultilabelStratifiedKFold(n_splits=n_split_in, shuffle=True, random_state=trial_seed)
        outer_cv = MultilabelStratifiedKFold(n_splits=n_split_out, shuffle=True, random_state=trial_seed)

        for model_name, mp in model_params.items():
            print(f"  Tuning and evaluating model: {model_name}")
            all_y_true = []
            all_y_pred_proba = []
            all_test_indices = []
            best_models = []

            outer_scores = []
            fold_train_histories = []
            fold_val_histories = []

            for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_bags, y)):
                print(f"    Evaluating fold {fold+1}/{n_split_out}")
                X_bags_train = [X_bags[i] for i in train_idx]
                X_bags_test = [X_bags[i] for i in test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                clf = RandomizedSearchCV(
                    estimator=mp['model'],
                    param_distributions=mp['params'],
                    n_iter=n_iter_search,
                    cv=inner_cv,
                    scoring=scorer,
                    refit=True,
                    n_jobs=1,
                    random_state=trial_seed*100+fold,
                    verbose=3
                )

                clf.fit(X_bags_train, y_train)
                best_model_instance = clf.best_estimator_
                fold_train_histories.append(best_model_instance.history_['train_loss'])
                fold_val_histories.append(best_model_instance.history_['val_loss'])

                y_pred_proba = clf.predict_proba(X_bags_test)

                if isinstance(y_pred_proba, list):
                    y_pred_proba = np.column_stack([prob[:, 1] for prob in y_pred_proba])
                elif isinstance(y_pred_proba, np.ndarray) and y_pred_proba.ndim == 3:
                    y_pred_proba = y_pred_proba[:, :, 1].T

                fold_score = average_precision_score(y_test, y_pred_proba, average='macro')
                outer_scores.append(fold_score)

                all_y_true.append(y_test)
                all_y_pred_proba.append(y_pred_proba)
                all_test_indices.append(test_idx)
                best_models.append(clf.best_estimator_)

            y_true_cv = all_y_true
            y_pred_proba_cv = all_y_pred_proba
            all_y_true = np.concatenate(all_y_true, axis=0)
            all_y_pred_proba = np.concatenate(all_y_pred_proba, axis=0)
            all_test_indices = np.concatenate(all_test_indices, axis=0)
            all_results.append({
                'trial': i,
                'model': model_name,
                'best_models': best_models,

                'mean_AP': np.mean(outer_scores),
                'std_AP': np.std(outer_scores, ddof=1),

                'y_true_cv': y_true_cv,
                'y_pred_proba_cv': y_pred_proba_cv,
                'oof_y_true': all_y_true,
                'oof_y_pred_proba': all_y_pred_proba,
                'oof_indices': all_test_indices,
                'train_histories': fold_train_histories,
                'val_histories': fold_val_histories
            })

    return all_results


def abmil_classifier_quick(X_bags, y, n_split_out=5, num_trials=3, random_state=42,
                            variants=('ABMIL',), hidden_dim=256, attention_dim=128,
                            dropout=0.1, learning_rate=1e-4, weight_decay=1e-6,
                            lstm_hidden_dim=None,
                            batch_size=4, n_epochs=20,
                            ensemble=True, ensemble_labels=(0,),
                            ensemble_n_optuna_trials=15, ensemble_n_split_in=3,
                            ensemble_n_epochs_max=40):
    """
    Fast, no-hyperparameter-search counterpart to abmil_classifier_tuned.

    Runs plain outer cross-validation only (no inner RandomizedSearchCV loop) --
    every fold fits ONE model per variant at a single fixed hyperparameter
    configuration, rather than searching n_iter_search configs per fold via an
    inner CV. This drops the search's `n_iter_search * n_split_in` extra fits per
    outer fold down to just 1, so it's a cheap way to get a first read on whether
    a variant (e.g. 'ABMIL' vs 'ABMIL_LSTM_residual_proj') looks promising before
    spending the time on a full nested-CV hyperparameter search.

    NOTE: when ensemble=True, the LinearProbe half of the model still runs its
    own internal nested-CV Optuna search inside fit() regardless of how "quick"
    the main model's config is -- ensemble_n_optuna_trials/ensemble_n_split_in/
    ensemble_n_epochs_max control the cost of that inner search.

    Output has the SAME structure/keys as abmil_classifier_tuned's results list,
    so downstream code (analyze_best_hyperparameters, plotting, etc.) doesn't
    need to branch on which function produced it -- 'best_models' here just holds
    the single fixed-config fitted model per fold, not a search winner (there's
    no search).

    Parameters
    ----------
    X_bags, y : as in abmil_classifier_tuned
    n_split_out : int
        Outer CV splits. (No inner CV / n_split_in here -- nothing to tune.)
    num_trials : int
        Repeated CV runs with different seeds, same as abmil_classifier_tuned.
    random_state : int
    variants : tuple of str
        Which architecture(s) to run, from VARIANT_WRAPPER_KWARGS -- e.g.
        ('ABMIL', 'ABMIL_LSTM_residual_proj', 'LSTM_only_residual').
    hidden_dim, attention_dim, dropout, learning_rate, weight_decay,
    lstm_hidden_dim, batch_size, n_epochs : fixed hyperparameters used for every
        fold/variant/trial. lstm_hidden_dim only matters for variants that use
        an LSTM; leave None to get the auto-derived default (hidden_dim //
        num_directions) for non-projected variants.
    ensemble, ensemble_labels, ensemble_n_optuna_trials, ensemble_n_split_in,
    ensemble_n_epochs_max : see ABMILSklearnWrapper -- controls the label-grouped
        simple/complex split and the independent LinearProbe tuning for the
        simple labels.

    Returns
    -------
    list of dict, one per (trial, variant), same schema as abmil_classifier_tuned.
    """
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

        for variant in variants:
            if variant not in VARIANT_WRAPPER_KWARGS:
                raise ValueError(f"Unknown variant {variant!r}. Choose from {list(VARIANT_WRAPPER_KWARGS)}")

            print(f"  Evaluating model (no search): {variant}")
            all_y_true = []
            all_y_pred_proba = []
            all_test_indices = []
            best_models = []

            outer_scores = []
            fold_train_histories = []
            fold_val_histories = []

            for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_bags, y)):
                print(f"    Fold {fold+1}/{n_split_out}")
                X_bags_train = [X_bags[i] for i in train_idx]
                X_bags_test = [X_bags[i] for i in test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                model = ABMILSklearnWrapper(
                    n_labels=y.shape[1], n_epochs=n_epochs, batch_size=batch_size,
                    device=device, ensemble=ensemble, ensemble_labels=list(ensemble_labels),
                    ensemble_n_optuna_trials=ensemble_n_optuna_trials,
                    ensemble_n_split_in=ensemble_n_split_in,
                    ensemble_n_epochs_max=ensemble_n_epochs_max,
                    random_state=trial_seed,
                    hidden_dim=hidden_dim, attention_dim=attention_dim, dropout=dropout,
                    learning_rate=learning_rate, weight_decay=weight_decay,
                    lstm_hidden_dim=lstm_hidden_dim,
                    **VARIANT_WRAPPER_KWARGS[variant],
                )

                # Single direct fit on the outer-train fold -- no inner CV, no search.
                model.fit(X_bags_train, y_train)
                fold_train_histories.append(model.history_['train_loss'])
                fold_val_histories.append(model.history_['val_loss'])

                y_pred_proba = model.predict_proba(X_bags_test)

                if isinstance(y_pred_proba, list):
                    y_pred_proba = np.column_stack([prob[:, 1] for prob in y_pred_proba])
                elif isinstance(y_pred_proba, np.ndarray) and y_pred_proba.ndim == 3:
                    y_pred_proba = y_pred_proba[:, :, 1].T

                fold_score = average_precision_score(y_test, y_pred_proba, average='macro')
                outer_scores.append(fold_score)

                all_y_true.append(y_test)
                all_y_pred_proba.append(y_pred_proba)
                all_test_indices.append(test_idx)
                best_models.append(model)

            y_true_cv = all_y_true
            y_pred_proba_cv = all_y_pred_proba
            all_y_true = np.concatenate(all_y_true, axis=0)
            all_y_pred_proba = np.concatenate(all_y_pred_proba, axis=0)
            all_test_indices = np.concatenate(all_test_indices, axis=0)
            all_results.append({
                'trial': i,
                'model': variant,
                'best_models': best_models,

                'mean_AP': np.mean(outer_scores),
                'std_AP': np.std(outer_scores, ddof=1),

                'y_true_cv': y_true_cv,
                'y_pred_proba_cv': y_pred_proba_cv,
                'oof_y_true': all_y_true,
                'oof_y_pred_proba': all_y_pred_proba,
                'oof_indices': all_test_indices,
                'train_histories': fold_train_histories,
                'val_histories': fold_val_histories
            })

    return all_results


def analyze_best_hyperparameters(abmil_results):
    """Analyzes the best hyperparameters from a list of results (unchanged)."""
    all_params = []

    for run_idx, run in enumerate(abmil_results):
        if 'best_models' in run:
            for model in run['best_models']:
                if hasattr(model, 'get_params'):
                    params = model.get_params(deep=False)
                elif hasattr(model, 'best_params_'):
                    params = model.best_params_
                else:
                    params = {k: v for k, v in model.__dict__.items()
                              if isinstance(v, (int, float, str, bool, list, tuple, type(None)))
                              and not k.startswith('_')}

                all_params.append(params)

    if not all_params:
        print("No hyperparameters found. Please verify the structure of your abmil_results.")
        return None

    df_params = pd.DataFrame(all_params)

    cols_to_drop = ['device', 'scaler', 'model_']
    cols_to_drop = [c for c in cols_to_drop if c in df_params.columns]
    df_params = df_params.drop(columns=cols_to_drop)

    for col in df_params.columns:
        df_params[col] = df_params[col].apply(lambda x: tuple(x) if isinstance(x, list) else x)

    varying_cols = [col for col in df_params.columns if df_params[col].nunique() > 1]

    if not varying_cols:
        print("All search parameters were identical across every run!\n")
        print(df_params.iloc[0].to_frame(name='Value'))
        return df_params

    df_varying = df_params[varying_cols]

    print("=" * 60)
    print(" MOST FREQUENT WINNING COMBINATIONS ")
    print("=" * 60)
    top_combinations = df_varying.value_counts().reset_index(name='Occurrence Count')
    print(top_combinations.to_string(index=False))
    print("\n" + "=" * 60)

    print(" MOST FREQUENT INDIVIDUAL PARAMETERS (MODE) ")
    print("=" * 60)
    for col in df_varying.columns:
        mode_val = df_varying[col].mode()[0]
        mode_count = (df_varying[col] == mode_val).sum()
        print(f" -> {col}: Most frequent choice was {mode_val} (found {mode_count}/{len(df_params)} times)")

    return top_combinations


def abmil_classifier_deployement(X_bags, y, n_split=5, random_state=42, variant='ABMIL'):
    """
    Trains a chosen variant in non-nested CV over a restricted hyperparameter grid,
    for finding the best hyperparameters to train a final model on 100% of the data.

    X_bags, y: as before.
    variant : str
        Which architecture to deploy-tune: 'ABMIL' (default, unchanged behavior),
        'ABMIL_LSTM', 'ABMIL_LSTM_residual', 'ABMIL_LSTM_residual_proj', 'LSTM_only',
        'LSTM_only_residual', 'LSTM_residual_proj', 'LSTM_last'.
    """
    y = np.array(y)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scorer = make_scorer(average_precision_score, average='macro', response_method='predict_proba')
    all_results = []

    ensemble = True
    ensemble_labels = [0]

    trial_seed = random_state

    np.random.seed(trial_seed)
    random.seed(trial_seed)
    torch.manual_seed(trial_seed)
    torch.cuda.manual_seed_all(trial_seed)

    wrapper = ABMILSklearnWrapper(
        n_labels=y.shape[1], n_epochs=20, batch_size=4,
        device=device,
        ensemble=ensemble, ensemble_labels=ensemble_labels,
        random_state=trial_seed,
        **VARIANT_WRAPPER_KWARGS[variant],
    )

    grid = {
        'hidden_dim': [256],
        'attention_dim': [128],
        'dropout': [0.1],
        'learning_rate': [1e-4, 3e-4, 1e-3],
        'weight_decay': [1e-6, 1e-4, 0.007],
    }
    if variant in _PROJ_VARIANTS:
        grid['lstm_hidden_dim'] = [128, 256]

    model_params = {
        variant: {
            'model': wrapper,
            'params': grid,
        }
    }

    cv = MultilabelStratifiedKFold(n_splits=n_split, shuffle=True, random_state=trial_seed)

    for model_name, mp in model_params.items():
        print(f" Tuning and evaluating model: {model_name}")
        best_models = []
        fold_train_histories = []
        fold_val_histories = []

        clf = GridSearchCV(
            estimator=mp['model'],
            param_grid=mp['params'],
            cv=cv,
            scoring=scorer,
            refit=True,
            n_jobs=1,
            verbose=3
        )

        clf.fit(X_bags, y)

        best_model_instance = clf.best_estimator_

        if hasattr(best_model_instance, 'history_'):
            fold_train_histories.append(best_model_instance.history_.get('train_loss', []))
            fold_val_histories.append(best_model_instance.history_.get('val_loss', []))

        best_models.append(best_model_instance)

        all_results.append({
            'model': model_name,
            'best_models': best_models,
            'train_histories': fold_train_histories,
            'val_histories': fold_val_histories,
            'best_params': clf.best_params_,
            'best_score': clf.best_score_
        })

    return all_results


def train_abmil_all_data(X_bags, y, random_state=42, variant='ABMIL',
                          hidden_dim=256, attention_dim=128, dropout=0.1,
                          learning_rate=1e-4, weight_decay=1e-6,
                          lstm_hidden_dim=None):
    """
    Trains a chosen variant on all the data using given hyperparameters
    (e.g. the best ones found from a previous tuning step).

    variant : str
        'ABMIL' (default, unchanged behavior), 'ABMIL_LSTM', 'ABMIL_LSTM_residual',
        'ABMIL_LSTM_residual_proj', 'LSTM_only', 'LSTM_only_residual',
        'LSTM_residual_proj', 'LSTM_last'.
    """
    y = np.array(y)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ensemble = True
    ensemble_labels = [0]

    trial_seed = random_state

    np.random.seed(trial_seed)
    random.seed(trial_seed)
    torch.manual_seed(trial_seed)
    torch.cuda.manual_seed_all(trial_seed)

    clf = ABMILSklearnWrapper(
        n_labels=y.shape[1], n_epochs=20, batch_size=4,
        device=device,
        ensemble=ensemble, ensemble_labels=ensemble_labels,
        random_state=trial_seed, hidden_dim=hidden_dim, attention_dim=attention_dim,
        dropout=dropout, learning_rate=learning_rate, weight_decay=weight_decay,
        lstm_hidden_dim=lstm_hidden_dim,
        **VARIANT_WRAPPER_KWARGS[variant],
    )

    print(f"Training production model ({variant}) on all available data...")
    clf.fit(X_bags, y)

    return clf


def evaluate_abmil_ood(fitted_wrapper, X_bags_ood):
    """Evaluates the fitted wrapper (any variant) on OOD data. Unchanged."""
    pt_model = fitted_wrapper.model_
    scaler = fitted_wrapper.scaler_
    device = fitted_wrapper.device

    pt_model.eval()

    y_pred_proba_ood, attention_weights_ood = predict_abmil(
        pt_model,
        X_bags_ood,
        scaler,
        device=device
    )
    return y_pred_proba_ood, attention_weights_ood


# ============================================================================
# QUICK USAGE EXAMPLE
# ============================================================================
#
# Run just your original ABMIL, no code changes needed:
#   results = abmil_classifier_tuned(X_bags, y, variants=('ABMIL',))
#
# Compare the core set of variants in one nested-CV sweep, including the new
# projected-residual LSTM variants:
#   results = abmil_classifier_tuned(
#       X_bags, y, num_trials=3,
#       variants=('ABMIL', 'ABMIL_LSTM_residual_proj', 'LSTM_residual_proj'),
#   )
#   df = pd.DataFrame([{'variant': r['model'], 'trial': r['trial'], 'mean_AP': r['mean_AP']}
#                       for r in results])
#   df.groupby('variant')['mean_AP'].agg(['mean', 'std'])
#
# Quick, no-search first look across variants (fast -- fixed hyperparams, no inner CV):
#   quick_results = abmil_classifier_quick(
#       X_bags, y, num_trials=3,
#       variants=('ABMIL', 'ABMIL_LSTM_residual_proj', 'LSTM_residual_proj'),
#       lstm_hidden_dim=64,  # only used by the LSTM variants
#   )
#
# Label-grouped ensemble example (type_a = simple LinearProbe, everything else
# = complex ABMIL+LSTM model), via the full Optuna-tuned nested-CV pipeline:
#   results = abmil_classifier_tuned_optuna(
#       X_bags, y, variants=('ABMIL_LSTM_residual_proj',),
#       ensemble=True, ensemble_labels=(0,),          # 0 == type_a
#       ensemble_n_optuna_trials=15, ensemble_n_split_in=3, ensemble_n_epochs_max=40,
#   )
#
# Train a single ABMIL_LSTM_residual_proj model directly (bypassing tuning) for
# a quick check:
#   model, history, scaler = train_abmil(
#       X_bags_train, y_train, X_bags_val, y_val, n_labels=y.shape[1],
#       use_lstm=True, lstm_residual_proj=True, lstm_hidden_dim=64, pooling='attention',
#   )
from pathlib import Path
import pickle
from src.feature_generation import pool_features

def mil_experiment(encoder_name : str,ensemble = False) :
    
    dir = Path("/idiap/temp/adeych/data")

    if encoder_name == "perch2":
       X_bags_dir = "perch2-bags.pkl"
    elif encoder_name == "effnetb0":
        X_bags_dir = "effnetb0-bags.pkl"
    elif encoder_name == "NLM_BEATs":
        X_bags_dir = "NLM-bags.pkl"
    else:
        raise ValueError(f"Unsupported encoder : {encoder_name}. Please choose from 'perch2', 'effnetb0', or 'NLM_BEATs'.")

    label_cols = ['type_a', 'type_b', 'type_c', 'type_d', 'echo']

    with open(dir / "feature_banks" / X_bags_dir, "rb") as f:
        X_bags = pickle.load(f)

    X_bags_pooled = pool_features(X_bags, windows=False,window_pooled=False,encoder = encoder_name)

    y = np.load(dir / "feature_banks" / "labels.npy")

    results_ABMIL = abmil_classifier_quick(
           X_bags_pooled, y, num_trials=5,
           variants=('ABMIL',),
           ensemble = ensemble,
       )
    results_ABMIL_LTSM_residual = abmil_classifier_quick(
           X_bags_pooled, y, num_trials=5,
           variants=('ABMIL_LSTM_residual',),
           ensemble = ensemble,
       )

    results_LTSM_only_residual = abmil_classifier_quick(
           X_bags_pooled, y, num_trials=5,
           variants=('LSTM_only_residual',),
           ensemble = ensemble,
       )

    results_LTSM_last =abmil_classifier_quick(
           X_bags_pooled, y, num_trials=5,
           variants=('LSTM_last',),
           ensemble = ensemble,
       )

    return results_ABMIL,results_ABMIL_LTSM_residual,results_LTSM_only_residual,results_LTSM_last
