"""
Multiple Instance Learning (MIL) for Multi-Label Bioacoustic Classification.

This module implements an Attention-Based Multiple Instance Learning (ABMIL) model
for multi-label classification of bioacoustic recordings, with an OPTIONAL temporal
context layer (a bidirectional LSTM) inserted before pooling.

Architectural variants are supported via `use_lstm`, `lstm_residual`, and `pooling`:

  1. 'ABMIL'               : use_lstm=False, pooling='attention'
                             -> your original ABMIL, unchanged.
  2. 'ABMIL_LSTM'          : use_lstm=True, lstm_residual=False, pooling='attention'
                             -> recurrent-context ABMIL. Instance embeddings are passed
                             through a BiLSTM to inject temporal context, THEN pooled with
                             the same gated attention as ABMIL. The LSTM output fully
                             replaces the pre-LSTM representation.
  3. 'ABMIL_LSTM_residual' : use_lstm=True, lstm_residual=True, pooling='attention'
                             -> same as above, but H_out = H + LSTM(H) (a skip connection
                             around the LSTM). This matches PS-DeVCEM's actual "residual
                             LSTM" block: attention still sees temporally-contextualized
                             features (their ablation shows this beats attention on raw
                             features), but the residual path lets the block gracefully
                             fall back toward the plain per-instance features if temporal
                             context isn't informative for a given label -- useful
                             insurance with a dataset this size.
  4. 'LSTM_only'           : use_lstm=True, pooling='mean' -> no attention at all.
                             Isolates whether temporal context alone (without learned
                             instance weighting) explains any gain, as a control arm.
  5. 'LSTM_last'           : use_lstm=True, pooling='last' -> classic RNN-sequence-
                             classifier style baseline using the LSTM's final state.

All variants share the same feature_fc -> [optional (residual) LSTM] -> pooling ->
classifier skeleton and the same hidden_dim, so they are directly comparable and
swappable via constructor arguments. Default behavior (use_lstm=False) is IDENTICAL
to the original ABMIL implementation -- nothing changes unless you opt in.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score,
    make_scorer,
    roc_auc_score,
)
import warnings
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from scipy.stats import loguniform
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin
warnings.filterwarnings('ignore')
import random


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
# ABMIL MODEL  (extended with optional LSTM temporal encoder)
# ============================================================================

class ABMIL(nn.Module):
    """
    Attention-Based Multiple Instance Learning, with an optional temporal
    context layer.

    Architecture:
    1. Feature extraction : Maps instance features through FC layer
    2. [Optional] Temporal encoder: BiLSTM over the ordered instance sequence,
       injecting neighboring-window context into each instance's representation
       before pooling (inspired by PS-DeVCEM's residual-temporal-feature attention)
    3. Pooling: 'attention' (ABMIL gated attention + weighted sum),
                'mean' (uniform mean pooling -- no attention, isolates the LSTM's
                        contribution as a standalone baseline),
                'last' (use the LSTM's final hidden state as the bag representation --
                        also attention-free, and only valid when use_lstm=True)
    4. Classification: Multi-label head (sigmoid)

    For multi-label + pooling='attention': trains one attention module per label
    (label-specific attention), same as before. 'mean'/'last' pooling produce a
    single bag representation shared across all labels (no per-label attention,
    since there is no attention at all in those modes).
    """

    def __init__(self, n_features, n_labels, hidden_dim=256, dropout=0.2,
                 attention_dim=128, label_specific_attention=True,
                 use_lstm=False, lstm_hidden_dim=None, lstm_bidirectional=True,
                 lstm_num_layers=1, lstm_residual=False, pooling='attention'):
        """
        Parameters
        ----------
        n_features : int
            Dimension of instance features
        n_labels : int
            Number of labels
        hidden_dim : int
            Hidden dimension for feature processing. Also the dimension the
            classifier and attention modules operate on -- when use_lstm=True,
            the LSTM's output is sized so it still equals hidden_dim, so nothing
            downstream needs to change.
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
            Hidden size per LSTM direction. Defaults to hidden_dim // 2 if
            bidirectional (so concatenated output == hidden_dim), else hidden_dim.
        lstm_bidirectional : bool
            Whether the LSTM reads the bag in both directions. Recommended True
            since you have the full recording available (not a streaming/causal
            setting).
        lstm_num_layers : int
            Number of stacked LSTM layers. Keep at 1 given your dataset size
            unless you have evidence more depth helps.
        lstm_residual : bool
            If True, wrap the LSTM in a skip connection: H_out = H + LSTM(H),
            instead of H_out = LSTM(H). This matches PS-DeVCEM's "residual LSTM"
            block. Only meaningful when use_lstm=True. Recommended default for
            new experiments: with a small dataset, the residual path lets the
            block fall back toward the plain per-instance features (i.e. behave
            more like ABMIL) if temporal context isn't informative for a given
            label, rather than forcing the LSTM to learn an identity mapping
            from scratch when it has little useful signal to add. Requires
            lstm_hidden_dim * num_directions == hidden_dim (guaranteed by the
            default lstm_hidden_dim, so only a concern if you override it).
        pooling : {'attention', 'mean', 'last'}
            'attention' -> standard ABMIL gated-attention pooling (default,
                            matches original behavior when use_lstm=False)
            'mean'      -> simple mean pooling over (optionally LSTM-contextualized)
                            instances, no attention. Use with use_lstm=True to get
                            a "temporal context, no learned weighting" ablation.
            'last'      -> use the LSTM's final hidden state as the bag
                            representation. Requires use_lstm=True, and is
                            incompatible with lstm_residual (there's no sequence
                            of per-instance outputs left to add a skip to -- the
                            residual only applies to the per-timestep outputs
                            used by 'attention'/'mean').
        """
        super().__init__()

        self.n_features = n_features
        self.n_labels = n_labels
        self.label_specific_attention = label_specific_attention
        self.use_lstm = use_lstm
        self.lstm_bidirectional = lstm_bidirectional
        self.lstm_residual = lstm_residual
        self.pooling = pooling

        if pooling not in ('attention', 'mean', 'last'):
            raise ValueError(f"pooling must be one of 'attention', 'mean', 'last', got {pooling!r}")
        if pooling == 'last' and not use_lstm:
            raise ValueError("pooling='last' requires use_lstm=True")
        if pooling == 'last' and lstm_residual:
            raise ValueError("lstm_residual is not applicable to pooling='last' "
                              "(it applies to per-timestep outputs, not the final state)")
        if lstm_residual and not use_lstm:
            raise ValueError("lstm_residual=True requires use_lstm=True")

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
            if lstm_residual and lstm_hidden_dim * num_directions != hidden_dim:
                raise ValueError(
                    f"lstm_residual=True requires lstm_hidden_dim * num_directions "
                    f"({lstm_hidden_dim * num_directions}) == hidden_dim ({hidden_dim}) "
                    f"so the LSTM output can be added back to the input. Leave "
                    f"lstm_hidden_dim=None to get this automatically, or adjust it."
                )
            self.lstm = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=lstm_hidden_dim,
                num_layers=lstm_num_layers,
                batch_first=True,
                bidirectional=lstm_bidirectional,
                dropout=dropout if lstm_num_layers > 1 else 0.0,
            )

        # Attention modules (only needed for pooling='attention')
        if pooling == 'attention':
            if label_specific_attention:
                self.attention_modules = nn.ModuleList([
                    self._build_attention(hidden_dim, attention_dim)
                    for _ in range(n_labels)
                ])
            else:
                self.attention = self._build_attention(hidden_dim, attention_dim)

        self.classifier = nn.Linear(hidden_dim, n_labels)

    @staticmethod
    def _build_attention(hidden_dim, attention_dim):
        """Build attention module."""
        return nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
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
        H_out : (n_instances, hidden_dim)
            LSTM outputs at every timestep (equals H unchanged if use_lstm=False).
            If lstm_residual=True, this is H + LSTM(H) rather than LSTM(H) alone.
        h_n : final hidden state tensor or None
            Only populated when use_lstm=True; needed for pooling='last'. NOTE:
            when lstm_residual=True, h_n is still the raw LSTM final state (the
            residual add only applies to the per-timestep outputs H_out) --
            pooling='last' is disallowed with lstm_residual=True in __init__ to
            avoid ambiguity here.
        """
        if not self.use_lstm:
            return H, None
        H_seq = H.unsqueeze(0)  # (1, n_instances, hidden_dim) -- batch size 1 per bag
        lstm_out, (h_n, c_n) = self.lstm(H_seq)
        lstm_out = lstm_out.squeeze(0)  # (n_instances, hidden_dim)
        H_out = H + lstm_out if self.lstm_residual else lstm_out
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
        H, h_n = self._apply_temporal_encoder(H)

        # --- No-attention pooling modes (used for the LSTM-only ablation) ---
        if self.pooling == 'mean':
            M = torch.mean(H, dim=0, keepdim=True)  # (1, hidden_dim)
            logits = self.classifier(M).squeeze(0)  # (n_labels,)
            return logits, None

        if self.pooling == 'last':
            # h_n: (num_layers * num_directions, batch=1, lstm_hidden_dim)
            if self.lstm_bidirectional:
                last = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (1, hidden_dim)
            else:
                last = h_n[-1]  # (1, hidden_dim)
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

                M = torch.sum(A * H, dim=0, keepdim=True)  # (1, hidden_dim)

                logit = self.classifier(M)[0, label_idx]
                logits_list.append(logit)
                attention_weights_list.append(A_squeezed)

            logits = torch.stack(logits_list)
            attention_weights = attention_weights_list if return_attention else None
        else:
            A = self.attention(H)  # (n_instances, 1)
            A = torch.softmax(A, dim=0)  # (n_instances, 1)

            M = torch.sum(A * H, dim=0, keepdim=True)  # (1, hidden_dim)

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
    device='cpu',
    verbose=True,
    random_state=42,
    ensemble=False,
    # --- new, all default to original ABMIL behavior ---
    use_lstm=False,
    lstm_hidden_dim=None,
    lstm_bidirectional=True,
    lstm_num_layers=1,
    lstm_residual=False,
    pooling='attention',
):
    """
    Train ABMIL model (optionally with an LSTM temporal-context layer, and/or
    attention-free pooling for the LSTM-only ablation).

    New parameters
    --------------
    use_lstm : bool
        Insert a BiLSTM before pooling. Default False = original ABMIL.
    lstm_hidden_dim, lstm_bidirectional, lstm_num_layers, lstm_residual : see ABMIL docstring
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
        pooling=pooling,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if not ensemble:
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

            if not ensemble:
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


def predict_abmil(model, X_bags, scaler, device='cpu'):
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
    Now also exposes use_lstm / lstm_* / pooling so the temporal variants can be
    tuned and compared through the exact same nested-CV machinery.
    """
    def __init__(self, hidden_dim=256, attention_dim=128, dropout=0.2,
                 learning_rate=1e-3, weight_decay=1e-5, batch_size=4,
                 n_epochs=20, n_labels=5, device='cpu', random_state=42,
                 ensemble=False, ensemble_labels=None, lr_C=1.0,
                 use_lstm=False, lstm_hidden_dim=None, lstm_bidirectional=True,
                 lstm_num_layers=1, lstm_residual=False, pooling='attention'):
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
        self.lr_C = lr_C
        self.use_lstm = use_lstm
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_bidirectional = lstm_bidirectional
        self.lstm_num_layers = lstm_num_layers
        self.lstm_residual = lstm_residual
        self.pooling = pooling

    def _get_ensemble_labels(self):
        return set(self.ensemble_labels) if self.ensemble_labels is not None else {0}

    def _meanpool(self, X):
        return np.array([bag.mean(axis=0) for bag in X])

    def fit(self, X, y):
        self.model_, self.history_, self.scaler_ = train_abmil(
            X_bags_train=X, y_train=y,
            n_labels=self.n_labels, n_epochs=self.n_epochs, batch_size=self.batch_size,
            learning_rate=self.learning_rate, weight_decay=self.weight_decay,
            hidden_dim=self.hidden_dim, attention_dim=self.attention_dim, dropout=self.dropout,
            label_specific_attention=True,
            device=self.device, verbose=False, random_state=self.random_state,
            ensemble=self.ensemble,
            use_lstm=self.use_lstm, lstm_hidden_dim=self.lstm_hidden_dim,
            lstm_bidirectional=self.lstm_bidirectional, lstm_num_layers=self.lstm_num_layers,
            lstm_residual=self.lstm_residual,
            pooling=self.pooling,
        )

        if self.ensemble:
            X_train_pool = self._meanpool(X)
            self.lr_scaler_ = StandardScaler()
            X_train_pool_scaled = self.lr_scaler_.fit_transform(X_train_pool)

            self.base_lr_classifiers_ = {}
            for k in self._get_ensemble_labels():
                base_lr = LogisticRegression(C=self.lr_C, max_iter=1000, random_state=self.random_state)
                base_lr.fit(X_train_pool_scaled, y[:, k])
                self.base_lr_classifiers_[k] = base_lr

        return self

    def predict_proba(self, X):
        abmil_preds, _ = predict_abmil(self.model_, X, self.scaler_, device=self.device)

        if not self.ensemble:
            return abmil_preds

        X_pool = self._meanpool(X)
        X_pool_scaled = self.lr_scaler_.transform(X_pool)

        preds = abmil_preds.copy()
        for k in self._get_ensemble_labels():
            base_lr_pred = self.base_lr_classifiers_[k].predict_proba(X_pool_scaled)[:, 1]
            preds[:, k] = base_lr_pred

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
#   'ABMIL'               -- original ABMIL, no LSTM
#   'ABMIL_LSTM'          -- recurrent-context ABMIL, LSTM output replaces H entirely
#   'ABMIL_LSTM_residual' -- same, but H_out = H + LSTM(H) (PS-DeVCEM-style residual
#                            LSTM). Recommended default for new "temporal ABMIL" runs.
#   'LSTM_only'           -- LSTM(+residual) + mean pooling, no attention (ablation
#                            arm to isolate temporal context from learned weighting)
#   'LSTM_only_residual'  -- same, with the residual connection
#   'LSTM_last'           -- LSTM final hidden state + no attention (alternate
#                            attention-free baseline; incompatible with lstm_residual)
VARIANT_WRAPPER_KWARGS = {
    'ABMIL':               dict(use_lstm=False, pooling='attention'),
    'ABMIL_LSTM':          dict(use_lstm=True,  pooling='attention', lstm_residual=False,
                                 lstm_bidirectional=True, lstm_num_layers=1),
    'ABMIL_LSTM_residual': dict(use_lstm=True,  pooling='attention', lstm_residual=True,
                                 lstm_bidirectional=True, lstm_num_layers=1),
    'LSTM_only':           dict(use_lstm=True,  pooling='mean',      lstm_residual=False,
                                 lstm_bidirectional=True, lstm_num_layers=1),
    'LSTM_only_residual':  dict(use_lstm=True,  pooling='mean',      lstm_residual=True,
                                 lstm_bidirectional=True, lstm_num_layers=1),
    'LSTM_last':           dict(use_lstm=True,  pooling='last',      lstm_residual=False,
                                 lstm_bidirectional=True, lstm_num_layers=1),
}

# Base hyperparameter search space, shared by all variants.
BASE_PARAM_GRID = {
    'hidden_dim': [64, 128, 256],
    'attention_dim': [32, 64, 128],  # ignored by LSTM_only / LSTM_last, harmless to search
    'dropout': [0.1, 0.3, 0.5],
    'learning_rate': loguniform(1e-4, 5e-3),
    'weight_decay': loguniform(1e-6, 1e-2),
    'lr_C': [0.1, 1.0, 10.0],
}


def _build_variant_wrapper_and_grid(variant, n_labels, device, trial_seed,
                                     ensemble=True, ensemble_labels=(0,), n_epochs=20, batch_size=4):
    """Builds the ABMILSklearnWrapper + param grid for a given architecture variant name."""
    if variant not in VARIANT_WRAPPER_KWARGS:
        raise ValueError(f"Unknown variant {variant!r}. Choose from {list(VARIANT_WRAPPER_KWARGS)}")

    wrapper = ABMILSklearnWrapper(
        n_labels=n_labels, n_epochs=n_epochs, batch_size=batch_size,
        device=device, ensemble=ensemble, ensemble_labels=list(ensemble_labels),
        random_state=trial_seed,
        **VARIANT_WRAPPER_KWARGS[variant],
    )
    params = dict(BASE_PARAM_GRID)
    return wrapper, params


# ============================================================================
# USAGE / TUNING PIPELINES
# ============================================================================

def abmil_classifier_tuned(X_bags, y, n_split_out=5, n_split_in=5, num_trials=5, random_state=42,
                            n_iter_search=4, variants=('ABMIL',)):
    """
    Trains and evaluates one or more architecture variants with nested cross-validation
    and hyperparameter tuning.

    X_bags: list of arrays, each array is (n_windows, n_features) for one recording
    y: (n_bags, n_labels) array of multi-label targets
    n_iter_search: int, number of parameter settings sampled in RandomizedSearchCV per fold
    n_split_out, n_split_in: outer/inner CV splits
    num_trials: int, number of repeated nested CV runs with different seeds
    random_state: int, base random seed
    variants: tuple of str
        Which architecture(s) to run in this sweep, drawn from
        VARIANT_WRAPPER_KWARGS: 'ABMIL' (default -- identical to original behavior),
        'ABMIL_LSTM', 'LSTM_only', 'LSTM_last'. Pass e.g.
        variants=('ABMIL', 'ABMIL_LSTM', 'LSTM_only') to compare all three in one call.

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
                            lr_C=0.1, batch_size=4, n_epochs=20,
                            ensemble=True, ensemble_labels=(0,)):
    """
    Fast, no-hyperparameter-search counterpart to abmil_classifier_tuned.

    Runs plain outer cross-validation only (no inner RandomizedSearchCV loop) --
    every fold fits ONE model per variant at a single fixed hyperparameter
    configuration, rather than searching n_iter_search configs per fold via an
    inner CV. This drops the search's `n_iter_search * n_split_in` extra fits per
    outer fold down to just 1, so it's a cheap way to get a first read on whether
    a variant (e.g. 'ABMIL' vs 'ABMIL_LSTM_residual') looks promising before
    spending the time on a full nested-CV hyperparameter search.

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
        ('ABMIL', 'ABMIL_LSTM_residual', 'LSTM_only_residual').
    hidden_dim, attention_dim, dropout, learning_rate, weight_decay, lr_C,
    batch_size, n_epochs : fixed hyperparameters used for every fold/variant/trial.
        Pick reasonable defaults up front, or pass in whatever a previous full
        search suggested as a sane starting point -- this function does not
        tune them.
    ensemble, ensemble_labels : same meaning as elsewhere (ultra-minority-class
        logistic-regression fallback).

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
                    random_state=trial_seed,
                    hidden_dim=hidden_dim, attention_dim=attention_dim, dropout=dropout,
                    learning_rate=learning_rate, weight_decay=weight_decay, lr_C=lr_C,
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
        'ABMIL_LSTM', 'LSTM_only', 'LSTM_last'.
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

    model_params = {
        variant: {
            'model': wrapper,
            'params': {
                'hidden_dim': [256],
                'attention_dim': [128],
                'dropout': [0.1],
                'learning_rate': [1e-4, 3e-4, 1e-3],
                'weight_decay': [1e-6, 1e-4, 0.007],
                **({'lr_C': [0.1]} if ensemble else {})
            }
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
                          learning_rate=1e-4, weight_decay=1e-6, lr_C=0.1):
    """
    Trains a chosen variant on all the data using given hyperparameters
    (e.g. the best ones found from a previous tuning step).

    variant : str
        'ABMIL' (default, unchanged behavior), 'ABMIL_LSTM', 'LSTM_only', 'LSTM_last'.
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
        dropout=dropout, learning_rate=learning_rate, weight_decay=weight_decay, lr_C=lr_C,
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
# Compare the core set of variants in one nested-CV sweep:
#   results = abmil_classifier_tuned(
#       X_bags, y, num_trials=3,
#       variants=('ABMIL', 'ABMIL_LSTM_residual', 'LSTM_only_residual'),
#   )
#   df = pd.DataFrame([{'variant': r['model'], 'trial': r['trial'], 'mean_AP': r['mean_AP']}
#                       for r in results])
#   df.groupby('variant')['mean_AP'].agg(['mean', 'std'])
#
# Also want the non-residual LSTM variants for comparison? Just add them to `variants`:
#   variants=('ABMIL', 'ABMIL_LSTM', 'ABMIL_LSTM_residual', 'LSTM_only', 'LSTM_only_residual')
#
# Quick, no-search first look across variants (fast -- fixed hyperparams, no inner CV):
#   quick_results = abmil_classifier_quick(
#       X_bags, y, num_trials=3,
#       variants=('ABMIL', 'ABMIL_LSTM_residual', 'LSTM_only_residual'),
#   )
#   df = pd.DataFrame([{'variant': r['model'], 'trial': r['trial'], 'mean_AP': r['mean_AP']}
#                       for r in quick_results])
#   df.groupby('variant')['mean_AP'].agg(['mean', 'std'])
#   # Once a variant or two look promising, run the full nested-CV search on just those:
#   results = abmil_classifier_tuned(X_bags, y, num_trials=3, variants=('ABMIL_LSTM_residual',))
#
# Train a single ABMIL_LSTM_residual model directly (bypassing tuning) for a quick check:
#   model, history, scaler = train_abmil(
#       X_bags_train, y_train, X_bags_val, y_val, n_labels=y.shape[1],
#       use_lstm=True, lstm_residual=True, pooling='attention',
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