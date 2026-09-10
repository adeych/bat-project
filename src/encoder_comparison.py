"""
Baseline classifier comparison: SVM / Logistic Regression (LinearProbe) /
Random Forest / MLP, evaluated via nested cross-validation with
GridSearchCV, on already mean-pooled (n_samples, n_features) inputs.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score, make_scorer
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from src.linear_probe import LinearProbeSklearnWrapper


# ============================================================================
# MLP MODEL
# ============================================================================

class MLP(BaseEstimator, ClassifierMixin):
    """
    Linear -> ReLU -> Dropout -> Linear multi-label classifier. 
    """
    _estimator_type = "classifier"

    def __init__(
        self,
        n_labels=5,
        input_dim=1536,
        hidden_dim=128,
        lr=1e-4,
        epochs=40,
        dropout=0.3,
        batch_size=4,
        device='cpu',
        random_state=None,
    ) -> None:
        self.n_labels = n_labels
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.dropout = dropout
        self.batch_size = batch_size
        self.device = device
        self.random_state = random_state

    def _build_model(self):
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.n_labels),
        )

    def fit(self, X, y, X_val=None, y_val=None, **kwargs):
        device = torch.device(self.device)

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_state)
            g = torch.Generator()
            g.manual_seed(self.random_state)
        else:
            g = None

        # sklearn Pipeline step-prefixed kwargs (e.g. pipeline.fit(X, y,
        # model__X_val=..., model__y_val=...)) -- inert until something
        # upstream actually passes validation data this way.
        if X_val is None and 'model__X_val' in kwargs:
            X_val = kwargs['model__X_val']
        if y_val is None and 'model__y_val' in kwargs:
            y_val = kwargs['model__y_val']

        X_tensor = torch.FloatTensor(np.asarray(X))
        y_tensor = torch.FloatTensor(np.asarray(y))

        has_validation = X_val is not None and y_val is not None
        if has_validation:
            X_val_tensor = torch.FloatTensor(np.asarray(X_val))
            y_val_tensor = torch.FloatTensor(np.asarray(y_val))

        self.model_ = self._build_model().to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(
            dataset, batch_size=min(self.batch_size, len(X)), shuffle=True, generator=g
        )

        self.history_ = {'train_loss': [], 'val_loss': [], 'val_ap': [], 'val_auc': []}

        for epoch in range(self.epochs):
            self.model_.train()
            epoch_train_loss = 0.0
            batch_count = 0

            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = self.model_(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()
                batch_count += 1

            self.history_['train_loss'].append(epoch_train_loss / batch_count)

            if has_validation:
                self.model_.eval()
                with torch.no_grad():
                    X_v, y_v = X_val_tensor.to(device), y_val_tensor.to(device)
                    val_logits = self.model_(X_v)
                    val_loss = criterion(val_logits, y_v)
                    self.history_['val_loss'].append(val_loss.item())

                    y_true_np = y_v.cpu().numpy()
                    y_pred_np = val_logits.detach().cpu().numpy()
                    self.history_['val_ap'].append(
                        average_precision_score(y_true_np, y_pred_np, average='macro')
                    )
                    self.history_['val_auc'].append(
                        roc_auc_score(y_true_np, y_pred_np, average='macro')
                    )
            else:
                self.history_['val_loss'].append(np.nan)
                self.history_['val_ap'].append(np.nan)
                self.history_['val_auc'].append(np.nan)

        self.classes_ = np.arange(self.n_labels)
        return self

    def predict_proba(self, X):
        device = torch.device(self.device)
        self.model_.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(np.asarray(X)).to(device)
            logits = self.model_(X_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs > threshold).astype(int)


# ============================================================================
# NESTED-CV BASELINE COMPARISON
# ============================================================================

def linear_probe_tuned(X, y, n_split_out=5, n_split_in=5, num_trials=5, random_state=42):
    """
    Performs nested cross-validation with hyperparameter tuning for multiple
    classifiers (SVM, Logistic Regression, Random Forest, MLP).

    - X: Feature matrix (numpy array).
    - y: Multi-label binary target matrix (numpy array).
    - n_split_out: Number of splits for the outer loop (model evaluation).
    - n_split_in: Number of splits for the inner loop (hyperparameter tuning).
    - num_trials: Number of repeated trials with different random seeds for robustness.
    - random_state: Base random seed for reproducibility.

    Returns a list of results, one per (trial, model), with the same schema
    as abmil_classifier_tuned_optuna / linear_probe_tuned_optuna: true
    labels, predicted probabilities, performance metrics, best_models (the
    fitted Pipeline per fold), best_params, and train/val loss histories
    (empty for SVM/RandomForest, which don't train by epoch).
    """
    scorer = make_scorer(average_precision_score, average='macro', response_method='predict_proba')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = []

    for i in range(num_trials):
        trial_seed = random_state + i
        print(f"Starting Trial {i+1}/{num_trials} with random_state={trial_seed}...")

        np.random.seed(trial_seed)
        random.seed(trial_seed)
        torch.manual_seed(trial_seed)
        torch.cuda.manual_seed_all(trial_seed)

        model_params = {
            'SVM': {
                'model': OneVsRestClassifier(SVC(
                            probability=True,
                            random_state=trial_seed)),
                'params': {
                    'model__estimator__C': [1, 10, 20],
                    'model__estimator__kernel': ['rbf', 'linear'],
                    'model__estimator__gamma': ['scale', 'auto', 0.01, 0.1]
                },
                'needs_scaler': True,
            },
            'Logistic Regression': {
                'model': LinearProbeSklearnWrapper(
                            n_labels=y.shape[1], n_epochs=40, batch_size=4, dropout=0.0,
                            learning_rate=1e-4, weight_decay=1e-5, device=device,
                            random_state=trial_seed, early_stopping=False),
                'params': {
                    'model__learning_rate': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
                    'model__n_epochs': [40],
                },
                'needs_scaler': False,  # train_linear_probe already fits its own StandardScaler
            },
            'Random Forest': {
                'model': RandomForestClassifier(
                            n_estimators=100,
                            random_state=trial_seed),
                'params': {
                    'model__n_estimators': [100],
                    'model__max_depth': [None, 10, 20]
                },
                'needs_scaler': True,  # harmless no-op for trees, kept for pipeline consistency
            },
            'MLP': {
                'model': MLP(
                    n_labels=y.shape[1],
                    input_dim=X.shape[1],
                    hidden_dim=128,
                    lr=1e-4,
                    epochs=40,
                    dropout=0.3,
                    device=device,
                    random_state=trial_seed
                ),
                'params': {
                    'model__lr': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
                    'model__hidden_dim': [128],
                    'model__epochs': [40],
                    'model__dropout': [0.3]
                },
                'needs_scaler': True,
            },
        }

        inner_cv = MultilabelStratifiedKFold(n_splits=n_split_in, shuffle=True, random_state=trial_seed)
        outer_cv = MultilabelStratifiedKFold(n_splits=n_split_out, shuffle=True, random_state=trial_seed)

        for model_name, mp in model_params.items():
            print(f"  Tuning and evaluating model: {model_name}")
            all_y_true = []
            all_y_pred_proba = []
            all_test_indices = []
            best_models = []
            all_best_params = []
            fold_train_histories = []
            fold_val_histories = []

            outer_scores = []

            for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
                print(f"    Evaluating fold {fold+1}/{n_split_out}")
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                steps = []
                if mp['needs_scaler']:
                    steps.append(('scaler', StandardScaler()))
                steps.append(('model', mp['model']))
                pipeline = Pipeline(steps)

                # PyTorch/CUDA-based models (MLP, Logistic Regression) must
                # not run under joblib's process-based parallelism -- each
                # subprocess would try to grab a CUDA context concurrently.
                n_jobs = 1 if model_name in ('MLP', 'Logistic Regression') else -1

                clf = GridSearchCV(estimator=pipeline, param_grid=mp['params'], cv=inner_cv,
                                    scoring=scorer, refit=True, n_jobs=n_jobs)

                clf.fit(X_train, y_train)

                best_pipeline = clf.best_estimator_
                underlying_model = best_pipeline.named_steps['model']
                if hasattr(underlying_model, 'history_'):
                    fold_train_histories.append(underlying_model.history_.get('train_loss', []))
                    fold_val_histories.append(underlying_model.history_.get('val_loss', []))
                else:
                    # SVM / RandomForest don't train by epoch -- nothing to record.
                    fold_train_histories.append([])
                    fold_val_histories.append([])

                y_pred_proba = clf.predict_proba(X_test)

                if isinstance(y_pred_proba, list):
                    y_pred_proba = np.column_stack([prob[:, 1] for prob in y_pred_proba])
                elif isinstance(y_pred_proba, np.ndarray) and y_pred_proba.ndim == 3:
                    y_pred_proba = y_pred_proba[:, :, 1].T

                fold_score = average_precision_score(y_test, y_pred_proba, average='macro')
                outer_scores.append(fold_score)

                all_y_true.append(y_test)
                all_y_pred_proba.append(y_pred_proba)
                all_test_indices.append(test_idx)
                best_models.append(best_pipeline)
                all_best_params.append(clf.best_params_)

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
                'val_histories': fold_val_histories,
                'best_params': all_best_params,
            })

    return all_results