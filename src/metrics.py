import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    auc,
    log_loss,
    brier_score_loss,
    balanced_accuracy_score,
    hamming_loss,
    accuracy_score,
    f1_score,
)
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')


#################################################################################
## METRIC CALCULATION AND SUMMARIZATION FUNCTIONS
#################################################################################

def calculate_ece(y_true, y_prob, n_bins=10, strategy='uniform'):
    """Calculates the Expected Calibration Error (ECE) for binary targets."""
    if strategy == 'uniform':
        bin_edges = np.linspace(0, 1, n_bins + 1)
    elif strategy == 'quantile':
        bin_edges = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))
        bin_edges = np.unique(bin_edges) # Handle overlapping quantiles
        if len(bin_edges) < 2:
            return 0.0
    
    ece = 0.0
    n_samples = len(y_true)
    
    for i in range(len(bin_edges) - 1):
        bin_idx = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i+1])
        if i == len(bin_edges) - 2: # Include right edge for the last bin
            bin_idx = bin_idx | (y_prob == bin_edges[i+1])
            
        bin_size = np.sum(bin_idx)
        if bin_size > 0:
            bin_acc = np.mean(y_true[bin_idx])
            bin_conf = np.mean(y_prob[bin_idx])
            ece += (bin_size / n_samples) * np.abs(bin_acc - bin_conf)
            
    return ece

def summarize(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return {
            "mean": values[0],
            "std": 0.0,
            "min": values[0],
            "max": values[0],
        }
    return {
        "mean": np.mean(values),
        "std": np.std(values, ddof=1),
        "min": np.min(values),
        "max": np.max(values),
    }

def compute_metrics(y_true, y_pred_proba, 
                    label_names=None, threshold=0.5):
    """Compute comprehensive multi-label metrics for a single evaluation."""
    y_pred_binary = (y_pred_proba > threshold).astype(int)
    
    n_labels = y_true.shape[1]
    if label_names is None:
        label_names = [f"Label {i}" for i in range(n_labels)]
    
    metrics = {
        'Macro-AUC': roc_auc_score(y_true, y_pred_proba, average='macro'),
        'cmAP': average_precision_score(y_true, y_pred_proba, average='macro'),
        #'Macro-Balanced Accuracy' : np.mean([balanced_accuracy_score(y_true[:, i], y_pred_binary[:, i]) for i in range(n_labels)]),
        'AP per label': {},
        #'Brier per label': {},
        #'Log-Loss per label': {}
    }
    
    #Per-label metrics
    for i, label_name in enumerate(label_names):
        # Average Precision (Recommended over trapezoidal PR-AUC)
        ap = average_precision_score(y_true[:, i], y_pred_proba[:, i])
        metrics['AP per label'][label_name] = ap
        
        # Error Scores
        #metrics['Brier per label'][label_name] = brier_score_loss(y_true[:, i], y_pred_proba[:, i])
        #metrics['Log-Loss per label'][label_name] = log_loss(y_true[:, i], y_pred_proba[:, i], labels=[0,1])
    
    # Aggregate macro scores
    metrics['Brier (macro)'] = np.mean( [brier_score_loss(y_true[:, i], y_pred_proba[:, i]) for i in range(n_labels)] )
    metrics['Log-Loss (macro)'] = np.mean( [log_loss(y_true[:, i], y_pred_proba[:, i], labels=[0,1]) for i in range(n_labels)] )
    
    return metrics

def compute_fold_metrics(y_true, y_pred_proba, label_names=None, threshold=0.5):
    fold_metrics = []
    for fold in range(len(y_true)):
        fold_metrics.append(compute_metrics(
            y_true[fold], 
            y_pred_proba[fold], 
            label_names, 
            threshold
        ))
    return fold_metrics


def compute_cv_stats(fold_metrics):
    """Compute metrics across cross-validation folds."""
    result = {}
    metrics = fold_metrics[0].keys()
    for metric in metrics:

        first_value = fold_metrics[0][metric]

        # scalar metric
        if np.isscalar(first_value):

            vals = [m[metric] for m in fold_metrics]
            result[metric] = summarize(vals)

        # nested dict metric
        elif isinstance(first_value, dict):

            result[metric] = {}

            labels = first_value.keys()

            for label in labels:
                vals = [m[metric][label] for m in fold_metrics]
                result[metric][label] = summarize(vals)

        else:
            raise TypeError(f"Unsupported metric type for key={metric}")
    
    return result

def result_summary(y_true, y_pred_proba, label_names=None, threshold=0.5,stats : bool = True) :
    oof_true = np.concatenate(y_true, axis=0)
    oof_pred_proba = np.concatenate(y_pred_proba, axis=0)
    fold_metrics = compute_fold_metrics(y_true, y_pred_proba, label_names, threshold)

    if stats :
        results = compute_cv_stats(fold_metrics)
    else :
        results = {
            "oof" :{
                "metrics" : compute_metrics(oof_true, oof_pred_proba, label_names, threshold),
                "true": oof_true,
                "pred_proba": oof_pred_proba
            },
            "cv" : {
                "stats" : compute_cv_stats(fold_metrics),
                "folds": fold_metrics
            }
        }
    return results

def compile_results(all_results,label_names=None,stats : bool = True,encoder : str = 'perch2') :
    if label_names is None:
        label_names = ['Type A', 'Type B', 'Type C', 'Type D', 'Echo']
    
    compiled_results = {}
    unique_models = list(set([r['model'] for r in all_results]))
    for model_name in unique_models:
        # 1. Gather all trials for this specific classifier
        model_trials = [r for r in all_results if r['model'] == model_name]
        y_true = []
        y_pred_proba = []
        for trial_data in model_trials:
            # These are already concatenated across the outer folds!
            y_true.append(trial_data['oof_y_true'])
            y_pred_proba.append(trial_data['oof_y_pred_proba'])
        # 2. Compute summary metrics for this model
        model_results = result_summary(y_true, y_pred_proba, label_names, stats=stats)
        compiled_results[encoder + " " + model_name] = model_results

    return compiled_results

def compile_results_folds(all_results,label_names=None,stats : bool = True,encoder : str = 'perch2') :
    if label_names is None:
        label_names = ['Type A', 'Type B', 'Type C', 'Type D', 'Echo']
    
    compiled_results = {}
    unique_models = list(set([r['model'] for r in all_results]))
    for model_name in unique_models:
        # 1. Gather all trials for this specific classifier
        model_trials = [r for r in all_results if r['model'] == model_name]
        y_true = []
        y_pred_proba = []
        for trial_data in model_trials:
            # These are already concatenated across the outer folds!
            y_true.append(trial_data['y_true_cv'])
            y_pred_proba.append(trial_data['y_pred_proba_cv'])
        # 2. Compute summary metrics for this model
        y_true = np.concatenate(y_true,axis = 0)
        y_pred_proba = np.concatenate(y_pred_proba,axis=0)
        model_results = result_summary(y_true, y_pred_proba, label_names, stats=stats)
        compiled_results[encoder + " " + model_name] = model_results

    return compiled_results


#################################################################################
## VISUALIZATION FUNCTIONS FOR METRICS
#################################################################################
def generate_metrics_table(all_results,label_names=None) :
    global_rows = []
    class_rows = []

    # Dynamic class names extracted safely from your labels key
    if label_names is None:
        class_names = ['Type A', 'Type B', 'Type C', 'Type D', 'Echo']
    else:
        class_names = label_names

    for model, stats in all_results.items():
        # 1. Parse Global Data Matrix
        global_row = {
            "Model": model,
            "Macro-AUC": f"{stats['Macro-AUC']['mean']:.3f} ± {stats['Macro-AUC']['std']:.3f}",
            "Macro-AP (cmAP)": f"{stats['cmAP']['mean']:.3f} ± {stats['cmAP']['std']:.3f}",
            #"Balanced Accuracy": f"{stats['Macro-Balanced Accuracy']['mean']:.3f} ± {stats['Macro-Balanced Accuracy']['std']:.3f}",
            "Brier Score ↓": f"{stats['Brier (macro)']['mean']:.3f} ± {stats['Brier (macro)']['std']:.3f}",
            "Log-Loss ↓": f"{stats['Log-Loss (macro)']['mean']:.3f} ± {stats['Log-Loss (macro)']['std']:.3f}"
        }
        global_rows.append(global_row)

        # 2. Parse Class-Specific Average Precision Data Matrix
        class_row = {"Model": model}
        for label in class_names:
            label_stats = stats['AP per label'][label]
            class_row[f"{label} AP"] = f"{label_stats['mean']:.3f} ± {label_stats['std']:.3f}"
        class_rows.append(class_row)

    # Convert arrays into clean Pandas DataFrames
    global_df = pd.DataFrame(global_rows)
    class_df = pd.DataFrame(class_rows)
    return global_df, class_df


def plot_comprehensive_boxplots(all_encoders_results, label_names=None):
    """
    Plots a 2x3 grid of subplots with layout:
    Row 1: Type A, Type B, Type C
    Row 2: Type D, Echo, cmAP
    Each subplot shows boxplots of the specified metric across different models and encoders, with a unified chance level indicated.
    Withstatistics over all folds and trials for each model.
    all_encoders_results : of format {encoder_name: [trial_dicts...]} where trial_dicts contain 'model', 'y_true_cv', 'y_pred_proba_cv' for each fold.
    label_names : list of class names corresponding to the columns in y_true and y_pred_proba.
    """
    if label_names is None:
        label_names = ['Type A', 'Type B', 'Type C', 'Type D', 'Echo']
        
    # --- Step 1: Parse Raw Fold Arrays into a Structured DataFrame ---
    rows = []
    baseline_stats = {} 
    
    for encoder_name, trials_list in all_encoders_results.items():
        baseline_stats[encoder_name] = {'cmAP': [], 'classes': {l: [] for l in label_names}}
        
        for trial in trials_list:
            model_name = trial['model']
            y_true_folds = trial['y_true_cv']          
            y_pred_proba_folds = trial['y_pred_proba_cv']  
            
            for fold_idx in range(len(y_true_folds)):
                y_true = y_true_folds[fold_idx]
                y_pred_proba = y_pred_proba_folds[fold_idx]
                
                fold_cmap = average_precision_score(y_true, y_pred_proba, average='macro')
                
                fold_class_aps = {}
                for idx, name in enumerate(label_names):
                    fold_class_aps[name] = average_precision_score(y_true[:, idx], y_pred_proba[:, idx])
                
                if any(x in model_name.lower() for x in ['prevalence', 'dummy', 'guesser']):
                    baseline_stats[encoder_name]['cmAP'].append(fold_cmap)
                    for name in label_names:
                        baseline_stats[encoder_name]['classes'][name].append(fold_class_aps[name])
                else:
                    clean_model_name = model_name
                    if any(x in model_name.lower() for x in ['logistic', 'regressor', 'regression']):
                        clean_model_name = 'lr'
                    elif 'svm' in model_name.lower():
                        clean_model_name = 'svm'
                    elif any(x in model_name.lower() for x in ['forest', 'rf']):
                        clean_model_name = 'rf'
                    elif 'mlp' in model_name.lower():
                        clean_model_name = 'mlp'
                        
                    row = {
                        'Encoder': encoder_name,
                        'Model': clean_model_name,
                        'cmAP': fold_cmap
                    }
                    for name in label_names:
                        row[f'AP_{name}'] = fold_class_aps[name]
                    rows.append(row)
                    
    df = pd.DataFrame(rows)
    
    model_order = ['lr', 'svm', 'rf', 'mlp']
    df = df[df['Model'].isin(model_order)]
    
    # --- Step 2: Establish Layout Structure (2 Rows x 3 Columns) ---
    sns.set_context("paper", font_scale=1.1)
    sns.set_style("whitegrid", {"grid.linestyle": "--", "grid.alpha": 0.5})
    
    pastel_palette = sns.color_palette("Pastel1", n_colors=3)
    encoders_list = list(all_encoders_results.keys())
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
            
            # Plot distributions showing outliers only
            sns.boxplot(
                data=df, x='Model', y=col_name, hue='Encoder',
                order=model_order, palette=encoder_colors,
                ax=ax, width=0.6, showfliers=True, 
                flierprops=dict(marker='o', markerfacecolor='gray', markersize=4, markeredgecolor='none', alpha=0.6),
                boxprops=dict(edgecolor='#4d4d4d', linewidth=1.2),
                whiskerprops=dict(color='#4d4d4d', linewidth=1.1),
                capprops=dict(color='#4d4d4d', linewidth=1.1),
                medianprops=dict(color='#2c3e50', linewidth=1.5)
            )
            
            # --- Step 4: Compute a Single Unified Chance Level ---
            chance_title_suffix = ""
            if encoders_list:
                first_encoder = encoders_list[0]
                
                if col_name == 'cmAP':
                    baseline_vals = baseline_stats[first_encoder]['cmAP']
                else:
                    class_name = col_name.replace('AP_', '')
                    baseline_vals = baseline_stats[first_encoder]['classes'][class_name]
                    
                if len(baseline_vals) > 0:
                    mean_baseline = np.mean(baseline_vals)
                    chance_title_suffix = f" | Chance level = {mean_baseline:.2f}"
            
            # --- Step 5: Refine Independent Scales & Clean multi-line Labels ---
            ax.set_title(display_title, fontsize=18, fontweight='bold', pad=18)
            ax.text(0.5, 1.03, chance_title_suffix, transform=ax.transAxes, 
                    fontsize=12, color='#666666', style='italic', ha='center')
            
            ax.set_xlabel("", fontsize=13)
            ax.set_ylabel("Average Precision (AP)" if c_idx == 0 else "", fontsize=14)
            
            # Formats names onto 2 lines, kept completely flat (rotation=0) and centered
            display_labels = ["Logistic\nRegressor", "SVM", "Random\nForest", "MLP"]
            ax.set_xticklabels(display_labels, rotation=0, ha='center', fontsize=13)
            ax.tick_params(axis='y', labelsize=14)
            
            # Auto-scale limits nicely based on data variance limits
            current_data = df[col_name].dropna()
            if not current_data.empty:
                ymin, ymax = current_data.min(), current_data.max()
                yrange = ymax - ymin if ymax != ymin else 0.1
                ax.set_ylim(max(0, ymin - 0.05 * yrange), min(1.02, ymax + 0.05 * yrange))
                
            ax.get_legend().remove()
                
    # --- Step 6: Generate a Unified Global External Legend Below Layout ---
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
    
    # Balanced layout adjustments to give breathing room for the bottom legend and multi-line labels
    plt.tight_layout(rect=[0, 0.12, 1, 0.95])
    plt.subplots_adjust(hspace=0.4, wspace=0.22)
    plt.show()

def plot_comprehensive_boxplots_no_echo(all_encoders_results, label_names=None):
    """
    Plots a 2x3 grid (now only 5 active subplots) excluding Echolocation.
    all_encoders_results : of format {encoder_name: [trial_dicts...]} where trial_dicts contain 'model', 'y_true_cv', 'y_pred_proba_cv' for each fold.
    label_names : list of class names corresponding to the columns in y_true and y_pred_proba.
    """
    # 1. Ensure Echo is removed from the processed labels
    if label_names is None:
        label_names = ['Type A', 'Type B', 'Type C', 'Type D']
    else:
        label_names = [l for l in label_names if 'echo' not in l.lower()]
        
    # --- Step 1: Parse Raw Fold Arrays ---
    rows = []
    baseline_stats = {} 
    
    for encoder_name, trials_list in all_encoders_results.items():
        baseline_stats[encoder_name] = {'cmAP': [], 'classes': {l: [] for l in label_names}}
        
        for trial in trials_list:
            model_name = trial['model']
            y_true_folds = trial['y_true_cv']          
            y_pred_proba_folds = trial['y_pred_proba_cv']  
            
            for fold_idx in range(len(y_true_folds)):
                y_true = y_true_folds[fold_idx]
                y_pred_proba = y_pred_proba_folds[fold_idx]
                
                fold_cmap = average_precision_score(y_true, y_pred_proba, average='macro')
                
                fold_class_aps = {}
                for idx, name in enumerate(label_names):
                    # Indexing assumes y_true/y_pred_proba still have the same columns
                    # If 'Echo' was index 4, it is ignored here because label_names doesn't contain it
                    fold_class_aps[name] = average_precision_score(y_true[:, idx], y_pred_proba[:, idx])
                
                if any(x in model_name.lower() for x in ['prevalence', 'dummy', 'guesser']):
                    baseline_stats[encoder_name]['cmAP'].append(fold_cmap)
                    for name in label_names:
                        baseline_stats[encoder_name]['classes'][name].append(fold_class_aps[name])
                else:
                    clean_model_name = model_name
                    if any(x in model_name.lower() for x in ['logistic', 'regressor', 'regression']):
                        clean_model_name = 'lr'
                    elif 'svm' in model_name.lower():
                        clean_model_name = 'svm'
                    elif any(x in model_name.lower() for x in ['forest', 'rf']):
                        clean_model_name = 'rf'
                    elif 'mlp' in model_name.lower():
                        clean_model_name = 'mlp'
                        
                    row = {'Encoder': encoder_name, 'Model': clean_model_name, 'cmAP': fold_cmap}
                    for name in label_names:
                        row[f'AP_{name}'] = fold_class_aps[name]
                    rows.append(row)
                    
    df = pd.DataFrame(rows)
    model_order = ['lr', 'svm', 'rf', 'mlp']
    df = df[df['Model'].isin(model_order)]
    
    # --- Step 2: Grid Setup ---
    sns.set_context("paper", font_scale=1.1)
    sns.set_style("whitegrid", {"grid.linestyle": "--", "grid.alpha": 0.5})
    
    pastel_palette = sns.color_palette("Pastel1", n_colors=len(all_encoders_results))
    encoders_list = list(all_encoders_results.keys())
    encoder_colors = {encoders_list[i]: pastel_palette[i] for i in range(len(encoders_list))}
    
    # Organize 5 targets into a 2x3 grid (last cell will be empty)
    targets = [(f'AP_{l}', l) for l in label_names] + [('cmAP', r'$cmAP$')]
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 12))
    axes_flat = axes.flatten()

    # --- Step 3: Visualization Loop ---
    for i, ax in enumerate(axes_flat):
        if i < len(targets):
            col_name, display_title = targets[i]
            
            sns.boxplot(data=df, x='Model', y=col_name, hue='Encoder',
                        order=model_order, palette=encoder_colors, ax=ax, width=0.6,
                        showfliers=True, flierprops=dict(marker='o', markersize=4, alpha=0.6),
                        boxprops=dict(edgecolor='#4d4d4d', linewidth=1.2))
            
            # Chance logic
            if encoders_list:
                first_enc = encoders_list[0]
                vals = baseline_stats[first_enc]['cmAP'] if col_name == 'cmAP' else \
                       baseline_stats[first_enc]['classes'][col_name.replace('AP_', '')]
                if vals: ax.text(0.5, 1.03, f"Chance: {np.mean(vals):.2f}", transform=ax.transAxes, ha='center', color='#666666')

                    # Increase font sizes across the board
            ax.set_title(display_title, fontsize=20, fontweight='bold', pad=25) # Increased pad

            
            ax.set_xlabel("", fontsize=16)
            ax.set_ylabel("Average Precision (AP)" if i % 3 == 0 else "", fontsize=18)

            # Larger tick labels
            ax.set_xticklabels(["Logistic\nReg", "SVM", "Random\nForest", "MLP"], fontsize=14)
            ax.tick_params(axis='y', labelsize=14)
            ax.get_legend().remove()
        else:
            ax.axis('off') # Hide the 6th empty plot

    # --- Step 4: Legend & Layout ---
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Encoders", loc='lower center', 
               ncol=len(encoders_list), bbox_to_anchor=(0.5, 0.02),
               fontsize=16, title_fontsize=18) # Larger legend
    
    # Adjust layout to accommodate larger text
    plt.tight_layout(rect=[0, 0.12, 1, 0.95])
    plt.show()

def plot_comprehensive_mlp_boxplots(balancing_results, augmented_results, label_names=None):
    """
    Plots a 2x3 grid of subplots evaluating 5 MLP strategies for a research paper:
    Row 1: Type A, Type B, Type C
    Row 2: Type D, Echo, cmAP
    
    Accepts the balancing results list and the data-augmented results list separately.
    balancing_results = list of dicts for MLP balancing strategies (e.g., Baseline, Class Weights, Focal Loss, Oversampled)
    with format: [{'model': 'MLP_FocalLoss', 'y_true_cv': [...], 'y_pred_proba_cv': [...]}, ...]
    augmented_results =  dict for the Data Augmentation strategy (with same structure as balancing_results entries)
    """
    if label_names is None:
        label_names = ['Type A', 'Type B', 'Type C', 'Type D', 'Echo']
        
    rows = []
    
    # --- Step 1: Parse Standard Balancing Results ---
    for trial_entry in balancing_results:
        model_name = trial_entry['model']      
        y_true_folds = trial_entry['y_true_cv']          
        y_pred_proba_folds = trial_entry['y_pred_proba_cv']  
        
        clean_name = model_name.replace('MLP_', '')
        if clean_name == 'ClassWeights':
            clean_name = 'Class Weights'
        elif clean_name == 'FocalLoss':
            clean_name = 'Focal Loss'
            
        for fold_idx in range(len(y_true_folds)):
            y_true = y_true_folds[fold_idx]
            y_pred_proba = y_pred_proba_folds[fold_idx]
            
            fold_cmap = average_precision_score(y_true, y_pred_proba, average='macro')
            row = {'Strategy': clean_name, 'cmAP': fold_cmap}
            for idx, name in enumerate(label_names):
                row[f'AP_{name}'] = average_precision_score(y_true[:, idx], y_pred_proba[:, idx])
            rows.append(row)
            
    # --- Step 2: Parse Data Augmented Results ---
    for trial_entry in augmented_results:
        y_true_folds = trial_entry['y_true_cv']          
        y_pred_proba_folds = trial_entry['y_pred_proba_cv']  
        
        for fold_idx in range(len(y_true_folds)):
            y_true = y_true_folds[fold_idx]
            y_pred_proba = y_pred_proba_folds[fold_idx]
            
            fold_cmap = average_precision_score(y_true, y_pred_proba, average='macro')
            row = {'Strategy': 'Augmented', 'cmAP': fold_cmap}
            for idx, name in enumerate(label_names):
                row[f'AP_{name}'] = average_precision_score(y_true[:, idx], y_pred_proba[:, idx])
            rows.append(row)
                    
    df = pd.DataFrame(rows)
    
    # Updated explicit order containing all 5 validation strategies
    strategy_order = ['Baseline', 'Class Weights', 'Focal Loss', 'Oversampled', 'Augmented']
    df = df[df['Strategy'].isin(strategy_order)]
    
    # --- Step 3: Establish Layout Structure (2 Rows x 3 Columns) ---
    sns.set_context("paper", font_scale=1.1)
    sns.set_style("whitegrid", {"grid.linestyle": "--", "grid.alpha": 0.5})
    
    # Added a matching complementary 5th color tone for Data Augmentation
    custom_palette = ['#8BBCE6', '#9EE4A5', '#F8B5C5', '#FAD390', '#D1B3F1']
    strategy_colors = {strategy_order[i]: custom_palette[i] for i in range(len(strategy_order))}
    
    row1_targets = [(f'AP_{label_names[0]}', label_names[0]), 
                    (f'AP_{label_names[1]}', label_names[1]), 
                    (f'AP_{label_names[2]}', label_names[2])]
    
    row2_targets = [(f'AP_{label_names[3]}', label_names[3]), 
                    (f'AP_{label_names[4]}', label_names[4]), 
                    ('cmAP', r'$cmAP$')]
    
    grid_targets = [row1_targets, row2_targets]
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=False)

    # --- Step 4: Run Visualization Iteration Loop ---
    for r_idx in range(2):
        for c_idx in range(3):
            ax = axes[r_idx, c_idx]
            col_name, display_title = grid_targets[r_idx][c_idx]
            
            sns.boxplot(
                data=df, x='Strategy', y=col_name, hue='Strategy',
                order=strategy_order, palette=strategy_colors,
                ax=ax, width=0.6, showfliers=True, legend=False,
                flierprops=dict(marker='o', markerfacecolor='gray', markersize=4, markeredgecolor='none', alpha=0.6),
                boxprops=dict(edgecolor='#4d4d4d', linewidth=1.2),
                whiskerprops=dict(color='#4d4d4d', linewidth=1.1),
                capprops=dict(color='#4d4d4d', linewidth=1.1),
                medianprops=dict(color='#2c3e50', linewidth=1.5)
            )
            
            # --- Step 5: Refine Aesthetics, Limits, and Label Padding ---
            ax.set_title(display_title, fontsize=12, fontweight='bold', pad=14)
            ax.set_xlabel("", fontsize=10)
            ax.set_ylabel("Average Precision (AP)" if c_idx == 0 else "", fontsize=12)
            
            # Formatting labels nicely onto two lines to maintain breathing room
            display_labels = ["Baseline\n(BCE)", "Class\nWeights", "Focal\nLoss", "Oversample", "Data\nAugmented"]
            ax.set_xticklabels(display_labels, rotation=0, ha='center', fontsize=13)
            ax.tick_params(axis='y', labelsize=14)
            
            # Auto-scale axis limits tightly based on data distribution boundaries
            current_data = df[col_name].dropna()
            if not current_data.empty:
                ymin, ymax = current_data.min(), current_data.max()
                yrange = ymax - ymin if ymax != ymin else 0.1
                ax.set_ylim(max(0, ymin - 0.08 * yrange), min(1.02, ymax + 0.08 * yrange))
                
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.42, wspace=0.22, bottom=0.1)
    plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import average_precision_score

def plot_abmil_vs_perch_boxplots(abmil_results, per2_results, label_names=None):
    """
    Plots a 2x3 grid of boxplots comparing Tuned ABMIL vs Perch 2.0 + Logistic Regression.
    Row 1: Type A, Type B, Type C AP
    Row 2: Type D, Echo, and overall cmAP
    abmil_results: List of dictionaries containing 'y_true_cv' and 'y_pred_proba_cv' for ABMIL trials.
    per2_results: List of dictionaries containing 'model', 'y_true_cv', and 'y_pred_proba_cv' for Perch 2.0 trials.
    """
    if label_names is None:
        label_names = ['Type A', 'Type B', 'Type C', 'Type D', 'Echo']
        
    rows = []
    
    # --- Step 1: Parse ABMIL Results ---
    for trial_entry in abmil_results:
        y_true_folds = trial_entry['y_true_cv']          
        y_pred_proba_folds = trial_entry['y_pred_proba_cv']  
        
        for fold_idx in range(len(y_true_folds)):
            y_true = np.array(y_true_folds[fold_idx])
            y_pred_proba = np.array(y_pred_proba_folds[fold_idx])
            
            fold_cmap = average_precision_score(y_true, y_pred_proba, average='macro')
            row = {'Framework': 'ABMIL', 'cmAP': fold_cmap}
            for idx, name in enumerate(label_names):
                row[f'AP_{name}'] = average_precision_score(y_true[:, idx], y_pred_proba[:, idx])
            rows.append(row)
            
    # --- Step 2: Parse Perch 2.0 Logistic Regression Results ---
    # Extract only Logistic Regression entries from your Perch dictionary storage
    perch_lr_entries = [entry for entry in per2_results if entry.get('model') == 'Logistic Regression']
    
    for trial_entry in perch_lr_entries:
        y_true_folds = trial_entry['y_true_cv']          
        y_pred_proba_folds = trial_entry['y_pred_proba_cv']  
        
        for fold_idx in range(len(y_true_folds)):
            y_true = np.array(y_true_folds[fold_idx])
            y_pred_proba = np.array(y_pred_proba_folds[fold_idx])
            
            fold_cmap = average_precision_score(y_true, y_pred_proba, average='macro')
            row = {'Framework': 'Logistic Regression', 'cmAP': fold_cmap}
            for idx, name in enumerate(label_names):
                row[f'AP_{name}'] = average_precision_score(y_true[:, idx], y_pred_proba[:, idx])
            rows.append(row)
                    
    df = pd.DataFrame(rows)
    framework_order = ['Logistic Regression', 'ABMIL']
    
    # --- Step 3: Aesthetic Grid Setup ---
    sns.set_context("paper", font_scale=1.1)
    sns.set_style("whitegrid", {"grid.linestyle": "--", "grid.alpha": 0.5})
    
    # Professional, highly-distinguishable contrasting dual-palette
    framework_colors = {'Logistic Regression': '#8BBCE6', 'ABMIL': '#FAD390'}
    
    row1_targets = [(f'AP_{label_names[0]}', label_names[0]), 
                    (f'AP_{label_names[1]}', label_names[1]), 
                    (f'AP_{label_names[2]}', label_names[2])]
    
    row2_targets = [(f'AP_{label_names[3]}', label_names[3]), 
                    (f'AP_{label_names[4]}', label_names[4]), 
                    ('cmAP', r'$cmAP$')]
    
    grid_targets = [row1_targets, row2_targets]
    fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharey=False)

    # --- Step 4: Iteration Plot Loop ---
    for r_idx in range(2):
        for c_idx in range(3):
            ax = axes[r_idx, c_idx]
            col_name, display_title = grid_targets[r_idx][c_idx]
            
            sns.boxplot(
                data=df, x='Framework', y=col_name, hue='Framework',
                order=framework_order, palette=framework_colors,
                ax=ax, width=0.4, showfliers=True, legend=False,
                flierprops=dict(marker='o', markerfacecolor='gray', markersize=4, markeredgecolor='none', alpha=0.6),
                boxprops=dict(edgecolor='#4d4d4d', linewidth=1.2),
                whiskerprops=dict(color='#4d4d4d', linewidth=1.1),
                capprops=dict(color='#4d4d4d', linewidth=1.1),
                medianprops=dict(color='#2c3e50', linewidth=1.5)
            )
            
            ax.set_title(display_title, fontsize=15, fontweight='bold', pad=12)
            ax.set_xlabel("", fontsize=15)
            ax.set_ylabel("Average Precision (AP)" if c_idx == 0 else "", fontsize=14)
            
            # Formatted labels on single lines since we only have two groups now
            ax.set_xticklabels(framework_order, fontsize=14, fontweight='bold')
            ax.tick_params(axis='y', labelsize=14)
            # Dynamic boundaries fitting the specific distributions
            current_data = df[col_name].dropna()
            if not current_data.empty:
                ymin, ymax = current_data.min(), current_data.max()
                yrange = ymax - ymin if ymax != ymin else 0.1
                ax.set_ylim(max(0, ymin - 0.1 * yrange), min(1.02, ymax + 0.1 * yrange))
                
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.35, wspace=0.22, bottom=0.1)
    plt.show()

def plot_comprehensive_calibration(encoder_results, label_names, n_bins=10, strategy="uniform"):
    """
    Plots a multi-model, multi-encoder calibration comparison grid.
    Each column represents a target label (Class).
    
    Parameters
    ----------
    encoder_results : dict
        A nested dictionary structured as:
        {
            "EffNet": [list of trial dicts containing 'model', 'oof_y_true', 'oof_y_pred_proba'],
            "Perch 2.0": [list of trial dicts ...]
        }
    label_names : list[str]
        Names of the target classes (e.g., ['Type A', 'Type B', 'Type C', 'Type D', 'Echolocation'])
    """
    sns.set_context("paper")
    sns.set_style("whitegrid")
    
    n_labels = len(label_names)
    
    # 1 Row for Calibration Curves, 1 Row for Histograms
    fig, axes = plt.subplots(2, n_labels, figsize=(4.5 * n_labels, 9))
    
    # Generate distinct, beautiful color palettes dynamically
    all_combinations = []
    for encoder_name in encoder_results.keys():
        # Look inside the first trial to check available classifier models
        models = sorted(list(set([r['model'] for r in encoder_results[encoder_name]])))
        for model in models:
            all_combinations.append((encoder_name, model))
            
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_combinations)))
    combo_colors = {combo: colors[idx] for idx, combo in enumerate(all_combinations)}
    
    legend_handles = {}

    # Iterate through each column (Each target class label)
    for class_idx, label_name in enumerate(label_names):
        ax_curve = axes[0, class_idx]
        ax_hist = axes[1, class_idx]
        
        # Perfect calibration reference line
        ax_curve.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.7, label='Perfect Calibration')
        
        for encoder_name, trial_list in encoder_results.items():
            models = sorted(list(set([r['model'] for r in trial_list])))
            
            for model_name in models:
                # Gather all independent trials matching this specific combination
                model_trials = [r for r in trial_list if r['model'] == model_name]
                
                trial_prob_true = []
                # Use standard uniform bin coordinates to ensure clean alignment during averaging
                common_bins = np.linspace(0, 1, n_bins)
                bin_centers = (common_bins[:-1] + common_bins[1:]) / 2
                
                # Accumulator for historical distribution checks
                all_pred_probas = []
                
                for trial in model_trials:
                    y_true = trial['oof_y_true'][:, class_idx]
                    y_prob = trial['oof_y_pred_proba'][:, class_idx]
                    all_pred_probas.extend(y_prob)
                    
                    # Calculate structural curve coordinates for this specific trial
                    p_true, p_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy=strategy)
                    
                    # Interp forces coordinates to map cleanly onto a shared x-axis layout for averaging
                    interp_true = np.interp(bin_centers, p_pred, p_true, left=np.nan, right=np.nan)
                    trial_prob_true.append(interp_true)
                
                # Average across Dimension 3 (Trials) safely ignoring any empty boundary bins
                mean_prob_true = np.nanmean(trial_prob_true, axis=0)
                
                # Plot setup
                combo_key = (encoder_name, model_name)
                current_color = combo_colors[combo_key]
                display_label = f"{encoder_name} - {model_name}"
                
                # --- Plot Calibration Curves ---
                # Mask out NaN bins where no predictions landed during testing
                valid_mask = ~np.isnan(mean_prob_true)
                line, = ax_curve.plot(bin_centers[valid_mask], mean_prob_true[valid_mask], 
                                      marker='o', markersize=4, linewidth=2, 
                                      color=current_color, alpha=0.85)
                
                if display_label not in legend_handles:
                    legend_handles[display_label] = line
                
                # --- Plot Density Histograms (Step/Outline styles keep multi-lines legible) ---
                ax_hist.hist(all_pred_probas, bins=common_bins, histtype='step', 
                             linewidth=1.5, color=current_color, alpha=0.75)
        
        # Formatting Top Subplot Row
        ax_curve.set_title(f"Calibration: {label_name}", fontsize=14, fontweight='bold')
        ax_curve.set_xlabel("Mean Predicted Probability", fontsize=10)
        ax_curve.set_ylabel("Fraction of Positives", fontsize=10)
        ax_curve.set_xlim(0, 1)
        ax_curve.set_ylim(0, 1)
        
        # Formatting Bottom Subplot Row
        ax_hist.set_title(f"Distribution: {label_name}", fontsize=14, fontweight='bold')
        ax_hist.set_xlabel("Predicted Probability", fontsize=10)
        ax_hist.set_ylabel("Density / Sample Count", fontsize=10)
        ax_hist.set_xlim(0, 1)
        ax_hist.set_yscale('log') # Log scale helps check minor boundaries when predictions stack at 0 or 1

    # Place a single unified legend block neatly below the chart grid
    fig.legend(legend_handles.values(), legend_handles.keys(), loc='lower center', 
               ncol=min(4, len(legend_handles)), bbox_to_anchor=(0.5, -0.06), 
               fontsize=11, frameon=True)
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

def plot_comprehensive_calibration2(linear_results, abmil_results, label_names, n_bins=10, strategy="uniform"):
    """
    Plots a multi-model calibration comparison grid of the a certains encoder linear results and abmil results.
    Columns track individual targets; top row renders curves, bottom row shows distribution density.
    linear_results = list of dicts with 'model', 'oof_y_true', 'oof_y_pred_proba' for linear models
    abmil_results = list of dicts with 'model', 'oof_y_true', 'oof_y_pred_proba' for ABMIL models
    """
    sns.set_context("paper")
    sns.set_style("whitegrid")
    
    n_labels = len(label_names)
    fig, axes = plt.subplots(2, n_labels, figsize=(4.5 * n_labels, 8.5))
    
    # Consolidate results and purge the non-competitive dummy strategy
    combined_results = [r for r in (linear_results + abmil_results) if r['model'] != 'Prevalence guesser']
    
    # Establish deterministic mapping for plot ordering and coloring
    #models = ['Logistic Regression', 'SVM', 'Random Forest', 'MLP', 'ABMIL']
    models = [ 'ABMIL']
    models = [m for m in models if m in set(r['model'] for r in combined_results)]
    
    # Distinct publication-ready hex palette mapping
    model_colors = {
        'Logistic Regression': '#1f77b4',
        'SVM': '#ff7f0e',
        'Random Forest': '#2ca02c',
        'MLP': '#9467bd',
        'ABMIL': '#d62728'
    }
    
    legend_handles = {}

    for class_idx, label_name in enumerate(label_names):
        ax_curve = axes[0, class_idx] if n_labels > 1 else axes[0]
        ax_hist = axes[1, class_idx] if n_labels > 1 else axes[1]
        
        # Diagonal target line
        ax_curve.plot([0, 1], [0, 1], linestyle='--', color='#7f7f7f', alpha=0.8, label='Perfect Calibration')
        
        for model_name in models:
            model_trials = [r for r in combined_results if r['model'] == model_name]
            if not model_trials:
                continue
                
            trial_prob_true = []
            common_bins = np.linspace(0, 1, n_bins)
            bin_centers = (common_bins[:-1] + common_bins[1:]) / 2
            all_pred_probas = []
            
            for trial in model_trials:
                y_true = trial['oof_y_true'][:, class_idx]
                y_prob = trial['oof_y_pred_proba'][:, class_idx]
                all_pred_probas.extend(y_prob)
                
                # Extract curve coords using OOF blocks
                p_true, p_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy=strategy)
                
                # Resample cleanly onto consistent bin alignment coordinates
                interp_true = np.interp(bin_centers, p_pred, p_true, left=np.nan, right=np.nan)
                trial_prob_true.append(interp_true)
            
            # Aggregate trial performance vectors safely ignoring unreached prediction bins
            mean_prob_true = np.nanmean(trial_prob_true, axis=0)
            current_color = model_colors.get(model_name, '#333333')
            
            # --- Draw Reliability Curves ---
            valid_mask = ~np.isnan(mean_prob_true)
            line, = ax_curve.plot(
                bin_centers[valid_mask], mean_prob_true[valid_mask], 
                marker='o', markersize=4, linewidth=2, 
                color=current_color, alpha=0.85
            )
            
            if model_name not in legend_handles:
                legend_handles[model_name] = line
            
            # --- Draw Density Histograms ---
            ax_hist.hist(
                all_pred_probas, bins=common_bins, histtype='step', 
                linewidth=1.5, color=current_color, alpha=0.8
            )
        
        # Subplot Layout and Design Polish
        ax_curve.set_title(f"Calibration: {label_name}", fontsize=12, fontweight='bold', pad=10)
        ax_curve.set_xlabel("Mean Predicted Probability", fontsize=12)
        ax_curve.set_ylabel("Fraction of Positives", fontsize=12)
        ax_curve.set_xlim(0, 1)
        ax_curve.set_ylim(0, 1)
        ax_curve.tick_params(axis='y', labelsize=14)
        ax_hist.tick_params(axis='x', labelsize=14)
        
        ax_hist.set_title(f"Distribution: {label_name}", fontsize=12, fontweight='bold', pad=10)
        ax_hist.set_xlabel("Predicted Probability", fontsize=12)
        ax_hist.set_ylabel("Log Sample Density", fontsize=12)
        ax_hist.set_xlim(0, 1)
        ax_hist.set_yscale('log')
        ax_hist.tick_params(axis='y', labelsize=14)
        ax_hist.tick_params(axis='x', labelsize=14)

    # Centralize universal shared legend mapping
    fig.legend(
        legend_handles.values(), legend_handles.keys(), loc='lower center', 
        ncol=len(legend_handles), bbox_to_anchor=(0.5, -0.05), 
        fontsize=10, frameon=True
    )
    
    plt.tight_layout(rect=[0, 0.02, 1, 1])
    plt.show()

def label_confusion(y_true,y_pred_proba,y_pred_binary=None, label_names=None, threshold=0.5) :
    """
    Analyzes which labels are predicted 'instead' of the true labels.
    """
    # 1. Convert proba to binary if binary isn't provided
    if y_pred_binary is None:
        y_pred_binary = (y_pred_proba >= threshold).astype(int)
    
    y_true = np.array(y_true)
    y_pred_binary = np.array(y_pred_binary)
    num_labels = y_true.shape[1]
    
    if label_names is None:
        label_names = [f"Label_{i}" for i in range(num_labels)]

    # 2. Initialize Confusion Matrix
    # Rows: The label that was SHOULD have been there (False Negative)
    # Cols: The label that was predicted WRONGLY (False Positive)
    confusion_mtx = np.zeros((num_labels, num_labels))

    # 3. Iterate through samples
    for i in range(len(y_true)):
        actual = y_true[i]
        pred = y_pred_binary[i]

        # Indices of missed labels (FN)
        missed = np.where((actual == 1) & (pred == 0))[0]
        # Indices of extra labels (FP)
        extra = np.where((actual == 0) & (pred == 1))[0]

        # If we missed something AND predicted something else wrongly
        for m_idx in missed:
            for e_idx in extra:
                confusion_mtx[m_idx, e_idx] += 1

    # 4. Wrap in DataFrame for easy viewing
    df_cm = pd.DataFrame(confusion_mtx, index=label_names, columns=label_names)
    
    return df_cm

def plot_loss_curves(all_results):
    """
    Plots the training and validation loss pathways side-by-side across configurations
    to visually demonstrate overfitting trends.
    all_results : list of dicts, each containing:
    {
        'model': str,  # e.g., 'MLP_FocalLoss'
        'train_histories': list of lists (each inner list is the train loss per epoch for a fold),
        'val_histories': list of lists (each inner list is the val loss per epoch for a fold)
    }
    """
    unique_models = list(set([res['model'] for res in all_results]))
    
    fig, axes = plt.subplots(1, len(unique_models), figsize=(6 * len(unique_models), 5), sharey=False)
    if len(unique_models) == 1:
        axes = [axes]
        
    for idx, model_name in enumerate(sorted(unique_models)):
        # Extract matches across all trials for this model variant
        model_entries = [res for res in all_results if res['model'] == model_name]
        
        # Aggregate arrays across folds and trials
        all_train = []
        all_val = []
        for entry in model_entries:
            all_train.extend(entry['train_histories'])
            all_val.extend(entry['val_histories'])
            
        # Convert lists into matrix footprints [Folds, Epochs]
        train_matrix = np.array(all_train)
        val_matrix = np.array(all_val)
        epochs = np.arange(1, train_matrix.shape[1] + 1)
        
        # Calculate mean loss curves
        mean_train = np.mean(train_matrix, axis=0)
        mean_val = np.mean(val_matrix, axis=0)
        
        # Standard deviation bounds for visual error shading
        std_train = np.std(train_matrix, axis=0)
        std_val = np.std(val_matrix, axis=0)
        
        ax = axes[idx]
        
        # Plot mean trends
        ax.plot(epochs, mean_train, label='Train Loss', color='royalblue', linewidth=2)
        ax.plot(epochs, mean_val, label='Val Loss', color='crimson', linewidth=2, linestyle='--')
        
        # Add error bounds shading
        ax.fill_between(epochs, mean_train - std_train, mean_train + std_train, color='royalblue', alpha=0.15)
        ax.fill_between(epochs, mean_val - std_val, mean_val + std_val, color='crimson', alpha=0.1)
        
        ax.set_title(f"{model_name}\n(Mean OOF Progression)", fontsize=13, fontweight='bold')
        ax.set_xlabel("Epochs", fontsize=11)
        if idx == 0:
            ax.set_ylabel("Loss Magnitude", fontsize=11)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper right')
        
    plt.tight_layout()
    plt.show()
