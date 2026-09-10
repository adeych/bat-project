"""
Functions to process raw cross-validation outputs into latex summary tables
"""

import numpy as np
from sklearn.metrics import average_precision_score
from src.metrics import calculate_ece
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

def process_folds_to_table(balancing_results, augmented_results):
    """
    Extracts scores from all 25 individual validation folds across the 5 trials,
    computes per-class AP scores per fold, and calculates the overall mean and SEM.
    """
    header_cols = ['Type A AP', 'Type B AP', 'Type C AP', 'Type D AP', 'Echo AP', 'cmAP']
    
    # Map your execution row labels to the raw outputs and internal dict names
    mapping = {
        'Baseline (BCE)':         (balancing_results, 'MLP_Baseline'),
        'Class Weights':          (balancing_results, 'MLP_ClassWeights'),
        'Focal Loss':             (balancing_results, 'MLP_FocalLoss'),
        'Iterative Oversampling': (balancing_results, 'MLP_Oversampled'),
        'Data Augmentation':      (augmented_results, 'MLP_Baseline')
    }
    
    formatted_results = {}
    
    for row_name, (results_list, model_key) in mapping.items():
        # Filter down to the matching model configuration
        model_entries = [entry for entry in results_list if entry['model'] == model_key]
        
        if not model_entries:
            print(f"Warning: No entries found for strategy '{row_name}' (key: '{model_key}')")
            continue
            
        # Lists to store metrics for all 25 individual folds
        fold_metrics = {col: [] for col in header_cols}
        
        for trial_entry in model_entries:
            # y_true_cv and y_pred_proba_cv are lists containing the arrays for each fold
            y_true_folds = trial_entry['y_true_cv']
            y_pred_folds = trial_entry['y_pred_proba_cv']
            
            # Loop through each fold in this trial (e.g., 5 folds per trial)
            for fold_true, fold_pred in zip(y_true_folds, y_pred_folds):
                
                # Calculate AP for each of your 5 classes for THIS SPECIFIC FOLD
                class_aps = []
                for class_idx in range(5):
                    # Edge case check: if a class happens to have 0 positive samples in a test fold,
                    # handle it gracefully to avoid sklearn exceptions.
                    if np.sum(fold_true[:, class_idx]) == 0:
                        ap = 0.0
                    else:
                        ap = average_precision_score(fold_true[:, class_idx], fold_pred[:, class_idx])
                    class_aps.append(ap)
                
                # Compute the macro mean (cmAP) for this individual fold
                fold_cmap = np.mean(class_aps)
                
                # Append metrics to our tracking array lists
                fold_metrics['Type A AP'].append(class_aps[0])
                fold_metrics['Type B AP'].append(class_aps[1])
                fold_metrics['Type C AP'].append(class_aps[2])
                fold_metrics['Type D AP'].append(class_aps[3])
                fold_metrics['Echo AP'].append(class_aps[4])
                fold_metrics['cmAP'].append(fold_cmap)
                
        # Calculate final stats across all collected folds (e.g., 25 folds total)
        formatted_results[row_name] = {}
        total_folds_collected = len(fold_metrics['cmAP'])
        
        for col in header_cols:
            all_fold_scores = np.array(fold_metrics[col])
            mean_val = np.mean(all_fold_scores)
            
            # Standard Error of the Mean (SEM) = sample standard deviation / sqrt(N)
            std_err = np.std(all_fold_scores, ddof=1) / np.sqrt(total_folds_collected) if total_folds_collected > 1 else 0.0
            
            formatted_results[row_name][col] = {
                'mean': mean_val,
                'std': std_err
            }
            
    # Return the clean LaTeX table generator string
    return generate_balancing_latex_table(formatted_results)


def generate_balancing_latex_table(results_dict):
    """
    Generates a LaTeX table for balancing techniques,
    automatically bolding the top-performing method for each metric column.
    """
    header_cols = ['Type A AP', 'Type B AP', 'Type C AP', 'Type D AP', 'Echo AP', 'cmAP']
    strategies = [
        'Baseline (BCE)',
        'Class Weights',
        'Focal Loss',
        'Iterative Oversampling',
        'Data Augmentation'
    ]
    
    max_means = {col: -1.0 for col in header_cols}
    for strat in strategies:
        if strat in results_dict:
            for col in header_cols:
                if results_dict[strat][col]['mean'] > max_means[col]:
                    max_means[col] = results_dict[strat][col]['mean']

    latex_str = []
    latex_str.append(r"\begin{table}[htbp]")
    latex_str.append(r"\centering")
    latex_str.append(r"\caption{Performance comparison of class-balancing and data augmentation techniques using Perch embeddings. Metrics report the mean ($\pm$ standard error of the mean) computed over 25 validation folds, with top values highlighted in bold.}")
    latex_str.append(r"\label{tab:balancing_results}")
    latex_str.append(r"\small")
    latex_str.append(r"\begin{tabular}{lcccccc}")
    latex_str.append(r"\toprule")
    latex_str.append(r"\textbf{Strategy} & \textbf{Type A AP} & \textbf{Type B AP} & \textbf{Type C AP} & \textbf{Type D AP} & \textbf{Echo AP} & \textbf{cmAP} \\")
    latex_str.append(r"\midrule")
    
    for strat in strategies:
        if strat not in results_dict:
            continue
            
        row_data = results_dict[strat]
        row_str = f"{strat}"
        
        for col in header_cols:
            mean = row_data[col]['mean']
            std  = row_data[col]['std']
            val_str = f"{mean:.3f} $\\pm$ {std:.3f}"
            
            if np.isclose(mean, max_means[col]):
                row_str += f" & \\textbf{{{val_str}}}"
            else:
                row_str += f" & {val_str}"
                
        row_str += r" \\"
        latex_str.append(row_str)
        
        if strat == 'Baseline (BCE)' or strat == 'Focal Loss':
            latex_str.append(r"\midrule")
            
    if latex_str[-1] == r"\midrule":
        latex_str.pop()
        
    latex_str.append(r"\bottomrule")
    latex_str.append(r"\end{tabular}")
    latex_str.append(r"\end{table}")
    
    return "\n".join(latex_str)

def process_encoder_folds_to_table(all_results_dict):
    """
    Processes the raw cross-validation outputs for the Encoder comparison,
    calculates per-class AP scores per fold across all 25 folds, 
    and calculates the overall mean and SEM.
    """
    header_cols = ['Type A AP', 'Type B AP', 'Type C AP', 'Type D AP', 'Echo AP', 'cmAP']
    encoders = ['ESP-EffNetB0', 'NatureLM-BEATs', 'Perch 2.0']
    classifiers = ['SVM', 'Logistic Regression', 'MLP', 'Random Forest']
    
    formatted_results = {}

    # --- 1. Process Chance-Level Baseline ('Prevalence guesser') ---
    # Look inside the 'ESP-EffNetB0' list to find entries where 'model' == 'Prevalence guesser'
    if 'ESP-EffNetB0' in all_results_dict:
        chance_entries = [entry for entry in all_results_dict['ESP-EffNetB0'] if entry.get('model') == 'Prevalence guesser']
        
        if chance_entries:
            formatted_results['Chance-Level'] = {}
            fold_metrics = {col: [] for col in header_cols}
            
            for trial_entry in chance_entries:
                y_true_folds = trial_entry['y_true_cv']
                y_pred_folds = trial_entry['y_pred_proba_cv']
                
                for fold_true, fold_pred in zip(y_true_folds, y_pred_folds):
                    class_aps = []
                    for class_idx in range(5):
                        if np.sum(fold_true[:, class_idx]) == 0:
                            ap = 0.0
                        else:
                            ap = average_precision_score(fold_true[:, class_idx], fold_pred[:, class_idx])
                        class_aps.append(ap)
                    
                    fold_cmap = np.mean(class_aps)
                    fold_metrics['Type A AP'].append(class_aps[0])
                    fold_metrics['Type B AP'].append(class_aps[1])
                    fold_metrics['Type C AP'].append(class_aps[2])
                    fold_metrics['Type D AP'].append(class_aps[3])
                    fold_metrics['Echo AP'].append(class_aps[4])
                    fold_metrics['cmAP'].append(fold_cmap)
                    
            total_folds_collected = len(fold_metrics['cmAP'])
            formatted_results['Chance-Level']['Chance'] = {}
            for col in header_cols:
                all_fold_scores = np.array(fold_metrics[col])
                mean_val = np.mean(all_fold_scores)
                std_err = np.std(all_fold_scores, ddof=1) / np.sqrt(total_folds_collected) if total_folds_collected > 1 else 0.0
                formatted_results['Chance-Level']['Chance'][col] = {'mean': mean_val, 'std': std_err}

    # --- 2. Process Deep Learning Encoders ---
    for enc in encoders:
        if enc not in all_results_dict:
            continue
            
        formatted_results[enc] = {}
        results_list = all_results_dict[enc]
        
        for clf in classifiers:
            model_entries = [entry for entry in results_list if entry.get('model') == clf]
            if not model_entries:
                continue
                
            fold_metrics = {col: [] for col in header_cols}
            
            for trial_entry in model_entries:
                y_true_folds = trial_entry['y_true_cv']
                y_pred_folds = trial_entry['y_pred_proba_cv']
                
                for fold_true, fold_pred in zip(y_true_folds, y_pred_folds):
                    class_aps = []
                    for class_idx in range(5):
                        if np.sum(fold_true[:, class_idx]) == 0:
                            ap = 0.0
                        else:
                            ap = average_precision_score(fold_true[:, class_idx], fold_pred[:, class_idx])
                        class_aps.append(ap)
                    
                    fold_cmap = np.mean(class_aps)
                    fold_metrics['Type A AP'].append(class_aps[0])
                    fold_metrics['Type B AP'].append(class_aps[1])
                    fold_metrics['Type C AP'].append(class_aps[2])
                    fold_metrics['Type D AP'].append(class_aps[3])
                    fold_metrics['Echo AP'].append(class_aps[4])
                    fold_metrics['cmAP'].append(fold_cmap)
            
            total_folds_collected = len(fold_metrics['cmAP'])
            formatted_results[enc][clf] = {}
            
            for col in header_cols:
                all_fold_scores = np.array(fold_metrics[col])
                mean_val = np.mean(all_fold_scores)
                std_err = np.std(all_fold_scores, ddof=1) / np.sqrt(total_folds_collected) if total_folds_collected > 1 else 0.0
                formatted_results[enc][clf][col] = {'mean': mean_val, 'std': std_err}
                
    return generate_encoder_latex_table(formatted_results)


def generate_encoder_latex_table(results_dict):
    """
    Generates a LaTeX table including a Chance-Level baseline at the top,
    grouping classifiers under their parent Encoders using multirow.
    """
    header_cols = ['Type A AP', 'Type B AP', 'Type C AP', 'Type D AP', 'Echo AP', 'cmAP']
    encoders = ['ESP-EffNetB0', 'NatureLM-BEATs', 'Perch 2.0']
    classifiers = ['SVM', 'Logistic Regression', 'MLP', 'Random Forest']
    
    max_means = {col: -1.0 for col in header_cols}
    for enc in encoders:
        if enc in results_dict:
            for clf in classifiers:
                if clf in results_dict[enc]:
                    for col in header_cols:
                        if results_dict[enc][clf][col]['mean'] > max_means[col]:
                            max_means[col] = results_dict[enc][clf][col]['mean']

    latex_str = []
    latex_str.append(r"\begin{table}[htbp]")
    latex_str.append(r"\centering")
    latex_str.append(r"\caption{Comparative performance evaluation across audio embedding encoders and downstream classifiers against a chance-level baseline. Metrics report the mean ($\pm$ standard error of the mean) computed over 25 validation folds, with top values highlighted in bold.}")
    latex_str.append(r"\label{tab:encoder_results}")
    latex_str.append(r"\footnotesize")
    latex_str.append(r"\def\arraystretch{1.2}")
    latex_str.append(r"\setlength{\tabcolsep}{5pt}")
    
    latex_str.append(r"\begin{tabular}{llcccccc}")
    latex_str.append(r"\toprule")
    latex_str.append(r"\textbf{Encoder} & \textbf{Classifier} & \textbf{Type A AP} & \textbf{Type B AP} & \textbf{Type C AP} & \textbf{Type D AP} & \textbf{Echo AP} & \textbf{cmAP} \\")
    latex_str.append(r"\midrule")
    
    # --- Print Baseline Row ---
    if 'Chance-Level' in results_dict:
        row_data = results_dict['Chance-Level']['Chance']
        row_str = r"\textit{Baseline} & Prevalence Guesser"
        for col in header_cols:
            mean = row_data[col]['mean']
            std  = row_data[col]['std']
            row_str += f" & {mean:.3f} $\\pm$ {std:.3f}"
        row_str += r" \\"
        latex_str.append(row_str)
        latex_str.append(r"\midrule")
    
    # --- Print Deep Learning Architecture Rows ---
    for enc in encoders:
        if enc not in results_dict:
            continue
            
        for idx, clf in enumerate(classifiers):
            if clf not in results_dict[enc]:
                continue
                
            row_data = results_dict[enc][clf]
            
            if idx == 0:
                row_str = f"\\multirow{{4}}{{*}}{{\\textbf{{{enc}}}}} & {clf}"
            else:
                row_str = f" & {clf}"
                
            for col in header_cols:
                mean = row_data[col]['mean']
                std  = row_data[col]['std']
                val_str = f"{mean:.3f} $\\pm$ {std:.3f}"
                
                if np.isclose(mean, max_means[col]):
                    row_str += f" & \\textbf{{{val_str}}}"
                else:
                    row_str += f" & {val_str}"
                    
            row_str += r" \\"
            latex_str.append(row_str)
            
        latex_str.append(r"\midrule")
        
    if latex_str[-1] == r"\midrule":
        latex_str.pop()
        
    latex_str.append(r"\bottomrule")
    latex_str.append(r"\end{tabular}")
    latex_str.append(r"\end{table}")
    
    return "\n".join(latex_str)


def process_encoder_folds_to_table_no_echo(all_results_dict):
    """
    Processes the raw cross-validation outputs, calculating scores for 
    the 4 main call types (excluding Echo).
    """
    # Updated: Removed 'Echo AP'
    header_cols = ['Type A AP', 'Type B AP', 'Type C AP', 'Type D AP', 'cmAP']
    encoders = ['ESP-EffNetB0', 'NatureLM-BEATs', 'Perch 2.0']
    classifiers = ['SVM', 'Logistic Regression', 'MLP', 'Random Forest']
    
    formatted_results = {}

    # --- 1. Process Chance-Level Baseline ---
    if 'ESP-EffNetB0' in all_results_dict:
        chance_entries = [entry for entry in all_results_dict['ESP-EffNetB0'] if entry.get('model') == 'Prevalence guesser']
        
        if chance_entries:
            formatted_results['Chance-Level'] = {}
            fold_metrics = {col: [] for col in header_cols}
            
            for trial_entry in chance_entries:
                y_true_folds = trial_entry['y_true_cv']
                y_pred_folds = trial_entry['y_pred_proba_cv']
                
                for fold_true, fold_pred in zip(y_true_folds, y_pred_folds):
                    class_aps = []
                    # Updated: range(4) to only process Types A-D
                    for class_idx in range(4):
                        if np.sum(fold_true[:, class_idx]) == 0:
                            ap = 0.0
                        else:
                            ap = average_precision_score(fold_true[:, class_idx], fold_pred[:, class_idx])
                        class_aps.append(ap)
                    
                    fold_cmap = np.mean(class_aps)
                    fold_metrics['Type A AP'].append(class_aps[0])
                    fold_metrics['Type B AP'].append(class_aps[1])
                    fold_metrics['Type C AP'].append(class_aps[2])
                    fold_metrics['Type D AP'].append(class_aps[3])
                    fold_metrics['cmAP'].append(fold_cmap)
                    
            total_folds_collected = len(fold_metrics['cmAP'])
            formatted_results['Chance-Level']['Chance'] = {}
            for col in header_cols:
                all_fold_scores = np.array(fold_metrics[col])
                mean_val = np.mean(all_fold_scores)
                std_err = np.std(all_fold_scores, ddof=1) / np.sqrt(total_folds_collected) if total_folds_collected > 1 else 0.0
                formatted_results['Chance-Level']['Chance'][col] = {'mean': mean_val, 'std': std_err}

    # --- 2. Process Deep Learning Encoders ---
    for enc in encoders:
        if enc not in all_results_dict: continue
            
        formatted_results[enc] = {}
        results_list = all_results_dict[enc]
        
        for clf in classifiers:
            model_entries = [entry for entry in results_list if entry.get('model') == clf]
            if not model_entries: continue
                
            fold_metrics = {col: [] for col in header_cols}
            
            for trial_entry in model_entries:
                y_true_folds = trial_entry['y_true_cv']
                y_pred_folds = trial_entry['y_pred_proba_cv']
                
                for fold_true, fold_pred in zip(y_true_folds, y_pred_folds):
                    class_aps = []
                    # Updated: range(4) to only process Types A-D
                    for class_idx in range(4):
                        if np.sum(fold_true[:, class_idx]) == 0:
                            ap = 0.0
                        else:
                            ap = average_precision_score(fold_true[:, class_idx], fold_pred[:, class_idx])
                        class_aps.append(ap)
                    
                    fold_cmap = np.mean(class_aps)
                    fold_metrics['Type A AP'].append(class_aps[0])
                    fold_metrics['Type B AP'].append(class_aps[1])
                    fold_metrics['Type C AP'].append(class_aps[2])
                    fold_metrics['Type D AP'].append(class_aps[3])
                    fold_metrics['cmAP'].append(fold_cmap)
            
            total_folds_collected = len(fold_metrics['cmAP'])
            formatted_results[enc][clf] = {}
            for col in header_cols:
                all_fold_scores = np.array(fold_metrics[col])
                mean_val = np.mean(all_fold_scores)
                std_err = np.std(all_fold_scores, ddof=1) / np.sqrt(total_folds_collected) if total_folds_collected > 1 else 0.0
                formatted_results[enc][clf][col] = {'mean': mean_val, 'std': std_err}
                
    return generate_encoder_latex_table_no_echo(formatted_results)

def generate_encoder_latex_table_no_echo(results_dict):
    """
    Generates an LaTeX table excluding Echo AP,
    grouping classifiers under their parent Encoders using multirow.
    """
    # Updated: Removed 'Echo AP'
    header_cols = ['Type A AP', 'Type B AP', 'Type C AP', 'Type D AP', 'cmAP']
    encoders = ['ESP-EffNetB0', 'NatureLM-BEATs', 'Perch 2.0']
    classifiers = ['SVM', 'Logistic Regression', 'MLP', 'Random Forest']
    
    max_means = {col: -1.0 for col in header_cols}
    for enc in encoders:
        if enc in results_dict:
            for clf in classifiers:
                if clf in results_dict[enc]:
                    for col in header_cols:
                        if results_dict[enc][clf][col]['mean'] > max_means[col]:
                            max_means[col] = results_dict[enc][clf][col]['mean']

    latex_str = []
    latex_str.append(r"\begin{table}[htbp]")
    latex_str.append(r"\centering")
    latex_str.append(r"\caption{Comparative performance evaluation across audio embedding encoders and downstream classifiers against a chance-level baseline. Metrics report the mean ($\pm$ standard error of the mean) computed over 25 validation folds, with top values highlighted in bold.}")
    latex_str.append(r"\label{tab:encoder_results}")
    latex_str.append(r"\footnotesize")
    latex_str.append(r"\def\arraystretch{1.2}")
    latex_str.append(r"\setlength{\tabcolsep}{5pt}")
    
    # Updated: tabular specification from {llcccccc} to {llccccc}
    latex_str.append(r"\begin{tabular}{llccccc}")
    latex_str.append(r"\toprule")
    # Updated: header row
    latex_str.append(r"\textbf{Encoder} & \textbf{Classifier} & \textbf{Type A AP} & \textbf{Type B AP} & \textbf{Type C AP} & \textbf{Type D AP} & \textbf{cmAP} \\")
    latex_str.append(r"\midrule")
    
    # --- Print Baseline Row ---
    if 'Chance-Level' in results_dict:
        row_data = results_dict['Chance-Level']['Chance']
        row_str = r"\textit{Baseline} & Prevalence Guesser"
        for col in header_cols:
            mean = row_data[col]['mean']
            std  = row_data[col]['std']
            row_str += f" & {mean:.3f} $\\pm$ {std:.3f}"
        row_str += r" \\"
        latex_str.append(row_str)
        latex_str.append(r"\midrule")
    
    # --- Print Deep Learning Architecture Rows ---
    for enc in encoders:
        if enc not in results_dict: continue
            
        for idx, clf in enumerate(classifiers):
            if clf not in results_dict[enc]: continue
                
            row_data = results_dict[enc][clf]
            
            if idx == 0:
                row_str = f"\\multirow{{4}}{{*}}{{\\textbf{{{enc}}}}} & {clf}"
            else:
                row_str = f" & {clf}"
                
            for col in header_cols:
                mean = row_data[col]['mean']
                std  = row_data[col]['std']
                val_str = f"{mean:.3f} $\\pm$ {std:.3f}"
                
                if np.isclose(mean, max_means[col]):
                    row_str += f" & \\textbf{{{val_str}}}"
                else:
                    row_str += f" & {val_str}"
                    
            row_str += r" \\"
            latex_str.append(row_str)
            
        latex_str.append(r"\midrule")
        
    if latex_str[-1] == r"\midrule":
        latex_str.pop()
        
    latex_str.append(r"\bottomrule")
    latex_str.append(r"\end{tabular}")
    latex_str.append(r"\end{table}")
    
    return "\n".join(latex_str)

def process_abmil_vs_perch_table(abmil_results, per2_results):
    """
    Computes summary metrics (Mean \pm SEM) across evaluation cross-validation folds
    and builds a LaTeX code block for comparing abmil and base probes results
    """
    header_cols = ['Type A AP', 'Type B AP', 'Type C AP', 'Type D AP', 'Echo AP', 'cmAP']
    label_names = ['Type A', 'Type B', 'Type C', 'Type D', 'Echo']
    
    summary_metrics = {}
    
    # Storage maps
    fold_metrics_abmil = {col: [] for col in header_cols}
    fold_metrics_perch = {col: [] for col in header_cols}

    # 1. Parse ABMIL
    for trial_entry in abmil_results:
        for f_true, f_pred in zip(trial_entry['y_true_cv'], trial_entry['y_pred_proba_cv']):
            f_true, f_pred = np.array(f_true), np.array(f_pred)
            class_aps = [average_precision_score(f_true[:, c], f_pred[:, c]) for c in range(5)]
            fold_metrics_abmil['Type A AP'].append(class_aps[0])
            fold_metrics_abmil['Type B AP'].append(class_aps[1])
            fold_metrics_abmil['Type C AP'].append(class_aps[2])
            fold_metrics_abmil['Type D AP'].append(class_aps[3])
            fold_metrics_abmil['Echo AP'].append(class_aps[4])
            fold_metrics_abmil['cmAP'].append(np.mean(class_aps))

    # 2. Parse Perch
    perch_lr_entries = [entry for entry in per2_results if entry.get('model') == 'Logistic Regression']
    for trial_entry in perch_lr_entries:
        for f_true, f_pred in zip(trial_entry['y_true_cv'], trial_entry['y_pred_proba_cv']):
            f_true, f_pred = np.array(f_true), np.array(f_pred)
            class_aps = [average_precision_score(f_true[:, c], f_pred[:, c]) for c in range(5)]
            fold_metrics_perch['Type A AP'].append(class_aps[0])
            fold_metrics_perch['Type B AP'].append(class_aps[1])
            fold_metrics_perch['Type C AP'].append(class_aps[2])
            fold_metrics_perch['Type D AP'].append(class_aps[3])
            fold_metrics_perch['Echo AP'].append(class_aps[4])
            fold_metrics_perch['cmAP'].append(np.mean(class_aps))

    # Process Stats
    for name, storage in [('Tuned ABMIL', fold_metrics_abmil), ('Perch 2.0 + LR', fold_metrics_perch)]:
        summary_metrics[name] = {}
        total_folds = len(storage['cmAP'])
        for col in header_cols:
            scores = np.array(storage[col])
            mean_val = np.mean(scores)
            sem_val = np.std(scores, ddof=1) / np.sqrt(total_folds) if total_folds > 1 else 0.0
            summary_metrics[name][col] = {'mean': mean_val, 'std': sem_val}

    # Identify Maximum values for bold highlights
    max_means = {col: max(summary_metrics['Tuned ABMIL'][col]['mean'], summary_metrics['Perch 2.0 + LR'][col]['mean']) for col in header_cols}

    # Generate LaTeX String Output
    latex_str = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Performance comparison between local frame aggregated embeddings and instance attention learning (ABMIL). Metrics denote mean ($\pm$ standard error of the mean) across validation iterations.}",
        r"\label{tab:abmil_vs_perch}",
        r"\footnotesize",
        r"\def\arraystretch{1.2}",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Framework Strategy} & \textbf{Type A AP} & \textbf{Type B AP} & \textbf{Type C AP} & \textbf{Type D AP} & \textbf{Echo AP} & \textbf{cmAP} \\",
        r"\midrule"
    ]

    for model_name in ['Perch 2.0 + LR', 'Tuned ABMIL']:
        row_str = f"\\textbf{{{model_name}}}" if "ABMIL" in model_name else model_name
        for col in header_cols:
            mean = summary_metrics[model_name][col]['mean']
            std = summary_metrics[model_name][col]['std']
            val_str = f"{mean:.3f} $\\pm$ {std:.3f}"
            
            if np.isclose(mean, max_means[col]):
                row_str += f" & \\textbf{{{val_str}}}"
            else:
                row_str += f" & {val_str}"
        row_str += r" \\"
        latex_str.append(row_str)

    latex_str.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ])

    return "\n".join(latex_str)

def generate_abmil_calibration_latex(abmil_results, label_names, n_bins=10, strategy="uniform"):
    """
    Computes calibration metrics for the ABMIL model across trials and 
    returns a LaTeX booktabs table string.
    """
    # Extract only ABMIL entries
    abmil_trials = [r for r in abmil_results if r['model'] == 'ABMIL']
    if not abmil_trials:
        raise ValueError("No results found with model identifier 'ABMIL'")
        
    metrics_accumulator = []
    
    # Evaluate every trial independently to preserve variance/std-dev
    for trial in abmil_trials:
        oof_true = trial['oof_y_true']
        oof_pred = trial['oof_y_pred_proba']
        
        trial_metrics = {}
        for c_idx, class_name in enumerate(label_names):
            y_t = oof_true[:, c_idx]
            y_p = oof_pred[:, c_idx]
            
            # Clip predictions slightly to avoid infinite log-loss penalties
            y_p_clipped = np.clip(y_p, 1e-15, 1 - 1e-15)
            
            trial_metrics[f"{class_name}_Brier"] = brier_score_loss(y_t, y_p)
            trial_metrics[f"{class_name}_LogLoss"] = log_loss(y_t, y_p_clipped, labels=[0, 1])
            trial_metrics[f"{class_name}_ECE"] = calculate_ece(y_t, y_p, n_bins=n_bins, strategy=strategy)
            
        metrics_accumulator.append(trial_metrics)
        
    df_trials = pd.DataFrame(metrics_accumulator)
    
    # Calculate means and standard deviations across trials
    means = df_trials.mean()
    stds = df_trials.std(ddof=1)
    
    # Begin LaTeX Table formatting construction
    latex_lines = [
        "\\begin{table}[htbp]",
        "  \\centering",
        "  \\caption{ABMIL Calibration Metrics Across Evaluated Target Cohorts}",
        "  \\label{tab:abmil_calibration}",
        "  \\begin{tabular}{lccc}",
        "    \\toprule",
        "    \\textbf{Evaluation Cohort} & \\textbf{Brier Score} & \\textbf{Log Score} & \\textbf{ECE} \\\\",
        "    \\midrule"
    ]
    
    # Append per-class rows
    for class_name in label_names:
        b_mean, b_std = means[f"{class_name}_Brier"], stds[f"{class_name}_Brier"]
        l_mean, l_std = means[f"{class_name}_LogLoss"], stds[f"{class_name}_LogLoss"]
        e_mean, e_std = means[f"{class_name}_ECE"], stds[f"{class_name}_ECE"]
        
        row_str = (
            f"    {class_name} & "
            f"${b_mean:.4f} \\pm {b_std:.4f}$ & "
            f"${l_mean:.4f} \\pm {l_std:.4f}$ & "
            f"${e_mean:.4f} \\pm {e_std:.4f}$ \\\\"
        )
        latex_lines.append(row_str)
        
    # Compute Macro Averages across all classes
    # --- Corrected Macro Average Calculations ---
    macro_brier_means = df_trials[[f"{c}_Brier" for c in label_names]].mean(axis=1)
    macro_log_means = df_trials[[f"{c}_LogLoss" for c in label_names]].mean(axis=1)
    macro_ece_means = df_trials[[f"{c}_ECE" for c in label_names]].mean(axis=1)
    
    # Inject isolation line for the summary statistics block
    latex_lines.append("    \\midrule")
    
    macro_row = (
        r"\textbf{Macro Average} &"
        f"${macro_brier_means.mean():.4f} \\pm {macro_brier_means.std(ddof=1):.4f}$ & "
        f"${macro_log_means.mean():.4f} \\pm {macro_log_means.std(ddof=1):.4f}$ & "
        f"${macro_ece_means.mean():.4f} \\pm {macro_ece_means.std(ddof=1):.4f}$ \\\\"
    )
    latex_lines.append(macro_row)
    
    # Close out structures safely
    latex_lines.extend([
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}"
    ])
    
    # Combine everything with system-native clean carriage line endings
    return "\n".join(latex_lines)

def evaluate_avex_multilabel(embeddings_dict, y_multilabel, label_name="tab:avex_multilabel"):
    """
    Evaluates bioacoustic embeddings strictly following the AVEX protocol for multi-label data.
    Computes Macro-Averaged Class-wise Retrieval-AUC and omits NMI (per AVEX guidelines).
    
    Parameters:
    -----------
    embeddings_dict : dict
        Keys are model strings (e.g., 'perch', 'effnetb0', 'naturelm-beats').
        Values are 2D arrays of shape (N_samples, N_dimensions).
    y_multilabel : array-like
        A 2D binary indicator matrix of shape (N_samples, N_classes).
    """
    y_multilabel = np.array(y_multilabel)
    n_samples, n_classes = y_multilabel.shape
    
    # Check if data is accidentally single-label
    if y_multilabel.ndim == 1 or n_classes == 1:
        raise ValueError("This function is explicitly for 2D multi-label indicator matrices.")

    results = []
    model_order = ['perch', 'effnet', 'nature']
    display_names = {
        'perch': 'Google Perch',
        'effnet': 'EfficientNet-B0',
        'nature': 'NatureLM-BEATs'
    }

    for target_key in model_order:
        # Match dictionary keys flexibly (case-insensitive keyword matching)
        matched_key = None
        for real_key in embeddings_dict.keys():
            if target_key in str(real_key).lower():
                matched_key = real_key
                break
                
        if matched_key is None:
            continue
            
        X = np.array(embeddings_dict[matched_key])
        
        if X.shape[0] != n_samples:
            print(f"⚠️ Skipping {matched_key}: Row mismatch (Embeddings: {X.shape[0]}, Labels: {n_samples}).")
            continue

        # Compute Pairwise Cosine Similarity Matrix
        sim_matrix = cosine_similarity(X)
        class_auc_means = []
        
        # AVEX Multi-label Retrieval: Evaluate retrieval performance class-by-class
        for c in range(n_classes):
            class_aucs = []
            for i in range(n_samples):
                # An item is a valid query for class 'c' only if that class is present in it
                if y_multilabel[i, c] == 1:
                    mask = np.ones(n_samples, dtype=bool)
                    mask[i] = False
                    
                    binary_targets = y_multilabel[mask, c]  # 1 if database item has class c, 0 if not
                    similarity_scores = sim_matrix[i, mask]
                    
                    # Compute ROC-AUC if both positive and negative matches exist in the database pool
                    if len(np.unique(binary_targets)) == 2:
                        auc = roc_auc_score(binary_targets, similarity_scores)
                        class_aucs.append(auc)
            
            # Record the average retrieval quality for this specific vocalization class
            if class_aucs:
                class_auc_means.append(np.mean(class_aucs))
                
        # Macro average across all classes (as reported in AVEX retrieval tables)
        macro_retrieval_auc = np.mean(class_auc_means) if class_auc_means else 0.0
        
        results.append({
            'Model': display_names[target_key],
            'Clustering NMI': 'N/A (Multi-label)',  # Left as N/A to strictly reflect the AVEX paradigm
            'Macro Retrieval-AUC': f"{macro_retrieval_auc:.4f}"
        })

    if not results:
        print("❌ Error: No models successfully matched or processed.")
        return ""

    # Generate LaTeX table code
    df_res = pd.DataFrame(results)
    df_res.rename(columns={
        'Model': '\\textbf{Encoder Model}',
        'Clustering NMI': '\\textbf{Clustering NMI}',
        'Macro Retrieval-AUC': '\\textbf{Macro Retrieval-AUC}'
    }, inplace=True)
    
    latex_body = df_res.to_latex(index=False, column_format='lcc', escape=False)
    
    latex_table = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\caption{Multi-Label Zero-Shot Embedding Evaluation: Macro-averaged Retrieval-AUC "
        "scores across overlapping call types. Clustering NMI is omitted per standard AVEX protocols "
        "for multi-label benchmarks.}\n"
        f"\\label{{{label_name}}}\n"
        f"{latex_body}"
        "\\end{table}"
    )
    
    return latex_table


def generate_master_species_table_with_totals(y_pred_proba, metadata_csv="ood_metadata.csv"):
    """Generates a master table of Average Precision (AP) scores for each bat species and call type,
      including a summary row for all recordings.
      metadata_csv: Path to the CSV file containing the metadata with true labels and species information.
      y_pred_proba: 2D array of shape (N_samples, N_classes) containing predicted probabilities for each class."""
    df = pd.read_csv(metadata_csv)
    class_cols = ['type_a', 'type_b', 'type_c', 'type_d', 'echo']
    
    # 1. Map model predictions
    for idx, col in enumerate(class_cols):
        df[f'pred_{col}'] = y_pred_proba[:, idx]
        
    # 2. Compute individual species rows
    species_rows = []
    for species, group_data in df.groupby('species_latin'):
        if len(group_data) < 2:
            continue
            
        entry = {'Bat Species': species}
        valid_aps = []
        
        for col in class_cols:
            y_true = group_data[col].values
            y_pred = group_data[f'pred_{col}'].values
            
            if len(np.unique(y_true)) > 1:
                ap_score = average_precision_score(y_true, y_pred)
                entry[f'{col.upper()} AP'] = f"{ap_score:.4f}"
                valid_aps.append(ap_score)
            else:
                entry[f'{col.upper()} AP'] = "-"
        
        if valid_aps:
            entry['cmap'] = f"{np.mean(valid_aps):.4f}"
        else:
            entry['cmap'] = "-"
            
        species_rows.append(entry)
        
    # Create and sort the species dataframe alphabetically
    master_table = pd.DataFrame(species_rows)
    if not master_table.empty:
        master_table.sort_values(by='Bat Species', inplace=True)
    
    # 3. Compute the GLOBAL summary row across ALL recordings
    global_entry = {'Bat Species': 'All Recordings'}
    global_aps = []
    
    for col in class_cols:
        y_true = df[col].values
        y_pred = df[f'pred_{col}'].values
        
        # Across the entire dataset, all classes will have 0s and 1s present
        global_ap = average_precision_score(y_true, y_pred)
        global_entry[f'{col.upper()} AP'] = f"{global_ap:.4f}"
        global_aps.append(global_ap)
        
    global_entry['cmap'] = f"{np.mean(global_aps):.4f}"
    
    # Append the summary row explicitly to the bottom
    master_table = pd.concat([master_table, pd.DataFrame([global_entry])], ignore_index=True)
    
    return master_table


def convert_to_latex_table(master_table_df, label="tab:master_species_ap"):
    """Converts the master species AP dataframe into a LaTeX table string with proper formatting."""
    df_latex = master_table_df.copy()
    
    # Scientific formatting: Italics for real species, Bold for the summary row
    df_latex['Bat Species'] = df_latex['Bat Species'].apply(
        lambda x: f"\\textbf{{{x}}}" if x == "All Recordings" else f"\\textit{{{x}}}"
    )
    
    # Map headers to clean academic styles
    header_mapping = {
        'Bat Species': '\\textbf{Bat Species}',
        'TYPE_A AP': '\\textbf{Type A AP}',
        'TYPE_B AP': '\\textbf{Type B AP}',
        'TYPE_C AP': '\\textbf{Type C AP}',
        'TYPE_D AP': '\\textbf{Type D AP}',
        'ECHO AP': '\\textbf{Echo AP}',
        'cmap': '\\textbf{cmAP}'
    }
    header_mapping = {k: v for k, v in header_mapping.items() if k in df_latex.columns}
    df_latex.rename(columns=header_mapping, inplace=True)
    
    # Generate the initial table body
    latex_body = df_latex.to_latex(
        index=False,
        column_format='lcccccc', 
        escape=False
    )
    
    # Inject a structural horizontal line directly above the summary row
    if "\\textbf{All Recordings}" in latex_body:
        latex_body = latex_body.replace(
            "\\textbf{All Recordings}", 
            "\\midrule\n\\textbf{All Recordings}"
        )
    
    latex_wrapper = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\caption{Master Evaluation Table: Out-of-Distribution (OOD) performance "
        "by species and global dataset configurations via Average Precision (AP) "
        "and class Mean Average Precision (cmAP).}\n"
        f"\\label{{{label}}}\n"
        "\\small\n"
        f"{latex_body}"
        "\\end{table}"
    )
    
    return latex_wrapper
