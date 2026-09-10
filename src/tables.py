"""
Functions to process raw cross-validation outputs into latex summary table for encoder comparison
"""

import numpy as np
from sklearn.metrics import average_precision_score
from src.metrics import calculate_ece
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

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

