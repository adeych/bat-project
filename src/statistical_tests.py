"""
Statistical tests for comparing encoder performance across different algorithms.
"""


import numpy as np
import pandas as pd
import scipy.stats as stats
import scikit_posthocs as sp
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FixedLocator
from sklearn.metrics import average_precision_score

def perform_encoder_statistical_analysis(all_encoders_results, label_names=None):
    """
    Performs a Friedman test followed by a  Nemenyi post-hoc test
    to compare encoder performance across matched cross-validation blocks.
    Bypasses scikit-posthocs to avoid NaN issues.
    """
    if label_names is None:
        label_names = ['Type A', 'Type B', 'Type C', 'Type D', 'Echo']

    # --- Step 1: Extract and Align Matched Samples ---
    structured_data = {}
    
    for encoder_name, trials_list in all_encoders_results.items():
        active_trial_idx = 0
        for trial in trials_list:
            model_name = trial['model']
            if any(x in model_name.lower() for x in ['prevalence', 'dummy', 'guesser']):
                continue
                
            y_true_folds = trial['y_true_cv']          
            y_pred_proba_folds = trial['y_pred_proba_cv']  
            
            if any(x in model_name.lower() for x in ['logistic', 'regressor', 'regression']):
                clean_model = 'Logistic Regression'
            elif 'svm' in model_name.lower():
                clean_model = 'SVM'
            elif any(x in model_name.lower() for x in ['forest', 'rf']):
                clean_model = 'Random Forest'
            elif 'mlp' in model_name.lower():
                clean_model = 'MLP'
            else:
                clean_model = model_name

            for fold_idx in range(len(y_true_folds)):
                y_true = y_true_folds[fold_idx]
                y_pred_proba = y_pred_proba_folds[fold_idx]
                
                for c_idx, class_name in enumerate(label_names):
                    ap_score = average_precision_score(y_true[:, c_idx], y_pred_proba[:, c_idx])
                    block_id = (clean_model, active_trial_idx, fold_idx, class_name)
                    
                    if block_id not in structured_data:
                        structured_data[block_id] = {}
                    structured_data[block_id][encoder_name] = ap_score
                    
            active_trial_idx += 1

    df_stat = pd.DataFrame.from_dict(structured_data, orient='index').dropna()
    encoders = list(df_stat.columns)
    
    print(f"=== Statistical Setup ===")
    print(f"Number of perfectly matched Blocks (N): {len(df_stat)}")
    print(f"Treatments (Encoders mapped): {encoders}\n")

    # --- Step 2: Friedman Test ---
    encoder_vectors = [df_stat[col].values for col in encoders]
    friedman_stat, p_value = stats.friedmanchisquare(*encoder_vectors)
    
    print(f"=== Friedman Test Results ===")
    print(f"Friedman $\chi^2$ Statistic: {friedman_stat:.4f}")
    print(f"p-value: {p_value:.4e}")
    
    alpha = 0.05
    if p_value < alpha:
        print(f"Result: Significant difference detected (p < {alpha}). Proceeding to Post-Hoc Nemenyi Test.\n")
        
        # --- Step 3: Compute Mean Ranks ---
        # rank ascending=False means highest AP score gets Rank 1
        ranks = df_stat.rank(axis=1, ascending=False)
        mean_ranks = ranks.mean()
        
        print("=== Mean Ranks (Lower is Better/Rank 1 is Top) ===")
        for enc in encoders:
            print(f"• {enc}: {mean_ranks[enc]:.3f}")
            
        # --- Step 4: Calculate Nemenyi CD Boundary ---
        k = len(encoders)
        N = len(df_stat)
        q_alpha = 3.314  # Studentized range critical value for k=3 groups, alpha=0.05
        cd_value = q_alpha * np.sqrt((k * (k + 1)) / (6 * N))
        print(f"\nCalculated Nemenyi Critical Difference (CD): {cd_value:.4f}")
        
        # --- Step 5: Direct Pairwise p-value Matrix Calculation ---
        # Using Studentized Range distribution property to compute exact p-values
        nemenyi_matrix = pd.DataFrame(1.0, index=encoders, columns=encoders)
        
        for i in range(k):
            for j in range(i + 1, k):
                enc1, enc2 = encoders[i], encoders[j]
                rank_diff = abs(mean_ranks[enc1] - mean_ranks[enc2])
                
                # Standard error for the rank difference estimate
                sei = np.sqrt((k * (k + 1)) / (6 * N))
                q_value = (rank_diff / sei) * np.sqrt(2) # Convert to standard studentized range scale
                
                # Approximate pairwise p-value using the studentized range survival function
                p_pairwise = stats.studentized_range.sf(q_value, k, np.inf)
                
                nemenyi_matrix.loc[enc1, enc2] = p_pairwise
                nemenyi_matrix.loc[enc2, enc1] = p_pairwise
                
        print("\n=== Nemenyi Pairwise p-value Matrix ===")
        print(nemenyi_matrix.to_string(float_format=lambda x: f"{x:.4e}" if x < 0.001 else f"{x:.4f}"))
        print("\n*Interpretation: Pairwise cells show exact p-values. If p < 0.05, performance difference is significant.")
        
    else:
        print(f"Result: No significant variance found among encoders (p >= {alpha}).")

    return df_stat

def perform_and_plot_nemenyi(all_encoders_results, label_names=None):
    """
    Performs Friedman + Nemenyi to compare encoder performance and plots a  
    significance heatmap with exact p-values rendered inside the cells.
    """
    if label_names is None:
        label_names = ['Type A', 'Type B', 'Type C', 'Type D', 'Echo']

    # --- Step 1: Extract and Align Matched Samples ---
    structured_data = {}
    for encoder_name, trials_list in all_encoders_results.items():
        active_trial_idx = 0
        for trial in trials_list:
            model_name = trial['model']
            if any(x in model_name.lower() for x in ['prevalence', 'dummy', 'guesser']):
                continue
                
            y_true_folds = trial['y_true_cv']          
            y_pred_proba_folds = trial['y_pred_proba_cv']  
            
            if any(x in model_name.lower() for x in ['logistic', 'regressor', 'regression']):
                clean_model = 'Logistic Regression'
            elif 'svm' in model_name.lower():
                clean_model = 'SVM'
            elif any(x in model_name.lower() for x in ['forest', 'rf']):
                clean_model = 'Random Forest'
            elif 'mlp' in model_name.lower():
                clean_model = 'MLP'
            else:
                clean_model = model_name

            for fold_idx in range(len(y_true_folds)):
                y_true = y_true_folds[fold_idx]
                y_pred_proba = y_pred_proba_folds[fold_idx]
                
                for c_idx, class_name in enumerate(label_names):
                    ap_score = average_precision_score(y_true[:, c_idx], y_pred_proba[:, c_idx])
                    block_id = (clean_model, active_trial_idx, fold_idx, class_name)
                    
                    if block_id not in structured_data:
                        structured_data[block_id] = {}
                    structured_data[block_id][encoder_name] = ap_score
            active_trial_idx += 1

    df_stat = pd.DataFrame.from_dict(structured_data, orient='index').dropna()
    encoders = list(df_stat.columns)
    k = len(encoders)
    N = len(df_stat)

    # --- Step 2: Global Friedman Test ---
    encoder_vectors = [df_stat[col].values for col in encoders]
    friedman_stat, global_p = stats.friedmanchisquare(*encoder_vectors)
    
    print(f"Friedman test p-value: {global_p:.4e}")
    if global_p >= 0.05:
        print("No globally significant differences found. Skipping Post-hoc heatmap.")
        return

    # --- Step 3: Compute Exact Nemenyi Pairwise p-values ---
    ranks = df_stat.rank(axis=1, ascending=False)
    mean_ranks = ranks.mean()
    
    p_matrix = pd.DataFrame(1.0, index=encoders, columns=encoders)
    for i in range(k):
        for j in range(i + 1, k):
            enc1, enc2 = encoders[i], encoders[j]
            rank_diff = abs(mean_ranks[enc1] - mean_ranks[enc2])
            sei = np.sqrt((k * (k + 1)) / (6 * N))
            q_value = (rank_diff / sei) * np.sqrt(2)
            
            p_pairwise = stats.studentized_range.sf(q_value, k, np.inf)
            p_matrix.loc[enc1, enc2] = p_pairwise
            p_matrix.loc[enc2, enc1] = p_pairwise

    # --- Step 4: Map Categorical Colors & Generate Text Annotations ---
    bin_matrix = pd.DataFrame(0, index=encoders, columns=encoders, dtype=float)
    annot_labels = np.empty((k, k), dtype=object)

    for i in range(k):
        for j in range(k):
            if i == j:
                bin_matrix.iloc[i, j] = np.nan
                annot_labels[i, j] = ""  # Leave main diagonal completely blank
                continue
                
            pval = p_matrix.iloc[i, j]
            
            # Format p-values dynamically based on scale
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

    # --- Step 5: Plotting Heatmap ---
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(7.5, 5.5))  # Slightly adjusted aspect ratio for annotations

    colors = ['#FCD7D7', "#CBE6CF", "#A9D1B6", "#7A9B85"]
    custom_cmap = ListedColormap(colors)

    # Render base heatmap with custom annotations array
    sns.heatmap(
        bin_matrix, 
        cmap=custom_cmap, 
        vmin=0, vmax=3, 
        annot=annot_labels, 
        fmt="",  # Required when passing a pre-formatted string array
        annot_kws={"fontsize": 10, "fontweight": "bold", "color": "#1c1c1c"},
        linewidths=1.2, 
        linecolor='#9e9e9e', 
        cbar=False, 
        square=True,
        ax=ax
    )

    ax.set_xticklabels(encoders, rotation=0, fontsize=11, fontweight='bold')
    ax.set_yticklabels(encoders, rotation=0, fontsize=11, fontweight='bold')
    ax.set_title("", fontsize=13, fontweight='bold', pad=15)

    # --- Step 6: Fixed Discrete Colorbar Layout ---
    cax = fig.add_axes([0.86, 0.28, 0.035, 0.44]) 
    norm = Normalize(vmin=0, vmax=4)
    mappable = ScalarMappable(norm=norm, cmap=custom_cmap)
    
    cb = fig.colorbar(
        mappable, 
        cax=cax, 
        boundaries=[0, 1, 2, 3, 4], 
        orientation='vertical'
    )
    
    cb.locator = FixedLocator([0.5, 1.5, 2.5, 3.5])
    cb.update_ticks()
    cb.ax.set_yticklabels(['NS', r'$p < 0.05$', r'$p < 0.01$', r'$p < 0.001$'], fontsize=11)
    cb.ax.tick_params(size=0)  
    cb.outline.set_edgecolor('#9e9e9e')  
    cb.outline.set_linewidth(1.2)

    plt.subplots_adjust(left=0.25, right=0.84, top=0.88, bottom=0.2)
    plt.show()


def evaluate_and_plot_mlp_strategies(balancing_results, augmented_results, label_names=None):
    """
    Performs Friedman + Nemenyi post-hoc statistical tests across 5 single MLP 
    training strategies and plots a heatmap.
    """
    if label_names is None:
        label_names = ['Type A', 'Type B', 'Type C', 'Type D', 'Echo']

    structured_data = {}

    # --- Step 1: Extract and Align Matched Samples from Balancing Group ---
    for trial_entry in balancing_results:
        model_name = trial_entry['model']      
        y_true_folds = trial_entry['y_true_cv']          
        y_pred_proba_folds = trial_entry['y_pred_proba_cv']  
        trial_idx = trial_entry['trial']
        
        # Format strings to clean handles
        clean_name = model_name.replace('MLP_', '')
        if clean_name == 'ClassWeights':
            clean_name = 'Class Weights'
        elif clean_name == 'FocalLoss':
            clean_name = 'Focal Loss'

        for fold_idx in range(len(y_true_folds)):
            y_true = y_true_folds[fold_idx]
            y_pred_proba = y_pred_proba_folds[fold_idx]
            
            for c_idx, class_name in enumerate(label_names):
                ap_score = average_precision_score(y_true[:, c_idx], y_pred_proba[:, c_idx])
                # Unique block token matching a specific environment state
                block_id = (trial_idx, fold_idx, class_name)
                
                if block_id not in structured_data:
                    structured_data[block_id] = {}
                structured_data[block_id][clean_name] = ap_score

    # --- Step 2: Extract and Align Matched Samples from Augmented Group ---
    for trial_entry in augmented_results:
        y_true_folds = trial_entry['y_true_cv']          
        y_pred_proba_folds = trial_entry['y_pred_proba_cv']  
        trial_idx = trial_entry['trial']

        for fold_idx in range(len(y_true_folds)):
            y_true = y_true_folds[fold_idx]
            y_pred_proba = y_pred_proba_folds[fold_idx]
            
            for c_idx, class_name in enumerate(label_names):
                ap_score = average_precision_score(y_true[:, c_idx], y_pred_proba[:, c_idx])
                block_id = (trial_idx, fold_idx, class_name)
                
                if block_id in structured_data:
                    structured_data[block_id]['Augmented'] = ap_score

    # Construct the final matrix and drop unaligned rows
    df_stat = pd.DataFrame.from_dict(structured_data, orient='index').dropna()
    strategies = ['Baseline', 'Class Weights', 'Focal Loss', 'Oversampled', 'Augmented']
    df_stat = df_stat[strategies]  # Fix clear display ordering
    
    k = len(strategies)
    N = len(df_stat)
    print(f"=== Statistical Setup ===")
    print(f"Number of perfectly matched Blocks (N): {N}")
    print(f"Strategies evaluated: {strategies}\n")

    # --- Step 3: Global Friedman Test ---
    strategy_vectors = [df_stat[col].values for col in strategies]
    friedman_stat, global_p = stats.friedmanchisquare(*strategy_vectors)
    
    print(f"=== Friedman Test Results ===")
    print(f"Friedman chi-square Statistic: {friedman_stat:.4f}")
    print(f"p-value: {global_p:.4e}")
    
    if global_p >= 0.05:
        print("Result: No globally significant differences detected. Skipping Post-Hoc heatmap.")
        return
    print("Result: Significant differences detected. Proceeding to Post-Hoc Nemenyi Test.\n")

    # --- Step 4: Compute Exact Nemenyi Pairwise p-values ---
    ranks = df_stat.rank(axis=1, ascending=False)
    mean_ranks = ranks.mean()
    
    p_matrix = pd.DataFrame(1.0, index=strategies, columns=strategies)
    for i in range(k):
        for j in range(i + 1, k):
            strat1, strat2 = strategies[i], strategies[j]
            rank_diff = abs(mean_ranks[strat1] - mean_ranks[strat2])
            sei = np.sqrt((k * (k + 1)) / (6 * N))
            q_value = (rank_diff / sei) * np.sqrt(2)
            
            p_pairwise = stats.studentized_range.sf(q_value, k, np.inf)
            p_matrix.loc[strat1, strat2] = p_pairwise
            p_matrix.loc[strat2, strat1] = p_pairwise

    # --- Step 5: Map Categorical Colors & Generate Text Annotations ---
    bin_matrix = pd.DataFrame(0, index=strategies, columns=strategies, dtype=float)
    annot_labels = np.empty((k, k), dtype=object)

    for i in range(k):
        for j in range(k):
            if i == j:
                bin_matrix.iloc[i, j] = np.nan
                annot_labels[i, j] = ""  # Leave main diagonal blank
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

    # --- Step 6: Plotting the Custom Significance Heatmap ---
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(8, 6.2))

    # Soft pink (NS) moving to deeper emerald tones for top-tier confidence
    colors = ['#FCD7D7', '#89DD94', '#009E32', '#00521A']
    custom_cmap = ListedColormap(colors)

    sns.heatmap(
        bin_matrix, 
        cmap=custom_cmap, 
        vmin=0, vmax=3, 
        annot=annot_labels, 
        fmt="",  
        annot_kws={"fontsize": 9.5, "fontweight": "bold", "color": "#1c1c1c"},
        linewidths=1.2, 
        linecolor='#9e9e9e', 
        cbar=False, 
        square=True,
        ax=ax
    )

    # Reformat category ticks to stack cleanly on 2 lines
    display_ticks = ["Baseline\n(BCE)", "Class\nWeights", "Focal\nLoss", "Iterative\nOversample", "Data\nAugmented"]
    ax.set_xticklabels(display_ticks, rotation=15, ha='right', fontsize=10.5, fontweight='bold')
    ax.set_yticklabels(display_ticks, rotation=0, fontsize=10.5, fontweight='bold')
    ax.set_title("MLP Optimization Significance\n(Nemenyi Pairwise p-values)", fontsize=13, fontweight='bold', pad=18)

    # --- Step 7: Construct Discrete Colorbar Legend Axis ---
    cax = fig.add_axes([0.88, 0.28, 0.035, 0.44]) 
    norm = Normalize(vmin=0, vmax=4)
    mappable = ScalarMappable(norm=norm, cmap=custom_cmap)
    
    cb = fig.colorbar(mappable, cax=cax, boundaries=[0, 1, 2, 3, 4], orientation='vertical')
    cb.locator = FixedLocator([0.5, 1.5, 2.5, 3.5])
    cb.update_ticks()
    cb.ax.set_yticklabels(['NS', r'$p < 0.05$', r'$p < 0.01$', r'$p < 0.001$'], fontsize=11)
    cb.ax.tick_params(size=0)  
    cb.outline.set_edgecolor('#9e9e9e')  
    cb.outline.set_linewidth(1.2)

    plt.subplots_adjust(left=0.25, right=0.85, top=0.88, bottom=0.22)
    plt.show()

def evaluate_and_plot_linear_probe_algorithms(all_results, label_names=None):
    """
    Performs Friedman + Nemenyi post-hoc statistical tests across 4 core classification
    algorithms applied to Perch 2.0 embeddings, prints mean ranks, and plots a heatmap.
    
    Parameters
    ----------
    all_results : list of dicts
        The output collected directly from the `linear_probe_tuned` function.
    label_names : list of str, optional
        Class names to split blocks. Defaults to your 5 bat call labels.
    """
    if label_names is None:
        label_names = ['Type A', 'Type B', 'Type C', 'Type D', 'Echo']

    structured_data = {}

    # --- Step 1: Extract and Align Matched Samples from all_results ---
    for trial_entry in all_results:
        model_name = trial_entry['model']      
        y_true_folds = trial_entry['y_true_cv']          
        y_pred_proba_folds = trial_entry['y_pred_proba_cv']  
        trial_idx = trial_entry['trial']
        
        # Skip prevalence guesser entirely during parsing
        if 'Prevalence' in model_name or 'Dummy' in model_name:
            continue
            
        clean_name = model_name.strip()

        for fold_idx in range(len(y_true_folds)):
            y_true = y_true_folds[fold_idx]
            y_pred_proba = y_pred_proba_folds[fold_idx]
            
            for c_idx, class_name in enumerate(label_names):
                ap_score = average_precision_score(y_true[:, c_idx], y_pred_proba[:, c_idx])
                block_id = (trial_idx, fold_idx, class_name)
                
                if block_id not in structured_data:
                    structured_data[block_id] = {}
                structured_data[block_id][clean_name] = ap_score

    # Construct the final matrix and drop any unaligned rows
    df_stat = pd.DataFrame.from_dict(structured_data, orient='index').dropna()
    
    # Core 4 strategies (Prevalence guesser removed)
    strategies = ['SVM', 'Logistic Regression', 'Random Forest', 'MLP']
    df_stat = df_stat[strategies]  
    
    k = len(strategies)
    N = len(df_stat)
    
    print(f"=== Statistical Setup ===")
    print(f"Number of perfectly matched Blocks (N): {N}")
    print(f"Algorithms evaluated: {strategies}\n")

    # --- Step 2: Compute and Print Mean Ranks ---
    # rank() assigns 1 to the highest score because ascending=False
    ranks = df_stat.rank(axis=1, ascending=False)
    mean_ranks = ranks.mean()
    
    print(f"=== Algorithm Ranks (Lower is Better) ===")
    sorted_ranks = mean_ranks.sort_values()
    for rank_idx, (algo, rank_val) in enumerate(sorted_ranks.items(), start=1):
        print(f"{rank_idx}. {algo:<25} Mean Rank: {rank_val:.4f}")
    print()

    # --- Step 3: Global Friedman Test ---
    strategy_vectors = [df_stat[col].values for col in strategies]
    friedman_stat, global_p = stats.friedmanchisquare(*strategy_vectors)
    
    print(f"=== Friedman Test Results ===")
    print(f"Friedman chi-square Statistic: {friedman_stat:.4f}")
    print(f"p-value: {global_p:.4e}")
    
    if global_p >= 0.05:
        print("Result: No globally significant differences detected. Skipping Post-Hoc heatmap.")
        return
    print("Result: Significant differences detected. Proceeding to Post-Hoc Nemenyi Test.\n")

    # --- Step 4: Compute Exact Nemenyi Pairwise p-values ---
    p_matrix = pd.DataFrame(1.0, index=strategies, columns=strategies)
    for i in range(k):
        for j in range(i + 1, k):
            strat1, strat2 = strategies[i], strategies[j]
            rank_diff = abs(mean_ranks[strat1] - mean_ranks[strat2])
            sei = np.sqrt((k * (k + 1)) / (6 * N))
            q_value = (rank_diff / sei) * np.sqrt(2)
            
            p_pairwise = stats.studentized_range.sf(q_value, k, np.inf)
            p_matrix.loc[strat1, strat2] = p_pairwise
            p_matrix.loc[strat2, strat1] = p_pairwise

    # --- Step 5: Map Categorical Colors & Generate Text Annotations ---
    bin_matrix = pd.DataFrame(0, index=strategies, columns=strategies, dtype=float)
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

    # --- Step 6: Plotting the Custom Significance Heatmap ---
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(8, 6))

    # Soft pink (NS) transitioning to emerald tones
    colors = ['#FCD7D7', "#CBE6CF", "#A9D1B6", "#7A9B85"]
    custom_cmap = ListedColormap(colors)

    sns.heatmap(
        bin_matrix, 
        cmap=custom_cmap, 
        vmin=0, vmax=3, 
        annot=annot_labels, 
        fmt="",  
        annot_kws={"fontsize": 9.5, "fontweight": "bold", "color": "#1c1c1c"},
        linewidths=1.2, 
        linecolor='#9e9e9e', 
        cbar=False, 
        square=True,
        ax=ax
    )

    # Reformat category ticks (Prevalence guesser omitted)
    display_ticks = ["SVM", "Logistic\nRegression", "Random\nForest", "MLP"]
    ax.set_xticklabels(display_ticks, rotation=15, ha='right', fontsize=10.5, fontweight='bold')
    ax.set_yticklabels(display_ticks, rotation=0, fontsize=10.5, fontweight='bold')
    ax.set_title("Perch 2.0 Embedding Classification Significance\n(Nemenyi Pairwise p-values)", fontsize=12, fontweight='bold', pad=18)

    # --- Step 7: Construct Discrete Colorbar Legend Axis ---
    cax = fig.add_axes([0.88, 0.28, 0.035, 0.44]) 
    norm = Normalize(vmin=0, vmax=4)
    mappable = ScalarMappable(norm=norm, cmap=custom_cmap)
    
    cb = fig.colorbar(mappable, cax=cax, boundaries=[0, 1, 2, 3, 4], orientation='vertical')
    cb.locator = FixedLocator([0.5, 1.5, 2.5, 3.5])
    cb.update_ticks()
    cb.ax.set_yticklabels(['NS', r'$p < 0.05$', r'$p < 0.01$', r'$p < 0.001$'], fontsize=11)
    cb.ax.tick_params(size=0)  
    cb.outline.set_edgecolor('#9e9e9e')  
    cb.outline.set_linewidth(1.2)

    plt.subplots_adjust(left=0.25, right=0.85, top=0.88, bottom=0.22)
    plt.show()
