from pathlib import Path 
import pickle
from src.feature_generation import pool_features
import numpy as np
from src.mil_multilabel import abmil_classifier_tuned_optuna
from src.linear_probe import linear_probe_tuned_optuna
import argparse


def mil_optimisation(variants,ensemble) :
    encoder_name = "perch2"
    dir = Path("/idiap/temp/adeych/data")
    X_bags_dir = "perch2-bags.pkl"
    with open(dir / "feature_banks" / X_bags_dir, "rb") as f:
        X_bags = pickle.load(f)
    X_bags_pooled = pool_features(X_bags, windows=False,window_pooled=False,encoder = encoder_name)
    y = np.load(dir / "feature_banks" / "labels.npy")

    results = abmil_classifier_tuned_optuna(X_bags_pooled, y, n_split_out=5, n_split_in=5, num_trials=5,
                                   random_state=42, n_optuna_trials=5, n_epochs_max=40,
                                   variants=variants,
                                   ensemble=ensemble, ensemble_labels=(0,),
                                   ensemble_n_optuna_trials=5, ensemble_n_split_in=3,
                                   ensemble_n_epochs_max=40)
    return results

def linear_probe_optimisation() :
    encoder_name = "perch2"
    dir = Path("/idiap/temp/adeych/data")
    X_bags_dir = "perch2-bags.pkl"
    with open(dir / "feature_banks" / X_bags_dir, "rb") as f:
        X_bags = pickle.load(f)
    X_bags_pooled = pool_features(X_bags, windows=False,window_pooled=False,encoder = encoder_name)
    X_bags_mean = pool_features(X_bags_pooled,windows = True, encoder = encoder_name)
    y = np.load(dir / "feature_banks" / "labels.npy")

    results = linear_probe_tuned_optuna(
           X_bags_mean, y, n_split_out=5, n_split_in=5, num_trials=5,
           n_optuna_trials=5, n_epochs_max=40, random_state=42)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", type=str, required=True,
                         help="Algorithm : ABMIL, ABMIL_LSTM, LSTM, LSTM_proj,ABMIL_LSTM_proj, Linear")
    parser.add_argument("--ensembling", action=argparse.BooleanOptionalAction, required=True,
                     help="True : have dedicated MLP for Type A")
    parser.add_argument("--name", type=str, required=True,
                            help="Name for result file")
    args = parser.parse_args()

    print("=" * 50)
    print("ABMIL + LSTM OPTIMISATION")
    print("=" * 50)

    if args.algorithm == "ABMIL" :
        variants = ('ABMIL',)
    elif args.algorithm == "ABMIL_LSTM" :
        variants = ('ABMIL_LSTM',)
    elif args.algorithm == "LSTM" :
        variants = ('LSTM_only',)
    elif args.algorithm == "LSTM_proj" :
        variants = ('LSTM_residual_proj',)
    elif args.algorithm == "ABMIL_LSTM_proj" :
        variants = ('ABMIL_LSTM_residual_proj',)    

    print(args.algorithm)
    print(args.ensembling)
    print(args.name)
    en_state = "en" if args.ensembling else "si"
    
    if args.algorithm == "Linear" :
        results = linear_probe_optimisation()
    else : 
        results = mil_optimisation(variants,args.ensembling)

    save_path = f"/idiap/home/adeych/bat-project/results/mil_{args.algorithm}_results_{en_state}_{args.name}.pkl"
    
    with open(save_path, "wb") as f:
        pickle.dump(results, f)

    print("\n" + "=" * 50)
    print("ALL CHECKS PASSED")
    print("=" * 50)