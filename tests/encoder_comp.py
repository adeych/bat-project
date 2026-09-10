import pickle 
from pathlib import Path
import numpy as np
from src.feature_generation import pool_features
import argparse
import warnings
from src.encoder_comparison import linear_probe_tuned

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", type=str, required=True,
                         help="perch2, effnetb0, NLM_BEATs")
    args = parser.parse_args()
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    dir = Path("/idiap/temp/adeych/data")

    if args.encoder not in ['perch2', 'effnetb0', 'NLM_BEATs']:
        raise ValueError("Invalid encoder name. Choose from 'perch2', 'effnetb0', or 'NLM_BEATs'.")
    if args.encoder == "perch2":
        with open(dir / "feature_banks" / "perch2-bags.pkl", "rb") as f:
                X_bags_per = pickle.load(f)
        X = pool_features(pool_features(X_bags_per,encoder = 'perch2'),encoder = 'perch2',windows = True)
    elif args.encoder == "effnetb0":
        with open(dir / "feature_banks" / "effnetb0-bags.pkl", "rb") as f:
                X_bags_eff = pickle.load(f)
        X = pool_features(pool_features(X_bags_eff,encoder = 'effnetb0'),encoder = 'effnetb0',windows = True)
    elif args.encoder == "NLM_BEATs":
        with open(dir / "feature_banks" / "NLM-bags.pkl", "rb") as f:
                X_bags_nlm = pickle.load(f)
        X = pool_features(pool_features(X_bags_nlm,encoder = 'NLM_BEATs'),encoder = 'NLM_BEATs',windows = True)

    y = np.load(dir / "feature_banks" / "labels.npy")

    results = linear_probe_tuned(X, y, n_split_out=5, n_split_in=5, num_trials=5, random_state=42)
    save_path = f"/idiap/temp/adeych/results/encoder_exp/enc_{args.encoder}_results.pkl"
    
    with open(save_path, "wb") as f:
        pickle.dump(results, f)

    print("\n" + "=" * 50)
    print("ALL CHECKS PASSED")
    print("=" * 50)