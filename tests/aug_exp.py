import argparse
import os
import sys
import argparse
import os
import sys
from huggingface_hub import hf_hub_download
from src.feature_generation import extract_feature, extract_encoder
from src.preprocessing import BioacousticDataset
import time
import numpy as np
from pathlib import Path
from src.augmented_mlp_training import load_encoder,run_cv_trials
from src.feature_generation import pool_features
from src.preprocessing import PipistrellePreprocessingPipeline,AudioCompose,TimeMasking,FrequencyMasking,TimeStretch,WaveformMixup
import pandas as pd
from pathlib import Path
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_name", type=str, required=True,
                         help="Encoder name, either perch2, NLM_BEATs, or effnetb0")
    args = parser.parse_args()

    print("=" * 50)
    print("DATA AUGMENTATION EXPERIMENT WITH ENCODER {}")
    print("=" * 50)

    
    dir = Path("/idiap/temp/adeych/data")
    encoder_name = args.encoder_name

    encoder = load_encoder(encoder_name, device="cuda")

    if encoder_name == "perch2":
       target_sr = 32000 
       pipeline = PipistrellePreprocessingPipeline(target_sr=target_sr, expansion_factor=5,window_sec =5,overlap=0.5)
       X_bags_dir = "perch2-bags.pkl"
    elif encoder_name == "effnetb0":
        target_sr = 16000
        pipeline = PipistrellePreprocessingPipeline(target_sr=target_sr, expansion_factor=10,window_sec =10,overlap=0.5)
        X_bags_dir = "effnetb0-bags.pkl"
    elif encoder_name == "NLM_BEATs":
        target_sr = 16000
        pipeline = PipistrellePreprocessingPipeline(target_sr=target_sr, expansion_factor=10,window_sec =10,overlap=0.5)
        X_bags_dir = "NLM-bags.pkl"
    else:
        raise ValueError(f"Unsupported encoder : {encoder_name}. Please choose from 'perch2', 'effnetb0', or 'NLM_BEATs'.")

    augment_pipeline = AudioCompose([
        (TimeMasking(), 0.3),
        (FrequencyMasking(), 0.3),
        (TimeStretch(target_sr=target_sr), 0.3),
    ])

    label_cols = ['type_a', 'type_b', 'type_c', 'type_d', 'echo']
    clip_df = pd.read_csv(dir / "bat_metadata.csv")  # one row per clip, with relative_path column
    root_dir = dir / "xenocanto-dataset"

    print("Started loading features")
    with open(dir / "feature_banks" / X_bags_dir, "rb") as f:
        X_bags = pickle.load(f)
    print("Finished loading features")
    window_level_list = pool_features(X_bags, windows=False,window_pooled=False,encoder = encoder_name)
    clean_window_embeddings = dict(zip(clip_df['relative_path'], window_level_list))
    X = pool_features(window_level_list,windows=True,encoder = encoder_name)

    y = np.load(dir / "feature_banks" / "labels.npy")

    all_results = run_cv_trials(clip_df, clean_window_embeddings, X, y, root_dir,
                   encoder, encoder_name, pipeline, augment_pipeline,
                   label_cols, n_splits=5, num_trials=2, aug_prob=0.2,
                   use_oversampling = True, ovs_percentage = 0.2,
                   mixup_prob=0.3, mixup_alpha=0.2,epochs =20,
                   random_state=42, device="cuda")

    save_path = f"/idiap/home/adeych/bat-project/results/mlp_{encoder_name}_results1.pkl"

    with open(save_path, "wb") as f:
        pickle.dump(all_results, f)


    print("\n" + "=" * 50)
    print("TEST DONE")
    print("=" * 50)