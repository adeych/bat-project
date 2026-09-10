import argparse
import os
import sys
import onnxruntime as ort #verify difference between onnxruntime and onnxruntime-gpu
from huggingface_hub import hf_hub_download
from src.feature_generation import extract_feature, extract_encoder
from src.preprocessing import BioacousticDataset
import time
import numpy as np
from pathlib import Path
import pickle
from src.feature_generation import build_perch_fb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=False, default="/idiap/temp/adeych/data",
                         help="Path to the temp/scratch directory containing your data")
    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("ENCODER EXTRACTION TEST")
    print("=" * 50)
    dir = Path(args.data_dir)

    batdata = BioacousticDataset(data_input=str(dir / "ood_metadata.csv"),
                             root_dir=str(dir / "domain-transfer-dataset"),
                             encoder = "perch2")

    results = build_perch_fb(batdata,device='cuda')
    embeddings, labels = results

    save_path = f"/idiap/temp/adeych/data/feature_banks/perch2_ood_features.pkl"
        
    with open(save_path, "wb") as f:
        pickle.dump(embeddings, f)

    
    save_path = f"/idiap/temp/adeych/data/feature_banks/perch2_ood_labels.pkl"
        
    with open(save_path, "wb") as f:
        pickle.dump(labels, f)

    
    print("\n" + "=" * 50)
    print("ALL CHECKS PASSED")
    print("=" * 50)