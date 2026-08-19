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

def perch_extraction(batdata, device='cpu',num_audio = 30):
    """
    Extracts features from the Perch 2.0 encoder for a given dataset.
    Returns a list of feature arrays and a corresponding array of labels.
    """
    feature_list = []
    label_list = []
    print(f"Dataset type: {type(batdata)}")
    
    model_path = hf_hub_download(
        repo_id="justinchuby/Perch-onnx",
        filename="perch_v2.onnx"
    )
    
    session = ort.InferenceSession(
        model_path, 
        providers=["CPUExecutionProvider"] if device=='cpu' else ["CUDAExecutionProvider", "CPUExecutionProvider"] 
    )

    for i in range(num_audio):
        windows, labels = batdata[i]
        windows = windows.numpy().astype(np.float32)
        
        batch_size = len(windows)
        feats = []
        for j in range(0, len(windows), batch_size):
            batch = windows[j:j+batch_size]
            out = session.run(["spatial_embedding"], {"inputs": batch})[0]
            feats.append(out)
        
        feats = np.concatenate(feats, axis=0)
        feature_list.append(feats)
        label_list.append(labels.numpy())

    return feature_list, np.array(label_list)

def nlm_extraction(batdata, device='cpu', num_audio=30):
    """
    Extracts features from the NLM_BEATs encoder for a given dataset.
    Returns a list of feature arrays and a corresponding array of labels.
    """
    feature_list = []
    label_list = []
    print(f"Dataset type: {type(batdata)}")
    
    encoder = extract_encoder("NLM_BEATs", device=device)

    for i in range(num_audio):
        windows, labels = batdata[i]
        feats = extract_feature(windows, encoder, "NLM_BEATs", device)
        feature_list.append(feats.numpy())
        label_list.append(labels.numpy())
            
    return feature_list, np.array(label_list)

def eff_extraction(batdata, device='cpu', num_audio=30):
    """
    Extracts features from the effnetb0 encoder for a given dataset.
    Returns a list of feature arrays and a corresponding array of labels.
    """
    feature_list = []
    label_list = []
    print(f"Dataset type: {type(batdata)}")
    
    encoder = extract_encoder("effnetb0", device=device)

    for i in range(num_audio):
        windows, labels = batdata[i]
        feats = extract_feature(windows, encoder, "effnetb0", device)
        feature_list.append(feats.numpy())
        label_list.append(labels.numpy())
            
    return feature_list, np.array(label_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=False, default="/idiap/temp/adeych/data",
                         help="Path to the temp/scratch directory containing your data")
    parser.add_argument("--num_audio", type=int, default=30,
                         help="Number of audio files to process for feature extraction")
    parser.add_argument("--device", type=str, default='all', choices=['cpu', 'cuda', 'all'],
                         help="Device to use for feature extraction (e.g., 'cpu' or 'cuda' or 'all')")
    parser.add_argument("--encoder", type=str, choices=['perch2', 'NLM_BEATs', 'effnetb0','all'], default='all',
                         help="Encoder model to use for feature extraction")
    args = parser.parse_args()

    print("\n" + "=" * 50)
    print("ENCODER EXTRACTION TEST")
    print("=" * 50)
    dir = Path(args.data_dir)

    if args.encoder == 'perch2' or args.encoder == 'all':
        batdata = BioacousticDataset(data_input=str(dir / "bat_metadata.csv"),
                             root_dir=str(dir / "xenocanto-dataset"),
                             encoder = "perch2")
        if args.device == 'cuda' or args.device == 'all':
            print("\n--- Extracting features using Perch 2.0 on GPU ---")
            t0 = time.perf_counter()
            perch_extraction(batdata, device='cuda', num_audio=args.num_audio)
            t1 = time.perf_counter()
            print(f"Perch 2.0 feature extraction on GPU took {t1 - t0:.2f} seconds")
        if args.device == 'cpu' or args.device == 'all':
            print("\n--- Extracting features using Perch 2.0 on CPU ---")
            t0 = time.perf_counter()
            perch_extraction(batdata, device='cpu', num_audio=args.num_audio)
            t1 = time.perf_counter()
            print(f"Perch 2.0 feature extraction on CPU took {t1 - t0:.2f} seconds")
    
    if args.encoder == 'NLM_BEATs' or args.encoder == 'all':
        batdata = BioacousticDataset(data_input=str(dir / "bat_metadata.csv"),
                             root_dir=str(dir / "xenocanto-dataset"),
                             encoder = "NLM_BEATs")
        if args.device == 'cuda' or args.device == 'all':
            print("\n--- Extracting features using NLM_BEATs on GPU ---")
            t0 = time.perf_counter()
            nlm_extraction(batdata, device='cuda', num_audio=args.num_audio)
            t1 = time.perf_counter()
            print(f"NLM_BEATs feature extraction on GPU took {t1 - t0:.2f} seconds")
        if args.device == 'cpu' or args.device == 'all':
            print("\n--- Extracting features using NLM_BEATs on CPU ---")
            t0 = time.perf_counter()
            nlm_extraction(batdata, device='cpu', num_audio=args.num_audio)
            t1 = time.perf_counter()
            print(f"NLM_BEATs feature extraction on CPU took {t1 - t0:.2f} seconds")

    if args.encoder == 'effnetb0' or args.encoder == 'all':
        batdata = BioacousticDataset(data_input=str(dir / "bat_metadata.csv"),
                             root_dir=str(dir / "xenocanto-dataset"),
                             encoder = "effnetb0")
        if args.device == 'cuda' or args.device == 'all':
            print("\n--- Extracting features using effnetb0 on GPU ---")
            t0 = time.perf_counter()
            eff_extraction(batdata, device='cuda', num_audio=args.num_audio)
            t1 = time.perf_counter()
            print(f"effnetb0 feature extraction on GPU took {t1 - t0:.2f} seconds")
        if args.device == 'cpu' or args.device == 'all':
            print("\n--- Extracting features using effnetb0 on CPU ---")
            t0 = time.perf_counter()
            eff_extraction(batdata, device='cpu', num_audio=args.num_audio)
            t1 = time.perf_counter()
            print(f"effnetb0 feature extraction on CPU took {t1 - t0:.2f} seconds")

    print("\n" + "=" * 50)
    print("ALL CHECKS PASSED")
    print("=" * 50)