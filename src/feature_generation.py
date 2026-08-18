import numpy as np
import torch
from tqdm import tqdm
from avex import load_model

import onnxruntime as ort
from huggingface_hub import hf_hub_download
#import tensorflow as tf
#import tensorflow_hub as hub


def extract_feature(window, encoder, model_name, device='cpu'):
    """
    Extracts features from a given window using the specified encoder model.
    - window: The input audio window (numpy array or torch tensor).
    - encoder: The loaded encoder model (PyTorch or TensorFlow).
    - model_name: The name of the model to use either 'effnetb0', 'NLM_BEATs', or 'perch2'.
    - device: The device to run the model on."""
    # Convert Torch tensor to NumPy for cross-framework compatibility
    if isinstance(window, torch.Tensor):
        window_np = window.cpu().numpy()
    else:
        window_np = window

    if model_name == 'perch2':
        # Perch expects a [Batch, Samples] float32 TF tensor
        # We must use tf.constant and ensure it's on the right device
        with tf.device('/GPU:0' if 'cuda' in str(device) else '/CPU:0'):
            # window_np shape is likely [Num_Windows, Samples]
            feats_tf = encoder(tf.constant(window_np, dtype=tf.float32))
            return torch.from_numpy(feats_tf['spatial_embedding'].numpy())
            
    else:
        # PyTorch models (EffNet/BEATs)
        encoder.eval()
        with torch.no_grad():
            # Ensure input is [Batch, Samples] or [Batch, 1, Samples]
            # avex usually wants [Batch, Samples]
            t_window = torch.from_numpy(window_np).to(device)
            feats = encoder(t_window)
            if isinstance(feats, dict): 
                feats = feats['x']
            return feats.cpu() # Return to CPU to avoid filling GPU RAM

def build_feature_bank(batdata, encoder, model_name, device='cpu'):
    """Builds a feature bank of all recordings by extracting features for each window in the dataset using the specified encoder."""
    feature_list = []
    label_list = []
    print(f"Dataset type: {type(batdata)}")
    
    for i in tqdm(range(len(batdata)), desc=f"Extracting {model_name}"):
        windows, labels = batdata[i]
        
        # Extract features for this specific file
        feats = extract_feature(windows, encoder, model_name, device)
        
        # feats is [Num_Windows, Embedding_Dim]
        feature_list.append(feats.numpy())
        label_list.append(labels.numpy())
            
    # Returns one list of features and a numpy array of labels
    return feature_list, np.array(label_list)

def build_perch_fb(batdata,device='cpu') : 
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

    for i in tqdm(range(len(batdata)), desc=f"Extracting perch 2.0"):
            windows, labels = batdata[i]
            windows = windows.numpy().astype(np.float32)
            
            batch_size = len(windows)
            feats = []
            for j in range(0,len(windows), batch_size) :
                batch = windows[j:j+batch_size]
                out = session.run(["spatial_embedding"], {"inputs": batch})[0]
                feats.append(out)
            # feats is [Num_Windows, Embedding_Dim]
            feats = np.concatenate(feats, axis=0)
            feature_list.append(feats)
            label_list.append(labels.numpy())

    return feature_list, np.array(label_list)
        


def extract_encoder(model_name, device='cpu'):
    """
    Loads the specified encoder model and returns it ready for feature extraction.
    model_name should be one of 'effnetb0', 'NLM_BEATs', or 'perch2'.
    """
    if model_name in ['effnetb0', 'NLM_BEATs', 'perch2']:
        # PyTorch logic
        model_key = "esp_aves2_effnetb0_all" if model_name == 'effnetb0' else "esp_aves2_naturelm_audio_v1_beats"
        encoder = load_model(model_key, device=device, return_features_only=True)
        encoder.to(device)
        return encoder
    else :
        raise ValueError(f"Unsupported model_name: {model_name}")
    #elif model_name == 'perch2':
    #    # TensorFlow logic
    #    suffix = "perch_v2_cpu" if device == 'cpu' else "perch_v2"
    #    perch_url = f"https://www.kaggle.com/models/google/bird-vocalization-classifier/frameworks/TensorFlow2/variations/{suffix}/versions/1"
    #    perch_model = hub.load(perch_url)
    #    return perch_model.signatures['serving_default'] 
    

def pool_features(features, windows : bool = False,window_pooled : bool = False, method : str ='mean',encoder : str = 'perch2'):
    """
    Applies pooling to the extracted features based on the specified method and encoder type.
        - If 'windows' is True, it applies pooling across the window dimension for each recording.
        - If 'window_pooled' is True, it applied patch pooling taking into account the fact that windows have been pooled.
        - The 'method' parameter determines whether to use mean or max pooling.
        - The 'encoder' parameter specifies the encoder type to determine the correct axes for pooling.
    """
    if windows :
        if method == 'mean':
            pooled_list = [np.mean(f, axis=0) for f in features]
            return np.stack(pooled_list)
        elif method == 'max':
            pooled_list = [np.max(f, axis=0) for f in features]
            return np.stack(pooled_list)
        else:
            raise ValueError(f"Unsupported pooling method: {method}")
    else :
        if encoder == 'effnetb0' :
            ax = (2,3)
        elif encoder == 'NLM_BEATs' :
            ax = 1
        elif encoder == 'perch2' :
            ax = (1,2)

        if window_pooled :
            if method == 'mean':
                return features.mean(axis=ax)
            elif method == 'max':
                return features.max(axis=ax)
            else:
                raise ValueError(f"Unsupported pooling method: {method}")
        else :
            if method == 'mean':
                return [f.mean(axis=ax) for f in features]
            elif method == 'max':
                return [f.max(axis=ax) for f in features]
            else:
                raise ValueError(f"Unsupported pooling method: {method}")