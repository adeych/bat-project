"""Audio Dataset and Preprocessing Pipeline for Pipistrelle Bat Recordings

"""
import os
import math
import random
from typing import List, Tuple, Union, Optional, Callable

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, sosfilt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.functional as AF
import torchaudio.transforms as AT

# ==========================================
# 1. Base Preprocessing Pipeline
# ==========================================

class PipistrellePreprocessingPipeline(torch.nn.Module):
    """
    This class encapsulates the entire audio preprocessing pipeline for the Pipistrellus Pipistrellus recordings, including:
    - Loading audio files
    - Applying bandpass filters to isolate relevant frequencies to species
    - Time expansion to make ultrasonic calls analysable for encoders
    - Windowing into fixed-size segments
    - Optional data augmentation (e.g., time shifting)"""

    def __init__(
        self, 
        target_sr=16000, 
        expansion_factor=10, 
        window_sec=10, 
        overlap=0.5,

    ) -> None :
        super().__init__()
        self.target_sr = target_sr
        self.expansion_factor = expansion_factor
        
        # Windowing parameters
        self.win_samples = int(window_sec * target_sr)
        self.hop_samples = int(self.win_samples * (1.0 - overlap))

    def load(self, file_path : str) -> Tuple[torch.Tensor, int]:
        """Loads audio using soundfile and applies 10x time expansion logic."""
        try:
            # Load using soundfile
            data, orig_sr = sf.read(file_path)

            # Convert to torch tensor [channels, time]
            audio = torch.from_numpy(data).float()
            # soundfile returns [time, channels] for stereo, so we transpose
            if audio.ndim == 1:
                audio = audio.unsqueeze(0) # Add channel dim for mono
            else:
                audio = audio.transpose(0, 1) # [T, C] -> [C, T]

            # Convert stereo to mono if necessary
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)

            return audio, orig_sr

        except Exception as e:
            print(f"\n[Warning] Skipping {file_path}: {e}")
            # Return 1 second of silence as a fallback to prevent NoneType errors
            return torch.zeros((1, self.target_sr)), self.target_sr
    
    def apply_time_expansion(self,audio: torch.Tensor, orig_sr: int) -> torch.Tensor:
        # Time Expansion logic
        virtual_sr = orig_sr // self.expansion_factor
        # Resample to target sample rate
        if virtual_sr != self.target_sr:
            audio = AF.resample(audio, orig_freq=virtual_sr, new_freq=self.target_sr)
        return audio

    def apply_bandpass(self, audio : torch.Tensor, orig_sr: int) -> torch.Tensor:
        """
        Applies a high-order Butterworth filter using SciPy for a 
        sharp cutoff before converting to PyTorch. Filters out low noise
        and 
        """
        # 1. Ensure orig_sr is an integer
        fs = float(orig_sr)
        # Convert torch tensor back to numpy temporarily for SciPy
        audio_np = audio.cpu().numpy()

        # 1.  8th-order Highpass at 15 kHz (48 dB/octave roll-off)
        sos_hp = butter(N=8, Wn=15000, btype='highpass', fs=fs, output='sos')
        audio_np = sosfilt(sos_hp, audio_np)

        return torch.from_numpy(audio_np).float()

    def window_audio(self, audio : torch.Tensor) -> torch.Tensor:
        """Cuts the 1D audio tensor into overlapping windows."""
        # Shape goes from [1, Total_Samples] -> [1, Num_Windows, Window_Samples]
        if audio.shape[1] < self.win_samples:
            # Pad with zeros if the file is shorter than window size
            pad_amount = self.win_samples - audio.shape[1]
            audio = torch.nn.functional.pad(audio, (0, pad_amount))
            
        windows = audio.unfold(-1, self.win_samples, self.hop_samples)
        # Remove channel dimension [Num_Windows, Window_Samples] 
        windows = windows.squeeze(0)
        return windows
    
    def forward(self, file_path : str,timeshift : bool = False) -> torch.Tensor:

        # 0. Load & Time Expand
        audio, orig_sr = self.load(file_path)
        # 1. Bandpass Filter
        audio = self.apply_bandpass(audio, orig_sr)
        #2. Time expand
        audio = self.apply_time_expansion(audio, orig_sr)
        # 4. Cut into Windows
        windows = self.window_audio(audio)
        
        return windows
    

# ==========================================
# 2. Augmentation Transforms and oversampling
# ==========================================
class WaveformMixup(nn.Module):
    """
    Blends two raw 1D audio waveforms and their corresponding multi-label targets.
    x_mixed = lambda * x1 + (1 - lambda) * x2
    y_mixed = lambda * y1 + (1 - lambda) * y2
    """
    def __init__(self, alpha: float = 0.2):
        super().__init__()
        self.alpha = alpha

    def forward(
        self, 
        audio1: torch.Tensor, 
        label1: torch.Tensor, 
        audio2: torch.Tensor, 
        label2: torch.Tensor
    ):
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample().item() if self.alpha > 0 else 1.0

        mixed_audio = lam * audio1 + (1 - lam) * audio2
        mixed_label = lam * label1 + (1 - lam) * label2

        return mixed_audio, mixed_label

class TimeMasking(nn.Module):
    """
    Time Masking by zeroing out contiguous chunks 
    of the raw 1D time domain signal.
    """
    def __init__(self, max_mask_ratio: float = 0.15, num_masks: int = 2):
        super().__init__()
        self.max_mask_ratio = max_mask_ratio
        self.num_masks = num_masks

    def forward(self, audio: torch.Tensor, target_sr: int = None) -> torch.Tensor:
        masked_audio = audio.clone()
        seq_len = audio.shape[-1]
        max_mask_len = int(seq_len * self.max_mask_ratio)

        for _ in range(self.num_masks):
            mask_len = random.randint(0, max_mask_len)
            if mask_len == 0:
                continue
            
            start = random.randint(0, seq_len - mask_len)
            masked_audio[..., start : start + mask_len] = 0.0

        return masked_audio

class FrequencyMasking(nn.Module):
    """
    Frequency Masking in raw audio by zeroing out a random
    frequency band using a SciPy bandstop (notch) filter.
    """
    def __init__(self, max_bandwidth_hz: float = 3000.0, num_masks: int = 1):
        super().__init__()
        self.max_bandwidth_hz = max_bandwidth_hz
        self.num_masks = num_masks

    def forward(self, audio: torch.Tensor, target_sr: int) -> torch.Tensor:
        nyquist = target_sr / 2.0
        audio_np = audio.cpu().numpy()

        for _ in range(self.num_masks):
            bandwidth = random.uniform(500.0, self.max_bandwidth_hz)
            # Pick a random center frequency within safe Nyquist boundaries
            f_low = random.uniform(100.0, nyquist - bandwidth - 100.0)
            f_high = f_low + bandwidth

            sos = butter(N=8, Wn=[f_low, f_high], btype='bandstop', fs=target_sr, output='sos')
            audio_np = sosfilt(sos, audio_np)

        return torch.from_numpy(audio_np).float().to(audio.device)

class TimeStretch(nn.Module):
    """
    Time stretches by resampling, then uses torchaudio PitchShift 
    to restore original frequencies.
    """
    def __init__(self, min_rate: float = 0.8, max_rate: float = 1.2):
        super().__init__()
        self.min_rate = min_rate
        self.max_rate = max_rate

    def forward(self, audio: torch.Tensor, target_sr: int) -> torch.Tensor:
        rate = random.uniform(self.min_rate, self.max_rate)
        if abs(rate - 1.0) < 1e-3:
            return audio

        orig_len = audio.shape[-1]
        virtual_sr = int(target_sr * rate)

        # 1. Resample (stretches time, but shifts pitch)
        stretched = AF.resample(audio, orig_freq=target_sr, new_freq=virtual_sr)

        # 2. Shift pitch back by -12 * log2(rate) semitones to restore true pitch
        n_steps = -12.0 * math.log2(rate)
        pitch_restored = AT.PitchShift(sample_rate=target_sr, n_steps=n_steps)(stretched)

        # 3. Trim or pad back to original window length
        if pitch_restored.shape[-1] < orig_len:
            pad = orig_len - pitch_restored.shape[-1]
            return torch.nn.functional.pad(pitch_restored, (0, pad))
        else:
            return pitch_restored[..., :orig_len]


class AudioCompose(nn.Module):
    """
    Applies a list of waveform augmentations to a batch/sequence of windows.
    Each window in the input tensor gets its own independent set of random transformations
    based on a per-transform probability.
    """
    def __init__(self, transforms_with_probs: List[Tuple[nn.Module, float]]):
        """
        Args:
            transforms_with_probs: A list of tuples pairing a transform instance 
                                   with its activation probability p (0.0 to 1.0).
                                   Example: [(TimeShift(), 0.5), (AddBackgroundNoise(...), 0.3)]
        """
        super().__init__()
        self.transforms = nn.ModuleList([t for t, _ in transforms_with_probs])
        self.probs = [p for _, p in transforms_with_probs]

    def forward(self, windows: torch.Tensor, target_sr: int) -> torch.Tensor:
        """
        Args:
            windows: Tensor of shape [Num_Windows, Window_Samples] or [Num_Windows, Channels, Window_Samples]
            target_sr: Target sample rate in Hz
        """
        # If no transforms are passed or tensor is empty, return original
        if len(self.transforms) == 0:
            return windows

        augmented_windows = []

        # Iterate over each window individually
        for i in range(windows.shape[0]):
            single_window = windows[i : i + 1]  # Preserve shape [1, Window_Samples]

            for transform, p in zip(self.transforms, self.probs):
                # Apply the specific transform on this specific window if random chance allows
                if random.random() < p:
                    # Check if the transform needs target_sr as an argument
                    try:
                        single_window = transform(single_window, target_sr=target_sr)
                    except TypeError:
                        single_window = transform(single_window)

            augmented_windows.append(single_window)

        # Re-stack into single tensor [Num_Windows, Window_Samples]
        return torch.cat(augmented_windows, dim=0)



def get_ir_per_label(self, y : np.ndarray) -> np.ndarray:
    """Calculates the Imbalance Ratio per Label (IRLBL)."""
    counts = np.sum(y, axis=0)
    max_count = np.max(counts)
    # Avoid division by zero for labels with 0 occurrences
    ir_per_label = max_count / (counts + 1e-9)
    return ir_per_label

def iterative_oversample(self, X, y, target_percentage=0.2,random_state = 42):
    """
    Randomly duplicates samples containing minority labels 
    until the distribution balances out.
    """
    rng = np.random.default_rng(random_state)
    # Convert to numpy for indexing if they aren't already
    X_resampled = list(X)
    y_resampled = list(y)
    current_counts = np.sum(y_resampled, axis=0)
    max_label_count = np.max(current_counts)
    # We want every label to at least reach a certain percentage 
    # of the majority label's count. 
    target_count = max_label_count * target_percentage
    # Indices of samples grouped by label for quick access
    label_to_indices = {i: np.where(y[:, i] == 1)[0] for i in range(y.shape[1])}
    # Keep adding samples until all labels meet the target
    balancing = True
    while balancing:
        ir_labels = get_ir_per_label(np.array(y_resampled))
        # Find labels that are still below target_count
        underrepresented_labels = np.where(np.sum(y_resampled, axis=0) < target_count)[0]
        if len(underrepresented_labels) == 0:
            balancing = False
            break
        # Focus on the most imbalanced label currently
        worst_label = underrepresented_labels[np.argmax(ir_labels[underrepresented_labels])]
        # Pick a random sample that contains this label
        possible_indices = label_to_indices[worst_label]
        if len(possible_indices) == 0:
            continue # Should not happen if label exists
        idx_to_clone = rng.choice(possible_indices)
        # Duplicate the sample
        y_resampled.append(y[idx_to_clone])
        X_resampled.append(X[idx_to_clone])
        # (Optional) Stop if we exceed a certain size to prevent infinite loops
        if len(y_resampled) > len(y) * 3:
            print("Reached safety limit (3x original size). Stopping.")
            break
    print(f"Final counts: {np.sum(y_resampled, axis=0).astype(int)}")
    return np.array(X_resampled), np.array(y_resampled)


# ==========================================
# 3. Dataset Class
# ==========================================

class BioacousticDataset(Dataset):
    """Custom Dataset class for loading and preprocessing pipistrelle bat audio recordings with optional data augmentation and resampling strategies.
    
    - data_input: Either a path to a CSV file containing metadata and labels or a pre-loaded pandas DataFrame.
    - root_dir: The base directory where audio files are located.  
    - noise_folder: Optional directory containing noise audio files for augmentation.
    - is_training: Whether the dataset is being used for training (enables augmentations).
    - resample: Whether to perform iterative oversampling to balance classes.
    - resample_augment: List of augmentations to apply only to the duplicated samples during resampling.
    - online_augment: List of augmentations to apply to all training samples during __getitem__.
    - time_shift: Whether to include time shifting as an augmentation option.
    - filter_echo: Whether to apply a bandreject filter to remove echolocation echoes during preprocessing.
    - overlap: The percentage of overlap between windows when segmenting audio.
    - encoder: The name of the encoder model for which the preprocessing pipeline should be optimized (e.g., 'perch2', 'effnetb0', 'NLM_BEATs').
    """
    def __init__(
        self, 
        data_input : str | pd.DataFrame, 
        root_dir : str, 
        is_training : bool =False,
        resample : bool = False, 
        resample_percentage : float = 0.2,
        augment : bool = False,
        augment_pipeline : Optional[AudioCompose] = None,
        overlap : float = 0.5,
        encoder : str ="perch2"
    ) -> None :
        
        self.root_dir = root_dir
        self.is_training = is_training

        #First element indicates if data augmentation should happen, other elements are the augmentations to apply during resampling
        self.augment = augment 
        self.augment_pipeline = augment_pipeline
    
        # Initialize processing pipeline
        if encoder == "perch2":
            self.pipeline = PipistrellePreprocessingPipeline(target_sr=32000, expansion_factor=5,window_sec =5,overlap=overlap)
        elif encoder == "effnetb0" or encoder == "NLM_BEATs":
            self.pipeline = PipistrellePreprocessingPipeline(target_sr=16000, expansion_factor=10,window_sec =10,overlap=overlap)
        else:
            raise ValueError(f"Unsupported encoder : {encoder}. Please choose from 'perch2', 'effnetb0', or 'NLM_BEATs'.")

        # 1. Load raw data
        if isinstance(data_input, str):
            df = pd.read_csv(data_input)
        elif isinstance(data_input, pd.DataFrame):
            df = data_input # Assume it's a DataFrame
        else :
            raise ValueError("data_input must be either a path to a CSV file or a pandas DataFrame.")

        # 2. Extract relative paths and labels of data
        label_cols = ['type_a', 'type_b', 'type_c', 'type_d', 'echo']
        X_paths = df['relative_path'].values
        y_labels = df[label_cols].values

        # 3. Resample data
        self.original_len = len(X_paths)
        if self.is_training and resample: 
            self.X, self.y = iterative_oversample(X_paths, y_labels,resample_percentage=resample_percentage)
        else:
            self.X, self.y = X_paths, y_labels


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir, self.X[idx])
        labels = torch.tensor(self.y[idx], dtype=torch.float32)
        
        
        # Run the entire audio preprocessing and augmentation pipeline
        # 1. Grab the raw audio windows [Num_Windows, 160000]
        windows = self.pipeline.forward(file_path)

        if self.is_training and self.augment:  # If augmentation is enabled
            windows = self.augment_pipeline(windows)

        return windows, labels
    