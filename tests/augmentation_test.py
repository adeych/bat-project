"""
Augmentation Visualizer for the Pipistrelle preprocessing pipeline.

Lets you:
  - grab a random (or specific) audio window via your PipistrellePreprocessingPipeline
  - apply any single transform (TimeMasking, FrequencyMasking, TimeStretch, WaveformMixup)
    or your full AudioCompose pipeline, and see before/after spectrograms + waveforms
  - "reroll the dice": every call to a transform re-triggers its internal randomness
    (they use `random.random()`/`random.uniform()`/Beta sampling inside forward()),
    so just re-running `viz.apply(transform)` on the SAME cached window gives you a
    fresh random draw of that transform. Use `viz.new_sample()` to grab a fresh
    audio window instead.

Usage (in a notebook):

    from your_module import PipistrellePreprocessingPipeline, TimeMasking, FrequencyMasking, TimeStretch, WaveformMixup
    from augmentation_visualizer import AugmentationVisualizer

    pipeline = PipistrellePreprocessingPipeline(target_sr=32000, expansion_factor=5, window_sec=5, overlap=0.5)
    audio_paths = [os.path.join(root_dir, p) for p in df['relative_path']]

    viz = AugmentationVisualizer(pipeline, audio_paths, target_sr=32000)

    viz.new_sample()                       # picks a random file + random window, plots it
    viz.apply(TimeMasking())               # apply + plot; rerun this cell to reroll the transform
    viz.apply(FrequencyMasking())
    viz.apply(TimeStretch())

    viz.compare_grid({                     # side-by-side grid of several transforms at once
        "TimeMasking": TimeMasking(),
        "FrequencyMasking": FrequencyMasking(),
        "TimeStretch": TimeStretch(),
    })

    viz.apply_mixup(WaveformMixup(alpha=0.2))   # blends current sample with a second random sample

    viz.apply(your_audio_compose_pipeline)  # test the full AudioCompose chain end-to-end
"""

import random
from typing import Callable, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import torch
import torchaudio.transforms as AT


class AugmentationVisualizer:
    def __init__(self, pipeline, audio_paths: Sequence[str], target_sr: int,
                 n_fft: int = 1024, hop_length: int = 256):
        """
        pipeline: an instance of PipistrellePreprocessingPipeline (already configured
                  with the target_sr/window_sec/overlap you want to test with)
        audio_paths: full file paths to sample from
        target_sr: sample rate your pipeline outputs (must match what your transforms expect)
        """
        self.pipeline = pipeline
        self.audio_paths = list(audio_paths)
        self.target_sr = target_sr
        self.spec_fn = AT.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=2)
        self.db_fn = AT.AmplitudeToDB()

        self.current_path: Optional[str] = None
        self.current_windows: Optional[torch.Tensor] = None
        self.current_window: Optional[torch.Tensor] = None  # shape [1, win_samples]

    # ------------------------------------------------------------------
    # Sample selection
    # ------------------------------------------------------------------
    def new_sample(self, path: Optional[str] = None, window_idx: Optional[int] = None,
                    plot: bool = True) -> torch.Tensor:
        """Grab a fresh audio file (random unless specified) and a window from it."""
        self.current_path = path or random.choice(self.audio_paths)
        self.current_windows = self.pipeline.forward(self.current_path)

        n_windows = self.current_windows.shape[0]
        idx = window_idx if window_idx is not None else random.randrange(n_windows)
        self.current_window = self.current_windows[idx : idx + 1]

        print(f"[sample] {self.current_path}  (window {idx}/{n_windows - 1})")
        if plot:
            self._plot_pair(self.current_window, self.current_window,
                             "Original", "Original")
        return self.current_window

    def _ensure_sample(self):
        if self.current_window is None:
            self.new_sample()

    # ------------------------------------------------------------------
    # Applying transforms
    # ------------------------------------------------------------------
    def apply(self, transform: Callable, name: Optional[str] = None, **kwargs) -> torch.Tensor:
        """
        Apply any single-input transform (or your AudioCompose) to the CURRENT window
        and plot before/after. Call again to reroll the transform's randomness on the
        same underlying sample.
        """
        self._ensure_sample()
        label = name or type(transform).__name__

        out = self._call_transform(transform, self.current_window, **kwargs)

        self._plot_pair(self.current_window, out, "Before", label)
        return out

    def apply_mixup(self, mixup_transform, other_path: Optional[str] = None,
                     other_window_idx: Optional[int] = None) -> torch.Tensor:
        """
        WaveformMixup needs a second (audio, label) pair. This blends the current
        sample with a second random (or specified) sample and plots all three.
        Labels are dummy zeros here since this is purely for visual inspection.
        """
        self._ensure_sample()
        other_path = other_path or random.choice(self.audio_paths)
        other_windows = self.pipeline.forward(other_path)
        idx = other_window_idx if other_window_idx is not None else random.randrange(other_windows.shape[0])
        other_window = other_windows[idx : idx + 1]

        dummy_label = torch.zeros(1)
        mixed, _ = mixup_transform(self.current_window, dummy_label, other_window, dummy_label)

        print(f"[mixup partner] {other_path} (window {idx})")
        fig, axes = plt.subplots(2, 3, figsize=(15, 6))
        self._spec(axes[0, 0], self.current_window, "Sample A (current)")
        self._spec(axes[0, 1], other_window, "Sample B (random)")
        self._spec(axes[0, 2], mixed, "Mixed")
        self._wave(axes[1, 0], self.current_window)
        self._wave(axes[1, 1], other_window)
        self._wave(axes[1, 2], mixed)
        plt.tight_layout()
        plt.show()
        return mixed

    def compare_grid(self, transforms: Dict[str, Callable], reroll_each: bool = True):
        """
        Apply several transforms to the SAME current window and show all spectrograms
        in one grid, for quick side-by-side comparison. Set reroll_each=False if you
        pass already-applied (deterministic) outputs you want to just plot.
        """
        self._ensure_sample()
        n = len(transforms) + 1
        cols = min(n, 4)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
        axes = axes.flatten() if n > 1 else [axes]

        self._spec(axes[0], self.current_window, "Original")

        for ax, (label, transform) in zip(axes[1:], transforms.items()):
            out = self._call_transform(transform, self.current_window) if reroll_each else transform
            self._spec(ax, out, label)

        for ax in axes[n:]:
            ax.axis("off")

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _call_transform(self, transform: Callable, window: torch.Tensor, **kwargs):
        try:
            return transform(window, target_sr=self.target_sr, **kwargs)
        except TypeError:
            return transform(window, **kwargs)

    def _spec(self, ax, audio: torch.Tensor, title: str):
        spec = self.spec_fn(audio.squeeze(0))
        spec_db = self.db_fn(spec)
        ax.imshow(spec_db.numpy(), origin="lower", aspect="auto", cmap="magma")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Freq bin")

    def _wave(self, ax, audio: torch.Tensor):
        ax.plot(audio.squeeze().numpy(), linewidth=0.4)
        ax.set_title("Waveform", fontsize=10)
        ax.set_xlim(0, audio.shape[-1])

    def _plot_pair(self, before: torch.Tensor, after: torch.Tensor,
                    label_before: str, label_after: str):
        fig, axes = plt.subplots(2, 2, figsize=(11, 6))
        self._spec(axes[0, 0], before, label_before)
        self._spec(axes[0, 1], after, label_after)
        self._wave(axes[1, 0], before)
        self._wave(axes[1, 1], after)
        plt.tight_layout()
        plt.show()