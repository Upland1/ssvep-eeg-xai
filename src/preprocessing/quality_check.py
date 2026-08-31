"""Artifact validation utilities for EEG windows."""

from __future__ import annotations

import numpy as np


def window_quality_flags(signal: np.ndarray, vpp_thresh: float = 200.0, std_range: tuple[float, float] = (0.5, 60.0)) -> tuple[np.ndarray, np.ndarray]:
    """Return per-channel Vp-p and std metrics and a binary noise mask."""
    vpp_per_ch = np.ptp(signal, axis=-1)
    std_per_ch = np.std(signal, axis=-1)
    is_noise = np.any(vpp_per_ch > vpp_thresh) or np.any(std_per_ch < std_range[0]) or np.any(std_per_ch > std_range[1])
    return vpp_per_ch, std_per_ch, is_noise


def segment_and_validate_eeg(signal: np.ndarray, fs: float, win_sec: float = 5.0, vpp_thresh: float = 200.0, std_range: tuple[float, float] = (0.5, 60.0)) -> tuple[list[np.ndarray], list[dict]]:
    """Split the EEG into fixed-length windows and mark noisy segments."""
    n_channels, n_samples = signal.shape
    samples_per_win = int(win_sec * fs)
    n_windows = n_samples // samples_per_win

    windows_data = []
    window_metadata = []

    for w in range(n_windows):
        start_idx = w * samples_per_win
        end_idx = start_idx + samples_per_win
        win_data = signal[:, start_idx:end_idx]

        vpp_per_ch, std_per_ch, is_noise = window_quality_flags(win_data, vpp_thresh=vpp_thresh, std_range=std_range)

        windows_data.append(win_data)
        window_metadata.append(
            {
                "window_id": w,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "start_time": start_idx / fs,
                "end_time": end_idx / fs,
                "is_noise": bool(is_noise),
                "max_vpp": float(np.max(vpp_per_ch)),
                "max_std": float(np.max(std_per_ch)),
            }
        )

    return windows_data, window_metadata
