"""Power spectral density extraction for condition windows."""

from __future__ import annotations

import numpy as np
from scipy.signal import periodogram


def process_condition_windows(
    eeg_data: np.ndarray,
    mark_channel: np.ndarray,
    fs: float,
    target_conditions: list[int] | tuple[int, ...] = (101, 102, 103, 104),
    sub_window_sec: float = 1.0,
    vpp_thresh: float = 200.0,
    std_range: tuple[float, float] = (0.5, 60.0),
):
    """Split condition recordings into valid subwindows and compute PSD in RAM."""
    samples_per_subwin = int(sub_window_sec * fs)
    diff_marks = np.diff(mark_channel, prepend=0)
    condition_records = {cond: [] for cond in target_conditions}

    for cond in target_conditions:
        event_indices = np.where((mark_channel == cond) & (diff_marks != 0))[0]

        if len(event_indices) == 0:
            event_indices = np.where(mark_channel == cond)[0]
            if len(event_indices) == 0:
                continue
            splits = np.where(np.diff(event_indices) > 1)[0] + 1
            blocks = np.split(event_indices, splits)
        else:
            blocks = [np.arange(idx, min(idx + int(5.0 * fs), eeg_data.shape[1])) for idx in event_indices]

        for block in blocks:
            n_sub = len(block) // samples_per_subwin
            for s in range(n_sub):
                start = block[s * samples_per_subwin]
                end = start + samples_per_subwin
                if end > eeg_data.shape[1]:
                    break

                sub_win = eeg_data[:, start:end]
                vpp = np.ptp(sub_win, axis=-1)
                std_dev = np.std(sub_win, axis=-1)

                if np.any(vpp > vpp_thresh) or np.any(std_dev < std_range[0]) or np.any(std_dev > std_range[1]):
                    continue

                freqs, psd = periodogram(sub_win, fs=fs, axis=-1)
                condition_records[cond].append({
                    "vpp": vpp,
                    "std": std_dev,
                    "psd": psd,
                    "freqs": freqs,
                })

    return condition_records


def prepare_feature_matrix(condition_records, target_conditions=(101, 102, 103, 104), freq_range=(5.0, 35.0)):
    """Flatten selected PSD features across channels and frequencies."""
    X = []
    y = []
    label_map = {cond: idx for idx, cond in enumerate(target_conditions)}

    for cond in target_conditions:
        records = condition_records[cond]
        for item in records:
            psd = item["psd"]
            freqs = item["freqs"]
            freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
            psd_selected = psd[:, freq_mask]
            X.append(psd_selected.flatten())
            y.append(label_map[cond])

    X = np.array(X)
    y = np.array(y)
    return X, y
