"""Signal plotting utilities for EEG time windows and spectra."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np


def plot_eeg_window(win_data: np.ndarray, fs: float, channel_names: list[str], meta: dict, offset: float = 50.0, output_path: str | None = None):
    """Plot the EEG window and mark whether it passed quality checks."""
    n_channels, samples = win_data.shape
    time_vector = np.arange(samples) / fs + meta["start_time"]

    plt.figure(figsize=(12, 6))
    for ch in range(n_channels):
        plt.plot(time_vector, win_data[ch, :] + (ch * offset), label=channel_names[ch])

    status = "NOISE / ARTIFACT" if meta["is_noise"] else "CLEAN"
    color = "red" if meta["is_noise"] else "green"
    plt.title(
        f"Window {meta['window_id']} [{meta['start_time']:.1f}s - {meta['end_time']:.1f}s] | Status: {status} (Max Vp-p: {meta['max_vpp']:.2f} uV)",
        color=color,
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude + Offset (uV)")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=150)
    plt.close()


def plot_average_psd_by_condition(condition_records, channel_names: list[str], target_freq_range=(1.0, 60.0), output_path: str = "outputs/figures/spectra/average_psd_by_condition.png"):
    """Create PSD spectra by condition and channel for multi-condition comparison."""
    fig, axes = plt.subplots(len(channel_names), 1, figsize=(14, max(7, 2.5 * len(channel_names))), sharex=True, squeeze=False)
    axes = axes[:, 0]

    for cond, records in condition_records.items():
        if len(records) == 0:
            continue
        all_psd = np.array([r["psd"] for r in records])
        freqs = records[0]["freqs"]
        avg_psd_by_channel = np.mean(all_psd, axis=0)
        mask = (freqs >= target_freq_range[0]) & (freqs <= target_freq_range[1])

        for ch_index, channel_name in enumerate(channel_names):
            spectrum_db = 10 * np.log10(avg_psd_by_channel[ch_index, mask] + 1e-12)
            axes[ch_index].plot(freqs[mask], spectrum_db, label=f"Condition {cond} (N={len(records)} wins)", linewidth=2)

    for ch_index, channel_name in enumerate(channel_names):
        axes[ch_index].set_ylabel(f"{channel_name}\nPSD (dB/Hz)")
        axes[ch_index].grid(True, alpha=0.3)
        axes[ch_index].legend(loc="upper right")

    axes[0].set_title("Average PSD by Condition and EEG Channel")
    axes[-1].set_xlabel("Frequency (Hz)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
