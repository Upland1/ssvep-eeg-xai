"""Signal plotting utilities for EEG time windows and spectra."""

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
    
def plot_stimulus_vs_baseline_psd(
    condition_records: dict,
    channel_names: list[str],
    stim_cond: int = 202,
    baseline_cond: int = 201,
    target_freq_range: tuple[float, float] = (1.0, 40.0),
    output_path: str | None = None,
):
    """Plot average PSD of the stimulus condition overlaid with the 201 cross baseline."""
    fig, axes = plt.subplots(len(channel_names), 1, figsize=(12, 2.5 * len(channel_names)), sharex=True)
    if len(channel_names) == 1:
        axes = [axes]

    baseline_psd = np.array([r["psd"] for r in condition_records.get(baseline_cond, [])])
    stim_psd = np.array([r["psd"] for r in condition_records.get(stim_cond, [])])

    freqs = condition_records[stim_cond][0]["freqs"] if stim_psd.size > 0 else condition_records[baseline_cond][0]["freqs"]
    mask = (freqs >= target_freq_range[0]) & (freqs <= target_freq_range[1])

    avg_base = np.mean(baseline_psd, axis=0) if baseline_psd.size > 0 else None
    avg_stim = np.mean(stim_psd, axis=0) if stim_psd.size > 0 else None

    for i, ch_name in enumerate(channel_names):
        if avg_base is not None:
            axes[i].plot(freqs[mask], 10 * np.log10(avg_base[i, mask] + 1e-12), label=f"Cross 201 Baseline (N={len(baseline_psd)})", color="gray", linestyle="--")
        if avg_stim is not None:
            axes[i].plot(freqs[mask], 10 * np.log10(avg_stim[i, mask] + 1e-12), label=f"Stimulus {stim_cond} (N={len(stim_psd)})", color="crimson", linewidth=1.8)

        axes[i].set_ylabel(f"{ch_name}\n(dB/Hz)")
        axes[i].legend(loc="upper right")
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Frequency (Hz)")
    axes[0].set_title(f"Spectral Comparison: Stimulus {stim_cond} vs Pre-Stimulus Fixation Cross {baseline_cond}")
    fig.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
