"""Validate EEG windows with channel- and window-level quality checks."""

import numpy as np


def summarize_window_quality(
    windows: np.ndarray,
    vpp_min: float = 5.0,
    vpp_max: float = 120.0,
    std_min: float = 1.0,
    std_max: float = 35.0,
) -> None:
    """Print Vpp/std statistics alongside the active thresholds."""
    if windows.ndim != 3:
        raise ValueError("windows must have shape (n_windows, n_channels, n_samples)")

    vpp = np.ptp(windows, axis=-1)
    std = np.std(windows, axis=-1)

    print(f"Windows: {windows.shape[0]} | Channels: {windows.shape[1]}")
    print(f"Vpp  observed: min={vpp.min():.3f} max={vpp.max():.3f} mean={vpp.mean():.3f} | threshold=[{vpp_min}, {vpp_max}]")
    print(f"Std  observed: min={std.min():.3f} max={std.max():.3f} mean={std.mean():.3f} | threshold=[{std_min}, {std_max}]")
    vpp_fail = ((vpp < vpp_min) | (vpp > vpp_max)).mean() * 100.0
    std_fail = ((std < std_min) | (std > std_max)).mean() * 100.0
    print(f"Channel-window entries failing Vpp bound: {vpp_fail:.1f}% | failing std bound: {std_fail:.1f}%")


def validate_eeg_windows(
    windows: np.ndarray,
    vpp_min: float = 5.0,
    vpp_max: float = 120.0,
    std_min: float = 1.0,
    std_max: float = 35.0,
) -> np.ndarray:
    """Return a mask for windows where every channel meets the limits."""
    if windows.ndim != 3:
        raise ValueError("windows must have shape (n_windows, n_channels, n_samples)")

    vpp_per_channel = np.ptp(windows, axis=-1)
    std_per_channel = np.std(windows, axis=-1)
    vpp_ok = np.all((vpp_per_channel >= vpp_min) & (vpp_per_channel <= vpp_max), axis=1)
    std_ok = np.all((std_per_channel >= std_min) & (std_per_channel <= std_max), axis=1)
    return (vpp_ok & std_ok).astype(int)


def filter_dataset(X: np.ndarray, y: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop invalid windows from a data matrix and its labels."""
    if X.shape[0] != y.shape[0] or X.shape[0] != mask.shape[0]:
        raise ValueError("X, y, and mask must contain the same number of windows")
    valid_idx = np.flatnonzero(mask == 1)
    return X[valid_idx], y[valid_idx]


def compute_channel_quality_stats(
    windows: np.ndarray,
    vpp_min: float = 5.0,
    vpp_max: float = 120.0,
    std_min: float = 1.0,
    std_max: float = 35.0,
) -> dict:
    """Return per-channel failure rates and mean signal statistics."""
    if windows.ndim != 3:
        raise ValueError("windows must have shape (n_windows, n_channels, n_samples)")

    vpp_per_channel = np.ptp(windows, axis=-1)
    std_per_channel = np.std(windows, axis=-1)

    channel_ok = (
        (vpp_per_channel >= vpp_min) & (vpp_per_channel <= vpp_max)
        & (std_per_channel >= std_min) & (std_per_channel <= std_max)
    )

    fail_fraction = 1.0 - channel_ok.mean(axis=0)

    return {
        "fail_fraction": fail_fraction,
        "mean_vpp": vpp_per_channel.mean(axis=0),
        "mean_std": std_per_channel.mean(axis=0),
    }


def identify_noisy_channels(
    windows: np.ndarray,
    channel_names: list[str] | None = None,
    vpp_min: float = 5.0,
    vpp_max: float = 120.0,
    std_min: float = 1.0,
    std_max: float = 35.0,
    fail_fraction_thresh: float = 0.5,
) -> tuple[list[int], dict]:
    """Return channels that fail the limits too often."""
    stats = compute_channel_quality_stats(
        windows, vpp_min=vpp_min, vpp_max=vpp_max, std_min=std_min, std_max=std_max
    )
    noisy_idx = [
        i for i, frac in enumerate(stats["fail_fraction"]) if frac > fail_fraction_thresh
    ]

    if channel_names is not None:
        if len(channel_names) != windows.shape[1]:
            raise ValueError(
                "channel_names length must match windows' channel axis "
                f"({len(channel_names)} names vs {windows.shape[1]} channels)"
            )
        stats["channel_names"] = list(channel_names)
        stats["noisy_channel_names"] = [channel_names[i] for i in noisy_idx]

    return noisy_idx, stats


def drop_noisy_channels(
    windows: np.ndarray,
    noisy_idx: list[int],
    channel_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str] | None, list[int]]:
    """Remove flagged channels and return the remaining indices and names."""
    if windows.ndim != 3:
        raise ValueError("windows must have shape (n_windows, n_channels, n_samples)")

    noisy_set = set(noisy_idx)
    keep_idx = [i for i in range(windows.shape[1]) if i not in noisy_set]
    if len(keep_idx) == 0:
        raise ValueError("All channels were flagged as noisy; nothing left to keep.")

    windows_clean = windows[:, keep_idx, :]
    remaining_names = [channel_names[i] for i in keep_idx] if channel_names is not None else None
    return windows_clean, remaining_names, keep_idx


def apply_artifact_quality_pipeline(
    windows: np.ndarray,
    y: np.ndarray,
    channel_names: list[str] | None = None,
    vpp_min: float = 5.0,
    vpp_max: float = 120.0,
    std_min: float = 1.0,
    std_max: float = 35.0,
    channel_fail_fraction_thresh: float = 0.5,
    verbose: bool = True,
    precomputed: dict | None = None,
) -> dict:
    """Apply channel- and window-level quality checks for one subject."""
    if precomputed is not None:
        valid_mask = np.asarray(precomputed["valid_mask"]).astype(int)
        if len(valid_mask) != len(y) or len(valid_mask) != windows.shape[0]:
            raise ValueError(
                f"precomputed valid_mask has {len(valid_mask)} entries for {windows.shape[0]} windows"
            )
        kept_idx = list(precomputed["channels_kept_idx"])
        windows_clean, y_clean = filter_dataset(windows[:, kept_idx, :], y, valid_mask)
        if verbose:
            print(f"[Shared QC] {len(kept_idx)}/{windows.shape[1]} channels kept | "
                  f"{int(valid_mask.sum())}/{len(valid_mask)} windows accepted.")
        return {
            "windows_clean": windows_clean,
            "y_clean": y_clean,
            "valid_mask": valid_mask,
            "channels_kept": precomputed.get("channels_kept"),
            "channels_kept_idx": kept_idx,
            "channels_dropped": precomputed.get("channels_dropped"),
            "channel_stats": precomputed.get("channel_stats", {}),
        }

    # Remove persistently bad channels.
    noisy_idx, channel_stats = identify_noisy_channels(
        windows,
        channel_names=channel_names,
        vpp_min=vpp_min,
        vpp_max=vpp_max,
        std_min=std_min,
        std_max=std_max,
        fail_fraction_thresh=channel_fail_fraction_thresh,
    )
    windows_ch, channels_kept, channels_kept_idx = drop_noisy_channels(
        windows, noisy_idx, channel_names
    )
    channels_dropped = channel_stats.get("noisy_channel_names")

    if verbose:
        n_total = windows.shape[1]
        n_kept = windows_ch.shape[1]
        dropped_label = channels_dropped if channels_dropped is not None else noisy_idx
        print(
            f"[Channel QC] {n_kept}/{n_total} channels kept. "
            f"Dropped: {dropped_label if dropped_label else 'none'}"
        )

    # Remove invalid windows.
    valid_mask = validate_eeg_windows(
        windows_ch, vpp_min=vpp_min, vpp_max=vpp_max, std_min=std_min, std_max=std_max
    )
    windows_clean, y_clean = filter_dataset(windows_ch, y, valid_mask)

    if verbose:
        n_win = len(valid_mask)
        n_acc = int(valid_mask.sum())
        print(f"[Window QC] {n_acc}/{n_win} windows accepted ({n_win - n_acc} rejected).")

    return {
        "windows_clean": windows_clean,
        "y_clean": y_clean,
        "valid_mask": valid_mask,
        "channels_kept": channels_kept,
        "channels_kept_idx": channels_kept_idx,
        "channels_dropped": channels_dropped,
        "channel_stats": channel_stats,
    }


# Legacy helpers for per-trial sub-window extraction.

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