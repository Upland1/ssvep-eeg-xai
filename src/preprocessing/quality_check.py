"""Artifact validation utilities for EEG windows.

Two-tier artifact quality pipeline
-----------------------------------
Tier 1 - Channel level:
    Every channel is passed through unfiltered until this module runs (no
    channel is dropped upstream during acquisition or segmentation). Here,
    `identify_noisy_channels` looks across *all* windows of a subject's
    recording and flags an electrode as chronically bad ("red channel") if
    it fails the Vpp/std limits too often. `drop_noisy_channels` then
    removes that electrode from the channel axis, for every window, so a
    single dead/noisy electrode does not contaminate the whole montage.

Tier 2 - Window level:
    On the surviving channels only, `validate_eeg_windows` rejects
    individual windows that still violate the same Vpp/std limits (e.g. a
    good electrode picking up a one-off movement artifact).

`apply_artifact_quality_pipeline` runs both tiers in the correct order and
returns the cleaned data plus a report of what was dropped and why.
"""

import numpy as np


def summarize_window_quality(
    windows: np.ndarray,
    vpp_min: float = 5.0,
    vpp_max: float = 120.0,
    std_min: float = 1.0,
    std_max: float = 35.0,
) -> None:
    """Print the actual Vpp/std distribution seen in `windows` next to the
    active thresholds -- run this on a small raw sample when a quality gate
    rejects everything, to see at a glance which bound is too strict for
    your device's actual signal scale, instead of guessing.
    """
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
    """Return a binary mask for windows whose channels meet EEG limits.

    A window is accepted (1) only if EVERY channel it contains falls within
    [vpp_min, vpp_max] peak-to-peak and [std_min, std_max] standard
    deviation. This is meant to run AFTER chronically bad channels have
    already been removed via `identify_noisy_channels` / `drop_noisy_channels`
    -- otherwise a single dead electrode would force every window in the
    recording to be rejected, even if the other channels are clean.

    Parameters
    ----------
    windows : np.ndarray, shape (n_windows, n_channels, n_samples)
    vpp_min, vpp_max : float
        Acceptable peak-to-peak voltage range (same units as `windows`,
        typically microvolts). Default lower bound raised to 5.0 uV so
        near-flat/disconnected channels are caught.
    std_min, std_max : float
        Acceptable standard-deviation range. Default lower bound raised to
        1.0 for the same reason.

    Returns
    -------
    np.ndarray of int, shape (n_windows,)
        1 = window accepted, 0 = window rejected.
    """
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
    """Compute, per channel, the fraction of windows that fail the quality limits.

    Runs BEFORE any window is discarded: it looks at every channel across
    every window to find electrodes that are bad most of the time (chronic
    fault), as opposed to a channel that is only occasionally noisy (which
    is handled later, per-window, by `validate_eeg_windows`).

    Parameters
    ----------
    windows : np.ndarray, shape (n_windows, n_channels, n_samples)

    Returns
    -------
    dict with:
      "fail_fraction" : np.ndarray, shape (n_channels,)
          Fraction of windows (0.0-1.0) in which that single channel falls
          outside [vpp_min, vpp_max] or [std_min, std_max].
      "mean_vpp" : np.ndarray, shape (n_channels,)
      "mean_std" : np.ndarray, shape (n_channels,)
    """
    if windows.ndim != 3:
        raise ValueError("windows must have shape (n_windows, n_channels, n_samples)")

    vpp_per_channel = np.ptp(windows, axis=-1)   # (n_windows, n_channels)
    std_per_channel = np.std(windows, axis=-1)   # (n_windows, n_channels)

    channel_ok = (
        (vpp_per_channel >= vpp_min) & (vpp_per_channel <= vpp_max)
        & (std_per_channel >= std_min) & (std_per_channel <= std_max)
    )  # (n_windows, n_channels), True = this channel is fine in this window

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
    """Flag electrodes that are bad often enough to drop entirely ("red channel").

    A channel is flagged noisy/dead if it fails the Vpp/std limits in more
    than `fail_fraction_thresh` of all windows (default: more than half the
    recording). This is deliberately a subject-level, whole-channel
    decision, so the channel count stays fixed across a subject's windows
    -- FBCSP, FBCCA and EEGNet all assume a constant number of channels.

    Parameters
    ----------
    windows : np.ndarray, shape (n_windows, n_channels, n_samples)
    channel_names : list[str], optional
        Channel labels aligned with the channel axis (e.g. ["PO3", "POz",
        "PO4", "PO8", "O1", "Oz", "O2"]). If given, names are attached to
        the returned stats for reporting.
    fail_fraction_thresh : float
        Fraction of windows a channel must fail before it is flagged.
        0.5 means "bad in more than half the recording".

    Returns
    -------
    noisy_idx : list[int]
        Channel-axis indices flagged as chronically bad, sorted ascending.
    stats : dict
        Output of `compute_channel_quality_stats`, plus "channel_names" and
        "noisy_channel_names" if `channel_names` was provided.
    """
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
    """Remove flagged channels from the channel axis of `windows`.

    Parameters
    ----------
    windows : np.ndarray, shape (n_windows, n_channels, n_samples)
    noisy_idx : list[int]
        Channel indices to remove (from `identify_noisy_channels`).
    channel_names : list[str], optional
        Channel labels aligned with `windows`' channel axis.

    Returns
    -------
    windows_clean : np.ndarray, shape (n_windows, n_channels - len(noisy_idx), n_samples)
    remaining_names : list[str] or None
        Names of the channels that were kept, in their original order.
    keep_idx : list[int]
        Original channel-axis indices that were kept, in order. Callers
        that already have a full-channel array elsewhere (e.g. the raw
        `windows` before feature extraction) can reuse this to slice it
        consistently: `windows[:, keep_idx, :]`.
    """
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
) -> dict:
    """Run the full two-tier artifact-quality pipeline for one subject.

    Tier 1 (channel level): all channels enter this function unfiltered.
    Electrodes that are chronically bad (flat/dead or overly noisy in more
    than `channel_fail_fraction_thresh` of windows) are identified and
    dropped for this subject's entire recording.

    Tier 2 (window level): on the surviving channels only, individual
    windows that still violate the Vpp/std limits are rejected. This is the
    existing `validate_eeg_windows` behaviour, applied after bad channels
    are removed so one dead electrode can no longer zero out every window.

    Parameters
    ----------
    windows : np.ndarray, shape (n_windows, n_channels, n_samples)
    y : np.ndarray, shape (n_windows,)
        Labels aligned with `windows`.
    channel_names : list[str], optional
    vpp_min, vpp_max, std_min, std_max : float
        Shared thresholds for both the channel-level and window-level
        checks. Defaults: Vpp in [5, 120], std in [1, 35].
    channel_fail_fraction_thresh : float
        Threshold used only for the channel-level (Tier 1) decision.
    verbose : bool
        If True, print a short QC report.

    Returns
    -------
    dict with:
      "windows_clean"     : np.ndarray, shape (n_clean, n_channels_kept, n_samples)
      "y_clean"           : np.ndarray, shape (n_clean,)
      "valid_mask"        : np.ndarray, shape (n_windows,)
          Window-level accept/reject mask, computed on the channel-reduced
          array (i.e. aligned with `windows`/`y`, before window filtering).
      "channels_kept"     : list[str] or None
      "channels_kept_idx" : list[int]
          Original channel-axis indices that were kept. Use this to slice
          any other full-channel array (e.g. the raw windows) consistently.
      "channels_dropped"  : list[str] or None
      "channel_stats"     : dict (fail_fraction / mean_vpp / mean_std per
                             original channel, from Tier 1)
    """
    # --- Tier 1: channel-level quality, computed on the FULL, unfiltered
    # channel set (no channel has been dropped yet at this point). ---
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

    # --- Tier 2: window-level quality, on the surviving channels only. ---
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


# ---------------------------------------------------------------------------
# Legacy helpers (unchanged) -- still used by the preprocessing pipeline for
# per-trial baseline/cue sub-window extraction.
# ---------------------------------------------------------------------------

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
