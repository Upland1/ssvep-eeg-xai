"""Power spectral density extraction with relative stimulus-baseline condition support."""
import numpy as np
from scipy.signal import periodogram

from src.preprocessing.quality_check import apply_artifact_quality_pipeline, validate_eeg_windows


def extract_continuous_psd_features(
    windows: np.ndarray,
    y: np.ndarray,
    fs: float = 250.0,
    freq_range: tuple[float, float] = (5.0, 35.0),
    use_db: bool = False,
    channel_names: list[str] | None = None,
    vpp_limits: tuple[float, float] = (5.0, 120.0),
    std_limits: tuple[float, float] = (1.0, 35.0),
    channel_fail_fraction_thresh: float = 0.5,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Validate windows (channel-level + window-level) and return flattened PSD features.

    Same two-tier artifact-quality pipeline as the FBCCA/FBCSP extractors:
    chronically bad channels are dropped for this subject first, then
    remaining noisy windows are rejected on the surviving channels. PSD is
    computed on the clean, channel-reduced data.

    Returns
    -------
    X_psd : np.ndarray, shape (n_clean, n_channels_kept * n_freq_bins)
    y_clean : np.ndarray, shape (n_clean,)
    valid_mask : np.ndarray, shape (n_windows,)
        Aligned with the ORIGINAL `windows` passed in.
    channel_report : dict
        See `extract_fbcca_features` docstring -- same structure.
    """
    qc = apply_artifact_quality_pipeline(
        windows,
        y,
        channel_names=channel_names,
        vpp_min=vpp_limits[0],
        vpp_max=vpp_limits[1],
        std_min=std_limits[0],
        std_max=std_limits[1],
        channel_fail_fraction_thresh=channel_fail_fraction_thresh,
        verbose=verbose,
    )
    clean_windows, y_clean = qc["windows_clean"], qc["y_clean"]

    if clean_windows.shape[0] == 0:
        raise ValueError("All windows were rejected by the validation mask.")

    freqs, psd = periodogram(clean_windows, fs=fs, axis=-1)
    freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    psd_band = psd[:, :, freq_mask]

    if use_db:
        psd_band = 10.0 * np.log10(psd_band + 1e-12)

    n_clean, n_channels, n_bins = psd_band.shape
    X_psd = psd_band.reshape(n_clean, n_channels * n_bins)
    channel_report = {
        "channels_kept": qc["channels_kept"],
        "channels_kept_idx": qc["channels_kept_idx"],
        "channels_dropped": qc["channels_dropped"],
        "channel_stats": qc["channel_stats"],
    }
    return X_psd, y_clean, qc["valid_mask"], channel_report


def extract_clean_subwindows_psd(
    eeg_segment: np.ndarray,
    fs: float,
    sub_window_sec: float = 1.0,
    vpp_thresh: float = 200.0,
    std_range: tuple[float, float] = (0.5, 60.0),
    quality_stats: dict[str, int] | None = None,
    vpp_min: float = 0.5,
) -> list[dict]:
    """Slice an EEG segment into sub-windows, filter artifacts, and extract PSD.

    Note: this is the raw-acquisition-time quality gate used during
    preprocessing (`run_preprocessing.py`), separate from the two-tier
    channel-drop pipeline used by the feature extractors above. `vpp_min`
    is set and passed explicitly here (default 0.5) so this stage never
    silently inherits `validate_eeg_windows`'s default -- if that shared
    default changes again for the feature-extraction quality gate, this
    raw-preprocessing stage is unaffected.
    """

    samples_per_sub = int(sub_window_sec * fs)
    n_sub = eeg_segment.shape[1] // samples_per_sub
    records = []
    subwindows = np.stack([
        eeg_segment[:, s * samples_per_sub:(s + 1) * samples_per_sub]
        for s in range(n_sub)
    ]) if n_sub else np.empty((0, eeg_segment.shape[0], samples_per_sub))
    # 
    valid_mask = validate_eeg_windows(
        subwindows,
        vpp_min=vpp_min,
        vpp_max=vpp_thresh,
        std_min=std_range[0],
        std_max=std_range[1],
    )
    if quality_stats is not None:
        quality_stats["total"] += n_sub
        quality_stats["rejected"] += int(np.count_nonzero(valid_mask == 0))
        quality_stats["accepted"] += int(np.count_nonzero(valid_mask == 1))

    for s in range(n_sub):
        if valid_mask[s] == 0:
            continue

        sub_win = subwindows[s]
        vpp = np.ptp(sub_win, axis=-1)
        std_dev = np.std(sub_win, axis=-1)

        freqs, psd = periodogram(sub_win, fs=fs, axis=-1)
        records.append({
            "time_data": sub_win,
            "vpp": vpp,
            "std": std_dev,
            "psd": psd,
            "freqs": freqs,
        })
    return records


def process_condition_windows_with_baseline(
    eeg_data: np.ndarray,
    mark_channel: np.ndarray,
    fs: float,
    target_conditions: tuple[int, ...] = (101, 102, 103, 104, 105),
    cue_condition: int = 202,
    cross_condition: int = 201,
    sub_window_sec: float = 1.0,
    vpp_thresh: float = 200.0,
    std_range: tuple[float, float] = (0.5, 60.0),
    stim_duration_sec: float = 5.0,
    cue_lookback_sec: float = 1.0,
    quality_stats: dict[str, int] | None = None,
) -> dict[int, list[dict]]:
    """Extracts condition windows ensuring exact subwindow counts and 202 lookback alignment."""
    condition_records: dict[int, list[dict]] = {
        cond: [] for cond in list(target_conditions) + [cue_condition, cross_condition]
    }

    diff_marks = np.diff(mark_channel, prepend=0)

    # 1. Process 202: 1.0 s window looking backwards from the cue trigger
    cue_indices = np.where((mark_channel == cue_condition) & (diff_marks != 0))[0]
    cue_samples = int(cue_lookback_sec * fs)

    for idx in cue_indices:
        start = max(0, idx - cue_samples)
        if (idx - start) == cue_samples:
            segment = eeg_data[:, start:idx]
            records = extract_clean_subwindows_psd(
                segment, fs, sub_window_sec, vpp_thresh, std_range, quality_stats
            )
            condition_records[cue_condition].extend(records)

    # 2. Process Stimulus Target Conditions (101, 102, 103, 104, 105)
    # Extracts all 5 sub-windows per trial (5.0s total)
    stim_samples = int(stim_duration_sec * fs)

    for cond in target_conditions:
        event_indices = np.where((mark_channel == cond) & (diff_marks != 0))[0]
        if len(event_indices) == 0:
            event_indices = np.where(mark_channel == cond)[0]
            if len(event_indices) == 0:
                continue
            splits = np.where(np.diff(event_indices) > 1)[0] + 1
            event_indices = [b[0] for b in np.split(event_indices, splits)]

        for event_idx in event_indices:
            end_idx = event_idx + stim_samples
            if end_idx <= eeg_data.shape[1]:
                stim_segment = eeg_data[:, event_idx:end_idx]
                records = extract_clean_subwindows_psd(
                    stim_segment, fs, sub_window_sec, vpp_thresh, std_range, quality_stats
                )
                condition_records[cond].extend(records)

    # 3. Process 201 (Fixation Cross)
    cross_indices = np.where((mark_channel == cross_condition) & (diff_marks != 0))[0]
    if len(cross_indices) > 0:
        for c_idx in cross_indices:
            end_c = min(eeg_data.shape[1], c_idx + int(2.0 * fs))
            cross_segment = eeg_data[:, c_idx:end_c]
            records = extract_clean_subwindows_psd(
                cross_segment, fs, sub_window_sec, vpp_thresh, std_range, quality_stats
            )
            condition_records[cross_condition].extend(records)

    return condition_records


def prepare_feature_matrix(
    condition_records: dict,
    target_conditions: tuple[int, ...] = (201, 202, 101, 102, 103, 104, 105),
    freq_range: tuple[float, float] = (5.0, 35.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten selected PSD features into X_psd, export X_time, and vector y."""
    X_psd = []
    X_time = []
    y = []

    for cond in target_conditions:
        if cond not in condition_records:
            continue
        records = condition_records[cond]
        for item in records:
            psd = item["psd"]
            freqs = item["freqs"]

            freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
            psd_selected = psd[:, freq_mask]

            X_psd.append(psd_selected.flatten())
            X_time.append(item.get("time_data", np.zeros((7, 250))))
            y.append(cond)

    return np.array(X_psd), np.array(X_time), np.array(y)

# Backwards compatibility alias
process_condition_windows = process_condition_windows_with_baseline