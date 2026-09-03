"""Power spectral density extraction with relative stimulus-baseline condition support."""
import numpy as np
from scipy.signal import periodogram


def extract_clean_subwindows_psd(
    eeg_segment: np.ndarray,
    fs: float,
    sub_window_sec: float = 1.0,
    vpp_thresh: float = 200.0,
    std_range: tuple[float, float] = (0.5, 60.0),
) -> list[dict]:
    """Slice an EEG segment into sub-windows, filter artifacts, and extract PSD."""
    samples_per_sub = int(sub_window_sec * fs)
    n_sub = eeg_segment.shape[1] // samples_per_sub
    records = []

    for s in range(n_sub):
        start = s * samples_per_sub
        end = start + samples_per_sub
        sub_win = eeg_segment[:, start:end]

        vpp = np.ptp(sub_win, axis=-1)
        std_dev = np.std(sub_win, axis=-1)

        # Artifact validation
        if np.any(vpp > vpp_thresh) or np.any(std_dev < std_range[0]) or np.any(std_dev > std_range[1]):
            continue

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
                segment, fs, sub_window_sec, vpp_thresh, std_range
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
                    stim_segment, fs, sub_window_sec, vpp_thresh, std_range
                )
                condition_records[cond].extend(records)

    # 3. Process 201 (Fixation Cross)
    cross_indices = np.where((mark_channel == cross_condition) & (diff_marks != 0))[0]
    if len(cross_indices) > 0:
        for c_idx in cross_indices:
            end_c = min(eeg_data.shape[1], c_idx + int(2.0 * fs))
            cross_segment = eeg_data[:, c_idx:end_c]
            records = extract_clean_subwindows_psd(
                cross_segment, fs, sub_window_sec, vpp_thresh, std_range
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