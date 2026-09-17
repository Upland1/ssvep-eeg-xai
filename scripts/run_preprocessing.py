"""Run EEG preprocessing and condition-based PSD feature extraction."""
from pathlib import Path
import numpy as np
import yaml

from src.features.psd_extraction import (
    prepare_feature_matrix,
    process_condition_windows_with_baseline,
)
from src.io.ebr_parser import load_ebr_file
from src.preprocessing.filters import apply_iir_bandpass
from src.visualization.signal_plots import plot_average_psd_by_condition
from src.preprocessing.quality_check import identify_noisy_channels, summarize_window_quality

def main():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "pipeline_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    base_dir = Path(__file__).resolve().parents[1]
    file_path = base_dir / config["paths"]["data_raw_dir"] / "S03" / "OO.ebr"

    recording = load_ebr_file(file_path)
    fs = recording["sampling_rate"]
    raw_matrix = recording["data"][0, :, 0, :]

    scalp_channels = config["channels"]["scalp"]
    eeg_indices = [i for i, name in enumerate(recording["channels"]) if name in scalp_channels]

    mark_idx = (
        recording["channels"].index(config["channels"]["mark_channel"])
        if config["channels"]["mark_channel"] in recording["channels"]
        else -1
    )
    mark_signal = raw_matrix[mark_idx, :] if mark_idx != -1 else np.zeros(raw_matrix.shape[1])

    filtered_eeg = apply_iir_bandpass(
        raw_matrix[eeg_indices, :],
        fs,
        lowcut=config["preprocessing"]["lowcut_hz"],
        highcut=config["preprocessing"]["highcut_hz"],
        order=config["preprocessing"]["filter_order"],
    )

    # Remove chronically bad channels before the per-window quality gate.
    samples_per_sub = int(config["preprocessing"]["sub_window_sec"] * fs)
    n_quality_windows = filtered_eeg.shape[1] // samples_per_sub
    quality_windows = np.stack([
        filtered_eeg[:, i * samples_per_sub:(i + 1) * samples_per_sub]
        for i in range(n_quality_windows)
    ])
    noisy_indices, _ = identify_noisy_channels(
        quality_windows,
        channel_names=scalp_channels,
        vpp_min=5,
        vpp_max=config["preprocessing"]["vpp_thresh_uv"],
        std_min=config["preprocessing"]["std_range"][0],
        std_max=config["preprocessing"]["std_range"][1],
    )
    if noisy_indices:
        noisy_names = [scalp_channels[i] for i in noisy_indices]
        print(f"Dropping chronically noisy channels: {noisy_names}")
        keep_indices = [i for i in range(len(scalp_channels)) if i not in noisy_indices]
        filtered_eeg = filtered_eeg[keep_indices, :]
        scalp_channels = [scalp_channels[i] for i in keep_indices]
        quality_windows = quality_windows[:, keep_indices, :]

    stimulus_targets = [c for c in config["project"]["target_conditions"] if c not in (201, 202)]
    quality_stats = {"total": 0, "accepted": 0, "rejected": 0}

    condition_records = process_condition_windows_with_baseline(
        filtered_eeg,
        mark_signal,
        fs,
        target_conditions=tuple(stimulus_targets),
        cue_condition=config["preprocessing"]["baseline"]["cue_condition"],
        cross_condition=config["preprocessing"]["baseline"]["cross_condition"],
        sub_window_sec=config["preprocessing"]["sub_window_sec"],
        vpp_thresh=config["preprocessing"]["vpp_thresh_uv"],
        std_range=tuple(config["preprocessing"]["std_range"]),
        stim_duration_sec=config["preprocessing"]["baseline"]["stimulus_duration_sec"],
        cue_lookback_sec=config["preprocessing"]["baseline"]["cue_lookback_sec"],
        quality_stats=quality_stats,
    )

    print(
        "Window quality validation: "
        f"{quality_stats['accepted']} accepted, "
        f"{quality_stats['rejected']} rejected, "
        f"{quality_stats['total']} total"
    )

    # Print validation of window balance
    print("\n--- Window Extraction Counts ---")
    total_stim = 0
    for cond in config["project"]["target_conditions"]:
        count = len(condition_records.get(cond, []))
        print(f"Condition {cond}: {count} windows")
        if cond in stimulus_targets:
            total_stim += count
    cue_count = len(condition_records.get(202, []))
    print(f"Total Stimulus Windows (101-105): {total_stim} | Cue (202) Windows: {cue_count}\n")

    feature_matrix_x, target_matrix_y = prepare_feature_matrix(
        condition_records,
        target_conditions=tuple(config["project"]["target_conditions"]),
        freq_range=tuple(config["features"]["psd_range_hz"]),
    )

    print(f"Feature matrix X shape: {feature_matrix_x.shape}")
    print(f"Target vector y shape: {target_matrix_y.shape}")

    output_dir = base_dir / config["paths"]["figures_dir"] / "spectra"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_average_psd_by_condition(
        condition_records,
        scalp_channels,
        target_freq_range=tuple(config["features"]["plotting_range_hz"]),
        output_path=str(output_dir / "average_psd_by_condition.png"),
    )

    processed_dir = base_dir / config["paths"]["data_processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    np.save(processed_dir / "X_psd_features.npy", feature_matrix_x)
    np.save(processed_dir / "y_labels.npy", target_matrix_y)

    print("Preprocessing completed successfully.")
    summarize_window_quality(
        quality_windows,
        vpp_min=0.5,
        vpp_max=config["preprocessing"]["vpp_thresh_uv"],
        std_min=config["preprocessing"]["std_range"][0],
        std_max=config["preprocessing"]["std_range"][1],
    )

if __name__ == "__main__":
    main()