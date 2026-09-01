"""Run EEG preprocessing and condition-based PSD feature extraction."""
import sys
from pathlib import Path
    
import numpy as np
import yaml

from src.features.psd_extraction import prepare_feature_matrix, process_condition_windows
from src.io.ebr_parser import load_ebr_file
from src.preprocessing.filters import apply_iir_bandpass
from src.visualization.signal_plots import plot_average_psd_by_condition


def main():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "pipeline_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    base_dir = Path(__file__).resolve().parents[1]
    file_path = base_dir / config["paths"]["data_raw_dir"] / "S01" / "OO.ebr"

    recording = load_ebr_file(file_path)
    fs = recording["sampling_rate"]
    raw_matrix = recording["data"][0, :, 0, :]

    scalp_channels = config["channels"]["scalp"]
    eeg_indices = [i for i, name in enumerate(recording["channels"]) if name in scalp_channels]
    mark_idx = recording["channels"].index(config["channels"]["mark_channel"]) if config["channels"]["mark_channel"] in recording["channels"] else -1
    mark_signal = raw_matrix[mark_idx, :] if mark_idx != -1 else np.zeros(raw_matrix.shape[1])

    filtered_eeg = apply_iir_bandpass(
        raw_matrix[eeg_indices, :],
        fs,
        lowcut=config["preprocessing"]["lowcut_hz"],
        highcut=config["preprocessing"]["highcut_hz"],
        order=config["preprocessing"]["filter_order"],
    )

    condition_records = process_condition_windows(
        filtered_eeg,
        mark_signal,
        fs,
        target_conditions=config["project"]["target_conditions"],
        sub_window_sec=config["preprocessing"]["sub_window_sec"],
        vpp_thresh=config["preprocessing"]["vpp_thresh_uv"],
        std_range=tuple(config["preprocessing"]["std_range"]),
    )

    feature_matrix_x, target_matrix_y = prepare_feature_matrix(
        condition_records,
        target_conditions=config["project"]["target_conditions"],
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


if __name__ == "__main__":
    main()
