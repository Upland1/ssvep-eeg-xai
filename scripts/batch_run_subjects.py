"""Automated batch preprocessing and evaluation across subject cohorts (S01 to S27)."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scripts.train_models import (
    FBCCA_TARGET_FREQS,
    extract_all_fbcca_features,
    extract_harmonic_features,
    run_evaluation,
    run_temporal_integration_evaluation,
)
from src.features.psd_extraction import (
    prepare_feature_matrix,
    process_condition_windows_with_baseline,
)
from src.io.ebr_parser import load_ebr_file
from src.models.spatial_spectral_cnn import (
    evaluate_temporal_smoothing,
    prepare_spatial_spectral_tensors,
    train_and_eval_fusion_cnn_cv,
)
from src.preprocessing.filters import apply_iir_bandpass


def preprocess_subject_if_needed(
    base_dir: Path, config: dict, subject_id: str, paradigm: str = "OO.ebr"
) -> Path:
    """Check if preprocessed data exists; otherwise parse raw .ebr and extract features."""
    subject_processed_dir = base_dir / config["paths"]["data_processed_dir"] / subject_id
    subject_processed_dir.mkdir(parents=True, exist_ok=True)

    psd_file = subject_processed_dir / "X_psd_features.npy"
    time_file = subject_processed_dir / "X_time_windows.npy"
    label_file = subject_processed_dir / "y_labels.npy"

    if psd_file.exists() and time_file.exists() and label_file.exists():
        return subject_processed_dir

    raw_file = base_dir / config["paths"]["data_raw_dir"] / subject_id / paradigm
    if not raw_file.exists():
        raise FileNotFoundError(f"Raw recording not found: {raw_file}")

    print(f"[{subject_id}] Preprocessing raw recording: {raw_file.name}...")
    recording = load_ebr_file(raw_file)
    fs = float(recording["sampling_rate"])
    raw_matrix = recording["data"][0, :, 0, :]

    scalp_channels = config["channels"]["scalp"]
    eeg_indices = [
        i for i, name in enumerate(recording["channels"]) if name in scalp_channels
    ]

    mark_name = config["channels"]["mark_channel"]
    mark_idx = (
        recording["channels"].index(mark_name)
        if mark_name in recording["channels"]
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

    stimulus_targets = [
        condition
        for condition in config["project"]["target_conditions"]
        if condition not in (201, 202)
    ]
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
    )

    X_psd, X_time, y = prepare_feature_matrix(
        condition_records,
        target_conditions=tuple(config["project"]["target_conditions"]),
        freq_range=tuple(config["features"]["psd_range_hz"]),
    )

    np.save(psd_file, X_psd)
    np.save(time_file, X_time)
    np.save(label_file, y)
    return subject_processed_dir


def evaluate_subject(subject_dir: Path, subject_id: str) -> dict:
    """Evaluate classical Shrinkage-LDA and dual-branch fusion CNN."""
    X_psd = np.load(subject_dir / "X_psd_features.npy")
    X_time = np.load(subject_dir / "X_time_windows.npy")
    y = np.load(subject_dir / "y_labels.npy")

    target_classes = (101, 102, 103, 104, 105)
    mask = np.isin(y, target_classes)
    X_psd_filt = X_psd[mask]
    X_time_filt = X_time[mask]
    y_filt = y[mask]

    X_harm = extract_harmonic_features(X_psd_filt)
    cache_fbcca = subject_dir / "X_fbcca_features.npy"
    if cache_fbcca.exists():
        X_fbcca = np.load(cache_fbcca)
    else:
        print(f"[{subject_id}] Extracting FBCCA correlation features...")
        X_fbcca = extract_all_fbcca_features(
            X_time_filt, fs=250.0, target_freqs=FBCCA_TARGET_FREQS
        )
        np.save(cache_fbcca, X_fbcca)

    X_hybrid = np.hstack([X_harm, X_fbcca])

    eval_results = run_evaluation(X_hybrid, y_filt, subject_id=subject_id)
    lda_entry = next(
        result for result in eval_results if result["Classifier"] == "Shrinkage-LDA"
    )
    lda_smoothed = run_temporal_integration_evaluation(
        X_hybrid, y_filt, window_size=2
    )

    X_psd_tensors = prepare_spatial_spectral_tensors(X_psd_filt)
    cnn_metrics = train_and_eval_fusion_cnn_cv(X_psd_tensors, X_fbcca, y_filt)
    cnn_smoothed = evaluate_temporal_smoothing(
        cnn_metrics["y_mapped"], cnn_metrics["oof_probabilities"], window_size=2
    )

    return {
        "Subject": subject_id,
        "LDA (1.0s)": lda_entry["Accuracy"],
        "LDA (2.0s Smoothed)": f"{lda_smoothed * 100:.2f}%",
        "Fusion CNN (1.0s)": f"{cnn_metrics['mean_accuracy'] * 100:.2f}%",
        "Fusion CNN (2.0s Smoothed)": f"{cnn_smoothed * 100:.2f}%",
        "LDA Recall Breakdown": lda_entry["Recall per Class"],
    }


def main():
    parser = argparse.ArgumentParser(description="Multi-Subject SSVEP Pipeline Benchmark")
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=["S01", "S02", "S03", "S04", "S05"],
        help="Subject IDs to evaluate (e.g., S01 S02)",
    )
    parser.add_argument(
        "--paradigm",
        type=str,
        default="OO.ebr",
        help="Target paradigm EBR file (OO.ebr, OOS.ebr, CBR.ebr, CBS.ebr)",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    config_path = base_dir / "configs" / "pipeline_config.yaml"
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    summary_records = []
    print("\n=================================================================")
    print(f"   STARTING BATCH EVALUATION FOR SUBJECTS: {', '.join(args.subjects)}")
    print(f"   PARADIGM: {args.paradigm}")
    print("=================================================================\n")

    for subject_id in args.subjects:
        try:
            subject_dir = preprocess_subject_if_needed(
                base_dir, config, subject_id, args.paradigm
            )
            metrics = evaluate_subject(subject_dir, subject_id)
            summary_records.append(metrics)
            print(
                f"--> [{subject_id}] Completed | LDA 1.0s: {metrics['LDA (1.0s)']} | "
                f"Fusion CNN 1.0s: {metrics['Fusion CNN (1.0s)']} | "
                f"Fusion CNN 2.0s: {metrics['Fusion CNN (2.0s Smoothed)']}"
            )
        except Exception as err:
            print(f"[!] Error processing {subject_id}: {err}")

    if summary_records:
        df_summary = pd.DataFrame(summary_records)
        print("\n" + "=" * 105)
        print("                        COHORT MULTI-SUBJECT BENCHMARK SUMMARY")
        print("=" * 105)
        print(df_summary.drop(columns=["LDA Recall Breakdown"]).to_string(index=False))
        print("=" * 105)

        print("\n--- Per-Class Recall Breakdown (Shrinkage-LDA) ---")
        for row in summary_records:
            print(f"{row['Subject']}: {row['LDA Recall Breakdown']}")


if __name__ == "__main__":
    main()
