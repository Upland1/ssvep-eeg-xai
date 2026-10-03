"""Compute and visualize model explainability (feature importance)."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler


def main():
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "data" / "processed"
    fig_dir = base_dir / "outputs" / "figures" / "xai_heatmaps"
    fig_dir.mkdir(parents=True, exist_ok=True)

    from scripts.train_models import (
        TARGET_FREQS,
        extract_all_fbcca_features,
        extract_harmonic_features,
        load_dataset,
    )

    X_psd, X_time, y = load_dataset(data_dir)
    if X_time is None:
        raise FileNotFoundError("X_time_windows.npy is required to compute FBCCA features.")

    target_classes = list(TARGET_FREQS)
    freq_labels = [f"{TARGET_FREQS[class_id]:g}Hz" for class_id in target_classes]

    X_harm = extract_harmonic_features(X_psd)
    X_fbcca = extract_all_fbcca_features(X_time, fs=250.0)
    X_hybrid = np.hstack([X_harm, X_fbcca])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_hybrid)

    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    clf.fit(X_scaled, y)

    weights = clf.coef_
    psd_weights = weights[:, :32]
    psd_weights_reshaped = psd_weights.reshape(5, 4, 8)

    channels = ["POz", "O1", "Oz", "O2"]
    harmonic_freqs = ["8.57", "10.91", "15", "20", "24", "17.14", "21.82", "30"]
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
    for class_idx, class_code in enumerate(target_classes):
        sns.heatmap(
            psd_weights_reshaped[class_idx],
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            xticklabels=harmonic_freqs,
            yticklabels=channels if class_idx == 0 else False,
            ax=axes[class_idx],
            cbar=class_idx == 4,
        )
        axes[class_idx].set_title(f"Class {class_code}")
        axes[class_idx].set_xlabel("Frequency (Hz)")

    axes[0].set_ylabel("Electrode Channel")
    fig.suptitle("Spatial-Spectral Feature Weights Across Occipital Channels")
    fig.tight_layout()
    spatial_output_path = fig_dir / "xai_spatial_spectral_weights.png"
    fig.savefig(spatial_output_path, dpi=150)
    plt.close(fig)
    print(f"Spatial-spectral XAI heatmap saved to: {spatial_output_path}")

    fbcca_weights = weights[:, -5:]
    plt.figure(figsize=(8, 4))
    sns.heatmap(
        fbcca_weights,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        xticklabels=[f"Corr({frequency})" for frequency in freq_labels],
        yticklabels=[
            f"Class {class_id} ({frequency})"
            for class_id, frequency in zip(target_classes, freq_labels)
        ],
    )
    plt.title("XAI: Model Sensitivity to FBCCA Reference Frequencies")
    plt.xlabel("FBCCA Correlation Features")
    plt.ylabel("Target Class")
    plt.tight_layout()

    output_path = fig_dir / "xai_fbcca_weights.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"XAI attribution heatmap saved to: {output_path}")


if __name__ == "__main__":
    main()