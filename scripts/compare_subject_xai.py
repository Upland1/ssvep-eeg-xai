"""Generate side-by-side Spatial-Spectral XAI heatmaps for S01 vs S04."""

from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_comparative_xai(model_dir: Path, output_fig: Path):
    s01_data = joblib.load(model_dir / "S01_shrinkage_lda.joblib")
    s04_data = joblib.load(model_dir / "S04_shrinkage_lda.joblib")

    classes = s01_data["classes"]
    channels = ["POz", "O1", "Oz", "O2"]
    harmonic_freqs = ["8.57", "10.91", "15", "20", "24", "17.14", "21.82", "30"]

    # Extract 32 spatial-spectral PSD weights per class: shape (5, 4, 8)
    w_s01 = s01_data["model"].coef_[:, :32].reshape(5, 4, 8)
    w_s04 = s04_data["model"].coef_[:, :32].reshape(5, 4, 8)

    fig, axes = plt.subplots(2, 5, figsize=(22, 7), sharex=True, sharey=True)

    for c_idx, class_id in enumerate(classes):
        # Row 0: S01 (Low SNR)
        sns.heatmap(
            w_s01[c_idx],
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0.0,
            xticklabels=harmonic_freqs if c_idx == 0 else False,
            yticklabels=channels if c_idx == 0 else False,
            ax=axes[0, c_idx],
            cbar=False,
        )
        axes[0, c_idx].set_title(f"S01 (Low-SNR) | Class {class_id}")

        # Row 1: S04 (High SNR)
        sns.heatmap(
            w_s04[c_idx],
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0.0,
            xticklabels=harmonic_freqs,
            yticklabels=channels if c_idx == 0 else False,
            ax=axes[1, c_idx],
            cbar=(c_idx == 4),
        )
        axes[1, c_idx].set_title(f"S04 (High-SNR) | Class {class_id}")
        axes[1, c_idx].set_xlabel("Harmonic Frequency (Hz)")

    axes[0, 0].set_ylabel("Occipital Electrode")
    axes[1, 0].set_ylabel("Occipital Electrode")
    fig.suptitle("Spatial-Spectral Attribution Weights: Low-SNR (S01) vs. High-SNR (S04)", fontsize=16)
    fig.tight_layout()

    output_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_fig, dpi=200)
    plt.close()
    print(f"Comparative XAI heatmap saved to: {output_fig}")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[1]
    plot_comparative_xai(
        model_dir=base_dir / "outputs" / "models",
        output_fig=base_dir / "outputs" / "figures" / "xai_heatmaps" / "xai_s01_vs_s04_comparison.png",
    )