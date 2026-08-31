"""Explainability plotting utilities for EEG saliency and SHAP heatmaps."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np


def plot_saliency_heatmap(scores: np.ndarray, channel_names: list[str], freq_values: np.ndarray, output_path: str = "outputs/figures/xai_heatmaps/saliency.png"):
    """Plot a spatial-spectral attribution map for a single example."""
    fig, ax = plt.subplots(figsize=(10, 5))
    img = ax.imshow(scores, aspect="auto", cmap="viridis", origin="lower")
    ax.set_yticks(np.arange(len(channel_names)))
    ax.set_yticklabels(channel_names)
    ax.set_xticks(np.linspace(0, scores.shape[1] - 1, min(10, scores.shape[1])).astype(int))
    ax.set_xticklabels(np.round(freq_values[np.linspace(0, len(freq_values) - 1, min(10, len(freq_values))).astype(int)], 1))
    ax.set_title("Input-gradient saliency")
    fig.colorbar(img, ax=ax, label="Attribution magnitude")
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
