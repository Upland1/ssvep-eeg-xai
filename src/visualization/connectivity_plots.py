"""
MNE-style connectivity diagrams for the condition-averaged matrices
produced by connectivity_extraction.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

try:
  from mne_connectivity.viz import plot_connectivity_circle
  _HAS_MNE_CONNECTIVITY = True
except ImportError:
  _HAS_MNE_CONNECTIVITY = False

# Label used for the no-stimulus baseline in figure titles/filenames.
CONDITION_LABELS = {
    201: "No stimulus (201)",
    101: "24 Hz",
    102: "20 Hz",
    103: "15 Hz",
    104: "10.91 Hz",
    105: "8.57 Hz",
}


def _require_mne_connectivity():
  if not _HAS_MNE_CONNECTIVITY:
    raise ImportError(
        "mne_connectivity is required for connectivity plotting. "
        "Install it with: pip install mne mne-connectivity"
    )


def plot_condition_connectivity(
    matrix: np.ndarray,
    channel_names: list[str],
    title: str,
    vmin: float | None = None,
    vmax: float | None = None,
    n_lines: int | None = None,
    colormap: str = "YlOrRd",
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
  """
  Plot a single condition's averaged connectivity matrix as an
  MNE-style circular diagram.
  """
  _require_mne_connectivity()

  n_channels = matrix.shape[0]
  if n_lines is None:
    n_lines = n_channels * (n_channels - 1) // 2

  fig, ax = plot_connectivity_circle(
      matrix,
      channel_names,
      n_lines=n_lines,
      title=title,
      vmin=vmin,
      vmax=vmax,
      colormap=colormap,
      ax=ax,
      show=False,
  )
  return fig, ax


def plot_all_conditions_panel(
    by_condition: dict,
    channel_names: list[str],
    condition_labels: dict[int, str] = CONDITION_LABELS,
    conditions: list[int] | None = None,
    colormap: str = "YlOrRd",
    shared_scale: bool = True,
    figsize_per_panel: float = 4.0,
) -> plt.Figure:
  """
  Side-by-side MNE-style circles for every condition, sharing one
  color scale by default so connection strength is visually comparable
  across the no-stimulus baseline and each SSVEP frequency.
  """
  _require_mne_connectivity()

  if conditions is None:
    conditions = sorted(by_condition, key=lambda c: (c != 201, c))

  valid = [c for c in conditions if by_condition.get(c, {}).get("matrix") is not None]
  if not valid:
    raise ValueError("No condition has a computed connectivity matrix to plot.")

  vmin = vmax = None
  if shared_scale:
    all_vals = []
    for c in valid:
      m = by_condition[c]["matrix"]
      n = m.shape[0]
      all_vals.append(m[np.triu_indices(n, k=1)])
    all_vals = np.concatenate(all_vals)
    vmin, vmax = float(all_vals.min()), float(all_vals.max())

  fig = plt.figure(figsize=(figsize_per_panel * len(valid), figsize_per_panel + 0.8))
  for idx, cond in enumerate(valid):
    ax = fig.add_subplot(1, len(valid), idx + 1, projection="polar")
    entry = by_condition[cond]
    label = condition_labels.get(cond, str(cond))
    n_win = entry["n_windows"]
    plot_connectivity_circle(
        entry["matrix"],
        channel_names,
        n_lines=len(channel_names) * (len(channel_names) - 1) // 2,
        title=f"{label}\n(n={n_win} windows)",
        vmin=vmin,
        vmax=vmax,
        colormap=colormap,
        padding=8.0,
        fontsize_title=11,
        ax=ax,
        show=False,
    )

  fig.subplots_adjust(top=0.88, bottom=0.08)
  return fig


def plot_difference_panel(
    diffs: dict[int, np.ndarray],
    channel_names: list[str],
    condition_labels: dict[int, str] = CONDITION_LABELS,
    conditions: list[int] | None = None,
    colormap: str = "RdBu_r",
    figsize_per_panel: float = 4.0,
    permutation_tests: dict[int, dict] | None = None,
    mask_nonsignificant: bool = False,
    use_fdr: bool = False,
) -> plt.Figure:
  """Side-by-side diagrams of stimulus-minus-matched-baseline connectivity,
  one per SSVEP frequency.

  Optionally annotates significance from permutation tests or masks non-significant edges.
  """
  _require_mne_connectivity()

  if conditions is None:
    conditions = sorted(diffs)
  valid = [c for c in conditions if diffs.get(c) is not None]
  if not valid:
    raise ValueError("No condition has a computed difference matrix to plot.")

  max_abs = max(float(np.abs(diffs[c]).max()) for c in valid)
  # Prevent degenerate color scale if max_abs is 0
  max_abs = max(max_abs, 1e-3)
  vmin, vmax = -max_abs, max_abs

  fig = plt.figure(figsize=(figsize_per_panel * len(valid), figsize_per_panel + 1.2))
  for idx, cond in enumerate(valid):
    ax = fig.add_subplot(1, len(valid), idx + 1, projection="polar")
    label = condition_labels.get(cond, str(cond))

    matrix_to_draw = diffs[cond].copy()
    title = f"{label}\n(stimulus - baseline)"

    if permutation_tests is not None and cond in permutation_tests and permutation_tests[cond] is not None:
      perm = permutation_tests[cond]
      p_val = perm["p_val_global_one_tailed"]
      stars = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
      n_sig = perm["n_significant_edges_fdr"] if use_fdr else perm["n_significant_edges_uncorrected"]
      total_e = perm["total_edges"]
      corr_lbl = "FDR" if use_fdr else "p<0.05"
      mean_d = perm["obs_mean_diff"]
      title = f"{label}\nΔ={mean_d:+.3f} (p={p_val:.3f}, {stars})\n{n_sig}/{total_e} edges ({corr_lbl})"

      if mask_nonsignificant:
        mask = perm["edge_fdr_mask"] if use_fdr else perm["edge_p_mask"]
        matrix_to_draw[~mask] = 0.0

    plot_connectivity_circle(
        matrix_to_draw,
        channel_names,
        n_lines=len(channel_names) * (len(channel_names) - 1) // 2,
        title=title,
        vmin=vmin,
        vmax=vmax,
        colormap=colormap,
        padding=8.0,
        fontsize_title=10,
        ax=ax,
        show=False,
    )

  fig.subplots_adjust(top=0.86, bottom=0.08)
  return fig


def plot_permutation_null_distributions(
    permutation_tests: dict[int, dict],
    condition_labels: dict[int, str] = CONDITION_LABELS,
    conditions: list[int] | None = None,
    figsize_per_panel: float = 3.4,
) -> plt.Figure:
  """Plot null distribution histograms from the permutation test against observed differences.

  Visualizes H0 (interchangeable stimulus/baseline windows) vs observed mean difference,
  clearly demonstrating whether each condition falls within null noise or outside it.
  """
  if conditions is None:
    conditions = sorted(permutation_tests)
  valid = [c for c in conditions if permutation_tests.get(c) is not None]
  if not valid:
    raise ValueError("No permutation tests available to plot.")

  fig, axes = plt.subplots(
      1, len(valid), figsize=(figsize_per_panel * len(valid), figsize_per_panel + 0.6), sharey=False
  )
  if len(valid) == 1:
    axes = [axes]

  for ax, cond in zip(axes, valid):
    perm = permutation_tests[cond]
    label = condition_labels.get(cond, str(cond))
    null_vals = perm["null_mean_diffs"]
    obs_val = perm["obs_mean_diff"]
    p_1t = perm["p_val_global_one_tailed"]
    ci_low, ci_high = perm["null_ci_95"]
    is_sig = p_1t < 0.05

    # Histogram of null distribution
    ax.hist(null_vals, bins=25, color="#5dade2", edgecolor="#2874a6", alpha=0.75, density=True)
    # Shaded 95% null confidence region
    ax.axvspan(ci_low, ci_high, color="#aed6f1", alpha=0.35, label="95% Null CI")
    # Vertical line for observed difference
    obs_color = "#e74c3c" if is_sig else "#7f8c8d"
    ax.axvline(obs_val, color=obs_color, linestyle="--", linewidth=2.5, label=f"Obs: {obs_val:+.3f}")
    ax.axvline(0.0, color="#2c3e50", linestyle=":", linewidth=1.0)

    stars = "***" if p_1t < 0.001 else "**" if p_1t < 0.01 else "*" if p_1t < 0.05 else "ns"
    ax.set_title(f"{label}\np = {p_1t:.4f} ({stars})\nObs Δ: {obs_val:+.4f}", fontsize=10, fontweight="bold")
    ax.set_xlabel("Coherence Difference (Δ)", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=7.5, loc="upper right")

  axes[0].set_ylabel("Null Density", fontsize=10)
  fig.suptitle("Permutation Test: Observed Difference vs Null Distribution (H₀: Stimulus = Rest)", fontsize=12, y=1.03)
  fig.tight_layout()
  return fig