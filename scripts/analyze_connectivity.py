"""
Functional connectivity analysis: no-stimulus baseline vs each SSVEP
frequency, per subject.
"""

import argparse
from pathlib import Path

import numpy as np

from src.features.connectivity_extraction import (
    BASELINE_CONDITION,
    CONDITION_FREQS,
    compute_condition_coherence,
    compute_condition_pdc,
    compute_condition_plv,
    compute_condition_wpli,
    extract_connectivity_by_condition,
)
from src.visualization.connectivity_plots import (
    CONDITION_LABELS,
    plot_all_conditions_panel,
    plot_difference_panel,
    plot_permutation_null_distributions,
)

FULL_MONTAGE = ["PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"]

METHODS = {
    "coherence": compute_condition_coherence,
  "plv": compute_condition_plv,
  "wpli": compute_condition_wpli,
  "pdc": compute_condition_pdc,
    # "granger": compute_granger_condition_matrix,  # queued
    # "mutual_info": compute_mi_condition_matrix,   # queued, not recommended as primary result
}


def resolve_channel_names(n_channels: int, full_montage: list[str] = FULL_MONTAGE) -> list[str] | None:
  """Match the actual channel count to known montage layouts."""
  if n_channels == len(full_montage):
    return list(full_montage)
  if n_channels == len(full_montage) - 1:
    print(
        f"[!] {n_channels} channels found (expected {len(full_montage)}); "
        "assuming PO7 is still missing from this data -- rerun "
        "preprocessing to include it."
    )
    return full_montage[1:]
  print(f"[!] Unexpected channel count ({n_channels}); channel names unavailable.")
  return None


def main():
  parser = argparse.ArgumentParser(
      description="Per-subject functional connectivity analysis: baseline vs each SSVEP frequency."
  )
  parser.add_argument("--subject", default="S03", help="Subject subdirectory (default: S03).")
  parser.add_argument(
      "--method", choices=list(METHODS.keys()), default="coherence",
      help="Connectivity method to use (default: coherence).",
  )
  parser.add_argument(
      "--permutations", type=int, default=200,
      help="Number of permutations for significance testing (default: 200, 0 to disable).",
  )
  parser.add_argument(
      "--alpha", type=float, default=0.05,
      help="Significance threshold alpha (default: 0.05).",
  )
  parser.add_argument(
      "--n-jobs", type=int, default=-1,
      help="Number of parallel jobs for permutation test (default: -1, all CPUs).",
  )
  parser.add_argument(
      "--seed", type=int, default=42,
      help="Random seed for permutations (default: 42).",
  )
  args = parser.parse_args()

  project_root = Path(__file__).resolve().parent.parent
  data_dir = project_root / "data" / "processed" / args.subject

  # 1. Load data -- keep baseline (201) alongside the 5 stimulus conditions,
  #    unlike every other benchmark script which drops 201/202 up front.
  windows_raw = np.load(data_dir / "X_time_windows.npy")
  y_raw = np.load(data_dir / "y_labels.npy")

  conditions_of_interest = [BASELINE_CONDITION] + sorted(CONDITION_FREQS)
  mask = np.isin(y_raw, conditions_of_interest)
  y_sel = y_raw[mask]
  windows = windows_raw[mask] if windows_raw.shape[0] == len(y_raw) else windows_raw[: len(y_sel)]

  print(f"Subject: {args.subject} | Method: {args.method} | Permutations: {args.permutations}")
  print(f"Loaded {windows.shape[0]} windows across {windows.shape[1]} channels.")
  for cond in conditions_of_interest:
    label = CONDITION_LABELS.get(cond, str(cond))
    print(f"  Condition {cond} ({label}): {int(np.sum(y_sel == cond))} windows before QC")

  # 2. Two-tier artifact quality (shared channel-drop decision across all
  #    six conditions), per-condition averaged connectivity matrices, and
  #    permutation significance testing.
  channel_names = resolve_channel_names(windows.shape[1])
  method_fn = METHODS[args.method]
  result = extract_connectivity_by_condition(
      windows,
      y_sel,
      fs=250.0,
      method_fn=method_fn,
      channel_names=channel_names,
      verbose=True,
      n_permutations=args.permutations,
      n_jobs=args.n_jobs,
      random_state=args.seed,
      alpha=args.alpha,
  )

  print("\n--- Condition-averaged connectivity summary ---")
  for cond, entry in result["by_condition"].items():
    label = CONDITION_LABELS.get(cond, str(cond))
    if entry["matrix"] is None:
      print(f"  {label}: NO CLEAN WINDOWS")
      continue
    n = entry["matrix"].shape[0]
    mean_conn = entry["matrix"][np.triu_indices(n, k=1)].mean()
    print(
        f"  {label:22s} | n={entry['n_windows']:3d} windows | "
        f"band={entry['band'][0]:.2f}-{entry['band'][1]:.2f} Hz | "
        f"mean {args.method}={mean_conn:.4f}"
    )
  if result["channels_dropped"]:
    print(f"Channels dropped for this subject: {result['channels_dropped']}")

  if result["diffs"] is not None:
    print("\n--- Stimulus vs rest, SAME frequency band (fair comparison) ---")
    for cond, diff_matrix in result["diffs"].items():
      label = CONDITION_LABELS.get(cond, str(cond))
      n = diff_matrix.shape[0]
      stim_val = result["by_condition"][cond]["matrix"][np.triu_indices(n, k=1)].mean()
      base_val = result["baseline_matched"][cond]["matrix"][np.triu_indices(n, k=1)].mean()
      mean_diff = diff_matrix[np.triu_indices(n, k=1)].mean()
      print(
          f"  {label:22s} | stimulus={stim_val:.4f} | rest (same band)={base_val:.4f} | "
          f"difference={mean_diff:+.4f}"
      )

  # Permutation testing report
  perm_tests = result.get("permutation_tests")
  if perm_tests:
    print(f"\n{'='*115}")
    print(f"--- Permutation Significance Test ({args.permutations} iterations, alpha={args.alpha}) ---")
    print(f"{'='*115}")
    header = (
        f"{'Condition':12s} | {'Stim':6s} | {'Rest':6s} | {'Obs Diff':9s} | {'95% Null CI':18s} | "
        f"{'p (increase)':14s} | {'p (2-tailed)':12s} | {'Sig Edges (p<0.05 / FDR)':26s} | {'Conclusion'}"
    )
    print(header)
    print("-" * 115)
    for cond, perm in perm_tests.items():
      if perm is None:
        continue
      label = CONDITION_LABELS.get(cond, str(cond))
      n = result["diffs"][cond].shape[0]
      stim_v = result["by_condition"][cond]["matrix"][np.triu_indices(n, k=1)].mean()
      base_v = result["baseline_matched"][cond]["matrix"][np.triu_indices(n, k=1)].mean()
      diff_v = perm["obs_mean_diff"]
      ci_str = f"[{perm['null_ci_95'][0]:+.3f}, {perm['null_ci_95'][1]:+.3f}]"

      p_1t = perm["p_val_global_one_tailed"]
      stars_1t = "***" if p_1t < 0.001 else "**" if p_1t < 0.01 else "*" if p_1t < 0.05 else "ns"
      p_1t_str = f"p={p_1t:.4f} {stars_1t:2s}"

      p_2t = perm["p_val_global_two_tailed"]
      stars_2t = "***" if p_2t < 0.001 else "**" if p_2t < 0.01 else "*" if p_2t < 0.05 else "ns"
      p_2t_str = f"p={p_2t:.4f} {stars_2t:2s}"

      edges_str = f"{perm['n_significant_edges_uncorrected']:2d}/{perm['total_edges']:2d} ({perm['n_significant_edges_fdr']:2d} FDR)"

      if p_1t < args.alpha:
        conclusion = "Significant (Genuine synchrony)"
      elif diff_v < 0 and p_2t >= args.alpha:
        conclusion = "Not significant (Null noise / ceiling effect)"
      elif p_1t >= args.alpha and p_2t >= args.alpha:
        conclusion = "Not significant (Within null variance)"
      else:
        conclusion = "Non-significant bidirectional change"

      print(
          f"{label:12s} | {stim_v:.4f} | {base_v:.4f} | {diff_v:+.4f}    | {ci_str:18s} | "
          f"{p_1t_str:14s} | {p_2t_str:12s} | {edges_str:26s} | {conclusion}"
      )
    print("=" * 115)

  # 3. Render and save all panels.
  kept_names = result["channels_kept"] or channel_names or [f"Ch{i}" for i in range(windows.shape[1])]

  out_dir = project_root / "outputs" / "figures" / "connectivity"
  out_dir.mkdir(parents=True, exist_ok=True)

  fig_raw = plot_all_conditions_panel(result["by_condition"], kept_names)
  raw_path = out_dir / f"connectivity_{args.method}_{args.subject}.png"
  fig_raw.savefig(raw_path, dpi=150, facecolor="black")
  print(f"\nRaw connectivity panel saved to: {raw_path}")

  if result["diffs"] is not None:
    # Difference panel with permutation p-values and asterisks in titles
    fig_diff = plot_difference_panel(
        result["diffs"], kept_names, permutation_tests=perm_tests, mask_nonsignificant=False
    )
    diff_path = out_dir / f"connectivity_{args.method}_diff_{args.subject}.png"
    fig_diff.savefig(diff_path, dpi=150, facecolor="black")
    print(f"Difference panel saved to: {diff_path}")

    # Significant-only difference panel (masked non-significant edges)
    if perm_tests:
      fig_diff_sig = plot_difference_panel(
          result["diffs"], kept_names, permutation_tests=perm_tests, mask_nonsignificant=True, use_fdr=True
      )
      diff_sig_path = out_dir / f"connectivity_{args.method}_diff_sig_fdr_{args.subject}.png"
      fig_diff_sig.savefig(diff_sig_path, dpi=150, facecolor="black")
      print(f"Significant-only (FDR) difference panel saved to: {diff_sig_path}")

      fig_null = plot_permutation_null_distributions(perm_tests)
      null_path = out_dir / f"connectivity_{args.method}_null_dist_{args.subject}.png"
      fig_null.savefig(null_path, dpi=150, facecolor="white", bbox_inches="tight")
      print(f"Null distribution histograms saved to: {null_path}")


if __name__ == "__main__":
  main()