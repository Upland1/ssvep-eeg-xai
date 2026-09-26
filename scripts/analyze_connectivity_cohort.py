"""
Cohort-wide functional connectivity analysis: all 4 methods, every
discovered subject.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.connectivity_extraction import (
    BASELINE_CONDITION,
    CONDITION_FREQS,
    compute_condition_coherence,
    compute_condition_pdc,
    compute_condition_plv,
    compute_condition_wpli,
    extract_connectivity_by_condition,
)
from src.visualization.connectivity_plots import CONDITION_LABELS

FULL_MONTAGE = ["PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"]
SUBJECT_DIR_PATTERN = re.compile(r"^S\d+$")

METHODS = {
    "coherence": compute_condition_coherence,
    "plv": compute_condition_plv,
    "wpli": compute_condition_wpli,
    "pdc": compute_condition_pdc,
}


def resolve_channel_names(n_channels: int, full_montage: list[str] = FULL_MONTAGE) -> list[str] | None:
  if n_channels == len(full_montage):
    return list(full_montage)
  if n_channels == len(full_montage) - 1:
    return full_montage[1:]
  return None


def discover_subjects(data_root: Path) -> list[str]:
  if not data_root.exists():
    return []
  return sorted(
      p.name for p in data_root.iterdir()
      if p.is_dir() and SUBJECT_DIR_PATTERN.match(p.name)
  )


def process_subject_method(
    sub: str, data_root: Path, method_name: str, method_fn, n_permutations: int, n_jobs: int, seed: int, alpha: float
) -> list[dict]:
  """Run one (subject, method) pair; returns one summary row per frequency."""
  sub_dir = data_root / sub
  if not (sub_dir / "X_time_windows.npy").exists():
    print(f"[-] Skipping {sub}: X_time_windows.npy not found.")
    return []

  windows_raw = np.load(sub_dir / "X_time_windows.npy")
  y_raw = np.load(sub_dir / "y_labels.npy")

  conditions_of_interest = [BASELINE_CONDITION] + sorted(CONDITION_FREQS)
  mask = np.isin(y_raw, conditions_of_interest)
  y_sel = y_raw[mask]
  windows = windows_raw[mask] if windows_raw.shape[0] == len(y_raw) else windows_raw[: len(y_sel)]
  channel_names = resolve_channel_names(windows.shape[1])

  try:
    result = extract_connectivity_by_condition(
        windows, y_sel, fs=250.0, method_fn=method_fn, channel_names=channel_names,
        verbose=False, n_permutations=n_permutations, n_jobs=n_jobs,
        random_state=seed, alpha=alpha,
    )
  except ValueError as e:
    print(f"[-] Skipping {sub} ({method_name}): {e}")
    return []

  perm_tests = result.get("permutation_tests") or {}
  rows = []
  for cond, f0 in CONDITION_FREQS.items():
    perm = perm_tests.get(cond)
    if perm is None or result["diffs"] is None or cond not in result["diffs"]:
      continue
    rows.append({
        "subject": sub,
        "method": method_name,
        "condition": cond,
        "frequency_hz": f0,
        "label": CONDITION_LABELS.get(cond, str(cond)),
        "obs_diff": perm["obs_mean_diff"],
        "p_one_tailed": perm["p_val_global_one_tailed"],
        "significant_uncorrected": perm["p_val_global_one_tailed"] < alpha,
        "n_edges_sig_uncorrected": perm["n_significant_edges_uncorrected"],
        "n_edges_sig_fdr": perm["n_significant_edges_fdr"],
        "total_edges": perm["total_edges"],
        "n_channels": windows.shape[1],
        "channels_dropped": ", ".join(result["channels_dropped"] or []) or "none",
    })
  return rows


def main():
  parser = argparse.ArgumentParser(
      description="Run all 4 connectivity methods across every discovered subject and summarize."
  )
  parser.add_argument("--permutations", type=int, default=100,
                       help="Permutations per (subject, method, frequency) (default: 100 -- "
                            "lower than the single-subject default of 200 since this multiplies "
                            "across the whole cohort; raise it for the final numbers).")
  parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs for each permutation test (default: -1).")
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--alpha", type=float, default=0.05)
  parser.add_argument("--methods", nargs="+", default=list(METHODS.keys()), choices=list(METHODS.keys()),
                       help="Which methods to run (default: all 4).")
  args = parser.parse_args()

  project_root = Path(__file__).resolve().parent.parent
  data_root = project_root / "data" / "processed"
  subjects = discover_subjects(data_root)

  print("=" * 80)
  print(f"   COHORT CONNECTIVITY SUMMARY -- methods={args.methods} (N={len(subjects)} subjects)")
  print("=" * 80)
  if not subjects:
    print(f"No subject folders matching 'S<digits>' found under {data_root}.")
    return

  all_rows = []
  for sub in subjects:
    for method_name in args.methods:
      print(f"\n[{sub}] running {method_name}...")
      rows = process_subject_method(
          sub, data_root, method_name, METHODS[method_name],
          args.permutations, args.n_jobs, args.seed, args.alpha,
      )
      all_rows.extend(rows)
      for r in rows:
        global_flag = "**" if r["p_one_tailed"] < 0.01 else ("*" if r["significant_uncorrected"] else "ns")
        print(
            f"    {r['label']:10s} | diff={r['obs_diff']:+.4f} | p={r['p_one_tailed']:.4f} {global_flag:2s} (global) | "
            f"edges {r['n_edges_sig_uncorrected']}/{r['total_edges']} uncorrected, {r['n_edges_sig_fdr']} FDR-robust"
        )

  if not all_rows:
    print("\nNo results collected -- check that data/processed/<subject>/ folders contain the expected files.")
    return

  df = pd.DataFrame(all_rows)

  # Cohort-level pivot: how many subjects show a significant increase
  # (FDR-backed, i.e. at least one edge survives correction) at each
  # frequency, per method -- the direct answer to "does the pattern hold".
  df["fdr_backed"] = df["n_edges_sig_fdr"] > 0
  pivot_fdr = df.pivot_table(index="method", columns="label", values="fdr_backed", aggfunc="sum")
  pivot_uncorr = df.pivot_table(index="method", columns="label", values="significant_uncorrected", aggfunc="sum")
  n_subjects_with_data = df["subject"].nunique()

  print("\n" + "=" * 80)
  print(f"COHORT PATTERN A -- subjects with >=1 individual channel-pair surviving FDR correction, out of {n_subjects_with_data}")
  print("(strictest criterion: at least one specific pair is individually robust to multiple-comparison correction)")
  print("=" * 80)
  print(pivot_fdr.to_string())
  print(f"\nCOHORT PATTERN B -- subjects where the GLOBAL aggregate test (mean diff across all pairs) reaches p<{args.alpha}")
  print("(looser criterion: the average increase across the whole network is real even if no single pair alone is)")
  print(pivot_uncorr.to_string())
  print("\nThese two can disagree in both directions -- a real, diffuse increase spread thinly across many")
  print("pairs can pass B but fail A; conversely a couple of very strong pairs can pass A even when B is borderline.")
  print("Report both, don't collapse them into one star.")

  out_dir = project_root / "reports"
  out_dir.mkdir(parents=True, exist_ok=True)
  out_path = out_dir / "connectivity_cohort_summary.csv"
  df.to_csv(out_path, index=False)
  print(f"\nFull per-subject/method/frequency results saved to: {out_path}")


if __name__ == "__main__":
  main()