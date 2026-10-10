"""Run cohort-wide connectivity analysis for all discovered subjects."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _find_root(start: Path) -> Path:
  for cand in [start, *start.parents]:
    if (cand / "configs" / "pipeline_config.yaml").exists():
      return cand
  return start.parents[3]


PROJECT_ROOT = _find_root(Path(__file__).resolve())
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.connectivity.connectivity_extraction import (  # noqa: E402
    BASELINE_CONDITION,
    CONDITION_FREQS,
    compute_condition_coherence,
    compute_condition_pdc,
    compute_condition_plv,
    compute_condition_wpli,
    extract_connectivity_by_condition,
)
from src.io.subject_data import (  # noqa: E402
    FULL_MONTAGE,
    QUALITY_MODES,
    discover_subjects,
    load_subject,
    resolve_channel_names,
)
from src.stats.nbs import nbs_sweep  # noqa: E402
from src.visualization.connectivity_plots import CONDITION_LABELS  # noqa: E402

# Legacy sub-window counts; built folders carry true trial ids.
SUB_WINDOWS_PER_TRIAL = {201: 2, 101: 5, 102: 5, 103: 5, 104: 5, 105: 5}

METHODS = {
    "coherence": compute_condition_coherence,
    "plv": compute_condition_plv,
    "wpli": compute_condition_wpli,
    "pdc": compute_condition_pdc,
}

__all__ = ["METHODS", "SUB_WINDOWS_PER_TRIAL", "FULL_MONTAGE", "discover_subjects", "resolve_channel_names",
           "output_suffix"]


def output_suffix(data_root: Path) -> str:
  """'' for the legacy data/processed folder, '_<name>' otherwise."""
  return "" if data_root.name == "processed" else f"_{data_root.name}"


def save_edge_level_results(result: dict, out_path: Path, nbs_masks: dict | None = None) -> None:
  """Save per-condition connectivity matrices and significance masks."""
  payload = {"channels_kept": np.array(result["channels_kept"] or [], dtype=str)}
  perm_tests = result.get("permutation_tests") or {}
  for cond in CONDITION_FREQS:
    entry = result["by_condition"].get(cond)
    if entry is None or entry["matrix"] is None or result["diffs"] is None or cond not in result["diffs"]:
      continue
    payload[f"stim_{cond}"] = entry["matrix"]
    payload[f"base_{cond}"] = result["baseline_matched"][cond]["matrix"]
    payload[f"diff_{cond}"] = result["diffs"][cond]
    perm = perm_tests.get(cond)
    if perm is not None:
      payload[f"p_{cond}"] = perm["p_val_edges_one_tailed"]
      payload[f"fdr_{cond}"] = perm["edge_fdr_mask"]
    if nbs_masks and cond in nbs_masks:
      payload[f"nbs_{cond}"] = nbs_masks[cond]
  out_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(out_path, **payload)


def process_subject_method(
    sub: str, data_root: Path, method_name: str, method_fn, n_permutations: int, n_jobs: int, seed: int, alpha: float,
    edges_dir: Path | None = None, quality_mode: str = "shared",
    nbs_thresholds: tuple = (1.5, 2.0, 2.5, 3.0), nbs_primary: float = 2.0,
) -> list[dict]:
  """Run one (subject, method) pair; returns one summary row per frequency."""
  sub_dir = data_root / sub
  if not (sub_dir / "X_time_windows.npy").exists():
    print(f"[-] Skipping {sub}: X_time_windows.npy not found.")
    return []

  conditions_of_interest = [BASELINE_CONDITION] + sorted(CONDITION_FREQS)
  d = load_subject(sub_dir, conditions_of_interest, quality_mode=quality_mode)
  windows, y_sel = d["windows"], d["y"]

  try:
    result = extract_connectivity_by_condition(
        windows, y_sel, fs=d["fs"], method_fn=method_fn, channel_names=d["channel_names"],
        verbose=False, n_permutations=n_permutations, n_jobs=n_jobs,
        random_state=seed, alpha=alpha, trial_ids=d["trial_ids"], quality=d["quality"],
    )
  except ValueError as e:
    print(f"[-] Skipping {sub} ({method_name}): {e}")
    return []

  perm_tests = result.get("permutation_tests") or {}

  # Run NBS on the existing permutation results.
  nbs = {}
  for cond, perm in perm_tests.items():
    if perm is None:
      continue
    obs, null = perm["obs_diff"], perm["null_diff_matrices"]
    if method_name == "pdc":   # Symmetrize directed connectivity.
      obs, null = (obs + obs.T) / 2, (null + null.transpose(0, 2, 1)) / 2
    nbs[cond] = nbs_sweep(obs, null, thresholds=tuple(sorted(set(nbs_thresholds) | {nbs_primary})), alpha=alpha)

  if edges_dir is not None:
    save_edge_level_results(result, edges_dir / f"{sub}_{method_name}.npz",
                            {c: r[nbs_primary]["sig_mask"] for c, r in nbs.items()})

  yc = result["y_clean"]
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
        "n_channels": len(result["channels_kept_idx"]),
        "channels_dropped": ", ".join(result["channels_dropped"] or []) or "none",
        "permutation_unit": perm["permutation_unit"],
        "nbs_threshold": nbs_primary,
        "nbs_max_size": nbs[cond][nbs_primary]["max_size"],
        "nbs_size_fraction": round(nbs[cond][nbs_primary]["max_size"] / nbs[cond][nbs_primary]["possible_edges"], 3),
        "nbs_p": nbs[cond][nbs_primary]["p_max"],
        "nbs_n_sig_edges": nbs[cond][nbs_primary]["n_sig_edges"],
        "nbs_significant": nbs[cond][nbs_primary]["n_sig_components"] > 0,
        **{f"nbs_size_z{t:g}": nbs[cond][t]["max_size"] for t in nbs_thresholds},
        **{f"nbs_p_z{t:g}": nbs[cond][t]["p_max"] for t in nbs_thresholds},
        "n_stim_windows": int(np.sum(yc == cond)),
        "n_baseline_windows": int(np.sum(yc == BASELINE_CONDITION)),
        "fs": d["fs"],
        "source": d["source"],
    })
  return rows


def main():
  parser = argparse.ArgumentParser(
      description="Run all 4 connectivity methods across every discovered subject and summarize."
  )
  parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "processed",
                      help="Folder with one S<digits> sub-folder per subject (default: data/processed)")
  parser.add_argument("--quality", choices=QUALITY_MODES, default="shared",
                      help="Quality decision for built folders (default: shared).")
  parser.add_argument("--permutations", type=int, default=100,
                       help="Permutations per (subject, method, frequency) (default: 100 -- "
                            "lower than the single-subject default of 200 since this multiplies "
                            "across the whole cohort; raise it for the final numbers).")
  parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs for each permutation test (default: -1).")
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--alpha", type=float, default=0.05)
  parser.add_argument("--methods", nargs="+", default=list(METHODS.keys()), choices=list(METHODS.keys()),
                       help="Which methods to run (default: all 4).")
  parser.add_argument("--subjects", nargs="+", default=None, help="Default: all discovered subjects.")
  parser.add_argument("--stats", choices=["nbs", "fdr"], default="nbs",
                      help="Edge-level criterion used in the summary (default: nbs). Both are always saved.")
  parser.add_argument("--nbs-thresholds", type=float, nargs="+", default=[1.5, 2.0, 2.5, 3.0],
                      help="Primary pseudo-z thresholds for the NBS sweep (default: 1.5 2.0 2.5 3.0).")
  parser.add_argument("--nbs-primary", type=float, default=2.0,
                      help="Threshold used for the significance mask and the summary (default: 2.0).")
  parser.add_argument("--edges-dir", default=None,
                       help="Where to save per-(subject, method) edge-level .npz files "
                            "(default: <project>/reports/connectivity_edges[_<data-root>]).")
  args = parser.parse_args()

  data_root = args.data_root if args.data_root.is_absolute() else PROJECT_ROOT / args.data_root
  suffix = output_suffix(data_root)
  edges_dir = Path(args.edges_dir) if args.edges_dir else PROJECT_ROOT / "reports" / f"connectivity_edges{suffix}"
  subjects = args.subjects or discover_subjects(data_root)

  print("=" * 80)
  print(f"   COHORT CONNECTIVITY SUMMARY -- methods={args.methods} (N={len(subjects)} subjects)")
  print(f"   data: {data_root} | quality: {args.quality}")
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
          edges_dir=edges_dir, quality_mode=args.quality,
          nbs_thresholds=tuple(args.nbs_thresholds), nbs_primary=args.nbs_primary,
      )
      all_rows.extend(rows)
      if rows:
        print(f"    fs={rows[0]['fs']:g} | {rows[0]['n_channels']} channels | "
              f"{rows[0]['n_baseline_windows']} baseline windows | {rows[0]['source']}")
      for r in rows:
        global_flag = "**" if r["p_one_tailed"] < 0.01 else ("*" if r["significant_uncorrected"] else "ns")
        print(
            f"    {r['label']:10s} | diff={r['obs_diff']:+.4f} | p={r['p_one_tailed']:.4f} {global_flag:2s} (global) | "
            f"edges {r['n_edges_sig_uncorrected']}/{r['total_edges']} uncorrected, {r['n_edges_sig_fdr']} FDR | "
            f"NBS z>{r['nbs_threshold']:g}: largest component {r['nbs_max_size']} edges "
            f"({100 * r['nbs_size_fraction']:.0f}%) p={r['nbs_p']:.3f}"
        )

  if not all_rows:
    print("\nNo results collected -- check that the data root contains the expected subject folders.")
    return

  df = pd.DataFrame(all_rows)

  # Count subjects with significant effects by method and frequency.
  df["fdr_backed"] = df["n_edges_sig_fdr"] > 0
  edge_col, edge_name = (("nbs_significant", f"a significant NBS component (z > {args.nbs_primary:g})")
                         if args.stats == "nbs" else ("fdr_backed", ">=1 channel-pair surviving FDR correction"))
  pivot_fdr = df.pivot_table(index="method", columns="label", values=edge_col, aggfunc="sum")
  pivot_uncorr = df.pivot_table(index="method", columns="label", values="significant_uncorrected", aggfunc="sum")
  n_subjects_with_data = df["subject"].nunique()

  print("\n" + "=" * 80)
  print(f"COHORT PATTERN A -- subjects with {edge_name}, out of {n_subjects_with_data}")
  print("(network-level criterion: a connected set of channel pairs larger than chance)" if args.stats == "nbs"
        else "(strictest criterion: at least one specific pair is individually robust to multiple-comparison correction)")
  print("=" * 80)
  print(pivot_fdr.to_string())
  print(f"\nCOHORT PATTERN B -- subjects where the GLOBAL aggregate test (mean diff across all pairs) reaches p<{args.alpha}")
  print("(looser criterion: the average increase across the whole network is real even if no single pair alone is)")
  print(pivot_uncorr.to_string())
  print("\nThese two can disagree in both directions -- a real, diffuse increase spread thinly across many")
  print("pairs can pass B but fail A; conversely a couple of very strong pairs can pass A even when B is borderline.")
  print("Report both, don't collapse them into one star.")

  # Compare component sizes across subjects.
  print("\n" + "=" * 80)
  print(f"GRAPH SIZE -- largest supra-threshold component (z > {args.nbs_primary:g}) as % of possible edges")
  print("(* = component significant by NBS, p < alpha)")
  print("=" * 80)
  df["size_cell"] = [f"{100 * f:3.0f}%{'*' if sig else ' '}" for f, sig in zip(df["nbs_size_fraction"], df["nbs_significant"])]
  for method, g in df.groupby("method"):
    piv = g.pivot_table(index="subject", columns="label", values="size_cell", aggfunc="first")
    print(f"\n[{method}]")
    print(piv.to_string())
  sweep_cols = [c for c in df.columns if c.startswith("nbs_size_z")]
  sweep = df.groupby(["method", "label"])[sweep_cols].mean().round(1)
  print("\nThreshold sweep -- mean largest-component size (edges) across subjects:")
  print(sweep.to_string())
  df = df.drop(columns=["size_cell"])

  out_dir = PROJECT_ROOT / "reports"
  out_dir.mkdir(parents=True, exist_ok=True)
  out_path = out_dir / f"connectivity_cohort_summary{suffix}.csv"
  df.to_csv(out_path, index=False)
  print(f"\nFull per-subject/method/frequency results saved to: {out_path}")
  print(f"Edge-level matrices (for cross-frequency comparison) saved under: {edges_dir}")


if __name__ == "__main__":
  main()