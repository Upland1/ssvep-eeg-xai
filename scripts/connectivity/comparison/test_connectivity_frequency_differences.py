"""Between-frequency SSVEP connectivity tests with a split-half reliability ceiling.

H0: trial windows from A and B are exchangeable. Trials are shuffled as whole
units; matched baselines stay fixed.

Reliability: each delta matrix is split into random halves, edge profiles are
correlated, and Spearman-Brown correction gives full-length reliability. The
observed similarity is compared against sqrt(rel_A * rel_B).

Limits: approximate trial IDs give only rough p-values; effects mix SNR and
true network differences.
"""

import argparse
import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import chi2

from scripts.connectivity.analysis.analyze_connectivity_cohort import (
    METHODS,
    SUB_WINDOWS_PER_TRIAL,
    discover_subjects,
    resolve_channel_names,
)
from scripts.connectivity.comparison.compare_connectivity_frequencies import (
    CONDITIONS,
    FREQ_LABEL,
    select_responders,
)
from src.features.connectivity.connectivity_extraction import (
    BASELINE_CONDITION,
    CONDITION_FREQS,
    _trial_permutation_split,
    benjamini_hochberg,
    extract_connectivity_by_condition,
)
from src.preprocessing.cv_utils import build_trial_ids

FS = 250.0
HALF_WIDTH = 1.0


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def trial_alignment_issues(y: np.ndarray, per_trial: dict[int, int]) -> list[str]:
  """Describe same-label blocks that are not divisible by their trial length."""
  issues, i, n = [], 0, len(y)
  while i < n:
    j = i
    while j < n and y[j] == y[i]:
      j += 1
    block, k = j - i, per_trial[int(y[i])]
    if block % k:
      issues.append(f"label {int(y[i])}: block of {block} not divisible by {k}")
    i = j
  return issues


def _offdiag(m: np.ndarray) -> np.ndarray:
  return m[~np.eye(m.shape[0], dtype=bool)]


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
  if a.std() == 0.0 or b.std() == 0.0:
    return float("nan")
  return float(np.corrcoef(a, b)[0, 1])


def _band(cond: int) -> tuple[float, float]:
  f0 = CONDITION_FREQS[cond]
  return (f0 - HALF_WIDTH, f0 + HALF_WIDTH)


def _pair_stats(da: np.ndarray, db: np.ndarray) -> tuple[float, float]:
  n = da.shape[0]
  triu = np.triu_indices(n, k=1)
  t_mag = float((da - db)[triu].mean())
  r = _pearson(_offdiag(da), _offdiag(db))
  return t_mag, (1.0 - r) if np.isfinite(r) else float("nan")


def fisher_combine(p_values: list[float]) -> float:
  p = np.clip(np.asarray([x for x in p_values if np.isfinite(x)], dtype=float), 1e-12, 1.0)
  if p.size == 0:
    return float("nan")
  return float(chi2.sf(-2.0 * np.sum(np.log(p)), 2 * p.size))


# ---------------------------------------------------------------------------
# Split-half reliability of one delta matrix
# ---------------------------------------------------------------------------

def _split_trials(rng: np.random.RandomState, tids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """Random trial-level halves (alternate assignment of shuffled trials)."""
  trials = rng.permutation(np.unique(tids))
  h1 = np.isin(tids, trials[0::2])
  return np.flatnonzero(h1), np.flatnonzero(~h1)


def _split_half_once(seed, stim, stim_tids, base, base_tids, band, method_fn) -> float:
  rng = np.random.RandomState(seed)
  s1, s2 = _split_trials(rng, stim_tids)
  b1, b2 = _split_trials(rng, base_tids)
  d1 = method_fn(stim[s1], fs=FS, freq_band=band) - method_fn(base[b1], fs=FS, freq_band=band)
  d2 = method_fn(stim[s2], fs=FS, freq_band=band) - method_fn(base[b2], fs=FS, freq_band=band)
  return _pearson(_offdiag(d1), _offdiag(d2))


def split_half_reliability(stim, stim_tids, base, base_tids, band, method_fn, n_splits, seed, n_jobs) -> float:
  seeds = np.random.RandomState(seed).randint(0, 2**31 - 1, size=n_splits)
  rs = Parallel(n_jobs=n_jobs)(
      delayed(_split_half_once)(int(s), stim, stim_tids, base, base_tids, band, method_fn) for s in seeds
  )
  rs = np.array([r for r in rs if np.isfinite(r)])
  if rs.size == 0:
    return float("nan")
  r_mean = float(np.tanh(np.mean(np.arctanh(np.clip(rs, -0.999, 0.999)))))
  return 2.0 * r_mean / (1.0 + r_mean)  # Spearman-Brown to full length


# ---------------------------------------------------------------------------
# Pairwise permutation test
# ---------------------------------------------------------------------------

def _pair_perm_worker(seed, pooled, pooled_tids, n_a, band_a, band_b, base_a, base_b, method_fn):
  rng = np.random.RandomState(seed)
  ia, ib = _trial_permutation_split(rng, pooled_tids, n_a)
  da = method_fn(pooled[ia], fs=FS, freq_band=band_a) - base_a
  db = method_fn(pooled[ib], fs=FS, freq_band=band_b) - base_b
  return da, db


def pairwise_permutation_test(
    stim_a, tid_a, stim_b, tid_b, cond_a, cond_b, delta_a, delta_b, base_a, base_b,
    method_fn, n_perm, seed, n_jobs, alpha=0.05,
) -> dict:
  if np.intersect1d(tid_a, tid_b).size:
    raise ValueError("trial ids of the two conditions overlap")
  pooled = np.concatenate([stim_a, stim_b], axis=0)
  pooled_tids = np.concatenate([tid_a, tid_b])
  seeds = np.random.RandomState(seed).randint(0, 2**31 - 1, size=n_perm)
  out = Parallel(n_jobs=n_jobs)(
      delayed(_pair_perm_worker)(
          int(s), pooled, pooled_tids, len(stim_a), _band(cond_a), _band(cond_b), base_a, base_b, method_fn
      ) for s in seeds
  )
  null_a = np.array([o[0] for o in out])
  null_b = np.array([o[1] for o in out])

  t_obs, d_obs = _pair_stats(delta_a, delta_b)
  null_stats = np.array([_pair_stats(a, b) for a, b in zip(null_a, null_b)])
  null_t, null_d = null_stats[:, 0], null_stats[:, 1]

  mu = float(np.mean(null_t))
  p_mag = float((1 + np.sum(np.abs(null_t - mu) >= abs(t_obs - mu))) / (n_perm + 1))
  valid_d = null_d[np.isfinite(null_d)]
  p_pat = (
      float((1 + np.sum(valid_d >= d_obs)) / (len(valid_d) + 1)) if np.isfinite(d_obs) and len(valid_d) else float("nan")
  )

  # Edge-level: which pairs differ? (two-sided, centred on each edge's null mean)
  n = delta_a.shape[0]
  triu = np.triu_indices(n, k=1)
  obs_edge = (delta_a - delta_b)[triu]
  null_edge = (null_a - null_b)[:, triu[0], triu[1]]
  mu_e = null_edge.mean(axis=0)
  p_edge = (1 + np.sum(np.abs(null_edge - mu_e) >= np.abs(obs_edge - mu_e), axis=0)) / (n_perm + 1)
  fdr = benjamini_hochberg(p_edge, alpha=alpha)

  return {
      "t_mag": t_obs, "p_mag": p_mag, "d_pat": d_obs, "p_pat": p_pat,
      "r_obs": 1.0 - d_obs if np.isfinite(d_obs) else float("nan"),
      "n_edges_fdr": int(fdr.sum()), "n_edges": int(len(fdr)),
  }


# ---------------------------------------------------------------------------
# Per-subject / per-method driver
# ---------------------------------------------------------------------------

def run_subject_method(sub, data_root, method_name, n_perm, n_splits, seed, n_jobs, skip_reliability):
  sub_dir = data_root / sub
  if not (sub_dir / "X_time_windows.npy").exists():
    print(f"[-] Skipping {sub}: X_time_windows.npy not found.")
    return [], []
  windows_raw = np.load(sub_dir / "X_time_windows.npy")
  y_raw = np.load(sub_dir / "y_labels.npy")
  conds = [BASELINE_CONDITION] + CONDITIONS
  mask = np.isin(y_raw, conds)
  y_sel = y_raw[mask]
  windows = windows_raw[mask] if windows_raw.shape[0] == len(y_raw) else windows_raw[: len(y_sel)]
  channel_names = resolve_channel_names(windows.shape[1])

  issues = trial_alignment_issues(y_sel, SUB_WINDOWS_PER_TRIAL)
  approx = bool(issues)
  if approx:
    print(f"  [!] {sub}: trial ids approximate ({'; '.join(issues)}) -- p-values for this subject are indicative only.")
  trial_ids = build_trial_ids(y_sel, sub_windows_per_trial=SUB_WINDOWS_PER_TRIAL)

  method_fn = METHODS[method_name]
  # n_permutations=0 -> only QC + observed matrices; we reuse its cleaned arrays.
  res = extract_connectivity_by_condition(
      windows, y_sel, fs=FS, method_fn=method_fn, channel_names=channel_names,
      verbose=False, n_permutations=0, trial_ids=trial_ids,
  )
  if res["diffs"] is None:
    print(f"[-] Skipping {sub} ({method_name}): no baseline windows after QC.")
    return [], []
  W, yc, tc = res["windows_clean"], res["y_clean"], res["trial_ids_clean"]
  base = W[yc == BASELINE_CONDITION]
  base_t = tc[yc == BASELINE_CONDITION]
  have = [c for c in CONDITIONS if c in res["diffs"]]

  rel_rows, rel = [], {}
  for c in have:
    if skip_reliability:
      rel[c] = float("nan")
      continue
    rel[c] = split_half_reliability(
        W[yc == c], tc[yc == c], base, base_t, _band(c), method_fn, n_splits, seed + c, n_jobs
    )
    rel_rows.append({"subject": sub, "method": method_name, "condition": c, "frequency": FREQ_LABEL[c],
                     "split_half_reliability": rel[c]})
    print(f"    reliability {FREQ_LABEL[c]:>8s}: {rel[c]:+.2f}")

  pair_rows = []
  for ca, cb in itertools.combinations(have, 2):
    r = pairwise_permutation_test(
        W[yc == ca], tc[yc == ca], W[yc == cb], tc[yc == cb], ca, cb,
        res["diffs"][ca], res["diffs"][cb],
        res["baseline_matched"][ca]["matrix"], res["baseline_matched"][cb]["matrix"],
        method_fn, n_perm, seed + 1000 * ca + cb, n_jobs,
    )
    ceiling = float(np.sqrt(rel[ca] * rel[cb])) if rel[ca] > 0 and rel[cb] > 0 else float("nan")
    r_dis = float(np.clip(r["r_obs"] / ceiling, -1, 1)) if np.isfinite(ceiling) and ceiling > 0.1 else float("nan")
    pair_rows.append({
        "subject": sub, "method": method_name, "freq_a": FREQ_LABEL[ca], "freq_b": FREQ_LABEL[cb],
        **r, "reliability_ceiling": ceiling, "r_disattenuated": r_dis, "trial_ids_approximate": approx,
    })
    print(f"    {FREQ_LABEL[ca]:>8s} vs {FREQ_LABEL[cb]:<8s} | mag {r['t_mag']:+.3f} p={r['p_mag']:.3f} | "
          f"pattern r={r['r_obs']:+.2f} p={r['p_pat']:.3f} | edges FDR {r['n_edges_fdr']}/{r['n_edges']} | ceiling {ceiling:.2f}")

  df = pd.DataFrame(pair_rows)
  if len(df):
    df["sig_bh_mag"] = benjamini_hochberg(df["p_mag"].values)
    df["sig_bh_pat"] = benjamini_hochberg(df["p_pat"].fillna(1.0).values)
  return df.to_dict("records"), rel_rows


def main():
  parser = argparse.ArgumentParser(description="Formal between-frequency connectivity tests + reliability ceiling.")
  parser.add_argument("--methods", nargs="+", default=["coherence", "wpli"], choices=list(METHODS))
  parser.add_argument("--subjects", nargs="+", default=None, help="Default: all discovered subjects.")
  parser.add_argument("--permutations", type=int, default=500,
                       help="Permutations per frequency pair (default 500; use >=1000 for final numbers).")
  parser.add_argument("--splits", type=int, default=10, help="Random split-half repetitions per frequency (default 10).")
  parser.add_argument("--skip-reliability", action="store_true")
  parser.add_argument("--responders", nargs="+", default=None)
  parser.add_argument("--responder-threshold", type=float, default=80.0)
  parser.add_argument("--benchmark-csv", default=None)
  parser.add_argument("--n-jobs", type=int, default=-1)
  parser.add_argument("--seed", type=int, default=42)
  args = parser.parse_args()

  project_root = Path(__file__).resolve().parents[2]  # scripts/connectivity/<file> -> project root
  data_root = project_root / "data" / "processed"
  out_dir = project_root / "reports" / "connectivity_comparison"
  out_dir.mkdir(parents=True, exist_ok=True)
  bench_csv = Path(args.benchmark_csv) if args.benchmark_csv else project_root / "reports" / "n_subjects_benchmark.csv"

  subjects = args.subjects or discover_subjects(data_root)
  responders, how = select_responders(subjects, args.responders, bench_csv, args.responder_threshold)
  print("=" * 90)
  print(f"BETWEEN-FREQUENCY TESTS  methods={args.methods}  subjects={subjects}  perms={args.permutations}")
  print(f"Responders ({how}): {responders or 'none'}")
  print("=" * 90)

  for method in args.methods:
    pair_all, rel_all = [], []
    for sub in subjects:
      print(f"\n[{sub}] {method}")
      pr, rr = run_subject_method(sub, data_root, method, args.permutations, args.splits,
                                  args.seed, args.n_jobs, args.skip_reliability)
      pair_all += pr
      rel_all += rr
    if not pair_all:
      continue
    df = pd.DataFrame(pair_all)
    df.to_csv(out_dir / f"{method}_pairwise_tests.csv", index=False)
    if rel_all:
      pd.DataFrame(rel_all).to_csv(out_dir / f"{method}_split_half_reliability.csv", index=False)

    print("\n" + "=" * 90)
    print(f"[{method}] GROUP SUMMARY -- responders only (non-responders have no SSVEP to compare)")
    print("=" * 90)
    if rel_all:
      rel_df = pd.DataFrame(rel_all)
      rel_df = rel_df[rel_df["subject"].isin(responders)]
      piv = rel_df.pivot_table(index="subject", columns="frequency", values="split_half_reliability")
      print("Split-half reliability of each delta matrix (Spearman-Brown; >~0.5 is usable):")
      print(piv.reindex(columns=[FREQ_LABEL[c] for c in CONDITIONS]).round(2).to_string())
    g = df[df["subject"].isin(responders)]
    if g.empty:
      print("No responder results.")
      continue
    print("\nPer frequency pair: subjects with p<0.05 / n, Fisher-combined p, mean disattenuated r")
    print(f"{'pair':<20s} | {'MAGNITUDE':^26s} | {'PATTERN':^40s}")
    for (fa, fb), grp in g.groupby(["freq_a", "freq_b"], sort=False):
      k = len(grp)
      mag = f"{int((grp.p_mag < .05).sum())}/{k}  comb p={fisher_combine(grp.p_mag.tolist()):.3f}"
      pat = (f"{int((grp.p_pat < .05).sum())}/{k}  comb p={fisher_combine(grp.p_pat.tolist()):.3f}  "
             f"r_dis={np.nanmean(grp.r_disattenuated) if grp.r_disattenuated.notna().any() else float('nan'):+.2f}")
      print(f"{fa:>8s} vs {fb:<8s} | {mag:^26s} | {pat:^40s}")
    if g["trial_ids_approximate"].any():
      print("\n[!] Includes subjects with approximate trial ids:",
            sorted(g.loc[g["trial_ids_approximate"], "subject"].unique()))

  print(f"\nCSVs saved under: {out_dir}")


if __name__ == "__main__":
  main()