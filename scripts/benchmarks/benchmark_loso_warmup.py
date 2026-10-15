"""LOSO with online (warm-up) per-subject normalization.

``lda_subjnorm`` in benchmark_loso.py z-scores the held-out subject's features
with statistics from that subject's whole session. A real new user has no
session yet, so here the statistics come only from the first K trials of the
recording (in time order, unlabeled). Those warm-up trials are not scored.

For every held-out subject and K the script reports, on the same scored
windows (all trials after the first K):
  full     statistics from the whole session (as lda_subjnorm)
  warmup   statistics from the first K trials only
  none     no per-subject normalization (as lda)

Build the data first, for example:
  python scripts/build_windows.py --window-sec 1 --stride-sec 0.25 --banks fbcca fbcsp

Usage from the repo root:
  python scripts/benchmarks/benchmark_loso_warmup.py --warmup-trials 1 2 5 10
  python scripts/benchmarks/benchmark_loso_warmup.py --channels PO3 POz PO4 O1 Oz O2
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_loso import (  # noqa: E402
    PROJECT_ROOT,
    STIM_CONDITIONS,
    SUBJECT_DIR,
    SWEEP_DIR,
    choose_channels,
    lda_loso,
    load_subject,
)


def window_starts(sub_dir: Path) -> np.ndarray:
  """Return window start samples, matching load_subject's window selection."""
  y = np.load(sub_dir / "y_labels.npy")
  sel = np.load(sub_dir / "valid_mask.npy").astype(bool) & np.isin(y, STIM_CONDITIONS)
  return np.load(sub_dir / "window_starts.npy")[sel]


def features(model: dict, d: dict) -> np.ndarray:
  """FBCSP + FBCCA features from a fitted LOSO model, before normalization."""
  csp = [c.transform(d["bands_csp"][:, b]) for b, c in enumerate(model["csps"])]
  return np.hstack(csp + [d["X_cca"]])


def accuracy(model: dict, f: np.ndarray, y: np.ndarray) -> float:
  return float((model["lda"].predict(model["scaler"].transform(f)) == y).mean() * 100)


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--root", type=Path, default=PROJECT_ROOT / "data" / "processed_continuous",
                  help="Folder holding the w<len>_s<stride> datasets (same as the sweeps)")
  ap.add_argument("--window", type=float, default=1.0, help="Window length in s (default 1.0)")
  ap.add_argument("--stride-sec", type=float, default=0.25)
  ap.add_argument("--subjects", nargs="*", default=None, help="Subset of subjects (default: all found)")
  ap.add_argument("--channels", nargs="+", default=["common"],
                  help="common (default) | all | explicit names, e.g. PO3 POz PO4 O1 Oz O2")
  ap.add_argument("--warmup-trials", type=int, nargs="+", default=[1, 2, 5, 10])
  ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "reports" / "loso_warmup")
  args = ap.parse_args()

  data_dir = None
  for p in sorted(args.root.iterdir()) if args.root.exists() else []:
    m = SWEEP_DIR.match(p.name)
    if p.is_dir() and m and np.isclose(float(m.group(1)), args.window) and np.isclose(float(m.group(2)), args.stride_sec):
      data_dir = p
  if data_dir is None:
    print(f"No w{args.window:g}_s{args.stride_sec:g} folder under {args.root}. Build it with scripts/build_windows.py.")
    return
  sub_dirs = {p.name: p for p in sorted(data_dir.iterdir()) if p.is_dir() and SUBJECT_DIR.match(p.name)}
  if args.subjects:
    sub_dirs = {s: p for s, p in sub_dirs.items() if s in args.subjects}
  channels, sub_dirs = choose_channels(sub_dirs, args.channels)
  print("=" * 90)
  print(f"LOSO warm-up normalization | {len(sub_dirs)} subjects | data {data_dir}")
  print(f"channels: {channels} | warm-up trials {args.warmup_trials}")
  print("=" * 90)

  t0 = time.time()
  data = {s: load_subject(p, channels) for s, p in sub_dirs.items()}
  for s, p in sub_dirs.items():
    data[s]["start"] = window_starts(p)

  records = []
  for test_sub, test in data.items():
    train = [data[s] for s in data if s != test_sub]
    norm = lda_loso(train, test, subject_norm=True)[1]["model"]
    plain = lda_loso(train, test, subject_norm=False)[1]["model"]
    f, f_plain, y, tid = features(norm, test), features(plain, test), test["y"], test["trial_ids"]
    full = (f - f.mean(axis=0)) / (f.std(axis=0) + 1e-8)
    # Trial IDs follow class order, so sort trials by their first window to get recording order.
    trials = np.unique(tid)
    order = trials[np.argsort([test["start"][tid == t].min() for t in trials])]

    line = [f"  {test_sub}"]
    for k in args.warmup_trials:
      if k >= len(order):
        continue
      warm = np.isin(tid, order[:k])
      scored = ~warm
      span = test["start"][warm].max() - test["start"][warm].min()
      mu, sd = f[warm].mean(axis=0), f[warm].std(axis=0)
      records.append({
          "Subject": test_sub,
          "Warm-up trials": k,
          "Warm-up windows": int(warm.sum()),
          "Warm-up classes": len(np.unique(y[warm])),
          "Warm-up span (s)": round(float(span / test["fs"] + test["window_sec"]), 1),
          "Scored windows": int(scored.sum()),
          "full (%)": round(accuracy(norm, full[scored], y[scored]), 2),
          "warmup (%)": round(accuracy(norm, ((f - mu) / (sd + 1e-8))[scored], y[scored]), 2),
          "none (%)": round(accuracy(plain, f_plain[scored], y[scored]), 2),
      })
      r = records[-1]
      line.append(f"K={k}: {r['warmup (%)']:5.1f} (full {r['full (%)']:5.1f}, none {r['none (%)']:5.1f})")
    print(" | ".join(line), flush=True)

  df = pd.DataFrame(records)
  args.out_dir.mkdir(parents=True, exist_ok=True)
  df.to_csv(args.out_dir / "summary.csv", index=False)
  print("=" * 90)
  print("COHORT (mean over held-out subjects, scored on trials after the warm-up; chance = 20%)")
  cohort = df.groupby("Warm-up trials").agg(**{
      "Classes seen": ("Warm-up classes", "mean"), "full %": ("full (%)", "mean"), "warmup %": ("warmup (%)", "mean"),
      "none %": ("none (%)", "mean"), "SD warmup": ("warmup (%)", "std")})
  cohort["warmup - full"] = cohort["warmup %"] - cohort["full %"]
  print(cohort.round(2).to_string())
  print(f"Saved summary.csv to {args.out_dir} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
  main()
