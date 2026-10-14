"""Simulate early decisions from saved window predictions.

Pass ``predictions.csv`` files from the sweep benchmarks. No model is retrained.
For each trial, windows are processed in time order:

  - ignore windows that start before ``--skip-sec``;
  - average the probabilities seen so far;
  - decide when the top probability reaches a threshold;
  - force the final decision if no threshold is reached.

Reports accuracy, decision time, forced decisions, and ITR. The best threshold
is selected per method and window across the cohort. FBCCA is skipped because
its scores are not probabilities.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

N_CLASSES = 5
PROB_COLS = [f"p_{c}" for c in (101, 102, 103, 104, 105)]
CODES = np.array([101, 102, 103, 104, 105])


def itr_bits_per_min(p: float, n: int, t_sec: float) -> float:
  if p <= 1.0 / n or t_sec <= 0:
    return 0.0
  bits = np.log2(n) + p * np.log2(p) + ((1 - p) * np.log2((1 - p) / (n - 1)) if p < 1 else 0.0)
  return float(bits * 60.0 / t_sec)


def simulate(pred: pd.DataFrame, thresholds: list[float], skip_sec: float) -> pd.DataFrame:
  """Return one decision row per trial and threshold."""
  rows = []
  keys = ["subject", "method", "window_sec", "run", "trial_id"]
  for key, g in pred[pred["offset_sec"] >= skip_sec - 1e-9].groupby(keys, sort=False):
    g = g.sort_values("offset_sec")
    probs = g[PROB_COLS].to_numpy()
    avg = np.cumsum(probs, axis=0) / np.arange(1, len(g) + 1)[:, None]
    top = avg.max(axis=1)
    t_avail = g["offset_sec"].to_numpy() + key[2]
    true = int(g["true"].iloc[0])
    for th in thresholds:
      hit = np.flatnonzero(top >= th)
      k, forced = (hit[0], False) if len(hit) else (len(g) - 1, True)
      rows.append((*key, th, true, int(CODES[avg[k].argmax()]), float(t_avail[k]), forced))
  return pd.DataFrame(rows, columns=keys + ["threshold", "true", "pred", "time_sec", "forced"])


def summarize(dec: pd.DataFrame, gaze_sec: float) -> pd.DataFrame:
  """Summarize accuracy, timing, forced decisions, and ITR per subject."""
  dec = dec.assign(correct=dec["true"] == dec["pred"])
  per_run = dec.groupby(["method", "window_sec", "threshold", "subject", "run"]).agg(
      acc=("correct", "mean"), time=("time_sec", "mean"), forced=("forced", "mean"), trials=("correct", "size"))
  s = per_run.groupby(["method", "window_sec", "threshold", "subject"]).mean().reset_index()
  s["itr"] = [itr_bits_per_min(a, N_CLASSES, t + gaze_sec) for a, t in zip(s["acc"], s["time"])]
  s[["acc", "forced"]] *= 100
  return s


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("predictions", type=Path, nargs="+", help="predictions.csv files from the sweep benchmarks")
  ap.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
  ap.add_argument("--skip-sec", type=float, default=0.5, help="Ignore windows starting earlier than this (default 0.5)")
  ap.add_argument("--gaze-sec", type=float, default=0.5)
  ap.add_argument("--out-dir", type=Path, default=Path("reports") / "dynamic_stopping")
  args = ap.parse_args()

  pred = pd.concat([pd.read_csv(p) for p in args.predictions], ignore_index=True)
  pred = pred[pred["method"] != "fbcca"]
  dec = simulate(pred, args.thresholds, args.skip_sec)
  per_sub = summarize(dec, args.gaze_sec)
  cohort = (per_sub.groupby(["method", "window_sec", "threshold"])
            [["acc", "time", "forced", "itr"]].mean().reset_index())

  args.out_dir.mkdir(parents=True, exist_ok=True)
  dec.to_csv(args.out_dir / "decisions.csv", index=False)
  per_sub.round(3).to_csv(args.out_dir / "per_subject.csv", index=False)
  cohort.round(3).to_csv(args.out_dir / "cohort.csv", index=False)

  # Select the highest-ITR threshold for each method and window.
  best = cohort.loc[cohort.groupby(["method", "window_sec"])["itr"].idxmax()]
  pd.set_option("display.width", 200)
  print(f"DYNAMIC STOPPING | skip first {args.skip_sec:g} s | ITR time = decision time + {args.gaze_sec:g} s")
  print("\nBest threshold per method and window (cohort mean):")
  print(best.rename(columns={"acc": "Acc %", "time": "Time s", "forced": "Forced %", "itr": "ITR"})
        .round(2).to_string(index=False))

  top = best.loc[best.groupby("method")["itr"].idxmax()]
  print("\nPer subject at each method's best setting:")
  for _, r in top.iterrows():
    sub = per_sub[(per_sub["method"] == r["method"]) & (per_sub["window_sec"] == r["window_sec"])
                  & (per_sub["threshold"] == r["threshold"])]
    print(f"  {r['method']} | {r['window_sec']:g} s windows, threshold {r['threshold']:g}")
    for _, s in sub.iterrows():
      print(f"    {s['subject']}: acc {s['acc']:5.1f}% | decision at {s['time']:.2f} s | "
            f"forced {s['forced']:4.1f}% | ITR {s['itr']:5.1f}")
  print(f"\nSaved decisions.csv, per_subject.csv, cohort.csv to {args.out_dir}")


if __name__ == "__main__":
  main()