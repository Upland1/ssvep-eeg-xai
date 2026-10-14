"""Leave-one-subject-out (LOSO) generalization benchmark.

For each subject S: train on ALL other subjects, test on S (never seen during
training). Repeat for every subject. This measures how well a model works on a
brand-new user with no calibration data, and is the cross-subject counterpart
of the within-subject numbers from ``benchmark_window_sweep.py`` /
``benchmark_nn_sweep.py`` (same data, same feature code, same networks).

Build the data first (same folder the sweeps use), e.g.:
  python scripts/build_windows.py --window-sec 1 --stride-sec 0.25 --banks fbcca fbcsp

Methods
  fbcca         FBCCA argmax. Needs no training, so its score is the
                calibration-free floor every trained method should beat.
  lda           FBCSP + FBCCA -> StandardScaler -> Shrinkage-LDA, fitted on the
                pooled training subjects.
  lda_subjnorm  Same, but each subject's features are z-scored with that
                subject's OWN mean/std before pooling (uses no labels of the
                test subject, only its unlabeled windows -- report it as
                "unsupervised per-subject normalization").
  ccnn / compact / eegnet
                The networks from benchmark_nn_sweep.py, trained on the pooled
                training subjects with its early stopping (validation = 20% of
                the training TRIALS, never the test subject).

Channels: CSP filters and CNN input layers need the same channels for every
subject, so by default only channels kept by the quality gate in EVERY subject
are used (``--channels common``). ``--channels all`` keeps all 8.

The run is resumable: every finished (method, test subject) fold is saved under
``<out-dir>/folds/`` and skipped on the next run, so a Colab disconnect only
loses the fold in progress. Rerun the same command to continue.

Usage (from the repo root):
  python scripts/benchmarks/benchmark_loso.py --methods fbcca lda lda_subjnorm
  python scripts/benchmarks/benchmark_loso.py --methods ccnn compact eegnet --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_window_sweep import (  # noqa: E402
    PROJECT_ROOT,
    STIM_CONDITIONS,
    SUBJECT_DIR,
    SWEEP_DIR,
    _standardize,
    fbcca_scores,
    prediction_rows,
    summarize,
)
from src.features.ssvep.fbcsp_extraction import MulticlassCSP  # noqa: E402

CLASSICAL = ("fbcca", "lda", "lda_subjnorm")
NETWORKS = ("ccnn", "compact", "eegnet")
METHODS = CLASSICAL + NETWORKS


# --------------------------------------------------------------------------- data

def read_meta(sub_dir: Path) -> dict:
  return json.loads((sub_dir / "windows_meta.json").read_text())


def choose_channels(sub_dirs: dict[str, Path], mode: str) -> list[str]:
  """Channels used for every subject: kept in all subjects, or the full montage."""
  metas = {s: read_meta(p) for s, p in sub_dirs.items()}
  montage = list(next(iter(metas.values()))["channels"])
  for s, m in metas.items():
    if list(m["channels"]) != montage:
      raise ValueError(f"{s} has channels {m['channels']}, expected {montage}; rebuild all subjects together")
  if mode == "all":
    return montage
  kept = [set(m["quality"]["channels_kept"]) for m in metas.values()]
  common = [c for c in montage if all(c in k for k in kept)]
  if len(common) < 2:
    raise ValueError(f"Only {common} survive the quality gate in every subject; use --channels all")
  return common


def load_subject(sub_dir: Path, channels: list[str]) -> dict:
  """Clean stimulus windows of one subject on the given channel list."""
  meta = read_meta(sub_dir)
  missing = [f for f in ("X_bands_fbcca.npy", "X_bands_fbcsp.npy") if not (sub_dir / f).exists()]
  if missing:
    raise FileNotFoundError(f"{sub_dir}: missing {missing}; rebuild with --banks fbcca fbcsp")
  idx = [list(meta["channels"]).index(c) for c in channels]
  y = np.load(sub_dir / "y_labels.npy")
  sel = np.load(sub_dir / "valid_mask.npy").astype(bool) & np.isin(y, STIM_CONDITIONS)
  fs = float(meta["fs"])
  offsets_path = sub_dir / "window_offsets.npy"
  offsets = np.load(offsets_path)[sel] if offsets_path.exists() else np.zeros(sel.sum(), dtype=np.int64)

  def take(name: str) -> np.ndarray:
    out = np.asarray(np.load(sub_dir / name, mmap_mode="r")[sel])
    return out[:, :, idx] if out.ndim == 4 else out[:, idx]

  d = {
      "X": take("X_time_windows.npy"),
      "bands_cca": take("X_bands_fbcca.npy"),
      "bands_csp": take("X_bands_fbcsp.npy"),
      "y": y[sel],
      "trial_ids": np.load(sub_dir / "trial_ids.npy")[sel],
      "offset_sec": offsets / fs,
      "fs": fs,
      "window_sec": float(meta.get("window_sec", meta["n_samples_per_window"] / fs)),
  }
  d["X_cca"] = fbcca_scores(d["bands_cca"], fs)
  return d


# ----------------------------------------------------------------------- methods

def lda_loso(train: list[dict], test: dict, subject_norm: bool) -> tuple[np.ndarray, dict]:
  """Pooled FBCSP+FBCCA Shrinkage-LDA; returns test probabilities and the fitted parts."""
  y_tr = np.concatenate([d["y"] for d in train])
  n_bands = test["bands_csp"].shape[1]
  csps = [MulticlassCSP(n_components=1).fit(np.concatenate([d["bands_csp"][:, b] for d in train]), y_tr)
          for b in range(n_bands)]

  def features(d: dict) -> np.ndarray:
    f = np.hstack([csp.transform(d["bands_csp"][:, b]) for b, csp in enumerate(csps)] + [d["X_cca"]])
    if subject_norm:  # this subject's own statistics; labels are not used
      f = (f - f.mean(axis=0)) / (f.std(axis=0) + 1e-8)
    return f

  X_tr = np.vstack([features(d) for d in train])
  scaler = StandardScaler().fit(X_tr)
  clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(scaler.transform(X_tr), y_tr)
  p = clf.predict_proba(scaler.transform(features(test)))
  probs = np.zeros((len(test["y"]), len(STIM_CONDITIONS)))
  for j, c in enumerate(clf.classes_):
    probs[:, np.flatnonzero(STIM_CONDITIONS == c)[0]] = p[:, j]
  train_acc = float((clf.predict(scaler.transform(X_tr)) == y_tr).mean() * 100)
  return probs, {"train": train_acc, "model": {"csps": csps, "scaler": scaler, "lda": clf,
                                               "subject_norm": subject_norm}}


def network_loso(name: str, train: list[dict], test: dict, args) -> tuple[np.ndarray, dict]:
  """Train one network on the pooled training subjects (early stopping on held-out training trials)."""
  import torch
  from benchmark_nn_sweep import _predict, fit_network

  label = {int(c): i for i, c in enumerate(STIM_CONDITIONS)}
  Xs = np.concatenate([_standardize(d["X"]) for d in train]).astype(np.float32)
  y = np.array([label[int(c)] for d in train for c in d["y"]], dtype=np.int64)
  # Trial ids restart at 0 in every subject; offset them so trials of different
  # subjects never share a group in the validation split.
  groups, offset = [], 0
  for d in train:
    groups.append(d["trial_ids"] + offset)
    offset += int(d["trial_ids"].max()) + 1
  model, info = fit_network(name, Xs, y, np.concatenate(groups), np.arange(len(y)), test["fs"], args, args.seed)
  X_te = torch.tensor(_standardize(test["X"]).astype(np.float32))
  info["model"] = model
  return _predict(model, X_te, torch.device(args.device)), info


# --------------------------------------------------------------------- reporting

def smoothed_acc(probs: np.ndarray, d: dict) -> float:
  """2-tap trial-aware smoothing on consecutive windows of the same trial (ordered by offset)."""
  order = np.lexsort((d["offset_sec"], d["trial_ids"]))
  p, tid, y = probs[order], d["trial_ids"][order], d["y"][order]
  same = np.r_[False, tid[1:] == tid[:-1]]
  smooth = p.copy()
  smooth[same] = (p[same] + p[np.flatnonzero(same) - 1]) / 2
  return float((STIM_CONDITIONS[smooth.argmax(1)] == y).mean() * 100)


def fold_path(out_dir: Path, method: str, sub: str) -> Path:
  return out_dir / "folds" / f"{method}__test-{sub}.npz"


def save_model(method: str, sub: str, model, args) -> None:
  out = args.models_dir / f"loso_{method}_test-{sub}"
  out.parent.mkdir(parents=True, exist_ok=True)
  if method in NETWORKS:
    import torch
    torch.save(model.state_dict(), out.with_suffix(".pt"))
  else:
    import joblib
    joblib.dump(model, out.with_suffix(".joblib"))


# -------------------------------------------------------------------------- main

def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--root", type=Path, default=PROJECT_ROOT / "data" / "processed_continuous",
                  help="Folder holding the w<len>_s<stride> datasets (same as the sweeps)")
  ap.add_argument("--window", type=float, default=1.0, help="Window length in s (default 1.0)")
  ap.add_argument("--stride-sec", type=float, default=0.25)
  ap.add_argument("--subjects", nargs="*", default=None, help="Subset of subjects (default: all found)")
  ap.add_argument("--methods", nargs="+", choices=METHODS, default=list(CLASSICAL))
  ap.add_argument("--channels", choices=("common", "all"), default="common")
  ap.add_argument("--no-augment", dest="augment", action="store_false")
  ap.add_argument("--epochs", type=int, default=150, help="Maximum epochs (early stopping usually ends sooner)")
  ap.add_argument("--patience", type=int, default=25)
  ap.add_argument("--val-frac", type=float, default=0.2)
  ap.add_argument("--batch-size", type=int, default=32)
  ap.add_argument("--lr", type=float, default=1e-3)
  ap.add_argument("--device", default=None, help="cpu / cuda (default: cuda if available)")
  ap.add_argument("--gaze-sec", type=float, default=0.5)
  ap.add_argument("--seed", type=int, default=42)
  ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "reports" / "loso")
  ap.add_argument("--save-models", action="store_true",
                  help="Also save each fold's fitted model under --models-dir (not needed for the benchmark)")
  ap.add_argument("--models-dir", type=Path, default=PROJECT_ROOT / "outputs" / "models" / "loso")
  args = ap.parse_args()

  if any(m in NETWORKS for m in args.methods) and args.device is None:
    import torch
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

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
  if len(sub_dirs) < 2:
    print(f"LOSO needs at least 2 subjects; found {list(sub_dirs)} in {data_dir}")
    return

  channels = choose_channels(sub_dirs, args.channels)
  print("=" * 90)
  print(f"LOSO | {len(sub_dirs)} subjects {list(sub_dirs)} | data {data_dir}")
  print(f"channels ({args.channels}): {channels} | methods {args.methods}"
        + (f" | device {args.device}, augment {'on' if args.augment else 'off'}" if args.device else ""))
  print("=" * 90)

  t0 = time.time()
  data = {}
  for s, p in sub_dirs.items():
    data[s] = load_subject(p, channels)
    d = data[s]
    print(f"  loaded {s}: {len(d['y'])} clean windows, {len(np.unique(d['trial_ids']))} trials, fs={d['fs']:g}")
  if len({d["fs"] for d in data.values()}) > 1:
    raise ValueError("Subjects have different sampling rates; rebuild them with the same settings")

  # ---- run every (method, held-out subject) fold, skipping finished ones
  for method in args.methods:
    for test_sub in data:
      path = fold_path(args.out_dir, method, test_sub)
      if path.exists():
        print(f"  [=] {method:13s} test={test_sub}: already done, skipping")
        continue
      test = data[test_sub]
      train = [data[s] for s in data if s != test_sub]
      ft = time.time()
      if method == "fbcca":
        probs, info = test["X_cca"] / test["X_cca"].sum(axis=1, keepdims=True), {}
      elif method in ("lda", "lda_subjnorm"):
        probs, info = lda_loso(train, test, subject_norm=method == "lda_subjnorm")
      else:
        probs, info = network_loso(method, train, test, args)
      if args.save_models and "model" in info:
        save_model(method, test_sub, info["model"], args)
      path.parent.mkdir(parents=True, exist_ok=True)
      np.savez(path, probs=probs, train=info.get("train", np.nan), val=info.get("val", np.nan),
               epoch=info.get("epoch", np.nan), seconds=time.time() - ft)
      acc = float((STIM_CONDITIONS[probs.argmax(1)] == test["y"]).mean() * 100)
      extra = f" | train {info['train']:.0f}%" if "train" in info else ""
      extra += f", val {info['val']:.0f}%, best epoch {info['epoch']}" if "val" in info else ""
      print(f"  [+] {method:13s} test={test_sub}: {acc:6.2f}%{extra} ({time.time() - ft:.0f} s)", flush=True)

  # ---- summary from every saved fold (including ones from earlier runs)
  records, preds = [], []
  for method in args.methods:
    for sub, d in data.items():
      path = fold_path(args.out_dir, method, sub)
      if not path.exists():
        continue
      f = np.load(path)
      probs = f["probs"]
      rec = summarize(sub, d, method, [probs], args.gaze_sec,
                      None if np.isnan(f["train"]) else [float(f["train"])])
      rec["2-tap Smoothed (%)"] = round(smoothed_acc(probs, d), 2)
      rec["Val Acc (%)"] = round(float(f["val"]), 2) if not np.isnan(f["val"]) else np.nan
      rec["Best Epoch"] = float(f["epoch"])
      rec["Channels"] = len(channels)
      records.append(rec)
      preds.append(prediction_rows(sub, d, method, 0, probs))
  if not records:
    return
  df = pd.DataFrame(records)
  args.out_dir.mkdir(parents=True, exist_ok=True)
  df.to_csv(args.out_dir / "summary.csv", index=False)
  pred_df = pd.concat(preds, ignore_index=True)
  pred_df.to_csv(args.out_dir / "predictions.csv", index=False)
  (pred_df.groupby(["method", "true", "pred"]).size().rename("count").reset_index()
   .to_csv(args.out_dir / "confusion.csv", index=False))

  print("=" * 90)
  print("PER SUBJECT (accuracy % when that subject is the unseen test subject)")
  print(df.pivot(index="Subject", columns="Method", values="Acc (%)")[
      [m for m in args.methods if m in df["Method"].unique()]].round(2).to_string())
  print("-" * 90)
  print("COHORT (mean +/- SD over held-out subjects; chance = 20%)")
  cohort = df.groupby("Method").agg(**{
      "Acc %": ("Acc (%)", "mean"), "SD": ("Acc (%)", "std"),
      "Smoothed %": ("2-tap Smoothed (%)", "mean"), "BalAcc %": ("BalAcc (%)", "mean"),
      "ITR": ("ITR (bits/min)", "mean"), "Train %": ("Train Acc (%)", "mean")})
  print(cohort.reindex([m for m in args.methods if m in cohort.index]).round(2).to_string())
  print(f"Saved summary.csv, predictions.csv, confusion.csv to {args.out_dir} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
  main()
