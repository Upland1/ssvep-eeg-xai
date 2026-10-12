"""Compare neural networks on window-sweep data.

Build the data first with ``scripts/build_windows.py``. The sweep compares
CCNN, Compact-CNN, and EEGNet with trial-grouped 5-fold CV. Each fold uses
20% of its training trials for validation and keeps the best validation loss.
Augmentation is on by default; use ``--no-augment`` to disable it.

Start with ``--windows 1`` before running the full sweep.
"""
from __future__ import annotations

import argparse
import copy
import importlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_window_sweep import (  # noqa: E402
    PROJECT_ROOT,
    STIM_CONDITIONS,
    SUBJECT_DIR,
    SWEEP_DIR,
    _standardize,
    load_sweep_subject,
    prediction_rows,
    summarize,
)

MODELS = ("ccnn", "compact", "eegnet")


def _import(names: list[str], attr: str):
  """Load an attribute from the first available module."""
  for name in names:
    try:
      return getattr(importlib.import_module(name), attr)
    except ModuleNotFoundError:
      continue
  raise ImportError(f"Could not find {attr} in any of {names}")


def build_model(name: str, n_channels: int, n_samples: int, fs: float, n_classes: int = len(STIM_CONDITIONS)) -> nn.Module:
  if name == "ccnn":
    CCNN = _import(["src.models.cnn.ccnn"], "CCNN")
    return CCNN(n_classes=n_classes, n_channels=n_channels, n_samples=n_samples, fs=fs)
  EEGNetSSVEP = _import(["src.models.cnn.eegnet"], "EEGNetSSVEP")
  kernel = int(fs // 2)
  if name == "compact":
    return EEGNetSSVEP(n_classes=n_classes, n_channels=n_channels, n_samples=n_samples,
                       F1=96, D=1, F2=96, kernel_length=kernel, dropout_rate=0.5)
  return EEGNetSSVEP(n_classes=n_classes, n_channels=n_channels, n_samples=n_samples,
                     kernel_length=kernel)


def _predict(model: nn.Module, X: torch.Tensor, device) -> np.ndarray:
  """Return batched softmax predictions to limit GPU memory use."""
  model.eval()
  with torch.no_grad():
    out = [torch.softmax(model(X[i:i + 64].to(device)), dim=1).cpu() for i in range(0, len(X), 64)]
  return torch.cat(out).numpy()


def train_fold(name: str, d: dict, Xs: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray,
               args, seed: int) -> tuple[np.ndarray, dict]:
  """Train one fold and return its test-fold probabilities."""
  model, info = fit_network(name, Xs, y, d["trial_ids"], tr, d["fs"], args, seed)
  return _predict(model, torch.tensor(Xs[te]), torch.device(args.device)), info


def fit_network(name: str, Xs: np.ndarray, y: np.ndarray, trial_ids: np.ndarray, tr: np.ndarray, fs: float,
                args, seed: int, n_classes: int = len(STIM_CONDITIONS)) -> tuple[nn.Module, dict]:
  """Train one network with trial-grouped validation and early stopping."""
  torch.manual_seed(seed)
  np.random.seed(seed)
  device = torch.device(args.device)

  inner = StratifiedGroupKFold(n_splits=max(2, int(round(1 / args.val_frac))), shuffle=True, random_state=seed)
  fit_rel, val_rel = next(inner.split(tr, y[tr], groups=trial_ids[tr]))
  fit, val = tr[fit_rel], tr[val_rel]

  if args.augment:
    SSVEPAugmentedDataset = _import(["src.preprocessing.dataset.augmentation"], "SSVEPAugmentedDataset")
    train_ds = SSVEPAugmentedDataset(Xs[fit], y[fit], seed=seed)
  else:
    train_ds = TensorDataset(torch.tensor(Xs[fit]), torch.tensor(y[fit]))
  loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                      generator=torch.Generator().manual_seed(seed))
  X_all = torch.tensor(Xs)
  y_val = torch.tensor(y[val])

  model = build_model(name, Xs.shape[1], Xs.shape[2], fs, n_classes).to(device)
  opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
  sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
  loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
  eval_loss = nn.CrossEntropyLoss()

  best = {"loss": np.inf, "epoch": 0, "state": None}
  for epoch in range(1, args.epochs + 1):
    model.train()
    for xb, yb in loader:
      opt.zero_grad()
      loss_fn(model(xb.to(device)), yb.to(device)).backward()
      opt.step()
    sched.step()
    p_val = _predict(model, X_all[val], device)
    v_loss = float(eval_loss(torch.log(torch.tensor(p_val) + 1e-8), y_val))
    if v_loss < best["loss"]:
      best = {"loss": v_loss, "epoch": epoch, "state": copy.deepcopy(model.state_dict()),
              "val_acc": float((p_val.argmax(1) == y[val]).mean() * 100)}
    elif epoch - best["epoch"] >= args.patience:
      break

  model.load_state_dict(best["state"])
  train_acc = float((_predict(model, X_all[fit], device).argmax(1) == y[fit]).mean() * 100)
  return model, {"train": train_acc, "val": best["val_acc"], "epoch": best["epoch"]}


def run_subject(sub: str, sub_dir: Path, args) -> tuple[list[dict], list[pd.DataFrame]]:
  d = load_sweep_subject(sub_dir)
  Xs = _standardize(d["X"]).astype(np.float32)
  label = {int(c): i for i, c in enumerate(STIM_CONDITIONS)}
  y = np.array([label[int(c)] for c in d["y"]], dtype=np.int64)
  records, preds = [], []
  line = [f"  {sub} | {d['window_sec']:g} s | {len(y)} windows"]

  for name in args.models:
    tag = name + ("+aug" if args.augment else "")
    runs, diag = [], []
    for r in range(args.repeats):
      seed = args.seed + r
      probs = np.zeros((len(y), len(STIM_CONDITIONS)))
      cv = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=seed)
      for tr, te in cv.split(Xs, y, groups=d["trial_ids"]):
        probs[te], info = train_fold(name, d, Xs, y, tr, te, args, seed)
        diag.append(info)
      runs.append(probs)
      preds.append(prediction_rows(sub, d, tag, r, probs))
    rec = summarize(sub, d, tag, runs, args.gaze_sec, [x["train"] for x in diag])
    rec["Val Acc (%)"] = round(float(np.mean([x["val"] for x in diag])), 2)
    rec["Best Epoch"] = round(float(np.mean([x["epoch"] for x in diag])), 1)
    records.append(rec)
    line.append(f"{tag} {rec['Acc (%)']:.1f}% (train {rec['Train Acc (%)']:.0f}%, "
                f"val {rec['Val Acc (%)']:.0f}%, ep {rec['Best Epoch']:.0f})")
  print(" | ".join(line), flush=True)
  return records, preds


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--root", type=Path, default=PROJECT_ROOT / "data" / "processed_continuous")
  ap.add_argument("--stride-sec", type=float, default=0.25)
  ap.add_argument("--windows", type=float, nargs="+", default=[1.0], help="Window lengths (default: 1.0)")
  ap.add_argument("--subjects", nargs="*", default=None)
  ap.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
  ap.add_argument("--no-augment", dest="augment", action="store_false")
  ap.add_argument("--folds", type=int, default=5)
  ap.add_argument("--repeats", type=int, default=1)
  ap.add_argument("--epochs", type=int, default=150, help="Maximum epochs (early stopping usually ends sooner)")
  ap.add_argument("--patience", type=int, default=25)
  ap.add_argument("--val-frac", type=float, default=0.2)
  ap.add_argument("--batch-size", type=int, default=32)
  ap.add_argument("--lr", type=float, default=1e-3)
  ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  ap.add_argument("--gaze-sec", type=float, default=0.5)
  ap.add_argument("--seed", type=int, default=42)
  ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "reports" / "nn_sweep")
  args = ap.parse_args()

  found = {}
  for p in sorted(args.root.iterdir()) if args.root.exists() else []:
    m = SWEEP_DIR.match(p.name)
    if p.is_dir() and m and np.isclose(float(m.group(2)), args.stride_sec):
      found[float(m.group(1))] = p
  lengths = [w for w in sorted(args.windows) if w in found]
  if not lengths:
    print(f"None of {args.windows} found as w<len>_s{args.stride_sec:g} under {args.root}")
    return

  print("=" * 100)
  print(f"NN SWEEP | lengths {lengths} s | models {args.models} | augment {'on' if args.augment else 'off'} | "
        f"max {args.epochs} epochs, patience {args.patience} | device {args.device}")
  print("=" * 100)
  t0 = time.time()
  records, preds = [], []
  for w in lengths:
    subs = sorted(p.name for p in found[w].iterdir() if p.is_dir() and SUBJECT_DIR.match(p.name))
    for sub in subs if args.subjects is None else [s for s in subs if s in args.subjects]:
      rec, pr = run_subject(sub, found[w] / sub, args)
      records += rec
      preds += pr

  df = pd.DataFrame(records)
  args.out_dir.mkdir(parents=True, exist_ok=True)
  df.to_csv(args.out_dir / "summary.csv", index=False)
  pred_df = pd.concat(preds, ignore_index=True)
  pred_df.to_csv(args.out_dir / "predictions.csv", index=False)
  (pred_df.groupby(["window_sec", "method", "true", "pred"]).size().rename("count").reset_index()
   .to_csv(args.out_dir / "confusion.csv", index=False))

  print("=" * 100)
  print("COHORT (mean over subjects)")
  cohort = df.groupby(["Window (s)", "Method"]).agg(
      **{"Acc %": ("Acc (%)", "mean"), "ITR": ("ITR (bits/min)", "mean"), "Train %": ("Train Acc (%)", "mean"),
         "Val %": ("Val Acc (%)", "mean"), "Epoch": ("Best Epoch", "mean")})
  print(cohort.round(1).to_string())
  print(f"Saved summary.csv, predictions.csv, confusion.csv to {args.out_dir} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
  main()