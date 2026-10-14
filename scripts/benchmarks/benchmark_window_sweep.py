"""Compare accuracy and ITR across analysis-window lengths.

Build sweep data first, for example:
  python scripts/build_windows.py --window-sec 0.5 1 1.5 2 3 4 \
      --stride-sec 0.25 --banks fbcca fbcsp

The sweep tests FBCCA, FBCSP+FBCCA LDA, and Compact-CNN without smoothing.
Trained methods use trial-grouped cross-validation so windows from one trial
stay in the same fold. Predictions are saved for later detection analysis.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


def _find_root(start: Path) -> Path:
  for cand in [start, *start.parents]:
    if (cand / "configs" / "pipeline_config.yaml").exists():
      return cand
  return start.parents[1]


PROJECT_ROOT = _find_root(Path(__file__).resolve())
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.ssvep.fbcsp_extraction import MulticlassCSP  # noqa: E402

STIM_CONDITIONS = np.array([101, 102, 103, 104, 105])
STIM_FREQS = {101: 24.0, 102: 20.0, 103: 15.0, 104: 10.9091, 105: 8.5714}
SWEEP_DIR = re.compile(r"^w(\d+(?:\.\d+)?)_s(\d+(?:\.\d+)?)$")
SUBJECT_DIR = re.compile(r"^S\d+$")
METHODS = ("fbcca", "lda", "cnn")


def load_sweep_subject(sub_dir: Path) -> dict:
  """Load one subject's stimulus windows with the shared quality decision applied."""
  meta = json.loads((sub_dir / "windows_meta.json").read_text())
  missing = [f for f in ("X_bands_fbcca.npy", "X_bands_fbcsp.npy") if not (sub_dir / f).exists()]
  if missing:
    raise FileNotFoundError(f"{sub_dir}: missing {missing}; rebuild with --banks fbcca fbcsp")
  y = np.load(sub_dir / "y_labels.npy")
  valid = np.load(sub_dir / "valid_mask.npy").astype(bool)
  sel = valid & np.isin(y, STIM_CONDITIONS)
  kept = meta["quality"]["channels_kept_idx"]
  offsets_path = sub_dir / "window_offsets.npy"
  offsets = np.load(offsets_path)[sel] if offsets_path.exists() else np.zeros(sel.sum(), dtype=np.int64)
  fs = float(meta["fs"])

  def take(name: str) -> np.ndarray:
    arr = np.load(sub_dir / name, mmap_mode="r")
    out = np.asarray(arr[sel])
    return out[:, :, kept] if out.ndim == 4 else out[:, kept]

  return {
      "X": take("X_time_windows.npy"),
      "bands_cca": take("X_bands_fbcca.npy"),
      "bands_csp": take("X_bands_fbcsp.npy"),
      "y": y[sel],
      "trial_ids": np.load(sub_dir / "trial_ids.npy")[sel],
      "offset_sec": offsets / fs,
      "fs": fs,
      "window_sec": float(meta.get("window_sec", meta["n_samples_per_window"] / fs)),
      "channels_kept": meta["quality"]["channels_kept"],
  }


def reference_bases(fs: float, n_samples: int, n_harmonics: int = 3) -> list[np.ndarray]:
  """Build orthonormal sine/cosine references for each target."""
  t = np.arange(n_samples) / fs
  bases = []
  for code in STIM_CONDITIONS:
    f = STIM_FREQS[int(code)]
    ref = np.column_stack([fn(2 * np.pi * h * f * t) for h in range(1, n_harmonics + 1) for fn in (np.sin, np.cos)])
    bases.append(np.linalg.qr(ref - ref.mean(axis=0))[0])
  return bases


def fbcca_scores(bands: np.ndarray, fs: float, n_harmonics: int = 3) -> np.ndarray:
  """Return FBCCA scores for each window and target."""
  n, n_bands, _, n_samples = bands.shape
  weights = np.array([k ** -1.25 + 0.25 for k in range(1, n_bands + 1)])
  bases = reference_bases(fs, n_samples, n_harmonics)
  scores = np.zeros((n, len(bases)))
  for b in range(n_bands):
    x = np.transpose(bands[:, b], (0, 2, 1))
    qx = np.linalg.qr(x - x.mean(axis=1, keepdims=True))[0]
    for k, qy in enumerate(bases):
      r = np.linalg.svd(np.transpose(qx, (0, 2, 1)) @ qy, compute_uv=False)[:, 0]
      scores[:, k] += weights[b] * np.clip(r, 0.0, 1.0) ** 2
  return scores


def lda_fold(d: dict, X_cca: np.ndarray, tr: np.ndarray, te: np.ndarray) -> np.ndarray:
  """Fit the LDA pipeline and return test probabilities."""
  y_tr = d["y"][tr]
  f_tr, f_te = [], []
  for b in range(d["bands_csp"].shape[1]):
    csp = MulticlassCSP(n_components=1).fit(d["bands_csp"][tr, b], y_tr)
    f_tr.append(csp.transform(d["bands_csp"][tr, b]))
    f_te.append(csp.transform(d["bands_csp"][te, b]))
  X_tr = np.hstack(f_tr + [X_cca[tr]])
  X_te = np.hstack(f_te + [X_cca[te]])
  scaler = StandardScaler().fit(X_tr)
  clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(scaler.transform(X_tr), y_tr)
  probs = np.zeros((len(te), len(STIM_CONDITIONS)))
  p = clf.predict_proba(scaler.transform(X_te))
  for j, c in enumerate(clf.classes_):
    probs[:, np.flatnonzero(STIM_CONDITIONS == c)[0]] = p[:, j]
  return probs


def _standardize(x: np.ndarray) -> np.ndarray:
  """Standardize each window and channel over time."""
  return (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-8)


def cnn_fold(d: dict, tr: np.ndarray, te: np.ndarray, args, seed: int) -> tuple[np.ndarray, float]:
  """Train Compact-CNN and return test probabilities and train accuracy."""
  import torch
  from torch import nn
  from torch.utils.data import DataLoader, TensorDataset

  from src.models.cnn.compact_cnn import CompactCNN

  torch.manual_seed(seed)
  np.random.seed(seed)
  device = torch.device(args.device)
  label = {int(c): i for i, c in enumerate(STIM_CONDITIONS)}
  X = torch.tensor(_standardize(d["X"]), dtype=torch.float32)
  y = torch.tensor([label[int(c)] for c in d["y"]], dtype=torch.long)

  model = CompactCNN(n_classes=len(STIM_CONDITIONS), n_channels=X.shape[1], n_samples=X.shape[2],
                     fs=d["fs"]).to(device)
  opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
  sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
  loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
  gen = torch.Generator().manual_seed(seed)
  loader = DataLoader(TensorDataset(X[tr], y[tr]), batch_size=args.batch_size, shuffle=True, generator=gen)

  for _ in range(args.epochs):
    model.train()
    for xb, yb in loader:
      opt.zero_grad()
      loss_fn(model(xb.to(device)), yb.to(device)).backward()
      opt.step()
    sched.step()

  model.eval()
  with torch.no_grad():
    def predict(idx):
      out = [torch.softmax(model(X[idx][i:i + 256].to(device)), dim=1).cpu() for i in range(0, len(idx), 256)]
      return torch.cat(out).numpy()
    train_acc = float((predict(tr).argmax(1) == y[tr].numpy()).mean() * 100)
    return predict(te), train_acc


def itr_bits_per_min(p: float, n: int, t_sec: float) -> float:
  """Return Wolpaw ITR in bits per minute."""
  if p <= 1.0 / n:
    return 0.0
  bits = np.log2(n) + p * np.log2(p) + ((1 - p) * np.log2((1 - p) / (n - 1)) if p < 1 else 0.0)
  return float(bits * 60.0 / t_sec)


def summarize(sub: str, d: dict, method: str, runs: list[np.ndarray], gaze_sec: float,
              train_accs: list[float] | None = None) -> dict:
  y = d["y"]
  accs = [accuracy_score(y, STIM_CONDITIONS[p.argmax(1)]) for p in runs]
  bals = [balanced_accuracy_score(y, STIM_CONDITIONS[p.argmax(1)]) for p in runs]
  acc = float(np.mean(accs))
  return {
      "Subject": sub,
      "Window (s)": d["window_sec"],
      "Method": method,
      "Windows": len(y),
      "Trials": len(np.unique(d["trial_ids"])),
      "Acc (%)": round(acc * 100, 2),
      "Acc SD": round(float(np.std(accs)) * 100, 2) if len(runs) > 1 else np.nan,
      "BalAcc (%)": round(float(np.mean(bals)) * 100, 2),
      "ITR (bits/min)": round(itr_bits_per_min(acc, len(STIM_CONDITIONS), d["window_sec"] + gaze_sec), 2),
      "Train Acc (%)": round(float(np.mean(train_accs)), 2) if train_accs else np.nan,
      "Runs": len(runs),
  }


def prediction_rows(sub: str, d: dict, method: str, run: int, probs: np.ndarray) -> pd.DataFrame:
  df = pd.DataFrame({
      "subject": sub, "window_sec": d["window_sec"], "method": method, "run": run,
      "trial_id": d["trial_ids"], "offset_sec": np.round(d["offset_sec"], 4),
      "true": d["y"], "pred": STIM_CONDITIONS[probs.argmax(1)],
  })
  for j, c in enumerate(STIM_CONDITIONS):
    df[f"p_{c}"] = np.round(probs[:, j], 5)
  return df


def run_subject(sub: str, sub_dir: Path, args) -> tuple[list[dict], list[pd.DataFrame]]:
  d = load_sweep_subject(sub_dir)
  X_cca = fbcca_scores(d["bands_cca"], d["fs"])
  records, preds = [], []
  line = [f"  {sub} | {d['window_sec']:g} s | {len(d['y'])} windows, {len(np.unique(d['trial_ids']))} trials"]

  if "fbcca" in args.methods:
    p = X_cca / X_cca.sum(axis=1, keepdims=True)
    records.append(summarize(sub, d, "fbcca", [p], args.gaze_sec))
    preds.append(prediction_rows(sub, d, "fbcca", 0, p))

  for method, repeats in (("lda", args.cv_repeats), ("cnn", args.cnn_repeats)):
    if method not in args.methods:
      continue
    runs, train_accs = [], []
    for r in range(repeats):
      seed = args.seed + r
      probs = np.zeros((len(d["y"]), len(STIM_CONDITIONS)))
      cv = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=seed)
      fold_train = []
      for tr, te in cv.split(d["X"], d["y"], groups=d["trial_ids"]):
        if method == "lda":
          probs[te] = lda_fold(d, X_cca, tr, te)
        else:
          probs[te], tacc = cnn_fold(d, tr, te, args, seed)
          fold_train.append(tacc)
      runs.append(probs)
      if fold_train:
        train_accs.append(float(np.mean(fold_train)))
      preds.append(prediction_rows(sub, d, method, r, probs))
    records.append(summarize(sub, d, method, runs, args.gaze_sec, train_accs or None))

  for rec in records:
    line.append(f"{rec['Method']} {rec['Acc (%)']:.1f}% / {rec['ITR (bits/min)']:.1f} b/m")
  print(" | ".join(line), flush=True)
  return records, preds


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--root", type=Path, default=PROJECT_ROOT / "data" / "processed_continuous",
                  help="Folder holding the w<len>_s<stride> datasets")
  ap.add_argument("--stride-sec", type=float, default=0.25, help="Stride of the datasets to use (default 0.25)")
  ap.add_argument("--windows", type=float, nargs="*", default=None, help="Window lengths to run (default: all found)")
  ap.add_argument("--subjects", nargs="*", default=None)
  ap.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
  ap.add_argument("--folds", type=int, default=5)
  ap.add_argument("--cv-repeats", type=int, default=5, help="CV repeats for LDA (cheap; default 5)")
  ap.add_argument("--cnn-repeats", type=int, default=1, help="CV repeats for the CNN (default 1)")
  ap.add_argument("--epochs", type=int, default=150)
  ap.add_argument("--batch-size", type=int, default=32)
  ap.add_argument("--lr", type=float, default=1e-3)
  ap.add_argument("--device", default=None, help="cpu / cuda (default: cuda if available)")
  ap.add_argument("--gaze-sec", type=float, default=0.5,
                  help="Gaze-shift time added to the window length in the ITR (default 0.5 s)")
  ap.add_argument("--seed", type=int, default=42)
  ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "reports" / "window_sweep")
  args = ap.parse_args()

  if "cnn" in args.methods and args.device is None:
    import torch
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

  found = {}
  for p in sorted(args.root.iterdir()) if args.root.exists() else []:
    m = SWEEP_DIR.match(p.name)
    if p.is_dir() and m and np.isclose(float(m.group(2)), args.stride_sec):
      found[float(m.group(1))] = p
  lengths = sorted(found) if args.windows is None else [w for w in sorted(args.windows) if w in found]
  if not lengths:
    print(f"No w<len>_s{args.stride_sec:g} folders found under {args.root}")
    return

  print("=" * 90)
  print(f"WINDOW SWEEP | lengths {lengths} s | stride {args.stride_sec:g} s | methods {args.methods}")
  print(f"LDA repeats {args.cv_repeats} | CNN repeats {args.cnn_repeats}, {args.epochs} epochs"
        + (f" on {args.device}" if "cnn" in args.methods else "") + f" | ITR time = window + {args.gaze_sec:g} s")
  print("=" * 90)

  records, preds = [], []
  t0 = time.time()
  for w in lengths:
    subs = sorted(p.name for p in found[w].iterdir() if p.is_dir() and SUBJECT_DIR.match(p.name))
    for sub in subs if args.subjects is None else [s for s in subs if s in args.subjects]:
      try:
        rec, pr = run_subject(sub, found[w] / sub, args)
      except (FileNotFoundError, ValueError) as exc:
        print(f"  [-] {sub} {w:g} s: skipped ({exc})")
        continue
      records += rec
      preds += pr

  df = pd.DataFrame(records)
  args.out_dir.mkdir(parents=True, exist_ok=True)
  df.to_csv(args.out_dir / "summary.csv", index=False)
  pred_df = pd.concat(preds, ignore_index=True)
  pred_df.to_csv(args.out_dir / "predictions.csv", index=False)

  # Save pooled confusion counts.
  conf = (pred_df.groupby(["window_sec", "method", "true", "pred"]).size()
          .rename("count").reset_index())
  conf.to_csv(args.out_dir / "confusion.csv", index=False)

  cohort = df.groupby(["Window (s)", "Method"]).agg(
      acc=("Acc (%)", "mean"), acc_sd=("Acc (%)", "std"), itr=("ITR (bits/min)", "mean"),
      train=("Train Acc (%)", "mean")).reset_index()
  print("=" * 90)
  print("COHORT (mean over subjects; SD = between subjects)")
  table = cohort.pivot(index="Window (s)", columns="Method", values=["acc", "itr"])
  table.columns = [f"{m} {'Acc %' if k == 'acc' else 'ITR'}" for k, m in table.columns]
  print(table.round(2).to_string())
  if "cnn" in args.methods:
    tr = cohort[cohort["Method"] == "cnn"].set_index("Window (s)")["train"]
    print("CNN train accuracy by window: " + ", ".join(f"{w:g}s {v:.1f}%" for w, v in tr.items()))
  print(f"Saved summary.csv, predictions.csv, confusion.csv to {args.out_dir} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
  main()