"""Aggregated NN benchmark: augmentation ON vs OFF, across ALL discovered subjects.

Runs the SAME architecture, quality pipeline, CV scheme and training
procedure twice per subject -- once with `--augment`, once without -- so
the augmentation effect can be judged on the cohort instead of anecdotally
on a single subject. Mirrors benchmark_n_subjects.py's auto-discovery
so it scales the same way (no hard-coded subject list).
"""

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

from src.preprocessing.quality_check import apply_artifact_quality_pipeline
from src.models.eegnet import EEGNetSSVEP
from src.models.compact_cnn import CompactCNN
from src.models.shallow_conv_net import ShallowConvNet
from src.preprocessing.augmentation import SSVEPAugmentedDataset
from src.preprocessing.cv_utils import build_trial_ids

FULL_MONTAGE = ["PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"]
SUBJECT_DIR_PATTERN = re.compile(r"^S\d+$")


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


def build_model(name: str, n_classes: int, n_channels: int, n_samples: int):
  if name == "eegnet":
    return EEGNetSSVEP(n_classes=n_classes, n_channels=n_channels, n_samples=n_samples, dropout_rate=0.25)
  if name == "compact_cnn":
    return CompactCNN(
        n_classes=n_classes, n_channels=n_channels, n_samples=n_samples,
        fs=250.0, lowest_freq_hz=8.5714, dropout_rate=0.5,
    )
  if name == "shallow_conv_net":
    return ShallowConvNet(n_classes=n_classes, n_channels=n_channels, n_samples=n_samples, dropout_rate=0.5)
  raise ValueError(f"Unknown model: {name}")


def run_cv(windows_clean, y_mapped, trial_ids, classes, model_name: str, args, augment: bool) -> tuple[float, float, float]:
  """Run one full 5-fold GROUP-stratified CV pass; returns (acc_1s, acc_smooth, mean_train_acc).

  Uses StratifiedGroupKFold with `trial_ids` as groups so every sub-window
  belonging to the same trial lands in the same fold -- consecutive
  sub-windows of a trial share near-identical artifacts/electrode state,
  so a plain StratifiedKFold split lets that leak into "accuracy" instead
  of measuring generalization to a genuinely new trial.

  mean_train_acc is the final-epoch TRAINING-set accuracy, averaged across
  folds -- a quick way to tell "not learning" (train acc also low) apart
  from "overfitting / high CV variance" (train acc high, test acc low).
  """
  sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
  oof_probs = np.zeros((len(y_mapped), len(classes)))
  fold_train_accs = []

  for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(windows_clean, y_mapped, groups=trial_ids), 1):
    X_tr, y_tr = windows_clean[train_idx], y_mapped[train_idx]
    X_te, y_te = windows_clean[test_idx], y_mapped[test_idx]

    mean_tr = np.mean(X_tr, axis=-1, keepdims=True)
    std_tr = np.std(X_tr, axis=-1, keepdims=True) + 1e-8
    X_tr_norm = (X_tr - mean_tr) / std_tr

    mean_te = np.mean(X_te, axis=-1, keepdims=True)
    std_te = np.std(X_te, axis=-1, keepdims=True) + 1e-8
    X_te_norm = (X_te - mean_te) / std_te

    t_X_te = torch.tensor(X_te_norm, dtype=torch.float32).unsqueeze(1).to(args.device)
    # Un-augmented view of the training fold, kept ONLY for measuring
    # final training-set accuracy at the end -- never used to compute
    # gradients, so it doesn't change what the model is trained on.
    t_X_tr_eval = torch.tensor(X_tr_norm, dtype=torch.float32).unsqueeze(1).to(args.device)
    t_y_tr_eval = torch.tensor(y_tr, dtype=torch.long)

    if augment:
      train_ds = SSVEPAugmentedDataset(X_tr_norm, y_tr, seed=42 + fold_idx)
    else:
      t_X_tr = torch.tensor(X_tr_norm, dtype=torch.float32).unsqueeze(1)
      t_y_tr = torch.tensor(y_tr, dtype=torch.long)
      train_ds = TensorDataset(t_X_tr, t_y_tr)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    # Seed BEFORE model construction so weight init and DataLoader
    # shuffling are deterministic across repeated runs -- without this,
    # two runs of the same command can differ by several points purely
    # from random init, which makes ON-vs-OFF comparisons unreliable.
    torch.manual_seed(42 + fold_idx)

    model = build_model(model_name, len(classes), windows_clean.shape[1], windows_clean.shape[2]).to(args.device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    model.train()
    for _ in range(args.epochs):
      for bx, by in train_loader:
        bx, by = bx.to(args.device), by.to(args.device)
        optimizer.zero_grad()
        loss = criterion(model(bx), by)
        loss.backward()
        optimizer.step()
      scheduler.step()

    model.eval()
    with torch.no_grad():
      fold_probs = torch.softmax(model(t_X_te), dim=-1).cpu().numpy()
      train_preds = torch.argmax(model(t_X_tr_eval), dim=-1).cpu().numpy()
    oof_probs[test_idx] = fold_probs
    fold_train_accs.append(accuracy_score(t_y_tr_eval.numpy(), train_preds) * 100.0)

  preds_1s = np.argmax(oof_probs, axis=1)
  acc_1s = accuracy_score(y_mapped, preds_1s) * 100.0

  preds_smooth = np.zeros_like(y_mapped)
  for i in range(len(y_mapped)):
    if i % 5 == 0:
      avg_p = oof_probs[i]
    else:
      avg_p = np.mean(oof_probs[i - 1:i + 1], axis=0)
    preds_smooth[i] = np.argmax(avg_p)
  acc_smooth = accuracy_score(y_mapped, preds_smooth) * 100.0

  return acc_1s, acc_smooth, float(np.mean(fold_train_accs))


def process_subject(sub: str, data_root: Path, model_name: str, args) -> dict | None:
  sub_dir = data_root / sub
  if not (sub_dir / "X_time_windows.npy").exists():
    print(f"[-] Skipping {sub}: X_time_windows.npy not found.")
    return None

  windows_raw = np.load(sub_dir / "X_time_windows.npy")
  y_raw = np.load(sub_dir / "y_labels.npy")
  stim_mask = np.isin(y_raw, [101, 102, 103, 104, 105])
  y_stim = y_raw[stim_mask]
  windows = windows_raw[stim_mask] if windows_raw.shape[0] == len(y_raw) else windows_raw[:len(y_stim)]
  channel_names = resolve_channel_names(windows.shape[1])

  qc = apply_artifact_quality_pipeline(windows, y_stim, channel_names=channel_names, verbose=False)
  windows_clean, y_clean = qc["windows_clean"], qc["y_clean"]

  # Trial ids computed on the PRE-quality-gate label array (same window
  # order qc["valid_mask"] indexes into), then filtered the same way as
  # y_clean -- so sub-windows of the same trial never split across folds.
  trial_ids_full = build_trial_ids(y_stim, sub_windows_per_trial=5)
  trial_ids_clean = trial_ids_full[qc["valid_mask"] == 1]

  classes = np.unique(y_clean)
  if len(y_clean) < 10 or len(classes) < 2:
    print(f"[-] Skipping {sub}: not enough clean windows/classes.")
    return None
  class_to_idx = {c: i for i, c in enumerate(classes)}
  y_mapped = np.array([class_to_idx[c] for c in y_clean])

  print(f"\n[{sub}] {len(y_clean)} clean windows, {windows_clean.shape[1]} channels | running {model_name}...")
  acc_1s_off, acc_smooth_off, train_acc_off = run_cv(windows_clean, y_mapped, trial_ids_clean, classes, model_name, args, augment=False)
  print(f"  augment OFF -> 1.0s: {acc_1s_off:.2f}% | 2.0s smoothed: {acc_smooth_off:.2f}% | train acc: {train_acc_off:.2f}%")
  acc_1s_on, acc_smooth_on, train_acc_on = run_cv(windows_clean, y_mapped, trial_ids_clean, classes, model_name, args, augment=True)
  print(f"  augment ON  -> 1.0s: {acc_1s_on:.2f}% | 2.0s smoothed: {acc_smooth_on:.2f}% | train acc: {train_acc_on:.2f}%")
  if train_acc_off < 40.0:
    print(f"  [!] Train accuracy is also low OFF ({train_acc_off:.2f}%) -- the model isn't fitting "
          f"this subject's training data either, not just failing to generalize. Likely an "
          f"optimization issue (LR/epochs/init), not a data-quality one.")

  return {
      "Subject": sub,
      "Clean Windows": len(y_clean),
      "1.0s OFF (%)": round(acc_1s_off, 2),
      "1.0s ON (%)": round(acc_1s_on, 2),
      "1.0s Delta": round(acc_1s_on - acc_1s_off, 2),
      "2.0s OFF (%)": round(acc_smooth_off, 2),
      "2.0s ON (%)": round(acc_smooth_on, 2),
      "2.0s Delta": round(acc_smooth_on - acc_smooth_off, 2),
      "Train Acc OFF (%)": round(train_acc_off, 2),
      "Train Acc ON (%)": round(train_acc_on, 2),
  }


def main():
  parser = argparse.ArgumentParser(
      description="Compare training-time augmentation ON vs OFF across all discovered subjects."
  )
  parser.add_argument("--model", choices=["eegnet", "compact_cnn", "shallow_conv_net"], default="eegnet")
  parser.add_argument("--epochs", type=int, default=150)
  parser.add_argument("--batch-size", type=int, default=32)
  parser.add_argument("--lr", type=float, default=1e-3)
  parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  args = parser.parse_args()

  project_root = Path(__file__).resolve().parents[3]
  data_root = project_root / "data" / "processed"
  subjects = discover_subjects(data_root)

  print("=" * 80)
  print(f"   AUGMENTATION A/B BENCHMARK -- model={args.model}  (N={len(subjects)} subjects)")
  print("=" * 80)

  if not subjects:
    print(f"No subject folders matching 'S<digits>' found under {data_root}.")
    return

  records = [r for r in (process_subject(sub, data_root, args.model, args) for sub in subjects) if r is not None]
  df = pd.DataFrame(records)

  print("\n" + "=" * 80)
  print(df.to_string(index=False))
  print("=" * 80)
  if len(df) > 0:
    print(
        f"Cohort mean | 1.0s OFF: {df['1.0s OFF (%)'].mean():.2f}% -> ON: {df['1.0s ON (%)'].mean():.2f}% "
        f"(Delta: {df['1.0s Delta'].mean():+.2f}) | "
        f"2.0s OFF: {df['2.0s OFF (%)'].mean():.2f}% -> ON: {df['2.0s ON (%)'].mean():.2f}% "
        f"(Delta: {df['2.0s Delta'].mean():+.2f})"
    )
    n_improved = (df["1.0s Delta"] > 0).sum()
    print(f"Subjects improved by augmentation (1.0s): {n_improved}/{len(df)}")

  out_path = project_root / "reports" / f"augmentation_ab_{args.model}.csv"
  out_path.parent.mkdir(parents=True, exist_ok=True)
  df.to_csv(out_path, index=False)
  print(f"Results saved to: {out_path}")


if __name__ == "__main__":
  main()