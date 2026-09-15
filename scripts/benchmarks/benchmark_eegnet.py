"""Stratified 5-Fold Cross-Validation Benchmark for EEGNet on Raw SSVEP Windows."""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from src.preprocessing.quality_check import filter_dataset, validate_eeg_windows
from src.models.eegnet import EEGNetSSVEP

parser = argparse.ArgumentParser(description="Benchmark EEGNet on raw SSVEP EEG segments.")
parser.add_argument("--subject", default="S03", help="Subject subdirectory (default: S03).")
parser.add_argument("--epochs", type=int, default=150, help="Training epochs per fold (default: 150).")
parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32).")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3).")
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()

project_root = Path(__file__).resolve().parents[2]
data_dir = project_root / "data" / "processed" / args.subject

# 1. Load data
windows_raw = np.load(data_dir / "X_time_windows.npy")
y_raw = np.load(data_dir / "y_labels.npy")
stim_mask = np.isin(y_raw, [101, 102, 103, 104, 105])
y_stim = y_raw[stim_mask]
windows = windows_raw[stim_mask] if windows_raw.shape[0] == len(y_raw) else windows_raw[:len(y_stim)]

# 2. Quality check (rejection of artifacts)
valid_mask = validate_eeg_windows(windows, vpp_min=0.5, vpp_max=120.0, std_min=0.1, std_max=35.0)
windows_clean, y_clean = filter_dataset(windows, y_stim, valid_mask)
print(f"Subject: {args.subject} | Verified Clean Windows: {len(y_clean)} | Device: {args.device}")

# 3. Class mapping 101-105 -> 0-4
classes = np.unique(y_clean)
class_to_idx = {c: i for i, c in enumerate(classes)}
y_mapped = np.array([class_to_idx[c] for c in y_clean])

# 4. Stratified 5-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_probs = np.zeros((len(y_clean), len(classes)))

for fold_idx, (train_idx, test_idx) in enumerate(skf.split(windows_clean, y_mapped), 1):
    X_tr, y_tr = windows_clean[train_idx], y_mapped[train_idx]
    X_te, y_te = windows_clean[test_idx], y_mapped[test_idx]

    # In-fold temporal standardization per channel
    # Per-window, per-channel temporal standardization (dim -1 = 256 timepoints)
    mean_tr = np.mean(X_tr, axis=-1, keepdims=True)
    std_tr = np.std(X_tr, axis=-1, keepdims=True) + 1e-8
    X_tr_norm = (X_tr - mean_tr) / std_tr

    mean_te = np.mean(X_te, axis=-1, keepdims=True)
    std_te = np.std(X_te, axis=-1, keepdims=True) + 1e-8
    X_te_norm = (X_te - mean_te) / std_te

    # Convert to PyTorch Tensors
    t_X_tr = torch.tensor(X_tr_norm, dtype=torch.float32).unsqueeze(1)  # (B, 1, 7, 256)
    t_y_tr = torch.tensor(y_tr, dtype=torch.long)
    t_X_te = torch.tensor(X_te_norm, dtype=torch.float32).unsqueeze(1).to(args.device)

    train_ds = TensorDataset(t_X_tr, t_y_tr)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    # Initialize model
    model = EEGNetSSVEP(
        n_classes=len(classes),
        n_channels=windows_clean.shape[1],
        n_samples=windows_clean.shape[2],
        dropout_rate=0.25
    ).to(args.device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    model.train()
    for epoch in range(args.epochs):
        for bx, by in train_loader:
            bx, by = bx.to(args.device), by.to(args.device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

    # Inference on out-of-fold test set
    model.eval()
    with torch.no_grad():
        test_logits = model(t_X_te)
        fold_probs = torch.softmax(test_logits, dim=-1).cpu().numpy()
        oof_probs[test_idx] = fold_probs

    fold_acc = accuracy_score(y_te, np.argmax(fold_probs, axis=1)) * 100.0
    print(f"  Fold {fold_idx}/5 | Accuracy: {fold_acc:.2f}%")

# 5. Compute Metrics
preds_1s = np.argmax(oof_probs, axis=1)
acc_1s = accuracy_score(y_mapped, preds_1s) * 100.0

# 2.0s causal trial-aware smoothing
preds_smooth = np.zeros_like(y_mapped)
for i in range(len(y_mapped)):
    if i % 5 == 0:
        avg_p = oof_probs[i]
    else:
        avg_p = np.mean(oof_probs[i - 1 : i + 1], axis=0)
    preds_smooth[i] = np.argmax(avg_p)
acc_smooth = accuracy_score(y_mapped, preds_smooth) * 100.0

print("\n" + "=" * 70)
print(f"            EEGNet BENCHMARK RESULTS (SUBJECT {args.subject})")
print("=" * 70)
print(f"Instantaneous 1.0s Accuracy:       {acc_1s:.2f}%")
print(f"Trial-Aware 2.0s Smoothed Accuracy: {acc_smooth:.2f}%")
print("=" * 70)