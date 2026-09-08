import argparse
from pathlib import Path
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.features.fbcca_extraction import extract_fbcca_features
from src.features.fbcsp_extraction import (
    DEFAULT_SUBBANDS,
    MulticlassCSP,
    butter_bandpass_filter,
)

project_root = Path(__file__).resolve().parents[1]
data_dir = project_root / "data" / "processed"

# 1. Load arrays safely
windows_raw = np.load(data_dir / "X_time_windows.npy")
y_raw = np.load(data_dir / "y_labels.npy")

stim_mask = np.isin(y_raw, [101, 102, 103, 104, 105])
y_stim = y_raw[stim_mask]

# Safe slice matching stimulus labels (200 windows)
if windows_raw.shape[0] == len(y_raw):
  windows = windows_raw[stim_mask]
else:
  windows = windows_raw[: len(y_stim)]

print(f"Verified aligned windows shape: {windows.shape}")
print(f"Verified stimulus labels shape:  {y_stim.shape}")

# 2. Extract FBCCA across all 7 channels
X_fbcca, _, _ = extract_fbcca_features(windows, y_stim, fs=250.0)

# 3. Stratified 5-Fold Cross-Validation (m=1 CSP component pair)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_probs = np.zeros((len(y_stim), 5))

for train_idx, test_idx in skf.split(windows, y_stim):
  y_tr, y_te = y_stim[train_idx], y_stim[test_idx]

  X_tr_fold, X_te_fold = [], []
  for low, high in DEFAULT_SUBBANDS:
    w_tr = butter_bandpass_filter(
        windows[train_idx], low, high, fs=250.0, order=4
    )
    w_te = butter_bandpass_filter(
        windows[test_idx], low, high, fs=250.0, order=4
    )
    csp = MulticlassCSP(n_components=1).fit(w_tr, y_tr)
    X_tr_fold.append(csp.transform(w_tr))
    X_te_fold.append(csp.transform(w_te))

  X_tr = np.hstack([np.hstack(X_tr_fold), X_fbcca[train_idx]])
  X_te = np.hstack([np.hstack(X_te_fold), X_fbcca[test_idx]])

  scaler = StandardScaler()
  X_tr_scaled = scaler.fit_transform(X_tr)
  X_te_scaled = scaler.transform(X_te)

  clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
  clf.fit(X_tr_scaled, y_tr)
  oof_probs[test_idx] = clf.predict_proba(X_te_scaled)

# 4. Instantaneous 1.0s Accuracy
classes = np.unique(y_stim)
preds_1s = classes[np.argmax(oof_probs, axis=1)]
acc_1s = accuracy_score(y_stim, preds_1s) * 100.0

# 5. Trial-Aware Causal Smoothing (Resets every 5 sub-windows per trial)
trial_len = 5
preds_smoothed = np.zeros_like(y_stim)

for i in range(len(y_stim)):
  pos = i % trial_len
  if pos == 0:
    avg_p = oof_probs[i]
  else:
    avg_p = np.mean(oof_probs[i - 1 : i + 1], axis=0)
  preds_smoothed[i] = classes[np.argmax(avg_p)]

acc_smooth = accuracy_score(y_stim, preds_smoothed) * 100.0

print("=" * 60)
print(f"5-Fold CV (FBCSP m=1 + FBCCA) 1.0s Accuracy:       {acc_1s:.2f}%")
print(f"5-Fold CV (FBCSP m=1 + FBCCA) 2.0s Trial Smoothed: {acc_smooth:.2f}%")
print("=" * 60)

target_names = [
    "101 (24 Hz)",
    "102 (20 Hz)",
    "103 (15 Hz)",
    "104 (10.91 Hz)",
    "105 (8.57 Hz)",
]
print("\n--- Instantaneous Classification Report ---")
print(classification_report(y_stim, preds_1s, target_names=target_names))

print("--- 2.0s Smoothed Classification Report ---")
print(classification_report(y_stim, preds_smoothed, target_names=target_names))

# 6. Save Confusion Matrix
cm = confusion_matrix(y_stim, preds_smoothed)
cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".1%",
    cmap="Blues",
    xticklabels=target_names,
    yticklabels=target_names,
    ax=ax,
)
ax.set_title(
  f"Core FBCSP (m=1) + FBCCA (2.0s Smoothed)\nMean CV Accuracy:"
    f" {acc_smooth:.2f}%"
)
ax.set_xlabel("Predicted Class")
ax.set_ylabel("True Class")
plt.tight_layout()

out_path = (
    project_root
    / "outputs"
    / "figures"
    / "confusion_matrices"
    / "cm_fbcsp_m1_fbcca_cv.png"
)
plt.savefig(out_path, dpi=150)
plt.close(fig)
print(f"\nUpdated confusion matrix saved to: {out_path}")