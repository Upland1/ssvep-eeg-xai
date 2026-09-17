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

FULL_MONTAGE = ["PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"]


def resolve_channel_names(n_channels: int, full_montage: list[str] = FULL_MONTAGE) -> list[str] | None:
  """Match the actual channel count to known montage layouts."""
  if n_channels == len(full_montage):
    return list(full_montage)
  if n_channels == len(full_montage) - 1:
    print(
        f"[!] {n_channels} channels found (expected {len(full_montage)}); "
        "assuming PO7 is still missing from this data -- rerun "
        "preprocessing to include it."
    )
    return full_montage[1:]
  print(f"[!] Unexpected channel count ({n_channels}); channel names unavailable.")
  return None


# 1. Parse CLI Arguments
parser = argparse.ArgumentParser(
    description="Evaluate out-of-fold confusion matrix and smoothing for SSVEP hybrid model."
)
parser.add_argument(
    "--subject",
    default=None,
    help="Subject subdirectory (e.g. S01, S03). Defaults to data/processed.",
)
args = parser.parse_args()

project_root = Path(__file__).resolve().parents[1]
data_dir = project_root / "data" / "processed"
if args.subject:
  data_dir = data_dir / args.subject

# 2. Load arrays safely
windows_raw = np.load(data_dir / "X_time_windows.npy")
y_raw = np.load(data_dir / "y_labels.npy")

stim_mask = np.isin(y_raw, [101, 102, 103, 104, 105])
y_stim = y_raw[stim_mask]
windows = (
    windows_raw[stim_mask]
    if windows_raw.shape[0] == len(y_raw)
    else windows_raw[: len(y_stim)]
)

# 3. Extract FBCCA with the two-tier quality gate (chronic bad channels
# dropped for this subject first, then remaining noisy windows rejected)
channel_names = resolve_channel_names(windows.shape[1])
X_fbcca, y_clean, valid_mask, ch_report = extract_fbcca_features(
    windows, y_stim, fs=250.0, channel_names=channel_names
)
windows_clean = windows[valid_mask == 1][:, ch_report["channels_kept_idx"], :]

print(f"Subject: {args.subject or 'Root'}")
print(f"Verified clean windows shape: {windows_clean.shape}")
print(f"Verified clean labels shape:  {y_clean.shape}")
if ch_report["channels_dropped"]:
  print(f"Channels dropped for this subject: {ch_report['channels_dropped']}")

# 4. Stratified 5-Fold Cross-Validation (m=1 CSP component pair + FBCCA)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_probs = np.zeros((len(y_clean), 5))

for train_idx, test_idx in skf.split(windows_clean, y_clean):
  y_tr, y_te = y_clean[train_idx], y_clean[test_idx]

  X_tr_fold, X_te_fold = [], []
  for low, high in DEFAULT_SUBBANDS:
    w_tr = butter_bandpass_filter(
        windows_clean[train_idx], low, high, fs=250.0, order=4
    )
    w_te = butter_bandpass_filter(
        windows_clean[test_idx], low, high, fs=250.0, order=4
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

  # Match probabilities strictly to sorted unique classes
  fold_probs = clf.predict_proba(X_te_scaled)
  for col_idx, cls_label in enumerate(clf.classes_):
    target_col = np.where(np.unique(y_clean) == cls_label)[0][0]
    oof_probs[test_idx, target_col] = fold_probs[:, col_idx]

# 5. Instantaneous 1.0s Accuracy
classes = np.unique(y_clean)
preds_1s = classes[np.argmax(oof_probs, axis=1)]
acc_1s = accuracy_score(y_clean, preds_1s) * 100.0

# 6. Trial-Aware Causal Smoothing (Resets every 5 sub-windows per trial)
trial_len = 5
preds_smoothed = np.zeros_like(y_clean)

for i in range(len(y_clean)):
  pos = i % trial_len
  if pos == 0:
    avg_p = oof_probs[i]
  else:
    avg_p = np.mean(oof_probs[i - 1 : i + 1], axis=0)
  preds_smoothed[i] = classes[np.argmax(avg_p)]

acc_smooth = accuracy_score(y_clean, preds_smoothed) * 100.0

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
print(classification_report(y_clean, preds_1s, target_names=target_names))

print("--- 2.0s Smoothed Classification Report ---")
print(classification_report(y_clean, preds_smoothed, target_names=target_names))

# 7. Save Confusion Matrix
cm = confusion_matrix(y_clean, preds_smoothed)
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
    f"Core FBCSP (m=1) + FBCCA (2.0s Smoothed) - {args.subject or 'S01'}\nMean"
    f" CV Accuracy: {acc_smooth:.2f}%"
)
ax.set_xlabel("Predicted Class")
ax.set_ylabel("True Class")
plt.tight_layout()

out_dir = project_root / "outputs" / "figures" / "confusion_matrices"
out_dir.mkdir(parents=True, exist_ok=True)
sub_tag = f"_{args.subject}" if args.subject else ""
out_path = out_dir / f"cm_fbcsp_m1_fbcca_cv{sub_tag}.png"
plt.savefig(out_path, dpi=150)
plt.close(fig)
print(f"\nUpdated confusion matrix saved to: {out_path}")