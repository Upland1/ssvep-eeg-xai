import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from src.features.fbcca_extraction import extract_fbcca_features
from src.preprocessing.cv_utils import build_trial_ids

# Full recorded montage (8 channels). The "visual" cluster is named
# explicitly here -- selecting by NAME rather than fixed position [3,4,5,6]
# is required now that PO7 is included as channel 0; a positional index
# would silently point at the wrong electrodes.
FULL_MONTAGE = ["PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"]
VISUAL_CHANNEL_NAMES = ["PO8", "O1", "Oz", "O2"]


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


def get_visual_channel_indices(
    n_channels: int,
    visual_names: list[str] = VISUAL_CHANNEL_NAMES,
    channel_names: list[str] | None = None,
) -> list[int]:
  """Return indices of the occipital/visual cluster, resolved by NAME."""
  names = channel_names if channel_names is not None else resolve_channel_names(n_channels)
  if names is not None:
    idx = [i for i, name in enumerate(names) if name in visual_names]
    if idx:
      return idx
  # Unknown montage: fall back to the old positional assumption.
  return [3, 4, 5, 6] if n_channels >= 7 else list(range(n_channels))


parser = argparse.ArgumentParser(
    description="Occipital Harmonic PSD + FBCCA hybrid benchmark."
)
parser.add_argument(
    "--subject",
    default=None,
    help="Subject subdirectory (e.g. S01, S04). Defaults to data/processed root.",
)
args = parser.parse_args()

project_root = Path(__file__).resolve().parents[2]
data_dir = project_root / "data" / "processed"
if args.subject:
  data_dir = data_dir / args.subject

windows_raw = np.load(data_dir / "X_time_windows.npy")
y_raw = np.load(data_dir / "y_labels.npy")

stim_mask = np.isin(y_raw, [101, 102, 103, 104, 105])
y_stim = y_raw[stim_mask]
windows = windows_raw[stim_mask] if windows_raw.shape[0] == len(y_raw) else windows_raw[:len(y_stim)]

# 1. Extract Occipital Harmonic Features (32 dims: 4 visual channels x 8 targeted harmonic bins)
visual_names = resolve_channel_names(windows.shape[1])
visual_ch = get_visual_channel_indices(windows.shape[1], channel_names=visual_names)
visual_ch_names = [visual_names[i] for i in visual_ch] if visual_names is not None else None
target_freqs = [8.57, 10.91, 15.0, 17.14, 20.0, 21.82, 24.0, 30.0]
freqs = np.linspace(0.0, 125.0, windows.shape[-1] // 2 + 1)
target_idx = [np.argmin(np.abs(freqs - f)) for f in target_freqs]

fft_vals = np.abs(np.fft.rfft(windows[:, visual_ch, :], axis=-1)) ** 2
X_harm = fft_vals[:, :, target_idx].reshape(windows.shape[0], -1)

# 2. Extract FBCCA features (5 dims). The artifact-quality pipeline still
# runs inside extract_fbcca_features, now over just the visual channels
# selected above -- it may drop one of them further if it's noisy for
# this subject.
X_fbcca, y_clean_fbcca, valid_mask, ch_report = extract_fbcca_features(
    windows[:, visual_ch, :], y_stim, fs=250.0, channel_names=visual_ch_names
)

# 3. Form Hybrid Matrix. NOTE: X_harm was computed on ALL windows (no
# quality gate), while X_fbcca/y_clean_fbcca only cover the windows that
# passed the quality gate -- align them before stacking.
X_harm_clean = X_harm[valid_mask == 1]
y_stim_clean = y_stim[valid_mask == 1]
X_hybrid = np.hstack([X_harm_clean, X_fbcca])

# Trial ids computed on the PRE-quality-gate label array, then filtered
# the same way as y_stim_clean -- sub-windows of the same 5.0s trial must
# never split across train/test folds.
trial_ids_full = build_trial_ids(y_stim, sub_windows_per_trial=5)
trial_ids_clean = trial_ids_full[valid_mask == 1]

# 4. Stratified GROUP 5-Fold Cross-Validation with Shrinkage-LDA
skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(y_stim_clean), dtype=int)

for train_idx, test_idx in skf.split(X_hybrid, y_stim_clean, groups=trial_ids_clean):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_hybrid[train_idx])
    X_te = scaler.transform(X_hybrid[test_idx])
    
    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    clf.fit(X_tr, y_stim_clean[train_idx])
    oof_preds[test_idx] = clf.predict(X_te)

acc = accuracy_score(y_stim_clean, oof_preds) * 100.0
print("=" * 60)
print(f"5-Fold CV Accuracy (Occipital Harmonic PSD + FBCCA): {acc:.2f}%")
if ch_report["channels_dropped"]:
    print(f"Visual channels dropped by quality gate: {ch_report['channels_dropped']}")
print("=" * 60)

target_names = ["101 (24 Hz)", "102 (20 Hz)", "103 (15 Hz)", "104 (10.91 Hz)", "105 (8.57 Hz)"]
print("\n" + classification_report(y_stim_clean, oof_preds, target_names=target_names))

cm = confusion_matrix(y_stim_clean, oof_preds)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm_norm, annot=True, fmt=".1%", cmap="Blues",
            xticklabels=target_names, yticklabels=target_names, ax=ax)
ax.set_title(f"Occipital (Harmonic PSD + FBCCA) Shrinkage-LDA\nMean CV Accuracy: {acc:.2f}%")
ax.set_xlabel("Predicted Class")
ax.set_ylabel("True Class")
plt.tight_layout()

out_dir = project_root / "outputs" / "figures" / "confusion_matrices"
out_dir.mkdir(parents=True, exist_ok=True)
sub_tag = f"_{args.subject}" if args.subject else ""
out_cm = out_dir / f"cm_harmonic_fbcca_cv{sub_tag}.png"
plt.savefig(out_cm, dpi=150)
plt.close(fig)
print(f"New confusion matrix saved to: {out_cm}")