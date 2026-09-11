from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.features.fbcca_extraction import extract_fbcca_features

project_root = Path(__file__).resolve().parents[1]
data_dir = project_root / "data" / "processed"

windows_raw = np.load(data_dir / "X_time_windows.npy")
y_raw = np.load(data_dir / "y_labels.npy")

stim_mask = np.isin(y_raw, [101, 102, 103, 104, 105])
y_stim = y_raw[stim_mask]
windows = windows_raw[stim_mask] if windows_raw.shape[0] == len(y_raw) else windows_raw[:len(y_stim)]

# 1. Extract Occipital Harmonic Features (32 dims: 4 visual channels x 8 targeted harmonic bins)
visual_ch = [3, 4, 5, 6] if windows.shape[1] >= 7 else list(range(windows.shape[1]))
target_freqs = [8.57, 10.91, 15.0, 17.14, 20.0, 21.82, 24.0, 30.0]
freqs = np.linspace(0.0, 125.0, windows.shape[-1] // 2 + 1)
target_idx = [np.argmin(np.abs(freqs - f)) for f in target_freqs]

fft_vals = np.abs(np.fft.rfft(windows[:, visual_ch, :], axis=-1)) ** 2
X_harm = fft_vals[:, :, target_idx].reshape(windows.shape[0], -1)

# 2. Extract FBCCA features (5 dims)
X_fbcca, _, _ = extract_fbcca_features(windows[:, visual_ch, :], y_stim, fs=250.0)

# 3. Form Hybrid Matrix (37 dims)
X_hybrid = np.hstack([X_harm, X_fbcca])

# 4. Stratified 5-Fold Cross-Validation with Shrinkage-LDA
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(y_stim), dtype=int)

for train_idx, test_idx in skf.split(X_hybrid, y_stim):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_hybrid[train_idx])
    X_te = scaler.transform(X_hybrid[test_idx])
    
    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    clf.fit(X_tr, y_stim[train_idx])
    oof_preds[test_idx] = clf.predict(X_te)

acc = accuracy_score(y_stim, oof_preds) * 100.0
print("=" * 60)
print(f"5-Fold CV Accuracy (Occipital Harmonic PSD + FBCCA): {acc:.2f}%")
print("=" * 60)

target_names = ["101 (24 Hz)", "102 (20 Hz)", "103 (15 Hz)", "104 (10.91 Hz)", "105 (8.57 Hz)"]
print("\n" + classification_report(y_stim, oof_preds, target_names=target_names))

cm = confusion_matrix(y_stim, oof_preds)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm_norm, annot=True, fmt=".1%", cmap="Blues",
            xticklabels=target_names, yticklabels=target_names, ax=ax)
ax.set_title(f"Occipital (Harmonic PSD + FBCCA) Shrinkage-LDA\nMean CV Accuracy: {acc:.2f}%")
ax.set_xlabel("Predicted Class")
ax.set_ylabel("True Class")
plt.tight_layout()

out_cm = project_root / "outputs" / "figures" / "confusion_matrices" / "cm_harmonic_fbcca_cv.png"
plt.savefig(out_cm, dpi=150)
plt.close(fig)
print(f"New confusion matrix saved to: {out_cm}")
