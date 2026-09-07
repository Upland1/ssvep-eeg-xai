from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np

from src.features.fbcca_extraction import extract_fbcca_features
from src.features.fbcsp_extraction import extract_fbcsp_features
from src.features.psd_extraction import extract_continuous_psd_features

parser = argparse.ArgumentParser(
    description="Verify and visualize PSD, FBCCA, and FBCSP extractions."
)
parser.add_argument(
    "--subject",
    help="Subject directory (e.g. S01). Defaults to root processed folder.",
)
args = parser.parse_args()

project_root = Path(__file__).resolve().parents[1]
data_dir = project_root / "data" / "processed"
if args.subject:
  data_dir = data_dir / args.subject

# 1. Load raw time windows and labels
windows_raw = np.load(data_dir / "X_time_windows.npy")
y_raw = np.load(data_dir / "y_labels.npy")

print(f"Loaded X_time_windows shape: {windows_raw.shape}")
print(f"Loaded y_labels shape:       {y_raw.shape}")

# 2. Filter labels to stimulus conditions (101-105)
stim_mask = np.isin(y_raw, [101, 102, 103, 104, 105])
y_stim = y_raw[stim_mask]  # This has 200 elements

# 3. Slice windows to match the 200 stimulus trials
if windows_raw.shape[0] == len(y_raw):
  windows = windows_raw[stim_mask]
else:
  # When X_time_windows contains raw blocks (e.g. 320), take the first 200 to match y_stim
  windows = windows_raw[: len(y_stim)]
  
# 2. Run all three extractors with artifact validation
X_psd, y_clean_psd, mask_psd = extract_continuous_psd_features(
    windows, y_stim, fs=250.0
)
X_fbcca, y_clean_fbcca, mask_fbcca = extract_fbcca_features(
    windows, y_stim, fs=250.0
)
X_fbcsp, y_clean_fbcsp, mask_fbcsp = extract_fbcsp_features(
    windows, y_stim, fs=250.0
)

# 3. Print verification and sanity metrics
print("=" * 60)
print("         BINARY VALIDATION MASK & MATRIX VERIFICATION")
print("=" * 60)
print(f"Total raw input windows:   {len(mask_psd)}")
print(f"Valid windows accepted (1): {int(np.sum(mask_psd))}")
print(f"Noisy windows rejected (0): {int(np.sum(mask_psd == 0))}")
print(f"Mask values sample:        {mask_psd[:15].tolist()}...")

masks_identical = np.array_equal(mask_psd, mask_fbcca) and np.array_equal(
    mask_psd, mask_fbcsp
)
labels_identical = np.array_equal(y_clean_psd, y_clean_fbcca) and np.array_equal(
    y_clean_psd, y_clean_fbcsp
)
print(f"\nMasks strictly identical across all 3 methods:  {masks_identical}")
print(f"Labels strictly identical across all 3 methods: {labels_identical}")

print("\n--- Output Dimensions & Ranges ---")
print(
    f"PSD:   Shape={X_psd.shape} | NaNs={np.isnan(X_psd).any()} |"
    f" Range=[{X_psd.min():.2f}, {X_psd.max():.2f}]"
)
print(
    f"FBCCA: Shape={X_fbcca.shape} | NaNs={np.isnan(X_fbcca).any()} |"
    f" Range=[{X_fbcca.min():.4f}, {X_fbcca.max():.4f}]"
)
print(
    f"FBCSP: Shape={X_fbcsp.shape} | NaNs={np.isnan(X_fbcsp).any()} |"
    f" Range=[{X_fbcsp.min():.4f}, {X_fbcsp.max():.4f}]"
)

classes, counts = np.unique(y_clean_psd, return_counts=True)
print("\n--- Valid Windows per Class ---")
for cls, count in zip(classes, counts):
  print(f"Condition {cls}: {count} windows")
print("=" * 60)

# 4. Multi-Panel Visual Inspection
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Panel 1: Mean FBCCA response profiles
target_labels = ["24 Hz", "20 Hz", "15 Hz", "10.91 Hz", "8.57 Hz"]
for cls in classes:
  idx = y_clean_fbcca == cls
  axes[0].plot(
      target_labels,
      np.mean(X_fbcca[idx], axis=0),
      marker="o",
      linewidth=2,
      label=f"Class {cls}",
  )
axes[0].set_title("FBCCA: Mean Template Correlation")
axes[0].set_xlabel("Reference Target Frequency")
axes[0].set_ylabel("Correlation Score (rho^2)")
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Panel 2: Continuous PSD energy distribution
axes[1].boxplot([X_psd[y_clean_psd == cls, :5].flatten() for cls in classes])
axes[1].set_xticklabels([f"Class {cls}" for cls in classes])
axes[1].set_title("PSD: Spectral Power (First 5 Bins)")
axes[1].set_ylabel("Power (uV^2/Hz)")
axes[1].grid(True, alpha=0.3)

# Panel 3: FBCSP normalized log-variances
axes[2].boxplot([X_fbcsp[y_clean_fbcsp == cls, :5].flatten() for cls in classes])
axes[2].set_xticklabels([f"Class {cls}" for cls in classes])
axes[2].set_title("FBCSP: Normalized Log-Variance (Sub-band 1)")
axes[2].set_ylabel("Log-Variance")
axes[2].grid(True, alpha=0.3)