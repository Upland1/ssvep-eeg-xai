from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

project_root = Path(__file__).resolve().parents[1]
data_dir = project_root / "data" / "processed"

windows = np.load(data_dir / "X_time_windows.npy")
y = np.load(data_dir / "y_labels.npy")

stim_mask = np.isin(y, [101, 102, 103, 104, 105])
windows = windows[stim_mask]
y = y[stim_mask]

# Occipital/visual channels are typically the last 4: [3, 4, 5, 6] in 7-channel montage
visual_channels = [3, 4, 5, 6] if windows.shape[1] >= 7 else list(range(windows.shape[1]))
windows_occ = windows[:, visual_channels, :]

# Compute FFT power spectrum (fs = 250 Hz, 256 samples -> ~0.98 Hz resolution)
fs = 250.0
n_samples = windows.shape[-1]
freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
fft_power = np.abs(np.fft.rfft(windows_occ, axis=-1)) ** 2
mean_power_across_channels = np.mean(fft_power, axis=1)

# Plot class spectra
classes = [101, 102, 103, 104, 105]
freq_labels = {101: "24 Hz", 102: "20 Hz", 103: "15 Hz", 104: "10.91 Hz", 105: "8.57 Hz"}

fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
mask_band = (freqs >= 5.0) & (freqs <= 35.0)

for i, cls in enumerate(classes):
    idx = (y == cls)
    avg_spectrum = np.mean(mean_power_across_channels[idx], axis=0)
    
    axes[i].plot(freqs[mask_band], avg_spectrum[mask_band], color="darkblue", linewidth=1.5)
    axes[i].set_title(f"Class {cls} ({freq_labels[cls]} Target) - Occipital Spectrum")
    axes[i].set_ylabel("Power")
    axes[i].grid(True, alpha=0.3)
    
    # Highlight expected target frequency
    target_f = float(freq_labels[cls].split()[0])
    axes[i].axvline(target_f, color="crimson", linestyle="--", label=f"Fundamental ({target_f} Hz)")
    axes[i].legend(loc="upper right")

axes[-1].set_xlabel("Frequency (Hz)")
plt.tight_layout()

fig_path = project_root / "outputs" / "figures" / "spectra" / "occipital_class_spectra_check.png"
fig_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_path, dpi=150)
plt.close(fig)
print(f"Occipital spectra diagnostic saved to: {fig_path}")
