import numpy as np
from scipy.linalg import eigh
from scipy.signal import butter, filtfilt
from src.preprocessing.signal.quality_check import apply_artifact_quality_pipeline

DEFAULT_SUBBANDS = [
    (7.0, 12.0),
    (13.0, 17.0),
    (18.0, 26.0),
    (27.0, 36.0),
    (38.0, 52.0),
]


def butter_bandpass_filter(
    data: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: float = 250.0,
    order: int = 4,
) -> np.ndarray:
  nyq = 0.5 * fs
  b, a = butter(
      order, [lowcut / nyq, min(highcut, nyq - 1.0) / nyq], btype='band'
  )
  return filtfilt(b, a, data, axis=-1)


class MulticlassCSP:

  def __init__(self, n_components: int = 1):
    self.n_components = n_components
    self.filters_ = {}

  def _compute_cov(self, windows: np.ndarray) -> np.ndarray:
    n_windows, n_channels, _ = windows.shape
    covs = np.zeros((n_channels, n_channels))
    for i in range(n_windows):
      win = windows[i]
      cov = np.dot(win, win.T)
      trace = np.trace(cov)
      covs += cov / (trace if trace > 0 else 1.0)
    return covs / n_windows

  def fit(self, X: np.ndarray, y: np.ndarray):
    classes = np.unique(y)
    self.filters_ = {}
    for cls in classes:
      idx_target = y == cls
      idx_rest = y != cls
      cov_target = self._compute_cov(X[idx_target])
      cov_rest = self._compute_cov(X[idx_rest])
      cov_sum = cov_target + cov_rest + 1e-6 * np.eye(cov_target.shape[0])
      eigenvalues, eigenvectors = eigh(cov_target, cov_sum)
      idx_sort = np.argsort(eigenvalues)[::-1]
      eigenvectors = eigenvectors[:, idx_sort]
      m = self.n_components
      w_top = eigenvectors[:, :m]
      w_bottom = eigenvectors[:, -m:]
      self.filters_[cls] = np.hstack([w_top, w_bottom])
    return self

  def transform(self, X: np.ndarray) -> np.ndarray:
    n_windows, n_channels, _ = X.shape
    features = []
    for i in range(n_windows):
      win = X[i]
      win_feats = []
      for cls, W in self.filters_.items():
        projected = np.dot(W.T, win)
        variances = np.maximum(np.var(projected, axis=-1), 1e-12)
        norm_log_var = np.log10(variances / np.sum(variances))
        win_feats.extend(norm_log_var)
      features.append(win_feats)
    return np.array(features)


def extract_fbcsp_features(
    windows: np.ndarray,
    y: np.ndarray,
    fs: float = 250.0,
    subbands: list[tuple[float, float]] = DEFAULT_SUBBANDS,
    n_components: int = 1,
    channel_names: list[str] | None = None,
    vpp_limits: tuple[float, float] = (5.0, 120.0),
    std_limits: tuple[float, float] = (1.0, 35.0),
    channel_fail_fraction_thresh: float = 0.5,
    verbose: bool = False,
    subband_windows: np.ndarray | None = None,
    quality: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
  """Validate windows (channel-level + window-level) and extract FBCSP features.

  bad channels are dropped for this subject first, then remaining
  noisy windows are rejected on the surviving channels. CSP is
  fit per sub-band on the clean, channel-reduced data.

  WARNING: this fits CSP on ALL windows passed in. For cross-validated
  accuracy, fit CSP inside each fold instead (as benchmark_n_subjects.py
  does); use this function only for visualisation / full-data fits.

  subband_windows : optional, shape (n_windows, n_subbands, n_channels, n_samples)
      Pre-filtered sub-band windows aligned with `windows` (one band per
      entry of `subbands`). When given, no filtering is done here. When
      None, each window is filtered on its own (legacy, with edge effects).

  quality : optional, the subject's shared quality decision (from
      `src.io.subject_data.load_subject(...)["quality"]`). When given, the
      two-tier gate is NOT re-run here; that decision (window mask + kept
      channels) is applied as-is, so every analysis uses the same windows.
  """
  qc = apply_artifact_quality_pipeline(
      windows,
      y,
      channel_names=channel_names,
      vpp_min=vpp_limits[0],
      vpp_max=vpp_limits[1],
      std_min=std_limits[0],
      std_max=std_limits[1],
      channel_fail_fraction_thresh=channel_fail_fraction_thresh,
      verbose=verbose,
      precomputed=quality,
  )
  clean_windows, y_clean = qc["windows_clean"], qc["y_clean"]

  if clean_windows.shape[0] == 0:
    raise ValueError("All windows were rejected by the validation mask.")

  if subband_windows is not None:
    if subband_windows.shape[0] != windows.shape[0]:
      raise ValueError(
          f"subband_windows {subband_windows.shape} is not aligned with windows {windows.shape}"
      )
    clean_bands = subband_windows[qc["valid_mask"] == 1][:, :, qc["channels_kept_idx"], :]
    band_list = [clean_bands[:, k] for k in range(clean_bands.shape[1])]
  else:
    band_list = [
        butter_bandpass_filter(clean_windows, lowcut, highcut, fs=fs, order=4)
        for lowcut, highcut in subbands
    ]

  subband_features = []
  for X_sb in band_list:
    csp = MulticlassCSP(n_components=n_components)
    csp.fit(X_sb, y_clean)
    subband_features.append(csp.transform(X_sb))

  X_fbcsp = np.hstack(subband_features)
  channel_report = {
      "channels_kept": qc["channels_kept"],
      "channels_kept_idx": qc["channels_kept_idx"],
      "channels_dropped": qc["channels_dropped"],
      "channel_stats": qc["channel_stats"],
  }
  return X_fbcsp, y_clean, qc["valid_mask"], channel_report