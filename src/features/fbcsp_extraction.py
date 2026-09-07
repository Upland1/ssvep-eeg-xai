import numpy as np
from scipy.linalg import eigh
from scipy.signal import butter, filtfilt
from src.preprocessing.quality_check import filter_dataset, validate_eeg_windows

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
    vpp_limits: tuple[float, float] = (0.5, 120.0),
    std_limits: tuple[float, float] = (0.1, 35.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Validate windows and extract FBCSP features.

  Returns: (X_fbcsp, y_clean, valid_mask)
  """
  # 1. Validation gate
  valid_mask = validate_eeg_windows(
      windows,
      vpp_min=vpp_limits[0],
      vpp_max=vpp_limits[1],
      std_min=std_limits[0],
      std_max=std_limits[1],
  )
  clean_windows, y_clean = filter_dataset(windows, y, valid_mask)

  if clean_windows.shape[0] == 0:
    raise ValueError("All windows were rejected by the validation mask.")

  subband_features = []
  for lowcut, highcut in subbands:
    X_sb = butter_bandpass_filter(clean_windows, lowcut, highcut, fs=fs, order=4)
    csp = MulticlassCSP(n_components=n_components)
    csp.fit(X_sb, y_clean)
    subband_features.append(csp.transform(X_sb))

  X_fbcsp = np.hstack(subband_features)
  return X_fbcsp, y_clean, valid_mask