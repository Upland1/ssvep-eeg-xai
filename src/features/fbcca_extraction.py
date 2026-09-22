import numpy as np
from scipy.signal import cheby1, filtfilt
from sklearn.cross_decomposition import CCA
from src.preprocessing.quality_check import apply_artifact_quality_pipeline

TARGET_FREQS = np.array([24.0, 20.0, 15.0, 10.9091, 8.5714])


def generate_reference_signals(
    f_target: float, fs: float, n_samples: int, n_harmonics: int = 4
) -> np.ndarray:
  t = np.arange(n_samples) / fs
  templates = []
  for h in range(1, n_harmonics + 1):
    templates.append(np.sin(2 * np.pi * h * f_target * t))
    templates.append(np.cos(2 * np.pi * h * f_target * t))
  return np.array(templates)


def get_chebyshev_subbands(fs: float) -> list[tuple[np.ndarray, np.ndarray]]:
  nyq = 0.5 * fs
  highcut = min(90.0, nyq - 1.0)
  lowcuts = [6.0, 14.0, 22.0]
  filters = []
  for low in lowcuts:
    b, a = cheby1(
        4, 3, [low / nyq, highcut / nyq], btype="bandpass", output="ba"
    )
    filters.append((b, a))
  return filters


def extract_fbcca_features(
    windows: np.ndarray,
    y: np.ndarray,
    fs: float = 250.0,
    target_freqs: np.ndarray = TARGET_FREQS,
    n_harmonics: int = 3,
    channel_names: list[str] | None = None,
    vpp_limits: tuple[float, float] = (5.0, 120.0),
    std_limits: tuple[float, float] = (1.0, 35.0),
    channel_fail_fraction_thresh: float = 0.5,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
  """Validate windows (channel-level + window-level) and extract FBCCA scores.

  Runs the two-tier artifact-quality pipeline first: chronically bad
  channels are identified and dropped for this subject, then
  individual windows still out of range on the surviving channels are
  rejected. FBCCA correlation scores are computed only on the
  clean, channel-reduced data.
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
  )
  clean_windows, y_clean = qc["windows_clean"], qc["y_clean"]

  if clean_windows.shape[0] == 0:
    raise ValueError("All windows were rejected by the validation mask.")

  n_windows, n_channels, n_samples = clean_windows.shape
  n_targets = len(target_freqs)
  subband_filters = get_chebyshev_subbands(fs)
  n_subbands = len(subband_filters)
  weights = np.array([n**-1.25 + 0.25 for n in range(1, n_subbands + 1)])

  reference_signals = []
  for f_i in target_freqs:
    ref = generate_reference_signals(
        f_i, fs, n_samples, n_harmonics=n_harmonics
    )
    reference_signals.append(ref)

  cca = CCA(n_components=1)
  X_fbcca = np.zeros((n_windows, n_targets))

  for w in range(n_windows):
    win = clean_windows[w]
    scores = np.zeros(n_targets)

    for i, ref in enumerate(reference_signals):
      r_sub = np.zeros(n_subbands)
      for sb_idx, (b, a) in enumerate(subband_filters):
        win_sb = filtfilt(b, a, win, axis=-1)
        try:
          cca.fit(win_sb.T, ref.T)
          x_score, y_score = cca.transform(win_sb.T, ref.T)
          r_sub[sb_idx] = np.corrcoef(x_score[:, 0], y_score[:, 0])[0, 1]
        except Exception:
          r_sub[sb_idx] = 0.0

      scores[i] = np.sum(weights * (r_sub**2))

    X_fbcca[w, :] = scores

  channel_report = {
      "channels_kept": qc["channels_kept"],
      "channels_kept_idx": qc["channels_kept_idx"],
      "channels_dropped": qc["channels_dropped"],
      "channel_stats": qc["channel_stats"],
  }
  return X_fbcca, y_clean, qc["valid_mask"], channel_report