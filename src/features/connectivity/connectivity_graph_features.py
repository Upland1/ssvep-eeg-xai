"""Per-window graph features for SSVEP classification.

Each window is converted to a small graph:
- nodes: per-channel power features at the 5 stimulation frequencies and
  their second harmonics,
- edges: pairwise coherence and phase features in each stimulation band.

Coherence is estimated from a few Welch segments inside the window. Phase is
computed from the full-window cross-spectrum and kept as a unit phasor. This
keeps the feature set compact while preserving frequency-specific phase
structure.
"""

import numpy as np
from scipy.signal import get_window

# Class order 101..105 (matches fbcca_extraction.TARGET_FREQS).
TARGET_FREQS = np.array([24.0, 20.0, 15.0, 10.9091, 8.5714])
HALF_WIDTH_HZ = 1.0


def _band_bins(freqs: np.ndarray, f0: float, half_width: float) -> np.ndarray:
  """FFT bins within +/-half_width of f0 (nearest bin if the band holds none)."""
  idx = np.flatnonzero(np.abs(freqs - f0) <= half_width)
  return idx if idx.size else np.array([int(np.argmin(np.abs(freqs - f0)))])


def _check(windows: np.ndarray) -> None:
  if windows.ndim != 3:
    raise ValueError("windows must have shape (n_windows, n_channels, n_samples)")


def node_band_power(
    windows: np.ndarray,
    fs: float = 250.0,
    target_freqs: np.ndarray = TARGET_FREQS,
    harmonics: tuple[int, ...] = (1, 2),
    half_width: float = HALF_WIDTH_HZ,
) -> np.ndarray:
  """log10 band power per channel at each target frequency and harmonic.

  Returns (n_windows, n_channels, len(harmonics) * len(target_freqs)),
  harmonic-major: [h1: 24, 20, 15, 10.9, 8.6 | h2: 48, 40, 30, 21.8, 17.1].
  """
  _check(windows)
  n, _, t = windows.shape
  x = windows - windows.mean(axis=-1, keepdims=True)
  w = get_window("hann", t)
  spec = np.fft.rfft(x * w, axis=-1)
  psd = np.abs(spec) ** 2 / (fs * np.sum(w ** 2))
  freqs = np.fft.rfftfreq(t, 1.0 / fs)
  feats = []
  for h in harmonics:
    for f0 in target_freqs:
      idx = _band_bins(freqs, h * f0, half_width)
      feats.append(psd[:, :, idx].mean(axis=-1))
  return np.log10(np.stack(feats, axis=-1) + 1e-12)


def edge_coherence(
    windows: np.ndarray,
    fs: float = 250.0,
    target_freqs: np.ndarray = TARGET_FREQS,
    nperseg: int = 128,
    step: int = 64,
    half_width: float = HALF_WIDTH_HZ,
) -> np.ndarray:
  """Per-window magnitude-squared coherence between all channel pairs.

  Welch inside the window (Hann, 50% overlap -> 3 segments for 256 samples),
  averaged over the bins of each target band. Returns
  (n_windows, n_channels, n_channels, len(target_freqs)); diagonal = 1.
  """
  _check(windows)
  n, c, t = windows.shape
  nperseg = min(nperseg, t)
  starts = np.arange(0, t - nperseg + 1, step)
  segs = np.stack([windows[:, :, s:s + nperseg] for s in starts], axis=2)  # (n, C, S, L)
  segs = segs - segs.mean(axis=-1, keepdims=True)
  w = get_window("hann", nperseg)
  spec = np.fft.rfft(segs * w, axis=-1)                                     # (n, C, S, F)
  freqs = np.fft.rfftfreq(nperseg, 1.0 / fs)
  sxy = np.einsum("ncsf,ndsf->ncdf", spec, spec.conj())
  sxx = np.einsum("ncsf,ncsf->ncf", spec, spec.conj()).real
  coh = np.abs(sxy) ** 2 / (sxx[:, :, None, :] * sxx[:, None, :, :] + 1e-20)
  return np.stack(
      [coh[..., _band_bins(freqs, f0, half_width)].mean(axis=-1) for f0 in target_freqs], axis=-1
  )


def edge_phase(
    windows: np.ndarray,
    fs: float = 250.0,
    target_freqs: np.ndarray = TARGET_FREQS,
    half_width: float = HALF_WIDTH_HZ,
) -> np.ndarray:
  """Unit phasor of the inter-channel phase difference at each target band.

  Returns (n_windows, n_channels, n_channels, len(target_freqs), 2) holding
  (cos, sin) of phase(channel_i) - phase(channel_j), from the cross-spectrum
  summed over the band's bins of the full-window FFT.
  """
  _check(windows)
  n, c, t = windows.shape
  x = windows - windows.mean(axis=-1, keepdims=True)
  w = get_window("hann", t)
  spec = np.fft.rfft(x * w, axis=-1)
  freqs = np.fft.rfftfreq(t, 1.0 / fs)
  out = []
  for f0 in target_freqs:
    idx = _band_bins(freqs, f0, half_width)
    cxy = np.einsum("ncf,ndf->ncd", spec[..., idx], spec[..., idx].conj())
    ph = cxy / (np.abs(cxy) + 1e-20)
    out.append(np.stack([ph.real, ph.imag], axis=-1))
  return np.stack(out, axis=3)


def upper_tri_flat(edges: np.ndarray) -> np.ndarray:
  """(n, C, C, ...) -> (n, n_pairs * prod(rest)): unique channel pairs only."""
  iu, ju = np.triu_indices(edges.shape[1], k=1)
  return edges[:, iu, ju].reshape(edges.shape[0], -1)


def build_graph_features(
    windows: np.ndarray, fs: float = 250.0, target_freqs: np.ndarray = TARGET_FREQS
) -> dict[str, np.ndarray]:
  """All per-window graph features for one subject's clean windows."""
  return {
      "node": node_band_power(windows, fs, target_freqs),
      "edge_coh": edge_coherence(windows, fs, target_freqs),
      "edge_phase": edge_phase(windows, fs, target_freqs),
  }