"""Cut event-locked EEG windows and optional filter-bank versions.

The legacy condition order is preserved: 201 cross, 202 cue, then 101-105
stimulus windows. Bands can be filtered continuously before cutting or
separately per window to reproduce the older edge-effect behavior.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import butter, cheby1, filtfilt

STIM_CONDITIONS = (101, 102, 103, 104, 105)
CROSS_CONDITION = 201
CUE_CONDITION = 202

FILTER_MODES = ("continuous", "per_window")


# Session loading

def load_session(path: str | Path, scalp_channels: list[str], mark_channel: str = "MARK") -> dict:
  """Load one `.ebr` session as continuous EEG (scalp channels) + MARK."""
  from src.io.ebr_parser import load_ebr_file

  rec = load_ebr_file(path)
  data = rec["data"][0, :, 0, :]
  names = rec["channels"]
  missing = [c for c in scalp_channels if c not in names]
  if missing:
    raise ValueError(f"{path}: channels {missing} not in recording {names}")
  if mark_channel not in names:
    raise ValueError(f"{path}: no '{mark_channel}' channel; cannot locate events.")
  fs = float(rec["sampling_rate"])
  if fs <= 0:
    raise ValueError(f"{path}: invalid sampling_rate {fs} in header.")
  idx = [names.index(c) for c in scalp_channels]
  return {
      "eeg": data[idx, :],
      "mark": data[names.index(mark_channel), :],
      "fs": fs,
      "channels": list(scalp_channels),
  }


# Window indexing

@dataclass
class WindowIndex:
  starts: np.ndarray      # Start sample for each window.
  labels: np.ndarray      # Condition code for each window.
  trial_ids: np.ndarray   # Shared trial id for related windows.
  sub_idx: np.ndarray     # Window position within its trial.


def event_onsets(mark: np.ndarray, code: int) -> np.ndarray:
  """Samples where MARK switches to `code` (rising edge of the pulse)."""
  change = np.diff(mark, prepend=0) != 0
  return np.flatnonzero((mark == code) & change)


def build_window_index(
    mark: np.ndarray,
    fs: float,
    stim_conditions: tuple[int, ...] = STIM_CONDITIONS,
    cross_condition: int = CROSS_CONDITION,
    cue_condition: int | None = CUE_CONDITION,
    sub_window_sec: float = 1.0,
    stim_duration_sec: float = 5.0,
    cross_duration_sec: float = 2.0,
    cue_lookback_sec: float = 1.0,
) -> WindowIndex:
  """Locate every analysis window in a continuous recording (legacy rules)."""
  n_total = len(mark)
  win = int(round(sub_window_sec * fs))
  starts, labels, trials, subs = [], [], [], []
  trial = 0

  def add(trial_starts, code):
    nonlocal trial
    for s_list in trial_starts:
      for k, s in enumerate(s_list):
        starts.append(s), labels.append(code), trials.append(trial), subs.append(k)
      if s_list:
        trial += 1

  # Cross windows, clipped at the recording end.
  cross = []
  for c in event_onsets(mark, cross_condition):
    end = min(n_total, c + int(round(cross_duration_sec * fs)))
    cross.append([c + k * win for k in range((end - c) // win)])
  add(cross, cross_condition)

  # Cue window immediately before the cue onset.
  if cue_condition is not None:
    look = int(round(cue_lookback_sec * fs))
    add([[c - look] for c in event_onsets(mark, cue_condition) if c - look >= 0], cue_condition)

  # Keep complete stimulus windows only.
  n_sub = int(round(stim_duration_sec / sub_window_sec))
  for code in stim_conditions:
    stim = []
    for e in event_onsets(mark, code):
      if e + n_sub * win <= n_total:
        stim.append([e + k * win for k in range(n_sub)])
    add(stim, code)

  return WindowIndex(
      starts=np.asarray(starts, dtype=np.int64),
      labels=np.asarray(labels, dtype=np.int64),
      trial_ids=np.asarray(trials, dtype=np.int64),
      sub_idx=np.asarray(subs, dtype=np.int64),
  )


def cut_windows(signal: np.ndarray, starts: np.ndarray, n_samples: int) -> np.ndarray:
  """(..., n_channels, n_total) -> (n_windows, ..., n_channels, n_samples)."""
  return np.stack([signal[..., s:s + n_samples] for s in starts])


# Filter banks

@dataclass
class BandFilter:
  name: str
  low: float
  high: float
  b: np.ndarray
  a: np.ndarray
  kind: str


def fbcca_filter_bank(fs: float) -> list[BandFilter]:
  """Build the FBCCA Chebyshev-I sub-bands."""
  nyq = 0.5 * fs
  high = min(90.0, nyq - 1.0)
  bank = []
  for low in (6.0, 14.0, 22.0):
    b, a = cheby1(4, 3, [low / nyq, high / nyq], btype="bandpass", output="ba")
    bank.append(BandFilter(f"fbcca_{low:g}-{high:g}", low, high, b, a, "cheby1(4, 3dB)"))
  return bank


def fbcsp_filter_bank(fs: float, subbands: list[tuple[float, float]] | None = None) -> list[BandFilter]:
  """FBCSP sub-bands: Butterworth order 4. Identical to `butter_bandpass_filter`."""
  from src.features.ssvep.fbcsp_extraction import DEFAULT_SUBBANDS

  nyq = 0.5 * fs
  bank = []
  for low, high in (subbands or DEFAULT_SUBBANDS):
    b, a = butter(4, [low / nyq, min(high, nyq - 1.0) / nyq], btype="band")
    bank.append(BandFilter(f"fbcsp_{low:g}-{high:g}", low, high, b, a, "butter(4)"))
  return bank


def band_windows(
    broadband: np.ndarray,
    starts: np.ndarray,
    n_samples: int,
    bank: list[BandFilter],
    mode: str,
) -> np.ndarray:
  """Return band windows with shape (windows, bands, channels, samples)."""
  if mode not in FILTER_MODES:
    raise ValueError(f"mode must be one of {FILTER_MODES}, got {mode!r}")
  if mode == "continuous":
    per_band = [cut_windows(filtfilt(f.b, f.a, broadband, axis=-1), starts, n_samples) for f in bank]
  else:
    wins = cut_windows(broadband, starts, n_samples)
    per_band = [filtfilt(f.b, f.a, wins, axis=-1) for f in bank]
  return np.stack(per_band, axis=1)


def select_clean_bands(bands: np.ndarray, valid_mask: np.ndarray, channels_kept_idx: list[int]) -> np.ndarray:
  """Apply the quality gate's window mask and channel selection to band windows."""
  return bands[np.asarray(valid_mask) == 1][:, :, channels_kept_idx, :]
