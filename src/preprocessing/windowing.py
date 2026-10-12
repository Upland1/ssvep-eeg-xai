"""Cut event-locked EEG windows and filter-bank bands."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import butter, cheby1, filtfilt

STIM_CONDITIONS = (101, 102, 103, 104, 105)
CROSS_CONDITION = 201
CUE_CONDITION = 202

FILTER_MODES = ("continuous", "per_window")


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


@dataclass
class WindowIndex:
  starts: np.ndarray      # window start samples
  labels: np.ndarray      # condition codes
  trial_ids: np.ndarray   # trial ids
  sub_idx: np.ndarray     # within-trial positions (0, 1, 2, ...)
  offsets: np.ndarray     # samples from the segment onset to the window start


def event_onsets(mark: np.ndarray, code: int) -> np.ndarray:
  """Samples where MARK switches to `code` (rising edge of the pulse)."""
  change = np.diff(mark, prepend=0) != 0
  return np.flatnonzero((mark == code) & change)


def segment_starts(onset: int, seg_len: int, win: int, step: int, n_total: int) -> list[int]:
  """Starts of every window that fits inside [onset, onset + seg_len) and the recording."""
  stop = min(onset + seg_len, n_total)
  if onset < 0 or stop - onset < win:
    return []
  return list(range(onset, stop - win + 1, step))


def windows_per_segment(seg_sec: float, window_sec: float, stride_sec: float, fs: float) -> int:
  """Number of windows that fit in a segment of `seg_sec` seconds."""
  seg, win, step = (int(round(x * fs)) for x in (seg_sec, window_sec, stride_sec))
  return 0 if seg < win else (seg - win) // step + 1


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
    stride_sec: float | None = None,
) -> WindowIndex:
  """Locate every analysis window in a continuous recording.

  Each condition is a segment: cross = [onset, onset + cross_duration),
  cue = [cue onset - lookback, cue onset), stimulus = [onset, onset + stim_duration).
  Windows of `sub_window_sec` are placed every `stride_sec` and must fit entirely
  inside their segment. `stride_sec=None` uses the window length (no overlap),
  which reproduces the legacy windows exactly. A stimulus trial is kept only if
  the whole stimulus segment is inside the recording (legacy rule).
  """
  if sub_window_sec <= 0:
    raise ValueError(f"window length must be > 0, got {sub_window_sec}")
  stride_sec = sub_window_sec if stride_sec is None else stride_sec
  if stride_sec <= 0:
    raise ValueError(f"stride must be > 0, got {stride_sec}")

  n_total = len(mark)
  win = int(round(sub_window_sec * fs))
  step = max(1, int(round(stride_sec * fs)))
  starts, labels, trials, subs, offs = [], [], [], [], []
  trial = 0

  def add(segments, code):
    nonlocal trial
    for onset, s_list in segments:
      for k, s in enumerate(s_list):
        starts.append(s), labels.append(code), trials.append(trial), subs.append(k), offs.append(s - onset)
      if s_list:
        trial += 1

  # Cross (rest) windows.
  cross_len = int(round(cross_duration_sec * fs))
  add([(c, segment_starts(c, cross_len, win, step, n_total)) for c in event_onsets(mark, cross_condition)],
      cross_condition)

  # Cue lookback windows.
  if cue_condition is not None:
    look = int(round(cue_lookback_sec * fs))
    add([(c - look, segment_starts(c - look, look, win, step, n_total))
         for c in event_onsets(mark, cue_condition)], cue_condition)

  # Stimulus windows (complete trials only).
  stim_len = int(round(stim_duration_sec * fs))
  for code in stim_conditions:
    add([(e, segment_starts(e, stim_len, win, step, n_total))
         for e in event_onsets(mark, code) if e + stim_len <= n_total], code)

  return WindowIndex(
      starts=np.asarray(starts, dtype=np.int64),
      labels=np.asarray(labels, dtype=np.int64),
      trial_ids=np.asarray(trials, dtype=np.int64),
      sub_idx=np.asarray(subs, dtype=np.int64),
      offsets=np.asarray(offs, dtype=np.int64),
  )


def cut_windows(signal: np.ndarray, starts: np.ndarray, n_samples: int) -> np.ndarray:
  """(..., n_channels, n_total) -> (n_windows, ..., n_channels, n_samples)."""
  return np.stack([signal[..., s:s + n_samples] for s in starts])


@dataclass
class BandFilter:
  name: str
  low: float
  high: float
  b: np.ndarray
  a: np.ndarray
  kind: str


def fbcca_filter_bank(fs: float) -> list[BandFilter]:
  """Return the FBCCA Chebyshev-I filter bank."""
  nyq = 0.5 * fs
  high = min(90.0, nyq - 1.0)
  bank = []
  for low in (6.0, 14.0, 22.0):
    b, a = cheby1(4, 3, [low / nyq, high / nyq], btype="bandpass", output="ba")
    bank.append(BandFilter(f"fbcca_{low:g}-{high:g}", low, high, b, a, "cheby1(4, 3dB)"))
  return bank


def fbcsp_filter_bank(fs: float, subbands: list[tuple[float, float]] | None = None) -> list[BandFilter]:
  """Return the FBCSP Butterworth filter bank."""
  from src.features.ssvep.fbcsp_extraction import DEFAULT_SUBBANDS

  nyq = 0.5 * fs
  bank = []
  for low, high in (subbands or DEFAULT_SUBBANDS):
    b, a = butter(4, [low / nyq, min(high, nyq - 1.0) / nyq], btype="band")
    bank.append(BandFilter(f"fbcsp_{low:g}-{high:g}", low, high, b, a, "butter(4)"))
  return bank


STIM_FREQS = {101: 24.0, 102: 20.0, 103: 15.0, 104: 10.9091, 105: 8.5714}


def harmonic_centers(freqs: dict[int, float] = STIM_FREQS, n_harmonics: int = 3) -> list[tuple[str, float]]:
  """Return harmonic center frequencies."""
  return [(f"h{k}_{f:.2f}", k * f) for f in freqs.values() for k in range(1, n_harmonics + 1)]


def intermediate_centers(freqs: dict[int, float] = STIM_FREQS) -> list[tuple[str, float]]:
  """Return centers between neighboring stimulus frequencies."""
  s = sorted(freqs.values())
  return [(f"mid_{(a + b) / 2:.2f}", (a + b) / 2) for a, b in zip(s[:-1], s[1:])]


def narrow_filter_bank(
    fs: float,
    centers: list[tuple[str, float]],
    half_width: float = 1.0,
    order: int = 2,
    max_hz: float | None = None,
    exclude_hz: list[float] = (),
) -> tuple[list[BandFilter], list[dict]]:
  """Build narrow Butterworth bands around the requested centers."""
  nyq = 0.5 * fs
  top = min(max_hz or nyq, nyq - 1.0)
  bank, skipped = [], []
  for name, fc in centers:
    low, high = fc - half_width, fc + half_width
    why = None
    if high >= top:
      why = f"above {top:g} Hz"
    elif any(low <= x <= high for x in exclude_hz):
      why = "contains the mains notch"
    if why:
      skipped.append({"name": name, "center_hz": round(fc, 3), "reason": why})
      continue
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    bank.append(BandFilter(name, round(low, 3), round(high, 3), b, a, f"butter({order})"))
  return bank, skipped


def band_overlaps(bank: list[BandFilter]) -> list[tuple[str, str, float]]:
  """Return overlapping band pairs and their overlap in Hz."""
  out = []
  for i, a in enumerate(bank):
    for b in bank[i + 1:]:
      ov = min(a.high, b.high) - max(a.low, b.low)
      if ov > 0:
        out.append((a.name, b.name, round(ov, 2)))
  return out


def band_windows(
    broadband: np.ndarray,
    starts: np.ndarray,
    n_samples: int,
    bank: list[BandFilter],
    mode: str,
) -> np.ndarray:
  """Return sub-band windows using the selected filtering mode."""
  if mode not in FILTER_MODES:
    raise ValueError(f"mode must be one of {FILTER_MODES}, got {mode!r}")
  if mode == "continuous":
    per_band = [cut_windows(filtfilt(f.b, f.a, broadband, axis=-1), starts, n_samples) for f in bank]
  else:
    wins = cut_windows(broadband, starts, n_samples)
    per_band = [filtfilt(f.b, f.a, wins, axis=-1) for f in bank]
  return np.stack(per_band, axis=1)


def select_clean_bands(bands: np.ndarray, valid_mask: np.ndarray, channels_kept_idx: list[int]) -> np.ndarray:
  """Apply the quality mask and channel selection to band windows."""
  return bands[np.asarray(valid_mask) == 1][:, :, channels_kept_idx, :]