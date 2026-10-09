"""Apply one quality gate to all windows from a subject."""
from __future__ import annotations

import numpy as np
from scipy.signal import filtfilt

from src.preprocessing.signal.quality_check import identify_noisy_channels, validate_eeg_windows

REASON_VALID = 0
REASON_OWN_GATE = 1
REASON_NEAR_ARTIFACT = 2
REASON_NAMES = {0: "valid", 1: "failed own Vpp/std gate", 2: "within guard of an artifact"}

DEFAULT_QUALITY = {
    "vpp_range_uv": [5.0, 120.0],
    "std_range_uv": [1.0, 35.0],
    "channel_fail_fraction": 0.5,
    "artifact_block_sec": 0.25,
    "guard_sec": "auto",
    "guard_rel_amplitude": 0.01,
}


def filter_ringing_sec(filters: list[tuple[np.ndarray, np.ndarray]], fs: float, rel_amplitude: float = 0.01) -> float:
  """Return the longest ringing time across the filters."""
  n = int(20 * fs)
  imp = np.zeros(n)
  imp[n // 2] = 1.0
  longest = 0
  for b, a in filters:
    h = np.abs(filtfilt(b, a, imp))
    above = np.flatnonzero(h > rel_amplitude * h.max())
    longest = max(longest, int(above.max() - n // 2))
  return longest / fs


def artifact_sample_mask(
    broadband: np.ndarray, fs: float, channels_idx: list[int], vpp_max: float, block_sec: float = 0.25
) -> np.ndarray:
  """Return a mask for artifact blocks in the recording."""
  blk = max(1, int(round(block_sec * fs)))
  n_total = broadband.shape[1]
  n_blocks = n_total // blk
  x = broadband[channels_idx, : n_blocks * blk].reshape(len(channels_idx), n_blocks, blk)
  bad_blocks = (np.ptp(x, axis=2) > vpp_max).any(axis=0)
  mask = np.zeros(n_total, dtype=bool)
  mask[: n_blocks * blk] = np.repeat(bad_blocks, blk)
  tail = broadband[channels_idx, n_blocks * blk:]
  if tail.shape[1] > 1 and (np.ptp(tail, axis=1) > vpp_max).any():
    mask[n_blocks * blk:] = True
  return mask


def dilate(mask: np.ndarray, n: int) -> np.ndarray:
  """Widen each True region by `n` samples on both sides."""
  if n <= 0:
    return mask.copy()
  idx = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
  out = np.zeros_like(mask)
  for start, stop in zip(idx[0::2], idx[1::2]):
    out[max(0, start - n): min(len(mask), stop + n)] = True
  return out


def subject_quality_gate(
    windows: np.ndarray,
    starts: np.ndarray,
    broadband: np.ndarray,
    fs: float,
    channel_names: list[str],
    filters: list[tuple[np.ndarray, np.ndarray]],
    params: dict | None = None,
) -> dict:
  """Return one quality decision for the subject's windows."""
  p = {**DEFAULT_QUALITY, **(params or {})}
  vpp_min, vpp_max = p["vpp_range_uv"]
  std_min, std_max = p["std_range_uv"]
  n_samples = windows.shape[-1]

  # Drop channels that fail too often.
  noisy_idx, stats = identify_noisy_channels(
      windows, channel_names=channel_names, vpp_min=vpp_min, vpp_max=vpp_max,
      std_min=std_min, std_max=std_max, fail_fraction_thresh=p["channel_fail_fraction"],
  )
  kept_idx = [i for i in range(windows.shape[1]) if i not in set(noisy_idx)]
  if not kept_idx:
    raise ValueError("All channels were flagged as noisy; nothing left to keep.")

  # Check each window on the kept channels.
  own_ok = validate_eeg_windows(windows[:, kept_idx], vpp_min, vpp_max, std_min, std_max).astype(bool)

  # Reject windows near artifacts in the continuous signal.
  if p["guard_sec"] in ("auto", None):
    guard_sec = filter_ringing_sec(filters, fs, p["guard_rel_amplitude"])
  else:
    guard_sec = float(p["guard_sec"])
  artifact = artifact_sample_mask(broadband, fs, kept_idx, vpp_max, p["artifact_block_sec"])
  widened = dilate(artifact, int(round(guard_sec * fs)))
  near = np.array([widened[s:s + n_samples].any() for s in starts], dtype=bool)

  reason = np.full(len(starts), REASON_VALID, dtype=np.int8)
  reason[own_ok & near] = REASON_NEAR_ARTIFACT
  reason[~own_ok] = REASON_OWN_GATE

  return {
      "valid_mask": (reason == REASON_VALID).astype(np.int8),
      "valid_mask_no_guard": own_ok.astype(np.int8),
      "reason": reason,
      "channels_kept_idx": kept_idx,
      "channels_kept": [channel_names[i] for i in kept_idx],
      "channels_dropped": [channel_names[i] for i in noisy_idx],
      "channel_fail_fraction": {channel_names[i]: round(float(f), 3) for i, f in enumerate(stats["fail_fraction"])},
      "guard_sec": round(guard_sec, 3),
      "artifact_fraction": round(float(artifact.mean()), 4),
      "params": p,
  }