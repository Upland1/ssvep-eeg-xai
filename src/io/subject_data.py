"""Load consistent windows and quality decisions for each subject."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

FULL_MONTAGE = ["PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"]
SUBJECT_DIR_PATTERN = re.compile(r"^S\d+$")
LEGACY_FS = 250.0
LEGACY_TRIAL_LENGTH = {201: 2, 202: 1, 101: 5, 102: 5, 103: 5, 104: 5, 105: 5}
QUALITY_MODES = ("shared", "shared-no-guard", "per-script")


def find_project_root(start: str | Path) -> Path:
  """Find the project root above `start`."""
  p = Path(start).resolve()
  for cand in [p, *p.parents]:
    if (cand / "configs" / "pipeline_config.yaml").exists():
      return cand
  raise FileNotFoundError(f"No configs/pipeline_config.yaml above {start}")


def discover_subjects(data_root: Path) -> list[str]:
  if not data_root.exists():
    return []
  return sorted(p.name for p in data_root.iterdir() if p.is_dir() and SUBJECT_DIR_PATTERN.match(p.name))


def resolve_channel_names(n_channels: int, full_montage: list[str] = FULL_MONTAGE) -> list[str] | None:
  """Return channel names for a full or legacy montage."""
  if n_channels == len(full_montage):
    return list(full_montage)
  if n_channels == len(full_montage) - 1:
    return full_montage[1:]
  return None


def is_built(sub_dir: Path) -> bool:
  return (sub_dir / "windows_meta.json").exists()


def load_subject(
    sub_dir: str | Path,
    conditions: list[int] | tuple[int, ...],
    quality_mode: str = "shared",
    bands: tuple[str, ...] = (),
    fs_override: float | None = None,
) -> dict:
  """Load one subject for the requested conditions."""
  if quality_mode not in QUALITY_MODES:
    raise ValueError(f"quality_mode must be one of {QUALITY_MODES}")
  sub_dir = Path(sub_dir)
  windows_all = np.load(sub_dir / "X_time_windows.npy", mmap_mode="r")
  y_all = np.load(sub_dir / "y_labels.npy")
  sel = np.isin(y_all, list(conditions))

  if not is_built(sub_dir):
    y = y_all[sel]
    windows = np.asarray(windows_all[sel] if windows_all.shape[0] == len(y_all) else windows_all[: len(y)])
    per_trial = {c: LEGACY_TRIAL_LENGTH.get(int(c), 5) for c in np.unique(y)}
    from src.preprocessing.dataset.cv_utils import build_trial_ids
    return {
        "windows": windows, "y": y, "trial_ids": build_trial_ids(y, per_trial), "bands": {},
        "fs": fs_override or LEGACY_FS, "channel_names": resolve_channel_names(windows.shape[1]),
        "quality": None, "built": False, "source": "legacy",
    }

  meta = json.loads((sub_dir / "windows_meta.json").read_text())
  q = meta.get("quality")
  out = {
      "windows": np.asarray(windows_all[sel]),
      "y": y_all[sel],
      "trial_ids": np.load(sub_dir / "trial_ids.npy")[sel],
      "bands": {b: np.asarray(np.load(sub_dir / f"X_bands_{b}.npy", mmap_mode="r")[sel]) for b in bands},
      "fs": fs_override or float(meta["fs"]),
      "channel_names": list(meta["channels"]),
      "quality": None,
      "built": True,
      "source": f"{meta['session_file']}, {meta['filter_mode']}, quality={quality_mode}",
      "meta": meta,
  }
  if quality_mode != "per-script":
    if q is None:
      raise ValueError(f"{sub_dir}: no quality decision saved; rebuild with scripts/build_windows.py")
    mask_file = "valid_mask.npy" if quality_mode == "shared" else "valid_mask_no_guard.npy"
    out["quality"] = {
        "valid_mask": np.load(sub_dir / mask_file)[sel],
        "channels_kept_idx": q["channels_kept_idx"],
        "channels_kept": q["channels_kept"],
        "channels_dropped": q["channels_dropped"],
    }
  return out