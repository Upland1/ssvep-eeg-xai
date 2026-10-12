"""Build analysis windows and filter-bank bands from raw `.ebr` files.

Filtering is continuous by default; use `--filter-mode per_window` for legacy
behavior. Quality can use filtered, raw, or both windows via
`--quality-source`. Use `--window-sec` and `--stride-sec` for custom windows.

Usage (from the repo root):
  python build_windows.py --filter-mode continuous
  python build_windows.py --filter-mode per_window
  python build_windows.py --subjects S03 S04 --window-sec 1 2
  python build_windows.py --quality-source raw --banks fbcca fbcsp
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from scipy.signal import filtfilt, iirfilter, iirnotch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.signal.filters import apply_iir_bandpass  # noqa: E402
from src.preprocessing.window_quality import QUALITY_SOURCES, REASON_NAMES, subject_quality_gate  # noqa: E402
from src.preprocessing.windowing import (  # noqa: E402
    FILTER_MODES,
    STIM_CONDITIONS,
    CROSS_CONDITION,
    CUE_CONDITION,
    band_windows,
    build_window_index,
    cut_windows,
    fbcca_filter_bank,
    band_overlaps,
    fbcsp_filter_bank,
    harmonic_centers,
    intermediate_centers,
    load_session,
    narrow_filter_bank,
    windows_per_segment,
)

SUBJECT_DIR = re.compile(r"^S\d+$")
BANK_NAMES = ("fbcca", "fbcsp", "harmonic", "intermediate")


def build_filter_banks(fs: float, cfg: dict, highcut: float, notch_hz: float | None) -> tuple[dict, list[dict]]:
  """Build the configured FBCCA, FBCSP, and optional narrow bands."""
  banks = {"fbcca": fbcca_filter_bank(fs), "fbcsp": fbcsp_filter_bank(fs)}
  fb = cfg.get("filter_bank") or {}
  skipped = []
  excl = [notch_hz] if notch_hz else []
  h = fb.get("harmonics") or {}
  if h.get("enabled", False):
    bank, sk = narrow_filter_bank(fs, harmonic_centers(n_harmonics=h.get("n_harmonics", 3)),
                                  h.get("half_width_hz", 1.0), h.get("order", 2), highcut, excl)
    banks["harmonic"] = bank
    skipped += sk
  m = fb.get("intermediate") or {}
  if m.get("enabled", False):
    bank, sk = narrow_filter_bank(fs, intermediate_centers(), m.get("half_width_hz", 1.0),
                                  m.get("order", 2), highcut, excl)
    banks["intermediate"] = bank
    skipped += sk
  return banks, skipped


def build_subject(raw_dir: Path, out_dir: Path, session: str, mode: str, cfg: dict,
                  guard_override: float | None = None, source_override: str | None = None,
                  window_sec: float | None = None, stride_sec: float | None = None,
                  save_banks: list[str] | None = None) -> dict:
  pre = cfg["preprocessing"]
  highcut = float(pre["highcut_hz"])
  notch_hz = pre.get("notch_hz") or None
  base = pre["baseline"]
  sess = load_session(raw_dir / f"{session}.ebr", cfg["channels"]["scalp"], cfg["channels"]["mark_channel"])
  fs = sess["fs"]
  window_sec = float(window_sec or pre["sub_window_sec"])
  stride_sec = float(stride_sec or window_sec)
  n_samples = int(round(window_sec * fs))

  # Filter the continuous signal and optionally remove mains noise.
  broadband = apply_iir_bandpass(sess["eeg"], fs, pre["lowcut_hz"], highcut, pre["filter_order"])
  notch_ba = None
  if notch_hz:
    notch_ba = iirnotch(notch_hz, pre.get("notch_q", 30.0), fs)
    broadband = filtfilt(*notch_ba, broadband, axis=-1)

  idx = build_window_index(
      sess["mark"], fs,
      cross_condition=base["cross_condition"],
      cue_condition=base["cue_condition"],
      sub_window_sec=window_sec,
      stride_sec=stride_sec,
      stim_duration_sec=base["stimulus_duration_sec"],
      cross_duration_sec=base.get("pre_stim_cross_sec", 2.0),
      cue_lookback_sec=base["cue_lookback_sec"],
  )
  X = cut_windows(broadband, idx.starts, n_samples)
  X_raw = cut_windows(sess["eeg"], idx.starts, n_samples)

  banks, skipped_bands = build_filter_banks(fs, cfg, highcut, notch_hz)

  # Build one quality decision for the subject.
  nyq = 0.5 * fs
  b_bb, a_bb = iirfilter(pre["filter_order"], [pre["lowcut_hz"] / nyq, highcut / nyq],
                         btype="bandpass", ftype="butter")
  continuous_filters = [(b_bb, a_bb)] + ([notch_ba] if notch_ba is not None else [])
  if mode == "continuous":
    continuous_filters += [(f.b, f.a) for bank in banks.values() for f in bank]
  q_params = dict(cfg.get("quality") or {})
  if guard_override is not None:
    q_params["guard_sec"] = guard_override
  if source_override is not None:
    q_params["source"] = source_override
  quality = subject_quality_gate(X, idx.starts, broadband, fs, sess["channels"], continuous_filters, q_params,
                                 raw_windows=X_raw, raw=sess["eeg"])

  out_dir.mkdir(parents=True, exist_ok=True)
  np.save(out_dir / "X_time_windows.npy", X)
  np.save(out_dir / "X_raw_windows.npy", X_raw)
  np.save(out_dir / "y_labels.npy", idx.labels)
  np.save(out_dir / "trial_ids.npy", idx.trial_ids)
  np.save(out_dir / "window_starts.npy", idx.starts)
  np.save(out_dir / "sub_idx.npy", idx.sub_idx)
  np.save(out_dir / "window_offsets.npy", idx.offsets)
  np.save(out_dir / "valid_mask.npy", quality["valid_mask"])
  np.save(out_dir / "valid_mask_no_guard.npy", quality["valid_mask_no_guard"])
  np.save(out_dir / "reject_reason.npy", quality["reason"])
  # All banks still feed the guard above; only the selected ones are saved.
  for name, bank in banks.items():
    if save_banks is not None and name not in save_banks:
      continue
    bands = band_windows(broadband, idx.starts, n_samples, bank, mode)
    np.save(out_dir / f"X_bands_{name}.npy", bands)

  labels, counts = np.unique(idx.labels, return_counts=True)
  reject_by_label = {
      str(int(c)): {REASON_NAMES[r]: int(np.sum((idx.labels == c) & (quality["reason"] == r))) for r in (1, 2, 3)}
      for c in labels
  }
  meta = {
      "created": datetime.now().isoformat(timespec="seconds"),
      "session_file": f"{session}.ebr",
      "fs": fs,
      "channels": sess["channels"],
      "n_samples_per_window": n_samples,
      "window_sec": window_sec,
      "stride_sec": stride_sec,
      "windows_per_trial": {
          "stimulus": windows_per_segment(base["stimulus_duration_sec"], window_sec, stride_sec, fs),
          "cross": windows_per_segment(base.get("pre_stim_cross_sec", 2.0), window_sec, stride_sec, fs),
          "cue": windows_per_segment(base["cue_lookback_sec"], window_sec, stride_sec, fs),
      },
      "banks_saved": sorted(banks) if save_banks is None else [b for b in sorted(banks) if b in save_banks],
      "filter_mode": mode,
      "broadband": {"low": pre["lowcut_hz"], "high": highcut, "order": pre["filter_order"],
                    "notch_hz": notch_hz, "applied": "continuous, zero-phase"},
      "bands": {k: [{"name": f.name, "low": f.low, "high": f.high, "type": f.kind} for f in v]
                for k, v in banks.items()},
      "bands_skipped": skipped_bands,
      "band_overlaps": [list(o) for o in band_overlaps([f for k in ("harmonic", "intermediate")
                                                        for f in banks.get(k, [])])],
      "label_counts": {str(int(k)): int(v) for k, v in zip(labels, counts)},
      "n_trials": int(idx.trial_ids.max() + 1) if len(idx.trial_ids) else 0,
      "quality": {
          "channels_kept": quality["channels_kept"],
          "channels_kept_idx": quality["channels_kept_idx"],
          "channels_dropped": quality["channels_dropped"],
          "channel_fail_fraction": quality["channel_fail_fraction"],
          "source": quality["params"]["source"],
          "guard_sec": quality["guard_sec"],
          "artifact_fraction_of_recording": quality["artifact_fraction"],
          "n_valid": int(quality["valid_mask"].sum()),
          "n_valid_no_guard": int(quality["valid_mask_no_guard"].sum()),
          "rejected_by_label": reject_by_label,
          "params": quality["params"],
      },
  }
  (out_dir / "windows_meta.json").write_text(json.dumps(meta, indent=2))
  return meta


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--filter-mode", choices=FILTER_MODES, default="continuous")
  ap.add_argument("--subjects", nargs="*", help="e.g. S01 S03 (default: every S<digits> folder in data/raw)")
  ap.add_argument("--session", default="OO", help="Session file stem inside each subject folder (default: OO)")
  ap.add_argument("--raw-root", type=Path, default=None)
  ap.add_argument("--out-root", type=Path, default=None,
                  help="Default: data/processed_<filter-mode>")
  ap.add_argument("--guard-sec", type=float, default=None,
                  help="Override the guard margin in seconds (default: config, 'auto' = filter ringing time)")
  ap.add_argument("--quality-source", choices=QUALITY_SOURCES, default=None,
                  help="Signal the quality limits are checked on (default: config quality.source)")
  ap.add_argument("--window-sec", type=float, nargs="+", default=None,
                  help="Window length(s) in seconds, e.g. 0.5 1 2 4 (default: config sub_window_sec)")
  ap.add_argument("--stride-sec", type=float, default=None,
                  help="Step between window starts in seconds (default: window length, no overlap)")
  ap.add_argument("--banks", nargs="*", choices=BANK_NAMES, default=None,
                  help="Filter banks to save (default: all; pass with no names to save none)")
  ap.add_argument("--config", type=Path, default=ROOT / "configs" / "pipeline_config.yaml")
  args = ap.parse_args()

  cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
  raw_root = args.raw_root or ROOT / cfg["paths"]["data_raw_dir"]
  base_out = args.out_root or ROOT / "data" / f"processed_{args.filter_mode}"
  subjects = args.subjects or sorted(p.name for p in raw_root.iterdir() if p.is_dir() and SUBJECT_DIR.match(p.name))

  pre = cfg["preprocessing"]
  tagged = args.window_sec is not None or args.stride_sec is not None
  for win in (args.window_sec or [float(pre["sub_window_sec"])]):
    stride = args.stride_sec or win
    out_root = base_out / f"w{win:g}_s{stride:g}" if tagged else base_out
    build_all(args, cfg, raw_root, out_root, subjects, win, stride)


def build_all(args, cfg: dict, raw_root: Path, out_root: Path, subjects: list[str], win: float, stride: float):
  """Build every subject for one window length."""
  pre = cfg["preprocessing"]
  print(f"\nBuilding windows | {win:g} s every {stride:g} s | mode={args.filter_mode} | "
        f"session={args.session} | out={out_root}")
  print(f"Broadband {pre['lowcut_hz']}-{pre['highcut_hz']} Hz | notch: {pre.get('notch_hz') or 'none'} | "
        f"quality source: {args.quality_source or (cfg.get('quality') or {}).get('source', 'filtered')}")
  shown = False
  for sub in subjects:
    try:
      meta = build_subject(raw_root / sub, out_root / sub, args.session, args.filter_mode, cfg, args.guard_sec,
                           args.quality_source, win, stride, args.banks)
    except (FileNotFoundError, ValueError) as exc:
      print(f"  [-] {sub}: skipped ({exc})")
      continue
    if not shown:
      shown = True
      extra = {k: len(v) for k, v in meta["bands"].items()}
      print(f"  bands per bank: {extra}")
      for sk in meta["bands_skipped"]:
        print(f"  [skip] {sk['name']} ({sk['center_hz']} Hz): {sk['reason']}")
      for a, b, ov in meta["band_overlaps"]:
        print(f"  [overlap] {a} and {b} share {ov} Hz")
      wpt = meta["windows_per_trial"]
      print(f"  windows per trial: stimulus {wpt['stimulus']}, cross {wpt['cross']}, cue {wpt['cue']}")
      for name, code in (("cross (rest)", CROSS_CONDITION), ("cue", CUE_CONDITION)):
        if str(code) not in meta["label_counts"]:
          print(f"  [warn] no {name} windows: the {win:g} s window is longer than that segment")
    q = meta["quality"]
    stim = [str(c) for c in STIM_CONDITIONS]
    own = sum(q["rejected_by_label"][c][REASON_NAMES[1]] for c in stim if c in q["rejected_by_label"])
    near = sum(q["rejected_by_label"][c][REASON_NAMES[2]] for c in stim if c in q["rejected_by_label"])
    raw = sum(q["rejected_by_label"][c][REASON_NAMES[3]] for c in stim if c in q["rejected_by_label"])
    print(f"  [+] {sub}: fs={meta['fs']:g} Hz | trials={meta['n_trials']} | "
          f"channels kept {len(q['channels_kept'])}/{len(meta['channels'])} (dropped: {q['channels_dropped'] or 'none'}) | "
          f"guard {q['guard_sec']:.2f} s | stimulus windows rejected: {own} own gate + {raw} raw gate + {near} near artifact | "
          f"valid {q['n_valid']}/{sum(meta['label_counts'].values())}")


if __name__ == "__main__":
  main()