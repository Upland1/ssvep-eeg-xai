"""Run a causal, pseudo-online SSVEP detector on an unseen recording.

Train on one run and replay another in ``--stride-sec`` chunks. Filtering
carries state between chunks, while windows are checked on a fixed clock.
Ignore windows before ``--skip-sec`` and decide when averaged probabilities
reach ``--threshold``; otherwise use the final class as a forced decision.

The test MARK channel is used only for scoring. The script also reports false
activations during the 5-second no-stimulation period before each target.
``--rest-class`` adds a sixth no-stimulation class and reports no target when
the threshold is not reached.

Before training, the first ``--check-sec`` seconds of the test stream are
checked for bad channels. If validation accuracy is below ``--min-val-acc``,
the model is retrained with ``--fallback-window-sec``; this choice uses only
the training run.

Use matching run types: OO/mOO for on-off stimulation and C/mC for
checkerboard stimulation, whose response is expected at twice the frequency.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, tf2sos

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_nn_sweep import _predict, fit_network  # noqa: E402
from benchmark_window_sweep import PROJECT_ROOT, STIM_CONDITIONS, _standardize  # noqa: E402

from src.preprocessing.signal.quality_check import identify_noisy_channels, validate_eeg_windows  # noqa: E402
from src.preprocessing.windowing import (  # noqa: E402
    CROSS_CONDITION, build_window_index, cut_windows, event_onsets, load_session,
)

N_CLASSES = len(STIM_CONDITIONS)
REST_SEC = 5.0   # fixation (2 s) + target presentation (2 s) + preparation (1 s)


class CausalFilter:
  """Apply band-pass and optional notch filtering while preserving state."""

  def __init__(self, fs: float, low: float, high: float, order: int, notch_hz: float | None, notch_q: float):
    sos = butter(order, [low, high], btype="band", fs=fs, output="sos")
    if notch_hz:
      sos = np.vstack([sos, tf2sos(*iirnotch(notch_hz, notch_q, fs))])
    self.sos, self.zi = sos, None

  def process(self, chunk: np.ndarray) -> np.ndarray:
    """Filter a (channels, samples) chunk."""
    if self.zi is None:
      self.zi = sosfilt_zi(self.sos)[:, None, :] * chunk[None, :, :1]
    out, self.zi = sosfilt(self.sos, chunk, axis=-1, zi=self.zi)
    return out


def make_filter(cfg: dict, fs: float) -> CausalFilter:
  pre = cfg["preprocessing"]
  return CausalFilter(fs, pre["lowcut_hz"], pre["highcut_hz"], pre["filter_order"],
                      pre.get("notch_hz") or None, pre.get("notch_q", 30.0))


def quality_limits(cfg: dict) -> dict:
  q = cfg.get("quality") or {}
  vpp, std = q.get("vpp_range_uv", [5.0, 120.0]), q.get("std_range_uv", [1.0, 35.0])
  return dict(vpp_min=vpp[0], vpp_max=vpp[1], std_min=std[0], std_max=std[1])


def classes(args) -> np.ndarray:
  return np.r_[STIM_CONDITIONS, CROSS_CONDITION] if args.rest_class else np.asarray(STIM_CONDITIONS)


def channel_check(path: Path, cfg: dict, args) -> set[int]:
  """Return channels that fail quality checks in the initial test stream."""
  sess = load_session(path, cfg["channels"]["scalp"], cfg["channels"]["mark_channel"])
  fs = sess["fs"]
  n = min(sess["eeg"].shape[1], int(round(args.check_sec * fs)))
  x = make_filter(cfg, fs).process(sess["eeg"][:, :n])
  win, step = int(round(args.window_sec * fs)), int(round(args.stride_sec * fs))
  W = np.stack([x[:, e - win:e] for e in range(win, n + 1, step)])
  bad, _ = identify_noisy_channels(W, channel_names=sess["channels"],
                                   fail_fraction_thresh=(cfg.get("quality") or {}).get("channel_fail_fraction", 0.5),
                                   **quality_limits(cfg))
  return set(int(i) for i in bad)


def train_on_run(path: Path, cfg: dict, args, exclude: set[int] = frozenset(), window_sec: float | None = None) -> dict:
  """Filter the training run, create windows, and train the network."""
  sess = load_session(path, cfg["channels"]["scalp"], cfg["channels"]["mark_channel"])
  fs = sess["fs"]
  x = make_filter(cfg, fs).process(sess["eeg"])
  base = cfg["preprocessing"]["baseline"]
  window_sec = window_sec or args.window_sec
  idx = build_window_index(sess["mark"], fs, sub_window_sec=window_sec, stride_sec=args.stride_sec,
                           stim_duration_sec=base["stimulus_duration_sec"],
                           # Treat the full pre-stimulus period as rest.
                           cross_duration_sec=REST_SEC if args.rest_class else base.get("pre_stim_cross_sec", 2.0),
                           cue_condition=None if args.rest_class else base["cue_condition"],
                           cue_lookback_sec=base["cue_lookback_sec"])
  cls = classes(args)
  stim = np.isin(idx.labels, cls)
  X = cut_windows(x, idx.starts[stim], int(round(window_sec * fs)))
  limits = quality_limits(cfg)
  noisy, _ = identify_noisy_channels(X, channel_names=sess["channels"],
                                     fail_fraction_thresh=(cfg.get("quality") or {}).get("channel_fail_fraction", 0.5),
                                     **limits)
  kept = [i for i in range(X.shape[1]) if i not in set(noisy) and i not in exclude]
  ok = validate_eeg_windows(X[:, kept], **limits).astype(bool)
  Xs = _standardize(X[ok][:, kept]).astype(np.float32)
  label = {int(c): i for i, c in enumerate(cls)}
  y = np.array([label[int(c)] for c in idx.labels[stim][ok]], dtype=np.int64)
  trials = idx.trial_ids[stim][ok]
  model, info = fit_network(args.model, Xs, y, trials, np.arange(len(y)), fs, args, args.seed, n_classes=len(cls))
  return {"model": model, "kept": kept, "fs": fs, "channels": sess["channels"], "n_train": len(y),
          "window_sec": window_sec,
          "n_train_trials": len(np.unique(trials)), **info}


def stream_run(path: Path, trained: dict, cfg: dict, args) -> dict:
  """Replay a run and return window ends, probabilities, and quality flags."""
  sess = load_session(path, cfg["channels"]["scalp"], cfg["channels"]["mark_channel"])
  fs, raw, kept = sess["fs"], sess["eeg"], trained["kept"]
  win, step = int(round(trained["window_sec"] * fs)), int(round(args.stride_sec * fs))
  filt = make_filter(cfg, fs)
  filtered = np.zeros_like(raw)
  limits = quality_limits(cfg)
  ends, windows = [], []
  for start in range(0, raw.shape[1] - step + 1, step):
    end = start + step
    filtered[:, start:end] = filt.process(raw[:, start:end])   # use data available so far
    if end >= win:
      ends.append(end)
      windows.append(filtered[kept, end - win:end])
  W = np.stack(windows)
  ok = validate_eeg_windows(W, **limits).astype(bool)
  probs = np.full((len(W), len(classes(args))), np.nan)
  if ok.any():
    # These windows contain only past samples, so batch prediction is equivalent.
    probs[ok] = _predict(trained["model"], torch.tensor(_standardize(W[ok]).astype(np.float32)), torch.device(args.device))
  return {"ends": np.array(ends), "probs": probs, "ok": ok, "mark": sess["mark"], "fs": fs, "win": win}


def decide(seg_start: int, seg_end: int, onset: int, s: dict, args) -> tuple[int | None, float, bool]:
  """Apply threshold stopping and return (class, time, forced)."""
  fs, win = s["fs"], s["win"]
  starts = s["ends"] - win
  sel = np.flatnonzero((starts >= seg_start + int(round(args.skip_sec * fs))) & (s["ends"] <= seg_end) & s["ok"])
  if len(sel) == 0:
    return None, np.nan, True
  avg = np.cumsum(s["probs"][sel], axis=0) / np.arange(1, len(sel) + 1)[:, None]
  target = avg[:, :N_CLASSES]                      # exclude the rest class
  hit = np.flatnonzero(target.max(axis=1) >= args.threshold)
  if len(hit):
    k = hit[0]
    return int(STIM_CONDITIONS[target[k].argmax()]), (s["ends"][sel[k]] - onset) / fs, False
  if args.rest_class:                              # no threshold: report no target
    return None, (s["ends"][sel[-1]] - onset) / fs, True
  k = len(sel) - 1
  return int(STIM_CONDITIONS[target[k].argmax()]), (s["ends"][sel[k]] - onset) / fs, True


def evaluate(s: dict, cfg: dict, args) -> tuple[pd.DataFrame, float]:
  """Score stimulation periods and check for false activations before them."""
  fs = s["fs"]
  stim_len = int(round(cfg["preprocessing"]["baseline"]["stimulus_duration_sec"] * fs))
  onsets = sorted((int(e), int(c)) for c in STIM_CONDITIONS for e in event_onsets(s["mark"], c))
  crosses = event_onsets(s["mark"], CROSS_CONDITION)
  rows, false_act = [], []
  for onset, code in onsets:
    pred, t, forced = decide(onset, onset + stim_len, onset, s, args)
    rows.append({"onset_s": onset / fs, "true": code, "pred": pred, "time_sec": t, "forced": forced})
    before = crosses[crosses < onset]
    if len(before):
      c = int(before[-1])
      p, _, f = decide(c, onset, c, s, args)
      false_act.append(p is not None and not f)
  return pd.DataFrame(rows), float(np.mean(false_act) * 100) if false_act else np.nan


def itr_bits_per_min(p: float, n: int, t_sec: float) -> float:
  if p <= 1.0 / n or not np.isfinite(t_sec) or t_sec <= 0:
    return 0.0
  bits = np.log2(n) + p * np.log2(p) + ((1 - p) * np.log2((1 - p) / (n - 1)) if p < 1 else 0.0)
  return float(bits * 60.0 / t_sec)


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--raw-root", type=Path, default=None, help="Default: config paths.data_raw_dir")
  ap.add_argument("--subjects", nargs="*", default=None)
  ap.add_argument("--train-session", default="OO")
  ap.add_argument("--test-session", default="mOO")
  ap.add_argument("--model", choices=("ccnn", "compact", "eegnet"), default="ccnn")
  ap.add_argument("--window-sec", type=float, default=0.5)
  ap.add_argument("--stride-sec", type=float, default=0.25)
  ap.add_argument("--skip-sec", type=float, default=0.5)
  ap.add_argument("--threshold", type=float, default=0.6)
  ap.add_argument("--gaze-sec", type=float, default=0.5)
  ap.add_argument("--min-val-acc", type=float, default=50.0,
                  help="Retrain with --fallback-window-sec if validation accuracy is below this (%%; 0 = off)")
  ap.add_argument("--fallback-window-sec", type=float, default=1.0)
  ap.add_argument("--check-sec", type=float, default=30.0,
                  help="Seconds at the start of the test stream used for the label-free channel check (0 = off)")
  ap.add_argument("--rest-class", action="store_true", help="Add a 6th 'no stimulation' class (Step 3b)")
  ap.add_argument("--no-augment", dest="augment", action="store_false")
  ap.add_argument("--epochs", type=int, default=150)
  ap.add_argument("--patience", type=int, default=25)
  ap.add_argument("--val-frac", type=float, default=0.2)
  ap.add_argument("--batch-size", type=int, default=32)
  ap.add_argument("--lr", type=float, default=1e-3)
  ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  ap.add_argument("--seed", type=int, default=42)
  ap.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "pipeline_config.yaml")
  ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "reports" / "pseudo_online")
  args = ap.parse_args()

  cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
  raw_root = args.raw_root or PROJECT_ROOT / cfg["paths"]["data_raw_dir"]
  subjects = args.subjects or sorted(p.name for p in raw_root.iterdir() if p.is_dir() and p.name.startswith("S"))
  if ("C" in args.train_session) != ("C" in args.test_session):
    print("[!] Training and test runs use different patterns (on-off vs checkerboard); "
          "checkerboard responses are expected at twice the frequency, so expect poor transfer.")

  print("=" * 100)
  print(f"PSEUDO-ONLINE | train {args.train_session} -> test {args.test_session} | {args.model} | "
        f"{args.window_sec:g} s windows every {args.stride_sec:g} s | skip {args.skip_sec:g} s | "
        f"{'6 classes (rest)' if args.rest_class else '5 classes'} | "
        f"threshold {args.threshold:g} | causal filtering | device {args.device}")
  print("=" * 100)
  records, decisions = [], []
  for sub in subjects:
    tr_path, te_path = raw_root / sub / f"{args.train_session}.ebr", raw_root / sub / f"{args.test_session}.ebr"
    if not (tr_path.exists() and te_path.exists()):
      print(f"  [-] {sub}: missing {args.train_session}.ebr or {args.test_session}.ebr")
      continue
    bad = channel_check(te_path, cfg, args) if args.check_sec > 0 else set()
    trained = train_on_run(tr_path, cfg, args, exclude=bad)
    if trained["val"] < args.min_val_acc and args.fallback_window_sec != args.window_sec:
      trained = train_on_run(tr_path, cfg, args, exclude=bad, window_sec=args.fallback_window_sec)
    s = stream_run(te_path, trained, cfg, args)
    dec, false_act = evaluate(s, cfg, args)
    dec.insert(0, "subject", sub)
    decisions.append(dec)
    acc = float((dec["pred"] == dec["true"]).mean())
    t = float(dec["time_sec"].mean())
    rec = {"Subject": sub, "Trials": len(dec), "Acc (%)": round(acc * 100, 2), "Time (s)": round(t, 2),
           ("Missed (%)" if args.rest_class else "Forced (%)"): round(float(dec["forced"].mean()) * 100, 1),
           "ITR (bits/min)": round(itr_bits_per_min(acc, N_CLASSES, t + args.gaze_sec), 2),
           "False act. (%)": round(false_act, 1),
           "Windows rejected (%)": round(float((~s["ok"]).mean()) * 100, 1),
           "Window (s)": trained["window_sec"],
           "Channels kept": len(trained["kept"]),
           "Dropped at check": ", ".join(trained["channels"][i] for i in sorted(bad)) or "none",
           "Train windows": trained["n_train"],
           "Val acc (%)": round(trained["val"], 1)}
    records.append(rec)
    miss = "Missed (%)" if args.rest_class else "Forced (%)"
    print(f"  {sub}: acc {rec['Acc (%)']:5.1f}% | decision at {rec['Time (s)']:.2f} s | {miss.split()[0].lower()} {rec[miss]:4.1f}% | "
          f"ITR {rec['ITR (bits/min)']:5.1f} | false act. in no-stim {rec['False act. (%)']:5.1f}% | "
          f"windows rejected {rec['Windows rejected (%)']:.1f}% | window {rec['Window (s)']:g} s | channels {rec['Channels kept']} "
          f"(check dropped: {rec['Dropped at check']}) | "
          f"train {rec['Train windows']} windows (val {rec['Val acc (%)']:.0f}%)", flush=True)

  if not records:
    return
  df = pd.DataFrame(records)
  args.out_dir.mkdir(parents=True, exist_ok=True)
  tag = f"{args.train_session}_to_{args.test_session}_{args.model}" + ("_rest" if args.rest_class else "")
  df.to_csv(args.out_dir / f"summary_{tag}.csv", index=False)
  pd.concat(decisions, ignore_index=True).to_csv(args.out_dir / f"decisions_{tag}.csv", index=False)
  m = df.mean(numeric_only=True)
  print("=" * 100)
  miss = "Missed (%)" if args.rest_class else "Forced (%)"
  print(f"Cohort | acc {m['Acc (%)']:.1f}% | decision at {m['Time (s)']:.2f} s | {miss.split()[0].lower()} {m[miss]:.1f}% | "
        f"ITR {m['ITR (bits/min)']:.1f} bits/min | false activations {m['False act. (%)']:.1f}%")
  print(f"Saved summary_{tag}.csv and decisions_{tag}.csv to {args.out_dir}")


if __name__ == "__main__":
  main()