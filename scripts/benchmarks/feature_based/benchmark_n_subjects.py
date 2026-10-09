"""Run the FBCSP + FBCCA + Shrinkage-LDA benchmark for all subjects.

Subjects are discovered under the data root and processed independently.
Use `--n-jobs` to run subjects in parallel.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


def _find_root(start: Path) -> Path:
  for cand in [start, *start.parents]:
    if (cand / "configs" / "pipeline_config.yaml").exists():
      return cand
  return start.parents[3]


PROJECT_ROOT = _find_root(Path(__file__).resolve())
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.ssvep.fbcca_extraction import extract_fbcca_features  # noqa: E402
from src.features.ssvep.fbcsp_extraction import (  # noqa: E402
    DEFAULT_SUBBANDS,
    MulticlassCSP,
    butter_bandpass_filter,
)
from src.io.subject_data import QUALITY_MODES, discover_subjects, load_subject  # noqa: E402

STIM_CONDITIONS = [101, 102, 103, 104, 105]


def _cv_once(seed, windows_clean, y_clean, trial_ids_clean, classes, X_fbcca, bands_fbcsp, fs, built):
  """Run one 5-fold CV and return the two accuracies."""
  n_clean = len(y_clean)
  skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
  oof_probs = np.zeros((n_clean, len(classes)))

  for train_idx, test_idx in skf.split(windows_clean, y_clean, groups=trial_ids_clean):
    y_tr = y_clean[train_idx]

    # Fit CSP on the training fold only.
    X_tr_fold, X_te_fold = [], []
    for b, (low, high) in enumerate(DEFAULT_SUBBANDS):
      if bands_fbcsp is not None:
        w_tr, w_te = bands_fbcsp[train_idx, b], bands_fbcsp[test_idx, b]
      else:
        w_tr = butter_bandpass_filter(windows_clean[train_idx], low, high, fs=fs, order=4)
        w_te = butter_bandpass_filter(windows_clean[test_idx], low, high, fs=fs, order=4)
      csp = MulticlassCSP(n_components=1).fit(w_tr, y_tr)
      X_tr_fold.append(csp.transform(w_tr))
      X_te_fold.append(csp.transform(w_te))

    X_tr = np.hstack([np.hstack(X_tr_fold), X_fbcca[train_idx]])
    X_te = np.hstack([np.hstack(X_te_fold), X_fbcca[test_idx]])

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    clf.fit(X_tr_sc, y_tr)
    probs = clf.predict_proba(X_te_sc)

    for col_idx, cls_label in enumerate(clf.classes_):
      t_col = np.where(classes == cls_label)[0][0]
      oof_probs[test_idx, t_col] = probs[:, col_idx]

  # Unsmooth accuracy.
  preds_1s = classes[np.argmax(oof_probs, axis=1)]
  acc_1s = accuracy_score(y_clean, preds_1s) * 100.0

  # Smooth with the previous window within each trial.
  preds_smooth = np.zeros_like(y_clean)
  for i in range(n_clean):
    if not built:
      same_trial = i % 5 != 0
    else:
      same_trial = i > 0 and trial_ids_clean[i] == trial_ids_clean[i - 1]
    avg_p = np.mean(oof_probs[i - 1:i + 1], axis=0) if same_trial else oof_probs[i]
    preds_smooth[i] = classes[np.argmax(avg_p)]
  acc_smooth = accuracy_score(y_clean, preds_smooth) * 100.0
  return acc_1s, acc_smooth


def process_subject(sub: str, data_root: Path, fs_override: float | None = None,
                    quality_mode: str = "shared", cv_repeats: int = 1) -> dict | None:
  """Run the full quality -> feature -> CV pipeline for one subject."""
  sub_dir = data_root / sub
  if not (sub_dir / "X_time_windows.npy").exists():
    print(f"[-] Skipping {sub}: X_time_windows.npy not found in {sub_dir}.")
    return None

  d = load_subject(sub_dir, STIM_CONDITIONS, quality_mode=quality_mode,
                   bands=("fbcca", "fbcsp"), fs_override=fs_override)
  windows, y_stim, fs = d["windows"], d["y"], d["fs"]
  bands_fbcca = d["bands"].get("fbcca")
  bands_fbcsp_all = d["bands"].get("fbcsp")

  # Compute quality and FBCCA features once for all CV folds.
  X_fbcca, y_clean, valid_mask, ch_report = extract_fbcca_features(
      windows, y_stim, fs=fs, channel_names=d["channel_names"], verbose=False,
      subband_windows=bands_fbcca, quality=d["quality"],
  )
  keep = valid_mask == 1
  kept_idx = ch_report["channels_kept_idx"]
  windows_clean = windows[keep][:, kept_idx, :]
  bands_fbcsp = bands_fbcsp_all[keep][:, :, kept_idx, :] if bands_fbcsp_all is not None else None
  n_clean = len(y_clean)
  classes = np.unique(y_clean)

  # Keep trial groups aligned with the filtered windows.
  trial_ids_clean = d["trial_ids"][keep]

  if n_clean < 10 or len(classes) < 2:
    print(f"[-] Skipping {sub}: not enough clean windows/classes after QC.")
    return None

  # Repeat CV with different seeds when requested.
  accs = [_cv_once(42 + r, windows_clean, y_clean, trial_ids_clean, classes, X_fbcca, bands_fbcsp, fs, d["built"])
          for r in range(cv_repeats)]
  accs = np.array(accs)
  acc_1s, acc_smooth = float(accs[:, 0].mean()), float(accs[:, 1].mean())

  channels_dropped = ch_report["channels_dropped"] or []
  print(
      f"  [+] {sub:5s} | fs={fs:g} | {d['source']} | Channels kept: {len(kept_idx)}/"
      f"{windows.shape[1]} (dropped: {channels_dropped or 'none'}) | "
      f"Clean: {n_clean:3d} | 1.0s: {acc_1s:6.2f}% | 2.0s Smoothed: {acc_smooth:6.2f}%"
      + (f" (mean of {cv_repeats} CV runs)" if cv_repeats > 1 else "")
  )

  return {
      "Subject": sub,
      "Channels Kept": len(kept_idx),
      "Channels Dropped": ", ".join(channels_dropped) if channels_dropped else "none",
      "Clean Windows": n_clean,
      "Rejected Windows": int(np.sum(valid_mask == 0)),
      "1.0s Acc (%)": round(acc_1s, 2),
      "2.0s Smoothed (%)": round(acc_smooth, 2),
      **({"1.0s SD": round(float(accs[:, 0].std()), 2), "2.0s SD": round(float(accs[:, 1].std()), 2),
          "CV repeats": cv_repeats} if cv_repeats > 1 else {}),
  }


def main():
  parser = argparse.ArgumentParser(
      description="FBCSP (m=1) + FBCCA -> Shrinkage-LDA benchmark across all discovered subjects."
  )
  parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "processed",
                      help="Folder with one S<digits> sub-folder per subject (default: data/processed)")
  parser.add_argument("--quality", choices=QUALITY_MODES, default="shared",
                      help="Which quality decision to use for built folders (default: shared).")
  parser.add_argument("--fs", type=float, default=None,
                      help="Override the sampling rate (e.g. 250 to reproduce the old setting).")
  parser.add_argument("--out", type=Path, default=None,
                      help="CSV path (default: reports/n_subjects_benchmark[_<data-root>][_<quality>].csv)")
  parser.add_argument("--cv-repeats", type=int, default=1,
                      help="Repeat the 5-fold CV with N different seeds and report the mean "
                           "(default 1 = seed 42 only, comparable with earlier runs; 10 recommended).")
  parser.add_argument(
      "--n-jobs", type=int, default=1,
      help="Parallel workers across subjects (default: 1 = sequential).",
  )
  args = parser.parse_args()

  data_root = args.data_root if args.data_root.is_absolute() else PROJECT_ROOT / args.data_root
  subjects = discover_subjects(data_root)

  print("=" * 75)
  print(f"   N-SUBJECT BENCHMARK: FBCSP (m=1) + FBCCA -> SHRINKAGE-LDA  (N={len(subjects)})")
  print(f"   data: {data_root} | quality: {args.quality}")
  print("=" * 75)

  if not subjects:
    print(f"No subject folders matching 'S<digits>' found under {data_root}.")
    return

  run = lambda sub: process_subject(sub, data_root, args.fs, args.quality, args.cv_repeats)  # noqa: E731
  if args.n_jobs == 1:
    records = [run(sub) for sub in subjects]
  else:
    try:
      from joblib import Parallel, delayed
    except ImportError:
      print("[!] joblib not installed; falling back to sequential execution.")
      records = [run(sub) for sub in subjects]
    else:
      records = Parallel(n_jobs=args.n_jobs)(
          delayed(process_subject)(sub, data_root, args.fs, args.quality, args.cv_repeats) for sub in subjects
      )

  records = [r for r in records if r is not None]

  df_results = pd.DataFrame(records)
  print("=" * 75)
  print(df_results.to_string(index=False))
  print("=" * 75)
  if len(df_results) > 0:
    print(
        f"Cohort mean | 1.0s: {df_results['1.0s Acc (%)'].mean():.2f}% | "
        f"2.0s Smoothed: {df_results['2.0s Smoothed (%)'].mean():.2f}%"
        + (f"  (each subject = mean of {args.cv_repeats} CV runs)" if args.cv_repeats > 1 else "")
    )

  suffix = "" if data_root.name == "processed" else f"_{data_root.name}"
  if data_root.name != "processed" and args.quality != "shared":
    suffix += f"_{args.quality.replace('-', '_')}"
  if args.cv_repeats > 1:
    suffix += f"_cv{args.cv_repeats}"
  out_path = args.out or PROJECT_ROOT / "reports" / f"n_subjects_benchmark{suffix}.csv"
  out_path.parent.mkdir(parents=True, exist_ok=True)
  df_results.to_csv(out_path, index=False)
  print(f"Results saved to: {out_path}")


if __name__ == "__main__":
  main()