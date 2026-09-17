"""Evaluate FBCSP + FBCCA -> Shrinkage-LDA across all available subjects.

subjects are auto-discovered from `data/processed/`, so
this scales any N subjects without editing the script. Each
subject is processed independently (its own channel-drop decision, its
own CV folds), so adding subjects only adds linear wall-clock time and since subjects are independent, the loop can
optionally run in parallel across CPU cores with `--n-jobs`.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.features.fbcca_extraction import extract_fbcca_features
from src.features.fbcsp_extraction import (
    DEFAULT_SUBBANDS,
    MulticlassCSP,
    butter_bandpass_filter,
)

# Full recorded montage (8 channels, per the project's actual acquisition
# setup). PO7 used to be hard-excluded upstream before any artifact-quality
# check ran -- it is now included here and left to the two-tier quality
# pipeline to decide, per subject, whether it should be dropped.
FULL_MONTAGE = ["PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"]
SUBJECT_DIR_PATTERN = re.compile(r"^S\d+$")  


def resolve_channel_names(n_channels: int, full_montage: list[str] = FULL_MONTAGE) -> list[str] | None:
  """Match the actual channel count in the data to known channel-name sets.

  Handles the current 8-channel montage, so nothing
  breaks silently if not every subject's files have been regenerated yet.
  """
  if n_channels == len(full_montage):
    return list(full_montage)
  if n_channels == len(full_montage) - 1:
    print(
        f"    [!] {n_channels} channels found (expected {len(full_montage)}); "
        "assuming a channel is missing from subject's raw windows -- "
        "rerun preprocessing for this subject to include it."
    )
    return full_montage[1:]
  print(f"    [!] Unexpected channel count ({n_channels}); using indices only.")
  return None


def discover_subjects(data_root: Path) -> list[str]:
  """Return every subject folder name under `data_root` matching S<digits>.

  Auto-discovery means adding subject n `data/processed/`
  is enough -- no script edits required.
  """
  if not data_root.exists():
    return []
  subjects = sorted(
      p.name for p in data_root.iterdir()
      if p.is_dir() and SUBJECT_DIR_PATTERN.match(p.name)
  )
  return subjects


def process_subject(sub: str, data_root: Path) -> dict | None:
  """Run the full quality -> feature -> CV pipeline for one subject.

  Self-contained on purpose: it only touches this subject's files and
  returns a plain dict, so it can be called either in a simple for-loop
  or dispatched to a worker process for parallel execution.
  """
  sub_dir = data_root / sub
  if not (sub_dir / "X_time_windows.npy").exists():
    print(f"[-] Skipping {sub}: X_time_windows.npy not found in {sub_dir}.")
    return None

  windows_raw = np.load(sub_dir / "X_time_windows.npy")
  y_raw = np.load(sub_dir / "y_labels.npy")

  # Filter target stimuli
  stim_mask = np.isin(y_raw, [101, 102, 103, 104, 105])
  y_stim = y_raw[stim_mask]
  windows = (
      windows_raw[stim_mask]
      if windows_raw.shape[0] == len(y_raw)
      else windows_raw[: len(y_stim)]
  )
  channel_names = resolve_channel_names(windows.shape[1])

  # Two-part artifact quality (channel-level drop, then window-level gate)
  # + FBCCA, computed once and shared across CV folds (FBCCA
  # templates are fixed sinusoidal references, not fit on the data).
  X_fbcca, y_clean, valid_mask, ch_report = extract_fbcca_features(
      windows, y_stim, fs=250.0, channel_names=channel_names, verbose=False
  )
  windows_clean = windows[valid_mask == 1][:, ch_report["channels_kept_idx"], :]
  n_clean = len(y_clean)
  classes = np.unique(y_clean)

  if n_clean < 10 or len(classes) < 2:
    print(f"[-] Skipping {sub}: not enough clean windows/classes after QC.")
    return None

  # Stratified 5-Fold Cross-Validation
  skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  oof_probs = np.zeros((n_clean, len(classes)))

  for train_idx, test_idx in skf.split(windows_clean, y_clean):
    y_tr, y_te = y_clean[train_idx], y_clean[test_idx]

    # In-fold FBCSP (fit strictly on the training fold to avoid leakage)
    X_tr_fold, X_te_fold = [], []
    for low, high in DEFAULT_SUBBANDS:
      w_tr = butter_bandpass_filter(
          windows_clean[train_idx], low, high, fs=250.0, order=4
      )
      w_te = butter_bandpass_filter(
          windows_clean[test_idx], low, high, fs=250.0, order=4
      )
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

  # 1.0s Accuracy
  preds_1s = classes[np.argmax(oof_probs, axis=1)]
  acc_1s = accuracy_score(y_clean, preds_1s) * 100.0

  # 2.0s Causal Trial-Aware Smoothing (resets every 5 sub-windows/trial)
  preds_smooth = np.zeros_like(y_clean)
  for i in range(len(y_clean)):
    if i % 5 == 0:
      avg_p = oof_probs[i]
    else:
      avg_p = np.mean(oof_probs[i - 1 : i + 1], axis=0)
    preds_smooth[i] = classes[np.argmax(avg_p)]
  acc_smooth = accuracy_score(y_clean, preds_smooth) * 100.0

  channels_dropped = ch_report["channels_dropped"] or []
  print(
      f"  [+] {sub:5s} | Channels kept: {len(ch_report['channels_kept_idx'])}/"
      f"{windows.shape[1]} (dropped: {channels_dropped or 'none'}) | "
      f"Clean: {n_clean:3d} | 1.0s: {acc_1s:6.2f}% | 2.0s Smoothed: {acc_smooth:6.2f}%"
  )

  return {
      "Subject": sub,
      "Channels Kept": len(ch_report["channels_kept_idx"]),
      "Channels Dropped": ", ".join(channels_dropped) if channels_dropped else "none",
      "Clean Windows": n_clean,
      "Rejected Windows": int(np.sum(valid_mask == 0)),
      "1.0s Acc (%)": round(acc_1s, 2),
      "2.0s Smoothed (%)": round(acc_smooth, 2),
  }


def main():
  parser = argparse.ArgumentParser(
      description="FBCSP + FBCCA -> Shrinkage-LDA benchmark across all discovered subjects."
  )
  parser.add_argument(
      "--n-jobs", type=int, default=1,
      help="Parallel workers across subjects (default: 1 = sequential). "
           "Each subject is fully independent, so this scales safely with "
           "cohort size, e.g. --n-jobs -1 to use all CPU cores for 40-45 subjects.",
  )
  args = parser.parse_args()

  project_root = Path(__file__).resolve().parents[2]
  data_root = project_root / "data" / "processed"
  subjects = discover_subjects(data_root)

  print("=" * 75)
  print(f"   N-SUBJECT BENCHMARK: FBCSP (m=1) + FBCCA -> SHRINKAGE-LDA  (N={len(subjects)})")
  print("=" * 75)

  if not subjects:
    print(f"No subject folders matching 'S<digits>' found under {data_root}.")
    return

  if args.n_jobs == 1:
    records = [process_subject(sub, data_root) for sub in subjects]
  else:
    try:
      from joblib import Parallel, delayed
    except ImportError:
      print("[!] joblib not installed; falling back to sequential execution.")
      records = [process_subject(sub, data_root) for sub in subjects]
    else:
      records = Parallel(n_jobs=args.n_jobs)(
          delayed(process_subject)(sub, data_root) for sub in subjects
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
    )

  out_path = project_root / "reports" / "n_subjects_benchmark.csv"
  out_path.parent.mkdir(parents=True, exist_ok=True)
  df_results.to_csv(out_path, index=False)
  print(f"Results saved to: {out_path}")


if __name__ == "__main__":
  main()