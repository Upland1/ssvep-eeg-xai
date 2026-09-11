"""Evaluate the Winning FBCSP (m=1) + FBCCA -> Shrinkage-LDA across 5 subjects."""

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

project_root = Path(__file__).resolve().parents[2]
data_root = project_root / "data" / "processed"
subjects = ["S01", "S02", "S03", "S04", "S05"]

records = []

print("=" * 75)
print("     5-SUBJECT BENCHMARK: FBCSP (m=1) + FBCCA -> SHRINKAGE-LDA")
print("=" * 75)

for sub in subjects:
  sub_dir = data_root / sub
  if not sub_dir.exists():
    print(f"[-] Skipping {sub}: directory {sub_dir} not found.")
    continue

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

  # Quality gate + FBCCA
  X_fbcca, y_clean, valid_mask = extract_fbcca_features(
      windows, y_stim, fs=250.0
  )
  windows_clean = windows[valid_mask == 1]
  n_clean = len(y_clean)

  # Stratified 5-Fold Cross-Validation
  skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  oof_probs = np.zeros((n_clean, 5))
  classes = np.unique(y_clean)

  for train_idx, test_idx in skf.split(windows_clean, y_clean):
    y_tr, y_te = y_clean[train_idx], y_clean[test_idx]

    # In-fold FBCSP
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

  # 2.0s Causal Smoothing
  preds_smooth = np.zeros_like(y_clean)
  for i in range(len(y_clean)):
    if i % 5 == 0:
      avg_p = oof_probs[i]
    else:
      avg_p = np.mean(oof_probs[i - 1 : i + 1], axis=0)
    preds_smooth[i] = classes[np.argmax(avg_p)]
  acc_smooth = accuracy_score(y_clean, preds_smooth) * 100.0

  records.append({
      "Subject": sub,
      "Clean Windows": n_clean,
      "Rejected": int(np.sum(valid_mask == 0)),
      "1.0s Acc (%)": round(acc_1s, 2),
      "2.0s Smoothed (%)": round(acc_smooth, 2),
  })
  print(
      f"  [+] {sub:5s} | Clean: {n_clean:3d} | 1.0s: {acc_1s:6.2f}% |"
      f" 2.0s Smoothed: {acc_smooth:6.2f}%"
  )

# Output Summary Table
df_results = pd.DataFrame(records)
print("=" * 75)
print(df_results.to_string(index=False))
print("=" * 75)

# Save to reports
out_path = project_root / "reports" / "five_subjects_benchmark.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
df_results.to_csv(out_path, index=False)
print(f"Results saved to: {out_path}")