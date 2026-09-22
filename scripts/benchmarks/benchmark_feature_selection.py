"""Benchmark Multiple Feature Selection Methods against Full Hybrid Space."""

import argparse
from pathlib import Path
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from src.features.fbcca_extraction import extract_fbcca_features
from src.features.fbcsp_extraction import (
    DEFAULT_SUBBANDS,
    MulticlassCSP,
    butter_bandpass_filter,
)
from src.features.feature_selection import select_k_best_fbcsp_features
from src.preprocessing.cv_utils import build_trial_ids

FULL_MONTAGE = ["PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"]


def resolve_channel_names(n_channels: int, full_montage: list[str] = FULL_MONTAGE) -> list[str] | None:
  """Match the actual channel count to known montage layouts."""
  if n_channels == len(full_montage):
    return list(full_montage)
  if n_channels == len(full_montage) - 1:
    print(
        f"[!] {n_channels} channels found (expected {len(full_montage)}); "
        "assuming PO7 is still missing from this data -- rerun "
        "preprocessing to include it."
    )
    return full_montage[1:]
  print(f"[!] Unexpected channel count ({n_channels}); channel names unavailable.")
  return None


parser = argparse.ArgumentParser(
    description="Benchmark Feature Selection on SSVEP Hybrid Space."
)
parser.add_argument(
    "--subject",
    default="S03",
    help="Subject subdirectory (e.g., S01, S03, S05). Defaults to S03.",
)
parser.add_argument(
    "--k-features",
    type=int,
    default=15,
    help="Number of FBCSP features to select (default: 15).",
)
args = parser.parse_args()

project_root = Path(__file__).resolve().parents[2]
data_dir = project_root / "data" / "processed" / args.subject

# 1. Load data
windows_raw = np.load(data_dir / "X_time_windows.npy")
y_raw = np.load(data_dir / "y_labels.npy")
stim_mask = np.isin(y_raw, [101, 102, 103, 104, 105])
y_stim = y_raw[stim_mask]
windows = (
    windows_raw[stim_mask]
    if windows_raw.shape[0] == len(y_raw)
    else windows_raw[: len(y_stim)]
)

# 2. Extract FBCCA & validation mask (two-tier: chronic bad channels dropped
# for this subject first, then remaining noisy windows rejected)
channel_names = resolve_channel_names(windows.shape[1])
X_fbcca, y_clean, valid_mask, ch_report = extract_fbcca_features(
    windows, y_stim, fs=250.0, channel_names=channel_names
)
windows_clean = windows[valid_mask == 1][:, ch_report["channels_kept_idx"], :]
print(f"Subject: {args.subject} | Verified clean windows: {len(y_clean)}")
if ch_report["channels_dropped"]:
  print(f"Channels dropped for this subject: {ch_report['channels_dropped']}")

# Trial ids computed on the PRE-quality-gate label array, then filtered
# the same way as y_clean -- sub-windows of the same 5.0s trial must never
# split across train/test folds.
trial_ids_full = build_trial_ids(y_stim, sub_windows_per_trial=5)
trial_ids_clean = trial_ids_full[valid_mask == 1]

# 3. Setup Cross-Validation
skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
classes = np.unique(y_clean)

# Containers for out-of-fold probability predictions
methods = [
    "1. Full Baseline (p=55)",
    f"2. ANOVA F-Score (p={args.k_features + 5})",
    f"3. Mutual Info (p={args.k_features + 5})",
    "4. Subband Pruning (p=45)",
]
oof_probs = {m: np.zeros((len(y_clean), len(classes))) for m in methods}

for fold_idx, (train_idx, test_idx) in enumerate(
    skf.split(windows_clean, y_clean, groups=trial_ids_clean), 1
):
    y_tr, y_te = y_clean[train_idx], y_clean[test_idx]

    # In-fold FBCSP transformation
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

    X_tr_fbcsp = np.hstack(X_tr_fold)
    X_te_fbcsp = np.hstack(X_te_fold)

    # In-fold scaling
    scaler_fbcsp = StandardScaler()
    X_tr_fbcsp_sc = scaler_fbcsp.fit_transform(X_tr_fbcsp)
    X_te_fbcsp_sc = scaler_fbcsp.transform(X_te_fbcsp)

    scaler_fbcca = StandardScaler()
    X_tr_fbcca_sc = scaler_fbcca.fit_transform(X_fbcca[train_idx])
    X_te_fbcca_sc = scaler_fbcca.transform(X_fbcca[test_idx])

    # Method 1: Full Baseline (50 FBCSP + 5 FBCCA = 55)
    X_tr_full = np.hstack([X_tr_fbcsp_sc, X_tr_fbcca_sc])
    X_te_full = np.hstack([X_te_fbcsp_sc, X_te_fbcca_sc])

    # Method 2: ANOVA F-Score Filter
    sel_anova, _ = select_k_best_fbcsp_features(
        X_tr_fbcsp_sc, y_tr, n_features_to_select=args.k_features, method="anova"
    )
    X_tr_anova = np.hstack([X_tr_fbcsp_sc[:, sel_anova], X_tr_fbcca_sc])
    X_te_anova = np.hstack([X_te_fbcsp_sc[:, sel_anova], X_te_fbcca_sc])

    # Method 3: Mutual Information Filter
    sel_mi, _ = select_k_best_fbcsp_features(
        X_tr_fbcsp_sc,
        y_tr,
        n_features_to_select=args.k_features,
        method="mutual_info",
    )
    X_tr_mi = np.hstack([X_tr_fbcsp_sc[:, sel_mi], X_tr_fbcca_sc])
    X_te_mi = np.hstack([X_te_fbcsp_sc[:, sel_mi], X_te_fbcca_sc])

    # Method 4: Physiological Subband Pruning (Keep Bands 1-4, Drop Band 5: 38-52 Hz)
    # Band 5 is columns 40..49
    bands_1_to_4 = list(range(40))
    X_tr_pruned = np.hstack([X_tr_fbcsp_sc[:, bands_1_to_4], X_tr_fbcca_sc])
    X_te_pruned = np.hstack([X_te_fbcsp_sc[:, bands_1_to_4], X_te_fbcca_sc])

    train_test_pairs = {
        "1. Full Baseline (p=55)": (X_tr_full, X_te_full),
        f"2. ANOVA F-Score (p={args.k_features + 5})": (X_tr_anova, X_te_anova),
        f"3. Mutual Info (p={args.k_features + 5})": (X_tr_mi, X_te_mi),
        "4. Subband Pruning (p=45)": (X_tr_pruned, X_te_pruned),
    }

    for name, (X_tr_m, X_te_m) in train_test_pairs.items():
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        clf.fit(X_tr_m, y_tr)
        probs = clf.predict_proba(X_te_m)
        for c_i, cls in enumerate(clf.classes_):
            t_col = np.where(classes == cls)[0][0]
            oof_probs[name][test_idx, t_col] = probs[:, c_i]


# 4. Evaluation Function
def score_predictions(probs: np.ndarray, y_true: np.ndarray):
    preds_1s = classes[np.argmax(probs, axis=1)]
    acc_1s = accuracy_score(y_true, preds_1s) * 100.0

    preds_smooth = np.zeros_like(y_true)
    for i in range(len(y_true)):
        if i % 5 == 0:
            avg_p = probs[i]
        else:
            avg_p = np.mean(probs[i - 1 : i + 1], axis=0)
        preds_smooth[i] = classes[np.argmax(avg_p)]
    acc_smooth = accuracy_score(y_true, preds_smooth) * 100.0
    return acc_1s, acc_smooth


# 5. Summary Table
print("\n" + "=" * 80)
print(f"       FEATURE SELECTION BENCHMARK SUMMARY (SUBJECT {args.subject})")
print("=" * 80)
print(
    f"{'Selection Method':<36} | {'Dim (p)':<8} | {'1.0s Acc (%)':<14} | {'2.0s Smoothed (%)'}"
)
print("-" * 80)

dims = [55, args.k_features + 5, args.k_features + 5, 45]
for name, p_dim in zip(methods, dims):
    acc_1s, acc_smooth = score_predictions(oof_probs[name], y_clean)
    print(f"{name:<36} | p={p_dim:<6} | {acc_1s:6.2f}%       | {acc_smooth:6.2f}%")
print("=" * 80)