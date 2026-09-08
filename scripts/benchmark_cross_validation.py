import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from src.features.psd_extraction import extract_continuous_psd_features
from src.features.fbcca_extraction import extract_fbcca_features
from src.features.fbcsp_extraction import MulticlassCSP, butter_bandpass_filter, DEFAULT_SUBBANDS
from src.models.sklearn_models import get_sklearn_model_suite

parser = argparse.ArgumentParser(description="Run 5-Fold Cross-Validation across single and hybrid feature representations.")
parser.add_argument("--subject", help="Subject folder (e.g., S01). Defaults to root processed.")
args = parser.parse_args()

project_root = Path(__file__).resolve().parents[1]
data_dir = project_root / "data" / "processed"
if args.subject:
    data_dir = data_dir / args.subject

# 1. Load data
windows_raw = np.load(data_dir / "X_time_windows.npy")
y_raw = np.load(data_dir / "y_labels.npy")

stim_mask = np.isin(y_raw, [101, 102, 103, 104, 105])
y_stim = y_raw[stim_mask]
windows = windows_raw[stim_mask] if windows_raw.shape[0] == len(y_raw) else windows_raw[:len(y_stim)]

# 2. Extract Base Features
print(f"Loaded {windows.shape[0]} windows across {windows.shape[1]} channels.")
print("Extracting feature representations...")

X_psd_cont, y_clean, _ = extract_continuous_psd_features(windows, y_stim, fs=250.0)
X_fbcca, _, _ = extract_fbcca_features(windows, y_stim, fs=250.0)

# Unsupervised FBCCA Baseline (argmax without any training)
label_map = {0: 101, 1: 102, 2: 103, 3: 104, 4: 105}
unsupervised_preds = np.array([label_map[i] for i in np.argmax(X_fbcca, axis=1)])
direct_fbcca_acc = accuracy_score(y_clean, unsupervised_preds) * 100.0
print(f"\n>>> Direct Unsupervised FBCCA Baseline (No ML, Argmax): {direct_fbcca_acc:.2f}% <<<\n")

# Targeted Harmonic PSD (4 visual channels x 8 harmonic peaks = 32 dims)
# Visual channels: O1, Oz, O2, POz (indices 3, 4, 5, 6 in 7-scalp montage)
keep_ch = [3, 4, 5, 6] if windows.shape[1] >= 7 else list(range(windows.shape[1]))
target_freqs = [8.57, 10.91, 15.0, 17.14, 20.0, 21.82, 24.0, 30.0]
freqs = np.linspace(0.0, 125.0, windows.shape[-1] // 2 + 1)
target_idx = [np.argmin(np.abs(freqs - f)) for f in target_freqs]

fft_vals = np.abs(np.fft.rfft(windows[:, keep_ch, :], axis=-1)) ** 2
X_psd_harm = fft_vals[:, :, target_idx].reshape(windows.shape[0], -1)

# 3. Setup Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
models = get_sklearn_model_suite()

# Exclude QDA on high-dim spaces to prevent rank failure
test_models = {k: v for k, v in models.items() if "QDA" not in k}

feature_spaces = {
    "1. Continuous PSD": (X_psd_cont, False),
    "2. Harmonic PSD (Visual Ch)": (X_psd_harm, False),
    "3. FBCCA Priors": (X_fbcca, False),
    "4. Hybrid: Harmonic PSD + FBCCA": (np.hstack([X_psd_harm, X_fbcca]), False),
    "5. FBCSP (Leak-Free CV)": (windows, True),
    "6. Hybrid: FBCSP + FBCCA": (windows, "fbcsp_fbcca"),
}

results = []

print("=" * 80)
print("             5-FOLD STRATIFIED CROSS-VALIDATION GENERALIZATION BENCHMARK")
print("=" * 80)

for feat_name, (data_obj, is_csp) in feature_spaces.items():
    print(f"\nEvaluating: {feat_name}...")
    
    for model_name, clf_template in test_models.items():
        fold_accs = []
        
        for train_idx, test_idx in skf.split(windows, y_clean):
            y_tr, y_te = y_clean[train_idx], y_clean[test_idx]
            
            # Proper Leak-Free CSP transformation inside the fold
            if is_csp is True:
                X_tr_fold, X_te_fold = [], []
                for low, high in DEFAULT_SUBBANDS:
                    w_tr = butter_bandpass_filter(data_obj[train_idx], low, high, fs=250.0)
                    w_te = butter_bandpass_filter(data_obj[test_idx], low, high, fs=250.0)
                    csp = MulticlassCSP(n_components=1).fit(w_tr, y_tr)
                    X_tr_fold.append(csp.transform(w_tr))
                    X_te_fold.append(csp.transform(w_te))
                X_tr = np.hstack(X_tr_fold)
                X_te = np.hstack(X_te_fold)
            elif is_csp == "fbcsp_fbcca":
                X_tr_fold, X_te_fold = [], []
                for low, high in DEFAULT_SUBBANDS:
                    w_tr = butter_bandpass_filter(data_obj[train_idx], low, high, fs=250.0)
                    w_te = butter_bandpass_filter(data_obj[test_idx], low, high, fs=250.0)
                    csp = MulticlassCSP(n_components=1).fit(w_tr, y_tr)
                    X_tr_fold.append(csp.transform(w_tr))
                    X_te_fold.append(csp.transform(w_te))
                X_tr = np.hstack([np.hstack(X_tr_fold), X_fbcca[train_idx]])
                X_te = np.hstack([np.hstack(X_te_fold), X_fbcca[test_idx]])
            else:
                X_tr, X_te = data_obj[train_idx], data_obj[test_idx]
            
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr)
            X_te_scaled = scaler.transform(X_te)
            
            clf = type(clf_template)(**clf_template.get_params())
            clf.fit(X_tr_scaled, y_tr)
            preds = clf.predict(X_te_scaled)
            fold_accs.append(accuracy_score(y_te, preds))
        
        mean_acc = np.mean(fold_accs) * 100.0
        std_acc = np.std(fold_accs) * 100.0
        results.append({
            "Feature Representation": feat_name,
            "Model Name": model_name,
            "CV Test Accuracy (%)": f"{mean_acc:.2f} ± {std_acc:.2f}%"
        })

df_cv = pd.DataFrame(results)
print("\n" + "=" * 80)
print("                           CROSS-VALIDATION RESULTS")
print("=" * 80)
# Group and show top performers
for feat in feature_spaces.keys():
    subset = df_cv[df_cv["Feature Representation"] == feat]
    print(f"\n--- {feat} ---")
    print(subset[["Model Name", "CV Test Accuracy (%)"]].to_string(index=False))