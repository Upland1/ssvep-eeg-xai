import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from src.features.fbcca_extraction import extract_fbcca_features
from src.features.fbcsp_extraction import extract_fbcsp_features
from src.features.psd_extraction import extract_continuous_psd_features
from src.models.sklearn_models import get_sklearn_model_suite

parser = argparse.ArgumentParser(
    description="Test Scikit-Learn Linear and Discriminant models on SSVEP features."
)
parser.add_argument(
    "--subject", help="Subject folder (e.g., S01). Defaults to root processed."
)
args = parser.parse_args()

project_root = Path(__file__).resolve().parents[1]
data_dir = project_root / "data" / "processed"
if args.subject:
  data_dir = data_dir / args.subject

# 1. Load raw time windows and labels
windows_raw = np.load(data_dir / "X_time_windows.npy")
y_raw = np.load(data_dir / "y_labels.npy")

stim_mask = np.isin(y_raw, [101, 102, 103, 104, 105])
y_stim = y_raw[stim_mask]
windows = (
    windows_raw[stim_mask]
    if windows_raw.shape[0] == len(y_raw)
    else windows_raw[: len(y_stim)]
)

print(f"Loaded {windows.shape[0]} windows across {windows.shape[1]} channels.")

# 2. Extract verified features using validation gates
print("\nExtracting feature spaces...")
X_psd, y_psd, mask_psd = extract_continuous_psd_features(
    windows, y_stim, fs=250.0
)
X_fbcca, y_fbcca, mask_fbcca = extract_fbcca_features(windows, y_stim, fs=250.0)
X_fbcsp, y_fbcsp, mask_fbcsp = extract_fbcsp_features(windows, y_stim, fs=250.0)

feature_sets = {
    "PSD (Continuous)": (X_psd, y_psd),
    "FBCCA (5 Priors)": (X_fbcca, y_fbcca),
    "FBCSP (Spatial Log-Var)": (X_fbcsp, y_fbcsp),
}

models = get_sklearn_model_suite()
records = []

print("\n" + "=" * 80)
print("             SCIKIT-LEARN MODEL SUITE - INITIAL FITTING TEST")
print("=" * 80)

for feat_name, (X, y) in feature_sets.items():
  print(f"\n---> Testing Feature Set: {feat_name} | Shape: {X.shape}")

  # Scaling is mandatory for regularized linear models (Ridge, Lasso, ElasticNet)
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)

  for model_name, clf in models.items():
    try:
      clf.fit(X_scaled, y)
      preds = clf.predict(X_scaled)
      acc = accuracy_score(y, preds) * 100.0

      records.append({
          "Feature Representation": feat_name,
          "Dimensions (p)": X.shape[1],
          "Model Section & Name": model_name,
          "Fit Accuracy (%)": f"{acc:.2f}%",
          "Status": "Converged",
      })
      print(f"  [{model_name:30s}] -> Fit Acc: {acc:6.2f}%")
    except Exception as e:
      records.append({
          "Feature Representation": feat_name,
          "Dimensions (p)": X.shape[1],
          "Model Section & Name": model_name,
          "Fit Accuracy (%)": "N/A",
          "Status": f"Error: {str(e)[:30]}",
      })
      print(f"  [{model_name:30s}] -> Failed: {e}")

# Summary Table
df_results = pd.DataFrame(records)
print("\n" + "=" * 80)
print("                           SUMMARY BENCHMARK TABLE")
print("=" * 80)
print(df_results.to_string(index=False))