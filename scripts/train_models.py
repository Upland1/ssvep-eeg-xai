"""Train and evaluate classifiers using targeted harmonic SSVEP features."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, recall_score


def extract_harmonic_features(X_raw, n_channels=7, freq_start=5.0, freq_end=35.0):
    """
    Extracts power specifically at the known SSVEP fundamentals and harmonics,
    reducing 217 dimensions down to pure signal bins.
    """
    # reconstruct frequency vector corresponding to 31 bins
    freqs = np.linspace(freq_start, freq_end, 31)
    
    # reshape X from (N, 217) back to (N, n_channels, 31)
    N = X_raw.shape[0]
    X_reshaped = X_raw.reshape(N, n_channels, 31)
    
    # SSVEP target frequencies and second harmonics
    target_freqs = [8.57, 10.91, 15.0, 20.0, 24.0, 17.14, 21.82, 30.0]
    
    # find closest frequency bin indices
    target_indices = [np.argmin(np.abs(freqs - f)) for f in target_freqs]
    target_indices = sorted(list(set(target_indices)))
    
    # select only targeted bins across all channels
    # shape: (N, n_channels, len(target_indices))
    X_targeted = X_reshaped[:, :, target_indices]
    
    # flatten to feature vector
    X_features = X_targeted.reshape(N, -1)
    return X_features

def compute_cca_correlations(eeg_window, fs=250.0, target_freqs=[8.5714, 10.9091, 15.0, 20.0, 24.0], nh=4):
    """
    eeg_window: shape (n_channels, n_samples)
    Returns: correlation coefficient for each target frequency
    """
    n_channels, n_samples = eeg_window.shape
    t = np.arange(n_samples) / fs
    correlations = []
    
    for f in target_freqs:
        # construct reference matrix Y: 2 * Nh rows
        Y = []
        for h in range(1, nh + 1):
            Y.append(np.sin(2 * np.pi * h * f * t))
            Y.append(np.cos(2 * np.pi * h * f * t))
        Y = np.array(Y)
        
        # CCA between (n_samples, n_channels) and (n_samples, 2*Nh)
        cca = CCA(n_components=1)
        cca.fit(eeg_window.T, Y.T)
        x_c, y_c = cca.transform(eeg_window.T, Y.T)
        r = np.corrcoef(x_c[:, 0], y_c[:, 0])[0, 1]
        correlations.append(r)
        
    return np.array(correlations)

def load_dataset(processed_dir: Path, target_classes=(101, 102, 103, 104, 105)):
    X = np.load(processed_dir / "X_psd_features.npy")
    y = np.load(processed_dir / "y_labels.npy")

    mask = np.isin(y, target_classes)
    return X[mask], y[mask]


def run_evaluation(X, y, subject_id="S01", n_splits=5):
    unique_classes = sorted(np.unique(y))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Linear Discriminant Analysis (LDA) and tuned SVMs are classical BCI standards
    models = {
        "LDA": LinearDiscriminantAnalysis(),
        "Linear-SVM (C=0.5)": SVC(kernel="linear", C=0.5),
        "RBF-SVM (C=2.0)": SVC(kernel="rbf", C=2.0, gamma="scale"),
        "Random-Forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        "MLP (32, 16)": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1500, alpha=0.1, random_state=42)
    }

    results = []

    for name, clf in models.items():
        all_true = []
        all_pred = []

        for train_idx, test_idx in skf.split(X, y):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)

            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_te)

            all_true.extend(y_te)
            all_pred.extend(preds)

        all_true = np.array(all_true)
        all_pred = np.array(all_pred)

        acc = accuracy_score(all_true, all_pred)
        recalls = recall_score(all_true, all_pred, labels=unique_classes, average=None)
        recall_str = " | ".join([f"{cls}: {r*100:.1f}%" for cls, r in zip(unique_classes, recalls)])

        results.append({
            "Subject": subject_id,
            "Classifier": name,
            "Accuracy": f"{acc * 100:.2f}%",
            "Recall per Class": recall_str
        })

    return pd.DataFrame(results)


def main():
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "data" / "processed"

    X_raw, y = load_dataset(data_dir)
    print(f"Original feature shape: {X_raw.shape}")

    # extract narrow-band harmonic features
    X_targeted = extract_harmonic_features(X_raw)
    print(f"Targeted harmonic feature shape: {X_targeted.shape}")

    df = run_evaluation(X_targeted, y)

    print("\n" + "=" * 90)
    print("   5-FOLD CV WITH TARGETED HARMONIC FEATURES (SUBJECT S01)")
    print("=" * 90)
    print(df.to_string(index=False))
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()