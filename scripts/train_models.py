"""Train and evaluate classifiers using Hybrid (Harmonic PSD + FBCCA) features."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt
from sklearn.cross_decomposition import CCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


TARGET_FREQS = {
    101: 24.0,
    102: 20.0,
    103: 15.0,
    104: 10.9091,
    105: 8.5714,
}
FBCCA_TARGET_FREQS = tuple(TARGET_FREQS.values())


def extract_harmonic_features(
    X_raw,
    n_channels=7,
    freq_start=5.0,
    freq_end=35.0,
    keep_channels=(1, 4, 5, 6),
):
    """Extract SSVEP harmonic power from the four visual channels."""
    freqs = np.linspace(freq_start, freq_end, 31)
    N = X_raw.shape[0]
    X_reshaped = X_raw.reshape(N, n_channels, 31)

    X_spatial = X_reshaped[:, keep_channels, :]

    target_freqs = [8.57, 10.91, 15.0, 20.0, 24.0, 17.14, 21.82, 30.0]
    target_indices = list(dict.fromkeys(np.argmin(np.abs(freqs - f)) for f in target_freqs))

    X_targeted = X_spatial[:, :, target_indices]
    return X_targeted.reshape(N, -1)


def get_subband_filter(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = max(0.01, lowcut / nyq)
    high = min(0.99, highcut / nyq)
    b, a = butter(order, [low, high], btype="bandpass")
    return b, a


def compute_fbcca_correlations(
    eeg_window,
    fs=250.0,
    target_freqs=FBCCA_TARGET_FREQS,
    num_subbands=3,
    nh_max=3,
):
    n_channels, n_samples = eeg_window.shape
    t = np.arange(n_samples) / fs
    num_targets = len(target_freqs)

    subband_lowcuts = [6.0, 14.0, 22.0][:num_subbands]
    highcut = 55.0
    weights = np.array([np.power(n + 1, -1.25) + 0.25 for n in range(num_subbands)])

    final_scores = np.zeros(num_targets)

    for sb_idx, lowcut in enumerate(subband_lowcuts):
        b, a = get_subband_filter(lowcut, highcut, fs)
        filtered_eeg = filtfilt(b, a, eeg_window, axis=-1)

        sb_corrs = []
        for f in target_freqs:
            valid_nh = max(1, min(nh_max, int(highcut // f)))
            Y = []
            for h in range(1, valid_nh + 1):
                Y.append(np.sin(2 * np.pi * h * f * t))
                Y.append(np.cos(2 * np.pi * h * f * t))
            Y = np.array(Y)

            cca = CCA(n_components=1)
            cca.fit(filtered_eeg.T, Y.T)
            x_c, y_c = cca.transform(filtered_eeg.T, Y.T)
            r = np.corrcoef(x_c[:, 0], y_c[:, 0])[0, 1]
            sb_corrs.append(0.0 if np.isnan(r) else r)

        final_scores += weights[sb_idx] * (np.array(sb_corrs) ** 2)

    return final_scores


def extract_all_fbcca_features(X_time, fs=250.0, target_freqs=FBCCA_TARGET_FREQS):
    fbcca_features = []
    for i in range(len(X_time)):
        scores = compute_fbcca_correlations(X_time[i], fs=fs, target_freqs=target_freqs)
        fbcca_features.append(scores)
    return np.array(fbcca_features)


def load_dataset(processed_dir: Path, target_classes=(101, 102, 103, 104, 105)):
    X_psd = np.load(processed_dir / "X_psd_features.npy")
    y = np.load(processed_dir / "y_labels.npy")
    time_file = processed_dir / "X_time_windows.npy"
    X_time = np.load(time_file) if time_file.exists() else None

    mask = np.isin(y, target_classes)
    return (X_psd[mask], X_time[mask] if X_time is not None else None, y[mask])


def plot_hybrid_confusion_matrix(
    X,
    y,
    output_path: Path,
    target_classes=(101, 102, 103, 104, 105),
):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_true, all_pred = [], []

    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        clf.fit(X_tr, y_tr)
        all_true.extend(y_te)
        all_pred.extend(clf.predict(X_te))

    cm = confusion_matrix(all_true, all_pred, labels=target_classes)
    class_labels = [
        f"{class_id}\n({TARGET_FREQS[class_id]:g}Hz)"
        for class_id in target_classes
    ]

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
    )
    plt.title("Shrinkage-LDA Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {output_path}")


def run_evaluation(X, y, subject_id="S01", n_splits=5):
    unique_classes = sorted(np.unique(y))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    models = {
        "Shrinkage-LDA": LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        "Logistic-Regression": LogisticRegression(max_iter=1000, C=1.0),
        "Linear-SVM (C=1.0)": SVC(kernel="linear", C=1.0),
        "RBF-SVM (C=2.0)": SVC(kernel="rbf", C=2.0, gamma="scale"),
        "Random-Forest": RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42),
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

    return results


def run_temporal_integration_evaluation(X, y, n_splits=5, window_size=2):
    """Evaluate causal moving-average decisions using out-of-fold LDA probabilities."""
    unique_classes = np.sort(np.unique(y))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_probabilities = np.zeros((len(y), len(unique_classes)))

    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])

        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        clf.fit(X_train, y[train_idx])
        probabilities = clf.predict_proba(X_test)

        for class_idx, class_id in enumerate(clf.classes_):
            target_idx = np.where(unique_classes == class_id)[0][0]
            oof_probabilities[test_idx, target_idx] = probabilities[:, class_idx]

    integrated_probabilities = np.zeros_like(oof_probabilities)
    for sample_idx in range(len(y)):
        start_idx = max(0, sample_idx - window_size + 1)
        integrated_probabilities[sample_idx] = np.mean(
            oof_probabilities[start_idx : sample_idx + 1], axis=0
        )

    predictions = unique_classes[np.argmax(integrated_probabilities, axis=1)]
    return accuracy_score(y, predictions)


def main():
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "data" / "processed"

    X_psd, X_time, y = load_dataset(data_dir)
    all_summary = []

    if X_time is not None:
        print("Extracting Harmonic PSD features (32 dimensions)...")
        X_harm = extract_harmonic_features(X_psd)

        print("Computing FBCCA features (5 dimensions)...")
        X_fbcca = extract_all_fbcca_features(X_time, fs=250.0)

        # Build Hybrid Matrix (32 harmonic PSD + 5 FBCCA features)
        X_hybrid = np.hstack([X_harm, X_fbcca])
        print(f"Hybrid feature matrix shape: {X_hybrid.shape}")

        plot_hybrid_confusion_matrix(
            X_hybrid,
            y,
            output_path=base_dir / "outputs" / "figures" / "confusion_matrices" / "cm_shrinkage_lda_hybrid.png",
        )

        print("Evaluating Supervised Classifiers on Hybrid (PSD + FBCCA) features...")
        hybrid_results = run_evaluation(X_hybrid, y, subject_id="S01")
        for res in hybrid_results:
            res["Classifier"] = f"Hybrid + {res['Classifier']}"
            all_summary.append(res)

        integrated_accuracy = run_temporal_integration_evaluation(
            X_hybrid, y, window_size=2
        )
        print(
            f"Shrinkage-LDA 2-window temporal integration accuracy: "
            f"{integrated_accuracy * 100:.2f}%"
        )

    df = pd.DataFrame(all_summary)
    print("\n" + "=" * 95)
    print("      SSVEP HYBRID (HARMONIC PSD + FBCCA) MODEL BENCHMARK (SUBJECT S01)")
    print("=" * 95)
    print(df.to_string(index=False))
    print("=" * 95 + "\n")


if __name__ == "__main__":
    main()