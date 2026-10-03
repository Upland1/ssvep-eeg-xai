"""Dual-Branch Fusion CNN for 5-Class SSVEP Classification.

Fuses:
  Branch 1 (Spatial-Spectral): 2D convolutions over Occipital Harmonics (N, 1, 4, 8)
  Branch 2 (Correlation Priors): Dense projection over 5-dim FBCCA Scores (N, 5)
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from scripts.train_models import (
    FBCCA_TARGET_FREQS,
    extract_all_fbcca_features,
    load_dataset,
)


class DualBranchFusionCNN(nn.Module):
    """Fuses 2D spatial-spectral harmonic representations with FBCCA correlation vectors."""

    def __init__(
        self,
        num_classes: int = 5,
        n_channels: int = 4,
        n_harmonics: int = 8,
        n_fbcca: int = 5,
        dropout_rate: float = 0.3,
    ):
        super().__init__()

        # Branch 1: Spatial-Spectral 2D CNN
        # Spatial filtering across channels: (B, 1, 4, 8) -> (B, 16, 1, 8)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(n_channels, 1), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Dropout2d(dropout_rate),
        )

        # Spectral filtering across harmonic bins: (B, 16, 1, 8) -> (B, 32, 1, 6)
        self.spectral_conv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(1, 3), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Dropout2d(dropout_rate),
        )

        self.branch_psd_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 1 * (n_harmonics - 2), 32),
            nn.ELU(),
            nn.Dropout(dropout_rate),
        )

        # Branch 2: FBCCA Correlation Prior MLP
        self.branch_fbcca_fc = nn.Sequential(
            nn.Linear(n_fbcca, 16),
            nn.BatchNorm1d(16),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(16, 16),
            nn.ELU(),
        )

        # Fusion & Classification Head: 32 (PSD) + 16 (FBCCA) = 48 dims
        self.classifier = nn.Sequential(
            nn.Linear(32 + 16, 32),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, num_classes),
        )

    def forward(self, x_psd: torch.Tensor, x_fbcca: torch.Tensor) -> torch.Tensor:
        # Branch 1 forward
        out_psd = self.spatial_conv(x_psd)
        out_psd = self.spectral_conv(out_psd)
        feat_psd = self.branch_psd_fc(out_psd)

        # Branch 2 forward
        feat_fbcca = self.branch_fbcca_fc(x_fbcca)

        # Multimodal fusion
        fused = torch.cat([feat_psd, feat_fbcca], dim=1)
        return self.classifier(fused)


def prepare_spatial_spectral_tensors(
    X_raw: np.ndarray,
    keep_channels: tuple[int, ...] = (1, 4, 5, 6),
    freq_start: float = 5.0,
    freq_end: float = 35.0,
) -> np.ndarray:
    """Extract and reshape occipital harmonic power into (N, 1, 4, 8) tensors."""
    freqs = np.linspace(freq_start, freq_end, 31)
    n_samples = X_raw.shape[0]
    X_spatial = X_raw.reshape(n_samples, 7, 31)[:, keep_channels, :]

    target_freqs = [8.57, 10.91, 15.0, 20.0, 24.0, 17.14, 21.82, 30.0]
    target_indices = [np.argmin(np.abs(freqs - f)) for f in target_freqs]

    X_tensors = X_spatial[:, :, target_indices]  # Shape: (N, 4, 8)
    return np.expand_dims(X_tensors, axis=1)    # Shape: (N, 1, 4, 8)


def train_and_eval_fusion_cnn_cv(
    X_psd_tensor: np.ndarray,
    X_fbcca: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    epochs: int = 140,
    lr: float = 1e-3,
    batch_size: int = 16,
) -> dict:
    """Evaluate the Dual-Branch CNN with Stratified 5-Fold Cross-Validation."""
    label_mapping = {val: idx for idx, val in enumerate(sorted(np.unique(y)))}
    y_mapped = np.array([label_mapping[val] for val in y])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_accuracies = []
    oof_predictions = np.zeros(len(y), dtype=int)
    oof_probabilities = np.zeros((len(y), len(label_mapping)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_psd_tensor, y_mapped), 1):
        X_psd_tr = X_psd_tensor[train_idx].copy()
        X_psd_te = X_psd_tensor[test_idx].copy()
        X_fbcca_tr = X_fbcca[train_idx].copy()
        X_fbcca_te = X_fbcca[test_idx].copy()
        y_tr, y_te = y_mapped[train_idx], y_mapped[test_idx]

        # Standardize PSD branch
        n_tr, c, h, w = X_psd_tr.shape
        n_te = X_psd_te.shape[0]
        scaler_psd = StandardScaler()
        X_psd_tr = scaler_psd.fit_transform(X_psd_tr.reshape(n_tr, -1)).reshape(n_tr, c, h, w)
        X_psd_te = scaler_psd.transform(X_psd_te.reshape(n_te, -1)).reshape(n_te, c, h, w)

        # Standardize FBCCA branch
        scaler_fbcca = StandardScaler()
        X_fbcca_tr = scaler_fbcca.fit_transform(X_fbcca_tr)
        X_fbcca_te = scaler_fbcca.transform(X_fbcca_te)

        train_ds = TensorDataset(
            torch.tensor(X_psd_tr, dtype=torch.float32),
            torch.tensor(X_fbcca_tr, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.long),
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        model = DualBranchFusionCNN(num_classes=len(label_mapping)).to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        model.train()
        for _ in range(epochs):
            for batch_psd, batch_fbcca, batch_y in train_loader:
                batch_psd = batch_psd.to(device)
                batch_fbcca = batch_fbcca.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                out = model(batch_psd, batch_fbcca)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()
            scheduler.step()

        model.eval()
        with torch.no_grad():
            x_psd_eval = torch.tensor(X_psd_te, dtype=torch.float32).to(device)
            x_fbcca_eval = torch.tensor(X_fbcca_te, dtype=torch.float32).to(device)
            logits = model(x_psd_eval, x_fbcca_eval)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1)

            oof_predictions[test_idx] = preds
            oof_probabilities[test_idx] = probs
            acc = float(np.mean(preds == y_te))
            fold_accuracies.append(acc)
            print(f"  Fold {fold}: {acc * 100:.2f}%")

    overall_acc = float(np.mean(oof_predictions == y_mapped))
    return {
        "fold_accuracies": fold_accuracies,
        "mean_accuracy": overall_acc,
        "oof_predictions": oof_predictions,
        "oof_probabilities": oof_probabilities,
        "y_mapped": y_mapped,
        "target_names": [str(k) for k in sorted(np.unique(y))],
    }


def evaluate_temporal_smoothing(y_true: np.ndarray, oof_probs: np.ndarray, window_size: int = 2) -> float:
    """Evaluate causal moving-average temporal smoothing over predicted probabilities."""
    smoothed_probs = np.zeros_like(oof_probs)
    for i in range(len(y_true)):
        start_idx = max(0, i - window_size + 1)
        smoothed_probs[i] = np.mean(oof_probs[start_idx : i + 1], axis=0)
    preds = np.argmax(smoothed_probs, axis=-1)
    return float(np.mean(preds == y_true))


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[2]
    processed_dir = base_dir / "data" / "processed"

    X_psd, X_time, y = load_dataset(processed_dir)
    if X_time is None:
        raise FileNotFoundError("X_time_windows.npy is required to compute FBCCA features.")

    # 1. Prepare Branch 1 (Spatial-Spectral Harmonics)
    X_psd_tensors = prepare_spatial_spectral_tensors(X_psd)

    # 2. Prepare Branch 2 (FBCCA Features)
    cache_fbcca = processed_dir / "X_fbcca_features.npy"
    if cache_fbcca.exists():
        X_fbcca = np.load(cache_fbcca)
    else:
        print("Extracting FBCCA features across raw time windows...")
        X_fbcca = extract_all_fbcca_features(X_time, fs=250.0, target_freqs=FBCCA_TARGET_FREQS)
        np.save(cache_fbcca, X_fbcca)

    # 3. Train & Evaluate
    metrics = train_and_eval_fusion_cnn_cv(X_psd_tensors, X_fbcca, y)

    smoothed_acc = evaluate_temporal_smoothing(
        metrics["y_mapped"], metrics["oof_probabilities"], window_size=2
    )

    print(f"\nDual-Branch Fusion CNN 5-Fold Mean Accuracy: {metrics['mean_accuracy'] * 100:.2f}%")
    print(f"Dual-Branch Fusion CNN 2-Window Smoothed Accuracy: {smoothed_acc * 100:.2f}%")
    print("\n--- Out-of-Fold Confusion Matrix ---")
    print(confusion_matrix(metrics["y_mapped"], metrics["oof_predictions"]))
    print(
        classification_report(
            metrics["y_mapped"],
            metrics["oof_predictions"],
            target_names=metrics["target_names"],
        )
    )