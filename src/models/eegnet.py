"""EEGNet architecture for 5-class SSVEP classification on raw time windows."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class Conv2dWithConstraint(nn.Conv2d):
    """2D convolution with an optional max-norm constraint."""

    def __init__(self, *args, max_norm: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.max_norm is not None:
            with torch.no_grad():
                self.weight.data = torch.renorm(
                    self.weight.data, p=2, dim=0, maxnorm=self.max_norm
                )
        return super().forward(x)


class LinearWithConstraint(nn.Linear):
    """Linear layer with an optional max-norm constraint on weight vectors."""

    def __init__(self, *args, max_norm: float = 0.25, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.max_norm is not None:
            with torch.no_grad():
                self.weight.data = torch.renorm(
                    self.weight.data, p=2, dim=0, maxnorm=self.max_norm
                )
        return super().forward(x)


class EEGNet(nn.Module):
    """Compact EEGNet for raw occipital EEG windows.

    Input shape: (batch, 1, channels=4, samples=250)
    Output shape: (batch, num_classes=5)
    """

    def __init__(
        self,
        num_classes: int = 5,
        n_channels: int = 4,
        n_samples: int = 250,
        fs: float = 250.0,
        f1: int = 8,
        d: int = 2,
        f2: int = 16,
        kernel_length_sec: float = 0.25,
        dropout_rate: float = 0.4,
    ):
        super().__init__()
        kernel_samples = int(fs * kernel_length_sec)

        self.block1 = nn.Sequential(
            nn.Conv2d(
                1,
                f1,
                kernel_size=(1, kernel_samples),
                padding=(0, kernel_samples // 2),
                bias=False,
            ),
            nn.BatchNorm2d(f1),
            Conv2dWithConstraint(
                f1,
                f1 * d,
                kernel_size=(n_channels, 1),
                groups=f1,
                bias=False,
                max_norm=1.0,
            ),
            nn.BatchNorm2d(f1 * d),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout_rate),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(
                f1 * d,
                f1 * d,
                kernel_size=(1, 16),
                padding=(0, 8),
                groups=f1 * d,
                bias=False,
            ),
            nn.Conv2d(f1 * d, f2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout_rate),
        )

        was_training = self.training
        self.eval()
        with torch.no_grad():
            feature_shape = self.block2(
                self.block1(torch.zeros(1, 1, n_channels, n_samples))
            ).shape
        if was_training:
            self.train()

        feat_dim = int(np.prod(feature_shape[1:]))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            LinearWithConstraint(feat_dim, num_classes, max_norm=0.25),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.block2(self.block1(x)))


def prepare_raw_time_tensors(
    X_time: np.ndarray,
    keep_channels: tuple[int, ...] = (1, 4, 5, 6),
) -> np.ndarray:
    """Select POz, O1, Oz, O2 and return shape (N, 1, 4, 250)."""
    X_occ = X_time[:, keep_channels, :]
    return np.expand_dims(X_occ, axis=1)


def train_and_eval_eegnet_cv(
    X_time_tensors: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    epochs: int = 150,
    lr: float = 1e-3,
    batch_size: int = 16,
) -> dict:
    """Evaluate EEGNet with stratified cross-validation."""
    label_mapping = {value: idx for idx, value in enumerate(sorted(np.unique(y)))}
    y_mapped = np.array([label_mapping[value] for value in y])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_accuracies = []
    oof_predictions = np.zeros(len(y), dtype=int)
    oof_probabilities = np.zeros((len(y), len(label_mapping)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_time_tensors, y_mapped), 1):
        X_tr = X_time_tensors[train_idx].copy()
        X_te = X_time_tensors[test_idx].copy()
        y_tr, y_te = y_mapped[train_idx], y_mapped[test_idx]

        n_tr, _, n_ch, n_s = X_tr.shape
        n_te = X_te.shape[0]
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr.reshape(n_tr, -1)).reshape(n_tr, 1, n_ch, n_s)
        X_te = scaler.transform(X_te.reshape(n_te, -1)).reshape(n_te, 1, n_ch, n_s)

        train_ds = TensorDataset(
            torch.tensor(X_tr, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.long),
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        model = EEGNet(
            num_classes=len(label_mapping),
            n_channels=n_ch,
            n_samples=n_s,
        ).to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        model.train()
        for _ in range(epochs):
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(batch_x), batch_y)
                loss.backward()
                optimizer.step()
            scheduler.step()

        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(X_te, dtype=torch.float32).to(device))
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()

        predictions = np.argmax(probabilities, axis=-1)
        oof_predictions[test_idx] = predictions
        oof_probabilities[test_idx] = probabilities
        fold_accuracy = np.mean(predictions == y_te)
        fold_accuracies.append(fold_accuracy)
        print(f"  Fold {fold}: {fold_accuracy * 100:.2f}%")

    overall_accuracy = np.mean(oof_predictions == y_mapped)
    return {
        "fold_accuracies": fold_accuracies,
        "mean_accuracy": overall_accuracy,
        "oof_predictions": oof_predictions,
        "oof_probabilities": oof_probabilities,
    }


if __name__ == "__main__":
    processed_dir = Path("data/processed")
    time_file = processed_dir / "X_time_windows.npy"
    label_file = processed_dir / "y_labels.npy"

    if not time_file.exists():
        raise FileNotFoundError(
            f"Missing {time_file}. Ensure extraction exported raw time slices."
        )

    X_time = np.load(time_file)
    y = np.load(label_file)
    target_conditions = [101, 102, 103, 104, 105]
    mask = np.isin(y, target_conditions)

    X_tensors = prepare_raw_time_tensors(X_time[mask])
    metrics = train_and_eval_eegnet_cv(X_tensors, y[mask])
    y_filtered = y[mask]
    target_conditions = sorted(np.unique(y_filtered))
    print("\n--- EEGNet Out-of-Fold Evaluation ---")
    print(confusion_matrix(
        np.array([target_conditions.index(label) for label in y_filtered]),
        metrics["oof_predictions"],
    ))
    print(classification_report(
        np.array([target_conditions.index(label) for label in y_filtered]),
        metrics["oof_predictions"],
        target_names=[str(condition) for condition in target_conditions],
    ))
    print(f"EEGNet 5-Fold Mean Accuracy: {metrics['mean_accuracy'] * 100:.2f}%")


class EEGNetSSVEP(EEGNet):
    """Compatibility wrapper for the benchmark-facing EEGNet API."""

    def __init__(
        self,
        n_classes: int = 5,
        n_channels: int = 7,
        n_samples: int = 256,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 125,
        dropout_rate: float = 0.25,
    ):
        super().__init__(
            num_classes=n_classes,
            n_channels=n_channels,
            n_samples=n_samples,
            f1=F1,
            d=D,
            f2=F2,
            kernel_length_sec=kernel_length / 250.0,
            dropout_rate=dropout_rate,
        )
