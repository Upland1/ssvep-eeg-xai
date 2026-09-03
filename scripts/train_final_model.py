"""Train and export final production model artifacts (DualBranchFusionCNN & Shrinkage-LDA)."""

import argparse
import joblib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from scripts.train_models import (
    FBCCA_TARGET_FREQS,
    extract_all_fbcca_features,
    extract_harmonic_features,
    load_dataset,
)
from src.models.spatial_spectral_cnn import (
    DualBranchFusionCNN,
    prepare_spatial_spectral_tensors,
)


def train_and_export_production_models(
    processed_dir: Path,
    output_dir: Path,
    subject_id: str = "S04",
    epochs: int = 140,
    lr: float = 1e-3,
    batch_size: int = 16,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    subject_processed = processed_dir / subject_id

    if not subject_processed.exists():
        raise FileNotFoundError(
            f"Processed directory not found for subject: {subject_processed}"
        )

    X_psd, X_time, y = load_dataset(subject_processed)
    if X_time is None:
        raise FileNotFoundError("X_time_windows.npy is required to extract FBCCA features.")

    target_classes = sorted(list(np.unique(y)))
    label_to_idx = {value: idx for idx, value in enumerate(target_classes)}
    idx_to_label = {idx: value for value, idx in label_to_idx.items()}
    y_mapped = np.array([label_to_idx[value] for value in y])

    print(f"[{subject_id}] Training final models on full dataset (N={len(y)} windows)...")

    X_psd_tensors = prepare_spatial_spectral_tensors(X_psd)
    cache_fbcca = subject_processed / "X_fbcca_features.npy"
    if cache_fbcca.exists():
        X_fbcca = np.load(cache_fbcca)
    else:
        print(f"[{subject_id}] Computing FBCCA features...")
        X_fbcca = extract_all_fbcca_features(
            X_time, fs=250.0, target_freqs=FBCCA_TARGET_FREQS
        )
        np.save(cache_fbcca, X_fbcca)

    X_harm = extract_harmonic_features(X_psd)
    X_hybrid = np.hstack([X_harm, X_fbcca])

    print(f"[{subject_id}] Fitting Shrinkage-LDA...")
    scaler_lda = StandardScaler()
    X_hybrid_scaled = scaler_lda.fit_transform(X_hybrid)

    lda_model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    lda_model.fit(X_hybrid_scaled, y)

    lda_artifact = {
        "model": lda_model,
        "scaler": scaler_lda,
        "classes": target_classes,
        "subject_id": subject_id,
        "feature_dim": X_hybrid.shape[1],
    }
    lda_path = output_dir / f"{subject_id}_shrinkage_lda.joblib"
    joblib.dump(lda_artifact, lda_path)
    print(f"  -> Shrinkage-LDA artifact saved to: {lda_path}")

    print(f"[{subject_id}] Training DualBranchFusionCNN ({epochs} epochs)...")
    n_samples, channels, height, width = X_psd_tensors.shape

    scaler_psd = StandardScaler()
    X_psd_scaled = scaler_psd.fit_transform(
        X_psd_tensors.reshape(n_samples, -1)
    ).reshape(n_samples, channels, height, width)

    scaler_fbcca = StandardScaler()
    X_fbcca_scaled = scaler_fbcca.fit_transform(X_fbcca)

    train_ds = TensorDataset(
        torch.tensor(X_psd_scaled, dtype=torch.float32),
        torch.tensor(X_fbcca_scaled, dtype=torch.float32),
        torch.tensor(y_mapped, dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualBranchFusionCNN(num_classes=len(target_classes)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    for epoch in range(epochs):
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

    cnn_artifact_path = output_dir / f"{subject_id}_fusion_cnn.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "scaler_psd": scaler_psd,
            "scaler_fbcca": scaler_fbcca,
            "label_to_idx": label_to_idx,
            "idx_to_label": idx_to_label,
            "subject_id": subject_id,
            "architecture": "DualBranchFusionCNN",
        },
        cnn_artifact_path,
    )
    print(f"  -> DualBranchFusionCNN artifact saved to: {cnn_artifact_path}")
    print(f"\n[{subject_id}] Production model training completed successfully.")


def main():
    parser = argparse.ArgumentParser(description="Train and save production SSVEP models.")
    parser.add_argument(
        "--subject", type=str, default="S04", help="Subject ID (e.g., S01, S04)"
    )
    parser.add_argument(
        "--save-dir", type=str, default="outputs/models", help="Output directory"
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    processed_dir = base_dir / "data" / "processed"
    save_dir = base_dir / args.save_dir

    train_and_export_production_models(
        processed_dir=processed_dir,
        output_dir=save_dir,
        subject_id=args.subject,
    )


if __name__ == "__main__":
    main()
