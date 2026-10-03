"""EEGNet architecture in PyTorch for raw multiclass SSVEP decoding.

Adapted from Lawhern et al. (2018) for inputs of shape (B, 1, C, T).
"""

import torch
import torch.nn as nn


class Conv2dWithConstraint(nn.Conv2d):
  """Conv2d layer with max-norm weight constraint for spatial filters."""

  def __init__(self, *args, max_norm: float = 1.0, **kwargs):
    super().__init__(*args, **kwargs)
    self.max_norm = max_norm

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if self.max_norm is not None:
      with torch.no_grad():
        norm = self.weight.norm(2, dim=(1, 2, 3), keepdim=True)
        desired = torch.clamp(norm, max=self.max_norm)
        self.weight.mul_(desired / (norm + 1e-8))
    return super().forward(x)


class EEGNetSSVEP(nn.Module):
  """Compact EEGNet architecture configured for SSVEP decoding.

  Parameters
  ----------
  n_classes : int
      Number of target classes (default: 5).
  n_channels : int
      Number of EEG scalp channels (default: 7).
  n_samples : int
      Number of time points per window (default: 256 for 1.0s at 250Hz).
  F1 : int
      Number of temporal filters (default: 8).
  D : int
      Depth multiplier for spatial filters per temporal filter (default: 2).
  F2 : int
      Number of pointwise filters in the separable convolution (default: 16).
  kernel_length : int
      Length of the temporal convolution kernel (default: 125, i.e., fs // 2).
  dropout_rate : float
      Dropout probability for regularization (default: 0.25).
  """

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
    super().__init__()
    self.n_classes = n_classes
    self.n_channels = n_channels
    self.n_samples = n_samples

    # =========================================================================
    # Block 1: Temporal Conv -> Depthwise Spatial Conv -> Pool
    # =========================================================================
    self.block1 = nn.Sequential(
        nn.Conv2d(
            in_channels=1,
            out_channels=F1,
            kernel_size=(1, kernel_length),
            padding=(0, kernel_length // 2),
            bias=False,
        ),
        nn.BatchNorm2d(F1),
        Conv2dWithConstraint(
            in_channels=F1,
            out_channels=F1 * D,
            kernel_size=(n_channels, 1),
            groups=F1,
            bias=False,
            max_norm=1.0,
        ),
        nn.BatchNorm2d(F1 * D),
        nn.ELU(),
        nn.AvgPool2d(kernel_size=(1, 4)),
        nn.Dropout(dropout_rate),
    )

    # =========================================================================
    # Block 2: Separable Convolution (Depthwise Temporal + Pointwise 1x1)
    # =========================================================================
    self.block2 = nn.Sequential(
        nn.Conv2d(
            in_channels=F1 * D,
            out_channels=F1 * D,
            kernel_size=(1, 16),
            padding=(0, 8),
            groups=F1 * D,
            bias=False,
        ),
        nn.Conv2d(
            in_channels=F1 * D,
            out_channels=F2,
            kernel_size=(1, 1),
            bias=False,
        ),
        nn.BatchNorm2d(F2),
        nn.ELU(),
        nn.AvgPool2d(kernel_size=(1, 8)),
        nn.Dropout(dropout_rate),
    )

    # Calculate dense input dimension dynamically
    with torch.no_grad():
      dummy = torch.zeros(1, 1, n_channels, n_samples)
      out = self.block2(self.block1(dummy))
      flattened_size = out.view(1, -1).size(1)

    # =========================================================================
    # Classifier Head
    # =========================================================================
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(flattened_size, n_classes),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass.

    Input shape: (B, 1, C, T) or (B, C, T)
    """
    if x.ndim == 3:
      x = x.unsqueeze(1)  # (B, 1, C, T)
    x = self.block1(x)
    x = self.block2(x)
    logits = self.classifier(x)
    return logits