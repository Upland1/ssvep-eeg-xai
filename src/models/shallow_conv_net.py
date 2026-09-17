"""ShallowConvNet architecture for EEG classification, in PyTorch.

Reference: Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J.,
Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F., Burgard, W.,
& Ball, T. (2017). "Deep learning with convolutional neural networks for
EEG decoding and visualization." Human Brain Mapping, 38(11), 5391-5420.

Distinct from both EEGNet and Compact-CNN in its nonlinearity: instead of
ELU, ShallowConvNet uses a square -> mean-pool -> log sequence, which
approximates computing the log-band-power of the filtered signal in each
pooling window -- conceptually close to what FBCSP's log-variance feature
does, but learned end-to-end via backprop instead of a fixed
eigendecomposition. It's a general-purpose EEG decoding baseline (not
SSVEP-specific like Compact-CNN), included here as a widely-cited
reference point from the broader BCI literature. This is a
faithful-in-spirit re-implementation of the paper's stage structure and
default filter/kernel sizes; it was not re-derived from the original
source code.
"""

import torch
import torch.nn as nn


def safe_log(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
  """Log nonlinearity with a numerical floor (square-pool output is >= 0,
  but can be exactly 0, where an unclamped log would be -inf/NaN)."""
  return torch.log(torch.clamp(x, min=eps))


class Square(nn.Module):
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x ** 2


class ShallowConvNet(nn.Module):
  """ShallowConvNet for EEG decoding (Schirrmeister et al., 2017).

  Parameters
  ----------
  n_classes : int
      Number of target classes (default: 5).
  n_channels : int
      Number of EEG channels (default: 7; works with any channel count --
      the spatial filter always collapses the channel axis to 1, so this
      model is unaffected by how many channels survive the quality gate).
  n_samples : int
      Number of time points per window (default: 256).
  F1 : int
      Number of temporal AND spatial filters (default: 40, per the
      original paper -- the two stages share this width).
  temporal_kernel_length : int
      Length of the first (temporal) conv kernel (default: 25 samples,
      i.e. 100 ms at 250 Hz, per the paper).
  pool_kernel_length : int
      Width of the mean-pooling window applied after the square
      nonlinearity (default: 75 samples, i.e. 300 ms at 250 Hz).
  pool_stride : int
      Stride of the mean-pooling window (default: 15 samples, i.e. 60 ms).
  dropout_rate : float
      Dropout probability before the classifier (default: 0.5, per the
      original paper).
  """

  def __init__(
      self,
      n_classes: int = 5,
      n_channels: int = 7,
      n_samples: int = 256,
      F1: int = 40,
      temporal_kernel_length: int = 25,
      pool_kernel_length: int = 75,
      pool_stride: int = 15,
      dropout_rate: float = 0.5,
  ):
    super().__init__()
    self.n_classes = n_classes
    self.n_channels = n_channels
    self.n_samples = n_samples

    # Guard against windows shorter than the paper's defaults (which were
    # tuned for ~4s trials at 250 Hz) -- clamp so the network never asks
    # for a kernel/pool larger than the data it's given.
    temporal_kernel_length = max(3, min(temporal_kernel_length, n_samples))
    conv_out_len = n_samples - temporal_kernel_length + 1
    pool_kernel_length = max(2, min(pool_kernel_length, conv_out_len))
    pool_stride = max(1, min(pool_stride, pool_kernel_length))

    # Stage 1: temporal conv, then spatial conv across ALL channels at
    # once (kernel = (C,1), no bias) -- collapses the channel axis to 1,
    # same role as CSP's spatial projection but learned end-to-end.
    self.temporal_conv = nn.Conv2d(1, F1, kernel_size=(1, temporal_kernel_length), bias=False)
    self.spatial_conv = nn.Conv2d(F1, F1, kernel_size=(n_channels, 1), bias=False)
    self.bn = nn.BatchNorm2d(F1)

    # Stage 2: square -> mean-pool -> log, approximating log-band-power
    # per pooling window (the paper's signature nonlinearity, distinct
    # from EEGNet/Compact-CNN's ELU).
    self.square = Square()
    self.pool = nn.AvgPool2d(kernel_size=(1, pool_kernel_length), stride=(1, pool_stride))
    self.dropout = nn.Dropout(dropout_rate)

    with torch.no_grad():
      dummy = torch.zeros(1, 1, n_channels, n_samples)
      out = self._features(dummy)
      flattened_size = out.view(1, -1).size(1)

    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(flattened_size, n_classes),
    )

  def _features(self, x: torch.Tensor) -> torch.Tensor:
    x = self.temporal_conv(x)
    x = self.spatial_conv(x)
    x = self.bn(x)
    x = self.square(x)
    x = self.pool(x)
    x = safe_log(x)
    x = self.dropout(x)
    return x

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass.

    Input shape: (B, 1, C, T) or (B, C, T)
    """
    if x.ndim == 3:
      x = x.unsqueeze(1)  # (B, 1, C, T)
    x = self._features(x)
    logits = self.classifier(x)
    return logits