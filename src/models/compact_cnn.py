"""Compact-CNN architecture for SSVEP classification, in PyTorch.

Reference: Waytowich, N., Lawhern, V. J., Garcia, J. O., Cummings, J.,
Faller, J., Sajda, P., & Vettel, J. M. (2018). "Compact Convolutional
Neural Networks for Classification of Asynchronous Steady-State Visual
Evoked Potentials." Journal of Neural Engineering, 15(6).

Unlike EEGNet (designed for broadband ERP/motor-imagery patterns via a
depthwise-separable temporal-then-spatial-then-separable pipeline),
Compact-CNN was designed specifically for SSVEP's narrowband oscillatory
structure: just two conv layers -- a spatial filter across all channels,
then a temporal filter whose kernel length is tied to the lowest
stimulation frequency in the paradigm, so it can span at least one full
cycle of the slowest target frequency. This is a faithful-in-spirit
re-implementation matching the paper's stage structure and the sizing
rule for the temporal kernel; exact hyperparameters (channel/temporal
filter counts, dropout) follow the paper's reported defaults but were not
re-derived from the original source code.
"""

import torch
import torch.nn as nn


class CompactCNN(nn.Module):
  """Compact CNN for SSVEP decoding (Waytowich et al., 2018).

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
  fs : float
      Sampling rate in Hz, used to size the temporal kernel.
  lowest_freq_hz : float
      Lowest SSVEP target frequency in the paradigm (default 8.5714 Hz,
      this project's slowest target). The temporal kernel length is set
      to round(fs / lowest_freq_hz) samples, so it spans at least one
      full cycle of the slowest oscillation the model needs to resolve.
  F1 : int
      Number of spatial filters (default 16, per the original paper).
  F2 : int
      Number of temporal filters (default 32, per the original paper).
  dropout_rate : float
      Dropout probability after the temporal conv (default 0.5, per the
      original paper -- higher than EEGNet's 0.25 since this network has
      far fewer, wider layers and a larger flattened head).
  """

  def __init__(
      self,
      n_classes: int = 5,
      n_channels: int = 7,
      n_samples: int = 256,
      fs: float = 250.0,
      lowest_freq_hz: float = 8.5714,
      F1: int = 16,
      F2: int = 32,
      dropout_rate: float = 0.5,
  ):
    super().__init__()
    self.n_classes = n_classes
    self.n_channels = n_channels
    self.n_samples = n_samples

    kernel_length = max(4, int(round(fs / lowest_freq_hz)))
    kernel_length = min(kernel_length, n_samples)  # never exceed the window
    self.kernel_length = kernel_length

    # Stage 1: spatial filter across ALL channels at once (kernel = (C,1)).
    # Collapses the channel axis to 1, analogous to a learned spatial
    # filter/beamformer -- similar intent to CSP's spatial projection, but
    # learned end-to-end instead of via generalized eigendecomposition.
    self.spatial_conv = nn.Sequential(
        nn.Conv2d(1, F1, kernel_size=(n_channels, 1), bias=False),
        nn.BatchNorm2d(F1),
    )

    # Stage 2: temporal filter sized to capture at least one full cycle of
    # the lowest SSVEP target frequency, isolating the narrowband
    # oscillatory structure SSVEP relies on (unlike EEGNet's broadband
    # kernel_length = fs // 2 convention).
    pad = kernel_length // 2
    self.temporal_conv = nn.Sequential(
        nn.Conv2d(F1, F2, kernel_size=(1, kernel_length), padding=(0, pad), bias=False),
        nn.BatchNorm2d(F2),
        nn.ELU(),
        nn.Dropout(dropout_rate),
    )

    # Calculate dense input dimension dynamically (mirrors eegnet.py's
    # own pattern in this codebase).
    with torch.no_grad():
      dummy = torch.zeros(1, 1, n_channels, n_samples)
      out = self.temporal_conv(self.spatial_conv(dummy))
      flattened_size = out.view(1, -1).size(1)

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
    x = self.spatial_conv(x)
    x = self.temporal_conv(x)
    logits = self.classifier(x)
    return logits