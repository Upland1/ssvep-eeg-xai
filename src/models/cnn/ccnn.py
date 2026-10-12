"""Complex-spectrum CNN for SSVEP classification."""
import numpy as np
import torch
import torch.nn as nn


class CCNN(nn.Module):
  """CNN that classifies the selected complex FFT spectrum."""

  def __init__(
      self,
      n_classes: int = 5,
      n_channels: int = 8,
      n_samples: int = 256,
      fs: float = 256.0,
      f_min: float = 3.0,
      f_max: float = 50.0,
      resolution_hz: float = 0.25,
      n_filters: int | None = None,
      spectral_kernel: int = 10,
      dropout_rate: float = 0.25,
  ):
    super().__init__()
    self.n_samples = n_samples
    self.nfft = max(n_samples, int(round(fs / resolution_hz)))
    freqs = np.fft.rfftfreq(self.nfft, d=1.0 / fs)
    idx = np.flatnonzero((freqs >= f_min) & (freqs <= f_max))
    self.register_buffer("bin_idx", torch.as_tensor(idx, dtype=torch.long))
    self.freqs = freqs[idx]
    nf = n_filters or 2 * n_channels

    self.spatial = nn.Sequential(
        nn.Conv2d(1, nf, kernel_size=(n_channels, 1)),
        nn.BatchNorm2d(nf),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
    )
    self.spectral = nn.Sequential(
        nn.Conv2d(nf, nf, kernel_size=(1, spectral_kernel)),
        nn.BatchNorm2d(nf),
        nn.ReLU(),
        nn.Dropout(dropout_rate),
    )
    with torch.no_grad():
      out = self.spectral(self.spatial(self.spectrum(torch.zeros(1, n_channels, n_samples)).unsqueeze(1)))
    self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(out[0].numel(), n_classes))

  def spectrum(self, x: torch.Tensor) -> torch.Tensor:
    """(B, C, T) time windows -> (B, C, 2 * n_bins) [real | imag] spectrum."""
    z = torch.fft.rfft(x, n=self.nfft, dim=-1).index_select(-1, self.bin_idx) / self.n_samples
    return torch.cat([z.real, z.imag], dim=-1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Input shape: (B, C, T) or (B, 1, C, T)."""
    if x.ndim == 4:
      x = x.squeeze(1)
    x = self.spectrum(x).unsqueeze(1)
    return self.classifier(self.spectral(self.spatial(x)))