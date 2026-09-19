"""Simple, standard EEG-window data augmentation for SSVEP training.

Applied ONLY inside the training DataLoader, on-the-fly, via
`SSVEPAugmentedDataset` -- validation/test data is never touched, so
nothing here can leak information across a train/test split. All four
methods act per-window on a (channels, samples) array and are standard,
simple choices in the EEG-augmentation literature -- deliberately chosen
over generative/learned augmentation for a first pass.

Methods
-------
- Gaussian noise: adds noise scaled to a fraction of the window's own
  per-channel std, simulating sensor/thermal noise.
- Amplitude scaling: multiplies the whole window by one random factor
  near 1.0, simulating electrode impedance / gain variation.
- Time shift: circularly shifts the window a few samples along time,
  simulating jitter in stimulus-onset alignment.
- Intra-class mixup: linearly interpolates two windows from the SAME
  class only. Mixing across classes would blend two different SSVEP
  stimulation frequencies into a signal that matches neither label, so
  this is restricted to same-class pairs by construction, not just by
  convention.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


def add_gaussian_noise(
    window: np.ndarray, std_frac: float = 0.05, rng: np.random.Generator | None = None
) -> np.ndarray:
  """Add per-channel Gaussian noise scaled to a fraction of that channel's own std."""
  rng = rng or np.random.default_rng()
  ch_std = window.std(axis=-1, keepdims=True)
  noise = rng.normal(0.0, std_frac, size=window.shape) * ch_std
  return window + noise


def scale_amplitude(
    window: np.ndarray,
    scale_range: tuple[float, float] = (0.9, 1.1),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
  """Multiply the whole window by one random scalar factor near 1.0."""
  rng = rng or np.random.default_rng()
  factor = rng.uniform(scale_range[0], scale_range[1])
  return window * factor


def time_shift(
    window: np.ndarray, max_shift_frac: float = 0.1, rng: np.random.Generator | None = None
) -> np.ndarray:
  """Circularly shift the window a few samples along the time axis."""
  rng = rng or np.random.default_rng()
  n_samples = window.shape[-1]
  max_shift = max(1, int(round(n_samples * max_shift_frac)))
  shift = int(rng.integers(-max_shift, max_shift + 1))
  if shift == 0:
    return window
  return np.roll(window, shift, axis=-1)


def mixup_intra_class(
    window_a: np.ndarray,
    window_b: np.ndarray,
    alpha: float = 0.4,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
  """Linearly interpolate two windows -- caller must ensure they share a class.

  Uses a Beta(alpha, alpha) mixing weight (the standard mixup recipe),
  reflected toward 0.5 so the result always leans toward whichever input
  it's closer to rather than averaging every single time.
  """
  rng = rng or np.random.default_rng()
  lam = float(rng.beta(alpha, alpha)) if alpha > 0 else 1.0
  lam = max(lam, 1.0 - lam)
  return lam * window_a + (1.0 - lam) * window_b


class SSVEPAugmentedDataset(Dataset):
  """Training-only Dataset wrapper: applies random augmentation on-the-fly.

  Wrap the TRAINING fold's arrays with this -- never validation/test.
  Each call to __getitem__ independently rolls the dice on whether to
  apply each augmentation, so every epoch sees a slightly different
  version of the data without duplicating anything on disk or in memory.

  Parameters
  ----------
  X : np.ndarray, shape (n_windows, n_channels, n_samples)
      Already per-window standardized, same as the un-augmented pipeline.
  y : np.ndarray, shape (n_windows,)
      Integer class labels.
  p_noise, p_scale, p_shift, p_mixup : float
      Independent probability of applying each augmentation to a given
      sample on a given draw (default 0.5 each).
  noise_std_frac, scale_range, shift_frac, mixup_alpha :
      Passed through to the corresponding function above.
  seed : int | None
      Optional seed for reproducibility (e.g. vary per CV fold).
  """

  def __init__(
      self,
      X: np.ndarray,
      y: np.ndarray,
      p_noise: float = 0.5,
      p_scale: float = 0.5,
      p_shift: float = 0.5,
      p_mixup: float = 0.5,
      noise_std_frac: float = 0.05,
      scale_range: tuple[float, float] = (0.9, 1.1),
      shift_frac: float = 0.1,
      mixup_alpha: float = 0.4,
      seed: int | None = None,
  ):
    self.X = X
    self.y = y
    self.p_noise = p_noise
    self.p_scale = p_scale
    self.p_shift = p_shift
    self.p_mixup = p_mixup
    self.noise_std_frac = noise_std_frac
    self.scale_range = scale_range
    self.shift_frac = shift_frac
    self.mixup_alpha = mixup_alpha
    self.rng = np.random.default_rng(seed)

    # Precompute class -> sample-index lookup once, so mixup can draw a
    # same-class partner in O(1) instead of scanning `y` every call.
    self._by_class: dict = {cls: np.flatnonzero(y == cls) for cls in np.unique(y)}

  def __len__(self) -> int:
    return len(self.X)

  def __getitem__(self, idx: int):
    window = self.X[idx].copy()
    label = self.y[idx]

    if self.p_mixup > 0 and self.rng.random() < self.p_mixup:
      same_class_idx = self._by_class[label]
      if len(same_class_idx) > 1:
        partner_idx = self.rng.choice(same_class_idx)
        window = mixup_intra_class(
            window, self.X[partner_idx], alpha=self.mixup_alpha, rng=self.rng
        )

    if self.p_scale > 0 and self.rng.random() < self.p_scale:
      window = scale_amplitude(window, scale_range=self.scale_range, rng=self.rng)

    if self.p_shift > 0 and self.rng.random() < self.p_shift:
      window = time_shift(window, max_shift_frac=self.shift_frac, rng=self.rng)

    if self.p_noise > 0 and self.rng.random() < self.p_noise:
      window = add_gaussian_noise(window, std_frac=self.noise_std_frac, rng=self.rng)

    return (
        torch.tensor(window, dtype=torch.float32),
        torch.tensor(label, dtype=torch.long),
    )