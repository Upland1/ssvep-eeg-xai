"""EEG filtering utilities."""

from __future__ import annotations

import numpy as np
from scipy.signal import filtfilt, iirfilter


def apply_iir_bandpass(data: np.ndarray, fs: float, lowcut: float = 1.0, highcut: float = 60.0, order: int = 4) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter along the time axis."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = iirfilter(order, [low, high], btype="bandpass", ftype="butter")
    return filtfilt(b, a, data, axis=-1)
