"""Utilities for loading and validating EBR EEG files.

This module wraps the binary EEG loader used in the original project scripts.
The goal is to keep the raw file parsing logic isolated from preprocessing,
feature extraction, and modeling code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np


def load_ebr_file(path: str | Path) -> Dict[str, Any]:
    """Load an EBR file from disk.

    The current implementation delegates to the project-specific loader module,
    which is expected to be available in the environment or repository root.
    """
    import ebr_file

    return ebr_file.load_ebr_file(str(path))


def resolve_recording_path(raw_root: str | Path, session: str, filename: str) -> Path:
    """Build the canonical path for a raw recording file."""
    return Path(raw_root) / session / filename


def select_scalp_channels(channels: list[str], scalp_names: list[str]) -> list[int]:
    """Return indices for the chosen scalp EEG channels."""
    return [i for i, name in enumerate(channels) if name in scalp_names]


def get_mark_channel_index(channels: list[str], mark_name: str = "MARK") -> int:
    """Return the index of the MARK channel, or -1 if unavailable."""
    if mark_name in channels:
        return channels.index(mark_name)
    return -1


def extract_scalp_signal(recording: Dict[str, Any], scalp_names: list[str]) -> tuple[np.ndarray, list[str]]:
    """Extract the EEG matrix for a single trial and band.

    Returns a 2D array of shape (n_channels, n_samples) and the corresponding
    channel labels.
    """
    raw_data = recording["data"]
    channels = recording["channels"]
    eeg_indices = select_scalp_channels(channels, scalp_names)
    eeg_matrix = raw_data[0, eeg_indices, 0, :]
    return eeg_matrix, [channels[i] for i in eeg_indices]
