"""Utilities for loading, parsing, and validating EBR/RAW EEG files directly."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def load_ebr_file(file: str | Path) -> Dict[str, Any]:
    """Load a binary EBR file directly into a structured dictionary."""
    file_str = str(file)
    if not os.path.exists(file_str):
        raise FileNotFoundError(f"The specified path does not exist: {file_str}")

    with open(file_str, "rb") as data_file:
        magic = data_file.readline().strip().lower()
        if magic != b"ebr binary 1.0":
            raise ValueError(f"File {file_str} is not a valid binary EBR file (header: {magic!r}).")

        data_type = "double"
        fs = 0.0
        ns = 0
        nb = 0
        nc = 0
        nt = 0
        bands: List[str] = []
        channels: List[str] = []
        trials: List[str] = []
        comments: List[str] = []
        marks: List[Tuple[int, str]] = []

        while True:
            line = data_file.readline().strip()
            if not line:
                continue

            if line.startswith(b"data_type"):
                data_type = line.split(b"data_type", 1)[1].strip().decode("utf-8")
            elif line.startswith(b"sampling_rate"):
                fs = float(line.split(b"sampling_rate", 1)[1].strip())
            elif line.startswith(b"samples"):
                ns = int(line.split(b"samples", 1)[1].strip())
            elif line.startswith(b"bands"):
                nb = int(line.split(b"bands", 1)[1].strip())
                bands = [""] * nb
            elif line.startswith(b"band_"):
                info = line.split(b"band_", 1)[1].split(b" ", 1)
                idx = int(info[0]) - 1
                bands[idx] = info[1].strip().decode("utf-8")
            elif line.startswith(b"channels"):
                nc = int(line.split(b"channels", 1)[1].strip())
                channels = [""] * nc
            elif line.startswith(b"channel_"):
                info = line.split(b"channel_", 1)[1].split(b" ", 1)
                idx = int(info[0]) - 1
                channels[idx] = info[1].strip().decode("utf-8")
            elif line.startswith(b"trials"):
                nt = int(line.split(b"trials", 1)[1].strip())
                trials = [""] * nt
            elif line.startswith(b"trial_"):
                info = line.split(b"trial_", 1)[1].split(b" ", 1)
                idx = int(info[0]) - 1
                trials[idx] = info[1].strip().decode("utf-8")
            elif line.startswith(b"comments"):
                ncomments = int(line.split(b"comments", 1)[1].strip())
                comments = [""] * ncomments
            elif line.startswith(b"comment_"):
                info = line.split(b"comment_", 1)[1].split(b" ", 1)
                idx = int(info[0]) - 1
                comments[idx] = info[1].strip().decode("utf-8")
            elif line.startswith(b"marks"):
                nmarks = int(line.split(b"marks", 1)[1].strip())
                marks = [(0, "")] * nmarks
            elif line.startswith(b"mark_"):
                info = line.split(b"mark_", 1)[1].split(b" ", 2)
                idx = int(info[0]) - 1
                mark_index = int(info[1])
                marks[idx] = (mark_index, info[2].strip().decode("utf-8"))
            elif line.startswith(b"end_header"):
                break

        data_size = nt * nc * nb * ns

        dtype_map = {
            "int8": (np.int8, 1),
            "char": (np.int8, 1),
            "uint8": (np.uint8, 1),
            "unsigned char": (np.uint8, 1),
            "int16": (np.int16, 2),
            "short": (np.int16, 2),
            "uint16": (np.uint16, 2),
            "unsigned short": (np.uint16, 2),
            "int32": (np.int32, 4),
            "int": (np.int32, 4),
            "uint32": (np.uint32, 4),
            "unsigned int": (np.uint32, 4),
            "int64": (np.int64, 8),
            "__int64": (np.int64, 8),
            "uint64": (np.uint64, 8),
            "unsigned __int64": (np.uint64, 8),
            "float": (np.float32, 4),
            "double": (np.float64, 8),
            "complex": (np.cdouble, 16),
            "class std::complex<double>": (np.cdouble, 16),
        }

        if data_type not in dtype_map:
            raise ValueError(f"Unsupported EBR data_type: {data_type}")

        target_dtype, byte_size = dtype_map[data_type]
        raw_bytes = data_file.read(byte_size * data_size)
        data = np.frombuffer(raw_bytes, dtype=target_dtype)

        if data_type not in ("complex", "class std::complex<double>"):
            data = data.astype(np.float64)

        data = data.reshape((nt, nc, nb, ns))

    return {
        "data_type": data_type,
        "sampling_rate": fs,
        "number_of_trials": nt,
        "trials": trials,
        "number_of_channels": nc,
        "channels": channels,
        "number_of_bands": nb,
        "bands": bands,
        "number_of_samples": ns,
        "number_of_comments": len(comments),
        "comments": comments,
        "number_of_marks": len(marks),
        "marks": marks,
        "data": data,
    }


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

    Returns a 2D array of shape (n_channels, n_samples) and the channel labels.
    """
    raw_data = recording["data"]
    channels = recording["channels"]
    eeg_indices = select_scalp_channels(channels, scalp_names)
    eeg_matrix = raw_data[0, eeg_indices, 0, :]
    return eeg_matrix, [channels[i] for i in eeg_indices]