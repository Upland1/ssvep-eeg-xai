"""Prepare graph inputs for the GNN without using Torch.

Variants combine node features, coherence, phase, and optional FBCCA scores.
Standardisation is fitted on the training fold and then applied to the test
fold.
"""

import numpy as np

VALID_TOKENS = {"nodes", "coh", "phase", "F"}


def parse_variant(variant: str) -> dict[str, bool]:
  tokens = variant.split("+")
  if tokens[0] != "nodes" or not set(tokens) <= VALID_TOKENS:
    raise ValueError(
        f"Invalid variant {variant!r}: must start with 'nodes' and use only "
        f"{sorted(VALID_TOKENS - {'nodes'})} after it."
    )
  return {"coh": "coh" in tokens, "phase": "phase" in tokens, "F": "F" in tokens}


def build_inputs(ds: dict, variant: str) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
  """Return (nodes, edges, graph) arrays for a variant.

  nodes : (n, C, 10)        edges : (n, C, C, De) or None        graph : (n, 5) or None
  """
  flags = parse_variant(variant)
  nodes = ds["node"].astype(np.float32)
  parts = []
  if flags["coh"]:
    parts.append(ds["edge_coh"])                                  # (n, C, C, 5)
  if flags["phase"]:
    ph = ds["edge_phase"]
    parts.append(ph.reshape(ph.shape[0], ph.shape[1], ph.shape[2], -1))   # (n, C, C, 10)
  edges = np.concatenate(parts, axis=-1).astype(np.float32) if parts else None
  graph = ds["X_fbcca"].astype(np.float32) if flags["F"] else None
  return nodes, edges, graph


def fit_standardizer(train: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
  """Per-position mean/std over the sample axis (axis 0), from TRAINING rows only."""
  return train.mean(axis=0, keepdims=True), train.std(axis=0, keepdims=True) + eps


def apply_standardizer(a: np.ndarray, stats: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
  return ((a - stats[0]) / stats[1]).astype(np.float32)


def standardize_fold(
    arrays: tuple[np.ndarray, np.ndarray | None, np.ndarray | None],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[tuple, tuple]:
  """Standardise (nodes, edges, graph) with statistics fit on `train_idx` only."""
  tr, te = [], []
  for a in arrays:
    if a is None:
      tr.append(None)
      te.append(None)
      continue
    stats = fit_standardizer(a[train_idx])
    tr.append(apply_standardizer(a[train_idx], stats))
    te.append(apply_standardizer(a[test_idx], stats))
  return tuple(tr), tuple(te)