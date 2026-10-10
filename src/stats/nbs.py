"""Network-Based Statistic (NBS) utilities for connectivity."""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def edge_pseudo_z(obs: np.ndarray, null: np.ndarray, two_tailed: bool = False) -> tuple[np.ndarray, np.ndarray]:
  """Standardize observed and null edge values by their null statistics."""
  mu = null.mean(axis=0)
  sd = null.std(axis=0)
  sd = np.where(sd > 0, sd, np.inf)
  z_obs = (obs - mu) / sd
  z_null = (null - mu) / sd
  if two_tailed:
    return np.abs(z_obs), np.abs(z_null)
  return z_obs, z_null


def _components(stat: np.ndarray, threshold: float) -> list[list[tuple[int, int]]]:
  """Return connected components above the threshold."""
  c = stat.shape[0]
  iu, ju = np.triu_indices(c, k=1)
  sup = stat[iu, ju] > threshold
  if not sup.any():
    return []
  adj = np.zeros((c, c), dtype=bool)
  adj[iu[sup], ju[sup]] = True
  adj |= adj.T
  _, labels = connected_components(csr_matrix(adj), directed=False)
  comps = {}
  for i, j in zip(iu[sup], ju[sup]):
    comps.setdefault(labels[i], []).append((int(i), int(j)))
  return sorted(comps.values(), key=len, reverse=True)


def nbs_test(z_obs: np.ndarray, z_null: np.ndarray, threshold: float, alpha: float = 0.05) -> dict:
  """Run NBS at one primary threshold."""
  c = z_obs.shape[0]
  null_max = np.array([max((len(comp) for comp in _components(z, threshold)), default=0) for z in z_null])
  n_perm = len(null_max)
  comps = []
  sig_mask = np.zeros((c, c), dtype=bool)
  for edges in _components(z_obs, threshold):
    size = len(edges)
    p = float((1 + np.sum(null_max >= size)) / (n_perm + 1))
    comps.append({"edges": edges, "size": size, "p": p})
    if p < alpha:
      for i, j in edges:
        sig_mask[i, j] = sig_mask[j, i] = True
  sig = [comp for comp in comps if comp["p"] < alpha]
  return {
      "threshold": threshold,
      "components": comps,
      "null_max": null_max,
      "sig_mask": sig_mask,
      "max_size": comps[0]["size"] if comps else 0,
      "p_max": comps[0]["p"] if comps else 1.0,
      "n_sig_components": len(sig),
      "n_sig_edges": int(sum(comp["size"] for comp in sig)),
      "possible_edges": c * (c - 1) // 2,
  }


def nbs_sweep(obs: np.ndarray, null: np.ndarray, thresholds=(1.5, 2.0, 2.5, 3.0),
              alpha: float = 0.05, two_tailed: bool = False) -> dict[float, dict]:
  """Run `nbs_test` at several primary thresholds on the same permutations."""
  z_obs, z_null = edge_pseudo_z(obs, null, two_tailed=two_tailed)
  return {t: nbs_test(z_obs, z_null, t, alpha) for t in thresholds}