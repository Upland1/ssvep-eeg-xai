"""Build and cache the graph features used by the connectivity benchmarks.

Each subject is cleaned with the same quality pipeline used elsewhere. The
result is saved as ``data/processed/<subject>/graph_dataset.npz`` with node
power, edge coherence, edge phase, FBCCA scores, labels, trial IDs, and the
retained channel names.

The script also prints a short check showing whether each feature is stronger
for its matching stimulation class.
"""

import argparse
from pathlib import Path

import numpy as np

from scripts.connectivity.analysis.analyze_connectivity_cohort import discover_subjects, resolve_channel_names
from src.features.connectivity.connectivity_graph_features import TARGET_FREQS, build_graph_features
from src.features.ssvep.fbcca_extraction import extract_fbcca_features
from src.preprocessing.dataset.cv_utils import build_trial_ids

CLASSES = [101, 102, 103, 104, 105]
FREQ_LABEL = ["24 Hz", "20 Hz", "15 Hz", "10.91 Hz", "8.57 Hz"]


def build_subject_dataset(sub: str, data_root: Path, fs: float = 250.0) -> dict | None:
  sub_dir = data_root / sub
  if not (sub_dir / "X_time_windows.npy").exists():
    print(f"[-] Skipping {sub}: X_time_windows.npy not found.")
    return None
  windows_raw = np.load(sub_dir / "X_time_windows.npy")
  y_raw = np.load(sub_dir / "y_labels.npy")
  stim_mask = np.isin(y_raw, CLASSES)
  y_stim = y_raw[stim_mask]
  windows = windows_raw[stim_mask] if windows_raw.shape[0] == len(y_raw) else windows_raw[: len(y_stim)]
  names = resolve_channel_names(windows.shape[1])

  X_fbcca, y_clean, valid_mask, report = extract_fbcca_features(windows, y_stim, fs=fs, channel_names=names)
  windows_clean = windows[valid_mask == 1][:, report["channels_kept_idx"], :]
  trial_ids = build_trial_ids(y_stim, sub_windows_per_trial=5)[valid_mask == 1]

  feats = build_graph_features(windows_clean, fs, TARGET_FREQS)
  kept = report["channels_kept"] or [f"ch{i}" for i in range(windows_clean.shape[1])]
  return {
      **feats,
      "X_fbcca": X_fbcca,
      "y": y_clean,
      "trial_ids": trial_ids,
      "channels_kept": np.array(kept, dtype=str),
  }


def load_or_build(sub: str, data_root: Path, rebuild: bool = False, fs: float = 250.0) -> dict | None:
  path = data_root / sub / "graph_dataset.npz"
  if path.exists() and not rebuild:
    z = np.load(path)
    return {k: z[k] for k in z.files}
  ds = build_subject_dataset(sub, data_root, fs)
  if ds is not None:
    np.savez_compressed(path, **ds)
  return ds


def sanity_report(sub: str, ds: dict) -> None:
  y = ds["y"]
  n, c = ds["node"].shape[:2]
  print(f"  {sub}: {n} clean windows | channels={[str(c) for c in ds['channels_kept']]} | "
        f"node {ds['node'].shape} | edge_coh {ds['edge_coh'].shape} | edge_phase {ds['edge_phase'].shape}")
  bad = [k for k in ("node", "edge_coh", "edge_phase", "X_fbcca") if not np.isfinite(ds[k]).all()]
  if bad:
    print(f"     [!] non-finite values in: {bad}")
  iu, ju = np.triu_indices(c, k=1)
  print("     own-band minus other-class (positive = feature is higher in matching-class windows)")
  print("     band        node log10 power | edge coherence | phase consistency")
  for k, cls in enumerate(CLASSES):
    own, oth = y == cls, y != cls
    if not own.any() or not oth.any():
      continue
    node_k = ds["node"][:, :, k].mean(axis=1)                       # fundamental power
    coh_k = ds["edge_coh"][:, iu, ju, k].mean(axis=1)
    ph = ds["edge_phase"][:, iu, ju, k, :]                            # (n, pairs, 2)
    # Consistency is the length of the mean phase vector.
    cons_own = np.linalg.norm(ph[own].mean(axis=0), axis=-1).mean()
    cons_oth = np.linalg.norm(ph[oth].mean(axis=0), axis=-1).mean()
    print(f"     {FREQ_LABEL[k]:>8s}   {node_k[own].mean() - node_k[oth].mean():+10.3f}      "
          f"{coh_k[own].mean() - coh_k[oth].mean():+10.3f}     {cons_own - cons_oth:+10.3f}")


def main():
  parser = argparse.ArgumentParser(description="Step 4: build per-window graph datasets per subject.")
  parser.add_argument("--subjects", nargs="+", default=None)
  parser.add_argument("--rebuild", action="store_true", help="Recompute even if graph_dataset.npz exists.")
  args = parser.parse_args()

  project_root = Path(__file__).resolve().parents[3]
  data_root = project_root / "data" / "processed"
  subjects = args.subjects or discover_subjects(data_root)
  print("=" * 90)
  print(f"GRAPH DATASET BUILD  subjects={subjects}")
  print("=" * 90)
  for sub in subjects:
    ds = load_or_build(sub, data_root, rebuild=args.rebuild)
    if ds is not None:
      sanity_report(sub, ds)
  print(f"\nSaved/loaded graph_dataset.npz under {data_root}/<subject>/")


if __name__ == "__main__":
  main()