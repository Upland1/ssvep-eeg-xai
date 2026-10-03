"""
Cross-frequency comparison of SSVEP functional connectivity (descriptive).

Input : the edge-level .npz files written by `analyze_connectivity_cohort.py`
        (reports/connectivity_edges/<subject>_<method>.npz).
Output: tidy CSVs + figures answering "do different stimulation frequencies
        recruit the same connectivity pattern, and where in the montage?"

Why it works on DELTA matrices (stimulus - matched baseline), never raw ones
---------------------------------------------------------------------------
Each stimulation frequency is scored in its OWN +/-1 Hz band, so raw
coherence/PLV values at 24 Hz and at 8.57 Hz are not on a common scale
(resting alpha alone lifts the 8-13 Hz band). The matched-baseline delta
removes whatever is already present at rest in that same band (resting
rhythms, volume conduction), so the deltas are the comparable quantity.

What is computed (per method, per subject, then per group)
-----------------------------------------------------------
1. Frequency-by-frequency SIMILARITY: Pearson correlation between the
   off-diagonal entries of two delta matrices (do 24 Hz and 20 Hz change
   the same channel pairs?), with a node-label permutation (QAP) p-value:
   the node labels of one matrix are shuffled together on rows and columns,
   which preserves the matrix's own structure and asks "is the agreement
   higher than for a random channel assignment?".
2. NODE STRENGTH per channel and frequency (mean delta to all other
   channels; for directed PDC the average of in- and out-flow) -- WHERE the
   change is concentrated.
3. Global mean delta and number of FDR-significant edges per frequency.

Group-level summaries are given for all subjects, for "responders" and for
"non-responders". Responders are defined from CLASSIFIER performance (the
2.0 s smoothed accuracy in reports/n_subjects_benchmark.csv, threshold set
by --responder-threshold) or explicitly via --responders -- deliberately NOT
from connectivity itself, which would be circular.

Caveats worth keeping in mind when reading the output
-----------------------------------------------------
- Deltas of different frequencies share a common spatial profile (nearby
  electrodes are always more coherent), so positive similarities are
  expected even for unrelated frequencies. Read the QAP p-values and the
  *differences* between pairs, not the raw level of r.
- With 7-8 channels there are only 21-28 undirected edges per matrix, and
  few subjects: treat everything here as descriptive until the cohort grows.
- Graph-efficiency-type measures are intentionally NOT computed: deltas can
  be negative, which makes path-length based measures ill-defined.
"""

import argparse
import itertools
import re
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FULL_MONTAGE = ["PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"]
CONDITIONS = [101, 102, 103, 104, 105]
FREQ_LABEL = {101: "24 Hz", 102: "20 Hz", 103: "15 Hz", 104: "10.91 Hz", 105: "8.57 Hz"}
FILE_PATTERN = re.compile(r"^(S\d+)_(.+)\.npz$")


# ---------------------------------------------------------------------------
# Loading / alignment
# ---------------------------------------------------------------------------

def embed_in_montage(matrix: np.ndarray, kept: list[str], montage: list[str] = FULL_MONTAGE) -> np.ndarray:
  """Place a (n_kept x n_kept) matrix into the full 8x8 montage grid.

  Channels a subject lost (Tier-1 drop, or legacy missing PO7) become NaN
  rows/columns, so subjects with different surviving channels can be
  compared and averaged on one common grid.
  """
  out = np.full((len(montage), len(montage)), np.nan)
  idx = [montage.index(n) for n in kept]
  out[np.ix_(idx, idx)] = matrix
  return out


def load_edge_file(path: Path) -> dict[int, dict[str, np.ndarray]]:
  """Return {condition: {"diff": 8x8, optional "fdr": 8x8}} for one file."""
  z = np.load(path)
  kept = [str(c) for c in z["channels_kept"]]
  if not kept:
    raise ValueError(f"{path.name}: no channel names stored; cannot align to the montage.")
  data = {}
  for cond in CONDITIONS:
    if f"diff_{cond}" not in z.files:
      continue
    entry = {"diff": embed_in_montage(z[f"diff_{cond}"], kept)}
    if f"fdr_{cond}" in z.files:
      entry["fdr"] = embed_in_montage(z[f"fdr_{cond}"].astype(float), kept)
    data[cond] = entry
  return data


def discover_edge_files(edges_dir: Path, methods: list[str] | None) -> dict[str, dict[str, Path]]:
  """{method: {subject: path}}"""
  found: dict[str, dict[str, Path]] = {}
  for p in sorted(edges_dir.glob("*.npz")):
    m = FILE_PATTERN.match(p.name)
    if not m:
      continue
    sub, method = m.group(1), m.group(2)
    if methods and method not in methods:
      continue
    found.setdefault(method, {})[sub] = p
  return found


def select_responders(
    subjects: list[str],
    explicit: list[str] | None,
    benchmark_csv: Path | None,
    threshold: float,
    column: str = "2.0s Smoothed (%)",
) -> tuple[list[str], str]:
  """Responder list + a human-readable note on how it was decided."""
  if explicit:
    return [s for s in explicit if s in subjects], "explicit --responders list"
  if benchmark_csv is not None and benchmark_csv.exists():
    df = pd.read_csv(benchmark_csv)
    if "Subject" in df.columns and column in df.columns:
      ok = df.loc[df[column] >= threshold, "Subject"].tolist()
      return [s for s in subjects if s in ok], f"{column} >= {threshold:g} in {benchmark_csv.name}"
  return [], "no responder definition available (pass --responders or --benchmark-csv)"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _offdiag(m: np.ndarray) -> np.ndarray:
  return m[~np.eye(m.shape[0], dtype=bool)]


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
  ok = np.isfinite(a) & np.isfinite(b)
  if ok.sum() < 6:
    return float("nan")
  a, b = a[ok], b[ok]
  if a.std() == 0.0 or b.std() == 0.0:
    return float("nan")
  return float(np.corrcoef(a, b)[0, 1])


def similarity_with_qap(
    a: np.ndarray, b: np.ndarray, n_perm: int, rng: np.random.Generator
) -> tuple[float, float]:
  """Pearson r between two delta matrices on their shared channels, plus a
  one-tailed (positive similarity) node-label permutation p-value."""
  present = ~np.all(np.isnan(a), axis=1) & ~np.all(np.isnan(b), axis=1)
  idx = np.flatnonzero(present)
  if len(idx) < 4:
    return float("nan"), float("nan")
  a_s, b_s = a[np.ix_(idx, idx)], b[np.ix_(idx, idx)]
  a_vec = _offdiag(a_s)
  r_obs = _pearson(a_vec, _offdiag(b_s))
  if not np.isfinite(r_obs) or n_perm <= 0:
    return r_obs, float("nan")
  exceed = 0
  for _ in range(n_perm):
    p = rng.permutation(len(idx))
    r = _pearson(a_vec, _offdiag(b_s[np.ix_(p, p)]))
    exceed += int(np.isfinite(r) and r >= r_obs)
  return r_obs, (1 + exceed) / (n_perm + 1)


def node_strength(m: np.ndarray) -> np.ndarray:
  """Mean delta from each channel to all others (NaN for absent channels).

  Symmetrised first so it also covers the directed PDC matrix (average of
  in- and out-flow).
  """
  s = (m + m.T) / 2.0
  s = s.copy()
  np.fill_diagonal(s, np.nan)
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    return np.nanmean(s, axis=1)


def global_mean_delta(m: np.ndarray) -> float:
  s = (m + m.T) / 2.0
  vals = s[np.triu_indices(s.shape[0], k=1)]
  return float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")


def n_fdr_edges(fdr: np.ndarray | None) -> float:
  if fdr is None:
    return float("nan")
  vals = fdr[np.triu_indices(fdr.shape[0], k=1)]
  return float(np.nansum(vals))


def fisher_mean(rs: list[np.ndarray]) -> np.ndarray:
  """Average correlation matrices across subjects via Fisher z."""
  stack = np.stack(rs)
  z = np.arctanh(np.clip(stack, -0.999, 0.999))
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    return np.tanh(np.nanmean(z, axis=0))


def nanmean_stack(arrs: list[np.ndarray]) -> np.ndarray:
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    return np.nanmean(np.stack(arrs), axis=0)


# ---------------------------------------------------------------------------
# Per-method analysis
# ---------------------------------------------------------------------------

def analyze_method(method: str, files: dict[str, Path], n_perm: int, seed: int) -> dict:
  rng = np.random.default_rng(seed)
  per_subject = {}
  for sub, path in files.items():
    try:
      per_subject[sub] = load_edge_file(path)
    except ValueError as e:
      print(f"[-] {e}")

  glob_rows, node_rows, sim_rows = [], [], []
  sim_r: dict[str, np.ndarray] = {}
  sim_p: dict[str, np.ndarray] = {}

  for sub, data in per_subject.items():
    conds = [c for c in CONDITIONS if c in data]
    for c in conds:
      m = data[c]["diff"]
      glob_rows.append({
          "subject": sub, "method": method, "condition": c, "frequency": FREQ_LABEL[c],
          "mean_delta": global_mean_delta(m), "n_fdr_edges": n_fdr_edges(data[c].get("fdr")),
      })
      for ch, v in zip(FULL_MONTAGE, node_strength(m)):
        node_rows.append({"subject": sub, "method": method, "condition": c,
                          "frequency": FREQ_LABEL[c], "channel": ch, "node_strength": v})

    r_mat = np.full((len(CONDITIONS), len(CONDITIONS)), np.nan)
    p_mat = np.full_like(r_mat, np.nan)
    np.fill_diagonal(r_mat, 1.0)
    for (i, ca), (j, cb) in itertools.combinations(list(enumerate(CONDITIONS)), 2):
      if ca not in data or cb not in data:
        continue
      r, p = similarity_with_qap(data[ca]["diff"], data[cb]["diff"], n_perm, rng)
      r_mat[i, j] = r_mat[j, i] = r
      p_mat[i, j] = p_mat[j, i] = p
      sim_rows.append({"subject": sub, "method": method, "freq_a": FREQ_LABEL[ca],
                       "freq_b": FREQ_LABEL[cb], "r": r, "p_qap": p})
    sim_r[sub], sim_p[sub] = r_mat, p_mat

  return {
      "per_subject": per_subject,
      "glob": pd.DataFrame(glob_rows),
      "node": pd.DataFrame(node_rows),
      "sim": pd.DataFrame(sim_rows),
      "sim_r": sim_r,
      "sim_p": sim_p,
  }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _heatmap(ax, m, labels_x, labels_y, vmin, vmax, cmap, title, annotate=True, stars=None, fmt="{:.2f}"):
  im = ax.imshow(m, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
  ax.set_xticks(range(len(labels_x)))
  ax.set_xticklabels(labels_x, rotation=45, ha="right", fontsize=8)
  ax.set_yticks(range(len(labels_y)))
  ax.set_yticklabels(labels_y, fontsize=8)
  ax.set_title(title, fontsize=10)
  if annotate:
    for i in range(m.shape[0]):
      for j in range(m.shape[1]):
        if np.isfinite(m[i, j]):
          star = "*" if (stars is not None and np.isfinite(stars[i, j]) and stars[i, j] < 0.05 and i != j) else ""
          dark = abs(m[i, j]) > 0.55 * max(abs(vmin), abs(vmax))
          ax.text(j, i, fmt.format(m[i, j]) + star, ha="center", va="center", fontsize=7,
                  color="white" if dark else "black")
  return im


def groups_for(subjects: list[str], responders: list[str]) -> dict[str, list[str]]:
  groups = {"all": list(subjects)}
  if responders:
    groups["responders"] = [s for s in subjects if s in responders]
    non = [s for s in subjects if s not in responders]
    if non:
      groups["non-responders"] = non
  return groups


def fig_similarity(method: str, res: dict, groups: dict[str, list[str]]) -> plt.Figure:
  subs = sorted(res["sim_r"])
  labels = [FREQ_LABEL[c] for c in CONDITIONS]
  panels = [(f"{s}", res["sim_r"][s], res["sim_p"][s]) for s in subs]
  for name, members in groups.items():
    members = [s for s in members if s in res["sim_r"]]
    if members:
      panels.append((f"{name}\n(n={len(members)}, mean r)", fisher_mean([res["sim_r"][s] for s in members]), None))
  n = len(panels)
  fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 4.4), squeeze=False, layout="constrained")
  for ax, (title, m, p) in zip(axes[0], panels):
    im = _heatmap(ax, m, labels, labels, -1, 1, "RdBu_r", title, stars=p)
  fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="Pearson r between delta matrices")
  fig.suptitle(f"[{method}] Similarity of stimulus-minus-baseline connectivity between frequencies "
               "(* = QAP p<0.05)", fontsize=11)
  return fig


def fig_node_strength(method: str, res: dict, groups: dict[str, list[str]]) -> plt.Figure:
  node = res["node"]
  names = list(groups)
  fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4.6), squeeze=False, layout="constrained")
  tables = {}
  for name in names:
    sub_df = node[node["subject"].isin(groups[name])]
    piv = sub_df.pivot_table(index="channel", columns="condition", values="node_strength", aggfunc="mean")
    tables[name] = piv.reindex(index=FULL_MONTAGE, columns=CONDITIONS)
  vmax = max(np.nanmax(np.abs(t.values)) for t in tables.values() if np.isfinite(t.values).any())
  vmax = max(float(vmax), 1e-3)
  for ax, name in zip(axes[0], names):
    im = _heatmap(ax, tables[name].values, [FREQ_LABEL[c] for c in CONDITIONS], FULL_MONTAGE,
                  -vmax, vmax, "RdBu_r", f"{name}\n(n={len(groups[name])})", fmt="{:+.2f}")
  fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="node strength (mean delta)")
  fig.suptitle(f"[{method}] Where does stimulation change connectivity? (per channel)", fontsize=11)
  return fig


def fig_delta_matrices(method: str, res: dict, members: list[str], group_name: str) -> plt.Figure:
  mats = {}
  for c in CONDITIONS:
    arrs = [res["per_subject"][s][c]["diff"] for s in members if c in res["per_subject"][s]]
    mats[c] = nanmean_stack(arrs) if arrs else np.full((8, 8), np.nan)
  vmax = max(max(float(np.nanmax(np.abs(m))) for m in mats.values() if np.isfinite(m).any()), 1e-3)
  fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(3.4 * len(CONDITIONS), 4.2), squeeze=False, layout="constrained")
  for ax, c in zip(axes[0], CONDITIONS):
    m = mats[c].copy()
    np.fill_diagonal(m, np.nan)
    im = _heatmap(ax, m, FULL_MONTAGE, FULL_MONTAGE, -vmax, vmax, "RdBu_r", FREQ_LABEL[c], annotate=False)
  fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="mean delta (stimulus - matched baseline)")
  fig.suptitle(f"[{method}] Mean delta connectivity per frequency -- {group_name} (n={len(members)})", fontsize=11)
  return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
  parser = argparse.ArgumentParser(description="Descriptive cross-frequency comparison of SSVEP connectivity.")
  parser.add_argument("--edges-dir", default=None, help="Default: <project>/reports/connectivity_edges")
  parser.add_argument("--methods", nargs="+", default=None, help="Subset of methods (default: all found).")
  parser.add_argument("--responders", nargs="+", default=None, help="Explicit responder subject list (e.g. S03 S04 S05).")
  parser.add_argument("--benchmark-csv", default=None,
                       help="Default: <project>/reports/n_subjects_benchmark.csv (classifier results).")
  parser.add_argument("--responder-threshold", type=float, default=80.0,
                       help="Min 2.0s smoothed accuracy (%%) to count as a responder (default 80).")
  parser.add_argument("--qap-perms", type=int, default=2000, help="Node-label permutations per similarity (default 2000).")
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--out-dir", default=None, help="CSV output dir. Default: <project>/reports/connectivity_comparison")
  parser.add_argument("--fig-dir", default=None, help="Default: <project>/outputs/figures/connectivity_comparison")
  args = parser.parse_args()

  project_root = Path(__file__).resolve().parents[2]
  edges_dir = Path(args.edges_dir) if args.edges_dir else project_root / "reports" / "connectivity_edges"
  out_dir = Path(args.out_dir) if args.out_dir else project_root / "reports" / "connectivity_comparison"
  fig_dir = Path(args.fig_dir) if args.fig_dir else project_root / "outputs" / "figures" / "connectivity_comparison"
  bench_csv = Path(args.benchmark_csv) if args.benchmark_csv else project_root / "reports" / "n_subjects_benchmark.csv"

  found = discover_edge_files(edges_dir, args.methods)
  if not found:
    print(f"No <subject>_<method>.npz files under {edges_dir}. Run analyze_connectivity_cohort.py first.")
    return
  out_dir.mkdir(parents=True, exist_ok=True)
  fig_dir.mkdir(parents=True, exist_ok=True)

  all_subjects = sorted({s for files in found.values() for s in files})
  responders, how = select_responders(all_subjects, args.responders, bench_csv, args.responder_threshold)
  print("=" * 80)
  print(f"CROSS-FREQUENCY CONNECTIVITY COMPARISON  methods={sorted(found)}  subjects={all_subjects}")
  print(f"Responders ({how}): {responders or 'none'}")
  print("=" * 80)

  for method, files in sorted(found.items()):
    print(f"\n##### {method} #####")
    res = analyze_method(method, files, args.qap_perms, args.seed)
    if res["glob"].empty:
      print("  no usable data.")
      continue
    groups = groups_for(sorted(res["sim_r"]), responders)

    glob_piv = res["glob"].pivot_table(index="subject", columns="frequency", values="mean_delta")
    glob_piv = glob_piv.reindex(columns=[FREQ_LABEL[c] for c in CONDITIONS])
    print("\nGlobal mean delta per subject x frequency (stimulus - matched baseline):")
    print(glob_piv.round(3).to_string())
    if res["glob"]["n_fdr_edges"].notna().any():
      fdr_piv = res["glob"].pivot_table(index="subject", columns="frequency", values="n_fdr_edges")
      print("\nFDR-significant edges per subject x frequency:")
      print(fdr_piv.reindex(columns=[FREQ_LABEL[c] for c in CONDITIONS]).to_string())

    print("\nFrequency-pair similarity of delta matrices (Fisher-z mean r | subjects with QAP p<0.05):")
    for gname, members in groups.items():
      sub_sim = res["sim"][res["sim"]["subject"].isin(members)]
      if sub_sim.empty:
        continue
      agg = sub_sim.groupby(["freq_a", "freq_b"]).agg(
          r_mean=("r", lambda x: float(np.tanh(np.nanmean(np.arctanh(np.clip(x, -0.999, 0.999)))))),
          n_sig=("p_qap", lambda x: int(np.sum(x < 0.05))),
          n=("r", "count"),
      ).reset_index()
      print(f"  -- {gname} (n={len(members)} subjects)")
      for _, row in agg.iterrows():
        print(f"     {row['freq_a']:>8s} vs {row['freq_b']:<8s} r={row['r_mean']:+.2f}  sig {row['n_sig']}/{row['n']}")

    res["glob"].to_csv(out_dir / f"{method}_global_delta.csv", index=False)
    res["node"].to_csv(out_dir / f"{method}_node_strength.csv", index=False)
    res["sim"].to_csv(out_dir / f"{method}_frequency_similarity.csv", index=False)

    fig = fig_similarity(method, res, groups)
    fig.savefig(fig_dir / f"{method}_frequency_similarity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    fig = fig_node_strength(method, res, groups)
    fig.savefig(fig_dir / f"{method}_node_strength.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    for gname, members in groups.items():
      if gname == "all" and len(groups) > 1:
        continue  # with a responder split, the per-group panels are the informative ones
      fig = fig_delta_matrices(method, res, members, gname)
      fig.savefig(fig_dir / f"{method}_delta_matrices_{gname.replace(' ', '_')}.png", dpi=150, bbox_inches="tight")
      plt.close(fig)

  print(f"\nCSVs saved under: {out_dir}")
  print(f"Figures saved under: {fig_dir}")


if __name__ == "__main__":
  main()