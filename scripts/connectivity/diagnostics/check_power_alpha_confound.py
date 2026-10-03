"""Check whether connectivity changes are explained by SSVEP power or alpha blocking.

A common driven component can increase coherence and wPLI without true
network coupling. The same issue is relevant near alpha frequencies, where
stimulus-related alpha suppression can oppose the SSVEP-driven phase effect.

Per subject and stimulation frequency, this script compares the mean
connectivity delta to:
- target_snr_db: evoked power increase at the fundamental,
- alpha_change_db: alpha-band change excluding the target band.

Power is computed as the mean Hann periodogram across retained occipital
channels; ratios use group means.

For each method, it reports Spearman correlations, within-subject centering,
and OLS R^2 for delta ~ target_snr_db + alpha_change_db.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import periodogram
from scipy.stats import spearmanr

from scripts.connectivity.analysis.analyze_connectivity_cohort import (
    discover_subjects,
    resolve_channel_names,
)
from scripts.connectivity.comparison.compare_connectivity_frequencies import (
    CONDITIONS,
    FREQ_LABEL,
    select_responders,
)
from src.features.connectivity.connectivity_extraction import BASELINE_CONDITION, CONDITION_FREQS
from src.preprocessing.quality_check import apply_artifact_quality_pipeline

FS = 250.0
HALF_WIDTH = 1.0
ALPHA_BAND = (8.0, 13.0)
OCCIPITAL = ["O1", "Oz", "O2"]


def occipital_indices(names: list[str] | None, n_channels: int) -> list[int]:
  if names is not None:
    idx = [i for i, n in enumerate(names) if n in OCCIPITAL]
    if idx:
      return idx
  return list(range(n_channels))


ALPHA_EXCLUDE_HALF_WIDTH = 1.5  # Hz around the target, removed from the alpha band


def mean_band_power(
    windows: np.ndarray,
    ch_idx: list[int],
    band: tuple[float, float],
    exclude: tuple[float, float] | None = None,
) -> float:
  """Mean periodogram power in `band` (optionally minus an excluded sub-band),
  averaged over windows and selected channels."""
  freqs, psd = periodogram(windows[:, ch_idx, :], fs=FS, window="hann", axis=-1)
  mask = (freqs >= band[0]) & (freqs <= band[1])
  if exclude is not None:
    mask &= ~((freqs >= exclude[0]) & (freqs <= exclude[1]))
  return float(psd[..., mask].mean())


def subject_power_rows(sub: str, data_root: Path) -> list[dict]:
  sub_dir = data_root / sub
  if not (sub_dir / "X_time_windows.npy").exists():
    print(f"[-] Skipping {sub}: X_time_windows.npy not found.")
    return []
  windows_raw = np.load(sub_dir / "X_time_windows.npy")
  y_raw = np.load(sub_dir / "y_labels.npy")
  mask = np.isin(y_raw, [BASELINE_CONDITION] + CONDITIONS)
  y_sel = y_raw[mask]
  windows = windows_raw[mask] if windows_raw.shape[0] == len(y_raw) else windows_raw[: len(y_sel)]
  names = resolve_channel_names(windows.shape[1])

  qc = apply_artifact_quality_pipeline(windows, y_sel, channel_names=names, verbose=False)
  W, y = qc["windows_clean"], qc["y_clean"]
  ch_idx = occipital_indices(qc["channels_kept"], W.shape[1])
  used = [qc["channels_kept"][i] for i in ch_idx] if qc["channels_kept"] else ch_idx

  base = W[y == BASELINE_CONDITION]
  if base.shape[0] == 0:
    print(f"[-] Skipping {sub}: no baseline windows after QC.")
    return []

  rows = []
  for cond in CONDITIONS:
    stim = W[y == cond]
    if stim.shape[0] == 0:
      continue
    f0 = CONDITION_FREQS[cond]
    band = (f0 - HALF_WIDTH, f0 + HALF_WIDTH)
    snr = 10.0 * np.log10(mean_band_power(stim, ch_idx, band) / mean_band_power(base, ch_idx, band))
    # Alpha power OUTSIDE the stimulated band: for 10.91 / 8.57 Hz the target
    # itself lies in 8-13 Hz, so including it would mix the SSVEP's own power
    # gain into the "alpha blocking" measure. Same exclusion for baseline.
    excl = (f0 - ALPHA_EXCLUDE_HALF_WIDTH, f0 + ALPHA_EXCLUDE_HALF_WIDTH)
    alpha = 10.0 * np.log10(
        mean_band_power(stim, ch_idx, ALPHA_BAND, exclude=excl)
        / mean_band_power(base, ch_idx, ALPHA_BAND, exclude=excl)
    )
    rows.append({
        "subject": sub, "condition": cond, "frequency": FREQ_LABEL[cond],
        "target_snr_db": snr, "alpha_change_db": alpha,
        "n_stim_windows": int(stim.shape[0]), "n_base_windows": int(base.shape[0]),
        "channels_used": ",".join(map(str, used)),
    })
  return rows


def center_by_subject(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
  out = df.copy()
  for c in cols:
    out[c] = out[c] - out.groupby("subject")[c].transform("mean")
  return out


def ols_r2(y: np.ndarray, X: np.ndarray) -> tuple[float, np.ndarray]:
  X1 = np.column_stack([np.ones(len(y)), X])
  beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
  resid = y - X1 @ beta
  ss_tot = float(np.sum((y - y.mean()) ** 2))
  return (1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else float("nan")), beta[1:]


def _spearman(a: pd.Series, b: pd.Series) -> tuple[float, float]:
  if len(a) < 4 or a.nunique() < 2 or b.nunique() < 2:
    return float("nan"), float("nan")
  r, p = spearmanr(a, b)
  return float(r), float(p)


def analyze_method(method: str, power: pd.DataFrame, delta_csv: Path, responders: list[str]) -> pd.DataFrame | None:
  if not delta_csv.exists():
    print(f"[{method}] {delta_csv.name} not found -- run compare_connectivity_frequencies.py first.")
    return None
  d = pd.read_csv(delta_csv)[["subject", "condition", "mean_delta"]]
  df = power.merge(d, on=["subject", "condition"], how="inner")
  if df.empty:
    print(f"[{method}] no overlapping subject/condition cells.")
    return None

  print(f"\n##### {method} #####")
  groups = {"all subjects": df}
  if responders:
    resp = df[df["subject"].isin(responders)]
    if resp["subject"].nunique() >= 2:
      groups["responders"] = resp

  for gname, g in groups.items():
    n_sub = g["subject"].nunique()
    gc = center_by_subject(g, ["mean_delta", "target_snr_db", "alpha_change_db"])
    print(f"\n  -- {gname} (n_subjects={n_sub}, cells={len(g)})")
    for col, label in (("target_snr_db", "target SNR (dB)"), ("alpha_change_db", "alpha change (dB)")):
      r_all, p_all = _spearman(g["mean_delta"], g[col])
      r_in, p_in = _spearman(gc["mean_delta"], gc[col])
      print(f"     delta vs {label:<17s} Spearman all-cells r={r_all:+.2f} (p={p_all:.3f}) | "
            f"within-subject r={r_in:+.2f} (p={p_in:.3f})")
    if len(g) >= 6:
      r2, coefs = ols_r2(g["mean_delta"].to_numpy(), g[["target_snr_db", "alpha_change_db"]].to_numpy())
      r2c, coefs_c = ols_r2(gc["mean_delta"].to_numpy(), gc[["target_snr_db", "alpha_change_db"]].to_numpy())
      print(f"     OLS delta ~ SNR + alpha: R2={r2:.2f} (coef SNR {coefs[0]:+.3f}, alpha {coefs[1]:+.3f}) | "
            f"within-subject R2={r2c:.2f}")

  inalpha = df[df["condition"].isin([104, 105])]
  outalpha = df[~df["condition"].isin([104, 105])]
  print("\n  Alpha-band targets (10.91, 8.57 Hz) vs the rest, mean over subjects:")
  print(f"     in-alpha  : delta={inalpha['mean_delta'].mean():+.3f}  alpha change={inalpha['alpha_change_db'].mean():+.2f} dB")
  print(f"     out-alpha : delta={outalpha['mean_delta'].mean():+.3f}  alpha change={outalpha['alpha_change_db'].mean():+.2f} dB")
  df["method"] = method
  return df


def make_figure(frames: list[pd.DataFrame], out_path: Path) -> None:
  n = len(frames)
  fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.6), squeeze=False, layout="constrained")
  subs = sorted({s for f in frames for s in f["subject"]})
  cmap = plt.get_cmap("tab10")
  markers = {101: "o", 102: "s", 103: "^", 104: "D", 105: "v"}
  for ax, f in zip(axes[0], frames):
    for i, s in enumerate(subs):
      for cond in CONDITIONS:
        cell = f[(f["subject"] == s) & (f["condition"] == cond)]
        if cell.empty:
          continue
        ax.scatter(cell["target_snr_db"], cell["mean_delta"], color=cmap(i), marker=markers[cond],
                   s=55, edgecolor="k", linewidth=0.4)
    ax.axhline(0, color="gray", lw=0.6)
    ax.axvline(0, color="gray", lw=0.6)
    ax.set_xlabel("target SNR (dB): stimulus vs baseline power at the fundamental")
    ax.set_ylabel("mean connectivity delta")
    ax.set_title(f"[{f['method'].iloc[0]}]")
    ax.grid(alpha=0.3)
  sub_handles = [plt.Line2D([], [], marker="o", ls="", color=cmap(i), label=s) for i, s in enumerate(subs)]
  freq_handles = [plt.Line2D([], [], marker=markers[c], ls="", color="gray", label=FREQ_LABEL[c]) for c in CONDITIONS]
  axes[0][-1].legend(handles=sub_handles + freq_handles, fontsize=7, loc="best", ncol=2)
  fig.suptitle("Connectivity delta vs SSVEP power at the fundamental (one point per subject x frequency)", fontsize=11)
  fig.savefig(out_path, dpi=150, bbox_inches="tight")
  plt.close(fig)


def main():
  parser = argparse.ArgumentParser(description="Step 3: is the connectivity delta explained by SSVEP power / alpha blocking?")
  parser.add_argument("--methods", nargs="+", default=["coherence", "wpli"])
  parser.add_argument("--subjects", nargs="+", default=None)
  parser.add_argument("--responders", nargs="+", default=None)
  parser.add_argument("--responder-threshold", type=float, default=80.0)
  parser.add_argument("--benchmark-csv", default=None)
  args = parser.parse_args()

  project_root = Path(__file__).resolve().parents[2]
  data_root = project_root / "data" / "processed"
  rep_dir = project_root / "reports" / "connectivity_comparison"
  fig_dir = project_root / "outputs" / "figures" / "connectivity_comparison"
  rep_dir.mkdir(parents=True, exist_ok=True)
  fig_dir.mkdir(parents=True, exist_ok=True)
  bench_csv = Path(args.benchmark_csv) if args.benchmark_csv else project_root / "reports" / "n_subjects_benchmark.csv"

  subjects = args.subjects or discover_subjects(data_root)
  responders, how = select_responders(subjects, args.responders, bench_csv, args.responder_threshold)
  print("=" * 90)
  print(f"POWER / ALPHA CONFOUND CHECK  subjects={subjects}")
  print(f"Responders ({how}): {responders or 'none'}")
  print("=" * 90)

  rows = []
  for sub in subjects:
    rows += subject_power_rows(sub, data_root)
  if not rows:
    print("No power results computed.")
    return
  power = pd.DataFrame(rows)
  power.to_csv(rep_dir / "power_confound_table.csv", index=False)

  piv_snr = power.pivot_table(index="subject", columns="frequency", values="target_snr_db")
  piv_alpha = power.pivot_table(index="subject", columns="frequency", values="alpha_change_db")
  order = [FREQ_LABEL[c] for c in CONDITIONS]
  print("\nTarget SNR (dB), stimulus vs baseline power in the target +/-1 Hz band:")
  print(piv_snr.reindex(columns=order).round(2).to_string())
  print("\nAlpha change (dB), 8-13 Hz excluding the target band, stimulus vs baseline (negative = alpha blocking):")
  print(piv_alpha.reindex(columns=order).round(2).to_string())

  frames = []
  for method in args.methods:
    res = analyze_method(method, power, rep_dir / f"{method}_global_delta.csv", responders)
    if res is not None:
      frames.append(res)
  if frames:
    pd.concat(frames).to_csv(rep_dir / "power_vs_connectivity_cells.csv", index=False)
    make_figure(frames, fig_dir / "connectivity_delta_vs_power.png")
    print(f"\nFigure saved to: {fig_dir / 'connectivity_delta_vs_power.png'}")
  print(f"CSVs saved under: {rep_dir}")


if __name__ == "__main__":
  main()