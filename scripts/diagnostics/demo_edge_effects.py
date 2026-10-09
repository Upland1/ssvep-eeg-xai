"""Show how filtering short windows creates edge effects.

Each filter-bank band is computed two ways: on the full recording before
cutting the window, and on the window alone. The difference shows how much
filtering is affected by the missing signal around the window edges.

Outputs (outputs/figures/edge_effects/ and reports/):
  edge_effects_<subject>.png   3-panel figure for the professor
  edge_effects_summary.csv     per subject x band: error at the edges vs centre

Usage:
  python scripts/diagnostics/demo_edge_effects.py                 # S03 figure + all-subject table
  python scripts/diagnostics/demo_edge_effects.py --subject S04
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.preprocessing.signal.filters import apply_iir_bandpass  # noqa: E402
from src.preprocessing.windowing import (  # noqa: E402
    STIM_CONDITIONS,
    band_windows,
    build_window_index,
    fbcca_filter_bank,
    fbcsp_filter_bank,
    load_session,
)

EDGE_SEC = 0.1  # first and last 100 ms of the window count as "edge"
STIM_FREQ = {101: 24.0, 102: 20.0, 103: 15.0, 104: 10.9091, 105: 8.5714}

# Palette (reference data-viz palette, light mode)
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C_CONT, C_PW = "#2a78d6", "#eb6834"   # categorical slots 1 and 2
EDGE_FILL = "#f1efe9"


def subject_errors(raw_root: Path, sub: str, session: str, cfg: dict) -> dict:
  pre = cfg["preprocessing"]
  sess = load_session(raw_root / sub / f"{session}.ebr", cfg["channels"]["scalp"], cfg["channels"]["mark_channel"])
  fs = sess["fs"]
  n = int(round(pre["sub_window_sec"] * fs))
  bb = apply_iir_bandpass(sess["eeg"], fs, pre["lowcut_hz"], pre["highcut_hz"], pre["filter_order"])
  idx = build_window_index(sess["mark"], fs)
  stim = np.isin(idx.labels, STIM_CONDITIONS)
  starts, labels = idx.starts[stim], idx.labels[stim]

  bank = fbcsp_filter_bank(fs) + fbcca_filter_bank(fs)
  cont = band_windows(bb, starts, n, bank, "continuous")   # (w, band, ch, t)
  pw = band_windows(bb, starts, n, bank, "per_window")
  rms = np.sqrt(np.mean(cont**2, axis=-1, keepdims=True))
  rel = np.abs(pw - cont) / np.maximum(rms, 1e-12)        # (w, band, ch, t)
  return {"fs": fs, "n": n, "bank": bank, "cont": cont, "pw": pw, "rel": rel,
          "labels": labels, "channels": sess["channels"]}


def summarise(sub: str, r: dict) -> list[dict]:
  e = int(round(EDGE_SEC * r["fs"]))
  profile = r["rel"].mean(axis=(0, 2))                     # (band, t)
  rows = []
  for b, f in enumerate(r["bank"]):
    edge = np.r_[profile[b, :e], profile[b, -e:]].mean()
    centre = profile[b, e:-e].mean()
    rows.append({"subject": sub, "band": f.name, "filter": f.kind,
                 "rel_error_edges_pct": round(100 * edge, 1),
                 "rel_error_centre_pct": round(100 * centre, 1),
                 "edge_to_centre_ratio": round(edge / centre, 1) if centre > 0 else np.nan})
  return rows


def plot_subject(sub: str, r: dict, out_png: Path) -> None:
  fs, n = r["fs"], r["n"]
  t = np.arange(n) / fs
  e = int(round(EDGE_SEC * fs))
  names = [f.name for f in r["bank"]]
  narrow = names.index("fbcsp_13-17")      # band holding the 15 Hz stimulus
  wide = names.index("fbcca_14-90")
  ch = r["channels"].index("Oz") if "Oz" in r["channels"] else 0

  # Example: the 15 Hz (103) window where the edge effect is largest on Oz.
  cand = np.flatnonzero(r["labels"] == 103)
  w = cand[np.argmax(r["rel"][cand, narrow, ch, :].mean(axis=-1))]

  plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
                       "xtick.color": INK_2, "ytick.color": INK_2, "axes.titlecolor": INK,
                       "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left"})
  fig = plt.figure(figsize=(11, 7.2), facecolor="white")
  gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.5, wspace=0.5)

  def shade(ax):
    ax.axvspan(0, EDGE_SEC, color=EDGE_FILL, zorder=0, lw=0)
    ax.axvspan(t[-1] - EDGE_SEC, t[-1], color=EDGE_FILL, zorder=0, lw=0)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
      ax.spines[s].set_visible(False)

  # A: example trace
  ax = fig.add_subplot(gs[0, :])
  shade(ax)
  ax.plot(t, r["cont"][w, narrow, ch], color=C_CONT, lw=2, label="Filtered continuously, then cut (reference)")
  ax.plot(t, r["pw"][w, narrow, ch], color=C_PW, lw=2, label="Cut first, filtered per 1 s window (old)")
  ax.set_title(f"A. {sub}, 15 Hz stimulus, channel {r['channels'][ch]}, band 13-17 Hz: same window, two ways")
  ax.set_xlabel("Time inside the 1 s window (s)")
  ax.set_ylabel("Amplitude (µV)")
  ax.set_xlim(0, t[-1])
  ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, labelcolor=INK)
  for x in (EDGE_SEC / 2, t[-1] - EDGE_SEC / 2):
    ax.text(x, 0.03, "edge", transform=ax.get_xaxis_transform(), ha="center", va="bottom", color=INK_2, fontsize=8)

  # B: error profile for a narrow and a wide band
  ax = fig.add_subplot(gs[1, 0])
  shade(ax)
  prof = 100 * r["rel"].mean(axis=(0, 2))
  for b, col, lab in ((narrow, INK, "FBCSP 13-17 Hz (narrow, Butterworth)"),
                      (wide, INK_2, "FBCCA 14-90 Hz (wide, Chebyshev)")):
    ax.plot(t, prof[b], color=col, lw=2, ls="-" if b == narrow else "--", label=lab)
  ax.set_title("B. Where in the window the error is (all stimulus windows)")
  ax.set_xlabel("Time inside the 1 s window (s)")
  ax.set_ylabel("Error, % of signal RMS")
  ax.set_xlim(0, t[-1])
  ax.set_ylim(0, None)
  ax.legend(frameon=False, loc="upper center", labelcolor=INK, fontsize=8)

  # C: every band, edges vs centre
  ax = fig.add_subplot(gs[1, 1])
  edge = np.array([np.r_[prof[b, :e], prof[b, -e:]].mean() for b in range(len(names))])
  centre = np.array([prof[b, e:-e].mean() for b in range(len(names))])
  y = np.arange(len(names))[::-1]
  h = 0.36
  ax.barh(y + h / 2 + 0.02, edge, height=h, color=INK, label=f"First/last {int(EDGE_SEC*1000)} ms")
  ax.barh(y - h / 2 - 0.02, centre, height=h, color="#a9a8a2", label="Rest of the window")
  for yi, v in zip(y, edge):
    ax.text(v + 1, yi + h / 2 + 0.02, f"{v:.0f}%", va="center", fontsize=8, color=INK)
  ax.set_yticks(y)
  ax.set_yticklabels([nm.replace("fbcsp_", "FBCSP ").replace("fbcca_", "FBCCA ") + " Hz" for nm in names])
  ax.set_xlabel("Mean error, % of signal RMS")
  ax.set_title(f"C. Every filter-bank band ({sub})")
  ax.grid(axis="x", color=GRID, lw=0.6)
  ax.set_axisbelow(True)
  for s in ("top", "right"):
    ax.spines[s].set_visible(False)
  ax.set_xlim(0, edge.max() * 1.18)
  ax.legend(frameon=False, loc="lower right", labelcolor=INK, fontsize=8)

  fig.suptitle("Edge effects: filtering each 1 s window on its own distorts the start and end of the window",
               x=0.07, ha="left", fontsize=12, fontweight="bold", color=INK)
  out_png.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(out_png, dpi=200, bbox_inches="tight")
  plt.close(fig)


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--subject", default="S03", help="Subject for the figure (default S03)")
  ap.add_argument("--session", default="OO")
  ap.add_argument("--config", type=Path, default=ROOT / "configs" / "pipeline_config.yaml")
  args = ap.parse_args()

  cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
  raw_root = ROOT / cfg["paths"]["data_raw_dir"]
  subjects = sorted(p.name for p in raw_root.iterdir() if p.is_dir() and re.match(r"^S\d+$", p.name))

  rows = []
  for sub in subjects:
    try:
      r = subject_errors(raw_root, sub, args.session, cfg)
    except (FileNotFoundError, ValueError) as exc:
      print(f"[-] {sub}: skipped ({exc})")
      continue
    rows += summarise(sub, r)
    if sub == args.subject:
      png = ROOT / "outputs" / "figures" / "edge_effects" / f"edge_effects_{sub}.png"
      plot_subject(sub, r, png)
      print(f"Figure: {png}")

  df = pd.DataFrame(rows)
  out_csv = ROOT / "reports" / "edge_effects_summary.csv"
  out_csv.parent.mkdir(parents=True, exist_ok=True)
  df.to_csv(out_csv, index=False)
  cohort = df.groupby("band", sort=False)[["rel_error_edges_pct", "rel_error_centre_pct"]].mean().round(1)
  print("\nCohort mean error (% of signal RMS), per-window vs continuous filtering:")
  print(cohort.to_string())
  print(f"\nTable: {out_csv}")


if __name__ == "__main__":
  main()