"""Check whether connectivity features improve frequency classification.

This step compares power-based features with edge coherence and phase using
the same grouped cross-validation protocol as the other benchmarks. It tests
whether connectivity adds useful information before training a graph model.

The final rule is only a descriptive guide for a small cohort: connectivity
passes when adding edge features improves the power baseline by at least two
accuracy points on average and for at least two-thirds of the responders.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.connectivity.analysis.analyze_connectivity_cohort import discover_subjects
from scripts.connectivity.comparison.compare_connectivity_frequencies import select_responders
from src.features.connectivity.connectivity_graph_features import upper_tri_flat
from scripts.connectivity.graph_modeling.build_graph_dataset import load_or_build

BASE = "N+F"
EDGE_SETS = ["Base+Ec", "Base+Ep", "Base+Ec+Ep"]


def make_feature_sets(ds: dict) -> dict[str, np.ndarray]:
  n = len(ds["y"])
  N = ds["node"].reshape(n, -1)
  F = ds["X_fbcca"]
  Ec = upper_tri_flat(ds["edge_coh"])
  Ep = upper_tri_flat(ds["edge_phase"])
  return {
      "N": N,
      "F": F,
      BASE: np.hstack([N, F]),
      "Ec": Ec,
      "Ep": Ep,
      "Ec+Ep": np.hstack([Ec, Ep]),
      "Base+Ec": np.hstack([N, F, Ec]),
      "Base+Ep": np.hstack([N, F, Ep]),
      "Base+Ec+Ep": np.hstack([N, F, Ec, Ep]),
  }


def oof_probabilities(X, y, folds, classes, k_select: int) -> np.ndarray:
  oof = np.zeros((len(y), len(classes)))
  for tr, te in folds:
    steps = [StandardScaler()]
    if k_select and X.shape[1] > k_select:
      steps.append(SelectKBest(f_classif, k=k_select))
    steps.append(LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    pipe = make_pipeline(*steps).fit(X[tr], y[tr])
    probs = pipe.predict_proba(X[te])
    for ci, cls in enumerate(pipe.classes_):
      oof[te, int(np.flatnonzero(classes == cls)[0])] = probs[:, ci]
  return oof


def score(oof: np.ndarray, y: np.ndarray, classes: np.ndarray) -> tuple[float, float]:
  """Return the 1.0 s accuracy and the 2.0 s causal-smoothed accuracy."""
  acc_1s = accuracy_score(y, classes[np.argmax(oof, axis=1)]) * 100.0
  preds = np.zeros_like(y)
  for i in range(len(y)):
    avg = oof[i] if i % 5 == 0 else np.mean(oof[i - 1:i + 1], axis=0)
    preds[i] = classes[np.argmax(avg)]
  return acc_1s, accuracy_score(y, preds) * 100.0


def evaluate_subject(sub: str, ds: dict, repeats: int, k_values: list[int]) -> list[dict]:
  y, groups = ds["y"], ds["trial_ids"]
  classes = np.unique(y)
  sets = make_feature_sets(ds)
  rows = []
  for r in range(repeats):
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42 + r)
    folds = list(skf.split(np.zeros(len(y)), y, groups=groups))
    for name, X in sets.items():
      for k in k_values:
        a1, a2 = score(oof_probabilities(X, y, folds, classes, k), y, classes)
        rows.append({"subject": sub, "set": name, "k_select": k, "repeat": r,
                     "dim": X.shape[1], "acc_1s": a1, "acc_2s": a2})
  return rows


def reference_table(path: Path) -> pd.DataFrame | None:
  if not path.exists():
    return None
  df = pd.read_csv(path)
  if "Subject" not in df.columns:
    return None
  return df.set_index("Subject")


def main():
  parser = argparse.ArgumentParser(description="Step 5: do connectivity features add to power-based features?")
  parser.add_argument("--subjects", nargs="+", default=None)
  parser.add_argument("--repeats", type=int, default=5, help="CV repeats with different fold shuffles (default 5).")
  parser.add_argument("--k-select", nargs="+", type=int, default=[0, 30],
                       help="In-fold ANOVA top-k values to try; 0 = no selection (default: 0 30).")
  parser.add_argument("--responders", nargs="+", default=None)
  parser.add_argument("--responder-threshold", type=float, default=80.0)
  parser.add_argument("--benchmark-csv", default=None)
  parser.add_argument("--rebuild", action="store_true")
  args = parser.parse_args()

  project_root = Path(__file__).resolve().parents[3]
  data_root = project_root / "data" / "processed"
  bench_csv = Path(args.benchmark_csv) if args.benchmark_csv else project_root / "reports" / "n_subjects_benchmark.csv"
  subjects = args.subjects or discover_subjects(data_root)
  responders, how = select_responders(subjects, args.responders, bench_csv, args.responder_threshold)
  ref = reference_table(bench_csv)

  print("=" * 96)
  print(f"CONNECTIVITY GATE  subjects={subjects}  repeats={args.repeats}  k_select={args.k_select}")
  print(f"Responders ({how}): {responders or 'none'}")
  print("=" * 96)

  rows = []
  for sub in subjects:
    ds = load_or_build(sub, data_root, rebuild=args.rebuild)
    if ds is None or len(np.unique(ds["y"])) < 2:
      continue
    print(f"[{sub}] {len(ds['y'])} windows, {ds['node'].shape[1]} channels ...")
    rows += evaluate_subject(sub, ds, args.repeats, args.k_select)
  if not rows:
    print("No results.")
    return

  df = pd.DataFrame(rows)
  out_dir = project_root / "reports"
  out_dir.mkdir(parents=True, exist_ok=True)
  df.to_csv(out_dir / "connectivity_gate_benchmark.csv", index=False)

  per_sub = df.groupby(["subject", "set", "k_select"], as_index=False)[["dim", "acc_1s", "acc_2s"]].mean()
  order = list(make_feature_sets(load_or_build(subjects[0], data_root)).keys())

  for k in args.k_select:
    label = "all features" if k == 0 else f"in-fold top-{k} ANOVA"
    sub_k = per_sub[per_sub["k_select"] == k]
    for metric, mname in (("acc_1s", "1.0 s accuracy (%)"), ("acc_2s", "2.0 s smoothed accuracy (%)")):
      piv = sub_k.pivot(index="set", columns="subject", values=metric).reindex(order)
      piv["mean(all)"] = piv.mean(axis=1)
      if responders:
        piv["mean(resp)"] = piv[[s for s in responders if s in piv.columns]].mean(axis=1)
      print(f"\n--- {mname}, {label} (mean over {args.repeats} repeats) ---")
      print(piv.round(1).to_string())

  if ref is not None:
    print("\nReference (existing benchmark, FBCSP m=1 + FBCCA -> Shrinkage-LDA):")
    cols = [c for c in ("1.0s Acc (%)", "2.0s Smoothed (%)") if c in ref.columns]
    print(ref.loc[[s for s in subjects if s in ref.index], cols].to_string())

  print("\n" + "=" * 96)
  print("DOES CONNECTIVITY ADD TO THE POWER-BASED REFERENCE?  (Base = N+F; delta in accuracy points, 1.0 s)")
  print("Per subject: mean delta [improved in x/R repeats]; verdict uses the responders.")
  print("=" * 96)
  verdicts = {}
  for k in args.k_select:
    label = "all features" if k == 0 else f"top-{k}"
    for es in EDGE_SETS:
      cells, deltas = [], []
      for sub in subjects:
        a = df[(df.subject == sub) & (df.set == BASE) & (df.k_select == k)].sort_values("repeat")["acc_1s"].to_numpy()
        b = df[(df.subject == sub) & (df.set == es) & (df.k_select == k)].sort_values("repeat")["acc_1s"].to_numpy()
        if len(a) == 0 or len(a) != len(b):
          continue
        d = b - a
        cells.append(f"{sub}:{d.mean():+.1f}[{int((d > 0).sum())}/{len(d)}]")
        if sub in responders:
          deltas.append(d.mean())
      mean_resp = float(np.mean(deltas)) if deltas else float("nan")
      n_imp = int(np.sum(np.array(deltas) >= 2.0)) if deltas else 0
      verdicts[(es, k)] = (mean_resp, n_imp, len(deltas))
      print(f"  {es:<11s} ({label:>9s}) | " + "  ".join(cells) + f" || resp mean {mean_resp:+.1f}")

  print("\nGate heuristic (descriptive): edge features add >= 2 pts on average AND in >= 2/3 of responders?")
  passed = [(es, k) for (es, k), (m, n_imp, n) in verdicts.items()
            if n and m >= 2.0 and n_imp >= int(np.ceil(2 * n / 3))]
  if passed:
    print("  PASS for:", ", ".join(f"{es} (k={k or 'all'})" for es, k in passed),
          "-> a graph model on these features is worth trying.")
  else:
    print("  NO PASS -> connectivity features do not add reliably to power-based ones here; a GNN on them is unlikely")
    print("  to beat FBCSP+FBCCA. Keep connectivity as an explanatory result, or move to learned-graph models on raw/spectral input.")
  print(f"\nPer-repeat results saved to: {out_dir / 'connectivity_gate_benchmark.csv'}")


if __name__ == "__main__":
  main()