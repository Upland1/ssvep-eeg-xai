"""Benchmark GNN variants on the cached per-window graph datasets.

The benchmark compares node features, edge features, and optional FBCCA
scores using the same grouped cross-validation and scoring as the LDA gate.
Standardisation is fitted within each training fold, and ``--smoke-test``
checks all variants with random data before a full run.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from scripts.connectivity.analysis.analyze_connectivity_cohort import discover_subjects
from scripts.connectivity.comparison.compare_connectivity_frequencies import select_responders
from scripts.connectivity.graph_modeling.benchmark_connectivity_gate import score
from scripts.connectivity.graph_modeling.build_graph_dataset import load_or_build
from src.features.connectivity.graph_inputs import build_inputs, standardize_fold
from src.models.graph.connectivity_gnn import ConnectivityGNN

DEFAULT_VARIANTS = ["nodes", "nodes+F", "nodes+coh+F", "nodes+phase+F", "nodes+coh+phase+F"]


def to_tensor(a, device):
  return None if a is None else torch.as_tensor(a, dtype=torch.float32, device=device)


def make_model(n_nodes, nodes, edges, graph, args) -> ConnectivityGNN:
  return ConnectivityGNN(
      n_nodes=n_nodes,
      n_node_feats=nodes.shape[-1],
      n_edge_feats=0 if edges is None else edges.shape[-1],
      n_graph_feats=0 if graph is None else graph.shape[-1],
      n_classes=5,
      hidden=args.hidden,
      n_layers=args.layers,
      dropout=args.dropout,
  )


def train_and_predict(tr, te, y_tr, n_nodes, args, seed: int) -> np.ndarray:
  """Train on one fold and return test probabilities and train accuracy."""
  torch.manual_seed(seed)
  dev = args.device
  nodes_tr, edges_tr, graph_tr = (to_tensor(a, dev) for a in tr)
  nodes_te, edges_te, graph_te = (to_tensor(a, dev) for a in te)
  y_t = torch.as_tensor(y_tr, dtype=torch.long, device=dev)

  model = make_model(n_nodes, tr[0], tr[1], tr[2], args).to(dev)
  criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
  optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
  scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

  n_tr = nodes_tr.shape[0]
  model.train()
  for _ in range(args.epochs):
    perm = torch.randperm(n_tr, device=dev)
    for i in range(0, n_tr, args.batch_size):
      idx = perm[i:i + args.batch_size]
      optimizer.zero_grad()
      out = model(
          nodes_tr[idx],
          None if edges_tr is None else edges_tr[idx],
          None if graph_tr is None else graph_tr[idx],
      )
      loss = criterion(out, y_t[idx])
      loss.backward()
      optimizer.step()
    scheduler.step()

  model.eval()
  with torch.no_grad():
    logits = model(nodes_te, edges_te, graph_te)
    train_pred = model(nodes_tr, edges_tr, graph_tr).argmax(dim=-1)
    train_acc = float((train_pred == y_t).float().mean().item()) * 100.0
    return torch.softmax(logits, dim=-1).cpu().numpy(), train_acc


def evaluate_variant(ds: dict, variant: str, repeats: int, args) -> list[dict]:
  y = ds["y"]
  classes = np.unique(y)
  y_mapped = np.searchsorted(classes, y)
  groups = ds["trial_ids"]
  arrays = build_inputs(ds, variant)
  n_nodes = arrays[0].shape[1]
  rows = []
  for r in range(repeats):
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42 + r)
    oof = np.zeros((len(y), len(classes)))
    train_accs = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(len(y)), y_mapped, groups=groups)):
      tr, te = standardize_fold(arrays, tr_idx, te_idx)
      oof[te_idx], tr_acc = train_and_predict(tr, te, y_mapped[tr_idx], n_nodes, args, seed=1000 * r + fold)
      train_accs.append(tr_acc)
    a1, a2 = score(oof, y_mapped, np.arange(len(classes)))
    rows.append({"variant": variant, "repeat": r, "acc_1s": a1, "acc_2s": a2,
                 "train_acc": float(np.mean(train_accs))})
  return rows


def smoke_test(args) -> None:
  """Check model shapes and gradients with random data."""
  torch.manual_seed(0)
  n_nodes, batch = 7, 4
  for variant in DEFAULT_VARIANTS:
    ds = {
        "node": np.random.randn(batch, n_nodes, 10),
        "edge_coh": np.random.rand(batch, n_nodes, n_nodes, 5),
        "edge_phase": np.random.randn(batch, n_nodes, n_nodes, 5, 2),
        "X_fbcca": np.random.rand(batch, 5),
    }
    nodes, edges, graph = (to_tensor(a, "cpu") for a in build_inputs(ds, variant))
    model = make_model(n_nodes, nodes, edges, graph, args)
    out = model(nodes, edges, graph)
    assert out.shape == (batch, 5), out.shape
    nn.CrossEntropyLoss()(out, torch.zeros(batch, dtype=torch.long)).backward()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  [ok] {variant:<22s} out={tuple(out.shape)}  params={n_params}")
  print("Smoke test passed.")


def main():
  parser = argparse.ArgumentParser(description="Step 6: GNN benchmark on per-window channel graphs.")
  parser.add_argument("--subjects", nargs="+", default=None)
  parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
  parser.add_argument("--repeats", type=int, default=3, help="CV repeats with different fold shuffles (default 3).")
  parser.add_argument("--epochs", type=int, default=150)
  parser.add_argument("--batch-size", type=int, default=32)
  parser.add_argument("--lr", type=float, default=1e-3)
  parser.add_argument("--hidden", type=int, default=32)
  parser.add_argument("--layers", type=int, default=2)
  parser.add_argument("--dropout", type=float, default=0.3)
  parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  parser.add_argument("--responders", nargs="+", default=None)
  parser.add_argument("--responder-threshold", type=float, default=80.0)
  parser.add_argument("--benchmark-csv", default=None)
  parser.add_argument("--smoke-test", action="store_true")
  args = parser.parse_args()

  if args.smoke_test:
    smoke_test(args)
    return

  project_root = Path(__file__).resolve().parents[3]
  data_root = project_root / "data" / "processed"
  bench_csv = Path(args.benchmark_csv) if args.benchmark_csv else project_root / "reports" / "n_subjects_benchmark.csv"
  subjects = args.subjects or discover_subjects(data_root)
  responders, how = select_responders(subjects, args.responders, bench_csv, args.responder_threshold)

  print("=" * 96)
  print(f"GNN BENCHMARK  subjects={subjects}  variants={args.variants}  repeats={args.repeats}  device={args.device}")
  print(f"Responders ({how}): {responders or 'none'}")
  print("=" * 96)

  rows = []
  for sub in subjects:
    ds = load_or_build(sub, data_root)
    if ds is None:
      continue
    for variant in args.variants:
      res = evaluate_variant(ds, variant, args.repeats, args)
      for rrow in res:
        rrow["subject"] = sub
      rows += res
      m1 = np.mean([x["acc_1s"] for x in res])
      m2 = np.mean([x["acc_2s"] for x in res])
      mt = np.mean([x["train_acc"] for x in res])
      print(f"[{sub}] {variant:<20s} 1.0s: {m1:6.2f}%  2.0s: {m2:6.2f}%  | train: {mt:6.2f}%")
  if not rows:
    print("No results.")
    return

  df = pd.DataFrame(rows)
  out_dir = project_root / "reports"
  out_dir.mkdir(parents=True, exist_ok=True)
  df.to_csv(out_dir / "gnn_benchmark.csv", index=False)

  per_sub = df.groupby(["subject", "variant"], as_index=False)[["acc_1s", "acc_2s", "train_acc"]].mean()
  # Add LDA results from the Step 5 gate when available.
  gate_csv = out_dir / "connectivity_gate_benchmark.csv"
  lda_rows = []
  if gate_csv.exists():
    g = pd.read_csv(gate_csv)
    g = g[(g.k_select == 0) & g.set.isin(["N+F", "Base+Ec"])]
    for (sub, st), grp in g.groupby(["subject", "set"]):
      lda_rows.append({"subject": sub, "variant": f"LDA {st}", "acc_1s": grp.acc_1s.mean(), "acc_2s": grp.acc_2s.mean()})
  per_sub = pd.concat([per_sub, pd.DataFrame(lda_rows)], ignore_index=True)
  order = args.variants + [v for v in ("LDA N+F", "LDA Base+Ec") if v in set(per_sub.variant)]

  for metric, name in (("acc_1s", "1.0 s accuracy (%)"), ("acc_2s", "2.0 s smoothed accuracy (%)")):
    piv = per_sub.pivot(index="variant", columns="subject", values=metric).reindex(order)
    piv["mean(all)"] = piv.mean(axis=1)
    if responders:
      piv["mean(resp)"] = piv[[s for s in responders if s in piv.columns]].mean(axis=1)
    print(f"\n--- {name} ---")
    print(piv.round(1).to_string())

  tr_piv = per_sub[per_sub.variant.isin(args.variants)].pivot(index="variant", columns="subject", values="train_acc")
  print("\n--- GNN training-set accuracy (%) -- diagnostic, compare with the 1.0 s test accuracy above ---")
  print(tr_piv.reindex(args.variants).round(1).to_string())
  print("  ~100% train but test far below LDA  -> overfitting: try --hidden 16 --dropout 0.5 or fewer --epochs")
  print("  train well below 100%               -> underfitting: try --epochs 300 --lr 3e-3 --dropout 0.1")
  print(f"\nPer-repeat results saved to: {out_dir / 'gnn_benchmark.csv'}")


if __name__ == "__main__":
  main()