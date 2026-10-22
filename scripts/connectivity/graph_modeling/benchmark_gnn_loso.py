"""Leave-one-subject-out (LOSO) benchmark for the edge-gated connectivity GNN.

For each subject S: train the GNN on ALL other subjects, test on S (never seen
during training). This is the cross-subject counterpart of
``benchmark_gnn.py`` (within-subject, trial-grouped 5-fold CV) and uses the
same data, channels and scoring as ``scripts/benchmarks/benchmark_loso.py``,
so its rows can be put next to the fbcca / lda / eegnet LOSO rows directly.

Build the data first (the same folder benchmark_loso.py reads):
  python scripts/build_windows.py --window-sec 1 --stride-sec 0.25 --banks fbcca fbcsp

What is computed per subject (once, in memory)
  nodes  log10 band power per channel at the 5 targets + 2nd harmonics (10)
  coh    per-window coherence between channel pairs in each target band (5)
  phase  per-window phase-difference phasor (cos, sin) in each band (10)
  F      FBCCA scores (5), added after pooling
Variants pick from these exactly as in benchmark_gnn.py ("nodes+F",
"nodes+coh+F", ...).

Normalization (``--norm``, both by default -> two rows per variant)
  pooled   features standardized with the pooled TRAINING subjects' mean/std
           (the direct analogue of the within-subject benchmark).
  subject  every subject's features are first z-scored with that subject's
           OWN mean/std over its windows. Uses no labels of the test subject,
           only its unlabeled windows; report it as "unsupervised per-subject
           normalization" (same idea as lda_subjnorm in benchmark_loso.py).

Training: AdamW / cosine / label smoothing as benchmark_gnn.py, plus early
stopping on 20% of the training TRIALS (grouped, never the test subject), as
the networks in benchmark_loso.py. ``--val-frac 0`` disables early stopping and
trains a fixed ``--epochs`` (the original within-subject protocol).
Each fold is trained with ``--seeds`` different seeds; accuracy is the mean.

Channels: the GNN learns one embedding per channel position, so every subject
needs the same channel list. Default ``--channels common`` keeps only the
channels kept by the quality gate in every subject (same as benchmark_loso.py).

Resumable: every finished (method, test subject) fold is saved under
``<out-dir>/folds/`` and skipped on the next run, so a Colab disconnect only
loses the fold in progress. A saved fold trained with different settings is
refused, so use a new --out-dir for a new configuration.

Usage:
  python scripts/connectivity/graph_modeling/benchmark_gnn_loso.py --smoke-test
  python scripts/connectivity/graph_modeling/benchmark_gnn_loso.py
  python scripts/connectivity/graph_modeling/benchmark_gnn_loso.py \
      --variants nodes+F nodes+coh+F nodes+coh+phase+F --norm subject --out-dir reports/loso_gnn_w1
  # within-subject reference on exactly the same windows/channels/training:
  python scripts/connectivity/graph_modeling/benchmark_gnn_loso.py --protocol within --out-dir reports/within_gnn
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))
from benchmark_loso import choose_channels, fold_path, load_subject, smoothed_acc  # noqa: E402
from benchmark_window_sweep import (  # noqa: E402  (also puts the repo root on sys.path)
    PROJECT_ROOT,
    STIM_CONDITIONS,
    SUBJECT_DIR,
    SWEEP_DIR,
    prediction_rows,
    summarize,
)
from src.features.connectivity.connectivity_graph_features import TARGET_FREQS, build_graph_features  # noqa: E402
from src.features.connectivity.graph_inputs import (  # noqa: E402
    apply_standardizer,
    build_inputs,
    fit_standardizer,
    parse_variant,
)

DEFAULT_VARIANTS = ["nodes+F", "nodes+coh+F"]
NORMS = ("pooled", "subject")


# --------------------------------------------------------------------------- data

def graph_subject(d: dict) -> dict:
  """Per-window graph features of one subject's clean stimulus windows."""
  feats = build_graph_features(np.asarray(d["X"], dtype=np.float64), d["fs"], TARGET_FREQS)
  return {**feats, "X_fbcca": np.asarray(d["X_cca"], dtype=np.float64)}


def subject_arrays(g: dict, variant: str, norm: str) -> tuple:
  """(nodes, edges, graph) of one subject, z-scored with its own statistics if norm == 'subject'."""
  arrays = build_inputs(g, variant)
  if norm == "subject":
    arrays = tuple(None if a is None else apply_standardizer(a, fit_standardizer(a)) for a in arrays)
  return arrays


def pool(per_subject: list[tuple]) -> tuple:
  return tuple(None if parts[0] is None else np.concatenate(parts) for parts in zip(*per_subject))


def standardize_with_train(train: tuple, test: tuple) -> tuple[tuple, tuple]:
  """Standardize every array with the pooled training subjects' statistics."""
  tr, te = [], []
  for a, b in zip(train, test):
    if a is None:
      tr.append(None)
      te.append(None)
      continue
    stats = fit_standardizer(a)
    tr.append(apply_standardizer(a, stats))
    te.append(apply_standardizer(b, stats))
  return tuple(tr), tuple(te)


def method_name(variant: str, norm: str) -> str:
  return f"gnn_{variant}" + ("_subjnorm" if norm == "subject" else "")


# ----------------------------------------------------------------------- training

def _predict(model, arrays: tuple, device, batch: int = 256) -> np.ndarray:
  import torch
  model.eval()
  n = arrays[0].shape[0]
  out = []
  with torch.no_grad():
    for i in range(0, n, batch):
      parts = [None if a is None else a[i:i + batch].to(device) for a in arrays]
      out.append(torch.softmax(model(*parts), dim=-1).cpu())
  return torch.cat(out).numpy()


def fit_gnn(train: tuple, y: np.ndarray, groups: np.ndarray, args, seed: int):
  """Train one GNN on the pooled training subjects; early stopping on held-out training trials."""
  import torch
  import torch.nn as nn
  from torch.optim import AdamW
  from torch.optim.lr_scheduler import CosineAnnealingLR

  from src.models.graph.connectivity_gnn import ConnectivityGNN

  torch.manual_seed(seed)
  np.random.seed(seed)
  dev = torch.device(args.device)
  idx = np.arange(len(y))
  if args.val_frac > 0:
    inner = StratifiedGroupKFold(n_splits=max(2, int(round(1 / args.val_frac))), shuffle=True, random_state=seed)
    fit, val = next(inner.split(idx, y, groups=groups))
  else:
    fit, val = idx, np.array([], dtype=int)

  T = tuple(None if a is None else torch.as_tensor(a, dtype=torch.float32, device=dev) for a in train)
  y_t = torch.as_tensor(y, dtype=torch.long, device=dev)
  nodes, edges, graph = train
  model = ConnectivityGNN(
      n_nodes=nodes.shape[1],
      n_node_feats=nodes.shape[-1],
      n_edge_feats=0 if edges is None else edges.shape[-1],
      n_graph_feats=0 if graph is None else graph.shape[-1],
      n_classes=len(STIM_CONDITIONS),
      hidden=args.hidden,
      n_layers=args.layers,
      dropout=args.dropout,
  ).to(dev)
  criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
  eval_loss = nn.CrossEntropyLoss()
  optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
  scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
  gen = torch.Generator().manual_seed(seed)
  fit_t = torch.as_tensor(fit)
  val_t = torch.as_tensor(val)

  def take(sel):
    return tuple(None if a is None else a[sel.to(dev)] for a in T)

  T_val = take(val_t) if len(val) else None
  best = {"loss": np.inf, "epoch": args.epochs, "state": None, "val_acc": np.nan}
  for epoch in range(1, args.epochs + 1):
    model.train()
    perm = fit_t[torch.randperm(len(fit_t), generator=gen)]
    for i in range(0, len(perm), args.batch_size):
      b = perm[i:i + args.batch_size]
      optimizer.zero_grad()
      criterion(model(*take(b)), y_t[b.to(dev)]).backward()
      optimizer.step()
    scheduler.step()
    if T_val is None:
      continue
    p_val = _predict(model, T_val, dev)
    v_loss = float(eval_loss(torch.log(torch.as_tensor(p_val) + 1e-8), torch.as_tensor(y[val])))
    if v_loss < best["loss"]:
      best = {"loss": v_loss, "epoch": epoch, "state": copy.deepcopy(model.state_dict()),
              "val_acc": float((p_val.argmax(1) == y[val]).mean() * 100)}
    elif epoch - best["epoch"] >= args.patience:
      break
  if best["state"] is not None:
    model.load_state_dict(best["state"])
  train_acc = float((_predict(model, take(fit_t), dev).argmax(1) == y[fit]).mean() * 100)
  return model, {"train": train_acc, "val": best["val_acc"], "epoch": best["epoch"]}


def gnn_fold(graphs: dict, data: dict, test_sub: str, variant: str, norm: str, args) -> tuple[np.ndarray, dict]:
  """Train on every subject except test_sub; return (seeds, n_test, 5) probabilities."""
  import torch

  label = {int(c): i for i, c in enumerate(STIM_CONDITIONS)}
  train_subs = [s for s in data if s != test_sub]
  tr = pool([subject_arrays(graphs[s], variant, norm) for s in train_subs])
  te = subject_arrays(graphs[test_sub], variant, norm)
  tr, te = standardize_with_train(tr, te)
  y = np.array([label[int(c)] for s in train_subs for c in data[s]["y"]], dtype=np.int64)
  # Trial ids restart in every subject; offset them so the validation split never mixes subjects' trials.
  groups, offset = [], 0
  for s in train_subs:
    groups.append(data[s]["trial_ids"] + offset)
    offset += int(data[s]["trial_ids"].max()) + 1
  groups = np.concatenate(groups)

  dev = torch.device(args.device)
  te_t = tuple(None if a is None else torch.as_tensor(a, dtype=torch.float32, device=dev) for a in te)
  probs, infos = [], []
  for k in range(args.seeds):
    model, info = fit_gnn(tr, y, groups, args, seed=args.seed + k)
    probs.append(_predict(model, te_t, dev))
    infos.append(info)
  mean = {key: float(np.mean([i[key] for i in infos])) for key in ("train", "val", "epoch")}
  return np.stack(probs), mean


def within_fold(g: dict, d: dict, variant: str, args) -> tuple[np.ndarray, dict]:
  """Within-subject reference on the SAME windows: trial-grouped 5-fold CV, out-of-fold probabilities."""
  import torch

  label = {int(c): i for i, c in enumerate(STIM_CONDITIONS)}
  y = np.array([label[int(c)] for c in d["y"]], dtype=np.int64)
  arrays = build_inputs(g, variant)
  dev = torch.device(args.device)
  probs = np.zeros((args.seeds, len(y), len(STIM_CONDITIONS)))
  infos = []
  for k in range(args.seeds):
    cv = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed + k)
    for tr, te in cv.split(np.zeros(len(y)), y, groups=d["trial_ids"]):
      a_tr, a_te = standardize_with_train(tuple(None if a is None else a[tr] for a in arrays),
                                          tuple(None if a is None else a[te] for a in arrays))
      model, info = fit_gnn(a_tr, y[tr], d["trial_ids"][tr], args, seed=args.seed + k)
      te_t = tuple(None if a is None else torch.as_tensor(a, dtype=torch.float32, device=dev) for a in a_te)
      probs[k, te] = _predict(model, te_t, dev)
      infos.append(info)
  mean = {key: float(np.mean([i[key] for i in infos])) for key in ("train", "val", "epoch")}
  return probs, mean


# ------------------------------------------------------------------------- driver

def fold_config(variant: str, norm: str, channels: list[str], args) -> str:
  """Settings that must match for a saved fold to be reused."""
  return json.dumps({
      "protocol": args.protocol, "folds": args.folds if args.protocol == "within" else None,
      "variant": variant, "norm": norm, "channels": channels, "seeds": args.seeds, "seed": args.seed,
      "epochs": args.epochs, "patience": args.patience, "val_frac": args.val_frac,
      "batch_size": args.batch_size, "lr": args.lr, "hidden": args.hidden, "layers": args.layers,
      "dropout": args.dropout,
  }, sort_keys=True)


def result_path(args, method: str, sub: str) -> Path:
  if args.protocol == "loso":
    return fold_path(args.out_dir, method, sub)
  return args.out_dir / "folds" / f"{method}__within-{sub}.npz"


def run_loso(data: dict, channels: list[str], args) -> pd.DataFrame | None:
  """Run (or resume) every (variant, norm, subject) fold, then summarize."""
  for v in args.variants:
    parse_variant(v)  # fail early on a typo
  if args.protocol == "within" and "subject" in args.norm:
    print("  --protocol within: per-subject normalization is the same as pooled here; running 'pooled' only")
    args.norm = ["pooled"]
  t0 = time.time()
  graphs = {}
  for s, d in data.items():
    graphs[s] = graph_subject(d)
    bad = [k for k in ("node", "edge_coh", "edge_phase", "X_fbcca") if not np.isfinite(graphs[s][k]).all()]
    if bad:
      raise ValueError(f"{s}: non-finite values in {bad}")
  print(f"  graph features for {len(graphs)} subjects in {time.time() - t0:.0f} s "
        f"(node {graphs[next(iter(graphs))]['node'].shape[1:]}, edges {graphs[next(iter(graphs))]['edge_coh'].shape[1:]})")

  methods = []
  for variant in args.variants:
    for norm in args.norm:
      method = method_name(variant, norm)
      methods.append(method)
      cfg = fold_config(variant, norm, channels, args)
      for test_sub in data:
        path = result_path(args, method, test_sub)
        if path.exists():
          saved = str(np.load(path)["config"])
          if saved != cfg:
            raise SystemExit(f"{path} was trained with other settings:\n  saved: {saved}\n  now:   {cfg}\n"
                             "Use a new --out-dir for a new configuration.")
          print(f"  [=] {method:28s} test={test_sub}: already done, skipping")
          continue
        ft = time.time()
        if args.protocol == "loso":
          probs, info = gnn_fold(graphs, data, test_sub, variant, norm, args)
        else:
          probs, info = within_fold(graphs[test_sub], data[test_sub], variant, args)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, probs=probs, train=info["train"], val=info["val"], epoch=info["epoch"],
                 seconds=time.time() - ft, config=cfg)
        accs = [(STIM_CONDITIONS[p.argmax(1)] == data[test_sub]["y"]).mean() * 100 for p in probs]
        val = f", val {info['val']:.0f}%, best epoch {info['epoch']:.0f}" if np.isfinite(info["val"]) else ""
        print(f"  [+] {method:28s} test={test_sub}: {np.mean(accs):6.2f}% (+/-{np.std(accs):.1f} over "
              f"{len(accs)} seeds) | train {info['train']:.0f}%{val} ({time.time() - ft:.0f} s)", flush=True)

  records, preds = [], []
  for method in methods:
    for sub, d in data.items():
      path = result_path(args, method, sub)
      if not path.exists():
        continue
      f = np.load(path)
      runs = list(f["probs"])
      rec = summarize(sub, d, method, runs, args.gaze_sec, [float(f["train"])])
      rec["Protocol"] = args.protocol
      rec["2-tap Smoothed (%)"] = round(float(np.mean([smoothed_acc(p, d) for p in runs])), 2)
      rec["Val Acc (%)"] = round(float(f["val"]), 2) if np.isfinite(f["val"]) else np.nan
      rec["Best Epoch"] = float(f["epoch"])
      rec["Channels"] = len(channels)
      records.append(rec)
      preds += [prediction_rows(sub, d, method, r, p) for r, p in enumerate(runs)]
  if not records:
    return None
  df = pd.DataFrame(records)
  args.out_dir.mkdir(parents=True, exist_ok=True)
  df.to_csv(args.out_dir / "summary.csv", index=False)
  pred_df = pd.concat(preds, ignore_index=True)
  pred_df.to_csv(args.out_dir / "predictions.csv", index=False)
  (pred_df.groupby(["method", "true", "pred"]).size().rename("count").reset_index()
   .to_csv(args.out_dir / "confusion.csv", index=False))

  present = [m for m in methods if m in set(df["Method"])]
  print("=" * 96)
  print("PER SUBJECT (accuracy % when that subject is the unseen test subject, mean over seeds)"
        if args.protocol == "loso" else
        f"PER SUBJECT, WITHIN-SUBJECT reference (trial-grouped {args.folds}-fold CV, mean over seeds)")
  print(df.pivot(index="Subject", columns="Method", values="Acc (%)")[present].round(2).to_string())
  print("-" * 96)
  print("COHORT (mean +/- SD over held-out subjects; chance = 20%)")
  cohort = df.groupby("Method").agg(**{
      "Acc %": ("Acc (%)", "mean"), "SD": ("Acc (%)", "std"),
      "Smoothed %": ("2-tap Smoothed (%)", "mean"), "BalAcc %": ("BalAcc (%)", "mean"),
      "ITR": ("ITR (bits/min)", "mean"), "Train %": ("Train Acc (%)", "mean")})
  print(cohort.reindex(present).round(2).to_string())
  if args.protocol == "loso":
    print("  Train % near 100 with test near chance -> the model memorizes the training subjects;")
    print("  compare the _subjnorm rows, and the fbcca / lda_subjnorm rows of benchmark_loso.py.")
  print(f"Saved summary.csv, predictions.csv, confusion.csv to {args.out_dir} ({time.time() - t0:.0f} s)")
  return df


# ---------------------------------------------------------------------- smoke test

def smoke_test(args) -> None:
  """End-to-end LOSO on synthetic SSVEP subjects (different gain and noise per subject)."""
  import tempfile

  rng = np.random.default_rng(0)
  fs, n_ch, n_trials, per_trial = 256.0, 4, 6, 5
  t = np.arange(int(fs)) / fs
  freqs = dict(zip(STIM_CONDITIONS, TARGET_FREQS))
  data = {}
  for s in range(4):
    gain, noise = 1.0 + s, 1.0 + 0.5 * s
    X, y, tid, off = [], [], [], []
    trial = 0
    for c in STIM_CONDITIONS:
      for _ in range(n_trials):
        ph = rng.uniform(0, 2 * np.pi)
        for w in range(per_trial):
          lag = w * 0.25
          sig = [gain * np.sin(2 * np.pi * freqs[c] * (t + lag) + ph + 0.3 * ch) for ch in range(n_ch)]
          X.append(np.array(sig) + noise * rng.standard_normal((n_ch, t.size)))
          y.append(c)
          tid.append(trial)
          off.append(lag)
        trial += 1
    X = np.array(X)
    # Stand-in for FBCCA scores: relative power at each target frequency.
    spec = np.abs(np.fft.rfft(X, axis=-1)) ** 2
    f_axis = np.fft.rfftfreq(t.size, 1 / fs)
    cca = np.stack([spec[:, :, np.argmin(np.abs(f_axis - f))].mean(1) for f in TARGET_FREQS], 1)
    data[f"S0{s + 1}"] = {"X": X, "y": np.array(y), "trial_ids": np.array(tid), "offset_sec": np.array(off),
                          "fs": fs, "window_sec": 1.0, "X_cca": cca / cca.sum(1, keepdims=True)}
  args.variants = ["nodes", "nodes+coh+phase+F"]
  args.norm = list(NORMS)
  args.epochs, args.patience, args.seeds = 30, 10, 2
  with tempfile.TemporaryDirectory() as tmp:
    args.out_dir = Path(tmp)
    df = run_loso(data, [f"ch{i}" for i in range(n_ch)], args)
    assert df is not None and len(df) == 4 * 4, len(df)
    print("\nRerun (every fold must be skipped):")
    t0 = time.time()
    run_loso(data, [f"ch{i}" for i in range(n_ch)], args)
    assert time.time() - t0 < 30
    print("\nWithin-subject reference protocol:")
    args.protocol, args.seeds, args.variants = "within", 1, ["nodes+F"]
    args.out_dir = Path(tmp) / "within"
    dfw = run_loso(data, [f"ch{i}" for i in range(n_ch)], args)
    assert dfw is not None and len(dfw) == 4 and set(dfw["Protocol"]) == {"within"}
  mean_acc = df["Acc (%)"].mean()
  assert mean_acc > 40, f"synthetic LOSO accuracy {mean_acc:.1f}% is too close to chance"
  print(f"\nSmoke test passed (mean synthetic LOSO accuracy {mean_acc:.1f}%, "
        f"within {dfw['Acc (%)'].mean():.1f}%).")


# -------------------------------------------------------------------------- main

def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--root", type=Path, default=PROJECT_ROOT / "data" / "processed_continuous",
                  help="Folder holding the w<len>_s<stride> datasets (same as benchmark_loso.py)")
  ap.add_argument("--window", type=float, default=1.0, help="Window length in s (default 1.0)")
  ap.add_argument("--stride-sec", type=float, default=0.25)
  ap.add_argument("--subjects", nargs="*", default=None, help="Subset of subjects (default: all found)")
  ap.add_argument("--channels", choices=("common", "all"), default="common")
  ap.add_argument("--protocol", choices=("loso", "within"), default="loso",
                  help="loso (default), or the within-subject reference on the same windows and channels")
  ap.add_argument("--folds", type=int, default=5, help="CV folds for --protocol within (default 5)")
  ap.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS,
                  help="GNN input variants, as in benchmark_gnn.py (default: nodes+F nodes+coh+F)")
  ap.add_argument("--norm", nargs="+", choices=NORMS, default=list(NORMS))
  ap.add_argument("--seeds", type=int, default=3, help="Training seeds per fold (default 3)")
  ap.add_argument("--seed", type=int, default=42)
  ap.add_argument("--epochs", type=int, default=150, help="Maximum epochs (early stopping usually ends sooner)")
  ap.add_argument("--patience", type=int, default=25)
  ap.add_argument("--val-frac", type=float, default=0.2, help="0 = no early stopping, fixed --epochs")
  ap.add_argument("--batch-size", type=int, default=32)
  ap.add_argument("--lr", type=float, default=1e-3)
  ap.add_argument("--hidden", type=int, default=32)
  ap.add_argument("--layers", type=int, default=2)
  ap.add_argument("--dropout", type=float, default=0.3)
  ap.add_argument("--device", default=None, help="cpu / cuda (default: cuda if available)")
  ap.add_argument("--gaze-sec", type=float, default=0.5)
  ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "reports" / "loso_gnn")
  ap.add_argument("--smoke-test", action="store_true", help="Synthetic end-to-end check, no data needed")
  args = ap.parse_args()

  if args.device is None:
    import torch
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
  if args.smoke_test:
    args.device = "cpu"
    smoke_test(args)
    return

  data_dir = None
  for p in sorted(args.root.iterdir()) if args.root.exists() else []:
    m = SWEEP_DIR.match(p.name)
    if p.is_dir() and m and np.isclose(float(m.group(1)), args.window) and np.isclose(float(m.group(2)), args.stride_sec):
      data_dir = p
  if data_dir is None:
    print(f"No w{args.window:g}_s{args.stride_sec:g} folder under {args.root}. Build it with scripts/build_windows.py.")
    return
  sub_dirs = {p.name: p for p in sorted(data_dir.iterdir()) if p.is_dir() and SUBJECT_DIR.match(p.name)}
  if args.subjects:
    sub_dirs = {s: p for s, p in sub_dirs.items() if s in args.subjects}
  if len(sub_dirs) < 2:
    print(f"LOSO needs at least 2 subjects; found {list(sub_dirs)} in {data_dir}")
    return

  channels = choose_channels(sub_dirs, args.channels)
  print("=" * 96)
  print(f"GNN LOSO | {len(sub_dirs)} subjects {list(sub_dirs)} | data {data_dir}")
  print(f"channels ({args.channels}): {channels} | variants {args.variants} | norm {args.norm}")
  print(f"device {args.device} | seeds {args.seeds} | epochs <= {args.epochs}, patience {args.patience}, "
        f"val {args.val_frac:g} of training trials")
  print("=" * 96)
  data = {}
  for s, p in sub_dirs.items():
    data[s] = load_subject(p, channels)
    d = data[s]
    print(f"  loaded {s}: {len(d['y'])} clean windows, {len(np.unique(d['trial_ids']))} trials, fs={d['fs']:g}")
  if len({d["fs"] for d in data.values()}) > 1:
    raise ValueError("Subjects have different sampling rates; rebuild them with the same settings")
  run_loso(data, channels, args)


if __name__ == "__main__":
  main()