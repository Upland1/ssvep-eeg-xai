"""Step 0: freeze the current (v1) pipeline state as a reference baseline.

Copies the processed data, result reports and config into an immutable
snapshot folder, records checksums/shapes/environment in a manifest, and
builds one summary table (`baseline_summary.csv`) that becomes the "v1"
column of the final v1-vs-v2 comparison.

Nothing in the working tree is modified or deleted; the snapshot is a copy.

Usage (from the repo root, ssvep-eeg-xai/):
    python scripts/freeze_baseline.py
    python scripts/freeze_baseline.py --tag v1 --note "pre edge-effect fix"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Reports produced BEFORE the trial-leakage fix. Kept for the record but
# filed under superseded/ so they are never mistaken for baseline numbers.
SUPERSEDED_REPORTS = {"five_subjects_benchmark.csv"}

COPY_TARGETS = [
    "data/processed",
    "reports",
    "outputs/reports",
    "configs",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd: list[str], cwd: Path) -> str | None:
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def describe_file(path: Path) -> dict:
    info = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    try:
        if path.suffix == ".npy":
            arr = np.load(path, mmap_mode="r", allow_pickle=False)
            info.update(shape=list(arr.shape), dtype=str(arr.dtype))
        elif path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as z:
                info["arrays"] = {k: list(z[k].shape) for k in z.files}
        elif path.suffix == ".csv":
            info["rows"] = int(len(pd.read_csv(path)))
    except Exception as exc:  # describe what we can; never fail the freeze
        info["read_error"] = str(exc)
    return info


def copy_tree(root: Path, snap: Path) -> list[Path]:
    copied = []
    for rel in COPY_TARGETS:
        src = root / rel
        if not src.exists():
            print(f"[-] not found, skipped: {rel}")
            continue
        for f in sorted(p for p in src.rglob("*") if p.is_file()):
            sub = "superseded" if f.name in SUPERSEDED_REPORTS else ""
            dst = snap / sub / f.relative_to(root)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst)
            copied.append(dst)
    return copied


def channel_audit(processed: Path) -> dict:
    """Record how many channels each subject's time windows actually have."""
    audit = {}
    for sub_dir in sorted(p for p in processed.glob("S*") if p.is_dir()):
        f = sub_dir / "X_time_windows.npy"
        y = sub_dir / "y_labels.npy"
        if f.exists():
            arr = np.load(f, mmap_mode="r")
            entry = {"n_windows": int(arr.shape[0]), "n_channels": int(arr.shape[1]),
                     "n_samples": int(arr.shape[2])}
            if y.exists():
                labels, counts = np.unique(np.load(y), return_counts=True)
                entry["label_counts"] = {str(int(k)): int(v) for k, v in zip(labels, counts)}
            audit[sub_dir.name] = entry
    return audit


def build_summary(reports: Path) -> pd.DataFrame:
    """One long table: pipeline, subject, metric, value."""
    rows = []

    def add(pipeline, subject, metric, value):
        rows.append({"pipeline": pipeline, "subject": subject, "metric": metric,
                     "value": round(float(value), 2)})

    f = reports / "n_subjects_benchmark.csv"
    if f.exists():
        d = pd.read_csv(f)
        for _, r in d.iterrows():
            add("FBCSP+FBCCA hybrid", r["Subject"], "acc_1s", r["1.0s Acc (%)"])
            add("FBCSP+FBCCA hybrid", r["Subject"], "acc_2s", r["2.0s Smoothed (%)"])
            add("FBCSP+FBCCA hybrid", r["Subject"], "channels_kept", r["Channels Kept"])

    for model in ("eegnet", "compact_cnn", "shallow_conv_net"):
        f = reports / f"augmentation_ab_{model}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        for _, r in d.iterrows():
            for aug in ("OFF", "ON"):
                name = f"{model} aug-{aug}"
                add(name, r["Subject"], "acc_1s", r[f"1.0s {aug} (%)"])
                add(name, r["Subject"], "acc_2s", r[f"2.0s {aug} (%)"])
                add(name, r["Subject"], "train_acc", r[f"Train Acc {aug} (%)"])

    f = reports / "gnn_benchmark.csv"
    if f.exists():
        d = pd.read_csv(f).groupby(["variant", "subject"])[["acc_1s", "acc_2s"]].mean()
        for (variant, sub), r in d.iterrows():
            add(f"GNN {variant}", sub, "acc_1s", r["acc_1s"])
            add(f"GNN {variant}", sub, "acc_2s", r["acc_2s"])

    f = reports / "connectivity_gate_benchmark.csv"
    if f.exists():
        d = pd.read_csv(f).groupby(["set", "subject"])[["acc_1s", "acc_2s"]].mean()
        for (fset, sub), r in d.iterrows():
            add(f"conn-gate {fset}", sub, "acc_1s", r["acc_1s"])
            add(f"conn-gate {fset}", sub, "acc_2s", r["acc_2s"])

    f = reports / "connectivity_cohort_summary.csv"
    if f.exists():
        d = pd.read_csv(f)
        for (method, sub), g in d.groupby(["method", "subject"]):
            add(f"connectivity {method}", sub, "n_freqs_sig_global", g["significant_uncorrected"].sum())
            add(f"connectivity {method}", sub, "n_edges_sig_fdr_total", g["n_edges_sig_fdr"].sum())

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    # Cohort mean row per pipeline/metric for quick reading.
    means = (summary.groupby(["pipeline", "metric"], as_index=False)["value"].mean()
             .assign(subject="MEAN"))
    means["value"] = means["value"].round(2)
    return pd.concat([summary, means], ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                    help="Repo root (default: parent of scripts/)")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--note", default="Per-window sub-band filtering, 1-60 Hz broadband, "
                    "5 classes, BH-FDR connectivity, leak-free StratifiedGroupKFold.")
    args = ap.parse_args()

    root = args.root.resolve()
    snap = root / "baselines" / f"{args.tag}_{datetime.now():%Y%m%d}"
    if snap.exists():
        sys.exit(f"[!] {snap} already exists. A baseline is frozen once; use another --tag.")
    snap.mkdir(parents=True)

    print(f"Freezing baseline into {snap.relative_to(root)}")
    copied = copy_tree(root, snap)

    git_hash = run(["git", "rev-parse", "HEAD"], root)
    git_dirty = run(["git", "status", "--porcelain"], root)
    manifest = {
        "tag": args.tag,
        "created": datetime.now().isoformat(timespec="seconds"),
        "note": args.note,
        "git_commit": git_hash,
        "git_uncommitted_changes": bool(git_dirty) if git_dirty is not None else None,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": (run([sys.executable, "-m", "pip", "freeze"], root) or "").splitlines(),
        "channel_audit": channel_audit(root / "data" / "processed"),
        "superseded_reports": sorted(SUPERSEDED_REPORTS),
        "files": {str(p.relative_to(snap)): describe_file(p) for p in copied},
    }
    (snap / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))

    summary = build_summary(root / "reports")
    summary.to_csv(snap / "baseline_summary.csv", index=False)

    # Make the snapshot read-only so it cannot be overwritten by accident.
    for p in snap.rglob("*"):
        if p.is_file():
            p.chmod(0o444)

    print(f"  files copied : {len(copied)}")
    print(f"  git commit   : {git_hash or 'not a git repo'}")
    chans = {k: v['n_channels'] for k, v in manifest['channel_audit'].items()}
    print(f"  channels     : {chans}")
    if not summary.empty:
        hyb = summary[(summary.pipeline == "FBCSP+FBCCA hybrid") & (summary.subject == "MEAN")]
        print("  hybrid mean  :", dict(zip(hyb.metric, hyb.value)))
    print("Done. Snapshot is read-only; MANIFEST.json + baseline_summary.csv are the v1 reference.")


if __name__ == "__main__":
    main()