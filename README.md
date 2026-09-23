# Steady Signal: An SSVEP Brain-Computer Interface Classification

The purpose of this project is to turn raw occipital EEG into a reliable,
frequency-locked command signal: a full research pipeline for 5-class
Steady-State Visually Evoked Potential (SSVEP) brain-computer interface
decoding. It covers raw `.ebr` acquisition, artifact quality control,
classical and hybrid feature extraction, classical ML and deep learning
classifiers, GPU-trained architecture benchmarking, data augmentation,
trial-grouped cross-validation, and an emerging functional connectivity
analysis.

> **Status:** the classical ML pipeline and the 5-subject flagship benchmark
> are complete and validated. Three deep learning architectures are
> implemented and benchmarked with an augmentation A/B protocol. Functional
> connectivity currently covers coherence only; phase-based, autoregressive,
> and information-theoretic methods are designed but not yet implemented
> (see [Roadmap](#roadmap)).

## Task

5-class SSVEP classification. Target stimulation frequencies and their
condition codes:

| Condition | Meaning | Frequency |
|---|---|---|
| 201 | Fixation cross ("sin estímulo" baseline) | Broadband 5–35 Hz |
| 202 | Cue (lookback window) | Used in preprocessing, not a target class |
| 101 | SSVEP target 1 | 24.00 Hz |
| 102 | SSVEP target 2 | 20.00 Hz |
| 103 | SSVEP target 3 | 15.00 Hz |
| 104 | SSVEP target 4 | 10.91 Hz |
| 105 | SSVEP target 5 | 8.57 Hz |

Recording montage (8 parieto-occipital channels): `PO7, PO3, POz, PO4, PO8,
O1, Oz, O2`. Sampling rate: 250 Hz. Trials are 5.0 s of stimulation, sliced
into five consecutive 1.0 s sub-windows.

## Pipeline overview

```
.ebr acquisition
      │
      ▼
IIR bandpass filter (Butterworth, order 4, 1-60 Hz, zero-phase)
      │
      ▼
Two-tier artifact quality gate
  Tier 1 — channel level: drop chronically bad electrodes (per subject)
  Tier 2 — window level: reject remaining noisy windows (Vpp / std)
      │
      ▼
Feature extraction (FBCSP / FBCCA / PSD)   ──or──   raw windows
      │                                              │
      ▼                                              ▼
Classical ML suite (LDA, QDA, Logistic          Deep learning (GPU)
Regression, Ridge, SGD)                         (EEGNet, Compact-CNN,
      │                                          ShallowConvNet)
      └───────────────────┬──────────────────────────┘
                           ▼
    StratifiedGroupKFold CV (grouped by trial id)
                           │
                           ▼
    Trial-aware causal smoothing (2-tap, resets every 5 sub-windows)
```

## Repository structure

```
src/
  io/
    ebr_parser.py             # Binary .ebr file loader/parser
  preprocessing/
    filters.py                 # Zero-phase Butterworth IIR bandpass
    quality_check.py           # Two-tier artifact-quality pipeline
    cv_utils.py                 # build_trial_ids() for trial-grouped CV
    augmentation.py             # Training-only on-the-fly augmentation
  features/
    fbcsp_extraction.py         # Filter-Bank CSP (one-vs-rest, 5 sub-bands)
    fbcca_extraction.py         # Filter-Bank CCA (harmonic reference signals)
    psd_extraction.py           # Continuous PSD + condition/baseline windowing
    feature_selection.py        # ANOVA / mutual-information feature ranking
  models/
    sklearn_models.py           # Classical ML model suite (sklearn)
    eegnet.py                   # EEGNet (Lawhern et al., 2018), PyTorch
    compact_cnn.py               # Compact-CNN (Waytowich et al., 2018), PyTorch
    shallow_conv_net.py          # ShallowConvNet (Schirrmeister et al., 2017), PyTorch
  visualization/
    signal_plots.py              # PSD-by-condition plotting

scripts/
  run_preprocessing.py           # Raw .ebr -> filtered, QC'd, PSD feature matrix
  diagnose_occipital_spectra.py  # Per-class occipital FFT spectrum diagnostic
  visualize_data.py              # Cross-check PSD / FBCCA / FBCSP extraction
  evaluate_confusion_matrix.py   # 5-fold CV confusion matrix (FBCSP+FBCCA hybrid)
  test_sklearn_models.py         # In-sample fit test across the classical suite

benchmarks/
  benchmark_cross_validation.py    # Single/hybrid feature-space CV comparison
  benchmark_feature_selection.py   # ANOVA / mutual-info / subband-pruning comparison
  benchmark_n_subjects.py          # Flagship FBCSP+FBCCA -> Shrinkage-LDA, all subjects
  benchmark_harmonic_fbcca.py      # Occipital harmonic PSD + FBCCA hybrid
  benchmark_eegnet.py              # EEGNet raw-window benchmark (+ --augment flag)
  benchmark_compact_cnn.py         # Compact-CNN raw-window benchmark
  benchmark_shallow_convnet.py     # ShallowConvNet raw-window benchmark
  benchmark_nn_augmentation.py     # Augmentation ON/OFF, all NN architectures, all subjects

connectivity/
  connectivity_extraction.py     # Coherence per-window + per-condition averaging
  connectivity_plots.py           # MNE-style circular connectivity diagrams
  analyze_connectivity.py         # Per-subject connectivity driver script
```

## Pipeline components

### Acquisition & preprocessing

- **`ebr_parser.py`** — parses the binary EBR format (header + trial/channel/
  band/mark metadata) into a structured dict, including the mark channel
  used to locate stimulus/cue/baseline events.
- **`filters.py`** — zero-phase Butterworth bandpass (`scipy.signal.filtfilt`,
  default 1–60 Hz, order 4).

### Artifact quality control (`quality_check.py`)

Two-tier pipeline, applied per subject:

1. **Channel level (Tier 1):** every channel enters unfiltered.
   `identify_noisy_channels` flags an electrode as chronically bad if it
   violates the Vpp/std thresholds in more than `channel_fail_fraction_thresh`
   (default 0.5) of all windows; `drop_noisy_channels` removes it from the
   subject's entire recording. Channel count is therefore not fixed across
   subjects — every downstream model, CNNs included, is built to tolerate a
   variable channel count.
2. **Window level (Tier 2):** on the surviving channels, `validate_eeg_windows`
   rejects individual windows still outside the Vpp/std bounds (default
   Vpp ∈ [5, 120] µV, std ∈ [1, 35] µV).

`apply_artifact_quality_pipeline` runs both tiers and returns the cleaned
windows/labels plus a per-channel report (kept/dropped channels, fail
fractions). All feature extractors (FBCSP, FBCCA, PSD) and all NN benchmark
scripts share this exact gate.

### Trial-grouped cross-validation (`cv_utils.py`)

Each 5.0 s trial is stored as 5 consecutive 1.0 s sub-windows. A plain
`StratifiedKFold` over sub-windows lets sub-windows from the *same* trial
land in both train and test folds, letting a high-capacity model
"recognize the trial" instead of the SSVEP frequency. `build_trial_ids`
reconstructs trial identity from the label array's contiguous same-label
blocks, and every benchmark script now uses `StratifiedGroupKFold` with
`groups=trial_ids`.

### Feature extraction

- **FBCSP (`fbcsp_extraction.py`)** — one-vs-rest multiclass CSP, 5 default
  sub-bands (`7–12, 13–17, 18–26, 27–36, 38–52 Hz`), Butterworth bandpass
  per sub-band, `m=1` component pair per class → **p = 50** features.
- **FBCCA (`fbcca_extraction.py`)** — canonical correlation against harmonic
  sinusoidal reference signals (3 harmonics by default) for each of the 5
  target frequencies, filtered through 3 Chebyshev-I sub-bands with
  power-law weighting → **p = 5** features.
- **PSD (`psd_extraction.py`)** — continuous periodogram-based PSD in a
  configurable frequency band (default 5–35 Hz), plus condition/baseline
  windowing utilities used by `run_preprocessing.py`.
- **Feature selection (`feature_selection.py`)** — ANOVA F-score or mutual
  information ranking over FBCSP features, computed strictly in-fold.

### Classifiers

**Classical ML (`sklearn_models.py`):** Ridge, Logistic Regression (L1/L2/
Elastic-Net), SGD (log-loss and passive-aggressive), standard LDA,
shrinkage-LDA (LSQR/eigen, Ledoit-Wolf auto shrinkage), and regularized QDA.
The flagship pipeline is **FBCSP + FBCCA (hybrid, p=55) → in-fold
StandardScaler → Shrinkage-LDA**, with 2-tap trial-aware causal smoothing.

**Deep learning (PyTorch, raw windows, GPU-capable):**

| Model | Source | Key idea |
|---|---|---|
| `EEGNetSSVEP` | Lawhern et al. (2018) | Temporal conv → depthwise spatial conv → separable conv; max-norm constrained spatial filters |
| `CompactCNN` | Waytowich et al. (2018) | Spatial filter + one temporal conv sized to span ≥1 cycle of the lowest target frequency (8.57 Hz) — SSVEP-specific narrowband design |
| `ShallowConvNet` | Schirrmeister et al. (2017) | Temporal + spatial conv → square → mean-pool → log nonlinearity, approximating log-band-power (general-purpose EEG baseline) |

All three collapse the channel axis via a spatial conv, so they tolerate
however many channels survive Tier-1 dropping for a given subject.

**Training convention**, shared by all three benchmark scripts and selecting
CUDA automatically when available (`--device cuda` if `torch.cuda.is_available()`,
else CPU): per-window per-channel temporal standardization computed in-fold,
`AdamW` (`weight_decay=1e-2`), `CosineAnnealingLR` schedule,
`CrossEntropyLoss(label_smoothing=0.05)`, 150 epochs/fold, batch size 32,
lr 1e-3, `StratifiedGroupKFold(n_splits=5, random_state=42)`.

### Data augmentation (`augmentation.py`)

Training-fold-only, applied on-the-fly (`SSVEPAugmentedDataset`), never
touching validation/test data:

- Gaussian noise (scaled to per-channel std)
- Amplitude scaling (±10% by default)
- Time shift (circular, jitter simulation)
- Intra-class mixup (Beta(α,α), reflected toward 0.5)

Enabled per-run via `--augment` on `benchmark_eegnet.py`, and evaluated
cohort-wide across all three architectures by `benchmark_nn_augmentation.py`.

### Functional connectivity — implemented and planned

**Implemented:**

- `connectivity_extraction.py` — magnitude-squared coherence, per-window
  matrix plus per-condition averaging; baseline = condition `201`
  (fixation cross, broadband 5–35 Hz) vs. narrow ±1 Hz bands around each
  stimulus fundamental; reuses `apply_artifact_quality_pipeline`.
- `connectivity_plots.py` — MNE-style circular diagrams via
  `mne_connectivity.viz.plot_connectivity_circle`, shared color scale
  across a 6-condition panel (baseline + 5 stimulus frequencies).
- `analyze_connectivity.py` — per-subject driver script (`--subject`,
  `--method`).

Verified on synthetic 8-channel data; not yet run against real subject
data.

**Designed, not yet implemented:** phase-based connectivity (PLV/PLI via
Hilbert-transform narrowband filtering), autoregressive connectivity
(Granger causality / PDC), and mutual-information connectivity.

## Usage

```bash
# Preprocess one subject's raw .ebr recording into a PSD feature matrix
python scripts/run_preprocessing.py

# Classical ML: fit and inspect accuracy across the full sklearn model suite
python scripts/test_sklearn_models.py --subject S03

# Flagship hybrid benchmark (all discovered subjects, trial-grouped CV)
python benchmarks/benchmark_n_subjects.py --n-jobs -1

# Single-subject EEGNet benchmark on GPU if available, with optional augmentation A/B
python benchmarks/benchmark_eegnet.py --subject S03 --epochs 150 --augment

# Augmentation ON vs OFF across all subjects, for a given architecture
python benchmarks/benchmark_nn_augmentation.py --model compact_cnn

# Functional connectivity (coherence) for one subject
python connectivity/analyze_connectivity.py --subject S03 --method coherence
```

Subject folders are auto-discovered under `data/processed/` (pattern
`S<digits>`); no script edits are needed to add subjects.

## Benchmark results

**Flagship pipeline: FBCSP (m=1, p=50) + FBCCA (p=5) → Shrinkage-LDA**,
5-fold `StratifiedGroupKFold` (trial-grouped), 1.0 s instantaneous /
2.0 s trial-aware smoothed accuracy. These are the corrected, validated
numbers — see **Trial-grouping fix** below for why an earlier version of
this table read differently.

| Subject | 1.0 s Acc | 2.0 s Smoothed |
|---|---|---|
| S01 | 61.00% | 67.50% |
| S02 | 56.12% | 61.73% |
| S03 | 86.80% | 90.36% |
| S04 | 97.46% | 97.97% |
| S05 | 95.00% | 95.50% |
| **Cohort mean** | **79.28%** | **82.61%** |

S02 is the hardest subject under the corrected evaluation.

*Per-subject clean/rejected window counts are tracked by `quality_check.py`'s
QC report and printed by each benchmark run, but are not reproduced here —
pull them from the actual console output or `reports/n_subjects_benchmark.csv`
of a given run rather than treating any fixed count as canonical.*

### Trial-grouping fix

All benchmark scripts originally split individual 1 s sub-windows with
plain `StratifiedKFold`, without grouping the 5 consecutive sub-windows
belonging to the same 5 s trial — sub-windows from one trial could land in
both train and test. This was confirmed on real S03 data (contiguous
blocks of ~40 windows = 8 trials × 5 sub-windows) and fixed project-wide
via `build_trial_ids()` + `StratifiedGroupKFold`.

Effect of the fix:

- Classical Shrinkage-LDA hybrid: cohort mean dropped ~4–5 pp
  (83.33%/87.57% → **79.28%/82.61%**) — relatively robust to the leak.
- Deep CNNs: dropped far more on some subjects (e.g. Compact-CNN up to
  −20 pp on individual subjects) — much more sensitive to the leak.
- EEGNet's measured augmentation benefit shrank from +7.34 pp to +2.01 pp
  once leakage was removed.

All 9 affected scripts are fixed: `benchmark_n_subjects.py`,
`benchmark_nn_augmentation.py`, `benchmark_cross_validation.py`,
`benchmark_feature_selection.py`, `evaluate_confusion_matrix.py`,
`benchmark_harmonic_fbcca.py`, `benchmark_eegnet.py`,
`benchmark_compact_cnn.py`, `benchmark_shallow_convnet.py`.

### Known open finding

EEGNet showed a large convergence gap on S01/S02 (~27–28%, near
chance-level for 5 classes) versus the classical hybrid's 61–56% on the
same subjects — a training-set-accuracy diagnostic was added to
`benchmark_nn_augmentation.py` to distinguish non-convergence from
overfitting before drawing further conclusions. Not yet resolved.

## Interpretation notes

- Per-subject accuracy varies substantially; the cohort mean should not be
  read in isolation from the per-subject table.
- Training accuracy in the augmentation reports is a diagnostic for
  convergence, not a substitute for cross-validated test accuracy.
- Connectivity values are descriptive, condition-level summaries and
  should be read alongside signal quality and the number of retained
  windows, not as standalone neurophysiological claims.

## Roadmap

1. ~~Classic ML analysis across all subjects~~ — done (flagship results above)
2. ~~Automatic noisy-channel discard criterion~~ — done, two-tier pipeline
3. Additional NN architectures for EEG — EEGNet, Compact-CNN, ShallowConvNet
   implemented; further architectures still open
4. ~~Simple EEG data-augmentation methods~~ — done (noise, scaling, shift, mixup)
5. NN architectures combined with augmentation — in progress
   (`benchmark_nn_augmentation.py`)
6. Functional connectivity analysis — coherence done and pending validation
   on real data; PLV/PLI, Granger/PDC, and mutual information designed but
   not implemented; a pseudo-online exploration planned after


## Requirements

- Python 3.10+
- `numpy`, `scipy`, `scikit-learn`, `pandas`
- `torch` (EEGNet / Compact-CNN / ShallowConvNet; CUDA optional but used
  automatically when available)
- `matplotlib`, `seaborn` (plots, confusion matrices)
- `mne-connectivity` (functional connectivity diagrams)
- `pyyaml` (pipeline configuration)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install mne mne-connectivity
```
