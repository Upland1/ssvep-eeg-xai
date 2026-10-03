# Steady Signal: SSVEP-EEG Classification, Connectivity and XAI

This repository provides a PyTorch / scikit-learn research pipeline for
**5-class Steady-State Visually Evoked Potential (SSVEP)** brain-computer
interface decoding from parieto-occipital EEG. It covers:

- raw binary `.ebr` acquisition parsing and a two-tier artifact quality gate,
- classical spectral/spatial features (FBCSP, FBCCA, harmonic PSD) with a
  Shrinkage-LDA flagship classifier,
- deep learning on raw windows (EEGNet, Compact-CNN, ShallowConvNet) with a
  training-only augmentation A/B protocol,
- functional connectivity (coherence, PLV, wPLI, PDC) with permutation
  statistics, cross-frequency comparison and confound checks,
- graph models on per-window channel graphs (connectivity gate + edge-gated GNN),
- explainability (feature-importance heatmaps across subjects).

> **Evaluation note.** Each 5 s trial is cut into five 1 s sub-windows. Every
> benchmark uses `StratifiedGroupKFold` **grouped by trial**, so sub-windows of
> the same trial never appear in both training and test folds. Earlier numbers
> computed with plain `StratifiedKFold` were inflated (see
> [Trial-grouping fix](#trial-grouping-fix)); all tables below use the
> corrected protocol.

## Task

| Condition | Meaning | Frequency |
|---|---|---|
| 101 | SSVEP target 1 | 24.00 Hz |
| 102 | SSVEP target 2 | 20.00 Hz |
| 103 | SSVEP target 3 | 15.00 Hz |
| 104 | SSVEP target 4 | 10.91 Hz |
| 105 | SSVEP target 5 | 8.57 Hz |
| 201 | Fixation cross (no-stimulus baseline) | used for connectivity baselines |
| 202 | Cue (lookback window) | used in preprocessing only |

Montage: `PO7, PO3, POz, PO4, PO8, O1, Oz, O2` at 250 Hz. Trials are 5.0 s of
stimulation, sliced into five consecutive 1.0 s sub-windows (256 samples).
Accuracies are reported per **1.0 s** window and after **2.0 s** trial-aware
causal smoothing (2-tap average that resets at every new trial).

## Results

All results: 5 subjects (S01–S05), 5-fold trial-grouped CV, chance level = 20%.

### Flagship classical pipeline

**FBCSP (m=1, p=50) + FBCCA (p=5) → in-fold StandardScaler → Shrinkage-LDA**
(`scripts/benchmarks/feature_based/benchmark_n_subjects.py`)

| Subject | Channels kept | Clean windows | 1.0 s accuracy (%) | 2.0 s smoothed (%) |
|---|---|---|---|---|
| S01 | 7 | 200 | 61.00 | 67.50 |
| S02 | 7 | 196 | 56.12 | 61.73 |
| S03 | 7 | 197 | 86.80 | 90.36 |
| S04 | 7 | 197 | 97.46 | 97.97 |
| S05 | 7 | 200 | 95.00 | 95.50 |
| **Mean** | | | **79.28** | **82.61** |

### Deep learning on raw windows (augmentation A/B)

Cohort mean accuracy, augmentation OFF → ON
(`scripts/benchmarks/neural_networks/benchmark_nn_augmentation.py`, 150 epochs/fold):

| Model | #params ¹ | 1.0 s OFF | 1.0 s ON | 2.0 s OFF | 2.0 s ON |
|---|---|---|---|---|---|
| ShallowConvNet | 14,485 | 73.18 | **75.70** | 76.52 | **78.92** |
| EEGNet | 2,349 | 64.22 | 66.24 | 67.56 | 68.77 |
| Compact-CNN | 56,021 | 36.80 | 53.96 | 41.23 | 57.69 |
| *Shrinkage-LDA flagship (reference)* | – | 79.28 | – | 82.61 | – |

<sub>¹ 7 channels × 256 samples, 5 classes.</sub>

Per subject (1.0 s, augmentation ON):

| Model | S01 | S02 | S03 | S04 | S05 |
|---|---|---|---|---|---|
| ShallowConvNet | 56.50 | 35.20 | 89.34 | 98.98 | 98.50 |
| EEGNet | 34.50 | 26.02 | 79.19 | 99.49 | 92.00 |
| Compact-CNN | 27.50 | 19.39 | 62.94 | 97.97 | 62.00 |

Augmentation helps every architecture on average and helps Compact-CNN most
(+17 pp). No network beats the classical flagship on the cohort mean; the gap
comes almost entirely from the two weak responders (S01, S02).

### Connectivity gate: do connectivity features add to power features?

Shrinkage-LDA on per-window graph features, mean over 5 CV repeats, 1.0 s
accuracy (`scripts/connectivity/graph_modeling/benchmark_connectivity_gate.py`).
`N` = node power at the 5 targets and their 2nd harmonics, `F` = FBCCA scores,
`Ec` = edge coherence, `Ep` = edge phase, `Base` = `N+F`.

| Feature set | Dim | S01 | S02 | S03 | S04 | S05 | Mean |
|---|---|---|---|---|---|---|---|
| Ec | 105 | 48.3 | 36.1 | 45.5 | 62.4 | 50.6 | 48.6 |
| Ec+Ep | 315 | 47.9 | 35.2 | 48.3 | 70.7 | 51.1 | 50.6 |
| N | 70 | 31.4 | 24.6 | 79.7 | 96.3 | 94.5 | 65.3 |
| F | 5 | 40.6 | 55.3 | 68.3 | 94.9 | 83.6 | 68.5 |
| Base (N+F) | 75 | 41.2 | 48.0 | 82.7 | 98.0 | 98.9 | 73.8 |
| Base+Ep | 285 | 44.9 | 44.3 | 83.5 | 98.3 | 96.3 | 73.5 |
| Base+Ec+Ep | 390 | 51.1 | 46.5 | 82.2 | 98.9 | 95.8 | 74.9 |
| Base+Ec | 180 | 57.5 | 51.1 | 84.0 | 98.1 | 97.2 | **77.6** |

Connectivity alone is far weaker than power. Adding coherence to the power
baseline helps the weak responders (S01 +16.3, S02 +3.1 pp) but not the
responders (S03 +1.3, S04 +0.1, S05 −1.7 pp), so the gate heuristic
(≥ 2 pp on average and in ≥ 2/3 of responders) does **not** pass.

### Graph neural network

Edge-gated GNN, mean over 3 CV repeats
(`scripts/connectivity/graph_modeling/benchmark_gnn.py`):

| Variant | #params | S02 1.0 s / 2.0 s | S03 1.0 s / 2.0 s | S05 1.0 s / 2.0 s |
|---|---|---|---|---|
| nodes+F | 5,150 | 38.6 / 46.1 | 77.8 / 82.2 | 90.0 / 95.0 |
| nodes+coh+F | 5,534 | 37.2 / 40.1 | 77.5 / 82.9 | 90.5 / 95.3 |

Coherence edges do not improve the GNN, consistent with the gate result.

### Functional connectivity statistics

Stimulus vs. matched no-stimulus baseline, trial-level permutation test
(200 permutations), FDR-corrected significant edges summed over the 5
frequencies (21 edges per frequency, 7 channels):

| Method | S01 | S02 | S03 | S04 | S05 |
|---|---|---|---|---|---|
| Coherence | 0 | 0 | 65 | 51 | 15 |
| wPLI | 0 | 3 | 38 | 33 | 18 |

Significant connectivity changes appear only in the subjects who also classify
well (S03–S05), which is why the confound checks below are needed before
reading them as network effects.

## Models

`src/models/` contains the following models:

| Model | File | Reference |
|---|---|---|
| Classical suite: Ridge, Logistic Regression (L1/L2/Elastic-Net), SGD, LDA, Shrinkage-LDA (Ledoit-Wolf), regularized QDA | `classical/sklearn_models.py` | scikit-learn |
| EEGNet | `cnn/eegnet.py` | Lawhern et al., 2018 |
| Compact-CNN (SSVEP) | `cnn/compact_cnn.py` | Waytowich et al., 2018 |
| ShallowConvNet | `cnn/shallow_conv_net.py` | Schirrmeister et al., 2017 |
| Dual-branch fusion CNN (occipital harmonics + FBCCA priors) | `cnn/spatial_spectral_cnn.py` | this repository |
| Edge-gated connectivity GNN | `graph/connectivity_gnn.py` | this repository |

All CNNs collapse the channel axis with a spatial convolution, so they accept
whatever number of channels survives the quality gate for a given subject.

**Deep learning training convention** (shared by all NN benchmarks): in-fold
per-window per-channel standardization, `AdamW` (weight decay 1e-2),
`CosineAnnealingLR`, `CrossEntropyLoss(label_smoothing=0.05)`, 150 epochs,
batch size 32, learning rate 1e-3, CUDA used automatically when available.

## Pipeline

```
.ebr acquisition
      │
      ▼
IIR bandpass (Butterworth, order 4, 1–60 Hz, zero-phase)
      │
      ▼
Two-tier artifact quality gate
  Tier 1 — channel level: drop chronically bad electrodes (per subject)
  Tier 2 — window level: reject remaining noisy windows (Vpp / std)
      │
      ├──────────────────────┬─────────────────────────┬──────────────────────┐
      ▼                      ▼                         ▼                      ▼
FBCSP / FBCCA / PSD     Raw 1 s windows          Connectivity            Per-window graphs
      │                      │                   (coh, PLV, wPLI, PDC)   (node power + edges)
      ▼                      ▼                         │                      │
Classical ML suite     EEGNet / Compact-CNN /          ▼                      ▼
(Shrinkage-LDA, ...)   ShallowConvNet (+ aug)    Permutation tests,     Connectivity gate,
      │                      │                   cross-frequency,       edge-gated GNN
      └──────────┬───────────┘                   confound checks
                 ▼
StratifiedGroupKFold (grouped by trial) → 2-tap trial-aware smoothing
```

### Preprocessing and quality control

- **`src/io/ebr_parser.py`** parses the binary EBR format (header, trial,
  channel, band and mark metadata), including the mark channel used to locate
  stimulus, cue and baseline events.
- **`src/preprocessing/signal/filters.py`**: zero-phase Butterworth bandpass.
- **`src/preprocessing/signal/quality_check.py`**: two-tier gate.
  1. *Channel level:* an electrode is dropped for the whole subject if it
     violates the Vpp/std bounds in more than 50% of windows
     (`channel_fail_fraction_thresh`).
  2. *Window level:* on the surviving channels, single windows outside the
     bounds are rejected.

  `apply_artifact_quality_pipeline` runs both tiers and returns clean windows,
  labels and a per-channel report. All feature extractors and NN benchmarks
  share this exact gate.

### Trial-grouped cross-validation

`src/preprocessing/dataset/cv_utils.py::build_trial_ids` reconstructs trial
identity from contiguous same-label blocks; every benchmark passes it as
`groups=` to `StratifiedGroupKFold`.

### Features

- **FBCSP** (`src/features/ssvep/fbcsp_extraction.py`): one-vs-rest multiclass
  CSP over 5 sub-bands (7–12, 13–17, 18–26, 27–36, 38–52 Hz), m=1 pair per class
  → 50 features.
- **FBCCA** (`src/features/ssvep/fbcca_extraction.py`): canonical correlation
  with harmonic sinusoidal references (3 harmonics) for each target, over
  Chebyshev-I sub-bands with power-law weighting → 5 features.
- **PSD** (`src/features/ssvep/psd_extraction.py`): continuous periodogram PSD
  (5–35 Hz) plus condition/baseline windowing.
- **Feature selection** (`src/features/selection/feature_selection.py`): ANOVA
  F-score or mutual-information ranking, fitted strictly in-fold.
- **Connectivity** (`src/features/connectivity/`): condition-level coherence
  (Welch on concatenated windows), PLV and wPLI from the cross-spectral density
  in a ±1 Hz band around each target, PDC; per-window graph features (node
  power at targets and 2nd harmonics, edge coherence and edge phase).

### Data augmentation

`src/preprocessing/dataset/augmentation.py::SSVEPAugmentedDataset` applies, on
the training fold only and on the fly: Gaussian noise scaled to channel std,
amplitude scaling (±10%), circular time shift and intra-class mixup.

### Connectivity analysis workflow

1. `scripts/connectivity/analysis/`: per-subject and cohort connectivity,
   stimulus vs. matched baseline, trial-level permutation tests with FDR.
2. `scripts/connectivity/comparison/`: descriptive cross-frequency comparison
   of baseline-corrected delta matrices (QAP similarity, node strength), and
   formal between-frequency tests with a split-half reliability ceiling.
3. `scripts/connectivity/diagnostics/`: checks whether connectivity deltas are
   explained by SSVEP power (target SNR) or alpha blocking.
4. `scripts/connectivity/graph_modeling/`: builds cached per-window graph
   datasets, runs the connectivity gate, then the GNN benchmark.

Outputs are written to `reports/` (CSVs) and `outputs/figures/`.

## Repository structure

```
src/
  io/                     ebr_parser.py
  preprocessing/
    signal/               filters.py, quality_check.py
    dataset/              cv_utils.py, augmentation.py
  features/
    ssvep/                fbcsp_extraction.py, fbcca_extraction.py, psd_extraction.py
    selection/            feature_selection.py
    connectivity/         connectivity_extraction.py, connectivity_graph_features.py, graph_inputs.py
  models/
    classical/            sklearn_models.py
    cnn/                  eegnet.py, compact_cnn.py, shallow_conv_net.py, spatial_spectral_cnn.py
    graph/                connectivity_gnn.py
  visualization/          signal_plots.py, connectivity_plots.py

scripts/
  preprocessing/          run_preprocessing.py
  exploration/            visualize_data.py, diagnose_occipital_spectra.py
  training/               train_models.py, train_final_model.py, batch_run_subjects.py
  evaluation/             evaluate_confusion_matrix.py, test_sklearn_models.py
  xai/                    run_xai.py, compare_subject_xai.py
  benchmarks/
    feature_based/        benchmark_n_subjects.py, benchmark_cross_validation.py,
                          benchmark_feature_selection.py, benchmark_harmonic_fbcca.py
    neural_networks/      benchmark_eegnet.py, benchmark_compact_cnn.py,
                          benchmark_shallow_convnet.py, benchmark_nn_augmentation.py
  connectivity/
    analysis/             analyze_connectivity.py, analyze_connectivity_cohort.py
    comparison/           compare_connectivity_frequencies.py, test_connectivity_frequency_differences.py
    diagnostics/          check_power_alpha_confound.py
    graph_modeling/       build_graph_dataset.py, benchmark_connectivity_gate.py, benchmark_gnn.py

configs/pipeline_config.yaml   # channels, filter, QC thresholds, windowing
```

## Usage

Run every script **as a module from the project root**, so the `src` and
`scripts` packages resolve:

```bash
# 1. Preprocess raw .ebr into windows and PSD features
python -m scripts.preprocessing.run_preprocessing

# 2. Classical flagship benchmark (all subjects found under data/processed/)
python -m scripts.benchmarks.feature_based.benchmark_n_subjects --n-jobs -1

# 3. Deep learning: one subject, or augmentation ON/OFF across all subjects
python -m scripts.benchmarks.neural_networks.benchmark_eegnet --subject S03 --augment
python -m scripts.benchmarks.neural_networks.benchmark_nn_augmentation --model shallow_conv_net

# 4. Connectivity: one subject, then the cohort, then comparisons
python -m scripts.connectivity.analysis.analyze_connectivity --subject S03 --method wpli
python -m scripts.connectivity.analysis.analyze_connectivity_cohort --methods coherence wpli
python -m scripts.connectivity.comparison.compare_connectivity_frequencies
python -m scripts.connectivity.diagnostics.check_power_alpha_confound

# 5. Graph models
python -m scripts.connectivity.graph_modeling.build_graph_dataset
python -m scripts.connectivity.graph_modeling.benchmark_connectivity_gate
python -m scripts.connectivity.graph_modeling.benchmark_gnn --smoke-test   # quick check
python -m scripts.connectivity.graph_modeling.benchmark_gnn

# 6. Explainability
python -m scripts.xai.run_xai
```

Subject folders are discovered automatically under `data/processed/`
(pattern `S<digits>`); adding a subject needs no code changes. Most scripts
accept `--help`.

## Trial-grouping fix

All benchmarks originally split 1 s sub-windows with plain `StratifiedKFold`,
so sub-windows from one 5 s trial could land in both train and test folds. This
was confirmed on real data and fixed project-wide with `build_trial_ids()` +
`StratifiedGroupKFold`. Effect:

- Shrinkage-LDA hybrid: cohort mean dropped from 83.33% / 87.57% to
  **79.28% / 82.61%**, so it was fairly robust to the leak.
- Deep CNNs dropped much more (Compact-CNN up to −20 pp on single subjects).
- EEGNet's apparent augmentation gain shrank from +7.34 pp to about +2 pp.

## Interpretation notes and open questions

- Accuracy varies strongly between subjects (S01/S02 vs. S03–S05); read the
  cohort mean together with the per-subject tables.
- On S01/S02, EEGNet reaches 78–86% training accuracy but stays near chance on
  test folds, which points to poor generalization rather than failure to fit.
- Connectivity results are condition-level, descriptive summaries. A common
  SSVEP-driven component can raise coherence and wPLI without true network
  coupling; `check_power_alpha_confound.py` tests this, and the effects should
  be read alongside signal quality and retained window counts.
- Not yet implemented: Granger causality and mutual-information connectivity,
  and a pseudo-online evaluation.

## Development environment

Results were produced on a single **NVIDIA GeForce RTX 3050 Laptop GPU (4 GB,
driver 580, CUDA 13.0)** under **Linux Mint 22.3**, using **Python 3.12** in a
virtual environment with:

- PyTorch 2.13
- NumPy 2.5, SciPy 1.18, pandas 3.0
- scikit-learn 1.9
- matplotlib 3.11, seaborn 0.13
- MNE 1.13, mne-connectivity 0.9
- PyYAML 6.0

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install mne mne-connectivity
```

`requirements.txt` lists minimum versions; `environment.yml` is provided for conda.

## Dataset

The EEG recordings (5 subjects, 8 parieto-occipital channels, binary `.ebr`
files per paradigm, e.g. `OO.ebr`, `C.ebr`) are **not distributed** with this
repository. Place them as:

```
data/raw/S01/OO.ebr
data/raw/S02/OO.ebr
...
```

Paths, channels and preprocessing thresholds are set in
`configs/pipeline_config.yaml`. Processed arrays are written to
`data/processed/<subject>/` and are git-ignored, as are `outputs/` and `reports/`.

## References

If you use the models or methods in this repository, please cite the original works:

```bibtex
@article{lawhern2018eegnet,
  title   = {EEGNet: a compact convolutional neural network for EEG-based brain--computer interfaces},
  author  = {Lawhern, Vernon J and Solon, Amelia J and Waytowich, Nicholas R and Gordon, Stephen M and Hung, Chou P and Lance, Brent J},
  journal = {Journal of Neural Engineering},
  volume  = {15},
  number  = {5},
  pages   = {056013},
  year    = {2018},
  doi     = {10.1088/1741-2552/aace8c}
}

@article{waytowich2018compact,
  title   = {Compact convolutional neural networks for classification of asynchronous steady-state visual evoked potentials},
  author  = {Waytowich, Nicholas and Lawhern, Vernon J and Garcia, Javier O and Cummings, Jennifer and Faller, Josef and Sajda, Paul and Vettel, Jean M},
  journal = {Journal of Neural Engineering},
  volume  = {15},
  number  = {6},
  pages   = {066031},
  year    = {2018},
  doi     = {10.1088/1741-2552/aae5d8}
}

@article{schirrmeister2017deep,
  title   = {Deep learning with convolutional neural networks for EEG decoding and visualization},
  author  = {Schirrmeister, Robin Tibor and Springenberg, Jost Tobias and Fiederer, Lukas Dominique Josef and Glasstetter, Martin and Eggensperger, Katharina and Tangermann, Michael and Hutter, Frank and Burgard, Wolfram and Ball, Tonio},
  journal = {Human Brain Mapping},
  volume  = {38},
  number  = {11},
  pages   = {5391--5420},
  year    = {2017},
  doi     = {10.1002/hbm.23730}
}

@article{chen2015fbcca,
  title   = {Filter bank canonical correlation analysis for implementing a high-speed SSVEP-based brain--computer interface},
  author  = {Chen, Xiaogang and Wang, Yijun and Gao, Shangkai and Jung, Tzyy-Ping and Gao, Xiaorong},
  journal = {Journal of Neural Engineering},
  volume  = {12},
  number  = {4},
  pages   = {046008},
  year    = {2015},
  doi     = {10.1088/1741-2560/12/4/046008}
}

@inproceedings{ang2008fbcsp,
  title     = {Filter Bank Common Spatial Pattern (FBCSP) in Brain-Computer Interface},
  author    = {Ang, Kai Keng and Chin, Zheng Yang and Zhang, Haihong and Guan, Cuntai},
  booktitle = {2008 IEEE International Joint Conference on Neural Networks},
  pages     = {2390--2397},
  year      = {2008},
  doi       = {10.1109/IJCNN.2008.4634130}
}

@article{vinck2011wpli,
  title   = {An improved index of phase-synchronization for electrophysiological data in the presence of volume-conduction, noise and sample-size bias},
  author  = {Vinck, Martin and Oostenveld, Robert and van Wingerden, Marijn and Battaglia, Francesco and Pennartz, Cyriel MA},
  journal = {NeuroImage},
  volume  = {55},
  number  = {4},
  pages   = {1548--1565},
  year    = {2011},
  doi     = {10.1016/j.neuroimage.2011.01.055}
}
```
