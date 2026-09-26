"""
Functional connectivity feature extraction for SSVEP EEG windows.
"""

import numpy as np
from joblib import Parallel, delayed
from scipy.signal import coherence, csd, welch
from src.preprocessing.quality_check import apply_artifact_quality_pipeline

# SSVEP fundamental frequency per stimulus condition code (matches
# fbcca_extraction.TARGET_FREQS, keyed by condition instead of by index).
CONDITION_FREQS = {
    101: 24.0,
    102: 20.0,
    103: 15.0,
    104: 10.9091,
    105: 8.5714,
}
BASELINE_CONDITION = 201
# Broadband range for the no-stimulus condition (matches psd_extraction's
# default freq_range) -- there's no single target frequency to center on.
BASELINE_BAND = (5.0, 35.0)
# Half-width, in Hz, of the narrow band used around each stimulus
# condition's fundamental frequency.
STIMULUS_BAND_HALFWIDTH_HZ = 1.0


# ---------------------------------------------------------------------------
# Per-window connectivity matrix (diagnostic use only -- see warning below)
# ---------------------------------------------------------------------------

def compute_coherence_matrix(
    window: np.ndarray,
    fs: float = 250.0,
    freq_band: tuple[float, float] | None = None,
    nperseg: int | None = None,
) -> np.ndarray:
  """
  Magnitude-squared coherence between every channel pair, band-averaged,
  computed on a SINGLE window.
  """
  if window.ndim != 2:
    raise ValueError("window must have shape (n_channels, n_samples)")

  n_channels, n_samples = window.shape
  nperseg = nperseg or min(n_samples, 128)
  conn = np.eye(n_channels)

  for i in range(n_channels):
    for j in range(i + 1, n_channels):
      f, cxy = coherence(window[i], window[j], fs=fs, nperseg=nperseg)
      if freq_band is not None:
        mask = (f >= freq_band[0]) & (f <= freq_band[1])
        val = float(np.mean(cxy[mask])) if mask.any() else 0.0
      else:
        val = float(np.mean(cxy))
      conn[i, j] = conn[j, i] = val

  return conn


# ---------------------------------------------------------------------------
# Condition-level coherence (the estimator actually used by the pipeline)
# ---------------------------------------------------------------------------

def compute_condition_coherence(
    condition_windows: np.ndarray,
    fs: float = 250.0,
    freq_band: tuple[float, float] | None = None,
    nperseg: int = 256,
) -> np.ndarray:
  """
  Condition-level coherence: concatenate ALL of a condition's clean
  windows into one long per-channel signal, then run Welch's method ONCE
  per channel pair -- instead of computing coherence per window and
  averaging the resulting matrices.
  """
  if condition_windows.ndim != 3:
    raise ValueError("condition_windows must have shape (n_windows, n_channels, n_samples)")

  n_windows, n_channels, n_samples = condition_windows.shape
  concat = condition_windows.transpose(1, 0, 2).reshape(n_channels, n_windows * n_samples)
  nperseg_eff = min(nperseg, concat.shape[-1])
  conn = np.eye(n_channels)

  for i in range(n_channels):
    for j in range(i + 1, n_channels):
      f, cxy = coherence(concat[i], concat[j], fs=fs, nperseg=nperseg_eff)
      if freq_band is not None:
        mask = (f >= freq_band[0]) & (f <= freq_band[1])
        val = float(np.mean(cxy[mask])) if mask.any() else 0.0
      else:
        val = float(np.mean(cxy))
      conn[i, j] = conn[j, i] = val

  return conn


def compute_phase_connectivity(
    windows: np.ndarray,
    fs: float = 250.0,
    freq_band: tuple[float, float] | None = None,
    metric: str = "wpli",
    nperseg: int | None = None,
) -> np.ndarray:
  """Compute condition-level PLV or wPLI from the cross-spectral density
  at the target band -- NOT from time-domain Hilbert-transform phase.

  FIXED (see review notes): the original implementation band-pass
  filtered each window in the time domain (filtfilt) and took the Hilbert
  transform's instantaneous phase. On this project's 256-sample (1.02s)
  windows with a narrow +/-1 Hz stimulus band, that produces SEVERE edge
  distortion: the amplitude envelope near each window's edges is only
  5-70% of its true mid-window value, and instantaneous frequency swings
  by several Hz or goes negative there -- confirmed with a controlled
  sinusoid test, and it does NOT resolve with more padding, a lower
  filter order, or a wider band (the culprit is the Hilbert transform's
  own edge effect on a short, non-periodic segment, not just the filter's
  settling time). Since the original code pooled ALL time samples
  (window edges included) into the PLV/wPLI average, every window was
  injecting corrupted samples into the estimate.

  This version instead gets the phase from the cross-spectral density at
  the band of interest -- the same Welch-segmented, properly-tapered
  machinery `compute_condition_coherence` already uses -- one complex
  value per window (or a few, if the window is long enough for multiple
  Welch segments), never a raw per-sample instantaneous phase. Verified:
  immune to injected edge-sample noise (PLV unchanged, 0.998 -> 0.998,
  when edge samples are corrupted) where the time-domain version would
  have been contaminated by construction.

  Parameters
  ----------
  windows : np.ndarray, shape (n_windows, n_channels, n_samples)
      All clean windows belonging to ONE condition.
  freq_band : (low, high) Hz -- required (unlike coherence, PLV/wPLI need
      a specific band to extract a meaningful phase from).
  metric : {"plv", "wpli"}
  nperseg : Welch segment length; defaults to min(n_samples, 128), same
      default as `compute_coherence_matrix`.

  Returns
  -------
  np.ndarray, shape (n_channels, n_channels)
  """
  if windows.ndim != 3:
    raise ValueError("windows must have shape (n_windows, n_channels, n_samples)")
  if metric not in {"plv", "wpli"}:
    raise ValueError(f"Unknown phase connectivity metric: {metric}")
  if freq_band is None:
    raise ValueError("freq_band is required for PLV/wPLI (need a target frequency to extract phase from)")

  n_windows, n_channels, n_samples = windows.shape
  nperseg = nperseg or min(n_samples, 128)
  low, high = freq_band
  conn = np.eye(n_channels)

  for i in range(n_channels):
    for j in range(i + 1, n_channels):
      # One (band-averaged) complex cross-spectral value per window.
      cross_vals = np.empty(n_windows, dtype=complex)
      for w in range(n_windows):
        f, pxy = csd(windows[w, i], windows[w, j], fs=fs, nperseg=nperseg)
        mask = (f >= low) & (f <= high)
        cross_vals[w] = pxy[mask].mean() if mask.any() else 0.0 + 0.0j

      if metric == "plv":
        value = float(np.abs(np.mean(np.exp(1j * np.angle(cross_vals)))))
      else:
        imag_vals = np.imag(cross_vals)
        denom = np.mean(np.abs(imag_vals))
        value = 0.0 if denom == 0.0 else float(np.abs(np.mean(imag_vals)) / denom)

      conn[i, j] = conn[j, i] = value

  return conn


def compute_condition_plv(
    condition_windows: np.ndarray,
    fs: float = 250.0,
    freq_band: tuple[float, float] | None = None,
) -> np.ndarray:
  """Compute phase-locking value for one condition."""
  return compute_phase_connectivity(condition_windows, fs=fs, freq_band=freq_band, metric="plv")


def compute_condition_wpli(
    condition_windows: np.ndarray,
    fs: float = 250.0,
    freq_band: tuple[float, float] | None = None,
) -> np.ndarray:
  """Compute weighted phase-lag index for one condition."""
  return compute_phase_connectivity(condition_windows, fs=fs, freq_band=freq_band, metric="wpli")


def fit_mvar_ols(windows: np.ndarray, p: int = 10) -> np.ndarray:
  """Fit a ridge-regularized MVAR model without crossing window boundaries."""
  if windows.ndim != 3:
    raise ValueError("windows must have shape (n_windows, n_channels, n_samples)")
  if p < 1 or p >= windows.shape[-1]:
    raise ValueError("p must be at least 1 and smaller than the number of samples")

  _, n_channels, n_samples = windows.shape
  n_observations = windows.shape[0] * (n_samples - p)
  x_matrix = np.empty((n_observations, p * n_channels), dtype=float)
  y_matrix = np.empty((n_observations, n_channels), dtype=float)

  row = 0
  for window in windows:
    for time_index in range(p, n_samples):
      x_matrix[row] = window[:, time_index - p:time_index][:, ::-1].reshape(-1)
      y_matrix[row] = window[:, time_index]
      row += 1

  gram = x_matrix.T @ x_matrix
  regularization = 1e-4 * np.trace(gram) / gram.shape[0]
  weights = np.linalg.solve(
      gram + regularization * np.eye(gram.shape[0]),
      x_matrix.T @ y_matrix,
  )

  coefficients = np.empty((p, n_channels, n_channels), dtype=float)
  for lag_index in range(p):
    # FIXED (see review notes): x_matrix's columns are built CHANNEL-major
    # -- for channel c, lag l (1-indexed), the column is at position
    # c*p + (l-1) (all p lags for channel 0, then all p lags for channel
    # 1, ...). The original code sliced `weights[lag*n_channels:(lag+1)*n_channels]`,
    # a CONTIGUOUS block -- correct only for a LAG-major layout (all
    # channels at lag 1, then all channels at lag 2, ...), which is the
    # opposite of what's actually built above. Verified with a fully
    # identifiable synthetic AR system (every (target, source, lag) given
    # a distinct value): the original slicing recovered coefficients with
    # errors on the same order as the coefficients themselves (~150-200%
    # relative error, i.e. structurally wrong, not just noisy), while the
    # fix below (a STRIDED selection matching the channel-major layout:
    # rows lag_index, lag_index+p, lag_index+2p, ...) recovered them
    # within ~5-10% relative error, consistent with ordinary finite-sample
    # OLS estimation noise.
    coefficients[lag_index] = weights[lag_index::p, :].T
  return coefficients


def compute_condition_pdc(
    windows: np.ndarray,
    fs: float = 250.0,
    freq_band: tuple[float, float] | None = None,
    order: int = 10,
) -> np.ndarray:
  """Compute band-averaged Partial Directed Coherence (PDC).

  The returned matrix is directed: entry ``[i, j]`` represents the flow
  from source channel ``j`` to destination channel ``i``.
  """
  if windows.ndim != 3:
    raise ValueError("windows must have shape (n_windows, n_channels, n_samples)")
  if not 0.0 < fs:
    raise ValueError("fs must be positive")
  if freq_band is not None:
    low, high = freq_band
    if not 0.0 <= low < high <= fs / 2:
      raise ValueError("freq_band must satisfy 0 <= low < high <= Nyquist frequency")

  coefficients = fit_mvar_ols(windows, p=order)
  n_channels = windows.shape[1]
  frequencies = (
      np.linspace(freq_band[0], freq_band[1], 15)
      if freq_band is not None
      else np.array([15.0])
  )
  pdc_sum = np.zeros((n_channels, n_channels), dtype=float)

  for frequency in frequencies:
    transfer = np.eye(n_channels, dtype=complex)
    for lag, coefficient in enumerate(coefficients, start=1):
      transfer -= coefficient * np.exp(-2j * np.pi * frequency * lag / fs)

    column_norms = np.sqrt(np.sum(np.abs(transfer) ** 2, axis=0))
    pdc_frequency = np.abs(transfer) / (column_norms[np.newaxis, :] + 1e-12)
    np.fill_diagonal(pdc_frequency, 0.0)
    pdc_sum += pdc_frequency

  return pdc_sum / len(frequencies)


# ---------------------------------------------------------------------------
# Condition-level bookkeeping
# ---------------------------------------------------------------------------

def band_for_condition(
    condition: int,
    condition_freqs: dict[int, float] = CONDITION_FREQS,
    baseline_condition: int = BASELINE_CONDITION,
    baseline_band: tuple[float, float] = BASELINE_BAND,
    half_width: float = STIMULUS_BAND_HALFWIDTH_HZ,
) -> tuple[float, float]:
  """Return the (low, high) Hz band to use for a given condition code.

  Stimulus conditions get a narrow band centered on their SSVEP
  fundamental; the baseline condition gets the broadband SSVEP-relevant
  range, since there's no single frequency to lock onto at rest.
  """
  if condition == baseline_condition:
    return baseline_band
  if condition in condition_freqs:
    f0 = condition_freqs[condition]
    return (f0 - half_width, f0 + half_width)
  raise ValueError(f"Unknown condition code: {condition!r}")


def compute_condition_connectivity(
    windows: np.ndarray,
    y: np.ndarray,
    fs: float = 250.0,
    conditions: list[int] | None = None,
    method_fn=compute_condition_coherence,
    condition_freqs: dict[int, float] = CONDITION_FREQS,
    baseline_condition: int = BASELINE_CONDITION,
    baseline_band: tuple[float, float] = BASELINE_BAND,
    half_width: float = STIMULUS_BAND_HALFWIDTH_HZ,
) -> dict[int, dict]:
  """
  Compute one connectivity matrix per condition.
  """
  if conditions is None:
    conditions = [baseline_condition] + sorted(condition_freqs)

  results = {}
  for cond in conditions:
    band = band_for_condition(
        cond, condition_freqs, baseline_condition, baseline_band, half_width
    )
    cond_windows = windows[y == cond]
    if cond_windows.shape[0] == 0:
      results[cond] = {"matrix": None, "n_windows": 0, "band": band}
      continue
    matrix = method_fn(cond_windows, fs=fs, freq_band=band)
    results[cond] = {
        "matrix": matrix,
        "n_windows": cond_windows.shape[0],
        "band": band,
    }
  return results


def compute_baseline_matched_connectivity(
    baseline_windows: np.ndarray,
    fs: float = 250.0,
    condition_freqs: dict[int, float] = CONDITION_FREQS,
    half_width: float = STIMULUS_BAND_HALFWIDTH_HZ,
    method_fn=compute_condition_coherence,
) -> dict[int, dict]:
  """Re-score the SAME no-stimulus windows in each stimulus condition's
  own narrow band, instead of one broadband number.
  """
  results = {}
  for cond, f0 in condition_freqs.items():
    band = (f0 - half_width, f0 + half_width)
    results[cond] = {
        "matrix": method_fn(baseline_windows, fs=fs, freq_band=band),
        "band": band,
    }
  return results


def compute_difference_matrices(
    by_condition: dict[int, dict], baseline_matched: dict[int, dict]
) -> dict[int, np.ndarray]:
  """stimulus_matrix - matched_baseline_matrix, per stimulus condition.

  This isolates the connectivity that stimulation ADDS on top of
  whatever's already present at rest (much of which, at these short
  occipital-parietal distances, is volume conduction / common-reference
  mixing rather than genuine coupling) -- a positive entry means that
  channel pair synchronizes MORE under stimulation than at rest in the
  same band; near-zero or negative entries mean stimulation added little
  or nothing there.
  """
  diffs = {}
  for cond, entry in by_condition.items():
    if cond not in baseline_matched or entry["matrix"] is None:
      continue
    diffs[cond] = entry["matrix"] - baseline_matched[cond]["matrix"]
  return diffs


# ---------------------------------------------------------------------------
# Statistical significance testing via permutation
# ---------------------------------------------------------------------------

def benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
  """Apply Benjamini-Hochberg FDR correction to a 1D array of p-values.

  Returns a boolean mask of the same shape where True indicates significance at FDR level alpha.
  """
  p_values = np.asarray(p_values)
  n = len(p_values)
  if n == 0:
    return np.array([], dtype=bool)
  sort_idx = np.argsort(p_values)
  sorted_p = p_values[sort_idx]
  thresh = (np.arange(1, n + 1) / n) * alpha
  is_sig = sorted_p <= thresh
  if np.any(is_sig):
    max_sig_idx = np.max(np.where(is_sig)[0])
    sig_sorted = np.zeros(n, dtype=bool)
    sig_sorted[: max_sig_idx + 1] = True
  else:
    sig_sorted = np.zeros(n, dtype=bool)
  sig = np.zeros(n, dtype=bool)
  sig[sort_idx] = sig_sorted
  return sig


def _run_single_permutation(
    seed: int,
    pooled: np.ndarray,
    n_stim: int,
    fs: float,
    freq_band: tuple[float, float],
    method_fn,
) -> np.ndarray:
  """Helper worker for running one permutation iteration."""
  rng = np.random.RandomState(seed)
  perm = rng.permutation(len(pooled))
  pseudo_stim = pooled[perm[:n_stim]]
  pseudo_base = pooled[perm[n_stim:]]
  m_stim = method_fn(pseudo_stim, fs=fs, freq_band=freq_band)
  m_base = method_fn(pseudo_base, fs=fs, freq_band=freq_band)
  return m_stim - m_base


def compute_permutation_test(
    stim_windows: np.ndarray,
    baseline_windows: np.ndarray,
    freq_band: tuple[float, float],
    fs: float = 250.0,
    n_permutations: int = 200,
    method_fn=compute_condition_coherence,
    n_jobs: int = -1,
    random_state: int = 42,
    alpha: float = 0.05,
) -> dict:
  """Non-parametric permutation test comparing stimulus vs matched baseline connectivity.

  Null hypothesis (H0): Window labels (stimulus vs rest) are exchangeable in this frequency band.
  Observed test statistic:
    - Global: mean of difference matrix over unique channel pairs (upper triangle).
    - Edge-level: difference matrix entries for each channel pair (i, j).

  Returns dict containing:
    - 'obs_diff': observed difference matrix (stimulus - baseline)
    - 'obs_mean_diff': observed mean connectivity increase across all channel pairs
    - 'null_mean_diffs': array of shape (n_permutations,) with null mean differences
    - 'null_diff_matrices': array of shape (n_permutations, C, C)
    - 'p_val_global_one_tailed': empirical p-value for stimulation > baseline
    - 'p_val_global_two_tailed': empirical p-value for |diff| != 0
    - 'p_val_edges_one_tailed': matrix (C, C) of one-tailed edge p-values
    - 'p_val_edges_two_tailed': matrix (C, C) of two-tailed edge p-values
    - 'edge_p_mask': boolean matrix (C, C) of uncorrected significant edges (p < alpha)
    - 'edge_fdr_mask': boolean matrix (C, C) of FDR-significant edges (q < alpha)
    - 'null_mean': mean of global null distribution
    - 'null_std': standard deviation of global null distribution
    - 'null_ci_95': tuple (2.5%, 97.5%) of global null distribution
    - 'n_significant_edges_uncorrected': count of uncorrected significant edges
    - 'n_significant_edges_fdr': count of FDR significant edges
    - 'total_edges': count of total unique edges
  """
  n_channels = stim_windows.shape[1]
  triu_idx = np.triu_indices(n_channels, k=1)
  n_edges = len(triu_idx[0])

  # 1. Observed matrices
  m_stim = method_fn(stim_windows, fs=fs, freq_band=freq_band)
  m_base = method_fn(baseline_windows, fs=fs, freq_band=freq_band)
  obs_diff = m_stim - m_base
  obs_mean_diff = float(obs_diff[triu_idx].mean())

  # 2. Pool windows and generate seeds
  pooled = np.concatenate([stim_windows, baseline_windows], axis=0)
  n_stim = len(stim_windows)
  master_rng = np.random.RandomState(random_state)
  perm_seeds = master_rng.randint(0, 2**31 - 1, size=n_permutations)

  # 3. Parallel permutation iterations
  null_diff_matrices = Parallel(n_jobs=n_jobs)(
      delayed(_run_single_permutation)(
          seed=int(s),
          pooled=pooled,
          n_stim=n_stim,
          fs=fs,
          freq_band=freq_band,
          method_fn=method_fn,
      )
      for s in perm_seeds
  )
  null_diff_matrices = np.array(null_diff_matrices)  # (n_permutations, n_channels, n_channels)
  null_mean_diffs = np.array([d[triu_idx].mean() for d in null_diff_matrices])

  # 4. Global p-values (with +1 pseudo-count to avoid p=0)
  p_val_global_one_tailed = float((1 + np.sum(null_mean_diffs >= obs_mean_diff)) / (n_permutations + 1))
  null_global_mean = float(np.mean(null_mean_diffs))
  null_global_std = float(np.std(null_mean_diffs))
  null_ci_95 = (
      float(np.percentile(null_mean_diffs, 2.5)),
      float(np.percentile(null_mean_diffs, 97.5)),
  )
  obs_dist_center = abs(obs_mean_diff - null_global_mean)
  p_val_global_two_tailed = float(
      (1 + np.sum(np.abs(null_mean_diffs - null_global_mean) >= obs_dist_center)) / (n_permutations + 1)
  )

  # 5. Edge-level p-values
  p_val_edges_one_tailed = np.ones((n_channels, n_channels), dtype=float)
  p_val_edges_two_tailed = np.ones((n_channels, n_channels), dtype=float)
  edge_p_mask = np.zeros((n_channels, n_channels), dtype=bool)
  edge_fdr_mask = np.zeros((n_channels, n_channels), dtype=bool)

  edge_p_1d = []
  for i, j in zip(*triu_idx):
    null_edge_vals = null_diff_matrices[:, i, j]
    obs_val = obs_diff[i, j]
    p_1t = float((1 + np.sum(null_edge_vals >= obs_val)) / (n_permutations + 1))
    p_val_edges_one_tailed[i, j] = p_val_edges_one_tailed[j, i] = p_1t
    edge_p_1d.append(p_1t)

    null_edge_mu = float(np.mean(null_edge_vals))
    p_2t = float(
        (1 + np.sum(np.abs(null_edge_vals - null_edge_mu) >= abs(obs_val - null_edge_mu)))
        / (n_permutations + 1)
    )
    p_val_edges_two_tailed[i, j] = p_val_edges_two_tailed[j, i] = p_2t

  edge_p_1d = np.array(edge_p_1d)
  fdr_sig_1d = benjamini_hochberg(edge_p_1d, alpha=alpha)
  uncorr_sig_1d = edge_p_1d < alpha

  for k, (i, j) in enumerate(zip(*triu_idx)):
    edge_p_mask[i, j] = edge_p_mask[j, i] = bool(uncorr_sig_1d[k])
    edge_fdr_mask[i, j] = edge_fdr_mask[j, i] = bool(fdr_sig_1d[k])

  return {
      "obs_diff": obs_diff,
      "obs_mean_diff": obs_mean_diff,
      "null_mean_diffs": null_mean_diffs,
      "null_diff_matrices": null_diff_matrices,
      "p_val_global_one_tailed": p_val_global_one_tailed,
      "p_val_global_two_tailed": p_val_global_two_tailed,
      "p_val_edges_one_tailed": p_val_edges_one_tailed,
      "p_val_edges_two_tailed": p_val_edges_two_tailed,
      "edge_p_mask": edge_p_mask,
      "edge_fdr_mask": edge_fdr_mask,
      "null_mean": null_global_mean,
      "null_std": null_global_std,
      "null_ci_95": null_ci_95,
      "n_significant_edges_uncorrected": int(np.sum(uncorr_sig_1d)),
      "n_significant_edges_fdr": int(np.sum(fdr_sig_1d)),
      "total_edges": n_edges,
  }


def extract_connectivity_by_condition(
    windows: np.ndarray,
    y: np.ndarray,
    fs: float = 250.0,
    method_fn=compute_condition_coherence,
    conditions: list[int] | None = None,
    channel_names: list[str] | None = None,
    vpp_limits: tuple[float, float] = (5.0, 120.0),
    std_limits: tuple[float, float] = (1.0, 35.0),
    channel_fail_fraction_thresh: float = 0.5,
    verbose: bool = False,
    baseline_condition: int = BASELINE_CONDITION,
    condition_freqs: dict[int, float] = CONDITION_FREQS,
    half_width: float = STIMULUS_BAND_HALFWIDTH_HZ,
    n_permutations: int = 0,
    n_jobs: int = -1,
    random_state: int = 42,
    alpha: float = 0.05,
) -> dict:
  """Run the two-tier artifact-quality pipeline once (shared channel-drop
  decision across all conditions for this subject), then compute:
    1. one connectivity matrix per condition (broadband baseline + narrow
       band per stimulus frequency) -- the raw/absolute view;
    2. the SAME no-stimulus windows re-scored in each stimulus condition's
       own narrow band (`baseline_matched`) -- for a fair, same-band
       comparison;
    3. the difference matrices (stimulus - matched baseline) -- the
       cleanest view of what stimulation actually adds;
    4. optional permutation significance testing for each difference matrix.
  """
  qc = apply_artifact_quality_pipeline(
      windows,
      y,
      channel_names=channel_names,
      vpp_min=vpp_limits[0],
      vpp_max=vpp_limits[1],
      std_min=std_limits[0],
      std_max=std_limits[1],
      channel_fail_fraction_thresh=channel_fail_fraction_thresh,
      verbose=verbose,
  )
  windows_clean, y_clean = qc["windows_clean"], qc["y_clean"]

  if windows_clean.shape[0] == 0:
    raise ValueError("All windows were rejected by the validation mask.")

  by_condition = compute_condition_connectivity(
      windows_clean, y_clean, fs=fs, conditions=conditions, method_fn=method_fn,
      condition_freqs=condition_freqs, baseline_condition=baseline_condition,
      half_width=half_width,
  )

  baseline_windows = windows_clean[y_clean == baseline_condition]
  baseline_matched = None
  diffs = None
  permutation_tests = None

  if baseline_windows.shape[0] > 0:
    baseline_matched = compute_baseline_matched_connectivity(
        baseline_windows, fs=fs, condition_freqs=condition_freqs,
        half_width=half_width, method_fn=method_fn,
    )
    diffs = compute_difference_matrices(by_condition, baseline_matched)

    if n_permutations > 0:
      if verbose:
        print(f"\n[Permutation Test] Running {n_permutations} permutations per frequency condition...")
      permutation_tests = {}
      for cond, f0 in condition_freqs.items():
        stim_windows = windows_clean[y_clean == cond]
        if stim_windows.shape[0] == 0:
          permutation_tests[cond] = None
          continue
        band = (f0 - half_width, f0 + half_width)
        permutation_tests[cond] = compute_permutation_test(
            stim_windows=stim_windows,
            baseline_windows=baseline_windows,
            freq_band=band,
            fs=fs,
            n_permutations=n_permutations,
            method_fn=method_fn,
            n_jobs=n_jobs,
            random_state=random_state,
            alpha=alpha,
        )

  return {
      "by_condition": by_condition,
      "baseline_matched": baseline_matched,
      "diffs": diffs,
      "permutation_tests": permutation_tests,
      "channels_kept": qc["channels_kept"],
      "channels_kept_idx": qc["channels_kept_idx"],
      "channels_dropped": qc["channels_dropped"],
  }