"""Trial-grouping utilities for leak-free cross-validation.

Each SSVEP trial (5.0s of stimulation) is sliced into several consecutive
1.0s sub-windows (5, given this project's sub_window_sec=1.0). Naively
splitting those sub-windows with plain StratifiedKFold lets sub-windows
from the SAME trial land in both the training and test fold -- since
consecutive sub-windows of one trial share near-identical artifacts,
electrode state, and background noise, a high-capacity model (especially
a CNN) can partly "recognize the trial" instead of learning the SSVEP
frequency, inflating CV accuracy.

`build_trial_ids` reconstructs which windows belong to the same trial
from the label array alone, using the fact that windows are stored in
contiguous same-label blocks, each block further divided into consecutive
chunks of `sub_windows_per_trial` (this was confirmed against this
project's actual data: 40 windows per condition = 8 trials x 5
sub-windows). Feed the result to `StratifiedGroupKFold` as `groups=` so
every sub-window of a trial is forced into the same fold.
"""

import numpy as np


def build_trial_ids(
  y: np.ndarray, sub_windows_per_trial: "int | dict[int, int]" = 5
) -> np.ndarray:
  """Assign a trial id to each window.

  Assumes windows are grouped into contiguous same-label blocks (one per
  condition occurrence), each block evenly divisible into consecutive
  chunks of `sub_windows_per_trial` windows -- one trial each, in order.

  Parameters
  ----------
  y : np.ndarray, shape (n_windows,)
      Condition/class label per window, in original acquisition order
      (call this BEFORE any quality-gate window rejection, then index
      the result with the same `valid_mask` used elsewhere, e.g.
      `trial_ids[valid_mask == 1]`, exactly like `y_clean`).
      sub_windows_per_trial : int or dict[label -> int]
      Number of consecutive sub-windows that make up one trial (default
      5, i.e. a 5.0s stimulus trial cut into 1.0s sub-windows). Pass a
      dict to use a different trial length per label -- e.g. for the
      connectivity analysis, which also uses the 201 fixation cross
      (2.0s = 2 sub-windows per trial):
      ``{201: 2, 101: 5, 102: 5, 103: 5, 104: 5, 105: 5}``.

  Returns
  -------
  trial_ids : np.ndarray of int, shape (n_windows,)
      Unique id per trial; every window belonging to the same trial gets
      the same id.
  """
  n = len(y)
  trial_ids = np.empty(n, dtype=int)
  next_id = 0
  i = 0
  while i < n:
    j = i
    while j < n and y[j] == y[i]:
      j += 1
    block_len = j - i
    if isinstance(sub_windows_per_trial, dict):
      per_trial = int(sub_windows_per_trial[int(y[i])])
    else:
      per_trial = int(sub_windows_per_trial)
    n_full_trials = block_len // per_trial
    remainder = block_len - n_full_trials * per_trial

    if remainder != 0:
      print(
          f"[!] build_trial_ids: a same-label block of length {block_len} "
          f"(label={y[i]!r}, starting at index {i}) is not evenly "
          f"divisible by sub_windows_per_trial={per_trial}. "
          "The trailing remainder is assigned its own trial id, but this "
          "usually means sub_windows_per_trial is wrong for this data -- "
          "verify before trusting the CV split."
      )

    pos = i
    for _ in range(n_full_trials):
      trial_ids[pos:pos + per_trial] = next_id
      next_id += 1
      pos += per_trial
    if remainder:
      trial_ids[pos:j] = next_id
      next_id += 1

    i = j

  return trial_ids