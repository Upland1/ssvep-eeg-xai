"""Feature Selection Strategies for SSVEP Hybrid Feature Spaces."""

from typing import List, Literal, Tuple
import numpy as np
from sklearn.feature_selection import f_classif, mutual_info_classif


def select_k_best_fbcsp_features(
    X_fbcsp_tr: np.ndarray,
    y_tr: np.ndarray,
    n_features_to_select: int = 15,
    method: Literal["anova", "mutual_info"] = "anova",
    random_state: int = 42,
) -> Tuple[List[int], np.ndarray]:
    """Select top K most discriminative FBCSP features using filter statistics.

    Computed strictly on in-fold training data to prevent leakage.

    Parameters
    ----------
    X_fbcsp_tr : np.ndarray
        In-fold scaled FBCSP feature matrix of shape (N_train, 50).
    y_tr : np.ndarray
        In-fold class labels of shape (N_train,).
    n_features_to_select : int
        Number of FBCSP features to retain.
    method : {"anova", "mutual_info"}
        Scoring metric for feature ranking.
    random_state : int
        Seed for mutual information estimation.

    Returns
    -------
    selected_indices : List[int]
        Sorted indices of top K selected FBCSP features.
    scores : np.ndarray
        Ranking scores for all 50 candidate features.
    """
    if method == "anova":
        f_scores, _ = f_classif(X_fbcsp_tr, y_tr)
        # Replace any potential NaNs from zero variance with 0
        scores = np.nan_to_num(f_scores, nan=0.0)
    elif method == "mutual_info":
        scores = mutual_info_classif(
            X_fbcsp_tr, y_tr, random_state=random_state
        )
    else:
        raise ValueError(f"Unknown selection method: {method}")

    # Rank features descending by score
    ranked_indices = np.argsort(scores)[::-1]
    selected_indices = sorted(ranked_indices[:n_features_to_select].tolist())

    return selected_indices, scores