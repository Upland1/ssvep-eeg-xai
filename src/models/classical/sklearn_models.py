"""Scikit-learn Linear and Discriminant Analysis Model Suite for SSVEP decoding."""

from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier


def get_sklearn_model_suite() -> dict:
  """Returns a catalog of scikit-learn models mapped to Sections 1.1 and 1.2.

  Updated for Scikit-Learn 1.8+ API standards.
  """
  return {
      # =========================================================================
      # 1.1. Linear Models (sklearn.linear_model)
      # =========================================================================
      # 1.1.2 Ridge Classification (L2 regularized closed-form)
      "1.1.2_RidgeClassifier": RidgeClassifier(
          alpha=1.0, class_weight="balanced"
      ),
      # 1.1.11 Logistic Regression (L2 penalty via l1_ratio=0.0)
      "1.1.11_LogReg_L2": LogisticRegression(
          l1_ratio=0.0,
          C=1.0,
          solver="lbfgs",
          max_iter=1000,
          class_weight="balanced",
      ),
      # 1.1.3 Logistic Regression (L1 / Lasso sparsity penalty via l1_ratio=1.0)
      "1.1.3_LogReg_L1_Lasso": LogisticRegression(
          l1_ratio=1.0,
          C=1.0,
          solver="saga",
          max_iter=3000,
          tol=1e-3,
          class_weight="balanced",
      ),
      # 1.1.5 Logistic Regression (Elastic-Net: L1 + L2)
      "1.1.5_LogReg_ElasticNet": LogisticRegression(
          l1_ratio=0.5,
          C=1.0,
          solver="saga",
          max_iter=3000,
          tol=1e-3,
          class_weight="balanced",
      ),
      # 1.1.13 Stochastic Gradient Descent (SGD log-loss)
      "1.1.13_SGD_LogLoss": SGDClassifier(
          loss="log_loss",
          penalty="elasticnet",
          l1_ratio=0.5,
          alpha=1e-3,
          max_iter=2000,
          class_weight="balanced",
          random_state=42,
      ),
      # 1.1.14 Passive-Aggressive formulation via SGDClassifier
      "1.1.14_PassiveAggressive_SGD": SGDClassifier(
          loss="hinge",
          penalty=None,
          learning_rate="pa1",
          eta0=1.0,
          max_iter=2000,
          random_state=42,
      ),
      # =========================================================================
      # 1.2. Linear and Quadratic Discriminant Analysis
      # =========================================================================
      # 1.2.1 Standard LDA (SVD solver, empirical covariance)
      "1.2.1_LDA_Standard_SVD": LinearDiscriminantAnalysis(solver="svd"),
      # 1.2.4 & 1.2.5 Regularized Shrinkage-LDA (LSQR + Ledoit-Wolf auto shrinkage)
      "1.2.4_Shrinkage_LDA_LSQR": LinearDiscriminantAnalysis(
          solver="lsqr", shrinkage="auto"
      ),
      # 1.2.4 & 1.2.5 Shrinkage-LDA (Eigen solver + Ledoit-Wolf auto shrinkage)
      "1.2.4_Shrinkage_LDA_Eigen": LinearDiscriminantAnalysis(
          solver="eigen", shrinkage="auto"
      ),
      # 1.2.2 Quadratic Discriminant Analysis (reg_param regularizes covariance towards spherical)
      "1.2.2_QDA_Regularized": QuadraticDiscriminantAnalysis(reg_param=0.5),
  }