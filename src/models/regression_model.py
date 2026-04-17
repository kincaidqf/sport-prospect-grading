"""Lasso regression model for NCAA stat → NBA performance prediction."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class LassoRegressionModel:
    """Cross-validated Lasso regression over numerical NCAA features."""

    def __init__(
        self,
        alpha_min: float = 1e-4,
        alpha_max: float = 1e2,
        alpha_steps: int = 100,
        max_iter: int = 10_000,
        cv_folds: int = 5,
    ) -> None:
        alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), alpha_steps)
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("lasso", LassoCV(alphas=alphas, max_iter=max_iter, cv=cv_folds, n_jobs=-1)),
        ])
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LassoRegressionModel":
        # TODO: implement training logic
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        # TODO: implement inference
        raise NotImplementedError

    def selected_features(self, feature_names: list[str]) -> list[str]:
        """Return feature names with non-zero lasso coefficients."""
        # TODO: implement after fit()
        raise NotImplementedError

    @property
    def alpha(self) -> float:
        return self.pipeline.named_steps["lasso"].alpha_
