"""Shared probability layer for the Probability Stacking Model (PSM Phase 1).

Provides constants, helpers, and the BaseModelBundle dataclass so that
classification and regression model families expose a uniform 4-class
probability interface consumed by multimodal.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.calibration import CalibratedClassifierCV, FrozenEstimator

# ── Constants ──────────────────────────────────────────────────────────────────

TIER_CLASS_NAMES: list[str] = ["bust", "bench", "starter", "star"]
TIER_LABELS: list[int]      = [0, 1, 2, 3]
TIER_THRESHOLDS: tuple      = (-0.5, 0.5, 1.5)
PROBA_COLUMNS: list[str]    = ["p_bust", "p_bench", "p_starter", "p_star"]

# ── Helpers ────────────────────────────────────────────────────────────────────

def normalize_proba(proba: np.ndarray) -> np.ndarray:
    """Normalize each row of a probability array so rows sum to 1.0.

    Clips negative values to 0 before normalizing to handle floating-point
    edge cases from calibration or CDF arithmetic.
    """
    p = np.clip(proba, 0.0, None)
    row_sums = p.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return p / row_sums


def proba_to_dataframe(proba: np.ndarray, index=None) -> pd.DataFrame:
    """Convert a (N, 4) probability array to a DataFrame with PROBA_COLUMNS."""
    if proba.shape[1] != 4:
        raise ValueError(f"Expected 4 probability columns, got {proba.shape[1]}")
    return pd.DataFrame(
        normalize_proba(proba),
        columns=PROBA_COLUMNS,
        index=index,
    )


def zscore_to_tier_proba(z_scores: np.ndarray, residual_std: float,
                          thresholds: tuple = TIER_THRESHOLDS) -> np.ndarray:
    """Convert regression z-score predictions to 4-class tier probabilities.

    Treats each prediction as a Gaussian distribution with mean=z and
    std=residual_std, then computes probability mass within each tier interval.
    Probabilities sum to exactly 1.0 by construction.
    """
    lo, mid, hi = thresholds
    sigma = max(residual_std, 1e-8)

    p_bust    = norm.cdf(lo,  loc=z_scores, scale=sigma)
    p_bench   = norm.cdf(mid, loc=z_scores, scale=sigma) - norm.cdf(lo,  loc=z_scores, scale=sigma)
    p_starter = norm.cdf(hi,  loc=z_scores, scale=sigma) - norm.cdf(mid, loc=z_scores, scale=sigma)
    p_star    = 1.0 - norm.cdf(hi, loc=z_scores, scale=sigma)

    return np.column_stack([p_bust, p_bench, p_starter, p_star])


def build_prefit_calibrator(base_estimator: Any, cal_X: Any,
                             cal_y: Any, method: str = "sigmoid") -> CalibratedClassifierCV:
    """Fit post-hoc calibration on a held-out calibration set without refitting the base.

    Uses FrozenEstimator (sklearn >= 1.8) so the base estimator is never refit.
    cal_X / cal_y must be the calibration split that was excluded from base training.
    """
    calibrator = CalibratedClassifierCV(
        estimator=FrozenEstimator(base_estimator), cv=None, method=method
    )
    calibrator.fit(cal_X, cal_y)
    return calibrator


# ── BaseModelBundle ────────────────────────────────────────────────────────────

@dataclass
class BaseModelBundle:
    """Wraps a fitted model and exposes a uniform predict_tier_proba interface.

    Fields
    ------
    name            : unique identifier for this bundle (e.g. "logistic_l2")
    task            : "classification" or "regression"
    estimator       : fitted sklearn pipeline (used for regression prediction
                      and as the base for classification before calibration)
    feature_cols    : ordered list of columns expected in input DataFrames
    proba_estimator : CalibratedClassifierCV for classification; None for regression
    residual_std    : training residual std for regression Gaussian CDF; 0.0 for classification
    thresholds      : tier boundary z-scores — should always equal TIER_THRESHOLDS
    """

    name: str
    task: str
    estimator: Any
    feature_cols: list[str]
    proba_estimator: Any
    residual_std: float
    thresholds: tuple

    def predict_tier_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame of shape (N, 4) with columns PROBA_COLUMNS.

        Probabilities are non-negative and sum to 1.0 per row.
        """
        X = df[self.feature_cols]

        if self.task == "classification":
            estimator = self.proba_estimator if self.proba_estimator is not None else self.estimator
            proba = estimator.predict_proba(X)
            return proba_to_dataframe(proba, index=df.index)

        if self.task == "regression":
            if self.residual_std <= 0.0:
                raise ValueError(
                    f"Bundle '{self.name}' has residual_std={self.residual_std}; "
                    "predict_tier_proba requires a positive residual_std for regression bundles."
                )
            z_scores = self.estimator.predict(X)
            proba = zscore_to_tier_proba(
                np.asarray(z_scores, dtype=float),
                self.residual_std,
                self.thresholds,
            )
            return proba_to_dataframe(proba, index=df.index)

        raise ValueError(f"Unknown task '{self.task}'; expected 'classification' or 'regression'.")


def load_text_tier_proba_table(path: str) -> pd.DataFrame:
    """Load Name/draft_year + PROBA_COLUMNS from a text-model tier CSV export."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"text tier proba CSV not found: {path}")
    raw = pd.read_csv(p)
    id_cols = ["Name", "draft_year"]
    for c in id_cols + list(PROBA_COLUMNS):
        if c not in raw.columns:
            raise ValueError(f"Text tier proba CSV missing column {c!r}: {path}")
    out = raw[id_cols + list(PROBA_COLUMNS)].copy()
    out["Name"] = out["Name"].astype(str).str.strip()
    out["draft_year"] = pd.to_numeric(out["draft_year"], errors="coerce")
    out = out.dropna(subset=id_cols + list(PROBA_COLUMNS))
    out["draft_year"] = out["draft_year"].astype(int)
    if out.duplicated(subset=id_cols).any():
        out = out.drop_duplicates(subset=id_cols, keep="last")
    return out.reset_index(drop=True)


def align_text_tier_proba_to_meta_cols(
    df: pd.DataFrame,
    lookup: pd.DataFrame,
    text_meta_key: str,
) -> pd.DataFrame:
    """Map text tier probabilities onto df rows as ``text__<text_meta_key>__p_*``; missing → uniform."""
    id_cols = ["Name", "draft_year"]
    for c in id_cols:
        if c not in df.columns:
            raise ValueError(
                f"Text tier proba merge requires column {c!r} on the input frame.",
            )
    side = df[id_cols].copy()
    side["Name"] = side["Name"].astype(str).str.strip()
    side["draft_year"] = pd.to_numeric(side["draft_year"], errors="coerce").astype("Int64")
    merged = side.merge(lookup, on=id_cols, how="left", validate="many_to_one")
    raw = merged[list(PROBA_COLUMNS)].to_numpy(dtype=float, copy=True)
    missing = np.isnan(raw).any(axis=1)
    if missing.any():
        raw[missing] = 0.25
    raw = normalize_proba(raw)
    out = pd.DataFrame(raw, columns=PROBA_COLUMNS, index=df.index)
    return out.rename(columns={c: f"text__{text_meta_key}__{c}" for c in PROBA_COLUMNS})
