"""Tests for regression and classification evaluation metrics."""
from __future__ import annotations

import numpy as np
import pytest

from src.training.evaluate import classification_metrics, regression_metrics


# ── Regression metrics ─────────────────────────────────────────────────────────

def test_regression_metrics_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    m = regression_metrics(y, y)
    assert m["mae"] == pytest.approx(0.0)
    assert m["rmse"] == pytest.approx(0.0)
    assert m["r2"] == pytest.approx(1.0)


def test_regression_metrics_keys():
    y = np.array([1.0, 2.0, 3.0])
    m = regression_metrics(y, y + 0.1)
    assert set(m.keys()) == {"mae", "rmse", "r2"}


def test_regression_metrics_positive_errors():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.5, 3.5])
    m = regression_metrics(y_true, y_pred)
    assert m["mae"] > 0
    assert m["rmse"] > 0


# ── Classification metrics ─────────────────────────────────────────────────────

def _make_multiclass_proba(y_true, n_classes=4, noise=0.1):
    n = len(y_true)
    proba = np.full((n, n_classes), noise / (n_classes - 1))
    for i, label in enumerate(y_true):
        proba[i, label] = 1.0 - noise
    return proba / proba.sum(axis=1, keepdims=True)


def test_classification_metrics_perfect_multiclass():
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    y_pred = y_true.copy()
    proba = _make_multiclass_proba(y_true, n_classes=4, noise=0.0)
    m = classification_metrics(y_true, y_pred, proba, class_names=["bust", "bench", "starter", "star"])
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["f1_macro"] == pytest.approx(1.0)
    assert m["auc"] == pytest.approx(1.0)


def test_classification_metrics_has_required_keys():
    y_true = np.array([0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 3])
    proba = _make_multiclass_proba(y_true, n_classes=4, noise=0.05)
    m = classification_metrics(y_true, y_pred, proba)
    for key in ("accuracy", "balanced_accuracy", "auc", "f1_macro"):
        assert key in m, f"Missing key: {key}"


def test_classification_metrics_per_class_keys():
    y_true = np.array([0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 3])
    proba = _make_multiclass_proba(y_true, n_classes=4, noise=0.05)
    names = ["bust", "bench", "starter", "star"]
    m = classification_metrics(y_true, y_pred, proba, class_names=names)
    for name in names:
        assert f"precision_{name}" in m
        assert f"recall_{name}" in m
        assert f"f1_{name}" in m


def test_classification_metrics_values_in_range():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 4, size=100)
    y_pred = rng.integers(0, 4, size=100)
    proba = rng.dirichlet(np.ones(4), size=100)
    m = classification_metrics(y_true, y_pred, proba)
    assert 0.0 <= m["accuracy"] <= 1.0
    assert 0.0 <= m["f1_macro"] <= 1.0
    assert 0.0 <= m["auc"] <= 1.0
