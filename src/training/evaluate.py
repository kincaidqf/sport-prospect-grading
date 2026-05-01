"""Evaluation metrics for regression, classification, and ranking tasks."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    cohen_kappa_score,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return MAE, RMSE, and R² for a regression model."""
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str] | None = None,
) -> dict:
    """Return accuracy, balanced_accuracy, AUC, macro F1, and per-class metrics.

    For multi-class (y_prob is 2-D with >2 columns): returns macro OvR ROC-AUC
    and per-class precision/recall/f1 keyed by class name.  For binary: returns
    standard ROC-AUC.  The 'auc' key is always present so callers never branch.
    """
    y_prob = np.asarray(y_prob)
    is_multiclass = y_prob.ndim == 2 and y_prob.shape[1] > 2

    out: dict = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }

    if is_multiclass:
        out["auc"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
        out["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

        labels = list(range(len(class_names))) if class_names is not None else None
        precision, recall, fscore, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=labels, zero_division=0
        )
        n_classes = len(precision)
        names = (class_names or [f"class_{i}" for i in range(n_classes)])[:n_classes]
        for i, name in enumerate(names):
            out[f"precision_{name}"] = float(precision[i])
            out[f"recall_{name}"] = float(recall[i])
            out[f"f1_{name}"] = float(fscore[i])
    else:
        out["auc"] = float(roc_auc_score(y_true, y_prob))

    return out


def ordinal_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str] | None = None,
) -> dict:
    """Return standard classification metrics plus ordinal-distance metrics."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    y_prob_arr = np.asarray(y_prob, dtype=float)

    out = classification_metrics(y_true, y_pred, y_prob, class_names=class_names)

    ordinal_error = np.abs(y_pred_arr - y_true_arr)
    max_distance = max(y_prob_arr.shape[1] - 1, 1)
    class_values = np.arange(y_prob_arr.shape[1], dtype=float)
    expected_class = y_prob_arr @ class_values

    out["top1_accuracy"] = out["accuracy"]
    out["within_one_accuracy"] = float(np.mean(ordinal_error <= 1.0))
    out["ordinal_mae"] = float(np.mean(ordinal_error))
    out["distance_weighted_accuracy"] = float(np.mean(1.0 - (ordinal_error / max_distance)))
    out["expected_class_mae"] = float(np.mean(np.abs(expected_class - y_true_arr)))
    kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    out["quadratic_weighted_kappa"] = float(kappa) if np.isfinite(kappa) else 0.0
    return out


def print_metrics(metrics: dict, prefix: str = "") -> None:
    label = f"[{prefix}] " if prefix else ""
    for k, v in metrics.items():
        print(f"{label}{k}: {v:.4f}")
