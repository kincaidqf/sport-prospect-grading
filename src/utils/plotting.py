"""Shared plotting utilities for tabular model training outputs."""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.features import get_coef_df, get_xgb_importance_df


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_ARTIFACT_DIR = os.path.join(PROJECT_ROOT, "outputs", "plots")


def _artifact_path(filename, artifact_dir=DEFAULT_ARTIFACT_DIR):
    os.makedirs(artifact_dir, exist_ok=True)
    return os.path.join(artifact_dir, filename)


def plot_feature_importance(results, col_info, target_mode, artifact_dir=DEFAULT_ARTIFACT_DIR):
    """Plot feature importance or coefficients for all fitted models."""
    numeric_cols = col_info["numeric_cols"]
    categorical_cols = col_info["categorical_cols"]
    ordinal_cols = col_info["ordinal_cols"]

    importance_data = {}
    for name, res in results.items():
        pipe = res["pipe"]
        if res["importance_kind"] == "coef":
            coef_df = get_coef_df(
                pipe,
                numeric_cols,
                categorical_cols,
                ordinal_cols,
                res["estimator_step"],
            )
            importance_data[name] = coef_df.rename(columns={"coefficient": "importance"})
        elif res["importance_kind"] == "xgb":
            importance_data[name] = get_xgb_importance_df(
                pipe,
                numeric_cols,
                categorical_cols,
                ordinal_cols,
            )

    n_models = len(importance_data)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 8))
    if n_models == 1:
        axes = [axes]
    fig.suptitle(f"Feature Importance — {target_mode}", fontsize=14, fontweight="bold")

    for ax, (name, imp_df) in zip(axes, importance_data.items()):
        top = imp_df.head(15).copy()
        if "abs_coef" in top.columns:
            top = top.sort_values("abs_coef", ascending=True)
            colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in top["importance"]]
            ax.barh(top["feature"], top["importance"], color=colors)
            ax.set_xlabel("Coefficient (standardized)")
            ax.axvline(0, color="black", linewidth=0.5)
        else:
            top = top.sort_values("importance", ascending=True)
            ax.barh(top["feature"], top["importance"], color="#3498db")
            ax.set_xlabel("Feature Importance (gain)")

        ax.set_title(name)
        ax.tick_params(axis="y", labelsize=9)

    plt.tight_layout()
    out_path = _artifact_path("feature_importance.png", artifact_dir)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")
    plt.show()

    plot_importance_heatmap(importance_data, target_mode, artifact_dir)


def plot_importance_heatmap(importance_data, target_mode, artifact_dir=DEFAULT_ARTIFACT_DIR):
    """Render a normalized cross-model feature-importance heatmap."""
    all_features = set()
    for imp_df in importance_data.values():
        all_features.update(imp_df["feature"].tolist())
    all_features = sorted(all_features)

    matrix = pd.DataFrame(index=all_features)
    for name, imp_df in importance_data.items():
        vals = imp_df.set_index("feature")
        if "abs_coef" in vals.columns:
            matrix[name] = vals["abs_coef"]
        else:
            matrix[name] = vals["importance"]
    matrix = matrix.fillna(0)

    matrix_norm = matrix.copy()
    for col in matrix_norm.columns:
        mx = matrix_norm[col].max()
        if mx > 0:
            matrix_norm[col] = matrix_norm[col] / mx

    matrix_norm["_mean"] = matrix_norm.mean(axis=1)
    matrix_norm = matrix_norm.sort_values("_mean", ascending=True).drop(columns=["_mean"])
    matrix_norm = matrix_norm.tail(20)

    fig, ax = plt.subplots(figsize=(8, max(6, len(matrix_norm) * 0.4)))
    im = ax.imshow(matrix_norm.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(matrix_norm.columns)))
    ax.set_xticklabels(matrix_norm.columns, fontsize=10)
    ax.set_yticks(range(len(matrix_norm.index)))
    ax.set_yticklabels(matrix_norm.index, fontsize=9)

    for i in range(len(matrix_norm.index)):
        for j in range(len(matrix_norm.columns)):
            val = matrix_norm.values[i, j]
            color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)

    ax.set_title(
        f"Normalized Feature Importance Comparison — {target_mode}",
        fontsize=12,
        fontweight="bold",
    )
    fig.colorbar(im, ax=ax, label="Relative importance (0–1)", shrink=0.8)
    plt.tight_layout()

    out_path = _artifact_path("importance_heatmap.png", artifact_dir)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path}")
    plt.show()


def plot_model_summary(results, target_mode, task_type, artifact_dir=DEFAULT_ARTIFACT_DIR):
    """Render a compact summary table of model metrics."""
    rows = []
    for name, res in results.items():
        row = {"Model": name}
        if task_type == "classification":
            row["Accuracy"] = res["accuracy"]
            row["ROC-AUC"] = res["auc"]
            row["C / alpha"] = f"{res.get('C', res.get('alpha', ''))}"
        else:
            row["R²"] = res["r2"]
            row["RMSE"] = res["rmse"]
            row["MAE"] = res["mae"]
            row["alpha"] = res.get("alpha", "")
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(max(8, 2.5 * len(summary_df.columns)), 1.2 + 0.5 * len(rows)))
    ax.axis("off")

    cell_text = []
    for _, row in summary_df.iterrows():
        formatted = []
        for value in row:
            formatted.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        cell_text.append(formatted)

    table = ax.table(
        cellText=cell_text,
        colLabels=summary_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.8)

    for idx in range(len(summary_df.columns)):
        table[0, idx].set_facecolor("#2c3e50")
        table[0, idx].set_text_props(color="white", fontweight="bold")

    for row_idx in range(len(rows)):
        color = "#ecf0f1" if row_idx % 2 == 0 else "white"
        for col_idx in range(len(summary_df.columns)):
            table[row_idx + 1, col_idx].set_facecolor(color)

    ax.set_title(
        f"Model Comparison — {task_type.title()} ({target_mode})",
        fontsize=13,
        fontweight="bold",
        pad=20,
    )

    plt.tight_layout()
    out_path = _artifact_path("model_summary.png", artifact_dir)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path}")
    plt.show()


def save_and_log(fig, filename, summary_metrics, artifact_dir=DEFAULT_ARTIFACT_DIR):
    """Save a plot locally and log it plus summary metrics to the active MLflow run."""
    out_path = _artifact_path(filename, artifact_dir)
    fig.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")

    import mlflow

    if mlflow.active_run() is not None:
        mlflow.log_artifact(out_path, artifact_path="plots")
        for name, metrics in summary_metrics.items():
            mlflow.log_metrics({f"{name.lower()}_{key}": value for key, value in metrics.items()})

    plt.show()
