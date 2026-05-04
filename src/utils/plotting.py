"""Shared plotting utilities for tabular model training outputs."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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
                step_name=res.get("estimator_step", "xgb"),
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
    plt.close(fig)

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
    plt.close(fig)


def plot_model_summary(results, target_mode, task_type, artifact_dir=DEFAULT_ARTIFACT_DIR):
    """Render a compact summary table of model metrics."""
    rows = []
    for name, res in results.items():
        row = {"Model": name}
        if task_type == "classification":
            row["Accuracy"] = res["accuracy"]
            row["F1-macro"] = res.get("f1_macro", float("nan"))
            row["Bal.Acc"] = res.get("balanced_accuracy", float("nan"))
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
    plt.close(fig)


def save_and_log(fig, filename, summary_metrics, artifact_dir=DEFAULT_ARTIFACT_DIR):
    """Save a plot locally and log summary metrics to the active MLflow run.

    Per spec, only loss-curve PNGs go to MLflow as artifacts — result plots stay local.
    """
    out_path = _artifact_path(filename, artifact_dir)
    fig.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")

    import mlflow

    if mlflow.active_run() is not None:
        for name, metrics in summary_metrics.items():
            mlflow.log_metrics({f"{name.lower()}_{key}": value for key, value in metrics.items()})

    plt.close(fig)


def _explicit_artifact_path(filename, artifact_dir):
    if artifact_dir is None:
        raise ValueError("artifact_dir is required for multimodal plots")
    return _artifact_path(filename, artifact_dir)


def _format_table_value(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


_MM_SHORT_NAMES = {
    "multimodal__stacker":         "Stacker (final)",
    "classification__logistic_l1": "CLF - L1 Logistic",
    "classification__logistic_l2": "CLF - L2 Logistic",
    "classification__xgboost":     "CLF - XGBoost",
    "regression__lasso":           "REG - Lasso",
    "regression__ridge":           "REG - Ridge",
    "regression__xgboost":         "REG - XGBoost",
}

_MM_LEGEND = [
    ("Acc",         "Exact accuracy — fraction predicted to the correct tier  (↑ better)"),
    ("Within 1",    "% where |predicted − actual| ≤ 1 tier; partial credit for near misses  (↑ better)"),
    ("Dist Wt Acc", "1 − |err|/3 per prediction; 1.0 exact / 0.67 one-off / 0.33 two-off / 0.0  (↑ better)"),
    ("Ord MAE",     "Mean absolute tier distance |predicted tier − actual tier|  (↓ better)"),
    ("Exp MAE",     "MAE of prob-weighted expected tier vs actual; captures calibration quality  (↓ better)"),
    ("QWK",         "Quadratic Weighted Kappa — kappa that penalizes large tier misses quadratically  (↑ better)"),
    ("F1 Macro",    "Unweighted mean of per-class F1 scores; not inflated by majority class  (↑ better)"),
]


def plot_multimodal_model_summary(summary_df, artifact_dir):
    """Render the multimodal model comparison table with a metric legend panel."""
    metric_cols = [
        "accuracy", "within_one_accuracy", "distance_weighted_accuracy",
        "ordinal_mae", "expected_class_mae", "quadratic_weighted_kappa", "f1_macro",
    ]
    display_cols = {
        "model": "Model",
        "accuracy": "Acc",
        "within_one_accuracy": "Within 1",
        "distance_weighted_accuracy": "Dist Wt Acc",
        "ordinal_mae": "Ord MAE",
        "expected_class_mae": "Exp MAE",
        "quadratic_weighted_kappa": "QWK",
        "f1_macro": "F1 Macro",
    }
    sel = ["model"] + [c for c in metric_cols if c in summary_df.columns]
    plot_df = summary_df[sel].copy()
    plot_df["model"] = plot_df["model"].map(lambda m: _MM_SHORT_NAMES.get(m, m))
    plot_df = plot_df.rename(columns=display_cols)

    stacker_rows = {
        i for i, m in enumerate(summary_df["model"])
        if "stacker" in str(m).lower()
    }

    n_rows = len(plot_df)
    fig = plt.figure(figsize=(14, n_rows * 0.6 + 5.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[n_rows + 1, 3.5], hspace=0.45)
    ax_tbl = fig.add_subplot(gs[0])
    ax_leg = fig.add_subplot(gs[1])
    ax_tbl.axis("off")
    ax_leg.axis("off")

    cell_text = [[_format_table_value(v) for v in row] for row in plot_df.to_numpy()]
    table = ax_tbl.table(
        cellText=cell_text,
        colLabels=plot_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.8)

    for col_idx in range(len(plot_df.columns)):
        table[0, col_idx].set_facecolor("#1c2833")
        table[0, col_idx].set_text_props(color="white", fontweight="bold")

    for row_idx in range(n_rows):
        is_stacker = row_idx in stacker_rows
        if is_stacker:
            bg = "#1a5276"
            fg = "white"
            fw = "bold"
        else:
            bg = "#eaf2f8" if row_idx % 2 == 0 else "white"
            fg = "black"
            fw = "normal"
        for col_idx in range(len(plot_df.columns)):
            table[row_idx + 1, col_idx].set_facecolor(bg)
            table[row_idx + 1, col_idx].set_text_props(color=fg, fontweight=fw)

    ax_tbl.set_title(
        "Multimodal Model Comparison\n"
        "(blue row = final stacker output; other rows = individual base model performance)",
        fontsize=12, fontweight="bold", pad=10,
    )

    legend_lines = ["Column definitions:"] + [
        f"  {abbr:<13} {desc}" for abbr, desc in _MM_LEGEND
    ]
    ax_leg.text(
        0.01, 0.97, "\n".join(legend_lines),
        transform=ax_leg.transAxes,
        fontsize=8.5, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8f9fa", edgecolor="#bdc3c7"),
    )

    out_path = _explicit_artifact_path("model_summary.png", artifact_dir)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path}")
    plt.close(fig)


def plot_ordinal_confusion_matrix(confusion_df, artifact_dir):
    """Ordinal confusion matrix: rows = actual tier, columns = predicted tier.

    Each cell shows the raw count and the row percentage (what fraction of
    actual-tier-X prospects were predicted as tier-Y). The diagonal is correct
    predictions; off-diagonal cells are errors. Darker = more predictions.
    """
    tier_display = [t.capitalize() for t in confusion_df.index]
    data = confusion_df.to_numpy(dtype=float)
    row_sums = data.sum(axis=1, keepdims=True)
    row_pct = np.divide(data, row_sums, out=np.zeros_like(data), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(row_pct, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(tier_display)))
    ax.set_xticklabels(tier_display, fontsize=11)
    ax.set_yticks(range(len(tier_display)))
    ax.set_yticklabels(tier_display, fontsize=11)
    ax.set_xlabel("Predicted tier", fontsize=12)
    ax.set_ylabel("Actual tier", fontsize=12)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            color = "white" if row_pct[i, j] > 0.55 else "black"
            ax.text(
                j, i,
                f"{int(data[i, j])}\n({row_pct[i, j]:.0%} of row)",
                ha="center", va="center", fontsize=9, color=color,
            )

    ax.set_title(
        "Confusion Matrix\n"
        "Rows = actual tier  |  Columns = predicted tier\n"
        "Cell: count and % of that actual tier predicted as each column tier",
        fontsize=11, fontweight="bold",
    )
    fig.colorbar(im, ax=ax, label="Row fraction (% of actual tier)", shrink=0.8)
    plt.tight_layout()
    out_path = _explicit_artifact_path("ordinal_confusion_matrix.png", artifact_dir)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path}")
    plt.close(fig)


def plot_ordinal_error_distribution(error_df, artifact_dir):
    """Bar chart of prediction errors, color-coded by severity, vs. random-chance baseline.

    Each bar shows what fraction of prospects were predicted exactly N tiers away
    from their true tier. Green = correct, red = worst possible miss.
    The dashed outline shows what a uniformly random model would produce.
    """
    _COLORS = ["#27ae60", "#f0b429", "#e67e22", "#c0392b"]
    _XLABELS = [
        "0 tiers off\n(correct)",
        "1 tier off\n(near miss)",
        "2 tiers off\n(bad miss)",
        "3 tiers off\n(worst miss)",
    ]
    # Uniform-random baseline: P(|pred-true|=k) over 4 classes, uniform prediction
    _BASELINE = [4/16, 6/16, 4/16, 2/16]

    pcts = error_df["percent"].tolist()
    counts = error_df["count"].tolist()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(pcts))

    bars = ax.bar(x, pcts, color=_COLORS, width=0.55, zorder=3,
                  edgecolor="white", linewidth=0.6)

    for xi, bl in zip(x, _BASELINE):
        ax.bar(xi, bl, width=0.55, color="none", edgecolor="#444",
               linewidth=1.5, linestyle="--", zorder=4)

    ax.bar([], [], color="none", edgecolor="#444", linewidth=1.5,
           linestyle="--", label="Random-chance baseline (uniform prediction)")

    for i, (bar, count, pct) in enumerate(zip(bars, counts, pcts)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            pct + 0.012,
            f"{count} prospects\n{pct:.1%}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(_XLABELS, fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_ylabel("Fraction of test prospects", fontsize=12)
    ax.set_ylim(0, max(pcts) * 1.45)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, axis="y", alpha=0.25, zorder=0)
    ax.set_title(
        "Prediction Error Distribution\n"
        "How many tiers off was the model? (lower error = better)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    out_path = _explicit_artifact_path("ordinal_error_distribution.png", artifact_dir)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path}")
    plt.close(fig)


def plot_expected_vs_true(pred_df, artifact_dir):
    """Strip chart: expected predicted tier vs. true tier, colored by discrete prediction.

    Each dot = one prospect in the test set. The y-axis is the model's probability-weighted
    expected tier (a continuous value between 0 and 3). The x-axis groups prospects by their
    true tier. Dot color = what tier the model actually predicted (argmax of probabilities).
    The black bar = median expected value for each true-tier group.
    Perfect calibration: median expected value should equal the true tier (diagonal).
    """
    _TIER_LABELS = ["Bust", "Bench", "Starter", "Star"]
    _PRED_COLORS = ["#5b6770", "#4c78a8", "#59a14f", "#e15759"]

    true_arr = pred_df["actual_tier"].to_numpy(dtype=int)
    expected = pred_df["expected_class"].to_numpy(dtype=float)
    pred_arr = pred_df["pred_tier"].to_numpy(dtype=int)

    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(9, 6))

    for tier_i in range(4):
        ax.axhspan(tier_i - 0.45, tier_i + 0.45, alpha=0.04, color="grey", zorder=0)

    for pred_tier in range(4):
        mask = pred_arr == pred_tier
        if not mask.any():
            continue
        jitter = rng.uniform(-0.22, 0.22, mask.sum())
        ax.scatter(
            true_arr[mask] + jitter, expected[mask],
            color=_PRED_COLORS[pred_tier], alpha=0.75, s=38,
            edgecolor="white", linewidth=0.4,
            label=f"Predicted: {_TIER_LABELS[pred_tier]}",
            zorder=3,
        )

    for i in range(4):
        mask = true_arr == i
        if mask.any():
            med = np.median(expected[mask])
            ax.hlines(med, i - 0.32, i + 0.32, color="black", linewidth=2.2, zorder=5)

    ax.plot([0, 3], [0, 3], color="#888", linestyle="--", linewidth=1.2,
            alpha=0.7, label="Perfect calibration (expected = actual)", zorder=2)

    for i in range(4):
        mask = true_arr == i
        ax.text(i, 3.35, f"n={mask.sum()}", ha="center", va="bottom",
                fontsize=9, color="#555")

    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(_TIER_LABELS, fontsize=12)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(_TIER_LABELS, fontsize=12)
    ax.set_xlim(-0.55, 3.55)
    ax.set_ylim(-0.35, 3.6)
    ax.set_xlabel("Actual tier", fontsize=12)
    ax.set_ylabel("Model's expected predicted tier\n(probability-weighted average of Bust=0, Bench=1, Starter=2, Star=3)", fontsize=10)
    ax.set_title(
        "Expected Prediction vs. True Tier\n"
        "Dot color = discrete predicted tier (argmax)  |  Black bar = median expected value per group",
        fontsize=12, fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=9, framealpha=0.88, ncols=2)
    ax.grid(True, axis="y", alpha=0.2, zorder=0)
    plt.tight_layout()
    out_path = _explicit_artifact_path("expected_vs_true.png", artifact_dir)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path}")
    plt.close(fig)


def plot_probability_mass_by_true_class(prob_mass_df, artifact_dir):
    """Heatmap: where does the model place its probability mass for each true tier?

    Each cell = average probability assigned to 'predicted tier' (column) across
    all test prospects whose true tier matches the row label.
    Each row sums to 1.00 (shown in rightmost annotation).
    A perfect model would show 1.00 on the diagonal and 0.00 everywhere else.
    Near-uniform rows indicate the model cannot distinguish that tier well.
    """
    col_labels = [c.replace("p_", "").capitalize() for c in prob_mass_df.columns]
    row_labels  = [r.capitalize() for r in prob_mass_df.index]
    data = prob_mass_df.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(data, cmap="YlGnBu", vmin=0.0, vmax=0.65, aspect="auto")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=12)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=12)
    ax.set_xlabel("Predicted tier", fontsize=12)
    ax.set_ylabel("Actual (true) tier", fontsize=12)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            text_color = "white" if val > 0.44 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=text_color)

    # Row sum annotations to right of last column
    n_cols = data.shape[1]
    for i, row in enumerate(data):
        ax.text(n_cols - 0.5 + 0.72, i, f"Σ = {row.sum():.3f}",
                ha="left", va="center", fontsize=9, color="#555", clip_on=False)

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_title(
        "Average Predicted Probability by True Tier\n"
        "Each row shows where the model places probability mass for that true tier.\n"
        "Ideal: 1.000 on the diagonal. Uniform rows = poor discrimination.",
        fontsize=11, fontweight="bold",
    )
    fig.colorbar(im, ax=ax, label="Mean predicted probability", shrink=0.8, pad=0.14)
    plt.tight_layout()
    out_path = _explicit_artifact_path("probability_mass_by_true_class.png", artifact_dir)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path}")
    plt.close(fig)


def plot_worst_misses_probability_bars(worst_df, artifact_dir, max_rows=20):
    """Plot stacked class probabilities for the worst ordinal misses."""
    plot_df = worst_df.head(max_rows).copy()
    if plot_df.empty:
        return

    prob_cols = ["p_bust", "p_bench", "p_starter", "p_star"]
    colors = ["#5b6770", "#4c78a8", "#59a14f", "#e15759"]
    labels = []
    for idx, row in plot_df.iterrows():
        name = str(row.get("Name", f"row {idx}"))
        labels.append(f"{name}\n{row['actual_tier_label']} -> {row['pred_tier_label']}")

    fig, ax = plt.subplots(figsize=(10, max(5, 0.55 * len(plot_df))))
    y = np.arange(len(plot_df))
    left = np.zeros(len(plot_df))
    for col, color in zip(prob_cols, colors):
        values = plot_df[col].to_numpy(dtype=float)
        ax.barh(y, values, left=left, color=color, label=col.replace("p_", ""))
        left += values

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Predicted probability")
    ax.set_title("Worst Misses: Predicted Probability Bars", fontsize=13, fontweight="bold")
    ax.legend(ncols=4, loc="lower center", bbox_to_anchor=(0.5, -0.18))
    plt.tight_layout()
    out_path = _explicit_artifact_path("worst_misses_probability_bars.png", artifact_dir)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path}")
    plt.close(fig)


def plot_best_hits_probability_bars(best_hits: dict, artifact_dir, max_rows=15):
    """Stacked tier probabilities for lottery-bust and late-star correct calls.

    Do not use ``df or default`` with pandas DataFrames (ambiguous truth value).
    """
    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, max(5.0, 0.45 * max_rows + 2.0)),
        constrained_layout=True,
    )
    panels = [
        ("lottery_bust_hits", "Lottery bust hits (pick ≤ 14)\ntrue bust · pred bust"),
        ("late_star_hits", "Late-pick star hits (pick ≥ 31)\ntrue star · pred star"),
    ]
    prob_cols = ["p_bust", "p_bench", "p_starter", "p_star"]
    colors = ["#5b6770", "#4c78a8", "#59a14f", "#e15759"]

    for ax, (key, title) in zip(axes, panels):
        raw = best_hits.get(key)
        if not isinstance(raw, pd.DataFrame) or raw.empty:
            ax.text(
                0.5, 0.5, "No matching rows",
                ha="center", va="center", fontsize=11, transform=ax.transAxes,
            )
            ax.set_axis_off()
            continue

        plot_df = raw.head(max_rows)
        labels = []
        for _, row in plot_df.iterrows():
            name = str(row.get("Name", "?"))
            pk = row.get("draft_pick", "")
            conf = float(row.get("confidence", 0.0))
            labels.append(f"{name}  (#{pk})\nconf={conf:.2f}")

        y = np.arange(len(plot_df))
        left = np.zeros(len(plot_df))
        for col, color in zip(prob_cols, colors):
            vals = plot_df[col].to_numpy(dtype=float)
            ax.barh(y, vals, left=left, color=color, label=col.replace("p_", ""))
            left += vals

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("Predicted probability")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(ncols=4, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12))

    out_path = _explicit_artifact_path("best_hits_probability_bars.png", artifact_dir)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path}")
    plt.close(fig)


def plot_stacker_contribution_heatmap(contribution_df, artifact_dir):
    """Row-normalized heatmap of stacker coefficient weight per base model and predicted tier.

    The stacker (a logistic regression) learns a weight for each base model's probability
    outputs. This plot shows what share of each base model's total coefficient magnitude
    goes toward predicting each tier. Values are row-normalized (each row sums to ~100%).
    A base model with high weight for 'Star' means the stacker relies on it most when
    deciding whether a prospect is a Star.
    """
    display_index = [_MM_SHORT_NAMES.get(idx, idx) for idx in contribution_df.index]
    col_labels = [c.capitalize() for c in contribution_df.columns]
    data = contribution_df.to_numpy(dtype=float)

    row_totals = data.sum(axis=1, keepdims=True)
    row_totals = np.where(row_totals == 0, 1.0, row_totals)
    data_norm = data / row_totals

    fig, ax = plt.subplots(figsize=(8, max(4, 0.75 * len(contribution_df))))
    im = ax.imshow(data_norm, aspect="auto", cmap="PuBuGn", vmin=0.0, vmax=0.5)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=12)
    ax.set_yticks(range(len(display_index)))
    ax.set_yticklabels(display_index, fontsize=10)
    ax.set_xlabel("Predicted tier (stacker output)", fontsize=12)
    ax.set_ylabel("Base model", fontsize=12)

    for i in range(data_norm.shape[0]):
        for j in range(data_norm.shape[1]):
            val = data_norm[i, j]
            text_color = "white" if val > 0.32 else "black"
            ax.text(j, i, f"{val:.1%}", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=text_color)

    ax.set_title(
        "Stacker Coefficient Weight by Base Model and Predicted Tier\n"
        "Each cell = share of this model's stacker weight allocated to predicting that tier.\n"
        "Row-normalized: each row sums to 100%.",
        fontsize=11, fontweight="bold",
    )
    fig.colorbar(im, ax=ax, label="Row-normalized coefficient weight", shrink=0.8,
                 format=plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    plt.tight_layout()
    out_path = _explicit_artifact_path("stacker_contribution_heatmap.png", artifact_dir)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path}")
    plt.close(fig)
