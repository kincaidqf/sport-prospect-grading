"""Reporting helpers for multimodal ordinal classification outputs."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.models.probability import PROBA_COLUMNS, TIER_CLASS_NAMES
from src.models.text_model import attach_scouting_text_columns
from src.training.evaluate import ordinal_classification_metrics
from src.utils.plotting import (
    plot_best_hits_probability_bars,
    plot_expected_vs_true,
    plot_multimodal_model_summary,
    plot_ordinal_confusion_matrix,
    plot_ordinal_error_distribution,
    plot_probability_mass_by_true_class,
    plot_stacker_contribution_heatmap,
    plot_worst_misses_probability_bars,
)


_METRIC_LEGEND = [
    ("accuracy",                  "Accuracy",                  "Fraction of prospects predicted to the exact correct tier",                           "correct / total",                               "higher"),
    ("balanced_accuracy",         "Balanced Accuracy",         "Per-class recall averaged equally — not inflated by majority class",                  "mean(recall per class)",                        "higher"),
    ("auc",                       "AUC-OvR",                   "Area Under ROC Curve; macro average over tiers using one-vs-rest",                    "macro OvR AUC",                                 "higher"),
    ("f1_macro",                  "F1 Macro",                  "Unweighted mean of per-class F1; treats all tiers equally",                           "mean(2·P·R / (P+R) per class)",                 "higher"),
    ("top1_accuracy",             "Top-1 Accuracy",            "Alias for accuracy",                                                                  "correct / total",                               "higher"),
    ("within_one_accuracy",       "Within-One Accuracy",       "Fraction where |predicted − actual| ≤ 1 tier",                                        "mean(|pred - true| ≤ 1)",                       "higher"),
    ("ordinal_mae",               "Ordinal MAE",               "Mean absolute tier distance between predicted and actual tier",                        "mean |pred − true|",                            "lower"),
    ("distance_weighted_accuracy","Distance-Weighted Accuracy","Partial credit: 1.0 exact / 0.67 one-off / 0.33 two-off / 0.0 three-off",             "mean(1 − |pred − true| / 3)",                   "higher"),
    ("expected_class_mae",        "Expected-Class MAE",        "MAE of prob-weighted expected predicted tier vs. actual; captures calibration quality","mean |Σ(p_i · i) − true|",                      "lower"),
    ("quadratic_weighted_kappa",  "Quadratic Weighted Kappa",  "Cohen's kappa penalizing large ordinal mistakes quadratically",                        "quadratic weighted kappa formula",               "higher"),
    ("precision_{tier}",          "Precision for tier",        "Of all prospects predicted as {tier}: fraction who were actually {tier}",              "TP / (TP + FP)",                                "higher"),
    ("recall_{tier}",             "Recall for tier",           "Of all actual {tier} prospects: fraction correctly identified",                        "TP / (TP + FN)",                                "higher"),
    ("f1_{tier}",                 "F1 for tier",               "Harmonic mean of precision and recall for {tier}",                                    "2 · precision · recall / (precision + recall)", "higher"),
]


TIER_LABELS = list(range(len(TIER_CLASS_NAMES)))


def _tier_name(value: int) -> str:
    if 0 <= int(value) < len(TIER_CLASS_NAMES):
        return TIER_CLASS_NAMES[int(value)]
    return str(value)


def build_enriched_predictions(
    test_df: pd.DataFrame,
    y_test,
    y_pred,
    test_proba: np.ndarray,
) -> pd.DataFrame:
    """Build the saved test prediction table with ordinal diagnostics."""
    y_true_arr = np.asarray(y_test, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    proba_arr = np.asarray(test_proba, dtype=float)

    id_cols = [c for c in ("Name", "draft_year", "draft_pick") if c in test_df.columns]
    pred_df = test_df[id_cols].copy() if id_cols else pd.DataFrame(index=test_df.index)
    pred_df["actual_tier"] = y_true_arr
    pred_df["pred_tier"] = y_pred_arr
    pred_df["actual_tier_label"] = [_tier_name(v) for v in y_true_arr]
    pred_df["pred_tier_label"] = [_tier_name(v) for v in y_pred_arr]
    pred_df["confidence"] = proba_arr.max(axis=1)

    for i, col in enumerate(PROBA_COLUMNS):
        pred_df[col] = proba_arr[:, i]

    ordinal_error = np.abs(y_pred_arr - y_true_arr)
    expected_class = proba_arr @ np.arange(proba_arr.shape[1], dtype=float)
    pred_df["ordinal_error"] = ordinal_error
    pred_df["within_one"] = ordinal_error <= 1
    pred_df["distance_weighted_credit"] = 1.0 - (ordinal_error / max(proba_arr.shape[1] - 1, 1))
    pred_df["expected_class"] = expected_class
    pred_df["expected_class_error"] = np.abs(expected_class - y_true_arr)
    return pred_df


def _summary_row(model_name: str, task: str, y_true, y_pred, y_prob) -> dict:
    metrics = ordinal_classification_metrics(
        y_true,
        y_pred,
        y_prob,
        class_names=TIER_CLASS_NAMES,
    )
    return {"model": model_name, "task": task, **metrics}


def build_model_summary(model, test_df: pd.DataFrame, y_test, y_pred, test_proba: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build final stacker and base-model summary tables."""
    rows = [_summary_row("multimodal__stacker", "multimodal", y_test, y_pred, test_proba)]
    base_rows = []

    comp = ((getattr(model, "cfg", None) or {}).get("model") or {}).get("composite_score")
    test_in = (
        attach_scouting_text_columns(test_df, comp)
        if getattr(model, "final_text_bundles", None)
        else test_df
    )

    for key in model.clf_models:
        bundle = model.final_clf_bundles[key]
        proba = bundle.predict_tier_proba(test_in)
        y_hat = np.argmax(proba.values, axis=1)
        base_rows.append(_summary_row(f"classification__{key}", "classification", y_test, y_hat, proba.values))

    for key in model.reg_models:
        bundle = model.final_reg_bundles[key]
        proba = bundle.predict_tier_proba(test_in)
        y_hat = np.argmax(proba.values, axis=1)
        base_rows.append(_summary_row(f"regression__{key}", "regression", y_test, y_hat, proba.values))

    for key in getattr(model, "text_models", []):
        bundle = model.final_text_bundles[key]
        proba = bundle.predict_tier_proba(test_in)
        y_hat = np.argmax(proba.values, axis=1)
        base_rows.append(_summary_row(f"text__{key}", "text", y_test, y_hat, proba.values))

    return pd.DataFrame(rows + base_rows), pd.DataFrame(base_rows)


def build_ordinal_error_distribution(pred_df: pd.DataFrame) -> pd.DataFrame:
    counts = pred_df["ordinal_error"].value_counts().reindex(TIER_LABELS, fill_value=0).sort_index()
    total = max(len(pred_df), 1)
    return pd.DataFrame({
        "ordinal_error": counts.index.astype(int),
        "count": counts.values.astype(int),
        "percent": counts.values / total,
    })


def build_confusion_table(y_test, y_pred) -> pd.DataFrame:
    matrix = confusion_matrix(y_test, y_pred, labels=TIER_LABELS)
    return pd.DataFrame(matrix, index=TIER_CLASS_NAMES, columns=TIER_CLASS_NAMES)


def build_best_hits_tables(pred_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Lottery bust hits (early pick, true & pred bust) and late steal stars (late pick, true & pred star)."""
    empty = {"lottery_bust_hits": pd.DataFrame(), "late_star_hits": pd.DataFrame()}
    if "draft_pick" not in pred_df.columns:
        return empty
    
    LOTTERY_MAX_PICK = 14
    LATE_STAR_MIN_PICK = 31

    dp = pd.to_numeric(pred_df["draft_pick"], errors="coerce")
    has_pick = dp.notna()
    if not has_pick.any():
        return empty

    tier_bust, tier_bench, tier_starter, tier_star = 0, 1, 2, 3
    lottery_bust = pred_df.loc[
        has_pick
        & (dp <= LOTTERY_MAX_PICK)
        & ((pred_df["actual_tier"] == tier_bust) | (pred_df["actual_tier"] == tier_bench))
        & ((pred_df["pred_tier"] == tier_bust) | (pred_df["pred_tier"] == tier_bench))
    ].copy()
    lottery_bust["_rk"] = pd.to_numeric(lottery_bust["draft_pick"], errors="coerce")
    lottery_bust = lottery_bust.sort_values(
        ["_rk", "confidence"], ascending=[True, False]
    ).drop(columns=["_rk"], errors="ignore")

    late_star = pred_df.loc[
        has_pick
        & (dp >= LATE_STAR_MIN_PICK)
        & ((pred_df["actual_tier"] == tier_starter) | (pred_df["actual_tier"] == tier_star))
        & ((pred_df["pred_tier"] == tier_starter) | (pred_df["pred_tier"] == tier_star))
    ].copy()
    late_star["_rk"] = pd.to_numeric(late_star["draft_pick"], errors="coerce")
    late_star = late_star.sort_values(
        ["confidence", "_rk"], ascending=[False, False]
    ).drop(columns=["_rk"], errors="ignore")

    return {"lottery_bust_hits": lottery_bust, "late_star_hits": late_star}


def build_probability_mass_by_true_class(pred_df: pd.DataFrame) -> pd.DataFrame:
    prob_mass = pred_df.groupby("actual_tier_label")[PROBA_COLUMNS].mean()
    return prob_mass.reindex(TIER_CLASS_NAMES).fillna(0.0)


def build_stacker_contributions(model) -> pd.DataFrame:
    """Aggregate absolute stacker coefficients by base model and output class."""
    coef = np.asarray(model.stacker.coef_, dtype=float)
    classes = list(model.stacker.classes_)

    base_models = [f"classification__{key}" for key in model.clf_models]
    base_models += [f"regression__{key}" for key in model.reg_models]
    base_models += [f"text__{key}" for key in getattr(model, "text_models", [])]
    out = pd.DataFrame(0.0, index=base_models, columns=TIER_CLASS_NAMES)

    for class_pos, cls in enumerate(classes):
        class_name = _tier_name(int(cls))
        for base_name in base_models:
            feature_idx = [
                idx for idx, col in enumerate(model.meta_cols)
                if col.startswith(f"{base_name}__")
            ]
            out.loc[base_name, class_name] = float(np.abs(coef[class_pos, feature_idx]).sum())

    return out


def _round_floats(df: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    out = df.copy()
    float_cols = out.select_dtypes(include="float").columns
    out[float_cols] = out[float_cols].round(decimals)
    return out


def write_multimodal_report(
    model,
    test_df: pd.DataFrame,
    y_test,
    y_pred,
    test_proba: np.ndarray,
    out_dir: str,
) -> dict[str, pd.DataFrame]:
    """Write multimodal tables and plots to the supplied run output directory."""
    os.makedirs(out_dir, exist_ok=True)

    pred_df = build_enriched_predictions(test_df, y_test, y_pred, test_proba)
    model_summary, base_model_summary = build_model_summary(model, test_df, y_test, y_pred, test_proba)
    error_distribution = build_ordinal_error_distribution(pred_df)
    confusion_df = build_confusion_table(y_test, y_pred)
    probability_mass = build_probability_mass_by_true_class(pred_df)
    worst_misses = pred_df.sort_values(
        ["ordinal_error", "expected_class_error", "confidence"],
        ascending=[False, False, False],
    )
    stacker_contributions = build_stacker_contributions(model)

    best_hits = build_best_hits_tables(pred_df)

    _round_floats(pred_df).to_csv(os.path.join(out_dir, "test_predictions.csv"), index=False)
    model.meta_features(test_df).round(4).to_csv(os.path.join(out_dir, "test_meta_features.csv"), index=True)
    _round_floats(base_model_summary).to_csv(os.path.join(out_dir, "base_model_summary.csv"), index=False)
    _round_floats(model_summary).to_csv(os.path.join(out_dir, "model_summary.csv"), index=False)
    _round_floats(worst_misses).to_csv(os.path.join(out_dir, "worst_misses.csv"), index=False)
    for bh_name, bh_df in best_hits.items():
        path = os.path.join(out_dir, f"{bh_name}.csv")
        if bh_df.empty:
            pd.DataFrame(
                [{"note": "No rows matched (need draft_pick; adjust model.multimodal.best_hits thresholds)."}],
            ).to_csv(path, index=False)
        else:
            _round_floats(bh_df).to_csv(path, index=False)
        print(f"[multimodal] best hits: {bh_name} → {len(bh_df)} rows → {path}")
    error_distribution.to_csv(os.path.join(out_dir, "ordinal_error_distribution.csv"), index=False)
    confusion_df.to_csv(os.path.join(out_dir, "confusion_matrix.csv"), index=True)
    _round_floats(probability_mass, decimals=4).to_csv(
        os.path.join(out_dir, "probability_mass_by_true_class.csv"), index=True
    )
    _round_floats(stacker_contributions, decimals=4).to_csv(
        os.path.join(out_dir, "stacker_contributions.csv"), index=True
    )

    legend_df = pd.DataFrame(
        _METRIC_LEGEND,
        columns=["metric", "full_name", "description", "formula", "direction"],
    )
    legend_df.to_csv(os.path.join(out_dir, "metric_legend.csv"), index=False)

    plot_multimodal_model_summary(model_summary, artifact_dir=out_dir)
    plot_ordinal_confusion_matrix(confusion_df, artifact_dir=out_dir)
    plot_ordinal_error_distribution(error_distribution, artifact_dir=out_dir)
    plot_expected_vs_true(pred_df, artifact_dir=out_dir)
    plot_probability_mass_by_true_class(probability_mass, artifact_dir=out_dir)
    plot_worst_misses_probability_bars(worst_misses, artifact_dir=out_dir, max_rows=20)
    plot_best_hits_probability_bars(best_hits, artifact_dir=out_dir, max_rows=15)
    plot_stacker_contribution_heatmap(stacker_contributions, artifact_dir=out_dir)

    return {
        "test_predictions": pred_df,
        "model_summary": model_summary,
        "base_model_summary": base_model_summary,
        "ordinal_error_distribution": error_distribution,
        "confusion_matrix": confusion_df,
        "probability_mass_by_true_class": probability_mass,
        "worst_misses": worst_misses,
        "best_hits": best_hits,
        "stacker_contributions": stacker_contributions,
    }
