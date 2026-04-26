"""Shared feature-name and feature-importance helpers for tabular models."""
from __future__ import annotations

import numpy as np
import pandas as pd
import mlflow


def get_all_feature_names(pipe, numeric_cols, categorical_cols, ordinal_cols):
    """Resolve expanded feature names after preprocessing."""
    preprocessor = pipe.named_steps["preprocessor"]
    cat_names = list(
        preprocessor.named_transformers_["cat"]
        .named_steps["onehot"]
        .get_feature_names_out(categorical_cols)
    )
    return list(numeric_cols) + cat_names + list(ordinal_cols)


def get_coef_df(pipe, numeric_cols, categorical_cols, ordinal_cols, step_name):
    """Return coefficients for a fitted linear model pipeline step.

    For multi-class models (coef_ shape [n_classes, n_features]) the mean
    absolute coefficient across classes is used so importance is still
    a single non-negative value per feature.
    """
    model = pipe.named_steps[step_name]
    all_names = get_all_feature_names(pipe, numeric_cols, categorical_cols, ordinal_cols)
    coefs = model.coef_
    if coefs.ndim == 2 and coefs.shape[0] > 1:
        coefficients = np.abs(coefs).mean(axis=0)
    else:
        coefficients = coefs.ravel()
    coef_df = pd.DataFrame({"feature": all_names, "coefficient": coefficients})
    coef_df["abs_coef"] = coef_df["coefficient"].abs()
    return coef_df.sort_values("abs_coef", ascending=False)


def get_lasso_coef_df(pipe, numeric_cols, categorical_cols, ordinal_cols):
    """Return only the non-zero Lasso coefficients."""
    coef_df = get_coef_df(pipe, numeric_cols, categorical_cols, ordinal_cols, "lasso")
    return coef_df[coef_df["coefficient"] != 0].copy()


def print_lasso_coefficients(pipe, numeric_cols, categorical_cols, ordinal_cols):
    """Print the retained Lasso coefficients."""
    all_names = get_all_feature_names(pipe, numeric_cols, categorical_cols, ordinal_cols)
    coef_df = get_lasso_coef_df(pipe, numeric_cols, categorical_cols, ordinal_cols)
    print(f"  {len(coef_df)} / {len(all_names)} features retained\n")
    for _, row in coef_df.iterrows():
        sign = "+" if row["coefficient"] > 0 else "-"
        print(f"  {sign} {row['feature']:<20}  coef = {row['coefficient']:+.4f}")


def get_xgb_importance_df(pipe, numeric_cols, categorical_cols, ordinal_cols):
    """Return fitted XGBoost feature importances."""
    xgb = pipe.named_steps["xgb"]
    all_names = get_all_feature_names(pipe, numeric_cols, categorical_cols, ordinal_cols)
    return pd.DataFrame({"feature": all_names, "importance": xgb.feature_importances_}).sort_values(
        "importance",
        ascending=False,
    )


def log_xgb_importances(pipe, numeric_cols, categorical_cols, ordinal_cols, top_n=15):
    """Log top XGBoost feature importances to MLflow."""
    importance_df = get_xgb_importance_df(pipe, numeric_cols, categorical_cols, ordinal_cols)
    for _, row in importance_df.head(top_n).iterrows():
        safe = row["feature"].replace(" ", "_").replace("%", "pct").replace("-", "_")
        mlflow.log_metric(f"imp_{safe}", row["importance"])


def print_xgb_importances(pipe, numeric_cols, categorical_cols, ordinal_cols, top_n=10):
    """Print the top XGBoost importances."""
    importance_df = get_xgb_importance_df(pipe, numeric_cols, categorical_cols, ordinal_cols)
    for _, row in importance_df.head(top_n).iterrows():
        bar = "#" * int(row["importance"] * 50)
        print(f"  {row['feature']:<20}  {row['importance']:.4f}  {bar}")
