"""Regression trainer for NBA draft prospect outcomes from NCAA stats."""
from __future__ import annotations

import os
from contextlib import nullcontext

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
# XGBRegressor is imported lazily inside train_selected_regression_models to prevent macOS
# OpenMP libomp conflicts at module initialization time.

from src.data.loader import (
    PROJECT_ROOT,
    PROSPECT_CONTEXT_MODE,
    RANDOM_STATE,
    TARGET_COL,
    TEST_SIZE,
    _assign_tier_thresholded,
    build_feature_matrix,
    load_data,
)
from src.models.probability import BaseModelBundle, TIER_THRESHOLDS
from src.training.evaluate import regression_metrics
from src.utils.features import (
    get_lasso_coef_df,
    log_xgb_importances,
    print_lasso_coefficients,
    print_xgb_importances,
)
from src.utils.mlflow_utils import (
    build_mlflow_context,
    log_candidate_summary,
    log_common_params,
    log_config_dict,
    log_data_summary,
    log_reproducibility_metadata,
    managed_run,
)
from src.utils.plotting import plot_feature_importance, plot_model_summary, save_and_log


MLFLOW_EXPERIMENT = "nba-draft-prospect-regression"
REGRESSION_TARGETS = {"plus_minus", "composite_score", "nba_role_zscore"}
TARGET_MODE = "nba_role_zscore"
USE_DRAFT_PICK = False
ARTIFACT_DIR = os.path.join(PROJECT_ROOT, "outputs", "plots")

REGRESSION_MODEL_REGISTRY: list[str] = ["lasso", "ridge", "xgboost"]
_DISPLAY_NAMES = {"lasso": "Lasso", "ridge": "Ridge", "xgboost": "XGBoost"}


# ── XGBoost fixed-param helpers ────────────────────────────────────────────────

def _first(x):
    return x[0] if isinstance(x, list) else x


def _get_oof_xgb_params_regression(cfg: dict) -> dict:
    """Return the single XGBoost parameter set used for OOF fold training.

    Checks model.multimodal.xgboost.oof_params first (set when multimodal
    pre-tuning is enabled). Falls back to the first value of each grid list
    in the regression xgboost config.
    """
    multimodal_xgb = (
        ((cfg or {}).get("model", {}) or {})
        .get("multimodal", {})
        .get("xgboost", {}) or {}
    )
    oof = multimodal_xgb.get("oof_params")
    if oof:
        return dict(oof)

    xgb_cfg = (
        ((cfg or {}).get("model", {}) or {})
        .get("regression", {})
        .get("xgboost", {}) or {}
    )
    return {
        "n_estimators":     _first(xgb_cfg.get("n_estimators",     [200])),
        "max_depth":        _first(xgb_cfg.get("max_depth",         [3])),
        "learning_rate":    _first(xgb_cfg.get("learning_rate",     [0.05])),
        "subsample":        _first(xgb_cfg.get("subsample",         [0.8])),
        "colsample_bytree": _first(xgb_cfg.get("colsample_bytree",  [1.0])),
        "min_child_weight": _first(xgb_cfg.get("min_child_weight",  [3])),
        "reg_alpha":        _first(xgb_cfg.get("reg_alpha",         [0.0])),
        "reg_lambda":       _first(xgb_cfg.get("reg_lambda",        [1.0])),
        "gamma":            _first(xgb_cfg.get("gamma",             [0.0])),
    }


# ── Shared training entry point ────────────────────────────────────────────────

def train_selected_regression_models(
    train_df: pd.DataFrame,
    target_col: str,
    cfg: dict,
    selected_models: list[str] | None = None,
    use_fixed_xgb_params: bool = False,
    mlflow_ctx=None,
    return_bundles: bool = True,
) -> dict[str, BaseModelBundle]:
    """Train selected regression models and return a BaseModelBundle per model.

    No calibration split is needed — tier probabilities are derived entirely via
    the Gaussian CDF algorithm using training residual_std stored in the bundle.
    Calling predict_tier_proba() on a bundle whose target is not nba_role_zscore
    will raise (residual_std is set to 0.0 as a guard).

    When mlflow_ctx is provided, a nested MLflow run is opened per model for
    training params and model artifact logging.  When mlflow_ctx is None (OOF
    fold use), no MLflow runs are created.

    Parameters
    ----------
    train_df             : training rows
    target_col           : continuous regression target column in train_df
    cfg                  : full project config dict
    selected_models      : subset of REGRESSION_MODEL_REGISTRY; None → all models
    use_fixed_xgb_params : skip GridSearchCV for XGBoost; use OOF fixed params
    mlflow_ctx           : MLflowContext for nested run creation; None → no logging
    return_bundles       : always True; kept for interface symmetry with classification
    """
    from xgboost import XGBRegressor  # noqa: PLC0415

    model_cfg = (cfg or {}).get("model", {}) or {}
    reg_cfg   = model_cfg.get("regression", {}) or {}
    xgb_cfg   = reg_cfg.get("xgboost") or {}

    prospect_context_mode    = model_cfg.get("prospect_context_mode", PROSPECT_CONTEXT_MODE)
    use_engineered_features  = bool(reg_cfg.get("use_engineered_features", False))
    use_pos_categorical      = bool(reg_cfg.get("use_pos_categorical", False))
    input_normalization_mode = model_cfg.get("input_normalization_mode", "global")
    use_draft_pick           = bool(reg_cfg.get("use_draft_pick", USE_DRAFT_PICK))

    alpha_min       = float(reg_cfg.get("alpha_min",   1e-4))
    alpha_max       = float(reg_cfg.get("alpha_max",   1e2))
    alpha_steps     = int(reg_cfg.get("alpha_steps",   100))
    linear_cv_folds = int(reg_cfg.get("cv_folds",      5))
    alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), alpha_steps)

    cv_folds     = xgb_cfg.get("cv_folds", 5)
    xgb_n_jobs   = xgb_cfg.get("n_jobs", 1)
    grid_n_jobs  = xgb_cfg.get("grid_n_jobs", 1)
    pre_dispatch = xgb_cfg.get("pre_dispatch", 1)

    preprocessor, numeric_cols, categorical_cols, ordinal_cols, passthrough_cols = build_feature_matrix(
        train_df,
        use_draft_pick=use_draft_pick,
        prospect_context_mode=prospect_context_mode,
        use_engineered_features=use_engineered_features,
        use_pos_categorical=use_pos_categorical,
        input_normalization_mode=input_normalization_mode,
    )
    feature_cols = numeric_cols + categorical_cols + ordinal_cols + passthrough_cols

    models_to_train = list(selected_models) if selected_models is not None else list(REGRESSION_MODEL_REGISTRY)
    unknown = [k for k in models_to_train if k not in REGRESSION_MODEL_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown regression model key(s): {unknown}. Valid: {REGRESSION_MODEL_REGISTRY}")

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    # Tier-probability output is only valid when target is nba_role_zscore.
    # For any other target, residual_std=0.0 ensures predict_tier_proba() raises.
    prob_target_mode = reg_cfg.get("probability_target_mode", "nba_role_zscore")
    zscore_col = TARGET_COL.get(prob_target_mode, "")
    is_zscore_target = (target_col == zscore_col)

    bundles: dict[str, BaseModelBundle] = {}

    for key in models_to_train:
        run_mgr = (
            managed_run(
                mlflow_ctx,
                run_name=f"{mlflow_ctx.parent_run_name}__{key}",
                nested=True,
                tags={"estimator": key},
            )
            if mlflow_ctx else nullcontext()
        )
        with run_mgr:
            if key == "lasso":
                model = LassoCV(alphas=alphas, cv=linear_cv_folds, max_iter=10000, random_state=RANDOM_STATE)
                pipe = Pipeline([("preprocessor", clone(preprocessor)), ("lasso", model)])
                pipe.fit(X_train, y_train)
                if mlflow_ctx:
                    log_common_params({
                        "model": key, "target": target_col,
                        "alpha": model.alpha_,
                        "alpha_search_min": alphas[0], "alpha_search_max": alphas[-1],
                        "alpha_search_n": len(alphas), "cv_folds": linear_cv_folds,
                        "random_seed": RANDOM_STATE, "n_train": len(X_train),
                        "use_draft_pick": use_draft_pick,
                        "use_fixed_xgb_params": use_fixed_xgb_params,
                    })
                    coef_df = get_lasso_coef_df(pipe, numeric_cols, categorical_cols, ordinal_cols)
                    mlflow.log_metric("n_nonzero_features", len(coef_df))
                    mlflow.sklearn.log_model(pipe, artifact_path=key)

            elif key == "ridge":
                model = RidgeCV(alphas=alphas, cv=linear_cv_folds)
                pipe = Pipeline([("preprocessor", clone(preprocessor)), ("ridge", model)])
                pipe.fit(X_train, y_train)
                if mlflow_ctx:
                    log_common_params({
                        "model": key, "target": target_col,
                        "alpha": model.alpha_,
                        "alpha_search_min": alphas[0], "alpha_search_max": alphas[-1],
                        "alpha_search_n": len(alphas), "cv_folds": linear_cv_folds,
                        "random_seed": RANDOM_STATE, "n_train": len(X_train),
                        "use_draft_pick": use_draft_pick,
                        "use_fixed_xgb_params": use_fixed_xgb_params,
                    })
                    mlflow.sklearn.log_model(pipe, artifact_path=key)

            elif key == "xgboost":
                if use_fixed_xgb_params:
                    fixed = _get_oof_xgb_params_regression(cfg)
                    pipe = Pipeline([
                        ("preprocessor", clone(preprocessor)),
                        ("xgb", XGBRegressor(
                            random_state=RANDOM_STATE, n_jobs=xgb_n_jobs, verbosity=0,
                            **fixed,
                        )),
                    ])
                    pipe.fit(X_train, y_train)
                    if mlflow_ctx:
                        log_common_params({
                            "model": key, "target": target_col,
                            "use_fixed_xgb_params": True, "n_train": len(X_train),
                            **fixed,
                        })
                        mlflow.sklearn.log_model(pipe, artifact_path=key)
                else:
                    param_grid = {
                        "xgb__n_estimators":     xgb_cfg.get("n_estimators",     [100, 200]),
                        "xgb__max_depth":        xgb_cfg.get("max_depth",        [2, 3]),
                        "xgb__learning_rate":    xgb_cfg.get("learning_rate",    [0.05, 0.1]),
                        "xgb__subsample":        xgb_cfg.get("subsample",        [0.7, 0.9]),
                        "xgb__colsample_bytree": xgb_cfg.get("colsample_bytree", [0.7, 1.0]),
                        "xgb__min_child_weight": xgb_cfg.get("min_child_weight", [3, 5, 10]),
                        "xgb__reg_alpha":        xgb_cfg.get("reg_alpha",        [0, 0.1, 1]),
                        "xgb__reg_lambda":       xgb_cfg.get("reg_lambda",       [1, 5, 10]),
                        "xgb__gamma":            xgb_cfg.get("gamma",            [0]),
                    }
                    xgb_base = Pipeline([
                        ("preprocessor", clone(preprocessor)),
                        ("xgb", XGBRegressor(
                            random_state=RANDOM_STATE, n_jobs=xgb_n_jobs, verbosity=0,
                        )),
                    ])
                    gs = GridSearchCV(
                        xgb_base, param_grid, cv=cv_folds,
                        scoring="r2", n_jobs=grid_n_jobs, pre_dispatch=pre_dispatch,
                    )
                    gs.fit(X_train, y_train)
                    pipe = gs.best_estimator_
                    if mlflow_ctx:
                        log_common_params({
                            "model": key, "target": target_col,
                            "cv_folds": cv_folds, "use_fixed_xgb_params": False,
                            "n_train": len(X_train),
                            **{k.replace("xgb__", ""): v for k, v in gs.best_params_.items()},
                            "best_cv_score": round(float(gs.best_score_), 4),
                            **{f"grid_{k.replace('xgb__','')}": str(v) for k, v in param_grid.items()},
                        })
                        log_xgb_importances(pipe, numeric_cols, categorical_cols, ordinal_cols)
                        mlflow.sklearn.log_model(pipe, artifact_path=key)

                    print(f"\n{'='*40}\n  XGBoost Feature Importances (top 10)\n{'='*40}")
                    print_xgb_importances(pipe, numeric_cols, categorical_cols, ordinal_cols)

            # Gaussian CDF conversion requires training residuals; only valid for zscore target
            if is_zscore_target:
                y_pred_train = pipe.predict(X_train)
                residual_std = float(np.std(np.asarray(y_train) - y_pred_train))
            else:
                residual_std = 0.0  # guard: predict_tier_proba() will raise

            bundles[key] = BaseModelBundle(
                name=key,
                task="regression",
                estimator=pipe,
                feature_cols=feature_cols,
                proba_estimator=None,
                residual_std=residual_std,
                thresholds=TIER_THRESHOLDS,
            )

    return bundles


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_results(results, y_test, col_info, target_mode=TARGET_MODE, plot_dir=ARTIFACT_DIR):
    _plot_regression(results, y_test, target_mode, plot_dir=plot_dir)
    plot_feature_importance(results, col_info, target_mode, artifact_dir=plot_dir)
    plot_model_summary(results, target_mode, "regression", artifact_dir=plot_dir)


_TIER_LABELS = ["Bust", "Bench", "Starter", "Star"]


def _plot_regression(results, y_test, target_mode, plot_dir):
    n = len(results)
    show_tiers = target_mode == "nba_role_zscore"
    n_rows = 3 if show_tiers else 2
    fig, axes = plt.subplots(n_rows, n, figsize=(6 * n, 5 * n_rows))
    fig.suptitle(f"NBA {target_mode} Prediction from College Stats", fontsize=14, fontweight="bold")

    actual_tiers = _assign_tier_thresholded(y_test).value_counts().sort_index() if show_tiers else None

    for col_idx, (name, res) in enumerate(results.items()):
        y_pred = res["y_pred"]

        ax = axes[0][col_idx]
        ax.scatter(y_test, y_pred, alpha=0.5, s=30, edgecolors="none")
        lim = max(abs(y_test.min()), abs(y_test.max()), abs(np.min(y_pred)), abs(np.max(y_pred))) + 1
        ax.plot([-lim, lim], [-lim, lim], "r--", linewidth=1, label="Perfect")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{name}  (R²={res['r2']:.3f})")
        ax.legend(fontsize=8)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

        ax2 = axes[1][col_idx]
        residuals = y_test - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.5, s=30, edgecolors="none")
        ax2.axhline(0, color="r", linestyle="--", linewidth=1)
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("Residual (Actual - Predicted)")
        ax2.set_title(f"{name} Residuals")

        if show_tiers:
            pred_series = pd.Series(y_pred, index=y_test.index)
            pred_tiers = _assign_tier_thresholded(pred_series).value_counts().sort_index()
            ax3 = axes[2][col_idx]
            x = np.arange(4)
            w = 0.35
            ax3.bar(x - w / 2, [actual_tiers.get(i, 0) for i in range(4)], width=w, label="Actual", color="steelblue")
            ax3.bar(x + w / 2, [pred_tiers.get(i, 0) for i in range(4)], width=w, label="Predicted", color="darkorange")
            ax3.set_xticks(x)
            ax3.set_xticklabels(_TIER_LABELS)
            ax3.set_title(f"{name} Tier Distribution")
            ax3.set_ylabel("Count")
            ax3.legend(fontsize=8)

    plt.tight_layout()
    save_and_log(
        fig,
        "regression_results.png",
        {name: {"r2": res["r2"], "rmse": res["rmse"], "mae": res["mae"]} for name, res in results.items()},
        artifact_dir=plot_dir,
    )


# ── Entry point ────────────────────────────────────────────────────────────────

def run(target_mode=TARGET_MODE, use_draft_pick=USE_DRAFT_PICK, df=None, cfg=None, run_name=None, tracking_uri=None):
    model_cfg                = (cfg or {}).get("model", {}) or {}
    composite_cfg            = model_cfg.get("composite_score") or {}
    reg_cfg                  = model_cfg.get("regression", {}) or {}
    target_score_mode        = (model_cfg.get("nba_role_score") or {}).get("target_score_mode", "global")
    prospect_context_mode    = model_cfg.get("prospect_context_mode", PROSPECT_CONTEXT_MODE)
    use_engineered_features  = bool(reg_cfg.get("use_engineered_features", False))
    use_pos_categorical      = bool(reg_cfg.get("use_pos_categorical", False))
    input_normalization_mode = model_cfg.get("input_normalization_mode", "global")
    use_draft_pick           = bool(reg_cfg.get("use_draft_pick", USE_DRAFT_PICK))
    selected_models          = reg_cfg.get("selected_models") or list(REGRESSION_MODEL_REGISTRY)

    df = load_data(composite_cfg=composite_cfg) if df is None else df
    target_col = TARGET_COL[target_mode]

    # Build col_info for plotting (build_feature_matrix called again inside train_selected_regression_models)
    _, numeric_cols, categorical_cols, ordinal_cols, passthrough_cols = build_feature_matrix(
        df,
        use_draft_pick=use_draft_pick,
        prospect_context_mode=prospect_context_mode,
        use_engineered_features=use_engineered_features,
        use_pos_categorical=use_pos_categorical,
        input_normalization_mode=input_normalization_mode,
    )
    col_info = {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "ordinal_cols": ordinal_cols,
    }
    feature_cols = numeric_cols + categorical_cols + ordinal_cols + passthrough_cols

    # run() owns the train/test split
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    y_test = test_df[target_col]

    print(f"\nTarget:  {target_mode}")
    print(f"Dataset: {len(df)} total players")
    print(f"Train:   {len(train_df)} | Test: {len(test_df)}")
    print(f"Features: {len(feature_cols)} raw columns → expanded after one-hot\n")

    mlflow_ctx = build_mlflow_context(
        cfg=cfg,
        model_type="regression",
        target_name=target_mode,
        fallback_experiment_name=MLFLOW_EXPERIMENT,
        tracking_uri=tracking_uri,
        run_name=run_name,
    )

    with managed_run(mlflow_ctx):
        if cfg is not None:
            log_config_dict(cfg)
        log_common_params({
            "model_family":            "regression",
            "target":                  target_mode,
            "target_score_mode":       target_score_mode,
            "input_normalization_mode": input_normalization_mode,
            "use_draft_pick":          use_draft_pick,
            "use_engineered_features": use_engineered_features,
            "use_pos_categorical":     use_pos_categorical,
            "selected_models":         str(selected_models),
        })
        log_data_summary(
            df,
            target_col=target_col,
            task="regression",
            test_size=TEST_SIZE,
            cv_folds=5,
            random_seed=RANDOM_STATE,
        )
        log_reproducibility_metadata(device="cpu")

        bundles = train_selected_regression_models(
            train_df, target_col, cfg,
            selected_models=selected_models,
            use_fixed_xgb_params=False,
            mlflow_ctx=mlflow_ctx,
        )

        results = {}
        for key, bundle in bundles.items():
            y_pred = bundle.estimator.predict(test_df[bundle.feature_cols])
            metrics = regression_metrics(y_test, y_pred)

            mlflow.log_metrics({f"{key}__{k}": v for k, v in metrics.items()})

            display = _DISPLAY_NAMES.get(key, key)
            estimator_step = "xgb" if key == "xgboost" else key
            results[display] = {
                "pipe":             bundle.estimator,
                "model":            bundle.estimator.named_steps.get(estimator_step),
                "y_pred":           y_pred,
                "r2":               metrics["r2"],
                "rmse":             metrics["rmse"],
                "mae":              metrics["mae"],
                "alpha":            getattr(bundle.estimator.named_steps.get(estimator_step), "alpha_", None),
                "importance_kind":  "xgb" if key == "xgboost" else "coef",
                "estimator_step":   estimator_step,
            }

            print(f"{'='*40}")
            print(f"  {display}")
            print(f"  R²   = {metrics['r2']:.4f}")
            print(f"  RMSE = {metrics['rmse']:.4f}")
            print(f"  MAE  = {metrics['mae']:.4f}")

        if "Lasso" in results:
            print(f"\n{'='*40}\n  Lasso Feature Importances (non-zero only)\n{'='*40}")
            print_lasso_coefficients(results["Lasso"]["pipe"], numeric_cols, categorical_cols, ordinal_cols)

        log_candidate_summary(results, task="regression")
        plot_results(results, y_test, col_info, target_mode=target_mode, plot_dir=mlflow_ctx.plot_dir)

    return results, y_test, col_info


if __name__ == "__main__":
    import yaml
    _cfg_path = os.path.join(PROJECT_ROOT, "src", "config", "config.yaml")
    with open(_cfg_path) as _f:
        _cfg = yaml.safe_load(_f)
    _reg_cfg = (_cfg.get("model", {}) or {}).get("regression", {}) or {}
    run(
        target_mode=_reg_cfg.get("target_mode", TARGET_MODE),
        use_draft_pick=_reg_cfg.get("use_draft_pick", USE_DRAFT_PICK),
        cfg=_cfg,
    )
