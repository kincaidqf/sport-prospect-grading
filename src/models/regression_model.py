"""Regression trainer for NBA draft prospect outcomes from NCAA stats."""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.data.loader import (
    PROJECT_ROOT,
    RANDOM_STATE,
    TARGET_COL,
    TEST_SIZE,
    build_feature_matrix,
    load_data,
)
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
REGRESSION_TARGETS = {"plus_minus", "composite_score"}
TARGET_MODE = "composite_score"
USE_DRAFT_PICK = False
ARTIFACT_DIR = os.path.join(PROJECT_ROOT, "outputs", "plots")


def train_and_evaluate(df, target_mode=TARGET_MODE, use_draft_pick=USE_DRAFT_PICK, mlflow_ctx=None, xgb_cfg=None):
    if target_mode not in REGRESSION_TARGETS:
        raise ValueError(
            f"regression_model only supports {sorted(REGRESSION_TARGETS)}; got {target_mode!r}"
        )

    preprocessor, numeric_cols, categorical_cols, ordinal_cols = build_feature_matrix(
        df,
        use_draft_pick=use_draft_pick,
    )
    feature_cols = numeric_cols + categorical_cols + ordinal_cols
    col = TARGET_COL[target_mode]
    train, test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    X_train, y_train = train[feature_cols], train[col]
    X_test, y_test = test[feature_cols], test[col]

    print(f"\nTarget:  {target_mode}")
    print(f"Dataset: {len(df)} total players")
    print(f"Train:   {len(train)} | Test: {len(test)}")
    print(f"Features: {len(feature_cols)} raw columns → expanded after one-hot\n")

    results = {}
    col_info = {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "ordinal_cols": ordinal_cols,
    }

    _run_regression(
        preprocessor,
        numeric_cols,
        categorical_cols,
        ordinal_cols,
        X_train,
        y_train,
        X_test,
        y_test,
        train,
        test,
        target_mode,
        use_draft_pick,
        results,
        mlflow_ctx,
        xgb_cfg=xgb_cfg,
    )
    return results, y_test, col_info


def _run_regression(
    preprocessor,
    numeric_cols,
    categorical_cols,
    ordinal_cols,
    X_train,
    y_train,
    X_test,
    y_test,
    train,
    test,
    target_mode,
    use_draft_pick,
    results,
    mlflow_ctx,
    xgb_cfg=None,
):
    alphas = np.logspace(-3, 2, 100)
    linear_cv_folds = 5

    for name, model in [
        ("Lasso", LassoCV(alphas=alphas, cv=linear_cv_folds, max_iter=10000, random_state=RANDOM_STATE)),
        ("Ridge", RidgeCV(alphas=alphas, cv=linear_cv_folds)),
    ]:
        run_manager = managed_run(
            mlflow_ctx,
            run_name=f"{mlflow_ctx.parent_run_name}__{name.lower()}",
            nested=True,
            tags={"estimator": name.lower()},
        ) if mlflow_ctx else mlflow.start_run(run_name=f"{name}_{target_mode}")
        with run_manager:
            pipe = Pipeline([("preprocessor", preprocessor), (name.lower(), model)])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            metrics = regression_metrics(y_test, y_pred)

            log_common_params(
                {
                    "model": name,
                    "target": target_mode,
                    "alpha": model.alpha_,
                    "alpha_search_min": alphas[0],
                    "alpha_search_max": alphas[-1],
                    "alpha_search_n": len(alphas),
                    "cv_folds": linear_cv_folds,
                    "random_seed": RANDOM_STATE,
                    "n_train": len(train),
                    "n_test": len(test),
                    "use_draft_pick": use_draft_pick,
                }
            )
            mlflow.log_metrics(metrics)

            if name == "Lasso":
                coef_df = get_lasso_coef_df(pipe, numeric_cols, categorical_cols, ordinal_cols)
                mlflow.log_metric("n_nonzero_features", len(coef_df))

            mlflow.sklearn.log_model(pipe, artifact_path=name.lower())
            results[name] = {
                "pipe": pipe,
                "model": model,
                "y_pred": y_pred,
                "r2": metrics["r2"],
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "alpha": model.alpha_,
                "importance_kind": "coef",
                "estimator_step": name.lower(),
            }

            print(f"{'='*40}")
            print(f"  {name} (alpha={model.alpha_:.4f})")
            print(f"  R²   = {metrics['r2']:.4f}")
            print(f"  RMSE = {metrics['rmse']:.4f}")
            print(f"  MAE  = {metrics['mae']:.4f}")

    cfg = xgb_cfg or {}
    param_grid = {
        "xgb__n_estimators": cfg.get("n_estimators", [100, 200]),
        "xgb__max_depth": cfg.get("max_depth", [2, 3]),
        "xgb__learning_rate": cfg.get("learning_rate", [0.05, 0.1]),
        "xgb__subsample": cfg.get("subsample", [0.7, 0.9]),
    }
    cv_folds = cfg.get("cv_folds", 5)
    xgb_n_jobs = cfg.get("n_jobs", 1)
    grid_n_jobs = cfg.get("grid_n_jobs", 1)
    pre_dispatch = cfg.get("pre_dispatch", 1)

    try:
        run_manager = managed_run(
            mlflow_ctx,
            run_name=f"{mlflow_ctx.parent_run_name}__xgboost",
            nested=True,
            tags={"estimator": "xgboost"},
        ) if mlflow_ctx else mlflow.start_run(run_name=f"XGBoost_{target_mode}")
        with run_manager:
            log_common_params(
                {
                    "model": "XGBoost",
                    "target": target_mode,
                    "cv_folds": cv_folds,
                    "xgb_n_jobs": xgb_n_jobs,
                    "grid_n_jobs": grid_n_jobs,
                    "pre_dispatch": pre_dispatch,
                    "n_estimators_grid": param_grid["xgb__n_estimators"],
                    "max_depth_grid": param_grid["xgb__max_depth"],
                    "learning_rate_grid": param_grid["xgb__learning_rate"],
                    "subsample_grid": param_grid["xgb__subsample"],
                    "random_seed": RANDOM_STATE,
                    "n_train": len(X_train),
                    "n_test": len(X_test),
                    "use_draft_pick": use_draft_pick,
                }
            )
            xgb_base = Pipeline(
                [
                    ("preprocessor", clone(preprocessor)),
                    (
                        "xgb",
                        XGBRegressor(
                            random_state=RANDOM_STATE,
                            n_jobs=xgb_n_jobs,
                            verbosity=0,
                            min_child_weight=5,
                        ),
                    ),
                ]
            )
            gs = GridSearchCV(
                xgb_base,
                param_grid,
                cv=cv_folds,
                scoring="r2",
                n_jobs=grid_n_jobs,
                pre_dispatch=pre_dispatch,
            )
            gs.fit(X_train, y_train)
            best = gs.best_estimator_
            y_pred_xgb = best.predict(X_test)
            metrics_xgb = regression_metrics(y_test, y_pred_xgb)

            log_common_params(
                {
                    **{key.replace("xgb__", ""): value for key, value in gs.best_params_.items()},
                    "best_cv_score": round(float(gs.best_score_), 4),
                }
            )
            mlflow.log_metrics(metrics_xgb)
            log_xgb_importances(best, numeric_cols, categorical_cols, ordinal_cols)
            mlflow.sklearn.log_model(best, artifact_path="xgboost")

        results["XGBoost"] = {
            "pipe": best,
            "model": best.named_steps["xgb"],
            "y_pred": y_pred_xgb,
            "r2": metrics_xgb["r2"],
            "rmse": metrics_xgb["rmse"],
            "mae": metrics_xgb["mae"],
            "alpha": None,
            "best_cv_score": round(float(gs.best_score_), 4),
            "importance_kind": "xgb",
            "estimator_step": "xgb",
        }

        print(f"{'='*40}")
        print(f"  XGBoost  (best: {gs.best_params_})")
        print(f"  R²   = {metrics_xgb['r2']:.4f}")
        print(f"  RMSE = {metrics_xgb['rmse']:.4f}")
        print(f"  MAE  = {metrics_xgb['mae']:.4f}")

        print(f"\n{'='*40}\n  XGBoost Feature Importances (top 10)\n{'='*40}")
        print_xgb_importances(best, numeric_cols, categorical_cols, ordinal_cols)

    except Exception as exc:
        import traceback
        print(f"\n[WARNING] XGBoost training failed and will be skipped: {exc}")
        traceback.print_exc()

    print(f"\n{'='*40}\n  Lasso Feature Importances (non-zero only)\n{'='*40}")
    print_lasso_coefficients(results["Lasso"]["pipe"], numeric_cols, categorical_cols, ordinal_cols)


def plot_results(results, y_test, col_info, target_mode=TARGET_MODE, plot_dir=ARTIFACT_DIR):
    _plot_regression(results, y_test, target_mode, plot_dir=plot_dir)
    plot_feature_importance(results, col_info, target_mode, artifact_dir=plot_dir)
    plot_model_summary(results, target_mode, "regression", artifact_dir=plot_dir)


def _plot_regression(results, y_test, target_mode, plot_dir):
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(6 * n, 10))
    fig.suptitle(f"NBA {target_mode} Prediction from College Stats", fontsize=14, fontweight="bold")

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

    plt.tight_layout()
    save_and_log(
        fig,
        "regression_results.png",
        {name: {"r2": res["r2"], "rmse": res["rmse"], "mae": res["mae"]} for name, res in results.items()},
        artifact_dir=plot_dir,
    )


def run(target_mode=TARGET_MODE, use_draft_pick=USE_DRAFT_PICK, df=None, cfg=None, run_name=None, tracking_uri=None):
    model_cfg     = (cfg or {}).get("model", {}) or {}
    composite_cfg = model_cfg.get("composite_score") or {}
    xgb_cfg       = model_cfg.get("regression", {}).get("xgboost") or {}
    df = load_data(composite_cfg=composite_cfg) if df is None else df
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
        log_common_params(
            {
                "model_family": "regression",
                "target": target_mode,
                "use_draft_pick": use_draft_pick,
            }
        )
        log_data_summary(
            df,
            target_col=TARGET_COL[target_mode],
            task="regression",
            test_size=TEST_SIZE,
            cv_folds=5,
            random_seed=RANDOM_STATE,
        )
        log_reproducibility_metadata(device="cpu")
        results, y_test, col_info = train_and_evaluate(
            df,
            target_mode=target_mode,
            use_draft_pick=use_draft_pick,
            mlflow_ctx=mlflow_ctx,
            xgb_cfg=xgb_cfg,
        )
        log_candidate_summary(results, task="regression")
        plot_results(results, y_test, col_info, target_mode=target_mode, plot_dir=mlflow_ctx.plot_dir)
    return results, y_test, col_info


if __name__ == "__main__":
    run()
