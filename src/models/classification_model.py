"""Classification trainer for NBA draft prospect outcomes from NCAA stats."""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data.loader import (
    PROJECT_ROOT,
    RANDOM_STATE,
    TARGET_COL,
    TEST_SIZE,
    build_feature_matrix,
    load_data,
)
from src.training.evaluate import classification_metrics
from src.utils.features import log_xgb_importances, print_xgb_importances
from src.utils.mlflow_utils import build_mlflow_context, log_common_params, log_config_dict, managed_run
from src.utils.plotting import plot_feature_importance, plot_model_summary, save_and_log


MLFLOW_EXPERIMENT = "nba-draft-prospect-classification"
CLASSIFICATION_TARGETS = {"became_starter", "survived_3yrs"}
TARGET_MODE = "survived_3yrs"
USE_DRAFT_PICK = False
ARTIFACT_DIR = os.path.join(PROJECT_ROOT, "outputs", "plots")


def train_and_evaluate(df, target_mode=TARGET_MODE, use_draft_pick=USE_DRAFT_PICK, mlflow_ctx=None, xgb_cfg=None):
    if target_mode not in CLASSIFICATION_TARGETS:
        raise ValueError(
            f"classification_model only supports {sorted(CLASSIFICATION_TARGETS)}; got {target_mode!r}"
        )

    preprocessor, numeric_cols, categorical_cols, ordinal_cols = build_feature_matrix(
        df,
        use_draft_pick=use_draft_pick,
    )
    feature_cols = numeric_cols + categorical_cols + ordinal_cols
    col = TARGET_COL[target_mode]
    y = df[col]
    train, test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    X_train, y_train = train[feature_cols], train[col]
    X_test, y_test = test[feature_cols], test[col]

    print(f"\nTarget:  {target_mode}")
    print(f"Classes: {dict(y.value_counts().sort_index())}")
    print(f"Dataset: {len(df)} total players")
    print(f"Train:   {len(train)} | Test: {len(test)}")
    print(f"Features: {len(feature_cols)} raw columns → expanded after one-hot\n")

    results = {}
    col_info = {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "ordinal_cols": ordinal_cols,
    }

    _run_classification(
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


def _run_classification(
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
    cs = np.logspace(-3, 2, 20)

    for name, penalty in [("LogisticL1", "l1"), ("LogisticL2", "l2")]:
        run_manager = managed_run(
            mlflow_ctx,
            run_name=f"{mlflow_ctx.parent_run_name}__{name.lower()}",
            nested=True,
            tags={"estimator": name.lower()},
        ) if mlflow_ctx else mlflow.start_run(run_name=f"{name}_{target_mode}")
        with run_manager:
            model = LogisticRegressionCV(
                Cs=cs,
                cv=5,
                penalty=penalty,
                solver="saga",
                max_iter=5000,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
            pipe = Pipeline([("preprocessor", preprocessor), ("clf", model)])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            y_prob = pipe.predict_proba(X_test)[:, 1]

            metrics = classification_metrics(y_test, y_pred, y_prob)
            best_c = model.C_[0]

            log_common_params(
                {
                    "model": name,
                    "target": target_mode,
                    "penalty": penalty,
                    "C": best_c,
                    "n_train": len(train),
                    "n_test": len(test),
                    "use_draft_pick": use_draft_pick,
                }
            )
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipe, artifact_path=name.lower())

            results[name] = {
                "pipe": pipe,
                "model": model,
                "y_pred": y_pred,
                "y_prob": y_prob,
                "accuracy": metrics["accuracy"],
                "auc": metrics["roc_auc"],
                "C": best_c,
                "importance_kind": "coef",
                "estimator_step": "clf",
            }

            print(f"{'='*40}")
            print(f"  {name} (C={best_c:.4f}, penalty={penalty})")
            print(f"  Accuracy = {metrics['accuracy']:.4f}")
            print(f"  ROC-AUC  = {metrics['roc_auc']:.4f}")
            print(classification_report(y_test, y_pred, target_names=["No", "Yes"], zero_division=0))

    cfg = xgb_cfg or {}
    param_grid = {
        "xgb__n_estimators": cfg.get("n_estimators", [100, 200]),
        "xgb__max_depth": cfg.get("max_depth", [2, 3]),
        "xgb__learning_rate": cfg.get("learning_rate", [0.05, 0.1]),
        "xgb__subsample": cfg.get("subsample", [0.7, 0.9]),
    }
    cv_folds = cfg.get("cv_folds", 5)

    try:
        xgb_base = Pipeline(
            [
                ("preprocessor", clone(preprocessor)),
                (
                    "xgb",
                    XGBClassifier(
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        verbosity=0,
                        min_child_weight=5,
                        eval_metric="logloss",
                    ),
                ),
            ]
        )
        gs = GridSearchCV(xgb_base, param_grid, cv=cv_folds, scoring="roc_auc", n_jobs=-1)
        gs.fit(X_train, y_train)
        best = gs.best_estimator_
        y_pred_xgb = best.predict(X_test)
        y_prob_xgb = best.predict_proba(X_test)[:, 1]
        metrics_xgb = classification_metrics(y_test, y_pred_xgb, y_prob_xgb)

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
                    **{key.replace("xgb__", ""): value for key, value in gs.best_params_.items()},
                    "n_train": len(X_train),
                    "n_test": len(X_test),
                    "use_draft_pick": use_draft_pick,
                }
            )
            mlflow.log_metrics(metrics_xgb)
            log_xgb_importances(best, numeric_cols, categorical_cols, ordinal_cols)
            mlflow.sklearn.log_model(best, artifact_path="xgboost")

        results["XGBoost"] = {
            "pipe": best,
            "model": best.named_steps["xgb"],
            "y_pred": y_pred_xgb,
            "y_prob": y_prob_xgb,
            "accuracy": metrics_xgb["accuracy"],
            "auc": metrics_xgb["roc_auc"],
            "C": None,
            "importance_kind": "xgb",
            "estimator_step": "xgb",
        }

        print(f"{'='*40}")
        print(f"  XGBoost  (best: {gs.best_params_})")
        print(f"  Accuracy = {metrics_xgb['accuracy']:.4f}")
        print(f"  ROC-AUC  = {metrics_xgb['roc_auc']:.4f}")
        print(classification_report(y_test, y_pred_xgb, target_names=["No", "Yes"], zero_division=0))

        print(f"\n{'='*40}\n  XGBoost Feature Importances (top 10)\n{'='*40}")
        print_xgb_importances(best, numeric_cols, categorical_cols, ordinal_cols)

    except Exception as exc:
        print(f"\n[WARNING] XGBoost training failed and will be skipped: {exc}")


def plot_results(results, y_test, col_info, target_mode=TARGET_MODE, plot_dir=ARTIFACT_DIR):
    _plot_classification(results, y_test, target_mode, plot_dir=plot_dir)
    plot_feature_importance(results, col_info, target_mode, artifact_dir=plot_dir)
    plot_model_summary(results, target_mode, "classification", artifact_dir=plot_dir)


def _plot_classification(results, y_test, target_mode, plot_dir):
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 9))
    fig.suptitle(f"NBA {target_mode} Classification from College Stats", fontsize=13, fontweight="bold")

    for col, (name, res) in enumerate(results.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            res["y_pred"],
            display_labels=["No", "Yes"],
            ax=axes[0][col],
        )
        axes[0][col].set_title(f"{name}\nAcc={res['accuracy']:.3f}")

        RocCurveDisplay.from_predictions(y_test, res["y_prob"], ax=axes[1][col], name=name)
        axes[1][col].set_title(f"ROC  AUC={res['auc']:.3f}")

    plt.tight_layout()
    save_and_log(
        fig,
        "classification_results.png",
        {name: {"accuracy": res["accuracy"], "roc_auc": res["auc"]} for name, res in results.items()},
        artifact_dir=plot_dir,
    )


def run(target_mode=TARGET_MODE, use_draft_pick=USE_DRAFT_PICK, df=None, cfg=None, run_name=None, tracking_uri=None):
    df = load_data() if df is None else df
    xgb_cfg = ((cfg or {}).get("model", {}).get("classification", {}).get("xgboost") or {})
    mlflow_ctx = build_mlflow_context(
        cfg=cfg,
        model_type="classification",
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
                "model_family": "classification",
                "target": target_mode,
                "use_draft_pick": use_draft_pick,
                "n_rows": len(df),
            }
        )
        results, y_test, col_info = train_and_evaluate(
            df,
            target_mode=target_mode,
            use_draft_pick=use_draft_pick,
            mlflow_ctx=mlflow_ctx,
            xgb_cfg=xgb_cfg,
        )
        plot_results(results, y_test, col_info, target_mode=target_mode, plot_dir=mlflow_ctx.plot_dir)
    return results, y_test, col_info


if __name__ == "__main__":
    run()
