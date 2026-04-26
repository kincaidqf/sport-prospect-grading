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
from src.training.evaluate import classification_metrics, cost_sensitive_predict, mean_cost
from src.utils.features import log_xgb_importances, print_xgb_importances
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


MLFLOW_EXPERIMENT = "nba-draft-prospect-classification"
CLASSIFICATION_TARGETS = {"became_starter", "prospect_tier"}
TARGET_MODE = "prospect_tier"
TIER_NAMES = ["Bust", "Contributor", "Star"]
USE_DRAFT_PICK = False
ARTIFACT_DIR = os.path.join(PROJECT_ROOT, "outputs", "plots")

# Cost matrix for ordinal 3-tier classes (Bust=0, Contributor=1, Star=2).
# cost[true, predicted]: a two-step error (Bust↔Star) costs 3× more than a
# one-step error (Bust↔Contributor or Contributor↔Star).
ORDINAL_COST_MATRIX = np.array([
    [0, 1, 3],  # true=Bust
    [1, 0, 1],  # true=Contributor
    [3, 1, 0],  # true=Star
], dtype=float)


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
    is_multiclass = target_mode == "prospect_tier"
    tier_labels   = TIER_NAMES if is_multiclass else ["No", "Yes"]

    cs = np.logspace(-3, 2, 20)
    linear_cv_folds = 5

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
                cv=linear_cv_folds,
                penalty=penalty,
                solver="saga",
                max_iter=5000,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
            pipe = Pipeline([("preprocessor", preprocessor), ("clf", model)])
            pipe.fit(X_train, y_train)
            proba  = pipe.predict_proba(X_test)
            y_prob = proba if is_multiclass else proba[:, 1]
            y_pred = (
                cost_sensitive_predict(proba, ORDINAL_COST_MATRIX)
                if is_multiclass
                else pipe.predict(X_test)
            )

            metrics = classification_metrics(y_test, y_pred, y_prob)
            if is_multiclass:
                metrics["mean_cost"] = mean_cost(
                    np.asarray(y_test), y_pred, ORDINAL_COST_MATRIX
                )
            best_c = model.C_[0]

            log_common_params(
                {
                    "model": name,
                    "target": target_mode,
                    "penalty": penalty,
                    "C": best_c,
                    "C_search_min": cs[0],
                    "C_search_max": cs[-1],
                    "C_search_n": len(cs),
                    "cv_folds": linear_cv_folds,
                    "random_seed": RANDOM_STATE,
                    "n_train": len(train),
                    "n_test": len(test),
                    "use_draft_pick": use_draft_pick,
                    "cost_sensitive": is_multiclass,
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
                "auc": metrics["auc"],
                "f1_macro": metrics.get("f1_macro", 0.0),
                "mean_cost": metrics.get("mean_cost"),
                "C": best_c,
                "importance_kind": "coef",
                "estimator_step": "clf",
            }

            print(f"{'='*40}")
            print(f"  {name} (C={best_c:.4f}, penalty={penalty})")
            print(f"  Accuracy = {metrics['accuracy']:.4f}")
            if is_multiclass:
                print(f"  F1-macro = {metrics['f1_macro']:.4f}")
                print(f"  MeanCost = {metrics['mean_cost']:.4f}  (cost-sensitive predictions)")
            print(f"  AUC      = {metrics['auc']:.4f}")
            print(classification_report(y_test, y_pred, target_names=tier_labels, zero_division=0))

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
            xgb_extra = (
                {"objective": "multi:softprob", "num_class": len(TIER_NAMES), "eval_metric": "mlogloss"}
                if is_multiclass
                else {"objective": "binary:logistic", "eval_metric": "logloss"}
            )
            xgb_base = Pipeline(
                [
                    ("preprocessor", clone(preprocessor)),
                    (
                        "xgb",
                        XGBClassifier(
                            random_state=RANDOM_STATE,
                            n_jobs=xgb_n_jobs,
                            verbosity=0,
                            min_child_weight=5,
                            **xgb_extra,
                        ),
                    ),
                ]
            )
            gs = GridSearchCV(
                xgb_base,
                param_grid,
                cv=cv_folds,
                scoring="f1_macro" if is_multiclass else "roc_auc",
                n_jobs=grid_n_jobs,
                pre_dispatch=pre_dispatch,
            )
            gs.fit(X_train, y_train)
            best = gs.best_estimator_
            proba_xgb  = best.predict_proba(X_test)
            y_prob_xgb = proba_xgb if is_multiclass else proba_xgb[:, 1]
            y_pred_xgb = (
                cost_sensitive_predict(proba_xgb, ORDINAL_COST_MATRIX)
                if is_multiclass
                else best.predict(X_test)
            )
            metrics_xgb = classification_metrics(y_test, y_pred_xgb, y_prob_xgb)
            if is_multiclass:
                metrics_xgb["mean_cost"] = mean_cost(
                    np.asarray(y_test), y_pred_xgb, ORDINAL_COST_MATRIX
                )

            log_common_params(
                {
                    **{key.replace("xgb__", ""): value for key, value in gs.best_params_.items()},
                    "best_cv_score": round(float(gs.best_score_), 4),
                    "cost_sensitive": is_multiclass,
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
            "auc": metrics_xgb["auc"],
            "f1_macro": metrics_xgb.get("f1_macro", 0.0),
            "mean_cost": metrics_xgb.get("mean_cost"),
            "C": None,
            "best_cv_score": round(float(gs.best_score_), 4),
            "importance_kind": "xgb",
            "estimator_step": "xgb",
        }

        print(f"{'='*40}")
        print(f"  XGBoost  (best: {gs.best_params_})")
        print(f"  Accuracy = {metrics_xgb['accuracy']:.4f}")
        if is_multiclass:
            print(f"  F1-macro = {metrics_xgb['f1_macro']:.4f}")
            print(f"  MeanCost = {metrics_xgb['mean_cost']:.4f}  (cost-sensitive predictions)")
        print(f"  AUC      = {metrics_xgb['auc']:.4f}")
        print(classification_report(y_test, y_pred_xgb, target_names=tier_labels, zero_division=0))

        print(f"\n{'='*40}\n  XGBoost Feature Importances (top 10)\n{'='*40}")
        print_xgb_importances(best, numeric_cols, categorical_cols, ordinal_cols)

    except Exception as exc:
        import traceback
        print(f"\n[WARNING] XGBoost training failed and will be skipped: {exc}")
        traceback.print_exc()


def plot_results(results, y_test, col_info, target_mode=TARGET_MODE, plot_dir=ARTIFACT_DIR):
    _plot_classification(results, y_test, target_mode, plot_dir=plot_dir)
    plot_feature_importance(results, col_info, target_mode, artifact_dir=plot_dir)
    plot_model_summary(results, target_mode, "classification", artifact_dir=plot_dir)


def _plot_classification(results, y_test, target_mode, plot_dir):
    is_multiclass  = target_mode == "prospect_tier"
    display_labels = TIER_NAMES if is_multiclass else ["No", "Yes"]

    n = len(results)
    fig_w = 6 * n if is_multiclass else 5 * n
    fig, axes = plt.subplots(2, n, figsize=(fig_w, 10 if is_multiclass else 9))
    fig.suptitle(f"NBA {target_mode} Classification from College Stats", fontsize=13, fontweight="bold")

    y_test_arr = np.asarray(y_test)

    for col, (name, res) in enumerate(results.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            res["y_pred"],
            display_labels=display_labels,
            ax=axes[0][col],
        )
        axes[0][col].set_title(f"{name}\nAcc={res['accuracy']:.3f}")

        if is_multiclass:
            ax = axes[1][col]
            x = np.arange(len(TIER_NAMES))
            width = 0.35
            actual_counts  = [int(np.sum(y_test_arr == i)) for i in range(len(TIER_NAMES))]
            pred_counts    = [int(np.sum(res["y_pred"] == i)) for i in range(len(TIER_NAMES))]
            ax.bar(x - width / 2, actual_counts, width, label="Actual",    color="#3498db")
            ax.bar(x + width / 2, pred_counts,   width, label="Predicted", color="#e67e22")
            ax.set_xticks(x)
            ax.set_xticklabels(TIER_NAMES)
            ax.set_ylabel("Count")
            ax.set_title(
                f"Tier Distribution\nAUC-OvR={res['auc']:.3f}  F1-macro={res.get('f1_macro', 0):.3f}"
            )
            ax.legend()
        else:
            RocCurveDisplay.from_predictions(y_test, res["y_prob"], ax=axes[1][col], name=name)
            axes[1][col].set_title(f"ROC  AUC={res['auc']:.3f}")

    plt.tight_layout()
    save_and_log(
        fig,
        "classification_results.png",
        {name: {"accuracy": res["accuracy"], "auc": res["auc"]} for name, res in results.items()},
        artifact_dir=plot_dir,
    )


def run(target_mode=TARGET_MODE, use_draft_pick=USE_DRAFT_PICK, df=None, cfg=None, run_name=None, tracking_uri=None):
    model_cfg     = (cfg or {}).get("model", {}) or {}
    composite_cfg = model_cfg.get("composite_score") or {}
    xgb_cfg       = model_cfg.get("classification", {}).get("xgboost") or {}
    df = load_data(composite_cfg=composite_cfg) if df is None else df
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
            }
        )
        log_data_summary(
            df,
            target_col=TARGET_COL[target_mode],
            task="classification",
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
        log_candidate_summary(results, task="classification")
        plot_results(results, y_test, col_info, target_mode=target_mode, plot_dir=mlflow_ctx.plot_dir)
    return results, y_test, col_info


if __name__ == "__main__":
    run()
