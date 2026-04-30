"""Classification trainer for NBA draft prospect outcomes from NCAA stats."""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report, f1_score
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
# XGBClassifier is imported lazily inside _run_classification to prevent macOS
# OpenMP libomp conflicts at module initialization time.

from src.data.loader import (
    PROJECT_ROOT,
    PROSPECT_CONTEXT_MODE,
    RANDOM_STATE,
    TARGET_COL,
    TEST_SIZE,
    CLASSIFICATION_EXCLUDED_NUMERIC,
    build_feature_matrix,
    load_data,
)
from src.training.evaluate import classification_metrics
from src.training.splits import get_chronological_split, get_random_split, get_repeated_stratified_cv
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
TIER_NAMES = ["Bust", "Bench", "Starter", "Star"]
TIER_CLASS_NAMES = ["bust", "bench", "starter", "star"]
USE_DRAFT_PICK = False  # never include draft_pick by default — it leaks scout judgment
ARTIFACT_DIR = os.path.join(PROJECT_ROOT, "outputs", "plots")


# ── Threshold tuning ───────────────────────────────────────────────────────────

def _tune_thresholds(proba_val: np.ndarray, y_val: np.ndarray, n_classes: int) -> np.ndarray:
    """Coordinate-descent search for per-class probability offsets maximizing macro F1.

    Never touches test data — caller must pass a held-out validation array.
    Returns an offset vector of shape (n_classes,) to add to predict_proba output.
    """
    offsets = np.zeros(n_classes)
    grid = np.linspace(-0.15, 0.15, 13)
    for cls_idx in range(n_classes):
        best_f1 = f1_score(y_val, np.argmax(proba_val + offsets, axis=1), average="macro", zero_division=0)
        for offset in grid:
            trial = offsets.copy()
            trial[cls_idx] = offset
            pred = np.argmax(proba_val + trial, axis=1)
            f1 = f1_score(y_val, pred, average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                offsets[cls_idx] = offset
    return offsets


# ── Core training ──────────────────────────────────────────────────────────────

def train_and_evaluate(
    df,
    target_mode=TARGET_MODE,
    use_draft_pick=USE_DRAFT_PICK,
    mlflow_ctx=None,
    xgb_cfg=None,
    clf_cfg=None,
    prospect_context_mode=PROSPECT_CONTEXT_MODE,
):
    if target_mode not in CLASSIFICATION_TARGETS:
        raise ValueError(
            f"classification_model only supports {sorted(CLASSIFICATION_TARGETS)}; got {target_mode!r}"
        )

    clf_cfg = clf_cfg or {}
    eval_mode = clf_cfg.get("eval_mode", "random")
    class_weight = clf_cfg.get("class_weight", None) or None
    threshold_tuning = bool(clf_cfg.get("threshold_tuning", False))

    preprocessor, numeric_cols, categorical_cols, ordinal_cols = build_feature_matrix(
        df,
        use_draft_pick=use_draft_pick,
        exclude_features=CLASSIFICATION_EXCLUDED_NUMERIC,
        prospect_context_mode=prospect_context_mode,
        use_engineered_features=False,
        use_pos_categorical=False,
    )
    feature_cols = numeric_cols + categorical_cols + ordinal_cols
    col = TARGET_COL[target_mode]
    y = df[col]

    print(f"\nTarget:    {target_mode}")
    print(f"Classes:   {dict(y.value_counts().sort_index())}")
    print(f"Dataset:   {len(df)} total players")
    print(f"Eval mode: {eval_mode}")
    print(f"Features:  {len(feature_cols)} raw columns → expanded after one-hot\n")

    col_info = {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "ordinal_cols": ordinal_cols,
    }
    results = {}

    if eval_mode == "chronological":
        train_df, val_df, test_df = get_chronological_split(df)
        X_train = train_df[feature_cols]
        y_train = train_df[col]
        X_val   = val_df[feature_cols]
        y_val   = val_df[col]
        X_test  = test_df[feature_cols]
        y_test  = test_df[col]
        print(
            f"Chronological split — "
            f"Train: {len(train_df)} (≤2018) | Val: {len(val_df)} (2019-2020) | "
            f"Test: {len(test_df)} (2021-2023)"
        )
        _run_classification(
            preprocessor, numeric_cols, categorical_cols, ordinal_cols,
            X_train, y_train, X_test, y_test,
            train_df, test_df, target_mode, use_draft_pick, results, mlflow_ctx,
            xgb_cfg=xgb_cfg, class_weight=class_weight,
            threshold_tuning=threshold_tuning, X_val=X_val, y_val=y_val,
            eval_mode=eval_mode,
        )
        return results, y_test, col_info

    elif eval_mode == "repeated_cv":
        cv = get_repeated_stratified_cv()
        X_all = df[feature_cols]
        y_all = df[col]
        print(f"Repeated CV: {cv.n_splits} splits × {cv.n_repeats} repeats")
        _run_repeated_cv(
            preprocessor, numeric_cols, categorical_cols, ordinal_cols,
            X_all, y_all, target_mode, use_draft_pick, results, mlflow_ctx,
            xgb_cfg=xgb_cfg, class_weight=class_weight, cv=cv,
        )
        # Final artifact fit on a random train/test split
        train_df, test_df = get_random_split(df, TEST_SIZE, RANDOM_STATE, col)
        X_test  = test_df[feature_cols]
        y_test  = test_df[col]
        return results, y_test, col_info

    else:  # random (default)
        train_df, test_df = get_random_split(df, TEST_SIZE, RANDOM_STATE, col)
        X_train = train_df[feature_cols]
        y_train = train_df[col]
        X_test  = test_df[feature_cols]
        y_test  = test_df[col]
        print(f"Train: {len(train_df)} | Test: {len(test_df)}")

        X_tr, X_val, y_tr, y_val = X_train, None, y_train, None
        if threshold_tuning:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train, test_size=0.15,
                stratify=y_train, random_state=RANDOM_STATE,
            )
            print(f"  Threshold tuning val: {len(X_val)} samples (held out from train)")

        _run_classification(
            preprocessor, numeric_cols, categorical_cols, ordinal_cols,
            X_tr, y_tr, X_test, y_test,
            train_df, test_df, target_mode, use_draft_pick, results, mlflow_ctx,
            xgb_cfg=xgb_cfg, class_weight=class_weight,
            threshold_tuning=threshold_tuning, X_val=X_val, y_val=y_val,
            eval_mode=eval_mode,
        )
        return results, y_test, col_info


def _run_repeated_cv(
    preprocessor,
    numeric_cols,
    categorical_cols,
    ordinal_cols,
    X_all,
    y_all,
    target_mode,
    use_draft_pick,
    results,
    mlflow_ctx,
    xgb_cfg=None,
    class_weight=None,
    cv=None,
):
    """Run repeated stratified CV for each model and record mean/std metrics."""
    is_multiclass = target_mode == "prospect_tier"
    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "balanced_accuracy": "balanced_accuracy",
    }
    if not is_multiclass:
        scoring["roc_auc"] = "roc_auc"

    cfg = xgb_cfg or {}
    cv_folds = cfg.get("cv_folds", 3)
    n_jobs = cfg.get("n_jobs", 1)

    from xgboost import XGBClassifier  # noqa: PLC0415
    xgb_extra = (
        {"objective": "multi:softprob", "num_class": len(TIER_NAMES), "eval_metric": "mlogloss"}
        if is_multiclass
        else {"objective": "binary:logistic", "eval_metric": "logloss"}
    )
    # XGBoost uses fixed params (first value from each grid) — cross_validate
    # doesn't support nested grid search; use random/chronological for full tuning.
    xgb_fixed = {
        "n_estimators":     cfg.get("n_estimators",     [200])[0],
        "max_depth":        cfg.get("max_depth",         [3])[0],
        "learning_rate":    cfg.get("learning_rate",     [0.05])[0],
        "subsample":        cfg.get("subsample",         [0.7])[0],
        "colsample_bytree": cfg.get("colsample_bytree",  [1.0])[0],
        "min_child_weight": cfg.get("min_child_weight",  [3])[0],
        "reg_alpha":        cfg.get("reg_alpha",         [0])[0],
        "reg_lambda":       cfg.get("reg_lambda",        [1])[0],
        "gamma":            cfg.get("gamma",             [0])[0],
    }

    model_specs = [
        ("LogisticL1", Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("clf", LogisticRegressionCV(
                Cs=np.logspace(-3, 2, 10), cv=cv_folds, penalty="l1",
                solver="saga", max_iter=5000, random_state=RANDOM_STATE,
                n_jobs=1, class_weight=class_weight,
                scoring="f1_macro" if is_multiclass else "roc_auc",
            )),
        ])),
        ("LogisticL2", Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("clf", LogisticRegressionCV(
                Cs=np.logspace(-3, 2, 10), cv=cv_folds, penalty="l2",
                solver="saga", max_iter=5000, random_state=RANDOM_STATE,
                n_jobs=1, class_weight=class_weight,
                scoring="f1_macro" if is_multiclass else "roc_auc",
            )),
        ])),
        ("XGBoost", Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("xgb", XGBClassifier(
                random_state=RANDOM_STATE, n_jobs=n_jobs, verbosity=0,
                **xgb_extra, **xgb_fixed,
            )),
        ])),
    ]

    sw_all = compute_sample_weight("balanced", y_all) if class_weight == "balanced" else None

    for name, pipe in model_specs:
        is_xgb = name == "XGBoost"
        run_manager = (
            managed_run(mlflow_ctx, run_name=f"{mlflow_ctx.parent_run_name}__{name.lower()}_cv", nested=True, tags={"estimator": name.lower(), "eval_mode": "repeated_cv"})
            if mlflow_ctx else mlflow.start_run(run_name=f"{name}_cv_{target_mode}")
        )
        with run_manager:
            # Note: XGBoost class balancing (sample_weight) is NOT applied in CV folds —
            # cross_validate doesn't subset fit_params arrays per fold. Balancing is
            # applied in the final artifact fit below.
            cv_res = cross_validate(
                pipe, X_all, y_all, cv=cv, scoring=scoring,
                n_jobs=n_jobs, return_train_score=False,
            )
            mean_f1 = float(np.mean(cv_res["test_f1_macro"]))
            std_f1  = float(np.std(cv_res["test_f1_macro"]))
            mean_ba = float(np.mean(cv_res["test_balanced_accuracy"]))
            mean_acc = float(np.mean(cv_res["test_accuracy"]))
            cv_metrics = {
                "cv_mean_f1_macro": mean_f1,
                "cv_std_f1_macro": std_f1,
                "cv_mean_balanced_accuracy": mean_ba,
                "cv_mean_accuracy": mean_acc,
            }
            mlflow.log_metrics(cv_metrics)
            log_common_params({
                "model": name, "target": target_mode,
                "cv_n_splits": cv.n_splits, "cv_n_repeats": cv.n_repeats,
                "use_draft_pick": use_draft_pick,
            })
            print(f"  {name}: CV F1-macro = {mean_f1:.4f} ± {std_f1:.4f}  |  Bal.Acc = {mean_ba:.4f}")
            # Fit final model on full data for artifact logging
            if is_xgb and sw_all is not None:
                pipe.fit(X_all, y_all, xgb__sample_weight=sw_all)
            else:
                pipe.fit(X_all, y_all)
            mlflow.sklearn.log_model(pipe, artifact_path=name.lower())
            results[name] = {
                "pipe": pipe,
                "y_pred": None,
                "y_prob": None,
                "accuracy": mean_acc,
                "auc": 0.0,
                "f1_macro": mean_f1,
                "balanced_accuracy": mean_ba,
                "cv_mean_f1_macro": mean_f1,
                "cv_std_f1_macro": std_f1,
                "C": None,
                "importance_kind": "xgb" if is_xgb else "coef",
                "estimator_step": "xgb" if is_xgb else "clf",
            }


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
    class_weight=None,
    threshold_tuning=False,
    X_val=None,
    y_val=None,
    eval_mode="random",
):
    is_multiclass = target_mode == "prospect_tier"
    tier_labels   = TIER_NAMES if is_multiclass else ["No", "Yes"]
    class_names   = TIER_CLASS_NAMES if is_multiclass else None

    cfg = xgb_cfg or {}
    cv_folds     = cfg.get("cv_folds", 3)
    xgb_n_jobs   = cfg.get("n_jobs", 1)
    grid_n_jobs  = cfg.get("grid_n_jobs", 1)
    pre_dispatch = cfg.get("pre_dispatch", 1)

    cs = np.logspace(-3, 2, 20)
    linear_cv_folds = 5

    # ── Logistic models ────────────────────────────────────────────────────────
    for name, penalty in [("LogisticL1", "l1"), ("LogisticL2", "l2")]:
        run_manager = (
            managed_run(mlflow_ctx, run_name=f"{mlflow_ctx.parent_run_name}__{name.lower()}", nested=True, tags={"estimator": name.lower()})
            if mlflow_ctx else mlflow.start_run(run_name=f"{name}_{target_mode}")
        )
        with run_manager:
            model = LogisticRegressionCV(
                Cs=cs,
                cv=linear_cv_folds,
                penalty=penalty,
                solver="saga",
                max_iter=5000,
                random_state=RANDOM_STATE,
                n_jobs=1,  # keep single-process; -1 causes macOS OpenMP conflicts with XGBoost
                class_weight=class_weight,
                scoring="f1_macro" if is_multiclass else "roc_auc",
            )
            pipe = Pipeline([("preprocessor", preprocessor), ("clf", model)])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            proba  = pipe.predict_proba(X_test)
            y_prob = proba if is_multiclass else proba[:, 1]

            metrics = classification_metrics(y_test, y_pred, y_prob, class_names=class_names)
            best_c = float(model.C_[0]) if hasattr(model.C_, "__len__") else float(model.C_)

            # Threshold tuning
            if threshold_tuning and X_val is not None and is_multiclass:
                proba_val = pipe.predict_proba(X_val)
                offsets   = _tune_thresholds(np.array(proba_val), np.array(y_val), proba.shape[1])
                y_pred_t  = np.argmax(proba + offsets, axis=1)
                f1_tuned  = float(f1_score(y_test, y_pred_t, average="macro", zero_division=0))
                mlflow.log_metrics({"default_f1_macro": metrics.get("f1_macro", 0.0), "tuned_f1_macro": f1_tuned})
                print(f"  {name} threshold tuning: default F1={metrics.get('f1_macro',0):.4f}  tuned F1={f1_tuned:.4f}")

            log_common_params({
                "model": name, "target": target_mode, "penalty": penalty,
                "C": best_c, "cv_folds": linear_cv_folds,
                "class_weight": str(class_weight), "eval_mode": eval_mode,
                "random_seed": RANDOM_STATE, "n_train": len(X_train), "n_test": len(X_test),
                "use_draft_pick": use_draft_pick,
            })
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
                "balanced_accuracy": metrics.get("balanced_accuracy", 0.0),
                "C": best_c,
                "importance_kind": "coef",
                "estimator_step": "clf",
            }

            print(f"{'='*40}")
            print(f"  {name} (C={best_c:.4f}, penalty={penalty})")
            print(f"  Accuracy     = {metrics['accuracy']:.4f}")
            print(f"  Balanced Acc = {metrics['balanced_accuracy']:.4f}")
            if is_multiclass:
                print(f"  F1-macro     = {metrics['f1_macro']:.4f}")
            print(f"  AUC          = {metrics['auc']:.4f}")
            if is_multiclass:
                for cls in TIER_CLASS_NAMES:
                    r = metrics.get(f"recall_{cls}", 0.0)
                    p = metrics.get(f"precision_{cls}", 0.0)
                    print(f"  {cls:<12}: precision={p:.3f}  recall={r:.3f}")
            print(classification_report(y_test, y_pred, target_names=tier_labels, zero_division=0))

    # ── XGBoost (lazy import to avoid macOS OpenMP conflict) ─
    from xgboost import XGBClassifier  # noqa: PLC0415
    param_grid = {
        "xgb__n_estimators":     cfg.get("n_estimators",     [100, 200]),
        "xgb__max_depth":        cfg.get("max_depth",        [2, 3]),
        "xgb__learning_rate":    cfg.get("learning_rate",    [0.05, 0.1]),
        "xgb__subsample":        cfg.get("subsample",        [0.7, 0.9]),
        "xgb__colsample_bytree": cfg.get("colsample_bytree", [1.0]),
        "xgb__min_child_weight": cfg.get("min_child_weight", [3, 10]),
        "xgb__reg_alpha":        cfg.get("reg_alpha",        [0, 0.1]),
        "xgb__reg_lambda":       cfg.get("reg_lambda",       [1, 5]),
        "xgb__gamma":            cfg.get("gamma",            [0]),
    }
    try:
        run_manager = (
            managed_run(mlflow_ctx, run_name=f"{mlflow_ctx.parent_run_name}__xgboost", nested=True, tags={"estimator": "xgboost"})
            if mlflow_ctx else mlflow.start_run(run_name=f"XGBoost_{target_mode}")
        )
        with run_manager:
            log_common_params({
                "model": "XGBoost", "target": target_mode,
                "cv_folds": cv_folds, "eval_mode": eval_mode,
                "xgb_n_jobs": xgb_n_jobs, "grid_n_jobs": grid_n_jobs,
                "class_weight": str(class_weight),
                # Use grid_ prefix so these don't collide with best-param keys logged later
                **{f"grid_{k.replace('xgb__','')}": str(v) for k, v in param_grid.items()},
                "random_seed": RANDOM_STATE,
                "n_train": len(X_train), "n_test": len(X_test),
                "use_draft_pick": use_draft_pick,
            })
            xgb_extra = (
                {"objective": "multi:softprob", "num_class": len(TIER_NAMES), "eval_metric": "mlogloss"}
                if is_multiclass
                else {"objective": "binary:logistic", "eval_metric": "logloss"}
            )
            xgb_base = Pipeline([
                ("preprocessor", clone(preprocessor)),
                ("xgb", XGBClassifier(
                    random_state=RANDOM_STATE, n_jobs=xgb_n_jobs, verbosity=0,
                    **xgb_extra,
                )),
            ])
            gs = GridSearchCV(
                xgb_base, param_grid, cv=cv_folds,
                scoring="f1_macro" if is_multiclass else "roc_auc",
                n_jobs=grid_n_jobs, pre_dispatch=pre_dispatch,
            )

            # Sample-weight class balancing for XGBoost (multiclass scale_pos_weight is insufficient)
            if class_weight == "balanced":
                sw = compute_sample_weight("balanced", y_train)
                gs.fit(X_train, y_train, xgb__sample_weight=sw)
            else:
                gs.fit(X_train, y_train)

            best = gs.best_estimator_
            y_pred_xgb = best.predict(X_test)
            proba_xgb  = best.predict_proba(X_test)
            y_prob_xgb = proba_xgb if is_multiclass else proba_xgb[:, 1]
            metrics_xgb = classification_metrics(y_test, y_pred_xgb, y_prob_xgb, class_names=class_names)

            # Threshold tuning
            if threshold_tuning and X_val is not None and is_multiclass:
                proba_val  = best.predict_proba(X_val)
                offsets    = _tune_thresholds(np.array(proba_val), np.array(y_val), proba_xgb.shape[1])
                y_pred_t   = np.argmax(proba_xgb + offsets, axis=1)
                f1_tuned   = float(f1_score(y_test, y_pred_t, average="macro", zero_division=0))
                mlflow.log_metrics({"default_f1_macro": metrics_xgb.get("f1_macro", 0.0), "tuned_f1_macro": f1_tuned})
                print(f"  XGBoost threshold tuning: default F1={metrics_xgb.get('f1_macro',0):.4f}  tuned F1={f1_tuned:.4f}")

            log_common_params({
                **{k.replace("xgb__", ""): v for k, v in gs.best_params_.items()},
                "best_cv_score": round(float(gs.best_score_), 4),
            })
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
                "balanced_accuracy": metrics_xgb.get("balanced_accuracy", 0.0),
                "C": None,
                "best_cv_score": round(float(gs.best_score_), 4),
                "importance_kind": "xgb",
                "estimator_step": "xgb",
            }

            print(f"{'='*40}")
            print(f"  XGBoost  (best: {gs.best_params_})")
            print(f"  Accuracy     = {metrics_xgb['accuracy']:.4f}")
            print(f"  Balanced Acc = {metrics_xgb['balanced_accuracy']:.4f}")
            if is_multiclass:
                print(f"  F1-macro     = {metrics_xgb['f1_macro']:.4f}")
            print(f"  AUC          = {metrics_xgb['auc']:.4f}")
            if is_multiclass:
                for cls in TIER_CLASS_NAMES:
                    r = metrics_xgb.get(f"recall_{cls}", 0.0)
                    p = metrics_xgb.get(f"precision_{cls}", 0.0)
                    print(f"  {cls:<12}: precision={p:.3f}  recall={r:.3f}")
            print(classification_report(y_test, y_pred_xgb, target_names=tier_labels, zero_division=0))
            print(f"\n{'='*40}\n  XGBoost Feature Importances (top 10)\n{'='*40}")
            print_xgb_importances(best, numeric_cols, categorical_cols, ordinal_cols)

    except Exception as exc:
        import traceback
        print(f"\n[WARNING] XGBoost training failed and will be skipped: {exc}")
        traceback.print_exc()



# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_results(results, y_test, col_info, target_mode=TARGET_MODE, plot_dir=ARTIFACT_DIR):
    _plot_classification(results, y_test, target_mode, plot_dir=plot_dir)
    plot_feature_importance(results, col_info, target_mode, artifact_dir=plot_dir)
    plot_model_summary(results, target_mode, "classification", artifact_dir=plot_dir)


def _plot_classification(results, y_test, target_mode, plot_dir):
    # Filter out models with no test predictions (e.g. repeated_cv final fits)
    plot_results_map = {name: res for name, res in results.items() if res.get("y_pred") is not None}
    if not plot_results_map:
        return

    is_multiclass  = target_mode == "prospect_tier"
    display_labels = TIER_NAMES if is_multiclass else ["No", "Yes"]

    n = len(plot_results_map)
    fig_w = 6 * n if is_multiclass else 5 * n
    fig, axes = plt.subplots(2, n, figsize=(fig_w, 10 if is_multiclass else 9))
    if n == 1:
        axes = [[axes[0]], [axes[1]]]
    fig.suptitle(f"NBA {target_mode} Classification from College Stats", fontsize=13, fontweight="bold")

    y_test_arr = np.asarray(y_test)

    for col_idx, (name, res) in enumerate(plot_results_map.items()):
        f1_macro = res.get("f1_macro", 0.0)
        ba       = res.get("balanced_accuracy", 0.0)
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            res["y_pred"],
            display_labels=display_labels,
            ax=axes[0][col_idx],
        )
        axes[0][col_idx].set_title(
            f"{name}\nAcc={res['accuracy']:.3f}  F1={f1_macro:.3f}  BalAcc={ba:.3f}"
        )

        if is_multiclass:
            ax = axes[1][col_idx]
            x = np.arange(len(TIER_NAMES))
            width = 0.35
            actual_counts = [int(np.sum(y_test_arr == i)) for i in range(len(TIER_NAMES))]
            pred_counts   = [int(np.sum(res["y_pred"] == i)) for i in range(len(TIER_NAMES))]
            ax.bar(x - width / 2, actual_counts, width, label="Actual",    color="#3498db")
            ax.bar(x + width / 2, pred_counts,   width, label="Predicted", color="#e67e22")
            ax.set_xticks(x)
            ax.set_xticklabels(TIER_NAMES)
            ax.set_ylabel("Count")
            ax.set_title(f"Tier Distribution\nAUC-OvR={res['auc']:.3f}  F1-macro={f1_macro:.3f}")
            ax.legend()
        else:
            RocCurveDisplay.from_predictions(y_test, res["y_prob"], ax=axes[1][col_idx], name=name)
            axes[1][col_idx].set_title(f"ROC  AUC={res['auc']:.3f}")

    plt.tight_layout()
    save_and_log(
        fig,
        "classification_results.png",
        {name: {"accuracy": res["accuracy"], "auc": res["auc"]} for name, res in plot_results_map.items()},
        artifact_dir=plot_dir,
    )


# ── Entry point ────────────────────────────────────────────────────────────────

def run(target_mode=TARGET_MODE, use_draft_pick=USE_DRAFT_PICK, df=None, cfg=None, run_name=None, tracking_uri=None):
    model_cfg             = (cfg or {}).get("model", {}) or {}
    composite_cfg         = model_cfg.get("composite_score") or {}
    clf_cfg               = model_cfg.get("classification", {}) or {}
    xgb_cfg               = clf_cfg.get("xgboost") or {}
    prospect_context_mode = model_cfg.get("prospect_context_mode", PROSPECT_CONTEXT_MODE)
    target_score_mode     = (model_cfg.get("nba_role_score") or {}).get("target_score_mode", "global")
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
        log_common_params({
            "model_family": "classification",
            "target": target_mode,
            "target_score_mode": target_score_mode,
            "use_draft_pick": use_draft_pick,
            "eval_mode": clf_cfg.get("eval_mode", "random"),
            "class_weight": str(clf_cfg.get("class_weight")),
            "threshold_tuning": clf_cfg.get("threshold_tuning", False),
        })
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
            clf_cfg=clf_cfg,
            prospect_context_mode=prospect_context_mode,
        )
        log_candidate_summary(results, task="classification", target_mode=target_mode)
        plot_results(results, y_test, col_info, target_mode=target_mode, plot_dir=mlflow_ctx.plot_dir)
    return results, y_test, col_info


if __name__ == "__main__":
    import yaml
    _cfg_path = os.path.join(PROJECT_ROOT, "src", "config", "config.yaml")
    with open(_cfg_path) as _f:
        _cfg = yaml.safe_load(_f)
    _clf_cfg = (_cfg.get("model", {}) or {}).get("classification", {}) or {}
    run(
        target_mode=_clf_cfg.get("target_mode", TARGET_MODE),
        use_draft_pick=_clf_cfg.get("use_draft_pick", USE_DRAFT_PICK),
        cfg=_cfg,
    )
