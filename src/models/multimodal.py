"""Multimodal orchestrator for the Probability Stacking Model (PSM Phase 1).

Trains classification and regression base models via out-of-fold cross-validation,
stacks their tier probabilities with a logistic meta-model, and exposes a unified
predict_proba interface for downstream use.
"""
from __future__ import annotations

import copy
import os

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from src.data.loader import (
    PROJECT_ROOT,
    PROSPECT_CONTEXT_MODE,
    RANDOM_STATE,
    TARGET_COL,
    CLASSIFICATION_EXCLUDED_NUMERIC,
    build_feature_matrix,
    load_data,
)
from src.training.splits import get_chronological_split
from src.models.classification_model import (
    CLASSIFICATION_MODEL_REGISTRY,
    train_selected_classification_models,
)
from src.models.probability import PROBA_COLUMNS, TIER_CLASS_NAMES, BaseModelBundle
from src.models.regression_model import (
    REGRESSION_MODEL_REGISTRY,
    train_selected_regression_models,
)
from src.models.multimodal_reporting import write_multimodal_report
from src.training.evaluate import ordinal_classification_metrics
from src.utils.mlflow_utils import (
    build_mlflow_context,
    log_common_params,
    log_config_dict,
    log_data_summary,
    log_reproducibility_metadata,
    managed_run,
)


MLFLOW_EXPERIMENT = "nba-draft-prospect-multimodal"
CLF_TARGET_COL = TARGET_COL["prospect_tier"]
REG_TARGET_COL = TARGET_COL["nba_role_zscore"]
TIER_NAMES = ["Bust", "Bench", "Starter", "Star"]


def _build_meta_cols(clf_models: list[str], reg_models: list[str]) -> list[str]:
    cols = []
    for key in clf_models:
        for c in PROBA_COLUMNS:
            cols.append(f"classification__{key}__{c}")
    for key in reg_models:
        for c in PROBA_COLUMNS:
            cols.append(f"regression__{key}__{c}")
    return cols


def _append_bundle_proba(
    meta_df: pd.DataFrame,
    bundles: dict[str, BaseModelBundle],
    task: str,
    models: list[str],
    df: pd.DataFrame,
) -> None:
    for key in models:
        proba_df = bundles[key].predict_tier_proba(df)
        for col in PROBA_COLUMNS:
            meta_df.loc[df.index, f"{task}__{key}__{col}"] = proba_df[col].values


def _make_task_cfg(cfg: dict, task: str, mm_cfg: dict) -> dict:
    out = copy.deepcopy(cfg)
    ue = mm_cfg.get("use_engineered_features") or {}
    up = mm_cfg.get("use_pos_categorical") or {}
    section = out.setdefault("model", {}).setdefault(task, {})
    section["use_engineered_features"] = bool(ue.get(task, False))
    section["use_pos_categorical"] = bool(up.get(task, False))
    return out


class MultimodalProspectModel:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        mm_cfg = (cfg.get("model", {}) or {}).get("multimodal", {}) or {}
        self._mm_cfg = mm_cfg

        self.cv_folds = int(mm_cfg.get("cv_folds", 5))
        self.output_dir = str(mm_cfg.get("output_dir", "outputs/multimodal"))

        base_models = mm_cfg.get("base_models", {}) or {}
        self.clf_models: list[str] = list(base_models.get("classification", ["logistic_l2", "xgboost"]))
        self.reg_models: list[str] = list(base_models.get("regression", ["ridge", "xgboost"]))

        cal_cfg = mm_cfg.get("calibration", {}) or {}
        self.cal_method = cal_cfg.get("method", "sigmoid")
        self.cal_size = float(cal_cfg.get("calibration_size", 0.15))

        xgb_cfg = mm_cfg.get("xgboost", {}) or {}
        self.pretune_oof = bool(xgb_cfg.get("pretune_oof_params", False))

        self.meta_cols: list[str] = _build_meta_cols(self.clf_models, self.reg_models)

        # Set by fit()
        self.final_clf_bundles: dict[str, BaseModelBundle] = {}
        self.final_reg_bundles: dict[str, BaseModelBundle] = {}
        self.stacker: LogisticRegression | None = None

    def fit(self, train_df: pd.DataFrame, mlflow_ctx=None) -> None:
        clf_cfg = _make_task_cfg(self.cfg, "classification", self._mm_cfg)
        reg_cfg = _make_task_cfg(self.cfg, "regression", self._mm_cfg)

        if self.pretune_oof:
            self._pretune_xgb(train_df)

        meta_train = pd.DataFrame(0.0, index=train_df.index, columns=self.meta_cols)
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=RANDOM_STATE)
        y_all = train_df[CLF_TARGET_COL]

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_df, y_all)):
            fold_full = train_df.iloc[train_idx].copy()
            fold_val  = train_df.iloc[val_idx].copy()

            fold_core, cal_df = train_test_split(
                fold_full, test_size=self.cal_size,
                stratify=fold_full[CLF_TARGET_COL], random_state=RANDOM_STATE,
            )

            clf_bundles = train_selected_classification_models(
                fold_core, CLF_TARGET_COL, clf_cfg,
                selected_models=self.clf_models,
                calibration_df=cal_df,
                use_fixed_xgb_params=True,
                mlflow_ctx=None,
            )
            reg_bundles = train_selected_regression_models(
                fold_core, REG_TARGET_COL, reg_cfg,
                selected_models=self.reg_models,
                use_fixed_xgb_params=True,
                mlflow_ctx=None,
            )

            _append_bundle_proba(meta_train, clf_bundles, "classification", self.clf_models, fold_val)
            _append_bundle_proba(meta_train, reg_bundles, "regression",     self.reg_models, fold_val)

        self.stacker = LogisticRegression(class_weight="balanced", max_iter=5000)
        self.stacker.fit(meta_train.values, y_all)

        final_core, final_cal = train_test_split(
            train_df, test_size=self.cal_size,
            stratify=train_df[CLF_TARGET_COL], random_state=RANDOM_STATE,
        )

        self.final_clf_bundles = train_selected_classification_models(
            final_core, CLF_TARGET_COL, clf_cfg,
            selected_models=self.clf_models,
            calibration_df=final_cal,
            use_fixed_xgb_params=True,
            mlflow_ctx=mlflow_ctx,
        )
        self.final_reg_bundles = train_selected_regression_models(
            final_core, REG_TARGET_COL, reg_cfg,
            selected_models=self.reg_models,
            use_fixed_xgb_params=True,
            mlflow_ctx=mlflow_ctx,
        )

    def meta_features(self, df: pd.DataFrame) -> pd.DataFrame:
        parts: list[pd.DataFrame] = []
        for key in self.clf_models:
            proba = self.final_clf_bundles[key].predict_tier_proba(df)
            parts.append(proba.rename(columns={c: f"classification__{key}__{c}" for c in PROBA_COLUMNS}))
        for key in self.reg_models:
            proba = self.final_reg_bundles[key].predict_tier_proba(df)
            parts.append(proba.rename(columns={c: f"regression__{key}__{c}" for c in PROBA_COLUMNS}))
        return pd.concat(parts, axis=1)[self.meta_cols]

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.stacker.predict_proba(self.meta_features(df).values)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.argmax(self.predict_proba(df), axis=1)

    def _pretune_xgb(self, train_df: pd.DataFrame) -> None:
        from sklearn.base import clone
        from xgboost import XGBClassifier  # noqa: PLC0415

        clf_cfg = _make_task_cfg(self.cfg, "classification", self._mm_cfg)
        model_cfg = clf_cfg.get("model", {}) or {}
        section = model_cfg.get("classification", {}) or {}
        xgb_cfg = section.get("xgboost", {}) or {}

        preprocessor, numeric_cols, categorical_cols, ordinal_cols, passthrough_cols = build_feature_matrix(
            train_df,
            use_draft_pick=bool(section.get("use_draft_pick", False)),
            exclude_features=CLASSIFICATION_EXCLUDED_NUMERIC,
            prospect_context_mode=model_cfg.get("prospect_context_mode", PROSPECT_CONTEXT_MODE),
            use_engineered_features=bool(section.get("use_engineered_features", False)),
            use_pos_categorical=bool(section.get("use_pos_categorical", False)),
            input_normalization_mode=model_cfg.get("input_normalization_mode", "global"),
        )
        feature_cols = numeric_cols + categorical_cols + ordinal_cols + passthrough_cols
        X = train_df[feature_cols]
        y = train_df[CLF_TARGET_COL]
        n_classes = len(sorted(y.unique()))

        cv_folds     = xgb_cfg.get("cv_folds", 3)
        xgb_n_jobs   = xgb_cfg.get("n_jobs", 1)
        grid_n_jobs  = xgb_cfg.get("grid_n_jobs", 1)
        pre_dispatch = xgb_cfg.get("pre_dispatch", 1)

        xgb_extra = {
            "objective": "multi:softprob",
            "num_class": n_classes,
            "eval_metric": "mlogloss",
        }
        param_grid = {
            "xgb__n_estimators":     xgb_cfg.get("n_estimators",     [100, 200]),
            "xgb__max_depth":        xgb_cfg.get("max_depth",        [2, 3]),
            "xgb__learning_rate":    xgb_cfg.get("learning_rate",    [0.05, 0.1]),
            "xgb__subsample":        xgb_cfg.get("subsample",        [0.7, 0.9]),
            "xgb__colsample_bytree": xgb_cfg.get("colsample_bytree", [1.0]),
            "xgb__min_child_weight": xgb_cfg.get("min_child_weight", [3, 10]),
            "xgb__reg_alpha":        xgb_cfg.get("reg_alpha",        [0, 0.1]),
            "xgb__reg_lambda":       xgb_cfg.get("reg_lambda",       [1, 5]),
            "xgb__gamma":            xgb_cfg.get("gamma",            [0]),
        }
        pipe = Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("xgb", XGBClassifier(
                random_state=RANDOM_STATE, n_jobs=xgb_n_jobs, verbosity=0, **xgb_extra,
            )),
        ])
        gs = GridSearchCV(
            pipe, param_grid, cv=cv_folds, scoring="f1_macro",
            n_jobs=grid_n_jobs, pre_dispatch=pre_dispatch,
        )
        gs.fit(X, y)

        best = {k.replace("xgb__", ""): v for k, v in gs.best_params_.items()}
        (
            self.cfg
            .setdefault("model", {})
            .setdefault("multimodal", {})
            .setdefault("xgboost", {})
        )["oof_params"] = best
        print(f"[multimodal] pre-tuned XGBoost oof_params: {best}  (cv_score={gs.best_score_:.4f})")


def run(df=None, cfg=None, run_name=None, tracking_uri=None):
    model_cfg     = (cfg.get("model", {}) or {})
    mm_cfg        = model_cfg.get("multimodal", {}) or {}
    composite_cfg = model_cfg.get("composite_score") or {}

    df = load_data(composite_cfg=composite_cfg) if df is None else df

    chron_train, chron_val, test_df = get_chronological_split(df)
    train_df = pd.concat([chron_train, chron_val], ignore_index=True)
    actual_test_size = len(test_df) / len(df)

    mlflow_ctx = build_mlflow_context(
        cfg=cfg,
        model_type="multimodal",
        target_name="prospect_tier",
        fallback_experiment_name=MLFLOW_EXPERIMENT,
        tracking_uri=tracking_uri,
        run_name=run_name,
    )

    model = MultimodalProspectModel(cfg)

    with managed_run(mlflow_ctx):
        if cfg is not None:
            log_config_dict(cfg)
        log_common_params({
            "model_family": "multimodal",
            "target":       "prospect_tier",
            "clf_models":   str(model.clf_models),
            "reg_models":   str(model.reg_models),
            "cal_size":     model.cal_size,
            "pretune_oof":  model.pretune_oof,
            "meta_cols":    str(model.meta_cols),
        })
        log_data_summary(
            df, target_col=CLF_TARGET_COL, task="classification",
            test_size=actual_test_size, cv_folds=model.cv_folds, random_seed=RANDOM_STATE,
        )
        log_reproducibility_metadata(device="cpu")

        model.fit(train_df, mlflow_ctx=mlflow_ctx)

        stacker_run_mgr = managed_run(
            mlflow_ctx,
            run_name=f"{mlflow_ctx.parent_run_name}__stacker",
            nested=True,
            tags={"estimator": "logistic_stacker"},
        )
        with stacker_run_mgr:
            log_common_params({
                "stacker":         "LogisticRegression",
                "class_weight":    "balanced",
                "max_iter":        5000,
                "n_meta_features": len(model.meta_cols),
            })
            mlflow.sklearn.log_model(model.stacker, artifact_path="stacker")

        test_proba = model.predict_proba(test_df)
        y_test = test_df[CLF_TARGET_COL]
        y_pred = np.argmax(test_proba, axis=1)

        metrics = ordinal_classification_metrics(y_test, y_pred, test_proba, class_names=TIER_CLASS_NAMES)
        ll = float(log_loss(y_test, test_proba))
        brier = float(np.mean([
            brier_score_loss((y_test == i).astype(int), test_proba[:, i])
            for i in range(4)
        ]))
        all_metrics = {**metrics, "log_loss": ll, "brier_score": brier}
        mlflow.log_metrics(all_metrics)

        print(f"\n{'='*50}\n  Multimodal Stacker Results\n{'='*50}")
        priority_metrics = [
            "top1_accuracy",
            "within_one_accuracy",
            "distance_weighted_accuracy",
            "ordinal_mae",
            "expected_class_mae",
            "quadratic_weighted_kappa",
            "f1_macro",
        ]
        printed = set()
        for key in priority_metrics:
            if key in all_metrics:
                print(f"  {key:<35} = {all_metrics[key]:.4f}")
                printed.add(key)
        for k, v in all_metrics.items():
            if k not in printed:
                print(f"  {k:<35} = {v:.4f}")

        out_dir = os.path.join(model.output_dir, run_name or "latest")
        write_multimodal_report(
            model=model,
            test_df=test_df,
            y_test=y_test,
            y_pred=y_pred,
            test_proba=test_proba,
            out_dir=out_dir,
        )

        print(f"[multimodal] outputs written to {out_dir}/")

    return model, y_test
