# PSM Phase 1: Reusable Tabular Probability Stacking

## Summary

Implement Phase 1 of the Probability Stacking Model as a direct step toward
`PROBABILITY_STACKING_MODEL.md`.

This phase builds the tabular foundation for the final model-agnostic stacking
architecture:

- regression and classification models remain independently trainable and improvable
- both model families expose a shared 4-class probability interface
- `multimodal.py` consumes selected base models from those families
- the first meta-model stacks tabular probabilities only
- text / DistilBERT integration is deferred to a later phase

The 4-class probability contract is:

```text
p_bust
p_bench
p_starter
p_star
pred_tier
confidence
```

Class order is always:

```python
["bust", "bench", "starter", "star"]
```

---

## Status

| Chunk | Scope | Status |
|-------|-------|--------|
| **Chunk 1** | 4-Class Tier Migration | ✅ DONE |
| **Chunk 2** | Shared Probability Layer (`probability.py`) | ✅ DONE |
| **Chunk 3** | Classification Model Refactor | ✅ DONE |
| **Chunk 4** | Regression Model Refactor | ✅ DONE |
| **Chunk 5** | Multimodal Orchestrator (`multimodal.py`) | ✅ DONE |
| **Chunk 6** | Wiring & Final Verification (`main.py`) | ✅ DONE |

---

## Implementation Chunks

Work through these chunks in order. After each chunk, run the acceptance gate and
confirm before starting the next.

---

### Chunk 1 — 4-Class Tier Migration ✅ DONE

**Acceptance gate passed.**

Files changed:
- `src/data/loader.py` — percentile tier path removed; z-score thresholds `[-0.5, 0.5, 1.5]`
  are the only `prospect_tier` definition; `_assign_tier_thresholded` is canonical
- `src/models/classification_inference.py` — `TIER_CLASS_NAMES = ["bust", "bench", "starter", "star"]`;
  probability columns renamed to `p_bust`, `p_bench`, `p_starter`, `p_star`
- `scripts/check_classification_contract.py` — expects 4 classes `[0,1,2,3]`, 4 probability
  columns, unpacks 5 values from `build_feature_matrix`
- `PROBABLITY_STACKING_MODEL.md` → `PROBABILITY_STACKING_MODEL.md` (renamed)
- All `build_feature_matrix` callers updated to unpack 5 values

---

### Chunk 2 — Shared Probability Layer ✅ DONE

**Acceptance gate passed.**

File created: `src/models/probability.py`

Contents:
- `TIER_CLASS_NAMES`, `TIER_LABELS`, `TIER_THRESHOLDS`, `PROBA_COLUMNS` constants
- `normalize_proba(proba)` — clips negatives, normalizes rows to sum 1.0
- `proba_to_dataframe(proba, index)` — converts `(N, 4)` array to `PROBA_COLUMNS` DataFrame
- `zscore_to_tier_proba(z_scores, residual_std, thresholds)` — Gaussian CDF conversion
- `build_prefit_calibrator(base_estimator, cal_X, cal_y, method)` — uses `FrozenEstimator`
- `BaseModelBundle` dataclass — `predict_tier_proba(df)` returns normalized `PROBA_COLUMNS` DataFrame;
  raises for regression bundles with `residual_std <= 0.0`

---

### Chunk 3 — Classification Model Refactor ✅ DONE

**Acceptance gate passed.**

Files changed: `src/models/classification_model.py`, `src/config/config.yaml`

Key additions:
- `CLASSIFICATION_MODEL_REGISTRY = ["logistic_l1", "logistic_l2", "xgboost"]`
- `_first(x)`, `_get_oof_xgb_params(cfg)` helpers
- `train_selected_classification_models(train_df, target_col, cfg, selected_models=None, calibration_df=None, use_fixed_xgb_params=False, mlflow_ctx=None, return_bundles=True)` — never creates its own calibration split; uses `nullcontext()` when `mlflow_ctx=None`
- `run()` rewritten to own calibration split and call `train_selected_classification_models`
- `_DISPLAY_NAMES` for backward-compat plotting keys
- Config additions: `model.classification.selected_models`, `model.classification.calibration.*`

---

### Chunk 4 — Regression Model Refactor ✅ DONE

**Acceptance gate passed.**

Files changed: `src/models/regression_model.py`, `src/config/config.yaml`

Key additions:
- `REGRESSION_MODEL_REGISTRY = ["lasso", "ridge", "xgboost"]`
- `_DISPLAY_NAMES = {"lasso": "Lasso", "ridge": "Ridge", "xgboost": "XGBoost"}`
- `_first(x)`, `_get_oof_xgb_params_regression(cfg)` helpers
- `train_selected_regression_models(train_df, target_col, cfg, selected_models=None, use_fixed_xgb_params=False, mlflow_ctx=None, return_bundles=True)` — computes `residual_std` from training residuals when target is `nba_role_zscore`; sets `residual_std=0.0` for any other target (guard); uses `nullcontext()` when `mlflow_ctx=None`
- `run()` rewritten to own the train/test split and call `train_selected_regression_models`; logs model-prefixed test metrics to parent MLflow run
- Pipeline step names: `"lasso"`, `"ridge"`, `"xgb"` (for XGBoost) — used for `estimator_step` in results dict and `bundle.estimator.named_steps` access
- Config additions: `model.regression.selected_models`, `model.regression.probability_target_mode`

---

### Chunk 5 — Multimodal Orchestrator ✅ DONE

**Acceptance gate passed.**

Files changed:
- `src/models/multimodal.py` *(new)* — `_build_meta_cols`, `_append_bundle_proba`, `_make_task_cfg` helpers;
  `MultimodalProspectModel` class with `fit`, `meta_features`, `predict_proba`, `predict`, `_pretune_xgb`;
  `run()` top-level entry point
- `src/config/config.yaml` — `model.multimodal.*` block added

---

### Chunk 5 — Multimodal Orchestrator (archived spec) ⏳ PENDING

**Files:**
- `src/models/multimodal.py` *(new)*
- `src/config/config.yaml` — add `model.multimodal.*` block

**Acceptance gate:**
```bash
uv run python -m compileall src/models/multimodal.py
```
Plus visual review of OOF/MLflow logic before proceeding to Chunk 6.

---

#### 5.1 Config Block to Add

Add this block to `src/config/config.yaml` under `model:`:

```yaml
model:
  multimodal:
    target_mode: prospect_tier
    cv_folds: 5
    meta_model: logistic_regression
    output_dir: outputs/multimodal
    base_models:
      classification: ["logistic_l2", "xgboost"]
      regression: ["ridge", "xgboost"]
    calibration:
      method: sigmoid
      calibration_size: 0.15
    use_engineered_features:
      classification: false
      regression: false
    use_pos_categorical:
      classification: false
      regression: true
    xgboost:
      pretune_oof_params: false
      oof_params:
        n_estimators: 200
        max_depth: 3
        learning_rate: 0.05
        subsample: 0.8
        colsample_bytree: 1.0
        min_child_weight: 3
        reg_alpha: 0.0
        reg_lambda: 1.0
        gamma: 0.0
```

---

#### 5.2 Imports

```python
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
    TEST_SIZE,
    CLASSIFICATION_EXCLUDED_NUMERIC,
    build_feature_matrix,
    load_data,
)
from src.models.classification_model import (
    CLASSIFICATION_MODEL_REGISTRY,
    train_selected_classification_models,
)
from src.models.probability import PROBA_COLUMNS, TIER_CLASS_NAMES, BaseModelBundle
from src.models.regression_model import (
    REGRESSION_MODEL_REGISTRY,
    train_selected_regression_models,
)
from src.training.evaluate import classification_metrics
from src.utils.mlflow_utils import (
    build_mlflow_context,
    log_common_params,
    log_config_dict,
    log_data_summary,
    log_reproducibility_metadata,
    managed_run,
)
```

#### 5.3 Module-Level Constants

```python
MLFLOW_EXPERIMENT = "nba-draft-prospect-multimodal"
CLF_TARGET_COL = TARGET_COL["prospect_tier"]
REG_TARGET_COL = TARGET_COL["nba_role_zscore"]
TIER_NAMES = ["Bust", "Bench", "Starter", "Star"]
```

---

#### 5.4 Module-Level Helpers

##### `_build_meta_cols(clf_models, reg_models) -> list[str]`

Returns the ordered list of meta-feature column names used throughout the class.
Order: all classification model probabilities first, then all regression model probabilities.
Within each model, order follows `PROBA_COLUMNS`.

```python
def _build_meta_cols(clf_models: list[str], reg_models: list[str]) -> list[str]:
    cols = []
    for key in clf_models:
        for c in PROBA_COLUMNS:
            cols.append(f"classification__{key}__{c}")
    for key in reg_models:
        for c in PROBA_COLUMNS:
            cols.append(f"regression__{key}__{c}")
    return cols
```

##### `_append_bundle_proba(meta_df, bundles, task, models, df)`

Fills `meta_df` in-place with tier probabilities from each bundle for the rows in `df`.
`meta_df` must already have the correct columns (built by `_build_meta_cols`).
`df.index` must be a subset of `meta_df.index`.

```python
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
```

##### `_make_task_cfg(cfg, task, mm_cfg) -> dict`

Returns a deep copy of `cfg` with per-task `use_engineered_features` and
`use_pos_categorical` overridden from the multimodal config block.
All other classification/regression settings (class_weight, use_draft_pick,
xgboost grid, calibration, selected_models, etc.) are preserved from the
base model configs.

```python
def _make_task_cfg(cfg: dict, task: str, mm_cfg: dict) -> dict:
    out = copy.deepcopy(cfg)
    ue = mm_cfg.get("use_engineered_features") or {}
    up = mm_cfg.get("use_pos_categorical") or {}
    section = out.setdefault("model", {}).setdefault(task, {})
    section["use_engineered_features"] = bool(ue.get(task, False))
    section["use_pos_categorical"] = bool(up.get(task, False))
    return out
```

Note: `use_draft_pick` is NOT overridden here. Classification inherits its value
from `cfg["model"]["classification"]["use_draft_pick"]` (false by default).
Regression inherits from `cfg["model"]["regression"]["use_draft_pick"]` (true).

---

#### 5.5 `MultimodalProspectModel` Class

##### `__init__(self, cfg)`

Parse and cache all settings from config. Initialize state containers to empty.

```python
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
```

##### `fit(self, train_df, mlflow_ctx=None)`

Orchestrates the full training flow. `run()` passes `mlflow_ctx` for the final base
model training only — OOF fold training always uses `mlflow_ctx=None`.

**Step-by-step:**

1. Build per-task cfgs using `_make_task_cfg` (call once before the OOF loop, not
   inside it, since the config is static).

2. **Optional pre-tuning (Strategy B):** if `self.pretune_oof`, call `self._pretune_xgb(train_df)`.
   This mutates `self.cfg["model"]["multimodal"]["xgboost"]["oof_params"]` in-place so that
   subsequent calls to `_get_oof_xgb_params` / `_get_oof_xgb_params_regression` pick up
   the tuned values automatically.

3. **OOF loop:** Initialize `meta_train` as a zero-filled DataFrame with index
   `train_df.index` and columns `self.meta_cols`.

   ```python
   meta_train = pd.DataFrame(0.0, index=train_df.index, columns=self.meta_cols)
   skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=RANDOM_STATE)
   y_all = train_df[CLF_TARGET_COL]

   for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_df, y_all)):
       fold_full = train_df.iloc[train_idx].copy()
       fold_val  = train_df.iloc[val_idx].copy()

       # Calibration carved from fold training portion
       fold_core, cal_df = train_test_split(
           fold_full, test_size=self.cal_size,
           stratify=fold_full[CLF_TARGET_COL], random_state=RANDOM_STATE,
       )

       # Train base models — no MLflow
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

       # Write OOF probabilities into meta_train
       _append_bundle_proba(meta_train, clf_bundles, "classification", self.clf_models, fold_val)
       _append_bundle_proba(meta_train, reg_bundles, "regression",     self.reg_models, fold_val)
   ```

   `fold_val` is a slice of the loaded DataFrame and contains all raw feature columns,
   so `bundle.predict_tier_proba(fold_val)` can select `bundle.feature_cols` from it. ✓

4. **Train stacker on OOF meta-features:**

   ```python
   self.stacker = LogisticRegression(class_weight="balanced", max_iter=5000)
   self.stacker.fit(meta_train.values, y_all)
   ```

   The stacker is trained ONCE on OOF predictions and is NOT retrained after the final
   base model fit.

5. **Final base model training** (runs full GridSearchCV; creates MLflow nested runs if
   `mlflow_ctx` is not None):

   Carve a calibration set from the full `train_df` for the final classification fit:

   ```python
   final_core, final_cal = train_test_split(
       train_df, test_size=self.cal_size,
       stratify=train_df[CLF_TARGET_COL], random_state=RANDOM_STATE,
   )

   self.final_clf_bundles = train_selected_classification_models(
       final_core, CLF_TARGET_COL, clf_cfg,
       selected_models=self.clf_models,
       calibration_df=final_cal,
       use_fixed_xgb_params=False,
       mlflow_ctx=mlflow_ctx,
   )
   self.final_reg_bundles = train_selected_regression_models(
       final_core, REG_TARGET_COL, reg_cfg,
       selected_models=self.reg_models,
       use_fixed_xgb_params=False,
       mlflow_ctx=mlflow_ctx,
   )
   ```

   Note on MLflow run naming: both classification and regression call `managed_run`
   with `run_name=f"{mlflow_ctx.parent_run_name}__{key}"`. If both families include
   `"xgboost"`, two nested runs will share the name `{parent}__xgboost` (different
   run IDs, same name). This is acceptable in MLflow. To disambiguate, the implementer
   may prefix names as `{parent}__clf__{key}` / `{parent}__reg__{key}` by modifying
   the `mlflow_ctx.parent_run_name` passed in, but this is optional.

##### `meta_features(self, df) -> pd.DataFrame`

Generate meta-feature DataFrame from the final fitted bundles. Returns a DataFrame
with `self.meta_cols` columns in the canonical order.

```python
def meta_features(self, df: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for key in self.clf_models:
        proba = self.final_clf_bundles[key].predict_tier_proba(df)
        parts.append(proba.rename(columns={c: f"classification__{key}__{c}" for c in PROBA_COLUMNS}))
    for key in self.reg_models:
        proba = self.final_reg_bundles[key].predict_tier_proba(df)
        parts.append(proba.rename(columns={c: f"regression__{key}__{c}" for c in PROBA_COLUMNS}))
    return pd.concat(parts, axis=1)[self.meta_cols]
```

The `[self.meta_cols]` reindex at the end enforces canonical column order and guards
against any concat ordering surprises.

##### `predict_proba(self, df) -> np.ndarray`

Returns `(N, 4)` stacker probability array.

```python
def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
    return self.stacker.predict_proba(self.meta_features(df).values)
```

##### `predict(self, df) -> np.ndarray`

Returns `(N,)` integer class labels `[0, 1, 2, 3]`.

```python
def predict(self, df: pd.DataFrame) -> np.ndarray:
    return np.argmax(self.predict_proba(df), axis=1)
```

##### `_pretune_xgb(self, train_df)`

Runs one GridSearchCV for XGBoost classification on the full `train_df`. Writes the
best params dict to `self.cfg["model"]["multimodal"]["xgboost"]["oof_params"]` so that
subsequent calls to `_get_oof_xgb_params` / `_get_oof_xgb_params_regression` in both
training functions pick them up automatically (those helpers check this path first).

No MLflow logging. Print best params.

Implementation outline:

```python
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

    cv_folds    = xgb_cfg.get("cv_folds", 3)
    xgb_n_jobs  = xgb_cfg.get("n_jobs", 1)
    grid_n_jobs = xgb_cfg.get("grid_n_jobs", 1)
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
```

---

#### 5.6 `run(df=None, cfg=None, run_name=None, tracking_uri=None)`

Top-level entry point called by `main.py`. Owns the stratified train/test split.
Returns `(model, y_test)`.

```python
def run(df=None, cfg=None, run_name=None, tracking_uri=None):
    model_cfg     = (cfg.get("model", {}) or {})
    mm_cfg        = model_cfg.get("multimodal", {}) or {}
    composite_cfg = model_cfg.get("composite_score") or {}

    df = load_data(composite_cfg=composite_cfg) if df is None else df
    y = df[CLF_TARGET_COL]

    # Stratified split — run() owns this
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE,
    )

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
            "cv_folds":     model.cv_folds,
            "clf_models":   str(model.clf_models),
            "reg_models":   str(model.reg_models),
            "cal_size":     model.cal_size,
            "pretune_oof":  model.pretune_oof,
            "n_train":      len(train_df),
            "n_test":       len(test_df),
            "meta_cols":    str(model.meta_cols),
        })
        log_data_summary(
            df, target_col=CLF_TARGET_COL, task="classification",
            test_size=TEST_SIZE, cv_folds=model.cv_folds, random_seed=RANDOM_STATE,
        )
        log_reproducibility_metadata(device="cpu")

        # Train (OOF loop + stacker + final base models)
        model.fit(train_df, mlflow_ctx=mlflow_ctx)

        # Log stacker as nested run
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
            mlflow.sklearn.log_model(model.stacker, name="stacker")

        # Evaluate on test set
        test_proba = model.predict_proba(test_df)
        y_test = test_df[CLF_TARGET_COL]
        y_pred = np.argmax(test_proba, axis=1)

        metrics = classification_metrics(y_test, y_pred, test_proba, class_names=TIER_CLASS_NAMES)
        ll = float(log_loss(y_test, test_proba))
        brier = float(np.mean([
            brier_score_loss((y_test == i).astype(int), test_proba[:, i])
            for i in range(4)
        ]))
        all_metrics = {**metrics, "log_loss": ll, "brier_score": brier}
        mlflow.log_metrics(all_metrics)

        print(f"\n{'='*50}\n  Multimodal Stacker Results\n{'='*50}")
        for k, v in all_metrics.items():
            print(f"  {k:<35} = {v:.4f}")

        # Write output CSVs
        out_dir = os.path.join(model.output_dir, run_name or "latest")
        os.makedirs(out_dir, exist_ok=True)

        # test_predictions.csv — use whatever identity cols are present
        id_cols = [c for c in ["Name", "draft_year"] if c in test_df.columns]
        pred_df = test_df[id_cols].copy() if id_cols else pd.DataFrame(index=test_df.index)
        pred_df["actual_tier"]  = y_test.values
        pred_df["pred_tier"]    = y_pred
        pred_df["confidence"]   = test_proba.max(axis=1)
        for i, col in enumerate(PROBA_COLUMNS):
            pred_df[col] = test_proba[:, i]
        pred_df.to_csv(os.path.join(out_dir, "test_predictions.csv"), index=False)

        # test_meta_features.csv
        test_mf = model.meta_features(test_df)
        test_mf.to_csv(os.path.join(out_dir, "test_meta_features.csv"), index=True)

        # base_model_summary.csv — each final bundle evaluated on test set
        summary_rows = []
        for key in model.clf_models:
            b = model.final_clf_bundles[key]
            proba = b.predict_tier_proba(test_df)
            y_hat = np.argmax(proba.values, axis=1)
            m = classification_metrics(y_test, y_hat, proba.values, class_names=TIER_CLASS_NAMES)
            summary_rows.append({"model": f"classification__{key}", "task": "classification", **m})
        for key in model.reg_models:
            b = model.final_reg_bundles[key]
            proba = b.predict_tier_proba(test_df)
            y_hat = np.argmax(proba.values, axis=1)
            m = classification_metrics(y_test, y_hat, proba.values, class_names=TIER_CLASS_NAMES)
            summary_rows.append({"model": f"regression__{key}", "task": "regression", **m})
        pd.DataFrame(summary_rows).to_csv(os.path.join(out_dir, "base_model_summary.csv"), index=False)

        print(f"[multimodal] outputs written to {out_dir}/")

    return model, y_test
```

---

#### 5.7 Implementation Notes and Gotchas

**Target columns.** `CLF_TARGET_COL = TARGET_COL["prospect_tier"]` and
`REG_TARGET_COL = TARGET_COL["nba_role_zscore"]`. Both columns are in the loaded
DataFrame. Regression bundles require `REG_TARGET_COL` to be in `fold_core` for
`train_selected_regression_models`, and `residual_std` is valid because
`model.regression.probability_target_mode: nba_role_zscore` is set in config.

**Feature columns in `fold_val`.** `fold_val` is a `.copy()` slice of the loaded
DataFrame, which contains all raw columns. `bundle.predict_tier_proba(fold_val)` calls
`estimator.predict(fold_val[bundle.feature_cols])`. Since `feature_cols` are raw
column names (not engineered), they exist in any slice of the loaded data. ✓

**`_make_task_cfg` call placement.** Call `_make_task_cfg` once before the OOF loop,
not inside it. Both `clf_cfg` and `reg_cfg` are constants for a given `fit()` call.

**OOF meta_train initialization.** Use `pd.DataFrame(0.0, ...)` not `np.zeros` so that
`.loc[df.index, col]` assignment works correctly with pandas index alignment.

**Stacker class ordering.** `LogisticRegression.predict_proba` returns columns in the
order of `stacker.classes_`. After fitting on `y_all` (which contains integer labels
0–3), `classes_` will be `[0, 1, 2, 3]`. The stacker's output column order matches
`TIER_CLASS_NAMES`. No reordering needed. Verify with `assert list(model.stacker.classes_) == [0, 1, 2, 3]`
during testing.

**MLflow context inside `fit`.** `fit()` accepts `mlflow_ctx=None`. It passes
`mlflow_ctx=None` to all OOF fold training calls, and passes the provided `mlflow_ctx`
to the final base model training calls only. The stacker is logged in `run()`, not
`fit()`, keeping `fit()` reusable without MLflow.

**Nested run name collision.** If `clf_models` and `reg_models` both contain `"xgboost"`,
`train_selected_classification_models` and `train_selected_regression_models` will each
create a nested run named `{parent}__xgboost`. MLflow allows duplicate names (different
run IDs), so this does not cause an error. The `tags={"estimator": "xgboost"}` tag does
not disambiguate by task. If this is a concern, pass a task-prefixed string as
`mlflow_ctx.parent_run_name` when calling each training function (requires creating a
shallow copy of the context or adding a prefix parameter).

**`confidence` column.** Defined as `test_proba.max(axis=1)` — the maximum probability
across the 4 stacker output classes for each test player.

**Brier score.** Computed as the mean of four one-vs-rest Brier scores (one per class),
consistent with a multiclass Brier score interpretation.

**`classification_metrics` signature.** Check that `classification_metrics(y_test, y_pred, proba, class_names=...)` accepts a `(N, 4)` proba array for multiclass AUC. The function is in `src/training/evaluate.py` — confirm its actual signature before calling.

**Output directory.** If `run_name` is `None`, fall back to `"latest"` as the subdirectory name. The implementer may prefer using `mlflow_ctx.parent_run_name` if it is always set.

---

#### 5.8 Chunk 5 Acceptance Gate Checklist

After writing the file, verify:

1. `uv run python -m compileall src/models/multimodal.py` — no syntax errors
2. All imports resolve (no circular imports; `classification_model` and `regression_model` do not import from `multimodal`)
3. `_build_meta_cols(["logistic_l2", "xgboost"], ["ridge", "xgboost"])` returns 16 columns (4 models × 4 classes)
4. OOF loop calls training functions with `mlflow_ctx=None` — confirmed by reading the code
5. Final base model training calls pass `mlflow_ctx=mlflow_ctx` — confirmed by reading the code
6. Stacker is fit on `meta_train` (OOF), not on final base model predictions
7. `meta_features(df)` uses `self.final_clf_bundles` / `self.final_reg_bundles` (not OOF fold bundles)
8. `test_predictions.csv` uses stacker `predict_proba` output, not any single base model

---

### Chunk 6 — Wiring & Final Verification PENDING

**Section:** [Wiring And Compatibility](#wiring-and-compatibility) + [Verification](#verification)

Files:
- `src/main.py` — add `"multimodal"` to the `--model` argument choices, include in
  macOS XGBoost OpenMP helper (`_ensure_macos_openmp_for_xgboost`), dispatch to
  `src.models.multimodal.run(...)`

Acceptance gate — full verification suite:
```bash
uv run python -m compileall src/models/probability.py src/models/classification_model.py src/models/regression_model.py src/models/multimodal.py src/models/classification_inference.py src/main.py scripts/check_classification_contract.py
uv run python scripts/check_classification_contract.py
uv run python src/main.py --model classification --run-name classification-after-psm-phase1-smoke
uv run python src/main.py --model regression --run-name regression-after-psm-phase1-smoke
uv run python src/main.py --model multimodal --run-name psm-phase1-smoke
```

---

## Prerequisite: 4-Class Tier Migration

> Completed in Chunk 1.

The z-score threshold path (`_assign_tier_thresholded` with `[-0.5, 0.5, 1.5]`) is the
only definition of `prospect_tier`. Class mapping: `bust=0`, `bench=1`, `starter=2`,
`star=3`. All references to `"contributor"` have been removed.

---

## Relationship To Probability Stacking Model

This phase implements the Stage 1 and Stage 2 skeleton from
`PROBABILITY_STACKING_MODEL.md`, but limits Stage 1 to tabular models.

The PSM document says Logistic Regression and XGBoost are placeholders and
should later be replaced by the best-performing models. Phase 1 supports that by
making regression/classification model selection configurable instead of
hard-coding a fixed stack.

The PSM document requires out-of-fold predictions for meta-model training.
Phase 1 must implement this immediately so the stacker is not trained on
in-sample base-model predictions.

The PSM document includes future text models and optional confidence features.
Phase 1 excludes both. Text integration and confidence/entropy/margin features
come later after the probability contract is stable.

---

## Key Implementation Changes

### Shared Probability Layer

> Implemented in `src/models/probability.py`.

### Classification Model Refactor

> Implemented in `src/models/classification_model.py`.

### Regression Model Refactor

> Implemented in `src/models/regression_model.py`.

### Multimodal Orchestrator

> See [Chunk 5](#chunk-5--multimodal-orchestrator-️-pending) for the detailed
> implementation plan.

### Wiring And Compatibility

Update `src/main.py`:

- add `"multimodal"` to the `--model` argument choices
- include `"multimodal"` in the macOS XGBoost OpenMP helper
  (`_ensure_macos_openmp_for_xgboost`)
- dispatch to `src.models.multimodal.run(...)`

---

## Outputs And Metrics

`multimodal.py` writes to:

```text
outputs/multimodal/{run_name}/test_predictions.csv
outputs/multimodal/{run_name}/test_meta_features.csv
outputs/multimodal/{run_name}/base_model_summary.csv
```

`test_predictions.csv` columns:

```text
Name           (if present in loaded data)
draft_year     (if present in loaded data)
actual_tier
pred_tier
confidence
p_bust
p_bench
p_starter
p_star
```

Log or print:

- accuracy
- balanced accuracy
- macro F1
- multiclass OvR AUC
- log loss
- multiclass Brier score (mean of per-class OvR Brier scores)

---

## Verification

```bash
uv run python -m compileall src/models/probability.py src/models/classification_model.py src/models/regression_model.py src/models/multimodal.py src/models/classification_inference.py src/main.py scripts/check_classification_contract.py
uv run python scripts/check_classification_contract.py
uv run python src/main.py --model classification --run-name classification-after-psm-phase1-smoke
uv run python src/main.py --model regression --run-name regression-after-psm-phase1-smoke
uv run python src/main.py --model multimodal --run-name psm-phase1-smoke
```

Acceptance criteria:

- `check_classification_contract.py` passes with `prospect_tier` reporting exactly 4 classes `[0, 1, 2, 3]`
- No remaining references to `"contributor"` as a tier name anywhere in `src/` or `scripts/`
- `build_feature_matrix` always returns exactly 5 values; all callers unpack 5
- Existing standalone classification run completes
- Existing standalone regression run completes
- Multimodal run completes
- Standalone classification/regression still train their default model sets
- Selected classification models output 4 tier probabilities that sum to 1.0 per row
- Selected regression models output 4 tier probabilities via Gaussian CDF when target is `nba_role_zscore`
- Multimodal meta-features dynamically reflect configured selected base models
- Multimodal stacker trains only on out-of-fold base-model probabilities
- OOF fold training creates zero MLflow runs
- Final base models and stacker appear as nested runs under the multimodal top-level run
- No base training function creates its own calibration split
- No leakage columns appear in base model feature columns or exported meta-features

---

## Explicit Phase 1 Exclusions

Do not implement text / DistilBERT stacking yet.

Do not add confidence, entropy, top-two margin, or metadata features yet.

Do not replace LogisticRegression stacker with XGBoost or an MLP yet.

Do not make calibrated probabilities alter standalone classification hard-prediction
metrics unless `use_calibrated_for_metrics: true` is explicitly set.
