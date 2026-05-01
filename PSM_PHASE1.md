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

## Implementation Chunks

Work through these chunks in order. After each chunk, run the acceptance gate and
confirm before starting the next.

| Chunk | Scope | Acceptance Gate |
|-------|-------|-----------------|
| **Chunk 1** | 4-Class Tier Migration | `check_classification_contract.py` passes with 4 classes |
| **Chunk 2** | Shared Probability Layer (`probability.py`) | `compileall src/models/probability.py` clean |
| **Chunk 3** | Classification Model Refactor | standalone classification run completes |
| **Chunk 4** | Regression Model Refactor | standalone regression run completes |
| **Chunk 5** | Multimodal Orchestrator (`multimodal.py`) | `compileall src/models/multimodal.py` clean, logic review |
| **Chunk 6** | Wiring & Final Verification (`main.py`) | full verification suite passes |

### Chunk 1 — 4-Class Tier Migration

**Section:** [Prerequisite: 4-Class Tier Migration](#prerequisite-4-class-tier-migration)

Files:
- `src/data/loader.py` — remove percentile tier path; z-score thresholds `[-0.5, 0.5, 1.5]` become the only definition of `prospect_tier`
- `src/models/classification_inference.py` — update `TIER_CLASS_NAMES`; rename probability columns to `p_bust`, `p_bench`, `p_starter`, `p_star`
- `scripts/check_classification_contract.py` — expect 4 classes `[0,1,2,3]`, labels `Bust/Bench/Starter/Star`, require 4 probability columns, unpack 5 values from `build_feature_matrix`
- Any other callers of `build_feature_matrix` that unpack only 4 values
- Rename `PROBABLITY_STACKING_MODEL.md` → `PROBABILITY_STACKING_MODEL.md`

Acceptance gate:
```bash
uv run python scripts/check_classification_contract.py
```
Must pass with `prospect_tier` reporting exactly 4 classes.

---

### Chunk 2 — Shared Probability Layer

**Section:** [Key Implementation Changes → Shared Probability Layer](#shared-probability-layer)

Files:
- `src/models/probability.py` *(new)* — constants, normalization helper, DataFrame conversion helper, Gaussian CDF converter, `BaseModelBundle` dataclass with `predict_tier_proba(df)`

Acceptance gate:
```bash
uv run python -m compileall src/models/probability.py
```

---

### Chunk 3 — Classification Model Refactor

**Section:** [Key Implementation Changes → Classification Model Refactor](#classification-model-refactor)

Files:
- `src/models/classification_model.py` — add model registry, `train_selected_classification_models()`, calibration split ownership, `use_fixed_xgb_params`
- `config.yaml` — add `model.classification.selected_models` and `model.classification.calibration.*`

Acceptance gate:
```bash
uv run python src/main.py --model classification --run-name classification-after-psm-phase1-smoke
```

---

### Chunk 4 — Regression Model Refactor

**Section:** [Key Implementation Changes → Regression Model Refactor](#regression-model-refactor)

Files:
- `src/models/regression_model.py` — add model registry, `train_selected_regression_models()`, `use_fixed_xgb_params`, guard for non-zscore targets
- `config.yaml` — add `model.regression.selected_models` and `model.regression.probability_target_mode`

Acceptance gate:
```bash
uv run python src/main.py --model regression --run-name regression-after-psm-phase1-smoke
```

---

### Chunk 5 — Multimodal Orchestrator

**Section:** [Key Implementation Changes → Multimodal Orchestrator](#multimodal-orchestrator)

Files:
- `src/models/multimodal.py` *(new)* — `MultimodalProspectModel` class, OOF loop, calibration carve-out, pre-tuning strategy A/B, stacker, final fit, MLflow nested runs, output CSVs
- `config.yaml` — add full `model.multimodal.*` block

Acceptance gate:
```bash
uv run python -m compileall src/models/multimodal.py
```
Plus visual review of OOF/MLflow logic before proceeding.

---

### Chunk 6 — Wiring & Final Verification

**Section:** [Wiring And Compatibility](#wiring-and-compatibility) + [Verification](#verification)

Files:
- `src/main.py` — add `"multimodal"` to `--model` choices, include in macOS OpenMP helper, dispatch to `multimodal.run()`

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

**This must be resolved before any other Phase 1 work begins.**

The codebase currently has a 3-class vs 4-class inconsistency that must be fully
eliminated. The 4-class path is the only acceptable definition going forward.

### Current state

| File | Current tier definition |
|------|------------------------|
| `classification_model.py` | 4-class: `["bust", "bench", "starter", "star"]` ✓ |
| `classification_inference.py` | 3-class stale: `["bust", "contributor", "star"]` ✗ |
| `check_classification_contract.py` | asserts `prospect_tier` has classes `[0, 1, 2]` ✗ |
| `config.yaml` `composite_score.tier_percentiles` | `[50, 80]` drives a 3-class percentile split ✗ |

### Required changes

**Data loader (`src/data/loader.py`)**

`prospect_tier` must be defined exclusively by the z-score threshold path, not
the percentile path:

- Thresholds: `[-0.5, 0.5, 1.5]` applied to `nba_role_zscore`
- Class mapping: `bust=0`, `bench=1`, `starter=2`, `star=3`
- Delete or disable the 3-class percentile tier path (`tier_percentiles`)
- `_assign_tier_thresholded` is the canonical function for `prospect_tier`
- All references to `"contributor"` as a tier name must be removed

**`src/models/classification_inference.py`**

```python
TIER_CLASS_NAMES = ["bust", "bench", "starter", "star"]
```

Update `predict_proba_stats` output column names accordingly:
`p_bust`, `p_bench`, `p_starter`, `p_star`.

**`scripts/check_classification_contract.py`**

- Expect `prospect_tier` classes `[0, 1, 2, 3]`
- Use labels `Bust`, `Bench`, `Starter`, `Star` (remove `Contributor`)
- Require probability columns `p_bust`, `p_bench`, `p_starter`, `p_star`
- Update assertion at check 3 to validate 4 classes

**`build_feature_matrix` return value**

`build_feature_matrix` must always return exactly 5 values:

```python
preprocessor, numeric_cols, categorical_cols, ordinal_cols, passthrough_cols = build_feature_matrix(...)
```

All callers that currently unpack only 4 values (including
`check_classification_contract.py`) must be updated to unpack 5.

### Acceptance gate

Do not proceed to any other Phase 1 work until:

```bash
uv run python scripts/check_classification_contract.py
```

passes all checks with `prospect_tier` reporting exactly 4 classes.

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

Also rename the existing misspelled file:

```text
PROBABLITY_STACKING_MODEL.md -> PROBABILITY_STACKING_MODEL.md
```

---

## Key Implementation Changes

### Shared Probability Layer

Create `src/models/probability.py`.

Add:

```python
TIER_CLASS_NAMES = ["bust", "bench", "starter", "star"]
TIER_LABELS = [0, 1, 2, 3]
TIER_THRESHOLDS = (-0.5, 0.5, 1.5)
PROBA_COLUMNS = ["p_bust", "p_bench", "p_starter", "p_star"]
```

Add helpers for:

- normalizing probability arrays (ensure they sum to 1.0)
- converting probability arrays into standard DataFrames with `PROBA_COLUMNS`
- converting regression z-score predictions to tier probabilities via Gaussian CDF
  (see algorithm below)
- building a prefit `CalibratedClassifierCV` compatible with current sklearn versions

#### Gaussian CDF regression → tier probability algorithm

Regression models predict a continuous `nba_role_zscore`. To convert predictions
to 4-class tier probabilities without introducing an extra model:

1. During training, compute training residuals: `residuals = y_train - y_pred_train`
2. Store `residual_std = residuals.std()` in the `BaseModelBundle`
3. During inference, for a predicted z-score `z`:

```python
from scipy.stats import norm

thresholds = [-0.5, 0.5, 1.5]
sigma = bundle.residual_std

p_bust    = norm.cdf(thresholds[0], loc=z, scale=sigma)
p_bench   = norm.cdf(thresholds[1], loc=z, scale=sigma) - norm.cdf(thresholds[0], loc=z, scale=sigma)
p_starter = norm.cdf(thresholds[2], loc=z, scale=sigma) - norm.cdf(thresholds[1], loc=z, scale=sigma)
p_star    = 1.0 - norm.cdf(thresholds[2], loc=z, scale=sigma)
```

This treats the regression model's uncertainty as Gaussian with standard deviation
equal to the empirical training residual std, and computes the probability mass
within each tier's z-score interval. No additional model is fit.

The probabilities sum to exactly 1.0 by construction.

#### `BaseModelBundle` dataclass

```python
@dataclass
class BaseModelBundle:
    name: str
    task: str                  # "classification" or "regression"
    estimator: Any             # fitted sklearn pipeline
    feature_cols: list[str]
    proba_estimator: Any       # CalibratedClassifierCV for classification, None for regression
    residual_std: float        # regression only; 0.0 for classification
    thresholds: tuple          # TIER_THRESHOLDS
```

`BaseModelBundle.predict_tier_proba(df)` must always return a DataFrame with
columns `PROBA_COLUMNS` in `TIER_CLASS_NAMES` order. Probabilities must sum to
1.0 per row.

- Classification bundles call `proba_estimator.predict_proba(df[feature_cols])`
- Regression bundles call `estimator.predict(df[feature_cols])` then apply the
  Gaussian CDF algorithm above using `residual_std` and `thresholds`

### Classification Model Refactor

Update `src/models/classification_model.py` so existing standalone training
still works, but the model family can also be called by `multimodal.py`.

Add a classification model registry with keys:

```python
"logistic_l1"
"logistic_l2"
"xgboost"
```

Add:

```python
def train_selected_classification_models(
    train_df,
    target_col,
    cfg,
    selected_models=None,
    calibration_df=None,
    use_fixed_xgb_params=False,
    return_bundles=True,
) -> dict[str, BaseModelBundle]:
    ...
```

#### Calibration split ownership

`train_selected_classification_models` **never** creates its own calibration
split. The caller always provides `calibration_df` or `None`:

- When `calibration_df` is provided: use it to fit `CalibratedClassifierCV` in
  `cv="prefit"` mode after training the base estimator on `train_df`
- When `calibration_df=None`: skip calibration; `proba_estimator` falls back to
  the base estimator's `predict_proba`

For standalone classification runs (`classification_model.run()`), the `run()`
function owns the calibration split and carves it out of its own train set before
calling `train_selected_classification_models`. The base function never splits
internally.

#### `use_fixed_xgb_params`

When `use_fixed_xgb_params=True`, XGBoost skips `GridSearchCV` and trains with
a single parameter set from config (see OOF section below). This flag is set
by `multimodal.py` during OOF folds.

#### Requirements

- `selected_models=None` trains all registered classification models.
- Standalone classification defaults remain equivalent to current behavior.
- Existing metrics, plots, MLflow logging, and model comparison behavior remain
  for standalone runs.
- Calibration is used for probability output only. It does not alter hard-prediction
  metrics unless `use_calibrated_for_metrics: true` is explicitly set.
- Default calibration method is sigmoid.

Add config:

```yaml
model:
  classification:
    selected_models: ["logistic_l1", "logistic_l2", "xgboost"]
    calibration:
      enabled: true
      method: sigmoid
      calibration_size: 0.15
      use_calibrated_for_metrics: false
```

### Regression Model Refactor

Update `src/models/regression_model.py` so existing standalone training still
works, but trained regression models can emit tier probabilities via the Gaussian
CDF algorithm.

Add a regression model registry with keys:

```python
"lasso"
"ridge"
"xgboost"
```

Add:

```python
def train_selected_regression_models(
    train_df,
    target_col,
    cfg,
    selected_models=None,
    use_fixed_xgb_params=False,
    return_bundles=True,
) -> dict[str, BaseModelBundle]:
    ...
```

#### Calibration split ownership

Regression bundles do not use `CalibratedClassifierCV`. There is no
`calibration_df` parameter. Tier-probability output is produced entirely by the
Gaussian CDF algorithm using `residual_std` stored in the bundle. No split is
needed.

#### `use_fixed_xgb_params`

Same meaning as in the classification refactor. Set by `multimodal.py` during
OOF folds.

#### Requirements

- `selected_models=None` trains all registered regression models.
- Standalone regression defaults remain equivalent to current behavior.
- Regression models continue optimizing the z-score target.
- Tier-probability output is only valid when target is `nba_role_zscore`.
  `BaseModelBundle.predict_tier_proba()` must raise if called on a regression
  bundle trained against a different target.
- Multimodal must require `nba_role_zscore` as the regression probability target.

Add config:

```yaml
model:
  regression:
    selected_models: ["lasso", "ridge", "xgboost"]
    probability_target_mode: nba_role_zscore
```

### Multimodal Orchestrator

Create `src/models/multimodal.py`.

It must not duplicate model definitions. It calls:

```python
train_selected_classification_models(...)
train_selected_regression_models(...)
```

Expose:

```python
class MultimodalProspectModel:
    def fit(self, train_df): ...
    def meta_features(self, df): ...
    def predict_proba(self, df): ...
    def predict(self, df): ...

def run(df=None, cfg=None, run_name=None, tracking_uri=None): ...
```

#### Splitting ownership

`MultimodalProspectModel` owns **all** data splitting. This includes:

- train/test split
- OOF fold splits
- calibration splits carved from each OOF fold's training portion

No base training function creates any split internally.

#### OOF hyperparameter strategy

XGBoost's `GridSearchCV` must **not** run inside the OOF loop — it would multiply
training cost by `cv_folds × grid_size` and produce inconsistent hyperparameters
across folds.

**Rule: OOF phase → fixed params. Final fit → full GridSearchCV.**

Two supported strategies for choosing OOF fixed XGBoost params:

**Strategy A — Config-specified (default):**
Add an `oof_params` sub-key under `model.multimodal.xgboost` in config. The user
explicitly sets the single parameter set used for all OOF folds. This is
reproducible and transparent.

```yaml
model:
  multimodal:
    xgboost:
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

**Strategy B — Pre-tuning pass (optional):**
Before the OOF loop begins, run a one-time `GridSearchCV` on the full training
set to find the best XGBoost params. Store those best params and reuse them as
fixed params for every OOF fold. This finds better params automatically at the
cost of one extra grid search.

Enable with:

```yaml
model:
  multimodal:
    xgboost:
      pretune_oof_params: true   # runs GridSearchCV once before OOF loop
```

When `pretune_oof_params: true`, the pre-tuning result overrides `oof_params`.
When `pretune_oof_params: false` (default), `oof_params` values are used as-is.

The final fit (after OOF) always runs full `GridSearchCV` regardless of which
strategy was used for OOF, ensuring final base models are optimally tuned.

Lasso and Ridge use `LassoCV` / `RidgeCV` (which already do their own internal
alpha search) and are not affected by this strategy.

#### Training flow

1. Load data with `load_data()`.
2. Use `prospect_tier` as target (4 classes).
3. Make a stratified train/test split (multimodal owns this split).
4. **If pre-tuning enabled:** run one `GridSearchCV` on full train set; store best
   XGBoost params as OOF fixed params.
5. Build out-of-fold meta-features with `StratifiedKFold(n_splits=cv_folds)`.
6. For each fold:
   a. Multimodal carves `cal_df` (size = `calibration_size` fraction) from the
      fold's training portion using a stratified split.
   b. Remaining fold training data becomes `fold_train_df`.
   c. Call `train_selected_classification_models(fold_train_df, ..., calibration_df=cal_df, use_fixed_xgb_params=True)` — no MLflow context.
   d. Call `train_selected_regression_models(fold_train_df, ..., use_fixed_xgb_params=True)` — no MLflow context.
   e. Call `bundle.predict_tier_proba(fold_val_df)` for each bundle.
   f. Write probability columns into the fold's meta-feature matrix row block.
7. Train stacker on OOF meta-features.
8. **Final fit:** call `train_selected_classification_models(full_train_df, ..., calibration_df=cal_df, use_fixed_xgb_params=False)` — with MLflow context, runs full GridSearchCV.
9. Call `train_selected_regression_models(full_train_df, ..., use_fixed_xgb_params=False)` — with MLflow context.
10. Generate test meta-features using final bundles.
11. Evaluate final stacker on test set.

#### Meta-feature naming

Meta-feature column names are dynamic and reflect the configured selected models:

```text
classification__logistic_l2__p_bust
classification__logistic_l2__p_bench
classification__logistic_l2__p_starter
classification__logistic_l2__p_star
regression__ridge__p_bust
regression__ridge__p_bench
regression__ridge__p_starter
regression__ridge__p_star
```

#### Stacker

```python
LogisticRegression(class_weight="balanced", max_iter=5000)
```

Add config:

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

### MLflow Logging Rules

OOF fold training produces intermediate models that are discarded after the loop.
These must not create MLflow runs.

**Log:**

- Final base models (retrained on full train set) — one nested run each
- Final stacker — one nested run
- Final evaluation metrics (accuracy, balanced accuracy, macro F1, OvR AUC, log
  loss, Brier score) — logged on the top-level multimodal run

**Do not log:**

- Any per-fold OOF models
- Per-fold metrics
- Intermediate GridSearchCV candidates

When calling `train_selected_classification_models` or
`train_selected_regression_models` during OOF folds, pass `mlflow_ctx=None`.
When calling them for the final fit, pass the multimodal MLflow context so that
nested runs are created under the top-level multimodal run.

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
Name
draft_year
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
- multiclass Brier score

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
