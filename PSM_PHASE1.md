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

- normalizing probability arrays
- converting probability arrays into standard DataFrames
- converting regression z-score predictions into tier probabilities using empirical calibration residuals
- building a prefit `CalibratedClassifierCV` compatible with current and older sklearn versions

Add a reusable `BaseModelBundle` dataclass that stores:

```python
name
task
estimator
feature_cols
proba_estimator
residuals
thresholds
```

`BaseModelBundle.predict_tier_proba(df)` must always return the standard
probability contract.

Classification bundles use `predict_proba()`. Regression bundles use
`predict()` on `nba_role_zscore`, then convert z-score predictions to tier
probabilities using calibration residuals and thresholds `[-0.5, 0.5, 1.5]`.

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
    return_bundles=True,
) -> dict[str, BaseModelBundle]:
    ...
```

Requirements:

- `selected_models=None` trains all current classification models.
- Standalone classification defaults remain equivalent to current behavior.
- Existing metrics, plots, MLflow logging, and model comparison behavior remain.
- Calibration is used for probability output, not to silently change hard-prediction metrics.
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
works, but trained regression models can emit tier probabilities.

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
    calibration_df=None,
    return_bundles=True,
) -> dict[str, BaseModelBundle]:
    ...
```

Requirements:

- `selected_models=None` trains all current regression models.
- Standalone regression defaults remain equivalent to current behavior.
- Regression models continue optimizing the z-score target.
- Tier-probability output is supported only when target is `nba_role_zscore`.
- Multimodal must require regression probability target `nba_role_zscore`.

Add config:

```yaml
model:
  regression:
    selected_models: ["lasso", "ridge", "xgboost"]
    probability_target_mode: nba_role_zscore
```

### Multimodal Orchestrator

Create `src/models/multimodal.py`.

It should not duplicate model definitions. It should call:

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

Training flow:

1. Load data with `load_data()`.
2. Use `prospect_tier` as target.
3. Make a stratified train/test split.
4. Build out-of-fold meta-features with `StratifiedKFold`.
5. For each fold:
   - train selected classification bundles
   - train selected regression bundles
   - predict held-out fold tier probabilities
   - write probability columns into the fold's meta-feature matrix
6. Train stacker on OOF meta-features.
7. Fit final selected base bundles on full train split.
8. Generate test meta-features.
9. Evaluate final stacker.

Meta-feature names are dynamic:

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

Use stacker:

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
```

### Wiring And Compatibility

Update `src/main.py`:

- add `"multimodal"` to model choices
- include `"multimodal"` in the macOS XGBoost OpenMP helper
- dispatch to `src.models.multimodal.run(...)`

Update `src/models/classification_inference.py`:

```python
TIER_CLASS_NAMES = ["bust", "bench", "starter", "star"]
```

Update `scripts/check_classification_contract.py`:

- expect classes `[0, 1, 2, 3]`
- use labels `Bust`, `Bench`, `Starter`, `Star`
- require probability columns `p_bust`, `p_bench`, `p_starter`, `p_star`

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

## Verification

Run:

```bash
uv run python -m compileall src/models/probability.py src/models/classification_model.py src/models/regression_model.py src/models/multimodal.py src/models/classification_inference.py src/main.py scripts/check_classification_contract.py
uv run python scripts/check_classification_contract.py
uv run python src/main.py --model classification --run-name classification-after-psm-phase1-smoke
uv run python src/main.py --model regression --run-name regression-after-psm-phase1-smoke
uv run python src/main.py --model multimodal --run-name psm-phase1-smoke
```

Acceptance criteria:

- Existing standalone classification run completes.
- Existing standalone regression run completes.
- Multimodal run completes.
- Standalone classification/regression still train their default model sets.
- Selected classification models output 4 tier probabilities.
- Selected regression models output 4 tier probabilities when trained against `nba_role_zscore`.
- Multimodal meta-features dynamically reflect configured selected base models.
- Multimodal stacker trains only on out-of-fold base-model probabilities.
- Final prediction probabilities sum to 1.0 per row.
- No leakage columns appear in base model feature columns or exported meta features.

## Explicit Phase 1 Exclusions

Do not implement text / DistilBERT stacking yet.

Do not add confidence, entropy, top-two margin, or metadata features yet.

Do not replace LogisticRegression stacker with XGBoost or an MLP yet.

Do not change the core target definition for `prospect_tier`.

Do not make calibrated probabilities alter standalone classification hard-prediction
metrics unless `use_calibrated_for_metrics: true` is explicitly set.
