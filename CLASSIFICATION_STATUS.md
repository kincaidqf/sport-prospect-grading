# Classification Pipeline — Implementation Status

## What Was Built

All six planned improvements are implemented and the contract tests pass. A full end-to-end smoke run (`uv run python src/main.py --model classification`) does **not** yet complete due to a macOS-specific crash described below.

---

## Completed Work

### 1. Metrics (F1-macro, Balanced Accuracy, Per-class)
- `src/training/evaluate.py`: `classification_metrics` now returns `balanced_accuracy`, `f1_macro`, and per-class `precision_{name}`, `recall_{name}`, `f1_{name}` keyed by class name (bust / contributor / star).
- Model selection uses `f1_macro` instead of raw accuracy.

### 2. Evaluation Protocols (`src/training/splits.py`)
- `get_random_split`: stratified random 80/20 split.
- `get_chronological_split`: train ≤2018 (~478 rows), val 2019–2020 (~114 rows), test 2021–2023 (~211 rows).
- `get_repeated_stratified_cv`: `RepeatedStratifiedKFold(n_splits=5, n_repeats=5)`.
- Config key `eval_mode: random | chronological | repeated_cv` wired through.

### 3. Engineered Features (`src/data/loader.py`)
Ten NCAA-only per-game / efficiency features added to `load_data()`:

| Feature | Formula |
|---|---|
| `to_pg` | TO / G |
| `oreb_pg` | ORebs / G |
| `dreb_pg` | DRebs / G |
| `fg3a_pg` | 3FGA / G |
| `ast_to` | AST / TO |
| `stocks_pg` | (STL + BLK) / G |
| `usage_proxy` | (FGA + 0.44·FTA + TO) / G |
| `efg_pct` | (FGM + 0.5·3FG) / FGA |
| `ts_pct` | PTS / (2·(FGA + 0.44·FTA)) |
| `height_in` | Ht parsed to inches |

`CLASSIFICATION_ENGINEERED_NUMERIC` list exported from `loader.py`. All have <1% nulls except `height_in` (~18.6% due to missing `Ht` column entries).

`Pos` added as OneHot categorical via `use_pos_categorical=True` in `build_feature_matrix`. `USE_DRAFT_PICK` constant set to `False`.

### 4. Class Balancing + Threshold Tuning
- `class_weight: balanced` in config; `LogisticRegressionCV` uses it directly.
- XGBoost uses `compute_sample_weight("balanced")` passed as `xgb__sample_weight` fit param.
- `_tune_thresholds()` implements coordinate-descent per-class probability offset tuning (validation set only; never touches test data). Activated via `threshold_tuning: true` in config.

### 5. Expanded Model Search (`src/models/classification_model.py`)
- `LogisticRegressionCV` scoring changed to `f1_macro` for `prospect_tier`.
- XGBoost `param_grid` expanded to 9 keys: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `reg_alpha`, `reg_lambda`, `gamma`.
- `ExtraTreesClassifier` added with grid: `n_estimators` [300,600] × `max_depth` [3,5,None] × `min_samples_leaf` [5,10,20].
- `GridSearchCV` scoring = `f1_macro` for multiclass, `roc_auc` for binary.

### 6. Multimodal-Ready Inference (`src/models/classification_inference.py`)
New file with:
- `predict_proba_stats(df, pipeline, feature_cols)`: returns DataFrame with `p_bust`, `p_contributor`, `p_star`, `pred_tier`, `confidence`. No post-draft fields.
- `export_stats_embeddings_or_proba(...)`: writes CSV with Name, draft_year, probability columns.
- `run_inference_on_merged_data(...)`: convenience wrapper.

### Contract Tests (`scripts/check_classification_contract.py`)
All 18 assertions pass:
1. No leakage columns (draft_pick / NBA outcome columns absent from feature list).
2. All 10 engineered features present with <50% nulls.
3. `prospect_tier` has exactly 3 classes (0=Bust, 1=Contributor, 2=Star).
4. Chronological split temporal ordering verified.
5. `predict_proba_stats` rows match input; probs sum to ~1.0; no `draft_pick` in output.

### Bug Fixes Applied
- MLflow param key collision: XGBoost grid params logged with `grid_` prefix to avoid collision when best params are logged separately.
- ExtraTrees step name: `get_xgb_importance_df`, `log_xgb_importances`, `print_xgb_importances` now accept `step_name` parameter so ExtraTrees (step `"etc"`) works correctly.
- `LogisticRegressionCV` changed from `n_jobs=-1` to `n_jobs=1` for macOS OpenMP safety.

---

## OpenMP SIGSEGV — RESOLVED

### Root Cause (discovered)
The `os.execve` re-exec in `main.py` set `DYLD_FALLBACK_LIBRARY_PATH` but did not set `OMP_NUM_THREADS` or `PYTHONUNBUFFERED`. In the re-exec'd process, `LogisticRegressionCV` with `solver='saga'` uses sklearn's OpenMP runtime to parallelize its inner loops. When XGBoost's `gs.fit()` then runs, XGBoost's libomp tries to initialize in a process that already has a different OpenMP runtime active — SIGSEGV in native code. `ExtraTreesClassifier` was removed to eliminate one conflict source, but the SAGA + XGBoost conflict remained.

### Fix Applied (`src/main.py`)
Two env vars added to the re-exec environment:
- `OMP_NUM_THREADS=1` — forces all OpenMP runtimes to use a single thread, preventing the multi-runtime parallelism conflict.
- `PYTHONUNBUFFERED=1` — ensures output is flushed immediately (previously caused buffered stdout to be lost on crash).

### Result
`uv run python src/main.py --model classification --run-name clf-improvements-smoke` exits 0 with all models training and all 4 plots saved. ExtraTreesClassifier has been permanently removed from the pipeline.

---

## Config State (`src/config/config.yaml`)

```yaml
classification:
  target_mode: prospect_tier
  use_draft_pick: false
  eval_mode: random
  class_weight: balanced
  threshold_tuning: false
  selection_metric: f1_macro
  xgboost:
    n_estimators: [100, 200]
    max_depth: [2, 3]
    learning_rate: [0.05, 0.1]
    subsample: [0.7, 0.9]
    colsample_bytree: [1.0]
    min_child_weight: [3, 10]
    reg_alpha: [0, 0.1]
    reg_lambda: [1, 5]
    gamma: [0]
    cv_folds: 3
    n_jobs: 1
    grid_n_jobs: 1
    pre_dispatch: 1
```

---

## Files Changed

| File | Status |
|---|---|
| `src/training/splits.py` | New |
| `src/models/classification_inference.py` | New |
| `scripts/check_classification_contract.py` | New |
| `src/models/classification_model.py` | Major rewrite |
| `src/data/loader.py` | Modified |
| `src/training/evaluate.py` | Modified |
| `src/utils/features.py` | Modified |
| `src/utils/plotting.py` | Modified |
| `src/utils/mlflow_utils.py` | Modified |
| `src/config/config.yaml` | Modified |
