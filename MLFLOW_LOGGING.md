# MLflow Logging

## Setup & Configuration

MLflow tracking is configured via `src/config/config.yaml` under the `logging.mlflow` section. All resolution is handled by `src/utils/mlflow_utils.py`.

```yaml
logging:
  mlflow:
    tracking_uri: null        # URI, path, or null → defaults to ./mlruns/
    artifact_location: null   # optional shared artifact root for new experiments
    experiment_name: null     # optional override; auto-derived if null
    experiment_prefix: nba-draft-ml
```

### Tracking URI resolution (priority order)

1. `--tracking-uri` CLI argument
2. `MLFLOW_TRACKING_URI` environment variable
3. `logging.mlflow.tracking_uri` in config
4. Default: `./mlruns/` (file-based, local)

Plain paths are automatically converted to `file://` URIs. `sqlite:///mlflow.db` in the root is an alternative local backend for the SQLite store.

### Starting the UI

```bash
# Local file-based (default)
uv run mlflow ui --backend-store-uri mlruns/

# Or via the convenience scripts
bash scripts/start_mlflow_ui.sh      # non-blocking, local mlruns/
bash scripts/start_mlflow_server.sh  # server mode, configurable port
```

---

## Experiments

Each model type logs to its own experiment:

| Model | Experiment name |
|-------|----------------|
| Regression | `nba-draft-prospect-regression` |
| Classification | `nba-draft-prospect-classification` |
| Text | `nba-draft-prospect-text` |
| Multimodal | `nba-draft-prospect-multimodal` |

Experiment names can be overridden globally via `logging.mlflow.experiment_name` or prefixed via `logging.mlflow.experiment_prefix`.

---

## Run Naming

When no `--run-name` is provided, runs are auto-named:

```
{model_type}-{target}-{timestamp}-{username}-{git_sha}
```

Example: `regression-nba_role_zscore-20260430-142305-kincaid-2bb2791`

This is deterministic enough to re-identify a run from the commit it was trained on.

---

## Run Tags

All runs share these tags (set by `build_mlflow_context`):

| Tag | Value |
|-----|-------|
| `model_type` | `regression`, `classification`, `text`, or `multimodal` |
| `target_name` | active target mode string |
| `user` | OS username |
| `hostname` | machine hostname |
| `git_branch` | current branch name |
| `git_commit` | full commit SHA |
| `tracking_uri` | resolved tracking URI |
| `project_root` | absolute path to repo root |

---

## Run Structure by Model

### Regression (`--model regression`)

**Parent run** logs:

*Setup params:*
- `model_family`, `target`, `target_score_mode`, `input_normalization_mode`
- `use_draft_pick`, `use_engineered_features`, `use_pos_categorical`, `selected_models`

*Data summary params:*
- `n_total`, `n_train`, `n_test`, `test_size`, `cv_folds`, `random_seed`, `split_type`
- `target_mean`, `target_std`, `target_min`, `target_max`

*Reproducibility params:*
- `python_version`, `device`
- `sklearn_version`, `xgboost_version`, `mlflow_version` (others if installed)

*Artifacts:*
- `config/config.json` — full resolved config snapshot
- `summary/candidate_summary_*.json` — per-model test metrics + `best_model`, `selection_metric`

*Test metrics (parent run, model-prefixed):*
- `{model}__r2`, `{model}__rmse`, `{model}__mae` for each of lasso, ridge, xgboost

**Nested run per estimator** (`{run_name}__{key}`, tagged `estimator={key}`):

*Lasso / Ridge:*
- Params: `model`, `target`, `alpha` (selected), `alpha_search_min/max/n`, `cv_folds`, `random_seed`, `n_train`, `use_draft_pick`
- Metrics: `n_nonzero_features` (Lasso only)
- Artifacts: `lasso/` or `ridge/` — full sklearn pipeline via `mlflow.sklearn.log_model`

*XGBoost (GridSearchCV):*
- Params: `model`, `target`, `cv_folds`, `n_train`, best params (`n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `reg_alpha`, `reg_lambda`, `gamma`), `best_cv_score`, `use_fixed_xgb_params`
- Params: full grid ranges as `grid_{param}` strings
- Params: top-15 XGBoost feature importances (gain-based, logged via `log_xgb_importances`)
- Artifacts: `xgboost/` — full sklearn pipeline

*XGBoost (fixed OOF params — when called from multimodal):*
- Params: all fixed params + `use_fixed_xgb_params=True`
- Artifacts: `xgboost/`

**Local plots** saved to `outputs/plots/{run_name}/`:
- `regression_results.png` — actual vs. predicted scatter, residual plot, tier distribution (actual vs. predicted counts) for each model
- `feature_importance.png` — coefficient bars (lasso/ridge) or XGBoost gain importance
- `model_summary.png` — summary bar chart across models

---

### Classification (`--model classification`)

**Parent run** logs:

*Setup params:*
- `model_family`, `target`, `target_score_mode`, `input_normalization_mode`
- `use_draft_pick`, `use_engineered_features`, `use_pos_categorical`
- `eval_mode` (`random`, `chronological`, or `repeated_cv`)
- `class_weight`, `threshold_tuning`, `calibration_enabled`, `calibration_size`, `selected_models`

*Data summary params* (classification version):
- `n_total`, `n_train`, `n_test`, `test_size`, `cv_folds`, `random_seed`, `split_type`
- `n_positive`, `n_negative`, `class_balance`

*Reproducibility params:* same as regression.

*Artifacts:* `config/config.json`, `summary/candidate_summary_*.json`

*Test metrics (parent run, model-prefixed):*
- `{key}__accuracy`, `{key}__balanced_accuracy`, `{key}__auc`, `{key}__f1_macro`
- `{key}__precision_{class}`, `{key}__recall_{class}` for each of bust/bench/starter/star
- `{key}__default_f1_macro` and `{key}__tuned_f1_macro` when `threshold_tuning=true`

**Nested run per estimator** (`{run_name}__{key}`, tagged `estimator={key}`):

*LogisticL1 / LogisticL2:*
- Params: `model`, `target`, `penalty`, `C` (selected), `cv_folds`, `class_weight`, `n_train`, `use_fixed_xgb_params`
- Artifacts: `logistic_l1/` or `logistic_l2/`

*XGBoost* — same as regression nested run structure, but scored on `f1_macro`.

**Repeated CV path (legacy `eval_mode: repeated_cv`):**
Nested runs per model are tagged `eval_mode=repeated_cv` and log:
- Metrics: `cv_mean_f1_macro`, `cv_std_f1_macro`, `cv_mean_balanced_accuracy`, `cv_mean_accuracy`
- Params: `cv_n_splits`, `cv_n_repeats`
- Artifacts: fitted pipeline

**Local plots** saved to `outputs/plots/{run_name}/`:
- `classification_results.png` — confusion matrix (top row) + tier distribution bar chart (bottom row)
- `feature_importance.png`, `model_summary.png`

---

### Text Model (`--model text`)

**Single parent run** (no nested runs). The text model has no estimator loop.

*Setup params:*
- `model_family="text"`, `target` (VORP)
- `pretrained`, `output_dim`, `freeze_base`, `max_length`
- `batch_size`, `epochs`, `lr`
- `n_total`, `n_train`, `n_val`, `n_test`

*Epoch-indexed metrics* (logged at each epoch via `log_epoch_metrics`, `step=epoch`):
- `train_loss`, `val_loss`

*Final test metrics* (logged once after training):
- Regression head: `r2`, `rmse`, `mae`
- `best_val_loss`
- Binary heads (AUC, average precision, accuracy at 0.5 threshold):
  - `star_auc`, `star_ap`, `star_acc`
  - `survived_auc`, `survived_ap`, `survived_acc`
  - `starter_auc`, `starter_ap`, `starter_acc`

*Artifacts:*
- `config/config.json`
- `model/` — full PyTorch model via `mlflow.pytorch.log_model`
- Checkpoint is separately saved to `outputs/checkpoints/text_model.pt` when `--save-path` is provided (not an MLflow artifact — written directly to disk)

**Local plots** saved to `outputs/plots/{run_name}/`:
- Predicted vs. actual VORP scatter
- Training/validation loss curve

---

### Multimodal (`--model multimodal`)

The multimodal run has a 3-level structure: parent → final base model nested runs → stacker nested run.

**OOF fold training (5 folds) does not log to MLflow.** `mlflow_ctx=None` is passed to `train_selected_classification_models` and `train_selected_regression_models` during cross-validation. Only the final base models (trained on the full training set after OOF) create nested runs.

**Parent run** logs:

*Setup params:*
- `model_family="multimodal"`, `target="prospect_tier"`
- `clf_models`, `reg_models` (lists of base model keys)
- `cal_size`, `pretune_oof`, `meta_cols`

*Data summary params:* classification version using `actual_test_size` (chronological test proportion)

*Reproducibility params:* same as other models.

*Artifacts:* `config/config.json`

*Test metrics* (parent run, from the stacker on the held-out test set):
- Ordinal classification metrics: `top1_accuracy`, `within_one_accuracy`, `distance_weighted_accuracy`, `ordinal_mae`, `expected_class_mae`, `quadratic_weighted_kappa`
- Standard classification metrics: `accuracy`, `balanced_accuracy`, `auc`, `f1_macro`
- Per-class: `precision_{class}`, `recall_{class}` for each tier
- Calibration: `log_loss`, `brier_score` (mean across 4 classes)

**Nested runs for final base models** — one per estimator in `clf_models` + `reg_models`:
- Same per-estimator logging as standalone classification/regression runs
- Always use fixed OOF params (`use_fixed_xgb_params=True`), so XGBoost does not re-run GridSearchCV
- Artifacts: `{key}/` sklearn pipeline per model

**Nested stacker run** (`{run_name}__stacker`, tagged `estimator=logistic_stacker`):
- Params: `stacker="LogisticRegression"`, `class_weight="balanced"`, `max_iter=5000`, `n_meta_features`
- Artifacts: `stacker/` — `mlflow.sklearn.log_model`

**Local outputs** written to `outputs/multimodal/{run_name}/` (not MLflow artifacts):
- `test_predictions.csv` — per-player predictions with ordinal diagnostics
- `test_meta_features.csv` — raw stacker input probabilities per player
- `base_model_summary.csv` — per-base-model ordinal metrics on the test set
- `model_summary.csv` — stacker-level summary metrics
- `worst_misses.csv` — highest-error predictions
- `ordinal_error_distribution.csv` — count/percent by error magnitude
- `confusion_matrix.csv`
- `probability_mass_by_true_class.csv` — average predicted proba per actual tier
- `stacker_contributions.csv` — absolute stacker coefficients per base model + class
- `metric_legend.csv` — metric definitions

---

## Shared Utility Reference (`src/utils/mlflow_utils.py`)

| Function | Purpose |
|----------|---------|
| `build_mlflow_context(...)` | Resolves URI, experiment, run name; sets tracking URI; returns `MLflowContext` |
| `managed_run(ctx, *, run_name, nested, tags)` | Context manager that merges `ctx.tags` into the run; use `nested=True` for child runs |
| `log_config_dict(cfg)` | Logs `cfg` as `config/config.json` artifact |
| `log_common_params(params)` | Coerces lists/dicts to strings, then calls `mlflow.log_params` |
| `log_epoch_metrics(metrics, epoch)` | Logs step-indexed metrics (`step=epoch`) |
| `log_reproducibility_metadata(device)` | Logs Python version + library versions |
| `log_data_summary(df, target_col, task, ...)` | Logs dataset counts, split config, and target distribution |
| `log_candidate_summary(results, task, ...)` | Logs `best_model`, `selection_metric`, and saves `summary/candidate_summary_*.json` |

The `nullcontext()` pattern is used in `train_selected_classification_models` and `train_selected_regression_models` so those functions can be called from OOF loops without logging. When `mlflow_ctx=None`, no run is opened and no metrics/params/artifacts are written.

---

## Viewing Results

```bash
# Open the tracking UI (default local backend)
uv run mlflow ui --backend-store-uri mlruns/

# Directly query a run's metrics from the CLI
uv run mlflow runs list --experiment-name nba-draft-prospect-multimodal
```

Local plots are always written to `outputs/plots/{run_name}/` regardless of MLflow availability, so visual output does not require the tracking server to be running.
