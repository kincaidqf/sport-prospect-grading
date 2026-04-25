# NBA Draft ML Research

## Goal

Predict NBA draft success using:

- College statistics (regression and classification)
- Scouting reports (NLP text model)
- Combined stats + scouting (multimodal model)

---

## Project Structure

```
sport-prospect-grading/
├── src/
│   ├── main.py                    # Entry point — CLI args, config loading, model dispatch
│   ├── config/
│   │   └── config.yaml            # Centralized config for all model types
│   ├── models/
│   │   ├── regression_model.py    # Lasso / Ridge / XGBoost regression
│   │   ├── classification_model.py# LogisticL1 / LogisticL2 / XGBoost classification
│   │   ├── text_model.py          # Transformer encoder on scouting reports
│   │   └── multimodal_model.py    # Stats + scouting joint model
│   ├── data/
│   │   ├── loader.py              # Tabular data loading, feature engineering, preprocessing
│   │   ├── dataset.py             # PyTorch Dataset for text/multimodal models
│   │   └── preprocessor.py        # Shared preprocessing utilities
│   ├── training/
│   │   ├── trainer.py             # Neural model training loop (grad clip, checkpointing)
│   │   └── evaluate.py            # Metrics: MAE, RMSE, R², accuracy, ROC-AUC
│   └── utils/
│       ├── device.py              # Auto-detects CPU / MPS / CUDA
│       ├── features.py            # Feature name expansion and importance helpers
│       ├── plotting.py            # Model summary and feature importance plots (local only)
│       └── mlflow_utils.py        # MLflow context, run naming, logging helpers
├── data/
│   ├── nba/                       # NBA master list and per-season stats
│   ├── ncaa/                      # NCAA master list and annual stats
│   └── scripts/                   # Data fetch, parse, and reconciliation scripts
├── notebooks/
│   └── 01_eda.ipynb
├── outputs/
│   └── plots/                     # Per-run local plot output (one subfolder per run name)
├── scripts/
│   ├── start_mlflow_ui.sh
│   └── start_mlflow_server.sh
├── MLFLOW_LOGGING.md              # MLflow logging specification
└── pyproject.toml
```

---

## Setup

### Prerequisites

- Python 3.11 or 3.12
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager

### Install

```bash
git clone <repo-url>
cd sport-prospect-grading
uv sync
cp .env.example .env
# Edit .env if needed (see Environment Variables below)
```

---

## Running Models

All models are launched through `src/main.py`. The `--model` flag selects the pipeline and is the primary switch.

### Basic usage

```bash
uv run python src/main.py --model regression
uv run python src/main.py --model classification
uv run python src/main.py --model text
```

### All CLI arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--model` | `regression \| classification \| text \| multimodal` | value from `config.yaml` | Selects which model pipeline to run. Overrides `model.type` in config. |
| `--config` | path | `src/config/config.yaml` | Path to the YAML config file. |
| `--run-name` | string | auto-generated | MLflow parent run name. **Use this to label and differentiate runs.** |
| `--epochs` | int | value from config | Override `training.epochs` for text/multimodal models. |
| `--output-dir` | path | value from config | Override the output directory. |
| `--tracking-uri` | URI or path | value from config or `MLFLOW_TRACKING_URI` env var | Override the MLflow tracking URI for this run only. |

### Naming runs

The `--run-name` flag is the recommended way to distinguish runs when comparing results in MLflow.

```bash
# Label a regression run clearly for comparison
uv run python src/main.py --model regression --run-name regression-baseline-no-pick
uv run python src/main.py --model regression --run-name regression-with-pick

# Label a classification run by target
uv run python src/main.py --model classification --run-name clf-survived3yrs-v1
```

If `--run-name` is omitted, a name is auto-generated using the pattern:

```
{model_type}-{target}-{YYYYMMDD-HHMMSS}-{user}-{git_sha}
```

Example: `regression-plus_minus-20260423-143012-kincaid-c79bc0f`

The run name is used as:
- The MLflow parent run name
- The subfolder name under `outputs/plots/` where local PNGs are saved

---

## Configuration

`src/config/config.yaml` controls all model behavior. CLI flags override specific fields at runtime; all other values come from the file.

### Regression (`--model regression`)

```yaml
model:
  regression:
    target_mode: plus_minus       # only supported target
    use_draft_pick: false         # set true to include draft pick position as a feature
    alpha_min: 1e-4               # Lasso/Ridge alpha search range (lower bound)
    alpha_max: 1e2                # Lasso/Ridge alpha search range (upper bound)
    alpha_steps: 100              # number of alpha candidates
    max_iter: 10000               # Lasso max iterations
    cv_folds: 5                   # CV folds for Lasso and Ridge
    xgboost:
      n_estimators: [100, 200]    # grid search values
      max_depth: [2, 3]
      learning_rate: [0.05, 0.1]
      subsample: [0.8]
      cv_folds: 3                 # CV folds for XGBoost grid search
      n_jobs: 1                    # XGBoost worker threads; keep 1 on macOS for stability
      grid_n_jobs: 1               # GridSearchCV worker processes
      pre_dispatch: 1              # Jobs queued ahead of active workers
```

Trains three models: **Lasso**, **Ridge**, **XGBoost**. Each gets its own nested MLflow run under the parent. XGBoost uses `GridSearchCV` over the values listed above.

### Classification (`--model classification`)

```yaml
model:
  classification:
    target_mode: survived_3yrs    # survived_3yrs | became_starter
    use_draft_pick: false
    xgboost:
      n_estimators: [100, 200]
      max_depth: [2, 3]
      learning_rate: [0.05, 0.1]
      subsample: [0.8]
      cv_folds: 3
      n_jobs: 1
      grid_n_jobs: 1
      pre_dispatch: 1
```

Trains three models: **LogisticL1**, **LogisticL2**, **XGBoost**. Target can be switched to `became_starter` to change the binary outcome being predicted.

On macOS, XGBoost needs an OpenMP runtime (`libomp.dylib`). The CLI will automatically restart tabular runs with `DYLD_FALLBACK_LIBRARY_PATH` when it finds `libomp` in common Homebrew or Conda locations.

### Text model (`--model text`)

```yaml
model:
  text:
    pretrained: "distilbert-base-uncased"
    output_dim: 128
    max_length: 512
    freeze_base: false            # set true to freeze transformer weights
training:
  batch_size: 32
  lr: 1e-3
  epochs: 50
  early_stopping_patience: 10
```

### Shared training settings

```yaml
training:
  batch_size: 32
  lr: 1e-3
  weight_decay: 1e-4
  epochs: 50
  early_stopping_patience: 10
  grad_clip: 1.0
  seed: 42
```

---

## MLflow Tracking

### What gets logged

Every run logs the following to MLflow, organized as a parent run with one nested child run per model:

**Parent run (top-level summary)**
- Model family, target variable, `use_draft_pick`
- Dataset size, train/test split sizes, test fraction, random seed
- Regression: target mean, std, min, max
- Classification: class balance, positive/negative counts
- Best model name and its test metric (`best_r2` or `best_auc`)
- Python version, sklearn/xgboost/mlflow versions, device
- Full resolved config as `config/config.json` artifact
- `candidate_summary.json` artifact listing every model's test metrics and CV score

**Each child run (one per estimator)**
- Estimator name, target, `use_draft_pick`, random seed
- Final selected hyperparameter (alpha for Lasso/Ridge, best grid params for XGBoost)
- Search space range (alpha min/max/n for linear models; C min/max/n for logistic)
- CV fold count
- XGBoost only: `best_cv_score` (best CV R²/ROC-AUC from grid search)
- Test metrics: R²/RMSE/MAE (regression) or accuracy/ROC-AUC (classification)
- Fitted model artifact

**Local outputs only (not uploaded to MLflow)**
- `regression_results.png` / `classification_results.png`
- `feature_importance.png`
- `importance_heatmap.png`
- `model_summary.png`

### Local plot output

All PNGs for a run are written to:

```
outputs/plots/{run_name}/
```

Each run gets its own subfolder named by the MLflow parent run name, so runs never overwrite each other.

### MLflow UI

```bash
# View runs in the default local store
uv run mlflow ui --backend-store-uri ./mlruns

# Or use the included script
bash scripts/start_mlflow_ui.sh
```

### Shared team tracking

To make runs visible across group members, point everyone to the same backend. Set `MLFLOW_TRACKING_URI` in `.env` or in `config.yaml` under `logging.mlflow.tracking_uri`:

```bash
# Shared server
MLFLOW_TRACKING_URI=http://host:5000

# Shared mounted folder
MLFLOW_TRACKING_URI=/Volumes/SharedDrive/sport-prospect-grading/mlruns
```

MLflow run names and the nested structure (parent + child per estimator) make it straightforward to filter and compare runs across team members in the UI.

---

## Environment Variables

Set these in `.env` at the project root:

| Variable | Description |
|---|---|
| `MLFLOW_TRACKING_URI` | MLflow backend URI (overrides config). Use for shared team tracking. |
| `MLFLOW_ARTIFACT_LOCATION` | Shared artifact root for new experiments. |
| `MLFLOW_RUN_NAME` | Default run name if `--run-name` is not passed. |
| `MODEL_PLOTS_DIR` | Override the local plot output directory. |
| `DEVICE` | Force compute device: `cpu`, `mps`, or `cuda`. |

---

## Data Scripts

```bash
# Fetch NBA season stats
uv run python data/scripts/fetch_nba_stats.py

# Parse NCAA stats from raw data
uv run python data/scripts/parse_ncaa_stats.py

# Reconcile NCAA and NBA data
uv run python data/scripts/reconcile_master.py

# Backfill missing NCAA stats from ESPN box scores (~2GB, ~2 min)
uv run python data/scripts/backfill_ncaa_stats.py
```

---

## Device Detection

`src/utils/device.py` auto-selects the best available device (CUDA > MPS > CPU). Override via `.env`:

```
DEVICE=mps    # force Apple Silicon GPU
DEVICE=cpu    # force CPU-only
```

Device is logged to MLflow as a reproducibility parameter on every run.
