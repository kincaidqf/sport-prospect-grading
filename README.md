# NBA Draft ML Research

## Goal

Predict NBA draft success using:

- College statistics (Regression)
- Scouting reports (NLP)
- Multimodal model

## Codebase Map

### Core Modules

#### `src/main.py`
Entry point that loads YAML config, resolves compute device, configures MLflow, and dispatches to the selected model pipeline (regression, classification, text, or multimodal).

#### `src/models/`
- **`regression_model.py`**: Lasso/Ridge regression model predicting NBA PLUS_MINUS (best season) from final-year NCAA college stats. Uses Lasso for feature selection due to high multicollinearity.
- **`classification_model.py`**: Logistic/XGBoost classification models for binary draft outcomes like `became_starter` and `survived_3yrs`.
- **`text_model.py`**: NLP encoder (ScoutingReportEncoder) for fine-tuning transformer models on scouting report texts.
- **`multimodal_model.py`**: Combines NCAA stats features + scouting report embeddings for joint prediction.

#### `src/training/`
- **`trainer.py`**: Training loop for neural models with gradient clipping, checkpoint saving, and loss tracking.
- **`evaluate.py`**: Evaluation metrics (MAE, RMSE, R², plus ranking metrics like precision@k).
- **`src/data/loader.py`**: Shared tabular data loading, target derivation, feature engineering, and sklearn preprocessing for stats-based models.
- **`src/utils/features.py`**: Shared feature-name expansion and feature-importance helpers.
- **`src/utils/plotting.py`**: Shared model-summary and feature-importance plotting utilities.
- **`src/utils/mlflow_utils.py`**: Shared MLflow tracking URI resolution, experiment setup, run naming, tags, and per-epoch logging helpers.

#### `src/utils/`
- **`device.py`**: Detects and logs compute device (CPU / MPS / CUDA).

#### `src/config/`
- **`config.yaml`**: Centralized configuration for model type, training hyperparameters, data paths, and output directory.

### Data Processing Scripts

#### `data/scripts/`
- **`fetch_nba_stats.py`**: Fetches season-level NBA stats (PLUS_MINUS, box scores) using nba-api for all players across draft classes.
- **`parse_ncaa_stats.py`**: Parses NCAA stats from raw data into structured DataFrames.
- **`reconcile_master.py`**: Reconciles NCAA and NBA data to link prospects with their professional performance.
- **`recover_nba_players.py`**: Recovers missing NBA players from the master list.

### Data Directories

#### `data/nba/`
- **`nba_master.csv`**: Master list of NBA draft prospects mapped to professional performance.
- **`nba_stats_best_season.csv`**: Best-season NBA stats (highest PLUS_MINUS) for each drafted prospect.
- **`season_cache/`**: Season-by-season NBA stats (2009–10 through 2025–26).

#### `data/ncaa/`
- **`ncaa_master.csv`**: Master list of NCAA prospects.
- **`ncaa_stats_master.csv`**: Aggregated NCAA statistics.
- **`YYYY-ZZZZ.csv`**: Annual NCAA stats files.

### Other
- **`notebooks/01_eda.ipynb`**: Exploratory data analysis notebook.
- **`cleaning-data.md`**: Documents the NCAA stats backfill process and results.
- **`pyproject.toml`**: Project metadata and dependencies (PyTorch, Transformers, scikit-learn, MLflow, Weights & Biases).

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager

### Install & run

```bash
# 1. Clone the repo
git clone <repo-url>
cd sport-prospect-grading

# 2. Install dependencies
uv sync

# 3. Copy and configure environment variables
cp .env.example .env
# Edit .env to set WANDB_API_KEY if using W&B online mode

# 4. Run any supported model through the main entrypoint
uv run python src/main.py --model regression
uv run python src/main.py --model classification
uv run python src/main.py --model text

# 5. (Optional) Launch the MLflow UI to inspect runs in the local default store
uv run mlflow ui --backend-store-uri ./mlruns
```

## MLflow logging

All supported training entrypoints now configure MLflow automatically and log:

- Hyperparameters and dataset split info
- Per-epoch loss curves for the text model
- Final evaluation metrics
- Prediction/summary plots
- Fitted sklearn or PyTorch models
- Run metadata tags including git branch, commit, hostname, and user

Local PNGs are also written per run under:

```text
outputs/plots/{mlflow_parent_run_name}/
```

### Default behavior

If you do nothing, runs are written to a local repo-relative MLflow store under `./mlruns`.

### Shared team tracking

To make runs visible across group members, everyone must point to the same MLflow backend. The simplest options are:

- A shared MLflow server, for example `MLFLOW_TRACKING_URI=http://host:5000`
- A shared mounted folder, for example `MLFLOW_TRACKING_URI=/Volumes/SharedDrive/sport-prospect-grading/mlruns`

Set the same `MLFLOW_TRACKING_URI` for every teammate, either in shell env or `.env`. If you create new experiments against a remote SQL backend, you can also set `MLFLOW_ARTIFACT_LOCATION` to a shared artifact directory.

### Naming conventions

Parent runs use this pattern by default:

```text
{model_type}-{target}-{timestamp}-{user}-{git_sha}
```

Tabular models create nested child runs per estimator like `__lasso`, `__ridge`, `__logisticl1`, and `__xgboost`, which makes model comparisons easy inside one training session.

Generated PNGs are stored in the parent run's local plot directory, so each run's visuals stay separate and do not overwrite each other.

### Helper scripts

You can use the included helpers so the team launches MLflow consistently:

```bash
bash scripts/start_mlflow_ui.sh
bash scripts/start_mlflow_server.sh
```

### Jupyter

```bash
uv run jupyter notebook notebooks/
```

### Data backfill

```bash
# Backfill missing NCAA stats from ESPN box scores (~2GB download, ~2 min)
uv run python data/scripts/backfill_ncaa_stats.py
```

## Device detection

`src/utils/device.py` auto-detects the best available device (MPS on Apple Silicon, CUDA if available, otherwise CPU). Override via the `DEVICE` env var in `.env`:

```
DEVICE=mps   # force Apple Silicon GPU
DEVICE=cpu   # force CPU
```
