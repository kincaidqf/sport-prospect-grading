# NBA Draft ML Research

## Goal

Predict NBA draft success using:

- College statistics (Regression)
- Scouting reports (NLP)
- Multimodal model

## Codebase Map

### Core Modules

#### `src/main.py`
Entry point that loads YAML config, resolves compute device, and dispatches to the selected model pipeline (regression, text, or multimodal).

#### `src/models/`
- **`regression_model.py`**: Lasso/Ridge regression model predicting NBA PLUS_MINUS (best season) from final-year NCAA college stats. Uses Lasso for feature selection due to high multicollinearity.
- **`text_model.py`**: NLP encoder (ScoutingReportEncoder) for fine-tuning transformer models on scouting report texts.
- **`multimodal_model.py`**: Combines NCAA stats features + scouting report embeddings for joint prediction.

#### `src/training/`
- **`trainer.py`**: Training loop for neural models with gradient clipping, checkpoint saving, and loss tracking.
- **`evaluate.py`**: Evaluation metrics (MAE, RMSE, R², plus ranking metrics like precision@k).

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

# 4. Run the regression model
uv run python src/models/regression_model.py

# 5. (Optional) Launch the MLflow UI to inspect runs
uv run mlflow ui
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
