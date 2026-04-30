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
- **`text_model.py`**: NLP encoder (ScoutingReportEncoder) for fine-tuning transformer models on scouting report texts. Pass `save_path=` to `train_and_evaluate_text_model` to persist weights for interpretability.
- **`interpret_text.py`**: Probes, aggregated occlusion, corpus log-odds, VADER sentiment correlations, and `outputs/interpretability/REPORT.md` for all prediction heads.
- **`multimodal_model.py`**: Combines NCAA stats features + scouting report embeddings for joint prediction.

#### `src/training/`
- **`trainer.py`**: Training loop for neural models with gradient clipping, checkpoint saving, and loss tracking.
- **`evaluate.py`**: Evaluation metrics (MAE, RMSE, R², plus ranking metrics like precision@k).

#### `src/data/`
- **`dataset.py`**: PyTorch Dataset classes (`ProspectStatsDataset`, `ScoutingReportDataset`) for loading NCAA stats and tokenized scouting reports.
- **`preprocessor.py`**: Feature engineering and train/val/test splitting with StandardScaler for numerical features.
- **`players.csv` / `players.jsonl`**: Master prospect data with scouting ratings and attributes.

#### `src/utils/`
- **`device.py`**: Utility to detect and log compute device (CPU/GPU).

#### `src/config/`
- **`config.yaml`**: Centralized YAML configuration for model type, training hyperparameters, data paths, and output directory.

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

### Configuration & Infrastructure

- **`Makefile`**: Build rules for common tasks.
- **`pyproject.toml`**: Project metadata and dependencies (PyTorch, Transformers, scikit-learn, MLflow, Weights & Biases).
- **`docker/`**: Docker configurations for containerized GPU/CPU training environments.
- **`notebooks/01_eda.ipynb`**: Exploratory data analysis notebook.
- **`mlruns/`**: MLflow experiment tracking artifacts and model snapshots.

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (package manager)
- Docker + Docker Compose (optional, for containerized runs)

### Local (recommended)

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

### Docker (CPU)

```bash
make build
make train
```

### Docker (GPU / RTX 5090)

```bash
make build-gpu
make train-gpu
```

### Text model interpretability

Train with a checkpoint path, then run interpretability (from repo root):

```bash
uv run python -c "from src.models.text_model import train_and_evaluate_text_model; train_and_evaluate_text_model(save_path='outputs/checkpoints/text_model.pt')"

uv run python -m src.models.interpret_text --checkpoint outputs/checkpoints/text_model.pt
```

Fast run (smaller/faster occlusion):

```bash
uv run python -m src.models.interpret_text --checkpoint outputs/checkpoints/text_model.pt --n-occlusion 20 --max-variants-per-report 40
```

Use `--retrain --checkpoint-out ...` to train and interpret in one step. Outputs land in `outputs/interpretability/`.

### Jupyter Notebook

```bash
# Local
uv run jupyter notebook notebooks/

# Docker
make notebook
```
