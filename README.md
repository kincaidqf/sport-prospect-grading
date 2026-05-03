# NBA Draft ML Research

## Goal

Predict NBA draft success using:

- College statistics (regression and classification)
- Scouting reports (NLP text model)
- Combined stats + scouting (multimodal model)

---

## Model Architecture

This project uses a **probability stacking ensemble** (PSM). Tabular classification and regression base models are trained via out-of-fold cross-validation; their probability outputs are stacked into a meta-feature matrix that a logistic regression meta-model learns to combine.

### Prospect Tier Labels

Tiers are derived from each player's `nba_role_zscore` — a composite of NBA performance statistics:

| Tier | Label | z-score |
|------|-------|---------|
| 0 | Bust | < −0.5 |
| 1 | Bench | −0.5 to 0.5 |
| 2 | Starter | 0.5 to 1.5 |
| 3 | Star | > 1.5 |

---

### Full System Overview

```mermaid
flowchart TD
    subgraph DATA["Data Layer"]
        NCAA["NCAA Stats\nPer-game averages, shooting splits,\nheight, class year, school tier"]
        NBA["NBA Outcomes\nrole z-score converted to tier label"]
    end

    subgraph PREP["Preprocessing Pipeline"]
        MERGE["Merge and Label\nncaa_master joined with nba_master"]
        FE["Feature Engineering\nPer-game normalization, team difficulty score,\nprospect context score, height deviation by position"]
        PIPE["sklearn ColumnTransformer\nMedian imputation → StandardScaler\nOptional: one-hot position, ordinal class year"]
    end

    subgraph BASE["Base Models — 4 total, trained via 5-Fold OOF CV"]
        subgraph CLF["Classification Branch"]
            LR["Logistic Regression\nsoftmax output, log loss"]
            XGB_C["XGBoost Classifier\ngradient boosted trees"]
        end
        subgraph REG["Regression Branch"]
            RIDGE["Ridge Regression\nMSE + L2 penalty"]
            XGB_R["XGBoost Regressor\ngradient boosted trees"]
        end
    end

    subgraph PROBA["Uniform Probability Interface — all models output 4-class probabilities"]
        CLF_P["Classification\npredict_proba() directly"]
        REG_P["Regression\nz-score prediction\n→ Gaussian CDF at thresholds -0.5, 0.5, 1.5\n→ 4-class tier probabilities"]
    end

    META["Meta-Feature Matrix: 16 columns\n4 models × p_bust, p_bench, p_starter, p_star"]
    STACKER["Logistic Regression Meta-Model\ntrained on OOF meta-features"]
    OUT["Final Prediction\nBust / Bench / Starter / Star  +  per-class confidence"]

    DATA --> PREP
    PREP --> BASE
    LR --> CLF_P
    XGB_C --> CLF_P
    RIDGE --> REG_P
    XGB_R --> REG_P
    CLF_P --> META
    REG_P --> META
    META --> STACKER
    STACKER --> OUT
```

---

### Logistic Regression Classifier

A linear model that learns a weighted combination of NCAA features and applies softmax to assign each prospect to one of four tiers. Two regularization variants are trained: **L1** (SAGA solver, drives unimportant feature weights to zero) and **L2** (LBFGS solver, shrinks all weights smoothly).

```mermaid
flowchart TD
    subgraph INPUT["Input Features — preprocessed"]
        FEAT["PPG, RPG, APG, BPG, SPG\nFG%, FT%, 3P%\nHeight Deviation from position average\nTeam Difficulty Score, Prospect Context Score\n~15 to 25 numeric features"]
    end

    subgraph PREP["Preprocessing Pipeline"]
        IMP["Median Imputation\nfills missing stats with column median"]
        SCALE["StandardScaler\nz = (x - mean) / std\ncenters and scales each feature"]
    end

    subgraph MODEL["LogisticRegressionCV"]
        LINEAR["Linear combination per class k\nz_k = w_k dot x + b_k\none weight vector per tier"]
        SOFT["Softmax Activation\nP(k) = exp(z_k) / sum over j of exp(z_j)\nConverts raw class scores to probabilities summing to 1"]
        subgraph REG["Regularization — penalizes large weights to prevent overfitting"]
            L1["L1 variant  (SAGA solver)\nPushes weak feature weights to exactly zero\nautomatically selects relevant features"]
            L2["L2 variant  (LBFGS solver)\nShrinks all weights proportionally\nsmoother, more stable solution"]
        end
        LOSS["Loss: Cross-Entropy\nL = -sum over k of  y_k * log P(k)\nMinimized during training"]
        CV["Inner 5-Fold Cross-Validation\nSearches 20 C values in 0.001 to 100\nC = 1 / regularization strength\nSelects best C automatically"]
    end

    subgraph CAL["Optional Post-hoc Calibration: Platt Scaling"]
        PLATT["Sigmoid function fit on held-out 15% of training data\nRecalibrates confidence scores without refitting the base model"]
    end

    OUT["Output: p_bust, p_bench, p_starter, p_star\nFour probabilities summing to 1.0"]

    INPUT --> PREP
    PREP --> LINEAR
    LINEAR --> SOFT
    L1 -.->|"penalizes weights during training"| LINEAR
    L2 -.->|"penalizes weights during training"| LINEAR
    CV -.->|"selects best C"| MODEL
    SOFT --> LOSS
    SOFT --> CAL
    CAL --> OUT
```

---

### XGBoost Classifier

An ensemble of shallow decision trees trained **sequentially** via gradient boosting. Each new tree corrects the residual errors of all previous trees, producing a powerful non-linear classifier without requiring manual feature engineering.

```mermaid
flowchart TD
    subgraph INPUT["Input Features — preprocessed"]
        FEAT["Same NCAA features as Logistic Regression"]
    end

    subgraph PREP["Preprocessing Pipeline"]
        IMP["Median Imputation"]
        SCALE["StandardScaler"]
    end

    subgraph MODEL["XGBClassifier — GridSearchCV 3-fold CV"]
        subgraph BOOST["Gradient Boosting Ensemble"]
            T1["Tree 1\nInitial rough prediction for all classes"]
            T2["Tree 2\nFits residual errors left by Tree 1"]
            TN["Tree N  (up to 200)\nEach tree corrects all previous errors\nTrees are shallow: max_depth 2 to 3"]
        end

        subgraph HP["Hyperparameters — grid-searched"]
            H1["max_depth: 2 to 3\nLimits tree depth to prevent overfitting"]
            H2["learning_rate: 0.05 to 0.1\nShrinks each tree contribution\nsmaller = more trees needed, less overfit"]
            H3["subsample: 0.7 to 0.9\nRandom row sampling per tree\nreduces correlation between trees"]
            H4["L1 reg_alpha + L2 reg_lambda\nmin_child_weight: 3 to 10\nfurther prevent overfitting on small leaves"]
        end

        PROB["objective: multi:softprob\nSoftmax over summed tree outputs\nProduces calibrated class probabilities"]
        LOSS["Eval metric: mlogloss  (multiclass cross-entropy)\nCV scoring: macro-averaged F1\nFeature importance computed via SHAP"]
    end

    OUT["Output: p_bust, p_bench, p_starter, p_star"]

    INPUT --> PREP
    PREP --> T1
    T1 --> T2
    T2 --> TN
    TN --> PROB
    HP -.->|"controls tree structure and regularization"| BOOST
    PROB --> LOSS
    PROB --> OUT
```

---

### Multi-Modal Probability Stacking

The multi-modal model is a **stacking ensemble**. Each base model independently produces 4-class probability estimates from the same NCAA input features. These are concatenated into a 16-column meta-feature matrix, and a logistic regression meta-model learns the optimal weighted combination across all base models.

Training uses **out-of-fold (OOF) cross-validation** so the meta-model trains on predictions the base models made on data they never saw during their own training — this prevents information leakage.

```mermaid
flowchart TD
    subgraph TRAIN["Full Labeled Training Set"]
        TD["NCAA Stats merged with NBA Outcomes\nprospect_tier: 0=Bust  1=Bench  2=Starter  3=Star"]
    end

    subgraph OOF["Phase 1 — Out-of-Fold Training: 5-Fold Stratified CV"]
        FOLDS["Each fold splits training data:\n80% train  split further into  85% core + 15% calibration hold-out\n20% validation  used only for scoring\nBase models train on core+calibration, predict on validation\nOOF probabilities collected across all 5 folds covering the full training set"]
    end

    subgraph BASE["Base Models — each outputs 4 probability columns per sample"]
        subgraph CLF["Classification Branch — predict_proba() called directly"]
            LR_B["Logistic Regression L2\npredict_proba() outputs p_bust, p_bench, p_starter, p_star"]
            XGB_B["XGBoost Classifier\npredict_proba() outputs p_bust, p_bench, p_starter, p_star"]
        end
        subgraph REG["Regression Branch — z-score converted to probabilities via Gaussian CDF"]
            RIDGE_B["Ridge Regression\npredict() → nba_role_zscore\nGaussian CDF evaluated at thresholds -0.5, 0.5, 1.5\n→ p_bust, p_bench, p_starter, p_star"]
            XGBR_B["XGBoost Regressor\npredict() → nba_role_zscore\nGaussian CDF evaluated at thresholds -0.5, 0.5, 1.5\n→ p_bust, p_bench, p_starter, p_star"]
        end
    end

    subgraph META["Meta-Feature Matrix: N samples × 16 columns"]
        MF["clf__logistic_l2:  p_bust  p_bench  p_starter  p_star\nclf__xgboost:     p_bust  p_bench  p_starter  p_star\nreg__ridge:       p_bust  p_bench  p_starter  p_star\nreg__xgboost:     p_bust  p_bench  p_starter  p_star"]
    end

    subgraph STACK["Phase 2 — Logistic Regression Meta-Model (Stacker)"]
        SLOG["Trained on OOF meta-features\nclass_weight=balanced handles class imbalance\nmax_iter=5000"]
        SSOFT["Softmax → final 4-class probability distribution"]
        SLOSS["Loss: Cross-Entropy\nOOF predictions prevent base-model leakage\n(meta-model only sees out-of-sample base predictions)"]
    end

    OUT["Final Output\nprospect_tier prediction  +  per-class confidence scores"]

    TRAIN --> OOF
    OOF --> BASE
    LR_B --> MF
    XGB_B --> MF
    RIDGE_B --> MF
    XGBR_B --> MF
    MF --> SLOG
    SLOG --> SSOFT
    SSOFT --> SLOSS
    SSOFT --> OUT
```

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

### Known issue: XGBoost runs stuck in MLflow

Regression and classification runs previously appeared to hang at `Running` in the MLflow UI, with no XGBoost output and no PNGs under `outputs/plots/{run_name}/`. The process was not hanging in Python; it was crashing during the XGBoost phase before the parent MLflow run context could close. Because plot generation happens after model training returns, the crash also prevented result plots from being written.

The fix is implemented in code and config:
- XGBoost now runs inside its own nested MLflow child run, so XGBoost work is tracked under `...__xgboost`.
- XGBoost and `GridSearchCV` default to single-worker execution through `n_jobs: 1`, `grid_n_jobs: 1`, and `pre_dispatch: 1`, avoiding unstable nested native parallelism.
- The project is pinned to Python 3.11/3.12 instead of floating to Python 3.13.
- Plotting uses the non-interactive Matplotlib `Agg` backend and closes figures after saving, so training scripts do not block on GUI display.
- On macOS, `src/main.py` restarts tabular runs with a `DYLD_FALLBACK_LIBRARY_PATH` pointing at a detected `libomp.dylib` when needed by XGBoost.

If old runs still show `Running`, they are stale records from a process that crashed before MLflow could mark them finished. New runs should complete normally and write plots to `outputs/plots/{run_name}/`.

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
