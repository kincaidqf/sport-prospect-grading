# Merge Notes: feat/classification-distribution → main

## Date
2026-04-30

## Summary
`feat/classification-distribution` (35 commits ahead of main) was fast-forward merged into main. No merge conflicts — main was a strict ancestor of the feature branch.

## What Changed

### New core modules (not on main)
- `src/models/classification_model.py` — full classification trainer (XGBoost + logistic regression), position-relative and global modes
- `src/models/classification_inference.py` — inference pipeline for saved classification models
- `src/data/loader.py` — shared data loading and feature engineering (replaces per-model loaders)
- `src/utils/features.py` — feature name resolution and importance helpers
- `src/utils/mlflow_utils.py` — shared MLflow context management
- `src/utils/plotting.py` — shared plotting utilities
- `src/training/splits.py` — chronological, random, and repeated-CV split strategies

### Files superseded / removed from main
| File (main) | Superseded by |
|---|---|
| `src/models/multimodal_model.py` (44 lines) | Placeholder; not yet implemented. Removed to avoid confusion — multimodal architecture is planned in `PROBABLITY_STACKING_MODEL.md` / `PSM_PHASE1.md` |
| `src/training/trainer.py` (59 lines) | Replaced by per-model `run()` entry points called from `main.py` |
| `docker/` directory (4 files) | Docker CPU/GPU setup removed; training runs locally via `uv run` |
| `Makefile` | Replaced by direct `uv run python -m src.main` invocations |
| `.env` | Never committed; `.env.example` preserved |

### Data changes
- `data/ncaa/ncaa_master.csv` — overwritten with current branch version (adds 2021-22 and 2022-23 NCAA seasons, backfilled position and class columns)
- `data/nba/nba_master.csv` — updated player records and nba_role_zscore column
- `data/nba/nba_stats_best_season.csv` — updated

### Data scripts preserved (all main scripts present + additions)
All scripts from main are preserved. Additional scripts added:
- `data/scripts/augment_new_seasons.py` — scrapes and appends new NCAA seasons
- `data/scripts/backfill_profile.py` — backfills position/class columns for older seasons
- `data/scripts/reconcile_from_ncaa_master.py` — reconciles player identities from ncaa_master

### text_model.py
The `feat/classification-distribution` version (470 lines) is significantly more complete than the main version (391 lines). It adds full MLflow integration (`managed_run`, `log_epoch_metrics`, etc.) while preserving the same model architecture (`ScoutingReportEncoder`, `TextProspectPredictor`).

### Tests added
54 tests across 5 files covering:
- Config structure and value validation
- Data loader CSV integrity, merge, feature matrix
- Evaluation metrics (regression and classification)
- Split strategies (random, chronological, repeated CV)
- Model public API contracts (imports, run() signatures, target mode constants)
- CLI dispatch (main.py argparse and routing)
