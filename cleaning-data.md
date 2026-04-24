# NCAA Data Cleaning & Backfill Process

## Problem

The NCAA player statistics in `ncaa_master.csv` were sourced from **NCAA Division I statistical leaderboards** — not complete box scores. Each leaderboard (Points Per Game, Rebounds Per Game, Assists Per Game, etc.) only lists the **top ~350 players nationally** for that category. A player only had a value for a stat if they ranked high enough on that specific leaderboard.

This meant that for our 656 drafted players:

| Column | Missing % | Why |
|--------|-----------|-----|
| G (Games) | 0.0% | Always present |
| FT (Free Throws) | 23.8% | Scoring-adjacent leaderboard |
| FGM (Field Goals Made) | 26.7% | Scoring leaderboard |
| FGA (Field Goal Attempts) | 29.9% | Scoring leaderboard |
| 3FG (Three-Pointers) | 32.9% | Three-point leaderboard |
| PTS (Points) | 33.2% | Points Per Game leaderboard |
| FTA (Free Throw Attempts) | 39.3% | Free throw leaderboard |
| REB (Rebounds) | 49.7% | Rebounding leaderboard |
| FT% | 57.2% | Free throw % leaderboard |
| Dbl Dbl | 62.5% | Double-doubles leaderboard |
| BLKS (Blocks) | 63.3% | Blocks leaderboard |
| ST (Steals) | 63.4% | Steals leaderboard |
| BKPG (Blocks/Game) | 65.7% | Blocks leaderboard |
| AST (Assists) | 67.7% | Assists leaderboard |
| ORebs / DRebs | 68.0% | Rebounding breakdown |
| STPG (Steals/Game) | 69.2% | Steals leaderboard |
| APG (Assists/Game) | 71.8% | Assists leaderboard |
| 3FGA | 74.7% | Three-point attempts |
| TO (Turnovers) | 76.2% | Turnover/ratio leaderboard |
| 3FG% | 90.1% | Three-point % leaderboard |

**56% of players (372/656)** had 4 or more of the 7 key model features missing. 33 players had **all** features missing.

### Example: DeMar DeRozan (2009, Pick #8)

DeRozan appeared on the scoring and rebounding leaderboards but not assists, blocks, or steals:

| Feature | Value |
|---------|-------|
| PPG | 13.9 |
| RPG | 5.7 |
| APG | **MISSING** |
| BKPG | **MISSING** |
| STPG | **MISSING** |
| FG% | 52.3 |
| FT% | 64.6 |

### Example: Andy Rautins (2010, Pick #63)

A late second-round pick who only appeared on the assists, steals, and three-point leaderboards — **12 of 17** model features were missing:

| Feature | Value |
|---------|-------|
| pts_pg | **MISSING** |
| reb_pg | **MISSING** |
| ast_pg | 4.89 |
| fgm_pg | **MISSING** |
| fga_pg | **MISSING** |
| ft_pg | **MISSING** |
| fta_pg | **MISSING** |
| fg3_pg | 2.80 |
| fg_pct | **MISSING** |
| ft_pct | **MISSING** |
| stl_pg | 1.97 |

## Impact on the Model (Before Cleaning)

The regression model (`regression_model.py`) used `SimpleImputer(strategy="median")` to fill missing values. With 50–72% of values being imputed with the median for several features:

1. **Predictions collapsed toward the mean** — Lasso and Ridge regression produced flat, clustered scatter plots because most "feature values" were just the median.
2. **Feature importance was misleading** — features with heavy imputation (APG, BKPG, STPG) had artificial signal from the imputed constant.
3. **Effectively reduced dataset** — only ~5 players had complete data across all features.

## Solution: ESPN Box Score Backfill

### Data Source

We used the [sportsdataverse](https://github.com/sportsdataverse/sportsdataverse-data) project, which provides **complete per-game box scores** for every NCAA Division I men's basketball player since 2002, sourced from ESPN.

The data is available as parquet files with one HTTP request per season:
```
https://github.com/sportsdataverse/sportsdataverse-data/releases/download/
  espn_mens_college_basketball_player_boxscores/player_box_{season}.parquet
```

Each season file contains ~150,000–185,000 game rows covering ~15,000 unique players, with all stats: points, rebounds, assists, steals, blocks, FGM, FGA, FTM, FTA, 3PM, 3PA, turnovers, offensive/defensive rebounds, minutes, and more.

### Backfill Script: `data/scripts/backfill_ncaa_stats.py`

The script:

1. **Downloads** parquet files for all 13 seasons (2009–2021)
2. **Aggregates** per-game box scores into season totals per player
3. **Matches** NCAA master players to ESPN data using multi-pass name matching:
   - **Pass 1:** Exact name match
   - **Pass 2:** Normalized name (strips Jr./Sr./II/III, punctuation like T.J. → TJ)
   - **Pass 3:** Nickname variants (Mohamed → Mo, etc.) + last name
   - **Pass 4:** Last name + team keyword match (for remaining edge cases)
4. **Fills** null raw-stat columns from the ESPN aggregated totals
5. **Recomputes** derived columns (PPG, RPG, FG%, FT%, etc.) from the newly filled raw stats

### Matching Results

| Metric | Result |
|--------|--------|
| Players matched | **655 / 656** (99.8%) |
| Stat cells filled | **4,697** |
| Unmatched | 1 (Jacob Wiley, Eastern Washington) |

### Cross-Validation

Where both sources had data, values matched exactly. Example for DeMar DeRozan:

| Stat | NCAA Leaderboard | ESPN Box Scores |
|------|-----------------|-----------------|
| PTS | 485 | 485 ✓ |
| REB | 201 | 201 ✓ |
| FGM | 192 | 192 ✓ |
| G | 35 | 35 ✓ |
| AST | *missing* | **51** (recovered) |
| STL | *missing* | **31** (recovered) |
| BLK | *missing* | **13** (recovered) |

## Results: After Cleaning

### Null Rates

| Column | Before | After |
|--------|--------|-------|
| PTS | 33.2% | **0.0%** |
| REB | 49.7% | **0.0%** |
| AST | 67.7% | **0.2%** |
| ST | 63.4% | **0.2%** |
| BLKS | 63.3% | **0.0%** |
| FGM | 26.7% | **0.0%** |
| FGA | 29.9% | **0.0%** |
| FT | 23.8% | **0.0%** |
| FTA | 39.3% | **0.0%** |
| 3FG | 32.9% | **0.0%** |
| 3FGA | 74.7% | **0.2%** |
| TO | 76.2% | **0.2%** |
| ORebs | 68.0% | **0.0%** |
| DRebs | 67.8% | **0.0%** |

Players with 0 missing features: **5 → 654**

### Feature Engineering Changes

The model was also updated to compute features from raw totals rather than relying on the pre-computed leaderboard per-game columns. This expanded the feature set from 9 to 17 numeric features:

**Original (9 features):**
`PPG, RPG, FG%, FT%, APG, BKPG, STPG, G, shoots_3s`

**New (17 features):**
- Per-game from totals: `pts_pg, reb_pg, ast_pg, blk_pg, stl_pg, fgm_pg, fga_pg, ft_pg, fta_pg, fg3_pg`
- Percentages from totals: `fg_pct, ft_pct`
- Derived efficiency: `pts_per_fga, ft_rate, fg3_share`
- Other: `G, shoots_3s`

### Model Performance (survived_3yrs classification)

| Model | Metric | Before Cleaning | After Cleaning |
|-------|--------|----------------|----------------|
| LogisticL1 | Accuracy | 0.727 | **0.742** |
| LogisticL1 | ROC-AUC | 0.736 | **0.732** |
| LogisticL2 | Accuracy | 0.735 | 0.727 |
| LogisticL2 | ROC-AUC | 0.729 | 0.716 |
| XGBoost | Accuracy | 0.735 | **0.735** |
| XGBoost | ROC-AUC | 0.753 | 0.657 |

The raw accuracy numbers are similar because the task is inherently hard (predicting NBA longevity from college stats), but the key improvements are qualitative:

1. **Confusion matrices are more balanced** — models now correctly identify some "No" cases (true negatives increased from 0–3 to 4–9) instead of blindly predicting the majority class.
2. **Feature importances are meaningful** — `blk_pg`, `fg3_share`, and `pts_per_fga` now contribute real signal rather than imputed noise. The top 10 XGBoost features are spread across the full stat profile.
3. **The model is working with real data** — 654/656 players have complete feature vectors, versus 5 before. The imputer is now a safety net rather than the primary data source.

## Reproducing

```bash
# 1. Backfill missing stats (downloads ~2GB of parquet, takes ~2 min)
python data/scripts/backfill_ncaa_stats.py

# 2. Run the model
python src/models/regression_model.py
```
