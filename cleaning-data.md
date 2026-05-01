# Data Cleaning & Pipeline

## Overview

The final dataset contains **787 players** (2009–2023 NBA draft classes) with matched NCAA and NBA records. Both `ncaa_master.csv` and `nba_master.csv` contain exactly these 787 players — the reconciliation step enforces that player sets match exactly.

The pipeline has two independent tracks (NCAA and NBA) that are joined at the end:

```
Raw NCAA leaderboard CSVs
  └─ parse_ncaa_stats.py ──────────────────┐
                                            ├─ ncaa_stats_master.csv
  └─ backfill_ncaa_stats.py ───────────────┘      │
  └─ augment_new_seasons.py (2022/2023) ─────────── ncaa_master.csv
  └─ backfill_profile.py (Cl/Pos/Ht) ─────────────┘

NBA API + Basketball-Reference
  └─ fetch_nba_stats.py ────────────────────┐
  └─ recover_nba_players.py ────────────────┤── nba_stats_best_season.csv
  └─ validate_recovered_players.py ─────────┘

                   │                              │
                   └── reconcile_from_ncaa_master.py ──► nba_master.csv
                                                        ncaa_master.csv (787 players each)
```

---

## NCAA Data Pipeline

### Problem: Leaderboard-Only Source Data

The NCAA player statistics were originally sourced from **NCAA Division I statistical leaderboards**, not complete box scores. Each leaderboard (Points Per Game, Rebounds Per Game, etc.) only lists the top ~350 players nationally for that category. A player only had a value for a stat if they ranked high enough on that specific leaderboard.

This produced severe missingness across the dataset:

| Column | Missing % | Why |
|--------|-----------|-----|
| G (Games) | 0.0% | Always present |
| FT | 23.8% | Scoring-adjacent leaderboard |
| FGM | 26.7% | Scoring leaderboard |
| FGA | 29.9% | Scoring leaderboard |
| 3FG | 32.9% | Three-point leaderboard |
| PTS | 33.2% | Points Per Game leaderboard |
| FTA | 39.3% | Free throw leaderboard |
| REB | 49.7% | Rebounding leaderboard |
| FT% | 57.2% | Free throw % leaderboard |
| Dbl Dbl | 62.5% | Double-doubles leaderboard |
| BLKS | 63.3% | Blocks leaderboard |
| ST | 63.4% | Steals leaderboard |
| BKPG | 65.7% | Blocks leaderboard |
| AST | 67.7% | Assists leaderboard |
| ORebs / DRebs | 68.0% | Rebounding breakdown |
| STPG | 69.2% | Steals leaderboard |
| APG | 71.8% | Assists leaderboard |
| 3FGA | 74.7% | Three-point attempts |
| TO | 76.2% | Turnover/ratio leaderboard |
| 3FG% | 90.1% | Three-point % leaderboard |

56% of players had 4 or more key features missing. 33 players had all features missing.

**Example — DeMar DeRozan (2009, Pick #8):** Appeared on scoring and rebounding leaderboards but not assists, blocks, or steals.

| Feature | Value |
|---------|-------|
| PPG | 13.9 |
| RPG | 5.7 |
| APG | **MISSING** |
| BKPG | **MISSING** |
| STPG | **MISSING** |
| FG% | 52.3 |
| FT% | 64.6 |

**Example — Andy Rautins (2010, Pick #63):** A late second-round pick who only appeared on the assists, steals, and three-point leaderboards — 12 of 17 model features were missing.

### Step 1: Parse Raw CSVs → `ncaa_stats_master.csv`

**Script:** `data/scripts/parse_ncaa_stats.py`

The raw annual NCAA files (`data/ncaa/YYYY-ZZZZ.csv`) contain multiple stat sections separated by NCAA headers (Points Per Game, Rebounds Per Game, etc.), each with its own column layout. This script:

1. Splits each annual file into sections at `"NCAA Men's Basketball"` delimiters
2. Parses each section's column headers (handles quoted fields via `csv.reader`)
3. Merges all sections per season on identity columns `(Name, Team, Cl, Ht, Pos)`, keeping the first non-null value when a column appears in multiple sections
4. Concatenates all seasons into a single `ncaa_stats_master.csv`

### Step 2: Build Initial `ncaa_master.csv`

**Script:** `data/scripts/reconcile_master.py`

From `ncaa_stats_master.csv` and `nba_stats_best_season.csv`, produces matched master files:

- Reads `players_list.txt` as the authoritative player registry
- For each player, selects their final college season (`{Y-1}-{Y}` for a player drafted in year Y)
- Keeps only players present in the player list **and** with data in both sources
- Guards against name collisions across draft classes (e.g., two players named Tony Mitchell)
- Writes `nba_master.csv` and `ncaa_master.csv` with exactly the same player sets

### Step 3: ESPN Box Score Backfill

**Script:** `data/scripts/backfill_ncaa_stats.py`

**Data source:** [sportsdataverse](https://github.com/sportsdataverse/sportsdataverse-data) ESPN box scores — complete per-game stats for every NCAA Division I player since 2002. One parquet file per season, downloaded from GitHub releases.

The script:

1. Downloads parquet files for all seasons in `ncaa_master.csv`
2. Aggregates per-game rows into season totals per player (filtering `did_not_play == False`)
3. Matches NCAA master players to ESPN data using a **4-pass name matching cascade**:
   - **Pass 1:** Exact name match
   - **Pass 2:** Normalized name — strips Jr./Sr./II/III, removes punctuation from initials (T.J. → TJ), collapses whitespace
   - **Pass 3:** Nickname variants (Mohamed → Mo, Christopher → Chris, etc.) + last name, disambiguated by team keyword when multiple matches
   - **Pass 4:** Last name only + team keyword match
4. Fills null raw-stat columns (`PTS`, `REB`, `AST`, `ST`, `BLKS`, `FGM`, `FGA`, `FT`, `FTA`, `3FG`, `3FGA`, `TO`, `ORebs`, `DRebs`) from ESPN aggregated totals
5. Recomputes derived columns (`PPG`, `RPG`, `APG`, `BKPG`, `STPG`, `FG%`, `FT%`, `3FG%`) from the filled raw stats

**Matching results:**

| Metric | Result |
|--------|--------|
| Players matched | 655 / 656 (99.8%) |
| Stat cells filled | 4,697 |
| Unmatched | 1 (Jacob Wiley, Eastern Washington) |

**Cross-validation** — where both sources had data, values matched exactly. For DeMar DeRozan:

| Stat | NCAA Leaderboard | ESPN Box Scores |
|------|-----------------|-----------------|
| PTS | 485 | 485 ✓ |
| REB | 201 | 201 ✓ |
| FGM | 192 | 192 ✓ |
| G | 35 | 35 ✓ |
| AST | *missing* | **51** (recovered) |
| STL | *missing* | **31** (recovered) |
| BLK | *missing* | **13** (recovered) |

### Step 4: Augment New Draft Classes

**Script:** `data/scripts/augment_new_seasons.py`

Adds NCAA stat rows for the **2022 and 2023 draft classes** — players who played in the 2021-22 and 2022-23 college seasons. These cohorts were not in the original leaderboard data. The script:

1. Reads `players_list.txt` and filters to target draft years `[2022, 2023]`
2. Skips players already present in `ncaa_master.csv`
3. Downloads sportsdataverse parquet files for those seasons
4. Applies the same 4-pass name matching (with team keyword disambiguation)
5. Constructs full `ncaa_master.csv`-compatible rows including per-game and percentage columns
6. Appends new rows and re-sorts by draft year / draft pick

Unmatched players from these classes are typically international prospects who played abroad rather than in NCAA Division I.

### Step 5: Profile Backfill for New Seasons

**Script:** `data/scripts/backfill_profile.py`

The ESPN box score source does not carry `Cl` (class year), `Pos` (position abbreviation), or `Ht` (height) — these come from scouting data. This script fills those columns for the 2021-22 and 2022-23 season rows added by `augment_new_seasons.py`:

- Joins `ncaa_master.csv` rows for target seasons against `data/scouting/players.csv` on `(Name, draft_year)`
- Maps full class year strings to NCAA abbreviations (Freshman → Fr., International → ---)
- Maps full position names to abbreviated codes (Point Guard / Shooting Guard → G, etc.)
- Writes height directly from scouting data

---

## NBA Data Pipeline

### Step 1: Fetch Best-Season Stats

**Script:** `data/scripts/fetch_nba_stats.py`

Collects NBA statistics for every player in `players_list.txt`:

1. **Resolves player IDs** using `nba_api`'s local static player lookup (no network call)
2. **Determines eligible seasons** — for a player drafted in year Y, their first 3 NBA seasons are `[Y, Y+1, Y+2]` (e.g., 2009 draft → 2009-10, 2010-11, 2011-12)
3. **Bulk-fetches `LeagueDashPlayerStats`** once per season (PerGame, Regular Season), then looks up each player by ID — avoids one API call per player
4. **Fetches VORP** from Basketball-Reference advanced stats tables (scrapes HTML; handles cases where the table is hidden inside an HTML comment)
5. **Selects the best season** using one of two modes (configured via `data.best_season_mode` in `config.yaml`):

   - **`composite` (default):** Min-max scales each role stat (`MIN`, `PTS`, `REB`, `AST`, `STL`, `BLK`) within the player's 3 candidate seasons, averages the scaled values, picks the highest. NaN stats contribute 0 rather than disqualifying a season. Final fallback: max PTS if all role stats are NaN.
   - **`vorp`:** Picks the season with the highest VORP. Falls back to composite when VORP is unavailable.

**Outputs:**
- `data/nba/nba_stats_best_season.csv` — per-player best season stats
- `data/nba/nba_stats_best_season_vorp.csv` — same plus VORP column

API calls use a 1-second delay between requests to respect rate limits.

### Step 2: Recovery Pass

**Script:** `data/scripts/recover_nba_players.py`

A second-pass resolution for players that `fetch_nba_stats.py` could not find (noted as `not_found_in_api`). Recovery is only attempted for NCAA players — international prospects who never registered in the NBA system are intentionally skipped.

Uses `CommonAllPlayers` (one API call) as the authoritative NBA player database, then applies a **7-strategy matching cascade** with era-windowing:

| Strategy | Method |
|----------|--------|
| 1 | Exact full-name match via `nba_api` static lookup |
| 2 | Unique last-name match within draft era window `[Y-1, Y+3]` |
| 3 | Last-name + first-initial disambiguation when multiple last-name hits |
| 4 | Initials expansion (BJ → B.J., AJ → A.J.) |
| 5 | Fuzzy last-name match (threshold ≥ 0.88), then disambiguate by first initial |
| 6 | Normalised full-name fuzzy match within era (threshold ≥ 0.80) |
| 7 | First/last name swap + fuzzy match |

If no match is found with the narrow era window, strategies 2–7 are retried with a wider window `[Y-2, Y+5]`.

Season data is **cached to disk** at `data/nba/season_cache/{season}.csv` so re-runs are fast.

**Final NBA stats outcomes (full player list):**

| Status | Count |
|--------|-------|
| With stats — original fetch (ok) | 1,090 |
| With stats — recovered | 123 |
| Resolved but no eligible-season data | 103 |
| International, not attempted | 124 |
| NCAA, exhaustively searched, never played | 355 |

The 355 unresolved NCAA players were draft prospects who never appeared in a single NBA regular-season game during their 3-year eligible window.

### Step 3: Validate Recovered Matches

**Script:** `data/scripts/validate_recovered_players.py`

An interactive CLI that presents each recovered player match (drafted name vs. matched NBA name) for human confirmation. Rejections are removed from both `nba_master.csv` and `ncaa_master.csv` atomically, keeping the player sets in sync.

```
#     Drafted Name                   Recovered NBA Name             [Method]
----------------------------------------------------------------------
1     Gal Mekel                      Gal Mekel                      [recovered:exact_static]
  Match correct? (y/n):
```

---

## Reconciliation

### `reconcile_from_ncaa_master.py` (current, preferred)

Treats `ncaa_master.csv` as the authoritative NCAA source (i.e., after all backfill and augment steps) and intersects it with valid NBA stats:

- Keeps only NBA rows with `note` starting with `ok` or `recovered:`
- Builds a `(player_name, draft_year)` key on both sides
- Writes the intersection to `nba_master.csv` and `ncaa_master.csv`
- Asserts that player key sets match exactly

### `reconcile_master.py` (legacy, initial build)

Rebuilds `ncaa_master.csv` from `ncaa_stats_master.csv` (the raw parsed leaderboard data), applying the "last season before draft" selection rule. This was the original reconciliation step used before backfill; `reconcile_from_ncaa_master.py` is now preferred since it treats the already-backfilled `ncaa_master.csv` as the source of truth.

---

## Results After Cleaning

### Null Rates in `ncaa_master.csv`

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

Players with 0 missing features: **5 → 784+**. The remaining ~0.2% nulls are legitimate (Jacob Wiley and a handful of players for whom ESPN box score data is unavailable for that season); the model's median imputer handles these as a safety net rather than as the primary data source.

---

## Feature Engineering

All model features are computed from raw totals in `src/data/loader.py` rather than relying on pre-computed leaderboard per-game columns. Per-game stats are derived as `total / G`; percentages as `made / attempted × 100`.

### Core numeric features (regression and classification)

| Feature | Derivation |
|---------|-----------|
| `pts_pg` | `PPG` (given) or `PTS / G` |
| `reb_pg` | `RPG` (given) or `REB / G` |
| `ast_pg` | `APG` (given) or `AST / G` |
| `blk_pg` | `BKPG` (given) or `BLKS / G` |
| `stl_pg` | `STPG` (given) or `ST / G` |
| `fgm_pg` | `FGM / G` |
| `fga_pg` | `FGA / G` |
| `ft_pg` | `FT / G` |
| `fta_pg` | `FTA / G` |
| `fg3_pg` | `3FG / G` |
| `fg_pct` | `FG%` (given) or `FGM / FGA × 100` |
| `ft_pct` | `FT%` (given) or `FT / FTA × 100` |
| `pts_per_fga` | `PTS / FGA` — shooting efficiency |
| `ft_rate` | `FTA / FGA` — drawing-fouls proxy |
| `fg3_share` | `3FG / FGM` — three-point reliance |
| `G` | Games played (regression only) |
| `height_dev` | `\|height_in − position_avg_height\|` — deviation from position mean |
| `team_difficulty_score` | School strength proxy (0.25–5.0 lookup) |

### Classification-only engineered features (when `use_engineered_features: true`)

| Feature | Derivation |
|---------|-----------|
| `to_pg` | `TO / G` |
| `efg_pct` | `(FGM + 0.5 × 3FG) / FGA` |
| `ts_pct` | `PTS / (2 × (FGA + 0.44 × FTA))` |

### Contextual features

- `prospect_context_score`: school difficulty × class standing^1.5 (configurable via `prospect_context_mode`)
- `position_group`: categorical (G / F / C), optionally included via `use_pos_categorical`
- `draft_pick`: optionally included via `use_draft_pick` (disabled by default — leaks scout judgment)

---

## Reproducing

Run scripts from the project root using `uv run`:

```bash
# 1. Parse raw NCAA leaderboard CSVs into a single master (only needed if raw CSVs change)
uv run python data/scripts/parse_ncaa_stats.py

# 2. Initial reconcile from parsed data (first-time setup only)
uv run python data/scripts/reconcile_master.py

# 3. Backfill missing NCAA stats from ESPN box scores
uv run python data/scripts/backfill_ncaa_stats.py

# 4. Add 2022/2023 draft class rows (first-time setup or when new classes are added)
uv run python data/scripts/augment_new_seasons.py

# 5. Backfill Cl/Pos/Ht for new seasons from scouting data
uv run python data/scripts/backfill_profile.py

# 6. Fetch NBA best-season stats (downloads ~50MB via NBA API, takes ~5 min with rate limiting)
uv run python data/scripts/fetch_nba_stats.py

# 7. Recover players not found in step 6 (second-pass resolution, downloads season cache)
uv run python data/scripts/recover_nba_players.py

# 8. Interactively validate recovered matches
uv run python data/scripts/validate_recovered_players.py

# 9. Reconcile final master files (intersect NCAA + NBA on matched player set)
uv run python data/scripts/reconcile_from_ncaa_master.py

# 10. Run a model
uv run python src/main.py --model regression --run-name baseline
```

Steps 1–5 only need to be re-run if the source NCAA data changes or new draft classes are added. Steps 6–9 only need to be re-run if the NBA data or player list changes. The season cache (`data/nba/season_cache/`) means step 7 is fast on re-runs.
