"""
Shared data loading and feature engineering for NBA draft prospect models.

Loads NCAA and NBA master CSVs, merges them, engineers features, and builds
sklearn ColumnTransformer preprocessors.
"""

import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder

# ── Paths ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NCAA_PATH    = os.path.join(PROJECT_ROOT, "data", "ncaa", "ncaa_master.csv")
NBA_PATH     = os.path.join(PROJECT_ROOT, "data", "nba", "nba_master.csv")
CACHE_DIR    = os.path.join(PROJECT_ROOT, "data", "nba", "season_cache")

# ── Feature config ─────────────────────────────────────────────────────────────

TEST_SIZE    = 0.2
RANDOM_STATE = 42

# Maps TARGET_MODE strings to DataFrame column names
TARGET_COL = {
    "plus_minus":      "PLUS_MINUS",
    "became_starter":  "became_starter",
    "prospect_tier":   "prospect_tier",
    "composite_score": "composite_score",
}

NUMERIC_FEATURES = [
    # core per-game (computed from totals)
    "pts_pg", "reb_pg", "ast_pg", "blk_pg", "stl_pg",
    "fgm_pg", "fga_pg", "ft_pg", "fta_pg", "fg3_pg",
    # percentages (computed from totals)
    "fg_pct", "ft_pct",
    # derived efficiency
    "pts_per_fga", "ft_rate", "fg3_share",
    # other
    "G", "shoots_3s",
]

# Raw counting-stat equivalents excluded from classification (per user spec:
# PTS, FGA, 3PG, REB, AST, BLKS, ST — Ratio and MP are not in the model).
CLASSIFICATION_EXCLUDED_NUMERIC: frozenset[str] = frozenset({
    "G"
})

DRAFT_PICK_FEATURE      = "draft_pick"
HEIGHT_FEATURE          = "Ht"
HEIGHT_DEV_FEATURE      = "height_dev"
CLASS_FEATURE           = "Cl"
CLASS_ORDER             = [["Fr.", "So.", "Jr.", "Sr."]]
POSITION_FEATURE        = "Pos"

PROSPECT_CONTEXT_FEATURE  = "prospect_context_score"
TEAM_DIFFICULTY_FEATURE   = "team_difficulty_score"
MPG_MINUTES_FEATURE       = "mpg_minutes"
PROSPECT_CONTEXT_MODE     = "individual"  # "composite" | "individual" | "both" | "none"

# ── Prospect Context Score config ──────────────────────────────────────────────

_TIER_1_TEAMS = {"Duke", "Kentucky", "Kansas", "North Carolina", "UCLA", "Arizona", "Michigan St."}
_TIER_2_TEAMS = {"Villanova", "Gonzaga", "Virginia", "Texas", "Baylor", "Florida", "Oregon", "Louisville", "Indiana"}
_TIER_3_TEAMS = {"Arkansas", "Auburn", "Alabama", "Tennessee", "Ohio St.", "Wisconsin", "Illinois",
                 "Texas Tech", "Houston", "Connecticut", "Marquette", "Creighton", "Xavier"}
_TIER_4_TEAMS = {"Seton Hall", "Providence", "Butler", "Saint Mary's", "VCU", "San Diego St.",
                 "Memphis", "Cincinnati", "BYU", "Dayton"}
_TIER_5_TEAMS = {"Georgia Tech", "Boston College", "Wake Forest", "Nebraska", "Minnesota", "DePaul", "Washington St."}

_TEAM_DIFFICULTY: dict[str, float] = {}
for _t, _score in [
    (_TIER_1_TEAMS, 1.00), (_TIER_2_TEAMS, 0.85), (_TIER_3_TEAMS, 0.70),
    (_TIER_4_TEAMS, 0.55), (_TIER_5_TEAMS, 0.40),
]:
    _TEAM_DIFFICULTY.update({team: _score for team in _t})

_CLASS_SCORE: dict[str, float] = {
    "Fr.": 1.00, "Fr": 1.00,
    "So.": 0.80, "So": 0.80,
    "Jr.": 0.60, "Jr": 0.60,
    "Sr.": 0.40, "Sr": 0.40,
}

_MPG_CAP = 30.0


# ── Helpers ────────────────────────────────────────────────────────────────────

def parse_height(ht_str):
    try:
        parts = str(ht_str).strip().split("-")
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            return int(parts[0]) * 12 + int(parts[1])
    except Exception:
        pass
    return np.nan


def _parse_mpg_to_minutes(val) -> float:
    """Convert a raw MPG cell to float minutes/game.

    The NCAA CSV stores MPG in two formats depending on the data source:
      - Numeric string of seconds/game (e.g. '2283.6' → 38.1 min)
      - 'MM:SS' string (e.g. '35:56' → 35.93 min)
    Returns np.nan when the value is missing or unparseable.
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    s = str(val).strip()
    if not s or s == "nan":
        return np.nan
    if ":" in s:
        parts = s.split(":")
        try:
            return float(parts[0]) + float(parts[1]) / 60.0
        except (ValueError, IndexError):
            return np.nan
    try:
        return float(s) / 60.0  # stored as seconds/game
    except ValueError:
        return np.nan


def compute_prospect_context_score(team: str, cl: str, mpg_minutes: float) -> float:
    """Multiplicative scalar: school difficulty × class standing × minutes fraction.

    Returns np.nan when mpg_minutes is NaN so the pipeline imputer can fill it.
    """
    if np.isnan(mpg_minutes):
        return np.nan
    difficulty   = _TEAM_DIFFICULTY.get(str(team).strip(), 0.25)
    class_score  = _CLASS_SCORE.get(str(cl).strip(), np.nan)
    if np.isnan(class_score):
        return np.nan
    minutes_score = min(mpg_minutes / _MPG_CAP, 1.0)
    return difficulty * (class_score ** 1.5) * minutes_score


def _compute_composite_score(df, w_min=0.55, w_gp=0.3, w_pm=0.15, nan_floor=-3.0):
    """Weighted z-score composite of NBA MIN, GP, and PLUS_MINUS.

    Players with no NBA data (all three stats NaN) receive nan_floor so they
    reliably land in the lowest tier after percentile binning.
    """
    def _zscore(s):
        mu, sigma = s.mean(), s.std()
        return (s - mu) / (sigma if sigma > 0 else 1.0)

    composite = w_min * _zscore(df["MIN"]) + w_gp * _zscore(df["GP"]) + w_pm * _zscore(df["PLUS_MINUS"])
    return composite.fillna(nan_floor)


def _assign_tier(composite, percentiles=(40, 80)):
    """Bin composite scores into 0=Bust, 1=Contributor, 2=Star (40/40/20 split).

    Cut points are derived from the given percentiles of the composite
    distribution, making it easy to shift tier boundaries via config.
    """
    cuts = np.percentile(composite, list(percentiles))
    return pd.Series(np.digitize(composite.values, cuts), index=composite.index, dtype=int)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(composite_cfg=None):
    ncaa = pd.read_csv(NCAA_PATH)
    nba  = pd.read_csv(NBA_PATH)

    df = ncaa.merge(
        nba[["player_name", "draft_year", "PLUS_MINUS", "MIN", "GP", "player_id"]],
        left_on=["Name", "draft_year"],
        right_on=["player_name", "draft_year"],
        how="inner",
    )

    _cfg             = composite_cfg or {}
    w_min            = _cfg.get("w_min", 0.35)
    w_gp             = _cfg.get("w_gp", 0.25)
    w_pm             = _cfg.get("w_plus_minus", 0.40)
    tier_percentiles = tuple(_cfg.get("tier_percentiles", (40, 80)))
    nan_floor        = float(_cfg.get("nan_floor", -3.0))

    # Derived targets
    df["became_starter"]  = (df["MIN"] >= 25).astype(int)
    df["composite_score"] = _compute_composite_score(df, w_min=w_min, w_gp=w_gp, w_pm=w_pm, nan_floor=nan_floor)
    df["prospect_tier"]   = _assign_tier(df["composite_score"], percentiles=tier_percentiles)

    # Feature engineering
    df["height_in"] = df[HEIGHT_FEATURE].apply(parse_height)
    df[CLASS_FEATURE]    = df[CLASS_FEATURE].str.strip().replace({"Fr": "Fr.", "So": "So.", "Jr": "Jr.", "Sr": "Sr."})
    df[POSITION_FEATURE] = df[POSITION_FEATURE].str.strip()

    pos_avg_ht = df.groupby(POSITION_FEATURE)["height_in"].transform("mean")
    df[HEIGHT_DEV_FEATURE] = (df["height_in"] - pos_avg_ht).abs()
    # Prospect context: parse MPG column, fall back to total seconds (MP) / G / 60
    _mpg_from_col = df["MPG"].apply(_parse_mpg_to_minutes)
    _mp_total     = pd.to_numeric(df["MP"], errors="coerce")
    _g_nonzero    = df["G"].replace(0, np.nan)
    _mpg_fallback = _mp_total / _g_nonzero / 60.0
    _mpg_minutes  = _mpg_from_col.fillna(_mpg_fallback)

    df[TEAM_DIFFICULTY_FEATURE]  = df["Team"].map(lambda t: _TEAM_DIFFICULTY.get(str(t).strip(), 0.25))
    df[MPG_MINUTES_FEATURE]      = _mpg_minutes.values
    df[PROSPECT_CONTEXT_FEATURE] = [
        compute_prospect_context_score(team, cl, mpg)
        for team, cl, mpg in zip(df["Team"], df[CLASS_FEATURE], _mpg_minutes)
    ]

    g = df["G"].replace(0, np.nan)
    df["pts_pg"] = df["PTS"]  / g
    df["reb_pg"] = df["REB"]  / g
    df["ast_pg"] = df["AST"]  / g
    df["blk_pg"] = df["BLKS"] / g
    df["stl_pg"] = df["ST"]   / g
    df["fgm_pg"] = df["FGM"]  / g
    df["fga_pg"] = df["FGA"]  / g
    df["ft_pg"]  = df["FT"]   / g
    df["fta_pg"] = df["FTA"]  / g
    df["fg3_pg"] = df["3FG"]  / g

    _backfill = [
        ("pts_pg", "PPG"), ("reb_pg", "RPG"), ("ast_pg", "APG"),
        ("blk_pg", "BKPG"), ("stl_pg", "STPG"),
    ]
    for computed, original in _backfill:
        df[computed] = df[computed].fillna(df[original])

    df["fg_pct"] = (df["FGM"] / df["FGA"].replace(0, np.nan)) * 100
    df["fg_pct"] = df["fg_pct"].fillna(df["FG%"])

    df["ft_pct"] = (df["FT"] / df["FTA"].replace(0, np.nan)) * 100
    df["ft_pct"] = df["ft_pct"].fillna(df["FT%"])

    df["pts_per_fga"] = df["PTS"] / df["FGA"].replace(0, np.nan)
    df["ft_rate"]     = df["FTA"] / df["FGA"].replace(0, np.nan)
    df["fg3_share"]   = df["3FG"].fillna(0) / df["FGM"].replace(0, np.nan)
    df["shoots_3s"]   = (df["3FG"].fillna(0) > 0).astype(float)

    print(f"\n{'='*50}")
    print(f"  Data Quality Report  ({len(df)} players)")
    print(f"{'='*50}")
    for col in NUMERIC_FEATURES:
        n_null = df[col].isna().sum()
        print(f"  {col:<14}: {n_null:4d} null ({100*n_null/len(df):5.1f}%)")
    total_null = df[NUMERIC_FEATURES].isna().sum(axis=1)
    print(f"\n  Players with 0 missing features: {(total_null == 0).sum()}")
    print(f"  Players with >50% missing:       {(total_null > len(NUMERIC_FEATURES)//2).sum()}")
    print(f"{'='*50}\n")

    return df


# ── Feature matrix ─────────────────────────────────────────────────────────────

def build_feature_matrix(df, use_draft_pick=False, exclude_features=None, prospect_context_mode=PROSPECT_CONTEXT_MODE):
    _exclude     = frozenset(exclude_features or [])
    numeric_cols = [f for f in NUMERIC_FEATURES + [HEIGHT_DEV_FEATURE, TEAM_DIFFICULTY_FEATURE] if f not in _exclude]
    if use_draft_pick:
        numeric_cols = numeric_cols + [DRAFT_PICK_FEATURE]
    if prospect_context_mode in ("composite", "both"):
        numeric_cols.append(PROSPECT_CONTEXT_FEATURE)
    if prospect_context_mode in ("individual", "both"):
        numeric_cols.append(MPG_MINUTES_FEATURE)
    categorical_cols = []
    # Cl (ordinal class standing) is included only in individual/both modes
    ordinal_cols = [CLASS_FEATURE] if prospect_context_mode in ("individual", "both") else []

    transformers = [
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ]), numeric_cols),
    ]
    if categorical_cols:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical_cols))
    if ordinal_cols:
        transformers.append(("ord", Pipeline([
            ("imputer",  SimpleImputer(strategy="most_frequent")),
            ("ordinal",  OrdinalEncoder(categories=CLASS_ORDER, handle_unknown="use_encoded_value", unknown_value=-1)),
            ("scaler",   StandardScaler()),
        ]), ordinal_cols))
    preprocessor = ColumnTransformer(transformers)

    return preprocessor, numeric_cols, categorical_cols, ordinal_cols
