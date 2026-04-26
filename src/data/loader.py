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

DRAFT_PICK_FEATURE = "draft_pick"
HEIGHT_FEATURE     = "Ht"
CLASS_FEATURE      = "Cl"
CLASS_ORDER        = [["Fr.", "So.", "Jr.", "Sr."]]
POSITION_FEATURE   = "Pos"
CONF_FEATURE       = "conf_tier"

POWER_5_TEAMS = {
    # ACC
    "Duke", "North Carolina", "Syracuse", "Louisville", "Virginia", "Florida St.",
    "Georgia Tech", "Boston College", "Clemson", "Notre Dame", "Miami (FL)",
    "Pittsburgh", "Wake Forest", "NC State", "Virginia Tech",
    # Big Ten
    "Michigan", "Michigan St.", "Ohio St.", "Indiana", "Illinois", "Iowa",
    "Minnesota", "Northwestern", "Wisconsin", "Purdue", "Maryland", "Nebraska",
    "Rutgers", "Penn St.",
    # Big 12
    "Kansas", "Texas", "Oklahoma", "Baylor", "Texas Tech", "Oklahoma St.",
    "Iowa St.", "Kansas St.", "West Virginia", "TCU",
    # SEC
    "Kentucky", "Florida", "Alabama", "Arkansas", "Auburn", "Georgia", "LSU",
    "Ole Miss", "Mississippi", "Mississippi St.", "Missouri", "South Carolina",
    "Tennessee", "Texas A&M", "Vanderbilt",
    # Pac-12
    "Arizona", "Arizona St.", "California", "Colorado", "Oregon", "Oregon St.",
    "Stanford", "UCLA", "USC", "Utah", "Washington", "Washington St.",
}

HIGH_MID_MAJOR_TEAMS = {
    "Georgetown", "St. John's", "Seton Hall", "Villanova", "Connecticut",
    "Providence", "Marquette", "DePaul", "Creighton", "Butler", "Xavier",
    "Gonzaga", "Memphis", "Houston", "Cincinnati", "SMU", "Wichita St.",
    "BYU", "San Diego St.", "Nevada", "UNLV", "Utah St.",
    "Dayton", "Davidson", "Rhode Island", "VCU", "Saint Mary's",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def parse_height(ht_str):
    try:
        parts = str(ht_str).strip().split("-")
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            return int(parts[0]) * 12 + int(parts[1])
    except Exception:
        pass
    return np.nan


def assign_conf_tier(team):
    if team in POWER_5_TEAMS:
        return 2
    if team in HIGH_MID_MAJOR_TEAMS:
        return 1
    return 0


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
    df[CONF_FEATURE]     = df["Team"].apply(assign_conf_tier)

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

def build_feature_matrix(df, use_draft_pick=False):
    numeric_cols     = NUMERIC_FEATURES + ["height_in", CONF_FEATURE]
    if use_draft_pick:
        numeric_cols = numeric_cols + [DRAFT_PICK_FEATURE]
    categorical_cols = [POSITION_FEATURE]
    ordinal_cols     = [CLASS_FEATURE]

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ]), numeric_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical_cols),
        ("ord", Pipeline([
            ("imputer",  SimpleImputer(strategy="most_frequent")),
            ("ordinal",  OrdinalEncoder(categories=CLASS_ORDER, handle_unknown="use_encoded_value", unknown_value=-1)),
            ("scaler",   StandardScaler()),
        ]), ordinal_cols),
    ])

    return preprocessor, numeric_cols, categorical_cols, ordinal_cols
