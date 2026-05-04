"""Generate 2026 class multimodal predictions with cleaned inputs.

Run:
  uv run python scripts/predict_2026_multimodal.py
"""
from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import yaml

# Ensure project root is on sys.path when run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import multimodal as mm
from src.data.loader import (
    CLASS_FEATURE,
    HEIGHT_FEATURE,
    POSITION_FEATURE,
    _team_difficulty,
    compute_prospect_context_score,
    parse_height,
)
from src.models.text_model import _clean_player_name

OUT_DIR = PROJECT_ROOT / "outputs" / "multimodal" / "2026_class_predictions"
CONFIG_PATH = PROJECT_ROOT / "src" / "config" / "config.yaml"
PLAYERS_PATH = PROJECT_ROOT / "data" / "scouting" / "players.csv"
PARQUET_URL = (
    "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"
    "espn_mens_college_basketball_player_boxscores/player_box_{season}.parquet"
)

STAT_MAP = {
    "PTS": "points",
    "REB": "rebounds",
    "AST": "assists",
    "ST": "steals",
    "BLKS": "blocks",
    "FGM": "field_goals_made",
    "FGA": "field_goals_attempted",
    "FT": "free_throws_made",
    "FTA": "free_throws_attempted",
    "3FG": "three_point_field_goals_made",
    "3FGA": "three_point_field_goals_attempted",
    "TO": "turnovers",
    "ORebs": "offensive_rebounds",
    "DRebs": "defensive_rebounds",
}

_CLASS_MAP = {
    "freshman": "Fr.",
    "fr": "Fr.",
    "fr.": "Fr.",
    "sophomore": "So.",
    "so": "So.",
    "so.": "So.",
    "junior": "Jr.",
    "jr": "Jr.",
    "jr.": "Jr.",
    "senior": "Sr.",
    "sr": "Sr.",
    "sr.": "Sr.",
}


def normalize_name(name: str) -> str:
    """Normalize player names for stable joins across data sources."""
    text = unicodedata.normalize("NFKD", str(name or ""))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower().strip()
    text = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_players_2026(players: pd.DataFrame) -> pd.DataFrame:
    out = players.loc[players["draft_year"] == 2026].copy()
    if out.empty:
        raise ValueError("No 2026 rows found in data/scouting/players.csv")

    # Mirror canonical workflow name cleaning from src.models.text_model.
    out["name_raw"] = out["name"].astype(str).str.strip()
    out["name"] = out["name_raw"].map(_clean_player_name)
    out["name_norm"] = out["name"].map(normalize_name)
    out["class_year"] = (
        out["class_year"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(_CLASS_MAP)
        .fillna("Fr.")
    )
    out["actual_pick"] = pd.to_numeric(out["actual_pick"], errors="coerce")
    out["big_board_rank"] = pd.to_numeric(out["big_board_rank"], errors="coerce")
    out["draft_pick_clean"] = out["actual_pick"].fillna(out["big_board_rank"]).fillna(60)
    out["position"] = out["position"].astype(str).str.strip()
    out["position_primary"] = out["position"].str.split("/").str[0].str.strip()
    out["height"] = out["height"].astype(str).str.strip()
    return out


def fetch_espn_season_totals(season: int = 2026) -> pd.DataFrame:
    box = pl.read_parquet(PARQUET_URL.format(season=season), use_pyarrow=True)
    aggregates = [
        pl.col("athlete_display_name").first().alias("espn_name"),
        pl.col("team_display_name").first().alias("Team"),
        pl.len().alias("G"),
    ]
    for ncaa_col, espn_col in STAT_MAP.items():
        if espn_col in box.columns:
            aggregates.append(pl.col(espn_col).sum().alias(ncaa_col))

    totals = (
        box.filter(pl.col("did_not_play") == False)
        .group_by("athlete_id")
        .agg(aggregates)
        .to_pandas()
    )
    totals["espn_name"] = totals["espn_name"].astype(str).str.strip()
    totals["name_norm"] = totals["espn_name"].map(normalize_name)

    # Coerce stats into numeric and null-out impossible negatives.
    num_cols = ["G", *STAT_MAP.keys()]
    for col in num_cols:
        totals[col] = pd.to_numeric(totals[col], errors="coerce")
        totals.loc[totals[col] < 0, col] = np.nan
    return totals


def match_players_to_stats(players_2026: pd.DataFrame, totals: pd.DataFrame) -> pd.DataFrame:
    exact = players_2026.merge(
        totals,
        on="name_norm",
        how="left",
        suffixes=("", "_espn"),
    )
    if exact["espn_name"].notna().all():
        return exact

    # Fallback pass for unmatched players: unique last-name match.
    unmatched_mask = exact["espn_name"].isna()
    unmatched = exact.loc[unmatched_mask, ["name", "name_norm"]].copy()
    totals = totals.copy()
    totals["last"] = totals["name_norm"].str.split().str[-1]
    unmatched["last"] = unmatched["name_norm"].str.split().str[-1]

    unique_last = (
        totals.groupby("last", as_index=False)
        .size()
        .query("size == 1")[["last"]]
        .merge(totals, on="last", how="left")
    )
    fallback = unmatched.merge(
        unique_last,
        on="last",
        how="left",
        suffixes=("", "_fallback"),
    )

    for col in ["espn_name", "Team", "G", *STAT_MAP.keys()]:
        exact.loc[unmatched_mask, col] = fallback[col].values
    return exact


def _per_game(total: pd.Series, games: pd.Series) -> pd.Series:
    total = pd.to_numeric(total, errors="coerce")
    games = pd.to_numeric(games, errors="coerce").replace(0, np.nan)
    return total / games


def _pct(made: pd.Series, att: pd.Series) -> pd.Series:
    made = pd.to_numeric(made, errors="coerce")
    att = pd.to_numeric(att, errors="coerce").replace(0, np.nan)
    return 100.0 * made / att


def build_model_input(matched: pd.DataFrame) -> pd.DataFrame:
    pred = pd.DataFrame(
        {
            "Name": matched["name"],
            "draft_year": 2026,
            "draft_pick": matched["draft_pick_clean"],
            "position": matched["position"],
            "Season": "2025-2026",
            "Team": matched["Team"],
            "Cl": matched["class_year"],
            "Pos": matched["position_primary"],
            "Ht": matched["height"],
            "G": matched["G"],
            "FGM": matched["FGM"],
            "3FG": matched["3FG"],
            "FT": matched["FT"],
            "PTS": matched["PTS"],
            "FGA": matched["FGA"],
            "3FGA": matched["3FGA"],
            "FTA": matched["FTA"],
            "REB": matched["REB"],
            "AST": matched["AST"],
            "BLKS": matched["BLKS"],
            "ST": matched["ST"],
            "TO": matched["TO"],
            "ORebs": matched["ORebs"],
            "DRebs": matched["DRebs"],
            "MP": np.nan,
            "MPG": np.nan,
        }
    )

    pred["PPG"] = _per_game(pred["PTS"], pred["G"])
    pred["RPG"] = _per_game(pred["REB"], pred["G"])
    pred["APG"] = _per_game(pred["AST"], pred["G"])
    pred["BKPG"] = _per_game(pred["BLKS"], pred["G"])
    pred["STPG"] = _per_game(pred["ST"], pred["G"])
    pred["FG%"] = _pct(pred["FGM"], pred["FGA"]).clip(0, 100)
    pred["3PG"] = _per_game(pred["3FG"], pred["G"])
    pred["3FG%"] = _pct(pred["3FG"], pred["3FGA"]).clip(0, 100)
    pred["FT%"] = _pct(pred["FT"], pred["FTA"]).clip(0, 100)

    pred[CLASS_FEATURE] = pred[CLASS_FEATURE].astype(str).str.strip().replace(_CLASS_MAP)
    pred[POSITION_FEATURE] = pred[POSITION_FEATURE].astype(str).str.strip()
    pred["height_in"] = pred[HEIGHT_FEATURE].apply(parse_height)
    pos_avg_ht = pred.groupby(POSITION_FEATURE)["height_in"].transform("mean")
    pred["height_dev"] = (pred["height_in"] - pos_avg_ht).abs()

    mpg_col = pd.to_numeric(pred["MPG"], errors="coerce")
    mp_total = pd.to_numeric(pred["MP"], errors="coerce")
    g_nonzero = pd.to_numeric(pred["G"], errors="coerce").replace(0, np.nan)
    pred["mpg_minutes"] = mpg_col.fillna(mp_total / g_nonzero / 60.0)
    pred["team_difficulty_score"] = pred["Team"].map(_team_difficulty)
    pred["prospect_context_score"] = [
        compute_prospect_context_score(team, cl)
        for team, cl in zip(pred["Team"], pred[CLASS_FEATURE])
    ]

    g = pd.to_numeric(pred["G"], errors="coerce").replace(0, np.nan)
    pred["pts_pg"] = pd.to_numeric(pred["PPG"], errors="coerce").fillna(pd.to_numeric(pred["PTS"], errors="coerce") / g)
    pred["reb_pg"] = pd.to_numeric(pred["RPG"], errors="coerce").fillna(pd.to_numeric(pred["REB"], errors="coerce") / g)
    pred["ast_pg"] = pd.to_numeric(pred["APG"], errors="coerce").fillna(pd.to_numeric(pred["AST"], errors="coerce") / g)
    pred["blk_pg"] = pd.to_numeric(pred["BKPG"], errors="coerce").fillna(pd.to_numeric(pred["BLKS"], errors="coerce") / g)
    pred["stl_pg"] = pd.to_numeric(pred["STPG"], errors="coerce").fillna(pd.to_numeric(pred["ST"], errors="coerce") / g)
    pred["fgm_pg"] = pd.to_numeric(pred["FGM"], errors="coerce") / g
    pred["fga_pg"] = pd.to_numeric(pred["FGA"], errors="coerce") / g
    pred["ft_pg"] = pd.to_numeric(pred["FT"], errors="coerce") / g
    pred["fta_pg"] = pd.to_numeric(pred["FTA"], errors="coerce") / g
    pred["fg3_pg"] = pd.to_numeric(pred["3FG"], errors="coerce") / g
    pred["fg_pct"] = pd.to_numeric(pred["FG%"], errors="coerce").fillna(
        (pd.to_numeric(pred["FGM"], errors="coerce") / pd.to_numeric(pred["FGA"], errors="coerce").replace(0, np.nan)) * 100
    )
    pred["ft_pct"] = pd.to_numeric(pred["FT%"], errors="coerce").fillna(
        (pd.to_numeric(pred["FT"], errors="coerce") / pd.to_numeric(pred["FTA"], errors="coerce").replace(0, np.nan)) * 100
    )
    pred["pts_per_fga"] = pd.to_numeric(pred["PTS"], errors="coerce") / pd.to_numeric(pred["FGA"], errors="coerce").replace(0, np.nan)
    pred["ft_rate"] = pd.to_numeric(pred["FTA"], errors="coerce") / pd.to_numeric(pred["FGA"], errors="coerce").replace(0, np.nan)
    pred["fg3_share"] = pd.to_numeric(pred["3FG"], errors="coerce").fillna(0) / pd.to_numeric(pred["FGM"], errors="coerce").replace(0, np.nan)

    turnovers = pd.to_numeric(pred["TO"], errors="coerce")
    pred["to_pg"] = turnovers / g
    pred["oreb_pg"] = pd.to_numeric(pred["ORebs"], errors="coerce") / g
    pred["dreb_pg"] = pd.to_numeric(pred["DRebs"], errors="coerce") / g
    pred["fg3a_pg"] = pd.to_numeric(pred["3FGA"], errors="coerce") / g
    pred["ast_to"] = pd.to_numeric(pred["AST"], errors="coerce") / turnovers.replace(0, np.nan)
    pred["stocks_pg"] = (pd.to_numeric(pred["ST"], errors="coerce").fillna(0) + pd.to_numeric(pred["BLKS"], errors="coerce").fillna(0)) / g
    pred["usage_proxy"] = (
        pd.to_numeric(pred["FGA"], errors="coerce").fillna(0)
        + 0.44 * pd.to_numeric(pred["FTA"], errors="coerce").fillna(0)
        + turnovers.fillna(0)
    ) / g
    fga_safe = pd.to_numeric(pred["FGA"], errors="coerce").replace(0, np.nan)
    pred["efg_pct"] = (
        pd.to_numeric(pred["FGM"], errors="coerce").fillna(0)
        + 0.5 * pd.to_numeric(pred["3FG"], errors="coerce").fillna(0)
    ) / fga_safe
    denom_ts = 2.0 * (
        pd.to_numeric(pred["FGA"], errors="coerce").fillna(0)
        + 0.44 * pd.to_numeric(pred["FTA"], errors="coerce").fillna(0)
    )
    pred["ts_pct"] = pd.to_numeric(pred["PTS"], errors="coerce") / denom_ts.replace(0, np.nan)
    return pred


def render_plots(result: pd.DataFrame, out_dir: Path) -> None:
    mass = result[["p_bust", "p_bench", "p_starter", "p_star"]].mean().rename(
        index={
            "p_bust": "Bust",
            "p_bench": "Bench",
            "p_starter": "Starter",
            "p_star": "Star",
        }
    )
    plt.figure(figsize=(8, 4.8))
    bars = plt.bar(mass.index, mass.values, color=["#d73027", "#fc8d59", "#91bfdb", "#4575b4"])
    plt.title("2026 Class: Average Predicted Tier Probability")
    plt.ylabel("Average probability")
    plt.ylim(0, max(0.4, mass.max() * 1.2))
    for bar, value in zip(bars, mass.values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.005, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "2026_probability_mass.png", dpi=220)
    plt.close()

    top20 = result.head(20).iloc[::-1]
    plt.figure(figsize=(8, 10))
    plt.barh(top20["name"], top20["p_star"], color="#4575b4")
    plt.title("Top 20 2026 Prospects by Star Probability (Multimodal)")
    plt.xlabel("Predicted P(Star)")
    plt.xlim(0, max(0.25, float(top20["p_star"].max()) * 1.1))
    plt.tight_layout()
    plt.savefig(out_dir / "2026_top20_star_probability.png", dpi=220)
    plt.close()


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f) or {}

    print("[2026] training multimodal model...")
    model, _ = mm.run(cfg=cfg, run_name="multimodal_2026_inference")

    print("[2026] loading and cleaning scouting players...")
    players = pd.read_csv(PLAYERS_PATH)
    players_2026 = clean_players_2026(players)

    print("[2026] fetching and cleaning ESPN 2025-26 totals...")
    totals = fetch_espn_season_totals(season=2026)

    print("[2026] matching players to season totals...")
    matched = match_players_to_stats(players_2026, totals)
    unmatched = matched.loc[matched["espn_name"].isna(), ["name", "position", "class_year", "name_norm"]].copy()
    unmatched.to_csv(OUT_DIR / "2026_unmatched_players.csv", index=False)

    print("[2026] building model features...")
    pred_df = build_model_input(matched)

    print("[2026] predicting...")
    proba = model.predict_proba(pred_df)
    labels = np.array(["Bust", "Bench", "Starter", "Star"])
    pred_idx = np.argmax(proba, axis=1)

    result = pd.DataFrame(
        {
            "name": pred_df["Name"],
            "draft_year": pred_df["draft_year"],
            "espn_match_name": matched["espn_name"],
            "pred_tier": labels[pred_idx],
            "confidence": proba.max(axis=1),
            "p_bust": proba[:, 0],
            "p_bench": proba[:, 1],
            "p_starter": proba[:, 2],
            "p_star": proba[:, 3],
        }
    ).sort_values(["p_star", "p_starter", "confidence"], ascending=False)

    csv_path = OUT_DIR / "predictions_2026_multimodal.csv"
    result.to_csv(csv_path, index=False)
    render_plots(result, OUT_DIR)

    print("\n[2026] done")
    print(f"  players: {len(result)}")
    print(f"  matched to ESPN totals: {result['espn_match_name'].notna().sum()}")
    print(f"  unmatched: {len(unmatched)}")
    print(f"  predictions: {csv_path}")
    print(f"  plots: {OUT_DIR / '2026_probability_mass.png'}")
    print(f"  plots: {OUT_DIR / '2026_top20_star_probability.png'}")
    print(f"  unmatched list: {OUT_DIR / '2026_unmatched_players.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
