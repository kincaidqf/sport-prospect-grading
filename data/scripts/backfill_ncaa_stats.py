"""
Backfill missing NCAA stats in ncaa_master.csv using sportsdataverse
per-game box scores (ESPN source, parquet files on GitHub).

Downloads one parquet file per season, aggregates season totals per player,
then fills in any null raw-stat columns in ncaa_master.csv by matching on
player name (with team-based disambiguation when needed).

Run from project root:
    python data/scripts/backfill_ncaa_stats.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

PROJECT_ROOT = Path(__file__).parent.parent.parent
NCAA_PATH = PROJECT_ROOT / "data" / "ncaa" / "ncaa_master.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "ncaa" / "ncaa_master.csv"

PARQUET_URL = (
    "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"
    "espn_mens_college_basketball_player_boxscores/player_box_{season}.parquet"
)

# Map ncaa_master Season strings to sportsdataverse season ints.
# sportsdataverse uses the *end* year: "2008-2009" -> 2009
def season_str_to_int(s: str) -> int:
    return int(s.split("-")[1])


# Columns to backfill: ncaa_master column -> sportsdataverse aggregation
STAT_MAP = {
    "PTS":  "points",
    "REB":  "rebounds",
    "AST":  "assists",
    "ST":   "steals",
    "BLKS": "blocks",
    "FGM":  "field_goals_made",
    "FGA":  "field_goals_attempted",
    "FT":   "free_throws_made",
    "FTA":  "free_throws_attempted",
    "3FG":  "three_point_field_goals_made",
    "3FGA": "three_point_field_goals_attempted",
    "TO":   "turnovers",
    "ORebs": "offensive_rebounds",
    "DRebs": "defensive_rebounds",
}

# Derived per-game / pct columns to recompute after backfill
DERIVED = {
    "PPG":  lambda r: r["PTS"] / r["G"] if pd.notna(r["PTS"]) and r["G"] > 0 else np.nan,
    "RPG":  lambda r: r["REB"] / r["G"] if pd.notna(r["REB"]) and r["G"] > 0 else np.nan,
    "APG":  lambda r: r["AST"] / r["G"] if pd.notna(r["AST"]) and r["G"] > 0 else np.nan,
    "BKPG": lambda r: r["BLKS"] / r["G"] if pd.notna(r["BLKS"]) and r["G"] > 0 else np.nan,
    "STPG": lambda r: r["ST"]  / r["G"] if pd.notna(r["ST"])  and r["G"] > 0 else np.nan,
    "FG%":  lambda r: 100 * r["FGM"] / r["FGA"] if pd.notna(r["FGM"]) and pd.notna(r["FGA"]) and r["FGA"] > 0 else np.nan,
    "FT%":  lambda r: 100 * r["FT"]  / r["FTA"] if pd.notna(r["FT"])  and pd.notna(r["FTA"]) and r["FTA"] > 0 else np.nan,
    "3FG%": lambda r: 100 * r["3FG"] / r["3FGA"] if pd.notna(r["3FG"]) and pd.notna(r["3FGA"]) and r["3FGA"] > 0 else np.nan,
}


def fetch_season_totals(season_int: int) -> pd.DataFrame:
    """Download one season's parquet and aggregate to per-player totals."""
    url = PARQUET_URL.format(season=season_int)
    print(f"  Downloading season {season_int} ...", end=" ", flush=True)
    df = pl.read_parquet(url, use_pyarrow=True)
    print(f"{len(df):,} game rows")

    # Aggregate per player
    agg_exprs = [
        pl.col("athlete_display_name").first().alias("espn_name"),
        pl.col("team_display_name").first().alias("espn_team"),
        pl.count().alias("espn_G"),
    ]
    for ncaa_col, espn_col in STAT_MAP.items():
        agg_exprs.append(pl.col(espn_col).sum().alias(ncaa_col))

    totals = (
        df.filter(pl.col("did_not_play") == False)
        .groupby("athlete_id")
        .agg(agg_exprs)
        .to_pandas()
    )
    totals["season_int"] = season_int
    return totals


def _normalize_name(name) -> str:
    """Strip suffixes, punctuation, and normalize for fuzzy matching."""
    import re
    if not isinstance(name, str):
        return ""
    n = name.lower().strip()
    # Remove Jr., Sr., II, III, IV
    n = re.sub(r"\b(jr\.?|sr\.?|ii|iii|iv)\b", "", n)
    # Remove periods from initials (T.J. -> TJ)
    n = n.replace(".", "")
    # Collapse whitespace
    n = re.sub(r"\s+", " ", n).strip()
    return n


# Common NCAA→ESPN nickname mappings
_NICKNAMES = {
    "mohamed": "mo",
    "robert": "rob",
    "william": "will",
    "james": "jim",
    "joseph": "joe",
    "nicholas": "nick",
    "nicolas": "nic",
    "christopher": "chris",
    "kenneth": "ken",
    "timothy": "tim",
    "daniel": "dan",
    "anthony": "ant",
}


def match_players(ncaa: pd.DataFrame, espn_totals: pd.DataFrame) -> dict:
    """
    Match ncaa_master rows to ESPN player totals.
    Returns {ncaa_index: espn_row} for matched players.

    Three-pass matching:
      1. Exact name
      2. Normalized name (strip Jr./Sr./III, punctuation)
      3. Last-name + team-keyword match
    """
    matches = {}

    # Pre-compute normalized ESPN names
    espn_totals = espn_totals.copy()
    espn_totals["_norm"] = espn_totals["espn_name"].apply(_normalize_name)

    for idx, row in ncaa.iterrows():
        season_int = season_str_to_int(row["Season"])
        name = row["Name"]
        ncaa_team = row["Team"].lower()

        candidates = espn_totals[espn_totals["season_int"] == season_int]

        # Pass 1: exact name
        exact = candidates[candidates["espn_name"] == name]
        if len(exact) == 1:
            matches[idx] = exact.iloc[0]
            continue
        if len(exact) > 1:
            matches[idx] = exact.iloc[0]
            continue

        # Pass 2: normalized name (handles Jr., T.J., etc.)
        norm = _normalize_name(name)
        norm_match = candidates[candidates["_norm"] == norm]
        if len(norm_match) == 1:
            matches[idx] = norm_match.iloc[0]
            continue
        if len(norm_match) > 1:
            # Disambiguate by team keyword
            for _, cand in norm_match.iterrows():
                if any(kw in cand["espn_team"].lower() for kw in ncaa_team.split()):
                    matches[idx] = cand
                    break
            else:
                matches[idx] = norm_match.iloc[0]
            continue

        # Pass 3: nickname first-name swap + last-name match
        parts = norm.split()
        if len(parts) >= 2:
            first, last = parts[0], parts[-1]
            # Try nickname variants of first name
            first_variants = {first}
            if first in _NICKNAMES:
                first_variants.add(_NICKNAMES[first])
            # Also check reverse mapping
            for full, nick in _NICKNAMES.items():
                if first == nick:
                    first_variants.add(full)

            for variant in first_variants:
                alt_norm = variant + " " + last
                alt_match = candidates[candidates["_norm"] == alt_norm]
                if len(alt_match) >= 1:
                    # Prefer team keyword match
                    for _, cand in alt_match.iterrows():
                        if any(kw in cand["espn_team"].lower() for kw in ncaa_team.split()):
                            matches[idx] = cand
                            break
                    else:
                        matches[idx] = alt_match.iloc[0]
                    break

            if idx in matches:
                continue

            # Pass 4: last-name only + team keyword (catches remaining)
            last_match = candidates[candidates["_norm"].str.endswith(" " + last)]
            if len(last_match) >= 1:
                for _, cand in last_match.iterrows():
                    espn_team_lower = cand["espn_team"].lower()
                    if any(kw in espn_team_lower for kw in ncaa_team.split()):
                        matches[idx] = cand
                        break

    return matches


def backfill(ncaa: pd.DataFrame, matches: dict) -> pd.DataFrame:
    """Fill null raw-stat columns from ESPN data, then recompute derived cols."""
    filled_count = 0

    for idx, espn_row in matches.items():
        for ncaa_col in STAT_MAP:
            if pd.isna(ncaa.at[idx, ncaa_col]) and pd.notna(espn_row[ncaa_col]):
                ncaa.at[idx, ncaa_col] = float(espn_row[ncaa_col])
                filled_count += 1

        # Also backfill G if somehow missing
        if pd.isna(ncaa.at[idx, "G"]) and pd.notna(espn_row["espn_G"]):
            ncaa.at[idx, "G"] = int(espn_row["espn_G"])

    # Recompute derived columns where the raw stat was just filled
    for col, func in DERIVED.items():
        for idx in matches:
            if pd.isna(ncaa.at[idx, col]):
                ncaa.at[idx, col] = func(ncaa.loc[idx])

    return ncaa, filled_count


def main():
    ncaa = pd.read_csv(NCAA_PATH)
    seasons_needed = sorted(ncaa["Season"].unique())
    season_ints = [season_str_to_int(s) for s in seasons_needed]

    print(f"ncaa_master.csv: {len(ncaa)} players, {len(seasons_needed)} seasons")
    print(f"Seasons to fetch: {season_ints}\n")

    # Print before stats
    raw_cols = list(STAT_MAP.keys())
    print("BEFORE backfill — null counts:")
    for col in raw_cols:
        n = ncaa[col].isna().sum()
        print(f"  {col:<6}: {n:4d} ({100*n/len(ncaa):.1f}%)")

    # Fetch all seasons
    print("\nFetching ESPN box scores...")
    all_totals = []
    for si in season_ints:
        all_totals.append(fetch_season_totals(si))
    espn_totals = pd.concat(all_totals, ignore_index=True)
    print(f"\nTotal ESPN player-seasons: {len(espn_totals):,}")

    # Match
    print("\nMatching players...")
    matches = match_players(ncaa, espn_totals)
    print(f"Matched {len(matches)} / {len(ncaa)} players")

    # Backfill
    ncaa, filled_count = backfill(ncaa, matches)
    print(f"Filled {filled_count} individual stat cells\n")

    # Print after stats
    print("AFTER backfill — null counts:")
    for col in raw_cols:
        n = ncaa[col].isna().sum()
        print(f"  {col:<6}: {n:4d} ({100*n/len(ncaa):.1f}%)")

    # Also show derived columns
    print("\nDerived columns:")
    for col in DERIVED:
        n = ncaa[col].isna().sum()
        print(f"  {col:<6}: {n:4d} ({100*n/len(ncaa):.1f}%)")

    # Show unmatched players
    unmatched_idx = set(range(len(ncaa))) - set(matches.keys())
    if unmatched_idx:
        print(f"\nUnmatched players ({len(unmatched_idx)}):")
        for idx in sorted(unmatched_idx)[:20]:
            row = ncaa.iloc[idx]
            print(f"  {row['Name']} ({row['Season']}, {row['Team']})")
        if len(unmatched_idx) > 20:
            print(f"  ... and {len(unmatched_idx) - 20} more")

    # Save
    ncaa.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
