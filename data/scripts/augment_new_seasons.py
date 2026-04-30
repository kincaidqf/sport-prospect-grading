"""
augment_new_seasons.py

Add NCAA stats rows to ncaa_master.csv for 2022 and 2023 draft prospects
(players who played in the 2021-2022 and 2022-2023 college seasons) using
the same sportsdataverse ESPN box-score parquet files used by backfill_ncaa_stats.py.

Run from project root:
    uv run python data/scripts/augment_new_seasons.py
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

PROJECT_ROOT = Path(__file__).parent.parent.parent
PLAYERS_LIST = PROJECT_ROOT / "data" / "scouting" / "players_list.txt"
NCAA_PATH = PROJECT_ROOT / "data" / "ncaa" / "ncaa_master.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "ncaa" / "ncaa_master.csv"

# Only augment these two new draft years
TARGET_DRAFT_YEARS = [2022, 2023]

PARQUET_URL = (
    "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"
    "espn_mens_college_basketball_player_boxscores/player_box_{season}.parquet"
)

STAT_MAP = {
    "PTS":   "points",
    "REB":   "rebounds",
    "AST":   "assists",
    "ST":    "steals",
    "BLKS":  "blocks",
    "FGM":   "field_goals_made",
    "FGA":   "field_goals_attempted",
    "FT":    "free_throws_made",
    "FTA":   "free_throws_attempted",
    "3FG":   "three_point_field_goals_made",
    "3FGA":  "three_point_field_goals_attempted",
    "TO":    "turnovers",
    "ORebs": "offensive_rebounds",
    "DRebs": "defensive_rebounds",
}


def parse_players_list(path: Path) -> list[dict]:
    out = []
    current_year = None
    line_re = re.compile(r"#\s*(\d+)\s{2,}(.+?)\s{2,}(.+)")
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip()
            ym = re.search(r"---\s*(\d{4})\s+Draft\s*---", line)
            if ym:
                current_year = int(ym.group(1))
                continue
            pm = line_re.search(line)
            if pm and current_year is not None:
                out.append({
                    "Name": pm.group(2).strip(),
                    "draft_year": current_year,
                    "draft_pick": int(pm.group(1)),
                    "position": pm.group(3).strip(),
                })
    return out


def fetch_season_totals(season_int: int) -> pd.DataFrame:
    url = PARQUET_URL.format(season=season_int)
    print(f"  Downloading season {season_int} ...", end=" ", flush=True)
    df = pl.read_parquet(url, use_pyarrow=True)
    print(f"{len(df):,} game rows")

    agg_exprs = [
        pl.col("athlete_display_name").first().alias("espn_name"),
        pl.col("team_display_name").first().alias("espn_team"),
        pl.len().alias("espn_G"),
    ]
    for ncaa_col, espn_col in STAT_MAP.items():
        if espn_col in df.columns:
            agg_exprs.append(pl.col(espn_col).sum().alias(ncaa_col))

    totals = (
        df.filter(pl.col("did_not_play") == False)
        .group_by("athlete_id")
        .agg(agg_exprs)
        .to_pandas()
    )
    totals["season_int"] = season_int
    return totals


def _normalize_name(name) -> str:
    if not isinstance(name, str):
        return ""
    n = name.lower().strip()
    n = re.sub(r"\b(jr\.?|sr\.?|ii|iii|iv)\b", "", n)
    n = n.replace(".", "")
    n = re.sub(r"\s+", " ", n).strip()
    return n


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


def find_match(name: str, team_hint: str, season_int: int, espn_totals: pd.DataFrame):
    """Return the best-matching ESPN row for a player, or None."""
    candidates = espn_totals[espn_totals["season_int"] == season_int].copy()
    if candidates.empty:
        return None

    candidates["_norm"] = candidates["espn_name"].apply(_normalize_name)
    team_keywords = team_hint.lower().split() if isinstance(team_hint, str) else []

    def team_match(espn_team: str) -> bool:
        t = espn_team.lower()
        return any(kw in t for kw in team_keywords) if team_keywords else True

    # Pass 1: exact name
    exact = candidates[candidates["espn_name"] == name]
    if len(exact) == 1:
        return exact.iloc[0]
    if len(exact) > 1:
        for _, row in exact.iterrows():
            if team_match(row["espn_team"]):
                return row
        return exact.iloc[0]

    # Pass 2: normalized name
    norm = _normalize_name(name)
    nm = candidates[candidates["_norm"] == norm]
    if len(nm) == 1:
        return nm.iloc[0]
    if len(nm) > 1:
        for _, row in nm.iterrows():
            if team_match(row["espn_team"]):
                return row
        return nm.iloc[0]

    # Pass 3: nickname variants + last name
    parts = norm.split()
    if len(parts) >= 2:
        first, last = parts[0], parts[-1]
        first_variants = {first}
        if first in _NICKNAMES:
            first_variants.add(_NICKNAMES[first])
        for full, nick in _NICKNAMES.items():
            if first == nick:
                first_variants.add(full)

        for variant in first_variants:
            alt = variant + " " + last
            am = candidates[candidates["_norm"] == alt]
            if len(am) >= 1:
                for _, row in am.iterrows():
                    if team_match(row["espn_team"]):
                        return row
                return am.iloc[0]

        # Pass 4: last-name only + team
        lm = candidates[candidates["_norm"].str.endswith(" " + last) | (candidates["_norm"] == last)]
        if len(lm) >= 1:
            for _, row in lm.iterrows():
                if team_match(row["espn_team"]):
                    return row

    return None


def build_row(meta: dict, espn_row: pd.Series, season_str: str) -> dict:
    """Construct a ncaa_master.csv row from player metadata + ESPN totals."""
    g = int(espn_row["espn_G"]) if pd.notna(espn_row.get("espn_G")) else np.nan

    def stat(col):
        v = espn_row.get(col, np.nan)
        return float(v) if pd.notna(v) else np.nan

    pts  = stat("PTS")
    reb  = stat("REB")
    ast  = stat("AST")
    blks = stat("BLKS")
    st   = stat("ST")
    fgm  = stat("FGM")
    fga  = stat("FGA")
    ft   = stat("FT")
    fta  = stat("FTA")
    fg3  = stat("3FG")
    fg3a = stat("3FGA")
    to_  = stat("TO")
    orebs = stat("ORebs")
    drebs = stat("DRebs")

    def per_game(x):
        return round(x / g, 1) if pd.notna(x) and pd.notna(g) and g > 0 else np.nan

    def pct(made, att):
        return round(100 * made / att, 1) if pd.notna(made) and pd.notna(att) and att > 0 else np.nan

    return {
        "Name":      meta["Name"],
        "draft_year": meta["draft_year"],
        "draft_pick": meta["draft_pick"],
        "position":   meta["position"],
        "Season":     season_str,
        "Team":       espn_row.get("espn_team", np.nan),
        "Cl":         np.nan,
        "Pos":        np.nan,
        "Ht":         np.nan,
        "G":          g,
        "FGM":        fgm,
        "3FG":        fg3,
        "FT":         ft,
        "PTS":        pts,
        "PPG":        per_game(pts),
        "FGA":        fga,
        "FG%":        pct(fgm, fga),
        "3PG":        per_game(fg3),
        "3FGA":       fg3a,
        "3FG%":       pct(fg3, fg3a),
        "FTA":        fta,
        "FT%":        pct(ft, fta),
        "REB":        reb,
        "RPG":        per_game(reb),
        "AST":        ast,
        "APG":        per_game(ast),
        "BLKS":       blks,
        "BKPG":       per_game(blks),
        "ST":         st,
        "STPG":       per_game(st),
        "TO":         to_,
        "Ratio":      np.nan,
        "Trpl Dbl":   np.nan,
        "Dbl Dbl":    np.nan,
        "MP":         np.nan,
        "MPG":        np.nan,
        "ORebs":      orebs,
        "DRebs":      drebs,
    }


def main():
    # Load existing ncaa_master
    ncaa = pd.read_csv(NCAA_PATH)
    existing_keys = set(ncaa["Name"] + "|" + ncaa["draft_year"].astype(str))
    print(f"ncaa_master.csv: {len(ncaa)} existing rows")

    # Parse players_list for target draft years only
    all_players = parse_players_list(PLAYERS_LIST)
    targets = [p for p in all_players if p["draft_year"] in TARGET_DRAFT_YEARS]
    new_players = [p for p in targets if (p["Name"] + "|" + str(p["draft_year"])) not in existing_keys]
    print(f"Target draft years {TARGET_DRAFT_YEARS}: {len(targets)} players, {len(new_players)} not yet in master")

    if not new_players:
        print("Nothing to add.")
        return

    # Determine which seasons to fetch
    season_ints = sorted({p["draft_year"] for p in new_players})  # end-year = draft_year
    print(f"\nFetching ESPN parquet data for seasons: {season_ints}")
    all_totals = []
    for si in season_ints:
        try:
            totals = fetch_season_totals(si)
            all_totals.append(totals)
        except Exception as e:
            print(f"  WARNING: could not fetch season {si}: {e}")
    if not all_totals:
        print("No ESPN data fetched — aborting.")
        return

    espn_totals = pd.concat(all_totals, ignore_index=True)
    print(f"Total ESPN player-seasons fetched: {len(espn_totals):,}")

    # Match and build new rows
    new_rows = []
    matched = []
    unmatched = []

    for player in new_players:
        draft_year = player["draft_year"]
        season_int = draft_year          # sportsdataverse end-year convention
        season_str = f"{draft_year - 1}-{draft_year}"

        espn_row = find_match(
            name=player["Name"],
            team_hint="",               # no team hint from players_list
            season_int=season_int,
            espn_totals=espn_totals,
        )

        if espn_row is not None:
            row = build_row(player, espn_row, season_str)
            new_rows.append(row)
            matched.append(player["Name"])
        else:
            unmatched.append((player["Name"], draft_year))

    print(f"\nMatched: {len(matched)} / {len(new_players)}")
    print(f"Unmatched (likely international or G-League): {len(unmatched)}")

    if unmatched:
        print("\nUnmatched players (sample, first 20):")
        for name, yr in unmatched[:20]:
            print(f"  {name} ({yr} draft)")
        if len(unmatched) > 20:
            print(f"  ... and {len(unmatched) - 20} more")

    if not new_rows:
        print("\nNo new rows to add.")
        return

    # Append and save
    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([ncaa, new_df], ignore_index=True)
    combined = combined.sort_values(["draft_year", "draft_pick"]).reset_index(drop=True)
    combined.to_csv(OUTPUT_PATH, index=False)
    print(f"\nAdded {len(new_rows)} rows. ncaa_master.csv now has {len(combined)} rows.")
    print(f"Saved to {OUTPUT_PATH}")

    # Summary by draft year
    print("\nNew rows by draft year:")
    new_df_summary = new_df.groupby("draft_year").size()
    print(new_df_summary.to_string())


if __name__ == "__main__":
    main()
