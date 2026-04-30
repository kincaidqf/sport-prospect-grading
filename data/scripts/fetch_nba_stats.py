"""
Fetch NBA stats for all players in players_list.txt.

Strategy (efficient):
  1. Determine all unique NBA seasons needed across every draft class.
  2. Pull LeagueDashPlayerStats once per season — gives PLUS_MINUS + full box.
  3. For each player, look up their stats across their 3 eligible seasons and
     keep the single best season (highest PLUS_MINUS).

Draft-year-to-season mapping example:
  2009 draft → 2009-10, 2010-11, 2011-12

Output: data/nba_stats_best_season.csv
"""

import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment
from nba_api.stats.endpoints import leaguedashplayerstats
from nba_api.stats.static import players as nba_players

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent
PLAYERS_LIST = ROOT / "data" / "scouting" / "players_list.txt"
OUTPUT_CSV = ROOT / "data" / "nba" / "nba_stats_best_season.csv"
OUTPUT_CSV_VORP = ROOT / "data" / "nba" / "nba_stats_best_season_vorp.csv"

# Delay between season-level API calls (seconds)
API_DELAY = 1.0

# Stat columns to keep in the output (PLUS_MINUS is the priority)
KEEP_COLS = [
    "PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "GP", "GS", "MIN",
    "FGM", "FGA", "FG_PCT",
    "FG3M", "FG3A", "FG3_PCT",
    "FTM", "FTA", "FT_PCT",
    "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF", "PTS",
    "PLUS_MINUS",
]


# ── Parsing ────────────────────────────────────────────────────────────────────

def normalize_player_name(name: str) -> str:
    """Normalize player names for cross-source matching."""
    cleaned = re.sub(r"[.\']", "", str(name)).lower()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

def parse_players_list(path: Path) -> list[dict]:
    """
    Parse players_list.txt into a list of dicts:
      player_name, draft_year, draft_pick, position
    """
    out = []
    current_year = None
    line_re = re.compile(r"#\s*(\d+)\s{2,}(.+?)\s{2,}(.+)")

    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip()
            year_match = re.search(r"---\s*(\d{4})\s+Draft\s*---", line)
            if year_match:
                current_year = int(year_match.group(1))
                continue
            player_match = line_re.search(line)
            if player_match and current_year is not None:
                out.append({
                    "player_name": player_match.group(2).strip(),
                    "draft_year": current_year,
                    "draft_pick": int(player_match.group(1)),
                    "position": player_match.group(3).strip(),
                })
    return out


# ── Season helpers ─────────────────────────────────────────────────────────────

def draft_year_to_seasons(draft_year: int) -> list[str]:
    """
    Return the 5 NBA season IDs for the first 5 seasons after a draft year.
    e.g. draft_year=2009 → ["2009-10", "2010-11", "2011-12", "2012-13", "2013-14"]
    """
    seasons = []
    for offset in range(5):
        start = draft_year + offset
        seasons.append(f"{start}-{str(start + 1)[-2:]}")
    return seasons


def current_nba_season() -> str:
    """Return the latest completed or current NBA season ID."""
    import datetime
    now = datetime.date.today()
    # NBA season starts in October; if before October use previous year's season
    if now.month >= 10:
        start = now.year
    else:
        start = now.year - 1
    return f"{start}-{str(start + 1)[-2:]}"


# ── Bulk season fetch ──────────────────────────────────────────────────────────

def fetch_season_stats(season: str) -> pd.DataFrame | None:
    """
    Fetch LeagueDashPlayerStats for a single season (PerGame).
    Returns a DataFrame indexed by PLAYER_ID, or None on failure.
    """
    try:
        dash = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            per_mode_detailed="PerGame",
            season_type_all_star="Regular Season",
            timeout=45,
        )
        df = dash.get_data_frames()[0]
        if df.empty:
            return None

        # Keep only the columns we need
        cols_present = [c for c in KEEP_COLS if c in df.columns]
        df = df[cols_present].copy()
        df["SEASON_ID"] = season
        df["PLAYER_NAME_NORM"] = df["PLAYER_NAME"].apply(normalize_player_name)
        df["PLUS_MINUS"] = pd.to_numeric(df["PLUS_MINUS"], errors="coerce")
        df["PTS"] = pd.to_numeric(df["PTS"], errors="coerce")
        return df
    except Exception as exc:
        print(f"  [WARN] Failed to fetch season {season}: {exc}")
        return None


def fetch_vorp_for_season(season: str) -> pd.DataFrame | None:
    """
    Fetch Basketball-Reference advanced stats for a season and return:
      PLAYER_NAME_NORM, VORP
    """
    end_year = int(season.split("-")[1])
    end_year += 2000
    url = f"https://www.basketball-reference.com/leagues/NBA_{end_year}_advanced.html"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        table = soup.find("table", {"id": "advanced"}) or soup.find("table", {"id": "advanced_stats"})
        if table is None:
            # Basketball Reference sometimes stores large tables in HTML comments.
            for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
                if "id=\"advanced\"" in comment or "id=\"advanced_stats\"" in comment:
                    comment_soup = BeautifulSoup(comment, "html.parser")
                    table = comment_soup.find("table", {"id": "advanced"}) or comment_soup.find("table", {"id": "advanced_stats"})
                    if table is not None:
                        break
        if table is None:
            return None

        rows = []
        for tr in table.find("tbody").find_all("tr"):
            if "class" in tr.attrs and "thead" in tr.attrs["class"]:
                continue
            player_cell = tr.find("td", {"data-stat": "player"}) or tr.find("th", {"data-stat": "name_display"}) or tr.find("td", {"data-stat": "name_display"})
            vorp_cell = tr.find("td", {"data-stat": "vorp"})
            if player_cell is None:
                continue
            player_name = player_cell.get_text(strip=True)
            vorp_val = pd.to_numeric(vorp_cell.get_text(strip=True) if vorp_cell else None, errors="coerce")
            rows.append({
                "PLAYER_NAME_NORM": normalize_player_name(player_name),
                "VORP": vorp_val,
            })
        if not rows:
            return None

        advanced = pd.DataFrame(rows).drop_duplicates(subset=["PLAYER_NAME_NORM"], keep="first")
        return advanced[["PLAYER_NAME_NORM", "VORP"]]
    except Exception as exc:
        print(f"  [WARN] Failed to fetch VORP for season {season}: {exc}")
        return None


# ── Player ID resolution ───────────────────────────────────────────────────────

def build_player_id_map(player_list: list[dict]) -> dict[str, int | None]:
    """
    Build a name → player_id mapping for all players in the list.
    Uses nba_api's local static lookup (no network call needed).
    """
    mapping: dict[str, int | None] = {}
    for entry in player_list:
        name = entry["player_name"]
        if name in mapping:
            continue
        results = nba_players.find_players_by_full_name(name)
        if results:
            # Prefer active player; otherwise first match
            active = [p for p in results if p.get("is_active", False)]
            mapping[name] = (active[0] if active else results[0])["id"]
        else:
            mapping[name] = None
    return mapping


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    # 1. Parse player list
    print("Parsing players list …")
    player_list = parse_players_list(PLAYERS_LIST)
    print(f"  {len(player_list)} players across all draft classes\n")

    # 2. Resolve player IDs (local static lookup, no API calls)
    print("Resolving NBA player IDs …")
    id_map = build_player_id_map(player_list)
    found = sum(1 for v in id_map.values() if v is not None)
    print(f"  {found}/{len(id_map)} names resolved\n")

    # 3. Determine all unique seasons needed
    max_season = current_nba_season()
    needed_seasons: set[str] = set()
    for entry in player_list:
        for s in draft_year_to_seasons(entry["draft_year"]):
            # Don't fetch future seasons (or the current season if very early)
            start_year = int(s.split("-")[0])
            max_start = int(max_season.split("-")[0])
            if start_year <= max_start:
                needed_seasons.add(s)

    seasons_sorted = sorted(needed_seasons)
    print(f"Fetching {len(seasons_sorted)} seasons from NBA API …")
    print(f"  Seasons: {seasons_sorted}\n")

    # 4. Bulk-fetch all seasons
    season_data: dict[str, pd.DataFrame] = {}
    season_vorp: dict[str, pd.DataFrame] = {}
    for i, season in enumerate(seasons_sorted, 1):
        print(f"  [{i:02d}/{len(seasons_sorted)}] {season} … ", end="", flush=True)
        df = fetch_season_stats(season)
        if df is not None:
            season_data[season] = df
            vorp_df = fetch_vorp_for_season(season)
            if vorp_df is not None:
                season_vorp[season] = vorp_df
                print(f"{len(df)} players, {len(vorp_df)} VORP rows")
            else:
                print(f"{len(df)} players, VORP unavailable")
        else:
            print("no data / future season")
        time.sleep(API_DELAY)

    print()

    # 5. For each player, find best eligible season
    results = []
    for entry in player_list:
        name = entry["player_name"]
        draft_year = entry["draft_year"]
        player_id = id_map.get(name)
        eligible = draft_year_to_seasons(draft_year)

        base = {**entry, "player_id": player_id}

        if player_id is None:
            results.append({**base, "note": "not_found_in_api"})
            continue

        # Collect rows for this player across eligible seasons
        candidate_rows = []
        for season in eligible:
            if season not in season_data:
                continue
            df = season_data[season]
            row = df[df["PLAYER_ID"] == player_id]
            if not row.empty:
                row_dict = row.iloc[0].to_dict()
                vorp_df = season_vorp.get(season)
                if vorp_df is not None:
                    name_norm = row_dict.get("PLAYER_NAME_NORM")
                    vorp_match = vorp_df[vorp_df["PLAYER_NAME_NORM"] == name_norm]
                    row_dict["VORP"] = (
                        pd.to_numeric(vorp_match.iloc[0]["VORP"], errors="coerce")
                        if not vorp_match.empty else pd.NA
                    )
                else:
                    row_dict["VORP"] = pd.NA
                candidate_rows.append(row_dict)

        if not candidate_rows:
            results.append({
                **base,
                "note": f"no_data_in_{eligible[0]}_to_{eligible[-1]}",
            })
            continue

        # Best season = highest PLUS_MINUS; fall back to PTS if all NaN
        candidates_df = pd.DataFrame(candidate_rows)
        candidates_df["PLUS_MINUS"] = pd.to_numeric(
            candidates_df["PLUS_MINUS"], errors="coerce"
        )

        if candidates_df["PLUS_MINUS"].notna().any():
            best = candidates_df.loc[candidates_df["PLUS_MINUS"].idxmax()]
        else:
            candidates_df["PTS"] = pd.to_numeric(candidates_df["PTS"], errors="coerce")
            best = candidates_df.loc[candidates_df["PTS"].idxmax()]

        results.append({**base, "note": "ok", **best.to_dict()})

    # 6. Write output CSVs
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(results)
    out_df_vorp = out_df.copy()

    # Prioritise key columns first
    priority = [
        "player_name", "draft_year", "draft_pick", "position",
        "player_id", "SEASON_ID", "PLUS_MINUS",
        "GP", "MIN", "PTS", "REB", "AST", "STL", "BLK",
        "FG_PCT", "FG3_PCT", "FT_PCT",
        "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
        "OREB", "DREB", "TOV", "PF",
        "TEAM_ABBREVIATION", "note",
    ]
    ordered = [c for c in priority if c in out_df.columns]
    remaining = [c for c in out_df.columns if c not in ordered and c != "VORP" and c != "PLAYER_NAME_NORM"]
    out_df = out_df[ordered + remaining]

    # Separate VORP output keeps base columns and appends VORP metadata.
    vorp_priority = priority + ["VORP"]
    vorp_ordered = [c for c in vorp_priority if c in out_df_vorp.columns]
    vorp_remaining = [c for c in out_df_vorp.columns if c not in vorp_ordered and c != "PLAYER_NAME_NORM"]
    out_df_vorp = out_df_vorp[vorp_ordered + vorp_remaining]

    out_df.to_csv(OUTPUT_CSV, index=False)
    out_df_vorp.to_csv(OUTPUT_CSV_VORP, index=False)

    # Summary
    ok = (out_df["note"] == "ok").sum()
    not_found = (out_df["note"] == "not_found_in_api").sum()
    no_data = len(out_df) - ok - not_found

    print(f"{'='*60}")
    print(f"Output → {OUTPUT_CSV}")
    print(f"VORP Output → {OUTPUT_CSV_VORP}")
    print(f"Total players      : {len(player_list)}")
    print(f"  With data (ok)   : {ok}")
    print(f"  Not in NBA API   : {not_found}")
    print(f"  No eligible data : {no_data}")

    not_found_names = out_df.loc[out_df["note"] == "not_found_in_api", "player_name"].tolist()
    if not_found_names:
        print(f"\nNot found in NBA API ({len(not_found_names)}):")
        for n in not_found_names[:20]:
            print(f"  {n}")
        if len(not_found_names) > 20:
            print(f"  … and {len(not_found_names) - 20} more")


if __name__ == "__main__":
    main()
