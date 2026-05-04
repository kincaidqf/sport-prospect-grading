"""
Fetch NBA stats for all players in players_list.txt.

Strategy (efficient):
  1. Determine all unique NBA seasons needed across every draft class.
  2. Pull LeagueDashPlayerStats once per season — gives PLUS_MINUS + full box.
  3. For each player, look up their stats across their first 3 eligible seasons
     and keep the single best season by composite role score (min-max scaled
     average of MIN, PTS, REB, AST, STL, BLK).

Per-season enrichment (optional best-season targets vs composite):
  - VORP from Basketball-Reference advanced tables (name match).
  - DARKO DPM (total only) from databallr (NBA player_id match; same calendar end-year as BRef).

Draft-year-to-season mapping example:
  2009 draft → 2009-10, 2010-11, 2011-12

Outputs (under data/nba/):
  - nba_stats_best_season.csv — base stats (no VORP/DPM columns).
  - nba_stats_best_season_vorp.csv — same rows plus VORP only.
  - nba_stats_best_season_darko.csv — same rows plus DPM (DARKO) only.

Run from repository root::

    PYTHONPATH=. python data/scripts/fetch_nba_stats.py

Optional: set ``DATABALLR_API_KEY`` and ``DATABALLR_HMAC_SECRET`` in the environment
if databallr rotates credentials (defaults match the public web client).
"""

import hashlib
import hmac
import os
import re
import time
import uuid
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import requests
import yaml
from bs4 import BeautifulSoup, Comment
from nba_api.stats.endpoints import leaguedashplayerstats
from nba_api.stats.static import players as nba_players

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent
PLAYERS_LIST = ROOT / "data" / "scouting" / "players_list.txt"
OUTPUT_CSV = ROOT / "data" / "nba" / "nba_stats_best_season.csv"
OUTPUT_CSV_VORP = ROOT / "data" / "nba" / "nba_stats_best_season_vorp.csv"
OUTPUT_CSV_DARKO = ROOT / "data" / "nba" / "nba_stats_best_season_darko.csv"
CONFIG_PATH = ROOT / "src" / "config" / "config.yaml"

# Delay between season-level API calls (seconds)
API_DELAY = 1.0

# databallr (https://databallr.com) — X-API-Key + HMAC secret match the public web client.
# Override with DATABALLR_API_KEY / DATABALLR_HMAC_SECRET if they rotate.
DATABALLR_API_BASE = "https://api.databallr.com/api"
_DATABALLR_KEY_DEFAULT = "d4c3b65c6bc3db82b1bf0bcf11659bfe2413c5bfdfdec5f545941e2a93a848aa"
_DATABALLR_HMAC_DEFAULT = "a7f8e2c9d4b1f6a3e0c5d8b2f7a4e1c6d9b3f0a5e8c2d7b4f1a6e3c0d5b8f2a9"

# Stats used for best-season selection (min-max scaled composite)
ROLE_STATS = ["MIN", "PTS", "REB", "AST", "STL", "BLK"]

# Stat columns to keep in the output
KEEP_COLS = [
    "PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "GP", "GS", "MIN",
    "FGM", "FGA", "FG_PCT",
    "FG3M", "FG3A", "FG3_PCT",
    "FTM", "FTA", "FT_PCT",
    "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF", "PTS",
    "PLUS_MINUS",
]


# ── Config ─────────────────────────────────────────────────────────────────────

def load_best_season_mode() -> str:
    """Read data.best_season_mode from config.yaml; default composite."""
    try:
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("data", {}).get("best_season_mode", "composite")
    except Exception:
        return "composite"


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
    Return the 3 NBA season IDs for the first 3 seasons after a draft year.
    e.g. draft_year=2009 → ["2009-10", "2010-11", "2011-12"]
    """
    seasons = []
    for offset in range(3):
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


def _nba_season_to_calendar_end_year(season: str) -> int:
    """e.g. '2024-25' → 2025 (matches Basketball-Reference league year)."""
    yy = int(season.split("-")[1])
    return yy + 2000


def _databallr_request_headers(method: str, path_with_query: str) -> dict[str, str]:
    """HMAC signing for /supabase/* routes (same scheme as databallr web client)."""
    api_key = os.environ.get("DATABALLR_API_KEY", _DATABALLR_KEY_DEFAULT)
    sign_key = os.environ.get("DATABALLR_HMAC_SECRET", _DATABALLR_HMAC_DEFAULT).encode()
    path_part, _, query = path_with_query.partition("?")
    sign_path = "/api" + path_part
    ts = str(int(time.time()))
    nonce = str(uuid.uuid4())
    payload = f"{ts}:{nonce}:{method.upper()}:{sign_path}:{query}"
    sig = hmac.new(sign_key, payload.encode(), hashlib.sha256).hexdigest()
    return {
        "X-API-Key": api_key,
        "X-Timestamp": ts,
        "X-Nonce": nonce,
        "X-Signature": sig,
        "Content-Type": "application/json",
        "User-Agent": "sport-prospect-grading/fetch_nba_stats",
    }


def fetch_darko_databallr_for_season(season: str) -> pd.DataFrame | None:
    """
    Fetch DARKO total DPM from databallr player_stats_with_metrics for one NBA season.

    Returns PLAYER_ID, DPM (deduped by PLAYER_ID), or None.
    """
    year = _nba_season_to_calendar_end_year(season)
    path = "/supabase/player_stats_with_metrics"
    all_rows: list[dict] = []
    page_size = 5000
    offset = 0

    try:
        while True:
            params = {
                "year": str(year),
                "playoffs": "false",
                "limit": str(page_size),
                "offset": str(offset),
                "select_fields": "nba_id,dpm",
                "order_by": "nba_id",
                "order_direction": "asc",
            }
            rel = f"{path}?{urlencode(params)}"
            url = DATABALLR_API_BASE + rel
            headers = _databallr_request_headers("GET", rel)
            resp = requests.get(url, headers=headers, timeout=60)
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            all_rows.extend(batch)
            if len(batch) < page_size:
                break
            offset += len(batch)
    except Exception as exc:
        print(f"  [WARN] Failed to fetch databallr DARKO for season {season}: {exc}")
        return None

    if not all_rows:
        return None

    df = pd.DataFrame(all_rows)
    if "nba_id" not in df.columns:
        return None
    df = df.rename(columns={"nba_id": "PLAYER_ID", "dpm": "DPM"})
    if "DPM" in df.columns:
        df["DPM"] = pd.to_numeric(df["DPM"], errors="coerce")
    df["PLAYER_ID"] = pd.to_numeric(df["PLAYER_ID"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["PLAYER_ID"]).drop_duplicates(subset=["PLAYER_ID"], keep="first")
    return df[["PLAYER_ID", "DPM"]].reset_index(drop=True)


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
    mode = load_best_season_mode()
    print(f"Best-season mode: {mode}\n")

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
    season_darko: dict[str, pd.DataFrame] = {}
    for i, season in enumerate(seasons_sorted, 1):
        print(f"  [{i:02d}/{len(seasons_sorted)}] {season} … ", end="", flush=True)
        df = fetch_season_stats(season)
        if df is not None:
            season_data[season] = df
            vorp_df = fetch_vorp_for_season(season)
            if vorp_df is not None:
                season_vorp[season] = vorp_df
            darko_df = fetch_darko_databallr_for_season(season)
            if darko_df is not None:
                season_darko[season] = darko_df
            v_msg = f"{len(vorp_df)} VORP" if vorp_df is not None else "VORP n/a"
            d_msg = f"{len(darko_df)} DPM" if darko_df is not None else "DPM n/a"
            print(f"{len(df)} players, {v_msg}, {d_msg}")
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
                darko_df = season_darko.get(season)
                pid = row_dict.get("PLAYER_ID")
                if darko_df is not None and pid is not None and not pd.isna(pid):
                    dm = darko_df[darko_df["PLAYER_ID"] == int(pid)]
                    if not dm.empty:
                        row_dict["DPM"] = dm.iloc[0].get("DPM", pd.NA)
                    else:
                        row_dict["DPM"] = pd.NA
                else:
                    row_dict["DPM"] = pd.NA
                candidate_rows.append(row_dict)

        if not candidate_rows:
            results.append({
                **base,
                "note": f"no_data_in_{eligible[0]}_to_{eligible[-1]}",
            })
            continue

        candidates_df = pd.DataFrame(candidate_rows)
        for col in ROLE_STATS:
            candidates_df[col] = pd.to_numeric(candidates_df[col], errors="coerce")
        if "VORP" in candidates_df.columns:
            candidates_df["VORP"] = pd.to_numeric(candidates_df["VORP"], errors="coerce")
        if "DPM" in candidates_df.columns:
            candidates_df["DPM"] = pd.to_numeric(candidates_df["DPM"], errors="coerce")

        best = None

        # vorp mode: pick the season with the highest VORP.
        if mode == "vorp" and "VORP" in candidates_df.columns and candidates_df["VORP"].notna().any():
            best = candidates_df.loc[candidates_df["VORP"].idxmax()]

        # darko mode: pick the season with the highest DARKO DPM.
        if best is None and mode == "darko" and "DPM" in candidates_df.columns and candidates_df["DPM"].notna().any():
            best = candidates_df.loc[candidates_df["DPM"].idxmax()]

        # composite mode (also fallback when impact metrics unavailable):
        # min-max scale each role stat within the player's candidate seasons,
        # average the scaled values, pick the highest. NaN stats contribute 0.
        # Final fallback to max PTS if all role stats are also NaN.
        if best is None:
            has_role_data = candidates_df[ROLE_STATS].notna().any().any()
            if has_role_data:
                scaled = pd.DataFrame(index=candidates_df.index)
                for col in ROLE_STATS:
                    col_min = candidates_df[col].min()
                    col_max = candidates_df[col].max()
                    if pd.notna(col_min) and pd.notna(col_max) and col_max > col_min:
                        scaled[col] = (candidates_df[col] - col_min) / (col_max - col_min)
                    else:
                        scaled[col] = 0.0
                candidates_df["_selection_score"] = scaled.fillna(0.0).mean(axis=1)
                best = candidates_df.loc[candidates_df["_selection_score"].idxmax()].drop("_selection_score")
            else:
                best = candidates_df.loc[candidates_df["PTS"].idxmax()]

        results.append({**base, "note": "ok", **best.to_dict()})

    # 6. Write output CSVs
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    full_df = pd.DataFrame(results)

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
    ordered = [c for c in priority if c in full_df.columns]
    _extra_targets = ("VORP", "DPM")
    remaining = [
        c for c in full_df.columns
        if c not in ordered and c not in _extra_targets and c != "PLAYER_NAME_NORM"
    ]
    out_df = full_df[ordered + remaining].copy()

    # VORP companion: base-shaped columns + VORP only (unchanged contract for downstream).
    vorp_priority = priority + ["VORP"]
    vorp_ordered = [c for c in vorp_priority if c in full_df.columns]
    vorp_remaining = [
        c for c in full_df.columns
        if c not in vorp_ordered and c not in ("DPM", "PLAYER_NAME_NORM")
    ]
    out_df_vorp = full_df[vorp_ordered + vorp_remaining].copy()

    # DARKO companion: same rows + DPM only.
    darko_priority = priority + ["DPM"]
    darko_ordered = [c for c in darko_priority if c in full_df.columns]
    darko_remaining = [
        c for c in full_df.columns
        if c not in darko_ordered and c not in ("VORP", "PLAYER_NAME_NORM")
    ]
    out_df_darko = full_df[darko_ordered + darko_remaining].copy()

    out_df.to_csv(OUTPUT_CSV, index=False)
    out_df_vorp.to_csv(OUTPUT_CSV_VORP, index=False)
    out_df_darko.to_csv(OUTPUT_CSV_DARKO, index=False)

    # Summary
    ok = (out_df["note"] == "ok").sum()
    not_found = (out_df["note"] == "not_found_in_api").sum()
    no_data = len(out_df) - ok - not_found

    print(f"{'='*60}")
    print(f"Output → {OUTPUT_CSV}")
    print(f"VORP Output → {OUTPUT_CSV_VORP}")
    print(f"DARKO Output → {OUTPUT_CSV_DARKO}")
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
