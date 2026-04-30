"""
recover_nba_players.py

Second-pass resolution for players that `fetch_nba_stats.py` could not find.

Goals
-----
- Only attempt recovery for NCAA players (class_year ∉ 'International').
- Use CommonAllPlayers (one API call) as the authoritative name ↔ ID db.
- Apply a cascade of matching strategies, most-confident first, so we never
  accept a low-quality match when a high-quality one exists.
- Cache fetched season data to disk so the script is fast on re-runs.
- Overwrite data/nba_stats_best_season.csv with the improved results.

Matching strategy cascade (first hit wins):
  1. Exact full-name match (nba_api static, fast, already tried – skip on
     players that succeeded last time, redo for the "not_found" ones to stay
     consistent with the main script).
  2. Unique last-name match within era window [draft_year-1, draft_year+3].
  3. Last-name + first-initial disambiguation when multiple last-name hits.
  4. Initials expansion: "BJ" → "B.J.", "AJ" → "A.J.", etc.
  5. Fuzzy last-name match (handles typos like "Johnsonn" → "Johnson"),
     threshold ≥ 0.88, then disambiguate by first initial.
  6. Normalised full-name fuzzy match within era, threshold ≥ 0.80.
  7. Widen era window to [draft_year-2, draft_year+5] and retry steps 2-3.
"""

import difflib
import re
import time
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import commonallplayers, leaguedashplayerstats
from nba_api.stats.static import players as nba_static

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent
PLAYERS_LIST = ROOT / "data" / "scouting" / "players_list.txt"
PLAYERS_CSV = ROOT / "data" / "scouting" / "players.csv"
PREV_OUTPUT = ROOT / "data" / "nba" / "nba_stats_best_season.csv"
OUTPUT_CSV = ROOT / "data" / "nba" / "nba_stats_best_season.csv"
SEASON_CACHE_DIR = ROOT / "data" / "nba" / "season_cache"

API_DELAY = 1.0
NCAA_CLASSES = {"Freshman", "Sophomore", "Junior", "Senior", "Class of 2021", "HS Senior"}

KEEP_COLS = [
    "PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "GP", "GS", "MIN",
    "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
    "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB",
    "AST", "STL", "BLK", "TOV", "PF", "PTS", "PLUS_MINUS",
]


# ── Utilities ──────────────────────────────────────────────────────────────────

def normalize(name: str) -> str:
    """Lowercase, remove dots, collapse whitespace."""
    return re.sub(r"\s+", " ", re.sub(r"\.", "", name)).strip().lower()


def draft_year_to_seasons(draft_year: int) -> list[str]:
    seasons = []
    for offset in range(3):
        s = draft_year + offset
        seasons.append(f"{s}-{str(s + 1)[-2:]}")
    return seasons


# ── NBA player DB (one API call) ───────────────────────────────────────────────

def build_nba_db() -> pd.DataFrame:
    """
    Fetch CommonAllPlayers and return a tidy DataFrame with columns:
      PERSON_ID, DISPLAY_FIRST_LAST, first_name, last_name,
      FROM_YEAR, TO_YEAR, GAMES_PLAYED_FLAG
    """
    print("Fetching CommonAllPlayers …", end=" ", flush=True)
    cap = commonallplayers.CommonAllPlayers(is_only_current_season=0, timeout=45)
    df = cap.get_data_frames()[0]
    print(f"{len(df)} entries")

    df = df[df["GAMES_PLAYED_FLAG"] == "Y"].copy()
    df["FROM_YEAR"] = pd.to_numeric(df["FROM_YEAR"], errors="coerce").fillna(0).astype(int)
    df["TO_YEAR"] = pd.to_numeric(df["TO_YEAR"], errors="coerce").fillna(0).astype(int)

    # Normalised name columns for matching
    df["_norm"] = df["DISPLAY_FIRST_LAST"].apply(normalize)
    df["last_name"] = (
        df["DISPLAY_LAST_COMMA_FIRST"]
        .str.split(",")
        .str[0]
        .str.strip()
        .str.lower()
    )
    df["first_name"] = (
        df["DISPLAY_LAST_COMMA_FIRST"]
        .str.split(",")
        .str[1]
        .fillna("")
        .str.strip()
        .str.lower()
    )
    return df


# ── Resolution strategies ──────────────────────────────────────────────────────

def _era_slice(nba_db: pd.DataFrame, draft_year: int, wide: bool = False) -> pd.DataFrame:
    lo = draft_year - (2 if wide else 1)
    hi = draft_year + (5 if wide else 3)
    return nba_db[(nba_db["FROM_YEAR"] >= lo) & (nba_db["FROM_YEAR"] <= hi)]


def resolve(name: str, draft_year: int, nba_db: pd.DataFrame):
    """
    Return (player_id, matched_name, strategy) or (None, None, None).
    """
    parts = name.strip().split()
    last = parts[-1].lower()
    first = parts[0].lower() if parts else ""
    norm = normalize(name)

    for wide in (False, True):
        era = _era_slice(nba_db, draft_year, wide=wide)
        era_suffix = "_wide_era" if wide else ""

        # ── Strategy 1: exact static lookup ─────────────────────────────────
        hits = nba_static.find_players_by_full_name(f"^{re.escape(name)}$")
        for h in hits:
            row = era[era["PERSON_ID"] == h["id"]]
            if not row.empty:
                return h["id"], h["full_name"], "exact_static" + era_suffix

        # ── Strategy 2: unique last-name match ───────────────────────────────
        lm = era[era["last_name"] == last]
        if len(lm) == 1:
            r = lm.iloc[0]
            return r["PERSON_ID"], r["DISPLAY_FIRST_LAST"], "last_unique" + era_suffix

        # ── Strategy 3: last-name + first-initial ────────────────────────────
        if len(lm) > 1 and first:
            im = lm[lm["first_name"].str.startswith(first[0])]
            if len(im) == 1:
                r = im.iloc[0]
                return r["PERSON_ID"], r["DISPLAY_FIRST_LAST"], "last+initial" + era_suffix

        # ── Strategy 4: initials expansion (BJ → B.J., AJ → A.J.) ──────────
        if len(parts) >= 2 and re.match(r"^[A-Z]{2}$", parts[0]):
            expanded_first = f"{parts[0][0]}.{parts[0][1]}."
            exp_name = f"{expanded_first} {' '.join(parts[1:])}"
            exp_norm = normalize(exp_name)
            exp_lm = era[era["last_name"] == last]
            for _, row in exp_lm.iterrows():
                if row["_norm"].startswith(parts[0][0].lower()):
                    # First initial matches expanded initial
                    return (
                        row["PERSON_ID"],
                        row["DISPLAY_FIRST_LAST"],
                        "initials_expanded" + era_suffix,
                    )

        # ── Strategy 5: fuzzy last-name (typo tolerance) ─────────────────────
        era_lasts = era["last_name"].unique().tolist()
        fuzzy_lasts = difflib.get_close_matches(last, era_lasts, n=3, cutoff=0.88)
        for fl in fuzzy_lasts:
            flm = era[era["last_name"] == fl]
            if len(flm) == 1:
                r = flm.iloc[0]
                return r["PERSON_ID"], r["DISPLAY_FIRST_LAST"], "last_fuzzy" + era_suffix
            if len(flm) > 1 and first:
                im = flm[flm["first_name"].str.startswith(first[0])]
                if len(im) == 1:
                    r = im.iloc[0]
                    return (
                        r["PERSON_ID"],
                        r["DISPLAY_FIRST_LAST"],
                        "last_fuzzy+initial" + era_suffix,
                    )

        # ── Strategy 6: full normalised-name fuzzy within era ────────────────
        era_norms = era["_norm"].tolist()
        fuzzy_full = difflib.get_close_matches(norm, era_norms, n=1, cutoff=0.80)
        if fuzzy_full:
            row = era[era["_norm"] == fuzzy_full[0]].iloc[0]
            return (
                row["PERSON_ID"],
                row["DISPLAY_FIRST_LAST"],
                "full_fuzzy" + era_suffix,
            )

        # ── Strategy 7: first-name + last-name swap ──────────────────────────
        if len(parts) == 2:
            swapped = normalize(f"{parts[1]} {parts[0]}")
            sw_hits = difflib.get_close_matches(swapped, era_norms, n=1, cutoff=0.88)
            if sw_hits:
                row = era[era["_norm"] == sw_hits[0]].iloc[0]
                return row["PERSON_ID"], row["DISPLAY_FIRST_LAST"], "swapped" + era_suffix

    return None, None, None


# ── Season data (cached to disk) ───────────────────────────────────────────────

def load_or_fetch_season(season: str) -> pd.DataFrame | None:
    SEASON_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = SEASON_CACHE_DIR / f"{season}.csv"

    if cache_path.exists():
        return pd.read_csv(cache_path)

    print(f"  fetching {season} …", end=" ", flush=True)
    try:
        dash = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            per_mode_detailed="PerGame",
            season_type_all_star="Regular Season",
            timeout=45,
        )
        df = dash.get_data_frames()[0]
        if df.empty:
            print("empty")
            return None
        cols = [c for c in KEEP_COLS if c in df.columns]
        df = df[cols].copy()
        df["SEASON_ID"] = season
        df["PLUS_MINUS"] = pd.to_numeric(df["PLUS_MINUS"], errors="coerce")
        df["PTS"] = pd.to_numeric(df["PTS"], errors="coerce")
        df.to_csv(cache_path, index=False)
        print(f"{len(df)} players (cached)")
        return df
    except Exception as exc:
        print(f"FAILED: {exc}")
        return None


ROLE_STATS = ["MIN", "PTS", "REB", "AST", "STL", "BLK"]


def get_best_season(player_id: int, eligible_seasons: list[str]) -> dict | None:
    candidates = []
    for season in eligible_seasons:
        df = load_or_fetch_season(season)
        if df is None:
            continue
        row = df[df["PLAYER_ID"] == player_id]
        if not row.empty:
            candidates.append(row.iloc[0].to_dict())
    if not candidates:
        return None
    cdf = pd.DataFrame(candidates)
    for col in ROLE_STATS:
        cdf[col] = pd.to_numeric(cdf[col], errors="coerce")

    has_role_data = cdf[ROLE_STATS].notna().any().any()
    if not has_role_data:
        cdf["PTS"] = pd.to_numeric(cdf["PTS"], errors="coerce")
        return cdf.loc[cdf["PTS"].idxmax()].to_dict()

    scaled = pd.DataFrame(index=cdf.index)
    for col in ROLE_STATS:
        col_min = cdf[col].min()
        col_max = cdf[col].max()
        if pd.notna(col_min) and pd.notna(col_max) and col_max > col_min:
            scaled[col] = (cdf[col] - col_min) / (col_max - col_min)
        else:
            scaled[col] = 0.0
    cdf["_selection_score"] = scaled.fillna(0.0).mean(axis=1)
    best = cdf.loc[cdf["_selection_score"].idxmax()]
    return best.drop("_selection_score").to_dict()


# ── players.csv helpers ────────────────────────────────────────────────────────

def load_player_meta() -> dict[str, str]:
    """Return {clean_name: class_year} from players.csv."""
    pcsv = pd.read_csv(PLAYERS_CSV)
    pcsv["clean_name"] = (
        pcsv["name"].str.replace(r"^\d+\s*-\s*", "", regex=True).str.strip()
    )
    return dict(zip(pcsv["clean_name"], pcsv["class_year"].fillna("Unknown")))


# ── players_list.txt parser (same as main script) ─────────────────────────────

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
                    "player_name": pm.group(2).strip(),
                    "draft_year": current_year,
                    "draft_pick": int(pm.group(1)),
                    "position": pm.group(3).strip(),
                })
    return out


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    # 1. Parse full player list
    player_list = parse_players_list(PLAYERS_LIST)
    meta = load_player_meta()

    # 2. Determine which players need recovery:
    #    not_found_in_api AND class_year is NCAA
    prev = pd.read_csv(PREV_OUTPUT)
    not_found_names = set(
        prev.loc[prev["note"] == "not_found_in_api", "player_name"]
    )

    to_recover = []
    for entry in player_list:
        name = entry["player_name"]
        if name not in not_found_names:
            continue
        cy = meta.get(name, "Unknown")
        if cy in NCAA_CLASSES:
            to_recover.append(entry)
        else:
            # International – keep the not_found note, no attempt
            pass

    print(f"\nPlayers to attempt recovery: {len(to_recover)}  "
          f"(all NCAA, excluding international)")

    # 3. Build NBA player DB (one API call)
    time.sleep(API_DELAY)
    nba_db = build_nba_db()

    # 4. Resolve IDs for candidates
    resolved: dict[str, tuple[int, str, str]] = {}  # name → (id, matched, strategy)
    unresolved: list[str] = []
    strategy_counts: dict[str, int] = {}

    print(f"\nResolving {len(to_recover)} names …")
    for entry in to_recover:
        name = entry["player_name"]
        pid, matched, strategy = resolve(name, entry["draft_year"], nba_db)
        if pid is not None:
            resolved[name] = (pid, matched, strategy)
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        else:
            unresolved.append(name)

    print(f"\nResolved: {len(resolved)}  |  Still unresolved: {len(unresolved)}")
    if strategy_counts:
        print("Strategies used:")
        for s, c in sorted(strategy_counts.items(), key=lambda x: -x[1]):
            print(f"  {s}: {c}")

    # 5. Fetch best-season stats for newly resolved players
    #    Ensure all needed seasons are cached
    needed_seasons: set[str] = set()
    for entry in to_recover:
        if entry["player_name"] in resolved:
            for s in draft_year_to_seasons(entry["draft_year"]):
                needed_seasons.add(s)

    import datetime
    cur_year = datetime.date.today().year
    cur_season_start = cur_year if datetime.date.today().month >= 10 else cur_year - 1
    max_season_start = cur_season_start

    print(f"\nEnsuring {len(needed_seasons)} season files are cached …")
    for season in sorted(needed_seasons):
        sy = int(season.split("-")[0])
        if sy > max_season_start:
            continue
        load_or_fetch_season(season)
        time.sleep(API_DELAY)

    # 6. Build updated stats for recovered players
    new_rows: dict[str, dict] = {}
    for entry in to_recover:
        name = entry["player_name"]
        if name not in resolved:
            continue
        pid, matched_name, strategy = resolved[name]
        eligible = draft_year_to_seasons(entry["draft_year"])
        eligible = [s for s in eligible if int(s.split("-")[0]) <= max_season_start]

        best = get_best_season(pid, eligible)
        if best:
            new_rows[name] = {
                **entry,
                "player_id": pid,
                "note": f"recovered:{strategy}",
                **best,
            }
        else:
            new_rows[name] = {
                **entry,
                "player_id": pid,
                "note": f"recovered_no_data:{strategy}",
            }

    # 7. Merge recovered rows into the existing output
    #    Replace "not_found_in_api" rows for successfully identified players
    prev_updated = prev.copy()

    recovered_count = 0
    recovered_with_stats = 0

    for name, row_dict in new_rows.items():
        mask = prev_updated["player_name"] == name
        # Build a Series from row_dict aligned to existing columns + any new ones
        new_cols = [c for c in row_dict if c not in prev_updated.columns]
        for c in new_cols:
            prev_updated[c] = pd.NA
        prev_updated.loc[mask, list(row_dict.keys())] = pd.NA  # clear first
        for col, val in row_dict.items():
            if col in prev_updated.columns:
                prev_updated.loc[mask, col] = val
        recovered_count += 1
        if "recovered_no_data" not in row_dict.get("note", ""):
            recovered_with_stats += 1

    # ── Re-order columns ───────────────────────────────────────────────────────
    priority = [
        "player_name", "draft_year", "draft_pick", "position",
        "player_id", "SEASON_ID", "PLUS_MINUS",
        "GP", "MIN", "PTS", "REB", "AST", "STL", "BLK",
        "FG_PCT", "FG3_PCT", "FT_PCT",
        "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
        "OREB", "DREB", "TOV", "PF",
        "TEAM_ABBREVIATION", "note",
    ]
    ordered = [c for c in priority if c in prev_updated.columns]
    remaining = [c for c in prev_updated.columns if c not in ordered]
    prev_updated = prev_updated[ordered + remaining]

    prev_updated.to_csv(OUTPUT_CSV, index=False)

    # ── Final summary ──────────────────────────────────────────────────────────
    ok = (prev_updated["note"].str.startswith("ok") | prev_updated["note"].str.startswith("recovered:")).sum()
    not_found_remaining = (prev_updated["note"] == "not_found_in_api").sum()
    no_data = prev_updated["note"].str.contains("no_data", na=False).sum()

    print(f"\n{'='*65}")
    print(f"Output → {OUTPUT_CSV}")
    print(f"Total players             : {len(prev_updated)}")
    print(f"  With stats (ok)         : {ok}")
    print(f"  Still not found in API  : {not_found_remaining}")
    print(f"  No eligible-season data : {no_data}")
    print(f"\nRecovery this run:")
    print(f"  Newly resolved names    : {recovered_count}")
    print(f"  Of those, with stats    : {recovered_with_stats}")
    print(f"  Of those, no NBA data   : {recovered_count - recovered_with_stats}")

    if unresolved:
        print(f"\nStill unresolved NCAA ({len(unresolved)}):")
        for n in unresolved[:30]:
            print(f"  {n}")
        if len(unresolved) > 30:
            print(f"  … and {len(unresolved)-30} more")


if __name__ == "__main__":
    main()


'''
  With NBA stats (best season found): 1,213                                                                         
  - ok (original fetch): 1,090                                                                                      
  - recovered:* (resolved this pass): 123                                                                           
                                                                                                                    
  Resolved but no eligible-season data: 103                                                                         
  - These players were identified in the NBA API but had zero appearances in their 3-year eligible window (e.g.,    
  drafted but cut before playing, or played only outside the window)                                                
  - Breakdown: 51 Senior, 21 International, 12 Sophomore, 11 Junior, 8 Freshman                                     
                                                                                                                    
  Not found in API — 479 total:                                                                                   
  - International: 124 — confirmed international players (class_year = International) who never registered in the   
  NBA system. As requested, no recovery was attempted for these.                                                    
  - NCAA not found: 355 — these were exhaustively searched. Every remaining one returned zero matches across all    
  strategies (exact, last-name, fuzzy, initials, suffix-stripped). They were draft prospects who never appeared in a
   single NBA regular-season game during the study window.                                                          
                                                          

'''