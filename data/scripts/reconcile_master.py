"""
reconcile_master.py

Reconcile nba_stats_best_season.csv and ncaa_stats_master.csv against
data/scouting/players_list.txt, producing two matched output files:

  data/nba/nba_master.csv   — one row per player, NBA stats
  data/ncaa/ncaa_master.csv — one row per player, NCAA stats (last season before draft)

Only players present in players_list.txt AND with data in both sources are
included. The player sets in both output files are identical.

NCAA season selection: for a player drafted in year Y, we use the NCAA season
"{Y-1}-{Y}" (i.e. their final college season immediately before the draft).
NBA selection: rows where `note` starts with "ok" or "recovered:" (players
with actual game stats; excludes not_found, no_data, and recovered_no_data).
"""

import re
from pathlib import Path

import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent
PLAYERS_LIST = ROOT / "data" / "scouting" / "players_list.txt"
NBA_INPUT = ROOT / "data" / "nba" / "nba_stats_best_season.csv"
NCAA_INPUT = ROOT / "data" / "ncaa" / "ncaa_stats_master.csv"
NBA_OUTPUT = ROOT / "data" / "nba" / "nba_master.csv"
NCAA_OUTPUT = ROOT / "data" / "ncaa" / "ncaa_master.csv"


# ── Parse players_list.txt ─────────────────────────────────────────────────────

def parse_players_list(path: Path) -> list[dict]:
    """Return list of {player_name, draft_year, draft_pick, position}."""
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
    # 1. Parse player list
    player_list = parse_players_list(PLAYERS_LIST)
    print(f"Players in list: {len(player_list)}")

    meta_df = pd.DataFrame(player_list).rename(columns={"player_name": "Name"})
    # Key for deduplication-safe matching: (name, draft_year)
    meta_df["_key"] = meta_df["Name"] + "|" + meta_df["draft_year"].astype(str)

    # 2. Load source files
    nba = pd.read_csv(NBA_INPUT)
    ncaa = pd.read_csv(NCAA_INPUT)

    # 3. Filter NBA ─────────────────────────────────────────────────────────────
    # Keep rows in the player list with actual game stats.
    list_names = set(meta_df["Name"])
    has_stats = nba["note"].str.startswith("ok") | nba["note"].str.startswith("recovered:")
    nba_filtered = nba[nba["player_name"].isin(list_names) & has_stats].copy()
    nba_filtered["draft_year"] = nba_filtered["draft_year"].astype(int)
    nba_filtered["_key"] = nba_filtered["player_name"] + "|" + nba_filtered["draft_year"].astype(str)
    print(f"Players in list with NBA stats: {len(nba_filtered)}")

    # 4. Filter NCAA ────────────────────────────────────────────────────────────
    # Merge player metadata into NCAA rows first, so each NCAA row is paired with
    # the specific (name, draft_year) entry it belongs to. This correctly handles
    # players sharing a name across different draft classes (e.g. two Tony Mitchells).
    meta_for_merge = meta_df[["Name", "draft_year", "draft_pick", "position"]].copy()
    meta_for_merge["expected_season"] = (
        (meta_for_merge["draft_year"] - 1).astype(str)
        + "-"
        + meta_for_merge["draft_year"].astype(str)
    )

    ncaa_merged = ncaa.merge(meta_for_merge, on="Name", how="inner")
    ncaa_filtered = ncaa_merged[
        ncaa_merged["Season"] == ncaa_merged["expected_season"]
    ].drop(columns=["expected_season"]).copy()

    # Guard against any remaining duplicates on (Name, draft_year)
    ncaa_filtered = ncaa_filtered.drop_duplicates(subset=["Name", "draft_year"])
    ncaa_filtered["_key"] = ncaa_filtered["Name"] + "|" + ncaa_filtered["draft_year"].astype(str)
    print(f"Players in list with NCAA last-season data: {len(ncaa_filtered)}")

    # 5. Intersection on (name, draft_year) ────────────────────────────────────
    common_keys = set(nba_filtered["_key"]) & set(ncaa_filtered["_key"])
    print(f"Players with both NBA and NCAA data: {len(common_keys)}")

    # 6. Build output DataFrames ────────────────────────────────────────────────
    nba_out = nba_filtered[nba_filtered["_key"].isin(common_keys)].drop(columns=["_key"]).copy()
    ncaa_out = ncaa_filtered[ncaa_filtered["_key"].isin(common_keys)].drop(columns=["_key"]).copy()

    # Reorder NCAA: identity columns first
    id_cols = ["Name", "draft_year", "draft_pick", "position", "Season", "Team", "Cl", "Pos"]
    id_cols = [c for c in id_cols if c in ncaa_out.columns]
    stat_cols = [c for c in ncaa_out.columns if c not in id_cols]
    ncaa_out = ncaa_out[id_cols + stat_cols]

    # Sort both by draft_year then draft_pick for consistent ordering
    nba_out = nba_out.sort_values(["draft_year", "draft_pick"]).reset_index(drop=True)
    ncaa_out = ncaa_out.sort_values(["draft_year", "draft_pick"]).reset_index(drop=True)

    # 7. Write output ───────────────────────────────────────────────────────────
    NBA_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    NCAA_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    nba_out.to_csv(NBA_OUTPUT, index=False)
    ncaa_out.to_csv(NCAA_OUTPUT, index=False)

    print(f"\nOutput:")
    print(f"  {NBA_OUTPUT}  ({len(nba_out)} rows, {len(nba_out.columns)} cols)")
    print(f"  {NCAA_OUTPUT}  ({len(ncaa_out)} rows, {len(ncaa_out.columns)} cols)")

    # Sanity check: player+year keys must match exactly
    nba_keys = set(nba_out["player_name"] + "|" + nba_out["draft_year"].astype(str))
    ncaa_keys = set(ncaa_out["Name"] + "|" + ncaa_out["draft_year"].astype(str))
    assert nba_keys == ncaa_keys, "Player sets do not match between outputs!"
    print(f"\nPlayer sets match: OK")


if __name__ == "__main__":
    main()
