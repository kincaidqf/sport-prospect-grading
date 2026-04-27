"""
reconcile_from_ncaa_master.py

Build matching nba_master.csv and ncaa_master.csv from the current
ncaa_master.csv (which already has the correct season per player).

Unlike reconcile_master.py (which rebuilds ncaa_master from the raw
ncaa_stats_master.csv), this script treats ncaa_master.csv as the
authoritative NCAA source and simply intersects it with valid NBA stats.

Run from project root:
    uv run python data/scripts/reconcile_from_ncaa_master.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent.parent
NCAA_PATH  = ROOT / "data" / "ncaa" / "ncaa_master.csv"
NBA_INPUT  = ROOT / "data" / "nba" / "nba_stats_best_season.csv"
NBA_OUTPUT = ROOT / "data" / "nba" / "nba_master.csv"

def main():
    ncaa = pd.read_csv(NCAA_PATH)
    nba  = pd.read_csv(NBA_INPUT)

    print(f"ncaa_master:          {len(ncaa)} players")
    print(f"nba_stats_best_season:{len(nba)} players")

    # Keep only NBA rows with actual game stats
    has_stats = nba["note"].str.startswith("ok") | nba["note"].str.startswith("recovered:")
    nba_valid = nba[has_stats].copy()
    nba_valid["draft_year"] = nba_valid["draft_year"].fillna(0).astype(int)
    nba_valid["_key"] = nba_valid["player_name"] + "|" + nba_valid["draft_year"].astype(str)
    print(f"NBA players with stats: {len(nba_valid)}")

    # Build key on ncaa side
    ncaa["_key"] = ncaa["Name"] + "|" + ncaa["draft_year"].astype(str)

    # Intersection
    common = set(ncaa["_key"]) & set(nba_valid["_key"])
    print(f"Intersection (both sides): {len(common)}")

    ncaa_out = ncaa[ncaa["_key"].isin(common)].drop(columns=["_key"]).copy()
    nba_out  = nba_valid[nba_valid["_key"].isin(common)].drop(columns=["_key"]).copy()

    # Consistent sort
    ncaa_out = ncaa_out.sort_values(["draft_year", "draft_pick"]).reset_index(drop=True)
    nba_out  = nba_out.sort_values(["draft_year", "draft_pick"]).reset_index(drop=True)

    # Sanity check
    ncaa_keys = set(ncaa_out["Name"] + "|" + ncaa_out["draft_year"].astype(str))
    nba_keys  = set(nba_out["player_name"] + "|" + nba_out["draft_year"].astype(str))
    assert ncaa_keys == nba_keys, "Player sets diverged after reconcile!"

    # Write
    ncaa_out.to_csv(NCAA_PATH, index=False)
    nba_out.to_csv(NBA_OUTPUT, index=False)

    print(f"\nncaa_master.csv → {len(ncaa_out)} rows  ({NCAA_PATH})")
    print(f"nba_master.csv  → {len(nba_out)} rows  ({NBA_OUTPUT})")
    print("Player sets match: OK")

    # Report dropped players
    dropped = set(ncaa["Name"] + "|" + ncaa["draft_year"].astype(str)) - common
    if dropped:
        print(f"\nDropped from ncaa_master (no NBA stats, {len(dropped)}):")
        dropped_df = ncaa[~ncaa["_key"].isin(common)][["Name","draft_year","draft_pick"]]
        print(dropped_df.to_string(index=False))

if __name__ == "__main__":
    main()
