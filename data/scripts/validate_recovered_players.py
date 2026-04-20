#!/usr/bin/env python3
"""Interactively validate recovered player matches in nba_master.csv.

For each row with note starting 'recovered:', shows the drafted player name
vs the recovered NBA player name and asks y/n. Rejections are removed from
both nba_master.csv and ncaa_master.csv.
"""

import pandas as pd
import sys

NBA_PATH = "data/nba/nba_master.csv"
NCAA_PATH = "data/ncaa/ncaa_master.csv"


def ask(prompt: str) -> bool:
    while True:
        ans = input(prompt).strip().lower()
        if ans in ("y", "n"):
            return ans == "y"
        print("  Please enter y or n.")


def main():
    nba = pd.read_csv(NBA_PATH)
    ncaa = pd.read_csv(NCAA_PATH)

    recovered_mask = nba["note"].str.startswith("recovered:", na=False)
    recovered = nba[recovered_mask]
    total = len(recovered)

    if total == 0:
        print("No recovered players found.")
        return

    print(f"\nFound {total} recovered player(s) to validate.\n")
    print(f"{'#':<5} {'Drafted Name':<30} {'Recovered NBA Name':<30} {'Method'}")
    print("-" * 90)

    to_drop_indices = []

    for i, (idx, row) in enumerate(recovered.iterrows(), 1):
        drafted = row["player_name"]
        recovered_name = row["PLAYER_NAME"]
        method = row["note"]
        print(f"{i:<5} {drafted:<30} {recovered_name:<30} [{method}]")
        ok = ask("  Match correct? (y/n): ")
        if not ok:
            to_drop_indices.append(idx)
            print(f"  -> Marked for removal.\n")
        else:
            print(f"  -> Accepted.\n")

    if not to_drop_indices:
        print("All recoveries accepted. No changes made.")
        return

    print(f"\nRemoving {len(to_drop_indices)} rejected row(s)...")

    rejected_rows = nba.loc[to_drop_indices]

    # Remove from nba_master
    nba_cleaned = nba.drop(index=to_drop_indices)
    nba_cleaned.to_csv(NBA_PATH, index=False)
    print(f"  nba_master.csv updated ({len(nba)} -> {len(nba_cleaned)} rows)")

    # Remove matching rows from ncaa_master by player_name + draft_year + draft_pick
    ncaa_before = len(ncaa)
    for _, row in rejected_rows.iterrows():
        match = (
            (ncaa["Name"] == row["player_name"])
            & (ncaa["draft_year"] == row["draft_year"])
            & (ncaa["draft_pick"] == row["draft_pick"])
        )
        ncaa = ncaa[~match]

    ncaa.to_csv(NCAA_PATH, index=False)
    print(f"  ncaa_master.csv updated ({ncaa_before} -> {len(ncaa)} rows)")

    print("\nDone.")


if __name__ == "__main__":
    main()
