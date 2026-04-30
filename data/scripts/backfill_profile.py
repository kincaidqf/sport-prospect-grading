"""
backfill_profile.py

Fills missing Cl, Pos, and Ht columns for 2021-2022 and 2022-2023 seasons
in ncaa_master.csv using data/scouting/players.csv.

Run from project root:
    uv run python data/scripts/backfill_profile.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
NCAA_PATH = PROJECT_ROOT / "data" / "ncaa" / "ncaa_master.csv"
SCOUTING_PATH = PROJECT_ROOT / "data" / "scouting" / "players.csv"

CLASS_MAP = {
    "Freshman":      "Fr.",
    "Sophomore":     "So.",
    "Junior":        "Jr.",
    "Senior":        "Sr.",
    "International": "---",
}

# Primary position (before any "/") drives the abbreviation
POS_MAP = {
    "Point Guard":    "G",
    "Shooting Guard": "G",
    "Small Forward":  "F",
    "Power Forward":  "F",
    "Center":         "C",
}


def abbrev_pos(pos: str) -> str:
    if not isinstance(pos, str):
        return np.nan
    primary = pos.split("/")[0].strip()
    return POS_MAP.get(primary, np.nan)


def main():
    ncaa = pd.read_csv(NCAA_PATH)
    scouting = pd.read_csv(SCOUTING_PATH)

    scouting["clean_name"] = (
        scouting["name"].str.replace(r"^\d+ - ", "", regex=True).str.strip()
    )

    lookup = (
        scouting[["clean_name", "draft_year", "height", "position", "class_year"]]
        .drop_duplicates(subset=["clean_name", "draft_year"])
        .rename(columns={"height": "s_height", "position": "s_position", "class_year": "s_class_year"})
    )

    target_mask = ncaa["Season"].isin(["2021-2022", "2022-2023"])
    targets = ncaa[target_mask].copy()

    merged = targets.merge(
        lookup,
        left_on=["Name", "draft_year"],
        right_on=["clean_name", "draft_year"],
        how="left",
    )

    matched = merged["s_height"].notna().sum()
    unmatched = merged["s_height"].isna().sum()
    print(f"Matched: {matched} / {len(merged)}")
    if unmatched:
        print("Unmatched players:")
        print(merged[merged["s_height"].isna()][["Name", "draft_year"]].to_string())

    ncaa.loc[target_mask, "Cl"] = merged["s_class_year"].map(CLASS_MAP).values
    ncaa.loc[target_mask, "Pos"] = merged["s_position"].apply(abbrev_pos).values
    ncaa.loc[target_mask, "Ht"] = merged["s_height"].values

    ncaa.to_csv(NCAA_PATH, index=False)
    print(f"\nSaved {NCAA_PATH}")

    # Verification
    updated = ncaa[target_mask][["Name", "draft_year", "Season", "Cl", "Pos", "Ht"]]
    print(f"\nSample of updated rows:\n{updated.head(10).to_string()}")
    print(f"\nNull counts in updated rows:")
    print(updated[["Cl", "Pos", "Ht"]].isna().sum().to_string())


if __name__ == "__main__":
    main()
