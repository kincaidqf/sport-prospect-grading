"""
Parse NCAA stats CSVs and combine into one master CSV.

Each source CSV has multiple stat sections separated by a 3-line NCAA header.
Each section has its own column layout. We merge all sections per year on
(Name, Team), keeping the first non-null value when a column appears in
multiple sections. A Season column is added from the filename.
"""

import csv
import re
from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).parent.parent.parent / "data" / "ncaa"
OUTPUT_PATH = Path(__file__).parent.parent.parent / "data" / "ncaa" / "ncaa_stats_master.csv"

# Columns that identify a player within a season — used as merge keys.
IDENTITY_COLS = ["Name", "Team", "Cl", "Ht", "Pos"]


def parse_file(path: Path) -> pd.DataFrame:
    """Parse one season's CSV into a single DataFrame (one row per player)."""
    season = path.stem  # e.g. "2020-2021"

    with open(path, newline="", encoding="latin-1") as fh:
        raw_lines = fh.readlines()

    # Split into sections. A new section starts with "NCAA Men's Basketball".
    section_starts = [
        i for i, line in enumerate(raw_lines) if line.strip() == "NCAA Men's Basketball"
    ]
    section_starts.append(len(raw_lines))  # sentinel

    sections: list[pd.DataFrame] = []

    for idx, start in enumerate(section_starts[:-1]):
        end = section_starts[idx + 1]
        section_lines = raw_lines[start:end]

        # Extract the stat-category name from the second header line.
        # e.g. "Division IPoints Per Game" -> "Points Per Game"
        category_line = section_lines[1].strip() if len(section_lines) > 1 else ""
        category = re.sub(r"^Division\s*I\s*", "", category_line).strip()

        # Find the column-header row (starts with "Rank" or "\"Rank\"").
        header_row_idx = None
        for i, line in enumerate(section_lines):
            if re.match(r'^"?Rank"?,', line.strip()):
                header_row_idx = i
                break

        if header_row_idx is None:
            continue  # no data rows in this section

        data_lines = section_lines[header_row_idx:]
        if len(data_lines) < 2:
            continue

        # Parse with csv reader so quoted fields are handled correctly.
        reader = csv.reader(data_lines)
        rows = list(reader)

        if not rows:
            continue

        columns = [c.strip() for c in rows[0]]
        data_rows = []
        for row in rows[1:]:
            # Skip blank or malformed rows.
            if not row or all(cell.strip() == "" for cell in row):
                continue
            # Pad or truncate to match column count.
            if len(row) < len(columns):
                row = row + [""] * (len(columns) - len(row))
            else:
                row = row[: len(columns)]
            data_rows.append([cell.strip() for cell in row])

        if not data_rows:
            continue

        df = pd.DataFrame(data_rows, columns=columns)

        # Drop the Rank column — not meaningful across sections.
        df = df.drop(columns=["Rank"], errors="ignore")

        # Replace empty strings with NaN.
        df = df.replace("", pd.NA)

        # Drop rows with no Name.
        df = df.dropna(subset=["Name"])

        sections.append(df)

    if not sections:
        return pd.DataFrame()

    # Merge all sections on identity columns.
    # Start with the first section and left-merge the rest,
    # keeping first non-null value for any duplicate columns.
    merged = sections[0].copy()

    for section_df in sections[1:]:
        # Determine which columns this section adds vs already exist.
        new_cols = [c for c in section_df.columns if c not in IDENTITY_COLS]
        existing_cols = [c for c in new_cols if c in merged.columns]
        truly_new_cols = [c for c in new_cols if c not in merged.columns]

        # Merge on identity columns (outer so we don't lose any player).
        right = section_df[IDENTITY_COLS + new_cols].copy()
        # Suffix existing columns to resolve conflicts.
        suffix = "_new"
        merged = merged.merge(right, on=IDENTITY_COLS, how="outer", suffixes=("", suffix))

        # For columns that appeared in both, fill NaN from the new version.
        for col in existing_cols:
            new_col = col + suffix
            if new_col in merged.columns:
                merged[col] = merged[col].combine_first(merged[new_col])
                merged = merged.drop(columns=[new_col])

    merged["Season"] = season
    return merged


def main() -> None:
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {DATA_DIR}")
        return

    all_seasons: list[pd.DataFrame] = []

    for path in csv_files:
        print(f"Parsing {path.name} ...")
        df = parse_file(path)
        if df.empty:
            print(f"  -> No data found, skipping.")
            continue
        print(f"  -> {len(df)} players, {len(df.columns)} columns")
        all_seasons.append(df)

    if not all_seasons:
        print("No data parsed.")
        return

    master = pd.concat(all_seasons, ignore_index=True, sort=False)

    # Put Season first, then identity columns, then stats.
    identity_first = ["Season"] + IDENTITY_COLS
    stat_cols = [c for c in master.columns if c not in identity_first]
    master = master[identity_first + stat_cols]

    master.to_csv(OUTPUT_PATH, index=False)
    print(f"\nMaster CSV written to {OUTPUT_PATH}")
    print(f"Total rows: {len(master)}, columns: {len(master.columns)}")
    print(f"Columns: {list(master.columns)}")


if __name__ == "__main__":
    main()
