"""Shared fixtures and paths for the test suite."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "src" / "config" / "config.yaml"
NCAA_PATH = PROJECT_ROOT / "data" / "ncaa" / "ncaa_master.csv"
NBA_PATH = PROJECT_ROOT / "data" / "nba" / "nba_master.csv"


@pytest.fixture(scope="session")
def cfg() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def ncaa_df() -> pd.DataFrame:
    return pd.read_csv(NCAA_PATH)


@pytest.fixture(scope="session")
def nba_df() -> pd.DataFrame:
    return pd.read_csv(NBA_PATH)


@pytest.fixture(scope="session")
def merged_df():
    """Load the merged NCAA+NBA dataframe via the shared loader."""
    from src.data.loader import load_data
    return load_data()
