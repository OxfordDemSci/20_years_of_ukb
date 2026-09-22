"""Shared Showcase list summaries are deterministic across Python hash seeds."""

from pathlib import Path
import sys

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from utils.shared_showcase import count_items, count_items_frac


def test_count_items_breaks_ties_alphabetically():
    values = pd.Series([["zeta", "alpha"], ["beta", "alpha", "zeta"]])
    assert count_items(values).index.tolist() == ["alpha", "zeta", "beta"]


def test_fractional_count_items_breaks_ties_alphabetically():
    values = pd.Series([["zeta", "alpha"], ["beta", "gamma"]])
    assert count_items_frac(values).index.tolist() == [
        "alpha",
        "beta",
        "gamma",
        "zeta",
    ]
