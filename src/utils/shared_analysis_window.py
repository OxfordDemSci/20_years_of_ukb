"""Fixed 2013–2025 analysis window, independent of the data snapshot date.

Raw source files and citation/attention snapshot totals are not rewritten. Apply this
filter before deriving metrics, fitting models, or selecting displayed records.
"""

from __future__ import annotations

import pandas as pd


ANALYSIS_START_DATE = pd.Timestamp("2013-01-01")
ANALYSIS_START_YEAR = ANALYSIS_START_DATE.year
ANALYSIS_END_DATE = pd.Timestamp("2025-12-31")
ANALYSIS_END_YEAR = ANALYSIS_END_DATE.year
_START_INCLUSIVE = ANALYSIS_START_DATE.tz_localize("UTC")
_END_EXCLUSIVE = (ANALYSIS_END_DATE + pd.Timedelta(days=1)).tz_localize("UTC")


def filter_analysis_window(
    frame: pd.DataFrame,
    *,
    year_col: str | None = "year",
    date_col: str | None = None,
) -> pd.DataFrame:
    """Copy rows dated from 1 January 2013 through 31 December 2025, inclusive.

    When available, both year and date must satisfy the window. An out-of-window year
    cannot be rescued by an in-window date (or vice versa). Missing/unparseable dates
    may fall back to a valid year; rows with neither are excluded. By default the
    date column is detected from ``date`` or ``publication_date``. Specify event
    columns explicitly for trials, patents, and other dated outcomes.
    """
    if date_col is None:
        date_col = next((name for name in ("date", "publication_date") if name in frame), None)
    has_year = year_col is not None and year_col in frame
    has_date = date_col is not None and date_col in frame
    if not has_year and not has_date:
        raise ValueError(
            "Cannot enforce the analysis window (2013-01-01 to 2025-12-31): "
            "no publication/event year or date column is available."
        )

    known = pd.Series(False, index=frame.index)
    within = pd.Series(True, index=frame.index)
    if has_year:
        years = pd.to_numeric(frame[year_col], errors="coerce")
        valid = years.notna() & years.between(1, 9999) & years.mod(1).eq(0)
        known |= valid
        within &= ~valid | years.between(ANALYSIS_START_YEAR, ANALYSIS_END_YEAR)
    if has_date:
        dates = pd.to_datetime(frame[date_col].astype("string"), errors="coerce", format="mixed", utc=True)
        valid = dates.notna()
        known |= valid
        within &= ~valid | (dates.ge(_START_INCLUSIVE) & dates.lt(_END_EXCLUSIVE))

    result = frame.loc[(known & within).fillna(False)].copy()
    result.attrs["analysis_start_date"] = ANALYSIS_START_DATE.date().isoformat()
    result.attrs["analysis_end_date"] = ANALYSIS_END_DATE.date().isoformat()
    return result
