"""Loader for the UK Biobank publication corpus, `showcase_plus_all_endpoint.parquet`.

This replaces the old `data/df_dimensions.xlsx`. Two differences matter to every
downstream notebook, and both are handled here rather than in each notebook:

1. NESTED COLUMNS ARE **JSON**, NOT PYTHON REPRs.
   The xlsx export wrote Python reprs, so the notebooks parsed them with
   `ast.literal_eval`. The parquet export writes JSON, which uses `null` / `true` /
   `false` — tokens `ast.literal_eval` rejects. Because the notebooks' parsers
   swallow exceptions and return `[]`, an unported notebook does not crash: it
   silently drops most of its data (e.g. every author whose affiliation has a
   `"state": null`). `parse_listcol` / `parse_dictcol` below try JSON first and fall
   back to `ast.literal_eval`, so they read both the parquet and any legacy pickle.

2. `research_org_country_names` NO LONGER EXISTS.
   It is now `research_org_countries`, a list of `{"id": "GB", "name": "United
   Kingdom"}` dicts rather than a list of bare name strings. `load_showcase()`
   rebuilds the old column from the new one so the paper-level geography code keeps
   working unchanged.

Typical use:

    from utils.shared_showcase import load_showcase
    df = load_showcase(columns=["id", "year", "times_cited", "authors"], parse=["authors"])
"""

from __future__ import annotations

import ast
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd

from utils import shared_paths as P

# Columns whose values are JSON blobs in the parquet. Anything listed in `parse=`
# must appear here (or be a real column of the file) or the load raises.
NESTED_COLUMNS = {
    "authors", "researchers", "research_orgs", "research_org_countries",
    "research_org_names", "research_org_cities", "research_org_types",
    "research_org_state_codes", "funders", "funder_countries", "concepts",
    "concepts_scores", "concepts_relevant", "mesh_terms", "clinical_trial_ids",
    "reference_ids", "referenced_pubs", "journal_lists", "open_access",
    "supporting_grant_ids", "editors", "journal",
} | {c for c in (
    "category_for", "category_for_2020", "category_rcdc", "category_hrcs_hc",
    "category_hrcs_rac", "category_bra", "category_hra", "category_icrp_cso",
    "category_icrp_ct", "category_sdg", "category_uoa",
)}


def parse_listcol(x):
    """Stringified list -> Python list. Already-a-list is kept; NaN/garbage -> [].

    Tries JSON first (the parquet's format), then Python-literal (legacy pickles).
    Never raises.
    """
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        try:
            v = json.loads(s)
        except Exception:
            try:
                v = ast.literal_eval(s)
            except Exception:
                return []
        return v if isinstance(v, list) else []
    return []


def parse_dictcol(x):
    """Stringified single dict -> Python dict; NaN/garbage -> {}. Never raises."""
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return {}
        try:
            v = json.loads(s)
        except Exception:
            try:
                v = ast.literal_eval(s)
            except Exception:
                return {}
        return v if isinstance(v, dict) else {}
    return {}


def item_names(cell) -> list:
    """Names from a list cell: handles list-of-dicts ({'name': ...}) and list-of-str."""
    out = []
    for el in parse_listcol(cell):
        if isinstance(el, dict):
            nm = el.get("name")
            if nm:
                out.append(nm)
        elif isinstance(el, str) and el.strip():
            out.append(el.strip())
    return out


def count_items(series, dedup_within_row=True, top=None):
    """Frequency of items across a list-valued column, counting each item at most
    once per row (dedup_within_row=True) so the number is 'documents containing X'."""
    c = Counter()
    for cell in series:
        items = item_names(cell)
        if dedup_within_row:
            items = set(items)
        c.update(items)
    s = pd.Series(c, dtype="int64").sort_values(ascending=False)
    return s.head(top) if top else s


def count_items_frac(series):
    """FRACTIONAL frequency: each row spreads a total weight of 1 equally across its unique
    items, so a trial listing many diseases does not dominate a trial listing one. The
    column sum then equals the number of rows carrying >=1 item (a paper/trial count)."""
    w = defaultdict(float)
    for cell in series:
        items = set(item_names(cell))
        if not items:
            continue
        f = 1.0 / len(items)
        for it in items:
            w[it] += f
    return pd.Series(w, dtype="float64").sort_values(ascending=False)


def load_showcase(columns: Optional[Sequence[str]] = None,
                  parse: Iterable[str] = (),
                  path: Path = P.SHOWCASE_PLUS,
                  cache: Optional[Path] = None) -> pd.DataFrame:
    """Read the showcase-plus parquet.

    columns : subset to read (None = all). `research_org_country_names` may be
              requested; it is synthesised from `research_org_countries`.
    parse   : nested columns to convert from JSON strings to Python objects.
    cache   : optional .pkl holding the already-parsed frame. It is reused only when
              newer than the parquet, so editing the source invalidates it.
    """
    path = Path(path)
    parse = list(parse)

    if cache is not None:
        cache = Path(cache)
        if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
            return pd.read_pickle(cache)

    # The legacy name is an alias, not a column of the file.
    want_country_names = columns is not None and "research_org_country_names" in columns
    read_cols = None
    if columns is not None:
        read_cols = [c for c in columns if c != "research_org_country_names"]
        if want_country_names and "research_org_countries" not in read_cols:
            read_cols.append("research_org_countries")

    df = pd.read_parquet(path, columns=read_cols)

    if (want_country_names or columns is None) and "research_org_countries" in df.columns:
        df["research_org_country_names"] = df["research_org_countries"].apply(item_names)

    for c in parse:
        if c not in df.columns:
            raise KeyError(f"cannot parse {c!r}: not a column of {path.name}. "
                           f"Available: {sorted(df.columns)[:12]}...")
        df[c] = df[c].apply(parse_listcol)

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(cache)
    return df
