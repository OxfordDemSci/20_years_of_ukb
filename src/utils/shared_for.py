"""
Fields of Research (FOR, 2020 ANZSRC) — shared L2/L4 helpers
============================================================

WHY THIS EXISTS
---------------
Dimensions exports have changed the FOR column name between snapshots:

    data/df_dimensions.xlsx (retired)                  ->  `category_for_2020`
    data/showcase/showcase+/showcase_plus_all_endpoint.parquet  ->  both, identical contents

The *contents* are identical in structure — the same 2020 ANZSRC taxonomy, the same
2-digit / 4-digit codes, the same divisions ('32 Biomedical and Clinical Sciences',
'42 Health Sciences', ...). Only the column name moved.

That rename was silent and dangerous, because the old helper began with

    def add_for_columns(df, col="category_for_2020"):
        if col not in df.columns:
            return df          # <- no error, no columns, no warning

so on the newer export it quietly produced NO `for_l2` / `for_l4` at all. Everything
downstream then read empty/absent FOR levels without a single traceback.

This module centralises the column resolution and the parsing so every notebook agrees,
and so a *missing* FOR column raises instead of vanishing.

THE TAXONOMY
------------
Each entry is {'id': ..., 'name': '<code> <label>'} where the code says the level:

    2-digit  -> L2 division   '32 Biomedical and Clinical Sciences'
    4-digit  -> L4 field      '3212 Ophthalmology and Optometry'   (parent division = '32')

`category_for*` is an UNORDERED multi-label set — no relevance score, list order is not
meaningful — so there is no true "primary" field. Callers that need one must say so.

THREE BEHAVIOURS, PRESERVED
---------------------------
The notebooks derive FOR levels three different ways. They are all supported here rather
than being silently unified, because they answer different questions:

  * `add_for_columns(df)`                     flat multi-label sets: L2 from 2-digit codes
                                              only  (01_analysis_05_* notebooks)
  * `add_for_columns(df, fold_parents=True)`  same, plus each L4's parent division folded
                                              into `for_l2`, so the division set is
                                              complete  (01_analysis_02_authors)
  * `primary_for_levels(cell, lookup)`        first-listed L2/L4 as a single label, for
                                              topic naming  (01_analysis_03_network_over_time)

`for_long(df)` is the fourth shape: the L2xL4 long table, one row per (paper, division,
field), carrying codes as well as labels. The 02 content notebooks each held their own copy
of that explosion; it lives here now.

Author: Jiani Y
Date: 2026-07-13
"""

import ast
import json
import warnings
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Known FOR column names, in preference order (newest export last, but either is fine).
FOR_COLUMNS: Tuple[str, ...] = ("category_for_2020", "category_for")


# =============================================================================
# COLUMN RESOLUTION — the whole point of this module
# =============================================================================
def resolve_for_column(df: pd.DataFrame, prefer: Optional[str] = None) -> str:
    """Name of the FOR column in `df`, whatever the export called it.

    `prefer` is honoured when present (so existing calls that pass "category_for_2020"
    keep working), and otherwise falls back to whichever known column exists — with a
    warning, because the caller clearly expected a different name.

    Raises KeyError if the frame carries no FOR column at all: a missing taxonomy should
    stop the pipeline, not silently produce empty levels.
    """
    if prefer and prefer in df.columns:
        return prefer
    present = [c for c in FOR_COLUMNS if c in df.columns]
    if not present:
        raise KeyError(
            "No Fields-of-Research column found. Expected one of "
            f"{list(FOR_COLUMNS)}; frame has {list(df.columns)[:25]}"
            f"{' ...' if len(df.columns) > 25 else ''}"
        )
    if prefer:
        warnings.warn(
            f"FOR column '{prefer}' not in this frame; using '{present[0]}' instead "
            "(the Dimensions export renamed it — contents are the same taxonomy).",
            stacklevel=2,
        )
    return present[0]


def has_for_column(df: pd.DataFrame) -> bool:
    """True if the frame carries any known FOR column."""
    return any(c in df.columns for c in FOR_COLUMNS)


# =============================================================================
# PARSING
# =============================================================================
def parse_listcol(x) -> list:
    """Stringified list -> Python list; already-a-list kept; unparseable/NaN -> [].

    JSON first (showcase_plus_all_endpoint.parquet writes JSON, where `null` /
    `true` are tokens `ast.literal_eval` rejects), then Python-literal for the
    legacy xlsx/pickle exports.
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
            except (ValueError, SyntaxError):
                return []
        return v if isinstance(v, list) else []
    return []


def split_for_name(name) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """'3212 Ophthalmology and Optometry' -> ('3212', 'Ophthalmology and Optometry', 'L4').

    Returns (None, None, None) for anything that isn't a numeric-coded FOR name.
    """
    if not isinstance(name, str) or not name.strip():
        return None, None, None
    parts = name.strip().split(" ", 1)
    code = parts[0]
    label = parts[1].strip() if len(parts) > 1 else ""
    if code.isdigit() and len(code) == 2:
        return code, label, "L2"
    if code.isdigit() and len(code) == 4:
        return code, label, "L4"
    return None, None, None


def split_for_levels(cell) -> Tuple[List[str], List[str]]:
    """(l2_labels, l4_labels) — two sorted lists — from one FOR category cell."""
    l2, l4 = set(), set()
    for c in parse_listcol(cell):
        if not isinstance(c, dict):
            continue
        _, label, level = split_for_name(c.get("name"))
        if level == "L2":
            l2.add(label)
        elif level == "L4":
            l4.add(label)
    return sorted(l2), sorted(l4)


def build_for_lookup(df: pd.DataFrame, col: Optional[str] = None) -> Dict[str, str]:
    """Global 2-digit division code -> division label, harvested from the data itself."""
    col = resolve_for_column(df, col)
    lut: Dict[str, str] = {}
    for cats in df[col].apply(parse_listcol):
        for c in cats:
            if isinstance(c, dict):
                code, label, level = split_for_name(c.get("name"))
                if level == "L2" and code not in lut:
                    lut[code] = label
    return lut


# =============================================================================
# THE DERIVATIONS
# =============================================================================
def add_for_columns(df: pd.DataFrame, col: Optional[str] = None,
                    fold_parents: bool = False,
                    lookup: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Attach `for_l2`, `for_l4` (list columns) and `n_for_l2` to `df`, in place.

    fold_parents=False  L2 = the 2-digit codes actually tagged on the paper.
    fold_parents=True   L2 additionally includes the parent division of every L4 code, so
                        a paper tagged only '3212' still counts under division '32'. This
                        is what 01_analysis_02_authors does; it yields a *complete*
                        division set and therefore higher L2 coverage.

    Raises KeyError when no FOR column exists (see resolve_for_column).
    """
    col = resolve_for_column(df, col)
    div_lut = lookup if lookup is not None else (build_for_lookup(df, col) if fold_parents else {})

    def _levels(cell):
        l2, l4 = set(), set()
        for c in parse_listcol(cell):
            if not isinstance(c, dict):
                continue
            code, label, level = split_for_name(c.get("name"))
            if level == "L2":
                l2.add(label)
            elif level == "L4":
                l4.add(label)
                if fold_parents:
                    parent = div_lut.get(code[:2])
                    if parent:
                        l2.add(parent)
        return sorted(l2), sorted(l4)

    levels = df[col].apply(_levels)
    df["for_l2"] = levels.apply(lambda t: t[0])
    df["for_l4"] = levels.apply(lambda t: t[1])
    df["n_for_l2"] = df["for_l2"].apply(len)
    return df


def for_long(df: pd.DataFrame, col: Optional[str] = None, id_col: str = "id",
             carry: Tuple[str, ...] = ("year",),
             parent_from_code: bool = True,
             lookup: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """One row per (paper, L2 division, L4 field) — codes AND labels, both levels.

    `add_for_columns` returns label lists and drops the codes, which is enough for a
    coverage count but not for anything that has to name a division or walk the hierarchy.
    This is the L2xL4 explosion the 02 content notebooks each carried inline; it lives here
    so the pairing rule is written once.

    parent_from_code=True   an L4's division is `code[:2]`, so every L4 gets a parent
                            whether or not the 2-digit code was tagged on the paper.
    parent_from_code=False  an L4 pairs only with 2-digit codes actually present on the
                            paper — the `l4c.startswith(l2c)` rule 02_content_2/3 used, and
                            the reason a paper tagged '3212' but not '32' would vanish.

    The two agree exactly on the current corpus (0 of 37,719 pairs differ), so the switch is
    there to keep that a checked claim rather than an assumption.

    Papers carrying no L4 field produce no rows: this is a table of pairs, not of papers.
    """
    col = resolve_for_column(df, col)
    div_lut = lookup if lookup is not None else build_for_lookup(df, col)
    carry = tuple(carry)
    missing = [c for c in (id_col, *carry) if c not in df.columns]
    if missing:
        raise KeyError(f"for_long: frame has no column(s) {missing}")

    rows = []
    for rec in df[[id_col, *carry, col]].itertuples(index=False, name=None):
        ident, *carried, cell = rec
        l2, l4 = {}, {}
        for c in parse_listcol(cell):
            if not isinstance(c, dict):
                continue
            code, label, level = split_for_name(c.get("name"))
            if level == "L2":
                l2[code] = label
            elif level == "L4":
                l4[code] = label
        for l4_code, l4_label in l4.items():
            parents = ({l4_code[:2]: div_lut.get(l4_code[:2])} if parent_from_code
                       else {k: v for k, v in l2.items() if l4_code.startswith(k)})
            for l2_code, l2_label in parents.items():
                rows.append((ident, *carried, l2_code, l2_label, l4_code, l4_label))

    return pd.DataFrame(
        rows, columns=[id_col, *carry, "l2_code", "l2_label", "l4_code", "l4_label"]
    )


def primary_for_levels(cell, lookup: Optional[Dict[str, str]] = None) -> Tuple[str, str]:
    """First-listed (L2, L4) labels for topic naming; 'Unclassified' when absent.

    'Primary' here means *first listed*, which is a labelling convenience, NOT a claim
    that Dimensions ranks the categories — it does not. An L4-only paper resolves its L2
    from the L4's parent division via `lookup`.
    """
    lookup = lookup or {}
    primary_l2 = primary_l4 = None
    for c in parse_listcol(cell):
        if not isinstance(c, dict):
            continue
        code, label, level = split_for_name(c.get("name"))
        if level == "L2":
            if primary_l2 is None:
                primary_l2 = label
        elif level == "L4":
            if primary_l4 is None:
                primary_l4 = label
            if primary_l2 is None:
                primary_l2 = lookup.get(code[:2])
    return primary_l2 or "Unclassified", primary_l4 or "Unclassified"


# =============================================================================
# CONVENIENCE
# =============================================================================
def load_papers(path=None, fold_parents: bool = False,
                verbose: bool = True) -> pd.DataFrame:
    """Load the paper corpus and attach FOR levels, whatever the FOR column is called.

    Defaults to the showcase-plus parquet; pass `path` to a .pkl to read a legacy export.
    """
    from utils import shared_paths as _P
    from utils.shared_showcase import load_showcase
    path = _P.SHOWCASE_PLUS if path is None else path
    df = load_showcase(path=path) if str(path).endswith(".parquet") else pd.read_pickle(path)
    add_for_columns(df, fold_parents=fold_parents)
    if verbose:
        print(f"{path}: {len(df):,} papers | FOR column: '{resolve_for_column(df)}'")
        print(for_coverage(df).to_string(index=False))
    return df


def for_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Small coverage report — how many papers actually carry L2 / L4 levels."""
    n = len(df)
    rows = []
    for lvl in ("for_l2", "for_l4"):
        if lvl not in df.columns:
            continue
        nonempty = int(df[lvl].apply(len).gt(0).sum())
        distinct = len({v for L in df[lvl] for v in L})
        rows.append({"level": lvl, "papers_with_>=1": nonempty,
                     "%": round(100 * nonempty / n, 1) if n else 0.0,
                     "distinct_labels": distinct})
    return pd.DataFrame(rows)
