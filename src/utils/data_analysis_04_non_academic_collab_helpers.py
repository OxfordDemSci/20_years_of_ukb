"""Helper functions for collaborator taxonomy notebook analysis.

This module centralizes data preparation, metrics, and plotting so the notebook can
focus on narrative, tables, and figures.
"""

from __future__ import annotations

import ast
import json
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler

from .shared_style import DEFAULT_DOT_MARKER_AREA, DEFAULT_MARKER_SIZE

# The canonical output first; the `output/` entries below it are pre-2026-08-26 homes,
# kept only so a stale file on someone's disk is still found rather than silently missed.
# `find_data_path` picks the NEWEST match, so a fresh run always wins over a stale one.
DEFAULT_DATA_CANDIDATES = [
    Path("data/analysis/non_academic/collaboration/non_academic_flagged_full_company.csv"),
    Path("output/non_academic_flagged_full_company.xlsx"),
    Path("output/non_academic_flagged_full_company.progress.csv"),
    Path("output/non_academic_flagged.xlsx"),
    Path("output/non_academic_flagged.progress.csv"),
    Path("output/non_academic_flagged_check500.xlsx"),
    Path("output/non_academic_flagged_check100.xlsx"),
]

NON_ACADEMIC_SECTOR_LABELS = [
    "University/HEI",
    "Hospital/Clinical",
    "Government/Public",
    "Research institute/Centre",
    "Nonprofit/Charity",
    "Company (non-UK)",
    "UK company",
    "Other/Unknown",
]

_COMPANY_SUFFIX_RE = re.compile(
    r"\b(inc|llc|ltd|plc|corp|corporation|gmbh|ag|sa|nv|oy|oyj|ab|as|bv|pte|sarl|limited)\b",
    re.IGNORECASE,
)


def _figure_dir() -> Path:
    """Where this notebook's figures go: output/figures/data_analysis/04_non_academic.

    The three plot functions below used to derive their default from `Path.cwd()` and a
    `is this "src"?` walk. `shared_paths.bootstrap()` makes cwd the repo ROOT, so that
    walk resolved to the repo root itself and every run dropped loose PDFs/PNGs at the
    top of the checkout. Anchor on the registry instead, like every other notebook.
    """
    from . import shared_paths as P

    P.FIG_NON_ACADEMIC.mkdir(parents=True, exist_ok=True)
    return P.FIG_NON_ACADEMIC


def find_data_path(
    candidates: Iterable[Path] | None = None,
    search_bases: Iterable[Path] | None = None,
) -> Path:
    """Find the newest available classifier output file from candidate paths."""
    candidate_list = list(candidates or DEFAULT_DATA_CANDIDATES)
    base_list = list(search_bases or [Path.cwd(), Path.cwd().parent])
    found: list[Path] = []

    for base in base_list:
        for rel in candidate_list:
            candidate = base / rel
            if candidate.exists():
                found.append(candidate)

    if found:
        return max(found, key=lambda p: p.stat().st_mtime)

    tried = [str(base / rel) for base in base_list for rel in candidate_list]
    raise FileNotFoundError(f"Could not find an analysis input file. Tried: {tried}")


def load_input_dataframe(data_path: Path | None = None) -> tuple[pd.DataFrame, Path]:
    """Load the raw classifier output dataframe."""
    path = data_path or find_data_path()
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported input file format: {path}")
    return df, path


def safe_parse(value: Any) -> Any:
    """Parse list/dict-like strings safely."""
    if isinstance(value, (list, dict)):
        return value
    if pd.isna(value):
        return None

    s = str(value).strip()
    if not s:
        return None
    if s.lower() in {"nan", "none", "null"}:
        return None
    if s in {"[]", "{}"}:
        return []

    if s[0] in "[{":
        try:
            return json.loads(s)
        except Exception:
            try:
                return ast.literal_eval(s)
            except Exception:
                return None

    return s


def is_empty_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if isinstance(value, str):
        s = value.strip()
        return (not s) or (s.lower() in {"nan", "none", "null", "[]", "{}"})
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) == 0
    return False


def parse_list(value: Any) -> list[Any]:
    parsed = safe_parse(value)
    if parsed is None:
        return []
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, tuple):
        return list(parsed)
    return [parsed]


def normalize_name(value: Any) -> str:
    if is_empty_value(value):
        return ""
    s = str(value).strip()
    s = re.sub(r"\s+", " ", s)
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def org_key(value: Any) -> str:
    s = normalize_name(value)
    if not s:
        return ""
    s = s.replace("’", "'").replace("‘", "'").replace("`", "'")
    return s.casefold()


def parse_col_as_list(df_in: pd.DataFrame, col_name: str) -> pd.Series:
    if col_name in df_in.columns:
        return df_in[col_name].apply(parse_list)
    return pd.Series([[] for _ in range(len(df_in))], index=df_in.index)


def extract_affiliation_names(authors_value: Any) -> list[str]:
    """Extract affiliation/org names at authorship level.

    One author contributes at most one mention per organization.
    """
    parsed = safe_parse(authors_value)
    if parsed is None:
        return []
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return []

    names: list[str] = []

    for author in parsed:
        if not isinstance(author, dict):
            continue

        per_author_orgs: dict[str, str] = {}

        affiliations = author.get("affiliations") or []
        if isinstance(affiliations, dict):
            affiliations = [affiliations]
        for aff in affiliations:
            if not isinstance(aff, dict):
                continue
            name = normalize_name(aff.get("name") or aff.get("raw_affiliation"))
            key = org_key(name)
            if key and key not in per_author_orgs:
                per_author_orgs[key] = name

        institutions = author.get("institutions") or []
        if isinstance(institutions, dict):
            institutions = [institutions]
        for inst in institutions:
            if not isinstance(inst, dict):
                continue
            name = normalize_name(inst.get("display_name") or inst.get("name"))
            key = org_key(name)
            if key and key not in per_author_orgs:
                per_author_orgs[key] = name

        names.extend(per_author_orgs.values())

    return names


def _rows_with_authorship(df: pd.DataFrame) -> pd.DataFrame:
    author_cols = [c for c in df.columns if "author" in c.lower()]
    if not author_cols:
        raise ValueError("No author columns found in the dataset.")

    mask = df[author_cols].apply(lambda row: any(not is_empty_value(v) for v in row), axis=1)
    return df[mask].copy()


def _subtract_lists(base_items: list[str], remove_items: list[str]) -> list[str]:
    remove_set = {org_key(x) for x in remove_items if org_key(x)}
    return [x for x in base_items if org_key(x) and org_key(x) not in remove_set]


def prepare_analysis_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Parse and normalize institution columns for analysis."""
    df = _rows_with_authorship(raw_df)

    df["author_institutions_list"] = parse_col_as_list(df, "author_institutions_list")
    df["academic_institutions_raw"] = parse_col_as_list(df, "academic_institutions")
    df["non_academic_institutions_raw"] = parse_col_as_list(df, "non_academic_institutions")
    df["company_institutions_raw"] = parse_col_as_list(df, "company_institutions")
    df["uk_company_institutions_raw"] = parse_col_as_list(df, "uk_company_institutions")

    df["all_institutions"] = df["author_institutions_list"].apply(
        lambda items: [normalize_name(x) for x in items if normalize_name(x)]
    )
    df["academic_institutions_norm"] = df["academic_institutions_raw"].apply(
        lambda items: [normalize_name(x) for x in items if normalize_name(x)]
    )
    df["non_academic_institutions_norm"] = df["non_academic_institutions_raw"].apply(
        lambda items: [normalize_name(x) for x in items if normalize_name(x)]
    )
    df["company_institutions_norm"] = df["company_institutions_raw"].apply(
        lambda items: [normalize_name(x) for x in items if normalize_name(x)]
    )
    df["uk_company_institutions_norm"] = df["uk_company_institutions_raw"].apply(
        lambda items: [normalize_name(x) for x in items if normalize_name(x)]
    )

    if "academic_institutions" not in df.columns or df["academic_institutions_norm"].map(len).sum() == 0:

        def derive_academic(row: pd.Series) -> list[str]:
            non_set = {org_key(x) for x in row["non_academic_institutions_norm"] if org_key(x)}
            return [x for x in row["all_institutions"] if org_key(x) and org_key(x) not in non_set]

        df["academic_institutions_norm"] = df.apply(derive_academic, axis=1)

    df["non_company_non_academic_institutions"] = df.apply(
        lambda row: _subtract_lists(
            row["non_academic_institutions_norm"],
            row["company_institutions_norm"],
        ),
        axis=1,
    )
    df["non_uk_company_institutions"] = df.apply(
        lambda row: _subtract_lists(
            row["company_institutions_norm"],
            row["uk_company_institutions_norm"],
        ),
        axis=1,
    )

    authors_col = None
    for candidate in ["authors", "authorships"]:
        if candidate in df.columns:
            authors_col = candidate
            break

    if authors_col is not None:
        df["affiliation_names"] = df[authors_col].apply(extract_affiliation_names)
    else:
        df["affiliation_names"] = [[] for _ in range(len(df))]

    # Keep readable aliases for downstream reporting.
    df["academic_institutions"] = df["academic_institutions_norm"]
    df["non_academic_institutions"] = df["non_academic_institutions_norm"]
    df["company_institutions"] = df["company_institutions_norm"]
    df["uk_company_institutions"] = df["uk_company_institutions_norm"]

    return df


def derive_flag(df: pd.DataFrame, flag_col: str, list_col: str) -> pd.Series:
    if flag_col in df.columns:
        return pd.to_numeric(df[flag_col], errors="coerce").fillna(0).astype(int)
    return (df[list_col].map(len) > 0).astype(int)


def add_flag_columns(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    df["non_academic_flag"] = derive_flag(df, "non_academic_flag", "non_academic_institutions_norm")
    df["academic_flag"] = derive_flag(df, "academic_flag", "academic_institutions_norm")
    df["company_flag"] = derive_flag(df, "company_flag", "company_institutions_norm")
    df["uk_company_flag"] = derive_flag(df, "uk_company_flag", "uk_company_institutions_norm")
    df["company_only_flag"] = derive_flag(df, "company_only_flag", "company_institutions_norm")
    df["uk_company_only_flag"] = derive_flag(df, "uk_company_only_flag", "uk_company_institutions_norm")
    df["academic_company_collab_flag"] = derive_flag(
        df,
        "academic_company_collab_flag",
        "company_institutions_norm",
    )
    df["academic_uk_company_collab_flag"] = derive_flag(
        df,
        "academic_uk_company_collab_flag",
        "uk_company_institutions_norm",
    )
    df["non_uk_company_flag"] = (df["non_uk_company_institutions"].map(len) > 0).astype(int)
    df["non_company_non_academic_flag"] = (
        df["non_company_non_academic_institutions"].map(len) > 0
    ).astype(int)
    return df


def build_row_authorship_mentions(affiliation_names: list[str], target_names: list[str]) -> list[str]:
    target_lookup: dict[str, str] = {}
    for name in target_names:
        key = org_key(name)
        if key and key not in target_lookup:
            target_lookup[key] = name

    if not target_lookup:
        return []

    matched: list[str] = []
    for aff_name in affiliation_names:
        key = org_key(aff_name)
        if key in target_lookup:
            matched.append(target_lookup[key])

    return matched


def add_authorship_mentions(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()

    def build_mentions_series(target_col: str) -> pd.Series:
        # Paper-level contribution mode:
        # use the normalized org list directly (deduped) and avoid affiliation-text matching.
        return df[target_col].map(_ordered_unique)

    df["non_academic_authorship_mentions"] = build_mentions_series("non_academic_institutions_norm")
    df["non_company_non_academic_authorship_mentions"] = build_mentions_series(
        "non_company_non_academic_institutions"
    )
    df["company_authorship_mentions"] = build_mentions_series("company_institutions_norm")
    df["uk_company_authorship_mentions"] = build_mentions_series("uk_company_institutions_norm")
    df["non_uk_company_authorship_mentions"] = build_mentions_series("non_uk_company_institutions")

    return df


def _parse_index_list(value: Any) -> list[int]:
    items = parse_list(value)
    out: list[int] = []
    for item in items:
        try:
            idx = int(float(item))
        except Exception:
            continue
        if idx >= 0:
            out.append(idx)
    return sorted(set(out))


def _safe_org_at(research_orgs: list[Any], idx: int) -> dict[str, Any] | None:
    if idx < 0 or idx >= len(research_orgs):
        return None
    org = research_orgs[idx]
    if isinstance(org, dict):
        return org
    return None


def _org_name(org: dict[str, Any] | None) -> str:
    if not isinstance(org, dict):
        return ""
    return normalize_name(org.get("name") or org.get("display_name"))


def _org_types(org: dict[str, Any] | None) -> set[str]:
    if not isinstance(org, dict):
        return set()
    out: set[str] = set()
    for t in parse_list(org.get("types")):
        s = normalize_name(t).casefold()
        if s:
            out.add(s)
    return out


def _is_uk_org_country(org: dict[str, Any] | None) -> bool:
    if not isinstance(org, dict):
        return False

    cc = normalize_name(org.get("country_code")).upper()
    if cc in {"GB", "UK", "GBR"}:
        return True

    country_name = normalize_name(org.get("country_name")).casefold()
    if country_name in {
        "united kingdom",
        "great britain",
        "england",
        "scotland",
        "wales",
        "northern ireland",
    }:
        return True

    return False


def classify_non_academic_sector(
    org: dict[str, Any] | None,
    *,
    is_company_index: bool = False,
    is_uk_company_index: bool = False,
) -> str:
    """Classify one organization into the collaborator taxonomy."""
    name = _org_name(org)
    lowered = name.casefold()
    types = _org_types(org)
    is_uk = _is_uk_org_country(org)

    # Index-based company labels from classifier are strongest signals, but UK
    # status is corrected using the organization's country metadata.
    if is_uk_company_index:
        return "UK company" if is_uk else "Company (non-UK)"
    if is_company_index:
        return "UK company" if is_uk else "Company (non-UK)"
    if "company" in types:
        return "UK company" if is_uk else "Company (non-UK)"

    if "healthcare" in types or any(
        k in lowered
        for k in [
            "hospital",
            "clinic",
            "clinical",
            "nhs",
            "medical center",
            "medical centre",
            "health system",
            "infirmary",
            "foundation trust",
        ]
    ):
        return "Hospital/Clinical"

    if "government" in types or any(
        k in lowered
        for k in [
            "ministry",
            "department of",
            "agency",
            "public health",
            "national institute",
            "national institutes",
            "institute for health and care research",
            "statens",
            "nih",
            "cdc",
        ]
    ):
        return "Government/Public"

    if "education" in types or any(
        k in lowered
        for k in [
            "university",
            "college",
            "school of medicine",
            "polytechnic",
        ]
    ):
        return "University/HEI"

    if "nonprofit" in types or "non-profit" in types or any(
        k in lowered
        for k in [
            "nonprofit",
            "non-profit",
            "charity",
            "foundation",
            "trust",
        ]
    ):
        return "Nonprofit/Charity"

    # Keep company heuristic after healthcare/public/university checks to avoid
    # false positives such as hospitals ending with "Inc".
    if _COMPANY_SUFFIX_RE.search(lowered) or any(
        k in lowered
        for k in [
            " biotechnology",
            " biotech",
            " therapeutics",
            " pharmaceuticals",
            " pharma",
        ]
    ):
        return "UK company" if is_uk else "Company (non-UK)"

    if "facility" in types or any(
        k in lowered
        for k in [
            "institute",
            "centre",
            "center",
            "laboratory",
            "laboratoire",
            "lab",
            "unit",
            "biobank",
            "consortium",
        ]
    ):
        return "Research institute/Centre"

    return "Other/Unknown"


def _sector_slug(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", label.casefold()).strip("_")
    return slug


def sector_flag_col(label: str) -> str:
    return f"sector_{_sector_slug(label)}_flag"


def sector_count_col(label: str) -> str:
    return f"sector_{_sector_slug(label)}_n"


def sector_institutions_col(label: str) -> str:
    return f"sector_{_sector_slug(label)}_institutions"


def sector_mentions_col(label: str) -> str:
    return f"sector_{_sector_slug(label)}_authorship_mentions"


def _ordered_unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = org_key(item)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _union_lists_keep_order(list_of_lists: list[list[str]]) -> list[str]:
    out: list[str] = []
    for items in list_of_lists:
        out.extend(items)
    return _ordered_unique(out)


def _finalize_non_academic_sector_columns(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    df["non_academic_sector_labels"] = df["non_academic_sector_records"].apply(
        lambda rows: [r.get("sector", "") for r in rows if normalize_name(r.get("sector"))]
    )
    df["non_academic_sector_names"] = df["non_academic_sector_records"].apply(
        lambda rows: [r.get("name", "") for r in rows if normalize_name(r.get("name"))]
    )
    df["non_academic_sector_counts"] = df["non_academic_sector_labels"].apply(
        lambda labels: dict(Counter(labels))
    )
    # Explicit collaborator taxonomy aliases (preferred semantic names).
    df["collaborator_sector_records"] = df["non_academic_sector_records"]
    df["collaborator_sector_labels"] = df["non_academic_sector_labels"]
    df["collaborator_sector_names"] = df["non_academic_sector_names"]
    df["collaborator_sector_counts"] = df["non_academic_sector_counts"]

    # Expand sector-specific institution lists from canonical sector records.
    for label in NON_ACADEMIC_SECTOR_LABELS:
        inst_col = sector_institutions_col(label)
        df[inst_col] = df["non_academic_sector_records"].apply(
            lambda rows, _label=label: _ordered_unique(
                [normalize_name(r.get("name")) for r in rows if normalize_name(r.get("name")) and r.get("sector") == _label]
            )
        )

    # Build sector-specific contribution lists at paper level only.
    # Do not use affiliation-text matching here.
    for label in NON_ACADEMIC_SECTOR_LABELS:
        inst_col = sector_institutions_col(label)
        mention_col = sector_mentions_col(label)
        df[mention_col] = df[inst_col]

    for label in NON_ACADEMIC_SECTOR_LABELS:
        slug = _sector_slug(label)
        df[f"sector_{slug}_flag"] = df["non_academic_sector_labels"].apply(
            lambda labels: int(label in set(labels))
        )
        df[f"sector_{slug}_n"] = df["non_academic_sector_labels"].apply(
            lambda labels: int(sum(1 for x in labels if x == label))
        )

    df["sector_company_any_flag"] = df["non_academic_sector_labels"].apply(
        lambda labels: int(any(x in {"Company (non-UK)", "UK company"} for x in labels))
    )
    df["sector_uk_company_flag"] = df["non_academic_sector_labels"].apply(
        lambda labels: int("UK company" in set(labels))
    )

    # Synchronize legacy analysis columns from taxonomy so downstream analysis is
    # taxonomy-backed even when using historical function names.
    non_academic_insts_col = [sector_institutions_col(label) for label in NON_ACADEMIC_SECTOR_LABELS]
    company_non_uk_col = sector_institutions_col("Company (non-UK)")
    uk_company_col = sector_institutions_col("UK company")
    company_cols = [company_non_uk_col, uk_company_col]
    non_company_cols = [c for c in non_academic_insts_col if c not in company_cols]

    df["non_academic_institutions_norm"] = df.apply(
        lambda row: _union_lists_keep_order([row[c] for c in non_academic_insts_col]),
        axis=1,
    )
    df["collaborator_institutions_norm"] = df["non_academic_institutions_norm"]
    df["company_institutions_norm"] = df.apply(
        lambda row: _union_lists_keep_order([row[c] for c in company_cols]),
        axis=1,
    )
    df["uk_company_institutions_norm"] = df[uk_company_col]
    df["non_uk_company_institutions"] = df[company_non_uk_col]
    df["non_company_non_academic_institutions"] = df.apply(
        lambda row: _union_lists_keep_order([row[c] for c in non_company_cols]),
        axis=1,
    )

    non_mentions_cols = [sector_mentions_col(label) for label in NON_ACADEMIC_SECTOR_LABELS]
    company_mentions_cols = [sector_mentions_col("Company (non-UK)"), sector_mentions_col("UK company")]
    non_company_mentions_cols = [c for c in non_mentions_cols if c not in company_mentions_cols]
    df["non_academic_authorship_mentions"] = df.apply(
        lambda row: _union_lists_keep_order([row[c] for c in non_mentions_cols]),
        axis=1,
    )
    df["collaborator_authorship_mentions"] = df["non_academic_authorship_mentions"]
    df["company_authorship_mentions"] = df.apply(
        lambda row: _union_lists_keep_order([row[c] for c in company_mentions_cols]),
        axis=1,
    )
    df["uk_company_authorship_mentions"] = df[sector_mentions_col("UK company")]
    df["non_uk_company_authorship_mentions"] = df[sector_mentions_col("Company (non-UK)")]
    df["non_company_non_academic_authorship_mentions"] = df.apply(
        lambda row: _union_lists_keep_order([row[c] for c in non_company_mentions_cols]),
        axis=1,
    )

    df["non_academic_institutions"] = df["non_academic_institutions_norm"]
    df["collaborator_institutions"] = df["collaborator_institutions_norm"]
    df["company_institutions"] = df["company_institutions_norm"]
    df["uk_company_institutions"] = df["uk_company_institutions_norm"]

    df["any_sector_collab_flag"] = (df["collaborator_institutions_norm"].map(len) > 0).astype(int)
    # Keep legacy column for compatibility, but semantics now equal
    # "has at least one collaborator taxonomy label".
    df["non_academic_flag"] = df["any_sector_collab_flag"]
    df["company_flag"] = (df["company_institutions_norm"].map(len) > 0).astype(int)
    df["uk_company_flag"] = (df["uk_company_institutions_norm"].map(len) > 0).astype(int)
    df["non_uk_company_flag"] = (df["non_uk_company_institutions"].map(len) > 0).astype(int)
    df["non_company_non_academic_flag"] = (
        df["non_company_non_academic_institutions"].map(len) > 0
    ).astype(int)

    if "academic_institutions_norm" in df.columns:
        df["academic_flag"] = (df["academic_institutions_norm"].map(len) > 0).astype(int)
        df["academic_company_collab_flag"] = (
            (df["academic_institutions_norm"].map(len) > 0)
            & (df["company_institutions_norm"].map(len) > 0)
        ).astype(int)
        df["academic_uk_company_collab_flag"] = (
            (df["academic_institutions_norm"].map(len) > 0)
            & (df["uk_company_institutions_norm"].map(len) > 0)
        ).astype(int)
    df["company_only_flag"] = (
        (df["company_institutions_norm"].map(len) > 0)
        & (df["non_company_non_academic_institutions"].map(len) == 0)
    ).astype(int)
    df["uk_company_only_flag"] = (
        (df["uk_company_institutions_norm"].map(len) > 0)
        & (df["non_uk_company_institutions"].map(len) == 0)
    ).astype(int)
    return df


def add_non_academic_sector_taxonomy(df_in: pd.DataFrame) -> pd.DataFrame:
    """Annotate each collaborator org mention with a coarse sector taxonomy."""
    df = df_in.copy()
    parsed_existing_records: pd.Series | None = None

    # Prefer taxonomy emitted directly by the classifier output when present.
    if "non_academic_sector_records" in df.columns:

        def parse_records(value: Any) -> list[dict[str, Any]]:
            parsed = parse_list(value)
            out: list[dict[str, Any]] = []
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                name = normalize_name(item.get("name"))
                sector = normalize_name(item.get("sector"))
                if not name or not sector:
                    continue
                if sector not in NON_ACADEMIC_SECTOR_LABELS:
                    continue
                raw_index = item.get("index", -1)
                try:
                    index_value = int(float(raw_index))
                except Exception:
                    index_value = -1
                rec = {
                    "index": index_value,
                    "name": name,
                    "sector": sector,
                    "country_code": normalize_name(item.get("country_code")).upper(),
                    "country_name": normalize_name(item.get("country_name")),
                    "types": parse_list(item.get("types")),
                    "is_company_index": bool(item.get("is_company_index")),
                    "is_uk_company_index": bool(item.get("is_uk_company_index")),
                }
                out.append(rec)
            return out

        parsed_existing_records = df["non_academic_sector_records"].apply(parse_records)
        df["non_academic_sector_records"] = parsed_existing_records

    if "research_orgs_list" not in df.columns:
        df["research_orgs_list"] = parse_col_as_list(df, "research_orgs")
    if "non_academic_indices_list" not in df.columns:
        if "non_academic_indices" in df.columns:
            df["non_academic_indices_list"] = df["non_academic_indices"].apply(_parse_index_list)
        else:
            df["non_academic_indices_list"] = [[] for _ in range(len(df))]
    if "academic_indices_list" not in df.columns:
        if "academic_indices" in df.columns:
            df["academic_indices_list"] = df["academic_indices"].apply(_parse_index_list)
        else:
            df["academic_indices_list"] = [[] for _ in range(len(df))]
    if "company_indices_list" not in df.columns:
        if "company_indices" in df.columns:
            df["company_indices_list"] = df["company_indices"].apply(_parse_index_list)
        else:
            df["company_indices_list"] = [[] for _ in range(len(df))]
    if "uk_company_indices_list" not in df.columns:
        if "uk_company_indices" in df.columns:
            df["uk_company_indices_list"] = df["uk_company_indices"].apply(_parse_index_list)
        else:
            df["uk_company_indices_list"] = [[] for _ in range(len(df))]

    def build_row_records(row: pd.Series) -> list[dict[str, Any]]:
        research_orgs = row.get("research_orgs_list", [])
        if not isinstance(research_orgs, list):
            research_orgs = parse_list(research_orgs)

        # Use the full collaborator set (academic + non-academic legacy indices)
        # to build the 8-sector taxonomy.
        collab_idx = list(row.get("non_academic_indices_list", [])) + list(
            row.get("academic_indices_list", [])
        )
        if not collab_idx and (
            row.get("non_academic_institutions_norm") or row.get("academic_institutions_norm")
        ):
            # Fallback: map collaborator names to research_orgs when index lists are missing.
            key_to_idx: dict[str, int] = {}
            for i, org in enumerate(research_orgs):
                if not isinstance(org, dict):
                    continue
                key = org_key(_org_name(org))
                if key and key not in key_to_idx:
                    key_to_idx[key] = i
            fallback_names = list(row.get("non_academic_institutions_norm", [])) + list(
                row.get("academic_institutions_norm", [])
            )
            for name in fallback_names:
                key = org_key(name)
                if key and key in key_to_idx:
                    collab_idx.append(key_to_idx[key])

        company_idx = set(row.get("company_indices_list", []))
        uk_company_idx = set(row.get("uk_company_indices_list", []))

        out: list[dict[str, Any]] = []
        for idx in sorted(set(collab_idx)):
            org = _safe_org_at(research_orgs, idx)
            if org is None:
                continue

            name = _org_name(org)
            if not name:
                continue

            sector = classify_non_academic_sector(
                org,
                is_company_index=idx in company_idx,
                is_uk_company_index=idx in uk_company_idx,
            )

            out.append(
                {
                    "index": int(idx),
                    "name": name,
                    "sector": sector,
                    "country_code": normalize_name(org.get("country_code")).upper(),
                    "country_name": normalize_name(org.get("country_name")),
                    "types": list(parse_list(org.get("types"))),
                    "is_company_index": bool(idx in company_idx),
                    "is_uk_company_index": bool(idx in uk_company_idx),
                }
            )

        return out

    has_indices_and_orgs = (
        "research_orgs_list" in df.columns
        and int(df["research_orgs_list"].map(len).sum()) > 0
        and (
            ("non_academic_indices_list" in df.columns and int(df["non_academic_indices_list"].map(len).sum()) > 0)
            or ("academic_indices_list" in df.columns and int(df["academic_indices_list"].map(len).sum()) > 0)
        )
    )
    if has_indices_and_orgs:
        df["non_academic_sector_records"] = df.apply(build_row_records, axis=1)
    elif parsed_existing_records is not None:
        df["non_academic_sector_records"] = parsed_existing_records
    else:
        df["non_academic_sector_records"] = [[] for _ in range(len(df))]
    return _finalize_non_academic_sector_columns(df)


def build_non_academic_sector_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize collaborator taxonomy sectors (mentions and papers)."""
    if "non_academic_sector_labels" not in df.columns:
        return pd.DataFrame(
            columns=[
                "sector",
                "institution_mentions",
                "papers",
                "mentions_pct",
                "papers_pct_of_non_academic_papers",
                "papers_pct_of_taxonomy_papers",
            ]
        )

    mention_counts: Counter = Counter()
    paper_counts: Counter = Counter()
    total_taxonomy_papers = int((df["non_academic_sector_labels"].map(len) > 0).sum())

    for labels in df["non_academic_sector_labels"]:
        for label in labels:
            mention_counts[label] += 1
        for label in set(labels):
            paper_counts[label] += 1

    total_mentions = int(sum(mention_counts.values()))

    rows: list[dict[str, Any]] = []
    for label in NON_ACADEMIC_SECTOR_LABELS:
        mentions = int(mention_counts.get(label, 0))
        papers = int(paper_counts.get(label, 0))
        rows.append(
            {
                "sector": label,
                "institution_mentions": mentions,
                "papers": papers,
                "mentions_pct": (100 * mentions / total_mentions) if total_mentions else 0.0,
                "papers_pct_of_non_academic_papers": (100 * papers / total_taxonomy_papers)
                if total_taxonomy_papers
                else 0.0,
                "papers_pct_of_taxonomy_papers": (100 * papers / total_taxonomy_papers)
                if total_taxonomy_papers
                else 0.0,
            }
        )

    return pd.DataFrame(rows)


def plot_non_academic_sector_breakdown(
    sector_df: pd.DataFrame,
    value_col: str = "institution_mentions",
) -> None:
    """Plot collaborator sector taxonomy breakdown."""
    if sector_df.empty or value_col not in sector_df.columns:
        print("No collaborator sector summary available for plotting.")
        return

    plot_df = sector_df.copy()
    plot_df = plot_df.sort_values(value_col, ascending=True)
    color_map = {
        "University/HEI": "#457B9D",
        "Hospital/Clinical": "#2A9D8F",
        "Government/Public": "#8D99AE",
        "Research institute/Centre": "#6A994E",
        "Nonprofit/Charity": "#A1C181",
        "Company (non-UK)": "#E9C46A",
        "UK company": "#D4AF37",
        "Other/Unknown": "#BDBDBD",
    }
    bar_colors = [color_map.get(s, "#BDBDBD") for s in plot_df["sector"]]

    plt.figure(figsize=(10.5, 5.5), dpi=300)
    plt.barh(
        plot_df["sector"],
        plot_df[value_col],
        color=bar_colors,
        edgecolor="black",
        linewidth=0.5,
    )
    xlabel = "Institution mentions" if value_col == "institution_mentions" else "Paper count"
    plt.xlabel(xlabel)
    plt.ylabel("Sector")
    plt.title("Collaborator taxonomy")
    plt.grid(True, axis="x", alpha=0.25)
    plt.tight_layout()
    plt.show()


def mention_coverage(df: pd.DataFrame, paper_col: str, mention_col: str) -> tuple[int, int, float]:
    paper_mask = df[paper_col].map(len) > 0
    mention_mask = df[mention_col].map(len) > 0
    total = int(paper_mask.sum())
    matched = int((paper_mask & mention_mask).sum())
    pct = round(100 * matched / total, 2) if total else 0.0
    return matched, total, pct


def build_counts(paper_series: pd.Series, mention_series: pd.Series) -> tuple[Counter, Counter, dict[str, str]]:
    mention_counts: Counter = Counter()
    paper_counts: Counter = Counter()
    labels: dict[str, str] = {}

    for paper_items, mention_items in zip(paper_series, mention_series):
        seen: set[str] = set()

        for name in paper_items:
            key = org_key(name)
            if not key:
                continue
            if key not in seen:
                paper_counts[key] += 1
                seen.add(key)
            labels.setdefault(key, name)

        for name in mention_items:
            key = org_key(name)
            if not key:
                continue
            mention_counts[key] += 1
            labels.setdefault(key, name)

    return mention_counts, paper_counts, labels


def top_table(paper_series: pd.Series, mention_series: pd.Series, n: int = 25) -> tuple[pd.DataFrame, Counter]:
    mention_counts, paper_counts, labels = build_counts(paper_series, mention_series)

    rows: list[dict[str, Any]] = []
    for key, papers in paper_counts.most_common(n):
        rows.append(
            {
                "org": labels.get(key, key),
                "papers": int(papers),
            }
        )

    return pd.DataFrame(rows), paper_counts


def build_top_sector_collaborator_tables(
    df: pd.DataFrame,
    n: int = 30,
) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    """Top collaborator tables for each taxonomy sector."""
    tables: dict[str, pd.DataFrame] = {}
    unique_counts: dict[str, int] = {}

    for label in NON_ACADEMIC_SECTOR_LABELS:
        inst_col = sector_institutions_col(label)
        mention_col = sector_mentions_col(label)
        if inst_col not in df.columns or mention_col not in df.columns:
            continue
        top_df, counts = top_table(df[inst_col], df[mention_col], n=n)
        tables[label] = top_df
        unique_counts[label] = len(counts)

    if "company_institutions_norm" in df.columns and "company_authorship_mentions" in df.columns:
        top_company_any_df, company_any_counts = top_table(
            df["company_institutions_norm"],
            df["company_authorship_mentions"],
            n=n,
        )
        tables["Company (any)"] = top_company_any_df
        unique_counts["Company (any)"] = len(company_any_counts)

    return tables, unique_counts


def build_top_collaborator_tables(df: pd.DataFrame, n: int = 30) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    """Backward-compatible wrapper returning legacy keys backed by taxonomy."""
    sector_tables, sector_unique = build_top_sector_collaborator_tables(df, n=n)

    tables = {
        "non_academic": top_table(df["non_academic_institutions_norm"], df["non_academic_authorship_mentions"], n=n)[0],
        "non_company_non_academic": top_table(
            df["non_company_non_academic_institutions"],
            df["non_company_non_academic_authorship_mentions"],
            n=n,
        )[0],
        "company": sector_tables.get("Company (any)", pd.DataFrame()),
        "uk_company": sector_tables.get("UK company", pd.DataFrame()),
        "non_uk_company": sector_tables.get("Company (non-UK)", pd.DataFrame()),
    }

    unique_counts = {
        "non_academic": len(build_counts(df["non_academic_institutions_norm"], df["non_academic_authorship_mentions"])[1]),
        "non_company_non_academic": len(
            build_counts(
                df["non_company_non_academic_institutions"],
                df["non_company_non_academic_authorship_mentions"],
            )[1]
        ),
        "company": int(sector_unique.get("Company (any)", 0)),
        "uk_company": int(sector_unique.get("UK company", 0)),
        "non_uk_company": int(sector_unique.get("Company (non-UK)", 0)),
    }

    return tables, unique_counts


def _pct(num: int | float, den: int | float) -> float:
    return round(100 * num / den, 2) if den else 0.0


def compute_summary_stats(df: pd.DataFrame, unique_counts: dict[str, int] | None = None) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "total_papers": int(len(df)),
        "papers_with_any_sector": int((df["non_academic_sector_labels"].map(len) > 0).sum()),
    }
    stats["pct_with_any_sector"] = _pct(stats["papers_with_any_sector"], stats["total_papers"])

    sector_summary = build_non_academic_sector_summary(df)
    stats["sector_summary_table"] = sector_summary

    for _, row in sector_summary.iterrows():
        label = str(row["sector"])
        slug = _sector_slug(label)
        stats[f"mentions_{slug}"] = int(row["institution_mentions"])
        stats[f"papers_{slug}"] = int(row["papers"])
        stats[f"mentions_pct_{slug}"] = float(row["mentions_pct"])
        pct_col = (
            "papers_pct_of_taxonomy_papers"
            if "papers_pct_of_taxonomy_papers" in row.index
            else "papers_pct_of_non_academic_papers"
        )
        stats[f"papers_pct_{slug}"] = float(row[pct_col])

        n_col = sector_count_col(label)
        stats[f"mean_{slug}_orgs"] = (
            round(float(df[n_col].mean()), 3) if n_col in df.columns else 0.0
        )

        inst_col = sector_institutions_col(label)
        mention_col = sector_mentions_col(label)
        if inst_col in df.columns and mention_col in df.columns:
            stats[f"coverage_{slug}"] = mention_coverage(df, inst_col, mention_col)
        else:
            stats[f"coverage_{slug}"] = (0, 0, 0.0)

    # Convenience combined company metrics (taxonomy-backed).
    company_any = int((df["sector_company_any_flag"] == 1).sum()) if "sector_company_any_flag" in df.columns else 0
    uk_company = int((df["sector_uk_company_flag"] == 1).sum()) if "sector_uk_company_flag" in df.columns else 0
    non_uk_company = int((df.get("sector_company_non_uk_flag", 0) == 1).sum()) if "sector_company_non_uk_flag" in df.columns else 0
    stats["papers_with_company_any"] = company_any
    stats["papers_with_uk_company"] = uk_company
    stats["papers_with_company_non_uk"] = non_uk_company
    stats["pct_with_company_any"] = _pct(company_any, stats["total_papers"])
    stats["pct_with_uk_company"] = _pct(uk_company, stats["total_papers"])
    stats["pct_with_company_non_uk"] = _pct(non_uk_company, stats["total_papers"])

    if unique_counts:
        for key, value in unique_counts.items():
            stats[f"unique_{_sector_slug(key)}"] = int(value)

    return stats


def summary_stats_as_lines(stats: dict[str, Any]) -> list[str]:
    lines = [
        f"Total papers: {stats['total_papers']}",
        f"Papers with any taxonomy sector collaborator: {stats['papers_with_any_sector']} ({stats['pct_with_any_sector']}%)",
        f"Papers with any company collaborator: {stats['papers_with_company_any']} ({stats['pct_with_company_any']}%)",
        f"Papers with Company (non-UK): {stats['papers_with_company_non_uk']} ({stats['pct_with_company_non_uk']}%)",
        f"Papers with UK company: {stats['papers_with_uk_company']} ({stats['pct_with_uk_company']}%)",
    ]

    for label in NON_ACADEMIC_SECTOR_LABELS:
        slug = _sector_slug(label)
        mentions = stats.get(f"mentions_{slug}", 0)
        papers = stats.get(f"papers_{slug}", 0)
        mention_pct = stats.get(f"mentions_pct_{slug}", 0.0)
        paper_pct = stats.get(f"papers_pct_{slug}", 0.0)
        mean_orgs = stats.get(f"mean_{slug}_orgs", 0.0)
        lines.append(
            f"{label}: {mentions} mentions ({mention_pct:.1f}%), {papers} papers ({paper_pct:.1f}%), mean orgs/paper={mean_orgs}"
        )
        unique_key = f"unique_{slug}"
        if unique_key in stats:
            lines.append(f"Unique {label} institutions: {stats[unique_key]}")
    return lines


def apply_project_plot_style() -> None:
    """Apply a publication-oriented style aligned with project notebooks."""
    plt.rcParams.update(
        {
            "font.family": "Helvetica",
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.axisbelow": True,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "lines.markersize": DEFAULT_MARKER_SIZE,
            "axes.prop_cycle": cycler(
                color=[
                    "#1b9e77",
                    "#d95f02",
                    "#7570b3",
                    "#e7298a",
                    "#66a61e",
                    "#e6ab02",
                    "#a6761d",
                    "#666666",
                ]
            ),
        }
    )


def plot_top_orgs(
    table: pd.DataFrame,
    title: str,
    xlabel: str,
    ylabel: str,
    color: str,
    n: int = 20,
    value_col: str = "papers",
) -> None:
    if table.empty:
        print(f"No data for: {title}")
        return

    plot_df = table.head(n).sort_values(value_col, ascending=True)

    plt.figure(figsize=(11, 7), dpi=150)
    plt.barh(
        plot_df["org"],
        plot_df[value_col],
        color=color,
        edgecolor="black",
        linewidth=0.3,
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.show()


def plot_collaborator_count_distributions(df: pd.DataFrame) -> None:
    nrows = 2
    ncols = 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 8), dpi=150)
    axes = axes.flatten()

    color_map = {
        "University/HEI": "#457B9D",
        "Hospital/Clinical": "#2A9D8F",
        "Government/Public": "#8D99AE",
        "Research institute/Centre": "#6A994E",
        "Nonprofit/Charity": "#A1C181",
        "Company (non-UK)": "#E9C46A",
        "UK company": "#D4AF37",
        "Other/Unknown": "#BDBDBD",
    }

    series_spec = []
    for idx, label in enumerate(NON_ACADEMIC_SECTOR_LABELS):
        if idx >= len(axes):
            break
        col = sector_count_col(label)
        if col not in df.columns:
            counts = pd.Series([0] * len(df))
        else:
            counts = df[col]
        series_spec.append((axes[idx], counts, f"{label} orgs per paper", color_map.get(label, "#BDBDBD")))

    for ax, counts, title, color in series_spec:
        nonzero = counts[counts > 0]
        if nonzero.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_title(title)
            continue

        max_count = int(nonzero.max())
        bins = np.arange(1, max_count + 2) - 0.5
        ax.hist(nonzero, bins=bins, color=color, edgecolor="black", linewidth=0.3)
        ax.set_title(title)
        ax.set_xlabel("Count")
        ax.set_ylabel("Papers")
        ax.set_xticks(range(1, max_count + 1))
        ax.grid(True, axis="y", alpha=0.2)

    for idx in range(len(series_spec), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.show()


def ensure_year_column(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        return df

    df["year"] = pd.to_datetime(df.get("date"), errors="coerce").dt.year.astype("Int64")
    return df


def _year_index(start_year: int, end_year: int) -> list[int]:
    return list(range(start_year, end_year + 1))


def _any_sector_collab_mask(df: pd.DataFrame) -> pd.Series:
    if "any_sector_collab_flag" in df.columns:
        return df["any_sector_collab_flag"] == 1
    if "collaborator_sector_labels" in df.columns:
        return df["collaborator_sector_labels"].map(len) > 0
    if "non_academic_sector_labels" in df.columns:
        return df["non_academic_sector_labels"].map(len) > 0

    sector_cols = [sector_flag_col(label) for label in NON_ACADEMIC_SECTOR_LABELS if sector_flag_col(label) in df.columns]
    if sector_cols:
        return df[sector_cols].fillna(0).astype(int).sum(axis=1) > 0

    if "non_academic_flag" in df.columns:
        return df["non_academic_flag"] == 1
    return pd.Series([False] * len(df), index=df.index)


def plot_cumulative_by_type(df: pd.DataFrame, start_year: int = 2014, end_year: int = 2025) -> None:
    year_range = _year_index(start_year, end_year)
    use_sector = all(sector_flag_col(label) in df.columns for label in NON_ACADEMIC_SECTOR_LABELS)

    if use_sector:
        color_map = {
            "University/HEI": "#457B9D",
            "Hospital/Clinical": "#2A9D8F",
            "Government/Public": "#8D99AE",
            "Research institute/Centre": "#6A994E",
            "Nonprofit/Charity": "#A1C181",
            "Company (non-UK)": "#5E548E",
            "UK company": "#D4AF37",
            "Other/Unknown": "#BDBDBD",
        }
        series_spec = [
            (label, sector_flag_col(label), color_map.get(label, "#BDBDBD"))
            for label in NON_ACADEMIC_SECTOR_LABELS
            if label != "University/HEI"
        ]
        title = (
            "Cumulative papers by taxonomy sector, excl. University/HEI "
            f"({start_year}-{end_year})"
        )
    else:
        series_spec = [
            ("Any taxonomy collaborator", "non_academic_flag", "#D4AF37"),
            ("Company (any)", "company_flag", "#2A9D8F"),
            ("UK company", "uk_company_flag", "#345995"),
            ("Non-UK company", "non_uk_company_flag", "#5E548E"),
        ]
        title = f"Cumulative papers by collaborator type ({start_year}-{end_year})"

    plt.figure(figsize=(10, 5), dpi=150)
    for label, col, color in series_spec:
        yearly = df[df[col] == 1].groupby("year").size().reindex(year_range, fill_value=0)
        plt.plot(
            year_range,
            yearly.cumsum().values,
            marker="o",
            linewidth=2,
            color=color,
            label=label,
        )
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel("Cumulative paper count")
    plt.xticks(year_range, rotation=45)
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_yearly_share_by_type(df: pd.DataFrame, start_year: int = 2014, end_year: int = 2025) -> None:
    year_range = _year_index(start_year, end_year)

    yearly_total = df.groupby("year").size().reindex(year_range, fill_value=0)
    use_sector = all(sector_flag_col(label) in df.columns for label in NON_ACADEMIC_SECTOR_LABELS)

    if use_sector:
        color_map = {
            "University/HEI": "#457B9D",
            "Hospital/Clinical": "#2A9D8F",
            "Government/Public": "#8D99AE",
            "Research institute/Centre": "#6A994E",
            "Nonprofit/Charity": "#A1C181",
            "Company (non-UK)": "#5E548E",
            "UK company": "#D4AF37",
            "Other/Unknown": "#BDBDBD",
        }
        series_spec = [
            (label, sector_flag_col(label), color_map.get(label, "#BDBDBD"))
            for label in NON_ACADEMIC_SECTOR_LABELS
        ]
        title = f"Share of papers by taxonomy sector ({start_year}-{end_year})"
    else:
        series_spec = [
            ("Any taxonomy collaborator", "non_academic_flag", "#D4AF37"),
            ("Company (any)", "company_flag", "#2A9D8F"),
            ("UK company", "uk_company_flag", "#345995"),
            ("Non-UK company", "non_uk_company_flag", "#5E548E"),
        ]
        title = f"Share of papers with collaborator types ({start_year}-{end_year})"

    plt.figure(figsize=(10, 5), dpi=150)
    for label, col, color in series_spec:
        yearly = df[df[col] == 1].groupby("year").size().reindex(year_range, fill_value=0)
        share = np.where(yearly_total > 0, yearly / yearly_total, 0)
        plt.plot(year_range, share * 100, marker="o", linewidth=2, color=color, label=label)
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel("Percent of papers")
    plt.xticks(year_range, rotation=45)
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_company_share_within_non_academic(
    df: pd.DataFrame,
    start_year: int = 2014,
    end_year: int = 2025,
) -> None:
    year_range = _year_index(start_year, end_year)
    any_sector_mask = _any_sector_collab_mask(df)
    yearly_non = df[any_sector_mask].groupby("year").size().reindex(year_range, fill_value=0)
    if "sector_company_any_flag" in df.columns:
        yearly_company = df[df["sector_company_any_flag"] == 1].groupby("year").size().reindex(year_range, fill_value=0)
    else:
        yearly_company = df[df["company_flag"] == 1].groupby("year").size().reindex(year_range, fill_value=0)
    if sector_flag_col("UK company") in df.columns:
        yearly_uk = df[df[sector_flag_col("UK company")] == 1].groupby("year").size().reindex(year_range, fill_value=0)
    else:
        yearly_uk = df[df["uk_company_flag"] == 1].groupby("year").size().reindex(year_range, fill_value=0)
    if sector_flag_col("Company (non-UK)") in df.columns:
        yearly_non_uk = df[df[sector_flag_col("Company (non-UK)")] == 1].groupby("year").size().reindex(year_range, fill_value=0)
    else:
        yearly_non_uk = df[df["non_uk_company_flag"] == 1].groupby("year").size().reindex(year_range, fill_value=0)

    share_company_in_non = np.where(yearly_non > 0, yearly_company / yearly_non, 0)
    share_uk_in_non = np.where(yearly_non > 0, yearly_uk / yearly_non, 0)
    share_non_uk_in_non = np.where(yearly_non > 0, yearly_non_uk / yearly_non, 0)

    plt.figure(figsize=(10, 5), dpi=150)
    plt.plot(year_range, share_company_in_non * 100, marker="o", linewidth=2, color="#2A9D8F", label="Company (any)")
    plt.plot(year_range, share_uk_in_non * 100, marker="o", linewidth=2, color="#D4AF37", label="UK company")
    plt.plot(year_range, share_non_uk_in_non * 100, marker="o", linewidth=2, color="#5E548E", label="Company (non-UK)")
    plt.title(f"Share of taxonomy-collaboration papers with company sectors ({start_year}-{end_year})")
    plt.xlabel("Year")
    plt.ylabel("Percent of taxonomy-collaboration papers")
    plt.xticks(year_range, rotation=45)
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.show()


def _collaboration_mix_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    priority = [
        "UK company",
        "Company (non-UK)",
        "Hospital/Clinical",
        "University/HEI",
        "Government/Public",
        "Research institute/Centre",
        "Nonprofit/Charity",
        "Other/Unknown",
    ]

    def pick_primary_sector(row: pd.Series) -> str:
        counts = row.get("collaborator_sector_counts", row.get("non_academic_sector_counts", {}))
        if not isinstance(counts, dict) or not counts:
            return "No taxonomy collaborator"
        best_label = None
        best_count = -1
        for label in priority:
            c = int(counts.get(label, 0))
            if c > best_count:
                best_count = c
                best_label = label
        return best_label or "No taxonomy collaborator"

    primary = df.apply(pick_primary_sector, axis=1)
    labels = ["No taxonomy collaborator"] + priority
    return {label: (primary == label) for label in labels}


def plot_collaboration_mix_stacked_area(
    df: pd.DataFrame,
    start_year: int = 2014,
    end_year: int = 2025,
) -> None:
    """Plot annual paper counts by mutually exclusive collaboration mix."""
    year_range = _year_index(start_year, end_year)
    masks = _collaboration_mix_masks(df)

    labels = list(masks.keys())
    stack_series = []
    for label in labels:
        counts = df[masks[label]].groupby("year").size().reindex(year_range, fill_value=0)
        stack_series.append(counts.values)

    colors = ["#9e9e9e", "#d4af37", "#5e548e", "#2a9d8f", "#457b9d", "#8d99ae", "#6a994e", "#a1c181", "#bdbdbd"]

    plt.figure(figsize=(12, 6), dpi=300)
    plt.stackplot(year_range, stack_series, labels=labels, colors=colors, alpha=0.9)
    plt.title(f"Annual paper volume by primary taxonomy sector ({start_year}-{end_year})")
    plt.xlabel("Year")
    plt.ylabel("Paper count")
    plt.xticks(year_range, rotation=45)
    plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    plt.tight_layout()
    plt.show()


def plot_collaboration_mix_share_stacked_area(
    df: pd.DataFrame,
    start_year: int = 2014,
    end_year: int = 2025,
) -> None:
    """Plot annual paper shares by mutually exclusive collaboration mix (100% stacked)."""
    year_range = _year_index(start_year, end_year)
    masks = _collaboration_mix_masks(df)
    labels = list(masks.keys())

    yearly_arrays = []
    for label in labels:
        yearly_counts = df[masks[label]].groupby("year").size().reindex(year_range, fill_value=0).astype(float)
        yearly_arrays.append(yearly_counts.values)

    stack = np.vstack(yearly_arrays)
    totals = stack.sum(axis=0)
    totals[totals == 0] = 1.0
    shares = 100.0 * stack / totals

    colors = ["#9e9e9e", "#d4af37", "#5e548e", "#2a9d8f", "#457b9d", "#8d99ae", "#6a994e", "#a1c181", "#bdbdbd"]
    plt.figure(figsize=(12, 6), dpi=300)
    plt.stackplot(year_range, shares, labels=labels, colors=colors, alpha=0.92)
    plt.title(f"Annual share by primary taxonomy sector ({start_year}-{end_year})")
    plt.xlabel("Year")
    plt.ylabel("Share of papers (%)")
    plt.ylim(0, 100)
    plt.xticks(year_range, rotation=45)
    plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    plt.tight_layout()
    plt.show()


def plot_flag_overlap_heatmap(
    df: pd.DataFrame,
    save_path: str | Path | None = None,
    save_formats: tuple[str, ...] = ("png", "svg", "pdf"),
    png_dpi: int = 400,
) -> dict[str, Path]:
    """Plot row-normalized overlap percentages between collaboration flags and save outputs."""
    flag_cols = []
    labels = []
    for label in NON_ACADEMIC_SECTOR_LABELS:
        col = sector_flag_col(label)
        if col in df.columns:
            flag_cols.append(col)
            labels.append(label)
    if not flag_cols:
        print("No taxonomy sector flag columns found for overlap heatmap.")
        return {}

    matrix = np.zeros((len(flag_cols), len(flag_cols)))
    for i, row_col in enumerate(flag_cols):
        row_mask = df[row_col] == 1
        row_total = int(row_mask.sum())
        for j, col_col in enumerate(flag_cols):
            both = int((row_mask & (df[col_col] == 1)).sum())
            matrix[i, j] = (100 * both / row_total) if row_total else 0.0

    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=300)
    im = ax.imshow(matrix, cmap="Spectral_r", vmin=0, vmax=100)
    ax.set_title("Collaboration flag overlap (% of row flag papers)")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticklabels(labels)
    ax.grid(False, which="both")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text_color = "white" if value >= 55 else "black"
            ax.text(j, i, f"{value:.1f}%", ha="center", va="center", color=text_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar_ticks = np.arange(0, 101, 20, dtype=float)
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([f"{int(t)}%" for t in cbar_ticks])
    cbar.set_label("Percent of Papers")

    plt.tight_layout()

    if save_path is not None:
        base_out = Path(save_path).with_suffix("")
    else:
        base_out = _figure_dir() / "collaboration_flag_overlap_heatmap"
    base_out.parent.mkdir(parents=True, exist_ok=True)

    saved_paths: dict[str, Path] = {}
    for fmt in save_formats:
        clean_fmt = fmt.lower().lstrip(".")
        out_path = base_out.with_suffix(f".{clean_fmt}")
        save_kwargs: dict[str, Any] = {"bbox_inches": "tight"}
        if clean_fmt == "png":
            save_kwargs["dpi"] = png_dpi
        fig.savefig(out_path, **save_kwargs)
        saved_paths[clean_fmt] = out_path

    plt.show()
    return saved_paths


def plot_citation_distribution_by_group(df: pd.DataFrame, citation_col: str = "times_cited") -> None:
    """Plot citation distributions across collaboration groups (log-scaled)."""
    if citation_col not in df.columns:
        print(f"Citation column not found: {citation_col}")
        return

    citations = pd.to_numeric(df[citation_col], errors="coerce").fillna(0)

    def sector_mask(label: str) -> pd.Series:
        col = sector_flag_col(label)
        if col in df.columns:
            return df[col] == 1
        return pd.Series([False] * len(df), index=df.index)

    any_sector_mask = _any_sector_collab_mask(df)
    groups = [
        ("No taxonomy collaborator", ~any_sector_mask, "#9e9e9e"),
        ("Hospital/Clinical", sector_mask("Hospital/Clinical"), "#2a9d8f"),
        ("University/HEI", sector_mask("University/HEI"), "#457b9d"),
        ("Company (non-UK)", sector_mask("Company (non-UK)"), "#5e548e"),
        ("UK company", sector_mask("UK company"), "#d4af37"),
    ]

    data = []
    labels = []
    colors = []

    for label, mask, color in groups:
        vals = np.log10(citations[mask] + 1)
        vals = vals.replace([np.inf, -np.inf], np.nan).dropna()
        if len(vals) == 0:
            continue
        data.append(vals.values)
        labels.append(f"{label}\n(n={len(vals):,})")
        colors.append(color)

    if not data:
        print("No citation data available for plotting.")
        return

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    box = ax.boxplot(
        data,
        patch_artist=True,
        labels=labels,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.5},
    )

    for patch, color in zip(box["boxes"], colors[: len(data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.6)

    ax.set_title("Citation distributions by taxonomy group")
    ax.set_ylabel("log10(citations + 1)")
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.show()


def plot_collaborator_concentration_curves(df: pd.DataFrame) -> None:
    """Plot collaborator concentration curves (papers covered vs collaborator rank)."""

    def paper_counter(series: pd.Series) -> Counter:
        counts: Counter = Counter()
        for items in series:
            seen = {org_key(x) for x in items if org_key(x)}
            for key in seen:
                counts[key] += 1
        return counts

    def curve_from_counter(counter: Counter) -> tuple[np.ndarray, np.ndarray]:
        vals = np.array(sorted(counter.values(), reverse=True), dtype=float)
        if len(vals) == 0:
            return np.array([]), np.array([])
        x = np.arange(1, len(vals) + 1) / len(vals)
        y = np.cumsum(vals) / vals.sum()
        return x, y

    series_spec: list[tuple[str, Counter, str]] = []
    for label, color in [
        ("Hospital/Clinical", "#2a9d8f"),
        ("University/HEI", "#457b9d"),
    ]:
        col = sector_institutions_col(label)
        if col in df.columns:
            series_spec.append((label, paper_counter(df[col]), color))
    series_spec.append(("Company (any)", paper_counter(df["company_institutions_norm"]), "#d4af37"))

    curves = []
    for label, counter, color in series_spec:
        x, y = curve_from_counter(counter)
        curves.append((label, x, y, color))

    if all(len(x) == 0 for _, x, _, _ in curves):
        print("No collaborator concentration data available.")
        return

    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot([0, 1], [0, 1], linestyle="--", color="#bdbdbd", linewidth=1.2, label="Uniform baseline")
    for label, x, y, color in curves:
        if len(x) == 0:
            continue
        plt.plot(x, y, linewidth=2.2, color=color, label=f"{label} collaborators")

    plt.title("Collaborator concentration curve")
    plt.xlabel("Fraction of collaborators (ranked by papers)")
    plt.ylabel("Fraction of papers covered")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()


def plot_yearly_median_log_citations_by_group(
    df: pd.DataFrame,
    citation_col: str = "times_cited",
    start_year: int = 2014,
    end_year: int = 2025,
    min_papers_per_point: int = 20,
) -> None:
    """Plot yearly median log10(citations+1) across collaboration groups."""
    if citation_col not in df.columns:
        print(f"Citation column not found: {citation_col}")
        return

    plot_df = ensure_year_column(df)
    year_range = _year_index(start_year, end_year)
    citations_log = np.log10(pd.to_numeric(plot_df[citation_col], errors="coerce").fillna(0) + 1)

    def sector_mask(label: str) -> pd.Series:
        col = sector_flag_col(label)
        if col in plot_df.columns:
            return plot_df[col] == 1
        return pd.Series([False] * len(plot_df), index=plot_df.index)

    any_sector_mask = _any_sector_collab_mask(plot_df)
    groups = [
        ("No taxonomy collaborator", ~any_sector_mask, "#9e9e9e"),
        ("Hospital/Clinical", sector_mask("Hospital/Clinical"), "#2a9d8f"),
        ("University/HEI", sector_mask("University/HEI"), "#457b9d"),
        ("Company (non-UK)", sector_mask("Company (non-UK)"), "#5e548e"),
        ("UK company", sector_mask("UK company"), "#d4af37"),
    ]

    plt.figure(figsize=(11, 6), dpi=300)
    for label, mask, color in groups:
        y_vals = []
        for year in year_range:
            year_mask = (plot_df["year"] == year) & mask
            vals = citations_log[year_mask].dropna()
            if len(vals) < min_papers_per_point:
                y_vals.append(np.nan)
            else:
                y_vals.append(float(vals.median()))
        plt.plot(year_range, y_vals, marker="o", linewidth=2, color=color, label=label)

    plt.title(f"Yearly median citation impact by taxonomy group ({start_year}-{end_year})")
    plt.xlabel("Year")
    plt.ylabel("Median log10(citations + 1)")
    plt.xticks(year_range, rotation=45)
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()


def _pick_journal_name(row: pd.Series) -> str:
    for col in ["journal_title_raw", "journal.title", "source_title.title", "publisher"]:
        if col in row.index and not is_empty_value(row[col]):
            return normalize_name(row[col])
    return ""


def build_journal_company_table(
    df: pd.DataFrame,
    top_n: int = 20,
    min_papers: int = 25,
) -> pd.DataFrame:
    """Build journal-level paper volume and company collaboration share table."""
    journal_df = df.copy()
    journal_df["any_sector_collab_flag"] = _any_sector_collab_mask(journal_df).astype(int)
    journal_df["journal_name"] = journal_df.apply(_pick_journal_name, axis=1)
    journal_df = journal_df[journal_df["journal_name"] != ""].copy()
    if journal_df.empty:
        return pd.DataFrame()

    grouped = (
        journal_df.groupby("journal_name", as_index=False)
        .agg(
            papers=("journal_name", "size"),
            company_papers=("company_flag", "sum"),
            uk_company_papers=("uk_company_flag", "sum"),
            any_sector_papers=("any_sector_collab_flag", "sum"),
        )
        .sort_values("papers", ascending=False)
    )

    grouped = grouped[grouped["papers"] >= min_papers].copy()
    if grouped.empty:
        return grouped

    grouped["company_share"] = grouped["company_papers"] / grouped["papers"]
    grouped["uk_company_share"] = grouped["uk_company_papers"] / grouped["papers"]
    grouped["any_sector_share"] = grouped["any_sector_papers"] / grouped["papers"]

    grouped = grouped.sort_values("papers", ascending=False).head(top_n).reset_index(drop=True)
    return grouped


def plot_top_journal_company_share(journal_df: pd.DataFrame, top_n: int = 15) -> None:
    """Plot company and UK-company shares for top journals."""
    if journal_df.empty:
        print("No journal data available for plotting.")
        return

    plot_df = journal_df.sort_values("papers", ascending=False).head(top_n).copy()
    plot_df = plot_df.sort_values("company_share", ascending=True)
    y = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    ax.barh(
        y,
        plot_df["company_share"] * 100,
        color="#2a9d8f",
        edgecolor="black",
        linewidth=0.4,
        alpha=0.85,
        label="Any company share",
    )
    ax.scatter(
        plot_df["uk_company_share"] * 100,
        y,
        color="#345995",
        s=DEFAULT_DOT_MARKER_AREA,
        zorder=3,
        label="UK company share",
    )

    for yi, papers in zip(y, plot_df["papers"]):
        ax.text(100.5, yi, f"n={int(papers)}", va="center", ha="left", fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["journal_name"])
    ax.set_xlim(0, 115)
    ax.set_xlabel("Share of papers (%)")
    ax.set_ylabel("Journal")
    ax.set_title("Company collaboration share among top journals")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(True, axis="x", alpha=0.25)
    plt.tight_layout()
    plt.show()


def plot_for_company_share_scatter(
    for_df: pd.DataFrame,
    min_papers: int = 30,
    max_labels: int = 10,
    save_path: str | Path | None = None,
    save_formats: tuple[str, ...] = ("png", "pdf", "svg"),
    png_dpi: int = 300,
) -> dict[str, Path]:
    """Plot FoR paper volume vs all-company share with matching color encoding and save outputs."""
    if for_df.empty:
        print("No FoR table available for plotting.")
        return {}

    plot_df = for_df[for_df["papers"] >= min_papers].copy()
    if plot_df.empty:
        print(f"No FoR categories meet min_papers={min_papers}.")
        return {}

    x = plot_df["papers"].astype(float)
    y = 100 * plot_df["company_share"].astype(float)
    c = 100 * plot_df["company_share"].astype(float)
    x_log = np.log10(x)
    span = float(x_log.max() - x_log.min()) if len(x_log) else 0.0
    if span <= 0:
        size = np.full(len(plot_df), 320.0)
    else:
        # Stretch the upper tail so high-volume FoR points read as clearly larger markers.
        x_norm = (x_log - x_log.min()) / span
        size = 100 + 1650 * np.power(x_norm, 2.1)

    fig, ax = plt.subplots(figsize=(10, 5.65), dpi=300)
    sc = ax.scatter(
        x,
        y,
        c=c,
        s=size,
        cmap="viridis",
        edgecolor="black",
        linewidth=0.4,
        alpha=0.9,
    )

    rank_for_labels = plot_df.assign(label_score=plot_df["company_share"] * np.log1p(plot_df["papers"]))
    rank_for_labels = rank_for_labels.sort_values("label_score", ascending=False)
    # For small FoR sets, label every point to avoid silently dropping one category.
    if len(rank_for_labels) <= 15:
        label_df = rank_for_labels
    else:
        label_df = rank_for_labels.head(max_labels)
    x_edge_cut = 10 ** (float(x_log.min()) + 0.82 * span) if span > 0 else float(x.iloc[0])
    edge_rank = 0
    label_text_overrides: dict[str, str] = {
        "Biomedical and Clinical Sciences": "Biomedical and\nClinical Sciences",
        "Agricultural, Veterinary and Food Sciences": "Agricultural,\nVeterinary and\nFood Sciences",
    }
    bbox_default = {"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.85}
    manual_label_layout: dict[str, dict[str, Any]] = {
        # User-requested manual placements.
        "Biological Sciences": {"xytext": (0, -16), "ha": "center", "va": "top"},
    }
    for _, row in label_df.iterrows():
        x_val = float(row["papers"])
        y_val = float(100 * row["company_share"])
        label = str(row["l2_for_name"])
        display_label = label_text_overrides.get(label, label)

        if label == "Health Sciences":
            ax.annotate(
                display_label,
                (x_val, y_val),
                textcoords="data",
                xytext=(x_val, 2.5),
                ha="center",
                va="center",
                fontsize=9,
                bbox=bbox_default,
                arrowprops={
                    "arrowstyle": "->",
                    "lw": 0.9,
                    "color": "#303030",
                    "connectionstyle": "arc3,rad=0.18",
                    "shrinkA": 0,
                    "shrinkB": 3,
                },
            )
            continue

        if label == "Biomedical and Clinical Sciences":
            ax.annotate(
                display_label,
                (x_val, y_val),
                textcoords="data",
                xytext=(x_val * 0.78, 10.0),
                ha="center",
                va="center",
                fontsize=9,
                bbox=bbox_default,
                arrowprops={
                    "arrowstyle": "->",
                    "lw": 0.9,
                    "color": "#303030",
                    "connectionstyle": "arc3,rad=-0.2",
                    "shrinkA": 0,
                    "shrinkB": 3,
                },
            )
            continue

        if label in manual_label_layout:
            layout = manual_label_layout[label]
            annotate_kwargs: dict[str, Any] = {
                "text": display_label,
                "xy": (x_val, y_val),
                "textcoords": "offset points",
                "xytext": layout["xytext"],
                "ha": layout["ha"],
                "va": layout["va"],
                "fontsize": 9,
            }
            if x_val >= x_edge_cut:
                # Keep curved leader arrows for right-side points.
                annotate_kwargs["bbox"] = bbox_default
                annotate_kwargs["arrowprops"] = {
                    "arrowstyle": "->",
                    "lw": 0.9,
                    "color": "#303030",
                    "connectionstyle": "arc3,rad=0.22",
                    "shrinkA": 0,
                    "shrinkB": 3,
                }
            ax.annotate(
                annotate_kwargs.pop("text"),
                annotate_kwargs.pop("xy"),
                **annotate_kwargs,
            )
            continue
        if x_val >= x_edge_cut:
            y_offset_cycle = [18, -18, 30, -30, 12, -12]
            y_offset = y_offset_cycle[edge_rank % len(y_offset_cycle)]
            rad = 0.28 if edge_rank % 2 == 0 else -0.28
            edge_rank += 1
            ax.annotate(
                display_label,
                (x_val, y_val),
                textcoords="offset points",
                xytext=(-105, y_offset),
                ha="right",
                va="center",
                fontsize=9,
                bbox=bbox_default,
                arrowprops={
                    "arrowstyle": "->",
                    "lw": 0.9,
                    "color": "#303030",
                    "connectionstyle": f"arc3,rad={rad}",
                    "shrinkA": 0,
                    "shrinkB": 3,
                },
            )
        else:
            ax.annotate(
                display_label,
                (x_val, y_val),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=9,
            )

    ax.set_xscale("log")
    ax.margins(x=0.08)
    y_low, y_high = ax.get_ylim()
    ax.set_ylim(y_low, y_high * 1.05)
    ax.set_xlabel("Paper count (log scale)")
    ax.set_ylabel("Company collaborator share (%)")
    ax.set_title("FoR level-2 volume vs company collaborator share")
    ax.grid(True, which="both", axis="both", alpha=0.2)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Company collaborator share (%)")
    plt.tight_layout()

    if save_path is not None:
        base_out = Path(save_path).with_suffix("")
    else:
        base_out = _figure_dir() / "for_level2_volume_vs_company_collaborator_share"
    base_out.parent.mkdir(parents=True, exist_ok=True)

    saved_paths: dict[str, Path] = {}
    for fmt in save_formats:
        clean_fmt = fmt.lower().lstrip(".")
        out_path = base_out.with_suffix(f".{clean_fmt}")
        save_kwargs: dict[str, Any] = {"bbox_inches": "tight"}
        if clean_fmt == "png":
            save_kwargs["dpi"] = png_dpi
        fig.savefig(out_path, **save_kwargs)
        saved_paths[clean_fmt] = out_path

    plt.show()
    return saved_paths


def build_yearly_metrics_table(
    df: pd.DataFrame,
    start_year: int = 2014,
    end_year: int = 2025,
) -> pd.DataFrame:
    """Build annual counts and shares for taxonomy sectors."""
    year_range = _year_index(start_year, end_year)
    out_rows: list[dict[str, Any]] = []
    sector_labels = [label for label in NON_ACADEMIC_SECTOR_LABELS if label != "Other/Unknown"]

    for year in year_range:
        sub = df[df["year"] == year]
        total = int(len(sub))
        any_sector_mask = _any_sector_collab_mask(sub)
        any_sector_papers = int(any_sector_mask.sum())
        row: dict[str, Any] = {
            "year": year,
            "papers_total": total,
            "papers_any_sector": any_sector_papers,
            "share_any_sector_pct": (100 * any_sector_papers / total) if total else 0.0,
        }
        for label in sector_labels:
            col = sector_flag_col(label)
            slug = _sector_slug(label)
            count = int((sub[col] == 1).sum()) if col in sub.columns else 0
            row[f"papers_{slug}"] = count
            row[f"share_{slug}_pct"] = (100 * count / total) if total else 0.0
        out_rows.append(row)

    out = pd.DataFrame(out_rows)
    if "share_company_non_uk_pct" in out.columns and "share_uk_company_pct" in out.columns:
        out["share_company_any_pct"] = out["share_company_non_uk_pct"] + out["share_uk_company_pct"]
        out["company_share_yoy_delta_pp"] = out["share_company_any_pct"].diff()
    else:
        out["company_share_yoy_delta_pp"] = np.nan
    return out


def plot_yearly_metrics_dashboard(yearly_df: pd.DataFrame) -> None:
    """Two-panel annual dashboard: taxonomy sector volumes and shares."""
    if yearly_df.empty:
        print("No yearly metrics available for plotting.")
        return

    years = yearly_df["year"].astype(int).tolist()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), dpi=300)
    color_map = {
        "Hospital/Clinical": "#2A9D8F",
        "University/HEI": "#457B9D",
        "Government/Public": "#8D99AE",
        "Research institute/Centre": "#6A994E",
        "Nonprofit/Charity": "#A1C181",
        "Company (non-UK)": "#5E548E",
        "UK company": "#D4AF37",
    }
    axes[0].plot(years, yearly_df["papers_total"], marker="o", linewidth=2, color="#6c757d", label="Total papers")
    for label in color_map:
        slug = _sector_slug(label)
        col = f"papers_{slug}"
        if col in yearly_df.columns:
            axes[0].plot(years, yearly_df[col], marker="o", linewidth=2, color=color_map[label], label=label)
    axes[0].set_title("Annual paper volumes by taxonomy sector")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Paper count")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))

    for label in color_map:
        slug = _sector_slug(label)
        col = f"share_{slug}_pct"
        if col in yearly_df.columns:
            axes[1].plot(years, yearly_df[col], marker="o", linewidth=2, color=color_map[label], label=label)
    axes[1].set_title("Annual taxonomy sector shares")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Share of papers (%)")
    axes[1].set_ylim(0, 100)
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))

    plt.tight_layout()
    plt.show()


def plot_company_geography_mix_over_time(
    df: pd.DataFrame,
    start_year: int = 2014,
    end_year: int = 2025,
) -> None:
    """Plot UK/non-UK composition among company-collaboration papers over time."""
    year_range = _year_index(start_year, end_year)

    uk_only = []
    non_uk_only = []
    mixed = []
    unknown = []

    for year in year_range:
        sub = df[(df["year"] == year) & (df["company_flag"] == 1)]
        total = len(sub)
        if total == 0:
            uk_only.append(0.0)
            non_uk_only.append(0.0)
            mixed.append(0.0)
            unknown.append(0.0)
            continue

        m_uk_only = ((sub["uk_company_flag"] == 1) & (sub["non_uk_company_flag"] == 0)).sum()
        m_non_uk_only = ((sub["uk_company_flag"] == 0) & (sub["non_uk_company_flag"] == 1)).sum()
        m_mixed = ((sub["uk_company_flag"] == 1) & (sub["non_uk_company_flag"] == 1)).sum()
        m_unknown = ((sub["uk_company_flag"] == 0) & (sub["non_uk_company_flag"] == 0)).sum()

        uk_only.append(100 * m_uk_only / total)
        non_uk_only.append(100 * m_non_uk_only / total)
        mixed.append(100 * m_mixed / total)
        unknown.append(100 * m_unknown / total)

    plt.figure(figsize=(11, 6), dpi=300)
    plt.stackplot(
        year_range,
        uk_only,
        non_uk_only,
        mixed,
        unknown,
        labels=["UK only", "Non-UK only", "UK + non-UK", "Unknown geo"],
        colors=["#345995", "#5e548e", "#2a9d8f", "#8d99ae"],
        alpha=0.92,
    )
    plt.title(f"Geographic composition of company-collaboration papers ({start_year}-{end_year})")
    plt.xlabel("Year")
    plt.ylabel("Share within company papers (%)")
    plt.ylim(0, 100)
    plt.xticks(year_range, rotation=45)
    plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    plt.tight_layout()
    plt.show()


def build_company_collaborator_churn_table(
    df: pd.DataFrame,
    start_year: int = 2014,
    end_year: int = 2025,
) -> pd.DataFrame:
    """Count new vs returning company collaborators by year."""
    year_range = _year_index(start_year, end_year)

    first_year: dict[str, int] = {}
    yearly_sets: dict[int, set[str]] = {year: set() for year in year_range}

    for _, row in df.iterrows():
        year = row.get("year")
        if pd.isna(year):
            continue
        year_int = int(year)
        if year_int not in yearly_sets:
            continue
        orgs = {org_key(x) for x in row["company_institutions_norm"] if org_key(x)}
        for org in orgs:
            yearly_sets[year_int].add(org)
            if org not in first_year or year_int < first_year[org]:
                first_year[org] = year_int

    rows: list[dict[str, Any]] = []
    for year in year_range:
        active = yearly_sets[year]
        new = {org for org in active if first_year.get(org) == year}
        returning = active - new
        rows.append(
            {
                "year": year,
                "active_company_collaborators": len(active),
                "new_company_collaborators": len(new),
                "returning_company_collaborators": len(returning),
                "new_share_pct": (100 * len(new) / len(active)) if active else 0.0,
            }
        )

    return pd.DataFrame(rows)


def plot_new_vs_returning_company_collaborators(churn_df: pd.DataFrame) -> None:
    """Plot annual counts of new vs returning company collaborators."""
    if churn_df.empty:
        print("No churn table available for plotting.")
        return

    years = churn_df["year"].astype(int).tolist()
    new_vals = churn_df["new_company_collaborators"].values
    returning_vals = churn_df["returning_company_collaborators"].values

    plt.figure(figsize=(11, 6), dpi=300)
    plt.bar(years, returning_vals, color="#2a9d8f", alpha=0.82, label="Returning")
    plt.bar(years, new_vals, bottom=returning_vals, color="#f4a261", alpha=0.92, label="New")
    plt.title("Annual new vs returning company collaborators")
    plt.xlabel("Year")
    plt.ylabel("Unique company collaborators")
    plt.xticks(years, rotation=45)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()


def build_top_company_year_matrix(
    df: pd.DataFrame,
    top_n: int = 15,
    start_year: int = 2014,
    end_year: int = 2025,
) -> pd.DataFrame:
    """Build matrix of paper counts by top company collaborator and year."""
    year_range = _year_index(start_year, end_year)

    total_counts: Counter = Counter()
    for items in df["company_institutions_norm"]:
        seen = {org_key(x): normalize_name(x) for x in items if org_key(x)}
        for key, name in seen.items():
            total_counts[(key, name)] += 1

    top_org_pairs = [pair for pair, _ in total_counts.most_common(top_n)]
    if not top_org_pairs:
        return pd.DataFrame()

    rows = []
    for key, name in top_org_pairs:
        year_counts = []
        for year in year_range:
            sub = df[df["year"] == year]
            c = 0
            for items in sub["company_institutions_norm"]:
                if key in {org_key(x) for x in items if org_key(x)}:
                    c += 1
            year_counts.append(c)
        rows.append([name] + year_counts)

    cols = ["org"] + [str(y) for y in year_range]
    return pd.DataFrame(rows, columns=cols)


def build_top_company_year_authorship_matrix(
    df: pd.DataFrame,
    top_n: int = 15,
    start_year: int = 2014,
    end_year: int = 2025,
) -> pd.DataFrame:
    """Build matrix of authorship contributions by top company collaborator and year."""
    year_range = _year_index(start_year, end_year)

    if "company_authorship_mentions" in df.columns:
        mention_series = df["company_authorship_mentions"]
    elif "affiliation_names" in df.columns and "company_institutions_norm" in df.columns:
        mention_series = df.apply(
            lambda row: build_row_authorship_mentions(
                row["affiliation_names"],
                row["company_institutions_norm"],
            ),
            axis=1,
        )
    else:
        mention_series = pd.Series([[] for _ in range(len(df))], index=df.index)

    labels: dict[str, str] = {}
    total_counts: Counter = Counter()
    for items in mention_series:
        for name in items:
            key = org_key(name)
            if not key:
                continue
            total_counts[key] += 1
            labels.setdefault(key, normalize_name(name))

    top_keys = [key for key, _ in total_counts.most_common(top_n)]
    if not top_keys:
        return pd.DataFrame()

    rows = []
    for key in top_keys:
        year_counts = []
        for year in year_range:
            year_mask = df["year"] == year
            count = 0
            for items in mention_series[year_mask]:
                for name in items:
                    if org_key(name) == key:
                        count += 1
            year_counts.append(count)
        rows.append([labels.get(key, key)] + year_counts)

    cols = ["org"] + [str(y) for y in year_range]
    return pd.DataFrame(rows, columns=cols)


def plot_top_company_heatmap(company_year_df: pd.DataFrame) -> None:
    """Plot heatmap of yearly paper counts for top company collaborators."""
    if company_year_df.empty:
        print("No top-company-by-year matrix available for plotting.")
        return

    plot_df = company_year_df.copy()
    org_labels = plot_df["org"].tolist()
    values = plot_df.drop(columns=["org"]).to_numpy(dtype=float)
    years = plot_df.columns[1:].tolist()

    fig, ax = plt.subplots(figsize=(11.5, max(5.5, 0.45 * len(org_labels))), dpi=300)
    im = ax.imshow(values, aspect="auto", cmap="Spectral_r")
    ax.set_title("Top company collaborators over time (paper counts)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Company")
    ax.set_xticks(np.arange(len(years)))
    ax.set_xticklabels(years, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(org_labels)))
    ax.set_yticklabels(org_labels)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if v <= 0:
                continue
            text_color = "white" if v >= np.nanpercentile(values, 70) else "black"
            ax.text(j, i, f"{int(v)}", ha="center", va="center", fontsize=8, color=text_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Paper count")
    plt.tight_layout()
    plt.show()


def _iter_for_texts(item: Any) -> Iterable[str]:
    if isinstance(item, dict):
        for key in ["name", "display_name", "label", "code"]:
            value = item.get(key)
            if not is_empty_value(value):
                yield normalize_name(value)
        return

    if isinstance(item, (list, tuple, set)):
        for sub in item:
            yield from _iter_for_texts(sub)
        return

    s = normalize_name(item)
    if s:
        yield s


def _parse_l2_name(text: str) -> tuple[str | None, str | None]:
    match = re.match(r"^(\d{2})(?!\d)\s*(.*)$", text)
    if not match:
        return None, None

    code = match.group(1)
    name = match.group(2).strip(" -:\t")
    if not name:
        name = text.strip()
    return code, name


def extract_level2_info(items: list[Any]) -> tuple[list[str], dict[str, str]]:
    codes: set[str] = set()
    local_names: dict[str, str] = {}

    for item in items:
        for text in _iter_for_texts(item):
            for m4 in re.findall(r"\b(\d{4})\b", text):
                codes.add(m4[:2])

            code2, name2 = _parse_l2_name(text)
            if code2:
                codes.add(code2)
                if name2:
                    local_names.setdefault(code2, name2)

    return sorted(codes), local_names


def add_for_level2_columns(df_in: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    df = df_in.copy()
    if "category_for_2020" not in df.columns:
        return df, {}

    df["category_for_2020_list"] = df["category_for_2020"].apply(parse_list)
    parsed_lvl2 = df["category_for_2020_list"].apply(extract_level2_info)
    df["category_for_2020_lvl2"] = parsed_lvl2.map(lambda x: x[0])

    name_votes: dict[str, Counter] = {}
    for _, local_names in parsed_lvl2:
        for code, name in local_names.items():
            name_votes.setdefault(code, Counter())[name] += 1

    name_map = {
        code: votes.most_common(1)[0][0]
        for code, votes in name_votes.items()
        if len(votes) > 0
    }

    return df, name_map


def build_for_share_table(df: pd.DataFrame, name_map: dict[str, str]) -> pd.DataFrame:
    if "category_for_2020_lvl2" not in df.columns:
        return pd.DataFrame()

    total_counts: Counter = Counter()
    company_any_counts: Counter = Counter()
    sector_counts: dict[str, Counter] = {label: Counter() for label in NON_ACADEMIC_SECTOR_LABELS}

    for _, row in df.iterrows():
        codes = row["category_for_2020_lvl2"]
        active_labels = [label for label in NON_ACADEMIC_SECTOR_LABELS if int(row.get(sector_flag_col(label), 0)) == 1]
        has_company_any = int(row.get("company_flag", 0)) == 1
        if not has_company_any:
            has_company_any = (
                int(row.get(sector_flag_col("Company (non-UK)"), 0)) == 1
                or int(row.get(sector_flag_col("UK company"), 0)) == 1
            )
        for code in codes:
            total_counts[code] += 1
            if has_company_any:
                company_any_counts[code] += 1
            for label in active_labels:
                sector_counts[label][code] += 1

    rows: list[dict[str, Any]] = []
    for code, total in total_counts.items():
        non_company_share = 0.0
        for label in NON_ACADEMIC_SECTOR_LABELS:
            if label in {"Company (non-UK)", "UK company"}:
                continue
            non_company_share += sector_counts[label].get(code, 0) / total if total else 0.0
        share_company_non_uk = sector_counts["Company (non-UK)"].get(code, 0) / total if total else 0.0
        share_uk_company = sector_counts["UK company"].get(code, 0) / total if total else 0.0
        share_company_any = company_any_counts.get(code, 0) / total if total else 0.0

        rows.append(
            {
                "code": code,
                "l2_for_name": name_map.get(code, code),
                "papers": int(total),
                "non_share": non_company_share,
                "company_share": share_company_any,
                "non_uk_company_share": share_company_non_uk,
                "uk_company_share": share_uk_company,
            }
        )

    out = pd.DataFrame(rows)
    for label in NON_ACADEMIC_SECTOR_LABELS:
        slug = _sector_slug(label)
        out[f"share_{slug}"] = out["code"].map(
            lambda code: sector_counts[label].get(code, 0) / total_counts.get(code, 1)
        )

    return out


def plot_for_share_table(for_df: pd.DataFrame, top_n: int = 15) -> None:
    if for_df.empty:
        print("No category_for_2020 level-2 codes found to plot.")
        return

    plot_df = for_df.sort_values("papers", ascending=False).head(top_n)
    plot_df = plot_df.sort_values("company_share", ascending=True)

    x = np.arange(len(plot_df))
    width = 0.2

    plt.figure(figsize=(12, 7), dpi=150)
    plt.barh(
        x - 1.5 * width,
        plot_df["share_hospital_clinical"] * 100,
        height=width,
        color="#2A9D8F",
        label="Hospital/Clinical share",
    )
    plt.barh(
        x - 0.5 * width,
        plot_df["share_university_hei"] * 100,
        height=width,
        color="#457B9D",
        label="University/HEI share",
    )
    plt.barh(
        x + 0.5 * width,
        plot_df["share_company_non_uk"] * 100,
        height=width,
        color="#5E548E",
        label="Company (non-UK) share",
    )
    plt.barh(
        x + 1.5 * width,
        plot_df["share_uk_company"] * 100,
        height=width,
        color="#D4AF37",
        label="UK company share",
    )
    plt.yticks(x, plot_df["l2_for_name"])
    plt.xlabel("Percent of papers")
    plt.ylabel("Level-2 FoR name")
    plt.title("Taxonomy sector shares by category_for_2020 level-2 FoR")
    plt.xlim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.show()


def build_uk_company_for_table(df: pd.DataFrame, name_map: dict[str, str]) -> pd.DataFrame:
    if "category_for_2020_lvl2" not in df.columns:
        return pd.DataFrame()

    total_counts: Counter = Counter()
    uk_counts: Counter = Counter()

    for codes, uk_flag in zip(df["category_for_2020_lvl2"], df["uk_company_flag"]):
        for code in codes:
            total_counts[code] += 1
            if uk_flag:
                uk_counts[code] += 1

    rows: list[dict[str, Any]] = []
    for code, total in total_counts.items():
        rows.append(
            {
                "code": code,
                "l2_for_name": name_map.get(code, code),
                "papers": int(total),
                "uk_company_share": uk_counts.get(code, 0) / total if total else 0,
            }
        )

    return pd.DataFrame(rows)


def plot_uk_company_for_table(uk_for_df: pd.DataFrame, top_n: int = 15) -> None:
    if uk_for_df.empty:
        print("No category_for_2020 level-2 codes found to plot.")
        return

    plot_df = uk_for_df.sort_values("papers", ascending=False).head(top_n)
    plot_df = plot_df.sort_values("uk_company_share", ascending=True)

    plt.figure(figsize=(10, 6), dpi=150)
    plt.barh(
        plot_df["l2_for_name"],
        plot_df["uk_company_share"] * 100,
        color="#345995",
        edgecolor="black",
        linewidth=0.3,
    )
    plt.title("UK company share by category_for_2020 level-2 FoR")
    plt.xlabel("Percent of papers")
    plt.ylabel("Level-2 FoR name")
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.show()


def extract_affiliations(authors_value: Any) -> list[dict[str, Any]]:
    parsed = safe_parse(authors_value)
    if parsed is None:
        return []
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return []

    affiliations: list[dict[str, Any]] = []
    for author in parsed:
        if not isinstance(author, dict):
            continue

        affs = author.get("affiliations") or []
        if isinstance(affs, dict):
            affs = [affs]
        for aff in affs:
            if not isinstance(aff, dict):
                continue
            name = aff.get("name") or aff.get("raw_affiliation")
            country = aff.get("country") or aff.get("country_name") or aff.get("country_code")
            if name or country:
                affiliations.append({"name": name, "country": country})

        insts = author.get("institutions") or []
        if isinstance(insts, dict):
            insts = [insts]
        for inst in insts:
            if not isinstance(inst, dict):
                continue
            name = inst.get("display_name") or inst.get("name")
            country = inst.get("country_code") or inst.get("country")
            if name or country:
                affiliations.append({"name": name, "country": country})

    return affiliations


def _get_authors_payload(row: pd.Series) -> Any:
    for col in ["authors", "authorships"]:
        if col in row.index and not is_empty_value(row[col]):
            return row[col]
    return None


def _extract_countries_for_orgset(row: pd.Series, org_col: str) -> list[Any]:
    target = {org_key(x) for x in row[org_col] if org_key(x)}
    if not target:
        return []

    countries: list[Any] = []
    payload = _get_authors_payload(row)
    for aff in extract_affiliations(payload):
        name = org_key(aff.get("name"))
        country = aff.get("country")
        if not country:
            continue
        if name and name in target:
            countries.append(country)

    return countries


def add_country_columns(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()

    df["non_academic_countries"] = df.apply(
        lambda row: _extract_countries_for_orgset(row, "non_academic_institutions_norm"),
        axis=1,
    )
    df["company_countries"] = df.apply(
        lambda row: _extract_countries_for_orgset(row, "company_institutions_norm"),
        axis=1,
    )

    if df["non_academic_countries"].map(len).sum() == 0 and "research_org_country_names" in df.columns:
        df["non_academic_countries"] = df.apply(
            lambda row: parse_list(row["research_org_country_names"])
            if row["non_academic_institutions_norm"]
            else [],
            axis=1,
        )

    if df["company_countries"].map(len).sum() == 0 and "research_org_country_names" in df.columns:
        df["company_countries"] = df.apply(
            lambda row: parse_list(row["research_org_country_names"])
            if row["company_institutions_norm"]
            else [],
            axis=1,
        )

    return df


def normalize_country(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        if value.get("name"):
            return str(value["name"]).strip()
        if value.get("id"):
            return str(value["id"]).strip()
    return str(value).strip()


def build_country_count_table(country_series: pd.Series) -> pd.DataFrame:
    countries: list[str] = []
    for items in country_series:
        for country in items:
            name = normalize_country(country)
            if name:
                countries.append(name)

    counts = Counter(countries)
    out = pd.DataFrame({"country": list(counts.keys()), "n_pubs": list(counts.values())})

    name_map = {
        "USA": "United States of America",
        "US": "United States of America",
        "UK": "United Kingdom",
        "Russia": "Russian Federation",
        "United States": "United States of America",
    }

    if out.empty:
        out["country_norm"] = []
    else:
        out["country_norm"] = out["country"].replace(name_map)

    return out


def load_world_geodata() -> Any:
    import geopandas as gpd

    world_url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"

    try:
        return gpd.read_file(world_url)
    except Exception:
        pass

    try:
        return gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    except Exception:
        return None


def plot_country_map(country_df: pd.DataFrame, title: str, cmap_name: str = "cividis", world: Any = None) -> None:
    if country_df.empty:
        print(f"No data to map: {title}")
        return

    world_df = world if world is not None else load_world_geodata()
    if world_df is None:
        print("World map data unavailable in this environment; skipped choropleth.")
        return

    world_df = world_df[world_df["ADMIN"] != "Antarctica"].copy()
    world_counts = world_df.merge(
        country_df,
        left_on="ADMIN",
        right_on="country_norm",
        how="left",
    )
    world_counts["n_pubs"] = world_counts["n_pubs"].fillna(0)

    nonzero = world_counts.loc[world_counts["n_pubs"] > 0, "n_pubs"]
    if nonzero.empty:
        print(f"No non-zero country counts to map: {title}")
        return

    cmap = plt.colormaps.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=nonzero.min(), vmax=nonzero.max())
    zero_color = "#E5E5E5"

    colors = [zero_color if n == 0 else mcolors.to_hex(cmap(norm(n))) for n in world_counts["n_pubs"]]

    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(16, 7.5), dpi=300)
    gs = GridSpec(1, 2, figure=fig, width_ratios=[0.15, 4.5], wspace=0.05)
    ax_cbar = fig.add_subplot(gs[0, 0])
    ax_map = fig.add_subplot(gs[0, 1])

    world_counts.plot(ax=ax_map, color=colors, linewidth=0.3, edgecolor="black")

    ax_map.set_title(title, pad=12)
    ax_map.set_aspect("auto")
    ax_map.margins(0)
    ax_map.set_anchor("W")
    ax_map.set_axis_off()

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    cbar = fig.colorbar(sm, cax=ax_cbar, orientation="vertical")
    cbar.set_label("Paper count")
    cbar.ax.yaxis.set_ticks_position("left")
    cbar.ax.yaxis.set_label_position("left")

    plt.show()


def _first_author_info_from_parsed(parsed: Any) -> tuple[str | None, int]:
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list) or not parsed:
        return None, 0

    authors = [a for a in parsed if isinstance(a, dict)]
    if not authors:
        return None, 0

    first = authors[0]
    surname = (
        first.get("last_name")
        or first.get("surname")
        or first.get("family")
        or first.get("family_name")
        or first.get("last")
    )

    if not surname:
        name = first.get("name") or first.get("display_name") or first.get("full_name")
        if isinstance(name, str) and name.strip():
            surname = re.split(r"\s+", name.strip())[-1]

    if surname:
        return str(surname).strip(), len(authors)

    return None, len(authors)


def first_author_label(row: pd.Series) -> str:
    for col in ["authors", "researchers"]:
        if col not in row.index or is_empty_value(row[col]):
            continue

        parsed = safe_parse(row[col])
        surname, n_authors = _first_author_info_from_parsed(parsed)
        if surname:
            return f"{surname} et al." if n_authors > 1 else surname

        raw = str(row[col])
        match = re.search(r"['\"]last_name['\"]\s*:\s*['\"]([^'\"]+)['\"]", raw)
        if match:
            surname = match.group(1).strip()
            n = row.get("authors_count")
            try:
                n_authors = int(float(n)) if pd.notna(n) else None
            except Exception:
                n_authors = None
            if n_authors is None:
                n_authors = max(1, len(re.findall(r"['\"]first_name['\"]\s*:", raw)))
            return f"{surname} et al." if n_authors > 1 else surname

    return "Unknown"


def format_year(row: pd.Series) -> str:
    if "year" in row.index and pd.notna(row["year"]):
        try:
            return str(int(float(row["year"])))
        except Exception:
            pass

    if "date" in row.index and pd.notna(row["date"]):
        d = row["date"]
        if isinstance(d, str) and len(d) >= 4 and d[:4].isdigit():
            return d[:4]
        try:
            return str(pd.to_datetime(d).year)
        except Exception:
            pass

    return "n.d."


def format_journal(row: pd.Series) -> str:
    for col in ["journal_title_raw", "journal.title", "source_title.title", "publisher"]:
        if col in row.index and pd.notna(row[col]) and str(row[col]).strip():
            return str(row[col]).strip()
    return "Unknown journal"


def format_doi(row: pd.Series) -> str:
    if "doi" not in row.index or is_empty_value(row["doi"]):
        return "DOI not available"

    doi = str(row["doi"]).strip()
    doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
    return f"https://doi.org/{doi}"


def top_n_by_flag(
    df: pd.DataFrame,
    flag_col: str,
    n: int = 5,
    citation_col: str = "times_cited",
) -> pd.DataFrame:
    if citation_col not in df.columns:
        raise ValueError(f"Citation column not found: {citation_col}")
    if flag_col not in df.columns:
        raise ValueError(f"Flag column not found: {flag_col}")

    sub = df[df[flag_col] == 1].copy()
    sub["_citations"] = pd.to_numeric(sub[citation_col], errors="coerce").fillna(0)
    sub = sub.sort_values("_citations", ascending=False)

    dedup_keys = [k for k in ["id", "doi", "title"] if k in sub.columns]
    if dedup_keys:
        sub = sub.drop_duplicates(subset=dedup_keys, keep="first")

    return sub.head(n)


def format_reference(row: pd.Series) -> str:
    author = first_author_label(row)
    year = format_year(row)
    title = str(row.get("title") or "Untitled").strip()
    journal = format_journal(row)
    doi = format_doi(row)
    cites = int(round(float(pd.to_numeric(row.get("_citations", 0), errors="coerce") or 0)))
    return f"{author} ({year}), {title}, {journal}, {doi}, {cites}"


def build_top_cited_sections(
    df: pd.DataFrame,
    top_n: int = 5,
    citation_col: str = "times_cited",
) -> dict[str, list[str]]:
    section_specs = [
        ("Top 5 cited papers with Hospital/Clinical collaborations", sector_flag_col("Hospital/Clinical")),
        ("Top 5 cited papers with Company (non-UK) collaborations", sector_flag_col("Company (non-UK)")),
        ("Top 5 cited papers with UK company collaborations", sector_flag_col("UK company")),
    ]

    out: dict[str, list[str]] = {}
    for heading, flag in section_specs:
        if flag not in df.columns:
            out[heading] = []
            continue
        top_df = top_n_by_flag(df, flag_col=flag, n=top_n, citation_col=citation_col)
        out[heading] = [format_reference(row) for _, row in top_df.iterrows()]

    return out


def plot_publication_figure(
    df: pd.DataFrame,
    start_year: int = 2016,
    end_year: int = 2025,
    citation_col: str = "times_cited",
    top_company_n: int = 15,
    figsize: tuple[float, float] = (16, 12),
    save_path: str | Path | None = None,
    save_formats: tuple[str, ...] = ("png", "pdf", "svg"),
) -> dict[str, Path]:
    """Create a 2x2 publication figure combining core collaboration visuals."""
    def _clean_company_label(label: str) -> str:
        # Remove parenthetical qualifiers like "(Sweden)" from company labels.
        clean = re.sub(r"\s*\([^)]*\)\s*", " ", label).strip()
        # Normalize Glaxo/Glasko naming variants to GSK.
        clean = re.sub(r"(?i)\b(?:glaxo|glasko)\s*smithkline\b", "GSK", clean)
        return clean

    def _split_label_mid(label: str) -> str:
        if len(label) <= 20:
            return label
        words = label.split()
        if len(words) <= 1:
            return label
        mid = len(words) // 2
        return " ".join(words[:mid]) + "\n" + " ".join(words[mid:])

    year_range = _year_index(start_year, end_year)

    sector_color_map = {
        "University/HEI": "#457b9d",
        "Hospital/Clinical": "#2a9d8f",
        "Government/Public": "#8d99ae",
        "Research institute/Centre": "#6a994e",
        "Nonprofit/Charity": "#a1c181",
        "Company (non-UK)": "#5e548e",
        "UK company": "#d4af37",
        "Other/Unknown": "#bdbdbd",
    }

    # Panel A: cumulative papers by taxonomy sectors.
    use_sector = all(sector_flag_col(label) in df.columns for label in NON_ACADEMIC_SECTOR_LABELS)
    if use_sector:
        panel_a_spec = [
            (label, sector_flag_col(label), sector_color_map.get(label, "#bdbdbd"))
            for label in NON_ACADEMIC_SECTOR_LABELS
        ]
    else:
        panel_a_spec = [
            ("Any taxonomy collaborator", "non_academic_flag", "#d4af37"),
            ("Company (any)", "company_flag", "#2a9d8f"),
            ("UK company", "uk_company_flag", "#345995"),
            ("Non-UK company", "non_uk_company_flag", "#5e548e"),
        ]

    # Panel B: citation distributions.
    if citation_col in df.columns:
        citations = pd.to_numeric(df[citation_col], errors="coerce").fillna(0)
    else:
        citations = pd.Series(np.zeros(len(df)), index=df.index)

    def sector_mask(label: str) -> pd.Series:
        col = sector_flag_col(label)
        if col in df.columns:
            return df[col] == 1
        return pd.Series([False] * len(df), index=df.index)

    if use_sector:
        citation_groups = [
            (label, sector_mask(label), sector_color_map.get(label, "#bdbdbd"))
            for label in NON_ACADEMIC_SECTOR_LABELS
        ]
    else:
        citation_groups = [
            ("Any taxonomy collaborator", df["non_academic_flag"] == 1, "#d4af37"),
            ("Company (any)", df["company_flag"] == 1, "#2a9d8f"),
            ("UK company", df["uk_company_flag"] == 1, "#345995"),
            ("Non-UK company", df["non_uk_company_flag"] == 1, "#5e548e"),
        ]
    citation_data = []
    citation_labels = []
    citation_colors = []
    for label, mask, color in citation_groups:
        vals = np.log10(citations[mask] + 1).replace([np.inf, -np.inf], np.nan).dropna()
        if len(vals) == 0:
            continue
        citation_data.append(vals.values)
        citation_labels.append(f"{_split_label_mid(label)}\n(n={len(vals):,})")
        citation_colors.append(color)

    # Panel C: annual new vs returning company collaborators.
    churn_df = build_company_collaborator_churn_table(df, start_year=start_year, end_year=end_year)

    # Panel D: top company collaborators heatmap.
    top_company_year_auth_df = build_top_company_year_authorship_matrix(
        df,
        top_n=top_company_n,
        start_year=start_year,
        end_year=end_year,
    )

    with plt.rc_context({"font.family": "Helvetica"}):
        fig = plt.figure(figsize=figsize, dpi=300)
        gs = fig.add_gridspec(2, 2, wspace=0.28, hspace=0.3)

        # A
        ax1 = fig.add_subplot(gs[0, 0])
        for label, col, color in panel_a_spec:
            yearly = df[df[col] == 1].groupby("year").size().reindex(year_range, fill_value=0)
            ax1.plot(
                year_range,
                yearly.cumsum().values,
                marker="o",
                linewidth=2,
                color=color,
                label=label,
            )
        ax1.set_title("a.", loc="left", fontweight="bold")
        ax1.set_xlabel("")
        ax1.set_ylabel("Cumulative paper count")
        ax1.set_xticks(year_range)
        ax1.tick_params(axis="x", rotation=45)
        ax1.grid(True, linestyle="--", alpha=0.3)
        legend_a = ax1.legend(frameon=True, loc="upper left", ncol=2, fontsize=8.5)
        legend_a.get_frame().set_facecolor("white")
        legend_a.get_frame().set_edgecolor("black")

        # B
        ax2 = fig.add_subplot(gs[0, 1])
        if citation_data:
            box = ax2.boxplot(
                citation_data,
                patch_artist=True,
                labels=citation_labels,
                vert=False,
                showfliers=False,
                medianprops={"color": "black", "linewidth": 1.5},
            )
            for patch, color in zip(box["boxes"], citation_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
                patch.set_edgecolor("black")
                patch.set_linewidth(0.6)
        else:
            ax2.text(0.5, 0.5, "No citation data", ha="center", va="center")
        ax2.set_title("b.", loc="left", fontweight="bold")
        ax2.set_xlabel("log10(citations + 1)")
        ax2.grid(True, linestyle="--", alpha=0.3)

        # C
        ax3 = fig.add_subplot(gs[1, 0])
        if not churn_df.empty:
            years = churn_df["year"].astype(int).tolist()
            returning_vals = churn_df["returning_company_collaborators"].values
            new_vals = churn_df["new_company_collaborators"].values
            ax3.bar(
                years,
                returning_vals,
                color="#345995",
                alpha=0.9,
                edgecolor="black",
                linewidth=0.6,
                label="Returning",
            )
            ax3.bar(
                years,
                new_vals,
                bottom=returning_vals,
                color="#d4af37",
                alpha=0.9,
                edgecolor="black",
                linewidth=0.6,
                label="New",
            )
            ax3.set_xticks(years)
            ax3.tick_params(axis="x", rotation=45)
            legend_c = ax3.legend(frameon=True, loc="upper left")
            legend_c.get_frame().set_facecolor("white")
            legend_c.get_frame().set_edgecolor("black")
        else:
            ax3.text(0.5, 0.5, "No churn data", ha="center", va="center")
        ax3.set_title("c.", loc="left", fontweight="bold")
        ax3.set_xlabel("")
        ax3.set_ylabel("Unique company collaborators")
        ax3.grid(True, linestyle="--", alpha=0.3)

        # D
        ax4 = fig.add_subplot(gs[1, 1])
        if not top_company_year_auth_df.empty:
            matrix = top_company_year_auth_df.drop(columns=["org"]).to_numpy(dtype=float)
            org_labels_raw = top_company_year_auth_df["org"].tolist()
            org_labels = [_split_label_mid(_clean_company_label(x)) for x in org_labels_raw]
            years = top_company_year_auth_df.columns[1:].tolist()

            heatmap_cmap = plt.get_cmap("Spectral_r").copy()
            heatmap_cmap.set_bad(color="white")
            masked_matrix = np.ma.masked_where(matrix <= 0, matrix)
            im = ax4.imshow(masked_matrix, aspect="auto", cmap=heatmap_cmap)
            ax4.set_xticks(np.arange(len(years)))
            ax4.set_xticklabels(years, rotation=45, ha="right")
            ax4.set_yticks(np.arange(len(org_labels)))
            ax4.set_yticklabels(org_labels)
            ax4.set_xlabel("")
            ax4.set_ylabel("")
            ax4.set_title("d.", loc="left", fontweight="bold")
            ax4.grid(False)
            for spine in ax4.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.8)
                spine.set_color("black")

            vmax = np.nanmax(matrix) if matrix.size else 0
            threshold = 0.55 * vmax if vmax else 1
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    val = matrix[i, j]
                    if val <= 0:
                        continue
                    color = "white" if val >= threshold else "black"
                    ax4.text(j, i, f"{int(val)}", ha="center", va="center", fontsize=7, color=color)

            cbar = fig.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
            cbar.set_label("Paper-level contributions")
        else:
            ax4.text(0.5, 0.5, "No company-year matrix", ha="center", va="center")
            ax4.set_axis_off()

        ax4.set_title("d.", loc="left", fontweight="bold")

        # Use tight layout as requested; suppress known colorbar/axes compatibility warning.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="This figure includes Axes that are not compatible with tight_layout",
                category=UserWarning,
            )
            plt.tight_layout(pad=0.6, w_pad=0.5, h_pad=0.5)

        if save_path is not None:
            base_out = Path(save_path).with_suffix("")
        else:
            base_out = _figure_dir() / "non_academic_collab_publication_figure"
        base_out.parent.mkdir(parents=True, exist_ok=True)

        saved_paths: dict[str, Path] = {}
        for fmt in save_formats:
            clean_fmt = fmt.lower().lstrip(".")
            out_path = base_out.with_suffix(f".{clean_fmt}")
            fig.savefig(out_path, bbox_inches="tight")
            saved_paths[clean_fmt] = out_path
        plt.show()

    return saved_paths


def load_and_prepare_dataframe(
    data_path: Path | None = None,
    require_sector_columns: bool = False,
) -> tuple[pd.DataFrame, Path, pd.DataFrame]:
    """Convenience wrapper for loading and preparing dataframe for analysis."""
    raw_df, path = load_input_dataframe(data_path=data_path)
    if require_sector_columns:
        required_cols = [
            "non_academic_sector_records",
            "non_academic_sector_labels",
        ]
        missing = [c for c in required_cols if c not in raw_df.columns]
        if missing:
            raise ValueError(
                "Input data is missing required taxonomy columns: "
                + ", ".join(missing)
                + ". Re-run non_academic_collab_classifier.ipynb to generate a "
                "taxonomy-enabled output file."
            )
    df = prepare_analysis_dataframe(raw_df)
    df = add_flag_columns(df)
    df = add_authorship_mentions(df)
    df = add_non_academic_sector_taxonomy(df)
    return df, path, raw_df
