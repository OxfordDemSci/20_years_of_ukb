"""Reusable data and statistical utilities for analysis 05.

The analysis has one canonical source: the Showcase+ all-endpoints-wide parquet.
All person-level longitudinal metrics use Dimensions researcher IDs. Authorships with
no researcher ID remain in paper-level descriptive denominators, but receive a
paper-local key so equal names are never silently treated as the same person.

The module deliberately distinguishes four units:

* ``papers``: one row per UK Biobank publication;
* ``authorships``: one row per parsed author-paper pair;
* ``institution_credits``: fractional paper credit split over authors and each
  author's reported affiliations;
* ``country_credits``: fractional paper credit split over authors and each author's
  distinct affiliation countries.

Consequently, every paper with parsed authors contributes exactly one unit of
authorship credit. Missing affiliation geography reduces only the geolocated total;
it does not redistribute unknown credit to observed countries or institutions.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from functools import cache

import numpy as np
import pandas as pd
from scipy import sparse as sp
from scipy.sparse.csgraph import connected_components

from . import shared_name_gender as NG
from . import shared_paths as P
from .shared_for import add_for_columns
from .shared_showcase import load_showcase, parse_listcol

FIRST_YEAR = 2013
LAST_COMPLETE_YEAR = 2025
HYPERAUTHOR_THRESHOLD = 100
LEIDEN_RESOLUTION = 1.0
LEIDEN_SEED = 48652
NETWORK_LAYOUT_ITERATIONS = 100
NETWORK_BACKBONE_REPEAT_LIMIT = 80_000
AUTHOR_CONCENTRATION_THRESHOLDS = (1, 5, 10, 25, 50)

SHOWCASE_COLUMNS = [
    "id",
    "title",
    "doi",
    "date",
    "year",
    "times_cited",
    "altmetric",
    "authors_count",
    "authors",
    "researchers",
    "research_orgs",
    "research_org_countries",
    "category_for_2020",
]


@dataclass
class CoreTables:
    """Tidy analysis tables built before the co-authorship projection."""

    authorships: pd.DataFrame
    affiliations: pd.DataFrame
    institution_credits: pd.DataFrame
    country_credits: pd.DataFrame
    author_metrics: pd.DataFrame
    institution_metrics: pd.DataFrame
    country_metrics: pd.DataFrame
    gender_by_year: pd.DataFrame
    gender_by_role: pd.DataFrame
    gender_by_field: pd.DataFrame
    gender_by_country: pd.DataFrame
    gender_by_institution: pd.DataFrame
    country_by_year: pd.DataFrame
    institution_by_year: pd.DataFrame
    organization_lookup: pd.DataFrame


@dataclass
class NetworkTables:
    """Final and temporal outputs from the resolved-author network."""

    metrics_by_year: pd.DataFrame
    author_metrics: pd.DataFrame
    community_membership: pd.DataFrame
    community_summary: pd.DataFrame
    collapsed_nodes: pd.DataFrame
    collapsed_edges: pd.DataFrame
    adjacency: sp.csr_matrix
    modularity: float


def load_author_papers(
    first_year: int = FIRST_YEAR,
    last_complete_year: int = LAST_COMPLETE_YEAR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Showcase+ and return ``(source_snapshot, complete_year_papers)``.

    The first frame retains provisional/future years for audit. The second is the
    canonical analytical cohort and has parsed FOR L2/L4 lists attached.
    """
    source = load_showcase(
        path=P.SHOWCASE_PLUS,
        columns=SHOWCASE_COLUMNS,
        parse=[
            "authors",
            "researchers",
            "research_orgs",
            "research_org_countries",
            "category_for_2020",
        ],
    )
    if source["id"].isna().any() or source["id"].duplicated().any():
        raise ValueError("Showcase+ must contain one non-missing row per publication ID")

    source = source.copy()
    source["year"] = pd.to_numeric(source["year"], errors="coerce").astype("Int64")
    source["times_cited"] = pd.to_numeric(
        source["times_cited"], errors="coerce"
    ).fillna(0.0)
    source["altmetric"] = pd.to_numeric(source["altmetric"], errors="coerce").fillna(0.0)
    source["authors_count"] = pd.to_numeric(source["authors_count"], errors="coerce")
    source["publication_date"] = pd.to_datetime(source["date"], errors="coerce")

    papers = source[source["year"].between(first_year, last_complete_year)].copy()
    papers["year"] = papers["year"].astype(int)
    add_for_columns(papers, col="category_for_2020", fold_parents=True)
    return source, papers


def _normalise_text(value) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text or None


def _normalise_key(value) -> str | None:
    text = _normalise_text(value)
    if text is None:
        return None
    return re.sub(r"[^a-z0-9]+", "-", text.casefold()).strip("-") or None


def _as_bool(value) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "true", "yes", "y"}
    return False


def _first_orcid(value) -> str | None:
    values = value if isinstance(value, list) else [value]
    for item in values:
        text = _normalise_text(item)
        if text:
            return text.removeprefix("https://orcid.org/")
    return None


def _counter_value(counters, key):
    counter = counters.get(key)
    return counter.most_common(1)[0][0] if counter else None


def build_organization_lookup(papers: pd.DataFrame) -> pd.DataFrame:
    """Resolve GRID organization attributes from every available source occurrence."""
    attrs = {
        name: defaultdict(Counter)
        for name in (
            "org_name",
            "country",
            "country_code",
            "city",
            "latitude",
            "longitude",
            "org_type",
        )
    }

    def add(org_id, **values):
        org_id = _normalise_text(org_id)
        if not org_id:
            return
        for key, value in values.items():
            if key not in attrs:
                continue
            if key == "org_type" and isinstance(value, list):
                for item in value:
                    if _normalise_text(item):
                        attrs[key][org_id][_normalise_text(item)] += 1
            elif value is not None and not (isinstance(value, float) and pd.isna(value)):
                attrs[key][org_id][value] += 1

    for orgs in papers["research_orgs"]:
        for org in parse_listcol(orgs):
            if not isinstance(org, dict):
                continue
            add(
                org.get("id"),
                org_name=org.get("name"),
                country=org.get("country_name"),
                country_code=org.get("country_code"),
                city=org.get("city_name"),
                latitude=org.get("latitude"),
                longitude=org.get("longitude"),
                org_type=org.get("types"),
            )

    for authors in papers["authors"]:
        for author in parse_listcol(authors):
            if not isinstance(author, dict):
                continue
            for affiliation in parse_listcol(author.get("affiliations")):
                if not isinstance(affiliation, dict):
                    continue
                add(
                    affiliation.get("id"),
                    org_name=affiliation.get("name"),
                    country=affiliation.get("country"),
                    country_code=affiliation.get("country_code"),
                    city=affiliation.get("city"),
                )

    org_ids = sorted(set().union(*(set(values) for values in attrs.values())))
    rows = []
    for org_id in org_ids:
        rows.append({
            "org_id": org_id,
            **{name: _counter_value(values, org_id) for name, values in attrs.items()},
        })
    return pd.DataFrame(rows)


@cache
def country_to_iso(country_code, country_name) -> tuple[str | None, str | None]:
    """Return ISO-2/ISO-3, reusing the project's patent-country resolver."""
    from . import shared_patent_utils as patent

    code = _normalise_text(country_code)
    iso2 = code.upper() if code and len(code) == 2 else None
    if iso2 is None:
        iso2 = patent.to_iso2(_normalise_text(country_name))
    iso3 = patent.iso2_to_iso3(iso2) if iso2 else None
    return iso2, iso3


def infer_name_gender(first_name) -> tuple[str, str, str]:
    """Backward-compatible direct inference wrapper around the shared utility."""
    result = NG.infer_name_gender(first_name)
    return result.category, result.strict_category, result.detail


def _complete_affiliation(affiliation: dict, lookup: dict) -> dict:
    result = dict(affiliation)
    org_id = _normalise_text(result.get("id"))
    known = lookup.get(org_id, {}) if org_id else {}
    aliases = {
        "name": "org_name",
        "country": "country",
        "country_code": "country_code",
        "city": "city",
        "latitude": "latitude",
        "longitude": "longitude",
        "org_type": "org_type",
    }
    for target, source in aliases.items():
        if result.get(target) is None:
            result[target] = known.get(source)
    return result


def build_authorship_tables(
    papers: pd.DataFrame,
    organization_lookup: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build authorship, affiliation, institution-credit and country-credit tables."""
    if organization_lookup is None:
        organization_lookup = build_organization_lookup(papers)
    lookup = organization_lookup.set_index("org_id").to_dict("index")

    authorship_rows = []
    affiliation_rows = []
    for paper in papers.itertuples(index=False):
        authors = [a for a in parse_listcol(paper.authors) if isinstance(a, dict)]
        for position, author in enumerate(authors, start=1):
            researcher_id = _normalise_text(author.get("researcher_id"))
            author_id = researcher_id or f"unresolved::{paper.id}::{position:04d}"
            first_name = _normalise_text(author.get("first_name")) or ""
            last_name = _normalise_text(author.get("last_name")) or ""
            authorship_rows.append({
                "author_id": author_id,
                "researcher_id": researcher_id,
                "identity_resolved": researcher_id is not None,
                "first_name": first_name,
                "last_name": last_name,
                "full_name": f"{first_name} {last_name}".strip(),
                "orcid": _first_orcid(author.get("orcid")),
                "current_org_id": _normalise_text(author.get("current_organization_id")),
                "paper_id": paper.id,
                "title": paper.title,
                "doi": paper.doi,
                "year": int(paper.year),
                "times_cited": float(paper.times_cited),
                "altmetric": float(paper.altmetric),
                "author_position": position,
                "source_team_size": len(authors),
                "authors_count_reported": paper.authors_count,
                "is_corresponding": _as_bool(author.get("corresponding")),
                "for_l2": paper.for_l2,
                "for_l4": paper.for_l4,
            })

            raw_affiliations = [
                _complete_affiliation(aff, lookup)
                for aff in parse_listcol(author.get("affiliations"))
                if isinstance(aff, dict)
            ]
            source = "reported"
            current_org = _normalise_text(author.get("current_organization_id"))
            if not raw_affiliations and current_org in lookup:
                known = lookup[current_org]
                raw_affiliations = [{
                    "id": current_org,
                    "name": known.get("org_name"),
                    "country": known.get("country"),
                    "country_code": known.get("country_code"),
                    "city": known.get("city"),
                    "latitude": known.get("latitude"),
                    "longitude": known.get("longitude"),
                    "org_type": known.get("org_type"),
                }]
                source = "current_org_backfill"
            if not raw_affiliations:
                raw_affiliations = [{}]
                source = "missing"

            for affiliation in raw_affiliations:
                org_id = _normalise_text(affiliation.get("id"))
                org_name = _normalise_text(affiliation.get("name"))
                country = _normalise_text(affiliation.get("country"))
                country_code = _normalise_text(affiliation.get("country_code"))
                iso2, iso3 = country_to_iso(country_code, country)
                institution_id = org_id or (
                    f"name::{_normalise_key(org_name)}" if org_name else None
                )
                affiliation_rows.append({
                    "author_id": author_id,
                    "researcher_id": researcher_id,
                    "identity_resolved": researcher_id is not None,
                    "paper_id": paper.id,
                    "year": int(paper.year),
                    "times_cited": float(paper.times_cited),
                    "altmetric": float(paper.altmetric),
                    "institution_id": institution_id,
                    "org_id": org_id,
                    "org_name": org_name,
                    "country": country,
                    "country_code": country_code,
                    "iso2": iso2,
                    "iso3": iso3,
                    "city": _normalise_text(affiliation.get("city")),
                    "latitude": affiliation.get("latitude"),
                    "longitude": affiliation.get("longitude"),
                    "org_type": _normalise_text(affiliation.get("org_type")),
                    "affiliation_source": source,
                    "for_l2": paper.for_l2,
                    "for_l4": paper.for_l4,
                })

    authorships = pd.DataFrame(authorship_rows)
    if authorships.empty:
        raise ValueError("No parsed authorships were found in the Showcase+ cohort")
    authorships = (
        authorships.sort_values(["paper_id", "author_position"])
        .drop_duplicates(["paper_id", "author_id"], keep="first")
        .reset_index(drop=True)
    )
    name_inference = NG.classify_authorship_names(authorships)
    authorships = pd.concat([authorships, name_inference], axis=1)
    authorships["team_size"] = authorships.groupby("paper_id")["author_id"].transform("size")
    authorships["authorship_credit"] = 1.0 / authorships["team_size"]
    authorships["position_fraction"] = (
        authorships["author_position"] / authorships["team_size"]
    )
    authorships["authorship_role"] = np.select(
        [
            authorships["team_size"].eq(1),
            authorships["author_position"].eq(1),
            authorships["author_position"].eq(authorships["team_size"]),
        ],
        ["Single author", "First author", "Last author"],
        default="Middle author",
    )

    affiliations = pd.DataFrame(affiliation_rows)
    affiliations = affiliations[affiliations["author_id"].isin(authorships["author_id"])]
    affiliations = affiliations.drop_duplicates(
        ["paper_id", "author_id", "institution_id", "iso3"], keep="first"
    ).reset_index(drop=True)
    affiliations = affiliations.merge(
        authorships[
            [
                "paper_id",
                "author_id",
                "authorship_credit",
                "team_size",
                "name_gender",
            ]
        ],
        on=["paper_id", "author_id"],
        how="inner",
        validate="many_to_one",
    )

    institution_credits = affiliations.dropna(subset=["institution_id"]).copy()
    # Conflicting source metadata can repeat one institution under multiple
    # country records. Institutional credit is allocated over distinct
    # author-paper institutions, not over those metadata variants.
    institution_credits = institution_credits.drop_duplicates(
        ["paper_id", "author_id", "institution_id"], keep="first"
    )
    institution_credits["n_author_institutions"] = institution_credits.groupby(
        ["paper_id", "author_id"]
    )["institution_id"].transform("nunique")
    institution_credits["credit"] = (
        institution_credits["authorship_credit"]
        / institution_credits["n_author_institutions"]
    )

    country_credits = affiliations.dropna(subset=["iso3"]).copy()
    country_credits = country_credits.drop_duplicates(
        ["paper_id", "author_id", "iso3"], keep="first"
    )
    country_credits["n_author_countries"] = country_credits.groupby(
        ["paper_id", "author_id"]
    )["iso3"].transform("nunique")
    country_credits["credit"] = (
        country_credits["authorship_credit"] / country_credits["n_author_countries"]
    )
    return authorships, affiliations, institution_credits, country_credits


def h_index(citations) -> int:
    values = np.sort(np.asarray(list(citations), dtype=float))[::-1]
    return int(np.sum(values >= np.arange(1, len(values) + 1)))


def g_index(citations) -> int:
    values = np.sort(np.asarray(list(citations), dtype=float))[::-1]
    if values.size == 0:
        return 0
    eligible = np.cumsum(values) >= np.arange(1, len(values) + 1) ** 2
    return int(np.flatnonzero(eligible)[-1] + 1) if eligible.any() else 0


def _citation_indices(citations):
    values = np.sort(np.asarray(list(citations), dtype=float))[::-1]
    ranks = np.arange(1, len(values) + 1)
    h = int(np.sum(values >= ranks))
    eligible = np.cumsum(values) >= ranks**2
    g = int(np.flatnonzero(eligible)[-1] + 1) if eligible.any() else 0
    return (
        h,
        g,
        int(np.sum(values >= 10)),
        float(np.mean(values == 0)) if values.size else np.nan,
    )


def _mode(series):
    values = series.dropna().astype(str)
    if values.empty:
        return None
    counts = values.value_counts()
    return min(counts[counts.eq(counts.max())].index)


def _weighted_modal(frame, group_col, value_col, weight_col="credit") -> pd.Series:
    values = frame.dropna(subset=[value_col]).copy()
    if values.empty:
        return pd.Series(dtype="object")
    scores = (
        values.groupby([group_col, value_col], observed=True)[weight_col]
        .sum()
        .reset_index(name="_weight")
        .sort_values([group_col, "_weight", value_col], ascending=[True, False, True])
        .drop_duplicates(group_col)
    )
    return scores.set_index(group_col)[value_col]


def _fast_modal(frame, group_col, value_col) -> pd.Series:
    """Vectorized deterministic mode, avoiding one Python callback per group."""
    values = frame[[group_col, value_col]].dropna()
    if values.empty:
        return pd.Series(dtype="object")
    counts = (
        values.groupby([group_col, value_col], observed=True)
        .size()
        .reset_index(name="_count")
        .sort_values([group_col, "_count", value_col], ascending=[True, False, True])
        .drop_duplicates(group_col)
    )
    return counts.set_index(group_col)[value_col]


def _weighted_modal_list(authorships, value_col) -> pd.Series:
    exploded = authorships[["author_id", value_col, "authorship_credit"]].explode(value_col)
    exploded = exploded.rename(columns={value_col: "value", "authorship_credit": "credit"})
    return _weighted_modal(exploded, "author_id", "value")


def _modal_assignment_metrics(
    frame: pd.DataFrame,
    value_col: str,
    prefix: str,
) -> pd.DataFrame:
    """Summarize an author's modal assignment with one unit conserved per paper."""
    rows = frame[["author_id", "paper_id", value_col]].dropna(
        subset=[value_col]
    ).drop_duplicates()
    if rows.empty:
        return pd.DataFrame()
    rows["n_assignments"] = rows.groupby(["author_id", "paper_id"])[
        value_col
    ].transform("nunique")
    rows["assignment_credit"] = 1.0 / rows["n_assignments"]
    scores = (
        rows.groupby(["author_id", value_col], observed=True)
        .agg(
            assignment_papers=("paper_id", "nunique"),
            assignment_credit=("assignment_credit", "sum"),
        )
        .reset_index()
    )
    observed = rows.groupby("author_id")["paper_id"].nunique()
    distinct = rows.groupby("author_id")[value_col].nunique()
    leading = (
        scores.sort_values(
            ["author_id", "assignment_credit", "assignment_papers", value_col],
            ascending=[True, False, False, True],
        )
        .drop_duplicates("author_id")
        .set_index("author_id")
    )
    result = leading[[value_col, "assignment_papers", "assignment_credit"]].rename(
        columns={
            value_col: f"modal_{prefix}",
            "assignment_papers": f"modal_{prefix}_papers",
            "assignment_credit": f"modal_{prefix}_assignment_credit",
        }
    )
    result[f"{prefix}_observed_papers"] = observed
    result[f"n_author_{prefix}_assignments"] = distinct
    result[f"modal_{prefix}_share"] = (
        result[f"modal_{prefix}_assignment_credit"]
        / result[f"{prefix}_observed_papers"]
    )
    return result


def build_author_metrics(
    authorships: pd.DataFrame,
    institution_credits: pd.DataFrame,
    country_credits: pd.DataFrame,
    organization_lookup: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per Dimensions-disambiguated author, including UKB-specific impact."""
    stable = authorships[authorships["identity_resolved"]].copy()
    stable["is_first_author"] = stable["authorship_role"].eq("First author")
    stable["is_middle_author"] = stable["authorship_role"].eq("Middle author")
    stable["is_last_author"] = stable["authorship_role"].eq("Last author")
    stable["is_single_author"] = stable["authorship_role"].eq("Single author")
    grouped = stable.groupby("author_id", sort=False)
    out = grouped.agg(
        n_ukb_papers=("paper_id", "nunique"),
        fractional_paper_credit=("authorship_credit", "sum"),
        first_ukb_year=("year", "min"),
        last_ukb_year=("year", "max"),
        total_ukb_citations=("times_cited", "sum"),
        mean_ukb_citations=("times_cited", "mean"),
        median_ukb_citations=("times_cited", "median"),
        max_ukb_citations=("times_cited", "max"),
        citation_variance=("times_cited", "var"),
        citation_skewness=("times_cited", "skew"),
        total_altmetric=("altmetric", "sum"),
        mean_altmetric=("altmetric", "mean"),
        n_first_author=("is_first_author", "sum"),
        n_middle_author=("is_middle_author", "sum"),
        n_last_author=("is_last_author", "sum"),
        n_single_author=("is_single_author", "sum"),
        n_corresponding=("is_corresponding", "sum"),
        mean_team_size=("team_size", "mean"),
    )
    for column in [
        "full_name",
        "first_name",
        "last_name",
        "orcid",
        "name_gender",
        "name_gender_dictionary",
        "name_gender_direct",
        "name_gender_strict",
        "name_gender_detail",
        "name_gender_dictionary_detail",
        "name_gender_method",
        "name_gender_query",
        "name_gender_vote_count",
        "name_gender_vote_margin",
        "name_gender_vote_share",
        "name_gender_source_count",
        "name_gender_direct_conflict",
        "name_gender_source_predictions",
        "name_gender_gender_guesser",
        "name_gender_gender_detector",
        "name_gender_nomquamgender",
        "name_gender_names_dataset",
        "name_gender_gename",
        "name_gender_nomquamgender_probability",
        "name_gender_nomquamgender_count",
        "name_gender_names_dataset_probability",
        "name_gender_unknown_reason",
        "name_gender_identity_conflict",
    ]:
        out[column] = _fast_modal(stable, "author_id", column)
    out["current_org_id"] = _fast_modal(stable, "author_id", "current_org_id")
    citation_lists = grouped["times_cited"].agg(list)
    citation_metrics = pd.DataFrame(
        [_citation_indices(values) for values in citation_lists],
        index=citation_lists.index,
        columns=[
            "ukb_h_index",
            "ukb_g_index",
            "ukb_i10_index",
            "proportion_uncited",
        ],
    )
    out = out.join(citation_metrics)
    out["ukb_h_index_per_paper"] = out["ukb_h_index"] / out["n_ukb_papers"]
    out["active_span_years"] = out["last_ukb_year"] - out["first_ukb_year"] + 1
    out["corresponding_share"] = out["n_corresponding"] / out["n_ukb_papers"]
    out["citation_share"] = (
        out["total_ukb_citations"] / out["total_ukb_citations"].sum()
    )
    stable = stable.assign(
        fractional_citation_credit=stable["times_cited"] * stable["authorship_credit"]
    )
    out["fractional_citation_credit"] = stable.groupby("author_id")[
        "fractional_citation_credit"
    ].sum()

    stable_inst = institution_credits[institution_credits["identity_resolved"]]
    institution_assignment = _modal_assignment_metrics(
        stable_inst, "institution_id", "institution"
    ).rename(columns={"modal_institution": "modal_institution_id"})
    out = out.join(institution_assignment)
    out["modal_institution"] = out["modal_institution_id"].map(
        _fast_modal(stable_inst, "institution_id", "org_name")
    )
    stable_country = country_credits[country_credits["identity_resolved"]]
    country_assignment = _modal_assignment_metrics(
        stable_country, "iso3", "country"
    ).rename(columns={"modal_country": "modal_country_iso3"})
    out = out.join(country_assignment)
    out["modal_country"] = out["modal_country_iso3"].map(
        _fast_modal(stable_country, "iso3", "country")
    )
    if organization_lookup is not None and not organization_lookup.empty:
        organizations = organization_lookup.set_index("org_id")
        out["current_institution"] = out["current_org_id"].map(
            organizations["org_name"]
        )
        out["current_country"] = out["current_org_id"].map(
            organizations["country"]
        )
        organization_iso3 = organizations.apply(
            lambda row: country_to_iso(
                row.get("country_code"), row.get("country")
            )[1],
            axis=1,
        )
        out["current_country_iso3"] = out["current_org_id"].map(organization_iso3)
    else:
        out["current_institution"] = None
        out["current_country"] = None
        out["current_country_iso3"] = None
    has_current_home = out["current_institution"].notna()
    out["home_institution"] = out["current_institution"].where(
        has_current_home, out["modal_institution"]
    )
    out["home_country"] = out["current_country"].where(
        has_current_home, out["modal_country"]
    )
    out["home_country_iso3"] = out["current_country_iso3"].where(
        has_current_home, out["modal_country_iso3"]
    )
    out["home_assignment_source"] = np.where(
        has_current_home, "current_organization", "modal_affiliation"
    )
    for level in ["for_l2", "for_l4"]:
        assignments = stable[["author_id", "paper_id", level]].explode(level)
        out = out.join(_modal_assignment_metrics(assignments, level, level))
    out.index.name = "researcher_id"
    return (
        out.reset_index()
        .sort_values(["ukb_h_index", "n_ukb_papers", "total_ukb_citations"], ascending=False)
        .reset_index(drop=True)
    )


def _entity_paper_table(credits: pd.DataFrame, entity_col: str) -> pd.DataFrame:
    return (
        credits.groupby([entity_col, "paper_id"], observed=True)
        .agg(
            year=("year", "first"),
            paper_credit=("credit", "sum"),
            times_cited=("times_cited", "first"),
            altmetric=("altmetric", "first"),
            for_l2=("for_l2", "first"),
        )
        .reset_index()
    )


def _entity_gender_metrics(credits, entity_col):
    known = credits[credits["name_gender"].isin(["Female", "Male"])].copy()
    totals = known.groupby(entity_col)["credit"].sum().rename("classified_name_credit")
    female = (
        known[known["name_gender"].eq("Female")]
        .groupby(entity_col)["credit"]
        .sum()
        .rename("female_name_credit")
    )
    out = pd.concat([totals, female], axis=1).fillna(0)
    out["female_name_share"] = 100 * out["female_name_credit"] / out[
        "classified_name_credit"
    ].replace(0, np.nan)
    return out


def build_institution_metrics(institution_credits: pd.DataFrame) -> pd.DataFrame:
    """Institution productivity, impact, collaboration, and name-gender composition."""
    paper = _entity_paper_table(institution_credits, "institution_id")
    grouped = paper.groupby("institution_id", observed=True)
    out = grouped.agg(
        fractional_paper_credit=("paper_credit", "sum"),
        unique_ukb_papers=("paper_id", "nunique"),
        first_ukb_year=("year", "min"),
        last_ukb_year=("year", "max"),
        total_ukb_citations=("times_cited", "sum"),
        mean_ukb_citations=("times_cited", "mean"),
        median_ukb_citations=("times_cited", "median"),
        max_ukb_citations=("times_cited", "max"),
        total_altmetric=("altmetric", "sum"),
    )
    citation_lists = grouped["times_cited"].agg(list)
    out["ukb_h_index"] = citation_lists.apply(h_index)
    out["ukb_g_index"] = citation_lists.apply(g_index)
    out["ukb_i10_index"] = citation_lists.apply(lambda x: int(np.sum(np.asarray(x) >= 10)))

    attrs = institution_credits.groupby("institution_id", observed=True).agg(
        institution=("org_name", _mode),
        org_id=("org_id", _mode),
        country=("country", _mode),
        country_iso3=("iso3", _mode),
        org_type=("org_type", _mode),
        unique_resolved_authors=(
            "researcher_id",
            lambda s: s.dropna().nunique(),
        ),
        author_affiliation_rows=("author_id", "size"),
    )
    out = out.join(attrs).join(_entity_gender_metrics(institution_credits, "institution_id"))

    field = paper[["institution_id", "paper_credit", "for_l2"]].explode("for_l2")
    field["field_credit"] = field["paper_credit"] / field.groupby(
        ["institution_id", field.index]
    )["for_l2"].transform("count").replace(0, np.nan)
    out["modal_for_l2"] = _weighted_modal(
        field.rename(columns={"for_l2": "field", "field_credit": "credit"}),
        "institution_id",
        "field",
    )

    partners = defaultdict(set)
    institution_counts = {}
    sole_counts = Counter()
    for paper_id, group in paper.groupby("paper_id"):
        institutions = sorted(set(group["institution_id"]))
        institution_counts[paper_id] = len(institutions)
        if len(institutions) == 1:
            sole_counts[institutions[0]] += 1
        for institution in institutions:
            partners[institution].update(set(institutions) - {institution})
    out["unique_institutional_collaborators"] = [
        len(partners[key]) for key in out.index
    ]
    paper["n_institutions"] = paper["paper_id"].map(institution_counts)
    out["mean_institutions_per_paper"] = paper.groupby("institution_id")[
        "n_institutions"
    ].mean()
    out["sole_institution_papers"] = [sole_counts[key] for key in out.index]
    out.index.name = "institution_id"
    return (
        out.reset_index()
        .sort_values(["fractional_paper_credit", "ukb_h_index"], ascending=False)
        .reset_index(drop=True)
    )


def build_country_metrics(
    country_credits: pd.DataFrame,
    papers: pd.DataFrame,
) -> pd.DataFrame:
    paper = _entity_paper_table(country_credits, "iso3")
    out = paper.groupby("iso3", observed=True).agg(
        fractional_paper_credit=("paper_credit", "sum"),
        unique_ukb_papers=("paper_id", "nunique"),
        first_ukb_year=("year", "min"),
        last_ukb_year=("year", "max"),
    )
    attrs = country_credits.groupby("iso3", observed=True).agg(
        country=("country", _mode),
        iso2=("iso2", _mode),
        unique_resolved_authors=("researcher_id", lambda s: s.dropna().nunique()),
        author_affiliation_rows=("author_id", "size"),
    )
    out = out.join(attrs).join(_entity_gender_metrics(country_credits, "iso3"))
    out["author_basis_unique_papers"] = out["unique_ukb_papers"]
    out["authors_per_paper"] = (
        out["author_affiliation_rows"] / out["author_basis_unique_papers"]
    )

    org_rows = []
    for paper_row in papers[["id", "research_org_countries"]].itertuples(index=False):
        for item in parse_listcol(paper_row.research_org_countries):
            if isinstance(item, dict):
                iso2, iso3 = country_to_iso(item.get("id"), item.get("name"))
                if iso3:
                    org_rows.append((paper_row.id, iso3, iso2, item.get("name")))
    org_basis = pd.DataFrame(
        org_rows, columns=["paper_id", "iso3", "iso2", "country"]
    ).drop_duplicates(["paper_id", "iso3"])
    if not org_basis.empty:
        org_counts = org_basis.groupby("iso3")["paper_id"].nunique()
        out = out.reindex(out.index.union(org_counts.index))
        out["org_basis_unique_papers"] = org_counts
        org_attrs = org_basis.groupby("iso3").agg(
            org_basis_iso2=("iso2", _mode),
            org_basis_country=("country", _mode),
        )
        out["iso2"] = out["iso2"].fillna(org_attrs["org_basis_iso2"])
        out["country"] = out["country"].fillna(org_attrs["org_basis_country"])
    out["org_basis_unique_papers"] = out.get("org_basis_unique_papers", np.nan)
    out["author_vs_org_paper_percent"] = (
        100 * out["author_basis_unique_papers"] / out["org_basis_unique_papers"]
    )
    out.index.name = "iso3"
    return (
        out.reset_index()
        .sort_values("fractional_paper_credit", ascending=False)
        .reset_index(drop=True)
    )


def _wilson_interval(successes, total, z=1.959963984540054):
    if total <= 0:
        return np.nan, np.nan
    p = successes / total
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator
    return 100 * (centre - margin), 100 * (centre + margin)


def _gender_summary(frame, group_cols, weight_col=None):
    work = frame.copy()
    weighted = weight_col is not None
    if weight_col is None:
        work["_weight"] = 1.0
        weight_col = "_weight"
    counts = (
        work.groupby([*group_cols, "name_gender"], observed=True)[weight_col]
        .sum()
        .unstack(fill_value=0)
    )
    for category in ["Female", "Male", "Unknown"]:
        if category not in counts:
            counts[category] = 0.0
    counts["classified"] = counts["Female"] + counts["Male"]
    counts["total"] = counts["classified"] + counts["Unknown"]
    counts["female_name_share"] = 100 * counts["Female"] / counts["classified"].replace(0, np.nan)
    counts["unknown_name_share"] = 100 * counts["Unknown"] / counts["total"].replace(0, np.nan)
    # Wilson intervals require integer Bernoulli trials. Fractional affiliation
    # credits are descriptive allocations, so no pseudo-precision is attached to them.
    intervals = (
        [(np.nan, np.nan)] * len(counts)
        if weighted
        else [
            _wilson_interval(row["Female"], row["classified"])
            for _, row in counts.iterrows()
        ]
    )
    counts["female_name_ci_low"] = [value[0] for value in intervals]
    counts["female_name_ci_high"] = [value[1] for value in intervals]
    return counts.reset_index()


def build_gender_tables(
    authorships: pd.DataFrame,
    country_credits: pd.DataFrame,
    institution_credits: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_year = _gender_summary(authorships, ["year"])

    by_role = _gender_summary(authorships, ["authorship_role"])
    corresponding = _gender_summary(
        authorships[authorships["is_corresponding"]], ["is_corresponding"]
    )
    if not corresponding.empty:
        corresponding["authorship_role"] = "Corresponding author"
        corresponding = corresponding.drop(columns="is_corresponding")
        by_role = pd.concat([by_role, corresponding], ignore_index=True)

    fields = authorships[["paper_id", "author_id", "for_l2", "name_gender"]].explode("for_l2")
    fields = fields.dropna(subset=["for_l2"])
    by_field = _gender_summary(fields, ["for_l2"])
    by_country = _gender_summary(country_credits, ["iso3", "country"], "credit")
    by_institution = _gender_summary(
        institution_credits, ["institution_id", "org_name"], "credit"
    )
    return by_year, by_role, by_field, by_country, by_institution


def _diversity_metrics(weights: pd.Series) -> dict:
    weights = weights[weights > 0].astype(float)
    if weights.empty:
        return {"hhi": np.nan, "effective_entities": np.nan, "top_1_share": np.nan,
                "top_10_share": np.nan}
    shares = weights / weights.sum()
    entropy = float(-(shares * np.log(shares)).sum())
    ordered = shares.sort_values(ascending=False)
    return {
        "hhi": float(np.square(shares).sum()),
        "effective_entities": float(np.exp(entropy)),
        "top_1_share": float(100 * ordered.head(1).sum()),
        "top_10_share": float(100 * ordered.head(10).sum()),
    }


def build_entity_time_table(
    credits: pd.DataFrame,
    entity_col: str,
    years=range(FIRST_YEAR, LAST_COMPLETE_YEAR + 1),
) -> pd.DataFrame:
    """Annual fractional totals and diversity/concentration for countries or institutions."""
    annual = credits.groupby(["year", entity_col], observed=True)["credit"].sum().reset_index()
    seen = set()
    rows = []
    for year in years:
        current = annual[annual["year"].eq(year)].set_index(entity_col)["credit"]
        seen.update(current.index)
        diversity = _diversity_metrics(current)
        rows.append({
            "year": year,
            "observed_entities": int(current.size),
            "cumulative_entities": len(seen),
            "geolocated_fractional_credit": float(current.sum()),
            **diversity,
        })
    return pd.DataFrame(rows)


def build_core_tables(papers: pd.DataFrame) -> CoreTables:
    organization_lookup = build_organization_lookup(papers)
    authorships, affiliations, institution_credits, country_credits = (
        build_authorship_tables(papers, organization_lookup)
    )
    author_metrics = build_author_metrics(
        authorships,
        institution_credits,
        country_credits,
        organization_lookup=organization_lookup,
    )
    institution_metrics = build_institution_metrics(institution_credits)
    country_metrics = build_country_metrics(country_credits, papers)
    gender_tables = build_gender_tables(
        authorships, country_credits, institution_credits
    )
    return CoreTables(
        authorships=authorships,
        affiliations=affiliations,
        institution_credits=institution_credits,
        country_credits=country_credits,
        author_metrics=author_metrics,
        institution_metrics=institution_metrics,
        country_metrics=country_metrics,
        gender_by_year=gender_tables[0],
        gender_by_role=gender_tables[1],
        gender_by_field=gender_tables[2],
        gender_by_country=gender_tables[3],
        gender_by_institution=gender_tables[4],
        country_by_year=build_entity_time_table(country_credits, "iso3"),
        institution_by_year=build_entity_time_table(institution_credits, "institution_id"),
        organization_lookup=organization_lookup,
    )


def _author_paper_incidence(authorships, max_team_size=None):
    stable = authorships[authorships["identity_resolved"]][
        ["author_id", "paper_id", "year"]
    ].drop_duplicates()
    paper_team = stable.groupby("paper_id")["author_id"].nunique()
    if max_team_size is not None:
        keep = paper_team[paper_team.le(max_team_size)].index
        stable = stable[stable["paper_id"].isin(keep)]
        paper_team = paper_team.loc[paper_team.index.intersection(keep)]
    authors = np.sort(stable["author_id"].unique())
    papers = np.sort(stable["paper_id"].unique())
    author_index = pd.Series(np.arange(len(authors)), index=authors)
    paper_index = pd.Series(np.arange(len(papers)), index=papers)
    rows = stable["author_id"].map(author_index).to_numpy()
    cols = stable["paper_id"].map(paper_index).to_numpy()
    incidence = sp.csr_matrix(
        (np.ones(len(stable), dtype=np.float32), (rows, cols)),
        shape=(len(authors), len(papers)),
    )
    team_sizes = np.asarray(incidence.sum(axis=0)).ravel()
    return incidence, authors, papers, team_sizes


def build_coauthor_matrices(authorships, max_team_size=None):
    """Raw shared-paper and team-size-normalized author adjacency matrices."""
    incidence, authors, papers, team_sizes = _author_paper_incidence(
        authorships, max_team_size=max_team_size
    )
    raw = (incidence @ incidence.T).tocsr()
    raw.setdiag(0)
    raw.eliminate_zeros()

    factors = np.zeros_like(team_sizes, dtype=np.float32)
    valid = team_sizes > 1
    factors[valid] = 1.0 / (team_sizes[valid] - 1.0)
    weighted_incidence = incidence @ sp.diags(np.sqrt(factors))
    fractional = (weighted_incidence @ weighted_incidence.T).tocsr()
    fractional.setdiag(0)
    fractional.eliminate_zeros()
    return raw, fractional, authors, papers, team_sizes


def summarize_network(raw: sp.csr_matrix, fractional: sp.csr_matrix) -> tuple[dict, np.ndarray, np.ndarray]:
    adjacency = raw.copy()
    adjacency.data = np.ones_like(adjacency.data, dtype=np.int8)
    n_nodes = adjacency.shape[0]
    degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    n_components, labels = connected_components(adjacency, directed=False)
    sizes = np.bincount(labels, minlength=n_components) if n_nodes else np.array([])
    giant = int(sizes.max()) if sizes.size else 0
    if giant:
        giant_label = int(np.argmax(sizes))
        giant_mask = labels == giant_label
        giant_adjacency = adjacency[giant_mask][:, giant_mask]
        giant_edges = int(giant_adjacency.nnz // 2)
        giant_density = (
            float(giant_adjacency.nnz / (giant * (giant - 1)))
            if giant > 1
            else 0.0
        )
    else:
        giant_edges = 0
        giant_density = 0.0
    upper_raw = sp.triu(raw, k=1).data
    median_tie = float(np.median(upper_raw)) if upper_raw.size else np.nan
    weak_ties = int(np.sum(upper_raw <= median_tie)) if upper_raw.size else 0
    strong_ties = int(np.sum(upper_raw > median_tie)) if upper_raw.size else 0
    mean_degree = float(degrees.mean()) if n_nodes else np.nan
    mean_tie = float(np.mean(upper_raw)) if upper_raw.size else np.nan
    metrics = {
        "n_nodes": int(n_nodes),
        "n_edges": int(adjacency.nnz // 2),
        "n_components": int(n_components),
        "n_isolates": int(np.sum(degrees == 0)),
        "giant_nodes": giant,
        "giant_edges": giant_edges,
        "giant_density": giant_density,
        "giant_fraction": giant / n_nodes if n_nodes else np.nan,
        "mean_degree": mean_degree,
        "avg_degree": mean_degree,
        "median_degree": float(np.median(degrees)) if n_nodes else np.nan,
        "max_degree": int(degrees.max()) if n_nodes else 0,
        "density": float(adjacency.nnz / (n_nodes * (n_nodes - 1))) if n_nodes > 1 else 0.0,
        "mean_tie_strength": mean_tie,
        "avg_tie_strength": mean_tie,
        "median_tie_strength": median_tie,
        "max_tie_strength": float(np.max(upper_raw)) if upper_raw.size else np.nan,
        "n_weak_ties": weak_ties,
        "n_strong_ties": strong_ties,
        "weak_tie_ratio": weak_ties / upper_raw.size if upper_raw.size else np.nan,
        "mean_fractional_strength": float(np.asarray(fractional.sum(axis=1)).mean())
        if n_nodes else np.nan,
    }
    return metrics, labels, degrees


def network_metrics_over_time(
    authorships: pd.DataFrame,
    first_year=FIRST_YEAR,
    last_year=LAST_COMPLETE_YEAR,
    hyperauthor_threshold=HYPERAUTHOR_THRESHOLD,
) -> pd.DataFrame:
    rows = []
    for year in range(first_year, last_year + 1):
        cumulative = authorships[authorships["year"].le(year)]
        for scenario, threshold in (
            ("All papers", None),
            (f"Teams <= {hyperauthor_threshold}", hyperauthor_threshold),
        ):
            raw, fractional, _, papers, team_sizes = build_coauthor_matrices(
                cumulative, max_team_size=threshold
            )
            metrics, _, _ = summarize_network(raw, fractional)
            rows.append({
                "year": year,
                "scenario": scenario,
                "n_papers": len(papers),
                "largest_team": int(team_sizes.max()) if team_sizes.size else 0,
                **metrics,
            })
    return pd.DataFrame(rows)


def _detect_leiden(fractional: sp.csr_matrix, resolution=LEIDEN_RESOLUTION, seed=LEIDEN_SEED):
    import igraph as ig

    ig.set_random_number_generator(random.Random(seed))
    graph = ig.Graph.Weighted_Adjacency(fractional, mode="undirected", loops=False)
    partition = graph.community_leiden(
        objective_function="modularity",
        weights="weight",
        resolution=resolution,
        n_iterations=-1,
    )
    membership = np.asarray(partition.membership, dtype=int)
    modularity = float(graph.modularity(membership.tolist(), weights="weight"))
    return membership, modularity


def _rank_communities(membership):
    counts = pd.Series(membership).value_counts()
    order = counts.sort_values(ascending=False).index.tolist()
    mapping = {community: rank for rank, community in enumerate(order, start=1)}
    return np.asarray([mapping[value] for value in membership], dtype=int)


def _top_labels(frame, group_col, value_col, n=3):
    counts = (
        frame.dropna(subset=[value_col])
        .groupby([group_col, value_col], observed=True)
        .size()
        .reset_index(name="n")
    )
    totals = counts.groupby(group_col)["n"].sum()
    counts["share"] = counts["n"] / counts[group_col].map(totals)
    counts = counts.sort_values([group_col, "n", value_col], ascending=[True, False, True])
    return counts.groupby(group_col).head(n)


def _attribute_distribution_by_group(
    frame: pd.DataFrame,
    group_col: str,
    value_col: str,
    missing_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return complete counts and a compact top-three summary by author group."""
    base = frame[[group_col, "researcher_id", value_col]].drop_duplicates(
        [group_col, "researcher_id"]
    )
    base[value_col] = base[value_col].fillna(missing_label)
    counts = (
        base.groupby([group_col, value_col], observed=True)["researcher_id"]
        .nunique()
        .reset_index(name="n_authors")
    )
    totals = base.groupby(group_col, observed=True)["researcher_id"].nunique()
    counts["group_authors"] = counts[group_col].map(totals)
    counts["author_share"] = counts["n_authors"] / counts["group_authors"]
    counts["_missing"] = counts[value_col].eq(missing_label)
    counts = counts.sort_values(
        [group_col, "n_authors", "_missing", value_col],
        ascending=[True, False, True, True],
    )
    counts["rank"] = counts.groupby(group_col, observed=True).cumcount() + 1

    leading = counts[counts["rank"].eq(1)].rename(
        columns={
            value_col: "leading_value",
            "n_authors": "leading_value_authors",
            "author_share": "leading_value_share",
        }
    )[
        [
            group_col,
            "group_authors",
            "leading_value",
            "leading_value_authors",
            "leading_value_share",
        ]
    ]
    top = counts[counts["rank"].le(3)].copy()
    top["_text"] = (
        top[value_col].astype(str)
        + " ("
        + top["n_authors"].astype(str)
        + ", "
        + (100 * top["author_share"]).round(1).astype(str)
        + "%)"
    )
    top_text = (
        top.groupby(group_col, observed=True)["_text"]
        .agg("; ".join)
        .rename("top_three")
    )
    summary = leading.merge(top_text, on=group_col, how="left")
    return summary, counts.drop(columns="_missing").reset_index(drop=True)


def community_intersection_tables(network: NetworkTables) -> OrderedDict:
    """Full author-characteristic distributions within network communities."""
    member = network.community_membership.copy()
    member["plot_node"] = np.where(
        member["community"].le(12),
        "Community " + member["community"].astype(str),
        "Other communities",
    )
    tables = OrderedDict()
    specifications = [
        ("for_l2", "modal_for_l2", "Unclassified"),
        ("for_l4", "modal_for_l4", "Unclassified"),
        ("institution", "modal_institution", "Unknown institution"),
        ("country", "modal_country", "Unknown country"),
        ("name_gender", "name_gender", "Unknown"),
    ]
    for stem, value_col, missing_label in specifications:
        summary, counts = _attribute_distribution_by_group(
            member, "community", value_col, missing_label
        )
        tables[f"network_community_{stem}_summary.csv"] = summary
        tables[f"network_community_{stem}_counts.csv"] = counts
        if stem in {"for_l2", "for_l4", "institution"}:
            collapsed_summary, collapsed_counts = _attribute_distribution_by_group(
                member, "plot_node", value_col, missing_label
            )
            tables[f"network_collapsed_{stem}_summary.csv"] = collapsed_summary
            tables[f"network_collapsed_{stem}_counts.csv"] = collapsed_counts
    return tables


def build_community_tables(
    raw,
    fractional,
    author_ids,
    network_author_metrics,
    resolution=LEIDEN_RESOLUTION,
    seed=LEIDEN_SEED,
    n_display=12,
):
    membership, modularity = _detect_leiden(fractional, resolution, seed)
    ranked = _rank_communities(membership)
    member = pd.DataFrame({
        "researcher_id": author_ids,
        "community": ranked,
    }).merge(network_author_metrics, on="researcher_id", how="left", validate="one_to_one")

    summary = member.groupby("community", observed=True).agg(
        n_authors=("researcher_id", "size"),
        mean_ukb_h_index=("ukb_h_index", "mean"),
        median_ukb_papers=("n_ukb_papers", "median"),
        female_name_share=(
            "name_gender",
            lambda s: 100 * s.eq("Female").sum() / max(s.isin(["Female", "Male"]).sum(), 1),
        ),
        modal_for_l2=("modal_for_l2", _mode),
        modal_for_l4=("modal_for_l4", _mode),
        modal_institution=("modal_institution", _mode),
        modal_country=("modal_country", _mode),
    ).reset_index()
    summary["author_share"] = 100 * summary["n_authors"] / len(member)
    summary["cumulative_author_share"] = summary.sort_values("community")[
        "author_share"
    ].cumsum()

    top_topics = _top_labels(member, "community", "modal_for_l4")
    top_institutions = _top_labels(member, "community", "modal_institution")
    summary["top_topics"] = summary["community"].map(
        top_topics.groupby("community").apply(
            lambda d: "; ".join(f"{v} ({100*s:.0f}%)" for v, s in zip(d["modal_for_l4"], d["share"])),
            include_groups=False,
        )
    )
    summary["top_institutions"] = summary["community"].map(
        top_institutions.groupby("community").apply(
            lambda d: "; ".join(f"{v} ({100*s:.0f}%)" for v, s in zip(d["modal_institution"], d["share"])),
            include_groups=False,
        )
    )

    upper = sp.triu(fractional, k=1).tocoo()
    edge_frame = pd.DataFrame({
        "source_community": ranked[upper.row],
        "target_community": ranked[upper.col],
        "weight": upper.data,
    })
    edge_frame = edge_frame[edge_frame["source_community"] != edge_frame["target_community"]]
    edge_frame[["source_community", "target_community"]] = np.sort(
        edge_frame[["source_community", "target_community"]], axis=1
    )
    community_edges = (
        edge_frame.groupby(["source_community", "target_community"], observed=True)["weight"]
        .sum()
        .reset_index()
    )

    display = set(summary.nlargest(n_display, "n_authors")["community"])
    collapse_map = {
        community: (f"Community {community}" if community in display else "Other communities")
        for community in summary["community"]
    }
    nodes = summary.copy()
    nodes["plot_node"] = nodes["community"].map(collapse_map)
    collapsed_nodes = nodes.groupby("plot_node", observed=True).agg(
        n_authors=("n_authors", "sum"),
        n_communities=("community", "size"),
        modal_for_l2=("modal_for_l2", _mode),
        modal_for_l4=("modal_for_l4", _mode),
        modal_institution=("modal_institution", _mode),
        female_name_share=("female_name_share", "mean"),
    ).reset_index()
    ce = community_edges.copy()
    ce["source"] = ce["source_community"].map(collapse_map)
    ce["target"] = ce["target_community"].map(collapse_map)
    ce = ce[ce["source"] != ce["target"]]
    ce[["source", "target"]] = np.sort(ce[["source", "target"]], axis=1)
    collapsed_edges = ce.groupby(["source", "target"], observed=True)["weight"].sum().reset_index()
    return member, summary, collapsed_nodes, collapsed_edges, modularity


def build_network_tables(
    authorships: pd.DataFrame,
    author_metrics: pd.DataFrame,
    first_year=FIRST_YEAR,
    last_year=LAST_COMPLETE_YEAR,
    hyperauthor_threshold=HYPERAUTHOR_THRESHOLD,
) -> NetworkTables:
    temporal = network_metrics_over_time(
        authorships, first_year, last_year, hyperauthor_threshold
    )
    raw, fractional, author_ids, _, _ = build_coauthor_matrices(authorships)
    final_metrics, component_labels, degrees = summarize_network(raw, fractional)
    component_sizes = np.bincount(component_labels)
    fractional_strength = np.asarray(fractional.sum(axis=1)).ravel()
    shared_paper_strength = np.asarray(raw.sum(axis=1)).ravel()
    net_author = pd.DataFrame({
        "researcher_id": author_ids,
        "coauthor_count": degrees.astype(int),
        "shared_paper_strength": shared_paper_strength,
        "fractional_collaboration_strength": fractional_strength,
        "component": component_labels,
        "component_size": component_sizes[component_labels],
    })
    net_author["in_giant_component"] = net_author["component_size"].eq(
        int(component_sizes.max())
    )
    combined = author_metrics.merge(net_author, on="researcher_id", how="left")
    member, summary, nodes, edges, modularity = build_community_tables(
        raw, fractional, author_ids, combined
    )
    net_author = net_author.merge(
        member[["researcher_id", "community"]], on="researcher_id", how="left"
    )

    temporal.attrs["final_metrics"] = final_metrics
    return NetworkTables(
        metrics_by_year=temporal,
        author_metrics=net_author,
        community_membership=member,
        community_summary=summary,
        collapsed_nodes=nodes,
        collapsed_edges=edges,
        adjacency=raw,
        modularity=modularity,
    )


def merge_author_network_metrics(
    author_metrics: pd.DataFrame,
    network: NetworkTables,
) -> pd.DataFrame:
    columns = [
        "researcher_id",
        "coauthor_count",
        "shared_paper_strength",
        "fractional_collaboration_strength",
        "component_size",
        "in_giant_component",
        "community",
    ]
    return author_metrics.merge(network.author_metrics[columns], on="researcher_id", how="left")


def network_figure_tables(network: NetworkTables) -> OrderedDict[str, pd.DataFrame]:
    """Return compact structural summaries used by the network supplement."""
    annual = (
        network.metrics_by_year.loc[
            network.metrics_by_year["scenario"].eq("All papers"),
            ["year", "n_nodes"],
        ]
        .sort_values("year")
        .rename(columns={"n_nodes": "cumulative_resolved_authors"})
        .reset_index(drop=True)
    )
    annual["new_resolved_authors"] = (
        annual["cumulative_resolved_authors"].diff()
        .fillna(annual["cumulative_resolved_authors"])
        .astype(int)
    )

    degree = (
        network.author_metrics["coauthor_count"]
        .fillna(0)
        .astype(int)
        .value_counts()
        .sort_index()
        .rename_axis("coauthor_count")
        .rename("n_authors")
        .reset_index()
    )
    degree["author_share_percent"] = 100 * degree["n_authors"] / degree["n_authors"].sum()
    degree["survival_percent"] = 100 * degree["n_authors"][::-1].cumsum()[::-1] / degree["n_authors"].sum()

    tie_strength = sp.triu(network.adjacency, k=1).data
    tie_labels = ["1 paper", "2 papers", "3-4 papers", "5+ papers"]
    tie_category = pd.cut(
        tie_strength,
        bins=[0, 1, 2, 4, np.inf],
        labels=tie_labels,
        ordered=True,
    )
    tie_distribution = (
        pd.Series(tie_category, name="strength_band")
        .value_counts(sort=False)
        .rename("n_ties")
        .reset_index()
    )
    tie_distribution["tie_share_percent"] = (
        100 * tie_distribution["n_ties"] / tie_distribution["n_ties"].sum()
    )

    components = (
        network.author_metrics[["component", "component_size"]]
        .drop_duplicates("component")
        .sort_values(["component_size", "component"], ascending=[False, True])
        .reset_index(drop=True)
    )
    components.insert(0, "component_rank", np.arange(1, len(components) + 1))
    giant_size = int(components["component_size"].max())
    components["component_class"] = np.select(
        [
            components["component_size"].eq(giant_size),
            components["component_size"].eq(1),
            components["component_size"].between(2, 5),
        ],
        ["Giant component", "Isolate", "Small (2-5)"],
        default="Intermediate (6-55)",
    )

    final = network.metrics_by_year.loc[
        network.metrics_by_year["year"].eq(LAST_COMPLETE_YEAR)
    ].set_index("scenario")
    sensitivity_metrics = [
        ("n_nodes", "Resolved authors"),
        ("n_edges", "Unique coauthor ties"),
        ("median_degree", "Median coauthor count"),
        ("giant_fraction", "Giant-component share"),
        ("mean_fractional_strength", "Mean normalized strength"),
    ]
    sensitivity_rows = []
    restricted_label = f"Teams <= {HYPERAUTHOR_THRESHOLD}"
    for metric, label in sensitivity_metrics:
        all_value = float(final.loc["All papers", metric])
        restricted_value = float(final.loc[restricted_label, metric])
        sensitivity_rows.append({
            "metric": metric,
            "label": label,
            "all_papers": all_value,
            "teams_at_most_100": restricted_value,
            "retained_percent": 100 * restricted_value / all_value,
        })
    sensitivity = pd.DataFrame(sensitivity_rows)

    return OrderedDict({
        "network_new_authors_by_year.csv": annual,
        "network_degree_distribution.csv": degree,
        "network_tie_strength_distribution.csv": tie_distribution,
        "network_component_rank_size.csv": components,
        "network_hyperauthorship_sensitivity.csv": sensitivity,
    })


def quality_audit(source, papers, core: CoreTables) -> pd.DataFrame:
    authorships = core.authorships
    gender_coverage = NG.inference_coverage(authorships).set_index("stage")
    gender_methods = NG.inference_method_counts(authorships)
    offline_direct_rows = int(
        authorships["name_gender_direct"].isin(NG.CLASSIFIED_CATEGORIES).sum()
    )
    offline_conflicts = int(authorships["name_gender_direct_conflict"].sum())
    accepted_majorities = int(
        gender_methods.loc[
            gender_methods["name_gender_method"].eq(
                "offline_ensemble_majority"
            ),
            "author_paper_pairs",
        ].sum()
    )
    identity_rows = int(
        gender_methods.loc[
            gender_methods["name_gender_method"].eq(
                "researcher_identity_consensus"
            ),
            "author_paper_pairs",
        ].sum()
    )
    identity_conflicts = int(
        authorships.loc[
            authorships["name_gender_identity_conflict"], "researcher_id"
        ].nunique()
    )
    library_audit = NG.offline_library_audit(authorships)
    unresolved_queries = NG.unresolved_query_queue(authorships)
    parsed_by_paper = authorships.groupby("paper_id").size()
    reported = papers.set_index("id")["authors_count"]
    comparable = pd.concat([parsed_by_paper.rename("parsed"), reported.rename("reported")], axis=1)
    country_credit = core.country_credits["credit"].sum()
    institution_credit = core.institution_credits["credit"].sum()
    institution_rows = core.affiliations.dropna(subset=["institution_id"])
    institution_metadata_variants = len(institution_rows) - len(
        institution_rows.drop_duplicates(["paper_id", "author_id", "institution_id"])
    )
    rows = [
        ("Source parquet records", len(source), "includes provisional years"),
        ("Records after 2025", int(source["year"].gt(LAST_COMPLETE_YEAR).sum()), "excluded from primary analysis"),
        ("Complete-year papers", len(papers), f"{FIRST_YEAR}-{LAST_COMPLETE_YEAR}"),
        ("Papers with parsed authors", authorships["paper_id"].nunique(), "author-based denominator"),
        ("Papers without parsed authors", int(len(papers) - authorships["paper_id"].nunique()), "retained in source audit only"),
        ("Parsed author-paper pairs", len(authorships), "deduplicated within paper"),
        ("Resolved author-paper pairs", int(authorships["identity_resolved"].sum()), "Dimensions researcher ID present"),
        ("Unresolved author-paper pairs", int((~authorships["identity_resolved"]).sum()), "paper-local IDs; never merged by name"),
        ("Unique resolved authors", core.author_metrics["researcher_id"].nunique(), "author table rows"),
        ("Author homes from current organization", int(core.author_metrics["home_assignment_source"].eq("current_organization").sum()), "GRID-resolved current organization"),
        ("Author homes from modal affiliation", int(core.author_metrics["home_assignment_source"].eq("modal_affiliation").sum()), "fallback includes authors with no resolved home"),
        ("Current-organization affiliation backfills", int(core.affiliations["affiliation_source"].eq("current_org_backfill").sum()), "used only when no paper affiliation was listed"),
        ("Authorship rows with no affiliation", int(core.affiliations["affiliation_source"].eq("missing").sum()), "excluded only from affiliation analyses"),
        ("Geolocated fractional paper credit", round(float(country_credit), 2), "out of papers with parsed authors"),
        ("Institutional fractional paper credit", round(float(institution_credit), 2), "out of papers with parsed authors"),
        ("Duplicate institution metadata variants collapsed", int(institution_metadata_variants), "same author-paper institution; counted once for credit"),
        (
            "Strict dictionary-classified name pairs",
            int(gender_coverage.loc["Strict dictionary", "classified_author_paper_pairs"]),
            "gender_guesser exact female/male calls",
        ),
        (
            "Expanded dictionary-classified name pairs",
            int(gender_coverage.loc["Expanded dictionary", "classified_author_paper_pairs"]),
            "adds mostly_female/mostly_male after shared name normalization",
        ),
        (
            "Primary name-classified author-paper pairs",
            int(gender_coverage.loc["Primary + identity linkage", "classified_author_paper_pairs"]),
            "offline ensemble plus conflict-safe researcher linkage",
        ),
        (
            "Offline ensemble-classified name pairs",
            offline_direct_rows,
            "five local libraries; no APIs or remote caches",
        ),
        (
            "Offline package-vote conflicts",
            offline_conflicts,
            "retained only when the configured supermajority rule was met",
        ),
        (
            "Accepted conflicting-vote majorities",
            accepted_majorities,
            (
                f">={NG.ENSEMBLE_MIN_MAJORITY_VOTES} winning votes and "
                f">={NG.ENSEMBLE_MIN_VOTE_MARGIN}-vote margin"
            ),
        ),
        (
            "Researcher-linkage name assignments",
            identity_rows,
            "unknown row linked only when all direct calls for that researcher agreed",
        ),
        (
            "Researchers with conflicting direct name calls",
            identity_conflicts,
            "all rows conservatively returned to Unknown",
        ),
        (
            "Unknown/androgynous name pairs",
            int(authorships["name_gender"].eq("Unknown").sum()),
            "kept in totals and excluded only from Female/Male share denominators",
        ),
        (
            "Unresolved normalized names after offline ensemble",
            len(unresolved_queries),
            "retained as Unknown; exported with all package votes",
        ),
        ("Papers missing FOR L2", int(papers["for_l2"].apply(len).eq(0).sum()), "kept as unclassified"),
        ("Papers with >=2 FOR L2 divisions", int(papers["for_l2"].apply(len).ge(2).sum()), "multi-label; no primary field assumed"),
        ("Largest parsed author team", int(authorships["team_size"].max()), "network sensitivity threshold is 100"),
        ("Papers with parsed list shorter than authors_count", int((comparable["parsed"] < comparable["reported"]).fillna(False).sum()), "possible source truncation"),
    ]
    rows.extend(
        (
            f"{row.library} classified name pairs",
            int(row.classified_author_paper_pairs),
            f"offline package {row.version}; {row.coverage_percent:.1f}% coverage",
        )
        for row in library_audit.itertuples(index=False)
    )
    return pd.DataFrame(rows, columns=["metric", "value", "interpretation"])


def validation_checks(
    source,
    papers,
    core: CoreTables,
    network: NetworkTables | None = None,
) -> pd.DataFrame:
    authorships = core.authorships
    paper_credit = authorships.groupby("paper_id")["authorship_credit"].sum()
    country_credit = core.country_credits.groupby("paper_id")["credit"].sum()
    institution_credit = core.institution_credits.groupby("paper_id")["credit"].sum()
    institution_authors = core.institution_credits[
        ["paper_id", "author_id"]
    ].drop_duplicates()
    country_authors = core.country_credits[["paper_id", "author_id"]].drop_duplicates()
    expected_institution_credit = (
        authorships.merge(
            institution_authors,
            on=["paper_id", "author_id"],
            how="inner",
            validate="one_to_one",
        )
        .groupby("paper_id")["authorship_credit"]
        .sum()
    )
    expected_country_credit = (
        authorships.merge(
            country_authors,
            on=["paper_id", "author_id"],
            how="inner",
            validate="one_to_one",
        )
        .groupby("paper_id")["authorship_credit"]
        .sum()
    )
    institution_delta = institution_credit.sub(
        expected_institution_credit, fill_value=0
    ).abs()
    country_delta = country_credit.sub(expected_country_credit, fill_value=0).abs()
    concentration = author_credit_concentration(core.author_metrics)
    productivity = author_productivity_bands(core.author_metrics)
    checks = [
        ("unique_source_publication_ids", not source["id"].duplicated().any(), source["id"].nunique(), len(source)),
        ("complete_year_window", papers["year"].between(FIRST_YEAR, LAST_COMPLETE_YEAR).all(), f"{papers['year'].min()}-{papers['year'].max()}", f"{FIRST_YEAR}-{LAST_COMPLETE_YEAR}"),
        ("unique_author_paper_pairs", not authorships.duplicated(["author_id", "paper_id"]).any(), len(authorships), len(authorships)),
        ("authorship_credit_sums_to_one", np.allclose(paper_credit, 1.0), float(np.abs(paper_credit - 1).max()), "<=1e-9"),
        ("country_credit_never_exceeds_one", bool(country_credit.le(1 + 1e-9).all()), float(country_credit.max()), "<=1"),
        ("institution_credit_never_exceeds_one", bool(institution_credit.le(1 + 1e-9).all()), float(institution_credit.max()), "<=1"),
        ("institution_credit_conserved", bool(institution_delta.le(1e-9).all()), float(institution_delta.max()), "<=1e-9"),
        ("country_credit_conserved", bool(country_delta.le(1e-9).all()), float(country_delta.max()), "<=1e-9"),
        ("fractional_credits_nonnegative", bool(core.institution_credits["credit"].ge(0).all() and core.country_credits["credit"].ge(0).all()), "nonnegative", "nonnegative"),
        ("ukb_h_index_bounded_by_papers", bool(core.author_metrics["ukb_h_index"].le(core.author_metrics["n_ukb_papers"]).all()), int(core.author_metrics["ukb_h_index"].max()), "<= n_ukb_papers"),
        ("author_credit_concentration_monotonic", bool(concentration["credit_share_percent"].is_monotonic_increasing and concentration["credit_share_percent"].between(0, 100).all()), ", ".join(f"{value:.1f}" for value in concentration["credit_share_percent"]), "nondecreasing; 0-100"),
        ("author_productivity_bands_exhaustive", int(productivity["n_authors"].sum()) == len(core.author_metrics), int(productivity["n_authors"].sum()), len(core.author_metrics)),
        ("gender_shares_bounded", bool(core.gender_by_year["female_name_share"].dropna().between(0, 100).all()), "0-100", "0-100"),
        (
            "name_gender_categories_valid",
            bool(authorships["name_gender"].isin(["Female", "Male", "Unknown"]).all()),
            ", ".join(sorted(authorships["name_gender"].unique())),
            "Female, Male, Unknown",
        ),
        (
            "name_gender_dictionary_sensitivity_ordered",
            int(authorships["name_gender_strict"].isin(["Female", "Male"]).sum())
            <= int(authorships["name_gender_dictionary"].isin(["Female", "Male"]).sum())
            <= int(authorships["name_gender_direct"].isin(["Female", "Male"]).sum()),
            "strict <= expanded <= enhanced direct",
            "strict <= expanded <= enhanced direct",
        ),
        (
            "resolved_name_categories_consistent",
            bool(
                authorships[authorships["identity_resolved"]]
                .loc[lambda frame: frame["name_gender"].isin(["Female", "Male"])]
                .groupby("researcher_id")["name_gender"]
                .nunique()
                .le(1)
                .all()
            ),
            "at most one classified category per researcher",
            "at most one classified category per researcher",
        ),
        (
            "offline_name_libraries_available",
            bool(NG.offline_library_versions()["available"].all()),
            ", ".join(
                NG.offline_library_versions().apply(
                    lambda row: f"{row['library']}={row['version']}", axis=1
                )
            ),
            "all pinned offline libraries available",
        ),
        (
            "offline_conflicting_votes_meet_majority_rule",
            bool(
                authorships.loc[
                    authorships["name_gender_direct_conflict"]
                    & authorships["name_gender_direct"].isin(
                        NG.CLASSIFIED_CATEGORIES
                    ),
                    ["name_gender_vote_count", "name_gender_vote_margin"],
                ]
                .assign(
                    valid=lambda frame: frame["name_gender_vote_count"].ge(
                        NG.ENSEMBLE_MIN_MAJORITY_VOTES
                    )
                    & frame["name_gender_vote_margin"].ge(
                        NG.ENSEMBLE_MIN_VOTE_MARGIN
                    )
                )["valid"]
                .all()
            ),
            (
                f">={NG.ENSEMBLE_MIN_MAJORITY_VOTES} votes; "
                f">={NG.ENSEMBLE_MIN_VOTE_MARGIN} margin"
            ),
            (
                f">={NG.ENSEMBLE_MIN_MAJORITY_VOTES} votes; "
                f">={NG.ENSEMBLE_MIN_VOTE_MARGIN} margin"
            ),
        ),
    ]
    if network is not None:
        all_years = network.metrics_by_year[network.metrics_by_year["scenario"].eq("All papers")]
        asymmetric = network.adjacency - network.adjacency.T
        checks.extend([
            ("network_years_complete", all_years["year"].tolist() == list(range(FIRST_YEAR, LAST_COMPLETE_YEAR + 1)), len(all_years), LAST_COMPLETE_YEAR - FIRST_YEAR + 1),
            ("network_author_coverage", network.author_metrics["researcher_id"].nunique() == core.author_metrics["researcher_id"].nunique(), network.author_metrics["researcher_id"].nunique(), core.author_metrics["researcher_id"].nunique()),
            ("community_membership_complete", network.community_membership["researcher_id"].nunique() == core.author_metrics["researcher_id"].nunique(), network.community_membership["researcher_id"].nunique(), core.author_metrics["researcher_id"].nunique()),
            ("network_adjacency_symmetric", asymmetric.nnz == 0, asymmetric.nnz, 0),
            ("network_adjacency_zero_diagonal", bool(np.allclose(network.adjacency.diagonal(), 0)), float(network.adjacency.diagonal().max(initial=0)), 0),
            ("network_edge_count_matches_adjacency", int(all_years.iloc[-1]["n_edges"]) == network.adjacency.nnz // 2, int(all_years.iloc[-1]["n_edges"]), network.adjacency.nnz // 2),
        ])
    return pd.DataFrame(checks, columns=["check", "passed", "observed", "expected"])


def gini(values) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values >= 0)]
    if values.size == 0 or values.sum() == 0:
        return np.nan
    values = np.sort(values)
    n = values.size
    return float((2 * np.sum(np.arange(1, n + 1) * values) / (n * values.sum())) - (n + 1) / n)


def author_credit_concentration(
    author_metrics: pd.DataFrame,
    thresholds=AUTHOR_CONCENTRATION_THRESHOLDS,
) -> pd.DataFrame:
    """Return credit held by the highest-credit share of resolved authors."""
    credit = pd.to_numeric(
        author_metrics["fractional_paper_credit"], errors="coerce"
    ).dropna().to_numpy(float)
    credit = np.sort(credit[credit >= 0])[::-1]
    if not len(credit) or credit.sum() <= 0:
        raise ValueError("Author concentration requires positive fractional credit")
    rows = []
    for percentage in thresholds:
        percentage = float(percentage)
        if not 0 < percentage <= 100:
            raise ValueError("Author concentration thresholds must be in (0, 100]")
        n_authors = max(1, math.ceil(len(credit) * percentage / 100))
        held_credit = float(credit[:n_authors].sum())
        rows.append({
            "top_author_percent": percentage,
            "n_authors": n_authors,
            "fractional_publication_credit": held_credit,
            "credit_share_percent": 100 * held_credit / credit.sum(),
        })
    return pd.DataFrame(rows)


def author_productivity_bands(author_metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize resolved authors in mutually exclusive UKB publication bands."""
    publication_counts = pd.to_numeric(
        author_metrics["n_ukb_papers"], errors="coerce"
    )
    if publication_counts.isna().any() or publication_counts.lt(1).any():
        raise ValueError(
            "Author productivity bands require positive publication counts"
        )

    labels = ["1 paper", "2-4 papers", "5-9 papers", "10+ papers"]
    bands = pd.cut(
        publication_counts,
        bins=[0, 1, 4, 9, np.inf],
        labels=labels,
        include_lowest=True,
    )
    result = (
        bands.value_counts(sort=False)
        .reindex(labels, fill_value=0)
        .rename("n_authors")
        .rename_axis("publication_band")
        .reset_index()
    )
    result["publication_band"] = result["publication_band"].astype(str)
    result["author_share_percent"] = (
        100 * result["n_authors"] / result["n_authors"].sum()
    )
    return result


def headline_statistics(core: CoreTables, network: NetworkTables) -> pd.DataFrame:
    gender = core.gender_by_year.set_index("year")
    institutions = core.institution_by_year.set_index("year")
    full_network = network.metrics_by_year[network.metrics_by_year["scenario"].eq("All papers")].set_index("year")
    concentration = author_credit_concentration(core.author_metrics)
    productivity = author_productivity_bands(core.author_metrics).set_index(
        "publication_band"
    )
    values = [
        ("Resolved authors", len(core.author_metrics), "authors"),
        ("Author-paper pairs", len(core.authorships), "authorships"),
        (
            "Authors with one UKB paper",
            productivity.loc["1 paper", "author_share_percent"],
            "% resolved authors",
        ),
        ("Author credit Gini", gini(core.author_metrics["fractional_paper_credit"]), "0=equality; 1=concentration"),
        *[
            (
                f"Credit held by top {percentage:g}% of authors",
                share,
                "% resolved-author fractional publication credit",
            )
            for percentage, share in concentration[
                ["top_author_percent", "credit_share_percent"]
            ].itertuples(index=False, name=None)
        ],
        ("Female-name share, 2013", gender.loc[2013, "female_name_share"], "% classified authorships"),
        ("Female-name share, 2025", gender.loc[2025, "female_name_share"], "% classified authorships"),
        ("Countries represented", int(core.country_metrics["fractional_paper_credit"].fillna(0).gt(0).sum()), "geolocated author-affiliation countries"),
        ("Institutions represented", len(core.institution_metrics), "identified affiliations"),
        ("Top-10 institution share, 2025", institutions.loc[2025, "top_10_share"], "% annual fractional credit"),
        ("Giant component, 2025", 100 * full_network.loc[2025, "giant_fraction"], "% resolved authors"),
        ("Leiden modularity", network.modularity, "fractional-edge weighted"),
    ]
    return pd.DataFrame(values, columns=["statistic", "value", "unit_or_definition"])


def legacy_artifact_crosswalk() -> pd.DataFrame:
    """Document how each substantive legacy output is retained after consolidation."""
    rows = [
        ("05_authors_1_metrics", "author_analytics.xlsx", "supplementary_author_characteristics.xlsx + author_metrics.csv", "retained and expanded; h-index explicitly UKB-specific"),
        ("05_authors_2 / 3_CHECK", "geographic maps (whole, fractional, intensity, org basis)", "Figure 1E + Supplementary Figure 3A-C + country_metrics.csv", "all four geographic views retained with harmonized ISO-3 entities"),
        ("05_geography_4 / 5", "static geography composite", "Figure 1E + Supplementary Figures 3-4", "separated into map-only and quantitative geography evidence"),
        ("05_geography_4 / 5", "geography evolution GIF", "Figure 1C + Supplementary Figure 4D", "cumulative country reach promoted to the headline; period-averaged diversity retained as a nonduplicative static summary"),
        ("05_authors_2 / 3_CHECK", "top-institution FOR bars", "Supplementary Figure 5", "retained with fractional multi-affiliation credit"),
        ("05_institutional_metrics", "institutional_analytics.xlsx", "supplementary_author_characteristics.xlsx + institution_metrics.csv", "retained and corrected for paper/field duplication"),
        ("05_authors_2 / 3_CHECK", "gender trends", "Figure 1D + Supplementary Figure 2", "trend promoted to headline; coverage/roles/fields in supplement"),
        ("05_network_over_time", "network evolution with metrics", "Figure 1A + Supplementary Figure 6 + network_metrics_by_year.csv", "retained as one static topology view, nonredundant structural summaries and a complete annual table"),
        ("05_network_over_time", "annual degree, component and tie-strength metrics", "network_metrics_by_year.csv", "retained for all papers and a hyperauthorship sensitivity"),
        ("05_network_over_time", "Leiden community network and CSV tables", "Figure 1A + Supplementary Figure 6A + community CSVs", "retained using igraph Leiden and fractional edge weights"),
        ("05_network_over_time", "paper and author topic/institution assignments", "paper_for_assignments.csv + author_metrics.csv + network_community_membership.csv", "retained without imposing a first-listed primary FOR category"),
        ("05_network_over_time", "full community L2/L4/institution count and summary tables", "network_community_*_counts.csv + network_community_*_summary.csv", "retained and expanded with country and name-category intersections"),
        ("05_network_over_time", "collapsed community L2/L4/institution tables", "Supplementary Figure 6A + network_collapsed_*_counts.csv + network_collapsed_*_summary.csv", "retained for the 12-community composition display"),
        ("05_authors_1_metrics", "author productivity distribution", "Supplementary Figure 1A + author_productivity_bands.csv", "complete survival curve and mutually exclusive band table retained in the author supplement"),
        ("05_authors_2 / 3_CHECK", "paper_combined", "Figure 1", "replaced by a six-panel synthesis across five analytical domains"),
    ]
    return pd.DataFrame(rows, columns=["legacy_source", "legacy_artifact", "successor", "status"])


def metric_definitions() -> pd.DataFrame:
    rows = [
        ("Resolved author", "An authorship carrying a Dimensions researcher_id; only these IDs are linked longitudinally."),
        ("Unresolved authorship", "An author-paper slot without a researcher_id. It receives a paper-local key and is never merged across papers by name."),
        ("UKB h-index", "Largest h for which an author or institution has h UK Biobank papers cited at least h times in the source snapshot."),
        ("UKB g-index", "Largest g for which the g most-cited UK Biobank papers received at least g squared citations in total."),
        ("Fractional publication credit", "Each paper contributes one unit split equally among parsed authors, then equally among each author's distinct affiliations or countries."),
        ("Author-credit concentration", "Cumulative share of resolved-author fractional publication credit held by authors ranked at or above a stated top-percent threshold."),
        ("Author productivity band", "Mutually exclusive grouping of resolved authors by their number of UK Biobank publications: 1, 2-4, 5-9 or 10+."),
        ("Modal author assignment", "An author's most frequent FOR, institution or country after each paper is split equally across that paper's multiple assignments; ties are deterministic."),
        ("Author home assignment", "Current organization metadata when its GRID record resolves; otherwise the author's modal observed affiliation, with the source flagged."),
        ("Female-name share", "Female-category author-paper pairs divided by Female- plus Male-category pairs; unknown and androgynous names are reported but excluded from this denominator."),
        ("Strict name rule", "Only gender_guesser exact female and male calls are classified; mostly_* and androgynous calls remain Unknown."),
        ("Expanded name rule", "Adds gender_guesser mostly_female/mostly_male calls after Unicode-safe cleanup, leading-initial removal and diacritic-normalized retry."),
        ("Offline ensemble name rule", f"Combines one local vote each from gender_guesser, gender_detector, nomquamgender, names-dataset and gename. Non-conflicting available votes are accepted; conflicts require at least {NG.ENSEMBLE_MIN_MAJORITY_VOTES} winning votes and a {NG.ENSEMBLE_MIN_VOTE_MARGIN}-vote margin."),
        ("Primary name rule", "Offline ensemble classification plus within-researcher propagation when all directly classified name observations agree; every row for identities with conflicting direct calls remains Unknown."),
        ("Name-category limitation", "Female and Male are binary statistical name categories, not observed sex or self-identified gender; Unknown is retained because error and non-classification are culturally patterned."),
        ("Effective entities", "Exponentiated Shannon entropy of annual fractional country or institution credit."),
        ("Institutional concentration", "Share of annual observed institutional fractional credit assigned to that year's leading one or ten institutions."),
        ("Coauthor tie", "An undirected link between two resolved authors appearing on at least one common paper."),
        ("Fractional collaboration strength", "For a paper with n resolved authors, each coauthor edge receives 1/(n-1), so each author's collaboration credit from that paper sums to one."),
        ("Giant component", "Largest connected component of the cumulative resolved-author coauthorship graph."),
        ("Hyperauthorship sensitivity", f"Parallel network calculation excluding papers with more than {HYPERAUTHOR_THRESHOLD} resolved authors."),
        ("Leiden community", f"Weighted Leiden modularity partition of the final fractional coauthor graph (resolution {LEIDEN_RESOLUTION:g}, seed {LEIDEN_SEED})."),
    ]
    return pd.DataFrame(rows, columns=["metric", "definition"])


def analysis_parameters(source, papers) -> pd.DataFrame:
    libraries = NG.offline_library_versions()
    library_versions = ", ".join(
        libraries.apply(lambda row: f"{row['library']}={row['version']}", axis=1)
    )
    rows = [
        ("input", P.raw_path(P.SHOWCASE_PLUS)),
        ("source_records", len(source)),
        ("primary_records", len(papers)),
        ("first_year", FIRST_YEAR),
        ("last_complete_year", LAST_COMPLETE_YEAR),
        ("provisional_years_excluded", ", ".join(map(str, sorted(source.loc[source["year"].gt(LAST_COMPLETE_YEAR), "year"].dropna().unique())))),
        ("author_identity", "Dimensions researcher_id"),
        ("unresolved_identity", "paper-local key; no cross-paper linkage"),
        ("name_gender_dictionary", "gender_guesser strict and expanded rules; offline"),
        ("name_gender_normalization", "Unicode-safe; skip leading initials; retry de-accented token"),
        ("name_gender_inference_mode", "offline package-bundled data only; no APIs or remote caches"),
        ("name_gender_offline_libraries", library_versions),
        ("name_gender_probability_threshold", NG.OFFLINE_MIN_PROBABILITY),
        ("name_gender_nomquam_min_count", NG.NOMQUAM_MIN_COUNT),
        ("name_gender_conflict_min_votes", NG.ENSEMBLE_MIN_MAJORITY_VOTES),
        ("name_gender_conflict_min_margin", NG.ENSEMBLE_MIN_VOTE_MARGIN),
        ("name_gender_identity_linkage", "Dimensions researcher_id; unanimous direct evidence only"),
        ("hyperauthor_threshold", HYPERAUTHOR_THRESHOLD),
        ("leiden_resolution", LEIDEN_RESOLUTION),
        ("leiden_seed", LEIDEN_SEED),
        ("network_layout", "igraph Large Graph Layout (LGL)"),
        ("network_layout_iterations", NETWORK_LAYOUT_ITERATIONS),
        ("network_backbone_repeated_tie_limit", NETWORK_BACKBONE_REPEAT_LIMIT),
        ("map_geometry", "Natural Earth 1:110m Admin 0 Countries"),
    ]
    return pd.DataFrame(rows, columns=["parameter", "value"])


def figure_captions() -> OrderedDict:
    return OrderedDict({
        "figure_01_caption.txt": (
            "Figure 5 | Characteristics and collaborative structure of authors using UK Biobank, "
            f"{FIRST_YEAR}-{LAST_COMPLETE_YEAR}. (A) Component-aware static coauthorship network with "
            "every resolved author plotted. Node colour distinguishes isolates, components of 2-5 authors, "
            "intermediate components and the giant component. All ties outside the giant component are "
            "shown; giant-component edges use a connected actual-edge backbone comprising a breadth-first "
            f"tree and up to {NETWORK_BACKBONE_REPEAT_LIMIT:,} repeated coauthorship ties. The stacked "
            "strip reports the share of resolved authors in each component class. (B) "
            "Cumulative shares of resolved-author fractional publication credit held by authors ranked "
            "in the top 1%, 5%, 10%, 25% and 50% of the credit distribution. (C) Cumulative number of "
            "unique author-affiliation countries observed through each publication year. (D) Annual share "
            "of author-paper pairs assigned to the Female name category among pairs assigned Female or "
            "Male, with 95% Wilson confidence intervals. Name categories are algorithmically inferred "
            "and are not self-identified gender. (E) Geographic distribution of fractional publication "
            "credit by author-affiliation country on a linear navy-to-yellow scale. (F) Annual "
            "concentration of observed institutional credit in the "
            "leading institution and leading ten institutions. "
            "Records from provisional year 2026 were excluded."
        ),
        "supplementary_figure_01_caption.txt": (
            "Supplementary Figure 1 | UK Biobank-specific author productivity and impact. (A) "
            "Complementary cumulative distribution of publication counts, annotated with the median, "
            "interquartile range, upper-tail share and maximum. (B) Distribution of UKB "
            "h-index values, truncated at the 99.5th percentile for legibility and annotated with "
            "distributional summaries. (C) Association between "
            "UKB publication count and UKB h-index, with the Spearman rank correlation reported. (D) Authors with the "
            "ten highest UKB h-index values, with publication and citation counts. Citation measures are snapshot values and "
            "the UKB h-index uses only publications in the analysed corpus."
        ),
        "supplementary_figure_02_caption.txt": (
            "Supplementary Figure 2 | Name-inferred gender composition of UK Biobank authorships. "
            "(A) Annual author-paper counts assigned Female, Male or Unknown/androgynous. (B) Annual "
            "classification coverage under strict dictionary, expanded dictionary, five-library "
            "offline ensemble and primary identity-linked rules. The ensemble uses only package-bundled "
            "local data with explicit disagreement thresholds; the primary rule adds conflict-safe "
            "linkage across records carrying the same Dimensions researcher ID. "
            "(C) Female-name share "
            "among classified names by authorship position, with 95% Wilson intervals. Corresponding "
            "authorship can overlap positional roles. (D) Female-name share in the 12 FOR divisions with "
            "the most classified author-paper memberships. Panel D uses a sequential blue scale for "
            "legibility, not a second variable. These binary categories are probabilistic proxies from "
            "names, not observed sex or self-identified gender; Unknown is retained because both error "
            "and non-classification vary culturally."
        ),
        "supplementary_figure_03_caption.txt": (
            "Supplementary Figure 3 | Geographic distribution of UK Biobank authorship. "
            "(A) Distinct papers by author-affiliation country. (B) Distinct papers by the source "
            "publication's research-organisation countries. Panels A and B use a shared logarithmic "
            "colour scale. (C) Author-paper rows per paper among countries represented on at least "
            "20 papers, shown on a separate logarithmic scale. Countries without eligible records are "
            "shown in light grey."
        ),
        "supplementary_figure_04_caption.txt": (
            "Supplementary Figure 4 | Geographic reach and diversification of UK Biobank authorship. "
            "(A) Countries with the largest shares of geolocated fractional publication credit. "
            "(B) Country-level agreement between distinct paper counts from author-affiliation metadata "
            "and source research-organisation metadata, with a one-to-one reference and Spearman rank "
            "correlation; labelled countries have the largest proportional discrepancies among countries "
            "with at least ten papers under both definitions. Separate callouts identify the shared "
            "minimum-count point and the country with the largest geometric mean count across definitions. "
            "(C) Period-specific shares of geolocated "
            "fractional publication credit for the eight "
            "leading countries across all years; colour uses a logarithmic scale and annotations report "
            "the exact percentage. (D) Mean annual effective "
            "number of countries in four publication-year periods, defined as exponentiated Shannon "
            "entropy of fractional country credit. Period aggregation reduces sensitivity to annual "
            "endpoint lag. Credit is split across authors and each author's distinct affiliation "
            "countries; papers with unresolved geography retain their author credit but do not contribute "
            "to geolocated totals."
        ),
        "supplementary_figure_05_caption.txt": (
            "Supplementary Figure 5 | Institutional participation in UK Biobank research. (A) Leading "
            "institutions by fractional publication credit, partitioned across all reported FOR L2 "
            "divisions with multi-label credit split equally. (B) Cumulative number of observed "
            "institutions. (C) Institutional UKB h-index against fractional publication credit, with the "
            "Spearman rank correlation reported; marker area reflects the number of resolved authors. "
            "Affiliations are fractional across authors "
            "and multiple affiliations."
        ),
        "supplementary_figure_06_caption.txt": (
            "Supplementary Figure 6 | UK Biobank coauthorship structure. (A) Modal FOR composition of "
            "the 12 largest Leiden communities in the full resolved-author network. (B) Newly resolved "
            "authors entering the cumulative network each year. (C) Association between UK Biobank "
            "publication count and the number of distinct coauthors among connected authors. Hexagon "
            "colour records author density on a logarithmic scale. (D) Distribution of unique coauthor ties by number of shared "
            "papers. (E) Rank-size distribution of non-giant connected components, with the giant-component "
            "endpoint reported separately to avoid compressing the smaller-component distribution. "
            f"(F) Percentage of each 2025 endpoint retained after excluding papers with more than "
            f"{HYPERAUTHOR_THRESHOLD} resolved authors. Complete annual trajectories remain available in "
            "network_metrics_by_year.csv."
        ),
    })


def methods_text(source, papers, core: CoreTables, network: NetworkTables) -> str:
    missing_papers = len(papers) - core.authorships["paper_id"].nunique()
    unresolved = int((~core.authorships["identity_resolved"]).sum())
    unknown = int(core.authorships["name_gender"].eq("Unknown").sum())
    coverage = NG.inference_coverage(core.authorships).set_index("stage")
    offline_assignments = int(
        core.authorships["name_gender_direct"].isin(NG.CLASSIFIED_CATEGORIES).sum()
    )
    accepted_conflicts = int(
        core.authorships["name_gender_method"]
        .eq("offline_ensemble_majority")
        .sum()
    )
    linked_assignments = int(
        core.authorships["name_gender_method"].eq(
            "researcher_identity_consensus"
        ).sum()
    )
    identity_conflicts = int(
        core.authorships.loc[
            core.authorships["name_gender_identity_conflict"], "researcher_id"
        ].nunique()
    )
    library_versions = ", ".join(
        NG.offline_library_versions().apply(
            lambda row: f"{row['library']} {row['version']}", axis=1
        )
    )
    return (
        "Author-characteristics analysis. The analysis used the Showcase+ all-endpoints-wide "
        f"publication parquet ({len(source):,} records at the source snapshot). Primary analyses "
        f"were restricted to complete publication years {FIRST_YEAR}-{LAST_COMPLETE_YEAR} "
        f"({len(papers):,} papers); provisional 2026 records were retained only for auditing. "
        "Nested author, affiliation, research-organization and Fields of Research (FOR 2020) "
        "records were parsed from their JSON representations. A tidy author-paper table was built "
        "from the authors field and deduplicated by publication ID and author identifier. "
        f"{missing_papers:,} complete-year papers had no parsed authors and therefore did not enter "
        "author-based analyses. Dimensions researcher_id values defined longitudinal identities. "
        f"The {unresolved:,} author-paper pairs lacking a researcher_id were retained in paper-level "
        "composition and affiliation denominators under paper-local keys, but were never linked "
        "across papers by name and were excluded from person-level impact and network metrics. "
        "UKB h-, g- and i10-indices used citations only to papers in this UK Biobank corpus; citation "
        "and Altmetric values were snapshot measures. Each paper contributed one fractional unit, "
        "split equally among parsed authors and then among each author's distinct affiliations or "
        "affiliation countries. When no paper-specific affiliation was listed, a resolvable current "
        "organization ID was used as a flagged backfill; unresolved affiliation credit was not "
        "redistributed. FOR is an unordered multi-label taxonomy: all L2/L4 labels were retained, and "
        "credit used in field-stratified figures was divided equally over labels. Author-level modal "
        "assignments conserved one unit per observed author-paper across multiple labels, institutions "
        "or countries. Author home institutions used resolvable current-organization metadata with the "
        "modal observed affiliation as an explicitly flagged fallback. Country identifiers "
        "were harmonized to ISO-3 and mapped to Natural Earth 1:110m Admin 0 geometry. "
        "Name-based categories were inferred through one shared, auditable and entirely offline "
        "pipeline used by every active project analysis. Given-name strings were cleaned without "
        "restricting Unicode alphabets; leading one-letter initials were skipped, and an "
        "accent-normalized retry was made when needed. The strict gender_guesser rule retained exact "
        "female/male calls, whereas the expanded rule grouped mostly_female/mostly_male as "
        f"Female/Male and classified {int(coverage.loc['Expanded dictionary', 'classified_author_paper_pairs']):,} "
        "author-paper pairs. The direct rule combined one vote from each installed local-data package: "
        f"{library_versions}. nomquamgender and names-dataset votes required probability >= "
        f"{NG.OFFLINE_MIN_PROBABILITY:.2f}; nomquamgender additionally required at least "
        f"{NG.NOMQUAM_MIN_COUNT:,} underlying records. Non-conflicting available votes were accepted. "
        f"When packages disagreed, classification required at least {NG.ENSEMBLE_MIN_MAJORITY_VOTES} "
        f"winning votes and a {NG.ENSEMBLE_MIN_VOTE_MARGIN}-vote margin; otherwise the name remained "
        "Unknown. Each package contributed at most one vote, including one internally consensus-checked "
        "vote across gender-detector's UK, US, Argentina and Uruguay tables. The ensemble classified "
        f"{offline_assignments:,} author-paper pairs directly, including {accepted_conflicts:,} assignments "
        "accepted under the conflicting-vote majority rule. No API client, network request or remotely "
        "populated name cache was used. For Dimensions-resolved researchers, an unknown row was "
        "assigned from other records only when every direct Female/Male call for that researcher agreed "
        f"({linked_assignments:,} rows recovered). All rows for the {identity_conflicts:,} researchers "
        "with conflicting direct calls were conservatively returned to Unknown. Unresolved library "
        f"conflicts, low-confidence, low-support, initial-only and missing cases remained Unknown ({unknown:,} "
        "author-paper pairs). Female-name shares used Female plus Male as the denominator, while Unknown "
        "was shown separately. These binary statistical name categories are neither observed sex nor "
        "self-identified gender. Name-based inference has culturally patterned error and non-classification "
        "(Lockhart, King and Munsch, 2023; doi:10.1038/s41562-023-01587-9); strict/expanded coverage, "
        "method counts and unresolved query tables are therefore supplied as sensitivity and audit "
        "outputs. Coauthorship "
        "networks included Dimensions-resolved authors. An undirected edge linked authors sharing at "
        "least one paper. Raw tie strength counted shared papers; fractional strength assigned each "
        "edge on an n-author paper weight 1/(n-1), so each author's collaboration credit from that "
        "paper summed to one. Cumulative annual metrics retained isolates. A parallel sensitivity "
        f"analysis excluded papers with more than {HYPERAUTHOR_THRESHOLD} resolved authors. Leiden "
        f"communities were estimated on the final fractional graph at resolution {LEIDEN_RESOLUTION:g} "
        f"with seed {LEIDEN_SEED}; the reported modularity was {network.modularity:.3f}. The main figure "
        "plotted every resolved author and every non-giant tie. For visualization only, the giant "
        "component was reduced to a connected actual-edge backbone comprising a breadth-first spanning "
        f"tree and up to {NETWORK_BACKBONE_REPEAT_LIMIT:,} repeated coauthorship ties, then positioned "
        f"with igraph's Large Graph Layout ({NETWORK_LAYOUT_ITERATIONS} iterations; seed {LEIDEN_SEED}). "
        "All reported network metrics used the complete graph. The main figure "
        "uses a compact synthesis across five domains; each supplementary figure is restricted to "
        "one domain. Complete community-by-FOR, institution, country and name-category distributions "
        "are supplied as count, share and rank tables rather than inferred from the plotted labels."
    )


def export_analysis_artifacts(
    registry,
    source: pd.DataFrame,
    papers: pd.DataFrame,
    core: CoreTables,
    network: NetworkTables,
) -> dict:
    """Export all analytical tables, workbook sheets, captions, methods, and crosswalk."""
    author_table = merge_author_network_metrics(core.author_metrics, network)
    audit = quality_audit(source, papers, core)
    checks = validation_checks(source, papers, core, network)
    headline = headline_statistics(core, network)
    concentration = author_credit_concentration(core.author_metrics)
    productivity = author_productivity_bands(core.author_metrics)
    gender_coverage = NG.inference_coverage(core.authorships)
    gender_coverage_by_year = NG.inference_coverage(core.authorships, "year")
    gender_methods = NG.inference_method_counts(core.authorships)
    gender_library_audit = NG.offline_library_audit(core.authorships)
    gender_library_versions = NG.offline_library_versions()
    unresolved_name_queries = NG.unresolved_query_queue(core.authorships)
    definitions = metric_definitions()
    parameters = analysis_parameters(source, papers)
    crosswalk = legacy_artifact_crosswalk()
    paper_for = papers[["id", "year", "title", "for_l2", "for_l4"]].rename(
        columns={"id": "paper_id"}
    )
    for column in ["for_l2", "for_l4"]:
        paper_for[column] = paper_for[column].apply(
            lambda values: "; ".join(map(str, values)) if values else "Unclassified"
        )
    membership_columns = [
        "researcher_id",
        "full_name",
        "community",
        "modal_for_l2",
        "modal_for_l4",
        "modal_institution",
        "modal_country",
        "name_gender",
    ]
    community_tables = community_intersection_tables(network)
    network_summaries = network_figure_tables(network)

    tables = OrderedDict({
        "author_metrics.csv": author_table,
        "author_credit_concentration.csv": concentration,
        "author_productivity_bands.csv": productivity,
        "paper_for_assignments.csv": paper_for,
        "institution_metrics.csv": core.institution_metrics,
        "country_metrics.csv": core.country_metrics,
        "gender_by_year.csv": core.gender_by_year,
        "gender_by_authorship_role.csv": core.gender_by_role,
        "gender_by_for_division.csv": core.gender_by_field,
        "gender_by_country.csv": core.gender_by_country,
        "gender_by_institution.csv": core.gender_by_institution,
        "name_gender_inference_coverage.csv": gender_coverage,
        "name_gender_inference_coverage_by_year.csv": gender_coverage_by_year,
        "name_gender_inference_method_counts.csv": gender_methods,
        "name_gender_offline_library_audit.csv": gender_library_audit,
        "name_gender_offline_library_versions.csv": gender_library_versions,
        "name_gender_unresolved_queries.csv": unresolved_name_queries,
        "country_metrics_by_year.csv": core.country_by_year,
        "institution_metrics_by_year.csv": core.institution_by_year,
        "network_metrics_by_year.csv": network.metrics_by_year,
        "network_author_metrics.csv": network.author_metrics,
        "network_community_membership.csv": network.community_membership[
            membership_columns
        ],
        "network_community_summary.csv": network.community_summary,
        "network_collapsed_nodes.csv": network.collapsed_nodes,
        "network_collapsed_edges.csv": network.collapsed_edges,
        "headline_summary_statistics.csv": headline,
        "data_quality_audit.csv": audit,
        "analysis_validation_checks.csv": checks,
        "metric_definitions.csv": definitions,
        "analysis_parameters.csv": parameters,
        "legacy_artifact_crosswalk.csv": crosswalk,
    })
    tables.update(community_tables)
    tables.update(network_summaries)
    saved_tables = {name: registry.save_table(frame, name) for name, frame in tables.items()}

    workbook_sheets = OrderedDict({
        "Author metrics": author_table,
        "Author concentration": concentration,
        "Author productivity bands": productivity,
        "Paper FOR assignments": paper_for,
        "Institution metrics": core.institution_metrics,
        "Country metrics": core.country_metrics,
        "Gender by year": core.gender_by_year,
        "Gender by role": core.gender_by_role,
        "Gender by FOR": core.gender_by_field,
        "Gender by country": core.gender_by_country,
        "Gender by institution": core.gender_by_institution,
        "Name inference coverage": gender_coverage,
        "Name coverage by year": gender_coverage_by_year,
        "Name inference methods": gender_methods,
        "Unresolved name queries": unresolved_name_queries,
        "Country by year": core.country_by_year,
        "Institution by year": core.institution_by_year,
        "Network by year": network.metrics_by_year,
        "Community summary": network.community_summary,
        "Community membership": network.community_membership[membership_columns],
        "Community L2 counts": community_tables[
            "network_community_for_l2_counts.csv"
        ],
        "Community L4 counts": community_tables[
            "network_community_for_l4_counts.csv"
        ],
        "Community inst counts": community_tables[
            "network_community_institution_counts.csv"
        ],
        "Community country counts": community_tables[
            "network_community_country_counts.csv"
        ],
        "Community gender counts": community_tables[
            "network_community_name_gender_counts.csv"
        ],
        "Network new authors": network_summaries[
            "network_new_authors_by_year.csv"
        ],
        "Network degree dist": network_summaries[
            "network_degree_distribution.csv"
        ],
        "Network tie strengths": network_summaries[
            "network_tie_strength_distribution.csv"
        ],
        "Network component sizes": network_summaries[
            "network_component_rank_size.csv"
        ],
        "Network sensitivity": network_summaries[
            "network_hyperauthorship_sensitivity.csv"
        ],
        "Headline statistics": headline,
        "Quality audit": audit,
        "Validation checks": checks,
        "Metric definitions": definitions,
        "Parameters": parameters,
        "Legacy crosswalk": crosswalk,
    })
    workbook = registry.save_workbook(
        workbook_sheets, "supplementary_author_characteristics.xlsx"
    )
    for filename, caption in figure_captions().items():
        registry.save_text(caption + "\n", filename)
    methods = registry.save_text(
        methods_text(source, papers, core, network) + "\n",
        "methods_author_characteristics.txt",
    )
    return {
        "author_table": author_table,
        "concentration": concentration,
        "productivity": productivity,
        "gender_coverage": gender_coverage,
        "gender_methods": gender_methods,
        "unresolved_name_queries": unresolved_name_queries,
        "audit": audit,
        "checks": checks,
        "headline": headline,
        "saved_tables": saved_tables,
        "workbook": workbook,
        "methods": methods,
    }
