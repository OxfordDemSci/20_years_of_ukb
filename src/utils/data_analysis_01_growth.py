"""Reusable computations and plotting primitives for the UK Biobank growth analysis."""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_rgb, to_rgba
from matplotlib.patches import Rectangle

from .shared_showcase import load_showcase
from .shared_style import compact_count as shared_compact_count
from .shared_style import marker_area as shared_marker_area
from .shared_style import panel_label as shared_panel_label
from .shared_style import sequential_colormap

ENDPOINT_SPECS = OrderedDict({
    "clinical_trials": {
        "label": "Clinical trials",
        "id_col": "clinical_trials__linked_ids",
        "event_col": "clinical_trials__start_date",
        "record_col": "clinical_trials__n_records",
        "event_label": "Trial starts",
    },
    "patents": {
        "label": "Patents",
        "id_col": "patents__linked_ids",
        "event_col": "patents__publication_year",
        "record_col": "patents__n_records",
        "event_label": "Patent publications",
    },
    "policy_documents": {
        "label": "Policy documents",
        "id_col": "policy_documents__linked_ids",
        "event_col": "policy_documents__year",
        "record_col": "policy_documents__n_records",
        "event_label": "Policy documents",
    },
    "grants": {
        "label": "Grants",
        "id_col": "grants__linked_ids",
        "event_col": "grants__start_year",
        "record_col": "grants__n_records",
        "event_label": "Grant starts",
    },
    "datasets": {
        "label": "Datasets",
        "id_col": "datasets__linked_ids",
        "event_col": "datasets__year",
        "record_col": "datasets__n_records",
        "event_label": "Dataset records",
    },
})

BASE_COLUMNS = [
    "id", "title", "date", "year", "type", "document_type",
    "authors_count", "times_cited", "altmetric", "altmetric_id",
    "open_access", "research_org_types", "research_org_countries",
]

NON_ACADEMIC_SECTORS = ("Company", "Government", "Healthcare", "Nonprofit")


def load_growth_papers(endpoint_specs=ENDPOINT_SPECS):
    """Load and normalize only the showcase columns required by this analysis."""
    endpoint_columns = [
        column
        for spec in endpoint_specs.values()
        for column in (spec["id_col"], spec["event_col"], spec["record_col"])
    ]
    parse_columns = [
        "open_access", "research_org_types", "research_org_countries",
        *[spec["id_col"] for spec in endpoint_specs.values()],
        *[spec["event_col"] for spec in endpoint_specs.values()],
    ]
    papers = load_showcase(
        columns=BASE_COLUMNS + endpoint_columns,
        parse=parse_columns,
    )
    papers["publication_date"] = pd.to_datetime(papers["date"], errors="coerce")
    papers["year"] = pd.to_numeric(papers["year"], errors="coerce").astype("Int64")
    papers["altmetric"] = pd.to_numeric(
        papers["altmetric"], errors="coerce"
    ).fillna(0.0)
    papers["times_cited"] = pd.to_numeric(
        papers["times_cited"], errors="coerce"
    ).fillna(0)
    papers["authors_count"] = pd.to_numeric(
        papers["authors_count"], errors="coerce"
    )
    return papers


def _normalized_strings(values):
    return {str(value).strip() for value in values if str(value).strip()}


def _country_ids(values):
    identifiers = set()
    for value in values:
        identifier = value.get("id") or value.get("name") if isinstance(value, dict) else value
        if identifier:
            identifiers.add(str(identifier).strip())
    return identifiers


def add_publication_indicators(
    papers,
    data_cutoff,
    endpoint_specs=ENDPOINT_SPECS,
    non_academic_sectors=NON_ACADEMIC_SECTORS,
):
    """Derive collaboration, attention, access, type, and endpoint flags."""
    papers = papers.copy()
    papers["org_type_set"] = papers["research_org_types"].apply(_normalized_strings)
    papers["country_set"] = papers["research_org_countries"].apply(_country_ids)
    papers["has_education_org"] = papers["org_type_set"].apply(
        lambda values: "Education" in values
    )
    papers["has_nonacademic_org"] = papers["org_type_set"].apply(
        lambda values: bool(values.intersection(non_academic_sectors))
    )
    papers["cross_sector_collaboration"] = (
        papers["has_education_org"] & papers["has_nonacademic_org"]
    )
    for sector in non_academic_sectors:
        papers[f"cross_sector_{sector.lower()}"] = (
            papers["has_education_org"]
            & papers["org_type_set"].apply(
                lambda values, sector=sector: sector in values
            )
        )

    papers["international_collaboration"] = papers["country_set"].apply(len).ge(2)
    papers["has_altmetric_attention"] = papers["altmetric"].gt(0)
    papers["is_open_access"] = papers["open_access"].apply(
        lambda values: "oa_all" in values
    )
    papers["is_preprint"] = papers["type"].eq("preprint")
    papers["future_dated"] = papers["publication_date"].gt(data_cutoff)
    for key, spec in endpoint_specs.items():
        papers[f"has_{key}"] = papers[spec["id_col"]].apply(bool)
    return papers


def _to_event_year(value):
    if value is None:
        return np.nan
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and not pd.isna(value):
        return int(value)
    text = str(value).strip()
    if not text:
        return np.nan
    try:
        return int(float(text))
    except ValueError:
        parsed = pd.to_datetime(text, errors="coerce")
        return np.nan if pd.isna(parsed) else int(parsed.year)


def _consensus_year(values):
    valid = pd.Series(values).dropna().astype(int)
    if valid.empty:
        return np.nan
    counts = valid.value_counts()
    return int(min(counts[counts.eq(counts.max())].index))


def reconstruct_endpoint_links(frame, key, spec):
    """Create paper-endpoint links, unique endpoint records, and alignment audit data."""
    rows = []
    mismatch_rows = 0
    missing_event_year_links = 0
    selected = frame[["id", "year", spec["id_col"], spec["event_col"]]]
    for paper_id, paper_year, endpoint_ids, event_values in selected.itertuples(
        index=False, name=None
    ):
        endpoint_ids = endpoint_ids or []
        event_values = event_values or []
        if len(endpoint_ids) != len(event_values):
            mismatch_rows += 1
        for position, endpoint_id in enumerate(endpoint_ids):
            if endpoint_id is None or not str(endpoint_id).strip():
                continue
            event_year = (
                _to_event_year(event_values[position])
                if position < len(event_values)
                else np.nan
            )
            missing_event_year_links += int(pd.isna(event_year))
            rows.append({
                "paper_id": str(paper_id),
                "paper_year": int(paper_year),
                "endpoint_id": str(endpoint_id),
                "event_year": event_year,
            })

    long = pd.DataFrame(
        rows,
        columns=["paper_id", "paper_year", "endpoint_id", "event_year"],
    )
    grouped = long.groupby("endpoint_id", sort=True)
    unique = grouped.agg(
        first_link_year=("paper_year", "min"),
        last_link_year=("paper_year", "max"),
        linked_papers=("paper_id", "nunique"),
        event_year_nunique=("event_year", "nunique"),
    )
    unique["event_year"] = grouped["event_year"].apply(_consensus_year)
    unique = unique.reset_index()

    recorded_records = int(
        pd.to_numeric(frame[spec["record_col"]], errors="coerce").fillna(0).sum()
    )
    audit = {
        "endpoint": key,
        "label": spec["label"],
        "parsed_link_rows": len(long),
        "recorded_n_records_sum": recorded_records,
        "record_count_difference": len(long) - recorded_records,
        "unique_endpoint_ids": unique["endpoint_id"].nunique(),
        "alignment_mismatch_paper_rows": mismatch_rows,
        "links_missing_event_year": missing_event_year_links,
        "unique_ids_with_event_year": unique["event_year"].notna().sum(),
        "unique_ids_with_conflicting_event_year": unique[
            "event_year_nunique"
        ].gt(1).sum(),
    }
    return long, unique, audit


def reconstruct_all_endpoints(frame, endpoint_specs=ENDPOINT_SPECS):
    endpoint_links = {}
    endpoint_unique = {}
    endpoint_audits = []
    for key, spec in endpoint_specs.items():
        long, unique, audit = reconstruct_endpoint_links(frame, key, spec)
        endpoint_links[key] = long
        endpoint_unique[key] = unique
        endpoint_audits.append(audit)
    return endpoint_links, endpoint_unique, endpoint_audits


def build_quality_audit(papers, endpoint_audits, data_cutoff):
    date_year_mismatch = (
        papers["publication_date"].notna()
        & papers["year"].notna()
        & papers["publication_date"].dt.year.ne(papers["year"])
    )
    rows = [
        {"check": "input_rows", "value": len(papers), "interpretation": "All parquet records"},
        {
            "check": "unique_publication_ids",
            "value": papers["id"].nunique(),
            "interpretation": "Expected to equal input rows",
        },
        {
            "check": "duplicate_publication_ids",
            "value": papers["id"].duplicated().sum(),
            "interpretation": "Expected zero",
        },
        {
            "check": "missing_publication_year",
            "value": papers["year"].isna().sum(),
            "interpretation": "Expected zero",
        },
        {
            "check": "missing_publication_date",
            "value": papers["publication_date"].isna().sum(),
            "interpretation": "Expected zero",
        },
        {
            "check": "date_year_mismatches",
            "value": date_year_mismatch.sum(),
            "interpretation": "Date year differs from year field",
        },
        {
            "check": "records_after_data_cutoff",
            "value": papers["future_dated"].sum(),
            "interpretation": f"Future-dated; excluded with {data_cutoff.year}",
        },
        {
            "check": f"records_in_incomplete_{data_cutoff.year}",
            "value": papers["year"].eq(data_cutoff.year).sum(),
            "interpretation": "Provisional year; excluded from primary analyses",
        },
    ]
    rows.extend(
        {
            "check": f"{audit['endpoint']}_array_alignment_mismatches",
            "value": audit["alignment_mismatch_paper_rows"],
            "interpretation": "ID and event-date arrays of different lengths",
        }
        for audit in endpoint_audits
    )
    return pd.DataFrame(rows)


def complete_year_corpus(papers, last_complete_year):
    complete = papers.loc[
        papers["year"].between(int(papers["year"].min()), last_complete_year)
    ].copy()
    complete["year"] = complete["year"].astype(int)
    return complete


def build_annual_outputs(
    complete,
    endpoint_unique,
    last_complete_year,
    endpoint_specs=ENDPOINT_SPECS,
):
    """Build annual paper, collaboration, endpoint, and event-year tables."""
    first_year = int(complete["year"].min())
    years = pd.Index(range(first_year, last_complete_year + 1), name="year")
    annual = pd.DataFrame(index=years)
    grouped = complete.groupby("year")

    def annual_sum(column):
        return grouped[column].sum().reindex(years, fill_value=0)

    annual["annual_papers"] = grouped.size().reindex(years, fill_value=0)
    annual["cumulative_papers"] = annual["annual_papers"].cumsum()
    annual["yoy_publication_growth_pct"] = annual["annual_papers"].pct_change().mul(100)
    annual["annual_cross_sector_papers"] = annual_sum("cross_sector_collaboration")
    annual["cumulative_cross_sector_papers"] = annual[
        "annual_cross_sector_papers"
    ].cumsum()
    annual["annual_international_papers"] = annual_sum("international_collaboration")
    annual["cumulative_international_papers"] = annual[
        "annual_international_papers"
    ].cumsum()
    annual["annual_altmetric_papers"] = annual_sum("has_altmetric_attention")
    annual["cumulative_altmetric_papers"] = annual["annual_altmetric_papers"].cumsum()
    annual["annual_altmetric_score_snapshot"] = annual_sum("altmetric")
    annual["cumulative_altmetric_score_snapshot"] = annual[
        "annual_altmetric_score_snapshot"
    ].cumsum()
    annual["annual_citations_snapshot"] = annual_sum("times_cited")
    annual["cumulative_citations_snapshot"] = annual[
        "annual_citations_snapshot"
    ].cumsum()
    annual["annual_open_access_papers"] = annual_sum("is_open_access")
    annual["annual_preprints"] = annual_sum("is_preprint")

    for column in [
        "cross_sector_collaboration",
        "international_collaboration",
        "has_altmetric_attention",
        "is_open_access",
        "is_preprint",
    ]:
        annual[f"share_{column}_pct"] = annual_sum(column).div(
            annual["annual_papers"]
        ).mul(100)

    annual["median_authors"] = grouped["authors_count"].median().reindex(years)
    annual["authors_q25"] = grouped["authors_count"].quantile(0.25).reindex(years)
    annual["authors_q75"] = grouped["authors_count"].quantile(0.75).reindex(years)
    annual["mean_authors"] = grouped["authors_count"].mean().reindex(years)

    for key in endpoint_specs:
        unique = endpoint_unique[key]
        new_records = (
            unique.loc[
                unique["first_link_year"].le(last_complete_year), "first_link_year"
            ]
            .value_counts()
            .reindex(years, fill_value=0)
            .sort_index()
        )
        annual[f"new_unique_{key}"] = new_records
        annual[f"cumulative_unique_{key}"] = new_records.cumsum()
        linked_papers = annual_sum(f"has_{key}")
        annual[f"annual_papers_linked_{key}"] = linked_papers
        annual[f"cumulative_papers_linked_{key}"] = linked_papers.cumsum()
        annual[f"share_papers_linked_{key}_pct"] = linked_papers.div(
            annual["annual_papers"]
        ).mul(100)

    sector_by_year = pd.DataFrame({"year": years})
    for sector in ["company", "government", "healthcare", "nonprofit"]:
        values = annual_sum(f"cross_sector_{sector}")
        sector_by_year[f"annual_education_{sector}"] = values.to_numpy()
        sector_by_year[f"cumulative_education_{sector}"] = values.cumsum().to_numpy()
        sector_by_year[f"share_education_{sector}_pct"] = (
            values.div(annual["annual_papers"]).mul(100).to_numpy()
        )

    event_year_metrics = pd.DataFrame({"year": years})
    event_year_baselines = []
    for key in endpoint_specs:
        event_years = endpoint_unique[key]["event_year"].dropna().astype(int)
        baseline = int(event_years.lt(first_year).sum())
        yearly = event_years.value_counts().reindex(years, fill_value=0).sort_index()
        event_year_metrics[f"new_{key}"] = yearly.to_numpy()
        event_year_metrics[f"cumulative_{key}"] = (
            baseline + yearly.cumsum()
        ).to_numpy()
        event_year_baselines.append({
            "endpoint": key,
            "events_before_primary_window": baseline,
            "events_after_last_complete_year": int(
                event_years.gt(last_complete_year).sum()
            ),
            "events_missing_year": int(endpoint_unique[key]["event_year"].isna().sum()),
        })

    return (
        annual,
        sector_by_year,
        event_year_metrics,
        pd.DataFrame(event_year_baselines),
    )


def top_share(values, fraction=0.01):
    values = pd.Series(values).fillna(0).clip(lower=0)
    total = values.sum()
    if total <= 0:
        return np.nan
    n_top = max(1, math.ceil(len(values) * fraction))
    return float(values.nlargest(n_top).sum() / total)


def growth_statistics(annual, baseline, latest):
    previous = latest - 1
    latest_growth = annual.loc[latest, "annual_papers"] / annual.loc[
        previous, "annual_papers"
    ] - 1
    cagr = (
        annual.loc[latest, "annual_papers"]
        / annual.loc[baseline, "annual_papers"]
    ) ** (1 / (latest - baseline)) - 1
    fit_years = np.arange(baseline, latest + 1)
    fit_counts = annual.loc[fit_years, "annual_papers"].to_numpy()
    slope, intercept = np.polyfit(fit_years - baseline, np.log(fit_counts), 1)
    fitted_log = intercept + slope * (fit_years - baseline)
    ss_res = np.square(np.log(fit_counts) - fitted_log).sum()
    ss_tot = np.square(np.log(fit_counts) - np.log(fit_counts).mean()).sum()
    return {
        "previous": previous,
        "latest_growth": latest_growth,
        "cagr": cagr,
        "loglinear_growth": np.expm1(slope),
        "loglinear_r2": 1 - ss_res / ss_tot,
        "doubling_years": np.log(2) / slope,
    }


def publication_milestones(
    complete,
    thresholds=(100, 1_000, 5_000, 10_000, 20_000),
):
    sorted_papers = complete.sort_values(["publication_date", "id"]).reset_index(drop=True)
    sorted_papers["cumulative_papers"] = np.arange(1, len(sorted_papers) + 1)
    rows = []
    for threshold in thresholds:
        reached = sorted_papers.loc[
            sorted_papers["cumulative_papers"].ge(threshold)
        ].head(1)
        if not reached.empty:
            rows.append({
                "cumulative_papers": threshold,
                "date_reached": reached.iloc[0]["publication_date"].date().isoformat(),
                "year": int(reached.iloc[0]["year"]),
            })
    return pd.DataFrame(rows)


def summarize_endpoints(
    complete,
    endpoint_links,
    endpoint_unique,
    last_complete_year,
    endpoint_specs=ENDPOINT_SPECS,
):
    first_year = int(complete["year"].min())
    rows = []
    for key, spec in endpoint_specs.items():
        unique = endpoint_unique[key]
        in_window = unique.loc[unique["first_link_year"].le(last_complete_year)]
        long = endpoint_links[key]
        rows.append({
            "endpoint": key,
            "label": spec["label"],
            f"linked_papers_{first_year}_{last_complete_year}": int(
                complete[f"has_{key}"].sum()
            ),
            "link_rows_from_complete_year_papers": int(
                long["paper_year"].le(last_complete_year).sum()
            ),
            f"unique_records_first_linked_by_{last_complete_year}": len(in_window),
            "event_year_coverage_pct": in_window["event_year"].notna().mean() * 100,
            "records_with_conflicting_event_year": int(
                in_window["event_year_nunique"].gt(1).sum()
            ),
            "median_linked_papers_per_record": in_window["linked_papers"].median(),
            "earliest_first_link_year": int(in_window["first_link_year"].min()),
            "latest_first_link_year": int(in_window["first_link_year"].max()),
        })
    return pd.DataFrame(rows)


def headline_summary(
    papers,
    complete,
    annual,
    growth_stats,
    baseline,
    latest,
    data_cutoff,
):
    first_year = int(complete["year"].min())
    previous = growth_stats["previous"]
    spearman = complete[["times_cited", "altmetric"]].corr(
        method="spearman"
    ).iloc[0, 1]
    rows = [
        {
            "metric": "All source records",
            "value": f"{len(papers):,}",
            "definition": f"Includes provisional {data_cutoff.year} records",
        },
        {
            "metric": "Papers in complete years",
            "value": f"{len(complete):,}",
            "definition": (
                f"Unique papers published {first_year}\N{EN DASH}{latest}"
            ),
        },
        {
            "metric": f"Papers published in {latest}",
            "value": f"{int(annual.loc[latest, 'annual_papers']):,}",
            "definition": "Latest complete publication year",
        },
        {
            "metric": f"Year-on-year growth in {latest}",
            "value": f"{growth_stats['latest_growth']:.1%}",
            "definition": f"Relative to {previous}",
        },
        {
            "metric": f"Publication CAGR, {baseline}\N{EN DASH}{latest}",
            "value": f"{growth_stats['cagr']:.1%}",
            "definition": "Compound annual growth in annual paper output",
        },
        {
            "metric": f"Log-linear annual growth, {baseline}\N{EN DASH}{latest}",
            "value": f"{growth_stats['loglinear_growth']:.1%}",
            "definition": f"OLS on log annual counts; R\N{SUPERSCRIPT TWO}={growth_stats['loglinear_r2']:.3f}",
        },
        {
            "metric": "Estimated publication doubling time",
            "value": f"{growth_stats['doubling_years']:.2f} years",
            "definition": f"From {baseline}\N{EN DASH}{latest} log-linear trend",
        },
        {
            "metric": "Cross-sector collaboration papers",
            "value": f"{int(complete['cross_sector_collaboration'].sum()):,}",
            "definition": (
                "Education plus Company, Government, Healthcare, or Nonprofit"
            ),
        },
        {
            "metric": "International collaboration papers",
            "value": f"{int(complete['international_collaboration'].sum()):,}",
            "definition": "At least two research-organization countries",
        },
        {
            "metric": "Papers with Altmetric attention",
            "value": f"{int(complete['has_altmetric_attention'].sum()):,}",
            "definition": "Positive aggregate Altmetric score",
        },
        {
            "metric": "Aggregate Altmetric score",
            "value": f"{complete['altmetric'].sum():,.0f}",
            "definition": "Current snapshot across complete-year papers",
        },
        {
            "metric": "Altmetric score held by top 1% of papers",
            "value": f"{top_share(complete['altmetric']):.1%}",
            "definition": "Concentration of current aggregate score",
        },
        {
            "metric": "Total citations",
            "value": f"{complete['times_cited'].sum():,.0f}",
            "definition": "Current Dimensions snapshot",
        },
        {
            "metric": "Citations held by top 1% of papers",
            "value": f"{top_share(complete['times_cited']):.1%}",
            "definition": "Concentration of current citation counts",
        },
        {
            "metric": "Citation\N{EN DASH}Altmetric Spearman correlation",
            "value": f"{spearman:.3f}",
            "definition": "Paper-level rank correlation, zero values retained",
        },
    ]
    return pd.DataFrame(rows)


def metric_definitions_table():
    return pd.DataFrame([
        {
            "metric": "Papers",
            "availability": "Available",
            "operational_definition": "Unique Dimensions publication IDs by publication year",
            "temporal_attribution": "Publication year",
        },
        {
            "metric": "Clinical trials",
            "availability": "Available",
            "operational_definition": "Paper has at least one clinical_trials__linked_ids; unique IDs retained for supplements",
            "temporal_attribution": "Publication year for paper indicator; earliest linked paper or trial start year in supplements",
        },
        {
            "metric": "Patents",
            "availability": "Available",
            "operational_definition": "Paper has at least one patents__linked_ids; unique IDs retained for supplements",
            "temporal_attribution": "Publication year for paper indicator; earliest linked paper or patent publication year in supplements",
        },
        {
            "metric": "Policy documents",
            "availability": "Available",
            "operational_definition": "Paper has at least one policy_documents__linked_ids; unique IDs retained for supplements",
            "temporal_attribution": "Publication year for paper indicator; earliest linked paper or policy year in supplements",
        },
        {
            "metric": "Grants",
            "availability": "Available",
            "operational_definition": "Paper has at least one grants__linked_ids; unique IDs retained for supplements",
            "temporal_attribution": "Earliest linked paper year; grant start year in supplement",
        },
        {
            "metric": "Datasets",
            "availability": "Available",
            "operational_definition": "Paper has at least one datasets__linked_ids; unique IDs retained for supplements",
            "temporal_attribution": "Publication year for paper indicator; earliest linked paper or dataset year in supplements",
        },
        {
            "metric": "Cross-sector collaboration",
            "availability": "Available",
            "operational_definition": "Education plus Company, Government, Healthcare, or Nonprofit organization type",
            "temporal_attribution": "Publication year",
        },
        {
            "metric": "International collaboration",
            "availability": "Available",
            "operational_definition": "At least two research-organization countries",
            "temporal_attribution": "Publication year",
        },
        {
            "metric": "Open access",
            "availability": "Available",
            "operational_definition": "open_access array contains oa_all",
            "temporal_attribution": "Publication year; access status is a current snapshot",
        },
        {
            "metric": "Preprints",
            "availability": "Available",
            "operational_definition": "Dimensions publication type equals preprint",
            "temporal_attribution": "Publication year",
        },
        {
            "metric": "Altmetric attention",
            "availability": "Available as aggregate score",
            "operational_definition": "Positive altmetric score; score is a current snapshot",
            "temporal_attribution": "Publication year, not mention date",
        },
        {
            "metric": "News mentions",
            "availability": "Unavailable",
            "operational_definition": "No news-specific count in the input parquet; not inferred",
            "temporal_attribution": "Not applicable",
        },
        {
            "metric": "Citations",
            "availability": "Available as current count",
            "operational_definition": "Dimensions times_cited",
            "temporal_attribution": "Publication year, not citation date",
        },
    ])


def main_indicator_specs(metric_colors):
    return OrderedDict({
        "patents": {
            "label": "Patents",
            "rate_label": "Patents",
            "inset_label": "Patents",
            "profile_label": "Patents",
            "matrix_label": "Patent",
            "flag_col": "has_patents",
            "annual_col": "annual_papers_linked_patents",
            "cumulative_col": "cumulative_papers_linked_patents",
            "share_col": "share_papers_linked_patents_pct",
            "color": metric_colors["patents"],
        },
        "clinical_trials": {
            "label": "Clinical trials",
            "rate_label": "Trials",
            "inset_label": "Clinical\ntrials",
            "profile_label": "Trials",
            "matrix_label": "Trial",
            "flag_col": "has_clinical_trials",
            "annual_col": "annual_papers_linked_clinical_trials",
            "cumulative_col": "cumulative_papers_linked_clinical_trials",
            "share_col": "share_papers_linked_clinical_trials_pct",
            "color": metric_colors["clinical_trials"],
        },
        "policy_documents": {
            "label": "Policy documents",
            "rate_label": "Policy",
            "inset_label": "Policy\ndocuments",
            "profile_label": "Policy",
            "matrix_label": "Policy",
            "flag_col": "has_policy_documents",
            "annual_col": "annual_papers_linked_policy_documents",
            "cumulative_col": "cumulative_papers_linked_policy_documents",
            "share_col": "share_papers_linked_policy_documents_pct",
            "color": metric_colors["policy_documents"],
        },
        "datasets": {
            "label": "Datasets",
            "rate_label": "Datasets",
            "inset_label": "Datasets",
            "profile_label": "Datasets",
            "matrix_label": "Data",
            "flag_col": "has_datasets",
            "annual_col": "annual_papers_linked_datasets",
            "cumulative_col": "cumulative_papers_linked_datasets",
            "share_col": "share_papers_linked_datasets_pct",
            "color": metric_colors["datasets"],
        },
        "cross_sector": {
            "label": "Cross-sector collaboration",
            "rate_label": "Cross-\nsector",
            "inset_label": "Cross-\nsector papers",
            "profile_label": "Cross-\nsector",
            "matrix_label": "Cross",
            "flag_col": "cross_sector_collaboration",
            "annual_col": "annual_cross_sector_papers",
            "cumulative_col": "cumulative_cross_sector_papers",
            "share_col": "share_cross_sector_collaboration_pct",
            "color": metric_colors["cross_sector"],
        },
        "altmetric": {
            "label": "Altmetric attention",
            "rate_label": "Altmetric",
            "inset_label": "Altmetric\nattention",
            "profile_label": "Altmetric",
            "matrix_label": "Alt.",
            "flag_col": "has_altmetric_attention",
            "annual_col": "annual_altmetric_papers",
            "cumulative_col": "cumulative_altmetric_papers",
            "share_col": "share_has_altmetric_attention_pct",
            "color": metric_colors["altmetric"],
        },
    })


def main_indicator_tables(complete, annual, indicators, latest):
    rate_table = pd.DataFrame([
        {
            "indicator": key,
            "metric": spec["label"],
            "publication_count": int(annual.loc[latest, spec["cumulative_col"]]),
            "publications_per_1000": (
                annual.loc[latest, spec["cumulative_col"]] / len(complete) * 1_000
            ),
            "color": spec["color"],
        }
        for key, spec in indicators.items()
    ])
    keys = list(indicators)
    flags = pd.DataFrame({
        key: complete[spec["flag_col"]].fillna(False).astype(bool)
        for key, spec in indicators.items()
    })
    conditional = pd.DataFrame(index=keys, columns=keys, dtype=float)
    for focal_key in keys:
        focal_mask = flags[focal_key]
        focal_count = int(focal_mask.sum())
        for other_key in keys:
            shared_count = int((focal_mask & flags[other_key]).sum())
            conditional.loc[focal_key, other_key] = (
                shared_count / focal_count * 100 if focal_count else np.nan
            )
    np.fill_diagonal(conditional.values, np.nan)

    cumulative_columns = [
        "cumulative_papers",
        *[spec["cumulative_col"] for spec in indicators.values()],
    ]
    if len(indicators) != 6:
        raise ValueError("Figure 1 requires the same six indicators in A, C and D")
    if not all(annual[column].is_monotonic_increasing for column in cumulative_columns):
        raise ValueError("A Figure 1 cumulative series is not monotonic")
    if not conditional.stack().between(0, 100).all():
        raise ValueError("Conditional-overlap percentages must lie between 0 and 100")

    label_map = {key: spec["label"] for key, spec in indicators.items()}
    overlap_export = conditional.rename(
        index=label_map,
        columns=label_map,
    ).reset_index(names="focal_indicator").round(1)
    return rate_table, conditional, overlap_export


def validate_growth_analysis(
    papers,
    complete,
    annual,
    endpoint_unique,
    indicators,
    indicator_rate_table,
    conditional_overlap,
    data_cutoff,
    last_complete_year,
    endpoint_specs=ENDPOINT_SPECS,
):
    """Check the invariants needed to interpret and reproduce the analysis.

    The returned table is suitable for export. Any failed check raises immediately,
    so figures cannot be produced from internally inconsistent intermediate data.
    """
    rows = []

    def add(scope, check, passed, observed, expected, interpretation):
        rows.append({
            "scope": scope,
            "check": check,
            "passed": bool(passed),
            "observed": observed,
            "expected": expected,
            "interpretation": interpretation,
        })

    duplicate_ids = int(papers["id"].duplicated().sum())
    missing_years = int(papers["year"].isna().sum())
    missing_dates = int(papers["publication_date"].isna().sum())
    comparable_dates = papers["publication_date"].notna() & papers["year"].notna()
    date_year_mismatches = int(
        papers.loc[comparable_dates, "publication_date"].dt.year.ne(
            papers.loc[comparable_dates, "year"]
        ).sum()
    )
    add(
        "source",
        "publication_ids_are_unique",
        duplicate_ids == 0,
        duplicate_ids,
        0,
        "Duplicate publication IDs would double-count papers.",
    )
    add(
        "source",
        "publication_years_are_complete",
        missing_years == 0,
        missing_years,
        0,
        "Every source record requires a publication year.",
    )
    add(
        "source",
        "publication_dates_are_complete",
        missing_dates == 0,
        missing_dates,
        0,
        "Publication dates are required for milestone dates.",
    )
    add(
        "source",
        "publication_date_years_agree",
        date_year_mismatches == 0,
        date_year_mismatches,
        0,
        "The date and year fields must identify the same publication cohort.",
    )

    first_year = int(complete["year"].min())
    complete_future = int(complete["future_dated"].sum())
    complete_after_window = int(complete["year"].gt(last_complete_year).sum())
    add(
        "corpus",
        "complete_year_corpus_ends_as_specified",
        complete_after_window == 0 and int(complete["year"].max()) == last_complete_year,
        f"{first_year}-{int(complete['year'].max())}",
        f"{first_year}-{last_complete_year}",
        "Primary estimates must exclude the provisional publication year.",
    )
    add(
        "corpus",
        "complete_year_corpus_has_no_future_dates",
        complete_future == 0,
        complete_future,
        0,
        f"No analytical record may fall after {data_cutoff.date()}.",
    )
    add(
        "corpus",
        "complete_year_publication_ids_are_unique",
        not complete["id"].duplicated().any(),
        int(complete["id"].duplicated().sum()),
        0,
        "Each publication contributes once to the complete-year corpus.",
    )

    expected_years = pd.Index(
        range(first_year, last_complete_year + 1),
        name=annual.index.name,
    )
    annual_total = int(annual["annual_papers"].sum())
    cumulative_total = int(annual.loc[last_complete_year, "cumulative_papers"])
    add(
        "annual",
        "annual_index_is_contiguous",
        annual.index.equals(expected_years),
        len(annual.index),
        len(expected_years),
        "Zero-count years must remain explicit in annual series.",
    )
    add(
        "annual",
        "annual_counts_sum_to_corpus",
        annual_total == len(complete),
        annual_total,
        len(complete),
        "Annual publication counts must partition the analytical corpus.",
    )
    add(
        "annual",
        "cumulative_total_matches_corpus",
        cumulative_total == len(complete),
        cumulative_total,
        len(complete),
        "The final cumulative count must equal the analytical corpus size.",
    )
    add(
        "annual",
        "cumulative_publications_are_monotonic",
        annual["cumulative_papers"].is_monotonic_increasing,
        bool(annual["cumulative_papers"].is_monotonic_increasing),
        True,
        "A cumulative publication series cannot decrease.",
    )

    rate_lookup = indicator_rate_table.set_index("indicator")
    for key, spec in indicators.items():
        flag_count = int(complete[spec["flag_col"]].fillna(False).sum())
        cumulative_count = int(annual.loc[last_complete_year, spec["cumulative_col"]])
        table_count = int(rate_lookup.loc[key, "publication_count"])
        observed_rate = float(rate_lookup.loc[key, "publications_per_1000"])
        expected_rate = flag_count / len(complete) * 1_000
        add(
            "indicator",
            f"{key}_annual_total_matches_flags",
            cumulative_count == flag_count,
            cumulative_count,
            flag_count,
            "Annual aggregation must reproduce publication-level indicator flags.",
        )
        add(
            "indicator",
            f"{key}_rate_matches_count",
            table_count == flag_count and np.isclose(observed_rate, expected_rate),
            f"{table_count}; {observed_rate:.6f}",
            f"{flag_count}; {expected_rate:.6f}",
            "Figure 1 rates must use the complete-year corpus denominator.",
        )

    off_diagonal = conditional_overlap.stack()
    add(
        "overlap",
        "conditional_overlap_is_bounded",
        off_diagonal.between(0, 100).all(),
        f"{off_diagonal.min():.3f}-{off_diagonal.max():.3f}",
        "0-100",
        "Conditional co-occurrence values are percentages.",
    )
    add(
        "overlap",
        "conditional_overlap_diagonal_is_omitted",
        pd.isna(np.diag(conditional_overlap)).all(),
        int(pd.isna(np.diag(conditional_overlap)).sum()),
        len(conditional_overlap),
        "Self-comparisons are omitted from Figure 1D.",
    )

    for key in endpoint_specs:
        unique = endpoint_unique[key]
        duplicate_endpoint_ids = int(unique["endpoint_id"].duplicated().sum())
        expected_total = int(unique["first_link_year"].le(last_complete_year).sum())
        annual_endpoint_total = int(
            annual.loc[last_complete_year, f"cumulative_unique_{key}"]
        )
        add(
            "endpoint",
            f"{key}_ids_are_unique",
            duplicate_endpoint_ids == 0,
            duplicate_endpoint_ids,
            0,
            "Linked records must be deduplicated by endpoint ID.",
        )
        add(
            "endpoint",
            f"{key}_first_link_total_matches_annual",
            annual_endpoint_total == expected_total,
            annual_endpoint_total,
            expected_total,
            "Endpoint first-link cohorts must reproduce the annual endpoint total.",
        )

    checks = pd.DataFrame(rows)
    failures = checks.loc[~checks["passed"], "check"].tolist()
    if failures:
        raise AssertionError(
            "Growth analysis validation failed: " + ", ".join(failures)
        )
    return checks


def growth_phase_windows(first_year, last_complete_year):
    candidates = [
        (max(first_year, 2013), min(2017, last_complete_year)),
        (max(first_year, 2018), min(2021, last_complete_year)),
        (max(first_year, 2022), last_complete_year),
    ]
    return [
        (f"{start_year}-{end_year}", start_year, end_year)
        for start_year, end_year in candidates
        if start_year <= end_year
    ]


def growth_period_summary(annual, complete, phase_windows):
    rows = []
    for period, start_year, end_year in phase_windows:
        first_output = float(annual.loc[start_year, "annual_papers"])
        last_output = float(annual.loc[end_year, "annual_papers"])
        elapsed_years = end_year - start_year
        cagr_pct = (
            ((last_output / first_output) ** (1 / elapsed_years) - 1) * 100
            if elapsed_years and first_output > 0
            else np.nan
        )
        phase_papers = int(annual.loc[start_year:end_year, "annual_papers"].sum())
        rows.append({
            "period": period,
            "papers_published": phase_papers,
            "share_of_complete_year_corpus_pct": phase_papers / len(complete) * 100,
            "annual_output_first_year": int(first_output),
            "annual_output_last_year": int(last_output),
            "annual_output_cagr_pct": cagr_pct,
            "median_yoy_growth_pct": annual.loc[
                start_year + 1:end_year, "yoy_publication_growth_pct"
            ].median(),
        })
    return pd.DataFrame(rows)


def indicator_accumulation_milestones(annual, indicators, last_complete_year):
    rows = []
    for spec in indicators.values():
        cumulative = annual[spec["cumulative_col"]]
        observed_total = int(cumulative.loc[last_complete_year])
        row = {
            "indicator": spec["label"],
            "observed_cumulative_papers": observed_total,
        }
        milestone_years = []
        for fraction in (0.25, 0.50, 0.75):
            reached = cumulative.loc[cumulative.ge(observed_total * fraction)]
            year = int(reached.index[0]) if observed_total and not reached.empty else pd.NA
            row[f"year_at_{int(fraction * 100)}pct"] = year
            milestone_years.append(year)
        row["years_from_25pct_to_75pct"] = (
            milestone_years[2] - milestone_years[0]
            if all(pd.notna(year) for year in milestone_years)
            else pd.NA
        )
        rows.append(row)
    return pd.DataFrame(rows)


def period_composition_summary(complete, phase_windows):
    rows = []
    for period, start_year, end_year in phase_windows:
        cohort = complete.loc[complete["year"].between(start_year, end_year)]
        rows.append({
            "period": period,
            "papers": len(cohort),
            "median_authors": cohort["authors_count"].median(),
            "authors_q25": cohort["authors_count"].quantile(0.25),
            "authors_q75": cohort["authors_count"].quantile(0.75),
            "cross_sector_pct": cohort["cross_sector_collaboration"].mean() * 100,
            "international_pct": cohort["international_collaboration"].mean() * 100,
            "preprint_pct": cohort["is_preprint"].mean() * 100,
        })
    return pd.DataFrame(rows)


def cross_sector_composition_summary(
    complete,
    non_academic_sectors=NON_ACADEMIC_SECTORS,
):
    labels = OrderedDict((sector.lower(), sector) for sector in non_academic_sectors)
    cross_sector_mask = complete["cross_sector_collaboration"]
    cross_sector_total = int(cross_sector_mask.sum())
    cross_sector_years = complete.loc[cross_sector_mask, "year"]
    rows = [{
        "non_academic_sector": "Any listed sector",
        "papers": cross_sector_total,
        "share_of_all_papers_pct": cross_sector_total / len(complete) * 100,
        "share_of_cross_sector_papers_pct": 100.0,
        "earliest_publication_year": int(cross_sector_years.min()),
        "latest_publication_year": int(cross_sector_years.max()),
    }]
    for key, label in labels.items():
        mask = complete[f"cross_sector_{key}"].fillna(False).astype(bool)
        count = int(mask.sum())
        years = complete.loc[mask, "year"]
        rows.append({
            "non_academic_sector": label,
            "papers": count,
            "share_of_all_papers_pct": count / len(complete) * 100,
            "share_of_cross_sector_papers_pct": (
                count / cross_sector_total * 100 if cross_sector_total else np.nan
            ),
            "earliest_publication_year": int(years.min()) if count else pd.NA,
            "latest_publication_year": int(years.max()) if count else pd.NA,
        })
    table = pd.DataFrame(rows)
    flag_columns = [f"cross_sector_{key}" for key in labels]
    sector_counts = complete[flag_columns].fillna(False).astype(bool).sum(axis=1)
    multiple_sector_count = int(sector_counts.ge(2).sum())
    return table, multiple_sector_count


def supplementary_indicator_specs(metric_colors):
    columns = OrderedDict({
        "Altmetric attention": "has_altmetric_attention",
        "Cross-sector": "cross_sector_collaboration",
        "International": "international_collaboration",
        "Patent": "has_patents",
        "Clinical trial": "has_clinical_trials",
        "Policy document": "has_policy_documents",
        "Dataset": "has_datasets",
    })
    colors = {
        "Altmetric attention": metric_colors["altmetric"],
        "Cross-sector": metric_colors["cross_sector"],
        "International": metric_colors["international"],
        "Patent": metric_colors["patents"],
        "Clinical trial": metric_colors["clinical_trials"],
        "Policy document": metric_colors["policy_documents"],
        "Dataset": metric_colors["datasets"],
    }
    return columns, colors


def indicator_overlap_tables(complete, indicator_columns):
    indicator_frame = complete[list(indicator_columns.values())].astype(bool)
    indicator_frame.columns = list(indicator_columns.keys())
    labels = indicator_frame.columns.tolist()
    jaccard = pd.DataFrame(index=labels, columns=labels, dtype=float)
    pair_counts = pd.DataFrame(index=labels, columns=labels, dtype=int)
    for left in labels:
        for right in labels:
            intersection = (indicator_frame[left] & indicator_frame[right]).sum()
            union = (indicator_frame[left] | indicator_frame[right]).sum()
            pair_counts.loc[left, right] = int(intersection)
            jaccard.loc[left, right] = intersection / union if union else np.nan
    prevalence = indicator_frame.mean().mul(100).sort_values()
    return jaccard, pair_counts, prevalence


@dataclass
class GrowthPlotter:
    """Small plotting primitives shared by all figures in the growth notebook."""

    style: Mapping
    years: pd.Index
    first_year: int
    last_complete_year: int
    baseline: int
    grid_kws: Mapping

    def panel_label(self, ax, label, x=-0.12, y=1.08, in_layout=True):
        return shared_panel_label(
            ax,
            label,
            self.style,
            x=x,
            y=y,
            va="top",
            in_layout=in_layout,
        )

    def format_year_axis(self, ax):
        ticks = [
            year
            for year in [2013, 2016, 2019, 2022, 2025]
            if year in self.years
        ]
        ax.set_xlim(self.first_year, self.last_complete_year + 0.35)
        ax.set_xticks(ticks)
        ax.tick_params(axis="x", rotation=0)
        ax.grid(axis="both", **self.grid_kws)
        ax.set_axisbelow(True)
        ax.set_xlabel("Publication year")

    @staticmethod
    def compact_count(value):
        return shared_compact_count(value)

    def inset_sparkline(self, ax, series, title, color):
        x = series.index.to_numpy(dtype=float)
        y = series.to_numpy(dtype=float)
        ax.set_facecolor("#FCFCFB")
        ax.plot(x, y, color=color, linewidth=1.8, zorder=2)
        ax.fill_between(x, y, color=color, alpha=0.11, zorder=1)
        ax.scatter(
            x[-1],
            y[-1],
            color=color,
            s=shared_marker_area(self.style, scale=0.38),
            zorder=3,
        )
        ax.set_xlim(self.first_year, self.last_complete_year + 0.35)
        ax.set_ylim(0, max(y[-1] * 1.28, 1))
        ax.set_xticks([self.first_year, self.baseline, self.last_complete_year])
        ax.set_yticks([0, y[-1] / 2, y[-1]])
        ax.tick_params(
            axis="x",
            which="major",
            length=4.5,
            width=1.25,
            color="#000000",
            bottom=True,
            top=False,
            right=False,
            labelbottom=False,
            labelleft=False,
        )
        ax.tick_params(
            axis="y",
            which="major",
            direction="out",
            length=3.2,
            width=0.9,
            color="#000000",
            left=True,
            right=False,
            labelleft=False,
        )
        ax.grid(False, which="both", axis="both")
        for spine_name, spine in ax.spines.items():
            spine.set_visible(spine_name in {"left", "bottom"})
            spine.set_color("#000000")
            spine.set_linewidth(max(self.style.get("axes_linewidth", 1.0), 1.3))
        ax.set_ylabel(
            title,
            rotation=0,
            ha="left",
            va="top",
            fontsize=self.style["annot_fs"] - 0.3,
            fontweight="semibold",
            multialignment="left",
            linespacing=0.95,
            color="#222222",
        )
        ax.yaxis.set_label_coords(0.055, 0.91)
        ax.text(
            0.945,
            0.91,
            self.compact_count(y[-1]),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=self.style["annot_fs"],
            fontweight="bold",
            color=color,
        )
        ax.text(
            0.055,
            0.08,
            str(self.first_year),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=self.style["annot_fs"] - 2,
            color="#6B7280",
        )
        ax.text(
            0.945,
            0.08,
            str(self.last_complete_year),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=self.style["annot_fs"] - 2,
            color="#6B7280",
        )


def plot_growth_and_reach(
    annual,
    indicators,
    indicator_rate_table,
    conditional_overlap,
    style,
    metric_colors,
    plotter,
    baseline,
    latest,
    cagr,
):
    """Build the four-panel main growth-and-reach figure."""
    years = annual.index
    indicator_keys = list(indicators)
    fig = plt.figure(figsize=style["figsize_main"], layout="constrained")
    grid = fig.add_gridspec(
        3,
        5,
        width_ratios=[1.0, 1.0, 1.0, 0.03, 1.90],
        height_ratios=[0.75, 1.05, 1.40],
        wspace=0.03,
        hspace=0.06,
    )
    ax_main = fig.add_subplot(grid[:, :3])
    ax_annual = fig.add_subplot(grid[0, 4])
    ax_rates = fig.add_subplot(grid[1, 4])
    ax_overlap = fig.add_subplot(grid[2, 4])

    main_x = annual.index.to_numpy(dtype=float)
    main_y = annual["cumulative_papers"].to_numpy(dtype=float)
    ax_main.plot(
        main_x,
        main_y,
        color=metric_colors["publications"],
        linewidth=3.4,
        zorder=3,
    )
    ax_main.fill_between(
        main_x,
        main_y,
        color=metric_colors["publications"],
        alpha=0.12,
        zorder=1,
    )
    ax_main.scatter(
        main_x[-1],
        main_y[-1],
        color=metric_colors["publications"],
        edgecolor="white",
        linewidth=1.0,
        s=shared_marker_area(style, scale=0.90),
        zorder=4,
    )
    ax_main.set_ylabel("Cumulative Publications")
    ax_main.yaxis.set_major_formatter(mticker.EngFormatter(sep=""))
    ax_main.set_ylim(0, main_y[-1] * 1.08)
    plotter.format_year_axis(ax_main)
    plotter.panel_label(ax_main, "A", x=-0.055, y=1.075)
    ax_main.annotate(
        f"{main_y[-1]:,.0f} papers",
        xy=(main_x[-1], main_y[-1]),
        xytext=(-12, 11),
        textcoords="offset points",
        ha="right",
        va="bottom",
        color=metric_colors["publications"],
        fontsize=style["label_fs"] + 1,
        fontweight="bold",
    )
    ax_main.add_patch(
        Rectangle(
            (0.0, 0.45),
            0.715,
            0.50,
            transform=ax_main.transAxes,
            facecolor="white",
            edgecolor="none",
            zorder=4.5,
        )
    )
    ax_main.spines["left"].set_zorder(5)
    for tick in ax_main.yaxis.get_major_ticks():
        tick.tick1line.set_zorder(5)
    inset_specs = [
        (spec["inset_label"], spec["cumulative_col"], spec["color"])
        for spec in indicators.values()
    ]
    inset_positions = [
        (0.04, 0.73, 0.205, 0.20),
        (0.265, 0.73, 0.205, 0.20),
        (0.49, 0.73, 0.205, 0.20),
        (0.04, 0.48, 0.205, 0.20),
        (0.265, 0.48, 0.205, 0.20),
        (0.49, 0.48, 0.205, 0.20),
    ]
    for (title, column, color), bounds in zip(inset_specs, inset_positions):
        plotter.inset_sparkline(
            ax_main.inset_axes(bounds),
            annual[column],
            title,
            color,
        )

    bar_colors = np.full(len(years), style["c_primary_light"], dtype=object)
    bar_colors[years.get_loc(baseline)] = style["c_gold"]
    bar_colors[years.get_loc(latest)] = metric_colors["publications"]
    ax_annual.bar(
        years,
        annual["annual_papers"],
        width=0.78,
        color=bar_colors,
        edgecolor="#000000",
        linewidth=0.8,
    )
    ax_annual.yaxis.set_major_formatter(mticker.EngFormatter(sep=""))
    ax_annual.set_ylim(0, annual["annual_papers"].max() * 1.20)
    plotter.format_year_axis(ax_annual)
    ax_annual.set_xticks([2013, 2018, 2021, 2025])
    plotter.panel_label(ax_annual, "B", y=1.13, in_layout=False)
    ax_annual.text(
        0.04,
        0.91,
        f"{annual.loc[latest, 'annual_papers'] / annual.loc[baseline, 'annual_papers']:.1f}"
        f"\N{MULTIPLICATION SIGN} the {baseline} output\n"
        f"{cagr:.1%} compound annual growth",
        transform=ax_annual.transAxes,
        ha="left",
        va="top",
        fontsize=style["annot_fs"],
        color="#374151",
        linespacing=1.35,
    )
    ax_annual.annotate(
        f"{int(annual.loc[latest, 'annual_papers']):,}",
        xy=(latest, annual.loc[latest, "annual_papers"]),
        xytext=(-2, 6),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=style["annot_fs"],
        fontweight="bold",
        color=metric_colors["publications"],
    )

    rate_y = np.arange(len(indicator_rate_table))[::-1]
    for y_position, (_, row) in zip(rate_y, indicator_rate_table.iterrows()):
        value = row["publications_per_1000"]
        color = row["color"]
        ax_rates.hlines(
            y_position,
            5,
            value,
            color=color,
            linewidth=2.0,
            alpha=0.65,
        )
        ax_rates.scatter(
            value,
            y_position,
            s=shared_marker_area(style),
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        ax_rates.text(
            value * 1.12,
            y_position,
            f"{value:.1f}",
            ha="left",
            va="center",
            fontsize=style["annot_fs"],
            fontweight="bold",
            color=color,
        )
    ax_rates.set_xscale("log")
    ax_rates.set_xlim(5, 1_000)
    ax_rates.set_ylim(-0.45, rate_y.max() + 0.45)
    ax_rates.set_xticks([10, 100, 1_000])
    ax_rates.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax_rates.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax_rates.set_yticks(
        rate_y,
        [spec["rate_label"] for spec in indicators.values()],
    )
    for tick_label in ax_rates.get_yticklabels():
        tick_label.set_linespacing(0.74)
        tick_label.set_multialignment("right")
        tick_label.set_verticalalignment("center")
    ax_rates.set_xlabel("Rate per 1,000 papers (log scale)")
    ax_rates.grid(axis="both", **plotter.grid_kws)
    ax_rates.set_axisbelow(True)
    plotter.panel_label(ax_rates, "C", y=1.13, in_layout=False)

    matrix_labels = [spec["matrix_label"] for spec in indicators.values()]
    matrix_row_labels = [spec["profile_label"] for spec in indicators.values()]
    matrix_colors = [spec["color"] for spec in indicators.values()]
    matrix_size = len(indicator_keys)
    for row_index, (row_key, row_color) in enumerate(
        zip(indicator_keys, matrix_colors)
    ):
        base_rgb = np.asarray(to_rgb(row_color))
        for column_index, column_key in enumerate(indicator_keys):
            is_diagonal = row_key == column_key
            value = conditional_overlap.loc[row_key, column_key]
            if is_diagonal:
                face_color = "#F2F2F2"
                annotation = "\N{EM DASH}"
                text_color = "#777777"
            else:
                alpha = 0.12 + 0.72 * value / 100
                face_rgb = (1 - alpha) * np.ones(3) + alpha * base_rgb
                face_color = tuple(face_rgb)
                annotation = f"{value:.0f}%"
                luminance = np.dot(face_rgb, [0.2126, 0.7152, 0.0722])
                text_color = "white" if luminance < 0.52 else "#222222"
            ax_overlap.add_patch(Rectangle(
                (column_index, row_index),
                1,
                1,
                facecolor=face_color,
                edgecolor="white",
                linewidth=1.2,
            ))
            ax_overlap.text(
                column_index + 0.5,
                row_index + 0.5,
                annotation,
                ha="center",
                va="center",
                fontsize=style["annot_fs"] - 1,
                fontweight="bold",
                color=text_color,
            )
    ax_overlap.set_xlim(0, matrix_size)
    ax_overlap.set_ylim(matrix_size, 0)
    ax_overlap.set_xticks(np.arange(matrix_size) + 0.5, matrix_labels)
    ax_overlap.set_yticks(np.arange(matrix_size) + 0.5, matrix_row_labels)
    for tick_label in ax_overlap.get_yticklabels():
        tick_label.set_linespacing(0.82)
        tick_label.set_multialignment("right")
        tick_label.set_verticalalignment("center")
    ax_overlap.tick_params(axis="x", pad=7)
    ax_overlap.set_xlabel("Co-occurring indicator")
    for spine in ax_overlap.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)
    plotter.panel_label(ax_overlap, "D", y=1.08, in_layout=False)
    return fig


def plot_annual_growth_supplement(
    annual,
    sector_by_year,
    style,
    metric_colors,
    plotter,
    baseline,
    last_complete_year,
):
    """Build the six-panel annual growth, collaboration, and reach supplement."""
    years = annual.index
    bar_colors = np.full(len(years), style["c_primary_light"], dtype=object)
    bar_colors[years.get_loc(baseline)] = style["c_gold"]
    bar_colors[years.get_loc(last_complete_year)] = metric_colors["publications"]

    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5), layout="constrained")
    axes = axes.ravel()
    axes[0].bar(
        years,
        annual["annual_papers"],
        color=bar_colors,
        width=0.78,
        edgecolor="#000000",
        linewidth=0.8,
    )
    axes[0].set_ylabel("Annual publications")
    axes[0].yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))

    growth = annual["yoy_publication_growth_pct"]
    recent_growth = growth.loc[baseline:last_complete_year]
    growth_colors = np.full(
        len(recent_growth), style["c_primary_light"], dtype=object
    )
    growth_colors[recent_growth.index.get_loc(baseline)] = style["c_gold"]
    growth_colors[recent_growth.index.get_loc(last_complete_year)] = metric_colors[
        "publications"
    ]
    growth_colors[recent_growth.lt(0).to_numpy()] = style["c_accent"]
    axes[2].bar(
        recent_growth.index,
        recent_growth,
        color=growth_colors,
        width=0.78,
        edgecolor="#000000",
        linewidth=0.8,
    )
    axes[2].axhline(0, color="#555555", linewidth=0.8)
    axes[2].set_ylabel("Year-on-year growth (%)")
    axes[2].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))

    axes[1].plot(
        years,
        annual["median_authors"],
        color=metric_colors["publications"],
        linewidth=2.2,
    )
    axes[1].fill_between(
        years.to_numpy(dtype=float),
        annual["authors_q25"].to_numpy(dtype=float),
        annual["authors_q75"].to_numpy(dtype=float),
        facecolor=to_rgba(metric_colors["publications"], 0.15),
        edgecolor="k",
        linewidth=0.9,
        linestyle="--",
        label="Interquartile\nrange",
    )
    axes[1].set_ylabel("Authors per paper")
    axes[1].legend(frameon=True, edgecolor="k", loc="upper right")

    axes[3].plot(
        years,
        annual["share_cross_sector_collaboration_pct"],
        color=metric_colors["cross_sector"],
        linewidth=2.2,
        label="Cross-sector",
    )
    axes[3].plot(
        years,
        annual["share_international_collaboration_pct"],
        color=metric_colors["international"],
        linewidth=2.2,
        label="International",
    )
    axes[3].plot(
        years,
        sector_by_year["share_education_company_pct"],
        color=style["c_gold"],
        linewidth=1.8,
        label="Education + company",
    )
    axes[3].set_ylabel("Share of annual publications (%)")
    axes[3].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    axes[3].legend(frameon=True, edgecolor="k", loc="best")

    axes[4].plot(
        years,
        annual["share_has_altmetric_attention_pct"],
        color=metric_colors["altmetric"],
        linewidth=2.2,
        label="Altmetric attention",
    )
    axes[4].plot(
        years,
        annual["share_is_open_access_pct"],
        color=metric_colors["open_access"],
        linewidth=2.2,
        label="Open access",
    )
    axes[4].plot(
        years,
        annual["share_is_preprint_pct"],
        color=metric_colors["preprints"],
        linewidth=1.8,
        label="Preprints",
    )
    axes[4].set_ylabel("Share of annual publications (%)")
    axes[4].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    axes[4].legend(frameon=True, edgecolor="k", loc="best")

    reach_lines = [
        ("grants", "Grants", metric_colors["grants"]),
        ("datasets", "Datasets", metric_colors["datasets"]),
        ("patents", "Patents", metric_colors["patents"]),
        ("policy_documents", "Policy", metric_colors["policy_documents"]),
        ("clinical_trials", "Trials", metric_colors["clinical_trials"]),
    ]
    for key, label, color in reach_lines:
        reach_share = annual[f"share_papers_linked_{key}_pct"].replace(0, np.nan)
        axes[5].plot(
            years,
            reach_share,
            linewidth=1.9,
            color=color,
            label=label,
        )
    axes[5].set_yscale("log")
    axes[5].set_ylim(0.05, 100)
    axes[5].set_ylabel("Publications linked to records (%, log scale)")
    axes[5].yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda value, _: f"{value:g}%")
    )
    axes[5].legend(frameon=True, edgecolor="k", ncol=2, loc="best")

    for label, ax in zip("ABCDEF", axes):
        plotter.panel_label(ax, label, x=-0.10, y=1.16)
        plotter.format_year_axis(ax)
    axes[2].set_xlim(baseline - 0.5, last_complete_year + 0.5)
    axes[2].set_xticks([2018, 2020, 2022, 2024])
    return fig


def plot_indicator_overlap_supplement(
    jaccard,
    prevalence,
    indicator_colors,
    style,
    metric_colors,
    plotter,
):
    """Build the Jaccard heatmap and indicator-prevalence supplement."""
    fig, (ax_heat, ax_bar) = plt.subplots(
        1,
        2,
        figsize=(14, 5.8),
        layout="constrained",
        gridspec_kw={"width_ratios": [1.35, 0.9]},
    )
    triangle_mask = np.triu(np.ones(jaccard.shape, dtype=bool))
    off_diagonal = jaccard.mask(np.eye(len(jaccard), dtype=bool)).stack()
    color_max = max(0.05, math.ceil(float(off_diagonal.max()) * 20) / 20)
    sns.heatmap(
        jaccard,
        mask=triangle_mask,
        annot=True,
        fmt=".2f",
        cmap=sequential_colormap(metric_colors["publications"]),
        vmin=0,
        vmax=color_max,
        square=True,
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": "Jaccard similarity"},
        ax=ax_heat,
    )
    heatmap_cbar = ax_heat.collections[0].colorbar
    heatmap_cbar.outline.set_edgecolor("k")
    heatmap_cbar.outline.set_linewidth(1.0)
    ax_heat.set_xlabel("")
    ax_heat.set_ylabel("")
    plt.setp(
        ax_heat.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    plotter.panel_label(ax_heat, "A")

    ax_bar.barh(
        prevalence.index,
        prevalence.values,
        color=[indicator_colors[label] for label in prevalence.index],
        edgecolor="#000000",
        linewidth=0.8,
    )
    for position, value in enumerate(prevalence.values):
        ax_bar.text(
            value + 0.8,
            position,
            f"{value:.1f}%",
            va="center",
            fontsize=style["annot_fs"],
        )
    ax_bar.set_xlim(0, max(prevalence.max() * 1.18, 10))
    ax_bar.set_xlabel("Share of publications (%)")
    ax_bar.set_ylabel("Reach indicator")
    ax_bar.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    ax_bar.grid(axis="both", **plotter.grid_kws)
    ax_bar.set_axisbelow(True)
    plotter.panel_label(ax_bar, "B")
    return fig
