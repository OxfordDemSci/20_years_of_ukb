"""Growth rankings from existing category aggregates, without fitting topic models."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data_analysis_02_content_panels as panels
from .shared_analysis_window import ANALYSIS_START_YEAR, ANALYSIS_END_YEAR

VOCABULARIES = {"for": ("FoR Level 4", "for_l4"),
                "rcdc": ("RCDC", "rcdc"), "topics": ("BERTopic", "topics")}


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer.")
    return int(value)


def category_growth_table(data, baseline_year=2018, end_year=2025, min_baseline_papers=10):
    """All categories, with whole-paper growth and the figures' composition shares.

    Missing classification years are unknown; a zero for an individual category
    in a classified year is an observed zero. Neither receives a fabricated CAGR.
    """
    start = _integer(baseline_year, "baseline_year")
    end = _integer(end_year, "end_year")
    minimum = _integer(min_baseline_papers, "min_baseline_papers")
    if not ANALYSIS_START_YEAR <= start < end <= ANALYSIS_END_YEAR:
        raise ValueError("Growth years must be ordered within the 2013–2025 analysis window.")
    if minimum < 1:
        raise ValueError("min_baseline_papers must be positive.")
    rows = []
    for vocabulary in VOCABULARIES:
        block = data[vocabulary]
        if not block["available"]:
            continue
        counts = block.get("paper_counts")
        if counts is None:
            if block["weight"] != "n_papers":
                raise ValueError("RCDC growth requires distinct paper_counts, not fractional weights.")
            counts = block["weights"]
        shares = block["share"]
        labels = (panels._topic_legend_labels(counts.columns) if vocabulary == "topics"
                  else {label: label for label in counts.columns})
        present = block["weights"].sum(axis=1).gt(0)
        known = {year: bool(present.get(year, False)) for year in (start, end - 1, end)}
        for category in counts.columns:
            def count(year):
                return float(counts.loc[year, category]) if known[year] else np.nan

            first, previous, last = count(start), count(end - 1), count(end)
            first_share = float(shares.loc[start, category]) if known[start] else np.nan
            last_share = float(shares.loc[end, category]) if known[end] else np.nan
            fold = last / first if first > 0 and np.isfinite(last) else np.nan
            if not (known[start] and known[end]):
                note = "Missing classification year"
            elif first == 0:
                note = "Absent in baseline year"
            elif first < minimum:
                note = f"Fewer than {minimum} baseline papers"
            elif last <= first:
                note = "No positive paper-count growth"
            else:
                note = "Eligible"
            rows.append({
                "vocabulary": vocabulary, "category": category,
                "display_label": str(labels[category]).replace("\n", " "),
                "baseline_year": start, "end_year": end, "previous_year": end - 1,
                "min_baseline_papers": minimum,
                "n_baseline": first, "n_previous": previous, "n_end": last,
                "paper_increase": last - first,
                "baseline_composition_pct": first_share, "end_composition_pct": last_share,
                "share_change_pp": last_share - first_share,
                "fold_change": fold, "growth_pct": 100 * (fold - 1),
                "cagr_pct": 100 * (fold ** (1 / (end - start)) - 1),
                "recent_growth_pct": 100 * (last / previous - 1)
                    if previous > 0 and np.isfinite(last) else np.nan,
                "eligible_cagr": note == "Eligible", "eligibility_note": note,
                "composition_counting": block["weight"],
                "baseline_composition_denominator": float(block["weights"].loc[start].sum())
                    if known[start] else np.nan,
                "end_composition_denominator": float(block["weights"].loc[end].sum())
                    if known[end] else np.nan,
            })
    return pd.DataFrame(rows)


def rank_growth_tables(all_rows, top_n=20):
    """Separate relative-count growth from increasing composition share."""
    top_n = _integer(top_n, "top_n")
    if top_n < 1:
        raise ValueError("top_n must be positive.")
    if all_rows.empty:
        return {name: all_rows.assign(rank=pd.Series(dtype=int)) for name in ("fastest", "share_gains")}
    definitions = {
        "fastest": (all_rows.eligible_cagr, "cagr_pct"),
        "share_gains": (all_rows.share_change_pp.gt(0) &
                        all_rows.n_end.ge(all_rows.min_baseline_papers), "share_change_pp"),
    }
    tables = {}
    for name, (eligible, metric) in definitions.items():
        table = all_rows.loc[eligible].sort_values(
            ["vocabulary", metric, "n_end", "category"], ascending=[True, False, False, True])
        table = table.groupby("vocabulary", sort=False).head(top_n).copy()
        table.insert(0, "rank", table.groupby("vocabulary", sort=False).cumcount() + 1)
        tables[name] = table.reset_index(drop=True)
    return tables


def growth_display_table(rows, baseline_year=2018, end_year=2025):
    """Compact manuscript view; complete precision and recent counts stay in CSV."""
    columns = {
        "display_label": "Category", "rank": "Rank",
        "n_baseline": f"Papers {baseline_year}", "n_end": f"Papers {end_year}",
        "cagr_pct": "Annualised growth (%)",
        "recent_growth_pct": f"Growth {end_year - 1}–{end_year} (%)",
        "baseline_composition_pct": f"Share {baseline_year} (%)",
        "end_composition_pct": f"Share {end_year} (%)", "share_change_pp": "Change (pp)",
    }
    view = rows.loc[:, list(columns)].rename(columns=columns).set_index("Category")
    for column in view:
        if column == "Rank" or column.startswith("Papers"):
            view[column] = view[column].astype("Int64")
        else:
            view[column] = view[column].round(2 if "Share" in column or "pp" in column else 1)
    return view


def growth_notes(baseline_year, end_year, min_baseline_papers):
    return [
        f"Publications are restricted to 2013–2025. Relative growth compares {baseline_year} with {end_year}; "
        f"CAGR = 100 × [(papers in {end_year} / papers in {baseline_year})^(1/{end_year-baseline_year}) − 1]. "
        f"The recent-growth column compares {end_year-1} with {end_year}.",
        f"Fastest-growth rankings require at least {min_baseline_papers} distinct papers in the baseline year "
        "and positive count growth. Ties are ordered by end-year paper count, then category name. "
        "Zero baselines have no defined percentage growth or CAGR. Small-baseline categories remain in "
        "the complete CSV and the separate ranking by percentage-point gain in composition share. "
        f"The share-gain ranking requires positive gain and at least {min_baseline_papers} end-year papers, "
        "with no baseline-count minimum.",
        "Counts are distinct publications within each category; a paper can contribute to several FoRs or RCDC tags. "
        "Shares match the content figures: FoR shares use all paper–field assignments; RCDC shares divide each "
        "classified paper equally across its tags; topics use non-outlier assigned papers. "
        "Missing classifications are excluded from these share denominators. Share change is in percentage points (pp).",
        "These are descriptive endpoint comparisons. Growth in paper counts and increased relative representation "
        "are different measures. The early 2013–2015 cohorts are too small for a stable default growth baseline; "
        "The default 2018 baseline matches the main publication-growth comparison. "
        "These tables supersede the archived naive FoR growth ranking, which allowed one-paper baselines.",
    ]


def export_growth_tables(data, registry, *, baseline_year=2018, end_year=2025,
                         min_baseline_papers=10, top_n=20):
    """CSV, editable Word and Excel exports, using already-computed category data."""
    all_rows = category_growth_table(data, baseline_year, end_year, min_baseline_papers)
    ranked = rank_growth_tables(all_rows, top_n=top_n)
    tables = {"content_category_growth_all.csv": all_rows,
              "content_fastest_growing_categories.csv": ranked["fastest"],
              "content_largest_share_gains.csv": ranked["share_gains"]}
    views, workbook = {}, {"All categories": all_rows}
    notes = growth_notes(baseline_year, end_year, min_baseline_papers)
    for vocabulary, (label, slug) in VOCABULARIES.items():
        rows = ranked["fastest"]
        if not data[vocabulary]["available"]:
            continue
        rows = rows.loc[rows.vocabulary.eq(vocabulary)]
        view = growth_display_table(rows, baseline_year, end_year)
        views[vocabulary] = view
        stem = f"content_fastest_growing_{slug}"
        tables[f"{stem}.csv"] = rows
        workbook[f"{label} growth"] = rows
        workbook[f"{label} share gains"] = ranked["share_gains"].loc[
            ranked["share_gains"].vocabulary.eq(vocabulary)]
        word_view = view.copy().astype(object)
        for column in word_view:
            digits = 0 if column == "Rank" or column.startswith("Papers") else (
                2 if "Share" in column or "pp" in column else 1)
            word_view[column] = view[column].map(
                lambda value, digits=digits: f"{value:,.{digits}f}" if pd.notna(value) else "—")
        registry.save_word_table(word_view, f"{stem}.docx",
            title=f"Fastest-growing {label} categories, {baseline_year}–{end_year}",
            notes=[f"Up to {top_n} categories ranked by compound annual growth rate; {len(rows)} eligible rows shown.", *notes],
            column_weights=[3.4, .5, .85, .85, 1.05, 1.15, .95, .95, .85])
    for name, frame in tables.items():
        registry.save_table(frame, name)
    registry.save_workbook(workbook, "content_growth_tables.xlsx")
    registry.save_text("\n\n".join(notes) + "\n", "content_growth_methods.txt")
    return {"all": all_rows, **ranked, "views": views}
