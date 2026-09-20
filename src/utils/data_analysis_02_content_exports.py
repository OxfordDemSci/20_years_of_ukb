"""Publication tables, captions and provenance for the consolidated content notebook."""

from __future__ import annotations

import pandas as pd

from utils import data_analysis_02_content_panels as panels
from utils.shared_analysis_window import ANALYSIS_START_DATE, ANALYSIS_END_DATE


def _caption(title, parts):
    return title + " " + " ".join(f"{letter}, {text}" for letter, text in parts.items()) + "\n"


def export_content_tables(data, registry):
    """Export the exact distributions and denominators used by the figures."""
    tables = {
        "content_annual_category_shares.csv": panels.category_share_table(data),
        "content_category_share_changes.csv": panels.category_change_table(data),
        "content_annual_coverage_and_breadth.csv": panels.annual_vocabulary_metrics(data),
        "content_displayed_band_coverage.csv": panels.band_summary(data),
        "content_topic_labels.csv": panels.topic_label_table(data),
    }
    # Retain the convenient matrix exports consumed by existing manuscript workflows.
    for key, name in (("for", "for_l4"), ("rcdc", "rcdc"), ("topics", "bertopic")):
        block = data[key]
        if block["available"]:
            tables[f"{name}_year_share.csv"] = block["share"].reset_index()
            tables[f"{name}_year_weights.csv"] = block["weights"].reset_index()
    parameters = {
        "analysis_start_date": str(ANALYSIS_START_DATE.date()),
        "analysis_end_date": str(ANALYSIS_END_DATE.date()),
        **data["counts"],
        "topic_results_available": data["topics"]["available"],
        "topic_results_source": data["topics"].get("source"),
        "for_counting": panels.WEIGHTS["for"],
        "rcdc_counting": panels.WEIGHTS["rcdc"],
        "topics_counting": panels.WEIGHTS["topics"],
    }
    tables["content_analysis_parameters.csv"] = pd.DataFrame(
        parameters.items(), columns=["parameter", "value"]
    )
    diagnostics = panels.load_topic_diagnostics(data)
    if diagnostics is not None:
        for key, frame in diagnostics.items():
            tables[f"content_topic_diagnostics_{key}.csv"] = frame
    for name, frame in tables.items():
        registry.save_table(frame, name)
    registry.save_text(
        _caption("Research composition and thematic evolution of UK Biobank publications.",
                 panels.MAIN_CAPTION),
        "02_01_figure_01_content_composition_caption.txt",
    )
    registry.save_text(
        _caption("Annual composition of leading research fields and RCDC categories.",
                 panels.SI_CAPTIONS["category_detail"]),
        "02_02_supplementary_figure_01_category_composition_caption.txt",
    )
    registry.save_text(
        _caption("Classification coverage and breadth of UK Biobank research.",
                 panels.SI_CAPTIONS["coverage"]),
        "02_03_supplementary_figure_02_coverage_and_breadth_caption.txt",
    )
    if diagnostics is not None:
        registry.save_text(
            _caption("Robustness and cluster persistence of the BERTopic analysis.",
                     panels.SI_CAPTIONS["topic_robustness"]),
            "02_04_supplementary_figure_03_topic_robustness_caption.txt",
        )
    registry.save_text(
        "The content analysis includes publications dated 1 January 2013 through "
        "31 December 2025, inclusive, from the canonical Showcase+ corpus. "
        "Missing dates may be resolved using publication year; contradictory or "
        "out-of-window dates are excluded by the shared analysis-window filter. "
        "Duplicate paper-category pairs are removed. FOR Level 4 composition uses "
        "whole paper-field assignments, normalised to the total number of assignments "
        "within each year; it is not the percentage of distinct papers carrying a field. "
        "RCDC composition divides one unit of credit equally across each paper's "
        "distinct tags and normalises across classified papers. Topic composition "
        "uses one non-outlier assignment per eligible paper. Missing classifications "
        "remain outside composition denominators and are quantified against the full "
        "yearly corpus in the coverage supplement. An empty classification year is "
        "shown as missing, never as a zero-percent composition. "
        "Main-panel categories are selected reproducibly by total in-window weight; "
        "all categories, annual denominators, and the shares covered by the displayed "
        "selection are exported. Topic waves are centred for display without "
        "renormalising the selected topics to 100%; vertical position has no meaning. "
        "Displayed topic labels are model-derived keyword summaries, with full labels "
        "retained in the topic-label CSV. Effective category diversity is the "
        "exponential of Shannon entropy. Category counts across vocabularies are not "
        "equivalent units. The small 2013 and 2014 cohorts require caution when "
        "interpreting early-year percentages or endpoint changes. "
        "Topic fitting is independent of figure generation: verified existing results "
        "are reused; otherwise five parameter configurations and five random seeds "
        "are compared using topic quality and assignment stability before the final "
        "global model is fitted. Coherence uses the same unigram, bigram and trigram "
        "analyser as the topic-word representation. The configured c-TF-IDF weighting "
        "is retained during outlier reassignment. All five seeds must succeed for "
        "all five parameter settings before selection. Publication year is not a "
        "model input feature.\n",
        "content_methods.txt",
    )
    return tables
