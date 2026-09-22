"""Publication tables and provenance for the consolidated content notebook."""

from __future__ import annotations

import pandas as pd

from utils import data_analysis_02_content_panels as panels
from utils import shared_paths as P
from utils.shared_analysis_window import ANALYSIS_END_DATE, ANALYSIS_START_DATE


def export_content_tables(data, registry):
    """Export the exact distributions and denominators used by the figures."""
    tables = {
        "content_annual_category_shares.csv": panels.category_share_table(data),
        "content_category_share_changes.csv": panels.category_change_table(data),
        "content_annual_coverage_and_breadth.csv": panels.annual_vocabulary_metrics(data),
        "content_displayed_band_coverage.csv": panels.band_summary(data),
        "content_topic_labels.csv": panels.topic_label_table(data),
    }
    tables.update(panel_source_tables(data))
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
        "selection are exported. Topic composition is stacked on a zero-based "
        "percentage axis without renormalising the selected topics to 100%. "
        "Selected topics use concise display descriptors reviewed against their leading "
        "keywords and example publication titles; other topics retain keyword summaries. "
        "Full model labels, display labels and label sources are retained in the "
        "topic-label CSV. Display labels do not change assignments or aggregation. "
        "Effective category diversity is the "
        "exponential of Shannon entropy. Category counts across vocabularies are not "
        "equivalent units. The small 2013 and 2014 cohorts require caution when "
        "interpreting early-year percentages or endpoint changes. "
        "Topic fitting is independent of figure generation: verified existing results "
        "are reused; otherwise five parameter configurations and five random seeds "
        "are compared using topic quality and assignment stability before the final "
        "global model is fitted. SPECTER document embeddings are reduced to five "
        "dimensions with UMAP before HDBSCAN clustering. Outliers are reassigned "
        "using embedding similarity where possible; coverage and topic shares use "
        "the resulting assignments. Coherence uses the same unigram, bigram and trigram "
        "analyser as the topic-word representation. The configured c-TF-IDF weighting "
        "is retained during outlier reassignment. All five seeds must succeed for "
        "all five parameter settings before selection. Publication year is not a "
        "model input feature.\n",
        "content_methods.txt",
    )
    return tables


def panel_source_tables(data):
    """Retain the panel ledger and matrix exports without a second analysis pass."""
    rows = [{"figure": "main", "panel": letter, "caption": caption}
            for letter, caption in panels.MAIN_CAPTION.items()]
    rows += [{"figure": f"si_{section}", "panel": letter, "caption": caption}
             for section, captions in panels.SI_CAPTIONS.items()
             for letter, caption in captions.items()]
    tables = {
        "panel_band_rules.csv": panels.band_summary(data),
        "panel_selection.csv": pd.DataFrame(rows),
    }
    for key, stem in (("topics", "topics"), ("for", "for_l4"), ("rcdc", "rcdc")):
        block = data[key]
        if not block["available"]:
            continue
        changes = (panels.leading_category_change_table(block, n=10) if key == "for"
                   else panels.start_end_table(block["share"], block["keep"]))
        tables[f"panel_{stem}_year_share.csv"] = block["share"].round(4).reset_index()
        tables[f"panel_{stem}_drawn_bands.csv"] = block["band"].round(4).reset_index()
        tables[f"panel_{stem}_start_vs_end.csv"] = changes.reset_index()
    return tables


def export_panel_manifest(registry):
    """List only figures generated in this run, not stale files in the output folder."""
    frame = pd.DataFrame([
        {"file": P.raw_path(path), "kb": path.stat().st_size // 1024,
         "modified": pd.Timestamp(path.stat().st_mtime, unit="s").round("s")}
        for path in sorted(registry.figure_paths)
    ], columns=["file", "kb", "modified"])
    registry.save_table(frame, "panel_manifest.csv")
    return frame
