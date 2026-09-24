"""Suggested notebook captions, kept separate from exported figure artwork."""

from pathlib import Path


CLINICAL_CAPTIONS = {
    "ct_conditions_wordcloud": "Conditions studied in UK Biobank-linked clinical trials. "
        "Word size reflects fractional frequency after normalising the supplied free-text "
        "conditions and merging the documented synonyms; each trial spreads one unit of "
        "weight across its distinct normalised conditions.",
    "ct_country_map": "Geographical distribution of UK Biobank-linked clinical trials. "
        "Countries are shaded by trial counts and the leading countries are annotated. "
        "Multinational trials can contribute to more than one country.",
    "ct_diseases_mesh": "MeSH categories recorded for UK Biobank-linked clinical trials. "
        "(A) Raw trial counts and (B) fractional trial weights for the leading categories. "
        "The supplied terms can include disease descriptors, tree ancestors and intervention terms.",
    "ct_rcdc_all": "RCDC classifications of UK Biobank-linked clinical trials, including disease "
        "and cross-cutting categories. (A) Raw trial counts. (B) Fractional trial weights.",
    "ct_rcdc_disease": "Disease-related RCDC classifications of UK Biobank-linked clinical trials "
        "after excluding the documented cross-cutting tags. (A) Raw trial counts. "
        "(B) Fractional trial weights. Trials can carry more than one disease category.",
    "ct_rcdc_icd": "Disease classifications mapped to ICD-10 chapters using the documented keyword "
        "rules. (A) Trial counts based on RCDC disease tags. (B) Comparison of MeSH- and "
        "RCDC-derived counts across the same chapters. Each trial counts once per chapter it touches.",
    "ct_diseases_mesh_leaf": "Specific MeSH disease descriptors assigned to UK Biobank-linked "
        "clinical trials, excluding tree ancestors and intervention terms. (A) Raw trial "
        "counts. (B) Fractional trial weights.",
    "ct_diseases_mesh_ancestors": "MeSH tree ancestors associated with trial conditions. "
        "(A) Raw trial counts. (B) Fractional trial weights. These are broader hierarchical "
        "categories rather than the specific disease descriptors shown in the leaf analysis.",
    "ct_icd_three_axes": "Comparison of disease classification routes for UK Biobank-linked "
        "clinical trials. (A) ICD-10 chapters derived from MeSH leaf descriptor IDs and their "
        "tree positions. (B) Chapter counts compared with the supplied MeSH terms and RCDC "
        "disease tags mapped by keyword rules. The two MeSH routes use blue shades; RCDC "
        "uses gold. A trial can contribute to multiple chapters.",
    "ct_mesh_leaf_condition_icd_heatmaps": "Relationships between MeSH leaf diseases and "
        "(A) ICD-10 chapters or (B) normalised free-text trial conditions. Cells give the "
        "percentage of trials carrying the row disease that also touch the column condition "
        "or chapter; a trial can contribute to multiple cells. White cells marked with a dash "
        "represent zero; the shared colour scale spans the observed non-zero percentages.",
    "ct_ukbb_papers_fields_concepts": "Characteristics of UK Biobank publications cited by "
        "clinical trials. (A) Fields of Research and (B) research concepts, summarised using "
        "fractional publication weights across each paper's assignments.",
    "ct_field_concept_heatmap": "Fields and concepts of UK Biobank publications cited by "
        "clinical trials. (A) Field composition. (B) Concept frequencies within each field; "
        "row labels report the number of papers assigned to that field. White cells marked "
        "with a dash represent zero; the colour scale spans the observed non-zero percentages.",
    "ct_papers_timeliness": "Timing and publication venues of UK Biobank papers cited by "
        "clinical trials. (A) Publication years of cited papers. (B) Positive paper-to-trial "
        "lags among trial-paper links. (C) The leading journals of cited papers. "
        "Panel B excludes zero and negative lags.",
}


def panel_caption(parts):
    return " ".join(f"({letter}) {text}" for letter, text in parts.items())


def growth_captions(first_year, last_year, snapshot_date):
    """Captions for the three growth figures, displayed rather than saved as files."""
    FIRST_YEAR, LAST_COMPLETE_YEAR, DATA_SNAPSHOT_DATE = first_year, last_year, snapshot_date
    main_caption = (
        f"Figure 2 | Growth and reach of UK Biobank research, {FIRST_YEAR}–{LAST_COMPLETE_YEAR}. "
        "(A) Cumulative unique publications; inset curves show cumulative publications linked "
        "to patents, clinical trials, policy documents or datasets, with cross-sector "
        "collaboration, or with a positive aggregate Altmetric score. Cross-sector collaboration "
        "requires at least one Education organization and at least one Company, Government, "
        "Healthcare or Nonprofit organization. (B) Annual publication output, with 2018 and 2025 "
        "highlighted; the annotation reports compound annual growth over that interval. (C) "
        "Publications carrying each of the same six indicators per 1,000 publications across "
        "2013–2025, shown on a logarithmic axis. (D) Directional co-occurrence among the same "
        "six indicators across the complete-year corpus. Each off-diagonal cell is the percentage "
        "of publications carrying the row indicator that also carry the column indicator; the "
        "diagonal is omitted. Row colors identify indicators consistently across A, C and D. "
        "Publications contribute once to each "
        "applicable indicator and indicators may overlap. Linked-endpoint indicators reflect links "
        "to events dated from 1 January 2013 through 31 December 2025 and may be lower for recent cohorts because links accrue over "
        "time. Citation and Altmetric values are snapshot measures. Records "
        "outside 2013–2025 or without usable dates were excluded. News-specific mention counts were unavailable."
    )

    supplementary_1_caption = (
        f"Supplementary Figure 1 | Overlap between forms of reach among UK Biobank "
        f"publications, {FIRST_YEAR}–{LAST_COMPLETE_YEAR}. (A) Pairwise Jaccard similarity "
        "between seven binary publication indicators: publications carrying both divided by "
        "publications carrying either. The nonredundant lower triangle is shown; "
        "self-comparisons are omitted. (B) Percentage of complete-year publications carrying "
        "each indicator. Publications may carry multiple indicators. Publications and linked events "
        "are restricted to 1 January 2013–31 December 2025; positive aggregate Altmetric scores "
        f"reflect the {DATA_SNAPSHOT_DATE.strftime('%d %B %Y')} snapshot."
    )

    supplementary_2_caption = (
        f"Supplementary Figure 2 | Annual growth, collaboration and reach of UK Biobank "
        f"publications, {FIRST_YEAR}–{LAST_COMPLETE_YEAR}. (A) Annual publication output. "
        "(B) Median authors per paper with the interquartile range. (C) Year-on-year growth "
        "from 2018, with 2018 and 2025 highlighted. (D) Annual shares of cross-sector and "
        "international collaborations and Education–Company collaborations. (E) Annual shares "
        "with Altmetric attention, open-access status and preprint publication type. (F) Annual "
        "shares linked to grants, datasets, patents, policy documents and clinical trials, shown "
        "on a logarithmic scale. Publications and linked events are restricted to 2013–2025. "
        "Altmetric scores and open-access status reflect "
        "the source snapshot; recent publication cohorts may be affected by indexing, linkage "
        "and attention-accrual latency."
    )


    return {"main": main_caption, "overlap": supplementary_1_caption, "annual": supplementary_2_caption}


def caption_for_name(name):
    """Reuse curated panel descriptions without loading any analytical data."""
    from . import shared_paths as P

    stem = Path(str(name)).stem
    if stem in CLINICAL_CAPTIONS:
        return CLINICAL_CAPTIONS[stem]
    if stem.startswith("02_"):
        from . import data_analysis_02_content_panels as C
        sections = {"category_composition": "category_detail", "category_rank_flow": "rank_flow",
                    "coverage_and_breadth": "coverage", "topic_robustness": "topic_robustness"}
        if stem.startswith(P.MAIN_FIGURE_STEMS[4]):
            return panel_caption(C.MAIN_CAPTION)
        for suffix, section in sections.items():
            if stem.endswith(suffix):
                return panel_caption(C.SI_CAPTIONS[section])
    elif stem.startswith("03_"):
        from . import data_analysis_03_academic_impact_panels as C
        if stem == P.MAIN_FIGURE_STEMS[5]:
            return panel_caption(C.MAIN_CAPTION)
        for suffix, section in (("impact_map_full", "impact_map"),
                                ("top_decile_share", "top_decile"), ("growth", "growth")):
            if "supplementary_figure" in stem and stem.endswith(suffix):
                return panel_caption(C.SI_CAPTIONS[section])
    elif stem.startswith("04_"):
        from . import data_analysis_04_non_academic_panels as C
        if stem == P.MAIN_FIGURE_STEMS[6]:
            return panel_caption(C.MAIN_CAPTION)
        for suffix, section in (("patents", "patents"), ("clinical_trials", "trials"),
                                ("policy", "policy"), ("altmetric", "altmetric"),
                                ("collaboration", "collaboration")):
            if "supplementary_figure" in stem and stem.endswith(suffix):
                return panel_caption(C.SI_CAPTIONS[section])
    elif stem.startswith("05_"):
        from .data_analysis_05_author_characteristics import figure_captions
        captions = figure_captions()
        if stem == P.MAIN_FIGURE_STEMS[3]:
            return captions["figure_03_caption.txt"]
        for number in range(1, 7):
            if f"supplementary_figure_{number:02d}_" in stem:
                return captions[f"supplementary_figure_{number:02d}_caption.txt"]
    return None


def suggest_caption(fig):
    """Conservative fallback: describe recorded labels, never infer findings."""
    if getattr(fig, "_ukb_caption", None):
        return fig._ukb_caption
    title = getattr(fig, "_ukb_description", "")
    parts = []
    for ax in fig.axes:
        if not ax.get_visible() or hasattr(ax, "_colorbar"):
            continue
        description = getattr(ax, "_ukb_description", "")
        x, y = (" ".join(value.split()) for value in (ax.get_xlabel(), ax.get_ylabel()))
        label = ax.get_title(loc="left")
        if not description:
            description = f"{y} by {x}" if x and y else y or x
        if description:
            parts.append(f"({label}) {description.rstrip('.')}." if label else
                         description.rstrip('.') + ".")
    text = " ".join(([title.rstrip('.') + '.'] if title else []) + parts)
    return text or "UK Biobank research: the plotted quantities and groups are identified by the figure labels and legend."
