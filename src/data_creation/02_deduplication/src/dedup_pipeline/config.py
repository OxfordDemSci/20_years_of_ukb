"""Configuration shared across the deduplication pipeline."""

# Ranked keys in descending importance for presence-only informativeness scoring.
RANKED_COLS = [
    "id", "date_imported_gbq", "year", "doi", "isbn", "title", "abstract", "category_for",
    "reference_ids", "type", "researcher_ids", "authors", "patent_ids", "journal", "concepts",
    "funding_details", "funder_orgs", "research_orgs", "altmetrics", "citations", "citations_count",
    "volume", "issue", "pages", "subtitles", "date", "date_online", "book_title", "document_type",
    "funder_countries", "concepts_relevant", "date_print", "source_id", "date_accepted",
    "date_submitted", "categories", "book_series_title", "proceedings_title", "parent_id",
    "conference", "supporting_grant_ids", "repository_dois", "arxiv_id", "open_access_categories",
    "open_access_categories_v2", "pmid", "pmcid", "date_normal", "date_inserted", "eisbn", "publisher",
    "event", "editors", "resulting_publication_doi", "mesh_headings", "mesh_terms", "category_bra",
    "category_hra", "category_hrcs_hc", "category_hrcs_rac", "category_icrp_cso", "category_icrp_ct",
    "category_rcdc", "category_sdg", "category_uoa", "copyright_statement", "acknowledgements",
    "clinical_trial_ids", "pubmed", "metrics", "source", "journal_lists", "funding_section",
    "research_org_cities", "research_org_city_names", "research_org_countries",
    "research_org_country_names", "research_org_state_codes", "research_org_state_names",
]

IGNORE_PATCH_COLS = {"id", "file", "cluster_id", "_score"}

EMPTY_CLUSTERS_SCHEMA = {"id": [], "file": [], "cluster_id": []}
EMPTY_WANTED_SCHEMA = {"file": [], "ids": [], "n_ids": []}
EMPTY_FILES_TO_TOUCH_SCHEMA = {"file": [], "path": [], "n_ids": []}
EMPTY_KEEP_SCHEMA = {
    "cluster_id": [],
    "canonical_id": [],
    "canonical_file": [],
    "best_score": [],
    "total_members": [],
}
EMPTY_PATCH_SCHEMA = {
    "cluster_id": [],
    "canonical_id": [],
    "canonical_file": [],
    "patch_json": [],
    "columns_filled_json": [],
    "merged_from_json": [],
}
EMPTY_LOSER_SCHEMA = {"cluster_id": [], "id": [], "file": []}


def build_presence_weights(zero_id_weight=False):
    """Return descending weights for columns used in presence-only scoring."""
    n = len(RANKED_COLS)
    weights = {column: float(n - i) for i, column in enumerate(RANKED_COLS)}
    if zero_id_weight:
        weights["id"] = 0.0
    return weights
