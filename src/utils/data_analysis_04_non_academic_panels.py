"""Assembled panels for analysis 04 — the non-academic impact figure family.

Part V of `04_non_academic.ipynb` owns the selection (which chart is main-paper, which is
SI); this module owns the two halves that selection needs and that a notebook cell is the
wrong place for:

    build_panel_data()      every aggregate the panels draw, computed once from the four
                            sources, returned as a dict of small frames
    draw_<name>(ax, D)      one chart into one caller-supplied axes, no figure of its own
    figure_main(D)          the main-paper panel;      figure_si_<section>(D) the SI ones

**Why re-drawn rather than pasted.** Parts I-IV each draw their charts with their own
figure size, type scale and palette — Part I uses the LCDS colours at dpi 400, Part II
carries a `colors_scheme` literal, and Part III has its own source style. Laid
side by side those differences read as four figures stapled together. Every function here
instead draws into an axes belonging to a figure the caller has already sized, under the
one `04_non_academic_panels` style section, so a panel's type is the panel's type and
"patents are steel blue" holds from panel A to panel F.

**Two inventories, used deliberately.** Publication reach uses Dimensions' full
reverse index, restricted to outcomes with known publication/start dates in the
2013–2025 analysis window. Descriptive patent panels use the existing detailed patent cohort,
filtered to the same publication window. Missing endpoint metadata cannot establish date eligibility
and therefore cannot enter the reach numerator. The raw source inventories are preserved.

**The sources, and which of them is the corpus.** Three of the five arms now read the WIDE
corpus export directly — one extraction, one index date — rather than a side CSV written by
a separate pull:

    publications     data/showcase/showcase+/showcase_plus_all_endpoints_wide.parquet
    patents          …the same file's `patents__*` block            (767 records, D43)
    clinical trials  …the same file's `clinical_trials__*` block    (195 records, 2026-09-22)
    policy           …the same file's `policy_documents__*` block   (449 records, 2026-09-22)
    altmetric        data/altmetric/altmetric.csv          (the real Explorer pull, D18 —
                     news mentions exist in NO other source, so this one stays a CSV)
    collaboration    data/analysis/non_academic/collaboration/…full_company.csv
                     (the LLM sector labels, keyed to corpus ids — an analysis output,
                     not a second retrieval)

The side CSVs at `data/analysis/non_academic/{clinical_trials,policy}/` are no longer read
for their records; `mesh_leaf_ids` is merged out of the trials one because the corpus block
does not carry it (see `LOCAL_ONLY_FIELDS` in the sources module).

The collaboration file is 339 MB, nearly all of it the `authors` column. The sector
taxonomy needs `research_orgs` and the classifier's index lists and *not* `authors`, so
this module reads ten columns instead of twenty-five and the taxonomy build takes ~10 s
rather than minutes. That is the only place it departs from the notebook's own pathway,
and it calls the notebook's own `add_non_academic_sector_taxonomy` to do the work.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd

from utils import shared_for as F
from utils import shared_paths as P
from utils import shared_rcdc as RCDC
from utils.shared_analysis_window import ANALYSIS_START_YEAR, ANALYSIS_END_YEAR, filter_analysis_window
from utils.data_analysis_04_non_academic_sources import filter_endpoint_links
from utils import shared_patent_utils as U
from utils.shared_showcase import endpoint_records

# =============================================================================
# The analysis window
# =============================================================================
# Paper and outcome dates use the shared inclusive 2013–2025 analysis window.
YEAR_MIN, YEAR_MAX = ANALYSIS_START_YEAR, ANALYSIS_END_YEAR
YEARS = list(range(YEAR_MIN, YEAR_MAX + 1))

#: Evidence streams, in the order they are told. Keys match `stream_colors` in
#: universal_settings.yml, so a panel asks the style for its colour by stream name.
STREAM_ORDER = ["patents", "clinical_trials", "policy", "altmetric", "collaboration"]

STREAM_LABELS = {
    "patents": "Patents",
    "clinical_trials": "Clinical trials",
    "policy": "Policy documents",
    "altmetric": "News & policy attention",
    "collaboration": "Non-academic collaboration",
}

#: The collaborator taxonomy's eight sectors, ordered public-sector first. Same strings as
#: `NON_ACADEMIC_SECTOR_LABELS` in the collaboration helpers — imported rather than
#: retyped where that module is available, listed here for the ordering.
SECTOR_ORDER = [
    "University/HEI",
    "Hospital/Clinical",
    "Research institute/Centre",
    "Government/Public",
    "Nonprofit/Charity",
    "Company (non-UK)",
    "UK company",
    "Other/Unknown",
]

#: The five sectors that are actually *non*-academic. University/HEI is in the taxonomy
#: because the classifier labels every org, not because a university is a non-academic
#: partner; Other/Unknown is a residual. Panels about "non-academic collaboration" use
#: this list, not SECTOR_ORDER, and the difference is the whole point of the figure.
NON_ACADEMIC_SECTORS = [
    "Hospital/Clinical",
    "Research institute/Centre",
    "Government/Public",
    "Nonprofit/Charity",
    "Company (non-UK)",
    "UK company",
]

COMPANY_SECTORS = ["Company (non-UK)", "UK company"]

#: The policy and attention figures are drawn in ONE colour family from end to end rather
#: than rotating the palette panel by panel. `stream_colors` still gives every stream its
#: own hue, because the main figure's panel A draws five of them on one axes and telling
#: them apart is the whole point of that panel. Inside a figure that is *about* one
#: stream, a second hue says only "different panel", which the panel letter already says,
#: and it leaves the reader hunting for a distinction that is not being drawn.
#:
#: The main attention cloud uses light blue with navy highlights. The supplementary
#: news/policy series use the same steel-blue/red identities as the earlier trend figure.
POLICY_PRIMARY = "steel_blue"     # #416FA0
POLICY_SECONDARY = "red"          # #E66859 - the UK / rest-of-world split
ATTENTION_PRIMARY = "steel_blue"  # #416FA0
ATTENTION_SECONDARY = "red"       # #E66859 - highlighted publications in the main cloud
ATTENTION_NEWS = "steel_blue"
ATTENTION_POLICY = "red"

#: The policy choropleth's ramp: white through the palette's three blues. Built from the
#: named palette rather than from a Matplotlib map so the map and the bars beside it are
#: the same four colours.
POLICY_MAP_RAMP = ["#FFFFFF", "blue", "steel_blue", "navy"]

# =============================================================================
# Organisation sector, for the clinical-trials "who runs them" panel
# =============================================================================
# Verbatim from Part I, §2, so the panel and the source part place the same organisation
# in the same sector. Two steps, because ~5% of orgs
# arrive with a non-committal GRID type ('Facility', 'Other') or none at all, and those
# were nearly all universities, companies and hospitals GRID had simply not classified.
ORG_TYPE_TO_SECTOR = {
    "Education": "Academia",
    "Company": "Industry",
    "Healthcare": "Healthcare",
    "Government": "Government",
    "Nonprofit": "Nonprofit",
}

# Tried IN ORDER, first match wins, and the order carries three decisions:
#   * industry and government are the most specific signals, so they go first;
#   * healthcare BEFORE academia, so a teaching hospital ("University Hospital of X",
#     "Centre Hospitalier Universitaire") counts as care delivery, not as a university;
#   * the research-institute rule has no trailing \b — accented characters
#     ("Investigación") are word characters, so \b would fail mid-word.
ORG_NAME_RULES = [
    ("Industry", r"\b(inc|corp|corporation|ltd|llc|gmbh|pharma\w*|therapeutics|biologics|"
                 r"biosciences|technologies|diagnostics|laboratories)\b"),
    ("Government", r"\b(ministry|statens|national institutes? of health|public health|"
                   r"serum institut|centers? for disease|agency)\b"),
    ("Healthcare", r"\b(hospital|hospitalier|medical cent(er|re)|clinic|health system|"
                   r"infirmary|nhs|clinical research facility)\b"),
    ("Academia", r"\b(univer\w+|college|school of medicine|institute of technology|"
                 r"polytechnic|université|academ\w+)\b"),
    ("Academia", r"\b(research institut|research cent|clinical research|biomedical research|"
                 r"institut|fondazione|centro de investiga|biocenter|foundation)"),
]

TRIAL_SECTOR_ORDER = ["Academia", "Healthcare", "Industry", "Government", "Nonprofit", "Other"]


def org_sector(org: dict) -> str:
    """Sector for one research-org dict: a clear GRID type wins, else infer from the name."""
    for org_type in (org.get("types") or []):
        if org_type in ORG_TYPE_TO_SECTOR:
            return ORG_TYPE_TO_SECTOR[org_type]
    name = (org.get("name") or "").lower()
    for sector, pattern in ORG_NAME_RULES:
        if re.search(pattern, name):
            return sector
    return "Other"


#: Keyword -> ICD-10 chapter, checked in order, first hit wins. Lifted verbatim from
#: Part I, §3.3, so the body map and the source part's own
#: ICD figures agree term for term. Circulatory is listed before nervous so "stroke"
#: (I60-I69) lands in circulatory rather than under a "sclerosis"-style nervous match.
ICD_CHAPTERS = [
    ("II  Neoplasms (C00–D48)",
     ["neoplasm", "cancer", "carcinoma", "tumor", "tumour", "lymphoma", "leukemia",
      "leukaemia", "melanoma", "sarcoma", "malignan", "adenoma"]),
    ("IV  Endocrine, nutritional & metabolic (E00–E90)",
     ["diabet", "obes", "overweight", "metaboli", "nutrition", "lipid", "cholesterol",
      "thyroid", "glucose", "insulin", "hyperglyc", "dyslipid", "hyperlipid"]),
    ("IX  Circulatory system (I00–I99)",
     ["cardiovascular", "cardiac", "heart", "coronary", "atrial fibrillation", "arrhythmia",
      "myocard", "vascular", "hypertension", "atheroscler", "thrombo", "ischemi", "ischaemi",
      "stroke", "aneurysm", "angina", "cardiomyopath", "embolism"]),
    ("V   Mental & behavioural (F00–F99)",
     ["depress", "anxiety", "psychiatr", "psychosis", "schizophren", "bipolar", "mood",
      "mental disorder", "substance", "addiction", "alcohol"]),
    ("VI  Nervous system (G00–G99)",
     ["alzheimer", "dementia", "parkinson", "epilep", "migraine", "sclerosis",
      "neurodegener", "cognitive", "neuropath"]),
    ("X   Respiratory system (J00–J99)",
     ["asthma", "copd", "respiratory", "pulmonary disease", "lung disease", "pneumon",
      "bronch"]),
    ("XI  Digestive system (K00–K93)",
     ["liver", "hepat", "nafld", "gastro", "digest", "bowel", "crohn", "colitis",
      "pancrea", "cirrhosis", "fatty liver", "steato"]),
    ("XIII Musculoskeletal (M00–M99)",
     ["arthr", "osteo", "musculoskeletal", "sarcopenia", "bone", "joint", "rheumat"]),
    ("XIV Genitourinary (N00–N99)",
     ["kidney", "renal", "urinary", "prostat", "nephro", "bladder"]),
    ("XV  Pregnancy & childbirth (O00–O99)",
     ["pregnan", "eclampsia", "obstetric", "gestational"]),
    ("I   Infectious & parasitic (A00–B99)",
     ["infect", "covid", "viral", "bacteri", "sepsis", "hiv"]),
]

#: Schematic chapter locations, not exact organ locations or evidence of causality.
#: Left/right refer to the viewer; systemic chapters are deliberately off the body.
ICD_BODY_POS = {
    "V   Mental & behavioural (F00–F99)":               (5.00, 13.20, "Mental/behavioural", "left"),
    "VI  Nervous system (G00–G99)":                     (5.00, 12.30, "Nervous", "right"),
    "X   Respiratory system (J00–J99)":                 (4.40, 10.90, "Respiratory", "left"),
    "IX  Circulatory system (I00–I99)":                 (5.50, 10.55, "Circulatory", "right"),
    "IV  Endocrine, nutritional & metabolic (E00–E90)": (4.75, 9.45, "Endocrine/metabolic", "left"),
    "XI  Digestive system (K00–K93)":                   (5.00, 8.60, "Digestive", "right"),
    "XIV Genitourinary (N00–N99)":                      (4.50, 7.75, "Genitourinary", "left"),
    "XV  Pregnancy & childbirth (O00–O99)":             (5.50, 7.75, "Pregnancy", "right"),
    "XIII Musculoskeletal (M00–M99)":                   (4.45, 5.60, "Musculoskeletal", "left"),
    "II  Neoplasms (C00–D48)":                          (7.40, 4.65, "Neoplasms", "systemic"),
    "I   Infectious & parasitic (A00–B99)":             (7.40, 3.30, "Infectious", "systemic"),
}

#: Condition-leaf MeSH terms that are not diseases. CTgov maps condition text like
#: "physical activity" or "healthy" onto behavioural or descriptive MeSH descriptors, so
#: they arrive in `mesh_leaf` looking exactly like a diagnosis. Same list, same reason, as
#: `MESH_LEAF_STOP` in Part I, §3.5 — kept identical so the
#: panel and the source notebook rank the same terms.
MESH_LEAF_STOP = {
    "Motor Activity", "Health Behavior", "Genetic Risk Score", "Psychological Well-Being",
    "Weight Loss", "Chronic Disease", "Body Weight", "Life Style", "Exercise",
}


# =============================================================================
# Small parsing helpers
# =============================================================================
def _lst(value) -> list:
    """Stringified list -> list. The CSVs round-trip their list columns as text."""
    return F.parse_listcol(value)


def _dct(value) -> dict:
    """Stringified dict -> dict; anything else -> {}."""
    if isinstance(value, dict):
        return value
    parsed = F.parse_listcol(f"[{value}]") if isinstance(value, str) else []
    return parsed[0] if parsed and isinstance(parsed[0], dict) else {}


def _clip_years(frame: pd.DataFrame, col: str = "year") -> pd.DataFrame:
    """Restrict to the analysis window and reindex onto every year in it (0-filled)."""
    out = frame[(frame[col] >= YEAR_MIN) & (frame[col] <= YEAR_MAX)].copy()
    return out


def _on_years(series: pd.Series) -> pd.Series:
    """A year-indexed count Series reindexed onto YEARS with 0 for the missing ones."""
    return series.reindex(YEARS, fill_value=0).astype(int)


def _wrap(label: str, width: int = 28) -> str:
    """Wrap a category label for a tick, on spaces, without importing textwrap per call."""
    words, lines, cur = str(label).split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def _names(cell) -> list[str]:
    """The `name` of every entry in a list column, whether dicts or bare strings."""
    out = []
    for value in _lst(cell):
        name = value.get("name") if isinstance(value, dict) else value
        if name and str(name).strip():
            out.append(str(name))
    return out


def _norm_icd_term(term: str) -> str:
    """Lowercase for keyword matching, defusing one substring trap.

    "Non-alcoholic Fatty Liver Disease" contains "alcohol", and the mental & behavioural
    rules are tested before digestive — so every NAFLD trial used to land in F00-F99.
    Stripping the negation makes it match "liver" -> XI Digestive, which is right
    (NAFLD is K76.0).
    """
    return term.lower().replace("non-alcoholic", " ").replace("nonalcoholic", " ")


def map_icd_chapter(term: str):
    """First-matching ICD-10 chapter for a disease term, else None (unmapped)."""
    normalised = _norm_icd_term(str(term))
    for chapter, keywords in ICD_CHAPTERS:
        if any(k in normalised for k in keywords):
            return chapter
    return None


def _shorten(text, limit: int = 68) -> str:
    """Truncate a paper title on a word boundary — a tick label, not a bibliography."""
    text = str(text).strip()
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut + " …"


def _for_column(frame: pd.DataFrame) -> str:
    """Which FOR column this table carries. Dimensions renamed it mid-project."""
    for col in ("category_for_2020", "category_for"):
        if col in frame.columns:
            return col
    raise KeyError("no category_for_2020 / category_for column to read divisions from")


def _for_divisions(cell) -> list[str]:
    """The FOR 2020 DIVISION names (2-digit, L2) in one category cell, deduped."""
    labels = set()
    for entry in _lst(cell):
        if not isinstance(entry, dict):
            continue
        _, label, level = F.split_for_name(entry.get("name"))
        if level == "L2" and label:
            labels.add(label)
    return sorted(labels)


def _top_counter(counter: Counter, n: int) -> pd.Series:
    """The n largest entries of a Counter as a Series, largest first."""
    items = counter.most_common(n)
    return pd.Series(dict(items), dtype="int64") if items else pd.Series(dtype="int64")


# =============================================================================
# 1. Loaders
# =============================================================================
#: Dimensions' own publication -> endpoint linkage, one column per stream, present only
#: in the WIDE corpus export. `patents__linked_ids` on a publication row lists every
#: patent Dimensions knows cites that paper.
LINKED_ID_COLUMNS = {
    "patents": "patents__linked_ids",
    "clinical_trials": "clinical_trials__linked_ids",
    "policy": "policy_documents__linked_ids",
}


def load_corpus() -> pd.DataFrame:
    """id / year / doi / times_cited / altmetric, plus the endpoint linkage if present.

    The linkage columns only exist in the wide export (317 columns); the narrow one has
    72 and none of them. They are requested opportunistically rather than unconditionally
    because `pd.read_parquet(columns=...)` raises on a column the file does not have, and
    a missing linkage should degrade this module to its pull-based fallback, not stop it.
    """
    import pyarrow.parquet as pq

    available = set(pq.read_schema(P.SHOWCASE_PLUS).names)
    wanted = ["id", "year", "doi", "times_cited", "altmetric"]
    wanted += ["date"] if "date" in available else []
    wanted += [c for c in LINKED_ID_COLUMNS.values() if c in available]
    corpus = filter_analysis_window(pd.read_parquet(P.SHOWCASE_PLUS, columns=wanted))
    corpus = filter_endpoint_links(corpus)
    corpus["year"] = pd.to_numeric(corpus["year"], errors="coerce")
    corpus["doi_clean"] = corpus["doi"].astype("string").str.strip().str.lower()
    return corpus


#: The patent fields the panels read, as they are named inside the corpus's `patents__*`
#: endpoint block. Everything the retired CSV export carried beyond these (cleaned abstracts,
#: topic lists, a primary-country pick) is derived and unused here — the panels read divisions
#: off the FOR column and countries off `assignee_countries`.
PATENT_ENDPOINT_FIELDS = (
    "id", "legal_status", "filing_status", "publication_year", "publication_date",
    "priority_year", "granted_year", "assignee_countries", "category_for_2020",
    "category_rcdc", "publication_ids",
)


def load_patents() -> pd.DataFrame:
    """Every patent citing a UK Biobank publication, from the corpus's own endpoint block.

    **Source repointed 2026-09-20 (D43); window filter kept from the 2026-09-21 pass.** This
    used to read `patents_modularized_export.csv`, derived from a separate, earlier patent
    query returning 513 records. The corpus — one run of the endpoint pipeline, which collects
    publications and all six endpoints together — carries **767**, a strict superset, at a
    single index date; 24 of the shared 513 had already drifted on `legal_status`. Reading the
    corpus closes a coverage gap and a date gap at once and needs no new extraction.

    The fixed analysis window is then applied exactly as it is to every other arm, on both the
    publication year and the publication date, so patents carry no special-cased window.

    Falls back to the CSV export when the corpus has no `patents__*` block (the narrow export),
    with a printed notice — the two give different counts, and a silent switch would be worse
    than either.
    """
    import pyarrow.parquet as pq

    available = set(pq.read_schema(P.SHOWCASE_PLUS).names)
    if f"patents__{PATENT_ENDPOINT_FIELDS[0]}" not in available:
        export = P.PATENT / "patents_modularized_export.csv"
        if not export.exists():
            raise FileNotFoundError(
                f"The corpus at {P.raw_path(P.SHOWCASE_PLUS)} carries no `patents__*` "
                f"endpoint columns, and the fallback export {P.raw_path(export)} is missing "
                f"too. Either point P.SHOWCASE_PLUS at the wide export, or run "
                f"Part II of 04_non_academic.ipynb through §5."
            )
        print(f"  ! corpus has no patents__* block — falling back to {P.raw_path(export)} "
              f"(the 513-patent pull, an older and smaller set)")
        pat = pd.read_csv(export)
    else:
        pat = endpoint_records("patents", PATENT_ENDPOINT_FIELDS)
        # Dimensions writes "N/A" where a status is unknown; the panels drop missing rows
        # rather than drawing a category for them.
        for column in ("legal_status", "filing_status"):
            pat[column] = pat[column].replace({"N/A": None})
        pat["legal_status_replaced"] = pat["legal_status"].replace(U.LEGAL_STATUS_DISPLAY)
        # `assignee_countries` arrives as {id, name} dicts here and as bare ISO-2 codes in the
        # retired CSV export. The panels count bare codes, so flatten to the id.
        pat["assignee_countries"] = pat["assignee_countries"].apply(
            lambda cell: [e.get("id") for e in _lst(cell)
                          if isinstance(e, dict) and e.get("id")]
        )

    n_linked = len(pat)
    pat = filter_analysis_window(pat, year_col="publication_year",
                                 date_col="publication_date")
    # How many linked patents the window left behind, carried the way `filter_analysis_window`
    # carries its own dates: a caption has to be able to say 696 OF 767 without re-reading.
    pat.attrs["n_linked_before_window"] = n_linked
    pat.attrs["n_outside_window"] = n_linked - len(pat)
    for column in ("publication_year", "priority_year", "granted_year"):
        pat[column] = pd.to_numeric(pat[column], errors="coerce")
    return pat


def load_trials() -> pd.DataFrame:
    """Clinical trials, with start_year and the parsed list columns the panels need.

    **Source repointed 2026-09-22**, the same move D43 made for patents: the records come
    out of the corpus's `clinical_trials__*` block rather than whatever CSV happens to sit
    in `data/analysis/non_academic/clinical_trials/`. Both hold the same 195 trials, so no
    number on any panel moves — what changes is that the arm can no longer silently fall
    onto an older extraction. `mesh_leaf_ids` is merged back from the CSV; see
    `LOCAL_ONLY_FIELDS` for why it is the one field the corpus cannot supply.
    """
    from utils.data_analysis_04_non_academic_sources import clinical_trials_records
    ct = filter_analysis_window(clinical_trials_records(),
                                year_col="start_year", date_col="start_date")
    ct["start_year"] = pd.to_datetime(ct["start_date"], errors="coerce").dt.year
    ct["study_type"] = ct["study_type"].fillna("Unknown")
    return ct


def load_policy() -> pd.DataFrame:
    """Policy documents, with the publisher country flattened out of its dict.

    **Source repointed 2026-09-22**, with `load_trials`: the corpus's
    `policy_documents__*` block instead of the side CSV. Same 449 documents either way.
    """
    from utils.data_analysis_04_non_academic_sources import policy_records
    pol = filter_analysis_window(policy_records(), year_col="year")
    pol["year"] = pd.to_numeric(pol["year"], errors="coerce")
    country = pol["publisher_org_country"].apply(_dct)
    pol["publisher_country"] = country.apply(lambda d: d.get("name") or "Unknown")
    pol["publisher_country_code"] = country.apply(lambda d: d.get("id") or "")
    pol["publisher_name"] = pol["publisher_org"].apply(_dct).apply(
        lambda d: d.get("name") or "Unknown"
    )
    return pol


def load_altmetric(corpus: pd.DataFrame) -> pd.DataFrame:
    """The Altmetric Explorer export joined to the corpus on a cleaned DOI.

    D18: this is the *real* pull, so `News mentions` is a measurement. The rebuilt-from-
    corpus substitute at `P.ALTMETRIC_DERIVED` carries a news column that is 0 for every
    row, meaning unknown — a figure drawn on it would be reporting an artefact of a
    missing file as a finding, so this refuses the substitute rather than falling back.
    """
    if not P.ALTMETRIC_CSV.exists():
        raise FileNotFoundError(
            f"{P.raw_path(P.ALTMETRIC_CSV)} is missing — that is the real Altmetric "
            f"Explorer export. The rebuilt table at {P.raw_path(P.ALTMETRIC_DERIVED)} "
            f"cannot substitute: its 'News mentions' is 0 for every row because no "
            f"source for it exists, and 0 there means unknown, not zero coverage."
        )
    alt = filter_analysis_window(pd.read_csv(P.ALTMETRIC_CSV), year_col="Year",
                                 date_col="Publication Date")
    alt["doi_clean"] = alt["DOI"].astype("string").str.strip().str.lower()
    alt["pub_date"] = pd.to_datetime(alt["Publication Date"], errors="coerce")
    alt["alt_year"] = alt["pub_date"].dt.year
    for col in ["Altmetric Attention Score", "News mentions", "Policy mentions"]:
        alt[col] = pd.to_numeric(alt[col], errors="coerce").fillna(0)
    alt["substantive"] = alt["News mentions"] + alt["Policy mentions"]

    # One corpus row per DOI before the join. Four DOIs appear twice in the corpus
    # (the same paper indexed under two Dimensions ids); left unhandled they duplicate
    # the Altmetric row and inflate every count taken off the merge. Keep the most-cited,
    # which is the same rule the source notebook applies on its side of the join.
    corpus_doi = (corpus[["id", "year", "doi_clean", "times_cited"]]
                  .dropna(subset=["doi_clean"])
                  .sort_values("times_cited", ascending=False)
                  .drop_duplicates(subset="doi_clean", keep="first"))
    joined = alt.merge(corpus_doi, on="doi_clean", how="inner")
    # Only eligible corpus papers enter the analysis; attention/citation totals remain
    # snapshot totals because per-event dates are not available.
    joined["year"] = joined["year"].fillna(joined["alt_year"])
    return filter_analysis_window(joined, year_col="year", date_col="pub_date")


def load_collaboration() -> pd.DataFrame:
    """The classified corpus with the eight sector labels derived, ten columns wide.

    `authors` is deliberately not read: it is ~95% of the 339 MB file and the taxonomy
    does not use it (the sector of an organisation comes from `research_orgs` plus the
    classifier's index lists). Section 7 of the source notebook, which does need authors
    for its journal table, is not reproduced in these panels.
    """
    from utils import data_analysis_04_non_academic_collab_helpers as h

    if not P.COLLAB_FLAGGED.exists():
        raise FileNotFoundError(
            "Missing saved organisation classifications: "
            f"{P.raw_path(P.COLLAB_FLAGGED)}. Restore the labelled dataset before "
            "assembling collaboration panels."
        )
    cols = [
        "id", "year", "times_cited", "category_for_2020", "research_orgs",
        "research_org_countries", "academic_indices", "non_academic_indices",
        "company_indices", "uk_company_indices",
    ]
    available = pd.read_csv(P.COLLAB_FLAGGED, nrows=0).columns
    cols += ["date"] if "date" in available else []
    collab = filter_analysis_window(pd.read_csv(P.COLLAB_FLAGGED, usecols=cols))
    collab["year"] = pd.to_numeric(collab["year"], errors="coerce")
    collab = h.add_non_academic_sector_taxonomy(collab)
    return collab


# =============================================================================
# 2. Aggregates
# =============================================================================
def _linked_paper_ids(frame: pd.DataFrame, corpus_ids: set[str],
                      col: str = "publication_ids") -> set[str]:
    """The corpus papers an artefact table references, as a set of Dimensions ids."""
    out: set[str] = set()
    for cell in frame[col]:
        out.update(pid for pid in _lst(cell) if pid in corpus_ids)
    return out


def linked_papers(corpus: pd.DataFrame, stream: str, pull: pd.DataFrame) -> tuple[set, str]:
    """Papers linked to eligible outcomes, preferring the wider publication index.

    ``load_corpus`` has already removed links to outcomes outside 2013–2025 or with
    unknown dates. Descriptive patent panels use the separate detailed pull. For
    narrow corpus exports without reverse linkage, the filtered artefact pull is
    the fallback. Return the linked paper IDs and their inventory provenance.
    """
    column = LINKED_ID_COLUMNS.get(stream)
    if column and column in corpus.columns:
        linked = corpus.loc[corpus[column].apply(lambda v: bool(_lst(v))), "id"]
        return set(linked), "Dimensions publication index, outcomes in 2013–2025"
    return _linked_paper_ids(pull, set(corpus["id"])), "artefact pull"


def _links_long(frame: pd.DataFrame, corpus_ids: set[str], year_col: str,
                col: str = "publication_ids") -> pd.DataFrame:
    """One row per (artefact, cited corpus paper) link, carrying the artefact's year."""
    rows = []
    for artefact_year, cell in zip(frame[year_col], frame[col]):
        for pid in _lst(cell):
            if pid in corpus_ids:
                rows.append((pid, artefact_year))
    return pd.DataFrame(rows, columns=["paper_id", "artefact_year"])


def build_reach(corpus, patents, trials, policy, altmetric, collab) -> pd.DataFrame:
    """How many UK Biobank papers carry each kind of non-academic linkage.

    One row per stream: the count of distinct corpus papers, and that as a share of the
    corpus. This is the figure that says how big each of the four arms actually is, and
    it is deliberately measured on the same denominator throughout.
    """
    corpus_ids = set(corpus["id"])
    n_corpus = len(corpus)

    alt_by_paper = altmetric.dropna(subset=["id"]).groupby("id")[
        ["News mentions", "Policy mentions", "Altmetric Attention Score"]
    ].max()

    company_flag = collab[[_sector_flag(s) for s in COMPANY_SECTORS]].max(axis=1)
    non_acad_flag = collab[[_sector_flag(s) for s in NON_ACADEMIC_SECTORS]].max(axis=1)

    rows = [
        ("patents", "Cited by a patent",
         len(linked_papers(corpus, "patents", patents)[0])),
        ("clinical_trials", "Cited by a clinical trial",
         len(linked_papers(corpus, "clinical_trials", trials)[0])),
        ("policy", "Cited by a policy document",
         len(linked_papers(corpus, "policy", policy)[0])),
        ("altmetric", "1+ news mentions",
         int((alt_by_paper["News mentions"] > 0).sum())),
        ("altmetric", "1+ Altmetric policy mentions",
         int((alt_by_paper["Policy mentions"] > 0).sum())),
        ("collaboration", "1+ non-academic collaborators",
         int(non_acad_flag.sum())),
        ("collaboration", "1+ company collaborators",
         int(company_flag.sum())),
    ]
    reach = pd.DataFrame(rows, columns=["stream", "linkage", "papers"])
    reach["pct"] = 100 * reach["papers"] / n_corpus
    reach["n_corpus"] = n_corpus
    sources = {stream: linked_papers(corpus, stream, pull)[1]
               for stream, pull in [("patents", patents),
                                    ("clinical_trials", trials),
                                    ("policy", policy)]}
    reach["source"] = reach["stream"].map(sources).fillna("")
    return reach


def build_reach_by_year(corpus, patents, trials, policy, altmetric, collab) -> pd.DataFrame:
    """Cumulative count of linked UK Biobank papers, by the paper's publication year.

    Every stream is measured in the same unit — *papers*, dated by when the paper came
    out — so the five lines are comparable. Counting artefacts instead (patents by
    filing year, trials by start year) would put five different units on one axis.
    """
    corpus_ids = set(corpus["id"])
    year_of = dict(zip(corpus["id"], corpus["year"]))

    def cumulative(paper_ids: Iterable[str]) -> pd.Series:
        years = pd.Series([year_of.get(pid) for pid in paper_ids]).dropna()
        counts = _on_years(years.astype(int).value_counts().sort_index())
        return counts.cumsum()

    alt_by_paper = altmetric.dropna(subset=["id"]).groupby("id")[
        ["News mentions", "Policy mentions"]
    ].max()
    news_ids = alt_by_paper.index[alt_by_paper["News mentions"] > 0]

    company_flag = collab[[_sector_flag(s) for s in COMPANY_SECTORS]].max(axis=1)

    series = {
        "patents": cumulative(linked_papers(corpus, "patents", patents)[0]),
        "clinical_trials": cumulative(linked_papers(corpus, "clinical_trials", trials)[0]),
        "policy": cumulative(linked_papers(corpus, "policy", policy)[0]),
        "altmetric": cumulative(news_ids),
        "collaboration": cumulative(collab.loc[company_flag == 1, "id"]),
    }
    return pd.DataFrame(series, index=YEARS).rename_axis("year")


def _sector_flag(label: str) -> str:
    """Column name of a sector's per-paper flag, as the collaboration helpers spell it."""
    slug = re.sub(r"[^a-z0-9]+", "_", label.casefold()).strip("_")
    return f"sector_{slug}_flag"


def _sector_count(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", label.casefold()).strip("_")
    return f"sector_{slug}_n"


# ---------------------------------------------------------------- patents ----
def build_patent_aggregates(patents: pd.DataFrame, corpus: pd.DataFrame) -> dict:
    """Everything the patent panels draw."""
    out = {}

    # Coverage, for the captions. The analysis window is enforced in `load_patents`, so every
    # aggregate below is already inside it; these two numbers are the only way a caption can
    # still say how many linked patents there were before it was applied.
    out["n_patents"] = int(len(patents))
    out["n_patents_linked"] = int(patents.attrs.get("n_linked_before_window", len(patents)))
    out["n_patents_outside_window"] = int(patents.attrs.get("n_outside_window", 0))

    # (a) filing status by publication year. `filing_status` is the two-way split
    # (Application / Grant); `legal_status_replaced` is the seven-way outcome.
    status = (patents.dropna(subset=["publication_year", "filing_status"])
              .astype({"publication_year": int})
              .groupby(["publication_year", "filing_status"]).size()
              .unstack(fill_value=0).sort_index())
    out["status_by_year"] = status

    legal = (patents.dropna(subset=["publication_year", "legal_status_replaced"])
             .astype({"publication_year": int})
             .groupby(["publication_year", "legal_status_replaced"]).size()
             .unstack(fill_value=0).sort_index())
    out["legal_status_by_year"] = legal

    # (b) assignee countries. `assignee_countries` is a list of ISO-2 codes; a patent with
    # assignees in two countries counts once in each, which is why the total exceeds 513.
    countries = Counter()
    for cell in patents["assignee_countries"]:
        countries.update({str(c) for c in _lst(cell) if c})
    out["countries"] = _top_counter(countries, 12)

    # (c) research divisions, read off the FOR column rather than off the notebook's
    # `top_level_topics`. That column maps every FOR entry to its 2-digit division code
    # but keeps the entry's own label, so division 32 arrives labelled both "Biomedical
    # and Clinical Sciences" (from the L2 entry) and "Clinical Sciences" (from an L4 one
    # beneath it) — which of the two a count lands under then depends on set iteration
    # order and is not stable between runs. `split_for_name` labels an L2 entry as an L2
    # entry, so a division is one division with one name.
    divisions_per_patent = patents[_for_column(patents)].apply(_for_divisions)
    topics = Counter()
    for labels in divisions_per_patent:
        topics.update(labels)
    out["topics"] = _top_counter(topics, 10)
    out["topics_per_patent"] = divisions_per_patent.map(len)

    # (d) country x topic, as a within-country percentage.
    rows = []
    for cell_c, labels in zip(patents["assignee_countries"], divisions_per_patent):
        iso = sorted({str(c) for c in _lst(cell_c) if c})
        if not iso or not labels:
            continue
        for country in iso:
            for label in labels:
                rows.append((country, label))
    ct = pd.DataFrame(rows, columns=["country", "topic"])
    top_c = [c for c in out["countries"].head(8).index]
    top_t = [t for t in out["topics"].head(8).index]
    pivot = (ct[ct["country"].isin(top_c) & ct["topic"].isin(top_t)]
             .groupby(["country", "topic"]).size().unstack(fill_value=0)
             .reindex(index=top_c, columns=top_t, fill_value=0))
    out["country_topic_pct"] = pivot.div(pivot.sum(axis=1).replace(0, np.nan), axis=0) * 100

    # (e) how long from paper to patent. Priority year is the patent's own earliest claim
    # date, so paper -> priority is the lag that matters; a negative lag is a patent whose
    # priority predates the paper it later cites, which happens and is kept.
    corpus_ids = set(corpus["id"])
    year_of = dict(zip(corpus["id"], corpus["year"]))
    lags = []
    for pyear, cell in zip(patents["priority_year"], patents["publication_ids"]):
        if pd.isna(pyear):
            continue
        for pid in _lst(cell):
            if pid in corpus_ids and pd.notna(year_of.get(pid)):
                lags.append(int(pyear) - int(year_of[pid]))
    out["paper_to_patent_lag"] = pd.Series(lags, dtype="int64")

    # (f) the most-cited UK Biobank papers, by how many patents reference them.
    counter = Counter()
    for cell in patents["publication_ids"]:
        counter.update(pid for pid in set(_lst(cell)) if pid in corpus_ids)
    top = _top_counter(counter, 12)
    titles = pd.read_parquet(P.SHOWCASE_PLUS, columns=["id", "title"]).set_index("id")["title"]
    out["top_cited_papers"] = pd.Series(
        top.values, index=[str(titles.get(i, i)) for i in top.index], dtype="int64"
    )
    return out


def build_trial_aggregates(trials: pd.DataFrame, corpus: pd.DataFrame) -> dict:
    """Everything the clinical-trial panels draw."""
    out = {}

    # (a) Calendar years in the analysis window; earlier starts are excluded.
    trials = filter_analysis_window(trials, year_col="start_year", date_col="start_date")
    by_year = (trials.dropna(subset=["start_year"]).astype({"start_year": int})
               .groupby(["start_year", "study_type"]).size().unstack(fill_value=0)
               .reindex(YEARS, fill_value=0).astype(int))
    out["start_year_by_type"] = by_year.rename(index=str)
    out["start_year_bins"] = {
        str(year): int(by_year.loc[year].sum()) for year in YEARS
    }

    # (b) lifecycle stage. The registry's nine `overall_status` values folded into five
    # stages — the same 9->5 map the source notebook's §4 uses.
    stage_map = {
        "Not yet recruiting": "Planned",
        "Recruiting": "Ongoing",
        "Enrolling by invitation": "Ongoing",
        "Active, not recruiting": "Ongoing",
        "Completed": "Completed",
        "Terminated": "Stopped early",
        "Withdrawn": "Stopped early",
        "Suspended": "Stopped early",
        "Unknown status": "Unknown",
    }
    stage = trials["overall_status"].map(stage_map).fillna("Unknown")
    stage_order = ["Planned", "Ongoing", "Completed", "Stopped early", "Unknown"]
    out["stage_by_type"] = (pd.crosstab(stage, trials["study_type"])
                            .reindex(stage_order, fill_value=0))

    # (c) conditions, two vocabularies. `mesh_leaf` is the condition-LEAF MeSH rebuilt
    # from the CTgov API (§3.5 of the source notebook) — the axis to quote, because the
    # shipped `mesh_terms` column merges tree ancestors and intervention terms into the
    # same list and its ranking is topped by scaffolding, not by diseases.
    leaf = Counter()
    for cell in trials["mesh_leaf"]:
        leaf.update({str(t) for t in _lst(cell)
                     if str(t).strip() and str(t) not in MESH_LEAF_STOP})
    out["mesh_leaf"] = _top_counter(leaf, 10)

    # RCDC, disease tags only: the stop list in shared_rcdc removes the research-area,
    # method and population tags that otherwise dominate a raw RCDC ranking (D11).
    rcdc_all, rcdc_disease = Counter(), Counter()
    for cell in trials["category_rcdc"]:
        names = {str(d.get("name")) for d in _lst(cell) if isinstance(d, dict) and d.get("name")}
        rcdc_all.update(names)
        rcdc_disease.update(n for n in names if RCDC.is_disease(n))
    out["rcdc_all"] = _top_counter(rcdc_all, 10)
    # RCDC spells some conditions at length ("Heart Disease - Coronary Heart Disease").
    # Trimmed to the tick width so a label stays on one line: two-line labels in the last
    # row of a fixed-height panel run off its bottom edge.
    rcdc_top = _top_counter(rcdc_disease, 10)
    rcdc_top.index = [_shorten(i, 34) for i in rcdc_top.index]
    out["rcdc_disease"] = rcdc_top

    # (d) ICD-10 chapters, for the body map. A trial is counted once per chapter it
    # touches, so the counts do not partition the 195 — a trial studying both obesity and
    # heart failure is in IV and in IX. Read off the SHIPPED `mesh_terms` column rather
    # than the condition-leaf rebuild, because the keyword rules were written against
    # that vocabulary and it covers 181 of 195 trials against the leaf column's 174.
    chapters, n_mapped = Counter(), 0
    for cell in trials["mesh_terms"]:
        hit = {map_icd_chapter(name) for name in _names(cell)}
        hit.discard(None)
        if hit:
            n_mapped += 1
        chapters.update(hit)
    out["icd_chapters"] = pd.Series(chapters, dtype="int64").sort_values(ascending=False)
    out["icd_mapped_trials"] = n_mapped

    # (e) planned enrollment. Heavily right-skewed (a few observational cohorts in the
    # hundreds of thousands), so the panel that draws this uses a log axis.
    out["enrollment"] = pd.to_numeric(trials["study_participants"], errors="coerce").dropna()

    # (f) where the trials are run, and WHO runs them. A trial spreads a weight of 1
    # evenly across its distinct (country, sector) organisation pairs, so the columns sum
    # to the number of located trials rather than counting a multi-site trial once per
    # site. Same fractional accounting as §7 of the source notebook.
    rows = []
    for cell in trials["research_orgs"]:
        pairs = {(o.get("country_name"), org_sector(o)) for o in _lst(cell)
                 if isinstance(o, dict) and o.get("country_name")}
        if not pairs:
            continue
        weight = 1.0 / len(pairs)
        rows.extend((country, sector, weight) for country, sector in pairs)
    long = pd.DataFrame(rows, columns=["country", "sector", "w"])
    matrix = long.pivot_table(index="country", columns="sector", values="w",
                              aggfunc="sum", fill_value=0)
    matrix = matrix.loc[matrix.sum(axis=1).sort_values(ascending=False).index].head(10)
    out["country_sector"] = matrix[[c for c in TRIAL_SECTOR_ORDER if c in matrix.columns]]

    # The whole-trial country count is kept as well: it is what the "trials per country"
    # line in the write-up quotes, and it is not recoverable from the fractional matrix.
    countries = Counter()
    for cell in trials["research_orgs"]:
        names = {o.get("country_name") for o in _lst(cell)
                 if isinstance(o, dict) and o.get("country_name")}
        countries.update(names)
    out["countries"] = _top_counter(countries, 10)

    # (g) paper -> trial lag, over every (trial, cited paper) link.
    corpus_ids = set(corpus["id"])
    year_of = dict(zip(corpus["id"], corpus["year"]))
    lags = []
    for syear, cell in zip(trials["start_year"], trials["publication_ids"]):
        if pd.isna(syear):
            continue
        for pid in _lst(cell):
            if pid in corpus_ids and pd.notna(year_of.get(pid)):
                lags.append(int(syear) - int(year_of[pid]))
    out["paper_to_trial_lag"] = pd.Series(lags, dtype="int64")

    # (h) how many UK Biobank papers each trial leans on.
    out["papers_per_trial"] = trials["publication_ids"].apply(
        lambda c: len([p for p in _lst(c) if p in corpus_ids])
    )
    return out


# ----------------------------------------------------------------- policy ----
def build_policy_aggregates(policy: pd.DataFrame, corpus: pd.DataFrame) -> dict:
    """Everything the policy panels draw."""
    out = {}
    corpus_ids = set(corpus["id"])

    # (a) documents per year, split UK / rest of world — the split the write-up asks
    # about, since UK Biobank is a UK asset and the question is whether its evidence
    # travels beyond UK policy.
    pol = policy.dropna(subset=["year"]).astype({"year": int})
    pol = pol[(pol["year"] >= YEAR_MIN) & (pol["year"] <= YEAR_MAX)]
    origin = np.where(pol["publisher_country_code"] == "GB", "United Kingdom", "Rest of world")
    out["by_year_origin"] = (pd.crosstab(pol["year"], origin)
                             .reindex(YEARS, fill_value=0)
                             .reindex(columns=["United Kingdom", "Rest of world"], fill_value=0))

    out["countries"] = _top_counter(Counter(policy["publisher_country"]), 12)
    # The same counts keyed on the ISO 3166-1 alpha-2 code the export ships, which is
    # what the choropleth joins the world geometry on. Kept beside the named-country
    # Series rather than replacing it: the names are what prose quotes, the codes are
    # what a map merges, and deriving one from the other at draw time is where a
    # choropleth silently loses a country.
    iso = policy.loc[policy["publisher_country_code"].astype(str).str.len() == 2,
                     "publisher_country_code"]
    out["countries_iso"] = iso.value_counts().astype("int64")
    publishers = _top_counter(Counter(policy["publisher_name"]), 8)
    # "Food and Agriculture Organization of the United Nations" is four wrapped lines in
    # a tick; the acronym-bearing tail is what identifies it, so trim rather than wrap.
    publishers.index = [_shorten(i, 46) for i in publishers.index]
    out["publishers"] = publishers

    # (b) what the policy documents are about: FOR 2020 divisions (L2) of the document.
    divisions = Counter()
    for cell in policy["category_for_2020"]:
        divisions.update(_for_divisions(cell))
    out["divisions"] = _top_counter(divisions, 10)

    # (c) how concentrated the citing is: papers cited by N policy documents.
    counter = Counter()
    for cell in policy["publication_ids"]:
        counter.update(pid for pid in set(_lst(cell)) if pid in corpus_ids)
    out["policy_docs_per_paper"] = pd.Series(list(counter.values()), dtype="int64")
    out["n_papers_cited"] = len(counter)

    titles = pd.read_parquet(P.SHOWCASE_PLUS, columns=["id", "title"]).set_index("id")["title"]
    top = _top_counter(counter, 8)
    out["top_cited_papers"] = pd.Series(
        top.values, index=[_shorten(titles.get(i, i)) for i in top.index], dtype="int64"
    )
    return out


# -------------------------------------------------------------- altmetric ----
def _annotate_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Attach first author and journal to a handful of scatter rows, for labelling.

    Read back from the parquet with a row filter rather than carried through the whole
    pipeline: `authors` is the widest column in the corpus and only these two rows need
    it, so pulling it for 26,109 publications to label 2 of them would be the expensive
    way round.
    """
    rows = rows.copy()
    ids = [i for i in rows["id"].dropna().unique().tolist()]
    if not ids:
        rows["first_author"] = []
        rows["journal"] = []
        return rows
    detail = pd.read_parquet(
        P.SHOWCASE_PLUS, columns=["id", "authors", "journal_title_raw"],
        filters=[("id", "in", ids)],
    )

    def first_author(cell) -> str:
        for entry in _lst(cell):
            if isinstance(entry, dict):
                name = entry.get("last_name") or entry.get("first_name") or ""
                if str(name).strip():
                    return str(name).strip()
        return "Author"

    detail["first_author"] = detail["authors"].apply(first_author)
    detail["journal"] = detail["journal_title_raw"].fillna("")
    return rows.merge(detail[["id", "first_author", "journal"]], on="id", how="left")


def build_altmetric_aggregates(altmetric: pd.DataFrame) -> dict:
    """Everything the altmetric panels draw."""
    out = {}
    alt = altmetric

    # The scatter's population, stated as a denominator rather than left implicit: a
    # paper needs a positive attention score AND at least one news-or-policy mention,
    # because both axes are logarithmic (D18's 5,601-paper figure).
    scatter = alt[(alt["substantive"] > 0) & (alt["Altmetric Attention Score"] > 0)].copy()
    out["scatter"] = scatter[
        ["id", "substantive", "News mentions", "Policy mentions",
         "Altmetric Attention Score", "times_cited", "year", "DOI"]
    ]
    # The two most-mentioned publications, with enough bibliographic detail to name them
    # on the panel. Part III annotated these and the annotation is
    # what turns an anonymous cloud into a claim a reader can check — the top-right point
    # is a specific paper, and saying which one costs two labels.
    out["scatter_top"] = _annotate_rows(scatter.nlargest(2, "substantive"))
    # Two contrasting lower-edge examples: lowest score per mention, and the most
    # mentioned paper with a score/mention ratio <= 3 (at least ten mentions).
    lower = scatter.loc[scatter["substantive"] >= 10].copy()
    lower["score_per_mention"] = lower["Altmetric Attention Score"] / lower["substantive"]
    lower = lower.loc[lower["score_per_mention"] <= 3]
    choices = pd.concat([lower.nsmallest(1, "score_per_mention"),
                         lower.nlargest(1, "substantive")]).drop_duplicates("id")
    out["scatter_lower"] = _annotate_rows(choices)
    central = scatter.loc[scatter["substantive"].between(5, 30)]
    out["scatter_center"] = _annotate_rows(central.nlargest(1, "Altmetric Attention Score"))

    out["score_distribution"] = alt.loc[
        alt["Altmetric Attention Score"] > 0, "Altmetric Attention Score"
    ]

    yearly = (alt.dropna(subset=["year"]).astype({"year": int})
              .groupby("year")[["News mentions", "Policy mentions"]].sum())
    out["mentions_by_year"] = yearly.reindex(YEARS, fill_value=0)

    # Coverage: of the papers published in each year, what share picked up any news, and
    # what share any policy mention. A rate, not a total, so the last years are readable
    # against the first despite the corpus growing 200-fold across the window.
    per_year = alt.dropna(subset=["year"]).astype({"year": int})
    grouped = per_year.groupby("year")
    coverage = pd.DataFrame({
        "papers": grouped.size(),
        "with_news": grouped["News mentions"].apply(lambda s: (s > 0).sum()),
        "with_policy": grouped["Policy mentions"].apply(lambda s: (s > 0).sum()),
    }).reindex(YEARS).fillna(0)
    coverage["pct_news"] = 100 * coverage["with_news"] / coverage["papers"].replace(0, np.nan)
    coverage["pct_policy"] = 100 * coverage["with_policy"] / coverage["papers"].replace(0, np.nan)
    out["coverage_by_year"] = coverage
    return out


# ---------------------------------------------------------- collaboration ----
def build_collaboration_aggregates(collab: pd.DataFrame) -> dict:
    """Everything the collaboration panels draw."""
    out = {}

    # (a) collaborator mentions and papers, per sector.
    summary = pd.DataFrame({
        "mentions": [int(collab[_sector_count(s)].sum()) for s in SECTOR_ORDER],
        "papers": [int(collab[_sector_flag(s)].sum()) for s in SECTOR_ORDER],
    }, index=SECTOR_ORDER)
    out["sector_summary"] = summary

    # (a2) how the sectors overlap, ROW-NORMALISED: cell (i, j) is the share of the
    # publications carrying sector i that also carry sector j. The matrix is deliberately
    # asymmetric, and the asymmetry is the finding — 90% of UK-company papers also carry
    # a university partner while 1.4% of university papers carry a UK company. On raw
    # counts every cell would be dominated by University/HEI, which is the one sector
    # nearly every paper has, and the six small sectors would be unreadable.
    flags = {s: collab[_sector_flag(s)] == 1 for s in SECTOR_ORDER}
    overlap = pd.DataFrame(index=SECTOR_ORDER, columns=SECTOR_ORDER, dtype=float)
    for row in SECTOR_ORDER:
        total = int(flags[row].sum())
        for col in SECTOR_ORDER:
            both = int((flags[row] & flags[col]).sum())
            overlap.loc[row, col] = 100 * both / total if total else np.nan
    out["flag_overlap"] = overlap
    out["flag_totals"] = pd.Series({s: int(flags[s].sum()) for s in SECTOR_ORDER})

    # (b) share of that year's papers carrying each non-academic sector. A share, because
    # the corpus grows substantially across the analysis window and a count panel would
    # only redraw that growth curve six times.
    year = collab.dropna(subset=["year"]).astype({"year": int})
    year = year[(year["year"] >= YEAR_MIN) & (year["year"] <= YEAR_MAX)]
    totals = _on_years(year["year"].value_counts().sort_index())
    shares = pd.DataFrame(index=YEARS)
    for sector in NON_ACADEMIC_SECTORS:
        counts = _on_years(year.loc[year[_sector_flag(sector)] == 1, "year"]
                           .value_counts().sort_index())
        shares[sector] = 100 * counts / totals.replace(0, np.nan)
    out["sector_share_by_year"] = shares
    out["papers_by_year"] = totals

    # (c) company collaboration, UK against everywhere else — the axis D27 flags as the
    # one with a residual quality caveat (4 orgs, ~3.9% of UK-company mentions, are UK
    # non-companies the GRID correction could not reach).
    company = pd.DataFrame(index=YEARS)
    for sector in COMPANY_SECTORS:
        counts = _on_years(year.loc[year[_sector_flag(sector)] == 1, "year"]
                           .value_counts().sort_index())
        company[sector] = counts
    out["company_by_year"] = company
    out["company_share_by_year"] = 100 * company.div(totals.replace(0, np.nan), axis=0)

    # (d) the organisations themselves, by sector.
    top_orgs = {}
    for sector in NON_ACADEMIC_SECTORS:
        counter = Counter()
        col = f"sector_{re.sub(r'[^a-z0-9]+', '_', sector.casefold()).strip('_')}_institutions"
        if col not in collab.columns:
            continue
        for cell in collab[col]:
            counter.update({str(v) for v in _lst(cell) if str(v).strip()})
        top_orgs[sector] = _top_counter(counter, 12)
    out["top_orgs_all"] = top_orgs
    out["top_orgs"] = top_orgs

    # (e) discipline: FOR 2020 division against the share of its papers with a company
    # collaborator. Where in the science does industry actually show up?
    company_flag = collab[[_sector_flag(s) for s in COMPANY_SECTORS]].max(axis=1)
    rows = []
    for cell, flag in zip(collab["category_for_2020"], company_flag):
        for label in _for_divisions(cell):
            rows.append((label, int(flag)))
    disc = pd.DataFrame(rows, columns=["division", "company"])
    by_div = disc.groupby("division")["company"].agg(["size", "sum"])
    by_div = by_div[by_div["size"] >= 100]
    by_div["pct_company"] = 100 * by_div["sum"] / by_div["size"]
    out["division_company_share"] = by_div.sort_values("pct_company", ascending=False)

    # (f) citation impact by collaboration type — one of the two questions a reviewer
    # always asks of a collaboration figure (the other being causality, which this
    # cannot answer and does not claim to).
    cites = pd.to_numeric(collab["times_cited"], errors="coerce")
    groups = {
        "Academic only": (company_flag == 0)
        & (collab[[_sector_flag(s) for s in NON_ACADEMIC_SECTORS]].max(axis=1) == 0),
        "Other non-academic": (company_flag == 0)
        & (collab[[_sector_flag(s) for s in NON_ACADEMIC_SECTORS]].max(axis=1) == 1),
        "With a company": company_flag == 1,
    }
    out["citations_by_group"] = {k: cites[v].dropna() for k, v in groups.items()}
    return out


# =============================================================================
# 3. One call that builds the lot
# =============================================================================
def build_panel_data(verbose: bool = True) -> dict:
    """Read the four sources and return every aggregate the panels draw.

    Roughly 40 s, nearly all of it the collaboration taxonomy and the two parquet reads.
    The result is a plain dict of small frames — small enough that a notebook can hold it,
    display any of it, and write it beside the figures as the numbers behind them.
    """
    def say(msg):
        if verbose:
            print(msg)

    say("corpus …")
    corpus = load_corpus()
    say(f"  {len(corpus):,} publications, {int(corpus.year.min())}-{int(corpus.year.max())}")

    say("patents …")
    patents = load_patents()
    say(f"  {len(patents):,} patents")

    say("clinical trials …")
    trials = load_trials()
    say(f"  {len(trials):,} trials")

    say("policy documents …")
    policy = load_policy()
    say(f"  {len(policy):,} documents")

    say("altmetric …")
    altmetric = load_altmetric(corpus)
    say(f"  {len(altmetric):,} rows joined to the corpus")

    say("collaboration (sector taxonomy) …")
    collab = load_collaboration()
    collab = collab.loc[collab["id"].isin(corpus["id"])].copy()
    say(f"  {len(collab):,} publications classified")

    data = {
        "corpus": corpus,
        "reach": build_reach(corpus, patents, trials, policy, altmetric, collab),
        "reach_by_year": build_reach_by_year(corpus, patents, trials, policy, altmetric, collab),
        "patents": build_patent_aggregates(patents, corpus),
        "trials": build_trial_aggregates(trials, corpus),
        "policy": build_policy_aggregates(policy, corpus),
        "altmetric": build_altmetric_aggregates(altmetric),
        "collaboration": build_collaboration_aggregates(collab),
        "counts": {
            "corpus": len(corpus),
            "patents": len(patents),
            "trials": len(trials),
            "policy": len(policy),
            "altmetric_rows": len(altmetric),
        },
    }
    say("done.")
    return data


# =============================================================================
# 4. Drawing primitives
# =============================================================================
# Every one of these takes an `ax` and returns it. None of them creates a figure, sets a
# figure size, or calls savefig: the assembler owns the page, so a panel cannot quietly
# impose its own geometry on the one it is sharing.

import matplotlib.pyplot as plt                                        # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm         # noqa: E402
from matplotlib.lines import Line2D                                    # noqa: E402
from matplotlib.patches import Patch                                   # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator               # noqa: E402
from contextlib import contextmanager                                  # noqa: E402

from utils.shared_style import (                                       # noqa: E402
    apply_typography, grid_on, load_style, palette, panel_label,
    semantic_colors, warm_cold_colormap,
)


from utils.data_analysis_04_non_academic_figures import (
    finalize_figure, savefig, PATENT_STATUS_COLORS, TRIAL_SECTOR_COLORS,
)


def _style():
    """The active style, so a draw function can read type sizes without an argument."""
    from utils import shared_style
    return shared_style._resolve(None)


@contextmanager
def _font_scale(factor):
    """Draw the enclosed figure with every font size multiplied by `factor`.

    The main panel and the SI panels want different type: the main figure is read at
    single-column or full width in a paper, the SI panels on a screen at full page. Rather
    than a second style section — which would be one more thing to keep in step with this
    one — the scale is a single number in `04_non_academic_panels` and this multiplies the
    `*_fs` keys through it, pushing the result into rcParams so axis labels and ticks move
    with the annotations a draw function sizes by hand.

    Restores the original style on the way out, including after an exception, so a failed
    render cannot leave the kernel drawing everything at 1.32x.
    """
    from utils import shared_style
    original = _style()
    if not factor or factor == 1:
        yield original
        return
    scaled = dict(original)
    for key in ("title_fs", "label_fs", "tick_fs", "annot_fs", "legend_fs",
                "body_fs", "panel_label_fs"):
        if key in scaled:
            scaled[key] = scaled[key] * factor
    shared_style.use_style(scaled)
    shared_style.apply_style(scaled)
    try:
        yield scaled
    finally:
        shared_style.use_style(original)
        shared_style.apply_style(original)


def _stream_colors() -> dict:
    return semantic_colors("stream_colors")


def _sector_colors() -> dict:
    return semantic_colors("sector_colors")


#: Median and mean rules, held constant across every histogram in the family. They were
#: borrowed from `stream_colors` at first, which meant the median line changed colour when
#: a stream did and could land the same hue as the bars underneath it.
REF_MEDIAN = ("--", "#E66859")      # palette red
REF_MEAN = (":", "#3D3D3D")         # neutral dark grey


def _ref_lines(ax, values, *, fmt="{:,.0f}", suffix="", loc="upper right"):
    """Draw the median (dashed red) and mean (dotted grey) rules and label them."""
    st = _style()
    for stat, (dash, color) in (("median", REF_MEDIAN), ("mean", REF_MEAN)):
        value = getattr(values, stat)()
        ax.axvline(value, linestyle=dash, color=color, linewidth=1.6,
                   label=f"{stat.title()} = {fmt.format(value)}{suffix}")
    ax.legend(loc=loc, fontsize=st["legend_fs"])
    return ax


def _fs_scale(figure: str) -> float:
    """Type scale for one figure, from `fs_scale` in the style section (default 1.0)."""
    return float((_style().get("fs_scale") or {}).get(figure, 1.0))


def _heat_cmap():
    """The colormap every heatmap in this family uses: the palette's cold-to-warm ramp.

    One map, not one per panel: a reader who has learned that navy is a low count and red
    a high one in panel D should not have to re-learn it in panel E.

    Empty cells are drawn WHITE, not navy. On a cold-to-warm ramp zero sits at the dark
    end, so an unmasked matrix renders "this combination does not occur" in the same
    saturated navy as "this combination is rare" — and in a block-diagonal matrix, where
    most cells are structurally empty, that turns the whole panel into a dark slab. The
    caller passes NaN for empty and this map paints it white.
    """
    cmap = warm_cold_colormap().copy()
    cmap.set_bad("white")
    return cmap


def _heat_values(frame):
    """Matrix values with zeros as NaN, so `_heat_cmap` paints them as empty."""
    values = frame.to_numpy(dtype=float)
    return values


def _heat_text_color(value, top):
    """Black on the pale middle of the ramp, white on both saturated ends.

    A cold-to-warm map is dark at the BOTTOM as well as at the top, so the usual
    "light text above 60%" rule leaves the smallest cells as dark-on-dark.
    """
    return "white" if value > top * 0.72 or value < top * 0.16 else "black"


def _seq_cmap(color, light="#FFFFFF"):
    """White -> `color`, for a heatmap that stays inside the project palette."""
    return LinearSegmentedColormap.from_list("brand_seq", [light, color], N=256)


def _thousands(ax, axis="y"):
    fmt = FuncFormatter(lambda v, _: f"{int(v):,}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def _tick_fontsize(labels, base) -> float:
    """Type size for category ticks: smaller once wrapping puts labels close together.

    A panel is a fixed height, so N rows of two-line labels collide at the size one row
    reads well at. Rather than leave that to be noticed in the PDF, shrink with the row
    count and with how many labels actually wrapped.
    """
    rows = len(labels)
    wrapped = sum("\n" in str(l) for l in labels)
    size = base
    if rows > 8:
        size -= 1
    if rows > 11:
        size -= 1
    if wrapped:
        size -= 1
    if wrapped > rows / 3:
        size -= 1
    return max(size, base - 4)


def _hbar(ax, series, color, xlabel, *, ylabel=None, annotate=True, pct_of=None,
          wrap=34, fmt="{:,.0f}"):
    """Horizontal bars from a largest-first Series, drawn largest at the top.

    **A bar that is annotated with its own value gets no grid.** The grid exists so a
    reader can estimate a bar's length off the axis; once the number is printed at the
    end of the bar there is nothing left to estimate, and the gridlines only add ink
    across the panel. Panels without annotations keep theirs.
    """
    st = _style()
    series = series.iloc[::-1]                       # barh draws bottom-up
    labels = [_wrap(i, wrap) for i in series.index]
    ax.tick_params(axis="y", labelsize=_tick_fontsize(labels, st["tick_fs"]))
    bars = ax.barh(labels, series.values, color=color,
                   edgecolor=st.get("edgecolor", "black"), linewidth=0.6)
    if annotate and len(series):
        span = max(series.values) or 1
        for bar, value in zip(bars, series.values):
            text = fmt.format(value)
            if pct_of:
                text += f"  ({100 * value / pct_of:.1f}%)"
            ax.text(value + span * 0.015, bar.get_y() + bar.get_height() / 2, text,
                    va="center", ha="left", fontsize=st["annot_fs"])
        ax.set_xlim(0, span * 1.22)
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if not annotate:
        grid_on(ax, axis="x")
    return ax


def _stacked_bars(ax, frame, colors, xlabel, ylabel, *, legend_title=None,
                  legend_loc="upper left", width=0.8, annotate=False,
                  segment_labels=True, min_label_share=0.07):
    """Stacked vertical bars: rows are the x categories, columns the stack segments.

    With `annotate`, each bar is labelled with its total and the grid is dropped — the
    same rule `_hbar` follows, for the same reason. Segments are labelled too unless
    `segment_labels=False`, which is what a seven-category stack needs: at that many
    slices most segments are a few pixels tall and their labels become a column of
    overlapping digits, so only the yearly total is worth printing.

    A segment smaller than `min_label_share` of the tallest bar is left unlabelled for
    the same reason, even when `segment_labels` is on.
    """
    st = _style()
    bottom = np.zeros(len(frame))
    x = np.arange(len(frame))
    totals = frame.sum(axis=1).to_numpy(dtype=float)
    tallest = max(totals.max(), 1)
    for column in frame.columns:
        values = frame[column].to_numpy(dtype=float)
        ax.bar(x, values, bottom=bottom, width=width, label=str(column),
               color=colors[str(column)] if isinstance(colors, dict) else colors,
               edgecolor=st.get("edgecolor", "black"), linewidth=0.5)
        if annotate and segment_labels:
            for xi, value, base in zip(x, values, bottom):
                if value >= tallest * min_label_share:
                    ax.text(xi, base + value / 2, f"{int(value):,}", ha="center",
                            va="center", fontsize=st["annot_fs"] - 1, color="white",
                            fontweight="bold")
        bottom += values
    if annotate:
        for xi, total in zip(x, totals):
            if total:
                ax.text(xi, total + tallest * 0.02, f"{int(total):,}", ha="center",
                        va="bottom", fontsize=st["annot_fs"])
        ax.set_ylim(0, tallest * 1.16)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in frame.index], rotation=45, ha="right")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(title=legend_title, loc=legend_loc, fontsize=st["legend_fs"],
              title_fontsize=st["legend_fs"])
    if not annotate:
        grid_on(ax, axis="y")
    _thousands(ax)
    return ax


def _lines(ax, frame, colors, xlabel, ylabel, *, marker="o", labels=None,
           legend_loc="upper left", legend_ncol=1, grid=True, legend_fs=None):
    """One line per column of `frame`, indexed on the x values.

    `legend_fs` overrides the style's legend size for this panel alone. A five-series
    legend inside a half-width panel is a box, not a caption: at the family's size it is
    wide enough to sit on the data whatever corner it is put in, and shrinking the type
    is what shrinks the box.
    """
    st = _style()
    for column in frame.columns:
        ax.plot(frame.index, frame[column].to_numpy(dtype=float),
                marker=marker, markersize=7, linewidth=1.8,
                markeredgecolor="white", markeredgewidth=0.5,
                color=colors[str(column)] if isinstance(colors, dict) else colors,
                label=(labels or {}).get(str(column), str(column)))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc=legend_loc, ncol=legend_ncol,
              fontsize=legend_fs if legend_fs is not None else st["legend_fs"])
    if grid:
        grid_on(ax)
    return ax


def _year_axis(ax, years=None, step=2):
    """Label the LAST year rather than the first — see shared_style.year_ticks."""
    from utils.shared_style import year_ticks
    years = list(years) if years is not None else YEARS
    ticks = year_ticks(min(years), max(years), step)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    return ax


# =============================================================================
# 5. The panels
# =============================================================================
# ------------------------------------------------------------------- shared --
def draw_reach(ax, D):
    """How many UK Biobank papers carry each kind of non-academic linkage."""
    reach = D["reach"].iloc[::-1]
    colors = _stream_colors()
    st = _style()
    bars = ax.barh(
        [_wrap(v, 26) for v in reach["linkage"]], reach["papers"],
        color=[colors[s] for s in reach["stream"]],
        edgecolor=st.get("edgecolor", "black"), linewidth=0.6,
    )
    # Log scale. The linkages span two orders of magnitude — 192 publications cited by a
    # trial against 18,372 with a non-academic co-author — and on a linear axis the three
    # citation bars are three pixels wide beside the collaboration bar. The point of the
    # panel is that these are all real routes at very different scales, which a linear
    # axis makes into "one route and some rounding error".
    ax.set_xscale("log")
    lo, hi = reach["papers"].min(), reach["papers"].max()
    ax.set_xlim(10 ** np.floor(np.log10(lo)), hi * 4)
    for bar, value, pct in zip(bars, reach["papers"], reach["pct"]):
        ax.text(value * 1.08, bar.get_y() + bar.get_height() / 2,
                f"{int(value):,}  ({pct:.1f}%)", va="center", fontsize=st["annot_fs"])
    ax.set_xlabel(f"UK Biobank publications (log, of {int(reach['n_corpus'].iloc[0]):,})")
    # No grid: every bar is labelled with its own count and share (see _hbar). No legend
    # either — each row is already named, the bar colour only carries the stream identity
    # across to panel B, and a legend box here sat on top of the largest bar.
    return ax


def draw_reach_by_year(ax, D):
    """Cumulative UK Biobank papers with each linkage, dated by the paper's own year."""
    frame = D["reach_by_year"].replace(0, np.nan)
    # No grid at all, and the legend bottom-right. Every series rises to the top-right, so
    # the lower right is the one empty quadrant; and on a log axis spanning four decades
    # even a major-only grid competes with five lines for the same space.
    # The five stream names are long ("Non-academic collaboration"), so the legend is set
    # two points under the family size: at the family size the box is wide enough that the
    # patent and trial lines run behind it on their way to the top right.
    _lines(ax, frame, _stream_colors(), "Publication year",
           "Cumulative publications (log)", labels=STREAM_LABELS,
           legend_loc="lower right", grid=False, legend_fs=_style()["legend_fs"] - 2)
    ax.set_yscale("log")
    _year_axis(ax)
    return ax


# ------------------------------------------------------------------ patents --
def draw_patents_by_year(ax, D):
    """Patents by publication year, split application vs granted."""
    frame = D["patents"]["status_by_year"]
    colors = _two_way("Application", "Grant")
    _stacked_bars(ax, frame[["Application", "Grant"]], colors,
                  "Patent publication year", "Patents", legend_title="Filing status",
                  annotate=True)
    return ax


def draw_patent_countries(ax, D):
    """Patents by assignee country.

    Counted once per patent per country: a patent with two US assignees is one US
    patent, not two. The ledger's "top US 268" row counts assignee OCCURRENCES instead,
    which is a different quantity off the same column — hence 244 here, not 268.
    """
    _hbar(ax, D["patents"]["countries"], _stream_colors()["patents"],
          "Patents with an assignee in the country", ylabel="Assignee country")
    return ax


def draw_patent_topics(ax, D):
    """Research divisions (FOR 2020, top level) the patents are classified into."""
    _hbar(ax, D["patents"]["topics"], _stream_colors()["patents"], "Patents",
          ylabel="Research division (FOR 2020)", wrap=26)
    return ax


def draw_patent_topic_count(ax, D):
    """How many divisions one patent spans."""
    st = _style()
    values = D["patents"]["topics_per_patent"]
    values = values[values > 0]
    bins = np.arange(0.5, values.max() + 1.5, 1)
    ax.hist(values, bins=bins, color=_stream_colors()["patents"],
            edgecolor=st.get("edgecolor", "black"), linewidth=0.6)
    _ref_lines(ax, values, fmt="{:.2f}")
    ax.set_xlabel("Research divisions per patent (at least 1)")
    ax.set_ylabel("Patents")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    grid_on(ax, axis="y")
    return ax


def draw_patent_country_topics(ax, D):
    """Research division x country, as a percentage within each country's own patents.

    Transposed relative to the underlying table: divisions run down the y axis and
    countries across the x. That puts the long labels on the axis that has room for them
    — two-digit country codes read fine rotated, "Information and Computing Sciences"
    does not — and it matches panel B, where the division is also the y category.
    """
    st = _style()
    pivot = D["patents"]["country_topic_pct"].T          # divisions x countries
    values = _heat_values(pivot)
    empty = (values == 0) | ~np.isfinite(values)
    image = ax.imshow(np.ma.masked_where(empty, values), aspect="auto", cmap=_heat_cmap(), vmin=0)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, fontsize=st["tick_fs"])
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels([_wrap(c, 22) for c in pivot.index],
                       fontsize=st["tick_fs"] - 2)
    ax.set_xlabel("Assignee country")
    ax.set_ylabel("Research division (FOR 2020)")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iat[i, j]
            if np.isfinite(value) and value > 0:
                ax.text(j, i, f"{value:.0f}", ha="center", va="center",
                        fontsize=st["annot_fs"] - 1,
                        color=_heat_text_color(value, pivot.to_numpy().max()))
    ax.figure.colorbar(image, ax=ax, fraction=0.03, pad=0.02,
                       label="% of the country's patents")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1)
    return ax


def draw_patent_legal_status(ax, D):
    """The seven-way legal outcome, by publication year."""
    frame = D["patents"]["legal_status_by_year"]
    order = [c for c in ["Active", "Application Pending", "Application Granted",
                         "Granted Patent Expired", "Application Ceased",
                         "Application Withdrawn", "Application Abandoned"]
             if c in frame.columns]
    colors = PATENT_STATUS_COLORS
    _stacked_bars(ax, frame[order], colors, "Patent publication year", "Patents",
                  legend_title="Legal status", annotate=True, segment_labels=False)
    st = _style()
    # Seven categories in a half-width panel: two columns of four, which is five rows of
    # legend once the title is counted. Three columns would run past the axes' right edge
    # at this width, and the single row the full-page version used needs a width this
    # panel does not have. The headroom is made above the bars rather than taken from
    # them, and it has to clear the per-year totals as well as the bars themselves — the
    # tallest bar is 124 patents, its total sits just above that, and the axis opens to
    # ~1.9x so the legend starts well clear of both.
    ax.set_ylim(0, frame.sum(axis=1).max() * 1.92)
    ax.legend(loc="upper left", ncol=2, columnspacing=1.0, handlelength=1.2,
              handletextpad=0.5, fontsize=st["legend_fs"],
                   title="Legal status", title_fontsize=st["legend_fs"])
    ax.tick_params(axis="x", labelrotation=0)
    ax.set_xticklabels([str(i) for i in frame.index], rotation=0, ha="center")
    return ax


def draw_patent_lag(ax, D):
    """Years from a UK Biobank paper to the priority date of a patent that cites it."""
    st = _style()
    lags = D["patents"]["paper_to_patent_lag"]
    bins = np.arange(lags.min() - 0.5, lags.max() + 1.5, 1)
    ax.hist(lags, bins=bins, color=_stream_colors()["patents"],
            edgecolor=st.get("edgecolor", "black"), linewidth=0.6)
    ax.axvline(lags.median(), linestyle=REF_MEDIAN[0], color=REF_MEDIAN[1],
               linewidth=1.6, label=f"Median = {lags.median():.0f} yr")
    ax.set_xlabel("Years from publication to patent priority date")
    ax.set_ylabel("Paper–patent links")
    ax.legend(loc="upper left", fontsize=st["legend_fs"])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    grid_on(ax, axis="y")
    return ax


# ---------------------------------------------------------- clinical trials --
#: Every two-category bar in the family uses this pair. A light/dark pair of the same hue
#: needs a legend to be read at all; blue against red separates at a glance and survives
#: the figure being printed small. It is the palette's own blue and red, so the panels
#: still belong to the same family as everything around them.
TWO_WAY_COLORS = ("steel_blue", "red")


def _two_way(first: str, second: str) -> dict:
    """Map two category names onto the shared blue/red pair, in that order."""
    return dict(zip((first, second), palette(*TWO_WAY_COLORS)))


def _trial_type_colors() -> dict:
    """Use the clinical notebook's blue/red identities without activating its style."""
    return load_style("04_non_academic_01_clinical_trials", activate=False)["trial_type_colors"]


def draw_trials_by_year(ax, D):
    """Trials beginning in 2013–2025, grouped by study type."""
    frame = D["trials"]["start_year_by_type"]
    colors = _trial_type_colors()
    order = [c for c in ["Interventional", "Observational"] if c in frame.columns]
    _stacked_bars(ax, frame[order], colors, "Trial start year", "Trials",
                  legend_title="Study type", annotate=True)
    return ax


def _trial_sector_colors() -> dict:
    """The six organisation sectors of §7 of the source notebook, in the project palette."""
    return TRIAL_SECTOR_COLORS


def draw_trial_country_sector(ax, D):
    """Where the trials are run, split by the sector of the organisation running them.

    Fractional weights, not counts: a trial spreads a weight of 1 evenly across its
    distinct (country, sector) organisation pairs, so the bars sum to the number of
    located trials rather than counting a multi-site trial once per site. Same accounting
    as §7 of the source notebook, which is why the totals print as whole numbers while
    the segments behind them are not integers.
    """
    st = _style()
    mat = D["trials"]["country_sector"].iloc[::-1]          # largest on top for barh
    colors = _trial_sector_colors()
    y = np.arange(len(mat))
    left = np.zeros(len(mat))
    for sector in mat.columns:
        values = mat[sector].to_numpy(dtype=float)
        ax.barh(y, values, left=left, label=sector, color=colors.get(sector, "#B8B8B8"),
                edgecolor=st.get("edgecolor", "black"), linewidth=0.4)
        left = left + values
    totals = mat.sum(axis=1).to_numpy()
    for yi, total in zip(y, totals):
        ax.text(total + totals.max() * 0.015, yi, f"{total:.0f}", va="center",
                fontsize=st["annot_fs"])
    ax.set_yticks(y)
    ax.set_yticklabels([_wrap(i, 20) for i in mat.index],
                       fontsize=_tick_fontsize(list(mat.index), st["tick_fs"]))
    ax.set_xlim(0, totals.max() * 1.16)
    ax.set_xlabel("Fractional trial weight")
    ax.set_ylabel("Country")
    ax.legend(title="Sector", loc="lower right", ncol=2, fontsize=st["legend_fs"],
              title_fontsize=st["legend_fs"])
    return ax                       # annotated bars carry no grid — see _hbar


def draw_trial_stage(ax, D):
    """Lifecycle stage, by study type — the nine registry statuses folded into five."""
    st = _style()
    frame = D["trials"]["stage_by_type"].iloc[::-1]
    colors = _trial_type_colors()
    y = np.arange(len(frame))
    height = 0.8 / max(len(frame.columns), 1)
    for i, column in enumerate(frame.columns):
        bars = ax.barh(y + i * height - 0.4 + height / 2, frame[column],
                       height=height, label=str(column),
                       color=colors.get(str(column), "#BDBDBD"),
                       edgecolor=st.get("edgecolor", "black"), linewidth=0.5)
        for bar, value in zip(bars, frame[column]):
            if value:
                ax.text(value + 1, bar.get_y() + bar.get_height() / 2, f"{int(value)}",
                        va="center", fontsize=st["annot_fs"] - 1)
    ax.set_yticks(y)
    ax.set_yticklabels(frame.index)
    ax.set_xlabel("Trials")
    ax.set_xlim(0, frame.to_numpy().max() * 1.45)
    ax.set_ylabel("Lifecycle stage")
    ax.legend(title="Study type", loc="lower right", fontsize=st["legend_fs"])
    # Annotated bars, so no grid — see _hbar for why.
    return ax


def _draw_trial_body_outline(ax):
    """A symmetric vector contour keeps the schematic crisp at publication scale."""
    from matplotlib.path import Path as MplPath
    from matplotlib.patches import Ellipse, PathPatch

    start = (4.65, 12.40)
    curves = [
        ((4.65, 12.10), (4.63, 11.94), (4.45, 11.85)),
        ((4.18, 11.73), (3.86, 11.73), (3.70, 11.40)),
        ((3.50, 11.01), (3.50, 10.61), (3.42, 10.20)),
        ((3.35, 9.64), (3.14, 8.68), (3.02, 8.08)),
        ((2.96, 7.78), (3.04, 7.52), (3.23, 7.56)),
        ((3.43, 7.60), (3.48, 8.13), (3.54, 8.46)),
        ((3.69, 9.12), (3.86, 9.72), (3.91, 10.26)),
        ((3.94, 10.61), (4.03, 10.60), (4.04, 10.26)),
        ((4.04, 9.70), (4.20, 9.27), (4.12, 8.70)),
        ((3.95, 8.10), (3.95, 7.45), (4.00, 6.87)),
        ((4.02, 6.27), (4.16, 5.73), (4.17, 5.18)),
        ((4.03, 4.53), (4.20, 3.51), (4.24, 2.78)),
        ((4.26, 2.44), (3.96, 2.32), (3.96, 2.12)),
        ((4.12, 1.98), (4.63, 2.03), (4.73, 2.10)),
        ((4.80, 2.32), (4.64, 2.56), (4.67, 2.89)),
        ((4.82, 3.76), (4.75, 4.57), (4.72, 5.16)),
        ((4.77, 5.90), (4.74, 6.72), (4.90, 7.18)),
        ((4.94, 7.30), (4.98, 7.33), (5.00, 7.33)),
    ]
    vertices, codes = [start], [MplPath.MOVETO]
    segments = []
    previous = start
    for first, second, end in curves:
        segments.append((previous, first, second, end))
        vertices.extend((first, second, end))
        codes.extend([MplPath.CURVE4] * 3)
        previous = end
    for beginning, first, second, _ in reversed(segments):
        vertices.extend((10 - x, y) for x, y in (second, first, beginning))
        codes.extend([MplPath.CURVE4] * 3)
    vertices.append(start)
    codes.append(MplPath.CLOSEPOLY)
    outline = dict(facecolor="white", edgecolor=palette("navy"), linewidth=1.05, zorder=1)
    ax.add_patch(PathPatch(MplPath(vertices, codes), gid="trial-body-outline", **outline))
    ax.add_patch(Ellipse((5, 13.08), 1.20, 1.52, gid="trial-body-head", **outline))


def draw_trial_bodymap(ax, D):
    """Schematic disease coverage, with aligned labels and a true area/count scale."""
    counts = D["trials"]["icd_chapters"]
    ax._ukb_annotation_fs = 12
    ax._ukb_legend_fs = 11
    _draw_trial_body_outline(ax)
    color = palette("steel_blue")
    area_per_trial = 8.
    for chapter, value in counts.items():
        if chapter not in ICD_BODY_POS or not np.isfinite(value) or value <= 0:
            continue
        x, y, short, side = ICD_BODY_POS[chapter]
        size = area_per_trial * value
        ax.scatter([x], [y], s=size, color=color, edgecolors="white",
                   linewidths=1, zorder=4, gid=f"trial-chapter:{chapter}")
        label = f"{short}\n{int(value)} trials"
        if side == "systemic":
            ax.annotate(label, (x, y), xytext=(18, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=12, zorder=5)
        else:
            left = side == "left"
            label = label.replace("/", "/\n")
            ax.annotate(
                label, (x, y), xytext=(2.75 if left else 7.25, y),
                ha="right" if left else "left", va="center", fontsize=12,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5),
                arrowprops=dict(arrowstyle="-", color=palette("navy"), linewidth=.7,
                                shrinkA=4, shrinkB=np.sqrt(size) / 2 + 2,
                                connectionstyle="arc3,rad=0"),
                zorder=5,
            )
    ax.text(7.25, 5.90, "Systemic", fontsize=12, color=palette("navy"),
            fontweight="bold", va="center")
    handles = [ax.scatter([], [], s=area_per_trial * value, color=color,
                          edgecolors="white", linewidths=1, label=str(value))
               for value in (10, 30, 50)]
    ax.legend(handles=handles, title="Trials", loc="lower center", ncol=3,
              fontsize=11, title_fontsize=11, borderpad=.8, handleheight=2,
              handletextpad=.5, columnspacing=1.2, labelspacing=.6)
    ax.set_xlim(-.3, 10.3)
    ax.set_ylim(.05, 14.25)
    ax.set_aspect("equal")
    ax.set_anchor("N")
    ax.axis("off")
    n_trials = len(D["trials"]["papers_per_trial"])
    note = (f"{D['trials']['icd_mapped_trials']} of {n_trials} trials mapped to at least one "
            "ICD-10 chapter. Marker area is directly proportional to trial count; body "
            "positions are schematic, not precise anatomical localisations.")
    notes = getattr(ax.figure, "_ukb_caption_notes", [])
    if note not in notes:
        ax.figure._ukb_caption_notes = [*notes, note]
    return ax


def draw_trial_conditions(ax, D):
    """The conditions the trials study — condition-LEAF MeSH, rebuilt from the registry.

    Not the shipped `mesh_terms` column: that merges the registry's match with its tree
    ancestors and with intervention-derived terms, so its ranking is topped by
    scaffolding ("Pathologic Processes") rather than by diseases.
    """
    _hbar(ax, D["trials"]["mesh_leaf"], _stream_colors()["clinical_trials"],
          "Trials", ylabel="Condition (MeSH)", wrap=26)
    return ax


def draw_trial_rcdc(ax, D):
    """RCDC disease tags, with the 56 cross-cutting tags stop-listed (D11)."""
    _hbar(ax, D["trials"]["rcdc_disease"], palette("steel_blue"),
          "Trials", ylabel="RCDC disease category", wrap=26)
    return ax


def draw_trial_enrollment(ax, D):
    """Planned enrollment per trial, on a log axis because it spans five orders."""
    st = _style()
    values = D["trials"]["enrollment"]
    values = values[values > 0]
    bins = np.logspace(np.log10(values.min()), np.log10(values.max()), 24)
    ax.hist(values, bins=bins, color=palette("cream"),
            edgecolor=st.get("edgecolor", "black"), linewidth=0.6)
    ax.set_xscale("log")
    _ref_lines(ax, values)
    ax.set_xlabel("Planned enrollment (log)")
    ax.set_ylabel("Trials")
    grid_on(ax, which="major")          # y only: `grid_on` leaves the log x axis bare
    return ax


def draw_trial_countries(ax, D):
    """Where the UK Biobank-citing trials are run (each country once per trial)."""
    _hbar(ax, D["trials"]["countries"], _stream_colors()["clinical_trials"],
          "Trials with an organisation in the country", ylabel="Country", wrap=22)
    return ax


def draw_trial_lag(ax, D):
    """Years from a UK Biobank paper to the start of a trial that cites it."""
    st = _style()
    lags = D["trials"]["paper_to_trial_lag"]
    bins = np.arange(lags.min() - 0.5, lags.max() + 1.5, 1)
    ax.hist(lags, bins=bins, color=_stream_colors()["clinical_trials"],
            edgecolor=st.get("edgecolor", "black"), linewidth=0.6)
    ax.axvline(lags.median(), linestyle=REF_MEDIAN[0], color=REF_MEDIAN[1],
               linewidth=1.6, label=f"Median = {lags.median():.0f} yr")
    ax.set_xlabel("Years from publication to trial start")
    ax.set_ylabel("Paper–trial links")
    ax.legend(loc="upper left", fontsize=st["legend_fs"])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    grid_on(ax, axis="y")
    return ax


# ------------------------------------------------------------------- policy --
def draw_policy_by_year(ax, D):
    """Policy documents citing UK Biobank research, UK-published against everywhere else."""
    frame = D["policy"]["by_year_origin"]
    colors = {"United Kingdom": palette(POLICY_PRIMARY),
              "Rest of world": palette(POLICY_SECONDARY)}
    _stacked_bars(ax, frame, colors, "Policy document year", "Policy documents",
                  legend_title="Publisher")
    return ax


def draw_policy_countries(ax, D):
    """The countries whose institutions publish those documents, as a ranked bar.

    Superseded on the SI page by `draw_policy_country_map`, which answers the same
    question geographically. Kept because `D["policy"]["countries"]` is the Series prose
    quotes the ranking off, and a bar of it is the cheapest way to check the map.
    """
    _hbar(ax, D["policy"]["countries"], palette(POLICY_PRIMARY),
          "Policy documents", ylabel="Publisher country", wrap=22)
    return ax


def draw_policy_country_map(ax, D):
    """Publisher country as a choropleth, on a log colour scale.

    **Why a map and not the ranked bar it replaces.** The ranking's finding is not which
    country is first — it is that a UK resource is cited by policy bodies on five
    continents, and that the second and third publishers (the United States and
    Switzerland, the latter almost entirely the WHO) are not British. A bar chart makes
    the reader assemble that geography from twelve country names; a map states it.

    **Log colour scale.** The counts run 1 to 111 with most countries in single figures,
    so a linear ramp paints everything outside the top three the same near-white and the
    map's whole message becomes "three countries exist".

    Countries with no citing document are grey, not the ramp's lightest blue: absent and
    minimal are different claims and a sequential ramp cannot make that distinction on
    its own. The named counts are printed for the leading publishers, because a
    choropleth alone cannot be read to a number.
    """
    import geopandas as gpd
    from matplotlib.colors import BoundaryNorm

    st = _style()
    counts = D["policy"]["countries_iso"]
    world = gpd.read_file(P.WORLD_SHP)
    # ISO_A2 is "-99" for a handful of countries in Natural Earth (France and Norway
    # among them, both of which publish here); ISO_A2_EH carries the code for those, so
    # the join takes _EH first and falls back. Without this France is drawn as missing.
    iso = world["ISO_A2_EH"].where(world["ISO_A2_EH"].astype(str).ne("-99"),
                                  world["ISO_A2"])
    world = world.assign(iso2=iso.astype(str))
    world["documents"] = world["iso2"].map(counts)
    world = world[world["NAME"] != "Antarctica"]
    matched = int(world["documents"].notna().sum())

    cmap = LinearSegmentedColormap.from_list(
        "policy_seq",
        [c if c.startswith("#") else palette(c) for c in POLICY_MAP_RAMP], N=256,
    )
    world.plot(column="documents", ax=ax, cmap=cmap,
               norm=LogNorm(vmin=1, vmax=float(counts.max())),
               edgecolor=st.get("edgecolor", "black"), linewidth=0.25,
               missing_kwds={"color": "white",
                             "edgecolor": st.get("edgecolor", "black"),
                             "linewidth": 0.25})
    ax.set_xlim(-179, 179)
    ax.set_ylim(-58, 84)
    ax.set_axis_off()
    ax.set_anchor("N")          # equal aspect leaves slack; spend it below, not above

    mappable = plt.cm.ScalarMappable(cmap=cmap,
                                     norm=LogNorm(vmin=1, vmax=float(counts.max())))
    cbar = ax.figure.colorbar(mappable, ax=ax, fraction=0.030, pad=0.02,
                              shrink=0.72)
    cbar.set_label("Policy documents (log)", fontsize=st["label_fs"] - 1)
    ticks = [t for t in (1, 3, 10, 30, 100) if t <= counts.max()]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([str(t) for t in ticks])
    cbar.ax.tick_params(labelsize=st["tick_fs"] - 1)

    # Below the map, not on it: the Pacific is the only clear water wide enough to hold
    # this box, and a caption sitting in the Pacific reads as if it belonged to the
    # countries around it.
    top = D["policy"]["countries"].head(5)
    ax.figure._ukb_caption_notes = [
        f"There are {len(counts)} publisher countries ({matched} mapped). Leading countries: "
        + "; ".join(f"{name}, {int(value)} documents" for name, value in top.items()) + "."]
    return ax


def draw_policy_publishers(ax, D):
    """The institutions themselves."""
    _hbar(ax, D["policy"]["publishers"], palette(POLICY_PRIMARY),
          "Policy documents", ylabel="Publishing institution", wrap=30)
    return ax


def draw_policy_divisions(ax, D):
    """What the policy documents are about, as FOR 2020 research divisions."""
    _hbar(ax, D["policy"]["divisions"], palette(POLICY_SECONDARY),
          "Policy documents", ylabel="Research division (FOR 2020)", wrap=30)
    return ax


def draw_policy_concentration(ax, D):
    """How many policy documents cite one paper — is the citing spread or concentrated?"""
    st = _style()
    values = D["policy"]["policy_docs_per_paper"]
    bins = np.arange(0.5, values.max() + 1.5, 1)
    ax.hist(values, bins=bins, color=palette(POLICY_PRIMARY),
            edgecolor=st.get("edgecolor", "black"), linewidth=0.6)
    ax.set_yscale("log")
    ax.set_xlabel("Policy documents citing the publication")
    ax.set_ylabel("UK Biobank publications (log)")
    ax.text(0.97, 0.94, f"{D['policy']['n_papers_cited']:,} publications cited\n"
                        f"median {values.median():.0f}, max {values.max():.0f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=st["annot_fs"],
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.8))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    grid_on(ax, axis="y")
    return ax


def draw_policy_top_papers(ax, D):
    """The UK Biobank publications policy documents reach for most often."""
    _hbar(ax, D["policy"]["top_cited_papers"], palette(POLICY_PRIMARY),
          "Policy documents citing the publication", ylabel="UK Biobank publication",
          wrap=44)
    ax.tick_params(axis="y", labelsize=_style()["tick_fs"] - 3)
    return ax


# ---------------------------------------------------------------- altmetric --
def _attention_cloud(ax, frame, xcolumn, xlabel, *, top_padding=.08, color=None,
                     outlined=False):
    """Use small uniform marks; the axes, rather than bubble area, encode values."""
    from matplotlib.colors import to_rgba

    ax._ukb_label_fs = 14
    ax._ukb_tick_fs = 12
    ax._ukb_annotation_fs = 12
    # Apply transparency to fills only, keeping the fine black outlines legible.
    ax.scatter(frame[xcolumn], frame["Altmetric Attention Score"],
               s=24 if outlined else 14, marker="o",
               facecolors=to_rgba(color or palette(ATTENTION_PRIMARY), .65 if outlined else .5),
               edgecolors="black" if outlined else "none",
               linewidths=.3 if outlined else 0,
               rasterized=True, zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Altmetric Attention Score")
    for column, setter, high_padding in (
        (xcolumn, ax.set_xlim, .13),
        ("Altmetric Attention Score", ax.set_ylim, top_padding),
    ):
        values = frame[column].to_numpy(dtype=float)
        values = values[np.isfinite(values) & (values > 0)]
        if values.size:
            lower, upper = np.log10([values.min(), values.max()])
            span = max(upper - lower, 1.)
            setter(10 ** (lower - .07 * span), 10 ** (upper + high_padding * span))
        else:
            setter(.8, 10)
    ax.grid(False, which="both")
    ax.tick_params(axis="both", which="major", labelsize=12, length=4, width=1)
    for side, spine in ax.spines.items():
        spine.set_visible(not outlined or side in ("left", "bottom"))
        spine.set_color("black")
        spine.set_linewidth(1)
    return ax


def _attention_caption_note(ax, text):
    notes = getattr(ax.figure, "_ukb_caption_notes", [])
    if text not in notes:
        ax.figure._ukb_caption_notes = [*notes, text]


def _attention_paper_label(row):
    author = row.get("first_author")
    author = str(author).strip() if pd.notna(author) and str(author).strip() else "Author"
    year = row.get("year")
    return f"{author} et al." + (f" ({int(year)})" if pd.notna(year) else "")


def _attention_callouts(ax, points, labels, obstacles, *, below=False, _first_candidate=0):
    """Route short curved leaders in display space, clear of boxes and marker disks."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    scale = fig.dpi / 72
    dots = ax.transData.transform(np.asarray(obstacles))
    anchors = ax.transData.transform(np.asarray(points))
    radii = np.full(len(dots), (np.sqrt(24) / 2 + .15 + 1.5) * scale)
    for anchor in anchors:
        radii[np.linalg.norm(dots - anchor, axis=1) < .1] = (np.sqrt(70) / 2 + 2) * scale
    bounds = ax.get_window_extent(renderer).padded(-5 * scale)
    boxes = [t.get_bbox_patch().get_window_extent(renderer).padded(3 * scale)
             for t in ax.texts if t.get_bbox_patch() is not None]
    paths = []
    artists = []
    order = list(range(len(points)))
    skip = _first_candidate
    for index in order:
        point, label = points[index], labels[index]
        lower_label = below[index] if isinstance(below, (list, tuple)) else below
        angles = (tuple(range(-15, -180, -15)) if lower_label else
                  (135, 150, 120, 180, 90, 60, 45, 30, 15, 0,
                   -15, -30, -45, -60, -90, -120, -135, -150, -165, 165))
        chosen = None
        for radius in (56, 72, 90, 115, 145, 180):
            for angle in angles:
                dx, dy = radius * np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
                for curve in (.20, -.20):
                    text = ax.annotate(
                        label, point, xytext=(dx, dy), textcoords="offset points",
                        ha=("center" if abs(dy) > abs(dx) * .5 else
                            "left" if dx > 1 else "right" if dx < -1 else "center"),
                        va="center", fontsize=12, zorder=5,
                        bbox=dict(boxstyle="round,pad=.25", facecolor="white",
                                  edgecolor=palette("navy"), linewidth=.7),
                        arrowprops=dict(arrowstyle="->", mutation_scale=9,
                                        color=palette("navy"), linewidth=.9,
                                        connectionstyle=f"arc3,rad={curve}", shrinkA=6, shrinkB=14),
                    )
                    text.update_positions(renderer)
                    text.update_bbox_position_size(renderer)
                    # FancyArrowPatch resolves point-based endpoint gaps when drawn.
                    text.arrow_patch.draw(renderer)
                    box = text.get_bbox_patch().get_window_extent(renderer).padded(2 * scale)
                    path = np.concatenate([segment(np.linspace(0, 1, 65))
                                           for segment, _ in text.arrow_patch.get_path().iter_bezier()])
                    nearest = np.maximum(np.maximum(box.p0 - dots, dots - box.p1), 0)
                    clear = (bounds.contains(*box.p0) and bounds.contains(*box.p1)
                             and not any(box.overlaps(other) for other in boxes)
                             and np.all(np.linalg.norm(nearest, axis=1) > radii))
                    if clear:
                        distances = np.linalg.norm(path[:, None, :] - dots[None, :, :], axis=2)
                        clear = np.all(distances > radii[None, :])
                    if clear:
                        clear = not any(np.any((path[:, 0] >= other.x0) & (path[:, 0] <= other.x1)
                                                & (path[:, 1] >= other.y0) & (path[:, 1] <= other.y1))
                                        for other in [box, *boxes])
                    if clear:
                        clear = not any(np.any((p[:, 0] >= box.x0) & (p[:, 0] <= box.x1)
                                                & (p[:, 1] >= box.y0) & (p[:, 1] <= box.y1)) for p in paths)
                    if clear:
                        clear = not any(np.min(np.linalg.norm(path[:, None, :] - p[None, :, :], axis=2))
                                        < 2 * scale for p in paths)
                    if clear:
                        if index == order[0] and skip:
                            skip -= 1
                            text.remove()
                            continue
                        chosen = text
                        boxes.append(box)
                        paths.append(path)
                        artists.append(text)
                        break
                    text.remove()
                if chosen is not None:
                    break
            if chosen is not None:
                break
        if chosen is None:
            if artists and _first_candidate < 40:
                for artist in artists:
                    artist.remove()
                return _attention_callouts(ax, points, labels, obstacles, below=below,
                                           _first_candidate=_first_candidate + 1)
            raise ValueError(f"No clear attention callout route for {label}; enlarge the panel.")
    return artists


def draw_altmetric_scatter(ax, D):
    """Attention against mentions, with clear upper- and lower-edge paper callouts."""
    scatter = D["altmetric"]["scatter"]
    _attention_cloud(ax, scatter, "substantive", "News and policy mentions",
                     top_padding=.30, outlined=True)
    top = D["altmetric"].get("scatter_top")
    if top is not None and len(top):
        top = top.head(2)
        central = D["altmetric"].get("scatter_center")
        if central is not None and len(central):
            top = pd.concat([top, central.head(1)], ignore_index=True)
            _attention_caption_note(
                ax, "Centre-left: highest attention among publications with five to thirty mentions.")
        n_upper = len(top)
        lower = D["altmetric"].get("scatter_lower")
        if lower is not None and len(lower):
            top = pd.concat([top, lower.head(2)], ignore_index=True)
        points = top[["substantive", "Altmetric Attention Score"]].to_numpy(dtype=float)
        ax.scatter(points[:, 0], points[:, 1], s=70, marker="o",
                   color=palette(ATTENTION_SECONDARY),
                   edgecolors="black", linewidths=.65, zorder=4)
        labels = [_attention_paper_label(row) for _, row in top.iterrows()]
        _attention_callouts(
            ax, points, labels,
            scatter[["substantive", "Altmetric Attention Score"]].to_numpy(),
            below=[i >= n_upper for i in range(len(top))],
        )
        if len(top) > n_upper:
            _attention_caption_note(
                ax, "Lower callouts: lowest-ratio and most-mentioned publications among those "
                "with at least ten mentions and attention-score/mention ratio <= 3. "
                "Selection is descriptive, not a research-quality assessment.")
    if len(scatter):
        scores = scatter["Altmetric Attention Score"]
        _attention_caption_note(
            ax, f"Panel E includes {len(scatter):,} publications "
            f"(mean attention score {scores.mean():,.1f}; median {scores.median():,.0f}; "
            f"maximum {scores.max():,.0f})."
        )
    return ax


def draw_altmetric_distribution(ax, D):
    """The attention score itself, over every paper Altmetric scored above zero."""
    st = _style()
    values = D["altmetric"]["score_distribution"]
    bins = np.logspace(0, np.log10(values.max()), 40)
    ax.hist(values, bins=bins, color=palette(ATTENTION_NEWS),
            edgecolor=st.get("edgecolor", "black"), linewidth=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvline(values.median(), linestyle=REF_MEDIAN[0], color=REF_MEDIAN[1],
               linewidth=1.6, label=f"Median: {values.median():,.0f}")
    ax.set_xlabel("Altmetric Attention Score (log)")
    ax.set_ylabel("UK Biobank publications (log)")
    ax.legend(loc="upper right", fontsize=st["legend_fs"])
    grid_on(ax)
    return ax


def draw_altmetric_mentions_by_year(ax, D):
    """Total news and total policy mentions, by the year of the paper that earned them.

    Two axes, because news outnumbers policy roughly a hundred to one and a shared axis
    would draw the policy series flat along the x axis.
    """
    st = _style()
    frame = D["altmetric"]["mentions_by_year"]
    line_news, = ax.plot(frame.index, frame["News mentions"], marker="o", markersize=5,
                         linewidth=1.8, color=palette(ATTENTION_NEWS),
                         markeredgecolor="white", label="News (left)")
    ax.set_xlabel("Publication year")
    ax.set_ylabel("News mentions")
    _thousands(ax)
    twin = ax.twinx()
    twin.spines["right"].set_visible(True)
    line_policy, = twin.plot(frame.index, frame["Policy mentions"], marker="s",
                             markersize=5, linewidth=1.8,
                             color=palette(ATTENTION_POLICY),
                             markeredgecolor="white", label="Policy (right)")
    twin.set_ylabel("Policy mentions")
    twin.grid(False)
    ax.legend(handles=[line_news, line_policy], loc="upper left",
              fontsize=st["legend_fs"])
    _year_axis(ax)
    grid_on(ax)
    return ax


def draw_altmetric_coverage(ax, D):
    """The share of each year's papers picking up any news / any policy mention.

    A rate rather than a total: the corpus grows substantially across the analysis window,
    so a count panel would only redraw that growth curve twice.
    """
    frame = D["altmetric"]["coverage_by_year"][["pct_news", "pct_policy"]]
    colors = {"pct_news": palette(ATTENTION_NEWS),
              "pct_policy": palette(ATTENTION_POLICY)}
    _lines(ax, frame, colors, "Publication year", "% of that year's publications",
           labels={"pct_news": "1+ news mentions", "pct_policy": "1+ policy mentions"},
           legend_loc="upper right")
    ax.lines[1].set_marker("s")
    _year_axis(ax)
    return ax


def draw_altmetric_vs_citations(ax, D):
    """Attention against citations, with cohort size and association in the caption."""
    scatter = D["altmetric"]["scatter"].dropna(subset=["times_cited"])
    scatter = scatter[scatter["times_cited"] > 0]
    _attention_cloud(ax, scatter, "times_cited", "Citations", color=palette(ATTENTION_NEWS))
    rho = scatter[["times_cited", "Altmetric Attention Score"]].corr(method="spearman").iat[0, 1]
    association = f"Spearman rho = {rho:.2f}" if np.isfinite(rho) else "Spearman rho is undefined"
    _attention_caption_note(
        ax, f"The attention-versus-citations panel includes {len(scatter):,} publications "
        f"with positive citations, attention scores and news or policy mentions; {association}."
    )
    return ax


# ------------------------------------------------------------ collaboration --
def draw_collab_sector_share(ax, D, *, colors=None):
    """Share of each year's publications carrying a collaborator in each sector."""
    frame = D["collaboration"]["sector_share_by_year"]
    _lines(ax, frame, _sector_colors() if colors is None else colors, "Publication year",
           "% of that year's publications", legend_loc="upper left", legend_ncol=2)
    _year_axis(ax)
    # Headroom for the six-entry legend rather than a legend placed over the data: the
    # top series runs to ~62%, so the axis is opened to 100 and the box sits above it.
    ax.set_ylim(0, 100)
    return ax


def draw_collab_sector_summary(ax, D, *, colors=None):
    """Collaborator organisation mentions, by sector."""
    summary = D["collaboration"]["sector_summary"]["mentions"].sort_values(ascending=False)
    colors = _sector_colors() if colors is None else colors
    st = _style()
    series = summary.iloc[::-1]
    bars = ax.barh([_wrap(i, 22) for i in series.index], series.values,
                   color=[colors[i] for i in series.index],
                   edgecolor=st.get("edgecolor", "black"), linewidth=0.6)
    span = series.max()
    for bar, value in zip(bars, series.values):
        ax.text(value + span * 0.015, bar.get_y() + bar.get_height() / 2, f"{value:,}",
                va="center", fontsize=st["annot_fs"])
    ax.set_xlim(0, span * 1.20)
    ax.set_xlabel("Collaborator organisation mentions")
    ax.set_ylabel("Collaborator sector")
    _thousands(ax, "x")
    return ax                     # annotated bars carry no grid — see _hbar


def draw_collab_sector_papers(ax, D, *, colors=None):
    """Publications with at least one collaborator in each sector."""
    summary = D["collaboration"]["sector_summary"]["papers"].sort_values(ascending=False)
    colors = _sector_colors() if colors is None else colors
    st = _style()
    series = summary.iloc[::-1]
    n_corpus = D["counts"]["corpus"]
    bars = ax.barh([_wrap(i, 22) for i in series.index], series.values,
                   color=[colors[i] for i in series.index],
                   edgecolor=st.get("edgecolor", "black"), linewidth=0.6)
    span = series.max()
    for bar, value in zip(bars, series.values):
        ax.text(value + span * 0.015, bar.get_y() + bar.get_height() / 2,
                f"{value:,} ({100 * value / n_corpus:.1f}%)",
                va="center", fontsize=st["annot_fs"])
    ax.set_xlim(0, span * 1.30)
    ax.set_xlabel(f"Publications (of {n_corpus:,})")
    ax.set_ylabel("Collaborator sector")
    _thousands(ax, "x")
    return ax                     # annotated bars carry no grid — see _hbar


def draw_collab_company_by_year(ax, D, *, colors=None):
    """Publications with a company collaborator, UK against non-UK.

    D27: about 4% of the UK-company mentions are UK organisations that are not companies
    and that the GRID country correction could not reach, so the UK series is a slight
    over-count. It is the smaller of the two series by an order of magnitude either way.
    """
    frame = D["collaboration"]["company_by_year"]
    colors = _sector_colors() if colors is None else colors
    _stacked_bars(ax, frame[COMPANY_SECTORS], colors, "Publication year",
                  "Publications with a company collaborator",
                  legend_title="Company sector")
    return ax


def draw_collab_top_companies(ax, D):
    """The company collaborators themselves, both sectors pooled."""
    top = D["collaboration"]["top_orgs"]
    pooled = Counter()
    for sector in COMPANY_SECTORS:
        pooled.update(top.get(sector, pd.Series(dtype="int64")).to_dict())
    series = _top_counter(pooled, 10)
    series.index = [_shorten(i, 34) for i in series.index]
    _hbar(ax, series, palette("steel_blue"), "Publications",
          ylabel="Company collaborator", wrap=34)
    return ax


def draw_collab_divisions(ax, D):
    """Where in the science industry shows up: company-collaboration rate by division."""
    frame = D["collaboration"]["division_company_share"].head(10)
    st = _style()
    series = frame["pct_company"].iloc[::-1]
    bars = ax.barh([_wrap(i, 26) for i in series.index], series.values,
                   color=palette("cream"),
                   edgecolor=st.get("edgecolor", "black"), linewidth=0.6)
    sizes = frame["size"].iloc[::-1]
    span = series.max()
    for bar, value, size in zip(bars, series.values, sizes):
        ax.text(value + span * 0.02, bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}%  (n={int(size):,})", va="center", fontsize=st["annot_fs"])
    ax.set_xlim(0, span * 1.45)
    ax.set_xlabel("Company-affiliated publications (% of division)")
    ax.set_ylabel("Research division (FOR 2020)")
    return ax                     # annotated bars carry no grid — see _hbar


def draw_collab_flag_overlap(ax, D):
    """Sector overlap, row-normalised: of the papers carrying sector i, what share also
    carries sector j.

    **Read it across the rows, not down the columns**, and the asymmetry is the point.
    The first column is near-saturated all the way down — whoever else a UK Biobank paper
    is written with, it is nearly always written with a university as well — while the UK
    company column is pale everywhere except its own diagonal. A company partnership sits
    on top of an academic one; it does not replace it.

    Row-normalised rather than counted because University/HEI carries 68,978 of the
    mentions and UK company 389: on raw counts the whole matrix would be one bright
    column and seven dark ones, and the six small sectors — which are what the figure is
    about — would be unreadable.

    The diagonal is 100% by construction and is drawn rather than blanked: it is the row
    the reader compares the rest of the row against.
    """
    st = _style()
    frame = D["collaboration"]["flag_overlap"]
    values = frame.to_numpy(dtype=float)
    image = ax.imshow(values, aspect="auto", cmap=_heat_cmap(), vmin=0, vmax=100)
    ax.set_xticks(range(values.shape[1]))
    ax.set_xticklabels([_wrap(c, 11) for c in frame.columns],
                       rotation=35, ha="right", fontsize=st["tick_fs"] - 2)
    # The row tick carries the row's own n: every cell in the row is a percentage OF
    # that number, and a row reading 90% off 365 papers and one reading 90% off 24,988
    # are not the same claim.
    # The n goes on its own line rather than beside the name: the row labels share a
    # half-width column with eight cells, and "Research institute/Centre (n=5,073)" on one
    # line takes a third of the panel's width before a single cell is drawn.
    totals = D["collaboration"]["flag_totals"]
    ax.set_yticks(range(values.shape[0]))
    ax.set_yticklabels([f"{label}\n(n={int(totals[label]):,})" for label in frame.index],
                       fontsize=st["tick_fs"] - 2)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            # No "%" in the cell: the axis label, the colourbar and the caption all
            # say the unit, and eight per-cell percent signs across a half-width panel
            # cost more room than they buy.
            ax.text(j, i, f"{value:.1f}", ha="center", va="center",
                    fontsize=st["annot_fs"] - 2,
                    color=_heat_text_color(value, 100))
    # The x label carries the whole sentence and there is no y label. The row ticks are
    # wide (a sector name over its n), and a y label outside them lands in whatever panel
    # shares the row — at half width there is no margin left to put it in.
    ax.set_xlabel("Co-occurring affiliation sector")
    cbar = ax.figure.colorbar(image, ax=ax, fraction=0.030, pad=0.02)
    cbar.set_label("% of the row's publications", fontsize=st["label_fs"] - 1)
    cbar.set_ticks([0, 20, 40, 60, 80, 100])
    cbar.set_ticklabels([f"{t}%" for t in (0, 20, 40, 60, 80, 100)])
    cbar.ax.tick_params(labelsize=st["tick_fs"] - 1)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1)
    ax.grid(False)
    return ax


def draw_collab_citations(ax, D):
    """Citation distribution by collaboration type. Association, not causation."""
    st = _style()
    groups = D["collaboration"]["citations_by_group"]
    order = ["Academic only", "Other non-academic", "With a company"]
    colors = [_sector_colors()["University/HEI"],
              _sector_colors()["Government/Public"],
              _sector_colors()["Company (non-UK)"]]
    data = [np.log10(groups[k][groups[k] > 0]) for k in order]
    parts = ax.violinplot(data, showextrema=False, widths=0.8)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(0.85)
        body.set_linewidth(0.6)
    for i, series in enumerate(data, start=1):
        ax.scatter([i], [series.median()], color="white", edgecolor="black",
                   zorder=5, s=28)
        ax.text(i, series.median(), f"  {10 ** series.median():.0f}",
                va="center", ha="left", fontsize=st["annot_fs"])
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels([_wrap(o, 14) for o in order])
    ax.set_ylabel("Times cited (log10)")
    ax.set_xlabel("Collaboration type (cited publications)")
    grid_on(ax, axis="y")
    return ax


# =============================================================================
# 6. The assembled figures
# =============================================================================
# One main-paper panel and five SI panels, one SI panel per evidence stream. The letter
# is the whole sub-panel title (`panel_label`); what each letter shows belongs in the
# caption, which `MAIN_CAPTION` / `SI_CAPTIONS` below carry so the figure and its caption
# are written in the same place and cannot drift apart.

MAIN_CAPTION = {
    "A": "Cumulative UK Biobank publications with non-academic linkages, dated by publication "
         "year (log scale). Patent, trial and policy links use Dimensions' publication index, "
         "restricted to outcomes published or started between 1 January 2013 and 31 December 2025.",
    "B": "Legal status of citing patent records from the corpus's endpoint block, by patent "
         "publication year (2013-2025). Bar labels show annual totals; records lacking status "
         "are omitted. Unlike A, this panel counts patents rather than distinct linked publications.",
    "C": "Citing clinical trials by start year and study type, covering each year from 2013 to 2025.",
    "D": "Share of each year's publications with at least one collaborator in each "
         "non-academic sector.",
    "E": "Altmetric Attention Score versus news and policy mentions for 2013-2025 publications "
         "with positive values on both axes. Both axes are logarithmic; each uniform small point "
         "is a publication. Counts are source-snapshot totals. Upper callouts identify the two "
         "most-mentioned publications.",
    "F": "Row-normalised affiliation-sector overlap: the percentage of publications in each "
         "row's sector also represented in each column's sector. Read across rows; denominators "
         "appear on the axis.",
}

SI_CAPTIONS = {
    "patents": {
        "A": "Research divisions (FOR 2020, top level) the patents are classified into.",
        "B": "Patents by publication year, split application against granted.",
        "C": "Number of research divisions one patent spans.",
        "D": "Research-division mix within each of the eight largest assignee countries.",
    },
    "trials": {
        "A": "Trial lifecycle stage, by study type (nine registry statuses folded into "
             "five).",
        "B": "Trials per ICD-10 chapter on a schematic body outline; marker area is "
             "directly proportional to the trial count, using the displayed size key. "
             "Positions indicate broad body systems, not precise organ locations. "
             "Neoplasms and infectious disease are systemic and sit off the body. "
             "A trial counts once per chapter it touches.",
        "C": "RCDC disease categories, with cross-cutting research-area tags removed.",
        "D": "Planned enrollment per trial (log scale).",
        "E": "Where the trials are run and who runs them: fractional trial weight by "
             "country and organisation sector.",
    },
    "policy": {
        "A": "Policy documents citing UK Biobank research, by year and publisher origin.",
        "B": "Publisher countries, on a logarithmic colour scale. White is a country with "
             "no citing document, which is a different statement from the ramp's "
             "lightest blue; the leading publishers are named with their counts.",
        "C": "Publishing institutions.",
        "D": "Research divisions (FOR 2020) the documents are classified into.",
    },
    "altmetric": {
        "A": "Distribution of the Altmetric Attention Score over every scored publication.",
        "B": "Total news (blue, left axis) and policy (red, right axis) mentions, by publication year.",
        "C": "Share of each year's publications with at least one news (blue) or policy (red) mention.",
        "D": "Attention against citations within the positive-attention, positive-mention "
             "cohort, restricted to publications with positive citations. Both axes are "
             "logarithmic; each point represents one publication.",
    },
    "collaboration": {
        "A": "Collaborator organisation mentions, by sector.",
        "B": "Publications with at least one collaborator in each sector.",
        "C": "Share of each year's publications with a collaborator in each non-academic "
             "sector.",
        "D": "Publications with a company collaborator, UK against non-UK.",
        "E": "The company collaborators themselves.",
        "F": "Company-collaboration rate by research division (divisions with 100+ "
             "publications).",
    },
}


def _grid_kw(height_ratios=None, width_ratios=None) -> dict:
    """`gridspec_kw` for whichever of the two ratio lists were given (possibly neither).

    Built in one place because both branches of `_assemble` need it and they used to
    disagree: the plain-grid branch honoured height ratios only, so a `width_ratios=`
    argument raised, and the spanning-slots branch silently dropped it.
    """
    kw = {}
    if height_ratios is not None:
        kw["height_ratios"] = list(height_ratios)
    if width_ratios is not None:
        kw["width_ratios"] = list(width_ratios)
    return kw


def _panel_grid(nrows, ncols, figsize, *, hspace=0.42, wspace=0.26,
                height_ratios=None, width_ratios=None):
    apply_typography()
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             gridspec_kw=_grid_kw(height_ratios, width_ratios) or None)
    finalize_figure(fig)
    fig.subplots_adjust(hspace=hspace, wspace=wspace)
    return fig, np.atleast_1d(axes).ravel()


def _assemble(spec, nrows, ncols, figsize, D, name, save=True, slots=None,
              hspace=0.42, wspace=0.26, height_ratios=None, width_ratios=None,
              label_panels=True):
    """Draw `spec` (an ordered list of draw functions) into a letter-labelled grid.

    Without `slots` the panels fill an `nrows` x `ncols` grid one cell each, in order.
    With `slots` — one `(rows, cols)` entry per draw function, each an int or a `slice` —
    a panel can span cells: a heatmap that needs the full page width takes
    `(2, slice(0, 2))`, a body map that needs height takes `(slice(0, 2), 1)`. Panel
    letters follow `spec` order regardless, so the letters and the reading order are the
    author's decision rather than a by-product of the geometry.

    `height_ratios` / `width_ratios` size the grid's rows and columns relative to each
    other (`[1.0, 1.3]` makes the right column 30% wider). Both work with or without
    `slots`, and both are lists as long as the grid's rows / columns — NOT as long as
    `spec`, which is a different number as soon as a panel spans a cell.
    """
    apply_typography()
    letters = "ABCDEFGHIJKL"
    if slots is None:
        fig, axes = _panel_grid(nrows, ncols, figsize, hspace=hspace, wspace=wspace,
                                height_ratios=height_ratios, width_ratios=width_ratios)
        for ax in axes[len(spec):]:                # an unfilled cell is removed, not blank
            ax.remove()
        axes = list(axes[:len(spec)])
    else:
        if len(slots) != len(spec):
            raise ValueError(f"{len(spec)} draw functions but {len(slots)} slots")
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(nrows, ncols, hspace=hspace, wspace=wspace,
                              **_grid_kw(height_ratios, width_ratios))
        axes = [fig.add_subplot(gs[rows, cols]) for rows, cols in slots]

    for ax, letter, draw in zip(axes, letters, spec):
        draw(ax, D)
        if label_panels:                # a one-panel figure has nothing to letter
            panel_label(ax, letter)
    finalize_figure(fig)
    if save:
        savefig(fig, name)
    return fig


def figure_main(D, save=True):
    """The main-paper panel: six charts on three rows, two to a row.

    The reach bars came out. They answered "how many publications carry each linkage",
    which is the same question `01_growth`'s reach panel answers on the same numbers —
    two figures in one paper making one point twice. `D["reach"]` is still built, so the
    counts remain quotable in the text and in `panel_selection.csv`.

    The attention scatter takes the freed row across both columns AND half again the
    height of the rows above it. It is the one panel with 5,601 points and two named
    outliers; the empty band the labels sit in is the space above a diagonal cloud, so
    making the panel taller is what shortens the leaders — at equal row height the boxes
    had to sit far left of their points to clear the data.

    F, the sector-overlap matrix, shares that row. It is an eight-by-eight matrix with a
    number in every cell, so it pays for the half width in type: the cells drop their
    percent signs (the axis, the colourbar and the caption all carry the unit), the row
    labels put their n on a second line, and the column labels wrap at eleven characters.
    What it buys is that the figure closes on one row rather than two, and that the two
    panels a reader is meant to hold together — the attention cloud and the structure of
    the collaboration D counts — are side by side instead of a page apart.

    The row is 1.55x the height of the rows above it, which is what both panels need. E's
    two named outliers sit in the empty band above a diagonal cloud, so height is what
    shortens their leaders; F is square by nature and would otherwise draw eight rows into
    the height of four.

    Drawn at the `main` type scale, larger than the SI panels because this one is read at
    figure width in a paper rather than full page on a screen.
    """
    st = _style()
    figsize = st["figsize_main"]
    with _font_scale(_fs_scale("main")):
        return _assemble(
            [draw_reach_by_year,            # A
             draw_patent_legal_status,      # B
             draw_trials_by_year,           # C
             draw_collab_sector_share,      # D
             draw_altmetric_scatter,        # E - bottom row, left
             draw_collab_flag_overlap],     # F - bottom row, right
            3, 2, (figsize[0], figsize[1] * 1.30),
            D, P.MAIN_FIGURE_STEMS[6], save=save,
            slots=[(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)],
            hspace=0.40, wspace=0.34, height_ratios=[1.0, 1.0, 1.55],
        )


def figure_si_patents(D, save=True):
    """Four patent panels on an ordinary 2x2 grid.

    The RCDC macro-cluster heatmap that once shared this page, and then had a page of its
    own, was removed on 2026-09-22 (D46): its Louvain communities could not carry their
    reviewed names onto the current patent cohort, and an unnamed partition is not a result.

    The paper-to-patent lag also came out earlier and stays out; its distribution is in
    `D["patents"]["paper_to_patent_lag"]`.
    """
    st = _style()
    with _font_scale(_fs_scale("si1_patents")):
        # The right column is the wider one: B stacks seven legal-status categories under
        # a two-column legend and D is an eight-by-eight heatmap, where A and C are a
        # ten-row bar chart and a four-bin histogram that read fine narrower.
        return _assemble(
            [draw_patent_topics,            # A
             draw_patents_by_year,          # B
             draw_patent_topic_count,       # C
             draw_patent_country_topics],   # D
            2, 2, st["figsize_si"], D, "04_02_supplementary_figure_01_patents", save=save,
            hspace=0.45, wspace=0.40, width_ratios=[1.0, 1.3],
        )


def figure_si_trials(D, save=True):
    """Five trial panels, with the body map spanning the right column's top two rows.

    The body map is equal-aspect, so height is what it needs: given two rows it draws a
    body rather than a squashed one, and its eleven labels have room to sit beside their
    dots. The paper-to-trial lag came out — it is the least load-bearing of the six, and
    the space buys the body map its rows.
    """
    st = _style()
    width, height = st["figsize_si"]
    with _font_scale(_fs_scale("si2_clinical_trials")):
        name = "04_03_supplementary_figure_02_clinical_trials"
        fig = _assemble(
            [draw_trial_stage,              # A  row 0, left
             draw_trial_bodymap,            # B  rows 0-1, right
             draw_trial_rcdc,               # C  row 1, left
             draw_trial_enrollment,         # D  row 2, left
             draw_trial_country_sector],    # E  row 2, right
            3, 2, (width, height * 1.30), D, name,
            save=False, slots=[(0, 0), (slice(0, 2), 1), (1, 0), (2, 0), (2, 1)],
            hspace=0.40, wspace=0.42, height_ratios=[1.0, 1.0, 1.05],
        )
        # Translate the complete schematic, preserving its equal-aspect anatomy.
        body = fig.axes[1]
        bounds = body.get_position(original=True)
        body.set_position([bounds.x0 - .04, bounds.y0, bounds.width, bounds.height])
        finalize_figure(fig)
        if save:
            savefig(fig, name)
        return fig


def figure_si_policy(D, save=True):
    """Four policy panels on a 2x2 grid, using the approved blue/red palette.

    **Two panels came out.** The concentration histogram (how many documents cite one
    paper) and the ranked list of most-cited papers were the page's two weakest claims:
    the first is a one-line fact — median 1, max 22 over 369 publications — that prose
    carries better than a log-axis histogram of a distribution with nothing in its tail,
    and the second is eight truncated paper titles, which is a table pretending to be a
    chart. Both aggregates are still built (`D["policy"]["policy_docs_per_paper"]`,
    `D["policy"]["top_cited_papers"]`) and both draw functions are still here, so either
    can be quoted or re-added without re-deriving anything.

    **B is now a choropleth** rather than the ranked country bar, which is where a
    geographic finding belongs; `draw_policy_countries` still draws the bar.

    The yearly origin split uses steel blue and red; publisher and division bars use
    those same colours. The geographic count scale remains sequential blue.
    """
    st = _style()
    width, height = st["figsize_si"]
    name = "04_04_supplementary_figure_03_policy"
    fig = _assemble(
        [draw_policy_by_year, draw_policy_country_map, draw_policy_publishers,
         draw_policy_divisions],
        2, 2, (width, height * 0.92), D, name,
        save=False, hspace=0.42, wspace=0.85, height_ratios=[0.85, 1.15],
    )
    for ax in fig.axes:
        ax._ukb_label_fs = 16
        ax._ukb_tick_fs = 12
        ax._ukb_annotation_fs = 12
        ax._ukb_legend_fs = 12
        ax._ukb_title_fs = 24
    for ax in (fig.axes[0], fig.axes[2], fig.axes[3]):
        ax.set_axisbelow(True)
        ax.grid(False, which="both")
        grid_on(ax, axis="both", which="major")
    finalize_figure(fig)
    if save:
        savefig(fig, name)
    return fig


def figure_si_altmetric(D, save=True):
    """A square 2x2 with shared news/policy colours and major-tick grids throughout."""
    st = _style()
    side = st["figsize_si"][0]
    name = "04_05_supplementary_figure_04_altmetric"
    fig = _assemble(
        [draw_altmetric_distribution, draw_altmetric_mentions_by_year,
         draw_altmetric_coverage, draw_altmetric_vs_citations],
        2, 2, (side, side), D, name, save=False, hspace=.28, wspace=.28,
    )
    fig.subplots_adjust(left=.09, right=.91, bottom=.085, top=.905)
    for ax in fig.axes:
        ax._ukb_label_fs = 14
        ax._ukb_tick_fs = 12
        ax._ukb_legend_fs = 12
        ax.set_box_aspect(1)
        ax.grid(False, which="both")
    # Only the primary axis in B draws a grid; a second grid would imply alignment
    # between two different units. Log axes get decade grids, never minor-tick hatching.
    for ax in fig.axes[:4]:
        grid_on(ax, which="major", log=True)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1)
    finalize_figure(fig)
    if save:
        savefig(fig, name, bbox_inches=None)
    return fig


def figure_si_collaboration(D, save=True):
    from functools import partial

    st = _style()
    colors = {**_sector_colors(), "Company (non-UK)": palette("steel_blue"),
              "Government/Public": palette("navy"), "UK company": palette("red")}
    return _assemble(
        [partial(draw_collab_sector_summary, colors=colors),
         partial(draw_collab_sector_papers, colors=colors),
         partial(draw_collab_sector_share, colors=colors),
         partial(draw_collab_company_by_year, colors=colors),
         draw_collab_top_companies, draw_collab_divisions],
        3, 2, (17, 14), D, "04_06_supplementary_figure_05_collaboration", save=save,
        hspace=0.45, wspace=0.55,
    )


#: name -> builder, in the order the notebook draws them.
FIGURES = {
    "main": figure_main,
    "si1_patents": figure_si_patents,
    "si2_clinical_trials": figure_si_trials,
    "si3_policy": figure_si_policy,
    "si4_altmetric": figure_si_altmetric,
    "si5_collaboration": figure_si_collaboration,
}
