"""Single source of truth for every path the analysis code touches.

Why this exists: the notebooks used to `os.chdir()` up to the repo root by testing
`Path.cwd().name == "src"`, then use relative strings for data and figures. That
test silently stopped working once the notebooks moved into `src/data_analysis/`,
and the relative strings had already drifted apart (`data/patent/` vs
`data/non_academic/patents/`, `clinic_trials` vs `clinical_trials`). Anchoring on
this file's own location removes both problems: paths resolve identically whether a
notebook is run from the repo root, from `src/`, or from `src/data_analysis/`.

Usage (top of every notebook / script):

    import sys
    from pathlib import Path
    ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "utils").is_dir())
    sys.path.insert(0, str(ROOT / "src"))
    from utils import shared_paths as P
    P.bootstrap()          # chdir to ROOT so any leftover relative path still works
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path

# utils/ -> src/ -> repo root
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"

DATA_ANALYSIS = SRC / "data_analysis"
DATA_CREATION = SRC / "data_creation"
UTILS = SRC / "utils"

DATA = ROOT / "data"

# -- the UK Biobank publication corpus (Dimensions records joined to the showcase) --
SHOWCASE = DATA / "showcase"
# Newest first. The WIDE export is the one to prefer: same 26,109 rows as the narrow
# one, but 317 columns instead of 72 because it carries the Dimensions endpoint linkage
# (`patents__linked_ids`, `clinical_trials__n_records`, and so on for all six endpoints).
# It is a strict superset, so a notebook written against the narrow file keeps working.
#
# It is built by `src/data_creation/01_scraper_full_endpoint_collection_and_merge/
# dimensions_endpoint_pipeline.py`, and it has landed in BOTH showcase directories over
# the project's life — hence two wide entries rather than one. The narrow file is kept
# last as a fallback, but note that anything reading a `*__linked_ids` column will raise
# on it rather than silently return nothing (`pd.read_parquet(columns=...)` is strict).
SHOWCASE_PLUS_CANDIDATES = (
    SHOWCASE / "showcase+" / "showcase_plus_all_endpoints_wide.parquet",
    SHOWCASE / "showcase_plus" / "showcase_plus_all_endpoints_wide.parquet",
    SHOWCASE / "showcase+" / "showcase_plus_all_endpoint.parquet",
)
SHOWCASE_PLUS = next(
    (path for path in SHOWCASE_PLUS_CANDIDATES if path.exists()),
    SHOWCASE_PLUS_CANDIDATES[0],
)

# -- raw Dimensions pulls: per-endpoint caches + the flattened exports --------------
DIMENSION = DATA / "dimension"
DIMENSION_CACHE = DIMENSION / "cache"
DIMENSION_FLAT = DIMENSION / "flat"

# -- analysis outputs, one directory per analysis notebook -------------------------
ANALYSIS = DATA / "analysis"
AUTHOR_ANALYSIS = ANALYSIS / "author_analysis"
ACADEMIC_IMPACT = ANALYSIS / "academic_impact"
# The category-count job's output directory. Named for the FOR run that came first, but
# it holds the partials of every classification system (counts.rcdc.*, counts.uoa.*, ...)
# because they are named per-category and share one merge.
FOR_COUNTS = ACADEMIC_IMPACT / "for_counts_out"
# The same tables built the other way: whole-database counts faceted straight out of
# the Dimensions API instead of counted off the VM corpus copy. Same schema, same
# filenames, so a notebook switches pathway by changing COUNTS_DIR and nothing else —
# see data_analysis_03_academic_impact_dimensions_api.py for what each can and cannot
# answer. The API path DOES carry citation weights (times_cited, mncs, and the measured
# per-field decile and median); what it cannot give is a fractional count on the
# background side, or a per-paper FCR there — a facet returns totals, not per-paper rows.
FOR_COUNTS_API = ACADEMIC_IMPACT / "for_counts_api"

# Analysis 02: training caches live here; current results use output/bertopic/.
# TOPIC_ASSIGNMENTS remains a recognised location for existing topic-result CSVs.
CONTENT = ANALYSIS / "content"
BERTOPIC_CACHE = CONTENT / "cache"
TOPIC_ASSIGNMENTS = CONTENT / "showcase_plus_id_topics.csv"

NON_ACADEMIC = ANALYSIS / "non_academic"
CLINICAL_TRIALS = NON_ACADEMIC / "clinical_trials"
PATENT = NON_ACADEMIC / "patent"
POLICY = NON_ACADEMIC / "policy"
# The collaborator-tagging arm. Its output used to land in `output/` next to the figures,
# which put an expensive, re-derivable *input* to the analysis among the deliverables. It
# is an analysis-derived table like every other file under `data/analysis/`, so it lives
# here with them.
COLLABORATION = NON_ACADEMIC / "collaboration"

# -- named files that more than one notebook reads ---------------------------------
AUTHOR_PAPER_CACHE = AUTHOR_ANALYSIS / "df_author_paper.parsed.pkl"
FIELD_COUNTS = FOR_COUNTS / "field_counts.parquet"
FIELD_TOTALS = FOR_COUNTS / "field_totals.parquet"
FIELD_COVERAGE = FOR_COUNTS / "field_coverage.parquet"
CT_CSV = CLINICAL_TRIALS / "clinical_trials.csv"
CT_UKBB_PAPERS = CLINICAL_TRIALS / "ct_ukbb_papers.csv"
# The legacy export contains the original full patent records plus derived columns.
# Prefer the canonical source when present; notebook analyses recompute derivations.
_PATENT_SOURCE_CANDIDATES = (
    PATENT / "patents_detailed.csv",
    DATA / "patent" / "df_with_iso_aggressive.csv",
)
PATENTS_DETAILED = next(
    (path for path in _PATENT_SOURCE_CANDIDATES if path.is_file()),
    _PATENT_SOURCE_CANDIDATES[0],
)
# RCDC macro-cluster partition (Louvain) behind §4.1 of the patents notebook. Reached as a
# bare "file/paten_rcdc_macro/..." until 2026-08-26 — a directory that has never existed
# here, so the notebook died on that cell. (Directory name misspelled on disk; kept.)
PATENT_RCDC_MACRO = PATENT / "paten_rcdc_macro"
_PATENT_RCDC_CANDIDATES = (
    PATENT_RCDC_MACRO / "cluster_label_summary_louvain.csv",
    DATA / "patent" / "cluster_label_summary_louvain.csv",
)
PATENT_RCDC_SUMMARY = next(
    (path for path in _PATENT_RCDC_CANDIDATES if path.is_file()),
    _PATENT_RCDC_CANDIDATES[0],
)
POLICY_CSV = POLICY / "policy_documents.csv"
# The real Altmetric Explorer export, with its audit trail and raw response cache beside
# it. Read as a bare "altmetric.csv" / "output/..." relative to the notebook's cwd until
# 2026-08-26 (so it resolved differently depending on where the kernel was launched), then
# pointed at `data/analysis/non_academic/` where no export had ever been placed. It is a
# SOURCE, not an analysis output, so it belongs under `data/` beside the other pulls.
ALTMETRIC_DIR = DATA / "altmetric"
ALTMETRIC_CSV = ALTMETRIC_DIR / "altmetric.csv"
# The Altmetric export above is NOT in the repo and has no provenance record. This
# is the substitute rebuilt from the corpus + the policy pull: same column names,
# so a real export can be dropped in later and the notebook switches by path alone.
# It is NOT equivalent — it carries no news mentions, and its "Policy mentions" are
# Dimensions policy citations. See data_analysis_04_non_academic_altmetric_from_corpus.py.
ALTMETRIC_DERIVED = NON_ACADEMIC / "altmetric_from_corpus.csv"
# CSV, not xlsx: `_write_frame` JSON-encodes the list columns, and `authors` on the
# largest paper in the corpus (2,119 author slots) serialises well past Excel's
# 32,767-character cell limit -- openpyxl would raise or silently truncate. The helpers
# read either suffix and parse the JSON back, so the choice costs nothing downstream.
COLLAB_FLAGGED = COLLABORATION / "non_academic_flagged_full_company.csv"
# Per-institution-set classifier cache, keyed on the institution tuple and fsynced per
# batch. It is what makes a re-run free, so it is named here rather than in the notebook.
COLLAB_CACHE = COLLABORATION / "collab_classifier_cache.jsonl"

# -- static reference data ----------------------------------------------------------
# Natural Earth 110m admin-0 countries, v5.1.1. The old constant named a directory that
# has never existed on this machine, so every choropleth raised; the shapefile is in
# `data/shapefile/`, with its provenance in the README beside it.
WORLD_SHP = DATA / "shapefile" / "ne_110m_admin_0_countries" / "ne_110m_admin_0_countries.shp"
# FOR id <-> code <-> name (80003 <-> 32 <-> 'Biomedical and Clinical Sciences'). It sits
# in doc/ rather than data/ because it ships with the repo and the methodology cites it.
FOR_2020_CODES = ROOT / "doc" / "category_for_2020_codes.csv"

# -- credentials --------------------------------------------------------------------
# Same shape and same gitignore rule as the Dimensions key in config/dsl.ini: the file
# itself is never committed, config/anthropic.ini.example shows the section and key names.
# Read only by the collaboration classifier, and only when ANTHROPIC_API_KEY is unset.
ANTHROPIC_INI = ROOT / "config" / "anthropic.ini"

# -- deliverables: everything a notebook exports lands under output/ -----------------
# The notebooks' savedirs are configured in universal_settings.yml; these constants are
# for code that writes a file directly.
OUTPUT = ROOT / "output"
OUTPUT_FIGURES = OUTPUT / "figures"
FIG_DATA_ANALYSIS = OUTPUT_FIGURES / "data_analysis"
FIG_DATA_CREATION = OUTPUT_FIGURES / "data_creation"
OUTPUT_TABLES = OUTPUT / "tables"
TABLE_DATA_ANALYSIS = OUTPUT_TABLES / "data_analysis"

FIG_GROWTH = FIG_DATA_ANALYSIS / "01_growth"
TABLE_GROWTH = TABLE_DATA_ANALYSIS / "01_growth"
# Note the level: 01_growth's tables sit under output/tables/data_analysis/, while 03's are
# directly under output/tables/. 02 follows 03, the more recent of the two. The split is an
# inconsistency, recorded rather than fixed here.
TABLE_CONTENT = OUTPUT_TABLES / "02_content"
TABLE_ACADEMIC_IMPACT = OUTPUT_TABLES / "03_academic_impact"
FIG_AUTHORS = FIG_DATA_ANALYSIS / "01_authors"
FIG_AUTHOR_CHARACTERISTICS = FIG_DATA_ANALYSIS / "05_author_characteristics"
TABLE_AUTHOR_CHARACTERISTICS = TABLE_DATA_ANALYSIS / "05_author_characteristics"
FIG_CONTENT = FIG_DATA_ANALYSIS / "02_content"
FIG_NETWORK = FIG_DATA_ANALYSIS / "02_network"
FIG_ACADEMIC_IMPACT = FIG_DATA_ANALYSIS / "03_academic_impact"
FIG_NON_ACADEMIC = FIG_DATA_ANALYSIS / "04_non_academic"
FIG_CLINICAL_TRIALS = FIG_NON_ACADEMIC / "clinical_trials"

# Compatibility names for older notebooks; all figures now live under output/.
FIG = OUTPUT_FIGURES
FIG_PATENT = FIG_NON_ACADEMIC / "patent"
FIG_GEOGRAPHY = FIG_AUTHOR_CHARACTERISTICS / "geography"


def bootstrap() -> None:
    """Put `src` on sys.path and make the repo root the working directory.

    Return nothing so a notebook's display hook cannot expose the absolute root.
    Callers can use ROOT internally. Safe to call repeatedly and from any starting
    directory inside the repo.
    """
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    if Path.cwd().resolve() != ROOT:
        os.chdir(ROOT)


def raw_path(path: Path) -> str:
    """Return a repository-relative POSIX path for logs, tables, and notebook output."""
    path = Path(path)
    if path.is_absolute():
        try:
            path = path.relative_to(ROOT)
        except ValueError:
            return path.name
    return path.as_posix()


def ensure_dirs() -> None:
    """Create the output directories that notebooks write into, if missing."""
    for d in (AUTHOR_ANALYSIS, ACADEMIC_IMPACT, FOR_COUNTS, FOR_COUNTS_API,
              CONTENT, BERTOPIC_CACHE, CLINICAL_TRIALS, PATENT,
              POLICY, COLLABORATION, DIMENSION_CACHE, DIMENSION_FLAT, OUTPUT_TABLES,
              FIG_GROWTH, TABLE_GROWTH, TABLE_CONTENT,
              FIG_AUTHORS, FIG_AUTHOR_CHARACTERISTICS,
              TABLE_ACADEMIC_IMPACT, TABLE_AUTHOR_CHARACTERISTICS,
              FIG_CONTENT, FIG_NETWORK, FIG_NON_ACADEMIC,
              FIG_CLINICAL_TRIALS, FIG_ACADEMIC_IMPACT, FIG_PATENT):
        d.mkdir(parents=True, exist_ok=True)


class ArtifactRegistry:
    """Track and export a notebook's figure, table, workbook, and text artifacts."""

    def __init__(self, table_dir: Path):
        self.table_dir = Path(table_dir)
        self.figure_paths: list[Path] = []
        self.table_paths: list[Path] = []

    @staticmethod
    def _register(paths: list[Path], path: Path) -> Path:
        """Record an artifact once, keeping notebook cell reruns idempotent."""
        path = Path(path)
        if path not in paths:
            paths.append(path)
        return path

    def save_table(self, frame, filename, index=False) -> Path:
        path = self.table_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=index)
        self._register(self.table_paths, path)
        print("saved", raw_path(path))
        return path

    def save_text(self, text: str, filename: str) -> Path:
        path = self.table_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        self._register(self.table_paths, path)
        print("saved", raw_path(path))
        return path

    def save_workbook(self, sheets: Mapping[str, object], filename: str) -> Path:
        import pandas as pd

        path = self.table_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for sheet_name, frame in sheets.items():
                frame.to_excel(writer, sheet_name=sheet_name[:31], index=False)
        self._register(self.table_paths, path)
        print("saved", raw_path(path))
        return path

    def record_figures(self, paths):
        paths = list(paths)
        for path in paths:
            self._register(self.figure_paths, path)
        return paths

    def save_word_table(self, frame, filename, **kwargs) -> Path:
        """Export an editable manuscript table and include it in the manifest."""
        from .shared_word import save_word_table

        path = save_word_table(frame, self.table_dir / filename, **kwargs)
        self._register(self.table_paths, path)
        print("saved", raw_path(path))
        return path

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def save_manifest(self, filename: str):
        import pandas as pd

        rows = [
            {
                "kind": "figure",
                "name": path.stem,
                "format": path.suffix.lstrip("."),
                "path": raw_path(path),
                "size_bytes": path.stat().st_size,
                "sha256": self._file_sha256(path),
            }
            for path in self.figure_paths
        ]
        rows.extend(
            {
                "kind": "table_or_text",
                "name": path.stem,
                "format": path.suffix.lstrip("."),
                "path": raw_path(path),
                "size_bytes": path.stat().st_size,
                "sha256": self._file_sha256(path),
            }
            for path in self.table_paths
        )
        manifest = pd.DataFrame(rows).sort_values(["kind", "name", "format"])
        path = self.table_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(path, index=False)
        print("saved", raw_path(path))
        return manifest, path
