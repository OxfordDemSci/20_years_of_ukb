"""Single source of truth for every path the analysis code touches.

Why this exists: the notebooks used to `os.chdir()` up to the repo root by testing
`Path.cwd().name == "src"`, then use relative strings ('data/...', 'fig/...'). That
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
SHOWCASE_PLUS_CANDIDATES = (
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
# answer (the API path has no fractional and no citation-weighted columns).
FOR_COUNTS_API = ACADEMIC_IMPACT / "for_counts_api"

# Analysis 02 (content / topic modelling). Registered here so 02's "five places" all
# exist — the 02 notebooks do not read these yet (they still carry Colab-era literals and
# are left untouched for now); wire them up when those notebooks are next opened.
CONTENT = ANALYSIS / "content"
BERTOPIC_CACHE = CONTENT / "cache"
TOPIC_ASSIGNMENTS = CONTENT / "showcase_plus_id_topics.csv"

NON_ACADEMIC = ANALYSIS / "non_academic"
CLINICAL_TRIALS = NON_ACADEMIC / "clinical_trials"
PATENT = NON_ACADEMIC / "patent"
POLICY = NON_ACADEMIC / "policy"

# -- named files that more than one notebook reads ---------------------------------
AUTHOR_PAPER_CACHE = AUTHOR_ANALYSIS / "df_author_paper.parsed.pkl"
FIELD_COUNTS = FOR_COUNTS / "field_counts.parquet"
FIELD_TOTALS = FOR_COUNTS / "field_totals.parquet"
FIELD_COVERAGE = FOR_COUNTS / "field_coverage.parquet"
CT_CSV = CLINICAL_TRIALS / "clinical_trials.csv"
CT_UKBB_PAPERS = CLINICAL_TRIALS / "ct_ukbb_papers.csv"
PATENTS_DETAILED = PATENT / "patents_detailed.csv"
POLICY_CSV = POLICY / "policy_documents.csv"
# Read as a bare "altmetric.csv" / "output/..." relative to the notebook's cwd until
# 2026-08-26, so both resolved differently depending on where the kernel was launched.
ALTMETRIC_CSV = NON_ACADEMIC / "altmetric.csv"
# The Altmetric export above is NOT in the repo and has no provenance record. This
# is the substitute rebuilt from the corpus + the policy pull: same column names,
# so a real export can be dropped in later and the notebook switches by path alone.
# It is NOT equivalent — it carries no news mentions, and its "Policy mentions" are
# Dimensions policy citations. See data_analysis_04_non_academic_altmetric_from_corpus.py.
ALTMETRIC_DERIVED = NON_ACADEMIC / "altmetric_from_corpus.csv"
COLLAB_FLAGGED = ROOT / "output" / "non_academic_flagged_full_company.xlsx"

# -- static reference data ----------------------------------------------------------
WORLD_SHP = DATA / "ne_110m_admin_0_countries" / "ne_110m_admin_0_countries.shp"
# FOR id <-> code <-> name (80003 <-> 32 <-> 'Biomedical and Clinical Sciences'). It sits
# in doc/ rather than data/ because it ships with the repo and the methodology cites it.
FOR_2020_CODES = ROOT / "doc" / "category_for_2020_codes.csv"

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
FIG_AUTHORS = FIG_DATA_ANALYSIS / "01_authors"
FIG_CONTENT = FIG_DATA_ANALYSIS / "02_content"
FIG_NETWORK = FIG_DATA_ANALYSIS / "02_network"
FIG_ACADEMIC_IMPACT = FIG_DATA_ANALYSIS / "03_academic_impact"
FIG_NON_ACADEMIC = FIG_DATA_ANALYSIS / "04_non_academic"
FIG_CLINICAL_TRIALS = FIG_NON_ACADEMIC / "clinical_trials"

# Pre-rearrangement figure tree. Still on disk (fig/network, fig/patent, fig/geography)
# and still the home of the patent figures, so it is kept rather than repointed.
FIG = ROOT / "fig"
FIG_PATENT = FIG / "patent"
FIG_GEOGRAPHY = FIG / "geography"


def bootstrap() -> Path:
    """Put `src` on sys.path and make the repo root the working directory.

    Returns ROOT. Safe to call repeatedly (re-running a notebook cell is the normal
    case), and safe to call from any starting directory inside the repo.
    """
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    if Path.cwd().resolve() != ROOT:
        os.chdir(ROOT)
    return ROOT


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
              POLICY, DIMENSION_CACHE, DIMENSION_FLAT, OUTPUT_TABLES,
              FIG_GROWTH, TABLE_GROWTH,
              FIG_AUTHORS, FIG_CONTENT, FIG_NETWORK, FIG_NON_ACADEMIC,
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
