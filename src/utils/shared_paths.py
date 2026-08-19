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
SHOWCASE_PLUS = SHOWCASE / "showcase_plus_all_endpoint.parquet"

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

# -- static reference data ----------------------------------------------------------
WORLD_SHP = DATA / "ne_110m_admin_0_countries" / "ne_110m_admin_0_countries.shp"

# -- deliverables: everything a notebook exports lands under output/ -----------------
# The notebooks' savedirs are configured in universal_settings.yml; these constants are
# for code that writes a file directly.
OUTPUT = ROOT / "output"
OUTPUT_FIGURES = OUTPUT / "figures"
FIG_DATA_ANALYSIS = OUTPUT_FIGURES / "data_analysis"
FIG_DATA_CREATION = OUTPUT_FIGURES / "data_creation"
OUTPUT_TABLES = OUTPUT / "tables"

FIG_AUTHORS = FIG_DATA_ANALYSIS / "01_authors"
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


def ensure_dirs() -> None:
    """Create the output directories that notebooks write into, if missing."""
    for d in (AUTHOR_ANALYSIS, ACADEMIC_IMPACT, FOR_COUNTS, FOR_COUNTS_API,
              CLINICAL_TRIALS, PATENT,
              POLICY, DIMENSION_CACHE, DIMENSION_FLAT, OUTPUT_TABLES,
              FIG_AUTHORS, FIG_NETWORK, FIG_NON_ACADEMIC, FIG_CLINICAL_TRIALS,
              FIG_ACADEMIC_IMPACT, FIG_PATENT):
        d.mkdir(parents=True, exist_ok=True)
