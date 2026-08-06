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
FOR_COUNTS = ACADEMIC_IMPACT / "for_counts"

NON_ACADEMIC = ANALYSIS / "non_academic"
CLINICAL_TRIALS = NON_ACADEMIC / "clinical_trials"
PATENT = NON_ACADEMIC / "patent"
POLICY = NON_ACADEMIC / "policy"

# -- named files that more than one notebook reads ---------------------------------
AUTHOR_PAPER_CACHE = AUTHOR_ANALYSIS / "df_author_paper.parsed.pkl"
FIELD_COUNTS = FOR_COUNTS / "field_counts.parquet"
FIELD_TOTALS = FOR_COUNTS / "field_totals.parquet"
CT_CSV = CLINICAL_TRIALS / "clinical_trials.csv"
CT_UKBB_PAPERS = CLINICAL_TRIALS / "ct_ukbb_papers.csv"
PATENTS_DETAILED = PATENT / "patents_detailed.csv"
POLICY_CSV = POLICY / "policy_documents.csv"

# -- static reference data + figure output -----------------------------------------
WORLD_SHP = DATA / "ne_110m_admin_0_countries" / "ne_110m_admin_0_countries.shp"

FIG = ROOT / "fig"
FIG_AUTHORS = FIG / "authors"
FIG_NETWORK = FIG / "network"
FIG_PATENT = FIG / "patent"
FIG_NON_ACADEMIC = FIG / "non_academic"

FILE = ROOT / "file"


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
    for d in (AUTHOR_ANALYSIS, ACADEMIC_IMPACT, FOR_COUNTS, CLINICAL_TRIALS, PATENT,
              POLICY, DIMENSION_CACHE, DIMENSION_FLAT,
              FIG_AUTHORS, FIG_NETWORK, FIG_PATENT, FIG_NON_ACADEMIC):
        d.mkdir(parents=True, exist_ok=True)
