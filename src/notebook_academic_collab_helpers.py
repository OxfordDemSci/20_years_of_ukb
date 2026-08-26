"""Compatibility shim: `notebook_academic_collab_helpers` -> the real module in utils/.

WHY THIS EXISTS
---------------
Two scripts import this module by its old, unqualified name:

    src/data_analysis/02_content_4_top10_broad_research_areas.py
    src/data_analysis/02_content_5_top10_fastest_growing_research_areas.py

The file behind that name had been renamed to
`src/data_analysis/04_notebook_academic_collab_helpers__TO_UTILS.py`, which is not a
valid Python module name (it starts with a digit), so **every one of those imports was
failing** with ModuleNotFoundError until 2026-08-26. The module now lives at its
convention-correct name, `utils/data_analysis_04_non_academic_collab_helpers.py`, and
this shim keeps those callers working.

Rebinding sys.modules rather than re-exporting names means `h.<anything>` resolves
against the real module -- all 95 public helpers, not just the ones an `import *` would
have picked up.

REMOVE THIS FILE when the three callers import the real module directly. The two
02_content_* scripts were deliberately left untouched in the 2026-08-26 pass (see
doc/decisions_04_non_academic.md), which is the only reason the shim is still needed.
"""

import sys

from utils import data_analysis_04_non_academic_collab_helpers as _impl

sys.modules[__name__] = _impl
