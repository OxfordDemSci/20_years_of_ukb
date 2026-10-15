#!/usr/bin/env python3
"""Stage 08 — classify every paper's collaborating organisations (companies and sectors).

WHY
---
Analysis 04's collaboration arm needs each publication's `research_orgs` labelled —
academic, UK company, non-UK company, and the other sector indices — by the Claude
classifier in `src/utils/data_analysis_04_non_academic_collab_classifier.py` (D16, D17,
D26). The notebook only ever READS that table (D15: the tagging step is skipped when its
output exists); this stage is what writes it.

It is the one paid stage of the data-creation run, so it is opt-in twice over: the runner
passes `--allow-api` only under `--allow-classification`, and the runner schedules it LAST,
after every other stage has succeeded, so a failure elsewhere never costs a
classification pass. Distinct institution sets are classified once and appended to
`P.COLLAB_CACHE` per batch (fsynced), so a killed run resumes where it stopped and a set
already in the cache is free. A pass that leaves any set unclassified raises and writes no
table (D25).

The corpus is read with exactly the columns `04_non_academic.ipynb` reads for this step,
and `load_or_classify` applies the shared 2013–2025 window, so the table is the one the
notebook would have written.

USAGE
-----
    python src/data_creation/08_collaborator_tagging/collaborator_tagging.py            # cache only
    python src/data_creation/08_collaborator_tagging/collaborator_tagging.py --allow-api
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils import shared_paths as P  # noqa: E402
from utils.data_creation_runtime import install_keyed_api_guard  # noqa: E402

#: The columns 04_non_academic.ipynb loads before classifying (its §1). The written table is
#: what every later section of 04 reads, so it must carry all of them.
CORPUS_COLUMNS = ["id", "title", "year", "date", "times_cited", "altmetric",
                  "category_for_2020", "research_orgs", "authors",
                  "journal_title_raw", "research_org_countries"]
PARSED_COLUMNS = ["research_orgs", "authors", "category_for_2020"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--corpus", type=Path, default=P.SHOWCASE_PLUS)
    parser.add_argument("--out", type=Path, default=P.COLLAB_FLAGGED)
    parser.add_argument("--cache", type=Path, default=P.COLLAB_CACHE)
    parser.add_argument("--allow-api", action="store_true",
                        default=os.environ.get("UKB_ALLOW_CLASSIFICATION", "").lower()
                        in {"1", "true", "yes"},
                        help="classify uncached institution sets with Claude (paid)")
    args = parser.parse_args(argv)
    install_keyed_api_guard()

    from utils import data_analysis_04_non_academic_collab_classifier as CLS
    from utils.shared_showcase import load_showcase

    corpus = load_showcase(columns=CORPUS_COLUMNS, parse=PARSED_COLUMNS, path=args.corpus)
    print(f"corpus: {len(corpus):,} publications from {P.raw_path(args.corpus)}")
    print(f"model {CLS.DEFAULT_MODEL}, effort {CLS.DEFAULT_EFFORT}, "
          f"API {'allowed' if args.allow_api else 'NOT allowed (cache only)'}")
    _, ran = CLS.load_or_classify(
        corpus,
        out_path=args.out,
        cache_path=args.cache,
        force=True,                 # the runner archived the old table; always rebuild
        allow_api=args.allow_api,
        model=CLS.DEFAULT_MODEL,
        effort=CLS.DEFAULT_EFFORT,
    )
    print(f"classified table written: {ran}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
