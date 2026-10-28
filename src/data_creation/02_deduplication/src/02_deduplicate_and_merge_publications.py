#!/usr/bin/env python3
"""Thin direct-run entry point.

Run from the folder root, for example:

    python src/02_deduplicate_and_merge_publications.py \
        --step5-dir ... --raw-base-dir ... --out-dir ...
"""

from dedup_pipeline.deduplication_cli import main


if __name__ == "__main__":
    main()
