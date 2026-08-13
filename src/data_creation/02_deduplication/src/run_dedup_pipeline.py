#!/usr/bin/env python3
"""Thin direct-run entry point.

Run from the folder root, for example:

    python src/run_dedup_pipeline.py --step5-dir ... --raw-base-dir ... --out-dir ...
"""

from dedup_pipeline.cli import main


if __name__ == "__main__":
    main()
