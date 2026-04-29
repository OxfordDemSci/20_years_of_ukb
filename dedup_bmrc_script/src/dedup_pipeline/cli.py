"""Command-line interface for the refactored deduplication pipeline."""

import argparse

from .pipeline import run_pipeline


def build_parser():
    parser = argparse.ArgumentParser(description="Stages A–E: clusters, routing, scoring, patches, apply")
    parser.add_argument("--step5-dir", required=True, help="folder with step-5 combo parquet files")
    parser.add_argument("--raw-base-dir", required=True, help="folder with raw shards, e.g. 000000000457.parquet files")
    parser.add_argument("--out-dir", required=True, help="base folder to write pipeline outputs")
    parser.add_argument("--min-group-size", type=int, default=2, help="minimum group size to consider duplicates")
    parser.add_argument("--ext", default=".parquet", help="raw shard filename extension, default .parquet")
    parser.add_argument("--no-exists-check", action="store_true", help="skip checking raw shard files on disk during Stage B")
    parser.add_argument("--zero-id-weight", action="store_true", help="if set, id contributes 0 to the informativeness score")
    parser.add_argument("--list-cap", type=int, default=200, help="cap for unioned list-like fields in patches; 0 = no cap")
    parser.add_argument("--apply-only-file", default=None, help="Stage E: process only this shard stem, e.g. 000000000457")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
