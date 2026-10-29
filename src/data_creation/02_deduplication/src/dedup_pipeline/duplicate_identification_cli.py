"""Command-line interface for duplicate identification."""

import argparse

from .duplicate_identification import (
    IdentificationConfig,
    identify_duplicates,
    load_column_aliases,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Identify duplicate publication records and write merge-compatible evidence parquet files"
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="one or more parquet files, directories, or glob patterns",
    )
    parser.add_argument("--out-dir", required=True, help="output directory consumed by Stage A")
    parser.add_argument(
        "--engine",
        choices=("sqlite", "duckdb"),
        default="sqlite",
        help="sqlite preserves the full small-data workflow; duckdb is disk-backed and scalable for exact matching",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("exact", "minhash"),
        default=("exact", "minhash"),
    )
    parser.add_argument(
        "--column-map",
        help="optional JSON mapping canonical field names to source-column aliases",
    )
    parser.add_argument("--exact-min-fields", type=int, default=2)
    parser.add_argument("--max-exact-block-size", type=int, default=100)
    parser.add_argument("--jaccard-threshold", type=float, default=0.80)
    parser.add_argument("--num-perm", type=int, default=128)
    parser.add_argument("--lsh-bands", type=int, default=16)
    parser.add_argument("--max-lsh-bucket-size", type=int, default=200)
    parser.add_argument("--min-title-tokens", type=int, default=3)
    parser.add_argument("--min-abstract-tokens", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--output-pair-chunk-size", type=int, default=25_000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-work-db", action="store_true")
    parser.add_argument(
        "--resume-normalization",
        action="store_true",
        help="reuse completed normalized parquet shards in the scalable engine",
    )
    parser.add_argument("--duckdb-memory-limit", default="8GB")
    parser.add_argument("--duckdb-threads", type=int)
    return parser


def main():
    args = build_parser().parse_args()
    config = IdentificationConfig(
        input_specs=args.input,
        out_dir=args.out_dir,
        engine=args.engine,
        methods=args.methods,
        column_aliases=load_column_aliases(args.column_map),
        exact_min_fields=args.exact_min_fields,
        max_exact_block_size=args.max_exact_block_size,
        jaccard_threshold=args.jaccard_threshold,
        num_perm=args.num_perm,
        lsh_bands=args.lsh_bands,
        max_lsh_bucket_size=args.max_lsh_bucket_size,
        min_title_tokens=args.min_title_tokens,
        min_abstract_tokens=args.min_abstract_tokens,
        random_seed=args.random_seed,
        output_pair_chunk_size=args.output_pair_chunk_size,
        overwrite=args.overwrite,
        keep_work_db=args.keep_work_db,
        resume_normalization=args.resume_normalization,
        duckdb_memory_limit=args.duckdb_memory_limit,
        duckdb_threads=args.duckdb_threads,
    )
    summary = identify_duplicates(config)
    exact_pairs = summary.get("outputs", {}).get("exact", {}).get("pairs", 0)
    minhash_pairs = sum(
        value.get("pairs", 0)
        for value in summary.get("outputs", {}).get("minhash", {}).values()
    )
    exact_label = "exact pair evidence" if args.engine == "duckdb" else "exact pairs"
    print(
        f"loaded {summary['records_loaded']} records; "
        f"identified {exact_pairs} {exact_label} and {minhash_pairs} verified text pairs"
    )


if __name__ == "__main__":
    main()
