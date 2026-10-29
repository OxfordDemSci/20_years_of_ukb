"""Disk-backed exact duplicate identification for very large parquet collections."""

from __future__ import annotations

import itertools
import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import duckdb
import polars as pl

from .duplicate_identification import EXACT_FIELDS, resolve_columns


def _clean_string(expr: pl.Expr, lowercase: bool = True) -> pl.Expr:
    value = expr.cast(pl.Utf8, strict=False).str.strip_chars().str.replace_all(r"\s+", " ")
    if lowercase:
        value = value.str.to_lowercase()
    return pl.when(
        value.is_null()
        | value.str.to_lowercase().is_in(["", "none", "null", "nan", "na", "n/a", "[]", "{}"])
    ).then(None).otherwise(value)


def _source_text_expr(source: str | None, dtype: pl.DataType | None, variant: str) -> pl.Expr:
    if source is None or dtype is None:
        return pl.lit(None, dtype=pl.Utf8)
    if isinstance(dtype, pl.Struct):
        names = {field.name for field in dtype.fields}
        preferred = [variant, "preferred", "original", "text", "value"]
        choices = [pl.col(source).struct.field(name) for name in preferred if name in names]
        if not choices:
            return pl.lit(None, dtype=pl.Utf8)
        return _clean_string(pl.coalesce(choices))
    if isinstance(dtype, pl.List):
        return _clean_string(pl.col(source).list.first())
    return _clean_string(pl.col(source))


def _identifier_expr(field_name: str, source: str | None) -> pl.Expr:
    if source is None:
        return pl.lit(None, dtype=pl.Utf8)
    value = _clean_string(pl.col(source))
    if field_name == "doi":
        return value.str.replace(
            r"^(?:https?://(?:dx\.)?doi\.org/|doi\s*:\s*)", ""
        ).str.strip_chars(" .")
    if field_name == "pmid":
        stripped = value.str.replace(r"^(?:pmid|pubmed)\s*[:.]?\s*", "")
        return pl.coalesce([stripped.str.extract(r"(\d+)", 1), stripped])
    if field_name == "pmcid":
        return value.str.replace(r"^pmcid\s*[:.]?\s*", "").str.replace_all(" ", "")
    if field_name == "arxiv_id":
        return (
            value.str.replace(r"^(?:https?://arxiv\.org/(?:abs|pdf)/|arxiv\s*:\s*)", "")
            .str.replace(r"\.pdf$", "")
            .str.replace(r"v\d+$", "")
        )
    if field_name in {"isbn", "eisbn"}:
        return value.str.replace_all(r"[^0-9x]", "")
    if field_name == "altmetric_id":
        return pl.coalesce([value.str.extract(r"(\d+)", 1), value])
    return value


def _collection_hash_expr(
    direct_source: str | None,
    collection_source: str | None,
    schema: Mapping[str, pl.DataType],
    seed: int,
) -> pl.Expr:
    if direct_source:
        return _clean_string(pl.col(direct_source))
    if collection_source is None:
        return pl.lit(None, dtype=pl.Utf8)
    dtype = schema[collection_source]
    present = pl.col(collection_source).is_not_null()
    if isinstance(dtype, pl.List):
        present = present & (pl.col(collection_source).list.len() > 0)
    return pl.when(present).then(
        pl.col(collection_source).hash(seed=seed).cast(pl.Utf8)
    ).otherwise(None)


def normalise_shard(
    path: Path,
    output_path: Path,
    metadata_path: Path,
    aliases: Mapping[str, Sequence[str]],
    node_offset: int,
    random_seed: int,
) -> dict[str, Any]:
    lazy = pl.scan_parquet(path)
    schema = lazy.collect_schema()
    resolved = resolve_columns(schema.names(), aliases)
    if resolved.get("id") is None:
        raise ValueError(f"{path} has no publication id column")

    exact_sources = {
        resolved.get("id"),
        resolved.get("author_list_hash"),
        resolved.get("authors"),
        resolved.get("reference_list_hash"),
        resolved.get("references"),
        *(resolved.get(field_name) for field_name in EXACT_FIELDS),
    }
    selected_sources = sorted(source for source in exact_sources if source is not None)
    frame = pl.read_parquet(path, columns=selected_sources)
    rows_seen = frame.height
    frame = frame.with_row_index("__source_row", offset=node_offset)
    id_expr = _clean_string(pl.col(resolved["id"]), lowercase=False).alias("id")

    author_hash = _collection_hash_expr(
        resolved.get("author_list_hash"),
        resolved.get("authors"),
        schema,
        random_seed,
    ).alias("author_list_hash")
    reference_hash = _collection_hash_expr(
        resolved.get("reference_list_hash"),
        resolved.get("references"),
        schema,
        random_seed + 1,
    ).alias("reference_list_hash")

    expressions = [
        pl.col("__source_row").cast(pl.UInt64).alias("node_id"),
        id_expr,
        pl.lit(path.stem).alias("file"),
        author_hash,
        *[
            _identifier_expr(field_name, resolved.get(field_name)).alias(field_name)
            for field_name in EXACT_FIELDS
            if field_name not in {"author_list_hash", "reference_list_hash"}
        ],
        reference_hash,
    ]
    normalised = frame.select(expressions)
    rows_missing_id = normalised.filter(pl.col("id").is_null()).height
    normalised = normalised.filter(pl.col("id").is_not_null())

    duplicates = normalised.filter(pl.col("id").is_duplicated()).select("id").head(100)
    duplicate_count = normalised.select(pl.col("id").is_duplicated().sum()).item()
    if duplicate_count:
        raise ValueError(
            f"{path} contains repeated publication IDs; examples: {duplicates['id'].to_list()}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalised.write_parquet(output_path, compression="zstd", statistics=True)
    nonmissing = {
        column: int(normalised.select(pl.col(column).is_not_null().sum()).item())
        for column in EXACT_FIELDS
    }
    metadata = {
        "input": str(path),
        "input_size": path.stat().st_size,
        "input_mtime_ns": path.stat().st_mtime_ns,
        "output": str(output_path),
        "file": path.stem,
        "rows_seen": rows_seen,
        "records_loaded": normalised.height,
        "rows_missing_id": rows_missing_id,
        "node_offset": node_offset,
        "nonmissing_counts": nonmissing,
        "columns": schema.names(),
        "resolved": resolved,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata


def normalise_inputs(
    paths: Sequence[Path],
    work_dir: Path,
    aliases: Mapping[str, Sequence[str]],
    random_seed: int,
    resume: bool,
) -> tuple[Path, dict[str, Any]]:
    normalised_dir = work_dir / "normalised"
    metadata_dir = work_dir / "normalisation_metadata"
    normalised_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "input_files": [str(path) for path in paths],
        "rows_seen": 0,
        "records_loaded": 0,
        "rows_missing_id": 0,
        "duplicate_node_key_count": 0,
        "duplicate_node_key_examples": [],
        "schemas": {},
        "nonmissing_counts": {
            name: 0 for name in EXACT_FIELDS
        },
    }
    node_offset = 0
    for index, path in enumerate(paths, 1):
        output_path = normalised_dir / f"{path.stem}.parquet"
        metadata_path = metadata_dir / f"{path.stem}.json"
        metadata = None
        if resume and output_path.exists() and metadata_path.exists():
            candidate = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                candidate.get("input_size") == path.stat().st_size
                and candidate.get("input_mtime_ns") == path.stat().st_mtime_ns
            ):
                metadata = candidate
        if metadata is None:
            metadata = normalise_shard(
                path,
                output_path,
                metadata_path,
                aliases,
                node_offset,
                random_seed,
            )
        node_offset += int(metadata["rows_seen"])
        summary["rows_seen"] += int(metadata["rows_seen"])
        summary["records_loaded"] += int(metadata["records_loaded"])
        summary["rows_missing_id"] += int(metadata["rows_missing_id"])
        summary["schemas"][path.name] = {
            "columns": metadata["columns"],
            "resolved": metadata["resolved"],
            "missing_canonical_fields": sorted(
                key for key, value in metadata["resolved"].items() if value is None
            ),
        }
        for column, count in metadata["nonmissing_counts"].items():
            summary["nonmissing_counts"][column] += int(count)
        print(
            f"[normalise {index}/{len(paths)}] {path.name}: "
            f"{metadata['records_loaded']:,}/{metadata['rows_seen']:,} rows"
        )
    return normalised_dir, summary


def _sql_path(path: Path) -> str:
    return str(path).replace("'", "''")


def identify_exact_compound_groups(
    normalised_dir: Path,
    out_dir: Path,
    work_dir: Path,
    max_block_size: int,
    memory_limit: str,
    threads: int | None,
    nonmissing_counts: Mapping[str, int],
) -> dict[str, Any]:
    database_path = work_dir / "exact_matching.duckdb"
    temp_dir = work_dir / "duckdb_temp"
    temp_dir.mkdir(exist_ok=True)
    connection = duckdb.connect(str(database_path))
    try:
        connection.execute(f"SET memory_limit='{memory_limit}'")
        connection.execute(f"SET temp_directory='{_sql_path(temp_dir)}'")
        connection.execute("SET preserve_insertion_order=false")
        if threads:
            connection.execute(f"SET threads={int(threads)}")
        parquet_glob = _sql_path(normalised_dir / "*.parquet")
        connection.execute(
            "CREATE OR REPLACE VIEW records AS "
            f"SELECT * FROM read_parquet('{parquet_glob}', union_by_name=true)"
        )

        total_groups = 0
        total_pairs = 0
        output_files = 0
        by_field_pair = {}
        active_fields = [field for field in EXACT_FIELDS if nonmissing_counts.get(field, 0) >= 2]
        combinations = list(itertools.combinations(active_fields, 2))
        for left_field, right_field in combinations:
            output_path = out_dir / f"exact_{left_field}_{right_field}.parquet"
            provenance = json.dumps(
                {
                    "method": "exact_compound_group",
                    "matched_fields": [left_field, right_field],
                    "n_fields": 2,
                },
                sort_keys=True,
            ).replace("'", "''")
            group_query = f"""
                SELECT "{left_field}", "{right_field}", COUNT(*) AS n
                FROM records
                WHERE "{left_field}" IS NOT NULL AND "{right_field}" IS NOT NULL
                GROUP BY "{left_field}", "{right_field}"
                HAVING COUNT(*) BETWEEN 2 AND {int(max_block_size)}
            """
            group_count, pair_count = connection.execute(
                f"SELECT COUNT(*), COALESCE(SUM(n * (n - 1) / 2), 0) FROM ({group_query})"
            ).fetchone()
            group_count = int(group_count)
            pair_count = int(pair_count)
            total_groups += group_count
            total_pairs += pair_count
            by_field_pair[f"{left_field}+{right_field}"] = {
                "groups": group_count,
                "implied_pairs": pair_count,
            }

            copy_query = f"""
                COPY (
                    WITH eligible AS ({group_query})
                    SELECT r.id, r.file,
                           'exact:{left_field}:{right_field}:' ||
                           md5(r."{left_field}" || chr(31) || r."{right_field}") AS match_group,
                           '{provenance}' AS columns
                    FROM records r
                    INNER JOIN eligible e
                      ON r."{left_field}" = e."{left_field}"
                     AND r."{right_field}" = e."{right_field}"
                ) TO '{_sql_path(output_path)}'
                (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
            """
            connection.execute(copy_query)
            output_files += 1
            print(
                f"[exact {output_files}/{len(combinations)}] {left_field}+{right_field}: "
                f"groups={group_count:,}, implied_pairs={pair_count:,}"
            )
        if not combinations:
            pl.DataFrame(
                schema={
                    "id": pl.Utf8,
                    "file": pl.Utf8,
                    "match_group": pl.Utf8,
                    "columns": pl.Utf8,
                }
            ).write_parquet(out_dir / "exact_no_eligible_fields.parquet", compression="zstd")
            output_files = 1
        return {
            "pairs": total_pairs,
            "groups": total_groups,
            "pair_evidence": total_pairs,
            "group_evidence": total_groups,
            "output_files": output_files,
            "max_block_size": max_block_size,
            "field_pairs": by_field_pair,
            "engine": "duckdb_compound_groups",
            "active_fields": active_fields,
        }
    finally:
        connection.close()


def identify_duplicates_scalable(config, paths, out_dir, aliases):
    if set(config.methods) != {"exact"}:
        raise ValueError(
            "the scalable engine currently supports --methods exact. Global MinHash over "
            "hundreds of millions of title/abstract records must be run as a separately "
            "blocked or distributed job; use the sqlite engine only for a reduced subset"
        )
    if config.exact_min_fields != 2:
        raise ValueError("the scalable compound-group engine requires --exact-min-fields 2")

    work_dir = out_dir / "_work_scalable"
    if work_dir.exists() and not config.resume_normalization:
        shutil.rmtree(work_dir)
    work_dir.mkdir(exist_ok=True)
    normalised_dir, summary = normalise_inputs(
        paths,
        work_dir,
        aliases,
        config.random_seed,
        config.resume_normalization,
    )
    summary["parameters"] = {
        "engine": "duckdb",
        "methods": list(config.methods),
        "exact_min_fields": config.exact_min_fields,
        "max_exact_block_size": config.max_exact_block_size,
        "duckdb_memory_limit": config.duckdb_memory_limit,
        "duckdb_threads": config.duckdb_threads,
        "resume_normalization": config.resume_normalization,
    }
    summary["outputs"] = {
        "exact": identify_exact_compound_groups(
            normalised_dir,
            out_dir,
            work_dir,
            config.max_exact_block_size,
            config.duckdb_memory_limit,
            config.duckdb_threads,
            summary["nonmissing_counts"],
        )
    }
    diagnostics_dir = out_dir / "diagnostics"
    diagnostics_dir.mkdir(exist_ok=True)
    (diagnostics_dir / "duplicate_identification_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    if not config.keep_work_db:
        shutil.rmtree(work_dir)
    return summary
