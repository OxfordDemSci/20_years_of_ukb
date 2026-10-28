"""Identify duplicate publication occurrences before the merge pipeline."""

from __future__ import annotations

import ast
import glob
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import polars as pl


EXACT_FIELDS = (
    "author_list_hash",
    "doi",
    "pmid",
    "pmcid",
    "arxiv_id",
    "altmetric_id",
    "isbn",
    "eisbn",
    "reference_list_hash",
)

TEXT_FIELDS = (
    "original_title",
    "preferred_title",
    "preferred_abstract",
)

DEFAULT_COLUMN_ALIASES = {
    "id": ("id", "publication_id", "dimensions_id", "pub_id"),
    "author_list_hash": ("author_list_hash", "authors_hash", "author_hash"),
    "authors": ("authors", "author_list", "authors_list", "author_names"),
    "doi": ("doi", "digital_object_identifier"),
    "pmid": ("pmid", "pubmed_id", "pubmed"),
    "pmcid": ("pmcid", "pubmed_central_id"),
    "arxiv_id": ("arxiv_id", "arxiv", "arxiv_identifier"),
    "altmetric_id": ("altmetric_id", "altmetrics_id", "altmetric"),
    "isbn": ("isbn", "print_isbn"),
    "eisbn": ("eisbn", "electronic_isbn", "ebook_isbn"),
    "reference_list_hash": ("reference_list_hash", "references_hash", "reference_hash"),
    "references": ("reference_ids", "references", "reference_id", "cited_reference_ids"),
    "original_title": ("original_title", "title_original"),
    "preferred_title": ("preferred_title", "title_preferred", "title"),
    "preferred_abstract": ("preferred_abstract", "abstract_preferred", "abstract"),
}

MISSING_TEXT = {"", "none", "null", "nan", "na", "n/a", "[]", "{}"}
TOKEN_RE = re.compile(r"[a-z0-9]+")
GLOB_CHARS = set("*?[")
MINHASH_PRIME = np.uint64(2_147_483_647)


@dataclass
class IdentificationConfig:
    input_specs: Sequence[str]
    out_dir: str
    engine: str = "sqlite"
    methods: Sequence[str] = ("exact", "minhash")
    column_aliases: Mapping[str, Sequence[str]] = field(default_factory=dict)
    exact_min_fields: int = 2
    max_exact_block_size: int = 100
    jaccard_threshold: float = 0.80
    num_perm: int = 128
    lsh_bands: int = 16
    max_lsh_bucket_size: int = 200
    min_title_tokens: int = 3
    min_abstract_tokens: int = 20
    random_seed: int = 42
    output_pair_chunk_size: int = 25_000
    overwrite: bool = False
    keep_work_db: bool = False
    resume_normalization: bool = False
    duckdb_memory_limit: str = "8GB"
    duckdb_threads: int | None = None


def merged_aliases(overrides: Mapping[str, Sequence[str]] | None = None) -> dict[str, tuple[str, ...]]:
    aliases = {key: tuple(values) for key, values in DEFAULT_COLUMN_ALIASES.items()}
    for key, values in (overrides or {}).items():
        if isinstance(values, str):
            values = [values]
        aliases[key] = tuple(dict.fromkeys([*values, *aliases.get(key, ())]))
    return aliases


def load_column_aliases(path: str | None) -> dict[str, Sequence[str]]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("column-map JSON must be an object mapping canonical names to aliases")
    return value


def expand_input_paths(input_specs: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    for spec in input_specs:
        expanded = os.path.expanduser(spec)
        candidate = Path(expanded)
        if any(char in expanded for char in GLOB_CHARS):
            paths.extend(Path(path) for path in glob.glob(expanded, recursive=True))
        elif candidate.is_dir():
            paths.extend(candidate.glob("*.parquet"))
        elif candidate.is_file():
            paths.append(candidate)
        else:
            raise FileNotFoundError(f"input does not exist: {spec}")

    parquet_paths = sorted({path.resolve() for path in paths if path.suffix.lower() == ".parquet"})
    if not parquet_paths:
        raise FileNotFoundError("no parquet inputs were found")

    stems: dict[str, Path] = {}
    for path in parquet_paths:
        if path.stem in stems:
            raise ValueError(
                "input parquet stems must be unique because the merge stage uses the stem as "
                f"the source shard key: {stems[path.stem]} and {path}"
            )
        stems[path.stem] = path
    return parquet_paths


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str):
        return value.strip().lower() in MISSING_TEXT
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False


def parse_structured_string(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if len(text) < 2 or (text[0], text[-1]) not in {("[", "]"), ("{", "}")}:
        return value
    try:
        return json.loads(text)
    except Exception:
        try:
            return ast.literal_eval(text)
        except Exception:
            return value


def normalise_space(value: Any, lowercase: bool = True) -> str | None:
    if is_missing(value):
        return None
    text = unicodedata.normalize("NFKC", str(value)).strip()
    text = re.sub(r"\s+", " ", text)
    if lowercase:
        text = text.casefold()
    return text or None


def canonical_json_value(value: Any) -> Any:
    value = parse_structured_string(value)
    if is_missing(value):
        return None
    if isinstance(value, dict):
        result = {}
        for key in sorted(value, key=lambda item: str(item)):
            normalised = canonical_json_value(value[key])
            if normalised is not None:
                result[normalise_space(key) or str(key)] = normalised
        return result or None
    if isinstance(value, (list, tuple)):
        result = [canonical_json_value(item) for item in value]
        return [item for item in result if item is not None] or None
    return normalise_space(value)


def hash_collection(value: Any, preserve_order: bool) -> str | None:
    canonical = canonical_json_value(value)
    if canonical is None:
        return None
    if not preserve_order and isinstance(canonical, list):
        canonical = sorted(
            {json.dumps(item, sort_keys=True, ensure_ascii=False) for item in canonical}
        )
    payload = json.dumps(canonical, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalise_identifier(field_name: str, value: Any) -> str | None:
    value = parse_structured_string(value)
    if isinstance(value, dict):
        for key in ("id", "value", "preferred"):
            if key in value and not is_missing(value[key]):
                value = value[key]
                break
    if isinstance(value, (list, tuple)):
        value = next((item for item in value if not is_missing(item)), None)
    text = normalise_space(value)
    if text is None:
        return None

    if field_name == "doi":
        text = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi\s*:\s*)", "", text)
        return text.rstrip(" .") or None
    if field_name == "pmid":
        text = re.sub(r"^(?:pmid|pubmed)\s*[:.]?\s*", "", text)
        match = re.search(r"\d+", text)
        return match.group(0) if match else text
    if field_name == "pmcid":
        text = re.sub(r"^pmcid\s*[:.]?\s*", "", text)
        return text.replace(" ", "")
    if field_name == "arxiv_id":
        text = re.sub(r"^(?:https?://arxiv\.org/(?:abs|pdf)/|arxiv\s*:\s*)", "", text)
        text = re.sub(r"\.pdf$", "", text)
        return re.sub(r"v\d+$", "", text)
    if field_name in {"isbn", "eisbn"}:
        value = re.sub(r"[^0-9x]", "", text)
        return value or None
    if field_name == "altmetric_id":
        match = re.search(r"\d+", text)
        return match.group(0) if match else text
    return text


def text_variant(value: Any, variant: str) -> str | None:
    value = parse_structured_string(value)
    if isinstance(value, dict):
        keys = (variant, "preferred", "original", "text", "value")
        value = next((value[key] for key in keys if key in value and not is_missing(value[key])), None)
    elif isinstance(value, (list, tuple)):
        value = next((item for item in value if not is_missing(item)), None)
    return normalise_space(value)


def tokenise(value: str | None) -> frozenset[str]:
    if not value:
        return frozenset()
    return frozenset(TOKEN_RE.findall(unicodedata.normalize("NFKC", value).casefold()))


def resolve_columns(schema_names: Sequence[str], aliases: Mapping[str, Sequence[str]]) -> dict[str, str | None]:
    lookup = {name.casefold(): name for name in schema_names}
    return {
        canonical: next((lookup[a.casefold()] for a in candidates if a.casefold() in lookup), None)
        for canonical, candidates in aliases.items()
    }


def row_value(row: Mapping[str, Any], resolved: Mapping[str, str | None], canonical: str) -> Any:
    source = resolved.get(canonical)
    return row.get(source) if source else None


def choose_hash(
    row: Mapping[str, Any],
    resolved: Mapping[str, str | None],
    hash_field: str,
    raw_field: str,
    preserve_order: bool,
) -> str | None:
    direct = normalise_space(row_value(row, resolved, hash_field))
    if direct:
        return direct
    return hash_collection(row_value(row, resolved, raw_field), preserve_order=preserve_order)


def create_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=FILE")
    columns = [*(f'"{name}" TEXT' for name in EXACT_FIELDS), *(f'"{name}" TEXT' for name in TEXT_FIELDS)]
    connection.execute(
        "CREATE TABLE records ("
        "node_id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "id TEXT NOT NULL, file TEXT NOT NULL, "
        + ", ".join(columns)
        + ", UNIQUE(file, id))"
    )
    return connection


def ingest_inputs(
    connection: sqlite3.Connection,
    paths: Sequence[Path],
    aliases: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "input_files": [str(path) for path in paths],
        "rows_seen": 0,
        "records_loaded": 0,
        "rows_missing_id": 0,
        "duplicate_node_key_count": 0,
        "duplicate_node_key_examples": [],
        "schemas": {},
        "nonmissing_counts": {name: 0 for name in (*EXACT_FIELDS, *TEXT_FIELDS)},
    }
    insert_columns = ["id", "file", *EXACT_FIELDS, *TEXT_FIELDS]
    placeholders = ",".join("?" for _ in insert_columns)
    quoted_columns = ",".join(f'"{name}"' for name in insert_columns)
    insert_sql = f"INSERT INTO records ({quoted_columns}) VALUES ({placeholders})"

    for path in paths:
        schema_names = pl.scan_parquet(path).collect_schema().names()
        resolved = resolve_columns(schema_names, aliases)
        if resolved.get("id") is None:
            raise ValueError(
                f"{path} has no publication id column; accepted aliases are {aliases['id']}"
            )
        selected = sorted({name for name in resolved.values() if name is not None})
        summary["schemas"][path.name] = {
            "columns": schema_names,
            "resolved": resolved,
            "missing_canonical_fields": sorted(key for key, value in resolved.items() if value is None),
        }

        frame = pl.read_parquet(path, columns=selected)
        rows_to_insert = []
        for row in frame.iter_rows(named=True):
            summary["rows_seen"] += 1
            publication_id = normalise_space(row_value(row, resolved, "id"), lowercase=False)
            if not publication_id:
                summary["rows_missing_id"] += 1
                continue

            author_hash = choose_hash(
                row, resolved, "author_list_hash", "authors", preserve_order=True
            )
            reference_hash = choose_hash(
                row, resolved, "reference_list_hash", "references", preserve_order=False
            )
            exact_values = {
                "author_list_hash": author_hash,
                "doi": normalise_identifier("doi", row_value(row, resolved, "doi")),
                "pmid": normalise_identifier("pmid", row_value(row, resolved, "pmid")),
                "pmcid": normalise_identifier("pmcid", row_value(row, resolved, "pmcid")),
                "arxiv_id": normalise_identifier("arxiv_id", row_value(row, resolved, "arxiv_id")),
                "altmetric_id": normalise_identifier(
                    "altmetric_id", row_value(row, resolved, "altmetric_id")
                ),
                "isbn": normalise_identifier("isbn", row_value(row, resolved, "isbn")),
                "eisbn": normalise_identifier("eisbn", row_value(row, resolved, "eisbn")),
                "reference_list_hash": reference_hash,
            }

            raw_title = row_value(row, resolved, "preferred_title")
            raw_original_title = row_value(row, resolved, "original_title")
            raw_abstract = row_value(row, resolved, "preferred_abstract")
            text_values = {
                "original_title": text_variant(
                    raw_original_title if not is_missing(raw_original_title) else raw_title,
                    "original",
                ),
                "preferred_title": text_variant(raw_title, "preferred"),
                "preferred_abstract": text_variant(raw_abstract, "preferred"),
            }

            values = {
                "id": publication_id,
                "file": path.stem,
                **exact_values,
                **text_values,
            }
            for name in (*EXACT_FIELDS, *TEXT_FIELDS):
                if values[name] is not None:
                    summary["nonmissing_counts"][name] += 1
            rows_to_insert.append(tuple(values[name] for name in insert_columns))

        try:
            connection.executemany(insert_sql, rows_to_insert)
        except sqlite3.IntegrityError:
            connection.rollback()
            for values in rows_to_insert:
                try:
                    connection.execute(insert_sql, values)
                except sqlite3.IntegrityError:
                    summary["duplicate_node_key_count"] += 1
                    if len(summary["duplicate_node_key_examples"]) < 100:
                        summary["duplicate_node_key_examples"].append(
                            {"file": values[1], "id": values[0]}
                        )
        connection.commit()
        summary["records_loaded"] = connection.execute("SELECT COUNT(*) FROM records").fetchone()[0]

    return summary


def initialise_exact_values(connection: sqlite3.Connection) -> None:
    connection.execute(
        "CREATE TABLE exact_values (node_id INTEGER NOT NULL, field TEXT NOT NULL, value TEXT NOT NULL)"
    )
    for field_name in EXACT_FIELDS:
        connection.execute(
            f'INSERT INTO exact_values SELECT node_id, ?, "{field_name}" FROM records '
            f'WHERE "{field_name}" IS NOT NULL AND "{field_name}" != ""',
            (field_name,),
        )
    connection.execute("CREATE INDEX exact_values_lookup ON exact_values(field, value, node_id)")
    connection.commit()


def empty_match_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={"id": pl.Utf8, "file": pl.Utf8, "match_group": pl.Utf8, "columns": pl.Utf8}
    )


def clean_existing_outputs(out_dir: Path) -> None:
    for path in out_dir.glob("*.parquet"):
        path.unlink()


def write_match_chunk(rows: list[dict[str, str]], path: Path) -> None:
    frame = pl.DataFrame(rows) if rows else empty_match_frame()
    frame.select(["id", "file", "match_group", "columns"]).write_parquet(
        path, compression="zstd"
    )


def identify_exact_matches(
    connection: sqlite3.Connection,
    out_dir: Path,
    min_fields: int,
    max_block_size: int,
    pair_chunk_size: int,
) -> dict[str, Any]:
    initialise_exact_values(connection)
    skipped = connection.execute(
        "SELECT field, COUNT(*) FROM ("
        "SELECT field, value, COUNT(*) AS n FROM exact_values GROUP BY field, value"
        ") WHERE n > ? GROUP BY field ORDER BY field",
        (max_block_size,),
    ).fetchall()
    connection.execute(
        "CREATE TEMP TABLE eligible_exact_values AS "
        "SELECT field, value FROM exact_values GROUP BY field, value "
        "HAVING COUNT(*) BETWEEN 2 AND ?",
        (max_block_size,),
    )
    connection.execute(
        "CREATE INDEX eligible_exact_lookup ON eligible_exact_values(field, value)"
    )

    query = """
        WITH pair_hits AS (
            SELECT a.node_id AS left_node, b.node_id AS right_node, a.field AS field
            FROM exact_values a
            JOIN eligible_exact_values e ON e.field = a.field AND e.value = a.value
            JOIN exact_values b
              ON b.field = a.field AND b.value = a.value AND b.node_id > a.node_id
        ), grouped AS (
            SELECT left_node, right_node, COUNT(DISTINCT field) AS n_fields,
                   GROUP_CONCAT(DISTINCT field) AS fields
            FROM pair_hits
            GROUP BY left_node, right_node
            HAVING COUNT(DISTINCT field) >= ?
        )
        SELECT grouped.left_node, grouped.right_node, grouped.n_fields, grouped.fields,
               left_record.id, left_record.file, right_record.id, right_record.file
        FROM grouped
        JOIN records left_record ON left_record.node_id = grouped.left_node
        JOIN records right_record ON right_record.node_id = grouped.right_node
        ORDER BY grouped.left_node, grouped.right_node
    """
    cursor = connection.execute(query, (min_fields,))
    chunk_index = 0
    pair_index = 0
    while True:
        batch = cursor.fetchmany(pair_chunk_size)
        if not batch:
            break
        output_rows: list[dict[str, str]] = []
        for left_node, right_node, n_fields, fields, left_id, left_file, right_id, right_file in batch:
            pair_index += 1
            matched_fields = sorted(set((fields or "").split(",")))
            group = f"exact:{pair_index:012d}"
            provenance = json.dumps(
                {
                    "method": "exact_two_field",
                    "matched_fields": matched_fields,
                    "n_fields": n_fields,
                    "nodes": [left_node, right_node],
                },
                sort_keys=True,
            )
            output_rows.extend(
                [
                    {"id": left_id, "file": left_file, "match_group": group, "columns": provenance},
                    {"id": right_id, "file": right_file, "match_group": group, "columns": provenance},
                ]
            )
        path = out_dir / f"exact_two_field_matches_{chunk_index:04d}.parquet"
        write_match_chunk(output_rows, path)
        chunk_index += 1

    if pair_index == 0:
        write_match_chunk([], out_dir / "exact_two_field_matches_0000.parquet")
        chunk_index = 1
    return {
        "pairs": pair_index,
        "output_files": chunk_index,
        "max_block_size": max_block_size,
        "skipped_oversized_blocks_by_field": dict(skipped),
    }


class MinHasher:
    def __init__(self, num_perm: int, seed: int):
        self.num_perm = num_perm
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, MINHASH_PRIME, size=num_perm, dtype=np.uint64)
        self.b = rng.integers(0, MINHASH_PRIME, size=num_perm, dtype=np.uint64)

    def signature(self, tokens: frozenset[str]) -> tuple[int, ...]:
        token_hashes = np.fromiter(
            (
                int.from_bytes(
                    hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "little"
                )
                % int(MINHASH_PRIME)
                for token in tokens
            ),
            dtype=np.uint64,
        )
        values = (self.a[:, None] * token_hashes[None, :] + self.b[:, None]) % MINHASH_PRIME
        return tuple(int(value) for value in values.min(axis=1))


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union_size = len(left | right)
    return len(left & right) / union_size if union_size else 0.0


def flush_minhash_rows(
    rows: list[dict[str, str]], out_dir: Path, field_name: str, chunk_index: int
) -> int:
    path = out_dir / f"minhash_{field_name}_matches_{chunk_index:04d}.parquet"
    write_match_chunk(rows, path)
    rows.clear()
    return chunk_index + 1


def identify_minhash_field(
    connection: sqlite3.Connection,
    out_dir: Path,
    field_name: str,
    threshold: float,
    num_perm: int,
    bands: int,
    max_bucket_size: int,
    min_tokens: int,
    random_seed: int,
    pair_chunk_size: int,
) -> dict[str, Any]:
    if num_perm % bands != 0:
        raise ValueError("num_perm must be divisible by lsh_bands")
    rows_per_band = num_perm // bands
    minhasher = MinHasher(num_perm=num_perm, seed=random_seed)
    buckets: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
    blocked_buckets: set[tuple[int, tuple[int, ...]]] = set()
    output_rows: list[dict[str, str]] = []
    seen_pairs: set[tuple[int, int]] = set()
    pair_index = 0
    chunk_index = 0
    documents_indexed = 0
    documents_skipped_short = 0

    @lru_cache(maxsize=10_000)
    def load_indexed_node(node_id: int) -> tuple[str, str, frozenset[str]]:
        record = connection.execute(
            f'SELECT id, file, "{field_name}" FROM records WHERE node_id = ?',
            (node_id,),
        ).fetchone()
        if record is None:
            raise RuntimeError(f"indexed node {node_id} is missing from the normalised store")
        other_id, other_file, other_text = record
        return other_id, other_file, tokenise(other_text)

    cursor = connection.execute(
        f'SELECT node_id, id, file, "{field_name}" FROM records '
        f'WHERE "{field_name}" IS NOT NULL AND "{field_name}" != "" ORDER BY node_id'
    )
    for node_id, publication_id, source_file, text in cursor:
        tokens = tokenise(text)
        if len(tokens) < min_tokens:
            documents_skipped_short += 1
            continue
        signature = minhasher.signature(tokens)
        band_keys = [
            (band, signature[band * rows_per_band : (band + 1) * rows_per_band])
            for band in range(bands)
        ]
        candidates: set[int] = set()
        for key in band_keys:
            if key not in blocked_buckets:
                candidates.update(buckets.get(key, ()))

        for other_node in sorted(candidates):
            pair = (other_node, node_id)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            other_id, other_file, other_tokens = load_indexed_node(other_node)
            similarity = jaccard(other_tokens, tokens)
            if similarity < threshold:
                continue
            pair_index += 1
            group = f"minhash:{field_name}:{pair_index:012d}"
            provenance = json.dumps(
                {
                    "method": "minhash_lsh_verified_jaccard",
                    "field": field_name,
                    "jaccard": round(similarity, 6),
                    "threshold": threshold,
                    "nodes": [other_node, node_id],
                },
                sort_keys=True,
            )
            output_rows.extend(
                [
                    {"id": other_id, "file": other_file, "match_group": group, "columns": provenance},
                    {"id": publication_id, "file": source_file, "match_group": group, "columns": provenance},
                ]
            )
            if pair_index % pair_chunk_size == 0:
                chunk_index = flush_minhash_rows(
                    output_rows, out_dir, field_name, chunk_index
                )

        documents_indexed += 1
        for key in band_keys:
            if key in blocked_buckets:
                continue
            bucket = buckets[key]
            bucket.append(node_id)
            if len(bucket) > max_bucket_size:
                blocked_buckets.add(key)
                del buckets[key]

    if output_rows:
        chunk_index = flush_minhash_rows(output_rows, out_dir, field_name, chunk_index)
    if pair_index == 0:
        write_match_chunk([], out_dir / f"minhash_{field_name}_matches_0000.parquet")
        chunk_index = 1
    return {
        "pairs": pair_index,
        "output_files": chunk_index,
        "documents_indexed": documents_indexed,
        "documents_skipped_below_min_tokens": documents_skipped_short,
        "blocked_lsh_buckets": len(blocked_buckets),
        "min_tokens": min_tokens,
    }


def validate_config(config: IdentificationConfig) -> None:
    if config.engine not in {"sqlite", "duckdb"}:
        raise ValueError("engine must be 'sqlite' or 'duckdb'")
    methods = set(config.methods)
    unknown = methods - {"exact", "minhash"}
    if unknown:
        raise ValueError(f"unknown identification methods: {sorted(unknown)}")
    if config.exact_min_fields < 2:
        raise ValueError("exact_min_fields must be at least 2")
    if not 0 < config.jaccard_threshold <= 1:
        raise ValueError("jaccard_threshold must be in (0, 1]")
    if config.num_perm <= 0 or config.lsh_bands <= 0:
        raise ValueError("num_perm and lsh_bands must be positive")
    if config.num_perm % config.lsh_bands:
        raise ValueError("num_perm must be divisible by lsh_bands")


def prepare_output_directory(out_dir: Path, overwrite: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = list(out_dir.glob("*.parquet"))
    if existing and not overwrite:
        raise FileExistsError(
            f"{out_dir} already contains parquet outputs; use --overwrite to replace matching outputs"
        )
    if overwrite:
        clean_existing_outputs(out_dir)


def identify_duplicates(config: IdentificationConfig) -> dict[str, Any]:
    validate_config(config)
    paths = expand_input_paths(config.input_specs)
    out_dir = Path(config.out_dir).expanduser().resolve()
    prepare_output_directory(out_dir, overwrite=config.overwrite)
    diagnostics_dir = out_dir / "diagnostics"
    aliases = merged_aliases(config.column_aliases)
    if config.engine == "duckdb":
        from .scalable_duplicate_identification import identify_duplicates_scalable

        return identify_duplicates_scalable(config, paths, out_dir, aliases)

    work_dir = out_dir / "_work"
    diagnostics_dir.mkdir(exist_ok=True)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir()
    database_path = work_dir / "normalised_records.sqlite3"
    connection = create_database(database_path)
    try:
        summary = ingest_inputs(connection, paths, aliases)
        summary["parameters"] = {
            "methods": list(config.methods),
            "exact_min_fields": config.exact_min_fields,
            "max_exact_block_size": config.max_exact_block_size,
            "jaccard_threshold": config.jaccard_threshold,
            "num_perm": config.num_perm,
            "lsh_bands": config.lsh_bands,
            "max_lsh_bucket_size": config.max_lsh_bucket_size,
            "random_seed": config.random_seed,
        }
        summary["outputs"] = {}

        if summary["duplicate_node_key_count"]:
            summary_path = diagnostics_dir / "duplicate_identification_summary.json"
            with summary_path.open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2, sort_keys=True)
            raise ValueError(
                "the merge pipeline cannot safely distinguish repeated (file, id) occurrences; "
                f"found {summary['duplicate_node_key_count']}, including "
                f"{summary['duplicate_node_key_examples'][:5]}"
            )

        if "exact" in config.methods:
            summary["outputs"]["exact"] = identify_exact_matches(
                connection=connection,
                out_dir=out_dir,
                min_fields=config.exact_min_fields,
                max_block_size=config.max_exact_block_size,
                pair_chunk_size=config.output_pair_chunk_size,
            )

        if "minhash" in config.methods:
            minhash_summary = {}
            for field_name in TEXT_FIELDS:
                min_tokens = (
                    config.min_abstract_tokens
                    if field_name == "preferred_abstract"
                    else config.min_title_tokens
                )
                minhash_summary[field_name] = identify_minhash_field(
                    connection=connection,
                    out_dir=out_dir,
                    field_name=field_name,
                    threshold=config.jaccard_threshold,
                    num_perm=config.num_perm,
                    bands=config.lsh_bands,
                    max_bucket_size=config.max_lsh_bucket_size,
                    min_tokens=min_tokens,
                    random_seed=config.random_seed,
                    pair_chunk_size=config.output_pair_chunk_size,
                )
            summary["outputs"]["minhash"] = minhash_summary

        summary_path = diagnostics_dir / "duplicate_identification_summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
    finally:
        connection.close()

    if not config.keep_work_db:
        shutil.rmtree(work_dir)
    return summary
