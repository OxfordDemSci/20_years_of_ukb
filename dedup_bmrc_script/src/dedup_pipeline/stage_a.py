"""Stage A: build duplicate clusters from step-5 matching outputs."""

import glob
import os
import sys

import polars as pl

from .config import EMPTY_CLUSTERS_SCHEMA
from .union_find import parent, reset_union_find, uf_find, uf_union


def list_step5_inputs(step5_dir):
    """List step-5 duplicate parquet files, excluding the aggregate file if present."""
    files = sorted(glob.glob(os.path.join(step5_dir, "*.parquet")))
    files = [f for f in files if os.path.basename(f) != "all_duplicate_ids.parquet"]
    if not files:
        print("no step-5 parquet files under:", step5_dir, file=sys.stderr)
    return files


def key_cols_for(path):
    """Infer matching-key columns from a step-5 parquet file."""
    cols = pl.scan_parquet(path).collect_schema().names()
    keys = [c for c in cols if c not in ("id", "file", "columns")]
    if not keys:
        raise RuntimeError(f"no key columns in {path}")
    return keys


def write_empty_clusters(clusters_map_path):
    pl.DataFrame(EMPTY_CLUSTERS_SCHEMA).write_parquet(clusters_map_path, compression="zstd")


def stage_a_build_clusters(step5_dir, clusters_map_path, min_group_size):
    """
    Build clusters_map.parquet from step-5 duplicate groups.

    Each publication occurrence is represented as `(id, file)`. If two occurrences
    appear in the same matched-key group, Stage A unions them. Overlapping groups
    across many step-5 files are therefore merged into final connected components.
    """
    reset_union_find()
    files = list_step5_inputs(step5_dir)
    if not files:
        write_empty_clusters(clusters_map_path)
        return

    total_unions = 0

    for i, path in enumerate(files, 1):
        base = os.path.basename(path)
        try:
            keys = key_cols_for(path)
            lf = pl.scan_parquet(path)
            lf = lf.with_columns([pl.col(k).cast(pl.Utf8) for k in keys])

            groups = (
                lf.group_by(keys)
                .agg(
                    pl.len().alias("n"),
                    pl.col("id").alias("ids"),
                    pl.col("file").alias("files"),
                )
                .filter(pl.col("n") >= min_group_size)
                .select(["ids", "files"])
                .collect(engine="streaming")
            )

            unions = 0
            for ids, files_col in zip(groups["ids"], groups["files"]):
                if len(ids) < 2:
                    continue
                root = (ids[0], files_col[0])
                for j in range(1, len(ids)):
                    uf_union(root, (ids[j], files_col[j]))
                    unions += 1

            total_unions += unions
            print(f"[A {i}/{len(files)}] {base}: unions={unions}")

        except Exception as exc:
            print(f"[warn] Stage A failed on {base}: {exc}", file=sys.stderr)

    rep2cid, next_cid, rows = {}, 0, []
    for node in parent.keys():
        rep = uf_find(node)
        if rep not in rep2cid:
            rep2cid[rep] = next_cid
            next_cid += 1
        rows.append({"id": node[0], "file": node[1], "cluster_id": rep2cid[rep]})

    if rows:
        pl.DataFrame(rows).write_parquet(clusters_map_path, compression="zstd")
    else:
        write_empty_clusters(clusters_map_path)

    print(f"[A] wrote {clusters_map_path} (total unions: {total_unions})")
