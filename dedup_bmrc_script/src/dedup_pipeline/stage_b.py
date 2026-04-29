"""Stage B: route cluster member ids to the raw shards that must be read."""

import os

import polars as pl

from .config import EMPTY_FILES_TO_TOUCH_SCHEMA, EMPTY_WANTED_SCHEMA


def write_empty_stage_b(out_dir):
    pl.DataFrame(EMPTY_WANTED_SCHEMA).write_parquet(
        os.path.join(out_dir, "wanted_ids_by_file.parquet"), compression="zstd"
    )
    pl.DataFrame(EMPTY_FILES_TO_TOUCH_SCHEMA).write_parquet(
        os.path.join(out_dir, "files_to_touch.parquet"), compression="zstd"
    )
    pl.DataFrame(EMPTY_FILES_TO_TOUCH_SCHEMA).write_parquet(
        os.path.join(out_dir, "missing_files.parquet"), compression="zstd"
    )


def stage_b_route_ids(clusters_map_path, raw_base_dir, out_dir, ext, check_exists=True):
    """
    Group cluster members by source shard.

    This prevents later stages from repeatedly opening the same raw shard for each
    duplicate group. Instead, each raw shard gets one list of wanted ids.
    """
    os.makedirs(out_dir, exist_ok=True)

    lf = pl.scan_parquet(clusters_map_path).select(["file", "id"])
    grouped = (
        lf.group_by("file")
        .agg([
            pl.col("id").n_unique().alias("n_ids"),
            pl.col("id").unique().alias("ids"),
        ])
        .collect(engine="streaming")
    )

    base = raw_base_dir.rstrip("/")
    grouped = grouped.with_columns(
        pl.format("{}/{}{}", pl.lit(base), pl.col("file"), pl.lit(ext)).alias("path")
    )

    if grouped.height == 0:
        write_empty_stage_b(out_dir)
        print("[B] clusters_map empty; wrote empty Stage B outputs.")
        return

    if check_exists:
        paths = grouped["path"].to_list()
        exists = [os.path.exists(p) for p in paths]
        grouped = grouped.with_columns(pl.Series("exists", exists))
        files_to_touch = grouped.filter(pl.col("exists")).select(["file", "path", "n_ids"])
        missing = grouped.filter(~pl.col("exists")).select(["file", "path", "n_ids"])
        if missing.height > 0:
            print(f"[B] missing shards under {base}: {missing.height}")
            print(missing.head(5))
    else:
        files_to_touch = grouped.select(["file", "path", "n_ids"])
        missing = pl.DataFrame(EMPTY_FILES_TO_TOUCH_SCHEMA)

    wanted = grouped.join(files_to_touch.select(["file"]), on="file", how="inner").select(
        ["file", "ids", "n_ids"]
    )

    wanted.write_parquet(os.path.join(out_dir, "wanted_ids_by_file.parquet"), compression="zstd")
    files_to_touch.write_parquet(os.path.join(out_dir, "files_to_touch.parquet"), compression="zstd")
    missing.write_parquet(os.path.join(out_dir, "missing_files.parquet"), compression="zstd")

    print(f"[B] wrote {os.path.join(out_dir, 'wanted_ids_by_file.parquet')} (rows={wanted.height})")
    print(f"[B] wrote {os.path.join(out_dir, 'files_to_touch.parquet')} (rows={files_to_touch.height})")
    if missing.height:
        print(f"[B] wrote {os.path.join(out_dir, 'missing_files.parquet')} (rows={missing.height})")
