"""Stage C: score duplicate occurrences and select one winner per cluster."""

import glob
import os

import polars as pl

from .config import EMPTY_KEEP_SCHEMA, RANKED_COLS, build_presence_weights


def write_empty_keep(out_dir, message):
    pl.DataFrame(EMPTY_KEEP_SCHEMA).write_parquet(
        os.path.join(out_dir, "keep_decisions.parquet"), compression="zstd"
    )
    print(message)


def score_exprs(present_cols, weights):
    """Build a Polars expression for presence-only informativeness scoring."""
    exprs = []
    for column in RANKED_COLS:
        if column not in present_cols:
            continue
        weight = weights.get(column, 0.0)
        if weight == 0.0:
            continue
        exprs.append(pl.col(column).is_not_null().cast(pl.Float64) * weight)

    if not exprs:
        return [pl.lit(0.0).alias("_score")]

    score = exprs[0]
    for expr in exprs[1:]:
        score = score + expr
    return [score.alias("_score")]


def stage_c_score_and_select(
    clusters_map_path,
    wanted_ids_path,
    raw_base_dir,
    out_dir,
    ext,
    zero_id_weight=False,
):
    """
    Read each touched raw shard, score only wanted rows, and select winners.

    Per-file scored rows are saved because Stage D later needs the winner and loser
    payloads to construct field-level merge patches.
    """
    os.makedirs(out_dir, exist_ok=True)
    parts_dir = os.path.join(out_dir, "scores_parts")
    os.makedirs(parts_dir, exist_ok=True)

    clusters = pl.read_parquet(clusters_map_path)
    wanted = pl.read_parquet(wanted_ids_path)

    if wanted.height == 0:
        write_empty_keep(out_dir, "[C] wanted_ids empty; wrote empty keep_decisions.")
        return

    weights = build_presence_weights(zero_id_weight=zero_id_weight)

    for i, row in enumerate(wanted.iter_rows(named=True), 1):
        file_stem, ids = row["file"], row["ids"]
        if not ids:
            continue

        shard_path = os.path.join(raw_base_dir.rstrip("/"), file_stem + ext)
        if not os.path.exists(shard_path):
            print(f"[warn] missing raw shard: {shard_path}")
            continue

        lf = pl.scan_parquet(shard_path).filter(pl.col("id").is_in(ids))
        present_cols = set(lf.collect_schema().names())
        if "file" not in present_cols:
            lf = lf.with_columns(pl.lit(file_stem).alias("file"))
            present_cols.add("file")

        score_cols = score_exprs(present_cols, weights)
        payload_cols = [c for c in RANKED_COLS if c in present_cols and c not in ("id", "file")]
        select_cols = ["id", "file"] + payload_cols + score_cols
        scored = lf.select(select_cols)

        cm_small = clusters.filter((pl.col("file") == file_stem) & (pl.col("id").is_in(ids))).lazy()
        scored = scored.join(cm_small, on=["id", "file"], how="inner")

        out_path = os.path.join(parts_dir, f"scores_{file_stem}.parquet")
        scored.sink_parquet(out_path, compression="zstd")
        print(f"[C {i}/{wanted.height}] wrote {out_path}")

    parts_paths = sorted(glob.glob(os.path.join(parts_dir, "scores_*.parquet")))
    if not parts_paths:
        write_empty_keep(out_dir, "[C] no scored rows; wrote empty keep_decisions.")
        return

    keep_frames = [
        pl.scan_parquet(path)
        .select(["cluster_id", "id", "file", "_score"])
        .collect(engine="streaming")
        for path in parts_paths
    ]

    if not keep_frames:
        write_empty_keep(out_dir, "[C] no scored rows; wrote empty keep_decisions.")
        return

    scored_all = pl.concat(keep_frames, how="vertical_relaxed")
    if scored_all.height == 0:
        write_empty_keep(out_dir, "[C] no scored rows; wrote empty keep_decisions.")
        return

    sorted_df = scored_all.sort(
        by=["cluster_id", "_score", "file", "id"],
        descending=[False, True, False, False],
    )

    keep = sorted_df.group_by("cluster_id").agg([
        pl.first("id").alias("canonical_id"),
        pl.first("file").alias("canonical_file"),
        pl.first("_score").alias("best_score"),
        pl.len().alias("total_members"),
    ])

    keep.write_parquet(os.path.join(out_dir, "keep_decisions.parquet"), compression="zstd")
    print(f"[C] keep_decisions written: {os.path.join(out_dir, 'keep_decisions.parquet')}")
