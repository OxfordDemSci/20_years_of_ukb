"""Stage E: apply loser deletions and winner patches to raw shards."""

import json
import os
import time
from datetime import datetime

import polars as pl


def shards_to_process(loser_list_path, merge_patches_path, only_file=None):
    shards = set()
    if os.path.exists(loser_list_path):
        loser_files = pl.read_parquet(loser_list_path, columns=["file"])
        shards.update(loser_files["file"].to_list())
    if os.path.exists(merge_patches_path):
        patch_files = pl.read_parquet(merge_patches_path, columns=["canonical_file"])
        shards.update(patch_files["canonical_file"].to_list())
    if only_file:
        return [only_file] if (not shards or only_file in shards) else []
    return sorted(shards)


def load_losers_for(shard_file, loser_list_path):
    if not os.path.exists(loser_list_path):
        return set()
    df = pl.read_parquet(loser_list_path, columns=["id", "file"])
    if df.height == 0:
        return set()
    ids = df.filter(pl.col("file") == shard_file).select("id").to_series().to_list()
    return set(ids)


def build_patch_df_for(shard_file, merge_patches_path):
    if not os.path.exists(merge_patches_path):
        return None, []

    mp = pl.read_parquet(merge_patches_path, columns=["canonical_id", "canonical_file", "patch_json"])
    if mp.height == 0:
        return None, []

    mp = mp.filter(pl.col("canonical_file") == shard_file)
    if mp.height == 0:
        return None, []

    rows = []
    patch_cols_all = set()
    for r in mp.iter_rows(named=True):
        pid = r["canonical_id"]
        patch_json = r["patch_json"]
        try:
            patch = json.loads(patch_json) if patch_json else {}
        except Exception:
            patch = {}
        if not isinstance(patch, dict):
            patch = {}

        row = {"id": pid}
        for key, value in patch.items():
            row[key] = value
            patch_cols_all.add(key)
        rows.append(row)

    if not rows:
        return None, []

    df = pl.DataFrame(rows)
    rename_map = {c: f"__patch_{c}" for c in df.columns if c != "id"}
    df = df.rename(rename_map)
    return df, list(patch_cols_all)


def atomic_replace(tmp_path, dest_path):
    os.replace(tmp_path, dest_path)


def write_stagee_progress(out_dir, summary):
    os.makedirs(out_dir, exist_ok=True)

    txt_path = os.path.join(out_dir, "progress_summary.txt")
    pq_path = os.path.join(out_dir, "progress_summary.parquet")
    hist_path = os.path.join(out_dir, "progress_history.tsv")

    lines = [
        f"timestamp: {summary['timestamp']}",
        f"elapsed_minutes: {summary['elapsed_minutes']}",
        f"processed_shards: {summary['processed_shards']}/{summary['total_shards']}",
        f"remaining_shards: {summary['remaining_shards']}",
        f"pct_complete: {summary['pct_complete']}",
        f"total_deleted_present: {summary['total_deleted_present']}",
        f"total_patched_present: {summary['total_patched_present']}",
        f"total_rows_before: {summary['total_rows_before']}",
        f"total_rows_after: {summary['total_rows_after']}",
    ]

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    pl.DataFrame([summary]).write_parquet(pq_path, compression="zstd")

    write_header = not os.path.exists(hist_path)
    with open(hist_path, "a", encoding="utf-8") as f:
        if write_header:
            f.write("\t".join(summary.keys()) + "\n")
        f.write("\t".join(str(summary[k]) for k in summary.keys()) + "\n")


def apply_to_shard(raw_base_dir, ext, shard_file, losers_set, patch_df, patch_cols, out_dir):
    """Apply deletions and patches to one raw shard using an atomic replace."""
    shard_path = os.path.join(raw_base_dir.rstrip("/"), shard_file + ext)
    if not os.path.exists(shard_path):
        print(f"[warn] missing shard: {shard_path}")
        return None

    loser_ids = list(losers_set) if losers_set else []

    lf_ids = pl.scan_parquet(shard_path).select(pl.col("id"))
    n_before = lf_ids.count().collect().item()

    n_del_present = 0
    if loser_ids:
        n_del_present = lf_ids.filter(pl.col("id").is_in(loser_ids)).count().collect().item()

    lf = pl.scan_parquet(shard_path)
    if loser_ids:
        lf = lf.filter(~pl.col("id").is_in(loser_ids))

    if patch_df is not None and patch_cols:
        existing_cols = set(lf.collect_schema().names())
        missing = [c for c in patch_cols if c not in existing_cols]
        if missing:
            lf = lf.with_columns([pl.lit(None).alias(c) for c in missing])

        lf = lf.join(patch_df.lazy(), on="id", how="left")

        updates = []
        for c in patch_cols:
            pcol = f"__patch_{c}"
            updates.append(pl.coalesce([pl.col(pcol), pl.col(c)]).alias(c))
        lf = lf.with_columns(updates)

        drop_cols = [f"__patch_{c}" for c in patch_cols]
        lf_cols = set(lf.collect_schema().names())
        lf = lf.drop([c for c in drop_cols if c in lf_cols])

    os.makedirs(os.path.join(out_dir, "changelogs"), exist_ok=True)
    tmp_path = shard_path + ".tmp"
    lf.sink_parquet(tmp_path, compression="zstd")
    atomic_replace(tmp_path, shard_path)

    n_after = pl.scan_parquet(shard_path).select(pl.count()).collect().item()

    n_patched_present = 0
    if patch_df is not None and patch_cols:
        ids_to_patch = patch_df["id"].to_list()
        if ids_to_patch:
            n_patched_present = (
                pl.scan_parquet(shard_path)
                .select(pl.col("id").is_in(ids_to_patch).sum())
                .collect()
                .item()
            )

    changelog_path = os.path.join(out_dir, "changelogs", f"{shard_file}.parquet")
    stats = {
        "file": shard_file,
        "path": shard_path,
        "num_rows_before": int(n_before),
        "num_deleted_present": int(n_del_present),
        "num_patched_present": int(n_patched_present),
        "num_rows_after": int(n_after),
    }

    pl.DataFrame([stats]).write_parquet(changelog_path, compression="zstd")

    print(
        f"[E] {shard_file}: -{n_del_present} deleted, "
        f"~{n_patched_present} patched, rows {n_before}→{n_after}"
    )

    return stats


def stage_e_apply(raw_base_dir, ext, loser_list_path, merge_patches_path, out_dir, only_file=None):
    """Apply Stage D decisions to raw shards."""
    os.makedirs(out_dir, exist_ok=True)
    shard_files = shards_to_process(loser_list_path, merge_patches_path, only_file)

    if only_file and not shard_files:
        print(f"[E] --only-file {only_file} not found among affected shards; nothing to do.")
        return
    if not shard_files:
        print("[E] no shards to process (nothing in loser_list/merge_patches).")
        return

    total_shards = len(shard_files)
    processed_shards = 0
    total_deleted_present = 0
    total_patched_present = 0
    total_rows_before = 0
    total_rows_after = 0

    started_at = time.time()
    last_progress_write = started_at

    for shard_file in sorted(shard_files):
        losers_set = load_losers_for(shard_file, loser_list_path)
        patch_df, patch_cols = build_patch_df_for(shard_file, merge_patches_path)

        stats = apply_to_shard(raw_base_dir, ext, shard_file, losers_set, patch_df, patch_cols, out_dir)
        del losers_set, patch_df, patch_cols

        if stats is None:
            continue

        processed_shards += 1
        total_deleted_present += stats["num_deleted_present"]
        total_patched_present += stats["num_patched_present"]
        total_rows_before += stats["num_rows_before"]
        total_rows_after += stats["num_rows_after"]

        now = time.time()
        should_write_progress = (
            processed_shards == total_shards
            or (processed_shards % 10 == 0)
            or ((now - last_progress_write) >= 600)
        )

        if should_write_progress:
            summary = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "elapsed_minutes": round((now - started_at) / 60.0, 2),
                "processed_shards": int(processed_shards),
                "total_shards": int(total_shards),
                "remaining_shards": int(total_shards - processed_shards),
                "pct_complete": round(100.0 * processed_shards / total_shards, 2),
                "total_deleted_present": int(total_deleted_present),
                "total_patched_present": int(total_patched_present),
                "total_rows_before": int(total_rows_before),
                "total_rows_after": int(total_rows_after),
            }

            write_stagee_progress(out_dir, summary)

            print(
                f"[E summary] processed {processed_shards}/{total_shards}, "
                f"total_deleted={total_deleted_present}, "
                f"total_patched={total_patched_present}"
            )

            last_progress_write = now
