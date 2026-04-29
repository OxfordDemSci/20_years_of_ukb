"""Stage D: build winner merge patches and loser list."""

import glob
import json
import os

import polars as pl

from .config import EMPTY_LOSER_SCHEMA, EMPTY_PATCH_SCHEMA, IGNORE_PATCH_COLS


def write_empty_stage_d(out_dir, message):
    pl.DataFrame(EMPTY_PATCH_SCHEMA).write_parquet(
        os.path.join(out_dir, "merge_patches.parquet"), compression="zstd"
    )
    pl.DataFrame(EMPTY_LOSER_SCHEMA).write_parquet(
        os.path.join(out_dir, "loser_list.parquet"), compression="zstd"
    )
    print(message)


def is_list_like(value):
    return isinstance(value, (list, tuple))


def coalesce_scalar(winner_val, loser_vals):
    """For scalar fields, fill winner only if the winner value is missing."""
    if winner_val is not None:
        return winner_val, None
    for loser_id, value in loser_vals:
        if value is not None:
            return value, loser_id
    return None, None


def merge_lists(winner_val, loser_vals, cap):
    """For list fields, take the ordered union of winner and loser values."""
    out, seen, src_ids = [], set(), set()

    def add_many(seq):
        changed = False
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
                changed = True
                if cap and len(out) >= cap:
                    break
        return changed

    if is_list_like(winner_val):
        add_many(winner_val)

    for loser_id, value in loser_vals:
        if is_list_like(value) and value:
            before = len(out)
            changed = add_many(value)
            if changed and len(out) > before:
                src_ids.add(loser_id)
        if cap and len(out) >= cap:
            break

    if len(out) == 0:
        return None, None
    if is_list_like(winner_val) and out == list(winner_val):
        return out, None
    return out, (sorted(list(src_ids)) if src_ids else None)


def build_patch_for_group(gdf, winner_row, list_cap):
    """Build one JSON-serialisable patch for one duplicate cluster."""
    patch, columns_filled, used_sources = {}, {}, set()

    losers = (
        gdf.filter(~((pl.col("id") == winner_row["id"]) & (pl.col("file") == winner_row["file"])))
        .sort(by=["_score", "file", "id"], descending=[True, False, False])
        .to_dicts()
    )
    if not losers:
        return {}, {}, []

    candidate_cols = [c for c in gdf.columns if c not in IGNORE_PATCH_COLS]

    for col in candidate_cols:
        win_val = winner_row.get(col, None)
        loser_vals = [((r["id"], r["file"]), r.get(col, None)) for r in losers]

        probe = win_val
        if probe is None:
            for _, value in loser_vals:
                if value is not None:
                    probe = value
                    break

        if is_list_like(probe):
            merged, src_ids = merge_lists(win_val, loser_vals, list_cap)
            if merged is not None and (win_val is None or merged != win_val):
                patch[col] = merged
                if src_ids:
                    columns_filled[col] = [f"{i}:{f}" for i, f in src_ids]
                    used_sources.update(src_ids)
        else:
            filled, src = coalesce_scalar(win_val, loser_vals)
            if win_val is None and filled is not None:
                patch[col] = filled
                if src:
                    columns_filled[col] = f"{src[0]}:{src[1]}"
                    used_sources.add(src)

    merged_from = [f"{i}:{f}" for i, f in sorted(list(used_sources))]
    return patch, columns_filled, merged_from


def normalise_cluster_key(cid):
    """Handle Polars partition keys that may be scalar or one-element tuples."""
    if isinstance(cid, tuple) and len(cid) == 1:
        cid = cid[0]
    try:
        return int(cid)
    except Exception:
        return cid


def stage_d_build_patches(scores_dir, keep_decisions_path, out_dir, list_cap=200):
    """
    Build compact field-wise patches for winners and a list of loser rows to delete.

    The patch records only fields that can enrich the canonical winner. It also
    records provenance: which loser supplied each filled field.
    """
    os.makedirs(out_dir, exist_ok=True)
    parts_glob = os.path.join(scores_dir, "scores_*.parquet")
    part_paths = sorted(glob.glob(parts_glob))

    if not part_paths:
        write_empty_stage_d(out_dir, "[D] no scores parts found; writing empty outputs.")
        return

    keep = pl.read_parquet(keep_decisions_path)
    if keep.height == 0:
        write_empty_stage_d(out_dir, "[D] keep_decisions empty; wrote empty Stage D outputs.")
        return

    part_dfs = [pl.read_parquet(path) for path in part_paths]
    df_all = pl.concat(part_dfs, how="diagonal_relaxed")
    if df_all.height == 0:
        write_empty_stage_d(out_dir, "[D] no scored rows found; wrote empty Stage D outputs.")
        return

    winners = keep.select(["cluster_id", "canonical_id", "canonical_file"])

    losers_df = (
        df_all.join(winners, on="cluster_id", how="inner")
        .filter(~((pl.col("id") == pl.col("canonical_id")) & (pl.col("file") == pl.col("canonical_file"))))
        .select(["cluster_id", pl.col("id"), pl.col("file")])
    )
    losers_df.write_parquet(os.path.join(out_dir, "loser_list.parquet"), compression="zstd")
    print(f"[D] loser_list written: {os.path.join(out_dir, 'loser_list.parquet')} (rows={losers_df.height})")

    keep_map = {
        r["cluster_id"]: (r["canonical_id"], r["canonical_file"])
        for r in keep.iter_rows(named=True)
    }
    grouped = df_all.partition_by("cluster_id", as_dict=True)

    out_rows, n_clusters = [], len(grouped)
    cap = None if (list_cap is None or int(list_cap) == 0) else int(list_cap)

    for idx, (cid, gdf) in enumerate(grouped.items(), 1):
        cid_key = normalise_cluster_key(cid)
        if cid_key not in keep_map:
            continue

        win_id, win_file = keep_map[cid_key]
        wdf = gdf.filter((pl.col("id") == win_id) & (pl.col("file") == win_file))
        if wdf.height == 0:
            continue

        winner_row = wdf.to_dicts()[0]
        patch, cols_filled, merged_from = build_patch_for_group(gdf, winner_row, cap)

        out_rows.append({
            "cluster_id": cid_key,
            "canonical_id": win_id,
            "canonical_file": win_file,
            "patch_json": json.dumps(patch, ensure_ascii=False),
            "columns_filled_json": json.dumps(cols_filled, ensure_ascii=False),
            "merged_from_json": json.dumps(merged_from, ensure_ascii=False),
        })

        if idx % 1000 == 0:
            print(f"[D] processed {idx}/{n_clusters} clusters...")

    if out_rows:
        pl.DataFrame(out_rows).write_parquet(os.path.join(out_dir, "merge_patches.parquet"), compression="zstd")
        print(f"[D] merge_patches written: {os.path.join(out_dir, 'merge_patches.parquet')} (rows={len(out_rows)})")
    else:
        pl.DataFrame(EMPTY_PATCH_SCHEMA).write_parquet(
            os.path.join(out_dir, "merge_patches.parquet"), compression="zstd"
        )
        print("[D] no patches needed; wrote empty merge_patches.")
