#!/usr/bin/env python3
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import pyarrow.parquet as pq

DATA_DIR = "/well/mills/projects/scientometric/data_copy"
OUTPUT_DIR = "/well/mills/projects/scientometric/data_copy_script"

API_MATCHED_CSV = f"{OUTPUT_DIR}/matched_ukb_full_api_with_abstract_1.csv"
FINAL_OUTPUT_CSV = f"{OUTPUT_DIR}/matched_ukb_full_1.csv"

CHECKPOINT_EVERY_N_FILES = 25
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "8"))

MATCHED_IDS: Set[str] = set()
API_META: Dict[str, Tuple[Optional[str], Optional[str], Optional[object]]] = {}


def p(msg: str):
    print(msg, flush=True)


def is_nonempty_text(x) -> bool:
    if pd.isna(x):
        return False
    return str(x).strip() != ""


def list_parquet_files(data_dir: str) -> List[str]:
    files = []
    for name in os.listdir(data_dir):
        full = os.path.join(data_dir, name)
        if os.path.isfile(full) and name.endswith(".parquet"):
            files.append(full)
    files.sort()
    return files


def get_parquet_columns(path: str) -> Set[str]:
    return set(pq.ParquetFile(path).schema_arrow.names)


def save_checkpoint(parts: List[pd.DataFrame]) -> int:
    if len(parts) == 0:
        out = pd.DataFrame(columns=["id", "title", "abstract", "year"])
    else:
        out = pd.concat(parts, ignore_index=True)
        out["id"] = out["id"].astype(str).str.strip()
        out = out[out["id"] != ""]
        out = out.drop_duplicates(subset=["id"], keep="first").sort_values("id").reset_index(drop=True)

    out.to_csv(FINAL_OUTPUT_CSV, index=False)
    return len(out)


def process_one_parquet(path: str):
    start = time.time()
    basename = os.path.basename(path)

    stats = {
        "file": basename,
        "status": "ok",
        "n_rows": 0,
        "n_nonempty_abstract": 0,
        "n_unique_candidates": 0,
        "n_intersection": 0,
        "elapsed_sec": 0.0,
        "error": None,
    }

    try:
        cols = get_parquet_columns(path)
        needed = [c for c in ["id", "title", "abstract", "year"] if c in cols]

        if "id" not in needed or "abstract" not in needed:
            stats["status"] = "missing_columns"
            stats["elapsed_sec"] = time.time() - start
            return stats, pd.DataFrame(columns=["id", "title", "abstract", "year"])

        df = pd.read_parquet(path, columns=needed)
        stats["n_rows"] = len(df)

        df = df[df["abstract"].apply(is_nonempty_text)].copy()
        stats["n_nonempty_abstract"] = len(df)

        if len(df) == 0:
            stats["elapsed_sec"] = time.time() - start
            return stats, pd.DataFrame(columns=["id", "title", "abstract", "year"])

        df["id"] = df["id"].astype(str).str.strip()
        df = df[df["id"] != ""].drop_duplicates(subset=["id"]).copy()
        stats["n_unique_candidates"] = len(df)

        hit_df = df[df["id"].isin(MATCHED_IDS)].copy()
        stats["n_intersection"] = len(hit_df)

        if len(hit_df) == 0:
            stats["elapsed_sec"] = time.time() - start
            return stats, pd.DataFrame(columns=["id", "title", "abstract", "year"])

        rows = []
        for row in hit_df.itertuples(index=False):
            pid = str(row.id).strip()

            local_title = getattr(row, "title", None) if hasattr(row, "title") else None
            local_abstract = getattr(row, "abstract", None) if hasattr(row, "abstract") else None
            local_year = getattr(row, "year", None) if hasattr(row, "year") else None

            api_title, api_abstract, api_year = API_META.get(pid, (None, None, None))

            title = local_title if is_nonempty_text(local_title) else api_title
            abstract = local_abstract if is_nonempty_text(local_abstract) else api_abstract
            year = local_year if is_nonempty_text(local_year) else api_year

            if is_nonempty_text(abstract):
                rows.append({
                    "id": pid,
                    "title": title,
                    "abstract": abstract,
                    "year": year,
                })

        out = pd.DataFrame(rows)
        if len(out) > 0:
            out = out.drop_duplicates(subset=["id"], keep="first").reset_index(drop=True)

        stats["elapsed_sec"] = time.time() - start
        return stats, out

    except Exception as e:
        stats["status"] = "error"
        stats["error"] = str(e)
        stats["elapsed_sec"] = time.time() - start
        return stats, pd.DataFrame(columns=["id", "title", "abstract", "year"])


def main():
    global MATCHED_IDS, API_META

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    p("=" * 100)
    p("Stage 2: local parquet scan + intersect with no-acronym Stage 1 API IDs")
    p(f"API_MATCHED_CSV  : {API_MATCHED_CSV}")
    p(f"FINAL_OUTPUT_CSV : {FINAL_OUTPUT_CSV}")
    p(f"MAX_WORKERS      : {MAX_WORKERS}")
    p("=" * 100)

    api_df = pd.read_csv(API_MATCHED_CSV)
    for col in ["id", "title", "abstract", "year"]:
        if col not in api_df.columns:
            api_df[col] = None

    api_df["id"] = api_df["id"].astype(str).str.strip()
    api_df = api_df[api_df["id"] != ""].drop_duplicates(subset=["id"], keep="first").reset_index(drop=True)

    MATCHED_IDS = set(api_df["id"].tolist())
    API_META = {
        row.id: (row.title, row.abstract, row.year)
        for row in api_df.itertuples(index=False)
    }

    p(f"[LOCAL] Loaded Stage 1 unique API ids: {len(MATCHED_IDS)}")

    parquet_files = list_parquet_files(DATA_DIR)
    total_files = len(parquet_files)
    p(f"[LOCAL] Found {total_files} local parquet files")

    start_all = time.time()
    output_parts: List[pd.DataFrame] = []
    seen_ids: Set[str] = set()

    cumulative_rows = 0
    cumulative_nonempty = 0
    cumulative_candidates = 0
    cumulative_intersections = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        future_to_path = {ex.submit(process_one_parquet, path): path for path in parquet_files}

        for done_idx, future in enumerate(as_completed(future_to_path), start=1):
            stats, part = future.result()

            cumulative_rows += stats["n_rows"]
            cumulative_nonempty += stats["n_nonempty_abstract"]
            cumulative_candidates += stats["n_unique_candidates"]
            cumulative_intersections += stats["n_intersection"]

            new_output = 0
            if len(part) > 0:
                part["id"] = part["id"].astype(str).str.strip()
                part = part[~part["id"].isin(seen_ids)].copy()
                if len(part) > 0:
                    seen_ids.update(part["id"].astype(str).tolist())
                    output_parts.append(part)
                    new_output = len(part)

            elapsed = (time.time() - start_all) / 60

            p(
                f"[LOCAL {done_idx}/{total_files}] {stats['file']} | "
                f"status={stats['status']} | rows={stats['n_rows']} | "
                f"nonempty={stats['n_nonempty_abstract']} | candidates={stats['n_unique_candidates']} | "
                f"intersections={stats['n_intersection']} | new_output={new_output} | "
                f"cumulative_output={len(seen_ids)} | file_time={stats['elapsed_sec']:.2f}s | "
                f"runtime={elapsed:.2f} min"
            )

            if stats["status"] == "error":
                p(f"[ERROR] {stats['file']}: {stats['error']}")

            if done_idx % CHECKPOINT_EVERY_N_FILES == 0 or done_idx == total_files:
                n = save_checkpoint(output_parts)
                p(f"[CHECKPOINT] after {done_idx} files | rows={n} | saved={FINAL_OUTPUT_CSV}")

    final_n = save_checkpoint(output_parts)
    elapsed = (time.time() - start_all) / 60

    p("=" * 100)
    p("[STAGE 2 DONE]")
    p(f"Cumulative rows loaded          : {cumulative_rows}")
    p(f"Cumulative non-empty abstracts  : {cumulative_nonempty}")
    p(f"Cumulative local candidates     : {cumulative_candidates}")
    p(f"Cumulative intersections        : {cumulative_intersections}")
    p(f"Final unique matched papers     : {final_n}")
    p(f"Final CSV                       : {FINAL_OUTPUT_CSV}")
    p(f"Runtime                         : {elapsed:.2f} min")
    p("=" * 100)


if __name__ == "__main__":
    main()
