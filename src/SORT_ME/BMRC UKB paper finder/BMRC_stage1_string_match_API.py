#!/usr/bin/env python3
from __future__ import annotations

import io
import os
import re
import time
from contextlib import redirect_stdout, redirect_stderr

import dimcli
import pandas as pd

API_KEY = "YOUR KEY HERE / SET AS ENVIRONMENT VARIABLE"

OUTPUT_DIR = "/well/mills/projects/scientometric/data_copy_script"
CACHE_DIR = f"{OUTPUT_DIR}/ukb_term_year_cache_no_acronym"

API_RAW_CSV = f"{OUTPUT_DIR}/matched_ukb_full_api_raw_1.csv"
API_MATCHED_CSV = f"{OUTPUT_DIR}/matched_ukb_full_api_with_abstract_1.csv"

LIMIT = 1000
SLEEP_SECONDS = 0.25
MAX_RETRIES = 8

START_YEAR = 1995
END_YEAR = 2026

SEARCH_TERMS = [
    '"UK Biobank"',
    '"UK Bio Bank"',
    '"United Kingdom Biobank"',
    '"United Kingdom Bio-bank"',
    '"United Kingdom Bio Bank"',
    '"biobank" AND "United Kingdom"',
    '"bio-bank" AND "united kingdom"',
    '"Bio Bank" AND "United Kingdom"',
    '"biobank" AND "UK"',
    '"bio-bank" AND "UK"',
    '"bio bank" AND "UK"',
]


def p(msg: str):
    print(msg, flush=True)


def safe_name(term: str) -> str:
    s = re.sub(r'[^A-Za-z0-9]+', '_', term).strip('_')
    return s[:100] if s else "term"


def dimensions_login(api_key: str):
    dimcli.login(key=api_key, endpoint="https://app.dimensions.ai/api/dsl/v2")
    return dimcli.Dsl()


def build_query(term: str, year: int, skip: int) -> str:
    return f'''
    search publications for """ {term} """
    where year={year} and abstract is not empty
    return publications[id + title + abstract + year] limit {LIMIT} skip {skip}
    '''


def run_query(dsl, query: str):
    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        result = dsl.query(query)
    captured = buf.getvalue()

    df = result.as_dataframe()
    if df is None or len(df) == 0:
        df = pd.DataFrame(columns=["id", "title", "abstract", "year"])

    for col in ["id", "title", "abstract", "year"]:
        if col not in df.columns:
            df[col] = None

    return df[["id", "title", "abstract", "year"]].copy(), captured, result


def scrape_term_year(dsl, term: str, year: int) -> pd.DataFrame:
    cache_path = f"{CACHE_DIR}/{safe_name(term)}_{year}.csv"

    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
        p(f"[CACHE] term={term} | year={year} | rows={len(df)}")
        return df

    parts = []
    skip = 0
    total_count = None
    page = 0

    while True:
        page += 1
        query = build_query(term, year, skip)

        part = None
        result = None
        last_err = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                part, captured, result = run_query(dsl, query)
                if captured.strip():
                    p(f"[DSL] term={term} | year={year} | page={page} | {captured.strip()[:300]}")
                break
            except Exception as e:
                last_err = e
                wait = min(120, SLEEP_SECONDS * (2 ** (attempt - 1)))
                p(f"[RETRY] term={term} | year={year} | page={page} | attempt={attempt}/{MAX_RETRIES} | error={e} | sleep={wait:.1f}s")
                time.sleep(wait)

        if part is None:
            p(f"[FAILED] term={term} | year={year} | error={last_err}")
            break

        if total_count is None:
            try:
                total_count = result["_stats"]["total_count"]
            except Exception:
                total_count = None
            p(f"[TERM-YEAR START] term={term} | year={year} | total_count={total_count}")

        n = len(part)
        p(f"[PAGE] term={term} | year={year} | page={page} | skip={skip} | rows={n}")

        if n == 0:
            break

        part["search_term"] = term
        parts.append(part)
        skip += n

        if total_count is not None and skip >= total_count:
            break
        if n < LIMIT:
            break
        if skip >= 50000:
            p(f"[WARNING] term={term} | year={year} reached 50,000 pagination cap. Consider splitting further.")
            break

        time.sleep(SLEEP_SECONDS)

    if len(parts) == 0:
        out = pd.DataFrame(columns=["id", "title", "abstract", "year", "search_term"])
    else:
        out = pd.concat(parts, ignore_index=True)

    out.to_csv(cache_path, index=False)
    p(f"[TERM-YEAR DONE] term={term} | year={year} | rows={len(out)} | saved={cache_path}")
    return out


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    p("=" * 100)
    p("Stage 1: notebook-style UKB Dimensions scrape, without UKB/UKBB acronym-only terms")
    p(f"Years: {START_YEAR}-{END_YEAR}")
    p(f"Number of terms: {len(SEARCH_TERMS)}")
    p(f"Cache dir: {CACHE_DIR}")
    p("=" * 100)

    dsl = dimensions_login(API_KEY)
    p("[API] Logged into Dimensions.")

    all_parts = []
    start_all = time.time()
    total_iters = len(SEARCH_TERMS) * (END_YEAR - START_YEAR + 1)
    done = 0

    for term in SEARCH_TERMS:
        for year in range(START_YEAR, END_YEAR + 1):
            done += 1
            t0 = time.time()

            p("\n" + "-" * 100)
            p(f"[PROGRESS] {done}/{total_iters} | term={term} | year={year}")

            df = scrape_term_year(dsl, term, year)
            if len(df) > 0:
                all_parts.append(df)

            elapsed = time.time() - t0
            total_elapsed = (time.time() - start_all) / 60

            current_raw = sum(len(x) for x in all_parts)
            if len(all_parts) > 0:
                current_unique = pd.concat(all_parts, ignore_index=True)["id"].astype(str).nunique()
            else:
                current_unique = 0

            p(f"[PROGRESS] term-year raw rows={len(df)} | cumulative_raw={current_raw} | cumulative_unique_ids={current_unique}")
            p(f"[TIME] this term-year={elapsed:.2f}s | total_runtime={total_elapsed:.2f} min")

    if len(all_parts) == 0:
        raw_df = pd.DataFrame(columns=["id", "title", "abstract", "year", "search_term"])
        dedup_df = pd.DataFrame(columns=["id", "title", "abstract", "year"])
    else:
        raw_df = pd.concat(all_parts, ignore_index=True)
        raw_df["id"] = raw_df["id"].astype(str).str.strip()
        raw_df = raw_df[raw_df["id"] != ""].copy()

        dedup_df = (
            raw_df
            .sort_values(["id", "year"])
            .drop_duplicates(subset=["id"], keep="first")
            [["id", "title", "abstract", "year"]]
            .sort_values("id")
            .reset_index(drop=True)
        )

    raw_df.to_csv(API_RAW_CSV, index=False)
    dedup_df.to_csv(API_MATCHED_CSV, index=False)

    total_elapsed = (time.time() - start_all) / 60
    p("=" * 100)
    p("[STAGE 1 DONE]")
    p(f"Raw rows across term-year searches : {len(raw_df)}")
    p(f"Unique matched publication ids     : {len(dedup_df)}")
    p(f"Raw API CSV                        : {API_RAW_CSV}")
    p(f"Deduplicated API CSV               : {API_MATCHED_CSV}")
    p(f"Total runtime                      : {total_elapsed:.2f} min")
    p("=" * 100)


if __name__ == "__main__":
    main()
