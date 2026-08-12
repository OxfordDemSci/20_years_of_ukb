import argparse
import csv
import html
import io
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SHOWCASE_URL = "https://biobank.ndph.ox.ac.uk/ukb/scdown.cgi?fmt=txt&id=19"
DIMENSIONS_AUTH_URL = "https://app.dimensions.ai/api/auth.json"
DIMENSIONS_QUERY_URL = "https://app.dimensions.ai/api/dsl/v2"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("ukb_showcase_dimensions_id_title.csv"))
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--unmatched-output", type=Path)
    parser.add_argument("--batch-size", type=int, default=400)
    parser.add_argument("--sleep", type=float, default=2.1)
    parser.add_argument("--retries", type=int, default=5)
    return parser.parse_args()


def clean_title(value):
    text = html.unescape(re.sub(r"<[^>]+>", "", value or ""))
    return re.sub(r"\s+", " ", text).strip()


def normalize_pmid(value):
    value = (value or "").strip()
    return value if value != "0" and re.fullmatch(r"\d+", value) else ""


def normalize_doi(value):
    return (value or "").strip().lower()


def request_json(url, payload=None, token=None, query=None, retries=5):
    headers = {}
    if token:
        headers["Authorization"] = f"JWT {token}"
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload).encode("utf-8")
    else:
        headers["Content-Type"] = "text/plain; charset=utf-8"
        body = query.encode("utf-8")
    for attempt in range(retries + 1):
        try:
            request = Request(url, data=body, headers=headers, method="POST")
            with urlopen(request, timeout=240) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code not in {429, 500, 502, 503, 504} or attempt == retries:
                raise RuntimeError(f"Dimensions HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError) as exc:
            if attempt == retries:
                raise RuntimeError(f"Dimensions request failed: {exc}") from exc
        time.sleep(min(60.0, 2.0**attempt))
    raise RuntimeError("Dimensions request failed.")


def authenticate(api_key, retries):
    response = request_json(DIMENSIONS_AUTH_URL, payload={"key": api_key}, retries=retries)
    token = response.get("token")
    if not token:
        raise RuntimeError("Dimensions authentication returned no token.")
    return token


def download_showcase(retries):
    for attempt in range(retries + 1):
        try:
            request = Request(SHOWCASE_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(request, timeout=240) as response:
                return response.read().decode("utf-8-sig")
        except (HTTPError, URLError, TimeoutError) as exc:
            if attempt == retries:
                raise RuntimeError(f"Showcase download failed: {exc}") from exc
        time.sleep(min(60.0, 2.0**attempt))
    raise RuntimeError("Showcase download failed.")


def deduplicate(rows, field, keep_blank=True):
    seen = set()
    result = []
    for row in rows:
        value = row[field]
        if keep_blank and not value:
            result.append(row)
        elif value not in seen:
            seen.add(value)
            result.append(row)
    return result


def parse_showcase(text):
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    required = {"pub_id", "pubmed_id", "doi", "title"}
    missing = required - set(reader.fieldnames or [])
    if missing:
        raise RuntimeError(f"Showcase data is missing columns: {sorted(missing)}")
    rows = [
        {
            "showcase_pub_id": (row.get("pub_id") or "").strip(),
            "pmid": normalize_pmid(row.get("pubmed_id")),
            "doi": normalize_doi(row.get("doi")),
            "showcase_title": clean_title(row.get("title")),
        }
        for row in reader
    ]
    rows = deduplicate(rows, "doi")
    rows = deduplicate(rows, "pmid")
    rows = deduplicate(rows, "showcase_pub_id", keep_blank=False)
    return rows


def batches(values, size):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def query_dimensions(token, field, values, batch_size, sleep, retries):
    values = sorted(set(values), key=int if field == "pmid" else None)
    results = {}
    total = (len(values) + batch_size - 1) // batch_size
    for number, batch in enumerate(batches(values, batch_size), start=1):
        quoted = ",".join(json.dumps(value) for value in batch)
        query = (
            f"search publications where {field} in [{quoted}] "
            "return publications[id+pmid+doi+title] limit 1000"
        )
        response = request_json(DIMENSIONS_QUERY_URL, token=token, query=query, retries=retries)
        publications = response.get("publications") or []
        for publication in publications:
            key = str(publication.get(field) or "").strip()
            if field == "doi":
                key = key.lower()
            publication_id = str(publication.get("id") or "").strip()
            if key and publication_id:
                results.setdefault(
                    key,
                    {
                        "id": publication_id,
                        "title": clean_title(publication.get("title")),
                        "pmid": str(publication.get("pmid") or "").strip(),
                        "doi": str(publication.get("doi") or "").strip().lower(),
                    },
                )
        print(f"{field} {number}/{total}: requested={len(batch)} returned={len(publications)}", flush=True)
        if sleep > 0:
            time.sleep(sleep)
    return results


def match_rows(rows, pmid_results, doi_results):
    matched = []
    unmatched = []
    seen_ids = set()
    for row in rows:
        publication = pmid_results.get(row["pmid"]) if row["pmid"] else None
        method = "pmid" if publication else ""
        if not publication and row["doi"]:
            publication = doi_results.get(row["doi"])
            method = "doi" if publication else ""
        if not publication:
            unmatched.append(row)
            continue
        if publication["id"] in seen_ids:
            continue
        seen_ids.add(publication["id"])
        matched.append(
            {
                "id": publication["id"],
                "title": publication["title"] or row["showcase_title"],
                "pmid": publication["pmid"] or row["pmid"],
                "doi": publication["doi"] or row["doi"],
                "showcase_pub_id": row["showcase_pub_id"],
                "match_method": method,
            }
        )
    return matched, unmatched


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise RuntimeError("--batch-size must be at least 1.")
    api_key = os.environ.get("DIMENSIONS_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Set DIMENSIONS_API_KEY before running the script.")
    rows = parse_showcase(download_showcase(args.retries))
    token = authenticate(api_key, args.retries)
    pmids = [row["pmid"] for row in rows if row["pmid"]]
    pmid_results = query_dimensions(token, "pmid", pmids, args.batch_size, args.sleep, args.retries)
    dois = [row["doi"] for row in rows if row["doi"] and (not row["pmid"] or row["pmid"] not in pmid_results)]
    doi_results = query_dimensions(token, "doi", dois, args.batch_size, args.sleep, args.retries)
    matched, unmatched = match_rows(rows, pmid_results, doi_results)
    write_csv(args.output, ["id", "title"], ({"id": row["id"], "title": row["title"]} for row in matched))
    if args.audit_output:
        write_csv(
            args.audit_output,
            ["id", "title", "pmid", "doi", "showcase_pub_id", "match_method"],
            matched,
        )
    if args.unmatched_output:
        write_csv(
            args.unmatched_output,
            ["showcase_pub_id", "pmid", "doi", "showcase_title"],
            unmatched,
        )
    print(
        json.dumps(
            {
                "showcase_rows_after_deduplication": len(rows),
                "mapped_dimensions_publications": len(matched),
                "unmatched_showcase_publications": len(unmatched),
                "output": str(args.output),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
