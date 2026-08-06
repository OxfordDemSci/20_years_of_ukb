#!/usr/bin/env python3
import argparse
import ast
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


AUTH_URL = "https://app.dimensions.ai/api/auth.json"
QUERY_URL = "https://app.dimensions.ai/api/dsl/v2"
API_KEY_ENV = "DIMENSIONS_API_KEY"

ENDPOINTS = {
    "clinical_trials": {
        "query_field": "publication_ids",
        "link_field": "publication_ids",
        "link_id": "clinical_trial_id",
        "mode": "reverse",
    },
    "grants": {
        "query_field": "id",
        "publication_field": "supporting_grant_ids",
        "link_id": "grant_id",
        "mode": "publication_ids",
    },
    "patents": {
        "query_field": "publication_ids",
        "link_field": "publication_ids",
        "link_id": "patent_id",
        "mode": "reverse",
    },
    "policy_documents": {
        "query_field": "publication_ids",
        "link_field": "publication_ids",
        "link_id": "policy_document_id",
        "mode": "reverse",
    },
    "datasets": {
        "query_field": "associated_publication_id",
        "link_field": "associated_publication_id",
        "link_id": "dataset_id",
        "mode": "single_reverse",
    },
    "source_titles": {
        "query_field": "id",
        "publication_field": "source_title",
        "link_id": "source_title_id",
        "mode": "publication_object_ids",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect linked Dimensions endpoints and merge them into one publication-level parquet."
    )
    parser.add_argument("--publications", required=True, type=Path, help="Publication endpoint parquet.")
    parser.add_argument("--output", required=True, type=Path, help="Output wide parquet.")
    parser.add_argument("--data-dir", type=Path, help="Relational endpoint and link parquet directory.")
    parser.add_argument("--checkpoint-dir", type=Path, help="Resumable API checkpoint directory.")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--sleep", type=float, default=2.1)
    parser.add_argument("--request-retries", type=int, default=5)
    return parser.parse_args()


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
        time.sleep(min(60.0, 2.0 ** attempt))
    raise RuntimeError("Dimensions request failed.")


def authenticate(api_key, retries):
    response = request_json(AUTH_URL, payload={"key": api_key}, retries=retries)
    token = response.get("token")
    if not token:
        raise RuntimeError("Dimensions authentication returned no token.")
    return token


def run_dsl(token, query, retries):
    return request_json(QUERY_URL, token=token, query=query, retries=retries)


def available_fields(token, source, retries):
    response = run_dsl(token, f"describe source {source}", retries)
    metadata = response.get("fields") or {}
    if isinstance(metadata, dict):
        fields = []
        for name, details in metadata.items():
            details = details if isinstance(details, dict) else {}
            if not details.get("deprecated") and not details.get("is_deprecated"):
                fields.append(name)
    else:
        fields = [
            item["name"]
            for item in metadata
            if item.get("name") and not item.get("deprecated") and not item.get("is_deprecated")
        ]
    fields = [field for field in fields if field != "score"]
    return ["id"] + sorted(set(fields) - {"id"})


def parse_collection(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "null"}:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return []
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        return [parsed]
    return []


def compact(value):
    if value is None:
        return None
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def normalized_record(record, fields):
    return {field: compact(record.get(field)) for field in fields}


def batches(values, size):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def batch_signature(source, query_field, fields, values):
    payload = json.dumps([source, query_field, fields, values], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_completed(path):
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def query_batch(token, source, query_field, values, fields, retries, sleep):
    quoted = ",".join(json.dumps(value) for value in values)
    field_expression = "+".join(fields)
    page_size = 1000
    while True:
        records = []
        offset = 0
        retry_with_smaller_pages = False
        while True:
            query = (
                f"search {source} where {query_field} in [{quoted}] "
                f"return {source}[{field_expression}] limit {page_size} skip {offset}"
            )
            response = run_dsl(token, query, retries)
            warnings = response.get("warnings") or response.get("_warnings") or []
            if warnings:
                if page_size == 1:
                    raise RuntimeError(f"Dimensions returned a truncation warning for {source}: {warnings}")
                page_size = max(1, page_size // 2)
                retry_with_smaller_pages = True
                break
            page = response.get(source) or []
            records.extend(page)
            if len(page) < page_size:
                return records
            offset += page_size
            if sleep > 0:
                time.sleep(sleep)
        if not retry_with_smaller_pages:
            return records


def collect_query(token, source, stage, query_field, values, fields, checkpoint_dir, batch_size, sleep, retries):
    values = sorted({str(value) for value in values if value is not None and str(value)})
    raw_path = checkpoint_dir / f"{source}.jsonl"
    done_path = checkpoint_dir / f"{stage}.done"
    completed = read_completed(done_path)
    total = (len(values) + batch_size - 1) // batch_size
    with raw_path.open("a", encoding="utf-8") as raw_handle, done_path.open("a", encoding="utf-8") as done_handle:
        for number, batch in enumerate(batches(values, batch_size), start=1):
            signature = batch_signature(source, query_field, fields, batch)
            if signature in completed:
                print(f"{stage} {number}/{total}: checkpoint", flush=True)
                continue
            records = query_batch(token, source, query_field, batch, fields, retries, sleep)
            for record in records:
                if record.get("id"):
                    raw_handle.write(json.dumps(normalized_record(record, fields), ensure_ascii=False) + "\n")
            raw_handle.flush()
            done_handle.write(signature + "\n")
            done_handle.flush()
            completed.add(signature)
            print(f"{stage} {number}/{total}: requested={len(batch)} returned={len(records)}", flush=True)
            if sleep > 0:
                time.sleep(sleep)


def load_records(path, fields):
    records = {}
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    if record.get("id"):
                        records[str(record["id"])] = record
    if not records:
        return pd.DataFrame(columns=fields)
    return pd.DataFrame(records.values()).reindex(columns=fields).sort_values("id").reset_index(drop=True)


def publication_ids_from_field(publications, field):
    values = set()
    pairs = []
    for publication_id, raw_value in publications[["id", field]].itertuples(index=False, name=None):
        for value in parse_collection(raw_value):
            endpoint_id = str(value)
            values.add(endpoint_id)
            pairs.append((str(publication_id), endpoint_id, f"publications.{field}"))
    return sorted(values), pairs


def publication_object_ids(publications, field):
    values = set()
    pairs = []
    for publication_id, raw_value in publications[["id", field]].itertuples(index=False, name=None):
        for value in parse_collection(raw_value):
            if isinstance(value, dict) and value.get("id"):
                endpoint_id = str(value["id"])
                values.add(endpoint_id)
                pairs.append((str(publication_id), endpoint_id, f"publications.{field}"))
    return sorted(values), pairs


def reverse_pairs(endpoint, link_field, publication_ids, source):
    pairs = []
    if endpoint.empty:
        return pairs
    for endpoint_id, raw_value in endpoint[["id", link_field]].itertuples(index=False, name=None):
        for value in parse_collection(raw_value):
            publication_id = str(value)
            if publication_id in publication_ids:
                pairs.append((publication_id, str(endpoint_id), f"{source}.{link_field}"))
    return pairs


def single_reverse_pairs(endpoint, link_field, publication_ids, source):
    pairs = []
    if endpoint.empty:
        return pairs
    for endpoint_id, raw_value in endpoint[["id", link_field]].itertuples(index=False, name=None):
        publication_id = "" if pd.isna(raw_value) else str(raw_value)
        if publication_id in publication_ids:
            pairs.append((publication_id, str(endpoint_id), f"{source}.{link_field}"))
    return pairs


def link_frame(pairs, link_id):
    grouped = {}
    for publication_id, endpoint_id, source in pairs:
        grouped.setdefault((publication_id, endpoint_id), set()).add(source)
    rows = [
        {
            "publication_id": publication_id,
            link_id: endpoint_id,
            "link_sources": json.dumps(sorted(sources), separators=(",", ":")),
        }
        for (publication_id, endpoint_id), sources in sorted(grouped.items())
    ]
    return pd.DataFrame(rows, columns=["publication_id", link_id, "link_sources"])


def save_parquet(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, engine="pyarrow", compression="zstd")


def collect_endpoint(token, name, config, publications, publication_ids, data_dir, checkpoint_dir, args):
    fields = available_fields(token, name, args.request_retries)
    required = {"id", config["query_field"]}
    if config.get("link_field"):
        required.add(config["link_field"])
    missing = required - set(fields)
    if missing:
        raise RuntimeError(f"{name} does not expose required fields: {sorted(missing)}")
    (checkpoint_dir / f"{name}_fields.json").write_text(json.dumps(fields, indent=2) + "\n", encoding="utf-8")
    seed_pairs = []
    mode = config["mode"]
    if mode == "publication_ids":
        values, seed_pairs = publication_ids_from_field(publications, config["publication_field"])
    elif mode == "publication_object_ids":
        values, seed_pairs = publication_object_ids(publications, config["publication_field"])
    else:
        values = sorted(publication_ids)
    collect_query(
        token,
        name,
        f"{name}_primary",
        config["query_field"],
        values,
        fields,
        checkpoint_dir,
        args.batch_size,
        args.sleep,
        args.request_retries,
    )
    if name == "clinical_trials" and "clinical_trial_ids" in publications.columns:
        direct_ids, direct_pairs = publication_ids_from_field(publications, "clinical_trial_ids")
        seed_pairs.extend(direct_pairs)
        collect_query(
            token,
            name,
            "clinical_trials_direct_ids",
            "id",
            direct_ids,
            fields,
            checkpoint_dir,
            args.batch_size,
            args.sleep,
            args.request_retries,
        )
    endpoint = load_records(checkpoint_dir / f"{name}.jsonl", fields)
    if mode == "reverse":
        seed_pairs.extend(reverse_pairs(endpoint, config["link_field"], publication_ids, name))
    elif mode == "single_reverse":
        seed_pairs.extend(single_reverse_pairs(endpoint, config["link_field"], publication_ids, name))
    links = link_frame(seed_pairs, config["link_id"])
    save_parquet(endpoint, data_dir / f"{name}.parquet")
    save_parquet(links, data_dir / f"links_publication_{name}.parquet")
    return endpoint, links


def decoded_value(value):
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, str) and value[:1] in {"[", "{"}:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def json_array(values):
    return json.dumps(values, ensure_ascii=False, separators=(",", ":"))


def endpoint_wide(publication_ids, name, config, endpoint, links):
    link_id = config["link_id"]
    endpoint_records = {
        str(record["id"]): record
        for record in endpoint.to_dict(orient="records")
    }
    linked = (
        links.groupby("publication_id")[link_id].apply(lambda values: sorted(set(values.astype(str)))).to_dict()
        if not links.empty
        else {}
    )
    rows = []
    for publication_id in publication_ids:
        linked_ids = linked.get(publication_id, [])
        records = [endpoint_records[endpoint_id] for endpoint_id in linked_ids if endpoint_id in endpoint_records]
        row = {
            "id": publication_id,
            f"{name}__linked_ids": json_array(linked_ids) if linked_ids else "",
            f"{name}__n_links": len(linked_ids),
            f"{name}__n_records": len(records),
        }
        for field in endpoint.columns:
            row[f"{name}__{field}"] = (
                json_array([decoded_value(record.get(field)) for record in records]) if records else ""
            )
        rows.append(row)
    return pd.DataFrame(rows)


def validate_publications(publications):
    if "id" not in publications.columns:
        raise RuntimeError("Publication parquet must contain an id column.")
    if publications["id"].isna().any():
        raise RuntimeError("Publication parquet contains missing ids.")
    if publications["id"].duplicated().any():
        raise RuntimeError("Publication parquet contains duplicate ids.")
    invalid = ~publications["id"].astype(str).str.fullmatch(r"pub\.\d+")
    if invalid.any():
        raise RuntimeError(f"Publication parquet contains {int(invalid.sum())} invalid Dimensions publication ids.")
    required = {"supporting_grant_ids", "source_title"}
    missing = required - set(publications.columns)
    if missing:
        raise RuntimeError(f"Publication parquet is missing linkage fields: {sorted(missing)}")


def main():
    args = parse_args()
    api_key = os.environ.get(API_KEY_ENV, "").strip()
    if not api_key:
        raise RuntimeError(f"Set {API_KEY_ENV} before running the pipeline.")
    if args.batch_size < 1:
        raise RuntimeError("--batch-size must be at least 1.")
    output = args.output.resolve()
    data_dir = (args.data_dir or output.with_name(f"{output.stem}_tables")).resolve()
    checkpoint_dir = (args.checkpoint_dir or data_dir / ".checkpoints").resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    publications = pd.read_parquet(args.publications)
    validate_publications(publications)
    publications = publications.copy()
    publications["id"] = publications["id"].astype(str)
    publication_ids = publications["id"].tolist()
    publication_id_set = set(publication_ids)
    save_parquet(publications, data_dir / "publications.parquet")
    token = authenticate(api_key, args.request_retries)
    collected = {}
    statistics = {"publications": {"records": len(publications), "links": len(publications)}}
    for name, config in ENDPOINTS.items():
        endpoint, links = collect_endpoint(
            token,
            name,
            config,
            publications,
            publication_id_set,
            data_dir,
            checkpoint_dir,
            args,
        )
        collected[name] = (endpoint, links)
        statistics[name] = {
            "records": len(endpoint),
            "links": len(links),
            "linked_publications": int(links["publication_id"].nunique()) if not links.empty else 0,
        }
    wide = publications.copy()
    for name, config in ENDPOINTS.items():
        endpoint, links = collected[name]
        aggregate = endpoint_wide(publication_ids, name, config, endpoint, links)
        wide = wide.merge(aggregate, on="id", how="left", validate="one_to_one", sort=False)
    if len(wide) != len(publications) or wide["id"].nunique() != len(publications):
        raise RuntimeError("Wide merge changed the publication row count or id uniqueness.")
    if wide.columns.duplicated().any():
        raise RuntimeError("Wide merge produced duplicate column names.")
    output.parent.mkdir(parents=True, exist_ok=True)
    save_parquet(wide, output)
    statistics["wide_output"] = {
        "path": str(output),
        "rows": len(wide),
        "columns": len(wide.columns),
        "unique_publication_ids": int(wide["id"].nunique()),
    }
    stats_path = output.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(statistics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(statistics, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
