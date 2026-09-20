"""Recover complete endpoint records embedded in the local Showcase+ snapshot.

The acquisition pipeline's ``endpoint_wide`` stores each endpoint field as a
parallel JSON array. These are complete fetched records, unlike ``linked_ids``,
which can also include entities whose metadata were never fetched. Recovery
checks array alignment and repeated records before writing a deduplicated CSV.
It never reconstructs metadata from links or overwrites an existing source CSV.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from utils import shared_paths as P
from utils.shared_analysis_window import filter_analysis_window
from utils.shared_showcase import parse_listcol


ENDPOINT_DATE_COLUMNS = {
    "patents": ("publication_year", "publication_date"),
    "clinical_trials": ("start_year", "start_date"),
    "policy_documents": ("year", None),
}


def filter_endpoint_links(corpus: pd.DataFrame, *, wide: pd.DataFrame | None = None,
                          source: Path | None = None) -> pd.DataFrame:
    """Limit reverse links to entities dated within the analysis window.

    Publication dates and outcome dates are separate: an eligible paper can cite
    or be cited by an outcome beyond the cutoff. Only links with dated endpoint
    metadata qualify. The source snapshot and its metadata arrays are unchanged.
    """
    result = corpus.copy()
    endpoints = [name for name in ENDPOINT_DATE_COLUMNS if name + "__linked_ids" in result]
    if not endpoints:
        return result
    if wide is None:
        source = Path(source) if source is not None else P.SHOWCASE_PLUS
        available = set(pq.read_schema(source).names)
        columns = []
        for name in endpoints:
            year, date = ENDPOINT_DATE_COLUMNS[name]
            columns.extend(name + "__" + field for field in ("id", "n_records", year, date)
                           if field and name + "__" + field in available)
        wide = pd.read_parquet(source, columns=columns)
    for name in endpoints:
        year, date = ENDPOINT_DATE_COLUMNS[name]
        wanted = [name + "__" + field for field in ("id", "n_records", year, date)
                  if field and name + "__" + field in wide]
        records = recover_endpoint_records(wide[wanted], name)
        eligible = set(filter_analysis_window(records, year_col=year, date_col=date)["id"])
        column = name + "__linked_ids"
        result[column] = result[column].apply(
            lambda values: [entity_id for entity_id in parse_listcol(values) if entity_id in eligible]
        )
        if name + "__n_links" in result:
            result[name + "__n_links"] = result[column].map(len)
    return result


def recover_endpoint_records(wide: pd.DataFrame, endpoint: str) -> pd.DataFrame:
    """Invert aligned endpoint arrays, rejecting incomplete or conflicting data."""
    prefix = endpoint + "__"
    count_col = prefix + "n_records"
    id_col = prefix + "id"
    fields = [c for c in wide.columns if c.startswith(prefix)
              and c[len(prefix):] not in {"linked_ids", "n_links", "n_records"}]
    if id_col not in fields or count_col not in wide:
        raise ValueError(f"{endpoint}: complete endpoint records are absent; linkage IDs are insufficient")
    counts = pd.to_numeric(wide[count_col], errors="coerce")
    invalid = ((wide[count_col].notna() & counts.isna()) | counts.lt(0)
               | (counts.notna() & counts.mod(1).ne(0)))
    if invalid.any():
        raise ValueError(f"{endpoint}: invalid record count {wide.loc[invalid, count_col].iloc[0]!r}")
    # A zero/missing count must not hide populated arrays from a malformed export.
    for row in wide.loc[counts.fillna(0).eq(0), fields].to_dict("records"):
        for column, value in row.items():
            if isinstance(value, str):
                if not value.strip():
                    continue  # The wide export uses blank strings for absent endpoint fields.
                try:
                    value = json.loads(value)
                except ValueError as exc:
                    raise ValueError(f"{column}: invalid endpoint JSON array") from exc
            if value is None or value is pd.NA or (isinstance(value, float) and pd.isna(value)):
                continue
            if not isinstance(value, list) or value:
                raise ValueError(f"{column}: populated or invalid array with zero/missing n_records")
    records = {}
    for row in wide.loc[counts.gt(0), fields + [count_col]].to_dict("records"):
        count = row[count_col]
        if int(count) != count or count < 0:
            raise ValueError(f"{endpoint}: invalid record count {count!r}")
        arrays = {}
        for column in fields:
            value = row[column]
            try:
                values = json.loads(value) if isinstance(value, str) else value
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{column}: invalid endpoint JSON array") from exc
            if not isinstance(values, list) or len(values) != count:
                raise ValueError(f"{column}: array length does not match n_records={count}")
            arrays[column[len(prefix):]] = values
        for index in range(int(count)):
            record = {field: values[index] for field, values in arrays.items()}
            entity_id = record["id"]
            if not isinstance(entity_id, str) or not entity_id.strip():
                raise ValueError(f"{endpoint}: endpoint record has no valid ID")
            if entity_id in records and records[entity_id] != record:
                raise ValueError(f"{endpoint}: conflicting copies of {entity_id}")
            records[entity_id] = record
    if not records:
        raise ValueError(f"{endpoint}: no full endpoint records found in the snapshot")
    return pd.DataFrame([records[key] for key in sorted(records)])


def ensure_endpoint_csv(endpoint: str, destination: Path, source: Path | None = None) -> Path:
    """Use the existing CSV, otherwise unpack full records with source provenance."""
    destination = Path(destination)
    if destination.exists():
        return destination
    source = Path(source) if source is not None else P.SHOWCASE_PLUS
    if not source.exists():
        raise FileNotFoundError(f"Missing {P.raw_path(destination)} and the full Showcase+ snapshot")
    prefix = endpoint + "__"
    columns = [name for name in pq.read_schema(source).names if name.startswith(prefix)]
    records = recover_endpoint_records(pd.read_parquet(source, columns=columns), endpoint)
    # Keep the source snapshot's nested lists/dicts intact in CSV-readable form.
    serialized = records.map(
        lambda value: json.dumps(value, ensure_ascii=False)
        if isinstance(value, (list, dict)) else value
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized.to_csv(destination, index=False)
    stat = source.stat()
    provenance = {
        "source": P.raw_path(source),
        "source_size_bytes": stat.st_size,
        "source_modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "recovered_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "records": len(records),
        "method": "Unpack complete parallel endpoint arrays; deduplicate identical records by ID",
        "scope": "Fetched entity records in this snapshot; linked IDs without records are excluded",
    }
    destination.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Recovered {len(records):,} {endpoint} records from the local Showcase+ snapshot")
    return destination


def ensure_clinical_trials_csv() -> Path:
    return ensure_endpoint_csv("clinical_trials", P.CT_CSV)


def ensure_policy_csv() -> Path:
    return ensure_endpoint_csv("policy_documents", P.POLICY_CSV)
