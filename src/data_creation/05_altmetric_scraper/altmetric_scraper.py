#!/usr/bin/env python3
# ruff: noqa: I001
"""Collect Altmetric Explorer data for the showcase_plus publication PIDs.

The Explorer API accepts up to 25,000 scholarly identifiers in a reusable
identifier list. This collector selects one PID per source row (DOI, then PMID,
arXiv ID, then ISBN), creates as many lists as required, downloads every result
page, and writes the five-column CSV consumed by the non-academic-impact
analysis.

The repository stores Explorer credentials separately. The API key identifies
the account, while the API secret signs each request and is never transmitted.
Both can also be overridden on the command line with file paths; credential
values themselves are deliberately not accepted as arguments.

Example::

    python src/data_creation/05_altmetric_scraper/altmetric_scraper.py

Default inputs and outputs::

    input:   data/showcase/showcase_plus/showcase_plus_all_endpoints_wide.parquet
    output:  data/altmetric/altmetric.csv
    audit:   data/altmetric/altmetric_explorer_audit.csv
    cache:   data/altmetric/altmetric_explorer_cache.jsonl
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import hmac
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


def _find_repo_root() -> Path:
    candidates = [
        Path.cwd(),
        *Path.cwd().parents,
        Path(__file__).resolve().parent,
        *Path(__file__).resolve().parents,
    ]
    for candidate in candidates:
        if (candidate / "src" / "utils").is_dir():
            return candidate
    raise RuntimeError(
        "Could not locate the repository root (expected src/utils). "
        "Run this script from inside the twenty_years_of_ukb checkout."
    )


ROOT = _find_repo_root()
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from utils import shared_paths as P


EXPLORER_API_BASE_URL = "https://www.altmetric.com/explorer/api"
DEFAULT_AUDIT = P.ALTMETRIC_CSV.parent / "altmetric_explorer_audit.csv"
DEFAULT_CACHE = P.ALTMETRIC_CSV.parent / "altmetric_explorer_cache.jsonl"
DEFAULT_KEY_FILE = next(
    (
        path
        for path in (
            ROOT / "keys" / "altmetric.txt",
            ROOT / "keys" / "altmetric_key.txt",
        )
        if path.is_file()
    ),
    ROOT / "keys" / "altmetric.txt",
)
DEFAULT_SECRET_FILE = ROOT / "keys" / "altmetric_secret.txt"

OUTPUT_COLUMNS = (
    "DOI",
    "Altmetric Attention Score",
    "News mentions",
    "Policy mentions",
    "Publication Date",
)

DEFAULT_PID_SPECS = (
    "doi=doi",
    "pmid=pmid",
    "arxiv=arxiv_id",
    "isbn=isbn",
)

PID_TYPE_ALIASES = {
    "doi": "doi",
    "pmid": "pmid",
    "pubmed": "pmid",
    "pubmed_id": "pmid",
    "arxiv": "arxiv",
    "arxiv_id": "arxiv",
    "isbn": "isbn",
}

RESPONSE_IDENTIFIER_KEYS = {
    "doi": ("dois", "doi"),
    "pmid": ("pubmed", "pubmed-ids", "pubmed_ids", "pmids", "pmid"),
    "arxiv": ("arxiv", "arxiv_ids"),
    "isbn": ("isbns", "isbn"),
}

AUTHORIZATION_ERROR_CODES = {
    "expired_api_key",
    "invalid_api_key",
    "invalid_digest",
    "missing_api_key",
    "no_api_access",
}


class AltmetricError(RuntimeError):
    """Base class for collection failures."""


class AltmetricAuthorizationError(AltmetricError):
    """Explorer credentials are invalid, expired, or not API-enabled."""


class AltmetricTemporaryError(AltmetricError):
    """A transient API or network failure persisted after retries."""


@dataclass(frozen=True)
class PidSpec:
    pid_type: str
    column: str


@dataclass(frozen=True)
class PidChoice:
    pid_type: str
    canonical_value: str
    submitted_value: str

    @property
    def key(self) -> str:
        return pid_key(self.pid_type, self.canonical_value)


class RateLimiter:
    def __init__(self, requests_per_second: float) -> None:
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        self.minimum_interval = 1.0 / requests_per_second
        self.last_request_started: float | None = None

    def wait(self) -> None:
        now = time.monotonic()
        if self.last_request_started is not None:
            remaining = self.minimum_interval - (now - self.last_request_started)
            if remaining > 0:
                time.sleep(remaining)
        self.last_request_started = time.monotonic()


class JsonlCache:
    """Append-only API cache; the final occurrence of a key is authoritative."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.entries: dict[str, Any] = {}
        self.hits = 0
        self.writes = 0
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    cache_key = str(entry["cache_key"])
                    payload = entry["payload"]
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"Malformed cache entry at {self.path}:{line_number}: {exc}"
                    ) from exc
                self.entries[cache_key] = payload

    def get(self, cache_key: str, *, refresh: bool) -> Any:
        if refresh or cache_key not in self.entries:
            return None
        self.hits += 1
        return self.entries[cache_key]

    def put(self, cache_key: str, payload: Any) -> None:
        entry = {
            "cache_key": cache_key,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "payload": payload,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(serialized + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.entries[cache_key] = payload
        self.writes += 1


def _payload_error(payload: Mapping[str, Any]) -> tuple[set[str], str]:
    errors = payload.get("errors")
    if not isinstance(errors, list):
        return set(), ""
    codes: set[str] = set()
    messages: list[str] = []
    for error in errors:
        if not isinstance(error, Mapping):
            continue
        code = str(error.get("code", "")).strip()
        detail = str(error.get("detail") or error.get("title") or "").strip()
        if code:
            codes.add(code)
        if code or detail:
            messages.append(": ".join(part for part in (code, detail) if part))
    return codes, "; ".join(messages)


class ExplorerClient:
    """Small Altmetric Explorer API client with signed requests and retries."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        *,
        base_url: str,
        requests_per_second: float,
        timeout: float,
        retries: int,
    ) -> None:
        if not api_key.strip() or not api_secret.strip():
            raise AltmetricAuthorizationError("Altmetric API credentials are empty.")
        self.api_key = api_key.strip()
        self.api_secret = api_secret.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.rate_limiter = RateLimiter(requests_per_second)

    def _digest(self, content: str, *, strip_secret_hyphens: bool = False) -> str:
        secret = self.api_secret.replace("-", "") if strip_secret_hyphens else self.api_secret
        return hmac.new(
            secret.encode("utf-8"),
            content.encode("utf-8"),
            hashlib.sha1,
        ).hexdigest()

    @staticmethod
    def _retry_delay(headers: Mapping[str, str], attempt: int) -> float:
        retry_after = headers.get("Retry-After")
        if retry_after:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                pass
        return min(120.0, (2.0**attempt) + random.uniform(0.0, 1.0))

    def _request_json(
        self,
        method: str,
        url: str,
        *,
        form: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        body = urlencode(form).encode("utf-8") if form is not None else None
        headers = {
            "Accept": "application/json",
            "User-Agent": "twenty-years-of-ukb-altmetric-collector/2.0",
        }
        if body is not None:
            headers["Content-Type"] = "application/x-www-form-urlencoded"

        for attempt in range(self.retries + 1):
            self.rate_limiter.wait()
            request = Request(url, data=body, headers=headers, method=method)
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    raw = response.read().decode("utf-8")
                    try:
                        payload = json.loads(raw)
                    except json.JSONDecodeError as exc:
                        if attempt == self.retries:
                            raise AltmetricTemporaryError(
                                f"Altmetric returned invalid JSON: {exc}"
                            ) from exc
                        time.sleep(self._retry_delay(response.headers, attempt))
                        continue
                    if not isinstance(payload, dict):
                        raise AltmetricError("Altmetric returned a non-object JSON response.")
                    codes, message = _payload_error(payload)
                    if codes & AUTHORIZATION_ERROR_CODES:
                        raise AltmetricAuthorizationError(message)
                    if codes:
                        raise AltmetricError(message)
                    return payload

            except HTTPError as exc:
                raw = exc.read().decode("utf-8", errors="replace")
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    payload = {}
                codes, message = _payload_error(payload)
                detail = message or re.sub(r"\s+", " ", raw).strip()[:500]
                if codes & AUTHORIZATION_ERROR_CODES or exc.code in {401, 403}:
                    raise AltmetricAuthorizationError(
                        detail or f"Altmetric returned HTTP {exc.code}."
                    ) from exc
                if exc.code == 429 or 500 <= exc.code <= 599:
                    if attempt < self.retries:
                        time.sleep(self._retry_delay(exc.headers, attempt))
                        continue
                    raise AltmetricTemporaryError(
                        f"Altmetric repeatedly returned HTTP {exc.code}: {detail}"
                    ) from exc
                raise AltmetricError(
                    f"Altmetric returned HTTP {exc.code}: {detail}"
                ) from exc

            except (URLError, TimeoutError, OSError) as exc:
                if attempt < self.retries:
                    time.sleep(self._retry_delay({}, attempt))
                    continue
                raise AltmetricTemporaryError(
                    f"Altmetric request failed repeatedly: {exc}"
                ) from exc

        raise AssertionError("unreachable")

    def create_identifier_list(self, identifiers: Sequence[str]) -> dict[str, Any]:
        content = "\n".join(identifiers)
        digest = self._digest(content, strip_secret_hyphens=True)
        payload = self._request_json(
            "POST",
            f"{self.base_url}/identifier_lists",
            form={"key": self.api_key, "digest": digest, "identifiers": content},
        )
        data = payload.get("data")
        if not isinstance(data, dict) or not data.get("id"):
            raise AltmetricError("Identifier-list response did not contain a list ID.")
        return data

    def research_outputs_page(
        self,
        identifier_list_id: str,
        *,
        page_number: int,
        page_size: int,
    ) -> dict[str, Any]:
        signing_content = f"identifier_list_id|{identifier_list_id}|scope|all"
        query = urlencode(
            {
                "key": self.api_key,
                "digest": self._digest(signing_content),
                "filter[identifier_list_id]": identifier_list_id,
                "filter[scope]": "all",
                "page[number]": page_number,
                "page[size]": page_size,
            }
        )
        return self._request_json(
            "GET",
            f"{self.base_url}/research_outputs?{query}",
        )


def canonical_pid_type(value: str) -> str:
    key = value.strip().lower().replace("-", "_")
    try:
        return PID_TYPE_ALIASES[key]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported Explorer identifier type {value!r}; use DOI, PMID, arXiv, or ISBN."
        ) from exc


def parse_pid_specs(values: Iterable[str]) -> list[PidSpec]:
    specs: list[PidSpec] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --pid {value!r}; expected TYPE=COLUMN.")
        raw_type, raw_column = value.split("=", 1)
        column = raw_column.strip()
        if not column:
            raise ValueError(f"Invalid --pid {value!r}; column name is empty.")
        spec = PidSpec(canonical_pid_type(raw_type), column)
        if spec not in specs:
            specs.append(spec)
    if not specs:
        raise ValueError("At least one --pid specification is required.")
    return specs


def available_columns(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Reading parquet requires pyarrow.") from exc
        return list(pq.ParquetFile(path).schema.names)
    if suffix in {".csv", ".txt"}:
        return list(pd.read_csv(path, nrows=0).columns)
    raise ValueError(f"Unsupported input format {path.suffix!r}; use .parquet or .csv.")


def resolve_pid_specs(
    requested: Sequence[str] | None,
    columns: Sequence[str],
) -> list[PidSpec]:
    available = set(columns)
    specs = parse_pid_specs(requested or DEFAULT_PID_SPECS)
    if requested:
        missing = [spec.column for spec in specs if spec.column not in available]
        if missing:
            raise ValueError(f"PID columns absent from input: {sorted(set(missing))}")
        return specs
    present = [spec for spec in specs if spec.column in available]
    if not present:
        raise ValueError("None of the default PID columns exist in the input.")
    return present


def read_source(path: Path, columns: Sequence[str], max_rows: int | None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(path, columns=list(columns))
        if max_rows is not None:
            frame = frame.head(max_rows)
    elif suffix in {".csv", ".txt"}:
        frame = pd.read_csv(path, usecols=list(columns), nrows=max_rows)
    else:
        raise ValueError(f"Unsupported input format {path.suffix!r}; use .parquet or .csv.")
    return frame.reset_index(drop=True)


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def expand_cell(value: Any) -> list[Any]:
    if is_missing(value):
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if not isinstance(value, str) and hasattr(value, "tolist"):
        converted = value.tolist()
        if isinstance(converted, list):
            return converted
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                try:
                    parsed = ast.literal_eval(stripped)
                except (MemoryError, RecursionError, SyntaxError, ValueError):
                    parsed = None
            if isinstance(parsed, (list, tuple, set)):
                return list(parsed)
        return [stripped]
    return [value]


def scalar_text(value: Any) -> str:
    if is_missing(value):
        return ""
    if isinstance(value, bool):
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return str(int(value)) if value.is_integer() else str(value)
    return str(value).strip()


def normalize_pid(pid_type: str, value: Any) -> str:
    text = scalar_text(value)
    if not text:
        return ""
    if pid_type == "doi":
        text = re.sub(r"^doi:\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(
            r"^https?://(?:dx\.)?doi\.org/", "", text, flags=re.IGNORECASE
        ).strip().lower()
        if not text.startswith("10.") or "/" not in text or re.search(r"\s", text):
            return ""
        return text
    if pid_type == "pmid":
        text = re.sub(r"^pmid:\s*", "", text, flags=re.IGNORECASE).strip()
        return text if re.fullmatch(r"\d+", text) else ""
    if pid_type == "arxiv":
        text = re.sub(r"^arxiv:\s*", "", text, flags=re.IGNORECASE).strip().lower()
        return text if text and not re.search(r"\s", text) else ""
    if pid_type == "isbn":
        text = re.sub(
            r"^isbn(?:-1[03])?:\s*", "", text, flags=re.IGNORECASE
        ).strip()
        text = re.sub(r"[\s-]+", "", text).upper()
        return text if re.fullmatch(r"[0-9X]+", text) else ""
    raise ValueError(f"Unsupported PID type: {pid_type}")


def submission_value(pid_type: str, canonical_value: str) -> str:
    if pid_type == "arxiv":
        return f"arXiv:{canonical_value}"
    return canonical_value


def pid_key(pid_type: str, canonical_value: str) -> str:
    return f"{pid_type}\u001f{canonical_value}"


def row_pid_choices(row: pd.Series, specs: Sequence[PidSpec]) -> list[PidChoice]:
    choices: list[PidChoice] = []
    seen: set[str] = set()
    for spec in specs:
        for raw_value in expand_cell(row.get(spec.column)):
            canonical = normalize_pid(spec.pid_type, raw_value)
            if not canonical:
                continue
            choice = PidChoice(
                spec.pid_type,
                canonical,
                submission_value(spec.pid_type, canonical),
            )
            if choice.key not in seen:
                seen.add(choice.key)
                choices.append(choice)
    return choices


def choose_pid(row: pd.Series, specs: Sequence[PidSpec]) -> PidChoice | None:
    choices = row_pid_choices(row, specs)
    return choices[0] if choices else None


def json_safe_value(value: Any) -> Any:
    if is_missing(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def source_date_fallback(
    row: pd.Series,
    publication_date_column: str | None,
    year_column: str | None,
) -> str:
    if publication_date_column:
        parsed = pd.to_datetime(row.get(publication_date_column), errors="coerce")
        if not pd.isna(parsed):
            return parsed.date().isoformat()
    if year_column:
        try:
            year = int(float(row.get(year_column)))
        except (TypeError, ValueError, OverflowError):
            year = 0
        if 1000 <= year <= 9999:
            return f"{year:04d}-01-01"
    return ""


def prepare_assignments(
    source: pd.DataFrame,
    specs: Sequence[PidSpec],
    *,
    publication_date_column: str | None,
    year_column: str | None,
) -> tuple[pd.DataFrame, list[str]]:
    audit_rows: list[dict[str, Any]] = []
    unique_tokens: dict[str, str] = {}
    input_columns = list(dict.fromkeys(spec.column for spec in specs))
    if "id" in source.columns and "id" not in input_columns:
        input_columns.insert(0, "id")

    for source_row, row in source.iterrows():
        choices = row_pid_choices(row, specs)
        choice = choices[0] if choices else None
        source_doi = next(
            (item.canonical_value for item in choices if item.pid_type == "doi"),
            "",
        )
        audit_row = {
            "source_row": source_row,
            **{
                f"input_{column}": json_safe_value(row.get(column))
                for column in input_columns
            },
            "submitted_pid_type": choice.pid_type if choice else "",
            "submitted_pid": choice.submitted_value if choice else "",
            "source_date_fallback": source_date_fallback(
                row, publication_date_column, year_column
            ),
            "_pid_key": choice.key if choice else "",
            "_all_pid_keys": tuple(item.key for item in choices),
            "_source_doi": source_doi,
        }
        audit_rows.append(audit_row)
        if choice is not None:
            unique_tokens.setdefault(choice.key, choice.submitted_value)

    return pd.DataFrame(audit_rows), list(unique_tokens.values())


def chunked(values: Sequence[str], size: int) -> list[list[str]]:
    return [list(values[start : start + size]) for start in range(0, len(values), size)]


def cached_identifier_list(
    client: ExplorerClient,
    cache: JsonlCache,
    identifiers: Sequence[str],
    *,
    refresh: bool,
) -> dict[str, Any]:
    content_hash = hashlib.sha256("\n".join(identifiers).encode("utf-8")).hexdigest()
    cache_key = f"identifier-list:{content_hash}"
    cached = cache.get(cache_key, refresh=refresh)
    if cached is not None:
        if not isinstance(cached, dict) or not cached.get("id"):
            raise RuntimeError(f"Invalid cached identifier list: {cache_key}")
        return cached
    data = client.create_identifier_list(identifiers)
    sanitized = {"id": str(data["id"]), "counts": data.get("counts", {})}
    cache.put(cache_key, sanitized)
    return sanitized


def _normalize_page(payload: Mapping[str, Any]) -> dict[str, Any]:
    data = payload.get("data")
    if not isinstance(data, list):
        raise AltmetricError("Research-outputs response did not contain a data list.")
    links = payload.get("links")
    meta = payload.get("meta")
    response_meta = meta.get("response", {}) if isinstance(meta, Mapping) else {}
    return {
        "data": data,
        "has_next": bool(links.get("next")) if isinstance(links, Mapping) else False,
        "total_results": response_meta.get("total-results"),
        "total_pages": response_meta.get("total-pages"),
    }


def fetch_identifier_list(
    client: ExplorerClient,
    cache: JsonlCache,
    identifier_list_id: str,
    *,
    batch_number: int,
    page_size: int,
    progress_every: int,
    refresh: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    page_number = 1
    while True:
        cache_key = f"research-outputs:{identifier_list_id}:{page_size}:{page_number}"
        page = cache.get(cache_key, refresh=refresh)
        if page is None:
            page = _normalize_page(
                client.research_outputs_page(
                    identifier_list_id,
                    page_number=page_number,
                    page_size=page_size,
                )
            )
            cache.put(cache_key, page)
        if not isinstance(page, dict) or not isinstance(page.get("data"), list):
            raise AltmetricError(f"Invalid cached research-output page: {cache_key}")

        for record in page["data"]:
            if isinstance(record, dict):
                records.append(
                    {"record": record, "batch": batch_number, "page": page_number}
                )

        if progress_every > 0 and (
            page_number % progress_every == 0 or not page.get("has_next")
        ):
            total_pages = page.get("total_pages") or "?"
            print(
                f"batch {batch_number}: page {page_number}/{total_pages}, "
                f"records {len(records):,}",
                flush=True,
            )
        if not page.get("has_next"):
            break
        page_number += 1
        if page_number > 1_000_000:
            raise RuntimeError("Pagination exceeded the safety limit.")
    return records


def response_pid_keys(record: Mapping[str, Any]) -> set[str]:
    attributes = record.get("attributes")
    if not isinstance(attributes, Mapping):
        return set()
    identifiers = attributes.get("identifiers")
    if not isinstance(identifiers, Mapping):
        return set()
    keys: set[str] = set()
    for pid_type, response_keys in RESPONSE_IDENTIFIER_KEYS.items():
        for response_key in response_keys:
            if response_key not in identifiers:
                continue
            for value in expand_cell(identifiers.get(response_key)):
                canonical = normalize_pid(pid_type, value)
                if canonical:
                    keys.add(pid_key(pid_type, canonical))
    return keys


def first_response_doi(record: Mapping[str, Any]) -> str:
    attributes = record.get("attributes")
    identifiers = attributes.get("identifiers", {}) if isinstance(attributes, Mapping) else {}
    if not isinstance(identifiers, Mapping):
        return ""
    for key in RESPONSE_IDENTIFIER_KEYS["doi"]:
        for value in expand_cell(identifiers.get(key)):
            doi = normalize_pid("doi", value)
            if doi:
                return doi
    return ""


def safe_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def safe_nonnegative_int(value: Any) -> int:
    try:
        number = int(float(value))
    except (TypeError, ValueError, OverflowError):
        return 0
    return max(0, number)


def build_outputs(
    fetched: Sequence[Mapping[str, Any]],
    assignments: pd.DataFrame,
    *,
    allow_missing_doi: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    unique_records: dict[str, Mapping[str, Any]] = {}
    duplicate_api_records = 0
    for wrapped in fetched:
        record = wrapped.get("record")
        if not isinstance(record, Mapping):
            continue
        altmetric_id = scalar_text(record.get("id"))
        key = f"altmetric:{altmetric_id}" if altmetric_id else f"record:{len(unique_records)}"
        if key in unique_records:
            duplicate_api_records += 1
            continue
        unique_records[key] = record

    source_rows_by_pid: dict[str, set[int]] = {}
    for assignment_index, row in assignments.iterrows():
        for key in row["_all_pid_keys"]:
            source_rows_by_pid.setdefault(str(key), set()).add(int(assignment_index))

    altmetric_ids_by_assignment: dict[int, set[str]] = {}
    output_rows: list[dict[str, Any]] = []
    included_altmetric_ids: set[str] = set()
    seen_dois: set[str] = set()
    records_without_doi = 0
    unmapped_records = 0

    for record in unique_records.values():
        altmetric_id = scalar_text(record.get("id"))
        record_pid_keys = response_pid_keys(record)
        matched_assignment_indices: set[int] = set()
        for key in record_pid_keys:
            matched_assignment_indices.update(source_rows_by_pid.get(key, set()))
        if not matched_assignment_indices:
            unmapped_records += 1
            continue
        for assignment_index in matched_assignment_indices:
            altmetric_ids_by_assignment.setdefault(assignment_index, set()).add(
                altmetric_id
            )

        attributes = record.get("attributes")
        if not isinstance(attributes, Mapping):
            attributes = {}
        mentions = attributes.get("mentions")
        if not isinstance(mentions, Mapping):
            mentions = {}
        api_doi = first_response_doi(record)
        source_dois = list(
            dict.fromkeys(
                str(assignments.at[index, "_source_doi"])
                for index in sorted(matched_assignment_indices)
                if assignments.at[index, "_source_doi"]
            )
        )
        doi = api_doi if api_doi in source_dois or not source_dois else source_dois[0]
        if not doi and not allow_missing_doi:
            records_without_doi += 1
            continue
        if doi and doi in seen_dois:
            duplicate_api_records += 1
            continue

        publication_date = scalar_text(attributes.get("publication-date"))
        if not publication_date:
            publication_date = next(
                (
                    str(assignments.at[index, "source_date_fallback"])
                    for index in sorted(matched_assignment_indices)
                    if assignments.at[index, "source_date_fallback"]
                ),
                "",
            )
        output_rows.append(
            {
                "DOI": doi,
                "Altmetric Attention Score": safe_float(attributes.get("altmetric-score")),
                "News mentions": safe_nonnegative_int(mentions.get("msm")),
                "Policy mentions": safe_nonnegative_int(mentions.get("policy")),
                "Publication Date": publication_date,
            }
        )
        if altmetric_id:
            included_altmetric_ids.add(altmetric_id)
        if doi:
            seen_dois.add(doi)

    output = pd.DataFrame(output_rows, columns=OUTPUT_COLUMNS)
    if not output.empty:
        output = output.sort_values(
            ["Publication Date", "DOI"], kind="stable", na_position="last"
        ).reset_index(drop=True)

    audit = assignments.copy()
    audit["altmetric_ids"] = [
        ";".join(sorted(altmetric_ids_by_assignment.get(index, set())))
        for index in audit.index
    ]
    audit["lookup_status"] = audit.apply(
        lambda row: (
            "no_valid_pid"
            if not row["_pid_key"]
            else "found"
            if row["altmetric_ids"]
            else "not_found"
        ),
        axis=1,
    )
    audit["output_included"] = audit["altmetric_ids"].map(
        lambda values: any(
            value in included_altmetric_ids for value in str(values).split(";") if value
        )
    )
    audit = audit.drop(columns=["_pid_key", "_all_pid_keys", "_source_doi"])

    counters = {
        "raw_api_records": len(fetched),
        "unique_api_records": len(unique_records),
        "duplicate_api_records": duplicate_api_records,
        "api_records_without_doi": records_without_doi,
        "api_records_unmapped_to_source_pid": unmapped_records,
        "matched_source_rows": int((audit["lookup_status"] == "found").sum()),
        "unmatched_source_rows": int((audit["lookup_status"] == "not_found").sum()),
        "output_rows": len(output),
    }
    return output, audit, counters


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def read_credential(path: Path, label: str) -> str:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Altmetric {label} file not found: {resolved}")
    value = resolved.read_text(encoding="utf-8").strip()
    if not value:
        raise AltmetricAuthorizationError(f"Altmetric {label} file is empty: {resolved}")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect Altmetric Explorer metrics for PIDs in the showcase_plus parquet."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=P.SHOWCASE_PLUS,
        help=f"Source parquet or CSV (default: {P.raw_path(P.SHOWCASE_PLUS)}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=P.ALTMETRIC_CSV,
        help=f"Final metrics CSV (default: {P.raw_path(P.ALTMETRIC_CSV)}).",
    )
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--api-key-file",
        type=Path,
        default=DEFAULT_KEY_FILE,
        help=f"Explorer API key file (default: {P.raw_path(DEFAULT_KEY_FILE)}).",
    )
    parser.add_argument(
        "--api-secret-file",
        type=Path,
        default=DEFAULT_SECRET_FILE,
        help=f"Explorer API secret file (default: {P.raw_path(DEFAULT_SECRET_FILE)}).",
    )
    parser.add_argument(
        "--pid",
        action="append",
        metavar="TYPE=COLUMN",
        help=(
            "PID fallback specification; repeat to set order. Defaults to "
            "doi=doi, pmid=pmid, arxiv=arxiv_id, isbn=isbn."
        ),
    )
    parser.add_argument("--publication-date-column")
    parser.add_argument("--year-column", default="year")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument(
        "--identifier-batch-size",
        type=int,
        default=25_000,
        help="PIDs per Explorer identifier list (default and API maximum: 25000).",
    )
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument(
        "--requests-per-second",
        type=float,
        default=2.0,
        help="Maximum request start rate (Altmetric asks Explorer clients to use <=2).",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N result pages; 0 disables periodic output.",
    )
    parser.add_argument(
        "--allow-missing-doi",
        action="store_true",
        help="Include API records without a DOI in the final CSV.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore cached identifier lists and result pages.",
    )
    parser.add_argument("--base-url", default=EXPLORER_API_BASE_URL, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.max_rows is not None and args.max_rows < 1:
        raise ValueError("--max-rows must be at least 1.")
    if not 1 <= args.identifier_batch_size <= 25_000:
        raise ValueError("--identifier-batch-size must be between 1 and 25000.")
    if not 1 <= args.page_size <= 100:
        raise ValueError("--page-size must be between 1 and 100.")
    if args.requests_per_second <= 0 or args.requests_per_second > 2:
        raise ValueError("--requests-per-second must be greater than 0 and no more than 2.")
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive.")
    if args.retries < 0:
        raise ValueError("--retries cannot be negative.")
    if args.progress_every < 0:
        raise ValueError("--progress-every cannot be negative.")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    columns = available_columns(input_path)
    specs = resolve_pid_specs(args.pid, columns)
    requested_columns = list(dict.fromkeys(spec.column for spec in specs))
    if "id" in columns:
        requested_columns.append("id")
    publication_date_column = args.publication_date_column
    if publication_date_column:
        if publication_date_column not in columns:
            raise ValueError(
                f"Publication-date column {publication_date_column!r} is absent."
            )
        requested_columns.append(publication_date_column)
    year_column: str | None = args.year_column
    if year_column and year_column in columns:
        requested_columns.append(year_column)
    else:
        year_column = None
    requested_columns = list(dict.fromkeys(requested_columns))

    source = read_source(input_path, requested_columns, args.max_rows)
    assignments, identifiers = prepare_assignments(
        source,
        specs,
        publication_date_column=publication_date_column,
        year_column=year_column,
    )
    if not identifiers:
        raise ValueError("No valid Explorer-supported PIDs were found in the input.")

    api_key = read_credential(args.api_key_file, "API key")
    api_secret = read_credential(args.api_secret_file, "API secret")
    client = ExplorerClient(
        api_key,
        api_secret,
        base_url=args.base_url,
        requests_per_second=args.requests_per_second,
        timeout=args.timeout,
        retries=args.retries,
    )
    cache = JsonlCache(args.cache)
    batches = chunked(identifiers, args.identifier_batch_size)

    print("input:", input_path)
    print("source rows:", f"{len(source):,}")
    print("unique submitted PIDs:", f"{len(identifiers):,}")
    print("PID fallback order:", " -> ".join(f"{s.pid_type}={s.column}" for s in specs))
    print("identifier-list batches:", len(batches))
    print("output:", args.output)
    print("cache:", args.cache)

    fetched: list[dict[str, Any]] = []
    recognized_counts: dict[str, int] = {}
    list_ids: list[str] = []
    for batch_number, batch in enumerate(batches, start=1):
        list_data = cached_identifier_list(
            client, cache, batch, refresh=args.refresh
        )
        list_id = str(list_data["id"])
        list_ids.append(list_id)
        counts = list_data.get("counts")
        if isinstance(counts, Mapping):
            for key, value in counts.items():
                recognized_counts[str(key)] = recognized_counts.get(str(key), 0) + safe_nonnegative_int(value)
        print(
            f"batch {batch_number}/{len(batches)}: {len(batch):,} PIDs, list {list_id}",
            flush=True,
        )
        fetched.extend(
            fetch_identifier_list(
                client,
                cache,
                list_id,
                batch_number=batch_number,
                page_size=args.page_size,
                progress_every=args.progress_every,
                refresh=args.refresh,
            )
        )

    output, audit, output_summary = build_outputs(
        fetched,
        assignments,
        allow_missing_doi=args.allow_missing_doi,
    )
    atomic_write_csv(output.loc[:, list(OUTPUT_COLUMNS)], args.output)
    atomic_write_csv(audit, args.audit_output)

    summary: dict[str, Any] = {
        "input_rows": len(source),
        "unique_submitted_pids": len(identifiers),
        "identifier_list_batches": len(batches),
        "identifier_list_ids": list_ids,
        "recognized_identifiers": recognized_counts,
        "cache_hits": cache.hits,
        "cache_writes": cache.writes,
        **output_summary,
    }
    print("\ncollection summary")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("saved:", args.output)
    print("saved:", args.audit_output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        AltmetricError,
        FileNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
