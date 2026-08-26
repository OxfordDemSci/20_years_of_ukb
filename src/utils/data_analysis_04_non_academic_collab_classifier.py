"""Institution-string parsing for the non-academic collaboration classifier.

WHY THIS EXISTS
---------------
This parser decides what text is sent to the LLM classifier, so it silently sets the
ceiling on classification quality: an affiliation it fails to extract can never be
labelled, and a department fragment it extracts instead of the institution gets labelled
as neither academic nor non-academic. It lived as a notebook cell, which meant the run
that produced `output/non_academic_flagged.xlsx` could not be reproduced without the
notebook's execution state. It is pure -- stdlib only, no globals, no I/O -- so it is
testable on its own, and that is the point of moving it.

WHAT IT HANDLES
---------------
`_try_parse` tries **JSON first, then `ast.literal_eval`**. That order matters: the
showcase-plus parquet stores nested columns as JSON (`null`/`true`/`false`), which
`ast.literal_eval` rejects. A literal-eval-first parser returns None on every parquet row
and the caller's `if not isinstance(...)` guard then drops the row without raising -- the
failure `src/utils/shared_showcase.py` documents at length. Keep JSON first.

Entry point: `build_institution_list(row)` -- a pandas row of affiliation-ish columns in,
a deduplicated list of at most MAX_INSTITUTIONS_PER_ROW institution strings out.

Extracted from `src/data_analysis/04_non_academic_05_collab_classifier.ipynb`
(cell 10) on 2026-08-26, unchanged apart from this docstring.
"""

from __future__ import annotations

import ast
import json
import re

INSTITUTION_KEYS = {
    "name",
    "institution",
    "organization",
    "organisation",
    "org",
    "affiliation",
    "affiliations",
    "raw_affiliation",
}

KEYWORDS = (
    "university",
    "college",
    "institute",
    "institut",
    "school",
    "hospital",
    "clinic",
    "centre",
    "center",
    "foundation",
    "trust",
    "ministry",
    "government",
    "council",
    "company",
    "inc",
    "ltd",
    "llc",
    "gmbh",
    "corp",
    "plc",
    "sa",
    "srl",
    "bv",
    "ag",
    "laboratory",
    "lab",
)

MAX_INSTITUTIONS_PER_ROW = 200
MAX_CHARS_PER_INSTITUTION = 150

_SPLIT_RE = re.compile(r"[;|]+")


def _normalize_text(s):
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\S+@\S+", "", s)
    s = re.sub(r"https?://\S+", "", s)
    return s.strip(" ,;|")


def _contains_keyword(s):
    s = s.lower()
    return any(k in s for k in KEYWORDS)


def _try_parse(s):
    s = s.strip()
    if not s or s[0] not in "[{":
        return None
    try:
        return json.loads(s)
    except Exception:
        try:
            return ast.literal_eval(s)
        except Exception:
            return None


def _extract_strings(obj):
    strings = []
    if obj is None:
        return strings
    if isinstance(obj, (list, tuple, set)):
        for item in obj:
            strings.extend(_extract_strings(item))
        return strings
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in INSTITUTION_KEYS:
                strings.extend(_extract_strings(value))
        return strings
    strings.append(str(obj))
    return strings


def _select_institution_segments(s):
    cleaned = _normalize_text(s)
    if not cleaned:
        return []
    parts = [p.strip() for p in _SPLIT_RE.split(cleaned) if p.strip()]
    selected = []
    for part in parts:
        comma_parts = [p.strip() for p in part.split(",") if p.strip()]
        if comma_parts:
            keyword_part = next((p for p in comma_parts if _contains_keyword(p)), None)
            part = keyword_part or comma_parts[0]
        part = _normalize_text(part)
        if part:
            if len(part) > MAX_CHARS_PER_INSTITUTION:
                part = part[:MAX_CHARS_PER_INSTITUTION].rstrip()
            selected.append(part)
    if not any(_contains_keyword(p) for p in selected):
        selected = selected[:2]
    return selected


def _extract_institutions_from_value(value):
    if value is None or (isinstance(value, float) and str(value) == "nan"):
        return []
    if isinstance(value, str):
        parsed = _try_parse(value)
        if parsed is not None:
            strings = _extract_strings(parsed)
        else:
            strings = [value]
    else:
        strings = _extract_strings(value)

    candidates = []
    for s in strings:
        candidates.extend(_select_institution_segments(s))
    return candidates


def _dedupe_keep_order(items):
    seen = set()
    out = []
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def build_institution_list(row):
    candidates = []
    for value in row:
        candidates.extend(_extract_institutions_from_value(value))
    candidates = _dedupe_keep_order(candidates)
    return candidates[:MAX_INSTITUTIONS_PER_ROW]


# =============================================================================
# PART 2 — THE CLASSIFICATION ENGINE (Anthropic / Claude)
# =============================================================================
# Ported from the OpenAI (`gpt-5-nano`) implementation that lived in the
# classifier notebook, on 2026-08-26. Four things changed and all four matter:
#
#   1. Provider. `anthropic.Anthropic()` + `claude-opus-5` replaces `OpenAI()`.
#   2. Output shape is now *guaranteed* rather than hoped for. The old code had
#      ~200 lines of JSON-repair (`_extract_json_fragment`, `_parse_json_object`,
#      a strict-retry, and a batch-splitting fallback) because the model could
#      return prose around the JSON. `output_config.format` with a json_schema
#      makes the first content block valid JSON against the schema, so all of
#      that machinery is gone rather than ported.
#   3. Concurrency is a thread pool, not asyncio. The SDK is sync, the work is
#      IO-bound, and a pool is testable without an event loop.
#   4. The cache is unchanged in spirit: keyed on the institution tuple, so a
#      re-run costs nothing for rows already seen. Re-running must not re-bill.
#
# CREDENTIALS: `anthropic.Anthropic()` resolves ANTHROPIC_API_KEY, then
# ANTHROPIC_AUTH_TOKEN, then an `ant auth login` profile. A Claude Code session
# is NOT a usable credential here — it has no API key to lend. See
# doc/04_non_academic_methodology.md §5.2.

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_BATCH_SIZE = 20
DEFAULT_MAX_CONCURRENCY = 8
DEFAULT_EFFORT = "low"          # bulk string classification; see note below
DEFAULT_MAX_TOKENS = 16000

# Effort is the cost lever, not the model. `low` is deliberate: this is
# short-string classification against an explicit rubric, not reasoning. Raise it
# to "medium" if spot-checks show borderline institutions being mislabelled, and
# record that as a decision — do not change it silently.

SYSTEM_PROMPT = (
    "You classify institution strings as academic organizations, non-academic organizations, "
    "companies, and UK-based companies. Only label entries that are clearly organizations. "
    "Ignore locations (cities, countries, states), departments, faculties, and generic terms "
    "like 'Education', 'Facility', or 'Healthcare'. "
    "Academic: universities, colleges, degree-granting schools, and explicitly "
    "university-affiliated hospitals. "
    "Non-academic: companies, industry, private sector, NGOs, charities, foundations, "
    "government agencies, and non-university hospitals or clinics. "
    "Company: a private-sector, for-profit company specifically (a subset of non-academic). "
    "UK company: a company clearly based in the United Kingdom "
    "(explicit UK/United Kingdom/England/Scotland/Wales/Northern Ireland signal, "
    "or a well-known UK company). "
    "If you are unsure about UK status or organizational status, do NOT label it."
)

# Every index refers to a position in that row's institution list. The schema is
# closed (`additionalProperties: false`) so a malformed response is a 400 from
# the API rather than a parse failure three functions downstream.
_ROW_PROPS = {
    "row": {"type": "integer", "description": "the row number given in the prompt"},
    "academic_indices": {"type": "array", "items": {"type": "integer"}},
    "non_academic_indices": {"type": "array", "items": {"type": "integer"}},
    "company_indices": {"type": "array", "items": {"type": "integer"}},
    "uk_company_indices": {"type": "array", "items": {"type": "integer"}},
}
BATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "rows": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": _ROW_PROPS,
                "required": list(_ROW_PROPS),
                "additionalProperties": False,
            },
        }
    },
    "required": ["rows"],
    "additionalProperties": False,
}

INDEX_FIELDS = ("academic", "non_academic", "company", "uk_company")


def build_batch_prompt(batch: Sequence[Sequence[str]]) -> str:
    """Render one batch of institution lists as a numbered prompt."""
    lines = [
        "Classify the institutions in each row below. Indices are 0-based and "
        "refer to positions within that row's list. Return one object per row, "
        "including rows where every list is empty.",
        "",
    ]
    for row_no, institutions in enumerate(batch):
        lines.append(f"Row {row_no}:")
        if not institutions:
            lines.append("  (no institutions)")
        for i, name in enumerate(institutions):
            lines.append(f"  {i}. {name}")
        lines.append("")
    return "\n".join(lines)


def _empty_result() -> dict[str, list[int]]:
    return {f"{f}_indices": [] for f in INDEX_FIELDS}


def classify_batch(client, batch: Sequence[Sequence[str]], *,
                   model: str = DEFAULT_MODEL,
                   effort: str = DEFAULT_EFFORT,
                   max_tokens: int = DEFAULT_MAX_TOKENS) -> list[dict[str, list[int]]]:
    """Classify one batch. Returns one dict of index lists per input row.

    Indices out of range for their row are dropped rather than trusted — the
    schema constrains the shape, not the values.
    """
    import json

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_batch_prompt(batch)}],
        output_config={
            "effort": effort,
            "format": {"type": "json_schema", "schema": BATCH_SCHEMA},
        },
    )
    if response.stop_reason == "refusal":
        detail = getattr(response, "stop_details", None)
        raise RuntimeError(
            f"classification refused (category={getattr(detail, 'category', None)})"
        )

    text = next(b.text for b in response.content if b.type == "text")
    payload = json.loads(text)

    out = [_empty_result() for _ in batch]
    for rec in payload.get("rows", []):
        row_no = rec.get("row")
        if not isinstance(row_no, int) or not (0 <= row_no < len(batch)):
            continue
        n = len(batch[row_no])
        for field in INDEX_FIELDS:
            key = f"{field}_indices"
            seen = {i for i in rec.get(key, []) if isinstance(i, int) and 0 <= i < n}
            out[row_no][key] = sorted(seen)
    return out


def _read_cache(path: Path) -> dict[tuple[str, ...], dict[str, list[int]]]:
    import json

    cache: dict[tuple[str, ...], dict[str, list[int]]] = {}
    if not path.exists():
        return cache
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                cache[tuple(rec["key"])] = rec["result"]
            except Exception:
                continue          # a torn last line from an interrupted run
    return cache


def _append_cache(path: Path, items: list[tuple[tuple[str, ...], dict]]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for key, result in items:
            fh.write(json.dumps({"key": list(key), "result": result}) + "\n")
        fh.flush()
        os.fsync(fh.fileno())     # a killed run must not lose paid-for results


def classify_institution_lists(
    institution_lists: Sequence[Sequence[str]],
    *,
    cache_path: Path | None = None,
    client=None,
    model: str = DEFAULT_MODEL,
    effort: str = DEFAULT_EFFORT,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    progress: Callable[[int, int], None] | None = None,
) -> list[dict[str, list[int]]]:
    """Classify every row, de-duplicated and cached.

    Rows with identical institution tuples are classified once. Anything already
    in `cache_path` costs nothing. Returns one result per input row, in order.
    """
    keys = [tuple(row) for row in institution_lists]
    cache = _read_cache(cache_path) if cache_path else {}

    todo = [k for k in dict.fromkeys(keys) if k and k not in cache]
    n_unique, n_cached = len(set(k for k in keys if k)), len(cache)
    print(f"  {len(keys):,} rows | {n_unique:,} distinct institution sets | "
          f"{n_cached:,} cached | {len(todo):,} to classify")

    if todo:
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        batches = [todo[i:i + batch_size] for i in range(0, len(todo), batch_size)]
        done = 0
        with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
            futures = {
                pool.submit(classify_batch, client, b, model=model, effort=effort): b
                for b in batches
            }
            for fut in as_completed(futures):
                batch = futures[fut]
                try:
                    results = fut.result()
                except Exception as exc:
                    # Report the type and the batch, never the raw traceback:
                    # affiliation strings are schema-only data (doc/data/showcase_plus.md).
                    print(f"  batch of {len(batch)} failed: {type(exc).__name__}")
                    continue
                if cache_path:
                    _append_cache(cache_path, list(zip(batch, results)))
                cache.update(dict(zip(batch, results)))
                done += len(batch)
                if progress:
                    progress(done, len(todo))

    return [cache.get(k, _empty_result()) for k in keys]


# ---------------------------------------------------------------- frame level

# The institution list handed to the model is built from `research_orgs`, in
# order, and NOTHING else. That is not a convenience — it is what makes the
# output usable. The downstream taxonomy
# (data_analysis_04_non_academic_collab_helpers.add_non_academic_sector_taxonomy
# -> build_row_records) treats every returned index as a position in
# `research_orgs`, and reads that org's GRID `types` and country to assign one of
# the 8 sectors. An index list built from any other column, or from a merged set
# of columns, silently indexes into the wrong list and mislabels sectors without
# raising. The old notebook selected its columns with a name-matching heuristic
# (any column matching institution/affiliation/organization/org) and did exactly
# that.
#
# The cost of the rule: an affiliation that appears only in `authors[].affiliations`
# and never reached `research_orgs` is invisible to the classifier. That is a real
# limit, recorded in doc/04_non_academic_methodology.md, not a thing to fix by
# widening the input list.

FLAG_COLUMNS = (
    "academic_flag", "non_academic_flag", "company_flag", "uk_company_flag",
    "uk_company_only_flag", "academic_uk_company_collab_flag",
)


def institution_lists_from_research_orgs(df, column: str = "research_orgs") -> list[list[str]]:
    """One list of org names per row, positionally aligned with `research_orgs`."""
    out: list[list[str]] = []
    for cell in df[column]:
        names = []
        for org in parse_listcol_local(cell):
            name = org.get("name") if isinstance(org, dict) else org
            names.append(_normalize_text(str(name)) if name else "")
        out.append(names)
    return out


def parse_listcol_local(x):
    """List-cell parser: already-a-list wins, else JSON, else Python literal."""
    if isinstance(x, list):
        return x
    parsed = _try_parse(x) if isinstance(x, str) else None
    return parsed if isinstance(parsed, list) else []


def attach_classification(df, results: Sequence[dict], institution_lists: Sequence[Sequence[str]]):
    """Add the index lists, the name lists and the flags to a copy of `df`."""
    out = df.copy()
    out["author_institutions_list"] = [list(x) for x in institution_lists]

    for field in INDEX_FIELDS:
        key = f"{field}_indices"
        idx = [r[key] for r in results]
        out[key] = idx
        out[f"{field}_institutions"] = [
            [names[i] for i in ix if i < len(names) and names[i]]
            for ix, names in zip(idx, institution_lists)
        ]

    for field in INDEX_FIELDS:
        out[f"{field}_flag"] = [bool(x) for x in out[f"{field}_indices"]]

    # A paper whose only non-academic partner is a UK company, and a paper where a
    # UK company and an academic institution appear together. Both are asked for
    # downstream and both are cheaper to define once, here.
    out["uk_company_only_flag"] = [
        bool(uk) and not (set(na) - set(uk))
        for uk, na in zip(out["uk_company_indices"], out["non_academic_indices"])
    ]
    out["academic_uk_company_collab_flag"] = [
        bool(uk) and bool(ac)
        for uk, ac in zip(out["uk_company_indices"], out["academic_indices"])
    ]
    return out


def load_or_classify(
    df,
    out_path: Path,
    *,
    cache_path: Path | None = None,
    force: bool = False,
    **kwargs: Any,
):
    """Return the classified frame, running the LLM pass only if it has to.

    If `out_path` already exists it is read and returned unchanged — the classified
    file is the expensive artefact and re-deriving it must never be accidental.
    Pass `force=True` to reclassify anyway (the per-row cache still applies, so
    this is cheap unless the cache is also gone).
    """
    import pandas as pd

    out_path = Path(out_path)
    if out_path.exists() and not force:
        loaded = (pd.read_excel(out_path) if out_path.suffix in {".xlsx", ".xls"}
                  else pd.read_csv(out_path))
        print(f"  reusing {out_path.name} ({len(loaded):,} rows) — no API calls, "
              f"no cost. Pass force=True to reclassify.")
        return loaded, False

    institution_lists = institution_lists_from_research_orgs(df)
    n_with = sum(1 for x in institution_lists if any(x))
    print(f"  classifying: {len(df):,} rows, {n_with:,} ({n_with/len(df):.1%}) "
          f"carrying >=1 research org")

    results = classify_institution_lists(
        institution_lists, cache_path=cache_path, **kwargs
    )
    out = attach_classification(df, results, institution_lists)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_frame(out, out_path)
    print(f"  wrote {out_path}")
    return out, True


def _write_frame(frame, path: Path) -> None:
    """Write xlsx if asked for, else csv. List columns are JSON-encoded so they
    survive the round trip — a Python repr would not reload as JSON."""
    import json as _json

    to_write = frame.copy()
    for col in to_write.columns:
        if to_write[col].map(lambda v: isinstance(v, (list, dict))).any():
            to_write[col] = to_write[col].map(
                lambda v: _json.dumps(v) if isinstance(v, (list, dict)) else v
            )
    if path.suffix in {".xlsx", ".xls"}:
        to_write.to_excel(path, index=False)
    else:
        to_write.to_csv(path, index=False)
