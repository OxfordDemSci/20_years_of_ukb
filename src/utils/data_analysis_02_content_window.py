"""Cohort-aware caches and provenance for the content-analysis topic models."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from utils.shared_analysis_window import ANALYSIS_END_DATE, ANALYSIS_END_YEAR


def validate_training_years(years):
    """Reject missing/future training years before model or cache access."""
    values = pd.to_numeric(pd.Series(list(years)), errors="coerce")
    if (values.empty or values.isna().any()
            or not (values.between(1, ANALYSIS_END_YEAR) & values.mod(1).eq(0)).all()):
        raise ValueError(
            f"Topic training requires dated papers through {ANALYSIS_END_DATE.date()} only."
        )
    return values


def topic_corpus_hash(ids, docs, years):
    """Key the complete ordered training input, including its publication cutoff."""
    years = validate_training_years(years)
    digest = hashlib.sha256(str(ANALYSIS_END_DATE.date()).encode())
    for paper_id, text, year in zip(ids, docs, years, strict=True):
        digest.update(json.dumps([str(paper_id), str(text), float(year)], ensure_ascii=False).encode())
        digest.update(b"\n")
    return digest.hexdigest()[:20]


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_topic_window_provenance(path, training_years):
    """Bind an exported topic table to the dates used to train its model."""
    years = validate_training_years(training_years)
    path = Path(path)
    provenance = {
        "analysis_end_date": str(ANALYSIS_END_DATE.date()),
        "training_min_year": int(years.min()),
        "training_max_year": int(years.max()),
        "training_documents": len(years),
        "artifact_sha256": _file_hash(path),
    }
    path.with_suffix(".analysis_window.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )


def require_topic_window_provenance(path):
    """Old topic tables cannot be made valid by hiding post-cutoff rows."""
    path = Path(path)
    sidecar = path.with_suffix(".analysis_window.json")
    message = (
        f"Outdated or unverified topic cache {path.name}: training must exclude papers after "
        f"{ANALYSIS_END_DATE.date()}. Rerun the filtered BERTopic analysis and retain "
        "the table's .analysis_window.json sidecar."
    )
    if not sidecar.is_file():
        raise FileNotFoundError(message)
    try:
        provenance = json.loads(sidecar.read_text(encoding="utf-8"))
        cutoff = pd.Timestamp(provenance["analysis_end_date"])
        valid = (
            cutoff <= ANALYSIS_END_DATE
            and int(provenance["training_max_year"]) <= ANALYSIS_END_YEAR
            and int(provenance["training_documents"]) > 0
            and provenance["artifact_sha256"] == _file_hash(path)
        )
    except (ValueError, TypeError, KeyError):
        valid = False
    if not valid:
        raise ValueError(message)
