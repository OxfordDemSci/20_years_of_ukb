"""Shared, offline, auditable name-category inference for every analysis.

The outputs are statistical name categories, not observed sex or self-identified
gender. Five packages with bundled local data each contribute at most one vote:
``gender_guesser``, ``gender_detector``, ``nomquamgender``, ``names_dataset`` and
``gename``. No API client, network request, or remotely populated cache is used.

The primary direct category accepts non-conflicting available votes. When libraries
disagree, a category is accepted only with at least three votes and a two-vote margin;
otherwise it remains Unknown. ``mostly_female`` and ``mostly_male`` from
``gender_guesser`` are deliberately collapsed to Female and Male in the expanded and
primary rules. Every package vote and ensemble decision is retained for audit.
"""

from __future__ import annotations

import csv
import math
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from functools import cache
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import pandas as pd

CLASSIFIED_CATEGORIES = frozenset({"Female", "Male"})
OFFLINE_MIN_PROBABILITY = 0.90
NOMQUAM_MIN_COUNT = 20
ENSEMBLE_MIN_MAJORITY_VOTES = 3
ENSEMBLE_MIN_VOTE_MARGIN = 2
OFFLINE_LIBRARIES = (
    ("gender_guesser", "gender-guesser", "name_gender_gender_guesser"),
    ("gender_detector", "gender-detector", "name_gender_gender_detector"),
    ("nomquamgender", "nomquamgender", "name_gender_nomquamgender"),
    ("names_dataset", "names-dataset", "name_gender_names_dataset"),
    ("gename", "gename", "name_gender_gename"),
)


@dataclass(frozen=True)
class LibraryPrediction:
    """One package's normalized local prediction."""

    category: str
    detail: str
    confidence: float | None = None
    count: int | None = None


@dataclass(frozen=True)
class NameGenderInference:
    """One direct inference with sensitivity stages and package provenance."""

    category: str
    dictionary_category: str
    strict_category: str
    detail: str
    dictionary_detail: str
    method: str
    query_name: str
    vote_count: int
    vote_margin: int
    vote_share: float | None
    source_count: int
    conflict: bool
    source_predictions: str
    gender_guesser_category: str
    gender_detector_category: str
    nomquamgender_category: str
    names_dataset_category: str
    gename_category: str
    nomquamgender_probability: float | None
    nomquamgender_count: int | None
    names_dataset_probability: float | None
    unknown_reason: str


@dataclass(frozen=True)
class NameCategorySummary:
    """Counts and percentages for nested Female/Male/Unknown categories."""

    total: int
    classified: int
    female: int
    male: int
    unknown: int
    coverage_percent: float
    female_percent_all: float
    female_percent_classified: float


def _dependency_error(package: str, error: Exception) -> ImportError:
    missing = ImportError(
        f"Offline name inference requires {package!r}. "
        "Install the pinned analysis dependencies with "
        "`python -m pip install -r requirements-analysis.txt`."
    )
    missing.__cause__ = error
    return missing


def offline_library_versions() -> pd.DataFrame:
    """Return pinned-package provenance without importing their large datasets."""
    rows = []
    for library, distribution, _ in OFFLINE_LIBRARIES:
        try:
            installed_version = version(distribution)
            available = True
        except PackageNotFoundError:
            installed_version = "not installed"
            available = False
        rows.append(
            {
                "library": library,
                "distribution": distribution,
                "version": installed_version,
                "available": available,
                "inference_mode": "offline bundled data",
            }
        )
    return pd.DataFrame(rows)


@cache
def _gender_guesser_detector():
    try:
        import gender_guesser.detector as gender_detector
    except ImportError as error:
        raise _dependency_error("gender-guesser", error)
    return gender_detector.Detector(case_sensitive=False)


@cache
def _nomquam_model():
    try:
        import nomquamgender as nqg
    except ImportError as error:
        raise _dependency_error("nomquamgender", error)
    return nqg.NBGC()


@cache
def _names_dataset():
    try:
        from names_dataset import NameDataset
    except ImportError as error:
        raise _dependency_error("names-dataset", error)
    # Loading surnames roughly doubles memory and cannot improve a given-name lookup.
    return NameDataset(load_first_names=True, load_last_names=False)


@cache
def _gename_table() -> dict[str, str]:
    try:
        from gename import Gender
    except ImportError as error:
        raise _dependency_error("gename", error)
    detector = Gender()
    rows = detector.cur.execute("SELECT name, gender FROM name_gender").fetchall()
    return {str(name).casefold(): str(category) for name, category in rows}


@cache
def _gender_detector_tables() -> dict[str, dict[str, str]]:
    """Load all four package datasets once, using the package's decision rule.

    ``gender-detector`` has a Python-2-era top-level import and reopens its CSV for
    every lookup. Importing its implementation module and indexing the bundled tables
    once preserves its classifier while making cohort-scale use practical. Latin-1 is
    required for the Argentina and Uruguay files shipped by version 0.1.0.
    """
    try:
        from gender_detector.gender_detector import GenderDetector
    except ImportError as error:
        raise _dependency_error("gender-detector", error)

    tables: dict[str, dict[str, str]] = {}
    for country in ("uk", "us", "ar", "uy"):
        detector = GenderDetector(country)
        table: dict[str, str] = {}
        with open(
            detector.country.file(),
            encoding="latin-1",
            newline="",
        ) as handle:
            rows = csv.reader(handle)
            next(rows, None)
            for row in rows:
                if len(row) < 5 or not row[0]:
                    continue
                try:
                    # The documented top-level API is broken on modern Python.
                    raw = detector._guess(row)
                except (TypeError, ValueError):
                    raw = None
                category = _normalise_category(raw, include_mostly=True)
                if category in CLASSIFIED_CATEGORIES:
                    table[row[0].casefold()] = category
        tables[country] = table
    return tables


def _letters(value: str) -> str:
    """Keep Unicode letters and intra-name punctuation."""
    return "".join(
        character
        for character in value
        if unicodedata.category(character).startswith("L") or character in "-'\u2019"
    )


def _ascii(value: str) -> str:
    """Return an ASCII approximation for Latin-script lookup tables."""
    decomposed = unicodedata.normalize("NFKD", value)
    return _letters(
        "".join(
            character
            for character in decomposed
            if not unicodedata.combining(character) and character.isascii()
        )
    )


def normalise_given_name(value) -> tuple[str, str]:
    """Return the first non-initial given token and its ASCII lookup form."""
    if not isinstance(value, str) or not value.strip():
        return "", ""
    for raw_token in value.split():
        token = _letters(raw_token)
        if sum(character.isalpha() for character in token) <= 1:
            continue
        return token, _ascii(token) or token
    return "", ""


def _normalise_category(raw, *, include_mostly: bool) -> str:
    label = str(raw or "").strip().casefold().replace("-", "_").replace(" ", "_")
    mapping = {
        "f": "Female",
        "female": "Female",
        "woman": "Female",
        "m": "Male",
        "male": "Male",
        "man": "Male",
    }
    if include_mostly:
        mapping.update(
            {
                "mostly_female": "Female",
                "mostly_male": "Male",
            }
        )
    return mapping.get(label, "Unknown")


def _candidates(token: str, query_name: str) -> tuple[str, ...]:
    values = [token]
    if query_name and query_name.casefold() != token.casefold():
        values.append(query_name)
    return tuple(dict.fromkeys(values))


def _gender_guesser_prediction(
    candidates: tuple[str, ...],
) -> tuple[LibraryPrediction, str, str, str]:
    first_raw = "unknown"
    for position, candidate in enumerate(candidates):
        raw = _gender_guesser_detector().get_gender(candidate)
        if position == 0:
            first_raw = raw
        expanded = _normalise_category(raw, include_mostly=True)
        if expanded in CLASSIFIED_CATEGORIES:
            strict = _normalise_category(raw, include_mostly=False)
            suffix = "" if position == 0 else ";ascii_retry"
            return (
                LibraryPrediction(expanded, f"{raw}{suffix}"),
                strict,
                expanded,
                raw,
            )
    return (
        LibraryPrediction("Unknown", first_raw),
        "Unknown",
        "Unknown",
        first_raw,
    )


def _gender_detector_prediction(candidates: tuple[str, ...]) -> LibraryPrediction:
    tables = _gender_detector_tables()
    for position, candidate in enumerate(candidates):
        by_country = {
            country: table.get(candidate.casefold(), "Unknown")
            for country, table in tables.items()
        }
        known = {
            value for value in by_country.values() if value in CLASSIFIED_CATEGORIES
        }
        details = ",".join(f"{key}={value}" for key, value in by_country.items())
        if len(known) == 1:
            suffix = "" if position == 0 else ";ascii_retry"
            return LibraryPrediction(known.pop(), details + suffix)
        if len(known) > 1:
            return LibraryPrediction("Unknown", "country_conflict;" + details)
    return LibraryPrediction("Unknown", "not_found")


def _nomquam_prediction(candidates: tuple[str, ...]) -> LibraryPrediction:
    for position, candidate in enumerate(candidates):
        used = candidate.casefold()
        record = _nomquam_model().reference.get(used)
        if record is None:
            continue
        sources, count, p_female = record[:3]
        if pd.isna(p_female):
            continue
        p_female = float(p_female)
        count = int(count)
        confidence = max(p_female, 1 - p_female)
        category = "Unknown"
        if count >= NOMQUAM_MIN_COUNT and confidence >= OFFLINE_MIN_PROBABILITY:
            category = "Female" if p_female >= 0.5 else "Male"
        suffix = "" if position == 0 else ";ascii_retry"
        detail = (
            f"used={used};p_female={p_female:.3f};sources={int(sources)};"
            f"count={count}{suffix}"
        )
        return LibraryPrediction(category, detail, confidence, count)
    return LibraryPrediction("Unknown", "not_found")


def _names_dataset_prediction(candidates: tuple[str, ...]) -> LibraryPrediction:
    for position, candidate in enumerate(candidates):
        # Direct access avoids deep-copying and expanding irrelevant country names.
        first_name = _names_dataset().first_names.get(candidate.strip().title()) or {}
        genders = first_name.get("gender") or {}
        female = float(genders.get("F", 0) or 0)
        male = float(genders.get("M", 0) or 0)
        confidence = max(female, male)
        if not genders:
            continue
        category = "Unknown"
        if confidence >= OFFLINE_MIN_PROBABILITY and female != male:
            category = "Female" if female > male else "Male"
        suffix = "" if position == 0 else ";ascii_retry"
        return LibraryPrediction(
            category,
            f"p_female={female:.3f};p_male={male:.3f}{suffix}",
            confidence,
        )
    return LibraryPrediction("Unknown", "not_found")


def _gename_prediction(candidates: tuple[str, ...]) -> LibraryPrediction:
    for position, candidate in enumerate(candidates):
        raw = _gename_table().get(candidate.casefold(), "U")
        category = _normalise_category(raw, include_mostly=True)
        if category in CLASSIFIED_CATEGORIES:
            suffix = "" if position == 0 else ";ascii_retry"
            return LibraryPrediction(category, f"{raw}{suffix}")
    return LibraryPrediction("Unknown", "not_found")


def _ensemble_decision(
    predictions: dict[str, LibraryPrediction],
) -> tuple[str, str, int, int, float | None, int, bool, str]:
    votes = [
        prediction.category
        for prediction in predictions.values()
        if prediction.category in CLASSIFIED_CATEGORIES
    ]
    source_count = len(votes)
    if not votes:
        return "Unknown", "unclassified", 0, 0, None, 0, False, "no_offline_prediction"

    counts = Counter(votes)
    ranked = counts.most_common()
    winner, winner_count = ranked[0]
    runner_up_count = ranked[1][1] if len(ranked) > 1 else 0
    margin = winner_count - runner_up_count
    conflict = len(counts) > 1
    vote_share = winner_count / source_count
    if not conflict:
        method = (
            "offline_single_source"
            if source_count == 1
            else "offline_ensemble_consensus"
        )
        return (
            winner,
            method,
            winner_count,
            margin,
            vote_share,
            source_count,
            False,
            "",
        )
    if (
        winner_count >= ENSEMBLE_MIN_MAJORITY_VOTES
        and margin >= ENSEMBLE_MIN_VOTE_MARGIN
    ):
        return (
            winner,
            "offline_ensemble_majority",
            winner_count,
            margin,
            vote_share,
            source_count,
            True,
            "",
        )
    return (
        "Unknown",
        "unclassified",
        winner_count,
        margin,
        vote_share,
        source_count,
        True,
        "offline_library_conflict",
    )


def infer_name_gender(value) -> NameGenderInference:
    """Infer one name using only package-bundled, offline data."""
    token, query_name = normalise_given_name(value)
    if not token:
        reason = (
            "missing_name"
            if not isinstance(value, str) or not value.strip()
            else "initials_only"
        )
        return NameGenderInference(
            "Unknown",
            "Unknown",
            "Unknown",
            reason,
            reason,
            "unclassified",
            "",
            0,
            0,
            None,
            0,
            False,
            "",
            "Unknown",
            "Unknown",
            "Unknown",
            "Unknown",
            "Unknown",
            None,
            None,
            None,
            reason,
        )

    candidates = _candidates(token, query_name)
    guesser, strict, expanded, dictionary_detail = _gender_guesser_prediction(
        candidates
    )
    predictions = {
        "gender_guesser": guesser,
        "gender_detector": _gender_detector_prediction(candidates),
        "nomquamgender": _nomquam_prediction(candidates),
        "names_dataset": _names_dataset_prediction(candidates),
        "gename": _gename_prediction(candidates),
    }
    (
        category,
        method,
        vote_count,
        vote_margin,
        vote_share,
        source_count,
        conflict,
        unknown_reason,
    ) = _ensemble_decision(predictions)
    source_predictions = " | ".join(
        f"{source}={prediction.category}[{prediction.detail}]"
        for source, prediction in predictions.items()
    )
    detail = (
        f"{method};votes={vote_count}/{source_count};margin={vote_margin};"
        f"conflict={str(conflict).lower()}"
    )
    return NameGenderInference(
        category,
        expanded,
        strict,
        detail,
        dictionary_detail,
        method,
        query_name,
        vote_count,
        vote_margin,
        vote_share,
        source_count,
        conflict,
        source_predictions,
        predictions["gender_guesser"].category,
        predictions["gender_detector"].category,
        predictions["nomquamgender"].category,
        predictions["names_dataset"].category,
        predictions["gename"].category,
        predictions["nomquamgender"].confidence,
        predictions["nomquamgender"].count,
        predictions["names_dataset"].confidence,
        unknown_reason,
    )


def classify_name_values(values) -> pd.DataFrame:
    """Classify distinct names once and map complete provenance to the input rows."""
    series = pd.Series(values, copy=False)
    unique = pd.unique(series.fillna(""))
    inferred = {value: infer_name_gender(value) for value in unique}
    rows = series.fillna("").map(lambda value: asdict(inferred[value]))
    result = pd.DataFrame(rows.tolist(), index=series.index)
    return result.rename(
        columns={
            "category": "name_gender_direct",
            "dictionary_category": "name_gender_dictionary",
            "strict_category": "name_gender_strict",
            "detail": "name_gender_detail",
            "dictionary_detail": "name_gender_dictionary_detail",
            "method": "name_gender_method",
            "query_name": "name_gender_query",
            "vote_count": "name_gender_vote_count",
            "vote_margin": "name_gender_vote_margin",
            "vote_share": "name_gender_vote_share",
            "source_count": "name_gender_source_count",
            "conflict": "name_gender_direct_conflict",
            "source_predictions": "name_gender_source_predictions",
            "gender_guesser_category": "name_gender_gender_guesser",
            "gender_detector_category": "name_gender_gender_detector",
            "nomquamgender_category": "name_gender_nomquamgender",
            "names_dataset_category": "name_gender_names_dataset",
            "gename_category": "name_gender_gename",
            "nomquamgender_probability": "name_gender_nomquamgender_probability",
            "nomquamgender_count": "name_gender_nomquamgender_count",
            "names_dataset_probability": "name_gender_names_dataset_probability",
            "unknown_reason": "name_gender_unknown_reason",
        }
    )


def classify_authorship_names(
    authorships: pd.DataFrame,
    *,
    name_col: str = "first_name",
    identity_col: str = "researcher_id",
    resolved_col: str = "identity_resolved",
) -> pd.DataFrame:
    """Classify authorships and propagate only unanimous identity evidence."""
    result = classify_name_values(authorships[name_col])
    result["name_gender"] = result["name_gender_direct"]
    result["name_gender_identity_conflict"] = False

    resolved = (
        authorships[resolved_col].fillna(False) & authorships[identity_col].notna()
    )
    evidence = pd.DataFrame(
        {
            "identity": authorships.loc[resolved, identity_col],
            "category": result.loc[resolved, "name_gender_direct"],
        }
    )
    evidence = evidence[evidence["category"].isin(CLASSIFIED_CATEGORIES)]
    sets = evidence.groupby("identity", observed=True)["category"].agg(
        lambda values: tuple(sorted(set(values)))
    )
    consensus = sets[sets.map(len).eq(1)].map(lambda values: values[0])
    conflicts = sets[sets.map(len).gt(1)].index

    propagated = authorships[identity_col].map(consensus)
    fill = resolved & result["name_gender"].eq("Unknown") & propagated.notna()
    result.loc[fill, "name_gender"] = propagated[fill]
    result.loc[fill, "name_gender_method"] = "researcher_identity_consensus"
    result.loc[fill, "name_gender_detail"] = "identity_consensus"
    result.loc[fill, "name_gender_unknown_reason"] = ""

    conflict = resolved & authorships[identity_col].isin(conflicts)
    result.loc[conflict, "name_gender"] = "Unknown"
    result.loc[conflict, "name_gender_method"] = "unclassified"
    result.loc[conflict, "name_gender_detail"] = "identity_conflict"
    result.loc[conflict, "name_gender_unknown_reason"] = "identity_conflict"
    result.loc[conflict, "name_gender_identity_conflict"] = True
    for column in [
        "name_gender_vote_count",
        "name_gender_vote_margin",
        "name_gender_source_count",
        "name_gender_nomquamgender_count",
    ]:
        result[column] = pd.to_numeric(result[column], errors="coerce").astype("Int64")
    return result


def researcher_names_from_records(values) -> list[str]:
    """Extract non-empty full names from one nested researcher-record list."""
    if not isinstance(values, list):
        return []
    names = []
    for researcher in values:
        if not isinstance(researcher, dict):
            continue
        first_name = str(researcher.get("first_name") or "").strip()
        last_name = str(researcher.get("last_name") or "").strip()
        full_name = " ".join(part for part in (first_name, last_name) if part)
        if full_name:
            names.append(full_name)
    return names


def classify_researcher_lists(researchers: pd.Series) -> pd.Series:
    """Classify nested researcher dictionaries with the project-wide pipeline."""
    rows = []
    for source_row, values in enumerate(researchers):
        values = values if isinstance(values, list) else []
        for position, researcher in enumerate(values):
            if not isinstance(researcher, dict):
                continue
            researcher_id = researcher.get("researcher_id") or researcher.get("id")
            rows.append(
                {
                    "source_row": source_row,
                    "position": position,
                    "researcher_id": researcher_id,
                    "identity_resolved": bool(researcher_id),
                    "first_name": researcher.get("first_name") or "",
                }
            )
    nested = [[] for _ in range(len(researchers))]
    if not rows:
        return pd.Series(
            nested,
            index=researchers.index,
            name="name_gender_categories",
            dtype=object,
        )
    flat = pd.DataFrame(rows)
    inferred = classify_authorship_names(flat)
    flat["name_gender"] = inferred["name_gender"]
    grouped = (
        flat.sort_values(["source_row", "position"])
        .groupby("source_row", sort=False)["name_gender"]
        .agg(list)
    )
    for source_row, categories in grouped.items():
        nested[int(source_row)] = categories
    return pd.Series(
        nested,
        index=researchers.index,
        name="name_gender_categories",
        dtype=object,
    )


def researcher_list_features(
    researchers: pd.Series,
    *,
    infer_name_categories: bool = True,
) -> pd.DataFrame:
    """Return reusable name features for nested Showcase researcher records.

    Name categories are canonical title-case values. When inference is disabled,
    each row receives an empty category list so downstream analysis can retain the
    same schema without loading any classifier.
    """
    names = researchers.map(researcher_names_from_records).rename("researcher_names")
    if infer_name_categories:
        categories = classify_researcher_lists(researchers)
    else:
        categories = pd.Series(
            [[] for _ in range(len(researchers))],
            index=researchers.index,
            name="name_gender_categories",
            dtype=object,
        )
    return pd.concat([names, categories], axis=1)


def summarize_name_categories(category_lists) -> NameCategorySummary:
    """Summarize nested name categories with both common denominators."""
    categories = []
    for values in category_lists:
        if not isinstance(values, (list, tuple, np.ndarray, pd.Series)):
            continue
        categories.extend(
            _normalise_category(value, include_mostly=True) for value in values
        )
    counts = Counter(categories)
    female = counts["Female"]
    male = counts["Male"]
    total = len(categories)
    classified = female + male
    return NameCategorySummary(
        total=total,
        classified=classified,
        female=female,
        male=male,
        unknown=total - classified,
        coverage_percent=100 * classified / total if total else math.nan,
        female_percent_all=100 * female / total if total else math.nan,
        female_percent_classified=(
            100 * female / classified if classified else math.nan
        ),
    )


def female_name_percentage(
    category_lists,
    *,
    denominator: str = "classified",
    empty: float = math.nan,
) -> float:
    """Return Female-category percentage using an explicit denominator.

    ``denominator='classified'`` excludes Unknown; ``'all'`` retains Unknown and
    therefore reports a conservative lower bound rather than a composition estimate.
    """
    summary = summarize_name_categories(category_lists)
    if denominator == "classified":
        return summary.female_percent_classified if summary.classified else float(empty)
    if denominator == "all":
        return summary.female_percent_all if summary.total else float(empty)
    raise ValueError("denominator must be 'classified' or 'all'")


def inference_coverage(
    authorships: pd.DataFrame,
    group_col: str | None = None,
) -> pd.DataFrame:
    """Return coverage under each sensitivity stage, optionally by year."""
    stages = [
        ("Strict dictionary", "name_gender_strict"),
        ("Expanded dictionary", "name_gender_dictionary"),
        ("Offline ensemble", "name_gender_direct"),
        ("Primary + identity linkage", "name_gender"),
    ]
    groups = (
        [(None, authorships)]
        if group_col is None
        else authorships.groupby(group_col, observed=True)
    )
    rows = []
    for group_value, frame in groups:
        for stage, column in stages:
            classified = frame[column].isin(CLASSIFIED_CATEGORIES)
            row = {
                "stage": stage,
                "classified_author_paper_pairs": int(classified.sum()),
                "unknown_author_paper_pairs": int((~classified).sum()),
                "total_author_paper_pairs": len(frame),
                "coverage_percent": (
                    100 * float(classified.mean()) if len(frame) else np.nan
                ),
            }
            if group_col is not None:
                row[group_col] = group_value
            rows.append(row)
    columns = ([group_col] if group_col is not None else []) + [
        "stage",
        "classified_author_paper_pairs",
        "unknown_author_paper_pairs",
        "total_author_paper_pairs",
        "coverage_percent",
    ]
    return pd.DataFrame(rows)[columns]


def inference_method_counts(authorships: pd.DataFrame) -> pd.DataFrame:
    """Count final assignments by method, category, and unresolved reason."""
    return (
        authorships.groupby(
            ["name_gender_method", "name_gender", "name_gender_unknown_reason"],
            dropna=False,
            observed=True,
        )
        .size()
        .rename("author_paper_pairs")
        .reset_index()
        .sort_values("author_paper_pairs", ascending=False)
        .reset_index(drop=True)
    )


def offline_library_audit(authorships: pd.DataFrame) -> pd.DataFrame:
    """Report coverage and agreement for every package vote."""
    rows = []
    versions = offline_library_versions().set_index("library")
    for library, _, column in OFFLINE_LIBRARIES:
        values = authorships[column]
        classified = values.isin(CLASSIFIED_CATEGORIES)
        comparable = classified & authorships["name_gender_direct"].isin(
            CLASSIFIED_CATEGORIES
        )
        agreement = values.eq(authorships["name_gender_direct"])
        rows.append(
            {
                "library": library,
                "version": versions.loc[library, "version"],
                "classified_author_paper_pairs": int(classified.sum()),
                "female_author_paper_pairs": int(values.eq("Female").sum()),
                "male_author_paper_pairs": int(values.eq("Male").sum()),
                "unknown_author_paper_pairs": int((~classified).sum()),
                "coverage_percent": 100 * float(classified.mean()),
                "agreement_with_direct_percent": (
                    100 * float(agreement[comparable].mean())
                    if comparable.any()
                    else math.nan
                ),
                "inference_mode": "offline bundled data",
            }
        )
    return pd.DataFrame(rows)


def unresolved_query_queue(authorships: pd.DataFrame) -> pd.DataFrame:
    """Prioritize names unresolved after every offline library and identity link."""
    unresolved = authorships[
        authorships["name_gender"].eq("Unknown")
        & authorships["name_gender_query"].fillna("").ne("")
    ].copy()
    columns = [
        "query_name",
        "author_paper_pairs",
        "unique_resolved_authors",
        "first_year",
        "last_year",
        "dictionary_result",
        "offline_predictions",
        "unknown_reason",
    ]
    if unresolved.empty:
        return pd.DataFrame(columns=columns)
    return (
        unresolved.groupby("name_gender_query", observed=True)
        .agg(
            author_paper_pairs=("paper_id", "size"),
            unique_resolved_authors=("researcher_id", "nunique"),
            first_year=("year", "min"),
            last_year=("year", "max"),
            dictionary_result=(
                "name_gender_dictionary_detail",
                lambda values: "; ".join(sorted(set(values))),
            ),
            offline_predictions=(
                "name_gender_source_predictions",
                lambda values: "; ".join(sorted(set(values))),
            ),
            unknown_reason=(
                "name_gender_unknown_reason",
                lambda values: "; ".join(sorted(set(values))),
            ),
        )
        .reset_index()
        .rename(columns={"name_gender_query": "query_name"})
        .sort_values(["author_paper_pairs", "query_name"], ascending=[False, True])
        .reset_index(drop=True)[columns]
    )
