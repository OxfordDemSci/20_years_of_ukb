"""Resume the scoped 25-fit BERTopic analysis, or reuse its completed results.

Only the cold modelling path imports BERTopic and its optional dependencies. This
module produces assignment/diagnostic tables; publication figures are made from
those tables separately, so a new figure never requires refitting a topic model.
"""

from __future__ import annotations

from collections import defaultdict
import gc
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd

from utils import shared_paths as P
from utils.data_analysis_02_content_window import (
    find_existing_topic_results, topic_corpus_hash, write_topic_window_provenance,
)
from utils.shared_analysis_window import (
    ANALYSIS_START_DATE, ANALYSIS_START_YEAR, ANALYSIS_END_DATE, ANALYSIS_END_YEAR,
    filter_analysis_window,
)
from utils.shared_showcase import load_showcase


EMBEDDING_MODEL_NAME = "allenai-specter"
EMBEDDING_BATCH_SIZE = 64
PIPELINE_VERSION = "v2-ngram-coherence-preserved-ctfidf"
COHERENCE_PROCESSES = 4
SEEDS = [11, 21, 42, 73, 101]
PARAM_GRID = [
    {"n_neighbors": 15, "min_cluster_size": 25, "min_samples": 5},
    {"n_neighbors": 25, "min_cluster_size": 35, "min_samples": 10},
    {"n_neighbors": 35, "min_cluster_size": 45, "min_samples": 10},
    {"n_neighbors": 45, "min_cluster_size": 55, "min_samples": 15},
    {"n_neighbors": 25, "min_cluster_size": 60, "min_samples": 20},
]
QUALITY_WEIGHTS = {"topic_diversity": 0.25, "outlier_rate": -0.80}
ROBUSTNESS_WEIGHTS = {"mean_ari": 0.25, "mean_nmi": 0.10, "topic_count_cv": -0.10}
PERSISTENCE_SENSITIVITY_WEIGHT = 0.15
SELECT_WITH_PERSISTENCE = False
LOW_PERSISTENCE_THRESHOLD = 0.10
_DOMAIN_STOPWORDS = {
    "study", "studies", "result", "results", "method", "methods",
    "conclusion", "conclusions", "background", "objective", "objectives",
    "analysis", "analyses", "data", "using", "used", "use", "based",
    "association", "associations", "associated", "effect", "effects",
    "participant", "participants", "uk", "ukb", "biobank", "united",
    "kingdom", "paper", "research", "et", "al",
}


def clean_value(value):
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def clean_topic_text(title, abstract=""):
    """Preserve the original notebook's title/abstract cleaning exactly."""
    text = f"{clean_value(title)}. {clean_value(abstract)}".strip()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\b©\s*\d{4}.*", " ", text)
    text = re.sub(r"\bCopyright\b.*", " ", text, flags=re.I)
    text = re.sub(
        r"\b(Background|Objective|Objectives|Aim|Aims|Methods|Results|Conclusion|Conclusions|Keywords|Funding|Conflict of interest)\s*:",
        " ", text, flags=re.I,
    )
    return re.sub(r"\s+", " ", text).strip()


def infer_column(frame, candidates, label, required=True):
    lower = {str(column).lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    if required:
        raise KeyError(f"Could not identify {label} column. Tried: {candidates}")
    return None


def extract_year(frame, year_column, date_column):
    years = (pd.to_numeric(frame[year_column], errors="coerce") if year_column
             else pd.Series(np.nan, index=frame.index))
    if date_column:
        fallback = frame[date_column].astype(str).str.extract(r"((?:19|20)\d{2})", expand=False)
        years = years.fillna(pd.to_numeric(fallback, errors="coerce"))
    return years.astype("Int64")


def _prepare_publications(frame, *, legacy=False):
    id_column = infer_column(frame, ["id", "publication_id", "pid", "showcase_plus_id"], "ID")
    title_column = infer_column(frame, ["title", "original_title"], "title")
    abstract_column = infer_column(frame, ["abstract", "description"], "abstract", required=False)
    year_column = infer_column(frame, ["year", "publication_year"], "year", required=False)
    date_column = infer_column(
        frame, ["date", "publication_date", "published_date", "date_online", "date_print"],
        "date", required=False,
    )
    if not legacy:
        frame = filter_analysis_window(frame, year_col=year_column, date_col=date_column)
    publications = pd.DataFrame({
        "id": frame[id_column].map(clean_value),
        "title": frame[title_column].map(clean_value),
        "abstract": frame[abstract_column].map(clean_value) if abstract_column else "",
        "year": extract_year(frame, year_column, date_column),
    })
    publications["topic_text"] = [
        clean_topic_text(title, abstract)
        for title, abstract in zip(publications["title"], publications["abstract"])
    ]
    publications = publications.loc[
        publications["id"].ne("") & publications["topic_text"].str.len().gt(20)
        & publications["year"].ge(2014 if legacy else ANALYSIS_START_YEAR)
    ].drop_duplicates("id", keep="first").reset_index(drop=True)
    if not legacy and publications.empty:
        raise ValueError(
            f"No usable topic inputs remain from {ANALYSIS_START_DATE.date()} "
            f"through {ANALYSIS_END_DATE.date()}."
        )
    return publications


def prepare_publications(frame):
    """Unique text-eligible publication inputs, within both date boundaries."""
    return _prepare_publications(frame)


def load_publications(path=None):
    return prepare_publications(load_showcase(path=Path(path) if path is not None else P.SHOWCASE_PLUS))


def grid_cache_paths(publications, cache_dir=None):
    """Scope checkpoints to the corpus, parameters and score implementation."""
    cache_dir = Path(cache_dir) if cache_dir is not None else P.BERTOPIC_CACHE
    key = topic_corpus_hash(publications["id"], publications["topic_text"], publications["year"])
    configuration = hashlib.sha256(json.dumps({
        "parameters": PARAM_GRID, "seeds": SEEDS, "embedding_model": EMBEDDING_MODEL_NAME,
        "pipeline_version": PIPELINE_VERSION,
    }, sort_keys=True).encode()).hexdigest()[:12]
    return (
        cache_dir / f"seed_grid_runs_{key}_{configuration}.csv",
        cache_dir / f"seed_assignments_{key}_{configuration}",
    )


def _validate_embeddings(values, n_documents, source):
    if (values.ndim != 2 or values.shape[0] != n_documents or values.shape[1] == 0
            or not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all()):
        raise ValueError(f"Invalid embeddings in {source}; no cache was overwritten.")
    return values


def _legacy_embeddings(publications, input_parquet, cache_dir):
    """Reuse frozen text embeddings only after reproducing their legacy identity.

    The old cache hashed ordered IDs and the first 1,000 cleaned characters. A
    matching reconstruction and an additional complete-text equality check bind
    each reused row to its current publication; this is not a reuse of an old
    fitted topic model. New 2013 rows are encoded independently before fitting
    exclusively on the scoped corpus.
    """
    if not list(cache_dir.glob("embeddings_allenai-specter_*docs_????????????.npy")):
        return None, np.arange(len(publications))
    legacy = _prepare_publications(pd.read_parquet(input_parquet), legacy=True)
    digest = hashlib.sha1()
    for paper_id, document in zip(legacy["id"], legacy["topic_text"]):
        digest.update(paper_id.encode(errors="ignore"))
        digest.update(b"\t")
        digest.update(document[:1000].encode(errors="ignore"))
        digest.update(b"\n")
    path = cache_dir / f"embeddings_allenai-specter_{len(legacy)}docs_{digest.hexdigest()[:12]}.npy"
    if not path.is_file():
        return None, np.arange(len(publications))
    source = _validate_embeddings(np.load(path, mmap_mode="r"), len(legacy), path)
    by_id = legacy.reset_index().set_index("id")
    values = np.empty((len(publications), source.shape[1]), dtype=source.dtype)
    missing = []
    for index, paper in publications.iterrows():
        if paper["id"] not in by_id.index or by_id.loc[paper["id"], "topic_text"] != paper["topic_text"]:
            missing.append(index)
        else:
            values[index] = source[int(by_id.loc[paper["id"], "index"])]
    print(f"[CACHE] Reused {len(publications) - len(missing):,} verified SPECTER embeddings; "
          f"{len(missing):,} to encode.", flush=True)
    return values, np.asarray(missing, dtype=int)


def get_embeddings(publications, input_parquet, cache_dir):
    cache_dir = Path(cache_dir)
    key = topic_corpus_hash(publications["id"], publications["topic_text"], publications["year"])
    path = cache_dir / f"embeddings_allenai-specter_{len(publications)}docs_{key}.npy"
    if path.is_file():
        return _validate_embeddings(np.load(path), len(publications), path)
    values, missing = _legacy_embeddings(publications, input_parquet, cache_dir)
    if len(missing):
        import torch
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device, local_files_only=True)
        except OSError:
            model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
        print(f"[ENCODE] {len(missing):,} documents on {device}.", flush=True)
        encoded = model.encode(
            publications.iloc[missing]["topic_text"].tolist(),
            batch_size=EMBEDDING_BATCH_SIZE, show_progress_bar=False,
            normalize_embeddings=True,
        )
        if values is None:
            values = np.empty((len(publications), encoded.shape[1]), dtype=encoded.dtype)
        values[missing] = encoded
        del model
    values = _validate_embeddings(values, len(publications), path)
    np.save(path, values)
    return values


def _stopwords():
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    return sorted(set(ENGLISH_STOP_WORDS) | _DOMAIN_STOPWORDS)


def build_vectorizer():
    from sklearn.feature_extraction.text import CountVectorizer
    return CountVectorizer(stop_words=_stopwords(), ngram_range=(1, 3), min_df=5, max_df=0.85)


def tokenise_documents(docs):
    """Use the topic vectorizer's identical 1–3-gram vocabulary for coherence."""
    analyzer = build_vectorizer().build_analyzer()
    return [analyzer(document) for document in docs]


def build_topic_model(params, seed):
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer
    from hdbscan import HDBSCAN
    from umap import UMAP

    return BERTopic(
        umap_model=UMAP(
            n_neighbors=params["n_neighbors"], n_components=5, min_dist=0.0,
            metric="cosine", random_state=seed, low_memory=True,
        ),
        hdbscan_model=HDBSCAN(
            min_cluster_size=params["min_cluster_size"], min_samples=params["min_samples"],
            metric="euclidean", cluster_selection_method="eom", prediction_data=True,
        ),
        vectorizer_model=build_vectorizer(),
        ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
        # Keyword labels use c-TF-IDF directly. MMR requires an embedding backend
        # and was a no-op in the previous notebook's precomputed-embedding path.
        calculate_probabilities=False, verbose=False,
    )


def reduce_outliers(model, docs, topics, embeddings):
    if -1 not in topics:
        return list(topics)
    try:
        reassigned = model.reduce_outliers(docs, topics, strategy="embeddings", embeddings=embeddings)
        import logging

        logger = logging.getLogger("BERTopic")
        # This warning concerns subsequent topic-reduction calls; reassignment is
        # our final topic update. Keep every other library warning visible.
        warning_filter = lambda record: not record.getMessage().startswith(
            "WARNING: Using a custom list of topic assignments may lead to errors"
        )
        logger.addFilter(warning_filter)
        try:
            model.update_topics(
                docs, topics=reassigned, vectorizer_model=build_vectorizer(),
                ctfidf_model=model.ctfidf_model,
            )
        finally:
            logger.removeFilter(warning_filter)
        return list(reassigned)
    except (ValueError, IndexError) as error:
        print(f"[WARN] Outlier reassignment unavailable: {str(error).splitlines()[0]}", flush=True)
        return list(topics)


def topic_words(model, top_n=10):
    return {
        int(topic): [word for word, _ in words[:top_n] if str(word).strip()]
        for topic, words in model.get_topics().items() if int(topic) != -1 and words
    }


def cluster_persistence_summary(hdbscan_model, topic_mapping=None):
    """Use HDBSCAN's original cluster numbering for persistence and cluster sizes."""
    persistence = np.asarray(hdbscan_model.cluster_persistence_, dtype=float)
    labels = np.asarray(hdbscan_model.labels_, dtype=int)
    clusters = np.arange(len(persistence), dtype=int)
    sizes = np.asarray([(labels == cluster).sum() for cluster in clusters])
    mapping = topic_mapping or {}
    table = pd.DataFrame({
        "hdbscan_cluster": clusters,
        "topic": [mapping.get(int(cluster), int(cluster)) for cluster in clusters],
        "cluster_persistence": persistence,
        "raw_cluster_size": sizes,
    })
    metrics = dict.fromkeys([
        "mean_cluster_persistence", "median_cluster_persistence", "weighted_cluster_persistence",
        "minimum_cluster_persistence", "low_persistence_fraction",
    ], np.nan)
    if len(persistence):
        metrics.update({
            "mean_cluster_persistence": float(persistence.mean()),
            "median_cluster_persistence": float(np.median(persistence)),
            "weighted_cluster_persistence": float(np.average(persistence, weights=np.maximum(sizes, 1))),
            "minimum_cluster_persistence": float(persistence.min()),
            "low_persistence_fraction": float(np.mean(persistence < LOW_PERSISTENCE_THRESHOLD)),
        })
    return metrics, table


def _topic_count_penalty(n_topics):
    return (8 - n_topics) * 0.05 if n_topics < 8 else (n_topics - 80) * 0.01 if n_topics > 80 else 0.0


def _fit_metrics(model, raw_labels, labels):
    words = topic_words(model)
    flat = [word for group in words.values() for word in group]
    diversity = len(set(flat)) / len(flat) if flat else np.nan
    n_topics = len(set(labels) - {-1})
    if n_topics == 0:
        raise ValueError("No non-outlier topics were identified by this parameter/seed fit.")
    outlier_rate = float(np.mean(labels == -1))
    penalty = _topic_count_penalty(n_topics)
    persistence, _ = cluster_persistence_summary(model.hdbscan_model)
    return {
        "n_topics": n_topics, "outlier_rate": outlier_rate,
        "topic_diversity": diversity, "topic_count_penalty": penalty,
        "raw_outlier_rate": float(np.mean(np.asarray(raw_labels) == -1)), **persistence,
    }, words


def _score_quality(metrics, words, tokens, dictionary, *, coherence_processes=None):
    from gensim.models import CoherenceModel

    coherence = float(CoherenceModel(
        topics=list(words.values()), texts=tokens, dictionary=dictionary,
        coherence="c_v", processes=COHERENCE_PROCESSES if coherence_processes is None else coherence_processes,
    ).get_coherence()) if words and len(dictionary) else np.nan
    diversity = metrics["topic_diversity"]
    score = (
        (0.0 if pd.isna(coherence) else coherence)
        + QUALITY_WEIGHTS["topic_diversity"] * (0.0 if pd.isna(diversity) else diversity)
        + QUALITY_WEIGHTS["outlier_rate"] * metrics["outlier_rate"] - metrics["topic_count_penalty"]
    )
    return {"coherence_cv": coherence, "quality_score": score}


def _quality_metrics(model, docs, tokens, dictionary, raw_labels, labels, *, coherence_processes=None):
    metrics, words = _fit_metrics(model, raw_labels, labels)
    metrics.update(_score_quality(metrics, words, tokens, dictionary, coherence_processes=coherence_processes))
    return metrics


def _fit_one_seed(parameter_index, params, seed, number, *, docs, embeddings,
                  tokens=None, dictionary=None, coherence_processes=None, defer_scoring=False):
    """Independent fit: return results to the coordinator, never write files."""
    import torch

    total = len(PARAM_GRID) * len(SEEDS)
    print(f"[GRID {number:02d}/{total}] Parameters {parameter_index}, seed {seed}.", flush=True)
    row = {"parameter_index": parameter_index, "seed": seed, **params, "status": "failed"}
    model, labels = None, None
    started = time.monotonic()
    try:
        model = build_topic_model(params, seed)
        raw_labels, _ = model.fit_transform(docs, embeddings=embeddings)
        labels = np.asarray(reduce_outliers(model, docs, raw_labels, embeddings), dtype=np.int32)
        fit_seconds = time.monotonic() - started
        row["fit_seconds"] = fit_seconds
        if defer_scoring:
            metrics, words = _fit_metrics(model, raw_labels, labels)
            row.update(metrics)
            row["_topic_words"] = words
            row["status"] = "unscored"
            print(f"[GRID {number:02d}/{total}] Fit {fit_seconds:.0f}s; ready for coherence.", flush=True)
        else:
            print(f"[GRID {number:02d}/{total}] Fit {fit_seconds:.0f}s; scoring coherence.", flush=True)
            row.update(_quality_metrics(
                model, docs, tokens, dictionary, raw_labels, labels,
                coherence_processes=coherence_processes,
            ))
            row["scoring_seconds"] = time.monotonic() - started - fit_seconds
            row["status"] = "ok"
    except Exception as error:
        row["error"] = f"{type(error).__name__}: {str(error).splitlines()[0]}"
        labels = None
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return row, labels


_SEED_WORKER_CONTEXT = None


def _initialise_seed_worker(fit_function, fit_kwargs):
    """Transfer the large shared input once per worker, not once per seed."""
    global _SEED_WORKER_CONTEXT
    _SEED_WORKER_CONTEXT = fit_function, fit_kwargs


def _parallel_seed_task(job):
    fit_function, fit_kwargs = _SEED_WORKER_CONTEXT
    return fit_function(*job, **fit_kwargs)


def _dispatch_seed_jobs(jobs, *, workers, fit_kwargs, fit_function=None):
    fit_function = _fit_one_seed if fit_function is None else fit_function
    if workers == 1:
        for job in jobs:
            yield fit_function(*job, **fit_kwargs)
        return
    from joblib import Parallel, delayed, parallel_config

    # Independent processes own their estimators. Scoring occurs in the parent,
    # avoiding nested gensim/loky process contexts and keeping the large vocabulary
    # in one process. Each fit worker gets one numerical-library thread.
    with parallel_config(backend="loky", inner_max_num_threads=1):
        with Parallel(
            n_jobs=workers, return_as="generator_unordered", batch_size=1,
            pre_dispatch=workers, initializer=_initialise_seed_worker,
            initargs=(fit_function, fit_kwargs),
        ) as parallel:
            yield from parallel(delayed(_parallel_seed_task)(job) for job in jobs)


def _worker_count(workers=None):
    value = os.environ.get("UKB_TOPIC_WORKERS", "1") if workers is None else workers
    if str(value) not in {"1", "2"}:
        raise ValueError("BERTopic workers must be 1 or 2 (or set UKB_TOPIC_WORKERS=1/2).")
    return int(value)


def run_seed_grid(publications, embeddings, cache_dir, *, workers=None):
    """Resume independent seed fits; only this coordinator commits checkpoints."""
    workers = _worker_count(workers)
    docs = publications["topic_text"].tolist()
    runs_path, assignment_dir = grid_cache_paths(publications, cache_dir)
    assignment_dir.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(runs_path).to_dict("records") if runs_path.is_file() else []
    completed = {(int(row["parameter_index"]), int(row["seed"])) for row in rows
                 if row.get("status") == "ok" and pd.notna(row.get("weighted_cluster_persistence"))}
    jobs = []
    total = len(PARAM_GRID) * len(SEEDS)
    for parameter_index, params in enumerate(PARAM_GRID):
        for seed_index, seed in enumerate(SEEDS):
            number = parameter_index * len(SEEDS) + seed_index + 1
            path = assignment_dir / f"params_{parameter_index}_seed_{seed}.npy"
            if (parameter_index, seed) in completed and path.is_file():
                labels = np.load(path)
                if labels.shape != (len(docs),):
                    raise ValueError(f"Invalid grid assignment length: {path}")
                print(f"[GRID {number:02d}/{total}] Reused parameters {parameter_index}, seed {seed}.", flush=True)
                continue
            jobs.append((parameter_index, params, seed, number))
    if not jobs:
        return pd.read_csv(runs_path), assignment_dir

    from gensim.corpora import Dictionary

    tokens = tokenise_documents(docs)
    dictionary = Dictionary(tokens)
    print(f"[GRID] {len(jobs)} fits remaining; {workers} fit worker(s), "
          f"{COHERENCE_PROCESSES} coherence processes in the coordinator.", flush=True)
    fit_kwargs = dict(docs=docs, embeddings=embeddings)
    if workers > 1:
        fit_kwargs["defer_scoring"] = True
    else:
        fit_kwargs.update(tokens=tokens, dictionary=dictionary, coherence_processes=COHERENCE_PROCESSES)
    results = _dispatch_seed_jobs(
        jobs, workers=workers, fit_kwargs=fit_kwargs,
    )
    for row, labels in results:
        parameter_index, seed = int(row["parameter_index"]), int(row["seed"])
        number = parameter_index * len(SEEDS) + SEEDS.index(seed) + 1
        if row["status"] == "unscored":
            words = row.pop("_topic_words")
            print(f"[GRID {number:02d}/{total}] Scoring coherence in coordinator.", flush=True)
            started = time.monotonic()
            try:
                row.update(_score_quality(row, words, tokens, dictionary))
                row["scoring_seconds"] = time.monotonic() - started
                row["status"] = "ok"
            except Exception as error:
                row["status"] = "failed"
                row["error"] = f"{type(error).__name__}: {str(error).splitlines()[0]}"
                labels = None
        if row["status"] == "ok":
            labels = np.asarray(labels, dtype=np.int32)
            if labels.shape != (len(docs),):
                raise ValueError(f"Invalid result length for parameters {parameter_index}, seed {seed}.")
            path = assignment_dir / f"params_{parameter_index}_seed_{seed}.npy"
            temporary = path.with_suffix(".tmp.npy")
            np.save(temporary, labels)
            temporary.replace(path)
            print(f"[GRID {number:02d}/{total}] {row['n_topics']} topics; "
                  f"coherence {row['coherence_cv']:.3f}; outliers {row['outlier_rate']:.1%}; "
                  f"scoring {row['scoring_seconds']:.0f}s.", flush=True)
        else:
            print(f"[FAIL {number:02d}/{total}] {row['error']}", flush=True)
        rows = [old for old in rows if (int(old["parameter_index"]), int(old["seed"])) != (parameter_index, seed)]
        rows.append(row)
        temporary = runs_path.with_suffix(".tmp.csv")
        pd.DataFrame(rows).sort_values(["parameter_index", "seed"]).to_csv(temporary, index=False)
        temporary.replace(runs_path)
    return pd.read_csv(runs_path), assignment_dir


def select_robust_model(seed_runs, assignment_dir):
    """Select parameters by quality plus ARI/NMI robustness, then the ARI medoid."""
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    successful = seed_runs.loc[seed_runs["status"].eq("ok")].copy()
    if successful.empty:
        raise RuntimeError("No successful BERTopic seed-grid runs; see seed_grid_runs.csv.")
    required = {(index, seed) for index in range(len(PARAM_GRID)) for seed in SEEDS}
    actual = [(int(row.parameter_index), int(row.seed)) for row in successful.itertuples()]
    missing = sorted(required - set(actual))
    unexpected = sorted(set(actual) - required)
    if missing or unexpected or len(set(actual)) != len(actual):
        details = []
        if missing:
            details.append("missing/failed " + ", ".join(f"parameters {index}/seed {seed}" for index, seed in missing))
        if unexpected:
            details.append("unexpected parameter/seed combinations")
        if len(set(actual)) != len(actual):
            details.append("duplicate successful parameter/seed combinations")
        raise RuntimeError(
            f"BERTopic selection requires all {len(PARAM_GRID)} parameter sets × {len(SEEDS)} seeds "
            "to succeed; " + "; ".join(details) + ". Resume the grid before selecting a final model."
        )
    stability_rows, pair_rows = [], []
    for parameter_index, group in successful.groupby("parameter_index", sort=True):
        assignments = {int(seed): np.load(Path(assignment_dir) / f"params_{int(parameter_index)}_seed_{int(seed)}.npy")
                       for seed in group["seed"]}
        aris, nmis, per_seed = [], [], defaultdict(list)
        for seed_a, seed_b in itertools.combinations(sorted(assignments), 2):
            ari = adjusted_rand_score(assignments[seed_a], assignments[seed_b])
            nmi = normalized_mutual_info_score(assignments[seed_a], assignments[seed_b])
            aris.append(ari)
            nmis.append(nmi)
            per_seed[seed_a].append(ari)
            per_seed[seed_b].append(ari)
            pair_rows.append({"parameter_index": int(parameter_index), "seed_a": seed_a,
                              "seed_b": seed_b, "ari": ari, "nmi": nmi})
        mean_seed_ari = {seed: float(np.mean(values)) for seed, values in per_seed.items()}
        medoid = max(mean_seed_ari, key=mean_seed_ari.get)
        mean_topics = float(group["n_topics"].mean())
        topic_cv = float(group["n_topics"].std(ddof=0) / max(mean_topics, 1))
        mean_quality = float(group["quality_score"].mean())
        mean_ari, mean_nmi = float(np.mean(aris)), float(np.mean(nmis))
        persistence = float(group["weighted_cluster_persistence"].mean())
        score = (mean_quality + ROBUSTNESS_WEIGHTS["mean_ari"] * mean_ari
                 + ROBUSTNESS_WEIGHTS["mean_nmi"] * mean_nmi
                 + ROBUSTNESS_WEIGHTS["topic_count_cv"] * topic_cv)
        stability_rows.append({
            "parameter_index": int(parameter_index), "n_seeds": len(group),
            "mean_quality_score": mean_quality,
            "mean_coherence_cv": float(group["coherence_cv"].mean()),
            "mean_topic_diversity": float(group["topic_diversity"].mean()),
            "mean_outlier_rate": float(group["outlier_rate"].mean()),
            "mean_raw_outlier_rate": float(group["raw_outlier_rate"].mean()),
            "mean_cluster_persistence": float(group["mean_cluster_persistence"].mean()),
            "median_cluster_persistence": float(group["median_cluster_persistence"].mean()),
            "mean_weighted_cluster_persistence": persistence,
            "mean_low_persistence_fraction": float(group["low_persistence_fraction"].mean()),
            "mean_n_topics": mean_topics, "sd_n_topics": float(group["n_topics"].std(ddof=0)),
            "topic_count_cv": topic_cv, "mean_pairwise_ari": mean_ari,
            "min_pairwise_ari": float(np.min(aris)), "mean_pairwise_nmi": mean_nmi,
            "min_pairwise_nmi": float(np.min(nmis)), "representative_seed": medoid,
            "representative_seed_mean_ari": mean_seed_ari[medoid],
            "robust_score": score,
            "robust_score_with_persistence": score + PERSISTENCE_SENSITIVITY_WEIGHT * persistence,
        })
    if not stability_rows:
        raise RuntimeError("BERTopic selection requires at least two successful seeds for a parameter set.")
    score_column = "robust_score_with_persistence" if SELECT_WITH_PERSISTENCE else "robust_score"
    stability = pd.DataFrame(stability_rows).sort_values(
        [score_column, "parameter_index"], ascending=[False, True], kind="stable",
    ).reset_index(drop=True)
    return stability, pd.DataFrame(pair_rows)


def _fit_topic_results(publications, input_parquet, output_dir, cache_dir, *, workers=1):
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    docs = publications["topic_text"].tolist()
    print(f"[BERTopic] {len(publications):,} publications, "
          f"{ANALYSIS_START_YEAR}–{ANALYSIS_END_YEAR}; {len(PARAM_GRID) * len(SEEDS)} seed-grid fits.", flush=True)
    embeddings = get_embeddings(publications, input_parquet, cache_dir)
    seed_runs, assignment_dir = run_seed_grid(publications, embeddings, cache_dir, workers=workers)
    seed_runs.to_csv(output_dir / "seed_grid_runs.csv", index=False)
    write_topic_window_provenance(output_dir / "seed_grid_runs.csv", publications["year"])
    stability, seed_pairs = select_robust_model(seed_runs, assignment_dir)
    stability.to_csv(output_dir / "bertopic_seed_robustness_summary.csv", index=False)
    seed_pairs.to_csv(output_dir / "bertopic_seed_pair_stability.csv", index=False)
    for name in ("bertopic_seed_robustness_summary.csv", "bertopic_seed_pair_stability.csv"):
        write_topic_window_provenance(output_dir / name, publications["year"])
    best = stability.iloc[0]
    parameter_index, seed = int(best["parameter_index"]), int(best["representative_seed"])
    print(f"[FINAL] Parameters {parameter_index}, representative seed {seed}.", flush=True)
    model = build_topic_model(PARAM_GRID[parameter_index], seed)
    raw_topics, _ = model.fit_transform(docs, embeddings=embeddings)
    topic_mapping = model.topic_mapper_.get_mappings(original_topics=True)
    persistence_metrics, persistence_table = cluster_persistence_summary(model.hdbscan_model, topic_mapping)
    topics = np.asarray(reduce_outliers(model, docs, raw_topics, embeddings), dtype=np.int32)
    words = topic_words(model)
    topic_ids = sorted(set(topics) - {-1})
    labels = {topic: " / ".join(words.get(topic, [])[:4]) for topic in topic_ids}
    assignments = publications[["id", "year", "title"]].copy()
    assignments["topic"] = topics
    assignments["topics"] = ["Outlier" if topic == -1 else f"T{topic}: {labels.get(topic, '')}" for topic in topics]
    pd.DataFrame({
        "topic": topic_ids, "topic_words": [", ".join(words.get(topic, [])) for topic in topic_ids],
        "short_label": [labels.get(topic, "") for topic in topic_ids],
    }).to_csv(output_dir / "bertopic_topic_labels.csv", index=False)
    persistence_table["short_label"] = persistence_table["topic"].map(labels)
    persistence_table.to_csv(output_dir / "bertopic_final_cluster_persistence.csv", index=False)
    for name in ("bertopic_topic_labels.csv", "bertopic_final_cluster_persistence.csv"):
        write_topic_window_provenance(output_dir / name, publications["year"])
    manifest = {
        "analysis_start_date": str(ANALYSIS_START_DATE.date()),
        "analysis_end_date": str(ANALYSIS_END_DATE.date()),
        "training_min_year": int(publications["year"].min()),
        "training_max_year": int(publications["year"].max()),
        "corpus_cache_key": topic_corpus_hash(publications["id"], docs, publications["year"]),
        "n_documents": len(assignments), "n_topics": len(topic_ids),
        "outlier_rate": float(np.mean(topics == -1)),
        "raw_outlier_rate": float(np.mean(np.asarray(raw_topics) == -1)),
        "parameter_index": parameter_index, "parameters": PARAM_GRID[parameter_index], "seed": seed,
        "embedding_model": EMBEDDING_MODEL_NAME, "hdbscan_persistence": persistence_metrics,
        "pipeline_version": PIPELINE_VERSION,
        "selection_score": "robust_score_with_persistence" if SELECT_WITH_PERSISTENCE else "robust_score",
    }
    (output_dir / "bertopic_final_run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # Commit completed result files after their supporting diagnostics. The compact
    # ID/topic table is the last artifact written and is the first reuse candidate.
    for filename, table in (
        ("bertopic_document_topic_assignments.csv", assignments),
        ("showcase_plus_id_topics.csv", assignments[["id", "topics"]]),
    ):
        path = output_dir / filename
        table.to_csv(path, index=False)
        write_topic_window_provenance(path, publications["year"])
    print(f"[SAVED] {len(topic_ids)} topics for {len(assignments):,} publications.", flush=True)
    return output_dir / "showcase_plus_id_topics.csv"


def _check_model_dependencies():
    """Validate the cold modelling environment without importing its packages."""
    from importlib.metadata import PackageNotFoundError, version
    from packaging.version import Version

    missing, versions = [], {}
    for package in ("bertopic", "sentence-transformers", "umap-learn", "hdbscan", "gensim", "torch", "scikit-learn", "joblib"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            missing.append(package)
    problems = []
    if missing:
        problems.append("missing " + ", ".join(missing))
    if "hdbscan" in versions and Version(versions["hdbscan"]) < Version("0.8.41"):
        problems.append(f"hdbscan {versions['hdbscan']} is incompatible; version >=0.8.41 is required")
    if problems:
        raise RuntimeError(
            "BERTopic cannot start: " + "; ".join(problems)
            + ". Install/update requirements-analysis.txt before fitting."
        )


def ensure_topic_results(output_dir=None, cache_dir=None, *, input_parquet=None,
                         force=False, train_if_missing=True, workers=None):
    """Return validated completed results, fitting only when explicitly necessary.

    An invalid existing result raises without silently recomputing or overwriting
    it. ``train_if_missing=False`` supports category-only analyses when no result
    exists. ``force=True`` is the deliberate override for a complete model refit.
    Optional ``workers=2`` (or ``UKB_TOPIC_WORKERS=2``) runs independent seed fits
    in two processes; the default remains serial with identical cache identities.
    """
    output_dir = Path(output_dir) if output_dir is not None else P.OUTPUT / "bertopic"
    if not force:
        existing = find_existing_topic_results(output_dir)
        if existing is not None:
            print(f"[SKIP] BERTopic results already exist: {P.raw_path(existing)}", flush=True)
            return existing
    if not train_if_missing:
        print("[SKIP] No completed topic results; model fitting is disabled.", flush=True)
        return None
    input_parquet = Path(input_parquet) if input_parquet is not None else P.SHOWCASE_PLUS
    cache_dir = Path(cache_dir) if cache_dir is not None else P.BERTOPIC_CACHE
    if not input_parquet.is_file():
        raise FileNotFoundError(f"Full-endpoint parquet not found: {input_parquet}")
    _check_model_dependencies()
    workers = _worker_count(workers)
    publications = load_publications(input_parquet)
    return _fit_topic_results(publications, input_parquet, output_dir, cache_dir, workers=workers)
