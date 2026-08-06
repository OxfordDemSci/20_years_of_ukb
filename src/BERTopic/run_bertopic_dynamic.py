#!/usr/bin/env python3
# UKB Showcase+ BERTopic dynamic topic modelling.
#
# Input:
#     Final Showcase+ CSV, one row per publication.
#
# Main design:
#     - Uses title + abstract as text input.
#     - Uses fixed scientific document embedding model: allenai-specter.
#     - Fits one global BERTopic model across all years.
#     - Computes dynamic topic counts/shares over publication year.

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import torch
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from hdbscan import HDBSCAN
from scipy.ndimage import gaussian_filter1d
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from umap import UMAP

import config

warnings.filterwarnings("ignore")

mpl.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 600,
        "font.size": 12,
        "axes.titlesize": 17,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

ID_CANDIDATES = [
    "showcase_id",
    "id",
    "publication_id",
    "paper_id",
    "dimensions_id",
    "dimensions_publication_id",
    "dim_id",
    "pub_id",
    "publication_ids",
    "record_id",
    "source_id",
    "doi",
    "pmid",
]
TITLE_CANDIDATES = ["title", "publication_title", "paper_title", "article_title"]
ABSTRACT_CANDIDATES = ["abstract", "description", "summary", "publication_abstract", "paper_abstract"]
YEAR_CANDIDATES = ["analysis_year", "year", "publication_year", "pub_year", "date_year"]
DATE_CANDIDATES = ["date", "publication_date", "published_date", "date_online", "date_print"]

CUSTOM_STOPWORDS = set(ENGLISH_STOP_WORDS) | {
    "uk",
    "united",
    "kingdom",
    "biobank",
    "ukb",
    "study",
    "studies",
    "data",
    "dataset",
    "datasets",
    "participants",
    "participant",
    "cohort",
    "cohorts",
    "analysis",
    "analyses",
    "using",
    "based",
    "associated",
    "association",
    "associations",
    "risk",
    "results",
    "methods",
    "background",
    "conclusion",
    "conclusions",
    "objective",
    "objectives",
    "aim",
    "aims",
    "keywords",
    "funding",
    "conflict",
    "interest",
}


def read_csv_safely(path: Path) -> pd.DataFrame:
    for enc in ["utf-8", "utf-8-sig", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False)


def norm_col(c: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")


def infer_col(
    df: pd.DataFrame,
    candidates: Sequence[str],
    required: bool = False,
    label: str = "column",
    override: Optional[str] = None,
) -> Optional[str]:
    if override:
        if override in df.columns:
            return override
        raise ValueError(f"Configured {label} column not found: {override}")

    norm_map = {norm_col(c): c for c in df.columns}

    for cand in candidates:
        key = norm_col(cand)
        if key in norm_map:
            return norm_map[key]

    for cand in candidates:
        key = norm_col(cand)
        for n, orig in norm_map.items():
            if key and (key in n or n in key):
                return orig

    if required:
        raise ValueError(
            f"Could not infer {label}. Tried {list(candidates)}. "
            f"Available columns include: {list(df.columns)[:80]}"
        )
    return None


def clean_str(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalise_title(x: Any) -> str:
    s = clean_str(x).lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return s.strip()


def make_short_hash(*parts: Any, n: int = 16) -> str:
    joined = "||".join(clean_str(p) for p in parts)
    return hashlib.sha1(joined.encode("utf-8", errors="ignore")).hexdigest()[:n]


def get_year_from_row(
    row: pd.Series,
    year_col: Optional[str] = None,
    date_col: Optional[str] = None,
) -> float:
    if year_col and year_col in row and not pd.isna(row[year_col]):
        try:
            y = int(float(row[year_col]))
            if 1800 <= y <= 2100:
                return y
        except Exception:
            pass

    if date_col and date_col in row and not pd.isna(row[date_col]):
        m = re.search(r"(19|20)\d{2}", str(row[date_col]))
        if m:
            return int(m.group(0))

    return np.nan


def clean_topic_text(title: Any, abstract: Any = "") -> str:
    title = clean_str(title)
    abstract = clean_str(abstract)
    text = (title + ". " + abstract).strip()

    text = re.sub(r"<[^>]+>", " ", text)
    text = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    text = re.sub(r"\b©\s*\d{4}.*", " ", text)
    text = re.sub(r"\bCopyright\b.*", " ", text, flags=re.I)
    text = re.sub(r"\bElectronic supplementary material is available.*", " ", text, flags=re.I)
    text = re.sub(r"\bSupplementary (data|material|materials) (are|is) available.*", " ", text, flags=re.I)

    text = re.sub(
        r"\b(Background|Objective|Objectives|Aim|Aims|Methods|Results|Conclusion|Conclusions|Keywords|Funding|Conflict of interest)\s*:",
        " ",
        text,
        flags=re.I,
    )

    text = re.sub(r"\s+", " ", text).strip()
    return text


def make_showcase_id(df: pd.DataFrame, id_col: Optional[str], title_col: str, year_col: str) -> pd.Series:
    if id_col and id_col in df.columns:
        ids = df[id_col].map(clean_str)
        missing = ids.eq("") | ids.str.lower().isin(["nan", "none", "null"])
    else:
        ids = pd.Series([""] * len(df), index=df.index)
        missing = pd.Series([True] * len(df), index=df.index)

    fallback = df.apply(
        lambda r: "titleyear:" + make_short_hash(normalise_title(r.get(title_col, "")), r.get(year_col, "")),
        axis=1,
    )
    ids.loc[missing] = fallback.loc[missing]
    return ids


def corpus_hash(docs: Sequence[str], years: Sequence[int]) -> str:
    h = hashlib.sha1()
    for doc, year in zip(docs, years):
        h.update(str(year).encode("utf-8"))
        h.update(b"\t")
        h.update(doc[:1000].encode("utf-8", errors="ignore"))
        h.update(b"\n")
    return h.hexdigest()[:12]


def save_mpl(fig: plt.Figure, fig_dir: Path, basename: str, dpi: int = 600) -> Tuple[Path, Path, Path]:
    png = fig_dir / f"{basename}.png"
    pdf = fig_dir / f"{basename}.pdf"
    svg = fig_dir / f"{basename}.svg"
    fig.tight_layout()
    fig.savefig(png, bbox_inches="tight", dpi=dpi)
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return png, pdf, svg


def tokenise_for_coherence(docs: Sequence[str]) -> List[List[str]]:
    tokenised = []
    for doc in docs:
        tokens = re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", doc.lower())
        tokens = [t for t in tokens if t not in CUSTOM_STOPWORDS and len(t) > 2]
        tokenised.append(tokens)
    return tokenised


def topic_words_from_model(topic_model: BERTopic, top_n: int = 10) -> Dict[int, List[str]]:
    topics = {}
    for topic_id, words_scores in topic_model.get_topics().items():
        if topic_id == -1:
            continue
        words = [w for w, _ in words_scores[:top_n] if isinstance(w, str) and w.strip()]
        if words:
            topics[int(topic_id)] = words
    return topics


def compute_topic_diversity(topic_words: Dict[int, List[str]]) -> float:
    all_words = [w for words in topic_words.values() for w in words]
    if not all_words:
        return np.nan
    return len(set(all_words)) / len(all_words)


def compute_coherence_cv(docs: Sequence[str], topic_words: Dict[int, List[str]]) -> float:
    if not topic_words:
        return np.nan

    tokenised = tokenise_for_coherence(docs)
    dictionary = Dictionary(tokenised)

    if len(dictionary) == 0:
        return np.nan

    topics = list(topic_words.values())

    try:
        cm = CoherenceModel(
            topics=topics,
            texts=tokenised,
            dictionary=dictionary,
            coherence="c_v",
        )
        return float(cm.get_coherence())
    except Exception:
        return np.nan


def build_vectorizer() -> CountVectorizer:
    return CountVectorizer(
        stop_words=list(CUSTOM_STOPWORDS),
        ngram_range=(1, 3),
        min_df=5,
        max_df=0.85,
    )


def build_topic_model(params: Dict[str, int], random_state: int) -> BERTopic:
    umap_model = UMAP(
        n_neighbors=params["n_neighbors"],
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=random_state,
        low_memory=True,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=params["min_cluster_size"],
        min_samples=params["min_samples"],
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    vectorizer_model = build_vectorizer()
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
    representation_model = MaximalMarginalRelevance(diversity=0.35)

    return BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        representation_model=representation_model,
        calculate_probabilities=False,
        verbose=False,
    )


def reduce_outliers_safely(
    topic_model: BERTopic,
    docs: Sequence[str],
    topics: List[int],
    embeddings: np.ndarray,
) -> List[int]:
    if -1 not in set(topics):
        return list(topics)

    try:
        new_topics = topic_model.reduce_outliers(
            list(docs),
            list(topics),
            strategy="embeddings",
            embeddings=embeddings,
        )
        try:
            topic_model.update_topics(
                list(docs),
                topics=new_topics,
                vectorizer_model=build_vectorizer(),
            )
        except Exception:
            pass
        return list(new_topics)
    except Exception:
        return list(topics)


def score_model(n_topics: int, outlier_rate: float, coherence: float, diversity: float) -> float:
    coherence = 0.0 if pd.isna(coherence) else float(coherence)
    diversity = 0.0 if pd.isna(diversity) else float(diversity)

    topic_penalty = 0.0
    if n_topics < 8:
        topic_penalty += (8 - n_topics) * 0.05
    if n_topics > 80:
        topic_penalty += (n_topics - 80) * 0.01

    return coherence + 0.25 * diversity - 0.80 * outlier_rate - topic_penalty


def prepare_input(input_csv: Path, output_dir: Path) -> pd.DataFrame:
    print(f"[1/8] Loading input CSV: {input_csv}")
    df_raw = read_csv_safely(input_csv)
    print(f"Loaded rows: {len(df_raw):,}")

    id_col = infer_col(df_raw, ID_CANDIDATES, required=False, label="id", override=config.ID_COL)
    title_col = infer_col(df_raw, TITLE_CANDIDATES, required=True, label="title", override=config.TITLE_COL)
    abstract_col = infer_col(df_raw, ABSTRACT_CANDIDATES, required=False, label="abstract", override=config.ABSTRACT_COL)
    year_col_raw = infer_col(df_raw, YEAR_CANDIDATES, required=False, label="year", override=config.YEAR_COL)
    date_col = infer_col(df_raw, DATE_CANDIDATES, required=False, label="date", override=config.DATE_COL)

    print("Detected columns:")
    print(f"  id      : {id_col}")
    print(f"  title   : {title_col}")
    print(f"  abstract: {abstract_col}")
    print(f"  year    : {year_col_raw}")
    print(f"  date    : {date_col}")

    df = df_raw.copy()

    if abstract_col is None:
        df["_abstract_for_topic"] = ""
        abstract_col = "_abstract_for_topic"

    df["analysis_year"] = df.apply(
        lambda r: get_year_from_row(r, year_col=year_col_raw, date_col=date_col),
        axis=1,
    )

    df = df[df["analysis_year"].notna()].copy()
    df["analysis_year"] = df["analysis_year"].astype(int)
    df = df[df["analysis_year"] >= int(config.MIN_YEAR)].copy()

    df["topic_text"] = df.apply(
        lambda r: clean_topic_text(r.get(title_col, ""), r.get(abstract_col, "")),
        axis=1,
    )

    df = df[df["topic_text"].str.len() >= 30].copy()
    df["showcase_plus_id"] = make_showcase_id(df, id_col, title_col, "analysis_year")

    before = len(df)
    df = df.drop_duplicates("showcase_plus_id").copy()
    after = len(df)

    print(f"Rows after usable text/year filtering and deduplication: {after:,} removed {before-after:,}")

    keep_cols = ["showcase_plus_id", "analysis_year", "topic_text"]

    for c in [id_col, title_col, abstract_col, year_col_raw, date_col]:
        if c and c in df.columns and c not in keep_cols:
            keep_cols.append(c)

    out = df[keep_cols].copy()

    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    out.to_csv(table_dir / "analysis_input_with_text_year.csv", index=False)

    year_counts = out["analysis_year"].value_counts().sort_index().reset_index()
    year_counts.columns = ["analysis_year", "n_papers"]
    year_counts.to_csv(table_dir / "yearly_paper_counts.csv", index=False)

    print("Yearly paper counts:")
    print(year_counts.to_string(index=False))

    return out


def get_or_make_embeddings(docs: List[str], years: List[int], output_dir: Path) -> np.ndarray:
    print("[2/8] Embedding documents with fixed model: allenai-specter")

    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    chash = corpus_hash(docs, years)
    cache_path = cache_dir / f"embeddings_allenai-specter_{len(docs)}docs_{chash}.npy"

    if cache_path.exists():
        print(f"Loading cached embeddings: {cache_path}")
        return np.load(cache_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = config.BATCH_SIZE_GPU if device == "cuda" else config.BATCH_SIZE_CPU

    print(f"Device: {device}; batch size: {batch_size}")

    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=device)

    embeddings = model.encode(
        docs,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    np.save(cache_path, embeddings)

    print(f"Saved embeddings: {cache_path}")
    print(f"Embedding matrix shape: {embeddings.shape}")

    return embeddings


def run_topic_grid(
    docs: List[str],
    embeddings: np.ndarray,
    output_dir: Path,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    print("[3/8] Testing BERTopic hyperparameters")

    if getattr(config, "MAX_DOCS_FOR_TOPIC_GRID", None):
        max_docs = int(config.MAX_DOCS_FOR_TOPIC_GRID)

        if len(docs) > max_docs:
            rng = np.random.default_rng(config.RANDOM_STATE)
            idx = rng.choice(len(docs), size=max_docs, replace=False)
            idx = np.sort(idx)
            grid_docs = [docs[i] for i in idx]
            grid_embeddings = embeddings[idx]
            print(f"Grid search uses a sample of {len(grid_docs):,} documents.")
        else:
            grid_docs = docs
            grid_embeddings = embeddings
    else:
        grid_docs = docs
        grid_embeddings = embeddings

    rows = []
    best_score = -1e9
    best_params = None

    for i, params in enumerate(config.PARAM_GRID, start=1):
        print(f"  Model {i}/{len(config.PARAM_GRID)}: {params}")

        try:
            topic_model = build_topic_model(params, random_state=config.RANDOM_STATE)
            topics, _ = topic_model.fit_transform(grid_docs, embeddings=grid_embeddings)
            topics = reduce_outliers_safely(topic_model, grid_docs, list(topics), grid_embeddings)

            topic_ids = sorted([t for t in set(topics) if t != -1])
            n_topics = len(topic_ids)
            outlier_rate = float(np.mean(np.array(topics) == -1))

            topic_words = topic_words_from_model(topic_model, top_n=10)
            diversity = compute_topic_diversity(topic_words)
            coherence = compute_coherence_cv(grid_docs, topic_words)
            score = score_model(n_topics, outlier_rate, coherence, diversity)

            row = {
                **params,
                "n_topics": n_topics,
                "outlier_rate": outlier_rate,
                "coherence_cv": coherence,
                "topic_diversity": diversity,
                "selection_score": score,
            }

            rows.append(row)

            print(
                f"    n_topics={n_topics}, outlier={outlier_rate:.3f}, "
                f"coherence={coherence:.3f}, diversity={diversity:.3f}, score={score:.3f}"
            )

            if score > best_score:
                best_score = score
                best_params = dict(params)

            del topic_model
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"    Failed: {e}")
            rows.append(
                {
                    **params,
                    "n_topics": np.nan,
                    "outlier_rate": np.nan,
                    "coherence_cv": np.nan,
                    "topic_diversity": np.nan,
                    "selection_score": -1e9,
                    "error": str(e),
                }
            )

    leaderboard = pd.DataFrame(rows).sort_values("selection_score", ascending=False)
    leaderboard.to_csv(output_dir / "tables" / "bertopic_hyperparameter_leaderboard.csv", index=False)

    if best_params is None:
        print("No successful grid model. Falling back to config.FINAL_PARAMS.")
        best_params = dict(config.FINAL_PARAMS)

    print(f"Selected params: {best_params}")
    return leaderboard, best_params


def fit_final_model(
    docs: List[str],
    years: List[int],
    ids: List[str],
    embeddings: np.ndarray,
    params: Dict[str, int],
    output_dir: Path,
) -> BERTopic:
    print("[4/8] Fitting final BERTopic model")

    topic_model = build_topic_model(params, random_state=config.RANDOM_STATE)
    topics, _ = topic_model.fit_transform(docs, embeddings=embeddings)
    topics = reduce_outliers_safely(topic_model, docs, list(topics), embeddings)

    print("[5/8] Saving topic assignments and metrics")

    table_dir = output_dir / "tables"
    model_dir = output_dir / "models" / "bertopic_showcase_plus"
    model_dir.mkdir(parents=True, exist_ok=True)

    topic_words = topic_words_from_model(topic_model, top_n=10)
    diversity = compute_topic_diversity(topic_words)
    coherence = compute_coherence_cv(docs, topic_words)

    topic_ids = sorted([t for t in set(topics) if t != -1])
    n_topics = len(topic_ids)
    outlier_rate = float(np.mean(np.array(topics) == -1))

    metrics = {
        **params,
        "embedding_model": config.EMBEDDING_MODEL_NAME,
        "n_documents": len(docs),
        "embedding_shape": list(embeddings.shape),
        "n_topics": n_topics,
        "outlier_rate": outlier_rate,
        "coherence_cv": coherence,
        "topic_diversity": diversity,
        "selection_score": score_model(n_topics, outlier_rate, coherence, diversity),
    }

    pd.DataFrame([metrics]).to_csv(table_dir / "bertopic_final_model_metrics.csv", index=False)

    doc_topics = pd.DataFrame(
        {
            "showcase_plus_id": ids,
            "analysis_year": years,
            "topic": topics,
            "topic_text": docs,
        }
    )
    doc_topics.to_csv(table_dir / "bertopic_document_topic_assignments.csv", index=False)

    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(table_dir / "bertopic_topic_info.csv", index=False)

    label_rows = []
    for topic_id in topic_ids:
        words = topic_words.get(topic_id, [])
        label_rows.append(
            {
                "topic": topic_id,
                "topic_words": ", ".join(words),
                "short_label": " / ".join(words[:4]),
            }
        )

    pd.DataFrame(label_rows).to_csv(table_dir / "bertopic_topic_labels.csv", index=False)

    try:
        topic_model.save(
            str(model_dir),
            serialization="safetensors",
            save_ctfidf=True,
            save_embedding_model=False,
        )
    except Exception as e:
        print(f"Model save with safetensors failed, trying default save. Error: {e}")

        try:
            topic_model.save(str(model_dir))
        except Exception as e2:
            print(f"Model save failed: {e2}")

    print("Final model metrics:")
    print(json.dumps(metrics, indent=2))

    return topic_model


def make_topic_year_tables_and_figures(
    topic_model: BERTopic,
    output_dir: Path,
    top_n: int,
) -> None:
    print("[6/8] Making dynamic topic tables and figures")

    table_dir = output_dir / "tables"
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    doc_topics = pd.read_csv(table_dir / "bertopic_document_topic_assignments.csv")
    doc_topics = doc_topics[doc_topics["topic"] != -1].copy()

    counts = (
        doc_topics.groupby(["analysis_year", "topic"])
        .size()
        .reset_index(name="n_papers")
    )

    yearly_total = (
        doc_topics.groupby("analysis_year")
        .size()
        .reset_index(name="year_total")
    )

    counts = counts.merge(yearly_total, on="analysis_year", how="left")
    counts["share"] = counts["n_papers"] / counts["year_total"]

    topic_total = doc_topics["topic"].value_counts().rename_axis("topic").reset_index(name="total_n")

    topic_first_last = []

    for topic, g in counts.groupby("topic"):
        g = g.sort_values("analysis_year")
        use_g = g[~g["analysis_year"].isin(getattr(config, "INCOMPLETE_YEARS", []))]

        if use_g.empty:
            use_g = g

        first_year = int(use_g["analysis_year"].min())
        last_year = int(use_g["analysis_year"].max())
        first_share = float(use_g.loc[use_g["analysis_year"] == first_year, "share"].mean())
        last_share = float(use_g.loc[use_g["analysis_year"] == last_year, "share"].mean())
        growth_ratio = (last_share + 1e-6) / (first_share + 1e-6)
        slope = np.polyfit(use_g["analysis_year"], use_g["share"], deg=1)[0] if len(use_g) >= 2 else 0.0

        topic_first_last.append(
            {
                "topic": topic,
                "first_year": first_year,
                "last_year": last_year,
                "first_share": first_share,
                "last_share": last_share,
                "growth_ratio": growth_ratio,
                "share_slope": slope,
            }
        )

    growth = topic_total.merge(pd.DataFrame(topic_first_last), on="topic", how="left")

    topic_words = topic_words_from_model(topic_model, top_n=8)
    growth["topic_words"] = growth["topic"].map(lambda t: ", ".join(topic_words.get(int(t), [])))
    growth["short_label"] = growth["topic"].map(lambda t: " / ".join(topic_words.get(int(t), [])[:4]))
    growth = growth.sort_values(["total_n", "share_slope"], ascending=[False, False])
    growth.to_csv(table_dir / "bertopic_topic_growth_table.csv", index=False)

    largest = growth.sort_values("total_n", ascending=False).head(max(8, top_n // 2))["topic"].tolist()
    eligible_growth = growth[growth["total_n"] >= 100].sort_values("share_slope", ascending=False)
    emerging = eligible_growth.head(top_n)["topic"].tolist()

    selected = []

    for t in largest + emerging:
        if t not in selected:
            selected.append(t)
        if len(selected) >= top_n:
            break

    selected = selected[:top_n]
    selected_growth = growth[growth["topic"].isin(selected)].copy()

    counts_sel = counts[counts["topic"].isin(selected)].copy()

    count_mat = counts_sel.pivot_table(
        index="analysis_year",
        columns="topic",
        values="n_papers",
        aggfunc="sum",
        fill_value=0,
    ).sort_index()

    share_mat = counts_sel.pivot_table(
        index="analysis_year",
        columns="topic",
        values="share",
        aggfunc="sum",
        fill_value=0,
    ).sort_index()

    label_map = {
        int(row["topic"]): f"T{int(row['topic'])}: {row['short_label']}"
        for _, row in selected_growth.iterrows()
    }

    count_mat_named = count_mat.rename(columns=label_map)
    share_mat_named = share_mat.rename(columns=label_map)

    count_mat_named.to_csv(table_dir / "bertopic_topic_year_counts_selected.csv")
    share_mat_named.to_csv(table_dir / "bertopic_topic_year_proportions_selected.csv")

    years = count_mat_named.index.to_numpy()
    y = share_mat_named.to_numpy().T
    labels = list(share_mat_named.columns)

    fig, ax = plt.subplots(figsize=(13, 8))
    ax.stackplot(years, y, labels=labels, alpha=0.88)
    ax.set_title("Dynamic topic shares in UKB Showcase+ publications")
    ax.set_xlabel("Publication year")
    ax.set_ylabel("Share of Showcase+ papers")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    save_mpl(fig, fig_dir, "bertopic_dynamic_streamgraph")

    fig, ax = plt.subplots(figsize=(13, 9))

    share_values = share_mat_named.copy()

    if len(share_values.index) >= 3:
        smooth_values = share_values.apply(lambda col: gaussian_filter1d(col.values, sigma=1.0), axis=0)
        smooth_df = pd.DataFrame(
            np.vstack(smooth_values.values).T,
            index=share_values.index,
            columns=share_values.columns,
        )
    else:
        smooth_df = share_values

    max_val = max(float(smooth_df.max().max()), 1e-6)
    offset = max_val * 1.25

    for i, col in enumerate(smooth_df.columns):
        baseline = i * offset
        vals = smooth_df[col].values
        ax.fill_between(smooth_df.index, baseline, baseline + vals, alpha=0.75)
        ax.plot(smooth_df.index, baseline + vals, linewidth=1)
        ax.text(
            smooth_df.index.min() - 0.2,
            baseline + max_val * 0.15,
            col,
            va="center",
            ha="right",
            fontsize=8,
        )

    ax.set_title("Topic evolution as ridge waves")
    ax.set_xlabel("Publication year")
    ax.set_yticks([])
    ax.set_xlim(smooth_df.index.min() - 1, smooth_df.index.max() + 0.5)
    save_mpl(fig, fig_dir, "bertopic_topic_ridge_waves")

    heat = share_mat_named.T

    fig_html = px.imshow(
        heat,
        aspect="auto",
        labels=dict(x="Publication year", y="Topic", color="Share"),
        title="BERTopic topic prevalence over time",
    )

    fig_html.write_html(fig_dir / "bertopic_topic_prevalence_heatmap.html")

    print(f"Selected {len(selected)} topics for figures.")
    print("Saved dynamic topic figures and matrices.")


def make_native_topics_over_time(
    topic_model: BERTopic,
    docs: List[str],
    years: List[int],
    output_dir: Path,
) -> None:
    print("[7/8] Running native BERTopic topics_over_time")

    table_dir = output_dir / "tables"
    fig_dir = output_dir / "figures"

    try:
        topics_over_time = topic_model.topics_over_time(
            docs,
            years,
            global_tuning=True,
            evolution_tuning=True,
            nr_bins=None,
        )

        topics_over_time.to_csv(table_dir / "bertopic_topics_over_time_native.csv", index=False)

        fig = topic_model.visualize_topics_over_time(
            topics_over_time,
            top_n_topics=config.TOP_N_TOPICS_FOR_FIGURES,
        )

        fig.write_html(fig_dir / "bertopic_native_topics_over_time.html")

        print("Saved native BERTopic topics-over-time outputs.")

    except Exception as e:
        print(f"Native topics_over_time failed, skipping. Error: {e}")


def write_manifest(
    input_csv: Path,
    output_dir: Path,
    params: Dict[str, int],
    n_docs: int,
) -> None:
    manifest = {
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "embedding_model": config.EMBEDDING_MODEL_NAME,
        "selected_params": params,
        "n_documents": n_docs,
        "min_year": config.MIN_YEAR,
        "incomplete_years": getattr(config, "INCOMPLETE_YEARS", []),
        "notes": [
            "Topic text is title + abstract.",
            "Embedding input uses light cleaning only.",
            "Corpus-generic words are removed from topic-word representation via CountVectorizer.",
            "One global BERTopic model is fitted across all years.",
        ],
    }

    with open(output_dir / "bertopic_run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run UKB Showcase+ BERTopic dynamic topic modelling.")
    parser.add_argument("--input_csv", type=str, default=None, help="Path to final Showcase+ CSV.")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory.")
    parser.add_argument("--no_grid", action="store_true", help="Skip hyperparameter grid and use config.FINAL_PARAMS.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_csv = Path(args.input_csv or config.SHOWCASE_PLUS_CSV_PATH)
    output_dir = Path(args.output_dir or config.OUTPUT_DIR)

    if "input your path" in str(input_csv):
        raise ValueError("Please edit config.py or pass --input_csv.")

    if "input your path" in str(output_dir):
        raise ValueError("Please edit config.py or pass --output_dir.")

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)

    for sub in ["tables", "figures", "models", "cache"]:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    df = prepare_input(input_csv, output_dir)

    docs = df["topic_text"].astype(str).tolist()
    years = df["analysis_year"].astype(int).tolist()
    ids = df["showcase_plus_id"].astype(str).tolist()

    embeddings = get_or_make_embeddings(docs, years, output_dir)

    if args.no_grid or not getattr(config, "RUN_TOPIC_GRID", True):
        print("[3/8] Skipping grid search; using config.FINAL_PARAMS")
        selected_params = dict(config.FINAL_PARAMS)
    else:
        _, selected_params = run_topic_grid(docs, embeddings, output_dir)

    topic_model = fit_final_model(
        docs=docs,
        years=years,
        ids=ids,
        embeddings=embeddings,
        params=selected_params,
        output_dir=output_dir,
    )

    make_topic_year_tables_and_figures(
        topic_model=topic_model,
        output_dir=output_dir,
        top_n=int(config.TOP_N_TOPICS_FOR_FIGURES),
    )

    make_native_topics_over_time(topic_model, docs, years, output_dir)

    write_manifest(
        input_csv=input_csv,
        output_dir=output_dir,
        params=selected_params,
        n_docs=len(docs),
    )

    print("[8/8] Done.")
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
