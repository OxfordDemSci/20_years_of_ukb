"""Saved three-model label analyses from the consolidated dataset notebook.

No tagging models are loaded. The optional semantic diagnostic reuses matching
saved coordinates, or runs only when explicitly enabled.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
from utils import shared_paths as P
from utils.shared_analysis_window import filter_analysis_window

MODEL_NAMES = ["qwen", "llama3_8b", "mistral_7b"]
COMBINED_LABELS_FILENAME = "matched_ukb_full_final_2013_2025_three_model_labels.csv"

def normalise_label(series):
    result = pd.Series(pd.NA, index=series.index, dtype="Int64")
    numeric = pd.to_numeric(series, errors="coerce")
    result.loc[numeric.eq(1)] = 1
    result.loc[numeric.eq(0)] = 0
    text = series.astype(str).str.strip().str.lower()
    result.loc[text.isin(["true", "yes", "y", "1", "1.0"])] = 1
    result.loc[text.isin(["false", "no", "n", "0", "0.0"])] = 0
    return result

def normalise_bool(series):
    if series.dtype == bool:
        return series.fillna(False).astype(bool)
    text = series.astype(str).str.strip().str.lower()
    return text.map({
        "true": True, "1": True, "1.0": True, "yes": True, "y": True,
        "false": False, "0": False, "0.0": False, "no": False, "n": False,
        "nan": False, "none": False, "<na>": False, "": False,
    }).fillna(False).astype(bool)

def clean_year(series):
    years = pd.to_numeric(series, errors="coerce")
    return years.where(years.between(1900, 2035) & years.mod(1).eq(0)).astype("Int64")


def resolve_combined_labels(input_path=None):
    value = input_path or os.environ.get("UKB_COMBINED_LABELS_CSV", "").strip()
    if value:
        path = Path(value).expanduser()
        path = path if path.is_absolute() else P.ROOT / path
        if not path.is_file():
            raise FileNotFoundError(f"Combined-label CSV not found: {path}")
        return path
    preferred = P.DATA / "analysis" / "dataset" / COMBINED_LABELS_FILENAME
    if preferred.is_file():
        return preferred
    names = {COMBINED_LABELS_FILENAME,
             "matched_ukb_full_final_since_2014_three_model_combined_labels.csv",
             "three_model_combined_labels.csv"}
    found = sorted(path for path in P.DATA.rglob("*labels.csv") if path.name in names)
    if len(found) > 1:
        raise ValueError("Multiple combined-label CSVs found; set UKB_COMBINED_LABELS_CSV explicitly.")
    return found[0] if found else None


def prepare_candidates(frame):
    """Normalise saved labels, filtering dates before deduplicating paper IDs."""
    required = {"id", "title", "abstract", "year"}
    for model in MODEL_NAMES:
        required.update({f"{model}_label", f"{model}_parse_ok"})
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Combined-label CSV is missing required columns: {missing}")
    combined = frame.copy()
    combined["id"] = combined["id"].astype("string").str.strip()
    combined = combined.loc[combined["id"].notna() & combined["id"].ne("")].copy()
    for column in ("title", "abstract"):
        combined[column] = combined[column].fillna("").astype(str)
    combined["year_int"] = clean_year(combined["year"])
    for column in ("date", "publication_date"):
        if column in combined:
            dates = pd.to_datetime(combined[column].astype("string"), errors="coerce", format="mixed", utc=True)
            combined["year_int"] = combined["year_int"].fillna(dates.dt.year.astype("Int64"))
            break
    combined = filter_analysis_window(combined, year_col="year_int")
    combined = combined.drop_duplicates("id", keep="first").reset_index(drop=True)
    for model in MODEL_NAMES:
        labels = normalise_label(combined[f"{model}_label"])
        combined[f"{model}_label"] = labels
        # An unusable label cannot count as a parsed TRUE or FALSE response.
        parsed = normalise_bool(combined[f"{model}_parse_ok"]) & labels.notna()
        combined[f"{model}_parse_ok"] = parsed
        combined[f"{model}_true"] = (parsed & labels.eq(1)).fillna(False).astype(bool)
        combined[f"{model}_false"] = (parsed & labels.eq(0)).fillna(False).astype(bool)
    combined["n_models_parsed"] = sum(combined[f"{model}_parse_ok"].astype(int) for model in MODEL_NAMES)
    combined["n_true_votes"] = sum(combined[f"{model}_true"].astype(int) for model in MODEL_NAMES)
    combined["n_false_votes"] = sum(combined[f"{model}_false"].astype(int) for model in MODEL_NAMES)
    combined["all_three_parsed"] = combined["n_models_parsed"].eq(3)
    combined["three_model_TRUE_agreement"] = combined["n_true_votes"].eq(3)
    combined["three_model_FALSE_agreement"] = combined["n_false_votes"].eq(3)
    combined["vote_signature"] = combined.apply(lambda row: " | ".join(
        f"{model}=" + ("NA" if not row[f"{model}_parse_ok"] else "T" if row[f"{model}_label"] == 1 else "F")
        for model in MODEL_NAMES), axis=1) if len(combined) else pd.Series(dtype=str)
    combined["consensus_group"] = np.select([
        combined["three_model_TRUE_agreement"], combined["three_model_FALSE_agreement"],
        combined["n_true_votes"].eq(2), combined["n_true_votes"].eq(1),
        combined["n_models_parsed"].gt(0),
    ], ["Three-model TRUE agreement", "Three-model FALSE agreement", "Two TRUE votes",
        "One TRUE vote", "No TRUE votes / parsed non-positive"], default="No parsed model labels")
    return combined


def _semantic_cache(output_dir, sample):
    """Reuse coordinates only for the identical sampled texts, labels and years."""
    coordinates = output_dir / "semantic_sample_with_coordinates.csv"
    metrics = output_dir / "semantic_metrics.csv"
    if not coordinates.is_file() or not metrics.is_file():
        return None
    cached = pd.read_csv(coordinates, dtype={"id": "string"})
    columns = ["id", "analysis_text", "binary_split", "year_int"]
    if not set(columns + ["semantic_x", "semantic_y"]).issubset(cached):
        raise ValueError("Saved semantic coordinates are malformed; no model was started.")
    actual = cached[columns].astype("string").fillna("").reset_index(drop=True)
    expected = sample[columns].astype("string").fillna("").reset_index(drop=True)
    if not actual.equals(expected):
        return None
    if not np.isfinite(cached[["semantic_x", "semantic_y"]].to_numpy(dtype=float)).all():
        raise ValueError("Saved semantic coordinates contain non-finite values.")
    summary = pd.read_csv(metrics)
    if not {"metric", "value"}.issubset(summary) or "SI_silhouette_index_cosine" not in set(summary.metric):
        raise ValueError("Saved semantic metrics are malformed.")
    return cached, summary


def run_agreement(input_path=None, *, output_dir=None, figure_dir=None,
                  run_semantic=False, max_tfidf_per_group=20_000,
                  max_semantic_per_group=3_000, seed=42, show_figures=True,
                  show_tables=False):
    """Run all original agreement/text analyses, with optional semantic encoding."""
    INPUT_PATH = resolve_combined_labels(input_path)
    if INPUT_PATH is None:
        reason = "Full candidate-level labels missing; set UKB_COMBINED_LABELS_CSV."
        print(f"[SKIP] Three-model agreement: {reason}")
        return {"section": "Three-model agreement", "status": "SKIP", "detail": reason}
    OUTPUT_DIR = Path(output_dir) if output_dir is not None else P.TABLE_DATA_ANALYSIS / "00_dataset" / "three_model_agreement"
    FIGURE_DIR = Path(figure_dir) if figure_dir is not None else P.FIG_DATA_ANALYSIS / "00_dataset" / "three_model_agreement"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    RUN_SEMANTIC_ANALYSIS = bool(run_semantic)
    MAX_TFIDF_PER_GROUP = int(max_tfidf_per_group)
    MAX_SEMANTIC_PER_GROUP = int(max_semantic_per_group)
    SEED = int(seed)

    import gc
    import re
    from IPython.display import display as _display

    def display(value):
        if show_tables:
            _display(value)
    from .data_analysis_00_figures import render_agreement_figures
    from sklearn.decomposition import PCA, TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import silhouette_score

    MODEL_NAMES = ["qwen", "llama3_8b", "mistral_7b"]
    MODEL_DISPLAY = {"qwen": "Qwen2.5-7B", "llama3_8b": "Llama3-8B", "mistral_7b": "Mistral-7B"}


    def analysis_text(frame):
        return (
            frame["title"].fillna("").astype(str).str.strip()
            + "\n"
            + frame["abstract"].fillna("").astype(str).str.strip()
        )


    def save_table(frame, filename):
        path = OUTPUT_DIR / filename
        frame.to_csv(path, index=False)
        # Paths are summarised once at the end.
        return path


    # Load and normalise saved candidate labels.
    combined = prepare_candidates(pd.read_csv(INPUT_PATH, low_memory=False, dtype={"id": "string"}))
    three_true = combined[combined["three_model_TRUE_agreement"]].copy()
    rest = combined[~combined["three_model_TRUE_agreement"]].copy()
    if three_true.empty or rest.empty:
        raise ValueError("Both consensus groups are required; use the full candidate-level export within 2013–2025.")
    save_table(combined, "combined_consensus_normalised.csv")
    save_table(three_true, "three_model_TRUE_agreement.csv")
    save_table(rest, "rest_NOT_three_model_TRUE_agreement.csv")


    # 5. Dataset overview
    overview = pd.DataFrame([
        {"metric": "total_candidates", "value": len(combined)},
        {"metric": "unique_ids", "value": combined["id"].nunique()},
        {"metric": "three_model_TRUE_agreement", "value": len(three_true)},
        {"metric": "rest_NOT_three_model_TRUE_agreement", "value": len(rest)},
        {"metric": "three_model_TRUE_agreement_percent", "value": 100 * len(three_true) / len(combined)},
        {"metric": "three_model_FALSE_agreement", "value": int(combined["three_model_FALSE_agreement"].sum())},
        {"metric": "all_three_parsed", "value": int(combined["all_three_parsed"].sum())},
        {"metric": "earliest_year", "value": combined["year_int"].min()},
        {"metric": "latest_year", "value": combined["year_int"].max()},
    ])
    save_table(overview, "dataset_overview.csv")
    display(overview)


    # 6. Consensus statistics and distributions
    model_summary = []
    for model in MODEL_NAMES:
        parsed = int(combined[f"{model}_parse_ok"].sum())
        true = int(combined[f"{model}_true"].sum())
        false = int(combined[f"{model}_false"].sum())
        model_summary.append({
            "model": model,
            "display_name": MODEL_DISPLAY[model],
            "parsed": parsed,
            "true": true,
            "false": false,
            "parse_rate_percent": 100 * parsed / len(combined),
            "true_percent_among_parsed": 100 * true / parsed if parsed else np.nan,
        })

    model_summary = pd.DataFrame(model_summary)
    vote_distribution = combined["n_true_votes"].value_counts().sort_index().rename_axis("n_true_votes").reset_index(name="n_candidates")
    group_distribution = combined["consensus_group"].value_counts().rename_axis("consensus_group").reset_index(name="n_candidates")
    signature_distribution = combined["vote_signature"].value_counts().rename_axis("vote_signature").reset_index(name="n_candidates")

    save_table(model_summary, "model_positive_rate_summary.csv")
    save_table(vote_distribution, "true_vote_distribution.csv")
    save_table(group_distribution, "consensus_group_distribution.csv")
    save_table(signature_distribution, "vote_signature_distribution.csv")

    display(model_summary)
    display(vote_distribution)
    display(group_distribution)
    display(signature_distribution)


    # 10. Pairwise model agreement
    agreement_matrix = pd.DataFrame(np.nan, index=MODEL_NAMES, columns=MODEL_NAMES)
    for model in MODEL_NAMES:
        if combined[f"{model}_parse_ok"].any():
            agreement_matrix.loc[model, model] = 1.0
    pairwise_rows = []

    for index, model_a in enumerate(MODEL_NAMES):
        for model_b in MODEL_NAMES[index + 1:]:
            mask = combined[f"{model_a}_parse_ok"] & combined[f"{model_b}_parse_ok"]
            agreement = (
                combined.loc[mask, f"{model_a}_label"].astype(int).to_numpy()
                == combined.loc[mask, f"{model_b}_label"].astype(int).to_numpy()
            ).mean() if mask.any() else np.nan
            agreement_matrix.loc[model_a, model_b] = agreement
            agreement_matrix.loc[model_b, model_a] = agreement
            pairwise_rows.append({
                "model_a": model_a,
                "model_b": model_b,
                "n_both_parsed": int(mask.sum()),
                "agreement_percent": 100 * agreement,
            })

    pairwise_agreement = pd.DataFrame(pairwise_rows)
    save_table(pairwise_agreement, "pairwise_model_agreement.csv")
    display(pairwise_agreement)

    # 11. Yearly consensus table and TRUE-versus-rest counts
    year_data = combined.dropna(subset=["year_int"]).copy()
    year_data["year_int"] = year_data["year_int"].astype(int)

    yearly = year_data.groupby("year_int").agg(
        total_candidates=("id", "count"),
        three_model_TRUE_agreement=("three_model_TRUE_agreement", "sum"),
        three_model_FALSE_agreement=("three_model_FALSE_agreement", "sum"),
        qwen_TRUE=("qwen_true", "sum"),
        llama3_8b_TRUE=("llama3_8b_true", "sum"),
        mistral_7b_TRUE=("mistral_7b_true", "sum"),
    ).reset_index()
    yearly["rest_NOT_three_model_TRUE_agreement"] = yearly["total_candidates"] - yearly["three_model_TRUE_agreement"]
    yearly["three_model_TRUE_agreement_rate_percent"] = 100 * yearly["three_model_TRUE_agreement"] / yearly["total_candidates"]
    yearly["rest_rate_percent"] = 100 - yearly["three_model_TRUE_agreement_rate_percent"]

    save_table(yearly, "yearly_three_model_consensus_trend.csv")
    display(yearly)

    # 14. Explicit UK Biobank mentions
    ukb_pattern = re.compile(
        r"\b(?:uk\s*biobank|u\.?k\.?\s*biobank|united\s+kingdom\s+biobank|ukb|ukbb)\b",
        flags=re.I,
    )
    combined["analysis_text"] = analysis_text(combined)
    combined["explicit_ukb_title_or_abstract"] = combined["analysis_text"].str.contains(ukb_pattern, na=False)
    combined["binary_split"] = np.where(
        combined["three_model_TRUE_agreement"],
        "Three-model TRUE agreement",
        "Rest: not three-model TRUE agreement",
    )

    explicit_summary = combined.groupby("binary_split").agg(
        n=("id", "count"),
        explicit_ukb=("explicit_ukb_title_or_abstract", "sum"),
    ).reset_index()
    explicit_summary["explicit_ukb_percent"] = 100 * explicit_summary["explicit_ukb"] / explicit_summary["n"]
    save_table(explicit_summary, "explicit_ukb_mention_summary.csv")
    display(explicit_summary)

    # 15. Keyword-category profile
    CATEGORY_PATTERNS = {
        "explicit UKB": r"\b(?:uk\s*biobank|u\.?k\.?\s*biobank|united\s+kingdom\s+biobank|ukb|ukbb)\b",
        "generic biobank": r"\b(?:biobank|biobanking|biobanks)\b",
        "UK population cue": r"\b(?:uk|british|england|scotland|wales|united kingdom)\b",
        "cohort / participants": r"\b(?:cohort|participants|population[- ]based|prospective|baseline assessment)\b",
        "genetics / genomics": r"\b(?:genetic|genomic|genotype|genotyping|gwas|polygenic|exome|sequencing)\b",
        "imaging": r"\b(?:imaging|mri|brain imaging|cardiac imaging|radiomics)\b",
        "linked records / EHR": r"\b(?:linked records|hospital episode statistics|hes|electronic health records|ehr|registry|registries)\b",
        "machine learning": r"\b(?:machine learning|deep learning|artificial intelligence|neural network|prediction model)\b",
        "cardiometabolic": r"\b(?:cardiovascular|heart|diabetes|obesity|metabolic|hypertension)\b",
        "cancer": r"\b(?:cancer|tumou?r|oncology|carcinoma|neoplasm)\b",
        "mental health / brain": r"\b(?:depression|anxiety|psychiatric|mental health|brain|cognition|dementia)\b",
        "other named biobank": r"\b(?:china kadoorie|biobank japan|finn?gen|all of us|million veteran|lifelines|decode|cartagene)\b",
    }

    category_rows = []
    for group_name, group in combined.groupby("binary_split"):
        text = analysis_text(group)
        for category, pattern in CATEGORY_PATTERNS.items():
            hits = text.str.contains(pattern, case=False, regex=True, na=False)
            category_rows.append({
                "binary_split": group_name,
                "category": category,
                "n_group": len(group),
                "n_with_category": int(hits.sum()),
                "percent_with_category": 100 * hits.mean(),
            })

    category_summary = pd.DataFrame(category_rows)
    save_table(category_summary, "keyword_category_summary.csv")
    display(category_summary)

    # 16. TF-IDF discriminative terms
    tfidf_true = three_true.sample(n=min(len(three_true), MAX_TFIDF_PER_GROUP), random_state=SEED)
    tfidf_rest = rest.sample(n=min(len(rest), MAX_TFIDF_PER_GROUP), random_state=SEED)
    tfidf_data = pd.concat([
        tfidf_true.assign(binary_split="Three-model TRUE agreement"),
        tfidf_rest.assign(binary_split="Rest: not three-model TRUE agreement"),
    ], ignore_index=True)
    tfidf_data["analysis_text"] = analysis_text(tfidf_data)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=5 if len(tfidf_data) >= 1_000 else 2,
        max_df=0.85,
        max_features=12_000,
    )
    matrix = vectorizer.fit_transform(tfidf_data["analysis_text"])
    terms = np.asarray(vectorizer.get_feature_names_out())
    true_mask = tfidf_data["binary_split"].eq("Three-model TRUE agreement").to_numpy()
    true_mean = np.asarray(matrix[true_mask].mean(axis=0)).ravel()
    rest_mean = np.asarray(matrix[~true_mask].mean(axis=0)).ravel()

    tfidf_terms = pd.DataFrame({
        "term": terms,
        "mean_tfidf_three_model_TRUE_agreement": true_mean,
        "mean_tfidf_rest": rest_mean,
        "difference_TRUE_minus_rest": true_mean - rest_mean,
    }).sort_values("difference_TRUE_minus_rest", ascending=False)
    save_table(tfidf_terms, "tfidf_discriminative_terms.csv")

    # Optional semantic separation diagnostic.
    semantic_true = three_true.sample(n=min(len(three_true), MAX_SEMANTIC_PER_GROUP), random_state=SEED)
    semantic_rest = rest.sample(n=min(len(rest), MAX_SEMANTIC_PER_GROUP), random_state=SEED)
    semantic_data = pd.concat([
        semantic_true.assign(binary_split="Three-model TRUE agreement"),
        semantic_rest.assign(binary_split="Rest: not three-model TRUE agreement"),
    ], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)
    semantic_data["analysis_text"] = analysis_text(semantic_data).str.slice(0, 3_500)
    labels = semantic_data["binary_split"].eq("Three-model TRUE agreement").astype(int).to_numpy()

    cached_semantic = _semantic_cache(OUTPUT_DIR, semantic_data)
    semantic_available = cached_semantic is not None or RUN_SEMANTIC_ANALYSIS
    if cached_semantic is not None:
        semantic_data, semantic_metrics = cached_semantic
        silhouette = float(semantic_metrics.set_index("metric").loc["SI_silhouette_index_cosine", "value"])
        print("[SKIP] Semantic encoding: using matching saved coordinates and metrics.")
    elif RUN_SEMANTIC_ANALYSIS:
        embedding_method = "sentence-transformers/all-MiniLM-L6-v2"
        try:
            from sentence_transformers import SentenceTransformer
            embedder = SentenceTransformer(embedding_method)
            embeddings = embedder.encode(
                semantic_data["analysis_text"].tolist(),
                batch_size=128,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            del embedder
            gc.collect()
        except Exception as error:
            print(f"Sentence embeddings unavailable ({error}); using TF-IDF/SVD fallback.")
            fallback = TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                min_df=3,
                max_df=0.9,
                max_features=12_000,
            )
            text_matrix = fallback.fit_transform(semantic_data["analysis_text"])
            components = min(100, text_matrix.shape[0] - 1, text_matrix.shape[1] - 1)
            embeddings = TruncatedSVD(n_components=components, random_state=SEED).fit_transform(text_matrix)
            embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
            embedding_method = "TF-IDF + TruncatedSVD fallback"

        silhouette = silhouette_score(embeddings, labels, metric="cosine")
        true_centroid = embeddings[labels == 1].mean(axis=0)
        rest_centroid = embeddings[labels == 0].mean(axis=0)
        true_centroid /= max(np.linalg.norm(true_centroid), 1e-12)
        rest_centroid /= max(np.linalg.norm(rest_centroid), 1e-12)
        centroid_similarity = float(np.dot(true_centroid, rest_centroid))

        coordinates = PCA(n_components=2, random_state=SEED).fit_transform(embeddings)
        semantic_data[["semantic_x", "semantic_y"]] = coordinates
        semantic_metrics = pd.DataFrame([
            {"metric": "embedding_method", "value": embedding_method},
            {"metric": "semantic_sample_size", "value": len(semantic_data)},
            {"metric": "semantic_sample_three_model_TRUE_agreement", "value": int(labels.sum())},
            {"metric": "semantic_sample_rest", "value": int((labels == 0).sum())},
            {"metric": "SI_silhouette_index_cosine", "value": silhouette},
            {"metric": "centroid_cosine_similarity", "value": centroid_similarity},
            {"metric": "centroid_cosine_distance", "value": 1 - centroid_similarity},
        ])
        save_table(semantic_data, "semantic_sample_with_coordinates.csv")
        save_table(semantic_metrics, "semantic_metrics.csv")
        display(semantic_metrics)


    else:
        print("[SKIP] Semantic encoding: no matching cache; enable RUN_SEMANTIC_ANALYSIS to compute it.")
    figure_files = render_agreement_figures(
        model_summary=model_summary, vote_distribution=vote_distribution,
        agreement_matrix=agreement_matrix, yearly=yearly,
        category_summary=category_summary, tfidf_terms=tfidf_terms, tfidf_data=tfidf_data,
        semantic_data=semantic_data if semantic_available else None,
        semantic_metrics=semantic_metrics if semantic_available else None,
        figure_dir=FIGURE_DIR, show_figures=show_figures,
    )
    table_files = sorted(OUTPUT_DIR.glob("*.csv"))
    figure_count = sum(path.suffix == ".png" for path in figure_files)
    print(f"[PASS] Three-model agreement: {len(combined):,} candidates; {len(table_files)} CSVs, {figure_count} combined figures.")
    return {"section": "Three-model agreement", "status": "PASS", "detail": f"{len(combined):,} candidates; {len(three_true):,} unanimous TRUE",
            "semantic_status": "PASS" if semantic_available else "SKIP", "tables": [str(p) for p in table_files], "figures": [str(p) for p in figure_files]}
