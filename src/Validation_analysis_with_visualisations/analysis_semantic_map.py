import argparse
import gc
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from common import ensure_real_path, make_text_for_analysis, savefig


def get_embeddings(texts, method="sentence-transformers/all-MiniLM-L6-v2"):
    if method == "tfidf_svd":
        return get_tfidf_svd_embeddings(texts), "TF-IDF + TruncatedSVD"
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(method)
        emb = model.encode(texts, batch_size=128, show_progress_bar=True, normalize_embeddings=True)
        del model
        gc.collect()
        return emb, method
    except Exception:
        return get_tfidf_svd_embeddings(texts), "TF-IDF + TruncatedSVD fallback"


def get_tfidf_svd_embeddings(texts):
    vec = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), min_df=3 if len(texts) >= 1000 else 2, max_df=0.9, max_features=12000)
    X = vec.fit_transform(texts)
    n_components = min(100, X.shape[1] - 1, X.shape[0] - 1)
    if n_components < 2:
        raise ValueError("Not enough features for semantic analysis.")
    emb = TruncatedSVD(n_components=n_components, random_state=42).fit_transform(X)
    return emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)


def run(three_true_path, rest_path, output_dir, max_per_group=3000, embedding_model="sentence-transformers/all-MiniLM-L6-v2", seed=42):
    os.makedirs(output_dir, exist_ok=True)
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    three_true = pd.read_csv(three_true_path)
    rest = pd.read_csv(rest_path)
    if len(three_true) > max_per_group:
        three_true = three_true.sample(n=max_per_group, random_state=seed)
    if len(rest) > max_per_group:
        rest = rest.sample(n=max_per_group, random_state=seed)
    df = pd.concat([
        three_true.assign(binary_split="Three-model TRUE agreement"),
        rest.assign(binary_split="Rest: not three-model TRUE agreement"),
    ], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    df["analysis_text"] = make_text_for_analysis(df).str.slice(0, 3500)
    labels = df["binary_split"].map({"Rest: not three-model TRUE agreement": 0, "Three-model TRUE agreement": 1}).values
    embeddings, used_method = get_embeddings(df["analysis_text"].tolist(), embedding_model)

    try:
        si = silhouette_score(embeddings, labels, metric="cosine")
    except Exception:
        si = np.nan
    emb_pos = embeddings[labels == 1]
    emb_rest = embeddings[labels == 0]
    c_pos = emb_pos.mean(axis=0)
    c_rest = emb_rest.mean(axis=0)
    c_pos = c_pos / max(np.linalg.norm(c_pos), 1e-12)
    c_rest = c_rest / max(np.linalg.norm(c_rest), 1e-12)
    centroid_cosine_similarity = float(np.dot(c_pos, c_rest))

    coords = PCA(n_components=2, random_state=seed).fit_transform(embeddings)
    df["semantic_x"] = coords[:, 0]
    df["semantic_y"] = coords[:, 1]
    df.to_csv(os.path.join(output_dir, "semantic_sample_with_coordinates.csv"), index=False)
    pd.DataFrame([
        {"metric": "embedding_method", "value": used_method},
        {"metric": "semantic_sample_size", "value": len(df)},
        {"metric": "SI_silhouette_index_cosine", "value": si},
        {"metric": "centroid_cosine_similarity", "value": centroid_cosine_similarity},
        {"metric": "centroid_cosine_distance", "value": 1 - centroid_cosine_similarity},
    ]).to_csv(os.path.join(output_dir, "semantic_metrics.csv"), index=False)

    plt.figure(figsize=(9, 7))
    for group in ["Rest: not three-model TRUE agreement", "Three-model TRUE agreement"]:
        sub = df[df["binary_split"].eq(group)]
        plt.scatter(sub["semantic_x"], sub["semantic_y"], s=12, alpha=0.55, label=f"{group} (n={len(sub):,})")
    plt.xlabel("Semantic component 1")
    plt.ylabel("Semantic component 2")
    plt.title(f"Semantic map\nSI / silhouette index = {si:.4f}")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    savefig(fig_dir, "semantic_map_TRUE_agreement_vs_rest.png")

    if "year_int" in df.columns and df["year_int"].notna().sum() > 0:
        plt.figure(figsize=(9, 7))
        sc = plt.scatter(df["semantic_x"], df["semantic_y"], c=pd.to_numeric(df["year_int"], errors="coerce"), s=12, alpha=0.60)
        plt.xlabel("Semantic component 1")
        plt.ylabel("Semantic component 2")
        plt.title("Semantic map coloured by publication year")
        plt.grid(alpha=0.25)
        plt.colorbar(sc, label="Publication year")
        plt.tight_layout()
        savefig(fig_dir, "semantic_map_by_year.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--three_true_path", default="INPUT your path for three_model_TRUE_agreement.csv here")
    p.add_argument("--rest_path", default="INPUT your path for rest_NOT_three_model_TRUE_agreement.csv here")
    p.add_argument("--output_dir", default="INPUT your path for analysis output folder here")
    p.add_argument("--max_per_group", type=int, default=3000)
    p.add_argument("--embedding_model", default="sentence-transformers/all-MiniLM-L6-v2")
    a = p.parse_args()
    run(
        ensure_real_path(a.three_true_path, "--three_true_path"),
        ensure_real_path(a.rest_path, "--rest_path"),
        ensure_real_path(a.output_dir, "--output_dir"),
        max_per_group=a.max_per_group,
        embedding_model=a.embedding_model,
    )


if __name__ == "__main__":
    main()
