import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

from common import ensure_real_path, make_text_for_analysis, savefig


def run(three_true_path, rest_path, output_dir, max_per_group=20000, seed=42):
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
    ], ignore_index=True)
    df["analysis_text"] = make_text_for_analysis(df)
    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), min_df=5 if len(df) >= 1000 else 2, max_df=0.85, max_features=12000)
    X = vectorizer.fit_transform(df["analysis_text"].fillna(""))
    terms = np.array(vectorizer.get_feature_names_out())
    pos_mask = df["binary_split"].eq("Three-model TRUE agreement").values
    rest_mask = df["binary_split"].eq("Rest: not three-model TRUE agreement").values
    pos_mean = np.asarray(X[pos_mask].mean(axis=0)).ravel()
    rest_mean = np.asarray(X[rest_mask].mean(axis=0)).ravel()
    out = pd.DataFrame({
        "term": terms,
        "mean_tfidf_three_model_TRUE_agreement": pos_mean,
        "mean_tfidf_rest": rest_mean,
        "difference_TRUE_minus_rest": pos_mean - rest_mean,
    }).sort_values("difference_TRUE_minus_rest", ascending=False)
    out.to_csv(os.path.join(output_dir, "tfidf_discriminative_terms.csv"), index=False)

    top_true = out.head(25).iloc[::-1]
    plt.figure(figsize=(10, 8))
    plt.barh(top_true["term"], top_true["difference_TRUE_minus_rest"])
    plt.xlabel("Mean TF-IDF difference: TRUE agreement minus rest")
    plt.title("Terms most associated with three-model TRUE agreement")
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    savefig(fig_dir, "tfidf_terms_associated_with_three_model_TRUE_agreement.png")

    top_rest = out.tail(25).sort_values("difference_TRUE_minus_rest", ascending=True).iloc[::-1]
    plt.figure(figsize=(10, 8))
    plt.barh(top_rest["term"], -top_rest["difference_TRUE_minus_rest"])
    plt.xlabel("Mean TF-IDF difference: rest minus TRUE agreement")
    plt.title("Terms most associated with the rest")
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    savefig(fig_dir, "tfidf_terms_associated_with_rest.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--three_true_path", default="INPUT your path for three_model_TRUE_agreement.csv here")
    p.add_argument("--rest_path", default="INPUT your path for rest_NOT_three_model_TRUE_agreement.csv here")
    p.add_argument("--output_dir", default="INPUT your path for analysis output folder here")
    p.add_argument("--max_per_group", type=int, default=20000)
    a = p.parse_args()
    run(
        ensure_real_path(a.three_true_path, "--three_true_path"),
        ensure_real_path(a.rest_path, "--rest_path"),
        ensure_real_path(a.output_dir, "--output_dir"),
        max_per_group=a.max_per_group,
    )


if __name__ == "__main__":
    main()
