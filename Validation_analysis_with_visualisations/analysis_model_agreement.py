import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt

from common import ensure_real_path, savefig


def run(combined_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    df = pd.read_csv(combined_path)
    models = ["qwen", "llama3_8b", "mistral_7b"]

    rows = []
    for m in models:
        parsed = int(df[f"{m}_parse_ok"].sum())
        true = int(df[f"{m}_true"].sum())
        false = int(df[f"{m}_false"].sum())
        rows.append({
            "model": m,
            "parsed": parsed,
            "true": true,
            "false": false,
            "true_percent_among_parsed": true / max(parsed, 1) * 100,
        })
    model_summary = pd.DataFrame(rows)
    model_summary.to_csv(os.path.join(output_dir, "model_positive_rate_summary.csv"), index=False)

    plt.figure(figsize=(8.5, 5))
    plt.bar(model_summary["model"], model_summary["true_percent_among_parsed"])
    plt.ylabel("TRUE percentage among parsed rows")
    plt.title("Model-level positive rate")
    plt.grid(axis="y", alpha=0.25)
    for i, v in enumerate(model_summary["true_percent_among_parsed"]):
        plt.text(i, v, f"{v:.1f}%", ha="center", va="bottom")
    plt.tight_layout()
    savefig(fig_dir, "model_positive_rate_summary.png")

    vote_dist = df["n_true_votes"].value_counts().sort_index().rename_axis("n_true_votes").reset_index(name="n_candidates")
    vote_dist.to_csv(os.path.join(output_dir, "true_vote_distribution.csv"), index=False)
    plt.figure(figsize=(8, 5))
    plt.bar(vote_dist["n_true_votes"].astype(str), vote_dist["n_candidates"])
    plt.xlabel("Number of models predicting TRUE")
    plt.ylabel("Number of candidates")
    plt.title("TRUE-vote distribution")
    plt.grid(axis="y", alpha=0.25)
    for i, v in enumerate(vote_dist["n_candidates"]):
        plt.text(i, v, f"{v:,}", ha="center", va="bottom")
    plt.tight_layout()
    savefig(fig_dir, "true_vote_distribution.png")

    group_dist = df["consensus_group"].value_counts().rename_axis("consensus_group").reset_index(name="n_candidates")
    group_dist.to_csv(os.path.join(output_dir, "consensus_group_distribution.csv"), index=False)
    plot = group_dist.sort_values("n_candidates", ascending=True)
    plt.figure(figsize=(10, 6))
    plt.barh(plot["consensus_group"], plot["n_candidates"])
    plt.xlabel("Number of candidates")
    plt.title("Consensus group distribution")
    plt.grid(axis="x", alpha=0.25)
    for i, v in enumerate(plot["n_candidates"]):
        plt.text(v, i, f" {v:,}", va="center")
    plt.tight_layout()
    savefig(fig_dir, "consensus_group_distribution.png")

    stats = []
    stats.extend([
        {"section": "overall", "metric": "total_candidates", "value": len(df)},
        {"section": "overall", "metric": "three_model_TRUE_agreement", "value": int(df["three_model_TRUE_agreement"].sum())},
        {"section": "overall", "metric": "rest_NOT_three_model_TRUE_agreement", "value": int((~df["three_model_TRUE_agreement"]).sum())},
        {"section": "overall", "metric": "all_three_parsed", "value": int(df["all_three_parsed"].sum())},
    ])
    for a_i, a in enumerate(models):
        for b in models[a_i + 1:]:
            mask = df[f"{a}_parse_ok"] & df[f"{b}_parse_ok"]
            agreement = float((df.loc[mask, f"{a}_label"].astype(int).values == df.loc[mask, f"{b}_label"].astype(int).values).mean()) if mask.sum() else float("nan")
            stats.append({"section": "pairwise_agreement", "metric": f"{a}_vs_{b}_n", "value": int(mask.sum())})
            stats.append({"section": "pairwise_agreement", "metric": f"{a}_vs_{b}_agreement_percent", "value": round(agreement * 100, 4)})
    pd.DataFrame(stats).to_csv(os.path.join(output_dir, "three_model_consensus_stats.csv"), index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--combined_path", default="INPUT your path for three_model_combined_labels.csv here")
    p.add_argument("--output_dir", default="INPUT your path for analysis output folder here")
    a = p.parse_args()
    run(ensure_real_path(a.combined_path, "--combined_path"), ensure_real_path(a.output_dir, "--output_dir"))


if __name__ == "__main__":
    main()
