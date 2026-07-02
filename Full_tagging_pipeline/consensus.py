import os

import numpy as np
import pandas as pd

from common import clean_year, extract_text, find_first_existing, normalise_bool_col, normalise_label_col
from tagger import prepare_config


def load_model_file(path, prefix, label_candidates, parse_candidates, raw_candidates):
    d = pd.read_csv(path)
    d["id"] = d["id"].astype(str).str.strip()
    d = d.drop_duplicates(subset=["id"], keep="first").reset_index(drop=True)

    label_col = find_first_existing(d.columns, label_candidates)
    parse_col = find_first_existing(d.columns, parse_candidates)
    raw_col = find_first_existing(d.columns, raw_candidates)
    if label_col is None:
        raise ValueError(f"Could not find label column for {prefix}: {d.columns.tolist()}")

    base = d[[c for c in ["id", "title", "abstract", "year"] if c in d.columns]].copy()
    if "title" in base.columns:
        base["title"] = base["title"].apply(extract_text)
    if "abstract" in base.columns:
        base["abstract"] = base["abstract"].apply(extract_text)
    if "year" in base.columns:
        base["year"] = pd.to_numeric(base["year"], errors="coerce")

    out = pd.DataFrame({"id": d["id"]})
    out[f"{prefix}_label"] = normalise_label_col(d[label_col])
    out[f"{prefix}_parse_ok"] = normalise_bool_col(d[parse_col]) if parse_col else out[f"{prefix}_label"].notna()
    out[f"{prefix}_raw_output"] = d[raw_col].fillna("").astype(str) if raw_col else ""
    return base, out


def combine_three_model_outputs(output_dir, qwen_config, llama_config, mistral_config, analysis_dir):
    os.makedirs(analysis_dir, exist_ok=True)
    qwen_cfg = prepare_config(qwen_config, output_dir)
    llama_cfg = prepare_config(llama_config, output_dir)
    mistral_cfg = prepare_config(mistral_config, output_dir)

    qwen_base, qwen_out = load_model_file(
        qwen_cfg["output_path"],
        "qwen",
        ["qwen_p2_label", "qwen_label"],
        ["qwen_p2_parse_ok", "qwen_parse_ok"],
        ["qwen_p2_raw_output", "qwen_raw_output"],
    )
    llama_base, llama_out = load_model_file(
        llama_cfg["output_path"],
        "llama3_8b",
        ["llama3_8b_p2_evalstyle_label", "llama3_8b_p2_label", "llama3_8b_label"],
        ["llama3_8b_p2_evalstyle_parse_ok", "llama3_8b_p2_parse_ok", "llama3_8b_parse_ok"],
        ["llama3_8b_p2_evalstyle_raw_output", "llama3_8b_p2_raw_output", "llama3_8b_raw_output"],
    )
    mistral_base, mistral_out = load_model_file(
        mistral_cfg["output_path"],
        "mistral_7b",
        ["mistral_7b_p2_evalstyle_label", "mistral_7b_p2_label", "mistral_7b_label"],
        ["mistral_7b_p2_evalstyle_parse_ok", "mistral_7b_p2_parse_ok", "mistral_7b_parse_ok"],
        ["mistral_7b_p2_evalstyle_raw_output", "mistral_7b_p2_raw_output", "mistral_7b_raw_output"],
    )

    base_all = pd.concat([qwen_base, llama_base, mistral_base], ignore_index=True, sort=False)
    base_all["id"] = base_all["id"].astype(str).str.strip()
    base_all = base_all[base_all["id"].ne("")]
    score = (
        base_all.get("title", pd.Series("", index=base_all.index)).fillna("").astype(str).str.len()
        + base_all.get("abstract", pd.Series("", index=base_all.index)).fillna("").astype(str).str.len()
        + base_all.get("year", pd.Series(np.nan, index=base_all.index)).notna().astype(int) * 100
    )
    base_all["_metadata_score"] = score
    base_all = base_all.sort_values("_metadata_score", ascending=False).drop_duplicates("id").drop(columns="_metadata_score")
    for c in ["title", "abstract"]:
        if c not in base_all.columns:
            base_all[c] = ""
        base_all[c] = base_all[c].fillna("").astype(str)
    if "year" not in base_all.columns:
        base_all["year"] = np.nan
    base_all["year"] = pd.to_numeric(base_all["year"], errors="coerce")

    combined = base_all.merge(qwen_out, on="id", how="left").merge(llama_out, on="id", how="left").merge(mistral_out, on="id", how="left")
    model_names = ["qwen", "llama3_8b", "mistral_7b"]
    for m in model_names:
        combined[f"{m}_label"] = combined.get(f"{m}_label", pd.Series(pd.NA, index=combined.index)).astype("Int64")
        combined[f"{m}_parse_ok"] = combined.get(f"{m}_parse_ok", pd.Series(False, index=combined.index)).fillna(False).astype(bool)
        combined[f"{m}_raw_output"] = combined.get(f"{m}_raw_output", pd.Series("", index=combined.index)).fillna("").astype(str)
        combined[f"{m}_true"] = combined[f"{m}_parse_ok"] & (combined[f"{m}_label"] == 1)
        combined[f"{m}_false"] = combined[f"{m}_parse_ok"] & (combined[f"{m}_label"] == 0)

    combined["year_int"] = combined["year"].apply(clean_year).astype("Int64")
    combined["n_models_parsed"] = sum(combined[f"{m}_parse_ok"].astype(int) for m in model_names)
    combined["n_true_votes"] = sum(combined[f"{m}_true"].astype(int) for m in model_names)
    combined["n_false_votes"] = sum(combined[f"{m}_false"].astype(int) for m in model_names)
    combined["three_model_TRUE_agreement"] = combined["qwen_true"] & combined["llama3_8b_true"] & combined["mistral_7b_true"]
    combined["three_model_FALSE_agreement"] = combined["qwen_false"] & combined["llama3_8b_false"] & combined["mistral_7b_false"]
    combined["all_three_parsed"] = combined["n_models_parsed"].eq(3)

    def signature(row):
        parts = []
        for m in model_names:
            if not bool(row[f"{m}_parse_ok"]):
                parts.append(f"{m}=NA")
            elif int(row[f"{m}_label"]) == 1:
                parts.append(f"{m}=T")
            else:
                parts.append(f"{m}=F")
        return " | ".join(parts)

    def group(row):
        if row["three_model_TRUE_agreement"]:
            return "Three-model TRUE agreement"
        if row["three_model_FALSE_agreement"]:
            return "Three-model FALSE agreement"
        if row["n_true_votes"] == 2:
            return "Two TRUE votes"
        if row["n_true_votes"] == 1:
            return "One TRUE vote"
        if row["n_true_votes"] == 0 and row["n_models_parsed"] > 0:
            return "No TRUE votes / parsed non-positive"
        return "No parsed model labels"

    combined["vote_signature"] = combined.apply(signature, axis=1)
    combined["consensus_group"] = combined.apply(group, axis=1)

    three_true = combined[combined["three_model_TRUE_agreement"]].copy()
    rest = combined[~combined["three_model_TRUE_agreement"]].copy()

    paths = {
        "combined": os.path.join(analysis_dir, "three_model_combined_labels.csv"),
        "three_true": os.path.join(analysis_dir, "three_model_TRUE_agreement.csv"),
        "rest": os.path.join(analysis_dir, "rest_NOT_three_model_TRUE_agreement.csv"),
    }
    combined.to_csv(paths["combined"], index=False)
    three_true.to_csv(paths["three_true"], index=False)
    rest.to_csv(paths["rest"], index=False)
    return paths
