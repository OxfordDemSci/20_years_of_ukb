import os
import time

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from common import (
    build_model_input,
    extract_first_json_obj,
    load_input_csv,
    load_llm,
    make_prompt_v2_balanced,
    normalise_bool_col,
    parse_llm_result,
    unload_model,
)


def prepare_config(config, output_dir):
    cfg = dict(config)
    cfg["output_path"] = os.path.join(output_dir, cfg["filename"])
    cfg["checkpoint_path"] = cfg["output_path"].replace(".csv", "_checkpoint.csv")
    cfg["stats_path"] = cfg["output_path"].replace(".csv", "_tagging_stats.csv")
    return cfg


def initialise_or_resume(df, cfg):
    run_key = cfg["run_key"]
    label_col = f"{run_key}_label"
    parse_col = f"{run_key}_parse_ok"
    raw_col = f"{run_key}_raw_output"

    if os.path.exists(cfg["checkpoint_path"]):
        ckpt = pd.read_csv(cfg["checkpoint_path"])
        ckpt["id"] = ckpt["id"].astype(str).str.strip()
        for c, default in [(label_col, np.nan), (parse_col, False), (raw_col, "")]:
            if c not in ckpt.columns:
                ckpt[c] = default
        out_df = df.merge(ckpt[["id", label_col, parse_col, raw_col]], on="id", how="left")
        out_df[parse_col] = normalise_bool_col(out_df[parse_col])
        out_df[label_col] = pd.to_numeric(out_df[label_col], errors="coerce")
        out_df[raw_col] = out_df[raw_col].fillna("").astype(str)
    else:
        out_df = df.copy()
        out_df[label_col] = np.nan
        out_df[parse_col] = False
        out_df[raw_col] = ""
    return out_df


def save_model_stats(out_df, cfg, input_path, runtime_s, batch_size, max_input_tokens, max_new_tokens, use_4bit):
    run_key = cfg["run_key"]
    label_col = f"{run_key}_label"
    parse_col = f"{run_key}_parse_ok"
    n_rows = len(out_df)
    n_parse_ok = int(out_df[parse_col].sum())
    n_true = int((out_df[label_col] == 1).sum())
    n_false = int((out_df[label_col] == 0).sum())

    stats_df = pd.DataFrame([
        {"metric": "model_id", "value": cfg["model_id"]},
        {"metric": "tag", "value": cfg["tag"]},
        {"metric": "prompt", "value": cfg["prompt_name"]},
        {"metric": "run_key", "value": run_key},
        {"metric": "input_path", "value": input_path},
        {"metric": "total_rows", "value": n_rows},
        {"metric": "unique_ids", "value": out_df["id"].nunique()},
        {"metric": "parsed_successfully", "value": n_parse_ok},
        {"metric": "parse_failed", "value": n_rows - n_parse_ok},
        {"metric": "parse_rate", "value": round(n_parse_ok / max(n_rows, 1), 6)},
        {"metric": "tagged_true_likely_ukb_use", "value": n_true},
        {"metric": "tagged_false_unlikely_ukb_use", "value": n_false},
        {"metric": "missing_labels", "value": int(out_df[label_col].isna().sum())},
        {"metric": "true_percentage_of_parsed", "value": round(n_true / max(n_parse_ok, 1) * 100, 4)},
        {"metric": "false_percentage_of_parsed", "value": round(n_false / max(n_parse_ok, 1) * 100, 4)},
        {"metric": "runtime_s", "value": round(runtime_s, 4)},
        {"metric": "runtime_min", "value": round(runtime_s / 60, 4)},
        {"metric": "batch_size", "value": batch_size},
        {"metric": "max_input_tokens", "value": max_input_tokens},
        {"metric": "max_new_tokens", "value": max_new_tokens},
        {"metric": "use_4bit", "value": use_4bit},
        {"metric": "output_path", "value": cfg["output_path"]},
        {"metric": "checkpoint_path", "value": cfg["checkpoint_path"]},
    ])
    stats_df.to_csv(cfg["stats_path"], index=False)
    return stats_df


def run_tagging_for_model(
    input_path,
    output_dir,
    config,
    batch_size=8,
    max_input_tokens=4096,
    max_new_tokens=80,
    checkpoint_every_batches=25,
    use_4bit=True,
):
    os.makedirs(output_dir, exist_ok=True)
    cfg = prepare_config(config, output_dir)
    df = load_input_csv(input_path)
    out_df = initialise_or_resume(df, cfg)

    run_key = cfg["run_key"]
    label_col = f"{run_key}_label"
    parse_col = f"{run_key}_parse_ok"
    raw_col = f"{run_key}_raw_output"
    remaining_idx = [i for i, ok in enumerate(out_df[parse_col].tolist()) if not bool(ok)]

    print(f"\n{cfg['tag']}: {len(remaining_idx):,} rows remaining")
    if not remaining_idx:
        out_df.to_csv(cfg["output_path"], index=False)
        save_model_stats(out_df, cfg, input_path, 0.0, batch_size, max_input_tokens, max_new_tokens, use_4bit)
        return cfg["output_path"]

    tokenizer, model = load_llm(cfg["model_id"], use_4bit=use_4bit)
    t0 = time.time()
    n_total = len(out_df)
    n_remaining = len(remaining_idx)
    num_batches = int(np.ceil(n_remaining / batch_size))

    for batch_no, start in enumerate(tqdm(range(0, n_remaining, batch_size), total=num_batches, desc=f"{cfg['tag']}-tag"), start=1):
        positions = remaining_idx[start:start + batch_size]
        batch = out_df.iloc[positions]
        prompts = [
            build_model_input(tokenizer, make_prompt_v2_balanced(r["title"], r["abstract"]))
            for _, r in batch.iterrows()
        ]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_input_tokens).to(model.device)
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen_only = output_ids[:, inputs["input_ids"].shape[1]:]
        texts = tokenizer.batch_decode(gen_only, skip_special_tokens=True)

        for pos, txt in zip(positions, texts):
            out_df.loc[pos, raw_col] = txt
            pred = parse_llm_result(extract_first_json_obj(txt))
            if pred is not None:
                out_df.loc[pos, label_col] = int(bool(pred))
                out_df.loc[pos, parse_col] = True
            else:
                out_df.loc[pos, label_col] = np.nan
                out_df.loc[pos, parse_col] = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if batch_no % checkpoint_every_batches == 0:
            out_df.to_csv(cfg["checkpoint_path"], index=False)
            print(f"checkpoint {batch_no}: parsed={int(out_df[parse_col].sum()):,}/{n_total:,}")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    runtime_s = time.time() - t0
    out_df.to_csv(cfg["checkpoint_path"], index=False)
    out_df.to_csv(cfg["output_path"], index=False)
    save_model_stats(out_df, cfg, input_path, runtime_s, batch_size, max_input_tokens, max_new_tokens, use_4bit)
    unload_model(model, tokenizer)
    return cfg["output_path"]
