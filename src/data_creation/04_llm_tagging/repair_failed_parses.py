import os
import shutil
import time

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from common import (
    build_model_input,
    extract_first_json_obj,
    extract_text,
    load_llm,
    make_prompt_v2_balanced,
    normalise_bool_col,
    parse_llm_result,
    robust_parse_llm_text,
    unload_model,
)
from tagger import prepare_config


def repair_failed_parses_for_model(
    output_dir,
    config,
    batch_size=8,
    max_input_tokens=4096,
    max_new_tokens=80,
    max_repair_attempts=3,
    checkpoint_every_batches=10,
    use_4bit=True,
):
    cfg = prepare_config(config, output_dir)
    run_key = cfg["run_key"]
    label_col = f"{run_key}_label"
    parse_col = f"{run_key}_parse_ok"
    raw_col = f"{run_key}_raw_output"

    if not os.path.exists(cfg["output_path"]):
        raise FileNotFoundError(cfg["output_path"])

    out_df = pd.read_csv(cfg["output_path"])
    out_df["id"] = out_df["id"].astype(str).str.strip()
    out_df["title"] = out_df["title"].apply(extract_text)
    out_df["abstract"] = out_df["abstract"].apply(extract_text)
    out_df["year"] = pd.to_numeric(out_df["year"], errors="coerce")

    for c, default in [(label_col, np.nan), (parse_col, False), (raw_col, "")]:
        if c not in out_df.columns:
            out_df[c] = default
    out_df[parse_col] = normalise_bool_col(out_df[parse_col])
    out_df[label_col] = pd.to_numeric(out_df[label_col], errors="coerce")
    out_df[raw_col] = out_df[raw_col].fillna("").astype(str)

    backup_path = cfg["output_path"].replace(".csv", "_before_parse_repair_backup.csv")
    if not os.path.exists(backup_path):
        shutil.copy2(cfg["output_path"], backup_path)

    before_ok = int(out_df[parse_col].sum())
    initial_failed_idx = out_df.index[(~out_df[parse_col]) | out_df[label_col].isna()].tolist()
    recovered_from_raw = 0

    for idx in initial_failed_idx:
        pred = robust_parse_llm_text(out_df.loc[idx, raw_col])
        if pred is not None:
            out_df.loc[idx, label_col] = int(bool(pred))
            out_df.loc[idx, parse_col] = True
            recovered_from_raw += 1

    out_df.to_csv(cfg["output_path"], index=False)
    out_df.to_csv(cfg["checkpoint_path"], index=False)

    remaining = out_df.index[(~out_df[parse_col]) | out_df[label_col].isna()].tolist()
    model_success = 0
    t0 = time.time()

    if remaining:
        tokenizer, model = load_llm(cfg["model_id"], use_4bit=use_4bit)
        for attempt in range(1, max_repair_attempts + 1):
            remaining = out_df.index[(~out_df[parse_col]) | out_df[label_col].isna()].tolist()
            if not remaining:
                break
            num_batches = int(np.ceil(len(remaining) / batch_size))
            for batch_no, start in enumerate(tqdm(range(0, len(remaining), batch_size), total=num_batches, desc=f"{cfg['tag']}-repair-{attempt}"), start=1):
                positions = remaining[start:start + batch_size]
                batch = out_df.iloc[positions]
                prompts = [
                    build_model_input(tokenizer, make_prompt_v2_balanced(r["title"], r["abstract"], repair_attempt=attempt))
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
                    if pred is None:
                        pred = robust_parse_llm_text(txt)
                    if pred is not None:
                        was_failed = (not bool(out_df.loc[pos, parse_col])) or pd.isna(out_df.loc[pos, label_col])
                        out_df.loc[pos, label_col] = int(bool(pred))
                        out_df.loc[pos, parse_col] = True
                        if was_failed:
                            model_success += 1
                    else:
                        out_df.loc[pos, label_col] = np.nan
                        out_df.loc[pos, parse_col] = False

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if batch_no % checkpoint_every_batches == 0:
                    out_df.to_csv(cfg["output_path"], index=False)
                    out_df.to_csv(cfg["checkpoint_path"], index=False)
        unload_model(model, tokenizer)

    runtime_s = time.time() - t0
    out_df.to_csv(cfg["output_path"], index=False)
    out_df.to_csv(cfg["checkpoint_path"], index=False)

    after_ok = int(out_df[parse_col].sum())
    stats = pd.DataFrame([
        {"metric": "model_id", "value": cfg["model_id"]},
        {"metric": "run_key", "value": run_key},
        {"metric": "rows_total", "value": len(out_df)},
        {"metric": "parse_ok_before", "value": before_ok},
        {"metric": "initial_failed_or_missing", "value": len(initial_failed_idx)},
        {"metric": "reparsed_from_existing_raw", "value": recovered_from_raw},
        {"metric": "model_repair_success", "value": model_success},
        {"metric": "parse_ok_after", "value": after_ok},
        {"metric": "label_missing_after", "value": int(out_df[label_col].isna().sum())},
        {"metric": "runtime_s", "value": round(runtime_s, 4)},
        {"metric": "output_path", "value": cfg["output_path"]},
        {"metric": "backup_path", "value": backup_path},
    ])
    repair_stats_path = cfg["output_path"].replace(".csv", "_parse_repair_stats.csv")
    stats.to_csv(repair_stats_path, index=False)
    return stats
