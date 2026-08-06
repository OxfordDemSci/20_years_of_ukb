import argparse
import os

import pandas as pd

from common import ensure_real_path, login_hf
from consensus import combine_three_model_outputs
from model_llama3 import MODEL_CONFIG as LLAMA_CONFIG
from model_mistral import MODEL_CONFIG as MISTRAL_CONFIG
from model_qwen import MODEL_CONFIG as QWEN_CONFIG
from repair_failed_parses import repair_failed_parses_for_model
from tagger import run_tagging_for_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="INPUT your path for input CSV here")
    parser.add_argument("--output_dir", default="INPUT your path for model outputs folder here")
    parser.add_argument("--analysis_dir", default="INPUT your path for consensus output folder here")
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_input_tokens", type=int, default=4096)
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--skip_tagging", action="store_true")
    parser.add_argument("--skip_repair", action="store_true")
    parser.add_argument("--use_4bit", action="store_true", default=True)
    args = parser.parse_args()

    input_path = ensure_real_path(args.input_path, "--input_path")
    output_dir = ensure_real_path(args.output_dir, "--output_dir")
    analysis_dir = ensure_real_path(args.analysis_dir, "--analysis_dir")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)
    login_hf(args.hf_token)

    configs = [QWEN_CONFIG, LLAMA_CONFIG, MISTRAL_CONFIG]
    if not args.skip_tagging:
        for cfg in configs:
            run_tagging_for_model(
                input_path=input_path,
                output_dir=output_dir,
                config=cfg,
                batch_size=args.batch_size,
                max_input_tokens=args.max_input_tokens,
                max_new_tokens=args.max_new_tokens,
                use_4bit=args.use_4bit,
            )

    repair_logs = []
    if not args.skip_repair:
        for cfg in configs:
            stats = repair_failed_parses_for_model(
                output_dir=output_dir,
                config=cfg,
                batch_size=args.batch_size,
                max_input_tokens=args.max_input_tokens,
                max_new_tokens=args.max_new_tokens,
                use_4bit=args.use_4bit,
            )
            stats["model_run_key"] = cfg["run_key"]
            repair_logs.append(stats)
        pd.concat(repair_logs, ignore_index=True).to_csv(os.path.join(output_dir, "parse_repair_round_log.csv"), index=False)

    paths = combine_three_model_outputs(output_dir, QWEN_CONFIG, LLAMA_CONFIG, MISTRAL_CONFIG, analysis_dir)
    print(paths)


if __name__ == "__main__":
    main()
