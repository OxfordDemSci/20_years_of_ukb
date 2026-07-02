import ast
import gc
import json
import os
import re
import warnings

import numpy as np
import pandas as pd
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def login_hf(hf_token=None):
    token = hf_token or os.getenv("HF_TOKEN", "").strip()
    if token:
        os.environ["HF_TOKEN"] = token
        login(token=token)
        return True
    return False


def extract_text(x):
    if pd.isna(x):
        return ""
    if not isinstance(x, str):
        return str(x).strip()
    s = x.strip()
    if s.startswith("{") and s.endswith("}"):
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, dict):
                if obj.get("preferred") is not None:
                    return str(obj["preferred"]).strip()
                for v in obj.values():
                    if v is not None and str(v).strip():
                        return str(v).strip()
        except Exception:
            pass
    return s


def load_input_csv(input_path):
    df = pd.read_csv(input_path)
    required = ["id", "title", "abstract", "year"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {df.columns.tolist()}")
    df = df[required].copy()
    df["id"] = df["id"].astype(str).str.strip()
    df["title"] = df["title"].apply(extract_text)
    df["abstract"] = df["abstract"].apply(extract_text)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[df["id"].ne("")]
    df = df[df["abstract"].fillna("").astype(str).str.strip().ne("")]
    return df.drop_duplicates(subset=["id"], keep="first").reset_index(drop=True)


def truncate_text(s, max_chars=3500):
    return str(s or "")[:max_chars]


def json_instruction():
    return """Return strict JSON only:
{"implies_UKB_use":true|false}
Do not include explanation, markdown, or any extra keys."""


def make_prompt_v2_balanced(title, abstract, repair_attempt=None):
    repair_note = ""
    if repair_attempt is not None:
        repair_note = f"""
This is an extra repair-generation round because the previous response could not be parsed.
Return exactly one valid JSON object only:
{{"implies_UKB_use":true}} or {{"implies_UKB_use":false}}
No explanation, no markdown, no extra keys, no text before or after the JSON.
Repair attempt: {repair_attempt}.
"""
    return f"""You will be given a scientific paper title and abstract.

Task: decide whether the paper likely used UK Biobank data or resources.

Important:
- Some true UK Biobank-use papers do not mention "UK Biobank" in the abstract.
- The paper may still use UK Biobank if the abstract describes a UK population-scale cohort, genetic/imaging/health-record analysis, or data-resource use that is consistent with UK Biobank.
- Do not require explicit words "UK Biobank" if the evidence strongly suggests use.

Return true when the title/abstract provides reasonable evidence that the paper analysed UK Biobank participants, data, samples, imaging, genetics, linked records, or a UK Biobank-derived cohort.

Return false when the paper is only about generic biobanking, ethics/governance, reviews, comparisons, or another named biobank.

{json_instruction()}
{repair_note}
title: \"\"\"{truncate_text(title, 700)}\"\"\"

abstract: \"\"\"{truncate_text(abstract, 3200)}\"\"\"

JSON:
"""


def extract_first_json_obj(text):
    raw = (text or "").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I | re.S).strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    m = re.search(r"\{.*?\}", raw, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return None


def parse_llm_result(obj):
    if not isinstance(obj, dict):
        return None
    val = obj.get("implies_UKB_use", None)
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        v = val.strip().lower()
        if v in {"true", "yes", "1"}:
            return True
        if v in {"false", "no", "0"}:
            return False
    return None


def parse_bool_value(val):
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, np.integer)) and val in {0, 1}:
        return bool(val)
    if isinstance(val, float) and val in {0.0, 1.0}:
        return bool(int(val))
    if isinstance(val, str):
        v = val.strip().lower().strip("\"'`.,;: ")
        if v in {"true", "yes", "y", "1", "positive", "likely", "likely true"}:
            return True
        if v in {"false", "no", "n", "0", "negative", "unlikely", "likely false"}:
            return False
    return None


def robust_parse_llm_text(text):
    raw = str(text or "").strip()
    raw = re.sub(r"^\s*```(?:json|JSON)?\s*", "", raw, flags=re.I)
    raw = re.sub(r"\s*```\s*$", "", raw, flags=re.I).strip()
    candidates = [raw] + [m.group(0) for m in re.finditer(r"\{[^{}]*\}", raw, flags=re.S)]
    seen = set()
    for cand in candidates:
        cand = cand.strip()
        if not cand or cand in seen:
            continue
        seen.add(cand)
        cand_norm = re.sub(r"\bTrue\b", "true", cand)
        cand_norm = re.sub(r"\bFalse\b", "false", cand_norm)
        cand_norm = re.sub(r"\bNone\b", "null", cand_norm)
        if "'" in cand_norm and '"' not in cand_norm:
            cand_norm = cand_norm.replace("'", '"')
        cand_norm = re.sub(r",\s*([}\]])", r"\1", cand_norm)
        obj = None
        try:
            obj = json.loads(cand_norm)
        except Exception:
            pass
        if obj is None:
            try:
                obj = ast.literal_eval(cand)
            except Exception:
                pass
        if isinstance(obj, dict):
            for key in [
                "implies_UKB_use", "implies_ukb_use", "ukb_use",
                "uses_uk_biobank", "uses_UKB", "used_UKB",
                "label", "prediction", "answer",
            ]:
                if key in obj:
                    parsed = parse_bool_value(obj.get(key))
                    if parsed is not None:
                        return parsed
    compact = re.sub(r"[^a-z0-9_ -]", " ", raw.strip().lower())
    compact = re.sub(r"\s+", " ", compact).strip()
    if compact in {"true", "yes", "1"}:
        return True
    if compact in {"false", "no", "0"}:
        return False
    m = re.search(r"implies[_\s-]*UKB[_\s-]*use\s*[:=]\s*(true|false|yes|no|1|0)", raw, flags=re.I)
    if m:
        return parse_bool_value(m.group(1))
    return None


def build_model_input(tokenizer, user_text):
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": user_text}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return user_text
    return user_text


def load_llm(model_id, use_4bit=True, dtype=torch.float16):
    token = os.getenv("HF_TOKEN", None)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
        token=token,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant_config = None
    if use_4bit and torch.cuda.is_available():
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    kwargs = dict(device_map="auto", torch_dtype=dtype, token=token, trust_remote_code=True)
    if quant_config is not None:
        kwargs["quantization_config"] = quant_config
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()
    return tokenizer, model


def unload_model(model=None, tokenizer=None):
    try:
        del model
    except Exception:
        pass
    try:
        del tokenizer
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def normalise_bool_col(s):
    if s.dtype == bool:
        return s.fillna(False).astype(bool)
    txt = s.astype(str).str.strip().str.lower()
    mapped = txt.map({
        "true": True, "1": True, "1.0": True, "yes": True, "y": True,
        "false": False, "0": False, "0.0": False, "no": False, "n": False,
        "nan": False, "none": False, "<na>": False, "": False,
    })
    return mapped.fillna(False).astype(bool)


def normalise_label_col(s):
    out = pd.Series(pd.NA, index=s.index, dtype="Int64")
    num = pd.to_numeric(s, errors="coerce")
    out.loc[num == 1] = 1
    out.loc[num == 0] = 0
    txt = s.astype(str).str.strip().str.lower()
    out.loc[txt.isin(["true", "yes", "y", "1", "1.0"])] = 1
    out.loc[txt.isin(["false", "no", "n", "0", "0.0"])] = 0
    return out


def clean_year(y):
    y = pd.to_numeric(y, errors="coerce")
    if pd.isna(y):
        return np.nan
    y = int(y)
    return y if 1900 <= y <= 2035 else np.nan


def find_first_existing(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None


def make_text_for_analysis(df):
    return df["title"].fillna("").astype(str).str.strip() + "\n" + df["abstract"].fillna("").astype(str).str.strip()


def savefig(fig_dir, name, dpi=220):
    import matplotlib.pyplot as plt
    os.makedirs(fig_dir, exist_ok=True)
    path = os.path.join(fig_dir, name)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return path


def ensure_real_path(value, arg_name):
    if not value or value.startswith("INPUT your path"):
        raise ValueError(f"Please set {arg_name}.")
    return value
