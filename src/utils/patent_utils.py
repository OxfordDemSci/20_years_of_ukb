"""
Patent Analysis Utility Functions
==================================

This module provides comprehensive utilities for patent analysis including:
- LLM-based classification for proteomics and drug development
- Data cleaning and preprocessing
- Country assignment and inference
- Topic parsing and categorization
- Visualization and plotting functions

Author: Jiani Y
Date: 2026-02-04
"""

import ast
import re
import json
from collections import Counter, defaultdict
from typing import Any, Optional, List, Dict, Tuple, Callable
import difflib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from bs4 import BeautifulSoup
from collections import Counter
import pandas as pd



import ast
import re
from collections import Counter
import pandas as pd
from typing import Any, Optional
import difflib


import ast
import pandas as pd
from collections import Counter



try:
    import pycountry
    _HAS_PYCOUNTRY = True
except Exception:
    _HAS_PYCOUNTRY = False

try:
    import geopandas as gpd
    _HAS_GEOPANDAS = True
except Exception:
    _HAS_GEOPANDAS = False


# ==============================================================================
# LLM PROMPT TEMPLATES
# ==============================================================================

PROTEOMICS_PROMPT_TEMPLATE = """
You are an expert in biomedical research and patent analysis.

Task:
Determine whether the following patent is related to PROTEOMICS.

Definition:
A patent is proteomics-related if it primarily concerns large-scale analysis,
identification, quantification, modification, or functional characterization
of proteins. This includes, for example:
- mass spectrometry–based proteomics
- protein biomarkers discovered at scale
- proteome-wide association studies
- protein interaction networks
- post-translational modification profiling

Exclude:
- patents focused only on a single protein as a drug target
- general genomics or transcriptomics without protein-level analysis
- purely therapeutic compositions without proteomic methodology

Output strictly in valid JSON with the following keys:
{{"label": "yes" or "no", "confidence": "high" | "medium" | "low", "rationale": "1-2 sentence explanation"}}

Patent abstract:
<<<
{abstract}
>>>
"""

DRUG_DEV_PROMPT_TEMPLATE = """
You are an expert in biomedical innovation, pharmacology, and patent analysis.

Task:
Determine whether the following patent is related to MEDICINE OR DRUG DEVELOPMENT.

Definition:
A patent is considered medicine/drug-development-related if its primary aim
concerns the discovery, development, optimization, validation, or therapeutic
use of medicines. This includes (but is not limited to):

- Identification or validation of therapeutic targets
- Small-molecule drugs, biologics, gene or cell therapies
- Drug repurposing
- Biomarkers explicitly used for treatment selection or stratification
- Therapeutic compositions, formulations, dosing, or delivery systems
- Methods of treatment or prevention of disease using a therapeutic agent

Exclude:
- Purely diagnostic inventions without therapeutic application
- Basic biological research without translational or therapeutic intent
- Medical devices not directly linked to drug delivery
- General research tools or assays without a drug development focus

Output strictly in valid JSON using the following schema:
{{"label": "yes" or "no",
  "development_stage": "target discovery" | "lead discovery" | "preclinical" | "clinical/therapeutic use" | "unclear",
  "confidence": "high" | "medium" | "low",
  "rationale": "1–2 sentences explaining the decision"}}

Patent abstract:
<<<
{abstract}
>>>
"""


# ==============================================================================
# DATA CLEANING FUNCTIONS
# ==============================================================================

def clean_patent_abstract(text: str) -> str:
    """
    Clean patent abstract text by removing HTML tags, normalizing spacing,
    and handling chemical notation.
    
    Args:
        text: Raw patent abstract text
        
    Returns:
        Cleaned text string
    """
    if pd.isna(text):
        return text

    s = str(text)

    # Repair common malformed tag patterns
    s = re.sub(r'<sub>(\d+)\)</sub>', r'<sub>\1</sub>)', s)

    # Convert <sub> and <sup> to plain-text markers before stripping tags
    s = re.sub(r'<sup>\s*([^<]+?)\s*</sup>', r'^\1', s, flags=re.IGNORECASE)
    s = re.sub(r'<sub>\s*([^<]+?)\s*</sub>', r'_\1', s, flags=re.IGNORECASE)

    # Strip remaining HTML/XML tags safely
    s = BeautifulSoup(s, "html.parser").get_text(separator=" ")

    # Normalize punctuation/spacing
    s = s.replace("\u00a0", " ")
    s = re.sub(r'\s+', ' ', s).strip()
    s = re.sub(r'\s+([,.;:])', r'\1', s)
    s = re.sub(r'\(\s+', '(', s)
    s = re.sub(r'\s+\)', ')', s)
    s = re.sub(r'\s*-\s*', '-', s)

    # Standardize chemical notation
    s = re.sub(r'\b([cC])_(\d+)\b', r'\1\2', s)

    return s


# ==============================================================================
# LLM CLASSIFICATION FUNCTIONS
# ==============================================================================

def classify_proteomics_ollama(
    abstract: str,
    client,
    model: str = "llama3.1:8b",
    temperature: float = 0.0
) -> Dict[str, Any]:
    """
    Classify a patent abstract as proteomics-related using Ollama LLM.
    
    Args:
        abstract: Patent abstract text
        client: Ollama client instance
        model: Model name to use
        temperature: Sampling temperature
        
    Returns:
        Dictionary with keys: label, confidence, rationale
    """
    prompt = PROTEOMICS_PROMPT_TEMPLATE.format(abstract=abstract)

    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature}
    )

    content = response["message"]["content"]

    try:
        json_str = content[content.find("{"):content.rfind("}")+1]
        parsed = json.loads(json_str)
    except Exception:
        parsed = {
            "label": None,
            "confidence": None,
            "rationale": content.strip()
        }

    return parsed


def classify_drug_dev_ollama(
    abstract: str,
    client,
    model: str = "llama3.1:8b",
    temperature: float = 0.0
) -> Dict[str, Any]:
    """
    Classify a patent abstract as drug development-related using Ollama LLM.
    
    Args:
        abstract: Patent abstract text
        client: Ollama client instance
        model: Model name to use
        temperature: Sampling temperature
        
    Returns:
        Dictionary with keys: label, development_stage, confidence, rationale
    """
    prompt = DRUG_DEV_PROMPT_TEMPLATE.format(abstract=abstract)

    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature}
    )

    content = response["message"]["content"]

    try:
        json_str = content[content.find("{"):content.rfind("}")+1]
        parsed = json.loads(json_str)
    except Exception:
        parsed = {
            "label": None,
            "development_stage": None,
            "confidence": None,
            "rationale": content.strip()
        }

    return parsed


def batch_classify_patents(
    df: pd.DataFrame,
    classify_func: Callable,
    client,
    abstract_col: str = "abstract_clean",
    result_cols: List[str] = None,
    checkpoint_path: str = None,
    save_every: int = 50,
    model: str = "llama3.1:8b",
    temperature: float = 0.0
) -> pd.DataFrame:
    """
    Batch classify patents with checkpoint saving.
    
    Args:
        df: DataFrame with patent data
        classify_func: Classification function (classify_proteomics_ollama or classify_drug_dev_ollama)
        client: Ollama client instance
        abstract_col: Column name containing abstracts
        result_cols: List of column names for results
        checkpoint_path: Path to save checkpoints
        save_every: Save checkpoint every N records
        model: Model name to use
        temperature: Sampling temperature
        
    Returns:
        DataFrame with classification results
    """
    from tqdm import tqdm
    
    if result_cols is None:
        result_cols = ["label", "confidence", "rationale"]
    
    # Load checkpoint if exists
    if checkpoint_path and pd.io.common.file_exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        df = pd.read_csv(checkpoint_path)
    else:
        df = df.copy()
    
    # Ensure result columns exist
    for col in result_cols:
        if col not in df.columns:
            df[col] = None
    
    for i in tqdm(range(len(df)), total=len(df)):
        # Skip if already processed
        if pd.notna(df.at[i, result_cols[0]]):
            continue
        
        abstract = df.at[i, abstract_col]
        
        if not isinstance(abstract, str) or len(abstract.strip()) == 0:
            for col in result_cols:
                df.at[i, col] = None
            continue
        
        res = classify_func(abstract, client=client, model=model, temperature=temperature)
        
        for col in result_cols:
            df.at[i, col] = res.get(col)
        
        # Save checkpoint
        if checkpoint_path and (i + 1) % save_every == 0:
            df.to_csv(checkpoint_path, index=False)
    
    # Final save
    if checkpoint_path:
        df.to_csv(checkpoint_path, index=False)
        print(f"Done. Results saved to {checkpoint_path}")
    
    return df


# ==============================================================================
# COUNTRY ASSIGNMENT FUNCTIONS
# ==============================================================================

# Fallback country name to ISO-2 mapping
_FALLBACK_NAME_TO_ISO = {
    'UNITED KINGDOM': 'GB',
    'UNITED STATES': 'US',
    'UNITED STATES OF AMERICA': 'US',
    'USA': 'US',
    'SOUTH KOREA': 'KR',
    'REPUBLIC OF KOREA': 'KR',
    'KOREA': 'KR',
    'JAPAN': 'JP',
    'GERMANY': 'DE',
    'FRANCE': 'FR',
    'CHINA': 'CN',
    "PEOPLE'S REPUBLIC OF CHINA": 'CN',
    'AUSTRALIA': 'AU',
    'CANADA': 'CA',
    'ITALY': 'IT',
    'SPAIN': 'ES',
}


def to_iso2(country_candidate: Optional[str]) -> Optional[str]:
    """
    Convert a country name or code to ISO-2 code (uppercase).
    
    Args:
        country_candidate: Country name or code string
        
    Returns:
        ISO-2 country code or None
    """
    if country_candidate is None:
        return None
    if not isinstance(country_candidate, str):
        return None
    s = country_candidate.strip()
    if not s:
        return None

    # Already 2-letter code
    if len(s) == 2 and s.isalpha():
        return s.upper()

    # 3-letter code -> map via pycountry or small map
    if len(s) == 3 and s.isalpha():
        if _HAS_PYCOUNTRY:
            try:
                c = pycountry.countries.get(alpha_3=s.upper())
                if c:
                    return c.alpha_2
            except Exception:
                pass
        _three_to_two = {'GBR': 'GB', 'USA': 'US', 'KOR': 'KR', 'JPN': 'JP', 'CHN': 'CN'}
        if s.upper() in _three_to_two:
            return _three_to_two[s.upper()]

    # Try pycountry name search
    if _HAS_PYCOUNTRY:
        try:
            candidates = pycountry.countries.search_fuzzy(s)
            if candidates:
                return candidates[0].alpha_2
        except Exception:
            pass

    # Exact fallback map
    key = s.upper()
    if key in _FALLBACK_NAME_TO_ISO:
        return _FALLBACK_NAME_TO_ISO[key]

    # Token match
    tokens = re.split(r'[,;/\-\(\)\s]+', key)
    for t in tokens:
        if t in _FALLBACK_NAME_TO_ISO:
            return _FALLBACK_NAME_TO_ISO[t]

    # Fuzzy fallback
    match = difflib.get_close_matches(key, list(_FALLBACK_NAME_TO_ISO.keys()), n=1, cutoff=0.8)
    if match:
        return _FALLBACK_NAME_TO_ISO[match[0]]

    return None


def _safe_literal_eval(val: Any) -> Any:
    """Safely parse string representations of lists/dicts."""
    if val is None:
        return None
    if isinstance(val, str):
        s = val.strip()
        if (s.startswith('[') and s.endswith(']')) or (s.startswith('{') and s.endswith('}')):
            try:
                return ast.literal_eval(s)
            except Exception:
                return val
    return val


def infer_country_from_name(name: Optional[str], use_low_confidence: bool = False) -> Optional[str]:
    """
    Rule-based inference of country from assignee name.
    
    Args:
        name: Assignee name string
        use_low_confidence: Whether to use low-confidence rules (e.g., 'Inc' -> US)
        
    Returns:
        ISO-2 country code or None
    """
    if not isinstance(name, str) or not name.strip():
        return None
    s = name.strip()
    s_lower = s.lower()

    # High-confidence markers
    high_confidence_rules = [
        (r'주식회사', 'KR'),
        (r'산학협력단', 'KR'),
        (r'재단법인', 'KR'),
        (r'대학교', 'KR'),
        (r'서울대학교', 'KR'),
        (r'연세대학교', 'KR'),
        (r'이화여자대학교', 'KR'),
        (r'株式会社', 'JP'),
        (r'有限会社', 'JP'),
        (r'\bplc\b', 'GB'),
        (r'\bpty\s?ltd\b', 'AU'),
        (r'\bgmbh\b', 'DE'),
    ]
    for pattern, iso in high_confidence_rules:
        if re.search(pattern, s_lower, flags=re.IGNORECASE):
            return iso

    # Low-confidence patterns
    if use_low_confidence:
        low_confidence_rules = [
            (r'\binc\.?\b', 'US'),
            (r'\bllc\b', 'US'),
            (r'\bltd\b', 'GB'),
            (r'\bsa\b', 'FR'),
        ]
        for pattern, iso in low_confidence_rules:
            if re.search(pattern, s_lower, flags=re.IGNORECASE):
                return iso

    # Detect trailing country in parentheses
    m = re.search(r'[\(\[,] *([A-Za-z \'\.]+?) *[\)\]]?$', s)
    if m:
        maybe_country = m.group(1)
        iso = to_iso2(maybe_country)
        if iso:
            return iso

    # Direct convert if string itself is country name
    iso = to_iso2(s)
    if iso:
        return iso

    return None


def assign_assignee_countries_with_iso(
    df: pd.DataFrame,
    assignee_col: str = 'assignees',
    assignee_names_col: str = 'assignee_names',
    use_low_confidence: bool = False,
    primary_pick: str = 'first'
) -> pd.DataFrame:
    """
    Assign country information to patents based on assignee data.
    
    Args:
        df: DataFrame with patent data
        assignee_col: Column with assignee structured data
        assignee_names_col: Column with assignee names
        use_low_confidence: Whether to use low-confidence inference rules
        primary_pick: How to pick primary country ('first' or 'most_common')
        
    Returns:
        DataFrame with added columns:
        - assignee_countries: list of ISO-2 codes
        - assignee_country_primary: primary ISO-2 code
        - assignee_country_source: source of inference
    """
    out_countries = []
    out_primary = []
    out_source = []

    for idx, row in df.iterrows():
        raw_assignees = row.get(assignee_col, None)
        assignee_names_val = row.get(assignee_names_col, None)

        parsed = _safe_literal_eval(raw_assignees)
        countries = []
        sources = []
        found_authority = False

        # 1) Authoritative country info from structured data
        if isinstance(parsed, list) and parsed:
            any_with_country = any(isinstance(x, dict) and (x.get('country_code') or x.get('country') or x.get('country_name')) for x in parsed)
            if any_with_country:
                for item in parsed:
                    if isinstance(item, dict):
                        cc = item.get('country_code') or item.get('country') or item.get('country_name')
                        if cc:
                            iso = to_iso2(cc)
                            if iso is None and isinstance(cc, str) and len(cc.strip()) == 2:
                                iso = cc.strip().upper()
                            countries.append(iso)
                            sources.append('authority' if iso else 'authority_raw')
                        else:
                            countries.append(None)
                            sources.append('unknown')
                found_authority = True

        # 2) Rule-based inference from assignee_names
        if not found_authority:
            parsed_names = _safe_literal_eval(assignee_names_val)
            used_names = []

            if isinstance(parsed_names, list):
                used_names = [n for n in parsed_names if isinstance(n, str) and n.strip()]
            elif isinstance(parsed_names, str) and parsed_names.strip():
                s = parsed_names.strip()
                if s.startswith('[') and s.endswith(']'):
                    try:
                        il = ast.literal_eval(s)
                        used_names = [n for n in il if isinstance(n, str) and n.strip()]
                    except Exception:
                        used_names = [s]
                else:
                    used_names = [s]

            if used_names:
                for nm in used_names:
                    iso = infer_country_from_name(nm, use_low_confidence)
                    countries.append(iso)
                    sources.append('rule_low_conf' if use_low_confidence and iso else ('rule_high_conf' if iso else 'unknown'))
            else:
                # Fallback to names inside parsed assignees
                if isinstance(parsed, list) and parsed:
                    for item in parsed:
                        if isinstance(item, dict):
                            name = item.get('name') or item.get('applicant') or None
                            iso = infer_country_from_name(name, use_low_confidence)
                            countries.append(iso)
                            sources.append('rule_low_conf' if use_low_confidence and iso else ('rule_high_conf' if iso else 'unknown'))
                        elif isinstance(item, str):
                            iso = infer_country_from_name(item, use_low_confidence)
                            countries.append(iso)
                            sources.append('rule_low_conf' if use_low_confidence and iso else ('rule_high_conf' if iso else 'unknown'))
                        else:
                            countries.append(None)
                            sources.append('unknown')

        # Normalize to uppercase ISO strings
        norm_countries = [c.upper() if isinstance(c, str) else None for c in countries]

        # Choose primary country
        primary = None
        source_label = 'unknown'
        if 'authority' in sources:
            for c, s in zip(norm_countries, sources):
                if s == 'authority' and c:
                    primary = c
                    source_label = 'authority'
                    break

        if primary is None:
            non_none = [c for c in norm_countries if c]
            if non_none:
                if primary_pick == 'first':
                    primary = non_none[0]
                elif primary_pick == 'most_common':
                    primary = Counter(non_none).most_common(1)[0][0]
                for c, s in zip(norm_countries, sources):
                    if c == primary:
                        source_label = s
                        break
            else:
                primary = None
                source_label = 'unknown'

        out_countries.append(norm_countries)
        out_primary.append(primary)
        out_source.append(source_label)

    df = df.copy()
    df['assignee_countries'] = out_countries
    df['assignee_country_primary'] = out_primary
    df['assignee_country_source'] = out_source
    return df


def iso2_to_iso3(iso2: str) -> Optional[str]:
    """
    Convert ISO-2 country code to ISO-3.
    
    Args:
        iso2: ISO-2 country code
        
    Returns:
        ISO-3 country code or None
    """
    if not _HAS_PYCOUNTRY:
        return None
    try:
        return pycountry.countries.get(alpha_2=iso2).alpha_3
    except Exception:
        return None


# ==============================================================================
# TOPIC PARSING FUNCTIONS
# ==============================================================================

def safe_parse_category(val: Any) -> List[Dict]:
    """
    Parse category_for field safely into list of dicts.
    
    Args:
        val: Raw category value
        
    Returns:
        List of category dictionaries
    """
    if val is None:
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return []
        try:
            parsed = ast.literal_eval(s)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            try:
                parsed = json.loads(s)
                return parsed if isinstance(parsed, list) else []
            except Exception:
                return []
    return []


def extract_topic_names(category_list: List[Dict]) -> List[str]:
    """
    Extract topic names from parsed category list.
    
    Args:
        category_list: List of category dictionaries
        
    Returns:
        List of topic names
    """
    return [
        d.get('name') for d in category_list
        if isinstance(d, dict) and d.get('name')
    ]


def parse_patent_topics(
    df: pd.DataFrame,
    category_col: str = 'category_for'
) -> pd.DataFrame:
    """
    Parse patent topics and add topics_list and n_topics columns.
    
    Args:
        df: DataFrame with patent data
        category_col: Column name containing category data
        
    Returns:
        DataFrame with added columns: topics_list, n_topics
    """
    df = df.copy()
    df['topics_list'] = df[category_col].apply(
        lambda x: extract_topic_names(safe_parse_category(x))
    )
    df['n_topics'] = df['topics_list'].apply(len)
    return df


def _extract_name_and_leading_code(topic_dict: Dict) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract leading code and display name from topic dict.
    
    Args:
        topic_dict: Dictionary with 'name' field
        
    Returns:
        Tuple of (leading_code, display_name)
    """
    if not isinstance(topic_dict, dict):
        return None, None
    name = topic_dict.get('name') or ''
    if not isinstance(name, str) or not name.strip():
        return None, None
    name = name.strip()
    
    m = re.match(r'^\s*(\d+)\s*(.*)$', name)
    if m:
        leading_code = m.group(1)
        rest = m.group(2).strip() or None
        return leading_code, rest
    else:
        return None, name


def _top_level_code_from_leading(leading_code: Optional[str]) -> Optional[str]:
    """
    Convert leading code to top-level code (first 2 digits).
    
    Args:
        leading_code: Full leading code string
        
    Returns:
        Top-level code (first 2 digits) or None
    """
    if not leading_code:
        return None
    s = str(leading_code)
    if len(s) >= 2:
        return s[:2]
    return s


def collapse_to_top_level(cat_parsed: List[Dict]) -> List[Tuple[Optional[str], Optional[str]]]:
    """
    Collapse category list to top-level codes and labels.
    
    Args:
        cat_parsed: List of parsed category dictionaries
        
    Returns:
        List of (top_code, label) tuples
    """
    results = []
    for d in cat_parsed:
        leading_code, name_rest = _extract_name_and_leading_code(d)
        top_code = _top_level_code_from_leading(leading_code)
        
        if top_code is None:
            label = d.get('name') or None
        else:
            if name_rest:
                label = name_rest
            else:
                full = d.get('name', '')
                m2 = re.match(r'^\s*\d+\s*(.*)$', full)
                label = m2.group(1).strip() if m2 and m2.group(1) else full
        results.append((top_code, label))
    return results


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def find_ukbb_papers(ids: Any, df_all_ukbb: Any) -> List[str]:
    """
    Find UKBB papers from publication IDs.
    
    Args:
        ids: Publication IDs (string or set)
        df_all_ukbb: DataFrame with all UKBB papers or an iterable of UKBB ids
        
    Returns:
        List of UKBB paper IDs
    """
    if isinstance(ids, str):
        ids_set = set(ast.literal_eval(ids))
    elif isinstance(ids, (list, tuple, set)):
        ids_set = set(ids)
    else:
        ids_set = set()

    if isinstance(df_all_ukbb, pd.DataFrame):
        ukbb_ids = set(df_all_ukbb['id'].tolist())
    else:
        ukbb_ids = set(df_all_ukbb) if df_all_ukbb is not None else set()

    ukbb_papers = [id for id in ids_set if id in ukbb_ids]
    return ukbb_papers


def plot_bar_matplotlib(
    counts_df: pd.DataFrame,
    top_n: int = 20,
    colors: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    savefile: Optional[str] = None
):
    """
    Plot bar chart of country counts.
    
    Args:
        counts_df: DataFrame with 'iso2' and 'count' columns
        top_n: Number of top countries to show
        colors: Bar color (default: '#345995')
        figsize: Figure size tuple
        savefile: Path to save figure
    """
    if colors is None:
        colors = '#345995'
    
    df = counts_df.sort_values('count', ascending=False).head(top_n).copy()
    plt.figure(figsize=figsize)
    plt.bar(df['iso2'], df['count'], color=colors)
    
    for i, row in df.iterrows():
        plt.text(row['iso2'], row['count'], str(row['count']), 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.xlabel('Country')
    plt.ylabel('Count')
    plt.title(f'Top {min(top_n, len(df))} countries by assignee occurrences')
    plt.tight_layout()
    
    if savefile:
        plt.savefig(savefile, dpi=200)
        print(f"Bar chart saved to: {savefile}")
    plt.show()


def build_country_count(df_with_iso):
    # build country -> count dict (ISO-2) and add iso3 code 
    
    counter = Counter()

    for countries in df_with_iso['assignee_countries']:
        if isinstance(countries, list):
            for c in countries:
                if isinstance(c, str):
                    counter[c.upper()] += 1

    country_count = dict(counter)

    # convert to DataFrame
    country_df = (
        pd.DataFrame(country_count.items(), columns=['iso2', 'count'])
        .sort_values('count', ascending=False)
        .reset_index(drop=True)
    )


    # convert ISO-2 to ISO-3
    def iso2_to_iso3(iso2):
        try:
            return pycountry.countries.get(alpha_2=iso2).alpha_3
        except Exception:
            return None

    country_df['iso3'] = country_df['iso2'].apply(iso2_to_iso3)
    country_df = country_df.dropna(subset=['iso3'])
    return country_df



# =========================
# Topic general functions 
# =========================




# ==============================================================================
# PLOTTING FUNCTIONS
# ==============================================================================





def plot_topics_distribution(df_patent, cat_col,figsize=(10,6),savefigure=False):
    """
    Plots the distribution of the number of topics per patent for a given category column.
    """
    avg_topics = df_patent['n_topics'].mean()
    median_topics = df_patent['n_topics'].median()

    print(f"Average number of topics per patent: {avg_topics:.2f}")
    print(f"Median number of topics per patent: {median_topics:.0f}")



    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(
        df_patent['n_topics'],
        bins=range(1, df_patent['n_topics'].max() + 2),
        edgecolor='black'
    )
    ax.axvline(avg_topics, linestyle='--', label=f'Mean = {avg_topics:.2f}')
    ax.axvline(median_topics, linestyle=':', label=f'Median = {median_topics:.0f}')

    ax.set_xlabel('Number of topics per patent')
    ax.set_ylabel('Number of patents')
    ax.set_title(f'Distribution of topics per patent ({cat_col.replace("category_", "")})')
    ax.legend()
    ax.tight_layout()
    if savefigure:
        plt.savefig(f'fig/patent/topics_distribution_{cat_col}.pdf')
        print(f"Topics distribution plot saved to: fig/patent/topics_distribution_{cat_col}.pdf")
    else:
        return fig, ax



def map_plotting(country_df, column_to_show_counts,figsize=(12, 8),savefigure=True):
    """
    function to plot the world map with patent counts.

    Args:
        country_df (pd.DataFrame): DataFrame with columns 'iso3' and 'count' for each country.
        column_to_show_counts (str): The column name to show counts for, e.g., 'assignees' or 'inventors'.
        figsize (tuple, optional): _description_. Defaults to (12, 8).
        savefigure (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """

    # merge world map with patent counts

    # load world shapefile 
    world = gpd.read_file(
        "data/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp"
    )
    world.columns = [c.lower() for c in world.columns]
    # merge patent counts into map
    world_patents = world.merge(
        country_df,
        how='left',
        left_on='iso_a3',
        right_on='iso3'
    )

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    world_patents.plot(
        column='count',
        ax=ax,
        legend=True,
        cmap='ocean_r',
        edgecolor='grey',
        missing_kwds={
            "color": "white",
            "label": "No patents"
        },
        legend_kwds={
            'label': "Number of {} occurrences".format(column_to_show_counts),
            'shrink': 0.6
        }
    )

    ax.set_title('Global distribution of patent {}'.format(column_to_show_counts), fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    if savefigure:
        plt.savefig('fig/patent/patent_countries_map_{}.pdf'.format(column_to_show_counts), dpi=500)    
    else:
        return fig, ax
    





def plot_filing_status_over_time(df_patent,col,figsize=(10, 6),savefigure=True, ax=None, title: Optional[str] = None):
    """
    Plot stacked bar chart of patent counts by filing status over publication years.

    Args:
        df_patent (pd.DataFrame): DataFrame containing patent data with 'publication_date' and filing status column
        col (str): Column name for filing status
        figsize (tuple, optional): Figure size tuple. Defaults to (10, 6).
        savefigure (bool, optional): Whether to save the figure. Defaults to True.
    """
    
    
    status_colors = {
        "Application Pending":"#4C72B0" ,        # teal
        "Application Ceased": "#8172B2",         # purple 
        "Active":"#55A868"  ,                     # blue 
        "Application Withdrawn": "#CCB974",      # mustard
        "Granted Patent Expired": "#DD8452",     # soft orange
        "Application Granted": "#64B5A7",        # green
        "Application Abandoned": "#C44E52"       # muted red
    }

    if col =='legal_status_replaced':
        replace_dict = {'Ceased':'Application Ceased', 
                        'Granted':'Application Granted', 
                        'Pending':'Application Pending', 
                        'Withdrawn':'Application Withdrawn',
                        'Abandoned':'Application Abandoned',
                        'Expired - Fee Related':'Granted Patent Expired'}
        df_patent['legal_status_replaced'] = df_patent['legal_status'].replace(replace_dict)

    # ensure datetime
    df_patent['publication_date'] = pd.to_datetime(
        df_patent['publication_date'], errors='coerce'
    )
    df_patent['publication_year'] = df_patent['publication_date'].dt.year



    # group
    filing_status_yearly = (
        df_patent
        .groupby(['publication_year', col])
        .size()
        .reset_index(name='counts')
    )

    # pivot for stacking
    pivot_df = filing_status_yearly.pivot(
        index='publication_year',
        columns=col,
        values='counts'
    ).fillna(0)

    pivot_df = pivot_df.sort_index()

    # totals per year
    totals = pivot_df.sum(axis=1)

    # plot
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    bottom = None
    for status in pivot_df.columns:
        values = pivot_df[status]
        bars = ax.bar(
            pivot_df.index,
            values,
            bottom=bottom,
            label=status,
            color=status_colors.get(status, None)
            #color=colors_scheme[pivot_df.columns.get_loc(status)+2]
        )

        # percentage annotations
        for year, bar, value in zip(pivot_df.index, bars, values):
            if value > 0 and totals.loc[year] > 2:
                pct = value / totals.loc[year] * 100
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + value / 2-0.5,
                    f'{pct:.1f}%',
                    ha='center',
                    va='center',
                    fontsize=8,
                    color ='white'
                )

        bottom = values if bottom is None else bottom + values

    # annotate total counts on top of bars
    for year in pivot_df.index:
        ax.text(
            year,
            totals.loc[year],
            f'{int(totals.loc[year])}',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )

    # labels & styling
    ax.set_xlabel('Publication Year')
    ax.set_ylabel('Number of Patents')
    if title is None:
        title = 'Patent Counts by Filing Status and Publication Year'
    ax.set_title(title)
    ax.legend(title='Filing Status',frameon=False, loc='upper left')
    ax.set_ylim(0, totals.max() * 1.1)  # add some headroom for annotations
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if created_fig:
        plt.tight_layout()
        if savefigure:
            plt.savefig('fig/patent/patent_filing_status_over_time.pdf', dpi=300)
        else:
            return fig, ax
    return fig, ax


def plot_patent_counts_by_filing_status(
    df: pd.DataFrame,
    date_col: str = 'publication_date',
    year_col: str = 'publication_year',
    filing_status_col: str = 'filing_status',
    colors: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (15, 6),
    savefile: Optional[str] = None
):
    """
    Plot stacked bar chart of patent counts by filing status over years.
    
    Args:
        df: DataFrame with patent data
        date_col: Column name for publication date
        year_col: Column name for publication year
        filing_status_col: Column name for filing status
        colors: List of colors for each filing status (default: ['#6E8B3D', '#345995'])
        figsize: Figure size tuple
        savefile: Path to save figure
    """
    if colors is None:
        colors = ['#6E8B3D', '#345995']
    
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df[year_col] = df[date_col].dt.year
    
    filing_status_yearly = (
        df.groupby([year_col, filing_status_col])
        .size()
        .reset_index(name='counts')
    )
    
    pivot_df = filing_status_yearly.pivot(
        index=year_col,
        columns=filing_status_col,
        values='counts'
    ).fillna(0).sort_index()
    
    totals = pivot_df.sum(axis=1)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bottom = None
    for idx, status in enumerate(pivot_df.columns):
        values = pivot_df[status]
        color = colors[idx % len(colors)]
        bars = ax.bar(
            pivot_df.index,
            values,
            bottom=bottom,
            label=status,
            color=color
        )
        
        # Percentage annotations
        for year, bar, value in zip(pivot_df.index, bars, values):
            if value > 0:
                pct = value / totals.loc[year] * 100
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + value / 2,
                    f'{pct:.1f}%',
                    ha='center',
                    va='center',
                    fontsize=8,
                    color='white'
                )
        
        bottom = values if bottom is None else bottom + values
    
    # Annotate total counts
    for year in pivot_df.index:
        ax.text(
            year,
            totals.loc[year],
            f'{int(totals.loc[year])}',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )
    
    ax.set_xlabel('Publication Year')
    ax.set_ylabel('Number of Patents')
    ax.set_title('Patent Counts by Filing Status and Publication Year')
    ax.legend(title='Filing Status')
    
    plt.tight_layout()
    if savefile:
        plt.savefig(savefile, dpi=300)
    plt.show()


def plot_patent_countries_map(
    df: pd.DataFrame,
    assignee_countries_col: str = 'assignee_countries',
    shapefile_path: str = None,
    colors: str = 'ocean_r',
    figsize: Tuple[int, int] = (15, 8),
    savefile: Optional[str] = None
):
    """
    Plot world map of patent assignee countries.
    
    Args:
        df: DataFrame with patent data including assignee_countries column
        assignee_countries_col: Column name with list of country ISO codes
        shapefile_path: Path to world shapefile
        colors: Colormap name (default: 'ocean_r')
        figsize: Figure size tuple
        savefile: Path to save figure
    """
    if not _HAS_GEOPANDAS:
        print("GeoPandas not available. Cannot plot map.")
        return
    
    if shapefile_path is None:
        print("Shapefile path required for map plotting.")
        return
    
    # Build country counter
    counter = Counter()
    for countries in df[assignee_countries_col]:
        if isinstance(countries, list):
            for c in countries:
                if isinstance(c, str):
                    counter[c.upper()] += 1
    
    country_count = dict(counter)
    country_df = (
        pd.DataFrame(country_count.items(), columns=['iso2', 'count'])
        .sort_values('count', ascending=False)
        .reset_index(drop=True)
    )
    
    country_df['iso3'] = country_df['iso2'].apply(iso2_to_iso3)
    country_df = country_df.dropna(subset=['iso3'])
    
    # Load world map
    world = gpd.read_file(shapefile_path)
    world.columns = [c.lower() for c in world.columns]
    
    world_patents = world.merge(
        country_df,
        how='left',
        left_on='iso_a3',
        right_on='iso3'
    )
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    world_patents.plot(
        column='count',
        ax=ax,
        legend=True,
        cmap=colors,
        edgecolor='black',
        missing_kwds={
            "color": "lightgrey",
            "label": "No patents"
        },
        legend_kwds={
            'label': "Number of assignee occurrences",
            'shrink': 0.6
        }
    )
    
    ax.set_title('Global distribution of patent assignees', fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    
    if savefile:
        plt.savefig(savefile, dpi=500)
    plt.show()


def plot_topics_histogram(
    df: pd.DataFrame,
    n_topics_col: str = 'n_topics',
    colors: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 5),
    savefile: Optional[str] = None
):
    """
    Plot histogram of number of topics per patent.
    
    Args:
        df: DataFrame with patent data
        n_topics_col: Column name with number of topics
        colors: Histogram color (default: '#345995')
        figsize: Figure size tuple
        savefile: Path to save figure
    """
    if colors is None:
        colors = '#345995'
    
    avg_topics = df[n_topics_col].mean()
    median_topics = df[n_topics_col].median()
    
    plt.figure(figsize=figsize)
    plt.hist(
        df[n_topics_col],
        bins=range(1, df[n_topics_col].max() + 2),
        edgecolor='black',
        color=colors[0] if isinstance(colors, (list, tuple)) else colors
    )
    plt.axvline(avg_topics, linestyle='--', color='red', label=f'Mean = {avg_topics:.2f}')
    plt.axvline(median_topics, linestyle=':', color='orange', label=f'Median = {median_topics:.0f}')
    
    plt.xlabel('Number of topics per patent')
    plt.ylabel('Number of patents')
    plt.title('Distribution of topics per patent')
    plt.legend()
    plt.tight_layout()
    
    if savefile:
        plt.savefig(savefile, dpi=300)
    plt.show()


def plot_top_topics_horizontal(
    topic_df: pd.DataFrame,
    topic_col: str = 'topic',
    count_col: str = 'count',
    top_n: int = 20,
    colors: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    savefile: Optional[str] = None
):
    """
    Plot horizontal bar chart of top patent topics.
    
    Args:
        topic_df: DataFrame with topic and count columns
        topic_col: Column name for topic names
        count_col: Column name for counts
        top_n: Number of top topics to show
        colors: Bar color (default: '#345995')
        figsize: Figure size tuple
        savefile: Path to save figure
    """
    if colors is None:
        colors = '#345995'
    
    plot_df = topic_df.head(top_n)
    
    plt.figure(figsize=figsize)
    plt.barh(plot_df[topic_col], plot_df[count_col], color=colors)
    plt.gca().invert_yaxis()
    
    plt.xlabel('Number of patents')
    plt.title(f'Top {top_n} patent topics')
    plt.tight_layout()
    
    if savefile:
        plt.savefig(savefile, dpi=300)
    plt.show()


def plot_collapsed_topics_horizontal(
    agg_df: pd.DataFrame,
    top_n: int = 20,
    fractional: bool = True,
    colors: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    savefile: Optional[str] = None
):
    """
    Plot horizontal bar chart of collapsed top-level topics.
    
    Args:
        agg_df: DataFrame with columns 'top_code', 'count', 'label'
        top_n: Number of top topics to show
        fractional: Whether counts are fractional
        colors: Bar color (default: '#345995')
        figsize: Figure size tuple
        savefile: Path to save figure
    """
    if colors is None:
        colors = '#345995'
    
    plot_df = agg_df.head(top_n).copy()
    plot_df['bar_label'] = plot_df.apply(
        lambda r: f"{r['top_code']} {r['label']}", axis=1
    )
    
    fig, ax = plt.subplots(figsize=figsize)
    
    y = np.arange(len(plot_df))
    bars = ax.barh(y, plot_df['count'], edgecolor='black', color=colors)
    
    # Annotate counts
    x_offset = plot_df['count'].max() * 0.01
    for bar, cnt in zip(bars, plot_df['count']):
        ax.text(
            bar.get_width() + x_offset,
            bar.get_y() + bar.get_height() / 2,
            f"{cnt:.0f}" if not fractional else f"{cnt:.2f}",
            va='center',
            ha='left',
            fontsize=9,
            fontweight='bold'
        )
    
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df['bar_label'])
    ax.invert_yaxis()
    
    ax.set_xlabel('Count (occurrences)' if not fractional else 'Fractional count (patent-weighted)')
    ax.set_title(f"Top {min(top_n, len(plot_df))} collapsed FOR top-level topics (high → low)")
    
    plt.tight_layout()
    if savefile:
        plt.savefig(savefile, dpi=300)
    plt.show()


def plot_drug_dev_by_country(
    df: pd.DataFrame,
    country_col: str = 'assignee_countries',
    year_col: str = 'year',
    weight_col: str = 'weight',
    uk_code: str = 'GB',
    colors: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 6),
    savefile: Optional[str] = None
):
    """
    Plot drug development patents by country (UK vs Others) over years.
    
    Args:
        df: DataFrame with exploded country data
        country_col: Column name for country codes
        year_col: Column name for years
        weight_col: Column name for fractional weights
        uk_code: ISO-2 code for UK (default: 'GB')
        colors: List of colors [UK_color, Others_color] (default: ['#6E8B3D', '#345995'])
        figsize: Figure size tuple
        savefile: Path to save figure
    """
    if colors is None:
        colors = ['#6E8B3D', '#345995']
    
    df_plot = (
        df.groupby([year_col, country_col])
        .agg({weight_col: 'sum'})
        .reset_index()
    )
    
    df_pivot = (
        df_plot.pivot(index=year_col, columns=country_col, values=weight_col)
        .sort_index()
    )
    
    df_pivot['UK'] = df_pivot.get(uk_code, 0)
    df_pivot['Others'] = df_pivot.drop(columns=[uk_code], errors='ignore').sum(axis=1)
    
    df_stacked = df_pivot[['UK', 'Others']]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    df_stacked.plot(
        kind='bar',
        stacked=True,
        ax=ax,
        color=colors
    )
    
    totals = df_stacked.sum(axis=1)
    
    for i, year in enumerate(df_stacked.index):
        uk = df_stacked.loc[year, 'UK']
        others = df_stacked.loc[year, 'Others']
        total = totals.loc[year]
        
        if uk > 0:
            ax.text(i, uk / 2, f'{uk:.1f}',
                   ha='center', va='center', fontsize=9, color='white')
        
        if others > 0:
            ax.text(i, uk + others / 2, f'{others:.1f}',
                   ha='center', va='center', fontsize=9, color='white')
    
    for i, total in enumerate(totals):
        ax.text(i, total, f'{total:.1f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_title('Drug Development Patents: UK vs Other Countries by Year')
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Patents (fractional count)')
    ax.legend(title='Assignee Country', loc='upper left')
    ax.set_xticklabels(df_stacked.index, rotation=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    if savefile:
        plt.savefig(savefile, dpi=300)
    plt.show()


def plot_development_stage_pies(
    df_stage: pd.DataFrame,
    country_col: str = 'assignee_countries',
    stage_col: str = 'development_stage',
    weight_col: str = 'weight',
    uk_code: str = 'GB',
    colors: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 6),
    savefile: Optional[str] = None
):
    """
    Plot pie charts comparing development stages between UK and other countries.
    
    Args:
        df_stage: DataFrame with aggregated stage data
        country_col: Column name for country codes
        stage_col: Column name for development stages
        weight_col: Column name for weights
        uk_code: ISO-2 code for UK (default: 'GB')
        colors: List of colors for stages (default: ['#345995', '#6E8B3D', '#D4AF37', '#B80C09'])
        figsize: Figure size tuple
        savefile: Path to save figure
    """
    if colors is None:
        colors = ['#345995', '#6E8B3D', '#D4AF37', '#B80C09']
    
    stage_order = df_stage[stage_col].dropna().unique().tolist()
    stage_color_map = dict(zip(stage_order, colors[::-1]))
    
    df_uk_stage = (
        df_stage[df_stage[country_col] == uk_code]
        .groupby(stage_col, as_index=False)[weight_col]
        .sum()
    )
    
    df_other_stage = (
        df_stage[df_stage[country_col] != uk_code]
        .groupby(stage_col, as_index=False)[weight_col]
        .sum()
    )
    
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    
    def draw_pie(ax, values, labels, title):
        wedges, texts, autotexts = ax.pie(
            values,
            labels=labels,
            autopct='%1.1f%%',
            colors=[stage_color_map.get(label, 'gray') for label in labels],
            startangle=140,
            labeldistance=1.15,
            pctdistance=0.75,
            wedgeprops=dict(linewidth=1, edgecolor='white')
        )
        
        for t in autotexts:
            t.set_color('white')
        
        ax.set_title(title, fontweight='bold')
        ax.axis('equal')
    
    draw_pie(
        axs[0],
        values=df_uk_stage[weight_col],
        labels=df_uk_stage[stage_col],
        title='UK'
    )
    
    draw_pie(
        axs[1],
        values=df_other_stage[weight_col],
        labels=df_other_stage[stage_col],
        title='Other Countries'
    )
    
    plt.suptitle('Drug Development Patents by Development Stage', fontsize=14)
    plt.tight_layout()
    
    if savefile:
        plt.savefig(savefile, dpi=300)
    plt.show()


def plot_top_cited_papers(
    df_papers: pd.DataFrame,
    title_col: str = 'title',
    count_col: str = 'count',
    top_n: int = 10,
    colors: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 7),
    savefile: Optional[str] = None
):
    """
    Plot horizontal bar chart of top cited UKBB papers.
    
    Args:
        df_papers: DataFrame with paper titles and citation counts
        title_col: Column name for paper titles
        count_col: Column name for citation counts
        top_n: Number of top papers to show
        colors: Bar color (default: '#345995')
        figsize: Figure size tuple
        savefile: Path to save figure
    """
    if colors is None:
        colors = '#345995'
    
    top_papers = df_papers.head(top_n)
    top_papers_ordered = top_papers.sort_values(count_col, ascending=True)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.barh(top_papers_ordered[title_col], top_papers_ordered[count_col], color=colors)
    ax.set_yticklabels(ax.get_yticklabels(), ha='right', fontsize=10)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Annotate counts
    for i, (count, title) in enumerate(zip(top_papers_ordered[count_col],
                                           top_papers_ordered[title_col])):
        ax.text(count + 0.5, i, str(count), va='center', fontsize=9, fontweight='bold')
    
    ax.set_xticklabels([])
    ax.set_xticks([])
    
    ax.annotate(
        f'Top {top_n} UKBB Papers Cited by Drug Development Patents',
        xy=(-6, 1.01),
        xycoords='axes fraction',
        ha='left',
        va='bottom',
        fontsize=12,
        fontweight='bold'
    )
    
    plt.subplots_adjust(left=0.8)
    
    if savefile:
        plt.savefig(savefile, dpi=300)
    plt.show()


def plot_model_agreement_pairwise(
    df_pairwise: pd.DataFrame,
    colors: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    savefile: Optional[str] = None
):
    """
    Plot pairwise agreement between models.
    
    Args:
        df_pairwise: DataFrame with columns 'Model 1', 'Model 2', 'Agreement %', 'Cohen\'s Kappa'
        colors: Bar color (default: '#345995')
        figsize: Figure size tuple
        savefile: Path to save figure
    """
    if colors is None:
        colors = '#345995'
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x_pos = np.arange(len(df_pairwise))
    bars = ax.bar(x_pos, df_pairwise['Agreement %'], color=colors, edgecolor='black')
    
    for i, (bar, val, kappa) in enumerate(zip(bars, df_pairwise['Agreement %'], df_pairwise['Cohen\'s Kappa'])):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{val:.1f}%\nκ={kappa:.2f}',
               ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    pair_labels = [f"{row['Model 1']}\nvs\n{row['Model 2']}" for _, row in df_pairwise.iterrows()]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(pair_labels, fontsize=9)
    
    ax.set_ylabel('Agreement (%)', fontsize=11)
    ax.set_title('Pairwise Agreement Between Models on Drug Development Labels', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    if savefile:
        plt.savefig(savefile, dpi=300)
    plt.show()


def plot_model_agreement_distribution(
    df_concordance: pd.DataFrame,
    yes_count_col: str = 'yes_count',
    colors: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    savefile: Optional[str] = None
):
    """
    Plot distribution of model agreement on labels.
    
    Args:
        df_concordance: DataFrame with yes_count column
        yes_count_col: Column name for yes count
        colors: Bar color (default: '#6E8B3D')
        figsize: Figure size tuple
        savefile: Path to save figure
    """
    if colors is None:
        colors = '#6E8B3D'
    
    fig, ax = plt.subplots(figsize=figsize)
    
    agreement_counts = df_concordance[yes_count_col].value_counts().sort_index()
    bars = ax.bar(agreement_counts.index, agreement_counts.values, color=colors, edgecolor='black')
    
    for bar, val in zip(bars, agreement_counts.values):
        pct = val / len(df_concordance) * 100
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
               f'{val}\n({pct:.1f}%)',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Number of models labeling as "yes"', fontsize=11)
    ax.set_ylabel('Number of patents', fontsize=11)
    ax.set_title('Distribution of Model Agreement on Drug Development Labels', fontsize=12, fontweight='bold')
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    if savefile:
        plt.savefig(savefile, dpi=300)
    plt.show()



#=============================================================================
# TOPIC PARSING FUNCTIONS 
#=============================================================================
def safe_parse_category(val):
    """Return list of dicts from category_for field; robust to strings like "[]"."""
    if val is None:
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return []
        try:
            parsed = ast.literal_eval(s)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            # best-effort: try JSON-like parsing fallback
            try:
                import json
                parsed = json.loads(s)
                return parsed if isinstance(parsed, list) else []
            except Exception:
                return []
    return []

def extract_name_and_leading_code(topic_dict):
    """
    Given a dict like {'id': '80017', 'name': '46 Information and Computing Sciences'},
    return (leading_code_str, display_name_str).
    leading_code_str: the numeric prefix extracted (e.g. '46' or '3202') or None.
    display_name_str: the name without the numeric prefix (e.g. 'Information and Computing Sciences').
    """
    if not isinstance(topic_dict, dict):
        return None, None
    name = topic_dict.get('name') or ''
    if not isinstance(name, str) or not name.strip():
        return None, None
    name = name.strip()
    # match leading number(s) possibly followed by spaces
    m = re.match(r'^\s*(\d+)\s*(.*)$', name)
    if m:
        leading_code = m.group(1)         # e.g. '3202' or '46'
        rest = m.group(2).strip() or None # e.g. 'Clinical Sciences' (may be empty)
        return leading_code, rest
    else:
        # no leading numeric code: return None code, whole name as label
        return None, name

def top_level_code_from_leading(leading_code):
    """
    Convert a leading code like '3202' -> top-level '32',
    '46' -> '46'. Returns None if leading_code is None/invalid.
    """
    if not leading_code:
        return None
    s = str(leading_code)
    # choose first two chars as top-level; if single-digit, keep it as-is
    if len(s) >= 2:
        return s[:2]
    return s

# -----------------------
# For each patent, turn its categories into a list of (top_level_code, top_level_label)
# -----------------------
def collapse_to_top_level(cat_parsed):
    """
    Input: list of topic dicts
    Output: list of tuples (top_code, top_label) possibly with duplicates
    """
    results = []
    for d in cat_parsed:
        leading_code, name_rest = extract_name_and_leading_code(d)
        top_code = top_level_code_from_leading(leading_code)
        # build a provisional label: prefer the broad-level name if available
        if top_code is None:
            # fallback: use full name if no numeric code
            label = d.get('name') or None
        else:
            # try to find a "top-level" label: if the topic name already begins with top_code text
            # name_rest may be the fine-grained label (e.g., 'Clinical Sciences')
            label = None
            if name_rest:
                label = name_rest
            else:
                # if no rest, fallback to the full 'name' without numeric prefix
                full = d.get('name', '')
                m2 = re.match(r'^\s*\d+\s*(.*)$', full)
                label = m2.group(1).strip() if m2 and m2.group(1) else full
        results.append((top_code, label))
    return results



def safe_parse_category(val):
    """Parse category_for field safely into list of dicts."""
    if val is None:
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []

def extract_topic_names(category_list):
    """Extract topic names from parsed category list."""
    return [
        d.get('name') for d in category_list
        if isinstance(d, dict) and d.get('name')
    ]
    
# Try pycountry for robust name->ISO conversion
try:
    import pycountry
    _HAS_PYCOUNTRY = True
except Exception:
    _HAS_PYCOUNTRY = False

# small fallback mapping; extend as needed
_FALLBACK_NAME_TO_ISO = {
    'UNITED KINGDOM': 'GB',
    'UNITED STATES': 'US',
    'UNITED STATES OF AMERICA': 'US',
    'USA': 'US',
    'SOUTH KOREA': 'KR',
    'REPUBLIC OF KOREA': 'KR',
    'KOREA': 'KR',
    'JAPAN': 'JP',
    'GERMANY': 'DE',
    'FRANCE': 'FR',
    'CHINA': 'CN',
    "PEOPLE'S REPUBLIC OF CHINA": 'CN',
    'AUSTRALIA': 'AU',
    'CANADA': 'CA',
    'ITALY': 'IT',
    'SPAIN': 'ES',
}

def to_iso2(country_candidate: Optional[str]) -> Optional[str]:
    """Convert a country name or code-like string to ISO-2 code (uppercase)."""
    if country_candidate is None:
        return None
    if not isinstance(country_candidate, str):
        return None
    s = country_candidate.strip()
    if not s:
        return None

    # Already 2-letter code
    if len(s) == 2 and s.isalpha():
        return s.upper()

    # 3-letter code -> map via pycountry or small map
    if len(s) == 3 and s.isalpha():
        if _HAS_PYCOUNTRY:
            try:
                c = pycountry.countries.get(alpha_3=s.upper())
                if c:
                    return c.alpha_2
            except Exception:
                pass
        _three_to_two = {'GBR': 'GB', 'USA': 'US', 'KOR': 'KR', 'JPN': 'JP', 'CHN': 'CN'}
        if s.upper() in _three_to_two:
            return _three_to_two[s.upper()]

    # Try pycountry name search
    if _HAS_PYCOUNTRY:
        try:
            # fuzzy search
            candidates = pycountry.countries.search_fuzzy(s)
            if candidates:
                return candidates[0].alpha_2
        except Exception:
            pass

    # Exact fallback map
    key = s.upper()
    if key in _FALLBACK_NAME_TO_ISO:
        return _FALLBACK_NAME_TO_ISO[key]

    # token match
    tokens = re.split(r'[,;/\-\(\)\s]+', key)
    for t in tokens:
        if t in _FALLBACK_NAME_TO_ISO:
            return _FALLBACK_NAME_TO_ISO[t]

    # fuzzy fallback against fallback keys
    match = difflib.get_close_matches(key, list(_FALLBACK_NAME_TO_ISO.keys()), n=1, cutoff=0.8)
    if match:
        return _FALLBACK_NAME_TO_ISO[match[0]]

    return None

def _safe_literal_eval(val: Any) -> Any:
    """Safely parse string representations of lists/dicts where possible."""
    if val is None:
        return None
    if isinstance(val, str):
        s = val.strip()
        if (s.startswith('[') and s.endswith(']')) or (s.startswith('{') and s.endswith('}')):
            try:
                return ast.literal_eval(s)
            except Exception:
                return val
    return val

def infer_country_from_name(name: Optional[str], use_low_confidence: bool = False) -> Optional[str]:
    """Rule-based inference returning ISO-2 or None."""
    if not isinstance(name, str) or not name.strip():
        return None
    s = name.strip()
    s_lower = s.lower()

    # high-confidence markers
    high_confidence_rules = [
        (r'주식회사', 'KR'),
        (r'산학협력단', 'KR'),
        (r'재단법인', 'KR'),
        (r'대학교', 'KR'),
        (r'서울대학교', 'KR'),
        (r'연세대학교', 'KR'),
        (r'이화여자대학교', 'KR'),
        (r'株式会社', 'JP'),
        (r'有限会社', 'JP'),
        (r'\bplc\b', 'GB'),
        (r'\bpty\s?ltd\b', 'AU'),
        (r'\bgmbh\b', 'DE'),
    ]
    for pattern, iso in high_confidence_rules:
        if re.search(pattern, s_lower, flags=re.IGNORECASE):
            return iso

    # low-confidence patterns (enable by flag)
    if use_low_confidence:
        low_confidence_rules = [
            (r'\binc\.?\b', 'US'),
            (r'\bllc\b', 'US'),
            (r'\bltd\b', 'GB'),
            (r'\bsa\b', 'FR'),
        ]
        for pattern, iso in low_confidence_rules:
            if re.search(pattern, s_lower, flags=re.IGNORECASE):
                return iso

    # detect trailing country in parentheses or after comma: "Name (Japan)" or "Name, Japan"
    m = re.search(r'[\(\[,] *([A-Za-z \'\.]+?) *[\)\]]?$', s)
    if m:
        maybe_country = m.group(1)
        iso = to_iso2(maybe_country)
        if iso:
            return iso

    # direct convert if the string itself is a country name
    iso = to_iso2(s)
    if iso:
        return iso

    return None

def assign_assignee_countries_with_iso(
    df: pd.DataFrame,
    assignee_col: str = 'assignees',
    assignee_names_col: str = 'assignee_names',
    use_low_confidence: bool = False,
    primary_pick: str = 'first'
) -> pd.DataFrame:
    """
    Assign country info using:
      1) authoritative country info inside `assignees` (list of dicts) if present
      2) fallback rule-based inference using `assignee_names` column
      3) fallback to names inside `assignees` dicts if assignee_names is missing
    Outputs columns:
      - assignee_countries: list of ISO-2 codes or None
      - assignee_country_primary: chosen ISO-2 or None
      - assignee_country_source: 'authority', 'rule_high_conf', 'rule_low_conf', 'authority_raw', or 'unknown'
    """
    out_countries = []
    out_primary = []
    out_source = []

    # iterate rows
    for idx, row in df.iterrows():
        raw_assignees = row.get(assignee_col, None)
        assignee_names_val = row.get(assignee_names_col, None)

        parsed = _safe_literal_eval(raw_assignees)
        countries = []
        sources = []
        found_authority = False

        # 1) If parsed structured list/dicts contains country info -> use it (authoritative)
        if isinstance(parsed, list) and parsed:
            any_with_country = any(isinstance(x, dict) and (x.get('country_code') or x.get('country') or x.get('country_name')) for x in parsed)
            if any_with_country:
                for item in parsed:
                    if isinstance(item, dict):
                        cc = item.get('country_code') or item.get('country') or item.get('country_name')
                        if cc:
                            iso = to_iso2(cc)  # normalise even if code or full name
                            # if iso None but cc looks like 2-letter, normalise uppercase
                            if iso is None and isinstance(cc, str) and len(cc.strip()) == 2:
                                iso = cc.strip().upper()
                            countries.append(iso)
                            sources.append('authority' if iso else 'authority_raw')
                        else:
                            # we will decide to infer from assignee_names first (below)
                            countries.append(None)
                            sources.append('unknown')
                found_authority = True

        # 2) If no authority, use assignee_names column for rule-based inference (preferred)
        if not found_authority:
            # if assignee_names_val exists and parse it: could be list or single name string
            parsed_names = _safe_literal_eval(assignee_names_val)
            #print(assignee_names_val,parsed_names)
            used_names = []

            if isinstance(parsed_names, list):
                # a list of strings
                used_names = [n for n in parsed_names if isinstance(n, str) and n.strip()]
            elif isinstance(parsed_names, str) and parsed_names.strip():
                # single string — could encode a list: "['A']" or plain "A"
                s = parsed_names.strip()
                if s.startswith('[') and s.endswith(']'):
                    try:
                        il = ast.literal_eval(s)
                        used_names = [n for n in il if isinstance(n, str) and n.strip()]
                    except Exception:
                        used_names = [s]
                else:
                    used_names = [s]
            else:
                used_names = []

            if used_names:
                #print(used_names)
                for nm in used_names:
                    iso = infer_country_from_name(nm, use_low_confidence)
                    countries.append(iso)
                    sources.append('rule_low_conf' if use_low_confidence and iso else ('rule_high_conf' if iso else 'unknown'))
            else:
                # assignee_names not present or empty -> fall back to names inside parsed assignees (if any)
                if isinstance(parsed, list) and parsed:
                    for item in parsed:
                        if isinstance(item, dict):
                            name = item.get('name') or item.get('applicant') or None
                            iso = infer_country_from_name(name, use_low_confidence)
                            countries.append(iso)
                            sources.append('rule_low_conf' if use_low_confidence and iso else ('rule_high_conf' if iso else 'unknown'))
                        elif isinstance(item, str):
                            iso = infer_country_from_name(item, use_low_confidence)
                            countries.append(iso)
                            sources.append('rule_low_conf' if use_low_confidence and iso else ('rule_high_conf' if iso else 'unknown'))
                        else:
                            countries.append(None)
                            sources.append('unknown')
                else:
                    # parsed not a list: try single dict or string
                    if isinstance(parsed, dict):
                        name = parsed.get('name') or parsed.get('applicant') or None
                        iso = infer_country_from_name(name, use_low_confidence)
                        countries.append(iso)
                        sources.append('rule_low_conf' if use_low_confidence and iso else ('rule_high_conf' if iso else 'unknown'))
                    elif isinstance(parsed, str) and parsed.strip():
                        # string might encode list
                        s = parsed.strip()
                        if s.startswith('[') and s.endswith(']'):
                            try:
                                inner = ast.literal_eval(s)
                                for item in inner:
                                    if isinstance(item, str):
                                        iso = infer_country_from_name(item, use_low_confidence)
                                        countries.append(iso)
                                        sources.append('rule_low_conf' if use_low_confidence and iso else ('rule_high_conf' if iso else 'unknown'))
                                    else:
                                        countries.append(None)
                                        sources.append('unknown')
                            except Exception:
                                iso = infer_country_from_name(parsed, use_low_confidence)
                                countries.append(iso)
                                sources.append('rule_low_conf' if use_low_confidence and iso else ('rule_high_conf' if iso else 'unknown'))
                        else:
                            iso = infer_country_from_name(parsed, use_low_confidence)
                            countries.append(iso)
                            sources.append('rule_low_conf' if use_low_confidence and iso else ('rule_high_conf' if iso else 'unknown'))
                    else:
                        countries.append(None)
                        sources.append('unknown')

        # normalise to uppercase ISO strings or None
        norm_countries = [c.upper() if isinstance(c, str) else None for c in countries]

        # choose primary country
        primary = None
        source_label = 'unknown'
        if 'authority' in sources:
            for c, s in zip(norm_countries, sources):
                if s == 'authority' and c:
                    primary = c
                    source_label = 'authority'
                    break

        if primary is None:
            non_none = [c for c in norm_countries if c]
            if non_none:
                if primary_pick == 'first':
                    primary = non_none[0]
                elif primary_pick == 'most_common':
                    primary = Counter(non_none).most_common(1)[0][0]
                # identify its source
                for c, s in zip(norm_countries, sources):
                    if c == primary:
                        source_label = s
                        break
            else:
                primary = None
                source_label = 'unknown'

        out_countries.append(norm_countries)
        out_primary.append(primary)
        out_source.append(source_label)

    df = df.copy()
    df['assignee_countries'] = out_countries
    df['assignee_country_primary'] = out_primary
    df['assignee_country_source'] = out_source
    return df


def save_clean_authors(row):
    val = row['authors']
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val)
        except Exception:
            parsed = []
    else:
        # print(f"Unexpected type for authors: {type(val)} in row {row.name}")
        parsed = []
    return parsed if isinstance(parsed, list) else []


def country_finder(author_str):
    if isinstance(author_str, str):
        return re.findall(r"'country_code': '([^']+)'", author_str)
    else:
        #print(f"Unexpected type for authors: {type(author_str)} {author_str}")
        return []
    
    
# =========================
# ipcr functions 
# =========================
def map_ipcr_code_to_standardized_code(code):
    front_code = code.split('/')[0]
    front_code = front_code[0:4]+f'{int(front_code[4:]):04d}'
    tail_code = code.split('/')[1] if '/' in code else ''
    tail_code = tail_code+'0'*(6-len(tail_code)) if tail_code else ''
    return front_code + ( tail_code if tail_code else '')


def map_ipc_topic_titles(df_ipc,codes):
    titles = []
    for code in codes:
        if code in df_ipc['symbol'].values:
            title = df_ipc.loc[df_ipc['symbol'] == code, 'title'].values[0]
            titles.append(title)
    return titles

def ipc_code_to_top_level(df_ipc,codes,level):
    """
    Given a list of IPC codes, return the corresponding top-level codes (Section, Class, or Subclass).
     - codes: list of IPC codes (e.g., ['A01', 'B23K 26/00'])
     - level: one of 'Section', 'Class', 'Subclass' to specify which top-level code to extract
    Returns: list of top-level codes corresponding to   the input codes
     - For example, if level='Section' and input code is 'A01', it would return 'A'. If level='Class' and input code is 'B23K 26/00', it would return 'B23'. If level='Subclass' and input code is '
    """
    top_codes = []
    for code in codes:
        if code in df_ipc['symbol'].values:
            top_code = df_ipc.loc[df_ipc['symbol'] == code, level].values[0]
            top_codes.append(top_code)
    return top_codes


import re
import pandas as pd
from bs4 import BeautifulSoup

def clean_patent_abstract(text: str) -> str:
    if pd.isna(text):
        return text

    s = str(text)

    # --- 1) Repair some common malformed tag patterns seen in scraped patents
    # Example: (c<sub>4</sub>-c<sub>6)</sub>  -> try to fix the "6)" inside sub
    s = re.sub(r'<sub>(\d+)\)</sub>', r'<sub>\1</sub>)', s)

    # --- 2) Convert <sub> and <sup> to explicit plain-text markers BEFORE stripping tags
    # r<sup>1</sup> -> r^1
    s = re.sub(r'<sup>\s*([^<]+?)\s*</sup>', r'^\1', s, flags=re.IGNORECASE)
    # c<sub>4</sub> -> c_4
    s = re.sub(r'<sub>\s*([^<]+?)\s*</sub>', r'_\1', s, flags=re.IGNORECASE)

    # --- 3) Strip remaining HTML/XML tags safely
    s = BeautifulSoup(s, "html.parser").get_text(separator=" ")

    # --- 4) Normalize punctuation/spacing
    s = s.replace("\u00a0", " ")                # non-breaking spaces
    s = re.sub(r'\s+', ' ', s).strip()          # collapse whitespace
    s = re.sub(r'\s+([,.;:])', r'\1', s)        # remove space before punctuation
    s = re.sub(r'\(\s+', '(', s)                # "( " -> "("
    s = re.sub(r'\s+\)', ')', s)                # " )" -> ")"
    s = re.sub(r'\s*-\s*', '-', s)              # normalize hyphen spacing

    # --- 5) Optional: standardize common "C4-C6" style ranges for NLP
    # Turn "c_4-c_6" or "C_4-C_6" into "C4-C6"
    s = re.sub(r'\b([cC])_(\d+)\b', r'\1\2', s)

    return s


def extract_all_countries(val: Any) -> List[str]:
    """Extract ISO country codes from a patent assignee country payload."""
    try:
        if val is None:
            return []
        if isinstance(val, str):
            try:
                val = ast.literal_eval(val)
            except Exception:
                return []
        if isinstance(val, list):
            countries = []
            for item in val:
                if isinstance(item, dict) and item.get('id'):
                    countries.append(item['id'])
                elif isinstance(item, str):
                    countries.append(item)
            return countries
    except Exception:
        pass
    return []


def _normalize_topic_code_list(topics: Any) -> List[str]:
    if not topics:
        return []
    if isinstance(topics, list):
        if topics and isinstance(topics[0], tuple):
            return [code for code, _ in topics if code]
        return [code for code in topics if code]
    return []


def analyze_topic_diversity(df_patent: pd.DataFrame, topics_col: str = 'top_level_topics') -> pd.DataFrame:
    """Return a copy of the dataframe with a topic_count column."""
    out = df_patent.copy()
    out['topic_count'] = out[topics_col].apply(lambda x: len(_normalize_topic_code_list(x)))
    return out


def build_topic_cooccurrence_network(
    df_patent: pd.DataFrame,
    topics_col: str = 'top_level_topics',
    min_weight: int = 2,
):
    """Build an undirected co-occurrence network from topic lists."""
    import networkx as nx
    from itertools import combinations

    pair_counter = Counter()
    for topics in df_patent[topics_col]:
        codes = sorted(set(_normalize_topic_code_list(topics)))
        for left, right in combinations(codes, 2):
            pair_counter[(left, right)] += 1

    graph = nx.Graph()
    for (left, right), weight in pair_counter.items():
        if weight >= min_weight:
            graph.add_edge(left, right, weight=weight)

    return graph, pair_counter


def plot_topic_cooccurrence_network(
    graph,
    code_to_label: Optional[Dict[str, str]] = None,
    figsize: Tuple[int, int] = (12, 10),
    savefile: Optional[str] = None,
):
    """Plot a topic co-occurrence network."""
    import networkx as nx

    if graph.number_of_nodes() == 0:
        print('No topic pairs found meeting the co-occurrence threshold.')
        return None, None

    if code_to_label is None:
        code_to_label = {}

    pos = nx.spring_layout(graph, seed=42, k=0.7)
    node_sizes = [400 + 250 * graph.degree(node) for node in graph.nodes()]
    edge_widths = [0.5 + graph[u][v].get('weight', 1) * 0.4 for u, v in graph.edges()]

    fig, ax = plt.subplots(figsize=figsize)
    nx.draw_networkx_edges(graph, pos, width=edge_widths, alpha=0.35, ax=ax)
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=node_sizes,
        node_color='#4C72B0',
        alpha=0.85,
        edgecolors='white',
        linewidths=1.5,
        ax=ax,
    )
    labels = {node: code_to_label.get(node, str(node)) for node in graph.nodes()}
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8, font_weight='bold', ax=ax)
    edge_labels = {(u, v): graph[u][v].get('weight', 1) for u, v in graph.edges()}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=7, ax=ax)
    ax.axis('off')
    plt.tight_layout()

    if savefile:
        plt.savefig(savefile, dpi=200, bbox_inches='tight')
        print(f'Network plot saved to: {savefile}')
    plt.show()
    return fig, ax


def analyze_country_topics(
    df_patent: pd.DataFrame,
    topics_col: str = 'top_level_topics',
    country_col: str = 'country_list',
    code_to_label: Optional[Dict[str, str]] = None,
    top_countries: int = 8,
    top_topics: int = 8,
):
    """Build a fractional country-topic matrix and summary tables."""
    if code_to_label is None:
        code_to_label = {}

    rows = []
    country_counter = Counter()
    topic_counter = Counter()

    for _, row in df_patent.iterrows():
        countries = extract_all_countries(row.get(country_col))
        topics = _normalize_topic_code_list(row.get(topics_col))
        unique_topics = sorted(set(topics))
        if not countries or not unique_topics:
            continue

        weight = 1.0 / len(countries)
        for country in countries:
            country_counter[country] += weight
            for code in unique_topics:
                topic_counter[code] += weight
                rows.append({
                    'country': country,
                    'topic_code': code,
                    'topic_label': code_to_label.get(code, f'Code {code}'),
                    'weight': weight,
                })

    df_country_topic = pd.DataFrame(rows)
    if df_country_topic.empty:
        return df_country_topic, pd.DataFrame(), [], [], country_counter, topic_counter

    topic_by_country = (
        df_country_topic.groupby(['country', 'topic_code', 'topic_label'])['weight']
        .sum()
        .reset_index(name='count')
    )

    top_country_codes = [country for country, _ in country_counter.most_common(top_countries)]
    top_topic_codes = [code for code, _ in topic_counter.most_common(top_topics)]

    pivot = (
        topic_by_country[topic_by_country['country'].isin(top_country_codes) & topic_by_country['topic_code'].isin(top_topic_codes)]
        .pivot(index='topic_code', columns='country', values='count')
        .fillna(0)
    )
    pivot = pivot.reindex(index=top_topic_codes, columns=top_country_codes, fill_value=0)

    return df_country_topic, pivot, top_country_codes, top_topic_codes, country_counter, topic_counter


def plot_country_topic_heatmap(
    pivot_country_topic: pd.DataFrame,
    code_to_label: Optional[Dict[str, str]] = None,
    figsize: Tuple[int, int] = (8, 6),
    savefile: Optional[str] = None,
    ax=None,
    cmap=None,
    title: Optional[str] = None,
):
    """Plot a normalized country-topic heatmap."""

    if code_to_label is None:
        code_to_label = {}

    if pivot_country_topic.empty:
        print('No country-topic data available for heatmap.')
        return None, None

    pivot_normalized = pivot_country_topic.div(pivot_country_topic.sum(axis=0), axis=1).fillna(0) * 100
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if cmap is None:
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#F5F7FA', '#4C72B0'])

    im = ax.imshow(pivot_normalized.values, cmap=cmap, aspect='auto', vmin=0, vmax=100)
    ax.set_xticks(range(len(pivot_normalized.columns)))
    ax.set_yticks(range(len(pivot_normalized.index)))
    ax.set_xticklabels([('UK' if col == 'GB' else col) for col in pivot_normalized.columns], rotation=0, ha='center')
    ax.set_yticklabels([code_to_label.get(code, code) for code in pivot_normalized.index])

    for i in range(len(pivot_normalized.index)):
        for j in range(len(pivot_normalized.columns)):
            value = pivot_normalized.values[i, j]
            if value > 0:
                ax.text(j, i, f'{value:.1f}%', ha='center', va='center', color='white', fontsize=7)

    if title is None:
        title = 'Patent Topics by Country (%)'
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel('Country')
    ax.set_ylabel('Topic Division')
    plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02, fraction=0.046)
    if created_fig:
        plt.tight_layout()

    if savefile:
        plt.savefig(savefile, dpi=200, bbox_inches='tight')
        print(f'Heatmap saved to: {savefile}')
    plt.show()
    return fig, ax


def plot_country_dominant_topics(
    topic_by_country: pd.DataFrame,
    top_countries: List[str],
    code_to_label: Optional[Dict[str, str]] = None,
    figsize: Tuple[int, int] = (8, 6),
    savefile: Optional[str] = None,
):
    """Plot the dominant topic per country as a horizontal bar chart."""
    if code_to_label is None:
        code_to_label = {}

    if topic_by_country.empty:
        print('No country-topic data available for dominant topic plot.')
        return None, None

    dominant_rows = []
    for country in top_countries:
        country_rows = topic_by_country[topic_by_country['country'] == country]
        if country_rows.empty:
            continue
        top_row = country_rows.nlargest(1, 'count').iloc[0]
        dominant_rows.append(top_row)

    if not dominant_rows:
        print('No dominant topic rows found.')
        return None, None

    dominant_df = pd.DataFrame(dominant_rows).sort_values('count', ascending=True)
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(dominant_df['country'], dominant_df['count'], color='#4C72B0', edgecolor='black', linewidth=1)
    for bar, topic_code, count in zip(bars, dominant_df['topic_code'], dominant_df['count']):
        topic_label = code_to_label.get(topic_code, topic_code)
        if len(topic_label) > 30:
            topic_label = topic_label[:30] + '...'
        ax.text(count + 0.5, bar.get_y() + bar.get_height() / 2, f'{topic_label} ({count:.2f})', va='center', fontsize=8)
    ax.set_xlabel('Fractional Patent Count in Dominant Topic')
    ax.set_ylabel('Country')
    ax.set_title('Dominant Topic per Country (Fractional)', fontsize=10, fontweight='bold')
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    if savefile:
        plt.savefig(savefile, dpi=200, bbox_inches='tight')
        print(f'Dominant-topic plot saved to: {savefile}')
    plt.show()
    return fig, ax


def plot_two_level_hierarchy(
    counts: pd.DataFrame,
    parent_code_col: str,
    parent_label_col: str,
    child_code_col: str,
    child_label_col: str,
    parent_name: str = 'Parent',
    child_name: str = 'Child',
    top_parent_n: int = 5,
    top_child_n: int = 8,
    figsize: Tuple[int, int] = (14, 20),
    savefile: Optional[str] = None,
):
    """Draw a bubble-tree style hierarchy plot for a two-level taxonomy."""
    if counts.empty:
        print(f'No data available for {parent_name} hierarchy plot.')
        return None, None

    parent_totals = counts.groupby([parent_code_col, parent_label_col])['count'].sum().reset_index()
    parent_totals = parent_totals.sort_values('count', ascending=False).head(top_parent_n)
    counts = counts.merge(parent_totals[[parent_code_col]], on=parent_code_col, how='inner')
    parent_order = parent_totals[parent_code_col].tolist()

    fig, ax = plt.subplots(figsize=figsize)
    colors_palette = plt.cm.Set3(range(len(parent_order)))
    parent_colors = dict(zip(parent_order, colors_palette))

    y_spacing = 1.5
    x_parent = 0
    x_child = 6
    y_current = 0

    for parent_code in parent_order:
        parent_data = counts[counts[parent_code_col] == parent_code]
        parent_label = parent_data[parent_label_col].iloc[0]
        parent_total = parent_data['count'].sum()

        children = (
            parent_data.groupby([child_code_col, child_label_col])['count']
            .sum()
            .reset_index()
            .sort_values('count', ascending=False)
            .head(top_child_n)
        )

        n_children = len(children)
        child_y_start = y_current
        child_y_spacing = 0.7
        parent_y = child_y_start + (n_children - 1) * child_y_spacing / 2

        ax.scatter(x_parent, parent_y, s=min(2200, 500 + parent_total * 6), color=parent_colors[parent_code], edgecolor='black', linewidth=2, zorder=3, alpha=0.85)
        ax.text(x_parent - 0.3, parent_y, f"{parent_code}\n{parent_label}\n({parent_total})", ha='right', va='center', fontsize=9, fontweight='bold')

        for child_idx, (_, child_row) in enumerate(children.iterrows()):
            child_code = child_row[child_code_col]
            child_label = child_row[child_label_col]
            child_count = child_row['count']
            child_y = child_y_start + child_idx * child_y_spacing
            ax.plot([x_parent, x_child], [parent_y, child_y], color=parent_colors[parent_code], linewidth=1.3, alpha=0.6, zorder=1)
            ax.scatter(x_child, child_y, s=min(900, 200 + child_count * 4), color=parent_colors[parent_code], edgecolor='gray', linewidth=1, zorder=2, alpha=0.55)
            child_text = child_label if len(str(child_label)) <= 35 else str(child_label)[:32] + '...'
            ax.text(x_child + 0.2, child_y, f"{child_code}: {child_text} ({child_count})", ha='left', va='center', fontsize=7)

        y_current += (n_children * child_y_spacing) + y_spacing

    ax.set_xlim(-2, 15)
    ax.set_ylim(-1, y_current)
    ax.axis('off')
    ax.text(x_parent, -0.5, parent_name, ha='center', va='top', fontsize=9, style='italic')
    ax.text(x_child, -0.5, child_name, ha='center', va='top', fontsize=9, style='italic')
    plt.tight_layout()

    if savefile:
        plt.savefig(savefile, dpi=200, bbox_inches='tight')
        print(f'Hierarchy plot saved to: {savefile}')
    plt.show()
    return fig, ax


def prepare_rcdc_macro_context(
    df_patent: pd.DataFrame,
    category_col: str = 'category_rcdc',
    summary_csv: str = 'file/paten_rcdc_macro/cluster_label_summary_louvain.csv',
    macro_cluster_names: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    """Prepare shared RCDC macro-cluster lookup tables for downstream plots."""
    df = df_patent.copy()

    if macro_cluster_names is None:
        macro_cluster_names = {
            0: 'Stem Cell & Regenerative Medicine',
            1: 'Biomedical Engineering & Digital Health',
            2: 'Aging & Neurodegenerative Disorders',
            3: 'Mental Health & Behavioral Disorders',
            4: 'Genomic & Precision Medicine',
            5: 'Metabolic & Hepatic Disorders',
            6: 'Infectious & Rare Disease Research ',
        }

    if category_col not in df.columns:
        raise KeyError(f"{category_col!r} column is missing from the input dataframe")

    df[category_col] = df[category_col].apply(safe_parse_category)

    cat_dict: Dict[str, str] = {}
    for cats in df[category_col]:
        for c in cats:
            if isinstance(c, dict) and 'name' in c and 'id' in c:
                cat_dict[str(c['id'])] = c['name']

    df_rcdc_macro_labels = pd.read_csv(summary_csv)
    df_rcdc_macro_labels['top_label_names'] = df_rcdc_macro_labels['top_labels'].apply(
        lambda x: [cat_dict.get(lab, 'Unknown') for lab in str(x).split(';')]
    )
    df_rcdc_macro_labels['cluster_names'] = df_rcdc_macro_labels['community'].map(macro_cluster_names)

    top_topic_dict = {
        x: str(y).split(';')
        for x, y in zip(df_rcdc_macro_labels['community'], df_rcdc_macro_labels['all_labels'])
    }
    code_to_macro = {
        str(code): macro
        for macro, code_list in top_topic_dict.items()
        for code in code_list
    }

    return {
        'df_rcdc_macro_labels': df_rcdc_macro_labels,
        'cat_dict': cat_dict,
        'macro_cluster_names': macro_cluster_names,
        'top_topic_dict': top_topic_dict,
        'code_to_macro': code_to_macro,
    }


def plot_rcdc_macro_hierarchy(
    df_patent: pd.DataFrame,
    rcdc_context: Dict[str, Any],
    category_col: str = 'category_rcdc',
    top_level_topics_col: str = 'top_level_topics',
    top_macro: int = 8,
    figsize: Tuple[int, int] = (14, 20),
    savefile: Optional[str] = None,
):
    """Plot the macro-cluster to RCDC category tree-style hierarchy."""
    import ast

    cat_dict = rcdc_context['cat_dict']
    macro_cluster_names = rcdc_context['macro_cluster_names']
    code_to_macro = rcdc_context['code_to_macro']

    rcdc_rows = []
    for _, row in df_patent.iterrows():
        raw_topics = row.get(category_col, [])

        if isinstance(raw_topics, str):
            try:
                raw_topics = ast.literal_eval(raw_topics)
            except Exception:
                raw_topics = []

        if not isinstance(raw_topics, list):
            continue

        for topic in raw_topics:
            if not isinstance(topic, dict):
                continue

            rcdc_id = topic.get('id')
            if rcdc_id is None:
                continue

            rcdc_id_str = str(rcdc_id)
            macro_code = code_to_macro.get(rcdc_id_str, code_to_macro.get(rcdc_id))
            if macro_code is None:
                continue

            rcdc_label = topic.get('name') or cat_dict.get(rcdc_id_str, cat_dict.get(rcdc_id, f'RCDC {rcdc_id_str}'))
            rcdc_rows.append({
                'macro_code': macro_code,
                'macro_label': macro_cluster_names.get(macro_code, f'Macro {macro_code}'),
                'rcdc_id': rcdc_id_str,
                'rcdc_label': rcdc_label,
            })

    hier_df_rcdc = pd.DataFrame(rcdc_rows)
    if hier_df_rcdc.empty:
        print('No category_rcdc hierarchy data available for dendrogram plot.')
        return None, None

    counts = (
        hier_df_rcdc
        .groupby(['macro_code', 'macro_label', 'rcdc_id', 'rcdc_label'])
        .size()
        .reset_index(name='count')
    )

    macro_totals = (
        counts.groupby(['macro_code', 'macro_label'])['count']
        .sum()
        .reset_index()
        .sort_values('count', ascending=False)
        .head(top_macro)
    )

    counts = counts.merge(macro_totals[['macro_code']], on='macro_code', how='inner')
    macro_order = macro_totals['macro_code'].tolist()

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.Set3(range(len(macro_order)))
    macro_colors = dict(zip(macro_order, colors))

    y_spacing = 1.5
    x_macro = 0
    x_rcdc = 6
    y_current = 0

    for macro_code in macro_order:
        macro_data = counts[counts['macro_code'] == macro_code]
        macro_label = macro_data['macro_label'].iloc[0]
        macro_total = macro_data['count'].sum()

        children = (
            macro_data.groupby(['rcdc_id', 'rcdc_label'])['count']
            .sum()
            .reset_index()
            .sort_values('count', ascending=False)
        )

        n_children = len(children)
        child_y_start = y_current
        child_y_spacing = 0.7
        macro_y = child_y_start + (n_children - 1) * child_y_spacing / 2

        ax.scatter(
            x_macro, macro_y, s=min(2200, 500 + macro_total * 6),
            color=macro_colors[macro_code], edgecolor='black', linewidth=2,
            zorder=3, alpha=0.85
        )
        ax.text(
            x_macro - 0.3, macro_y,
            f"{macro_code}\n{macro_label}\n({macro_total})",
            ha='right', va='center', fontsize=9, fontweight='bold'
        )

        for child_idx, (_, child_row) in enumerate(children.iterrows()):
            rcdc_id = child_row['rcdc_id']
            rcdc_label = child_row['rcdc_label']
            rcdc_count = child_row['count']
            child_y = child_y_start + child_idx * child_y_spacing

            ax.plot([x_macro, x_rcdc], [macro_y, child_y], color=macro_colors[macro_code], linewidth=1.3, alpha=0.6, zorder=1)
            ax.scatter(
                x_rcdc, child_y,
                s=min(900, 200 + rcdc_count * 4),
                color=macro_colors[macro_code], edgecolor='gray', linewidth=1,
                zorder=2, alpha=0.55
            )
            child_text = rcdc_label if len(str(rcdc_label)) <= 35 else str(rcdc_label)[:32] + '...'
            ax.text(x_rcdc + 0.2, child_y, f"{rcdc_id}: {child_text} ({rcdc_count})", ha='left', va='center', fontsize=7)

        y_current += (n_children * child_y_spacing) + y_spacing

    ax.set_xlim(-2, 15)
    ax.set_ylim(-1, y_current)
    ax.set_aspect('auto')
    ax.axis('off')
    ax.text(x_macro, -0.5, 'Macro Cluster', ha='center', va='top', fontsize=9, style='italic')
    ax.text(x_rcdc, -0.5, 'RCDC Category', ha='center', va='top', fontsize=9, style='italic')
    plt.tight_layout()

    if savefile:
        plt.savefig(savefile, dpi=300, bbox_inches='tight')
    plt.show()
    return fig, ax


def plot_rcdc_macro_heatmap(
    df_patent: pd.DataFrame,
    rcdc_context: Dict[str, Any],
    category_col: str = 'category_rcdc',
    top_n_categories: int = 5,
    normalize: bool = True,
    figsize: Tuple[int, int] = (13, 8),
    savefile: Optional[str] = None,
):
    """Plot the macro-cluster by RCDC-category heatmap used in the original notebook."""
    import ast
    import textwrap
    from matplotlib.colors import LinearSegmentedColormap
    import seaborn as sns

    cat_dict = rcdc_context['cat_dict']
    macro_cluster_names = rcdc_context['macro_cluster_names']
    top_topic_dict = rcdc_context['top_topic_dict']

    top_topic_dict_str = {
        str(macro_code): {str(topic_id) for topic_id in topic_ids}
        for macro_code, topic_ids in top_topic_dict.items()
    }

    macro_topic_counts = {str(macro_code): Counter() for macro_code in macro_cluster_names.keys()}

    for _, row in df_patent.iterrows():
        raw_topics = row.get(category_col, [])
        if isinstance(raw_topics, str):
            try:
                raw_topics = ast.literal_eval(raw_topics)
            except Exception:
                raw_topics = []

        if not isinstance(raw_topics, list):
            continue

        tot_topics = len(raw_topics) if raw_topics else 1
        for topic in raw_topics:
            if not isinstance(topic, dict):
                continue

            topic_id = str(topic.get('id'))
            if topic_id is None:
                continue

            for macro_code, allowed_topic_ids in top_topic_dict_str.items():
                if topic_id in allowed_topic_ids:
                    macro_topic_counts[macro_code][topic_id] += 1 if not normalize else 1 / tot_topics
                    break

    heatmap_data = []
    for macro_code, macro_label in macro_cluster_names.items():
        macro_code = str(macro_code)
        top_topics = macro_topic_counts[macro_code].most_common(top_n_categories)
        heatmap_data.append((macro_label, [(cat_dict.get(topic_id, topic_id), count) for topic_id, count in top_topics]))

    rcdc_labels = []
    for _, topics in heatmap_data:
        for topic_label, _ in topics:
            if topic_label not in rcdc_labels:
                rcdc_labels.append(topic_label)

    heatmap_matrix = np.zeros((len(heatmap_data), len(rcdc_labels)))
    for i, (_, topics) in enumerate(heatmap_data):
        for topic_label, count in topics:
            j = rcdc_labels.index(topic_label)
            heatmap_matrix[i, j] = count

    heatmap_df = pd.DataFrame(heatmap_matrix, index=[m for m, _ in heatmap_data], columns=rcdc_labels).T
    col_order = heatmap_df.columns.tolist()
    row_category = heatmap_df.idxmax(axis=1)
    row_value = heatmap_df.max(axis=1)
    sorted_df = (
        heatmap_df.assign(
            _category=row_category,
            _value=row_value,
            _category_order=row_category.map({col: i for i, col in enumerate(col_order)})
        )
        .sort_values(by=['_category_order', '_value'], ascending=[True, False])
        .drop(columns=['_category', '_value', '_category_order'])
        .replace(0, np.nan)
    )

    macro_topic_totals = {label: int(sum(macro_topic_counts[str(code)].values())) for code, label in macro_cluster_names.items()}
    #cmap = LinearSegmentedColormap.from_list('rcdc_macro_cmap', ['#345995', '#B80C09'])
    cmap="YlGnBu"
    fig, ax = plt.subplots(figsize=figsize)
    dis_df = sorted_df.copy()
    if normalize:
        for col in dis_df.columns:
            col_sum = macro_topic_totals.get(col, 0)
            if col_sum > 0:
                dis_df[col] = dis_df[col] / col_sum

    sns.heatmap(
        dis_df,
        annot=True,
        fmt='.1%',
        cmap=cmap,
        cbar_kws={'label': "Fraction of Macro Cluster's Topics" if normalize else 'Patent Count'},
        annot_kws={'size': 8},
        ax=ax,
    )

    for i in range(5, len(dis_df), 5):
        ax.axhline(i, color='grey', linewidth=1, linestyle='--')

    wrapped_labels = [textwrap.fill(label, width=14) for label in dis_df.columns]
    ax.set_xticklabels(wrapped_labels, rotation=0, ha='center', fontsize=9)
    ax.set_title('Top 5 RCDC Categories per Macro Cluster', fontsize=12, fontweight='bold', pad=20)
    ax.set_ylabel('RCDC Category', fontweight='bold')
    ax.set_xlabel('Macro Cluster', fontweight='bold')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)
    plt.tight_layout()

    if savefile:
        plt.savefig(savefile, format='svg')
    plt.show()
    return fig, ax, sorted_df
