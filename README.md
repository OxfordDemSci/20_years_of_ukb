# 20 years of UK Biobank

A place to develop analysis and infrastructure related to the history of the UK Biobank —
what its publication corpus contains, how it grew, what it is about, how much of the world's
literature it represents, and what it has led to outside academia.

## Setup

```bash
mamba create -n ukbb20 python=3.12 pip  # once, if the environment does not exist
mamba activate ukbb20
python -m pip install -r requirements-analysis.txt
```

`requirements-analysis.txt` covers the active notebooks, shared analysis helpers and
notebook runner. It preserves the fixed name-inference datasets and patent Louvain
implementation. Use a dedicated environment; it does not include unrelated packages
from a general-purpose base environment. Data-creation pipelines retain their own
requirements files under `src/data_creation/`.

Optional dependencies are installed only for the corresponding features:

- `python -m pip install bitsandbytes` for CUDA 4-bit validation models.
- `python -m pip install anthropic` for explicitly enabled paid collaboration
  classification; analysis of saved labels does not require it.
- `python -m pip install leidenalg` for the optional patent-clustering comparison.
- `python -m pip install xlrd` if loading legacy `.xls` collaboration labels;
  `.xlsx` support is included through `openpyxl`.

Helvetica regular and bold must be installed as system fonts; pip cannot supply
them. Model checkpoints and the research data are also separate from Python packages.

The corpus is `data/showcase/showcase+/showcase_plus_all_endpoint.parquet` (26,109
publications, 2013–2026). **`data/`, `output/`, `logs/` and `doc/` are gitignored**,
so a fresh clone has code and settings only.

Figures are saved under `output/figures/`, including patent figures in
`output/figures/data_analysis/04_non_academic/patent/`. All notebook PNG exports use
500 dpi, enforced by `PNG_DPI` in `src/utils/shared_style.py`; inline display resolution
is configured separately. All figures use Helvetica; titles are uppercase, bold and
left-aligned through the shared typography helpers. Rerun a notebook to regenerate
existing figures with these settings.

The author-characteristics notebook exports the top 30 authors by individual UK Biobank
h-index to `output/tables/data_analysis/05_author_characteristics/` as
`top_30_authors_by_ukb_h_index.csv` and an editable `.docx` table. The author-indexed
ranking uses 2013–2025 publications and runs immediately after loading the cohort,
before the network and field-normalised citation analyses. Word export uses `python-docx`,
included in `requirements-analysis.txt`.

The consolidated `src/data_analysis/02_content.ipynb` combines BERTopic and category
composition. It exports one main figure (FOR L4 and RCDC above the topic waves), two
classification supplements, a third topic-robustness supplement when verified diagnostics
are available, source tables, captions and an artifact manifest. The previous
two content notebooks are preserved under `_archived/` and do not run in the batch.

Completed topic-result CSVs in `output/bertopic/` or registered legacy locations are
reused before importing any modelling libraries. Matching `.analysis_window.json`
provenance must verify 2013–2025 training; invalid caches stop without starting a refit.
If results are missing, the five-configuration, five-seed robustness analysis runs once;
embeddings and successful parameter/seed runs are cached independently. Set
`RUN_TOPIC_MODELLING = False` to inspect the classification analyses without fitting,
or `FORCE_TOPIC_REFIT = True` to deliberately refit. Plotting cells only read results.
The default grid runs serially. On a machine with enough CPU and memory, optionally set
`UKB_TOPIC_WORKERS=2` (or call `ensure_topic_results(workers=2)`) to use two independent
fit processes. Coherence scoring and checkpoint writes remain in the coordinator;
completed seeds are reused with the same corpus/configuration keys in either mode.

Never hardcode a path. Every notebook opens with the same bootstrap header and takes its
paths from the registry:

```python
import sys
from pathlib import Path

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "utils").is_dir())
sys.path.insert(0, str(ROOT / "src"))
from utils import shared_paths as P
P.bootstrap()
```

Load the corpus through `utils.shared_showcase.load_showcase()`, never `pd.read_parquet`
directly — the nested columns are JSON and hand-rolled parsers fail *silently* against them.

## The analyses

| slug | what it asks | docs |
|---|---|---|
| `00_dataset` | is the corpus clean and complete? | — |
| `01_growth` | how did publication volume grow? | — |
| `02_content` | what is the research about? | [methodology](doc/02_content_methodology.md) · [report](doc/02_content_report.md) |
| `03_academic_impact` | how much of the world's work in a field is UK Biobank's, and does it land better? | [methodology](doc/03_academic_impact_methodology.md) · [report](doc/03_academic_impact_report.md) |
| `04_non_academic` | what has it led to — trials, patents, policy, attention, collaboration? | [methodology](doc/04_non_academic_methodology.md) · [report](doc/04_non_academic_report.md) |
| `05_authors` | who writes it, where, and with whom? | [report](doc/05_authors_report.md) *(partly superseded)* |

Notebooks run in slug order; within a slug, in numeric order. Several are currently blocked
or broken — [`doc/STATE.md`](doc/STATE.md) lists exactly which, and why.

All analyses use **1 January 2013–31 December 2025**, inclusive, defined in
`src/utils/shared_analysis_window.py`. Publication and linked-event inputs are filtered
before metrics, rankings, model fitting, and plotting. A year or date outside this
window excludes a record, and records with no usable date or
year are excluded. Raw source files remain intact, so source-inventory audits can still
report all 26,109 records, including the excluded 2026 publications.

`load_showcase()` reads the raw source. Analysis code must apply
`filter_analysis_window()` before using it. BERTopic caches are tied to the eligible
training corpus; old topic exports require retraining with the full 2013–2025 window and a matching
`.analysis_window.json` provenance file before reuse.

The publication/event cutoff is distinct from measurement time: citation totals retain
the **26 August 2026 snapshot**, and Altmetric totals retain their available source
snapshot. These sources do not provide the histories needed to reconstruct those
totals as of December 2025.

`04_non_academic_04_collaboration.ipynb` analyses saved organisation classifications.
It can rebuild the labelled table from a complete cache without credentials. Missing
classifications are reported as missing data; paid classification requires explicitly
setting `UKB_ALLOW_CLASSIFICATION=1` and configuring Anthropic separately.

### Run all analysis notebooks

With your analysis environment active:

```bash
bash run_analysis_notebooks.sh --list   # preview filename order
bash run_analysis_notebooks.sh          # run sequentially, continuing after failures
```

Alternatively, select its interpreter with `PYTHON=/path/to/env/bin/python bash
run_analysis_notebooks.sh`. The runner requires `nbclient`, `nbformat` and `ipykernel`
in addition to the analysis dependencies. It uses that interpreter for every kernel.

Each notebook stops at its first error. Terminal output is one result line per
notebook, with a brief error when needed, followed by totals. Error messages appear in
`output/notebook_runs/<run>/summary.csv`, alongside compact logs identifying the
failing cell and executed notebook copies. Tracebacks are omitted, and notebook
outputs are kept in the copies rather than repeated in logs. Original notebooks are
preserved; their code still writes the usual analysis outputs. The script exits nonzero if any notebook fails.
Use `--from 03_academic_impact_01_for_analysis.ipynb` to start later in the list,
`--only 04_non_academic_02_patents 04_non_academic_03_altmetric` to run selected
notebooks in their usual order, or `--timeout 3600` to limit each cell to one hour
(the default is unlimited).

The consolidated `00_dataset.ipynb` runs saved three-model agreement analyses and
labelled validation in separate sections. The two previous `00_` notebooks are
preserved in `_archived/` and excluded from the batch. Run the combined notebook with
`bash run_analysis_notebooks.sh --only 00_dataset`.

Set `UKB_COMBINED_LABELS_CSV` to the local full candidate-level three-model combined
labels CSV. A uniquely named
`three_model_combined_labels.csv` (or its original long filename) under `data/`
is detected automatically. Showcase+ is not a substitute for this input. Matching
semantic coordinates/metrics are reused; new semantic encoding requires
`RUN_SEMANTIC_ANALYSIS = True` (or `UKB_RUN_SEMANTIC_ANALYSIS=1`).

Validation reuses all six `predictions_*.csv` tables under `output/validation/`
(overridable with `UKB_VALIDATION_OUTPUT_DIR`) after checking dates, IDs and labels.
Generating predictions requires `RUN_VALIDATION_INFERENCE = True` (or
`UKB_RUN_VALIDATION_INFERENCE=1`), `UKB_VALIDATION_POSITIVE_CSV`,
`UKB_VALIDATION_NEGATIVE_CSV`, and explicit `UKB_VALIDATION_N_POS` /
`UKB_VALIDATION_N_NEG` evaluation sample sizes. The named labelled files under
`data/validation/` are the default sources. Inputs are checked before model imports
or downloads; there is no interactive login or automatic package installation.
The historical pre-2014 negative filename does not bypass the shared 2013–2025
window. Encoder baselines retain their original in-sample threshold calibration.

Missing data/results are reported as skipped sections; malformed inputs fail after
both sections have been attempted. The status table is saved to
`output/tables/data_analysis/00_dataset/00_dataset_status.csv`. Agreement tables
use the `three_model_agreement/` subdirectory; validation tables remain under
`output/validation/`. Figures are collected under
`output/figures/data_analysis/00_dataset/` as 500-DPI PNGs and PDFs.
The publication figure set combines candidate agreement and annual trends (`00_01`),
mention/keyword profiles and TF-IDF contrasts (`00_02`), validation performance
(`00_03`), and agreement across six prompts (`00_04`). Optional cached semantic
maps are combined in `00_05`. Each has a manuscript caption in a matching
`_caption.txt`; detailed source tables are preserved. Unavailable inputs skip the
corresponding figure rather than generate placeholder results. The notebook lists
which CSVs underlie each figure. No standalone per-prompt figures, SVGs or JSON
figure sidecars are produced.

`02_content` uses the local Showcase+ parquet, writes topic results to
`output/bertopic/`, and caches embeddings under `data/analysis/content/cache/`.
Figures go to `output/figures/data_analysis/02_content/` and publication tables to
`output/tables/02_content/`. It does not install dependencies automatically.
Run it alone with `bash run_analysis_notebooks.sh --only 02_content`.
Academic field/citation analyses require the original
`data/analysis/academic_impact/for_counts_api/` cache. Set `UKB_FOR_COUNTS_DIR` if
it is stored elsewhere. This contains both UK Biobank and whole-database counts,
the per-paper snapshot (`api_ukbb_records.json`), field-year reference means
(`api_whole.for.parquet`) and thresholds (`field_thresholds.for.csv`). The
Showcase+ corpus alone cannot recreate the whole-database comparison.
The extra citation notebook uses local Showcase+ directly and writes to
`output/non_topic_figures/`.

When canonical non-academic inputs are absent, patents can use the complete legacy
export `data/patent/df_with_iso_aggressive.csv`; existing derived country/topic
columns are recalculated. Clinical-trial and policy CSVs can be recovered from the
complete fetched records embedded in Showcase+, with strict array-alignment and
duplicate checks. Each recovered CSV has a `.provenance.json` sidecar. Unfetched
linked IDs are excluded, so these are snapshot records, not an additional data pull.
The clinical MeSH section still requires an enriched `mesh_leaf_ids` column and
`mesh_tree_numbers.pkl`; it does not fetch this enrichment automatically.

## Documentation

`doc/` is local-only (not in git), so its dated correction blocks are the only history those
documents have. **Start here:**

- [`doc/STATE.md`](doc/STATE.md) — where the project is right now: active decisions, the
  ledger of established figures, and the open questions. One page; read it first.
- [`doc/conventions.md`](doc/conventions.md) — this project's deltas from the analyst's
  universal standard: the storage layout, the slugs, the deliberate departures.
- [`doc/data/`](doc/data/) — one file per source: provenance, dictionary, sensitivity tier,
  and traps.
- `doc/decisions_*.md` — the append-only decision log. Do not read it front to back;
  `STATE.md` exists so you do not have to.

**The ledger in `STATE.md` is authority.** A result that contradicts it is reporting a bug
until proven otherwise.

## Layout

```
src/data_creation/    acquisition, dedup, matching, LLM tagging
src/data_analysis/    the numbered analysis notebooks
src/utils/            shared_paths (registry), shared_style, shared_<domain>, per-analysis modules
config/               credentials — *.ini gitignored, *.ini.example committed
universal_settings.yml   the plotting style for the whole project
```
