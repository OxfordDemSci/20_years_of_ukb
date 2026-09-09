# 20 years of UK Biobank

A place to develop analysis and infrastructure related to the history of the UK Biobank —
what its publication corpus contains, how it grew, what it is about, how much of the world's
literature it represents, and what it has led to outside academia.

## Setup

```bash
mamba activate ukbb20        # miniforge3; there is no environment.yml yet
```

The corpus is `data/showcase/showcase+/showcase_plus_all_endpoint.parquet` (26,109
publications, 2013–2026). **`data/`, `output/`, `fig/`, `logs/` and `doc/` are gitignored**,
so a fresh clone has code and settings only.

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

`04_non_academic_04_collaboration.ipynb` tags collaborators with an LLM before analysing
them. That step needs `pip install anthropic` and an `ANTHROPIC_API_KEY` (or `ant auth
login`), and it is **skipped whenever its output file already exists** — so re-running the
analysis costs nothing.

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
