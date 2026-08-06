# Deduplication pipeline: staged A–E implementation


## Folder structure

```text
.
├── README.md
├── requirements.txt
└── src/
    ├── run_dedup_pipeline.py
    └── dedup_pipeline/
        ├── __init__.py
        ├── cli.py
        ├── config.py
        ├── pipeline.py
        ├── stage_a.py
        ├── stage_b.py
        ├── stage_c.py
        ├── stage_d.py
        ├── stage_e.py
        └── union_find.py
```

### What each source file does

| File | Purpose |
|---|---|
| `src/run_dedup_pipeline.py` | Simple direct-run entry point. Use this if you want to run the whole pipeline from the folder root. |
| `src/dedup_pipeline/cli.py` | Defines command-line arguments. |
| `src/dedup_pipeline/pipeline.py` | Orchestrates Stage A → Stage B → Stage C → Stage D → Stage E. |
| `src/dedup_pipeline/config.py` | Stores ranked columns, empty output schemas, and scoring weight construction. |
| `src/dedup_pipeline/union_find.py` | Disjoint-set / union-find implementation used by Stage A. |
| `src/dedup_pipeline/stage_a.py` | Builds duplicate clusters from step-5 matching parquet files. |
| `src/dedup_pipeline/stage_b.py` | Groups duplicate ids by raw shard so each raw file is touched once later. |
| `src/dedup_pipeline/stage_c.py` | Scores candidate duplicate rows and selects one winner per cluster. |
| `src/dedup_pipeline/stage_d.py` | Builds field-wise winner enrichment patches and loser deletion list. |
| `src/dedup_pipeline/stage_e.py` | Applies deletes and patches to raw shards using atomic replacement. |

---

## Input

1. Raw datasets (`0000–4999`), stored as parquet shards.
2. Lots of parquet files containing duplicated data based on distinct matching criteria, for example:
   - title and author match
   - `pmid + doi + reference_id` match
   - Jaccard match by title
   - Jaccard match by abstract
   - other column-combination duplicate files from the earlier step-5 process

The step-5 parquet files are expected to contain at least:

```text
id, file, <one or more matching-key columns>
```

The `file` column should be the raw shard stem, for example:

```text
000000000457
```

The raw file path is then constructed as:

```text
<raw-base-dir>/<file><ext>
```

For the usual setting:

```text
<raw-base-dir>/000000000457.parquet
```

---

## The problem

Deduplicate input-1 using input-2, choose the most informative record per duplicate group, merge missing fields into it, and update the original files safely.

More concretely, we need to:

1. Merge duplicate evidence from many matching criteria.
2. Build final duplicate clusters across raw shards.
3. Pick one canonical record per cluster.
4. Use loser records to fill missing fields in the winner.
5. Delete loser records from the original raw shards.
6. Write the changes safely so the raw dataset is not left half-written if something fails during output.

---

## Why not do it straightforwardly

If we use the “normal” method — pair every item inside each group and, per group, open the winner/loser files to read, patch, and delete — we get:

```text
quadratic link creation through all input-2:  ∑ n²
```

and:

```text
O(G · k) full-file rewrites
```

where:

- `G` is the number of duplicate groups
- `k` is the number of raw files touched per group

This becomes expensive because the same raw shard may be opened and rewritten many times across different duplicate groups.

---

## My strategy

We read each input-2 file once, read each input-1 file at most twice, write each touched raw file once, and achieve approximately linear time complexity.

The key idea is to separate the problem into staged data products:

1. First build global duplicate clusters.
2. Then group all needed raw ids by shard.
3. Then score candidate rows in shard batches.
4. Then build compact merge/delete decisions.
5. Finally update each affected raw shard once.

This avoids repeated per-cluster raw-file reads and repeated per-cluster raw-file rewrites.

---

# Stage A: Merge all input-2 data with the disjoint-set algorithm

## Problem

Given many input-2 files, each listing publications that match on a specific set of keys, we must merge all overlapping matches into final duplicate clusters across files.

For example, one file may say:

```text
A = B by title + author
```

Another file may say:

```text
B = C by doi + pmid
```

Even if `A` and `C` never appear together in the same row group, they should still end up in the same final duplicate cluster because they are connected through `B`.

## Problem formulation and method

Treat each publication occurrence as a node:

```text
(id, file)
```

Treat each “same as” relation from a matched duplicate group as an edge.

This turns the task into computing connected components of a sparse graph. Stage A applies the disjoint-set algorithm, also known as union-find, to merge connected nodes efficiently.

For each step-5 parquet file:

1. Infer the matching-key columns by excluding `id`, `file`, and `columns`.
2. Group rows by those matching keys.
3. Keep only groups with at least `min_group_size` records.
4. Union all records in the group.
5. After all step-5 files are processed, assign a `cluster_id` to each connected component.

## Complexity

```text
O(E × almost-constant(V)) ≈ linear
```

where:

- `V` is the number of publication occurrences `(id, file)`
- `E` is the number of duplicate links induced by the step-5 files

Union-find has an inverse-Ackermann factor, which is practically constant for real datasets.

## Output

```text
OUT/clusters_map.parquet
```

Columns:

| Column | Meaning |
|---|---|
| `id` | Publication id. |
| `file` | Raw shard stem where the publication occurs. |
| `cluster_id` | Final duplicate cluster id. |

Each row says: this `(id, file)` occurrence belongs to this final duplicate cluster.

---

# Stage B: Build per input-1 file fetch lists

## Problem

We must score only the rows in input-1 that belong to any duplicate cluster, without repeatedly opening files.

If duplicate clusters are processed independently, the same raw shard may be read repeatedly. Stage B prevents this by building a per-shard list of all ids needed later.

## Method

Group `clusters_map.parquet` by `file` to collect the unique set of `id`s needed from each raw shard.

For each shard:

```text
file -> [ids needed from this file]
```

The code also constructs the expected raw shard path:

```text
<raw-base-dir>/<file><ext>
```

If existence checking is enabled, Stage B separates found shards from missing shards.

## Complexity

```text
O(N)
```

where `N` is the number of clustered publication occurrences.

## Outputs

```text
OUT/stageB/wanted_ids_by_file.parquet
OUT/stageB/files_to_touch.parquet
OUT/stageB/missing_files.parquet
```

### `wanted_ids_by_file.parquet`

Columns:

| Column | Meaning |
|---|---|
| `file` | Raw shard stem. |
| `ids` | List of publication ids needed from this shard. |
| `n_ids` | Number of unique ids needed from this shard. |

This is the main routing file for Stage C.

### `files_to_touch.parquet`

Columns:

| Column | Meaning |
|---|---|
| `file` | Raw shard stem. |
| `path` | Full expected raw shard path. |
| `n_ids` | Number of duplicate-related ids in this shard. |

This records which raw shards actually exist and need to be touched.

### `missing_files.parquet`

Columns:

| Column | Meaning |
|---|---|
| `file` | Raw shard stem. |
| `path` | Expected raw shard path. |
| `n_ids` | Number of ids that would have been needed from this missing shard. |

This is useful for debugging path problems.

---

# Stage C: Scoring for the most informative one per cluster

## Problem

For each duplicate cluster, decide which occurrence to keep using a transparent “informativeness” score.

The winner should usually be the row with more useful fields present. This is especially important when duplicate records come from different sources or import stages and one row is more complete than another.

## Method

Stage C calculates a presence-only score using a ranked list of important columns.

The ranked columns are stored in:

```text
src/dedup_pipeline/config.py
```

Earlier columns receive larger weights. A row receives the weight for a column if that column is present and non-null.

In simplified form:

```text
score(row) = sum(weight(column) for each non-null ranked column in row)
```

The optional argument:

```text
--zero-id-weight
```

sets the weight of `id` to zero. This is useful because every row should normally have an `id`, so counting it may not help distinguish informativeness.

Stage C reads raw shards according to `wanted_ids_by_file.parquet`, meaning it does not scan every raw row unnecessarily. For each shard, it writes a scored part file containing both:

1. the score used for winner selection
2. payload columns needed later for patch construction

Then it concatenates the scored rows and chooses the winner per `cluster_id` by sorting:

```text
cluster_id ascending
_score descending
file ascending
id ascending
```

This gives deterministic tie-breaking.

## Complexity

```text
O(N)
```

where `N` is the number of duplicate-related raw rows that must be scored.

## Outputs

```text
OUT/stageC/keep_decisions.parquet
OUT/stageC/scores_parts/scores_<file>.parquet
```

### `keep_decisions.parquet`

Columns:

| Column | Meaning |
|---|---|
| `cluster_id` | Duplicate cluster id. |
| `canonical_id` | Publication id selected as the winner. |
| `canonical_file` | Raw shard stem containing the winner. |
| `best_score` | Informativeness score of the winner. |
| `total_members` | Number of scored members in this cluster. |

This file only answers: where is the winner?

### `scores_parts/scores_<file>.parquet`

Each touched raw shard gets one score part file.

Typical columns:

| Column | Meaning |
|---|---|
| `id` | Publication id. |
| `file` | Raw shard stem. |
| ranked payload columns | Useful fields that may be merged into the winner later. |
| `_score` | Presence-only informativeness score. |
| `cluster_id` | Duplicate cluster id. |

These files are kept because Stage D needs both winner and loser payload values to build merge patches.

---

# Stage D: Build merge patches for winners

## Problem

The winner may still be missing useful fields held by losers. We need a compact, deterministic patch to enrich the winner before deleting the losers.

For example, the selected winner may have the better title, abstract, and Digital Object Identifier (DOI), but a loser may contain a useful `pmcid`, `funding_details`, or extra list entries. We do not want to keep the loser row, but we may want to preserve its useful information.

## Method

For each `cluster_id`, Stage D uses:

```text
OUT/stageC/keep_decisions.parquet
OUT/stageC/scores_parts/scores_*.parquet
```

It identifies:

1. the winner row
2. all loser rows
3. fields that can fill or enrich the winner

The patching rule is:

### Scalar fields

For scalar fields, only fill the winner if the winner value is missing and a loser has a non-null value.

Example:

```text
winner.pmcid = null
loser.pmcid  = PMC123
patch.pmcid  = PMC123
```

### List-like fields

For list-like fields, merge the winner list with loser lists using an ordered union, with an optional cap controlled by:

```text
--list-cap
```

Default:

```text
--list-cap 200
```

Use this to prevent extremely large list-valued fields from growing without bound. Setting `--list-cap 0` means no cap.

### Provenance

Stage D records which loser supplied each patched field. This is useful for debugging and auditability.

## Complexity

```text
O(N)
```

where `N` is the number of scored duplicate-related rows.

## Outputs

```text
OUT/stageD/merge_patches.parquet
OUT/stageD/loser_list.parquet
```

### `merge_patches.parquet`

Columns:

| Column | Meaning |
|---|---|
| `cluster_id` | Duplicate cluster id. |
| `canonical_id` | Winner publication id. |
| `canonical_file` | Raw shard stem containing the winner. |
| `patch_json` | JSON dictionary of fields to apply to the winner. |
| `columns_filled_json` | JSON provenance map showing which loser filled which column. |
| `merged_from_json` | JSON list of loser records that contributed information. |

This is compact: it stores only the fields that need to be patched, not full rows.

### `loser_list.parquet`

Columns:

| Column | Meaning |
|---|---|
| `cluster_id` | Duplicate cluster id. |
| `id` | Loser publication id to delete. |
| `file` | Raw shard stem containing the loser. |

This is the delete list for Stage E.

---

# Stage E: Update input-1 files

## Problem

Apply deletes and patches to input-1 efficiently and safely on shared storage.

The raw shards are the final data files, so we want to avoid partial writes and repeated rewrites.

## Method

Stage E uses:

```text
OUT/stageD/loser_list.parquet
OUT/stageD/merge_patches.parquet
```

It determines all affected raw shards from:

1. files containing losers
2. files containing winners that need patches

For each affected raw shard:

1. Read the raw shard.
2. Remove loser rows.
3. Join patch rows by `id`.
4. Fill winner fields from patch columns using coalescing.
5. Write to a temporary file:

```text
<raw-shard>.tmp
```

6. Atomically replace the original shard with the temporary file.
7. Write a changelog file for that shard.

This means each affected raw shard is rewritten once.

The argument:

```text
--apply-only-file 000000000457
```

can be used to test Stage E on one shard only.

## Complexity

```text
one read + one write per affected input-1 file
```

or, roughly:

```text
2N over touched raw shard data
```

where `N` is the size of affected raw shards.

## Outputs

Stage E modifies the raw input files in place using atomic replacement.

It also writes logs under:

```text
OUT/stageE/
```

### `stageE/changelogs/<file>.parquet`

One changelog is written per processed shard.

Columns:

| Column | Meaning |
|---|---|
| `file` | Raw shard stem. |
| `path` | Full raw shard path. |
| `num_rows_before` | Number of rows before delete/patch. |
| `num_deleted_present` | Number of loser ids actually found and deleted from this shard. |
| `num_patched_present` | Number of winner ids found in the shard after patching. |
| `num_rows_after` | Number of rows after delete/patch. |

### `stageE/progress_summary.txt`

Human-readable latest progress summary.

### `stageE/progress_summary.parquet`

Machine-readable latest progress summary.

### `stageE/progress_history.tsv`

Append-only history of progress snapshots. It is updated every 10 processed shards, at the end, or after roughly 10 minutes.

---

# Complete output layout

After running the pipeline, the output folder should look like this:

```text
OUT/
├── clusters_map.parquet
├── stageB/
│   ├── wanted_ids_by_file.parquet
│   ├── files_to_touch.parquet
│   └── missing_files.parquet
├── stageC/
│   ├── keep_decisions.parquet
│   └── scores_parts/
│       ├── scores_000000000001.parquet
│       ├── scores_000000000002.parquet
│       └── ...
├── stageD/
│   ├── merge_patches.parquet
│   └── loser_list.parquet
└── stageE/
    ├── changelogs/
    │   ├── 000000000001.parquet
    │   ├── 000000000002.parquet
    │   └── ...
    ├── progress_summary.txt
    ├── progress_summary.parquet
    └── progress_history.tsv
```

---

# How to run

From the folder root:

```bash
python src/run_dedup_pipeline.py \
  --step5-dir /well/mills/projects/scientometric/cache/columndeduplication \
  --raw-base-dir /well/mills/projects/scientometric/data_copy \
  --out-dir /well/mills/projects/scientometric/data_copy_cache/dedup_run \
  --min-group-size 2 \
  --ext .parquet \
  --zero-id-weight \
  --list-cap 200
```

Alternative module-style run from inside `src/`:

```bash
cd src
python -m dedup_pipeline.cli \
  --step5-dir /well/mills/projects/scientometric/cache/columndeduplication \
  --raw-base-dir /well/mills/projects/scientometric/data_copy \
  --out-dir /well/mills/projects/scientometric/data_copy_cache/dedup_run \
  --min-group-size 2 \
  --ext .parquet \
  --zero-id-weight \
  --list-cap 200
```

---

# Useful inspection commands

## Check the output folder

```bash
find /well/mills/projects/scientometric/data_copy_cache/dedup_run -maxdepth 3 -type f | head -50
```

## Read Stage A cluster map

```bash
python - <<'PY'
import polars as pl
p = "/well/mills/projects/scientometric/data_copy_cache/dedup_run/clusters_map.parquet"
df = pl.read_parquet(p)
print(df.head())
print("rows:", df.height)
print("clusters:", df["cluster_id"].n_unique())
PY
```

## Check missing raw files from Stage B

```bash
python - <<'PY'
import polars as pl
p = "/well/mills/projects/scientometric/data_copy_cache/dedup_run/stageB/missing_files.parquet"
df = pl.read_parquet(p)
print(df.head(20))
print("missing rows:", df.height)
PY
```

## Check winner decisions from Stage C

```bash
python - <<'PY'
import polars as pl
p = "/well/mills/projects/scientometric/data_copy_cache/dedup_run/stageC/keep_decisions.parquet"
df = pl.read_parquet(p)
print(df.head())
print("winner decisions:", df.height)
PY
```

## Check number of losers from Stage D

```bash
python - <<'PY'
import polars as pl
p = "/well/mills/projects/scientometric/data_copy_cache/dedup_run/stageD/loser_list.parquet"
df = pl.read_parquet(p)
print(df.head())
print("losers:", df.height)
PY
```

## Check patches from Stage D

```bash
python - <<'PY'
import polars as pl
p = "/well/mills/projects/scientometric/data_copy_cache/dedup_run/stageD/merge_patches.parquet"
df = pl.read_parquet(p)
print(df.head())
print("patch rows:", df.height)
PY
```

## Check latest Stage E progress

```bash
cat /well/mills/projects/scientometric/data_copy_cache/dedup_run/stageE/progress_summary.txt
```

## Watch a Slurm log

```bash
tail -f /path/to/your/slurm-output.log
```

---

# Important safety notes

Stage E updates the raw parquet shards in place.

Before running Stage E on the real raw directory, it is safer to do one of the following:

1. Run the whole pipeline on a copied subset of raw shards.
2. Run with `--apply-only-file <file_stem>` first.
3. Make sure the raw directory is backed up or recoverable.

Example single-shard Stage E test through the full pipeline:

```bash
python src/run_dedup_pipeline.py \
  --step5-dir /well/mills/projects/scientometric/cache/columndeduplication \
  --raw-base-dir /well/mills/projects/scientometric/data_copy \
  --out-dir /well/mills/projects/scientometric/data_copy_cache/dedup_run_test_one \
  --min-group-size 2 \
  --ext .parquet \
  --zero-id-weight \
  --list-cap 200 \
  --apply-only-file 000000000457
```

This still runs Stages A–D normally, but Stage E only applies changes to one affected raw shard.


---

# Command-line arguments

| Argument | Required? | Default | Meaning |
|---|---:|---:|---|
| `--step5-dir` | yes | none | Folder containing step-5 duplicate parquet files. |
| `--raw-base-dir` | yes | none | Folder containing raw parquet shards. |
| `--out-dir` | yes | none | Folder where pipeline outputs are written. |
| `--min-group-size` | no | `2` | Minimum matched group size to treat as duplicates. |
| `--ext` | no | `.parquet` | Raw shard filename extension. |
| `--no-exists-check` | no | off | Skip Stage B raw-file existence checking. |
| `--zero-id-weight` | no | off | Make `id` contribute zero to the informativeness score. |
| `--list-cap` | no | `200` | Maximum length for merged list-like fields; `0` means no cap. |
| `--apply-only-file` | no | none | In Stage E, process only one shard stem. Useful for testing. |

---

# Short version of the algorithm

```text
Input-2 duplicate files
        │
        ▼
Stage A: union-find over (id, file) nodes
        │
        ▼
clusters_map.parquet
        │
        ▼
Stage B: group ids by raw shard
        │
        ▼
wanted_ids_by_file.parquet
        │
        ▼
Stage C: read wanted rows, score, choose winners
        │
        ▼
keep_decisions.parquet + scores_parts/*.parquet
        │
        ▼
Stage D: build winner patches + loser list
        │
        ▼
merge_patches.parquet + loser_list.parquet
        │
        ▼
Stage E: update each affected raw shard once
        │
        ▼
Deduplicated raw shards + changelogs
```
