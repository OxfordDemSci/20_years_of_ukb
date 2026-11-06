#!/bin/bash
# =============================================================================
# data_analysis_05_author_gender_corpus.sh — BMRC/Slurm driver for the gender-by-field job
# =============================================================================
#
# The script is both the submitter and the job body: every job it submits re-enters this
# same file with PHASE set. Site handling follows the aleatoric_luck NK_grid launcher
# (launch/README.md, launch/profiles/bmrc.sh): an explicit project account, `module purge;
# module load` in every job, dependency wheels built on the login node because BMRC
# compute nodes cannot reach PyPI, a commit-pinned checkout, and progress read from the
# filesystem rather than by polling squeue.
#
# USAGE (from a login node, in a clean checkout of the repository)
#   ./…gender_corpus.sh wheels         # 1. build the dependency wheels (login node, network)
#   ./…gender_corpus.sh setup-env      # 2. one small job: the offline venv on a compute node
#   ./…gender_corpus.sh filelist       # 3. enumerate the corpus
#   ./…gender_corpus.sh check-inputs   # 4. the three files copied from the laptop
#   ./…gender_corpus.sh schema         # 5. one job: schema groups + year coverage. READ IT.
#   ./…gender_corpus.sh local          # 6. (optional) the whole chain on a few files
#   ./…gender_corpus.sh submit         # 7. names -> classify -> identity -> count x2 -> merge
#   ./…gender_corpus.sh status         #    what is done, per unit
#   ./…gender_corpus.sh resume         #    resubmit only unfinished tasks, then the rest
#
# Required:  ACCOUNT=<project account> (never defaulted, as in NK_grid)
# Override anything by exporting it first, e.g.
#   ACCOUNT=mills.prj OUTDIR=/well/mills/users/$USER/gender_corpus ./…gender_corpus.sh submit
# DRY_RUN=1 prints every sbatch command instead of submitting it.
# =============================================================================

set -euo pipefail

# Slurm runs a COPY of this script from its spool directory, so a job cannot find the
# repository from BASH_SOURCE. The submitter exports GC_HERE; a job inherits it.
HERE="${GC_HERE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
REPO="$(cd "$HERE/../.." && pwd)"
SELF="$HERE/$(basename "${BASH_SOURCE[0]}")"
[[ -n "${GC_SELF:-}" ]] && SELF="$GC_SELF"
PY_SCRIPT="$HERE/data_analysis_05_author_gender_corpus.py"
REQUIREMENTS="$HERE/data_analysis_05_author_gender_corpus_requirements.txt"

# ------------------------------------------------------------- site (BMRC profile)
# Unset -> the BMRC module; set to "" for a machine without environment modules.
PYTHON_MODULE="${PYTHON_MODULE-Python/3.11.3-GCCcore-12.3.0}"
PARTITION="${PARTITION:-long}"
CONSTRAINT="${CONSTRAINT:-skl-compat}"      # 'none' sends no constraint
QOS="${QOS:-}"
ACCOUNT="${ACCOUNT:-}"

# ------------------------------------------------------------- data and outputs
CORPUS="${CORPUS:-/well/mills/projects/scientometric/data_copy}"
OUTDIR="${OUTDIR:-$HOME/gender_corpus_out}"   # on shared storage, never inside CORPUS
INPUTS="${INPUTS:-$OUTDIR/inputs}"           # ukbb_ids.txt, ukbb_for_codes.csv, for_2020_codes.txt
FILELIST="${FILELIST:-$OUTDIR/files.txt}"

# The environment lives beside the checkout, not in it and not in /tmp (too small on
# compute nodes), and is shared by every run that uses the same requirements.
ENV_ROOT="${ENV_ROOT:-$(dirname "$REPO")/gender-corpus-envs}"
WHEELHOUSE="${WHEELHOUSE:-$ENV_ROOT/wheels}"
VENV="${VENV:-$ENV_ROOT/py311}"

# ------------------------------------------------------------- design
FIRST_YEAR="${FIRST_YEAR:-2015}"
LAST_YEAR="${LAST_YEAR:-2025}"
NUM_SHARDS="${NUM_SHARDS:-200}"      # corpus-read array size; ~25 files per task
BUCKETS="${BUCKETS:-32}"             # classify / identity array size
ARRAY_LIMIT="${ARRAY_LIMIT:-50}"     # concurrent tasks per array: spares the shared filesystem
BG_LABEL="${BG_LABEL:-background}"
UKBB_LABEL="${UKBB_LABEL:-ukbb}"

# ------------------------------------------------------------- resources per phase
TIME_READ="${TIME_READ:-12:00:00}";  MEM_READ="${MEM_READ:-16G}"      # names, count
TIME_CLASSIFY="${TIME_CLASSIFY:-06:00:00}"; MEM_CLASSIFY="${MEM_CLASSIFY:-12G}"
TIME_IDENTITY="${TIME_IDENTITY:-04:00:00}"; MEM_IDENTITY="${MEM_IDENTITY:-24G}"
TIME_MERGE="${TIME_MERGE:-02:00:00}"; MEM_MERGE="${MEM_MERGE:-16G}"
TIME_SCHEMA="${TIME_SCHEMA:-06:00:00}"; MEM_SCHEMA="${MEM_SCHEMA:-8G}"
DRY_RUN="${DRY_RUN:-0}"

RUN_JSON="$OUTDIR/run.json"

# =============================================================================
# helpers
# =============================================================================
die() { echo "!! $*" >&2; exit 2; }

load_python() {
  if [[ -n "$PYTHON_MODULE" ]]; then
    command -v module >/dev/null 2>&1 || die "module unavailable in this shell"
    module purge
    module load "$PYTHON_MODULE"
  fi
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
  export PYTHONUNBUFFERED=1
}

py() { "$VENV/bin/python" "$PY_SCRIPT" "$@"; }

commit() { git -C "$REPO" rev-parse HEAD; }

require_clean_checkout() {
  # Tracked files only: a run records its commit and every job checks it again, so the
  # code that counted is always the code at that commit.
  [[ -z "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]] \
    || die "uncommitted changes in $REPO — commit them, or develop in a separate clone"
}

# The run's identity: commit and design. A different one needs a new OUTDIR, because
# partials from two designs would merge into one table without complaint.
run_signature() {
  printf '{"commit": "%s", "first_year": %s, "last_year": %s, "num_shards": %s, "buckets": %s, "filelist": "%s", "codes": "%s", "ids": "%s"}\n' \
    "$(commit)" "$FIRST_YEAR" "$LAST_YEAR" "$NUM_SHARDS" "$BUCKETS" "$FILELIST" \
    "$INPUTS/ukbb_for_codes.csv" "$INPUTS/ukbb_ids.txt"
}

check_run_json() {
  local want; want="$(run_signature)"
  if [[ -s "$RUN_JSON" ]]; then
    [[ "$(cat "$RUN_JSON")" == "$want" ]] || die "$RUN_JSON records a different commit or design:
  recorded: $(cat "$RUN_JSON")
  now:      $want
Use a new OUTDIR (or check out the recorded commit)."
  else
    echo "$want" > "$RUN_JSON"
  fi
}

check_inputs() {
  for f in ukbb_ids.txt ukbb_for_codes.csv for_2020_codes.txt; do
    [[ -s "$INPUTS/$f" ]] || die "missing $INPUTS/$f — run make-inputs on the laptop and copy it here"
  done
  echo "ids    $(wc -l < "$INPUTS/ukbb_ids.txt") UK Biobank paper ids"
  echo "codes  $(($(wc -l < "$INPUTS/ukbb_for_codes.csv") - 1)) UK Biobank FoR codes (L2 + L4)"
  echo "vocab  $(wc -l < "$INPUTS/for_2020_codes.txt") ANZSRC 2020 codes"
}

check_schema() {
  local groups="$OUTDIR/schema/schema_groups.csv"
  [[ -s "$groups" ]] || die "no $groups — run 'schema' and read it before submitting"
  local errors
  errors=$("$VENV/bin/python" -c "import pandas as pd; print(int(pd.read_csv('$groups').n_errors.sum()))")
  if [[ "$errors" != 0 && "${ALLOW_SCHEMA_ERRORS:-0}" != 1 ]]; then
    die "$errors file(s) in $groups cannot be read by this job. Resolve them, or set
ALLOW_SCHEMA_ERRORS=1 to run without them (merge will then need ALLOW_FAILED=1)."
  fi
}

preflight() {
  [[ -n "$ACCOUNT" ]] || die "ACCOUNT is required (the Slurm project account)"
  require_clean_checkout
  [[ -x "$VENV/bin/python" ]] || die "no environment at $VENV — run 'wheels' then 'setup-env'"
  [[ -s "$FILELIST" ]] || die "no $FILELIST — run 'filelist'"
  check_inputs >/dev/null
  check_schema
  mkdir -p "$OUTDIR/slurm"
  check_run_json
  # Travels to every job through --export=ALL.
  export GC_HERE="$HERE" GC_SELF="$SELF" GC_COMMIT="$(commit)"
  export PYTHON_MODULE CORPUS OUTDIR INPUTS FILELIST ENV_ROOT WHEELHOUSE VENV FIRST_YEAR \
         LAST_YEAR NUM_SHARDS BUCKETS BG_LABEL UKBB_LABEL
}

# submit NAME ARRAY DEPENDENCY TIME MEM [VAR=value ...] -> prints the job id
submit() {
  local name=$1 array=$2 dep=$3 time=$4 mem=$5; shift 5
  local out="$OUTDIR/slurm/%x.%j.out"
  [[ -n "$array" ]] && out="$OUTDIR/slurm/%x.%A_%a.out"
  local args=(--parsable --job-name="$name" --account="$ACCOUNT" --partition="$PARTITION"
              --nodes=1 --ntasks=1 --cpus-per-task=1 --time="$time" --mem="$mem"
              --output="$out" --error="$out")
  [[ -n "$CONSTRAINT" && "$CONSTRAINT" != none ]] && args+=(--constraint="$CONSTRAINT")
  [[ -n "$QOS" ]] && args+=(--qos="$QOS")
  [[ -n "$array" ]] && args+=(--array="${array}%${ARRAY_LIMIT}")
  [[ -n "$dep" ]] && args+=(--dependency="afterok:$dep")
  args+=(--export="ALL,$(IFS=,; echo "$*")")
  if [[ "$DRY_RUN" == 1 ]]; then
    echo "  sbatch ${args[*]} $SELF" >&2
    echo "DRY-$name"
  else
    sbatch "${args[@]}" "$SELF"
  fi
}

# Missing array indices of one unit (resume), or the whole range (submit).
indices() {
  local unit=$1 n=$2 resume=$3
  if [[ "$resume" == 1 ]]; then
    py status --out "$OUTDIR" --num-shards "$NUM_SHARDS" --buckets "$BUCKETS" \
      --arms "$BG_LABEL,$UKBB_LABEL" --missing-only "$unit"
  else
    echo "0-$((n - 1))"
  fi
}

# The whole chain. With resume=1 a unit submits only its unfinished tasks — and once a
# unit has run again, every unit after it runs in full, because its inputs changed.
chain() {
  local resume=$1 dep="" ids forced=0 arm_jobs=() spec label filter
  ids=$(indices names "$NUM_SHARDS" "$resume")
  if [[ -n "$ids" ]]; then
    dep=$(submit gc-names "$ids" "" "$TIME_READ" "$MEM_READ" PHASE=names); forced=1
    echo "names    $dep  (tasks $ids)"
  fi
  for unit in classify identity; do
    [[ "$forced" == 1 ]] && ids="0-$((BUCKETS - 1))" || ids=$(indices "$unit" "$BUCKETS" "$resume")
    if [[ -n "$ids" ]]; then
      local t m; [[ $unit == classify ]] && { t=$TIME_CLASSIFY; m=$MEM_CLASSIFY; } || { t=$TIME_IDENTITY; m=$MEM_IDENTITY; }
      dep=$(submit "gc-$unit" "$ids" "$dep" "$t" "$m" PHASE="$unit"); forced=1
      echo "$unit $dep  (tasks $ids)"
    fi
  done
  for spec in "$BG_LABEL:exclude" "$UKBB_LABEL:include"; do
    label="${spec%%:*}"; filter="${spec##*:}"
    [[ "$forced" == 1 ]] && ids="0-$((NUM_SHARDS - 1))" || ids=$(indices "count.$label" "$NUM_SHARDS" "$resume")
    if [[ -n "$ids" ]]; then
      local jid; jid=$(submit "gc-count-$label" "$ids" "$dep" "$TIME_READ" "$MEM_READ" \
                       PHASE=count LABEL="$label" ID_FILTER="$filter")
      arm_jobs+=("$jid"); echo "count    $jid  arm $label (--id-filter $filter, tasks $ids)"
    fi
  done
  local after="$dep"
  if [[ ${#arm_jobs[@]} -gt 0 ]]; then after=$(IFS=:; echo "${arm_jobs[*]}"); fi
  local mid; mid=$(submit gc-merge "" "$after" "$TIME_MERGE" "$MEM_MERGE" PHASE=merge)
  echo "merge    $mid  (waits for ${after:-nothing})"
  echo "progress: $0 status   (reads done markers; no squeue polling needed)"
}

# =============================================================================
# job body: every submitted job re-enters here with PHASE set
# =============================================================================
if [[ -n "${PHASE:-}" && -n "${SLURM_JOB_ID:-}" ]]; then
  load_python
  echo "$PHASE task ${SLURM_ARRAY_TASK_ID:-single} on $(hostname) at $(date)"
  if [[ "$PHASE" != setup-env && -n "${GC_COMMIT:-}" && "$(commit)" != "$GC_COMMIT" ]]; then
    die "checkout moved from $GC_COMMIT to $(commit) after submission — refusing to compute"
  fi
  task="${SLURM_ARRAY_TASK_ID:-0}"
  allow=(); [[ "${ALLOW_FAILED:-0}" == 1 ]] && allow=(--allow-failed)
  case "$PHASE" in
    setup-env)
      python3 -m venv "$VENV"
      "$VENV/bin/pip" install --no-index --find-links "$WHEELHOUSE" -r "$REQUIREMENTS"
      "$VENV/bin/python" - "$HERE" <<'EOF'
import sys
sys.path.insert(0, sys.argv[1])
import pyarrow, pandas, shared_name_gender as NG
print("pyarrow", pyarrow.__version__, "| pandas", pandas.__version__)
print(NG.offline_library_versions().to_string(index=False))
for name in ("Anna", "John", "J."):
    r = NG.infer_name_gender(name)
    print(f"  {name!r:8} -> {r.category} ({r.method}{', ' + r.unknown_reason if r.unknown_reason else ''})")
EOF
      ;;
    schema)
      py schema --file-list "$FILELIST" --out "$OUTDIR/schema" \
        --for-vocab "$INPUTS/for_2020_codes.txt" ;;
    names)
      py names --file-list "$FILELIST" --shard "$task" --num-shards "$NUM_SHARDS" \
        --buckets "$BUCKETS" --first-year "$FIRST_YEAR" --last-year "$LAST_YEAR" --out "$OUTDIR" ;;
    classify|identity)
      py "$PHASE" --bucket "$task" --buckets "$BUCKETS" --num-shards "$NUM_SHARDS" \
        --out "$OUTDIR" ${allow[@]+"${allow[@]}"} ;;
    count)
      py count --file-list "$FILELIST" --shard "$task" --num-shards "$NUM_SHARDS" \
        --buckets "$BUCKETS" --first-year "$FIRST_YEAR" --last-year "$LAST_YEAR" \
        --ids "$INPUTS/ukbb_ids.txt" --id-filter "$ID_FILTER" --label "$LABEL" \
        --codes "$INPUTS/ukbb_for_codes.csv" --out "$OUTDIR" ;;
    merge)
      py merge --out "$OUTDIR" --num-shards "$NUM_SHARDS" --arms "$BG_LABEL,$UKBB_LABEL" ${allow[@]+"${allow[@]}"} ;;
    *) die "unknown PHASE '$PHASE'" ;;
  esac
  echo "done at $(date)"
  exit 0
fi

# =============================================================================
# subcommands
# =============================================================================
cmd="${1:-help}"
case "$cmd" in
  wheels)
    # Login node, with network: build every dependency into a wheel, including the two
    # name packages published only as source archives, with the compute nodes' Python.
    load_python
    mkdir -p "$WHEELHOUSE"
    python3 -m pip wheel -r "$REQUIREMENTS" -w "$WHEELHOUSE"
    echo "wheels -> $WHEELHOUSE"; ls "$WHEELHOUSE"
    ;;

  setup-env)
    [[ -n "$ACCOUNT" ]] || die "ACCOUNT is required"
    [[ -d "$WHEELHOUSE" ]] || die "no $WHEELHOUSE — run 'wheels' first"
    mkdir -p "$OUTDIR/slurm"
    export GC_HERE="$HERE" GC_SELF="$SELF" PYTHON_MODULE VENV WHEELHOUSE OUTDIR
    jid=$(submit gc-setup-env "" "" 01:00:00 8G PHASE=setup-env)
    echo "setup-env job $jid — log in $OUTDIR/slurm/gc-setup-env.$jid.out"
    ;;

  filelist)
    mkdir -p "$OUTDIR"
    # Sorted, so every array task derives the same shard from the same list.
    find -L "$CORPUS" -type f -name '*.parquet' | sort > "$FILELIST"
    echo "$(wc -l < "$FILELIST") parquet files -> $FILELIST"
    ;;

  check-inputs)
    check_inputs
    ;;

  schema)
    [[ -n "$ACCOUNT" ]] || die "ACCOUNT is required"
    [[ -s "$FILELIST" ]] || die "no $FILELIST — run 'filelist'"
    check_inputs >/dev/null
    mkdir -p "$OUTDIR/slurm" "$OUTDIR/schema"
    export GC_HERE="$HERE" GC_SELF="$SELF" PYTHON_MODULE VENV OUTDIR INPUTS FILELIST
    jid=$(submit gc-schema "" "" "$TIME_SCHEMA" "$MEM_SCHEMA" PHASE=schema)
    echo "schema job $jid — then read $OUTDIR/schema/schema_groups.csv and the job log"
    ;;

  schema-summary)
    py schema-summary --out "$OUTDIR/schema"
    ;;

  submit)
    preflight
    [[ -e "$OUTDIR/done" ]] && die "$OUTDIR already holds a run — use 'resume', or a new OUTDIR"
    chain 0
    ;;

  resume)
    preflight
    chain 1
    ;;

  status)
    py status --out "$OUTDIR" --num-shards "$NUM_SHARDS" --buckets "$BUCKETS" \
      --arms "$BG_LABEL,$UKBB_LABEL"
    [[ -s "$OUTDIR/gender_field_counts.parquet" ]] && echo "merged: $OUTDIR/gender_field_counts.parquet"
    ;;

  local)
    # No Slurm: the whole chain in this shell, one shard, for a handful of files — e.g.
    #   head -5 files.txt > few.txt; FILELIST=few.txt OUTDIR=… ./…gender_corpus.sh local
    # Run it inside an interactive job (srun --pty), not on the login node.
    load_python
    [[ -x "$VENV/bin/python" ]] || die "no environment at $VENV"
    yr=(--first-year "$FIRST_YEAR" --last-year "$LAST_YEAR")
    py names --file-list "$FILELIST" --buckets "$BUCKETS" "${yr[@]}" --out "$OUTDIR"
    py classify --buckets "$BUCKETS" --out "$OUTDIR"
    py identity --buckets "$BUCKETS" --out "$OUTDIR"
    for spec in "$BG_LABEL:exclude" "$UKBB_LABEL:include"; do
      py count --file-list "$FILELIST" --buckets "$BUCKETS" "${yr[@]}" \
        --ids "$INPUTS/ukbb_ids.txt" --id-filter "${spec##*:}" --label "${spec%%:*}" \
        --codes "$INPUTS/ukbb_for_codes.csv" --out "$OUTDIR"
    done
    py merge --out "$OUTDIR" --arms "$BG_LABEL,$UKBB_LABEL"
    ;;

  *)
    sed -n '2,28p' "${BASH_SOURCE[0]}"
    ;;
esac
