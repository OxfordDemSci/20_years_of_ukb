#!/bin/bash
# =============================================================================
# academic_impact_field_counts.sh — BMRC/Slurm driver for the FOR count job
# =============================================================================
#
# The script is both the submitter and the job body: `submit` sbatch's this same
# file as an array, and the array tasks re-enter it with SLURM_ARRAY_TASK_ID set.
#
# USAGE (from the login node)
#   ./academic_impact_field_counts.sh filelist   # 1. enumerate the corpus
#   ./academic_impact_field_counts.sh probe      # 2. measure before you size (DO THIS)
#   ./academic_impact_field_counts.sh submit     # 3. array + dependent merge
#   ./academic_impact_field_counts.sh local      # (no Slurm) whole corpus on one core
#
# Override anything by exporting it first, e.g.
#   CORPUS=/well/.../parquet NUM_SHARDS=20 ./academic_impact_field_counts.sh submit
#
# Pick the classification system with CATEGORY (default 'for'); e.g.
#   CATEGORY=rcdc ./...field_counts.sh local        # NIH disease/condition topics
# Partials are named per-category, so several categories can share one OUTDIR.
#
# READ THE PROBE OUTPUT BEFORE SUBMITTING
# ---------------------------------------
# We only read id/year/type + the one category column. On the sample file that is
# ~0.16 % of the bytes, so "230 GB" is not the workload — expect a few hundred MB of I/O and
# ~20 minutes on ONE core for the whole corpus. `probe` prints the actual percentage
# for the real files. If it comes back small, NUM_SHARDS=10 is already generous and
# NUM_SHARDS=1 would do; only raise it if probe shows the projection is expensive.
# Over-sharding a cheap job wastes queue time and makes 5,000 tiny files worse, not
# better, on a shared filesystem.
# =============================================================================

#SBATCH --job-name=for-counts
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=%x.%A_%a.out
#SBATCH --error=%x.%A_%a.err

set -euo pipefail

# ---------------------------------------------------------------- configuration
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${PY_SCRIPT:-$HERE/data_analysis_03_academic_impact_field_counts.py}"

# The corpus directory on the VM, and where to put results (NOT inside the corpus).
CORPUS="${CORPUS:-/well/CHANGE/ME/dimensions_parquet}"
OUTDIR="${OUTDIR:-$HOME/for_counts_out}"

# UKBB paper ids. Make this once with `make-ids` below.
IDS="${IDS:-$OUTDIR/ukbb_ids.txt}"

FILELIST="${FILELIST:-$OUTDIR/files.txt}"
NUM_SHARDS="${NUM_SHARDS:-10}"
LABEL="${LABEL:-background}"
ID_FILTER="${ID_FILTER:-exclude}"     # exclude = the non-UKBB background arm
CATEGORY="${CATEGORY:-for}"           # classification system: for|rcdc|uoa|hrcs_hc|...
                                      # partials are named per-category, so several
                                      # categories can share one OUTDIR and one merge.

# BMRC gives you Python through modules; a conda env works too. Set PYTHON to skip.
MODULE_LOAD="${MODULE_LOAD:-Python/3.11.3-GCCcore-12.3.0}"
PYTHON="${PYTHON:-}"

setup_python() {
  if [[ -z "$PYTHON" ]]; then
    if command -v module &>/dev/null && [[ -n "$MODULE_LOAD" ]]; then
      module load "$MODULE_LOAD" || true
    fi
    PYTHON="$(command -v python3 || command -v python)"
  fi
  echo "python: $PYTHON"
  "$PYTHON" -c 'import pyarrow; print("pyarrow", pyarrow.__version__)' \
    || { echo "pyarrow missing: pip install --user pyarrow" >&2; exit 1; }
}

mkdir -p "$OUTDIR"

# ------------------------------------------------------------ the array task body
# When Slurm runs this file as an array task, do the shard and exit. Placed before
# the CLI dispatch so the submitted script needs no arguments.
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  setup_python
  echo "shard ${SLURM_ARRAY_TASK_ID}/${NUM_SHARDS} on $(hostname) at $(date)"
  "$PYTHON" "$PY_SCRIPT" count --category "$CATEGORY" \
    --file-list "$FILELIST" \
    --shard "$SLURM_ARRAY_TASK_ID" --num-shards "$NUM_SHARDS" \
    --exclude-ids "$IDS" --id-filter "$ID_FILTER" \
    --label "$LABEL" --out "$OUTDIR"
  echo "done at $(date)"
  exit 0
fi

# ------------------------------------------------------------------- subcommands
cmd="${1:-help}"
case "$cmd" in

  filelist)
    # -L follows symlinked corpora (and still refuses to loop); sorted so every array
    # task derives the same shard assignment from the same list.
    find -L "$CORPUS" -type f -name '*.parquet' | sort > "$FILELIST"
    echo "$(wc -l < "$FILELIST") parquet files -> $FILELIST"
    ;;

  make-ids)
    # Extract the UKBB paper ids once, from the UKBB parquet, into a plain id list.
    setup_python
    src="${UKBB_PARQUET:?export UKBB_PARQUET=/path/to/showcase_plus_all_endpoint.parquet}"
    "$PYTHON" - "$src" "$IDS" <<'EOF'
import sys, pyarrow.parquet as pq
ids = pq.read_table(sys.argv[1], columns=["id"]).column(0).to_pylist()
ids = sorted({i for i in ids if i})
open(sys.argv[2], "w").write("\n".join(ids) + "\n")
print(f"{len(ids):,} ids -> {sys.argv[2]}")
EOF
    ;;

  probe)
    # Footer-only: touches ~64 KB per file, safe to run on the login node.
    setup_python
    "$PYTHON" "$PY_SCRIPT" probe --category "$CATEGORY" --file-list "$FILELIST" --out "$OUTDIR"
    echo
    echo "Look at 'projected read' above. If it is a small % of the corpus, this job is"
    echo "minutes of work — lower NUM_SHARDS rather than raising it."
    ;;

  submit)
    [[ -s "$FILELIST" ]] || { echo "no $FILELIST — run 'filelist' first" >&2; exit 1; }
    [[ -s "$IDS" ]] || { echo "no $IDS — run 'make-ids' first" >&2; exit 1; }
    last=$((NUM_SHARDS - 1))
    jid=$(sbatch --parsable --array=0-"$last" \
          --export=ALL,CORPUS="$CORPUS",OUTDIR="$OUTDIR",IDS="$IDS",\
FILELIST="$FILELIST",NUM_SHARDS="$NUM_SHARDS",LABEL="$LABEL",ID_FILTER="$ID_FILTER",\
CATEGORY="$CATEGORY" \
          "${BASH_SOURCE[0]}")
    echo "array job $jid (0-$last)"
    # afterok: the merge must not run on a partial set of shards, or the counts are
    # silently short. If a shard fails, fix it and rerun just that --array=N.
    mid=$(sbatch --parsable --dependency=afterok:"$jid" \
          --job-name=for-merge --partition=short --time=00:30:00 --mem=16G \
          --output="$OUTDIR/merge.%j.out" \
          --wrap="${PYTHON:-python3} $PY_SCRIPT merge --out $OUTDIR")
    echo "merge job $mid (waits for $jid)"
    echo "watch: squeue -u \$USER -j $jid,$mid"
    ;;

  local)
    # Single process, no Slurm. Given the projection cost, this is often the right answer.
    setup_python
    "$PYTHON" "$PY_SCRIPT" count --category "$CATEGORY" --file-list "$FILELIST" \
      --exclude-ids "$IDS" --id-filter "$ID_FILTER" --label "$LABEL" --out "$OUTDIR"
    "$PYTHON" "$PY_SCRIPT" merge --out "$OUTDIR"
    ;;

  merge)
    setup_python
    "$PYTHON" "$PY_SCRIPT" merge --out "$OUTDIR"
    ;;

  *)
    sed -n '2,40p' "${BASH_SOURCE[0]}"
    ;;
esac
