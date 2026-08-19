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
#   ./academic_impact_field_counts.sh make-ids   # 2. the UKBB id list, once
#   ./academic_impact_field_counts.sh probe      # 3. measure before you size (DO THIS)
#   ./academic_impact_field_counts.sh submit     # 4. the whole chain + dependent merge
#   ./academic_impact_field_counts.sh local      # (no Slurm) whole corpus on one core
#
# Override anything by exporting it first, e.g.
#   CORPUS=/well/.../parquet NUM_SHARDS=20 ./academic_impact_field_counts.sh submit
#
# Pick the classification system(s) with CATEGORY (default 'for'); e.g.
#   CATEGORY=rcdc      ./...field_counts.sh local   # NIH disease/condition topics
#   CATEGORY=for,rcdc  ./...field_counts.sh submit  # both, in one chain
# Partials are named per-category, so several categories share one OUTDIR and one
# merge. Counting two systems costs barely more than one — the projected read is
# dominated by id/year/type, and a second category column adds a few hundred KB per
# file — so listing them together is cheaper than two separate submissions, which
# would also recompute the (category-independent) citation thresholds twice.
#
# CITATION WEIGHTING AND WHY BOTH ARMS RUN HERE
# ---------------------------------------------
# WEIGHTS (default 'cit,fcr,top10,top10f,mncs') adds citation-weighted columns beside
# counts — see the .py docstring for what each one means. Two consequences for this
# driver:
#
#   1. Citations accrue over time, so the two arms MUST come from the same snapshot.
#      `submit` therefore counts both arms off this one FILELIST — background with
#      --id-filter exclude, UK Biobank with --id-filter include — instead of leaving
#      the UKBB arm to a separate local run whose citation counts are as-of a
#      different day and silently higher.
#   2. top10 / top10f / mncs need a reference (a decile cut-off, an expected citation
#      count) computed BEFORE counting, so `submit` runs two extra passes first: a
#      citdist array over the whole corpus (no id filter — "top 10 % of the
#      literature" means the literature), then a single thresholds job. Drop those
#      weights from WEIGHTS and both passes are skipped.
#
#      top10 needs only a (year, type) reference. top10f and mncs need one per
#      CATEGORY — the top tenth OF THAT FIELD, and that field's mean citations — so
#      citdist runs once per system in CATEGORY. That is the pass that lets the
#      analysis say "top 10 % of epidemiology" rather than "top 10 % of everything",
#      and gives a field-normalised score for the years Dimensions has not published
#      an FCR for.
#
# The chain `submit` builds, each step waiting on the last (afterok):
#      citdist array -> thresholds -> [background array | ukbb array] -> merge
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
NUM_SHARDS="${NUM_SHARDS:-5}"
LABEL="${LABEL:-background}"
ID_FILTER="${ID_FILTER:-exclude}"     # exclude = the non-UKBB background arm
CATEGORY="${CATEGORY:-for}"           # one system or a comma-separated list:
                                      # for|rcdc|uoa|hrcs_hc|... partials are named
                                      # per-category, so several share one OUTDIR
                                      # and one merge.

# 'for,rcdc' -> 'for rcdc', so every caller loops the same way.
category_list() { echo "${CATEGORY//,/ }"; }

# Citation weights, comma-separated; empty string = paper counts only, exactly as
# before. The derived weights pull in the citdist + thresholds passes automatically.
WEIGHTS="${WEIGHTS:-cit,fcr,top10,top10f,mncs}"
THRESHOLDS="${THRESHOLDS:-$OUTDIR/cit_thresholds.parquet}"
EXPECTED="${EXPECTED:-$OUTDIR/cit_expected.parquet}"
PERCENTILE="${PERCENTILE:-10}"

# Both arms off one file list, so their citation counts share one as-of date.
UKBB_LABEL="${UKBB_LABEL:-ukbb}"

# Assembled once here so `submit`, `local` and the array body cannot drift apart.
weight_args() {
  [[ -z "$WEIGHTS" ]] && return 0
  printf -- '--weights %s' "$WEIGHTS"
  [[ "$WEIGHTS" == *top10* ]] && printf -- ' --top-thresholds %s' "$THRESHOLDS"
  [[ "$WEIGHTS" == *mncs*  ]] && printf -- ' --expected %s' "$EXPECTED"
}

# top10f and mncs are scored against a reference for the CATEGORY a paper is being
# counted into, so citdist has to build one per classification system; top10 alone
# needs only the (year, type) fallback, which every citdist run writes.
needs_reference() { [[ "$WEIGHTS" == *top10* || "$WEIGHTS" == *mncs* ]]; }
needs_per_category() { [[ "$WEIGHTS" == *top10f* || "$WEIGHTS" == *mncs* ]]; }

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
# the CLI dispatch so the submitted script needs no arguments. PHASE says which pass
# this array is (count by default, so an old submitted job still means what it did).
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  setup_python
  echo "${PHASE:-count} shard ${SLURM_ARRAY_TASK_ID}/${NUM_SHARDS} on $(hostname) at $(date)"
  case "${PHASE:-count}" in
    citdist)
      # No id filter on purpose: the reference population is the whole literature, not
      # either arm. One pass per category when a per-category reference is needed —
      # each pass also rewrites the identical (year, type) fallback, which is cheap
      # next to a second read of the corpus.
      if needs_per_category; then
        for cat in $(category_list); do
          echo "--- citdist category $cat"
          "$PYTHON" "$PY_SCRIPT" citdist --category "$cat" \
            --file-list "$FILELIST" \
            --shard "$SLURM_ARRAY_TASK_ID" --num-shards "$NUM_SHARDS" --out "$OUTDIR"
        done
      else
        "$PYTHON" "$PY_SCRIPT" citdist \
          --file-list "$FILELIST" \
          --shard "$SLURM_ARRAY_TASK_ID" --num-shards "$NUM_SHARDS" --out "$OUTDIR"
      fi
      ;;
    count)
      for cat in $(category_list); do
        echo "--- category $cat"
        # shellcheck disable=SC2046  # deliberate word splitting: weight_args is flags
        "$PYTHON" "$PY_SCRIPT" count --category "$cat" \
          --file-list "$FILELIST" \
          --shard "$SLURM_ARRAY_TASK_ID" --num-shards "$NUM_SHARDS" \
          $(weight_args) \
          --exclude-ids "$IDS" --id-filter "$ID_FILTER" \
          --label "$LABEL" --out "$OUTDIR"
      done
      ;;
    *)
      echo "unknown PHASE '$PHASE'" >&2; exit 1 ;;
  esac
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
    # Footer-only: touches ~64 KB per file, safe to run on the login node. The weight
    # columns are included so the printed projection is the one the real run pays.
    setup_python
    for cat in $(category_list); do
      "$PYTHON" "$PY_SCRIPT" probe --category "$cat" --file-list "$FILELIST" \
        ${WEIGHTS:+--weights "$WEIGHTS"} --out "$OUTDIR"
    done
    echo
    echo "Look at 'projected read' above. If it is a small % of the corpus, this job is"
    echo "minutes of work — lower NUM_SHARDS rather than raising it."
    ;;

  submit)
    [[ -s "$FILELIST" ]] || { echo "no $FILELIST — run 'filelist' first" >&2; exit 1; }
    [[ -s "$IDS" ]] || { echo "no $IDS — run 'make-ids' first" >&2; exit 1; }
    # Resolve an absolute interpreter path now, on the login node where the module
    # loads correctly — the merge job's --wrap runs on a compute node that may not
    # have it on PATH, and would otherwise silently fall back to the system python3.
    setup_python
    last=$((NUM_SHARDS - 1))
    # sbatch splits --export on commas, and WEIGHTS is itself a comma-separated list —
    # spelling it inside --export would deliver WEIGHTS=cit and two junk names. So the
    # shared configuration travels through the environment (--export=ALL copies the
    # submitting shell) and only the comma-free per-job scalars are named explicitly.
    export CORPUS OUTDIR IDS FILELIST NUM_SHARDS CATEGORY PY_SCRIPT WEIGHTS THRESHOLDS \
           PERCENTILE UKBB_LABEL EXPECTED
    watch=""
    dep=""            # what the count arrays wait for; empty until top10 says otherwise

    # The cut-offs are a property of the corpus, not of a classification system or an
    # arm, so an existing thresholds file is reused across categories and re-runs.
    # FORCE_THRESHOLDS=1 recomputes them (a new corpus snapshot is the reason to).
    if needs_reference && [[ -s "$THRESHOLDS" && "${FORCE_THRESHOLDS:-0}" != 1 ]]; then
      echo "reusing $THRESHOLDS (FORCE_THRESHOLDS=1 to recompute)"
    elif needs_reference; then
      cid=$(sbatch --parsable --array=0-"$last" --job-name=cit-dist \
            --export=ALL,PHASE=citdist "${BASH_SOURCE[0]}")
      echo "citdist array $cid (0-$last)"
      tid=$(sbatch --parsable --dependency=afterok:"$cid" \
            --job-name=cit-thresh --partition=short --time=00:30:00 --mem=16G \
            --output="$OUTDIR/thresholds.%j.out" \
            --wrap="${PYTHON:-python3} $PY_SCRIPT thresholds --out $OUTDIR --percentile $PERCENTILE")
      echo "thresholds job $tid (waits for $cid)"
      dep="--dependency=afterok:$tid"
      watch="$cid,$tid"
    fi

    # Both arms, same file list, one snapshot. They are independent of each other, so
    # they queue in parallel and only the merge waits for both.
    arm_jobs=()
    for spec in "$LABEL:$ID_FILTER" "$UKBB_LABEL:include"; do
      arm_label="${spec%%:*}"; arm_filter="${spec##*:}"
      jid=$(sbatch --parsable --array=0-"$last" --job-name="counts-$arm_label" \
            $dep --export=ALL,PHASE=count,LABEL="$arm_label",ID_FILTER="$arm_filter" \
            "${BASH_SOURCE[0]}")
      echo "count array $jid — arm '$arm_label' (--id-filter $arm_filter, 0-$last," \
           "categories: $(category_list))"
      arm_jobs+=("$jid")
      watch="${watch:+$watch,}$jid"
    done

    # afterok: the merge must not run on a partial set of shards, or the counts are
    # silently short. If a shard fails, fix it and rerun just that --array=N.
    after=$(IFS=:; echo "${arm_jobs[*]}")
    mid=$(sbatch --parsable --dependency=afterok:"$after" \
          --job-name=for-merge --partition=short --time=00:30:00 --mem=16G \
          --output="$OUTDIR/merge.%j.out" \
          --wrap="${PYTHON:-python3} $PY_SCRIPT merge --out $OUTDIR")
    echo "merge job $mid (waits for ${after//:/, })"
    echo "watch: squeue -u \$USER -j $watch,$mid"
    ;;

  local)
    # Single process, no Slurm. Given the projection cost, this is often the right answer.
    setup_python
    if needs_reference && [[ ! -s "$THRESHOLDS" || "${FORCE_THRESHOLDS:-0}" == 1 ]]; then
      if needs_per_category; then
        for cat in $(category_list); do
          "$PYTHON" "$PY_SCRIPT" citdist --category "$cat" --file-list "$FILELIST" \
            --out "$OUTDIR"
        done
      else
        "$PYTHON" "$PY_SCRIPT" citdist --file-list "$FILELIST" --out "$OUTDIR"
      fi
      "$PYTHON" "$PY_SCRIPT" thresholds --out "$OUTDIR" --percentile "$PERCENTILE"
    fi
    for cat in $(category_list); do
      for spec in "$LABEL:$ID_FILTER" "$UKBB_LABEL:include"; do
        # shellcheck disable=SC2046  # deliberate word splitting: weight_args is flags
        "$PYTHON" "$PY_SCRIPT" count --category "$cat" --file-list "$FILELIST" \
          $(weight_args) \
          --exclude-ids "$IDS" --id-filter "${spec##*:}" --label "${spec%%:*}" \
          --out "$OUTDIR"
      done
    done
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
