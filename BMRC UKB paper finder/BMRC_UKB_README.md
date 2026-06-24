# BMRC UKB full-text matching runbook

## 1. Copy files to BMRC path

```bash
mkdir -p /well/mills/projects/scientometric/data_copy_script/logs
mkdir -p /well/mills/projects/scientometric/data_copy_script/ukb_term_year_cache_no_acronym

cp BMRC_stage1_api_no_acronym.py /well/mills/projects/scientometric/data_copy_script/
cp BMRC_stage2_local_no_acronym.py /well/mills/projects/scientometric/data_copy_script/
```

## 2. Load environment

```bash
module load Miniforge3
PYTHON=/well/mills/projects/scientometric/venvs/dedup_env/bin/python

$PYTHON --version
$PYTHON -m pip install --upgrade pip
$PYTHON -m pip install dimcli pyarrow fastparquet tqdm pandas
```

## 3. Syntax check

```bash
$PYTHON -m py_compile /well/mills/projects/scientometric/data_copy_script/BMRC_stage1_api_no_acronym.py
$PYTHON -m py_compile /well/mills/projects/scientometric/data_copy_script/BMRC_stage2_local_no_acronym.py
```

## 4. Create Stage 1 sbatch

```bash
cat > /well/mills/projects/scientometric/data_copy_script/run_BMRC_stage1_api_no_acronym.sbatch <<'EOF'
#!/bin/bash
#SBATCH -J ukb_api_noacr
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --time=1-06:00:00
#SBATCH -o /well/mills/projects/scientometric/data_copy_script/logs/ukb_api_noacr_%j.out
#SBATCH -e /well/mills/projects/scientometric/data_copy_script/logs/ukb_api_noacr_%j.err

set -euo pipefail
module load Miniforge3 || true
PYTHON=/well/mills/projects/scientometric/venvs/dedup_env/bin/python

$PYTHON --version
$PYTHON /well/mills/projects/scientometric/data_copy_script/BMRC_stage1_api_no_acronym.py
EOF
```

## 5. Create Stage 2 sbatch

```bash
cat > /well/mills/projects/scientometric/data_copy_script/run_BMRC_stage2_local_no_acronym.sbatch <<'EOF'
#!/bin/bash
#SBATCH -J ukb_loc_noacr
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=1-06:00:00
#SBATCH -o /well/mills/projects/scientometric/data_copy_script/logs/ukb_loc_noacr_%j.out
#SBATCH -e /well/mills/projects/scientometric/data_copy_script/logs/ukb_loc_noacr_%j.err

set -euo pipefail
module load Miniforge3 || true
PYTHON=/well/mills/projects/scientometric/venvs/dedup_env/bin/python
export MAX_WORKERS=8

$PYTHON --version
$PYTHON /well/mills/projects/scientometric/data_copy_script/BMRC_stage2_local_no_acronym.py
EOF
```

## 6. Submit chained jobs

```bash
jid1=$(sbatch --parsable /well/mills/projects/scientometric/data_copy_script/run_BMRC_stage1_api_no_acronym.sbatch)
jid2=$(sbatch --parsable --dependency=afterok:$jid1 /well/mills/projects/scientometric/data_copy_script/run_BMRC_stage2_local_no_acronym.sbatch)

echo "Stage1 job: $jid1"
echo "Stage2 job: $jid2"
```

## 7. Monitor

```bash
squeue -u $USER
tail -f /well/mills/projects/scientometric/data_copy_script/logs/ukb_api_noacr_${jid1}.out
tail -f /well/mills/projects/scientometric/data_copy_script/logs/ukb_loc_noacr_${jid2}.out
```

## 8. Check outputs

```bash
ls -lh /well/mills/projects/scientometric/data_copy_script/matched_ukb_full_api_raw_1.csv
ls -lh /well/mills/projects/scientometric/data_copy_script/matched_ukb_full_api_with_abstract_1.csv
ls -lh /well/mills/projects/scientometric/data_copy_script/matched_ukb_full_1.csv

$PYTHON - <<'EOF'
import pandas as pd
for p in [
    "/well/mills/projects/scientometric/data_copy_script/matched_ukb_full_api_raw_1.csv",
    "/well/mills/projects/scientometric/data_copy_script/matched_ukb_full_api_with_abstract_1.csv",
    "/well/mills/projects/scientometric/data_copy_script/matched_ukb_full_1.csv",
]:
    df = pd.read_csv(p)
    print(p, len(df), df["id"].nunique() if "id" in df.columns else "no id")
EOF
```
