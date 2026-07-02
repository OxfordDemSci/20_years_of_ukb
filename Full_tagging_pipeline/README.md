# Full tagging pipeline

## Contains

- model_qwen.py: Qwen model settings.
- model_llama3.py: Llama3 model settings.
- model_mistral.py: Mistral model settings.
- tagger.py: model tagging functions.
- repair_failed_parses.py: failed-parse repair round.
- consensus.py: three-model TRUE agreement and rest split.
- run_three_model_pipeline.py: full execution script.
- common.py: shared helpers.

## Run

pip install -r requirements.txt

python run_three_model_pipeline.py

Edit placeholder input/output paths before running.

## Requirements

See requirements.txt.
