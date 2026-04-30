#!/usr/bin/env bash
set -e

python -m src.main \
  --backend vllm \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --input data/conll04_test.json \
  --output outputs/vllm_conll04_test.json \
  --gold data/conll04_test_gold.json \
  --limit 20
