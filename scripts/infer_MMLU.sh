#!/bin/bash

########################
# Settings
########################
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PARENT_DIR=$(dirname "$SCRIPT_DIR")
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DEVICE_ID="${DEVICE_ID:-5}"
PYTHON_BIN="${PYTHON_BIN:-python}"
VLLM_SRC="${VLLM_SRC:-$PARENT_DIR/hidden_vllm}"

########################
# Hyper-parameters 
########################
SEED=42
LAMDA=0.4
TEMPERATURE=0.8
TOP_P=0.8

# Default parameters (can be overridden from CLI)
LOGIC_RESULT_PATH="./dataset/MMLU-ProX-Lite_2col_tsv_by_lang/routing_results_polymath_ds7b_new.json"
TEST_MODE="CLSP_cost"
GROUP_NUM=3
SAMPLING_SIZE=29
PRUNING_RATE=0.8
MODEL_NAME="new_RL-qwen-7B_correct"
MODEL_PATH="${MODEL_PATH:-${ULXCOT_MODEL_PATH:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}}"
MAX_TOKEN_NUMS=5120
OUTPUT_PATH="./dataset/MMLU-ProX-Lite_2col_tsv_by_lang/${MODEL_NAME}_${TEMPERATURE}_${TOP_P}/${TEST_MODE}/${MAX_TOKEN_NUMS}/"
DATASET="MMLU"

for QUERY_DIFF in single
do
# Run Python script
echo "Running process_query.py..."

export PYTHONPATH="${VLLM_SRC}:${PYTHONPATH}"
CUDA_VISIBLE_DEVICES=${DEVICE_ID} "${PYTHON_BIN}" "${PARENT_DIR}/process_query.py" \
  --logic_result_path "$LOGIC_RESULT_PATH" \
  --test_mode "$TEST_MODE" \
  --group_num "$GROUP_NUM" \
  --model_path "$MODEL_PATH"\
  --sampling_size "$SAMPLING_SIZE" \
  --output_path "$OUTPUT_PATH" \
  --query_diff "$QUERY_DIFF" \
  --seed "$SEED" \
  --lamda "$LAMDA"\
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
  --prune_ratio "$PRUNING_RATE"\
  --max_token_nums "$MAX_TOKEN_NUMS"\
  --dataset "$DATASET"
done


