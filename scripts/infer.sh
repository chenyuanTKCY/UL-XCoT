#!/bin/bash

########################
# Settings
########################
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PARENT_DIR=$(dirname "$SCRIPT_DIR")
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
DEVICE_ID="${DEVICE_ID:-0}"
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
LOGIC_RESULT_PATH="./dataset/polymath/routing_results_polymath_ds7b_4.json"
TEST_MODE="ours"
GROUP_NUM=3
SAMPLING_SIZE=9
MODEL_NAME="new_RL-qwen-7B_correct"
MODEL_PATH="${MODEL_PATH:-${ULXCOT_MODEL_PATH:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}}"
PRUNING_RATE=0.7


declare -A MAX_TOKEN_NUMS_dict
MAX_TOKEN_NUMS_dict["low"]=2048
MAX_TOKEN_NUMS_dict["medium"]=4096
MAX_TOKEN_NUMS_dict["high"]=5120
MAX_TOKEN_NUMS_dict["top"]=10240
OUTPUT_PATH="./dataset/polymath/${MODEL_NAME}_${TEMPERATURE}_${TOP_P}/${TEST_MODE}/layer4/"

for QUERY_DIFF in low
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
  --prune_ratio "$PRUNING_RATE" \
  --max_token_nums "${MAX_TOKEN_NUMS_dict[$QUERY_DIFF]}"
done
