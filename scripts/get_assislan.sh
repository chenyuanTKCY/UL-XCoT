#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PARENT_DIR="$( dirname "$SCRIPT_DIR" )"

PYTHON_BIN="${PYTHON_BIN:-python}"

SAMPLING_SIZE=9
LOGIC_RESULT_PATH="./dataset/polymath/routing_results_polymath_ds7b_12.json"
# LOGIC_RESULT_PATH="./dataset/MMLU-ProX-Lite_2col_tsv_by_lang/routing_results_polymath_ds7b_new.json"
MODEL_PATH="${MODEL_PATH:-${ULXCOT_MODEL_PATH:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}}"
DEVICE="${DEVICE:-1}"
GET_SIMILAR_LANGUAGES_MODE="single"


echo "Running get_assistant_languages.py..."
"${PYTHON_BIN}" "${PARENT_DIR}/get_assistant_languages.py" \
  --sampling_size "$SAMPLING_SIZE" \
  --logic_result_path "$LOGIC_RESULT_PATH" \
  --model_path "$MODEL_PATH" \
  --device "$DEVICE" \
  --get_similar_languages_mode "$GET_SIMILAR_LANGUAGES_MODE" \
  --dataset "polymath"
