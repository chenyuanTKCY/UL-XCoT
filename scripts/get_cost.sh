#!/bin/bash
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PARENT_DIR=$(dirname "$SCRIPT_DIR")
PYTHON_BIN="${PYTHON_BIN:-python}"

PATH_TO_RESULTS="${PATH_TO_RESULTS:-./dataset/polymath/new_RL-qwen-7B_correct_0.8_0.6/SC_mo/}"

for DIFF in medium
do
MODE="single"
echo "Running get_cost.py..."
"${PYTHON_BIN}" "${PARENT_DIR}/get_cost.py" \
  --path "$PATH_TO_RESULTS" \
  --mode "$MODE" \
  --diff "$DIFF" \
  --dataset "polymath"
done
