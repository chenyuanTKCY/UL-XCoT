#!/bin/bash
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PARENT_DIR=$(dirname "$SCRIPT_DIR")
PYTHON_BIN="${PYTHON_BIN:-python}"

DATA_PATH="${DATA_PATH:-./dataset/polymath/new_RL-qwen-7B_correct_0.8_0.6/SC_mo/}"
MODEL_NAME="${MODEL_NAME:-gpt-4o-mini}"
API_KEY="${OPENAI_API_KEY:-}"
SEED=42
AUTOCAP=False

if [ -z "$API_KEY" ]; then
  echo "OPENAI_API_KEY is not set."
  exit 1
fi

for SINGLE_MODE in top;
do
echo "Running eval_answer.py..."
"${PYTHON_BIN}" "${PARENT_DIR}/eval_answer.py" \
  --data_path "$DATA_PATH" \
  --model_name "$MODEL_NAME" \
  --api_key "$API_KEY" \
  --single_mode "$SINGLE_MODE" \
  --seed "$SEED" \
  --autocap "$AUTOCAP" \
  --dataset "polymath"
done
