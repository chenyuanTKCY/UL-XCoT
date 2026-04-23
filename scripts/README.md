# Script Entry Points

This directory keeps the minimal public entry points for the paper.
Many one-off sweeps and machine-specific variants were removed before release.

## Retained Scripts

- `infer.sh`: main UL-XCoT inference entry on PolyMath
- `CLSP_acc.sh`: representative CLSP baseline
- `self_consistency.sh`: representative self-consistency baseline
- `infer_MMLU.sh`: MMLU evaluation entry
- `get_assislan.sh`: routing / candidate-language generation
- `get_acc.sh`: answer evaluation, requires `OPENAI_API_KEY`
- `get_cost.sh`: token-cost summarization

## Runtime Variables

The retained scripts support environment-variable overrides for public use:

- `ULXCOT_MODEL_PATH`: model path or model identifier
- `PYTHON_BIN`: python executable, default `python`
- `VLLM_SRC`: source directory added to `PYTHONPATH`, default `hidden_vllm`
- `DEVICE_ID`: CUDA device id for the script

Example:

```bash
export ULXCOT_MODEL_PATH=/path/to/DeepSeek-R1-Distill-Qwen-7B
export PYTHON_BIN=/home/cyzhang/miniconda/envs/LangRouter/bin/python
export VLLM_SRC=$PWD/hidden_vllm
bash scripts/infer.sh
```

## Notes

- The recommended environment setup is `conda env create -f environment.yml` from the repository root.
- In the original environment, the scripts were run from the `LangRouter` conda environment.
- `hidden_vllm` is a customized local vLLM backend and should be installed in that environment with `pip install -e hidden_vllm`.
- The scripts still reflect the paper's example hyperparameter settings.
- If you need additional sweeps, copy a retained script and modify it locally instead of keeping all temporary variants in the repository.
