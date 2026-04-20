# UL-XCoT

Official repository for the paper:

**Less Languages, Less Tokens: An Efficient Unified Logic Cross-lingual Chain-of-Thought Reasoning Framework**

Chenyuan Zhang, Qiguang Chen, Xie Chen, Zhuotao Tian, Bowen Xing, Meishan Zhang, Libo Qin, Baotian Hu, Min Zhang

Paper PDF: [paper/3459_Less_Languages_Less_Token.pdf](paper/3459_Less_Languages_Less_Token.pdf)

## Overview

UL-XCoT is an efficient cross-lingual chain-of-thought (XCoT) self-consistency framework for multilingual reasoning.
The method improves inference efficiency in two ways:

- less languages: select a small candidate language set in a unified logic space
- less tokens: prune low-quality reasoning trajectories during decoding with early stopping

The paper evaluates UL-XCoT on:

- PolyMath across 18 languages
- Global-MMLU-Lite across 29 languages

using DeepSeek-R1-Distill-Qwen-7B as the main backbone.

## Method Summary

The repository is organized around the main stages of the paper:

1. Unified Logic Mechanism
   Projects multilingual hidden states into a shared logic space so they can be compared across languages.
2. Candidate Language Selection
   Uses routing metadata to choose a small subset of helpful auxiliary languages for each query.
3. Dynamic CoT Pruning
   Monitors reasoning trajectories during decoding and early-stops low-quality branches.
4. Voting
   Aggregates the remaining trajectories to produce the final answer.

## Repository Layout

```text
UL-XCoT/
├── README.md
├── requirements.txt
├── process_query.py
├── get_assistant_languages.py
├── eval_answer.py
├── get_cost.py
├── dataset/
├── scripts/
├── utils/
├── paper/
└── hidden_vllm/    # git submodule
```

Key files:

- [process_query.py](process_query.py): main inference entry for UL-XCoT and retained baselines
- [get_assistant_languages.py](get_assistant_languages.py): generates routing / candidate-language metadata
- [eval_answer.py](eval_answer.py): evaluates answer accuracy
- [get_cost.py](get_cost.py): summarizes token cost
- [utils/language_router.py](utils/language_router.py): language routing logic
- [utils/early_stop.py](utils/early_stop.py): dynamic pruning / early stopping logic
- [utils/inference_utils.py](utils/inference_utils.py): inference wrapper

## Lightweight Public Release

This GitHub version is intentionally lightweight.
Large generated outputs, parameter sweeps, cached reasoning traces, and heavyweight dataset payloads were removed.

What remains:

- the core code path for the paper
- the retained public shell entry points
- lightweight benchmark inputs and routing metadata
- the paper PDF

For dataset details, see [dataset/README.md](dataset/README.md).

## Setup

### 1. Clone the repository

```bash
git clone --recurse-submodules https://github.com/chenyuanTKCY/UL-XCoT.git
cd UL-XCoT
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

### 2. Prepare the environment

```bash
conda env create -f environment.yml
conda activate LangRouter
```

The file [environment.yml](environment.yml) was exported from the original runtime environment used for this project.

### 3. Install `hidden_vllm`

`hidden_vllm` is a customized vLLM implementation used in this project to expose hidden states during decoding.
It is required by the inference pipeline.

```bash
cd hidden_vllm
pip install -e .
cd ..
```

### 4. Set runtime paths

The public scripts no longer depend on hard-coded local machine paths.
They use environment variables instead.

Recommended variables:

```bash
export ULXCOT_MODEL_PATH=/path/to/DeepSeek-R1-Distill-Qwen-7B
export PYTHON_BIN=/home/cyzhang/miniconda/envs/LangRouter/bin/python
export VLLM_SRC=$PWD/hidden_vllm
```

Notes:

- `ULXCOT_MODEL_PATH` should point to the model directory or model identifier used for inference.
- `VLLM_SRC` is added to `PYTHONPATH` by the retained shell scripts.
- In the original paper environment, the main runtime lived in `/home/cyzhang/miniconda/envs/LangRouter/`.
- `hidden_vllm` should be installed locally in that environment because it implements the hidden-state extraction used by UL-XCoT.
- Some scripts assume a GPU environment similar to the paper setup.

## Quick Start

### 1. Generate routing metadata

```bash
bash scripts/get_assislan.sh
```

This corresponds to candidate language selection metadata generation.

### 2. Run UL-XCoT

```bash
bash scripts/infer.sh
```

### 3. Run representative baselines

```bash
bash scripts/CLSP_acc.sh
bash scripts/self_consistency.sh
bash scripts/translate_to_EN.sh
bash scripts/infer_MMLU.sh
```

### 4. Evaluate accuracy

```bash
export OPENAI_API_KEY=your_key_here
bash scripts/get_acc.sh
```

### 5. Summarize token cost

```bash
bash scripts/get_cost.sh
```

For script-level notes, see [scripts/README.md](scripts/README.md).

## Dataset

The repository keeps only the lightweight subset required for code understanding and limited reproduction.

Main retained data:

- `dataset/polymath/input/`
- `dataset/MMLU-ProX-Lite_2col_tsv_by_lang/*/{test,validation}.tsv`
- `dataset/MMLU-ProX-Lite_2col_tsv_by_lang/routing_results_polymath_ds7b_new.json`

See [dataset/README.md](dataset/README.md) for data provenance and role descriptions.

## Paper Assets

The paper PDF is stored in [paper/3459_Less_Languages_Less_Token.pdf](paper/3459_Less_Languages_Less_Token.pdf).

## Citation

If you find this repository useful, please cite:

```bibtex
@article{zhang2026ulxcot,
  title={Less Languages, Less Tokens: An Efficient Unified Logic Cross-lingual Chain-of-Thought Reasoning Framework},
  author={Zhang, Chenyuan and Chen, Qiguang and Chen, Xie and Tian, Zhuotao and Xing, Bowen and Zhang, Meishan and Qin, Libo and Hu, Baotian and Zhang, Min},
  year={2026}
}
```

## Acknowledgements

- The paper uses a customized vLLM-based backend tracked here as the `hidden_vllm` submodule.
- Public release cleanup removed large experiment outputs so the repository can serve as a paper-facing code release.
