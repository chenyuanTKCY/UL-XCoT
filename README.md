<div align="center">

# Less Languages, Less Tokens

## UL-XCoT

### An Efficient Unified Logic Cross-lingual Chain-of-Thought Reasoning Framework

[![Paper](https://img.shields.io/badge/Paper-PDF-B31B1B?style=for-the-badge)](paper/3459_Less_Languages_Less_Token.pdf)
![Version](https://img.shields.io/badge/Version-v1.0-2F6BFF?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Public%20Release-0A7F5A?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Task](https://img.shields.io/badge/Task-XCoT%20Reasoning-6A5ACD?style=for-the-badge)

**Official repository for the UL-XCoT paper**

Chenyuan Zhang, Qiguang Chen, Xie Chen, Zhuotao Tian, Bowen Xing, Meishan Zhang, Libo Qin, Baotian Hu, Min Zhang

</div>

## 🔥 News

- `v1.0` public release of the paper-facing repository.
- The paper PDF is available in this repository at [paper/3459_Less_Languages_Less_Token.pdf](paper/3459_Less_Languages_Less_Token.pdf).
- The release keeps the core code, lightweight benchmark inputs, and public scripts for UL-XCoT.

## 💡 Overview

Cross-lingual chain-of-thought (XCoT) reasoning can improve multilingual problem solving, but it often introduces substantial inference overhead. UL-XCoT addresses this efficiency bottleneck from two complementary directions:

1. **Less Languages**: route each query to a small set of helpful auxiliary languages in a unified logic space instead of reasoning over many languages.
2. **Less Tokens**: prune low-quality reasoning trajectories during decoding with early stopping to reduce unnecessary token generation.

In this repository, UL-XCoT is mainly evaluated on:

- **PolyMath** across 18 languages
- **Global-MMLU-Lite / MMLU-ProX-Lite** style multilingual evaluation across 29 languages

The main backbone used in the paper is **DeepSeek-R1-Distill-Qwen-7B**.

## 🧠 Method

The repository is organized around four main stages of the framework:

1. **Unified Logic Mechanism**  
   Project multilingual hidden states into a shared logic space so reasoning behavior can be compared across languages.
2. **Candidate Language Selection**  
   Use routing metadata to select a small auxiliary-language set for each query.
3. **Dynamic CoT Pruning**  
   Monitor decoding trajectories and early-stop low-confidence branches.
4. **Voting**  
   Aggregate the remaining trajectories to produce the final prediction.

## 🎯 Installation

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

The file [environment.yml](environment.yml) was exported from the original runtime environment used in this project.

### 3. Install `hidden_vllm`

`hidden_vllm` is a customized vLLM backend used to expose hidden states during decoding. It is required by the UL-XCoT inference pipeline.

```bash
cd hidden_vllm
pip install -e .
cd ..
```

### 4. Set runtime paths

The public scripts use environment variables instead of machine-specific hard-coded paths.

```bash
export ULXCOT_MODEL_PATH=/path/to/DeepSeek-R1-Distill-Qwen-7B
export PYTHON_BIN=/home/cyzhang/miniconda/envs/LangRouter/bin/python
export VLLM_SRC=$PWD/hidden_vllm
```

Notes:

- `ULXCOT_MODEL_PATH` should point to the model directory or model identifier used for inference.
- `PYTHON_BIN` should point to the Python executable in your runtime environment.
- `VLLM_SRC` is added to `PYTHONPATH` by the retained shell scripts.
- Some scripts assume a GPU environment similar to the paper setup.

## 🚀 Quick Start

### 1. Generate routing metadata

```bash
bash scripts/get_assislan.sh
```

This step prepares candidate-language routing information for later inference.

### 2. Run UL-XCoT inference

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

### 4. Evaluate answer accuracy

```bash
export OPENAI_API_KEY=your_key_here
bash scripts/get_acc.sh
```

### 5. Summarize token cost

```bash
bash scripts/get_cost.sh
```

For more script-level notes, see [scripts/README.md](scripts/README.md).

## 📦 Dataset

This GitHub release is intentionally lightweight and keeps only the benchmark inputs and metadata needed for code understanding and limited reproduction.

Main retained resources:

- `dataset/polymath/input/`
- `dataset/MMLU-ProX-Lite_2col_tsv_by_lang/*/{test,validation}.tsv`
- `dataset/MMLU-ProX-Lite_2col_tsv_by_lang/routing_results_polymath_ds7b_new.json`

For detailed data provenance, structure, and omitted artifacts, see [dataset/README.md](dataset/README.md).

## 🖨️ File Structure

```text
UL-XCoT/
├── README.md
├── environment.yml
├── requirements.txt
├── process_query.py
├── get_assistant_languages.py
├── eval_answer.py
├── get_cost.py
├── dataset/
│   ├── polymath/
│   └── MMLU-ProX-Lite_2col_tsv_by_lang/
├── scripts/
│   ├── infer.sh
│   ├── CLSP_acc.sh
│   ├── self_consistency.sh
│   ├── translate_to_EN.sh
│   ├── infer_MMLU.sh
│   ├── get_assislan.sh
│   ├── get_acc.sh
│   └── get_cost.sh
├── utils/
├── paper/
└── hidden_vllm/
```

Key files:

- `process_query.py`: main inference entry for UL-XCoT and retained baselines
- `get_assistant_languages.py`: candidate-language routing generation
- `eval_answer.py`: answer evaluation
- `get_cost.py`: token-cost statistics
- `utils/language_router.py`: language routing logic
- `utils/early_stop.py`: dynamic pruning / early stopping logic
- `utils/inference_utils.py`: inference wrapper



## ✒️ Citation

If you find this repository useful for your research, please consider citing:

```bibtex
@article{zhang2026ulxcot,
  title={Less Languages, Less Tokens: An Efficient Unified Logic Cross-lingual Chain-of-Thought Reasoning Framework},
  author={Zhang, Chenyuan and Chen, Qiguang and Chen, Xie and Tian, Zhuotao and Xing, Bowen and Zhang, Meishan and Qin, Libo and Hu, Baotian and Zhang, Min},
  booktitle={Proc. of ACL 2026},
  year={2026}
}
```

## 🙏 Acknowledgements

- The project uses a customized vLLM backend tracked as the `hidden_vllm` submodule.
- This public release focuses on the core code path and paper-facing materials rather than full internal experiment outputs.

## 📮 Contact

Please create a GitHub issue in this repository if you have questions, suggestions, or reproduction-related feedback.
