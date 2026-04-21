# Dataset Overview

This directory contains the lightweight public data release used by the UL-XCoT repository. It keeps the benchmark inputs needed for code understanding and limited reproduction, while excluding large generated outputs, cached traces, and internal sweep artifacts.

## Included Resources

The public `dataset/` directory keeps two main benchmark resources:

- `polymath/input/`: multilingual PolyMath benchmark inputs organized by language and difficulty
- `MMLU-ProX-Lite_2col_tsv_by_lang/`: per-language TSV files for the multilingual MMLU-ProX-Lite setting

In addition, the MMLU directory keeps one retained routing metadata file:

- `MMLU-ProX-Lite_2col_tsv_by_lang/routing_results_polymath_ds7b_new.json`

## Benchmark Background

### `polymath/input/`

`polymath/input/` is the main benchmark family used in this repository.

PolyMath is a multilingual mathematical reasoning benchmark designed to evaluate how well large language models reason across languages, rather than only in English. The benchmark covers **18 languages** and organizes problems into **4 difficulty levels** from easy to hard. Its goal is to provide a more discriminative testbed for multilingual math reasoning, with attention to language diversity, reasoning difficulty, and translation quality.

In this repository, PolyMath is the primary benchmark for studying:

- native-language multilingual reasoning
- auxiliary-language routing
- cross-lingual chain-of-thought efficiency
- token reduction under dynamic pruning

The retained public version is organized as:

- one directory per language
- four difficulty splits per language: `low`, `medium`, `high`, and `top`

Languages currently included in this release:

- `ar`, `bn`, `de`, `en`, `es`, `fr`, `id`, `it`, `ja`, `ko`, `ms`, `pt`, `ru`, `sw`, `te`, `th`, `vi`, `zh`

How it is used in this repo:

- `utils/config.py` points the default `language_sample_dir` to `./dataset/polymath/input`
- `process_query.py` uses the PolyMath setting as the main multilingual reasoning benchmark
- the retained inference scripts are primarily organized around this benchmark

Reference work:

- PolyMath: *Evaluating Mathematical Reasoning in Multilingual Contexts*  
  Paper: https://huggingface.co/papers/2504.18428

### `MMLU-ProX-Lite_2col_tsv_by_lang/`

`MMLU-ProX-Lite_2col_tsv_by_lang/` is the code-ready release of the multilingual MMLU-ProX-Lite benchmark used by this project.

MMLU-ProX is a multilingual extension of MMLU-Pro for evaluating advanced LLM reasoning across languages and cultural contexts. The broader MMLU-ProX benchmark emphasizes **parallel questions across languages**, making direct cross-lingual comparison easier. The **Lite** version is a smaller evaluation subset designed for more efficient benchmarking while still preserving multilingual coverage. The current public release used here covers **29 languages**, with `test.tsv` and `validation.tsv` files stored separately for each language.

In this repository, MMLU-ProX-Lite serves as a second evaluation setting beyond PolyMath. It is mainly used to check whether UL-XCoT's routing and pruning behavior generalizes outside the math-focused benchmark.

How it is used in this repo:

- `utils/file_processor.py` reads `test.tsv` from this directory
- `scripts/infer_MMLU.sh` uses this directory as the main evaluation input
- `routing_results_polymath_ds7b_new.json` provides one retained routing metadata example for this setting

Languages currently included in this release:

- `af`, `ar`, `bn`, `cs`, `de`, `en`, `es`, `fr`, `hi`, `hu`, `id`, `it`, `ja`, `ko`, `mr`, `ne`, `pt`, `ru`, `sr`, `sw`, `te`, `th`, `uk`, `ur`, `vi`, `wo`, `yo`, `zh`, `zu`

Reference work:

- MMLU-ProX: *A Multilingual Benchmark for Advanced Large Language Model Evaluation*  
  Project page: https://mmluprox.github.io/  
  Dataset card: https://huggingface.co/datasets/li-lab/MMLU-ProX-Lite  
  Paper: https://huggingface.co/papers/2503.10497

## What Is Not Included

This GitHub version does not include every dataset artifact used during the original research cycle. Several files referenced by legacy experiment paths are intentionally omitted because they are generated, large, or straightforward to reconstruct locally.

Notably, the public release does not currently include:

- `dataset/polymath/translate/`
- `dataset/polymath/routing_results_*.json`
- large model outputs under dataset subdirectories
- parameter sweep folders such as `new_RL-*`, `RL-*`, `llama_*`, and `qwen*`

This means:

- the core benchmark inputs are available
- some routing metadata must be regenerated locally if you want to reproduce every internal experiment variant
- some older helper scripts still contain optional paths that assume those generated files exist

## Why Only A Lightweight Release

The original working directory contained many experiment byproducts:

- generated inference outputs
- self-consistency runs
- cached reasoning traces
- hyperparameter sweeps
- dataset-side analysis artifacts

Those files are not appropriate for a paper-facing source repository because they are large and derived from execution rather than canonical source data.

The public version therefore focuses on:

- benchmark inputs
- code-facing evaluation files
- a minimal retained routing metadata example
- concise documentation describing what was removed
