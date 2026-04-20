# Dataset Layout

This directory contains the lightweight public data release for the UL-XCoT paper repository.
It keeps the benchmark inputs needed to understand the task setup and run limited reproduction, while excluding large generated outputs and heavyweight intermediate artifacts.

## What Is Included

The public `dataset/` directory currently keeps two code-facing resources:

- `polymath/input/`: multilingual PolyMath benchmark inputs organized by language and difficulty
- `MMLU-ProX-Lite_2col_tsv_by_lang/`: per-language TSV files for the multilingual MMLU-ProX-Lite setting

In addition, the MMLU directory keeps one routing metadata file:

- `MMLU-ProX-Lite_2col_tsv_by_lang/routing_results_polymath_ds7b_new.json`

## Data Provenance And Role In This Project

### `polymath/input/`

PolyMath is the main benchmark family used in this repository.
It contains multilingual math reasoning problems and is the primary testbed for the paper's analysis of cross-lingual reasoning and auxiliary-language routing.

The retained public version is organized as:

- one directory per language
- four difficulty splits per language: `low`, `medium`, `high`, and `top`

Languages currently included in the public release are:

- `ar`, `bn`, `de`, `en`, `es`, `fr`, `id`, `it`, `ja`, `ko`, `ms`, `pt`, `ru`, `sw`, `te`, `th`, `vi`, `zh`

How it is used in this repo:

- `utils/config.py` points the default `language_sample_dir` to `./dataset/polymath/input`
- `process_query.py` uses the PolyMath setting as the main multilingual reasoning benchmark
- the retained inference scripts are written around this benchmark family

Role in the paper:

- evaluate multilingual reasoning performance under the native-language setting
- compare routed auxiliary-language inference against direct decoding
- study how language choice affects reasoning quality and token efficiency

### `MMLU-ProX-Lite_2col_tsv_by_lang/`

This directory is the normalized, code-ready release of the multilingual MMLU-ProX-Lite benchmark used by this project.
Each language has `test.tsv` and `validation.tsv` files in a simple two-column TSV layout so the evaluation pipeline can read them directly.

How it is used in this repo:

- `utils/file_processor.py` reads `test.tsv` from this directory
- `scripts/infer_MMLU.sh` uses this directory as its main evaluation input
- `routing_results_polymath_ds7b_new.json` provides one retained routing metadata example for this setting

Role in the paper:

- serve as a second multilingual evaluation setting beyond PolyMath
- check whether the routing behavior generalizes outside the main math benchmark

## Files Not Included In The Public Release

This GitHub version does not include every dataset artifact used during the research cycle.
Several files referenced by legacy experiment paths are intentionally omitted because they are generated, large, or easy to reconstruct locally.

Notably, the public release does not currently include:

- `dataset/polymath/translate/`
- `dataset/polymath/routing_results_*.json`
- large model outputs under dataset subdirectories
- parameter sweep folders such as `new_RL-*`, `RL-*`, `llama_*`, and `qwen*`

This means:

- the core benchmark inputs are available
- some auxiliary-language routing metadata must be regenerated locally if you want to reproduce every internal experiment variant
- some older helper scripts still contain optional paths that assume these generated files exist

## Why The Repository Keeps Only A Subset

The original working directory contained many experiment byproducts:

- generated inference outputs
- self-consistency runs
- cached reasoning traces
- hyperparameter sweeps
- dataset-side analysis artifacts

Those files are not appropriate for a paper-facing GitHub repository because they are large, derived from code execution, and not the canonical source data.

The public version therefore focuses on:

- benchmark inputs
- code-facing evaluation files
- a minimal routing metadata example
- documentation describing what was removed and why

If you need the full internal outputs, regenerate them locally or host them separately from the source repository.
