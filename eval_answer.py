########################################################################
#
# @author : Chenyuan Zhang
# @when : Winter Semester 2025/2026
# @where : Harbin Institute of Technology
# @title : Eval Answer
# @component: root
# @file : eval_answer.py
#
########################################################################

import os
import random
import re
import json
import numpy as np
import argparse
from typing import List, Dict
import torch.multiprocessing as mp
from multiprocessing import Manager
from collections import defaultdict

from tqdm import tqdm

from utils.answer_judge import answerJudge
# from utils.file_processor import read_pkl_file  # currently unused
from utils.config import polymath_LANG_DICT, polymath_LANG_LIST, mmlu_LANG_LIST


def judge_equal(pred, answer, scale: int = 100) -> bool:
    """
    Compare prediction and ground truth numerically with a simple tolerance:
    - Convert both to float
    - Multiply by `scale` (default 100, i.e., keep 2 decimal places)
    - Round and cast to int
    - Check if they are exactly equal

    This reduces common floating-point issues like 1.999999 vs 2.0.
    """

    try:
        pred_scaled = int(round(float(pred) * scale))
        ans_scaled  = int(round(float(answer) * scale))
        return pred_scaled == ans_scaled
    except Exception:
        # If parsing fails (non-numeric prediction), treat as unequal.
        return False


def read_jsonl_file(file_path: str, max_lines: int | None = None):
    """
    Read a JSONL file and return a list of dicts.
    If max_lines is not None, only read at most max_lines lines.
    """
    responses = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            dic_responses = json.loads(line)
            responses.append(dic_responses)

            if max_lines is not None and i >= max_lines:
                break
    return responses



def get_single_pred_from_box_pattern(response: Dict):
    """
    Extract the content inside the last LaTeX \\boxed{...} from the assistant's
    final message for a single response.

    Returns:
        pred          : str or None (None if no boxed expression is found)
        origin_answer : ground-truth answer from response["origin"]["answer"]
    """
    origin_answer = response["origin"]["answer"]

    # Use the last paragraph of the last assistant message (heuristic)
    pred_answer = (
        response["message"][-1]["content"]
        .split("\n\n")[-1]
        .replace(",", "")
        .replace("\n", " ")
        .replace("\r", " ")
    )

    # Match LaTeX \boxed{...}
    pattern = r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}'
    pred_list = re.findall(pattern, pred_answer)

    if len(pred_list) == 0:
        pred = None
    else:
        pred = pred_list[-1].strip()  # use the last boxed expression

    return pred, origin_answer


def get_pred_from_box_pattern(responses: List[Dict], seed:int = None, sampled_scores = None, vote_by_query_id: bool = True, ):
    """
    For a list of responses of a single language:

    - Group responses by query_id.
    - For each query_id, extract predictions for all samples.
    - Perform majority voting on predictions that have a boxed expression.
    - If no sample has a boxed result for a query, the prediction is None.

    Returns:
        pred_list       : list of predictions (str or None) per query
        origin_answers  : list of ground-truth answers per query
    """
    if not vote_by_query_id:
        pred_, origin_answer_ = [], []
        for resp in responses:
            pred, oa = get_single_pred_from_box_pattern(resp)
            pred_.append(pred)
            origin_answer_.append(oa)
        return pred_, origin_answer_
    # Group by query_id
    responses_by_id = defaultdict(list)
    
    for idx, response in enumerate(responses):
        query_id = response["query_id"]
        # query_id = idx
        # print(query_id)
        responses_by_id[query_id].append(response)

    pred_, origin_answer_ = [], []

    for query_id, response_group in responses_by_id.items():
        voting_dict = {}
        origin_answer = None

        for resp_id, resp in enumerate(response_group):
            
            pred, oa = get_single_pred_from_box_pattern(resp)

            # Record origin_answer once per query_id
            if origin_answer is None:
                origin_answer = oa

            # Skip samples without a boxed prediction
            if pred is None:
                continue

            if pred not in voting_dict:
                voting_dict[pred] = 0
            if sampled_scores is not None:
                # print(type(sampled_scores))
                try:
                    voting_dict[pred] += sampled_scores[query_id][str(resp["sample_lang_id"])]
                except:

                    print(f"{str(resp['sample_lang_id'])} not in{sampled_scores[query_id]}")
            else:
                voting_dict[pred] += 1

        # If no boxed prediction exists for this query, prediction is None
        if len(voting_dict) == 0:
            pred_max = None
        else:
            if seed is not None:
                random.seed(seed)
            # pred_max = random.choice(list(voting_dict.keys()))
            pred_max = max(voting_dict.items(), key=lambda x: (x[1], random.random()))[0]

        pred_.append(pred_max)
        origin_answer_.append(origin_answer)

    return pred_, origin_answer_

def looks_numeric(s: str) -> bool:
    """
    Check whether a string looks like a pure numeric answer.
    """
    s = s.strip()
    return bool(re.fullmatch(r'[-+]?(?:\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?', s))

def eval_single_lang_answer(
    lang_id: int,
    data_list: List[List[Dict]],
    model_name: str,
    api_key: str,
    correct_all,
    seed: int = None,
    sampled_scores = None
):
    """
    Worker function for a single language.

    Args:
        lang_id    : language index, also used as process index in mp.spawn
        data_list  : list of length num_langs; each item is a list of responses
        model_name : answer judge model name
        api_key    : API key for the answer judge model
        correct_all: Manager().dict; we write {lang_id: [0/1, ...]} into it
    """
    # Initialize answerJudge inside each process to avoid pickling issues.
    answerJudger = answerJudge(model_name=model_name, api_key=api_key)

    responses = data_list[lang_id]
    sampled_scores = sampled_scores[lang_id] if sampled_scores is not None else None
    
    _pred, _origin_answer = get_pred_from_box_pattern(responses, seed, sampled_scores, vote_by_query_id=True)

    correct = []
    for pred, origin_answer in tqdm(
        list(zip(_pred, _origin_answer)),
        total=len(_pred),
        desc=f"Lang {lang_id}"
    ):
        # No prediction at all -> count as incorrect.
        if pred is None:
            correct.append(0)
            continue

        # 1. Handle purely numeric answers first.
        if looks_numeric(pred) and looks_numeric(origin_answer):
            # Use local numeric comparison only; do not call the LLM judge.
            if judge_equal(pred, origin_answer):
                correct.append(1)
            else:
                correct.append(0)
            continue  # Done with this sample.

        # 2. For non-numeric answers, fall back to the LLM judge.
        if answerJudger.get_answer(pred, origin_answer):
            print(f"[Lang {lang_id}] LLM judge: Correct prediction found.")
            correct.append(1)
        else:
            correct.append(0)

    correct_all[lang_id] = correct


def eval_answer(rank: int, data_path: str, model_name: str, api_key: str, seed: int = None, autocap: bool =False, dataset: str = "polymath"):
    """
    Evaluate a single difficulty level (low/medium/high/top).

    Args:
        rank      : 0/1/2/3 -> low/medium/high/top
        data_path : root directory of the dataset
        model_name: answer judge model name
        api_key   : API key for the answer judge model
    """
    sampled_scores = None
    if autocap:
        with open("dataset/polymath/new_RL-qwen-7B_correct_0.8_0.6/autocap_auxiliary/0.4/auxiliary_set_scores.json", 
                    'r', 
                    encoding='utf-8') as f:
            sampled_scores = json.load(f)
    

    diff_list = ["low", "medium", "high", "top"]
    diff = diff_list[rank]
    lang_list = polymath_LANG_LIST if dataset == "polymath" else mmlu_LANG_LIST
    if dataset == "mmlu":
        diff = "single"
        # print(diff)
    if sampled_scores is not None:
        sampled_scores = sampled_scores[diff]
        sampled_scores = np.array(sampled_scores).T.tolist()  # Transpose to 18 x 125 x 6.
    # print("!")
    data_all = []  # [num_langs][list_of_responses]
    # Here we assume there are 16 languages and directory structure:
    # data_path/diff/{lang_id}/0.jsonl
    for lang_id in tqdm(range(len(lang_list)), desc="Loading jsonl"):
    # for lang_id in range(len(lang_list)):
        file_path = os.path.join(data_path, diff, str(lang_id), "0.jsonl")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            # continue
        responses = read_jsonl_file(file_path=file_path)
        data_all.append(responses)
        
    # correct_all: {lang_id: [0/1, ...]}
    with Manager() as manager:
        correct_all = manager.dict()

        mp.spawn(
            eval_single_lang_answer,
            args=(data_all, model_name, api_key, correct_all, seed, sampled_scores),
            nprocs=len(data_all),
            join=True
        )

        # Convert manager dict to a normal list ordered by lang_id
        correct_all_list = [correct_all[i] for i in range(len(data_all))]

    # Compute per-language accuracy
    lang_accuracy = [
        (sum(c) / len(c) if len(c) > 0 else 0.0)
        for c in correct_all_list
    ]

    # Compute overall accuracy
    total_correct = sum(sum(c) for c in correct_all_list)
    total_count = sum(len(c) for c in correct_all_list)
    overall_accuracy = total_correct / total_count if total_count > 0 else 0.0

    # Write results to file
    output_dir = os.path.join(data_path, diff)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "accuracy_0.jsonl")
    data_to_save = {}
    for lang_id, lang in enumerate(lang_list):
        try:
            data_to_save[f"Accuracy for {lang}"] = lang_accuracy[lang_id]
        except IndexError:
            data_to_save[f"Accuracy for {lang}"] = 0.0
    data_to_save["Overall accuracy"] = overall_accuracy
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, indent=4, ensure_ascii=False)


def parallel_eval(
    data_path: str,
    model_name: str = "deepseek-reasoner",
    api_key: str = "sk-xxx",
    single_mode: str | None = "high",
    seed: int = None,
    autocap: bool = False,
    dataset: str = "polymath"
):
    """
    Top-level evaluation entry.

    Args:
        data_path  : root directory of the dataset
        model_name : answer judge model name
        api_key    : API key for the answer judge model
        single_mode: one of {"low","medium","high","top"} or None.
                     - If set to a difficulty string, only that difficulty
                       will be evaluated.
                     - If None, all difficulties are evaluated sequentially.
    """
    diff_list = ["low", "medium", "high", "top"]

    if single_mode is not None:

        if single_mode not in diff_list and dataset != "mmlu":
            raise ValueError(
                f"single_mode must be one of {diff_list} or None, got {single_mode}"
            )
        rank = 0
        if dataset == "polymath":
            rank = diff_list.index(single_mode)
        print(f"[Eval] Only evaluate difficulty: {single_mode}")
        eval_answer(rank, data_path, model_name, api_key, seed, autocap, dataset)
    else:
        print("[Eval] Evaluate all difficulties: low, medium, high, top")
        for rank, diff in enumerate(diff_list):
            print(f"\n===== Evaluating difficulty: {diff} =====")
            eval_answer(rank, data_path, model_name, api_key, seed, autocap)


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("true", "1", "yes", "y", "t"):
        return True
    if v in ("false", "0", "no", "n", "f"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

if __name__ == "__main__":
    # Use "spawn" to be compatible with most platforms (including Windows).
    mp.set_start_method("spawn", force=True)

    argparser = argparse.ArgumentParser(
        prog="eval_answer.py",
        description="Evaluate the answers generated by the model on the dataset."
    )
    argparser.add_argument(
        "--data_path",
        type=str,
        default="./dataset/polymath/ds1.5b/output/",
        help="Path to the dataset directory."
    )
    argparser.add_argument(
        "--model_name",
        type=str,
        default="deepseek-reasoner",
        help="Name of the answer judging model."
    )
    argparser.add_argument(
        "--api_key",
        type=str,
        default="sk-xxx",
        help="API key for the answer judging model."
    )
    argparser.add_argument(
        "--single_mode",
        type=str,
        default="high",
        help=(
            'One of {"low","medium","high","top"} or "None". '
            'If specified, only evaluate this difficulty level. '
            'If set to "None" (string), all difficulties are evaluated.'
        )
    )
    argparser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )
    argparser.add_argument("--autocap", type=str2bool, default=False)
    argparser.add_argument("--dataset", type=str, default="polymath", help="Dataset name (e.g., 'polymath' or 'mmlu').")

    args = argparser.parse_args()

    # Also support string "None" from CLI to mean Python None.
    single_mode = args.single_mode
    if isinstance(single_mode, str) and single_mode.lower() == "none":
        single_mode = None

    print("Use AutoCAP auxiliary scores:", args.autocap)
    parallel_eval(
        data_path=args.data_path,
        model_name=args.model_name,
        api_key=args.api_key,
        single_mode=single_mode,
        seed = args.seed,
        autocap = args.autocap,
        dataset = args.dataset
    )
