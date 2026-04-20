########################################################################
#
# @author : Chenyuan Zhang
# @when : Winter Semester 2025/2026
# @where : Harbin Institute of Technology
# @title : Get Cost
# @component: root
# @file : get_cost.py
#
########################################################################

import re
import os
import json
import ast
import argparse
import numpy as np
from typing import List, Dict
from utils.answer_judge import answerJudge
from utils.file_processor import read_pkl_file
from utils.config import polymath_LANG_DICT, polymath_LANG_LIST, mmlu_LANG_LIST
from tqdm import tqdm


def get_cost(all_path: str, diff: str, dataset:str="polymath"):
    the_whole_latency, the_whole_token_length = [], []
    lang_list = polymath_LANG_LIST if dataset == "polymath" else mmlu_LANG_LIST
    # if dataset == "mmlu":

    item_record = {}
    for lang_id in tqdm(range(len(lang_list))):
        path = f"{all_path}{diff}/{lang_id}/inference_data_all_0.jsonl"
        # 如果判断不存在则跳过
        if not os.path.exists(path):
            continue
        all_latency, all_token_length = [], []
        with open(path, 'r', encoding='utf-8') as f:
            for _, line in enumerate(f, 1):
                _dict = json.loads(line)
                all_latency.append(_dict['all_latency'])
                all_token_length.append(_dict['all_token_lengths'])
            avg_latency = np.mean(np.array(all_latency))
            avg_token_length = np.mean(np.array(all_token_length))
            write_path = f"{all_path}{diff}/{lang_id}/cost_overall.txt"
            with open(write_path, 'w', encoding='utf-8') as wf:
                wf.write(f"Language: {lang_list[lang_id]}, Avg Latency: {avg_latency}, Avg Token Length: {avg_token_length}\n")
            item_record[lang_list[lang_id]] = f"Avg Latency: {avg_latency}, Avg Token Length: {avg_token_length}\n"
            the_whole_latency.append(avg_latency)
            the_whole_token_length.append(avg_token_length)
    overall_write_path = f"{all_path}{diff}/cost_overall.txt"
    the_whole_avg_latency = np.mean(np.array(the_whole_latency))
    the_whole_avg_token_length = np.mean(np.array(the_whole_token_length))
    with open(overall_write_path, 'w', encoding='utf-8') as wf:
        for lang in lang_list:
            if lang not in list(item_record.keys()):
                continue
            wf.write(f"Language: {lang}, {item_record[lang]}")
        wf.write(f"Overall Avg Latency: {the_whole_avg_latency}, Overall Avg Token Length: {the_whole_avg_token_length}\n")
        # item_record['avg_latency'] = the_whole_avg_latency
        # item_record['avg_token_length'] = the_whole_avg_token_length
        # f"Avg Latency: {the_whole_avg_latency}, Avg Token Length: {the_whole_avg_token_length}"
        # wf.write(json.dumps(item_record ,ensure_ascii=False) + "\n")

def get_all_cost(all_path:str):
    for diff in ["low", "medium", "high", "top"]:
        get_cost(all_path, diff)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        prog = "get_cost",
        description = "get latency cost and token length for different languages",
    )
    argparser.add_argument(
        "--path",
        type=str,
        default="./dataset/polymath/ds1.5b/new_output/",
        help="the path to the latency cost file",
    )
    argparser.add_argument(
        "--mode",
        type=str,
        default="single",
        help="the mode to run the script",
    )
    argparser.add_argument(
        "--diff",
        type=str,
        default="high",
        help="the mode to run the script",
    )
    argparser.add_argument(
        "--dataset",
        type=str,
        default="polymath",
        help="the dataset to run the script",
    )

    args = argparser.parse_args()

    if args.mode == "single":
        get_cost(argparser.parse_args().path, argparser.parse_args().diff, argparser.parse_args().dataset)
    else:
       get_all_cost(argparser.parse_args().path)