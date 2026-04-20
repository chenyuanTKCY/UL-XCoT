########################################################################
#
# @author : Chenyuan Zhang
# @when : Winter Semester 2025/2026
# @where : Harbin Institute of Technology
# @title : Process Query
# @component: root
# @file : process_query.py
#
########################################################################


import torch
import argparse
import random
import math
import json
import torch.multiprocessing as mp
from utils.config import language_model_dir, language_sample_dir,  mgsm_LANG_DICT, polymath_LANG_DICT, mmlu_LANG_DICT, \
                        model_initialization, model_initialization_parallel, mmlu_LANG_LIST, \
                        polymath_output_path, polymath_LANG_LIST
from utils.language_router import LanguageRouter, RoutingResults
from utils.file_processor import read_allfiles_in_dir_byline, read_allfiles_in_dir,\
                                read_file_line_by_line, read_polymath, read_MMLU
from utils.inference_utils import ModelInference
from typing import Tuple, List
from copy import deepcopy, copy
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer


def infer_language_with_model(
    inference_model: ModelInference = None,
    language_query_set: List = None,
    sampled_quardruple: Dict = None,
    test_mode: str = "ours",
    diff: str = "low",
    group_id: int = 0,
    sampling_size: int = 3,
    lamda: float = 0.4,
    seed: int = 42,
    prune_ratio: float = 0.6,
    window_size_scaling: float = 3.0,
    max_token_nums: int = 2048,
    dataset:str = "polymath"
):
    """
    Extend each query with sampled multilingual variants and perform inference.
    Handles both direct inference and sampled language expansion.
    """

    sampled_languages = sampled_quardruple.get("similar_languages")
    ms = sampled_quardruple.get("ms")
    gamma = sampled_quardruple.get("gamma")
    
    if sampled_languages is not None:
        extended_queries = []
        if test_mode == "orign" or test_mode == "raw":
            sampled_languages = [[i] for i in range(len(polymath_LANG_LIST))]
        elif test_mode == "CLSP" or test_mode == "CLSP_cost":
            if dataset == "polymath":
                 sampled_languages = [[i for i in range(len(polymath_LANG_LIST)) if i != j] 
                     for j in range(len(polymath_LANG_LIST))]
            elif dataset == "MMLU":
                sampled_languages = [[i for i in range(len(mmlu_LANG_LIST)) if i != j] 
                        for j in range(len(mmlu_LANG_LIST))]
            print(f"----------------------++++++++++++++++++++{test_mode}+++++++++++++++++++++-----------------------")
        elif test_mode == "self_consistency" or test_mode == "self_consistency_acc" or test_mode == "self_consistency_cost" or test_mode == "SC_mo":
            num_languages = len(polymath_LANG_LIST)
            sampled_languages = [[i] * sampling_size for i in range(num_languages)]
            print(sampled_languages)
            # print(f"----------------------++++++++++++++++++++{test_mode}+++++++++++++++++++++-----------------------")
        elif test_mode == "ablation_top_k":
            sampled_languages = [[i for i in range(len(polymath_LANG_LIST)) if i != j] 
                     for j in range(len(polymath_LANG_LIST))]
        elif test_mode == "autocap_auxiliary":
            sampled_languages = [[i for i in range(len(polymath_LANG_LIST))] 
                     for j in range(len(polymath_LANG_LIST))]
        elif test_mode == "autocap" or test_mode == "autocap_cost":
            with open("dataset/polymath/new_RL-qwen-7B_correct_0.8_0.6/autocap_auxiliary_new/10240/auxiliary_set.json", 
                      'r', 
                      encoding='utf-8') as f:
                sampled_languages = json.load(f)
        elif test_mode == "translate_to_EN":
            sampled_languages = [[i] for i in range(len(polymath_LANG_LIST))]
        elif test_mode == "translate_to_EN_sc":
            num_languages = len(polymath_LANG_LIST)
            sampled_languages = [[i] * sampling_size for i in range(num_languages)]           

        if type(sampled_languages) is list:
            for query_id, query in enumerate(language_query_set):
                # Deep copy to avoid overwriting base structure
                new_query = deepcopy(query)

                # For each language in the sampled group, extend with related languages' text
                for lang_idx, lang_samples in enumerate(sampled_languages):
                    # print(f"------------------------LENGTH{len(lang_samples)}---------------------------")
                    new_query[lang_idx]=[query[lang][0] for lang in lang_samples[:sampling_size]]
                    # print(f"------------------------LENGTH{len(new_query[lang_idx])}---------------------------")
                extended_queries.append(new_query)
        else:
            sampled_languages = sampled_languages[diff]
            for query_id, query in enumerate(language_query_set):
                new_query = deepcopy(query)
                # print(len(sampled_languages))
                # print(len(sampled_languages[query_id]))
                for lang_idx, lang_samples in enumerate(sampled_languages[query_id]):
                    if test_mode == "ours" or test_mode == "autocap"  or test_mode == "CLSP_acc" or "autocap_cost":
                        new_query[lang_idx] = [query[lang][0] for lang in lang_samples[:sampling_size]]
                    elif test_mode == "ablation_random":
                        keep_size = math.ceil(sampling_size * (1-prune_ratio))
                        # print(prune_ratio)
                        # use  random sampling to select languages with seed
                        random.seed(seed)
                        sampled_langs = random.sample(lang_samples, keep_size)
                        new_query[lang_idx] = [query[lang][0] for lang in sampled_langs]



                extended_queries.append(new_query)

        # Run model inference with extended queries
        inference_model.inference(
        language_query_set=extended_queries,
        ms_components=ms,
        gamma_components=gamma,
        mode=test_mode,
        diff = diff,
        group_id = group_id,
        max_token_nums=max_token_nums,
        lamda=lamda,
        seed=seed,
        prune_ratio=prune_ratio,
        window_size_scaling=window_size_scaling,
        sampling_size=sampling_size,
        dataset = dataset
        )



def run_inference(model_path, sampled_quardruple, 
                  test_mode,  query_diff, data_infer,output_path, 
                  sampling_size, temperature=0.8, top_p=0.95, lamda=0.4, seed=42,prune_ratio=0.6, window_size_scaling=3.0, max_token_nums=2048, dataset="polymath"):
    # print(f"------------------------LENGTH{sampling_size}---------------------------")
    language_dict = polymath_LANG_DICT if dataset == "polymath" else mmlu_LANG_DICT
    inference_model = ModelInference(model=model_path, 
                                     enable_prefix_caching=True,
                                     temperature=temperature,
                                     top_p=top_p,
                                     language_dict=language_dict,
                                     output=output_path, 
                                     max_model_len=20000,
                                     )
    infer_language_with_model(
        inference_model=inference_model,
        language_query_set=data_infer,
        sampled_quardruple=sampled_quardruple,
        test_mode=test_mode,
        diff=query_diff,
        group_id=0,
        sampling_size=sampling_size,
        lamda=lamda,
        seed=seed,
        prune_ratio=prune_ratio,
        window_size_scaling=window_size_scaling,
        max_token_nums=max_token_nums,
        dataset=dataset
    )




def sequential_inference( model_path, sampled_quardruple, sampling_size=3, 
                         test_mode="ours", group_num=5, query_diff="top",
                         output_path = polymath_output_path, temperature=0.8, top_p=0.95, lamda=0.4, seed=42, prune_ratio=0.6, window_size_scaling=3.0, max_token_nums=2048, dataset="polymath"):
    if dataset == "polymath":

        language_query_set = read_polymath(mode="infer", diff=query_diff)
        if test_mode == "translate_to_EN" or test_mode == "translate_to_EN_sc":
            language_query_set = read_polymath(mode="infer", diff=query_diff, translate_to_en=True)
    elif dataset == "MMLU":
        language_query_set = read_MMLU(mode="infer")
    run_inference(model_path=model_path, 
                  sampled_quardruple = sampled_quardruple, 
                  test_mode=test_mode, 
                  query_diff=query_diff,
                  data_infer=language_query_set,
                  output_path=output_path,
                  sampling_size=sampling_size,
                  temperature=temperature,
                  top_p=top_p,
                  lamda=lamda,
                  seed=seed,
                  prune_ratio=prune_ratio,
                  window_size_scaling=window_size_scaling,
                  max_token_nums=max_token_nums,
                  dataset=dataset)


if __name__ == "__main__":

    argparser = argparse.ArgumentParser(prog = "process_query",description="Process Query for Dataset")
    argparser.add_argument("--logic_result_path", type=str, default="./dataset/polymath/routing_results_polymath.json", help="The path of the logic routing result file.")
    argparser.add_argument("--test_mode", type=str, default="ours", help="The test mode for inference, options are 'ours' or 'orign' or 'CLSP'.")
    argparser.add_argument("--group_num", type=int, default=5, help="The number of groups for inference.")
    argparser.add_argument("--sampling_size", type=int, default=3, help="The number of sampled languages for inference.")
    # argparser.add_argument("--devices", type=str, default="5,5,5,5,5", help="The list of devices for parallel inference, separated by commas.")
    argparser.add_argument("--model_path", type=str, default=language_model_dir, help="The path of the language model.")
    argparser.add_argument("--output_path", type=str, default=polymath_output_path, help="The output path for inference results.")
    argparser.add_argument("--query_diff", type=str, default="top", help="The difficulty level of the query set.")
    argparser.add_argument("--temperature", type=float, default=0.8, help="The temperature for sampling during inference.")
    argparser.add_argument("--top_p", type=float, default=0.95, help="The top_p value for nucleus sampling during inference.")
    argparser.add_argument("--lamda", type=float, default=0.4, help="The lamda parameter for inference.")
    argparser.add_argument("--seed", type=int, default=42, help="The random seed for inference.")
    argparser.add_argument("--prune_ratio", type=float, default=0.6, help="The prune ratio for inference.")
    argparser.add_argument("--window_size_scaling", type=float, default=3.0, help="The window size scaling factor for inference.")
    argparser.add_argument("--max_token_nums", type=int, default=2048, help="The maximum number of tokens for inference.")
    argparser.add_argument("--dataset", type=str, default="polymath", help="The dataset for inference, options are 'polymath' or 'MMLU'.")
    
    
    args = argparser.parse_args()

    if args.logic_result_path:
        ## INFER: ST-BoN for an efficient inference adding the sampled auxilary languages
        with open(args.logic_result_path, 'r', encoding='utf-8') as f:
            sampled_quardruple = json.load(f)
        # print(sampled_quardruple.get("similar_languages"))
        sequential_inference(model_path= args.model_path, 
                            sampled_quardruple =sampled_quardruple,
                            sampling_size=args.sampling_size,
                            test_mode=args.test_mode,
                            group_num=args.group_num,
                            query_diff=args.query_diff,
                            output_path = args.output_path,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            lamda=args.lamda,
                            seed=args.seed,
                            prune_ratio=args.prune_ratio,
                            window_size_scaling=args.window_size_scaling,
                            max_token_nums=args.max_token_nums,
                            dataset=args.dataset)
    else:
        print("[Warning] Please provide the logic routing result file path! ")

    

    
