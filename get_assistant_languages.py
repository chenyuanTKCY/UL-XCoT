########################################################################
#
# @author : Chenyuan Zhang
# @when : Winter Semester 2025/2026
# @where : Harbin Institute of Technology
# @title : Get Assistant Languages
# @component: root
# @file : get_assistant_langugages.py
#
########################################################################

import torch
import argparse
import json
import numpy as np
import torch.multiprocessing as mp
from utils.config import language_model_dir, language_sample_dir,  mgsm_LANG_DICT, \
                        model_initialization, model_initialization_parallel, \
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


def sample_language_with_model(model_path_use, language_matrix_set: List = None, language_router: LanguageRouter = None, sampling_size:int = 3, 
                               logic_result_path:str = "./dataset/polymath/routing_results_polymath.json", device:int = 0, get_similar_languages_mode : str = "batch", dataset: str = "polymath") -> None:
    '''
        This function is used to search the language for sampling in inference.
    '''
    ## SETUP: Get the similar languages of logic wth the same context

    # Initialize the language model and tokenizer
    _model, _tokenizer = model_initialization_parallel(model_path_use = model_path_use ,sampling_size=1, mode='same', indicated_device=device)

    # Initialize the language router
    model_ = _model[0]
    tokenizer_ = _tokenizer[0]
    language_router = LanguageRouter(model = model_, tokenizer = tokenizer_)

    # Read the language set
    if dataset == "polymath":
        language_matrix_set = read_polymath(mode="sample")
        # print(len(language_matrix_set[0]))
    elif dataset == "MMLU":
        language_matrix_set = read_MMLU(mode="sample")

    # print(np.array(language_matrix_set).shape)
    # sampled_quardruple has 4 elements: similar_languages, ma, ms, gamma
    sampled_quardruple = language_router.route(language_matrix_set=language_matrix_set, sampling_size= sampling_size, get_similar_languages_mode=get_similar_languages_mode, dataset=dataset)
    def _to_jsonable(o):
        # Torch / numpy scalar -> Python int/float
        if hasattr(o, "item"):
            return o.item()

        # numpy array or torch tensor -> Python list
        if hasattr(o, "tolist"):
            return o.tolist()

        return str(o)
    # write to json
    with open(logic_result_path, 'w', encoding='utf-8') as f:
        json.dump({
            "similar_languages": sampled_quardruple.similar_languages,
            "ma": sampled_quardruple.ma,
            "ms": sampled_quardruple.ms,
            "gamma": sampled_quardruple.gamma
        }, f, ensure_ascii=False, indent=4, default=_to_jsonable)



if __name__ == "__main__":
    argparser = argparse.ArgumentParser(prog = "get_assistant_languages", 
                                        description = "Get the assistant languages with linguistic difference, " \
                                        "linguistic coefficency and share explanation.")
    argparser.add_argument("--sampling_size", type=int, default=3, help="The number of  languages for inference.")
    argparser.add_argument("--logic_result_path", type=str, default="./dataset/polymath/routing_results_polymath.json", help="The path of the logic routing result file.")
    argparser.add_argument("--model_path", type=str, default=language_model_dir, help="The path of the language model.")
    argparser.add_argument("--device", type=int, default=0, help="The device for model inference.")
    argparser.add_argument("--get_similar_languages_mode", type=str, default="batch", help="The mode for getting similar languages, options are 'batch' or 'single'.")
    argparser.add_argument("--dataset", type=str, default="polymath", help="The dataset for getting similar languages, options are 'polymath' or 'mmlu'.")
    args = argparser.parse_args()
    sample_language_with_model(
        sampling_size= args.sampling_size,
        device = args.device,
        model_path_use = args.model_path,
        logic_result_path= args.logic_result_path,
        get_similar_languages_mode= args.get_similar_languages_mode,
        dataset = args.dataset
    )