########################################################################
#
# @author : Chenyuan Zhang
# @when : Winter Semester 2025/2026
# @where : Harbin Institute of Technology
# @title : Language Router
# @component: utils
# @file : language_router.py
#
########################################################################

import torch
import numpy as np
import time
from transformers import PreTrainedModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, List, Union
from .hidden_states_getter import ModelBasedProcessor, ApiBasedProcessor
from .config import ms_save_dir
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from .expression_processor import expression_split, get_similar_languages
from.file_processor import read_polymath, read_MMLU
from dataclasses import dataclass




class LanguageRouter:
    def __init__(self, 
                 model: Optional[AutoModelForCausalLM] = None,
                 tokenizer: Optional[AutoTokenizer] = None,
                 api_key: Optional[str] = None,
                 api_base: Optional[str] = None) -> None:
        self._processor = self._initialize_processor(model, tokenizer, api_key, api_base)



    # Initialize the language processing model or API client
    def _initialize_processor(self, model: Optional[AutoModelForCausalLM],tokenizer: Optional[AutoTokenizer], api_key: Optional[str], api_base: Optional[str]) -> Union[ModelBasedProcessor, ApiBasedProcessor]:
        
        if model and tokenizer:
            return ModelBasedProcessor(model = model, tokenizer = tokenizer)
        elif api_key and api_base:
            return ApiBasedProcessor(api_key = api_key, api_base = api_base)
        else:
            raise ValueError("Either a model or API key and base URL must be provided.")
    


    # Route the language matrix set to the most similar languages
    def route(self, language_matrix_set: List, sampling_size: int = 3, lamda: float = 0.5, get_similar_languages_mode : str = "batch", dataset = "polymath") -> List:

        '''Route the language matrix set to the most similar languages
        Args:
            language_matrix_set (List): The language matrix set to be used.
            language_query_set (List): The language query set to be routed.
            sampling_size (int, necessary): The number of similar languages to be returned. Defaults to 3.

        Returns:
            List: The List of similar languages.
        '''
        # Safety check for sampling size
        if sampling_size <= 0:
            raise ValueError("Sampling size must be a positive integer.")
        


        # Step 1 Get m, each ma, ms, gamma at every layer
        m = self._processor.get_layer_hidden_states(language_matrix_set)
        m = m.mean(dim=3)
        ma, gamma, ms = [], [], []

        # split the final_token_expression into ma and ms
        for m_layer in tqdm(m, desc="Splitting language matrix layers"):
            ma_layer, ms_layer, gamma_layer = expression_split(m_layer)
            ma.append(ma_layer)
            ms.append(ms_layer)
            gamma.append(gamma_layer)
        



        # Step 2 Utilize ms to substract the logic space of final token expression in order to sampling
        # get the final token expression
        # [layer_num-1, d_model, lang_num, lang_sample]
        if get_similar_languages_mode == "batch":
            final_token_expression = self._processor.get_layer_hidden_states(language_matrix_set= language_matrix_set, ms_components = ms)
            final_token_expression = final_token_expression[-15].squeeze(0) # [d, l, lang_sample]
            final_token_expression = final_token_expression.mean(dim=2)  # [d, l]
            # save final_token_expression
            # torch.save(final_token_expression, "./results/linguistic_embedding/final_token_expression.pt")
            # get the similar languages
            similar_languages = get_similar_languages(final_token_expression, sampling_size = sampling_size)

        else:
            similar_languages = {}
            # Polymath read logic
            if dataset == "polymath":
                for diff in ["low", "medium", "high", "top"]:
                    temp_similar_languages = []
                    # [125, lang]
                    single_dataset = read_polymath(mode="sample_single", diff=diff)
                    single_dataset = np.array(single_dataset).T.tolist()
                    final_token_expression = self._processor.get_layer_hidden_states(language_matrix_set= single_dataset, ms_components = ms)
                    if diff == "low":
                        torch.save(final_token_expression, "./results/linguistic_embedding/final_token_expression_None.pt")            
                    final_token_expression = final_token_expression[12].squeeze(0) # [d,  l, lang_sample]
                    final_token_expression = final_token_expression.permute(2,0,1)  # [lang_sample, d, l]
                    # print(final_token_expression.shape)
                    # print(final_token_expression.size(0))
                    for i in range(final_token_expression.size(0)):
                        single_final_token_expression = final_token_expression[i]
                        single_similar_languages = get_similar_languages(single_final_token_expression, sampling_size = sampling_size)
                        temp_similar_languages.append(single_similar_languages)
                    similar_languages[diff] = temp_similar_languages
            elif dataset == "MMLU":
                for diff in ["single"]:
                    temp_similar_languages = []
                    # [125, lang]
                    single_dataset = read_MMLU(mode="sample_single")
                    single_dataset = np.array(single_dataset).T.tolist()
                    final_token_expression = self._processor.get_layer_hidden_states(language_matrix_set= single_dataset, ms_components = ms)
                    if diff == "low":
                        torch.save(final_token_expression, "./results/linguistic_embedding/final_token_expression_None.pt")            
                    final_token_expression = final_token_expression[-15].squeeze(0) # [d,  l, lang_sample]
                    final_token_expression = final_token_expression.permute(2,0,1)  # [lang_sample, d, l]
                    # print(final_token_expression.shape)
                    # print(final_token_expression.size(0))
                    for i in range(final_token_expression.size(0)):
                        single_final_token_expression = final_token_expression[i]
                        single_similar_languages = get_similar_languages(single_final_token_expression, sampling_size = sampling_size)
                        temp_similar_languages.append(single_similar_languages)
                    similar_languages[diff] = temp_similar_languages
        # Step 3 Save routing quadruple
        return RoutingResults(
            similar_languages=similar_languages,
            ma=ma,
            ms=ms,
            gamma=gamma
        )

@dataclass
class RoutingResults:
    '''
        Class to store the results of the language routing process.
    '''
    similar_languages: List
    ma: List
    ms: List
    gamma: List