########################################################################
#
# @author : Chenyuan Zhang
# @when : Winter Semester 2025/2026
# @where : Harbin Institute of Technology
# @title : Use model or api to get hidden states
# @component: utils
# @file : hidden_states_getter.py
#
########################################################################


import torch
import time
import json
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from .config  import ms_save_dir, gamma_save_dir, ma_diff_save_dir, mgsm_LANG_DICT, mgsm_output_path
from .expression_processor import expression_split, get_similar_languages, HookModule
from transformers import PreTrainedModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from typing import Optional, List, Tuple, Union
from queue import Queue
from threading import Thread, Lock
from openai import OpenAI
from scipy.linalg import svd
from abc import ABC, abstractmethod

from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


class ModelBasedProcessor:
    '''
        Class for processing expressions using a language model.
    '''

    def __init__(self, model: AutoModelForCausalLM | List = None, tokenizer: AutoTokenizer | List = None):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = 8
    

    def _get_transformer_layer_nums(self) -> int:
        '''
            Get the number of transformer layers in the model.
        '''
        return len(self.model.model.layers)
    
    def get_layer_hidden_states(self, language_matrix_set: List, ms_components: List = None) -> torch.Tensor:
        '''
            Get the hidden states from each layer for the language matrix set.
        '''
        hooks = HookModule(self.model)
        if ms_components is not None:
            middle_layers = list(range(2, 27))
            # print(len(ms_components[0]))
            ms_components = torch.tensor(ms_components).to(self.model.device)
            hooks.add_layer_hooks(middle_layers, lambda h, i: hooks.middle_layer_logic_extract(h, i, ms_components, lamda=0.75))
        
        self.model.eval()
        all_layer_hidden_states_all_list = []
            
        with torch.no_grad():

            # Process each language block in the language matrix set
            progress_bar =tqdm(language_matrix_set, desc="get layer hidden states")
            for lang_block in progress_bar:
                if type(lang_block[0]) is dict:
                    lan = lang_block[0]['lang_id']
                else:
                    lan = lang_block[0][0]['lang_id']
                progress_bar.set_description(f"Processing lang {lan}")
                texts = [item['text'] if type(item) == dict else item[0]['text'] for item in lang_block]
                
                # Record [l_i, num_layer, d_model]
                processed_hidden_states = [None] * (self._get_transformer_layer_nums() + 1)

                # Process texts in batches
                for i in tqdm(range(0, len(texts), self.batch_size), desc = "Processing Batches", leave=False):
                    batch_texts = texts[i:i+self.batch_size]
                    
                    # Tokenize the batch texts
                    inputs = self.tokenizer(batch_texts,
                                            return_tensors='pt',
                                            padding = True,
                                            truncation = True,
                                            max_length = 1024,).to(self.model.device)

                    attention_mask = inputs['attention_mask']  # [batch, seq_len]
                    # Get model outputs with hidden states
                    outputs = self.model(
                        **inputs,
                        output_hidden_states=True,
                        return_dict=True
                    )

                    hidden_states = outputs.hidden_states  # Tuple of (layer_num + 1) elements
                    
                    # Collect hidden states from each layer
                    for layer_idx in range(len(hidden_states)):
                        
                        # Get the final token hidden state for the batch

                        layer_hidden = hidden_states[layer_idx]
                        last_token_positions = attention_mask.sum(dim=1) - 1  # The positions of the last tokens
                        # [batch_size, hidden_dim]
                        layer_hidden = layer_hidden[torch.arange(layer_hidden.size(0)), last_token_positions]  
                        if processed_hidden_states[layer_idx] is None:
                            processed_hidden_states[layer_idx] = layer_hidden # Update with last token hidden states
                        else:
                           processed_hidden_states[layer_idx] = torch.cat((processed_hidden_states[layer_idx], layer_hidden), dim=0)
                    
                hidden_states_tensor = torch.stack(processed_hidden_states, dim=0)  # [layer_num, batch_size, d_model]
                hidden_states_tensor = hidden_states_tensor[1:] # [layer_num-1, lang_sample, d_model]
                # hidden_states_tensor_mean = hidden_states_tensor.mean(dim=1)  # Mean over the sample dimension [layer_num, d_model]
                
                all_layer_hidden_states_all_list.append(hidden_states_tensor)
        # [lang_id, layer_num-1, lang_sample, d_model]
        all_layer_hidden_states_all = torch.stack(all_layer_hidden_states_all_list, dim=0).permute(1,3,0,2)  # [layer_num-1, d_model, lang_num, lang_sample]
        
        hooks.remove_hooks()
        return all_layer_hidden_states_all
        


class ApiBasedProcessor:
    '''
        Class for processing expressions using an API client.
    '''

    def __init__(self, api_key: str, api_base: str):
        self.api_key = api_key
        self.api_base = api_base
    def get_layer_hidden_states(self, language_matrix_set: List, ms_components: List = None) -> List:
        '''
            Get the final token expression from the language matrix set.
        '''

        for idx in range(len(language_matrix_set)):
            for idy in range(len(language_matrix_set[idx])):
                sequences = language_matrix_set[idx][idy]['text']
                
                # Here We get the final token expression from the API

        return []

