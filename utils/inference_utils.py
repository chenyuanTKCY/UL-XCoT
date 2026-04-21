########################################################################
#
# @author : Chenyuan Zhang
# @when : Winter Semester 2025/2026
# @where : Harbin Institute of Technology
# @title : Multilingual Efficient Inference 
# @component: utils
# @file : inference_utils.py
#
########################################################################

import torch
import time
import math
import pickle
import threading
import json
import os
import numpy as np
from .config  import ms_save_dir, gamma_save_dir, \
ma_diff_save_dir, mgsm_LANG_DICT, mgsm_output_path, polymath_output_path, polymath_LANG_DICT, polymath_LANG_NOTE_DICT
from .early_stop import VoteManager, ConfPerReqLogitsProcessor, LogicObject
from typing import Optional, List, Tuple, Union, Dict
from collections import defaultdict
from queue import Queue
from threading import Thread, Lock, Barrier
from openai import OpenAI
from scipy.linalg import svd
from abc import ABC, abstractmethod

from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

#--------------------------------------------------------------#
import copy
from vllm import LLM, SamplingParams
from typing import List, Dict, Any, Optional
from .synthesis_output import SynthesisOutput
#--------------------------------------------------------------#


class ModelInference():
    '''
        This class is used to handle the model inference for multilingual tasks
    '''
    def __init__(self, 
                model: str,
                output: str,
                temperature: float = 0.8,
                top_p: float = 0.95,
                language_dict: Dict = None,
                **vllm_kwargs) -> None:
        self.model = model
        self.lang_dict = language_dict
        self.output = output
        self.temperature = temperature
        self.top_p = top_p
        self.model_name = model
        self.vllm_kwargs = vllm_kwargs
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        
        # Initialize vLLM
        default_kwargs = {
            "tensor_parallel_size": len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")),
            "enable_prefix_caching": True,
            "trust_remote_code": True,
            "use_v2_block_manager":True, 
            "enforce_eager": True,  # 确保不使用 CUDA Graph，避免与 hook 冲突
        }
        print(f"---------------------{default_kwargs['tensor_parallel_size']} GPU(s) will be used for vLLM inference---------------------")
        default_kwargs.update(vllm_kwargs)

        print("Initializing vLLM engine...")
        llm_init_start = time.time()
        self.llm = LLM(model=model, **default_kwargs)
        llm_init_time = time.time() - llm_init_start
        print(f"vLLM engine initialized in {llm_init_time:.2f} seconds")
        
        # Initialize tokenizer
        print("Initializing tokenizer...")
        tokenizer_init_start = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        tokenizer_init_time = time.time() - tokenizer_init_start
        print(f"Tokenizer initialized in {tokenizer_init_time:.2f} seconds")
        
        # Store initialization times
        self.init_times = {
            'llm_init_time': llm_init_time,
            'tokenizer_init_time': tokenizer_init_time
        }
    


    def inference(self, language_query_set: List = None, 
                  sampled_set: List = None, ms_components: List = None, 
                  gamma_components: List =None, mode:str = "ours", diff:str = "high", 
                  group_id:int = 0, max_token_nums: int = 1024, lamda: float = 0.4, seed: int = 42, prune_ratio: float = 0.6, window_size_scaling: int = 3, sampling_size:int = 3, dataset:str = "polymath"):
        # language_query_set = np.array(language_query_set).transpose(1,0,2).tolist()
        try:
            test_arr = np.array(language_query_set, dtype=object)
            print(f"形状检查: {test_arr.shape}")
            
            language_query_set = np.array(language_query_set).transpose(1,0,2).tolist()
        except ValueError as e:
            print(f"数组转换错误: {e}")
            print(f"数据类型: {type(language_query_set)}")
        BATCH = 125
        BATCH_MODES = {"self_consistency", "self_consistency_acc", "CLSP_acc", "autocap_auxiliary", "raw", "autocap", "origin", "translate_to_EN"}

        def chunks(lst, n):
            for i in range(0, len(lst), n):
                yield i, lst[i:i+n]   # 返回 (batch_start_id, batch_items)

        for lang_id, queries in tqdm(enumerate(language_query_set),
                                    total=len(language_query_set),
                                    desc="Processing language queries"):
            if mode == "autocap_auxiliary" and lang_id > 0:
                break
            if (mode == "translate_to_EN" or mode == "translate_to_EN_sc") and (lang_id == 3 or lang_id < 2):
                continue

            if mode in BATCH_MODES:
                if mode == "CLSP_acc":
                    queries = [queries[i] for i in range(len(queries)) if i % 10 == 1]
                for batch_start_id, batch in tqdm(list(chunks(queries, BATCH)),
                                                total=(len(queries) + BATCH - 1) // BATCH,
                                                desc="Processing language query"):

                    # 把这个 batch 里的 queries_sample 拼起来
                    query_all = [x for queries_sample in batch for x in queries_sample]

                    self._inference(
                        query_id=batch_start_id,
                        lang_id=lang_id,
                        queries=query_all,
                        ms_component=ms_components,
                        gamma_component=gamma_components,
                        mode=mode,
                        diff=diff,
                        group_id=group_id,
                        max_token_nums=max_token_nums,
                        lamda=lamda,
                        seed=seed,
                        prune_ratio=prune_ratio,
                        window_size_scaling=window_size_scaling,
                        sampling_size=sampling_size,
                        dataset = dataset
                    )
            else:
                for query_id, queries_sample in tqdm(enumerate(queries),
                                                    total=len(queries),
                                                    desc="Processing language query"):
                    if mode in ["CLSP_cost", "autocap_cost", "self_consistency_cost", "ours", "CLSP_acc", "translate_to_EN_sc"] and query_id % 100 != 1:
                        continue
                    self._inference(
                        query_id=query_id,
                        lang_id=lang_id,
                        queries=queries_sample,
                        ms_component=ms_components,
                        gamma_component=gamma_components,
                        mode=mode,
                        diff=diff,
                        group_id=group_id,
                        max_token_nums=max_token_nums,
                        lamda=lamda,
                        seed=seed,
                        prune_ratio=prune_ratio,
                        window_size_scaling=window_size_scaling,
                        sampling_size=sampling_size,
                        dataset = dataset
                    )

                       

    def _inference(self, 
                   query_id: int, 
                   lang_id: int, queries, 
                   ms_component=None, 
                   gamma_component=None, 
                   mode="ours", 
                   diff="high", 
                   group_id = 0,
                   sampling_params: Optional[SamplingParams] = None,
                   max_token_nums: int = 1024,
                   lamda: float = 0.4,
                   seed: int = 42,
                   prune_ratio: float = 0.6,
                   window_size_scaling: float = 3.0,
                   sampling_size: int = 3,
                   dataset: str = "polymath"
                   ):
        """
        Main entry point for inference.

        Args:
            query_id (int): Identifier for current inference batch.
            queries (List): Query list with language IDs and text.
            ms_component (List): Middle-space components.
            gamma_component (List): Gamma components per language.
        """
        
        
        prompts, sample_lang_ids, messages = self._prepare_inputs(queries, diff, mode, dataset)
        total_start_time = time.time()
        # step 1: Prepare inputs
        budget = len(prompts)
        manager = VoteManager(sampling_size=budget)
        # Create output object
        output = SynthesisOutput()
        output.mode = mode
        output.llm_init_time = self.init_times['llm_init_time']
        output.tokenizer_init_time = self.init_times['tokenizer_init_time']
        output.config = {
            "model": self.model_name,
            "mode": mode,
        }
        if mode == "ours" or mode == "ablation_top_k" or mode == "stbon" or mode == "SC_mo":
            print("---------------------------+++EARLY STOP ENABLED+++---------------------------")
            sampling_params = SamplingParams(
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=max_token_nums,
                logprobs=20,
                logits_processors=[
                    ConfPerReqLogitsProcessor(
                        sampling_size=budget,
                        manager=manager,
                        eos_token_id=sampling_params.eos_token_id if sampling_params else self.tokenizer.eos_token_id,
                        window_size_scaling=window_size_scaling,
                        seed=seed,
                        prune_ratio=prune_ratio,
                        logicobject=LogicObject(ms_components=ms_component, gamma_components=gamma_component,lamda=lamda)
                    )
                ],
            )
        else:
            sampling_params = SamplingParams(
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=max_token_nums,
                logprobs=20,
            )

        # Generate all traces at once
        print(f"Generating {budget} traces...", sampling_params)
        generation_start = time.time()
        vllm_outputs = self.llm.generate(prompts, sampling_params)
        output.generation_time = time.time() - generation_start
        
        # Process results
        processing_start = time.time()
        
        def process_batch_results_offline(batch_outputs,  manager) -> Dict[str, Any]:
            """Process batch results from vLLM for offline mode"""
            question_outputs = []
            for output_list in batch_outputs:
                question_outputs += output_list.outputs
            # Process all traces for this question
            traces = []
            total_tokens = 0
            
            for seq_id, output in enumerate(question_outputs):
                
                trace_data = process_output_offline(output, seq_id, manager.is_early_stop[seq_id], query_id)
                traces.append(trace_data)
                total_tokens += trace_data["num_tokens"]
            
            return {
                'traces': traces,
                'total_tokens': total_tokens,
                'num_traces': len(traces)
            }

        def process_output_offline(output, seq_id: int, is_seq_early_stop, query_id) -> Dict[str, Any]:
            """Process a single vLLM output for offline mode - stores full confidence array"""
            text = output.text
            if not is_seq_early_stop:
                if mode == "self_consistency" or mode == "self_consistency_acc" or mode == "CLSP_acc"  or mode == "autocap_auxiliary" or mode== "raw" or mode == "autocap" or mode == "translate_to_EN":
                    query_id = (seq_id // sampling_size) + query_id
                self.save_result(
                text = text,
                messages = messages,
                diff = diff,
                lang_id = lang_id,
                group_id= group_id,
                query_id= query_id,
                queries = queries,
                lang_ids = sample_lang_ids,
                i = seq_id)
            token_ids = output.token_ids
            
            # Calculate confidence but don't store full logprobs
            
            return {
                "stop_reason": output.finish_reason,
                "text": text,
                "token_ids": token_ids,
                "num_tokens": len(token_ids) if token_ids else 0,
            }
        processed_results = process_batch_results_offline(vllm_outputs,  manager)
        output.all_traces = processed_results['traces']
        output.total_tokens = processed_results['total_tokens']
        output.total_traces_count = len(output.all_traces)
        output.avg_tokens_per_trace = output.total_tokens / output.total_traces_count if output.total_traces_count > 0 else 0
        
        # Basic voting (for backward compatibility)
        # self._perform_basic_voting(output)
        
        output.processing_time = time.time() - processing_start

        output.total_time = time.time() - total_start_time
        # output.print_summary()
        self.save_latent(output, diff, group_id, query_id, lang_id, manager, mode)



    def _prepare_inputs(self, queries, diff, mode, dataset):
        print(mode)
        
        step_num =  5 if diff == "medium" else 6 if diff == "high" else 8 if diff == "top" else 5
        texts = []
        sample_lang_ids = []
        messages = []
        if mode == "raw":
            for i, query in enumerate(queries):
                lang_name_contraction = list(self.lang_dict.keys())[int(query['lang_id'])]
                # Construct the system instruction
                instruction = f"""
                {query['text']}
                {polymath_LANG_NOTE_DICT[lang_name_contraction]}
                """
                # Apply chat template
                message = [{"role": "user", "content": instruction}]
                text = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True, reasoning_effort="low")
                texts.append(text)
                sample_lang_ids.append(int(query['lang_id']))
                messages.append(message)
        elif mode == "autocap_auxiliary":
            for i, query in enumerate(queries):
                lang_name_contraction = list(self.lang_dict.keys())[int(query['lang_id'])]
                # Construct the system instruction
                instruction = f"""
                You are an expert in multilingual understanding and cross-lingual reasoning.

                Your task: Given a sample written in a **{lang_name_contraction}**, select **9 OTHER languages** (cross-lingual) that are optimal to support reasoning transfer for this sample.

                IMPORTANT CONSTRAINTS:
                - You MUST do **cross-lingual** selection: **do NOT select the same language as the source language**.
                - If the source language appears in the Language Options list, it is **ineligible** and must be excluded.
                - Select languages only from the provided Language Options.

                Selection criteria:
                - Prioritize linguistic proximity and transferability using **language family / branch / typology**.


                Make brief step-by-step instructions:
                1. **Infer Source Language**: Identify the most likely source language of the sample (choose one label from the Language Options). State it explicitly.
                2. **Selection Rationale**: Choose **at least three eligible target languages** (excluding the source language). Briefly justify each choice using family/branch/typology (and optionally affinity proxy: High/Medium/Low).
                3. **Alignment Score**: For each selected target language, assign an **alignment_score ∈ [0, 1]** reflecting compatibility for cross-lingual reasoning transfer, primarily from family/branch proximity (and optionally affinity proxy).
                4. **Center Language**: Designate **exactly one** selected target language as the pivot (**center=True**). All others must be **center=False**.
                5. **Conclusion Output (STRICT FORMAT)**: Output ONLY the following JSON-like line (no extra text), exactly in this format:

                Target Language=[{{"language":"L1","alignment_score":S1,"center":true/false}},{{...}}]

                Language Options:
                - Arabic: Afro-Asiatic, Semitic
                - Bengali: Indo-European, Indo-Aryan
                - German: Indo-European, Germanic
                - English: Indo-European, Germanic
                - Spanish: Indo-European, Romance
                - French: Indo-European, Romance
                - Indonesian: Austronesian, Malayo-Polynesian
                - Italian: Indo-European, Romance
                - Japanese: Japonic, Japanese
                - Korean: Koreanic, Korean
                - Malay: Austronesian, Malayo-Polynesian
                - Portuguese: Indo-European, Romance
                - Russian: Indo-European, Slavic
                - Swahili: Niger-Congo, Bantu
                - Telugu: Dravidian, South-Central Dravidian
                - Chinese: Sino-Tibetan, Sinitic

                Sample:
                {query['text']}
                """

                # Apply chat template
                message = [{"role": "user", "content": instruction}]
                text = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
                texts.append(text)
                sample_lang_ids.append(int(query['lang_id']))
                messages.append(message)
        else:
            
            for i, query in enumerate(queries):
                lang_name = list(self.lang_dict.values())[int(query['lang_id'])]
                if mode == "translate_to_EN" or mode == "translate_to_EN_sc":
                    lang_name = "English"
                # Construct the system instruction
                if dataset == "polymath":
                    instruction = f"""
                    You are an expert in mathematical / geometric reasoning reasoning.
                    Think strictly in {lang_name}. Do not use any other language.

                    FORMAT:
                    <think>
                    Step 1: ...
                    Step 2: ...
                    ...
                    Step N: ...
                    </think>
                    $\\boxed{"FINAL_ANSWER"}$

                    HARD RULES:
                    1) All intermediate reasoning MUST be inside a single <think>...</think> block, written only in {lang_name}.
                    2) At most {step_num} numbered steps. Be concise and avoid repetition.
                    3) Outside </think> you may output ONE line only: a single LaTeX boxed answer in the form $\\boxed{...}$.
                    4) Do NOT restate the problem. Do NOT add any explanation, comments, or extra text after the boxed answer.
                    5) If the result is an expression, keep it simplified. If numeric, give an exact value when possible.

                    QUESTION:
                    {query['text']}

                    NOTES:
                    - Use standard math notation. Keep symbols/variables as-is.
                    - If you reach a conclusion early, stop immediately and output the boxed answer.

                    Note: Please put the final answer in the $\\boxed{{}}$ format.
                    """
                elif dataset == "MMLU":
                    instruction = f"""
                    You are a careful expert problem solver.

                    LANGUAGE:
                    - Write all intermediate work strictly in {lang_name}. Do not use any other language.

                    OUTPUT FORMAT (follow exactly):
                    <think>
                    Step 1: ...
                    Step 2: ...
                    </think>
                    $\\boxed{{X}}$

                    HARD RULES:
                    1) Put ALL intermediate work inside a single <think>...</think> block (only {lang_name}).
                    2) Use at most {step_num} steps. Write only key computations/checks, no extra explanations.
                    3) After </think>, output EXACTLY ONE line: $\\boxed{{X}}$ where X is one letter A–J. No other text, no blank lines.
                    4) Do NOT restate the problem or options.
                    5) You MUST always include the <think> block (at least Step 1), even if you conclude early.
                    6) If the problem is ambiguous or missing info, choose the best option using the most consistent interpretation.

                    PROCEDURE (inside <think>):
                    - Identify what is being asked.
                    - Compute/derive the result.
                    - Compare against the options and select the single best letter.

                    QUESTION:
                    {query['text']}
                    """                 

                # Apply chat template
                message = [{"role": "user", "content": instruction}]
                text = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
                texts.append(text)
                sample_lang_ids.append(int(query['lang_id']))
                messages.append(message)
            

            
        return texts, sample_lang_ids, messages
    


    def save_result(self, text:str, messages:str, diff:str, lang_id:int, group_id:int, query_id:int, queries:List, lang_ids:List, i:int):
        save_res = [messages[i]] + [{"role": "assistant", "content": text}]
        file_path = os.path.join(self.output, f"{diff}", f"{lang_id}", f"{group_id}.jsonl")
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "a", encoding="utf8") as file:
            file.write(json.dumps({"query_id": query_id, "sample_lang_id":lang_ids[i], "message": save_res, "origin": queries[i]}, ensure_ascii=False)+"\n")
    

    def save_latent(self, output, diff, group_id, query_id, lang_id, manager, mode):
        data_to_save = {
            'query_id': query_id,
            'all_latency': output.generation_time,
            'all_token_lengths': output.total_tokens,
            'no_divergency_step': manager.no_divergency_step,
            'divergency_step': manager.divergency_step,
        }
        if mode == "self_consistency" or mode == "self_consistency_acc" or mode == "CLSP_acc":
            output_file = os.path.join(
                self.output + f"{diff}/",
                f"inference_data_all_{group_id}.jsonl"
            )          
        else:
            # JSONL 文件路径
            output_file = os.path.join(
                self.output + f"{diff}/{lang_id}/",
                f"inference_data_all_{group_id}.jsonl"
            )

        # 确保文件夹存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # 追加写入 JSONL
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data_to_save, ensure_ascii=False) + "\n")
