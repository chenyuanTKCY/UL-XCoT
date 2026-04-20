########################################################################
#
# @author : Chenyuan Zhang
# @when : Winter Semester 2025/2026
# @where : Harbin Institute of Technology
# @title : Expression Processor
# @component: utils
# @file : model_processor.py
#
########################################################################

import torch
import time
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
from .config  import ms_save_dir, gamma_save_dir, ma_diff_save_dir
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


class HookModule:
    def __init__(self, model: AutoModelForCausalLM = None):
        self.model = model
        self.hook_list = []

    def add_layer_hooks(self, target_layers, modification_func):
        """
        Dynamically attach hooks to specific layers of the model.

        Args:
            model: Transformer model instance.
            target_layers (List[int]): Layer indices to hook.
            modification_func (Callable): Function to modify hidden states.

        Returns:
            List: Hook handles to be removed after inference.
        """

        hooks, layers = [], self.model.model.layers

        def create_hook(layer_idx):
            def hook_fn(module, _, output):
                # Modify hidden states during forward pass
                if torch.is_tensor(output):
                    return modification_func(output, layer_idx)
                elif isinstance(output, tuple) and len(output) > 0:
                    modified = modification_func(output[0], layer_idx)
                    return (modified,) + output[1:]
                return output
            return hook_fn

        for layer_idx in target_layers:
            if -len(layers) < layer_idx < len(layers):
                hooks.append(layers[layer_idx].register_forward_hook(create_hook(layer_idx)))
            else:
                print(f"Warning: Layer {layer_idx} is out of valid range.")

        self.hook_list.extend(hooks)

    def middle_layer_logic_extract(self, hidden_states, layer_idx, ms_components, lamda=0.1):
        """
        Modify hidden states by removing the linguistic bias subspace.

        Args:
            hidden_states (torch.Tensor): Hidden representations.
            layer_idx (int): Current layer index.
            ms_components (torch.Tensor): Middle-space components.
            lamda (float): Scaling coefficient.

        Returns:
            torch.Tensor: Modified hidden states.
        """
        M = ms_components[layer_idx]
        ling_diff = lamda * (hidden_states.float() @ M.float() @ M.float().T).to(hidden_states.dtype)
        return hidden_states - ling_diff

    def inverse_middle_layer_logic(self, hidden_states, layer_idx, ms_components, gamma_components, lang_id, lamda=0.1):
        """
        Reconstruct hidden states with language-specific features.

        Args:
            hidden_states (torch.Tensor): Hidden states after modification.
            layer_idx (int): Current layer index.
            ms_components (torch.Tensor): Middle-space components.
            gamma_components (torch.Tensor): Gamma components per language.
            lang_id (int): Index of language.
            lamda (float): Scaling coefficient.

        Returns:
            torch.Tensor: Restored hidden states.
        """
        M = ms_components[layer_idx - 1]
        G = gamma_components[layer_idx - 1]
        ling_diff = lamda * (M @ G[lang_id]).to(hidden_states.dtype)
        return hidden_states + ling_diff


    def remove_hooks(self):
        """
        Remove all registered hooks from the model.
        """
        for hook in self.hook_list:
            hook.remove()
        self.hook_list = []





def expression_split(final_token_expression: torch.Tensor,
                    r:int = 5,) -> Tuple:
    '''
        Split the final token expression into mr and ml components.
        args:
            final_token_expression: The mean embedding to be splitted.
            r: The rank of the subspace.
            rank_method: The method to rank the mr component.
        returns:
            ma_components (d, 1).
            ms_components (d, r).
            Gamma_components (n_languages, r).
    '''
    # print(final_token_expression.shape)
    final_token_expression = final_token_expression.cpu().to(torch.float32).numpy()
    # r = np.linalg.matrix_rank(final_token_expression)
    d, n_languages = final_token_expression.shape
    
    # Approximate 
    # ones_vector_shape: (n_languages, 1)
    ones_vector = np.ones((n_languages, 1))

    # M_a_prime_shape: (d, 1)
    M_a_prime = (1/d) * final_token_expression @ ones_vector

    # Center the matrix   M_centered_shape: (d, n_languages)
    M_centered = final_token_expression - np.outer(M_a_prime, ones_vector.T)
    
    # Using SVD to the matrix u_shape: (d, r), s_shape: (r,), v_shape: (r, n_languages)
    U, _, Vt = svd(M_centered, full_matrices=False)
    # M_s_shape: (d, r)
    M_s_prime = U[:, :r]
    Gamma_prime = Vt[:r, :].T

    # M_prime_shape: (d, n_languages)
    M_prime = np.outer(M_a_prime, ones_vector.T) + M_s_prime @ Gamma_prime.T
    
    def _pseudoinverse_svd(A: np.ndarray, tol: float = 1e-15) -> np.array:
        '''
            Compute the pseudoinverse of a matrix using SVD.
            args:
                A: The input matrix.
                tol: The tolerance for singular values.
            returns:
                A_pinv: The pseudoinverse of the input matrix.
        '''
        
        U, s, Vt = svd(A, full_matrices=False)
        
        s_pseudo = np.zeros_like(s)
        s_pseudo[s > tol] = 1.0 / s[s > tol]
        Sigma_pseudo = np.diag(s_pseudo)

        A_pseudo = Vt.T @ Sigma_pseudo @ U.T
        return A_pseudo
    M_prime_pinv = _pseudoinverse_svd(M_prime.T) @ ones_vector

    # Force orthogonalization
    denominator = np.linalg.norm(M_prime_pinv)**2
    
    # M_a_shape: (d, 1)
    M_a = M_prime_pinv / denominator
    
    # Center the matrix after orthogonalization
    M_centered_ortho = M_prime - np.outer(M_a, ones_vector)
    U_ortho, _, Vt_ortho = svd(M_centered_ortho, full_matrices=False)
    
    # M_s_shape: (d, r)
    M_s = U_ortho[:, :r]
    # Gamma_shape: (r, n_languages)
    Gamma = Vt_ortho[:r, :].T
    ma_diff_file_name = "ma_diff_"+time.strftime("%Y%m%d_%H%M%S")+".npy"
    ms_file_name = "ms_"+time.strftime("%Y%m%d_%H%M%S")+".npy"
    gamma_file_name = "gamma_"+time.strftime("%Y%m%d_%H%M%S")+".npy"

    # np.save(ms_save_dir+ms_file_name, M_s)
    # np.save(gamma_save_dir+gamma_file_name, Gamma)
    # np.save(ma_diff_save_dir+ma_diff_file_name, M_a)

    return (M_a.tolist(), M_s.tolist(), Gamma.tolist())
    



def get_similar_languages(hidden_state: torch.Tensor, sampling_size: int) -> List:
    '''
        Get the most similar languages based on the ms components.
    '''
    # print(f"--------------------{sampling_size}-----------------------")
    hidden_state = hidden_state.cpu().to(torch.float32).numpy()

    # Compute the norm of each row in the hidden state matrix
    hidden_state_norm = np.linalg.norm(hidden_state.T, axis=1, keepdims=True)

    # Normalize the hidden state matrix
    normalized_hidden_state = hidden_state.T / hidden_state_norm


    def _calculate_cosine_similarity(query: np.array, candidates: np.array) -> np.array:
        '''
            Calculate the cosine similarity between the query and candidates.
            args:
                query: The query vector.
                candidates: The candidate vectors.
            returns:
                similarities: The cosine similarity scores.
        '''
        similarities = candidates @ query.T

        query_norm = np.linalg.norm(query)
        candidates_norm = np.linalg.norm(candidates, axis=1)

        # Compute the cosine similarity
        similarities = similarities / (query_norm * candidates_norm)

        return similarities
    

    result = []
    for i in range(len(normalized_hidden_state)):
        query = normalized_hidden_state[i][:]
        similarities = _calculate_cosine_similarity(query, normalized_hidden_state)

        # Sort the similarities in descending order
        sorted_indices = np.argsort(similarities)[::-1]

        # Get the top k similar indices, excluding self
        
        filtered_indices = [idx for idx in sorted_indices if idx != i]
        top_k_indices = filtered_indices[:sampling_size]

        result.append(top_k_indices)

    return result



def get_coe_feature(all_hidden: torch.Tensor)->torch.Tensor:
    '''
        Compute the Coefficient of Efficiency (COE) feature for the given hidden states.
        args:
            all_hidden: The hidden states tensor of shape (batch_size, seq_length, hidden_size).
        returns:
            coe_features: The COE values for each sample in the batch.
    '''


    # all_hidden shape: (layer_id, [bath_size, seq_length, hidden_size])
    Z_l = torch.stack([layer.mean(dim=1) for layer in all_hidden], dim = 0)  # shape: (layer_id, batch_size, hidden_size)
    def M(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.norm(a - b, p = 2, dim = -1)
    
    def A(a, b, eps=1e-9):
        a_norm = F.normalize(a, dim=-1)
        b_norm = F.normalize(b, dim=-1)
        cos = (a_norm * b_norm).sum(dim=-1).clamp(-1 + eps, 1 - eps)
        return torch.acos(cos)
    
    L, B, _ = Z_l.shape
    F_h = torch.zeros(B, device = Z_l.device)

    M_ref = M(Z_l[0], Z_l[-1])
    A_ref = A(Z_l[0], Z_l[-1])
    M_ref = torch.clamp(M_ref, min=1e-6)
    A_ref = torch.clamp(A_ref, min=1e-6)
    for layer_id in range(L - 1):
        M_current = M(Z_l[layer_id], Z_l[layer_id + 1])
        A_current = A(Z_l[layer_id], Z_l[layer_id + 1])
        F_h_l = (M_current / (M_ref + 1e-8)) - (A_current / (A_ref + 1e-8))
        F_h += F_h_l
    
    return (F_h / (L - 1)).unsqueeze(-1)  # shape: (B, 1)

