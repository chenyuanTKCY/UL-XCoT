########################################################################
#
# @author : Chenyuan Zhang
# @when : Winter Semester 2025/2026
# @where : Harbin Institute of Technology
# @title : Hook vllm and process early stopping 
# @component: utils
# @file : early_stop.py
#
########################################################################

import torch
import math
import os
import torch.nn.functional as F
from typing import List, Optional




class LogicObject:
    """
    LogicObject applies language-specific middle-space (MS) projection
    and γ-based linguistic correction to the transformer hidden states.

    Args:
        ms_components (List[torch.Tensor]): List of projection matrices M_l for each layer.
        gamma_components (List[torch.Tensor]): Optional γ-components per layer.
        lamda (float): Scaling factor for linguistic correction.
    """
    def __init__(
        self,
        ms_components: Optional[List[torch.Tensor]] = None,
        gamma_components: Optional[List[torch.Tensor]] = None,
        lamda: float = 0.4
    ):
        self.ms_components = [torch.tensor(m) for m in ms_components] if ms_components is not None else None
        self.gamma_components = [torch.tensor(g) for g in gamma_components] if gamma_components is not None else None
        self.lamda = lamda

    def process_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply MS-based linguistic correction to hidden states of selected layers.

        Args:
            hidden_states (torch.Tensor):
                Shape [num_layers, batch_size, hidden_size]

        Returns:
            torch.Tensor:
                Processed hidden states with the same shape as input.
        """
        num_layers, batch_size, hidden_size = hidden_states.shape

        # Layers to process (example: 5 → 14 inclusive)
        process_layers = list(range(2, num_layers))

        # Safety check
        if self.ms_components is None:
            return hidden_states

        if len(self.ms_components) < max(process_layers) + 1:
            raise ValueError(
                f"ms_components length={len(self.ms_components)} is smaller than "
                f"the required layer index {max(process_layers)}."
            )

        # Clone to avoid in-place modification of upstream values
        processed = hidden_states.clone()

        for layer_idx in process_layers:
            # Extract hidden states for this layer: shape [batch, hidden]
            H_l = hidden_states[layer_idx]  # [B, D]

            # Projection matrix for this layer: shape [D, d] (you defined it)
            M = self.ms_components[layer_idx].to(H_l.device)

            # Linguistic difference term:
            #     Δ = λ * (H @ M @ Mᵀ)
            #
            # Shape:
            #   H    : [B, D]
            #   M    : [D, d]
            #   Mᵀ   : [d, D]
            #   HMMᵀ : [B, D]
            ling_diff = self.lamda * (H_l.float() @ M.float() @ M.float().T)
            ling_diff = ling_diff.to(H_l.dtype)

            # Subtract linguistic component
            processed[layer_idx] = H_l - ling_diff

        return processed


# ============================================================
# 1. Global VoteManager
#    - Single instance per request
#    - Shared by ALL ConfPerReqLogitsProcessor clones
# ============================================================

class VoteManager:
    """
    Global controller shared across all LogitsProcessor instances.

    Why this is needed:
    -------------------
    vLLM internally clones / deep-copies logits processors when:
      - creating per-request processors
      - doing multiprocessing / worker dispatch

    As a result, any state stored on `self.xxx` inside the processor
    is NOT guaranteed to be shared across decoding steps or across
    different sampling paths.

    To keep a consistent view of:
        - which traces are early-stopped
        - voting scores
        - which trace is the final winner
        - running statistics of hidden states
    we store them in this external manager and pass it BY REFERENCE
    into each processor instance.
    """

    def __init__(self, sampling_size: int):
        """
        Args:
            sampling_size: number of parallel sampling traces (e.g. n in SamplingParams)
        """
        self.sampling_size = sampling_size

        # Per-trace early-stop flags
        self.is_early_stop: List[bool] = [False] * sampling_size

        # Per-trace accumulated votes inside the voting window
        self.voting_result: List[int] = [0] * sampling_size

        # Final winner trace id (None until decided)
        self.winner: Optional[List] = None

        # Voting window [start_len, end_len] in terms of token_length(anchor)
        self.window: Optional[List[int]] = None

        # Whether we have already finished voting and frozen the result
        self.voting_done: bool = False

        # Running-average hidden states: shape [L, sampling_size, H]
        self.avg_hidden_states: Optional[torch.Tensor] = None

        # Number of tokens processed for each trace
        self.token_length: List[int] = [0] * sampling_size

        self.no_divergency_step = 0
        self.divergency_step = 0

    def reset(self):
        """
        Optional helper if you reuse the same manager across multiple high-level requests.
        """
        self.is_early_stop = [False] * self.sampling_size
        self.voting_result = [0] * self.sampling_size
        self.winner = None
        self.window = None
        self.voting_done = False
        self.avg_hidden_states = None
        self.token_length = [0] * self.sampling_size
        self.no_divergency_step = 0
        self.divergency_step = 0


# ============================================================
# 2. Logits Processor using VoteManager
# ============================================================

class ConfPerReqLogitsProcessor:
    """
    A value-based early-stop controller for multi-sampling reasoning.

    This processor:
      - observes hidden-state trajectories of each sampling path (trace)
      - computes CoE_R (Change of Embedding) features
      - detects divergence among traces
      - opens a voting window
      - identifies the most consistent trace as the "winner"
      - forces all "loser" traces to emit EOS (soft early stop)

    All persistent state is stored in a shared VoteManager instance.
    """

    def __init__(
        self,
        sampling_size: int,
        manager: VoteManager,
        logicobject: Optional[LogicObject] = None,
        eos_token_id: int = 23,
        window_size_scaling: float = 2.0,
        prune_ratio: float = 0.4,
        seed: int = 42,
    ):
        """
        Args:
            sampling_size      : number of parallel sampling traces
            manager            : shared VoteManager instance
            eos_token_id       : token id to force when early-stopping a trace
            window_size_scaling: voting window length ~ scaling * start_length
        """
        self.sampling_size = sampling_size
        self.manager = manager          # shared across processor clones
        self.eos_token_id = eos_token_id
        self.window_size_scaling = window_size_scaling
        self.logicobject = logicobject
        print(f"------------prune_ratio: {prune_ratio}---------------")
        self.prune_ratio = prune_ratio
        self.keep_num = math.ceil((1-self.prune_ratio) * sampling_size)
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)

    # --------------------------------------------------------
    # Main entry point: called by vLLM once per trace per step
    # --------------------------------------------------------
    def __call__(
        self,
        prompt_tokens_ids: List[int],
        past_tokens_ids: List[int],
        logits: torch.Tensor,
        hidden_states: torch.Tensor,  # expected: [num_layers, sampling_size, hidden_size]
        seq_id: int,                  # trace id in [0, sampling_size-1]
    ) -> torch.Tensor:
        """
        This method is executed for EACH sampling trace at EACH decoding step.

        Important assumptions:
        ----------------------
        - `hidden_states` contains hidden states for ALL traces:
              shape = [L, sampling_size, H]
          If your hook only gives [L,1,H], you need to modify the model runner
          to collect and pass all traces together.
        """
        seq_id = seq_id % self.sampling_size

        # 1) Update global state & voting logic
        if (not self.manager.voting_done) and (hidden_states is not None):
            self._update_state_and_maybe_vote(hidden_states, seq_id)

        # 2) If this trace has been marked early-stop → force EOS
        if 0 <= seq_id < self.sampling_size and self.manager.is_early_stop[seq_id]:
            # Hard-mask logits: only EOS is allowed
            # NOTE: we clone logits in-place; if you want to avoid side-effects,
            # you can do `logits = logits.clone()` first.
            val_to_keep = logits[self.eos_token_id].item()
            logits[:] = float("-inf")
            logits[self.eos_token_id] = val_to_keep
            return logits
        # 3) Otherwise, return logits unchanged
        return logits

    # --------------------------------------------------------
    # Internal: running-mean update + divergence + voting
    # --------------------------------------------------------
    def _update_state_and_maybe_vote(
        self,
        hidden_states: torch.Tensor,
        seq_id: int,
    ) -> None:
        """
        Full pipeline:
          - identify active traces
          - pick an anchor trace (smallest active seq_id)
          - update running mean hidden states
          - compute CoE features
          - if divergence detected → open voting window
          - inside window → accumulate votes for the most central trace
          - after window → finalize winner, early-stop losers
        """

        # 0) Determine which traces are still active
        active = [i for i in range(self.sampling_size)
                  if not self.manager.is_early_stop[i]]
        if len(active) == 0:
            # Nothing left to manage
            self.manager.voting_done = True
            return

        # Anchor trace: only this trace will perform global updates
        anchor = min(active)
        if seq_id != anchor:
            # Other traces just read manager state; they do not write here.
            return

        # Shape check
        if hidden_states.dim() != 3:
            # Unexpected shape: skip to avoid crashing
            return
        L, B, H = hidden_states.shape
        if B != self.sampling_size:
            # We expect hidden_states to contain ALL traces at once.
            # If not, your model runner / hook needs adjustment.
            return

        # ------------------ 1) Running mean update ------------------
        temp_hidden_states = hidden_states.detach().clone()
        temp_hidden_states = self.logicobject.process_hidden_states(temp_hidden_states)
        if self.manager.avg_hidden_states is None:
            # First step: initialize running mean with current hidden_states
            
            self.manager.avg_hidden_states = temp_hidden_states
            for i in active:
                self.manager.token_length[i] = 1
        else:
            # Incremental running mean for active traces
            for i in active:
                self.manager.token_length[i] += 1
                count = self.manager.token_length[i]
                w = 1.0 / float(count)
                self.manager.avg_hidden_states[:, i, :] += w * (
                    temp_hidden_states[:, i, :] - self.manager.avg_hidden_states[:, i, :]
                )

        # ------------------ 2) Compute scalar CoE_R per trace ------------------
        coe_values: List[torch.Tensor] = []
        active_ids_for_coe: List[int] = []

        for i in active:
            # [L, H] → [L, 1, H] for CoEScoreInfo
            hs_i = self.manager.avg_hidden_states[:, i, :].unsqueeze(1)
            # if 10 < self.manager.token_length[i] < 20:
                # save_path = f'./results/hidden_states_analysis/ablation/hidden_states_trace_{i}_len_{self.manager.token_length[i]}.pt'
                # os.makedirs(os.path.dirname(save_path), exist_ok=True)
                # torch.save(hs_i, save_path)
            coe_r = CoEScoreInfo(hs_i).compute_CoE_R()
            if not torch.isnan(coe_r):
                active_ids_for_coe.append(i)
                coe_values.append(coe_r)

        if len(active_ids_for_coe) < 2:
            # Cannot compare divergence with < 2 candidates
            return

        all_coe = torch.stack(coe_values)  # [num_active]
        curr_len = self.manager.token_length[anchor]

        # ------------------ 3) Detect divergence → open voting window ------------------
        if (self.manager.window is None) and (not self.manager.voting_done) and curr_len>=10:
            if self._detect_divergence(all_coe):
                # torch.save(self.manager.avg_hidden_states, f'./results/model_weights_{seq_id}.pt')
                start = curr_len
                # ensure window length >= 1
                end = start + max(1, int(self.window_size_scaling * start))
                self.manager.window = [start, end]

        # ------------------ 4) If window exists: vote or finalize ------------------
        window = self.manager.window
        if window is None:
            return

        start, end = window

        # -------- Inside window: accumulate votes --------
        if start <= curr_len <= end:

            coe_vals = all_coe   # [num_active]
            num = coe_vals.numel()
            k = min(self.keep_num, num)

            diverged = self._detect_divergence(coe_vals)

            if diverged:
                # Divergence detected -> select the top-K most central traces.
                self.manager.divergency_step += 1
                local_ids = self._detect_best_multi(coe_vals, k=k)
            else:
                # No divergence detected -> randomly select K traces.
                self.manager.no_divergency_step += 1
                local_ids = torch.randperm(num, generator=self.rng)[:k].tolist()

            # Map local ids back to global trace ids.
            for lid in local_ids:
                gid = active_ids_for_coe[lid]
                self.manager.voting_result[gid] += 1

            return
        # -------- Window passed → finalize winners --------
        elif curr_len > end and (not self.manager.voting_done):

            votes = torch.tensor(
                [self.manager.voting_result[i] for i in active_ids_for_coe],
                dtype=torch.float32,
            )

            # top-k based on votes
            num = min(self.keep_num, votes.numel())
            top_vals, top_idx = torch.topk(votes, k=num)

            diverged = self._detect_divergence(top_vals)

            winner_local_ids = top_idx.tolist()


            # Map local winner ids back to global ids.
            winners = [active_ids_for_coe[lid] for lid in winner_local_ids]
            self.manager.winners = winners

            # Early-stop the losing traces.
            for gid in active_ids_for_coe:
                # if gid not in winners:
                if gid in winners:
                    self.manager.is_early_stop[gid] = True
                    print(f"[Voting] Trace {gid} stopped at length "
                        f"{self.manager.token_length[gid]}")

            self.manager.voting_done = True
            self.manager.window = None
    # --------------------------------------------------------
    # Divergence detection on scalar CoE features
    # --------------------------------------------------------
    @staticmethod
    def _detect_divergence(
        all_coe_features: torch.Tensor,
        eps_abs: float = 0.05, #0.05
        eps_rel: float = 0.09, # 0.09
        min_spread_ratio: float = 2.75,   # max_rel must be much larger than mean_rel to count as divergence
    ) -> bool:
        vals = all_coe_features.view(-1).to(torch.float32)
        if vals.numel() < 2:
            return False

        diff = torch.abs(vals.unsqueeze(0) - vals.unsqueeze(1))

        denom = torch.maximum(vals.abs().unsqueeze(0), vals.abs().unsqueeze(1))
        denom = torch.clamp(denom, min=1e-3)
        rel_diff = diff / denom

        max_diff = diff.max().item()
        max_rel  = rel_diff.max().item()
        mean_rel = rel_diff.mean().item()

        # First check absolute and relative thresholds.
        if not (max_diff > eps_abs and max_rel > eps_rel):
            return False

        # Then check whether the largest deviation is truly prominent,
        # rather than all traces moving together.
        if mean_rel <= 1e-6:
            return False  # Almost identical values; only minor fluctuations.
        if max_rel < mean_rel * min_spread_ratio:
            return False  # No clearly separated pair.

        return True



    # --------------------------------------------------------
    # Pick the "most central" sample by cosine similarity
    # --------------------------------------------------------
    @staticmethod
    def _detect_best_multi(all_coe_features: torch.Tensor, k: int = 1):
        """
        Given N scalar CoE values, compute their centrality and return
        the indices of the top-k most 'central' samples.

        Centrality_i = -sum_j |coe_i - coe_j|
        (Lower total deviation means more central)
        """
        coe = all_coe_features.view(-1)   # [N]
        N = coe.size(0)

        if N <= k:
            return list(range(N))

        # pairwise absolute differences matrix: |c_i - c_j|
        # shape: [N, N]
        diffs = torch.abs(coe.unsqueeze(0) - coe.unsqueeze(1))

        # centrality score = negative total deviation
        # Lower total deviation is better, so negate it and maximize.
        centrality = -diffs.sum(dim=1)    # [N]

        # select top-k
        _, topk_idx = torch.topk(centrality, k=k, largest=True)
        return topk_idx.tolist()



# ============================================================
# 3. CoE Score Computation
# ============================================================

class CoEScoreInfo:
    """
    Compute magnitude and angular Change of Embedding (CoE)
    across transformer layers for a single token trajectory.
    """

    def __init__(self, hidden_states: torch.Tensor):
        """
        Args:
            hidden_states: tensor of shape [num_layers, 1, d_model]
        """
        # Remove the batch dimension → [L, D]
        self.hidden_states = hidden_states.squeeze(1)

    # ------------------ Magnitude CoE ------------------
    def compute_CoE_Mag(self):
        """
        Magnitude-based CoE:
          - normalize layer-to-layer differences by the total change
            between first and last layer
        """
        hs = self.hidden_states  # [L, D]
        L = hs.size(0)
        if L < 2:
            z = torch.zeros(1, device=hs.device, dtype=hs.dtype)
            return z, z.mean(), z.var(unbiased=False)

        denom = torch.norm(hs[-1] - hs[0], p=2) + 1e-12
        deltas = hs[1:] - hs[:-1]                        # [L-1, D]
        mag_norm = torch.norm(deltas, dim=1, p=2) / denom  # [L-1]

        return mag_norm, mag_norm.mean(), mag_norm.var(unbiased=False)

    # ------------------ Angular CoE ------------------
    def compute_CoE_Ang(self):
        """
        Angular-based CoE:
          - measures how the direction of the representation changes
            across layers, normalized by the total angle between
            first and last layer.
        """
        hs = self.hidden_states  # [L, D]
        L = hs.size(0)
        if L < 2:
            z = torch.zeros(1, device=hs.device, dtype=hs.dtype)
            return z, z.mean(), z.var(unbiased=False)

        # Total angle between first and last layer
        cos_total = torch.dot(hs[-1], hs[0]) / (
            torch.norm(hs[-1], p=2) * torch.norm(hs[0], p=2) + 1e-12
        )
        cos_total = torch.clamp(cos_total, -1.0, 1.0)
        total_angle = torch.acos(cos_total) + 1e-12

        angles = []
        for i in range(L - 1):
            a, b = hs[i + 1], hs[i]
            cos_sim = torch.dot(a, b) / (
                torch.norm(a, p=2) * torch.norm(b, p=2) + 1e-12
            )
            cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
            angle = torch.acos(cos_sim) / total_angle
            angles.append(angle)

        angles = torch.stack(angles)  # [L-1]
        return angles, angles.mean(), angles.var(unbiased=False)

    # ------------------ CoE_R: mag_mean - ang_mean ------------------
    def compute_CoE_R(self):
        """
        CoE_R = mean(magnitude CoE) - mean(angular CoE)
        A single scalar summarizing both aspects.
        """
        _, mag_ave, _ = self.compute_CoE_Mag()
        _, ang_ave, _ = self.compute_CoE_Ang()
        return mag_ave - ang_ave

