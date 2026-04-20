########################################################################
#
# @author : Chenyuan Zhang
# @when : Winter Semester 2025/2026
# @where : Harbin Institute of Technology
# @title : Dynamic Barrier with Adaptive Divergence Detection 
# @component: utils
# @file : dynamic_barrier.py
#
########################################################################

import threading
import time
import torch
from threading import Lock, Barrier
from typing import List, Dict
from queue import Queue
from .expression_processor import get_coe_feature


class EarlyStop:
    def __init__(self, n):
        self.events = [threading.Event() for _ in range(n)]
    def set(self, i): self.events[i].set()
    def is_set(self, i): return self.events[i].is_set()
    def show(self):
        print([e.is_set() for e in self.events])


class DynamicBarrier:
    """
    Synchronization barrier with adaptive divergence detection.
    - Detects first divergence round (c)
    - Votes for best thread in window [c, c+t]
    - After c+t, selects winner and terminates others
    - Reconstructs barrier so only the winner continues streaming
    """
    def __init__(self, initial_parties: int):
        self.lock = Lock()
        self.active_ranges = set(range(initial_parties))
        self.barrier = None
        self._create_barrier()

        self.comparison_results: Dict[int, Dict] = {}
        self.round_number = 1

        # Adaptive divergence point (None until first mismatch)
        self.c = None
        self.t = None  # Voting window length after divergence

        # Current best thread during voting
        self.current_winner = None

    def _create_barrier(self):
        """
        Create or recreate the barrier according to active threads.
        Ensures at least 1 participant to avoid dead barrier state.
        """
        if hasattr(self, "barrier") and self.barrier:
            try:
                self.barrier.abort()
            except:
                pass

        parties = max(1, len(self.active_ranges))
        self.barrier = Barrier(parties)
        print(f"[Init] Barrier created with {parties} participant(s). Active = {self.active_ranges}")

    def thread_finish(self, tid: int):
        """
        Mark thread as finished (exits competition).
        """
        if tid in self.active_ranges:
            self.active_ranges.remove(tid)
        print(f"[Exit] Thread {tid} removed. Active → {self.active_ranges}")

    def _detect_divergence(self, eps_rel: float = 1e-2, eps_abs: float = 5e-2) -> bool:
        """
        Returns True iff divergence is detected in the *current round* among *active* threads.

        Logic:
        1) Only compare features from active threads that belong to the current round.
        2) Stack features as (n, D) tensors on CPU float32 for stable, device-agnostic math.
        3) Compute all pairwise L2 distances.
        4) Use an adaptive threshold:
            threshold = max(eps_rel * mean(||F_i||), eps_abs)
            where eps_rel scales with feature magnitude, and eps_abs guards tiny-scale noise.
        5) If any pairwise distance exceeds the threshold, we declare divergence.
        """
        # Only compare features from the current round & currently active threads.
        actives = sorted(self.active_ranges)
        if len(self.comparison_results) < len(actives):
            return False
        if not all(self.comparison_results[tid]['round'] == self.round_number for tid in actives):
            return False

        # Collect features, cast to CPU float32, and flatten to 1D; shape => (n, D).
        F = torch.stack([
            self.comparison_results[tid]['coe_feature']
                .to('cpu', dtype=torch.float32)
                .reshape(-1)
            for tid in actives
        ], dim=0)  # (n, D)

        print(F)
        # With fewer than 2 participants, divergence is undefined.
        if F.size(0) <= 1:
            return False

        # Pairwise L2 distances between rows of F; shape => (n*(n-1)/2,).
        pd = torch.pdist(F, p=2)

        # Adaptive scale: average feature norm; keep a small floor to avoid zero scale.
        scale = F.norm(dim=1).mean().item()

        # Threshold uses both relative and absolute floors to avoid over-sensitivity.
        thr = max(eps_rel * max(scale, 1e-6), eps_abs)

        # Divergence if any pair exceeds the threshold.
        return bool(torch.any(pd > thr))



    def get_comparison_results(self) -> int:
        actives = sorted(self.active_ranges)
        if len(actives) == 1:
            return actives[0]
        F = torch.stack([
            self.comparison_results[tid]['coe_feature'].to(dtype=torch.float32, device='cpu').reshape(-1)
            for tid in actives
        ], dim=0)
        D = torch.cdist(F, F, p=2)
        n = D.shape[0]
        avg = (D.sum(dim=1) - torch.diag(D)) / (n - 1)
        idx = int(torch.argmin(avg).item())
        return actives[idx]


    def wait_at_barrier(self, thread_id: int, coe_feature: torch.Tensor, counts: List[int]) -> bool:
        """
        Main barrier logic:
        1) Store feature
        2) If first divergence, set c
        3) If inside voting window [c, c+t], vote for winner
        4) If after window, keep only best thread and stop others
        """
        with self.lock:
            if thread_id not in self.active_ranges:
                print(f"[Skip] Thread {thread_id} inactive.")
                return False

            self.comparison_results[thread_id] = {
                'coe_feature': coe_feature.detach().cpu().clone(),
                'timestamp': time.time(),
                'round': self.round_number
            }

        try:
            print(f"[Wait] Thread {thread_id} waiting at barrier (round={self.round_number}).")
            self.barrier.wait(timeout=10)

            with self.lock:
                all_in = (len(self.comparison_results) == len(self.active_ranges))
                if all_in:
                    print(f"[Round] All arrived round {self.round_number}")

                    # Step 1 — detect divergence
                    if self.c is None and self._detect_divergence():
                        self.c = self.round_number
                        self.t = 2 * self.c
                        print(f"[Adaptive-C] Divergence at round {self.c}")

                    # Step 2 — voting window [c, c+t]
                    if self.c is not None and (self.c <= self.round_number < self.c + self.t):
                        k = self.get_comparison_results()
                        print(f"[Vote] Round {self.round_number}: Thread {k} wins")
                        counts[k] += 1
                        self.current_winner = max(range(len(counts)), key=lambda x: counts[x])

                    # Step 3 — after window, enforce early stop
                    if self.c is not None and (self.round_number >= self.c + self.t):
                        winner = max(range(len(counts)), key=lambda x: counts[x])
                        self.current_winner = winner

                        print(f"[EARLY-STOP] Winner: Thread {winner} (score={counts[winner]})")
                        losers = [tid for tid in list(self.active_ranges) if tid != winner]
                        for loser in losers:
                            print(f"[Deactivate] Thread {loser}")
                            self.thread_finish(loser)
                            

                        # Rebuild barrier for single winner
                        self._create_barrier()

                    # Prepare next round
                    self.comparison_results.clear()
                    self.round_number += 1

            self.barrier.wait(timeout=10)
            return True

        except threading.BrokenBarrierError:
            with self.lock:
                print(f"[BrokenBarrier] Thread {thread_id} rebuilding barrier...")
                self._create_barrier()
            return False

        except Exception as e:
            print(f"[BarrierError] Thread {thread_id}: {e}")
            return False



def consume_streamer(
    i,
    streamer,
    useful_hidden_queue: Queue,
    results: List[str],
    counts: List[int],
    barrier_manager: DynamicBarrier,
    real_time_features: Dict,
    lock: threading.Lock,
    print_live: bool = False,
    is_keep: bool = True,
    stop_flags: EarlyStop = None,
):
    '''
    Consume generated text chunks from the streamer,
    process hidden states to extract features,
    and coordinate with the dynamic barrier for adaptive early stopping.
    Args:
        i: Thread ID
        streamer: Text generation streamer
        useful_hidden_queue: Queue for hidden states
        results: Shared list to store final outputs
        counts: Shared list to store comparison counts
        barrier_manager: DynamicBarrier instance
        real_time_features: Shared dict for real-time features
        lock: Threading lock for synchronization
        print_live: Whether to print generated text live
        is_keep: Whether to keep the output if not early stopped
    
    '''
    buf = []
    feature_count = 0
    c_val = 0
    timeout = 300
    start_time = time.time()
    # t = competitive window length after divergence
    t = 0

    early_stopped = False

    try:
        for chunk in streamer:



            buf.append(chunk)
            if print_live:
                print(chunk, end="", flush=True)

            # Consume as many hidden states as available
            while not useful_hidden_queue.empty():
                hidden_state = useful_hidden_queue.get()
                coe_feature = get_coe_feature(all_hidden=hidden_state)
                

                # Local early-stop check: once c is known and window passed
                c_val = barrier_manager.c if barrier_manager.c is not None else c_val
                t = barrier_manager.t if barrier_manager.t is not None else t
                winner = barrier_manager.current_winner

                if (
                    c_val > 0 and 
                    feature_count > (c_val + t) and
                    winner is not None
                ):
                    if  i != winner:
                        print(f"[Thread {i}] Early stopped at feature {feature_count}. Winner is {winner}.")
                        early_stopped = True
                        is_keep = False
                        if stop_flags is not None:
                            if not stop_flags.is_set(i):
                                stop_flags.set(i)
                            print("********************************")
                            stop_flags.show()
                        break
                    else:
                        print(f"[Thread {i}] Continuing as winner at feature {feature_count}.")
                        
                current_time = time.strftime("%H:%M:%S")
                print(f"Thread{i} generated feature {feature_count} at {current_time}")
                feature_count += 1

                # Barrier sync & adaptive early-stop comparison
                success = barrier_manager.wait_at_barrier(
                    thread_id=i,
                    coe_feature=coe_feature,
                    counts=counts,
                )

                # Update winner info from barrier manager
                winner = barrier_manager.current_winner

                if not success:
                    print(f"Thread {i} skipped comparison for this round.")
                    continue

    except Exception as e:
        print(f"Error in streamer consumer thread {i}: {e}")

    finally:
        # Properly unregister thread
        barrier_manager.thread_finish(i)

        # Save final output for this thread
        if feature_count is not None and c_val is not None and (is_keep == False or feature_count < (c_val + t)):
            results[i] = ""
            print(f"[END] Thread{i} finished generation with {feature_count} features. Without saving output.")
        else:
            results[i] = "".join(buf)
            print(f"[END] Thread{i} finished generation with {feature_count} features. Saved output.")

        # Mark this thread as finished by setting its count to -1
        counts[i] = -1



def consume_streamer_self_consistency(
    i,
    streamer,
    results: List[str],
    barrier_manager: DynamicBarrier,
    print_live: bool = True,
):
    '''

    Consume generated text chunks from the streamer,
    process hidden states to extract features,
    and coordinate with the dynamic barrier for adaptive early stopping.
    Args:
        i: Thread ID
        streamer: Text generation streamer
        results: Shared list to store final outputs
        barrier_manager: DynamicBarrier instance
        print_live: Whether to print generated text live
    
    '''
    
    buf = []
    feature_count = 0



    try:
        for chunk in streamer:

            buf.append(chunk)
            if print_live:
                print(chunk, end="", flush=True)

    except Exception as e:
        print(f"Error in streamer consumer thread {i}: {e}")

    finally:
        # Properly unregister thread
        barrier_manager.thread_finish(i)
        results[i] = "".join(buf)
        print(f"[END] Thread{i} finished generation with {feature_count} features. Saved output.")

