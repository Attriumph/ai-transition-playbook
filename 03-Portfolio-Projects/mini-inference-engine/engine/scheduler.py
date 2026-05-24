"""
Continuous Batching Scheduler
==============================

This module implements a continuous batching scheduler for LLM inference,
based on the iteration-level scheduling approach introduced by Orca (Yu et al.,
OSDI 2022) and refined in vLLM (Kwon et al., SOSP 2023).

Static Batching vs. Continuous Batching
---------------------------------------

**Static batching** (the naive approach):

    1. Collect N requests into a batch.
    2. Pad all prompts to the same length.
    3. Run autoregressive decoding until ALL sequences hit max_length or EOS.
    4. Only then admit new requests.

    Problem: If one sequence finishes in 10 tokens and another needs 500,
    the short one wastes 490 decode steps worth of GPU compute. The GPU is
    doing useless work on padding tokens.

    Throughput: O(batch_size / max_sequence_length_in_batch)

**Continuous batching** (this implementation):

    1. At EVERY iteration (forward pass), re-evaluate the batch composition.
    2. Remove finished sequences immediately, reclaim their memory.
    3. Admit new requests from the waiting queue mid-batch.
    4. If memory is exhausted, preempt lowest-priority running sequences.

    Benefit: GPU utilization stays near 100% because we never waste compute
    on finished sequences. New requests start generating tokens within one
    iteration of arriving.

    Throughput: O(total_tokens_generated / time), decoupled from individual
    sequence lengths.

Three-Queue State Machine
--------------------------

::

    ┌──────────┐   allocate blocks    ┌──────────┐
    │ WAITING  │ ──────────────────→  │ RUNNING  │
    └──────────┘                      └──┬───┬───┘
         ↑                     finished │   │ preempt
         │                             ↓   ↓
         │ retry               ┌──────┐ ┌──────────┐
         └─────────────────────│ DONE │ │ SWAPPED  │
                               └──────┘ └──────────┘
                                             │
                                             │ swap back in
                                             ↓
                                        ┌──────────┐
                                        │ RUNNING  │
                                        └──────────┘

- **WAITING**: Requests that have arrived but haven't been allocated GPU
  memory yet. Sorted by arrival time (FCFS).
- **RUNNING**: Requests actively being decoded on GPU. They have allocated
  KV cache blocks.
- **SWAPPED**: Requests whose KV cache has been moved to CPU memory due to
  GPU memory pressure. They'll be swapped back when space opens up.

References
----------
- Yu et al., "Orca: A Distributed Serving System for Transformer-Based
  Generative Models", OSDI 2022.
- Kwon et al., "Efficient Memory Management for Large Language Model Serving
  with PagedAttention", SOSP 2023.
"""

from __future__ import annotations

import enum
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Set, Tuple

from .kv_cache import KVCacheManager, Device


# ---------------------------------------------------------------------------
# Sequence and SequenceGroup — the units of work
# ---------------------------------------------------------------------------

class SequenceStatus(enum.Enum):
    """Lifecycle states for a single sequence."""
    WAITING = "waiting"
    RUNNING = "running"
    SWAPPED = "swapped"
    FINISHED_STOPPED = "finished_stopped"      # Hit a stop token or stop string
    FINISHED_LENGTH = "finished_length"        # Hit max_tokens limit
    FINISHED_ABORTED = "finished_aborted"      # Client cancelled or error

    @property
    def is_finished(self) -> bool:
        return self in (
            SequenceStatus.FINISHED_STOPPED,
            SequenceStatus.FINISHED_LENGTH,
            SequenceStatus.FINISHED_ABORTED,
        )


@dataclass
class Sequence:
    """A single sequence being generated.

    In simple sampling (temperature > 0, no beam search), each request
    produces exactly one Sequence. In beam search, a request produces
    ``beam_width`` Sequences that share a common prompt prefix.

    Attributes:
        seq_id: Globally unique sequence identifier.
        prompt_token_ids: The tokenized prompt.
        output_token_ids: Tokens generated so far (grows each iteration).
        status: Current lifecycle state.
        arrival_time: Timestamp when the request arrived.
        max_tokens: Maximum number of tokens to generate.
        stop_token_ids: Set of token IDs that signal generation should stop.
    """
    seq_id: int
    prompt_token_ids: List[int]
    output_token_ids: List[int] = field(default_factory=list)
    status: SequenceStatus = SequenceStatus.WAITING
    arrival_time: float = field(default_factory=time.monotonic)
    max_tokens: int = 256
    stop_token_ids: Set[int] = field(default_factory=lambda: {2})  # Default EOS=2

    @property
    def num_prompt_tokens(self) -> int:
        """Number of tokens in the prompt."""
        return len(self.prompt_token_ids)

    @property
    def num_output_tokens(self) -> int:
        """Number of tokens generated so far."""
        return len(self.output_token_ids)

    @property
    def num_total_tokens(self) -> int:
        """Total tokens (prompt + generated)."""
        return self.num_prompt_tokens + self.num_output_tokens

    @property
    def is_finished(self) -> bool:
        """Whether this sequence has terminated."""
        return self.status.is_finished

    @property
    def is_prefill(self) -> bool:
        """Whether this sequence hasn't started decoding yet (needs prefill)."""
        return self.num_output_tokens == 0

    def append_token(self, token_id: int) -> None:
        """Append a newly generated token and check stopping conditions."""
        self.output_token_ids.append(token_id)

        # Check stop conditions
        if token_id in self.stop_token_ids:
            self.status = SequenceStatus.FINISHED_STOPPED
        elif self.num_output_tokens >= self.max_tokens:
            self.status = SequenceStatus.FINISHED_LENGTH


@dataclass
class SequenceGroup:
    """A group of sequences originating from a single request.

    For greedy/sampling decoding: contains exactly 1 Sequence.
    For beam search: contains ``beam_width`` Sequences sharing a prompt.

    The SequenceGroup is the unit that the scheduler operates on — it
    admits, preempts, and completes entire groups atomically.

    Attributes:
        request_id: Unique request identifier (from the API layer).
        sequences: The sequence(s) in this group.
        arrival_time: When the request arrived (for FCFS ordering).
        priority: Lower value = higher priority. Default is arrival time.
    """
    request_id: str
    sequences: List[Sequence]
    arrival_time: float = field(default_factory=time.monotonic)
    priority: float = field(default=0.0)

    def __post_init__(self) -> None:
        if self.priority == 0.0:
            self.priority = self.arrival_time

    @property
    def is_finished(self) -> bool:
        """A group is finished when ALL its sequences are finished."""
        return all(seq.is_finished for seq in self.sequences)

    @property
    def num_sequences(self) -> int:
        return len(self.sequences)

    def get_unfinished_sequences(self) -> List[Sequence]:
        """Return sequences that are still generating."""
        return [seq for seq in self.sequences if not seq.is_finished]

    def get_max_num_total_tokens(self) -> int:
        """Maximum total token count across all sequences in the group."""
        return max(seq.num_total_tokens for seq in self.sequences)

    def get_max_num_blocks_needed(self, block_size: int) -> int:
        """Maximum blocks needed across all sequences in the group.

        Used by the scheduler to check if there's enough memory to
        keep this group running.
        """
        max_tokens = self.get_max_num_total_tokens()
        return (max_tokens + block_size - 1) // block_size


# ---------------------------------------------------------------------------
# Preemption mode
# ---------------------------------------------------------------------------

class PreemptionMode(enum.Enum):
    """How to reclaim memory from a preempted sequence.

    SWAP:
        Move KV cache blocks from GPU to CPU. Fast to resume (just swap
        back), but requires CPU memory budget. This is the preferred mode.

    RECOMPUTE:
        Discard the KV cache entirely. To resume, the prompt must be
        re-processed (prefill again). Uses no CPU memory but wastes GPU
        compute on re-prefill. Used as fallback when CPU memory is also
        exhausted.
    """
    SWAP = "swap"
    RECOMPUTE = "recompute"


# ---------------------------------------------------------------------------
# Scheduler Output — what the model runner consumes
# ---------------------------------------------------------------------------

@dataclass
class SchedulerOutput:
    """The output of one scheduling iteration.

    This dataclass tells the model runner exactly what to do in the next
    forward pass. It contains:

    - Which sequences to run (and their block tables for PagedAttention).
    - Which sequences to prefill (first forward pass, prompt processing).
    - Which sequences are in decode phase (autoregressive, one token each).
    - Block swap operations to execute on the cache engine.

    Attributes:
        scheduled_sequence_groups: SequenceGroups selected for this iteration.
        prefill_seq_ids: Sequence IDs that need prefill (prompt processing).
        decode_seq_ids: Sequence IDs in decode phase.
        blocks_to_swap_in: CPU→GPU block mappings to execute before forward pass.
        blocks_to_swap_out: GPU→CPU block mappings to execute before forward pass.
        blocks_to_copy: Source→Dest GPU block copies (for Copy-on-Write).
        preempted_groups: SequenceGroups that were preempted this iteration.
        num_batched_tokens: Total tokens in this iteration's batch.
    """
    scheduled_sequence_groups: List[SequenceGroup] = field(default_factory=list)
    prefill_seq_ids: List[int] = field(default_factory=list)
    decode_seq_ids: List[int] = field(default_factory=list)
    blocks_to_swap_in: Dict[int, int] = field(default_factory=dict)
    blocks_to_swap_out: Dict[int, int] = field(default_factory=dict)
    blocks_to_copy: Dict[int, int] = field(default_factory=dict)
    preempted_groups: List[SequenceGroup] = field(default_factory=list)
    num_batched_tokens: int = 0

    @property
    def is_empty(self) -> bool:
        """Whether there's nothing to do this iteration."""
        return (
            len(self.scheduled_sequence_groups) == 0
            and len(self.blocks_to_swap_in) == 0
            and len(self.blocks_to_swap_out) == 0
        )

    @property
    def has_swap_operations(self) -> bool:
        """Whether there are pending swap operations (adds latency)."""
        return bool(self.blocks_to_swap_in or self.blocks_to_swap_out)


# ---------------------------------------------------------------------------
# Scheduler configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SchedulerConfig:
    """Configuration parameters for the scheduler.

    Attributes:
        max_num_seqs: Maximum number of sequences in the running batch.
            This is the primary knob for trading latency vs throughput.
            Higher values increase throughput but also increase per-token
            latency.
        max_num_batched_tokens: Maximum total tokens across all sequences
            in a single iteration. Limits GPU memory usage during prefill.
        max_model_len: Maximum sequence length (prompt + generation).
            Requests exceeding this are rejected.
        preemption_mode: Default preemption strategy (SWAP or RECOMPUTE).
        delay_factor: Fraction of max waiting time before admitting new
            requests over swapped ones. 0.0 = always prefer waiting (FCFS).
    """
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 4096
    max_model_len: int = 8192
    preemption_mode: PreemptionMode = PreemptionMode.SWAP
    delay_factor: float = 0.0


# ---------------------------------------------------------------------------
# Scheduler — the core continuous batching engine
# ---------------------------------------------------------------------------

class Scheduler:
    """Continuous batching scheduler for LLM inference.

    This scheduler implements iteration-level request multiplexing. At every
    forward pass (iteration), it:

    1. **Checks running sequences** for completion (EOS / max tokens).
       Finished sequences' blocks are freed immediately.

    2. **Handles preemption** if GPU memory is too fragmented or exhausted.
       Low-priority running sequences are swapped to CPU or recomputed.

    3. **Admits new requests** from the waiting queue, allocating GPU blocks
       for their prompts. Requests are admitted in FCFS order until either
       ``max_num_seqs`` or ``max_num_batched_tokens`` is reached, or GPU
       blocks are exhausted.

    4. **Produces a SchedulerOutput** that the model runner uses to execute
       the next forward pass.

    The scheduler never touches model weights or runs forward passes — it
    purely manages *which* sequences run and *where* their KV cache lives.

    Example::

        from engine.kv_cache import CacheConfig, KVCacheManager

        cache_config = CacheConfig(
            block_size=16, num_layers=32, num_kv_heads=8, head_dim=128,
            dtype_size=2, num_gpu_blocks=1000, num_cpu_blocks=2000,
        )
        kv_manager = KVCacheManager(cache_config)

        scheduler_config = SchedulerConfig(max_num_seqs=32)
        scheduler = Scheduler(scheduler_config, kv_manager)

        # Add requests
        scheduler.add_request("req-1", prompt_token_ids=[1, 2, 3, 4, 5])
        scheduler.add_request("req-2", prompt_token_ids=[10, 20, 30])

        # Run one scheduling iteration
        output = scheduler.schedule()
        # → output.scheduled_sequence_groups contains the batch to execute
        # → output.prefill_seq_ids contains sequences needing prompt processing
    """

    def __init__(
        self,
        config: SchedulerConfig,
        kv_cache_manager: KVCacheManager,
    ) -> None:
        self.config = config
        self.kv_cache_manager = kv_cache_manager
        self.block_size = kv_cache_manager.block_size

        # The three queues — the heart of continuous batching
        #
        # WAITING: Newly arrived requests, not yet allocated GPU memory.
        #   Ordered by arrival time (FCFS). This is a deque for O(1)
        #   popleft when admitting requests.
        #
        # RUNNING: Requests actively being decoded. Their KV cache is on GPU.
        #   Ordered by priority (arrival time) for preemption decisions.
        #
        # SWAPPED: Preempted requests whose KV cache is on CPU.
        #   Will be swapped back to GPU when space opens up.
        self.waiting: Deque[SequenceGroup] = deque()
        self.running: List[SequenceGroup] = []
        self.swapped: List[SequenceGroup] = []

        # Monotonically increasing sequence ID counter
        self._next_seq_id: int = 0

        # Metrics
        self._num_completed_requests: int = 0
        self._num_preemptions: int = 0

    # ---- Public API ----

    def add_request(
        self,
        request_id: str,
        prompt_token_ids: List[int],
        max_tokens: int = 256,
        stop_token_ids: Optional[Set[int]] = None,
    ) -> SequenceGroup:
        """Add a new inference request to the waiting queue.

        This is called by the API server when a new request arrives. The
        request is NOT immediately scheduled — it waits in the ``waiting``
        queue until the next ``schedule()`` call admits it.

        Args:
            request_id: Unique request ID (from the client).
            prompt_token_ids: Tokenized prompt.
            max_tokens: Maximum tokens to generate.
            stop_token_ids: Token IDs that signal generation should stop.

        Returns:
            The created SequenceGroup.

        Raises:
            ValueError: If the prompt exceeds ``max_model_len``.
        """
        if len(prompt_token_ids) > self.config.max_model_len:
            raise ValueError(
                f"Prompt length {len(prompt_token_ids)} exceeds "
                f"max_model_len {self.config.max_model_len}."
            )

        if stop_token_ids is None:
            stop_token_ids = {2}  # Default EOS token

        seq = Sequence(
            seq_id=self._next_seq_id,
            prompt_token_ids=prompt_token_ids,
            max_tokens=max_tokens,
            stop_token_ids=stop_token_ids,
        )
        self._next_seq_id += 1

        seq_group = SequenceGroup(
            request_id=request_id,
            sequences=[seq],
        )
        self.waiting.append(seq_group)
        return seq_group

    def abort_request(self, request_id: str) -> None:
        """Abort a request by its ID. Frees any allocated resources.

        Called when a client disconnects or cancels a request.
        """
        # Check all queues
        for queue in [self.waiting, self.running, self.swapped]:
            for group in list(queue):
                if group.request_id == request_id:
                    for seq in group.sequences:
                        seq.status = SequenceStatus.FINISHED_ABORTED
                        if seq.seq_id in self.kv_cache_manager._block_tables:
                            self.kv_cache_manager.free_sequence(seq.seq_id)
                    queue.remove(group)
                    return

    def schedule(self) -> SchedulerOutput:
        """Run one scheduling iteration.

        This is the main scheduling loop, called once per forward pass.
        The algorithm proceeds in three phases:

        Phase 1: Process running sequences
            - Remove finished sequences, free their blocks.
            - Check if running sequences need new blocks (crossed a
              block boundary during the last decode step).
            - If blocks can't be allocated, preempt lowest-priority sequences.

        Phase 2: Swap in previously preempted sequences
            - If there are swapped sequences and enough GPU blocks, swap
              them back in. Swapped sequences get priority over new requests
              because they've already done partial work.

        Phase 3: Admit new requests from the waiting queue
            - Allocate blocks for new prompts.
            - Respect ``max_num_seqs`` and ``max_num_batched_tokens`` limits.

        Returns:
            SchedulerOutput describing the next forward pass.

        Note:
            This method is the single most important function in the
            entire serving engine. Every design decision here directly
            impacts throughput, latency, and fairness.
        """
        output = SchedulerOutput()

        # ----------------------------------------------------------------
        # Phase 1: Process currently running sequences
        # ----------------------------------------------------------------
        if self.running:
            self._process_running_sequences(output)

        # ----------------------------------------------------------------
        # Phase 2: Try to swap in preempted sequences
        # ----------------------------------------------------------------
        # We only swap in if there are no pending swap-outs (to avoid
        # thrashing GPU↔CPU transfers in the same iteration).
        if self.swapped and not output.blocks_to_swap_out:
            self._try_swap_in(output)

        # ----------------------------------------------------------------
        # Phase 3: Admit new requests from the waiting queue
        # ----------------------------------------------------------------
        if self.waiting:
            self._admit_waiting_requests(output)

        # ----------------------------------------------------------------
        # Build final output
        # ----------------------------------------------------------------
        output.num_batched_tokens = self._count_batched_tokens(output)
        return output

    def has_pending_requests(self) -> bool:
        """Return True if there are any unfinished requests."""
        return bool(self.waiting or self.running or self.swapped)

    def get_num_unfinished_requests(self) -> int:
        """Return the total number of unfinished requests across all queues."""
        return len(self.waiting) + len(self.running) + len(self.swapped)

    # ---- Phase 1: Process running sequences ----

    def _process_running_sequences(self, output: SchedulerOutput) -> None:
        """Handle currently running sequences: finish, allocate, or preempt.

        For each running sequence group:
        1. If finished → remove from running, free blocks.
        2. If needs a new block (crossed boundary) → try to allocate.
        3. If allocation fails → preempt the lowest-priority group.
        """
        # Sort by priority (lowest = highest priority = oldest arrival)
        # so that we preempt the NEWEST requests first (they've done the
        # least work, so preempting them wastes the least compute).
        self.running.sort(key=lambda g: g.priority)

        # We'll rebuild the running list
        still_running: List[SequenceGroup] = []
        # Preemption candidates — processed in reverse priority order
        preemption_candidates: List[SequenceGroup] = []

        for group in self.running:
            if group.is_finished:
                # All sequences in this group are done
                self._free_group(group)
                self._num_completed_requests += 1
                continue

            # Check if any unfinished sequence needs a new block
            needs_new_block = False
            for seq in group.get_unfinished_sequences():
                # A new block is needed when the sequence length crosses
                # a block boundary. We check if the current number of
                # allocated blocks can hold all the tokens.
                blocks_allocated = self.kv_cache_manager.get_block_table(
                    seq.seq_id
                ).num_blocks
                blocks_needed = self.kv_cache_manager.blocks_needed_for_tokens(
                    seq.num_total_tokens
                )
                if blocks_needed > blocks_allocated:
                    needs_new_block = True
                    break

            if needs_new_block:
                # Try to allocate the new block
                if self.kv_cache_manager.can_allocate(1):
                    for seq in group.get_unfinished_sequences():
                        blocks_allocated = self.kv_cache_manager.get_block_table(
                            seq.seq_id
                        ).num_blocks
                        blocks_needed = self.kv_cache_manager.blocks_needed_for_tokens(
                            seq.num_total_tokens
                        )
                        if blocks_needed > blocks_allocated:
                            self.kv_cache_manager.allocate_block(
                                seq.seq_id, Device.GPU
                            )
                    still_running.append(group)
                else:
                    # Can't allocate — this group becomes a preemption candidate
                    preemption_candidates.append(group)
            else:
                still_running.append(group)

        # Handle preemption: free blocks from candidates to make room
        # Process in reverse priority (newest first = least work wasted)
        for group in reversed(preemption_candidates):
            self._preempt_group(group, output)

        self.running = still_running

        # Add running groups to the output
        for group in self.running:
            output.scheduled_sequence_groups.append(group)
            for seq in group.get_unfinished_sequences():
                if seq.is_prefill:
                    output.prefill_seq_ids.append(seq.seq_id)
                else:
                    output.decode_seq_ids.append(seq.seq_id)

    def _preempt_group(
        self,
        group: SequenceGroup,
        output: SchedulerOutput,
    ) -> None:
        """Preempt a sequence group to free GPU memory.

        Depending on the preemption mode:
        - SWAP: Move KV cache to CPU. Fast to resume but uses CPU memory.
        - RECOMPUTE: Discard KV cache. Must re-prefill to resume.

        Args:
            group: The sequence group to preempt.
            output: SchedulerOutput to record swap operations.
        """
        self._num_preemptions += 1

        if self.config.preemption_mode == PreemptionMode.SWAP:
            # Check if CPU has enough blocks
            num_blocks_needed = sum(
                self.kv_cache_manager.get_block_table(seq.seq_id).num_blocks
                for seq in group.get_unfinished_sequences()
            )
            if self.kv_cache_manager.get_num_free_cpu_blocks() >= num_blocks_needed:
                # Swap to CPU
                for seq in group.get_unfinished_sequences():
                    mapping = self.kv_cache_manager.swap_out(seq.seq_id)
                    output.blocks_to_swap_out.update(mapping)
                    seq.status = SequenceStatus.SWAPPED
                self.swapped.append(group)
                output.preempted_groups.append(group)
                return

        # Fallback: RECOMPUTE mode (or CPU is full)
        # Discard KV cache entirely — will need to re-prefill
        for seq in group.get_unfinished_sequences():
            self.kv_cache_manager.free_sequence(seq.seq_id)
            seq.status = SequenceStatus.WAITING
            # Reset output tokens — the partial work is lost
            seq.output_token_ids = []
        self.waiting.appendleft(group)  # Re-add with high priority
        output.preempted_groups.append(group)

    # ---- Phase 2: Swap in preempted sequences ----

    def _try_swap_in(self, output: SchedulerOutput) -> None:
        """Try to swap preempted sequences back from CPU to GPU.

        Swapped sequences get priority over new waiting requests because:
        1. They've already done partial work (tokens generated).
        2. Swapping in is cheaper than re-prefilling (just memcpy, no compute).
        """
        still_swapped: List[SequenceGroup] = []

        for group in self.swapped:
            # Check if we have room in the running batch
            if len(self.running) >= self.config.max_num_seqs:
                still_swapped.append(group)
                continue

            # Check if GPU has enough blocks for this group
            num_blocks_needed = sum(
                self.kv_cache_manager.get_block_table(seq.seq_id).num_blocks
                for seq in group.get_unfinished_sequences()
            )
            if not self.kv_cache_manager.can_allocate(num_blocks_needed):
                still_swapped.append(group)
                continue

            # Swap in!
            for seq in group.get_unfinished_sequences():
                mapping = self.kv_cache_manager.swap_in(seq.seq_id)
                output.blocks_to_swap_in.update(mapping)
                seq.status = SequenceStatus.RUNNING

            self.running.append(group)
            output.scheduled_sequence_groups.append(group)
            for seq in group.get_unfinished_sequences():
                output.decode_seq_ids.append(seq.seq_id)

        self.swapped = still_swapped

    # ---- Phase 3: Admit new waiting requests ----

    def _admit_waiting_requests(self, output: SchedulerOutput) -> None:
        """Admit new requests from the waiting queue.

        Requests are admitted in FCFS order, subject to:
        1. ``max_num_seqs`` limit (batch size cap).
        2. ``max_num_batched_tokens`` limit (total tokens cap).
        3. Available GPU blocks.

        This is where continuous batching shines: we don't wait for the
        entire batch to finish before admitting new work. Every single
        iteration is an opportunity to fill empty batch slots.
        """
        num_running = len(self.running)
        num_batched_tokens = self._count_batched_tokens(output)

        while self.waiting:
            # Check batch size limit
            if num_running >= self.config.max_num_seqs:
                break

            group = self.waiting[0]  # Peek at the next request

            # Calculate tokens this group would add to the batch
            group_tokens = sum(
                seq.num_prompt_tokens if seq.is_prefill else 1
                for seq in group.get_unfinished_sequences()
            )

            # Check token budget
            if num_batched_tokens + group_tokens > self.config.max_num_batched_tokens:
                break

            # Calculate blocks needed for the prompt
            blocks_needed = 0
            for seq in group.sequences:
                blocks_needed += self.kv_cache_manager.blocks_needed_for_tokens(
                    seq.num_prompt_tokens
                )

            # Check GPU memory
            if not self.kv_cache_manager.can_allocate(blocks_needed):
                break

            # Admit the request!
            self.waiting.popleft()

            # Allocate blocks for each sequence in the group
            for seq in group.sequences:
                self.kv_cache_manager.allocate_initial_blocks(
                    seq.seq_id,
                    seq.num_prompt_tokens,
                    Device.GPU,
                )
                seq.status = SequenceStatus.RUNNING

            self.running.append(group)
            output.scheduled_sequence_groups.append(group)

            for seq in group.sequences:
                output.prefill_seq_ids.append(seq.seq_id)

            num_running += 1
            num_batched_tokens += group_tokens

    # ---- Helpers ----

    def _free_group(self, group: SequenceGroup) -> None:
        """Free all KV cache blocks for a finished sequence group."""
        for seq in group.sequences:
            self.kv_cache_manager.free_sequence(seq.seq_id)

    def _count_batched_tokens(self, output: SchedulerOutput) -> int:
        """Count total tokens in the current batch.

        Prefill sequences contribute their full prompt length.
        Decode sequences contribute 1 token each.
        """
        total = 0
        for group in output.scheduled_sequence_groups:
            for seq in group.get_unfinished_sequences():
                if seq.is_prefill:
                    total += seq.num_prompt_tokens
                else:
                    total += 1  # Decode: one new token per iteration
        return total

    # ---- Metrics and debugging ----

    def get_stats(self) -> Dict[str, object]:
        """Return scheduler statistics for monitoring."""
        return {
            "waiting": len(self.waiting),
            "running": len(self.running),
            "swapped": len(self.swapped),
            "completed": self._num_completed_requests,
            "preemptions": self._num_preemptions,
            "gpu_blocks_free": self.kv_cache_manager.get_num_free_gpu_blocks(),
            "cpu_blocks_free": self.kv_cache_manager.get_num_free_cpu_blocks(),
        }

    def __repr__(self) -> str:
        return (
            f"Scheduler("
            f"waiting={len(self.waiting)}, "
            f"running={len(self.running)}, "
            f"swapped={len(self.swapped)}, "
            f"completed={self._num_completed_requests})"
        )


# ---------------------------------------------------------------------------
# Self-test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from .kv_cache import CacheConfig, KVCacheManager

    print("=" * 70)
    print("Continuous Batching Scheduler — Demo")
    print("=" * 70)

    # Set up KV cache with limited blocks to demonstrate preemption
    cache_config = CacheConfig(
        block_size=16,
        num_layers=32,
        num_kv_heads=8,
        head_dim=128,
        dtype_size=2,
        num_gpu_blocks=20,   # Intentionally small to force preemption
        num_cpu_blocks=50,
    )
    kv_manager = KVCacheManager(cache_config)

    scheduler_config = SchedulerConfig(
        max_num_seqs=4,
        max_num_batched_tokens=512,
    )
    scheduler = Scheduler(scheduler_config, kv_manager)

    # Add some requests with varying prompt lengths
    for i in range(6):
        prompt = list(range(30 + i * 20))  # Varying lengths: 30, 50, 70, ...
        scheduler.add_request(
            request_id=f"req-{i}",
            prompt_token_ids=prompt,
            max_tokens=50,
        )
        print(f"Added req-{i} with prompt length {len(prompt)}")

    # Run scheduling iterations
    for iteration in range(5):
        print(f"\n--- Iteration {iteration} ---")
        output = scheduler.schedule()
        print(f"Scheduled groups: {len(output.scheduled_sequence_groups)}")
        print(f"Prefill seqs: {output.prefill_seq_ids}")
        print(f"Decode seqs: {output.decode_seq_ids}")
        print(f"Preempted: {len(output.preempted_groups)}")
        print(f"Swaps out: {len(output.blocks_to_swap_out)}")
        print(f"Scheduler: {scheduler}")
        print(f"Stats: {scheduler.get_stats()}")

        # Simulate token generation for running sequences
        for group in output.scheduled_sequence_groups:
            for seq in group.get_unfinished_sequences():
                seq.append_token(42)  # Dummy token
