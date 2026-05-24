"""
PagedAttention KV Cache Manager
================================

This module implements the core memory management algorithms from the
PagedAttention paper (Kwon et al., SOSP 2023). The key insight is to treat
GPU KV cache memory exactly like an operating system manages physical RAM:

    ┌─────────────────────────────────────────────────────────────┐
    │  OS Virtual Memory              KV Cache (PagedAttention)   │
    │  ─────────────────              ─────────────────────────   │
    │  Virtual page         →         Logical KV block            │
    │  Physical page frame  →         Physical KV block (GPU)     │
    │  Page table           →         Block table                 │
    │  malloc / free        →         allocate / free_block       │
    │  fork() + CoW         →         Beam search sharing         │
    │  Swap to disk         →         Swap KV blocks to CPU       │
    └─────────────────────────────────────────────────────────────┘

Why This Matters
----------------
Naive KV cache allocation pre-reserves a contiguous buffer of size
``max_seq_len × num_layers × 2 × num_heads × head_dim × dtype_size``
for *every* request, even if the actual generated sequence is much shorter.
This causes 60–80% internal fragmentation.

PagedAttention eliminates this by allocating fixed-size blocks on demand,
just like OS paging eliminates the need for contiguous physical memory.

Block Budget Formula
--------------------
The total number of physical blocks that fit in GPU memory is:

    total_blocks = gpu_memory_for_kv // block_memory

where:

    block_memory = block_size × num_layers × 2 × num_kv_heads × head_dim × dtype_size

The factor of 2 accounts for both Key and Value tensors stored per layer.

References
----------
- Kwon et al., "Efficient Memory Management for Large Language Model Serving
  with PagedAttention", SOSP 2023. https://arxiv.org/abs/2309.06180
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Data types and configuration
# ---------------------------------------------------------------------------

class Device(enum.Enum):
    """Device where a physical block resides."""
    GPU = "gpu"
    CPU = "cpu"


@dataclass(frozen=True)
class CacheConfig:
    """Configuration for the KV cache geometry.

    These parameters are determined by the model architecture and the
    operator's choice of block size and dtype.

    Attributes:
        block_size: Number of tokens per block. Typical values: 16 or 32.
            Smaller blocks reduce internal fragmentation but increase
            block-table overhead. 16 is the vLLM default.
        num_layers: Number of transformer layers (e.g., 32 for Llama-3-8B).
        num_kv_heads: Number of key/value attention heads. For GQA models
            this is less than num_attention_heads (e.g., 8 for Llama-3-8B
            which uses GQA with 32 Q heads and 8 KV heads).
        head_dim: Dimension of each attention head (e.g., 128).
        dtype_size: Bytes per element. FP16/BF16 = 2, FP8/INT8 = 1.
        num_gpu_blocks: Total physical blocks available on GPU.
        num_cpu_blocks: Total physical blocks available on CPU (for swapping).
    """
    block_size: int = 16
    num_layers: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    dtype_size: int = 2  # FP16 / BF16
    num_gpu_blocks: int = 0
    num_cpu_blocks: int = 0

    @property
    def block_memory_bytes(self) -> int:
        """Memory consumed by a single physical block across all layers.

        Formula:
            block_size × num_layers × 2 × num_kv_heads × head_dim × dtype_size
                                      ↑
                              (Key + Value tensors)

        Example (Llama-3-8B, FP16, block_size=16):
            16 × 32 × 2 × 8 × 128 × 2 = 4,194,304 bytes = 4 MB per block
        """
        return (
            self.block_size
            * self.num_layers
            * 2  # Key + Value
            * self.num_kv_heads
            * self.head_dim
            * self.dtype_size
        )

    @property
    def block_memory_mb(self) -> float:
        """Block memory in megabytes, for human-readable reporting."""
        return self.block_memory_bytes / (1024 * 1024)


# ---------------------------------------------------------------------------
# Physical Block — a fixed-size chunk of GPU/CPU memory
# ---------------------------------------------------------------------------

@dataclass
class PhysicalBlock:
    """A physical block of KV cache memory on a device.

    Analogous to a **page frame** in OS virtual memory. Each physical block
    holds KV data for exactly ``block_size`` tokens across all layers.

    Attributes:
        device: The device (GPU or CPU) where this block resides.
        block_id: Unique identifier within the device's block pool.
        ref_count: Number of logical blocks currently mapping to this
            physical block. When ref_count > 1, the block is shared
            (Copy-on-Write). When ref_count drops to 0, the block can
            be reclaimed.
    """
    device: Device
    block_id: int
    ref_count: int = 0

    def is_shared(self) -> bool:
        """Return True if multiple logical blocks reference this physical block.

        A shared block must be copied before mutation (Copy-on-Write).
        This occurs during beam search when multiple beam candidates share
        a common prefix.
        """
        return self.ref_count > 1


# ---------------------------------------------------------------------------
# Block Allocator — manages the free pool for one device
# ---------------------------------------------------------------------------

class BlockAllocator:
    """Manages a pool of physical KV cache blocks on a single device.

    This is analogous to the **physical page frame allocator** in an OS
    kernel. It maintains a free list of block IDs and hands them out on
    demand.

    The allocator is intentionally simple: a stack-based free list. In a
    production system (vLLM), the allocator also coordinates with a
    ``CacheEngine`` that manages the actual GPU tensors. Here, we model
    the allocation logic without the tensor backing store.

    Attributes:
        device: GPU or CPU.
        num_blocks: Total number of blocks in the pool.
        block_size: Tokens per block (for reference, not used in allocation).

    Example::

        allocator = BlockAllocator(device=Device.GPU, num_blocks=1000, block_size=16)

        # Allocate a block (like a page fault allocating a frame)
        block = allocator.allocate()
        assert block.ref_count == 1

        # Free the block (like process exit releasing frames)
        allocator.free(block)
        assert allocator.get_num_free_blocks() == 1000
    """

    def __init__(self, device: Device, num_blocks: int, block_size: int) -> None:
        self.device = device
        self.num_blocks = num_blocks
        self.block_size = block_size

        # The free list: a stack of available PhysicalBlock objects.
        # Using a list as a stack (append/pop) gives O(1) alloc/free.
        self._free_blocks: List[PhysicalBlock] = [
            PhysicalBlock(device=device, block_id=i, ref_count=0)
            for i in range(num_blocks)
        ]

        # Index for O(1) lookup by block_id (used during CoW and swap).
        self._all_blocks: Dict[int, PhysicalBlock] = {
            block.block_id: block for block in self._free_blocks
        }

    def allocate(self) -> PhysicalBlock:
        """Allocate a single physical block from the free pool.

        Returns:
            A PhysicalBlock with ref_count set to 1.

        Raises:
            RuntimeError: If no free blocks are available. The caller
                (KVCacheManager) should check availability before calling
                this, or handle preemption.
        """
        if not self._free_blocks:
            raise RuntimeError(
                f"Out of free {self.device.value} blocks! "
                f"Total pool size: {self.num_blocks}. "
                "The scheduler should preempt or reject requests before "
                "reaching this state."
            )
        block = self._free_blocks.pop()
        block.ref_count = 1
        return block

    def free(self, block: PhysicalBlock) -> None:
        """Return a physical block to the free pool.

        The block's ref_count is decremented. It is only returned to the
        free list when ref_count reaches 0 (i.e., no remaining references).

        This handles the Copy-on-Write case: if a block is shared by
        multiple sequences (ref_count > 1), freeing one reference does
        not release the physical memory.

        Args:
            block: The block to release.

        Raises:
            ValueError: If the block's ref_count is already 0.
        """
        if block.ref_count <= 0:
            raise ValueError(
                f"Double-free detected for block {block.block_id} "
                f"on {block.device.value}. ref_count is already {block.ref_count}."
            )
        block.ref_count -= 1
        if block.ref_count == 0:
            self._free_blocks.append(block)

    def increase_ref(self, block: PhysicalBlock) -> None:
        """Increment the reference count of a physical block.

        Used for Copy-on-Write: when a beam search fork occurs, the
        child sequence's logical block maps to the same physical block
        as the parent. Instead of copying the KV data, we just bump
        the ref_count. The copy is deferred until one of them writes
        new tokens (Copy-on-Write semantics).

        Args:
            block: The physical block to add a reference to.
        """
        block.ref_count += 1

    def get_num_free_blocks(self) -> int:
        """Return the number of currently available blocks."""
        return len(self._free_blocks)

    def get_num_allocated_blocks(self) -> int:
        """Return the number of currently allocated blocks."""
        return self.num_blocks - len(self._free_blocks)

    def get_block(self, block_id: int) -> PhysicalBlock:
        """Look up a physical block by its ID.

        Args:
            block_id: The unique block identifier.

        Returns:
            The PhysicalBlock object.

        Raises:
            KeyError: If the block_id is not in the pool.
        """
        if block_id not in self._all_blocks:
            raise KeyError(
                f"Block ID {block_id} not found in {self.device.value} pool."
            )
        return self._all_blocks[block_id]

    def __repr__(self) -> str:
        return (
            f"BlockAllocator(device={self.device.value}, "
            f"total={self.num_blocks}, "
            f"free={self.get_num_free_blocks()}, "
            f"allocated={self.get_num_allocated_blocks()})"
        )


# ---------------------------------------------------------------------------
# Sequence Block Table — per-sequence logical-to-physical mapping
# ---------------------------------------------------------------------------

@dataclass
class BlockTable:
    """Maps a sequence's logical block indices to physical blocks.

    Analogous to a **page table** in OS virtual memory. Each sequence
    has its own BlockTable that translates logical block numbers
    (0, 1, 2, ...) to physical block locations on GPU or CPU.

    Attributes:
        logical_to_physical: Ordered list of PhysicalBlock references.
            Index ``i`` represents logical block ``i`` for this sequence.
    """
    logical_to_physical: List[PhysicalBlock] = field(default_factory=list)

    @property
    def num_blocks(self) -> int:
        """Number of logical blocks currently allocated for this sequence."""
        return len(self.logical_to_physical)

    def append(self, block: PhysicalBlock) -> None:
        """Append a new physical block mapping (sequence grew by one block)."""
        self.logical_to_physical.append(block)

    def get_physical_block(self, logical_idx: int) -> PhysicalBlock:
        """Get the physical block for a given logical block index."""
        if logical_idx < 0 or logical_idx >= len(self.logical_to_physical):
            raise IndexError(
                f"Logical block index {logical_idx} out of range "
                f"[0, {len(self.logical_to_physical)})."
            )
        return self.logical_to_physical[logical_idx]

    def set_physical_block(self, logical_idx: int, block: PhysicalBlock) -> None:
        """Replace the physical block at a given logical index.

        Used during Copy-on-Write: when a shared block needs to be mutated,
        we allocate a new physical block, copy the data, and update the
        mapping here.
        """
        if logical_idx < 0 or logical_idx >= len(self.logical_to_physical):
            raise IndexError(
                f"Logical block index {logical_idx} out of range "
                f"[0, {len(self.logical_to_physical)})."
            )
        self.logical_to_physical[logical_idx] = block

    def get_physical_block_ids(self) -> List[int]:
        """Return a list of physical block IDs for the model runner.

        The model runner needs this to construct the block-table tensor
        that the PagedAttention kernel uses to look up KV cache data.
        """
        return [block.block_id for block in self.logical_to_physical]


# ---------------------------------------------------------------------------
# KV Cache Manager — top-level orchestrator
# ---------------------------------------------------------------------------

class KVCacheManager:
    """Top-level KV cache memory manager implementing PagedAttention.

    This class coordinates block allocation across GPU and CPU devices,
    manages per-sequence block tables, and implements Copy-on-Write for
    beam search.

    Architecture::

        ┌───────────────────────────────────┐
        │         KVCacheManager            │
        │                                   │
        │  ┌─────────────┐ ┌─────────────┐ │
        │  │ GPU Block   │ │ CPU Block   │ │
        │  │ Allocator   │ │ Allocator   │ │
        │  │ (1000 blks) │ │ (2000 blks) │ │
        │  └─────────────┘ └─────────────┘ │
        │                                   │
        │  Block Tables (per sequence):     │
        │  seq_0: [phys_3, phys_7, phys_1]  │
        │  seq_1: [phys_5, phys_9]          │
        │  seq_2: [phys_3, phys_12, ...]    │
        │          ↑ shared (CoW)           │
        └───────────────────────────────────┘

    The scheduler interacts with this manager to:
    1. Check if there's enough free blocks to admit a new request.
    2. Allocate blocks as sequences generate tokens.
    3. Free blocks when sequences finish.
    4. Swap blocks between GPU and CPU during preemption.
    5. Fork block tables during beam search (Copy-on-Write).

    Example::

        config = CacheConfig(
            block_size=16,
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            dtype_size=2,
            num_gpu_blocks=1000,
            num_cpu_blocks=2000,
        )
        manager = KVCacheManager(config)

        # New sequence arrives — allocate initial blocks for its prompt
        seq_id = 0
        manager.allocate_initial_blocks(seq_id, prompt_length=45)

        # During decoding, allocate one more block when the sequence
        # crosses a block boundary
        manager.allocate_block(seq_id, device=Device.GPU)

        # Sequence finishes — free all its blocks
        manager.free_sequence(seq_id)
    """

    def __init__(self, config: CacheConfig) -> None:
        self.config = config
        self.block_size = config.block_size

        # Create per-device block allocators
        self.gpu_allocator = BlockAllocator(
            device=Device.GPU,
            num_blocks=config.num_gpu_blocks,
            block_size=config.block_size,
        )
        self.cpu_allocator = BlockAllocator(
            device=Device.CPU,
            num_blocks=config.num_cpu_blocks,
            block_size=config.block_size,
        )

        # Per-sequence block tables: seq_id → BlockTable
        self._block_tables: Dict[int, BlockTable] = {}

    # ---- Query methods ----

    def get_num_free_gpu_blocks(self) -> int:
        """Number of free physical blocks on GPU."""
        return self.gpu_allocator.get_num_free_blocks()

    def get_num_free_cpu_blocks(self) -> int:
        """Number of free physical blocks on CPU."""
        return self.cpu_allocator.get_num_free_blocks()

    def can_allocate(self, num_required_blocks: int) -> bool:
        """Check if the GPU has enough free blocks for a new allocation.

        The scheduler calls this before admitting a new request. If this
        returns False, the scheduler must either wait or preempt.

        Args:
            num_required_blocks: Number of blocks the new sequence needs
                (typically ``ceil(prompt_length / block_size)``).
        """
        return self.gpu_allocator.get_num_free_blocks() >= num_required_blocks

    def blocks_needed_for_tokens(self, num_tokens: int) -> int:
        """Calculate how many blocks are needed to store ``num_tokens``.

        Args:
            num_tokens: Number of tokens (e.g., prompt length).

        Returns:
            Number of blocks (ceiling division).
        """
        return (num_tokens + self.block_size - 1) // self.block_size

    def get_block_table(self, seq_id: int) -> BlockTable:
        """Get the block table for a sequence.

        Raises:
            KeyError: If the sequence has not been registered.
        """
        if seq_id not in self._block_tables:
            raise KeyError(f"Sequence {seq_id} has no block table.")
        return self._block_tables[seq_id]

    def get_block_table_ids(self, seq_id: int) -> List[int]:
        """Get the physical block IDs for a sequence (for the model runner)."""
        return self.get_block_table(seq_id).get_physical_block_ids()

    # ---- Allocation methods ----

    def allocate_initial_blocks(
        self,
        seq_id: int,
        prompt_length: int,
        device: Device = Device.GPU,
    ) -> BlockTable:
        """Allocate the initial set of blocks for a new sequence.

        Called when a prompt is first admitted to the running batch.
        Allocates enough blocks to hold the full prompt.

        Args:
            seq_id: Unique sequence identifier.
            prompt_length: Number of tokens in the prompt.
            device: Device to allocate on (default GPU).

        Returns:
            The newly created BlockTable for this sequence.

        Raises:
            RuntimeError: If not enough free blocks are available.
        """
        num_blocks = self.blocks_needed_for_tokens(prompt_length)
        allocator = self._get_allocator(device)

        if allocator.get_num_free_blocks() < num_blocks:
            raise RuntimeError(
                f"Cannot allocate {num_blocks} blocks for sequence {seq_id}: "
                f"only {allocator.get_num_free_blocks()} free on {device.value}."
            )

        block_table = BlockTable()
        for _ in range(num_blocks):
            physical_block = allocator.allocate()
            block_table.append(physical_block)

        self._block_tables[seq_id] = block_table
        return block_table

    def allocate_block(
        self,
        seq_id: int,
        device: Device = Device.GPU,
    ) -> PhysicalBlock:
        """Allocate a single additional block for an existing sequence.

        Called during decoding when the sequence crosses a block boundary
        (i.e., ``len(tokens) % block_size == 0``).

        Args:
            seq_id: Sequence identifier.
            device: Device to allocate on.

        Returns:
            The newly allocated PhysicalBlock.
        """
        allocator = self._get_allocator(device)
        block = allocator.allocate()
        self._block_tables[seq_id].append(block)
        return block

    # ---- Deallocation methods ----

    def free_sequence(self, seq_id: int) -> None:
        """Free all blocks owned by a sequence.

        Called when a sequence finishes (EOS token or max length reached).
        Decrements ref_count on each physical block; blocks with ref_count=0
        are returned to the free pool.

        Args:
            seq_id: Sequence to free.
        """
        if seq_id not in self._block_tables:
            return  # Already freed or never allocated

        block_table = self._block_tables.pop(seq_id)
        for physical_block in block_table.logical_to_physical:
            allocator = self._get_allocator(physical_block.device)
            allocator.free(physical_block)

    # ---- Copy-on-Write for beam search ----

    def fork_sequence(self, parent_seq_id: int, child_seq_id: int) -> BlockTable:
        """Create a CoW fork of a sequence's block table (beam search).

        Instead of copying all KV cache data, we share the physical blocks
        by incrementing their ref_count. This is exactly like ``fork()`` in
        Unix: the child gets a copy of the page table pointing to the same
        physical frames, and we defer actual copying until a write occurs.

        Args:
            parent_seq_id: The source sequence to fork from.
            child_seq_id: The new sequence that shares the parent's cache.

        Returns:
            The child's new BlockTable (sharing physical blocks with parent).
        """
        parent_table = self.get_block_table(parent_seq_id)
        child_table = BlockTable()

        for physical_block in parent_table.logical_to_physical:
            # Increment ref_count — this block is now shared (CoW)
            allocator = self._get_allocator(physical_block.device)
            allocator.increase_ref(physical_block)
            child_table.append(physical_block)

        self._block_tables[child_seq_id] = child_table
        return child_table

    def copy_on_write(
        self,
        seq_id: int,
        logical_block_idx: int,
    ) -> Tuple[PhysicalBlock, PhysicalBlock]:
        """Perform a Copy-on-Write for a shared block.

        When a sequence needs to write to a block that has ref_count > 1
        (shared via beam search), we must:
        1. Allocate a new physical block.
        2. Copy the data from the old block to the new one (in practice,
           this would be a GPU memcpy — here we just track the metadata).
        3. Update the block table to point to the new block.
        4. Decrement the old block's ref_count.

        Args:
            seq_id: The sequence that needs to write.
            logical_block_idx: Which logical block is being written to.

        Returns:
            Tuple of (old_block, new_block) for the caller to issue
            the actual GPU memcpy.
        """
        block_table = self.get_block_table(seq_id)
        old_block = block_table.get_physical_block(logical_block_idx)

        if not old_block.is_shared():
            # No CoW needed — this sequence has exclusive access
            return old_block, old_block

        # Allocate a fresh block on the same device
        allocator = self._get_allocator(old_block.device)
        new_block = allocator.allocate()

        # Update the block table to point to the new block
        block_table.set_physical_block(logical_block_idx, new_block)

        # Release our reference to the old block
        allocator.free(old_block)

        return old_block, new_block

    # ---- Swap operations (GPU ↔ CPU for preemption) ----

    def swap_out(self, seq_id: int) -> Dict[int, int]:
        """Swap a sequence's blocks from GPU to CPU.

        Called by the scheduler when a sequence is preempted due to memory
        pressure. The KV cache data is moved to CPU memory so the GPU blocks
        can be reused.

        Args:
            seq_id: Sequence to swap out.

        Returns:
            Mapping of GPU block_id → CPU block_id (for the cache engine
            to perform the actual data transfer).
        """
        block_table = self.get_block_table(seq_id)
        gpu_to_cpu_mapping: Dict[int, int] = {}

        new_logical_to_physical: List[PhysicalBlock] = []
        for gpu_block in block_table.logical_to_physical:
            if gpu_block.device != Device.GPU:
                # Already on CPU (shouldn't happen in normal flow)
                new_logical_to_physical.append(gpu_block)
                continue

            # Allocate a CPU block
            cpu_block = self.cpu_allocator.allocate()
            gpu_to_cpu_mapping[gpu_block.block_id] = cpu_block.block_id

            # Free the GPU block
            self.gpu_allocator.free(gpu_block)

            new_logical_to_physical.append(cpu_block)

        block_table.logical_to_physical = new_logical_to_physical
        return gpu_to_cpu_mapping

    def swap_in(self, seq_id: int) -> Dict[int, int]:
        """Swap a sequence's blocks from CPU back to GPU.

        Called by the scheduler when a previously swapped sequence is
        being resumed.

        Args:
            seq_id: Sequence to swap in.

        Returns:
            Mapping of CPU block_id → GPU block_id.
        """
        block_table = self.get_block_table(seq_id)
        cpu_to_gpu_mapping: Dict[int, int] = {}

        new_logical_to_physical: List[PhysicalBlock] = []
        for cpu_block in block_table.logical_to_physical:
            if cpu_block.device != Device.CPU:
                new_logical_to_physical.append(cpu_block)
                continue

            # Allocate a GPU block
            gpu_block = self.gpu_allocator.allocate()
            cpu_to_gpu_mapping[cpu_block.block_id] = gpu_block.block_id

            # Free the CPU block
            self.cpu_allocator.free(cpu_block)

            new_logical_to_physical.append(gpu_block)

        block_table.logical_to_physical = new_logical_to_physical
        return cpu_to_gpu_mapping

    # ---- Internal helpers ----

    def _get_allocator(self, device: Device) -> BlockAllocator:
        """Get the block allocator for a given device."""
        if device == Device.GPU:
            return self.gpu_allocator
        elif device == Device.CPU:
            return self.cpu_allocator
        else:
            raise ValueError(f"Unknown device: {device}")

    # ---- Status and debugging ----

    def get_status(self) -> Dict[str, object]:
        """Return a summary of current memory state for debugging."""
        return {
            "gpu_free_blocks": self.gpu_allocator.get_num_free_blocks(),
            "gpu_allocated_blocks": self.gpu_allocator.get_num_allocated_blocks(),
            "gpu_total_blocks": self.gpu_allocator.num_blocks,
            "cpu_free_blocks": self.cpu_allocator.get_num_free_blocks(),
            "cpu_allocated_blocks": self.cpu_allocator.get_num_allocated_blocks(),
            "cpu_total_blocks": self.cpu_allocator.num_blocks,
            "active_sequences": len(self._block_tables),
            "block_memory_mb": self.config.block_memory_mb,
        }

    def __repr__(self) -> str:
        return (
            f"KVCacheManager(\n"
            f"  gpu={self.gpu_allocator},\n"
            f"  cpu={self.cpu_allocator},\n"
            f"  active_sequences={len(self._block_tables)},\n"
            f"  block_size={self.block_size},\n"
            f"  block_memory={self.config.block_memory_mb:.2f} MB\n"
            f")"
        )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def create_kv_cache_manager(
    *,
    block_size: int = 16,
    num_layers: int = 32,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    dtype_size: int = 2,
    gpu_memory_for_kv_bytes: int = 0,
    cpu_memory_for_kv_bytes: int = 0,
) -> KVCacheManager:
    """Factory function that calculates block counts from raw memory budgets.

    This bridges the gap between "I have X GB of GPU memory for KV cache"
    and the block-level allocation interface.

    Args:
        block_size: Tokens per block.
        num_layers: Transformer layers.
        num_kv_heads: Number of KV attention heads.
        head_dim: Attention head dimension.
        dtype_size: Bytes per element (2 for FP16/BF16, 1 for FP8/INT8).
        gpu_memory_for_kv_bytes: GPU memory budget for KV cache in bytes.
        cpu_memory_for_kv_bytes: CPU memory budget for KV cache in bytes.

    Returns:
        A configured KVCacheManager instance.

    Example::

        # Llama-3-8B on a 24GB GPU with 8GB available for KV cache
        manager = create_kv_cache_manager(
            block_size=16,
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            dtype_size=2,                       # FP16
            gpu_memory_for_kv_bytes=8 * (1024**3),  # 8 GB
            cpu_memory_for_kv_bytes=16 * (1024**3),  # 16 GB
        )
        print(manager)
    """
    config = CacheConfig(
        block_size=block_size,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype_size=dtype_size,
    )

    block_mem = config.block_memory_bytes
    num_gpu_blocks = gpu_memory_for_kv_bytes // block_mem if block_mem > 0 else 0
    num_cpu_blocks = cpu_memory_for_kv_bytes // block_mem if block_mem > 0 else 0

    config = CacheConfig(
        block_size=block_size,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype_size=dtype_size,
        num_gpu_blocks=num_gpu_blocks,
        num_cpu_blocks=num_cpu_blocks,
    )

    return KVCacheManager(config)


# ---------------------------------------------------------------------------
# Self-test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("KV Cache Manager — PagedAttention Demo")
    print("=" * 70)

    # Simulate Llama-3-8B config with 8 GB KV cache budget
    manager = create_kv_cache_manager(
        block_size=16,
        num_layers=32,
        num_kv_heads=8,
        head_dim=128,
        dtype_size=2,
        gpu_memory_for_kv_bytes=8 * (1024 ** 3),
        cpu_memory_for_kv_bytes=16 * (1024 ** 3),
    )
    print(f"\nInitialized: {manager}")
    print(f"Status: {manager.get_status()}")

    # Simulate three sequences arriving
    print("\n--- Allocating sequences ---")
    for seq_id, prompt_len in [(0, 45), (1, 120), (2, 200)]:
        blocks_needed = manager.blocks_needed_for_tokens(prompt_len)
        print(f"Seq {seq_id}: prompt_len={prompt_len}, blocks_needed={blocks_needed}")
        manager.allocate_initial_blocks(seq_id, prompt_len)

    print(f"\nAfter allocation: {manager.gpu_allocator}")

    # Simulate beam search fork (Copy-on-Write)
    print("\n--- Beam search fork (seq_0 → seq_3) ---")
    manager.fork_sequence(parent_seq_id=0, child_seq_id=3)
    parent_blocks = manager.get_block_table_ids(0)
    child_blocks = manager.get_block_table_ids(3)
    print(f"Parent block IDs: {parent_blocks}")
    print(f"Child  block IDs: {child_blocks}  (same physical blocks — CoW!)")
    parent_table = manager.get_block_table(0)
    print(f"Shared block ref_counts: {[b.ref_count for b in parent_table.logical_to_physical]}")

    # Simulate CoW trigger
    print("\n--- Copy-on-Write trigger on seq_3, block 0 ---")
    old, new = manager.copy_on_write(seq_id=3, logical_block_idx=0)
    print(f"Old block: {old.block_id} (ref_count={old.ref_count})")
    print(f"New block: {new.block_id} (ref_count={new.ref_count})")

    # Simulate swap out
    print("\n--- Swapping out seq_1 to CPU ---")
    mapping = manager.swap_out(seq_id=1)
    print(f"GPU→CPU block mapping: {mapping}")
    print(f"GPU free after swap: {manager.get_num_free_gpu_blocks()}")

    # Free a finished sequence
    print("\n--- Freeing finished seq_2 ---")
    manager.free_sequence(seq_id=2)
    print(f"GPU free after free: {manager.get_num_free_gpu_blocks()}")

    print(f"\nFinal status: {manager.get_status()}")
