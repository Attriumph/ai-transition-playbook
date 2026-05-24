"""
KV Cache Memory Calculator
===========================

A mathematical VRAM budget planner for LLM inference. This utility answers
critical capacity-planning questions:

    1. "How much GPU memory does the KV cache need for N concurrent
       sequences of length L?"

    2. "After loading Llama-3-70B in FP16, how many KV cache blocks
       fit in my remaining GPU memory?"

    3. "What's the maximum batch size I can sustain at 4096 context
       length on an 80GB A100?"

    4. "How much memory do I save by quantizing KV cache to FP8?"

Memory Breakdown for LLM Inference
-----------------------------------

GPU memory is consumed by three major components:

    ┌─────────────────────────────────────────────────────────┐
    │                    GPU VRAM (e.g., 80 GB)               │
    │                                                         │
    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
    │  │ Model Weights │  │  KV Cache    │  │  Activations │  │
    │  │  (fixed)      │  │ (dynamic)    │  │ (transient)  │  │
    │  │              │  │              │  │              │  │
    │  │ ~14 GB (7B)  │  │ 0 – 60+ GB  │  │ ~1-2 GB     │  │
    │  │ ~140 GB (70B)│  │ (grows with  │  │ (per-batch)  │  │
    │  │              │  │  batch size) │  │              │  │
    │  └──────────────┘  └──────────────┘  └──────────────┘  │
    └─────────────────────────────────────────────────────────┘

The KV cache is the only component that scales with concurrent users,
making it the critical bottleneck for serving throughput.

KV Cache Memory Formula
------------------------

Per-token KV cache memory:

    kv_per_token = num_layers × 2 × num_kv_heads × head_dim × dtype_size
                                 ↑
                         (Key + Value tensors)

Per-sequence KV cache memory:

    kv_per_seq = kv_per_token × seq_length

Total KV cache for a batch:

    kv_total = kv_per_seq × batch_size

Block-level calculation:

    block_memory = block_size × kv_per_token
    total_blocks = available_gpu_memory // block_memory

Supported dtypes:
    - FP16 / BF16: 2 bytes per element (standard)
    - FP8 (E4M3 / E5M2): 1 byte per element (H100 native)
    - INT8: 1 byte per element (quantized KV cache)

References
----------
- Kwon et al., "Efficient Memory Management for Large Language Model Serving
  with PagedAttention", SOSP 2023.
- Pope et al., "Efficiently Scaling Transformer Inference", MLSys 2023.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class KVDtype(Enum):
    """Supported data types for KV cache storage.

    Each variant stores (name, bytes_per_element, description).
    """
    FP16 = ("fp16", 2, "IEEE 754 half-precision (16-bit)")
    BF16 = ("bf16", 2, "Brain floating-point (16-bit)")
    FP8 = ("fp8", 1, "8-bit float (E4M3 or E5M2, H100 native)")
    INT8 = ("int8", 1, "8-bit integer (quantized KV cache)")

    def __init__(self, label: str, size: int, description: str) -> None:
        self.label = label
        self.size = size
        self.description = description

    @classmethod
    def from_string(cls, s: str) -> "KVDtype":
        """Parse a dtype string like 'fp16', 'bf16', 'fp8', 'int8'."""
        normalized = s.lower().strip()
        for member in cls:
            if member.label == normalized:
                return member
        valid = ", ".join(m.label for m in cls)
        raise ValueError(f"Unknown dtype '{s}'. Valid options: {valid}")


# ---------------------------------------------------------------------------
# Model configuration presets
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    """Model architecture parameters relevant to KV cache sizing.

    Attributes:
        name: Human-readable model name.
        num_layers: Number of transformer layers (decoder blocks).
        num_attention_heads: Total number of query attention heads.
        num_kv_heads: Number of key/value heads. For GQA models, this is
            less than num_attention_heads. For MHA, they're equal.
        head_dim: Dimension of each attention head.
        hidden_dim: Model hidden dimension (for reference).
        vocab_size: Vocabulary size (for reference).
        model_size_billions: Approximate parameter count in billions.
        weight_memory_gb: Approximate GPU memory for model weights in FP16.
    """
    name: str
    num_layers: int
    num_attention_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_dim: int
    vocab_size: int
    model_size_billions: float
    weight_memory_gb: float


# Well-known model presets for quick calculations
MODEL_PRESETS: Dict[str, ModelConfig] = {
    "llama-3-8b": ModelConfig(
        name="Llama-3-8B",
        num_layers=32,
        num_attention_heads=32,
        num_kv_heads=8,       # GQA: 4:1 ratio
        head_dim=128,
        hidden_dim=4096,
        vocab_size=128256,
        model_size_billions=8.03,
        weight_memory_gb=16.06,
    ),
    "llama-3-70b": ModelConfig(
        name="Llama-3-70B",
        num_layers=80,
        num_attention_heads=64,
        num_kv_heads=8,       # GQA: 8:1 ratio
        head_dim=128,
        hidden_dim=8192,
        vocab_size=128256,
        model_size_billions=70.6,
        weight_memory_gb=141.0,
    ),
    "llama-3-405b": ModelConfig(
        name="Llama-3.1-405B",
        num_layers=126,
        num_attention_heads=128,
        num_kv_heads=8,       # GQA: 16:1 ratio
        head_dim=128,
        hidden_dim=16384,
        vocab_size=128256,
        model_size_billions=405.0,
        weight_memory_gb=810.0,
    ),
    "mistral-7b": ModelConfig(
        name="Mistral-7B",
        num_layers=32,
        num_attention_heads=32,
        num_kv_heads=8,       # GQA
        head_dim=128,
        hidden_dim=4096,
        vocab_size=32000,
        model_size_billions=7.24,
        weight_memory_gb=14.48,
    ),
    "qwen2-72b": ModelConfig(
        name="Qwen2-72B",
        num_layers=80,
        num_attention_heads=64,
        num_kv_heads=8,       # GQA
        head_dim=128,
        hidden_dim=8192,
        vocab_size=152064,
        model_size_billions=72.7,
        weight_memory_gb=145.4,
    ),
}


# ---------------------------------------------------------------------------
# Memory Calculator
# ---------------------------------------------------------------------------

@dataclass
class MemoryReport:
    """Detailed memory breakdown report.

    All memory values are in bytes unless suffixed with _gb or _mb.

    Attributes:
        model_name: Name of the model.
        kv_dtype: Data type used for KV cache.
        kv_per_token_bytes: Memory per token across all layers.
        kv_per_token_per_layer_bytes: Memory per token per layer.
        kv_per_sequence_bytes: Memory for one sequence at max length.
        kv_total_batch_bytes: Memory for the full batch.
        block_size: Tokens per block.
        block_memory_bytes: Memory per block.
        num_blocks_available: Blocks that fit in available GPU memory.
        max_concurrent_sequences: Max sequences at given context length.
        model_weight_memory_bytes: Memory consumed by model weights.
        gpu_total_memory_bytes: Total GPU memory.
        gpu_available_for_kv_bytes: GPU memory available for KV cache.
        activation_memory_estimate_bytes: Estimated activation memory.
        seq_length: Target sequence length for calculations.
        batch_size: Target batch size for calculations.
        num_layers: Number of transformer layers.
        num_kv_heads: Number of KV attention heads.
        head_dim: Head dimension.
    """
    model_name: str
    kv_dtype: KVDtype
    kv_per_token_bytes: int
    kv_per_token_per_layer_bytes: int
    kv_per_sequence_bytes: int
    kv_total_batch_bytes: int
    block_size: int
    block_memory_bytes: int
    num_blocks_available: int
    max_concurrent_sequences: int
    model_weight_memory_bytes: int
    gpu_total_memory_bytes: int
    gpu_available_for_kv_bytes: int
    activation_memory_estimate_bytes: int
    seq_length: int
    batch_size: int
    num_layers: int
    num_kv_heads: int
    head_dim: int

    def _bytes_to_gb(self, b: int) -> float:
        return b / (1024 ** 3)

    def _bytes_to_mb(self, b: int) -> float:
        return b / (1024 ** 2)

    def print_report(self) -> str:
        """Generate a formatted, human-readable memory report."""
        lines = []
        sep = "=" * 72
        thin_sep = "-" * 72

        lines.append(sep)
        lines.append(f"  KV Cache Memory Report: {self.model_name}")
        lines.append(sep)

        # Model architecture
        lines.append("")
        lines.append("  Model Architecture:")
        lines.append(thin_sep)
        lines.append(f"    Layers:             {self.num_layers}")
        lines.append(f"    KV Heads:           {self.num_kv_heads}")
        lines.append(f"    Head Dim:           {self.head_dim}")
        lines.append(f"    KV Dtype:           {self.kv_dtype.label} ({self.kv_dtype.description})")
        lines.append(f"    Block Size:         {self.block_size} tokens")

        # Per-token KV memory
        lines.append("")
        lines.append("  Per-Token KV Cache Memory:")
        lines.append(thin_sep)
        lines.append(
            f"    Per layer:          {self.kv_per_token_per_layer_bytes:,} bytes "
            f"({self._bytes_to_mb(self.kv_per_token_per_layer_bytes):.4f} MB)"
        )
        lines.append(
            f"    formula: 2 × {self.num_kv_heads} × {self.head_dim} × "
            f"{self.kv_dtype.size} = {self.kv_per_token_per_layer_bytes:,} bytes"
        )
        lines.append(
            f"    All layers:         {self.kv_per_token_bytes:,} bytes "
            f"({self._bytes_to_mb(self.kv_per_token_bytes):.4f} MB)"
        )
        lines.append(
            f"    formula: {self.num_layers} × {self.kv_per_token_per_layer_bytes:,} = "
            f"{self.kv_per_token_bytes:,} bytes"
        )

        # Per-sequence KV memory
        lines.append("")
        lines.append(f"  Per-Sequence KV Cache (seq_len={self.seq_length:,}):")
        lines.append(thin_sep)
        lines.append(
            f"    Memory:             {self.kv_per_sequence_bytes:,} bytes "
            f"({self._bytes_to_gb(self.kv_per_sequence_bytes):.4f} GB)"
        )

        # Batch KV memory
        lines.append("")
        lines.append(f"  Batch KV Cache (batch_size={self.batch_size}, seq_len={self.seq_length:,}):")
        lines.append(thin_sep)
        lines.append(
            f"    Memory:             {self.kv_total_batch_bytes:,} bytes "
            f"({self._bytes_to_gb(self.kv_total_batch_bytes):.4f} GB)"
        )

        # GPU memory breakdown
        lines.append("")
        lines.append("  GPU Memory Breakdown:")
        lines.append(thin_sep)
        lines.append(
            f"    Total GPU Memory:   {self._bytes_to_gb(self.gpu_total_memory_bytes):.1f} GB"
        )
        lines.append(
            f"    Model Weights:      {self._bytes_to_gb(self.model_weight_memory_bytes):.2f} GB"
        )
        lines.append(
            f"    Activations (est.): {self._bytes_to_gb(self.activation_memory_estimate_bytes):.2f} GB"
        )
        lines.append(
            f"    Available for KV:   {self._bytes_to_gb(self.gpu_available_for_kv_bytes):.2f} GB"
        )

        # Block-level analysis
        lines.append("")
        lines.append("  PagedAttention Block Analysis:")
        lines.append(thin_sep)
        lines.append(
            f"    Block memory:       {self.block_memory_bytes:,} bytes "
            f"({self._bytes_to_mb(self.block_memory_bytes):.2f} MB)"
        )
        lines.append(
            f"    formula: {self.block_size} × {self.num_layers} × 2 × "
            f"{self.num_kv_heads} × {self.head_dim} × {self.kv_dtype.size}"
        )
        lines.append(
            f"    Available blocks:   {self.num_blocks_available:,}"
        )
        lines.append(
            f"    Total block memory: "
            f"{self._bytes_to_gb(self.num_blocks_available * self.block_memory_bytes):.2f} GB"
        )

        # Capacity estimates
        blocks_per_seq = math.ceil(self.seq_length / self.block_size)
        lines.append("")
        lines.append(f"  Capacity Estimates (at seq_len={self.seq_length:,}):")
        lines.append(thin_sep)
        lines.append(f"    Blocks per sequence: {blocks_per_seq}")
        lines.append(f"    Max concurrent seqs: {self.max_concurrent_sequences}")

        if self.max_concurrent_sequences > 0:
            effective_throughput_tokens = self.max_concurrent_sequences * self.seq_length
            lines.append(
                f"    Max total KV tokens: {effective_throughput_tokens:,}"
            )

        # Dtype comparison
        lines.append("")
        lines.append("  Dtype Comparison (same config, varying precision):")
        lines.append(thin_sep)
        lines.append(f"    {'Dtype':<8} {'Per-Token':>12} {'Per-Seq':>12} {'Max Seqs':>10} {'Savings':>10}")
        lines.append(f"    {'─'*8} {'─'*12} {'─'*12} {'─'*10} {'─'*10}")

        fp16_per_token = self.num_layers * 2 * self.num_kv_heads * self.head_dim * 2
        for dtype in KVDtype:
            per_token = self.num_layers * 2 * self.num_kv_heads * self.head_dim * dtype.size
            per_seq = per_token * self.seq_length
            block_mem = self.block_size * per_token
            n_blocks = self.gpu_available_for_kv_bytes // block_mem if block_mem > 0 else 0
            b_per_seq = math.ceil(self.seq_length / self.block_size)
            max_seqs = n_blocks // b_per_seq if b_per_seq > 0 else 0
            savings = f"{(1 - dtype.size / 2) * 100:.0f}%" if dtype.size < 2 else "baseline"
            lines.append(
                f"    {dtype.label:<8} {self._bytes_to_mb(per_token):>9.4f} MB "
                f"{self._bytes_to_gb(per_seq):>9.4f} GB "
                f"{max_seqs:>10,} "
                f"{savings:>10}"
            )

        lines.append("")
        lines.append(sep)

        report = "\n".join(lines)
        print(report)
        return report


class MemoryCalculator:
    """Calculate KV cache memory requirements for LLM inference.

    This calculator provides exact memory sizing based on model architecture
    parameters. It supports custom configurations and well-known model presets.

    The core formula:

        kv_per_token = num_layers × 2 × num_kv_heads × head_dim × dtype_size

    where the factor of 2 accounts for both Key and Value tensors.

    Example::

        calc = MemoryCalculator(
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            dtype=KVDtype.FP16,
        )
        report = calc.calculate(
            gpu_memory_gb=24.0,
            model_size_gb=16.0,
            seq_length=4096,
            batch_size=32,
        )
        report.print_report()
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: KVDtype = KVDtype.FP16,
        block_size: int = 16,
        model_name: str = "Custom Model",
    ) -> None:
        """Initialize the calculator with model architecture parameters.

        Args:
            num_layers: Number of transformer decoder layers.
            num_kv_heads: Number of key/value attention heads. For GQA models,
                this is typically num_attention_heads / group_size (e.g., 8
                for Llama-3-8B which has 32 Q heads grouped into 8 KV heads).
            head_dim: Dimension of each attention head (e.g., 128).
            dtype: Data type for KV cache storage.
            block_size: Number of tokens per PagedAttention block.
            model_name: Name for the report header.
        """
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.block_size = block_size
        self.model_name = model_name

    @classmethod
    def from_preset(
        cls,
        preset_name: str,
        dtype: KVDtype = KVDtype.FP16,
        block_size: int = 16,
    ) -> "MemoryCalculator":
        """Create a calculator from a well-known model preset.

        Args:
            preset_name: One of: llama-3-8b, llama-3-70b, llama-3-405b,
                mistral-7b, qwen2-72b.
            dtype: KV cache data type.
            block_size: Tokens per block.

        Returns:
            Configured MemoryCalculator.

        Raises:
            KeyError: If the preset name is not recognized.
        """
        preset_key = preset_name.lower().strip()
        if preset_key not in MODEL_PRESETS:
            valid = ", ".join(MODEL_PRESETS.keys())
            raise KeyError(
                f"Unknown model preset '{preset_name}'. "
                f"Valid presets: {valid}"
            )
        config = MODEL_PRESETS[preset_key]
        return cls(
            num_layers=config.num_layers,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            dtype=dtype,
            block_size=block_size,
            model_name=config.name,
        )

    # ---- Core calculations ----

    def kv_per_token_per_layer_bytes(self) -> int:
        """Memory for one token's KV cache in a single layer.

        Formula: 2 × num_kv_heads × head_dim × dtype_size

        The factor of 2 is for Key and Value tensors.
        """
        return 2 * self.num_kv_heads * self.head_dim * self.dtype.size

    def kv_per_token_bytes(self) -> int:
        """Memory for one token's KV cache across ALL layers.

        Formula: num_layers × 2 × num_kv_heads × head_dim × dtype_size
        """
        return self.num_layers * self.kv_per_token_per_layer_bytes()

    def kv_per_sequence_bytes(self, seq_length: int) -> int:
        """Memory for one complete sequence's KV cache.

        Formula: seq_length × kv_per_token_bytes
        """
        return seq_length * self.kv_per_token_bytes()

    def kv_batch_bytes(self, batch_size: int, seq_length: int) -> int:
        """Total KV cache memory for a batch of sequences.

        Formula: batch_size × seq_length × kv_per_token_bytes
        """
        return batch_size * self.kv_per_sequence_bytes(seq_length)

    def block_memory_bytes(self) -> int:
        """Memory consumed by a single PagedAttention block.

        Formula: block_size × num_layers × 2 × num_kv_heads × head_dim × dtype_size
        """
        return self.block_size * self.kv_per_token_bytes()

    def calculate_available_blocks(
        self,
        gpu_memory_bytes: int,
        model_weight_bytes: int,
        activation_memory_bytes: int = 0,
        overhead_fraction: float = 0.05,
    ) -> int:
        """Calculate how many KV cache blocks fit in available GPU memory.

        The available memory for KV cache is:

            available = gpu_total - model_weights - activations - overhead

        Then:

            num_blocks = available // block_memory

        Args:
            gpu_memory_bytes: Total GPU memory in bytes.
            model_weight_bytes: Memory consumed by model weights in bytes.
            activation_memory_bytes: Estimated activation memory. If 0,
                uses a heuristic of ~2% of GPU memory.
            overhead_fraction: Fraction of GPU memory reserved for CUDA
                context, fragmentation, etc. Default 5%.

        Returns:
            Number of blocks that fit in the remaining GPU memory.
        """
        overhead = int(gpu_memory_bytes * overhead_fraction)

        if activation_memory_bytes == 0:
            # Heuristic: activations are roughly 1-2% of total GPU memory
            # for typical batch sizes. This is a rough estimate.
            activation_memory_bytes = int(gpu_memory_bytes * 0.02)

        available = gpu_memory_bytes - model_weight_bytes - activation_memory_bytes - overhead
        available = max(0, available)

        block_mem = self.block_memory_bytes()
        if block_mem == 0:
            return 0
        return available // block_mem

    def max_concurrent_sequences(
        self,
        num_blocks: int,
        seq_length: int,
    ) -> int:
        """Calculate max concurrent sequences given available blocks.

        Args:
            num_blocks: Total available KV cache blocks.
            seq_length: Target sequence length.

        Returns:
            Maximum number of sequences that can run concurrently.
        """
        blocks_per_seq = math.ceil(seq_length / self.block_size)
        if blocks_per_seq == 0:
            return 0
        return num_blocks // blocks_per_seq

    # ---- Full report generation ----

    def calculate(
        self,
        gpu_memory_gb: float = 80.0,
        model_size_gb: float = 16.0,
        seq_length: int = 4096,
        batch_size: int = 32,
        activation_memory_gb: float = 0.0,
    ) -> MemoryReport:
        """Run a full memory analysis and produce a detailed report.

        Args:
            gpu_memory_gb: Total GPU memory in GB (e.g., 80 for A100-80GB).
            model_size_gb: Model weights memory in GB (e.g., 16 for 8B FP16).
            seq_length: Target sequence length for capacity analysis.
            batch_size: Target batch size for total memory calculation.
            activation_memory_gb: Activation memory estimate in GB. If 0,
                uses a heuristic.

        Returns:
            A MemoryReport with all calculated values.
        """
        GB = 1024 ** 3

        gpu_memory_bytes = int(gpu_memory_gb * GB)
        model_weight_bytes = int(model_size_gb * GB)
        activation_bytes = int(activation_memory_gb * GB) if activation_memory_gb > 0 else 0

        num_blocks = self.calculate_available_blocks(
            gpu_memory_bytes=gpu_memory_bytes,
            model_weight_bytes=model_weight_bytes,
            activation_memory_bytes=activation_bytes,
        )

        # Recalculate actual activation estimate for the report
        if activation_bytes == 0:
            activation_bytes = int(gpu_memory_bytes * 0.02)

        overhead_bytes = int(gpu_memory_bytes * 0.05)
        available_for_kv = max(
            0,
            gpu_memory_bytes - model_weight_bytes - activation_bytes - overhead_bytes,
        )

        max_seqs = self.max_concurrent_sequences(num_blocks, seq_length)

        return MemoryReport(
            model_name=self.model_name,
            kv_dtype=self.dtype,
            kv_per_token_bytes=self.kv_per_token_bytes(),
            kv_per_token_per_layer_bytes=self.kv_per_token_per_layer_bytes(),
            kv_per_sequence_bytes=self.kv_per_sequence_bytes(seq_length),
            kv_total_batch_bytes=self.kv_batch_bytes(batch_size, seq_length),
            block_size=self.block_size,
            block_memory_bytes=self.block_memory_bytes(),
            num_blocks_available=num_blocks,
            max_concurrent_sequences=max_seqs,
            model_weight_memory_bytes=model_weight_bytes,
            gpu_total_memory_bytes=gpu_memory_bytes,
            gpu_available_for_kv_bytes=available_for_kv,
            activation_memory_estimate_bytes=activation_bytes,
            seq_length=seq_length,
            batch_size=batch_size,
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
        )


# ---------------------------------------------------------------------------
# CLI interface
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="KV Cache Memory Calculator — VRAM budget planner for LLM inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Llama-3-8B on a 24GB GPU
  python -m engine.memory_calculator \\
      --num-layers 32 --num-kv-heads 8 --head-dim 128 \\
      --dtype fp16 --gpu-memory-gb 24 --model-size-gb 16

  # Use a preset
  python -m engine.memory_calculator \\
      --preset llama-3-70b --dtype fp16 \\
      --gpu-memory-gb 80 --seq-length 8192

  # Compare FP16 vs FP8 on Llama-3-8B
  python -m engine.memory_calculator \\
      --preset llama-3-8b --dtype fp8 \\
      --gpu-memory-gb 24 --seq-length 4096

Available presets: llama-3-8b, llama-3-70b, llama-3-405b, mistral-7b, qwen2-72b
        """,
    )

    # Model architecture (manual)
    arch_group = parser.add_argument_group("Model Architecture (manual)")
    arch_group.add_argument("--num-layers", type=int, help="Number of transformer layers")
    arch_group.add_argument("--num-kv-heads", type=int, help="Number of KV attention heads")
    arch_group.add_argument("--head-dim", type=int, default=128, help="Head dimension (default: 128)")

    # Model preset
    parser.add_argument(
        "--preset",
        type=str,
        choices=list(MODEL_PRESETS.keys()),
        help="Use a well-known model preset instead of manual config",
    )

    # KV cache settings
    cache_group = parser.add_argument_group("KV Cache Settings")
    cache_group.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp16", "bf16", "fp8", "int8"],
        help="KV cache data type (default: fp16)",
    )
    cache_group.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="Tokens per PagedAttention block (default: 16)",
    )

    # GPU and memory
    gpu_group = parser.add_argument_group("GPU Memory")
    gpu_group.add_argument(
        "--gpu-memory-gb",
        type=float,
        default=80.0,
        help="Total GPU memory in GB (default: 80, i.e., A100-80GB)",
    )
    gpu_group.add_argument(
        "--model-size-gb",
        type=float,
        default=0.0,
        help="Model weights memory in GB. If using a preset and not specified, "
             "uses the preset's default.",
    )
    gpu_group.add_argument(
        "--activation-memory-gb",
        type=float,
        default=0.0,
        help="Activation memory estimate in GB (0 = auto-estimate)",
    )

    # Workload
    workload_group = parser.add_argument_group("Workload Parameters")
    workload_group.add_argument(
        "--seq-length",
        type=int,
        default=4096,
        help="Target sequence length for capacity analysis (default: 4096)",
    )
    workload_group.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Target batch size for total memory calculation (default: 32)",
    )

    return parser


def main() -> None:
    """CLI entry point for the memory calculator."""
    parser = build_parser()
    args = parser.parse_args()

    dtype = KVDtype.from_string(args.dtype)

    if args.preset:
        calc = MemoryCalculator.from_preset(
            preset_name=args.preset,
            dtype=dtype,
            block_size=args.block_size,
        )
        # Use preset's default model size if not explicitly given
        if args.model_size_gb == 0.0:
            args.model_size_gb = MODEL_PRESETS[args.preset].weight_memory_gb
    else:
        if args.num_layers is None or args.num_kv_heads is None:
            parser.error(
                "Either --preset or both --num-layers and --num-kv-heads "
                "must be specified."
            )
        calc = MemoryCalculator(
            num_layers=args.num_layers,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            dtype=dtype,
            block_size=args.block_size,
        )
        if args.model_size_gb == 0.0:
            parser.error(
                "--model-size-gb is required when not using a preset."
            )

    report = calc.calculate(
        gpu_memory_gb=args.gpu_memory_gb,
        model_size_gb=args.model_size_gb,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        activation_memory_gb=args.activation_memory_gb,
    )
    report.print_report()


if __name__ == "__main__":
    main()
