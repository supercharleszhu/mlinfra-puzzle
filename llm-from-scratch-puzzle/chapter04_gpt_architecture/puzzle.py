from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int
    context_length: int
    embedding_dim: int
    num_layers: int
    num_heads: int


def projection_parameter_counts(
    embedding_dim: int, expansion: int = 4
) -> dict[str, int]:
    # LLM_TODO_CH04_01_COUNTS
    raise NotImplementedError("count weights and biases for attention and FFN")


def estimate_gpt2(
    config: GPTConfig, bytes_per_parameter: int = 4
) -> tuple[int, int]:
    """Return (parameter count, weights-only bytes), assuming tied output weights."""
    # LLM_TODO_CH04_02_ESTIMATE
    raise NotImplementedError("sum embeddings, transformer blocks, and final norm")
