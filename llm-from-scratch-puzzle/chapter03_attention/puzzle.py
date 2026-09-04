from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def simple_self_attention(inputs: Tensor) -> tuple[Tensor, Tensor]:
    """Return context vectors and attention weights with inputs as Q, K, and V."""
    # ATTENTION_TODO_CH03_00_SIMPLE
    raise NotImplementedError("compute scaled scores, weights, and context")


class CausalSelfAttention(nn.Module):
    """One causal attention head with learned Q, K, and V projections."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        context_length: int,
        dropout: float = 0.0,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        # ATTENTION_TODO_CH03_00_CAUSAL
        raise NotImplementedError(
            "create Q/K/V projections, dropout, and a causal-mask buffer"
        )

    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError(
            "project Q/K/V, mask future scores, normalize, and mix values"
        )


class MultiHeadCausalAttention(nn.Module):
    """Parallel causal heads implemented with shared projection matrices."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        context_length: int,
        num_heads: int,
        dropout: float = 0.0,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        # ATTENTION_TODO_CH03_00_MULTIHEAD
        raise NotImplementedError(
            "validate head width and create Q/K/V/output projections"
        )

    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError(
            "split heads, apply causal attention, and merge the head axis"
        )


class RawProjection(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs @ self.weight


def transfer_raw_to_linear(raw_weight: Tensor, linear: nn.Linear) -> None:
    # LLM_TODO_CH03_01_TRANSFER
    raise NotImplementedError("copy the raw matrix using nn.Linear's layout")


class MultiHeadWrapper(nn.Module):
    def __init__(self, heads: list[nn.Module]) -> None:
        super().__init__()
        self.heads = nn.ModuleList(heads)

    def forward(self, inputs: Tensor) -> Tensor:
        # LLM_TODO_CH03_02_MULTIHEAD
        raise NotImplementedError("concatenate every head on the feature axis")


def validate_attention_dimensions(embedding_dim: int, num_heads: int) -> int:
    # LLM_TODO_CH03_03_DIMENSIONS
    raise NotImplementedError("validate divisibility and return per-head width")
