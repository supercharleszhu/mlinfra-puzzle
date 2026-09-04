from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def simple_self_attention(inputs: Tensor) -> tuple[Tensor, Tensor]:
    if inputs.ndim != 3:
        raise ValueError("inputs must have shape (batch, tokens, channels)")
    scores = inputs @ inputs.transpose(-2, -1)
    weights = torch.softmax(scores / math.sqrt(inputs.shape[-1]), dim=-1)
    return weights @ inputs, weights


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        context_length: int,
        dropout: float = 0.0,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        if min(input_dim, output_dim, context_length) <= 0:
            raise ValueError("dimensions and context length must be positive")
        self.query = nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.key = nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.value = nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(context_length, context_length, dtype=torch.bool),
                diagonal=1,
            ),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        token_count = inputs.shape[1]
        if token_count > self.causal_mask.shape[0]:
            raise ValueError("input exceeds configured context length")
        queries = self.query(inputs)
        keys = self.key(inputs)
        values = self.value(inputs)
        scores = queries @ keys.transpose(-2, -1)
        scores = scores.masked_fill(
            self.causal_mask[:token_count, :token_count], -torch.inf
        )
        weights = torch.softmax(scores / math.sqrt(keys.shape[-1]), dim=-1)
        return self.dropout(weights) @ values


class MultiHeadCausalAttention(nn.Module):
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
        if min(input_dim, output_dim, context_length, num_heads) <= 0:
            raise ValueError("dimensions, context length, and heads must be positive")
        if output_dim % num_heads:
            raise ValueError("output_dim must be divisible by num_heads")
        self.head_dim = output_dim // num_heads
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.query = nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.key = nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.value = nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.output = nn.Linear(output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(context_length, context_length, dtype=torch.bool),
                diagonal=1,
            ),
        )

    def _split_heads(self, tensor: Tensor) -> Tensor:
        batch, tokens, _ = tensor.shape
        return tensor.view(
            batch, tokens, self.num_heads, self.head_dim
        ).transpose(1, 2)

    def forward(self, inputs: Tensor) -> Tensor:
        batch, token_count, _ = inputs.shape
        if token_count > self.causal_mask.shape[0]:
            raise ValueError("input exceeds configured context length")
        queries = self._split_heads(self.query(inputs))
        keys = self._split_heads(self.key(inputs))
        values = self._split_heads(self.value(inputs))
        scores = queries @ keys.transpose(-2, -1)
        scores = scores.masked_fill(
            self.causal_mask[:token_count, :token_count], -torch.inf
        )
        weights = torch.softmax(scores / math.sqrt(self.head_dim), dim=-1)
        context = self.dropout(weights) @ values
        merged = context.transpose(1, 2).contiguous().view(
            batch, token_count, self.output_dim
        )
        return self.output(merged)


class RawProjection(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs @ self.weight


def transfer_raw_to_linear(raw_weight: Tensor, linear: nn.Linear) -> None:
    expected = (linear.in_features, linear.out_features)
    if tuple(raw_weight.shape) != expected:
        raise ValueError(f"expected raw shape {expected}, got {tuple(raw_weight.shape)}")
    with torch.no_grad():
        linear.weight.copy_(raw_weight.transpose(0, 1))
        if linear.bias is not None:
            linear.bias.zero_()


class MultiHeadWrapper(nn.Module):
    def __init__(self, heads: list[nn.Module]) -> None:
        super().__init__()
        if not heads:
            raise ValueError("at least one head is required")
        self.heads = nn.ModuleList(heads)

    def forward(self, inputs: Tensor) -> Tensor:
        return torch.cat([head(inputs) for head in self.heads], dim=-1)


def validate_attention_dimensions(embedding_dim: int, num_heads: int) -> int:
    if embedding_dim <= 0 or num_heads <= 0:
        raise ValueError("dimensions must be positive")
    if embedding_dim % num_heads:
        raise ValueError("embedding_dim must be divisible by num_heads")
    return embedding_dim // num_heads


def main() -> None:
    torch.manual_seed(7)
    inputs = torch.randn(2, 5, 4)

    simple_context, simple_weights = simple_self_attention(inputs)
    assert simple_context.shape == inputs.shape
    assert torch.allclose(
        simple_weights.sum(dim=-1),
        torch.ones_like(simple_weights.sum(dim=-1)),
    )

    causal = CausalSelfAttention(4, 4, context_length=5)
    changed_future = inputs.clone()
    changed_future[:, 3:] += 100
    assert torch.allclose(
        causal(inputs)[:, :3],
        causal(changed_future)[:, :3],
        atol=1e-5,
        rtol=1e-5,
    )

    multihead = MultiHeadCausalAttention(4, 8, context_length=5, num_heads=2)
    assert multihead(inputs).shape == (2, 5, 8)
    assert torch.allclose(
        multihead(inputs)[:, :3],
        multihead(changed_future)[:, :3],
        atol=1e-5,
        rtol=1e-5,
    )

    raw = RawProjection(4, 3)
    nn.init.normal_(raw.weight)
    linear = nn.Linear(4, 3)
    transfer_raw_to_linear(raw.weight, linear)
    assert torch.allclose(raw(inputs), linear(inputs))

    wrapper = MultiHeadWrapper([nn.Linear(4, 2), nn.Linear(4, 2)])
    assert wrapper(inputs).shape == (2, 5, 4)
    assert validate_attention_dimensions(768, 12) == 64
    try:
        validate_attention_dimensions(10, 3)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid head split was accepted")
    print(
        "chapter03: attention implementations, projection transfer, "
        "multi-head wrapper, and dimension checks passed"
    )


if __name__ == "__main__":
    main()
