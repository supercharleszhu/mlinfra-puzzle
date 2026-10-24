from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence

import regex
import torch
from torch import Tensor, nn


GPT2_SPLIT_PATTERN = regex.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+|"""
    r""" ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def _merge_pair(
    pieces: tuple[bytes, ...], pair: tuple[bytes, bytes]
) -> tuple[bytes, ...]:
    merged: list[bytes] = []
    index = 0
    while index < len(pieces):
        if (
            index + 1 < len(pieces)
            and pieces[index] == pair[0]
            and pieces[index + 1] == pair[1]
        ):
            merged.append(pair[0] + pair[1])
            index += 2
        else:
            merged.append(pieces[index])
            index += 1
    return tuple(merged)


def train_bpe(
    text: str,
    vocab_size: int,
    special_tokens: Sequence[str] = (),
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a deterministic byte-level BPE vocabulary on a small corpus."""

    if vocab_size < 256 + len(special_tokens):
        raise ValueError(
            "vocab_size must fit 256 byte tokens and every special token"
        )
    if len(set(special_tokens)) != len(special_tokens):
        raise ValueError("special tokens must be unique")

    vocab = {index: bytes([index]) for index in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    special_pattern = (
        regex.compile(
            "|".join(
                regex.escape(token)
                for token in sorted(special_tokens, key=len, reverse=True)
            )
        )
        if special_tokens
        else None
    )
    ordinary_parts = (
        special_pattern.split(text) if special_pattern is not None else [text]
    )
    words: Counter[tuple[bytes, ...]] = Counter()
    for part in ordinary_parts:
        if not part or part in special_tokens:
            continue
        for match in GPT2_SPLIT_PATTERN.finditer(part):
            encoded = match.group().encode("utf-8")
            words[tuple(bytes([value]) for value in encoded)] += 1

    merges: list[tuple[bytes, bytes]] = []
    while len(vocab) < vocab_size:
        pair_counts: Counter[tuple[bytes, bytes]] = Counter()
        for pieces, count in words.items():
            for pair in zip(pieces, pieces[1:]):
                pair_counts[pair] += count
        if not pair_counts:
            break

        best_pair = max(
            pair_counts,
            key=lambda pair: (pair_counts[pair], pair),
        )
        merged_piece = best_pair[0] + best_pair[1]
        if merged_piece in vocab.values():
            break
        vocab[len(vocab)] = merged_piece
        merges.append(best_pair)

        updated: Counter[tuple[bytes, ...]] = Counter()
        for pieces, count in words.items():
            updated[_merge_pair(pieces, best_pair)] += count
        words = updated

    return vocab, merges


class Tokenizer:
    """Byte-level BPE tokenizer with explicit special-token handling."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: Sequence[tuple[bytes, bytes]],
        special_tokens: Sequence[str] = (),
    ) -> None:
        if len(set(vocab.values())) != len(vocab):
            raise ValueError("vocabulary byte strings must be unique")
        self.vocab = dict(vocab)
        self.token_to_id = {
            token_bytes: token_id
            for token_id, token_bytes in self.vocab.items()
        }
        self.merge_ranks = {
            pair: rank for rank, pair in enumerate(merges)
        }
        self.special_token_ids = {}
        for token in special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in self.token_to_id:
                raise ValueError(
                    f"special token {token!r} is missing from the vocabulary"
                )
            self.special_token_ids[token] = self.token_to_id[token_bytes]
        self.special_pattern = (
            regex.compile(
                "("
                + "|".join(
                    regex.escape(token)
                    for token in sorted(
                        self.special_token_ids, key=len, reverse=True
                    )
                )
                + ")"
            )
            if self.special_token_ids
            else None
        )

    def _encode_piece(self, piece: str) -> list[int]:
        tokens = tuple(
            bytes([value]) for value in piece.encode("utf-8")
        )
        while len(tokens) > 1:
            ranked_pairs = [
                (self.merge_ranks[pair], pair)
                for pair in zip(tokens, tokens[1:])
                if pair in self.merge_ranks
            ]
            if not ranked_pairs:
                break
            _, best_pair = min(ranked_pairs)
            tokens = _merge_pair(tokens, best_pair)
        try:
            return [self.token_to_id[token] for token in tokens]
        except KeyError as error:
            raise ValueError(
                f"token bytes {error.args[0]!r} are missing from the vocabulary"
            ) from error

    def encode(self, text: str) -> list[int]:
        parts = (
            self.special_pattern.split(text)
            if self.special_pattern is not None
            else [text]
        )
        token_ids: list[int] = []
        for part in parts:
            if not part:
                continue
            if part in self.special_token_ids:
                token_ids.append(self.special_token_ids[part])
                continue
            for match in GPT2_SPLIT_PATTERN.finditer(part):
                token_ids.extend(self._encode_piece(match.group()))
        return token_ids

    def encode_iterable(self, texts: Iterable[str]) -> Iterable[int]:
        for text in texts:
            yield from self.encode(text)

    def decode(self, ids: Sequence[int]) -> str:
        try:
            encoded = b"".join(self.vocab[token_id] for token_id in ids)
        except KeyError as error:
            raise ValueError(f"unknown token ID {error.args[0]}") from error
        return encoded.decode("utf-8", errors="replace")


class Linear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device=None, dtype=None
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype,
            )
        )
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, std=std)

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs @ self.weight.transpose(-1, -2)


class Embedding(nn.Module):
    def __init__(
        self, num_embeddings: int, embedding_dim: int, device=None, dtype=None
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(
                num_embeddings,
                embedding_dim,
                device=device,
                dtype=dtype,
            )
        )
        nn.init.trunc_normal_(self.weight)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(
        self, d_model: int, eps: float = 1e-5, device=None, dtype=None
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        input_dtype = inputs.dtype
        values = inputs.float()
        rms = torch.sqrt(
            values.square().mean(dim=-1, keepdim=True) + self.eps
        )
        normalized = values / rms
        return (normalized * self.weight.float()).to(input_dtype)


class SwiGLU(nn.Module):
    def __init__(
        self, d_model: int, d_ff: int, device=None, dtype=None
    ) -> None:
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, inputs: Tensor) -> Tensor:
        gate = torch.nn.functional.silu(self.w1(inputs))
        return self.w2(gate * self.w3(inputs))


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device=None,
    ) -> None:
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("RoPE head dimension must be even")
        frequencies = 1.0 / (
            theta
            ** (
                torch.arange(0, d_k, 2, device=device, dtype=torch.float32)
                / d_k
            )
        )
        positions = torch.arange(
            max_seq_len, device=device, dtype=torch.float32
        )
        angles = torch.outer(positions, frequencies)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(
        self, inputs: Tensor, token_positions: Tensor
    ) -> Tensor:
        cos = self.cos[token_positions].to(inputs.dtype)
        sin = self.sin[token_positions].to(inputs.dtype)
        while cos.ndim < inputs.ndim:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        even = inputs[..., 0::2]
        odd = inputs[..., 1::2]
        rotated = torch.stack(
            (even * cos - odd * sin, even * sin + odd * cos),
            dim=-1,
        )
        return rotated.flatten(start_dim=-2)


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        context_length: int,
        theta: float = 10_000.0,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(
            d_model, d_model, device=device, dtype=dtype
        )
        self.rope = RotaryPositionalEmbedding(
            theta, self.head_dim, context_length, device=device
        )

    def _split_heads(self, values: Tensor) -> Tensor:
        batch, sequence, _ = values.shape
        return values.reshape(
            batch, sequence, self.num_heads, self.head_dim
        ).transpose(1, 2)

    def forward(self, inputs: Tensor) -> Tensor:
        batch, sequence, width = inputs.shape
        query = self._split_heads(self.q_proj(inputs))
        key = self._split_heads(self.k_proj(inputs))
        value = self._split_heads(self.v_proj(inputs))
        positions = torch.arange(sequence, device=inputs.device)
        query = self.rope(query, positions)
        key = self.rope(key, positions)

        scores = query @ key.transpose(-1, -2)
        scores = scores / math.sqrt(self.head_dim)
        causal_mask = torch.ones(
            sequence,
            sequence,
            dtype=torch.bool,
            device=inputs.device,
        ).triu(diagonal=1)
        scores = scores.masked_fill(causal_mask, -torch.inf)
        probabilities = torch.softmax(scores.float(), dim=-1).to(
            query.dtype
        )
        context = probabilities @ value
        context = context.transpose(1, 2).reshape(batch, sequence, width)
        return self.output_proj(context)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        context_length: int,
        theta: float = 10_000.0,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(
            d_model, device=device, dtype=dtype
        )
        self.attention = CausalMultiHeadSelfAttention(
            d_model,
            num_heads,
            context_length,
            theta=theta,
            device=device,
            dtype=dtype,
        )
        self.ffn_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = inputs + self.attention(self.attention_norm(inputs))
        return hidden + self.ffn(self.ffn_norm(hidden))


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float = 10_000.0,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.token_embeddings = Embedding(
            vocab_size, d_model, device=device, dtype=dtype
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    context_length,
                    theta=theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.output = Linear(
            d_model, vocab_size, device=device, dtype=dtype
        )

    def forward(self, token_ids: Tensor) -> Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token IDs must have shape [batch, sequence]")
        if token_ids.shape[1] > self.context_length:
            raise ValueError("input exceeds the configured context length")
        hidden = self.token_embeddings(token_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.output(self.final_norm(hidden))


def cross_entropy_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Mean cross entropy using an explicit stable log-sum-exp."""

    if logits.shape[:-1] != targets.shape:
        raise ValueError("targets must match every non-vocabulary dimension")
    logits_float = logits.float()
    max_logits = logits_float.max(dim=-1, keepdim=True).values
    log_normalizer = max_logits.squeeze(-1) + torch.log(
        torch.exp(logits_float - max_logits).sum(dim=-1)
    )
    target_logits = logits_float.gather(
        dim=-1, index=targets.unsqueeze(-1)
    ).squeeze(-1)
    return (log_normalizer - target_logits).mean()


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ) -> None:
        if lr < 0 or eps < 0 or weight_decay < 0:
            raise ValueError("lr, eps, and weight_decay must be non-negative")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError("AdamW betas must lie in [0, 1)")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["first_moment"] = torch.zeros_like(parameter)
                    state["second_moment"] = torch.zeros_like(parameter)
                state["step"] += 1
                first_moment = state["first_moment"]
                second_moment = state["second_moment"]
                first_moment.mul_(beta1).add_(
                    gradient, alpha=1.0 - beta1
                )
                second_moment.mul_(beta2).addcmul_(
                    gradient, gradient, value=1.0 - beta2
                )

                parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                bias_correction1 = 1.0 - beta1 ** state["step"]
                bias_correction2 = 1.0 - beta2 ** state["step"]
                step_size = group["lr"] / bias_correction1
                denominator = (
                    second_moment.sqrt() / math.sqrt(bias_correction2)
                )
                denominator.add_(group["eps"])
                parameter.addcdiv_(
                    first_moment, denominator, value=-step_size
                )
        return loss


def run_checks() -> None:
    special = "<|endoftext|>"
    text = "low lower lowest 你好"
    vocab, merges = train_bpe(
        text, vocab_size=280, special_tokens=(special,)
    )
    tokenizer = Tokenizer(vocab, merges, special_tokens=(special,))
    sample = f"lower 你好{special}"
    token_ids = tokenizer.encode(sample)
    assert tokenizer.decode(token_ids) == sample

    torch.manual_seed(336)
    model = TransformerLM(
        vocab_size=64,
        context_length=16,
        num_layers=2,
        d_model=32,
        num_heads=4,
        d_ff=64,
    )
    inputs = torch.randint(0, 64, (2, 8))
    targets = torch.randint(0, 64, (2, 8))
    logits = model(inputs)
    assert logits.shape == (2, 8, 64)
    loss = cross_entropy_loss(logits, targets)
    assert torch.isfinite(loss)
    loss.backward()

    before = model.output.weight.detach().clone()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    assert not torch.equal(before, model.output.weight)

    custom_parameter = nn.Parameter(torch.tensor([1.0, -2.0]))
    reference_parameter = nn.Parameter(custom_parameter.detach().clone())
    custom_parameter.grad = torch.tensor([0.25, -0.5])
    reference_parameter.grad = custom_parameter.grad.clone()
    custom_optimizer = AdamW(
        [custom_parameter], lr=2e-3, weight_decay=0.1
    )
    reference_optimizer = torch.optim.AdamW(
        [reference_parameter], lr=2e-3, weight_decay=0.1
    )
    custom_optimizer.step()
    reference_optimizer.step()
    torch.testing.assert_close(custom_parameter, reference_parameter)
    print(
        "Chapter 01 CS336 checks passed: "
        f"{len(vocab)} BPE tokens, loss={loss.item():.4f}"
    )


if __name__ == "__main__":
    run_checks()
