from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from torch import Tensor, nn
from torch.optim import Optimizer


@dataclass(frozen=True)
class DecodingProfile:
    temperature: float
    top_k: int


@dataclass(frozen=True)
class GPTConfig:
    name: str
    embedding_dim: int
    num_layers: int
    num_heads: int
    nominal_parameters: int


def sample_frequencies(
    logits: Tensor, temperature: float, draws: int, seed: int
) -> Tensor:
    # LLM_TODO_CH05_01_TEMPERATURE
    raise NotImplementedError("sample seeded token counts from scaled logits")


def choose_decoding_profile(task: str) -> DecodingProfile:
    # LLM_TODO_CH05_02_PROFILES
    raise NotImplementedError("choose deterministic or creative settings")


def generate_deterministic(
    next_logits: Callable[[Tensor], Tensor],
    prompt: list[int],
    max_new_tokens: int,
    top_k: int = 1,
) -> list[int]:
    # LLM_TODO_CH05_03_DETERMINISTIC
    raise NotImplementedError("append the sole top-k candidate at each step")


def make_checkpoint(
    model: nn.Module, optimizer: Optimizer, step: int
) -> dict[str, Any]:
    # LLM_TODO_CH05_04_CHECKPOINT
    raise NotImplementedError("capture model, optimizer, and progress state")


def restore_checkpoint(
    checkpoint: dict[str, Any], model: nn.Module, optimizer: Optimizer
) -> int:
    raise NotImplementedError("restore both state dictionaries and return step")


def split_cross_entropy(
    train_logits: Tensor,
    train_targets: Tensor,
    validation_logits: Tensor,
    validation_targets: Tensor,
) -> tuple[float, float]:
    # LLM_TODO_CH05_05_CROSS_ENTROPY
    raise NotImplementedError("evaluate the two explicitly supplied toy splits")


def select_gpt2_config(name: str) -> GPTConfig:
    # LLM_TODO_CH05_06_CONFIG
    raise NotImplementedError("return metadata only; do not allocate a model")
