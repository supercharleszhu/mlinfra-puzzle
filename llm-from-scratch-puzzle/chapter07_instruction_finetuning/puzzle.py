from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from torch import Tensor, nn


@dataclass(frozen=True)
class MemoryPlan:
    num_examples: int
    sequence_length: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    tokens_per_update: int
    estimated_activation_mib: float


def format_instruction(
    record: Mapping[str, str], style: Literal["alpaca_like", "phi_like"]
) -> str:
    # LLM_TODO_CH07_01_FORMAT
    raise NotImplementedError("render two original prompt layouts")


def mask_prompt_labels(
    input_ids: Tensor, prompt_length: int, ignore_index: int = -100
) -> Tensor:
    # LLM_TODO_CH07_02_MASK
    raise NotImplementedError("clone labels and mask the prompt prefix")


def build_memory_plan(
    num_examples: int,
    average_tokens: int,
    max_sequence_length: int,
    hidden_dim: int,
    activation_budget_mib: float,
) -> MemoryPlan:
    # LLM_TODO_CH07_03_PLAN
    raise NotImplementedError("derive a bounded microbatch and accumulation plan")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float = 1.0) -> None:
        super().__init__()
        # LLM_TODO_CH07_04_LORA
        raise NotImplementedError("freeze base and create trainable low-rank factors")

    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError("add the scaled low-rank update")
