from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from torch import Tensor, nn


def pad_and_mask(
    sequences: Sequence[Sequence[int]], pad_id: int, max_length: int | None = None
) -> tuple[Tensor, Tensor]:
    # LLM_TODO_CH06_01_PADDING
    raise NotImplementedError("right-pad rows and return a Boolean validity mask")


def configure_trainable(
    model: nn.Module, mode: Literal["head", "full"], head_name: str = "classifier"
) -> int:
    # LLM_TODO_CH06_02_UNFREEZE
    raise NotImplementedError("set requires_grad and return the trainable count")


def select_sequence_representation(
    hidden_states: Tensor,
    attention_mask: Tensor,
    position: Literal["first", "last_valid"],
) -> Tensor:
    # LLM_TODO_CH06_03_REPRESENTATION
    raise NotImplementedError("select position zero or gather each final valid row")
