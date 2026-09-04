from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from torch import Tensor, nn


def pad_and_mask(
    sequences: Sequence[Sequence[int]], pad_id: int, max_length: int | None = None
) -> tuple[Tensor, Tensor]:
    if not sequences:
        raise ValueError("at least one sequence is required")
    longest = max(len(sequence) for sequence in sequences)
    width = longest if max_length is None else max_length
    if width < longest or width <= 0:
        raise ValueError("max_length must fit every sequence and be positive")
    tokens = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(sequences), width), dtype=torch.bool)
    for row, sequence in enumerate(sequences):
        length = len(sequence)
        if length:
            tokens[row, :length] = torch.tensor(sequence, dtype=torch.long)
            mask[row, :length] = True
    return tokens, mask


def configure_trainable(
    model: nn.Module, mode: Literal["head", "full"], head_name: str = "classifier"
) -> int:
    if mode not in {"head", "full"}:
        raise ValueError("mode must be 'head' or 'full'")
    found_head = False
    for name, parameter in model.named_parameters():
        is_head = name == head_name or name.startswith(f"{head_name}.")
        found_head = found_head or is_head
        parameter.requires_grad = mode == "full" or is_head
    if mode == "head" and not found_head:
        raise ValueError(f"no parameters found below {head_name!r}")
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def select_sequence_representation(
    hidden_states: Tensor,
    attention_mask: Tensor,
    position: Literal["first", "last_valid"],
) -> Tensor:
    if hidden_states.ndim != 3 or attention_mask.shape != hidden_states.shape[:2]:
        raise ValueError("expected hidden (batch, sequence, width) and matching mask")
    if position == "first":
        return hidden_states[:, 0]
    if position != "last_valid":
        raise ValueError("position must be 'first' or 'last_valid'")
    lengths = attention_mask.to(torch.long).sum(dim=1)
    if torch.any(lengths == 0):
        raise ValueError("every row needs at least one valid token")
    row_ids = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    return hidden_states[row_ids, lengths - 1]


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(3, 3)
        self.classifier = nn.Linear(3, 2)


def main() -> None:
    tokens, mask = pad_and_mask([[2, 4, 6], [8]], pad_id=0, max_length=4)
    assert tokens.tolist() == [[2, 4, 6, 0], [8, 0, 0, 0]]
    assert mask.sum(dim=1).tolist() == [3, 1]

    model = TinyClassifier()
    selective = configure_trainable(model, "head")
    full = configure_trainable(model, "full")
    assert selective == 8 and full > selective

    hidden = torch.arange(2 * 4 * 3).reshape(2, 4, 3)
    first = select_sequence_representation(hidden, mask, "first")
    last = select_sequence_representation(hidden, mask, "last_valid")
    assert torch.equal(first, hidden[:, 0])
    assert torch.equal(last[0], hidden[0, 2])
    assert torch.equal(last[1], hidden[1, 0])
    print("chapter06: padding, unfreezing, and representation checks passed")


if __name__ == "__main__":
    main()
