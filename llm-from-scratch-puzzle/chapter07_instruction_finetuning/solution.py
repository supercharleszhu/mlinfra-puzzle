from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import torch
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
    instruction = record.get("instruction", "").strip()
    details = record.get("input", "").strip() or "(no extra details)"
    response = record.get("output", "").strip()
    if not instruction:
        raise ValueError("record needs a nonempty instruction")
    if style == "alpaca_like":
        return (
            f"### Assignment\n{instruction}\n\n"
            f"### Available details\n{details}\n\n"
            f"### Expected reply\n{response}"
        )
    if style == "phi_like":
        return (
            f"<|user|>\nComplete this assignment: {instruction}\n"
            f"Useful details: {details}\n<|assistant|>\n{response}"
        )
    raise ValueError("style must be 'alpaca_like' or 'phi_like'")


def mask_prompt_labels(
    input_ids: Tensor, prompt_length: int, ignore_index: int = -100
) -> Tensor:
    if input_ids.ndim != 1 or not 0 <= prompt_length <= input_ids.numel():
        raise ValueError("expected 1-D ids and a prefix length within the row")
    labels = input_ids.clone()
    labels[:prompt_length] = ignore_index
    return labels


def build_memory_plan(
    num_examples: int,
    average_tokens: int,
    max_sequence_length: int,
    hidden_dim: int,
    activation_budget_mib: float,
) -> MemoryPlan:
    if min(num_examples, average_tokens, max_sequence_length, hidden_dim) <= 0:
        raise ValueError("dataset and shape values must be positive")
    if activation_budget_mib <= 0:
        raise ValueError("activation budget must be positive")
    sequence_length = min(average_tokens, max_sequence_length)
    # A declared toy heuristic: fp32 values times eight retained activation groups.
    per_example_bytes = sequence_length * hidden_dim * 4 * 8
    budget_bytes = activation_budget_mib * 2**20
    if per_example_bytes > budget_bytes:
        raise ValueError("budget is below the one-example activation estimate")
    micro_batch = min(num_examples, max(1, int(budget_bytes // per_example_bytes)))
    accumulation = math.ceil(min(32, num_examples) / micro_batch)
    estimated_mib = micro_batch * per_example_bytes / 2**20
    return MemoryPlan(
        num_examples=num_examples,
        sequence_length=sequence_length,
        micro_batch_size=micro_batch,
        gradient_accumulation_steps=accumulation,
        tokens_per_update=micro_batch * accumulation * sequence_length,
        estimated_activation_mib=estimated_mib,
    )


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float = 1.0) -> None:
        super().__init__()
        if rank <= 0 or rank > min(base.in_features, base.out_features):
            raise ValueError("rank must fit both linear dimensions")
        self.base = base
        self.rank = rank
        self.scale = alpha / rank
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, inputs: Tensor) -> Tensor:
        update = (inputs @ self.lora_a.transpose(0, 1)) @ self.lora_b.transpose(0, 1)
        return self.base(inputs) + update * self.scale


def main() -> None:
    record = {
        "instruction": "Sort the symbols by code point.",
        "input": "z, a, m",
        "output": "a, m, z",
    }
    alpaca_like = format_instruction(record, "alpaca_like")
    phi_like = format_instruction(record, "phi_like")
    assert "### Assignment" in alpaca_like
    assert "<|user|>" in phi_like and alpaca_like != phi_like

    labels = mask_prompt_labels(torch.tensor([4, 5, 6, 7, 8]), prompt_length=3)
    assert labels.tolist() == [-100, -100, -100, 7, 8]

    plan = build_memory_plan(10_000, 128, 256, 256, activation_budget_mib=8)
    assert plan.sequence_length == 128
    assert plan.estimated_activation_mib <= 8
    assert plan.gradient_accumulation_steps >= 1

    torch.manual_seed(9)
    base = nn.Linear(16, 12)
    inputs = torch.randn(3, 16)
    expected = base(inputs).detach()
    fully_trainable = sum(parameter.numel() for parameter in base.parameters())
    lora = LoRALinear(base, rank=2, alpha=4)
    lora_trainable = sum(
        parameter.numel() for parameter in lora.parameters() if parameter.requires_grad
    )
    assert torch.allclose(lora(inputs), expected)
    assert lora_trainable == 2 * (16 + 12) < fully_trainable
    print("chapter07: formatting, masking, planning, and LoRA checks passed")


if __name__ == "__main__":
    main()
