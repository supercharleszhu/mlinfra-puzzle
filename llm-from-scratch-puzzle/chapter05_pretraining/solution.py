from __future__ import annotations

import copy
import io
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
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


CONFIGS = {
    "124M": GPTConfig("124M", 768, 12, 12, 124_000_000),
    "1558M": GPTConfig("1558M", 1_600, 48, 25, 1_558_000_000),
}


def sample_frequencies(
    logits: Tensor, temperature: float, draws: int, seed: int
) -> Tensor:
    if logits.ndim != 1 or temperature <= 0 or draws <= 0:
        raise ValueError("use 1-D logits, positive temperature, and positive draws")
    probabilities = torch.softmax(logits / temperature, dim=0)
    generator = torch.Generator(device=logits.device).manual_seed(seed)
    samples = torch.multinomial(
        probabilities, draws, replacement=True, generator=generator
    )
    return torch.bincount(samples, minlength=logits.numel())


def choose_decoding_profile(task: str) -> DecodingProfile:
    profiles = {
        "deterministic": DecodingProfile(temperature=1.0, top_k=1),
        "creative": DecodingProfile(temperature=1.2, top_k=20),
    }
    try:
        return profiles[task]
    except KeyError as error:
        raise ValueError(f"unknown task profile: {task}") from error


def generate_deterministic(
    next_logits: Callable[[Tensor], Tensor],
    prompt: list[int],
    max_new_tokens: int,
    top_k: int = 1,
) -> list[int]:
    if not prompt or max_new_tokens < 0:
        raise ValueError("prompt must be nonempty and token count nonnegative")
    if top_k != 1:
        raise ValueError("this deterministic exercise requires top_k=1")
    output = list(prompt)
    for _ in range(max_new_tokens):
        context = torch.tensor(output, dtype=torch.long).unsqueeze(0)
        logits = next_logits(context)
        if logits.ndim != 2 or logits.shape[0] != 1:
            raise ValueError("next_logits must return shape (1, vocabulary)")
        output.append(int(torch.argmax(logits[0]).item()))
    return output


def make_checkpoint(
    model: nn.Module, optimizer: Optimizer, step: int
) -> dict[str, Any]:
    if step < 0:
        raise ValueError("step must be nonnegative")
    return {
        "model": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "step": step,
    }


def restore_checkpoint(
    checkpoint: dict[str, Any], model: nn.Module, optimizer: Optimizer
) -> int:
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["step"])


def split_cross_entropy(
    train_logits: Tensor,
    train_targets: Tensor,
    validation_logits: Tensor,
    validation_targets: Tensor,
) -> tuple[float, float]:
    train_loss = F.cross_entropy(train_logits, train_targets)
    validation_loss = F.cross_entropy(validation_logits, validation_targets)
    return float(train_loss.item()), float(validation_loss.item())


def select_gpt2_config(name: str) -> GPTConfig:
    try:
        return CONFIGS[name]
    except KeyError as error:
        raise ValueError(f"available configs: {', '.join(CONFIGS)}") from error


def _train_step(model: nn.Module, optimizer: Optimizer) -> None:
    inputs = torch.tensor([[1.0], [2.0]])
    targets = torch.tensor([[2.0], [4.0]])
    optimizer.zero_grad()
    loss = F.mse_loss(model(inputs), targets)
    loss.backward()
    optimizer.step()


def main() -> None:
    logits = torch.tensor([0.0, 1.0, 2.0])
    cold = sample_frequencies(logits, 0.2, 2_000, seed=4)
    warm = sample_frequencies(logits, 2.0, 2_000, seed=4)
    assert cold[-1] > warm[-1]
    assert choose_decoding_profile("deterministic").top_k == 1
    assert choose_decoding_profile("creative").temperature > 1.0

    def toy_next(context: Tensor) -> Tensor:
        target = (int(context[0, -1]) + 1) % 5
        result = torch.zeros(1, 5)
        result[0, target] = 10.0
        return result

    assert generate_deterministic(toy_next, [0], 4) == [0, 1, 2, 3, 4]

    torch.manual_seed(3)
    uninterrupted = nn.Linear(1, 1, bias=False)
    initial = copy.deepcopy(uninterrupted.state_dict())
    uninterrupted_optimizer = torch.optim.SGD(
        uninterrupted.parameters(), lr=0.05, momentum=0.9
    )
    _train_step(uninterrupted, uninterrupted_optimizer)
    _train_step(uninterrupted, uninterrupted_optimizer)

    interrupted = nn.Linear(1, 1, bias=False)
    interrupted.load_state_dict(initial)
    interrupted_optimizer = torch.optim.SGD(
        interrupted.parameters(), lr=0.05, momentum=0.9
    )
    _train_step(interrupted, interrupted_optimizer)
    buffer = io.BytesIO()
    torch.save(make_checkpoint(interrupted, interrupted_optimizer, step=1), buffer)
    buffer.seek(0)

    resumed = nn.Linear(1, 1, bias=False)
    resumed_optimizer = torch.optim.SGD(resumed.parameters(), lr=0.05, momentum=0.9)
    payload = torch.load(buffer, map_location="cpu", weights_only=True)
    assert restore_checkpoint(payload, resumed, resumed_optimizer) == 1
    _train_step(resumed, resumed_optimizer)
    assert torch.allclose(resumed.weight, uninterrupted.weight)

    # These are deliberately assigned toy splits, not inferred corpus members.
    train_loss, validation_loss = split_cross_entropy(
        torch.tensor([[3.0, 0.0], [0.0, 3.0]]),
        torch.tensor([0, 1]),
        torch.tensor([[0.4, 0.6], [0.6, 0.4]]),
        torch.tensor([0, 1]),
    )
    assert train_loss < validation_loss
    assert select_gpt2_config("124M").num_layers == 12
    assert select_gpt2_config("1558M").nominal_parameters == 1_558_000_000
    print("chapter05: six deterministic toy exercises passed")


if __name__ == "__main__":
    main()
