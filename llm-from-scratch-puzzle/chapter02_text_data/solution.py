from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


def reconstruct_text(token_ids: Sequence[int], vocab: Mapping[int, bytes]) -> str:
    try:
        encoded = b"".join(vocab[token_id] for token_id in token_ids)
    except KeyError as error:
        raise ValueError(f"unknown token id: {error.args[0]}") from error
    return encoded.decode("utf-8")


class SlidingWindowDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, token_ids: Sequence[int], max_length: int, stride: int) -> None:
        if max_length <= 0 or stride <= 0:
            raise ValueError("max_length and stride must be positive")
        values = torch.tensor(token_ids, dtype=torch.long)
        self.windows = [
            (values[start : start + max_length], values[start + 1 : start + max_length + 1])
            for start in range(0, len(values) - max_length, stride)
        ]

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.windows[index]


def make_loader(
    token_ids: Sequence[int], max_length: int, stride: int, batch_size: int
) -> DataLoader[tuple[Tensor, Tensor]]:
    dataset = SlidingWindowDataset(token_ids, max_length, stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)


def main() -> None:
    pieces = {0: b"A", 1: b"\xe4", 2: b"\xbd\xa0", 3: b"!"}
    assert reconstruct_text([0, 1, 2, 3], pieces) == "A你!"

    loader = make_loader(range(12), max_length=4, stride=3, batch_size=2)
    inputs, targets = next(iter(loader))
    assert inputs.shape == targets.shape == (2, 4)
    assert inputs[:, 0].tolist() == [0, 3]
    assert torch.equal(targets[:, :-1], inputs[:, 1:])
    print("chapter02: byte reconstruction and stride checks passed")


if __name__ == "__main__":
    main()
