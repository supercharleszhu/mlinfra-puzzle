from __future__ import annotations

from collections.abc import Mapping, Sequence

from torch import Tensor
from torch.utils.data import DataLoader, Dataset


def reconstruct_text(token_ids: Sequence[int], vocab: Mapping[int, bytes]) -> str:
    """Join byte pieces and decode only after the complete byte stream exists."""
    # LLM_TODO_CH02_01_BYTES
    raise NotImplementedError("reconstruct the byte stream before UTF-8 decoding")


class SlidingWindowDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, token_ids: Sequence[int], max_length: int, stride: int) -> None:
        # LLM_TODO_CH02_02_WINDOWS
        raise NotImplementedError("construct shifted input/target windows")

    def __len__(self) -> int:
        raise NotImplementedError("return the number of complete windows")

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        raise NotImplementedError("return one input window and its shifted target")


def make_loader(
    token_ids: Sequence[int], max_length: int, stride: int, batch_size: int
) -> DataLoader[tuple[Tensor, Tensor]]:
    raise NotImplementedError("wrap SlidingWindowDataset in a deterministic loader")
