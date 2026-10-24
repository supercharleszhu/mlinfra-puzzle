from __future__ import annotations

import ast
import itertools
import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
KERNEL = {
    "display_name": "Python 3 (PyTorch)",
    "language": "python",
    "name": "python3",
}
CELL_IDS = itertools.count()


def markdown(text: str) -> dict[str, object]:
    return {
        "id": f"cell-{next(CELL_IDS):04d}",
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(text).strip().splitlines(keepends=True),
    }


def code(text: str) -> dict[str, object]:
    return {
        "id": f"cell-{next(CELL_IDS):04d}",
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(text).strip().splitlines(keepends=True),
    }


def solution_node_source(name: str) -> str:
    path = ROOT / "chapter01_cs336_transformer_e2e" / "solution.py"
    source_text = path.read_text(encoding="utf-8")
    module = ast.parse(source_text)
    for node in module.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name:
            return ast.get_source_segment(source_text, node) or ""
        if isinstance(node, ast.Assign):
            if any(
                isinstance(target, ast.Name) and target.id == name
                for target in node.targets
            ):
                return ast.get_source_segment(source_text, node) or ""
    raise KeyError(f"{name} not found in {path}")


def chapter01() -> list[dict[str, object]]:
    return [
        markdown(
            """
            # Chapter 01: CS336 Transformer LM end to end

            This chapter builds a small language-model stack from its byte
            representation through one optimizer update. It follows the public
            Stanford CS336 Assignment 1 interfaces while using an independently
            written implementation and compact, synthetic checks.

            The implementation has three connected stages:

            1. A byte-level BPE trainer starts with all 256 byte values,
               repeatedly merges the most frequent adjacent pair, and records
               merge rank. The tokenizer applies those ranks within
               regex-based pre-tokenization boundaries and preserves explicit
               special tokens.
            2. A decoder-only Transformer combines learned token embeddings,
               pre-normalized causal attention, rotary positions, SwiGLU
               feed-forward layers, residual connections, and an output
               projection. Every tensor remains small enough for CPU execution.
            3. Stable cross entropy feeds gradients to a custom AdamW update
               with first and second moments, bias correction, and decoupled
               weight decay.

            The final cell verifies Unicode tokenization, model shapes,
            forward and backward propagation, and a real parameter update.
            """
        ),
        code(
            """
            from __future__ import annotations

            import math
            from collections import Counter
            from collections.abc import Iterable, Sequence

            import regex
            import torch
            from torch import Tensor, nn
            """
        ),
        markdown(
            """
            ## 1. Byte-level BPE

            UTF-8 characters may span several bytes, so the base vocabulary is
            bytes rather than Unicode code points. Pre-tokenization only
            determines where merges may occur; it must not decode partial byte
            pieces. Training counts adjacent pairs weighted by pre-token
            frequency, chooses a deterministic winner, merges all occurrences,
            and appends the merged byte string to the vocabulary.

            Encoding repeats the learned merges in rank order. Decoding is the
            inverse byte concatenation followed by one UTF-8 decode using the
            replacement character for malformed external token sequences.
            Special tokens bypass ordinary pre-tokenization and map to one
            reserved ID.
            """
        ),
        code(
            "\n\n".join(
                solution_node_source(name)
                for name in (
                    "GPT2_SPLIT_PATTERN",
                    "_merge_pair",
                    "train_bpe",
                    "Tokenizer",
                )
            )
        ),
        markdown(
            """
            ## 2. Transformer language model

            `Linear` stores weights in `[output, input]` order and computes
            `x @ W.T`. Token IDs index an embedding table. RMSNorm performs its
            reduction in FP32 before restoring the input dtype, while SwiGLU
            multiplies a SiLU gate by a second learned projection.

            RoPE rotates adjacent feature pairs by position-dependent angles.
            Query and key tensors have shape `[batch, heads, sequence,
            head_dim]`; broadcasting the cached sine and cosine tables across
            batch and head dimensions applies the same positional basis to
            every head.

            Causal attention masks positions above the main diagonal before
            softmax. Each pre-normalized block adds attention and feed-forward
            residuals. The complete model validates its context length and
            returns logits with shape `[batch, sequence, vocabulary]`.
            """
        ),
        code(
            "\n\n".join(
                solution_node_source(name)
                for name in (
                    "Linear",
                    "Embedding",
                    "RMSNorm",
                    "SwiGLU",
                    "RotaryPositionalEmbedding",
                    "CausalMultiHeadSelfAttention",
                    "TransformerBlock",
                    "TransformerLM",
                )
            )
        ),
        markdown(
            """
            ## 3. Loss and AdamW

            Stable cross entropy subtracts the maximum logit before
            exponentiation. The selected target logit is then subtracted from
            the log normalizer and averaged over all token positions.

            AdamW tracks exponential first and second moments for every
            parameter. Bias correction compensates for zero initialization of
            those moments. Weight decay is applied directly to the parameter,
            separately from the gradient moments, which distinguishes AdamW
            from adding an L2 penalty to Adam's gradient.
            """
        ),
        code(
            "\n\n".join(
                solution_node_source(name)
                for name in ("cross_entropy_loss", "AdamW")
            )
        ),
        markdown(
            """
            ## 4. Local end-to-end verification

            The check trains a tiny BPE vocabulary containing multibyte Unicode
            and a special token, then requires an exact encode/decode
            round-trip. A two-layer CPU Transformer processes random token IDs,
            produces finite cross entropy, completes backpropagation, and takes
            one custom AdamW step. A second check compares the custom optimizer
            against `torch.optim.AdamW` for the same parameter and gradient.
            """
        ),
        code(solution_node_source("run_checks") + "\n\nrun_checks()"),
        markdown(
            """
            ## Concept checks

            - A null character is a real Unicode code point even when its
              printed representation appears empty. UTF-8 must be decoded from
              a complete byte stream because individual bytes of a multibyte
              character are not valid standalone text.
            - Transformer parameter memory is not the full training footprint.
              Gradients, AdamW's two moment tensors, activations, temporary
              attention tensors, and allocator overhead must also be included.
            - Feed-forward projections commonly dominate per-token arithmetic
              at fixed sequence length, while attention's quadratic sequence
              term becomes increasingly important as context grows.
            - End-to-end correctness requires checking interfaces between
              components: tokenizer IDs must fit the embedding vocabulary,
              head dimensions must divide model width, RoPE dimensions must be
              even, targets must align with logits, and optimizer state must
              follow each parameter's device and dtype.
            """
        ),
    ]


def main() -> None:
    notebook = {
        "cells": chapter01(),
        "metadata": {
            "kernelspec": KERNEL,
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    destination = (
        ROOT
        / "chapter01_cs336_transformer_e2e"
        / "chapter01.ipynb"
    )
    destination.write_text(
        json.dumps(notebook, ensure_ascii=True, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"WROTE {destination.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
