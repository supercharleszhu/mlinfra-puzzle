from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
KERNEL = {
    "display_name": "Python 3 (PyTorch)",
    "language": "python",
    "name": "python3",
}


def markdown(text: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(text).strip().splitlines(keepends=True),
    }


def code(text: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(text).strip().splitlines(keepends=True),
    }


def exercise(
    number: str,
    title: str,
    objective: str,
    explanation: str,
    walkthrough: str,
    hints: str,
    scaffold: str,
    check: str,
) -> list[dict[str, object]]:
    return [
        markdown(
            f"""
            ## Exercise {number}: {title}

            **Objective.** {objective}

            {explanation}

            ### Data-flow walkthrough

            {walkthrough}

            ### Hints and common mistakes

            {hints}
            """
        ),
        code(scaffold),
        markdown(
            """
            ### Check your implementation

            Run the next cell only after replacing the TODO above. A useful
            check tests behavior, not just whether the function returns.
            """
        ),
        code(check),
    ]


def intro(
    chapter: str,
    title: str,
    source_url: str,
    overview: str,
) -> list[dict[str, object]]:
    return [
        markdown(
            f"""
            # Chapter {chapter}: {title}

            {overview}

            This notebook is an independently written practice companion. It
            does not reproduce the book's prose, figures, data, or solutions.
            Read the [Chinese chapter]({source_url}) for the complete narrative
            and exercise statements, and use the
            [canonical MIT companion code](https://github.com/rasbt/LLMs-from-scratch)
            as an additional reference.

            **Prerequisites:** Python 3.10+ and PyTorch 2.2+. Every example is
            synthetic, deterministic, CPU-friendly, and makes no network call.

            **Workflow**

            1. Read the concept and trace the shapes by hand.
            2. Replace one `LLM_TODO` and its `NotImplementedError`.
            3. Run that exercise's check cell.
            4. Explain the result before moving to the next exercise.
            5. Compare with `solution.py` only after making a serious attempt.
            """
        )
    ]


def finish(chapter: str, questions: str) -> list[dict[str, object]]:
    return [
        markdown(
            f"""
            ## Chapter {chapter} synthesis

            {questions}

            After answering these questions, run the chapter's `solution.py`
            from a terminal to compare behavior. The reference file is compact
            by design; the reasoning in this notebook is the main lesson.
            """
        )
    ]


def chapter02() -> list[dict[str, object]]:
    cells = intro(
        "02",
        "Text data as model-ready tensors",
        "https://github.com/skindhu/Build-A-Large-Language-Model-CN/blob/main/cn-Book/2.%E5%A4%84%E7%90%86%E6%96%87%E6%9C%AC%E6%95%B0%E6%8D%AE.md",
        """
        A language model never receives text directly. A tokenizer maps text to
        integer IDs, an embedding table maps IDs to vectors, and a sampling
        pipeline groups IDs into input/target windows. These exercises isolate
        two easy-to-miss boundaries: bytes are not characters, and next-token
        targets are shifted views of the same token stream.
        """,
    )
    cells.append(
        code(
            """
            from __future__ import annotations

            from collections.abc import Mapping, Sequence

            import torch
            from torch import Tensor
            from torch.utils.data import DataLoader, Dataset
            """
        )
    )
    cells += exercise(
        "2.1",
        "Reconstruct byte-token pieces",
        "Join token byte pieces in ID order and decode the complete stream once.",
        """
        UTF-8 uses a variable number of bytes per character. A tokenizer may
        store a byte sequence in pieces whose boundaries do not align with
        Unicode character boundaries. Decoding each piece independently can
        therefore fail even when their concatenation is valid text.

        Treat tokenization as two mappings: token IDs select byte strings, then
        the concatenated byte string is decoded. If IDs are
        $[t_0,\\ldots,t_n]$ and $V(t)$ returns bytes, the reconstructed stream is
        $B=V(t_0)\\Vert\\cdots\\Vert V(t_n)$. Only then compute
        `B.decode("utf-8")`.
        """,
        """
        In the toy vocabulary, the Chinese character is split across two token
        pieces: `b"\\xe4"` and `b"\\xbd\\xa0"`. Neither is a complete UTF-8
        character alone, but their concatenation is `b"\\xe4\\xbd\\xa0"`.
        Preserve token order, join as bytes, and decode the final sequence.
        """,
        """
        Do not call `.decode()` inside the loop. Surface an unknown token ID as
        a useful `ValueError` rather than silently dropping it. Joining Python
        strings is also wrong here because the vocabulary values are bytes.
        """,
        """
        def reconstruct_text(
            token_ids: Sequence[int], vocab: Mapping[int, bytes]
        ) -> str:
            # LLM_TODO_CH02_01_BYTES
            raise NotImplementedError(
                "join all byte pieces before UTF-8 decoding"
            )
        """,
        """
        pieces = {0: b"A", 1: b"\\xe4", 2: b"\\xbd\\xa0", 3: b"!"}
        assert reconstruct_text([0, 1, 2, 3], pieces) == "A\\u4f60!"
        try:
            reconstruct_text([99], pieces)
        except ValueError:
            pass
        else:
            raise AssertionError("unknown IDs should be reported")
        print("Exercise 2.1 passed")
        """,
    )
    cells += exercise(
        "2.2",
        "Create shifted sliding windows",
        "Build deterministic input/target windows and verify their shape and stride.",
        """
        Autoregressive training asks the model to predict the next token at
        every position. For a token stream $x_0,x_1,\\ldots$, a window beginning
        at $s$ uses input $[x_s,\\ldots,x_{s+L-1}]$ and target
        $[x_{s+1},\\ldots,x_{s+L}]$. The two tensors have the same length, but
        the target is shifted by one.

        `stride` controls the distance between starting positions, not the
        distance between tokens inside one window. A stride smaller than
        `max_length` creates overlapping examples; a larger stride skips parts
        of the stream. Only complete input/target pairs should be emitted.
        """,
        """
        For IDs `0..11`, `max_length=4`, and `stride=3`, starts are `0, 3, 6`.
        The first two inputs in a batch of two are `[0,1,2,3]` and
        `[3,4,5,6]`; their targets are `[1,2,3,4]` and `[4,5,6,7]`.
        The first column therefore reveals the start-position stride.
        """,
        """
        The range must leave one extra token for the shifted target. Validate
        positive lengths and strides. Keep `shuffle=False` for this reasoning
        exercise so batch order remains inspectable.
        """,
        """
        class SlidingWindowDataset(Dataset[tuple[Tensor, Tensor]]):
            def __init__(
                self, token_ids: Sequence[int], max_length: int, stride: int
            ) -> None:
                # LLM_TODO_CH02_02_WINDOWS
                raise NotImplementedError("construct complete shifted windows")

            def __len__(self) -> int:
                raise NotImplementedError("return the window count")

            def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
                raise NotImplementedError("return one input/target pair")


        def make_loader(
            token_ids: Sequence[int],
            max_length: int,
            stride: int,
            batch_size: int,
        ) -> DataLoader[tuple[Tensor, Tensor]]:
            raise NotImplementedError("create a deterministic DataLoader")
        """,
        """
        loader = make_loader(range(12), max_length=4, stride=3, batch_size=2)
        inputs, targets = next(iter(loader))
        assert inputs.shape == targets.shape == (2, 4)
        assert inputs[:, 0].tolist() == [0, 3]
        assert torch.equal(targets[:, :-1], inputs[:, 1:])
        print("Exercise 2.2 passed")
        """,
    )
    cells += finish(
        "02",
        """
        - Why can a valid character fail to decode when a token piece is viewed alone?
        - How does decreasing stride change data reuse and training cost?
        - Why does a length-$L$ input require $L+1$ source tokens?
        """,
    )
    return cells


def chapter03() -> list[dict[str, object]]:
    cells = intro(
        "03",
        "Attention dimensions and projection layouts",
        "https://github.com/skindhu/Build-A-Large-Language-Model-CN/blob/main/cn-Book/3.%E5%AE%9E%E7%8E%B0%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6.md",
        """
        Attention is mostly disciplined shape manipulation. These exercises
        focus on matrix-storage conventions, concatenating independent heads,
        and validating the divisibility condition that makes a multi-head split
        possible.
        """,
    )
    cells.append(
        code(
            """
            from __future__ import annotations

            import math

            import torch
            from torch import Tensor, nn
            """
        )
    )
    cells += exercise(
        "implementation A",
        "Simple self-attention",
        "Implement the complete score-normalize-mix path using the input as Q, K, and V.",
        """
        Self-attention creates a context-dependent representation for every
        token. In the simplest version, the same input tensor supplies queries,
        keys, and values. Pairwise scores are
        $S=XX^T/\\sqrt{D}$, row-wise weights are
        $A=\\operatorname{softmax}(S)$, and context is $C=AX$.

        For input shape `[B,T,D]`, scores and weights have shape `[B,T,T]`.
        Each row of `A` sums to one and describes how one query position mixes
        all value positions. The scale $\\sqrt{D}$ keeps score magnitude from
        growing unchecked as channel width increases.
        """,
        """
        Start with `[B,T,D]`. Transpose only the final two axes to obtain
        `[B,D,T]`; batch matrix multiplication then yields `[B,T,T]`. Applying
        weights `[B,T,T]` to values `[B,T,D]` returns `[B,T,D]`.
        """,
        """
        Softmax belongs on the key axis (`dim=-1`). Do not transpose the batch
        axis. Verify both output shape and row sums; a shape-only test cannot
        detect normalization on the wrong axis.
        """,
        """
        def simple_self_attention(inputs: Tensor) -> tuple[Tensor, Tensor]:
            # ATTENTION_TODO_CH03_00_SIMPLE
            raise NotImplementedError(
                "compute scaled scores, weights, and context"
            )
        """,
        """
        torch.manual_seed(1)
        implementation_inputs = torch.randn(2, 5, 4)
        context, weights = simple_self_attention(implementation_inputs)
        assert context.shape == implementation_inputs.shape
        assert weights.shape == (2, 5, 5)
        assert torch.allclose(
            weights.sum(dim=-1),
            torch.ones_like(weights.sum(dim=-1)),
        )
        print("Simple attention passed")
        """,
    )
    cells += exercise(
        "implementation B",
        "Learned causal Q/K/V attention",
        "Add learned projections and prevent every position from reading future tokens.",
        """
        Learned attention projects the input into queries, keys, and values:
        $Q=XW_Q$, $K=XW_K$, and $V=XW_V$. Scores are
        $QK^T/\\sqrt{D_h}$. A causal decoder must hide score entries where key
        position $j$ is greater than query position $i$.

        Store the upper-triangular mask as a registered Boolean buffer. It then
        follows the module across devices without becoming a trainable
        parameter. Replace masked scores with negative infinity before softmax,
        making their resulting probability zero.
        """,
        """
        Q, K, and V each have `[B,T,Dout]`; scores have `[B,T,T]`. Slice a
        `[context,context]` mask to `[:T,:T]`. The output after weights times V
        is `[B,T,Dout]`. Changing tokens at positions 3 and 4 must not alter
        outputs at positions 0 through 2.
        """,
        """
        Mask before softmax, scale by key width, and reject sequences longer
        than the configured context. Dropout applies to attention weights during
        training; use dropout zero for deterministic checks.
        """,
        """
        class CausalSelfAttention(nn.Module):
            def __init__(
                self,
                input_dim: int,
                output_dim: int,
                context_length: int,
                dropout: float = 0.0,
                qkv_bias: bool = False,
            ) -> None:
                super().__init__()
                # ATTENTION_TODO_CH03_00_CAUSAL
                raise NotImplementedError(
                    "create projections, dropout, and a causal-mask buffer"
                )

            def forward(self, inputs: Tensor) -> Tensor:
                raise NotImplementedError(
                    "project, mask, normalize, and mix values"
                )
        """,
        """
        causal = CausalSelfAttention(4, 4, context_length=5)
        changed = implementation_inputs.clone()
        changed[:, 3:] += 100
        original_prefix = causal(implementation_inputs)[:, :3]
        changed_prefix = causal(changed)[:, :3]
        assert original_prefix.shape == (2, 3, 4)
        assert torch.allclose(original_prefix, changed_prefix, atol=1e-5, rtol=1e-5)
        print("Causal attention passed")
        """,
    )
    cells += exercise(
        "implementation C",
        "Efficient multi-head causal attention",
        "Project all heads together, split a head axis, attend in parallel, and merge.",
        """
        Multi-head attention uses several smaller attention spaces in parallel.
        One dense projection still produces width $D_{out}$, then reshaping
        exposes `H` heads of width $D_h=D_{out}/H`. The score tensor becomes
        `[B,H,T,T]`, so every head has its own attention matrix.

        After attention, transpose `[B,H,T,Dh]` to `[B,T,H,Dh]` and merge the
        final two axes back to `[B,T,Dout]`. An output projection can then mix
        information across heads. The causal mask broadcasts over batch and
        head dimensions.
        """,
        """
        With `Dout=8` and `H=2`, each head width is four. Q/K/V move from
        `[B,T,8]` to `[B,2,T,4]`. Scores are `[B,2,T,T]`; merged context returns
        to `[B,T,8]`.
        """,
        """
        Require exact divisibility. Call `.contiguous()` after transposing and
        before `view`, because transpose changes strides. Scale by per-head
        width, not the full output width.
        """,
        """
        class MultiHeadCausalAttention(nn.Module):
            def __init__(
                self,
                input_dim: int,
                output_dim: int,
                context_length: int,
                num_heads: int,
                dropout: float = 0.0,
                qkv_bias: bool = False,
            ) -> None:
                super().__init__()
                # ATTENTION_TODO_CH03_00_MULTIHEAD
                raise NotImplementedError(
                    "create projections and record head dimensions"
                )

            def forward(self, inputs: Tensor) -> Tensor:
                raise NotImplementedError(
                    "split heads, attend causally, and merge heads"
                )
        """,
        """
        multihead = MultiHeadCausalAttention(
            4, 8, context_length=5, num_heads=2
        )
        output = multihead(implementation_inputs)
        changed_output = multihead(changed)
        assert output.shape == (2, 5, 8)
        assert torch.allclose(
            output[:, :3], changed_output[:, :3], atol=1e-5, rtol=1e-5
        )
        print("Multi-head causal attention passed")
        """,
    )
    cells += exercise(
        "3.1",
        "Transfer projection weights",
        "Copy a raw matrix projection into `nn.Linear` without changing outputs.",
        """
        Suppose a hand-written layer stores $W_{raw}$ with shape
        `[in_features, out_features]` and computes $XW_{raw}$. PyTorch's
        `nn.Linear` stores its weight as `[out_features, in_features]` and
        computes $XW_{linear}^{T}+b$. Equivalent behavior therefore requires
        $W_{linear}=W_{raw}^{T}$.

        This is a storage-layout issue, not a mathematical change to the
        projection. Copy under `torch.no_grad()` so setup is not added to the
        autograd graph. A bias must be zeroed when the raw projection has none.
        """,
        """
        With `X: [B,T,Din]` and raw weight `[Din,Dout]`, the raw output is
        `[B,T,Dout]`. `nn.Linear` accepts the same input but internally
        transposes its `[Dout,Din]` parameter. Matching numerical output is a
        stronger check than comparing shapes.
        """,
        """
        Do not assign a new tensor to `linear.weight`; copy into the existing
        parameter. Check both dimensions before transposing. Forgetting the bias
        can produce a plausible but shifted result.
        """,
        """
        class RawProjection(nn.Module):
            def __init__(self, in_features: int, out_features: int) -> None:
                super().__init__()
                self.weight = nn.Parameter(
                    torch.empty(in_features, out_features)
                )

            def forward(self, inputs: Tensor) -> Tensor:
                return inputs @ self.weight


        def transfer_raw_to_linear(
            raw_weight: Tensor, linear: nn.Linear
        ) -> None:
            # LLM_TODO_CH03_01_TRANSFER
            raise NotImplementedError("copy using nn.Linear's storage layout")
        """,
        """
        torch.manual_seed(7)
        raw = RawProjection(4, 3)
        nn.init.normal_(raw.weight)
        linear = nn.Linear(4, 3)
        transfer_raw_to_linear(raw.weight, linear)
        sample = torch.randn(2, 5, 4)
        assert torch.allclose(raw(sample), linear(sample))
        print("Exercise 3.1 passed")
        """,
    )
    cells += exercise(
        "3.2",
        "Concatenate multiple heads",
        "Combine head outputs on the feature axis and reason about total width.",
        """
        A simple teaching implementation can run several attention heads
        independently and concatenate their outputs. If each head produces
        `[B,T,Dh]` and there are $H$ heads, concatenation on the last axis gives
        `[B,T,H\\cdot Dh]`.

        Concatenating on the sequence axis would instead alter token count,
        destroying the alignment between positions. The batch and sequence
        dimensions must remain unchanged because every head describes the same
        examples and token positions.
        """,
        """
        Two toy heads mapping width 4 to width 2 each receive `[2,5,4]`. Each
        returns `[2,5,2]`; concatenation along `dim=-1` returns `[2,5,4]`.
        To preserve a desired model width $D$, choose $Dh=D/H$.
        """,
        """
        Require at least one head and register heads in `nn.ModuleList`, or
        parameters will not appear in the wrapper's state dictionary. Use
        `dim=-1`, not `dim=1`.
        """,
        """
        class MultiHeadWrapper(nn.Module):
            def __init__(self, heads: list[nn.Module]) -> None:
                super().__init__()
                if not heads:
                    raise ValueError("at least one head is required")
                self.heads = nn.ModuleList(heads)

            def forward(self, inputs: Tensor) -> Tensor:
                # LLM_TODO_CH03_02_MULTIHEAD
                raise NotImplementedError(
                    "concatenate head outputs on the feature axis"
                )
        """,
        """
        wrapper = MultiHeadWrapper([nn.Linear(4, 2), nn.Linear(4, 2)])
        output = wrapper(torch.randn(2, 5, 4))
        assert output.shape == (2, 5, 4)
        print("Exercise 3.2 passed")
        """,
    )
    cells += exercise(
        "3.3",
        "Validate per-head dimensions",
        "Check that an embedding width can be split evenly across attention heads.",
        """
        Efficient multi-head attention reshapes a model-width axis $D$ into
        `[H, Dh]`, where $Dh=D/H$. A reshape cannot create or discard elements,
        so $D$ must be divisible by $H$. GPT-2 small uses $D=768$ and $H=12$,
        yielding $Dh=64$.

        The full flow for a projected tensor is
        `[B,T,D] -> [B,T,H,Dh] -> [B,H,T,Dh]`. The transpose places heads before
        tokens so batched attention scores become `[B,H,T,T]`.
        """,
        """
        `768 / 12 = 64`, so each head receives 64 channels. A width of 10 with
        3 heads is invalid because no integer per-head width preserves all ten
        channels.
        """,
        """
        Reject zero and negative dimensions before using modulo. Returning a
        truncated integer division for a non-divisible width hides a later
        reshape error and makes debugging harder.
        """,
        """
        def validate_attention_dimensions(
            embedding_dim: int, num_heads: int
        ) -> int:
            # LLM_TODO_CH03_03_DIMENSIONS
            raise NotImplementedError(
                "validate divisibility and return per-head width"
            )
        """,
        """
        assert validate_attention_dimensions(768, 12) == 64
        try:
            validate_attention_dimensions(10, 3)
        except ValueError:
            pass
        else:
            raise AssertionError("an uneven split should fail")
        print("Exercise 3.3 passed")
        """,
    )
    cells += finish(
        "03",
        """
        - Why does `nn.Linear.weight` look transposed relative to $XW$ notation?
        - Which tensor dimensions describe examples, positions, heads, and channels?
        - Where would a wrong concatenation axis first become observable?
        """,
    )
    return cells


def chapter04() -> list[dict[str, object]]:
    cells = intro(
        "04",
        "Reasoning about GPT architecture size",
        "https://github.com/skindhu/Build-A-Large-Language-Model-CN/blob/main/cn-Book/4.%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AE%9E%E7%8E%B0%E4%B8%80%E4%B8%AA%E7%94%A8%E4%BA%8E%E6%96%87%E6%9C%AC%E7%94%9F%E6%88%90%E7%9A%84%20GPT%20%E6%A8%A1%E5%9E%8B.md",
        """
        Before allocating a model, parameter arithmetic reveals where capacity
        and memory go. These exercises derive the dominant attention and
        feed-forward terms, then estimate complete GPT-2-style model sizes from
        configuration values.
        """,
    )
    cells.append(
        code(
            """
            from __future__ import annotations

            from dataclasses import dataclass


            @dataclass(frozen=True)
            class GPTConfig:
                vocab_size: int
                context_length: int
                embedding_dim: int
                num_layers: int
                num_heads: int
            """
        )
    )
    cells += exercise(
        "4.1",
        "Count attention and feed-forward parameters",
        "Derive parameter counts from layer dimensions rather than instantiating modules.",
        """
        A dense projection from $D_{in}$ to $D_{out}$ has
        $D_{in}D_{out}$ weights and $D_{out}$ biases. A standard attention block
        has four width-preserving projections: query, key, value, and output.
        Its leading weight term is therefore $4D^2$.

        A feed-forward network expands from $D$ to $ED$ and contracts back.
        Its leading weight term is $D(ED)+(ED)D=2ED^2$. With expansion $E=4$,
        this is $8D^2$, approximately twice attention's projection parameters.
        Bias terms matter for exact counts but not for the dominant ratio.
        """,
        """
        For width `D=768`, Q/K/V/output each use a `768 x 768` weight. The FFN
        uses `768 x 3072` and `3072 x 768`. Return exact totals including the
        stated biases, then compare the ratio instead of relying on memory.
        """,
        """
        Count parameters, not activations. Do not multiply by sequence length or
        batch size; shared weights are reused at every position. State whether
        biases are included so another implementation can reproduce the count.
        """,
        """
        def projection_parameter_counts(
            embedding_dim: int, expansion: int = 4
        ) -> dict[str, int]:
            # LLM_TODO_CH04_01_COUNTS
            raise NotImplementedError(
                "count exact attention and FFN weights plus biases"
            )
        """,
        """
        counts = projection_parameter_counts(768)
        assert set(counts) == {"attention", "feed_forward"}
        ratio = counts["feed_forward"] / counts["attention"]
        assert 1.9 < ratio < 2.1
        print(f"Exercise 4.1 passed: FFN/attention = {ratio:.3f}")
        """,
    )
    cells += exercise(
        "4.2",
        "Estimate complete GPT-2 variants",
        "Compute tied-output parameter and weights-only memory estimates without allocation.",
        """
        A GPT-2-style estimate combines token embeddings $V\\times D$, position
        embeddings $T\\times D$, $L$ Transformer blocks, and a final layer norm.
        With pre-normalization and an expansion of four, one block contributes
        approximately $12D^2$ plus linear-in-$D$ bias and normalization terms.

        If the output classifier shares the token-embedding matrix, do not count
        another $V\\times D$ matrix. Weights-only memory is
        `parameter_count * bytes_per_parameter`. Training requires substantially
        more memory for gradients, optimizer state, activations, and temporary
        buffers; this exercise deliberately estimates only stored weights.
        """,
        """
        GPT-2 small uses vocabulary 50,257, context 1,024, width 768, and 12
        blocks. An XL-style configuration uses width 1,600, 48 blocks, and 25
        heads. Arithmetic should place the latter near 1.6 billion parameters
        without constructing a multi-gigabyte module.
        """,
        """
        Keep units explicit: decimal billions differ from binary GiB. Validate
        that width is divisible by head count even though head count does not
        change the dense projection total. Avoid double-counting tied output
        embeddings.
        """,
        """
        def estimate_gpt2(
            config: GPTConfig, bytes_per_parameter: int = 4
        ) -> tuple[int, int]:
            # LLM_TODO_CH04_02_ESTIMATE
            raise NotImplementedError(
                "sum tied embeddings, blocks, and final normalization"
            )
        """,
        """
        small = GPTConfig(50_257, 1_024, 768, 12, 12)
        xl = GPTConfig(50_257, 1_024, 1_600, 48, 25)
        small_params, _ = estimate_gpt2(small)
        xl_params, xl_bytes = estimate_gpt2(xl)
        assert 120_000_000 < small_params < 130_000_000
        assert 1_500_000_000 < xl_params < 1_650_000_000
        assert xl_bytes == xl_params * 4
        print(f"Exercise 4.2 passed: XL weights = {xl_bytes / 2**30:.2f} GiB")
        """,
    )
    cells += finish(
        "04",
        """
        - Why does the FFN dominate one block's parameter count at expansion four?
        - Why does changing head count leave the dense projection count unchanged?
        - Which major training-memory categories are absent from weights-only memory?
        """,
    )
    return cells


def chapter05() -> list[dict[str, object]]:
    cells = intro(
        "05",
        "Decoding, evaluation, and resumable training",
        "https://github.com/skindhu/Build-A-Large-Language-Model-CN/blob/main/cn-Book/5.%E5%9C%A8%E6%97%A0%E6%A0%87%E8%AE%B0%E6%95%B0%E6%8D%AE%E9%9B%86%E4%B8%8A%E8%BF%9B%E8%A1%8C%E9%A2%84%E8%AE%AD%E7%BB%83.md",
        """
        Training and generation use the same logits differently. Training scores
        known targets with cross entropy; generation transforms logits into a
        token choice. This chapter also treats checkpointing and configuration
        selection as correctness problems rather than incidental plumbing.
        """,
    )
    cells.append(
        code(
            """
            from __future__ import annotations

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
            """
        )
    )
    cells += exercise(
        "5.1",
        "Observe temperature sampling",
        "Scale logits, sample reproducibly, and compare token-frequency concentration.",
        """
        Temperature rescales logits before softmax:
        $p_i=\\exp(z_i/\\tau)/\\sum_j\\exp(z_j/\\tau)$. For $0<\\tau<1$, logit
        differences are amplified and probability concentrates on the largest
        logit. For $\\tau>1$, differences shrink and the distribution flattens.

        Sampling frequencies are random estimates of probabilities, so exact
        counts should not be asserted. A seeded generator makes a demonstration
        reproducible, while enough draws make the concentration trend visible.
        Temperature zero is handled by greedy selection, not division by zero.
        """,
        """
        For logits `[0,1,2]`, compare 2,000 draws at temperatures 0.2 and 2.0.
        The last token should dominate much more strongly in the cold sample.
        Return a count for every vocabulary entry, including entries not drawn.
        """,
        """
        Apply temperature to logits, not probabilities. Use replacement because
        each generation step samples from the full vocabulary. Pass an explicit
        `torch.Generator` instead of changing global RNG state.
        """,
        """
        def sample_frequencies(
            logits: Tensor, temperature: float, draws: int, seed: int
        ) -> Tensor:
            # LLM_TODO_CH05_01_TEMPERATURE
            raise NotImplementedError(
                "sample seeded counts from temperature-scaled logits"
            )
        """,
        """
        logits = torch.tensor([0.0, 1.0, 2.0])
        cold = sample_frequencies(logits, 0.2, 2_000, seed=4)
        warm = sample_frequencies(logits, 2.0, 2_000, seed=4)
        assert cold.sum() == warm.sum() == 2_000
        assert cold[-1] > warm[-1]
        print("Exercise 5.1 passed")
        """,
    )
    cells += exercise(
        "5.2",
        "Choose a decoding profile",
        "Represent deterministic and creative decoding goals as explicit settings.",
        """
        `top_k` limits sampling to the $k$ largest logits; temperature reshapes
        probabilities within the retained set. A deterministic workflow can set
        `top_k=1`, while exploratory generation can retain more candidates and
        use a temperature above one.

        These are task policies, not universal quality settings. A profile
        should be named by intent so callers do not scatter unexplained numeric
        constants throughout generation code.
        """,
        """
        Return a dataclass for `"deterministic"` and `"creative"`. The former
        must retain one candidate. The latter should retain multiple candidates
        and use a temperature above one in this toy policy.
        """,
        """
        Reject unknown profile names instead of silently choosing a default.
        Avoid claiming that one profile is always better; generation goals and
        model calibration differ.
        """,
        """
        def choose_decoding_profile(task: str) -> DecodingProfile:
            # LLM_TODO_CH05_02_PROFILES
            raise NotImplementedError("map a named intent to explicit settings")
        """,
        """
        assert choose_decoding_profile("deterministic").top_k == 1
        creative = choose_decoding_profile("creative")
        assert creative.top_k > 1 and creative.temperature > 1.0
        print("Exercise 5.2 passed")
        """,
    )
    cells += exercise(
        "5.3",
        "Generate deterministically",
        "Append the unique top-1 token at each autoregressive step.",
        """
        Autoregressive generation repeatedly evaluates the current context,
        selects a next token, appends it, and uses the extended sequence on the
        next iteration. With `top_k=1`, sampling and argmax are equivalent
        because only one candidate survives.

        The function here receives a callback returning next-token logits with
        shape `[1,V]`. Keeping that contract small separates generation logic
        from any particular model implementation.
        """,
        """
        The toy callback makes the successor of token $t$ equal to
        $(t+1)\\bmod 5`. Starting with `[0]` and generating four tokens should
        yield `[0,1,2,3,4]`.
        """,
        """
        Select from the final logit row and append a Python integer. Validate the
        callback shape. Do not mutate the caller's prompt list unexpectedly;
        begin with a copy.
        """,
        """
        def generate_deterministic(
            next_logits: Callable[[Tensor], Tensor],
            prompt: list[int],
            max_new_tokens: int,
            top_k: int = 1,
        ) -> list[int]:
            # LLM_TODO_CH05_03_DETERMINISTIC
            raise NotImplementedError("append one top-ranked token per step")
        """,
        """
        def toy_next(context: Tensor) -> Tensor:
            target = (int(context[0, -1]) + 1) % 5
            result = torch.zeros(1, 5)
            result[0, target] = 10.0
            return result


        prompt = [0]
        generated = generate_deterministic(toy_next, prompt, 4)
        assert generated == [0, 1, 2, 3, 4]
        assert prompt == [0]
        print("Exercise 5.3 passed")
        """,
    )
    cells += exercise(
        "5.4",
        "Checkpoint and resume optimizer state",
        "Save model parameters, optimizer dynamics, and progress as one logical checkpoint.",
        """
        Resuming training requires more than model weights. Optimizers can carry
        momentum, running averages, and step counters that affect the next
        update. A useful checkpoint stores both state dictionaries plus explicit
        progress such as the completed step.

        State dictionaries contain tensors that may later mutate, so an
        in-memory snapshot should be deep-copied. Loading must target compatible
        model and optimizer structures. In production, also preserve scheduler,
        scaler, RNG, and data-order state when exact reproducibility matters.
        """,
        """
        Train a one-weight linear model for one step, snapshot it, restore into
        a fresh model and momentum optimizer, then take the second step. Its
        weight should match a model trained for two uninterrupted steps.
        """,
        """
        Restore optimizer state after constructing the optimizer over the new
        model's parameters. Do not store live `state_dict()` references in an
        in-memory checkpoint. Return the saved step so the caller can continue
        logging and scheduling correctly.
        """,
        """
        def make_checkpoint(
            model: nn.Module, optimizer: Optimizer, step: int
        ) -> dict[str, Any]:
            # LLM_TODO_CH05_04_CHECKPOINT
            raise NotImplementedError("snapshot model, optimizer, and step")


        def restore_checkpoint(
            checkpoint: dict[str, Any],
            model: nn.Module,
            optimizer: Optimizer,
        ) -> int:
            raise NotImplementedError("restore both state dictionaries")
        """,
        """
        import copy


        def train_step(model: nn.Module, optimizer: Optimizer) -> None:
            x = torch.tensor([[1.0], [2.0]])
            y = torch.tensor([[2.0], [4.0]])
            optimizer.zero_grad()
            loss = F.mse_loss(model(x), y)
            loss.backward()
            optimizer.step()


        torch.manual_seed(3)
        model = nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
        train_step(model, optimizer)
        saved = make_checkpoint(model, optimizer, step=1)

        resumed = nn.Linear(1, 1, bias=False)
        resumed_optimizer = torch.optim.SGD(
            resumed.parameters(), lr=0.05, momentum=0.9
        )
        assert restore_checkpoint(saved, resumed, resumed_optimizer) == 1
        before = copy.deepcopy(resumed.weight)
        train_step(resumed, resumed_optimizer)
        assert not torch.equal(before, resumed.weight)
        print("Exercise 5.4 passed")
        """,
    )
    cells += exercise(
        "5.5",
        "Compare train and validation cross entropy",
        "Calculate losses on explicitly supplied toy splits and interpret them cautiously.",
        """
        For one target class $y$, cross entropy is
        $-\\log(\\operatorname{softmax}(z)_y)$. PyTorch accepts a batch of logits
        `[N,C]` and integer targets `[N]`, averages the per-example losses, and
        remains numerically stable by combining log-softmax with indexing.

        Train/validation loss is evidence about performance on those specific
        splits. Similar values do not prove that data was absent from
        pretraining, while a gap alone does not identify its cause. This toy
        exercise demonstrates mechanics only and deliberately avoids claims
        about membership of any real corpus.
        """,
        """
        The training logits strongly favor their targets, while validation
        logits are less decisive. Compute each loss independently and return
        Python floats. The expected relation is train loss below validation
        loss, not a memorized numeric constant.
        """,
        """
        Pass raw logits to `cross_entropy`; do not apply softmax first. Ensure
        target dtype is integer class indices. Keep split labels explicit so no
        accidental data reuse occurs.
        """,
        """
        def split_cross_entropy(
            train_logits: Tensor,
            train_targets: Tensor,
            validation_logits: Tensor,
            validation_targets: Tensor,
        ) -> tuple[float, float]:
            # LLM_TODO_CH05_05_CROSS_ENTROPY
            raise NotImplementedError("score both supplied splits")
        """,
        """
        train_loss, validation_loss = split_cross_entropy(
            torch.tensor([[3.0, 0.0], [0.0, 3.0]]),
            torch.tensor([0, 1]),
            torch.tensor([[0.4, 0.6], [0.6, 0.4]]),
            torch.tensor([0, 1]),
        )
        assert train_loss < validation_loss
        print("Exercise 5.5 passed")
        """,
    )
    cells += exercise(
        "5.6",
        "Switch model configuration metadata",
        "Select GPT-2 variant metadata without allocating either model.",
        """
        Configuration selection should be separate from model construction.
        This lets tooling inspect dimensions, reject invalid names, and estimate
        resources without allocating hundreds of millions of parameters.

        The nominal names `124M` and `1558M` are labels used for this exercise.
        Exact counts depend on conventions such as tied embeddings and bias
        inclusion. Store the dimensions and nominal count as immutable metadata.
        """,
        """
        A lookup for `124M` should report 12 layers; `1558M` should report width
        1,600, 48 layers, 25 heads, and its nominal parameter label. No
        `nn.Module` is created.
        """,
        """
        Return a clear error listing valid names. Do not silently fall back to a
        smaller configuration: accidental allocation choices can be expensive.
        """,
        """
        CONFIGS = {
            "124M": GPTConfig("124M", 768, 12, 12, 124_000_000),
            "1558M": GPTConfig("1558M", 1_600, 48, 25, 1_558_000_000),
        }


        def select_gpt2_config(name: str) -> GPTConfig:
            # LLM_TODO_CH05_06_CONFIG
            raise NotImplementedError("look up metadata without allocation")
        """,
        """
        assert select_gpt2_config("124M").num_layers == 12
        xl = select_gpt2_config("1558M")
        assert (xl.embedding_dim, xl.num_layers, xl.num_heads) == (1_600, 48, 25)
        print("Exercise 5.6 passed")
        """,
    )
    cells += finish(
        "05",
        """
        - How do temperature and top-k change different parts of token selection?
        - Which state is required for exact training continuation beyond this toy checkpoint?
        - What can validation loss support, and what conclusions require stronger evidence?
        """,
    )
    return cells


def chapter06() -> list[dict[str, object]]:
    cells = intro(
        "06",
        "Classification fine-tuning mechanics",
        "https://github.com/skindhu/Build-A-Large-Language-Model-CN/blob/main/cn-Book/6.%E7%94%A8%E4%BA%8E%E5%88%86%E7%B1%BB%E4%BB%BB%E5%8A%A1%E7%9A%84%E5%BE%AE%E8%B0%83.md",
        """
        Classification reuses token representations but changes how they are
        pooled and which parameters are updated. These exercises make padding,
        parameter freezing, and sequence-position selection explicit.
        """,
    )
    cells.append(
        code(
            """
            from __future__ import annotations

            from collections.abc import Sequence
            from typing import Literal

            import torch
            from torch import Tensor, nn
            """
        )
    )
    cells += exercise(
        "6.1",
        "Pad sequences and preserve validity",
        "Right-pad variable-length token rows and return a Boolean attention mask.",
        """
        Dense batches require a common sequence length. Right-padding places
        real tokens first and fills the remainder with `pad_id`. A mask records
        which positions are valid so attention, pooling, and loss logic can
        ignore padding.

        Padding to the longest row in a batch minimizes wasted work; padding to
        a fixed maximum can simplify shapes but may add substantial computation.
        The mask, not the numeric pad ID alone, is the reliable semantic signal.
        """,
        """
        For rows `[1,2,3]` and `[4]`, automatic length is three. Padded tokens
        are `[[1,2,3],[4,pad,pad]]`; the Boolean mask is
        `[[T,T,T],[T,F,F]]`. An explicit `max_length` may truncate longer rows
        and pad shorter ones.
        """,
        """
        Return tensors with identical `[B,T]` shape. Decide truncation before
        padding, validate positive explicit lengths, and use Boolean mask dtype.
        Do not infer validity later from hidden-state values.
        """,
        """
        def pad_and_mask(
            sequences: Sequence[Sequence[int]],
            pad_id: int,
            max_length: int | None = None,
        ) -> tuple[Tensor, Tensor]:
            # LLM_TODO_CH06_01_PADDING
            raise NotImplementedError("right-pad and return a validity mask")
        """,
        """
        tokens, mask = pad_and_mask([[1, 2, 3], [4]], pad_id=0)
        assert tokens.tolist() == [[1, 2, 3], [4, 0, 0]]
        assert mask.dtype == torch.bool
        assert mask.tolist() == [[True, True, True], [True, False, False]]
        print("Exercise 6.1 passed")
        """,
    )
    cells += exercise(
        "6.2",
        "Select trainable parameters",
        "Toggle between classifier-only and full-model fine-tuning and count parameters.",
        """
        Freezing a parameter sets `requires_grad=False`, preventing gradient
        storage and optimizer updates for that parameter. Head-only training
        adapts a small classifier while preserving the backbone; full
        fine-tuning exposes every parameter to updates.

        A configuration helper should first set every parameter consistently,
        then enable only the named head when requested. Returning the trainable
        parameter count makes the policy observable and testable.
        """,
        """
        In a toy module with `backbone` and `classifier`, head mode should leave
        only classifier weight and bias trainable. Full mode should make the
        count equal the model's total parameter count.
        """,
        """
        Iterate over parameters rather than modules because a module may own
        multiple tensors. Validate the head name and mode. Changing
        `requires_grad` does not automatically rebuild an already-created
        optimizer.
        """,
        """
        def configure_trainable(
            model: nn.Module,
            mode: Literal["head", "full"],
            head_name: str = "classifier",
        ) -> int:
            # LLM_TODO_CH06_02_UNFREEZE
            raise NotImplementedError(
                "set requires_grad according to the selected policy"
            )
        """,
        """
        toy_model = nn.ModuleDict({
            "backbone": nn.Linear(4, 4),
            "classifier": nn.Linear(4, 2),
        })
        head_count = configure_trainable(toy_model, "head")
        expected_head = sum(p.numel() for p in toy_model["classifier"].parameters())
        assert head_count == expected_head
        full_count = configure_trainable(toy_model, "full")
        assert full_count == sum(p.numel() for p in toy_model.parameters())
        print("Exercise 6.2 passed")
        """,
    )
    cells += exercise(
        "6.3",
        "Choose a sequence representation",
        "Select either position zero or each row's last valid hidden state.",
        """
        A decoder produces hidden states `[B,T,D]`. A classifier needs one
        vector per sequence, so a policy must choose a position. Position zero
        is easy to index but in causal attention it has seen only itself. The
        last valid token can incorporate all earlier valid tokens.

        Variable lengths mean the last valid index differs by row. If a Boolean
        mask contains `L` true values in a right-padded row, the index is
        `L-1`. Gathering with batch indices returns `[B,D]`.
        """,
        """
        For valid lengths `[3,1]`, gather hidden states at positions `[2,0]`.
        Compare this with selecting `hidden_states[:,0,:]`, which always uses
        position zero regardless of sequence length.
        """,
        """
        Reject rows with no valid token. Keep index tensors on the same device
        as hidden states. Do not use `hidden_states[:,-1]` because that selects
        padding for shorter rows.
        """,
        """
        def select_sequence_representation(
            hidden_states: Tensor,
            attention_mask: Tensor,
            position: Literal["first", "last_valid"],
        ) -> Tensor:
            # LLM_TODO_CH06_03_REPRESENTATION
            raise NotImplementedError("select first or gather last-valid states")
        """,
        """
        hidden = torch.arange(2 * 3 * 2).reshape(2, 3, 2)
        mask = torch.tensor([[True, True, True], [True, False, False]])
        first = select_sequence_representation(hidden, mask, "first")
        last = select_sequence_representation(hidden, mask, "last_valid")
        assert torch.equal(first, hidden[:, 0])
        assert torch.equal(last, torch.stack([hidden[0, 2], hidden[1, 0]]))
        print("Exercise 6.3 passed")
        """,
    )
    cells += finish(
        "06",
        """
        - Which operations need the padding mask besides attention?
        - Why must the optimizer be reconsidered after changing trainability?
        - How does causal direction affect the information available at position zero?
        """,
    )
    return cells


def chapter07() -> list[dict[str, object]]:
    cells = intro(
        "07",
        "Instruction data, masking, and parameter-efficient adaptation",
        "https://github.com/skindhu/Build-A-Large-Language-Model-CN/blob/main/cn-Book/7.%E6%8C%87%E4%BB%A4%E9%81%B5%E5%BE%AA%E5%BE%AE%E8%B0%83.md",
        """
        Instruction fine-tuning turns structured records into token sequences,
        chooses which tokens contribute to loss, and balances model adaptation
        against memory cost. The final exercise introduces LoRA as a low-rank
        update to a frozen linear layer.
        """,
    )
    cells.append(
        code(
            """
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
            """
        )
    )
    cells += exercise(
        "7.1",
        "Format instruction records",
        "Render one record in two original prompt layouts while preserving field roles.",
        """
        A prompt template serializes structured fields into plain text. The
        exact delimiters are model conventions; the important invariant is that
        instruction, optional context, and expected response remain
        distinguishable and appear in a consistent order.

        This notebook uses independently worded `alpaca_like` and `phi_like`
        layouts for mechanics only. They are not copies of upstream templates.
        Formatting should be deterministic so token counts and masks can be
        reproduced.
        """,
        """
        The record contains an assignment, optional details, and a response.
        The first style uses Markdown-like section labels; the second uses
        user/assistant role tokens. Empty details receive an explicit placeholder
        so record structure remains visible.
        """,
        """
        Strip surrounding whitespace but do not discard meaningful content.
        Reject an empty instruction and an unknown style. Keep the response in
        the returned training string; inference formatting would omit it.
        """,
        """
        def format_instruction(
            record: Mapping[str, str],
            style: Literal["alpaca_like", "phi_like"],
        ) -> str:
            # LLM_TODO_CH07_01_FORMAT
            raise NotImplementedError("render two original prompt layouts")
        """,
        """
        record = {
            "instruction": "Sort the symbols by code point.",
            "input": "z, a, m",
            "output": "a, m, z",
        }
        style_a = format_instruction(record, "alpaca_like")
        style_b = format_instruction(record, "phi_like")
        assert record["instruction"] in style_a and record["output"] in style_a
        assert "<|user|>" in style_b and "<|assistant|>" in style_b
        assert style_a != style_b
        print("Exercise 7.1 passed")
        """,
    )
    cells += exercise(
        "7.2",
        "Mask prompt labels",
        "Exclude a prompt prefix from next-token loss with an ignore index.",
        """
        Language-model inputs and labels are often copies shifted by one token.
        For response-focused instruction tuning, prompt-token label positions can
        be replaced with an `ignore_index` such as `-100`. PyTorch cross entropy
        omits those positions from its reduction.

        Masking labels does not remove prompt tokens from the input. The model
        still reads the instruction as context; it simply receives no direct
        loss for reproducing that prefix.
        """,
        """
        For IDs `[4,5,6,7,8]` and prompt length three, cloned labels become
        `[-100,-100,-100,7,8]`. The original input tensor must remain unchanged.
        """,
        """
        Clone before assignment. Validate that prompt length is between zero and
        sequence length. In a shifted-label pipeline, carefully define whether
        the boundary refers to input or target positions.
        """,
        """
        def mask_prompt_labels(
            input_ids: Tensor,
            prompt_length: int,
            ignore_index: int = -100,
        ) -> Tensor:
            # LLM_TODO_CH07_02_MASK
            raise NotImplementedError("clone labels and mask the prompt prefix")
        """,
        """
        ids = torch.tensor([4, 5, 6, 7, 8])
        labels = mask_prompt_labels(ids, prompt_length=3)
        assert labels.tolist() == [-100, -100, -100, 7, 8]
        assert ids.tolist() == [4, 5, 6, 7, 8]
        print("Exercise 7.2 passed")
        """,
    )
    cells += exercise(
        "7.3",
        "Build a bounded memory plan",
        "Derive a toy microbatch and accumulation plan without downloading a dataset.",
        """
        Dataset size, sequence length, hidden width, and batch size affect
        memory differently. This exercise uses an explicitly labeled heuristic:
        FP32 values times eight retained activation groups. It is not a hardware
        predictor, but it demonstrates how to turn a budget into a bounded
        microbatch.

        Gradient accumulation separates microbatch memory from effective update
        size. Several small forward/backward passes can contribute gradients
        before one optimizer step. The token count per update is
        `micro_batch * accumulation * sequence_length`.
        """,
        """
        Clamp average tokens to `max_sequence_length`, estimate bytes per
        example, choose the largest microbatch under the activation budget, and
        select enough accumulation steps to approach a small target effective
        batch. Report the estimate transparently in MiB.
        """,
        """
        Reject a budget that cannot fit one estimated example. Do not claim the
        estimate includes parameters, optimizer state, allocator fragmentation,
        or attention's implementation-specific temporaries.
        """,
        """
        def build_memory_plan(
            num_examples: int,
            average_tokens: int,
            max_sequence_length: int,
            hidden_dim: int,
            activation_budget_mib: float,
        ) -> MemoryPlan:
            # LLM_TODO_CH07_03_PLAN
            raise NotImplementedError(
                "derive bounded microbatch and accumulation values"
            )
        """,
        """
        plan = build_memory_plan(
            10_000, 128, 256, 256, activation_budget_mib=8
        )
        assert plan.sequence_length == 128
        assert 0 < plan.estimated_activation_mib <= 8
        assert plan.micro_batch_size >= 1
        assert plan.gradient_accumulation_steps >= 1
        print("Exercise 7.3 passed")
        """,
    )
    cells += exercise(
        "7.4",
        "Add a LoRA update",
        "Freeze a base linear layer and learn a scaled low-rank residual.",
        """
        A linear layer computes $xW^T+b$. LoRA freezes $W$ and adds
        $\\Delta W=BA$, where $A\\in\\mathbb{R}^{r\\times D_{in}}$ and
        $B\\in\\mathbb{R}^{D_{out}\\times r}$. The forward pass becomes
        $xW^T+b+(\\alpha/r)xA^TB^T$.

        If $r$ is much smaller than both feature dimensions, trainable values
        drop from roughly $D_{in}D_{out}$ to
        $r(D_{in}+D_{out})$. Initializing $B$ to zero makes the initial update
        zero, so wrapping a layer does not immediately change its output.
        """,
        """
        Wrap a `16 -> 12` linear layer with rank two. Trainable low-rank factors
        contain `2*(16+12)=56` values, fewer than the base layer's trainable
        weight and bias. Before training, wrapped and base outputs should match.
        """,
        """
        Freeze both base weight and bias. Register factors as `nn.Parameter`.
        Validate rank bounds and apply transposes according to PyTorch's linear
        convention. Zero-initialize only one factor; zeroing both can delay
        useful gradients.
        """,
        """
        class LoRALinear(nn.Module):
            def __init__(
                self, base: nn.Linear, rank: int, alpha: float = 1.0
            ) -> None:
                super().__init__()
                # LLM_TODO_CH07_04_LORA
                raise NotImplementedError(
                    "freeze base and create trainable low-rank factors"
                )

            def forward(self, inputs: Tensor) -> Tensor:
                raise NotImplementedError("add the scaled low-rank update")
        """,
        """
        torch.manual_seed(9)
        base = nn.Linear(16, 12)
        sample = torch.randn(3, 16)
        expected = base(sample).detach()
        base_count = sum(p.numel() for p in base.parameters())
        lora = LoRALinear(base, rank=2, alpha=4)
        trainable = sum(p.numel() for p in lora.parameters() if p.requires_grad)
        assert torch.allclose(lora(sample), expected)
        assert trainable == 2 * (16 + 12) < base_count
        print("Exercise 7.4 passed")
        """,
    )
    cells += finish(
        "07",
        """
        - Which prompt fields must be present during training versus inference?
        - Why are prompt tokens retained as inputs when their labels are masked?
        - What does the memory heuristic omit, and how would you measure reality?
        - Why does zero-initializing one LoRA factor preserve the base output?
        """,
    )
    return cells


NOTEBOOKS = {
    "chapter02_text_data/chapter02.ipynb": chapter02,
    "chapter03_attention/chapter03.ipynb": chapter03,
    "chapter04_gpt_architecture/chapter04.ipynb": chapter04,
    "chapter05_pretraining/chapter05.ipynb": chapter05,
    "chapter06_classification/chapter06.ipynb": chapter06,
    "chapter07_instruction_finetuning/chapter07.ipynb": chapter07,
}


def main() -> None:
    for relative_path, build_cells in NOTEBOOKS.items():
        notebook = {
            "cells": build_cells(),
            "metadata": {
                "kernelspec": KERNEL,
                "language_info": {"name": "python", "version": "3"},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        destination = ROOT / relative_path
        destination.write_text(
            json.dumps(notebook, ensure_ascii=True, indent=1) + "\n",
            encoding="utf-8",
        )
        print(f"WROTE {destination.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
