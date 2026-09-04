from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int
    context_length: int
    embedding_dim: int
    num_layers: int
    num_heads: int


def projection_parameter_counts(
    embedding_dim: int, expansion: int = 4
) -> dict[str, int]:
    if embedding_dim <= 0 or expansion <= 0:
        raise ValueError("dimensions must be positive")
    hidden = embedding_dim * expansion
    attention = 4 * (embedding_dim * embedding_dim + embedding_dim)
    feed_forward = embedding_dim * hidden + hidden + hidden * embedding_dim + embedding_dim
    return {"attention": attention, "feed_forward": feed_forward}


def estimate_gpt2(
    config: GPTConfig, bytes_per_parameter: int = 4
) -> tuple[int, int]:
    if min(
        config.vocab_size,
        config.context_length,
        config.embedding_dim,
        config.num_layers,
        config.num_heads,
        bytes_per_parameter,
    ) <= 0:
        raise ValueError("configuration values must be positive")
    if config.embedding_dim % config.num_heads:
        raise ValueError("embedding_dim must be divisible by num_heads")

    width = config.embedding_dim
    embeddings = (config.vocab_size + config.context_length) * width
    # Per block: QKV + output projections, 4x FFN, and two layer norms.
    transformer_block = 12 * width * width + 13 * width
    final_layer_norm = 2 * width
    parameters = embeddings + config.num_layers * transformer_block + final_layer_norm
    return parameters, parameters * bytes_per_parameter


def main() -> None:
    counts = projection_parameter_counts(768)
    assert counts["feed_forward"] > counts["attention"]

    small = GPTConfig(50_257, 1_024, 768, 12, 12)
    xl = GPTConfig(50_257, 1_024, 1_600, 48, 25)
    small_parameters, small_bytes = estimate_gpt2(small)
    xl_parameters, xl_bytes = estimate_gpt2(xl)
    assert small_parameters == 124_439_808
    assert xl_parameters == 1_557_611_200
    assert small_bytes == small_parameters * 4
    assert 1.5e9 < xl_parameters < 1.6e9
    print(
        "chapter04: counts passed "
        f"(small={small_parameters:,}, XL={xl_parameters:,}, "
        f"XL fp32={xl_bytes / 2**30:.2f} GiB)"
    )


if __name__ == "__main__":
    main()
