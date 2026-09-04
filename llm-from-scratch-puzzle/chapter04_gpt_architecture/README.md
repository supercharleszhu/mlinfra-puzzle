# Chapter 04 - GPT architecture

Parameter arithmetic explains model structure before any memory is allocated.
This chapter derives the dominant attention and feed-forward terms, then builds
a transparent weights-only estimate for GPT-2-style variants.

| Exercise | Concept | Completion check |
| --- | --- | --- |
| 4.1 | Attention versus 4x FFN parameter counts | FFN is approximately 2x |
| 4.2 | Full tied-embedding estimate | Small and XL ranges are plausible |

Start with [`chapter04.ipynb`](chapter04.ipynb). It distinguishes weights from
activations and decimal parameters from binary memory units. The Chinese [chapter
4](https://github.com/skindhu/Build-A-Large-Language-Model-CN/blob/main/cn-Book/4.%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AE%9E%E7%8E%B0%E4%B8%80%E4%B8%AA%E7%94%A8%E4%BA%8E%E6%96%87%E6%9C%AC%E7%94%9F%E6%88%90%E7%9A%84%20GPT%20%E6%A8%A1%E5%9E%8B.md)
and [canonical code repo](https://github.com/rasbt/LLMs-from-scratch) provide the
full source exercises. Counts here are transparent estimates, not training
results.
