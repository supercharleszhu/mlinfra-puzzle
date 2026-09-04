# Chapter 06 - Classification

Classification reuses a language model's sequence representations but must
define padding semantics, a trainability policy, and a pooling position.

| Exercise | Concept | Completion check |
| --- | --- | --- |
| 6.1 | Right-padding and validity masks | Tokens and mask have `[B,T]` shape |
| 6.2 | Head-only versus full tuning | Trainable counts match policy |
| 6.3 | First versus last-valid state | Each row gathers the intended token |

Start with [`chapter06.ipynb`](chapter06.ipynb). It traces variable-length
gather indices and explains why causal position zero has limited context. For
the complete source exercises, use Chinese [chapter
6](https://github.com/skindhu/Build-A-Large-Language-Model-CN/blob/main/cn-Book/6.%E7%94%A8%E4%BA%8E%E5%88%86%E7%B1%BB%E4%BB%BB%E5%8A%A1%E7%9A%84%E5%BE%AE%E8%B0%83.md)
and the [canonical MIT companion](https://github.com/rasbt/LLMs-from-scratch).
The local data and implementation are original, small demonstrations.
