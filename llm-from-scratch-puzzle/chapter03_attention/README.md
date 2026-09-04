# Chapter 03 - Attention

Attention implementations become manageable when every matrix layout and
tensor axis has a precise meaning. This chapter now starts with a complete,
independently written implementation lab that follows the conceptual
progression of the Chinese chapter:

1. use input vectors directly as Q, K, and V;
2. add learned projections and a causal mask;
3. split projected channels into parallel heads and merge them again.

The lab traces `[B,T,D] -> [B,H,T,Dh] -> [B,H,T,T]`, checks that attention
rows normalize to one, and proves causality by changing future inputs while
requiring earlier outputs to remain unchanged.

| Implementation checkpoint | Core result |
| --- | --- |
| Simple self-attention | Scaled scores, row softmax, weighted values |
| Causal Q/K/V attention | Learned projections plus future-token mask |
| Efficient multi-head attention | Parallel heads and output projection |

| Exercise | Concept | Completion check |
| --- | --- | --- |
| 3.1 | Raw projection versus `nn.Linear` storage | Numerical outputs match |
| 3.2 | Concatenate independent heads | Batch/sequence stay fixed |
| 3.3 | Split model width across heads | 768 / 12 gives width 64 |

Start with [`chapter03.ipynb`](chapter03.ipynb), complete the three
`ATTENTION_TODO` checkpoints, then solve the three mapped book exercises. Use
`puzzle.py` for a script-only attempt and `solution.py` only after completing
the implementation. Consult the Chinese [chapter
3](https://github.com/skindhu/Build-A-Large-Language-Model-CN/blob/main/cn-Book/3.%E5%AE%9E%E7%8E%B0%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6.md)
and [MIT companion repository](https://github.com/rasbt/LLMs-from-scratch) for
the full exercise statements. This implementation is an independent toy.
