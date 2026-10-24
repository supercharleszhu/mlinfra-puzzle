# Chapter 01 - CS336 Transformer end to end

This chapter turns the first Stanford CS336 assignment into one locally
executable walkthrough:

1. train a byte-level BPE vocabulary and tokenize Unicode text;
2. build `Linear`, `Embedding`, RMSNorm, SwiGLU, RoPE, causal multi-head
   attention, Transformer blocks, and a complete language model;
3. compute numerically stable cross entropy and update the model with a custom
   AdamW optimizer;
4. run a CPU smoke test covering tokenization, forward propagation, backward
   propagation, and an optimizer step.

Start with [`chapter01.ipynb`](chapter01.ipynb). The notebook and
[`solution.py`](solution.py) contain the same completed implementation; the
script is convenient for terminal validation.

The chapter is an independently written implementation based on the public
[Stanford CS336](https://stanford-cs336.github.io/) assignment interface. It
does not reproduce the assignment handout or upstream solution code.

Run locally from the repository root:

```bash
llm-from-scratch-puzzle/.venv/bin/python \
  llm-from-scratch-puzzle/chapter01_cs336_transformer_e2e/solution.py
```
