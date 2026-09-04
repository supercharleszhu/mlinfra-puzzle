# Chapter 07 - Instruction fine-tuning

Instruction fine-tuning serializes structured examples, chooses which tokens
contribute to loss, and balances adaptation capacity against memory.

| Exercise | Concept |
| --- | --- |
| 7.1 | Two independently worded prompt layouts |
| 7.2 | Prompt-label masking with `ignore_index` |
| 7.3 | Microbatch and accumulation planning |
| 7.4 | Low-rank adaptation of a frozen linear layer |

Start with [`chapter07.ipynb`](chapter07.ipynb). It derives the LoRA update
$BA$, traces factor shapes, and states exactly what the toy memory heuristic
omits. Obtain the complete exercises from Chinese [chapter
7](https://github.com/skindhu/Build-A-Large-Language-Model-CN/blob/main/cn-Book/7.%E6%8C%87%E4%BB%A4%E9%81%B5%E5%BE%AA%E5%BE%AE%E8%B0%83.md)
and use the [canonical MIT companion](https://github.com/rasbt/LLMs-from-scratch)
as the code reference. No upstream prose, templates, data, or code is
reproduced.
