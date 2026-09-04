# Chapter 02 - Text data

Text becomes model input through a sequence of representations: Unicode text,
token byte pieces, integer IDs, and fixed-length training windows. This chapter
focuses on the boundaries where those representations are commonly confused.

| Exercise | Concept | Completion check |
| --- | --- | --- |
| 2.1 | Join byte-token pieces before UTF-8 decoding | Split multibyte character round-trips |
| 2.2 | Create shifted next-token windows | Shapes, targets, and stride agree |

Start with [`chapter02.ipynb`](chapter02.ipynb). It explains variable-width
UTF-8, derives the shifted-window indices, and includes toy checks. Use
`puzzle.py` for a script-only attempt and `solution.py` after finishing.

From the repository root:

```bash
python3 llm-from-scratch-puzzle/chapter02_text_data/puzzle.py
```

See the Chinese [chapter
2](https://github.com/skindhu/Build-A-Large-Language-Model-CN/blob/main/cn-Book/2.%E5%A4%84%E7%90%86%E6%96%87%E6%9C%AC%E6%95%B0%E6%8D%AE.md)
and [canonical companion](https://github.com/rasbt/LLMs-from-scratch) for the
full source exercises. The tasks and toy bytes here are original.
