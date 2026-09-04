# LLM from Scratch Puzzles

Twenty independently written PyTorch exercises accompany chapters 2-7 of
Sebastian Raschka's *Build a Large Language Model (From Scratch)*. The primary
learning path is six explanation-rich Jupyter notebooks. Each notebook derives
the key equations, traces tensor shapes, walks through a synthetic example,
calls out common mistakes, and ends each exercise with an executable check.

The notebooks use deterministic toy data and tiny models so they run on CPU.
They are concept practice, not copies of the book's prose, figures, data, or
implementations. Read the linked chapter before or alongside each notebook for
the complete book context.

## Prerequisites

- Python 3.10+
- PyTorch 2.2–2.x

For a CPU-only environment:

```bash
python3 -m venv llm-from-scratch-puzzle/.venv
llm-from-scratch-puzzle/.venv/bin/python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  -r llm-from-scratch-puzzle/requirements.txt
```

No exercise downloads a corpus or model.

## Notebook and exercise map

| Notebook | Exercises |
| --- | --- |
| [`chapter02.ipynb`](chapter02_text_data/chapter02.ipynb) | 2.1 reconstruct split byte-token pieces; 2.2 inspect sliding-window batch shape and stride |
| [`chapter03.ipynb`](chapter03_attention/chapter03.ipynb) | Build simple, causal Q/K/V, and efficient multi-head attention; then map exercises 3.1-3.3 |
| [`chapter04.ipynb`](chapter04_gpt_architecture/chapter04.ipynb) | 4.1 compare FFN and attention parameter counts; 4.2 estimate GPT-2 variant parameters and weights-only memory |
| [`chapter05.ipynb`](chapter05_pretraining/chapter05.ipynb) | 5.1 temperature sampling; 5.2 decoding profiles; 5.3 deterministic generation; 5.4 checkpoint resume; 5.5 split loss; 5.6 config selection |
| [`chapter06.ipynb`](chapter06_classification/chapter06.ipynb) | 6.1 padding and masks; 6.2 selective unfreezing; 6.3 first-token versus last-valid-token representations |
| [`chapter07.ipynb`](chapter07_instruction_finetuning/chapter07.ipynb) | 7.1 prompt formats; 7.2 label masking; 7.3 memory planning; 7.4 LoRA |

Each chapter also retains a script-friendly `puzzle.py`, compact `solution.py`,
and focused `README.md`. The notebook and Python puzzle expose the same 20
named book-exercise objectives. Chapter 03 adds three implementation
checkpoints so attention is built before its exercise-specific variations.

## Recommended workflow

1. Open a chapter notebook and read its linked source chapter.
2. Predict shapes and results before running any code.
3. Replace one TODO and run only that exercise's check cell.
4. Answer the synthesis questions in your own words.
5. Compare with `solution.py` only after completing the attempt.

## Commands

Run from the repository root:

```bash
# Start Jupyter, then open the six chapter notebooks listed above.
llm-from-scratch-puzzle/.venv/bin/python -m jupyter lab

# Validate notebook structure and every reference solution.
python3 llm-from-scratch-puzzle/scripts/check_notebooks.py
llm-from-scratch-puzzle/.venv/bin/python \
  llm-from-scratch-puzzle/scripts/check_solutions.py
```

`BLANKS.md` is the greppable implementation checklist. Regenerate notebooks
after intentionally changing their shared curriculum structure:

```bash
python3 llm-from-scratch-puzzle/scripts/generate_notebooks.py
```

## Sources, attribution, and scope

Use the book/source for the complete exercise statements and surrounding
explanations:

- Chinese chapters: [2](https://github.com/skindhu/Build-A-Large-Language-Model-CN/blob/main/cn-Book/2.%E5%A4%84%E7%90%86%E6%96%87%E6%9C%AC%E6%95%B0%E6%8D%AE.md),
  [3](https://github.com/skindhu/Build-A-Large-Language-Model-CN/blob/main/cn-Book/3.%E5%AE%9E%E7%8E%B0%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6.md),
  [4](https://github.com/skindhu/Build-A-Large-Language-Model-CN/blob/main/cn-Book/4.%E4%BB%8E%E9%9B%B6%E5%BC%80%E5%A7%8B%E5%AE%9E%E7%8E%B0%E4%B8%80%E4%B8%AA%E7%94%A8%E4%BA%8E%E6%96%87%E6%9C%AC%E7%94%9F%E6%88%90%E7%9A%84%20GPT%20%E6%A8%A1%E5%9E%8B.md),
  [5](https://github.com/skindhu/Build-A-Large-Language-Model-CN/blob/main/cn-Book/5.%E5%9C%A8%E6%97%A0%E6%A0%87%E8%AE%B0%E6%95%B0%E6%8D%AE%E9%9B%86%E4%B8%8A%E8%BF%9B%E8%A1%8C%E9%A2%84%E8%AE%AD%E7%BB%83.md),
  [6](https://github.com/skindhu/Build-A-Large-Language-Model-CN/blob/main/cn-Book/6.%E7%94%A8%E4%BA%8E%E5%88%86%E7%B1%BB%E4%BB%BB%E5%8A%A1%E7%9A%84%E5%BE%AE%E8%B0%83.md),
  and [7](https://github.com/skindhu/Build-A-Large-Language-Model-CN/blob/main/cn-Book/7.%E6%8C%87%E4%BB%A4%E9%81%B5%E5%BE%AA%E5%BE%AE%E8%B0%83.md)
- Canonical MIT-licensed companion code:
  [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)

All curriculum code and explanations here are original. They do not adapt
upstream solution code or reproduce book text, so no upstream code or book
content is bundled. Numerical demonstrations are toy calculations; they make
no claim about full-training quality or membership of any real dataset.
