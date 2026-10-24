# LLM from Scratch Puzzles

This track starts with a complete CS336-style Transformer implementation, then
moves to classification and instruction fine-tuning:

| Chapter | Notebook | Focus |
| --- | --- | --- |
| 01 | [`chapter01.ipynb`](chapter01_cs336_transformer_e2e/chapter01.ipynb) | Byte-level BPE, Transformer LM, stable cross entropy, and AdamW |
| 02 | [`chapter02.ipynb`](chapter02_classification/chapter02.ipynb) | Padding, selective unfreezing, and sequence representations |
| 03 | [`chapter03.ipynb`](chapter03_instruction_finetuning/chapter03.ipynb) | Prompt formatting, label masking, memory planning, and LoRA |

Chapter 01 is a completed end-to-end reference with executable checks.
Chapters 02 and 03 remain fill-in-the-blank exercises with compact reference
solutions. The former Chapters 02–05 were removed because their tokenization,
attention, architecture, and training material is now consolidated in the
CS336 chapter.

## Local setup

```bash
python3 -m venv llm-from-scratch-puzzle/.venv
llm-from-scratch-puzzle/.venv/bin/python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu "torch>=2.2,<3"
llm-from-scratch-puzzle/.venv/bin/python -m pip install \
  -r llm-from-scratch-puzzle/requirements.txt
```

All checks use synthetic data and run on CPU without downloading a corpus or
model.

## Commands

Run from the repository root:

```bash
# Execute every compact reference solution.
llm-from-scratch-puzzle/.venv/bin/python \
  llm-from-scratch-puzzle/scripts/check_solutions.py

# Validate notebook structure and embedded Python.
python3 llm-from-scratch-puzzle/scripts/check_notebooks.py

# Execute the complete CS336 notebook.
llm-from-scratch-puzzle/.venv/bin/jupyter nbconvert \
  --to notebook --execute \
  llm-from-scratch-puzzle/chapter01_cs336_transformer_e2e/chapter01.ipynb \
  --output cs336-chapter01-executed.ipynb --output-dir /tmp
```

Regenerate the CS336 Chapter 01 notebook after changing its solution source:

```bash
python3 llm-from-scratch-puzzle/scripts/generate_notebooks.py
```

## Sources and scope

Chapter 01 follows the public interfaces and topics from
[Stanford CS336](https://stanford-cs336.github.io/) using independently written
code and explanations. Chapters 02–03 are original practice companions for
Sebastian Raschka's *Build a Large Language Model (From Scratch)* and link to
their source context from each chapter README.
