# Day 07 - Asynchronous Pipelines

**Goal:** replace CTA-wide lockstep barriers with a producer/consumer ring
buffer managed by `PipelineAsync`.

Start with [`day07.ipynb`](day07.ipynb). It develops the synchronized baseline,
the four stage-state transitions, circular-buffer reuse, shared-storage cost,
`producer.tail()`, and pipeline debugging before presenting the TODO scaffold.
It follows official Notebook
[`09_async_pipeline.ipynb`](../notebooks/09_async_pipeline.ipynb); use the
vendored notebook for the original NVIDIA walkthrough.

Two warps share `N` staging slots:

1. The producer calls `acquire_and_advance()`, writes one slot, then `commit()`.
2. The consumer calls `wait_and_advance()`, reads that slot, then `release()`.
3. Pipeline phase bits let each slot be reused safely around the ring.
4. `producer.tail()` prevents the producer warp from retiring before all
   expected consumer arrivals complete.

Fill the five TODOs in `puzzle.py`. The expected output is the sequence
`0..length-1`, proving that no value was overwritten before consumption.

```bash
python puzzle.py --length 16 --stages 3
python solution.py --length 16 --stages 3
```

This lesson uses only H100-supported asynchronous barriers and runs on SM80+.
