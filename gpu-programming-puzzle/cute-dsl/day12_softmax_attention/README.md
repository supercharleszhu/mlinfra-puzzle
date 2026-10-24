# Day 12 - Softmax Attention and Simplified FlashAttention-2

**Goal:** derive scaled dot-product attention, turn stable softmax into an
online recurrence, and implement the same one-pass forward algorithm in
Triton and CuTe DSL without materializing the full attention matrix.

This lesson follows the algorithm and implementation discussion in
[From FlashAttention to FlashAttention 2](https://supercharleszhu.github.io/posts/from-flashattention-to-flashattention-2/),
especially its explanations of Q-block ownership, delayed normalization, and
warp partitioning along Q rows.

## 1. What softmax attention computes

For one batch-head, let:

- `Q` have shape `(N_q, D)`;
- `K` and `V` have shape `(N_k, D)`;
- `D` be the head dimension.

Scaled dot-product attention is:

$$
S = \frac{QK^T}{\sqrt{D}},
\qquad
P_{ij} =
\frac{\exp(S_{ij})}
{\sum_{r=0}^{N_k-1}\exp(S_{ir})},
\qquad
O = PV.
$$

Each query row compares with every key row. Softmax converts those scores into
positive weights summing to one, and the output is a weighted combination of
the value rows.

The stable form subtracts each score row's maximum:

$$
m_i = \max_j S_{ij},
\qquad
P_{ij} =
\frac{\exp(S_{ij}-m_i)}
{\sum_r \exp(S_{ir}-m_i)}.
$$

Subtracting the maximum does not change the result because the same factor
$e^{-m_i}$ appears in every numerator and the denominator. It prevents large
positive scores from overflowing `exp`.

## 2. Why ordinary attention is memory-hungry

A direct implementation commonly creates:

```text
scores        = Q @ K.T       # N_q x N_k
probabilities = softmax(...)  # N_q x N_k
output        = probabilities @ V
```

The score/probability matrix is quadratic in sequence length. With
`N_q=N_k=N`, it contains `N^2` elements even though the final output contains
only `N*D`. Writing and rereading that matrix from HBM is the main problem
FlashAttention removes.

FlashAttention tiles K/V and keeps only a small score tile on chip:

```text
one Q row/block
    for each K/V tile:
        score tile = Q tile @ K tile.T
        update online softmax state
        update output accumulator with V tile
    normalize once and store O
```

The complete `N x N` matrix is never stored in HBM.

## 3. One-pass softmax attention equation

For one query row, maintain three values after processing some K/V tiles:

$$
m = \max_{\text{processed }j} s_j,
$$

$$
\ell = \sum_{\text{processed }j} e^{s_j-m},
$$

$$
U = \sum_{\text{processed }j} e^{s_j-m}v_j.
$$

`U` is deliberately **unnormalized**. The current attention output would be
`U / l`, but FlashAttention-2 delays that division until all tiles have been
processed.

For a new score tile $S_t$ and matching value tile $V_t$, compute:

$$
\widetilde{m}_t = \operatorname{rowmax}(S_t),
\qquad
m' = \max(m, \widetilde{m}_t),
$$

$$
\alpha = e^{m-m'},
\qquad
P_t = e^{S_t-m'},
$$

$$
\ell' = \alpha\ell + \operatorname{rowsum}(P_t),
$$

$$
U' = \alpha U + P_tV_t.
$$

After the final tile:

$$
O = \frac{U}{\ell}.
$$

These updates preserve the invariant:

$$
\ell' = \sum_{\text{all processed }j}e^{s_j-m'},
\qquad
U' = \sum_{\text{all processed }j}e^{s_j-m'}v_j.
$$

The factor `alpha` rescales all old terms from the old maximum to the new
maximum. This is why the algorithm remains numerically stable even when a
later tile contains a much larger score.

The normalized FA1 form updates `O` after each tile. The equations are
equivalent, but the unnormalized FA2 form avoids repeatedly dividing and
rescaling every output element:

$$
O' =
\frac{\alpha\ell}{\ell'}O
+ \frac{P_tV_t}{\ell'}.
$$

## 4. What FlashAttention-2 changes

The referenced deep dive highlights three changes from FlashAttention-1:

1. **Delay output normalization.** Accumulate `U` and divide once at the end,
   reducing non-matmul rescaling work.
2. **Assign each Q block to one CTA.** The CTA loops over all K/V blocks and
   owns its output, avoiding forward atomics.
3. **Partition warps only along Q rows.** Every warp sees the full K-column
   extent for its Q rows, so softmax does not require cross-warp exchange.

A production FA2 kernel also uses tensor-core MMA, swizzled shared memory,
asynchronous K/V pipelines, FP16/BF16 inputs, causal/edge masks, and a
specialized backward pass. This day isolates the forward algorithm before
those mechanisms.

## 5. Correctness and benchmark contract

The two teaching paths intentionally use different API shapes:

```text
Triton Q/K/V/O: [batch_heads, sequence_length, head_dim]
CuTe   Q/O:     [M, head_dim]
CuTe   K/V:     [N, head_dim]
```

The Triton path uses FP16 Q/K/V/output with FP32 online-softmax and output
accumulation. The CuTe path keeps Q/K/V/output storage in FP32, but
`MmaTF32Op` converts matrix-operand fragments to TF32 and executes both
`QK^T` and `PV` on tensor cores with FP32 accumulation.

Removing the batch-head dimension keeps the CuTe ownership lesson focused:
grid `x` enumerates 64-row Q tiles, and a four-warp CTA owns each tile. Each
warp owns 16 Q rows and traverses the complete K-column extent for those
rows. There is no grid `y` batch-head coordinate or batch-head term in global
indexing.

The reference explicitly materializes scores only for testing and always
computes it in FP32 before converting to the input dtype:

```python
scale = 1.0 / math.sqrt(head_dim)
scores = q.float() @ k.float().transpose(-1, -2) * scale
reference = (
    torch.softmax(scores, dim=-1) @ v.float()
).to(q.dtype)
```

Each implementation is immediately followed by its own correctness and
benchmark cells. Correctness includes sequence lengths 17, 33, and 64 so edge
tiles and partial Q blocks are exercised. Each benchmark:

- compiles and launches the implementation before timing;
- checks its output against the same PyTorch reference;
- uses `triton.testing.do_bench`;
- reports forward matmul FLOPs:

$$
\text{FLOPs} \approx 4 B_h N^2 D.
$$

The count includes `QK^T` and `PV` but omits exponentials, reductions, and
rescaling, so the reported TFLOP/s is an attention convention rather than a
complete instruction count.

## 6. Implementations

Both implementations apply the same `(m, l, U)` invariant to matrix tiles.
They differ in storage dtype, tile shape, and how their tensor-core ownership
is expressed.

### 6.1 Triton implementation

This implementation follows the forward path in
[Triton Tutorial 06: Fused Attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html),
with a deliberately narrow feature set:

- FlashAttention-2 forward only;
- non-causal attention;
- FP16 Q/K/V and FP16 output;
- FP32 online-softmax and output accumulation;
- four warps, with no warp specialization;
- no FP8, tensor descriptors, autotuning, or backward kernel.

One program owns a block of 64 Q rows rather than one row. It keeps the Q tile
resident and processes K/V blocks of 32 or 64 rows:

```text
Q_tile: BLOCK_M x D
K_tile: BLOCK_N x D
V_tile: BLOCK_N x D

scores = Q_tile @ K_tile.T / sqrt(D)
m_new = max(m, rowmax(scores))
alpha = exp2(m - m_new)
p = exp2(scores - m_new[:, None])
l = alpha * l + rowsum(p)
U = alpha[:, None] * U + p.to(fp16) @ V_tile
m = m_new
```

`tl.dot` lowers both matrix products to tensor-core operations. `U` is a
`BLOCK_M x D` FP32 accumulator. The `BLOCK_M x BLOCK_N` score/probability tile
stays on chip, so the kernel still never creates an `N x N` output.

The kernel uses base-2 exponentiation like the official tutorial. Multiplying
the softmax scale by `log2(e)` preserves the same mathematical softmax while
mapping to `exp2`.

The notebook places `test_triton_attention()` and
`triton_benchmark_suite()` directly after the implementation.

### 6.2 CuTe implementation

The referenced FA2 head-dimension-64 SM80 kernel uses a `128 x 128` CTA tile,
four warps, and a warp layout stacked only along M:

```cpp
Flash_fwd_kernel_traits<64, 128, 128, 4, ...>
TiledMMA<Atom, Layout<Shape<_4, _1, _1>>, Tile<_64, _16, _16>>
```

The CuTe DSL lesson preserves that defining scheduling rule while using a
`64 x 64` tile:

```text
four-warp CTA
Q/O API shape: [M, D], K/V API shape: [N, D]
├── Q tile:      64 x D, loaded once
├── K/V tile:   64 x D, shared by all four warps
├── warp 0:     Q rows  0..15, all 64 K columns
├── warp 1:     Q rows 16..31, all 64 K columns
├── warp 2:     Q rows 32..47, all 64 K columns
├── warp 3:     Q rows 48..63, all 64 K columns
├── scores:     64 x 64 = TF32 TiledMma(Q, K^T)
└── U registers: 64 x D = persistent TF32 MMA accumulator
```

FA2 can use a larger tile because BF16 operands consume half as much shared
memory and its score/probability fragments remain in registers. This
intermediate TF32 lesson now keeps scores and online-softmax state in
registers, but still converts the final probability fragment through one FP32
shared-memory tile before `PV`. `64 x 64` keeps that representation practical
while adding the production kernel's four-warp row schedule.

The implementation is functional rather than stateful:

```python
@cute.kernel
def simplified_flash_attention2_kernel(...):
    ...


@cute.jit
def simplified_flash_attention2(m_q, m_k, m_v, m_o):
    query_length, head_dim = m_q.shape
    key_length, _ = m_k.shape
    ...
```

There is no attention class or `self`; compile-time dimensions come directly
from the 2D input tensor shapes. The launcher selects an `m16n8k4` TF32 atom
for `D=4 mod 8` and an `m16n8k8` atom otherwise.

Q/K/V remain FP32 in global and shared memory. Copying their FP32 partitions
into the fragments created by:

```python
mma_op = MmaTF32Op((16, 8, 8))
tiled_mma = cute.make_tiled_mma(
    mma_op, atom_layout_mnk=(4, 1, 1)
)
```

performs the TF32 operand conversion required by
`mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32`. Both matrix products
therefore use tensor cores while their accumulators remain FP32.

CuTe DSL 4.4.2 does not export `cute.nvgpu.warp.MmaTF32Op`. The lesson
includes a small compatibility operation that constructs the same SM80 MMA
atom through `MmaAtomSM80Type`; newer releases use the public class:

```python
MmaTF32Op = getattr(
    cute.nvgpu.warp, "MmaTF32Op", MmaTF32CompatOp
)
```

For CTA thread `t`, the explicit hierarchy is:

```python
warp = t // 32
lane = t % 32
```

`atom_layout_mnk=(4, 1, 1)` places the four warp MMA atoms along M and none
along N or K. Warp `w` therefore owns Q rows `16*w : 16*(w+1)` and sees every
K column for those rows. This is the essential FA2 rule: softmax rows never
span warps, so no cross-warp reduction is required.

After `QK^T`, the FP32 score accumulator remains in registers. The raw
`m16n8` C fragment gives each lane two logical rows and two columns from each
8-column MMA atom. Across the eight N atoms in a 64-key tile, one lane owns
16 scores for each of its two rows:

```python
column = n_block * 8 + (lane % 4) * 2 + value
```

The four lanes sharing an MMA row reduce their local maxima and sums with
`quad_reduce_max` and `quad_reduce_sum`, matching FA2's `Allreduce<4>`. The
running maximum, denominator, and unnormalized output stay in registers. Each
K/V iteration updates:

$$
m' = \max(m, \operatorname{rowmax}(S_t)),
$$

$$
\ell' = e^{m-m'}\ell + \operatorname{rowsum}(e^{S_t-m'}),
$$

$$
U' = e^{m-m'}U
   + \operatorname{TF32MMA}(e^{S_t-m'}, V_t).
$$

The resulting probabilities are written once to FP32 shared memory and copied
into TF32 A fragments for `PV`. This is one remaining difference from FA2,
which converts and retiles `acc_s` directly into the `PV` A-register layout.
The persistent `U` tile stays in MMA C registers across the entire K/V loop:

```python
r_output.fill(0.0)
for each_kv_tile:
    r_output[row] *= previous_scale[row]
    r_output += tiled_mma(probabilities, values)
r_output[row] /= running_sum[row]
```

For `m16n8k8`, each lane owns four C values per output atom: two values from
row `lane // 4` and two from row `lane // 4 + 8`. The CTA row adds the
warp's offset:

```python
row = warp * 16 + lane // 4 + (accumulator // 2) * 8
```

That ownership determines which online-softmax scale applies to each output
register. The normalized register fragment is copied to shared memory only
once for the final coalesced output store. Partial Q and K/V tiles are
zero/mask padded, so sequence lengths need not be multiples of 64.

The notebook places `test_cute_attention()` and `cute_benchmark_suite()`
directly after the implementation.

## 7. Compare the implementations

Triton and CuTe both process 64 Q rows and up to 64 keys per update. Triton
uses FP16 tensor-core `tl.dot`; CuTe uses FP32 storage and TF32 `TiledMma`.
CuTe keeps scores, row reductions, and persistent output in registers. Only
the probability fragment still passes through shared memory before `PV`.
Neither implementation writes the full score or probability matrix to HBM.

Representative Triton B300 measurements retain 16 batch-heads:

| `BH x N x D` | Triton ms | Triton TFLOP/s |
|---|---:|---:|
| `16 x 256 x 64` | 0.0102 | 26.23 |
| `16 x 512 x 64` | 0.0123 | 87.34 |

The single-head comparison also measures the direct PyTorch expression:

```python
torch.softmax((q @ k.T) / math.sqrt(head_dim), dim=-1) @ v
```

FP32 matmul precision is set to `high`, allowing the same TF32 tensor-core
arithmetic category as the CuTe kernel. This implementation is mathematically
naive because it materializes the complete score and probability matrices,
but its two matrix multiplications use highly optimized cuBLAS kernels.

| `N x D` | Naive ms | Naive TFLOP/s | CuTe ms | CuTe TFLOP/s | CuTe speedup |
|---|---:|---:|---:|---:|---:|
| `256 x 64` | 0.0210 | 0.80 | 0.0408 | 0.41 | 0.52x |
| `512 x 64` | 0.0215 | 3.12 | 0.0719 | 0.93 | 0.30x |
| `4096 x 64` | 0.1054 | 40.75 | 0.4997 | 8.59 | 0.21x |

The four-warp `64 x 64` schedule plus register softmax is about 23-24% faster
than the preceding one-warp `16 x 16` version at the two smaller shapes.
Against the immediately preceding four-warp shared-score version, register
softmax improves latency by about 8% at `256 x 64`, 10% at `512 x 64`, and
11% at `4096 x 64`.

The teaching kernel is still about 1.9x slower at `256 x 64`, 3.3x slower at
`512 x 64`, and 4.7x slower at `4096 x 64`. A single head launches only four,
eight, or 64 CTAs respectively, so even the largest case does not fill all
148 B300 SMs in one wave. Production attention exposes batch and head
dimensions to the launch grid, and FA2 additionally retiles probabilities
directly in registers, pipelines asynchronous copies, and uses swizzled
lower-precision shared storage.

Run:

```bash
python day12_softmax_attention/solution.py
python day12_softmax_attention/solution.py --benchmark
```

## 8. Nsight Compute: why fused CuTe is slower

The benchmark establishes the application result:

```text
N=512, D=64
naive PyTorch: 0.0215 ms
four-warp CuTe: 0.0719 ms
```

The CuTe kernel is about `3.3x` slower even though it avoids materializing the
complete score and probability matrices. Nsight Compute explains why.

### 8.1 Reproduce the profile

`profile_target.py` compiles and warms up first, checks correctness, then uses
`cudaProfilerStart/Stop` around exactly one measured invocation. The naïve
region contains its four GPU kernels; the CuTe region contains one fused
kernel.

On the B300 pod:

```bash
cd /tmp/gpu-programming-puzzle-cute-dsl/day12_softmax_attention
PYTHON_BIN=../.venv/bin/python \
SEQUENCE_LENGTH=512 HEAD_DIM=64 \
bash collect_profiles.sh
```

The script creates raw CSV and `.ncu-rep` files under `reports/`. Open the
reports in the Nsight Compute UI for source and SASS inspection. The compact
checked-in snapshot is `profile_summary.csv`.

NCU replays kernels to collect counters, so its durations are perturbed.
Use the normal benchmark for final latency and NCU for relative evidence.
The profiled durations preserve the same result:

```text
naive stages total: 31.040 us
fused CuTe kernel: 110.496 us
ratio: 3.56x
```

### 8.2 Counter snapshot

The naïve expression launches four kernels:

| Path/stage | Grid blocks | Duration (NCU) | SM throughput | Tensor active | Registers/thread | Shared/block |
|---|---:|---:|---:|---:|---:|---:|
| CuTe fused | 8 | 110.496 us | 0.67% | 0.37% | 255 | 82.9 KB |
| Naïve `QK^T` | 64 | 7.392 us | 5.23% | 2.83% | 88 | 50.2 KB |
| Naïve scale | 256 | 5.024 us | 1.80% | 0% | 32 | 1.0 KB |
| Naïve softmax | 128 | 6.880 us | 6.53% | 0% | 48 | 1.0 KB |
| Naïve `PV` | 16 | 11.744 us | 1.63% | 0.89% | 69 | 227.3 KB |

The B300 has 148 SMs. For `M=512`, the fused kernel launches only:

```text
ceil(512 / 64) = 8 CTAs
```

At most eight SMs receive work in that wave, only 5.4% of the GPU. Each CTA
then serially processes eight 64-key tiles because online softmax makes the
K/V loop sequential. Production attention obtains enough parallel CTAs from
batch and head dimensions; this teaching API deliberately removed those
dimensions.

By comparison, the naïve stages expose grids of 64, 256, 128, and 16 blocks.
Although they launch separately, cuBLAS/CUTLASS and PyTorch spread each stage
over much more of the GPU.

### 8.3 Occupancy is constrained by the teaching representation

The CuTe kernel uses:

```text
255 registers/thread
82.944 KB shared memory/CTA
4 warps/CTA
```

On this B300, registers limit an SM to two resident CTAs. That is at most
eight resident warps per participating SM before other limits. NCU reports
only 6.24% active warps and 0.49% issue-active cycles.

Register softmax pushes the kernel to the 255-register ceiling and introduces
3,712 local-load sectors and 640 local-store sectors. Keeping row state in
shared memory removes those spills but is slower, so the checked-in version
accepts this small spill cost. The dominant limitation remains reduced
residency and almost no grid-level parallelism.

### 8.4 The remaining shared-memory path is still expensive

Register softmax removes the score store and reload. The kernel still stages
Q/K/V with simple layouts and writes probabilities to shared memory before
reloading them for `PV`. These unswizzled paths produce:

```text
shared-load bank conflicts:  261,341
shared-store bank conflicts: 281,600
total:                       542,941
```

The four naïve library kernels report only 140 shared-memory conflicts in
total. The CuTe stall distribution matches the shared-memory-heavy source:

| Stall reason | CuTe |
|---|---:|
| Short scoreboard, including shared/MIO dependencies | 32.86% |
| Long scoreboard, including L1TEX/global dependencies | 39.06% |
| Fixed-latency wait | 14.60% |
| CTA barrier | 1.33% |

The kernel is therefore not mainly waiting at `sync_threads()`. It is waiting
for data dependencies created by synchronous global/shared copies and the
remaining probability shared-memory path. NCU also reports zero `LDGSTS`
instructions, confirming that this version has no `cp.async` pipeline. By
contrast, the naïve `QK^T` GEMM executes 5,120 `LDGSTS` instructions.

### 8.5 Tensor cores are present but poorly utilized

The fused kernel executes tensor-core MMA for both matrix products, but the
GPU-wide tensor pipeline is active for only 0.37% of elapsed cycles. The
optimized naïve `QK^T` GEMM reaches 2.83% despite the small problem.

NCU also reports:

```text
CuTe tensor-pipe instructions:       32,768
naïve GEMM tensor-pipe instructions: 16,896
```

The teaching kernel builds both products from fine-grained `m16n8` warp MMA
atoms. The library GEMMs use tuned larger tiles; the `PV` kernel is an SM100
two-SM TMA kernel. The libraries perform the same mathematical products with
coarser instruction granularity, optimized epilogues, and architecture-aware
copy pipelines.

### 8.6 Fusion saves traffic, but traffic is not the bottleneck

Summing the NCU rows:

| Path | DRAM reads | L2 bytes |
|---|---:|---:|
| CuTe fused | 0.44 MB | 2.23 MB |
| Naïve four stages | 3.64 MB | 14.23 MB |

Fusion reduces L2 traffic by about `6.4x` and DRAM reads by about `8.3x`.
Nevertheless, the basic profile reports less than 3% DRAM throughput for
every naïve stage and 0.05% for CuTe. The `512 x 512` intermediates are small
enough to benefit heavily from B300 cache, so avoiding them does not offset
low launch parallelism and inefficient on-chip movement.

The evidence chain is:

```text
single-head API
  -> only 8 fused CTAs for M=512
  -> most of 148 SMs idle
  -> 255 registers + 83 KB shared memory constrain residency
  -> unswizzled Q/K/V/P paths create 543K shared conflicts
  -> scoreboard stalls dominate and tensor utilization stays below 1%
  -> tuned multi-kernel PyTorch is faster despite moving more bytes
```

## 9. Next optimization steps

Advance one mechanism at a time:

1. Implement FA2's `convert_layout_acc_Aregs` equivalent for TF32 CuTe DSL,
   converting and retiling register probabilities directly into the `PV` A
   operand without the remaining shared-memory tile.
2. Add FA2-style swizzled Q/K/V layouts to reduce the 543K shared bank
   conflicts.
3. Add 128-bit `cp.async` global-to-shared copies, then double-buffer K/V to
   overlap the next tile's loads with current-tile MMA.
4. Expand the four-warp tile from `64 x 64` toward FA2's `128 x 128` shape
   after removing the FP32 probability shared tile.
5. Restore batch/head launch dimensions so production-sized workloads expose
   enough CTAs to fill the GPU.
6. Add causal masks, logsumexp storage, and backward recomputation.
7. Treat TMA, warp specialization, and architecture-specific tensor-memory
   paths as separate advanced steps.
