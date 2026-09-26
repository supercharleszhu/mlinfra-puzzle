# Remote Pod Guide: Correctness, Benchmark, and Profiling

This guide shows how to run the GEMM challenge on a remote GPU pod through `kubectl`.

Current dev pod, subject to change:

```bash
export GEMM_NS=kk-flyte-adhoc
export GEMM_POD=a9lfvz8vvnrm4vpb86dg-n0-0
export GEMM_REMOTE_DIR=/tmp/gpu-programming-puzzle
```

If the pod changes, update only `GEMM_POD` and `GEMM_NS`.

For the CuTe tutorial chapter:

```bash
# Generic/Ampere-compatible and SM90 Hopper lessons on the provided H100 pod:
cuda-cpp/scripts/upload_run_advanced_day.sh --day 20 --solution
cuda-cpp/scripts/upload_run_advanced_day.sh --day 22 --solution
cuda-cpp/scripts/upload_run_advanced_day.sh --day 23 --solution
cuda-cpp/scripts/upload_run_advanced_day.sh --day 24 --solution

# SM100 Blackwell lessons require a separate B200/SM100 pod:
cuda-cpp/scripts/upload_run_advanced_day.sh --day 25 --solution --pod "$B200_POD"
cuda-cpp/scripts/upload_run_advanced_day.sh --day 28 --solution --pod "$B200_POD"
```

The first tutorial run verifies the approved, pinned NVIDIA CUTLASS v4.6.1
checkout, then uploads its CuTe headers and tutorial helper sources. The
selected local challenge or solution `.cu` is compiled directly with NVCC and
cached by source hash. No plugin, internet script, or CuTe DSL package is
installed or executed. Pass `--dry-run` to check blanks and source selection
without compiling.

## 1. Check pod and GPU access

```bash
kubectl -n "$GEMM_NS" get pod "$GEMM_POD" -o wide
kubectl -n "$GEMM_NS" exec "$GEMM_POD" -- nvidia-smi
kubectl -n "$GEMM_NS" exec "$GEMM_POD" -- bash -lc 'python3 --version && nvcc --version || true'
```

If the pod has multiple containers, first find the container names:

```bash
kubectl -n "$GEMM_NS" get pod "$GEMM_POD" \
  -o jsonpath='{range .spec.containers[*]}{.name}{"\n"}{end}'
```

Then add `-c <container-name>` to every `kubectl exec` and `kubectl cp` command below.

## 2. Copy the standalone challenge to the pod

Preferred executable helper:

```bash
cd gpu-programming-puzzle
cuda-cpp/scripts/upload_to_pod.sh
cuda-cpp/scripts/upload_to_pod.sh --with-cutlass   # needed for solution days 13-19
```

From the `mlinfra-puzzle` repo root:

```bash
kubectl -n "$GEMM_NS" exec "$GEMM_POD" -- rm -rf "$GEMM_REMOTE_DIR"
kubectl -n "$GEMM_NS" cp ./gpu-programming-puzzle "$GEMM_POD:$GEMM_REMOTE_DIR"
```

`kubectl cp` requires `tar` inside the container. If it fails, use an explicit tar stream:

```bash
tar -C . -cf - gpu-programming-puzzle | \
  kubectl -n "$GEMM_NS" exec -i "$GEMM_POD" -- \
  bash -lc 'rm -rf /tmp/gpu-programming-puzzle && tar -C /tmp -xf -'
```

## 3. Confirm files and TODOs on the pod

```bash
kubectl -n "$GEMM_NS" exec "$GEMM_POD" -- bash -lc "
  cd $GEMM_REMOTE_DIR &&
  find cuda-cpp/challenge -maxdepth 1 -type f | sort &&
  grep -R \"GEMM_TODO\" -n cuda-cpp/challenge | head -40
"
```

The kernels are intentionally incorrect until you replace the `GEMM_TODO_*` placeholders.

## 4. Solution correctness and benchmark run

Preferred executable helper:

```bash
cd gpu-programming-puzzle
cuda-cpp/scripts/run_remote_day.sh --day 1
cuda-cpp/scripts/run_remote_day.sh --day 9 --sizes "128 256" --iters 50
cuda-cpp/scripts/run_remote_day.sh --day 14 --include-cutlass
cuda-cpp/scripts/run_remote_day.sh --all
```

Or upload and run in one command:

```bash
cuda-cpp/scripts/run_remote_matrix.sh --day 1
cuda-cpp/scripts/run_remote_matrix.sh --day 14 --with-cutlass
```

Use the complete solution first to confirm the pod can compile and run kernels:

```bash
kubectl -n "$GEMM_NS" exec "$GEMM_POD" -- bash -lc "
  cd $GEMM_REMOTE_DIR &&
  python3 cuda-cpp/solutions/python/benchmark_solution.py \
    --sizes 128 \
    --dtype float32 \
    --warmup 1 \
    --iters 2
"
```

Expected behavior:

- The first run compiles a PyTorch CUDA extension, so it can take a few minutes.
- The table should include `torch`, `01_naive`, `02_coalesced`, ..., `07_warptiling`.
- The `max_error` column should be near zero for FP32 kernels.

For Tensor Core solution kernels:

```bash
kubectl -n "$GEMM_NS" exec "$GEMM_POD" -- bash -lc "
  cd $GEMM_REMOTE_DIR &&
  python3 cuda-cpp/solutions/python/benchmark_solution.py \
    --sizes 128 256 \
    --dtype float16 \
    --warmup 5 \
    --iters 20
"
```

To benchmark a single solution kernel:

```bash
kubectl -n "$GEMM_NS" exec "$GEMM_POD" -- bash -lc "
  cd $GEMM_REMOTE_DIR &&
  python3 cuda-cpp/solutions/python/benchmark_solution.py \
    --sizes 1024 \
    --dtype float32 \
    --kernels 06_vectorize \
    --warmup 20 \
    --iters 100
"
```

CUTLASS solution files 13-19 require CUTLASS headers. If the pod has a full LeetCUDA checkout with `./cutlass`, run from that checkout or set `CUTLASS_DIR`, then add `--include-cutlass`. Days 13-19 additionally require Hopper/SM90+ and build with `sm_90a`.

## 5. Exercise correctness run

After filling a file and wiring the Python extension loader in `python/benchmark_gemm_challenge.py`, run a small correctness pass first:

```bash
kubectl -n "$GEMM_NS" exec "$GEMM_POD" -- bash -lc "
  cd $GEMM_REMOTE_DIR &&
  python3 python/benchmark_gemm_challenge.py \
    --sizes 128 256 \
    --dtype float32 \
    --warmup 3 \
    --iters 5
"
```

Use the `max error` column as the first gate. Suggested tolerances:

| dtype | Initial tolerance |
| --- | ---: |
| FP32 | `1e-3` to `1e-4` for small sizes |
| FP16 | `1e-2` to `1e-1` depending on accumulation/output |
| BF16 | `1e0` is a reasonable first smoke-test tolerance for larger Hopper BF16 runs |

If correctness fails, do not benchmark yet. Debug the first size that fails.

## 6. Exercise benchmark run

Once correctness passes, increase sizes and iterations:

```bash
kubectl -n "$GEMM_NS" exec "$GEMM_POD" -- bash -lc "
  cd $GEMM_REMOTE_DIR &&
  python3 python/benchmark_gemm_challenge.py \
    --sizes 128 256 512 1024 2048 4096 \
    --dtype float32 \
    --warmup 20 \
    --iters 100 | tee /tmp/gemm_benchmark.txt
"
```

Copy the benchmark table back:

```bash
kubectl -n "$GEMM_NS" cp "$GEMM_POD:/tmp/gemm_benchmark.txt" ./gemm_benchmark.txt
```

For Tensor Core days, switch dtype:

```bash
kubectl -n "$GEMM_NS" exec "$GEMM_POD" -- bash -lc "
  cd $GEMM_REMOTE_DIR &&
  python3 python/benchmark_gemm_challenge.py \
    --sizes 512 1024 2048 4096 \
    --dtype float16 \
    --warmup 20 \
    --iters 100
"
```

## 7. Nsight Compute profile

Profile one kernel/size at a time. Start with a small targeted run to keep reports readable:

```bash
kubectl -n "$GEMM_NS" exec "$GEMM_POD" -- bash -lc "
  cd $GEMM_REMOTE_DIR &&
  ncu --set full \
      --target-processes all \
      --force-overwrite \
      -o /tmp/gemm_day_profile \
      python3 python/benchmark_gemm_challenge.py \
        --sizes 1024 \
        --dtype float32 \
        --warmup 5 \
        --iters 10
"
```

For profiling a known-good solution kernel, use:

```bash
kubectl -n "$GEMM_NS" exec "$GEMM_POD" -- bash -lc "
  cd $GEMM_REMOTE_DIR &&
  ncu --set roofline \
      --target-processes all \
      --force-overwrite \
      -o /tmp/gemm_solution_profile \
      python3 cuda-cpp/solutions/python/benchmark_solution.py \
        --sizes 128 \
        --dtype float32 \
        --kernels 01_naive \
        --warmup 1 \
        --iters 1
"
```

Copy the Nsight report back:

```bash
kubectl -n "$GEMM_NS" cp "$GEMM_POD:/tmp/gemm_day_profile.ncu-rep" ./gemm_day_profile.ncu-rep
kubectl -n "$GEMM_NS" cp "$GEMM_POD:/tmp/gemm_solution_profile.ncu-rep" ./gemm_solution_profile.ncu-rep
```

Useful metrics to inspect after each day:

| Day | What to look for |
| --- | --- |
| 1-2 | Global memory load/store coalescing and memory throughput. |
| 3 | Shared-memory load/store activity and `__syncthreads()` overhead. |
| 4-7 | Reduced shared-memory pressure, register use, occupancy, and stall reasons. |
| 8-12 | Tensor Core instruction presence, e.g. `HMMA`/`MMA`, plus pipeline stalls. |
| 13-14 | CUTLASS tile shape, stages, swizzle/scheduler effects, and achieved TFLOPS. |
| 13-19 | Handwritten Hopper TMA/WGMMA and CUTLASS Hopper schedule choices, `sm_90a` compile behavior, and achieved TFLOPS on H100. |

## 8. Interactive debugging shell

```bash
kubectl -n "$GEMM_NS" exec -it "$GEMM_POD" -- bash
cd /tmp/gpu-programming-puzzle
```

Inside the pod, common checks:

```bash
nvidia-smi
python3 - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.get_device_name())
print(torch.cuda.get_device_capability())
PY
grep -R "GEMM_TODO" -n cuda
```

## 9. Cleanup

```bash
kubectl -n "$GEMM_NS" exec "$GEMM_POD" -- bash -lc "
  rm -rf $GEMM_REMOTE_DIR /tmp/gemm_benchmark.txt \
    /tmp/gemm_day_profile.ncu-rep /tmp/gemm_solution_profile.ncu-rep
"
```
