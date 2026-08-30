#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Upload one GEMM hard-way challenge day to a Kubernetes GPU pod, then run correctness + benchmark.

Usage:
  scripts/upload_run_one_day.sh --day DAY [options]

Options:
  --day DAY           Day number: 1..19.
  --namespace NS      Kubernetes namespace. Default: $GEMM_NS or kk-flyte-adhoc
  --pod POD           Pod name. Default: $GEMM_POD or a9lfvz8vvnrm4vpb86dg-n0-0
  --remote-dir DIR    Default: $GEMM_REMOTE_DIR or /tmp/cutlass-puzzle
  --container NAME    Optional container name for multi-container pods.
  --sizes "S..."      Space-separated matrix sizes. Overrides mode defaults.
  --easy              Quick benchmark sizes through 1024. Default.
  --verbose           Full explore-gemm benchmark sweep through 8192.
  --dtype DTYPE       float32, float16, or bfloat16. Default depends on day.
  --warmup N          Warmup iterations. Default: 10
  --iters N           Benchmark iterations. Default: 100
  --runs N            Repeat each benchmark size N times. Default: 5
  --tolerance VALUE   Max-error correctness tolerance. Default depends on dtype.
  --arch ARCH         nvcc GPU architecture. Default: current GPU capability.
  --result-dir DIR    Local folder for successful run logs. Default: challenge ./result
  --with-cutlass      Upload CUTLASS headers for days 13-19.
  --cutlass-dir DIR   CUTLASS checkout. Default: $CUTLASS_DIR, ./cutlass, ../LeetCUDA/cutlass.
  --verbose-build     Show PyTorch extension build output.
  -h, --help          Show this help.

Examples:
  scripts/upload_run_one_day.sh --day 1
  scripts/upload_run_one_day.sh --day 9 --verbose
  scripts/upload_run_one_day.sh --day 9 --sizes "256 512 1024" --iters 50 --runs 5
  scripts/upload_run_one_day.sh --day 14 --with-cutlass --cutlass-dir ../LeetCUDA/cutlass
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
challenge_dir="$(cd "$script_dir/.." && pwd)"

namespace="${GEMM_NS:-kk-flyte-adhoc}"
pod="${GEMM_POD:-a9lfvz8vvnrm4vpb86dg-n0-0}"
remote_dir="${GEMM_REMOTE_DIR:-/tmp/cutlass-puzzle}"
container=""
day=""
sizes=""
dtype=""
warmup=10
iters=100
runs=5
tolerance=""
arch=""
result_dir="$challenge_dir/result"
with_cutlass=0
cutlass_dir="${CUTLASS_DIR:-}"
verbose_build=0
benchmark_mode="easy"
easy_sizes="64 96 128 256 512 768 1024"
explore_gemm_sizes="64 96 128 256 512 768 1024 1536 2048 3072 4096 8192"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --day)
      day="$2"
      shift 2
      ;;
    --namespace)
      namespace="$2"
      shift 2
      ;;
    --pod)
      pod="$2"
      shift 2
      ;;
    --remote-dir)
      remote_dir="$2"
      shift 2
      ;;
    --container)
      container="$2"
      shift 2
      ;;
    --sizes)
      sizes="$2"
      shift 2
      ;;
    --easy)
      benchmark_mode="easy"
      shift
      ;;
    --verbose)
      benchmark_mode="verbose"
      shift
      ;;
    --dtype)
      dtype="$2"
      shift 2
      ;;
    --warmup)
      warmup="$2"
      shift 2
      ;;
    --iters)
      iters="$2"
      shift 2
      ;;
    --runs)
      runs="$2"
      shift 2
      ;;
    --tolerance)
      tolerance="$2"
      shift 2
      ;;
    --arch)
      arch="$2"
      shift 2
      ;;
    --result-dir)
      result_dir="$2"
      shift 2
      ;;
    --with-cutlass)
      with_cutlass=1
      shift
      ;;
    --cutlass-dir)
      cutlass_dir="$2"
      shift 2
      ;;
    --verbose-build)
      verbose_build=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$day" ]]; then
  echo "Pass --day DAY" >&2
  usage >&2
  exit 2
fi

case "$day" in
  1) day_file="01_naive.cu" ;;
  2) day_file="02_kernel_global_mem_coalesce.cu" ;;
  3) day_file="03_kernel_shared_mem.cu" ;;
  4) day_file="04_kernel_blocktiling_1d.cu" ;;
  5) day_file="05_kernel_blocktiling_2d.cu" ;;
  6) day_file="06_kernel_vectorize.cu" ;;
  7) day_file="07_kernel_warptiling.cu" ;;
  8) day_file="08_kernel_warptiling_all_dtypes.cu" ;;
  9) day_file="09_kernel_tensorcore_naive.cu" ;;
  10) day_file="10_kernel_tensorcore_warptiled.cu" ;;
  11) day_file="11_kernel_tensorcore_double_buffered.cu" ;;
  12) day_file="12_kernel_tensorcore_async.cu" ;;
  13) day_file="13_kernel_fastcu_matmul2_manual_tma_wgmma.cu" ;;
  14) day_file="14_kernel_fastcu_matmul3_big_tile.cu" ;;
  15) day_file="15_kernel_fastcu_matmul4_warp_specialized.cu" ;;
  16) day_file="16_kernel_fastcu_matmul6_persistent.cu" ;;
  17) day_file="17_kernel_fastcu_matmul10_tma_store.cu" ;;
  18) day_file="18_kernel_fastcu_matmul12_final.cu" ;;
  19) day_file="19_kernel_cutlass3_hopper_tunable.cu" ;;
  *)
    echo "Unsupported day: $day" >&2
    exit 2
    ;;
esac

if [[ "$day" -ge 13 && "$with_cutlass" -ne 1 ]]; then
  echo "Day $day needs CUTLASS headers; pass --with-cutlass." >&2
  exit 2
fi

kubectl_args=(-n "$namespace")
if [[ -n "$container" ]]; then
  kubectl_args+=(-c "$container")
fi

echo "Checking pod: namespace=$namespace pod=$pod"
kubectl -n "$namespace" get pod "$pod" -o wide

echo "Uploading day $day ($day_file) and support files to $pod:$remote_dir"
kubectl "${kubectl_args[@]}" exec "$pod" -- bash -lc "
  set -euo pipefail
  mkdir -p '$remote_dir/cuda' '$remote_dir/python'
  rm -rf '$remote_dir/build'
"

upload_files=(
  "cuda/$day_file"
  cuda/challenge_todo.cuh
  cuda/gemm_kernels.cuh
  cuda/utils.cuh
  python/benchmark_one_day.py
)

tar -C "$challenge_dir" -cf - "${upload_files[@]}" | \
  kubectl "${kubectl_args[@]}" exec -i "$pod" -- tar -C "$remote_dir" -xf -

if [[ "$with_cutlass" -eq 1 ]]; then
  if [[ -z "$cutlass_dir" ]]; then
    for candidate in \
      "$challenge_dir/cutlass" \
      "$challenge_dir/../LeetCUDA/cutlass" \
      "$HOME/LeetCUDA/cutlass"; do
      if [[ -d "$candidate/include/cutlass" ]]; then
        cutlass_dir="$candidate"
        break
      fi
    done
  fi

  if [[ -z "$cutlass_dir" || ! -d "$cutlass_dir/include/cutlass" ]]; then
    echo "--with-cutlass requested, but CUTLASS headers were not found. Pass --cutlass-dir DIR or set CUTLASS_DIR." >&2
    exit 1
  fi

  echo "Uploading CUTLASS headers from $cutlass_dir"
  kubectl "${kubectl_args[@]}" exec "$pod" -- bash -lc "mkdir -p '$remote_dir/solutions/third-party/cutlass'"
  tar -cf - -C "$cutlass_dir" include -C "$cutlass_dir/tools/util" include | \
    kubectl "${kubectl_args[@]}" exec -i "$pod" -- \
      bash -lc "tar -C '$remote_dir/solutions/third-party/cutlass' -xf -"
fi

run_args=(--day "$day" --warmup "$warmup" --iters "$iters" --runs "$runs")
if [[ -n "$arch" ]]; then
  run_args+=(--arch "$arch")
fi
if [[ -n "$sizes" ]]; then
  run_sizes="$sizes"
elif [[ "$benchmark_mode" == "verbose" ]]; then
  run_sizes="$explore_gemm_sizes"
else
  run_sizes="$easy_sizes"
fi
run_args+=(--sizes $run_sizes)
if [[ -n "$dtype" ]]; then
  run_args+=(--dtype "$dtype")
fi
if [[ -n "$tolerance" ]]; then
  run_args+=(--tolerance "$tolerance")
fi
if [[ "$verbose_build" -eq 1 ]]; then
  run_args+=(--verbose-build)
fi

printf -v quoted_args " %q" "${run_args[@]}"
echo "Running correctness + benchmark"
timestamp="$(date +%Y%m%d_%H%M%S)"
result_file="$result_dir/day$(printf '%02d' "$day")_${timestamp}.log"
tmp_result="$(mktemp)"
cleanup_tmp_result() {
  rm -f "$tmp_result"
}
trap cleanup_tmp_result EXIT

set +e
{
  echo "# GEMM hard-way challenge day $day"
  echo "# timestamp: $timestamp"
  echo "# namespace: $namespace"
  echo "# pod: $pod"
  echo "# remote_dir: $remote_dir"
  echo "# benchmark_mode: $benchmark_mode"
  echo "# command: python3 python/benchmark_one_day.py$quoted_args"
  echo
  kubectl "${kubectl_args[@]}" exec "$pod" -- bash -lc "
    set -euo pipefail
    cd '$remote_dir'
    python3 python/benchmark_one_day.py$quoted_args
  "
} 2>&1 | tee "$tmp_result"
run_status=${PIPESTATUS[0]}
set -e

if [[ "$run_status" -eq 0 ]]; then
  mkdir -p "$result_dir"
  mv "$tmp_result" "$result_file"
  trap - EXIT
  echo "Saved successful run output to $result_file"
else
  echo "Run failed; not saving result log." >&2
  exit "$run_status"
fi
