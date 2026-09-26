#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Upload and run one NVIDIA CuTe C++ tutorial lesson (Days 20-28).

Usage:
  cuda-cpp/scripts/upload_run_advanced_day.sh --day DAY [options]

Options:
  --day DAY           Day number: 20..28.
  --solution          Run the completed solution. Default: challenge.
  --namespace NS      Default: $GEMM_NS or kk-flyte-adhoc
  --pod POD           Default: $GEMM_POD or a9lfvz8vvnrm4vpb86dg-n0-0
  --remote-dir DIR    Default: /tmp/gpu-programming-puzzle-advanced
  --cutlass-dir DIR   Pinned CUTLASS v4.6.1 checkout.
  --iters N           Reserved compatibility option; tutorial mains time themselves.
  --jobs N            Parallel NVCC compiler threads. Default: 2
  --build-only        Compile without executing the example.
  --dry-run           Check blanks and print the selected local source.
  --verbose-build     Print the NVCC command.
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
challenge_dir="$(cd "$script_dir/../.." && pwd)"
namespace="${GEMM_NS:-kk-flyte-adhoc}"
pod="${GEMM_POD:-a9lfvz8vvnrm4vpb86dg-n0-0}"
remote_dir="/tmp/gpu-programming-puzzle-advanced"
cutlass_dir="$challenge_dir/.cache/cutlass-v4.6.1"
commit="e05f953a5b3d38adc240df2ff928e0421c2abba3"
day=""
source="challenge"
iters=20
jobs=2
build_only=0
dry_run=0
verbose_build=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --day) day="$2"; shift 2 ;;
    --solution) source="solution"; shift ;;
    --namespace) namespace="$2"; shift 2 ;;
    --pod) pod="$2"; shift 2 ;;
    --remote-dir) remote_dir="$2"; shift 2 ;;
    --cutlass-dir) cutlass_dir="$2"; shift 2 ;;
    --iters) iters="$2"; shift 2 ;;
    --jobs) jobs="$2"; shift 2 ;;
    --build-only) build_only=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    --verbose-build) verbose_build=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! "$day" =~ ^(20|21|22|23|24|25|26|27|28)$ ]]; then
  echo "--day must be between 20 and 28" >&2
  exit 2
fi

if [[ ! -d "$cutlass_dir/.git" ]]; then
  "$script_dir/setup_advanced_cutlass.sh" "$cutlass_dir"
fi
actual_commit="$(git -C "$cutlass_dir" rev-parse HEAD)"
if [[ "$actual_commit" != "$commit" ]]; then
  echo "--cutlass-dir must point to pinned CUTLASS v4.6.1 ($commit)" >&2
  exit 1
fi

kubectl -n "$namespace" get pod "$pod" >/dev/null
kubectl -n "$namespace" exec "$pod" -- bash -lc \
  "mkdir -p '$remote_dir/cuda-cpp/challenge' '$remote_dir/cuda-cpp/solutions' '$remote_dir/python'"

tar -C "$challenge_dir" -cf - \
  cuda-cpp/challenge cuda-cpp/solutions python/benchmark_advanced_day.py | \
  kubectl -n "$namespace" exec -i "$pod" -- tar -C "$remote_dir" -xf -

remote_commit="$(
  kubectl -n "$namespace" exec "$pod" -- bash -lc \
    "cat '$remote_dir/official-cutlass/.cutlass-advanced-commit' 2>/dev/null || true" \
    | tail -n 1
)"
if [[ "$remote_commit" != "$commit" ]]; then
  echo "Uploading pinned CUTLASS v4.6.1 C++ sources"
  kubectl -n "$namespace" exec "$pod" -- bash -lc \
    "rm -rf '$remote_dir/official-cutlass' && mkdir -p '$remote_dir/official-cutlass'"
  tar \
    --exclude=.git \
    --exclude=build \
    --exclude=docs \
    --exclude=media \
    --exclude=python \
    --exclude=test \
    -C "$cutlass_dir" -cf - . | \
    kubectl -n "$namespace" exec -i "$pod" -- \
      tar -C "$remote_dir/official-cutlass" -xf -
  kubectl -n "$namespace" exec "$pod" -- bash -lc \
    "printf '%s\n' '$commit' > '$remote_dir/official-cutlass/.cutlass-advanced-commit'"
fi

run_args=(--day "$day" --source "$source" --cutlass-dir official-cutlass
  --iters "$iters" --jobs "$jobs")
if [[ "$build_only" -eq 1 ]]; then
  run_args+=(--build-only)
fi
if [[ "$dry_run" -eq 1 ]]; then
  run_args+=(--dry-run)
fi
if [[ "$verbose_build" -eq 1 ]]; then
  run_args+=(--verbose-build)
fi
printf -v quoted_args " %q" "${run_args[@]}"
kubectl -n "$namespace" exec "$pod" -- bash -lc \
  "cd '$remote_dir' && python3 python/benchmark_advanced_day.py$quoted_args"
