#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Upload the CUTLASS puzzle curriculum to a Kubernetes GPU pod.

Usage:
  scripts/upload_to_pod.sh [options]

Options:
  --namespace NS       Kubernetes namespace. Default: $GEMM_NS or kk-flyte-adhoc
  --pod POD            Pod name. Default: $GEMM_POD or a9lfvz8vvnrm4vpb86dg-n0-0
  --remote-dir DIR     Default: $GEMM_REMOTE_DIR or /tmp/cutlass-puzzle
  --container NAME     Optional container name for multi-container pods.
  --with-cutlass       Also upload CUTLASS headers for days 13-19.
  --cutlass-dir DIR    CUTLASS checkout. Default: $CUTLASS_DIR, ./cutlass, ../LeetCUDA/cutlass.
  -h, --help           Show this help.

Examples:
  scripts/upload_to_pod.sh
  scripts/upload_to_pod.sh --pod my-new-pod --with-cutlass
EOF
}

challenge_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

namespace="${GEMM_NS:-kk-flyte-adhoc}"
pod="${GEMM_POD:-a9lfvz8vvnrm4vpb86dg-n0-0}"
remote_dir="${GEMM_REMOTE_DIR:-/tmp/cutlass-puzzle}"
container=""
with_cutlass=0
cutlass_dir="${CUTLASS_DIR:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
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
    --with-cutlass)
      with_cutlass=1
      shift
      ;;
    --cutlass-dir)
      cutlass_dir="$2"
      shift 2
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

if [[ ! -d "$challenge_dir" ]]; then
  echo "Challenge folder not found: $challenge_dir" >&2
  exit 1
fi

kubectl_args=(-n "$namespace")
if [[ -n "$container" ]]; then
  kubectl_args+=(-c "$container")
fi

echo "Checking pod: namespace=$namespace pod=$pod"
kubectl -n "$namespace" get pod "$pod" -o wide

echo "Uploading $challenge_dir -> $pod:$remote_dir"
tar -C "$challenge_dir" \
  --exclude='.git' \
  --exclude='**/__pycache__' \
  --exclude='**/build' \
  --exclude='result' \
  -cf - . | \
  kubectl "${kubectl_args[@]}" exec -i "$pod" -- \
    bash -lc "rm -rf '$remote_dir' && mkdir -p '$remote_dir' && tar -C '$remote_dir' -xf -"

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

  echo "Uploading CUTLASS headers for days 13-19 from $cutlass_dir"
  kubectl "${kubectl_args[@]}" exec "$pod" -- \
    bash -lc "mkdir -p '$remote_dir/solutions/third-party/cutlass'"
  tar -cf - -C "$cutlass_dir" include -C "$cutlass_dir/tools/util" include | \
    kubectl "${kubectl_args[@]}" exec -i "$pod" -- \
      bash -lc "tar -C '$remote_dir/solutions/third-party/cutlass' -xf -"
fi

echo "Upload complete. Remote contents:"
kubectl "${kubectl_args[@]}" exec "$pod" -- \
  bash -lc "cd '$remote_dir' && find . -maxdepth 2 -type f | sort | head -60"
