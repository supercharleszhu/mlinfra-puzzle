#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Upload the Triton track and prepare Jupyter on a Kubernetes GPU pod.

Usage:
  scripts/setup_remote_notebook.sh [options]

Options:
  --namespace NS    Default: $GEMM_NS or kk-flyte-adhoc
  --pod POD         Default: $GEMM_POD or axx5w8ksbsbqz9b5wmnj-n0-0
  --remote-dir DIR  Default: /tmp/gpu-programming-puzzle-triton
  --verify          Run Day 01 correctness after setup.
  -h, --help        Show this help.
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
track_dir="$(cd "$script_dir/.." && pwd)"
namespace="${GEMM_NS:-kk-flyte-adhoc}"
pod="${GEMM_POD:-axx5w8ksbsbqz9b5wmnj-n0-0}"
remote_dir="/tmp/gpu-programming-puzzle-triton"
verify=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --namespace) namespace="$2"; shift 2 ;;
    --pod) pod="$2"; shift 2 ;;
    --remote-dir) remote_dir="$2"; shift 2 ;;
    --verify) verify=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

kubectl -n "$namespace" get pod "$pod" >/dev/null
kubectl -n "$namespace" exec "$pod" -- mkdir -p "$remote_dir"
tar \
  --exclude='.venv' \
  --exclude='__pycache__' \
  --exclude='.ipynb_checkpoints' \
  -C "$track_dir" -cf - . | \
  kubectl -n "$namespace" exec -i "$pod" -- tar -C "$remote_dir" -xf -

kubectl -n "$namespace" exec "$pod" -- bash -lc "
  python3 -m venv --system-site-packages '$remote_dir/.venv'
  '$remote_dir/.venv/bin/python' -c 'import torch, triton'
  '$remote_dir/.venv/bin/python' -m pip install --disable-pip-version-check \
    -r '$remote_dir/requirements.txt'
  JUPYTER_PLATFORM_DIRS=1 '$remote_dir/.venv/bin/python' -m ipykernel install \
    --prefix '$remote_dir/jupyter-kernel' \
    --name triton-puzzle \
    --display-name 'Triton (GPU)'
"

if [[ "$verify" -eq 1 ]]; then
  kubectl -n "$namespace" exec "$pod" -- bash -lc "
    cd '$remote_dir'
    '$remote_dir/.venv/bin/python' scripts/check_curriculum.py
    '$remote_dir/.venv/bin/python' day01_vector_add/solution.py
  "
fi

cat <<EOF
Triton notebook environment is ready.

Start Jupyter:
  kubectl -n $namespace exec -it $pod -- bash -lc \\
    "cd '$remote_dir' && JUPYTER_PATH='$remote_dir/jupyter-kernel/share/jupyter' \\
     '$remote_dir/.venv/bin/python' -m jupyterlab --ip=0.0.0.0 --port=8889 --no-browser"

In another terminal:
  kubectl -n $namespace port-forward pod/$pod 8889:8889
EOF
