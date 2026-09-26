#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Upload the CuTe DSL track and prepare Jupyter on a Kubernetes GPU pod.

Usage:
  scripts/setup_remote_notebook.sh [options]

Options:
  --namespace NS    Default: $GEMM_NS or kk-flyte-adhoc
  --pod POD         Default: $GEMM_POD or a9lfvz8vvnrm4vpb86dg-n0-0
  --remote-dir DIR  Default: /tmp/gpu-programming-puzzle-cute-dsl
  --verify          Execute official Notebook 01 after setup.
  -h, --help        Show this help.
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dsl_dir="$(cd "$script_dir/.." && pwd)"
namespace="${GEMM_NS:-kk-flyte-adhoc}"
pod="${GEMM_POD:-a9lfvz8vvnrm4vpb86dg-n0-0}"
remote_dir="/tmp/gpu-programming-puzzle-cute-dsl"
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
  -C "$dsl_dir" -cf - . | \
  kubectl -n "$namespace" exec -i "$pod" -- tar -C "$remote_dir" -xf -

driver_major="$(
  kubectl -n "$namespace" exec "$pod" -- \
    nvidia-smi --query-gpu=driver_version --format=csv,noheader |
    head -n 1 | cut -d. -f1
)"
if (( driver_major >= 580 )); then
  cuda_python_spec='cuda-python>=13,<14'
else
  cuda_python_spec='cuda-python>=12,<13'
fi

kubectl -n "$namespace" exec "$pod" -- bash -lc "
  python3 -m venv --system-site-packages '$remote_dir/.venv'
  '$remote_dir/.venv/bin/python' -m pip install --disable-pip-version-check \
    'nvidia-cutlass-dsl>=4.6,<5' '$cuda_python_spec' 'numpy>=1.24' \
    jupyter nbconvert ipykernel
  JUPYTER_PLATFORM_DIRS=1 '$remote_dir/.venv/bin/python' -m ipykernel install \
    --prefix '$remote_dir/jupyter-kernel' \
    --name cutlass-cute-dsl \
    --display-name 'CuTe DSL (H100)'
"

if [[ "$verify" -eq 1 ]]; then
  kubectl -n "$namespace" exec "$pod" -- bash -lc "
    cd '$remote_dir'
    JUPYTER_PATH='$remote_dir/jupyter-kernel/share/jupyter' \
      '$remote_dir/.venv/bin/python' -m nbconvert \
      --to notebook \
      --execute notebooks/01_hello_world.ipynb \
      --output '$remote_dir/01_hello_world.executed.ipynb' \
      --ExecutePreprocessor.kernel_name=cutlass-cute-dsl \
      --ExecutePreprocessor.timeout=600
  "
fi

cat <<EOF
CuTe DSL notebook environment is ready.

Start Jupyter:
  kubectl -n $namespace exec -it $pod -- bash -lc \\
    "cd '$remote_dir' && JUPYTER_PATH='$remote_dir/jupyter-kernel/share/jupyter' \\
     '$remote_dir/.venv/bin/python' -m jupyterlab --ip=0.0.0.0 --port=8888 --no-browser"

In another terminal:
  kubectl -n $namespace port-forward pod/$pod 8888:8888
EOF
