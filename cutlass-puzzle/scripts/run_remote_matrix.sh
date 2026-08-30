#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Convenience wrapper: upload the challenge, then run selected days remotely.

Usage:
  scripts/run_remote_matrix.sh --day DAY [options]
  scripts/run_remote_matrix.sh --all [options]

Options are forwarded to run_remote_day.sh. This wrapper additionally accepts:
  --skip-upload       Do not upload before running.
  --with-cutlass      Upload CUTLASS headers and pass --include-cutlass.

Examples:
  scripts/run_remote_matrix.sh --day 1
  scripts/run_remote_matrix.sh --day 14 --with-cutlass
  scripts/run_remote_matrix.sh --all --skip-upload
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
skip_upload=0
with_cutlass=0
declare -a forwarded=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-upload)
      skip_upload=1
      shift
      ;;
    --with-cutlass)
      with_cutlass=1
      forwarded+=(--include-cutlass)
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      forwarded+=("$1")
      shift
      ;;
  esac
done

if [[ "$skip_upload" -eq 0 ]]; then
  upload_args=()
  if [[ "$with_cutlass" -eq 1 ]]; then
    upload_args+=(--with-cutlass)
  fi
  "$script_dir/upload_to_pod.sh" "${upload_args[@]}"
fi

"$script_dir/run_remote_day.sh" "${forwarded[@]}"

