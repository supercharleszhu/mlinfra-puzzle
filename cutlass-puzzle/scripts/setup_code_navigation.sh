#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
challenge_dir="$(cd "$script_dir/.." && pwd)"
repo_dir="$(cd "$challenge_dir/.." && pwd)"
cutlass_dir="${CUTLASS_DIR:-$challenge_dir/.cache/cutlass-v4.6.1}"
build_dir="${1:-$repo_dir/build/cutlass-puzzle-navigation}"

if [[ ! -f "$cutlass_dir/include/cute/tensor.hpp" ]]; then
  echo "Pinned CUTLASS headers not found at $cutlass_dir" >&2
  echo "Run $script_dir/setup_advanced_cutlass.sh first." >&2
  exit 1
fi

cmake \
  -S "$challenge_dir" \
  -B "$build_dir" \
  -DCUTLASS_DIR="$cutlass_dir" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

echo "Compilation database ready at $build_dir/compile_commands.json"
