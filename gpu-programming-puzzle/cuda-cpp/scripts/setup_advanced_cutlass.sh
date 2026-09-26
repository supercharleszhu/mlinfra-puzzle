#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
challenge_dir="$(cd "$script_dir/../.." && pwd)"
destination="${1:-$challenge_dir/.cache/cutlass-v4.6.1}"
commit="e05f953a5b3d38adc240df2ff928e0421c2abba3"

if [[ ! -d "$destination/.git" ]]; then
  mkdir -p "$(dirname "$destination")"
  git clone --depth 1 --branch v4.6.1 https://github.com/NVIDIA/cutlass.git "$destination"
fi

actual="$(git -C "$destination" rev-parse HEAD)"
if [[ "$actual" != "$commit" ]]; then
  echo "Expected CUTLASS v4.6.1 commit $commit, found $actual" >&2
  exit 1
fi

printf '%s\n' "$actual" > "$destination/.cutlass-advanced-commit"
echo "Pinned NVIDIA CuTe tutorial sources ready at $destination"
