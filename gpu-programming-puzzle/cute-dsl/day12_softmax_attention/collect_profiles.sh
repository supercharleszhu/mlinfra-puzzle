#!/usr/bin/env bash
set -euo pipefail

day_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-python3}"
output_dir="${1:-$day_dir/reports}"
sequence_length="${SEQUENCE_LENGTH:-512}"
head_dim="${HEAD_DIM:-64}"

mkdir -p "$output_dir"

metrics=(
  gpu__time_duration.sum
  dram__bytes_read.sum
  dram__bytes_write.sum
  lts__t_bytes.sum
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum
  l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum
  l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum
  sm__warps_active.avg.pct_of_peak_sustained_active
  sm__issue_active.avg.pct_of_peak_sustained_elapsed
  sm__throughput.avg.pct_of_peak_sustained_elapsed
  sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed
  sm__inst_executed.sum
  sm__inst_executed_pipe_tensor.sum
  sm__sass_inst_executed_op_ldgsts.sum
  smsp__warp_issue_stalled_barrier_per_warp_active
  smsp__warp_issue_stalled_short_scoreboard_per_warp_active
  smsp__warp_issue_stalled_long_scoreboard_per_warp_active
  smsp__warp_issue_stalled_wait_per_warp_active
  launch__registers_per_thread
  launch__shared_mem_per_block
)
metric_list="$(IFS=,; echo "${metrics[*]}")"

for variant in naive cute; do
  report="$output_dir/${variant}_profile"
  ncu \
    --profile-from-start off \
    --target-processes all \
    --metrics "$metric_list" \
    --force-overwrite \
    -o "$report" \
    "$python_bin" "$day_dir/profile_target.py" \
      --variant "$variant" \
      --sequence-length "$sequence_length" \
      --head-dim "$head_dim"
  ncu --import "$report.ncu-rep" --csv --page raw \
    > "$report.csv"
done

printf 'Reports written to %s\n' "$output_dir"
