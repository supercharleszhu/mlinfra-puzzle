#!/usr/bin/env bash
set -euo pipefail

day_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-python3}"
output_dir="${1:-$day_dir/reports}"
m="${M:-4096}"
n="${N:-4096}"

mkdir -p "$output_dir"

metrics=(
  gpu__time_duration.sum
  dram__bytes_read.sum
  dram__bytes_write.sum
  dram__throughput.avg.pct_of_peak_sustained_elapsed
  l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum
  l1tex__t_requests_pipe_lsu_mem_global_op_st.sum
  l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
  l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum
  l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio
  l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum
  sm__warps_active.avg.pct_of_peak_sustained_active
  launch__registers_per_thread
  launch__shared_mem_per_block_dynamic
)
metric_list="$(IFS=,; echo "${metrics[*]}")"

variants=(
  naive-read
  naive-write
  shared-unpadded
  shared-padded
  shared-swizzled
)

for variant in "${variants[@]}"; do
  name="${variant//-/_}"
  report="$output_dir/$name"

  if [[ "$variant" == "naive-read" ]]; then
    kernel_regex='regex:.*naive_coalesced_read_kernel.*'
  elif [[ "$variant" == "naive-write" ]]; then
    kernel_regex='regex:.*naive_coalesced_write_kernel.*'
  else
    kernel_regex='regex:.*shared_transpose_kernel.*'
  fi

  ncu \
    --target-processes all \
    --kernel-name "$kernel_regex" \
    --launch-count 1 \
    --metrics "$metric_list" \
    --force-overwrite \
    -o "$report" \
    "$python_bin" "$day_dir/profile_target.py" \
      --M "$m" --N "$n" --variant "$variant"

  ncu --import "$report.ncu-rep" --csv --page raw \
    > "$report.csv"
done

"$python_bin" "$day_dir/solution.py" \
  --naive-read "$output_dir/naive_read.csv" \
  --naive-write "$output_dir/naive_write.csv" \
  --unpadded "$output_dir/shared_unpadded.csv" \
  --padded "$output_dir/shared_padded.csv" \
  --swizzled "$output_dir/shared_swizzled.csv"
