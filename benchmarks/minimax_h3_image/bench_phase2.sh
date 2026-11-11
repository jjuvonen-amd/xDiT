#!/bin/bash
# Phase 2: rank the top BF16 u8 configs with more iterations, FP8 rows, and the
# teacher-forced quality of FP8 against the 2048² reference dumps.
B=/app/xDiT/benchmarks/minimax_h3_image/bench.sh
U8="--ulysses_degree 8 --use_parallel_vae --attention_backend AITER_BF16 --use_torch_compile"
U8FP8A="--ulysses_degree 8 --use_parallel_vae --attention_backend AITER_FP8 --use_torch_compile"
port=29660
run() {  # name nproc size envs args...
  local name=$1 nproc=$2 size=$3 envs=$4
  shift 4
  port=$((port + 1))
  env PORT=$port $envs $B "$name" "$nproc" "$size" "$@"
}
for size in 2048 4096; do
  s=${size:0:1}
  for rep in a b; do
    run p${s}_u8_tc_$rep 8 $size "" $U8 --num_iterations 10
    run p${s}_u8_tc_ro_$rep 8 $size COMPILE_MODE=reduce-overhead $U8 --num_iterations 10
    run p${s}_u8_tc_sdma_ov_$rep 8 $size "" $U8 --ulysses_a2a sdma --ulysses_sdma_overlap_qkv --num_iterations 10
  done
  run p${s}_u8_fp8g_ro 8 $size COMPILE_MODE=reduce-overhead $U8 --use_fp8_gemms --num_iterations 10
  run p${s}_u8_fp8ga_ro 8 $size COMPILE_MODE=reduce-overhead $U8FP8A --use_fp8_gemms --num_iterations 10
done

# Teacher-forced quality at 2048² (eager is enough: the kernels are the same).
prompt="An overhead photograph of a rustic farm-to-table spread on a weathered wooden table: a round loaf of sourdough on burlap, wedges of cheese on a dark board, a copper bowl of vegetables, vine tomatoes, olives and walnuts."
teach() {  # name args...
  local name=$1
  shift
  port=$((port + 1))
  mkdir -p /output/quality/$name
  echo "== teacher $name ($*)"
  TEACHER_DIR=/output/ref/t2i_2048/r1 torchrun --nproc-per-node 8 --master-port $port /app/xDiT/benchmarks/minimax_h3_image/xdit_run.py \
    --model MiniMax-H3-Image --height 2048 --width 2048 --seed 42 --prompt "$prompt" \
    --ulysses_degree 8 --output_directory /output/quality/$name "$@" 2>&1 | grep "\[teacher\]"
}
teach u8_bf16 --attention_backend AITER_BF16
teach u8_fp8g --attention_backend AITER_BF16 --use_fp8_gemms
teach u8_fp8ga --attention_backend AITER_FP8 --use_fp8_gemms
