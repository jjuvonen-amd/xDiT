#!/bin/bash
# BF16 benchmark matrix for MiniMax-H3-Image, run inside the container on a whole node.
# Usage: bench_matrix.sh [filter-regex]
B=/app/xDiT/benchmarks/minimax_h3_image/bench.sh
U8="--ulysses_degree 8 --use_parallel_vae --attention_backend AITER_BF16 --use_torch_compile"
U4="--ulysses_degree 4 --use_parallel_vae --attention_backend AITER_BF16 --use_torch_compile"
SDMA="--ulysses_a2a sdma"
configs=(
  "m2_u1_eager|1|2048|--attention_backend AITER_BF16"
  "m2_u1_tc|1|2048|--attention_backend AITER_BF16 --use_torch_compile"
  "m2_u8_tc|8|2048|$U8"
  "m2_u8_tc_tetp|8|2048|$U8 --text_encoder_tp_degree 8"
  "m2_u8_tc_sdma|8|2048|$U8 $SDMA"
  "m2_u8_tc_sdma_ov|8|2048|$U8 $SDMA --ulysses_sdma_overlap_qkv"
  "m2_u8_tc_ro|8|2048|$U8|COMPILE_MODE=reduce-overhead"
  "m2_u8_tc_rms|8|2048|$U8|AITER_RMSNORM=1"
  "m2_u4_tc|4|2048|$U4"
  "m4_u1_eager|1|4096|--attention_backend AITER_BF16"
  "m4_u1_tc|1|4096|--attention_backend AITER_BF16 --use_torch_compile"
  "m4_u8_tc|8|4096|$U8"
  "m4_u8_tc_sdma|8|4096|$U8 $SDMA"
  "m4_u8_tc_sdma_ov|8|4096|$U8 $SDMA --ulysses_sdma_overlap_qkv"
  "m4_u8_tc_ro|8|4096|$U8|COMPILE_MODE=reduce-overhead"
  "m4_u4_tc|4|4096|$U4"
)
port=29630
for entry in "${configs[@]}"; do
  IFS='|' read -r name nproc size args envs <<< "$entry"
  [[ -n "$1" && ! "$name" =~ $1 ]] && continue
  port=$((port + 1))
  env PORT=$port $envs $B "$name" "$nproc" "$size" $args
done
