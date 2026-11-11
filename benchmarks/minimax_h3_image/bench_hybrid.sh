#!/bin/bash
# Hybrid attention schedule: BF16 attention for the first N steps, FP8 after; FP8 GEMMs throughout.
HYB="--use_hybrid_attn_schedule --hybrid_attn_high_precision_backend AITER_BF16 --hybrid_attn_low_precision_backend AITER_FP8"
prompt="An overhead photograph of a rustic farm-to-table spread on a weathered wooden table: a round loaf of sourdough on burlap, wedges of cheese on a dark board, a copper bowl of vegetables, vine tomatoes, olives and walnuts."
for n in 2 3; do
  echo "== teacher hybrid$n"
  mkdir -p /output/quality/hybrid$n
  TEACHER_DIR=/output/ref/t2i_2048/r1 torchrun --nproc-per-node 8 --master-port 2971$n /app/xDiT/benchmarks/minimax_h3_image/xdit_run.py \
    --model MiniMax-H3-Image --height 2048 --width 2048 --seed 42 --prompt "$prompt" --ulysses_degree 8 \
    --use_fp8_gemms $HYB --num_hybrid_attn_high_precision_steps $n --output_directory /output/quality/hybrid$n \
    > /output/quality/hybrid$n/log 2>&1
  grep -o "\[teacher\] step [0-9]: rel-L2 [0-9.e+-]*" /output/quality/hybrid$n/log
  grep -E "Error|error:" /output/quality/hybrid$n/log | head -3
  PORT=2972$n /output/bench.sh h4_u8_fp8_hybrid$n 8 4096 --ulysses_degree 8 --use_parallel_vae --use_torch_compile \
    --use_fp8_gemms $HYB --num_hybrid_attn_high_precision_steps $n --num_iterations 10
done
