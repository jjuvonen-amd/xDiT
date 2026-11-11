#!/bin/bash
# Headline numbers on the final code (compile mode defaults to reduce-overhead).
B=/app/xDiT/benchmarks/minimax_h3_image/bench.sh
U8="--ulysses_degree 8 --use_parallel_vae --attention_backend AITER_BF16 --use_torch_compile"
PORT=29701 $B f2_u1_eager 1 2048 --attention_backend AITER_BF16 --num_iterations 5
PORT=29702 $B f2_u1_tc 1 2048 --attention_backend AITER_BF16 --use_torch_compile --num_iterations 5
PORT=29703 $B f4_u1_tc 1 4096 --attention_backend AITER_BF16 --use_torch_compile --num_iterations 5
PORT=29704 $B f2_u8_tc 8 2048 $U8 --num_iterations 10
PORT=29705 $B f4_u8_tc 8 4096 $U8 --num_iterations 10
