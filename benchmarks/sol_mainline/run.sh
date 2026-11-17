#!/usr/bin/env bash
# Sol-Attn on aiter mainline: end-to-end rows for Wan2.2-T2V and MiniMax-H3.
#
# Runs in container jjuvonen-sol-mainline (~/sol_mainline/xDiT at /app/xDiT, aiter main built in
# /app/external/aiter). Each row takes 8 whole GPUs through brr; GPU core dumps are off.
# Outputs: /output/<model>/<row>_s<seed>/ (host: ~/sol_mainline/output). Rows with a finished
# output are skipped.
#
# Usage: tools/run.sh ROW [ROW ...]            e.g. tools/run.sh wan_dense wan_fp8sol
#        SEEDS="1234 1 2" tools/run.sh wan_fp8sol
#        BRR_ARGS="--node" tools/run.sh ...     (timing: whole node, queued behind other jobs)
set -uo pipefail

CONTAINER=${CONTAINER:-jjuvonen-sol-mainline}
SRC=${SRC:-/app/xDiT}   # xDiT checkout to run, e.g. /work/xDiT-b
BRR_ARGS=${BRR_ARGS:---gpus 8 --exclusive}
SEEDS=${SEEDS:-1234}
EXTRA=${EXTRA:-}
TAG=${TAG:-}   # suffix for the output directory, e.g. TAG=_main

WAN_PROMPT="industrial robot arm working on a conveyor belt"
WAN_COMMON="--model Wan-AI/Wan2.2-T2V-A14B-Diffusers --height 720 --width 1280 --num_frames 81 \
  --num_inference_steps 40 --ulysses_degree 8 --use_torch_compile --use_parallel_vae \
  --warmup_calls 0 --num_iterations 1 --cross_attention_backend aiter"

H3_PROMPT="A street musician plays an acoustic guitar and sings on a busy city sidewalk at sunset, passers-by stop to listen"
H3_COMMON="--model MiniMaxAI/MiniMax-H3 --task t2va --ulysses_degree 8 --use_torch_compile \
  --warmup_calls 0 --num_iterations 1"

declare -A ROW=(
  [wan_dense]="wan|--attention_backend aiter_bf16"
  [wan_bf16sol]="wan|--attention_backend aiter_bf16_sol"
  [wan_fp8sol]="wan|--attention_backend aiter_fp8_sol"
  [wan_fp8sol_tilt2]="wan|--attention_backend aiter_fp8_sol --solattn_tilt_rank 2"
  [wan_bf16sol_tilt2]="wan|--attention_backend aiter_bf16_sol --solattn_tilt_rank 2"
  [wan_fp8sol_sched]="wan|--attention_backend aiter_fp8_sol --solattn_beta_schedule=0.5:-0.5"
  [h3_dense]="h3|--attention_backend aiter_bf16"
  [h3_bf16sol]="h3|--attention_backend aiter_bf16_sol"
  [h3_fp8sol]="h3|--attention_backend aiter_fp8_sol"
)

for row in "$@"; do
  spec=${ROW[$row]:-}
  [[ -z $spec ]] && { echo "unknown row $row" >&2; exit 2; }
  model=${spec%%|*}; flags=${spec#*|}
  if [[ $model == wan ]]; then common=$WAN_COMMON; prompt=$WAN_PROMPT; else common=$H3_COMMON; prompt=$H3_PROMPT; fi
  for seed in $SEEDS; do
    out=/output/$model/${row}${TAG}_s$seed
    host_out=$HOME/sol_mainline/output/$model/${row}${TAG}_s$seed
    if compgen -G "$host_out/*.mp4" > /dev/null; then echo "skip $row s$seed"; continue; fi
    echo "run $row s$seed -> $host_out"
    docker exec "$CONTAINER" bash -c "mkdir -p $out && cd $SRC && export PYTHONPATH=$SRC && ulimit -c 0 && \
      /opt/brr/brr run $BRR_ARGS -- xdit $common $flags $EXTRA --seed $seed \
      --prompt \"$prompt\" --output_directory $out > $out/out.log 2>&1"
    echo "  rc=$? $(grep -hoE "Average time over [0-9]+ runs: [0-9.]+s" $host_out/out.log | tail -1)"
  done
done
