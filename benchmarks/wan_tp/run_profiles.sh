#!/usr/bin/env bash
# Profile Wan2.2-T2V-A14B under tensor and sequence parallelism; one xdit run per config.
#
#   bash benchmarks/wan_tp/run_profiles.sh CONFIG [CONFIG ...]
#
# CONFIG names:
#   sp8                       Ulysses over 8 GPUs, no TP (the reference)
#   tp<N>_x<cross>[_ov|_fused|_sdma]       Megatron TP+SP over N GPUs
#   tp<N>_u<M>_x<cross>[_ov|_fused|_sdma]  TP N x Ulysses M
# where <cross> is "sharded" (cross-attention split too: 3 all-gathers + 3 reduce-scatters
# per block) or "repl" (cross-attention whole on each rank's rows: 2 + 2). "_ov" pipelines
# each collective against its GEMM in chunks (XDIT_TP_COMM_OVERLAP=chunked); "_fused" uses
# torch's symm_mem fused AG+GEMM / GEMM+RS ops (XDIT_TP_COMM_OVERLAP=symm_mem; NVLink only);
# "_sdma" gathers each all-gather + GEMM on SDMA copy engines (XDIT_TP_COMM_OVERLAP=sdma; ROCm).
#
# Environment:
#   OUTPUT_DIR  where <config>/{log.txt,profile_trace_rank_<r>.json.gz} go (default ./output/profiles)
#   MODEL       default Wan-AI/Wan2.2-T2V-A14B-Diffusers
#   STEPS       denoising steps per profiled iteration (default 4; both experts run)
#   ATTN        --attention_backend, e.g. FLASH_4 or CUDNN (default: the model's)
#   CHUNKS      XDIT_TP_OVERLAP_CHUNKS for _ov configs (default 4)
#   XDIT_TP_SDMA_CONSUMER, XDIT_TP_SDMA_TRIGGER  passed through for _sdma configs
#   PROFILE     1 to profile one iteration after one warmup (default), 0 for a warmed-up timed run
#   EXTRA_ARGS  appended to every xdit command line
#
# Summarize the traces with benchmarks/wan_tp/summarize_traces.py.
set -u

OUTPUT_DIR=${OUTPUT_DIR:-./output/profiles}
MODEL=${MODEL:-Wan-AI/Wan2.2-T2V-A14B-Diffusers}
STEPS=${STEPS:-4}
CHUNKS=${CHUNKS:-4}
PROFILE=${PROFILE:-1}

if [ $# -eq 0 ]; then
  sed -n '2,26p' "$0"
  exit 1
fi

common=(--model "$MODEL" --height 720 --width 1280 --num_frames 81 --num_inference_steps "$STEPS"
        --prompt "A cat walking on grass" --seed 42 --use_torch_compile)
# A timed run warms up first, so one-off costs left after the compile warmup stay out of the timing.
if [ "$PROFILE" = 1 ]; then
  common+=(--profile --profile_wait 0 --profile_warmup 1 --profile_active 1)
else
  common+=(--warmup_calls 1)
fi
[ -n "${ATTN:-}" ] && common+=(--attention_backend "$ATTN")
# shellcheck disable=SC2206  # EXTRA_ARGS is a whitespace-separated argument list by design.
[ -n "${EXTRA_ARGS:-}" ] && common+=(${EXTRA_ARGS})

status=0
for name in "$@"; do
  cross=sharded
  overlap=none
  if [ "$name" = sp8 ]; then
    parallel=(--ulysses_degree 8)
  elif [[ $name =~ ^tp([0-9]+)(_u([0-9]+))?_x(sharded|repl)(_ov|_fused|_sdma)?$ ]]; then
    parallel=(--tensor_parallel_degree "${BASH_REMATCH[1]}")
    [ -n "${BASH_REMATCH[3]}" ] && parallel+=(--ulysses_degree "${BASH_REMATCH[3]}")
    [ "${BASH_REMATCH[4]}" = repl ] && cross=replicated
    [ "${BASH_REMATCH[5]}" = _ov ] && overlap=chunked
    [ "${BASH_REMATCH[5]}" = _fused ] && overlap=symm_mem
    [ "${BASH_REMATCH[5]}" = _sdma ] && overlap=sdma
  else
    echo "unknown config: $name" >&2
    status=1
    continue
  fi
  out=$OUTPUT_DIR/$name
  mkdir -p "$out"
  echo "=== $name: ${parallel[*]}, cross-attention $cross, overlap $overlap"
  # graph_breaks: a collective breaking a compiled graph shows up in the log.
  XDIT_WAN_TP_CROSS_ATTENTION=$cross XDIT_TP_COMM_OVERLAP=$overlap XDIT_TP_OVERLAP_CHUNKS=$CHUNKS \
    TORCH_LOGS=graph_breaks xdit "${common[@]}" "${parallel[@]}" --output_directory "$out" > "$out/log.txt" 2>&1
  code=$?
  echo "=== $name exit=$code: $(grep -oE '(Profiling i|I)teration [0-9]+ completed in [0-9.]+s' "$out/log.txt" | tail -1)"
  [ $code -ne 0 ] && status=$code
done
exit $status
