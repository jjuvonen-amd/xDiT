#!/usr/bin/env bash
# Collect results for the SDMA all-gather + GEMM (XDIT_TP_COMM_OVERLAP=sdma) on one 8-GPU
# ROCm node: op and Wan parity tests, SDMA engine-plan bandwidth, the per-site benchmark at
# Wan2.2 widths, a sweep over wider GEMMs (where the GEMM can outlast the gather), and Wan2.2
# TP8/TP4 profiles and 40-step timings next to no overlap and chunked overlap. Run inside the
# xDiT container:
#
#   bash benchmarks/wan_tp/collect_sdma.sh
#
# Environment:
#   OUTPUT_ROOT  results directory (default /output/sdma_results)
#   ATTN         --attention_backend for the Wan runs (default: the model's)
#   CHUNKS       pieces / chunks for the Wan runs, chunked and sdma alike (default 4)
#
# If the op tests fail with the GPU-released trigger but pass with the host one, the
# later steps run with XDIT_TP_SDMA_TRIGGER=host (logged). Every timing step first logs
# `rocm-smi --showpids`, so results taken while other jobs used the GPUs can be spotted.
set -u
cd "$(dirname "$0")/../.." || exit 1

OUT=${OUTPUT_ROOT:-/output/sdma_results}
export CHUNKS=${CHUNKS:-4}
mkdir -p "$OUT"
failures=()

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/collect.log"; }

gpu_check() {
  local users
  users=$(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+ / && $2 != "gpuagent"')
  printf -- '--- %s before %s\n%s\n' "$(date -Is)" "$1" "$users" >> "$OUT/gpu_users.log"
  if [ -n "$users" ]; then
    log "WARNING: other processes are using the GPUs before $1 (see gpu_users.log); its timings are not comparable"
  fi
}

# step NAME CMD...: run CMD, log to NAME.log, record failures without stopping.
step() {
  local name=$1
  shift
  log "== $name"
  "$@" > "$OUT/$name.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    failures+=("$name (exit $rc)")
    log "   FAILED (exit $rc), see $name.log"
  fi
  return $rc
}

log "results in $OUT"

# 1. Environment.
step env python -c "
import torch
print('torch', torch.__version__, 'hip', torch.version.hip)
print('devices', torch.cuda.device_count(), torch.cuda.get_device_name())
" || exit 1
git -c safe.directory="$PWD" log --oneline -1 >> "$OUT/env.log" 2>&1
cat "$OUT/env.log"

# 2. Tests. The op test covers both triggers; pick the one that works for the runs below.
export XDIT_TP_SDMA_TRIGGER=gpu
OP_TEST=tests/integration/accelerator/model_executor/layers/test_sdma_all_gather.py
if ! step op_test_gpu_trigger python -m pytest -q -p no:cacheprovider "$OP_TEST" -k gpu; then
  if step op_test_host_trigger python -m pytest -q -p no:cacheprovider "$OP_TEST" -k host; then
    export XDIT_TP_SDMA_TRIGGER=host
    log "   the GPU-released trigger failed and the host one passed: using XDIT_TP_SDMA_TRIGGER=host"
  else
    log "   the SDMA op fails with both triggers; stopping"
    exit 1
  fi
fi
step parity_tests python -m pytest -q -p no:cacheprovider \
  tests/integration/accelerator/model_executor/test_wan_tensor_parallel.py -k sdma

# 3. Raw 7-peer SDMA bandwidth on torch's buffers: reads (pull) vs writes (push), the latter on
#    the read-preferred and on the write-preferred engines (what the op's push uses).
gpu_check copy_direction
step copy_direction torchrun --nproc_per_node 8 benchmarks/wan_tp/symm_mem_copy_paths.py \
  --paths hsa_local_sdma_fixed,hsa_local_sdma_push,hsa_local_sdma_push_out --output-dir "$OUT/copy_direction"

# 4. Per site at Wan2.2 TP8 widths, and 5. wider column-parallel GEMMs (AG sites), for 2, 4
#    and 8 pieces next to chunked overlap with as many chunks; push (the default), and pull
#    once for comparison.
gpu_check fusion_bench_wan
step fusion_bench_wan torchrun --nproc_per_node 8 benchmarks/tp_comm_gemm_overlap.py --sdma \
  --sdma-trigger "$XDIT_TP_SDMA_TRIGGER" --chunks 2 4 8 --output-dir "$OUT/fusion_bench_wan"
step fusion_bench_wan_pull torchrun --nproc_per_node 8 benchmarks/tp_comm_gemm_overlap.py --sdma \
  --sdma-direction pull --sdma-trigger "$XDIT_TP_SDMA_TRIGGER" --chunks 2 4 8 --output-dir "$OUT/fusion_bench_wan_pull"
for shape in 5120:27648 5120:55296 8192:57344; do
  h=${shape%:*} f=${shape#*:}
  gpu_check "sweep_h${h}_f${f}"
  step "sweep_h${h}_f${f}" torchrun --nproc_per_node 8 benchmarks/tp_comm_gemm_overlap.py --sdma \
    --sdma-trigger "$XDIT_TP_SDMA_TRIGGER" --chunks 2 4 8 --hidden "$h" --ffn "$f" --sites qkv ffn_up \
    --output-dir "$OUT/sweep_h${h}_f${f}"
done

# 6. Wan2.2 T2V 720p/81f: profiled 4-step iteration per config, in one session. The SDMA
#    copies are ROCr operations, which torch's profiler labels as host-to-device copies.
gpu_check profiles
step profiles env OUTPUT_DIR="$OUT/profiles" bash benchmarks/wan_tp/run_profiles.sh \
  tp8_xrepl tp8_xrepl_ov tp8_xrepl_sdma
step summary python benchmarks/wan_tp/summarize_traces.py --root "$OUT/profiles"

# 7. Full 40-step timings, warmed up.
gpu_check timed40
step timed40 env OUTPUT_DIR="$OUT/timed40" PROFILE=0 STEPS=40 \
  bash benchmarks/wan_tp/run_profiles.sh tp8_xrepl tp8_xrepl_ov tp8_xrepl_sdma tp4_xrepl_ov tp4_xrepl_sdma

log "== done (trigger: $XDIT_TP_SDMA_TRIGGER, chunks: $CHUNKS)"
grep -hE "^=== .* exit=" "$OUT"/profiles*.log "$OUT"/timed40*.log 2>/dev/null | tee -a "$OUT/collect.log"
if [ ${#failures[@]} -gt 0 ]; then
  log "failed steps: ${failures[*]}"
  exit 1
fi
log "all steps passed"
