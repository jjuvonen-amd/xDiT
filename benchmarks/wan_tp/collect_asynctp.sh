#!/usr/bin/env bash
# Collect torch AsyncTP (symm_mem fused AG+GEMM / GEMM+RS) results on one 8-GPU node:
# environment, tests, P2P bandwidth, per-site comm/GEMM benchmarks, Wan2.2 TP4/TP8
# profiles and 40-step timings. Run inside the xDiT container, from any directory:
#
#   bash benchmarks/wan_tp/collect_asynctp.sh
#
# Environment:
#   OUTPUT_ROOT  results directory (default /output/asynctp_results)
#   ATTN         --attention_backend for the Wan runs (default: the model's)
#   SKIP_TESTS=1 skip the unit and parity tests
#
# Takes about two hours. Every timing step first logs `rocm-smi --showpids` (or
# nvidia-smi) so results taken while other jobs used the GPUs can be spotted.
set -u
cd "$(dirname "$0")/../.." || exit 1

OUT=${OUTPUT_ROOT:-/output/asynctp_results}
mkdir -p "$OUT"
failures=()

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/collect.log"; }

# Record other processes on the GPUs; timings taken next to them are not comparable.
gpu_check() {
  local users=""
  if command -v rocm-smi >/dev/null; then
    users=$(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+ / && $2 != "gpuagent"')
  elif command -v nvidia-smi >/dev/null; then
    users=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader)
  fi
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

# 1. Environment. Stop here if the stack is not the one under test.
step env python -c "
import torch, torch.distributed._symmetric_memory as s
print('torch', torch.__version__, 'hip', torch.version.hip, 'cuda', torch.version.cuda)
print('devices', torch.cuda.device_count(), torch.cuda.get_device_name())
print('native ROCm AsyncTP entry point:', hasattr(s, '_fused_all_gather_matmul_native_rocm'))
" || exit 1
{
  dpkg -l 2>/dev/null | awk '/amdrocm-runtime-dev[0-9]/{print $2, $3}'
  git log --oneline -1
} >> "$OUT/env.log" 2>&1
cat "$OUT/env.log"

# 2. Tests. Failures are recorded; the measurements below still run.
if [ "${SKIP_TESTS:-0}" != 1 ]; then
  step unit_tests env CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= python -m pytest -q -p no:cacheprovider tests/unit -m "not accelerator"
  step parity_tests python -m pytest -q -p no:cacheprovider tests/integration/accelerator/model_executor/test_wan_tensor_parallel.py
fi

# 3. P2P read bandwidth: one peer vs all peers in turn vs all at once.
for p in 8 4; do
  gpu_check "p2p_tp$p"
  step "p2p_tp$p" torchrun --nproc_per_node "$p" benchmarks/wan_tp/p2p_bandwidth.py --output-dir "$OUT/p2p"
done

# 4. Per-site comm + GEMM: RCCL baseline, chunked overlap, torch fused (decomposed path at Wan shapes).
for p in 2 4 8; do
  gpu_check "fusion_bench_tp$p"
  step "fusion_bench_tp$p" torchrun --nproc_per_node "$p" benchmarks/tp_comm_gemm_overlap.py --symm-mem \
    --output-dir "$OUT/fusion_bench"
done
# The native AsyncTP kernel only takes 2048 < gathered rows <= 4096: TP8, 512 rows per GPU, with and without it.
gpu_check native_window
step native_window_on env TORCH_SYMM_MEM_ENABLE_NATIVE_ASYNC_TP=1 torchrun --nproc_per_node 8 \
  benchmarks/tp_comm_gemm_overlap.py --symm-mem --seq 4096 --sites qkv ffn_up --output-dir "$OUT/fusion_bench_native_on"
step native_window_off torchrun --nproc_per_node 8 \
  benchmarks/tp_comm_gemm_overlap.py --symm-mem --seq 4096 --sites qkv ffn_up --output-dir "$OUT/fusion_bench_native_off"

# 5. Wan2.2 T2V 720p/81f: profiled 4-step iteration per config, in one session.
gpu_check profiles
step profiles env OUTPUT_DIR="$OUT/profiles" bash benchmarks/wan_tp/run_profiles.sh \
  sp8 tp8_xrepl tp8_xrepl_fused tp8_xrepl_ov tp4_xrepl tp4_xrepl_fused tp4_xrepl_ov
step summary python benchmarks/wan_tp/summarize_traces.py --root "$OUT/profiles"

# 6. Full 40-step timings, warmed up.
gpu_check timed40
step timed40 env OUTPUT_DIR="$OUT/timed40" PROFILE=0 STEPS=40 bash benchmarks/wan_tp/run_profiles.sh \
  sp8 tp8_xrepl tp8_xrepl_fused tp8_xrepl_ov tp4_xrepl_fused tp4_xrepl_ov

log "== done"
grep -hE "^=== .* exit=" "$OUT/profiles.log" "$OUT/timed40.log" 2>/dev/null | tee -a "$OUT/collect.log"
if [ ${#failures[@]} -gt 0 ]; then
  log "failed steps: ${failures[*]}"
  exit 1
fi
log "all steps passed"
