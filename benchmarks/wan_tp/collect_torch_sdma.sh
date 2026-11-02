#!/usr/bin/env bash
# Collect results for torch's symm_mem fused all-gather + matmul and matmul + reduce-scatter on
# their SDMA path, i.e. a torch patched with patches/pytorch and run with
# TORCH_SYMM_MEM_ENABLE_ROCM_SDMA=1, on one 8-GPU ROCm node:
#   1. env: torch, GPU arch, whether the patch is applied (applies it if not);
#   2. PyTorch's own AsyncTP tests from test/distributed/test_symmetric_memory.py, patched like
#      torch (its base commit is fetched from GitHub), with the SDMA path off and on, and the new
#      AsyncTPRocmSdmaTest; then the Wan parity tests that run the stock ops on that path;
#   3. the per-site benchmark at Wan2.2 TP8 widths: stock symm_mem, symm_mem on the SDMA path
#      (eager and replayed from a CUDA graph), xfuser's own SDMA ops and chunked overlap;
#   4. Wan2.2 T2V TP8 720p/81f, 40 steps: torch's symm_mem ops (XDIT_TP_COMM_OVERLAP=symm_mem)
#      with the SDMA path off and on, next to chunked overlap and xfuser's SDMA ops with the SDMA
#      reduce-scatter. E2E=0 skips this step.
# Run inside the xDiT container:
#
#   bash benchmarks/wan_tp/collect_torch_sdma.sh
#
# Environment: OUTPUT_ROOT (default /output/torch_sdma_results), CHUNKS (pieces per rank for every
# SDMA and chunked run, default 4), E2E (default 1). Sets TORCH_SYMM_MEM_ROCM_SDMA_ALLOW_UNTESTED=1
# and XDIT_TP_SDMA_ALLOW_UNTESTED=1, so it also runs on GPU architectures the paths are not yet
# validated on (the arch is logged).
set -u
cd "$(dirname "$0")/../.." || exit 1

OUT=${OUTPUT_ROOT:-/output/torch_sdma_results}
export CHUNKS=${CHUNKS:-4}
export TORCH_SYMM_MEM_ROCM_SDMA_ALLOW_UNTESTED=1 XDIT_TP_SDMA_ALLOW_UNTESTED=1
PATCH_FILE=patches/pytorch/rocm_sdma_async_tp.patch
mkdir -p "$OUT"
failures=()

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/collect.log"; }

# Processes on the GPUs other than the gpuagent monitor (none of ours run between steps).
gpu_users() {
  if command -v rocm-smi >/dev/null; then
    rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+[ \t]/ && $2 != "gpuagent"'
  elif command -v amd-smi >/dev/null; then
    amd-smi process --json 2>/dev/null | python -c "
import json, sys
for g in json.load(sys.stdin):
    for p in g['process_list']:
        info = p.get('process_info', p)
        if isinstance(info, dict) and 'pid' in info:
            print(f\"gpu {g['gpu']}: pid {info['pid']}, {info['mem_usage']['value']} B\")"
  else
    echo "UNCHECKED: neither rocm-smi nor amd-smi is available"
  fi
}

gpu_check() {
  local users
  users=$(gpu_users)
  printf -- '--- %s before %s\n%s\n' "$(date -Is)" "$1" "${users:-(none)}" >> "$OUT/gpu_users.log"
  [ -z "$users" ] || log "WARNING: before $1: $users (see gpu_users.log); its timings may not be comparable"
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

# 1. Environment and the patch.
step env python -c "
import torch
print('torch', torch.__version__, 'hip', torch.version.hip)
print('devices', torch.cuda.device_count(), torch.cuda.get_device_name())
print('arch', torch.cuda.get_device_properties(0).gcnArchName)
" || exit 1
git -c safe.directory="$PWD" log --oneline -1 >> "$OUT/env.log" 2>&1
if ! bash patches/pytorch/apply_to_installed_torch.sh --check | grep -q "^applied"; then
  step apply_patch bash patches/pytorch/apply_to_installed_torch.sh || exit 1
fi
bash patches/pytorch/apply_to_installed_torch.sh --check >> "$OUT/env.log"
cat "$OUT/env.log"

# 2. PyTorch's test file at the patch's base commit, with the patch's test part applied. For a
#    torch built from an older commit, test helpers it lacks are stubbed in that copy only.
upstream_tests() {
  local base dir
  base=$(sed -n 's/^base-commit: //p' "$PATCH_FILE")
  dir="$OUT/upstream"
  mkdir -p "$dir/test/distributed"
  curl -sfL "https://raw.githubusercontent.com/pytorch/pytorch/$base/test/distributed/test_symmetric_memory.py" \
    -o "$dir/test/distributed/test_symmetric_memory.py" || return 1
  python - "$PATCH_FILE" <<'EOF' | patch -p1 -d "$dir" || return 1
import re, sys
keep = False
for line in open(sys.argv[1]):
    if line.startswith("diff --git "):
        keep = re.match(r"diff --git a/test/", line) is not None
    if keep:
        sys.stdout.write(line)
EOF
  python - "$dir/test/distributed/test_symmetric_memory.py" <<'EOF'
import sys
import torch.testing._internal.common_utils as cu
path = sys.argv[1]
text = open(path).read()
stubs = []
if not hasattr(cu, "requires_accelerator"):
    text = text.replace("    requires_accelerator,\n", "", 1)
    stubs.append("requires_accelerator = lambda f: f")
if not hasattr(cu, "requires_accelerator_p2p_access"):
    text = text.replace(
        "    requires_accelerator_p2p_access,\n", "    requires_cuda_p2p_access as requires_accelerator_p2p_access,\n", 1
    )
if stubs:
    text = text.replace("\ntest_contexts = ", "\n" + "\n".join(stubs) + "  # stub: not in this torch build\ntest_contexts = ", 1)
open(path, "w").write(text)
print("stubbed for this torch build:", stubs or "nothing")
EOF
}
python -c "import expecttest" 2>/dev/null || pip install -q expecttest > /dev/null 2>&1
if step upstream_test_file upstream_tests; then
  TEST_FILE="$OUT/upstream/test/distributed/test_symmetric_memory.py"
  gpu_check upstream_tests
  step asynctp_tests_sdma_off python -m pytest -q -p no:cacheprovider "$TEST_FILE" -k "AsyncTPTest and not Rocm"
  step asynctp_tests_sdma_on env TORCH_SYMM_MEM_ENABLE_ROCM_SDMA=1 \
    python -m pytest -q -p no:cacheprovider "$TEST_FILE" -k "AsyncTPTest and not Rocm"
  step asynctp_rocm_sdma_tests python -m pytest -q -p no:cacheprovider "$TEST_FILE" -k AsyncTPRocmSdmaTest
fi
step wan_parity_tests python -m pytest -q -p no:cacheprovider \
  tests/integration/accelerator/model_executor/test_wan_tensor_parallel.py -k patched_torch

# 3. Per site at Wan2.2 TP8 widths.
gpu_check fusion_bench_wan
step fusion_bench_wan torchrun --nproc_per_node 8 benchmarks/tp_comm_gemm_overlap.py --symm-mem \
  --symm-mem-sdma --graph --sdma --chunks 2 4 8 --output-dir "$OUT/fusion_bench_wan"

# 4. Wan2.2 T2V TP8, 40 steps.
if [ "${E2E:-1}" = 1 ]; then
  gpu_check timed40
  step timed40 env OUTPUT_DIR="$OUT/timed40" PROFILE=0 STEPS=40 \
    bash benchmarks/wan_tp/run_profiles.sh tp8_xrepl_ov tp8_xrepl_fused
  step timed40_torch_sdma env OUTPUT_DIR="$OUT/timed40_torch_sdma" PROFILE=0 STEPS=40 \
    TORCH_SYMM_MEM_ENABLE_ROCM_SDMA=1 TORCH_SYMM_MEM_ROCM_SDMA_CHUNKS="$CHUNKS" \
    bash benchmarks/wan_tp/run_profiles.sh tp8_xrepl_fused
  step timed40_xfuser_sdma env OUTPUT_DIR="$OUT/timed40_xfuser_sdma" PROFILE=0 STEPS=40 \
    XDIT_TP_SDMA_REDUCE_SCATTER=on bash benchmarks/wan_tp/run_profiles.sh tp8_xrepl_sdma
fi

log "== done (chunks: $CHUNKS)"
grep -hE "^=== .* exit=" "$OUT"/timed40*.log 2>/dev/null | tee -a "$OUT/collect.log"
if [ ${#failures[@]} -gt 0 ]; then
  log "failed steps: ${failures[*]}"
  exit 1
fi
log "all steps passed"
