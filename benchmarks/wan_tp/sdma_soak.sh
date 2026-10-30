#!/usr/bin/env bash
# Soak test for the SDMA TP ops (XDIT_TP_COMM_OVERLAP=sdma with XDIT_TP_SDMA_REDUCE_SCATTER=on) on
# one 8-GPU ROCm node. Until DURATION_MIN is used up, alternates:
#   stress  benchmarks/wan_tp/sdma_flag_stress.py, all modes, every other time next to RCCL traffic
#   wan     one xdit process running ITERATIONS Wan2.2-T2V TP8 720p/81f 40-step generations with the
#           same seed and --determinism_check 1 (an iteration whose video differs from the first's
#           is logged as diverged)
# while each GPU's VRAM use is sampled every 30 s (rocm-smi, or amd-smi in containers). Run inside the xDiT container:
#
#   DURATION_MIN=120 bash benchmarks/wan_tp/sdma_soak.sh
#
# Environment: DURATION_MIN (default 120), ITERATIONS per wan phase (default 10), OUTPUT_ROOT
# (default /output/sdma_soak), CHUNKS (default 4), ATTN. At the end it writes summary.md: stress
# mismatches and waits that gave up, failed or aborted runs, diverged iterations, the spread of
# iteration times within each wan phase, and each GPU's VRAM range within each wan phase. The soak
# passes when the first four are zero, iteration times stay within 2 % of each phase's first, and
# VRAM is flat within a phase.
set -u
cd "$(dirname "$0")/../.." || exit 1

OUT=${OUTPUT_ROOT:-/output/sdma_soak}
DURATION_MIN=${DURATION_MIN:-120}
ITERATIONS=${ITERATIONS:-10}
export CHUNKS=${CHUNKS:-4}
mkdir -p "$OUT"
deadline=$(($(date +%s) + DURATION_MIN * 60))
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/soak.log"; }

phase=none
# VRAM used per GPU in bytes, from rocm-smi or (in containers that only ship it) amd-smi.
vram_json() {
  if command -v rocm-smi >/dev/null; then
    rocm-smi --showmeminfo vram --json 2>/dev/null | python -c "
import json, sys
d = json.load(sys.stdin)
print(json.dumps({k: int(v.get('VRAM Total Used Memory (B)', 0)) for k, v in d.items() if k.startswith('card')}))"
  else
    amd-smi metric --mem-usage --json 2>/dev/null | python -c "
import json, sys
d = json.load(sys.stdin)
print(json.dumps({f\"gpu{g['gpu']}\": int(g['mem_usage']['used_vram']['value']) << 20 for g in d['gpu_data']}))"
  fi
}

sample_vram() {
  while true; do
    echo "{\"t\": $(date +%s), \"phase\": \"$(cat "$OUT/phase")\", \"used\": $(vram_json)}" >> "$OUT/vram.jsonl"
    sleep 30
  done
}
echo none > "$OUT/phase"
sample_vram &
sampler=$!
trap 'kill $sampler 2>/dev/null' EXIT

round=0
while [ "$(date +%s)" -lt "$deadline" ]; do
  round=$((round + 1))
  bg=""
  [ $((round % 2)) = 0 ] && bg=--background-rccl
  echo "stress_$round" > "$OUT/phase"
  log "== stress round $round ${bg:-(no background traffic)}"
  torchrun --nproc_per_node 8 benchmarks/wan_tp/sdma_flag_stress.py $bg --output-dir "$OUT/stress_$round" \
    > "$OUT/stress_$round.log" 2>&1
  log "   exit $?"
  [ "$(date +%s)" -lt "$deadline" ] || break
  echo "wan_$round" > "$OUT/phase"
  log "== wan round $round: $ITERATIONS generations"
  OUTPUT_DIR="$OUT/wan_$round" PROFILE=0 STEPS=40 XDIT_TP_SDMA_REDUCE_SCATTER=on \
    EXTRA_ARGS="--num_iterations $ITERATIONS --determinism_check 1" \
    bash benchmarks/wan_tp/run_profiles.sh tp8_xrepl_sdma > "$OUT/wan_$round.log" 2>&1
  log "   $(grep -E '^=== .* exit=' "$OUT/wan_$round.log" | tail -1)"
done
echo none > "$OUT/phase"

python - "$OUT" <<'EOF'
import glob, json, re, statistics, sys
from pathlib import Path

out = Path(sys.argv[1])
lines = ["# SDMA soak", ""]
mismatches = gave_up = 0
for md in sorted(out.glob("stress_*/sdma_flag_stress_tp*.md")):
    for row in md.read_text().splitlines():
        cells = [c.strip() for c in row.split("|")[1:-1]]
        if len(cells) == 7 and cells[3].isdigit():
            mismatches += int(cells[4])
            gave_up += int(cells[5])
failed = [p.name for p in out.glob("*.log") if re.search(r"exit (?!0\b)\d+|exit=[1-9]", p.read_text())]
diverged = sum(len(re.findall(r"diverged", p.read_text())) for p in out.glob("wan_*/tp8_xrepl_sdma/log.txt"))
lines += [
    f"- stress rounds: {len(list(out.glob('stress_*.log')))}; mismatched elements: {mismatches}; flag waits that gave up: {gave_up}",
    f"- runs that failed or aborted: {failed or 'none'}",
    f"- diverged iterations (determinism check): {diverged}",
    "",
    "| wan round | iterations | first s | min s | max s | max spread vs first |",
    "|---|---|---|---|---|---|",
]
for log in sorted(out.glob("wan_*/tp8_xrepl_sdma/log.txt"), key=lambda p: int(p.parts[-3].split("_")[1])):
    times = [float(t) for t in re.findall(r"Iteration \d+ completed in ([0-9.]+)s", log.read_text())]
    if times:
        spread = max(abs(t - times[0]) for t in times) / times[0]
        lines.append(f"| {log.parts[-3]} | {len(times)} | {times[0]:.1f} | {min(times):.1f} | {max(times):.1f} | {spread:.1%} |")
samples = [json.loads(s) for s in (out / "vram.jsonl").read_text().splitlines() if s.strip()] if (out / "vram.jsonl").exists() else []
lines += ["", "VRAM used within each wan round, per GPU (MiB, min-max over samples taken mid-round):", ""]
for phase in sorted({s["phase"] for s in samples if s["phase"].startswith("wan_")}, key=lambda p: int(p.split("_")[1])):
    group = [s for s in samples if s["phase"] == phase][1:]  # skip the sample taken while the model loaded
    if not group:
        continue
    ranges = {card: (min(s["used"][card] for s in group) >> 20, max(s["used"][card] for s in group) >> 20) for card in group[0]["used"]}
    lines.append(f"- {phase}: " + ", ".join(f"{card} {lo}-{hi}" for card, (lo, hi) in sorted(ranges.items())))
text = "\n".join(lines) + "\n"
(out / "summary.md").write_text(text)
print(text)
EOF
