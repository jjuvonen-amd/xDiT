#!/bin/bash
# One MiniMax-H3-Image benchmark config inside the container.
# Usage: bench.sh <name> <nproc> <size> [extra runner args...]
# Writes /output/bench/<name>/{log,*.png,timings.json}; prints the summary lines.
name=$1 nproc=$2 size=$3
shift 3
out=/output/bench/$name
mkdir -p "$out"
prompt="An overhead photograph of a rustic farm-to-table spread on a weathered wooden table: a round loaf of sourdough on burlap, wedges of cheese on a dark board, a copper bowl of vegetables, vine tomatoes, olives and walnuts."
echo "loadavg start: $(cat /proc/loadavg)" > "$out/log"
TIMING=1 torchrun --nproc-per-node "$nproc" --master-port ${PORT:-29620} /app/xDiT/benchmarks/minimax_h3_image/xdit_run.py \
  --model MiniMax-H3-Image --height "$size" --width "$size" --seed 42 --prompt "$prompt" \
  --warmup_calls 1 --num_iterations 3 --output_directory "$out" "$@" >> "$out/log" 2>&1
rc=$?
echo "loadavg end: $(cat /proc/loadavg)" >> "$out/log"
echo "== $name (nproc $nproc, ${size}², $*) rc=$rc"
grep -E "\[timing\]|Iteration [0-9]+ completed|Average time|loadavg|Error|error:" "$out/log" | grep -v "^\[aiter\]" | tail -12
