#!/bin/bash
# One MiniMax-H3-Image run through the xDiT runner, inside the container.
# Usage: run.sh <out_subdir> [extra xdit args...]
#   env: SIZE (2048), PROMPT_FILE (default: the model card's T2I prompt), SEED (42)
out=/output/runs/$1
shift
size=${SIZE:-2048}
prompt=${PROMPT:-"An overhead photograph of a rustic farm-to-table spread on a weathered wooden table: a round loaf of sourdough on burlap, wedges of cheese on a dark board, a copper bowl of vegetables, vine tomatoes, olives and walnuts."}
mkdir -p "$out"
xdit --model MiniMax-H3-Image --height "$size" --width "$size" --seed "${SEED:-42}" \
  --prompt "$prompt" --output_directory "$out" "$@" 2>&1 | grep -v -E "Warning|warnings.warn|^W1[01]" | tee "$out/out.log" | tail -25
