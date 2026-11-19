#!/usr/bin/env bash
# FLUX.3 Sol-Attn end-to-end rows: klein / klein base x 2048² / 4096² x dense / BF16 Sol / FP8 Sol x
# prompt, Ulysses 8, torch.compile (blocks in reduce-overhead), dense AITER text stage.
#
# Runs in container jjuvonen-sol-mainline; outputs /output/flux3/<model><size>_<attn>_<prompt>[TAG]_s<seed>/.
# Rows with an image are skipped.
#
# Usage: tools/run_flux3.sh MODEL SIZE ATTN PROMPT [...]   one row, e.g. "b 4k fp8sol hands"
#        tools/run_flux3.sh matrix                          every model/size/attn/prompt
#   env: SRC (checkout, default /app/xDiT), SEED, TAG, EXTRA, BRR_ARGS (e.g. "--node" for timing)
set -uo pipefail

CONTAINER=${CONTAINER:-jjuvonen-sol-mainline}
SRC=${SRC:-/app/xDiT}
BRR_ARGS=${BRR_ARGS:---gpus 8 --exclusive}
SEED=${SEED:-1234}
TAG=${TAG:-}
EXTRA=${EXTRA:-}

declare -A PROMPTS=(
  [sign]="A weathered wooden sign outside a small bakery reads 'FRESH BREAD DAILY - OPEN 7AM' in hand-painted white letters. Morning sunlight on a red brick wall, potted geraniums below the sign, shallow depth of field."
  [hands]="Close-up portrait of an elderly woman with deep wrinkles holding a steaming cup of tea in both hands, wearing a cream knitted wool sweater. Soft window light from the left, detailed skin texture, fingers and fabric."
  [street]="A crowded night market street in Taipei, dozens of people walking between food stalls, glowing neon signs with Chinese characters, steam rising from cooking pots, wet pavement reflecting the lights."
  [macro]="Macro photograph of dew drops on a spider web strung between red autumn leaves and a grey knitted scarf. Fine wool fibers, leaf veins and individual droplets visible, soft golden morning backlight."
)
declare -A MODELS=([k]="FLUX.3-image-klein" [b]="FLUX.3-image-klein-base")
declare -A SIZES=([2k]=2048 [4k]=4096)
declare -A ATTN=(
  [dense]="--attention_backend aiter_bf16"
  [bf16sol]="--attention_backend aiter_bf16_sol --cross_attention_backend aiter"
  [fp8sol]="--attention_backend aiter_fp8_sol --cross_attention_backend aiter"
  [fp8sol_ramp]="--attention_backend aiter_fp8_sol --cross_attention_backend aiter --solattn_beta_schedule=0.5:-0.5"
)

run_row() {  # model size attn prompt
  local m=$1 s=$2 a=$3 p=$4
  local name="${m}${s}_${a}_${p}${TAG}_s${SEED}"
  local out=/output/flux3/$name host=$HOME/sol_mainline/output/flux3/$name
  if compgen -G "$host/*.png" > /dev/null; then echo "skip $name"; return; fi
  echo "run $name"
  docker exec "$CONTAINER" bash -c "mkdir -p $out && cd $SRC && export PYTHONPATH=$SRC && ulimit -c 0 && \
    /opt/brr/brr run $BRR_ARGS -- xdit --model ${MODELS[$m]} --prompt \"${PROMPTS[$p]}\" --seed $SEED \
    --height ${SIZES[$s]} --width ${SIZES[$s]} --ulysses_degree 8 --use_torch_compile \
    --warmup_calls 0 --num_iterations 1 ${ATTN[$a]} $EXTRA --output_directory $out > $out/out.log 2>&1"
  echo "  rc=$? $(grep -hoE 'Average time over [0-9]+ runs: [0-9.]+s' $host/out.log | tail -1)"
}

if [[ ${1:-} == matrix ]]; then
  for m in k b; do for s in 2k 4k; do for a in dense bf16sol fp8sol; do for p in hands sign street macro; do
    run_row $m $s $a $p
  done; done; done; done
else
  while (( $# >= 4 )); do run_row "$1" "$2" "$3" "$4"; shift 4; done
fi
