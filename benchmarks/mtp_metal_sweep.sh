#!/usr/bin/env bash
# Sweep MTP draft-depth / adaptive / p_min on the Metal draft path.
#
# This machine drifts (thermals + background load), so configs are interleaved
# round-robin rather than run in blocks, and each is scored by its best run.
# Compare best-of-N across configs; means are contaminated by contention spikes.
#
# Usage: benchmarks/mtp_metal_sweep.sh [reps] [config-set]
#   config-set: all (default) | depth | adaptive
set -u

REPS="${1:-3}"
SET="${2:-all}"
BIN=./target/release/llama-sinks
BASE="$HOME/Downloads/models/e2b/gemma-4-E2B-it-Q4_K_M.gguf"
MTP="$HOME/Downloads/models/e2b/mtp-gemma-4-E2B-it-F16.gguf"

if [ "$SET" = depth ]; then
  # The interesting range: default 4 is too deep for this workload.
  CONFIGS=(
    "steps=4 (baseline):LLAMA_MTP_DRAFT_STEPS=4"
    "steps=2:LLAMA_MTP_DRAFT_STEPS=2"
    "steps=3:LLAMA_MTP_DRAFT_STEPS=3"
    "steps=4 adaptive:LLAMA_MTP_DRAFT_STEPS=4 LLAMA_MTP_ADAPTIVE=1"
  )
elif [ "$SET" = adaptive ]; then
  CONFIGS=(
    "steps=4 (baseline):LLAMA_MTP_DRAFT_STEPS=4"
    "steps=4 adaptive:LLAMA_MTP_DRAFT_STEPS=4 LLAMA_MTP_ADAPTIVE=1"
    "steps=6 adaptive:LLAMA_MTP_DRAFT_STEPS=6 LLAMA_MTP_ADAPTIVE=1"
    "steps=7 adaptive:LLAMA_MTP_DRAFT_STEPS=7 LLAMA_MTP_ADAPTIVE=1"
  )
else
  CONFIGS=(
    "steps=4 (baseline):LLAMA_MTP_DRAFT_STEPS=4"
    "steps=5:LLAMA_MTP_DRAFT_STEPS=5"
    "steps=6:LLAMA_MTP_DRAFT_STEPS=6"
    "steps=7:LLAMA_MTP_DRAFT_STEPS=7"
    "steps=7 adaptive:LLAMA_MTP_DRAFT_STEPS=7 LLAMA_MTP_ADAPTIVE=1"
    "steps=7 p_min=0.5:LLAMA_MTP_DRAFT_STEPS=7 LLAMA_MTP_P_MIN=0.5"
    "steps=7 p_min=0.75:LLAMA_MTP_DRAFT_STEPS=7 LLAMA_MTP_P_MIN=0.75"
  )
fi

declare -a BEST ACC TPF FWD RUNS
for i in "${!CONFIGS[@]}"; do BEST[$i]=0; RUNS[$i]=""; done

for rep in $(seq "$REPS"); do
  for i in "${!CONFIGS[@]}"; do
    label="${CONFIGS[$i]%%:*}"
    envs="${CONFIGS[$i]#*:}"
    out=$(env $envs MTP_BACKEND=metal ATTENTION_KERNEL=auto \
      LLAMA_KV_CACHE_TYPE=q4_0 LLAMA_CTX_SIZE=8192 \
      "$BIN" --gpu "$BASE" --mtp "$MTP" 2>&1)
    tps=$(echo "$out" | sed -n 's/.*Throughput: \([0-9.]*\).*/\1/p')
    [ -z "$tps" ] && continue
    RUNS[$i]="${RUNS[$i]} $tps"
    if awk "BEGIN{exit !($tps > ${BEST[$i]})}"; then
      BEST[$i]=$tps
      ACC[$i]=$(echo "$out" | sed -n 's/.*Accepted: [0-9]* (\([0-9.]*\)%).*/\1/p')
      TPF[$i]=$(echo "$out" | sed -n 's|.*Tokens / main forward: \([0-9.]*\).*|\1|p')
      FWD[$i]=$(echo "$out" | sed -n 's/.*Main-model forwards: \([0-9]*\).*/\1/p')
    fi
    printf '  rep%s %-22s %s tok/s\n' "$rep" "$label" "$tps"
  done
done

echo
printf '%-24s %8s  %7s  %8s  %6s   %s\n' CONFIG BEST ACCEPT TOK/FWD FWDS RUNS
for i in "${!CONFIGS[@]}"; do
  printf '%-24s %8s  %6s%%  %8s  %6s  %s\n' \
    "${CONFIGS[$i]%%:*}" "${BEST[$i]}" "${ACC[$i]:-?}" "${TPF[$i]:-?}" \
    "${FWD[$i]:-?}" "${RUNS[$i]}"
done
