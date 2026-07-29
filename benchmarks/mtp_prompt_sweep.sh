#!/usr/bin/env bash
# Validate MTP draft depth across prompt types after the narrow-N verify win
# (E28). Interleaved best-of-N to fight thermal/contention drift.
#
# Usage: benchmarks/mtp_prompt_sweep.sh [reps]
# Requires bash 3.2+ (macOS /bin/bash is fine — no associative arrays).
set -u

REPS="${1:-2}"
BIN=./target/release/llama-sinks
BASE="$HOME/Downloads/models/e2b/gemma-4-E2B-it-Q4_K_M.gguf"
MTP="$HOME/Downloads/models/e2b/mtp-gemma-4-E2B-it-F16.gguf"

# Prompt presets from mtp_bench_prompt(); max tokens keeps essay/explain bounded.
PROMPTS="bubble_sort:200 fibonacci:300 essay:400 explain:400 qa:250 json:128"

# Depth configs: 3 was previous bubble_sort optimum; 4 is default + E28 optimum;
# adaptive should track whichever accept regime the prompt lands in.
DEPTH_LABELS=("steps=3" "steps=4" "steps=4 adaptive")
DEPTH_ENVS=(
  "LLAMA_MTP_DRAFT_STEPS=3"
  "LLAMA_MTP_DRAFT_STEPS=4"
  "LLAMA_MTP_DRAFT_STEPS=4 LLAMA_MTP_ADAPTIVE=1"
)

printf '%-12s %-18s %8s  %7s  %8s  %6s  %6s   %s\n' \
  PROMPT CONFIG BEST ACCEPT TOK/FWD FWDS TOKS RUNS

for prompt_spec in $PROMPTS; do
  prompt="${prompt_spec%%:*}"
  max_tok="${prompt_spec#*:}"

  n=${#DEPTH_LABELS[@]}
  BEST=(); ACC=(); TPF=(); FWD=(); TOKS=(); RUNS=()
  i=0
  while [ $i -lt $n ]; do
    BEST[$i]=0; RUNS[$i]=""
    i=$((i + 1))
  done

  for rep in $(seq "$REPS"); do
    i=0
    while [ $i -lt $n ]; do
      label="${DEPTH_LABELS[$i]}"
      envs="${DEPTH_ENVS[$i]}"
      out=$(env $envs MTP_BACKEND=metal ATTENTION_KERNEL=auto \
        LLAMA_KV_CACHE_TYPE=q4_0 LLAMA_CTX_SIZE=8192 \
        "$BIN" --gpu "$BASE" --mtp "$MTP" \
        --prompt "$prompt" --max-tokens "$max_tok" 2>&1)
      tps=$(echo "$out" | sed -n 's/.*Throughput: \([0-9.]*\).*/\1/p')
      if [ -z "$tps" ]; then
        printf '  FAIL %s %s\n' "$prompt" "$label"
        i=$((i + 1))
        continue
      fi
      RUNS[$i]="${RUNS[$i]} $tps"
      if awk "BEGIN{exit !($tps > ${BEST[$i]})}"; then
        BEST[$i]=$tps
        ACC[$i]=$(echo "$out" | sed -n 's/.*Accepted: [0-9]* (\([0-9.]*\)%).*/\1/p')
        TPF[$i]=$(echo "$out" | sed -n 's|.*Tokens / main forward: \([0-9.]*\).*|\1|p')
        FWD[$i]=$(echo "$out" | sed -n 's/.*Main-model forwards: \([0-9]*\).*/\1/p')
        TOKS[$i]=$(echo "$out" | sed -n 's/.*Tokens: \([0-9]*\).*/\1/p')
      fi
      printf '  rep%s %-12s %-18s %s tok/s\n' "$rep" "$prompt" "$label" "$tps"
      i=$((i + 1))
    done
  done

  i=0
  while [ $i -lt $n ]; do
    printf '%-12s %-18s %8s  %6s%%  %8s  %6s  %6s  %s\n' \
      "$prompt" "${DEPTH_LABELS[$i]}" "${BEST[$i]}" "${ACC[$i]:-?}" \
      "${TPF[$i]:-?}" "${FWD[$i]:-?}" "${TOKS[$i]:-?}" "${RUNS[$i]}"
    i=$((i + 1))
  done
  echo
done
