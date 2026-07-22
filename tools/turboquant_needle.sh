#!/usr/bin/env bash
# P0 TurboQuant needle smoke — buries a fact in a long prompt, asks for recall.
# Usage:
#   ./tools/turboquant_needle.sh
# Optional: K_BITS=3 V_BITS=2 RW=128 CTX=4096 NGEN=64 MODEL=...
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${MODEL:-$HOME/Downloads/models/e2b/gemma-4-E2B-it-Q4_K_M.gguf}"
K_BITS="${K_BITS:-3}"
V_BITS="${V_BITS:-2}"
RW="${RW:-128}"
CTX="${CTX:-4096}"
NEEDLE="${NEEDLE:-AURORA-7749}"
NGEN="${NGEN:-64}"
BIN="${BIN:-$ROOT/target/release/llama-sinks}"

if [[ ! -x "$BIN" ]]; then
  echo "missing binary: $BIN (cargo build --release first)" >&2
  exit 1
fi

# Pad so the prompt is long enough that the needle is not "free" in short context.
# Rough: ~40 chars/sentence * repeats → a few k chars (tokenizer will map to tokens).
PROMPT="$(NEEDLE="$NEEDLE" python3 - <<'PY'
import os
needle = os.environ["NEEDLE"]
pad = ("The ocean currents redistribute heat across the planet. " * 120)
print(
    f"Context:\n{pad}\n"
    f"Important fact buried here: The secret project code name is {needle}.\n"
    f"{pad}\n"
    f"Question: What is the secret project code name? Reply with only the code name."
)
PY
)"

OUT=$(mktemp)
trap 'rm -f "$OUT"' EXIT

ATTENTION_KERNEL=specialized \
LLAMA_KV_CACHE_TYPE=turboquant \
TURBOQUANT_K_BITS="$K_BITS" \
TURBOQUANT_V_BITS="$V_BITS" \
TURBOQUANT_RESIDUAL_WINDOW="$RW" \
LLAMA_CTX_SIZE="$CTX" \
LLAMA_MAX_TOKENS="$NGEN" \
LLAMA_TEMPERATURE=0 \
LLAMA_MIN_P=0 \
LLAMA_PROMPT="$PROMPT" \
"$BIN" --gpu "$MODEL" 2>&1 | tee "$OUT"

# Only score model output after the templated turn (avoid matching needle in the prompt echo).
# generate_gemma4_gpu prints the full prompt first, then completion tokens.
if awk 'found{print} /<start_of_turn>model/{found=1}' "$OUT" | grep -q "$NEEDLE"; then
  echo "PASS needle=$NEEDLE K${K_BITS}/V${V_BITS} rw=$RW ctx=$CTX"
  exit 0
else
  echo "FAIL needle=$NEEDLE K${K_BITS}/V${V_BITS} rw=$RW ctx=$CTX" >&2
  echo "(If the model only wrote the essay, rebuild after LLAMA_PROMPT support landed.)" >&2
  exit 2
fi
