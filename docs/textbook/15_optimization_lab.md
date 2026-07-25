# 15 — Optimization Lab Method

## Purpose

Teach **how this repo investigates performance**, not a forever-true leaderboard.
Authoritative numbers and corpses of ideas: [`../../AGENTS.md`](../../AGENTS.md).

---

## Golden rules

1. **One change at a time** + cool machine / second run (thermal).  
2. Report **ctx length** with tok/s (25 vs 200 vs 4k are different sports).  
3. Separate **prefill** vs **decode** benches.  
4. Correctness first — fast garbage is worthless (`AGENTS` #11, #15).  
5. Kill ideas with data; log them so you don’t revive them.

---

## Benchmark commands

```bash
# Decode
LLAMA_KV_CACHE_TYPE=q4_0 ATTENTION_KERNEL=auto \
  ./target/release/llama-sinks --gpu MODEL.gguf \
  --bench-decode --bench-decode-tokens 25,200

# Prefill
LLAMA_KV_CACHE_TYPE=q4_0 LLAMA_MAX_PREFILL_SEQ=4096 \
  ./target/release/llama-sinks --gpu MODEL.gguf \
  --bench-prefill --bench-prefill-tokens 2048,4096
```

Microbenches: `--bench-matvec`, `--bench-mul-mm`, `--bench-mv-ext`.

Ablation: `PROFILE_ABLATE=...`, `PREFILL_TIMING=1`, `PROFILE_DISPATCHES=1`.

---

## Experiment taxonomy (from AGENTS)

| Class | Examples | Lesson |
|-------|----------|--------|
| Overhead | dispatch counting | Real but not the ctx gap |
| Kernel geometry | KQ_NR0, NSG | Empirically tune |
| Algorithm switch | flash vs tile-free, MWG | Ctx-dependent winners |
| Fusion | FUSED_DECODE, mega | Often wash if bandwidth-bound |
| Routing | ATTENTION_KERNEL=auto | Best profile so far |
| Correctness×perf | KV append, GQA | Perf without quality is fake |
| Prefill dtype | PLE f16 | Huge wins possible |
| MTP packaging | tiled verify, ext matvec | Product-level tok/s |

---

## Current best decode posture (as of AGENTS summary)

```text
ATTENTION_KERNEL=auto
LLAMA_KV_CACHE_TYPE=q4_0
fused decode executor
```

Expect ~fused short-ctx peak and flatter long-ctx than specialized-only; still a
few tok/s under llama.cpp at mid context — gap not closed.

---

## How to add your own experiment page

Copy this template into `AGENTS.md`:

```markdown
## N. Title
**What**: …
**Result**: … tok/s @ … ctx …
**Conclusion**: keep / revert / conditional
```

---

## Reading bottlenecks

| If… | Look at… |
|-----|----------|
| tok/s falls with ctx | Attention bandwidth / kernel family |
| Short ctx already slow | Matvec / fusion / dispatch |
| Prefill ≪ llama @4k | mul_mm, flash NSG, PLE dtype |
| MTP ≪ baseline | Verify phase, accept rate |
| Multi-user latency | slots, queue, prefill tick |

---

## Checklist

- [ ] Run one decode bench and interpret 25 vs 200.  
- [ ] Name the hybrid auto win and its KV bug.  
- [ ] Know prefill’s biggest historical win (PLE f16).  
- [ ] Can reject a “faster” kernel that fails Hello/ZEBRA tests.

**Next:** [16_glossary_and_drills.md](16_glossary_and_drills.md)
