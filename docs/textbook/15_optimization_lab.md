# 15 — The Optimization Lab: How to Not Fool Yourself

`AGENTS.md` is a log of 15 numbered experiments plus a dozen more, and the
striking thing about it is how many are **failures**: no effect, worse, or
"correct but within noise." That is not a sign of bad work. That is what
performance work looks like when it is done honestly.

This chapter is the methodology that produced those entries, and the tooling
you use to run your own.

---

## Part A — The rules

**1. Never optimize without a measurement first.** The four-way branch in the
MLP (Ch 08) exists because someone measured each. `fastMathEnabled` (#2) took
five minutes to try and had *zero* effect, because the kernel is
bandwidth-bound — a fact a measurement would have predicted for free.

**2. Predict before you measure.** Write down the expected speedup and the
mechanism. If you cannot state the mechanism ("this removes 14 MB of weight
reads per token"), you are guessing. When the prediction misses, you have
learned something about the machine; when you had no prediction, you have
learned nothing.

**3. One variable per run.** Every env var in this codebase exists so that a
single change can be isolated at runtime instead of being tangled into a
rebuild.

**4. Run it twice, cold and warm.** M1 Pro throttles. `AGENTS.md` E22/E23
explicitly note "cool exact-4096" because a warm machine reads 5–10% slower.
A 3% "improvement" measured warm-then-cold is noise with a story attached.

**5. Write down the failures.** The reason `AGENTS.md` is valuable is that it
stops you from re-running experiment #3 (`KQ_NR0=2`, 42.8 vs 46.1 — worse) for
the third time. Failed experiments are the expensive knowledge.

**6. Check correctness on every perf change.** #11 hit **59.6 tok/s** — a 6
tok/s win over llama.cpp — while emitting `くださいまして` forever. Speed
without a correctness gate is a random number generator.

---

## Part B — The measurement toolkit

### B.1 End-to-end throughput

```bash
cargo run --release -- --bench-decode --bench-decode-tokens 200
```

Generation length matters enormously, because the gap this project is chasing
*grows with context*:

```text
25 tok  → ~54.7 tok/s   (ties llama.cpp)
200 tok → ~47.9 tok/s   (−12%)
400 tok → ~41   tok/s
```

Always report the token count with the number. "47 tok/s" alone is not a
measurement.

### B.2 Phase timing

```bash
PROFILE_PHASES=1   # per-phase host-side timing (commits + waits between phases)
PROFILE_GPU=1      # GPU timestamps via encoder marks
PROFILE_DECODE=1   # decode-specific breakdown
PROFILE_DISPATCHES=1  # count dispatches per token
```

`PROFILE_PHASES` inserts `end_encoding` + `commit` + `wait_until_completed`
between phases (you can see it in the decode path around the PLE pre-pass).
That *changes* total time — it serializes what would otherwise overlap. Use it
for **relative** attribution, never for absolute throughput.

### B.3 Ablation — the most underrated tool

```bash
PROFILE_ABLATE=SKIP_mlp    # skip MLP entirely; output is garbage, timing is real
PROFILE_ABLATE=SKIP_ple
PROFILE_ABLATE=SKIP_attn
PROFILE_ABLATE=SKIP_all    # the floor: command-buffer plumbing only
```

Deleting work is the cheapest possible way to price it, and it needs no
instrumentation that could perturb what you are measuring. Always get
`SKIP_all` first — that is the baseline your Δ values are differences from
(135 ms at 4k prefill; see Ch 10 Part H).

This is how E22 found the PLE problem. Nobody suspected the PLE block until
`SKIP_ple` showed **1555 ms** of a 4k prefill disappearing.

### B.4 Comparing against the reference

```bash
./benchmarks/compare_llama.sh
python3 benchmarks/benchmark_local.py
python3 benchmarks/prefill_benchmark.py
```

llama.cpp is the yardstick: 53.9 tok/s decode, 593.9 tok/s pp4096 with FA=1 on
the same machine/model. When you match it, stop. When you beat it, get
suspicious and check correctness (see #11).

### B.5 Correctness gates

Cheapest first, and run all of them before believing any win:

```bash
python3 benchmarks/prefill_correctness.py    # prefill vs decode agreement
python3 benchmarks/compare_outputs.py        # vs reference outputs
MTP_VERIFY_CROSSCHECK=1                      # parallel verify == sequential
```

Plus two by hand:

- **`Hello.`** — catches gross breakage in seconds.
- **Needle test** — plant `ZEBRA42` mid-context, ask for it back. This is the
  only cheap test that exercises long-range attention, shared-KV anchors, and
  KV layout together. Short prompts pass happily with a broken cache.

---

## Part C — The knob catalog

Every one of these is a runtime switch, which means every one of them is an
experiment you can run in a minute. Grouped by what they let you isolate.

**Attention routing**

| Var | Values | Purpose |
|---|---|---|
| `ATTENTION_KERNEL` | `auto` / `specialized` / `ggml` / `mwg` | The big one (AGENTS #13, #14) |
| `FLASH_ATTN`, `PREFILL_FLASH_ATTN` | 0/1 | Flash vs non-flash |
| `ATTENTION_GQA_Q4`, `ATTENTION_GQA_F16` | 0/1 | GQA tiled kernels (opt-in after #11) |
| `TILED_EXT_MIN_Q` | int (default 2) | Tiled-ext threshold (MTP M4) |
| `MV_EXT_NSG` | int | Simdgroups for ext matvec |

**Fusion ladder** — each of these turns one rung off, so you can price it:

```text
FUSED_DECODE, MEGA_KERNEL, FUSED_QKV, FUSED_Q_ATTN, FUSED_K_ATTN,
FUSED_KV_ATTN, FUSED_RMSNORM_ACC, FUSED_RMSNORM_MLP,
FUSED_RMSNORM_MLP_KQUANT, FUSED_MLP_GELU_DOWN, FUSED_MLP_PLE
```

**MLP / matvec variants**

```text
MATVEC_KERNEL, MLP_GATE_UP_GGML, MLP_GATE_UP_DUAL, MLP_GELU_F16,
MLP_FUSED_GELU_GGML, MLP_FUSED_GELU_GGML_R2S4, PACKED_MLP_GATE_UP
```

**Prefill**

```text
PREFILL_MUL_MM, MUL_MM_MIN_SEQ, PREFILL_QKV_HSD, PREFILL_QKV_STACKED,
PREFILL_GATE_UP_STACKED, PREFILL_GATE_UP_EXT_GELU, PREFILL_MLP_F16,
PREFILL_MLP_GATE_F16_DST, PREFILL_GPU_ROPE, LLAMA_MAX_PREFILL_SEQ,
BENCH_PREFILL_EXACT, PREFILL_TIMING
```

**Quantization / KV**

```text
WEIGHT_FORMAT, LLAMA_KV_CACHE_TYPE, Q6K_TO_Q4, Q3_LAYER_START, Q3_LAYER_END
```

**MTP**

```text
LLAMA_MTP_DRAFT_STEPS, LLAMA_MTP_ADAPTIVE, LLAMA_MTP_P_MIN,
LLAMA_MTP_DRAFT_TOP_K, MTP_VERIFY_SEQUENTIAL, MTP_VERIFY_DECODE_BATCH,
MTP_VERIFY_DECODE_FA, MTP_VERIFY_CROSSCHECK, MTP_PREFILL_PARALLEL,
MTP_TREE_SPEC, MTP_TREE_BRANCH, MTP_LAYER_BISECT, MTP_STOP_LAYER
```

**Runtime / serving**

```text
LLAMA_CTX_SIZE, METAL_N_CB, LLAMA_QUEUE_DEPTH, LLAMA_KV_POOL_SLOTS,
LLAMA_REQUEST_TIMEOUT_SECS, LLAMA_PREFILL_TOKENS_PER_TICK
```

`MTP_LAYER_BISECT` / `MTP_STOP_LAYER` deserve a note: they let you run only
the first N layers, which is **binary search for a numerical divergence**.
When two paths disagree, bisect on layer count until you find the first layer
where they differ. Far faster than reading kernels.

---

## Part D — Case studies, read as method

### D.1 A dead end that taught the model (#1, #2, #4, #6, #7)

Dispatch overhead: 455 dispatches/token × ~5.2 µs = 2.38 ms. Real! But the
gap at 200 tok versus the 25-tok baseline is 2.6 ms, and dispatch overhead is
**constant** while the gap **grows with context**. Therefore not the cause.

That reasoning pattern — *does the suspect's magnitude scale the way the
symptom scales?* — closed four more experiments quickly. `fastMathEnabled` (no
effect: bandwidth-bound), `FUSED_DECODE=0` (same tok/s: bottleneck in both
paths), non-flash attention (same), 32-thread ggml vec kernel (same as
256-thread → thread count is not the limiter).

Five experiments, one conclusion: **the attention kernel's scaling with
context is the problem, not its constant costs.** That is worth five
experiments.

### D.2 A win that was really a dtype bug (E22)

PLE was 1555 ms of a 4k prefill. Not a kernel problem: F32 tensors were being
requantized to Q4_0, landing on a slow batch path. Keeping them dense f16 with
`mul_mm_f16`: ~230 ms.

Method: ablation found the bucket, then *reading the loader* found the cause.
No kernel was written. **When a bucket is surprisingly large, suspect the data
type before the kernel.**

### D.3 A win from a threshold someone else calibrated (M4, and Ch 10 D.3)

`TILED_EXT_MIN_Q` was 20 because llama.cpp switches vec→tiled at 20. But
llama.cpp's sub-20 path is a good vec kernel and ours is a bad fallback, so
the threshold was wrong *for us*. Set it to 2: verify 44 → 36 ms, e2e 37.7 →
42.4 tok/s.

**Every ported constant is a hypothesis about the alternative path.**
Re-measure it in your own codebase.

### D.4 A "win" that was a correctness regression (#11)

GQA tiled attention on the fused path: 59.6 tok/s, better than llama.cpp —
and garbage output. Root cause was semantic: the decomposed GQA path appended
KV *before* attention instead of attending with f32 K/V then appending.

Reverted to opt-in, shared-KV layers only. **Order a correctness check before
you celebrate**, and note the specific invariant you violated (here: fused vs
explicit append ordering, Ch 06 Part D.1).

### D.5 A correct change that did nothing (M7)

Fused gate‖up+GeLU for MTP verify: correct, clean, default-on, worth 0.5–1.5
tok/s — inside run-to-run noise. And the log says why, in advance-compatible
terms: *weight bandwidth for gate+up is unchanged; only activation scratch and
one dispatch are saved.*

Being able to explain why a change did not help is as valuable as a win,
because it prunes a whole family of similar ideas ("fuse more glue") in one
entry.

---

## Part E — A protocol you can follow

```text
1. Establish baseline
   - cold machine, note the config, run twice, record both numbers
   - state the token count and context length

2. Locate the cost
   - PROFILE_ABLATE for buckets (get SKIP_all first)
   - PROFILE_DISPATCHES to check the dispatch count matches your model of it
   - PROFILE_PHASES for relative attribution only

3. Form a hypothesis
   - "X costs N ms because <mechanism>"
   - predict the improvement magnitude BEFORE running

4. Change one variable
   - prefer an existing env var; add one if the change is structural

5. Measure
   - twice, cold; compare against the recorded baseline, not from memory

6. Verify correctness
   - Hello. + needle test + the relevant benchmarks/ script

7. Record in AGENTS.md
   - what, result, conclusion — including "no effect" and "worse"
```

Step 7 is the one people skip, and it is the one that compounds.

---

## Part F — Where this project currently stands

Decode: ~50 tok/s at 200+ tokens with `ATTENTION_KERNEL=auto` + Q4_0 KV, vs
llama.cpp 53.9. Short-context peak ~53.5. Prefill: 581–591 tok/s at exact
4096, vs llama.cpp 593.9 — effectively matched.

So prefill is done and decode has ~4 tok/s left. The open hypotheses, verbatim
from `AGENTS.md`: KV-cache Q4_0 write bandwidth during decode, pipeline bubbles
between attention and MLP, and K-norm/RoPE path differences vs llama.cpp.

Untried experiments are listed there too (E17 threshold sweep, E18 ggml vs
specialized at 400–512 tok, E19 KV layout microbench for head_dim=256, E20 MLP
variant sweep, E21 command-buffer pipelining). If you want to contribute a
number rather than a guess, start with one of those — they are specified
precisely enough to run.

---

---

## Part G — Benchmarking discipline on a laptop

Every number in `AGENTS.md` was measured on an M1 Pro under a fan curve, which
means noise is not a rounding detail — it is the main threat to every conclusion
in this chapter.

### G.1 The noise sources, in order of size

| Source | Typical effect | Control |
|---|---|---|
| Thermal throttling | **5–15%** over a long run | cool between runs; report cold vs warm separately |
| Other processes (browser, IDE, indexers) | 2–10% | close them; measure twice |
| First-run compile / page-in | first token much slower | warm up, discard the first run |
| Context length drift between runs | 5%+ | fix generated token count exactly |
| Sampling randomness changing text length | varies | greedy, or fixed seed and fixed max_tokens |

The log's own convention reflects this: it reports things like "42.4 / 43.3
tok/s (2 runs each)" and calls ±2 tok/s noise. Treat any single-run difference
under ~4% as unmeasured.

### G.2 A protocol that survives review

```text
1. Build release. Confirm the binary you are about to run is the one you built.
2. Close everything else. Let the machine idle 60 s.
3. Warm up: one full run, discarded.
4. Run config A three times, alternating with config B three times (A,B,A,B,A,B).
   Alternating defeats thermal drift; three consecutive A runs do not.
5. Report min, median, and max for each. Compare medians.
6. If the medians differ by less than the within-config spread, you measured
   nothing. Say so.
7. Record the exact command line, env vars, model file, and generated token
   count alongside the numbers.
```

Step 4 is the one people skip and it is the one that matters: A-then-B is
confounded with cold-then-hot, and that confound is worth more than most of the
optimizations in this chapter.

### G.3 Fix the workload, not just the config

Decode throughput depends on context length (Ch 07 Part I), so "50 tok/s" is
meaningless without it. Use `--bench-decode --bench-decode-tokens 25,200` and
report both, exactly as the log does. When comparing against llama.cpp, match:
model file, quantization, KV cache type, flash-attention setting, context
length, and generated token count. A mismatch in any one of them can produce the
entire gap you are trying to explain.

---

## Part H — Writing up an experiment

`AGENTS.md` is the most valuable file in the repo because it records failures
with reasons. Keep it that way. The template that makes an entry useful later:

```markdown
### N. Short name (date)

**What**: the change, precisely — file, flag, constant, and the value before
and after.

**Hypothesis**: what you expected and *why* (with the arithmetic: bytes,
dispatches, occupancy).

**Result**: numbers with their conditions (model, KV type, context, generated
tokens, runs). Include the baseline measured in the same session.

**Correctness**: what you verified and how (reference compare, needle test,
crosscheck flag).

**Conclusion**: kept / reverted / opt-in, and the *reason* — ideally a sentence
that would have let you predict this without running it.
```

The last line is the whole point. `AGENTS.md` #1 concludes "dispatch overhead is
constant; the gap grows with context" — a scaling argument that immediately
retired four other candidate explanations. That sentence was worth more than the
measurement that produced it.

Two rules for the log:

- **Record negative results.** Ten of the fifteen numbered entries are negative
  and they are why nobody re-runs those experiments.
- **Record the bugs you fixed on the way**, even boring ones (#9's buffer offset,
  #10's `kernel void`). They are the honest cost of porting a kernel, and the
  next person will hit them.

## Part I — Exercises

1. Run `--bench-decode` at 25, 200, and 400 tokens on a cold machine. Do your
   numbers match the table in Part B.1? If not, what differs (thermals, model,
   KV type, attention kernel)?

2. Run `PROFILE_ABLATE=SKIP_all`, then `SKIP_mlp`, `SKIP_attn`, `SKIP_ple` at
   4k prefill. Build your own version of Ch 10 Part H's table.

3. `PROFILE_DISPATCHES=1` on one decode token. Compare against the ~13–14
   per-layer estimate in Ch 08 Part F. Which branch does your MLP take?

4. Pick E17 (hybrid threshold sweep: 64 / 128 / 256). Write the hypothesis and
   predicted result *first*, then run it at 200 and 400 tokens, then check
   output quality at each threshold. Write the AGENTS.md entry.

5. Set `ATTENTION_KERNEL=ggml` and measure at 25 and 200 tokens. Explain the
   flatness using the mechanism from #13.

6. Turn off one fusion rung at a time (`FUSED_QKV=0`, then
   `FUSED_RMSNORM_ACC=0`, …). Which rung is worth the most? Does the ordering
   match your bandwidth model?

7. Deliberately reproduce a "fast but wrong" result: enable `ATTENTION_GQA_Q4=1`
   on KV-owning layers if your build allows it. Note the tok/s and the output.
   Then explain the invariant you violated in one sentence.

---

## Checklist

- [ ] I can name the six rules and why each exists.
- [ ] I know why `PROFILE_PHASES` numbers are relative, not absolute.
- [ ] I always get `SKIP_all` before interpreting any ablation Δ.
- [ ] I know the reasoning that closed experiments #1, #2, #4, #6, #7 at once.
- [ ] I run a needle test before believing a perf win.
- [ ] I would write down a change that did nothing, and say why.
- [ ] I know the current gap, the config that achieves it, and one untried
      experiment I could run today.

**Next:** [16_glossary_and_drills.md](16_glossary_and_drills.md) — self-test
before you claim you know this codebase.

- Take the last measurement you made on this project and rewrite it in the
  Part H template. If you cannot fill in "Hypothesis" with arithmetic, you
  measured before you predicted.
- Design an A/B for `METAL_N_CB=1` vs `2` that satisfies Part G.2, then run it.
- Two configs measure 43.3 and 42.1 tok/s, single runs each. Write the honest
  one-sentence conclusion.
