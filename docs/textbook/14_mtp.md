# 14 — MTP: Speculative Decoding with a Draft Head

Decode is bandwidth-bound: every token reads ~2.5 GB of weights, and the ALUs
idle. That waste is an opportunity — if you could *verify several candidate
tokens in one forward*, you would pay roughly one token's weight bandwidth for
several tokens of output.

That is speculative decoding. Gemma4's variant is **MTP** (multi-token
prediction): the checkpoint ships a small draft head trained to predict the
next token from the target model's own final hidden state.

Files:

```text
src/speculative.rs     # MtpDraftHead: the small model + its forward
src/gemma4_mtp.rs      # Gemma4MtpAssistant: draft chain orchestration
src/mtp_serve.rs       # MtpScheduler: serial serving loop
src/draft_tree.rs       # (experimental) tree drafting
```

---

## Part A — The idea, with the arithmetic

Standard decode: 1 forward → 1 token.

Speculative decode:

```text
1. draft:  cheap model proposes d candidate tokens        (d small forwards)
2. verify: target model scores all d+1 positions at once  (1 batched forward)
3. accept: keep the longest prefix where target == draft
4. rewind: drop KV rows for rejected drafts
```

The output is **identical to greedy decoding of the target model** whenever
you accept only on exact match — verification is what guarantees that. You are
not trading quality for speed; you are trading *wasted bandwidth* for speed.

Why it can win: a batch-3 verify forward reads the weights once, the same as a
batch-1 decode. If all 3 are accepted you got 3 tokens for ~1.6× the cost of
1 (measured; see Part F).

Why it can lose: rejected drafts are pure waste — draft forwards plus verify
rows for tokens you discard. The break-even depends on accept rate and on how
cheap the draft is.

### A.1 The expected-tokens model

With per-token accept probability `p` and `d` drafts, expected accepted
tokens ≈ `(1 − p^(d+1)) / (1 − p)`. At the measured `p ≈ 0.42`:

```text
d=1: 1.42   d=2: 1.60   d=3: 1.67   d=4: 1.70   d=6: 1.72
```

It saturates fast. That is exactly what the draft-steps sweep in `AGENTS.md`
found empirically — 2/3/4/6/7 steps all landed at a flat 42–43.8 tok/s.
**Once you see this curve, "add more draft steps" stops being a plausible
lever**, and the only real levers are draft *quality* (raise `p`) and verify
*cost*.

---

## Part B — The draft head

`MtpDraftHead` is a single transformer-ish block shipped inside the GGUF under
the `gemma4-assistant.*` namespace (Ch 03 Part B):

```text
gemma4-assistant.nextn_predict_layers      (default 4)
gemma4-assistant.embedding_length_out      (default 1536)   hidden_backbone
gemma4-assistant.embedding_length          (default 256)    draft hidden
gemma4-assistant.vocab_size                (default 262144)
```

It takes **two** inputs:

1. The embedding of the current token (scaled by `√hidden_backbone`).
2. `h_nextn` — the target model's final hidden state for that position.

Input 2 is what makes it accurate despite being tiny: it does not have to
understand the context from scratch, it gets the target's own summary of it.

Crucially, the draft head **attends to the target model's KV cache**:

```200:211:src/gemma4_mtp.rs
let (next_token, h_next) = self.head.forward_draft_step(
    &target.ctx,
    &self.scratch,
    &token_embedding,
    if step == 0 { initial_activation } else { &self.embd_nextn },
    target.total_tokens as u32,
    &target.k_cache,
    &target.v_cache,
    target.kv_seq_len,
    target.kv_capacity,
    target.kv_cache_type,
);
```

No separate draft cache to maintain, no divergence between two caches — and
another reason the draft is cheap.

### B.1 The `h_nextn` capture site (and the trap)

Where exactly do you read the target's hidden state: before or after the final
output norm?

`AGENTS.md` M5 records the full story. The generic `llama-graph.h` comment
says `t_h_nextn` is the "hidden state before final output norm." Following
that comment and switching capture from `normed_buf` to `hidden_buf`:

```text
accept rate: 42.5% → 21.8%
tok/s:       41    → 30
```

Reverted. `gemma4.cpp` sets `t_h_nextn` **after** `output_norm` — matching
transformers, vLLM, and SGLang for this architecture. The existing post-norm
capture was already right.

Two lessons, both general:

- **A comment in a generic header does not override the architecture-specific
  implementation.** Check where the value is actually assigned for *your*
  model.
- **Accept rate is a sharp diagnostic.** A halving of accept rate with
  unchanged output text says "the draft is being fed something subtly wrong,"
  not "the draft is bad." Watch it like you watch loss curves.

The same experiment did keep one real fix: `forward_prefill_parallel_self` was
reading row 0 instead of the **last** row for multi-token prefill. Off-by-one
in which row's hidden state you hand to the draft head.

---

## Part C — Drafting a chain

```167:229:src/gemma4_mtp.rs
pub fn draft_chain(&mut self, initial_token: usize, initial_activation: &[f32],
                   steps: usize, target: &Gemma4GpuModel, p_min: f32)
    -> Result<Vec<usize>, String>
{
    if steps == 0 { return Ok(Vec::new()); }
    if initial_activation.len() != self.head.hidden_backbone { return Err(...); }
    if target.kv_seq_len == 0 { return Err("target KV cache is empty".to_string()); }

    self.initial_activation = initial_activation.to_vec();
    let mut draft_token = initial_token;
    let mut drafts = Vec::with_capacity(steps);

    for step in 0..steps {
        let mut token_embedding = target.token_embedding_raw(draft_token)?;
        let scale = (self.head.hidden_backbone as f32).sqrt();
        for v in token_embedding.iter_mut() { *v *= scale; }

        let (next_token, h_next) = self.head.forward_draft_step(
            &target.ctx, &self.scratch, &token_embedding,
            if step == 0 { initial_activation } else { &self.embd_nextn },
            target.total_tokens as u32,
            &target.k_cache, &target.v_cache,
            target.kv_seq_len, target.kv_capacity, target.kv_cache_type,
        );
        self.gpu_passes += 1;
        draft_token = next_token as usize;
        self.embd_nextn = h_next;

        if p_min > 0.0 {
            let logits = MetalContext::read_buffer(&self.scratch.logits, self.head.vocab);
            let prob = draft_token_confidence(&logits, draft_token, draft_top_k());
            drafts.push(draft_token);
            if prob < p_min { break; }
        } else {
            drafts.push(draft_token);
        }
    }
    Ok(drafts)
}
```

Read the chain structure: it is **autoregressive within the draft**. Step 0
uses the target's activation; step *k* > 0 uses `embd_nextn`, the draft head's
own output from step *k−1*. So the draft is a little model rolling forward on
its own predictions, which is precisely why accuracy decays with depth (Part
A.1).

Note `if p_min > 0.0` gates a `read_buffer` — a **GPU→CPU sync** per draft
step. That is why the code avoids it by default: the sync costs more than the
occasional wasted draft.

### C.1 Draft confidence, and matching llama.cpp exactly

```248:259:src/gemma4_mtp.rs
fn draft_token_confidence(logits: &[f32], token: usize, top_k: usize) -> f32 {
    if top_k == 0 || top_k >= logits.len() {
        // full-vocab softmax
        ...
    }
    // else: softmax normalized over the top_k largest logits only
}
```

With the comment above it:

```232:243:src/gemma4_mtp.rs
/// Candidate-set size for the draft confidence softmax. llama.cpp's draft-mtp
/// sampler is top_k=10: the greedy token's probability is normalized over the
/// top 10 candidates only, not the full vocab. 0 = full-vocab softmax.
fn draft_top_k() -> usize { /* LLAMA_MTP_DRAFT_TOP_K, default 10 */ }
```

This is `AGENTS.md` M6. The original implementation used a **full-vocab**
softmax, so probabilities were systematically lower than llama.cpp's, and any
`p_min` threshold cut drafts far earlier than intended. With a 262k vocab the
difference is large: mass spread over the tail suppresses `p` even when the
top candidate dominates its real competitors.

Outcome of fixing it: accept rate rises with `p_min` (0.3 → 44%, 0.5 → 46.5%,
0.75 → 50.5%) but tok/s stays inside run-to-run noise — the extra draft
passes eat the accept gain. So `p_min` stays opt-in.

Worth internalizing: **when you port a threshold, port the distribution it
was calibrated on.** Same lesson as `TILED_EXT_MIN_Q` (Ch 10 Part D.3), in a
different disguise.

### C.2 Adaptive depth

```281:286:src/mtp_serve.rs
let tail_steps = effective_draft_tail_steps(
    &stats.accepted_per_cycle,
    self.draft_steps,
    self.adaptive,
);
let n_draft = tail_steps + 1;
```

Recent accept history feeds the next cycle's draft depth: accepting a lot →
draft deeper; getting rejected → draft shallower. Given the flat sweep in Part
A.1 this is more about avoiding waste in bad stretches than about winning in
good ones.

---

## Part D — Verify, accept, rewind

The serving loop (`mtp_serve.rs` ~311–367) is the heart of it:

```320:367:src/mtp_serve.rs
let mut verify_batch = Vec::with_capacity(drafted.len() + 1);
verify_batch.push(id_last);
verify_batch.extend_from_slice(&drafted);

let verify_tokens = match self.model.forward_verify_batch(&verify_batch) { ... };

let mut ids: Vec<usize> = Vec::with_capacity(drafted.len() + 1);
let mut n_accepted = 0usize;
for i in 0..drafted.len() {
    let pred = verify_tokens[i];
    ids.push(pred);
    if pred != drafted[i] { break; }
    n_accepted += 1;
}
if n_accepted == drafted.len() {
    ids.push(verify_tokens[drafted.len()]);
}
stats.record_cycle(drafted.len(), n_accepted);

// Roll back KV for rejected drafts.
let rewind = (drafted.len() - n_accepted) as u32;
if rewind > 0 { self.model.truncate_kv(rewind); }

let i_h = n_accepted.min(verify_batch.len() - 1);
mtp_hidden = self.model.prefill_hidden_activation_at(i_h);
id_last = *ids.last().unwrap();
ids
```

Four subtleties, each worth pausing on.

**1. `verify_batch` starts with `id_last`, not with a draft.** Position *i* of
the verify output is the target's prediction *given* tokens up to *i*. To
score draft[0] you must include the token before it. Hence `d+1` rows for `d`
drafts.

**2. The accept loop pushes `pred`, not `drafted[i]`.** On a mismatch, the
target's own token is still pushed before breaking. That token is correct by
construction — it is what plain greedy decode would have produced — so a
rejection is never a wasted cycle. You always advance at least one token.
This is the invariant that makes speculative decoding *safe*: worst case, one
token per cycle, exactly like non-speculative decode.

**3. The bonus token.** If every draft was accepted, `verify_tokens[d]` is a
free extra token — the target already computed it in the same forward. `d`
accepted drafts yield `d+1` emitted tokens.

**4. Rewind.** Verify wrote KV rows for all `d+1` positions. Rejected rows
must go: `truncate_kv(rewind)` drops the tail. Get this wrong and the cache
contains keys/values for tokens that were never emitted — the model then
attends to a phantom continuation. The symptom is textbook: coherent for a
while, then drifting into repetition, exactly like `AGENTS.md` #15's essay
failure.

Also note `i_h = n_accepted.min(verify_batch.len() - 1)`: the hidden state
handed to the next draft cycle must come from the row of the **last accepted**
position, clamped to the batch. Same class of off-by-one as M5's last-row fix.

### D.1 Verify paths

Three implementations exist, and the default was hard-won:

| Path | Env | What it does |
|---|---|---|
| `forward_verify_parallel` | default | One prefill-style batched chunk over all `d+1` rows |
| sequential | `MTP_VERIFY_SEQUENTIAL=1` | One decode forward per row |
| decode-batch | `MTP_VERIFY_DECODE_BATCH=1` | Batched decode kernels |

Sequential verify was the original default and it made MTP **slower than
non-MTP** (~25 tok/s vs ~44): 90% of wall time in verify, because *d+1*
sequential decodes cost *d+1* full weight reads — destroying the entire
premise.

Parallel verify runs one prefill-style chunk: weights read once, causal
attention across the rows, batched matvecs. When you enable it, keep
`MTP_VERIFY_CROSSCHECK=1` on during development — it asserts parallel ==
sequential for every row, which is how the two original garbage bugs were
found (draft-head attention scratch sized to `hidden_head` instead of
`max_head_dim=512`, and an f16 MLP cast feeding the f32 matvec fallback).

### D.2 Why verify needs its own kernels

Verify sits in an awkward size regime: `seq` 2–8. Too big for decode matvec
(one activation row), too small for prefill `mul_mm` (which wants dozens of
rows to fill its tiles). Measured: forcing `mul_mm` at these sizes via
`MUL_MM_MIN_SEQ=1` gave **20 tok/s**.

So three things were built specifically for this regime:

- **M2 — `mul_mv_ext` K-quant kernels** (`matvec_ggml_ext_q{4,6}K_nx8_r{2..5}`,
  ported from llama.cpp). Dequantize a weight row once, dot it against all
  batch rows. Weight bandwidth amortized over the batch without needing
  matmul-sized tiles. Ch 05 Part G.
- **M3 — batched `lm_head`.** Verify originally computed logits per row with
  `encode_matvec_auto_at_view`, reading the ~440 MB vocab matrix **once per
  row**. Replaced with a single `encode_prefill_projection_auto_batch_view`
  over all rows. +1 tok/s, and an obvious-in-hindsight bug: the fix is
  "read the biggest matrix in the model once."
- **M4 — `TILED_EXT_MIN_Q=2`.** The tiled attention gate, covered in Ch 10
  Part D.3: 44 → 36 ms verify GPU, e2e 37.7 → 42.4 tok/s. The single biggest
  MTP win.
- **M7 — fused gate‖up+GeLU ext matvec.** Correct, default-on, and worth
  ~0.5–1.5 tok/s, i.e. **within noise**. Expected: weight bandwidth for
  gate+up is unchanged; only activation scratch and one gelu dispatch are
  saved. A good example of a clean change that does not move the number, and
  of writing that down rather than claiming a win.

---

## Part E — Serial serving

```text
MtpScheduler: one request at a time, model-owned KV cache
Scheduler:    many requests, KV pool slots
```

MTP does not do continuous batching. Reason: the verify forward is itself a
batch, and the accept/rewind cycle mutates cache length in ways that are
awkward to interleave with other requests' rows. Ch 13 Part B.2's
`from_existing` adapter is what lets verify reuse the pool-shaped prefill
kernels against the model-owned single cache.

The loop shape mirrors the main scheduler's checks in the same order — max
tokens, cancellation, timeout — then draft, verify, accept, and emit each
accepted token through `emit_token` (which applies stop conditions per token,
since a cycle can produce several).

There is also a fallback worth noticing:

```311:318:src/mtp_serve.rs
let accepted_ids: Vec<usize> = if drafted.is_empty() {
    // Rare fallback: plain single-token decode on global KV.
    let next_logits = self.model.forward_single_token(id_last);
    mtp_hidden = self.model.last_hidden_activation();
    let next = sampling::argmax(&next_logits);
    stats.record_cycle(0, 0);
    id_last = next;
    vec![next]
}
```

If drafting produced nothing (e.g. `p_min` cut at step 0), fall back to a
normal decode step. The speculative path never becomes a hard dependency.

---

## Part F — Results and where the ceiling is

E2B Q4_K_M + F16 draft head, 399-token essay, adaptive draft, ~42% accept,
1.85 tokens per forward:

| Config | tok/s |
|--------|-------|
| Non-MTP baseline (`auto`) | 43.5–44.5 |
| MTP sequential verify (old default) | 23.5–26 |
| MTP parallel verify + ext matvec | 34.8 |
| + batched lm_head | 37.7 |
| + tiled ext attention (default) | **42.4** (auto) / **43.1** (specialized) |

Read that table honestly: **MTP has clawed its way back to roughly the
non-MTP baseline.** It is not yet a win on this hardware/model pair.

Where the remaining cost is: verify at seq=3 takes ~36 ms versus ~22 ms for a
single decode — **1.6× for 3 rows**. Ablation puts ~12 ms of that in the MLP,
where a pure weight-bandwidth model predicts ~1.1×, not 1.6×. The three
weight streams (gate/up/down) plus occupancy effects are the suspects; M7
proved it is *not* the gelu glue.

And the structural ceiling: at 42% accept and 1.85 tokens/forward, even a
**free** verify caps you at 1.85× per-forward cost. So the two real levers are:

1. **Draft quality** — 42% → 60%+ would change the arithmetic materially.
2. **Verify phase timing** — find where the 1.6× MLP tax actually lives.

Everything else has been tried and logged.

---

---

## Part F.1 — The speculative-decoding cost model, and why it caps out here

Every result in Part F is predicted by one equation. Let

- `α` = acceptance rate (fraction of drafted tokens that verify correctly)
- `k` = draft steps per cycle
- `c_d` = cost of one draft-head forward, relative to one target forward
- `c_v(n)` = cost of verifying `n` rows, relative to one target forward

Then the expected tokens per cycle is `T(k, α) = (1 − α^{k+1}) / (1 − α)`
(you always get at least the one token the target itself produces), and the cost
per cycle is `k·c_d + c_v(k+1)`. Speedup over plain decode is:

```text
S = T(k, α) / ( k·c_d + c_v(k+1) )
```

Plug in this engine's measured numbers — `α ≈ 0.42`, `c_d ≈ 0.1` (the draft head
is one small F16 layer), `c_v(3) ≈ 1.6` (Part D: verify of 3 rows costs ~36 ms
against ~22 ms for a single decode; assume it grows linearly,
`c_v(n) = 1 + 0.3(n − 1)`):

```text
k = 2:  T = 1 + 0.42 + 0.176 = 1.60    cost = 0.2 + 1.6 = 1.80   S = 0.89
k = 3:  T = 1.67                        cost = 0.3 + 1.9  = 2.20  S = 0.76
```

Both are **below 1.0**, and yet the measured end-to-end result is roughly *equal*
to the non-MTP baseline (42.4 vs 43.5–44.5 tok/s). That near-agreement is the
useful part: the model says MTP as configured is approximately a wash, and the
bench agrees. It also tells you precisely which term must change.

### Sensitivity: which term is worth attacking

Hold the others fixed and vary one:

| Change | S at k=2 | Comment |
|---|---|---|
| baseline (α=0.42, c_v=1.6) | 0.89 | measured |
| α → 0.60 | 1.09 | draft quality |
| α → 0.75 | 1.28 | draft quality |
| c_v(3) → 1.1 | 1.23 | perfect batching |
| α → 0.60 **and** c_v → 1.1 | 1.51 | |

Two conclusions, and they are the reason Part F's "remaining gap" section reads
the way it does:

1. **`c_v` is bounded below by 1.0 and is already at 1.6.** The theoretical best
   case — verifying 3 rows for the price of 1, since the weights are read once —
   would buy ~50%. `AGENTS.md` M7 chased part of this (fused gate∥up+GeLU) and
   got a wash, because the cost is three weight streams and occupancy, not glue.
   The ceiling on this term is visible and modest.
2. **`α` has no ceiling below 1.0 and is the only term with room.** Going from
   42% to 60% is worth more than a perfect verify kernel. That is a *model*
   problem — draft-head capacity and training — not a kernel problem.

This is why the sweeps in Part F look flat. `p_min` raises `α` to 46–50% but
raises the number of draft passes at the same time, so `k·c_d` grows exactly as
fast as `T` does. Any lever that trades draft cost for acceptance moves along a
curve the equation already knows about; only lifting `α` at fixed cost moves the
curve itself.

### The lesson that generalizes

Speculative decoding is only a win when the draft is *cheap* and *right*. Both
words matter, and the equation tells you which one you are short on. Before
optimizing a verify kernel, measure `α` — if it is under ~50% with a
near-free draft, no amount of kernel work will produce a speedup, and you will
spend a week proving it.

## Part G — Exercises

1. With `p = 0.6`, compute expected accepted tokens for `d = 1..6`. At what
   `d` does the marginal gain drop below 0.05 tokens?

2. Draft produces `[A, B, C]`; verify returns `[A, X, C, D]`. What is emitted?
   What is `n_accepted`? What is `rewind`? Which row does `i_h` select?

3. Why must `verify_batch` include `id_last`? What would happen if it started
   at `drafted[0]`?

4. Explain why pushing `pred` (not `drafted[i]`) on mismatch keeps output
   identical to greedy decoding.

5. Skip `truncate_kv`. Describe what the cache contains and predict the
   text-level symptom. Which `AGENTS.md` entry describes the same class of
   failure?

6. Why is full-vocab softmax "wrong" for `p_min` against llama.cpp parity?
   Construct a two-number example with a 262k vocab where full-vocab and
   top-10 probabilities differ by more than 2×.

7. Verify at seq=3 costs 1.6× a single decode. Write the bandwidth model that
   predicts 1.1×, then list three mechanisms that could account for the gap.

8. Run with `MTP_VERIFY_CROSSCHECK=1`. What exactly does it compare, and why
   is it too slow to leave on?

9. Derive `T(k, α) = (1 − α^{k+1})/(1 − α)` from scratch. Why is the exponent
   `k + 1` rather than `k`?

10. Using Part F.1's model, find the `α` at which `k = 4` beats `k = 2`, holding
    `c_d = 0.1` and `c_v(n) = 1 + 0.3(n − 1)`. Interpret the answer.

11. Suppose verify became free (`c_v = 1`). What `α` would you need for a 1.5×
    speedup at `k = 3`? Is that plausible for a one-layer draft head?

## Checklist

- [ ] I can explain why speculative decoding preserves greedy output exactly.
- [ ] I can derive the expected-tokens formula and explain the flat sweep.
- [ ] I know what `h_nextn` is, where it is captured, and why M5 was wrong.
- [ ] I can walk verify → accept → rewind and justify every index.
- [ ] I know why sequential verify destroyed the premise.
- [ ] I can name the three kernel families built for the seq 2–8 regime.
- [ ] I know the current accept rate, tok/s, and the two remaining levers.

**Next:** [15_optimization_lab.md](15_optimization_lab.md) — the measurement
discipline that produced every number in this chapter.
