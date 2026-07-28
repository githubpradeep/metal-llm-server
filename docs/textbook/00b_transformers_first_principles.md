# 00b — Transformers From First Principles

This chapter derives everything the rest of the textbook assumes. Not "here is
the attention formula" — we build attention from the problem it solves, prove
the properties that later kernels exploit, and do the arithmetic by hand.

Work through it with a pen. Every part ends with exercises; the numbers in
them are the numbers you will need in Ch 05–07.

---

# Part 1 — What an inference server actually computes

## 1.1 The function

A language model is one function:

```text
f(token_ids[0..T]) → logits ∈ R^vocab      # scores for the token at position T
```

`logits[v]` is an unnormalized score for vocabulary item `v`. That is all a
model is. Everything else — chat, tools, streaming — is bookkeeping around
repeated calls to `f`.

## 1.2 Autoregression

```text
ids = tokenize(prompt)
loop:
    logits = f(ids)
    next   = sample(logits)
    emit(next)
    ids.push(next)
    until next == EOS or budget exhausted
```

Naively this is catastrophic: generating 500 tokens means 500 calls to `f`,
each over a growing sequence. If `f` costs `O(T²)` for a sequence of length
`T`, total cost is `O(T³)`. Nobody could serve that.

Two observations rescue it, and they are the entire systems story of this
repo:

**Observation A — causality means the past never changes.** Token 7's
representation cannot depend on token 8. So when you append a token, every
computation for earlier positions is *bit-identical* to last step. Cache it.
That is the KV cache (Part 9), and it turns per-token cost from `O(T²)` into
`O(T)`.

**Observation B — the prompt can be processed all at once.** The whole prompt
is known upfront, so its positions can be computed in parallel, as matrices.

Those two observations create two completely different computational regimes:

| | Prefill | Decode |
|---|---|---|
| Query rows per forward | many (a chunk: 128–4096) | exactly 1 |
| Dominant operation | matrix × matrix | matrix × vector |
| Weight bytes per useful token | amortized over the chunk | **all of them, every token** |
| Bottleneck | compute (FLOPs) | memory bandwidth |
| Kernels here | `mul_mm`, `flash_attn_ext` | `matvec_*`, flash decode |
| Metric people quote | tokens/s prefill (~590) | tokens/s decode (~50) |

**The one sentence to keep:** decode re-reads nearly every weight in the model
to produce one token; prefill reads each weight once and reuses it across
hundreds of positions. Ch 00c Part A turns that into numbers, and Ch 05 turns
those numbers into kernel choices.

## 1.3 Why the ratio is so lopsided

E4B has roughly 2.5 GB of weights in Q4_0. At M1 Pro's ~200 GB/s that is
~12.5 ms of pure reading, i.e. a hard ceiling near 80 tok/s no matter how
clever your kernels are. Meanwhile the *arithmetic* for one token is about
5 GFLOP, which at ~3.2 TFLOPS takes ~1.6 ms.

So decode spends ~8× more time waiting for memory than computing. Every
optimization in `AGENTS.md` that failed, failed because it optimized compute
(`fastMathEnabled`, thread counts) in a regime where compute is free.

### Exercises 1

1. A model has 4 GB of weights. Bandwidth is 200 GB/s. What is the theoretical
   maximum decode rate? What if you quantize to half the bytes?
2. Why does a 4096-token prompt at 590 tok/s (≈7 s) not contradict a decode
   rate of 50 tok/s? Express both in bytes moved per token.
3. If causality did *not* hold (a token could attend to future tokens), which
   of the two observations in 1.2 would survive? What would that do to serving?

---

# Part 2 — Tokens, embeddings, and the residual stream

## 2.1 Tokens

Text → tokenizer → integers in `[0, vocab)`. Gemma4's vocab is 262144, which
is large, and that size shows up everywhere: `lm_head` is `[262144, hidden]`
(~380 MB in Q4_0), the softmax in Part 10 is over 262k values, and the draft
confidence calculation in Ch 14 had a bug *because* the vocab is this big.

Special tokens matter. Chat turns are wrapped by a template (roles, channels,
tool markers) before tokenization — Ch 11 Part B. The ids are not invented by
the server; they come from GGUF metadata (Ch 03 Part B).

## 2.2 Embedding, and the √hidden scale

Token id → row of an embedding matrix `E` of shape `[vocab, hidden]`:

```text
h₀ = E[token] * √hidden_size
```

Where does that scale come from? Embedding rows are initialized with variance
around `1/hidden`, so a raw row has L2 norm around 1. But the residual stream
downstream expects vectors whose *per-component* magnitude is O(1), not whose
*norm* is O(1). Multiplying by `√hidden` converts one to the other:

```text
‖E[t]‖ ≈ 1  ⇒  components ≈ 1/√hidden
× √hidden   ⇒  components ≈ 1
```

Omit it and every component entering layer 0 is ~50× too small (for
hidden=2560). RMSNorm will rescale the *direction* correctly, so the model
still produces fluent text — just systematically wrong text. This is the first
example of the pattern you will meet a dozen times in this codebase:

> **A missing scale factor does not crash. It degrades.**

In this codebase: `decode_embed_into` in `gemma4_gpu_model.rs`. The MTP draft
head applies the same scale to its own input (`gemma4_mtp.rs` ~196: `let scale
= (self.head.hidden_backbone as f32).sqrt();`).

## 2.3 The residual stream as a communication bus

Every sub-block has this shape:

```text
h ← h + SubBlock(Norm(h))
```

Note what it is *not*: `h ← SubBlock(h)`. The block never replaces the state;
it **adds** to it. Three consequences worth internalizing:

**1. It is a bus, not a pipeline.** Think of `h` as a shared workspace of
`hidden` channels. Each block reads it, computes a contribution, and writes
that contribution back in. Layer 30 can read something layer 2 wrote, because
nothing erased it.

**2. Identity is the default.** If a sub-block outputs zeros, the layer is a
no-op and the network still works. That is why training deep stacks is
possible at all, and it is also why *bugs* behave the way they do:

| Bug | Effect on `h` | What you see |
|---|---|---|
| Sub-block output zeroed | layer becomes identity | slightly worse output, no crash |
| Residual add dropped (`h = out`) | history destroyed each layer | total garbage |
| Residual scaled by 1.1 | drift compounds over 42 layers | fluent, off-distribution text |

**3. Everything is in the same vector space.** All 42 layers read and write the
same `hidden`-dimensional space, which is why the final `lm_head` can be the
same matrix as the input embedding (weight tying, Ch 03 Part C).

Decode: `h` is `[hidden]`. Prefill: `[seq, hidden]`, with every row
independent through the MLP and norms, and coupled only inside attention.

### Exercises 2

1. hidden = 2560. A raw embedding row has norm ≈ 1. What is the typical
   component magnitude before and after the `√hidden` scale?
2. You accidentally write `h = post_norm(out)` instead of `h += post_norm(out)`
   in one of 42 layers. Predict the output quality. Now in all 42.
3. Why can `lm_head` share weights with `E`? What property of the residual
   stream makes that even type-check conceptually?

---

# Part 3 — RMSNorm

## 3.1 Deriving it from LayerNorm

LayerNorm standardizes each vector:

```text
μ  = mean(x)
σ² = mean((x − μ)²)
y  = (x − μ)/sqrt(σ² + ε) * γ + β
```

Four things: center, scale, learned gain, learned bias. Which of them earns
its cost?

Empirically, in transformer residual streams, `μ ≈ 0` already (the stream is a
sum of many roughly zero-mean contributions), and the bias `β` is redundant
with biases elsewhere. What actually matters is **preventing scale drift**: as
contributions accumulate, `‖h‖` grows, and without normalization the softmax
and activations saturate.

Drop centering and bias, and you get RMSNorm:

```text
rms(x) = sqrt( (1/N) Σ x_i²  + ε )
y_i    = (x_i / rms(x)) * w_i
```

One pass, one weight vector. Note that ε sits **inside** the sqrt, added to
the mean of squares. Implementations differ on this and it produces small
persistent numeric differences — match the reference (Ch 08 Part A.1).

Geometrically: RMSNorm projects `x` onto the sphere of radius `√N` and then
rescales each axis by `w`. It preserves direction and discards magnitude.

## 3.2 Where Gemma4 puts norms

Standard "pre-norm" transformer:

```text
h ← h + Attn(Norm(h))
h ← h + MLP(Norm(h))
```

Gemma4 adds a **second** norm on the way *out* of each sub-block:

```text
h ← h + post_attn_norm( Attn( input_norm(h) ) )
h ← h + post_ff_norm( MLP( pre_ff_norm(h) ) )
```

Why: the input norm controls what the sub-block *sees*; the output norm
controls what it *contributes*. With 42 layers all writing into one bus, the
second norm keeps any single layer from dominating. This is why the engine has
a fused `rmsnorm_acc` kernel (`acc[i] += x[i] * inv_rms * w[i]`) — the
post-norm and the residual add happen together, three times per layer
(Ch 08 Part B).

## 3.3 QK-norm, and why attention scale becomes 1.0

Gemma4 normalizes Q and K **per head**, before RoPE:

```text
Q_h = RMSNorm_{N=head_dim}(Q_h) * q_norm_weight
K_h = RMSNorm_{N=head_dim}(K_h) * k_norm_weight
```

Now derive the consequence. Classic attention divides scores by `√d`:

```text
score = (q · k) / √d
```

The reason for `√d` is variance control: if `q` and `k` have independent
components with variance 1, then `q · k` is a sum of `d` such products, so its
variance is `d` and its magnitude is `~√d`. Divide by `√d` to keep scores O(1)
so softmax does not saturate.

But QK-norm *already* fixes the magnitude of `q` and `k` — with learned
per-channel gains that the model tuned during training. The `1/√d` correction
is now redundant, and worse, it double-corrects. So:

```text
Gemma4: attention_scale = 1.0        # QK-norm handles scaling
```

There is a comment saying exactly that in `gemma4_gpu_model.rs`, and you can
see `scale = 1.0f32` at the prefill attention call site (Ch 10 Part D.2).

**Mental model: QK-norm is a learned replacement for the hand-picked `1/√d`.**

If you port a kernel from a codebase that hardcodes `1/√d`, you must remove
it. And if you see attention output that is too flat (nearly uniform weights),
suspect a stray `1/√d` still dividing.

V is also normalized — with **no** learned weight
(`rmsnorm_per_head_noweight`).

### Exercises 3

1. Write RMSNorm for `x = [3, 4]`, `w = [1, 1]`, ε = 0. Then for `w = [2, 0.5]`.
2. Two vectors differ only in magnitude (`x` and `10x`). What does RMSNorm
   produce for each? What information has been discarded?
3. `q` and `k` have i.i.d. components with variance 1 and `d = 128`. What is
   the standard deviation of `q · k`? Now with `d = 512`. Explain why `1/√d`
   is the right correction and why QK-norm makes it redundant.
4. You keep both QK-norm and `1/√d` with d=512. By what factor are your scores
   too small? What happens to the softmax distribution?

---

# Part 4 — Attention, derived

## 4.1 The problem attention solves

Position `i` needs information from earlier positions, but *which* earlier
positions depends on content, not on distance. "The **cat** that the dog
chased was **grey**" — "grey" needs "cat", four words back through an
intervening clause. A fixed-window convolution cannot express that; a
content-addressed lookup can.

So build a soft dictionary lookup:

```text
each position j publishes:  a key k_j (what I am about)  and a value v_j (what I offer)
each position i asks:       a query q_i (what I need)
```

Three design decisions turn that into attention:

**Decision 1 — similarity is a dot product.** `q · k` is large when the
vectors align. It is cheap (one FMA per dimension), differentiable, and
expressible as a matmul, which is what hardware is fastest at.

**Decision 2 — the lookup is soft.** A hard argmax over keys is not
differentiable and throws away partial matches. Softmax gives a weighted
average that is differentiable everywhere and reduces to argmax as scores
sharpen:

```text
α_j = exp(s_j) / Σ_k exp(s_k)         # α ≥ 0, Σα = 1
```

Softmax specifically (rather than, say, normalizing the scores directly) has
two properties the model wants: it is positive, and it is
translation-invariant in the scores (`s + c` gives the same `α`), which makes
the numerical trick in 4.5 possible.

**Decision 3 — the output is a convex combination of values.** `out_i = Σ_j
α_{i,j} v_j` lives in the convex hull of the values, so it cannot explode.

Together:

```text
score_{i,j} = (q_i · k_j) * scale       # scale = 1.0 for Gemma4 (Part 3.3)
α_{i,·}     = softmax over allowed j
out_i       = Σ_j α_{i,j} v_j
```

## 4.2 Masks: causal and sliding

**Causal.** For generation, position `i` must not see `j > i`, or the model
would train to cheat and be unusable at inference. Implement by setting
disallowed scores to `−∞` before the softmax (`exp(−∞) = 0`).

**Sliding window.** Additionally require `j > i − W`. Now attention is local,
and per-token cost stops growing with context. Gemma4 uses sliding layers for
most of its depth and full-attention layers periodically, which is Ch 02's
main subject.

Note what the mask does to cost: a materialized `[S, S]` score matrix at
S=4096 is 16.7M floats = 67 MB, most of which is masked away. That is why real
kernels never materialize it — they compute tiles and apply the mask inside
the kernel from index arithmetic (Ch 10 Part D.2).

## 4.3 Heads, and why more than one

One attention head can only compute one weighted average per position. But a
token often needs several different things at once — its syntactic subject,
the topic, the nearest quotation mark. So run `n_q` independent attentions in
parallel, each with its own projections and its own `d = head_dim` subspace,
then concatenate and mix:

```text
Q = X W_Q      # [S, n_q  * d]
K = X W_K      # [S, n_kv * d]
V = X W_V      # [S, n_kv * d]
out_h = Attn(Q_h, K_h, V_h)   for each head h
out   = concat(out_0 … out_{n_q-1}) W_O      # [S, hidden]
```

`W_O` is what lets heads interact; without it, each head would write to a
disjoint slice of the residual stream.

## 4.4 GQA, with the memory argument

In classic MHA, `n_kv = n_q`: every query head has its own K/V. Then the KV
cache stores `n_q` heads per layer per position.

**That is the problem.** Cache size is:

```text
2 (K and V) × n_kv × head_dim × seq × layers × bytes_per_element
```

For E4B-like dimensions with `n_q = 20`, F16, 8192 context, 42 layers, and
head_dim 128, MHA would need:

```text
2 × 20 × 128 × 8192 × 42 × 2 B ≈ 3.5 GB     per sequence
```

Per sequence. Four concurrent requests would need 14 GB of KV cache alone.

**GQA** fixes it: use fewer KV heads than query heads, and let groups of query
heads share:

```text
num_kv_groups = n_q / n_kv                    # E4B: 20 / 4 = 5
kv_head_for(q_head) = q_head / num_kv_groups
```

With `n_kv = 4` the same cache is `4/20 = 1/5` the size: ~0.7 GB instead of
3.5 GB. Quality cost is small; memory saving is 5×. Nobody serves MHA at long
context anymore.

```129:136:src/gemma4_config.rs
pub fn num_kv_groups(&self) -> usize {
    self.num_attention_heads / self.num_key_value_heads
}

pub fn layer_num_kv_groups(&self, layer_idx: usize) -> usize {
    let kv = self.layer_num_kv_heads(layer_idx);
    if kv == 0 { 1 } else { self.num_attention_heads / kv }
}
```

GQA also changes the *shape of the computation*, which matters for kernels:
five query heads want the same K/V rows. You can either dispatch per query
head and read the KV tile five times (simple, default), or dispatch per KV
head and share the tile across five queries in threadgroup memory (bandwidth
win, but the sharing protocol must match the cache layout). The second is
`ATTENTION_GQA_Q4=1`, and it produced garbage when it was made default —
`AGENTS.md` #11, discussed in Ch 02 Part 3.

An incorrect `q → kv` mapping is the canonical silent bug here: every head
still gets *some* plausible K/V, so output stays grammatical.

## 4.5 Softmax numerically, and the online form

### Stability first

```text
α_j = exp(s_j) / Σ_k exp(s_k)
```

`exp(90)` overflows f32. Use translation invariance — subtract the max:

```text
m = max_k s_k
α_j = exp(s_j − m) / Σ_k exp(s_k − m)
```

Now the largest exponent is `exp(0) = 1`, and the denominator is in `[1, n]`.
Never implement softmax without this.

### The online (flash) form

Stable softmax appears to need two passes: one to find `m`, one to accumulate.
For attention that would mean materializing all `kv_seq` scores — exactly what
we cannot afford. Flash attention removes the need by carrying running state
and *retroactively correcting* it.

State: running max `m`, running denominator `ℓ`, running output accumulator
`acc` (unnormalized).

For each new tile with scores `s` and values `V`:

```text
m_new = max(m, max(s))
r     = exp(m − m_new)               # rescale factor for old state
ℓ     = ℓ * r + Σ_j exp(s_j − m_new)
acc   = acc * r + Σ_j exp(s_j − m_new) * V_j
out   = acc / ℓ                      # only at the very end
```

The key insight in one line: **all previously accumulated terms were divided
by `exp(m_old)`; multiplying by `r = exp(m_old − m_new)` converts them to be
divided by `exp(m_new)` instead.** Both `ℓ` and `acc` must be rescaled, since
both carry that factor.

### Worked example — prove it

Scores in two tiles: tile0 = `[1, 3]`, tile1 = `[2]`. Values `V = [10, 20, 30]`.

One-shot:

```text
m = 3
e = [e^{-2}, e^{0}, e^{-1}] ≈ [0.135, 1.000, 0.368]
ℓ = 1.503
α ≈ [0.090, 0.665, 0.245]
out ≈ 0.090·10 + 0.665·20 + 0.245·30 ≈ 21.55
```

Online:

```text
after tile0:  m=3,  ℓ = 0.135 + 1 = 1.135
              acc = 0.135·10 + 1·20 = 21.35

tile1 (s=2):  m_new = max(3, 2) = 3      → r = exp(0) = 1
              ℓ   = 1.135·1 + e^{-1} = 1.503
              acc = 21.35·1 + 0.368·30 = 32.39
              out = 32.39 / 1.503 ≈ 21.55        ✓ identical
```

Now the interesting case — tile1 has score **5**, which raises the max:

```text
m_new = 5,  r = exp(3 − 5) = e^{-2} ≈ 0.135
ℓ   = 1.135·0.135 + exp(0)      = 0.153 + 1     = 1.153
acc = 21.35·0.135 + 1·30        = 2.88 + 30     = 32.88
out = 32.88 / 1.153 ≈ 28.5      # dominated by the new high-scoring value ✓
```

Sanity check that by hand with one-shot softmax over `[1, 3, 5]` and
`V = [10, 20, 30]` — you should get the same number. Do it; that check is
what makes the invariant stick.

**Forget the rescale of `acc` and you get attention mass that drifts as
context grows** — fine on short prompts, wrong on long ones. This engine keeps
`(m, ℓ, old_factor, inv_ℓ)` in a four-element threadgroup array
(`shared_update[4]`) and applies them in `flash_softmax_tile`; Ch 07 Part D
walks the code.

## 4.6 Prefill vs decode algebra

**Prefill** (many queries): `out = softmax(Q Kᵀ ⊙ mask) V` — genuinely a pair
of matmuls, tiled so the `[S, S]` intermediate never exists in memory.

**Decode** (one query): there is no `Q Kᵀ` matmul, only a matrix-vector
product against the cache:

```text
for t in [kv_start, kv_seq):  s_t = scale * (q · k_t)
α = softmax(s)
out = Σ_t α_t v_t
```

And one extra obligation that prefill does not have: **append the new `(k, v)`
to the cache** so the next token can see this one. Whether that append is done
by the attention kernel or by a separate dispatch is the source of the most
instructive bug in `AGENTS.md` (#15) — see Ch 06 Part D.1.

### Exercises 4

1. Compute softmax of `[2, 4, 6]` by hand, stably. Then of `[102, 104, 106]`.
   Confirm they are equal and explain why.
2. Run the online algorithm on three tiles: `[1]`, `[4]`, `[2]`, with
   `V = [10, 20, 30]`. Show `(m, ℓ, acc)` after each tile and verify against
   one-shot.
3. Redo exercise 2 but "forget" to rescale `acc` (rescale only `ℓ`). How wrong
   is the answer? Does it get worse with more tiles?
4. E4B: `n_q = 20`, `n_kv = 4`. Which KV head does query head 13 use? Which
   query heads share KV head 2?
5. Compute the F16 KV cache size for MHA (`n_kv = 20`) vs GQA (`n_kv = 4`) at
   head_dim 128, 8192 context, 42 layers. How many concurrent sequences fit in
   8 GB in each case?
6. Why does an incorrect `q → kv` head mapping produce grammatical output
   rather than a crash?

---

# Part 5 — RoPE

## 5.1 Why rotation instead of addition

Absolute position embeddings add a learned vector: `h = E[t] + p_pos`. Two
problems. The model must learn what each absolute index means, and it
generalizes poorly past training lengths.

What attention actually needs is **relative** position: "how far back is this
key?" So find an encoding where the *score* depends only on `m − n`.

RoPE does it with rotation. Take a pair of channels as a complex number and
rotate it by an angle proportional to position:

```text
z = x_a + i·x_b        # a pair of channels
z' = z · e^{i·pos·θ}   # rotate by pos·θ
```

Now the inner product of a rotated query at position `m` with a rotated key at
position `n`:

```text
⟨z_q e^{i m θ}, z_k e^{i n θ}⟩ = Re( z_q e^{i m θ} · conj(z_k e^{i n θ}) )
                                = Re( z_q conj(z_k) · e^{i (m−n) θ} )
```

The absolute positions cancel; only `m − n` survives. That is the whole
theorem, and it is worth deriving once by hand because it explains the
constraints: the rotation must be applied to **both** q and k, with the same
θ per channel pair, or the cancellation fails.

Different channel pairs get different θ (geometrically spaced), so the model
sees position at many frequencies at once — fast-rotating pairs resolve nearby
offsets, slow-rotating pairs carry long-range information.

## 5.2 The exact formulas this engine uses

For half-index `d ∈ [0, head_dim/2)`:

```text
inv_freq_d = 1 / (θ_base^{2d / head_dim}) / rope_factor
angle_d    = pos * inv_freq_d
q'[d]        = q[d]        * cos(angle_d) − q[d + half] * sin(angle_d)
q'[d + half] = q[d + half] * cos(angle_d) + q[d]        * sin(angle_d)
```

That is a 2×2 rotation applied to the pair `(q[d], q[d + half])`.

**Pairing matters.** This is **Neox style**: channel `d` pairs with `d +
head_dim/2`. The alternative (GPT-J style) pairs adjacent channels `(0,1),
(2,3), …`. Both are valid rotations; they are *not* interchangeable, because
the weights were trained against one specific pairing. Use the wrong one and
every score is subtly wrong — fluent nonsense again.

From the table-fill kernel:

```1987:2001:src/shaders/llama.metal
    if (d < p.rope_angles) {
        float inv_freq = 1.0f / (pow(p.theta, 2.0f * float(d) / float(p.head_dim))) / p.factor;
        float angle = pos * inv_freq;
        float c = cos(angle);
        float s = sin(angle);
        cos_packed[base + d] = c;
        cos_packed[base + d + half_dim] = c;
        sin_packed[base + d] = s;
        sin_packed[base + d + half_dim] = s;
    } else {
        cos_packed[base + d] = 1.0f;
        // ...
        sin_packed[base + d] = 0.0f;
```

Two implementation details in that snippet are worth naming:

**The tables are written twice**, at `d` and at `d + half_dim`. Both halves of
the pair need the same `cos`/`sin`, and storing both means the consuming
kernel indexes by channel with no branch or extra arithmetic — it just reads
`cos[i]` for whatever channel `i` it holds. Memory for simplicity: a good
trade in a kernel that runs 42× per token.

**Partial rotary is implemented as identity, not as a branch.** When
`d ≥ rope_angles`, the kernel writes `cos = 1, sin = 0`, which makes the
rotation the identity for those channels. So `partial_rotary_factor = 0.25`
(only a quarter of the head's channels carry position) costs the consuming
kernel exactly nothing — no divergence, no bounds check. All the complexity
lives in the table.

That is a pattern worth stealing: **push conditionals into precomputed data
when the data is small and the consumer is hot.**

## 5.3 Worked micro-numbers

`head_dim = 4`, `θ = 10000`, `pos = 1`, full rotary so `rope_angles = 2`:

```text
half = 2, so pairs are (q0, q2) and (q1, q3)

d=0: inv_freq = 1/10000^{0/4·2} = 1/10000^0    = 1
     angle = 1·1 = 1 rad        → cos ≈ 0.540, sin ≈ 0.841
d=1: inv_freq = 1/10000^{2/4}   = 1/100        = 0.01
     angle = 0.01               → cos ≈ 1.000, sin ≈ 0.010

q = [1, 2, 3, 4]
q'0 = q0·cos0 − q2·sin0 = 1·0.540 − 3·0.841 = −1.983
q'2 = q2·cos0 + q0·sin0 = 3·0.540 + 1·0.841 =  2.461
q'1 = q1·cos1 − q3·sin1 = 2·1.000 − 4·0.010 =  1.960
q'3 = q3·cos1 + q1·sin1 = 4·1.000 + 2·0.010 =  4.020
```

Observe: the `d=0` pair rotated a lot (1 radian), the `d=1` pair barely moved
(0.01 rad). At `pos = 100` the `d=1` pair would rotate 1 rad and the `d=0`
pair would have wrapped around 100/2π ≈ 16 times. That spread is the point —
high-frequency pairs encode local offsets, low-frequency pairs encode global
ones.

Now verify the relative-position property numerically: that is Exercise 3
below, and it is the single most convincing way to believe RoPE.

## 5.4 Gemma4's two RoPE configurations

| Layer type | head_dim | θ_base | Partial rotary |
|---|---|---|---|
| Sliding | 128 | ~10 000 | full (factor 1.0) |
| Full / global | 512 | ~1 000 000 | **0.25** |

Why a larger θ for full-attention layers: a bigger base makes all frequencies
slower, so angles stay distinguishable out to much longer distances. Sliding
layers only ever look back `W` tokens, so they can afford fast frequencies and
finer local resolution. Full layers must resolve offsets of tens of thousands,
so they need slow ones.

```162:166:src/gemma4_config.rs
pub fn full_partial_rotary_factor(&self) -> f64 {
    self.rope_parameters.as_ref()
        .and_then(|r| r.full_attention.as_ref())
        .map_or(0.25, |c| c.partial_rotary_factor)
}
```

RoPE is applied **after** QK-norm (norm the vector, then rotate it — rotation
preserves norm, so the order is not arbitrary but it is also not reversible in
effect). Fusing norm + RoPE + attention into one kernel is a real speed win
and a real correctness boundary: `AGENTS.md` #11's follow-up found that
*splitting* QK-norm+RoPE out of the attention kernel changed results.

## 5.5 Decode vs prefill table fill

- **Decode:** one dispatch, `rope_fill_decode`, fills a packed buffer
  `cos_packed[layer * max_head_dim + d]` for the current position, for all
  layers at once (Ch 09).
- **Prefill:** `rope_fill_prefill_batch` fills a per-layer table covering
  `[start_pos, start_pos + S)` (Ch 10 Part C.3).

Per-layer tables because θ and rotary dims differ by layer type. Reusing one
layer's table for another is a bug whose signature is "the last layer's angles
applied everywhere," and because command buffers execute *after* encoding, a
table overwritten mid-encode affects earlier dispatches too. That class of bug
bit the MTP draft path historically.

### Exercises 5

1. `head_dim = 8`, θ = 10000. Compute `inv_freq_d` for `d = 0..3`. Which pair
   rotates fastest?
2. With `head_dim = 512` and `partial_rotary_factor = 0.25`, how many half-
   indices get real trig? How many channels are pass-through?
3. **The important one.** Take `q = k = [1, 0, 1, 0]` (head_dim 4, θ=10000).
   Rotate `q` to position 5 and `k` to position 3, take the dot product. Now
   rotate `q` to 12 and `k` to 10 and take the dot product. Same? Explain in
   terms of 5.1.
4. Swap Neox pairing for GPT-J pairing in your head: for head_dim 8, list the
   channel pairs under each scheme. Why can't weights trained on one be used
   with the other?
5. Why does the kernel write `cos` into both `d` and `d + half_dim` instead of
   having the consumer compute the index?

---

# Part 6 — The MLP

## 6.1 What it is for

Attention moves information *between* positions. It does no per-position
nonlinear computation to speak of — the output is a convex combination of
values. The MLP is where each position transforms its own content: it is the
"thinking" step, and it holds most of the parameters.

```text
gate = W_gate · h            # [hidden] → [intermediate]
up   = W_up   · h            # [hidden] → [intermediate]
mid  = GeLU(gate) ⊙ up       # elementwise
out  = W_down · mid          # [intermediate] → [hidden]
```

E4B: `2560 → 10240 → 2560`. The 4× expansion is standard: project into a wider
space where features are more separable, apply a nonlinearity, project back.

## 6.2 Why gating

A plain FFN is `W_2 · act(W_1 h)` — two matrices. The gated variant computes
**two** parallel projections and lets one modulate the other multiplicatively.
`up` is the content; `GeLU(gate)` is a soft, learned per-channel mask over it.

Cost: one extra `[intermediate, hidden]` matrix (50% more MLP parameters and
50% more MLP bandwidth). Benefit: better quality per parameter, consistently
enough that essentially every modern LLM uses a gated MLP.

Gemma uses **GeLU** as the activation. Llama-style models use SiLU/Swish
(`SwiGLU`). They are similar in shape and not interchangeable — the weights
were trained against one.

## 6.3 GeLU exactly

```text
GeLU(x) = x · Φ(x)                            # Φ = standard normal CDF
        ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))   # the tanh approximation
```

Intuition: GeLU is a smooth gate. For very negative `x`, `Φ(x) ≈ 0` and the
unit is off; for very positive, `Φ(x) ≈ 1` and it passes through. Unlike ReLU
it is smooth at 0, and unlike ReLU it is slightly negative for small negative
inputs — a small amount of signal survives.

The engine uses the **tanh approximation** (`gelu_pytorch_tanh`), matching
PyTorch's `approximate='tanh'`, which is what Gemma was exported with. The
exact `erf` version differs by ~1e-3, which is enough to flip token choices on
close calls. Match the training stack (Ch 08 Part C.3).

## 6.4 Where the bytes are

Per layer, Q4_0 (4.5 bits/weight), E4B:

```text
gate: 10240 × 2560 × 4.5/8 ≈ 14.7 MB
up:   same                 ≈ 14.7 MB
down: 2560 × 10240         ≈ 14.7 MB
                     total ≈ 44 MB per layer
× 42 layers                ≈ 1.85 GB
```

Against ~2.5 GB total per decode token, the MLP is roughly **three quarters of
all weight traffic**. That is why Ch 05 (quantization) and Ch 08's fusion
ladder exist, and why prefill phase timing (Ch 10 Part H) puts MLP at 54%.

### Exercises 6

1. Compute GeLU(−2), GeLU(0), GeLU(1), GeLU(3) using the tanh form. Sketch it.
2. Why is `GeLU(gate) ⊙ up` more expressive than `GeLU(W₁h)` alone? Describe
   what a channel of `gate` can do to a channel of `up`.
3. Compute the F16 byte count for the same three matrices. What decode rate
   ceiling does that imply at 200 GB/s (MLP only, 42 layers)?
4. Fusing gate and up into one kernel avoids writing two `[10240]` scratch
   buffers. How many bytes does that save per layer? Compare to the 44 MB of
   weights read. What does that tell you about the value of fusion here?

---

# Part 7 — Assembling one Gemma4 block

## 7.1 The generic block, for contrast

```text
h ← h + Attn(Norm(h))
h ← h + MLP(Norm(h))
```

## 7.2 What Gemma4 actually does

```text
# ── Attention branch ─────────────────────────────────────────────
n    = input_layernorm(h)
Q    = W_q · n                                  # always
K, V = W_k · n, W_v · n                         # ONLY if layer.has_kv
Q    = RoPE( QK-norm(Q) )                       # per-head norm, then rotate
K    = RoPE( QK-norm(K) )                       # if has_kv
V    = V-norm(V)                                # no learned weight
append (K, V) → cache[layer.kv_source_layer]    # if has_kv
attn = Attention(Q, K_cache[src], V_cache[src]) # src = kv_source_layer
o    = W_o · attn
h   ← h + post_attention_norm(o)

# ── MLP branch ───────────────────────────────────────────────────
n    = pre_feedforward_layernorm(h)
m    = W_down · ( GeLU(W_gate · n) ⊙ (W_up · n) )
h   ← h + post_feedforward_layernorm(m)

# ── PLE branch (Gemma4-specific) ─────────────────────────────────
g    = GeLU(W_ple_gate · h) ⊙ context[layer]    # context from the pre-pass
p    = W_ple_proj · g
h   ← h + post_per_layer_input_norm(p)

# ── Depth stabilization ──────────────────────────────────────────
h   ← h * layer_scalar
```

Four things here have no equivalent in a Llama block, and each one is a place
where a from-scratch port silently goes wrong:

**1. Double norms around every sub-block** (Part 3.2). Three `rmsnorm_acc`
calls per layer.

**2. PLE** — a third residual contribution, per layer, that re-injects
token identity mixed with a projection of the current state. Ch 08 Part D
builds it; the short version is that each layer gets its own small
token-conditioned bias, so token identity does not have to survive 42 layers
of mixing in the residual stream alone.

**3. Shared KV.** The last `num_kv_shared_layers` layers have `has_kv =
false`: they compute Q, they attend, but they never compute or store K/V —
they read an earlier layer's cache. This is the engine's invariant #1, and it
touches loading (no `attn_k`/`attn_v` tensors exist), append (skip), attention
(read from `kv_source_layer`), and stride math (use the anchor's head_dim).
Ch 02 Part 5 and Ch 06 Part F.

**4. `layer_scalar`.** A per-layer constant multiply. Trivially easy to omit;
produces slow compounding drift over 42 layers.

## 7.3 Why the order cannot be permuted

Some orderings are load-bearing:

- **QK-norm before RoPE.** Normalize, then rotate. Rotation preserves norm, so
  doing it the other way changes nothing about magnitudes — but it changes
  *which* vector the learned per-channel norm weights are applied to, and the
  weights were trained one way.
- **Append after computing K, before (or fused with) attention.** The current
  token must be visible to its own attention. Ch 06 Part D.1.
- **V-norm before append**, since the cache stores post-norm values.
- **PLE after the MLP branch**, reading the already-updated `h`.

Everything in this list is a one-line change that yields plausible output. You
cannot debug them by reading generated text; you need reference comparison
(Ch 15 Part B.5).

### Exercises 7

1. Redraw 7.2 from memory. Check which sub-blocks read `h` and which read a
   normed copy.
2. Layer 30 is shared-KV with anchor 22. List every step in 7.2 that layer 30
   skips and every step it still performs.
3. You swap QK-norm and RoPE. Which mathematical property makes the *norms* of
   Q and K unaffected? What is still wrong?
4. Count the RMSNorm invocations in one layer (including per-head ones). Now
   multiply by 42.

---

# Part 8 — Logit softcapping

After the final norm and `lm_head`:

```text
logits ← cap · tanh(logits / cap)          # cap = 30 for Gemma4
```

What this does: `tanh` is roughly the identity near 0 and saturates at ±1, so
small logits pass through nearly unchanged while extreme logits are squashed
into `(−cap, +cap)`.

```text
logit  1 → 30·tanh(0.033) ≈  1.0     (unchanged)
logit 10 → 30·tanh(0.333) ≈  9.6     (slightly compressed)
logit 60 → 30·tanh(2.0)   ≈ 29.0     (heavily compressed)
logit 300 → ≈ 30                     (saturated)
```

Why bother: it bounds confidence. A model that outputs a logit of 300 for one
token has effectively zero entropy — sampling, temperature, and penalties all
stop having any effect. Softcapping keeps the distribution manipulable and
was part of Gemma's training, so **it is not optional at inference** — remove
it and your sampling behaves differently from the reference implementation.

Read from GGUF metadata as `gemma4.final_logit_softcapping` (default 30.0,
Ch 03 Part B). Applied on GPU in the sample path, or on CPU after readback in
the logits path.

### Exercises 8

1. Compute the softcapped value of logits 5, 20, 45, 100 with cap 30.
2. Two tokens have logits 60 and 55. Compute their softmax ratio before and
   after softcapping. Did the model become less certain?
3. Why would omitting softcap make `temperature` and `min_p` less effective?

---

# Part 9 — The KV cache, derived

## 9.1 The complexity argument

Without a cache, producing token `T` means running the whole model over `T`
positions: `O(T)` work in the MLP and `O(T²)` in attention, per layer. Doing
that for every generated token gives `O(T³)` total for the sequence.

With a cache, generating token `T` costs:

```text
MLP, norms, projections:  O(1) in T      # one row of activations
attention:                O(T)           # one query against T cached keys
```

So per-token cost is linear in context, and total is `O(T²)` — but with a tiny
constant, because the `O(T)` part (reading the cache) is much cheaper than the
`O(1)`-in-T part (reading all the weights) until context gets long.

That last point is the interesting one, and it is exactly what `AGENTS.md` is
about: at 25 tokens of context, weight reads dominate completely and this
engine matches llama.cpp. At 200+, the attention term is large enough that
*how well your attention kernel scales* starts to matter, and a 12% gap opens.

## 9.2 What must be stored

For each layer that owns KV, each position, each KV head: one `K` vector and
one `V` vector of `head_dim` values. Post-RoPE and post-norm (Part 7.3), so
nothing needs recomputing on read.

```text
bytes = 2 × n_kv × head_dim × seq × layers × bytes_per_element
```

F16, E4B sliding dims, 8192 context, 42 layers:

```text
2 × 4 × 128 × 8192 × 42 × 2 B ≈ 705 MB
```

And full-attention layers use head_dim 512, so they cost 4× that per layer —
Ch 13 Part A.2 does the mixed calculation properly (~1.06 GB per sequence in
F16, ~0.30 GB in Q4_0).

## 9.3 Quantizing the cache

The cache is read every token for every layer, so it is bandwidth in exactly
the same way weights are. Quantize it with the same block format:

```86:93:src/gemma4_config.rs
pub fn bytes_per_row(&self, head_dim: usize) -> usize {
    assert!(head_dim % 32 == 0, "head_dim must be a multiple of 32 for quantized KV cache");
    match self {
        KvCacheType::F16 => head_dim * 2,
        KvCacheType::Q8_0 => (head_dim / 32) * 34,
        KvCacheType::Q4_0 => (head_dim / 32) * 18,
    }
}
```

Q4_0 is ~3.5× smaller than F16 (18 bytes per 32 values vs 64). Cost: the
attention kernel must dequantize on the fly, and stored keys lose precision.
In practice quality holds up because attention is an average and errors
partially cancel — but verify on your workload, not on faith.

Env: `LLAMA_KV_CACHE_TYPE=q4_0`. Layout and address math: Ch 06 Part C.

### Exercises 9

1. Derive the `O(T³)` figure for cache-free generation of `T` tokens. Which
   term dominates?
2. Compute per-token bytes read at context 200 and context 4000 for E4B with
   Q4_0 KV: weights (~2.5 GB) plus cache. At what context does the cache reach
   10% of the total?
3. Why can K be stored post-RoPE? What would you have to store instead if you
   wanted to be able to re-position a cached prefix?
4. Q8_0 uses 34 bytes per 32 values. What is the overhead per value versus
   plain int8, and where does it go?

---

# Part 10 — Sampling

Given `logits ∈ R^vocab`, choose a token.

**Greedy / argmax.** Take the max. Deterministic, and the baseline that
speculative decoding must reproduce exactly (Ch 14 Part A).

**Temperature.** `logits ← logits / T` before softmax. `T < 1` sharpens
(more deterministic), `T > 1` flattens. `T → 0` becomes argmax.

**Top-k.** Keep the `k` largest logits, renormalize over just those. Note
this changes probabilities, not just candidates — which is exactly the
subtlety that broke MTP's draft-confidence threshold in `AGENTS.md` M6: a
top-10 softmax and a full-262k softmax give very different probabilities for
the same token.

**Min-p.** Keep tokens with `p ≥ min_p × p_max` — an adaptive cutoff. When the
model is confident, few tokens survive; when it is unsure, many do. Generally
better behaved than a fixed top-k.

**Repetition / frequency penalties.** Subtract from the logits of tokens
already generated. This is why `ActiveRequest` carries `generated_tokens`
(Ch 12 Part A.2) — sampling is stateful in the request, not just in the
logits.

In this engine sampling is CPU-side after logits readback (`sampling.rs`,
called from `prepare_decode_token`), with an optional GPU argmax path. At 262k
vocab this is not free, but it is not the bottleneck either — Ch 12 Part G
notes it becomes measurable at large batch.

Policy on top of sampling — EOS floors and first-token blocklists — lives in
the scheduler, not here (Ch 12 Part D.1).

### Exercises 10

1. Logits `[1, 2, 3]`. Compute the softmax at `T = 1`, `T = 0.5`, `T = 2`.
2. A distribution has `p_max = 0.6`. With `min_p = 0.1`, what is the
   probability cutoff? Now with `p_max = 0.15`. How many candidates survive in
   each case, qualitatively?
3. Vocab 262144. A token has logit 20; the next best is 15; the remaining
   262142 are near 0. Compute its probability under full-vocab softmax and
   under top-10 softmax. Why does the difference break a fixed `p_min`?

---

# Part 11 — Worked shapes, one decode token

E4B, sliding layer:

```text
h                    [2560]
n = norm(h)          [2560]
Q = W_q n            [20 × 128] = [2560]
K, V                 [4 × 128]  = [512] each
KV cache row (Q4_0)  4 heads × (128/32) × 18 B = 288 B per position, per tensor
scores               [kv_seq] per query head
attn out             [20 × 128] = [2560]
o = W_o attn         [2560]
gate, up             [10240] each
mid                  [10240]
down                 [2560]
ple gate             [256]
ple proj             [2560]
logits               [262144]
```

Full-attention layer, same model: head_dim 512, so `Q` is `[20 × 512] =
[10240]`, `K/V` are `[4 × 512] = [2048]`, and one Q4_0 cache row is
`4 × 16 × 18 = 1152` bytes. **Four times the attention work and cache traffic
on those layers** — which is why kernels are compiled per head_dim (`h128`,
`h256`, `h512`) and why scratch must be sized for the maximum (Ch 17 Part A).

---

# Part 12 — Where each concept lives in this repo

| Concept | File | Chapter |
|---|---|---|
| Config, GQA, RoPE params, KV type | `gemma4_config.rs` | 02, 17 |
| Embedding, forward decode/prefill | `gemma4_gpu_model.rs` | 09, 10 |
| Per-layer fused encode | `decode_fused.rs` | 09 |
| Norms, RoPE, attention, KV kernels | `shaders/llama.metal` | 06, 07, 08 |
| Quantized matvec / matmul | `shaders/ggml_mul_mv_q4.metal`, `ggml_mul_mm_q4.metal` | 05 |
| Sampling | `sampling.rs` | 10 (this ch), 12 |
| Serving loop | `scheduler.rs`, `server.rs` | 11, 12 |

Do not open a kernel until you can redraw Part 7.2 from memory. The kernels
are fusions of those steps, and they are unreadable if you do not already know
which steps they are fusing.

---

## Checklist (closed book)

- [ ] Explain prefill vs decode, with bytes-per-token for each.
- [ ] Derive why the embedding is scaled by `√hidden_size`.
- [ ] Explain the residual stream as a bus, and the three bug modes in 2.3.
- [ ] Derive RMSNorm from LayerNorm and justify each omission.
- [ ] Explain why `attention_scale = 1.0` follows from QK-norm.
- [ ] Derive attention from "content-addressed lookup" in three decisions.
- [ ] Write the GQA mapping for `n_q=20, n_kv=4` and the memory it saves.
- [ ] Run the two-tile online softmax including the max-increase case.
- [ ] Prove RoPE's relative-position property, and write the Neox update.
- [ ] Explain how `partial_rotary_factor` is implemented with no branch.
- [ ] State what shared-KV layers skip and what they still do.
- [ ] Compute Q4_0 KV bytes per position for both E4B layer types.

**Next:** [00c_gpu_and_metal_fundamentals.md](00c_gpu_and_metal_fundamentals.md)
