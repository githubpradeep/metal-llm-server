# Mixture of Experts in Gemma 4 A4B

*Learning notes on how sparse MLPs show up in this Metal inference engine.*

Dense transformers activate all parameters on every token. As models grow, that coupling of *capacity* (how much the model can store) and *compute* (how much work each token does) becomes expensive on device. **Mixture of Experts (MoE)** weakens the coupling: keep a large pool of specialist feed-forward networks (“experts”), but route each token through only a few of them. Total parameter count can be large; FLOPs per token stay closer to a smaller dense model.

This note explains the MoE design used for **Gemma 4 26B-A4B** in this repository—what the architecture is doing conceptually, how routing and combination are defined, and how those ideas are mapped onto Metal execution, quantized weights, and a small expert cache sized for machines with limited unified memory (e.g. 16 GB).

It is intentionally *not* a survey of Switch Transformer, Mixtral, or DeepSeek MoE. Those systems share the sparse-MLP idea but differ in gating, activation functions, and attention stacks. Here we stay close to what the code implements.

> Equations below use plain text / Unicode so they render in Cursor, GitHub, and terminals without a LaTeX plugin.

---

## Motivation

### Capacity without proportional FLOPs

For a dense Transformer, a rough rule of thumb is that decode FLOPs per token scale like **2 × (active parameters)**. If every weight participates on every token, bigger models are strictly more expensive at inference.

MoE changes the accounting. Let **N** be the number of experts and **k** the number activated per token. Ignoring shared layers for a moment:

```text
active FFN params  ≈  (k / N) × total expert params
```

Gemma 4 A4B is marketed in that spirit: on the order of **26B total** parameters with roughly **4B active**—enough capacity to specialize, without paying dense-26B bandwidth on every step.

### What stays dense

A subtle but important point for reading this codebase: **MoE does not sparsify attention**. QKV projections, attention over the KV cache, RoPE, and residual connections around attention behave like dense Gemma 4 (E2B/E4B). Sparsity lives in the **MLP / FFN block** after attention.

So when debugging “MoE is broken,” it helps to ask first whether the failure is in attention (shared with dense models) or in the sparse MLP path (A4B-specific).

---

## Background: the dense MLP we are replacing

In a standard decoder layer, after attention and its residual, the feed-forward block is usually:

```text
h' = h + W_down · ( σ(W_gate · Norm(h)) ⊙ (W_up · Norm(h)) )
```

where **σ** is an elementwise nonlinearity (here **GeLU**), **⊙** is elementwise multiply, and **W_gate**, **W_up**, **W_down** are the gate, up, and down projections. That entire FFN runs for every token.

MoE replaces *this* block’s compute pattern—not the surrounding residual stream discipline.

---

## Gemma 4 A4B MoE: shared expert + routed experts

Gemma’s A4B design is closer to “dense FFN in parallel with a sparse MoE” than to “experts only.”

For a hidden state **h** in R^d (A4B uses d ≈ 2816):

1. **Shared FFN** S(·) — always evaluated (same role as a normal MLP).
2. **Router** — produces scores over **N** experts.
3. **Top-k experts** {E_i₁, …, E_iₖ} — evaluated and mixed.
4. **Combine** — normalized shared and routed branches are added, then folded back into the residual stream.

Schematically:

```text
        h
        │
   attention (+ residual)
        │
        ├──────────────────────┐
        ▼                      ▼
   shared FFN S(h)        router → top-k experts
        │                      │
   Norm₁(S)               Norm₂(Σ w_j E_{i_j})
        │                      │
        └──────────┬───────────┘
                   ▼
              h ← h + Norm(·)
```

Dense E2B/E4B layers simply leave the MoE branch absent (`moe = None`) and run only **S**.

### Routing

Let **N** be `num_experts` and **k** be `num_experts_used` (from GGUF metadata `gemma4.expert_count` / `gemma4.expert_used_count`).

The router is a linear map after a weightless RMSNorm and a learned scale **s** (stored prefolded with 1/√d):

```text
z = W_r · ( s ⊙ RMSNorm_∅(h) )     ∈ R^N
```

We convert logits to probabilities with a full softmax, then keep the top **k**:

```text
p_i = exp(z_i) / Σ_j exp(z_j)

T   = TopK(p, k)                         # indices of the k largest p_i

w_i = p_i / Σ_{j ∈ T} p_j                # for i ∈ T  (renormalize)
```

Two design choices matter in practice:

- **Full softmax before top-k**, not a sigmoid gate or grouped routing. This matches Gemma’s conversion, not DeepSeek-style √softplus gating.
- **Renormalization** of the surviving weights so Σ_{i ∈ T} w_i = 1. Dropping mass without renorm would silently shrink the routed branch.

In code this is `softmax_topk_renorm` in `gemma4_moe.rs`.

### Expert function

Each expert is a GeLU MLP with its own intermediate width d_ff^(exp) (`expert_feed_forward_length`):

```text
E_i(x) = W_down,i · ( GeLU(W_gate,i · x) ⊙ (W_up,i · x) )
```

Weights are stored quantized: fused gate∥up as **Q4_K**, down as **Q5_1** (most layers) or **Q8_0** (typically the last). A static per-expert scale **s_i** multiplies the down path at combine time.

### Combination

Let **u** be the shared FFN output and **v** the weighted expert sum. Dual post-norms precede the add:

```text
u' = RMSNorm(u; θ₁)
v' = RMSNorm(v; θ₂)

h  ← h + RMSNorm(u' + v'; θ_post)
```

with

```text
v = Σ_{i ∈ T} (w_i · s_i) · E_i( RMSNorm(h; θ_pre2) )
```

The dual norms (`post_ffw_norm_1`, `post_ffw_norm_2`) are easy to miss when porting from Mixtral-style graphs that only normalize once.

---

## Mapping ideas → this codebase

### Where MoE sits in the forward pass

Per layer, decode follows:

1. Attention (+ residual) → `hidden_buf`
2. **MoE MLP or dense MLP** → updates `hidden_buf`
3. Optional PLE (usually inactive on A4B)
4. Layer scalar

The fused “mega-decode” executor is disabled whenever any layer has MoE. Sparse MLP bookkeeping does not currently fit that fused graph; the legacy per-layer command-buffer loop is the source of truth.

### Files

| Concern | Location |
|---------|----------|
| Config flags (`is_moe`, expert dims) | `gemma4_config.rs` |
| Top-k, byte views, LFU cache, CPU checks | `gemma4_moe.rs` |
| Load + `run_moe_mlp_decode` / prefill | `gemma4_gpu_model.rs` |
| Encode APIs | `gpu.rs` |
| `slots8` / `sum8` kernels | `shaders/ggml_mul_mv_q4.metal` |

### Load-time detection

Metadata alone is not sufficient. A layer becomes MoE if `blk.{i}.ffn_gate_inp.weight` exists; dimensions must be positive. Expert tensors are **memory-mapped** (Metal no-copy views over the GGUF), while the router and norms are materialized into ordinary buffers. mmap avoids duplicating tens of gigabytes at load time, but it shifts the runtime problem to *which expert pages stay hot*.

---

## Inference mechanics

### The CPU routing barrier

Architecturally, routing could be entirely on GPU. The current implementation scores on GPU, then:

1. waits for the attention+router command buffer,
2. reads logits to the host,
3. runs `softmax_topk_renorm`,
4. plans expert slots and issues expert work.

This is deliberately simple—easy to log, easy to cross-check—but it inserts a **host sync on every MoE layer of every token**. Any future optimization that keeps top-k on device would remove that barrier without changing the math above.

### Overlapping shared compute with expert I/O

Once top-k indices are known, two things must happen: evaluate the shared FFN, and ensure the selected experts’ weights are resident.

The implementation treats these as concurrent when possible:

- **Misses**: `pread` expert blobs into slot buffers (with `F_RDADVISE` on macOS), in parallel threads.
- **Meanwhile**: encode the shared FFN on the GPU queue.
- **Then**: run hit experts immediately; run miss experts after fills complete.

Optional `MOE_PARALLEL_QUEUE=1` places expert matvecs on a second Metal queue so they can overlap shared MLP even more aggressively.

Latency is therefore closer to

```text
T ≈ T_router+sync
  + max(T_shared, T_miss_I/O)
  + T_expert_matvecs
  + T_combine
```

than to a strict sum of all four—**when** the cache hits often enough that T_miss_I/O is small.

### Why an expert slot cache exists

All **N** experts exist in the mmap’d file, but only **k** are needed per token. Naively letting the GPU fault through cold mmap pages thrashes unified memory—especially on 16 GB machines.

The **LFU slot cache** (`ExpertSlotCache`) keeps a fixed number of expert (gate∥up, down) copies in `StorageModeShared` buffers:

- Hits reuse parked weights.
- Misses evict low-frequency entries (with periodic count decay) and refill from disk.
- Default slot counts snap to 16 / 24 / 32 from free RAM (~110 MB per slot is the planning constant). Comments note that on a busy 16 GB M1 Pro, 32 slots can thrash while 16 behaves better.

This is the same *systems* idea used by several on-device MoE engines (often nicknamed TurboFieldfare / ds4-style): **treat SSD + page cache as the true expert store; RAM holds only a working set.**

### Fused multi-expert kernels

For **k ≤ 8**, an optional path (`MOE_FUSED_SLOTS=1`) issues:

1. one `matvec_ggml_q4_K_gelu_mul_slots8` dispatch (`threadgroup_position.z` = expert),
2. one `sum8` down kernel (Q5_1 or Q8_0)

so experts do not serialize on a single intermediate buffer. The math is unchanged; only the scheduling granularity changes.

---

## Prefill versus decode

Decode (one new token) is the well-exercised MoE path.

Prefill is harder because **each token may select a different expert set**. The default policy therefore falls back to **sequential** MoE MLP evaluation over prompt tokens (attention may still batch). `MOE_PARALLEL_PREFILL=1` enables an experimental path that batches attention then loops MoE per row; it has historically been fragile, which is why it is opt-in.

From a systems perspective: MoE’s win on FLOPs does not automatically transfer to prefill wall-clock time unless expert grouping (token→expert sorting) is implemented. That grouping is still future work here.

---

## Ablations and observability

Because MoE sits beside a full shared MLP, we can peel the system like an onion:

| Knob | What it isolates |
|------|------------------|
| `MOE_DISABLE=1` | Dense shared MLP only; router/experts ignored |
| `MOE_SHARED_ONLY=1` | Shared path + dual-norm combine, expert contribution zeroed |
| `MOE_CROSSCHECK=1` | CPU reference expert FFN vs Metal |
| `MOE_LOG=1` | Top-k indices/weights, health of hidden states |
| `MOE_PROFILE=1` | Phase times: router read, plan, fill, shared wait, experts |
| `MOE_CACHE_STATS=1` | Slot hit/miss rates |

A useful mental order when outputs degrade: disable MoE entirely → shared-only → enable experts with logging/crosscheck → inspect cache hit rate under load.

---

## What this design is *not*

It is easy to project other MoE papers onto this code. A few mismatches:

- **Not Mixtral-only-experts**: A4B always runs a shared FFN.
- **Not DeepSeek routing**: no hash-MoE bootstrap, no √softplus affinity, no correction-bias top-k.
- **Not SwiGLU experts**: GeLU, matching Gemma’s dense MLPs.
- **Not MLA / CSA / HCA**: attention remains Gemma’s GQA + sliding / full pattern with shared-KV layers.

What *is* portable to other MoEs is the systems layer: mmap’d experts, LFU working sets, I/O overlapped with shared compute, and optional fused multi-expert dispatches.

---

## Summary

1. **Problem.** Dense scaling ties capacity to per-token FLOPs; MoE loosens that tie by activating **k** of **N** FFN experts.
2. **Gemma A4B form.** Shared dense FFN in parallel with softmax top-k GeLU experts; dual post-norms; residual combine.
3. **Scope.** Only the MLP block is sparse; attention is dense Gemma 4.
4. **Runtime.** GPU router logits → CPU top-k → LFU-resident experts with miss I/O overlapped against shared MLP → combine.
5. **Memory.** Experts live in a large mmap; a small slot cache is what makes decode viable on 16 GB-class unified memory.
6. **Limits today.** Host routing sync every layer; cautious sequential MoE prefill; fused mega-decode disabled for MoE models.

The center of gravity in code is `run_moe_mlp_decode` in `gemma4_gpu_model.rs`, with routing/cache primitives in `gemma4_moe.rs`. Reading those two—with the equations above in mind—is usually enough to navigate the rest.

---

## Appendix: tensor names and formats

Per layer `blk.{i}.*` in GGUF:

| Tensor | Role | Typical format |
|--------|------|----------------|
| `ffn_gate_inp.{weight,scale}` | Router | F32→f16 weights; scale prefolded |
| `ffn_gate_up_exps.weight` | All experts’ gate∥up | Q4_K, mmap |
| `ffn_down_exps.{weight,scale}` | All experts’ down + scales | Q5_1 or Q8_0, mmap |
| `pre_ffw_norm_2` | Expert pre-norm | F32 |
| `post_ffw_norm_1` / `post_ffw_norm_2` | Shared / routed post-norms | F32 |
| `ffn_gate` / `ffn_up` / `ffn_down`, `ffn_norm`, `post_ffw_norm` | Shared FFN | Often Q8_0 on A4B |

---

## References (conceptual)

- Shazeer et al., 2017. *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.*
- Fedus et al., 2021. *Switch Transformers.*
- Jiang et al., 2024. *Mixtral of Experts.*
- Gemma 4 model cards / GGUF metadata (`gemma4.expert_*`) as realized in this engine.

*(The papers above provide vocabulary; the equations and runtime behavior in this note follow the repository’s Gemma 4 A4B path.)*
