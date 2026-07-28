# 08 — Norms, MLP, and PLE: The Other 60% of a Layer

Attention gets the attention. But on decode, the MLP moves **more weight
bytes than attention does**, and Gemma4's PLE block is a whole extra
sub-network that most transformer tutorials never mention because no other
popular model has it.

This chapter is a tutorial. You will:

1. Derive RMSNorm and read the exact Metal kernel that computes it
2. Understand `rmsnorm_acc` — the "residual epilogue" used three times per layer
3. Walk the **four-branch** MLP decision in `encode_fused_mlp_layer` and know
   why each branch exists
4. Build the PLE block from scratch: the pre-pass (5 dispatches) and the
   per-layer block (2–3 dispatches)

Open these:

```text
src/shaders/llama.metal        # rmsnorm ~1582, rmsnorm_acc ~1711, gelu_mul ~2696, ple_matvec_gelu_q4 ~1090
src/decode_fused.rs            # encode_fused_mlp_layer ~531, encode_fused_ple_layer ~724
src/gemma4_gpu_model.rs        # PLE pre-pass ~3442-3489
```

---

## Part A — RMSNorm from first principles

### A.1 The formula and why it is not LayerNorm

LayerNorm centers and scales:

```text
y = (x − mean(x)) / sqrt(var(x) + ε) * γ + β
```

RMSNorm drops centering and the bias:

```text
rms(x) = sqrt( (1/N) Σ x_i²  + ε )
y_i    = (x_i / rms(x)) * w_i
```

Why it works: in a residual stream the mean is already near zero and the
thing that actually destabilizes training is **scale drift**. RMSNorm fixes
scale with one pass and one weight vector. Cheaper: no mean, no variance,
no bias.

Note where ε sits: **inside** the sqrt, added to the mean of squares. Some
implementations add it outside. If you port a kernel with ε in the wrong
place you get a tiny, maddening numeric difference that only shows up as
slightly different tokens after a few hundred steps.

### A.2 Now read the kernel

```1582:1620:src/shaders/llama.metal
kernel void rmsnorm(
    device const float* x [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& dim [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    threadgroup float shared_sum[256];

    float partial_sum = 0.0f;
    for (uint i = tid; i < dim; i += tg_size) {
        float val = x[i];
        partial_sum += val * val;
    }
    shared_sum[tid] = partial_sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv_rms = rsqrt(shared_sum[0] / float(dim) + eps);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < dim; i += tg_size) {
        out[i] = x[i] * inv_rms * weight[i];
    }
}
```

Three phases, and every GPU norm you will ever read has these same three:

**Phase 1 — strided partial sums.** Thread `tid` handles indices
`tid, tid+tg_size, tid+2*tg_size, …`. For `dim=2560` and `tg_size=256`,
each thread touches exactly 10 elements. The stride (not a contiguous
block per thread) is deliberate: at any instant the 256 threads are reading
256 *adjacent* floats, which coalesces into wide memory transactions.

**Phase 2 — tree reduction.** Halve the active thread count each step:

```text
stride=128:  shared_sum[0..128) += shared_sum[128..256)
stride=64:   shared_sum[0..64)  += shared_sum[64..128)
...
stride=1:    shared_sum[0] += shared_sum[1]
```

`log2(256) = 8` steps, each followed by a barrier. Why the barrier *inside*
the loop? Because thread 0 reading `shared_sum[1]` at stride=1 must be sure
thread 1 finished its stride=2 write. Different simdgroups are not lockstep
with each other — 256 threads span 8 simdgroups.

**Phase 3 — normalize and write.** Same strided pattern, now applying
`inv_rms` and the per-channel weight.

`rsqrt` is one instruction — cheaper than `1.0/sqrt(x)`.

### A.3 Per-head RMSNorm (QK-norm)

Same three phases, but `N = head_dim` and each head reduces independently.
For E4B's 20 heads × 128 dims you get 20 separate sums, not one over 2560.

This is why `rmsnorm_per_head` exists as its own kernel, and why the fused
attention kernel does its own inline reduction over `HEAD_DIM` (Ch 07 Part D.3).
Using the wrong N is a classic bug: the output *looks* normalized, magnitudes
are just subtly wrong, and quality degrades without a crash.

`rmsnorm_per_head_noweight` is the V-norm variant: no weight multiply.

---

## Part B — `rmsnorm_acc`: the residual epilogue

Gemma4 wraps each sub-block output in a norm before adding to the residual:

```text
h ← h + post_norm(sub_output)
```

Naively that is three kernels: rmsnorm, then vec_add. This engine fuses it:

```1711:1745:src/shaders/llama.metal
kernel void rmsnorm_acc(
    device float* acc [[buffer(0)]],       // hidden — read AND written
    device const float* x [[buffer(1)]],   // sub-block output
    device const float* weight [[buffer(2)]],
    ...
) {
    // ... identical phases 1 and 2 over x ...
    float inv_rms = rsqrt(shared_sum[0] / float(dim) + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < dim; i += tg_size) {
        acc[i] += x[i] * inv_rms * weight[i];       // += , not =
    }
}
```

The only difference from `rmsnorm` is `acc[i] +=` instead of `out[i] =`.
That single character saves a full read+write pass over `hidden` per
sub-block, three times per layer, 42 layers, every token.

**Do the arithmetic.** hidden = 2560 floats = 10 KB. Three sub-blocks ×
42 layers = 126 extra round trips of 10 KB read + 10 KB write ≈ 2.5 MB of
avoidable traffic per token. At ~200 GB/s that is ~12 µs/token — small but
free.

You will see `encode_rmsnorm_acc_view` at the end of all three sub-blocks in
`decode_fused.rs`. When you see it, read it as: **"normalize this and fold it
into the residual."**

---

## Part C — The MLP block

### C.1 The math

```text
n    = pre_feedforward_layernorm(h)
gate = W_gate · n        # [intermediate]
up   = W_up   · n        # [intermediate]
mid  = GeLU(gate) ⊙ up   # elementwise
down = W_down · mid      # [hidden]
h   ← h + post_feedforward_layernorm(down)
```

E4B: `2560 → 10240 → 2560`. Three big matrices per layer.

### C.2 Why gating at all

A plain FFN is `W_2 · act(W_1 x)`. The gated variant computes **two**
projections and uses one to modulate the other. Empirically better per
parameter; costs one extra matrix of bandwidth. Gemma uses GeLU as the
activation (not SiLU/Swish as in Llama's SwiGLU).

### C.3 GeLU, exactly

```metal
kernel void gelu_mul(
    device const float* gate, device const float* up, device float* out,
    constant uint& n, uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    out[gid] = gelu_pytorch_tanh(gate[gid]) * up[gid];
}
```

`gelu_pytorch_tanh` is the tanh approximation:

```text
GeLU(x) ≈ 0.5 · x · (1 + tanh( sqrt(2/π) · (x + 0.044715 x³) ))
```

This matches PyTorch `gelu(approximate='tanh')`, which is what Gemma was
trained/exported with. The exact erf version differs by ~1e-3 — enough to
diverge token choices on borderline logits. **Match the training stack.**

Note this kernel is a trivial `dispatch_threads` map: one thread per output
element, no threadgroup memory, no barriers. Compare with `rmsnorm` (needs a
reduction) — elementwise ops are the easy case.

### C.4 The four-branch decision in `encode_fused_mlp_layer`

Now read the real code. There are four paths, and each exists for a
measured reason. The `n` counter tallies dispatches for `PROFILE_DISPATCHES`.

**Branch 1 — K-quant, gate/up both Q4_K, fusion enabled (the common Q4_K_M case):**

```549:563:src/decode_fused.rs
self.ctx.encode_rmsnorm_qk_gelu_mul_kquant_at_view(
    encoder,
    &layer.gate_proj, &layer.up_proj,
    scratch.hidden, 0,
    &layer.pre_feedforward_layernorm_weight,
    scratch.inv_rms,
    scratch.gelu, 0,
    intermediate_size, hidden_size, eps,
);
n += 2;
```

**Two dispatches** for norm + gate + up + GeLU. The first computes
`inv_rms` of `hidden` into a tiny buffer; the second is a Q4_K matvec that
reads `hidden`, scales on the fly by `w_norm[i] * inv_rms`, dequantizes
gate-row *i* and up-row *i*, and writes `GeLU(gate·x) * (up·x)` straight
into `scratch.gelu`.

What was eliminated: a `normed` buffer (2560 floats written then read), a
`gate` buffer and an `up` buffer (10240 floats each), and a separate
`gelu_mul` dispatch. Note what was **not** eliminated: reading both weight
matrices. That is why fusion here is a modest win, not a 2× win (Ch 05 Part F).

**Branch 2 — K-quant gate/up but fusion disabled** (`FUSED_RMSNORM_MLP_KQUANT=0`):
explicit `rmsnorm` into `scratch.normed`, then `encode_matvec_qk_gelu_mul_at_view`.
Still fuses gate+up+GeLU, just not the norm. This branch exists so you can
A/B the norm fusion without rebuilding.

**Branch 3 — K-quant but gate/up are *not* both Q4_K** (e.g. one is Q6_K):

(`src/decode_fused.rs` ~587–621, as a dispatch sketch:)

```text
encode_rmsnorm_view(...);                       // n += 1
encode_matvec_quant_layer(gate_proj → scratch.gate);
encode_matvec_quant_layer(up_proj   → scratch.up);
encode_gelu_mul(gate, up → gelu);               // n += 3
```

The fully decomposed fallback. The fused kernel is Q4_K-specific; mixed
formats fall back to generic per-tensor matvecs. **This is why per-tensor
format tags matter** (Ch 05 Part E) — one Q6_K gate tensor changes the
dispatch count from 2 to 4 for that layer.

**Branch 4 — plain Q4_0 with a packed gate‖up buffer:**

```642:659:src/decode_fused.rs
self.ctx.encode_mlp_fused_q4_gelu_down_packed_from_hidden_at_view(
    encoder,
    &layer.gate_up_proj,     // interleaved gate‖up
    &layer.down_proj,
    &layer.pre_feedforward_layernorm_weight,
    scratch.hidden, 0, scratch.inv_rms, 0,
    scratch.up, 0, scratch.down, 0,
    hidden_size, intermediate_size, eps,
);
n += 3;
```

This one goes furthest: norm + gate‖up + GeLU **and** the down projection,
three dispatches for the entire MLP. It requires the interleaved
`gate_up_proj` layout built at load time (`PACKED_MLP_GATE_UP`, default on)
so one kernel can walk gate-row *i* and up-row *i* adjacently.

Then, for every branch except 4 (which already did down):

```625:634:src/decode_fused.rs
self.encode_matvec_quant_layer(encoder, &layer.down_proj,
    scratch.gelu, scratch.down, hidden_size, intermediate_size, layer.weight_format);
n += 1;
```

On Q4_K_M, `down_proj` is typically **Q6_K** (higher precision where the
model is most sensitive) — one reason branch 3 shows up.

And every branch ends identically:

```711:719:src/decode_fused.rs
self.ctx.encode_rmsnorm_acc_view(
    encoder, scratch.hidden, scratch.down,
    &layer.post_feedforward_layernorm_weight, hidden_size, eps);
n += 1;
```

### C.5 Dispatch budget table

| Branch | Condition | Dispatches (incl. down + acc) |
|--------|-----------|-------------------------------|
| 1 | Q4_K gate+up, fusion on | 2 + 1 + 1 = **4** |
| 2 | Q4_K gate+up, norm fusion off | 2 + 1 + 1 = **4** |
| 3 | mixed K-quant formats | 4 + 1 + 1 = **6** |
| 4 | Q4_0 packed gate‖up | 3 + 1 = **4** |

Run with `PROFILE_DISPATCHES=1` and check which branch your model takes.
If you expected 4 and see 6, your gate/up tensors are not both Q4_K.

---

## Part D — PLE: Gemma4's per-layer embedding block

This is the part with no equivalent in Llama, so there is no folklore to
lean on. Build it carefully.

### D.1 What problem it solves

Normally a token's identity enters the network **once**, at the embedding
layer, and everything after that is a transformed residual. PLE gives every
layer its own token-conditioned signal: a per-layer embedding table lookup,
mixed with a projection of the current hidden state, injected as an extra
residual contribution.

Think of it as: *"each layer gets a small, token-specific bias vector,
modulated by what the residual stream currently looks like."*

Dimensions: `ple_dim = hidden_size_per_layer_input = 256` (vs hidden 2560).
Small per layer, but 42 layers of it.

### D.2 The two inputs

**Input 1 — token identity** (CPU-side gather, before any GPU work):

```text
decode_ple_into(token_id):
    ple_token_id_buf[layer*ple_dim + i] = PLE_E[token][layer*ple_dim + i] * sqrt(ple_dim)
```

One flat buffer of `n_layers * ple_dim` floats. Note the `√ple_dim` scale,
mirroring the `√hidden_size` scale on the main embedding.

**Input 2 — context projection** (GPU pre-pass, depends on the current hidden
state). Here is the real code, and it is exactly four dispatches:

```3446:3489:src/gemma4_gpu_model.rs
if !__ablate.skip_ple() {
    // Step 2a: context_proj = per_layer_model_projection @ embed
    self.ctx.encode_matvec_auto_view(
        encoder,
        &self.per_layer_model_projection_weight,
        &self.hidden_buf,
        &self.ple_context_proj_buf,
        ple_total_dim as u32,
        hidden_size as u32,
    );
    // Step 2b: context_proj *= 1/sqrt(hidden_size)
    self.ctx.encode_vec_scale(
        encoder, &self.ple_context_proj_buf, &self.ple_combined_buf,
        ple_total_dim as u32, context_proj_scale);
    // Step 2c: RMSNorm per layer
    self.ctx.encode_rmsnorm_per_head_view(
        encoder, &self.ple_combined_buf,
        &self.per_layer_projection_norm_weight,
        &self.ple_context_proj_buf,
        num_layers as u32, ple_dim as u32, eps);
    // Step 3: combined = (context_proj + token_identity) * 1/sqrt(2)
    self.ctx.encode_vec_add(
        encoder, &self.ple_context_proj_buf, &self.ple_token_id_buf,
        &self.ple_combined_buf, ple_total_dim as u32);
    self.ctx.encode_vec_scale(
        encoder, &self.ple_combined_buf, &self.ple_context_proj_buf,
        ple_total_dim as u32, ple_input_scale);
}
```

Read it as a pipeline over one big buffer of `n_layers × ple_dim` values:

| Step | Operation | Shape |
|------|-----------|-------|
| 2a | `W_model_proj @ hidden` | `[2560] → [42*256] = [10752]` |
| 2b | scale by `1/√hidden_size` | elementwise |
| 2c | RMSNorm **per layer** (`rmsnorm_per_head` with N=ple_dim, 42 "heads") | per-256 chunk |
| 3 | `+ token_identity`, then scale by `1/√2` | elementwise |

Two details worth pausing on:

- **`rmsnorm_per_head` is reused as "rmsnorm per layer."** The kernel does
  not care whether the 42 chunks are attention heads or layer slices; it
  reduces over `ple_dim` independently for each. Nice example of a kernel
  being a shape contract, not a semantic one.
- **The `1/√2`** is the standard "average two unit-scale signals without
  doubling variance" factor. Two contributions (context + identity) each
  roughly unit-scale sum to ~√2 scale; dividing restores it.
- The buffers **ping-pong**: `ple_context_proj_buf → ple_combined_buf →
  ple_context_proj_buf → …`. The final result lands in
  `ple_context_proj_buf`, which is what layers read. Follow the arrows
  carefully — an off-by-one in the ping-pong silently feeds an
  un-normalized vector to every layer.

The comment in the code notes this replaced 42 per-layer copy-out
dispatches: because the result is contiguous, layer `L` just reads at byte
offset `L * ple_dim * 4`.

### D.3 The per-layer PLE block

```724:797:src/decode_fused.rs
fn encode_fused_ple_layer(...) -> u32 {
    let ple_off = (layer_idx as u32 * ple_dim * 4) as u64;   // slice of the pre-pass output

    if gpu::fused_mlp_ple_enabled()
        && gpu::weight_buf_is_q4(&layer.per_layer_input_gate_weight, ple_dim, hidden_size)
    {
        self.ctx.encode_ple_matvec_gelu_q4_at_view(
            encoder,
            &layer.per_layer_input_gate_weight,
            scratch.hidden, 0,
            scratch.ple_ctx, ple_off,          // context slice for THIS layer
            scratch.ple_normed, 0,
            ple_dim, hidden_size);
        n += 1;
    } else {
        // K-quant gate → ggml Q4_K/Q6_K matvec, then separate gelu_mul
        self.encode_matvec_auto_layer(encoder, &layer.per_layer_input_gate_weight,
            scratch.hidden, scratch.gate, ple_dim, hidden_size);
        self.ctx.encode_gelu_mul_at(encoder, scratch.gate, 0,
            scratch.ple_ctx, ple_off, scratch.ple_normed, 0, ple_dim);
        n += 2;
    }

    self.encode_matvec_auto_layer(encoder, &layer.per_layer_projection_weight,
        scratch.ple_normed, scratch.ple_projected, hidden_size, ple_dim);
    n += 1;
    self.ctx.encode_rmsnorm_acc_view(encoder, scratch.hidden, scratch.ple_projected,
        &layer.post_per_layer_input_norm_weight, hidden_size, eps);
    n += 1;
    n
}
```

So the math per layer is:

```text
gate      = W_inp_gate · h                      # [2560] → [256]
gated     = GeLU(gate) ⊙ context[layer]         # [256], context from pre-pass
projected = W_ple_proj · gated                  # [256] → [2560]
h        ← h + post_ple_norm(projected)
```

The fast path folds matvec + GeLU + multiply-by-context into **one** kernel:

```1090:1102:src/shaders/llama.metal
kernel void ple_matvec_gelu_q4(
    device const uchar* W [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device const float* context [[buffer(2)]],
    device float* y [[buffer(3)]],
    constant uint& M [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    ...
) {
    matvec_q4_gelu_mul_body<4>(W, x, context, y, M, K, tgid, sgid, lane, Q4F_SG);
}
```

It is a normal Q4 matvec (4 rows per simdgroup) whose epilogue is
`y[row] = GeLU(dot) * context[row]` instead of `y[row] = dot`. The `context`
pointer is already offset to this layer's slice by the host (`ple_off`).

Note the structural asymmetry: PLE **projects up** from 256 → 2560 at the
end, so `per_layer_projection_weight` is `[2560, 256]` — a wide-output,
narrow-input matvec. Different shape regime from the MLP's `[10240, 2560]`.

### D.4 The performance trap that mattered (AGENTS E22)

On Q4_K_M GGUFs, `per_layer_model_projection` (the pre-pass weight) and some
PLE tensors are stored as **F32 or F16**, not K-quant. An earlier version of
the loader requantized them to Q4_0, which pushed the pre-pass onto a slow
`projection_q4_batch` path.

Result at 4k prefill: PLE bucket **~1555 ms**. Keeping them dense f16 and
using `mul_mm_f16`: **~230 ms**. That single dtype decision was worth more
than several kernel-tuning experiments.

Lesson: *"small" ops with the wrong dtype path can dominate.* Ablation
(`PROFILE_ABLATE`) is how it was found — the PLE bucket did not look
suspicious until it was measured.

---

## Part E — Layer scalar

After attention + MLP + PLE:

```text
hidden *= layer.layer_scalar
```

A per-layer constant from the checkpoint (depth stabilization). One tiny
`vec_scale` dispatch. It is trivially easy to omit in a from-scratch port
and produces slowly-compounding wrongness — outputs stay grammatical while
drifting from the reference model. Grep `layer_scalar` and confirm it is
applied exactly once per layer.

---

## Part F — Putting one layer's dispatch budget together

For a KV-owning Q4_K_M layer on the fused path, roughly:

```text
Attention:  2 (norm+QKV)  + 1 (full fused attn) + 1 (O proj) + 1 (acc)  ≈ 5
MLP:        2 (norm+gate‖up+gelu) + 1 (down) + 1 (acc)                  ≈ 4
PLE:        1–2 (gate+gelu) + 1 (proj) + 1 (acc)                        ≈ 3–4
Scalar:     1
                                                              ≈ 13–14 per layer
```

× 42 layers, minus the cheaper shared-KV layers (no K/V projection, no
append), lands around **450–600 dispatches per token**, plus rope fill, the PLE
pre-pass (5), final norm, and lm_head. `AGENTS.md` #1 measured **455** on the
fused path — compare your own with `PROFILE_DISPATCHES=1`.

At ~5 µs of CPU-side encode overhead per dispatch you get ~2.4–3 ms/token of
pure overhead — real, measurable, and (per AGENTS #1) still not the thing
that makes long-context decode slower than llama.cpp.

---

## Part G — Exercises

1. Rewrite `rmsnorm` from memory. Then diff against the file. Did you put ε
   inside the sqrt? Did you barrier inside the reduction loop?

2. For `dim=10240` and `tg_size=256`: how many elements does each thread
   handle in phase 1? How many reduction steps in phase 2?

3. Trace which MLP branch your model takes: print `layer.weight_format`,
   `gate_proj.format`, `up_proj.format`, `down_proj.format` for layer 0.
   Predict the dispatch count, then verify with `PROFILE_DISPATCHES=1`.

4. In the PLE pre-pass, list the buffer each step reads and writes. Confirm
   the final result is in `ple_context_proj_buf`. What breaks if step 2c
   writes to `ple_combined_buf` instead?

5. Why can `rmsnorm_per_head` serve both QK-norm (heads) and PLE (layers)?
   State the kernel's contract in one sentence with no reference to
   attention.

6. `gelu_mul` uses `dispatch_threads`, `rmsnorm` uses a threadgroup with
   barriers. Explain the difference in one sentence.

7. Estimate the bandwidth saved by `rmsnorm_acc` vs `rmsnorm` + `vec_add`
   over one full token (42 layers × 3 sub-blocks, hidden=2560).

---

## Checklist

- [ ] I can write RMSNorm's three GPU phases and justify every barrier.
- [ ] I know what the `+=` in `rmsnorm_acc` saves.
- [ ] I can name all four MLP branches and the condition that selects each.
- [ ] I know why `down_proj` being Q6_K changes the dispatch count.
- [ ] I can draw the 5-dispatch PLE pre-pass with buffer names and scales.
- [ ] I can explain `1/√2` and `1/√hidden_size` in the PLE path.
- [ ] I know why PLE-as-Q4_0 cost ~1.3 s at 4k prefill.

**Next:** [09_decode_path.md](09_decode_path.md) assembles attention (Ch 07)
and this chapter into one token; [10_prefill_path.md](10_prefill_path.md)
does the batched version.
