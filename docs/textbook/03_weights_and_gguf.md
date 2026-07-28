# 03 — Weights: From GGUF Bytes to GPU Buffers

Every inference bug that is not a kernel bug is a **loading** bug: a
transposed tensor, a wrong dtype, a tensor requantized onto a slow path, a
name that silently did not match so a weight stayed zero. This chapter makes
you fluent in the file format and the load path so you can debug those in
minutes instead of days.

Open:

```text
src/gguf.rs                    # format reader + CPU dequant
src/gemma4_gpu_model.rs        # from_gguf ~1650, config parse ~9052
src/gpu.rs                     # buffer_from_* upload helpers
tools/gguf_inspect.py          # inspect any file from the shell
```

---

## Part A — The GGUF file format, byte by byte

GGUF is intentionally boring: a header, a key/value metadata table, a tensor
table, then one big aligned blob of tensor data.

```text
┌─────────────────────────────────────────┐
│ magic: u32  = 0x46554747 ("GGUF")       │
│ version: u32 = 2 or 3                   │
│ tensor_count: u64                       │
│ kv_count: u64                           │
├─────────────────────────────────────────┤
│ metadata: kv_count entries              │
│   key: gguf string (u64 len + bytes)    │
│   value_type: u32                       │
│   value: depends on type                │
├─────────────────────────────────────────┤
│ tensor table: tensor_count entries      │
│   name: gguf string                     │
│   n_dims: u32                           │
│   dims: [u64; n_dims]                   │
│   ggml_type: u32                        │
│   offset: u64   (relative to data blob) │
├─────────────────────────────────────────┤
│ padding to `general.alignment` (def 32) │
├─────────────────────────────────────────┤
│ tensor data blob                        │
└─────────────────────────────────────────┘
```

Here is the reader, essentially complete:

```214:257:src/gguf.rs
pub fn open<P: AsRef<Path>>(path: P) -> Self {
    let file = File::open(&path).expect("failed to open gguf file");
    let mmap = unsafe { Mmap::map(&file) }.expect("failed to mmap gguf file");
    let (version, metadata, tensors, data_offset) = {
        let mut c = Cursor { buf: &mmap, pos: 0 };
        let magic = c.u32();
        assert_eq!(magic, GGUF_MAGIC, "not a GGUF file (bad magic)");
        let version = c.u32();
        assert!(version == 2 || version == 3, "unsupported GGUF version {}", version);
        let tensor_count = c.u64() as usize;
        let kv_count = c.u64() as usize;

        let mut metadata = HashMap::with_capacity(kv_count);
        for _ in 0..kv_count {
            let key = c.gstr();
            let vt = c.u32();
            let val = c.value(vt);
            metadata.insert(key, val);
        }

        let mut tensors = HashMap::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = c.gstr();
            let n_dims = c.u32() as usize;
            let dims: Vec<u64> = (0..n_dims).map(|_| c.u64()).collect();
            let ggml_type = c.u32();
            let offset = c.u64();
            tensors.insert(name.clone(), TensorInfo { name, ggml_type, dims, offset });
        }

        let alignment = match metadata.get("general.alignment") { ... _ => DEFAULT_ALIGNMENT };
        let data_offset = align_up(c.pos, alignment);
        (version, metadata, tensors, data_offset)
    };
    Gguf { mmap, version, metadata, tensors, data_offset }
}
```

Three things to internalize:

**1. `mmap`, not `read`.** The file is mapped, not copied. A 4 GB model does
not become 4 GB of resident heap; pages fault in as touched. `unsafe` because
the OS can change the mapping under you if the file is modified — do not edit
a GGUF while a server has it open.

**2. Offsets are relative.** `info.offset` is relative to the data blob, so
absolute position is `data_offset + info.offset`:

```267:270:src/gguf.rs
fn tensor_bytes(&self, info: &TensorInfo) -> &[u8] {
    let start = self.data_offset + info.offset as usize;
    &self.mmap[start..start + info.byte_len()]
}
```

**3. Parsing is a straight-line walk.** No seeking, no index — you must read
the metadata table to know where the tensor table starts, and the tensor
table to know where data starts. This is why the whole header is parsed
eagerly into `HashMap`s in `open`.

### A.1 Dimension order will confuse you once

```99:110:src/gguf.rs
/// ne0 (innermost / contiguous dim). For a [out,in] weight this is `in`.
pub fn ne0(&self) -> usize { self.dims.first().copied().unwrap_or(1) as usize }
/// Number of rows = product of all dims except ne0. For a [out,in] weight: `out`.
pub fn n_rows(&self) -> usize {
    if self.dims.len() <= 1 { 1 } else { self.dims[1..].iter().product::<u64>() as usize }
}
```

GGUF stores `dims[0]` as the **innermost/contiguous** dimension, the opposite
of how you write matrix shapes in a paper. For `gate_proj` with
`[out=10240, in=2560]`:

```text
dims        = [2560, 10240]     # ne0 = 2560 = in
ne0()       = 2560              # contiguous run: one output row's inputs
n_rows()    = 10240             # number of output rows
```

Which is exactly what a matvec wants: row `i` of the weight is contiguous, so
`dot(W[i], x)` is a linear scan. If you ever compute a stride with the dims
swapped, the symptom is not a crash — it is output that looks like structured
noise. Print `dims`, `ne0()`, `n_rows()` for one tensor and match them
against the architecture before trusting anything.

### A.2 Byte length from block geometry

```48:61:src/gguf.rs
fn block_spec(t: u32) -> (usize, usize) {
    match t {
        ggml_type::F32  => (1, 4),
        ggml_type::F16  => (1, 2),
        ggml_type::BF16 => (1, 2),
        ggml_type::Q4_0 => (32, 18),
        ggml_type::Q4_1 => (32, 20),
        ggml_type::Q8_0 => (32, 34),
        ggml_type::Q4_K => (QK_K, 144),   // QK_K = 256
        ggml_type::Q5_K => (QK_K, 176),
        ggml_type::Q6_K => (QK_K, 210),
        _ => panic!("unsupported ggml type id {}", t),
    }
}
```

```111:116:src/gguf.rs
pub fn byte_len(&self) -> usize {
    let (epb, bpb) = block_spec(self.ggml_type);
    let n = self.num_elements();
    assert!(n % epb == 0, "tensor {} elems {} not divisible by block {}", self.name, n, epb);
    (n / epb) * bpb
}
```

That `assert!` is doing real work. A Q4_K tensor whose element count is not a
multiple of 256 is either misparsed or genuinely unsupported — either way you
want to know at load, not at first inference.

The table itself is worth memorizing, and worth *deriving* rather than
memorizing (see Ch 05):

| Type | elems/block | bytes/block | bits/weight | What's in the block |
|------|-------------|-------------|-------------|---------------------|
| Q4_0 | 32 | 18 | 4.5 | f16 scale + 16 packed bytes |
| Q4_1 | 32 | 20 | 5.0 | f16 scale + f16 min + 16 bytes |
| Q8_0 | 32 | 34 | 8.5 | f16 scale + 32 int8 |
| Q4_K | 256 | 144 | 4.5 | 2 f16 + 12 B of 6-bit sub-scales + 128 B nibbles |
| Q5_K | 256 | 176 | 5.5 | Q4_K + 32 B of 5th bits |
| Q6_K | 256 | 210 | 6.56 | 128 B low + 64 B high + 16 int8 scales + f16 |

**Trap:** Q4_0 and Q4_K are both 4.5 bits/weight, so a Q4_0 and a Q4_K tensor
of the same shape have the **same byte length**. Byte length can never tell
you which one you have; only `ggml_type` can. Ch 05 Part E covers why this
one bites so hard.

### A.3 Reading a single row without dequantizing anything

```280:284:src/gguf.rs
pub fn tensor_row_bytes(&self, name: &str, row: usize, row_stride: usize) -> &[u8] {
    let info = self.tensor(name).unwrap_or_else(|| panic!("tensor not found: {}", name));
    let start = self.data_offset + info.offset as usize + row * row_stride;
    &self.mmap[start..start + row_stride]
}
```

This is how embedding lookups work: `token_embd` is a `[262144, 2560]` table
— hundreds of megabytes — and a decode step needs exactly one row. `mmap` +
row arithmetic gives you that row with one page fault and no allocation.
Dequantizing the whole table to keep it "ready" would be strictly worse.

---

## Part B — Metadata: the config lives in the file

Config is not hardcoded; it is read from metadata keys, with defaults:

```9052:9087:src/gemma4_gpu_model.rs
g.get_f32("gemma4.attention.layer_norm_rms_epsilon").unwrap_or(1e-6) as f64;
g.get_u32("gemma4.context_length").unwrap_or(131072) as usize;
g.get_f32("gemma4.final_logit_softcapping").unwrap_or(30.0);
g.get_arr_bool("gemma4.attention.sliding_window_pattern")
let full_theta    = g.get_f32("gemma4.rope.freq_base").unwrap_or(1_000_000.0) as f64;
let sliding_theta = g.get_f32("gemma4.rope.freq_base_swa").unwrap_or(10_000.0) as f64;
g.get_u32("gemma4.rope.dimension_count_swa").unwrap_or(head_dim as u32) as f64;
```

Read that block as an architecture summary of Gemma4:

- **Two RoPE bases** — `1e6` for full-attention layers, `1e4` for sliding.
  Long-range layers need slower frequencies (Ch 02, Ch 00b).
- **`sliding_window_pattern`** — a bool array, one entry per layer, that
  *is* the SWA/full layout. Not inferred, not a formula: read from the file.
- **`dimension_count_swa`** — partial rotary, potentially different per
  attention type.
- **`final_logit_softcapping = 30.0`** — `logits = 30 * tanh(logits/30)`.

The architecture gate:

```1650:1650:src/gemma4_gpu_model.rs
let arch = g.get_str("general.architecture").unwrap_or("");
```

The MTP draft model uses a parallel namespace (`src/speculative.rs`):

```text
gemma4-assistant.nextn_predict_layers      (default 4)
gemma4-assistant.embedding_length_out      (default 1536)
gemma4-assistant.embedding_length          (default 256)
gemma4-assistant.vocab_size                (default 262144)
gemma4-assistant.rope.freq_base / _swa
gemma4-assistant.attention.sliding_window
gemma4-assistant.final_logit_softcapping
gemma4-assistant.attention.sliding_window_pattern
```

Same reader, different prefix. If you add a draft-model field, add it here,
not to a struct literal.

The tokenizer also lives in metadata (`gguf.rs` ~463+):

```text
tokenizer.ggml.model            # "llama" (SPM) / "gpt2" (BPE) / ...
tokenizer.ggml.tokens           # array of strings, index = token id
tokenizer.ggml.merges           # BPE merge rules
tokenizer.ggml.token_type       # normal / control / user-defined / ...
tokenizer.ggml.bos_token_id / eos / unknown_token_id
```

`token_type` is how the server knows ids like 1, 105, 106 are control tokens
— the same ids that appear in the scheduler's first-token blocklist (Ch 12
Part D.1). Nothing in this stack invents a token id; they all come from here.

---

## Part C — Name → field: where weights actually land

Loading is a lot of string lookups. When one silently misses, that weight
stays whatever the buffer was initialized to. Know the map:

| GGUF-ish name | Engine field | Notes |
|---|---|---|
| `token_embd.weight` | `token_embedding` | Often BF16/F16; row-gathered per token, scaled by `√hidden_size` |
| `blk.N.attn_norm.weight` | `layer.input_layernorm_weight` | Pre-attention RMSNorm |
| `blk.N.attn_q.weight` | `layer.q_proj` | `[q_out, hidden]` |
| `blk.N.attn_k.weight` | `layer.k_proj` | **Absent at runtime for shared-KV layers** |
| `blk.N.attn_v.weight` | `layer.v_proj` | Same |
| `blk.N.attn_q_norm.weight` | `layer.q_norm_weight` | Per-head, N=head_dim |
| `blk.N.attn_k_norm.weight` | `layer.k_norm_weight` | Per-head |
| `blk.N.attn_output.weight` | `layer.o_proj` | `[hidden, q_out]` |
| `blk.N.post_attention_norm.weight` | `layer.post_attention_layernorm_weight` | Feeds `rmsnorm_acc` |
| `blk.N.ffn_gate.weight` | `layer.gate_proj` | `[intermediate, hidden]` |
| `blk.N.ffn_up.weight` | `layer.up_proj` | Same shape |
| `blk.N.ffn_down.weight` | `layer.down_proj` | `[hidden, intermediate]`, often Q6_K |
| `blk.N.ffn_norm.weight` | `layer.pre_feedforward_layernorm_weight` | |
| `blk.N.post_ffw_norm.weight` | `layer.post_feedforward_layernorm_weight` | |
| PLE gate | `layer.per_layer_input_gate_weight` | `[ple_dim, hidden]` |
| PLE projection | `layer.per_layer_projection_weight` | `[hidden, ple_dim]` |
| PLE model projection | `per_layer_model_projection_weight` | `[n_layers*ple_dim, hidden]`, keep dense f16 |
| `output_norm.weight` | `final_norm_weight` | |
| `output.weight` / tied embd | `lm_head` | `[vocab, hidden]` ≈ 440 MB |

Notes that save hours:

- **Shared-KV layers have no K/V projections at all** — not zeroed, absent.
  `layer.has_kv == false` is the flag every code path checks (Ch 02, Ch 06,
  Ch 10 Part C.2).
- **Derived layouts** exist alongside the originals: `gate_up_proj` is gate
  and up interleaved for the packed MLP kernel (`PACKED_MLP_GATE_UP`),
  `qkv_stacked` likewise for prefill. These are *built at load time* from the
  same bytes. When you add a fused kernel, you usually add a layout, and then
  you own keeping the two in sync.
- **`lm_head` may be tied** to `token_embd`. If it is, there is one buffer
  and two names for it — do not free one.

### C.1 `BufferView`: buffer + offset + format

Weights do not each get their own `MTLBuffer`. They are suballocated, and a
`BufferView` names a slice plus its interpretation:

```text
BufferView { buffer: &MTLBuffer, offset: u64, format: weight_fmt::* }
```

`format` is the thing that cannot be inferred (see A.2: Q4_0 and Q4_K are
byte-identical in length). Every dispatch decision in `decode_fused.rs` keys
off it:

```rust
let gate_up_q4k = layer.gate_proj.format == weight_fmt::Q4_K
    && layer.up_proj.format   == weight_fmt::Q4_K;
```

Get `format` wrong and the wrong kernel dequantizes with the wrong formula:
no crash, no assert, just garbage or subtly-wrong logits. **`format` is not
metadata; it is a correctness invariant.**

---

## Part D — Dtype decisions, and the 1.3-second lesson

Not every tensor should be quantized. From `AGENTS.md` E22:

> PLE `inp_gate`/`proj` are **F32** on Q4_K_M but `qw()` requantized them to
> Q4_0 → slow `projection_q4_batch`. Keep dense **f16** + `mul_mm_f16`.
> Result: PLE Δ **~1555 → ~230 ms** at 4k prefill.

Unpack why this happened, because the shape of the mistake is common:

1. The loader had a general rule: "quantize weights to Q4_0 for the GPU."
2. That rule was written for the big matrices, where 4.5 bits/weight is a
   large bandwidth win.
3. PLE's tensors are small and shipped as F32/F16. Quantizing them bought
   almost nothing in bandwidth and pushed them onto a batch path that was
   never optimized.
4. Cost: 1.3 seconds per 4k prefill — larger than several kernel
   optimizations combined.

The general rule: **the fast path for a tensor depends on its size and its
source dtype, not only on the model's nominal quantization.** Big quantized
matrices → quant matvec/matmul. Small dense tensors → keep dense, use f16
matmul.

Related cleanup in the same experiment: delete `model.q4cache` after load.
Holding the requantized copy wasted memory for weights nothing used again.

---

### D.1 The weight cache, and why staleness is a load-bearing concept

Requantizing every big matrix at startup costs seconds. So the loader persists
the result next to the model:

| File | Contents | Consumer |
|---|---|---|
| `model.q4cache` | GPU-ready weights in the engine's own split layout | uploaded to Metal buffers |
| `model.embed.cache` | token embedding table | mmapped, read on CPU (Ch 09 Part B) |

The embedding table is separate for a good reason: decode looks up exactly one
row per token on the **CPU**, so keeping the ~1 GB table out of GPU memory buys
back slots for KV cache (Ch 13 Part E.1). The two files are written together and
must be treated as one unit.

The interesting part is the error handling, which reads like a list of bugs
someone actually hit:

```971:1000:src/gemma4_gpu_model.rs
                "  Stale Q4 cache (old interleaved layout). Delete model.q4cache and model.embed.cache, then re-run to re-quantize with GGUF layout."
```

Four distinct failure states get distinct messages: *stale layout* (the cache
was written before the packing changed), *partially f16* (written before the
all-Q4 decision), *half-present* (`model.embed.cache` exists, `model.q4cache`
does not), and *unrecognized format*. All four resolve the same way — delete both
files and re-run — but they are reported separately because the ambiguous
version ("cache error") sends people debugging kernels.

Two rules follow, and they will save you an afternoon each:

1. **Any change to weight packing, quantization choice, or dtype policy
   invalidates the cache.** If you edit the loader and see no behavior change,
   you are running yesterday's weights. Delete both files first, then measure.
2. **A stale cache is a silent-wrong-answer generator, not a crash.** The layout
   checks catch the shapes they know about; a subtler change (same byte length,
   different interleave — see Ch 05 Part E on Q4_0 and Q4_K having *identical*
   sizes) can slip through and produce fluent nonsense.

That second rule is the same class of bug as the KV-layout mistakes in Ch 06:
a size check is not a format check.

---

## Part E — Verifying a load in five minutes

Do this whenever you touch loading, or when output is "almost right."

**1. Inspect the file before blaming the code:**

```bash
python3 tools/gguf_inspect.py /path/to/model.gguf | head -50
```

Check `general.architecture`, layer count, `sliding_window_pattern` length,
and per-tensor types. If `blk.24.attn_k.weight` is absent, layer 24 is a
shared-KV layer — expected, not a bug.

**2. Check config against the file.** `hidden_size`, `num_attention_heads`,
`head_dim`, `intermediate_size`, `hidden_size_per_layer_input`,
`sliding_window`, both RoPE thetas. A default silently swallowing a missing
key is the quietest possible failure — `unwrap_or(1e-6)` looks fine right up
until the checkpoint used `1e-5`.

**3. Check formats per tensor.** Print `format` for `q_proj`, `gate_proj`,
`up_proj`, `down_proj` on layer 0. On Q4_K_M expect mostly Q4_K with Q6_K on
`down_proj`. Now predict which MLP branch runs (Ch 08 Part C.4) and confirm
with `PROFILE_DISPATCHES=1`.

**4. Sanity-check magnitudes.** Dequantize a few rows on CPU and look at the
range. Norm weights near 1.0; projection weights small and roughly
zero-centered. All zeros means a name miss. Values in the thousands means a
scale/dims misread.

**5. Run the smoke prompts.** `Hello.` should produce coherent text. Then a
needle test (`ZEBRA42` mid-context) to exercise long-range attention — that
is what catches KV-layout and shared-layer-anchor mistakes, which look
perfect on short prompts.

---

## Part F — Exercises

1. `gate_proj` is `[10240, 2560]` in Q4_K. Compute `dims`, `ne0()`,
   `n_rows()`, `num_elements()`, `byte_len()`. Then redo it for Q6_K and for
   F16.

2. A tensor has `num_elements() = 2570` and type Q4_K. What does `byte_len()`
   do? Why is failing here better than proceeding?

3. Given only `byte_len()` and `dims`, can you distinguish Q4_0 from Q4_K?
   Prove your answer with the numbers.

4. Explain how `tensor_row_bytes` makes a 262k-row embedding table cheap.
   What would the alternative cost at load time and at steady state?

5. `token_embd` is BF16 in many Gemma GGUFs. Where does the BF16→F32
   conversion happen, and why is that acceptable per decode step?

6. Trace `sliding_window_pattern` from metadata to a dispatch decision: which
   field does it set, which struct holds it, which kernel argument does it
   eventually change?

7. Suppose you add a `blk.N.ffn_gate_b.weight` bias tensor to a checkpoint but
   forget to load it. What is the observable symptom, and how would you catch
   it with step 4 of Part E?

---

## Checklist

- [ ] I can draw the GGUF layout and say why parsing is sequential.
- [ ] I know `ne0` is the innermost dim and what that means for matvec.
- [ ] I can compute `byte_len()` for any type in the table from the block spec.
- [ ] I know Q4_0 and Q4_K have identical byte lengths, and why that matters.
- [ ] I know why `mmap` + row slicing beats eager dequant for embeddings.
- [ ] I can name the metadata keys that define Gemma4's two attention types.
- [ ] I know what `BufferView.format` protects against.
- [ ] I can tell the PLE-dtype story and state its general lesson.

**Next:** [04_metal_runtime.md](04_metal_runtime.md) — how these buffers get
bound to kernels.
