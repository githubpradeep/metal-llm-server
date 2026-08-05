# DeepSeek-V4-Flash native port

Status tracker for the clean-room Rust+Metal implementation in `src/deepseek4/`.
ds4 (`reference/ds4/`) is studied for algorithms and used as a logit/token oracle —
kernels and host encode paths are written in-tree (not pasted from ds4).

## Scope

- Flash only (`deepseek4`, 43 layers)
- antirez/ds4 GGUFs only (`~/models/dsv4-0731/…imatrix-0731.gguf`)
- SSD expert streaming required on ≤24–128 GB machines

## Phase checklist

| Phase | Status | Notes |
|-------|--------|-------|
| 0 Scaffold | done | `Dsv4Config`, `GpuModel` dispatch, this doc |
| 1 Loader + IQ2/Q2_K | done | GGUF validate, IQ2_XXS/Q2_K in `gguf.rs` + `quant.rs` + `dsv4_moe.metal` |
| 2 mHC | done | `hc.rs` + `dsv4_hc.metal`; output HC head = n_hc sigmoid weights |
| 3 SWA attention | done | LoRA-Q / MQA latent / **grouped LoRA-O** / tail RoPE / sinks / inv-RoPE |
| 4 CSA/HCA | done | learned compressor (`compressor.rs`), FP8 NoPE round, mixed attn + top-k |
| 5 MoE | done | √softplus, hash-MoE, SwiGLU, IQ2/Q2_K experts via SSD |
| 6 SSD streaming | done | `ssd.rs` expert cache with pread + LRU slots |
| 7 Prefill + product | done | CLI `--dsv4-*`, chat-v2 encode, arch dispatch, `DSV4_PREFILL_CHUNK` |

## Oracle commands (ds4)

```bash
cd reference/ds4
./ds4 -m ~/models/dsv4-0731/…imatrix-0731.gguf --inspect
./ds4 -m … --ssd-streaming --ctx 1024 --nothink -n 32 -p "Say hello in one short sentence."
./ds4 -m … --ssd-streaming --dump-logits /tmp/ds4_logits.json -p "Hi" -n 1
```

## llama-sinks CLI

```bash
# Inspect metadata + validate tensors (no full weight residency beyond mmap)
cargo run --release -- --dsv4-inspect ~/models/dsv4-0731/….gguf

# Unit tests (HC Sinkhorn, MoE top-k, IQ2 layout)
cargo test deepseek4 -- --nocapture

# Load + short greedy gen (SSD streaming; CPU dense path is slow)
cargo run --release -- --dsv4-gen ~/models/dsv4-0731/….gguf --nothink -n 8 -p "Say hello."
```

## Study anchors in ds4 (read, don't copy)

| Topic | Where to read |
|-------|----------------|
| Flash shape | `ds4.c` `DS4_SHAPE_FLASH` ~535 |
| compress ratios | `ds4_expected_layer_compress_ratio` ~1065 |
| metadata keys | `config_validate_deepseek4_model` ~5568 |
| Sinkhorn | `hc_split_sinkhorn_one` ~9656 |
| Grouped LoRA-O | `layer_grouped_out_one` / `matvec_q8_0_grouped_rows` ~10420 |
| Compressor | `compressor_decode_one` ~12445 |
| √softplus router | ~10650 |
| IQ2 block | `metal/moe.metal` `block_iq2_xxs` / `dequantize_iq2_xxs` |
| Layer forward | `layer_forward_raw_swa_one` ~13459 |

## Known gaps vs ds4 parity

- Indexer FP4 QAT + full CSA mask still simplified (top-k on compressed K dots)
- YaRN extras on compressed-layer RoPE not fully ported
- Dense path is CPU matvec (Metal IQ2/Q2_K/HC compile; full graph encode TBD)
- Expect slow tok/s until Metal graph encode matches ds4 overlap policy
- Next correctness gate: `--dump-logits` from ds4 vs our first-token logits MAE

## Correctness fixes landed this slice

- Grouped LoRA-O (`attn_output_a` is `[4096, 8192]`, not flat heads×dim)
- Flash KV is single `head_dim` latent (K≡V), not 2×head_dim
- Output HC fn is `[16384, 4]` sigmoid stream weights (not full Sinkhorn mix)
- **Every token re-seeds HC from its embedding** (ds4 `hc_from_plain_embedding` each decode step)
- Chat encode: `<｜User｜>…<｜Assistant｜></think>` for `--nothink`
- Learned CSA/HCA compressor (score-weighted pool + RoPE + E4M3FN NoPE)
- HC expand comb index `dst + src*n_hc` (match ds4 post; Sinkhorn fill is transposed)
- Tail RoPE with `theta_extrap *= base^(-2/n_rot)`; YaRN on compressed layers
- Sink-aware softmax (sink in denom only, max init from sink)
