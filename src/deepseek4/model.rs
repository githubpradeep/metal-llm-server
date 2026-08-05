//! DeepSeek-V4-Flash model load + session state (compact weight storage).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::gguf::{ggml_type, Gguf};

use super::config::{dsv4_config_from_gguf, Dsv4Config};
use super::dense_matvec::{bytes_to_f16_vec, matvec_f16_weights, matvec_q8_0};
use super::gguf_validate::{print_inspect, validate_dsv4_gguf};
use super::compressor::CompressorState;
use super::kv::{LayerKvConfig, LayerKvState};
use super::metal_ctx::Dsv4Metal;
use super::quant::{IQ2_XXS_BLOCK_BYTES, Q2_K_BLOCK_BYTES, QK_K};
use super::ssd::{estimate_expert_bytes, ExpertBlobLayout, ExpertKey, ExpertSsdCache};

#[derive(Debug, Clone)]
pub enum DenseW {
    F32 { data: Vec<f32>, dims: Vec<usize> },
    F16 { data: Vec<u16>, dims: Vec<usize> },
    Q8 { data: Vec<u8>, dims: Vec<usize> },
}

impl DenseW {
    pub fn from_gguf(g: &Gguf, name: &str) -> Self {
        let info = g.tensor(name).unwrap_or_else(|| panic!("missing {name}"));
        let dims: Vec<usize> = info.dims.iter().map(|&d| d as usize).collect();
        let t = info.ggml_type;
        match t {
            ggml_type::F32 => DenseW::F32 {
                data: g.dequant_to_f32(name),
                dims,
            },
            ggml_type::F16 => DenseW::F16 {
                data: bytes_to_f16_vec(g.tensor_raw(name)),
                dims,
            },
            ggml_type::Q8_0 => DenseW::Q8 {
                data: g.tensor_raw(name).to_vec(),
                dims,
            },
            ggml_type::BF16 => DenseW::F32 {
                data: g.dequant_to_f32(name),
                dims,
            },
            _ => DenseW::F32 {
                data: g.dequant_to_f32(name),
                dims,
            },
        }
    }

    pub fn dims(&self) -> &[usize] {
        match self {
            DenseW::F32 { dims, .. } | DenseW::F16 { dims, .. } | DenseW::Q8 { dims, .. } => dims,
        }
    }

    /// ggml weight layout: dims[0]=ne0 (rows of matrix / input for matvec),
    /// dims[1]=ne1 (columns / output rows for standard ggml mul_mat).
    /// For y = W x with W [ne1, ne0] in math (out, in), ggml stores ne0=in, ne1=out.
    pub fn matvec_ggml(&self, x: &[f32], out: &mut [f32]) {
        let dims = self.dims();
        let n_in = dims[0];
        let n_out = if dims.len() > 1 { dims[1] } else { 1 };
        assert_eq!(x.len(), n_in, "x len {} != ne0 {}", x.len(), n_in);
        assert_eq!(out.len(), n_out, "out len {} != ne1 {}", out.len(), n_out);
        self.matvec(x, n_out, n_in, out);
    }

    pub fn nelems(&self) -> usize {
        match self {
            DenseW::F32 { data, .. } => data.len(),
            DenseW::F16 { data, .. } => data.len(),
            DenseW::Q8 { data, .. } => data.len() / 34 * 32,
        }
    }

    pub fn matvec(&self, x: &[f32], n_out: usize, n_in: usize, out: &mut [f32]) {
        self.matvec_rows(x, 0, n_out, n_in, out);
    }

    /// Matvec over a contiguous slice of output rows `[row_start, row_start + n_out)`.
    /// Used for grouped LoRA-O where each group owns `rank` rows of a shared weight.
    pub fn matvec_rows(
        &self,
        x: &[f32],
        row_start: usize,
        n_out: usize,
        n_in: usize,
        out: &mut [f32],
    ) {
        assert_eq!(out.len(), n_out);
        assert_eq!(x.len(), n_in);
        match self {
            DenseW::F32 { data: w, .. } => {
                let need = (row_start + n_out) * n_in;
                assert!(
                    w.len() >= need,
                    "f32 weight len {} < need {} (start={} out={} in={})",
                    w.len(),
                    need,
                    row_start,
                    n_out,
                    n_in
                );
                for r in 0..n_out {
                    let mut acc = 0.0f32;
                    let row = &w[(row_start + r) * n_in..(row_start + r + 1) * n_in];
                    for i in 0..n_in {
                        acc += row[i] * x[i];
                    }
                    out[r] = acc;
                }
            }
            DenseW::F16 { data: w, .. } => {
                let need = (row_start + n_out) * n_in;
                assert!(
                    w.len() >= need,
                    "f16 weight len {} < need {}",
                    w.len(),
                    need
                );
                matvec_f16_weights(
                    &w[row_start * n_in..(row_start + n_out) * n_in],
                    x,
                    n_out,
                    n_in,
                    out,
                );
            }
            DenseW::Q8 { data: w, .. } => {
                assert_eq!(n_in % 32, 0, "q8 n_in must be multiple of 32");
                let blocks_per_row = n_in / 32;
                let row_bytes = blocks_per_row * 34;
                let need = (row_start + n_out) * row_bytes;
                assert!(
                    w.len() >= need,
                    "q8 weight len {} < need {} (start={} out={} in={})",
                    w.len(),
                    need,
                    row_start,
                    n_out,
                    n_in
                );
                matvec_q8_0(
                    &w[row_start * row_bytes..(row_start + n_out) * row_bytes],
                    x,
                    n_out,
                    n_in,
                    out,
                );
            }
        }
    }

    pub fn as_f32_scale_base(&self) -> Vec<f32> {
        match self {
            DenseW::F32 { data: v, .. } => v.clone(),
            DenseW::F16 { data: v, .. } => v.iter().map(|&h| crate::gpu::f16_to_f32(h)).collect(),
            DenseW::Q8 { .. } => panic!("scale/base should be f32/f16"),
        }
    }
}

#[derive(Debug)]
pub struct LayerWeights {
    pub hc_attn_fn: DenseW,
    pub hc_attn_scale: Vec<f32>,
    pub hc_attn_base: Vec<f32>,
    pub attn_norm: Vec<f32>,
    pub attn_q_a: DenseW,
    pub attn_q_a_norm: Vec<f32>,
    pub attn_q_b: DenseW,
    pub attn_kv: DenseW,
    pub attn_kv_a_norm: Vec<f32>,
    pub attn_sinks: Vec<f32>,
    pub attn_output_a: DenseW,
    pub attn_output_b: DenseW,
    pub hc_ffn_fn: DenseW,
    pub hc_ffn_scale: Vec<f32>,
    pub hc_ffn_base: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub ffn_gate_inp: DenseW,
    pub exp_probs_b: Option<Vec<f32>>,
    pub ffn_gate_shexp: DenseW,
    pub ffn_up_shexp: DenseW,
    pub ffn_down_shexp: DenseW,
    pub tid2eid: Option<Vec<i32>>,
    pub attn_compressor: Option<CompressorWeights>,
    pub indexer_compressor: Option<CompressorWeights>,
}

#[derive(Debug)]
pub struct CompressorWeights {
    pub ape: DenseW,
    pub kv: DenseW,
    pub gate: DenseW,
    pub norm: Vec<f32>,
}

pub struct Dsv4GpuModel {
    pub path: PathBuf,
    pub cfg: Dsv4Config,
    pub metal: Arc<Dsv4Metal>,
    pub token_embd: DenseW,
    pub output_weight: DenseW,
    pub output_norm: Vec<f32>,
    pub output_hc_fn: DenseW,
    pub output_hc_scale: Vec<f32>,
    pub output_hc_base: Vec<f32>,
    pub layers: Vec<LayerWeights>,
    pub kv: Vec<LayerKvState>,
    pub attn_comp_state: Vec<Option<CompressorState>>,
    pub idx_comp_state: Vec<Option<CompressorState>>,
    pub hc: Vec<f32>,
    pub pos: usize,
    pub ssd: ExpertSsdCache,
    pub ssd_streaming: bool,
    pub nothink: bool,
    pub expert_gate_type: u32,
    pub expert_down_type: u32,
    pub gate_row_bytes: usize,
    pub up_row_bytes: usize,
    pub down_row_bytes: usize,
}

fn load_f32(g: &Gguf, name: &str) -> Vec<f32> {
    g.dequant_to_f32(name)
}

fn load_i32(g: &Gguf, name: &str) -> Vec<i32> {
    let raw = g.tensor_raw(name);
    (0..raw.len() / 4)
        .map(|i| {
            let o = i * 4;
            i32::from_le_bytes([raw[o], raw[o + 1], raw[o + 2], raw[o + 3]])
        })
        .collect()
}

impl Dsv4GpuModel {
    pub fn inspect_only(path: impl AsRef<Path>) -> Dsv4Config {
        let g = Gguf::open(path);
        let cfg = dsv4_config_from_gguf(&g);
        validate_dsv4_gguf(&g, &cfg).expect("validate");
        print_inspect(&g, &cfg);
        cfg
    }

    pub fn load_from_gguf(
        path: impl AsRef<Path>,
        ssd_streaming: bool,
        cache_experts_mib: Option<usize>,
    ) -> Self {
        let path = path.as_ref().to_path_buf();
        println!("Loading DeepSeek-V4-Flash from {}", path.display());
        let g = Gguf::open(&path);
        let cfg = dsv4_config_from_gguf(&g);
        validate_dsv4_gguf(&g, &cfg).expect("validate");
        print_inspect(&g, &cfg);

        let metal = Dsv4Metal::new();
        println!("  dsv4 Metal kernels compiled OK");

        let token_embd = DenseW::from_gguf(&g, "token_embd.weight");
        let output_weight = DenseW::from_gguf(&g, "output.weight");
        let output_norm = load_f32(&g, "output_norm.weight");
        let output_hc_fn = DenseW::from_gguf(&g, "output_hc_fn.weight");
        let output_hc_scale = load_f32(&g, "output_hc_scale.weight");
        let output_hc_base = load_f32(&g, "output_hc_base.weight");

        let expert_gate_type = g.tensor_type("blk.0.ffn_gate_exps.weight");
        let expert_down_type = g.tensor_type("blk.0.ffn_down_exps.weight");
        let gate_row_bytes = match expert_gate_type {
            ggml_type::IQ2_XXS => (cfg.n_embd / QK_K) * IQ2_XXS_BLOCK_BYTES,
            ggml_type::Q4_K => (cfg.n_embd / QK_K) * 144,
            ggml_type::Q8_0 => (cfg.n_embd / 32) * 34,
            t => panic!("unsupported expert gate type {t}"),
        };
        let up_row_bytes = gate_row_bytes;
        let down_row_bytes = match expert_down_type {
            ggml_type::Q2_K => (cfg.n_ff_exp / QK_K) * Q2_K_BLOCK_BYTES,
            ggml_type::Q4_K => (cfg.n_ff_exp / QK_K) * 144,
            ggml_type::Q8_0 => (cfg.n_ff_exp / 32) * 34,
            t => panic!("unsupported expert down type {t}"),
        };

        let expert_bytes = estimate_expert_bytes(cfg.n_embd, cfg.n_ff_exp);
        let budget_mib = cache_experts_mib.unwrap_or(if ssd_streaming { 2048 } else { 4096 });
        let n_slots = ExpertSsdCache::slots_for_budget(budget_mib, expert_bytes);
        println!(
            "  SSD expert cache: {n_slots} slots (~{budget_mib} MiB, {:.2} MiB/expert)",
            expert_bytes as f64 / (1024.0 * 1024.0)
        );
        let mut ssd = ExpertSsdCache::new(&path, n_slots).expect("SSD cache");

        let mut layers = Vec::with_capacity(cfg.n_layer);
        let mut attn_comp_state = Vec::with_capacity(cfg.n_layer);
        let mut idx_comp_state = Vec::with_capacity(cfg.n_layer);
        for il in 0..cfg.n_layer {
            let ratio = cfg.compress_ratio(il);
            let attn_compressor = if ratio != 0 {
                Some(CompressorWeights {
                    ape: DenseW::from_gguf(&g, &format!("blk.{il}.attn_compressor_ape.weight")),
                    kv: DenseW::from_gguf(&g, &format!("blk.{il}.attn_compressor_kv.weight")),
                    gate: DenseW::from_gguf(&g, &format!("blk.{il}.attn_compressor_gate.weight")),
                    norm: load_f32(&g, &format!("blk.{il}.attn_compressor_norm.weight")),
                })
            } else {
                None
            };
            let indexer_compressor = if ratio == 4 {
                Some(CompressorWeights {
                    ape: DenseW::from_gguf(&g, &format!("blk.{il}.indexer_compressor_ape.weight")),
                    kv: DenseW::from_gguf(&g, &format!("blk.{il}.indexer_compressor_kv.weight")),
                    gate: DenseW::from_gguf(
                        &g,
                        &format!("blk.{il}.indexer_compressor_gate.weight"),
                    ),
                    norm: load_f32(&g, &format!("blk.{il}.indexer_compressor_norm.weight")),
                })
            } else {
                None
            };
            let lw = LayerWeights {
                hc_attn_fn: DenseW::from_gguf(&g, &format!("blk.{il}.hc_attn_fn.weight")),
                hc_attn_scale: load_f32(&g, &format!("blk.{il}.hc_attn_scale.weight")),
                hc_attn_base: load_f32(&g, &format!("blk.{il}.hc_attn_base.weight")),
                attn_norm: load_f32(&g, &format!("blk.{il}.attn_norm.weight")),
                attn_q_a: DenseW::from_gguf(&g, &format!("blk.{il}.attn_q_a.weight")),
                attn_q_a_norm: load_f32(&g, &format!("blk.{il}.attn_q_a_norm.weight")),
                attn_q_b: DenseW::from_gguf(&g, &format!("blk.{il}.attn_q_b.weight")),
                attn_kv: DenseW::from_gguf(&g, &format!("blk.{il}.attn_kv.weight")),
                attn_kv_a_norm: load_f32(&g, &format!("blk.{il}.attn_kv_a_norm.weight")),
                attn_sinks: load_f32(&g, &format!("blk.{il}.attn_sinks.weight")),
                attn_output_a: DenseW::from_gguf(&g, &format!("blk.{il}.attn_output_a.weight")),
                attn_output_b: DenseW::from_gguf(&g, &format!("blk.{il}.attn_output_b.weight")),
                hc_ffn_fn: DenseW::from_gguf(&g, &format!("blk.{il}.hc_ffn_fn.weight")),
                hc_ffn_scale: load_f32(&g, &format!("blk.{il}.hc_ffn_scale.weight")),
                hc_ffn_base: load_f32(&g, &format!("blk.{il}.hc_ffn_base.weight")),
                ffn_norm: load_f32(&g, &format!("blk.{il}.ffn_norm.weight")),
                ffn_gate_inp: DenseW::from_gguf(&g, &format!("blk.{il}.ffn_gate_inp.weight")),
                exp_probs_b: if g.has_tensor(&format!("blk.{il}.exp_probs_b.bias")) {
                    Some(load_f32(&g, &format!("blk.{il}.exp_probs_b.bias")))
                } else {
                    None
                },
                ffn_gate_shexp: DenseW::from_gguf(&g, &format!("blk.{il}.ffn_gate_shexp.weight")),
                ffn_up_shexp: DenseW::from_gguf(&g, &format!("blk.{il}.ffn_up_shexp.weight")),
                ffn_down_shexp: DenseW::from_gguf(&g, &format!("blk.{il}.ffn_down_shexp.weight")),
                tid2eid: if cfg.is_hash_layer(il) {
                    Some(load_i32(&g, &format!("blk.{il}.ffn_gate_tid2eid.weight")))
                } else {
                    None
                },
                attn_compressor,
                indexer_compressor,
            };
            layers.push(lw);
            attn_comp_state.push(CompressorState::new(ratio, cfg.head_dim));
            idx_comp_state.push(if ratio == 4 {
                CompressorState::new(ratio, cfg.n_indexer_head_dim)
            } else {
                None
            });

            let gate_base = g.tensor_file_offset(&format!("blk.{il}.ffn_gate_exps.weight"));
            let up_base = g.tensor_file_offset(&format!("blk.{il}.ffn_up_exps.weight"));
            let down_base = g.tensor_file_offset(&format!("blk.{il}.ffn_down_exps.weight"));
            let gate_expert_bytes = gate_row_bytes * cfg.n_ff_exp;
            let up_expert_bytes = up_row_bytes * cfg.n_ff_exp;
            let down_expert_bytes = down_row_bytes * cfg.n_embd;
            for e in 0..cfg.n_expert {
                ssd.register(
                    ExpertKey {
                        layer: il as u16,
                        expert: e as u16,
                    },
                    ExpertBlobLayout {
                        gate_offset: gate_base + (e as u64) * gate_expert_bytes as u64,
                        gate_bytes: gate_expert_bytes,
                        up_offset: up_base + (e as u64) * up_expert_bytes as u64,
                        up_bytes: up_expert_bytes,
                        down_offset: down_base + (e as u64) * down_expert_bytes as u64,
                        down_bytes: down_expert_bytes,
                    },
                );
            }
            if il % 8 == 0 || il + 1 == cfg.n_layer {
                println!("  loaded dense layer {}/{}", il + 1, cfg.n_layer);
            }
        }

        let kv: Vec<_> = (0..cfg.n_layer)
            .map(|il| LayerKvState::new(LayerKvConfig::from_cfg(&cfg, il), 2048))
            .collect();

        println!("  DeepSeek-V4-Flash ready (ssd_streaming={ssd_streaming})");
        let hc_elems = cfg.hc_state_elems();
        Self {
            path,
            cfg,
            metal,
            token_embd,
            output_weight,
            output_norm,
            output_hc_fn,
            output_hc_scale,
            output_hc_base,
            layers,
            kv,
            attn_comp_state,
            idx_comp_state,
            hc: vec![0.0; hc_elems],
            pos: 0,
            ssd,
            ssd_streaming,
            nothink: false,
            expert_gate_type,
            expert_down_type,
            gate_row_bytes,
            up_row_bytes,
            down_row_bytes,
        }
    }

    pub fn reset(&mut self) {
        self.pos = 0;
        self.hc.fill(0.0);
        for kv in &mut self.kv {
            kv.clear();
        }
        for s in &mut self.attn_comp_state {
            if let Some(st) = s.as_mut() {
                st.clear();
            }
        }
        for s in &mut self.idx_comp_state {
            if let Some(st) = s.as_mut() {
                st.clear();
            }
        }
    }

    pub fn embed(&self, token: usize) -> Vec<f32> {
        let e = self.cfg.n_embd;
        let mut out = vec![0.0f32; e];
        match &self.token_embd {
            DenseW::F32 { data: w, .. } => out.copy_from_slice(&w[token * e..(token + 1) * e]),
            DenseW::F16 { data: w, .. } => {
                for i in 0..e {
                    out[i] = crate::gpu::f16_to_f32(w[token * e + i]);
                }
            }
            DenseW::Q8 { .. } => panic!("embd should be f16/f32"),
        }
        out
    }
}
