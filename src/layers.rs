use ndarray::{Array2, Array3, s};
use crate::cache::StreamingKVCache;
use crate::config::LlamaConfig;
use crate::quantize::QuantizedLinear;
use crate::weights::ModelWeights;

// ─── RMS Norm ────────────────────────────────────────────────────────────────

pub struct RMSNorm {
    pub weight: Vec<f32>,
    pub eps: f32,
}

impl RMSNorm {
    pub fn new(weight: Vec<f32>, eps: f64) -> Self {
        RMSNorm { weight, eps: eps as f32 }
    }

    pub fn forward(&self, x: &Array3<f32>) -> Array3<f32> {
        let shape = x.shape();
        let batch = shape[0];
        let seq = shape[1];
        let dim = shape[2];
        let x_raw = x.as_slice().unwrap();
        let mut out = vec![0.0f32; batch * seq * dim];

        for bs in 0..(batch * seq) {
            let offset = bs * dim;
            let row = &x_raw[offset..offset + dim];
            let mut sum_sq = 0.0f32;
            for &v in row.iter() {
                sum_sq += v * v;
            }
            let inv_rms = 1.0 / (sum_sq / dim as f32 + self.eps).sqrt();
            let out_row = &mut out[offset..offset + dim];
            for d in 0..dim {
                out_row[d] = row[d] * inv_rms * self.weight[d];
            }
        }
        Array3::from_shape_vec((batch, seq, dim), out).unwrap()
    }

    #[inline]
    pub fn forward_vec(&self, x: &[f32], out: &mut [f32]) {
        let dim = self.weight.len();
        let mut sum_sq = 0.0f32;
        for &v in x.iter() {
            sum_sq += v * v;
        }
        let inv_rms = 1.0 / (sum_sq / dim as f32 + self.eps).sqrt();
        for d in 0..dim {
            out[d] = x[d] * inv_rms * self.weight[d];
        }
    }
}

// ─── Rotary Embedding ────────────────────────────────────────────────────────

pub struct RotaryEmbedding {
    pub inv_freq: Vec<f32>,
}

impl RotaryEmbedding {
    pub fn new(dim: usize, _max_position_embeddings: usize, base: f64) -> Self {
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1.0 / base.powf(i as f64 / dim as f64) as f32)
            .collect();
        RotaryEmbedding { inv_freq }
    }

    pub fn forward(&self, position_ids: &[Vec<i64>]) -> (Vec<f32>, Vec<f32>) {
        let batch = position_ids.len();
        let seq_len = position_ids[0].len();
        let half_dim = self.inv_freq.len();
        let full_dim = half_dim * 2;

        let mut cos_data = vec![0.0f32; batch * seq_len * full_dim];
        let mut sin_data = vec![0.0f32; batch * seq_len * full_dim];

        for b in 0..batch {
            for s_idx in 0..seq_len {
                let pos = position_ids[b][s_idx] as f32;
                let offset = (b * seq_len + s_idx) * full_dim;
                for (i, &freq) in self.inv_freq.iter().enumerate() {
                    let angle = pos * freq;
                    let c = angle.cos();
                    let s_val = angle.sin();
                    cos_data[offset + i] = c;
                    cos_data[offset + i + half_dim] = c;
                    sin_data[offset + i] = s_val;
                    sin_data[offset + i + half_dim] = s_val;
                }
            }
        }
        (cos_data, sin_data)
    }
}

// ─── Rotary Position Embedding Application ───────────────────────────────────

pub fn apply_rotary_pos_emb_flat(
    q: &mut [f32],
    k: &mut [f32],
    cos: &[f32],
    sin: &[f32],
    num_heads: usize,
    num_kv_heads: usize,
    seq: usize,
    head_dim: usize,
) {
    let half = head_dim / 2;

    for s_idx in 0..seq {
        let cs_offset = s_idx * head_dim;

        for h in 0..num_heads {
            let q_offset = (h * seq + s_idx) * head_dim;
            for d in 0..half {
                let q1 = q[q_offset + d];
                let q2 = q[q_offset + d + half];
                let c = cos[cs_offset + d];
                let s = sin[cs_offset + d];
                q[q_offset + d] = q1 * c - q2 * s;
                q[q_offset + d + half] = q2 * c + q1 * s;
            }
        }

        for h in 0..num_kv_heads {
            let k_offset = (h * seq + s_idx) * head_dim;
            for d in 0..half {
                let k1 = k[k_offset + d];
                let k2 = k[k_offset + d + half];
                let c = cos[cs_offset + d];
                let s = sin[cs_offset + d];
                k[k_offset + d] = k1 * c - k2 * s;
                k[k_offset + d + half] = k2 * c + k1 * s;
            }
        }
    }
}

// ─── SiLU ────────────────────────────────────────────────────────────────────

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

// ─── MLP (Quantized) ─────────────────────────────────────────────────────────

pub struct LlamaMLP {
    pub gate_up_proj: QuantizedLinear,
    pub down_proj: QuantizedLinear,
    pub intermediate_size: usize,
}

impl LlamaMLP {
    pub fn new(config: &LlamaConfig, layer_idx: usize, weights: &ModelWeights) -> Self {
        let gate_w = weights.get_2d(&format!("model.layers.{}.mlp.gate_proj.weight", layer_idx));
        let up_w = weights.get_2d(&format!("model.layers.{}.mlp.up_proj.weight", layer_idx));
        let down_w = weights.get_2d(&format!("model.layers.{}.mlp.down_proj.weight", layer_idx));

        let intermediate = config.intermediate_size;
        let hidden = config.hidden_size;

        // Fuse gate + up
        let mut fused = vec![0.0f32; 2 * intermediate * hidden];
        let gate_slice = gate_w.as_slice().unwrap();
        let up_slice = up_w.as_slice().unwrap();
        fused[..intermediate * hidden].copy_from_slice(gate_slice);
        fused[intermediate * hidden..].copy_from_slice(up_slice);

        let gate_up_q = QuantizedLinear::from_f32(&fused, 2 * intermediate, hidden);
        let down_q = QuantizedLinear::from_f32(down_w.as_slice().unwrap(), hidden, intermediate);

        LlamaMLP {
            gate_up_proj: gate_up_q,
            down_proj: down_q,
            intermediate_size: intermediate,
        }
    }

    #[inline]
    pub fn forward_vec(&self, x: &[f32], output: &mut [f32]) {
        let intermediate = self.intermediate_size;
        let mut gate_up = vec![0.0f32; 2 * intermediate];
        self.gate_up_proj.forward_vec(x, &mut gate_up);

        let mut activated = vec![0.0f32; intermediate];
        for i in 0..intermediate {
            let g = gate_up[i];
            activated[i] = (g / (1.0 + (-g).exp())) * gate_up[i + intermediate];
        }

        self.down_proj.forward_vec(&activated, output);
    }

    pub fn forward(&self, x: &Array3<f32>) -> Array3<f32> {
        let shape = x.shape();
        let batch = shape[0];
        let seq = shape[1];
        let m = batch * seq;

        let x_raw = x.as_slice().unwrap();
        let mut gate_up = vec![0.0f32; m * 2 * self.intermediate_size];
        self.gate_up_proj.forward_batch(x_raw, m, &mut gate_up);

        let mut activated = vec![0.0f32; m * self.intermediate_size];
        for bs in 0..m {
            let offset = bs * (2 * self.intermediate_size);
            let out_offset = bs * self.intermediate_size;
            for i in 0..self.intermediate_size {
                let g = gate_up[offset + i];
                activated[out_offset + i] = silu(g) * gate_up[offset + self.intermediate_size + i];
            }
        }

        let mut output = vec![0.0f32; m * self.down_proj.out_features];
        self.down_proj.forward_batch(&activated, m, &mut output);

        Array3::from_shape_vec((batch, seq, self.down_proj.out_features), output).unwrap()
    }
}

// ─── Attention (Quantized) ───────────────────────────────────────────────────

pub struct LlamaAttention {
    pub q_proj: QuantizedLinear,
    pub k_proj: QuantizedLinear,
    pub v_proj: QuantizedLinear,
    pub o_proj: QuantizedLinear,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_kv_groups: usize,
    pub layer_idx: usize,
}

impl LlamaAttention {
    pub fn new(config: &LlamaConfig, layer_idx: usize, weights: &ModelWeights) -> Self {
        let head_dim = config.head_dim();
        let q_w = weights.get_2d(&format!("model.layers.{}.self_attn.q_proj.weight", layer_idx));
        let k_w = weights.get_2d(&format!("model.layers.{}.self_attn.k_proj.weight", layer_idx));
        let v_w = weights.get_2d(&format!("model.layers.{}.self_attn.v_proj.weight", layer_idx));
        let o_w = weights.get_2d(&format!("model.layers.{}.self_attn.o_proj.weight", layer_idx));

        let hidden = config.hidden_size;
        let q_out = config.num_attention_heads * head_dim;
        let kv_out = config.num_key_value_heads * head_dim;

        LlamaAttention {
            q_proj: QuantizedLinear::from_f32(q_w.as_slice().unwrap(), q_out, hidden),
            k_proj: QuantizedLinear::from_f32(k_w.as_slice().unwrap(), kv_out, hidden),
            v_proj: QuantizedLinear::from_f32(v_w.as_slice().unwrap(), kv_out, hidden),
            o_proj: QuantizedLinear::from_f32(o_w.as_slice().unwrap(), hidden, q_out),
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim,
            num_kv_groups: config.num_key_value_groups(),
            layer_idx,
        }
    }

    pub fn forward(
        &self,
        hidden_states: &Array3<f32>,
        cos: &[f32],
        sin: &[f32],
        kv_cache: &mut StreamingKVCache,
    ) -> Array3<f32> {
        let shape = hidden_states.shape();
        let bsz = shape[0];
        let q_len = shape[1];
        let head_dim = self.head_dim;
        let num_heads = self.num_heads;
        let num_kv_heads = self.num_kv_heads;
        let hidden_size = num_heads * head_dim;

        if q_len == 1 && bsz == 1 {
            // ─── Fast single-token decode ───
            let x = hidden_states.as_slice().unwrap();

            let q_size = num_heads * head_dim;
            let kv_size = num_kv_heads * head_dim;
            let mut q_buf = vec![0.0f32; q_size];
            let mut k_buf = vec![0.0f32; kv_size];
            let mut v_buf = vec![0.0f32; kv_size];

            self.q_proj.forward_vec(x, &mut q_buf);
            self.k_proj.forward_vec(x, &mut k_buf);
            self.v_proj.forward_vec(x, &mut v_buf);

            // Apply rotary embeddings in-place
            apply_rotary_pos_emb_flat(
                &mut q_buf, &mut k_buf,
                cos, sin, num_heads, num_kv_heads, 1, head_dim,
            );

            // Update KV cache
            let kv_seq = kv_cache.update(
                &k_buf, &v_buf, 1, num_kv_heads, head_dim, self.layer_idx,
            );

            let scale = 1.0 / (head_dim as f32).sqrt();
            let (k_cache, _, k_cap) = kv_cache.get_key_slice(self.layer_idx);
            let (v_cache, _, v_cap) = kv_cache.get_value_slice(self.layer_idx);

            // Attention (virtual GQA)
            let mut attn_out = vec![0.0f32; hidden_size];
            self.attention_single_token(
                &q_buf, k_cache, v_cache, k_cap, v_cap, kv_seq, scale, &mut attn_out,
            );

            // O projection
            let mut output = vec![0.0f32; hidden_size];
            self.o_proj.forward_vec(&attn_out, &mut output);

            Array3::from_shape_vec((1, 1, hidden_size), output).unwrap()
        } else {
            // ─── Multi-token prefill ───
            let x_raw = hidden_states.as_slice().unwrap();
            let m = bsz * q_len;

            let mut q_raw = vec![0.0f32; m * num_heads * head_dim];
            let mut k_raw = vec![0.0f32; m * num_kv_heads * head_dim];
            let mut v_raw = vec![0.0f32; m * num_kv_heads * head_dim];

            self.q_proj.forward_batch(x_raw, m, &mut q_raw);
            self.k_proj.forward_batch(x_raw, m, &mut k_raw);
            self.v_proj.forward_batch(x_raw, m, &mut v_raw);

            // Transpose: (seq, heads, head_dim) → (heads, seq, head_dim)
            let mut q_transposed = vec![0.0f32; num_heads * q_len * head_dim];
            let mut k_transposed = vec![0.0f32; num_kv_heads * q_len * head_dim];
            let mut v_transposed = vec![0.0f32; num_kv_heads * q_len * head_dim];

            for s_idx in 0..q_len {
                let src_base = s_idx * num_heads * head_dim;
                for h in 0..num_heads {
                    let src_offset = src_base + h * head_dim;
                    let dst_offset = (h * q_len + s_idx) * head_dim;
                    q_transposed[dst_offset..dst_offset + head_dim]
                        .copy_from_slice(&q_raw[src_offset..src_offset + head_dim]);
                }
                let ksrc_base = s_idx * num_kv_heads * head_dim;
                for h in 0..num_kv_heads {
                    let src_offset = ksrc_base + h * head_dim;
                    let dst_offset = (h * q_len + s_idx) * head_dim;
                    k_transposed[dst_offset..dst_offset + head_dim]
                        .copy_from_slice(&k_raw[src_offset..src_offset + head_dim]);
                    v_transposed[dst_offset..dst_offset + head_dim]
                        .copy_from_slice(&v_raw[src_offset..src_offset + head_dim]);
                }
            }

            apply_rotary_pos_emb_flat(
                &mut q_transposed, &mut k_transposed,
                cos, sin, num_heads, num_kv_heads, q_len, head_dim,
            );

            let kv_seq = kv_cache.update(
                &k_transposed, &v_transposed, q_len, num_kv_heads, head_dim, self.layer_idx,
            );

            let scale = 1.0 / (head_dim as f32).sqrt();
            let (k_cache, _, k_cap) = kv_cache.get_key_slice(self.layer_idx);
            let (v_cache, _, v_cap) = kv_cache.get_value_slice(self.layer_idx);

            let mut attn_out = vec![0.0f32; num_heads * q_len * head_dim];
            self.attention_multi_token(
                &q_transposed, k_cache, v_cache, k_cap, v_cap,
                q_len, kv_seq, scale, &mut attn_out,
            );

            // Transpose back
            let mut o_input = vec![0.0f32; m * hidden_size];
            for s_idx in 0..q_len {
                for h in 0..num_heads {
                    let src_offset = (h * q_len + s_idx) * head_dim;
                    let dst_offset = s_idx * hidden_size + h * head_dim;
                    o_input[dst_offset..dst_offset + head_dim]
                        .copy_from_slice(&attn_out[src_offset..src_offset + head_dim]);
                }
            }

            let mut output = vec![0.0f32; m * hidden_size];
            self.o_proj.forward_batch(&o_input, m, &mut output);

            Array3::from_shape_vec((bsz, q_len, hidden_size), output).unwrap()
        }
    }

    #[inline(never)]
    fn attention_single_token(
        &self,
        q: &[f32],
        k_buf: &[f32],
        v_buf: &[f32],
        k_cap: usize,
        v_cap: usize,
        kv_seq: usize,
        scale: f32,
        out: &mut [f32],
    ) {
        let head_dim = self.head_dim;
        let num_heads = self.num_heads;
        let num_kv_groups = self.num_kv_groups;

        let mut scores = vec![0.0f32; kv_seq];

        for h in 0..num_heads {
            let kv_h = h / num_kv_groups;
            let q_offset = h * head_dim;
            let k_head_base = kv_h * k_cap * head_dim;
            let v_head_base = kv_h * v_cap * head_dim;

            let mut max_score = f32::NEG_INFINITY;
            for kv in 0..kv_seq {
                let k_offset = k_head_base + kv * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[q_offset + d] * k_buf[k_offset + d];
                }
                let s = dot * scale;
                scores[kv] = s;
                if s > max_score {
                    max_score = s;
                }
            }

            let mut sum = 0.0f32;
            for s in scores[..kv_seq].iter_mut() {
                *s = (*s - max_score).exp();
                sum += *s;
            }
            let inv_sum = 1.0 / sum;
            for s in scores[..kv_seq].iter_mut() {
                *s *= inv_sum;
            }

            let out_offset = h * head_dim;
            let out_slice = &mut out[out_offset..out_offset + head_dim];
            out_slice.fill(0.0);
            for kv in 0..kv_seq {
                let w = scores[kv];
                if w > 1e-10 {
                    let v_offset = v_head_base + kv * head_dim;
                    for d in 0..head_dim {
                        out_slice[d] += w * v_buf[v_offset + d];
                    }
                }
            }
        }
    }

    fn attention_multi_token(
        &self,
        q: &[f32],
        k_buf: &[f32],
        v_buf: &[f32],
        k_cap: usize,
        v_cap: usize,
        q_len: usize,
        kv_seq: usize,
        scale: f32,
        out: &mut [f32],
    ) {
        let head_dim = self.head_dim;
        let num_heads = self.num_heads;
        let num_kv_groups = self.num_kv_groups;

        let mut attn_row = vec![0.0f32; kv_seq];

        for h in 0..num_heads {
            let kv_h = h / num_kv_groups;
            let k_head_base = kv_h * k_cap * head_dim;
            let v_head_base = kv_h * v_cap * head_dim;

            for qi in 0..q_len {
                let q_offset = (h * q_len + qi) * head_dim;

                let mut max_val = f32::NEG_INFINITY;
                for ki in 0..kv_seq {
                    let k_offset = k_head_base + ki * head_dim;
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[q_offset + d] * k_buf[k_offset + d];
                    }
                    let s = dot * scale;
                    attn_row[ki] = s;
                    if s > max_val {
                        max_val = s;
                    }
                }

                if q_len == kv_seq {
                    for ki in (qi + 1)..kv_seq {
                        attn_row[ki] = f32::NEG_INFINITY;
                    }
                    max_val = attn_row[..=qi].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                } else if q_len < kv_seq {
                    let new_start = kv_seq - q_len;
                    for ki in (new_start + qi + 1)..kv_seq {
                        attn_row[ki] = f32::NEG_INFINITY;
                    }
                    max_val = attn_row[..kv_seq].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                }

                let mut sum = 0.0f32;
                for w in attn_row[..kv_seq].iter_mut() {
                    *w = (*w - max_val).exp();
                    sum += *w;
                }
                let inv_sum = 1.0 / sum;
                for w in attn_row[..kv_seq].iter_mut() {
                    *w *= inv_sum;
                }

                let out_offset = (h * q_len + qi) * head_dim;
                let out_slice = &mut out[out_offset..out_offset + head_dim];
                out_slice.fill(0.0);
                for ki in 0..kv_seq {
                    let w = attn_row[ki];
                    if w > 1e-10 {
                        let v_offset = v_head_base + ki * head_dim;
                        for d in 0..head_dim {
                            out_slice[d] += w * v_buf[v_offset + d];
                        }
                    }
                }
            }
        }
    }
}

// ─── Decoder Layer ───────────────────────────────────────────────────────────

pub struct LlamaDecoderLayer {
    pub self_attn: LlamaAttention,
    pub mlp: LlamaMLP,
    pub input_layernorm: RMSNorm,
    pub post_attention_layernorm: RMSNorm,
}

impl LlamaDecoderLayer {
    pub fn new(config: &LlamaConfig, layer_idx: usize, weights: &ModelWeights) -> Self {
        let input_ln_w = weights.get_1d(&format!(
            "model.layers.{}.input_layernorm.weight", layer_idx
        ));
        let post_ln_w = weights.get_1d(&format!(
            "model.layers.{}.post_attention_layernorm.weight", layer_idx
        ));

        LlamaDecoderLayer {
            self_attn: LlamaAttention::new(config, layer_idx, weights),
            mlp: LlamaMLP::new(config, layer_idx, weights),
            input_layernorm: RMSNorm::new(input_ln_w, config.rms_norm_eps),
            post_attention_layernorm: RMSNorm::new(post_ln_w, config.rms_norm_eps),
        }
    }

    #[inline]
    pub fn forward_vec(
        &self,
        hidden_states: &[f32],
        cos: &[f32],
        sin: &[f32],
        kv_cache: &mut StreamingKVCache,
        output: &mut [f32],
    ) {
        let dim = self.input_layernorm.weight.len();
        let mut normed = vec![0.0f32; dim];

        self.input_layernorm.forward_vec(hidden_states, &mut normed);

        let hidden_size = self.self_attn.num_heads * self.self_attn.head_dim;
        let normed_arr = Array3::from_shape_vec((1, 1, hidden_size), normed).unwrap();
        let attn_output = self.self_attn.forward(&normed_arr, cos, sin, kv_cache);
        let attn_slice = attn_output.as_slice().unwrap();

        let mut residual1 = vec![0.0f32; dim];
        for d in 0..dim {
            residual1[d] = hidden_states[d] + attn_slice[d];
        }

        let mut normed2 = vec![0.0f32; dim];
        self.post_attention_layernorm.forward_vec(&residual1, &mut normed2);

        let mut mlp_out = vec![0.0f32; dim];
        self.mlp.forward_vec(&normed2, &mut mlp_out);

        for d in 0..dim {
            output[d] = residual1[d] + mlp_out[d];
        }
    }

    pub fn forward(
        &self,
        hidden_states: &Array3<f32>,
        cos: &[f32],
        sin: &[f32],
        kv_cache: &mut StreamingKVCache,
    ) -> Array3<f32> {
        let residual = hidden_states.clone();
        let normed = self.input_layernorm.forward(hidden_states);
        let attn_output = self.self_attn.forward(&normed, cos, sin, kv_cache);
        let hidden_states = &residual + &attn_output;

        let residual = hidden_states.clone();
        let normed = self.post_attention_layernorm.forward(&hidden_states);
        let mlp_output = self.mlp.forward(&normed);
        &residual + &mlp_output
    }
}
