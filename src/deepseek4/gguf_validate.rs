//! Validate DeepSeek-V4-Flash GGUF tensor presence and types (antirez/ds4 layout).

use crate::gguf::{ggml_type, Gguf};

use super::config::Dsv4Config;

fn require_tensor(g: &Gguf, name: &str) {
    assert!(
        g.has_tensor(name),
        "deepseek4 GGUF missing required tensor: {name}"
    );
}

fn require_type(g: &Gguf, name: &str, allowed: &[u32]) {
    require_tensor(g, name);
    let t = g.tensor_type(name);
    assert!(
        allowed.contains(&t),
        "tensor {name} has type {}, expected one of {:?}",
        crate::gguf::ggml_type_name(t),
        allowed
            .iter()
            .map(|&x| crate::gguf::ggml_type_name(x))
            .collect::<Vec<_>>()
    );
}

/// Layer tensor names for Flash. `ratio` and `hash` select optional tensors.
pub fn layer_tensor_names(il: usize, ratio: u32, is_hash: bool) -> Vec<String> {
    let mut names = vec![
        format!("blk.{il}.hc_attn_fn.weight"),
        format!("blk.{il}.hc_attn_scale.weight"),
        format!("blk.{il}.hc_attn_base.weight"),
        format!("blk.{il}.attn_norm.weight"),
        format!("blk.{il}.attn_q_a.weight"),
        format!("blk.{il}.attn_q_a_norm.weight"),
        format!("blk.{il}.attn_q_b.weight"),
        format!("blk.{il}.attn_kv.weight"),
        format!("blk.{il}.attn_kv_a_norm.weight"),
        format!("blk.{il}.attn_sinks.weight"),
        format!("blk.{il}.attn_output_a.weight"),
        format!("blk.{il}.attn_output_b.weight"),
        format!("blk.{il}.hc_ffn_fn.weight"),
        format!("blk.{il}.hc_ffn_scale.weight"),
        format!("blk.{il}.hc_ffn_base.weight"),
        format!("blk.{il}.ffn_norm.weight"),
        format!("blk.{il}.ffn_gate_inp.weight"),
        format!("blk.{il}.ffn_gate_exps.weight"),
        format!("blk.{il}.ffn_up_exps.weight"),
        format!("blk.{il}.ffn_down_exps.weight"),
        format!("blk.{il}.ffn_gate_shexp.weight"),
        format!("blk.{il}.ffn_up_shexp.weight"),
        format!("blk.{il}.ffn_down_shexp.weight"),
    ];
    if ratio != 0 {
        names.extend([
            format!("blk.{il}.attn_compressor_ape.weight"),
            format!("blk.{il}.attn_compressor_kv.weight"),
            format!("blk.{il}.attn_compressor_gate.weight"),
            format!("blk.{il}.attn_compressor_norm.weight"),
        ]);
    }
    if ratio == 4 {
        names.extend([
            format!("blk.{il}.indexer.attn_q_b.weight"),
            format!("blk.{il}.indexer.proj.weight"),
            format!("blk.{il}.indexer_compressor_ape.weight"),
            format!("blk.{il}.indexer_compressor_kv.weight"),
            format!("blk.{il}.indexer_compressor_gate.weight"),
            format!("blk.{il}.indexer_compressor_norm.weight"),
        ]);
    }
    if is_hash {
        names.push(format!("blk.{il}.ffn_gate_tid2eid.weight"));
    }
    // Optional expert bias.
    // names.push(format!("blk.{il}.exp_probs_b.bias"));
    names
}

pub fn validate_dsv4_gguf(g: &Gguf, cfg: &Dsv4Config) -> Result<(), String> {
    let arch = g
        .get_str("general.architecture")
        .ok_or_else(|| "missing general.architecture".to_string())?;
    if arch != "deepseek4" {
        return Err(format!("unsupported architecture {arch:?}; need deepseek4"));
    }

    require_tensor(g, "token_embd.weight");
    require_tensor(g, "output_norm.weight");
    require_tensor(g, "output.weight");
    require_tensor(g, "output_hc_base.weight");
    require_tensor(g, "output_hc_fn.weight");
    require_tensor(g, "output_hc_scale.weight");

    for il in 0..cfg.n_layer {
        let ratio = cfg.compress_ratio(il);
        let is_hash = cfg.is_hash_layer(il);
        for name in layer_tensor_names(il, ratio, is_hash) {
            if !g.has_tensor(&name) {
                // Some GGUFs use `.bias` for exp_probs; skip soft-required.
                if name.contains("exp_probs") {
                    continue;
                }
                return Err(format!("missing tensor {name}"));
            }
        }
        // Expert quants: gate/up IQ2_XXS, down Q2_K (asymmetric Flash IQ2).
        require_type(
            g,
            &format!("blk.{il}.ffn_gate_exps.weight"),
            &[ggml_type::IQ2_XXS, ggml_type::Q4_K, ggml_type::Q8_0],
        );
        require_type(
            g,
            &format!("blk.{il}.ffn_up_exps.weight"),
            &[ggml_type::IQ2_XXS, ggml_type::Q4_K, ggml_type::Q8_0],
        );
        require_type(
            g,
            &format!("blk.{il}.ffn_down_exps.weight"),
            &[ggml_type::Q2_K, ggml_type::Q4_K, ggml_type::Q8_0],
        );
        require_type(
            g,
            &format!("blk.{il}.ffn_gate_shexp.weight"),
            &[ggml_type::Q8_0, ggml_type::Q4_K],
        );
        if is_hash {
            require_type(
                g,
                &format!("blk.{il}.ffn_gate_tid2eid.weight"),
                &[ggml_type::I32],
            );
        }
    }

    Ok(())
}

pub fn print_inspect(g: &Gguf, cfg: &Dsv4Config) {
    let name = g.get_str("general.name").unwrap_or("(unnamed)");
    println!("model: {name}");
    println!("arch:  deepseek4 (Flash)");
    println!(
        "layers: {}  embd: {}  vocab: {}",
        cfg.n_layer, cfg.n_embd, cfg.n_vocab
    );
    println!(
        "attention: heads={} kv_heads={} head_dim={} swa={}",
        cfg.n_head, cfg.n_head_kv, cfg.head_dim, cfg.n_swa
    );
    println!(
        "indexer: heads={} head_dim={} top_k={}",
        cfg.n_indexer_head, cfg.n_indexer_head_dim, cfg.n_indexer_top_k
    );
    println!(
        "experts: count={} used={} shared={} ff={} hash_layers={}",
        cfg.n_expert, cfg.n_expert_used, cfg.n_expert_shared, cfg.n_ff_exp, cfg.n_hash_layer
    );
    println!(
        "mHC: streams={} sinkhorn_iters={}",
        cfg.n_hc, cfg.n_hc_sinkhorn_iter
    );
    println!("train context: {}", cfg.context_length);
    println!("compress_ratios: {:?}", cfg.compress_ratios);
    let mut type_bytes: std::collections::BTreeMap<u32, (usize, u64)> =
        std::collections::BTreeMap::new();
    for info in g.tensors.values() {
        let e = type_bytes.entry(info.ggml_type).or_insert((0, 0));
        e.0 += 1;
        e.1 += info.byte_len() as u64;
    }
    println!("tensor types:");
    for (t, (count, bytes)) in type_bytes {
        println!(
            "  {:8} {:5} tensors, {:.2} GiB",
            crate::gguf::ggml_type_name(t),
            count,
            bytes as f64 / (1024.0 * 1024.0 * 1024.0)
        );
    }
    for name in [
        "blk.0.hc_attn_fn.weight",
        "blk.0.hc_attn_scale.weight",
        "blk.0.hc_attn_base.weight",
        "blk.0.attn_kv.weight",
        "blk.0.attn_output_a.weight",
        "blk.0.attn_output_b.weight",
        "output_hc_fn.weight",
        "output_hc_scale.weight",
        "output_hc_base.weight",
    ] {
        if let Some(info) = g.tensor(name) {
            println!(
                "  dim {name}: {:?} type={}",
                info.dims,
                crate::gguf::ggml_type_name(info.ggml_type)
            );
        }
    }
}
