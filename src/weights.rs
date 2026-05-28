use std::collections::HashMap;
use std::fs;
use std::path::Path;

use ndarray::Array2;
use safetensors::SafeTensors;
use serde::Deserialize;

use crate::config::LlamaConfig;

#[derive(Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

/// Stores all model weights as f32 2D arrays (or 1D stored as 2D with dim 1).
pub struct ModelWeights {
    pub tensors: HashMap<String, Vec<f32>>,
    pub shapes: HashMap<String, Vec<usize>>,
}

impl ModelWeights {
    pub fn load(model_dir: &str) -> (Self, LlamaConfig) {
        let config_path = Path::new(model_dir).join("config.json");
        let config_str = fs::read_to_string(&config_path)
            .expect("Failed to read config.json");
        let config: LlamaConfig = serde_json::from_str(&config_str)
            .expect("Failed to parse config.json");

        let mut tensors: HashMap<String, Vec<f32>> = HashMap::new();
        let mut shapes: HashMap<String, Vec<usize>> = HashMap::new();

        // Try sharded loading first
        let index_path = Path::new(model_dir).join("model.safetensors.index.json");
        let shard_files: Vec<String> = if index_path.exists() {
            let index_str = fs::read_to_string(&index_path).unwrap();
            let index: SafetensorsIndex = serde_json::from_str(&index_str).unwrap();
            let mut files: Vec<String> = index.weight_map.values().cloned().collect();
            files.sort();
            files.dedup();
            files
        } else {
            vec!["model.safetensors".to_string()]
        };

        for shard_file in &shard_files {
            let shard_path = Path::new(model_dir).join(shard_file);
            let data = fs::read(&shard_path)
                .unwrap_or_else(|_| panic!("Failed to read shard: {}", shard_file));
            let safetensors = SafeTensors::deserialize(&data)
                .expect("Failed to deserialize safetensors");

            for (name, tensor_view) in safetensors.tensors() {
                let shape: Vec<usize> = tensor_view.shape().to_vec();
                let dtype = tensor_view.dtype();
                let raw_data = tensor_view.data();

                let float_data: Vec<f32> = match dtype {
                    safetensors::Dtype::F32 => {
                        raw_data
                            .chunks_exact(4)
                            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                            .collect()
                    }
                    safetensors::Dtype::F16 => {
                        raw_data
                            .chunks_exact(2)
                            .map(|b| {
                                let bits = u16::from_le_bytes([b[0], b[1]]);
                                half_to_f32(bits)
                            })
                            .collect()
                    }
                    safetensors::Dtype::BF16 => {
                        raw_data
                            .chunks_exact(2)
                            .map(|b| {
                                let bits = u16::from_le_bytes([b[0], b[1]]);
                                bf16_to_f32(bits)
                            })
                            .collect()
                    }
                    _ => panic!("Unsupported dtype: {:?}", dtype),
                };

                shapes.insert(name.to_string(), shape);
                tensors.insert(name.to_string(), float_data);
            }
        }

        println!("All shards loaded and tensors extracted successfully.");
        (ModelWeights { tensors, shapes }, config)
    }

    /// Get a weight as a 2D array (rows x cols).
    pub fn get_2d(&self, key: &str) -> Array2<f32> {
        let actual_key = if key == "lm_head.weight" && !self.tensors.contains_key(key) {
            "model.embed_tokens.weight"
        } else {
            key
        };
        let data = self.tensors.get(actual_key)
            .unwrap_or_else(|| panic!("Weight not found: {}", actual_key));
        let shape = self.shapes.get(actual_key).unwrap();
        assert_eq!(shape.len(), 2, "Expected 2D tensor for key: {}", actual_key);
        Array2::from_shape_vec((shape[0], shape[1]), data.clone())
            .expect("Shape mismatch")
    }

    /// Get a weight as a 1D vector.
    pub fn get_1d(&self, key: &str) -> Vec<f32> {
        let data = self.tensors.get(key)
            .unwrap_or_else(|| panic!("Weight not found: {}", key));
        data.clone()
    }

    /// Get raw weight data (handles lm_head fallback).
    pub fn get_1d_raw(&self, key: &str) -> Vec<f32> {
        let actual_key = if key == "lm_head.weight" && !self.tensors.contains_key(key) {
            "model.embed_tokens.weight"
        } else {
            key
        };
        let data = self.tensors.get(actual_key)
            .unwrap_or_else(|| panic!("Weight not found: {}", actual_key));
        data.clone()
    }
}

/// Convert f16 bits to f32.
fn half_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign << 31);
        }
        // Subnormal
        let mut e = 1u32;
        let mut m = mant;
        while (m & 0x400) == 0 {
            m <<= 1;
            e += 1;
        }
        let f_exp = (127 - 15 + 1 - e) as u32;
        let f_mant = (m & 0x3FF) << 13;
        f32::from_bits((sign << 31) | (f_exp << 23) | f_mant)
    } else if exp == 31 {
        let f_exp = 0xFFu32;
        let f_mant = mant as u32;
        f32::from_bits((sign << 31) | (f_exp << 23) | (f_mant << 13))
    } else {
        let f_exp = (exp as i32 - 15 + 127) as u32;
        let f_mant = mant << 13;
        f32::from_bits((sign << 31) | (f_exp << 23) | f_mant)
    }
}

/// Convert bf16 bits to f32.
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}
