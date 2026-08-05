//! Thin dispatch over Gemma4 vs DeepSeek-V4 GPU models.

use crate::deepseek4::Dsv4GpuModel;
use crate::gemma4_gpu_model::Gemma4GpuModel;
use crate::gguf::Gguf;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelArch {
    Gemma4,
    DeepSeek4,
}

pub fn detect_arch(path: &str) -> ModelArch {
    if path.ends_with(".gguf") {
        let g = Gguf::open(path);
        match g.get_str("general.architecture").unwrap_or("") {
            "deepseek4" => ModelArch::DeepSeek4,
            "gemma4" => ModelArch::Gemma4,
            other => panic!("unsupported GGUF architecture {other:?}"),
        }
    } else {
        ModelArch::Gemma4
    }
}

pub enum GpuModel {
    Gemma4(Gemma4GpuModel),
    DeepSeek4(Dsv4GpuModel),
}

impl GpuModel {
    pub fn load_gguf(path: &str, ssd_streaming: bool) -> Self {
        match detect_arch(path) {
            ModelArch::Gemma4 => {
                GpuModel::Gemma4(Gemma4GpuModel::load_from_gguf(path))
            }
            ModelArch::DeepSeek4 => {
                GpuModel::DeepSeek4(Dsv4GpuModel::load_from_gguf(path, ssd_streaming, None))
            }
        }
    }

    pub fn arch(&self) -> ModelArch {
        match self {
            GpuModel::Gemma4(_) => ModelArch::Gemma4,
            GpuModel::DeepSeek4(_) => ModelArch::DeepSeek4,
        }
    }
}
