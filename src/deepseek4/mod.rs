//! DeepSeek-V4-Flash native engine (clean-room kernels; ds4 as study/oracle).

pub mod attn;
pub mod compressor;
pub mod config;
pub mod dense_matvec;
pub mod forward;
pub mod forward_metal;
pub mod gguf_validate;
pub mod hc;
pub mod kv;
pub mod metal_ctx;
pub mod model;
pub mod moe;
pub mod quant;
pub mod serve;
pub mod ssd;

pub use config::{dsv4_config_from_gguf, Dsv4Config};
pub use model::Dsv4GpuModel;

/// DeepSeek chat specials (antirez Flash GGUF vocab indices).
pub const DSV4_BOS_ID: u32 = 0;
pub const DSV4_EOS_ID: u32 = 1;
pub const DSV4_USER_ID: u32 = 128_803;
pub const DSV4_ASSISTANT_ID: u32 = 128_804;
pub const DSV4_THINK_START_ID: u32 = 128_821;
pub const DSV4_THINK_END_ID: u32 = 128_822;

/// Match ds4 `encode_chat_prompt` for a single user turn (Flash / nothink|think).
/// Pushes specials by id (not string encode) so markers stay atomic.
///
/// ds4 CLI default system is `"You are a helpful assistant"` (see `ds4_cli.c`);
/// pass `None` to omit system text (empty system in `encode_chat_prompt`).
pub fn encode_chat_user_ids(
    tokenizer: &tokenizers::Tokenizer,
    user: &str,
    nothink: bool,
    system: Option<&str>,
) -> Vec<usize> {
    let mut ids = Vec::new();
    ids.push(DSV4_BOS_ID as usize);
    if let Some(sys) = system {
        if !sys.is_empty() {
            let content = tokenizer
                .encode(sys, false)
                .expect("encode system content");
            ids.extend(content.get_ids().iter().map(|&t| t as usize));
        }
    }
    ids.push(DSV4_USER_ID as usize);
    let content = tokenizer
        .encode(user, false)
        .expect("encode user content");
    ids.extend(content.get_ids().iter().map(|&t| t as usize));
    ids.push(DSV4_ASSISTANT_ID as usize);
    ids.push(if nothink {
        DSV4_THINK_END_ID as usize
    } else {
        DSV4_THINK_START_ID as usize
    });
    ids
}

/// ds4 CLI default system string (Flash chat).
pub const DSV4_DEFAULT_SYSTEM: &str = "You are a helpful assistant";

/// String form for logging / HF template parity (specials must be AddedTokens).
pub fn encode_chat_user(user: &str, nothink: bool, system: Option<&str>) -> String {
    let think = if nothink { "</think>" } else { "<think>" };
    let mut s = String::from("<｜begin▁of▁sentence｜>");
    if let Some(sys) = system {
        if !sys.is_empty() {
            s.push_str(sys);
        }
    }
    s.push_str(&format!("<｜User｜>{user}<｜Assistant｜>{think}"));
    s
}
