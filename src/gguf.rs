//! Minimal GGUF (v2/v3) reader: header, metadata KV table, tensor table, and
//! CPU dequantizers for the quant types used by Gemma-4 GGUFs
//! (Q4_0, Q4_1, Q5_1, Q8_0, Q4_K, Q5_K, Q6_K, plus F16/BF16/F32 passthrough).
//!
//! The Q4_0 block layout is byte-identical to `gpu::quantize_q4_0`, so quantized
//! weights round-trip losslessly through `dequant_to_f32` -> `buffer_from_f32_as_q4`.

use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use crate::gpu::{bf16_to_f32, f16_to_f32};

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" little-endian
const DEFAULT_ALIGNMENT: usize = 32;
const QK_K: usize = 256;

/// ggml tensor type ids (subset).
pub mod ggml_type {
    pub const F32: u32 = 0;
    pub const F16: u32 = 1;
    pub const Q4_0: u32 = 2;
    pub const Q4_1: u32 = 3;
    pub const Q5_1: u32 = 7;
    pub const Q8_0: u32 = 8;
    pub const Q2_K: u32 = 10;
    pub const Q4_K: u32 = 12;
    pub const Q5_K: u32 = 13;
    pub const Q6_K: u32 = 14;
    pub const IQ2_XXS: u32 = 16;
    pub const I32: u32 = 26;
    pub const BF16: u32 = 30;
}

pub fn ggml_type_name(t: u32) -> &'static str {
    match t {
        ggml_type::F32 => "F32",
        ggml_type::F16 => "F16",
        ggml_type::Q4_0 => "Q4_0",
        ggml_type::Q4_1 => "Q4_1",
        ggml_type::Q5_1 => "Q5_1",
        ggml_type::Q8_0 => "Q8_0",
        ggml_type::Q2_K => "Q2_K",
        ggml_type::Q4_K => "Q4_K",
        ggml_type::Q5_K => "Q5_K",
        ggml_type::Q6_K => "Q6_K",
        ggml_type::IQ2_XXS => "IQ2_XXS",
        ggml_type::I32 => "I32",
        ggml_type::BF16 => "BF16",
        _ => "UNKNOWN",
    }
}

/// (elements_per_block, bytes_per_block) for a ggml type.
pub(crate) fn block_spec(t: u32) -> (usize, usize) {
    match t {
        ggml_type::F32 => (1, 4),
        ggml_type::F16 => (1, 2),
        ggml_type::BF16 => (1, 2),
        ggml_type::I32 => (1, 4),
        ggml_type::Q4_0 => (32, 18),
        ggml_type::Q4_1 => (32, 20),
        ggml_type::Q5_1 => (32, 24),
        ggml_type::Q8_0 => (32, 34),
        ggml_type::Q2_K => (QK_K, 84),
        ggml_type::Q4_K => (QK_K, 144),
        ggml_type::Q5_K => (QK_K, 176),
        ggml_type::Q6_K => (QK_K, 210),
        ggml_type::IQ2_XXS => (QK_K, 66),
        _ => panic!("unsupported ggml type id {}", t),
    }
}

#[derive(Debug, Clone)]
pub enum MetaValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    Str(String),
    ArrU32(Vec<u32>),
    ArrI32(Vec<i32>),
    ArrF32(Vec<f32>),
    ArrBool(Vec<bool>),
    ArrStr(Vec<String>),
    /// Arrays of types we don't specifically need, kept as raw count for skipping.
    ArrOther(usize),
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub ggml_type: u32,
    pub dims: Vec<u64>,
    /// Offset relative to the start of the tensor data blob.
    pub offset: u64,
}

impl TensorInfo {
    pub fn num_elements(&self) -> usize {
        self.dims.iter().product::<u64>() as usize
    }
    /// ne0 (innermost / contiguous dim). For a [out,in] weight this is `in`.
    pub fn ne0(&self) -> usize {
        self.dims.first().copied().unwrap_or(1) as usize
    }
    /// Number of rows = product of all dims except ne0. For a [out,in] weight: `out`.
    pub fn n_rows(&self) -> usize {
        if self.dims.len() <= 1 {
            1
        } else {
            self.dims[1..].iter().product::<u64>() as usize
        }
    }
    pub fn byte_len(&self) -> usize {
        let (epb, bpb) = block_spec(self.ggml_type);
        let n = self.num_elements();
        assert!(n % epb == 0, "tensor {} elems {} not divisible by block {}", self.name, n, epb);
        (n / epb) * bpb
    }
}

pub struct Gguf {
    mmap: Mmap,
    pub version: u32,
    pub metadata: HashMap<String, MetaValue>,
    pub tensors: HashMap<String, TensorInfo>,
    data_offset: usize,
}

struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn u8(&mut self) -> u8 {
        let v = self.buf[self.pos];
        self.pos += 1;
        v
    }
    fn read(&mut self, n: usize) -> &'a [u8] {
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        s
    }
    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.read(4).try_into().unwrap())
    }
    fn i32(&mut self) -> i32 {
        i32::from_le_bytes(self.read(4).try_into().unwrap())
    }
    fn u64(&mut self) -> u64 {
        u64::from_le_bytes(self.read(8).try_into().unwrap())
    }
    fn i64(&mut self) -> i64 {
        i64::from_le_bytes(self.read(8).try_into().unwrap())
    }
    fn u16(&mut self) -> u16 {
        u16::from_le_bytes(self.read(2).try_into().unwrap())
    }
    fn i16(&mut self) -> i16 {
        i16::from_le_bytes(self.read(2).try_into().unwrap())
    }
    fn f32(&mut self) -> f32 {
        f32::from_le_bytes(self.read(4).try_into().unwrap())
    }
    fn f64(&mut self) -> f64 {
        f64::from_le_bytes(self.read(8).try_into().unwrap())
    }
    fn gstr(&mut self) -> String {
        let n = self.u64() as usize;
        String::from_utf8_lossy(self.read(n)).into_owned()
    }
    /// Read a single scalar metadata value of the given value-type id.
    fn scalar(&mut self, vt: u32) -> MetaValue {
        match vt {
            0 => MetaValue::U8(self.u8()),
            1 => MetaValue::I8(self.u8() as i8),
            2 => MetaValue::U16(self.u16()),
            3 => MetaValue::I16(self.i16()),
            4 => MetaValue::U32(self.u32()),
            5 => MetaValue::I32(self.i32()),
            6 => MetaValue::F32(self.f32()),
            7 => MetaValue::Bool(self.u8() != 0),
            8 => MetaValue::Str(self.gstr()),
            10 => MetaValue::U64(self.u64()),
            11 => MetaValue::I64(self.i64()),
            12 => MetaValue::F64(self.f64()),
            other => panic!("unexpected scalar value type {}", other),
        }
    }
    fn value(&mut self, vt: u32) -> MetaValue {
        if vt != 9 {
            return self.scalar(vt);
        }
        // ARRAY
        let at = self.u32();
        let n = self.u64() as usize;
        match at {
            4 => MetaValue::ArrU32((0..n).map(|_| self.u32()).collect()),
            5 => MetaValue::ArrI32((0..n).map(|_| self.i32()).collect()),
            6 => MetaValue::ArrF32((0..n).map(|_| self.f32()).collect()),
            7 => MetaValue::ArrBool((0..n).map(|_| self.u8() != 0).collect()),
            8 => MetaValue::ArrStr((0..n).map(|_| self.gstr()).collect()),
            other => {
                // Consume to stay aligned, but don't materialize.
                for _ in 0..n {
                    let _ = self.scalar(other);
                }
                MetaValue::ArrOther(n)
            }
        }
    }
}

impl Gguf {
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
                tensors.insert(
                    name.clone(),
                    TensorInfo { name, ggml_type, dims, offset },
                );
            }

            let alignment = match metadata.get("general.alignment") {
                Some(MetaValue::U32(a)) => *a as usize,
                Some(MetaValue::U64(a)) => *a as usize,
                _ => DEFAULT_ALIGNMENT,
            };
            let data_offset = align_up(c.pos, alignment);
            (version, metadata, tensors, data_offset)
        };

        Gguf { mmap, version, metadata, tensors, data_offset }
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    fn tensor_bytes(&self, info: &TensorInfo) -> &[u8] {
        let start = self.data_offset + info.offset as usize;
        &self.mmap[start..start + info.byte_len()]
    }

    /// Raw on-disk block bytes for a tensor (no dequant), for native GPU upload.
    pub fn tensor_raw(&self, name: &str) -> &[u8] {
        let info = self.tensor(name).unwrap_or_else(|| panic!("tensor not found: {}", name));
        self.tensor_bytes(info)
    }

    /// Raw byte slice for a single row (`row` of `n_rows`) of a tensor, for
    /// decoding one lookup-table row on demand (no full-tensor dequant).
    pub fn tensor_row_bytes(&self, name: &str, row: usize, row_stride: usize) -> &[u8] {
        let info = self.tensor(name).unwrap_or_else(|| panic!("tensor not found: {}", name));
        let start = self.data_offset + info.offset as usize + row * row_stride;
        &self.mmap[start..start + row_stride]
    }

    pub fn tensor_file_offset(&self, name: &str) -> u64 {
        let info = self
            .tensor(name)
            .unwrap_or_else(|| panic!("tensor not found: {}", name));
        (self.data_offset + info.offset as usize) as u64
    }

    /// ggml type id of a tensor (see `ggml_type`).
    pub fn tensor_type(&self, name: &str) -> u32 {
        self.tensor(name)
            .unwrap_or_else(|| panic!("tensor not found: {}", name))
            .ggml_type
    }

    // ---- metadata accessors ----

    pub fn get(&self, key: &str) -> Option<&MetaValue> {
        self.metadata.get(key)
    }

    pub fn get_u32(&self, key: &str) -> Option<u32> {
        match self.metadata.get(key)? {
            MetaValue::U32(v) => Some(*v),
            MetaValue::I32(v) => Some(*v as u32),
            MetaValue::U64(v) => Some(*v as u32),
            MetaValue::U16(v) => Some(*v as u32),
            _ => None,
        }
    }

    pub fn get_u64(&self, key: &str) -> Option<u64> {
        match self.metadata.get(key)? {
            MetaValue::U64(v) => Some(*v),
            MetaValue::U32(v) => Some(*v as u64),
            MetaValue::I32(v) => Some(*v as u64),
            _ => None,
        }
    }

    pub fn get_f32(&self, key: &str) -> Option<f32> {
        match self.metadata.get(key)? {
            MetaValue::F32(v) => Some(*v),
            MetaValue::F64(v) => Some(*v as f32),
            _ => None,
        }
    }

    pub fn get_bool(&self, key: &str) -> Option<bool> {
        match self.metadata.get(key)? {
            MetaValue::Bool(v) => Some(*v),
            _ => None,
        }
    }

    pub fn get_str(&self, key: &str) -> Option<&str> {
        match self.metadata.get(key)? {
            MetaValue::Str(v) => Some(v.as_str()),
            _ => None,
        }
    }

    pub fn get_arr_bool(&self, key: &str) -> Option<&[bool]> {
        match self.metadata.get(key)? {
            MetaValue::ArrBool(v) => Some(v.as_slice()),
            _ => None,
        }
    }

    pub fn get_arr_u32(&self, key: &str) -> Option<&[u32]> {
        match self.metadata.get(key)? {
            MetaValue::ArrU32(v) => Some(v.as_slice()),
            _ => None,
        }
    }

    pub fn get_arr_i32(&self, key: &str) -> Option<&[i32]> {
        match self.metadata.get(key)? {
            MetaValue::ArrI32(v) => Some(v.as_slice()),
            _ => None,
        }
    }

    /// `gemma4.feed_forward_length` is a scalar on E4B and a per-layer array on E2B
    /// (double-wide MLP on KV-shared layers).
    pub fn get_usize_list(&self, key: &str) -> Option<Vec<usize>> {
        match self.metadata.get(key)? {
            MetaValue::U32(v) => Some(vec![*v as usize]),
            MetaValue::I32(v) => Some(vec![*v as usize]),
            MetaValue::U64(v) => Some(vec![*v as usize]),
            MetaValue::ArrU32(v) => Some(v.iter().map(|&x| x as usize).collect()),
            MetaValue::ArrI32(v) => Some(v.iter().map(|&x| x as usize).collect()),
            _ => None,
        }
    }

    pub fn get_arr_str(&self, key: &str) -> Option<&[String]> {
        match self.metadata.get(key)? {
            MetaValue::ArrStr(v) => Some(v.as_slice()),
            _ => None,
        }
    }

    pub fn get_arr_f32(&self, key: &str) -> Option<&[f32]> {
        match self.metadata.get(key)? {
            MetaValue::ArrF32(v) => Some(v.as_slice()),
            _ => None,
        }
    }

    // ---- dequantization ----

    /// Dequantize an entire tensor to f32 in ggml-natural (row-major [..,ne1,ne0]) order.
    pub fn dequant_to_f32(&self, name: &str) -> Vec<f32> {
        let info = self.tensor(name).unwrap_or_else(|| panic!("tensor not found: {}", name));
        let bytes = self.tensor_bytes(info);
        let mut out = Vec::with_capacity(info.num_elements());
        dequant_blocks(info.ggml_type, bytes, info.num_elements(), |chunk| {
            out.extend_from_slice(chunk)
        });
        out
    }

    /// Dequantize an entire tensor directly to bf16 little-endian bytes, streaming
    /// block-by-block so the full f32 image is never materialized (used for the
    /// multi-GB per-layer embedding table).
    pub fn dequant_to_bf16_bytes(&self, name: &str) -> Vec<u8> {
        let info = self.tensor(name).unwrap_or_else(|| panic!("tensor not found: {}", name));
        let bytes = self.tensor_bytes(info);
        let mut out = Vec::with_capacity(info.num_elements() * 2);
        dequant_blocks(info.ggml_type, bytes, info.num_elements(), |chunk| {
            for &v in chunk {
                out.extend_from_slice(&f32_to_bf16(v).to_le_bytes());
            }
        });
        out
    }
}

fn align_up(x: usize, align: usize) -> usize {
    ((x + align - 1) / align) * align
}

/// (elements_per_block, bytes_per_block) for a ggml type — public for tooling.
pub fn type_block_spec(t: u32) -> (usize, usize) {
    block_spec(t)
}

/// Dequantize a raw block byte slice of the given ggml type to f32. Used by the
/// K-quant kernel self-test to produce a CPU reference from native blocks.
pub fn dequant_type_to_f32(ggml_type: u32, bytes: &[u8], total_elems: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(total_elems);
    dequant_blocks(ggml_type, bytes, total_elems, |chunk| out.extend_from_slice(chunk));
    out
}

/// Dequantize a single row (`elems` values) of a tensor stored in `ggml_type`
/// block layout into `out` (length `elems`). Used to decode one embedding/PLE
/// lookup-table row directly from native GGUF blocks (no full-tensor conversion).
pub fn dequant_row_to_f32(ggml_type: u32, row_bytes: &[u8], elems: usize, out: &mut [f32]) {
    assert!(out.len() >= elems, "dequant_row_to_f32: out too small");
    let mut n = 0;
    dequant_blocks(ggml_type, row_bytes, elems, |chunk| {
        out[n..n + chunk.len()].copy_from_slice(chunk);
        n += chunk.len();
    });
    assert_eq!(n, elems, "dequant_row_to_f32: row did not fill elems");
}

/// ggml token type ids (tokenizer.ggml.token_type).
mod token_type {
    pub const CONTROL: i32 = 3;
    pub const USER_DEFINED: i32 = 4;
}

/// Build a HuggingFace `tokenizers::Tokenizer` from the embedded GGUF tokenizer
/// metadata (BPE vocab + merges + special tokens + BOS post-processor).
pub fn build_tokenizer_from_gguf(path: &str) -> tokenizers::Tokenizer {
    use tokenizers::models::bpe::BPE;
    use tokenizers::pre_tokenizers::metaspace::{Metaspace, PrependScheme};
    use tokenizers::processors::template::TemplateProcessing;
    use tokenizers::{AddedToken, Tokenizer};

    let g = Gguf::open(path);

    let model = g.get_str("tokenizer.ggml.model").unwrap_or("");
    assert!(
        model == "gemma4"
            || model == "llama"
            || model == "gemma"
            || model == "gemma3"
            || model == "deepseek"
            || model == "deepseek4"
            || model == "gpt2"
            || model == "bpe",
        "unexpected tokenizer.ggml.model '{}'",
        model
    );

    let tokens = g
        .get_arr_str("tokenizer.ggml.tokens")
        .expect("tokenizer.ggml.tokens missing");
    let merges = g
        .get_arr_str("tokenizer.ggml.merges")
        .expect("tokenizer.ggml.merges missing");
    let token_types = g.get_arr_i32("tokenizer.ggml.token_type");

    // token -> id
    let mut vocab: ahash::AHashMap<String, u32> = ahash::AHashMap::with_capacity(tokens.len());
    for (i, t) in tokens.iter().enumerate() {
        vocab.insert(t.clone(), i as u32);
    }

    // "A B" -> (A, B); pieces never contain a raw space (SentencePiece uses U+2581).
    let merges_vec: Vec<(String, String)> = merges
        .iter()
        .filter_map(|m| m.split_once(' ').map(|(a, b)| (a.to_string(), b.to_string())))
        .collect();

    let is_gpt2_bpe = model == "gpt2" || model == "bpe" || model == "deepseek" || model == "deepseek4";

    let unk_token = g
        .get_u32("tokenizer.ggml.unknown_token_id")
        .and_then(|id| tokens.get(id as usize).cloned())
        .unwrap_or_else(|| {
            if is_gpt2_bpe {
                String::new()
            } else {
                "<unk>".to_string()
            }
        });

    let mut bpe_builder = BPE::builder().vocab_and_merges(vocab, merges_vec);
    if is_gpt2_bpe {
        // GPT-2 style: byte-level BPE, no UNK.
        if !unk_token.is_empty() {
            bpe_builder = bpe_builder.unk_token(unk_token);
        }
        bpe_builder = bpe_builder.byte_fallback(false).fuse_unk(false);
    } else {
        bpe_builder = bpe_builder
            .unk_token(unk_token)
            .byte_fallback(true)
            .fuse_unk(true);
    }
    let bpe = bpe_builder
        .build()
        .expect("failed to build BPE model from GGUF");

    let mut tok = Tokenizer::new(bpe);

    if is_gpt2_bpe {
        use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
        use tokenizers::pre_tokenizers::byte_level::ByteLevel;
        // add_prefix_space=false matches typical DeepSeek/GPT-2 GGUF exports.
        tok.with_pre_tokenizer(Some(ByteLevel::new(false, true, false)));
        tok.with_decoder(Some(ByteLevelDecoder::new(true, false, false)));
    } else {
        // SentencePiece-style whitespace handling: spaces <-> U+2581 ("▁").
        let prepend = if g.get_bool("tokenizer.ggml.add_space_prefix").unwrap_or(false) {
            PrependScheme::First
        } else {
            PrependScheme::Never
        };
        tok.with_pre_tokenizer(Some(Metaspace::new('\u{2581}', prepend, true)));
        tok.with_decoder(Some(Metaspace::new('\u{2581}', prepend, true)));
    }

    // Register control / user-defined tokens as special so they tokenize atomically.
    if let Some(types) = token_types {
        let specials: Vec<AddedToken> = tokens
            .iter()
            .enumerate()
            .filter(|(i, _)| {
                matches!(
                    types.get(*i).copied(),
                    Some(token_type::CONTROL) | Some(token_type::USER_DEFINED)
                )
            })
            .map(|(_, t)| AddedToken::from(t.clone(), true))
            .collect();
        tok.add_special_tokens(&specials);
    }

    // BOS post-processor (encode(text, true) prepends BOS) when requested.
    let add_bos = g.get_bool("tokenizer.ggml.add_bos_token").unwrap_or(!is_gpt2_bpe);
    if add_bos {
        if let Some(bos_id) = g.get_u32("tokenizer.ggml.bos_token_id") {
            if let Some(bos_tok) = tokens.get(bos_id as usize) {
                let post = TemplateProcessing::builder()
                    .try_single(format!("{} $A", bos_tok))
                    .unwrap()
                    .special_tokens(vec![(bos_tok.as_str(), bos_id)])
                    .build()
                    .expect("failed to build BOS template processor");
                tok.with_post_processor(Some(post));
            }
        }
    }

    tok
}

/// Convert f32 -> bf16 (round to nearest even on the truncated mantissa bit).
fn f32_to_bf16(v: f32) -> u16 {
    let bits = v.to_bits();
    let rounding_bias = 0x7FFF + ((bits >> 16) & 1);
    ((bits + rounding_bias) >> 16) as u16
}

/// 6-bit scale/min unpacking for Q4_K / Q5_K (matches ggml `get_scale_min_k4`).
#[inline]
fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        let d = q[j] & 63;
        let m = q[j + 4] & 63;
        (d, m)
    } else {
        let d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

/// Dequantize `total_elems` worth of `data` for the given ggml type, feeding the
/// output to `sink` in block-sized f32 chunks (in natural element order).
fn dequant_blocks(ggml_type: u32, data: &[u8], total_elems: usize, mut sink: impl FnMut(&[f32])) {
    match ggml_type {
        ggml_type::F32 => {
            let mut buf = [0.0f32; 256];
            let mut i = 0;
            while i < total_elems {
                let n = (total_elems - i).min(256);
                for k in 0..n {
                    let o = (i + k) * 4;
                    buf[k] = f32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]);
                }
                sink(&buf[..n]);
                i += n;
            }
        }
        ggml_type::F16 => {
            let mut buf = [0.0f32; 256];
            let mut i = 0;
            while i < total_elems {
                let n = (total_elems - i).min(256);
                for k in 0..n {
                    let o = (i + k) * 2;
                    buf[k] = f16_to_f32(u16::from_le_bytes([data[o], data[o + 1]]));
                }
                sink(&buf[..n]);
                i += n;
            }
        }
        ggml_type::BF16 => {
            let mut buf = [0.0f32; 256];
            let mut i = 0;
            while i < total_elems {
                let n = (total_elems - i).min(256);
                for k in 0..n {
                    let o = (i + k) * 2;
                    buf[k] = bf16_to_f32(u16::from_le_bytes([data[o], data[o + 1]]));
                }
                sink(&buf[..n]);
                i += n;
            }
        }
        ggml_type::Q4_0 => {
            let mut out = [0.0f32; 32];
            let nb = total_elems / 32;
            for b in 0..nb {
                let base = b * 18;
                let d = f16_to_f32(u16::from_le_bytes([data[base], data[base + 1]]));
                let qs = &data[base + 2..base + 18];
                for i in 0..16 {
                    let q_lo = (qs[i] & 0x0F) as i32 - 8;
                    let q_hi = (qs[i] >> 4) as i32 - 8;
                    out[i] = q_lo as f32 * d;
                    out[i + 16] = q_hi as f32 * d;
                }
                sink(&out);
            }
        }
        ggml_type::Q4_1 => {
            let mut out = [0.0f32; 32];
            let nb = total_elems / 32;
            for b in 0..nb {
                let base = b * 20;
                let d = f16_to_f32(u16::from_le_bytes([data[base], data[base + 1]]));
                let m = f16_to_f32(u16::from_le_bytes([data[base + 2], data[base + 3]]));
                let qs = &data[base + 4..base + 20];
                for i in 0..16 {
                    let q_lo = (qs[i] & 0x0F) as f32;
                    let q_hi = (qs[i] >> 4) as f32;
                    out[i] = q_lo * d + m;
                    out[i + 16] = q_hi * d + m;
                }
                sink(&out);
            }
        }
        ggml_type::Q5_1 => {
            // block_q5_1: half d, half m, qh[4], qs[16] = 24 bytes / 32 values
            let mut out = [0.0f32; 32];
            let nb = total_elems / 32;
            for b in 0..nb {
                let base = b * 24;
                let d = f16_to_f32(u16::from_le_bytes([data[base], data[base + 1]]));
                let m = f16_to_f32(u16::from_le_bytes([data[base + 2], data[base + 3]]));
                let qh = u32::from_le_bytes([
                    data[base + 4],
                    data[base + 5],
                    data[base + 6],
                    data[base + 7],
                ]);
                let qs = &data[base + 8..base + 24];
                for i in 0..16 {
                    let xh_0 = (((qh >> (i + 0)) << 4) & 0x10) as f32;
                    let xh_1 = (((qh >> (i + 12)) ) & 0x10) as f32;
                    let x0 = ((qs[i] & 0x0F) as f32) + xh_0;
                    let x1 = ((qs[i] >> 4) as f32) + xh_1;
                    out[i] = x0 * d + m;
                    out[i + 16] = x1 * d + m;
                }
                sink(&out);
            }
        }
        ggml_type::Q8_0 => {
            let mut out = [0.0f32; 32];
            let nb = total_elems / 32;
            for b in 0..nb {
                let base = b * 34;
                let d = f16_to_f32(u16::from_le_bytes([data[base], data[base + 1]]));
                let qs = &data[base + 2..base + 34];
                for i in 0..32 {
                    out[i] = (qs[i] as i8) as f32 * d;
                }
                sink(&out);
            }
        }
        ggml_type::Q4_K => {
            let nb = total_elems / QK_K;
            let mut out = [0.0f32; QK_K];
            for b in 0..nb {
                let base = b * 144;
                let d = f16_to_f32(u16::from_le_bytes([data[base], data[base + 1]]));
                let dmin = f16_to_f32(u16::from_le_bytes([data[base + 2], data[base + 3]]));
                let scales = &data[base + 4..base + 16];
                let qs = &data[base + 16..base + 144];
                let mut is = 0;
                let mut qoff = 0;
                let mut y = 0;
                while y < QK_K {
                    let (sc1, m1) = get_scale_min_k4(is, scales);
                    let (sc2, m2) = get_scale_min_k4(is + 1, scales);
                    let d1 = d * sc1 as f32;
                    let mn1 = dmin * m1 as f32;
                    let d2 = d * sc2 as f32;
                    let mn2 = dmin * m2 as f32;
                    for l in 0..32 {
                        out[y + l] = d1 * (qs[qoff + l] & 0x0F) as f32 - mn1;
                    }
                    for l in 0..32 {
                        out[y + 32 + l] = d2 * (qs[qoff + l] >> 4) as f32 - mn2;
                    }
                    qoff += 32;
                    is += 2;
                    y += 64;
                }
                sink(&out);
            }
        }
        ggml_type::Q5_K => {
            let nb = total_elems / QK_K;
            let mut out = [0.0f32; QK_K];
            for b in 0..nb {
                let base = b * 176;
                let d = f16_to_f32(u16::from_le_bytes([data[base], data[base + 1]]));
                let dmin = f16_to_f32(u16::from_le_bytes([data[base + 2], data[base + 3]]));
                let scales = &data[base + 4..base + 16];
                let qh = &data[base + 16..base + 48];
                let qs = &data[base + 48..base + 176];
                let mut is = 0;
                let mut qoff = 0;
                let mut y = 0;
                let mut u1: u8 = 1;
                let mut u2: u8 = 2;
                while y < QK_K {
                    let (sc1, m1) = get_scale_min_k4(is, scales);
                    let (sc2, m2) = get_scale_min_k4(is + 1, scales);
                    let d1 = d * sc1 as f32;
                    let mn1 = dmin * m1 as f32;
                    let d2 = d * sc2 as f32;
                    let mn2 = dmin * m2 as f32;
                    for l in 0..32 {
                        let hi = if qh[l] & u1 != 0 { 16.0 } else { 0.0 };
                        out[y + l] = d1 * ((qs[qoff + l] & 0x0F) as f32 + hi) - mn1;
                    }
                    for l in 0..32 {
                        let hi = if qh[l] & u2 != 0 { 16.0 } else { 0.0 };
                        out[y + 32 + l] = d2 * ((qs[qoff + l] >> 4) as f32 + hi) - mn2;
                    }
                    qoff += 32;
                    is += 2;
                    y += 64;
                    u1 = u1.wrapping_shl(2);
                    u2 = u2.wrapping_shl(2);
                }
                sink(&out);
            }
        }
        ggml_type::Q6_K => {
            // block_q6_K: ql[128], qh[64], scales[16] (i8), d (half). 210 bytes / 256.
            let nb = total_elems / QK_K;
            let mut out = [0.0f32; QK_K];
            for b in 0..nb {
                let base = b * 210;
                let ql = &data[base..base + 128];
                let qh = &data[base + 128..base + 192];
                let sc = &data[base + 192..base + 208];
                let d = f16_to_f32(u16::from_le_bytes([data[base + 208], data[base + 209]]));
                // Two 128-element halves; each consumes 64 ql, 32 qh, 8 scales.
                for half in 0..2 {
                    let ql = &ql[half * 64..];
                    let qh = &qh[half * 32..];
                    let sc = &sc[half * 8..];
                    let y = half * 128;
                    for l in 0..32 {
                        let is = l / 16;
                        let q1 = ((ql[l] & 0x0F) as i32 | (((qh[l] >> 0) & 3) as i32) << 4) - 32;
                        let q2 = ((ql[l + 32] & 0x0F) as i32 | (((qh[l] >> 2) & 3) as i32) << 4) - 32;
                        let q3 = ((ql[l] >> 4) as i32 | (((qh[l] >> 4) & 3) as i32) << 4) - 32;
                        let q4 = ((ql[l + 32] >> 4) as i32 | (((qh[l] >> 6) & 3) as i32) << 4) - 32;
                        out[y + l] = d * (sc[is] as i8) as f32 * q1 as f32;
                        out[y + l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
                        out[y + l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
                        out[y + l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
                    }
                }
                sink(&out);
            }
        }
        ggml_type::Q2_K => {
            let nb = total_elems / QK_K;
            let mut out = [0.0f32; QK_K];
            for b in 0..nb {
                let base = b * 84;
                dequant_q2_k_block_into(&data[base..base + 84], &mut out);
                sink(&out);
            }
        }
        ggml_type::IQ2_XXS => {
            let nb = total_elems / QK_K;
            let mut out = [0.0f32; QK_K];
            for b in 0..nb {
                let base = b * 66;
                dequant_iq2_xxs_block_into(&data[base..base + 66], &mut out);
                sink(&out);
            }
        }
        ggml_type::I32 => {
            let mut buf = [0.0f32; 256];
            let mut i = 0;
            while i < total_elems {
                let n = (total_elems - i).min(256);
                for k in 0..n {
                    let o = (i + k) * 4;
                    let v = i32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]);
                    buf[k] = v as f32;
                }
                sink(&buf[..n]);
                i += n;
            }
        }
        other => panic!("dequant: unsupported ggml type {}", other),
    }
}

fn dequant_q2_k_block_into(block: &[u8], out: &mut [f32]) {
    let scales = &block[0..16];
    let qs = &block[16..80];
    let d = f16_to_f32(u16::from_le_bytes([block[80], block[81]]));
    let dmin = f16_to_f32(u16::from_le_bytes([block[82], block[83]]));
    for idx in 0..QK_K {
        let group = idx / 16;
        let l = idx % 16;
        let q_base = 32 * (group / 8) + 16 * (group & 1);
        let shift = ((group / 2) & 3) * 2;
        let q = ((qs[q_base + l] as u32) >> shift) & 0x03;
        let sc = scales[group];
        out[idx] = d * ((sc & 0x0f) as f32) * (q as f32) - dmin * ((sc >> 4) as f32);
    }
}

// ggml IQ2_XXS format tables (public quant constants).
static IQ2XXS_GRID: [u64; 256] = [
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
];

static KSIGNS_IQ2XS: [u8; 128] = [
    0, 129, 130, 3, 132, 5, 6, 135, 136, 9, 10, 139, 12, 141, 142, 15, 144, 17, 18, 147, 20, 149,
    150, 23, 24, 153, 154, 27, 156, 29, 30, 159, 160, 33, 34, 163, 36, 165, 166, 39, 40, 169, 170,
    43, 172, 45, 46, 175, 48, 177, 178, 51, 180, 53, 54, 183, 184, 57, 58, 187, 60, 189, 190, 63,
    192, 65, 66, 195, 68, 197, 198, 71, 72, 201, 202, 75, 204, 77, 78, 207, 80, 209, 210, 83, 212,
    85, 86, 215, 216, 89, 90, 219, 92, 221, 222, 95, 96, 225, 226, 99, 228, 101, 102, 231, 232,
    105, 106, 235, 108, 237, 238, 111, 240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123,
    252, 125, 126, 255,
];

static KMASK_IQ2XS: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

fn dequant_iq2_xxs_block_into(block: &[u8], out: &mut [f32]) {
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let qs = &block[2..66];
    for ib32 in 0..8 {
        let q2 = &qs[ib32 * 8..ib32 * 8 + 8];
        let aux32_g = u32::from_le_bytes([q2[0], q2[1], q2[2], q2[3]]);
        let aux32_s = u32::from_le_bytes([q2[4], q2[5], q2[6], q2[7]]);
        let aux8 = aux32_g.to_le_bytes();
        let scale = 0.125 * d * (2.0 * ((aux32_s >> 28) as f32) + 1.0);
        let base = ib32 * 32;
        for l in 0..4 {
            let g = IQ2XXS_GRID[aux8[l] as usize].to_le_bytes();
            let signs = KSIGNS_IQ2XS[((aux32_s >> (7 * l)) & 127) as usize];
            for j in 0..8 {
                let sign = if signs & KMASK_IQ2XS[j] != 0 {
                    -1.0
                } else {
                    1.0
                };
                out[base + l * 8 + j] = scale * (g[j] as f32) * sign;
            }
        }
    }
}
