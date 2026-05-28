/// Flat KV cache stored as contiguous Vec<f32> for zero-copy slice access.
/// Layout: (batch=1, num_kv_heads, seq, head_dim) stored row-major.
/// This avoids ndarray overhead and allows direct pointer arithmetic in attention.

pub struct StreamingKVCache {
    /// Flat storage per layer: Vec<f32> of shape (num_kv_heads * seq * head_dim)
    pub key_cache: Vec<Option<Vec<f32>>>,
    pub value_cache: Vec<Option<Vec<f32>>>,
    /// Current sequence length in cache per layer
    pub seq_lens: Vec<usize>,
    /// Capacity (max seq before realloc) per layer
    pub capacities: Vec<usize>,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub sink_size: usize,
    pub window_size: usize,
    pub total_tokens: usize,
}

impl StreamingKVCache {
    pub fn new(sink_size: usize, window_size: usize) -> Self {
        StreamingKVCache {
            key_cache: Vec::new(),
            value_cache: Vec::new(),
            seq_lens: Vec::new(),
            capacities: Vec::new(),
            num_kv_heads: 0,
            head_dim: 0,
            sink_size,
            window_size: window_size.max(1),
            total_tokens: 0,
        }
    }

    pub fn num_items(&self) -> usize {
        self.total_tokens
    }

    pub fn cached_length(&self) -> usize {
        if self.seq_lens.is_empty() {
            0
        } else {
            self.seq_lens[0]
        }
    }

    fn ensure_size(&mut self, layer_idx: usize) {
        while self.key_cache.len() <= layer_idx {
            self.key_cache.push(None);
            self.value_cache.push(None);
            self.seq_lens.push(0);
            self.capacities.push(0);
        }
    }

    /// Initialize dimensions on first call.
    fn init_dims(&mut self, num_kv_heads: usize, head_dim: usize) {
        if self.num_kv_heads == 0 {
            self.num_kv_heads = num_kv_heads;
            self.head_dim = head_dim;
        }
    }

    /// Append new keys/values. `new_keys` is a flat slice of shape (num_kv_heads, new_seq, head_dim).
    /// Returns (key_slice, value_slice, current_seq_len) for this layer.
    pub fn update(
        &mut self,
        new_keys: &[f32],
        new_values: &[f32],
        new_seq: usize,
        num_kv_heads: usize,
        head_dim: usize,
        layer_idx: usize,
    ) -> usize {
        self.init_dims(num_kv_heads, head_dim);
        self.ensure_size(layer_idx);

        let head_new_stride = new_seq * head_dim;
        let keep_total = self.sink_size + self.window_size;

        if self.key_cache[layer_idx].is_none() {
            // First time: allocate with capacity for keep_total
            let cap = keep_total.max(new_seq);
            let buf_size = num_kv_heads * cap * head_dim;
            let mut k_buf = vec![0.0f32; buf_size];
            let mut v_buf = vec![0.0f32; buf_size];

            // Copy new data in
            for h in 0..num_kv_heads {
                let src_offset = h * head_new_stride;
                let dst_offset = h * cap * head_dim;
                k_buf[dst_offset..dst_offset + head_new_stride]
                    .copy_from_slice(&new_keys[src_offset..src_offset + head_new_stride]);
                v_buf[dst_offset..dst_offset + head_new_stride]
                    .copy_from_slice(&new_values[src_offset..src_offset + head_new_stride]);
            }

            self.key_cache[layer_idx] = Some(k_buf);
            self.value_cache[layer_idx] = Some(v_buf);
            self.seq_lens[layer_idx] = new_seq;
            self.capacities[layer_idx] = cap;
        } else {
            let cur_seq = self.seq_lens[layer_idx];
            let new_total = cur_seq + new_seq;
            let cap = self.capacities[layer_idx];

            if new_total <= cap {
                // Append in-place
                let k_buf = self.key_cache[layer_idx].as_mut().unwrap();
                let v_buf = self.value_cache[layer_idx].as_mut().unwrap();

                for h in 0..num_kv_heads {
                    let src_offset = h * head_new_stride;
                    let dst_offset = h * cap * head_dim + cur_seq * head_dim;
                    k_buf[dst_offset..dst_offset + head_new_stride]
                        .copy_from_slice(&new_keys[src_offset..src_offset + head_new_stride]);
                    v_buf[dst_offset..dst_offset + head_new_stride]
                        .copy_from_slice(&new_values[src_offset..src_offset + head_new_stride]);
                }
                self.seq_lens[layer_idx] = new_total;
            } else {
                // Need to grow or evict
                let new_cap = (new_total * 2).max(keep_total);
                let buf_size = num_kv_heads * new_cap * head_dim;
                let mut new_k_buf = vec![0.0f32; buf_size];
                let mut new_v_buf = vec![0.0f32; buf_size];

                let k_buf = self.key_cache[layer_idx].as_ref().unwrap();
                let v_buf = self.value_cache[layer_idx].as_ref().unwrap();

                // Copy existing data
                for h in 0..num_kv_heads {
                    let old_offset = h * cap * head_dim;
                    let new_offset = h * new_cap * head_dim;
                    new_k_buf[new_offset..new_offset + cur_seq * head_dim]
                        .copy_from_slice(&k_buf[old_offset..old_offset + cur_seq * head_dim]);
                    new_v_buf[new_offset..new_offset + cur_seq * head_dim]
                        .copy_from_slice(&v_buf[old_offset..old_offset + cur_seq * head_dim]);

                    // Append new
                    let src_offset = h * head_new_stride;
                    let dst_offset = new_offset + cur_seq * head_dim;
                    new_k_buf[dst_offset..dst_offset + head_new_stride]
                        .copy_from_slice(&new_keys[src_offset..src_offset + head_new_stride]);
                    new_v_buf[dst_offset..dst_offset + head_new_stride]
                        .copy_from_slice(&new_values[src_offset..src_offset + head_new_stride]);
                }

                self.key_cache[layer_idx] = Some(new_k_buf);
                self.value_cache[layer_idx] = Some(new_v_buf);
                self.seq_lens[layer_idx] = new_total;
                self.capacities[layer_idx] = new_cap;
            }
        }

        // Track absolute token count from layer 0
        if layer_idx == 0 {
            self.total_tokens += new_seq;
        }

        // Evict middle tokens if cache exceeds budget
        let cur_seq = self.seq_lens[layer_idx];
        if cur_seq > keep_total {
            self.evict(layer_idx, num_kv_heads, head_dim);
        }

        self.seq_lens[layer_idx]
    }

    /// Evict middle tokens: keep first sink_size + last window_size.
    fn evict(&mut self, layer_idx: usize, num_kv_heads: usize, head_dim: usize) {
        let cur_seq = self.seq_lens[layer_idx];
        let cap = self.capacities[layer_idx];
        let keep_total = self.sink_size + self.window_size;
        let tail_start = cur_seq - self.window_size;

        let k_buf = self.key_cache[layer_idx].as_mut().unwrap();
        let v_buf = self.value_cache[layer_idx].as_mut().unwrap();

        // Move tail tokens right after sink tokens (in-place per head)
        for h in 0..num_kv_heads {
            let base = h * cap * head_dim;
            let sink_end = base + self.sink_size * head_dim;
            let tail_src = base + tail_start * head_dim;
            let tail_len = self.window_size * head_dim;

            // Copy tail to right after sink (may overlap if window is large, use copy_within)
            k_buf.copy_within(tail_src..tail_src + tail_len, sink_end);
            v_buf.copy_within(tail_src..tail_src + tail_len, sink_end);
        }

        self.seq_lens[layer_idx] = keep_total;
    }

    /// Get raw key slice for a layer. Layout: head h starts at h * capacity * head_dim.
    /// Actual data for head h is [h*cap*hd .. h*cap*hd + seq_len*hd].
    #[inline]
    pub fn get_key_slice(&self, layer_idx: usize) -> (&[f32], usize, usize) {
        let buf = self.key_cache[layer_idx].as_ref().unwrap();
        let seq = self.seq_lens[layer_idx];
        let cap = self.capacities[layer_idx];
        (buf.as_slice(), seq, cap)
    }

    #[inline]
    pub fn get_value_slice(&self, layer_idx: usize) -> (&[f32], usize, usize) {
        let buf = self.value_cache[layer_idx].as_ref().unwrap();
        let seq = self.seq_lens[layer_idx];
        let cap = self.capacities[layer_idx];
        (buf.as_slice(), seq, cap)
    }
}
