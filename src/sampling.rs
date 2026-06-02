use rand::Rng;

/// Sample with temperature and min-p filtering.
/// temperature: controls randomness (0.0 = greedy, 1.0 = neutral, >1.0 = more random)
/// min_p: minimum probability threshold relative to top token (0.05 = keep tokens with prob > 5% of max)
pub fn sample(logits: &[f32], temperature: f32, min_p: f32) -> usize {
    let vocab_size = logits.len();

    // Temperature 0 = greedy
    if temperature < 1e-6 {
        return argmax(logits);
    }

    // Apply temperature and compute softmax with max subtraction for numerical stability
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let mut probs = vec![0.0f32; vocab_size];
    let mut sum = 0.0f32;
    for i in 0..vocab_size {
        let scaled = (logits[i] - max_logit) / temperature;
        probs[i] = scaled.exp();
        sum += probs[i];
    }
    let inv_sum = 1.0 / sum;
    for p in probs.iter_mut() {
        *p *= inv_sum;
    }

    // Min-p filtering: remove tokens with prob < min_p * max_prob
    let p_max = probs.iter().cloned().fold(0.0f32, f32::max);
    let threshold = p_max * min_p;

    let mut filtered_sum = 0.0f32;
    for p in probs.iter_mut() {
        if *p < threshold {
            *p = 0.0;
        } else {
            filtered_sum += *p;
        }
    }

    // Renormalize and sample
    if filtered_sum > 1e-9 {
        let inv_filtered = 1.0 / filtered_sum;
        for p in probs.iter_mut() {
            *p *= inv_filtered;
        }
        multinomial_sample(&probs)
    } else {
        argmax(logits)
    }
}

/// Legacy interface used by CLI
pub fn min_p_sampling(logits: &[f32], p_base: f32) -> usize {
    sample(logits, 0.7, p_base)
}

/// Sample from a probability distribution.
fn multinomial_sample(probs: &[f32]) -> usize {
    let mut rng = rand::thread_rng();
    let r: f32 = rng.gen();
    let mut cdf = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cdf += p;
        if r < cdf {
            return i;
        }
    }
    probs.len() - 1
}

/// Return index of maximum value.
pub fn argmax(values: &[f32]) -> usize {
    let mut max_idx = 0;
    let mut max_val = f32::NEG_INFINITY;
    for (i, &v) in values.iter().enumerate() {
        if v > max_val {
            max_val = v;
            max_idx = i;
        }
    }
    max_idx
}
