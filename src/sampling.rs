use rand::Rng;

/// Min-p sampling: filter tokens below p_base * max_prob, then sample.
pub fn min_p_sampling(logits: &[f32], p_base: f32) -> usize {
    let vocab_size = logits.len();

    // Compute softmax (without max subtraction to match Python baseline)
    let mut probs = vec![0.0f32; vocab_size];
    let mut sum = 0.0f32;
    for i in 0..vocab_size {
        probs[i] = logits[i].exp();
        sum += probs[i];
    }
    if sum > 0.0 {
        for p in probs.iter_mut() {
            *p /= sum;
        }
    }

    // Find max probability
    let p_max = probs.iter().cloned().fold(0.0f32, f32::max);
    let p_scaled = p_max * p_base;

    // Zero out tokens below threshold
    let mut filtered_sum = 0.0f32;
    for p in probs.iter_mut() {
        if *p < p_scaled {
            *p = 0.0;
        } else {
            filtered_sum += *p;
        }
    }

    // Renormalize and sample
    if filtered_sum > 1e-9 {
        for p in probs.iter_mut() {
            *p /= filtered_sum;
        }
        multinomial_sample(&probs)
    } else {
        // Fallback to argmax
        argmax(logits)
    }
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
