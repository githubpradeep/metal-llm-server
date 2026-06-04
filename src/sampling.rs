use rand::Rng;
use std::collections::HashMap;

pub struct SamplingParams {
    pub temperature: f32,
    pub min_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub frequency_penalty: f32,
}

/// Sample with temperature and min-p filtering.
/// temperature: controls randomness (0.0 = greedy, 1.0 = neutral, >1.0 = more random)
/// min_p: minimum probability threshold relative to top token (0.05 = keep tokens with prob > 5% of max)
pub fn sample(logits: &[f32], temperature: f32, min_p: f32) -> usize {
    sample_with_params(
        logits,
        &SamplingParams {
            temperature,
            min_p,
            top_k: 0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
        },
        &[],
    )
}

pub fn sample_with_params(logits: &[f32], params: &SamplingParams, history: &[usize]) -> usize {
    let vocab_size = logits.len();
    let mut adjusted = logits.to_vec();

    apply_repetition_penalty(&mut adjusted, history, params.repetition_penalty);
    apply_frequency_penalty(&mut adjusted, history, params.frequency_penalty);

    if params.top_k > 0 {
        apply_top_k(&mut adjusted, params.top_k);
    }

    // Temperature 0 = greedy
    if params.temperature < 1e-6 {
        return argmax(&adjusted);
    }

    // Apply temperature and compute softmax with max subtraction for numerical stability
    let max_logit = adjusted.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let mut probs = vec![0.0f32; vocab_size];
    let mut sum = 0.0f32;
    for i in 0..vocab_size {
        let scaled = (adjusted[i] - max_logit) / params.temperature;
        probs[i] = scaled.exp();
        sum += probs[i];
    }
    let inv_sum = 1.0 / sum;
    for p in probs.iter_mut() {
        *p *= inv_sum;
    }

    // Min-p filtering: remove tokens with prob < min_p * max_prob
    let p_max = probs.iter().cloned().fold(0.0f32, f32::max);
    let threshold = p_max * params.min_p.max(0.0);

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
        argmax(&adjusted)
    }
}

fn apply_repetition_penalty(logits: &mut [f32], history: &[usize], penalty: f32) {
    if penalty <= 0.0 || (penalty - 1.0).abs() < f32::EPSILON {
        return;
    }

    for &token in history {
        if let Some(logit) = logits.get_mut(token) {
            if *logit > 0.0 {
                *logit /= penalty;
            } else {
                *logit *= penalty;
            }
        }
    }
}

fn apply_frequency_penalty(logits: &mut [f32], history: &[usize], penalty: f32) {
    if penalty.abs() < f32::EPSILON {
        return;
    }

    let mut counts = HashMap::new();
    for &token in history {
        *counts.entry(token).or_insert(0usize) += 1;
    }

    for (token, count) in counts {
        if let Some(logit) = logits.get_mut(token) {
            *logit -= penalty * count as f32;
        }
    }
}

fn apply_top_k(logits: &mut [f32], top_k: usize) {
    if top_k == 0 || top_k >= logits.len() {
        return;
    }

    let mut top = logits.to_vec();
    let (_, kth, _) = top.select_nth_unstable_by(top_k - 1, |a, b| {
        b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
    });
    let threshold = *kth;

    for logit in logits.iter_mut() {
        if *logit < threshold {
            *logit = f32::NEG_INFINITY;
        }
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
