use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Sampling parameters mirrored from the Audio8 Python runtime defaults.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 50,
        }
    }
}

/// Deterministic sampler backed by ChaCha8. Ports the runtime's top-k +
/// top-p + temperature multinomial sampling.
pub struct Sampler {
    rng: ChaCha8Rng,
}

impl Sampler {
    /// Create a sampler with a fixed seed (Python default: 42).
    pub fn new(seed: u64) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    /// Sample an index from `logits` (assumed already filtered to the allowed
    /// token range by the caller).
    pub fn sample(&mut self, logits: &[f32], params: &SamplingParams) -> usize {
        sample_with(logits, params, &mut self.rng)
    }
}

/// Stateless sampling used by tests and the AR loops.
pub fn sample_with<R: Rng>(logits: &[f32], params: &SamplingParams, rng: &mut R) -> usize {
    if logits.is_empty() {
        return 0;
    }
    if logits.len() == 1 {
        return 0;
    }

    let temperature = params.temperature.max(1e-6);
    // Scaled logits -> probabilities (numerically stable softmax).
    let max = logits.iter().cloned().fold(f32::MIN, f32::max);
    let exp: Vec<f32> = logits
        .iter()
        .map(|&l| ((l - max) / temperature).exp())
        .collect();
    let sum: f32 = exp.iter().sum();
    let mut probs: Vec<f32> = exp.iter().map(|&e| e / sum).collect();

    // top-k: keep only the k largest probabilities.
    if params.top_k > 0 && params.top_k < probs.len() {
        let mut indices: Vec<usize> = (0..probs.len()).collect();
        indices.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal));
        let cutoff = probs[indices[params.top_k - 1]];
        for p in probs.iter_mut() {
            if *p < cutoff {
                *p = 0.0;
            }
        }
    }

    // top-p: keep the smallest set whose cumulative prob >= p.
    if params.top_p > 0.0 && params.top_p < 1.0 {
        let mut indices: Vec<usize> = (0..probs.len()).collect();
        indices.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal));
        let mut cum = 0.0;
        let mut keep = vec![false; probs.len()];
        for &i in &indices {
            cum += probs[i];
            keep[i] = true;
            if cum >= params.top_p {
                break;
            }
        }
        for (i, p) in probs.iter_mut().enumerate() {
            if !keep[i] {
                *p = 0.0;
            }
        }
    }

    // Renormalize and sample multinomially.
    let sum: f32 = probs.iter().sum();
    if sum <= 0.0 {
        // Degenerate: fall back to argmax.
        return logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
    }

    let mut roll = rng.random_range(0.0..sum);
    for (i, p) in probs.iter().enumerate() {
        roll -= *p;
        if roll <= 0.0 {
            return i;
        }
    }
    probs.len() - 1
}

/// Convenience wrapper that samples once from a fresh seeded RNG.
pub fn sample(logits: &[f32], params: &SamplingParams) -> usize {
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    sample_with(logits, params, &mut rng)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_with_fixed_seed() {
        let params = SamplingParams::default();
        let logits = [0.1, 0.9, 2.0, 0.5, 1.2];
        let mut a = Sampler::new(42);
        let mut b = Sampler::new(42);
        for _ in 0..10 {
            assert_eq!(a.sample(&logits, &params), b.sample(&logits, &params));
        }
    }

    #[test]
    fn different_seed_gives_different_streams() {
        let params = SamplingParams::default();
        let logits = [0.1, 0.9, 2.0, 0.5, 1.2, 3.0, 0.2];
        let mut a = Sampler::new(1);
        let mut b = Sampler::new(2);
        let sa: Vec<usize> = (0..50).map(|_| a.sample(&logits, &params)).collect();
        let sb: Vec<usize> = (0..50).map(|_| b.sample(&logits, &params)).collect();
        assert_ne!(sa, sb);
    }

    #[test]
    fn samples_within_range() {
        let params = SamplingParams::default();
        let logits: Vec<f32> = (0..100).map(|i| (i as f32 / 10.0).sin()).collect();
        let mut s = Sampler::new(7);
        for _ in 0..200 {
            let idx = s.sample(&logits, &params);
            assert!(idx < logits.len());
        }
    }

    #[test]
    fn single_token_returns_zero() {
        assert_eq!(sample(&[3.5], &SamplingParams::default()), 0);
    }

    #[test]
    fn empty_logits_returns_zero() {
        assert_eq!(sample(&[], &SamplingParams::default()), 0);
    }

    #[test]
    fn very_low_temperature_approximates_argmax() {
        let params = SamplingParams {
            temperature: 1e-6,
            top_p: 0.0,
            top_k: 0,
        };
        let logits = [0.1, 0.9, 5.0, 0.5, 1.2];
        assert_eq!(sample(&logits, &params), 2);
    }
}
