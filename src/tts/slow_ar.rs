use anyhow::{anyhow, Result};

use super::manifest::Manifest;

/// Slow AR (semantic) session wrapper. Holds KV caches for all layers.
///
/// NOT yet implemented — requires the ONNX sessions (FASE 8 of the roadmap).
#[allow(dead_code)]
pub struct SlowAr;

impl SlowAr {
    pub fn load(_model_path: &str, _manifest: &Manifest) -> Result<Self> {
        Err(anyhow!("Slow AR not yet implemented (FASE 8)"))
    }

    pub fn step(
        &mut self,
        _codes: &[i64],
        _input_pos: &[i64],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        Err(anyhow!("Slow AR not yet implemented (FASE 8)"))
    }
}
