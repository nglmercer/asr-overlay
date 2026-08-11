use anyhow::{anyhow, Result};

use super::manifest::Manifest;

/// Fast AR (codebook) session wrapper.
///
/// NOT yet implemented — requires the ONNX sessions (FASE 9 of the roadmap).
#[allow(dead_code)]
pub struct FastAr;

impl FastAr {
    pub fn load(_model_path: &str, _manifest: &Manifest) -> Result<Self> {
        Err(anyhow!("Fast AR not yet implemented (FASE 9)"))
    }

    pub fn step(
        &mut self,
        _slow_hidden: &[f32],
        _token_id: i64,
        _use_slow_hidden: bool,
        _position: i64,
    ) -> Result<Vec<f32>> {
        Err(anyhow!("Fast AR not yet implemented (FASE 9)"))
    }
}
