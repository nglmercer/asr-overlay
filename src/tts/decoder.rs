use anyhow::{anyhow, Result};

/// Codec decoder session. Turns `codes[1,10,T]` into a mono f32 waveform at
/// 44.1 kHz.
///
/// NOT yet implemented — requires the ONNX session (FASE 11 of the roadmap).
#[allow(dead_code)]
pub struct CodecDecoder;

impl CodecDecoder {
    pub fn load(_model_path: &str) -> Result<Self> {
        Err(anyhow!("Codec decoder not yet implemented (FASE 11)"))
    }

    pub fn decode(&mut self, _codes: &[i64]) -> Result<Vec<f32>> {
        Err(anyhow!("Codec decoder not yet implemented (FASE 11)"))
    }
}
