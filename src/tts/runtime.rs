use anyhow::{anyhow, Result};

use super::sampler::SamplingParams;

/// Synthesized audio output.
#[derive(Debug, Clone)]
pub struct AudioOutput {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

/// High-level Audio8 TTS runtime: prompt → slow AR → fast AR → decoder.
///
/// NOT yet implemented — this is the orchestration layer that wires the AR
/// loops together (FASES 10-12 of the roadmap). It requires the ONNX models
/// and a registered voice profile.
#[allow(dead_code)]
pub struct Audio8 {
    model_dir: String,
    voice: String,
    manifest: super::Manifest,
}

impl Audio8 {
    pub fn load(model_dir: &str, voice: &str) -> Result<Self> {
        let manifest = super::Manifest::load(&format!("{model_dir}/runtime_manifest.json"))?;
        Ok(Self {
            model_dir: model_dir.to_string(),
            voice: voice.to_string(),
            manifest,
        })
    }

    pub fn synthesize(
        &self,
        _text: &str,
        _params: &SamplingParams,
    ) -> Result<AudioOutput> {
        Err(anyhow!(
            "Audio8 synthesis not yet implemented (FASES 10-12); model dir: {}",
            self.model_dir
        ))
    }
}
