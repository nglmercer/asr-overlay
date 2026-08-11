use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

/// A registered speaker profile: the reference text plus its 10 codebooks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceProfile {
    pub reference_text: String,
    /// `[num_codebooks, T]` acoustic codes produced by the registration encoder.
    pub codes: Vec<Vec<i64>>,
}

/// Load a profile saved as `{meta.json, codes.bin}`.
pub fn load_profile(_dir: &str) -> Result<VoiceProfile> {
    Err(anyhow!(
        "Voice profile loading not yet implemented (FASE 16)"
    ))
}

/// Save a profile to `{dir}/meta.json` and `{dir}/codes.bin`.
pub fn save_profile(_dir: &str, _profile: &VoiceProfile) -> Result<()> {
    Err(anyhow!(
        "Voice profile saving not yet implemented (FASE 16)"
    ))
}
