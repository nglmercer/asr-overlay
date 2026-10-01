// Mirrors the Rust serde shapes in src/config.rs and src/caption.rs.
// Field names stay snake_case so configs pass through untouched.

export interface OverlayStyle {
  font_family: string
  font_size: number
  font_weight: number
  color: string
  background: string
  padding: number
  radius: number
  text_shadow: string
  position: string
  align: string
  max_width: number
  margin: number
  uppercase: boolean
  fade_ms: number
  custom_css: string
}

export type ProviderKind =
  | 'phonon'
  | 'ark'
  | 'openai_compatible'
  | 'voxtral_realtime'
  | 'gemini_live_transcribe'
  | 'gemini_live_translate'

export interface ProviderConfig {
  kind: ProviderKind
  base_url: string
  model: string
  api_key_env: string
  target_language: string
  streaming_delay_ms: number
}

export interface AsrConfig {
  port: number
  language: string
  max_new_tokens: number
  auto_start: boolean
  vad_threshold: number
  vad_silence_ms: number
  vad_min_ms: number
  vad_max_ms: number
  min_audio_ms: number
  input_device: string
  threads: number
  style: OverlayStyle
  provider: ProviderConfig
}

export interface Caption {
  text: string
  start_ms: number
  end_ms: number
  partial?: boolean
  final_?: boolean
}

export type WsMessage =
  | ({ type: 'caption' } & Caption)
  | { type: 'style'; style: OverlayStyle }

export interface DeviceEntry {
  id: string
  label: string
  default: boolean
}

export interface StatusResponse {
  running: boolean
  listening: boolean
  has_caption: boolean
}

/** Same values as `OverlayStyle::default()` in Rust. */
export function defaultStyle(): OverlayStyle {
  return {
    font_family: '-apple-system, "Segoe UI", Roboto, sans-serif',
    font_size: 28,
    font_weight: 600,
    color: '#ffffff',
    background: 'transparent',
    padding: 8,
    radius: 8,
    text_shadow: '0 2px 6px rgba(0,0,0,0.9), 0 0 2px #000',
    position: 'bottom',
    align: 'center',
    max_width: 90,
    margin: 40,
    uppercase: false,
    fade_ms: 200,
    custom_css: '',
  }
}

export function defaultProvider(): ProviderConfig {
  return {
    kind: 'phonon',
    base_url: '',
    model: '',
    api_key_env: '',
    target_language: '',
    streaming_delay_ms: 100,
  }
}

/** Same values as `AsrConfig::default()` in Rust. */
export function defaultConfig(): AsrConfig {
  return {
    port: 8080,
    language: 'auto',
    max_new_tokens: 256,
    auto_start: true,
    vad_threshold: 0.005,
    vad_silence_ms: 600,
    vad_min_ms: 300,
    vad_max_ms: 30000,
    min_audio_ms: 8000,
    input_device: '',
    threads: 8,
    style: defaultStyle(),
    provider: defaultProvider(),
  }
}

/** Fill a freshly fetched config: old servers omit `provider` entirely. */
export function normalizeConfig(raw: Partial<AsrConfig>): AsrConfig {
  const fallback = defaultConfig()
  return {
    ...fallback,
    ...raw,
    style: { ...fallback.style, ...(raw.style ?? {}) },
    provider: { ...fallback.provider, ...(raw.provider ?? {}) },
  }
}
