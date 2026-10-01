<script setup lang="ts">
import type { AsrConfig } from '../../shared/types'

defineProps<{
  config: AsrConfig
  isArk: boolean
  isHttp: boolean
  isStreaming: boolean
  isTranslate: boolean
  modelPlaceholder: string
}>()
</script>

<template>
  <div class="panel">
    <h2>Provider</h2>
    <div class="field">
      <label for="provider_kind">Transcription provider</label>
      <select id="provider_kind" v-model="config.provider.kind">
        <option value="phonon">Phonon-2 (local CPU server, English)</option>
        <option value="ark">Local ARK-ASR (on-device)</option>
        <option value="openai_compatible">OpenAI-compatible HTTP</option>
        <option value="voxtral_realtime">Mistral Voxtral realtime</option>
        <option value="gemini_live_transcribe">Gemini Live transcription</option>
        <option value="gemini_live_translate">Gemini Live translation</option>
      </select>
    </div>
    <p class="hint">
      Switching provider, model, base URL, or API-key variable restarts listening automatically.
    </p>
    <p class="hint" v-if="config.provider.kind === 'phonon'">
      Start the separate CPU server with <code>fermion serve phonon-2 --port 8000</code>.
      The default base URL is http://127.0.0.1:8000/v1. English only; no GPU required.
    </p>
    <div class="field" id="provider_base_url_field" v-if="isHttp">
      <label for="provider_base_url">Base URL</label>
      <input
        type="text"
        id="provider_base_url"
        placeholder="http://localhost:8000/v1"
        v-model.trim="config.provider.base_url"
      />
    </div>
    <div class="field" id="provider_model_field" v-if="!isArk">
      <label for="provider_model">Model</label>
      <input
        type="text"
        id="provider_model"
        :placeholder="modelPlaceholder"
        v-model.trim="config.provider.model"
      />
    </div>
    <div class="field" id="provider_api_key_env_field" v-if="!isArk">
      <label for="provider_api_key_env">API key environment variable</label>
      <input
        type="text"
        id="provider_api_key_env"
        placeholder="MISTRAL_API_KEY"
        spellcheck="false"
        v-model.trim="config.provider.api_key_env"
      />
    </div>
    <p class="hint" id="provider_api_key_env_hint" v-if="!isArk">
      Only the variable <em>name</em> is stored — never the key itself. Leave empty for servers
      without authentication.
    </p>
    <div class="field" id="provider_target_language_field" v-if="isTranslate">
      <label for="provider_target_language">Target language</label>
      <input
        type="text"
        id="provider_target_language"
        placeholder="es"
        v-model.trim="config.provider.target_language"
      />
    </div>
    <div class="field" id="provider_streaming_delay_field" v-if="isStreaming">
      <label for="provider_streaming_delay_ms">Streaming chunk (ms)</label>
      <input
        type="number"
        id="provider_streaming_delay_ms"
        min="20"
        max="1000"
        step="10"
        v-model.number="config.provider.streaming_delay_ms"
      />
    </div>
    <p class="hint" id="provider_streaming_delay_hint" v-if="isStreaming">
      How much audio is buffered into each chunk sent to realtime providers. Around 100 ms is what
      they expect.
    </p>
  </div>
</template>
