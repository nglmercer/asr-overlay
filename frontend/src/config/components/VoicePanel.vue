<script setup lang="ts">
import type { AsrConfig } from '../../shared/types'

defineProps<{
  config: AsrConfig
  isArk: boolean
}>()
</script>

<template>
  <div class="panel" id="panel_vad">
    <h2>Voice detection</h2>
    <div class="row">
      <div class="field">
        <label for="vad_threshold">Threshold</label>
        <input
          type="number"
          id="vad_threshold"
          step="0.001"
          min="0"
          max="1"
          v-model.number="config.vad_threshold"
        />
      </div>
      <div class="field">
        <label for="vad_silence_ms">Silence (ms)</label>
        <input
          type="number"
          id="vad_silence_ms"
          step="50"
          min="100"
          v-model.number="config.vad_silence_ms"
        />
      </div>
    </div>
    <div class="row">
      <div class="field">
        <label for="vad_min_ms">Min (ms)</label>
        <input type="number" id="vad_min_ms" step="50" min="50" v-model.number="config.vad_min_ms" />
      </div>
      <div class="field">
        <label for="vad_max_ms">Max (ms)</label>
        <input
          type="number"
          id="vad_max_ms"
          step="500"
          min="1000"
          v-model.number="config.vad_max_ms"
        />
      </div>
    </div>
    <template v-if="isArk">
      <div class="row" id="field_min_audio_ms">
        <div class="field">
          <label for="min_audio_ms">Pad short audio to (ms)</label>
          <input
            type="number"
            id="min_audio_ms"
            step="500"
            min="0"
            v-model.number="config.min_audio_ms"
          />
        </div>
      </div>
      <p class="hint" id="min_audio_ms_hint">
        Short clips make the model fall back on its own priors and answer in the wrong language, so
        anything shorter than this is padded with silence. Higher is more accurate and slower — 8000
        is the measured sweet spot, 6000 already drifts. ARK-ASR only; other providers receive audio
        as segmented.
      </p>
    </template>
  </div>
</template>
