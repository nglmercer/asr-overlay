<script setup lang="ts">
import type { AsrConfig, DeviceEntry } from '../../shared/types'
import { LANGUAGES } from '../composables/useSettings'

defineProps<{
  config: AsrConfig
  devices: DeviceEntry[]
  isArk: boolean
  languageHint: string
}>()

function deviceLabel(device: DeviceEntry): string {
  return device.default ? `${device.label} (default)` : device.label
}

function savedLabel(inputDevice: string): string {
  const saved = (inputDevice || '').replace(/^pw:/, '')
  return saved ? `${saved} (unavailable)` : 'System default'
}
</script>

<template>
  <div class="panel">
    <h2>Recognition</h2>
    <div class="field">
      <label for="input_device">Microphone</label>
      <select id="input_device" v-model="config.input_device">
        <option value="">System default</option>
        <option
          v-if="config.input_device && !devices.some((d) => d.id === config.input_device)"
          :value="config.input_device"
        >
          {{ savedLabel(config.input_device) }}
        </option>
        <option v-for="device in devices" :key="device.id" :value="device.id">
          {{ deviceLabel(device) }}
        </option>
      </select>
    </div>
    <p class="hint">Changing the microphone restarts capture automatically.</p>
    <div class="field">
      <label for="language">Language</label>
      <select id="language" v-model="config.language">
        <option v-for="[code, label] in LANGUAGES" :key="code" :value="code">
          {{ label }}
        </option>
      </select>
    </div>
    <p class="hint" id="language-hint">{{ languageHint }}</p>
    <div class="row" v-if="isArk">
      <div class="field" id="field_threads">
        <label for="threads">Threads</label>
        <input type="number" id="threads" min="1" max="32" v-model.number="config.threads" />
      </div>
      <div class="field" id="field_max_new_tokens">
        <label for="max_new_tokens">Max tokens</label>
        <input
          type="number"
          id="max_new_tokens"
          min="16"
          max="1024"
          step="16"
          v-model.number="config.max_new_tokens"
        />
      </div>
    </div>
    <p class="hint" id="ark-only-hint" v-if="isArk">
      Threads and max tokens only apply to the local ARK-ASR model.
    </p>
  </div>
</template>
