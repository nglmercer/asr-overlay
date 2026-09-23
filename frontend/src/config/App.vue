<script setup lang="ts">
import { useSettings } from './composables/useSettings'
import RecognitionPanel from './components/RecognitionPanel.vue'
import ProviderPanel from './components/ProviderPanel.vue'
import VoicePanel from './components/VoicePanel.vue'
import StylePanels from './components/StylePanels.vue'
import PreviewPane from './components/PreviewPane.vue'

const settings = useSettings()
</script>

<template>
  <header>
    <h1>ASR Overlay</h1>
    <span id="status" class="pill" :class="{ live: settings.listening.value }">
      {{ settings.listening.value ? '● listening' : '○ idle' }}
    </span>
    <span class="grow"></span>
    <button id="toggle" class="secondary" @click="settings.toggleListening()">
      {{ settings.listening.value ? 'Stop listening' : 'Start listening' }}
    </button>
    <span id="save-state" class="pill" :class="{ live: settings.saveLive.value }">
      {{ settings.saveText.value }}
    </span>
  </header>

  <div class="layout">
    <div>
      <RecognitionPanel
        :config="settings.config"
        :devices="settings.devices.value"
        :is-ark="settings.isArk.value"
        :language-hint="settings.languageHint.value"
      />
      <ProviderPanel
        :config="settings.config"
        :is-ark="settings.isArk.value"
        :is-http="settings.isHttp.value"
        :is-streaming="settings.isStreaming.value"
        :is-translate="settings.isTranslate.value"
        :model-placeholder="settings.modelPlaceholder.value"
      />
      <VoicePanel v-if="!settings.isStreaming.value" :config="settings.config" :is-ark="settings.isArk.value" />
      <StylePanels :config="settings.config" />
    </div>

    <div>
      <PreviewPane
        v-model:preview-text="settings.previewText.value"
        v-model:photo-backdrop="settings.photoBackdrop.value"
        :live-pill="settings.livePill.value"
        :live-pill-live="settings.livePillLive.value"
        :overlay-url="settings.overlayUrl"
        @render="settings.renderPreview()"
      />
    </div>
  </div>
</template>
