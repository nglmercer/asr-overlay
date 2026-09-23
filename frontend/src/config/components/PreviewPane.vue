<script setup lang="ts">
const previewText = defineModel<string>('previewText', { required: true })
const photoBackdrop = defineModel<boolean>('photoBackdrop', { required: true })

defineProps<{
  livePill: string
  livePillLive: boolean
  overlayUrl: string
}>()

defineEmits<{
  render: []
}>()
</script>

<template>
  <div class="panel">
    <h2>Preview</h2>
    <div class="stage-bar">
      <input
        type="text"
        id="preview-text"
        style="flex: 1"
        v-model="previewText"
        @input="$emit('render')"
      />
      <button
        class="secondary"
        id="bg-toggle"
        @click="photoBackdrop = !photoBackdrop"
      >
        Backdrop
      </button>
      <span class="pill" id="live-pill" :class="{ live: livePillLive }">{{ livePill }}</span>
    </div>
    <!-- Same markup and same stylesheet generator as the real
             overlay, so this cannot drift from what OBS shows. -->
    <div id="preview-stage" :class="{ photo: photoBackdrop }">
      <div id="preview-caption"><span class="caption-text visible">{{ previewText }}</span></div>
    </div>
    <p class="hint" style="margin-top: 12px">
      OBS Browser Source URL:
      <code id="overlay-url">{{ overlayUrl }}</code>
    </p>
  </div>
</template>
