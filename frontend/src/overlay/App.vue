<script setup lang="ts">
import { onMounted, onUnmounted, ref } from 'vue'
import { OverlayStyle } from '../shared/style'
import { connectStream, fetchSettings } from '../shared/api'

const text = ref('')
const connected = ref(false)
const statusText = ref('connecting…')

let disconnect: (() => void) | null = null

onMounted(() => {
  // Style comes from /settings on load, then live over the WebSocket so
  // edits in the config page show up in OBS without reloading the source.
  fetchSettings()
    .then((config) => OverlayStyle.applyStyle(config.style, '#caption', 'overlay-style'))
    .catch(() => OverlayStyle.applyStyle(null, '#caption', 'overlay-style'))

  // Keep the last caption visible until a newer one arrives. An empty
  // caption clears the overlay.
  disconnect = connectStream({
    onCaption: (caption) => {
      text.value = caption.text
    },
    onStyle: (style) => {
      OverlayStyle.applyStyle(style, '#caption', 'overlay-style')
    },
    onOpen: () => {
      connected.value = true
      statusText.value = '● live'
    },
    onClose: () => {
      connected.value = false
      statusText.value = 'reconnecting…'
    },
  })
})

onUnmounted(() => {
  disconnect?.()
})
</script>

<template>
  <!-- The caption is plain DOM text, not an image: style it with the config
       page, or with your own CSS in an OBS Browser Source. -->
  <div id="caption"><span class="caption-text" :class="{ visible: text !== '' }">{{ text }}</span></div>
  <div id="status" :class="{ live: connected }">{{ statusText }}</div>
</template>
