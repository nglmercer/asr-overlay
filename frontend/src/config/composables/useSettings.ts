// Server sync for the settings page.
//
// Everything autosaves. Style edits also stream to connected overlays
// immediately so OBS updates while you drag a slider, while the full
// config is persisted to disk on a slower debounce.
import { computed, onMounted, onUnmounted, reactive, ref, watch } from 'vue'
import type { AsrConfig, DeviceEntry } from '../../shared/types'
import { defaultConfig } from '../../shared/types'
import { OverlayStyle } from '../../shared/style'
import {
  connectStream,
  fetchDevices,
  fetchSettings,
  fetchStatus,
  pushStyle,
  saveSettings,
  startListening,
  stopListening,
} from '../../shared/api'

// ARK-ASR's 19 supported languages. It auto-detects, so this is a label
// for the operator rather than a switch that changes decoding.
export const LANGUAGES: Array<[string, string]> = [
  ['auto', 'Auto-detect'],
  ['zh', 'Chinese'],
  ['en', 'English'],
  ['de', 'German'],
  ['ja', 'Japanese'],
  ['fr', 'French'],
  ['ko', 'Korean'],
  ['es', 'Spanish'],
  ['pl', 'Polish'],
  ['it', 'Italian'],
  ['ro', 'Romanian'],
  ['hu', 'Hungarian'],
  ['cs', 'Czech'],
  ['nl', 'Dutch'],
  ['fi', 'Finnish'],
  ['hr', 'Croatian'],
  ['sk', 'Slovak'],
  ['sl', 'Slovene'],
  ['et', 'Estonian'],
  ['lt', 'Lithuanian'],
]

export function useSettings() {
  // Defaults first so the form renders instantly; the server state fills
  // in on mount without triggering a save.
  const config = reactive<AsrConfig>(defaultConfig())
  const devices = ref<DeviceEntry[]>([])
  const saveText = ref('Saved')
  const listening = ref(false)
  const previewText = ref('The quick brown fox jumps over the lazy dog.')
  const livePill = ref('preview')
  const livePillLive = ref(false)
  const photoBackdrop = ref(false)

  const saveLive = computed(() => saveText.value === 'Saved')
  const overlayUrl = `${location.origin}/`

  // --- provider-driven visibility --------------------------------------
  const providerKind = computed(() => config.provider.kind)
  const isArk = computed(() => providerKind.value === 'ark')
  const isHttp = computed(() => providerKind.value === 'openai_compatible')
  const isStreaming = computed(
    () =>
      providerKind.value === 'voxtral_realtime' ||
      providerKind.value === 'gemini_live_transcribe' ||
      providerKind.value === 'gemini_live_translate',
  )
  const isTranslate = computed(() => providerKind.value === 'gemini_live_translate')

  const modelPlaceholder = computed(() => {
    switch (providerKind.value) {
      case 'openai_compatible':
        return 'whisper-large-v3'
      case 'voxtral_realtime':
        return 'voxtral-mini-latest'
      case 'gemini_live_transcribe':
      case 'gemini_live_translate':
        return 'gemini-2.0-flash-live-001'
      default:
        return ''
    }
  })

  const languageHint = computed(() =>
    isHttp.value
      ? 'Sent to the transcription server as a hint; use Auto-detect to let the server decide.'
      : 'ARK-ASR detects the spoken language on its own, so this is a label only — it does not change or translate the output.',
  )

  // --- preview -----------------------------------------------------------
  const renderPreview = () => {
    OverlayStyle.applyStyle(config.style, '#preview-caption', 'preview-style')
  }

  // --- server sync ---------------------------------------------------------
  let suspendSave = true
  let styleTimer: ReturnType<typeof setTimeout> | null = null
  let saveTimer: ReturnType<typeof setTimeout> | null = null
  let saving = false
  let pending = false

  const flush = () => {
    if (saving) {
      pending = true
      return
    }
    saving = true
    saveText.value = 'Saving…'
    // Snapshot: the config keeps changing under us while saving.
    const payload = JSON.parse(JSON.stringify(config)) as AsrConfig
    saveSettings(payload)
      .then(() => {
        saveText.value = 'Saved'
      })
      .catch((e: Error) => {
        saveText.value = `Save failed: ${e.message}`
      })
      .finally(() => {
        saving = false
        if (pending) {
          pending = false
          flush()
        }
      })
  }

  const autosave = () => {
    saveText.value = 'Unsaved…'
    if (saveTimer) {
      clearTimeout(saveTimer)
    }
    saveTimer = setTimeout(flush, 600)
  }

  const onEdit = () => {
    renderPreview()
    if (styleTimer) {
      clearTimeout(styleTimer)
    }
    styleTimer = setTimeout(() => pushStyle({ ...config.style }), 150)
    autosave()
  }

  watch(
    config,
    () => {
      if (!suspendSave) {
        onEdit()
      }
    },
    { deep: true },
  )

  // Don't lose the last keystroke if the tab closes mid-debounce.
  const handleBeforeUnload = () => {
    if (saveTimer) {
      clearTimeout(saveTimer)
      saveTimer = null
      navigator.sendBeacon?.(
        '/settings',
        new Blob([JSON.stringify(config)], { type: 'application/json' }),
      )
    }
  }

  // --- status --------------------------------------------------------------
  const refreshStatus = () => {
    fetchStatus()
      .then((s) => {
        listening.value = s.listening
      })
      .catch(() => {})
  }

  const toggleListening = () => {
    const run = listening.value ? stopListening() : startListening()
    run.finally(refreshStatus)
  }

  // --- lifecycle -------------------------------------------------------------
  let disconnect: (() => void) | null = null
  let statusTimer: ReturnType<typeof setInterval> | null = null

  onMounted(() => {
    window.addEventListener('beforeunload', handleBeforeUnload)
    // Devices first, so the saved microphone can be selected.
    fetchDevices()
      .then((list) => {
        devices.value = list
      })
      .catch(() => {})
      .finally(() => {
        fetchSettings()
          .then((cfg) => {
            suspendSave = true
            Object.assign(config, JSON.parse(JSON.stringify(cfg)) as AsrConfig)
            suspendSave = false
            renderPreview()
          })
          .catch(() => {
            suspendSave = false
          })
      })

    refreshStatus()
    statusTimer = setInterval(refreshStatus, 3000)

    // Show real captions in the preview while listening.
    disconnect = connectStream({
      onCaption: (caption) => {
        if (caption.text) {
          previewText.value = caption.text
          renderPreview()
          livePill.value = 'live caption'
          livePillLive.value = true
        }
      },
      onStyle: () => {},
    })
    renderPreview()
  })

  onUnmounted(() => {
    window.removeEventListener('beforeunload', handleBeforeUnload)
    if (statusTimer) {
      clearInterval(statusTimer)
    }
    if (styleTimer) {
      clearTimeout(styleTimer)
    }
    if (saveTimer) {
      clearTimeout(saveTimer)
    }
    disconnect?.()
  })

  return {
    config,
    devices,
    saveText,
    saveLive,
    listening,
    previewText,
    livePill,
    livePillLive,
    photoBackdrop,
    overlayUrl,
    isArk,
    isHttp,
    isStreaming,
    isTranslate,
    modelPlaceholder,
    languageHint,
    toggleListening,
    renderPreview,
  }
}

export type SettingsState = ReturnType<typeof useSettings>
