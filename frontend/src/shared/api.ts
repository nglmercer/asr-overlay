// Typed client for the Rust server's HTTP + WebSocket API.
import type {
  AsrConfig,
  Caption,
  DeviceEntry,
  OverlayStyle,
  StatusResponse,
  WsMessage,
} from './types'
import { normalizeConfig } from './types'

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init)
  if (!response.ok) {
    throw new Error(`HTTP ${response.status} for ${path}`)
  }
  return (await response.json()) as T
}

export async function fetchSettings(): Promise<AsrConfig> {
  return normalizeConfig(await json<Partial<AsrConfig>>('/settings'))
}

export async function saveSettings(config: AsrConfig): Promise<void> {
  const response = await fetch('/settings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  })
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`)
  }
}

/** Live style preview: streams to connected overlays without persisting. */
export function pushStyle(style: OverlayStyle): void {
  fetch('/style', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(style),
  }).catch(() => {})
}

export function fetchDevices(): Promise<DeviceEntry[]> {
  return json<DeviceEntry[]>('/devices')
}

export function fetchStatus(): Promise<StatusResponse> {
  return json<StatusResponse>('/status')
}

export async function startListening(): Promise<void> {
  await fetch('/start', { method: 'POST' })
}

export async function stopListening(): Promise<void> {
  await fetch('/stop', { method: 'POST' })
}

export interface CaptionHandlers {
  onCaption: (caption: Caption) => void
  onStyle: (style: OverlayStyle) => void
  onOpen?: () => void
  onClose?: () => void
}

/**
 * Open the caption/style socket, reconnecting on drops. Returns a closer.
 * Late captions never overwrite a newer line: messages apply in arrival
 * order on the single socket.
 */
export function connectStream(handlers: CaptionHandlers): () => void {
  let closed = false
  let socket: WebSocket | null = null
  let timer: ReturnType<typeof setTimeout> | null = null

  const connect = () => {
    if (closed) {
      return
    }
    const proto = location.protocol === 'https:' ? 'wss' : 'ws'
    socket = new WebSocket(`${proto}://${location.host}/ws`)
    socket.onopen = () => handlers.onOpen?.()
    socket.onmessage = (event: MessageEvent) => {
      try {
        const msg = JSON.parse(event.data as string) as WsMessage | null
        if (!msg) {
          return
        }
        if (msg.type === 'style') {
          handlers.onStyle(msg.style)
        } else if (msg.type === 'caption') {
          handlers.onCaption(msg)
        }
      } catch {
        // Ignore malformed frames.
      }
    }
    socket.onclose = () => {
      socket = null
      handlers.onClose?.()
      if (!closed) {
        timer = setTimeout(connect, 2000)
      }
    }
    socket.onerror = () => socket?.close()
  }

  connect()
  return () => {
    closed = true
    if (timer) {
      clearTimeout(timer)
    }
    socket?.close()
  }
}
