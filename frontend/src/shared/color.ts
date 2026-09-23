// Colour helpers for the settings page.
//
// The colour inputs only understand #rrggbb, but the stored values are
// arbitrary CSS. Sync only when the text side is a plain hex colour.

export function isHex(value: string): boolean {
  return /^#[0-9a-f]{6}$/i.test((value || '').trim())
}

interface RgbaParts {
  r: number
  g: number
  b: number
  a: number
}

function rgbaToParts(value: string): RgbaParts | null {
  const match = /^rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*([\d.]+)\s*)?\)$/i.exec(
    (value || '').trim(),
  )
  if (!match) {
    return null
  }
  const [, r, g, b, a] = match
  return { r: +r, g: +g, b: +b, a: a === undefined ? 1 : +a }
}

export function toHex(value: string): string {
  if (isHex(value)) {
    return value
  }
  const parts = rgbaToParts(value)
  if (!parts) {
    return '#000000'
  }
  const hex = (n: number) => n.toString(16).padStart(2, '0')
  return `#${hex(parts.r)}${hex(parts.g)}${hex(parts.b)}`
}

export function alphaOf(value: string): number {
  if ((value || '').trim() === 'transparent') {
    return 0
  }
  const parts = rgbaToParts(value)
  return parts ? Math.round(parts.a * 100) : 100
}

export function composeBackground(hex: string, alphaPercent: number): string {
  const alpha = +alphaPercent / 100
  if (alpha <= 0) {
    return 'transparent'
  }
  const r = parseInt(hex.slice(1, 3), 16)
  const g = parseInt(hex.slice(3, 5), 16)
  const b = parseInt(hex.slice(5, 7), 16)
  return alpha >= 1 ? hex : `rgba(${r}, ${g}, ${b}, ${alpha.toFixed(2)})`
}
