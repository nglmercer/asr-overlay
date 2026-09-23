<script setup lang="ts">
import { computed } from 'vue'
import type { AsrConfig } from '../../shared/types'
import { alphaOf, composeBackground, isHex, toHex } from '../../shared/color'

const props = defineProps<{
  config: AsrConfig
}>()

// Colour picker <-> CSS text, in both directions. The pickers only
// understand #rrggbb; the text fields hold arbitrary CSS.
const colorHex = computed({
  get: () => toHex(props.config.style.color),
  set: (value: string) => {
    props.config.style.color = value
  },
})

const bgHex = computed({
  get: () => toHex(props.config.style.background),
  set: (value: string) => {
    props.config.style.background = composeBackground(value, bgAlpha.value)
  },
})

const bgAlpha = computed({
  get: () => alphaOf(props.config.style.background),
  set: (value: number) => {
    props.config.style.background = composeBackground(toHex(props.config.style.background), value)
  },
})

const onColorText = (value: string) => {
  props.config.style.color = value
  if (isHex(value)) {
    // Picker follows plain hex; anything else leaves it alone.
    colorHex.value = value
  }
}
</script>

<template>
  <div class="panel">
    <h2>Text</h2>
    <div class="field">
      <label for="font_family">Font family</label>
      <input type="text" id="font_family" v-model="config.style.font_family" />
    </div>
    <div class="row">
      <div class="field">
        <label for="font_size">Size (px)</label>
        <input
          type="number"
          id="font_size"
          min="8"
          max="200"
          v-model.number="config.style.font_size"
        />
      </div>
      <div class="field">
        <label for="font_weight">Weight</label>
        <select id="font_weight" v-model.number="config.style.font_weight">
          <option>300</option>
          <option>400</option>
          <option>500</option>
          <option>600</option>
          <option>700</option>
          <option>800</option>
          <option>900</option>
        </select>
      </div>
    </div>
    <div class="row">
      <div class="field">
        <label for="color">Text colour</label>
        <input type="color" id="color" v-model="colorHex" />
      </div>
      <div class="field">
        <label for="color_text">Hex / CSS</label>
        <input
          type="text"
          id="color_text"
          :value="config.style.color"
          @input="onColorText(($event.target as HTMLInputElement).value)"
        />
      </div>
    </div>
    <div class="field">
      <label for="text_shadow">Text shadow (CSS)</label>
      <input type="text" id="text_shadow" v-model="config.style.text_shadow" />
    </div>
    <div class="field inline">
      <input type="checkbox" id="uppercase" v-model="config.style.uppercase" />
      <label for="uppercase" style="margin: 0">Uppercase</label>
    </div>
  </div>

  <div class="panel">
    <h2>Box &amp; layout</h2>
    <div class="row">
      <div class="field">
        <label for="bg_color">Background</label>
        <input type="color" id="bg_color" v-model="bgHex" />
      </div>
      <div class="field">
        <label for="bg_alpha">Opacity</label>
        <input type="range" id="bg_alpha" min="0" max="100" v-model.number="bgAlpha" />
      </div>
    </div>
    <div class="field">
      <label for="background">Background (CSS)</label>
      <input type="text" id="background" v-model="config.style.background" />
    </div>
    <div class="row">
      <div class="field">
        <label for="padding">Padding</label>
        <input type="number" id="padding" min="0" max="80" v-model.number="config.style.padding" />
      </div>
      <div class="field">
        <label for="radius">Radius</label>
        <input type="number" id="radius" min="0" max="80" v-model.number="config.style.radius" />
      </div>
    </div>
    <div class="row">
      <div class="field">
        <label for="position">Position</label>
        <select id="position" v-model="config.style.position">
          <option value="bottom">Bottom</option>
          <option value="center">Center</option>
          <option value="top">Top</option>
        </select>
      </div>
      <div class="field">
        <label for="align">Align</label>
        <select id="align" v-model="config.style.align">
          <option value="center">Center</option>
          <option value="left">Left</option>
          <option value="right">Right</option>
        </select>
      </div>
    </div>
    <div class="row">
      <div class="field">
        <label for="margin">Margin (px)</label>
        <input type="number" id="margin" min="0" max="600" v-model.number="config.style.margin" />
      </div>
      <div class="field">
        <label for="max_width">Width (%)</label>
        <input
          type="number"
          id="max_width"
          min="10"
          max="100"
          v-model.number="config.style.max_width"
        />
      </div>
    </div>
    <div class="field">
      <label for="fade_ms">Fade in (ms)</label>
      <input
        type="number"
        id="fade_ms"
        min="0"
        max="2000"
        step="50"
        v-model.number="config.style.fade_ms"
      />
    </div>
  </div>

  <div class="panel">
    <h2>Custom CSS</h2>
    <p class="hint">
      Appended to the overlay stylesheet. The caption is real DOM text:
      <code>#caption .caption-text</code>.
    </p>
    <textarea
      id="custom_css"
      spellcheck="false"
      placeholder="#caption .caption-text { letter-spacing: 0.02em; }"
      v-model="config.style.custom_css"
    ></textarea>
  </div>
</template>
