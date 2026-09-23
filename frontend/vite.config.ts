import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// Multi-page build: the overlay (index.html) and the settings page
// (config.html) are separate entries. Output names are stable — no content
// hashes — because the Rust server embeds the bundle and serves fixed
// routes: /index.js, /config.js, /style.js, /vendor.js.
export default defineConfig({
  plugins: [vue()],
  build: {
    outDir: 'dist',
    // Localhost-served app: keep bundles readable and greppable instead of
    // minified. Stable chunk names also let the server address each file.
    minify: false,
    // Every browser that can run this app (including OBS's Chromium)
    // supports modulepreload natively; skip the extra polyfill chunk so
    // the output is exactly the files the server routes serve.
    polyfillModulePreload: false,
    rollupOptions: {
      input: {
        index: 'index.html',
        config: 'config.html',
      },
      output: {
        entryFileNames: '[name].js',
        chunkFileNames: '[name].js',
        assetFileNames: '[name][extname]',
        codeSplitting: {
          groups: [
            { name: 'vendor', test: /node_modules/ },
            { name: 'style', test: /src\/shared/ },
          ],
        },
      },
    },
  },
  server: {
    // `npm run dev` against a running backend (`cargo run`): API and
    // WebSocket traffic proxies to the Rust server.
    proxy: {
      '/settings': 'http://localhost:8080',
      '/devices': 'http://localhost:8080',
      '/status': 'http://localhost:8080',
      '/start': 'http://localhost:8080',
      '/stop': 'http://localhost:8080',
      '/style': 'http://localhost:8080',
      '/caption.png': 'http://localhost:8080',
      '/ws': { target: 'ws://localhost:8080', ws: true },
    },
  },
})
