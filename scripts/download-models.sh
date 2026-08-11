#!/usr/bin/env bash
# download-models.sh — fetch the models asr-overlay needs.
#
# Usage:
#   ./scripts/download-models.sh whisper [small|base|medium|large-v3-turbo]
#   ./scripts/download-models.sh audio8            # Audio8 TTS ONNX INT4
#   ./scripts/download-models.sh audio8 --with-encoder   # + voice registration
#   ./scripts/download-models.sh all
#
# Uses plain curl — no Python, no huggingface_hub CLI.
set -euo pipefail

cd "$(dirname "$0")/.."

WHISPER_BASE="https://huggingface.co/ggerganov/whisper.cpp/resolve/main"
AUDIO8_BASE="https://huggingface.co/Audio8/Audio8-TTS-Preview-0.6B-ONNX-INT4/resolve/main"

fetch() { # $1=url  $2=dest
    local url="$1" dest="$2"
    if [ -s "$dest" ]; then
        echo "  ok   $dest (already present)"
        return
    fi
    mkdir -p "$(dirname "$dest")"
    echo "  get  $dest"
    curl -fL --progress-bar -o "$dest.part" "$url"
    mv "$dest.part" "$dest"
}

download_whisper() {
    local size="${1:-small}"
    echo "== Whisper ggml-$size =="
    fetch "$WHISPER_BASE/ggml-$size.bin" "models/whisper/ggml-$size.bin"
    echo "Run with: cargo run -- --model models/whisper/ggml-$size.bin"
}

download_audio8() {
    echo "== Audio8 TTS Preview 0.6B ONNX INT4 (~572 MiB) =="
    # .onnx and .onnx.data must stay side by side — never separate them.
    for f in \
        runtime_manifest.json \
        tokenizer/tokenizer.json \
        slow_ar_int4.onnx slow_ar_int4.onnx.data \
        fast_ar_int4.onnx fast_ar_int4.onnx.data \
        codec_decoder_fp16.onnx codec_decoder_fp16.onnx.data
    do
        fetch "$AUDIO8_BASE/$f" "models/audio8/$f"
    done

    if [ "${1:-}" = "--with-encoder" ]; then
        echo "== Voice registration encoder (total ~968 MiB) =="
        for f in \
            registration/registration_manifest.json \
            registration/codec_encoder_fp16.onnx \
            registration/codec_encoder_fp16.onnx.data
        do
            fetch "$AUDIO8_BASE/$f" "models/audio8/$f"
        done
    fi
}

case "${1:-}" in
    whisper) download_whisper "${2:-small}" ;;
    audio8)  download_audio8 "${2:-}" ;;
    all)     download_whisper small; download_audio8 "${2:-}" ;;
    *)
        echo "usage: $0 {whisper [size] | audio8 [--with-encoder] | all}" >&2
        exit 1
        ;;
esac
