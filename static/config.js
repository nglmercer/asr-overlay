(() => {
    // ARK-ASR's 19 supported languages. It auto-detects, so this is a label
    // for the operator rather than a switch that changes decoding.
    const LANGUAGES = [
        ['auto', 'Auto-detect'], ['zh', 'Chinese'], ['en', 'English'],
        ['de', 'German'], ['ja', 'Japanese'], ['fr', 'French'],
        ['ko', 'Korean'], ['es', 'Spanish'], ['pl', 'Polish'],
        ['it', 'Italian'], ['ro', 'Romanian'], ['hu', 'Hungarian'],
        ['cs', 'Czech'], ['nl', 'Dutch'], ['fi', 'Finnish'],
        ['hr', 'Croatian'], ['sk', 'Slovak'], ['sl', 'Slovene'],
        ['et', 'Estonian'], ['lt', 'Lithuanian'],
    ];

    const $ = (id) => document.getElementById(id);
    const previewCaption = document.querySelector('#preview-caption .caption-text');

    // Fields that live on config.style, with their input type.
    const STYLE_FIELDS = [
        ['font_family', 'text'], ['font_size', 'number'], ['font_weight', 'text'],
        ['color_text', 'text'], ['background', 'text'], ['padding', 'number'],
        ['radius', 'number'], ['text_shadow', 'text'], ['position', 'text'],
        ['align', 'text'], ['max_width', 'number'], ['margin', 'number'],
        ['uppercase', 'checkbox'], ['fade_ms', 'number'], ['custom_css', 'text'],
    ];
    const ROOT_FIELDS = [
        ['language', 'text'], ['input_device', 'text'], ['threads', 'number'], ['max_new_tokens', 'number'],
        ['vad_threshold', 'float'], ['vad_silence_ms', 'number'],
        ['vad_min_ms', 'number'], ['vad_max_ms', 'number'],
    ];

    let config = null;

    const languageSelect = $('language');
    for (const [code, label] of LANGUAGES) {
        const option = document.createElement('option');
        option.value = code;
        option.textContent = label;
        languageSelect.appendChild(option);
    }

    // --- colour helpers -----------------------------------------------------
    // The colour inputs only understand #rrggbb, but the stored values are
    // arbitrary CSS. Sync only when the text side is a plain hex colour.
    const isHex = (value) => /^#[0-9a-f]{6}$/i.test((value || '').trim());

    const rgbaToParts = (value) => {
        const match = /^rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*([\d.]+)\s*)?\)$/i
            .exec((value || '').trim());
        if (!match) return null;
        const [, r, g, b, a] = match;
        return { r: +r, g: +g, b: +b, a: a === undefined ? 1 : +a };
    };

    const toHex = (value) => {
        if (isHex(value)) return value;
        const parts = rgbaToParts(value);
        if (!parts) return '#000000';
        const hex = (n) => n.toString(16).padStart(2, '0');
        return `#${hex(parts.r)}${hex(parts.g)}${hex(parts.b)}`;
    };

    const alphaOf = (value) => {
        if ((value || '').trim() === 'transparent') return 0;
        const parts = rgbaToParts(value);
        return parts ? Math.round(parts.a * 100) : 100;
    };

    const composeBackground = () => {
        const hex = $('bg_color').value;
        const alpha = +$('bg_alpha').value / 100;
        if (alpha <= 0) return 'transparent';
        const r = parseInt(hex.slice(1, 3), 16);
        const g = parseInt(hex.slice(3, 5), 16);
        const b = parseInt(hex.slice(5, 7), 16);
        return alpha >= 1 ? hex : `rgba(${r}, ${g}, ${b}, ${alpha.toFixed(2)})`;
    };

    // --- form <-> config ----------------------------------------------------
    const readStyle = () => ({
        font_family: $('font_family').value,
        font_size: +$('font_size').value,
        font_weight: +$('font_weight').value,
        color: $('color_text').value,
        background: $('background').value,
        padding: +$('padding').value,
        radius: +$('radius').value,
        text_shadow: $('text_shadow').value,
        position: $('position').value,
        align: $('align').value,
        max_width: +$('max_width').value,
        margin: +$('margin').value,
        uppercase: $('uppercase').checked,
        fade_ms: +$('fade_ms').value,
        custom_css: $('custom_css').value,
    });

    const fillForm = (cfg) => {
        const s = cfg.style;
        $('language').value = cfg.language;
        // The device list arrives separately; keep the saved value selected
        // even if the device is currently unplugged.
        if (!$('input_device').querySelector(`option[value="${CSS.escape(cfg.input_device || '')}"]`)) {
            const option = document.createElement('option');
            option.value = cfg.input_device || '';
            option.textContent = cfg.input_device || 'System default';
            $('input_device').appendChild(option);
        }
        $('input_device').value = cfg.input_device || '';
        $('threads').value = cfg.threads;
        $('max_new_tokens').value = cfg.max_new_tokens;
        $('vad_threshold').value = cfg.vad_threshold;
        $('vad_silence_ms').value = cfg.vad_silence_ms;
        $('vad_min_ms').value = cfg.vad_min_ms;
        $('vad_max_ms').value = cfg.vad_max_ms;

        $('font_family').value = s.font_family;
        $('font_size').value = s.font_size;
        $('font_weight').value = s.font_weight;
        $('color_text').value = s.color;
        $('color').value = toHex(s.color);
        $('background').value = s.background;
        $('bg_color').value = toHex(s.background);
        $('bg_alpha').value = alphaOf(s.background);
        $('padding').value = s.padding;
        $('radius').value = s.radius;
        $('text_shadow').value = s.text_shadow;
        $('position').value = s.position;
        $('align').value = s.align;
        $('max_width').value = s.max_width;
        $('margin').value = s.margin;
        $('uppercase').checked = s.uppercase;
        $('fade_ms').value = s.fade_ms;
        $('custom_css').value = s.custom_css;
    };

    const renderPreview = () => {
        const style = readStyle();
        window.OverlayStyle.applyStyle(style, '#preview-caption', 'preview-style');
        previewCaption.textContent = $('preview-text').value;
        previewCaption.classList.add('visible');
    };

    // --- server sync --------------------------------------------------------
    let saveTimer = null;
    const pushStyle = () => {
        // Live-update connected overlays without saving the whole config on
        // every keystroke.
        clearTimeout(saveTimer);
        saveTimer = setTimeout(() => {
            fetch('/style', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(readStyle()),
            }).catch(() => {});
        }, 250);
    };

    const onEdit = () => {
        renderPreview();
        pushStyle();
    };

    const save = () => {
        if (!config) return;
        const updated = {
            ...config,
            language: $('language').value,
            input_device: $('input_device').value,
            threads: +$('threads').value,
            max_new_tokens: +$('max_new_tokens').value,
            vad_threshold: parseFloat($('vad_threshold').value),
            vad_silence_ms: +$('vad_silence_ms').value,
            vad_min_ms: +$('vad_min_ms').value,
            vad_max_ms: +$('vad_max_ms').value,
            style: readStyle(),
        };
        const button = $('save');
        button.disabled = true;
        button.textContent = 'Saving…';
        fetch('/settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(updated),
        })
            .then((r) => {
                config = updated;
                button.textContent = r.ok ? 'Saved ✓' : 'Failed';
            })
            .catch(() => { button.textContent = 'Failed'; })
            .finally(() => {
                button.disabled = false;
                setTimeout(() => { button.textContent = 'Save'; }, 1500);
            });
    };

    const refreshStatus = () => {
        fetch('/status')
            .then((r) => r.json())
            .then((s) => {
                const pill = $('status');
                pill.textContent = s.listening ? '● listening' : '○ idle';
                pill.classList.toggle('live', !!s.listening);
                $('toggle').textContent = s.listening ? 'Stop listening' : 'Start listening';
                $('toggle').dataset.listening = s.listening ? '1' : '';
            })
            .catch(() => {});
    };

    $('toggle').addEventListener('click', () => {
        const path = $('toggle').dataset.listening ? '/stop' : '/start';
        fetch(path, { method: 'POST' }).finally(refreshStatus);
    });
    $('save').addEventListener('click', save);

    // Wire every input to the preview.
    for (const [id] of [...STYLE_FIELDS, ...ROOT_FIELDS]) {
        const el = $(id);
        if (el) el.addEventListener('input', onEdit);
    }
    $('preview-text').addEventListener('input', renderPreview);

    // Colour picker <-> CSS text, in both directions.
    $('color').addEventListener('input', () => { $('color_text').value = $('color').value; onEdit(); });
    $('color_text').addEventListener('input', () => {
        if (isHex($('color_text').value)) $('color').value = $('color_text').value;
        onEdit();
    });
    const syncBackground = () => { $('background').value = composeBackground(); onEdit(); };
    $('bg_color').addEventListener('input', syncBackground);
    $('bg_alpha').addEventListener('input', syncBackground);
    $('background').addEventListener('input', () => {
        $('bg_color').value = toHex($('background').value);
        $('bg_alpha').value = alphaOf($('background').value);
        onEdit();
    });

    $('bg-toggle').addEventListener('click', () => {
        $('preview-stage').classList.toggle('photo');
    });

    $('overlay-url').textContent = `${location.origin}/`;

    // Show real captions in the preview while listening.
    const connect = () => {
        const proto = location.protocol === 'https:' ? 'wss' : 'ws';
        const ws = new WebSocket(`${proto}://${location.host}/ws`);
        ws.onmessage = (ev) => {
            try {
                const msg = JSON.parse(ev.data);
                if (msg && msg.type === 'caption' && msg.text) {
                    $('preview-text').value = msg.text;
                    renderPreview();
                    $('live-pill').textContent = 'live caption';
                    $('live-pill').classList.add('live');
                }
            } catch (_) { /* ignore */ }
        };
        ws.onclose = () => setTimeout(connect, 2000);
        ws.onerror = () => ws.close();
    };

    // Devices first, so fillForm can select the saved one.
    fetch('/devices')
        .then((r) => r.json())
        .then((devices) => {
            const select = $('input_device');
            select.innerHTML = '';
            const auto = document.createElement('option');
            auto.value = '';
            auto.textContent = 'System default';
            select.appendChild(auto);
            for (const device of devices) {
                const option = document.createElement('option');
                option.value = device.name;
                option.textContent = device.default ? `${device.name} (default)` : device.name;
                select.appendChild(option);
            }
        })
        .catch(() => {})
        .finally(() => {
            fetch('/settings')
                .then((r) => r.json())
                .then((cfg) => {
                    config = cfg;
                    fillForm(cfg);
                    renderPreview();
                })
                .catch(() => {});
        });

    refreshStatus();
    setInterval(refreshStatus, 3000);
    connect();
})();
