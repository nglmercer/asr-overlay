(() => {
    const captionEl = document.querySelector('#caption .caption-text');
    const statusEl = document.getElementById('status');

    const applyStyle = (style) =>
        window.OverlayStyle.applyStyle(style, '#caption', 'overlay-style');

    const showCaption = (text) => {
        captionEl.textContent = text;
        captionEl.classList.add('visible');
    };

    const hideCaption = () => {
        captionEl.classList.remove('visible');
        captionEl.textContent = '';
    };

    // Style comes from /settings on load, then live over the WebSocket so
    // edits in the config page show up in OBS without reloading the source.
    fetch('/settings')
        .then((r) => r.json())
        .then((config) => applyStyle(config.style))
        .catch(() => applyStyle(null));

    // Keep the last caption visible until a newer one arrives. An empty
    // caption clears the overlay.
    const connect = () => {
        const proto = location.protocol === 'https:' ? 'wss' : 'ws';
        const ws = new WebSocket(`${proto}://${location.host}/ws`);

        ws.onopen = () => {
            statusEl.textContent = '● live';
            statusEl.classList.add('live');
        };

        ws.onmessage = (ev) => {
            try {
                const msg = JSON.parse(ev.data);
                if (!msg) return;
                if (msg.type === 'style') {
                    applyStyle(msg.style);
                } else if (msg.type === 'caption') {
                    if (msg.text) showCaption(msg.text);
                    else hideCaption();
                }
            } catch (_) { /* ignore malformed frames */ }
        };

        ws.onclose = () => {
            statusEl.textContent = 'reconnecting…';
            statusEl.classList.remove('live');
            setTimeout(connect, 2000);
        };

        ws.onerror = () => ws.close();
    };

    connect();
})();
