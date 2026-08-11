(() => {
    const captionEl = document.getElementById('caption');
    const statusEl = document.getElementById('status');

    const showCaption = (text) => {
        captionEl.textContent = text;
        captionEl.classList.add('visible');
    };

    const hideCaption = () => {
        captionEl.classList.remove('visible');
        captionEl.textContent = '';
    };

    // Keep the last caption visible until a newer one arrives. If a caption is
    // empty (Whisper returned nothing), clear the overlay.
    const connect = () => {
        const proto = location.protocol === 'https:' ? 'wss' : 'ws';
        const ws = new WebSocket(`${proto}://${location.host}/ws`);

        ws.onopen = () => {
            statusEl.textContent = '● live';
            statusEl.classList.add('live');
        };

        ws.onmessage = (ev) => {
            try {
                const caption = JSON.parse(ev.data);
                if (caption && typeof caption.text === 'string') {
                    if (caption.text) showCaption(caption.text);
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
