// Turns an OverlayStyle (from /settings) into CSS.
//
// The overlay page and the config page's preview both call this, so what you
// see in the preview is produced by exactly the same code that styles OBS.
(() => {
    const POSITIONS = {
        bottom: (m) => `bottom: ${m}px; top: auto; transform: none;`,
        top: (m) => `top: ${m}px; bottom: auto; transform: none;`,
        center: () => `top: 50%; bottom: auto; transform: translateY(-50%);`,
    };

    // The caption text is inserted with textContent, never innerHTML, so it
    // cannot inject markup. custom_css is author-supplied and goes into a
    // <style> element by design.
    const buildCss = (style, scope) => {
        const s = style || {};
        const position = (POSITIONS[s.position] || POSITIONS.bottom)(s.margin ?? 40);
        const align = s.align || 'center';
        // `left`/`right` alignment should hug that edge, not sit centred.
        const justify =
            align === 'left' ? 'flex-start' : align === 'right' ? 'flex-end' : 'center';

        return `
${scope} {
    position: absolute;
    left: 0;
    right: 0;
    ${position}
    display: flex;
    justify-content: ${justify};
    pointer-events: none;
}
${scope} .caption-text {
    display: inline-block;
    max-width: ${s.max_width ?? 90}%;
    box-sizing: border-box;
    font-family: ${s.font_family || 'sans-serif'};
    font-size: ${s.font_size ?? 28}px;
    font-weight: ${s.font_weight ?? 600};
    color: ${s.color || '#fff'};
    background: ${s.background || 'transparent'};
    padding: ${s.padding ?? 8}px ${(s.padding ?? 8) * 1.6}px;
    border-radius: ${s.radius ?? 8}px;
    text-shadow: ${s.text_shadow || 'none'};
    text-align: ${align};
    text-transform: ${s.uppercase ? 'uppercase' : 'none'};
    word-wrap: break-word;
    overflow-wrap: anywhere;
    white-space: pre-wrap;
    opacity: 0;
    transition: opacity ${s.fade_ms ?? 200}ms ease-in;
}
${scope} .caption-text.visible { opacity: 1; }
${s.custom_css || ''}
`;
    };

    // Injects (or replaces) a <style> element carrying the generated CSS.
    const applyStyle = (style, scope, elementId) => {
        let el = document.getElementById(elementId);
        if (!el) {
            el = document.createElement('style');
            el.id = elementId;
            document.head.appendChild(el);
        }
        el.textContent = buildCss(style, scope);
    };

    window.OverlayStyle = { buildCss, applyStyle };
})();
