use anyhow::{anyhow, Result};
use cairo::{Context, Format, ImageSurface, Operator};
use std::path::Path;

/// How the caption text should be drawn.
#[derive(Debug, Clone)]
pub struct RenderStyle {
    /// Font family, resolved via fontconfig (e.g. "DejaVu Sans", "Sans").
    pub font_family: String,
    pub font_size: f64,
    /// RGBA 0.0..1.0
    pub text_rgba: [f64; 4],
    /// Background behind the text (semi-transparent by default).
    pub bg_rgba: [f64; 4],
    /// Corner radius of the background pill.
    pub corner_radius: f64,
    /// Padding around the text inside the background.
    pub padding: f64,
    /// Maximum canvas width; text is clipped/soft-wrapped at this bound.
    pub max_width: f64,
    /// Max lines of soft-wrap before truncation with an ellipsis.
    pub max_lines: usize,
}

impl Default for RenderStyle {
    fn default() -> Self {
        Self {
            font_family: "DejaVu Sans".to_string(),
            font_size: 42.0,
            text_rgba: [1.0, 1.0, 1.0, 1.0],
            bg_rgba: [0.0, 0.0, 0.0, 0.6],
            corner_radius: 12.0,
            padding: 16.0,
            max_width: 1600.0,
            max_lines: 3,
        }
    }
}

/// Render `text` to a transparent RGBA surface and encode it as PNG bytes.
///
/// The canvas auto-sizes to the wrapped text so OBS can use it directly as a
/// transparent overlay image. No camera frames are involved — this only draws
/// the subtitle text.
pub fn render_caption_png(text: &str, style: &RenderStyle) -> Result<Vec<u8>> {
    let (width, height, lines) = layout(text, style)?;
    let surface = ImageSurface::create(Format::ARgb32, width, height)?;
    let ctx = Context::new(&surface)?;

    // Transparent base.
    ctx.set_operator(Operator::Clear);
    ctx.paint()?;
    ctx.set_operator(Operator::Over);

    // Background pill behind the whole block of lines.
    draw_pill(&ctx, width as f64, height as f64, style)?;

    // Text lines, top-left anchored inside the padding.
    ctx.set_font_face(&font_face(style)?);
    ctx.set_font_size(style.font_size);
    ctx.set_source_rgba(style.text_rgba[0], style.text_rgba[1], style.text_rgba[2], style.text_rgba[3]);

    let line_height = text_height(&ctx, style) + 6.0;
    let mut y = style.padding + text_ascent(&ctx, style);
    for (i, line) in lines.iter().enumerate() {
        if i > 0 {
            y += line_height;
        }
        ctx.move_to(style.padding, y);
        ctx.show_text(line)?;
    }

    surface.flush();
    let mut out = Vec::new();
    surface.write_to_png(&mut out).map_err(|e| anyhow!("encode png: {e}"))?;
    Ok(out)
}

/// Write the PNG to `path` (used for debugging / as a static fallback).
pub fn render_caption_to_file(text: &str, style: &RenderStyle, path: &Path) -> Result<()> {
    let png = render_caption_png(text, style)?;
    std::fs::write(path, png).map_err(|e| anyhow!("write {}: {e}", path.display()))
}

/// Compute the auto-fit canvas size and the (soft-wrapped) lines to draw.
fn layout(text: &str, style: &RenderStyle) -> Result<(i32, i32, Vec<String>)> {
    if text.trim().is_empty() {
        return Err(anyhow!("empty caption"));
    }

    // Probe surface to measure text.
    let probe = ImageSurface::create(Format::ARgb32, 1, 1)?;
    let ctx = Context::new(&probe)?;
    ctx.set_font_face(&font_face(style)?);
    ctx.set_font_size(style.font_size);

    let lines = wrap(&ctx, text, style);
    let max_line_w = lines
        .iter()
        .map(|l| text_width(&ctx, l))
        .fold(0.0_f64, f64::max);

    let line_height = text_height(&ctx, style) + 6.0;
    let content_w = max_line_w.max(1.0);
    let content_h = line_height * lines.len().max(1) as f64 - 6.0;

    let width = (content_w + 2.0 * style.padding).ceil() as i32;
    let height = (content_h + 2.0 * style.padding).ceil() as i32;
    Ok((width, height, lines))
}

/// Soft-wrap text to `max_width` (and `max_lines`, with ellipsis).
fn wrap(ctx: &Context, text: &str, style: &RenderStyle) -> Vec<String> {
    let max_inner = style.max_width - 2.0 * style.padding;
    let mut lines: Vec<String> = Vec::new();

    for paragraph in text.split('\n') {
        let mut current = String::new();
        for word in paragraph.split_whitespace() {
            let candidate = if current.is_empty() {
                word.to_string()
            } else {
                format!("{current} {word}")
            };
            if text_width(ctx, &candidate) <= max_inner || current.is_empty() {
                current = candidate;
            } else {
                lines.push(current);
                current = word.to_string();
            }
        }
        if !current.is_empty() {
            lines.push(current);
        }
    }

    if lines.len() > style.max_lines {
        lines.truncate(style.max_lines);
        let last = lines.last_mut().expect("max_lines > 0");
        if text_width(ctx, last) + text_width(ctx, "…") > max_inner {
            while text_width(ctx, last) + text_width(ctx, "…") > max_inner && last.chars().count() > 1 {
                last.pop();
            }
        }
        last.push('…');
    }
    lines
}

fn font_face(style: &RenderStyle) -> Result<cairo::FontFace> {
    cairo::FontFace::toy_create(
        &style.font_family,
        cairo::FontSlant::Normal,
        cairo::FontWeight::Normal,
    )
    .map_err(|e| anyhow!("font {e}"))
}

fn text_width(ctx: &Context, text: &str) -> f64 {
    ctx.text_extents(text).map(|te| te.width()).unwrap_or(0.0)
}

fn text_height(ctx: &Context, _style: &RenderStyle) -> f64 {
    ctx.font_extents().map(|fe| fe.ascent() + fe.descent()).unwrap_or(0.0)
}

fn text_ascent(ctx: &Context, _style: &RenderStyle) -> f64 {
    ctx.font_extents().map(|fe| fe.ascent()).unwrap_or(0.0)
}

/// Draw a rounded-rect background that fits the whole text block.
fn draw_pill(ctx: &Context, width: f64, height: f64, style: &RenderStyle) -> Result<()> {
    ctx.set_source_rgba(style.bg_rgba[0], style.bg_rgba[1], style.bg_rgba[2], style.bg_rgba[3]);
    let r = style.corner_radius.min(width / 2.0).min(height / 2.0);
    rounded_rect(ctx, 0.0, 0.0, width, height, r);
    ctx.fill().map_err(|e| anyhow!("fill: {e}"))
}

/// Append a rounded rectangle path to `ctx`.
fn rounded_rect(ctx: &Context, x: f64, y: f64, w: f64, h: f64, r: f64) {
    ctx.new_path();
    ctx.arc(x + w - r, y + r, r, -std::f64::consts::FRAC_PI_2, 0.0);
    ctx.arc(x + w - r, y + h - r, r, 0.0, std::f64::consts::FRAC_PI_2);
    ctx.arc(x + r, y + h - r, r, std::f64::consts::FRAC_PI_2, std::f64::consts::PI);
    ctx.arc(x + r, y + r, r, std::f64::consts::PI, std::f64::consts::PI + std::f64::consts::FRAC_PI_2);
    ctx.close_path();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_non_empty_png() {
        let png = render_caption_png("Hola mundo", &RenderStyle::default()).unwrap();
        // PNG magic.
        assert_eq!(&png[..8], &[137, 80, 78, 71, 13, 10, 26, 10]);
    }

    #[test]
    fn rejects_empty_text() {
        assert!(render_caption_png("", &RenderStyle::default()).is_err());
        assert!(render_caption_png("   ", &RenderStyle::default()).is_err());
    }

    #[test]
    fn wraps_long_text() {
        let style = RenderStyle {
            max_width: 300.0,
            max_lines: 2,
            ..RenderStyle::default()
        };
        let png = render_caption_png(&"palabra ".repeat(60), &style).unwrap();
        assert_eq!(&png[..8], &[137, 80, 78, 71, 13, 10, 26, 10]);
    }

    #[test]
    fn writes_file() {
        let path = std::env::temp_dir().join("asr-overlay-caption-test.png");
        render_caption_to_file("Prueba de subtítulo", &RenderStyle::default(), &path).unwrap();
        assert!(path.exists());
        let _ = std::fs::remove_file(path);
    }
}
