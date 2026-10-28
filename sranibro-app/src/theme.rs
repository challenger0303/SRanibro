//! Design tokens for the UI — one source of truth so nothing is eyeballed.
//!
//! Semantic colors, a type ramp, an 8px spacing scale, radii, and shared
//! frame/animation helpers. Components are built from these, the way a real
//! design system enforces consistency.

use eframe::egui::{self, Color32};

// --- surfaces (refined cool Fluent-dark) ---
pub const BG: Color32 = Color32::from_rgb(0x0f, 0x11, 0x15);
pub const NAV_BG: Color32 = Color32::from_rgb(0x0b, 0x0d, 0x11);
pub const SURFACE: Color32 = Color32::from_rgb(0x17, 0x1a, 0x21);
pub const INNER: Color32 = Color32::from_rgb(0x0b, 0x0d, 0x11);
pub const BORDER: Color32 = Color32::from_rgb(0x2b, 0x31, 0x3a);
pub const BORDER_STRONG: Color32 = Color32::from_rgb(0x3a, 0x41, 0x4d);
// Faint chip for healthy pipeline nodes: a hair darker than SURFACE so nodes read
// as grouped, WITHOUT a border (avoids the boxed-tile-grid look). Subtler than INNER.
pub const NODE_BG: Color32 = Color32::from_rgb(0x13, 0x16, 0x1c);

// --- text ---
pub const TEXT1: Color32 = Color32::from_rgb(0xe6, 0xea, 0xf0);
pub const TEXT2: Color32 = Color32::from_rgb(0x8f, 0x98, 0xa8);
// TEXT3 = muted-but-LEGIBLE: 4.73:1 on SURFACE (was #5f6776 ≈ 3.06:1, below WCAG
// AA 4.5 for small text — captions/log/metadata are systemic, not decorative).
pub const TEXT3: Color32 = Color32::from_rgb(0x7c, 0x86, 0x96);
// DECO = the old dim tone, for pure DECORATION only (inactive connectors, hairline
// fills) where low contrast is intentional — never for text the user must read.
pub const DECO: Color32 = Color32::from_rgb(0x5f, 0x67, 0x76);

// --- accent + semantic ---
pub const ACCENT: Color32 = Color32::from_rgb(0x4c, 0xc2, 0xff);
pub const ACCENT_BG: Color32 = Color32::from_rgb(0x10, 0x22, 0x2e);
pub const OK: Color32 = Color32::from_rgb(0x35, 0xd0, 0x7f);
pub const OK_BG: Color32 = Color32::from_rgb(0x10, 0x21, 0x1b);
pub const OK_BORDER: Color32 = Color32::from_rgb(0x23, 0x47, 0x36);
pub const WARN: Color32 = Color32::from_rgb(0xe6, 0xb4, 0x50);
pub const ERR: Color32 = Color32::from_rgb(0xff, 0x5c, 0x5c);
pub const LED_OFF: Color32 = Color32::from_rgb(0x3a, 0x41, 0x4d);

// --- design scale ---
// The whole console is a faithful, uniformly-scaled reproduction of the
// `03_Pro_Console.html` mockup: EVERY dimension is `<mockup_px> * S`, so the
// layout keeps the mockup's exact proportions while staying readable on 4K.
// `S` is the single global scale; bumping it scales the entire UI together.
pub const S: f32 = 1.3;

// Mockup design size, in mockup px (rail 48 + main padding 14*2 + content 1100;
// height = stacked sections). The window is sized to hold this exactly.
// ~70% of the original 1176 — the equal-width layout was too landscape-wide.
pub const DESIGN_W: f32 = 824.0;
// Holds the equal-width eye card + the (compact) 2-lane branched pipeline.
pub const DESIGN_H: f32 = 565.0;

// Fixed, non-resizable window. CRITICAL UNIT NOTE: eframe's `with_inner_size`
// and egui's layout both use LOGICAL POINTS (the OS scale, here 1.5, multiplies
// both to physical px equally). So the window and the content live in the SAME
// unit: `content_w/h` are simply `WIN_W/WIN_H`. Do NOT multiply the window by
// ppp, and do NOT divide content by ppp — doing both (the old bug) left the
// window 1.5x bigger than the content (large empty space bottom-right).
// `available_*` over-reports the surface and must never be used for layout.
pub const WIN_W: f32 = DESIGN_W * S;
pub const WIN_H: f32 = DESIGN_H * S;

pub const NAV_W: f32 = 48.0 * S;
pub const MAIN_PAD: f32 = 14.0 * S; // central panel padding (mockup main padding 14)
pub const CARD_PAD: f32 = 14.0 * S; // card inner padding (mockup card padding 14)

/// Visible content width in logical points (== the window width; same unit).
pub fn content_w(_ctx: &egui::Context) -> f32 {
    WIN_W
}

/// Visible content height in logical points (== the window height; same unit).
pub fn content_h() -> f32 {
    WIN_H
}

// --- spacing + radius, all scaled (mockup px * S) ---
pub const SP2: f32 = 8.0 * S; // small gaps (mockup 8/9)
pub const SP3: f32 = 12.0 * S; // inter-section gap (mockup 12)
pub const SP4: f32 = 16.0 * S;
pub const R_CARD: f32 = 10.0 * S; // mockup card radius 10
pub const R_INNER: f32 = 8.0 * S; // mockup inner radius 7-9
pub const R_BOX: f32 = 9.0 * S; // eye-camera box radius (mockup 9)

/// Apply fonts (native Segoe UI + Consolas if present) and the dark visuals.
pub fn apply(ctx: &egui::Context) {
    // Leave pixels_per_point at the monitor's native value (stable). Forcing it
    // here fought with eframe's per-frame reset and caused scale flicker.
    let mut fonts = egui::FontDefinitions::default();
    if let Ok(b) = std::fs::read("C:/Windows/Fonts/segoeui.ttf") {
        fonts
            .font_data
            .insert("segoe".into(), egui::FontData::from_owned(b).into());
        if let Some(f) = fonts.families.get_mut(&egui::FontFamily::Proportional) {
            f.insert(0, "segoe".into());
        }
    }
    if let Ok(b) = std::fs::read("C:/Windows/Fonts/consola.ttf") {
        fonts
            .font_data
            .insert("consola".into(), egui::FontData::from_owned(b).into());
        if let Some(f) = fonts.families.get_mut(&egui::FontFamily::Monospace) {
            f.insert(0, "consola".into());
        }
    }
    ctx.set_fonts(fonts);

    let mut v = egui::Visuals::dark();
    v.override_text_color = Some(TEXT1);
    v.panel_fill = BG;
    v.window_fill = BG;
    v.faint_bg_color = SURFACE;
    v.extreme_bg_color = INNER;
    v.selection.bg_fill = ACCENT.gamma_multiply(0.35);
    v.selection.stroke = egui::Stroke::new(1.0, ACCENT);
    // Panel separators (top-bar bottom edge, nav-rail right edge) -> dark hairline.
    v.widgets.noninteractive.bg_stroke = egui::Stroke::new(1.0, BORDER);
    v.widgets.hovered.bg_stroke = egui::Stroke::new(1.0, ACCENT);
    v.widgets.active.bg_stroke = egui::Stroke::new(1.0, ACCENT);
    ctx.set_visuals(v);
}

/// A standard card: surface fill, hairline border, rounded, padded.
pub fn card() -> egui::Frame {
    egui::Frame::default()
        .fill(SURFACE)
        .stroke(egui::Stroke::new(1.0, BORDER))
        .rounding(R_CARD)
        .inner_margin(egui::Margin::same(CARD_PAD))
}

/// Linear color interpolation (for animated state transitions).
pub fn lerp_color(a: Color32, b: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    let m = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t).round() as u8;
    Color32::from_rgba_unmultiplied(
        m(a.r(), b.r()),
        m(a.g(), b.g()),
        m(a.b(), b.b()),
        m(a.a(), b.a()),
    )
}

// --- type ramp (monospace-dominant, like the mockup; scaled by S) ---
pub fn h3(s: &str) -> egui::RichText {
    egui::RichText::new(s)
        .monospace()
        .size(13.0 * S)
        .strong()
        .color(TEXT1)
}
pub fn label(s: &str) -> egui::RichText {
    egui::RichText::new(s)
        .monospace()
        .size(11.0 * S)
        .color(TEXT2)
}
/// Proportional (Segoe UI) body text for PROSE — instruction/help SENTENCES, where
/// monospace hurts readability and rhythm. Data, identifiers, paths, rates, and the
/// log stay monospace (the console identity); this is for things you read, not scan.
pub fn prose(s: &str) -> egui::RichText {
    egui::RichText::new(s).size(11.0 * S).color(TEXT2)
}
pub fn num(s: &str) -> egui::RichText {
    egui::RichText::new(s)
        .monospace()
        .size(11.0 * S)
        .color(TEXT1)
}
