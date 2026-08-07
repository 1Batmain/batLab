//! File purpose: Bridges a CPU-side tensor produced step by step (the diffusion
//! sampler's latent) to a GPU buffer the visualiser can bind and render live.
//!
//! Training needs no such bridge: the model's output buffer already lives on the
//! GPU, so the visualiser binds it and reads whatever the last pass wrote. The
//! reverse diffusion chain is the opposite shape — [`crate::sample_diffusion`]
//! keeps its latent as a `Vec<f32>` on the CPU and never uploads it, since the
//! model is only ever asked to predict from it. This type gives that latent a
//! GPU home so the same visualiser can display it, at the cost of one small
//! upload per denoising step.

use std::sync::Arc;

use crate::GpuContext;
use wgpu::util::DeviceExt as _;

/// The rule drawn between the two panes, one entry per column, as raw tensor
/// values — the shader maps `-1 → black` and `+1 → white`.
///
/// Without it the window shows one image with a seam down the middle and
/// nothing saying there is a seam. A user watching a live run read it as a
/// single picture that had come apart — "cut in two, the left decorrelated
/// from the right" — which is a fair description of a frame whose halves are
/// *supposed* to differ, and an alarming one if you think you are looking at
/// one image. A light/dark/light rule is the cheapest thing that cannot be
/// mistaken for content: no sample from this model contains a one-pixel
/// saturated-black column flanked by two saturated-white ones.
pub const SEPARATOR_COLUMNS: [f32; 3] = [1.0, -1.0, 1.0];

/// How many panes the frame carries.
///
/// ```text
///   Both  — width = 2 * W + 3          X0Only — width = W
///   ┌───────────────┬─┬───────────────┐  ┌───────────────┐
///   │      x_t      │▕│    x0_hat     │  │    x0_hat     │
///   │   (noisy)     │▕│ (what the     │  │               │
///   │               │▕│  model sees)  │  │               │
///   └───────────────┴─┴───────────────┘  └───────────────┘
/// ```
///
/// [`LiveView::X0Only`] is the perpetual default, and the reason is not screen
/// real estate: *"on n'a même pas besoin de la fenêtre de gauche"*. A drift is
/// something to look at, and the double view puts the picture in half a window
/// at the wrong aspect ratio, beside a field of noise that competes with it for
/// attention. The noise is diagnostic, so it stays one key away — `[x]` — for
/// when the question is what the latent is doing rather than what the image is.
///
/// A single pane is also the only view whose frame is **square**, which is the
/// real geometry of a CIFAR sample; the visualiser letterboxes to whatever
/// ratio it is handed, so the window simply opens right.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiveView {
    /// x̂₀ alone, at the image's own aspect ratio.
    X0Only,
    /// x_t, the rule, then x̂₀ — the diagnostic view.
    Both,
}

impl Default for LiveView {
    fn default() -> Self {
        Self::Both
    }
}

impl LiveView {
    /// Full width of the composed frame for a given pane width — what the
    /// visualiser must be told, as opposed to the model's own width.
    pub fn frame_width(self, pane_width: u32) -> u32 {
        match self {
            LiveView::X0Only => pane_width.max(1),
            // The `+ 3` is the rule: the visualiser is told this number, so the
            // gutter is real pixels in the buffer rather than something the
            // shader would have to know about.
            LiveView::Both => pane_width.max(1) * 2 + SEPARATOR_COLUMNS.len() as u32,
        }
    }

    /// `[x]` swaps the two.
    pub fn toggle(self) -> Self {
        match self {
            LiveView::X0Only => LiveView::Both,
            LiveView::Both => LiveView::X0Only,
        }
    }

    /// What the window's title bar says it is showing.
    pub fn caption(self) -> &'static str {
        match self {
            LiveView::X0Only => "x̂₀ (estimation)",
            LiveView::Both => "gauche: x_t (bruité)  |  droite: x̂₀ (estimation)",
        }
    }
}

/// A GPU-resident frame the visualiser renders while the sampler fills it.
///
/// The buffer is laid out as a **single wide image** — see [`LiveView`] for the
/// two layouts. The visualiser's shader is a plain row-major indexer
/// (`buf[(y * width + x) * channels + c]`), so a wide image is rendered side by
/// side with no shader change and no second window — the panes are just spans
/// of one tensor.
pub struct LiveFrame {
    gpu: Arc<GpuContext>,
    buffer: Arc<wgpu::Buffer>,
    /// Width of a single pane, in pixels.
    pane_width: u32,
    height: u32,
    channels: u32,
    view: LiveView,
    /// Scratch row-interleaved staging area, reused across steps so a 256-step
    /// run does not allocate 256 times.
    staging: Vec<f32>,
}

impl LiveFrame {
    /// Allocates the double-width frame for a `width × height × channels` model
    /// output. Contents start at zero, which the shader renders mid-grey.
    pub fn new(gpu: Arc<GpuContext>, width: u32, height: u32, channels: u32) -> Self {
        Self::with_view(gpu, width, height, channels, LiveView::Both)
    }

    /// The same, in the layout of `view`.
    ///
    /// The buffer is sized for the view, so switching views means a new
    /// `LiveFrame` and a re-registration with the visualiser — the window has
    /// to be told the new width anyway, and a buffer that could hold either
    /// would leave half of itself rendered as content in the narrow one.
    pub fn with_view(
        gpu: Arc<GpuContext>,
        width: u32,
        height: u32,
        channels: u32,
        view: LiveView,
    ) -> Self {
        let pane_width = width.max(1);
        let height = height.max(1);
        let channels = channels.max(1);
        let element_count = (view.frame_width(pane_width) * height * channels) as usize;
        let staging = vec![0.0f32; element_count];

        let buffer = gpu
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("live_denoise_frame"),
                contents: bytemuck::cast_slice::<f32, u8>(&staging),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        Self {
            gpu,
            buffer: Arc::new(buffer),
            pane_width,
            height,
            channels,
            view,
            staging,
        }
    }

    /// The buffer to hand to the visualiser as its render source.
    pub fn buffer(&self) -> Arc<wgpu::Buffer> {
        Arc::clone(&self.buffer)
    }

    /// Full width of the composed frame — what the visualiser must be told, not
    /// the model's own width.
    pub fn frame_width(&self) -> u32 {
        self.view.frame_width(self.pane_width)
    }

    pub fn view(&self) -> LiveView {
        self.view
    }

    pub fn frame_height(&self) -> u32 {
        self.height
    }

    pub fn channels(&self) -> u32 {
        self.channels
    }

    /// Uploads one denoising step: `latent` into the left pane, `x0_hat` into the
    /// right.
    ///
    /// Both slices are the model's own tensor layout (row-major, interleaved
    /// channels). Short or oversized slices are tolerated — the copy is clamped
    /// rather than panicking, because a display path must never be able to abort
    /// a sampling run.
    pub fn publish(&mut self, latent: &[f32], x0_hat: &[f32]) {
        compose_frame(
            &mut self.staging,
            latent,
            x0_hat,
            self.pane_width,
            self.height,
            self.channels,
            self.view,
        );

        // Queued on the same queue the visualiser renders from, so the next
        // frame it presents observes this write — no fence or readback needed.
        self.gpu.queue().write_buffer(
            &self.buffer,
            0,
            bytemuck::cast_slice::<f32, u8>(&self.staging),
        );
    }
}

/// Composes the frame the visualiser would display, as a plain CPU tensor of
/// `live_frame_width(width) × height × channels` — no GPU, no window.
///
/// The window is what users report on ("the image is cut in two"), so the
/// headless path can write out exactly what it would have shown, through the
/// same composition the live path uses. A screenshot proves what one screen
/// did; this proves what the buffer holds.
pub fn compose_live_frame(
    latent: &[f32],
    x0_hat: &[f32],
    width: u32,
    height: u32,
    channels: u32,
) -> Vec<f32> {
    compose_live_frame_view(latent, x0_hat, width, height, channels, LiveView::Both)
}

/// The same, in the layout of `view`.
pub fn compose_live_frame_view(
    latent: &[f32],
    x0_hat: &[f32],
    width: u32,
    height: u32,
    channels: u32,
    view: LiveView,
) -> Vec<f32> {
    let (width, height, channels) = (width.max(1), height.max(1), channels.max(1));
    let mut staging = vec![0.0f32; (view.frame_width(width) * height * channels) as usize];
    compose_frame(&mut staging, latent, x0_hat, width, height, channels, view);
    staging
}

/// Width of the frame [`compose_live_frame`] returns, for a given pane width.
pub fn live_frame_width(pane_width: u32) -> u32 {
    LiveView::Both.frame_width(pane_width)
}

/// Lays the two panes and the rule into `staging`, row-major with interleaved
/// channels — the exact indexing the shader performs.
///
/// Split out of [`LiveFrame::publish`] so it can be tested at all: the rest of
/// `LiveFrame` needs a GPU, this does not, and the layout is the part that can
/// be silently wrong. The previous mission's validation counted published
/// frames and sampled `latent[0]` / `x0[0]`; neither would have noticed a
/// stride that interleaved the panes instead of stacking them side by side.
///
/// Total in its inputs: writes every element of `staging` it owns, including
/// the rule, on every call.
fn compose_frame(
    staging: &mut [f32],
    latent: &[f32],
    x0_hat: &[f32],
    pane_width: u32,
    height: u32,
    channels: u32,
    view: LiveView,
) {
    let pane_stride = (pane_width * channels) as usize;
    if view == LiveView::X0Only {
        // One pane, no rule: the frame IS the estimate, row for row.
        for row in 0..height as usize {
            let dst = row * pane_stride;
            copy_row(&mut staging[dst..dst + pane_stride], x0_hat, dst);
        }
        return;
    }
    let rule_stride = SEPARATOR_COLUMNS.len() * channels as usize;
    let frame_stride = pane_stride * 2 + rule_stride;

    for row in 0..height as usize {
        let src = row * pane_stride;
        let dst = row * frame_stride;
        copy_row(&mut staging[dst..dst + pane_stride], latent, src);

        let rule = &mut staging[dst + pane_stride..dst + pane_stride + rule_stride];
        for (column, value) in SEPARATOR_COLUMNS.iter().enumerate() {
            // The rule is a column of the image, so it carries every channel —
            // a grey value on channel 0 alone would tint, not separate.
            rule[column * channels as usize..(column + 1) * channels as usize].fill(*value);
        }

        copy_row(
            &mut staging[dst + pane_stride + rule_stride..dst + frame_stride],
            x0_hat,
            src,
        );
    }
}

/// Copies one row of a pane, zero-filling whatever the source does not cover.
fn copy_row(dst: &mut [f32], src: &[f32], src_offset: usize) {
    // Clamp the offset before slicing: `&src[7..7]` panics on a 2-element slice
    // even though it asks for nothing, so an empty span still has to be taken
    // from within bounds.
    let start = src_offset.min(src.len());
    let available = (src.len() - start).min(dst.len());
    dst[..available].copy_from_slice(&src[start..start + available]);
    dst[available..].fill(0.0);
}

#[cfg(test)]
mod tests {
    use super::{LiveView, SEPARATOR_COLUMNS, compose_frame, copy_row};

    fn frame_width_of(pane_width: u32) -> u32 {
        LiveView::Both.frame_width(pane_width)
    }

    /// Composes a frame the way `publish` does, returning the staging buffer.
    fn compose(latent: &[f32], x0_hat: &[f32], w: u32, h: u32, c: u32) -> Vec<f32> {
        let mut staging = vec![f32::NAN; (frame_width_of(w) * h * c) as usize];
        compose_frame(&mut staging, latent, x0_hat, w, h, c, LiveView::Both);
        assert!(
            staging.iter().all(|v| !v.is_nan()),
            "compose_frame left part of the buffer unwritten"
        );
        staging
    }

    /// The layout claim, pinned element by element.
    ///
    /// Every source value is unique and encodes its own `(x, y)`, so any
    /// misindexing shows up as a wrong number rather than as a plausible
    /// image: a transposed row/column swaps 10y+x for 10x+y, a pane-stride
    /// off-by-one shifts the whole right pane, and interleaving the panes
    /// column-by-column (the "desynchronised" failure a user would suspect)
    /// scrambles both halves.
    #[test]
    fn compose_frame_lays_the_panes_out_row_major_around_the_rule() {
        let (w, h) = (4u32, 3u32);
        let latent: Vec<f32> = (0..h).flat_map(|y| (0..w).map(move |x| (10 * y + x) as f32)).collect();
        let x0_hat: Vec<f32> = latent.iter().map(|v| -v).collect();

        let staging = compose(&latent, &x0_hat, w, h, 1);

        let [a, b, c] = SEPARATOR_COLUMNS;
        assert_eq!(
            staging,
            vec![
                0.0, 1.0, 2.0, 3.0, a, b, c, -0.0, -1.0, -2.0, -3.0, //
                10.0, 11.0, 12.0, 13.0, a, b, c, -10.0, -11.0, -12.0, -13.0, //
                20.0, 21.0, 22.0, 23.0, a, b, c, -20.0, -21.0, -22.0, -23.0,
            ]
        );
    }

    /// The same claim read the way the eye reads it: a checkerboard on the
    /// left, a horizontal ramp on the right. If the buffer were interleaved or
    /// transposed, the checkerboard would smear and the ramp would run down
    /// instead of across — the two patterns fail in visibly different ways, so
    /// this is the test that says what the window should *look* like.
    #[test]
    fn a_checkerboard_and_a_ramp_survive_the_composition_side_by_side() {
        let (w, h) = (6u32, 4u32);
        let checker: Vec<f32> = (0..h)
            .flat_map(|y| (0..w).map(move |x| if (x + y) % 2 == 0 { 1.0 } else { -1.0 }))
            .collect();
        let ramp: Vec<f32> = (0..h)
            .flat_map(|_| (0..w).map(|x| x as f32 / (w - 1) as f32))
            .collect();

        let staging = compose(&checker, &ramp, w, h, 1);
        let stride = frame_width_of(w) as usize;
        let right = w as usize + SEPARATOR_COLUMNS.len();

        for y in 0..h as usize {
            let row = &staging[y * stride..(y + 1) * stride];

            // Left: still a checkerboard — alternating along x, phase flipping
            // with y. A transpose would keep it alternating, so the phase is
            // what carries the information here.
            for x in 0..w as usize {
                let expected = if (x + y) % 2 == 0 { 1.0 } else { -1.0 };
                assert_eq!(row[x], expected, "left pane at ({x}, {y})");
            }

            // Right: still a ramp — strictly increasing along x, identical on
            // every row. A transpose makes it constant along x instead.
            for x in 1..w as usize {
                assert!(
                    row[right + x] > row[right + x - 1],
                    "ramp not increasing at ({x}, {y})"
                );
            }
            assert_eq!(row[right], 0.0);
            assert_eq!(row[right + w as usize - 1], 1.0);
        }
    }

    /// The rule has to be an unbroken column on every row and every channel,
    /// or it reads as a dashed artefact — which would make the window look
    /// *more* broken, not less.
    #[test]
    fn the_rule_is_a_full_height_column_on_every_channel() {
        let (w, h, c) = (3u32, 5u32, 3u32);
        let pixels = (w * h * c) as usize;
        let latent = vec![0.5f32; pixels];
        let x0_hat = vec![-0.5f32; pixels];

        let staging = compose(&latent, &x0_hat, w, h, c);
        let stride = (frame_width_of(w) * c) as usize;

        for y in 0..h as usize {
            let row = &staging[y * stride..(y + 1) * stride];
            for (column, value) in SEPARATOR_COLUMNS.iter().enumerate() {
                let base = (w as usize + column) * c as usize;
                for channel in 0..c as usize {
                    assert_eq!(row[base + channel], *value, "rule at ({column}, {y}, {channel})");
                }
            }
            // …and the rule must not have eaten into either pane.
            assert!(row[..w as usize * c as usize].iter().all(|v| *v == 0.5));
            assert!(row[base_of_right_pane(w, c)..].iter().all(|v| *v == -0.5));
        }
    }

    fn base_of_right_pane(w: u32, c: u32) -> usize {
        (w as usize + SEPARATOR_COLUMNS.len()) * c as usize
    }

    /// The documented control for a wandering run: at t=0 the sampler's output
    /// *is* its x̂₀ estimate (`x0_estimate_is_the_x0_the_reverse_step_uses`),
    /// so the two panes must land on the same image. If they ever differ there,
    /// the split is real and not a display artefact — which is precisely the
    /// question a user asks when they see two dissimilar halves.
    #[test]
    fn identical_sources_produce_two_identical_panes() {
        let (w, h) = (5u32, 5u32);
        let image: Vec<f32> = (0..w * h).map(|i| (i as f32 / 7.0).sin()).collect();

        let staging = compose(&image, &image, w, h, 1);
        let stride = frame_width_of(w) as usize;
        let right = w as usize + SEPARATOR_COLUMNS.len();

        for y in 0..h as usize {
            let row = &staging[y * stride..(y + 1) * stride];
            assert_eq!(
                row[..w as usize],
                row[right..right + w as usize],
                "panes diverge on row {y} despite identical sources"
            );
        }
    }

    /// The single-pane view is the estimate and **nothing else**: no rule, no
    /// second half, and the frame is square when the image is.
    ///
    /// A pane written at the wrong stride would still look like a picture — a
    /// sheared one — so the values encode their own `(x, y)`: any misindexing
    /// shows up as a wrong number rather than as a plausible image.
    #[test]
    fn the_single_view_is_the_estimate_alone_at_the_images_own_size() {
        let (w, h) = (4u32, 4u32);
        let latent: Vec<f32> = (0..w * h).map(|i| -1000.0 - i as f32).collect();
        let x0_hat: Vec<f32> = (0..h)
            .flat_map(|y| (0..w).map(move |x| (10 * y + x) as f32))
            .collect();

        assert_eq!(LiveView::X0Only.frame_width(w), w, "the frame is one pane wide");
        let mut staging = vec![f32::NAN; (w * h) as usize];
        compose_frame(&mut staging, &latent, &x0_hat, w, h, 1, LiveView::X0Only);

        assert_eq!(staging, x0_hat, "the frame must be x̂₀, row for row");
        // The rule is a saturated black column flanked by two white ones; a
        // single-pane frame that still drew one would put a stripe through the
        // picture.
        assert!(
            !staging.windows(3).any(|w| w == SEPARATOR_COLUMNS),
            "the single view drew the separator rule"
        );
        // And nothing of the latent leaked in — every latent value is < -1000.
        assert!(staging.iter().all(|v| *v > -1.0));
    }

    /// `[x]` is a round trip, and the two widths differ by exactly the rule.
    #[test]
    fn toggling_the_view_swaps_the_two_layouts_and_nothing_else() {
        assert_eq!(LiveView::X0Only.toggle(), LiveView::Both);
        assert_eq!(LiveView::Both.toggle(), LiveView::X0Only);
        assert_eq!(LiveView::X0Only.toggle().toggle(), LiveView::X0Only);
        assert_eq!(
            LiveView::Both.frame_width(32),
            2 * LiveView::X0Only.frame_width(32) + SEPARATOR_COLUMNS.len() as u32
        );
        // Inference and the headless composer keep the double view: only the
        // perpetual path asked for a single one.
        assert_eq!(LiveView::default(), LiveView::Both);
        assert_eq!(super::live_frame_width(32), LiveView::Both.frame_width(32));
    }

    #[test]
    fn copy_row_transfers_the_requested_span() {
        let src = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut dst = vec![0.0; 3];
        copy_row(&mut dst, &src, 3);
        assert_eq!(dst, vec![4.0, 5.0, 6.0]);
    }

    /// A truncated tensor must leave the rest of the pane black instead of
    /// panicking mid-run: the visualiser is a spectator, never a failure mode
    /// for sampling.
    #[test]
    fn copy_row_zero_fills_beyond_a_short_source() {
        let src = vec![1.0, 2.0];
        let mut dst = vec![9.0; 4];
        copy_row(&mut dst, &src, 1);
        assert_eq!(dst, vec![2.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn copy_row_handles_offset_past_the_end() {
        let src = vec![1.0, 2.0];
        let mut dst = vec![9.0; 2];
        copy_row(&mut dst, &src, 7);
        assert_eq!(dst, vec![0.0, 0.0]);
    }
}
