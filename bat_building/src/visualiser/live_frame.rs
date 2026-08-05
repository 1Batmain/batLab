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

/// A GPU-resident frame the visualiser renders while the sampler fills it.
///
/// The buffer is laid out as a **single image twice as wide** as the model's
/// output: the current latent x_t on the left, the model's clipped x0 estimate
/// on the right. The visualiser's shader is a plain row-major indexer
/// (`buf[(y * width + x) * channels + c]`), so a double-width image is rendered
/// side by side with no shader change and no second window — the panes are just
/// two halves of one tensor.
///
/// ```text
///   width = 2 * W                     one row of the buffer
///   ┌───────────────┬───────────────┐  ┌──────────┬──────────┐
///   │               │               │  │  x_t row │ x0_hat   │
///   │      x_t      │    x0_hat     │  │  (W px)  │ row (W)  │
///   │   (noisy)     │ (what the     │  └──────────┴──────────┘
///   │               │  model sees)  │
///   └───────────────┴───────────────┘
/// ```
pub struct LiveFrame {
    gpu: Arc<GpuContext>,
    buffer: Arc<wgpu::Buffer>,
    /// Width of a single pane, in pixels (the buffer is twice this).
    pane_width: u32,
    height: u32,
    channels: u32,
    /// Scratch row-interleaved staging area, reused across steps so a 256-step
    /// run does not allocate 256 times.
    staging: Vec<f32>,
}

impl LiveFrame {
    /// Allocates the double-width frame for a `width × height × channels` model
    /// output. Contents start at zero, which the shader renders mid-grey.
    pub fn new(gpu: Arc<GpuContext>, width: u32, height: u32, channels: u32) -> Self {
        let pane_width = width.max(1);
        let height = height.max(1);
        let channels = channels.max(1);
        let element_count = (pane_width * 2 * height * channels) as usize;
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
        self.pane_width * 2
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
        let pane_stride = (self.pane_width * self.channels) as usize;
        let frame_stride = pane_stride * 2;

        for row in 0..self.height as usize {
            let src = row * pane_stride;
            let dst = row * frame_stride;
            copy_row(&mut self.staging[dst..dst + pane_stride], latent, src);
            copy_row(
                &mut self.staging[dst + pane_stride..dst + frame_stride],
                x0_hat,
                src,
            );
        }

        // Queued on the same queue the visualiser renders from, so the next
        // frame it presents observes this write — no fence or readback needed.
        self.gpu.queue().write_buffer(
            &self.buffer,
            0,
            bytemuck::cast_slice::<f32, u8>(&self.staging),
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
    use super::copy_row;

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
