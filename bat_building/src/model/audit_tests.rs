//! File purpose: Diagnostic tests written during the training-pipeline audit.
//!
//! These tests are NOT regression guards for current behaviour — several of
//! them are expected to FAIL against the implementation as it stands. Each
//! failure IS the finding; see `AUDIT_TRAINING.md` at the repo root.

use crate::gpu_context::GpuContext;
use crate::model::{ConvolutionType, Dim3, Infer, LayerTypes, Model, PaddingMode};
use std::sync::Arc;

/// CPU reference: single-channel 3x3 "same" convolution, stride 1, zero padding.
fn reference_same_conv3x3(input: &[f32], h: usize, w: usize, kernel: &[f32], bias: f32) -> Vec<f32> {
    let pad = 1i32;
    let mut out = vec![0.0f32; h * w];
    for oy in 0..h as i32 {
        for ox in 0..w as i32 {
            let mut sum = bias;
            for ky in 0..3i32 {
                for kx in 0..3i32 {
                    let iy = oy + ky - pad;
                    let ix = ox + kx - pad;
                    if iy < 0 || iy >= h as i32 || ix < 0 || ix >= w as i32 {
                        continue; // zero padding
                    }
                    sum += input[iy as usize * w + ix as usize] * kernel[(ky * 3 + kx) as usize];
                }
            }
            out[oy as usize * w + ox as usize] = sum;
        }
    }
    out
}

/// Finding #1 — `convolution.wgsl` never applies the padding offset, so
/// `PaddingMode::Same` produces a shifted (and out-of-bounds-clamped)
/// convolution instead of a centered, zero-padded one.
///
/// With a kernel whose only non-zero tap is the CENTER, a correct "same"
/// convolution is the identity. Anything else proves the spatial mapping is
/// wrong.
#[test]
fn same_padding_conv_is_centered_and_zero_padded() {
    pollster::block_on(async {
        let gpu = Arc::new(GpuContext::new_headless().await);
        let (h, w) = (4usize, 4usize);

        let mut model: Model<Infer> = Model::new(gpu.clone()).await;
        model
            .add_layer(LayerTypes::Convolution(ConvolutionType::new(
                Dim3::new((h as u32, w as u32, 1)),
                1,
                Dim3::new((3, 3, 1)),
                1,
                PaddingMode::Same,
            )))
            .unwrap();
        model.build_model().unwrap();

        // Only the center tap is 1 -> a correct same-conv is the identity.
        let kernel = vec![
            0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0,
        ];
        let layer = model.layers.first().unwrap();
        gpu.queue.write_buffer(
            layer.buffers.forward[1].as_ref(),
            0,
            bytemuck::cast_slice(&kernel),
        );
        gpu.queue
            .write_buffer(layer.buffers.forward[2].as_ref(), 0, &0.0f32.to_le_bytes());

        let input: Vec<f32> = (0..(h * w)).map(|i| (i + 1) as f32).collect();
        let got = model.predict(&input);
        let expected = reference_same_conv3x3(&input, h, w, &kernel, 0.0);

        assert_eq!(
            got, expected,
            "\nSame-padding conv is neither centered nor zero-padded.\n\
             input:    {input:?}\n\
             expected: {expected:?}  (identity: only the center tap is 1)\n\
             got:      {got:?}\n"
        );
    });
}
