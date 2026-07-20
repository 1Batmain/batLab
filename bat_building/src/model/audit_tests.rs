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

// ---------------------------------------------------------------------------
// Finding #2 — the diffusion timestep is a deterministic function of the
// sample index, because both derive from the same linear counter
// `step * batch_size + batch_offset` (diffusion.rs:172-173).
//
// Consequence: a given image only ever sees `gcd(sample_count, schedule_len)`
// of the `schedule_len` timesteps, for the entire run. DDPM requires
// t ~ U{0, T-1} drawn INDEPENDENTLY of x_0.
// ---------------------------------------------------------------------------
#[test]
fn diffusion_timestep_is_decorrelated_from_sample_index() {
    use std::collections::HashSet;

    // CIFAR-10-like dataset against the project's 256-step schedule.
    let sample_count = 50_000usize;
    let schedule_len = 256usize;
    let batch_size = 16usize;

    // Replicates diffusion.rs::train_step_batch_inner exactly.
    let pair_for = |n: usize| {
        let step = n / batch_size;
        let batch_offset = n % batch_size;
        let sample_index = (step * batch_size + batch_offset) % sample_count;
        let diffusion_step = (step * batch_size + batch_offset) % schedule_len;
        (sample_index, diffusion_step)
    };

    // Walk 20 full epochs and collect every timestep sample #0 is ever paired with.
    let mut seen: HashSet<usize> = HashSet::new();
    for n in 0..(sample_count * 20) {
        let (sample_index, diffusion_step) = pair_for(n);
        if sample_index == 0 {
            seen.insert(diffusion_step);
        }
    }

    assert_eq!(
        seen.len(),
        schedule_len,
        "\nSample #0 only ever sees {} of the {schedule_len} timesteps, over 20 epochs.\n\
         Observed set: {:?}\n\
         The timestep is fully determined by the sample index: both are\n\
         `(step*batch_size + batch_offset) % _`. Coverage is capped at\n\
         gcd(sample_count, schedule_len) = {}.\n",
        seen.len(),
        {
            let mut v: Vec<_> = seen.iter().copied().collect();
            v.sort_unstable();
            v
        },
        {
            fn gcd(a: usize, b: usize) -> usize {
                if b == 0 { a } else { gcd(b, a % b) }
            }
            gcd(sample_count, schedule_len)
        }
    );
}

// ---------------------------------------------------------------------------
// Finding #3 — the linear schedule uses the DDPM betas (1e-4 .. 0.02), which
// are calibrated for T = 1000, at T = 256. The forward process therefore never
// reaches (approximately) pure noise, so q(x_T) != N(0, I) while the sampler
// starts from N(0, I).
// ---------------------------------------------------------------------------
#[test]
fn schedule_reaches_approximately_pure_noise_at_final_step() {
    use crate::training::LinearNoiseSchedule;

    let schedule = LinearNoiseSchedule::new_linear(256, 1e-4, 0.02);
    let terminal = schedule.alpha_bar(schedule.len() - 1);
    let residual_signal = terminal.sqrt();

    assert!(
        terminal < 1e-3,
        "\nalpha_bar(T-1) = {terminal:.6} (sqrt = {residual_signal:.4}).\n\
         x_T still retains {:.1}% of the original signal, so q(x_T) is far from\n\
         N(0, I) — yet sampling starts from pure Gaussian noise.\n\
         Reference: DDPM at T=1000 with these betas gives alpha_bar_T = 4.0e-5.\n",
        residual_signal * 100.0
    );
}
