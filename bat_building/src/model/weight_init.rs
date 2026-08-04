//! File purpose: weight-initialisation schemes for trainable layers.

use serde::{Deserialize, Serialize};

/// How `BufferInit::RandomWeights` buffers are filled at build time.
///
/// `Uniform` is the default so nothing changes unless a run asks for something
/// else: it is the historical `U(-0.1, 0.1)` draw, identical for every layer
/// regardless of its shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum WeightInit {
    /// `U(-0.1, 0.1)`, blind to the layer's fan-in.
    #[default]
    #[serde(rename = "uniform", alias = "Uniform")]
    Uniform,
    /// He / Kaiming normal: `N(0, 2 / fan_in)`, the variance that keeps the
    /// forward activation scale constant through a stack of ReLU-family layers.
    #[serde(rename = "he", alias = "He", alias = "kaiming")]
    He,
}

impl WeightInit {
    pub fn label(self) -> &'static str {
        match self {
            WeightInit::Uniform => "uniform",
            WeightInit::He => "he",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "uniform" => Some(WeightInit::Uniform),
            "he" | "kaiming" => Some(WeightInit::He),
            _ => None,
        }
    }
}

/// XorShift32 — the PRNG the uniform initialisation has always used, kept so
/// the default scheme reproduces its exact stream.
struct XorShift32(u32);

impl XorShift32 {
    fn new(seed: u32) -> Self {
        // 0 is the absorbing state of xorshift; never let a seed reach it.
        Self(if seed == 0 { 2463534242 } else { seed })
    }

    fn next_u32(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        self.0
    }

    /// Uniform in `(0, 1]` — never exactly 0, so `ln(u)` in Box–Muller is finite.
    fn next_unit(&mut self) -> f32 {
        (self.next_u32() as f64 + 1.0) as f32 / (u32::MAX as f64 + 1.0) as f32
    }
}

/// Historical draw: `U(-0.1, 0.1)` from a fixed seed.
///
/// The seed is deliberately NOT varied per layer — that is what the code has
/// always done, and changing it would silently move the baseline every run is
/// compared against.
pub(crate) fn uniform_weights(count: usize) -> Vec<f32> {
    let mut rng = XorShift32::new(2463534242);
    (0..count)
        .map(|_| (rng.next_u32() as f32 / u32::MAX as f32) * 0.2 - 0.1)
        .collect()
}

/// He normal: `N(0, sqrt(2 / fan_in))`, via Box–Muller on the same XorShift32.
///
/// `seed` varies per layer: with one shared stream every layer of the same size
/// would receive the *identical* weight matrix, which is exactly the degeneracy
/// a careful initialisation is supposed to avoid.
pub(crate) fn he_weights(count: usize, fan_in: u32, seed: u32) -> Vec<f32> {
    let std = (2.0 / fan_in.max(1) as f32).sqrt();
    let mut rng = XorShift32::new(seed);
    let mut out = Vec::with_capacity(count);
    while out.len() < count {
        let u1 = rng.next_unit();
        let u2 = rng.next_unit();
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = std::f32::consts::TAU * u2;
        out.push(radius * angle.cos() * std);
        if out.len() < count {
            out.push(radius * angle.sin() * std);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weight_init_defaults_to_uniform() {
        assert_eq!(WeightInit::default(), WeightInit::Uniform);
        assert_eq!(WeightInit::parse("He"), Some(WeightInit::He));
        assert_eq!(WeightInit::parse("xavier"), None);
    }

    /// The default path must be bit-identical to the historical
    /// `Layer::init_random_weights`, or every prior run becomes incomparable.
    #[test]
    fn uniform_reproduces_the_historical_stream() {
        let mut state: u32 = 2463534242;
        let historical: Vec<f32> = (0..64)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state as f32 / u32::MAX as f32) * 0.2 - 0.1
            })
            .collect();
        assert_eq!(uniform_weights(64), historical);
    }

    #[test]
    fn he_matches_its_target_standard_deviation() {
        for fan_in in [9u32, 144, 1152] {
            let values = he_weights(20_000, fan_in, 12345);
            let mean = values.iter().map(|&v| v as f64).sum::<f64>() / values.len() as f64;
            let var = values
                .iter()
                .map(|&v| (v as f64 - mean).powi(2))
                .sum::<f64>()
                / values.len() as f64;
            let want = 2.0 / fan_in as f64;
            assert!(
                mean.abs() < 0.02 * want.sqrt(),
                "fan_in {fan_in}: mean {mean}"
            );
            assert!(
                (var / want - 1.0).abs() < 0.05,
                "fan_in {fan_in}: variance {var} vs target {want}"
            );
        }
    }

    /// Deeper layers have a larger fan-in and must therefore start smaller —
    /// the whole point of the scheme, and what the flat ±0.1 draw ignores.
    #[test]
    fn he_scales_down_as_fan_in_grows() {
        let rms = |fan_in: u32| {
            let v = he_weights(8_000, fan_in, 7);
            (v.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
        };
        let shallow = rms(27); // 3×3×3
        let deep = rms(1152); // 3×3×128
        assert!(
            deep < shallow * 0.25,
            "a 1152-fan-in layer should start far smaller than a 27-fan-in one \
             ({deep} vs {shallow})"
        );
        // …and the uniform draw does not: same spread whatever the layer.
        let uniform_rms = {
            let v = uniform_weights(8_000);
            (v.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
        };
        assert!((uniform_rms - 0.1 / 3.0f64.sqrt()).abs() < 0.005);
    }

    #[test]
    fn he_streams_differ_between_layers() {
        assert_ne!(he_weights(32, 27, 1), he_weights(32, 27, 2));
    }

    #[test]
    fn he_requests_of_odd_length_are_exact() {
        assert_eq!(he_weights(7, 9, 3).len(), 7);
    }

    /// End-to-end through the build path: each convolution's weights must come
    /// out at its OWN fan-in's scale, and two stacked layers must not share a
    /// stream. Exercises the plumbing, not just the helper.
    #[test]
    fn built_model_applies_he_per_layer() {
        use crate::gpu_context::GpuContext;
        use crate::model::debug::read_back_f32;
        use crate::model::layer_types::LayerType;
        use crate::model::{ConvolutionType, Dim3, LayerTypes, LossMethod, Model, PaddingMode};
        use std::sync::Arc;

        pollster::block_on(async {
            let gpu = Arc::new(GpuContext::new_headless().await);

            let build = |gpu: Arc<GpuContext>, init: WeightInit| async move {
                let mut model = Model::new_training(gpu, 0.01, 1, LossMethod::MeanSquared).await;
                model.set_weight_init(init);
                // 3 in → 16 → 32 channels: fan_in 27 then 144.
                model
                    .add_layer(LayerTypes::Convolution(ConvolutionType::new(
                        Dim3::new((8, 8, 3)),
                        16,
                        Dim3::new((3, 3, 3)),
                        1,
                        PaddingMode::Same,
                    )))
                    .unwrap();
                model
                    .add_layer(LayerTypes::Convolution(ConvolutionType::new(
                        Dim3::default(),
                        32,
                        Dim3::new((3, 3, 16)),
                        1,
                        PaddingMode::Same,
                    )))
                    .unwrap();
                model.build().unwrap();
                model
            };

            let rms_of = |model: &Model<crate::model::Training>, layer: usize| -> f64 {
                let l = &model.layers[layer];
                let idx =
                    l.ty.get_optimizer_bindings()
                        .expect("conv is trainable")
                        .weights_forward_index;
                let buf = &l.buffers.forward[idx];
                let w = read_back_f32(model.gpu.as_ref(), buf, buf.size()).expect("readback");
                (w.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / w.len() as f64).sqrt()
            };

            let he = build(gpu.clone(), WeightInit::He).await;
            for (layer, fan_in) in [(0usize, 27.0f64), (1, 144.0)] {
                let want = (2.0 / fan_in).sqrt();
                let got = rms_of(&he, layer);
                assert!(
                    (got / want - 1.0).abs() < 0.1,
                    "layer {layer}: rms {got} vs he target {want}"
                );
            }

            // The default must be untouched: same flat spread at both depths.
            let uniform = build(gpu.clone(), WeightInit::Uniform).await;
            let flat = 0.1 / 3.0f64.sqrt();
            for layer in [0usize, 1] {
                let got = rms_of(&uniform, layer);
                assert!(
                    (got - flat).abs() < 0.005,
                    "layer {layer}: uniform rms {got} vs {flat}"
                );
            }
        });
    }
}
