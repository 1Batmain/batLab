//! File purpose: the exponential moving average of the weights — its decay, its
//! warmup, and the uniform the EMA pass reads.
//!
//! # Why the weights are averaged at all
//!
//! A diffusion model is sampled from, not evaluated: what matters is not the
//! loss of the last SGD iterate but the quality of the images the weights
//! produce. The last iterate is a point bouncing inside a basin at the scale of
//! the learning rate; the average of the recent iterates sits nearer the middle
//! of that basin. DDPM (Ho et al. 2020) generates from an EMA of decay 0.9999
//! and every implementation since has kept the trick, because it costs one
//! buffer and one fused multiply-add per weight.
//!
//! # The warmup, and why it is not optional
//!
//! `ema ← d·ema + (1−d)·w` with a decay of 0.999 has a time constant of a
//! thousand steps. Started at the random initialisation, it would still be
//! mostly random noise a thousand steps in — the average would be *worse* than
//! the raw weights over the whole early run.
//!
//! Two mechanisms, both applied here, remove that:
//!
//! 1. the shadow is **seeded with the weights themselves** when the pass is
//!    built (see `Layer::create_ema_pass`), so it never holds anything the model
//!    did not hold first;
//! 2. the decay is **ramped**: `d(t) = min(decay, (1+t)/(10+t))`.
//!
//! The ramp is TensorFlow's `ExponentialMovingAverage(num_updates=t)` rule, and
//! it is what this file implements. It is preferred to the bias-correction form
//! (`ema / (1 − dᵗ)`, which is Adam's trick applied to a zero-initialised
//! shadow) for one reason: bias correction assumes the shadow started at zero,
//! which would forbid seeding it with the weights — and a shadow that starts at
//! zero is a shadow whose first checkpoint is unusable. The ramp composes with
//! the seeding instead of contradicting it.
//!
//! Read the ramp as "average over the last ~10 steps at t=10, the last ~100 at
//! t≈1000": `d(1) = 0.182`, `d(10) = 0.55`, `d(100) = 0.918`, `d(1000) = 0.991`,
//! and the nominal 0.999 takes over at t ≈ 9990.

use serde::{Deserialize, Serialize};

/// The EMA of a run: one decay, and the warmup that gets it there.
///
/// `None` on a model means no EMA at all — no buffers, no dispatch, and a
/// checkpoint that is byte-identical to what the same run produced before this
/// file existed.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct EmaConfig {
    /// The nominal decay, the value the ramp converges to. 0.999 is the usual
    /// choice for runs of a few thousand steps; DDPM's 0.9999 only makes sense
    /// past ~100 k steps, where its 10 000-step time constant fits in the run.
    pub decay: f32,
}

impl EmaConfig {
    /// Rejects a decay that would make the average meaningless.
    ///
    /// `1.0` freezes the shadow at the initial weights for ever and `0.0` makes
    /// it a copy of the weights — both are silently useless rather than loudly
    /// wrong, which is exactly the kind of setting that costs a night of GPU.
    pub fn parse(value: &str) -> Option<Self> {
        let decay: f32 = value.trim().parse().ok()?;
        Self::new(decay)
    }

    pub fn new(decay: f32) -> Option<Self> {
        if decay.is_finite() && decay > 0.0 && decay < 1.0 {
            Some(Self { decay })
        } else {
            None
        }
    }

    /// The decay actually applied at the 1-based step `t`: `min(decay, (1+t)/(10+t))`.
    ///
    /// Computed in f64 for the same reason Adam's bias corrections are — the
    /// ratio is a hair below 1 for most of a run, and the shader has no business
    /// deriving it from a step counter it would have to be handed anyway.
    pub fn effective_decay(&self, t: u64) -> f32 {
        let t = t.max(1) as f64;
        let ramp = (1.0 + t) / (10.0 + t);
        ramp.min(self.decay as f64) as f32
    }
}

/// Uniform layout shared with `shader/ema.wgsl` (16 bytes: one f32 + padding).
#[derive(Debug, Clone, Copy)]
pub(crate) struct EmaSpecs {
    pub decay: f32,
}

impl EmaSpecs {
    pub(crate) const BYTES: usize = 16;

    pub(crate) fn to_bytes(self) -> [u8; Self::BYTES] {
        let mut out = [0u8; Self::BYTES];
        out[0..4].copy_from_slice(&self.decay.to_le_bytes());
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_decay_outside_the_open_unit_interval_is_refused() {
        assert_eq!(EmaConfig::parse("0.999"), Some(EmaConfig { decay: 0.999 }));
        assert_eq!(EmaConfig::parse(" 0.9 "), Some(EmaConfig { decay: 0.9 }));
        assert_eq!(EmaConfig::parse("1.0"), None);
        assert_eq!(EmaConfig::parse("0"), None);
        assert_eq!(EmaConfig::parse("-0.5"), None);
        assert_eq!(EmaConfig::parse("nope"), None);
        assert_eq!(EmaConfig::parse("nan"), None);
    }

    /// The warmup numbers quoted in this file's documentation, checked against
    /// the closed form. A ramp that drifts from its documentation is a ramp
    /// nobody can reason about at 3 a.m.
    #[test]
    fn the_warmup_ramp_matches_its_closed_form() {
        let ema = EmaConfig { decay: 0.999 };
        for (t, expected) in [
            (1u64, 2.0 / 11.0),
            (10, 11.0 / 20.0),
            (100, 101.0 / 110.0),
            (1000, 1001.0 / 1010.0),
        ] {
            let got = ema.effective_decay(t);
            assert!(
                (got as f64 - expected).abs() < 1e-6,
                "t={t}: {got} vs {expected}"
            );
        }
        // Past the crossover the nominal decay takes over and never moves again.
        assert_eq!(ema.effective_decay(100_000), 0.999);
        assert_eq!(ema.effective_decay(u64::MAX), 0.999);
        // t=0 must not divide by anything or read as "no averaging at all":
        // the counter is 1-based at the first update.
        assert_eq!(ema.effective_decay(0), ema.effective_decay(1));
    }

    /// The ramp is monotone: a step never averages LESS than the step before
    /// it. An off-by-one that flipped the ratio would still look plausible at a
    /// single point.
    #[test]
    fn the_warmup_ramp_only_ever_rises() {
        let ema = EmaConfig { decay: 0.99 };
        let mut previous = 0.0f32;
        for t in 1..5000u64 {
            let d = ema.effective_decay(t);
            assert!(d >= previous, "t={t}: {d} < {previous}");
            assert!(d <= 0.99, "t={t}: {d} above the nominal decay");
            previous = d;
        }
    }

    #[test]
    fn ema_specs_pack_to_sixteen_bytes() {
        let bytes = EmaSpecs { decay: 0.5 }.to_bytes();
        assert_eq!(bytes.len(), 16);
        assert_eq!(f32::from_le_bytes(bytes[0..4].try_into().unwrap()), 0.5);
        assert!(bytes[4..].iter().all(|b| *b == 0));
    }
}
