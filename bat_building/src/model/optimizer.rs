//! File purpose: optimiser selection and hyperparameters shared by the model graph and the configs.

use serde::{Deserialize, Serialize};

/// Which weight-update rule the per-layer optimiser pass runs.
///
/// `Sgd` is the default so existing configs and checkpoints keep their exact
/// prior behaviour: the enum is `#[serde(default)]`-friendly and old config
/// files (which have no `optimizer` field) deserialise to `Sgd`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum OptimizerKind {
    #[default]
    #[serde(rename = "sgd", alias = "Sgd", alias = "SGD")]
    Sgd,
    #[serde(rename = "adam", alias = "Adam", alias = "ADAM")]
    Adam,
}

impl OptimizerKind {
    pub fn label(self) -> &'static str {
        match self {
            OptimizerKind::Sgd => "sgd",
            OptimizerKind::Adam => "adam",
        }
    }

    /// Parses the `--optimizer` CLI value / config string.
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "sgd" => Some(OptimizerKind::Sgd),
            "adam" => Some(OptimizerKind::Adam),
            _ => None,
        }
    }

    /// Number of extra f32 state slots the optimiser keeps per trainable scalar
    /// (Adam keeps the first and second moment: 2).
    pub fn state_slots_per_parameter(self) -> u64 {
        match self {
            OptimizerKind::Sgd => 0,
            OptimizerKind::Adam => 2,
        }
    }
}

/// Adam hyperparameters (Kingma & Ba 2015 defaults).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdamHyperparameters {
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
}

impl Default for AdamHyperparameters {
    fn default() -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }
}

impl AdamHyperparameters {
    /// Bias-correction denominators `(1 - β1^t, 1 - β2^t)` for a 1-based step
    /// counter `t`.
    ///
    /// Computed on the CPU in f64 and passed to the shader as a uniform: `t`
    /// grows without bound over a run and `pow(β, t)` in f32 loses the little
    /// precision that matters exactly where the correction stops mattering,
    /// while the CPU side has the step counter anyway.
    pub fn bias_corrections(&self, t: u64) -> (f32, f32) {
        let t = t.max(1) as i32;
        let bc1 = 1.0 - (self.beta1 as f64).powi(t);
        let bc2 = 1.0 - (self.beta2 as f64).powi(t);
        // A correction of exactly 0 can only come from β=0; clamp so the shader
        // never divides by zero.
        (
            bc1.max(f64::MIN_POSITIVE) as f32,
            bc2.max(f64::MIN_POSITIVE) as f32,
        )
    }
}

/// Uniform layout shared with `shader/adam.wgsl` (std140-compatible: 8 f32 = 32 bytes).
#[derive(Debug, Clone, Copy)]
pub(crate) struct AdamSpecs {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    /// Multiplied into the accumulated gradient before the moments are updated.
    ///
    /// SGD folds the batch mean into the learning rate (`lr / batch_size`)
    /// because its update is linear in the gradient. Adam's is not — it is
    /// (very nearly) invariant to a rescaling of `g`, so scaling `lr` would
    /// leave the effective step unchanged. The batch mean must therefore be
    /// taken on the gradient itself.
    pub grad_scale: f32,
    /// `1 - β1^t`
    pub bias_correction1: f32,
    /// `1 - β2^t`
    pub bias_correction2: f32,
}

impl AdamSpecs {
    pub(crate) const BYTES: usize = 32;

    pub(crate) fn to_bytes(self) -> [u8; Self::BYTES] {
        let mut out = [0u8; Self::BYTES];
        let fields = [
            self.lr,
            self.beta1,
            self.beta2,
            self.eps,
            self.grad_scale,
            self.bias_correction1,
            self.bias_correction2,
            0.0,
        ];
        for (i, value) in fields.iter().enumerate() {
            out[i * 4..(i + 1) * 4].copy_from_slice(&value.to_le_bytes());
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optimizer_kind_defaults_to_sgd() {
        assert_eq!(OptimizerKind::default(), OptimizerKind::Sgd);
    }

    #[test]
    fn optimizer_kind_parses_case_insensitively() {
        assert_eq!(OptimizerKind::parse("Adam"), Some(OptimizerKind::Adam));
        assert_eq!(OptimizerKind::parse(" SGD "), Some(OptimizerKind::Sgd));
        assert_eq!(OptimizerKind::parse("rmsprop"), None);
    }

    /// A config written before the optimiser was selectable must still load,
    /// and must load as SGD.
    #[test]
    fn missing_optimizer_field_deserialises_to_sgd() {
        #[derive(serde::Deserialize)]
        struct Cfg {
            #[serde(default)]
            optimizer: OptimizerKind,
        }
        let cfg: Cfg = serde_json::from_str("{}").unwrap();
        assert_eq!(cfg.optimizer, OptimizerKind::Sgd);
        let cfg: Cfg = serde_json::from_str(r#"{"optimizer":"adam"}"#).unwrap();
        assert_eq!(cfg.optimizer, OptimizerKind::Adam);
    }

    #[test]
    fn bias_corrections_match_closed_form() {
        let hp = AdamHyperparameters::default();
        for t in [1u64, 2, 5, 50, 1000] {
            let (bc1, bc2) = hp.bias_corrections(t);
            // The reference uses the betas as they are actually stored (f32),
            // promoted to f64 — 0.999 is not exactly representable in f32 and
            // the gap compounds over a thousand steps.
            let expect1 = 1.0 - (hp.beta1 as f64).powi(t as i32);
            let expect2 = 1.0 - (hp.beta2 as f64).powi(t as i32);
            assert!(
                (bc1 as f64 - expect1).abs() < 1e-7,
                "t={t}: bc1 {bc1} vs {expect1}"
            );
            assert!(
                (bc2 as f64 - expect2).abs() < 1e-7,
                "t={t}: bc2 {bc2} vs {expect2}"
            );
            // …and still within f32 noise of the ideal hyperparameters.
            assert!((bc2 as f64 - (1.0 - 0.999f64.powi(t as i32))).abs() < 1e-4);
        }
        // t=1 is the strongest correction: m̂ = m / 0.1, v̂ = v / 0.001.
        let (bc1, bc2) = hp.bias_corrections(1);
        assert!((bc1 - 0.1).abs() < 1e-6);
        assert!((bc2 - 0.001).abs() < 1e-6);
    }

    #[test]
    fn adam_specs_pack_to_32_bytes_in_declaration_order() {
        let specs = AdamSpecs {
            lr: 1.0,
            beta1: 2.0,
            beta2: 3.0,
            eps: 4.0,
            grad_scale: 5.0,
            bias_correction1: 6.0,
            bias_correction2: 7.0,
        };
        let bytes = specs.to_bytes();
        assert_eq!(bytes.len(), 32);
        let decoded: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(decoded, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.0]);
    }
}
