//! File purpose: Implements optimizer functionality for model execution, state, or diagnostics.

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Optimizer {
    Sgd,
}
