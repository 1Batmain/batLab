use serde::{Deserialize, Serialize};

use crate::types::{ActivationMethod, Dim3, PaddingMode};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum LayerSpec {
    Convolution(ConvolutionLayerSpec),
    Activation(ActivationLayerSpec),
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ConvolutionLayerSpec {
    pub nb_kernel: u32,
    pub dim_kernel: Dim3,
    pub stripe: u32,
    pub mode: PaddingMode,
    pub dim_input: Option<Dim3>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ActivationLayerSpec {
    pub method: ActivationMethod,
    pub dim_input: Option<Dim3>,
}
