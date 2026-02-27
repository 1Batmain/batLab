use encase::ShaderType;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PaddingMode {
    Valid,
    Same,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Optimizer {
    Sgd,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ActivationMethod {
    Relu,
    Linear,
}

#[derive(ShaderType, Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct Dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
    pub _padding: u32,
}

impl Dim3 {
    pub fn new(i: (u32, u32, u32)) -> Self {
        Self { x: i.0, y: i.1, z: i.2, _padding: 0 }
    }

    pub fn length(&self) -> u32 {
        self.x * self.y * self.z
    }
}

/// Uniform struct for convolution layer parameters.
/// Maps to WGSL uniform buffer binding.
#[derive(ShaderType, Debug, Clone, Copy)]
pub struct ConvolutionSpecUniform {
    pub dim_input: Dim3,
    pub stride: u32,
    pub padding_mode: u32,  // 0 = Valid, 1 = Same
}
