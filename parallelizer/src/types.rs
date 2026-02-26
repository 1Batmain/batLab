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

#[derive(ShaderType, Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Dim3 {
    pub fn new() -> Self {
        Self { x: 0, y: 0, z: 0 }
    }

    pub fn length(&self) -> u32 {
        self.x * self.y * self.z
    }
}
