//! File purpose: Implements types functionality for model execution, state, or diagnostics.

use encase::ShaderType;
use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
pub enum PaddingMode {
    #[default]
    Valid,
    Same,
}

#[derive(ShaderType, Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct Dim3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
    pub _padding: u32,
}

impl std::ops::Mul<u32> for Dim3 {
    type Output = Dim3;

    fn mul(self, fct: u32) -> Dim3 {
        Dim3 {
            x: self.x * fct,
            y: self.y * fct,
            z: self.z * fct,
            _padding: self._padding,
        }
    }
}

impl Dim3 {
    pub fn new(i: (u32, u32, u32)) -> Self {
        Self {
            x: i.0,
            y: i.1,
            z: i.2,
            _padding: 0,
        }
    }

    pub fn bytes_size(&self) -> u32 {
        self.x * self.y * self.z * std::mem::size_of::<u32>() as u32
    }

    pub fn length(&self) -> u32 {
        self.x * self.y * self.z
    }
}

#[derive(Debug, Clone)]
pub(crate) struct BufferSpec {
    pub size: u32,
    pub usage: wgpu::BufferUsages,
    pub visibility: wgpu::ShaderStages,
    pub ty: wgpu::BindingType,
}
