use crate::model::types::{Dim3, PaddingMode};
use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, Copy)]
pub enum ModelError {
    InvalidStride {
        stride: u32,
    },
    KernelLargerThanInput {
        input: Dim3,
        kernel: Dim3,
        mode: PaddingMode,
    },
}

impl Display for ModelError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelError::InvalidStride { stride } => {
                write!(f, "invalid convolution stride: {}, expected > 0", stride)
            }
            ModelError::KernelLargerThanInput {
                input,
                kernel,
                mode,
            } => write!(
                f,
                "kernel larger than input for {:?} convolution: input=({}, {}, {}), kernel=({}, {}, {})",
                mode, input.x, input.y, input.z, kernel.x, kernel.y, kernel.z
            ),
        }
    }
}

impl Error for ModelError {}
