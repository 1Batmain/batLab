use crate::model::types::{Dim3, PaddingMode};
use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone)]
pub enum ModelError {
    InvalidStride {
        stride: u32,
    },
    InvalidScaleFactor {
        scale_factor: u32,
    },
    InvalidGroupCount {
        channels: u32,
        groups: u32,
    },
    KernelLargerThanInput {
        input: Dim3,
        kernel: Dim3,
        mode: PaddingMode,
    },
    NoLayersToMark,
    DuplicateSavedOutput {
        key: String,
    },
    MissingSavedOutput {
        key: String,
    },
    DuplicateSavedGradient {
        key: String,
    },
    ConcatSpatialMismatch {
        input: Dim3,
        skip: Dim3,
    },
}

impl Display for ModelError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelError::InvalidStride { stride } => {
                write!(f, "invalid convolution stride: {}, expected > 0", stride)
            }
            ModelError::InvalidScaleFactor { scale_factor } => {
                write!(
                    f,
                    "invalid upsample scale factor: {}, expected > 0",
                    scale_factor
                )
            }
            ModelError::InvalidGroupCount { channels, groups } => {
                write!(
                    f,
                    "invalid group count: {groups} for {channels} channels, expected > 0 and to divide the channel count"
                )
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
            ModelError::NoLayersToMark => {
                write!(f, "cannot mark a saved output before adding a layer")
            }
            ModelError::DuplicateSavedOutput { key } => {
                write!(f, "saved output key '{key}' is already registered")
            }
            ModelError::MissingSavedOutput { key } => {
                write!(f, "saved output key '{key}' was not found")
            }
            ModelError::DuplicateSavedGradient { key } => {
                write!(
                    f,
                    "multiple skip gradients target the same saved output '{key}'"
                )
            }
            ModelError::ConcatSpatialMismatch { input, skip } => write!(
                f,
                "concat requires matching spatial dimensions: input=({}, {}, {}), skip=({}, {}, {})",
                input.x, input.y, input.z, skip.x, skip.y, skip.z
            ),
        }
    }
}

impl Error for ModelError {}
