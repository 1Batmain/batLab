//! File purpose: Module entry point for training; wires submodules and shared exports.

pub mod dataset;
pub mod diffusion;
pub mod schedule;

use crate::model::{Dim3, Model};
use std::error::Error;
use std::fmt;

pub use dataset::{GpuDataset, GpuDatasetError};
pub use diffusion::DiffusionTask;
pub use schedule::LinearNoiseSchedule;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Workgroups {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl Workgroups {
    pub const fn x(x: u32) -> Self {
        Self { x, y: 1, z: 1 }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskPassSpec {
    pub label: &'static str,
    pub entrypoint: &'static str,
    pub workgroups: Workgroups,
}

#[derive(Debug, Clone)]
pub enum TrainingTaskError {
    EmptyModel,
    TargetLengthMismatch {
        expected: usize,
        actual: usize,
    },
    InvalidBatchSize {
        batch_size: usize,
    },
    DatasetError {
        message: String,
    },
    InvalidLayout {
        input: Dim3,
        output: Dim3,
        message: &'static str,
    },
}

impl fmt::Display for TrainingTaskError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TrainingTaskError::EmptyModel => write!(f, "cannot configure task on an empty model"),
            TrainingTaskError::TargetLengthMismatch { expected, actual } => write!(
                f,
                "invalid clean target length: expected {expected}, got {actual}"
            ),
            TrainingTaskError::InvalidBatchSize { batch_size } => {
                write!(f, "invalid batch size: {batch_size} (must be > 0)")
            }
            TrainingTaskError::DatasetError { message } => {
                write!(f, "dataset error: {message}")
            }
            TrainingTaskError::InvalidLayout {
                input,
                output,
                message,
            } => write!(
                f,
                "{message} (input={}x{}x{}, output={}x{}x{})",
                input.x, input.y, input.z, output.x, output.y, output.z
            ),
        }
    }
}

impl Error for TrainingTaskError {}

pub trait TrainingTask {
    fn name(&self) -> &'static str;
    fn configure(&mut self, input: Dim3, output: Dim3) -> Result<(), TrainingTaskError>;
    fn pass_specs(&self) -> &[TaskPassSpec];
}

pub struct Trainer<T: TrainingTask> {
    task: T,
}

impl<T: TrainingTask> Trainer<T> {
    pub fn new(task: T) -> Self {
        Self { task }
    }

    pub fn task(&self) -> &T {
        &self.task
    }

    pub fn task_mut(&mut self) -> &mut T {
        &mut self.task
    }

    pub fn configure_for_model<State>(
        &mut self,
        model: &Model<State>,
    ) -> Result<(), TrainingTaskError> {
        let input = model.input_dim().ok_or(TrainingTaskError::EmptyModel)?;
        let output = model.output_dim().ok_or(TrainingTaskError::EmptyModel)?;
        self.task.configure(input, output)
    }
}
