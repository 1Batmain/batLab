//! File purpose: entry point for the whole diffusion machinery — training
//! (`dataset`, `diffusion`, `weighting`, `Trainer`/`TrainingTask`) AND inference
//! (`sampler`, `drift`, `perpetual`), over a shared `schedule` and the `metrics`
//! instrumentation. The folder is named `training/` for history only; the
//! inference path lives here too, and by design — see
//! `docs/reports/AUDIT_SIMPLIFICATION.md` §T1 for why the sampler stays here and
//! why the *directory* (not this or that file) is the misnomer. Wires submodules
//! and shared exports.

pub mod dataset;
pub mod diffusion;
pub mod drift;
pub mod eval;
pub mod metrics;
pub mod perpetual;
pub mod sampler;
pub mod schedule;
pub mod weighting;

use crate::model::{Dim3, Model};
use std::error::Error;
use std::fmt;

pub use dataset::{DatasetPayload, GpuDataset, GpuDatasetError, decode_u8};
pub use diffusion::DiffusionTask;
pub use eval::{BaselineBucket, EvalConfig, EvalReport, MseBucket, evaluate};
pub use drift::{AsyncNoisePredictor, DriftFrame, DriftWalk, NoisePredictor};
pub use metrics::{
    BucketStat, DenoiseStepStat, MetricsLogger, ProbeConfig, Stats, log_probe, log_train_loss,
    log_trajectory, probe_diffusion,
};
pub use sampler::{
    BASE_NOISE_FOLD, DenoiseFrame, ReverseStep, base_noise_seed, compose_diffusion_input,
    path_seed, predict_epsilon, reverse_step, reverse_step_async, reverse_step_from_epsilon,
    reverse_step_seed, sample_diffusion,
};
pub use perpetual::{
    CLIMB_TEMPO_RATIO, DriftAction, DriftPhase, MIN_FLUX_LEVEL, MIN_RENOISE_DEPTH, PerpetualDrift,
    PerpetualOrigin, PerpetualRegime, RENOISE_DEPTH_STEP,
};
pub use schedule::{LinearNoiseSchedule, PosteriorVariance};
pub use weighting::{DEFAULT_SNR_GAMMA, LossWeighting};

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
