use serde::{Deserialize, Serialize};

use crate::model::types::{ActivationMethod, Dim3, Optimizer, PaddingMode};

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub(crate) enum SavedLayerArchitecture {
    Convolution {
        nb_kernel: u32,
        dim_kernel: Dim3,
        stripe: u32,
        mode: PaddingMode,
        dim_input: Dim3,
        dim_output: Dim3,
    },
    Activation {
        method: ActivationMethod,
        dim_input: Dim3,
        dim_output: Dim3,
    },
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct SavedLayer {
    pub(crate) architecture: SavedLayerArchitecture,
    pub(crate) weights: Vec<f32>,
    pub(crate) bias: Vec<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct SavedTrainingSpec {
    pub(crate) lr: f32,
    pub(crate) batch_size: u32,
    pub(crate) optimizer: Optimizer,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct SavedModel {
    pub(crate) version: u32,
    pub(crate) training: Option<SavedTrainingSpec>,
    pub(crate) layers: Vec<SavedLayer>,
}
