use crate::model::error::ModelError;
use crate::model::layer_types::LayerType;
use crate::model::types::{BufferSpec, Dim3};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OptimizerMethod {
    Sgd,
}

#[derive(Debug, Clone)]
pub struct OptimizerType {
    pub method: OptimizerMethod,
    pub dim_input: Dim3,
    pub dim_output: Dim3,
}

impl OptimizerType {
    pub fn new(method: OptimizerMethod, dim_input: Dim3) -> Self {
        Self {
            method,
            dim_input,
            dim_output: dim_input,
        }
    }
}

impl LayerType for OptimizerType {
    fn get_entrypoint(&self) -> &str {
        match self.method {
            OptimizerMethod::Sgd => "sgd",
        }
    }
    fn get_byte_weights(&self) -> u32 {
        todo!();
    }
    fn get_dim_input(&self) -> Dim3 {
        self.dim_input
    }
    fn get_dim_output(&self) -> Dim3 {
        self.dim_output
    }
    fn set_dim_input(&mut self, input: Dim3) {
        self.dim_input = input;
    }
    fn set_dim_output(&mut self) -> Result<Dim3, ModelError> {
        self.dim_output = self.dim_input;
        Ok(self.dim_output)
    }

    fn get_buffers_specs(&self) -> Vec<(String, BufferSpec)> {
        todo!();
    }

    fn get_spec_uniform_bytes_size(&self) -> u32 {
        todo!();
    }

    fn get_spec_uniform_bytes(&self) -> Vec<u8> {
        todo!();
    }
    fn get_back_buffers_specs(&self) -> Vec<(String, BufferSpec)> {
        todo!();
    }
}
