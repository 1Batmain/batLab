//! File purpose: Implements dataset logic used by the training pipeline.

use crate::gpu_context::GpuContext;
use std::error::Error;
use std::fmt;

const DEFAULT_CHUNK_BYTES: usize = 64 * 1024 * 1024;
const MAX_DYNAMIC_CHUNK_BYTES: usize = 512 * 1024 * 1024;
const GPU_MEMORY_CHUNK_FRACTION: u64 = 8;

#[derive(Debug, Clone)]
pub struct GpuDataset {
    samples: Vec<Vec<f32>>,
    chunk_buffer: wgpu::Buffer,
    chunk_sample_capacity: usize,
    loaded_chunk_start: Option<usize>,
    loaded_chunk_count: usize,
    staging_cpu: Vec<f32>,
    sample_count: usize,
    sample_len: usize,
}

#[derive(Debug, Clone)]
pub enum GpuDatasetError {
    EmptyDataset,
    InvalidSampleLength {
        expected: usize,
        actual: usize,
        sample_index: usize,
    },
    InvalidFlatLength {
        total: usize,
        sample_len: usize,
    },
    SampleTooLarge {
        sample_len: usize,
        max_chunk_bytes: usize,
    },
    SampleIndexOutOfBounds {
        sample_index: usize,
        sample_count: usize,
    },
}

impl fmt::Display for GpuDatasetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuDatasetError::EmptyDataset => write!(f, "dataset must contain at least one sample"),
            GpuDatasetError::InvalidSampleLength {
                expected,
                actual,
                sample_index,
            } => write!(
                f,
                "invalid sample length at index {sample_index}: expected {expected}, got {actual}"
            ),
            GpuDatasetError::InvalidFlatLength { total, sample_len } => write!(
                f,
                "flat dataset length {total} is not divisible by sample length {sample_len}"
            ),
            GpuDatasetError::SampleTooLarge {
                sample_len,
                max_chunk_bytes,
            } => write!(
                f,
                "single sample ({sample_len} f32) exceeds max chunk size {max_chunk_bytes} bytes"
            ),
            GpuDatasetError::SampleIndexOutOfBounds {
                sample_index,
                sample_count,
            } => write!(
                f,
                "sample index {sample_index} out of bounds for dataset with {sample_count} samples"
            ),
        }
    }
}

impl Error for GpuDatasetError {}

impl GpuDataset {
    fn select_max_chunk_bytes(gpu: &GpuContext, dataset_total_bytes: usize) -> usize {
        let limits = gpu.device.limits();
        let binding_cap = limits.max_storage_buffer_binding_size as u64;
        let buffer_cap = limits.max_buffer_size;
        let gpu_cap = gpu.specs().memory_size();
        let hard_cap = buffer_cap
            .min(binding_cap)
            .min(gpu_cap)
            .min(MAX_DYNAMIC_CHUNK_BYTES as u64) as usize;

        let target_from_gpu = (gpu_cap / GPU_MEMORY_CHUNK_FRACTION) as usize;
        let target = target_from_gpu
            .max(DEFAULT_CHUNK_BYTES)
            .min(hard_cap.max(1));
        target.min(dataset_total_bytes.max(1))
    }

    pub fn from_samples(
        gpu: &GpuContext,
        samples: Vec<Vec<f32>>,
        sample_len: usize,
    ) -> Result<Self, GpuDatasetError> {
        if samples.is_empty() {
            return Err(GpuDatasetError::EmptyDataset);
        }
        if sample_len == 0 {
            return Err(GpuDatasetError::InvalidFlatLength {
                total: 0,
                sample_len,
            });
        }
        for (sample_index, sample) in samples.iter().enumerate() {
            if sample.len() != sample_len {
                return Err(GpuDatasetError::InvalidSampleLength {
                    expected: sample_len,
                    actual: sample.len(),
                    sample_index,
                });
            }
        }
        let sample_count = samples.len();
        let sample_bytes = sample_len * std::mem::size_of::<f32>();
        let dataset_total_bytes = sample_count.saturating_mul(sample_bytes);
        let max_chunk_bytes = Self::select_max_chunk_bytes(gpu, dataset_total_bytes);
        if sample_bytes > max_chunk_bytes {
            return Err(GpuDatasetError::SampleTooLarge {
                sample_len,
                max_chunk_bytes,
            });
        }
        let chunk_sample_capacity = (max_chunk_bytes / sample_bytes).max(1);
        let chunk_bytes = (chunk_sample_capacity * sample_bytes) as u64;
        let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("training_dataset_chunk"),
            size: chunk_bytes.max(4),
            usage: wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        Ok(Self {
            samples,
            chunk_buffer: buffer,
            chunk_sample_capacity,
            loaded_chunk_start: None,
            loaded_chunk_count: 0,
            staging_cpu: Vec::new(),
            sample_count,
            sample_len,
        })
    }

    pub fn sample_count(&self) -> usize {
        self.sample_count
    }

    pub fn sample_len(&self) -> usize {
        self.sample_len
    }

    pub fn gpu_buffer_bytes(&self) -> u64 {
        self.chunk_buffer.size()
    }

    pub fn copy_sample_to(
        &mut self,
        gpu: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        sample_index: usize,
        destination: &wgpu::Buffer,
    ) -> Result<(), GpuDatasetError> {
        if sample_index >= self.sample_count {
            return Err(GpuDatasetError::SampleIndexOutOfBounds {
                sample_index,
                sample_count: self.sample_count,
            });
        }
        self.ensure_chunk_loaded(gpu, sample_index);
        let sample_bytes = (self.sample_len * std::mem::size_of::<f32>()) as u64;
        let local_index = sample_index.saturating_sub(self.loaded_chunk_start.unwrap_or(0));
        let source_offset = local_index as u64 * sample_bytes;
        encoder.copy_buffer_to_buffer(
            &self.chunk_buffer,
            source_offset,
            destination,
            0,
            sample_bytes,
        );
        Ok(())
    }

    fn ensure_chunk_loaded(&mut self, gpu: &GpuContext, sample_index: usize) {
        if let Some(start) = self.loaded_chunk_start {
            let end = start + self.loaded_chunk_count;
            if (start..end).contains(&sample_index) {
                return;
            }
        }

        let chunk_start = (sample_index / self.chunk_sample_capacity) * self.chunk_sample_capacity;
        let chunk_end = (chunk_start + self.chunk_sample_capacity).min(self.sample_count);
        let chunk_count = chunk_end - chunk_start;

        self.staging_cpu.clear();
        self.staging_cpu.reserve(chunk_count * self.sample_len);
        for sample in &self.samples[chunk_start..chunk_end] {
            self.staging_cpu.extend_from_slice(sample);
        }
        gpu.queue.write_buffer(
            &self.chunk_buffer,
            0,
            bytemuck::cast_slice(&self.staging_cpu),
        );
        self.loaded_chunk_start = Some(chunk_start);
        self.loaded_chunk_count = chunk_count;
    }
}
