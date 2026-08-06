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
    /// How many times a chunk has been uploaded to the GPU.
    ///
    /// Each load is a `queue.write_buffer` of the WHOLE chunk (64 MiB on this
    /// machine), so this counter is the single most expensive thing the dataset
    /// does. It exists because the cost is invisible otherwise: nothing in the
    /// training loop mentions it, and a shuffled sampler makes it depend on the
    /// batch size and on the dataset/chunk ratio rather than on anything the
    /// caller wrote.
    chunk_loads: u64,
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
            chunk_loads: 0,
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

    /// Whole-chunk uploads performed so far. See the field's comment.
    pub fn chunk_loads(&self) -> u64 {
        self.chunk_loads
    }

    /// Samples one chunk holds — the quantity that decides, together with the
    /// dataset size, how much of the above a shuffled batch costs.
    pub fn chunk_sample_capacity(&self) -> usize {
        self.chunk_sample_capacity
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

    /// Copy a whole batch into `destination`, sample `i` of the list landing at
    /// slot `i` (offset `i * sample_len` floats).
    ///
    /// This is NOT `copy_sample_to` in a loop into one encoder, and the
    /// difference is a correctness one. `ensure_chunk_loaded` uploads through
    /// `queue.write_buffer`, and wgpu applies every pending queue write BEFORE
    /// the command buffers submitted after it. Two samples from two different
    /// chunks encoded into a single encoder would therefore both read the chunk
    /// loaded last: the first copy would silently pick up the wrong image, with
    /// no validation error and no crash — just a batch quietly trained on
    /// duplicated data.
    ///
    /// So the batch is grouped by chunk and each resident chunk gets its own
    /// submission. A dataset that fits in one chunk — the common case — is one
    /// group and one submission, which is also the point of the exercise.
    pub fn copy_samples_to(
        &mut self,
        gpu: &GpuContext,
        sample_indices: &[usize],
        destination: &wgpu::Buffer,
    ) -> Result<(), GpuDatasetError> {
        for &sample_index in sample_indices {
            if sample_index >= self.sample_count {
                return Err(GpuDatasetError::SampleIndexOutOfBounds {
                    sample_index,
                    sample_count: self.sample_count,
                });
            }
        }
        let sample_bytes = (self.sample_len * std::mem::size_of::<f32>()) as u64;
        let capacity = self.chunk_sample_capacity;

        // Slot order is preserved: the grouping is by chunk, but every copy
        // still writes to the destination slot of its own position.
        let mut pending: Vec<(usize, usize)> = sample_indices
            .iter()
            .copied()
            .enumerate()
            .map(|(slot, sample_index)| (slot, sample_index))
            .collect();
        pending.sort_by_key(|(_, sample_index)| sample_index / capacity);

        let mut cursor = 0usize;
        while cursor < pending.len() {
            let chunk = pending[cursor].1 / capacity;
            let end = pending[cursor..]
                .iter()
                .position(|(_, s)| s / capacity != chunk)
                .map(|offset| cursor + offset)
                .unwrap_or(pending.len());

            self.ensure_chunk_loaded(gpu, pending[cursor].1);
            let chunk_start = self.loaded_chunk_start.unwrap_or(0);
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            for &(slot, sample_index) in &pending[cursor..end] {
                let local_index = sample_index.saturating_sub(chunk_start);
                encoder.copy_buffer_to_buffer(
                    &self.chunk_buffer,
                    local_index as u64 * sample_bytes,
                    destination,
                    slot as u64 * sample_bytes,
                    sample_bytes,
                );
            }
            gpu.queue.submit([encoder.finish()]);
            cursor = end;
        }
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
        self.chunk_loads += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// A batch must upload each chunk it needs AT MOST ONCE.
    ///
    /// `copy_sample_to` in a loop did not guarantee this: with a shuffled
    /// sampler, consecutive samples of a batch land in unrelated chunks, so the
    /// old path re-uploaded a whole chunk on most samples. On CIFAR-10 grey
    /// (50 000 samples of 4 KiB, 64 MiB chunks => 4 chunks) a batch of 16 costs
    /// ~12.25 whole-chunk uploads that way, against ~3.96 when grouped — and a
    /// chunk upload is 64 MiB of host-to-device traffic, i.e. by far the most
    /// expensive thing the dataset does.
    #[test]
    fn a_batch_uploads_each_chunk_at_most_once() {
        pollster::block_on(async {
            let gpu = Arc::new(crate::gpu_context::GpuContext::new_headless().await);
            let sample_len = 8;
            let samples: Vec<Vec<f32>> = (0..32)
                .map(|s| (0..sample_len).map(|i| (s * sample_len + i) as f32).collect())
                .collect();
            let mut dataset = GpuDataset::from_samples(gpu.as_ref(), samples, sample_len).unwrap();
            let destination = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("test_batch_destination"),
                size: (8 * sample_len * std::mem::size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            // Deliberately out of order and spread out: the grouping must not
            // depend on the indices arriving sorted.
            let batch = [31usize, 3, 17, 0, 28, 9, 22, 5];
            dataset
                .copy_samples_to(gpu.as_ref(), &batch, &destination)
                .unwrap();
            let after_first = dataset.chunk_loads();
            assert!(
                after_first <= batch.len().div_ceil(dataset.chunk_sample_capacity()).max(1) as u64,
                "a batch of {} uploaded {after_first} chunks",
                batch.len()
            );

            // A second batch over already-resident data must upload nothing.
            dataset
                .copy_samples_to(gpu.as_ref(), &batch, &destination)
                .unwrap();
            assert_eq!(
                dataset.chunk_loads(),
                after_first,
                "a second batch over a resident chunk re-uploaded it"
            );
        });
    }

    /// Slot order follows the CALLER's order, not the chunk grouping.
    #[test]
    fn samples_land_in_the_slot_they_were_asked_for() {
        pollster::block_on(async {
            let gpu = Arc::new(crate::gpu_context::GpuContext::new_headless().await);
            let sample_len = 4;
            let samples: Vec<Vec<f32>> = (0..16)
                .map(|s| vec![s as f32; sample_len])
                .collect();
            let mut dataset = GpuDataset::from_samples(gpu.as_ref(), samples, sample_len).unwrap();
            let batch = [11usize, 2, 7];
            let destination = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("test_slot_destination"),
                size: (batch.len() * sample_len * std::mem::size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            dataset
                .copy_samples_to(gpu.as_ref(), &batch, &destination)
                .unwrap();

            let values = crate::model::debug::read_back_f32(
                gpu.as_ref(),
                &destination,
                destination.size(),
            )
            .unwrap();
            for (slot, &sample_index) in batch.iter().enumerate() {
                for offset in 0..sample_len {
                    assert_eq!(
                        values[slot * sample_len + offset],
                        sample_index as f32,
                        "slot {slot} does not hold sample {sample_index}"
                    );
                }
            }
        });
    }
}
