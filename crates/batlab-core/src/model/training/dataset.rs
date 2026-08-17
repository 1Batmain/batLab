//! GpuDataset: chunking, residency and the 8-bit → f32 decode path.

use crate::gpu_context::GpuContext;
use std::error::Error;
use std::fmt;

const DEFAULT_CHUNK_BYTES: usize = 64 * 1024 * 1024;
/// Share of GPU capacity a STREAMED chunk may take — small, because a streamed
/// chunk is re-uploaded on every miss (a recurring cost).
const GPU_MEMORY_CHUNK_FRACTION: u64 = 8;
/// Share of GPU capacity a RESIDENT dataset may take — twice the streaming
/// fraction, for the opposite reason: a dataset that fits is uploaded once and
/// removes the traffic. On this machine the two are 3.5 and 7 GiB; ImageNet 32×32
/// in 8-bit (3.94 GB) lands between them — streamed under one rule, resident under
/// this one, which is the point.
const GPU_MEMORY_RESIDENT_FRACTION: u64 = 4;

/// What the dataset holds on the host, and therefore what crosses to the GPU. The
/// two variants are the two `.batraw` encodings, kept apart to the device instead
/// of normalised to f32 on load — the whole of BATRAW3: an 8-bit corpus stays
/// 8-bit until a shader widens it, so file, host RAM and chunk upload are a quarter.
#[derive(Debug, Clone)]
pub enum DatasetPayload {
    /// One `Vec<f32>` per sample. Every dataset that predates BATRAW3, plus any
    /// whose geometry had to be resampled on the way in.
    Floats(Vec<Vec<f32>>),
    /// `sample_count * sample_len` bytes, flat, sample `i` starting at
    /// `i * sample_len`. Widened on the GPU by `shader/dataset_decode.wgsl`.
    Bytes(Vec<u8>),
}

/// The `[0,255] → [-1,1]` widening — exact (256 inputs, no rounding), which is why
/// BATRAW3 is lossless and `to_u8` is its inverse. The GPU builds a 256-entry table
/// from THIS and is pinned to it by `the_gpu_decode_agrees_with_the_cpu_one`.
pub fn decode_u8(value: u8) -> f32 {
    value as f32 / 127.5 - 1.0
}

impl DatasetPayload {
    /// Samples the payload holds, given the length one sample has.
    pub fn sample_count(&self, sample_len: usize) -> usize {
        match self {
            DatasetPayload::Floats(samples) => samples.len(),
            DatasetPayload::Bytes(bytes) => {
                if sample_len == 0 {
                    0
                } else {
                    bytes.len() / sample_len
                }
            }
        }
    }

    /// Bytes one sample occupies — 4× the value count for f32, 1× for u8.
    ///
    /// This is the number that decides how many samples a chunk holds, so it is
    /// also the number that makes BATRAW3 cheaper: same chunk, four times the
    /// residency, a quarter of the misses.
    pub fn sample_bytes(&self, sample_len: usize) -> usize {
        match self {
            DatasetPayload::Floats(_) => sample_len * std::mem::size_of::<f32>(),
            DatasetPayload::Bytes(_) => sample_len,
        }
    }

    /// One sample as f32, decoding if it has to.
    ///
    /// For the callers that genuinely need CPU-side pixels — the metrics probe
    /// and the perpetual seed image — and for nothing else: the training path
    /// never materialises a sample on the host.
    pub fn sample_f32(&self, index: usize, sample_len: usize) -> Option<Vec<f32>> {
        match self {
            DatasetPayload::Floats(samples) => samples.get(index).cloned(),
            DatasetPayload::Bytes(bytes) => {
                let start = index.checked_mul(sample_len)?;
                let end = start.checked_add(sample_len)?;
                bytes
                    .get(start..end)
                    .map(|slice| slice.iter().copied().map(decode_u8).collect())
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct GpuDataset {
    payload: DatasetPayload,
    chunk_buffer: wgpu::Buffer,
    chunk_sample_capacity: usize,
    loaded_chunk_start: Option<usize>,
    loaded_chunk_count: usize,
    staging_cpu: Vec<u8>,
    /// The decode pipeline, built on the first 8-bit copy and never for an f32
    /// dataset: an old-format run allocates nothing new and takes the same path
    /// it always did.
    decode: Option<DecodePass>,
    sample_count: usize,
    sample_len: usize,
    /// How many times a chunk has been uploaded to the GPU.
    ///
    /// Each load is a `write_buffer` of the WHOLE chunk (64 MiB), the dataset's
    /// single most expensive act — counted because it is otherwise invisible (a
    /// shuffled sampler makes it depend on batch and dataset/chunk ratio, nothing
    /// the caller wrote).
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

/// Bytes of one dataset chunk on a device with these caps. Pure and public: the
/// inventory calls the SAME function ([`crate::resources::plan_dataset`]) so the
/// prediction cannot drift from the allocation. Two cases:
///
/// - RESIDENT — the whole dataset fits one allocatable buffer within the residency
///   budget: take all of it, one upload for the run's life (reachable for CIFAR
///   and ImageNet 32×32 under native limits, not the WebGPU baseline).
/// - STREAMED — it does not fit: size to the streaming budget, a shuffled batch
///   pays for the chunks it lands in (what the browser still gets).
///
/// `.min(binding_cap)` matters as much as the buffer cap — the chunk is a storage
/// binding for the decode shader, so a binding limit is a residency limit (the one
/// that binds on web).
pub fn select_chunk_bytes(
    binding_cap: u64,
    buffer_cap: u64,
    gpu_cap: u64,
    dataset_total_bytes: u64,
) -> u64 {
    let dataset = dataset_total_bytes.max(1);
    let allocatable = buffer_cap.min(binding_cap).min(gpu_cap).max(1);

    let resident_budget = (gpu_cap / GPU_MEMORY_RESIDENT_FRACTION).max(DEFAULT_CHUNK_BYTES as u64);
    if dataset <= allocatable.min(resident_budget) {
        return dataset;
    }

    let streaming_budget = (gpu_cap / GPU_MEMORY_CHUNK_FRACTION).max(DEFAULT_CHUNK_BYTES as u64);
    streaming_budget.min(allocatable).min(dataset)
}

impl GpuDataset {
    fn select_max_chunk_bytes(gpu: &GpuContext, dataset_total_bytes: usize) -> usize {
        let limits = gpu.device.limits();
        select_chunk_bytes(
            limits.max_storage_buffer_binding_size as u64,
            limits.max_buffer_size,
            gpu.specs().memory_size(),
            dataset_total_bytes as u64,
        ) as usize
    }

    /// A dataset of f32 samples — every caller that predates BATRAW3.
    ///
    /// Byte for byte the path it always was: [`DatasetPayload::Floats`] is
    /// copied straight from the chunk to the batch slot, no shader involved.
    pub fn from_samples(
        gpu: &GpuContext,
        samples: Vec<Vec<f32>>,
        sample_len: usize,
    ) -> Result<Self, GpuDatasetError> {
        for (sample_index, sample) in samples.iter().enumerate() {
            if sample.len() != sample_len {
                return Err(GpuDatasetError::InvalidSampleLength {
                    expected: sample_len,
                    actual: sample.len(),
                    sample_index,
                });
            }
        }
        Self::from_payload(gpu, DatasetPayload::Floats(samples), sample_len)
    }

    /// A dataset in whichever encoding it arrived in.
    ///
    /// The encoding decides one number — [`DatasetPayload::sample_bytes`] — and
    /// everything else follows from it: how many samples a chunk holds, how
    /// often a shuffled batch misses, how many bytes each miss costs. An 8-bit
    /// corpus gets four times the residency out of the same chunk.
    pub fn from_payload(
        gpu: &GpuContext,
        payload: DatasetPayload,
        sample_len: usize,
    ) -> Result<Self, GpuDatasetError> {
        if sample_len == 0 {
            return Err(GpuDatasetError::InvalidFlatLength {
                total: 0,
                sample_len,
            });
        }
        if let DatasetPayload::Bytes(bytes) = &payload
            && !bytes.len().is_multiple_of(sample_len)
        {
            return Err(GpuDatasetError::InvalidFlatLength {
                total: bytes.len(),
                sample_len,
            });
        }
        let sample_count = payload.sample_count(sample_len);
        if sample_count == 0 {
            return Err(GpuDatasetError::EmptyDataset);
        }
        let sample_bytes = payload.sample_bytes(sample_len);
        let dataset_total_bytes = sample_count.saturating_mul(sample_bytes);
        let max_chunk_bytes = Self::select_max_chunk_bytes(gpu, dataset_total_bytes);
        if sample_bytes > max_chunk_bytes {
            return Err(GpuDatasetError::SampleTooLarge {
                sample_len,
                max_chunk_bytes,
            });
        }
        let chunk_sample_capacity = (max_chunk_bytes / sample_bytes).max(1);
        // Rounded up to a word: `write_buffer` refuses a size that is not a
        // multiple of 4, and the decode shader reads the chunk as `array<u32>`.
        // An 8-bit sample whose length is not a multiple of 4 (a 3-value test
        // fixture, a 5×5 greyscale image) would otherwise be rejected by the
        // driver rather than by us.
        let chunk_bytes = ((chunk_sample_capacity * sample_bytes) as u64).next_multiple_of(4);
        let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("training_dataset_chunk"),
            size: chunk_bytes.max(4),
            usage: wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        Ok(Self {
            payload,
            chunk_buffer: buffer,
            chunk_sample_capacity,
            loaded_chunk_start: None,
            loaded_chunk_count: 0,
            staging_cpu: Vec::new(),
            decode: None,
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

    /// The whole dataset's size in its payload encoding — what would cross the
    /// boundary if it were uploaded in one go, and what it does cross once when
    /// it is resident.
    pub fn payload_bytes(&self) -> u64 {
        (self.sample_count * self.payload.sample_bytes(self.sample_len)) as u64
    }

    /// Chunks the dataset is cut into. `1` means resident: one upload for the
    /// life of the run, and no dataset traffic per step afterwards.
    pub fn chunk_count(&self) -> usize {
        self.sample_count
            .div_ceil(self.chunk_sample_capacity.max(1))
            .max(1)
    }

    /// Whether the whole dataset sits on the GPU at once.
    pub fn is_resident(&self) -> bool {
        self.chunk_count() == 1
    }

    /// Copy a whole batch into `destination`, sample `i` at slot `i`. NOT one copy
    /// per sample into one encoder — a correctness point: `write_buffer` uploads
    /// apply BEFORE the command buffers after them, so two samples from two chunks
    /// in one encoder would both read the chunk loaded last (a silent wrong image).
    /// So the batch is grouped by chunk, one submission each; a single-chunk
    /// dataset (the common case) is one group. An f32 chunk is copied
    /// buffer-to-buffer, an 8-bit one widened by a compute pass; the grouping is shared.
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
            let group: Vec<(usize, usize)> = pending[cursor..end]
                .iter()
                .map(|&(slot, sample_index)| (slot, sample_index - chunk_start))
                .collect();

            match self.payload {
                DatasetPayload::Floats(_) => {
                    let sample_bytes = (self.sample_len * std::mem::size_of::<f32>()) as u64;
                    let mut encoder = gpu.device.create_command_encoder(&Default::default());
                    for &(slot, local_index) in &group {
                        encoder.copy_buffer_to_buffer(
                            &self.chunk_buffer,
                            local_index as u64 * sample_bytes,
                            destination,
                            slot as u64 * sample_bytes,
                            sample_bytes,
                        );
                    }
                    gpu.submit([encoder.finish()]);
                }
                DatasetPayload::Bytes(_) => self.decode_group(gpu, &group, destination),
            }
            cursor = end;
        }
        Ok(())
    }

    /// Widen one chunk's worth of 8-bit samples into their batch slots.
    ///
    /// `group` is `(destination slot, index inside the resident chunk)`, which
    /// is exactly the pair the shader indexes by — see `dataset_decode.wgsl`.
    fn decode_group(
        &mut self,
        gpu: &GpuContext,
        group: &[(usize, usize)],
        destination: &wgpu::Buffer,
    ) {
        let sample_len = self.sample_len;
        let pass = self
            .decode
            .get_or_insert_with(|| DecodePass::new(gpu, group.len().max(1)));
        pass.ensure_plan_capacity(gpu, group.len());

        let plan: Vec<u32> = group
            .iter()
            .flat_map(|&(slot, local_index)| [local_index as u32, slot as u32])
            .collect();
        gpu.write_buffer(&pass.plan, 0, bytemuck::cast_slice(&plan));
        gpu.write_buffer(
            &pass.spec,
            0,
            bytemuck::cast_slice(&[sample_len as u32, group.len() as u32, 0u32, 0u32]),
        );

        // The bind group is rebuilt rather than cached because `destination`
        // belongs to the caller — the diffusion prepass owns it and rebuilds it
        // whenever the batch changes. One bind group per chunk group per step,
        // next to a forward and a backward pass, is not a cost worth a cache
        // keyed on a buffer identity that could go stale.
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dataset_decode_bg"),
            layout: &pass.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.chunk_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: pass.plan.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: destination.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: pass.spec.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: pass.palette.as_entire_binding(),
                },
            ],
        });

        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        let threads = (sample_len * group.len()) as u32;
        let workgroups = threads.div_ceil(64);
        gpu.compute_pass(
            &mut encoder,
            workgroups,
            || "dataset · decode_u8".to_string(),
            |compute| {
                compute.set_pipeline(&pass.pipeline);
                compute.set_bind_group(0, &bind_group, &[]);
                let (x, y) = crate::model::layer::dispatch_grid(workgroups);
                compute.dispatch_workgroups(x, y, 1);
            },
        );
        gpu.submit([encoder.finish()]);
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
        match &self.payload {
            DatasetPayload::Floats(samples) => {
                self.staging_cpu
                    .reserve(chunk_count * self.sample_len * std::mem::size_of::<f32>());
                for sample in &samples[chunk_start..chunk_end] {
                    self.staging_cpu
                        .extend_from_slice(bytemuck::cast_slice(sample));
                }
            }
            DatasetPayload::Bytes(bytes) => {
                let from = chunk_start * self.sample_len;
                let to = chunk_end * self.sample_len;
                self.staging_cpu.extend_from_slice(&bytes[from..to]);
            }
        }
        // `write_buffer` refuses a size that is not a multiple of 4. The tail
        // padding is never read: the shader only addresses bytes below
        // `chunk_count * sample_len`.
        while !self.staging_cpu.len().is_multiple_of(4) {
            self.staging_cpu.push(0);
        }
        gpu.write_buffer(&self.chunk_buffer, 0, &self.staging_cpu);
        self.loaded_chunk_start = Some(chunk_start);
        self.loaded_chunk_count = chunk_count;
        self.chunk_loads += 1;
    }
}

/// The compute pass that widens an 8-bit chunk, built once per dataset.
///
/// Only the plan and the spec are rewritten per call; the pipeline and its
/// layout outlive every batch.
#[derive(Debug, Clone)]
struct DecodePass {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    /// `(local index, destination slot)` per sample of the group.
    plan: wgpu::Buffer,
    /// `sample_len`, `pair_count`, and two words of padding to reach the 16-byte
    /// minimum a uniform binding has.
    spec: wgpu::Buffer,
    /// The 256 decoded values. Written once, read by every dispatch — see the
    /// comment on binding 4 in `dataset_decode.wgsl` for why the shader looks
    /// them up instead of computing them.
    palette: wgpu::Buffer,
    plan_pairs: usize,
}

impl DecodePass {
    fn new(gpu: &GpuContext, pairs: usize) -> Self {
        let device = &gpu.device;
        let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dataset_decode_bgl"),
            entries: &[
                storage(0, true),
                storage(1, true),
                storage(2, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage(4, true),
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("dataset_decode"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "shader/dataset_decode.wgsl"
            ))),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("dataset_decode_layout"),
            bind_group_layouts: &[&layout],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("dataset_decode_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("dataset_decode"),
            cache: None,
            compilation_options: Default::default(),
        });
        let spec = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dataset_decode_spec"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let palette = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dataset_decode_palette"),
            size: 256 * std::mem::size_of::<f32>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let values: Vec<f32> = (0..=255u8).map(decode_u8).collect();
        gpu.write_buffer(&palette, 0, bytemuck::cast_slice(&values));

        Self {
            pipeline,
            layout,
            plan: Self::plan_buffer(gpu, pairs),
            spec,
            palette,
            plan_pairs: pairs,
        }
    }

    fn plan_buffer(gpu: &GpuContext, pairs: usize) -> wgpu::Buffer {
        gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dataset_decode_plan"),
            size: (pairs.max(1) * 2 * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn ensure_plan_capacity(&mut self, gpu: &GpuContext, pairs: usize) {
        if pairs > self.plan_pairs {
            self.plan = Self::plan_buffer(gpu, pairs);
            self.plan_pairs = pairs;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// A batch must upload each chunk it needs AT MOST ONCE.
    ///
    /// One copy per sample did not guarantee this: with a shuffled
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

    fn batch_destination(gpu: &GpuContext, floats: usize) -> wgpu::Buffer {
        gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test_decode_destination"),
            size: (floats * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        })
    }

    /// The GPU widening and the CPU one must be the SAME function.
    ///
    /// They are written twice — once in WGSL for the training path, once in Rust
    /// for the probe and the seed image — and a disagreement between them is a
    /// brightness shift applied to half the pipeline, with nothing to signal it.
    /// So all 256 byte values go through both and are compared exactly, not
    /// within a tolerance: `v / 127.5 - 1` is representable in f32 on both sides
    /// and there is no reason for the results to merely be close.
    #[test]
    fn the_gpu_decode_agrees_with_the_cpu_one() {
        pollster::block_on(async {
            let gpu = Arc::new(crate::gpu_context::GpuContext::new_headless().await);
            let sample_len = 256;
            // Sample 0 is 0..=255 in order, sample 1 the reverse: between them
            // every value is decoded at an even and at an odd byte offset, which
            // is what the shader's shift-and-mask has to get right.
            let mut bytes: Vec<u8> = (0..=255u8).collect();
            bytes.extend((0..=255u8).rev());
            let mut dataset =
                GpuDataset::from_payload(gpu.as_ref(), DatasetPayload::Bytes(bytes), sample_len)
                    .unwrap();
            assert_eq!(dataset.sample_count(), 2);

            let destination = batch_destination(gpu.as_ref(), 2 * sample_len);
            dataset
                .copy_samples_to(gpu.as_ref(), &[0, 1], &destination)
                .unwrap();
            let values =
                crate::model::debug::read_back_f32(gpu.as_ref(), &destination, destination.size())
                    .unwrap();

            for byte in 0..=255u8 {
                assert_eq!(
                    values[byte as usize],
                    decode_u8(byte),
                    "the GPU decoded {byte} differently from decode_u8"
                );
                assert_eq!(
                    values[sample_len + 255 - byte as usize],
                    decode_u8(byte),
                    "the GPU decoded {byte} differently at an odd offset"
                );
            }
            // The convention itself, spelled out where it is easy to check
            // against the loader: black is -1, white is +1.
            assert_eq!(values[0], -1.0);
            assert_eq!(values[255], 1.0);
        });
    }

    /// Slot order survives the decode path too.
    ///
    /// The f32 path preserves it with a copy offset; the 8-bit path preserves it
    /// with the `.y` of a plan entry, which is a second implementation of the
    /// same promise and therefore a second thing that can be wrong.
    #[test]
    fn an_8_bit_batch_lands_in_the_slot_it_was_asked_for() {
        pollster::block_on(async {
            let gpu = Arc::new(crate::gpu_context::GpuContext::new_headless().await);
            let sample_len = 4;
            let bytes: Vec<u8> = (0..16u8)
                .flat_map(|s| std::iter::repeat_n(s, sample_len))
                .collect();
            let mut dataset =
                GpuDataset::from_payload(gpu.as_ref(), DatasetPayload::Bytes(bytes), sample_len)
                    .unwrap();
            let batch = [11usize, 2, 7];
            let destination = batch_destination(gpu.as_ref(), batch.len() * sample_len);
            dataset
                .copy_samples_to(gpu.as_ref(), &batch, &destination)
                .unwrap();

            let values =
                crate::model::debug::read_back_f32(gpu.as_ref(), &destination, destination.size())
                    .unwrap();
            for (slot, &sample_index) in batch.iter().enumerate() {
                for offset in 0..sample_len {
                    assert_eq!(
                        values[slot * sample_len + offset],
                        decode_u8(sample_index as u8),
                        "slot {slot} does not hold sample {sample_index}"
                    );
                }
            }
        });
    }

    /// The point of the format, as a number.
    ///
    /// Same device rule, same images, four times the residency — so a quarter of
    /// the chunks, so a quarter of the misses. Stated on the pure sizing rule
    /// rather than on two built datasets because the interesting scale is
    /// ImageNet 32×32 (1.28 M images), and no test is going to allocate the
    /// 15.7 GiB the f32 encoding of it would take.
    ///
    /// The clamp at the end is why this is phrased as "at least": once a whole
    /// dataset fits in one chunk, the 8-bit encoding stops buying residency and
    /// starts buying the thing residency was for — it buys ALL of it, and the
    /// chunk count falls to one.
    #[test]
    fn an_8_bit_chunk_holds_four_times_the_images() {
        // The rule's inputs, as a real device reports them; only the ratio is
        // under test, so the exact caps do not matter as long as both sides see
        // the same ones.
        let caps = (128 << 20, 1 << 30, 8u64 << 30);
        let sample_len = 3072u64; // CIFAR-10 / ImageNet 32×32 RGB
        for count in [50_000u64, 1_281_167] {
            let capacity = |sample_bytes: u64| {
                select_chunk_bytes(caps.0, caps.1, caps.2, count * sample_bytes) / sample_bytes
            };
            let floats = capacity(sample_len * 4);
            let bytes = capacity(sample_len);
            assert!(
                bytes >= floats * 4 || bytes >= count,
                "{count} samples: an 8-bit chunk holds {bytes}, four f32 chunks hold {}",
                floats * 4
            );
            // Rounded up (last chunk short both sides): four f32 chunks of images
            // fit one 8-bit chunk, so the count falls to a quarter.
            assert!(
                count.div_ceil(bytes) <= count.div_ceil(floats).div_ceil(4),
                "{count} samples: {} 8-bit chunks against {} f32 ones",
                count.div_ceil(bytes),
                count.div_ceil(floats)
            );
        }

        // ImageNet 32×32 RGB (1 281 167 × 3072 bytes): 15.7 GB f32 vs 3.9 GB u8 —
        // the f32 figure is what makes the corpus impractical, not the corpus.
        let imagenet = 1_281_167u64 * sample_len;
        assert_eq!(imagenet * 4 / 1_000_000_000, 15);
        assert_eq!(imagenet / 1_000_000_000, 3);
    }

    /// The same claim where it is actually allocated: two real datasets, same
    /// images, and the 8-bit one resident in fewer chunks.
    ///
    /// Pinned to the **web** profile, and that is the test's subject rather than
    /// a detail: chunking is what a device with a 128 MiB binding cap does, and
    /// under native limits both datasets would simply be resident and the
    /// comparison would have nothing left to compare.
    #[test]
    fn a_real_8_bit_dataset_is_resident_in_fewer_chunks() {
        pollster::block_on(async {
            let gpu = Arc::new(
                crate::gpu_context::GpuContext::new_headless_with(
                    crate::gpu_context::GpuLimitsProfile::Web,
                )
                .await,
            );
            let sample_len = 3072;
            let count = 40_000;
            let floats = GpuDataset::from_samples(
                gpu.as_ref(),
                vec![vec![0.0; sample_len]; count],
                sample_len,
            )
            .unwrap();
            let bytes = GpuDataset::from_payload(
                gpu.as_ref(),
                DatasetPayload::Bytes(vec![0u8; count * sample_len]),
                sample_len,
            )
            .unwrap();

            assert!(
                bytes.chunk_sample_capacity() >= floats.chunk_sample_capacity() * 4
                    || bytes.chunk_sample_capacity() == count,
                "8-bit chunk holds {}, f32 chunk holds {}",
                bytes.chunk_sample_capacity(),
                floats.chunk_sample_capacity()
            );
            assert!(
                count.div_ceil(bytes.chunk_sample_capacity())
                    <= count.div_ceil(floats.chunk_sample_capacity()).div_ceil(4),
                "{} 8-bit chunks against {} f32 ones",
                count.div_ceil(bytes.chunk_sample_capacity()),
                count.div_ceil(floats.chunk_sample_capacity())
            );
        });
    }

    /// A byte payload that is not a whole number of samples is a truncated file,
    /// and must be refused rather than silently short by one image.
    #[test]
    fn a_ragged_8_bit_payload_is_refused() {
        pollster::block_on(async {
            let gpu = Arc::new(crate::gpu_context::GpuContext::new_headless().await);
            let result =
                GpuDataset::from_payload(gpu.as_ref(), DatasetPayload::Bytes(vec![0u8; 10]), 4);
            assert!(
                matches!(result, Err(GpuDatasetError::InvalidFlatLength { .. })),
                "a ragged byte payload was accepted"
            );
        });
    }

    /// A sample length that is not a multiple of 4 must still work: the chunk is
    /// padded to a word for `write_buffer`, and the padding is never read.
    #[test]
    fn an_8_bit_sample_of_odd_length_still_decodes() {
        pollster::block_on(async {
            let gpu = Arc::new(crate::gpu_context::GpuContext::new_headless().await);
            let sample_len = 3;
            let bytes: Vec<u8> = vec![1, 2, 3, 250, 251, 252];
            let mut dataset =
                GpuDataset::from_payload(gpu.as_ref(), DatasetPayload::Bytes(bytes), sample_len)
                    .unwrap();
            let destination = batch_destination(gpu.as_ref(), 2 * sample_len);
            dataset
                .copy_samples_to(gpu.as_ref(), &[1, 0], &destination)
                .unwrap();
            let values =
                crate::model::debug::read_back_f32(gpu.as_ref(), &destination, destination.size())
                    .unwrap();
            let want: Vec<f32> = [250u8, 251, 252, 1, 2, 3].iter().copied().map(decode_u8).collect();
            assert_eq!(values, want);
        });
    }

    /// A dataset that fits is uploaded ONCE, and no step after the first pays
    /// for it.
    ///
    /// This is the whole of what the native limits profile buys on the data
    /// side. Asserted on `chunk_loads` rather than on a duration, because the
    /// property is "no upload happens", not "uploads are fast" — on the unified
    /// memory this is developed on a 128 MiB chunk copy costs about a
    /// millisecond against a step of seconds, so a timing would have measured
    /// nothing and would have passed just as well if the uploads had stayed.
    #[test]
    fn a_dataset_that_fits_is_uploaded_once_and_never_again() {
        pollster::block_on(async {
            let gpu = Arc::new(
                crate::gpu_context::GpuContext::new_headless_with(
                    crate::gpu_context::GpuLimitsProfile::Native,
                )
                .await,
            );
            let sample_len = 3072;
            let count = 20_000; // 61 MiB in 8-bit: resident on any real adapter
            let mut dataset = GpuDataset::from_payload(
                gpu.as_ref(),
                DatasetPayload::Bytes(vec![7u8; count * sample_len]),
                sample_len,
            )
            .unwrap();
            assert!(
                dataset.is_resident(),
                "{} chunks — a 61 MiB dataset must be resident under native limits",
                dataset.chunk_count()
            );

            let destination = batch_destination(gpu.as_ref(), 32 * sample_len);
            // Batches drawn from all over the dataset, which is exactly the
            // access pattern that made streaming expensive.
            for round in 0..16 {
                let batch: Vec<usize> = (0..32)
                    .map(|slot| (round * 977 + slot * 613) % count)
                    .collect();
                dataset
                    .copy_samples_to(gpu.as_ref(), &batch, &destination)
                    .unwrap();
            }
            assert_eq!(
                dataset.chunk_loads(),
                1,
                "a resident dataset re-uploaded its chunk"
            );
        });
    }

    /// The CPU-side accessor the probe and the seed image use must agree with
    /// the GPU one — same values, same order.
    #[test]
    fn the_cpu_accessor_decodes_the_same_sample_the_gpu_does() {
        let bytes: Vec<u8> = (0..12u8).collect();
        let payload = DatasetPayload::Bytes(bytes);
        assert_eq!(payload.sample_count(4), 3);
        assert_eq!(
            payload.sample_f32(2, 4).unwrap(),
            vec![decode_u8(8), decode_u8(9), decode_u8(10), decode_u8(11)]
        );
        assert!(payload.sample_f32(3, 4).is_none());
    }
}
