//! File purpose: Implements diffusion logic used by the training pipeline.

use super::weighting::{LossWeighting, TimestepSampler};
use super::{
    GpuDataset, LinearNoiseSchedule, TaskPassSpec, TrainingTask, TrainingTaskError, Workgroups,
};
use crate::model::{Dim3, Model, Training};
use encase::{ShaderSize, ShaderType, StorageBuffer};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct DiffusionTask {
    schedule: LinearNoiseSchedule,
    timestep_channels: usize,
    pass_specs: Vec<TaskPassSpec>,
    configured_input: Option<Dim3>,
    configured_output: Option<Dim3>,
    prepare_pass: Option<DiffusionPreparePass>,
    shuffle: SampleShuffle,
    weighting: LossWeighting,
    sampler: TimestepSampler,
}

/// Per-epoch permutation of the dataset.
///
/// The sample index used to be `(step*batch_size + batch_offset) % sample_count`,
/// i.e. strictly sequential: the presentation order was identical at every
/// epoch. The permutation is regenerated whenever the epoch changes and is
/// derived from the epoch number alone, so runs stay reproducible.
#[derive(Debug, Clone, Default)]
pub(crate) struct SampleShuffle {
    order: Vec<usize>,
    epoch: Option<u64>,
}

impl SampleShuffle {
    pub(crate) fn sample_index(
        &mut self,
        position: usize,
        epoch: u64,
        sample_count: usize,
    ) -> usize {
        if sample_count == 0 {
            return 0;
        }
        if self.epoch != Some(epoch) || self.order.len() != sample_count {
            self.order = (0..sample_count).collect();
            let mut rng = SplitMix64::new(epoch.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x5DEE_CE66);
            for i in (1..sample_count).rev() {
                let j = (rng.next_u64() % (i as u64 + 1)) as usize;
                self.order.swap(i, j);
            }
            self.epoch = Some(epoch);
        }
        self.order[position % sample_count]
    }
}

/// Minimal SplitMix64 — enough to shuffle indices and draw timesteps, with no
/// external RNG dependency.
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

#[derive(Debug, Clone)]
struct DiffusionPreparePass {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    clean_target: wgpu::Buffer,
    specs: wgpu::Buffer,
    model_input: Arc<wgpu::Buffer>,
    target_noise: Arc<wgpu::Buffer>,
    expected_target_len: usize,
    /// Workgroups for ONE sample; the encode multiplies by the batch it is
    /// asked for, so a batch-1 call on a batch-16 graph dispatches a prefix.
    workgroups_per_sample: u32,
    /// Samples the pass was built for — the size of `specs` and `clean_target`.
    batch: u32,
}

/// One entry per sample of the batch. Only `alpha_bar`, `step` and `seed`
/// actually differ between entries; the shape fields are replicated so the
/// shader has everything it needs from a single indexed read.
#[derive(ShaderType, Clone, Copy)]
struct DiffusionPrepareUniform {
    alpha_bar: f32,
    step: u32,
    seed: u32,
    input_channels: u32,
    signal_channels: u32,
    timestep_channels: u32,
    pixel_count: u32,
    /// Schedule length T, so the shader's timestep embedding normalises `step`
    /// to `[0, 1]` exactly like the CPU `LinearNoiseSchedule::timestep_embedding`
    /// used at inference. The two MUST agree or the network is conditioned on
    /// one signal and sampled with another.
    total_steps: u32,
}

impl DiffusionTask {
    pub fn new(schedule: LinearNoiseSchedule) -> Self {
        Self::new_with_weighting(schedule, LossWeighting::default())
    }

    /// Same task, but the training timestep is drawn from `p(t) ∝ w(t)` instead
    /// of uniformly — see `weighting.rs` for why that is exactly a `w(t)`-weighted
    /// loss and why it is done here rather than in the loss shader.
    pub fn new_with_weighting(schedule: LinearNoiseSchedule, weighting: LossWeighting) -> Self {
        let sampler = TimestepSampler::new(&schedule, weighting);
        Self {
            schedule,
            timestep_channels: 0,
            pass_specs: Vec::new(),
            configured_input: None,
            configured_output: None,
            prepare_pass: None,
            shuffle: SampleShuffle::default(),
            weighting,
            sampler,
        }
    }

    pub fn schedule(&self) -> &LinearNoiseSchedule {
        &self.schedule
    }

    pub fn loss_weighting(&self) -> LossWeighting {
        self.weighting
    }

    /// Share of the timestep draws landing in each of the four probe buckets,
    /// for the run banner. Empty for the uniform sampler.
    pub fn timestep_bucket_mass(&self, buckets: usize) -> Vec<f64> {
        let p = self.sampler.probabilities();
        if p.is_empty() || buckets == 0 {
            return Vec::new();
        }
        let span = p.len().div_ceil(buckets);
        (0..buckets)
            .map(|b| {
                let lo = (b * span).min(p.len());
                let hi = ((b + 1) * span).min(p.len());
                p[lo..hi].iter().sum()
            })
            .collect()
    }

    pub fn timestep_channels(&self) -> usize {
        self.timestep_channels
    }

    /// GPU bytes the prepass allocates on top of the model, for the resource
    /// banner. Both buffers now carry the batch axis, so both scale with it —
    /// which is the point the banner has to convey: raising the batch raises
    /// GPU memory linearly, and it is the first thing that will hit
    /// `max_storage_buffer_binding_size` on a large model.
    pub fn estimated_prepare_gpu_bytes_for_batch(&self, output: Dim3, batch: u32) -> u64 {
        Self::prepare_target_bytes(output, batch)
            .saturating_add(Self::prepare_specs_bytes(batch))
    }

    /// The clean image batch the prepass noises — one slot per sample.
    ///
    /// Split out of the total so the resource inventory can name the two
    /// buffers separately: they are the two allocations `build_prepare_pass`
    /// makes, and a page that says "prepass: 2 MiB" without saying which half
    /// grows with the batch has not said anything.
    pub fn prepare_target_bytes(output: Dim3, batch: u32) -> u64 {
        (output.length() as u64 * std::mem::size_of::<f32>() as u64 * batch.max(1) as u64).max(4)
    }

    /// The per-sample spec array the prepass shader indexes by sample.
    pub fn prepare_specs_bytes(batch: u32) -> u64 {
        (DiffusionPrepareUniform::SHADER_SIZE.get() as u64 * batch.max(1) as u64).max(4)
    }

    pub fn estimated_prepare_gpu_bytes(&self, output: Dim3) -> u64 {
        self.estimated_prepare_gpu_bytes_for_batch(output, 1)
    }

    /// Encode the per-sample spec array the shader indexes by sample.
    fn encode_prepare_specs(specs: &[DiffusionPrepareUniform]) -> Vec<u8> {
        let mut buffer = StorageBuffer::new(Vec::new());
        buffer
            .write(&specs.to_vec())
            .expect("failed to encode diffusion prepare specs");
        buffer.into_inner()
    }

    pub fn train_step_report(
        &mut self,
        model: &mut Model<Training>,
        clean_target: &[f32],
        diffusion_step: usize,
        seed: u64,
    ) -> Result<f32, TrainingTaskError> {
        let input = model.input_dim().ok_or(TrainingTaskError::EmptyModel)?;
        let output = model.output_dim().ok_or(TrainingTaskError::EmptyModel)?;
        if !same_dims(self.configured_input, input) || !same_dims(self.configured_output, output) {
            self.configure(input, output)?;
        }

        let alpha_bar = self.schedule.alpha_bar(diffusion_step);
        let step = diffusion_step.min(self.schedule.len().saturating_sub(1));
        let seed = fold_seed(seed);
        let specs_bytes = Self::encode_prepare_specs(&[DiffusionPrepareUniform {
            alpha_bar,
            step: step as u32,
            seed,
            input_channels: input.z,
            signal_channels: output.z,
            timestep_channels: self.timestep_channels as u32,
            pixel_count: output.x * output.y,
            total_steps: self.schedule.len() as u32,
        }]);

        let pass = self.ensure_prepare_pass(model, input, output)?;
        if clean_target.len() != pass.expected_target_len {
            return Err(TrainingTaskError::TargetLengthMismatch {
                expected: pass.expected_target_len,
                actual: clean_target.len(),
            });
        }

        model
            .gpu
            .queue
            .write_buffer(&pass.clean_target, 0, bytemuck::cast_slice(clean_target));
        model.gpu.write_buffer(&pass.specs, 0, &specs_bytes);

        // One sample, whatever the graph was built for: this entry point takes
        // a single CPU-side target and fills slot 0.
        let loss = model.train_step_report_with_prepass(|encoder| {
            pass.encode_with_batch(encoder, 1);
        });
        Ok(loss)
    }

    /// `(sample_index, timestep)` for each slot of the batch.
    ///
    /// Factored out so the batched path and the sequential reference of
    /// `train_step_batch_sequential` cannot drift: both draw from the same
    /// shuffle, the same counter and the same sampler, which is what makes the
    /// two paths comparable sample by sample.
    fn plan_batch(
        &mut self,
        sample_count: usize,
        step: usize,
        batch_size: usize,
        seed: u64,
    ) -> Vec<(usize, usize)> {
        let schedule_len = self.schedule.len();
        (0..batch_size)
            .map(|batch_offset| {
                let counter = step.wrapping_mul(batch_size).wrapping_add(batch_offset);
                let (epoch, position) = if sample_count == 0 {
                    (0, 0)
                } else {
                    ((counter / sample_count) as u64, counter % sample_count)
                };
                let sample_index = self.shuffle.sample_index(position, epoch, sample_count);
                (
                    sample_index,
                    self.sampler.draw(counter, schedule_len, seed),
                )
            })
            .collect()
    }

    pub fn train_step_report_batch(
        &mut self,
        model: &mut Model<Training>,
        dataset: &mut GpuDataset,
        step: usize,
        batch_size: usize,
        seed: u64,
    ) -> Result<Option<f32>, TrainingTaskError> {
        self.train_step_batch_inner(model, dataset, step, batch_size, seed, true)
    }

    pub fn train_step_batch(
        &mut self,
        model: &mut Model<Training>,
        dataset: &mut GpuDataset,
        step: usize,
        batch_size: usize,
        seed: u64,
    ) -> Result<(), TrainingTaskError> {
        let _ = self.train_step_batch_inner(model, dataset, step, batch_size, seed, false)?;
        Ok(())
    }

    fn train_step_batch_inner(
        &mut self,
        model: &mut Model<Training>,
        dataset: &mut GpuDataset,
        step: usize,
        batch_size: usize,
        seed: u64,
        report_last_loss: bool,
    ) -> Result<Option<f32>, TrainingTaskError> {
        if batch_size == 0 {
            return Err(TrainingTaskError::InvalidBatchSize { batch_size });
        }
        let input = model.input_dim().ok_or(TrainingTaskError::EmptyModel)?;
        let output = model.output_dim().ok_or(TrainingTaskError::EmptyModel)?;
        if !same_dims(self.configured_input, input) || !same_dims(self.configured_output, output) {
            self.configure(input, output)?;
        }
        let schedule = self.schedule.clone();
        let schedule_len = schedule.len();
        let timestep_channels = self.timestep_channels as u32;

        // Resolved up front: the shuffle borrows `self`, while `pass` below
        // borrows it mutably for the rest of the function.
        let batch_plan = self.plan_batch(dataset.sample_count(), step, batch_size, seed);

        // The graph carries the batch, so the graph must have been built for
        // it. Anything else would silently train on a prefix (or read past the
        // end of the spec array).
        if model.batch() as usize != batch_size {
            return Err(TrainingTaskError::InvalidBatchSize { batch_size });
        }

        let pass = self.ensure_prepare_pass(model, input, output)?;
        if dataset.sample_len() != pass.expected_target_len {
            return Err(TrainingTaskError::TargetLengthMismatch {
                expected: pass.expected_target_len,
                actual: dataset.sample_len(),
            });
        }

        // Per-sample specs. Each entry is computed by EXACTLY the expression
        // that computed it when the batch was unrolled on the CPU — same
        // `batch_offset`, same `sample_index`, same `fold_seed`. That identity
        // is what makes sample i of the batched step draw the same noise and
        // the same timestep as sample i of the sequential step, and hence what
        // makes an old-path/new-path training run comparable.
        let specs: Vec<DiffusionPrepareUniform> = batch_plan
            .iter()
            .enumerate()
            .map(|(batch_offset, &(sample_index, diffusion_step))| {
                let step_seed = seed ^ ((batch_offset as u64) << 32) ^ sample_index as u64;
                DiffusionPrepareUniform {
                    alpha_bar: schedule.alpha_bar(diffusion_step),
                    step: diffusion_step as u32,
                    seed: fold_seed(step_seed),
                    input_channels: input.z,
                    signal_channels: output.z,
                    timestep_channels,
                    pixel_count: output.x * output.y,
                    total_steps: schedule_len as u32,
                }
            })
            .collect();
        let specs_bytes = Self::encode_prepare_specs(&specs);
        model.gpu.write_buffer(&pass.specs, 0, &specs_bytes);

        // Dataset -> clean_target, one slot per sample. Submitted separately
        // (and grouped by chunk) because chunk loading goes through
        // `queue.write_buffer`, which would otherwise be applied ahead of the
        // copies encoded before it — see `GpuDataset::copy_samples_to`.
        let sample_indices: Vec<usize> = batch_plan.iter().map(|&(index, _)| index).collect();
        let gpu = model.gpu.clone();
        dataset
            .copy_samples_to(gpu.as_ref(), &sample_indices, &pass.clean_target)
            .map_err(|err| TrainingTaskError::DatasetError {
                message: err.to_string(),
            })?;

        // ONE encoder, ONE submit: prepare, forward, loss, backward and the
        // optimiser update for the whole batch. This is the change the mission
        // is about — it used to be `1 + 16 + 1` submits at batch 16.
        let loss = model.train_step_report_batched(batch_size, report_last_loss, |encoder| {
            pass.encode(encoder);
        });

        Ok(loss)
    }

    /// The pre-batching training step, kept verbatim as a **test reference**.
    ///
    /// One `CommandEncoder` and one `submit` per sample, gradients accumulated
    /// with `+=` between samples, optimiser applied once at the end. This is
    /// the code the batched path has to agree with, and keeping it executable
    /// — rather than freezing a fixture — means the equivalence tests compare
    /// two things that both still run.
    ///
    /// Requires a model built at `batch = 1`.
    #[cfg(test)]
    pub(crate) fn train_step_batch_sequential(
        &mut self,
        model: &mut Model<Training>,
        dataset: &mut GpuDataset,
        step: usize,
        batch_size: usize,
        seed: u64,
    ) -> Result<Option<f32>, TrainingTaskError> {
        if batch_size == 0 {
            return Err(TrainingTaskError::InvalidBatchSize { batch_size });
        }
        assert_eq!(
            model.batch(),
            1,
            "the sequential reference needs a batch-1 graph"
        );
        let input = model.input_dim().ok_or(TrainingTaskError::EmptyModel)?;
        let output = model.output_dim().ok_or(TrainingTaskError::EmptyModel)?;
        if !same_dims(self.configured_input, input) || !same_dims(self.configured_output, output) {
            self.configure(input, output)?;
        }
        let schedule = self.schedule.clone();
        let schedule_len = schedule.len();
        let timestep_channels = self.timestep_channels as u32;
        let batch_plan = self.plan_batch(dataset.sample_count(), step, batch_size, seed);

        let pass = self.ensure_prepare_pass(model, input, output)?;
        if dataset.sample_len() != pass.expected_target_len {
            return Err(TrainingTaskError::TargetLengthMismatch {
                expected: pass.expected_target_len,
                actual: dataset.sample_len(),
            });
        }
        let gpu = model.gpu.clone();

        let mut last_loss = None;
        model.begin_batch_accumulation();
        for (batch_offset, &(sample_index, diffusion_step)) in batch_plan.iter().enumerate() {
            let step_seed = seed ^ ((batch_offset as u64) << 32) ^ sample_index as u64;
            let specs_bytes = Self::encode_prepare_specs(&[DiffusionPrepareUniform {
                alpha_bar: schedule.alpha_bar(diffusion_step),
                step: diffusion_step as u32,
                seed: fold_seed(step_seed),
                input_channels: input.z,
                signal_channels: output.z,
                timestep_channels,
                pixel_count: output.x * output.y,
                total_steps: schedule_len as u32,
            }]);
            model.gpu.write_buffer(&pass.specs, 0, &specs_bytes);
            dataset
                .copy_samples_to(gpu.as_ref(), &[sample_index], &pass.clean_target)
                .map_err(|err| TrainingTaskError::DatasetError {
                    message: err.to_string(),
                })?;

            if batch_offset + 1 == batch_size {
                last_loss = model.train_step_report_with_prepass_no_opt(|encoder| {
                    pass.encode_with_batch(encoder, 1);
                });
            } else {
                model.train_step_with_prepass_no_opt(|encoder| {
                    pass.encode_with_batch(encoder, 1);
                });
            }
        }
        model.finish_batch_accumulation(batch_size);

        Ok(last_loss)
    }

    fn ensure_prepare_pass(
        &mut self,
        model: &Model<Training>,
        input: Dim3,
        output: Dim3,
    ) -> Result<&mut DiffusionPreparePass, TrainingTaskError> {
        let model_input = model
            .layers
            .first()
            .ok_or(TrainingTaskError::EmptyModel)?
            .buffers
            .forward[0]
            .clone();
        let target_noise = model
            .loss_layer
            .as_ref()
            .ok_or(TrainingTaskError::EmptyModel)?
            .buffers
            .forward[1]
            .clone();

        let needs_rebuild = match self.prepare_pass.as_ref() {
            None => true,
            Some(state) => {
                !Arc::ptr_eq(&state.model_input, &model_input)
                    || !Arc::ptr_eq(&state.target_noise, &target_noise)
                    || state.expected_target_len != output.length() as usize
                    // A resized batch means a resized spec array and a resized
                    // clean_target; the model's buffers were rebuilt with it.
                    || state.batch != model.batch()
            }
        };

        if needs_rebuild {
            self.prepare_pass = Some(Self::build_prepare_pass(
                model,
                input,
                output,
                model_input,
                target_noise,
            ));
        }

        Ok(self.prepare_pass.as_mut().expect("prepare pass missing"))
    }

    fn build_prepare_pass(
        model: &Model<Training>,
        input: Dim3,
        output: Dim3,
        model_input: Arc<wgpu::Buffer>,
        target_noise: Arc<wgpu::Buffer>,
    ) -> DiffusionPreparePass {
        let device = &model.gpu.device;
        let batch = model.batch().max(1);
        let expected_target_len = output.length() as usize;
        let target_bytes =
            (expected_target_len * batch as usize * std::mem::size_of::<f32>()) as u64;
        let specs_size = DiffusionPrepareUniform::SHADER_SIZE.get() as u64 * batch as u64;

        let clean_target = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("diffusion_clean_target"),
            // COPY_SRC so tests can read back what the dataset copy landed.
            size: target_bytes.max(4),
            usage: wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        // A storage buffer, not a uniform: it is an array indexed by the sample,
        // and a uniform array of structs would pay 16-byte element alignment for
        // nothing.
        let specs = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("diffusion_prepare_specs"),
            size: specs_size.max(4),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("diffusion_prepare_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("diffusion_prepare_bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: clean_target.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: specs.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: model_input.as_ref().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: target_noise.as_ref().as_entire_binding(),
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("diffusion_prepare"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "shader/diffusion_prepare.wgsl"
            ))),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("diffusion_prepare_layout"),
            bind_group_layouts: &[&bgl],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("diffusion_prepare_pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("diffusion_prepare"),
            cache: None,
            compilation_options: Default::default(),
        });

        DiffusionPreparePass {
            pipeline,
            bind_group,
            clean_target,
            specs,
            model_input,
            target_noise,
            expected_target_len,
            workgroups_per_sample: input.length().div_ceil(64),
            batch,
        }
    }
}

impl DiffusionPreparePass {
    fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        self.encode_with_batch(encoder, self.batch);
    }

    fn encode_with_batch(&self, encoder: &mut wgpu::CommandEncoder, batch: u32) {
        let batch = batch.clamp(1, self.batch.max(1));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("diffusion_prepare_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        let (x, y) =
            crate::model::layer::dispatch_grid(self.workgroups_per_sample * batch);
        pass.dispatch_workgroups(x, y, 1);
    }
}

impl TrainingTask for DiffusionTask {
    fn name(&self) -> &'static str {
        "diffusion"
    }

    fn configure(&mut self, input: Dim3, output: Dim3) -> Result<(), TrainingTaskError> {
        if input.x != output.x || input.y != output.y {
            return Err(TrainingTaskError::InvalidLayout {
                input,
                output,
                message: "diffusion requires matching spatial input/output dims",
            });
        }
        if input.z < output.z {
            return Err(TrainingTaskError::InvalidLayout {
                input,
                output,
                message: "diffusion requires input channels >= output channels",
            });
        }
        self.timestep_channels = input.z.saturating_sub(output.z) as usize;
        self.configured_input = Some(input);
        self.configured_output = Some(output);
        self.prepare_pass = None;
        self.pass_specs = vec![TaskPassSpec {
            label: "diffusion_prepare",
            entrypoint: "diffusion_prepare",
            workgroups: Workgroups::x(input.length().div_ceil(64)),
        }];
        Ok(())
    }

    fn pass_specs(&self) -> &[TaskPassSpec] {
        &self.pass_specs
    }
}

fn fold_seed(seed: u64) -> u32 {
    let mixed = seed ^ (seed >> 32);
    (mixed as u32)
        .wrapping_mul(1664525)
        .wrapping_add(1013904223)
}

/// Draws t ~ U{0, schedule_len-1}, independently of which sample is paired with
/// it.
///
/// The timestep used to be `counter % schedule_len` while the sample index was
/// `counter % sample_count` — both derived from the same linear counter, so the
/// timestep was a deterministic function of the image. A given sample only ever
/// saw `gcd(sample_count, schedule_len)` distinct timesteps (16 of 256 for
/// CIFAR-10), whereas DDPM requires t drawn independently of x_0.
pub(crate) fn diffusion_step_for(counter: usize, schedule_len: usize, seed: u64) -> usize {
    if schedule_len == 0 {
        return 0;
    }
    let mut rng = SplitMix64::new(seed ^ (counter as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    (rng.next_u64() % schedule_len as u64) as usize
}

/// The same draw as [`diffusion_step_for`], stopped one step earlier: the raw
/// uniform in `[0, 1)` that a non-uniform sampler feeds to its inverse CDF.
/// Keeping the stream identical means the two samplers see the same randomness
/// and differ only by the mapping applied to it.
pub(crate) fn uniform_unit_for(counter: usize, seed: u64) -> f64 {
    let mut rng = SplitMix64::new(seed ^ (counter as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    // Top 53 bits: the exactly-representable mantissa range of f64.
    (rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

fn same_dims(saved: Option<Dim3>, target: Dim3) -> bool {
    let Some(saved) = saved else {
        return false;
    };
    saved.x == target.x && saved.y == target.y && saved.z == target.z
}

#[cfg(test)]
mod tests {
    use super::DiffusionTask;
    use crate::gpu_context::GpuContext;
    use crate::model::{ActivationMethod, ActivationType, Dim3, LayerTypes, LossMethod, Model};
    use crate::training::{GpuDataset, LinearNoiseSchedule, TrainingTask};
    use std::sync::Arc;

    #[test]
    fn diffusion_task_rejects_spatial_mismatch() {
        let schedule = LinearNoiseSchedule::new_linear(8, 1e-4, 0.02);
        let mut task = DiffusionTask::new(schedule);
        let err = task
            .configure(Dim3::new((32, 32, 8)), Dim3::new((16, 16, 3)))
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("matching spatial input/output dims")
        );
    }

    #[test]
    fn diffusion_task_computes_timestep_channels() {
        let schedule = LinearNoiseSchedule::new_linear(8, 1e-4, 0.02);
        let mut task = DiffusionTask::new(schedule);
        task.configure(Dim3::new((32, 32, 8)), Dim3::new((32, 32, 3)))
            .unwrap();
        assert_eq!(task.timestep_channels(), 5);
        assert_eq!(task.pass_specs().len(), 1);
        assert_eq!(task.pass_specs()[0].entrypoint, "diffusion_prepare");
        assert_eq!(
            task.pass_specs()[0].workgroups,
            super::Workgroups::x(32 * 32 * 8 / 64)
        );
    }

    #[test]
    fn diffusion_task_gpu_prepare_pass_executes() {
        pollster::block_on(async {
            let gpu = Arc::new(GpuContext::new_headless().await);
            let mut model = Model::new_training(gpu, 0.01, 1, LossMethod::MeanSquared).await;
            model
                .add_layer(LayerTypes::Activation(ActivationType::new(
                    ActivationMethod::Linear,
                    Dim3::new((4, 4, 3)),
                )))
                .unwrap();
            model.build().unwrap();

            let schedule = LinearNoiseSchedule::new_linear(8, 1e-4, 0.02);
            let mut task = DiffusionTask::new(schedule);
            let samples = vec![vec![0.0f32; 4 * 4 * 3]];
            let mut dataset = GpuDataset::from_samples(model.gpu.as_ref(), samples, 4 * 4 * 3)
                .expect("failed to upload gpu dataset");
            let loss = task
                .train_step_report_batch(&mut model, &mut dataset, 0, 1, 42)
                .unwrap();
            assert!(loss.is_some());
            assert!(loss.unwrap().is_finite());
        });
    }

    /// The batched prepass must give every sample bit-for-bit the noise and the
    /// timestep embedding the sequential prepass gave it.
    ///
    /// This is the invariant the whole validation plan rests on: if sample i of
    /// a batch draws different noise than it used to, an old-path/new-path
    /// training comparison measures nothing, because the two runs are no longer
    /// solving the same problem.
    ///
    /// The comparison is between two *code paths*, with no duplicated formula:
    /// the batched step is run once, then `train_step_batch_sequential` — the
    /// pre-batching loop, kept executable — is run with batch sizes 1, 2, ... B,
    /// each leaving slot 0 holding the prepass output of its last sample. At
    /// `step = 0` the counter is `0 * batch_size + offset == offset`, so slot j
    /// of the batch of B and the last slot of the batch of j+1 are the same
    /// (sample, timestep, batch_offset) triple — and must therefore produce the
    /// same bits.
    #[test]
    fn batched_prepass_gives_each_sample_the_noise_it_had_alone() {
        pollster::block_on(async {
            const BATCH: usize = 4;
            let gpu = Arc::new(GpuContext::new_headless().await);

            // input z > output z: a diffusion model must be conditioned on t.
            let build = |batch: u32| {
                let gpu = gpu.clone();
                async move {
                    let mut model =
                        Model::new_training(gpu, 0.01, batch, LossMethod::MeanSquared).await;
                    model
                        .add_layer(LayerTypes::Convolution(
                            crate::model::layer_types::ConvolutionType::new(
                                Dim3::new((4, 4, 3)),
                                1,
                                Dim3::new((3, 3, 3)),
                                1,
                                crate::model::PaddingMode::Same,
                            ),
                        ))
                        .unwrap();
                    model.build().unwrap();
                    model
                }
            };
            let mut batched = build(BATCH as u32).await;
            let mut single = build(1).await;

            let sample_len = 4 * 4;
            let samples: Vec<Vec<f32>> = (0..BATCH)
                .map(|s| {
                    (0..sample_len)
                        .map(|i| (s as f32 + 1.0) * 0.37 - i as f32 * 0.011)
                        .collect()
                })
                .collect();
            let mut dataset_batched =
                GpuDataset::from_samples(gpu.as_ref(), samples.clone(), sample_len).unwrap();
            let mut dataset_single =
                GpuDataset::from_samples(gpu.as_ref(), samples, sample_len).unwrap();

            let schedule = LinearNoiseSchedule::new_linear(64, 1e-4, 0.02);
            let seed = 0xC0FFEE_u64;
            let mut task_batched = DiffusionTask::new(schedule.clone());
            let mut task_single = DiffusionTask::new(schedule);

            task_batched
                .train_step_report_batch(&mut batched, &mut dataset_batched, 0, BATCH, seed)
                .unwrap();
            let all_input = read_forward(&batched, 0, 0);
            let all_noise = read_loss_forward(&batched, 1);

            let input_len = 4 * 4 * 3;
            for j in 0..BATCH {
                task_single
                    .train_step_batch_sequential(
                        &mut single,
                        &mut dataset_single,
                        0,
                        j + 1,
                        seed,
                    )
                    .unwrap();
                let alone_input = read_forward(&single, 0, 0);
                let alone_noise = read_loss_forward(&single, 1);

                let batched_slice = &all_input[j * input_len..(j + 1) * input_len];
                let differing = alone_input[..input_len]
                    .iter()
                    .zip(batched_slice)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                assert_eq!(
                    differing, 0,
                    "\nx_t of sample {j} differs between the batched and the sequential \
                     prepass ({differing}/{input_len} elements)\n\
                     sequential[..6]: {:?}\nbatched[..6]:    {:?}\n",
                    &alone_input[..6],
                    &batched_slice[..6],
                );

                let batched_noise = &all_noise[j * sample_len..(j + 1) * sample_len];
                let differing = alone_noise[..sample_len]
                    .iter()
                    .zip(batched_noise)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                assert_eq!(
                    differing, 0,
                    "\ntarget noise of sample {j} differs between the batched and the \
                     sequential prepass ({differing}/{sample_len} elements)\n"
                );
            }

            // Not vacuous: the samples must actually have drawn different noise,
            // otherwise "every slice matches" is true for the wrong reason.
            let first = &all_noise[..sample_len];
            let last = &all_noise[(BATCH - 1) * sample_len..BATCH * sample_len];
            assert!(
                first.iter().zip(last).any(|(a, b)| (a - b).abs() > 1e-3),
                "every sample of the batch drew the same noise — the per-sample seed \
                 is not reaching the shader"
            );
        });
    }

    fn read_forward(model: &Model<crate::model::Training>, layer: usize, binding: usize) -> Vec<f32> {
        let buffer = &model.layers[layer].buffers.forward[binding];
        crate::model::debug::read_back_f32(model.gpu.as_ref(), buffer, buffer.size()).unwrap()
    }

    fn read_loss_forward(model: &Model<crate::model::Training>, binding: usize) -> Vec<f32> {
        let buffer = &model.loss_layer.as_ref().unwrap().buffers.forward[binding];
        crate::model::debug::read_back_f32(model.gpu.as_ref(), buffer, buffer.size()).unwrap()
    }

    /// Kept from PR #6: consecutive draws within a batch must not collapse onto
    /// a single timestep. Strengthened — the draw is now random, so the property
    /// is "spread across the schedule", not "the identity sequence".
    #[test]
    fn diffusion_step_progression_does_not_alias_single_batch() {
        use std::collections::HashSet;
        let steps: HashSet<usize> = (0..8)
            .map(|counter| super::diffusion_step_for(counter, 8, 0))
            .collect();
        assert!(
            steps.len() >= 4,
            "8 consecutive draws over 8 timesteps collapsed to {steps:?}"
        );
    }

    /// Finding #2 — the timestep must be independent of the sample it is paired
    /// with. Walks the same (sample, timestep) pairing the training loop builds
    /// and checks that one given sample eventually sees the whole schedule.
    #[test]
    fn timestep_is_decorrelated_from_sample_index() {
        use std::collections::HashSet;

        let sample_count = 1_024usize;
        let schedule_len = 256usize;
        let batch_size = 16usize;
        // Sample #0 is drawn once per epoch, so coverage is bounded by the epoch
        // count. 2000 epochs makes full coverage of a 256-step schedule the
        // overwhelmingly likely outcome for a uniform draw (expected number of
        // never-hit timesteps: 256 * e^-7.8 < 0.2), while the old lattice
        // pairing would stay pinned at gcd(sample_count, schedule_len).
        let epochs = 2_000usize;
        let mut shuffle = super::SampleShuffle::default();

        let mut seen: HashSet<usize> = HashSet::new();
        for counter in 0..(sample_count * epochs) {
            let step = counter / batch_size;
            let seed = (step as u64) << 32; // what main.rs passes per step
            let epoch = (counter / sample_count) as u64;
            let position = counter % sample_count;
            let sample_index = shuffle.sample_index(position, epoch, sample_count);
            if sample_index == 0 {
                seen.insert(super::diffusion_step_for(counter, schedule_len, seed));
            }
        }

        let lattice = gcd(sample_count, schedule_len);
        assert_eq!(
            seen.len(),
            schedule_len,
            "sample #0 saw {} of the {schedule_len} timesteps over {epochs} epochs \
             (the coupled pairing would cap it at gcd = {lattice})",
            seen.len()
        );
    }

    fn gcd(a: usize, b: usize) -> usize {
        if b == 0 { a } else { gcd(b, a % b) }
    }

    /// The dataset presentation order must differ between epochs.
    #[test]
    fn sample_order_is_shuffled_between_epochs() {
        let sample_count = 512usize;
        let mut shuffle = super::SampleShuffle::default();

        let order_of = |shuffle: &mut super::SampleShuffle, epoch: u64| -> Vec<usize> {
            (0..sample_count)
                .map(|position| shuffle.sample_index(position, epoch, sample_count))
                .collect()
        };

        let epoch0 = order_of(&mut shuffle, 0);
        let epoch1 = order_of(&mut shuffle, 1);

        assert_ne!(
            epoch0, epoch1,
            "presentation order is identical across epochs"
        );

        // Each epoch must still be a permutation — every sample seen exactly once.
        for order in [&epoch0, &epoch1] {
            let mut sorted = (*order).clone();
            sorted.sort_unstable();
            assert_eq!(sorted, (0..sample_count).collect::<Vec<_>>());
        }
    }
}
