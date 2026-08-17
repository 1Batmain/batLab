//! The convolution layer type: shapes, GPU bindings, and the backward-pass
//! reduction-lane heuristic.

use crate::model::error::ModelError;
use crate::model::layer_types::{
    Batch, batched_bytes,
    BackwardBufferBinding, BackwardBufferSource, BufferInit, ForwardBufferBinding,
    ForwardBufferSource, LayerType, OptimizerBindings, ShaderDescriptor,
};
use crate::model::types::{BufferSpec, Dim3, PaddingMode};
use encase::{ShaderSize, ShaderType, UniformBuffer};
use wgpu::BufferUsages;

#[derive(Debug, Default, Clone, Copy)]
pub struct ConvolutionType {
    pub nb_kernel: u32,
    pub dim_kernel: Dim3,
    pub stride: u32,
    pub mode: PaddingMode,
    pub dim_input: Dim3,
    pub dim_output: Dim3,
}

#[derive(ShaderType, Clone, Copy)]
pub struct ConvolutionUniform {
    pub nb_kernel: u32,
    pub stride: u32,
    pub padding_mode: u32, // 0 = Valid, 1 = Same
    /// Cooperating threads per sum for the TWO reductions: `grad_weights` in the
    /// low half of the word, `grad_bias` in the high half (see
    /// [`ConvolutionType::pack_lanes`]). Packed into one word — the word that was
    /// explicit padding — so the uniform's size and every field offset stay pinned
    /// against the legacy fixtures. The two halves differ because the reductions do:
    /// `grad_weights` has thousands of sums and wants few lanes, `grad_bias` has one
    /// per kernel and wants as many as it can get.
    pub reduction_lanes: u32,
    pub dim_kernel: Dim3,
    pub dim_input: Dim3,
    pub dim_output: Dim3,
}

impl ConvolutionType {
    pub fn new(
        dim_input: Dim3,
        nb_kernel: u32,
        dim_kernel: Dim3,
        stride: u32,
        mode: PaddingMode,
    ) -> Self {
        Self {
            nb_kernel,
            dim_kernel,
            stride,
            mode,
            dim_input,
            dim_output: Dim3::default(),
        }
    }

    fn kernel_bytes(&self) -> u32 {
        self.dim_kernel.bytes_size() * self.nb_kernel
    }

    /// Number of output positions the `grad_weights` / `grad_bias` reductions
    /// sum over.
    fn output_positions(&self) -> u32 {
        self.dim_output.x * self.dim_output.y
    }

    /// How many threads cooperate on one `grad_weights` / `grad_bias` sum.
    /// Splitting a sum across lanes buys parallelism but costs a barrier'd tree
    /// reduction, so it only pays when there are too few sums to keep the GPU busy.
    /// Rule: fewest lanes that put ~`TARGET_THREADS` threads in flight, never below
    /// `MIN_POSITIONS_PER_LANE` positions per lane, never past `MAX_LANES`.
    ///
    /// `MAX_LANES` is 32, NOT 64: at 64 a workgroup holds a single slot, so
    /// consecutive threads no longer walk consecutive `kz` and the slot layout's
    /// coalescing is lost (measured +35% on conv4).
    ///
    /// `TARGET_THREADS`/`MIN_POSITIONS_PER_LANE` are per-machine knobs read from
    /// [`crate::tuning`] (env-overridable), calibrated by `bench_conv_reduction_lanes`.
    /// The sweep tables and recalibration history: `KERNEL_HUNT.md`, `GPU_PROFILE.md`
    /// §6. Structural caveat still open (`positions` is per-sample, the loop covers
    /// `batch ×` more): `BATCH_DISPATCH.md` §9.
    pub(crate) fn reduction_lanes(sums: u32, positions: u32) -> u32 {
        let target_threads = crate::tuning::conv_target_threads();
        let min_positions_per_lane = crate::tuning::conv_min_positions_per_lane();
        const MAX_LANES: u32 = WG_SIZE / 2;

        let mut lanes = 1u32;
        while lanes < MAX_LANES
            && sums * lanes < target_threads
            && positions / (lanes * 2) >= min_positions_per_lane
        {
            lanes *= 2;
        }
        lanes
    }

    /// Lanes for the `grad_weights` reduction. Capped at the number of output ROWS:
    /// `conv_back_weights` walks `(sample, oy)` rows, so a lane past `OH` gets no
    /// row and idles. `OH` not `OH * batch` because the uniform is written once at
    /// build time without the batch — conservative in the only safe direction. No
    /// model here hits it (`OH >= 32 >= lanes`); a guard for wide-and-short shapes.
    pub(crate) fn reduction_lanes_for_weights(&self) -> u32 {
        let lanes = Self::reduction_lanes(
            self.dim_kernel.length() * self.nb_kernel,
            self.output_positions(),
        );
        // Largest power of two ≤ the row count (the tree reduction halves to 1).
        let rows = self.dim_output.x.max(1);
        lanes.min(1 << rows.ilog2())
    }

    /// Lanes for the `grad_bias` reduction — its OWN sum count (`nb_kernel`), not
    /// the weights' one: feeding the weights' choice to both left the bias one
    /// workgroup of 48 threads walking 32 768 positions (9.8 ms, latency-bound).
    /// Deliberately does NOT split the position axis across workgroups (a scratch
    /// buffer + second pass), which the sweep showed recovers ~nothing more.
    pub(crate) fn reduction_lanes_for_bias(&self) -> u32 {
        Self::reduction_lanes(self.nb_kernel, self.output_positions())
    }

    /// The two lane counts as the shader reads them: `grad_weights` in the low
    /// half of the word, `grad_bias` in the high half. Both are powers of two
    /// no greater than 32, so neither can ever overflow its half.
    pub(crate) fn pack_lanes(weight_lanes: u32, bias_lanes: u32) -> u32 {
        debug_assert!(weight_lanes <= WG_SIZE && bias_lanes <= WG_SIZE);
        (weight_lanes & 0xffff) | (bias_lanes << 16)
    }

    /// Independent sums a single workgroup carries (`lanes * slots == 64`).
    fn reduction_slots(&self) -> u32 {
        WG_SIZE / self.reduction_lanes_for_weights()
    }

    fn bias_slots(&self) -> u32 {
        WG_SIZE / self.reduction_lanes_for_bias()
    }
}

/// `@workgroup_size(64)` in every convolution shader.
const WG_SIZE: u32 = 64;

impl LayerType for ConvolutionType {
    fn get_forward_shader(&self) -> ShaderDescriptor {
        ShaderDescriptor {
            label: "convolution",
            source: include_str!("../shader/convolution.wgsl"),
        }
    }

    fn get_backward_shader(&self) -> Option<ShaderDescriptor> {
        Some(ShaderDescriptor {
            label: "back_convolution",
            source: include_str!("../shader/back_convolution.wgsl"),
        })
    }

    fn has_weights(&self) -> bool {
        true
    }

    fn get_dim_input(&self) -> Dim3 {
        self.dim_input
    }

    fn get_dim_output(&self) -> Dim3 {
        self.dim_output
    }

    fn get_forward_buffer_bindings(&self, batch: Batch) -> Vec<ForwardBufferBinding> {
        self.get_buffers_specs(batch)
            .into_iter()
            .map(|(name, spec)| ForwardBufferBinding {
                init: match name.as_str() {
                    "weights" => BufferInit::RandomWeights,
                    "specs" => BufferInit::SpecsUniform,
                    _ => BufferInit::None,
                },
                source: if name == "input" {
                    ForwardBufferSource::PreviousOutput
                } else {
                    ForwardBufferSource::Allocate
                },
                name,
                spec,
            })
            .collect()
    }

    fn get_back_buffer_bindings(&self, batch: Batch) -> Vec<BackwardBufferBinding> {
        self.get_back_buffers_specs(batch)
            .into_iter()
            .enumerate()
            .map(|(index, (name, spec))| BackwardBufferBinding {
                name,
                spec,
                source: match index {
                    0 => BackwardBufferSource::Forward(0),
                    1 => BackwardBufferSource::Forward(1),
                    2 => BackwardBufferSource::Forward(3),
                    3 => BackwardBufferSource::IncomingGradient,
                    _ => BackwardBufferSource::Allocate,
                },
            })
            .collect()
    }

    fn get_back_grad_input_index(&self) -> Option<usize> {
        Some(4)
    }

    fn get_optimizer_bindings(&self) -> Option<OptimizerBindings> {
        Some(OptimizerBindings {
            weight_count: (self.dim_kernel.bytes_size() * self.nb_kernel) / 4,
            weights_forward_index: 1,
            bias_forward_index: 2,
            grad_weights_backward_index: 5,
            grad_bias_backward_index: 6,
        })
    }

    fn get_weight_fan_in(&self) -> Option<u32> {
        // One output value sums kh·kw·c_in products.
        Some((self.dim_kernel.x * self.dim_kernel.y * self.dim_kernel.z).max(1))
    }

    fn get_back_entrypoints(&self) -> Vec<&'static str> {
        // Three separate sub-passes to avoid write races:
        //   1. grad_input  — one thread per input element
        //   2. grad_weights — one thread per weight element
        //   3. grad_bias    — one thread per kernel
        vec!["conv_back_input", "conv_back_weights", "conv_back_bias"]
    }

    fn get_back_workgroup_counts(&self, batch: Batch) -> Vec<u32> {
        // A workgroup carries `reduction_slots()` independent sums, each split
        // across `64 / slots` lanes. The batch splits the three sub-passes in two:
        // `grad_input` writes one activation per thread (scales with batch);
        // `grad_weights`/`grad_bias` write PARAMETERS — as many sums as weights
        // whatever the batch, so the batch enters inside the kernel as
        // `batch * OH * OW` positions (one `+=` per weight per STEP, not per sample).
        // The bias pass gets its own slot count (`nb_kernel` sums, not the weights').
        vec![
            (self.dim_input.length() * batch).div_ceil(WG_SIZE),
            (self.dim_kernel.length() * self.nb_kernel).div_ceil(self.reduction_slots()),
            self.nb_kernel.div_ceil(self.bias_slots()),
        ]
    }

    fn set_dim_input(&mut self, input: Dim3) {
        self.dim_input = input;
    }

    fn set_dim_output(&mut self) -> Result<Dim3, ModelError> {
        if self.stride == 0 {
            return Err(ModelError::InvalidStride {
                stride: self.stride,
            });
        }
        // The forward/backward shaders iterate the kernel's channel axis over
        // dim_input.z and size the weight buffer from dim_kernel; a mismatch
        // silently corrupts the convolution (this exact footgun broke the
        // diffusion timestep-conditioning fix).
        if self.dim_kernel.z != self.dim_input.z {
            return Err(ModelError::KernelChannelMismatch {
                input: self.dim_input,
                kernel: self.dim_kernel,
            });
        }
        let x = match self.mode {
            PaddingMode::Valid => {
                let delta = self.dim_input.x.checked_sub(self.dim_kernel.x).ok_or(
                    ModelError::KernelLargerThanInput {
                        input: self.dim_input,
                        kernel: self.dim_kernel,
                        mode: self.mode,
                    },
                )?;
                (delta / self.stride) + 1
            }
            PaddingMode::Same => self.dim_input.x.div_ceil(self.stride),
        };
        let y = match self.mode {
            PaddingMode::Valid => {
                let delta = self.dim_input.y.checked_sub(self.dim_kernel.y).ok_or(
                    ModelError::KernelLargerThanInput {
                        input: self.dim_input,
                        kernel: self.dim_kernel,
                        mode: self.mode,
                    },
                )?;
                (delta / self.stride) + 1
            }
            PaddingMode::Same => self.dim_input.y.div_ceil(self.stride),
        };
        let z = self.nb_kernel;
        self.dim_output = Dim3::new((x, y, z));
        Ok(self.dim_output)
    }

    fn get_buffers_specs(&self, batch: Batch) -> Vec<(String, BufferSpec)> {
        vec![
            // [0] input — shared with previous layer's output
            (
                "input".to_string(),
                BufferSpec {
                    size: batched_bytes(self.dim_input, batch).max(4),
                    usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC | BufferUsages::STORAGE,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                },
            ),
            // [1] weights — initialised with random values
            (
                "weights".to_string(),
                BufferSpec {
                    size: self.kernel_bytes().max(4),
                    usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC | BufferUsages::STORAGE,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                },
            ),
            // [2] bias — initialised to zero
            (
                "bias".to_string(),
                BufferSpec {
                    size: (self.nb_kernel * 4).max(4),
                    usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC | BufferUsages::STORAGE,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                },
            ),
            // [3] specs uniform
            (
                "specs".to_string(),
                BufferSpec {
                    size: self.get_spec_uniform_bytes_size().max(4),
                    usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(
                            std::num::NonZeroU64::new(self.get_spec_uniform_bytes_size() as u64)
                                .unwrap(),
                        ),
                    },
                },
            ),
            // [4] output
            (
                "output".to_string(),
                BufferSpec {
                    size: batched_bytes(self.dim_output, batch).max(4),
                    usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC | BufferUsages::STORAGE,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                },
            ),
        ]
    }

    fn get_back_buffers_specs(&self, batch: Batch) -> Vec<(String, BufferSpec)> {
        // Backward bind group (shared by all three sub-passes): [0] fwd_input,
        // [1] weights, [2] specs (shared from forward), [3] grad_output (incoming),
        // [4] grad_input (outgoing), [5] grad_weights, [6] grad_bias.
        let read_storage = |size: u32| BufferSpec {
            size: size.max(4),
            usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC | BufferUsages::STORAGE,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
        };
        let write_storage = |size: u32| BufferSpec {
            size: size.max(4),
            usage: BufferUsages::COPY_DST | BufferUsages::COPY_SRC | BufferUsages::STORAGE,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
        };
        vec![
            (
                "fwd_input".to_string(),
                read_storage(batched_bytes(self.dim_input, batch)),
            ),
            ("weights".to_string(), read_storage(self.kernel_bytes())),
            (
                "specs".to_string(),
                BufferSpec {
                    size: self.get_spec_uniform_bytes_size().max(4),
                    usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: Some(
                            std::num::NonZeroU64::new(self.get_spec_uniform_bytes_size() as u64)
                                .unwrap(),
                        ),
                    },
                },
            ),
            (
                "grad_output".to_string(),
                read_storage(batched_bytes(self.dim_output, batch)),
            ),
            (
                "grad_input".to_string(),
                write_storage(batched_bytes(self.dim_input, batch)),
            ),
            (
                "grad_weights".to_string(),
                write_storage(self.kernel_bytes()),
            ),
            ("grad_bias".to_string(), write_storage(self.nb_kernel * 4)),
        ]
    }

    fn get_spec_uniform_bytes_size(&self) -> u32 {
        ConvolutionUniform::SHADER_SIZE.get() as u32
    }

    fn get_spec_uniform_bytes(&self) -> Vec<u8> {
        let uniform = ConvolutionUniform {
            nb_kernel: self.nb_kernel,
            stride: self.stride,
            padding_mode: match self.mode {
                PaddingMode::Valid => 0,
                PaddingMode::Same => 1,
            },
            reduction_lanes: Self::pack_lanes(
                self.reduction_lanes_for_weights(),
                self.reduction_lanes_for_bias(),
            ),
            dim_kernel: self.dim_kernel,
            dim_input: self.dim_input,
            dim_output: self.dim_output,
        };
        let mut buffer = UniformBuffer::new(Vec::new());
        buffer
            .write(&uniform)
            .expect("failed to encode convolution uniform");
        buffer.into_inner()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_dim_output_rejects_kernel_depth_not_matching_input_channels() {
        // input has 3 channels but kernel depth is 1 — the exact footgun that
        // silently mis-sized the weight buffer when adding timestep channels.
        let mut conv = ConvolutionType::new(
            Dim3::new((8, 8, 3)),
            4,
            Dim3::new((3, 3, 1)),
            1,
            PaddingMode::Same,
        );
        assert!(matches!(
            conv.set_dim_output(),
            Err(ModelError::KernelChannelMismatch { .. })
        ));
    }

    #[test]
    fn set_dim_output_accepts_matching_kernel_depth() {
        let mut conv = ConvolutionType::new(
            Dim3::new((8, 8, 3)),
            4,
            Dim3::new((3, 3, 3)),
            1,
            PaddingMode::Same,
        );
        assert!(conv.set_dim_output().is_ok());
    }
}
