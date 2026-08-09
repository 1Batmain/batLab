//! File purpose: Defines the convolution layer type, shapes, and GPU bindings used by the model graph.

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
    /// Cooperating threads per sum, for the **two** reductions: the
    /// `grad_weights` split in the low half of the word, the `grad_bias` split
    /// in the high half (see [`ConvolutionType::pack_lanes`]).
    ///
    /// One word and not two because this uniform's layout is pinned: bytes
    /// 48..64 are a `Dim3` whose trailing word WGSL reads as `vec3` padding, so
    /// there is no free slot at the end to grow into without moving offsets the
    /// legacy fixtures bind against. It occupies the word that used to be
    /// explicit padding, so the uniform's size and every other field offset are
    /// unchanged.
    ///
    /// The two halves differ because the two reductions do: `grad_weights` has
    /// thousands of independent sums and wants few lanes, `grad_bias` has as
    /// many sums as there are kernels — 48 on the layer that cost 9,8 ms on a
    /// single workgroup — and wants as many lanes as it can get.
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
    ///
    /// Splitting a sum across lanes buys parallelism but costs a tree reduction
    /// with its barriers, so it only pays when there are too few sums to keep
    /// the GPU busy on their own. `conv3` has 9216 weights — already plenty —
    /// and measurably *lost* time when every sum was split 16 ways; `conv1` has
    /// 432 weights and 1024 positions, and gains ~9x from splitting maximally.
    ///
    /// The rule is therefore: use the fewest lanes that still put roughly
    /// `TARGET_THREADS` threads in flight, never splitting so far that a lane
    /// has fewer than `MIN_POSITIONS_PER_LANE` positions left to sum (below
    /// that the reduction costs more than the sum it replaces), and never past
    /// `MAX_LANES`.
    ///
    /// `MAX_LANES` is 32, not 64, on purpose: at 64 lanes a workgroup holds a
    /// single slot, so consecutive threads no longer walk consecutive `kz` and
    /// the coalescing the slot layout exists for is lost. Measured on conv4
    /// that costs 35% (0.0220 ms at 32 lanes vs 0.0297 at 64), and on conv1 it
    /// buys nothing (0.0291 vs 0.0290).
    ///
    /// Calibrated against `bench_conv_reduction_lanes`, which sweeps every
    /// legal lane count per layer. The resulting choices land within 1.4% of
    /// the per-layer optimum on all four convolutions of Greyscale_Diffusion:
    ///
    /// | layer | sums | positions | picked | best measured |
    /// |---|---:|---:|---:|---:|
    /// | conv1 |  432 | 1024 | 32 | 32 (0.0291 ms) |
    /// | conv2 | 4608 |  256 | 16 |  4 (0.0576 ms; 16 gives 0.0584) |
    /// | conv3 | 9216 |  256 |  8 |  8 (0.0965 ms) |
    /// | conv4 |  288 | 1024 | 32 | 32 (0.0220 ms) |
    ///
    /// # `TARGET_THREADS` has been recalibrated twice, and the second time says
    /// why the first expired
    ///
    /// 65 536 came from `bench_conv_reduction_lanes` — one kernel dispatched
    /// 200 times in a loop, on `Greyscale_Diffusion`, **before the batch axis
    /// existed**. `GPU_PROFILE.md` §6 re-swept it on the real step and got
    /// 262 144 (−4,2 % to −5,9 %).
    ///
    /// That value then expired the moment `conv_back_weights`' inner loop got
    /// cheap (`KERNEL_HUNT.md` §3): a lane used to pay four integer divisions
    /// per product and now pays two loads and an FMA, so the barriers of a deep
    /// split no longer buy back what they cost. Re-swept a third time, Σ of the
    /// timed passes of one step, minimum over 8 armed steps, two interleaved
    /// passes over the sweep agreeing to 0,3 %:
    ///
    /// | TARGET_THREADS | XL b8 | XL b32 | L b8 | L b32 |
    /// |---:|---:|---:|---:|---:|
    /// |    32 768        | 40.7 | **154.5** | 18.6 | 71.3 |
    /// | **131 072**      | **40.4** | 155.3 | **18.6** | **71.1** |
    /// |    262 144 (was) | 41.2 | 159.8 | 20.1 | 79.5 |
    /// |    524 288       | 45.0 | 179.8 |    — |    — |
    /// |  1 048 576       | 52.4 | 213.1 |    — |    — |
    ///
    /// 32 768 and 131 072 are a tie; the tie is broken **towards parallelism**
    /// (at 32 768 the biggest layers fall to a single lane per sum) because a
    /// device with less throughput per thread loses more to a starved dispatch
    /// than to a barrier. That is a portability argument, not a measurement.
    ///
    /// This recalibrates the constant; it does **not** fix the structural point
    /// `BATCH_DISPATCH.md` §9 raises — `positions` here is still the count *per
    /// sample*, while the loop covers `batch ×` more. Doing that properly means
    /// getting the batch into the uniform's lane word, and is left open.
    ///
    /// # The two numbers are knobs, not constants
    ///
    /// Both came out of a sweep on one Mac, and both describe *that machine's*
    /// balance between parallelism and barrier cost. They are read from
    /// [`crate::tuning`] (`BATLAB_CONV_TARGET_THREADS`,
    /// `BATLAB_CONV_MIN_POSITIONS_PER_LANE`) so the same sweep is one shell
    /// loop on a GPU nobody here owns — see `KERNEL_HUNT.md` §Portabilité. The
    /// defaults are unchanged.
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

    /// Lanes for the `grad_weights` reduction: one sum per weight, and there
    /// are thousands of them, so the rule usually answers "barely split".
    ///
    /// Capped at the number of **output rows**, because that is what the lanes
    /// now split. `conv_back_weights` walks `(sample, oy)` rows and runs `ox`
    /// densely inside them, so a lane numbered past `OH` would be handed no row
    /// at all on a batch of one — a thread asked for and then left idle. `OH`
    /// and not `OH * batch` because the uniform is written once at build time
    /// and does not know the batch; the cap is therefore conservative in the
    /// only direction that is safe. No layer of any model here is affected
    /// (every one has `OH >= 32 >= lanes`); it is a guard for the wide-and-short
    /// shapes nothing in this repo builds yet.
    pub(crate) fn reduction_lanes_for_weights(&self) -> u32 {
        let lanes = Self::reduction_lanes(
            self.dim_kernel.length() * self.nb_kernel,
            self.output_positions(),
        );
        // The largest power of two no greater than the row count — the lane
        // count has to stay a power of two for the tree reduction to halve it
        // down to 1.
        let rows = self.dim_output.x.max(1);
        lanes.min(1 << rows.ilog2())
    }

    /// Lanes for the `grad_bias` reduction — its **own** count, not the
    /// weights' one.
    ///
    /// The bias pass has exactly `nb_kernel` sums: 48 on `Color_Diffusion_XL`'s
    /// L26, against 41 472 for its weights. Feeding the weights' choice to both
    /// (which is what this did) left the bias with `nb_kernel / slots`
    /// workgroups — **one** workgroup, 48 threads, on the UpsampleConv of the
    /// same shape, walking 32 768 positions each. Measured 9,8 ms for a pass
    /// that reads 6,3 Mio: 0,6 Gio/s, i.e. latency-bound on a thread count that
    /// cannot cover it. The same rule fed the bias's own sum count answers 32
    /// lanes and 24 workgroups instead of 1.
    ///
    /// What this deliberately does **not** do is split the position axis across
    /// workgroups, which would need a scratch buffer and a second pass to
    /// combine them. Costed rather than assumed: 1 → 1536 threads already
    /// recovers essentially the whole pass, and the remaining ~0,3 ms of the
    /// step is not worth a buffer and a dispatch per layer.
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
        // grad_weights / grad_bias no longer run one thread per output element:
        // a workgroup carries `reduction_slots()` independent sums, each split
        // across `64 / slots` cooperating lanes.
        //
        // The batch axis splits the three sub-passes in two:
        //   - `grad_input` writes one activation per thread, so it scales with
        //     the batch like any elementwise pass;
        //   - `grad_weights` / `grad_bias` write *parameters*. There are
        //     exactly as many sums as there are weights whatever the batch, so
        //     their workgroup count is unchanged and the batch enters inside
        //     the kernel, as `batch * OH * OW` positions to reduce instead of
        //     `OH * OW`. That is the whole point of the exercise: one `+=` per
        //     weight and per step instead of one per weight and per sample.
        //
        // The bias pass gets its own slot count: it reduces `nb_kernel` sums,
        // not `dim_kernel.length() * nb_kernel` of them, and inheriting the
        // weights' split was what left it on a single workgroup.
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
        // Backward bind group layout (all three sub-passes share this layout):
        //   [0] fwd_input    — shared from forward[0]
        //   [1] weights      — shared from forward[1]
        //   [2] specs        — shared from forward[3]
        //   [3] grad_output  — incoming gradient (from next layer or loss)
        //   [4] grad_input   — outgoing gradient to previous layer  (NEW)
        //   [5] grad_weights — accumulated weight gradients         (NEW)
        //   [6] grad_bias    — accumulated bias gradients           (NEW)
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
