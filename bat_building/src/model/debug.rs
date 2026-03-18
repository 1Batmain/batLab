//! File purpose: Implements debug functionality for model execution, state, or diagnostics.

use std::fmt;

use crate::gpu_context::GpuContext;
use crate::model::layer::Layer;
use crate::model::layer_types::{LayerType, LayerTypes};

// ---------------------------------------------------------------------------
// GPU buffer readback
// ---------------------------------------------------------------------------

/// Read an f32 buffer back from the GPU synchronously.
/// Returns `None` if the buffer has no `COPY_SRC` usage (e.g. uniform buffers).
pub(crate) fn read_back_f32(
    gpu: &GpuContext,
    buf: &wgpu::Buffer,
    size_bytes: u64,
) -> Option<Vec<f32>> {
    if size_bytes == 0 {
        return Some(vec![]);
    }
    if !buf.usage().contains(wgpu::BufferUsages::COPY_SRC) {
        return None;
    }

    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("debug_staging"),
        size: size_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, size_bytes);
    gpu.queue.submit([encoder.finish()]);

    let slice = staging.slice(..);
    let (tx, rx) = futures::channel::oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });

    if gpu
        .device
        .poll(wgpu::PollType::wait_indefinitely())
        .is_err()
    {
        return None;
    }
    match pollster::block_on(async { rx.await }) {
        Ok(Ok(())) => {}
        _ => return None,
    }

    let values = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let bytes = slice.get_mapped_range();
        bytemuck::cast_slice::<u8, f32>(&bytes).to_vec()
    })) {
        Ok(values) => values,
        Err(_) => {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| staging.unmap()));
            return None;
        }
    };
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| staging.unmap()));
    Some(values)
}

// ---------------------------------------------------------------------------
// f32 slice summary
// ---------------------------------------------------------------------------

const PREVIEW_LEN: usize = 6;

pub(crate) fn summarize_f32(values: &[f32]) -> String {
    let preview: Vec<String> = values
        .iter()
        .take(PREVIEW_LEN)
        .map(|v| format!("{v:.4}"))
        .collect();
    let ellipsis = if values.len() > PREVIEW_LEN {
        ", ..."
    } else {
        ""
    };

    let has_nan = values.iter().any(|v| v.is_nan());
    let has_inf = values.iter().any(|v| v.is_infinite());
    let finite: Vec<f32> = values.iter().copied().filter(|v| v.is_finite()).collect();

    let stats = if !finite.is_empty() {
        let min = finite.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = finite.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean = finite.iter().sum::<f32>() / finite.len() as f32;
        format!("min={min:.4} max={max:.4} mean={mean:.4}")
    } else {
        "no finite values".to_string()
    };

    let mut warnings = String::new();
    if has_nan {
        warnings.push_str(" ⚠ NaN");
    }
    if has_inf {
        warnings.push_str(" ⚠ Inf");
    }

    format!(
        "[{}{}] {} ({} elems){}",
        preview.join(", "),
        ellipsis,
        stats,
        values.len(),
        warnings
    )
}

// ---------------------------------------------------------------------------
// LayerDebugView — formats a single layer with GPU buffer readback
// ---------------------------------------------------------------------------

pub(crate) struct LayerDebugView<'a> {
    pub idx: usize,
    pub layer: &'a Layer,
    pub gpu: &'a GpuContext,
}

impl fmt::Debug for LayerDebugView<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let layer = self.layer;

        let ty_name = match &layer.ty {
            LayerTypes::Convolution(_) => "Convolution",
            LayerTypes::Activation(_) => "Activation",
            LayerTypes::Concat(_) => "Concat",
            LayerTypes::FullyConnected(_) => "FullyConnected",
            LayerTypes::GroupNorm(_) => "GroupNorm",
            LayerTypes::UpsampleConv(_) => "UpsampleConv",
            LayerTypes::Loss(_) => "Loss",
        };

        let dim_in = layer.ty.get_dim_input();
        let dim_out = layer.ty.get_dim_output();

        let mut s = f.debug_struct(&format!("Layer[{}]::{}", self.idx, ty_name));

        s.field(
            "input_dim",
            &format!(
                "{}x{}x{} ({} elems, {} B)",
                dim_in.x,
                dim_in.y,
                dim_in.z,
                dim_in.length(),
                dim_in.bytes_size()
            ),
        );
        s.field(
            "output_dim",
            &format!(
                "{}x{}x{} ({} elems, {} B)",
                dim_out.x,
                dim_out.y,
                dim_out.z,
                dim_out.length(),
                dim_out.bytes_size()
            ),
        );

        s.field("fwd_entrypoint", &layer.ty.get_entrypoint());
        s.field("fwd_workgroups", &layer.num_workgroups);

        let back_eps = layer.ty.get_back_entrypoints();
        let back_wgs = layer.ty.get_back_workgroup_counts();
        if !back_eps.is_empty() {
            let bwd_passes: Vec<String> = back_eps
                .iter()
                .zip(&back_wgs)
                .map(|(ep, wg)| format!("{ep} ({wg} wg)"))
                .collect();
            s.field("bwd_passes", &bwd_passes);
        }

        // Forward buffers
        let fwd_specs = layer.ty.get_buffers_specs();
        let fwd_descs: Vec<String> = layer
            .buffers
            .forward
            .iter()
            .enumerate()
            .map(|(i, buf)| {
                let name = fwd_specs.get(i).map(|(n, _)| n.as_str()).unwrap_or("?");
                let id = std::sync::Arc::as_ptr(buf) as usize;
                let size = buf.size();
                let content_str = match read_back_f32(self.gpu, buf, size) {
                    Some(v) => summarize_f32(&v),
                    None => "(uniform — no readback)".to_string(),
                };
                format!("[{i}] {name:<14} id={id:#x}  {size:>9} B  {content_str}")
            })
            .collect();
        s.field("fwd_buffers", &fwd_descs);

        // Backward buffers (present after build())
        if let Some(bwd_bufs) = &layer.buffers.backward {
            let bwd_specs = layer.ty.get_back_buffers_specs();
            let bwd_descs: Vec<String> = bwd_bufs
                .iter()
                .enumerate()
                .map(|(i, buf)| {
                    let name = bwd_specs.get(i).map(|(n, _)| n.as_str()).unwrap_or("?");
                    let id = std::sync::Arc::as_ptr(buf) as usize;
                    let size = buf.size();
                    let content_str = match read_back_f32(self.gpu, buf, size) {
                        Some(v) => summarize_f32(&v),
                        None => "(uniform — no readback)".to_string(),
                    };
                    format!("[{i}] {name:<14} id={id:#x}  {size:>9} B  {content_str}")
                })
                .collect();
            s.field("bwd_buffers", &bwd_descs);
        }

        if let Some(opt) = &layer.opt_pass {
            s.field("opt_workgroups", &opt.num_workgroups);
        }

        s.finish()
    }
}
