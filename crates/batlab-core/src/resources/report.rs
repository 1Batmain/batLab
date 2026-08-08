//! File purpose: Renders a [`GpuInventory`] as text, once, for every reader.
//!
//! The CLI (`--resources`) and the TUI's Resources page print the same lines
//! from the same function. That is deliberate: a page and a script that
//! disagree about a number are worse than either alone, and the numbers here
//! are the kind someone quotes.
//!
//! Width is a parameter rather than an assumption. The repository's own habit
//! (`UX_NAV.md`, the architecture panel) is to elide in the middle and say so,
//! never to truncate in silence — a table cut short reads as a complete table.

use super::{format_bytes, GpuInventory, Kind, ProfileSource, Workload};
use crate::gpu_context::GpuLimitsProfile;
use crate::transfers::TransferRate;

/// Measured traffic, to print beside the predicted memory.
///
/// Optional throughout: the inventory is computable with no GPU at all, and a
/// page that cannot measure says so rather than showing a plausible zero.
#[derive(Debug, Clone, Copy, Default)]
pub struct MeasuredTransfers {
    /// Per training step, averaged over the steps that were run.
    pub training_step: Option<TransferRate>,
    /// Per reverse (denoising) step of the sampler.
    pub inference_step: Option<TransferRate>,
    /// Bytes uploaded once, when the graph was built: the weights themselves.
    pub build_upload_bytes: Option<u64>,
    /// Reverse steps in one image — how many times the per-step row happens.
    pub steps_per_image: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct ReportOptions {
    pub width: usize,
    /// Layer rows to show, `None` for all. The TUI passes what fits.
    pub max_layer_rows: Option<usize>,
    /// Whether to draw the proportional bars.
    pub bars: bool,
}

impl Default for ReportOptions {
    fn default() -> Self {
        Self {
            width: 100,
            max_layer_rows: None,
            bars: true,
        }
    }
}

/// The whole page, as lines.
pub fn report_lines(
    inventory: &GpuInventory,
    measured: Option<&MeasuredTransfers>,
    options: ReportOptions,
) -> Vec<String> {
    let mut out = Vec::new();
    // A page narrower than this cannot hold a byte figure and a label at all;
    // clamping here means every section can assume it.
    let options = ReportOptions {
        width: options.width.max(28),
        ..options
    };
    let width = options.width;
    device_section(inventory, width, &mut out);
    out.push(String::new());
    workload_section(inventory, width, &mut out);
    out.push(String::new());
    memory_section(inventory, options, &mut out);
    out.push(String::new());
    layers_section(inventory, options, &mut out);
    out.push(String::new());
    limits_section(inventory, width, &mut out);
    out.push(String::new());
    execution_model_section(inventory, options, &mut out);
    if let Some(measured) = measured {
        out.push(String::new());
        traffic_section(inventory, measured, width, &mut out);
    }
    out.push(String::new());
    verdict_section(inventory, width, &mut out);
    out
}

fn device_section(inventory: &GpuInventory, width: usize, out: &mut Vec<String>) {
    let device = &inventory.device;
    let qualifier = match device.source {
        ProfileSource::Measured => "this machine",
        ProfileSource::Hypothetical => "hypothetical",
    };
    out.push(heading(&format!("DEVICE  {} ({qualifier})", device.name), width));
    if device.source == ProfileSource::Measured {
        out.push(wrap(
            &format!(
                "  {} · {} · {}",
                device.backend,
                device.device_type,
                if device.unified_memory {
                    "unified memory (host and GPU share it)"
                } else {
                    "discrete memory"
                }
            ),
            width,
        ));
    }
    // Named first, because every number under it follows from it: the limits
    // below are what `request_device` was ASKED for, not what the adapter can
    // do, and the gap between the two profiles is four orders of magnitude on
    // this machine. A page that showed 256 MiB without saying "because we asked
    // for the web baseline" reads as a hardware fact.
    if let Some(profile) = device.limits_profile {
        out.push(row(
            "limits profile",
            &format!(
                "{} — {}",
                profile.label(),
                match profile {
                    GpuLimitsProfile::Native => "the adapter's own limits",
                    GpuLimitsProfile::Web =>
                        "the WebGPU baseline, what a browser grants unasked",
                }
            ),
            width,
        ));
    }
    out.push(row(
        "max_buffer_size",
        &format_bytes(device.max_buffer_size),
        width,
    ));
    out.push(row(
        "max_storage_buffer_binding_size",
        &format_bytes(device.max_storage_buffer_binding_size),
        width,
    ));
    // Shown next to the device's, and captioned, because the two are four
    // orders of magnitude apart on this machine and the codebase spells both
    // `max_buffer_size`. It is what the dataset sizes its chunks from.
    out.push(row(
        "adapter would allow, per buffer",
        &format_bytes(device.adapter_max_buffer_size),
        width,
    ));
    out.push(row(
        "max_compute_workgroups/dimension",
        &device.max_compute_workgroups_per_dimension.to_string(),
        width,
    ));
    out.push(row(
        "max_storage_buffers/stage",
        &device.max_storage_buffers_per_shader_stage.to_string(),
        width,
    ));
    match device.memory_budget {
        Some(budget) => out.push(row(
            "memory budget (stated)",
            &format_bytes(budget),
            width,
        )),
        // Said out loud, because its absence is the reason the verdict is about
        // per-binding limits rather than about a fill level.
        None => out.push(wrap(
            "  memory budget — not exposed: neither wgpu nor WebGPU reports one, so \
             the verdict below is about per-binding limits, not about a fill level.",
            width,
        )),
    }
}

fn workload_section(inventory: &GpuInventory, width: usize, out: &mut Vec<String>) {
    let description = match inventory.workload {
        Workload::Inference => "inference · batch 1 (the sampler runs one path at a time)".to_string(),
        Workload::Training {
            optimizer,
            ema,
            diffusion,
            ..
        } => format!(
            "training · batch {} · {optimizer:?}{}{}",
            inventory.batch,
            if ema { " · EMA" } else { "" },
            if diffusion { " · diffusion prepass" } else { "" }
        ),
    };
    out.push(heading(&format!("WORKLOAD  {description}"), width));
    out.push(wrap(
        &format!(
            "  {} trainable scalars, {} of weights",
            thousands(inventory.parameters),
            format_bytes(inventory.bytes_of(Kind::Weights))
        ),
        width,
    ));
}

fn memory_section(inventory: &GpuInventory, options: ReportOptions, out: &mut Vec<String>) {
    let width = options.width;
    let total = inventory.total_bytes();
    let (scaling, fixed) = inventory.batch_split();
    out.push(heading(
        &format!(
            "GPU MEMORY  {} total · {} resident · {} streamed",
            format_bytes(total),
            format_bytes(inventory.resident_bytes()),
            format_bytes(total - inventory.resident_bytes())
        ),
        width,
    ));
    for (kind, bytes) in inventory.by_kind() {
        let share = if total == 0 {
            0.0
        } else {
            bytes as f64 / total as f64
        };
        let bar = if options.bars && width >= 72 {
            format!("{}  ", bar_of(share, 14))
        } else {
            String::new()
        };
        // The `×batch` marker is a fixed-width column rather than a suffix:
        // as a suffix it made every batch-scaling row two characters longer
        // than the rest, and a table whose numbers do not line up is a table
        // nobody compares down a column.
        let value = format!(
            "{:>6} {:>10}  {bar}{:>5.1}%",
            if kind.scales_with_batch() {
                "×batch"
            } else {
                ""
            },
            format_bytes(bytes),
            share * 100.0,
        );
        out.push(row(kind.label(), &value, width));
    }
    out.push(wrap(
        &format!(
            "  → {} scales with the batch, {} does not.",
            format_bytes(scaling),
            format_bytes(fixed)
        ),
        width,
    ));
}

fn layers_section(inventory: &GpuInventory, options: ReportOptions, out: &mut Vec<String>) {
    out.push(heading(
        "PER LAYER  parameters · GPU bytes",
        options.width,
    ));
    let rows = &inventory.layers;
    let limit = options.max_layer_rows.unwrap_or(rows.len()).max(3);
    if rows.len() <= limit {
        for row in rows {
            out.push(layer_line(row, options.width));
        }
        return;
    }
    // Elide in the MIDDLE and say how much was elided — the head and the tail
    // are where the interesting layers are, and a silent truncation reads like
    // a complete list.
    let head = limit / 2;
    let tail = limit - head - 1;
    for row in &rows[..head] {
        out.push(layer_line(row, options.width));
    }
    out.push(format!("  … {} more layers …", rows.len() - head - tail));
    for row in &rows[rows.len() - tail..] {
        out.push(layer_line(row, options.width));
    }
}

fn layer_line(footprint: &super::LayerFootprint, width: usize) -> String {
    row(
        &format!("{:>2} {}", footprint.index, footprint.display),
        &format!(
            "{:>9} {:>10}",
            thousands(footprint.parameters),
            format_bytes(footprint.bytes)
        ),
        width,
    )
}

fn limits_section(inventory: &GpuInventory, width: usize, out: &mut Vec<String>) {
    out.push(heading("HEADROOM", width));
    if let Some(largest) = inventory.largest_allocation() {
        out.push(wrap(
            &format!(
                "  largest buffer — {} ({}), against a {} binding cap.",
                format_bytes(largest.bytes),
                largest.label,
                format_bytes(inventory.device.max_storage_buffer_binding_size)
            ),
            width,
        ));
    }
    let per_dim = inventory.device.max_compute_workgroups_per_dimension as u64;
    let split = if inventory.largest_dispatch > per_dim {
        ", dispatched as a 2-D grid"
    } else {
        ""
    };
    out.push(wrap(
        &format!(
            "  largest dispatch — {} workgroups against {per_dim} per dimension{split}.",
            thousands(inventory.largest_dispatch)
        ),
        width,
    ));
    if inventory.attention_probs_bytes > 0 {
        out.push(wrap(
            &format!(
                "  attention — {} of softmax rows, N² per sample over every attention \
                 layer: the term that grows quadratically with the resolution it \
                 attends over.",
                format_bytes(inventory.attention_probs_bytes)
            ),
            width,
        ));
    }
}

fn execution_model_section(
    inventory: &GpuInventory,
    options: ReportOptions,
    out: &mut Vec<String>,
) {
    out.push(heading("EXECUTION MODEL", options.width));
    // The question this page was built to answer, answered in words before it
    // is answered in numbers.
    out.push(wrap(
        "  resident — the whole model is uploaded once, at build time, and stays \
         on the GPU: weights, gradients, optimiser state and every activation \
         buffer. Nothing is paged in layer by layer; a step touches no host \
         memory for the model itself.",
        options.width,
    ));
    match inventory.dataset {
        // One chunk is a different execution model, not a smaller number of the
        // same one: the dataset stops being streamed at all, and the per-step
        // host→GPU traffic goes to zero rather than down. Worth its own
        // sentence, because it is the difference a limits profile makes.
        Some(plan) if plan.chunk_count <= 1 => out.push(wrap(
            &format!(
                "  resident — the dataset too. All {} of it sits in one buffer, uploaded \
                 once at the first step and never again: a batch of {} costs NO host→GPU \
                 dataset traffic. (This needs a device that will bind a buffer that big — \
                 see the limits profile above.)",
                format_bytes(plan.total_bytes),
                inventory.batch,
            ),
            options.width,
        )),
        Some(plan) => out.push(wrap(
            &format!(
                "  streamed — the dataset, and only the dataset. {} is cut into {} \
                 chunk(s) of {}, and exactly ONE is resident at a time. A shuffled \
                 batch of {} touches {:.2} distinct chunks per step, i.e. about {} \
                 of host→GPU traffic.",
                format_bytes(plan.total_bytes),
                plan.chunk_count,
                format_bytes(plan.chunk_bytes),
                inventory.batch,
                plan.expected_uploads_per_step,
                format_bytes(plan.expected_upload_bytes_per_step as u64)
            ),
            options.width,
        )),
        None => out.push(wrap(
            "  streamed — nothing: this inventory was asked without a dataset.",
            options.width,
        )),
    }
    if inventory.workload == Workload::Inference {
        out.push(wrap(
            "  the reverse chain, however, keeps its latent on the CPU: every \
             denoising step writes the composed input up and maps the predicted ε \
             back down. One full round trip per step, 256 per image.",
            options.width,
        ));
    }
}

fn traffic_section(
    inventory: &GpuInventory,
    measured: &MeasuredTransfers,
    width: usize,
    out: &mut Vec<String>,
) {
    out.push(heading("TRAFFIC PER STEP (measured, not estimated)", width));
    if let Some(bytes) = measured.build_upload_bytes {
        out.push(row(
            "build, uploaded once",
            &format_bytes(bytes),
            width,
        ));
    }
    let step_row = |label: &str, rate: TransferRate| {
        row(
            label,
            &format!(
                "{:>10} up {:>10} down {:>4.1} rt {:>4.1} sub",
                format_bytes(rate.host_to_device_bytes as u64),
                format_bytes(rate.device_to_host_bytes as u64),
                rate.round_trips,
                rate.submits
            ),
            width,
        )
    };
    if let Some(rate) = measured.training_step {
        out.push(step_row("train step", rate));
    }
    if let Some(rate) = measured.inference_step {
        out.push(step_row("reverse step", rate));
        if measured.steps_per_image > 0 {
            out.push(wrap(
                &format!(
                    "  → {} reverse steps per image: {} CPU↔GPU round trips, {} moved \
                     in total. The latent lives on the CPU between steps.",
                    measured.steps_per_image,
                    (rate.round_trips * measured.steps_per_image as f64).round() as u64,
                    format_bytes(
                        ((rate.host_to_device_bytes + rate.device_to_host_bytes)
                            * measured.steps_per_image as f64) as u64
                    )
                ),
                width,
            ));
        }
    }
    let _ = inventory;
}

fn verdict_section(inventory: &GpuInventory, width: usize, out: &mut Vec<String>) {
    if inventory.fits() {
        out.push(heading("VERDICT  fits", width));
    } else {
        out.push(heading("VERDICT  DOES NOT FIT", width));
        for obstacle in &inventory.obstacles {
            out.push(wrap(&format!("  · {obstacle}"), width));
        }
    }
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

/// `  label ......... value`, right-aligned to `width`, never wider than it.
///
/// When the two cannot share a line the value drops to its own indented line
/// rather than being cut: a number cut in half is worse than a number moved.
fn row(label: &str, value: &str, width: usize) -> String {
    let value_len = value.chars().count();
    let available = width.saturating_sub(3 + value_len);
    if available < 8 {
        return format!(
            "  {}\n    {}",
            elide(label, width.saturating_sub(2)),
            elide(value, width.saturating_sub(4))
        );
    }
    format!(
        "  {:<available$} {}",
        elide(label, available),
        value,
        available = available
    )
}

/// A section title, never wider than the page.
fn heading(text: &str, width: usize) -> String {
    elide(text, width)
}

fn bar_of(share: f64, width: usize) -> String {
    let filled = ((share * width as f64).round() as usize).min(width);
    format!("{}{}", "█".repeat(filled), "·".repeat(width - filled))
}

/// Thin-space thousands, so a seven-digit parameter count is readable.
pub fn thousands(value: u64) -> String {
    let digits = value.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(' ');
        }
        out.push(c);
    }
    out
}

/// Cut a string to `width`, marking the cut. Never silently.
fn elide(text: &str, width: usize) -> String {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= width {
        return text.to_string();
    }
    if width <= 1 {
        return "…".to_string();
    }
    chars[..width - 1].iter().collect::<String>() + "…"
}

/// Wrap a paragraph to `width`, keeping the two-space indent of continuation
/// lines so a wrapped sentence still reads as one item.
fn wrap(text: &str, width: usize) -> String {
    let width = width.max(24);
    let mut lines: Vec<String> = Vec::new();
    let mut current = String::new();
    for word in text.split_whitespace() {
        let candidate_len = if current.is_empty() {
            word.chars().count() + 2
        } else {
            current.chars().count() + 1 + word.chars().count()
        };
        if !current.is_empty() && candidate_len > width {
            lines.push(current);
            current = format!("    {word}");
        } else if current.is_empty() {
            current = format!("  {word}");
        } else {
            current.push(' ');
            current.push_str(word);
        }
    }
    if !current.is_empty() {
        lines.push(current);
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::{inventory, DeviceProfile, InventoryRequest};

    fn small_request() -> InventoryRequest {
        use crate::config::{ActivationMethod, LayerDraft, PaddingMode};
        InventoryRequest {
            layers: vec![
                LayerDraft::Convolution {
                    dim_input: (8, 8, 3),
                    nb_kernel: 4,
                    dim_kernel: (3, 3, 3),
                    stride: 1,
                    padding: PaddingMode::Same,
                    save_key: None,
                },
                LayerDraft::Activation {
                    dim_input: (8, 8, 4),
                    method: ActivationMethod::Silu,
                    save_key: None,
                },
            ],
            input_size: (8, 8, 3),
            batch: 4,
            workload: Workload::Inference,
            dataset: None,
            live_frame: false,
        }
    }

    /// A narrow terminal must not produce a line wider than it is: the repo's
    /// rule is to elide, never to overflow.
    #[test]
    fn every_line_fits_the_width_it_was_given() {
        let inv = inventory(&small_request(), &DeviceProfile::hypothetical("x", None)).unwrap();
        for width in [40usize, 60, 80, 120] {
            let lines = report_lines(
                &inv,
                None,
                ReportOptions {
                    width,
                    max_layer_rows: Some(4),
                    bars: true,
                },
            );
            for line in lines.iter().flat_map(|l| l.split('\n')) {
                assert!(
                    line.chars().count() <= width,
                    "width {width}: line of {} chars: {line:?}",
                    line.chars().count()
                );
            }
        }
    }

    /// An elided layer table says how many rows it hid.
    #[test]
    fn an_elided_table_says_what_it_hid() {
        let mut request = small_request();
        // Ten activation layers so there is something to elide.
        for _ in 0..8 {
            request.layers.push(crate::config::LayerDraft::Activation {
                dim_input: (8, 8, 4),
                method: crate::config::ActivationMethod::Relu,
                save_key: None,
            });
        }
        let inv = inventory(&request, &DeviceProfile::hypothetical("x", None)).unwrap();
        let lines = report_lines(
            &inv,
            None,
            ReportOptions {
                width: 100,
                max_layer_rows: Some(5),
                bars: false,
            },
        );
        let elision = lines
            .iter()
            .find(|l| l.contains("more layers"))
            .expect("no elision marker in a table that had to be cut");
        assert!(
            elision.contains("6 more layers"),
            "the marker must name the count: {elision:?}"
        );
    }

    /// The verdict is the line a reader looks for first, and it must name the
    /// obstacle rather than just refusing.
    #[test]
    fn a_refused_inventory_prints_the_limit_it_met() {
        let mut device = DeviceProfile::hypothetical("cramped", Some(1024));
        device.max_storage_buffer_binding_size = 512;
        let inv = inventory(&small_request(), &device).unwrap();
        let text = report_lines(&inv, None, ReportOptions::default()).join("\n");
        assert!(text.contains("DOES NOT FIT"), "{text}");
        assert!(
            text.contains("max_storage_buffer_binding_size"),
            "the verdict must name the limit: {text}"
        );
    }

    #[test]
    fn thousands_groups_digits() {
        assert_eq!(thousands(0), "0");
        assert_eq!(thousands(999), "999");
        assert_eq!(thousands(1_234_567), "1 234 567");
    }
}
