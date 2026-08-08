//! Tests for the GPU resource inventory.
//!
//! The load-bearing one is `the_predicted_inventory_matches_a_real_model`: the
//! prediction is a *mirror* of `Model::build`, and a mirror that has drifted is
//! worse than no mirror at all, because it answers confidently. Everything else
//! here checks a property of the prediction; that one checks the prediction.

use super::*;
use crate::config::{ActivationMethod, LayerDraft, PaddingMode};
use crate::model::LossMethod;
use crate::model::optimizer::OptimizerKind;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// A U-shaped stack: a skip that is saved, a stride-2 descent, an upsample, a
/// concat that fuses them, and an attention block.
///
/// Deliberately not a chain of convolutions. Concat is the only layer whose
/// three tensors have different per-sample lengths, `UpsampleConv` is the only
/// one whose output is larger than its input, and attention is the only one
/// with an `N²` scratch — a fixture without them would let a walker that
/// mishandles any of the three pass.
fn u_shaped_stack() -> (Vec<LayerDraft>, (u32, u32, u32)) {
    let input = (8, 8, 3);
    let layers = vec![
        LayerDraft::Convolution {
            dim_input: input,
            nb_kernel: 8,
            dim_kernel: (3, 3, 3),
            stride: 1,
            padding: PaddingMode::Same,
            save_key: None,
        },
        LayerDraft::GroupNorm {
            dim_input: (8, 8, 8),
            num_groups: 4,
            save_key: None,
        },
        LayerDraft::Activation {
            dim_input: (8, 8, 8),
            method: ActivationMethod::Silu,
            save_key: Some("skip8".to_string()),
        },
        LayerDraft::Convolution {
            dim_input: (8, 8, 8),
            nb_kernel: 16,
            dim_kernel: (3, 3, 8),
            stride: 2,
            padding: PaddingMode::Same,
            save_key: None,
        },
        LayerDraft::Attention {
            dim_input: (4, 4, 16),
            save_key: None,
        },
        LayerDraft::UpsampleConv {
            dim_input: (4, 4, 16),
            scale_factor: 2,
            nb_kernel: 8,
            dim_kernel: (3, 3, 16),
            padding: PaddingMode::Same,
            save_key: None,
        },
        LayerDraft::Concat {
            dim_input: (8, 8, 8),
            dim_skip: (8, 8, 8),
            skip_key: "skip8".to_string(),
            save_key: None,
        },
        LayerDraft::Convolution {
            dim_input: (8, 8, 16),
            nb_kernel: 1,
            dim_kernel: (3, 3, 16),
            stride: 1,
            padding: PaddingMode::Same,
            save_key: None,
        },
    ];
    (layers, input)
}

fn request(batch: u32, workload: Workload) -> InventoryRequest {
    let (layers, input_size) = u_shaped_stack();
    InventoryRequest {
        layers,
        input_size,
        batch,
        workload,
        dataset: None,
        live_frame: false,
    }
}

fn training(optimizer: OptimizerKind, ema: bool) -> Workload {
    Workload::Training {
        optimizer,
        ema,
        loss: LossMethod::MeanSquared,
        // The prepass is added by the trainer, not by `Model::build`, so the
        // comparison against a real built model turns it off.
        diffusion: false,
    }
}

fn test_device() -> DeviceProfile {
    DeviceProfile::hypothetical("test", None)
}

// ---------------------------------------------------------------------------
// The one that matters: prediction vs a model actually built on the GPU
// ---------------------------------------------------------------------------

/// The inventory must equal, byte for byte, what a model built from the same
/// layers allocates on the device.
///
/// `Model::estimated_gpu_bytes` sums the real `wgpu::Buffer`s, deduplicated by
/// `Arc` identity — so it charges a shared activation once, exactly like the
/// walk here charges an aliased binding once. That equality is the whole claim
/// of this module, and it is asserted at three batch sizes so that a walker
/// which got the batch axis onto the wrong half of the table cannot pass.
#[test]
fn the_predicted_inventory_matches_a_real_model() {
    pollster::block_on(async {
        use crate::model::{Model, Training};
        use std::sync::Arc;

        let gpu = Arc::new(crate::gpu_context::GpuContext::new_headless().await);
        let device = DeviceProfile::from_gpu(gpu.as_ref());
        let (layers, input_size) = u_shaped_stack();

        for (batch, optimizer, ema) in [
            (1u32, OptimizerKind::Sgd, false),
            (4, OptimizerKind::Adam, false),
            (4, OptimizerKind::Adam, true),
            (7, OptimizerKind::Adam, false),
        ] {
            let mut model = Model::<Training>::new_training_with_optimizer(
                Arc::clone(&gpu),
                1e-3,
                batch,
                LossMethod::MeanSquared,
                optimizer,
            )
            .await;
            if ema {
                model.set_ema(crate::model::EmaConfig::new(0.999));
            }
            for draft in &layers {
                model.add_draft(draft).expect("layer rejected");
            }
            model.build().expect("build failed");

            let predicted = inventory(
                &InventoryRequest {
                    layers: layers.clone(),
                    input_size,
                    batch,
                    workload: training(optimizer, ema),
                    dataset: None,
                    live_frame: false,
                },
                &device,
            )
            .expect("inventory failed");

            assert_eq!(
                predicted.total_bytes(),
                model.estimated_gpu_bytes(),
                "batch {batch}, {optimizer:?}, ema={ema}: predicted {} against {} really allocated",
                format_bytes(predicted.total_bytes()),
                format_bytes(model.estimated_gpu_bytes())
            );

            // Not just the total: the parameter count derived from the weight
            // BUFFERS must agree with the count derived from the config, which
            // `config::tests` in turn pins against the scalars a real
            // checkpoint holds. Two independent routes to the same number.
            let from_config: u64 = layers.iter().map(|l| l.parameter_count()).sum();
            assert_eq!(
                predicted.parameters, from_config,
                "batch {batch}: {} parameters from the buffers, {from_config} from the config",
                predicted.parameters
            );
        }
    });
}

/// The inference inventory must equal a model built for inference — which is
/// the forward graph alone.
///
/// Worth its own test because the difference between the two workloads is not a
/// scale factor: inference drops the loss layer, the whole backward chain, the
/// optimiser state and the EMA, and pins the batch to one. Getting *that* wrong
/// is how a page tells someone their 8 GB card cannot run a model it runs fine.
#[test]
fn the_inference_inventory_matches_a_model_built_for_inference() {
    pollster::block_on(async {
        use crate::model::{Infer, Model};
        use std::sync::Arc;

        let gpu = Arc::new(crate::gpu_context::GpuContext::new_headless().await);
        let device = DeviceProfile::from_gpu(gpu.as_ref());
        let (layers, input_size) = u_shaped_stack();

        let mut model = Model::<Infer>::new(Arc::clone(&gpu)).await;
        for draft in &layers {
            model.add_draft(draft).expect("layer rejected");
        }
        model.build_model().expect("build failed");

        let predicted = inventory(
            &InventoryRequest {
                layers: layers.clone(),
                input_size,
                batch: 1,
                workload: Workload::Inference,
                dataset: None,
                live_frame: false,
            },
            &device,
        )
        .expect("inventory failed");

        assert_eq!(predicted.total_bytes(), model.estimated_gpu_bytes());
        // And it is genuinely smaller than the training graph, not equal to it.
        let train = inventory(&request(1, training(OptimizerKind::Adam, false)), &device).unwrap();
        assert!(
            predicted.total_bytes() * 2 < train.total_bytes(),
            "inference {} is not meaningfully smaller than training {}",
            format_bytes(predicted.total_bytes()),
            format_bytes(train.total_bytes())
        );
    });
}

// ---------------------------------------------------------------------------
// Properties of the inventory, no GPU needed
// ---------------------------------------------------------------------------

/// The whole point of the batch column: activations scale, parameters do not.
#[test]
fn the_batch_moves_the_activations_and_leaves_the_parameters_alone() {
    let device = test_device();
    let one = inventory(&request(1, training(OptimizerKind::Adam, false)), &device).unwrap();
    let eight = inventory(&request(8, training(OptimizerKind::Adam, false)), &device).unwrap();

    for kind in [
        Kind::Weights,
        Kind::ParameterGradients,
        Kind::OptimizerState,
        Kind::Uniforms,
    ] {
        assert_eq!(
            one.bytes_of(kind),
            eight.bytes_of(kind),
            "{} moved with the batch",
            kind.label()
        );
    }
    for kind in [
        Kind::Activations,
        Kind::ActivationGradients,
        Kind::AttentionScratch,
        Kind::LossScratch,
    ] {
        assert_eq!(
            one.bytes_of(kind) * 8,
            eight.bytes_of(kind),
            "{} did not scale with the batch",
            kind.label()
        );
    }
}

/// Adam carries two moments per trainable scalar, SGD carries none, and an EMA
/// adds one more copy of the weights.
#[test]
fn the_optimizer_and_the_ema_are_multiples_of_the_weights() {
    let device = test_device();
    let sgd = inventory(&request(2, training(OptimizerKind::Sgd, false)), &device).unwrap();
    let adam = inventory(&request(2, training(OptimizerKind::Adam, false)), &device).unwrap();
    let adam_ema = inventory(&request(2, training(OptimizerKind::Adam, true)), &device).unwrap();

    assert_eq!(sgd.bytes_of(Kind::OptimizerState), 0, "SGD keeps no state");
    assert_eq!(
        adam.bytes_of(Kind::OptimizerState),
        2 * adam.bytes_of(Kind::Weights),
        "Adam must hold m and v for every scalar"
    );
    assert_eq!(
        adam_ema.bytes_of(Kind::Ema),
        adam_ema.bytes_of(Kind::Weights),
        "the EMA is one more copy of the weights"
    );
    assert_eq!(adam.bytes_of(Kind::Ema), 0, "no EMA was asked for");
}

/// Attention's softmax rows are `N²` per sample, and the inventory must be able
/// to say so — `ATTENTION.md` calls this the term that decides whether an
/// attention block is affordable at a given resolution.
#[test]
fn the_attention_probs_are_quadratic_in_the_positions() {
    let device = test_device();
    let small = inventory(&request(1, Workload::Inference), &device).unwrap();
    // The fixture attends over a 4×4 map: 16 positions, 16² floats per sample.
    assert_eq!(small.attention_probs_bytes, 16 * 16 * 4);

    let mut doubled = request(1, Workload::Inference);
    // Attend over 8×8 instead — four times the positions, sixteen times the
    // softmax block.
    doubled.layers[3] = LayerDraft::Convolution {
        dim_input: (8, 8, 8),
        nb_kernel: 16,
        dim_kernel: (3, 3, 8),
        stride: 1,
        padding: PaddingMode::Same,
        save_key: None,
    };
    doubled.layers[4] = LayerDraft::Attention {
        dim_input: (8, 8, 16),
        save_key: None,
    };
    doubled.layers[5] = LayerDraft::UpsampleConv {
        dim_input: (8, 8, 16),
        scale_factor: 1,
        nb_kernel: 8,
        dim_kernel: (3, 3, 16),
        padding: PaddingMode::Same,
        save_key: None,
    };
    let big = inventory(&doubled, &device).unwrap();
    assert_eq!(
        big.attention_probs_bytes,
        small.attention_probs_bytes * 16,
        "4x the positions must be 16x the softmax block"
    );
}

/// A layer's input IS the previous layer's output: one allocation, not two.
///
/// This is the sharing rule that decides the total, and the easiest one to get
/// wrong in a re-implementation — counting both sides would roughly double the
/// activation post, silently and plausibly.
#[test]
fn a_shared_activation_is_charged_once() {
    let device = test_device();
    let inv = inventory(&request(1, Workload::Inference), &device).unwrap();
    let outputs = inv
        .allocations
        .iter()
        .filter(|a| a.label == "output")
        .count();
    let inputs = inv
        .allocations
        .iter()
        .filter(|a| a.label == "input")
        .count();
    assert_eq!(
        outputs, 8,
        "each of the 8 layers owns exactly one output buffer"
    );
    assert_eq!(
        inputs, 1,
        "only the first layer allocates an input; every other one binds its \
         predecessor's output"
    );
}

// ---------------------------------------------------------------------------
// The verdict
// ---------------------------------------------------------------------------

/// A model that does not fit must say which limit it met, and a model that fits
/// must say nothing.
#[test]
fn the_verdict_names_the_limit_it_met() {
    let (layers, input_size) = u_shaped_stack();
    let request = InventoryRequest {
        layers,
        input_size,
        batch: 8,
        workload: training(OptimizerKind::Adam, false),
        dataset: None,
        live_frame: false,
    };

    let roomy = DeviceProfile::hypothetical("roomy", Some(8 * 1024 * 1024 * 1024));
    let inv = inventory(&request, &roomy).unwrap();
    assert!(inv.fits(), "obstacles on a roomy card: {:?}", inv.obstacles);

    // A budget just under the total: over budget, and nothing else.
    let total = inv.total_bytes();
    let tight = DeviceProfile::hypothetical("tight", Some(total - 1));
    let inv = inventory(&request, &tight).unwrap();
    assert!(!inv.fits());
    assert!(
        matches!(inv.obstacles.as_slice(), [Obstacle::OverBudget { .. }]),
        "expected a budget obstacle alone, got {:?}",
        inv.obstacles
    );

    // A device whose per-binding cap is smaller than one activation buffer:
    // the total is irrelevant, the build fails on a single allocation. This is
    // the WebGPU failure a browser hands out, and it is why the cap is checked
    // per buffer and not only in aggregate.
    let mut cramped = DeviceProfile::hypothetical("cramped", Some(8 * 1024 * 1024 * 1024));
    cramped.max_storage_buffer_binding_size = 4096;
    let inv = inventory(&request, &cramped).unwrap();
    assert!(!inv.fits());
    assert!(
        inv.obstacles
            .iter()
            .any(|o| matches!(o, Obstacle::BindingTooLarge { .. })),
        "expected a per-binding obstacle, got {:?}",
        inv.obstacles
    );
    // …and the uniforms are not flagged: they are bound as uniforms, and 32
    // bytes would otherwise trip a 4 KiB storage cap for no reason.
    assert!(
        !inv.obstacles.iter().any(|o| matches!(
            o,
            Obstacle::BindingTooLarge { label, .. } if label.contains("specs")
        )),
        "a uniform was judged against the storage binding cap"
    );
}

/// The batch ceiling must be the *last* batch that fits — the one after it must
/// not.
#[test]
fn the_batch_ceiling_is_the_last_batch_that_fits() {
    let (layers, input_size) = u_shaped_stack();
    let request = InventoryRequest {
        layers,
        input_size,
        batch: 1,
        workload: training(OptimizerKind::Adam, false),
        dataset: None,
        live_frame: false,
    };

    // A budget chosen to sit between two batches rather than at a round number.
    let one = inventory(&request, &DeviceProfile::hypothetical("x", None))
        .unwrap()
        .total_bytes();
    let eight = inventory(
        &request.clone().with_batch(8),
        &DeviceProfile::hypothetical("x", None),
    )
    .unwrap()
    .total_bytes();
    let budget = (one + eight) / 2;
    let device = DeviceProfile::hypothetical("mid", Some(budget));

    let ceiling = max_batch_that_fits(&request, &device, 4096)
        .unwrap()
        .expect("batch 1 must fit");
    assert!(
        inventory(&request.clone().with_batch(ceiling), &device)
            .unwrap()
            .fits(),
        "the reported ceiling {ceiling} does not fit"
    );
    assert!(
        !inventory(&request.clone().with_batch(ceiling + 1), &device)
            .unwrap()
            .fits(),
        "batch {} fits too, so {ceiling} was not the ceiling",
        ceiling + 1
    );

    // A card too small for a single sample answers `None`, not `Some(0)`.
    let hopeless = DeviceProfile::hypothetical("hopeless", Some(1024));
    assert_eq!(max_batch_that_fits(&request, &hopeless, 4096).unwrap(), None);
}

/// The ceiling is honoured even when memory would allow more, so a caller
/// asking "up to 1024?" never gets 2048 back.
#[test]
fn the_search_respects_its_ceiling() {
    let (layers, input_size) = u_shaped_stack();
    let request = InventoryRequest {
        layers,
        input_size,
        batch: 1,
        workload: training(OptimizerKind::Adam, false),
        dataset: None,
        live_frame: false,
    };
    let device = DeviceProfile::hypothetical("huge", Some(u64::MAX / 4));
    assert_eq!(
        max_batch_that_fits(&request, &device, 64).unwrap(),
        Some(64),
        "an unbounded card must return the ceiling itself"
    );
}

// ---------------------------------------------------------------------------
// The dataset
// ---------------------------------------------------------------------------

/// The predicted chunk plan must be the one `GpuDataset` really allocates.
///
/// This test exists because the prediction was wrong, by exactly a factor of
/// two, and nothing else noticed. `GpuDataset` sizes its chunk from
/// `GpuSpecs::memory_size()` — the **adapter's** `max_buffer_size`, 28 GiB on
/// this Mac — while the profile was handing `select_chunk_bytes` the
/// **device's** 256 MiB. Two fields spelled `max_buffer_size`, four orders of
/// magnitude apart, and the page confidently reported 64 MiB chunks against a
/// dataset that had allocated 128 MiB ones. Every downstream figure — the
/// resident footprint, the chunk count, the predicted traffic per step — was
/// half of the truth.
///
/// So the plan is checked against an allocation, not against arithmetic.
#[test]
fn the_predicted_chunk_plan_is_the_one_the_dataset_allocates() {
    pollster::block_on(async {
        use crate::training::GpuDataset;
        use std::sync::Arc;

        let gpu = Arc::new(crate::gpu_context::GpuContext::new_headless().await);
        let device = DeviceProfile::from_gpu(gpu.as_ref());

        // The last case is the one that bites, and it has to be this big.
        //
        // `select_chunk_bytes` ends on `target.min(dataset_total)`, so on a
        // dataset smaller than one chunk EVERY candidate rule agrees: the chunk
        // is the dataset. The wrong figure and the right one only diverge above
        // the cap — which is why the bug survived a suite full of small
        // fixtures, and why one case here allocates a real 137 MiB.
        let cases: [(usize, u64); 3] = [(16, 64), (1024, 64), (3072, 12_000)];
        for (sample_len, sample_count) in cases {
            let samples: Vec<Vec<f32>> = (0..sample_count)
                .map(|s| vec![s as f32; sample_len])
                .collect();
            let dataset =
                GpuDataset::from_samples(gpu.as_ref(), samples, sample_len).expect("dataset");

            let plan = plan_dataset(
                &device,
                DatasetSpec {
                    sample_count,
                    sample_bytes: sample_len as u64 * 4,
                },
                16,
            );
            assert_eq!(
                plan.chunk_sample_capacity as usize,
                dataset.chunk_sample_capacity(),
                "{sample_count}×{sample_len}: predicted {} samples per chunk, the dataset holds {}",
                plan.chunk_sample_capacity,
                dataset.chunk_sample_capacity()
            );
            // The buffer is the allocation the inventory charges for, so the
            // prediction has to be that buffer and not merely no larger.
            assert_eq!(
                plan.chunk_bytes,
                dataset.gpu_buffer_bytes(),
                "{sample_count}×{sample_len}: predicted a chunk of {} against a buffer of {}",
                format_bytes(plan.chunk_bytes),
                format_bytes(dataset.gpu_buffer_bytes())
            );
        }
    });
}

/// The dataset is the one post that is streamed, and the plan must say so in
/// numbers: how big a chunk is, how many there are, and how many a shuffled
/// batch touches.
#[test]
fn the_dataset_plan_counts_chunks_and_the_uploads_a_batch_costs() {
    // A device whose storage bindings cap at 128 MiB and whose adapter would
    // allocate no more than 256 MiB in one go: the chunk rule then lands on its
    // 64 MiB floor. Both figures are stated, because the rule reads BOTH — the
    // adapter's for its target, the device's for its cap — and a fixture that
    // set only one would model a machine that does not exist.
    let mut device = DeviceProfile::hypothetical("mac", None);
    device.max_storage_buffer_binding_size = 128 * 1024 * 1024;
    device.max_buffer_size = 128 * 1024 * 1024;
    device.adapter_max_buffer_size = 256 * 1024 * 1024;

    let cifar_grey = DatasetSpec {
        sample_count: 50_000,
        sample_bytes: 32 * 32 * 4,
    };
    let plan = plan_dataset(&device, cifar_grey, 16);
    assert_eq!(plan.chunk_bytes, 64 * 1024 * 1024);
    assert_eq!(plan.chunk_count, 4, "195 MiB in 64 MiB chunks is 4 chunks");

    // The oracle is written out here, independently of the implementation, and
    // per chunk — because the last chunk is short. 50 000 samples in chunks of
    // 32 768 is 32 768 + 17 232, not two equal halves, and treating it as two
    // equal halves is how the first version of this over-predicted the traffic.
    let capacity = plan.chunk_sample_capacity;
    let total_samples = 50_000f64;
    let mut held: Vec<f64> = Vec::new();
    let mut remaining = 50_000u64;
    while remaining > 0 {
        let take = remaining.min(capacity);
        held.push(take as f64);
        remaining -= take;
    }
    assert_eq!(held.len(), 4, "the fixture must exercise a short last chunk");
    assert!(
        held[3] < held[0],
        "the last chunk holds {} of {} — a fixture where it is full proves nothing",
        held[3],
        held[0]
    );
    let expected_uploads: f64 = held
        .iter()
        .map(|h| 1.0 - (1.0 - h / total_samples).powi(16))
        .sum();
    let expected_bytes: f64 = held
        .iter()
        .map(|h| (1.0 - (1.0 - h / total_samples).powi(16)) * h * 4096.0)
        .sum();
    assert!(
        (plan.expected_uploads_per_step - expected_uploads).abs() < 1e-9,
        "{} against the closed form {expected_uploads}",
        plan.expected_uploads_per_step
    );
    assert!(
        (plan.expected_upload_bytes_per_step - expected_bytes).abs() < 1.0,
        "{} bytes against the closed form {expected_bytes}",
        plan.expected_upload_bytes_per_step
    );
    // And it must never claim more traffic than there is dataset.
    assert!(
        plan.expected_upload_bytes_per_step <= plan.total_bytes as f64,
        "predicted {} of traffic from a {} dataset",
        format_bytes(plan.expected_upload_bytes_per_step as u64),
        format_bytes(plan.total_bytes)
    );

    // A dataset that fits in one chunk is one upload, whatever the batch — and
    // then zero, because the chunk stays resident.
    let tiny = DatasetSpec {
        sample_count: 100,
        sample_bytes: 4096,
    };
    let plan = plan_dataset(&device, tiny, 64);
    assert_eq!(plan.chunk_count, 1);
    assert_eq!(plan.expected_uploads_per_step, 1.0);
}

/// The chunk that is resident is charged to the inventory — and it is the
/// *chunk*, not the dataset: 195 MiB of CIFAR-10 costs 64 MiB of GPU.
#[test]
fn only_the_resident_chunk_is_charged_not_the_whole_dataset() {
    let mut device = DeviceProfile::hypothetical("mac", None);
    device.max_storage_buffer_binding_size = 128 * 1024 * 1024;
    device.max_buffer_size = 128 * 1024 * 1024;
    device.adapter_max_buffer_size = 256 * 1024 * 1024;

    let mut req = request(16, training(OptimizerKind::Adam, false));
    req.dataset = Some(DatasetSpec {
        sample_count: 50_000,
        sample_bytes: 32 * 32 * 4,
    });
    let inv = inventory(&req, &device).unwrap();

    assert_eq!(inv.bytes_of(Kind::DatasetChunk), 64 * 1024 * 1024);
    assert_eq!(
        inv.total_bytes() - inv.resident_bytes(),
        64 * 1024 * 1024,
        "the streamed chunk is exactly the difference between total and resident"
    );
    assert!(
        inv.dataset.unwrap().total_bytes > 3 * inv.bytes_of(Kind::DatasetChunk),
        "the fixture must be a dataset that does NOT fit on the GPU, or it \
         proves nothing about streaming"
    );
}

// ---------------------------------------------------------------------------
// Planning the graph
// ---------------------------------------------------------------------------

/// The planner must chain dimensions from the input, not read the `dim_input`
/// the draft was written with — those two disagree the moment anyone edits a
/// layer, and the GPU follows the chain.
#[test]
fn the_planner_chains_dimensions_instead_of_trusting_the_draft() {
    let (mut layers, input_size) = u_shaped_stack();
    // Poison the second layer's recorded input. The chain says (8,8,8).
    if let LayerDraft::GroupNorm { dim_input, .. } = &mut layers[1] {
        *dim_input = (99, 99, 99);
    }
    let planned = plan_graph(&layers, input_size).expect("planning failed");
    let dim = planned.layers[1].get_dim_input();
    assert_eq!(
        (dim.x, dim.y, dim.z),
        (8, 8, 8),
        "the planner believed the draft instead of the layer before it"
    );
}

/// A concat whose skip target does not exist is an error, not a guess.
#[test]
fn a_concat_without_its_skip_is_refused() {
    let (mut layers, input_size) = u_shaped_stack();
    if let LayerDraft::Concat { skip_key, .. } = &mut layers[6] {
        *skip_key = "nowhere".to_string();
    }
    assert!(matches!(
        plan_graph(&layers, input_size),
        Err(crate::model::ModelError::MissingSavedOutput { .. })
    ));
}

/// Bytes are formatted in the binary units the limits are expressed in.
#[test]
fn bytes_read_in_the_units_the_limits_use() {
    assert_eq!(format_bytes(512), "512 B");
    assert_eq!(format_bytes(134_217_728), "128.0 MiB");
    assert_eq!(format_bytes(8 * 1024 * 1024 * 1024), "8.00 GiB");
}
