use std::sync::Arc;

use bat_tui::{ActivationMethod, LayerDraft, PaddingMode, RunMode};
use parallelizer::{
    ActivationMethod as PActivation, ActivationType, ConvolutionType, Dim3, GpuContext,
    LayerTypes, LossMethod as PLoss, Model, PaddingMode as PPadding,
};

fn main() {
    let config = match bat_tui::run() {
        Ok(c) => c,
        Err(_) => return,
    };

    match config.run.mode.clone() {
        RunMode::Train(train_cfg) => {
            let (tx, rx) = std::sync::mpsc::channel::<bat_tui::TrainingEvent>();
            let config_clone = config.clone();

            std::thread::spawn(move || {
                let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
                rt.block_on(async {
                    let gpu = Arc::new(GpuContext::new_headless().await);
                    let mut model =
                        Model::new_training(gpu, train_cfg.lr, train_cfg.batch_size, PLoss::MeanSquared)
                            .await;

                    for draft in &config_clone.layers {
                        let layer = convert_layer(draft);
                        model.add_layer(layer).expect("failed to add layer");
                    }
                    model.build();

                    let input = vec![0.5_f32; config_clone.input_elem_count()];
                    let target = vec![1.0_f32; config_clone.output_elem_count()];

                    for step in 0..train_cfg.steps {
                        model.train_step(&input, &target);
                        let _ = tx.send(bat_tui::TrainingEvent::Step { step, loss: 0.0 });
                    }
                    let _ = tx.send(bat_tui::TrainingEvent::Done);
                });
            });

            bat_tui::run_monitor(config, rx).ok();
        }
        RunMode::Infer => {
            println!("Inference mode — not yet implemented.");
        }
    }
}

fn convert_layer(draft: &LayerDraft) -> LayerTypes {
    match draft {
        LayerDraft::Convolution {
            dim_input,
            nb_kernel,
            dim_kernel,
            stride,
            padding,
        } => LayerTypes::Convolution(ConvolutionType::new(
            Dim3::new(*dim_input),
            *nb_kernel,
            Dim3::new(*dim_kernel),
            *stride,
            convert_padding(padding),
        )),
        LayerDraft::Activation { method, .. } => {
            LayerTypes::Activation(ActivationType::new(convert_activation(method), Dim3::default()))
        }
    }
}

fn convert_padding(p: &PaddingMode) -> PPadding {
    match p {
        PaddingMode::Valid => PPadding::Valid,
        PaddingMode::Same => PPadding::Same,
    }
}

fn convert_activation(a: &ActivationMethod) -> PActivation {
    match a {
        ActivationMethod::Relu => PActivation::Relu,
        ActivationMethod::Linear => PActivation::Linear,
    }
}
