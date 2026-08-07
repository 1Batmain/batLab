//! File purpose: What each parameter of the TUI actually does, in one panel.
//!
//! Every form in this application asks for numbers that decide hours of GPU
//! time, and every one of them had exactly one affordance: a label. "Magnitude"
//! is not an explanation, and the answer — *1,0 is the nominal DDPM chain, and
//! below it the images get smoother by converging on the dataset mean* — was
//! sitting in a report nobody reads while typing into a form.
//!
//! So the text here is **sourced**, not invented: each entry cites the report it
//! comes from, and the numbers in it are measured numbers from that report, not
//! rules of thumb. When a report says something uncomfortable — that averaging
//! denoising paths blurs rather than enriches, that the seed toggle hides a bug
//! class — the panel says it too. A help panel that only repeats the label is
//! decoration; one that tells you the measured trap is an instrument.
//!
//! Adding a field to a form without adding its entry here is not an error: the
//! panel simply falls silent for that field. It is a hint, never a gate.

use super::app::Screen;

/// One field's worth of explanation.
pub struct HelpEntry {
    /// Repeated at the top of the panel, so a narrow terminal that clipped the
    /// form still says which field is being explained.
    pub title: &'static str,
    /// Paragraphs. Wrapped by the renderer; blank strings are blank lines.
    pub body: &'static [&'static str],
    /// The report this was taken from, without the `docs/reports/` prefix.
    pub source: Option<&'static str>,
}

/// The explanation for the field at `field_idx` of `screen`, if there is one.
///
/// The indices are the ones the screen's own `*_FIELD_NAMES` use, so the two
/// stay aligned by construction — a field inserted in the middle of a form
/// shifts both.
pub fn help_for(screen: Screen, field_idx: usize) -> Option<&'static HelpEntry> {
    let table: &[HelpEntry] = match screen {
        Screen::InputSize => INPUT_SIZE_HELP,
        Screen::TrainingParams => TRAINING_HELP,
        Screen::InferenceParams => INFERENCE_HELP,
        Screen::PerpetualParams => PERPETUAL_HELP,
        Screen::TrainingControl => TRAINING_CONTROL_HELP,
        Screen::WeightSelector => return Some(&WEIGHTS_HELP),
        _ => return None,
    };
    table.get(field_idx)
}

// ---------------------------------------------------------------------------
// Model input geometry
// ---------------------------------------------------------------------------

static INPUT_SIZE_HELP: &[HelpEntry] = &[
    HelpEntry {
        title: "Width",
        body: &[
            "Largeur de l'image que le modèle voit et produit. Elle doit \
             correspondre au dataset : CIFAR-10 est en 32×32.",
            "",
            "Confirmer cet écran efface la liste des couches : la géométrie est \
             le premier maillon de la chaîne, tout ce qui suit était dimensionné \
             sur l'ancienne.",
        ],
        source: None,
    },
    HelpEntry {
        title: "Height",
        body: &[
            "Hauteur de l'image. Même remarque que la largeur : elle vient du \
             dataset, pas d'un choix libre.",
        ],
        source: None,
    },
    HelpEntry {
        title: "Channels",
        body: &[
            "Canaux d'entrée — et il en faut STRICTEMENT PLUS qu'en sortie. Le \
             surplus porte l'embedding du timestep t : 3→1 en niveaux de gris, \
             7→3 en couleur.",
            "",
            "Sans ce surplus le réseau ne reçoit jamais t. ε̂ ne peut plus être \
             qu'une moyenne sur tous les t (std ≈ 0,6 pour une cible de 1,0), la \
             chaîne inverse amplifie le résidu ×174 sur 256 pas, et \
             l'échantillonnage sature en blanc.",
        ],
        source: Some("INSIGHTS_TRAINING.md"),
    },
];

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

static WEIGHTS_HELP: HelpEntry = HelpEntry {
    title: "Poids de départ",
    body: &[
        "Le défaut est de CONTINUER depuis les poids du modèle — latest.ckpt, le \
         fichier que tout entraînement réécrit.",
        "",
        "« Start from random weights » repart de zéro : le checkpoint n'est plus \
         alors que la destination d'écriture du run.",
        "",
        "Un checkpoint d'une autre architecture est refusé par le moteur, mais \
         seulement après avoir construit le modèle sur GPU — d'où la règle : \
         éditer les couches remet ce choix sur « random ».",
    ],
    source: None,
};

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------

static TRAINING_HELP: &[HelpEntry] = &[
    HelpEntry {
        title: "Learning Rate",
        body: &[
            "Le pas d'apprentissage. Avec Adam, 1e-3 est le défaut éprouvé : il \
             atteint dès le pas 75 un niveau que SGD n'atteint jamais en 1500 \
             pas, sur les quatre tranches de t.",
            "",
            "Le plateau utile va de 1e-3 à 3e-3. 3e-3 converge deux fois plus \
             vite mais dégrade un peu la moyenne ; 3e-4 est nettement moins bon \
             (0,00649 contre 0,00266 en haut-t). 1e-3 est le choix de prudence : \
             la tolérance au lr baisse quand le réseau s'approfondit.",
        ],
        source: Some("OPTIMIZER_ADAM.md"),
    },
    HelpEntry {
        title: "Batch Size",
        body: &[
            "Nombre d'échantillons traités par pas. 16 est la valeur de tous les \
             runs de référence : 1,84× plus rapide qu'à batch 1, et un pas entier \
             tient en UNE soumission GPU au lieu de 18 (temps CPU ÷13).",
            "",
            "Au-delà, l'accélération retombe (1,26× à batch 64) même si le débit \
             par échantillon continue de progresser — 62 ms contre 105 ms à batch \
             16. La mémoire GPU, elle, croît linéairement avec le batch.",
        ],
        source: Some("BATCH_DISPATCH.md"),
    },
    HelpEntry {
        title: "Steps",
        body: &[
            "Nombre de pas d'optimisation. Repères mesurés : 600 pas suffisent à \
             un smoke lisible avec Adam (std(ε̂) déjà à 0,99 pour une cible de \
             1,0) ; la planche couleur finale a demandé 20 000 pas, soit ≈ 7 h.",
            "",
            "En SGD la tranche t bas plafonnait dès ~4000 pas et remontait même \
             ensuite : plus de pas n'achète pas toujours quelque chose. Une loss \
             de batch qui descend ne suffit pas — c'est la loss par tranche de t \
             qui dit si le modèle utilise t.",
        ],
        source: Some("COLOR_MODEL.md, SCALE_UNET.md"),
    },
    HelpEntry {
        title: "Start from random",
        body: &[
            "Décoché (le défaut), le run REPREND les poids choisis à l'étape \
             précédente et continue de les entraîner.",
            "",
            "Coché, il repart de zéro. Utile pour comparer une architecture à \
             elle-même, ou quand les poids d'un run précédent sont suspects.",
            "",
            "Éditer l'architecture le coche d'office : des poids d'une autre \
             géométrie ne seraient de toute façon pas chargeables.",
        ],
        source: None,
    },
];

// ---------------------------------------------------------------------------
// Inference
// ---------------------------------------------------------------------------

static INFERENCE_HELP: &[HelpEntry] = &[
    HelpEntry {
        title: "Random Seed",
        body: &[
            "Random tire une graine neuve à chaque run. Manual rejoue exactement \
             la même image : à graine et paramètres égaux, la sortie est la \
             même.",
        ],
        source: None,
    },
    HelpEntry {
        title: "Seed",
        body: &[
            "La graine fixe le latent de départ x_T — le bruit d'où part la \
             chaîne inverse.",
            "",
            "C'est le paramètre qui a coûté le plus cher du projet : le sampler \
             XORait le pas de diffusion dans la graine pendant que le champ de \
             bruit XORait l'index du pixel. Les deux se composaient, les 256 pas \
             ne tiraient qu'un seul champ permuté, et toutes les images sortaient \
             en bandes horizontales. Corrigé sans réentraîner : banding 15,07 → \
             1,23, diversité ×15,6.",
        ],
        source: Some("ANISOTROPY_HUNT.md"),
    },
    HelpEntry {
        title: "Denoising Paths",
        body: &[
            "Nombre de trajectoires de débruitage, moyennées à l'arrivée.",
            "",
            "À lire avant d'augmenter : les N chemins partent tous du MÊME x_T, \
             donc la moyenne floute au lieu d'enrichir. Mesuré à graines égales, \
             paths=3 divise la diversité inter-graines par 1,8 — et masquait une \
             partie du banding plutôt que de le corriger (2,41 contre 4,66). \
             Le défaut, 1, est le réglage honnête.",
        ],
        source: Some("AUDIT_TRAINING.md, SCALE_UNET.md"),
    },
    HelpEntry {
        title: "Magnitude",
        body: &[
            "Échelle du bruit σ_t·z réinjecté à chaque pas de la chaîne inverse. \
             1,0 = chaîne DDPM nominale, c'est-à-dire la distribution correcte.",
            "",
            "Plus bas donne des images plus lisses, mais qui convergent vers la \
             moyenne du dataset : à 0 la chaîne est déterministe et huit graines \
             rendent quasiment la même image plate (intra_image_std 0,029).",
            "",
            "Le repère utile n'est pas inter_seed_std — il est proportionnel à \
             magnitude, il mesure donc surtout le bruit non débruité. C'est \
             intra_image_std qui compte : il doit s'approcher de 0,206 (le \
             dataset) PAR LE BAS, pas le dépasser. À magnitude 1,0 il monte à \
             0,283 : les images ne sont pas plus variées, elles sont plus \
             bruitées.",
        ],
        source: Some("LOSS_WEIGHTING.md, ANISOTROPY_HUNT.md"),
    },
];

// ---------------------------------------------------------------------------
// Perpetual
// ---------------------------------------------------------------------------

static PERPETUAL_HELP: &[HelpEntry] = &[
    HelpEntry {
        title: "Random Seed",
        body: &[
            "Random tire une graine neuve. Manual repart toujours de la même \
             image de départ — utile pour rejouer une dérive qu'on a aimée.",
            "",
            "En cours de run, [r] re-tire une graine, donc une AUTRE image du \
             dataset, sans tout relancer.",
        ],
        source: Some("IMG2IMG_DRIFT.md"),
    },
    HelpEntry {
        title: "Seed",
        body: &[
            "La dérive part d'une VRAIE image du dataset (cifar10_grey ou \
             cifar10_rgb selon les canaux de sortie du modèle) : la graine dit \
             LAQUELLE.",
            "",
            "Elle ne fixe donc plus un latent de bruit pur — le run n'a plus à \
             descendre 256 pas avant que quoi que ce soit arrive, il remonte \
             depuis l'image dès la première frame, au niveau réglé ci-dessous.",
            "",
            "Sans dataset trouvable, repli sur l'ancienne ouverture : bruit pur \
             en haut du schedule, annoncé à l'écran.",
        ],
        source: Some("IMG2IMG_DRIFT.md, ANISOTROPY_HUNT.md"),
    },
    HelpEntry {
        title: "Magnitude",
        body: &[
            "Échelle du bruit réinjecté à chaque pas. 1,0 = chaîne nominale.",
            "",
            "En dérive perpétuelle, la baisser lisse l'image mais la tire vers la \
             moyenne du dataset : la dérive devient calme et vide. Les campagnes \
             perpetual tournent à 1,0.",
        ],
        source: Some("LOSS_WEIGHTING.md, PERPETUAL_FLUX.md"),
    },
    HelpEntry {
        title: "Renoise Depth (t_r)",
        body: &[
            "Le niveau de bruit auquel chaque cycle remonte — et, en régime flux, \
             le seul niveau où le run vit (il s'appelle alors t*).",
            "",
            "16 à 32 est le réglage recommandé : assez de mouvement pour que \
             l'image travaille, assez peu pour que le panneau de gauche reste \
             lisible. Plus haut, ça bouge plus fort : |Δx_t| médiane 10,7 à \
             t*=16, 20,4 à 64, 28,5 à 128.",
            "",
            "C'est le cadran qui « jauge le bruit qu'on réinjecte », et il agit \
             dès la première frame : le run part d'une image réelle, il n'a pas \
             de descente initiale à faire.",
        ],
        source: Some("PERPETUAL_FLUX.md, IMG2IMG_DRIFT.md"),
    },
    HelpEntry {
        title: "Steps / second",
        body: &[
            "La cadence imposée au sampler, et elle est nécessaire : sans \
             throttle le modèle tient ~500 pas/s, un cycle t_r=64 boucle huit \
             fois par seconde — c'est un clignotement, pas une dérive. 30 est le \
             réglage retenu.",
            "",
            "C'est une consigne, pas une promesse : au plafond (228 demandés) le \
             sampler rend 200 à 211. Le panneau du moniteur affiche les deux \
             nombres, mesure et consigne.",
        ],
        source: Some("PERPETUAL_INFERENCE.md"),
    },
    HelpEntry {
        title: "Regime",
        body: &[
            "errance — descend jusqu'à t=0, l'image se résout complètement, puis \
             rebondit à t_r.",
            "",
            "respiration — plancher à t_r/2 : l'image ne se résout jamais.",
            "",
            "flux — tient un seul niveau t* et n'en repart jamais : ni phase, ni \
             point de rebroussement.",
            "",
            "errance et respiration ne se figent plus pendant la remontée — le \
             modèle tourne aussi en montant (0 frame gelée sur 179, contre \
             179/179 avant). Il reste leur tournant de cycle, une frame.",
        ],
        source: Some("PERPETUAL_FLUX.md, IMG2IMG_DRIFT.md"),
    },
];

// ---------------------------------------------------------------------------
// Training controls, mid-run
// ---------------------------------------------------------------------------

static TRAINING_CONTROL_HELP: &[HelpEntry] = &[
    HelpEntry {
        title: "Learning Rate",
        body: &[
            "Le pas d'apprentissage, changeable sans arrêter le run. Avec Adam, \
             1e-3 est le défaut éprouvé et le plateau utile va jusqu'à 3e-3.",
            "",
            "Le baisser en fin de run est la façon habituelle d'affiner sans \
             repartir d'un checkpoint.",
        ],
        source: Some("OPTIMIZER_ADAM.md"),
    },
    HelpEntry {
        title: "Batch Size",
        body: &[
            "Changer le batch en cours de run RECONSTRUIT le modèle. La \
             reconstruction préserve poids, biais, moments Adam et compteur de \
             pas de l'optimiseur — rien n'est perdu, mais ce n'est pas gratuit.",
            "",
            "La mémoire GPU croît linéairement avec le batch : c'est par là qu'un \
             run se fait tuer.",
        ],
        source: Some("BATCH_DISPATCH_DESIGN.md"),
    },
    HelpEntry {
        title: "Total Steps",
        body: &[
            "Où le run s'arrête. L'allonger en cours de route évite de relancer \
             depuis un checkpoint.",
            "",
            "À ne pas confondre avec les pas du schedule de diffusion (256), qui \
             sont une propriété du modèle et ne se règlent pas ici.",
        ],
        source: None,
    },
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tui::app::{
        INFERENCE_PARAM_FIELD_NAMES, INPUT_SIZE_FIELD_NAMES, PERPETUAL_PARAM_FIELD_NAMES,
        TRAINING_CONTROL_FIELD_NAMES, TRAINING_PARAM_FIELD_NAMES,
    };

    /// The tables are indexed by field position, so a form that grows a field
    /// in the middle silently re-points every entry after it at the wrong
    /// label. Comparing the titles against the field names is what catches
    /// that — the panel would otherwise confidently explain the wrong
    /// parameter, which is worse than explaining nothing.
    #[test]
    fn every_help_entry_sits_on_the_field_it_describes() {
        let forms: [(Screen, &[&str]); 5] = [
            (Screen::InputSize, &INPUT_SIZE_FIELD_NAMES),
            (Screen::TrainingParams, &TRAINING_PARAM_FIELD_NAMES),
            (Screen::InferenceParams, &INFERENCE_PARAM_FIELD_NAMES),
            (Screen::PerpetualParams, &PERPETUAL_PARAM_FIELD_NAMES),
            (Screen::TrainingControl, &TRAINING_CONTROL_FIELD_NAMES),
        ];
        for (screen, names) in forms {
            for (index, name) in names.iter().enumerate() {
                let entry = help_for(screen, index).unwrap_or_else(|| {
                    panic!("{screen:?} field {index} ({name}) has no help entry")
                });
                assert_eq!(
                    entry.title, *name,
                    "{screen:?} field {index} is '{name}' but the panel explains \
                     '{}' — the table has drifted out of step with the form",
                    entry.title
                );
            }
            assert!(
                help_for(screen, names.len()).is_none(),
                "{screen:?} has a help entry past its last field"
            );
        }
    }

    /// A field with nothing to say must fall silent, not panic and not show the
    /// previous field's text.
    #[test]
    fn a_screen_off_the_forms_has_no_help() {
        for screen in [
            Screen::ModelList,
            Screen::ModelActions,
            Screen::LayerBuilder,
            Screen::Monitor,
            Screen::RenameModel,
        ] {
            assert!(help_for(screen, 0).is_none(), "{screen:?}");
        }
    }

    /// Every claim with a number in it has to say where the number came from.
    /// The panel's whole value is that it is not making things up.
    #[test]
    fn a_cited_source_is_a_report_that_exists() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../docs/reports")
            .canonicalize()
            .expect("docs/reports must be reachable from the crate");
        let mut checked = 0;
        for screen in Screen::ALL {
            for index in 0..8 {
                let Some(entry) = help_for(screen, index) else {
                    continue;
                };
                let Some(source) = entry.source else { continue };
                for report in source.split(", ") {
                    assert!(
                        root.join(report).is_file(),
                        "{screen:?}/{index} cites {report}, which is not in docs/reports/"
                    );
                    checked += 1;
                }
            }
        }
        assert!(checked >= 10, "only {checked} citations were checked");
    }
}
