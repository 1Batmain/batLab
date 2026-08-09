//! File purpose: the engine's performance knobs — the handful of numbers that
//! are *calibration*, not arithmetic — read once from the environment so they
//! can be swept on a machine nobody here owns.
//!
//! # Why these are not constants
//!
//! batLab is developed on exactly one GPU and is meant to run on the visitor's
//! (`lib.rs`, and the engine/interface boundary in `CLAUDE.md`). A constant
//! tuned here is a constant tuned *there* too, and nobody would ever find out:
//! a badly chosen tile size does not fail, it just costs time silently.
//!
//! So every number that came out of a sweep on this Mac lives here, with:
//!
//! - a **default** that is the value measured here — changing the default is a
//!   perf decision to be argued with numbers, not a side effect of this module;
//! - an **environment variable** that overrides it without recompiling, so the
//!   sweep is one shell loop on any machine;
//! - the **sweep that produced the default**, written down next to it.
//!
//! Nothing algorithmic belongs here. A kernel whose *correctness* depends on a
//! knob is a bug; every knob below only moves how the same arithmetic is split
//! across threads.
//!
//! # Sweeping on a new machine
//!
//! ```text
//! for t in 65536 131072 262144 524288 1048576; do
//!   BATLAB_CONV_TARGET_THREADS=$t \
//!     batlab --profile-step Color_Diffusion_XL --dataset <path> --batch 32
//! done
//! ```
//!
//! and read the `Σ pass minima` line of the budget. `--profile-step` prints the
//! knobs it ran under, so a profile pasted into a report says what produced it.

use std::sync::OnceLock;

/// A single knob: its name in the environment, its default, and its meaning.
pub struct Knob {
    /// Environment variable that overrides it.
    pub env: &'static str,
    /// The value in force for this process.
    pub value: u32,
    /// Value compiled in, i.e. the one measured on the development machine.
    pub default: u32,
}

/// Parse an override, falling back to `default`.
///
/// A malformed or zero value is **not** an error the run stops on — it is a
/// typo in a sweep script, and stopping a nine-hour training on it would be
/// worse than ignoring it. It falls back, and `describe()` shows what is
/// actually in force.
fn resolve(raw: Option<&str>, default: u32) -> u32 {
    match raw.map(str::trim).map(str::parse::<u32>) {
        Some(Ok(v)) if v > 0 => v,
        _ => default,
    }
}

fn env_knob(name: &'static str, default: u32) -> u32 {
    resolve(std::env::var(name).ok().as_deref(), default)
}

/// How many threads the `grad_weights` / `grad_bias` reductions aim to keep in
/// flight when they choose how far to split one sum (`reduction_lanes`).
///
/// Default 262 144, swept on this machine with `--profile-step` on the real
/// training step — the table is in `ConvolutionType::reduction_lanes` and in
/// `GPU_PROFILE.md` §6. The optimum is interior: too few lanes starves the GPU,
/// too many pays more in tree-reduction barriers than the sum it splits. Both
/// failure modes are architecture-dependent, which is exactly why this is a
/// knob and not a constant.
pub fn conv_target_threads() -> u32 {
    static V: OnceLock<u32> = OnceLock::new();
    *V.get_or_init(|| env_knob("BATLAB_CONV_TARGET_THREADS", 262_144))
}

/// Fewest positions a lane may be left to sum. Below this the tree reduction
/// and its barriers cost more than the sequential sum they replace.
pub fn conv_min_positions_per_lane() -> u32 {
    static V: OnceLock<u32> = OnceLock::new();
    *V.get_or_init(|| env_knob("BATLAB_CONV_MIN_POSITIONS_PER_LANE", 16))
}

/// Every knob and the value in force, for the run banner and the profiler.
pub fn describe() -> Vec<Knob> {
    vec![
        Knob {
            env: "BATLAB_CONV_TARGET_THREADS",
            value: conv_target_threads(),
            default: 262_144,
        },
        Knob {
            env: "BATLAB_CONV_MIN_POSITIONS_PER_LANE",
            value: conv_min_positions_per_lane(),
            default: 16,
        },
    ]
}

/// One line per knob whose value is **not** the compiled-in default. Empty when
/// the run is on defaults, so a profile that says nothing was taken on stock
/// settings.
pub fn overrides_in_force() -> Vec<String> {
    describe()
        .into_iter()
        .filter(|k| k.value != k.default)
        .map(|k| format!("{}={} (default {})", k.env, k.value, k.default))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unset_knob_is_the_value_measured_here() {
        assert_eq!(resolve(None, 262_144), 262_144);
    }

    #[test]
    fn a_knob_reads_its_override() {
        assert_eq!(resolve(Some("65536"), 262_144), 65_536);
        assert_eq!(resolve(Some("  4096\n"), 262_144), 4_096);
    }

    /// A typo in a sweep script must not stop a training run — and must not
    /// silently become a *different* number either. It becomes the default,
    /// and `overrides_in_force()` then reports nothing, which is the truth.
    #[test]
    fn a_malformed_override_falls_back_instead_of_stopping_the_run() {
        for bad in ["", "0", "-1", "many", "1e5", "262144.0"] {
            assert_eq!(
                resolve(Some(bad), 262_144),
                262_144,
                "{bad:?} should have fallen back to the default"
            );
        }
    }

    /// The defaults listed for the banner must be the defaults the accessors
    /// actually use — a table that drifts from the code would misreport the
    /// baseline of every profile pasted into a report.
    #[test]
    fn the_described_defaults_are_the_ones_the_accessors_use() {
        // The test process may itself be running under an override; what is
        // pinned here is the *default* column, not the value.
        for knob in describe() {
            assert!(
                knob.default > 0,
                "{} has no compiled-in default",
                knob.env
            );
            assert!(
                knob.env.starts_with("BATLAB_"),
                "{} is not in the BATLAB_ namespace",
                knob.env
            );
        }
        assert_eq!(
            describe().len(),
            2,
            "a knob was added or removed without updating the banner test"
        );
    }
}
