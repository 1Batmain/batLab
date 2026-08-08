//! File purpose: Counts what actually crosses the host↔GPU boundary, so that
//! "do we upload everything once, or piece by piece?" is answered by a
//! measurement instead of by an assertion.
//!
//! # Why counters and not prints
//!
//! The traffic of this engine is invisible in its source: `queue.write_buffer`
//! appears in twenty places, none of them says how many bytes it moves per
//! training step, and the most expensive one — a whole 64 MiB dataset chunk —
//! is issued by a cache miss three call levels down. A `println!` in each would
//! answer the question once and then have to be deleted; a counter answers it
//! every time anyone asks.
//!
//! # Cost
//!
//! Four relaxed atomic adds per submission, on a path that is already issuing a
//! command buffer to a driver. Nothing reads them unless something asks for a
//! [`TransferSnapshot`], and nothing is allocated to hold them. There is no
//! "enable" flag because a flag would be a second thing to get wrong, and
//! because a `fetch_add(Relaxed)` next to a `queue.submit` is not measurable.

use std::sync::atomic::{AtomicU64, Ordering};

/// Live counters, owned by the [`crate::GpuContext`] every path already shares.
#[derive(Debug, Default)]
pub struct TransferCounters {
    host_to_device_bytes: AtomicU64,
    host_to_device_writes: AtomicU64,
    device_to_host_bytes: AtomicU64,
    device_to_host_reads: AtomicU64,
    submits: AtomicU64,
    command_buffers: AtomicU64,
}

impl TransferCounters {
    pub fn record_write(&self, bytes: u64) {
        self.host_to_device_bytes
            .fetch_add(bytes, Ordering::Relaxed);
        self.host_to_device_writes.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_read(&self, bytes: u64) {
        self.device_to_host_bytes
            .fetch_add(bytes, Ordering::Relaxed);
        self.device_to_host_reads.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_submit(&self, command_buffers: u64) {
        self.submits.fetch_add(1, Ordering::Relaxed);
        self.command_buffers
            .fetch_add(command_buffers, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> TransferSnapshot {
        TransferSnapshot {
            host_to_device_bytes: self.host_to_device_bytes.load(Ordering::Relaxed),
            host_to_device_writes: self.host_to_device_writes.load(Ordering::Relaxed),
            device_to_host_bytes: self.device_to_host_bytes.load(Ordering::Relaxed),
            device_to_host_reads: self.device_to_host_reads.load(Ordering::Relaxed),
            submits: self.submits.load(Ordering::Relaxed),
            command_buffers: self.command_buffers.load(Ordering::Relaxed),
        }
    }
}

/// A reading of the counters at one instant.
///
/// Differences of two snapshots are what anyone actually wants — "per training
/// step", "per denoising step" — so [`TransferSnapshot::since`] is the primary
/// operation and the absolute numbers are mostly a way to get one.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TransferSnapshot {
    /// Bytes written into GPU buffers with `queue.write_buffer`: dataset
    /// chunks, model inputs, weight initialisation, every uniform rewrite.
    pub host_to_device_bytes: u64,
    pub host_to_device_writes: u64,
    /// Bytes mapped back to the CPU: the loss, the metrics probes, and — the
    /// interesting one — the model's output on every reverse step.
    pub device_to_host_bytes: u64,
    pub device_to_host_reads: u64,
    /// Calls to `queue.submit`, whatever they carried.
    pub submits: u64,
    /// Command buffers those submissions carried.
    pub command_buffers: u64,
}

impl TransferSnapshot {
    /// What moved between `earlier` and this reading.
    pub fn since(self, earlier: TransferSnapshot) -> TransferSnapshot {
        TransferSnapshot {
            host_to_device_bytes: self
                .host_to_device_bytes
                .saturating_sub(earlier.host_to_device_bytes),
            host_to_device_writes: self
                .host_to_device_writes
                .saturating_sub(earlier.host_to_device_writes),
            device_to_host_bytes: self
                .device_to_host_bytes
                .saturating_sub(earlier.device_to_host_bytes),
            device_to_host_reads: self
                .device_to_host_reads
                .saturating_sub(earlier.device_to_host_reads),
            submits: self.submits.saturating_sub(earlier.submits),
            command_buffers: self.command_buffers.saturating_sub(earlier.command_buffers),
        }
    }

    /// The same totals divided by a number of steps — the per-step figures the
    /// resources page reports.
    pub fn per(self, steps: u64) -> TransferRate {
        let steps = steps.max(1) as f64;
        TransferRate {
            host_to_device_bytes: self.host_to_device_bytes as f64 / steps,
            device_to_host_bytes: self.device_to_host_bytes as f64 / steps,
            round_trips: self.device_to_host_reads as f64 / steps,
            submits: self.submits as f64 / steps,
        }
    }
}

/// Per-step traffic.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TransferRate {
    pub host_to_device_bytes: f64,
    pub device_to_host_bytes: f64,
    /// Readbacks per step. Each one is a `map_async` plus a blocking poll: the
    /// CPU waits for the GPU to drain, which is why this number matters more
    /// than the bytes next to it.
    pub round_trips: f64,
    pub submits: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A snapshot difference is what every reported figure is built on, and
    /// `since` must not go negative when counters are read out of order.
    #[test]
    fn a_difference_of_snapshots_is_what_moved_between_them() {
        let counters = TransferCounters::default();
        counters.record_write(1000);
        counters.record_submit(1);
        let first = counters.snapshot();

        counters.record_write(250);
        counters.record_read(40);
        counters.record_submit(2);
        let second = counters.snapshot();

        let delta = second.since(first);
        assert_eq!(delta.host_to_device_bytes, 250);
        assert_eq!(delta.device_to_host_bytes, 40);
        assert_eq!(delta.device_to_host_reads, 1);
        assert_eq!(delta.submits, 1);
        assert_eq!(delta.command_buffers, 2);

        // Reversed, it saturates at zero rather than wrapping around u64.
        assert_eq!(first.since(second), TransferSnapshot::default());
    }

    /// There must be no second door.
    ///
    /// The counters are only true if every host↔GPU crossing in the engine goes
    /// through [`crate::GpuContext`]. A single `queue.write_buffer` left behind
    /// does not fail, does not warn, and does not show up in any number — it
    /// just makes the reported traffic quietly too small, which is the one
    /// failure mode a traffic report must not have. So the rule is checked
    /// mechanically rather than remembered.
    ///
    /// Test files are exempt: they build fixtures, and a fixture's uploads are
    /// not the engine's traffic.
    #[test]
    fn nothing_in_the_engine_bypasses_the_counted_queue() {
        fn walk(dir: &std::path::Path, offenders: &mut Vec<String>) {
            for entry in std::fs::read_dir(dir).expect("failed to read the engine sources") {
                let path = entry.expect("bad directory entry").path();
                if path.is_dir() {
                    walk(&path, offenders);
                    continue;
                }
                let name = path.file_name().unwrap_or_default().to_string_lossy();
                if !name.ends_with(".rs") || name.contains("test") {
                    continue;
                }
                // `gpu_context.rs` is where the door is: it is the one file
                // allowed to touch the queue, because it counts what it passes.
                if name == "gpu_context.rs" {
                    continue;
                }
                let source = std::fs::read_to_string(&path).expect("failed to read a source file");
                // Comments name the wgpu call all over this crate; only code
                // counts, so lines whose first non-space character starts a
                // comment are skipped.
                for (number, line) in source.lines().enumerate() {
                    let code = line.trim_start();
                    if code.starts_with("//") || code.starts_with("///") {
                        continue;
                    }
                    // Assembled from halves so that this line is not itself an
                    // offender — the detector has to be allowed to name what it
                    // detects.
                    if code.contains(concat!("queue", ".write_buffer"))
                        || code.contains(concat!("queue", ".submit"))
                    {
                        offenders.push(format!("{}:{}: {}", path.display(), number + 1, code));
                    }
                }
            }
        }

        let mut offenders = Vec::new();
        walk(
            &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src"),
            &mut offenders,
        );
        assert!(
            offenders.is_empty(),
            "these bypass GpuContext::write_buffer/submit, so their bytes are \
             invisible to every traffic figure the project reports:\n{}",
            offenders.join("\n")
        );
    }

    #[test]
    fn a_rate_is_the_total_divided_by_the_steps() {
        let counters = TransferCounters::default();
        for _ in 0..4 {
            counters.record_write(100);
            counters.record_read(10);
            counters.record_submit(1);
        }
        let rate = counters.snapshot().per(4);
        assert_eq!(rate.host_to_device_bytes, 100.0);
        assert_eq!(rate.device_to_host_bytes, 10.0);
        assert_eq!(rate.round_trips, 1.0);
        assert_eq!(rate.submits, 1.0);
        // Zero steps must not divide by zero.
        assert_eq!(counters.snapshot().per(0).submits, 4.0);
    }
}
