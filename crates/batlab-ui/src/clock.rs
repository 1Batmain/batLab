//! File purpose: Wall-clock time as text — the stamp a checkpoint's file name
//! carries, and the date/size a listing shows beside it.
//!
//! # Why local time, and why by hand
//!
//! `std` gives a `SystemTime` and nothing else: no calendar, no timezone. The
//! two things this module owes the rest of the program are a *sortable* stamp
//! for file names and a *readable* date for the screen, and both have to be in
//! the user's own timezone — "hier soir" is a local-time notion, and a
//! checkpoint written at 22:41 in Paris must not be filed under `2041` UTC on
//! one line and shown as 22:41 on another. So the civil breakdown goes through
//! `localtime_r`, the only portable-enough way to ask the C library which
//! offset was in force *at that instant* (a fixed offset would be wrong twice a
//! year).
//!
//! # The one property the stamp must have
//!
//! `YYYY-MM-DD_HHMM` sorts the same way lexicographically and chronologically.
//! That is not cosmetic: `pretrained_weights/` is listed by name in several
//! places, and it is what makes "the newest is the last one" true without
//! anyone having to stat the files.

use std::time::{SystemTime, UNIX_EPOCH};

/// A wall-clock instant broken into the fields a human reads, in local time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CivilTime {
    pub year: i32,
    /// 1–12.
    pub month: u32,
    /// 1–31.
    pub day: u32,
    pub hour: u32,
    pub minute: u32,
    pub second: u32,
}

/// The local civil time at `at`, or `None` for an instant the C library cannot
/// place (before the epoch, or out of `time_t` range).
pub fn local(at: SystemTime) -> Option<CivilTime> {
    let secs = at.duration_since(UNIX_EPOCH).ok()?.as_secs();
    let secs = i64::try_from(secs).ok()?;
    // SAFETY: `localtime_r` fills the caller's `tm` and returns null on
    // failure; nothing is retained and no global state is read back.
    unsafe {
        let raw = secs as libc::time_t;
        let mut tm: libc::tm = std::mem::zeroed();
        if libc::localtime_r(&raw, &mut tm).is_null() {
            return None;
        }
        Some(CivilTime {
            year: tm.tm_year + 1900,
            month: (tm.tm_mon + 1) as u32,
            day: tm.tm_mday as u32,
            hour: tm.tm_hour as u32,
            minute: tm.tm_min as u32,
            second: tm.tm_sec as u32,
        })
    }
}

impl CivilTime {
    /// The form a checkpoint's file name carries: `2026-08-08_1041`.
    ///
    /// Zero-padded throughout, biggest unit first — which is the whole point,
    /// see the module header.
    pub fn stamp(&self) -> String {
        format!(
            "{:04}-{:02}-{:02}_{:02}{:02}",
            self.year, self.month, self.day, self.hour, self.minute
        )
    }

    /// The form a listing shows: `2026-08-08 10:41`.
    ///
    /// Deliberately the same digits as the stamp, in the same order: a user who
    /// reads `run-2026-08-08_1041.ckpt` on one line and `2026-08-08 10:41` on
    /// the next does not have to be told they are the same fact. A month name
    /// would read shorter and would have to pick a language the rest of the
    /// screen has not picked.
    pub fn short(&self) -> String {
        format!(
            "{:04}-{:02}-{:02} {:02}:{:02}",
            self.year, self.month, self.day, self.hour, self.minute
        )
    }
}

/// The stamp for `at`, falling back to the raw epoch seconds when the calendar
/// is unavailable.
///
/// The fallback keeps the two properties that matter — unique per run and
/// sorted chronologically (epoch seconds are both) — so a run never fails to
/// name its checkpoint because a timezone database is missing.
pub fn stamp(at: SystemTime) -> String {
    match local(at) {
        Some(civil) => civil.stamp(),
        None => format!(
            "epoch{}",
            at.duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0)
        ),
    }
}

/// The short local date of `at` for display, or `None` when it cannot be placed
/// — callers show "unknown date" rather than an invented one.
pub fn short(at: SystemTime) -> Option<String> {
    local(at).map(|civil| civil.short())
}

/// A byte count as a human reads it, in the same decimal units the training
/// loop already prints checkpoint sizes in (`bytes / 1e6` = MB).
pub fn human_bytes(bytes: u64) -> String {
    const KB: f64 = 1.0e3;
    const MB: f64 = 1.0e6;
    let n = bytes as f64;
    if n < KB {
        format!("{bytes} B")
    } else if n < MB {
        format!("{:.0} kB", n / KB)
    } else {
        format!("{:.1} MB", n / MB)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn civil(year: i32, month: u32, day: u32, hour: u32, minute: u32) -> CivilTime {
        CivilTime {
            year,
            month,
            day,
            hour,
            minute,
            second: 0,
        }
    }

    /// The property the whole naming scheme rests on: sorting the stamps as
    /// text is sorting the instants as time. Every pair here crosses a boundary
    /// where a lazier format (no padding, day first) would invert.
    #[test]
    fn a_stamp_sorts_as_text_exactly_as_it_sorts_in_time() {
        let ordered = [
            civil(2025, 12, 31, 23, 59),
            civil(2026, 1, 1, 0, 0),
            civil(2026, 1, 1, 9, 5),
            civil(2026, 1, 1, 10, 41),
            civil(2026, 1, 9, 8, 0),
            civil(2026, 1, 10, 8, 0),
            civil(2026, 8, 8, 10, 41),
            civil(2026, 10, 1, 0, 0),
        ];
        let stamps: Vec<String> = ordered.iter().map(CivilTime::stamp).collect();
        let mut sorted = stamps.clone();
        sorted.sort();
        assert_eq!(
            stamps, sorted,
            "chronological order and lexicographic order must agree"
        );
    }

    #[test]
    fn the_stamp_and_the_shown_date_carry_the_same_digits() {
        let at = civil(2026, 8, 8, 10, 41);
        assert_eq!(at.stamp(), "2026-08-08_1041");
        assert_eq!(at.short(), "2026-08-08 10:41");
    }

    /// A `SystemTime` has to come back out as a plausible calendar date — this
    /// is the only part `localtime_r` owns, and it is worth one round trip.
    #[test]
    fn a_recent_instant_lands_on_a_plausible_calendar_date() {
        let now = local(SystemTime::now()).expect("now must be placeable");
        assert!(now.year >= 2024 && now.year < 2200, "year: {}", now.year);
        assert!((1..=12).contains(&now.month), "month: {}", now.month);
        assert!((1..=31).contains(&now.day), "day: {}", now.day);
        assert!(now.hour < 24 && now.minute < 60);

        // An hour later is still the same instant plus an hour, whichever way
        // the calendar rolls: the stamps must differ and stay ordered.
        let later = SystemTime::now() + Duration::from_secs(3600);
        assert!(stamp(later) > stamp(SystemTime::now()));
    }

    #[test]
    fn sizes_read_in_the_units_the_run_prints() {
        assert_eq!(human_bytes(0), "0 B");
        assert_eq!(human_bytes(999), "999 B");
        assert_eq!(human_bytes(12_800), "13 kB");
        assert_eq!(human_bytes(14_200_000), "14.2 MB");
    }
}
