//! Renderer port #7: the submitting lane's last waits
//! (`VICTORY_PORT_JOURNAL=1`, off by default; the switch is
//! [`crate::watch::port_journal`]).
//!
//! What the lane ("victory-submit", `crate::queue`'s line) waited for most
//! with ports 1, 2, 5 and 6 on (runs guard-4, guard-6, merge-4): the heaps'
//! records (4.5-5.2 ms a frame, `descriptor::SyncedHeaps` dropped before the
//! lists are sent), then the journal's lock (`crate::watch`, ~2.8 ms; all
//! threads 8-9 ms). Two parts, both under the one switch:
//!
//! - **The heaps' records** (this module; renderer patch #4's part 1,
//!   `port/lane` 97033e5a, brought over): a round of the heap-sync thread is
//!   stamped when the title submits (`descriptor::heap_sync_stamp`), so it
//!   can share a round that began while the submission waited in the line, and
//!   the lane waits for that round, or for a later one if (a) a wait of its
//!   queue that was met in memory went since (the queue's floor,
//!   `queue/behind.rs` `State::floors`) or (b) a step or GPU landing wrote, or
//!   may have written, a heap's title memory ([`prep_wrote`], which raises
//!   `descriptor::SyncRounds`'s floor for both paths).
//! - **The journal's lock** (`crate::watch`): the pages' notes are kept in
//!   parts of their own, each with its own lock; noting an answer that
//!   changes nothing of the standing asking takes the journal's lock
//!   shared, and the parts it notes in alone. See `watch::port_journal`.
//!
//! Why a submission's records are what they were: a round reads the title's
//! records as they are when it runs. The records a submission's lists use
//! are written by the title before its submit call, or, for a part behind a
//! wait for a memory value or a fence, before that wait is met. A round
//! asked for at the submit call (or, behind a wait, the first round to begin
//! after the lane saw the wait met) begins after those writes: it sees all
//! of them, as the round begun at the preparing did. What our own steps
//! write into a heap's title memory while preparing moves the floor up for
//! every submission prepared after it. Not covered (as not before, where the
//! round raced the steps): shader writes landing in a heap's memory.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Whether the switch is on (`crate::watch::port_journal`).
pub(crate) fn on() -> bool {
    crate::watch::port_journal()
}

// ---------------------------------------------------------------------------
// The heaps' records: rounds asked for at the title's submit.

thread_local! {
    /// On the lane: the round of the heaps' records the submission being
    /// prepared needs (0: none known, a round is asked for as before).
    static MARK: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// The lane is about to do a submission that needs round `mark`.
pub(crate) fn set_mark(mark: u64) {
    MARK.set(mark);
}

/// The round the submission being prepared needs; 0 if none is known.
fn mark() -> u64 {
    MARK.get()
}

/// The round every submission prepared from now on needs at least: one that
/// began after a step of a submission being prepared wrote, or may have
/// written, title memory of a heap of descriptors ([`prep_wrote`]).
static PREP_FLOOR: AtomicU64 = AtomicU64::new(0);

/// A step of the submission being prepared wrote, or may have written, title
/// memory of a heap of descriptors: every submission prepared after it waits
/// for a round begun after now. (Whether the switch is on or not: it may
/// change in an A/B run.)
pub(crate) fn prep_wrote() {
    PREP_FLOOR.fetch_max(crate::descriptor::round_mark(), Relaxed);
    STATS[PREP_WRITES].fetch_add(1, Relaxed);
}

/// The round the submission about to be prepared needs ([`mark`], at least
/// [`PREP_FLOOR`]); 0 if none is known (or the switch is off).
pub(crate) fn needed() -> u64 {
    if !on() {
        return 0;
    }
    match mark() {
        0 => {
            STATS[NO_MARK].fetch_add(1, Relaxed);
            0
        }
        m => m.max(PREP_FLOOR.load(Relaxed)),
    }
}

// ---------------------------------------------------------------------------
// Counting.

/// Since the last frame line: rounds of the heaps' records asked for at a
/// submit; submissions that waited for a round asked earlier; preparings
/// that wrote a heap's memory (or may have); nanoseconds the lane waited for
/// the heaps' records, and how many times, with the switch and without.
/// Also preparings with no mark (a round asked then, as before) and those
/// whose round had to be asked for at the preparing (a wait's floor or our
/// own writes asked for a later one).
pub(crate) static STATS: [AtomicU64; 9] = [const { AtomicU64::new(0) }; 9];
pub(crate) const ROUNDS_ASKED: usize = 0;
pub(crate) const EARLY_ROUNDS: usize = 1;
const PREP_WRITES: usize = 2;
const HEAPS_NS_ON: usize = 3;
const HEAPS_NS_OFF: usize = 4;
const HEAPS_ON: usize = 5;
const HEAPS_OFF: usize = 6;
const NO_MARK: usize = 7;
pub(crate) const LATE_ROUNDS: usize = 8;

/// The lane waited `ns` for the heaps' records; `on`: with the switch.
pub(crate) fn heaps_waited(ns: u64, on: bool) {
    let (time, count) = if on { (HEAPS_NS_ON, HEAPS_ON) } else { (HEAPS_NS_OFF, HEAPS_OFF) };
    STATS[time].fetch_add(ns, Relaxed);
    STATS[count].fetch_add(1, Relaxed);
}

/// Whether the switch is on, or changes in this run (`VICTORY_AB`).
fn named() -> bool {
    #[cfg(feature = "diagnostics")]
    if crate::ab::bit("VICTORY_PORT_JOURNAL").is_some() {
        return true;
    }
    on()
}

/// Renderer port #7's line of the frame lines, per frame.
pub(crate) fn line(frames: f64) {
    let s: Vec<f64> = STATS.iter().map(|s| s.swap(0, Relaxed) as f64).collect();
    let journal = crate::watch::port_journal_counts();
    if !named() {
        return;
    }
    let f = frames.max(1.0);
    // The heaps' wait per frame, with the switch and without (an A/B line
    // covers both).
    let (on_ms, off_ms) = (s[HEAPS_NS_ON] / 1e6 / f, s[HEAPS_NS_OFF] / 1e6 / f);
    tracing::info!(
        "renderer port 7 (VICTORY_PORT_JOURNAL=1) per frame: heaps' records: the lane waited {on_ms:.2} ms ({:.1} preparings) with \
         the switch, {off_ms:.2} ms ({:.1}) without; {:.1} rounds asked for at the title's submit, {:.1} preparings found theirs \
         asked earlier, {:.1} asked it at the preparing, {:.1} had no mark; {:.1} preparings wrote a heap's memory; journal: {}",
        s[HEAPS_ON] / f,
        s[HEAPS_OFF] / f,
        s[ROUNDS_ASKED] / f,
        s[EARLY_ROUNDS] / f,
        s[LATE_ROUNDS] / f,
        s[NO_MARK] / f,
        s[PREP_WRITES] / f,
        journal.per_frame(f),
    );
}
