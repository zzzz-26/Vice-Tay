//! Which title memory the title wrote since a host copy took it (decision
//! 0007, "Copies of title memory"). The runtime reserves title GPU memory
//! with write watching (`victory-hle` xmem), so the kernel keeps a dirty bit
//! per 4 KiB page; asking for a range's written pages also clears them. Host
//! copies of the same memory can overlap (nested buffers, textures in
//! buffers), so what one asks is kept here for the others: a journal of
//! each page's last write, numbered by the asking.
//!
//! Physical-page views (xmem's sections) cannot be write-watched. There a
//! copy leaves each 64 KiB block it took read-only, and our vectored handler
//! takes the first write to it: it notes the block as written and gives the
//! write access back. The runtime tells us when it maps or unmaps views
//! ([`title_memory_remapped`]), which resets their protection. Blocks found
//! wholly uncommitted are skipped while they remain so; one query can
//! confirm a whole region, including ordinary reservations between views.
//!
//! The title shows one physical page at several addresses, and writes it
//! through one while the card reads it at another. So a block is clean, and
//! known by its bit, only while every view of its page is read-only
//! ([`cover`]): a write through any of them faults, opens them all and takes
//! clean and bit from all of them ([`open`]). A view mapped anew of a page
//! that is guarded elsewhere has the others looked at again
//! ([`views_mapped`]). What we write ourselves is told at one address and
//! counts at all ([`writing`], [`landed_beside`], [`wrote`]). Not so for a
//! page shown in more than [`MANY_VIEWS`] places (what the title shows where
//! it has nothing). `VICTORY_VIEW_GUARD=0`: each address for itself, as
//! before.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::{Bound, RangeBounds};
use std::sync::{Condvar, Mutex, RwLock};

/// The unit tracked and copied: 64 KiB, the console's page (the kernel's
/// 4 KiB pages fold into it). Finer units made the checks of the world's
/// big buffers cost more than the copies they saved.
pub(crate) const PAGE: usize = 64 << 10;
/// The kernel's page, the unit of its answers.
const KERNEL_PAGE: usize = 4096;

/// The parts of a page.
const PARTS: usize = PAGE / KERNEL_PAGE;
/// All of them.
const WHOLE: u16 = u16::MAX;

/// Whether the parts of a page that the kernel names are kept, and a refresh
/// copies those (`VICTORY_PAGE_PARTS=0`: the 64 KiB page, as before).
///
/// The kernel says which 4 KiB pages were written, and a copy took the 64 KiB
/// around each: 29 MB a frame written so were most of 114 MB copied, and as
/// much sent up and hashed again on the card (session play-1125-0203, run
/// cpu-2). What is checked is as before, a page.
fn parts_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PAGE_PARTS").is_none_or(|v| v != "0"))
}

impl Shard {
    const fn new() -> Shard {
        Shard { written: BTreeMap::new(), parts: std::collections::HashMap::with_hasher(std::hash::BuildHasherDefault::new()) }
    }

    /// Page `page` was found written by asking `now`: the parts in `parts`
    /// (a bit each). The asking that found it written before (0: none).
    ///
    /// A page's notes only go up: with the journal held alone `now` is the
    /// newest number there is, as it always was; noted under the shared lock
    /// (renderer port #7) another asking with a higher number may have
    /// noted it first.
    fn note(&mut self, page: usize, parts: u16, now: u64) -> u64 {
        let before = self.written.get(&page).copied().unwrap_or(0);
        self.written.insert(page, before.max(now));
        if parts == WHOLE || !parts_on() {
            if now >= before {
                self.parts.remove(&page);
            } else if let Some(all) = self.parts.get_mut(&page) {
                all.iter_mut().for_each(|at| *at = (*at).max(now));
            }
            return before;
        }
        let all = self.parts.entry(page).or_insert([before; PARTS]);
        for (k, at) in all.iter_mut().enumerate() {
            if parts >> k & 1 != 0 {
                *at = (*at).max(now);
            }
        }
        before
    }

    /// The parts of `page` written by an asking after `taken`.
    fn parts_since(&self, page: usize, taken: u64) -> u16 {
        match self.parts.get(&page) {
            Some(all) => all.iter().enumerate().filter(|(_, at)| **at > taken).fold(0, |m, (k, _)| m | 1 << k),
            None => WHOLE,
        }
    }
}

/// The pages of a part of the journal ([`Notes`]): pieces of
/// `1 << SHARD_SHIFT` pages (64 MiB), every [`SHARDS`]th piece in one part.
const SHARD_SHIFT: u32 = 10;
/// The parts: a bit each in a `u64` ([`shards_of`]).
const SHARDS: usize = 64;

fn shard_of(page: usize) -> usize {
    (page >> SHARD_SHIFT) % SHARDS
}

/// The parts that hold pages `[first, last)`, a bit each.
fn shards_of(first: usize, last: usize) -> u64 {
    if first >= last {
        return 0;
    }
    let (a, b) = (first >> SHARD_SHIFT, (last - 1) >> SHARD_SHIFT);
    if b - a + 1 >= SHARDS {
        return u64::MAX;
    }
    (a..=b).fold(0, |m, piece| m | 1 << (piece % SHARDS))
}

/// The parts that hold `pages`, a bit each.
fn shards_of_pages(pages: impl IntoIterator<Item = usize>) -> u64 {
    pages.into_iter().fold(0, |m, page| m | 1 << shard_of(page))
}

/// The pages' notes, in parts of their own, each under a lock of its own
/// (renderer port #7). Only reached through [`JOURNAL`]: held alone, its
/// holder changes them with no part's lock; held shared, a reader reads a
/// part at a time, and who notes or needs a number that counts holds the
/// parts it touches ([`Notes::hold`]).
struct Notes([RwLock<Shard>; SHARDS]);

/// A part of [`Notes`].
struct Shard {
    /// Page number → the asking that last found it written.
    written: BTreeMap<usize, u64>,
    /// The same for each of a page's sixteen parts ([`KERNEL_PAGE`]), where
    /// the parts are known ([`parts_on`]); a page that is not here was
    /// written whole, as far as anyone knows.
    parts: std::collections::HashMap<usize, [u64; PARTS], std::hash::BuildHasherDefault<PageHash>>,
}

impl Notes {
    const fn new() -> Notes {
        Notes([const { RwLock::new(Shard::new()) }; SHARDS])
    }

    /// The part of `page`, with the journal held alone.
    fn shard_mut(&mut self, page: usize) -> &mut Shard {
        self.0[shard_of(page)].get_mut().unwrap_or_else(|e| e.into_inner())
    }

    /// A part to read, the journal held shared or alone.
    fn read(&self, shard: usize) -> std::sync::RwLockReadGuard<'_, Shard> {
        if let Ok(s) = self.0[shard].try_read() {
            return s;
        }
        let began = std::time::Instant::now();
        let s = self.0[shard].read().unwrap_or_else(|e| e.into_inner());
        port7_waited(P7_PART_WAITS, began);
        s
    }

    /// See [`Shard::note`]; the journal held alone.
    fn note(&mut self, page: usize, parts: u16, now: u64) -> u64 {
        self.shard_mut(page).note(page, parts, now)
    }

    /// See [`Shard::parts_since`].
    fn parts_since(&self, page: usize, taken: u64) -> u16 {
        self.read(shard_of(page)).parts_since(page, taken)
    }

    /// The notes of pages `[first, last)`, one shard at a time, until the
    /// predicate says no. Large ranges visit each shard once.
    fn each(&self, first: usize, last: usize, mut each: impl FnMut(usize, u64) -> bool) -> bool {
        match pieces(first, last) {
            Some(mut pieces) => pieces.all(|(from, to)| {
                let s = self.read(shard_of(from));
                s.written.range(from..to).all(|(&p, &w)| each(p, w))
            }),
            None => (0..SHARDS).all(|shard| {
                self.read(shard).written.range(first..last).all(|(&p, &w)| each(p, w))
            }),
        }
    }

    /// The notes of pages `[first, last)` in page order. A part at a time.
    fn range(&self, first: usize, last: usize) -> Vec<(usize, u64)> {
        let mut out = Vec::new();
        self.each(first, last, |p, w| {
            out.push((p, w));
            true
        });
        if pieces(first, last).is_none() {
            out.sort_unstable();
        }
        out
    }

    /// The parts `shards` (a bit each), held until dropped: for writing
    /// (`write`) or reading. Taken in order: whoever waits for a part holds
    /// only lower ones, so no two holders wait for each other.
    fn hold(&self, shards: u64, write: bool) -> Held<'_> {
        let mut held = Held { shards, read: Vec::new(), write: Vec::new() };
        let mut m = shards;
        while m != 0 {
            let shard = m.trailing_zeros() as usize;
            m &= m - 1;
            let lock = &self.0[shard];
            if write {
                let guard = match lock.try_write() {
                    Ok(g) => g,
                    Err(_) => {
                        let began = std::time::Instant::now();
                        let g = lock.write().unwrap_or_else(|e| e.into_inner());
                        port7_waited(P7_PART_WAITS, began);
                        g
                    }
                };
                held.write.push(guard);
            } else {
                held.read.push(self.read(shard));
            }
        }
        held
    }
}

/// The pieces of pages `[first, last)` in page order, each in one part;
/// `None` for a range of more pieces than is worth walking (then each part
/// is looked at once).
fn pieces(first: usize, last: usize) -> Option<impl Iterator<Item = (usize, usize)>> {
    let (a, end) = if first < last {
        (first >> SHARD_SHIFT, ((last - 1) >> SHARD_SHIFT) + 1)
    } else {
        (0, 0)
    };
    if end - a > 4 * SHARDS {
        return None;
    }
    Some((a..end).map(move |piece| {
        let step = 1 << SHARD_SHIFT;
        (first.max(piece * step), last.min((piece + 1).saturating_mul(step)))
    }))
}

/// Parts of [`Notes`] held ([`Notes::hold`]).
struct Held<'a> {
    shards: u64,
    read: Vec<std::sync::RwLockReadGuard<'a, Shard>>,
    write: Vec<std::sync::RwLockWriteGuard<'a, Shard>>,
}

impl Held<'_> {
    /// The place among those held of the part of `page`.
    fn place(&self, page: usize) -> usize {
        let shard = shard_of(page);
        assert!(self.shards >> shard & 1 != 0, "page {page:#x}: its part is not held");
        (self.shards & ((1u64 << shard) - 1)).count_ones() as usize
    }

    fn shard(&self, page: usize) -> &Shard {
        let at = self.place(page);
        if self.write.is_empty() { &self.read[at] } else { &self.write[at] }
    }

    /// See [`Shard::note`]; held for writing.
    fn note(&mut self, page: usize, parts: u16, now: u64) -> u64 {
        let at = self.place(page);
        self.write[at].note(page, parts, now)
    }

    /// The notes of pages `[first, last)` in page order; their parts held.
    fn range(&self, first: usize, last: usize) -> Vec<(usize, u64)> {
        match pieces(first, last) {
            Some(pieces) => pieces
                .flat_map(|(from, to)| self.shard(from).written.range(from..to).map(|(&p, &w)| (p, w)))
                .collect(),
            None => {
                let all = if self.write.is_empty() {
                    self.read.iter().flat_map(|s| s.written.range(first..last)).map(|(&p, &w)| (p, w)).collect::<Vec<_>>()
                } else {
                    self.write.iter().flat_map(|s| s.written.range(first..last)).map(|(&p, &w)| (p, w)).collect::<Vec<_>>()
                };
                let mut all = all;
                all.sort_unstable();
                all
            }
        }
    }
}

/// Of the pages of a copy that were written after it took them (page number,
/// the asking it took the page at): the parts to take again, a bit for each
/// 4 KiB, in the same order.
pub(crate) fn parts_written(pages: &[(usize, u64)]) -> Vec<u16> {
    let j = journal();
    pages.iter().map(|&(page, taken)| j.notes.parts_since(page, taken)).collect()
}

/// Numbers the askings; 0 is "never". Counted up with the journal held
/// (shared will do: what notes an answer holds it alone).
static ASKED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn next_asking() -> u64 {
    ASKED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1
}

/// A title-memory range whose changed pages another subsystem needs to read.
pub(crate) struct Tracked {
    start: usize,
    end: usize,
    seen: Mutex<BTreeMap<usize, u64>>,
}

/// Tracks a runtime GPU-memory range, including physical-page views when
/// `views` is true. Other process memory is not tracked.
pub(crate) fn track(start: usize, len: usize, views: bool) -> Option<Tracked> {
    let end = start.checked_add(len)?;
    if len == 0 || writes(start, end, views).is_none() {
        return None;
    }
    Some(Tracked {
        start,
        end,
        seen: Mutex::default(),
    })
}

struct Journal {
    /// Per page, the asking that last found it written, and its parts'
    /// ([`Shard`]).
    notes: Notes,
    /// Page ranges asked ahead for a submission's refreshes ([`ask_ahead`])
    /// and the asking that did: `writes` answers for a range inside one
    /// from the journal, without another kernel call.
    ahead: Ahead,
    ahead_asked: u64,
    /// The title mapping revision when the standing query began.
    ahead_mapping: u64,
    /// A verified raw copy let this query stand past a title-memory write.
    kept_after_raw: bool,
    /// The ahead asking stands for a whole submission ([`ask_once`]): no
    /// other ahead asking is made, and `ahead` stays, until
    /// [`ask_once_end`].
    kept: bool,
}

impl Journal {
    fn covers(&self, first: usize, last: usize) -> bool {
        (!self.kept_after_raw || self.ahead_mapping == changes()) && self.ahead.holds(first, last)
    }
}

/// Read by many threads at once (a submission's refreshes run on several),
/// written only to note an answer: the kernel is asked, and view memory
/// guarded, with the range held in [`ASKING`] and not under this lock. Under
/// one lock for all of it those threads ran one at a time: 22 of 96 looks at
/// the submitting thread, and one in six of those at the six threads beside
/// it, found it waiting for that lock (run pooloff1).
static JOURNAL: RwLock<Journal> = RwLock::new(Journal {
    notes: Notes::new(),
    ahead: Ahead::new(),
    ahead_asked: 0,
    ahead_mapping: 0,
    kept_after_raw: false,
    kept: false,
});

/// The page ranges of [`Journal::ahead`] that are not contained in another.
/// Starts and ends both increase, so one binary search finds the range
/// with the furthest end among those starting at or before the query.
#[derive(Default)]
struct Ahead {
    ranges: Vec<(usize, usize)>,
}

impl Ahead {
    const fn new() -> Self {
        Self { ranges: Vec::new() }
    }

    fn from_ranges(mut ranges: Vec<(usize, usize)>) -> Self {
        ranges.sort_unstable_by_key(|&(start, end)| (start, std::cmp::Reverse(end)));
        let mut furthest_end = None;
        ranges.retain(|&(_, end)| {
            if furthest_end.is_some_and(|previous| end <= previous) {
                return false;
            }
            furthest_end = Some(end);
            true
        });
        Self { ranges }
    }

    fn clear(&mut self) {
        self.ranges.clear();
    }

    /// Whether one of the ranges holds pages `[first, last)`.
    fn holds(&self, first: usize, last: usize) -> bool {
        let past = self.ranges.partition_point(|&(start, _)| start <= first);
        past != 0 && self.ranges[past - 1].1 >= last
    }

    /// Adds a range if no single entry covers it, dropping entries it contains.
    fn insert_uncovered(&mut self, first: usize, last: usize) -> bool {
        let mut at = self.ranges.partition_point(|&(start, _)| start <= first);
        if at > 0 {
            let (start, end) = self.ranges[at - 1];
            if last <= end {
                return false;
            }
            if start == first {
                at -= 1;
            }
        }
        let end = at + self.ranges[at..].partition_point(|&(_, end)| end <= last);
        drop(self.ranges.splice(at..end, [(first, last)]));
        true
    }
}

fn journal() -> std::sync::RwLockReadGuard<'static, Journal> {
    if let Ok(j) = JOURNAL.try_read() {
        return j;
    }
    let began = std::time::Instant::now();
    let j = waited(1, || JOURNAL.read().unwrap_or_else(|e| e.into_inner()));
    port7_waited(P7_READ_WAITS, began);
    j
}

fn journal_mut() -> std::sync::RwLockWriteGuard<'static, Journal> {
    port7_count(P7_ALONE, 1);
    if let Ok(j) = JOURNAL.try_write() {
        return j;
    }
    let began = std::time::Instant::now();
    let j = waited(2, || JOURNAL.write().unwrap_or_else(|e| e.into_inner()));
    port7_waited(P7_ALONE_WAITS, began);
    j
}

/// Renderer port #7 (`VICTORY_PORT_JOURNAL=1`, off by default; read here,
/// and A/B-able in one run with `VICTORY_AB=<s>,VICTORY_PORT_JOURNAL`): the
/// journal's lock no longer makes the threads that read it wait for those
/// that note in it, nor those for each other, unless the standing asking
/// changes. (The switch also covers `crate::port_journal`'s heaps' records.)
///
/// What it was (runs guard-4, merge-4; ports 1, 2, 5, 6 on): every note in
/// the journal ([`JOURNAL`]) took it alone: the answers of askings found
/// written (`writes`, a submission's one asking, what shaders put in beside a
/// guard), and also what has nothing to change: each title buffer copy told
/// ([`wrote`]) with no asking standing, every flush's [`ask_ahead_done`] and
/// [`ask_once_end`]. 200 such takings a frame waited, 3-3.6 ms; the readers
/// (a refresh's every look, ~10,000 a frame on seven threads) waited 480
/// times, 5.4-5.9 ms; ~2.8 ms of it was the submitting thread's.
///
/// What it is with the switch: the pages' notes are in [`SHARDS`] parts of
/// their own ([`Notes`]), each with a lock of its own. What changes the
/// standing asking (its ranges, its number, whether it stands) still takes
/// the journal alone, and so does what our steps write while it stands
/// ([`wrote`], which moves its number). Noting an answer takes the journal
/// shared and the parts of its pages for writing ([`Notes::hold`]): an
/// asking about a range (`writes`), what shaders put in beside a guard
/// ([`landed_beside`]), and the pages a submission's asking found (noted
/// before its ranges and number are set, alone and briefly). What has
/// nothing to change looks first under the shared lock.
///
/// Why nothing is lost: an answer `(number, pages)` must hold every note of
/// a lower number of those pages. A number is taken only holding the parts
/// of the pages it notes or answers for (who takes a fresh number to answer
/// with holds the parts of its range for reading, switch or not), so a note
/// of a lower number is in, or its number is higher. The standing asking's
/// number is set with the journal held alone, after every shared holder that
/// took a lower one let go. A page's note only goes up ([`Shard::note`]).
/// Parts are taken in order, the journal's lock before them and never while
/// holding them, and no holder of a part waits for anything else: no two
/// wait for each other. Tests: `port7_tests`.
///
/// Measured (Series S, ports 1, 2, 5, 6 on; reports/PORT-PATCH-7.md): the
/// journal's waits 8-9 ms a frame of all threads -> ~0.8 ms (for parts), the
/// lane's 1.0 -> 0.0 ms; with the heaps' records (`crate::port_journal`)
/// the lane works 42.9 -> 40.4 ms and the frame is 1.3-2.9 ms faster in the
/// same run (A/B p7-1, p7-2), 46.1 -> 45.1-45.2 ms across runs.
pub(crate) fn port_journal() -> bool {
    #[cfg(feature = "diagnostics")]
    if let Some(on) = crate::ab::pick("VICTORY_PORT_JOURNAL") {
        return on;
    }
    #[cfg(test)]
    if PORT_JOURNAL_IN_TEST.get() {
        return true;
    }
    // Read here: the switch of renderer port #7.
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_PORT_JOURNAL").is_ok_and(|v| v.trim() == "1"))
}

#[cfg(test)]
thread_local! {
    /// [`port_journal`], for what this thread does.
    pub(crate) static PORT_JOURNAL_IN_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

thread_local! {
    /// Whether this thread is the submitting lane ("victory-submit").
    static ON_LANE: bool = std::thread::current().name() == Some("victory-submit");
}

/// Renderer port #7's counts since the last line: takings of the journal
/// alone; of them, those that waited and nanoseconds; waits for it shared
/// and nanoseconds; the lane's waits for it (both kinds) and nanoseconds;
/// takings alone found needless and left out; notings done shared; waits for
/// a part and nanoseconds.
static PORT7: [std::sync::atomic::AtomicU64; 12] = [const { std::sync::atomic::AtomicU64::new(0) }; 12];
const P7_ALONE: usize = 0;
/// (A count and its nanoseconds next to it, as the others below.)
const P7_ALONE_WAITS: usize = 1;
const P7_READ_WAITS: usize = 3;
const P7_LANE_WAITS: usize = 5;
const P7_SKIPPED: usize = 7;
const P7_SHARED_NOTES: usize = 8;
const P7_PART_WAITS: usize = 9;

fn port7_count(what: usize, n: u64) {
    PORT7[what].fetch_add(n, std::sync::atomic::Ordering::Relaxed);
}

/// A wait of kind `what` (a count, its nanoseconds next) that began then.
fn port7_waited(what: usize, began: std::time::Instant) {
    let ns = began.elapsed().as_nanos() as u64;
    port7_count(what, 1);
    port7_count(what + 1, ns);
    if what != P7_PART_WAITS && ON_LANE.with(|l| *l) {
        port7_count(P7_LANE_WAITS, 1);
        port7_count(P7_LANE_WAITS + 1, ns);
    }
}

/// Renderer port #7's journal counts, taken ([`crate::port_journal::line`]).
pub(crate) struct Port7Counts([u64; 12]);

pub(crate) fn port_journal_counts() -> Port7Counts {
    Port7Counts(std::array::from_fn(|i| PORT7[i].swap(0, std::sync::atomic::Ordering::Relaxed)))
}

impl Port7Counts {
    /// For the line, per frame.
    pub(crate) fn per_frame(&self, frames: f64) -> String {
        let n = |i: usize| self.0[i] as f64 / frames;
        let ms = |i: usize| self.0[i] as f64 / 1e6 / frames;
        format!(
            "the lane waited for its lock {:.1} times, {:.2} ms; all threads waited to take it alone {:.1} times ({:.2} ms) of {:.1} \
             takings alone, shared {:.1} times ({:.2} ms); {:.1} takings alone found needless, {:.1} notings done shared, {:.1} waits \
             for a part ({:.2} ms)",
            n(P7_LANE_WAITS),
            ms(P7_LANE_WAITS + 1),
            n(P7_ALONE_WAITS),
            ms(P7_ALONE_WAITS + 1),
            n(P7_ALONE),
            n(P7_READ_WAITS),
            ms(P7_READ_WAITS + 1),
            n(P7_SKIPPED),
            n(P7_SHARED_NOTES),
            n(P7_PART_WAITS),
            ms(P7_PART_WAITS + 1),
        )
    }
}

/// Since the last line: waits and nanoseconds of waiting, of all threads, for
/// the ranges held ([`ASKING`]), the journal to read it, the journal to note
/// in it, and the guards (faults of the title's among them).
#[cfg(feature = "diagnostics")]
static WAITS: [[std::sync::atomic::AtomicU64; 2]; 4] = [const { [const { std::sync::atomic::AtomicU64::new(0) }; 2] }; 4];

#[cfg(feature = "diagnostics")]
fn waited<T>(what: usize, f: impl FnOnce() -> T) -> T {
    use std::sync::atomic::Ordering::Relaxed;
    let began = std::time::Instant::now();
    let got = f();
    WAITS[what][0].fetch_add(1, Relaxed);
    WAITS[what][1].fetch_add(began.elapsed().as_nanos() as u64, Relaxed);
    got
}

#[cfg(not(feature = "diagnostics"))]
fn waited<T>(_: usize, f: impl FnOnce() -> T) -> T {
    f()
}

/// [`WAITS`], taken, for the log: per frame.
#[cfg(feature = "diagnostics")]
pub(crate) fn lock_waits(frames: f64) -> String {
    let of = |what: usize| {
        let [n, ns] = [0, 1].map(|i| WAITS[what][i].swap(0, std::sync::atomic::Ordering::Relaxed));
        format!("{:.0} {:.2} ms", n as f64 / frames, ns as f64 / 1e6 / frames)
    };
    format!("ranges held {}, journal read {}, journal noted in {}, guards {}", of(0), of(1), of(2), of(3))
}

/// The page ranges some thread is asking about: asking forgets what it
/// answers, so until the answer is in the journal no other thread may ask
/// about any of the same pages (it would be told they were not written).
/// Taken before the journal's lock, never under it.
static ASKING: (Mutex<Vec<(usize, usize)>>, Condvar) = (Mutex::new(Vec::new()), Condvar::new());

/// Pages `[first, last)` held in [`ASKING`] until dropped.
struct Asking(usize, usize);

impl Asking {
    fn hold(first: usize, last: usize) -> Asking {
        let mut held = ASKING.0.lock().unwrap_or_else(|e| e.into_inner());
        if held.iter().any(|&(a, b)| a < last && first < b) {
            held = waited(0, || {
                let mut held = held;
                while held.iter().any(|&(a, b)| a < last && first < b) {
                    held = ASKING.1.wait(held).unwrap_or_else(|e| e.into_inner());
                }
                held
            });
        }
        held.push((first, last));
        Asking(first, last)
    }

    /// Everything: an asking for a whole submission.
    fn hold_all() -> Asking {
        Asking::hold(0, usize::MAX)
    }
}

impl Drop for Asking {
    fn drop(&mut self) {
        let mut held = ASKING.0.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(i) = held.iter().position(|&r| r == (self.0, self.1)) {
            held.swap_remove(i);
        }
        drop(held);
        ASKING.1.notify_all();
    }
}

thread_local! {
    /// Room for the kernel's answer.
    static FOUND: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Whether an asking walks the pages without the kernel's reset, and resets
/// only the pages it found written after (`VICTORY_ASK_NORESET=1`; off by
/// default). What is answered is the same: a page found written is reset
/// before the caller copies it, so a write after the walk is in the copy or
/// seen by the next asking; a page not found is not reset.
fn no_reset() -> bool {
    #[cfg(feature = "diagnostics")]
    if let Some(on) = crate::ab::pick("VICTORY_ASK_NORESET") {
        return on;
    }
    // Read here: askings that reset only what they found.
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_ASK_NORESET").is_some_and(|v| v == "1"))
}

/// Resets the write watch of the 4 KiB pages at `found` (in order), a run of
/// neighbours at a time.
fn reset_found(found: &[usize]) {
    use windows_sys::Win32::System::Memory::ResetWriteWatch;
    let mut i = 0;
    while i < found.len() {
        let mut k = i + 1;
        while k < found.len() && found[k] == found[k - 1] + KERNEL_PAGE {
            k += 1;
        }
        // SAFETY: pages the kernel just reported write-watched.
        unsafe { ResetWriteWatch(found[i] as *const _, (k - i) * KERNEL_PAGE) };
        i = k;
    }
}

/// Whether a submission's askings are made on several threads at once
/// (`VICTORY_ASK_SPREAD=1`; off by default): the same askings of the same
/// ranges, all held by the asking thread meanwhile.
fn spread() -> bool {
    #[cfg(feature = "diagnostics")]
    if let Some(on) = crate::ab::pick("VICTORY_ASK_SPREAD") {
        return on;
    }
    // Read here: askings on several threads.
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_ASK_SPREAD").is_some_and(|v| v == "1"))
}

/// `each(i)` for every `i` below `n`, on the threads kept for landing if
/// [`spread`] and there are enough; what each returns, in order.
fn spread_each<T: Send>(n: usize, each: impl Fn(usize) -> T + Sync) -> Vec<T> {
    if !spread() || n < 4 {
        return (0..n).map(each).collect();
    }
    let out: Vec<Mutex<Option<T>>> = (0..n).map(|_| Mutex::new(None)).collect();
    crate::land::each_under(n, &|i| {
        let v = each(i);
        *out[i].lock().unwrap_or_else(|e| e.into_inner()) = Some(v);
    });
    out.into_iter().filter_map(|m| m.into_inner().unwrap_or_else(|e| e.into_inner())).collect()
}

/// The pages of `[first, last)` the kernel says were written since it was
/// last asked, in order, each once, with their written parts; `None` if the range is not write-watched
/// memory of one reservation. The caller holds the range ([`Asking`]).
fn kernel_written(first: usize, last: usize) -> Option<Vec<(usize, u16)>> {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    FOUND.with(|found| {
        let mut found = found.borrow_mut();
        let mut count = (last - first) * PAGE / KERNEL_PAGE;
        if found.len() < count {
            found.resize(count, 0);
        }
        let mut granularity = 0;
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        let keep = no_reset();
        // SAFETY: the answer buffer holds `count` entries.
        let failed = unsafe {
            GetWriteWatch(
                if keep { 0 } else { WRITE_WATCH_FLAG_RESET },
                (first * PAGE) as *const _,
                (last - first) * PAGE,
                found.as_mut_ptr().cast(),
                &mut count,
                &mut granularity,
            )
        } != 0;
        #[cfg(feature = "diagnostics")]
        {
            tally(10, u64::from(failed));
            tally(11, started.elapsed().as_nanos() as u64);
        }
        if failed {
            // (On the timeline: how many askings the kernel refuses.)
            crate::timeline::span("asking refused").end();
            return None;
        }
        if keep {
            reset_found(&found[..count]);
        }
        #[cfg(feature = "diagnostics")]
        {
            tally(12, count as u64);
            use crate::copystat::{C, add};
            add(C::Watches, 1);
            add(C::WatchUs, started.elapsed().as_micros() as u64);
            slow(started.elapsed().as_micros() as u64);
            tally(0, 1);
            tally(1, ((last - first) * (PAGE >> 10)) as u64);
        }
        let mut pages: Vec<(usize, u16)> = Vec::new();
        for &at in &found[..count] {
            let (page, part) = (at / PAGE, 1u16 << (at % PAGE / KERNEL_PAGE));
            match pages.last_mut() {
                Some((last, parts)) if *last == page => *parts |= part,
                _ => pages.push((page, part)),
            }
        }
        Some(pages)
    })
}

/// Ranges of one submission closer than this are asked about together:
/// walking the pages between them costs less than another kernel call (a
/// call that resets written pages costs 30-70 microseconds whatever the
/// range).
const AHEAD_GAP: usize = (8 << 20) / PAGE;
/// The most one asking covers (its answer needs 8 bytes per 4 KiB page).
const AHEAD_MAX: usize = (1 << 30) / PAGE;
/// Asks the kernel about many ranges `(start, end)` at once, before the
/// [`writes`] of each: ranges close together are one asking, and `writes`
/// then answers from the journal until [`ask_ahead_done`]. The world's
/// submissions refresh thousands of host copies (6,664 askings a frame,
/// 456 ms, in run 14). Ranges that are not write-watched, or lie alone, are
/// left to `writes`.
pub(crate) fn ask_ahead(ranges: impl Iterator<Item = (usize, usize)>) {
    // Skip building and sorting the batch while its submission's asking stands.
    if JOURNAL.try_read().is_ok_and(|journal| journal.kept) {
        return;
    }
    ask_ahead_held(ranges, false);
}

/// [`ask_ahead`]; `held`: the caller holds everything ([`Asking::hold_all`]).
fn ask_ahead_held(ranges: impl Iterator<Item = (usize, usize)>, held: bool) {
    let mut pieces: Vec<(usize, usize)> = ranges
        // Not view memory known by its bits: the kernel would refuse the
        // asking (and, asked together with others, theirs).
        .filter(|r| r.0 < r.1 && !published(r.0, r.1) && !refused_as_view(r.0, r.1))
        .map(|(start, end)| (start / PAGE, end.div_ceil(PAGE)))
        .collect();
    if pieces.len() < 2 {
        return;
    }
    pieces.sort_unstable();
    // The submission's one asking stands ([`ask_once`]): what it does not
    // hold is left to `writes`.
    if journal().kept {
        return;
    }
    let _all = (!held).then(Asking::hold_all);
    let (mut asked, mut written) = (Vec::new(), Vec::new());
    let mut runs = Vec::new();
    let mut i = 0;
    while i < pieces.len() {
        let first = pieces[i].0;
        let mut last = pieces[i].1;
        let mut k = i + 1;
        while k < pieces.len()
            && pieces[k].0 <= last + AHEAD_GAP
            && last.max(pieces[k].1) - first <= AHEAD_MAX
        {
            last = last.max(pieces[k].1);
            k += 1;
        }
        runs.push(i..k);
        i = k;
    }
    for (a, w) in spread_each(runs.len(), |r| {
        let (mut a, mut w) = (Vec::new(), Vec::new());
        ask_run(&pieces[runs[r].clone()], &mut a, &mut w);
        (a, w)
    }) {
        asked.extend(a);
        written.extend(w);
    }
    let ahead = Ahead::from_ranges(asked);
    note_hot(&written);
    if port_journal() {
        // Renderer port #7: the pages noted shared, under their parts; then
        // the number and the ranges, alone (no shared holder with a lower
        // number is left by then).
        let now = note_shared(&written);
        let mut j = journal_mut();
        j.ahead_asked = now;
        j.ahead = ahead;
        return;
    }
    let mut j = journal_mut();
    j.ahead_asked = next_asking();
    j.ahead = ahead;
    let now = j.ahead_asked;
    for (page, parts) in written {
        j.notes.note(page, parts, now);
    }
}

/// Renderer port #7: notes `written` (pages and their parts) as found by an
/// asking with a number taken now, holding the journal shared and their
/// parts for writing; the number.
fn note_shared(written: &[(usize, u16)]) -> u64 {
    let j = journal();
    let mut held = j.notes.hold(shards_of_pages(written.iter().map(|w| w.0)), true);
    let now = next_asking();
    for &(page, parts) in written {
        held.note(page, parts, now);
    }
    port7_count(P7_SHARED_NOTES, 1);
    now
}

/// Whether the kernel would refuse to ask about `[start, end)` for its view
/// memory (renderer port #6): read-only by our hand, or known a view
/// ([`VIEWS`]). Such a range is left to [`writes`], which goes to the guards
/// at once; asked with others it failed their asking too, which was then
/// asked again by halves.
fn refused_as_view(start: usize, end: usize) -> bool {
    if !port_guard() || !light() {
        return false;
    }
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    let refused = PROTECTED.any(first, last) || VIEWS.any(first, last);
    if refused {
        port6_count(P6_NOT_ASKED, 1);
    }
    refused
}

/// Since the last frame line: the kernel askings that took a millisecond
/// or more, their microseconds, and the longest asking's. An asking walks
/// page tables, some ten microseconds for most ranges; in run 75 the 2,644
/// askings of a frame took 219 ms in the lines where the pass over the
/// heap's views ran long, 25-50 ms in the others. These numbers say
/// whether a few askings wait for something or all of them are slow.
#[cfg(feature = "diagnostics")]
static SLOW: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

#[cfg(feature = "diagnostics")]
fn slow(us: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    if us >= 1000 {
        SLOW[0].fetch_add(1, Relaxed);
        SLOW[1].fetch_add(us, Relaxed);
    }
    SLOW[2].fetch_max(us, Relaxed);
}

/// [`SLOW`], taken.
#[cfg(feature = "diagnostics")]
pub(crate) fn slow_asks() -> [u64; 3] {
    [0, 1, 2].map(|i| SLOW[i].swap(0, std::sync::atomic::Ordering::Relaxed))
}

/// Whether a thread asks, between a submission's askings, about the memory
/// they found written (`VICTORY_ASK_EARLY=<ms>` between its rounds; off
/// unless given: in run cpu-2 the submissions' own askings were none the
/// cheaper for its 650 a frame, 12 to 18 ms where 9 were).
///
/// An asking's cost is the pages it finds written and resets, some 1.4
/// microseconds each, and the title writes 7,000 a frame: the first
/// submission that refreshes its 174 MB heap stood 8 to 10 ms in one call
/// (session play-1125-0203). Asked a little at a time while the title
/// records, the same pages are in the journal when the submission comes, and
/// its own asking finds those written since.
///
/// Nothing is seen later than before: a page goes into the journal as found
/// written by an asking of its own number, as from any thread that asks, and
/// the submission still asks for itself. A page the title writes again after
/// such a round faults once more.
fn early() -> Option<std::time::Duration> {
    static EVERY: std::sync::OnceLock<Option<std::time::Duration>> = std::sync::OnceLock::new();
    *EVERY.get_or_init(|| {
        let ms = match std::env::var("VICTORY_ASK_EARLY") {
            Ok(v) => v.trim().parse::<f64>().unwrap_or(2.0),
            Err(_) => 0.0,
        };
        std::time::Duration::try_from_secs_f64(ms / 1000.0).ok().filter(|d| !d.is_zero())
    })
}

/// The pieces of memory (by [`EARLY_PIECE`] pages) that submissions' askings
/// found written lately: how many pages each, halved every round of 64.
static HOT: Mutex<BTreeMap<usize, u32>> = Mutex::new(BTreeMap::new());
/// The pieces asked in a round, at most.
const EARLY_MOST: usize = 24;
/// The pages of a piece: 2 MiB.
const EARLY_PIECE: usize = (2 << 20) / PAGE;
/// Told when a submission asked.
static HOT_MOVED: Condvar = Condvar::new();

/// Since the last frame line: pieces asked early, the microseconds of it.
static EARLY: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// [`EARLY`], taken.
#[cfg(feature = "diagnostics")]
pub(crate) fn early_asks() -> [u64; 2] {
    [0, 1].map(|i| EARLY[i].swap(0, std::sync::atomic::Ordering::Relaxed))
}

/// A submission's asking found `written` (pages).
fn note_hot(written: &[(usize, u16)]) {
    let Some(every) = early() else {
        return;
    };
    let mut hot = HOT.lock().unwrap_or_else(|e| e.into_inner());
    for (page, _) in written {
        let n = hot.entry(page / EARLY_PIECE).or_default();
        *n = (*n + 1).min(1 << 10);
    }
    HOT_MOVED.notify_one();
    drop(hot);
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        let _ = std::thread::Builder::new().name("victory-ask-early".into()).spawn(move || ask_early(every));
    });
}

fn ask_early(every: std::time::Duration) {
    use std::sync::atomic::Ordering::Relaxed;
    // So many rounds without a submission's asking: wait for one.
    const IDLE: u32 = 100;
    let mut round = 0u32;
    let mut idle = 0;
    let mut last = 0u64;
    loop {
        std::thread::sleep(every);
        let pieces: Vec<usize> = {
            let mut hot = HOT.lock().unwrap_or_else(|e| e.into_inner());
            let now = ASKED.load(Relaxed);
            idle = if now == last { idle + 1 } else { 0 };
            while idle > IDLE {
                hot = HOT_MOVED.wait(hot).unwrap_or_else(|e| e.into_inner());
                idle = 0;
            }
            round += 1;
            if round.is_multiple_of(64) {
                hot.retain(|_, n| {
                    *n /= 2;
                    *n > 0
                });
            }
            // The most written: a piece with a few pages is as dear to ask as
            // one with many.
            let mut most: Vec<(u32, usize)> = hot.iter().filter(|h| *h.1 >= 16).map(|(&p, &n)| (n, p)).collect();
            most.sort_unstable_by(|a, b| b.cmp(a));
            most.into_iter().take(EARLY_MOST).map(|(_, p)| p).collect()
        };
        let began = std::time::Instant::now();
        let mut asked = 0;
        for piece in pieces {
            // (A submission's asking stands: what it holds is answered from
            // the journal, and costs nothing here.)
            if published(piece * EARLY_PIECE * PAGE, (piece + 1) * EARLY_PIECE * PAGE) {
                continue;
            }
            asked += 1;
            let _ = writes(piece * EARLY_PIECE * PAGE, (piece + 1) * EARLY_PIECE * PAGE, false);
        }
        EARLY[0].fetch_add(asked, Relaxed);
        EARLY[1].fetch_add(began.elapsed().as_micros() as u64, Relaxed);
        last = ASKED.load(Relaxed);
    }
}

/// The submission's refreshes are done: `writes` asks the kernel again.
pub(crate) fn ask_ahead_done() {
    // Renderer port #7: nothing to change is seen shared.
    if port_journal() {
        let j = journal();
        if j.kept || j.ahead.ranges.is_empty() {
            port7_count(P7_SKIPPED, 1);
            return;
        }
    }
    let mut j = journal_mut();
    if !j.kept {
        j.ahead.clear();
    }
}

/// Whether a submission asks the kernel once, before its first refresh,
/// about every range its lists refresh (`VICTORY_ASK_ONCE=1`; off by
/// default).
///
/// A submission's refreshes are done in groups, between the title's own
/// buffer copies (each copy flushes the refreshes recorded before it), and
/// every group asked the kernel again: 2,644 askings a frame, 70 a
/// submission. With the switch the journal answers all of them from one
/// asking made when the submission starts.
///
/// What changes: memory the title's other threads write while the
/// submission is being prepared is seen by the next submission's asking,
/// not by this one's later groups. What this submission's lists read was
/// written before they were submitted, so before the asking. Our own
/// writes into title memory while preparing are told to the journal
/// ([`wrote`]) or end the asking ([`ask_once_end`]).
pub(crate) fn once() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // On by default since run 81 (with card buffers: 375-450 ms a frame
    // against 573-647, the picture unchanged over four minutes);
    // `VICTORY_ASK_ONCE=0` asks per range, as before.
    *ON.get_or_init(|| std::env::var_os("VICTORY_ASK_ONCE").is_none_or(|v| v != "0"))
}

/// Tests that go through the submission's one asking ([`ask_once`]) hold
/// this while they run: the asking is global, so two tests asking at once end
/// each other's, and a write one of them waits for can be taken by the other.
#[cfg(test)]
pub(crate) fn one_asking_test() -> std::sync::MutexGuard<'static, ()> {
    static ONE: std::sync::Mutex<()> = std::sync::Mutex::new(());
    ONE.lock().unwrap_or_else(|e| e.into_inner())
}

/// With [`once`]: asks about `ranges` (everything a submission's lists
/// refresh) as [`ask_ahead`] does, and lets that asking stand until
/// [`ask_once_end`].
pub(crate) fn ask_once(ranges: &[(usize, usize)]) {
    if !once() {
        return;
    }
    let mut spent = crate::timeline::span("asking: ahead");
    ask_once_end();
    // Both parts under one hold of everything: a second hold waited for the
    // other threads' askings again (0.5 ms a time, run fills-ab3), and no
    // other asking comes between the two parts.
    let _all = Asking::hold_all();
    let mapping = changes();
    ask_ahead_held(ranges.iter().copied(), true);
    spent.next("asking: the rest");
    ask_rest(ranges, true, mapping);
    spent.end();
}

/// [`ask_once`] after its [`ask_ahead`]: what that left to `writes`, and the
/// asking stands.
fn ask_rest(ranges: &[(usize, usize)], held: bool, mapping: u64) {
    let _all = (!held).then(Asking::hold_all);
    let mut ahead = {
        let mut j = journal_mut();
        std::mem::take(&mut j.ahead)
    };
    // `ask_ahead` leaves a range that lies alone to `writes`, which asks
    // once per flush: with one asking for the submission that was every
    // such range asked again at each of its 70 flushes (run 76: 2,300
    // askings a frame with the switch on, as without it). They are asked
    // here, once; what is not write-watched stays with `writes`.
    let mut written = Vec::new();
    // Keep the order of the first queries. The index only checks coverage.
    let mut alone_ranges: Vec<(usize, usize)> = Vec::new();
    let mut alone_index = Ahead::new();
    for &(start, end) in ranges {
        if start >= end || published(start, end) || refused_as_view(start, end) {
            continue;
        }
        let (first, last) = (start / PAGE, end.div_ceil(PAGE));
        if !ahead.holds(first, last) && alone_index.insert_uncovered(first, last) {
            alone_ranges.push((first, last));
        }
    }
    let answers = spread_each(alone_ranges.len(), |r| {
        let (first, last) = alone_ranges[r];
        #[cfg(feature = "diagnostics")]
        let asking = std::time::Instant::now();
        let found = kernel_written(first, last);
        #[cfg(feature = "diagnostics")]
        alone(first * PAGE, last * PAGE, found.as_ref().map_or(0, Vec::len), asking.elapsed());
        found
    });
    for (&(first, last), found) in alone_ranges.iter().zip(answers) {
        if let Some(pages) = found {
            written.extend(pages);
            ahead.insert_uncovered(first, last);
        }
    }
    note_hot(&written);
    // A number of its own, and the one this asking stands under: between
    // the two parts another thread may have asked about a range of the
    // second, copied it, and the title written it since. Under the first
    // part's number that write was older than the copy, and never copied.
    if port_journal() {
        // Renderer port #7: noted shared first (see `ask_ahead_held`).
        let now = note_shared(&written);
        let mut j = journal_mut();
        j.ahead_asked = now;
        j.ahead_mapping = mapping;
        j.kept_after_raw = false;
        j.kept = !ahead.ranges.is_empty();
        j.ahead = ahead;
        return;
    }
    let mut j = journal_mut();
    let now = next_asking();
    j.ahead_asked = now;
    j.ahead_mapping = mapping;
    j.kept_after_raw = false;
    for (page, parts) in written {
        j.notes.note(page, parts, now);
    }
    j.kept = !ahead.ranges.is_empty();
    j.ahead = ahead;
}

/// Diagnostics: the ranges a submission's asking asks about alone
/// ([`ask_rest`]), by (start, end): askings, 64 KiB pages found written,
/// microseconds.
#[cfg(feature = "diagnostics")]
static ALONE: Mutex<Option<std::collections::HashMap<(usize, usize), (u64, u64, u64)>>> = Mutex::new(None);

#[cfg(feature = "diagnostics")]
fn alone(start: usize, end: usize, pages: usize, took: std::time::Duration) {
    let mut all = ALONE.lock().unwrap_or_else(|e| e.into_inner());
    let e = all.get_or_insert_with(Default::default).entry((start, end)).or_default();
    *e = (e.0 + 1, e.1 + pages as u64, e.2 + took.as_micros() as u64);
}

/// Diagnostics, for the frame lines: [`ALONE`] per frame, the busiest.
#[cfg(feature = "diagnostics")]
pub(crate) fn alone_line(frames: f64) {
    let all = ALONE.lock().unwrap_or_else(|e| e.into_inner()).take().unwrap_or_default();
    if all.is_empty() {
        return;
    }
    let (n, pages, us) = all.values().fold((0, 0, 0), |s, v| (s.0 + v.0, s.1 + v.1, s.2 + v.2));
    let mut all: Vec<_> = all.into_iter().collect();
    all.sort_unstable_by_key(|(_, v)| std::cmp::Reverse(v.2));
    let top: Vec<String> = all
        .iter()
        .take(12)
        .map(|((s, e), (k, p, u))| {
            format!("{s:#x}+{:#x} {:.1}x {:.1} pages {:.2} ms", e - s, *k as f64 / frames, *p as f64 / frames, *u as f64 / 1e3 / frames)
        })
        .collect();
    tracing::info!(
        "ranges a submission's asking asks about alone, per frame: {:.1} askings of {} ranges, {:.1} pages of 64 KiB found written, \
         {:.2} ms; the costliest: {}",
        n as f64 / frames,
        all.len(),
        pages as f64 / frames,
        us as f64 / 1e3 / frames,
        top.join("; ")
    );
}

/// The submission's one asking stands no longer: its refreshes are done,
/// or title memory was written in a way the journal was not told.
pub(crate) fn ask_once_end() {
    // Renderer port #7: none standing is seen shared.
    if port_journal() && !journal().kept {
        port7_count(P7_SKIPPED, 1);
        return;
    }
    let mut j = journal_mut();
    if j.kept {
        j.kept = false;
        j.kept_after_raw = false;
        j.ahead.clear();
    }
}

/// We wrote title memory `[at, at + len)` while a submission's one asking
/// stands (the title's buffer copies, done at submission): its pages count
/// as written after that asking, so the refreshes that follow copy them.
pub(crate) fn wrote(at: usize, len: usize) {
    // Renderer port #7: with no asking standing there is nothing to note,
    // which is seen shared. (Standing, its number moves: alone, as before.)
    if port_journal() && (len == 0 || !journal().kept) {
        port7_count(P7_SKIPPED, 1);
        return;
    }
    let mut j = journal_mut();
    if !j.kept || len == 0 {
        return;
    }
    let now = next_asking();
    // What is copied from now on counts as copied after these writes (fork
    // PR 410): answers older than the writes in them made every refresh
    // until the next asking copy the pages again.
    j.ahead_asked = now;
    for page in at / PAGE..at.saturating_add(len).div_ceil(PAGE) {
        j.notes.note(page, WHOLE, now);
    }
    for (_, others) in elsewhere(at, len) {
        for other in others {
            j.notes.note(other / PAGE, WHOLE, now);
        }
    }
}

/// Bound the extra kernel probe for a raw copy to 256 KiB.
pub(crate) const RAW_COPY_WATCH_LIMIT: usize = 256 << 10;

pub(crate) struct RawCopyNote {
    first: usize,
    last: usize,
    mapping: u64,
}

/// Prepare a note for a raw copy while its mapping is held. Covered ranges
/// were write-watched when queried. Confirm the current destination with a
/// non-resetting probe, since a raw address can name a mapped alias.
pub(crate) fn prepare_raw_copy_note(_reading: &Reading, at: usize, len: usize) -> Option<RawCopyNote> {
    if len == 0 {
        return None;
    }
    let end = at.checked_add(len)?;
    let (first, last) = (at / PAGE, end.div_ceil(PAGE));
    let mapping = changes();
    {
        let j = journal();
        if !j.kept || j.ahead_mapping != mapping || !j.covers(first, last) {
            return None;
        }
    }

    let mut addresses = [0usize; RAW_COPY_WATCH_LIMIT / KERNEL_PAGE];
    if last - first > addresses.len() / (PAGE / KERNEL_PAGE) {
        return None;
    }
    let bytes = (last - first) * PAGE;
    let mut count = bytes / KERNEL_PAGE;
    let mut granularity = 0;
    // SAFETY: the answer fits in `addresses`. Flag 0 keeps the reported bits.
    let result = unsafe {
        windows_sys::Win32::System::Memory::GetWriteWatch(
            0,
            (first * PAGE) as *const _,
            bytes,
            addresses.as_mut_ptr().cast(),
            &mut count,
            &mut granularity,
        )
    };
    (result == 0).then_some(RawCopyNote { first, last, mapping })
}

impl RawCopyNote {
    pub(crate) fn written(self, _reading: &Reading) -> bool {
        let mut j = journal_mut();
        if !j.kept {
            return true;
        }
        if changes() != self.mapping || j.ahead_mapping != self.mapping || !j.covers(self.first, self.last) {
            return false;
        }
        let now = next_asking();
        j.ahead_asked = now;
        j.kept_after_raw = true;
        for page in self.first..self.last {
            j.notes.note(page, WHOLE, now);
        }
        true
    }
}

/// The blocks of `[at, at + len)` whose pages the title shows at other
/// addresses too, and there.
fn elsewhere(at: usize, len: usize) -> Vec<(usize, Vec<usize>)> {
    if !view_guard() || len == 0 {
        return Vec::new();
    }
    crate::inplace::shown_elsewhere(at / BLOCK * BLOCK, at.saturating_add(len))
}

/// One kernel asking for `pieces` (page ranges in order); by halves if it
/// fails, as a run across two reservations or into memory that is not
/// write-watched does. A piece alone is left to [`writes`]. Adds what was
/// asked to `asked` and the pages found written to `written`.
fn ask_run(pieces: &[(usize, usize)], asked: &mut Vec<(usize, usize)>, written: &mut Vec<(usize, u16)>) {
    if pieces.len() < 2 {
        return;
    }
    let first = pieces[0].0;
    let last = pieces.iter().map(|p| p.1).max().unwrap_or(first);
    if let Some(pages) = kernel_written(first, last) {
        written.extend(pages);
        asked.push((first, last));
        return;
    }
    let (a, b) = pieces.split_at(pieces.len() / 2);
    ask_run(a, asked, written);
    ask_run(b, asked, written);
}

/// What [`writes`] would say of `[start, end)` from the journal alone, if
/// the range lies in one asked ahead ([`ask_ahead`], [`ask_once`]): the
/// pages written since tracking began and the asking that last found each
/// written, with the number of the asking that stands (as [`writes`] gives
/// it). `None` if the kernel would have to be asked.
pub(crate) fn asked(start: usize, end: usize) -> Option<(u64, Vec<(usize, u64)>)> {
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    let j = journal();
    j.covers(first, last).then(|| (j.ahead_asked, j.notes.range(first, last)))
}

/// The pages of `[start, end)` (whole pages) as last written, for pages
/// written since tracking began: `(page number, asking)` in page order, and
/// this asking's number, which is higher than any in the list but those the
/// kernel just reported. `None` if the memory is not write-watched (nor,
/// with `guarded`, view memory, which is then guarded).
pub(crate) fn writes(start: usize, end: usize, guarded: bool) -> Option<(u64, Vec<(usize, u64)>)> {
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    // Asked ahead with the submission's other ranges: the journal has it.
    // (The standing number changes only with the journal held alone, so its
    // notes are all in: a part at a time will do.)
    let ahead = |j: &Journal| j.covers(first, last).then(|| (j.ahead_asked, j.notes.range(first, last)));
    // A fresh number and the notes of the range, its parts held meanwhile
    // (renderer port #7, switch or not: who notes shared takes its number
    // holding the parts it notes in).
    let fresh = || {
        let j = journal();
        let held = j.notes.hold(shards_of(first, last), false);
        (next_asking(), held.range(first, last))
    };
    if let Some(known) = ahead(&journal()) {
        return Some(known);
    }
    // Read-only since an asking noted it: the bits first, then the journal.
    if guarded && published(start, end) {
        #[cfg(feature = "diagnostics")]
        {
            tally(2, 1);
            crate::faultbook::note(crate::faultbook::What::Known, first, last, 0);
        }
        return Some(fresh());
    }
    // Renderer port #6: view memory, which the kernel would refuse, and the
    // caller does not ask the guards (textures, which then guard by a way of
    // their own): no kernel call, ~300 a frame (run guard-2).
    if !guarded && refused_as_view(start, end) {
        return None;
    }
    // Asked, and the answer noted, with the range held: nothing else asks
    // about any of it in between.
    let _range = Asking::hold(first, last);
    if let Some(known) = ahead(&journal()) {
        return Some(known);
    }
    let mut guards_asked = false;
    // (View memory read-only by our hand: the kernel would refuse. With
    // renderer port #6 also view memory known as such, [`VIEWS`].)
    let protected = guarded && light() && PROTECTED.any(first, last);
    let viewed = guarded && !protected && known_views(first, last);
    if viewed {
        port6_count(P6_NOT_ASKED, 1);
    }
    let refused = protected || viewed;
    let pages = match if refused { None } else { kernel_written(first, last) } {
        Some(pages) => pages,
        None => {
            guards_asked = true;
            if !guarded {
                return None;
            }
            #[cfg(feature = "diagnostics")]
            let guarding = std::time::Instant::now();
            let mut written = guard(first * PAGE, last * PAGE).map(|pages| pages.into_iter().map(|p| (p, WHOLE)).collect::<Vec<_>>());
            if written.is_none() && viewed {
                // Not a view after all (other memory there now): forgotten as
                // one, and the kernel asked as before.
                (first..last).for_each(|block| VIEWS.clear(block));
                if let Some(pages) = kernel_written(first, last) {
                    guards_asked = false;
                    written = Some(pages);
                }
            }
            #[cfg(feature = "diagnostics")]
            {
                use crate::copystat::{C, add};
                let us = guarding.elapsed().as_micros() as u64;
                add(C::GuardUs, us);
                if written.is_some() {
                    add(C::Watches, 1);
                    add(C::WatchUs, us);
                    slow(us);
                }
            }
            written?
        }
    };
    if pages.is_empty() {
        // Nothing to note: the others may go on reading.
        return Some(fresh());
    }
    let answer = if port_journal() {
        // Renderer port #7: shared, the range's parts held for writing (the
        // range is held too: no other asking notes any of it meanwhile).
        let j = journal();
        let mut held = j.notes.hold(shards_of(first, last) | shards_of_pages(pages.iter().map(|p| p.0)), true);
        let now = next_asking();
        for &(page, parts) in &pages {
            held.note(page, parts, now);
        }
        port7_count(P7_SHARED_NOTES, 1);
        (now, held.range(first, last))
    } else {
        let mut j = journal_mut();
        let now = next_asking();
        for &(page, parts) in &pages {
            j.notes.note(page, parts, now);
        }
        (now, j.notes.range(first, last))
    };
    // Noted: what the guards made clean of it is known by its bits from
    // now on. (The range is still held: no other asking of any of it came
    // between.)
    if guards_asked && light() {
        publish(&pages.iter().map(|p| p.0).collect::<Vec<_>>());
    }
    Some(answer)
}

/// Whether nothing of `[start, end)` was written that a copy with the notes
/// `copied` (the asking after which each page from page `base` on was taken)
/// and `ranged` (the asking after which a range that holds this one was
/// taken whole, or 0) does not hold, as [`writes`] would answer with nothing
/// asked: the submission's asking covers the range, or its blocks are known
/// by their bits. `false` if something would have to be asked.
pub(crate) fn nothing_new(start: usize, end: usize, base: usize, copied: &[u64], ranged: u64) -> bool {
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    // The bits first, then the journal, as in `writes`.
    let known = published(start, end);
    #[cfg(feature = "diagnostics")]
    if known {
        crate::faultbook::note(crate::faultbook::What::Known, first, last, 1);
    }
    let j = journal();
    if !known && !j.covers(first, last) {
        return false;
    }
    let Some(taken) = first.checked_sub(base).and_then(|from| copied.get(from..last - base)) else {
        return false;
    };
    (ranged != 0 || taken.iter().all(|&at| at != 0))
        && j.notes.each(first, last, |page, write| write <= taken[page - first].max(ranged))
}

impl Tracked {
    /// Takes changed pages of `[from, to)`, cut to the tracked range. A
    /// partial page stays changed so a later take can read its other bytes.
    pub(crate) fn take(&self, from: usize, to: usize) -> Vec<(usize, usize)> {
        let (from, to) = (from.max(self.start), to.min(self.end));
        if from >= to {
            return Vec::new();
        }
        let Some((_, pages)) = writes(from, to, true) else {
            return Vec::new();
        };
        let mut seen = self.seen.lock().unwrap_or_else(|e| e.into_inner());
        let mut new: Vec<(usize, u64)> = Vec::new();
        for (page, write) in pages {
            let last = seen.get(&page).copied();
            if last.is_some_and(|last| last >= write) || (page * PAGE).max(from) >= ((page + 1) * PAGE).min(to) {
                continue;
            }
            // Keep partial pages dirty: the caller has not taken the rest.
            if (page * PAGE).max(self.start) >= from
                && ((page + 1) * PAGE).min(self.end) <= to
            {
                seen.insert(page, write);
            }
            new.push((page, last.unwrap_or(0)));
        }
        // Of a page taken before, the parts written since, where they are
        // known (`VICTORY_TAKE_PARTS=0`: the page).
        let parts = if crate::toggled!("VICTORY_TAKE_PARTS") { parts_written(&new) } else { vec![WHOLE; new.len()] };
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        for (&(page, last), &parts) in new.iter().zip(&parts) {
            let mut parts = if last == 0 { WHOLE } else { parts };
            while parts != 0 {
                let first = parts.trailing_zeros() as usize;
                let past = first + (!(parts >> first)).trailing_zeros() as usize;
                parts &= if past >= PARTS { 0 } else { u16::MAX << past };
                let (start, end) = ((page * PAGE + first * KERNEL_PAGE).max(from), (page * PAGE + past.min(PARTS) * KERNEL_PAGE).min(to));
                if start >= end {
                    continue;
                }
                match ranges.last_mut() {
                    Some((_, last)) if *last == start => *last = end,
                    _ => ranges.push((start, end)),
                }
            }
        }
        ranges
    }
}

/// Guarded blocks of view memory.
const BLOCK: usize = PAGE;

/// Whether the refreshes of host copies go the light way
/// (`VICTORY_LIGHT_REFRESH=0`: as before): view memory nothing wrote since
/// it was last asked about is known by its bits ([`PUBLISHED`]), a copy
/// nothing of whose range changed is told so before any thread is woken
/// for it (`crate::host`, `Shadow::nothing_new`), and the ends of a range,
/// pages the range holds in part, are not copied again at every refresh
/// (`Shadow::copy_stale`).
///
/// A frame asked about view memory some 1,500 times, for nearly all of it
/// to hear that nothing was written, on up to seven threads at once, each
/// asking under the guards' one lock and behind [`ASKING`]'s: 94 ms a frame
/// of all threads' time (run record-1), most of it waiting.
pub(crate) fn light() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LIGHT_REFRESH").is_none_or(|v| v != "0"))
}

/// Renderer port #6 (`VICTORY_PORT_GUARD=1`, off by default; read here, and
/// A/B-able in one run with `VICTORY_AB=<s>,VICTORY_PORT_GUARD`): the guards'
/// one lock no longer makes the title's write faults, and the refreshes'
/// looks at view memory, wait for one another.
///
/// What it was (run best-1, Series S, ports 1, 2 and 5 on): some 350 write
/// faults a frame, each taking [`GUARDS`] alone to give one block its write
/// access back, and some 220 refreshes a frame guarding view memory block by
/// block under the same lock (every `VirtualProtect` of theirs under it):
/// 450 waits for the lock, 58-62 ms a frame of all threads' time, 20 of it the
/// refreshes' (2.5-4.5 ms of the submitting thread's), the rest the title's
/// threads, mostly queued behind one another.
///
/// What it is with the switch: [`GUARDS`] is a lock for readers and one
/// writer. The sets of blocks in it are of atomic words, so what changes a
/// block's state needs no lock of all of them: a block's own claim does
/// ([`BUSY`]). A fault, a refresh guarding its range and a block made known
/// by its bits ([`publish`]) take the lock shared and the claims of the
/// blocks they change, so they wait only for those who change the same
/// block (a `VirtualProtect` of it) and for the few things that still take
/// the lock alone (a mapping changed, a kernel read, memory freed, the holes
/// looked at, ...), which take no claims: holding the lock alone, nobody
/// holds one. Who waits for a claim holds no other claim, or only claims of
/// lower blocks (faults take theirs in order); who holds claims and wants
/// more only tries for them (a refresh: what it cannot claim it leaves as it
/// is, written, told so at the next asking), so no two wait for each other.
///
/// Also with the switch:
/// - view memory is not asked of the kernel, which refuses it
///   ([`VIEWS`]; ~300 refused askings a frame);
/// - the planning of a submission's refreshes is made beside the submitting
///   thread ([`port_guard_plan`]);
/// - only if asked for (`VICTORY_GUARD_HOT=<ms>`, [`hot_rest`]): a block the
///   title writes again soon after each time it was taken rests longer.
///   Measured worse (run guard-6: 45.6 -> 47.8 ms a frame): off.
pub(crate) fn port_guard() -> bool {
    #[cfg(feature = "diagnostics")]
    if let Some(on) = crate::ab::pick("VICTORY_PORT_GUARD") {
        return on;
    }
    #[cfg(test)]
    if PORT_GUARD_IN_TEST.get() {
        return true;
    }
    // Read here: the switch of renderer port #6.
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_PORT_GUARD").is_ok_and(|v| v.trim() == "1"))
}

/// With renderer port #6: whether a submission's look at which refreshes
/// copy nothing (`Shadow::nothing_new`, before the threads are woken for the
/// others) is made on the threads beside the submitting one, as it already
/// was for more than 256 refreshes (`VICTORY_PORT_GUARD_PLAN=0`: on the
/// submitting thread, as before; A/B-able by that name).
///
/// A look walks the journal's every page ever written in the range; one
/// submission a frame spent up to 6 ms in it alone, the six threads beside it
/// idle (runs lane-1, guard-3).
pub(crate) fn port_guard_plan() -> bool {
    if !port_guard() {
        return false;
    }
    #[cfg(feature = "diagnostics")]
    if let Some(on) = crate::ab::pick("VICTORY_PORT_GUARD_PLAN") {
        return on;
    }
    // Read here: renderer port #6's planning beside the submitting thread.
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PORT_GUARD_PLAN").is_none_or(|v| v != "0"))
}

#[cfg(test)]
thread_local! {
    /// [`hot_rest`] in milliseconds, for what this thread does (0: as set).
    pub(crate) static HOT_IN_TEST: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    /// [`port_guard`], for what this thread does.
    pub(crate) static PORT_GUARD_IN_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The blocks a thread holding [`GUARDS`] shared is changing the state of
/// (renderer port #6, [`port_guard`]): a bit each, set by who changes it
/// ([`Claims`]).
static BUSY: Bits = Bits::new();

/// Claims of blocks ([`BUSY`]), let go when dropped.
struct Claims(Vec<usize>);

impl Claims {
    /// The claims of `blocks` (in order, each once), waited for one after
    /// the other: who waits holds only claims of lower blocks.
    fn wait_for(blocks: &[usize]) -> Claims {
        debug_assert!(blocks.windows(2).all(|w| w[0] < w[1]));
        let mut spun = 0u64;
        for &block in blocks {
            let mut tries = 0u32;
            while !BUSY.try_set(block) {
                tries += 1;
                spun += 1;
                if tries < 64 {
                    std::hint::spin_loop();
                } else {
                    // Another thread's `VirtualProtect` of the block.
                    std::thread::yield_now();
                }
            }
        }
        if spun != 0 {
            port6_count(P6_CLAIM_SPINS, spun);
        }
        Claims(blocks.to_vec())
    }

    /// The claim of `block` if it is free; nothing waited for.
    fn try_one(&mut self, block: usize) -> bool {
        let got = BUSY.try_set(block);
        if got {
            self.0.push(block);
        } else {
            port6_count(P6_CLAIMS_MISSED, 1);
        }
        got
    }

    fn none() -> Claims {
        Claims(Vec::new())
    }

    fn holds(&self, block: usize) -> bool {
        self.0.contains(&block)
    }
}

impl Drop for Claims {
    fn drop(&mut self) {
        for &block in &self.0 {
            BUSY.clear(block);
        }
    }
}

/// Renderer port #6's counts since the last line: faults opened under the
/// shared lock; nanoseconds they waited for the lock; refreshes' guards
/// done shared; blocks a refresh could not claim (left written); spins for
/// claims; blocks rested longer for being hot; kernel askings left out for
/// view memory; blocks found hot.
static PORT6: [std::sync::atomic::AtomicU64; 8] = [const { std::sync::atomic::AtomicU64::new(0) }; 8];
const P6_FAULTS: usize = 0;
const P6_FAULT_WAIT_NS: usize = 1;
const P6_GUARDS: usize = 2;
const P6_CLAIMS_MISSED: usize = 3;
const P6_CLAIM_SPINS: usize = 4;
const P6_RESTED_HOT: usize = 5;
const P6_NOT_ASKED: usize = 6;
const P6_HOT: usize = 7;

fn port6_count(what: usize, n: u64) {
    PORT6[what].fetch_add(n, std::sync::atomic::Ordering::Relaxed);
}

/// Renderer port #6's line of the frame lines, per frame.
pub(crate) fn port_guard_line(frames: f64) {
    let n: Vec<f64> = PORT6.iter().map(|c| c.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / frames).collect();
    if !port_guard() && n.iter().all(|&v| v == 0.0) {
        return;
    }
    tracing::info!(
        "renderer port 6 (VICTORY_PORT_GUARD=1) per frame: faults opened under the shared lock {:.1} ({:.2} ms waiting for it), \
         refreshes' guards done shared {:.1}, blocks left written for want of their claim {:.1}, spins for claims {:.0}; \
         blocks found hot {:.1}, rests longer for it {:.1} (VICTORY_GUARD_HOT); kernel askings left out for view memory {:.1}",
        n[P6_FAULTS],
        n[P6_FAULT_WAIT_NS] / 1e6,
        n[P6_GUARDS],
        n[P6_CLAIMS_MISSED],
        n[P6_CLAIM_SPINS],
        n[P6_HOT],
        n[P6_RESTED_HOT],
        n[P6_NOT_ASKED],
    );
}

/// The blocks known to be view memory (renderer port #6): guarded by us
/// once, or found a view by [`guard`]. The kernel refuses to ask about them
/// (its write watch is not kept for views), and with the switch such a
/// range goes to the guards at once. Forgotten when the memory is unmapped
/// or freed; a block wrongly here is only looked at by the guards, which
/// find it no view ([`is_view`]) and say the range is not watched.
static VIEWS: Bits = Bits::new();

/// Whether a block of `[first, last)` is known view memory and the kernel
/// need not be asked ([`VIEWS`]).
fn known_views(first: usize, last: usize) -> bool {
    port_guard() && light() && VIEWS.any(first, last)
}

/// A bit per block of the address space, in pieces made when first needed
/// and never let go.
struct Bits([std::sync::atomic::AtomicPtr<BitsPiece>; Bits::PIECES]);

/// The bits of 4 GiB.
type BitsPiece = [std::sync::atomic::AtomicU64; Bits::WORDS];

impl Bits {
    const WORDS: usize = 1 << 10;
    /// 128 TiB, what a process has.
    const PIECES: usize = 1 << 15;

    const fn new() -> Bits {
        Bits([const { std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()) }; Bits::PIECES])
    }

    fn piece(&self, word: usize) -> Option<&BitsPiece> {
        let piece = self.0.get(word / Bits::WORDS)?.load(std::sync::atomic::Ordering::Acquire);
        // SAFETY: null, or a piece made by `set`, which is never freed.
        unsafe { piece.as_ref() }
    }

    fn set(&self, block: usize) {
        use std::sync::atomic::Ordering::{AcqRel, Acquire, SeqCst};
        let word = block / 64;
        let Some(place) = self.0.get(word / Bits::WORDS) else {
            return;
        };
        if place.load(Acquire).is_null() {
            let piece: Box<BitsPiece> = Box::new([const { std::sync::atomic::AtomicU64::new(0) }; Bits::WORDS]);
            let piece = Box::into_raw(piece);
            if place.compare_exchange(std::ptr::null_mut(), piece, AcqRel, Acquire).is_err() {
                // SAFETY: made just now, and another thread's is there.
                drop(unsafe { Box::from_raw(piece) });
            }
        }
        if let Some(piece) = self.piece(word) {
            piece[word % Bits::WORDS].fetch_or(1 << (block % 64), SeqCst);
        }
    }

    /// Sets it if it was not set: whether it was not.
    fn try_set(&self, block: usize) -> bool {
        self.set_was(block) == Some(false)
    }

    /// Sets it; whether it was set before (`None`: out of the address space).
    fn set_was(&self, block: usize) -> Option<bool> {
        use std::sync::atomic::Ordering::{AcqRel, Acquire, SeqCst};
        let word = block / 64;
        let place = self.0.get(word / Bits::WORDS)?;
        if place.load(Acquire).is_null() {
            let piece: Box<BitsPiece> = Box::new([const { std::sync::atomic::AtomicU64::new(0) }; Bits::WORDS]);
            let piece = Box::into_raw(piece);
            if place.compare_exchange(std::ptr::null_mut(), piece, AcqRel, Acquire).is_err() {
                // SAFETY: made just now, and another thread's is there.
                drop(unsafe { Box::from_raw(piece) });
            }
        }
        let bit = 1 << (block % 64);
        self.piece(word).map(|piece| piece[word % Bits::WORDS].fetch_or(bit, SeqCst) & bit != 0)
    }

    /// Whether it was set; it is not now.
    fn clear_was(&self, block: usize) -> bool {
        let word = block / 64;
        let bit = 1 << (block % 64);
        self.piece(word)
            .is_some_and(|piece| piece[word % Bits::WORDS].fetch_and(!bit, std::sync::atomic::Ordering::SeqCst) & bit != 0)
    }

    /// Those set of `[first, last)`, in order (`last` at most the address
    /// space's end).
    fn set_in(&self, first: usize, last: usize) -> impl Iterator<Item = usize> + '_ {
        let last = last.min(Bits::PIECES * Bits::WORDS * 64).max(first);
        let mut word = first / 64;
        let end = last.div_ceil(64);
        let mut bits = 0u64;
        std::iter::from_fn(move || {
            loop {
                if bits != 0 {
                    let bit = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    return Some((word - 1) * 64 + bit);
                }
                if word >= end {
                    return None;
                }
                match self.piece(word) {
                    Some(piece) => {
                        bits = piece[word % Bits::WORDS].load(std::sync::atomic::Ordering::SeqCst) & in_word(word * 64, first, last);
                        word += 1;
                    }
                    None => word = (word / Bits::WORDS + 1) * Bits::WORDS,
                }
            }
        })
    }

    /// How many are set.
    fn count(&self) -> usize {
        self.0
            .iter()
            .filter_map(|p| {
                // SAFETY: null, or a piece made by `set`, which is never freed.
                unsafe { p.load(std::sync::atomic::Ordering::Acquire).as_ref() }
            })
            .map(|piece| piece.iter().map(|w| w.load(std::sync::atomic::Ordering::SeqCst).count_ones() as usize).sum::<usize>())
            .sum()
    }

    /// Sixty-four of them.
    fn word(&self, word: usize) -> u64 {
        self.piece(word).map_or(0, |piece| piece[word % Bits::WORDS].load(std::sync::atomic::Ordering::SeqCst))
    }

    /// Whether it was set, and is no more.
    fn take(&self, block: usize) -> bool {
        self.any(block, block + 1) && {
            let (word, bit) = (block / 64, 1 << (block % 64));
            self.piece(word)
                .is_some_and(|piece| piece[word % Bits::WORDS].fetch_and(!bit, std::sync::atomic::Ordering::SeqCst) & bit != 0)
        }
    }

    fn clear(&self, block: usize) {
        let word = block / 64;
        if let Some(piece) = self.piece(word) {
            // Ahead of the protection's change, and seen by whoever the
            // writer tells of its write.
            piece[word % Bits::WORDS].fetch_and(!(1 << (block % 64)), std::sync::atomic::Ordering::SeqCst);
        }
    }

    /// Whether a bit of `[first, last)` is set.
    fn any(&self, first: usize, last: usize) -> bool {
        let mut block = first;
        while block < last {
            let word = block / 64;
            let until = ((word + 1) * 64).min(last);
            let mask = (u64::MAX >> (64 - (until - block))) << (block % 64);
            match self.piece(word) {
                Some(piece) => {
                    if piece[word % Bits::WORDS].load(std::sync::atomic::Ordering::SeqCst) & mask != 0 {
                        return true;
                    }
                    block = until;
                }
                // None of the piece's.
                None => block = (word / Bits::WORDS + 1) * Bits::WORDS * 64,
            }
        }
        false
    }

    /// Whether the bits of all of `[first, last)` are set (not so of none).
    fn all(&self, first: usize, last: usize) -> bool {
        let mut block = first;
        while block < last {
            let word = block / 64;
            let until = ((word + 1) * 64).min(last);
            let mask = (u64::MAX >> (64 - (until - block))) << (block % 64);
            let Some(piece) = self.piece(word) else {
                return false;
            };
            if piece[word % Bits::WORDS].load(std::sync::atomic::Ordering::Acquire) & mask != mask {
                return false;
            }
            block = until;
        }
        first < last
    }
}

/// The blocks that are read-only by our hand, not written since, and of
/// which the journal holds every write from before that: for those, what
/// [`writes`] answers is in the journal, and neither the kernel, nor the
/// guards' lock, nor [`ASKING`] need be asked.
///
/// Why that is so: a block's bit is set under the guards' lock while the
/// block is clean, by the asking that made it clean, after that asking noted
/// in the journal what it found ([`publish`]). It is cleared under the same
/// lock, before the block becomes writable again ([`open`]) or is forgotten
/// ([`title_memory_remapped`]). So while the bit is set nothing can have
/// written the block through this address, and who sees the bit and looks
/// at the journal after that finds every earlier write in it. A write that
/// comes while the bit is looked at is no different from one that comes just
/// after an asking: the next asking finds the block open.
///
/// So too for the blocks of placeholders that were copied as zeros
/// (`Guards::held`): nothing can be there before the runtime maps it, and
/// it tells of that ([`title_memory_remapped`]) before the title is.
static PUBLISHED: Bits = Bits::new();

/// The blocks read-only by our hand, or about to be: set under the guards'
/// lock ahead of the protection, cleared under it when the block is open
/// again. Who finds none set has no guard to open and need not wait for the
/// lock ([`writing`]): some 700 pieces a frame are landed in memory that is
/// no view.
static PROTECTED: Bits = Bits::new();

/// The blocks the title's bytes were put in beside their guards
/// ([`put_beside`]) and that no asking has told as written since. Such a
/// block is as it was else: read-only, clean, and an asking that finds its
/// bit here takes the bit and tells it as written.
///
/// The bit is set when the bytes are in, at every view of the page, and
/// [`PUBLISHED`]'s is cleared after that. So an asking either finds the bit
/// or came before the title's thread went on from its write; and who takes
/// the bit reads the bytes after.
static BESIDE: Bits = Bits::new();

/// How much was put into a block beside its guard since it was last told as
/// written (`crate::titlewrite::MOST`), by a few bits of its number.
static PUT: [std::sync::atomic::AtomicU32; 1 << 13] = [const { std::sync::atomic::AtomicU32::new(0) }; 1 << 13];

fn put_count(block: usize) -> &'static std::sync::atomic::AtomicU32 {
    &PUT[block % PUT.len()]
}

/// Whether [`put_beside`] put it there.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Beside {
    Done,
    /// The block has had its share.
    Full,
    /// Not guarded, or there is no other way to its page, or the runtime is
    /// changing a mapping.
    Cannot,
}

/// Whether a block of `[at, at + len)` is (or is about to be) read-only by
/// our hand. `len` is not 0.
#[inline]
pub(crate) fn guards_any(at: usize, len: usize) -> bool {
    PROTECTED.any(at / BLOCK, (at.saturating_add(len) - 1) / BLOCK + 1)
}

/// The title's bytes for a guarded block, put in by `put` at our own view of
/// the same page: it is given what to add to an address of the block (as by
/// `wrapping_add`). `weight`: how much of the block's share it is.
pub(crate) fn put_beside(block: usize, weight: u32, put: impl FnOnce(usize)) -> Beside {
    use std::sync::atomic::Ordering::Relaxed;
    if !light() || !view_guard() || !PROTECTED.any(block, block + 1) {
        return Beside::Cannot;
    }
    let count = put_count(block);
    if count.load(Relaxed) + weight > crate::titlewrite::MOST {
        return Beside::Full;
    }
    // No mapping changes until all is told: a view of the page that came
    // after the look at them would not be.
    let Some(_reading) = title_memory_reading_now() else {
        return Beside::Cannot;
    };
    let Some((page, shown)) = crate::inplace::beside_page(block * BLOCK) else {
        return Beside::Cannot;
    };
    put(page.wrapping_sub(block * BLOCK));
    count.fetch_add(weight, Relaxed);
    for view in shown.iter().filter(|&&view| view != 0) {
        BESIDE.set(view / BLOCK);
    }
    for view in shown.iter().filter(|&&view| view != 0) {
        PUBLISHED.clear(view / BLOCK);
    }
    Beside::Done
}

/// A faulting store of the title's leaves `block` as it is: whether it may be
/// left out (until that has been so often, since the block was taken, that
/// the faults cost more than opening it).
pub(crate) fn nothing_put(block: usize) -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    let count = put_count(block);
    count.load(Relaxed) < crate::titlewrite::MOST && count.fetch_add(1, Relaxed) < crate::titlewrite::MOST
}

/// The title is about to write block by block around `at`: as by a fault.
pub(crate) fn opened_for_writing(at: usize) {
    open_faulted(at);
}

/// Opens the guarded block at `at` for a write of the title's that faulted
/// there (or is about to be made), and notes it opened: how many blocks were
/// opened (0: it was not guarded, or another thread opened it first), and how
/// long the guards' lock was waited for.
///
/// With renderer port #6 the lock is taken shared and the claims of the block
/// and of its page's other views are waited for, in order: the block's state
/// is changed by nobody else meanwhile, and other faults go on beside this.
fn open_faulted(at: usize) -> (usize, std::time::Duration) {
    let block = at / BLOCK;
    let began = std::time::Instant::now();
    if !port_guard() {
        let g = guards();
        let waited = began.elapsed();
        let opened = open(&g, at, at + 1);
        if opened != 0 {
            g.opened(block);
        }
        return (opened, waited);
    }
    let g = guards_shared();
    let waited = began.elapsed();
    port6_count(P6_FAULT_WAIT_NS, waited.as_nanos() as u64);
    let mut views: Vec<usize> = vec![block];
    loop {
        let _claims = Claims::wait_for(&views);
        if !g.protected.contains(block) {
            return (0, waited);
        }
        let given = if g.aliased.contains(block) {
            // (The views as the table has them now, as `open` looks them up;
            // a view mapped after this opens what is guarded of the page
            // again, holding the lock alone: `views_mapped`.)
            let mut now: Vec<usize> = crate::inplace::pages_of(block * BLOCK).into_iter().map(|p| p / BLOCK).collect();
            now.push(block);
            now.sort_unstable();
            now.dedup();
            if now.iter().any(|b| views.binary_search(b).is_err()) {
                // Their claims too, in order: again.
                views = now;
                continue;
            }
            Some(now)
        } else {
            None
        };
        let opened = open_given(&g, at, at + 1, given.as_deref());
        if opened != 0 {
            g.opened(block);
            port6_count(P6_FAULTS, 1);
        }
        return (opened, waited);
    }
}

/// An asking made blocks of `pages` clean and noted them in the journal.
fn publish(pages: &[usize]) {
    if port_guard() {
        // A block's claim while its bit is set: a fault opening it meanwhile
        // clears the bit after this sets it, never before.
        let g = guards_shared();
        for &page in pages {
            let _claim = Claims::wait_for(&[page]);
            if g.clean.contains(page) || g.held.contains(page) {
                PUBLISHED.set(page);
                if BESIDE.any(page, page + 1) {
                    PUBLISHED.clear(page);
                }
            }
        }
        return;
    }
    let g = guards();
    for &page in pages {
        if g.clean.contains(page) || g.held.contains(page) {
            PUBLISHED.set(page);
            // (Put there since: whoever did clears the bit after this is
            // set, or its own bit is found here.)
            if BESIDE.any(page, page + 1) {
                PUBLISHED.clear(page);
            }
        }
    }
}

/// Whether placeholders are known for what they are (`VICTORY_KNOWN_HOLES=0`:
/// every hole is looked at at every asking, as before).
///
/// Most ranges asked about have holes, reservations the title maps its pages
/// into as it streams (231,000 regions, 1,125 of a frame's 1,530 asked view
/// ranges, run next1-1). Such a range was never known whole: each asking put
/// a question to Windows for each of its holes, under the guards' lock, and
/// 111 ms a frame of all threads' time went into waiting for that lock.
fn known_holes() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_KNOWN_HOLES").is_none_or(|v| v != "0") && light())
}

/// Whether `block` lies in a placeholder: a reservation in which pages come
/// to be only by mapping a view (or an allocation) in its place, which an
/// ordinary commit cannot do.
fn placeholder(block: usize) -> bool {
    /// `MEMORY_REGION_INFORMATION` (ntddk).
    #[repr(C)]
    #[derive(Default)]
    struct Region {
        allocation_base: usize,
        allocation_protect: u32,
        region_type: u32,
        region_size: usize,
        commit_size: usize,
        partition_id: usize,
        node_preference: usize,
    }
    #[link(name = "ntdll", kind = "raw-dylib")]
    unsafe extern "system" {
        fn NtQueryVirtualMemory(
            process: *mut std::ffi::c_void,
            at: *const std::ffi::c_void,
            class: u32,
            out: *mut std::ffi::c_void,
            len: usize,
            returned: *mut usize,
        ) -> i32;
    }
    // (Class 3 answers with the kind of memory in that word, not the flags.)
    const MEMORY_REGION_INFORMATION_EX: u32 = 7;
    const PLACEHOLDER_RESERVATION: u32 = 1 << 8;
    let mut region = Region::default();
    // SAFETY: the answer's room and its size.
    let status = unsafe {
        NtQueryVirtualMemory(
            windows_sys::Win32::System::Threading::GetCurrentProcess(),
            (block * BLOCK) as *const _,
            MEMORY_REGION_INFORMATION_EX,
            (&raw mut region).cast(),
            size_of::<Region>(),
            std::ptr::null_mut(),
        )
    };
    status >= 0 && region.region_type & PLACEHOLDER_RESERVATION != 0
}

/// The runtime gave the title memory at `[at, at + len)`, allocated or
/// committed: what was known of holes there is asked again.
pub fn title_memory_allocated(at: usize, len: usize) {
    if len == 0 || !known_holes() {
        return;
    }
    let g = guards();
    let blocks: Vec<usize> = g.held.range(at / BLOCK..at.saturating_add(len).div_ceil(BLOCK)).collect();
    for block in blocks {
        g.held.remove(block);
        PUBLISHED.clear(block);
    }
}

/// Once a frame: a hundredth of the known holes are looked at, in case one
/// was filled with nobody telling ([`title_memory_allocated`],
/// [`title_memory_remapped`]), which no title we know does.
pub(crate) fn look_at_holes() {
    const PARTS: usize = 100;
    if !known_holes() {
        return;
    }
    static FROM: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let blocks: Vec<usize> = {
        let g = guards();
        let from = FROM.load(std::sync::atomic::Ordering::Relaxed);
        let part: Vec<usize> = g.held.range(from..).take(g.held.len() / PARTS + 1).collect();
        FROM.store(part.last().map_or(0, |b| b + 1), std::sync::atomic::Ordering::Relaxed);
        part
    };
    let mut until = 0;
    for block in blocks {
        if block >= until {
            until = uncommitted_end(block);
        }
        if block >= until {
            // (Or unmapped memory was mapped since the part was taken: told
            // already, and forgotten.)
            let g = guards();
            if g.held.remove(block) {
                PUBLISHED.clear(block);
                static SAID: std::sync::Once = std::sync::Once::new();
                SAID.call_once(|| {
                    tracing::warn!(
                        address = format_args!("{:#x}", block * BLOCK),
                        "a placeholder of the title's was filled with nobody telling (VICTORY_KNOWN_HOLES=0 asks every time)"
                    );
                });
            }
        }
    }
}

/// Whether a block of `[start, end)` is view memory read-only by our hand:
/// the kernel keeps no written pages of it and would refuse an asking.
pub(crate) fn guarded(start: usize, end: usize) -> bool {
    PROTECTED.any(start / BLOCK, end.div_ceil(BLOCK))
}

/// Whether nothing of `[start, end)` can have been written since an asking
/// noted it in the journal ([`PUBLISHED`]).
fn published(start: usize, end: usize) -> bool {
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    // (In this order: who puts something in beside a guard sets the one bit,
    // then clears the other, which an asking that is publishing the block may
    // just be setting.)
    light() && PUBLISHED.all(first, last) && !BESIDE.any(first, last)
}

/// Since the last frame line: what asking about title memory was. Kernel
/// askings and the KiB they walked; view ranges answered by their bits, by
/// the guards whole clean, and block by block; blocks made read-only; blocks
/// opened by ourselves; nanoseconds waiting for the guards' lock and
/// nanoseconds of changing protections (all threads); questions to Windows
/// about holes; kernel askings refused; nanoseconds in kernel askings; the
/// 4 KiB pages they found written.
#[cfg(feature = "diagnostics")]
static ASKS: [std::sync::atomic::AtomicU64; 13] = [const { std::sync::atomic::AtomicU64::new(0) }; 13];

#[cfg(feature = "diagnostics")]
fn tally(what: usize, n: u64) {
    ASKS[what].fetch_add(n, std::sync::atomic::Ordering::Relaxed);
}

/// [`ASKS`], taken.
#[cfg(feature = "diagnostics")]
pub(crate) fn asks() -> [u64; 13] {
    std::array::from_fn(|i| ASKS[i].swap(0, std::sync::atomic::Ordering::Relaxed))
}

/// The guards' lock, the wait for it counted.
fn guards_timed() -> std::sync::RwLockWriteGuard<'static, Guards> {
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    let g = guards();
    #[cfg(feature = "diagnostics")]
    tally(7, started.elapsed().as_nanos() as u64);
    g
}

/// [`guards_shared`], the wait for it counted.
fn guards_timed_shared() -> std::sync::RwLockReadGuard<'static, Guards> {
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    let g = guards_shared();
    #[cfg(feature = "diagnostics")]
    tally(7, started.elapsed().as_nanos() as u64);
    g
}

/// A set of blocks, a bit each: a fault, or a look at a range of thousands of
/// blocks, is a few words of it and no walk through a tree of every guarded
/// block there is. Of atomic words (renderer port #6): a thread holding
/// [`GUARDS`] shared changes the bits of the blocks it holds the claims of
/// ([`BUSY`]), while others read and change other blocks' bits of the same
/// words. Who holds the lock alone sees and changes them as before.
struct Blocks(Bits);

impl Blocks {
    /// The blocks of 4 GiB (a piece of [`Bits`]).
    #[cfg(test)]
    const IN_PIECE: usize = Bits::WORDS * 64;

    const fn new() -> Blocks {
        Blocks(Bits::new())
    }

    fn word(&self, word: usize) -> u64 {
        self.0.word(word)
    }

    fn contains(&self, block: usize) -> bool {
        self.word(block / 64) >> (block % 64) & 1 != 0
    }

    /// Whether it was not there.
    fn insert(&self, block: usize) -> bool {
        self.0.set_was(block) == Some(false)
    }

    /// Whether it was there.
    fn remove(&self, block: usize) -> bool {
        self.0.clear_was(block)
    }

    fn insert_all(&self, blocks: std::ops::Range<usize>) {
        for block in blocks {
            self.insert(block);
        }
    }

    /// Those of `range`, in order.
    fn range(&self, range: impl RangeBounds<usize>) -> impl Iterator<Item = usize> + '_ {
        let first = match range.start_bound() {
            Bound::Included(&first) => first,
            Bound::Excluded(&before) => before.saturating_add(1),
            Bound::Unbounded => 0,
        };
        // (Exclusive. No block's number is near the top.)
        let last = match range.end_bound() {
            Bound::Included(&last) => last.saturating_add(1),
            Bound::Excluded(&last) => last,
            Bound::Unbounded => usize::MAX,
        }
        .max(first);
        self.0.set_in(first, last)
    }

    fn len(&self) -> usize {
        self.0.count()
    }
}

/// Of the 64 blocks from `at` on, those in `[from, to)`, a bit each.
fn in_word(at: usize, from: usize, to: usize) -> u64 {
    let low = from.saturating_sub(at).min(64);
    let high = to.saturating_sub(at).min(64);
    if low >= high { 0 } else { (u64::MAX >> (64 - (high - low))) << low }
}

/// The bits of a word that are set, by number.
struct BitsOf(u64);

impl Iterator for BitsOf {
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        let bit = (self.0 != 0).then(|| self.0.trailing_zeros() as usize)?;
        self.0 &= self.0 - 1;
        Some(bit)
    }
}

/// Whether a write fault opens the view that was written and leaves the
/// page's other views read-only, not clean any more
/// (`VICTORY_OPEN_ONE_VIEW=0`: opens them all, as before).
///
/// The others were opened (a change of protection each, under the guards'
/// lock, with other faults waiting) and made read-only again when the page
/// was next taken, though the title writes through one of them: 586 faults a
/// frame opened 880 blocks (run faults-1). A write through one left as it was
/// faults, and opens it.
fn open_one_view() -> bool {
    crate::toggled!("VICTORY_OPEN_ONE_VIEW")
}

/// How long a block the title has just opened by a write is left open by
/// those that take it (`VICTORY_GUARD_REST=<microseconds>`, 3,000 unless
/// given; `0`: not at all, as before).
///
/// 220 of a frame's 586 faults came within 3 ms after a refresh had taken the
/// block and made it read-only, while the title was still writing it, and
/// what a refresh takes is asked about again by the next within a millisecond
/// (run faults-1): made read-only, written, opened, up to a dozen times a
/// frame. A block that rests counts as written at each asking, so nothing is
/// missed; it is read-only at the first asking after its rest, which the
/// title's later writes are seen by.
fn rest() -> std::time::Duration {
    #[cfg(test)]
    {
        REST_IN_TEST.get()
    }
    #[cfg(not(test))]
    {
        static US: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
        if !crate::toggled!("VICTORY_GUARD_REST") {
            return std::time::Duration::ZERO;
        }
        let us = *US.get_or_init(|| {
            std::env::var("VICTORY_GUARD_REST").ok().and_then(|v| v.trim().parse().ok()).filter(|us| *us > 1).unwrap_or(3000)
        });
        std::time::Duration::from_micros(us)
    }
}

#[cfg(test)]
thread_local! {
    /// [`rest`], for what this thread asks.
    static REST_IN_TEST: std::cell::Cell<std::time::Duration> = const { std::cell::Cell::new(std::time::Duration::ZERO) };
}

struct Guards {
    /// Blocks we made read-only.
    protected: Blocks,
    /// Of those, the ones not written since a copy took them.
    clean: Blocks,
    /// Blocks already copied as zeros. While they remain wholly uncommitted,
    /// the copy can skip hundreds of MB of failing kernel reads per world frame.
    absent: Blocks,
    /// Of those, the ones in placeholders ([`known_holes`]).
    held: Blocks,
    /// Blocks that were absent and turned out to be other memory than views
    /// when something was there again: never guarded, copied every time.
    foreign: BTreeSet<usize>,
    /// Blocks (first, past the last) the kernel writes now
    /// ([`kernel_writing`]), and blocks the runtime has unmapped in a change
    /// not yet done ([`title_memory_unmapping`]). Not in the sets above, and
    /// left as they are by [`guard`], which reports them written.
    kernel: Vec<(usize, usize)>,
    changing: Vec<(usize, usize)>,
    /// Blocks whose page is shown at other addresses too and that were made
    /// read-only together with those (which are in it as well): what opens
    /// one opens them all. (A write's range can be huge, `fread`'s, so only
    /// these are looked up.)
    aliased: Blocks,
    /// Blocks opened by a write fault, and when ([`rest`]); and how hot each
    /// is ([`hot_rest`]). Behind a lock of its own: faults note in it holding
    /// [`GUARDS`] shared.
    opened: Mutex<std::collections::HashMap<usize, Opened, std::hash::BuildHasherDefault<PageHash>>>,
}

/// When a write fault opened a block, and how often it had been written again
/// soon after it was taken ([`hot_rest`]).
#[derive(Clone, Copy)]
struct Opened {
    at: std::time::Instant,
    hot: u8,
}

/// How long a block the title writes again soon after each time it is taken
/// rests at most (renderer port #6, `VICTORY_GUARD_HOT=<ms>`; unset or 0:
/// off, [`rest`] for all). A block opened by a fault within its rest and
/// 50 ms more after it last was rests twice as long the next time, up to
/// this; one that stayed clean longer rests [`rest`] again. Such a block is
/// told as written at every asking while it rests (as a resting block always
/// was), so what is missed is nothing; what it costs is a look at its bytes
/// at every refresh of a copy over it instead of a protection change and a
/// fault each time.
fn hot_rest() -> std::time::Duration {
    #[cfg(test)]
    if HOT_IN_TEST.get() != 0 {
        return std::time::Duration::from_millis(HOT_IN_TEST.get());
    }
    #[cfg(feature = "diagnostics")]
    if crate::ab::pick("VICTORY_GUARD_HOT") == Some(false) {
        return std::time::Duration::ZERO;
    }
    static MS: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let ms = *MS.get_or_init(|| {
        std::env::var("VICTORY_GUARD_HOT").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(0).min(1000)
    });
    if ms == 0 || !port_guard() {
        return std::time::Duration::ZERO;
    }
    std::time::Duration::from_millis(ms)
}

impl Opened {
    /// How long a block opened so rests.
    fn rest(&self, rest: std::time::Duration) -> std::time::Duration {
        if self.hot == 0 || rest.is_zero() {
            return rest;
        }
        let most = hot_rest();
        if most.is_zero() {
            return rest;
        }
        rest.saturating_mul(1 << self.hot.min(16)).min(most.max(rest))
    }
}

impl Guards {
    fn left_alone(&self, block: usize) -> bool {
        self.kernel.iter().chain(&self.changing).any(|&(first, last)| (first..last).contains(&block))
    }

    /// Whether `block` is left open for now: the title has just written it.
    fn rests(&self, block: usize, rest: std::time::Duration) -> bool {
        if rest.is_zero() {
            return false;
        }
        let opened = self.opened.lock().unwrap_or_else(|e| e.into_inner());
        opened.get(&block).is_some_and(|o| {
            let elapsed = o.at.elapsed();
            let longer = o.rest(rest);
            if elapsed >= rest && elapsed < longer {
                port6_count(P6_RESTED_HOT, 1);
            }
            elapsed < longer
        })
    }

    /// A write fault opened `block`.
    fn opened(&self, block: usize) {
        const MANY: usize = 1 << 13;
        let rest = rest();
        let mut opened = self.opened.lock().unwrap_or_else(|e| e.into_inner());
        if opened.len() >= MANY {
            let keep = std::time::Duration::from_millis(50).max(hot_rest() + std::time::Duration::from_millis(50));
            opened.retain(|_, o| o.at.elapsed() < keep);
            if opened.len() >= MANY {
                opened.clear();
            }
        }
        put_count(block).store(0, std::sync::atomic::Ordering::Relaxed);
        let now = std::time::Instant::now();
        let hot = match opened.get(&block) {
            // Written again within its rest and a little more: hotter.
            Some(o) if !hot_rest().is_zero() && now.duration_since(o.at) < o.rest(rest) + std::time::Duration::from_millis(50) => {
                port6_count(P6_HOT, 1);
                o.hot.saturating_add(1).min(16)
            }
            _ => 0,
        };
        opened.insert(block, Opened { at: now, hot });
    }
}

/// Read by many threads at once with renderer port #6 ([`port_guard`]), held
/// alone otherwise and by what changes more than single blocks' states.
static GUARDS: RwLock<Guards> = RwLock::new(Guards {
    protected: Blocks::new(),
    clean: Blocks::new(),
    absent: Blocks::new(),
    held: Blocks::new(),
    foreign: BTreeSet::new(),
    kernel: Vec::new(),
    changing: Vec::new(),
    aliased: Blocks::new(),
    opened: Mutex::new(std::collections::HashMap::with_hasher(std::hash::BuildHasherDefault::new())),
});

/// The guards, alone.
fn guards() -> std::sync::RwLockWriteGuard<'static, Guards> {
    if let Ok(g) = GUARDS.try_write() {
        return g;
    }
    waited(3, || GUARDS.write().unwrap_or_else(|e| e.into_inner()))
}

/// The guards, shared (renderer port #6): what changes a block's state holds
/// its claim ([`Claims`]).
fn guards_shared() -> std::sync::RwLockReadGuard<'static, Guards> {
    if let Ok(g) = GUARDS.try_read() {
        return g;
    }
    waited(3, || GUARDS.read().unwrap_or_else(|e| e.into_inner()))
}

/// The 4 KiB pages of title memory that what shaders wrote was put into,
/// and the frame it last was: in pieces made when first needed and never let
/// go. Some 20,000 pages a frame are noted by the relay and the landing
/// threads: in a map behind one lock that was 309 waits for it and 280 looks
/// at its table in the 40 s of a profile, as many as all else of the landings'
/// notes (run faults-5).
struct Landed([std::sync::atomic::AtomicPtr<LandedPiece>; Landed::PIECES]);

/// Of 256 MiB: the frame (its low 31 bits and one), 0 for never.
type LandedPiece = [std::sync::atomic::AtomicU32; Landed::PAGES];

impl Landed {
    const PAGES: usize = 1 << 16;
    /// 128 TiB, what a process has.
    const PIECES: usize = 1 << 19;

    fn stamp(frame: u64) -> u32 {
        (frame as u32 & 0x7fff_ffff) + 1
    }

    fn note(&self, page: usize, frame: u64) {
        use std::sync::atomic::Ordering::{AcqRel, Acquire, Relaxed};
        let Some(place) = self.0.get(page / Landed::PAGES) else {
            return;
        };
        let mut piece = place.load(Acquire);
        if piece.is_null() {
            let made: Box<LandedPiece> = Box::new([const { std::sync::atomic::AtomicU32::new(0) }; Landed::PAGES]);
            let made = Box::into_raw(made);
            piece = match place.compare_exchange(std::ptr::null_mut(), made, AcqRel, Acquire) {
                Ok(_) => made,
                Err(there) => {
                    // SAFETY: made just now, and another thread's is there.
                    drop(unsafe { Box::from_raw(made) });
                    there
                }
            };
        }
        // SAFETY: a piece, which is never freed.
        let at = unsafe { &(&*piece)[page % Landed::PAGES] };
        let stamp = Landed::stamp(frame);
        // (Most are noted several times a frame: no write then.)
        if at.load(Relaxed) != stamp {
            at.store(stamp, Relaxed);
        }
    }

    /// Whether `page` was noted in a frame from `since` on, up to `now`.
    fn since(&self, page: usize, since: u64, now: u64) -> bool {
        let Some(place) = self.0.get(page / Landed::PAGES) else {
            return false;
        };
        // SAFETY: null, or a piece, which is never freed.
        let Some(piece) = (unsafe { place.load(std::sync::atomic::Ordering::Acquire).as_ref() }) else {
            return false;
        };
        let stamp = piece[page % Landed::PAGES].load(std::sync::atomic::Ordering::Relaxed);
        stamp != 0 && u64::from(Landed::stamp(now).wrapping_sub(stamp) & 0x7fff_ffff) <= now.saturating_sub(since)
    }
}

static LANDED: Landed = Landed([const { std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()) }; Landed::PIECES]);

/// A page's number, spread.
#[derive(Default)]
struct PageHash(u64);

impl std::hash::Hasher for PageHash {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.write_u64(u64::from(byte));
        }
    }
    fn write_u64(&mut self, n: u64) {
        self.0 = (self.0 ^ n).wrapping_mul(0x9e37_79b9_7f4a_7c15).rotate_left(26).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    }
    fn write_usize(&mut self, n: usize) {
        self.write_u64(n as u64);
    }
}

/// What shaders wrote is about to go into title memory at `[at, at + len)`.
pub(crate) fn landing(at: usize, len: usize) {
    {
        let frame = crate::frame::current();
        let (first, last) = (at / KERNEL_PAGE, (at + len.max(1) - 1) / KERNEL_PAGE);
        (first..=last).for_each(|page| LANDED.note(page, frame));
        for (block, others) in elsewhere(at, len) {
            let block = block / KERNEL_PAGE;
            let here = first.max(block)..=last.min(block + BLOCK / KERNEL_PAGE - 1);
            for other in others {
                here.clone().for_each(|page| LANDED.note(page - block + other / KERNEL_PAGE, frame));
            }
        }
    }
    writing(at, len);
}


/// What shaders wrote went into title memory through another view of the
/// same pages (`crate::inplace::beside`): `pages` (of 4 KiB, by number)
/// hold some of it, all of them in `spans` (address, bytes).
///
/// Nothing faulted and no guard was opened. A guarded block stays read-only,
/// clean, and known by its bit ([`PUBLISHED`]), all of which say that the
/// title wrote nothing there: our write is an entry of the journal, which
/// every answer about the block comes from, the quick ones and the guards'
/// alike (a block whose guard is off they report anyway). So neither the
/// guards' lock nor a protection is touched here, or by the next refresh.
/// Noted after the bytes are in, never before: a copy that read the block
/// before they were did so with the number of an asking made before that,
/// which is under the one given here, and takes it again; noted before, a
/// copy could read the older bytes with a number above the note's.
///
/// Returns the note's number, and for each 64 KiB block noted the number of
/// the block's last write before (0: none is known).
pub(crate) fn landed_beside(pages: &[usize], spans: &[(usize, usize)]) -> (u64, Vec<(usize, u64)>) {
    let mut before = Vec::with_capacity(spans.len());
    let noted;
    // The same pages at other addresses: copies of the memory there have not
    // what was put in either.
    let elsewhere: Vec<(usize, Vec<usize>)> = spans.iter().flat_map(|&(at, len)| elsewhere(at, len)).collect();
    {
        let frame = crate::frame::current();
        for &page in pages {
            LANDED.note(page, frame);
        }
        for (block, others) in &elsewhere {
            let first = block / KERNEL_PAGE;
            for &page in pages.iter().filter(|&&page| (first..first + BLOCK / KERNEL_PAGE).contains(&page)) {
                others.iter().for_each(|other| LANDED.note(page - first + other / KERNEL_PAGE, frame));
            }
        }
    }
    {
        let mut blocks: Vec<(usize, u16)> = Vec::with_capacity(spans.len());
        for &page in pages {
            let (block, part) = (page / PARTS, 1u16 << (page % PARTS));
            match blocks.last_mut() {
                Some((last, parts)) if *last == block => *parts |= part,
                _ => blocks.push((block, part)),
            }
        }
        let others = || elsewhere.iter().flat_map(|(_, others)| others.iter().map(|other| other / PAGE));
        if port_journal() {
            // Renderer port #7: shared, the parts of its pages held.
            let j = journal();
            let mut held = j.notes.hold(shards_of_pages(blocks.iter().map(|b| b.0).chain(others())), true);
            let now = next_asking();
            noted = now;
            for (block, parts) in blocks {
                before.push((block, held.note(block, parts, now)));
            }
            for other in others() {
                held.note(other, WHOLE, now);
            }
            port7_count(P7_SHARED_NOTES, 1);
        } else {
            let mut j = journal_mut();
            let now = next_asking();
            noted = now;
            for (block, parts) in blocks {
                before.push((block, j.notes.note(block, parts, now)));
            }
            for other in others() {
                j.notes.note(other, WHOLE, now);
            }
        }
    }
    for &(at, len) in spans {
        // A placed texture there changes with no descriptor written.
        crate::descriptor::host::remapped(at, len);
    }
    for other in elsewhere.into_iter().flat_map(|(_, others)| others) {
        crate::descriptor::host::remapped(other, BLOCK);
    }
    (noted, before)
}

/// Whether every write of the title's to 64 KiB block `block` so far is in
/// the journal ([`PUBLISHED`]).
pub(crate) fn block_published(block: usize) -> bool {
    published(block * BLOCK, (block + 1) * BLOCK)
}

/// Whether shaders' writes went into a page of `[at, at + len)` in the last
/// `frames` frames.
pub(crate) fn landed_lately(at: usize, len: usize, frames: u64) -> bool {
    let now = crate::frame::current();
    (at / 4096..=(at + len.max(1) - 1) / 4096).any(|page| LANDED.since(page, now.saturating_sub(frames), now))
}

/// Whether every block of `[at, at + len)` is view memory written since a
/// host copy last took it: guarded then, and the guard is off. (Written by
/// the title, or by us putting a flight's shader writes into it, if not
/// beside the guard, [`landed_beside`].) Not for
/// blocks no copy ever took, which were never guarded: the caller's range
/// lies in a copy a flight was given, which took all of it. (Nor for blocks
/// read-only by way of another view of their page: whether they were written
/// since is not known.)
pub(crate) fn written_since_taken(at: usize, len: usize) -> bool {
    if len == 0 {
        return false;
    }
    let g = guards();
    (at / BLOCK..=(at + len - 1) / BLOCK)
        .all(|b| !g.clean.contains(b) && !g.absent.contains(b) && !g.protected.contains(b))
}

/// For view memory: the pages of `[start, end)`'s blocks written since
/// they were last guarded (all of a block's pages), guarding them again
/// before the caller copies them. `None` if the memory is not a view.
fn guard(start: usize, end: usize) -> Option<Vec<usize>> {
    // A range whose first block the guards hold already (read-only by our
    // hand, or noted absent) is a view: no question to Windows. The
    // question walks the page tables to the end of the pages that share the
    // first one's state, and a guarded view is one long run of read-only
    // pages: 0.6 ms for the title's 174 MB heap, 16 times a frame, and most
    // of the 68 ms a frame of "guards" (run 58).
    // `VICTORY_GUARD_ASK=1` asks every time, as before.
    static ASK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    let known = !*ASK.get_or_init(|| std::env::var_os("VICTORY_GUARD_ASK").is_some_and(|v| v == "1"))
        && (PROTECTED.any(first, first + 1) || {
            // (Renderer port #6: shared, a look only.)
            let of = |g: &Guards| g.protected.contains(first) || g.absent.contains(first);
            if port_guard() { of(&guards_timed_shared()) } else { of(&guards_timed()) }
        });
    if !known && !is_view(start, end) {
        return None;
    }
    install();
    // The holes that an ordinary commit may have filled, with no word of the
    // runtime's. Windows is asked with the lock not held (the range is, no
    // other thread asks about it: `ASKING`); what the runtime tells meanwhile
    // takes the block out of the set, and it is looked at as any other.
    let unsure: Vec<usize> = {
        let of = |g: &Guards| -> Option<Vec<usize>> {
            if g.clean.range(first..last).count() == last - first && !BESIDE.any(first, last) {
                return None;
            }
            Some(g.absent.range(first..last).filter(|block| !g.held.contains(*block)).collect())
        };
        // (Renderer port #6: shared, a look only. A block a fault opens
        // meanwhile was clean, read-only and unwritten when looked at.)
        let unsure = if port_guard() { of(&guards_timed_shared()) } else { of(&guards_timed()) };
        match unsure {
            Some(unsure) => unsure,
            None => {
                // Nothing written, mapped or new since the range was last taken.
                #[cfg(feature = "diagnostics")]
                tally(3, 1);
                return Some(Vec::new());
            }
        }
    };
    let mut came: Vec<(usize, bool)> = Vec::new();
    let mut uncommitted_until = first;
    for block in unsure {
        // One question for the blocks of a region.
        if block >= uncommitted_until {
            uncommitted_until = uncommitted_end(block);
            #[cfg(feature = "diagnostics")]
            tally(9, 1);
        }
        if block >= uncommitted_until {
            // The hole may have been given up, and its address be the
            // title's heap's, or code's, now: not ours to guard.
            came.push((block, is_view(block * BLOCK, (block + 1) * BLOCK)));
        }
    }
    let rest = rest();
    if port_guard() {
        // (Renderer port #6: known for a view from now on.)
        if !VIEWS.all(first, last) {
            (first..last).for_each(|block| VIEWS.set(block));
        }
        return Some(guard_shared(first, last, &came, rest));
    }
    let mut g = guards_timed();
    #[cfg(feature = "diagnostics")]
    tally(4, 1);
    for &(block, view) in &came {
        if g.absent.remove(block) && !view {
            g.foreign.insert(block);
        }
    }
    // The range's blocks that are neither clean nor known to be holes. A
    // frame asks about some 10 GB of view memory, 150,000 blocks, nearly all
    // clean.
    let mut written = Vec::new();
    for word in first / 64..last.div_ceil(64) {
        let known = g.clean.word(word) & !BESIDE.word(word) | g.absent.word(word);
        written.extend(BitsOf(!known & in_word(word * 64, first, last)).map(|bit| word * 64 + bit));
    }
    let mut run: Option<(usize, usize)> = None;
    for &block in &written {
        if let Some((from, to)) = run
            && to != block
        {
            protect(&g, from, to);
            run = None;
            // Faults of the title's wait for the lock: each block is done on
            // its own, and looked at anew below.
            drop(g);
            g = guards_timed();
        }
        // Told as written by this, and read by the caller after.
        if BESIDE.take(block) {
            put_count(block).store(0, std::sync::atomic::Ordering::Relaxed);
        }
        if g.protected.contains(block) {
            // Read-only by way of another view of its page, or not clean
            // because one could not be made so. (Or clean, and bytes were put
            // in beside the guard: it stays.)
            if let Some((from, to)) = run.take() {
                protect(&g, from, to);
            }
            if g.clean.contains(block) {
                continue;
            }
            if cover(&g, block, block + 1, None) {
                g.clean.insert(block);
            }
        } else if !g.foreign.contains(&block) && !g.left_alone(block) && !g.rests(block, rest) {
            run = Some((run.map_or(block, |r| r.0), block + 1));
        } else if let Some((from, to)) = run.take() {
            protect(&g, from, to);
        }
    }
    if let Some((from, to)) = run {
        protect(&g, from, to);
    }
    drop(g);
    #[cfg(feature = "diagnostics")]
    crate::faultbook::note(crate::faultbook::What::Asked, first, last, written.len() as u64);
    Some(written)
}

/// The rest of [`guard`] with renderer port #6: the lock shared, and a
/// block's claim held while its state changes. What a fault opens meanwhile
/// is opened before or after the block's state is looked at and changed,
/// never in between; a block whose claim another holds (a fault, a cover of
/// another range's) is left as it is, written: told so now, and looked at
/// again at the next asking.
fn guard_shared(first: usize, last: usize, came: &[(usize, bool)], rest: std::time::Duration) -> Vec<usize> {
    if !came.is_empty() {
        // (Changes no block's protection: the sets of what is where.)
        let mut g = guards_timed();
        for &(block, view) in came {
            if g.absent.remove(block) && !view {
                g.foreign.insert(block);
            }
        }
    }
    port6_count(P6_GUARDS, 1);
    let g = guards_timed_shared();
    #[cfg(feature = "diagnostics")]
    tally(4, 1);
    let mut written = Vec::new();
    for word in first / 64..last.div_ceil(64) {
        let known = g.clean.word(word) & !BESIDE.word(word) | g.absent.word(word);
        written.extend(BitsOf(!known & in_word(word * 64, first, last)).map(|bit| word * 64 + bit));
    }
    let mut run: Option<(usize, usize)> = None;
    for &block in &written {
        if let Some((from, to)) = run
            && to != block
        {
            protect_claimed(&g, from, to);
            run = None;
        }
        // Told as written by this, and read by the caller after.
        if BESIDE.take(block) {
            put_count(block).store(0, std::sync::atomic::Ordering::Relaxed);
        }
        if g.protected.contains(block) {
            if let Some((from, to)) = run.take() {
                protect_claimed(&g, from, to);
            }
            if g.clean.contains(block) {
                continue;
            }
            // (No other claim is held here.)
            let mut claims = Claims::wait_for(&[block]);
            if g.protected.contains(block) && !g.clean.contains(block) && cover(&g, block, block + 1, Some(&mut claims)) {
                g.clean.insert(block);
            }
        } else if !g.foreign.contains(&block) && !g.left_alone(block) && !g.rests(block, rest) {
            run = Some((run.map_or(block, |r| r.0), block + 1));
        } else if let Some((from, to)) = run.take() {
            protect_claimed(&g, from, to);
        }
    }
    if let Some((from, to)) = run {
        protect_claimed(&g, from, to);
    }
    drop(g);
    #[cfg(feature = "diagnostics")]
    crate::faultbook::note(crate::faultbook::What::Asked, first, last, written.len() as u64);
    written
}

/// [`protect`] of `[from, to)` with the lock shared: of the blocks whose
/// claims can be had now, each run, its other views' blocks only where their
/// claims can be had too.
fn protect_claimed(g: &Guards, from: usize, to: usize) {
    let mut claims = Claims::none();
    let mut start = None;
    for block in from..to {
        if claims.try_one(block) {
            start.get_or_insert(block);
        } else if let Some(s) = start.take() {
            protect_run(g, s, block, &mut claims);
        }
    }
    if let Some(s) = start {
        protect_run(g, s, to, &mut claims);
    }
}

fn protect_run(g: &Guards, from: usize, to: usize, claims: &mut Claims) {
    #[cfg(feature = "diagnostics")]
    crate::faultbook::note(crate::faultbook::What::Taken, from, to, 0);
    protect_here(g, from, to);
    cover(g, from, to, Some(claims));
}

/// Whether `[start, end)` is view memory. The title's views lie in
/// placeholders, mapped and unmapped block by block as it streams, so a
/// range can begin in a block that is not mapped right now: the first
/// pages of the range that are there decide. A range with nothing there
/// counts as a view (its blocks are noted absent, and looked at again when
/// the runtime maps them).
///
/// Looking at the first address only took such a range for "not a view":
/// its copy was marked not watched for good and copied whole at every
/// refresh, around its unmapped pages (run 54: 300 such ranges, 570 MB
/// copied a frame for them, 255 copies a frame reading around holes).
/// `VICTORY_GUARD_HOLES=0` looks at the first address only, as before.
fn is_view(start: usize, end: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_MAPPED, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    static HOLES: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let holes = *HOLES.get_or_init(|| std::env::var_os("VICTORY_GUARD_HOLES").is_none_or(|v| v != "0"));
    let mut at = start;
    loop {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
        if n == 0 {
            return false;
        }
        if info.Type == MEM_MAPPED {
            return true;
        }
        // Committed memory that is no view (the title's heaps, ours), or
        // the answer as it was before.
        if info.State == MEM_COMMIT || !holes {
            return false;
        }
        // A hole (a placeholder, or free): on to what follows it.
        let next = (info.BaseAddress as usize).saturating_add(info.RegionSize);
        if next <= at || next >= end {
            return true;
        }
        at = next;
    }
}

/// Makes blocks `[from, to)` read-only and notes them guarded and clean;
/// block by block if they span views, leaving out any that fail, and
/// noting those not committed (unmapped) absent. The same physical pages
/// at every other view of them are made read-only too: a write through
/// one of those is this memory written, and only a fault can say so.
/// `VICTORY_VIEW_GUARD=0` leaves the other views as they are, as before.
fn protect(g: &Guards, from: usize, to: usize) {
    #[cfg(feature = "diagnostics")]
    crate::faultbook::note(crate::faultbook::What::Taken, from, to, 0);
    protect_here(g, from, to);
    cover(g, from, to, None);
}

/// Views of one page past which its views are not guarded together.
pub(crate) const MANY_VIEWS: usize = 64;

/// Whether all the views of a guarded page are guarded together
/// (`VICTORY_VIEW_GUARD`).
fn view_guard() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_VIEW_GUARD").is_none_or(|v| v != "0"))
}

/// Diagnostics: blocks guarded together with their other views, blocks
/// opened because another view of theirs was written, and blocks opened
/// because a view of theirs was mapped; since the start.
#[cfg(feature = "diagnostics")]
pub(crate) static ALIASES: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// Of the blocks `[from, to)`, which are read-only by our hand where they
/// can be: every other view of their pages is made read-only too. Where one
/// cannot be (the kernel writes through it now, or the runtime is changing
/// it), a write through it would not be seen: that block is not clean, and
/// told as written at every asking, until it can. Whether all are covered.
///
/// With the lock shared (renderer port #6), `claims` holds the claims of
/// `[from, to)`'s blocks, and gets those of the other views' blocks it can
/// have now; a view whose block's claim cannot be had is not covered.
fn cover(g: &Guards, from: usize, to: usize, mut claims: Option<&mut Claims>) -> bool {
    let mut all = true;
    for (page, others) in elsewhere(from * BLOCK, (to - from) * BLOCK) {
        let block = page / BLOCK;
        if !g.protected.contains(block) {
            continue; // not there, or not ours to guard
        }
        #[cfg(feature = "diagnostics")]
        ALIASES[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        g.aliased.insert(block);
        let mut covered = true;
        for other in others {
            covered &= protect_alias(g, other / BLOCK, claims.as_deref_mut());
        }
        if !covered {
            all = false;
            g.clean.remove(block);
        }
    }
    all
}

/// Makes `block` read-only: another view of a page a copy is about to take
/// through its own, so that a write through this one faults too. It is not
/// noted clean: the copies of title memory at this view have not taken the
/// page ([`guard`] tells such a block as written, then notes it clean).
/// One that is clean stays so: had the page been written since its copies
/// took it, [`open`] would have opened it. Whether it is read-only now.
fn protect_alias(g: &Guards, block: usize, claims: Option<&mut Claims>) -> bool {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, PAGE_READWRITE, VirtualProtect};
    if let Some(claims) = claims
        && !claims.holds(block)
        && !claims.try_one(block)
    {
        // Another changes it now (a fault opens it): not covered.
        return false;
    }
    g.aliased.insert(block);
    if g.protected.contains(block) {
        return true;
    }
    if g.left_alone(block) || g.rests(block, rest()) {
        return false;
    }
    // Ahead of the protection.
    PROTECTED.set(block);
    let mut old = 0;
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    // SAFETY: title view memory; only its protection changes.
    let done = unsafe { VirtualProtect((block * BLOCK) as *const _, BLOCK, PAGE_READONLY, &mut old) } != 0;
    #[cfg(feature = "diagnostics")]
    {
        tally(5, 1);
        tally(8, started.elapsed().as_nanos() as u64);
    }
    if done && matches!(old, PAGE_READWRITE | PAGE_READONLY) {
        g.protected.insert(block);
        #[cfg(feature = "diagnostics")]
        crate::faultbook::note(crate::faultbook::What::Alias, block, 0, 0);
        return true;
    }
    if done {
        // Not for us to guard: as it was.
        // SAFETY: as above.
        unsafe { VirtualProtect((block * BLOCK) as *const _, BLOCK, old, &mut old) };
    }
    PROTECTED.clear(block);
    false
}

/// [`protect`] for one range, without the other views of its pages.
fn protect_here(g: &Guards, from: usize, to: usize) {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, PAGE_READWRITE, VirtualProtect};
    let set = |from: usize, to: usize| {
        let mut old = 0;
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        // SAFETY: title view memory; only its protection changes.
        let mut done = unsafe {
            VirtualProtect(
                (from * BLOCK) as *const _,
                (to - from) * BLOCK,
                PAGE_READONLY,
                &mut old,
            ) != 0
        };
        if done && !matches!(old, PAGE_READWRITE | PAGE_READONLY) {
            // What is there is not for us to guard (code, or memory the
            // title protects itself): as it was.
            // SAFETY: as above.
            unsafe { VirtualProtect((from * BLOCK) as *const _, (to - from) * BLOCK, old, &mut old) };
            done = false;
        }
        #[cfg(feature = "diagnostics")]
        {
            tally(5, (to - from) as u64);
            tally(8, started.elapsed().as_nanos() as u64);
        }
        done
    };
    let note = |g: &Guards, from: usize, to: usize| {
        g.protected.insert_all(from..to);
        g.clean.insert_all(from..to);
    };
    // Ahead of the protection (`PROTECTED`).
    (from..to).for_each(|block| PROTECTED.set(block));
    if set(from, to) {
        note(g, from, to);
        return;
    }
    let mut uncommitted_until = from;
    for block in from..to {
        // (Not one question a block of a region that is not there.)
        if block >= uncommitted_until && set(block, block + 1) {
            note(g, block, block + 1);
            continue;
        }
        PROTECTED.clear(block);
        if block >= uncommitted_until {
            uncommitted_until = uncommitted_end(block);
            #[cfg(feature = "diagnostics")]
            tally(9, 1);
        }
        if block < uncommitted_until {
            g.absent.insert(block);
            if known_holes() && placeholder(block) {
                g.held.insert(block);
            }
        }
    }
}

/// The end (exclusive) of the whole uncommitted blocks starting at `block`.
fn uncommitted_end(block: usize) -> usize {
    use windows_sys::Win32::System::Memory::{
        MEM_FREE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery((block * BLOCK) as *const _, &mut info, size_of_val(&info)) };
    if n != 0 && matches!(info.State, MEM_RESERVE | MEM_FREE) {
        (info.BaseAddress as usize).saturating_add(info.RegionSize) / BLOCK
    } else {
        block
    }
}

/// Gives write access back to the guarded blocks of `[start, end)` and
/// notes them written; how many there were. The same physical pages at
/// every other view of them are written by that too (the write that faulted,
/// or is ours, `crate::host::copy_mapped`, goes through one view): they are
/// clean no longer, whether they are opened as well ([`open_one_view`]) or
/// not.
fn open(g: &Guards, start: usize, end: usize) -> usize {
    open_given(g, start, end, None)
}

/// [`open`]; `given`: the blocks of the page's views, for a fault opening one
/// block with the lock shared, which holds their claims ([`open_faulted`]).
fn open_given(g: &Guards, start: usize, end: usize, given: Option<&[usize]>) -> usize {
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    if first >= last {
        return 0;
    }
    if first + 1 == last {
        let protected = g.protected.contains(first);
        if !g.aliased.contains(first) {
            if protected {
                unguard(g, &[first]);
                return 1;
            }
            return 0;
        }
    }
    let mut blocks: Vec<usize> = g.protected.range(first..last).collect();
    let here = blocks.len();
    let aliased: Vec<usize> = g.aliased.range(first..last).collect();
    if !aliased.is_empty() {
        let one = open_one_view();
        for block in aliased {
            g.aliased.remove(block);
            let others: Vec<usize> = match given {
                Some(given) => given.to_vec(),
                None => crate::inplace::pages_of(block * BLOCK).into_iter().map(|p| p / BLOCK).collect(),
            };
            for other in others {
                if one {
                    // Before the first can be written.
                    g.clean.remove(other);
                    PUBLISHED.clear(other);
                    continue;
                }
                g.aliased.remove(other);
                if !(first..last).contains(&other) && g.protected.contains(other) {
                    blocks.push(other);
                }
            }
        }
        #[cfg(feature = "diagnostics")]
        ALIASES[1].fetch_add((blocks.len() - here) as u64, std::sync::atomic::Ordering::Relaxed);
    }
    unguard(g, &blocks);
    here
}

/// Gives write access back to `blocks`, guarded ones.
fn unguard(g: &Guards, blocks: &[usize]) {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, PAGE_READWRITE, VirtualProtect};
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    // All of them before the first can be written: a write through one is a
    // write of the others' bytes.
    for &block in blocks {
        g.protected.remove(block);
        g.clean.remove(block);
        PUBLISHED.clear(block);
    }
    for &block in blocks {
        let mut old = 0;
        // SAFETY: a block we made read-only.
        unsafe {
            if VirtualProtect((block * BLOCK) as *const _, BLOCK, PAGE_READWRITE, &mut old) != 0
                && !matches!(old, PAGE_READONLY | PAGE_READWRITE)
            {
                // Not what we left there: the address is other memory's
                // (code's, say), which keeps its protection.
                VirtualProtect((block * BLOCK) as *const _, BLOCK, old, &mut old);
            }
        }
        PROTECTED.clear(block);
    }
    #[cfg(feature = "diagnostics")]
    if !blocks.is_empty() {
        tally(8, started.elapsed().as_nanos() as u64);
    }
}

/// Before our own write into title memory `[at, at + len)` through the
/// kernel (which does not fault into our handler): opens its guarded blocks.
/// The runtime calls it (as `kernel_will_write`) before the title's kernel
/// reads into its own memory.
pub fn writing(at: usize, len: usize) {
    // The length is the title's (`fread`'s size times count, saturated).
    let last = at.saturating_add(len).div_ceil(BLOCK);
    if !light() || PROTECTED.any(at / BLOCK, last) {
        let g = guards_timed();
        let opened = open(&g, at, at.saturating_add(len));
        #[cfg(feature = "diagnostics")]
        {
            tally(6, opened as u64);
            crate::faultbook::note(crate::faultbook::What::Ours, at / BLOCK, last, opened as u64);
        }
        let _ = opened;
    }
    // A placed texture there changes with no descriptor written.
    crate::descriptor::host::remapped(at, len);
}

/// The kernel writes title memory `[at, at + len)` until what this returns
/// is dropped ([`kernel_writing`]).
pub struct KernelWrite(usize, usize);

/// The runtime is about to have the kernel write title memory
/// `[at, at + len)` (a file read): its guarded blocks are opened, as by
/// [`writing`], and no refresh guards them again before what this returns is
/// dropped, at the end of that call. One that did would make the kernel's
/// write fail (the kernel does not fault into our handler): a read cut
/// short, which the title takes for what the file holds.
pub fn kernel_writing(at: usize, len: usize) -> KernelWrite {
    let (first, last) = (at / BLOCK, at.saturating_add(len).div_ceil(BLOCK));
    {
        let mut g = guards_timed();
        open(&g, at, at.saturating_add(len));
        if len != 0 {
            // Nothing known of them stands while they are left alone.
            let absent: Vec<usize> = g.absent.range(first..last).collect();
            for block in absent {
                g.absent.remove(block);
                if g.held.remove(block) {
                    PUBLISHED.clear(block);
                }
            }
            g.kernel.push((first, last));
        }
    }
    crate::descriptor::host::remapped(at, len);
    KernelWrite(first, if len == 0 { first } else { last })
}

impl Drop for KernelWrite {
    fn drop(&mut self) {
        if self.0 == self.1 {
            return;
        }
        let mut g = guards();
        if let Some(i) = g.kernel.iter().position(|&r| r == (self.0, self.1)) {
            g.kernel.swap_remove(i);
        }
    }
}

/// What `VICTORY_PAGE_WRITES` asks to be done with a write the guards do
/// not see once its bytes are there ([`title_memory_written`]), as bits:
/// 1, the guarded blocks under it are opened and count as written (host
/// copies of buffers and textures read through tables take them again);
/// 2, the placed textures picked by index that lie under it are due again;
/// 4, both also for reads to plain addresses when they have finished.
/// Unset or `guards`: bit 1 alone. `0`: nothing, counted only (the default
/// until run 199). `1`: bits 1 and 2. `slots`: bit 2 alone. `all`:
/// everything.
/// (Bit 1 by default: in every run 10,000 to 50,000 guarded blocks of host
/// copies lay under such writes and kept the bytes from before; runs 197
/// and 198 opened them, 9,766 and 10,796 blocks, with nothing worse for
/// it. Bit 2 makes texture slots due and is the one that flooded the
/// queue in an earlier form: still asked for by name.)
pub(crate) fn page_writes() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_PAGE_WRITES").as_deref().map(str::trim) {
        Ok("0") => 0,
        Ok("1") => 3,
        Ok("slots") => 2,
        Ok("all") => 7,
        _ => 1,
    })
}

/// Since the start: writes into physical pages through the runtime's own
/// views that landed at addresses the title has mapped (one for each run
/// of addresses); their bytes; guarded blocks that lay under them (host
/// copies that would otherwise keep the bytes from before); reads to
/// plain addresses that finished.
pub(crate) static WRITTEN: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// The bytes of a write the guards do not see are there: title memory
/// `[at, at + len)`. `pages`: DirectStorage wrote them into physical pages
/// through a view of the runtime's own, and this is an address where the
/// title has those pages mapped (no title address faulted, nothing was
/// mapped or unmapped; the shortcut this file's header names). Otherwise a
/// read to a plain address has finished, of which [`writing`] was told
/// before it began.
///
/// Upstream's c75fdba does the same for its trackers (its `Writing` marks
/// the pages once more when dropped; page destinations are reported at
/// each address showing them). Here it is counted always and acted on as
/// [`page_writes`] says. Called without the runtime's page-table lock;
/// takes the guards' lock, then, alone, the lock of the ranges kept for
/// the texture pass.
pub fn title_memory_written(at: usize, len: usize, pages: bool) {
    use std::sync::atomic::Ordering::Relaxed;
    if len == 0 {
        return;
    }
    let mode = page_writes();
    let end = at.saturating_add(len);
    if pages {
        WRITTEN[0].fetch_add(1, Relaxed);
        WRITTEN[1].fetch_add(len as u64, Relaxed);
    } else {
        WRITTEN[3].fetch_add(1, Relaxed);
        if mode & 4 == 0 {
            return;
        }
    }
    {
        let g = guards();
        let guarded = g.protected.range(at / BLOCK..end.div_ceil(BLOCK)).count() as u64;
        if pages {
            WRITTEN[2].fetch_add(guarded, Relaxed);
        }
        if mode & 1 != 0 && guarded != 0 {
            open(&g, at, end);
        }
    }
    // A placed texture there changed with no descriptor written.
    crate::descriptor::host::page_written(at, len);
}

/// The runtime mapped a view of the physical bytes `[offset, offset + len)`
/// of `section` (`crate::inplace::title_memory_view`): the view is
/// writable, so the same bytes guarded at another view must be checked
/// again - and are, the pages noted written, which also has the next check
/// guard the new view ([`cover`]).
pub(crate) fn views_mapped(section: usize, offset: u64, len: usize) {
    if !view_guard() {
        return;
    }
    let ranges = crate::inplace::views_of_section(section, offset, len);
    // (A block is noted in `PROTECTED` ahead of the look at its page's views
    // that did not find this one yet.)
    if !ranges.iter().any(|&(at, n)| PROTECTED.any(at / BLOCK, (at + n).div_ceil(BLOCK))) {
        return;
    }
    // All of them, once one is guarded: another may have been made clean
    // since that look, relying on those that are opened here. And all
    // together: none may be writable while another is clean (who asks about
    // that one by its bits takes no lock).
    let g = guards();
    let blocks: Vec<usize> =
        ranges.iter().flat_map(|&(at, n)| g.protected.range(at / BLOCK..(at + n).div_ceil(BLOCK)).collect::<Vec<_>>()).collect();
    #[cfg(feature = "diagnostics")]
    ALIASES[2].fetch_add(blocks.len() as u64, std::sync::atomic::Ordering::Relaxed);
    unguard(&g, &blocks);
}

/// The runtime is about to unmap a view of which it maps a part again at
/// once ([`title_memory_unmapping`] of all of it comes next): the title's
/// threads may write through that part before we are told of it
/// ([`views_mapped`]). What is guarded of its pages at other views is opened,
/// and not guarded again before the change is done ([`Remapping`]).
pub fn title_memory_splitting(at: usize, len: usize) {
    let others: Vec<usize> = elsewhere(at, len).into_iter().flat_map(|(_, others)| others).collect();
    if others.is_empty() {
        return;
    }
    let mut g = guards();
    // (The view itself too: the part that stays is read-only until then. And
    // all of them told as written before the first is writable.)
    for &other in &others {
        g.clean.remove(other / BLOCK);
        PUBLISHED.clear(other / BLOCK);
    }
    open(&g, at, at.saturating_add(len));
    for other in others {
        open(&g, other, other + BLOCK);
        if CHANGING.get().0 {
            CHANGING.set((true, true));
            g.changing.push((other / BLOCK, other / BLOCK + 1));
        }
    }
}

/// Held shared by a read of title memory in place, and alone by the runtime
/// while it changes what title memory shows.
///
/// The runtime does not change a mapping in one step. To unmap one page of
/// a view it unmaps the whole view and maps what stays again
/// (`xmem.rs` `unmap`), and a view it maps is a reservation until the next
/// call commits it (`map`, then `commit`). For that moment pages the title
/// never gave up are "reserved, never committed", and a thread reading
/// them in place takes an access violation: run 71 in `Layout::detile`,
/// run 74 in the hash before it, both under `Staging::refresh_now`, which
/// had asked whether the texture was all mapped and been told yes. Asking
/// cannot help; not changing the mapping while such a read is under way
/// does.
static MAPPING: std::sync::RwLock<()> = std::sync::RwLock::new(());

/// What the runtime holds while it changes what title memory shows
/// ([`title_memory_remapping`]).
pub struct Remapping(#[allow(dead_code)] std::sync::RwLockWriteGuard<'static, ()>);

impl Drop for Remapping {
    fn drop(&mut self) {
        // The change is done: what is there now may be guarded.
        if CHANGING.replace((false, false)).1 {
            guards().changing.clear();
        }
        CHANGES.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }
}

/// How often the runtime began or ended a change of what title memory
/// shows: odd while one is under way.
static CHANGES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

thread_local! {
    /// This thread holds [`Remapping`]; and put blocks in
    /// [`Guards::changing`].
    static CHANGING: std::cell::Cell<(bool, bool)> = const { std::cell::Cell::new((false, false)) };
}

/// For [`again_after_change`]: to be asked before what may fail.
pub(crate) fn changes() -> u64 {
    CHANGES.load(std::sync::atomic::Ordering::SeqCst)
}

/// `f` failed to read or write title memory, at some time since [`changes`]
/// said `since`. If the runtime changed a mapping in that time, the pages
/// may have been away for that moment only ([`MAPPING`]): again, when the
/// change is done. Whether `f` did it.
pub(crate) fn again_after_change(since: u64, mut f: impl FnMut() -> bool) -> bool {
    let mut since = since;
    for _ in 0..4 {
        let now = change_done();
        if now == since {
            return false;
        }
        since = now;
        if f() {
            return true;
        }
    }
    false
}

/// Returns when the change of a mapping that is under way, if one is, is
/// over (or has been going on for 2 ms): pages it took away for its moment
/// are back. Another may have begun.
pub(crate) fn change_over() {
    let now = changes();
    if now.is_multiple_of(2) {
        return;
    }
    let started = std::time::Instant::now();
    while changes() == now && started.elapsed().as_millis() < 2 {
        for _ in 0..50 {
            std::hint::spin_loop();
        }
    }
}

/// [`changes`], when no change is under way (or one has been for a second).
fn change_done() -> u64 {
    let mut tries = 0;
    let mut started = None;
    loop {
        let now = changes();
        if now.is_multiple_of(2) {
            return now;
        }
        tries += 1;
        if tries < 100 {
            std::hint::spin_loop();
            continue;
        }
        if started.get_or_insert_with(std::time::Instant::now).elapsed().as_secs() >= 1 {
            return now;
        }
        // SAFETY: gives the processor to another thread.
        unsafe { windows_sys::Win32::System::Threading::Sleep(u32::from(tries > 200)) };
    }
}

/// Faults of title threads that met a change of mapping and went on after
/// it ([`came_back`]).
static MET: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// A title thread's access to `at` (`write`: a write) faulted. Whether it
/// can be made now: if a change of mapping is under way, after it.
///
/// The console changes one page's mapping and no other's. Here the rest of
/// a view is unmapped and mapped again ([`MAPPING`]), and a title thread that
/// uses a page of it just then, which the title never gave up, faulted
/// ("reserved, never committed", or read-only for the moment between our
/// forgetting a guarded block and its unmapping). It waits here instead.
/// Memory that is not there with no change under way is not waited for.
fn came_back(at: usize, write: bool) -> bool {
    let there = || {
        if !write {
            return readable_view(at);
        }
        // (A refresh may have guarded the page that came back.)
        open_faulted(at).0 != 0 || writable(at)
    };
    let before = changes();
    if there() || again_after_change(before, there) {
        MET.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return true;
    }
    false
}

/// Whether `at` is in a committed page of a view that can be read.
fn readable_view(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_MAPPED, MEMORY_BASIC_INFORMATION, PAGE_READONLY, PAGE_READWRITE, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0
        && info.State == MEM_COMMIT
        && info.Type == MEM_MAPPED
        && matches!(info.Protect, PAGE_READONLY | PAGE_READWRITE)
}

/// Title memory that could not be written with what was to go there (what a
/// shader wrote, a title's copy): counted, and said at the 1st, 2nd, 4th, 8th
/// and so on. A page the title unmapped loses nothing by it; a page that is
/// there does.
pub(crate) fn lost(what: &'static str, at: usize, len: usize) {
    static LOST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = LOST.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    if n.is_power_of_two() {
        let there = writable(at);
        tracing::warn!(n, what, at = format_args!("{at:#x}"), len, there, "title memory not written");
    }
}


/// The runtime is about to change a mapping or free a title allocation.
/// Until this guard is dropped, reads of title memory in place stay paused
/// ([`title_memory_reading`]). Hold it through the whole change.
pub fn title_memory_remapping() -> Remapping {
    // For the title's own threads ([`came_back`]).
    install();
    let met = MET.load(std::sync::atomic::Ordering::Relaxed);
    if met != 0 {
        static SAID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        if met >= 2 * SAID.load(std::sync::atomic::Ordering::Relaxed) {
            SAID.store(met, std::sync::atomic::Ordering::Relaxed);
            tracing::info!(met, "accesses of title threads that waited for a change of mapping");
        }
    }
    let changing = MAPPING.write().unwrap_or_else(|e| e.into_inner());
    CHANGES.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    CHANGING.set((true, false));
    Remapping(changing)
}

/// What a read of title memory in place holds ([`title_memory_reading`]).
pub struct Reading(#[allow(dead_code)] std::sync::RwLockReadGuard<'static, ()>);

/// For reading title memory in place: until what it returns is dropped,
/// the runtime changes no mapping, so memory found mapped stays mapped.
/// Not to be taken twice by one thread, nor held across anything that
/// maps title memory.
pub fn title_memory_reading() -> Reading {
    Reading(MAPPING.read().unwrap_or_else(|e| e.into_inner()))
}

/// [`title_memory_reading`] if it can be had at once: `None` while the
/// runtime changes a mapping (or waits to), for a caller that has another
/// way to read and would rather not wait.
pub fn title_memory_reading_now() -> Option<Reading> {
    match MAPPING.try_read() {
        Ok(guard) => Some(Reading(guard)),
        Err(std::sync::TryLockError::Poisoned(e)) => Some(Reading(e.into_inner())),
        Err(std::sync::TryLockError::WouldBlock) => None,
    }
}

/// The runtime mapped title memory `[at, at + len)`: what it shows is new,
/// so it is no longer guarded (nor known unmapped), and read-write.
///
/// The view is there before we are told of it, and a copy's refresh on
/// another thread may have guarded its blocks in between (it finds blocks
/// it knew absent committed, and holds [`GUARDS`], which we wait for).
/// Forgetting such a block and leaving it read-only would make the title's
/// next write to it an access violation nobody takes (`on_write` finds it
/// neither guarded nor writable) and a kernel read into it fail, so the
/// write access is given back.
pub fn title_memory_remapped(at: usize, len: usize) {
    crate::texback::remapped(at, len);
    open(&title_memory_changed(at, len), at, at.saturating_add(len));
}

/// The runtime is about to unmap title memory `[at, at + len)`: it is
/// nothing of ours any more. Its protection goes with it. Until the change
/// is done ([`Remapping`]) no refresh guards what is there: one that did
/// before the unmapping would leave the address guarded, clean and known by
/// its bits ([`PUBLISHED`]) with no view behind it.
pub fn title_memory_unmapping(at: usize, len: usize) {
    crate::texback::remapped(at, len);
    forget_views(at / BLOCK, at.saturating_add(len).div_ceil(BLOCK));
    let mut g = title_memory_changed(at, len);
    if CHANGING.get().0 && len != 0 {
        CHANGING.set((true, true));
        let (first, last) = (at / BLOCK, at.saturating_add(len).div_ceil(BLOCK));
        match g.changing.last_mut() {
            Some(before) if before.1 == first => before.1 = last,
            _ => g.changing.push((first, last)),
        }
    }
    let blocks: Vec<usize> = g.protected.range(at / BLOCK..at.saturating_add(len).div_ceil(BLOCK)).collect();
    for block in blocks {
        g.protected.remove(block);
        g.clean.remove(block);
        PUBLISHED.clear(block);
        PROTECTED.clear(block);
    }
}

/// The title is about to free or decommit memory of its own,
/// `[at, at + len)`, or with no length the whole allocation at `at`: what
/// was noted of its blocks (guarded, clean, absent) would be taken for true
/// of the next memory at the address.
pub fn title_memory_freed(at: usize, len: usize) {
    if len != 0 {
        forget_views(at / BLOCK, at.saturating_add(len).div_ceil(BLOCK));
    } else if VIEWS.any(at / BLOCK, at / BLOCK + 1) {
        forget_views(at / BLOCK, allocation_end(at, at).div_ceil(BLOCK).max(at / BLOCK + 1));
    }
    let mut g = guards();
    let first = at / BLOCK;
    let noted = [g.protected.range(first..).next(), g.absent.range(first..).next(), g.foreign.range(first..).next().copied()];
    let Some(noted) = noted.into_iter().flatten().min() else {
        return;
    };
    let end = if len != 0 {
        at.saturating_add(len).div_ceil(BLOCK)
    } else {
        allocation_end(at, noted * BLOCK).div_ceil(BLOCK)
    };
    if noted >= end {
        return;
    }
    let absent: Vec<usize> = g.absent.range(first..end).collect();
    for block in absent {
        g.absent.remove(block);
        if g.held.remove(block) {
            PUBLISHED.clear(block);
        }
    }
    let foreign: Vec<usize> = g.foreign.range(first..end).copied().collect();
    for block in foreign {
        g.foreign.remove(&block);
    }
    // (Memory that is still a view is not freed by that call, which fails:
    // it must not stay read-only.)
    open(&g, at, end * BLOCK);
}

/// The end of the allocation at `at`, if `inside` lies in it; else `at`.
fn allocation_end(at: usize, inside: usize) -> usize {
    use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
    let ask = |address: usize| {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(address as *const _, &mut info, size_of_val(&info)) };
        (n != 0 && info.AllocationBase as usize == at)
            .then(|| (info.BaseAddress as usize).saturating_add(info.RegionSize))
    };
    let mut end = at;
    let mut next = ask(inside.max(at));
    while let Some(to) = next {
        end = to;
        next = ask(to);
    }
    end
}

/// What is known of `[first, last)` as view memory ([`VIEWS`]) is forgotten.
fn forget_views(first: usize, last: usize) {
    if !VIEWS.any(first, last) {
        return;
    }
    let set: Vec<usize> = VIEWS.set_in(first, last).collect();
    for block in set {
        VIEWS.clear(block);
    }
}

/// Tells what keeps track of what title memory shows; the guards, with no
/// block of the range known unmapped.
fn title_memory_changed(at: usize, len: usize) -> std::sync::RwLockWriteGuard<'static, Guards> {
    // A placed texture there arrives, goes or changes with no descriptor
    // written.
    crate::descriptor::host::remapped(at, len);
    // The views the runtime told us of there are no longer what it shows.
    crate::inplace::views_gone(at, len);
    // And nothing stays in place over it.
    crate::inplace::title_memory_freeing(at, len);
    let mut g = guards();
    let range = at / BLOCK..at.saturating_add(len).div_ceil(BLOCK);
    let absent: Vec<usize> = g.absent.range(range.clone()).collect();
    for block in absent {
        g.absent.remove(block);
        if g.held.remove(block) {
            PUBLISHED.clear(block);
        }
    }
    let foreign: Vec<usize> = g.foreign.range(range.clone()).copied().collect();
    for block in foreign {
        g.foreign.remove(&block);
    }
    let aliased: Vec<usize> = g.aliased.range(range).collect();
    for block in aliased {
        g.aliased.remove(block);
    }
    g
}

/// Registers the handler that takes writes to guarded blocks, once, ahead
/// of the crash reporter's.
pub(crate) fn install() {
    use windows_sys::Win32::Foundation::{
        EXCEPTION_ACCESS_VIOLATION, NTSTATUS, STATUS_GUARD_PAGE_VIOLATION,
    };
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
        EXCEPTION_POINTERS,
    };
    unsafe extern "system" fn on_write(info: *mut EXCEPTION_POINTERS) -> i32 {
        let _fault = crate::timeline::span("a fault looked at");
        // SAFETY: the system passes the exception being dispatched.
        let r = unsafe { &*(*info).ExceptionRecord };
        const READ: usize = 0;
        const WRITE: usize = 1;
        // A fault of our own copies (`crate::safecopy`): whatever it is, if
        // it is not a block we guard, the copy fails and goes on from there.
        let ours = crate::safecopy::failing_exit(r.ExceptionAddress as usize);
        // SAFETY: as above.
        let context = unsafe { (*info).ContextRecord };
        let fail = move |exit: usize| {
            // SAFETY: the context the system passes, to go on from.
            unsafe { (*context).Rip = exit as u64 };
            EXCEPTION_CONTINUE_EXECUTION
        };
        if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS
            || r.NumberParameters < 2
            || r.ExceptionInformation[0] != WRITE
        {
            return match ours {
                Some(exit) => {
                    // A guard page of title memory fails the copy as the kernel's read would: the
                    // fault cleared its guard flag, so it is set again (or counted, if it cannot be).
                    if r.ExceptionCode == STATUS_GUARD_PAGE_VIOLATION {
                        rearm_guard((r.NumberParameters >= 2).then(|| r.ExceptionInformation[1]));
                    }
                    fail(exit)
                }
                None if r.ExceptionCode == EXCEPTION_ACCESS_VIOLATION as NTSTATUS
                    && r.NumberParameters >= 2
                    && r.ExceptionInformation[0] == READ
                    && came_back(r.ExceptionInformation[1], false) =>
                {
                    EXCEPTION_CONTINUE_EXECUTION
                }
                None => EXCEPTION_CONTINUE_SEARCH,
            };
        }
        let at = r.ExceptionInformation[1];
        let block = at / BLOCK;
        if ours.is_none()
            && PROTECTED.any(block, block + 1)
            // SAFETY: the context the system passes, to go on from.
            && crate::titlewrite::store(unsafe { &mut *context }, at)
        {
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        // Another thread may have opened the block since this one faulted
        // (then nothing waits for the lock), and a copy may have guarded it
        // again since that (then it is ours again, not read-only for some
        // other reason).
        for _ in 0..3 {
            if !PROTECTED.any(block, block + 1) && writable(at) {
                #[cfg(feature = "diagnostics")]
                crate::faultbook::note(crate::faultbook::What::Open, block, 0, 0);
                return EXCEPTION_CONTINUE_EXECUTION;
            }
            #[cfg(feature = "diagnostics")]
            let began = std::time::Instant::now();
            // (Let go again before the next look: the runtime wants the
            // guards to end a change.)
            let (opened, _waited) = open_faulted(at);
            #[cfg(feature = "diagnostics")]
            let waited = _waited;
            if opened != 0 {
                // Renderer port #2's fallback says who wrote a block of the heap.
                crate::port_descriptors::faulted(at, r.ExceptionAddress as usize);
                #[cfg(feature = "diagnostics")]
                {
                    crate::copystat::add(crate::copystat::C::GuardFaults, 1);
                    let ns = waited.as_nanos().min(u128::from(u32::MAX)) as usize;
                    let held = began.elapsed().saturating_sub(waited).as_nanos() as u64;
                    crate::faultbook::note(crate::faultbook::What::Fault, block, opened << 32 | ns, held);
                    crate::faultbook::note(crate::faultbook::What::Site, r.ExceptionAddress as usize, block, 0);
                }
                return EXCEPTION_CONTINUE_EXECUTION;
            }
        }
        if writable(at) {
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        match ours {
            Some(exit) => fail(exit),
            None if came_back(at, true) => EXCEPTION_CONTINUE_EXECUTION,
            None => EXCEPTION_CONTINUE_SEARCH,
        }
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: the handler only touches our guard state and protections.
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(on_write));
    });
}

/// Sets the guard flag again on the page holding `at` (the address the fault
/// reported, if it did), with the protection the page has now: the fault
/// cleared the flag. False, and counted by [`not_rearmed`], if the address is
/// not known, the page is not committed any more, or its protection cannot be
/// set.
fn rearm_guard(at: Option<usize>) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_GUARD, VirtualProtect, VirtualQuery,
    };
    let done = at.is_some_and(|at| {
        let page = at / KERNEL_PAGE * KERNEL_PAGE;
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        let mut old = 0;
        // SAFETY: valid out pointers of the right size; the protection change is of one page, and
        // only if that page is committed.
        unsafe {
            VirtualQuery(page as *const _, &mut info, size_of_val(&info)) != 0
                && info.State == MEM_COMMIT
                && VirtualProtect(page as *const _, KERNEL_PAGE, info.Protect | PAGE_GUARD, &mut old) != 0
        }
    });
    if !done {
        not_rearmed(at);
    }
    done
}

/// Guard pages that a copy stopped at and that could not be guarded again.
static NOT_REARMED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Counts a guard page that could not be guarded again; the first is said
/// (the title may then not grow its stack, or catch that guard, at the page).
fn not_rearmed(at: Option<usize>) {
    if NOT_REARMED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) == 0 {
        tracing::warn!(
            at = format_args!("{:#x}", at.unwrap_or(0)),
            "guard not re-armed: a copy of title memory took a guard page and it could not be set again (0: no address)"
        );
    }
}

/// Whether `at` is committed read-write memory now.
fn writable(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READWRITE, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0 && info.State == MEM_COMMIT && info.Protect == PAGE_READWRITE
}

#[cfg(test)]
mod parts_tests {
    use super::*;

    #[test]
    fn ranges_asked_ahead_hold_what_lies_in_one_of_them() {
        let all = [(30, 40), (12, 14), (10, 16), (15, 35), (40, 48), (10, 20), (10, 20), (50, 51)];
        let indexed = Ahead::from_ranges(all.to_vec());
        let mut ahead = Ahead::new();
        let mut asked = Vec::new();
        for (first, last) in all {
            let uncovered = !asked.iter().any(|&(a, b)| a <= first && last <= b);
            assert_eq!(ahead.insert_uncovered(first, last), uncovered, "{first}..{last}");
            if uncovered {
                asked.push((first, last));
            }
        }
        assert_eq!(ahead.ranges, indexed.ranges);
        for first in 0..60 {
            for last in first + 1..62 {
                let slow = all.iter().any(|&(a, b)| a <= first && last <= b);
                assert_eq!(ahead.holds(first, last), slow, "{first}..{last}");
            }
        }
        assert!(!Ahead::new().holds(0, 1));
        ahead.clear();
        assert!(ahead.ranges.is_empty() && !ahead.holds(10, 11));
    }

    #[test]
    fn the_parts_of_a_page_written_since_it_was_taken() {
        let mut j = Shard::new();
        assert_eq!(j.parts_since(7, 0), WHOLE);
        assert_eq!(j.note(7, WHOLE, 3), 0);
        // Parts 0 and 5 at 4, part 5 again at 6: at 2 the whole page is due.
        assert_eq!(j.note(7, 0b10_0001, 4), 3);
        assert_eq!(j.note(7, 0b10_0000, 6), 4);
        assert_eq!(j.parts_since(7, 2), WHOLE);
        assert_eq!(j.parts_since(7, 3), 0b10_0001);
        assert_eq!(j.parts_since(7, 4), 0b10_0000);
        assert_eq!(j.parts_since(7, 5), 0b10_0000);
        assert_eq!(j.parts_since(7, 6), 0);
        // Written whole: all of it, whoever took it before.
        assert_eq!(j.note(7, WHOLE, 8), 6);
        assert_eq!(j.parts_since(7, 7), WHOLE);
        // A page never written before: what was not named was never written.
        assert_eq!(j.note(9, 0b100, 9), 0);
        assert_eq!(j.parts_since(9, 5), 0b100);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };
    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    #[test]
    fn written_pages_are_reported_to_every_asker() {
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                4 * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(p, 0);
        let (start, end) = (p, p + 4 * PAGE);
        // The journal may hold pages of another test's memory that was here.
        let (none, before) = writes(start, end, false).expect("watched");
        assert!(before.iter().all(|&(_, w)| w < none), "{before:?}");
        // SAFETY: committed pages of ours.
        unsafe { ((p + PAGE + 8) as *mut u8).write(1) };
        let (now, after) = writes(start, end, false).expect("watched");
        let new = |w: Vec<(usize, u64)>| w.into_iter().filter(|w| w.1 > none).collect::<Vec<_>>();
        assert_eq!(new(after), [(p / PAGE + 1, now)]);
        // Another asker after the kernel's bit was cleared still sees it.
        let (later, again) = writes(p + PAGE, p + 2 * PAGE, false).expect("watched");
        assert!(later > now);
        assert_eq!(new(again), [(p / PAGE + 1, now)]);
        let mut plain = vec![0u8; PAGE];
        assert!(
            writes(
                plain.as_mut_ptr() as usize,
                plain.as_ptr() as usize + PAGE,
                true
            )
            .is_none()
        );
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// A kernel read into a guarded view block fails; opened first, it
    /// lands and the block counts as written.
    #[test]
    fn kernel_reads_into_guarded_blocks_after_opening() {
        use std::io::{Read, Seek, Write};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile,
        };
        // SAFETY: a pagefile section of two blocks, mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK).Value as usize
        };
        assert_ne!(view, 0);
        assert!(guard(view, view + 2 * BLOCK).is_some()); // now read-only
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"from the disk").unwrap();
        // `ReadFile` inside: the kernel writes the view.
        let read = |file: &mut std::fs::File| {
            file.rewind().unwrap();
            // SAFETY: 13 bytes of our view, nothing else refers to them.
            let into = unsafe { std::slice::from_raw_parts_mut((view + 100) as *mut u8, 13) };
            file.read(into).is_ok_and(|n| n == 13)
        };
        assert!(
            !read(&mut file),
            "the kernel does not fault into our handler"
        );
        writing(view + 100, 13);
        assert!(read(&mut file));
        // SAFETY: the bytes just read.
        let landed = unsafe { std::slice::from_raw_parts((view + 100) as *const u8, 13) };
        assert_eq!(landed, b"from the disk");
        assert_eq!(
            guard(view, view + 2 * BLOCK),
            Some((view / PAGE..view / PAGE + 1).collect())
        );
        // Before the view goes, as the runtime does: the address may be
        // another test's the moment it is free.
        title_memory_unmapping(view, 2 * BLOCK);
        // SAFETY: our view.
        unsafe {
            UnmapViewOfFile(
                windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: view as *mut _,
                },
            )
        };
    }

    /// A placeholder of `blocks` blocks, each its own.
    fn placeholder(blocks: usize) -> usize {
        use windows_sys::Win32::System::Memory::{
            MEM_PRESERVE_PLACEHOLDER, MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: a fresh placeholder, split.
        unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                blocks * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            for block in 0..blocks - 1 {
                assert_ne!(
                    VirtualFree((p + block * BLOCK) as *mut _, BLOCK, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER),
                    0
                );
            }
            p
        }
    }

    /// Private memory of `protect` in place of the placeholder of one block.
    fn private_at(at: usize, protect: u32) {
        use windows_sys::Win32::System::Memory::{MEM_REPLACE_PLACEHOLDER, VirtualAlloc2};
        // SAFETY: our placeholder.
        let got = unsafe {
            VirtualAlloc2(
                windows_sys::Win32::System::Threading::GetCurrentProcess(),
                at as *const _,
                BLOCK,
                MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER,
                protect,
                std::ptr::null_mut(),
                0,
            )
        };
        assert_eq!(got as usize, at);
    }

    fn protection(at: usize) -> u32 {
        use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
        // SAFETY: plain data the query fills in.
        unsafe {
            let mut info: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
            VirtualQuery(at as *const _, &mut info, size_of_val(&info));
            info.Protect
        }
    }

    /// A refresh between the opening of a kernel read's destination and the
    /// read itself leaves it open: the read lands, and is seen.
    #[test]
    fn a_refresh_leaves_open_what_the_kernel_is_writing() {
        use std::io::{Read, Seek, Write};
        use windows_sys::Win32::System::Memory::{CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile};
        // SAFETY: a pagefile section of two blocks, mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK)
        };
        let at = view.Value as usize;
        assert_ne!(at, 0);
        let pages = |block: usize| at / PAGE + block..at / PAGE + block + 1;
        assert!(writes(at, at + 2 * BLOCK, true).is_some());
        assert!(published(at, at + 2 * BLOCK) || !light());
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"from the disk").unwrap();
        file.rewind().unwrap();
        let writing = kernel_writing(at + 100, 13);
        // The refresh: the block being written is new to it every time, and
        // stays open; the other is as it was.
        for _ in 0..2 {
            assert_eq!(guard(at, at + 2 * BLOCK), Some(pages(0).collect()));
            let (now, found) = writes(at, at + BLOCK, true).expect("a view");
            assert!(found.contains(&(at / PAGE, now)));
            assert!(writable(at) && !writable(at + BLOCK));
            assert!(!published(at, at + BLOCK));
        }
        // SAFETY: 13 bytes of our view, nothing else refers to them.
        let into = unsafe { std::slice::from_raw_parts_mut((at + 100) as *mut u8, 13) };
        assert_eq!(file.read(into).expect("the kernel's write lands"), 13);
        assert_eq!(into, b"from the disk");
        drop(writing);
        // Seen by the next refresh, and guarded again.
        assert_eq!(guard(at, at + 2 * BLOCK), Some(pages(0).collect()));
        assert!(!writable(at));
        assert_eq!(guard(at, at + 2 * BLOCK), Some(Vec::new()));
        title_memory_unmapping(at, 2 * BLOCK);
        // SAFETY: our view.
        unsafe { UnmapViewOfFile(view) };
    }

    /// A hole the kernel is to write is no known hole meanwhile, nor is one
    /// the runtime has begun to change; after that it is again.
    #[test]
    fn what_is_left_alone_is_no_known_hole() {
        let p = placeholder(3);
        let known = |block: usize| published(p + block * BLOCK, p + (block + 1) * BLOCK);
        let ask = || writes(p, p + 3 * BLOCK, true).expect("as a view");
        ask();
        let holes = known_holes();
        assert!((0..3).all(known) == holes);
        let writing = kernel_writing(p + BLOCK + 5, 100);
        for _ in 0..2 {
            assert!(known(0) == holes && !known(1) && known(2) == holes);
            let (now, written) = ask();
            assert!(written.contains(&(p / PAGE + 1, now)));
            assert!(written.iter().all(|w| w.0 == p / PAGE + 1 || w.1 < now));
        }
        assert!(!guards().held.contains(p / BLOCK + 1));
        drop(writing);
        ask();
        assert!((0..3).all(known) == holes);
        {
            let _changing = title_memory_remapping();
            title_memory_unmapping(p, BLOCK);
            ask();
            assert!(!known(0) && !guards().held.contains(p / BLOCK));
        }
        ask();
        assert!((0..3).all(known) == holes);
        title_memory_freed(p, BLOCK);
        assert!(!known(0) && !guards().held.contains(p / BLOCK));
        title_memory_unmapping(p, 3 * BLOCK);
        // SAFETY: our placeholders.
        unsafe {
            for block in 0..3 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
        }
    }

    /// A refresh during a change of mapping guards nothing the runtime has
    /// begun to unmap: no note of it stays behind.
    #[test]
    fn nothing_is_guarded_between_our_forgetting_it_and_its_unmapping() {
        use windows_sys::Win32::System::Memory::{CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile};
        // SAFETY: a pagefile section of a block, mapped read-write.
        let view = unsafe {
            let section =
                CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE, 0, BLOCK as u32, std::ptr::null());
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, BLOCK)
        };
        let at = view.Value as usize;
        assert_ne!(at, 0);
        assert!(writes(at, at + BLOCK, true).is_some());
        let changing = title_memory_remapping();
        title_memory_unmapping(at, BLOCK);
        assert!(writes(at, at + BLOCK, true).is_some());
        {
            let g = guards();
            assert!(!g.protected.contains(at / BLOCK) && !g.clean.contains(at / BLOCK));
        }
        assert!(!published(at, at + BLOCK));
        // SAFETY: our view.
        unsafe { UnmapViewOfFile(view) };
        drop(changing);
        assert!(!guards().left_alone(at / BLOCK));
    }

    /// Memory at an address where blocks were guarded or absent is not
    /// taken for them: code stays code, private memory is not guarded.
    #[test]
    fn other_memory_at_a_guarded_address_keeps_its_protection() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER, MapViewOfFile3, PAGE_EXECUTE_READ,
            PAGE_EXECUTE_READWRITE, UnmapViewOfFile2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        let p = placeholder(3);
        // SAFETY: a section's block in place of the first placeholder.
        let view = unsafe {
            let section =
                CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE, 0, BLOCK as u32, std::ptr::null());
            assert!(!section.is_null());
            MapViewOfFile3(
                section,
                GetCurrentProcess(),
                p as *const _,
                0,
                BLOCK,
                MEM_REPLACE_PLACEHOLDER,
                PAGE_READWRITE,
                std::ptr::null_mut(),
                0,
            )
        };
        assert_eq!(view.Value as usize, p);
        // One block guarded, two absent.
        assert_eq!(guard(p, p + 3 * BLOCK).map(|w| w.len()), Some(3));
        // Not told of what comes: the view goes, code is there; and private
        // memory in the holes.
        // SAFETY: our view.
        assert_ne!(unsafe { UnmapViewOfFile2(GetCurrentProcess(), view, MEM_PRESERVE_PLACEHOLDER) }, 0);
        private_at(p, PAGE_EXECUTE_READ);
        private_at(p + BLOCK, PAGE_READWRITE);
        private_at(p + 2 * BLOCK, PAGE_EXECUTE_READWRITE);
        // (The runtime's allocations it tells of, or the holes stay known.)
        title_memory_allocated(p + BLOCK, 2 * BLOCK);
        let as_made = || {
            assert_eq!(protection(p), PAGE_EXECUTE_READ);
            assert_eq!(protection(p + BLOCK), PAGE_READWRITE);
            assert_eq!(protection(p + 2 * BLOCK), PAGE_EXECUTE_READWRITE);
        };
        let holes = [p / PAGE + 1, p / PAGE + 2];
        for _ in 0..2 {
            let written = guard(p, p + 3 * BLOCK).expect("as a view");
            assert!(holes.iter().all(|page| written.contains(page)), "copied every time");
            as_made();
            let g = guards();
            assert!(holes.iter().all(|&block| !g.held.contains(block) && !g.absent.contains(block)));
        }
        writing(p, BLOCK);
        as_made();
        title_memory_unmapping(p, 3 * BLOCK);
        // SAFETY: our allocations.
        unsafe {
            for block in 0..3 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
        }
    }

    /// Memory the title frees leaves no note of its blocks, whether it names
    /// its length or the allocation.
    #[test]
    fn freed_memory_leaves_no_notes() {
        let p = placeholder(3);
        let noted = |block: usize| {
            let g = guards();
            let block = p / BLOCK + block;
            g.protected.contains(block) || g.clean.contains(block) || g.absent.contains(block)
        };
        assert!(guard(p, p + 3 * BLOCK).is_some());
        assert!(noted(0) && noted(1) && noted(2));
        // The middle allocation.
        title_memory_freed(p + BLOCK, 0);
        assert!(noted(0) && !noted(1) && noted(2));
        title_memory_freed(p, BLOCK);
        assert!(!noted(0) && noted(2));
        // Nothing noted there: nothing to do, and nothing beside it.
        title_memory_freed(p + BLOCK, 0);
        assert!(noted(2));
        title_memory_freed(p + 2 * BLOCK, 0);
        assert!(!noted(2));
        // SAFETY: our placeholders.
        unsafe {
            for block in 0..3 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
        }
    }

    /// A write between another thread's asking about a range and the second
    /// part of a submission's one asking is later than what that thread
    /// copied.
    #[test]
    fn the_rest_of_the_one_asking_has_a_number_of_its_own() {
        let _one = one_asking_test();
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe {
            VirtualAlloc(std::ptr::null(), 4 * PAGE, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE)
        } as usize;
        assert_ne!(p, 0);
        // One range: `ask_ahead` leaves it.
        let ranges = [(p, p + 4 * PAGE)];
        ask_ahead(ranges.into_iter());
        let (copied, _) = writes(p, p + 4 * PAGE, false).expect("watched");
        // SAFETY: committed pages of ours.
        unsafe { ((p + PAGE) as *mut u8).write_volatile(1) };
        ask_rest(&ranges, false, changes());
        let last = journal().notes.range(p / PAGE + 1, p / PAGE + 2).first().map(|n| n.1);
        ask_once_end();
        assert!(last.is_some_and(|last| last > copied), "{last:?} is not after {copied}");
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    #[test]
    fn a_raw_copy_in_watched_memory_keeps_the_other_asked_ranges() {
        let _one = one_asking_test();
        let reading = title_memory_reading();
        // SAFETY: a private write-watched allocation owned by the test.
        let p = unsafe {
            VirtualAlloc(std::ptr::null(), 4 * PAGE, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE)
        } as usize;
        assert_ne!(p, 0);
        let other = (p + 2 * PAGE, p + 3 * PAGE);
        ask_once(&[(p, p + PAGE), other]);
        let (old, other_notes) = asked(other.0, other.1).expect("queried before the copy");
        assert!(prepare_raw_copy_note(&reading, other.1, 1).is_none());
        let note = prepare_raw_copy_note(&reading, p, 1).expect("watched destination");
        // SAFETY: one byte in the allocation above.
        unsafe { (p as *mut u8).write_volatile(0x53) };
        assert!(note.written(&reading));
        let (now, notes) = asked(p, p + PAGE).expect("the query still covers the destination");
        assert!(now > old);
        assert!(notes.contains(&(p / PAGE, now)));
        assert_eq!(asked(other.0, other.1), Some((now, other_notes)));
        assert!(kernel_written(p / PAGE, p / PAGE + 1).is_some_and(|pages| !pages.is_empty()));
        drop(reading);
        {
            use windows_sys::Win32::System::Memory::MEM_DECOMMIT;
            let _changing = title_memory_remapping();
            title_memory_freed(p, 4 * PAGE);
            // SAFETY: decommit and recommit the test's reservation in place.
            unsafe {
                assert_ne!(VirtualFree(p as *mut _, 4 * PAGE, MEM_DECOMMIT), 0);
                assert_eq!(VirtualAlloc(p as *const _, 4 * PAGE, MEM_COMMIT, PAGE_READWRITE) as usize, p);
                (other.0 as *mut u8).write_volatile(0x65);
            }
        }
        assert!(asked(other.0, other.1).is_none());
        assert!(writes(other.0, other.1, false).is_some_and(|(_, pages)| {
            pages.iter().any(|&(page, wrote)| page == other.0 / PAGE && wrote > now)
        }));
        ask_once_end();

        // A stale address in the index is not proof of a current watched allocation.
        let reading = title_memory_reading();
        // SAFETY: another private allocation, with no write watch enabled.
        let plain = unsafe { VirtualAlloc(std::ptr::null(), PAGE, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) } as usize;
        assert_ne!(plain, 0);
        {
            let mut j = journal_mut();
            j.ahead = Ahead::from_ranges(vec![(plain / PAGE, plain / PAGE + 1)]);
            j.ahead_mapping = changes();
            j.kept = true;
        }
        assert!(prepare_raw_copy_note(&reading, plain, 1).is_none());
        ask_once_end();
        // SAFETY: freed once, after all test queries.
        unsafe {
            VirtualFree(plain as *mut _, 0, MEM_RELEASE);
            VirtualFree(p as *mut _, 0, MEM_RELEASE);
        }
    }

    #[test]
    #[ignore = "benchmark"]
    fn a_raw_copy_between_freshly_queried_ranges() {
        let _one = one_asking_test();
        let reading = title_memory_reading();
        let ranges: Vec<_> = (0..16)
            .map(|_| {
                // SAFETY: separate write-watched allocations owned by the benchmark.
                let p = unsafe {
                    VirtualAlloc(std::ptr::null(), PAGE, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE)
                } as usize;
                assert_ne!(p, 0);
                (p, p + PAGE)
            })
            .collect();
        let dst = ranges[0].0;
        let mut samples = [Vec::new(), Vec::new()];
        for trial in 0..256 {
            for keep in [trial % 2 == 0, trial % 2 != 0] {
                ask_once(&ranges);
                let began = std::time::Instant::now();
                let note = keep.then(|| prepare_raw_copy_note(&reading, dst, 1).expect("watched destination"));
                // SAFETY: the first byte of the benchmark's first allocation.
                unsafe { (dst as *mut u8).write_volatile(trial as u8) };
                if let Some(note) = note {
                    assert!(note.written(&reading));
                } else {
                    ask_once_end();
                }
                for &(start, end) in &ranges {
                    std::hint::black_box(writes(start, end, false).expect("watched"));
                }
                samples[usize::from(keep)].push(began.elapsed().as_secs_f64() * 1e6);
                ask_once_end();
            }
        }
        for sample in &mut samples {
            sample.sort_by(f64::total_cmp);
        }
        println!("raw copy and 16 refresh queries: reask {:.2} us, keep {:.2} us (medians)", samples[0][128], samples[1][128]);
        for (start, _) in ranges {
            // SAFETY: freed once, after all benchmark queries.
            unsafe { VirtualFree(start as *mut _, 0, MEM_RELEASE) };
        }
    }

    /// The runtime maps a view and then tells us; a copy's refresh on another
    /// thread can guard it in between. Told, it is the title's to write: a
    /// block left read-only and forgotten would be an access violation in
    /// title code.
    #[test]
    fn a_block_guarded_before_we_were_told_of_its_view_can_be_written() {
        use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile,
        };
        // SAFETY: a pagefile section of two blocks, mapped read-write; the
        // view holds it.
        let view = unsafe {
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let view = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK);
            CloseHandle(section);
            view
        };
        let at = view.Value as usize;
        assert_ne!(at, 0);
        // The refresh, then the runtime's word.
        assert!(guard(at, at + 2 * BLOCK).is_some());
        assert!(!writable(at) && !writable(at + BLOCK));
        title_memory_remapped(at, 2 * BLOCK);
        assert!(writable(at) && writable(at + BLOCK));
        // SAFETY: inside the view; a fault would end the process.
        unsafe { ((at + 8) as *mut u8).write_volatile(1) };
        // Looked at again as new, and guarded again.
        assert_eq!(
            guard(at, at + 2 * BLOCK),
            Some((at / PAGE..at / PAGE + 2).collect())
        );
        assert!(!writable(at));
        title_memory_unmapping(at, 2 * BLOCK);
        // SAFETY: our view.
        unsafe { UnmapViewOfFile(view) };
    }

    #[test]
    fn writes_in_partly_committed_view_blocks_are_reported() {
        use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            SEC_RESERVE, UnmapViewOfFile,
        };

        const SIZE: usize = 3 * BLOCK;
        struct View(MEMORY_MAPPED_VIEW_ADDRESS);
        impl Drop for View {
            fn drop(&mut self) {
                title_memory_unmapping(self.0.Value as usize, SIZE);
                // SAFETY: our view is still mapped.
                unsafe { UnmapViewOfFile(self.0) };
            }
        }
        // SAFETY: a new reserved pagefile section. The view holds its own
        // reference after we close the handle, including on assertion failure.
        let view = unsafe {
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE | SEC_RESERVE,
                0,
                SIZE as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let view = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, SIZE);
            CloseHandle(section);
            assert!(!view.Value.is_null());
            View(view)
        };
        let start = view.0.Value as usize;
        let commit = |at: usize, len| {
            // SAFETY: these pages belong to our reserved view.
            let committed = unsafe { VirtualAlloc(at as _, len, MEM_COMMIT, PAGE_READWRITE) };
            assert_eq!(committed as usize, at);
        };
        let early = start + BLOCK + 3 * KERNEL_PAGE;
        let late = start + 2 * BLOCK + 3 * KERNEL_PAGE;
        commit(start, BLOCK);
        commit(early, KERNEL_PAGE);
        let refresh = || guard(start, start + SIZE).unwrap();
        assert_eq!(refresh(), [start / PAGE, early / PAGE, late / PAGE]);
        // The last block is still entirely reserved.
        assert!(!refresh().contains(&(late / PAGE)));

        commit(late, KERNEL_PAGE); // Ordinary commit, without a runtime mapping notification.
        for at in [early, late] {
            assert!(writable(at));
            // SAFETY: each address is inside its committed read-write page.
            unsafe { (at as *mut u8).write_volatile(1) };
        }
        assert_eq!(refresh(), [early / PAGE, late / PAGE]);
    }

    /// A range of view memory whose first block is not mapped is a view
    /// all the same: its blocks are reported once, then the ones written
    /// or mapped since.
    #[test]
    fn a_range_that_begins_in_an_unmapped_block_is_guarded() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, UnmapViewOfFile2,
            VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: a fresh placeholder of three blocks and a pagefile
        // section of three; the middle block is mapped first.
        let (p, section) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                3 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (3 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            (p, section)
        };
        let map = |block: usize| {
            // SAFETY: splits the placeholder at the block and maps the
            // section's block there.
            let view = unsafe {
                VirtualFree(
                    (p + block * BLOCK) as *mut _,
                    BLOCK,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                );
                MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    (p + block * BLOCK) as *const _,
                    (block * BLOCK) as u64,
                    BLOCK,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(view.Value as usize, p + block * BLOCK);
            view
        };
        let pages = |block: usize| (p + block * BLOCK) / PAGE..(p + (block + 1) * BLOCK) / PAGE;
        let all: Vec<usize> = (0..3).flat_map(pages).collect();
        // Nothing mapped at all: a view with every block absent.
        assert!(is_view(p, p + 3 * BLOCK));
        let middle = map(1);
        title_memory_remapped(p + BLOCK, BLOCK);
        // The range begins in the placeholder, its middle block is a view.
        assert!(is_view(p, p + 3 * BLOCK));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(all));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()), "nothing written since");
        // SAFETY: inside the mapped middle block.
        unsafe { ((p + BLOCK + 8) as *mut u8).write_volatile(1) };
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(1).collect()));
        let first = map(0);
        title_memory_remapped(p, BLOCK);
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(0).collect()));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()));
        // Memory of ours that is no view stays none.
        let mut local = [0u8; 16];
        let at = local.as_mut_ptr() as usize;
        assert!(!is_view(at, at + local.len()));
        title_memory_unmapping(p, 3 * BLOCK);
        // SAFETY: our views and placeholder.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), first, MEM_PRESERVE_PLACEHOLDER);
            UnmapViewOfFile2(GetCurrentProcess(), middle, MEM_PRESERVE_PLACEHOLDER);
            for block in 0..3 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
    }

    /// Blocks of a placeholder between views are reported once, then not
    /// again (they cannot be guarded) until the runtime maps them.
    #[test]
    fn unmapped_blocks_are_reported_once_until_mapped() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, UnmapViewOfFile2,
            VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: a fresh placeholder of three blocks and a pagefile
        // section of two; block 0 shows the section's first block, as the
        // runtime maps physical pages.
        let (p, section) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                3 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            (p, section)
        };
        let map = |block: usize| {
            // SAFETY: splits the placeholder at the block and maps the
            // section's block there.
            let view = unsafe {
                VirtualFree(
                    (p + block * BLOCK) as *mut _,
                    BLOCK,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                );
                MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    (p + block * BLOCK) as *const _,
                    (block * BLOCK) as u64,
                    BLOCK,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(view.Value as usize, p + block * BLOCK);
            view
        };
        let first = map(0);
        let pages = |block: usize| (p + block * BLOCK) / PAGE..(p + (block + 1) * BLOCK) / PAGE;
        let all: Vec<usize> = (0..3).flat_map(pages).collect();
        assert_eq!(guard(p, p + 3 * BLOCK), Some(all));
        assert_eq!(
            guard(p, p + 3 * BLOCK),
            Some(Vec::new()),
            "nothing written since"
        );
        let second = map(1);
        title_memory_remapped(p + BLOCK, BLOCK);
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(1).collect()));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()));
        title_memory_unmapping(p, 3 * BLOCK);
        // SAFETY: our views and placeholder.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), first, MEM_PRESERVE_PLACEHOLDER);
            UnmapViewOfFile2(GetCurrentProcess(), second, MEM_PRESERVE_PLACEHOLDER);
            for block in 0..3 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
    }

    /// A pagefile section mapped twice: two views of the same physical
    /// pages, the second left for the test; the section handle, the two
    /// addresses.
    fn two_views(bytes: usize) -> (usize, usize, usize) {
        use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile,
        };
        // SAFETY: a fresh pagefile section, mapped twice; the views keep
        // the section alive and the handle is closed by the caller.
        unsafe {
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                bytes as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let one = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, bytes).Value as usize;
            let two = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, bytes).Value as usize;
            assert_ne!(one, 0);
            assert_ne!(two, 0);
            crate::inplace::title_memory_view(one, bytes, section as usize, 0);
            crate::inplace::title_memory_view(two, bytes, section as usize, 0);
            (section as usize, one, two)
        }
    }

    /// The section of [`two_views`] is closed and its views unmapped; the
    /// guard is told they are gone first.
    fn drop_views(section: usize, views: &[usize], bytes: usize) {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Memory::{MEMORY_MAPPED_VIEW_ADDRESS, UnmapViewOfFile};
        for &view in views {
            title_memory_remapped(view, bytes);
            // SAFETY: a view of the section mapped by `two_views`.
            unsafe {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: view as *mut _,
                })
            };
        }
        // SAFETY: the section's last reference.
        unsafe { CloseHandle(section as *mut std::ffi::c_void) };
    }

    /// The title wrote the pages of a guarded texture through another view
    /// of them: the fault at that view is the guard's to see, and the
    /// guarded view's next check reports the pages written.
    #[test]
    fn a_write_through_another_view_is_reported_at_the_guarded_one() {
        const SIZE: usize = 2 * BLOCK;
        let (section, guarded, other) = two_views(SIZE);
        let pages = |block: usize| (guarded + block * BLOCK) / PAGE;
        // The first check takes it in (and every view of its pages).
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()), "nothing written yet");
        // Writing through the other view faults into the handler, which
        // opens that view and the guarded one: the next check sees it.
        // SAFETY: the second view, read-write once the handler has opened it.
        unsafe { ((other + BLOCK + 8) as *mut u8).write(1) };
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()));
        drop_views(section, &[guarded, other], SIZE);
    }

    /// A view of guarded pages is mapped anew: it is writable, so the
    /// guarded view checks its pages again (and guards the new view).
    #[test]
    fn a_view_mapped_beside_a_guarded_one_makes_it_check_again() {
        const SIZE: usize = 2 * BLOCK;
        let (section, guarded, other) = two_views(SIZE);
        let pages = |block: usize| (guarded + block * BLOCK) / PAGE;
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()), "nothing written yet");
        title_memory_remapped(other, SIZE);
        crate::inplace::title_memory_view(other, SIZE, section, 0);
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()));
        drop_views(section, &[guarded, other], SIZE);
    }

    /// A write of the runtime's own, told at one view of the pages
    /// (`writing`): the other views' next checks report the pages written.
    #[test]
    fn a_write_told_at_one_view_is_reported_at_the_others() {
        const SIZE: usize = 2 * BLOCK;
        let (section, guarded, other) = two_views(SIZE);
        let pages = |block: usize| (guarded + block * BLOCK) / PAGE;
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()), "nothing written yet");
        writing(other + BLOCK, BLOCK);
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()));
        // Half a page and the next: both blocks, not just the first's.
        writing(other + BLOCK / 2, BLOCK);
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        drop_views(section, &[guarded, other], SIZE);
    }

    /// Both views of the pages are checked, each for a copy of its own
    /// (the title shows one physical page at several addresses, and each
    /// address has its host copy): a write through either is reported at
    /// both, whichever is checked first. Guarding the pages for the first
    /// view's check must not note the second view clean, or its copy would
    /// keep the bytes from before the write.
    #[test]
    fn a_write_is_reported_at_every_checked_view() {
        const SIZE: usize = 2 * BLOCK;
        let (section, one, two) = two_views(SIZE);
        let pages = |view: usize, block: usize| (view + block * BLOCK) / PAGE;
        assert_eq!(guard(one, one + SIZE), Some(vec![pages(one, 0), pages(one, 1)]));
        assert_eq!(guard(two, two + SIZE), Some(vec![pages(two, 0), pages(two, 1)]), "the second view's first check");
        assert_eq!(guard(one, one + SIZE), Some(Vec::new()), "nothing written yet");
        assert_eq!(guard(two, two + SIZE), Some(Vec::new()), "nothing written yet");
        // SAFETY: the second view, read-write once the handler has opened it.
        unsafe { ((two + BLOCK + 8) as *mut u8).write(1) };
        assert_eq!(guard(one, one + SIZE), Some(vec![pages(one, 1)]));
        assert_eq!(guard(two, two + SIZE), Some(vec![pages(two, 1)]), "written through this view, checked second");
        // SAFETY: the first view, as above.
        unsafe { ((one + 8) as *mut u8).write(1) };
        assert_eq!(guard(two, two + SIZE), Some(vec![pages(two, 0)]));
        assert_eq!(guard(one, one + SIZE), Some(vec![pages(one, 0)]), "written through this view, checked second");
        assert_eq!(guard(one, one + SIZE), Some(Vec::new()));
        assert_eq!(guard(two, two + SIZE), Some(Vec::new()));
        drop_views(section, &[one, two], SIZE);
    }

    /// A write opens the view it goes through and no other: the others stay
    /// read-only, are told as written, and take writes of their own.
    #[test]
    fn a_write_opens_the_view_it_goes_through() {
        let (section, one, two) = two_views(BLOCK);
        let page = |view: usize| view / PAGE;
        for view in [one, two] {
            assert_eq!(guard(view, view + BLOCK), Some(vec![page(view)]));
        }
        // SAFETY (all): a view, opened by the handler.
        unsafe { ((two + 8) as *mut u8).write(1) };
        assert!(writable(two) && !writable(one));
        unsafe { ((one + 8) as *mut u8).write(1) };
        assert!(writable(two) && writable(one));
        assert_eq!(guard(one, one + BLOCK), Some(vec![page(one)]));
        assert_eq!(guard(two, two + BLOCK), Some(vec![page(two)]));
        // Written at the view that was left as it was, after the other.
        unsafe { ((two + 8) as *mut u8).write(1) };
        assert_eq!(guard(one, one + BLOCK), Some(vec![page(one)]), "not clean, though never opened");
        assert!(!writable(two) && !writable(one));
        assert_eq!(guard(two, two + BLOCK), Some(vec![page(two)]));
        assert_eq!(guard(one, one + BLOCK), Some(Vec::new()));
        // Of a page that was told at one view, a kernel write to the other
        // finds it open.
        unsafe { ((two + 8) as *mut u8).write(1) };
        assert!(!writable(one));
        writing(one, 8);
        assert!(writable(one));
        drop_views(section, &[one, two], BLOCK);
    }

    /// A block the title has just written is not made read-only, and is told
    /// as written at each asking, at every view of it, until it has rested.
    #[test]
    fn a_block_just_written_rests() {
        let (section, one, two) = two_views(BLOCK);
        let page = one / PAGE;
        let rests = |rests: bool| REST_IN_TEST.set(std::time::Duration::from_secs(if rests { 3600 } else { 0 }));
        assert_eq!(guard(one, one + BLOCK), Some(vec![page]));
        assert_eq!(guard(one, one + BLOCK), Some(Vec::new()));
        rests(true);
        assert_eq!(guard(one, one + BLOCK), Some(Vec::new()), "nothing written: as it was");
        // SAFETY (all): a view, opened by the handler.
        unsafe { ((two + 8) as *mut u8).write(1) };
        for _ in 0..3 {
            assert_eq!(guard(one, one + BLOCK), Some(vec![page]));
            assert!(writable(two), "no fault for the next write");
            assert!(writes(two, two + BLOCK, true).is_some() && !published(two, two + BLOCK) && !published(one, one + BLOCK));
            unsafe { ((two + 8) as *mut u8).write(1) };
        }
        rests(false);
        assert_eq!(guard(one, one + BLOCK), Some(vec![page]));
        assert!(!writable(two) && !writable(one));
        assert_eq!(guard(one, one + BLOCK), Some(Vec::new()));
        drop_views(section, &[one, two], BLOCK);
    }

    /// What is put in beside a guard is told at every view, once, and the
    /// block stays read-only; until it has had its share.
    #[test]
    fn what_the_title_puts_in_beside_a_guard_is_told() {
        let (section, one, two) = two_views(BLOCK);
        crate::title_memory_section(section, BLOCK as u64, true);
        let page = |view: usize| view / PAGE;
        for view in [one, two] {
            assert_eq!(guard(view, view + BLOCK), Some(vec![page(view)]));
        }
        assert!(writes(one, one + BLOCK, true).is_some() && writes(two, two + BLOCK, true).is_some());
        put_count(one / BLOCK).store(0, std::sync::atomic::Ordering::Relaxed);
        let put = |byte: u8| loop {
            // SAFETY: a byte of the section's only page.
            let done = put_beside(one / BLOCK, 1, |delta| unsafe { ((one + 100).wrapping_add(delta) as *mut u8).write(byte) });
            // (Other tests hold changes of mapping.)
            if done != Beside::Cannot {
                return done;
            }
        };
        for round in 1..4 {
            assert_eq!(put(round), Beside::Done);
            // SAFETY: the view the title reads it by.
            assert_eq!(unsafe { ((two + 100) as *const u8).read_volatile() }, round);
            assert!(!writable(one) && !writable(two) && !published(one, one + BLOCK) && !published(two, two + BLOCK));
            for view in [two, one] {
                assert_eq!(guard(view, view + BLOCK), Some(vec![page(view)]));
                assert_eq!(guard(view, view + BLOCK), Some(Vec::new()));
            }
        }
        for _ in 0..crate::titlewrite::MOST {
            assert_eq!(put(9), Beside::Done);
        }
        assert_eq!(put(9), Beside::Full);
        // A refresh between gives it its share again.
        assert_eq!(guard(one, one + BLOCK), Some(vec![page(one)]));
        assert_eq!(put(9), Beside::Done);
        // Opened by a fault, it is told by that.
        assert_eq!(guard(one, one + BLOCK), Some(vec![page(one)]));
        // SAFETY: a view, opened by the handler.
        unsafe { ((one + 8) as *mut u8).write(1) };
        assert_eq!(put_beside(one / BLOCK, 1, |_| panic!("it is not guarded")), Beside::Cannot);
        assert_eq!(guard(one, one + BLOCK), Some(vec![page(one)]));
        assert_eq!(guard(one, one + BLOCK), Some(Vec::new()));
        drop_views(section, &[one, two], BLOCK);
        crate::title_memory_section_gone(section);
    }

    /// Stores that fault, and copies, as the title makes them: what is there
    /// after, the block's guard and what the next asking says.
    #[test]
    fn stores_and_copies_of_the_title_leave_the_guard() {
        use std::sync::atomic::{AtomicU32, Ordering};
        unsafe extern "system-unwind" fn real(to: *mut u8, from: *const u8, n: usize) -> *mut u8 {
            // SAFETY: by the caller.
            unsafe { std::ptr::copy(from, to, n) };
            to
        }
        const SIZE: usize = 2 * BLOCK;
        let (section, one, two) = two_views(SIZE);
        crate::title_memory_section(section, SIZE as u64, true);
        let pages = |view: usize| vec![view / PAGE, view / PAGE + 1];
        assert_eq!(guard(one, one + SIZE), Some(pages(one)));
        assert_eq!(guard(two, two + SIZE), Some(pages(two)));
        let copy = crate::titlewrite::import("memcpy", real as *const () as usize);
        crate::titlewrite::ON_IN_TEST.set(true);
        let copy = if copy == real as *const () as usize {
            crate::titlewrite::copy_for_test(real as *const () as usize)
        } else {
            copy
        };
        // SAFETY: of that signature.
        let copy: unsafe extern "system-unwind" fn(*mut u8, *const u8, usize) -> *mut u8 = unsafe { std::mem::transmute(copy) };
        // No change of mapping meanwhile but the other tests' moments of it:
        // a store that meets one opens the block, as before.
        let mut kept = 0;
        let quiet = || {
            while title_memory_reading_now().is_none() {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        };
        for round in 1..200u32 {
            quiet();
            // SAFETY (all): words of the view.
            unsafe { ((one + 64) as *mut u32).write_volatile(round) };
            let was = unsafe { AtomicU32::from_ptr((two + 64) as *mut u32) }.fetch_sub(1, Ordering::SeqCst);
            assert_eq!((was, unsafe { ((one + 64) as *const u32).read_volatile() }), (round, round - 1));
            let bytes = vec![round as u8; 3000 + round as usize];
            // Over the end of the first block.
            assert_eq!(unsafe { copy((one + BLOCK - 1000) as *mut u8, bytes.as_ptr(), bytes.len()) } as usize, one + BLOCK - 1000);
            let there = unsafe { std::slice::from_raw_parts((two + BLOCK - 1000) as *const u8, bytes.len()) };
            assert!(there == bytes && unsafe { ((two + BLOCK + 2000 + round as usize) as *const u8).read_volatile() } != round as u8);
            kept += usize::from(!writable(one) && !writable(two));
            for view in [one, two] {
                assert_eq!(guard(view, view + SIZE), Some(pages(view)), "round {round}");
                assert_eq!(guard(view, view + SIZE), Some(Vec::new()), "round {round}");
            }
        }
        assert!(kept > 10, "the guard stayed {kept} times: {:?}", crate::titlewrite::DONE);
        // What is there already is not written: nothing to tell.
        for k in 0..10 {
            // SAFETY (all): as above.
            unsafe { ((one + 64) as *mut u32).write_volatile(198) };
            let same = unsafe { std::slice::from_raw_parts((one + BLOCK - 900) as *const u8, 1500) }.to_vec();
            unsafe { copy((one + BLOCK - 900) as *mut u8, same.as_ptr(), same.len()) };
            assert!(!writable(one) && !writable(two), "{k}");
        }
        assert_eq!(guard(one, one + SIZE), Some(Vec::new()));
        // Many stores to a block: it is opened.
        quiet();
        for k in 0..20 {
            unsafe { ((one + 128 + k * 4) as *mut u32).write_volatile(7) };
        }
        assert!(writable(one));
        crate::titlewrite::ON_IN_TEST.set(false);
        drop_views(section, &[one, two], SIZE);
        crate::title_memory_section_gone(section);
    }

    /// Of a page taken before, what is taken again is the kernel's pages
    /// written since; where they are not known, all of it.
    #[test]
    fn what_is_taken_again_is_what_was_written() {
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe {
            VirtualAlloc(std::ptr::null(), 4 * PAGE, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE)
        } as usize;
        assert_ne!(p, 0);
        let write = |at: usize| {
            // SAFETY: committed pages of ours.
            unsafe { ((p + at) as *mut u8).write_volatile(1) }
        };
        let tracked = track(p, 4 * PAGE, false).expect("watched");
        // (Other tests' memory was here.)
        (0..4).for_each(|page| write(page * PAGE + 5));
        assert_eq!(tracked.take(p, p + 4 * PAGE), [(p, p + 4 * PAGE)], "never taken");
        assert!(tracked.take(p, p + 4 * PAGE).is_empty());
        write(PAGE + 5000);
        write(PAGE + 8192);
        write(PAGE + 65535);
        write(2 * PAGE);
        assert_eq!(
            tracked.take(p, p + 4 * PAGE),
            [(p + PAGE + 4096, p + PAGE + 3 * 4096), (p + 2 * PAGE - 4096, p + 2 * PAGE + 4096)]
        );
        write(2 * PAGE + 70);
        // Two askings' writes are in what is taken after both.
        assert!(writes(p, p + 4 * PAGE, false).is_some());
        write(2 * PAGE + 9000);
        assert_eq!(tracked.take(p, p + 4 * PAGE), [(p + 2 * PAGE, p + 2 * PAGE + 4096), (p + 2 * PAGE + 8192, p + 2 * PAGE + 12288)]);
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// The frames pages were last landed in.
    #[test]
    fn landed_pages_are_known_by_their_frame() {
        static LANDED: Landed = Landed([const { std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()) }; Landed::PIECES]);
        let landed = &LANDED;
        for page in [0, 5, Landed::PAGES - 1, Landed::PAGES, 7 * Landed::PAGES + 3] {
            assert!(!landed.since(page, 0, 100));
            landed.note(page, 100);
            assert!(landed.since(page, 100, 100) && landed.since(page, 0, 100) && landed.since(page, 90, 110));
            assert!(!landed.since(page, 101, 110));
            landed.note(page, 0x1_0000_0005);
            assert!(landed.since(page, 0x1_0000_0000, 0x1_0000_0007) && !landed.since(page, 0x1_0000_0006, 0x1_0000_0007));
            assert!(!landed.since(page + 1, 0, 0x1_0000_0007));
        }
        landed.note(usize::MAX / 4096, 1);
        assert!(!landed.since(usize::MAX / 4096, 0, 1));
    }

    /// The set of blocks against another of the same.
    #[test]
    fn blocks_are_a_set() {
        let (set, mut other) = (Blocks::new(), BTreeSet::new());
        let mut seed = 1u64;
        let mut next = |n: usize| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as usize % n
        };
        for round in 0..40_000u32 {
            // About the ends of pieces and words.
            let block = [0, 63, Blocks::IN_PIECE, 3 * Blocks::IN_PIECE - 70][next(4)] + next(140);
            if next(3) == 0 {
                assert_eq!(set.remove(block), other.remove(&block));
            } else {
                assert_eq!(set.insert(block), other.insert(block));
            }
            assert_eq!(set.contains(block), other.contains(&block));
            if round.is_multiple_of(64) {
                let a = [0, 63, Blocks::IN_PIECE, 3 * Blocks::IN_PIECE - 70][next(4)] + next(140);
                let b = a + [next(10), next(300), 2 * Blocks::IN_PIECE + next(100)][next(3)];
                assert!(set.range(a..b).eq(other.range(a..b).copied()), "{a}..{b}");
                assert!(set.range(a..).eq(other.range(a..).copied()) && set.range(..).eq(other.iter().copied()));
                assert!(set.range(a..=b).eq(other.range(a..=b).copied()));
                assert_eq!(set.len(), other.len());
            }
        }
        assert_eq!(set.range(5..usize::MAX).next(), other.range(5..).next().copied());
    }

    /// What is known of a block by its bit is so at every view of its page:
    /// the write through one takes the bits of all.
    #[test]
    fn bits_go_at_every_view() {
        const SIZE: usize = 2 * BLOCK;
        let (section, one, two) = two_views(SIZE);
        let ask = |at: usize| writes(at, at + SIZE, true).expect("a view");
        for at in [one, two, one, two] {
            ask(at);
        }
        assert!(published(one, one + SIZE) && published(two, two + SIZE));
        let (before, _) = ask(two);
        // SAFETY: the first view, opened by the handler.
        unsafe { ((one + BLOCK + 8) as *mut u8).write(1) };
        assert!(published(one, one + BLOCK) && published(two, two + BLOCK), "the other block is as it was");
        assert!(!published(one + BLOCK, one + SIZE) && !published(two + BLOCK, two + SIZE));
        let told = |at: usize| ask(at).1.iter().find(|w| w.0 == (at + BLOCK) / PAGE).map_or(0, |w| w.1);
        assert!(told(two) > before && told(one) > before);
        drop_views(section, &[one, two], SIZE);
    }

    /// While the kernel writes through one view, the same page at another
    /// is told as written at every asking, and the first is not guarded.
    #[test]
    fn what_the_kernel_writes_through_is_not_guarded_for_another_view() {
        let (section, one, two) = two_views(BLOCK);
        let page = one / PAGE;
        assert_eq!(guard(one, one + BLOCK), Some(vec![page]));
        assert_eq!(guard(one, one + BLOCK), Some(Vec::new()));
        let writing = kernel_writing(two, 100);
        for _ in 0..3 {
            assert_eq!(guard(one, one + BLOCK), Some(vec![page]), "the kernel may be writing it");
            assert!(writable(two));
        }
        assert!(!written_since_taken(one, 8), "by whom is not known");
        drop(writing);
        assert_eq!(guard(one, one + BLOCK), Some(vec![page]));
        assert!(!writable(two));
        assert_eq!(guard(one, one + BLOCK), Some(Vec::new()));
        // SAFETY: opened by the handler.
        unsafe { ((two + 8) as *mut u8).write(1) };
        assert_eq!(guard(one, one + BLOCK), Some(vec![page]));
        drop_views(section, &[one, two], BLOCK);
    }

    /// What shaders wrote, put in past the guards and told at one view, is
    /// told at the others, whose copies have not got it; and so is what we
    /// wrote while a submission's one asking stands.
    #[test]
    fn what_is_put_in_beside_the_guards_is_told_at_every_view() {
        let (section, one, two) = two_views(BLOCK);
        writes(two, two + BLOCK, true).expect("a view");
        let (before, _) = writes(two, two + BLOCK, true).expect("a view");
        assert!(published(two, two + BLOCK));
        landed_beside(&[one / 4096 + 3], &[(one + 3 * 4096, 16)]);
        let (_, told) = writes(two, two + BLOCK, true).expect("a view");
        assert!(told.iter().any(|w| w.0 == two / PAGE && w.1 > before), "{told:?} after {before}");
        assert!(landed_lately(two + 3 * 4096, 16, 1) && !landed_lately(two + 4 * 4096, 16, 1));
        drop_views(section, &[one, two], BLOCK);
    }

    /// A view is cut in two by the runtime: until that is done, its pages
    /// count as written wherever else they are shown.
    #[test]
    fn a_view_being_split_leaves_its_pages_unguarded_elsewhere() {
        const SIZE: usize = 3 * BLOCK;
        let (section, one, two) = two_views(SIZE);
        let pages: Vec<usize> = (0..3).map(|k| one / PAGE + k).collect();
        assert_eq!(guard(one, one + SIZE), Some(pages.clone()));
        assert_eq!(guard(one, one + SIZE), Some(Vec::new()));
        let changing = title_memory_remapping();
        title_memory_splitting(two, SIZE);
        title_memory_unmapping(two, SIZE);
        for _ in 0..2 {
            assert_eq!(guard(one, one + SIZE), Some(pages.clone()), "written through the part that stays");
        }
        // The first and last page stay mapped.
        for k in [0, 2] {
            title_memory_remapped(two + k * BLOCK, BLOCK);
            crate::inplace::title_memory_view(two + k * BLOCK, BLOCK, section, (k * BLOCK) as u64);
        }
        drop(changing);
        assert_eq!(guard(one, one + SIZE), Some(pages));
        assert_eq!(guard(one, one + SIZE), Some(Vec::new()));
        assert!(!writable(two) && writable(two + BLOCK) && !writable(two + 2 * BLOCK));
        drop_views(section, &[one, two], SIZE);
    }

    /// A page shown in more than `MANY_VIEWS` views is guarded at each by
    /// itself: the others stay writable, and nothing looks through them all.
    #[test]
    fn a_page_of_many_views_is_guarded_one_view_at_a_time() {
        let (section, one, two) = two_views(BLOCK);
        let filler = one + (1 << 45);
        for n in 0..MANY_VIEWS {
            crate::inplace::title_memory_view(filler + n * BLOCK, BLOCK, section, 0);
        }
        assert_eq!(guard(one, one + BLOCK), Some(vec![one / PAGE]));
        assert!(writable(two), "another view of it is left alone");
        // SAFETY: the guarded view, opened by the handler.
        unsafe { ((one + 8) as *mut u8).write(1) };
        assert_eq!(guard(one, one + BLOCK), Some(vec![one / PAGE]));
        title_memory_remapped(filler, MANY_VIEWS * BLOCK);
        drop_views(section, &[one, two], BLOCK);
    }

    /// A write or mapping that runs past the end of the address space (a
    /// title's `fread` with a huge count) covers up to the end; the sum used
    /// to overflow. Up there no test has guarded blocks.
    #[test]
    fn ranges_past_the_address_space_end_do_not_overflow() {
        let top = usize::MAX - 16;
        writing(top, PAGE);
        title_memory_remapped(top, usize::MAX);
    }
}

#[cfg(test)]
mod mapping_tests {
    use super::{title_memory_reading, title_memory_remapping};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, mpsc};
    use std::time::Duration;

    /// A change of mapping waits for a read in place that is under way, and
    /// a read in place waits for a change of mapping.
    #[test]
    fn reads_in_place_and_remappings_wait_for_each_other() {
        // A reader holds; the change does not begin until it lets go.
        let changed = Arc::new(AtomicBool::new(false));
        let reading = title_memory_reading();
        let (began, wait) = mpsc::channel();
        let change = {
            let changed = changed.clone();
            std::thread::spawn(move || {
                began.send(()).unwrap();
                let _changing = title_memory_remapping();
                changed.store(true, Ordering::SeqCst);
            })
        };
        wait.recv().unwrap();
        std::thread::sleep(Duration::from_millis(100));
        assert!(!changed.load(Ordering::SeqCst), "the mapping changed under a read in place");
        drop(reading);
        change.join().unwrap();
        assert!(changed.load(Ordering::SeqCst));
        // A change holds; the read does not begin until it is done.
        let read = Arc::new(AtomicBool::new(false));
        let changing = title_memory_remapping();
        let (began, wait) = mpsc::channel();
        let reader = {
            let read = read.clone();
            std::thread::spawn(move || {
                began.send(()).unwrap();
                let _reading = title_memory_reading();
                read.store(true, Ordering::SeqCst);
            })
        };
        wait.recv().unwrap();
        std::thread::sleep(Duration::from_millis(100));
        assert!(!read.load(Ordering::SeqCst), "a read in place began during a change of mapping");
        drop(changing);
        reader.join().unwrap();
        assert!(read.load(Ordering::SeqCst));
    }
}

#[cfg(test)]
mod many_askers_tests {
    use super::{PAGE, writes};
    use windows_sys::Win32::System::Memory::*;

    /// Threads that write pages and ask about overlapping ranges at the same
    /// time: whoever wrote a page before asking is told of that write (by
    /// its own asking or by one that took the kernel's answer before it).
    #[test]
    fn no_write_is_lost_between_askers() {
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        const PAGES: usize = 48;
        const THREADS: usize = 8;
        // SAFETY: a reservation of our own, freed at the end.
        let at = unsafe {
            VirtualAlloc(std::ptr::null(), PAGES * PAGE, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE)
        } as usize;
        assert!(at != 0 && at.is_multiple_of(PAGE));
        std::thread::scope(|s| {
            for t in 0..THREADS {
                s.spawn(move || {
                    let mut last = [0u64; PAGES];
                    let mut seed = t as u64 * 7919 + 1;
                    let mut next = |n: usize| {
                        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                        (seed >> 33) as usize % n
                    };
                    for _ in 0..4000 {
                        let page = next(PAGES);
                        let from = page - next(page.min(12) + 1);
                        let to = page + 1 + next((PAGES - page).min(12));
                        // SAFETY: committed memory of ours; what the racing
                        // writers leave in it does not matter.
                        unsafe { ((at + page * PAGE + t * 8) as *mut u64).write_volatile(1) };
                        let (now, written) = writes(at + from * PAGE, at + to * PAGE, false).expect("write-watched");
                        let number = written.iter().find(|w| w.0 == at / PAGE + page).map_or(0, |w| w.1);
                        assert!(number > last[page], "page {page}: {number} is not after {}", last[page]);
                        for l in &mut last[from..to] {
                            *l = now;
                        }
                    }
                });
            }
        });
        // SAFETY: the reservation made above.
        unsafe { VirtualFree(at as *mut _, 0, MEM_RELEASE) };
    }

    /// The same of view memory, which is guarded: what its bits answer
    /// misses no write either, however often blocks are opened and guarded
    /// again between.
    #[test]
    fn no_write_to_views_is_lost_between_askers() {
        writes_to_views_are_not_lost::<1>(0, 0);
    }

    /// The same with renderer port #6 ([`super::port_guard`]): faults and
    /// refreshes beside one another under the shared lock, each block's state
    /// changed by the holder of its claim.
    #[test]
    fn no_write_to_views_is_lost_between_askers_with_port_6() {
        writes_to_views_are_not_lost::<1>(0, 1);
    }

    /// Pages at three addresses, a view coming and going, with port #6; and
    /// with half the threads under the shared lock and half under the lock
    /// alone (the switch changes in an A/B run).
    #[test]
    fn no_write_through_another_view_is_lost_with_port_6() {
        writes_to_views_are_not_lost::<3>(0, 1);
        writes_to_views_are_not_lost::<3>(0, 2);
    }

    /// Resting blocks with port #6, hot ones resting longer.
    #[test]
    fn no_write_to_resting_views_is_lost_with_port_6() {
        writes_to_views_are_not_lost::<3>(50, 1);
        writes_to_views_are_not_lost::<1>(50, 2);
    }

    /// The same with blocks that rest after a write ([`super::rest`]).
    #[test]
    fn no_write_to_resting_views_is_lost_between_askers() {
        writes_to_views_are_not_lost::<1>(50, 0);
        writes_to_views_are_not_lost::<3>(50, 0);
    }

    /// The same of pages shown at three addresses, each of which has copies
    /// of its own: a write through any of them, or put past the guards, is
    /// told at all. A fourth view comes and goes meanwhile, and is written
    /// too. (The first assertion fails within a few rounds if a guard holds
    /// for one address only, `VICTORY_VIEW_GUARD=0`.)
    #[test]
    fn no_write_through_another_view_is_lost_between_askers() {
        writes_to_views_are_not_lost::<3>(0, 0);
    }

    /// `port`: 0, renderer port #6 off; 1, on in all threads; 2, on in every
    /// other thread.
    fn writes_to_views_are_not_lost<const VIEWS: usize>(rest: u64, port: u8) {
        let rest = std::time::Duration::from_micros(rest);
        const PAGES: usize = 12;
        const THREADS: usize = 8;
        // SAFETY: views of a section of our own.
        let (section, all): (usize, [usize; VIEWS]) = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (PAGES * PAGE) as u32,
                std::ptr::null(),
            );
            crate::title_memory_section(section as usize, (PAGES * PAGE) as u64, true);
            let all = std::array::from_fn(|_| {
                let at = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, PAGES * PAGE).Value as usize;
                assert!(at != 0 && at.is_multiple_of(PAGE));
                super::title_memory_remapped(at, PAGES * PAGE);
                crate::title_memory_view(at, PAGES * PAGE, section as usize, 0);
                at
            });
            (section as usize, all)
        };
        let by_bits = std::sync::atomic::AtomicUsize::new(0);
        let port6 = |what: usize| super::PORT6[what].load(std::sync::atomic::Ordering::Relaxed);
        let port6_before = [super::P6_FAULTS, super::P6_GUARDS].map(port6);
        let past = std::sync::atomic::AtomicUsize::new(0);
        let title_beside = std::sync::atomic::AtomicUsize::new(0);
        std::thread::scope(|s| {
            if VIEWS > 1 {
                s.spawn(move || {
                    super::REST_IN_TEST.set(rest);
                    super::PORT_GUARD_IN_TEST.set(port == 1);
                    let mut last = [0u64; VIEWS];
                    for round in 0..300usize {
                        let (page, asked) = (round * 5 % PAGES, round % VIEWS);
                        // SAFETY (all): a view of one page of the section,
                        // this thread's alone.
                        let one = unsafe {
                            MapViewOfFile(section as _, FILE_MAP_WRITE, 0, (page * PAGE) as u32, PAGE).Value as usize
                        };
                        assert_ne!(one, 0);
                        super::title_memory_remapped(one, PAGE);
                        crate::title_memory_view(one, PAGE, section, (page * PAGE) as u64);
                        unsafe { ((one + THREADS * 8) as *mut u64).write_volatile(round as u64) };
                        let (now, written) =
                            writes(all[asked] + page * PAGE, all[asked] + (page + 1) * PAGE, true).expect("a view");
                        let number = written.first().map_or(0, |w| w.1);
                        assert!(number > last[asked], "page {page}, through a new view: {number} is not after {}", last[asked]);
                        last[asked] = now.max(last[asked]);
                        {
                            let _changing = super::title_memory_remapping();
                            super::title_memory_unmapping(one, PAGE);
                            unsafe { UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: one as *mut _ }) };
                        }
                        last = [last[asked]; VIEWS];
                    }
                });
            }
            for t in 0..THREADS {
                let (by_bits, past, title_beside) = (&by_bits, &past, &title_beside);
                s.spawn(move || {
                    super::REST_IN_TEST.set(rest);
                    super::PORT_GUARD_IN_TEST.set(port == 1 || port == 2 && t % 2 == 1);
                    super::HOT_IN_TEST.set(if rest.is_zero() { 0 } else { 20 });
                    let mut last = [[0u64; PAGES]; VIEWS];
                    let mut seed = t as u64 * 7919 + 1;
                    let mut next = |n: usize| {
                        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                        (seed >> 33) as usize % n
                    };
                    for _ in 0..500 {
                        let page = next(PAGES);
                        let from = page - next(page.min(3) + 1);
                        let to = page + 1 + next((PAGES - page).min(3));
                        // Written through one view, asked about at another.
                        let (through, asked) = (all[next(VIEWS)], next(VIEWS));
                        #[allow(unused_assignments)]
                        let mut how = 0;
                        let (at, last) = (all[asked], &mut last[asked]);
                        if next(4) != 0 {
                            // As in the test above, or after a word of ours,
                            // or past the guard.
                            let word = through + page * PAGE + t * 8;
                            // SAFETY (both): as above.
                            // (Waited for: other tests hold changes of mapping.)
                            // (Some as a copy of the title's, or a store of it, does.)
                            if next(3) == 0
                                && super::put_beside(word / PAGE, 1, |delta| unsafe {
                                    (word.wrapping_add(delta) as *mut u64).write_volatile(3);
                                }) == super::Beside::Done
                            {
                                title_beside.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                let (now, written) = writes(at + from * PAGE, at + to * PAGE, true).expect("a view");
                                let number = written.iter().find(|w| w.0 == at / PAGE + page).map_or(0, |w| w.1);
                                assert!(number > last[page], "page {page}, beside: {number} is not after {}", last[page]);
                                for l in &mut last[from..to] {
                                    *l = now;
                                }
                                continue;
                            }
                            let reading = (next(3) == 0).then(super::title_memory_reading);
                            how = usize::from(reading.is_some());
                            let beside = reading.as_ref().and_then(|_| crate::inplace::beside(word, 8));
                            if let Some(to) = beside {
                                unsafe { (to as *mut u64).write_volatile(2) };
                                super::landed_beside(&[word / 4096], &[(word, 8)]);
                                past.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                assert_eq!(unsafe { (word as *const u64).read_volatile() }, 2, "put elsewhere");
                            } else {
                                how += 2;
                                if next(4) == 0 {
                                    super::writing(word, 8);
                                    how += 4;
                                }
                                unsafe { (word as *mut u64).write_volatile(1) };
                            }
                        } else {
                            // Nothing written: what is asked counts as before.
                            by_bits.fetch_add(
                                usize::from(super::published(at + from * PAGE, at + to * PAGE)),
                                std::sync::atomic::Ordering::Relaxed,
                            );
                            let (now, _) = writes(at + from * PAGE, at + to * PAGE, true).expect("a view");
                            for l in &mut last[from..to] {
                                *l = now;
                            }
                            continue;
                        }
                        let (now, written) = writes(at + from * PAGE, at + to * PAGE, true).expect("a view");
                        let number = written.iter().find(|w| w.0 == at / PAGE + page).map_or(0, |w| w.1);
                        assert!(
                            number > last[page],
                            "{VIEWS} views, page {page}, through {:x} ({how}), asked at {:x}: {number} is not after {}",
                            through / PAGE + page,
                            at / PAGE + page,
                            last[page]
                        );
                        for l in &mut last[from..to] {
                            *l = now;
                        }
                    }
                });
            }
        });
        assert!(by_bits.into_inner() > 0, "the bits answered none");
        if port != 0 {
            let after = [super::P6_FAULTS, super::P6_GUARDS].map(port6);
            assert!(after[0] > port6_before[0] && after[1] > port6_before[1], "port 6: no fault or guard under the shared lock");
        }
        assert!(past.into_inner() > 0, "nothing was put past the guards");
        assert!(title_beside.into_inner() > 0, "nothing of the title's was put in beside the guards");
        // (The view stays: at its address other tests' memory would be told
        // of these writes.)
    }

    /// A range of a placeholder with views in it is known whole, its holes
    /// among it, until something is mapped into one, unmapped from it or
    /// allocated there. A hole that is no placeholder's is asked about.
    #[test]
    fn holes_are_known_until_the_runtime_tells() {
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        const BLOCKS: usize = 8;
        // SAFETY (all): a placeholder and a section of our own, which stay.
        let (at, section) = unsafe {
            let at = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                BLOCKS * PAGE,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            let section =
                CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE, 0, (BLOCKS * PAGE) as u32, std::ptr::null());
            (at, section)
        };
        let map = |block: usize| unsafe {
            let to = at + block * PAGE;
            VirtualFree(to as *mut _, PAGE, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER);
            let view = MapViewOfFile3(
                section,
                GetCurrentProcess(),
                to as *const _,
                (block * PAGE) as u64,
                PAGE,
                MEM_REPLACE_PLACEHOLDER,
                PAGE_READWRITE,
                std::ptr::null_mut(),
                0,
            );
            assert_eq!(view.Value as usize, to);
            super::title_memory_remapped(to, PAGE);
        };
        map(1);
        map(5);
        assert!(super::placeholder(at / PAGE) && super::placeholder(at / PAGE + 7) && !super::placeholder(at / PAGE + 5));
        let end = at + BLOCKS * PAGE;
        let ask = |since: u64| -> (u64, Vec<usize>) {
            let (now, written) = writes(at, end, true).expect("a view");
            (now, written.into_iter().filter(|w| w.1 > since).map(|w| w.0 - at / PAGE).collect())
        };
        let (first, all) = ask(0);
        assert_eq!(all, (0..BLOCKS).collect::<Vec<_>>());
        assert!(super::published(at, end), "with its holes");
        // Written; mapped into; unmapped.
        unsafe { ((at + 5 * PAGE) as *mut u8).write_volatile(1) };
        assert!(!super::published(at, end));
        let (second, all) = ask(first);
        assert_eq!(all, [5]);
        map(2);
        assert!(!super::published(at, end));
        let (third, all) = ask(second);
        assert_eq!(all, [2]);
        unsafe { ((at + 2 * PAGE) as *mut u8).write_volatile(1) };
        let (fourth, all) = ask(third);
        assert_eq!(all, [2]);
        super::title_memory_unmapping(at + 5 * PAGE, PAGE);
        unsafe {
            let view = MEMORY_MAPPED_VIEW_ADDRESS { Value: (at + 5 * PAGE) as *mut _ };
            UnmapViewOfFile2(GetCurrentProcess(), view, MEM_PRESERVE_PLACEHOLDER);
        }
        let (fifth, all) = ask(fourth);
        assert_eq!(all, [5]);
        assert!(super::published(at, end));
        // Told of memory there, though there is none: asked about again.
        super::title_memory_allocated(at, PAGE);
        assert!(!super::published(at, end));
        let (_, all) = ask(fifth);
        assert_eq!(all, []);
        // A reservation of the ordinary kind is no placeholder.
        let plain = unsafe { VirtualAlloc(std::ptr::null(), PAGE, MEM_RESERVE, PAGE_NOACCESS) } as usize;
        assert!(!super::placeholder(plain / PAGE));
        super::look_at_holes();
        assert!(!super::published(at, end), "block 0 stays asked about");
        assert!(super::published(at + PAGE, end), "the others are as they were");
        super::title_memory_unmapping(at, BLOCKS * PAGE);
    }

    /// Pages mapped into a placeholder, written and unmapped again, each by a
    /// thread of its own as the runtime does it, while others ask about all
    /// of it: what was written, and that a page came or went, is told to
    /// whoever asks after it, and blocks are known by their bits meanwhile.
    #[test]
    fn no_mapping_is_lost_between_askers() {
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        const BLOCKS: usize = 12;
        // SAFETY (all): a placeholder and a section of our own, which stay.
        let (at, section) = unsafe {
            let at = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                BLOCKS * PAGE,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            for block in 0..BLOCKS - 1 {
                VirtualFree((at + block * PAGE) as *mut _, PAGE, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER);
            }
            let section =
                CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE, 0, (BLOCKS * PAGE) as u32, std::ptr::null());
            (at, section as usize)
        };
        let end = at + BLOCKS * PAGE;
        let (stop, by_bits) = (AtomicBool::new(false), AtomicUsize::new(0));
        let number = |written: &[(usize, u64)], block: usize| written.iter().find(|w| w.0 == at / PAGE + block).map_or(0, |w| w.1);
        std::thread::scope(|s| {
            let mappers: Vec<_> = (0..4usize)
                .map(|t| {
                    let number = &number;
                    s.spawn(move || {
                        for round in 0..120 {
                            let block = t * 3 + round % 3;
                            let to = at + block * PAGE;
                            let (before, _) = writes(at, end, true).expect("a view");
                            unsafe {
                                let view = MapViewOfFile3(
                                    section as _,
                                    GetCurrentProcess(),
                                    to as *const _,
                                    (block * PAGE) as u64,
                                    PAGE,
                                    MEM_REPLACE_PLACEHOLDER,
                                    PAGE_READWRITE,
                                    std::ptr::null_mut(),
                                    0,
                                );
                                assert_eq!(view.Value as usize, to);
                                super::title_memory_remapped(to, PAGE);
                                (to as *mut u8).write_volatile(round as u8);
                            }
                            let (mapped, written) = writes(at, end, true).expect("a view");
                            assert!(number(&written, block) > before, "mapped and written: not told");
                            unsafe { ((to + 9) as *mut u8).write_volatile(round as u8) };
                            let (again, written) = writes(to, to + PAGE, true).expect("a view");
                            assert!(number(&written, block) > mapped, "written again: not told");
                            super::title_memory_unmapping(to, PAGE);
                            unsafe {
                                let view = MEMORY_MAPPED_VIEW_ADDRESS { Value: to as *mut _ };
                                assert_ne!(UnmapViewOfFile2(GetCurrentProcess(), view, MEM_PRESERVE_PLACEHOLDER), 0);
                            }
                            let _ = again;
                        }
                    })
                })
                .collect();
            for t in 0..3usize {
                let (stop, by_bits) = (&stop, &by_bits);
                s.spawn(move || {
                    let mut k = t;
                    while !stop.load(Ordering::Relaxed) {
                        k = (k * 5 + 3) % 64;
                        let (from, to) = (at + k % 6 * PAGE, end - k / 8 % 5 * PAGE);
                        by_bits.fetch_add(usize::from(super::published(from, to)), Ordering::Relaxed);
                        assert!(writes(from, to, true).is_some());
                    }
                });
            }
            let failed = mappers.into_iter().map(|m| m.join()).filter(Result::is_err).count();
            stop.store(true, Ordering::Relaxed);
            assert_eq!(failed, 0);
        });
        assert!(by_bits.into_inner() > 0, "the bits answered none");
        // All unmapped: known whole, and nothing but holes.
        writes(at, end, true).expect("a view");
        assert!(super::published(at, end));
        super::title_memory_unmapping(at, BLOCKS * PAGE);
    }

    /// A block's bit goes with everything that makes it the title's to write,
    /// or forgets it.
    #[test]
    fn bits_go_with_the_guard() {
        // SAFETY: a view of a section of our own, which stays.
        let at = unsafe {
            let section =
                CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE, 0, (2 * PAGE) as u32, std::ptr::null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * PAGE).Value as usize
        };
        let end = at + 2 * PAGE;
        let takes: [&dyn Fn(); 4] = [
            &|| super::title_memory_remapped(at, PAGE),
            // SAFETY: the view.
            &|| unsafe { ((at + 8) as *mut u8).write_volatile(1) },
            &|| super::writing(at, 1),
            // (Told again, as of a view mapped there next: it is read-only.)
            &|| {
                super::title_memory_unmapping(at, PAGE);
                assert!(!super::published(at, end), "forgotten");
                super::title_memory_remapped(at, PAGE);
                let mut old = 0;
                // SAFETY: the view.
                unsafe { VirtualProtect(at as *const _, PAGE, PAGE_READWRITE, &mut old) };
            },
        ];
        for (i, take) in takes.iter().enumerate() {
            writes(at, end, true).expect("a view");
            assert!(super::published(at, end), "{i}");
            take();
            assert!(!super::published(at, end), "{i}");
            assert!(super::published(at + PAGE, end), "{i}: the other block");
        }
        super::title_memory_unmapping(at, 2 * PAGE);
    }

    /// A range's bits are all of its blocks', whatever lies beside them.
    #[test]
    fn bits_of_ranges() {
        static BITS: super::Bits = super::Bits::new();
        let blocks = [0usize, 1, 63, 64, 65, 127, 128, 200, 65535, 65536, 65537, 1 << 20, (1 << 31) - 1];
        for &b in &blocks {
            assert!(!BITS.all(b, b + 1));
            BITS.set(b);
            assert!(BITS.all(b, b + 1));
        }
        assert!(BITS.all(0, 2) && BITS.all(63, 66) && BITS.all(127, 129) && BITS.all(65535, 65538));
        assert!(!BITS.all(0, 3) && !BITS.all(62, 64) && !BITS.all(65534, 65536) && !BITS.all(5, 5));
        for b in 300..700 {
            BITS.set(b);
        }
        assert!(BITS.all(300, 700) && !BITS.all(299, 700) && !BITS.all(300, 701));
        BITS.clear(511);
        assert!(!BITS.all(300, 700) && BITS.all(300, 511) && BITS.all(512, 700));
        assert!(!BITS.all(1 << 40, (1 << 40) + 1), "outside the address space");
    }
}

/// Renderer port #7 ([`port_journal`]): the journal's notes in parts.
#[cfg(test)]
mod port7_tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    use windows_sys::Win32::System::Memory::*;

    fn rng(seed: &mut u64, n: usize) -> usize {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*seed >> 33) as usize % n.max(1)
    }

    /// The parts of a range, its pieces, and what is read of it a part at a
    /// time or holding its parts, over ranges across the parts' bounds.
    #[test]
    fn parts_pieces_and_ranges_agree() {
        const P: usize = 1 << SHARD_SHIFT;
        assert_eq!(shards_of(0, 0), 0);
        assert_eq!(shards_of(0, 1), 1);
        assert_eq!(shards_of(P - 1, P + 1), 0b11);
        assert_eq!(shards_of(63 * P, 65 * P), 1 << 63 | 1);
        assert_eq!(shards_of(5, 5 + SHARDS * P), u64::MAX);
        assert_eq!(
            pieces(P - 2, 2 * P + 3).unwrap().collect::<Vec<_>>(),
            [(P - 2, P), (P, 2 * P), (2 * P, 2 * P + 3)]
        );
        assert_eq!(pieces(7, 4).unwrap().next(), None);
        assert_eq!(
            pieces(usize::MAX - 1, usize::MAX).unwrap().next(),
            Some((usize::MAX - 1, usize::MAX))
        );
        assert!(pieces(0, 1 << 40).is_none());
        let notes = Notes::new();
        let mut want: BTreeMap<usize, u64> = BTreeMap::new();
        let mut seed = 7;
        for n in 1..4000u64 {
            let page = rng(&mut seed, 70 * P);
            let mut held = notes.hold(shards_of_pages([page]), true);
            held.note(page, WHOLE, n);
            want.insert(page, n);
        }
        for _ in 0..500 {
            let first = rng(&mut seed, 70 * P);
            let last = first + rng(&mut seed, 3 * P);
            let expected: Vec<(usize, u64)> = want.range(first..last).map(|(&p, &w)| (p, w)).collect();
            assert_eq!(notes.range(first, last), expected, "{first}..{last} a part at a time");
            assert_eq!(notes.hold(shards_of(first, last), false).range(first, last), expected, "{first}..{last} held");
        }
        // The whole space: each part once.
        assert_eq!(notes.range(0, usize::MAX / 2), want.iter().map(|(&p, &w)| (p, w)).collect::<Vec<_>>());
        assert_eq!(notes.hold(u64::MAX, false).range(0, usize::MAX / 2).len(), want.len());
    }

    /// Noted out of order (shared noters' numbers may come in any order): a
    /// page's note, and its parts', only go up.
    #[test]
    fn notes_only_go_up() {
        let mut s = Shard::new();
        assert_eq!(s.note(3, WHOLE, 9), 0);
        assert_eq!(s.note(3, 0b11, 5), 9);
        assert_eq!(s.written[&3], 9);
        assert_eq!(s.parts_since(3, 8), WHOLE, "all of it at 9");
        assert_eq!(s.note(3, 0b100, 12), 9);
        assert_eq!(s.parts_since(3, 9), 0b100);
        // An older whole note leaves the newer part as it is.
        assert_eq!(s.note(3, WHOLE, 10), 12);
        assert_eq!(s.parts_since(3, 10), 0b100);
        assert_eq!(s.parts_since(3, 9), WHOLE);
        assert_eq!(s.written[&3], 12);
    }

    /// Many threads note pages (holding their parts for writing, a number
    /// taken under them) and read ranges with a fresh number (holding their
    /// parts for reading), the parts taken in any mix: every reader sees
    /// every note of a lower number of its range's pages (what keeps a write
    /// from being lost), and nobody waits for ever. (A watchdog fails the
    /// test after 60 s.)
    #[test]
    fn a_fresh_number_sees_every_lower_note_and_nobody_deadlocks() {
        const P: usize = 1 << SHARD_SHIFT;
        const THREADS: usize = 8;
        const ROUNDS: usize = 3000;
        static NOTES: Notes = Notes::new();
        static NUMBER: AtomicU64 = AtomicU64::new(0);
        let (done, wait) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            // Per thread: what it noted (page, number) and what it read
            // (first, last, number, the notes read).
            type Read = (usize, usize, u64, Vec<(usize, u64)>);
            let logs: Vec<(Vec<(usize, u64)>, Vec<Read>)> = std::thread::scope(|s| {
                let all: Vec<_> = (0..THREADS)
                    .map(|t| {
                        s.spawn(move || {
                            let mut seed = t as u64 * 104_729 + 3;
                            let (mut noted, mut read) = (Vec::new(), Vec::new());
                            for _ in 0..ROUNDS {
                                let first = rng(&mut seed, 200 * P);
                                if rng(&mut seed, 2) == 0 {
                                    // A few pages, some far apart.
                                    let mut pages: Vec<usize> =
                                        (0..1 + rng(&mut seed, 6)).map(|_| first + rng(&mut seed, 3 * P)).collect();
                                    if rng(&mut seed, 4) == 0 {
                                        pages.push(rng(&mut seed, 200 * P));
                                    }
                                    let mut held = NOTES.hold(shards_of_pages(pages.iter().copied()), true);
                                    let now = NUMBER.fetch_add(1, Relaxed) + 1;
                                    for &p in &pages {
                                        held.note(p, WHOLE, now);
                                        noted.push((p, now));
                                    }
                                } else {
                                    let last = first + 1 + rng(&mut seed, 3 * P);
                                    let held = NOTES.hold(shards_of(first, last), false);
                                    let now = NUMBER.fetch_add(1, Relaxed) + 1;
                                    read.push((first, last, now, held.range(first, last)));
                                }
                            }
                            (noted, read)
                        })
                    })
                    .collect();
                all.into_iter().map(|h| h.join().unwrap()).collect()
            });
            let mut by_page: std::collections::HashMap<usize, Vec<u64>> = Default::default();
            for (noted, _) in &logs {
                for &(p, n) in noted {
                    by_page.entry(p).or_default().push(n);
                }
            }
            by_page.values_mut().for_each(|v| v.sort_unstable());
            let mut checked = 0;
            for (_, read) in &logs {
                for (first, last, now, seen) in read {
                    let seen: std::collections::HashMap<usize, u64> = seen.iter().copied().collect();
                    for (&p, ns) in by_page.iter().filter(|(p, _)| (*first..*last).contains(*p)) {
                        // The highest note below the reader's number.
                        let below = ns.partition_point(|n| n < now);
                        if below > 0 {
                            let wanted = ns[below - 1];
                            let got = seen.get(&p).copied().unwrap_or(0);
                            assert!(got >= wanted, "page {p}: read {got} with number {now}, note {wanted} missed");
                            checked += 1;
                        }
                    }
                }
            }
            assert!(checked > 1000, "only {checked} checks");
            done.send(()).unwrap();
        });
        wait.recv_timeout(std::time::Duration::from_secs(60)).expect("the parts' holders deadlocked (or a check failed)");
    }

    /// The journal's functions themselves, with the switch on in all threads
    /// and in every other one (as in an A/B run): writers and askers of one
    /// region; a submission's one asking standing over another region, its
    /// thread writing there (told with `wrote` or found by its asking), and
    /// other threads asking about that region meanwhile. Whoever wrote a page
    /// before asking is told of that write.
    #[test]
    fn no_write_is_lost_with_port_7() {
        let _one = one_asking_test();
        askers_and_a_submission(1);
        askers_and_a_submission(2);
    }

    /// `port`: 1, renderer port #7 on in all threads; 2, in every other.
    fn askers_and_a_submission(port: u8) {
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        const PAGES: usize = 48;
        const HALF: usize = PAGES / 2;
        const THREADS: usize = 6;
        // SAFETY: a reservation of our own, freed at the end.
        let at = unsafe {
            VirtualAlloc(std::ptr::null(), PAGES * PAGE, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE)
        } as usize;
        assert!(at != 0 && at.is_multiple_of(PAGE));
        let shared_before = PORT7[P7_SHARED_NOTES].load(Relaxed);
        let skipped_before = PORT7[P7_SKIPPED].load(Relaxed);
        std::thread::scope(|s| {
            // Writers and askers of the first half.
            for t in 0..THREADS {
                s.spawn(move || {
                    PORT_JOURNAL_IN_TEST.set(port == 1 || t % 2 == 0);
                    let mut last = [0u64; HALF];
                    let mut seed = t as u64 * 7919 + 1;
                    for _ in 0..2500 {
                        let page = rng(&mut seed, HALF);
                        let from = page - rng(&mut seed, page.min(8) + 1);
                        let to = page + 1 + rng(&mut seed, (HALF - page).min(8));
                        // SAFETY: committed memory of ours.
                        unsafe { ((at + page * PAGE + t * 8) as *mut u64).write_volatile(1) };
                        let (now, written) = writes(at + from * PAGE, at + to * PAGE, false).expect("write-watched");
                        let number = written.iter().find(|w| w.0 == at / PAGE + page).map_or(0, |w| w.1);
                        assert!(number > last[page], "page {page}: {number} is not after {}", last[page]);
                        for l in &mut last[from..to] {
                            *l = now;
                        }
                        // Now and then a no-op of the standing asking's.
                        if rng(&mut seed, 16) == 0 {
                            wrote(at + page * PAGE, 8);
                            ask_ahead_done();
                        }
                    }
                });
            }
            // Askers of the second half that write nothing there: they take
            // the kernel's answers from the submission's thread.
            for t in 0..2 {
                s.spawn(move || {
                    PORT_JOURNAL_IN_TEST.set(port == 1 || t == 0);
                    let mut seed = t as u64 * 31 + 5;
                    for _ in 0..3000 {
                        let page = HALF + rng(&mut seed, HALF);
                        let to = (page + 1 + rng(&mut seed, 4)).min(PAGES);
                        let _ = writes(at + page * PAGE, at + to * PAGE, false);
                    }
                });
            }
            // The submission: writes pages of the second half, asks once over
            // all of it, writes more and tells `wrote`, reads everything from
            // the standing asking, ends it.
            s.spawn(move || {
                let mut last = [0u64; HALF];
                let mut seed = 99;
                for round in 0..400 {
                    PORT_JOURNAL_IN_TEST.set(port == 1 || round % 2 == 0);
                    let before: Vec<usize> = (0..3).map(|_| rng(&mut seed, HALF)).collect();
                    for &p in &before {
                        // SAFETY: committed memory of ours.
                        unsafe { ((at + (HALF + p) * PAGE) as *mut u64).write_volatile(round) };
                    }
                    let ranges: Vec<(usize, usize)> =
                        (0..4).map(|k| (at + (HALF + k * 6) * PAGE, at + (HALF + k * 6 + 6) * PAGE)).collect();
                    ask_once(&ranges);
                    let told = rng(&mut seed, HALF);
                    // SAFETY: committed memory of ours.
                    unsafe { ((at + (HALF + told) * PAGE + 64) as *mut u64).write_volatile(round) };
                    wrote(at + (HALF + told) * PAGE + 64, 8);
                    let mut read: Vec<usize> = before.iter().copied().chain([told]).collect();
                    read.sort_unstable();
                    read.dedup();
                    for p in read {
                        let (now, written) =
                            writes(at + (HALF + p) * PAGE, at + (HALF + p + 1) * PAGE, false).expect("write-watched");
                        let number = written.iter().find(|w| w.0 == at / PAGE + HALF + p).map_or(0, |w| w.1);
                        assert!(number > last[p], "round {round}, page {p}: {number} is not after {}", last[p]);
                        last[p] = now;
                    }
                    ask_ahead_done();
                    ask_once_end();
                }
            });
        });
        assert!(PORT7[P7_SHARED_NOTES].load(Relaxed) > shared_before, "the shared notings ran");
        assert!(PORT7[P7_SKIPPED].load(Relaxed) > skipped_before, "needless takings alone were left out");
        // SAFETY: the reservation made above.
        unsafe { VirtualFree(at as *mut _, 0, MEM_RELEASE) };
    }
}

#[cfg(test)]
mod rearm_tests {
    use super::*;
    use std::sync::atomic::Ordering::Relaxed;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_DECOMMIT, MEM_RELEASE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, PAGE_GUARD, PAGE_READWRITE,
        VirtualAlloc, VirtualFree, VirtualQuery,
    };

    /// A committed page is guarded again; one that is not committed any more is
    /// not, and that is counted, as is a fault that gave no address.
    #[test]
    fn a_page_that_cannot_be_guarded_again_is_counted() {
        // SAFETY: a fresh page of our own.
        let page = unsafe { VirtualAlloc(std::ptr::null(), 4096, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) } as usize;
        assert_ne!(page, 0);
        let before = NOT_REARMED.load(Relaxed);
        assert!(rearm_guard(Some(page)));
        let mut info = MEMORY_BASIC_INFORMATION::default();
        // SAFETY: a valid out pointer of the right size.
        assert_ne!(unsafe { VirtualQuery(page as *const _, &mut info, size_of_val(&info)) }, 0);
        assert_ne!(info.Protect & PAGE_GUARD, 0, "a committed page is guarded again");
        // SAFETY: our page; decommitted, it is reserved and not committed.
        unsafe { VirtualFree(page as *mut _, 4096, MEM_DECOMMIT) };
        assert!(!rearm_guard(Some(page)), "a page that is not committed cannot be guarded");
        assert!(!rearm_guard(None), "a fault that gave no address");
        assert_eq!(NOT_REARMED.load(Relaxed), before + 2);
        // SAFETY: our allocation.
        unsafe { VirtualFree(page as *mut _, 0, MEM_RELEASE) };
    }
}
