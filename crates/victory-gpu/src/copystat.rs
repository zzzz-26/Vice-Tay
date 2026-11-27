//! Diagnostics: where the submission copies of title memory go (which
//! ranges, how often, of what memory) and what limits their rate, logged
//! with the frame-rate line. `VICTORY_COPY_MODE=plain` copies without the
//! mapped-page guard, `none` skips the copies (both for timing only). Set
//! `VICTORY_MEMORY_CATEGORIES=1` for one extra periodic line that gathers the
//! memory categories already counted here into one approximation.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Counters since the last line.
pub(crate) enum C {
    Submissions,
    Copies,
    CopyBytes,
    Refreshes,
    RefreshBytes,
    Skipped,
    SkippedBytes,
    Runs,
    RunUs,
    Queries,
    QueryUs,
    MoveUs,
    Unmapped,
    /// Write-watch askings and their time.
    Watches,
    WatchUs,
    /// Bytes of watched ranges checked, and of those found unchanged.
    Checked,
    Unchanged,
    /// Copies that met an unmapped page.
    Fallbacks,
    /// Writes taken by the guard on view memory.
    GuardFaults,
    /// Texture bytes detiled after a watched write; hashed; detiled after a
    /// hash change.
    TexWatched,
    TexHashed,
    TexChanged,
    /// Changes of guarded textures the hash found and the guard did not see.
    TexHot,
    /// Frames presented since the last line (what the line divides by).
    Frames,
    /// Calls of list methods that record into a host list, and the
    /// nanoseconds they took, summed over the title's recording threads.
    RecordCalls,
    RecordNs,
    /// Waiting for host pipelines being made (`pipeline::make`).
    PipelineWaitUs,
    /// Dynamic descriptor slots: tables asked for, refused (no free run),
    /// and the nanoseconds of the searches; the slots the tables took.
    DynamicAsked,
    DynamicRefused,
    DynamicNs,
    DynamicSlots,
    /// Of the refused: those with enough slots free in all, but no run of
    /// them together (the slots in pieces, not taken).
    DynamicPieces,
    /// Tables that, refused, waited for the flights in the air to land
    /// before they asked again (`crate::draw`'s `dynamic_or_settle`), the
    /// microseconds of those waits (counted in "a blocking fence wait"
    /// too), and the tables refused again after it: their draws or
    /// dispatches are dropped.
    DynamicSettles,
    DynamicSettleUs,
    DynamicLost,
    /// Marker texts for device-removed data (`HostList::note`).
    NoteNs,
    /// Ranges given to shaders (steps), their snapshots, and the bytes
    /// compared after the submission.
    GpuWrites,
    SnapshotUs,
    BackCompared,
    /// Reading around unmapped pages after a failed bulk read.
    SparseUs,
    /// Guarding view memory (part of the write-watch askings' time).
    GuardUs,
    /// Bytes copied whole because their memory is neither write-watched
    /// nor a view (`Shadow::update_free`).
    WholeBytes,
    /// Submissions with ranges shaders write (they need the GPU's results
    /// before they return).
    BackSubmissions,
    /// The title's frame waits (`frame::wait`) and the time slept in them,
    /// over all its threads.
    PaceWaits,
    PaceUs,
    /// After a present: until the title's first list call and until its
    /// first submission; and from its last submission to its next present.
    UntilRecordUs,
    UntilSubmitUs,
    TailUs,
    /// Resetting host lists (`HostList::reset`).
    ResetNs,
    /// What shaders wrote: finding the 64 KiB pieces that differ, how many
    /// do, and putting them into title memory.
    BackScanUs,
    BackDiffer,
    BackSyncUs,
    /// Texture uploads recorded, and left out as unchanged.
    TexUploads,
    TexSkipped,
    /// Held submissions a present ran although their waits were not met.
    Forced,
    /// Ranges named through unordered access views by shaders that cannot
    /// store to buffers, not given to shaders as written, and their bytes.
    GpuWritesLeftOut,
    GpuWritesLeftOutBytes,
    /// Waits for memory values looked at (`list::ValueWait::met`) and the
    /// nanoseconds the looks took, on every thread.
    WaitChecks,
    WaitCheckNs,
    /// Host lists made (`Host::new_list`: for the lists a queue holds and
    /// for the parts of lists cut at waits) and the nanoseconds that took.
    HostLists,
    HostListNs,
    /// Parts of held submissions that ran while the part behind them could
    /// have run with them (its waits were met already).
    PartsTogether,
    /// Microseconds of the copies counted in `WholeBytes`.
    WholeUs,
    /// Submissions by how long the card was waited for (under 2 ms, 2-8,
    /// 8-32, longer), and the microseconds waited in each class.
    CardUnder2,
    Card2To8,
    Card8To32,
    CardOver32,
    CardUnder2Us,
    Card2To8Us,
    Card8To32Us,
    CardOver32Us,
    /// Submissions that hand nothing back to the title (no range shaders
    /// write, no value written when done), and the microseconds the card
    /// was waited for them.
    NothingBack,
    NothingBackUs,
    /// Waits for every earlier shader write recorded into host lists: for
    /// the title's barrier groups, for its barriers on buffers, and the
    /// indirect commands bounded (each a dispatch of ours between two such
    /// waits).
    WaitsForGroups,
    WaitsForBuffers,
    IndirectBounded,
}
const COUNTERS: usize = 80;
/// One of some sets of counters, each in lines of the processor's memory of
/// its own: a set for all threads had them wait for each other's lines at
/// every list call (what counts here is summed up for the log alone).
#[repr(align(64))]
struct Bank<const N: usize>([AtomicU64; N]);

/// The sets.
const BANKS: usize = 32;

static COUNT: [Bank<COUNTERS>; BANKS] = [const { Bank([const { AtomicU64::new(0) }; COUNTERS]) }; BANKS];

/// This thread's set.
#[inline]
fn bank() -> usize {
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    thread_local!(static MINE: std::cell::Cell<usize> = const { std::cell::Cell::new(usize::MAX) });
    // (Asked by the allocator too: nothing here allocates.)
    MINE.try_with(|mine| {
        if mine.get() == usize::MAX {
            mine.set(NEXT.fetch_add(1, Relaxed) % BANKS);
        }
        mine.get()
    })
    .unwrap_or(0)
}

pub(crate) fn add(c: C, n: u64) {
    COUNT[bank()].0[c as usize].fetch_add(n, Relaxed);
}

/// The counter, all sets together, as it is now (it goes back to 0 with
/// [`log`]).
pub(crate) fn peek(c: C) -> u64 {
    let i = c as usize;
    COUNT.iter().map(|b| b.0[i].load(Relaxed)).sum()
}

/// Counter `i`, all sets together, from now on 0.
fn taken(i: usize) -> u64 {
    COUNT.iter().map(|b| b.0[i].swap(0, Relaxed)).sum()
}

/// How copies run: 0 guarded (as always), 1 plain, 2 not at all.
pub(crate) fn mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_COPY_MODE").as_deref() {
        Ok("plain") => 1,
        Ok("none") => 2,
        _ => 0,
    })
}

/// Per source range (start, length): copies, bytes, microseconds, memory type.
type Ranges = HashMap<(usize, usize), (u64, u64, u64, u32)>;
static RANGES: Mutex<Option<Ranges>> = Mutex::new(None);

/// Per range copied whole because its memory is not watched (start,
/// length): copies and microseconds.
static WHOLE: Mutex<Option<HashMap<(usize, usize), (u64, u64)>>> = Mutex::new(None);

/// `len` bytes at `src`, memory that is not watched, were copied whole in
/// `us`.
pub(crate) fn whole(src: usize, len: usize, us: u64) {
    add(C::WholeUs, us);
    let mut map = WHOLE.lock().unwrap_or_else(|e| e.into_inner());
    let e = map.get_or_insert_with(HashMap::new).entry((src, len)).or_insert((0, 0));
    e.0 += 1;
    e.1 += us;
}

/// The longest wait for the card since the last line: microseconds, and
/// what ran (lists and pipelines).
static LONGEST: Mutex<(u64, String)> = Mutex::new((0, String::new()));

/// The waits for the card since the last line by what ran (lists and
/// pipelines): how many, and their microseconds.
static WAITS_BY_WORK: Mutex<Option<HashMap<String, (u64, u64)>>> = Mutex::new(None);

/// With `VICTORY_GPU_TIMING`: the commands' time on the card since the last
/// line by what they ran (a pipeline, or the kind of command): how many,
/// their microseconds, and the longest one's.
static COMMANDS_BY_WORK: Mutex<Option<HashMap<String, (u64, u64, u64)>>> = Mutex::new(None);

/// What a list of a submission is, for a timing run's sums by list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Timed {
    /// A run's prelude: refreshes of the copies its commands read.
    Prelude,
    /// The title's list, the run of that place in the submission.
    Title(usize),
    /// Ours in front of the lists, the steps done on the card.
    Steps,
    /// The compare of copies in the card's memory, behind the lists.
    Compare,
    /// Another of ours (the timeline's stamps).
    Other,
}

impl Timed {
    fn index(self) -> usize {
        match self {
            Timed::Prelude => 0,
            Timed::Title(_) => 1,
            Timed::Steps => 2,
            Timed::Compare => 3,
            Timed::Other => 4,
        }
    }
}

/// With `VICTORY_GPU_TIMING`, per kind of list ([`Timed`]): lists, their
/// microseconds from sending to the card's signal, and the microseconds of
/// the commands noted in them; then the least time of any list (what a
/// submission of nothing costs, about).
static LISTS_TIMED: Mutex<([[u64; 3]; 5], u64)> = Mutex::new(([[0; 3]; 5], u64::MAX));

/// A list of kind `kind` took `wall` microseconds from its sending to the
/// card's signal, `noted` of them in its noted commands (timing run).
pub(crate) fn list_took(kind: Timed, wall: u64, noted: u64) {
    let mut t = LISTS_TIMED.lock().unwrap_or_else(|e| e.into_inner());
    let e = &mut t.0[kind.index()];
    e[0] += 1;
    e[1] += wall;
    e[2] += noted;
    t.1 = t.1.min(wall);
}

/// A command took `ms` on the card (timing run). `op` is its note: the
/// kind and size first, then the pipeline in brackets, then what it binds.
pub(crate) fn command_took(op: &str, ms: f64) {
    let us = (ms * 1e3) as u64;
    let mut by = COMMANDS_BY_WORK.lock().unwrap_or_else(|e| e.into_inner());
    let e = by.get_or_insert_with(HashMap::new).entry(command_key(op)).or_insert((0, 0, 0));
    e.0 += 1;
    e.1 += us;
    e.2 = e.2.max(us);
}

/// What a command's note is summed under: its pipeline (`[compute ...]`,
/// `[mesh ..., pixel ...]`) with the command's first word, or without a
/// pipeline its first two words.
fn command_key(op: &str) -> String {
    let head = op.split(" | ").next().unwrap_or(op);
    let word = head.split(' ').next().unwrap_or("");
    match (head.find('['), head.find(']')) {
        (Some(a), Some(b)) if a < b => format!("{word} {}", &head[a..=b]),
        _ => head.split(' ').take(2).collect::<Vec<_>>().join(" "),
    }
}

/// The card was waited for `us` for a submission; `what` names it if it is
/// the longest since the last line.
pub(crate) fn card_waited(us: u64, what: impl FnOnce() -> String) {
    let (n, t) = match us {
        0..2_000 => (C::CardUnder2, C::CardUnder2Us),
        2_000..8_000 => (C::Card2To8, C::Card2To8Us),
        8_000..32_000 => (C::Card8To32, C::Card8To32Us),
        _ => (C::CardOver32, C::CardOver32Us),
    };
    add(n, 1);
    add(t, us);
    // Every submission is named (some forty a frame): the same lists run
    // every frame, and the eight longest waits of a line were one list
    // eight times.
    let text = what();
    {
        let mut by = WAITS_BY_WORK.lock().unwrap_or_else(|e| e.into_inner());
        let e = by.get_or_insert_with(HashMap::new).entry(text.clone()).or_insert((0, 0));
        e.0 += 1;
        e.1 += us;
    }
    let mut longest = LONGEST.lock().unwrap_or_else(|e| e.into_inner());
    if us > longest.0 {
        *longest = (us, text);
    }
}

/// Whether an experiment on barriers is on (`VICTORY_BARRIER_TEST`): bit 0
/// for `groups` (the title's barrier groups record no wait for earlier
/// shader writes), bit 1 for `indirect` (the pass that bounds an indirect
/// command runs without its two such waits; its state changes stay), both
/// bits for `both`. For timing only: without the waits a shader may read
/// what another has not finished writing.
pub(crate) fn barrier_test() -> u8 {
    static TEST: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *TEST.get_or_init(|| match std::env::var("VICTORY_BARRIER_TEST").as_deref() {
        Ok("groups") => 1,
        Ok("indirect") => 2,
        Ok("both") => 3,
        _ => 0,
    })
}

/// Per range compared after a submission (start, length): times, and the
/// bytes of it written back to the title.
static COMPARED: Mutex<Option<HashMap<(usize, usize), (u64, u64)>>> = Mutex::new(None);

/// `len` bytes at `src` were compared after a submission and `written` of
/// them put back into title memory.
pub(crate) fn compared(src: usize, len: usize, written: usize) {
    let mut map = COMPARED.lock().unwrap_or_else(|e| e.into_inner());
    let e = map.get_or_insert_with(HashMap::new).entry((src, len)).or_insert((0, 0));
    e.0 += 1;
    e.1 += written as u64;
}

/// A copy of `len` bytes from `src` (memory type `kind`) took `us`.
pub(crate) fn range(src: usize, len: usize, us: u64, kind: u32) {
    let mut map = RANGES.lock().unwrap_or_else(|e| e.into_inner());
    let e = map
        .get_or_insert_with(HashMap::new)
        .entry((src, len))
        .or_insert((0, 0, 0, kind));
    e.0 += 1;
    e.1 += len as u64;
    e.2 += us;
}

/// Our own allocations alive now, by size of block (under 4 KiB, to 64 KiB,
/// to 1 MiB, larger): bytes, then blocks. "Our own" is the Rust code of
/// every crate in the process: in diagnostics builds [`Counting`] stands in
/// front of the selected allocator. With the system allocator, the memory
/// map's "process heap segments"
/// hold the blocks under about half a megabyte of ours, of the graphics
/// runtime and of the driver alike (3.5 GB in run 60's gameplay, growing by
/// some 2 MB a frame, and nothing said whose): this says which part is
/// ours. Bigger blocks are allocations of their own in the map.
static OWN: [Bank<8>; BANKS] = [const { Bank([const { AtomicU64::new(0) }; 8]) }; BANKS];

/// Count `i` of [`OWN`], all sets together (a block made by one thread and
/// released by another leaves one set below zero: the sum is right).
fn owned(i: usize) -> u64 {
    OWN.iter().fold(0u64, |sum, b| sum.wrapping_add(b.0[i].load(Relaxed)))
}

/// Formats already-collected byte counts for the optional category line.
/// Values are not summed here: several categories overlap, so only the
/// individual numbers are meaningful.
fn category_text(fields: &[(&str, u64)]) -> String {
    let mut out = String::new();
    for (name, bytes) in fields {
        if !out.is_empty() {
            out.push_str(", ");
        }
        out.push_str(&format!("{name} {:.0} MB", *bytes as f64 / (1 << 20) as f64));
    }
    out
}

/// The class of [`OWN`] a block of `size` bytes counts in.
fn block_class(size: usize) -> usize {
    match size {
        0..0x1000 => 0,
        0x1000..0x1_0000 => 1,
        0x1_0000..0x10_0000 => 2,
        _ => 3,
    }
}

fn own(size: usize, made: bool) {
    let class = block_class(size);
    let own = &OWN[bank()].0;
    if made {
        own[class].fetch_add(size as u64, Relaxed);
        own[4 + class].fetch_add(1, Relaxed);
    } else {
        own[class].fetch_sub(size as u64, Relaxed);
        own[4 + class].fetch_sub(1, Relaxed);
    }
}

/// Our own blocks of [`BIG`] bytes and more alive now: where each is (0:
/// none) and its bytes. The memory map lists the process's big allocations
/// by size without saying whose they are (run 65: one of 544 MB, one of
/// 401 MB, two of 174 MB, two of 115 MB); the sizes here say which of them
/// are ours.
static BIG_AT: [std::sync::atomic::AtomicUsize; 64] = [const { std::sync::atomic::AtomicUsize::new(0) }; 64];
static BIG_BYTES: [AtomicU64; 64] = [const { AtomicU64::new(0) }; 64];
const BIG: usize = 32 << 20;

/// A block of [`BIG`] bytes or more was made at `at`, or released there.
fn big(at: *mut u8, size: usize, made: bool) {
    if size < BIG {
        return;
    }
    for (place, bytes) in BIG_AT.iter().zip(&BIG_BYTES) {
        if made {
            if place.compare_exchange(0, at as usize, Relaxed, Relaxed).is_ok() {
                bytes.store(size as u64, Relaxed);
                return;
            }
        } else if place.load(Relaxed) == at as usize {
            bytes.store(0, Relaxed);
            place.store(0, Relaxed);
            return;
        }
    }
}

/// The sizes of our own big blocks alive, largest first, in MB.
fn big_blocks() -> String {
    let mut sizes: Vec<u64> = BIG_BYTES.iter().map(|b| b.load(Relaxed)).filter(|&b| b != 0).collect();
    sizes.sort_unstable_by(|a, b| b.cmp(a));
    let text: Vec<String> = sizes.iter().map(|b| format!("{:.2}", *b as f64 / (1 << 20) as f64)).collect();
    if text.is_empty() { "none".into() } else { text.join(", ") }
}

/// The selected allocator with every block counted in [`OWN`] (two atomic
/// additions per allocation and per release: diagnostics builds only).
struct Counting;

#[cfg(feature = "mimalloc")]
const INNER: mimalloc::MiMalloc = mimalloc::MiMalloc;
#[cfg(not(feature = "mimalloc"))]
const INNER: std::alloc::System = std::alloc::System;

// SAFETY: every call goes to the selected allocator unchanged; only
// counters are kept beside it.
unsafe impl std::alloc::GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        // SAFETY: the caller's contract, passed on.
        let at = unsafe { INNER.alloc(layout) };
        if !at.is_null() {
            own(layout.size(), true);
            big(at, layout.size(), true);
        }
        at
    }

    unsafe fn alloc_zeroed(&self, layout: std::alloc::Layout) -> *mut u8 {
        // SAFETY: the caller's contract, passed on.
        let at = unsafe { INNER.alloc_zeroed(layout) };
        if !at.is_null() {
            own(layout.size(), true);
            big(at, layout.size(), true);
        }
        at
    }

    unsafe fn dealloc(&self, at: *mut u8, layout: std::alloc::Layout) {
        own(layout.size(), false);
        big(at, layout.size(), false);
        // SAFETY: the caller's contract, passed on.
        unsafe { INNER.dealloc(at, layout) };
    }

    unsafe fn realloc(&self, at: *mut u8, layout: std::alloc::Layout, size: usize) -> *mut u8 {
        // SAFETY: the caller's contract, passed on.
        let moved = unsafe { INNER.realloc(at, layout, size) };
        if !moved.is_null() {
            own(layout.size(), false);
            own(size, true);
            big(at, layout.size(), false);
            big(moved, size, true);
        }
        moved
    }
}

#[global_allocator]
static HEAP: Counting = Counting;

#[cfg(test)]
mod command_tests {
    /// Commands are summed by pipeline and first word; those without a
    /// pipeline by their first two words.
    #[test]
    fn commands_are_named_by_their_pipeline() {
        use super::command_key;
        assert_eq!(
            command_key("dispatch 182x1x1 [compute 24f989b98dcb03f0] | 24f989b98dcb03f0: u4=0xa6b080 [heap 341380: x]"),
            "dispatch [compute 24f989b98dcb03f0]"
        );
        assert_eq!(
            command_key("indirect kind 10 max 1 at 0xd9358fce0 [mesh 0750cb2d975c7563, pixel 085d374c5a795549] | t"),
            "indirect [mesh 0750cb2d975c7563, pixel 085d374c5a795549]"
        );
        assert_eq!(command_key("texture upload 0x271b61c28b0 (2048x2048x1 f99 mips 10) sub 0"), "texture upload");
        assert_eq!(command_key("alias copy of 0x26f66aca780 into 0x26fbdc93890"), "alias copy");
    }
}

#[cfg(test)]
mod own_tests {
    use super::{block_class, category_text, owned};
    use std::sync::atomic::Ordering::Relaxed;

    #[test]
    fn the_category_line_formats_megabytes() {
        let fields = [("first", 1u64 << 20), ("zero", 0), ("last", 5u64 << 20)];
        assert_eq!(
            category_text(&fields),
            "first 1 MB, zero 0 MB, last 5 MB"
        );
        assert_eq!(fields[0].0, "first");
    }

    #[test]
    fn blocks_count_in_their_size_class() {
        assert_eq!(block_class(0), 0);
        assert_eq!(block_class(4095), 0);
        assert_eq!(block_class(4096), 1);
        assert_eq!(block_class(65535), 1);
        assert_eq!(block_class(65536), 2);
        assert_eq!(block_class((1 << 20) - 1), 2);
        assert_eq!(block_class(1 << 20), 3);
    }

    /// A block of 32 MiB or more is listed by its size while it lives.
    #[test]
    fn big_blocks_are_listed_while_they_live() {
        let size = (48usize << 20) + 4096;
        let listed = |at: usize| {
            super::BIG_AT
                .iter()
                .zip(&super::BIG_BYTES)
                .any(|(place, bytes)| place.load(Relaxed) == at && bytes.load(Relaxed) == size as u64)
        };
        let block = vec![0u8; size];
        let at = block.as_ptr() as usize;
        assert!(listed(at));
        assert!(super::big_blocks().contains("48.00"));
        drop(block);
        assert!(!listed(at));
    }

    /// A big block shows in the counters while it lives (other tests
    /// allocate small blocks meanwhile: only the big class is looked at,
    /// and only for at least this block).
    #[test]
    fn a_block_is_counted_while_it_lives() {
        let block = vec![1u8; 8 << 20];
        assert!(owned(3) >= 8 << 20);
        assert!(owned(7) >= 1);
        drop(block);
        assert!(owned(7) < u64::MAX / 2, "no count below zero");
    }
}

/// The process's memory in bytes: in physical memory, and committed
/// (private bytes).
pub(crate) fn process_memory() -> (u64, u64) {
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut c = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    (c.WorkingSetSize as u64, c.PagefileUsage as u64)
}

/// The PC's memory in bytes: physical memory free, and what is left to
/// commit.
fn system_memory() -> (u64, u64) {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    // SAFETY: plain data the call fills in.
    let mut m: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    m.dwLength = size_of::<MEMORYSTATUSEX>() as u32;
    // SAFETY: a sized structure.
    unsafe { GlobalMemoryStatusEx(&mut m) };
    (m.ullAvailPhys, m.ullAvailPageFile)
}

/// Every how many memory lines (one each 5 s) the address space is walked
/// for the memory map line ([`log_memory_map`]): `VICTORY_MEMORY_MAP=<n>`,
/// 60 if unset (about every 5 minutes; 6 before), `0` leaves the walk out.
/// (This module is in diagnostics builds only.)
///
/// The walk asks Windows about every region, 250,000 of them in a street
/// (0.9 s), and its questions hold the process's memory map: the submitting
/// thread's write-watch askings and the title's faults wait meanwhile (run
/// lane2-0: two askings of 11 and 13 ms, each ending with one of the walk's
/// questions).
fn memory_map() -> u64 {
    static EVERY: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *EVERY.get_or_init(|| match std::env::var("VICTORY_MEMORY_MAP") {
        Ok(v) => v.trim().parse().unwrap_or(60),
        Err(_) => 60,
    })
}

/// Whether to print the optional one-line category summary. Off by default.
fn memory_categories() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_MEMORY_CATEGORIES"))
}

/// Asks which of the pages in `asked` (address, answer) are in physical
/// memory and adds the bytes each stands for (kind, bytes) to its kind.
fn in_memory(asked: &mut Vec<[usize; 2]>, stands: &mut Vec<(usize, u64)>, present: &mut [u64; 10]) {
    use windows_sys::Win32::System::ProcessStatus::K32QueryWorkingSetEx;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    // SAFETY: our own process; `asked` holds that many entries of two
    // words each, as the call fills them in.
    let ok = unsafe {
        K32QueryWorkingSetEx(GetCurrentProcess(), asked.as_mut_ptr().cast(), (asked.len() * 16) as u32)
    } != 0;
    if ok {
        for (a, &(kind, bytes)) in asked.iter().zip(stands.iter()) {
            // Bit 0 of the answer: the page is in the process's memory.
            if a[1] & 1 != 0 {
                present[kind] += bytes;
            }
        }
    }
    asked.clear();
    stands.clear();
}

/// Logs what the process has committed by kind, from a walk of its address
/// space, and how much of each kind is in physical memory (one page in
/// sixteen is asked about). The kinds tell the title's own memory from
/// ours: the title's heaps and page views lie at console addresses (below
/// 1 TiB), everything the host allocates lies far above. Run 30, seen
/// from outside at 18.0 GB private: 4.6 GB the title's heaps, 2.6 GB in
/// 162 segments of the process heap, 2.7 GB in 707 allocations of 4 MiB
/// (windows of title memory), 0.46 GB the pipeline library's file; mapped
/// and so not private: 6.2 GB the title's page pool, 2.0 GB the driver's
/// shader cache file.
/// shortcut: the walk takes about 0.6 s on a thread of its own, and a
/// query of a region of gigabytes holds up other threads' memory calls
/// for its length (89 ms for the 6.2 GB pool), so it runs with every
/// sixth memory line only.
fn log_memory_map() {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_FREE, MEM_IMAGE, MEM_MAPPED, MEMORY_BASIC_INFORMATION, PAGE_WRITECOMBINE,
        VirtualQuery,
    };
    const KINDS: [&str; 10] = [
        "the title's heaps (private, console addresses)",
        "views of the title's pages (mapped, console addresses)",
        "the title's page pool (mapped, 1 GB and more)",
        "process heap segments (private, 15.8 MB each)",
        "windows of title memory and other 4 MiB allocations",
        "write-combined (buffers shared with the card)",
        "other private allocations of 1 MB and more",
        "other private allocations under 1 MB",
        "files and other mappings at host addresses",
        "programs and libraries",
    ];
    /// Console addresses lie below this.
    const HOST: usize = 1 << 40;
    /// One page asked about stands for this much.
    const STEP: usize = 64 << 10;
    let started = std::time::Instant::now();
    // Allocation, start, bytes, committed, type, protection.
    type Region = (usize, usize, usize, bool, u32, u32);
    let mut regions: Vec<Region> = Vec::new();
    let mut at = 0usize;
    loop {
        // SAFETY: plain data filled in by the query.
        let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        if unsafe { VirtualQuery(at as *const _, &mut m, size_of_val(&m)) } == 0 {
            break;
        }
        if m.State != MEM_FREE {
            regions.push((
                m.AllocationBase as usize,
                m.BaseAddress as usize,
                m.RegionSize,
                m.State == MEM_COMMIT,
                m.Type,
                m.Protect,
            ));
        }
        let next = (m.BaseAddress as usize).saturating_add(m.RegionSize);
        if next <= at {
            break;
        }
        at = next;
    }
    // Each allocation's whole size, reserved parts included.
    let mut whole: HashMap<usize, usize> = HashMap::new();
    for r in &regions {
        *whole.entry(r.0).or_default() += r.2;
    }
    let kind = |r: &Region| -> usize {
        let size = whole.get(&r.0).copied().unwrap_or(0);
        let console = r.0 < HOST;
        if r.4 == MEM_IMAGE {
            9
        } else if r.4 == MEM_MAPPED {
            match (console, size >= 1 << 30) {
                (false, _) => 8,
                (true, true) => 2,
                (true, false) => 1,
            }
        } else if console {
            0
        } else if size == 0xfd_0000 {
            3
        } else if size == 4 << 20 {
            4
        } else if r.5 & PAGE_WRITECOMBINE != 0 {
            5
        } else if size >= 1 << 20 {
            6
        } else {
            7
        }
    };
    let mut committed = [0u64; 10];
    let mut present = [0u64; 10];
    let mut asked: Vec<[usize; 2]> = Vec::with_capacity(4096);
    let mut stands: Vec<(usize, u64)> = Vec::with_capacity(4096);
    for r in regions.iter().filter(|r| r.3) {
        let k = kind(r);
        committed[k] += r.2 as u64;
        let mut offset = 0;
        while offset < r.2 {
            asked.push([r.1 + offset, 0]);
            stands.push((k, STEP.min(r.2 - offset) as u64));
            if asked.len() == 4096 {
                in_memory(&mut asked, &mut stands, &mut present);
            }
            offset += STEP;
        }
    }
    if !asked.is_empty() {
        in_memory(&mut asked, &mut stands, &mut present);
    }
    let text: Vec<String> = (0..KINDS.len())
        .filter(|&k| committed[k] > 0)
        .map(|k| format!("{} {} MB ({} MB in memory)", KINDS[k], committed[k] >> 20, present[k] >> 20))
        .collect();
    tracing::info!(
        "memory map ({} regions, {} ms): {}",
        regions.len(),
        started.elapsed().as_millis(),
        text.join("; ")
    );
    // The private allocations the kinds above leave as "other" (and the
    // write-combined ones), grouped by their size: the sizes name their
    // owners (the pipeline library's file to the byte, a snapshot buffer
    // of a title heap's size, the driver's blocks). Committed bytes by
    // allocation first, then allocations by (kind, whole size).
    let mut by_allocation: HashMap<usize, (usize, u64)> = HashMap::new();
    for r in regions.iter().filter(|r| r.3) {
        let k = kind(r);
        if k == 5 || k == 6 {
            by_allocation.entry(r.0).or_insert((k, 0)).1 += r.2 as u64;
        }
    }
    let mut groups: HashMap<(usize, usize), (u64, u64)> = HashMap::new();
    for (base, &(k, bytes)) in &by_allocation {
        let group = groups.entry((k, whole.get(base).copied().unwrap_or(0))).or_default();
        group.0 += 1;
        group.1 += bytes;
    }
    let mut groups: Vec<((usize, usize), (u64, u64))> = groups.into_iter().collect();
    groups.sort_by_key(|g| std::cmp::Reverse(g.1.1));
    let biggest: Vec<String> = groups
        .iter()
        .take(20)
        .map(|&((k, size), (n, bytes))| {
            format!(
                "{n} of {:.2} MB{}: {} MB",
                size as f64 / (1 << 20) as f64,
                if k == 5 { " write-combined" } else { "" },
                bytes >> 20
            )
        })
        .collect();
    tracing::info!(
        "memory map, other private allocations by size, the {} biggest groups of {} (committed): {}",
        biggest.len(),
        groups.len(),
        biggest.join("; ")
    );
}

/// Logs where the memory is: the process and the PC, then what this crate
/// holds by kind (alive now), then the host buffers made since the start.
fn log_memory() {
    use std::sync::atomic::Ordering::Relaxed;
    let mb = |v: u64| v as f64 / (1 << 20) as f64;
    let (working, private) = process_memory();
    let (free, left) = system_memory();
    let copies = crate::shadow::memory();
    let staged = crate::texture::memory();
    let pool = crate::pool::memory();
    let textures = [0, 1].map(|i| crate::resource::HOST_TEXTURES[i].load(Relaxed));
    let made: Vec<u64> = crate::host::MADE.iter().map(|m| m.load(Relaxed)).collect();
    let windows = crate::shadow::windows();
    let stagings = crate::texture::let_go();
    let census = crate::resource::census();
    let kinds: Vec<u64> = crate::resource::KINDS.iter().map(|k| k.load(Relaxed)).collect();
    let card = crate::host::host()
        .and_then(|h| h.video_memory())
        .unwrap_or_default();
    // Everything the system says of the two memory groups, beside the
    // budget and use the lines below print.
    if let Some(h) = crate::host::host() {
        h.log_video_memory("with the memory line");
    }
    // What else holds memory: the map, on a thread of its own, with every
    // sixth line; and what the shader table and the pipeline library keep.
    static LINES: AtomicU64 = AtomicU64::new(0);
    static WALKING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    let every = memory_map();
    if every > 0 && LINES.fetch_add(1, Relaxed) % every == every - 1 && !WALKING.swap(true, Relaxed) {
        let walk = std::thread::Builder::new().name("victory-memory-map".into()).spawn(|| {
            log_memory_map();
            WALKING.store(false, Relaxed);
        });
        if walk.is_err() {
            WALKING.store(false, Relaxed);
        }
    }
    let kept = crate::pipeline::kept();
    tracing::info!(
        "memory kept beside the lines below: {} shader containers {:.0} MB, {} translations \
         {:.0} MB; the pipeline library's file {:.0} MB (its last save wrote {:.0} MB)",
        kept[0],
        mb(kept[1]),
        kept[2],
        mb(kept[3]),
        mb(kept[4]),
        mb(kept[5]),
    );
    tracing::info!(
        "memory of textures: {} title textures have no host copy yet ({:.0} MB of title memory); \
         host copies the host renders to or writes: {} alive {:.0} MB; host copies the title \
         fills: {} alive {:.0} MB ({} of them placed), {} let go {:.0} MB and {} made again \
         {:.0} MB since the start; {} host copies made in all; passes that made none for lack of \
         room: {}; the card: {:.0} MB in use of a budget of {:.0} MB, {:.0} MB of the PC's memory \
         shared with it in use",
        census[2],
        mb(census[3]),
        census[0],
        mb(census[1]),
        census[4],
        mb(census[5]),
        census[6],
        kinds[2],
        mb(kinds[3]),
        kinds[4],
        mb(kinds[5]),
        kinds[6],
        crate::draw::HEAP_TEXTURES_FULL.load(Relaxed),
        mb(card[1]),
        mb(card[0]),
        mb(card[3]),
    );
    {
        let [textures, staging] = pool;
        tracing::info!("memory in blocks (VICTORY_TEXTURE_POOL=0 turns them off): {textures} MB of textures, {staging} MB of their staging buffers");
    }
    let targets = crate::resource::targets_census();
    tracing::info!(
        "host copies the sweep never lets go: {} the host renders to or writes {:.0} MB ({} of them placed {:.0} MB);          of them {} have title memory that is not mapped now {:.0} MB, {} were not used for 30 s and are held by no          list and no slot {:.0} MB, {} are both {:.0} MB; and {} copies of textures the title fills that the host          copied into {:.0} MB",
        targets[0],
        mb(targets[1]),
        targets[10],
        mb(targets[11]),
        targets[2],
        mb(targets[3]),
        targets[4],
        mb(targets[5]),
        targets[6],
        mb(targets[7]),
        targets[8],
        mb(targets[9]),
    );
    tracing::info!(
        "memory let go since the start: {} windows outside buffers {:.0} MB, {} staging buffers \
         {:.0} MB; of the windows alive, {:.0} MB lie between the lowest and highest address \
         asked for in each",
        windows[0],
        mb(windows[1]),
        stagings[0],
        mb(stagings[1]),
        mb(windows[2]),
    );
    tracing::info!(
        "memory: {:.0} MB private ({:.0} MB of it in memory; the PC has {:.0} MB free and {:.0} MB \
         left to commit); host copies of title memory: {} alive {:.0} MB, of them {} windows \
         outside buffers {:.0} MB; snapshot buffers kept: {} reserving {:.0} MB; staging of \
         textures: {} alive {:.0} MB, {} kept by their texture; host textures: {} alive {:.0} MB, \
         {} placed in title memory; made so far: {} upload buffers {:.0} MB, {} shared buffers \
         {:.0} MB, {} card buffers {:.0} MB",
        mb(private),
        mb(working),
        mb(free),
        mb(left),
        copies[0],
        mb(copies[1]),
        copies[2],
        mb(copies[3]),
        copies[4],
        mb(copies[5]),
        staged[0],
        mb(staged[1]),
        staged[2],
        textures[0],
        mb(textures[1]),
        crate::resource::placed_textures(),
        made[0],
        mb(made[1]),
        made[2],
        mb(made[3]),
        made[4],
        mb(made[5]),
    );
    tracing::info!(
        "memory, placed textures: {} of them were asked for again with other filler words in their description          (one resource and host copy each since the filler left the key; VICTORY_PLACED_FILLER=1 keeps it)",
        crate::resource::placed_filler_twins()
    );
    // What the line above leaves open: what is alive of the buffers "made
    // so far", what the windows hold and what holds them, and what our own
    // code has allocated.
    let alive: Vec<u64> = crate::host::ALIVE.iter().map(|a| a.load(Relaxed)).collect();
    let window = crate::shadow::window_use();
    let (views, [constant, shader, unordered, sampler, console_sampler, placed]) =
        crate::descriptor::views_kept();
    let own: Vec<u64> = (0..8).map(owned).collect();
    tracing::info!(
        "memory, what holds it: {} host lists alive; their scratch memory: {} chunks {:.0} MB, of          them {} chunks {:.0} MB spare ({} taken from the spare ones so far); their other upload          buffers: {} alive {:.0} MB; windows outside buffers (grid {} KiB): {:.0} MB of their          pages ever copied, {} shown by descriptor slots {:.0} MB, {} held by lists or the pass          over the heap {:.0} MB, {} asked for without a length {:.0} MB; buffer views of the heap          whose slot showed another copy than the one brought up to date: {}; distinct views          kept: {} ({} free) {:.0} MB; our own allocations: {:.0} MB in {} blocks (under 4 KiB          {:.0} MB in {}, to 64 KiB {:.0} MB in {}, to 1 MiB {:.0} MB in {}, larger {:.0} MB in {})",
        alive[0],
        alive[1],
        mb(alive[2]),
        alive[3],
        mb(alive[4]),
        alive[5],
        alive[6],
        mb(alive[7]),
        window[7] >> 10,
        mb(window[0]),
        window[1],
        mb(window[2]),
        window[3],
        mb(window[4]),
        window[5],
        mb(window[6]),
        crate::draw::HEAP_VIEWS_ELSEWHERE.load(Relaxed),
        views[0],
        views[1],
        mb(views[2]),
        mb(own[..4].iter().sum()),
        own[4..].iter().sum::<u64>(),
        mb(own[0]),
        own[4],
        mb(own[1]),
        own[5],
        mb(own[2]),
        own[6],
        mb(own[3]),
        own[7],
    );
    // Round 2: what the view table still holds and what goes around it,
    // host list counts and memory sampling, the card's share that is the layer's own or
    // the title's no more, and which big blocks are ours.
    let lists: Vec<u64> = crate::host::LISTS.iter().map(|l| l.load(Relaxed)).collect();
    tracing::info!(
        "memory, more: views named whole in their records since the start: {} constant buffer \
         views, {} views of placed memory; of the view table's entries {} are constant buffer \
         views and {} views of placed memory; shapes of placed views kept: {}; host lists: {} \
         made by the host, {} preludes made for them, summed process-private byte delta {:.0} MB \
         across list-creation intervals (includes other threads); twins of depth textures: {:.0} MB; host copies of \
         placed textures whose title memory is not mapped any more (estimate): {:.0} MB, of \
         them still shown by a descriptor slot {:.0} MB; the pipeline library's mapped file \
         let go from memory so far: {:.0} MB; our own blocks of 32 MB and more, in MB: {}",
        views[6],
        views[7],
        views[3],
        views[4],
        views[5],
        lists[0],
        lists[1],
        lists[2] as i64 as f64 / (1 << 20) as f64,
        mb(census[7]),
        mb(census[8]),
        mb(census[9]),
        mb(kept[6]),
        big_blocks(),
    );
    tracing::info!(
        "memory, view slots (occupied+free, slot MB): constant {}+{} {:.2}, shader {}+{} {:.2}, \
         unordered {}+{} {:.2}, sampler {}+{} {:.2}, console sampler {}+{} {:.2}, placed {}+{} {:.2}",
        constant.occupied, constant.free, mb(constant.slot_bytes()),
        shader.occupied, shader.free, mb(shader.slot_bytes()),
        unordered.occupied, unordered.free, mb(unordered.slot_bytes()),
        sampler.occupied, sampler.free, mb(sampler.slot_bytes()),
        console_sampler.occupied, console_sampler.free, mb(console_sampler.slot_bytes()),
        placed.occupied, placed.free, mb(placed.slot_bytes()),
    );
    // Round 3: whether windows hold the same memory more than once (they
    // can, once they are cut finer than 4 MiB), and how many host lists
    // the title made and never used.
    tracing::info!(
        "memory, windows: the {:.0} MB of windows lie over {:.0} MB of title memory (the rest is \
         memory that is in more than one window, and is copied once for each); host lists never \
         reset since they were made: {}",
        mb(copies[3]),
        mb(crate::shadow::window_cover()),
        lists[3],
    );
    if memory_categories() {
        let fields: &[(&str, u64)] = &[
            ("working set", working),
            ("private total", private),
            ("free physical", free),
            ("commit headroom", left),
            ("host lists alive", alive[0]),
            ("lists never reset", lists[3]),
            ("list scratch live", alive[2]),
            ("list scratch spare", alive[4]),
            ("list other uploads", alive[7]),
            ("private growth making lists", lists[2]),
            ("title window copies", copies[1]),
            ("window pages copied", window[0]),
            ("window held by lists", window[4]),
            ("window without a length", window[6]),
            ("texture host copies", census[1]),
            ("texture title-fill copies", census[5]),
            ("texture pool", pool[0] << 20),
            ("staging pool", pool[1] << 20),
            ("live texture staging", staged[1]),
            ("descriptor views", views[2]),
            ("shader containers", kept[1]),
            ("translations", kept[3]),
            ("pipeline library file", kept[4]),
            ("our rust heap", own[..4].iter().sum()),
            ("made upload buffers so far", made[1]),
            ("made shared buffers so far", made[3]),
            ("made card buffers so far", made[5]),
        ];
        tracing::info!(
            "memory by category (VICTORY_MEMORY_CATEGORIES=1, approximate and overlapping, not a sum): {}",
            category_text(fields)
        );
    }
}

/// Microseconds since the first call.
fn now_us() -> u64 {
    static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    EPOCH.get_or_init(std::time::Instant::now).elapsed().as_micros() as u64
}

/// When the last frame was presented and when the last submission returned
/// ([`now_us`]), and whether a list call and a submission followed that
/// present yet.
static PRESENTED_AT: AtomicU64 = AtomicU64::new(0);
static SUBMITTED_AT: AtomicU64 = AtomicU64::new(0);
static RECORDED_SINCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
static SUBMITTED_SINCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// What a thread that submits did between its submissions and in them
/// (microseconds; list calls in nanoseconds).
#[derive(Default, Clone, Copy)]
struct Submitter {
    submissions: u64,
    /// From a submission's return to the next one's call: wall time, of it
    /// the thread running (user, kernel), and inside our list calls.
    between: u64,
    between_user: u64,
    between_kernel: u64,
    between_ours: u64,
    /// Waiting for the submission lock; running inside submissions.
    lock: u64,
    inside_user: u64,
    inside_kernel: u64,
}

/// By thread: the submitting threads, and list-call time (nanoseconds,
/// calls) of every recording thread.
static SUBMITTERS: Mutex<Vec<(u32, Submitter)>> = Mutex::new(Vec::new());
static RECORDERS: Mutex<Vec<(u32, u64, u64)>> = Mutex::new(Vec::new());

thread_local! {
    /// This thread's last submission: when it returned ([`now_us`]) and the
    /// thread's processor time then (user, kernel).
    static LEFT: std::cell::Cell<Option<(u64, u64, u64)>> = const { std::cell::Cell::new(None) };
    /// The thread's processor time when its running submission was called.
    static ENTERED: std::cell::Cell<(u64, u64)> = const { std::cell::Cell::new((0, 0)) };
    /// List-call nanoseconds of this thread since its last submission.
    static OURS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    /// List-call time not yet added to `RECORDERS` (nanoseconds, calls).
    static UNSENT: std::cell::Cell<(u64, u64)> = const { std::cell::Cell::new((0, 0)) };
}

fn thread_id() -> u32 {
    // SAFETY: a plain query.
    unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() }
}

fn micros(t: windows_sys::Win32::Foundation::FILETIME) -> u64 {
    (u64::from(t.dwHighDateTime) << 32 | u64::from(t.dwLowDateTime)) / 10
}

/// This thread's processor time so far (user, kernel), in microseconds.
fn thread_times() -> (u64, u64) {
    use windows_sys::Win32::Foundation::FILETIME;
    use windows_sys::Win32::System::Threading::{GetCurrentThread, GetThreadTimes};
    let zero = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
    let (mut created, mut exited, mut kernel, mut user) = (zero, zero, zero, zero);
    // SAFETY: our own thread and four valid out pointers.
    unsafe { GetThreadTimes(GetCurrentThread(), &mut created, &mut exited, &mut kernel, &mut user) };
    (micros(user), micros(kernel))
}

/// The process's processor time so far (user, kernel), in microseconds.
fn process_times() -> (u64, u64) {
    use windows_sys::Win32::Foundation::FILETIME;
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetProcessTimes};
    let zero = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
    let (mut created, mut exited, mut kernel, mut user) = (zero, zero, zero, zero);
    // SAFETY: our own process and four valid out pointers.
    unsafe { GetProcessTimes(GetCurrentProcess(), &mut created, &mut exited, &mut kernel, &mut user) };
    (micros(user), micros(kernel))
}

fn submitter(all: &mut Vec<(u32, Submitter)>, id: u32) -> &mut Submitter {
    let at = match all.iter().position(|s| s.0 == id) {
        Some(at) => at,
        None => {
            all.push((id, Submitter::default()));
            all.len() - 1
        }
    };
    &mut all[at].1
}

/// A list call on this thread that began at `started` has returned.
pub(crate) fn recorded(started: std::time::Instant) {
    let took = started.elapsed();
    let ns = took.as_nanos() as u64;
    crate::timeline::called(took);
    add(C::RecordCalls, 1);
    add(C::RecordNs, ns);
    if !RECORDED_SINCE.load(Relaxed) && !RECORDED_SINCE.swap(true, Relaxed) {
        let began = now_us().saturating_sub(ns / 1000);
        add(C::UntilRecordUs, began.saturating_sub(PRESENTED_AT.load(Relaxed)));
    }
    OURS.with(|o| o.set(o.get() + ns));
    let (unsent, calls) = UNSENT.with(|u| u.get());
    let (unsent, calls) = (unsent + ns, calls + 1);
    // By thread, a couple of milliseconds at a time.
    if unsent < 2_000_000 {
        UNSENT.with(|u| u.set((unsent, calls)));
        return;
    }
    UNSENT.with(|u| u.set((0, 0)));
    let id = thread_id();
    let mut all = RECORDERS.lock().unwrap_or_else(|e| e.into_inner());
    match all.iter_mut().find(|r| r.0 == id) {
        Some(r) => (r.1, r.2) = (r.1 + unsent, r.2 + calls),
        None => all.push((id, unsent, calls)),
    }
}

/// A submission is called on this thread (before it waits for the
/// submission lock).
pub(crate) fn submission_begins() {
    let now = now_us();
    INSIDE.fetch_add(1, Relaxed);
    if !SUBMITTED_SINCE.load(Relaxed) && !SUBMITTED_SINCE.swap(true, Relaxed) {
        add(C::UntilSubmitUs, now.saturating_sub(PRESENTED_AT.load(Relaxed)));
    }
    let (user, kernel) = thread_times();
    ENTERED.with(|e| e.set((user, kernel)));
    let ours = OURS.with(|o| o.replace(0));
    if let Some((left, u, k)) = LEFT.with(|l| l.get()) {
        let mut all = SUBMITTERS.lock().unwrap_or_else(|e| e.into_inner());
        let s = submitter(&mut all, thread_id());
        s.between += now.saturating_sub(left);
        s.between_user += user.saturating_sub(u);
        s.between_kernel += kernel.saturating_sub(k);
        s.between_ours += ours;
    }
}

/// The submission returns; it waited `lock` microseconds for the
/// submission lock.
pub(crate) fn submission_ends(lock: u64) {
    let now = now_us();
    INSIDE.fetch_sub(1, Relaxed);
    SUBMITTED_AT.store(now, Relaxed);
    let (user, kernel) = thread_times();
    let (u, k) = ENTERED.with(|e| e.get());
    LEFT.with(|l| l.set(Some((now, user, kernel))));
    OURS.with(|o| o.set(0));
    let mut all = SUBMITTERS.lock().unwrap_or_else(|e| e.into_inner());
    let s = submitter(&mut all, thread_id());
    s.submissions += 1;
    s.lock += lock;
    s.inside_user += user.saturating_sub(u);
    s.inside_kernel += kernel.saturating_sub(k);
}

/// The submissions being prepared now.
static INSIDE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// What the frame is at, for the sampling profiler (`VICTORY_PROFILE`): 3
/// in a submission, else 0 after a present until the first list call, 1 until
/// the first submission, 2 behind it.
pub(crate) fn phase() -> u32 {
    if INSIDE.load(Relaxed) > 0 {
        3
    } else if SUBMITTED_SINCE.load(Relaxed) {
        2
    } else {
        u32::from(RECORDED_SINCE.load(Relaxed))
    }
}

/// The title calls its present (before our own copy to the window runs).
pub(crate) fn present_begins() {
    add(C::TailUs, now_us().saturating_sub(SUBMITTED_AT.load(Relaxed)));
}

/// A frame was presented.
pub(crate) fn presented() {
    add(C::Frames, 1);
    PRESENTED_AT.store(now_us(), Relaxed);
    RECORDED_SINCE.store(false, Relaxed);
    SUBMITTED_SINCE.store(false, Relaxed);
}

fn page_faults() -> u64 {
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut c = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    u64::from(c.PageFaultCount)
}

/// Logs the counters per frame, and the ranges that took most bytes. The
/// counters run from line to line, so they are divided by the frames
/// presented since the last line, not by `frames` (the frames of the last
/// second or more: a 4 s frame after an unlogged 4 s frame showed two
/// frames' counts as one's).
pub(crate) fn log(frames: f64) {
    let counted = taken(C::Frames as usize);
    let frames = if counted > 0 { counted as f64 } else { frames };
    static FAULTS: AtomicU64 = AtomicU64::new(0);
    let faults = page_faults();
    let faults = faults - FAULTS.swap(faults, Relaxed).min(faults);
    let c: Vec<f64> = (0..COUNTERS).map(|i| taken(i) as f64 / frames).collect();
    let mb = |v: f64| v / 1e6;
    let ms = |v: f64| v / 1e3;
    tracing::info!(
        "copies per frame: {:.1} submissions; {:.0} title copies {:.1} MB; {:.0} refreshes {:.1} MB, \
         {:.0} skipped {:.1} MB; {:.0} runs {:.1} ms; {:.0} queries {:.2} ms, moves {:.2} ms, \
         {:.2} MB unmapped; {:.0} page faults; {:.0} watches {:.2} ms, {:.1} MB checked, {:.1} MB \
         unchanged; {:.1} fallbacks; {:.0} guard faults; textures {:.1} MB watched and \
         changed, {:.1} MB hashed, {:.1} MB of them changed, {:.1} changes the guard did not see",
        c[C::Submissions as usize],
        c[C::Copies as usize],
        mb(c[C::CopyBytes as usize]),
        c[C::Refreshes as usize],
        mb(c[C::RefreshBytes as usize]),
        c[C::Skipped as usize],
        mb(c[C::SkippedBytes as usize]),
        c[C::Runs as usize],
        ms(c[C::RunUs as usize]),
        c[C::Queries as usize],
        ms(c[C::QueryUs as usize]),
        ms(c[C::MoveUs as usize]),
        mb(c[C::Unmapped as usize]),
        faults as f64 / frames,
        c[C::Watches as usize],
        ms(c[C::WatchUs as usize]),
        mb(c[C::Checked as usize]),
        mb(c[C::Unchanged as usize]),
        c[C::Fallbacks as usize],
        c[C::GuardFaults as usize],
        mb(c[C::TexWatched as usize]),
        mb(c[C::TexHashed as usize]),
        mb(c[C::TexChanged as usize]),
        c[C::TexHot as usize],
    );
    tracing::info!(
        "recording per frame ({frames:.0} frames): {:.0} list calls {:.1} ms (all recording \
         threads); {:.1} ms waiting for pipelines; dynamic slots: {:.0} tables asked, {:.0} \
         refused ({:.1} with enough slots free, but in pieces), {:.1} ms searching, {:.0} slots taken, \
         {:.1} tables waited {:.1} ms for flights to land, {:.1} refused again (dropped); notes {:.1} ms; \
         shader-written ranges: {:.0} steps in \
         {:.1} submissions, snapshots {:.1} ms, {:.1} MB compared back; sparse reads {:.1} ms; \
         guards {:.1} ms; {:.1} MB copied whole (memory not watched); {:.1} frame waits \
         sleeping {:.1} ms (all threads); ranges of shaders that cannot store to buffers: \
         {:.0} steps {:.1} MB left out",
        c[C::RecordCalls as usize],
        c[C::RecordNs as usize] / 1e6,
        ms(c[C::PipelineWaitUs as usize]),
        c[C::DynamicAsked as usize],
        c[C::DynamicRefused as usize],
        c[C::DynamicPieces as usize],
        c[C::DynamicNs as usize] / 1e6,
        c[C::DynamicSlots as usize],
        c[C::DynamicSettles as usize],
        ms(c[C::DynamicSettleUs as usize]),
        c[C::DynamicLost as usize],
        c[C::NoteNs as usize] / 1e6,
        c[C::GpuWrites as usize],
        c[C::BackSubmissions as usize],
        ms(c[C::SnapshotUs as usize]),
        mb(c[C::BackCompared as usize]),
        ms(c[C::SparseUs as usize]),
        ms(c[C::GuardUs as usize]),
        mb(c[C::WholeBytes as usize]),
        c[C::PaceWaits as usize],
        ms(c[C::PaceUs as usize]),
        c[C::GpuWritesLeftOut as usize],
        mb(c[C::GpuWritesLeftOutBytes as usize]),
    );
    let [together, written, mapped] =
        [0, 1, 2].map(|n| crate::watch::ALIASES[n].load(std::sync::atomic::Ordering::Relaxed));
    tracing::info!(
        "guards over several views of a page, since the start: {together} blocks guarded with their other views, \
         {written} views opened by a write through another, {mapped} ranges opened because a view of them was mapped"
    );
    tracing::info!(
        "waits for the write trackers' locks per frame, all threads (number, time): {}",
        crate::watch::lock_waits(frames)
    );
    let [copies, blocks, opened, stores, unknown, full, stayed, copied_same, records] =
        [0, 1, 2, 3, 4, 5, 6, 7, 8].map(|n| crate::titlewrite::DONE[n].load(std::sync::atomic::Ordering::Relaxed));
    tracing::info!(
        "the title's writes beside the guards, since the start: {copies} copies (VICTORY_COPY_BESIDE=1), {blocks} blocks, {opened} opened; \
         {stores} stores (VICTORY_STORE_BESIDE=1), {unknown} not understood, {full} blocks opened for too many; left out for changing nothing: {stayed} stores, {copied_same} parts of copies, {records} copies of descriptors; \
         the busiest stores (left out/changed): {}",
        crate::titlewrite::busiest()
    );
    // Where the wall clock of a frame goes: the phases of a frame, the
    // processor time of the whole process, and what each submitting and
    // recording thread did.
    static CPU: Mutex<(u64, u64)> = Mutex::new((0, 0));
    let (user, kernel) = process_times();
    let (user, kernel) = {
        let mut last = CPU.lock().unwrap_or_else(|e| e.into_inner());
        let spent = (user.saturating_sub(last.0), kernel.saturating_sub(last.1));
        *last = (user, kernel);
        (spent.0 as f64 / 1e3 / frames, spent.1 as f64 / 1e3 / frames)
    };
    tracing::info!(
        "frame account per frame: after a present {:.1} ms until the first list call and {:.1} ms \
         until the first submission; {:.1} ms from the last submission to the present call; \
         lists reset in {:.1} ms; the process ran {user:.0} ms user + {kernel:.0} ms kernel on \
         all threads; what shaders wrote: {:.1} ms finding {:.0} pieces that differ, {:.1} ms \
         putting them back; texture uploads: {:.0} recorded, {:.0} left out; {:.1} held \
         submissions run at the present with their waits not met; waits for memory values: \
         {:.0} looks {:.1} ms; {:.0} host lists made {:.1} ms; {:.0} parts of held submissions \
         ran alone that the next part could have run with",
        ms(c[C::UntilRecordUs as usize]),
        ms(c[C::UntilSubmitUs as usize]),
        ms(c[C::TailUs as usize]),
        c[C::ResetNs as usize] / 1e6,
        ms(c[C::BackScanUs as usize]),
        c[C::BackDiffer as usize],
        ms(c[C::BackSyncUs as usize]),
        c[C::TexUploads as usize],
        c[C::TexSkipped as usize],
        c[C::Forced as usize],
        c[C::WaitChecks as usize],
        c[C::WaitCheckNs as usize] / 1e6,
        c[C::HostLists as usize],
        c[C::HostListNs as usize] / 1e6,
        c[C::PartsTogether as usize],
    );
    // What is copied whole, largest first: memory nobody tells us about.
    let whole = WHOLE.lock().unwrap_or_else(|e| e.into_inner()).take();
    let mut whole: Vec<_> = whole.into_iter().flatten().collect();
    whole.sort_by_key(|&((_, len), (n, _))| std::cmp::Reverse(len as u64 * n));
    tracing::info!(
        "  copied whole (memory not watched): {:.1} ms per frame in {} ranges; the largest follow",
        ms(c[C::WholeUs as usize]),
        whole.len()
    );
    for ((src, len), (n, us)) in whole.into_iter().take(6) {
        tracing::info!(
            "  copied whole from {src:#x}+{len:#x}: {:.1}x/frame, {:.2} MB/frame, {:.2} ms/frame",
            n as f64 / frames,
            (len as u64 * n) as f64 / 1e6 / frames,
            us as f64 / 1e3 / frames
        );
    }
    // The waiting for the card: many short waits or a few long ones, how
    // much of it is for submissions the title gets nothing back from, and
    // the longest one.
    let longest = std::mem::take(&mut *LONGEST.lock().unwrap_or_else(|e| e.into_inner()));
    tracing::info!(
        "  waiting for the card per frame: {:.1} submissions under 2 ms ({:.1} ms), {:.1} of 2-8 ms \
         ({:.1} ms), {:.1} of 8-32 ms ({:.1} ms), {:.1} longer ({:.1} ms); {:.1} submissions hand \
         nothing back to the title ({:.1} ms of the waiting); the longest since the last line: \
         {:.1} ms, {}",
        c[C::CardUnder2 as usize],
        ms(c[C::CardUnder2Us as usize]),
        c[C::Card2To8 as usize],
        ms(c[C::Card2To8Us as usize]),
        c[C::Card8To32 as usize],
        ms(c[C::Card8To32Us as usize]),
        c[C::CardOver32 as usize],
        ms(c[C::CardOver32Us as usize]),
        c[C::NothingBack as usize],
        ms(c[C::NothingBackUs as usize]),
        longest.0 as f64 / 1e3,
        longest.1
    );
    let by = WAITS_BY_WORK.lock().unwrap_or_else(|e| e.into_inner()).take();
    let mut by: Vec<_> = by.into_iter().flatten().collect();
    by.sort_by_key(|&(_, (_, us))| std::cmp::Reverse(us));
    for (what, (n, us)) in by.into_iter().take(12) {
        tracing::info!(
            "  the card's time by what ran: {:.1} ms a frame in {:.1} submissions a frame: {}",
            us as f64 / 1e3 / frames,
            n as f64 / frames,
            what.chars().take(700).collect::<String>()
        );
    }
    // The lists of a timing run by kind: the title's, what of their time
    // their commands account for, and ours.
    let (timed, least) = std::mem::replace(&mut *LISTS_TIMED.lock().unwrap_or_else(|e| e.into_inner()), ([[0; 3]; 5], u64::MAX));
    if timed.iter().any(|t| t[0] > 0) {
        let per = |us: u64| us as f64 / 1e3 / frames;
        let n = |c: u64| c as f64 / frames;
        let all: u64 = timed.iter().map(|t| t[1]).sum();
        let lists: u64 = timed.iter().map(|t| t[0]).sum();
        let [prelude, title, steps, compare, other] = timed;
        tracing::info!(
            "  the card's time by list (timing run), per frame, from sending to the signal: {:.1} ms in {:.0} lists; the \
             title's {:.1} ms in {:.0} (their noted commands {:.1}, the rest {:.1}: what of ours is recorded among them, \
             barriers, and the sending); preludes {:.1} ms in {:.0} (refreshes of copies); steps on the card {:.1} ms in \
             {:.0}; compares of copies in the card's memory {:.1} ms in {:.0}; others of ours {:.1} ms in {:.0}; the least \
             a list took {:.0} us ({:.1} ms a frame at that for every list)",
            per(all),
            n(lists),
            per(title[1]),
            n(title[0]),
            per(title[2]),
            per(title[1].saturating_sub(title[2])),
            per(prelude[1]),
            n(prelude[0]),
            per(steps[1]),
            n(steps[0]),
            per(compare[1]),
            n(compare[0]),
            per(other[1]),
            n(other[0]),
            if least == u64::MAX { 0 } else { least },
            if least == u64::MAX { 0.0 } else { per(least * lists) },
        );
    }
    let by = COMMANDS_BY_WORK.lock().unwrap_or_else(|e| e.into_inner()).take();
    let mut by: Vec<_> = by.into_iter().flatten().collect();
    by.sort_by_key(|&(_, (_, us, _))| std::cmp::Reverse(us));
    // All of them together, and by their first word: the forty lines below
    // are the heaviest only (100 of 226 ms waited in run 102), and what the
    // commands do not account for is the layer's own lists (the card
    // copies' passes, preludes) and the waits themselves.
    if !by.is_empty() {
        let mut kinds: Vec<(String, u64, u64)> = Vec::new();
        for (what, (n, us, _)) in &by {
            let word = what.split(' ').next().unwrap_or("").to_owned();
            match kinds.iter_mut().find(|k| k.0 == word) {
                Some(k) => (k.1, k.2) = (k.1 + n, k.2 + us),
                None => kinds.push((word, *n, *us)),
            }
        }
        kinds.sort_by_key(|k| std::cmp::Reverse(k.2));
        let all: u64 = kinds.iter().map(|k| k.2).sum();
        let text: Vec<String> =
            kinds.iter().map(|(word, n, us)| format!("{word} {:.1} ms in {:.0}", *us as f64 / 1e3 / frames, *n as f64 / frames)).collect();
        tracing::info!(
            "  the card's time by command (timing run), all {} kinds together: {:.1} ms a frame; by first word: {}",
            by.len(),
            all as f64 / 1e3 / frames,
            text.join(", ")
        );
    }
    for (what, (n, us, longest)) in by.into_iter().take(40) {
        tracing::info!(
            "  the card's time by command (timing run): {:.2} ms a frame in {:.1} commands a frame, the longest {:.2} ms: {what}",
            us as f64 / 1e3 / frames,
            n as f64 / frames,
            longest as f64 / 1e3
        );
    }
    tracing::info!(
        "  waits for every earlier shader write, per frame: {:.0} for the title's barrier groups, {:.0} \
         for its barriers on buffers; {:.0} indirect commands bounded (a dispatch of ours between two \
         such waits each); barrier experiment {}",
        c[C::WaitsForGroups as usize],
        c[C::WaitsForBuffers as usize],
        c[C::IndirectBounded as usize],
        barrier_test()
    );
    // What is compared after submissions, largest first, and how much of
    // it shaders had written.
    let compared = COMPARED.lock().unwrap_or_else(|e| e.into_inner()).take();
    let mut compared: Vec<_> = compared.into_iter().flatten().collect();
    compared.sort_by_key(|&((_, len), (n, _))| std::cmp::Reverse(len as u64 * n));
    tracing::info!("  compared after submissions: {} ranges; the largest follow", compared.len());
    for ((src, len), (n, written)) in compared.into_iter().take(8) {
        tracing::info!(
            "  compared {src:#x}+{len:#x}: {:.1}x/frame, {:.2} MB/frame, {:.3} MB/frame of it written back",
            n as f64 / frames,
            (len as u64 * n) as f64 / 1e6 / frames,
            written as f64 / 1e6 / frames
        );
    }
    let mut submitters = std::mem::take(&mut *SUBMITTERS.lock().unwrap_or_else(|e| e.into_inner()));
    submitters.sort_by_key(|s| std::cmp::Reverse(s.1.submissions));
    for (id, s) in submitters.into_iter().take(4) {
        let per = |v: u64| v as f64 / 1e3 / frames;
        tracing::info!(
            "  thread {id} submitted {:.1}x/frame: between its submissions {:.1} ms (running \
             {:.1} ms user + {:.1} ms kernel; {:.1} ms in list calls), in them running {:.1} ms \
             user + {:.1} ms kernel, {:.1} ms waiting for the submission lock",
            s.submissions as f64 / frames,
            per(s.between),
            per(s.between_user),
            per(s.between_kernel),
            s.between_ours as f64 / 1e6 / frames,
            per(s.inside_user),
            per(s.inside_kernel),
            per(s.lock),
        );
    }
    let mut recorders = std::mem::take(&mut *RECORDERS.lock().unwrap_or_else(|e| e.into_inner()));
    recorders.sort_by_key(|r| std::cmp::Reverse(r.1));
    let busiest: Vec<String> = recorders
        .iter()
        .take(6)
        .map(|r| format!("thread {} {:.0} ms in {:.0} calls", r.0, r.1 as f64 / 1e6 / frames, r.2 as f64 / frames))
        .collect();
    tracing::info!("  list calls by thread: {} threads; {}", recorders.len(), busiest.join(", "));
    log_memory();
    #[cfg(feature = "diagnostics")]
    crate::watch::alone_line(frames);
    let ranges = RANGES.lock().unwrap_or_else(|e| e.into_inner()).take();
    let mut ranges: Vec<_> = ranges.into_iter().flatten().collect();
    ranges.sort_by_key(|&(_, (_, bytes, _, _))| std::cmp::Reverse(bytes));
    // The kind of memory of each range, asked once per address: a question
    // walks the page tables to the end of the pages alike, up to 10 ms in
    // the title's heaps, and holds the memory map meanwhile (run lane2-0:
    // 10 ms on the submitting thread, when this ran there). A line may name
    // the kind an address had when first seen.
    static KINDS: Mutex<Option<HashMap<usize, u32>>> = Mutex::new(None);
    for ((src, len), (n, bytes, us, _)) in ranges.into_iter().take(10) {
        use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
        let known = KINDS.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(HashMap::new).get(&src).copied();
        let kind = known.unwrap_or_else(|| {
            // SAFETY: plain data filled in by the query.
            let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            // SAFETY: a valid out pointer of the right size.
            unsafe { VirtualQuery(src as *const _, &mut info, size_of_val(&info)) };
            KINDS.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(HashMap::new).insert(src, info.Type);
            info.Type
        });
        tracing::info!(
            "  copied from {src:#x}+{len:#x} (memory {kind:#x}): {:.1}x/frame, {:.2} MB/frame, {:.2} ms/frame",
            n as f64 / frames,
            bytes as f64 / 1e6 / frames,
            us as f64 / 1e3 / frames
        );
    }
}
