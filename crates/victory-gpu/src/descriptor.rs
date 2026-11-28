//! Descriptor heaps: zeroed memory, with handles that are plain addresses
//! (CPU and GPU alike, as on the console).
//!
//! Descriptor sizes: views 32 bytes and samplers 16, the size of RDNA 2's
//! image and sampler descriptors, so a title that copies descriptors itself
//! moves whole ones. Render target and depth views are ours alone; 128 bytes
//! leaves room for whatever stride a title assumes.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use crate::object::{self, DESCRIPTOR_HEAP, E_INVALIDARG, HRESULT, Iid, Object, Table, This, guid};
use crate::resource::{Memory, PAGEABLE_IIDS};

#[cfg(windows)]
pub(crate) mod host;

const HEAP_IIDS: [Iid; 4] = [
    guid(
        0x8efb471d,
        0x616c,
        0x4f49,
        [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct HeapDesc {
    kind: u32,
    count: u32,
    flags: u32,
    nodes: u32,
}

/// Newer titles (the title) pass the description followed by
/// memory of their own for the descriptors: an address and its size in
/// bytes, or a null address to have the heap allocate (inferred from the
/// title's call sites; older titles pass the plain description).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct Placement {
    address: u64,
    size: u64,
}

#[repr(C)]
pub(crate) struct Heap {
    object: Object,
    desc: HeapDesc,
    /// The first descriptor: in `memory`, or in the title's own memory.
    pub(crate) start: u64,
    memory: Option<Memory>,
    /// Shader-visible view and sampler heaps get a host heap with the same
    /// descriptor order when a draw first uses them (decision 0007, step
    /// 3); slots written before wait in `pending`.
    #[cfg(windows)]
    host: std::sync::OnceLock<Option<host::HostHeap>>,
    #[cfg(windows)]
    pending: std::sync::Mutex<Vec<u32>>,
    /// Changed pages of a placed shader-visible heap. Some titles copy
    /// descriptor records into it as bytes, without a descriptor API call.
    #[cfg(windows)]
    written: Option<crate::watch::Tracked>,
}

pub(crate) fn heap<'a>(this: This) -> &'a Heap {
    // SAFETY: called from descriptor heap methods only.
    unsafe { &*this.cast::<Heap>() }
}

impl Heap {
    /// Whether this is a sampler heap.
    pub(crate) fn samplers(&self) -> bool {
        self.desc.kind == SAMPLERS
    }

    /// The host heap, made on first use from what the title wrote so far.
    /// Made late because a host view heap the size of the title's
    /// costs over 100 MB of commit, which its boot needs for its page pool.
    #[cfg(windows)]
    pub(crate) fn host(&self) -> Option<&host::HostHeap> {
        if let Some(h) = self.host.get() {
            return h.as_ref();
        }
        if self.desc.flags & SHADER_VISIBLE == 0 || !matches!(self.desc.kind, VIEWS_KIND | SAMPLERS)
        {
            return None;
        }
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(h) = self.host.get() {
            return h.as_ref();
        }
        // A recorder can take the fast path as soon as the OnceLock is
        // initialized. Build and replay queued writes outside `SYNC_ROUNDS`,
        // then publish `self.host` and raise the heap-sync floor in one
        // `SYNC_ROUNDS` critical section so no thread that observes `Some` on
        // the fast path can reach `begin_sync_heaps_for` before the floor is
        // raised. Lock order is `pending` -> `SYNC_ROUNDS` (nothing takes the
        // reverse).
        let built = increment(self.desc.kind).and_then(|step| {
            let step = u64::from(step);
            let h = host::HostHeap::new(self.samplers(), self.desc.count)?;
            for i in pending.drain(..) {
                h.mirror(i, self.start + u64::from(i) * step, false);
            }
            Some(h)
        });
        if built.is_some() && sync_aside() {
            let mut rounds = SYNC_ROUNDS.0.lock().unwrap_or_else(|e| e.into_inner());
            let _ = self.host.set(built);
            rounds.note_heap();
        } else {
            let _ = self.host.set(built);
        }
        self.host.get().and_then(|h| h.as_ref())
    }

    /// The handle of descriptor `index` of this heap.
    #[cfg(windows)]
    pub(crate) fn handle(&self, index: u32) -> Option<u64> {
        let step = u64::from(increment(self.desc.kind)?);
        (index < self.desc.count).then(|| self.start + u64::from(index) * step)
    }

    /// The descriptor index of `handle` in this heap, if it lies in it.
    pub(crate) fn index(&self, handle: u64) -> Option<u32> {
        let step = u64::from(increment(self.desc.kind)?);
        let at = handle.checked_sub(self.start)?;
        let index = at / step;
        (index < u64::from(self.desc.count)).then_some(index as u32)
    }
}

const VIEWS_KIND: u32 = 0;
const SAMPLERS: u32 = 1;
const SHADER_VISIBLE: u32 = 1;

/// Shader-visible heaps by address range, for mirroring descriptor writes:
/// start, end, heap.
#[cfg(windows)]
///
/// Held shared while a heap found here is used (by every thread that writes
/// descriptors, and by [`sync_heaps`]), alone only where heaps come and go.
static SHADER_HEAPS: std::sync::RwLock<Vec<(u64, u64, usize)>> = std::sync::RwLock::new(Vec::new());

/// Makes the host descriptor for what the title wrote at `handle`, if it
/// is in a shader-visible heap. Keeps `SHADER_HEAPS` locked for the duration
/// so `Heap::drop` cannot free the heap concurrently.
#[cfg(windows)]
fn mirror(handle: u64) {
    let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
    let Some(&(_, _, at)) = heaps
        .iter()
        .find(|&&(start, end, _)| (start..end).contains(&handle))
    else {
        return;
    };
    // SAFETY: registered heaps stay alive until `Heap::drop`, which takes
    // the `SHADER_HEAPS` lock we hold here.
    let heap = unsafe { &*(at as *const Heap) };
    let Some(index) = heap.index(handle) else {
        return;
    };
    // Renderer port #2: its record is looked at again with the title's own.
    crate::port_descriptors::mark(handle);
    let mut pending = heap.pending.lock().unwrap_or_else(|e| e.into_inner());
    match heap.host.get() {
        None => pending.push(index),
        Some(host) => {
            drop(pending);
            if let Some(host) = host {
                host.mirror(index, handle, false);
            }
        }
    }
}

/// Whether [`sync_heaps`] is done (`VICTORY_HEAP_SYNC=0`: not).
#[cfg(windows)]
fn sync_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_HEAP_SYNC").is_none_or(|v| v != "0"))
}

/// Whether a submission's [`sync_heaps`] is done by a thread of its own
/// beside the rest of the preparing (`VICTORY_HEAP_SYNC_ASIDE=0`: in front of
/// it, as before).
///
/// It took 13 ms a frame of the submitting thread (session play-1125-0203:
/// 230 changed heap pages and 470,000 slots in 30 submissions). What it makes
/// are the host heap's descriptors, which the card reads: it has to be done
/// before the lists go to the queue, and nothing of the preparing looks at
/// them (what a draw reads through a table it made itself, when recorded).
#[cfg(windows)]
fn sync_aside() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| sync_on() && std::env::var_os("VICTORY_HEAP_SYNC_ASIDE").is_none_or(|v| v != "0"))
}

/// How long that thread waits for a submission before it makes a round of its
/// own (`VICTORY_HEAP_SYNC_EARLY=<ms>`, 4 if not given; `0`: none). The title
/// copies its records while it records, and the first submission of a frame
/// found all of them: 8 ms waited for (run cpu-2). A round made meanwhile
/// leaves it those written since.
#[cfg(windows)]
fn sync_early() -> Option<std::time::Duration> {
    static EVERY: std::sync::OnceLock<Option<std::time::Duration>> = std::sync::OnceLock::new();
    *EVERY.get_or_init(|| {
        let ms = std::env::var("VICTORY_HEAP_SYNC_EARLY").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(4u64);
        (ms > 0).then(|| std::time::Duration::from_millis(ms))
    })
}

/// State of the asynchronous heap-sync worker (`victory-heap-sync`).
#[cfg(windows)]
#[derive(Default)]
struct SyncRounds {
    /// Highest round number requested by a submission.
    asked: u64,
    /// Rounds started by the worker (incremented before `sync_heaps()` runs,
    /// for both asked and early rounds).
    started: u64,
    /// Rounds completed by the worker (incremented after `sync_heaps()`
    /// finishes, for both asked and early rounds).
    done: u64,
    /// Minimum round required after a shader-visible host heap is lazily
    /// initialized (e.g. during `VICTORY_THIN=1` translation after enqueue):
    /// earlier rounds skipped that heap while its `OnceLock` was empty.
    floor: u64,
}

#[cfg(windows)]
impl SyncRounds {
    /// Minimum round number that is guaranteed to start *after* any title
    /// descriptor write already visible to this thread.
    fn stamp(&self) -> u64 {
        self.started + 1
    }

    /// Raises `floor` after a new host heap has been published into
    /// `Heap::host`, so stamped submissions wait for a round that observes it.
    fn note_heap(&mut self) {
        self.floor = self.floor.max(self.started + 1);
    }

    /// Determines the round a caller must wait for (if any) and updates
    /// `asked`. Returns `(wait_for_round, should_notify)`.
    fn request(&mut self, stamp: Option<u64>) -> (Option<u64>, bool) {
        let mine = match stamp {
            Some(s) if s > 0 => {
                let need = s.max(self.floor);
                if self.done >= need {
                    return (None, false);
                }
                need
            }
            _ => self.started + 1,
        };
        let notify = self.asked < mine;
        if notify {
            self.asked = mine;
        }
        (Some(mine), notify)
    }

    /// Whether a caller is waiting for a round that has not finished yet.
    fn has_asked(&self) -> bool {
        self.asked > self.done
    }

    /// Marks the start of a round (before `sync_heaps()` reads write-watch).
    fn start_round(&mut self) {
        self.started += 1;
    }

    /// Marks the completion of a round (after `sync_heaps()` finishes).
    fn finish_round(&mut self) {
        self.done += 1;
    }
}

/// The thread's work: rounds asked for, started, done, and new-heap floor.
#[cfg(windows)]
static SYNC_ROUNDS: (std::sync::Mutex<SyncRounds>, std::sync::Condvar) = (
    std::sync::Mutex::new(SyncRounds { asked: 0, started: 0, done: 0, floor: 0 }),
    std::sync::Condvar::new(),
);

/// Returns the earliest heap-sync round guaranteed to begin *after* title
/// descriptor writes made up to this moment (used both to stamp submissions at
/// enqueue time in `queue::behind::submit` and to raise a queue's post-wait
/// floor when a wait is met). Returns `0` when asynchronous heap sync is off.
#[cfg(windows)]
pub(crate) fn heap_sync_stamp() -> u64 {
    if !sync_aside() {
        return 0;
    }
    SYNC_ROUNDS.0.lock().unwrap_or_else(|e| e.into_inner()).stamp()
}

/// [`sync_heaps`], begun; done when it is dropped (with [`sync_aside`]), or
/// when this returns.
#[cfg(windows)]
pub(crate) struct SyncedHeaps(Option<u64>);

/// Begins (or reuses an already-completed/in-flight) heap-sync round that
/// covers all title descriptor writes up to `stamp` (or up to now if `None`).
#[cfg(windows)]
pub(crate) fn begin_sync_heaps_for(stamp: Option<u64>) -> SyncedHeaps {
    if !sync_aside() {
        sync_heaps();
        return SyncedHeaps(None);
    }
    static STARTED: std::sync::Once = std::sync::Once::new();
    static THREAD: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    STARTED.call_once(|| {
        let made = std::thread::Builder::new().name("victory-heap-sync".into()).spawn(|| {
            crate::cores::keep_on_fast_cores("victory-heap-sync");
            // (The submitting thread waits for its rounds.)
            let mut priority = crate::cores::Priority::new();
            let rounds = || SYNC_ROUNDS.0.lock().unwrap_or_else(|e| e.into_inner());
            /// A round is over, however it ends (asked or early).
            struct Over;
            impl Drop for Over {
                fn drop(&mut self) {
                    SYNC_ROUNDS.0.lock().unwrap_or_else(|e| e.into_inner()).finish_round();
                    SYNC_ROUNDS.1.notify_all();
                }
            }
            // Between the rounds asked for, while submissions come.
            let mut last = std::time::Instant::now();
            loop {
                let mut r = rounds();
                let mut asked = r.has_asked();
                while !asked {
                    let Some(every) = sync_early().filter(|_| last.elapsed().as_millis() < 500) else {
                        r = SYNC_ROUNDS.1.wait(r).unwrap_or_else(|e| e.into_inner());
                        asked = r.has_asked();
                        continue;
                    };
                    let (back, over) = SYNC_ROUNDS.1.wait_timeout(r, every).unwrap_or_else(|e| e.into_inner());
                    r = back;
                    asked = r.has_asked();
                    if over.timed_out() {
                        break;
                    }
                }
                // Count the round as started *before* releasing the lock and
                // calling `sync_heaps()`, so any submission stamped after this
                // point knows it must wait for the next round.
                r.start_round();
                drop(r);
                priority.keep();
                if asked {
                    last = std::time::Instant::now();
                }
                let _over = Over;
                let _round =
                    crate::timeline::span(if asked { "the heaps' records, for a submission" } else { "the heaps' records, early" });
                let _ = std::panic::catch_unwind(sync_heaps);
            }
        });
        THREAD.store(made.is_ok(), std::sync::atomic::Ordering::Release);
    });
    if !THREAD.load(std::sync::atomic::Ordering::Acquire) {
        sync_heaps();
        return SyncedHeaps(None);
    }
    let mut rounds = SYNC_ROUNDS.0.lock().unwrap_or_else(|e| e.into_inner());
    let (mine, notify) = rounds.request(stamp);
    drop(rounds);
    if notify {
        SYNC_ROUNDS.1.notify_all();
    }
    SyncedHeaps(mine)
}

/// Renderer port #7 (`crate::port_journal`, from renderer patch #4): [`begin_sync_heaps_for`] for a
/// submission that needs round `mark` (one that began after what it waits
/// for was written, via [`crate::port_journal::prep_wrote`]): asked for if it has not
/// been. 0: as [`begin_sync_heaps_for`] without a stamp.
#[cfg(windows)]
pub(crate) fn begin_sync_heaps_after(mark: u64) -> SyncedHeaps {
    if mark == 0 || !sync_aside() {
        return begin_sync_heaps_for(None);
    }
    // (The thread is started by the first `begin_sync_heaps_for`.)
    let mut rounds = SYNC_ROUNDS.0.lock().unwrap_or_else(|e| e.into_inner());
    if rounds.done == 0 && rounds.asked == 0 {
        drop(rounds);
        return begin_sync_heaps_for(None);
    }
    let (mine, notify) = rounds.request(Some(mark));
    drop(rounds);
    if notify {
        SYNC_ROUNDS.1.notify_all();
        crate::port_journal::STATS[crate::port_journal::LATE_ROUNDS].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    } else {
        crate::port_journal::STATS[crate::port_journal::EARLY_ROUNDS].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    SyncedHeaps(mine)
}

/// Renderer port #7: the number of a round of the heaps' records that
/// begins after this call, asked for if none is waiting to begin (at the
/// title's submit: the round runs while the submission waits in the line).
/// 0 if the records are not synced by a thread of their own (or it has not
/// started yet).
#[cfg(windows)]
pub(crate) fn round_after_now() -> u64 {
    if !sync_aside() {
        return 0;
    }
    let mut rounds = SYNC_ROUNDS.0.lock().unwrap_or_else(|e| e.into_inner());
    if rounds.done == 0 && rounds.asked == 0 {
        return 0;
    }
    let round = rounds.stamp();
    if rounds.asked < round {
        rounds.asked = round;
        drop(rounds);
        SYNC_ROUNDS.1.notify_all();
        crate::port_journal::STATS[crate::port_journal::ROUNDS_ASKED].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    round
}

/// Renderer port #7: the number of the first round that begins after this
/// call, not asked for ([`begin_sync_heaps_after`] asks for it if nobody did).
/// Also raises `SyncRounds::floor` so stamped submissions queued before a
/// preparation-time write to a placed descriptor heap still wait for a round
/// that observes it.
#[cfg(windows)]
pub(crate) fn round_mark() -> u64 {
    if !sync_aside() {
        return 0;
    }
    let mut rounds = SYNC_ROUNDS.0.lock().unwrap_or_else(|e| e.into_inner());
    if rounds.done == 0 && rounds.asked == 0 {
        return 0;
    }
    rounds.note_heap();
    rounds.floor
}

/// Whether `[at, at + len)` touches the title memory of a placed
/// shader-visible heap of descriptors.
#[cfg(windows)]
pub(crate) fn in_shader_heap(at: usize, len: usize) -> bool {
    let (from, to) = (at as u64, at as u64 + len as u64);
    SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner()).iter().any(|&(start, end, _)| from < end && start < to)
}

/// Whether any `(from, to)` title-address range in `ranges` overlaps a placed
/// shader-visible heap of descriptors, holding `SHADER_HEAPS` read lock once.
#[cfg(windows)]
pub(crate) fn any_in_shader_heap(ranges: impl IntoIterator<Item = (usize, usize)>) -> bool {
    let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
    if heaps.is_empty() {
        return false;
    }
    ranges.into_iter().any(|(from, to)| {
        if to <= from {
            return false;
        }
        let (from, to) = (from as u64, to as u64);
        heaps.iter().any(|&(start, end, _)| from < end && start < to)
    })
}

#[cfg(windows)]
impl Drop for SyncedHeaps {
    fn drop(&mut self) {
        let Some(mine) = self.0 else {
            return;
        };
        let began = std::time::Instant::now();
        let mut rounds = SYNC_ROUNDS.0.lock().unwrap_or_else(|e| e.into_inner());
        while rounds.done < mine {
            rounds = SYNC_ROUNDS.1.wait(rounds).unwrap_or_else(|e| e.into_inner());
        }
        drop(rounds);
        SYNCED[6].fetch_add(began.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Mirrors records copied directly into placed shader-visible heaps. Calls
/// to `Create*View` are already mirrored by [`mirror`]; raw CPU copies leave
/// only write-watch pages to reveal which slots changed.
///
/// Upstream VIctory 86a5ed6 does the same and counts 1,300 to 14,400 such
/// slots a frame in the world of the title this fork works on, against a
/// few hundred views made by calls; without it those host slots stayed
/// null, or went on showing what they showed before.
/// `VICTORY_HEAP_SYNC=0`: not done, as before. [`SYNCED`] counts.
#[cfg(windows)]
pub(crate) fn sync_heaps() {
    use std::sync::atomic::Ordering::Relaxed;
    if !sync_on() {
        return;
    }
    let started = std::time::Instant::now();
    let (mut pages, mut looked, mut mirrored, mut tabled, mut later) = (0u64, 0u64, 0u64, 0u64, 0u64);
    let (mut views, mut bytes) = (0u64, 0u64);
    let mut spent = [0u64; 3];
    let since = |at: std::time::Instant| at.elapsed().as_nanos() as u64;
    let mut kinds = [0u64; 5];
    let mut job_ns = [0u64; 3];
    let mut short = 0u64;
    let mut wait: Vec<u32> = Vec::new();
    let frame = crate::frame::current();
    let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
    for &(start, end, at) in heaps.iter() {
        // SAFETY: registered heaps are live (removed when dropped).
        let heap = unsafe { &*(at as *const Heap) };
        let (Some(Some(host)), Some(written), Some(step)) =
            (heap.host.get(), &heap.written, increment(heap.desc.kind))
        else {
            continue;
        };
        let step = u64::from(step);
        // Capped host heaps have no counterparts for the title slots reserved
        // for per-draw tables; raw writes there must not be mirrored.
        let mirrored_count = heap.desc.count.min(host.count());
        // The records of some frames ago that no draw read through a table
        // since: a shader may pick them by index.
        let before = host.copied_before(frame);
        let of_tables = std::sync::atomic::AtomicU64::new(0);
        let (pages_a_job, records_a_job) = sync_jobs();
        let phase = crate::timeline::span("the heaps' records: those that waited");
        crate::land::each_view(true, before.len().div_ceil(records_a_job), &|n| {
            for &index in before.chunks(records_a_job).nth(n).unwrap_or_default() {
                if host.copied_tabled(index, frame) {
                    of_tables.fetch_add(1, Relaxed);
                } else {
                    host.mirror(index, start + u64::from(index) * step, false);
                }
            }
        });
        phase.end();
        let of_tables = of_tables.into_inner();
        tabled += of_tables;
        mirrored += before.len() as u64 - of_tables;
        kinds[4] += before.len() as u64 - of_tables;
        // Those of the changed pages, made together at the end.
        let mut make: Vec<u32> = Vec::new();
        // Ranges of the heap to look at, and whether a range is one the
        // fallback of renderer port #2 looks at whole.
        let mut parts: Vec<(usize, usize, bool)> = Vec::new();
        let port = crate::port_descriptors::adopted(start);
        if port {
            // Renderer port #2: the slots the title's patched stores and our
            // calls marked, in place of the changed pages; and a few of the
            // pages the watch says were written, looked at whole.
            let phase = crate::timeline::span("the heaps' records: the marks taken (renderer port 2)");
            let marked = crate::port_descriptors::take(start).unwrap_or_default();
            phase.end();
            if crate::port_descriptors::ask_due(frame) {
                let began = std::time::Instant::now();
                let phase = crate::timeline::span("the heaps' records: asking (renderer port 2's fallback)");
                let taken = written.take(start as usize, end as usize);
                phase.end();
                spent[0] += since(began);
                crate::port_descriptors::asked(&taken);
            }
            parts = port_scan_parts(
                &marked,
                crate::port_descriptors::to_check(frame),
                start as usize,
                end as usize,
                step as usize,
            );
        } else {
            let began = std::time::Instant::now();
            let phase = crate::timeline::span("the heaps' records: asking");
            let taken = written.take(start as usize, end as usize);
            phase.end();
            spent[0] += since(began);
            for (range_from, range_end) in taken {
                let mut page = range_from & !(crate::watch::PAGE - 1);
                while page < range_end {
                    parts.push((page.max(range_from), (page + crate::watch::PAGE).min(range_end), false));
                    page += crate::watch::PAGE;
                }
            }
        }
        // The marks' runs are short: more of them a job.
        let pages_a_job = if port { pages_a_job.max(parts.len() / 48) } else { pages_a_job };
        // A few pages a thread at a time: read, compared with what was last
        // seen there and sorted out, beside one another (14 ms a frame of one
        // thread at 160 pages, run faults-6).
        let began = std::time::Instant::now();
        let copied = host.copied_begin(step as usize);
        let samplers = heap.samplers();
        let in_place = sync_in_place() && step == 32 && crate::safecopy::on();
        let found: Vec<std::sync::Mutex<SyncPart>> = parts.chunks(pages_a_job).map(|_| Default::default()).collect();
        let phase = crate::timeline::span("the heaps' records: pages read and compared");
        crate::land::each_view(true, found.len(), &|n| {
            let mut out = SyncPart::default();
            let mut page_bytes: Vec<u8> = Vec::new();
            let mut changed: Vec<u32> = Vec::new();
            let mut bits: Vec<u64> = Vec::new();
            for &(from, to, check) in parts.chunks(pages_a_job).nth(n).unwrap_or_default() {
                let checking = check.then(std::time::Instant::now);
                let slots = changed_slots(from, to, start as usize, step, mirrored_count);
                // A record that straddles the range's ends is left
                // to the range that holds its first byte.
                let first = ((from - start as usize) as u64).div_ceil(step).max(slots.start);
                let lead = (start + first * step) as usize;
                let records: &[u8] = if in_place {
                    // Read where they are, each record once, and compared
                    // as it is read (`host::Copied::changes_in_place`).
                    let at = std::time::Instant::now();
                    let count = (to.saturating_sub(lead) as u64 / step).min(slots.end.saturating_sub(first));
                    changed.clear();
                    // SAFETY: as below; title memory is read by a copy that
                    // fails where a page is not there.
                    let whole = unsafe { copied.changes_in_place(first as u32, lead as *const u8, count as usize, &mut bits, &mut changed) };
                    out.ns[1] += since(at);
                    // (A run of marked slots is no page; nor does it ask the guards.)
                    if !port || check {
                        out.pages += 1;
                        out.views += u64::from(crate::watch::guarded(from, to));
                    }
                    out.bytes += (to - from) as u64;
                    out.looked += count;
                    out.short += u64::from(!whole);
                    // What was read of the changed records is in the records
                    // last seen now.
                    // SAFETY: this part's records alone, which nothing else
                    // writes meanwhile.
                    unsafe { copied.seen(first as u32, count as usize) }
                } else {
                    // The page's records are read through the kernel, into a
                    // copy: a page the title unmaps meanwhile fails the read
                    // where 2,048 reads in place would fault.
                    let at = std::time::Instant::now();
                    page_bytes.resize(to - from, 0);
                    if !crate::host::read_into(from as *const u8, &mut page_bytes) {
                        continue;
                    }
                    out.ns[0] += since(at);
                    out.pages += 1;
                    out.views += u64::from(crate::watch::guarded(from, to));
                    out.bytes += (to - from) as u64;
                    let Some(records) = page_bytes.get(lead.wrapping_sub(from)..) else {
                        continue;
                    };
                    let count = (records.len() as u64 / step).min(slots.end.saturating_sub(first));
                    out.looked += count;
                    // Only the records that are not the ones last seen
                    // there (`host::Copied::changes`).
                    changed.clear();
                    let at = std::time::Instant::now();
                    // SAFETY: the parts are ranges of the heap apart from one
                    // another, and only records that lie whole in one are its.
                    unsafe { copied.changes(first as u32, &records[..(count * step) as usize], &mut changed) };
                    out.ns[1] += since(at);
                    records
                };
                let at = std::time::Instant::now();
                for &index in &changed {
                    let at = (u64::from(index) - first) as usize * step as usize;
                    let word = |n: usize| {
                        records
                            .get(at + 4 * n..at + 4 * n + 4)
                            .and_then(|b| b.try_into().ok())
                            .map_or(0, u32::from_le_bytes)
                    };
                    if check {
                        // Renderer port #2's fallback: what no mark told of.
                        crate::port_descriptors::missed(index, start + u64::from(index) * step, !samplers && word(0) == VIEW_TAG, host.copied_tabled(index, frame));
                    }
                    if !samplers && word(0) == VIEW_TAG {
                        // A draw's own table: the draw makes what it
                        // reads of it (`draw.rs` `tables`).
                        if host.copied_tabled(index, frame) {
                            out.tabled += 1;
                            continue;
                        }
                        // Buffer views and placed textures get no host
                        // slot from this anyway, they wait for a draw
                        // or a pass: noted some frames on, unless a
                        // draw's table turns out to hold them.
                        if matches!(word(2) >> KIND_SHIFT, CONSTANT | PLACED) && host.copied_waits() {
                            out.wait.push(index);
                            continue;
                        }
                    }
                    // No view, in a slot that shows none: nothing to make or
                    // unmake (32,000 of 41,000 a frame on the water, run
                    // b2-water).
                    if word(0) != VIEW_TAG && host.shows_none(index) {
                        out.nothing += 1;
                        continue;
                    }
                    out.make.push(index);
                    out.kinds[if samplers {
                        3
                    } else if word(0) != VIEW_TAG {
                        0
                    } else if word(2) >> KIND_SHIFT == 0 {
                        1
                    } else {
                        2
                    }] += 1;
                }
                out.ns[2] += since(at);
                if let Some(began) = checking {
                    crate::port_descriptors::CHECKED[4].fetch_add(since(began), Relaxed);
                }
            }
            *found[n].lock().unwrap_or_else(|e| e.into_inner()) = out;
        });
        phase.end();
        drop(copied);
        for part in found {
            let mut part = part.into_inner().unwrap_or_else(|e| e.into_inner());
            pages += part.pages;
            views += part.views;
            bytes += part.bytes;
            looked += part.looked;
            short += part.short;
            tabled += part.tabled;
            kinds[0] += part.nothing;
            later += part.wait.len() as u64;
            mirrored += part.make.len() as u64;
            wait.append(&mut part.wait);
            make.append(&mut part.make);
            for (kind, n) in kinds.iter_mut().zip(part.kinds) {
                *kind += n;
            }
            for (ns, n) in job_ns.iter_mut().zip(part.ns) {
                *ns += n;
            }
        }
        spent[1] += since(began);
        let began = std::time::Instant::now();
        let phase = crate::timeline::span("the heaps' records: made");
        crate::land::each_view(true, make.len().div_ceil(records_a_job), &|n| {
            for &index in make.chunks(records_a_job).nth(n).unwrap_or_default() {
                host.mirror(index, start + u64::from(index) * step, false);
            }
        });
        phase.end();
        spent[2] += since(began);
        if !wait.is_empty() {
            host.copied_wait(frame, std::mem::take(&mut wait));
        }
    }
    drop(heaps);
    let micros = started.elapsed().as_micros() as u64;
    for (count, n) in SYNCED.iter().zip([pages, looked, mirrored, micros, tabled, later]) {
        count.fetch_add(n, Relaxed);
    }
    let [taking, reading, making] = spent;
    for (count, n) in SYNC_SPLIT.iter().zip([views, bytes, taking, reading, making]) {
        count.fetch_add(n, Relaxed);
    }
    for (count, n) in SYNCED_KINDS.iter().zip(kinds) {
        count.fetch_add(n, Relaxed);
    }
    for (count, n) in SYNC_JOBS_NS.iter().zip(job_ns) {
        count.fetch_add(n, Relaxed);
    }
    SYNC_SHORT.fetch_add(short, Relaxed);
}

/// Pages [`sync_heaps`] read in place of which a part could not be read
/// ([`sync_in_place`]), since the last frame line.
#[cfg(windows)]
static SYNC_SHORT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Whether [`sync_heaps`] reads a changed page's records where they are, each
/// once, and compares each with the one last seen as it reads it, 32 bytes at
/// a time (`VICTORY_SYNC_IN_PLACE=0`: into a copy of the page first, then
/// compared, as before).
///
/// The copy and the compare were each a pass over all of the page: 7.4 and
/// 5.3 ms a frame of the six threads for 35 MB (run heapsync-ab1, in the
/// street). A page the title unmaps meanwhile fails where the copy failed;
/// the records before the place are taken as they were read.
///
/// On by default since run heapsync-ab2 (A/B in the street: the pass 12.7 ms
/// a frame against 15.7-16.2, the copy and compare 7.4 ms of its threads
/// against 13.3, the submitting thread's wait for it 2.0 ms against 2.2-2.6;
/// the picture the same).
#[cfg(windows)]
fn sync_in_place() -> bool {
    #[cfg(feature = "diagnostics")]
    if let Some(on) = crate::ab::pick("VICTORY_SYNC_IN_PLACE") {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SYNC_IN_PLACE").is_none_or(|v| v != "0"))
}

/// Of [`sync_heaps`]'s page jobs since the last frame line, on all threads:
/// nanoseconds reading pages, comparing them with what was last seen, and
/// sorting out the records that changed.
#[cfg(windows)]
static SYNC_JOBS_NS: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// Pages and records [`sync_heaps`] gives a thread at a time
/// (`VICTORY_SYNC_FINE=1`: [`SYNC_PAGES_FINE`] and [`SYNC_AT_A_TIME_FINE`];
/// otherwise [`SYNC_PAGES`] and [`SYNC_AT_A_TIME`], as before).
///
/// A round for a submission finds a dozen pages or so: by eight a job that
/// was two jobs, on two of the six threads, while the submission waited
/// for the round (2 to 3.4 ms at a time, run city4).
#[cfg(windows)]
fn sync_jobs() -> (usize, usize) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let default_fine = || *ON.get_or_init(|| std::env::var_os("VICTORY_SYNC_FINE").is_some_and(|v| v != "0"));
    #[cfg(feature = "diagnostics")]
    let fine = crate::ab::pick("VICTORY_SYNC_FINE").unwrap_or_else(default_fine);
    #[cfg(not(feature = "diagnostics"))]
    let fine = default_fine();
    if fine { (SYNC_PAGES_FINE, SYNC_AT_A_TIME_FINE) } else { (SYNC_PAGES, SYNC_AT_A_TIME) }
}

#[cfg(windows)]
const SYNC_PAGES_FINE: usize = 2;

#[cfg(windows)]
const SYNC_AT_A_TIME_FINE: usize = 64;

/// Of those mirrored, since the last frame line: no view, a view of an object,
/// another, a sampler; waited and no draw came.
#[cfg(windows)]
static SYNCED_KINDS: [std::sync::atomic::AtomicU64; 5] = [const { std::sync::atomic::AtomicU64::new(0) }; 5];

/// What a thread found on its pages of a heap, for [`sync_heaps`].
#[cfg(windows)]
#[derive(Default)]
struct SyncPart {
    pages: u64,
    views: u64,
    bytes: u64,
    looked: u64,
    tabled: u64,
    nothing: u64,
    make: Vec<u32>,
    wait: Vec<u32>,
    kinds: [u64; 5],
    /// Nanoseconds reading, comparing, sorting out ([`SYNC_JOBS_NS`]).
    ns: [u64; 3],
    /// Pages read in place not all there ([`SYNC_SHORT`]).
    short: u64,
}

/// Pages [`sync_heaps`] gives a thread at a time.
#[cfg(windows)]
const SYNC_PAGES: usize = 8;

/// Records [`sync_heaps`] gives a thread at a time.
#[cfg(windows)]
const SYNC_AT_A_TIME: usize = 256;

/// Since the last frame line, of [`sync_heaps`]: changed pages of placed
/// shader-visible heaps looked at, slots on them, slots whose record was
/// not the one last seen and was mirrored, the microseconds it took; then
/// new records left to the draws that read them through a table, and new
/// records that waited for one ([`host::HostHeap::copied_wait`]); and the
/// microseconds submissions waited for it ([`sync_aside`]).
#[cfg(windows)]
pub(crate) static SYNCED: [std::sync::atomic::AtomicU64; 7] = [const { std::sync::atomic::AtomicU64::new(0) }; 7];

/// The same, of [`SYNCED`]'s pages: those in view memory that was guarded;
/// bytes read; nanoseconds asking what was written, reading it and comparing
/// it with what was last seen, making views.
#[cfg(windows)]
static SYNC_SPLIT: [std::sync::atomic::AtomicU64; 5] = [const { std::sync::atomic::AtomicU64::new(0) }; 5];

/// With the frame line: what [`sync_heaps`] did per frame over `frames`.
/// Nothing while no heap page changed.
#[cfg(windows)]
pub(crate) fn sync_line(frames: f64) {
    crate::port_descriptors::line(frames);
    crate::draw::in_place_indirect_line(frames);
    let [twins, moved, lost] = [0, 1, 2].map(|i| host::TWINS[i].swap(0, std::sync::atomic::Ordering::Relaxed));
    if twins > 0 {
        tracing::info!(
            "raw twins of buffer views made per frame (VICTORY_RAW_VIEWS=0: none): {:.0}, of them {:.1} that do not begin on 16 bytes ({:.1} null)",
            twins as f64 / frames,
            moved as f64 / frames,
            lost as f64 / frames
        );
    }
    let [pages, looked, mirrored, micros, tabled, later, waited] = [0, 1, 2, 3, 4, 5, 6].map(|i| SYNCED[i].swap(0, std::sync::atomic::Ordering::Relaxed));
    let split = [0, 1, 2, 3, 4].map(|i| SYNC_SPLIT[i].swap(0, std::sync::atomic::Ordering::Relaxed) as f64);
    let kinds = [0, 1, 2, 3, 4].map(|i| SYNCED_KINDS[i].swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / frames);
    let jobs = [0, 1, 2].map(|i| SYNC_JOBS_NS[i].swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / 1e6 / frames);
    let quick = [0, 1].map(|i| host::SPANS_QUICK[i].swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / frames);
    let short = SYNC_SHORT.swap(0, std::sync::atomic::Ordering::Relaxed);
    if pages == 0 {
        return;
    }
    tracing::info!(
        "the heaps' records' jobs per frame (VICTORY_SYNC_FINE={}, VICTORY_SYNC_IN_PLACE={}): on all threads {:.2} ms reading pages, \
         {:.2} comparing (in place: reading too), {:.2} sorting out; writes of slots {:.0} leaving the spans alone (VICTORY_SPANS_QUICK), \
         {:.0} under their lock; {} pages read in place not all there",
        u8::from(sync_jobs().0 == SYNC_PAGES_FINE),
        u8::from(sync_in_place()),
        jobs[0],
        jobs[1],
        jobs[2],
        quick[0],
        quick[1],
        short
    );
    tracing::info!(
        "descriptor records the title copied into its heap as bytes (VICTORY_HEAP_SYNC=0: not looked for): per frame \
         {:.1} changed heap pages, {:.0} slots looked at, {:.1} whose record was new and was mirrored, {:.1} new ones of \
         draws' tables left to the draws, {:.1} that waited for a draw first, {:.2} ms ({:.2} ms waited for); of the mirrored \
         {:.0} no views (with those of slots that show none), {:.0} of objects, {:.0} others, {:.0} samplers, {:.0} that waited and no draw came; \
         {:.1} of the pages guarded views, {:.1} MB read; {:.2} ms asking, {:.2} reading and comparing, {:.2} making",
        pages as f64 / frames,
        looked as f64 / frames,
        mirrored as f64 / frames,
        tabled as f64 / frames,
        later as f64 / frames,
        micros as f64 / 1000.0 / frames,
        waited as f64 / 1000.0 / frames,
        kinds[0],
        kinds[1],
        kinds[2],
        kinds[3],
        kinds[4],
        split[0] / frames,
        split[1] / (1 << 20) as f64 / frames,
        split[2] / 1e6 / frames,
        split[3] / 1e6 / frames,
        split[4] / 1e6 / frames
    );
}

/// Title descriptor slots in a changed byte range that have host counterparts.
/// `range_end` is exclusive.
#[cfg(windows)]
fn changed_slots(
    range_from: usize,
    range_end: usize,
    heap_start: usize,
    step: u64,
    mirrored_count: u32,
) -> std::ops::Range<u64> {
    let from = (range_from - heap_start) as u64;
    let to = (range_end - heap_start) as u64;
    let first = from / step;
    let last = to.div_ceil(step).min(u64::from(mirrored_count));
    first.min(last)..last
}

/// Marks own their slots, even if a wider fallback read would fail. The
/// fallback takes only the complete slots between the sorted marked runs.
/// The write watch supplies each fallback page at most once.
#[cfg(windows)]
fn port_scan_parts(
    marked: &[(u32, u32)],
    pages: Vec<usize>,
    heap_start: usize,
    heap_end: usize,
    step: usize,
) -> Vec<(usize, usize, bool)> {
    let mut parts: Vec<_> = marked
        .iter()
        .map(|&(first, count)| {
            let from = heap_start + first as usize * step;
            (from, from + count as usize * step, false)
        })
        .collect();
    let marked_parts = parts.len();
    for page in pages {
        let from = page.max(heap_start);
        let to = (page + crate::watch::PAGE).min(heap_end);
        if from >= to {
            continue;
        }
        let mut from = heap_start + (from - heap_start).div_ceil(step) * step;
        let to = heap_start + (to - heap_start) / step * step;
        let first_mark = parts[..marked_parts].partition_point(|&(_, end, _)| end <= from);
        for n in first_mark..marked_parts {
            let (mark_from, mark_to, _) = parts[n];
            if from >= to || mark_from >= to {
                break;
            }
            if from < mark_from {
                parts.push((from, mark_from, true));
            }
            from = mark_to;
        }
        if from < to {
            parts.push((from, to, true));
        }
    }
    parts
}

/// The title's view heap that shaders of a list without one get their views
/// from (`draw.rs`, `bind`): the title's structure builder binds root views
/// only, and the host shows those, and memory by address, through heap slots.
/// The biggest there is. `VICTORY_LENT_HEAP=0`: none, such a dispatch is
/// dropped, as before.
#[cfg(windows)]
pub(crate) fn lent_heap() -> Option<usize> {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("VICTORY_LENT_HEAP").is_none_or(|v| v != "0")) {
        return None;
    }
    let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
    heaps
        .iter()
        // SAFETY: registered heaps are live (removed when dropped).
        .map(|&(_, _, h)| (h, unsafe { &*(h as *const Heap) }))
        .filter(|(_, heap)| !heap.samplers() && matches!(heap.host.get(), Some(Some(_))))
        .max_by_key(|(_, heap)| heap.desc.count)
        .map(|(h, _)| h)
}

/// `f` of the title heap at `heap`, if that heap still exists; it stays
/// until `f` is done.
#[cfg(windows)]
pub(crate) fn with_heap<T>(heap: usize, f: impl FnOnce(&Heap) -> T) -> Option<T> {
    let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
    // SAFETY: registered heaps are live (removed when dropped).
    heaps.iter().any(|&(_, _, h)| h == heap).then(|| f(unsafe { &*(heap as *const Heap) }))
}

/// Gives back dynamic slots of the host heap behind the title heap at
/// `heap`, if its host heap still has the identity that took the slots.
#[cfg(windows)]
pub(crate) fn release_dynamic(heap: usize, id: u64, first: u32, n: u32) {
    let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
    if heaps.iter().any(|&(_, _, h)| h == heap)
        // SAFETY: registered heaps are live (removed when dropped).
        && let Some(Some(host)) = unsafe { &*(heap as *const Heap) }.host.get()
    {
        host.release(id, first, n);
    }
}

/// Runs `f` with the host heap of the title's largest view heap, if there is
/// one (nothing is waited for).
#[cfg(windows)]
pub(crate) fn with_view_heap<T>(f: impl FnOnce(&host::HostHeap) -> T) -> Option<T> {
    let heaps = SHADER_HEAPS.try_read().ok()?;
    let host = heaps
        .iter()
        // SAFETY: registered heaps are live (removed when dropped).
        .map(|&(_, _, at)| unsafe { &*(at as *const Heap) })
        .filter(|heap| !heap.samplers())
        .filter_map(|heap| heap.host.get()?.as_ref())
        .max_by_key(|host| host.count())?;
    Some(f(host))
}

/// Makes dynamic slot `index` of the host heap behind the title heap at
/// `heap`, if that heap still exists, the constant buffer view of `size`
/// bytes of `buffer` at `gpu`.
#[cfg(windows)]
pub(crate) fn constant_slot(
    heap: usize,
    index: u32,
    buffer: &windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    gpu: u64,
    size: u32,
) {
    let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
    if heaps.iter().any(|&(_, _, h)| h == heap)
        // SAFETY: registered heaps are live (removed when dropped).
        && let Some(Some(host)) = unsafe { &*(heap as *const Heap) }.host.get()
    {
        host.constant_at(index, buffer, gpu, size);
    }
}

#[cfg(windows)]
impl Drop for Heap {
    fn drop(&mut self) {
        let at = std::ptr::from_ref::<Heap>(self) as usize;
        let mut heaps = SHADER_HEAPS.write().unwrap_or_else(|e| e.into_inner());
        heaps.retain(|&(_, _, h)| h != at);
        crate::port_descriptors::let_go(self.start);
    }
}

/// Bytes per descriptor, by heap type (views, samplers, render targets,
/// depth).
fn increment(kind: u32) -> Option<u32> {
    [32, 16, 128, 128].get(kind as usize).copied()
}

/// Device method.
pub(crate) extern "system" fn increment_size(_: This, kind: u32) -> u32 {
    increment(kind).unwrap_or(0)
}

/// What our render target views hold: the resource and the view
/// description, turned into a host view when a command uses them.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct RtvRecord {
    tag: u32,
    described: u32,
    pub(crate) resource: This,
    desc: [u32; 6],
}
const RTV_TAG: u32 = u32::from_le_bytes(*b"vRTV");
const _: () = assert!(size_of::<RtvRecord>() <= 112);

impl RtvRecord {
    /// All the view it records is made from: records with equal keys have
    /// equal views (of the same host texture).
    #[cfg(windows)]
    pub(crate) fn key(&self) -> (usize, [u32; 7]) {
        let mut words = [self.described; 7];
        words[1..].copy_from_slice(&self.desc);
        (self.resource as usize, words)
    }

    /// The view description in the host's type (the same layout).
    #[cfg(windows)]
    pub(crate) fn desc(
        &self,
    ) -> Option<windows::Win32::Graphics::Direct3D12::D3D12_RENDER_TARGET_VIEW_DESC> {
        const TEXTURE_2D: u32 = 4;
        self.host_desc(TEXTURE_2D)
    }

    /// The depth view description in the host's type (the same layout).
    #[cfg(windows)]
    pub(crate) fn depth_desc(
        &self,
    ) -> Option<windows::Win32::Graphics::Direct3D12::D3D12_DEPTH_STENCIL_VIEW_DESC> {
        const TEXTURE_2D: u32 = 3;
        self.host_desc(TEXTURE_2D)
    }

    /// The description words as a host type, the format as the host has it.
    /// Without one, a typeless host copy gets a 2D view (`dimension`) in
    /// the title's format, which the host needs spelled out.
    #[cfg(windows)]
    fn host_desc<T>(&self, dimension: u32) -> Option<T> {
        assert!(size_of::<T>() <= size_of::<[u32; 6]>());
        let mut words = if self.described != 0 {
            self.desc
        } else {
            [
                crate::resource::view_format(self.resource)?,
                dimension,
                0,
                0,
                0,
                0,
            ]
        };
        // A target or depth view: of a target, in its stand-in format.
        words[0] = crate::resource::target_format(crate::resource::pc_format(words[0]));
        // SAFETY: both host view descriptions are 24 bytes in this layout.
        Some(unsafe { (&raw const words).cast::<T>().read_unaligned() })
    }
}

/// Device method: records the view at `handle` (descriptor memory).
pub(crate) extern "system" fn create_rtv(
    _: This,
    resource: This,
    desc: *const [u32; 6],
    handle: u64,
) {
    if handle == 0 {
        return;
    }
    // SAFETY: the title passes a description or null.
    let described = unsafe { desc.as_ref() };
    let record = RtvRecord {
        tag: RTV_TAG,
        described: u32::from(described.is_some()),
        resource,
        desc: described.copied().unwrap_or_default(),
    };
    // SAFETY: the handle is a descriptor slot, at least 112 bytes.
    unsafe { (handle as *mut RtvRecord).write_unaligned(record) };
}

/// Device method: a depth view, recorded as render target views are.
pub(crate) extern "system" fn create_dsv(
    this: This,
    resource: This,
    desc: *const [u32; 6],
    handle: u64,
) {
    create_rtv(this, resource, desc, handle);
    if handle != 0 {
        // SAFETY: just written by `create_rtv`.
        unsafe { (handle as *mut u32).write_unaligned(DSV_TAG) };
    }
}
const DSV_TAG: u32 = u32::from_le_bytes(*b"vDSV");

/// A shader view or sampler as the title described it (descriptions in the
/// host's layout, kept as words), for the host to build its own from.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum View {
    /// Constant buffer: GPU address and size.
    Constant(u64, u32),
    /// Shader resource: resource, description (none: the resource's own).
    Shader(usize, Option<[u32; 10]>),
    /// Unordered access: resource, counter resource, description.
    Unordered(usize, usize, Option<[u32; 10]>),
    Sampler([u32; 13]),
    /// The console's own sampler description: flags, filters per axis,
    /// addressing, LOD and anisotropy settings, border color, then
    /// console-only tuning words.
    ConsoleSampler([u32; 27]),
    /// A view of memory without a resource object: whether it is unordered
    /// access, the resource description, then the view's (the plain one
    /// followed by the memory address and console fields).
    Placed(bool, [u32; 14], [u32; 16]),
}

impl View {
    #[cfg(all(windows, feature = "diagnostics"))]
    fn kind_index(&self) -> usize {
        match self {
            Self::Constant(..) => 0,
            Self::Shader(..) => 1,
            Self::Unordered(..) => 2,
            Self::Sampler(..) => 3,
            Self::ConsoleSampler(..) => 4,
            Self::Placed(..) => 5,
        }
    }

    /// Whether the view reads buffer memory (constant buffers, buffer views
    /// of resources or of memory) or memory the title placed a texture in:
    /// such views get host slots only for draws.
    #[cfg(windows)]
    fn is_buffer(&self) -> bool {
        match self {
            View::Constant(..) | View::Placed(..) => true,
            View::Shader(r, _) | View::Unordered(r, _, _) => {
                crate::resource::buffer_bytes(*r as This).is_some()
            }
            View::Sampler(_) | View::ConsoleSampler(_) => false,
        }
    }
}

/// Every distinct view in use; a descriptor holds a tag, an index into it
/// and the entry's generation, which survive titles copying descriptors as
/// bytes. A view of a resource lives as long as the resource: when the
/// title releases it, its entries are freed for reuse under a new
/// generation, so descriptors (and host slots made from them) naming the
/// old one go stale instead of naming whatever is made at the same address
/// next.
///
/// Views without a resource object have nothing that releases them, and a
/// title copies descriptors as bytes, so nothing says when the last
/// descriptor naming an entry is gone: such entries were kept for the run.
/// The title writes some 14,000 constant buffer views a frame,
/// each at the address its constants have that frame: 2,450,139 entries
/// after four minutes of play, 1,047 MB with the index, and 11,000 more
/// every second (run 65). So the two kinds that carry an address are not
/// entries any more: their records name them whole ([`constant_words`],
/// [`placed_words`]), and the table holds only what is bounded: views of
/// resource objects (freed with the resource), samplers (as many as there
/// are sampler states), and the shapes of placed views (their descriptions
/// without the address: as many as the title has kinds of textures and
/// buffers, however often it moves them).
/// `VICTORY_VIEW_INLINE=0`: every view is an entry, as before.
#[cfg_attr(test, derive(Default))]
struct Views {
    /// Each entry's view and generation.
    list: Vec<(View, u32)>,
    /// Count slot changes so the memory log never has to walk the table.
    /// A free slot counts toward the kind it last held until it is reused.
    #[cfg(all(windows, feature = "diagnostics"))]
    kind_usage: [ViewKindUsage; 6],
    index: std::collections::BTreeMap<View, u32>,
    /// Entries naming each resource.
    by_resource: std::collections::BTreeMap<usize, Vec<u32>>,
    free: Vec<u32>,
    /// The shapes of views of placed memory, and where each is by its
    /// hash ([`shape_key`]): keyed by the shape itself the index held a
    /// second copy of each, 2.4 million of them (some 900 MB with the list)
    /// 18 minutes into the world.
    shapes: Vec<Shape>,
    shape_index: std::collections::BTreeMap<u64, u32>,
    /// Entries of views that no resource releases (constant buffer views
    /// and views of placed memory that did not fit a record), oldest first,
    /// with the generation each was made at: past [`ownerless_cap`] the
    /// oldest is freed, so they cannot grow for the whole run.
    ownerless: std::collections::VecDeque<(u32, u32)>,
}

#[cfg(all(windows, feature = "diagnostics"))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ViewKindUsage {
    pub(crate) occupied: u64,
    pub(crate) free: u64,
}

#[cfg(all(windows, feature = "diagnostics"))]
impl ViewKindUsage {
    /// Bytes in the slots themselves, excluding spare capacity and indexes.
    pub(crate) fn slot_bytes(self) -> u64 {
        (self.occupied + self.free) * size_of::<(View, u32)>() as u64
    }
}

/// Read far more often than written: every slot a draw reads through a
/// table is looked up here, on several recording threads at once in the
/// world. Readers share it; only a view not seen before (a new table entry
/// or shape) and a released resource take it alone.
static VIEWS: std::sync::RwLock<Views> = std::sync::RwLock::new(Views {
    list: Vec::new(),
    #[cfg(all(windows, feature = "diagnostics"))]
    kind_usage: [ViewKindUsage { occupied: 0, free: 0 }; 6],
    index: std::collections::BTreeMap::new(),
    by_resource: std::collections::BTreeMap::new(),
    free: Vec::new(),
    shapes: Vec::new(),
    shape_index: std::collections::BTreeMap::new(),
    ownerless: std::collections::VecDeque::new(),
});

/// How many table entries of views without a resource are kept
/// ([`Views::ownerless`]). Once the shapes of placed views are full (the
/// title makes far more of them than it has kinds of textures: 4,194,304
/// after half an hour of play), every placed view it writes becomes such an
/// entry, some 15,000 a second: kept for the run they took 16.7 GB after 70
/// minutes and the frame rate fell by half. A descriptor names the view it
/// was written with for a frame or a few, so the oldest of a million are
/// long out of use; one naming a freed entry reads as stale (no view).
/// `VICTORY_OWNERLESS_VIEWS=<n>` keeps `n` (0: all, as before).
fn ownerless_cap() -> usize {
    static CAP: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| {
        std::env::var("VICTORY_OWNERLESS_VIEWS")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(1 << 20)
    })
}
pub(crate) const VIEW_TAG: u32 = u32::from_le_bytes(*b"vVIE");

/// A record is the tag and two words, `i` and `generation` (the host keeps
/// them as one 64-bit key per slot). The top two bits of `generation` say
/// what the other 62 bits are: 0, entry `i` of the table at that
/// generation; [`CONSTANT`], a constant buffer view, whole; [`PLACED`], a
/// view of placed memory: its address and the number of its shape.
const KIND_SHIFT: u32 = 30;
/// Table generations stay below the kind bits (and above 0).
const GENERATIONS: u32 = (1 << KIND_SHIFT) - 1;
const CONSTANT: u32 = 1;
const PLACED: u32 = 2;
/// Bits of a constant buffer view's address in its record: any address a
/// program can have. The 15 bits above them are its size.
const CONSTANT_ADDRESS_BITS: u32 = 47;
/// Bits of a placed view's address in its record: placed memory is title
/// memory, which lies below 1 TiB. The 22 bits above them number its shape.
const PLACED_ADDRESS_BITS: u32 = 40;
const SHAPES: usize = 1 << (62 - PLACED_ADDRESS_BITS);

/// A view of placed memory without its address: whether it is unordered
/// access, the resource description, and the view's words but the two that
/// hold the address (10 and 11).
type Shape = (bool, [u32; 14], [u32; 14]);

/// Whether constant buffer views and views of placed memory are named whole
/// in their records ([`Views`]). On by default; `VICTORY_VIEW_INLINE=0`
/// makes every view a table entry, as before.
fn inline() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_VIEW_INLINE").is_none_or(|v| v != "0"))
}

/// Views named whole in their records since the start: constant buffer
/// views, views of placed memory (for the memory line).
static NAMED_WHOLE: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// The two record words of the 62 bits `payload` of kind `kind`.
fn kind_words(kind: u32, payload: u64) -> [u32; 2] {
    [payload as u32, kind << KIND_SHIFT | (payload >> 32) as u32]
}

/// The 62 bits two record words carry beside their kind.
fn payload(i: u32, generation: u32) -> u64 {
    u64::from(generation & GENERATIONS) << 32 | u64::from(i)
}

/// The record words that name the constant buffer view of `size` bytes at
/// `address` whole; `None` if they cannot: an address past 47 bits, or a
/// size of 16 KiB or more that is no multiple of 256 or reaches 4 MiB (the
/// size has 15 bits: bytes below 16 KiB, else units of 256 bytes). Such a
/// view is a table entry.
fn constant_words(address: u64, size: u32) -> Option<[u32; 2]> {
    if address >> CONSTANT_ADDRESS_BITS != 0 {
        return None;
    }
    let code = if size < 1 << 14 {
        u64::from(size)
    } else if size.is_multiple_of(256) && size >> 8 < 1 << 14 {
        1 << 14 | u64::from(size >> 8)
    } else {
        return None;
    };
    Some(kind_words(CONSTANT, code << CONSTANT_ADDRESS_BITS | address))
}

/// The constant buffer view [`constant_words`] named.
fn constant_of(i: u32, generation: u32) -> View {
    let payload = payload(i, generation);
    let code = (payload >> CONSTANT_ADDRESS_BITS) as u32;
    let size = if code >> 14 == 0 { code } else { (code & 0x3fff) << 8 };
    View::Constant(payload & ((1 << CONSTANT_ADDRESS_BITS) - 1), size)
}

/// A placed view's shape and its address.
fn shape_of(unordered: bool, resource: &[u32; 14], desc: &[u32; 16]) -> (Shape, u64) {
    let mut rest = [0u32; 14];
    rest[..10].copy_from_slice(&desc[..10]);
    rest[10..].copy_from_slice(&desc[12..]);
    (
        (unordered, *resource, rest),
        u64::from(desc[10]) | u64::from(desc[11]) << 32,
    )
}

fn shape_key(shape: &Shape) -> u64 {
    use std::hash::BuildHasher;
    std::hash::BuildHasherDefault::<std::hash::DefaultHasher>::default().hash_one(shape)
}

/// The record words that name the view of placed memory at `address` with
/// shape number `shape`.
fn placed_words(shape: u32, address: u64) -> [u32; 2] {
    kind_words(PLACED, u64::from(shape) << PLACED_ADDRESS_BITS | address)
}

impl Views {
    /// The number of `shape`, a new one if it is not there yet; `None` when
    /// the shapes are full (the view is then a table entry).
    fn shape(&mut self, shape: Shape) -> Option<u32> {
        let key = shape_key(&shape);
        if let Some(&at) = self.shape_index.get(&key) {
            // shortcut: of two shapes with one hash the second is a table entry, as when the shapes are full.
            return (self.shapes[at as usize] == shape).then_some(at);
        }
        if self.shapes.len() >= SHAPES {
            return None;
        }
        let at = self.shapes.len() as u32;
        self.shapes.push(shape);
        self.shape_index.insert(key, at);
        #[cfg(feature = "diagnostics")]
        if matches!(at + 1, 1024 | 16384 | 131_072 | 1_048_576) {
            self.log_shape_words();
        }
        Some(at)
    }

    /// The number of `shape` if it is there already.
    fn known_shape(&self, shape: &Shape) -> Option<u32> {
        let &at = self.shape_index.get(&shape_key(shape))?;
        (self.shapes[at as usize] == *shape).then_some(at)
    }

    /// How many distinct values each word of the shapes kept takes: the
    /// words that tell shapes apart. Logged as the count of shapes passes a
    /// few marks, to see which fields make shapes many.
    #[cfg(feature = "diagnostics")]
    fn log_shape_words(&self) {
        let distinct = |word: &dyn Fn(&Shape) -> u32| {
            self.shapes.iter().map(word).collect::<std::collections::BTreeSet<_>>().len()
        };
        let resource: Vec<usize> = (0..14).map(|w| distinct(&|s: &Shape| s.1[w])).collect();
        let view: Vec<usize> = (0..14).map(|w| distinct(&|s: &Shape| s.2[w])).collect();
        let unordered = distinct(&|s: &Shape| u32::from(s.0));
        tracing::info!(
            shapes = self.shapes.len(),
            unordered,
            ?resource,
            ?view,
            "shapes of placed views: distinct values of each word (resource description words 0-13; view description words 0-9 and 12-15)"
        );
    }

    /// The view of placed memory [`placed_words`] named.
    fn placed_of(&self, i: u32, generation: u32) -> Option<View> {
        let payload = payload(i, generation);
        let address = payload & ((1 << PLACED_ADDRESS_BITS) - 1);
        let &(unordered, resource, rest) = self.shapes.get((payload >> PLACED_ADDRESS_BITS) as usize)?;
        let mut desc = [0u32; 16];
        desc[..10].copy_from_slice(&rest[..10]);
        (desc[10], desc[11]) = (address as u32, (address >> 32) as u32);
        desc[12..].copy_from_slice(&rest[10..]);
        Some(View::Placed(unordered, resource, desc))
    }

    /// The record words of `view` as a table entry, if it has one.
    fn entry_of(&self, view: &View) -> Option<[u32; 2]> {
        let &i = self.index.get(view)?;
        Some([i, self.list[i as usize].1])
    }

    /// The record words of `view` as a table entry: its entry (a new or a
    /// freed one if the view is not there yet) and that entry's generation.
    fn entry(&mut self, view: View) -> [u32; 2] {
        self.entry_kept(view, ownerless_cap())
    }

    /// [`entry`](Self::entry), keeping at most `cap` entries of views
    /// without a resource (0: all).
    fn entry_kept(&mut self, view: View, cap: usize) -> [u32; 2] {
        let ownerless = view.resources() == [0, 0] && matches!(view, View::Constant(..) | View::Placed(..));
        let fresh = !self.index.contains_key(&view);
        let words = self.entry_made(view);
        if ownerless && fresh {
            self.ownerless.push_back((words[0], words[1]));
            while cap != 0 && self.ownerless.len() > cap {
                let (i, generation) = self.ownerless.pop_front().expect("longer than cap");
                self.free_ownerless(i, generation);
            }
        }
        words
    }

    /// Frees entry `i` if it still holds the view it had at `generation`
    /// (it may have been freed and made again for another view since).
    fn free_ownerless(&mut self, i: u32, generation: u32) {
        let Some((view, current)) = self.list.get(i as usize).cloned() else {
            return;
        };
        if current != generation || self.index.get(&view) != Some(&i) {
            return;
        }
        self.free_entry(i, &view, generation);
    }

    fn free_entry(&mut self, i: u32, view: &View, generation: u32) {
        self.index.remove(view);
        self.list[i as usize].1 = ((generation + 1) & GENERATIONS).max(1);
        #[cfg(all(windows, feature = "diagnostics"))]
        {
            let usage = &mut self.kind_usage[view.kind_index()];
            usage.occupied -= 1;
            usage.free += 1;
        }
        self.free.push(i);
    }

    fn entry_made(&mut self, view: View) -> [u32; 2] {
        let i = match self.index.get(&view) {
            Some(&i) => i,
            None => {
                let i = match self.free.pop() {
                    Some(i) => {
                        #[cfg(all(windows, feature = "diagnostics"))]
                        {
                            let old_kind = self.list[i as usize].0.kind_index();
                            self.kind_usage[old_kind].free -= 1;
                        }
                        self.list[i as usize].0 = view.clone();
                        i
                    }
                    None => {
                        self.list.push((view.clone(), 1));
                        self.list.len() as u32 - 1
                    }
                };
                #[cfg(all(windows, feature = "diagnostics"))]
                {
                    self.kind_usage[view.kind_index()].occupied += 1;
                }
                for r in view.resources().into_iter().filter(|&r| r != 0) {
                    self.by_resource.entry(r).or_default().push(i);
                }
                self.index.insert(view, i);
                i
            }
        };
        [i, self.list[i as usize].1]
    }
}

/// The two words that name `view` in a record: the view itself where its
/// kind allows ([`Views`]), else its table entry.
fn words_for(view: View) -> [u32; 2] {
    use std::sync::atomic::Ordering::Relaxed;
    if inline() {
        match &view {
            View::Constant(address, size) => {
                if let Some(words) = constant_words(*address, *size) {
                    NAMED_WHOLE[0].fetch_add(1, Relaxed);
                    return words;
                }
            }
            View::Placed(unordered, resource, desc) => {
                let (shape, address) = shape_of(*unordered, resource, desc);
                if address >> PLACED_ADDRESS_BITS == 0 {
                    // Nearly every placed view has a shape seen before.
                    let known = views_read().known_shape(&shape);
                    let shape = known.or_else(|| views_write().shape(shape));
                    if let Some(shape) = shape {
                        NAMED_WHOLE[1].fetch_add(1, Relaxed);
                        return placed_words(shape, address);
                    }
                }
            }
            _ => {}
        }
    }
    // A view written before keeps its entry and generation.
    if let Some(words) = views_read().entry_of(&view) {
        return words;
    }
    views_write().entry(view)
}

fn views_read() -> std::sync::RwLockReadGuard<'static, Views> {
    VIEWS.read().unwrap_or_else(|e| e.into_inner())
}

fn views_write() -> std::sync::RwLockWriteGuard<'static, Views> {
    VIEWS.write().unwrap_or_else(|e| e.into_inner())
}

/// For the memory line: the distinct views kept ([`Views`]), those of them
/// free for reuse, and the bytes the table and its index hold for them (an
/// estimate: an entry of the table, and one of the index with its share of
/// a tree node; the shapes likewise); then the entries that are constant
/// buffer views and views of placed memory (those that did not fit a
/// record, or all of them with `VICTORY_VIEW_INLINE=0`), the shapes of
/// placed views kept, and the views named whole in their records since the
/// start: constant buffer views, views of placed memory. Both the summary and
/// the per-kind slot counts come from the same read lock.
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn views_kept() -> ([u64; 8], [ViewKindUsage; 6]) {
    use std::sync::atomic::Ordering::Relaxed;
    let views = views_read();
    let each = size_of::<(View, u32)>() as u64;
    let shape = size_of::<Shape>() as u64;
    let [constant, .., placed] = views.kind_usage;
    let summary = [
        views.list.len() as u64,
        views.free.len() as u64,
        views.list.capacity() as u64 * each
            + views.index.len() as u64 * each * 3 / 2
            + views.shapes.capacity() as u64 * shape
            + views.shape_index.len() as u64 * size_of::<(u64, u32)>() as u64 * 3 / 2,
        constant.occupied + constant.free,
        placed.occupied + placed.free,
        views.shapes.len() as u64,
        NAMED_WHOLE[0].load(Relaxed),
        NAMED_WHOLE[1].load(Relaxed),
    ];
    (summary, views.kind_usage)
}

impl View {
    /// The resource objects the view names.
    fn resources(&self) -> [usize; 2] {
        match *self {
            View::Shader(r, _) => [r, 0],
            View::Unordered(r, counter, _) => [r, counter],
            _ => [0, 0],
        }
    }

    /// The texture object the view is of, if that texture has no host copy
    /// yet (`crate::resource::host_waits`): a slot written with such a
    /// view is left null until the texture is first used.
    #[cfg(windows)]
    fn waiting_texture(&self) -> Option<usize> {
        match *self {
            View::Shader(r, _) | View::Unordered(r, _, _)
                if crate::resource::host_waits(r as This) =>
            {
                Some(r)
            }
            _ => None,
        }
    }
}

/// Title slots (their handles) written with a view of a texture object
/// that had no host copy then, by texture. A title texture gets its host
/// copy on first use (`resource.rs` `Hosted`), and the title
/// writes each texture's views into its view heap when it creates the
/// texture: making those host slots would make every host copy at once.
/// They are made when the texture is: translated shaders index the heap
/// directly, so a slot is not always made again by the draw that reads it.
#[cfg(windows)]
static WAITING: std::sync::Mutex<std::collections::BTreeMap<usize, Vec<u64>>> =
    std::sync::Mutex::new(std::collections::BTreeMap::new());

/// Textures given their host copy since [`make_waiting`] last ran.
#[cfg(windows)]
static MADE_LATE: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());

/// Title texture `resource` was given its host copy (called under that
/// texture's own lock: nothing else is locked here).
#[cfg(windows)]
pub(crate) fn texture_made(resource: usize) {
    MADE_LATE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push(resource);
}

/// The slot at `handle` was written with a view of texture `resource`,
/// which has no host copy: it waits for one.
#[cfg(windows)]
fn wait_for(resource: usize, handle: u64) {
    /// A texture with more slots waiting than this is in use for all we
    /// know (its views are being copied about): its host copy is made.
    const MOST: usize = 64;
    let many = {
        let mut waiting = WAITING.lock().unwrap_or_else(|e| e.into_inner());
        let slots = waiting.entry(resource).or_default();
        if !slots.contains(&handle) {
            slots.push(handle);
        }
        slots.len() > MOST
    };
    if many {
        // Makes it, and notes it made.
        let _ = crate::resource::host_copy(resource as This);
    } else if !crate::resource::host_waits(resource as This) {
        // Made by another thread since the slot was looked at: that making
        // may have been taken by `make_waiting` before this slot was noted.
        texture_made(resource);
    }
}

/// Makes the host slots that waited for a texture's host copy, for the
/// textures made since the last call. Called before a submission runs, by
/// the thread that submits: a slot is read when a list runs.
#[cfg(windows)]
pub(crate) fn make_waiting() {
    let made = std::mem::take(&mut *MADE_LATE.lock().unwrap_or_else(|e| e.into_inner()));
    for resource in made {
        let handles = WAITING
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&resource);
        for handle in handles.into_iter().flatten() {
            // Only if the slot still holds a view of that texture: the
            // title may have written it since (and a view of another
            // texture without a host copy must go on waiting).
            // (Looked up first: the handle is read only while its heap is
            // there.)
            // (The heap stays while the lock is held: `mirror`.)
            let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
            let Some(&(_, _, at)) = heaps.iter().find(|&&(start, end, _)| (start..end).contains(&handle)) else {
                continue;
            };
            // SAFETY: a registered heap, live while the lock is held.
            let heap = unsafe { &*(at as *const Heap) };
            let Some(index) = heap.index(handle) else {
                continue;
            };
            let same = match view(handle) {
                Some(View::Shader(r, _) | View::Unordered(r, _, _)) => r == resource,
                _ => false,
            };
            if same && let Some(Some(host)) = heap.host.get() {
                host.mirror(index, handle, true);
            }
        }
    }
}

fn record(handle: u64, view: View) {
    if handle == 0 {
        return;
    }
    let [i, generation] = words_for(view);
    // SAFETY: the handle is a descriptor slot, at least 16 bytes.
    unsafe { (handle as *mut [u32; 3]).write_unaligned([VIEW_TAG, i, generation]) };
    #[cfg(windows)]
    mirror(handle);
}

/// The title released `resource`: the views naming it go stale.
pub(crate) fn forget(resource: usize) {
    #[cfg(windows)]
    WAITING
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .remove(&resource);
    let mut views = views_write();
    let Some(entries) = views.by_resource.remove(&resource) else {
        return;
    };
    for i in entries {
        let (view, generation) = views.list[i as usize].clone();
        // Entries freed through a view's other resource may hold another
        // view by now.
        if views.index.get(&view) != Some(&i) || !view.resources().contains(&resource) {
            continue;
        }
        // The view's other resource (a counter) keeps a stale entry, which
        // `forget` of that resource skips.
        views.free_entry(i, &view, generation);
    }
}

/// The view the record words `i` and `generation` name: the one they hold
/// whole, or the view of entry `i` if `generation` is still its own.
fn view_at(i: u32, generation: u32) -> Option<View> {
    match generation >> KIND_SHIFT {
        0 => {
            let views = views_read();
            let (view, current) = views.list.get(i as usize)?;
            (*current == generation).then(|| view.clone())
        }
        CONSTANT => Some(constant_of(i, generation)),
        PLACED => views_read().placed_of(i, generation),
        _ => None,
    }
}

/// The words of the record at `handle`, if they are a view's.
pub(crate) fn record_at(handle: u64) -> Option<[u32; 3]> {
    if handle == 0 {
        return None;
    }
    // SAFETY: the handle is a descriptor slot the title passes back.
    let words = unsafe { (handle as *const [u32; 3]).read_unaligned() };
    (words[0] == VIEW_TAG).then_some(words)
}

/// The view recorded at `handle`, if one is (and its resource is alive).
pub(crate) fn view(handle: u64) -> Option<View> {
    if handle == 0 {
        return None;
    }
    // SAFETY: the handle is a descriptor slot the title passes back.
    let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
    if tag != VIEW_TAG {
        return None;
    }
    view_at(i, generation)
}

/// A description of `N` words, if the title passes one.
fn words<const N: usize>(desc: *const [u32; N]) -> Option<[u32; N]> {
    // SAFETY: the title passes a description or null.
    (!desc.is_null()).then(|| unsafe { desc.read_unaligned() })
}

/// Device method.
pub(crate) extern "system" fn create_cbv(_: This, desc: *const [u32; 4], handle: u64) {
    if let Some([lo, hi, size, _]) = words(desc) {
        let address = u64::from(lo) | u64::from(hi) << 32;
        record(handle, View::Constant(address, size));
    }
}

/// Device method.
pub(crate) extern "system" fn create_srv(
    _: This,
    resource: This,
    desc: *const [u32; 10],
    handle: u64,
) {
    record(handle, View::Shader(resource as usize, words(desc)));
}

/// Device method.
pub(crate) extern "system" fn create_uav(
    _: This,
    resource: This,
    counter: This,
    desc: *const [u32; 10],
    handle: u64,
) {
    let view = View::Unordered(resource as usize, counter as usize, words(desc));
    record(handle, view);
}

/// Copy only the fields of a placed resource description. Padding can differ
/// between two views of the same texture and the caller may leave it unwritten.
fn placed_resource_words(desc: *const [u32; 14]) -> Option<[u32; 14]> {
    if desc.is_null() {
        return None;
    }
    let src = desc.cast::<u32>();
    let mut fields = [0; 14];
    // SAFETY: the caller supplies a 56-byte ResourceDesc, possibly unaligned.
    // Dimension occupies word 0. Alignment through Flags occupy words 2..13.
    unsafe {
        fields[0] = src.read_unaligned();
        fields[2..13].copy_from_slice(&src.add(2).cast::<[u32; 11]>().read_unaligned());
    }
    Some(fields)
}

/// The words of a placed view's description with those its kind does not
/// use set to zero. The first ten words are the PC's view description: for a
/// shader resource view, the format, the dimension, the component mapping,
/// a word of padding (the union that follows starts on 8 bytes) and the
/// union's six words; for an unordered access view, the format, the
/// dimension and the union's eight words. Each dimension uses only the head
/// of the union, and the title leaves the rest unwritten: run vt-shapes saw
/// 131,392 values in word 3 and 36,413 in word 9 of a million shapes, each
/// a new shape for the same view. Words 10 and on (the address and the
/// console's own fields) are kept; an unknown dimension keeps all.
fn placed_view_words(desc: *const [u32; 16], unordered: bool) -> Option<[u32; 16]> {
    let mut d = words(desc)?;
    // Words of the union each dimension uses (D3D12_BUFFER_SRV and the like).
    let (start, used) = if unordered {
        // D3D12_UAV_DIMENSION: buffer, 1D, 1D array, 2D, 2D array, 2D MS,
        // 2D MS array, 3D.
        const USED: [usize; 9] = [0, 7, 1, 3, 2, 4, 0, 2, 3];
        (2, USED.get(d[1] as usize).copied())
    } else {
        // D3D12_SRV_DIMENSION: buffer, 1D, 1D array, 2D, 2D array, 2D MS,
        // 2D MS array, 3D, cube, cube array, acceleration structure.
        const USED: [usize; 12] = [0, 5, 3, 5, 4, 6, 0, 2, 3, 3, 5, 2];
        (4, USED.get(d[1] as usize).copied())
    };
    if let Some(used) = used.filter(|_| d[1] != 0) {
        if !unordered {
            d[3] = 0;
        }
        d[start + used..10].fill(0);
    }
    Some(d)
}

/// Device methods 0x230 and 0x238: shader resource and unordered access
/// views of memory the title placed itself, `(resource desc, view desc,
/// handle)`.
pub(crate) extern "system" fn create_placed_srv(
    _: This,
    resource: *const [u32; 14],
    desc: *const [u32; 16],
    handle: u64,
) {
    if let (Some(resource), Some(desc)) = (placed_resource_words(resource), placed_view_words(desc, false)) {
        record(handle, View::Placed(false, resource, desc));
    }
}

pub(crate) extern "system" fn create_placed_uav(
    _: This,
    resource: *const [u32; 14],
    desc: *const [u32; 16],
    handle: u64,
) {
    if let (Some(resource), Some(desc)) = (placed_resource_words(resource), placed_view_words(desc, true)) {
        record(handle, View::Placed(true, resource, desc));
    }
}

/// Device method.
pub(crate) extern "system" fn create_sampler(_: This, desc: *const [u32; 13], handle: u64) {
    if let Some(desc) = words(desc) {
        record(handle, View::Sampler(desc));
    }
}

/// Device method 0x1e8.
pub(crate) extern "system" fn create_sampler_x(_: This, desc: *const [u32; 27], handle: u64) {
    if let Some(desc) = words(desc) {
        record(handle, View::ConsoleSampler(desc));
    }
}

/// Device method: `count` descriptors of `kind` from `src` to `dst`.
pub(crate) extern "system" fn copy_simple(_: This, count: u32, dst: u64, src: u64, kind: u32) {
    let Some(step) = increment(kind).map(u64::from) else {
        return;
    };
    let len = u64::from(count) * step;
    if dst != 0 && src != 0 {
        // SAFETY: both ranges are descriptor slots the title owns.
        #[cfg(windows)]
        unsafe {
            crate::titlewrite::put(dst as *mut u8, src as *const u8, len as usize)
        };
        // SAFETY: as above.
        #[cfg(not(windows))]
        unsafe {
            std::ptr::copy(src as *const u8, dst as *mut u8, len as usize)
        };
        #[cfg(windows)]
        for i in 0..u64::from(count) {
            mirror(dst + i * step);
        }
    }
}

/// Device method: ranges of descriptors (start and count each) gathered
/// from the sources and scattered to the destinations, in order.
#[allow(clippy::too_many_arguments)] // the method's own arguments
pub(crate) extern "system" fn copy(
    _: This,
    dst_ranges: u32,
    dst_starts: *const u64,
    dst_sizes: *const u32,
    src_ranges: u32,
    src_starts: *const u64,
    src_sizes: *const u32,
    kind: u32,
) {
    let Some(step) = increment(kind) else {
        return;
    };
    // Each range as (start, count); a null size array means one descriptor
    // per range.
    let ranges = |n: u32, starts: *const u64, sizes: *const u32| -> Vec<(u64, u32)> {
        (0..n as usize)
            .map(|i| {
                // SAFETY: the title passes `n` starts and sizes (or null sizes).
                unsafe {
                    (
                        *starts.add(i),
                        if sizes.is_null() { 1 } else { *sizes.add(i) },
                    )
                }
            })
            .collect()
    };
    if dst_starts.is_null() || src_starts.is_null() {
        return;
    }
    let src = ranges(src_ranges, src_starts, src_sizes);
    let mut dst = ranges(dst_ranges, dst_starts, dst_sizes).into_iter();
    let (mut to, mut room) = dst.next().unwrap_or((0, 0));
    for (mut from, mut left) in src {
        while left > 0 {
            while room == 0 {
                let Some(next) = dst.next() else { return };
                (to, room) = next;
            }
            let n = left.min(room);
            copy_simple(std::ptr::null(), n, to, from, kind);
            let bytes = u64::from(n) * u64::from(step);
            (to, from, room, left) = (to + bytes, from + bytes, room - n, left - n);
        }
    }
}

/// The view recorded at `handle`, if one is.
pub(crate) fn rtv_record(handle: u64) -> Option<RtvRecord> {
    tagged_record(handle, RTV_TAG)
}

/// The depth view recorded at `handle`, if one is.
pub(crate) fn dsv_record(handle: u64) -> Option<RtvRecord> {
    tagged_record(handle, DSV_TAG)
}

fn tagged_record(handle: u64, tag: u32) -> Option<RtvRecord> {
    if handle == 0 {
        return None;
    }
    // SAFETY: the handle is a descriptor slot the title passes back.
    let record = unsafe { (handle as *const RtvRecord).read_unaligned() };
    (record.tag == tag).then_some(record)
}

/// Device method.
pub(crate) extern "system" fn create_heap(
    _: This,
    desc_ptr: *const HeapDesc,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc_ptr.as_ref() }) else {
        return E_INVALIDARG;
    };
    let Some(size) = increment(desc.kind).map(|i| u64::from(i) * u64::from(desc.count.max(1)))
    else {
        return E_INVALIDARG;
    };
    // SAFETY: read only as a hint: a title that passes the plain
    // description has its own stack or data after it.
    let placement = unsafe { desc_ptr.add(1).cast::<Placement>().read_unaligned() };
    // shortcut: a placement is recognised by its size matching the count,
    // since the plain and extended descriptions share the slot and the
    // interface version; find the flag that tells them apart if a title
    // trips over this.
    let placed = placement.address != 0 && placement.size == size;
    tracing::debug!(?desc, placed, "descriptor heap");
    let (start, memory) = if placed {
        if placement.address.checked_add(size).is_none() {
            return E_INVALIDARG;
        }
        (placement.address, None)
    } else {
        match Memory::zeroed(size, 64) {
            Ok(memory) => (memory.address(), Some(memory)),
            Err(hr) => return hr,
        }
    };
    let visible = desc.flags & SHADER_VISIBLE != 0 && matches!(desc.kind, VIEWS_KIND | SAMPLERS);
    let heap = Box::new(Heap {
        object: Object::new(&FUNCTIONS),
        desc,
        start,
        memory,
        #[cfg(windows)]
        host: std::sync::OnceLock::new(),
        #[cfg(windows)]
        pending: Default::default(),
        #[cfg(windows)]
        written: (placed && visible)
            .then(|| crate::watch::track(start as usize, size as usize, true))
            .flatten(),
    });
    // Renderer port #2: the title's own stores tell this heap's slots.
    #[cfg(windows)]
    if placed && visible && desc.kind == VIEWS_KIND {
        crate::port_descriptors::adopt(start, desc.count);
    }
    #[cfg(windows)]
    if visible {
        let at = std::ptr::from_ref::<Heap>(&heap) as usize;
        let mut heaps = SHADER_HEAPS.write().unwrap_or_else(|e| e.into_inner());
        heaps.push((start, start + size, at));
    }
    #[cfg(not(windows))]
    let _ = visible;
    object::hand_out(DESCRIPTOR_HEAP, heap, iid, &HEAP_IIDS, out)
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    let found = object::is(iid, &HEAP_IIDS).then_some(this);
    object::answer(DESCRIPTOR_HEAP, iid, found, out)
}
extern "system" fn release(this: This) -> u32 {
    object::release_boxed::<Heap>(this)
}
extern "system" fn desc(this: This, out: *mut HeapDesc) {
    if !out.is_null() {
        // SAFETY: checked non-null; the title passes room for a description.
        unsafe { out.write(heap(this).desc) };
    }
}
/// The same address for the CPU and the GPU start.
extern "system" fn start(this: This) -> u64 {
    heap(this).start
}

/// IUnknown 3, object 4, device child 1, heap 3.
static FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<DESCRIPTOR_HEAP, 24>());
    f[0] = query as *const ();
    f[2] = release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    f[8] = desc as *const (); // 0x40
    f[9] = start as *const (); // 0x48 CPU
    f[10] = start as *const (); // 0x50 GPU
    Table(f)
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::{S_OK, slot};

    /// What newer titles pass; tests use it throughout, so that reading the
    /// placement stays inside the value.
    #[repr(C)]
    struct Extended(HeapDesc, Placement);

    fn create(desc: HeapDesc, placement: Placement, out: &mut This) -> HRESULT {
        let extended = Extended(desc, placement);
        create_heap(std::ptr::null(), &extended.0, &HEAP_IIDS[0], out)
    }

    #[cfg(windows)]
    #[test]
    fn raw_heap_sync_stops_before_dynamic_host_slots() {
        let start = 0x1000usize;
        let step = 32;
        let capped = 5;
        assert_eq!(
            changed_slots(
                start + 3 * step + 4,
                start + 4 * step + 8,
                start,
                step as u64,
                capped
            ),
            3..5
        );
        assert_eq!(
            changed_slots(
                start + 6 * step,
                start + 9 * step,
                start,
                step as u64,
                capped
            ),
            5..5
        );
    }

    #[cfg(windows)]
    #[test]
    fn marked_and_fallback_scans_own_disjoint_slots() {
        let page = crate::watch::PAGE;
        let step = 32;
        let start = page;
        let per_page = (page / step) as u32;
        let marked = [(per_page - 1, 2), (per_page + 4, 2), (3 * per_page - 1, 2)];
        let parts = port_scan_parts(
            &marked,
            vec![start + 3 * page, start + page],
            start,
            start + 4 * page,
            step,
        );
        assert_eq!(
            parts,
            [
                (start + page - step, start + page + step, false),
                (start + page + 4 * step, start + page + 6 * step, false),
                (start + 3 * page - step, start + 3 * page + step, false),
                (start + 3 * page + step, start + 4 * page, true),
                (start + page + step, start + page + 4 * step, true),
                (start + page + 6 * step, start + 2 * page, true),
            ]
        );
    }

    #[cfg(windows)]
    #[test]
    fn a_mark_owns_its_slot_when_the_slot_straddles_a_fallback_page() {
        let page = crate::watch::PAGE;
        let step = 32;
        let start = page - step / 2;
        let marked = [(0, 2)];
        let parts = port_scan_parts(&marked, vec![page], start, start + 2 * page, step);
        assert_eq!(
            parts,
            [(start, start + 2 * step, false), (start + 2 * step, start + page, true)]
        );
    }

    #[test]
    fn heaps_hold_count_descriptors_from_their_start() {
        let null = std::ptr::null();
        let desc = HeapDesc {
            kind: 0,
            count: 1000,
            flags: 1,
            nodes: 0,
        };
        let none = Placement {
            address: 0,
            size: 0,
        };
        let mut h = null;
        assert_eq!(create(desc, none, &mut h), S_OK);
        let cpu = slot::<extern "system" fn(This) -> u64>(h, 0x48)(h);
        assert_eq!(cpu % 64, 0);
        assert_eq!(slot::<extern "system" fn(This) -> u64>(h, 0x50)(h), cpu);
        assert_eq!(increment_size(null, 0), 32);
        // SAFETY: the heap holds 1000 descriptors of 32 bytes.
        unsafe { (cpu as *mut u8).add(999 * 32 + 31).write(1) };
        assert_eq!(increment_size(null, 9), 0);
        let bad = HeapDesc { kind: 9, ..desc };
        assert_eq!(create(bad, none, &mut h), E_INVALIDARG);
    }

    /// A descriptor copied into a placed heap as bytes is mirrored before
    /// the next submission, even though no descriptor method saw the copy.
    #[cfg(windows)]
    #[test]
    fn records_copied_as_bytes_are_made_at_the_next_sync() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        let memory = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                crate::watch::PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        };
        assert!(!memory.is_null());
        let desc = HeapDesc {
            kind: SAMPLERS,
            count: 64,
            flags: SHADER_VISIBLE,
            nodes: 1,
        };
        let placement = Placement {
            address: memory as u64,
            size: 64 * 16,
        };
        let mut h: This = std::ptr::null();
        assert_eq!(create(desc, placement, &mut h), S_OK);
        if let Some(host) = heap(h).host() {
            let mut elsewhere = [0u64; 2];
            let source = elsewhere.as_mut_ptr() as u64;
            let sampler = [21, 3, 3, 3, 0, 1, 0, 0, 0, 0, 0, 0x7f7f_ffff, 0];
            create_sampler(std::ptr::null(), &sampler, source);
            // SAFETY: the heap slot and source both hold 16 descriptor bytes.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    source as *const u8,
                    memory.cast::<u8>().add(5 * 16),
                    16,
                )
            };
            assert!(!host.made(5));
            // Another test's begin_sync_heaps_for can have the heap-sync thread
            // running a sync of its own: that one may take this page's write
            // and make the record a moment after this sync returns.
            let mut made = false;
            for _ in 0..200 {
                sync_heaps();
                made = host.made(5);
                if made {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            assert!(made);
        }
        slot::<extern "system" fn(This) -> u32>(h, 0x10)(h);
        // SAFETY: this allocation belongs to the test.
        unsafe { VirtualFree(memory, 0, MEM_RELEASE) };
    }

    /// Two placed views alike but for the words their kind does not use
    /// (the padding after the component mapping, the union's unused tail)
    /// are one record; a used word still tells them apart.
    #[test]
    fn placed_views_ignore_unused_view_words() {
        let mut slot = [0u32; 8];
        let handle = slot.as_mut_ptr() as u64;
        let null = std::ptr::null();
        let resource = [1, 0, 0, 0, 0x4000, 0, 1, 0x10001, 0, 1, 0, 1, 0, 0];
        // A structured buffer view: words 4-5 first element, 6 count, 7
        // stride, 8 flags; 3 and 9 are not used.
        let srv = [0, 1, 0x1688, 0, 16, 0, 64, 16, 0, 0, 0x1000, 0x6, 0, 0, 0, 0];
        // An unordered buffer view: words 2-3 first element, 4 count, 5
        // stride, 6-7 counter offset, 8 flags; 9 is not used.
        let uav = [0, 1, 16, 0, 64, 16, 0, 0, 0, 0, 0x1000, 0x6, 0, 0, 0, 0];
        for (unordered, create, desc, unused, used) in [
            (false, create_placed_srv as extern "system" fn(_, _, _, _), srv, [3usize, 9], 6usize),
            (true, create_placed_uav, uav, [9, 9], 4),
        ] {
            create(null, &resource, &desc, handle);
            let first = slot;
            assert_eq!(view(handle), Some(View::Placed(unordered, resource, desc)));
            let mut noisy = desc;
            for w in unused {
                noisy[w] = 0xdead_beef;
            }
            create(null, &resource, &noisy, handle);
            assert_eq!(slot, first, "unused words left out");
            let mut other = desc;
            other[used] += 1;
            create(null, &resource, &other, handle);
            assert_ne!(slot, first, "a used word counts");
        }
        // A 2D texture view uses four words of the union (4-7): 8 and 9 go.
        let mut tex = [28, 4, 0x1688, 0x77, 0, 1, 0, 0, 0x55, 0x66, 0x2000, 0x6, 0, 0, 0, 0];
        create_placed_srv(null, &resource, &tex, handle);
        tex[3] = 0;
        (tex[8], tex[9]) = (0, 0);
        assert_eq!(view(handle), Some(View::Placed(false, resource, tex)));
    }

    #[test]
    fn placed_views_share_resources_across_description_padding() {
        let mut slot = [0u32; 8];
        let handle = slot.as_mut_ptr() as u64;
        let view_desc = [0u32; 16];
        let null = std::ptr::null();
        let mut resource = [3, 0, 0, 0, 8, 0, 8, 0x10001, 28, 1, 0, 0, 0, 0];
        let expected = resource;
        for (unordered, create) in [
            (false, create_placed_srv as extern "system" fn(_, _, _, _)),
            (true, create_placed_uav),
        ] {
            create(null, &expected, &view_desc, handle);
            let first = slot;
            resource[1] = 0xaaaa_aaaa;
            resource[13] = 0xbbbb_bbbb;
            create(null, &resource, &view_desc, handle);
            assert_eq!(slot, first);
            assert_eq!(
                view(handle),
                Some(View::Placed(unordered, expected, view_desc))
            );
            let mut other = resource;
            other[12] = 4;
            create(null, &other, &view_desc, handle);
            assert_ne!(slot, first);
        }
    }

    #[test]
    fn views_survive_copies() {
        let null = std::ptr::null();
        let mut heap = vec![0u64; 4 * 8]; // four 32-byte slots
        let at = |i: usize| heap.as_ptr() as u64 + 32 * i as u64;
        let (s0, s1, s2, s3) = (at(0), at(1), at(2), at(3));
        create_cbv(null, &[0x1000, 1, 256, 0], s0);
        assert_eq!(view(s0), Some(View::Constant(0x1_0000_1000, 256)));
        create_srv(null, null, std::ptr::null(), s1);
        assert_eq!(view(s1), Some(View::Shader(0, None)));
        // The same view twice is one entry.
        create_srv(null, null, std::ptr::null(), s2);
        // SAFETY: both slots are in `heap`.
        assert_eq!(unsafe { *(s1 as *const [u32; 3]) }, unsafe {
            *(s2 as *const [u32; 3])
        });
        // Two ranges of one into one range of two.
        let (dst, dst_n, src, src_n) = ([s2], [2u32], [s0, s1], [1u32, 1]);
        copy(
            null,
            1,
            dst.as_ptr(),
            dst_n.as_ptr(),
            2,
            src.as_ptr(),
            src_n.as_ptr(),
            0,
        );
        assert_eq!(view(s2), view(s0));
        assert_eq!(view(s3), view(s1));
        assert_eq!(view(at(0) + 8), None);
        heap[0] = 0;
    }

    /// Constant buffer views are named whole in their records: any address
    /// a program can have, any size below 16 KiB, multiples of 256 bytes
    /// up to 4 MiB; what does not fit says so.
    #[test]
    fn constant_views_fit_their_records() {
        let top = (1u64 << CONSTANT_ADDRESS_BITS) - 1;
        for address in [0u64, 0x100, 0x2602_0000, 0xd_6b60_0000, 0x7ffd_a34f_0000, top] {
            for size in [0u32, 1, 255, 256, 4000, (1 << 14) - 1, 1 << 14, 65536, (1 << 22) - 256] {
                let [i, generation] = constant_words(address, size).expect("fits");
                assert_eq!(generation >> KIND_SHIFT, CONSTANT);
                assert_eq!(view_at(i, generation), Some(View::Constant(address, size)), "{address:#x} {size}");
            }
        }
        assert_eq!(constant_words(top + 1, 256), None, "an address past 47 bits");
        assert_eq!(constant_words(0x1000, (1 << 14) + 1), None, "16 KiB or more, no multiple of 256");
        assert_eq!(constant_words(0x1000, 1 << 22), None, "4 MiB");
        // Different views never share their words.
        assert_ne!(constant_words(0x1000, 256), constant_words(0x1000, 512));
        assert_ne!(constant_words(0x1000, 256), constant_words(0x1100, 256));
        assert_ne!(constant_words(0x1000, 1 << 14), constant_words(0x1000, 64));
    }

    /// Each slot is counted once as it is freed and reused across kinds.
    #[cfg(all(windows, feature = "diagnostics"))]
    #[test]
    fn diagnostic_counts_follow_distinct_entries_including_reused_slots() {
        let check = |views: &Views| {
            let mut expected = [ViewKindUsage::default(); 6];
            for view in views.index.keys() {
                expected[view.kind_index()].occupied += 1;
            }
            for &i in &views.free {
                expected[views.list[i as usize].0.kind_index()].free += 1;
            }
            assert_eq!(views.kind_usage, expected);
            assert_eq!(
                expected.iter().map(|usage| usage.slot_bytes()).sum::<u64>(),
                (views.list.len() * size_of::<(View, u32)>()) as u64
            );
        };
        let mut views = Views::default();
        let shader = View::Shader(1, None);
        let [i, generation] = views.entry(shader.clone());
        views.free_entry(i, &shader, generation);
        assert_eq!(views.kind_usage[1], ViewKindUsage { occupied: 0, free: 1 });
        check(&views);
        let constant = View::Constant(0x1234, 256);
        assert_eq!(views.entry(constant.clone()), [0, 2]);
        assert_eq!(views.entry(constant), [0, 2]);
        let placed = View::Placed(false, [0; 14], [0; 16]);
        let index = views.entry(placed.clone());
        assert_eq!(views.entry(placed), index);
        views.entry(View::Unordered(1, 2, None));
        views.entry(View::Sampler([0; 13]));
        views.entry(View::ConsoleSampler([0; 27]));
        assert_eq!(views.kind_usage.map(|usage| usage.occupied), [1, 0, 1, 1, 1, 1]);
        check(&views);
        // Under a cap of 2, placed views push the constant one out; its
        // freed entry, reused, changes kind.
        for n in 1..6u32 {
            views.entry_kept(View::Placed(false, [n; 14], [n; 16]), 2);
            check(&views);
        }
        views.entry_kept(View::Constant(0x5678, 512), 2);
        check(&views);
    }

    /// Entries of views no resource releases stay under their cap: the
    /// oldest go stale, the newest still name their views, entries are
    /// reused rather than added, and views of resources are never dropped.
    #[test]
    fn ownerless_entries_are_capped() {
        let mut views = Views::default();
        let placed = |n: u32| View::Placed(false, [n; 14], [n; 16]);
        let named = |views: &Views, words: [u32; 2], view: &View| {
            views.list.get(words[0] as usize).is_some_and(|(v, g)| *g == words[1] && v == view)
        };
        let words: Vec<[u32; 2]> = (0..10).map(|n| views.entry_kept(placed(n), 4)).collect();
        assert_eq!(views.index.len(), 4, "four kept");
        assert!(views.list.len() <= 5, "entries reused, not added: {}", views.list.len());
        for n in 0..6 {
            assert!(!named(&views, words[n], &placed(n as u32)), "entry {n} stale");
        }
        for n in 6..10 {
            assert!(named(&views, words[n], &placed(n as u32)), "entry {n} kept");
        }
        // The same view written again keeps its words.
        assert_eq!(views.entry_kept(placed(9), 4), words[9]);
        // A view of a resource is freed with the resource only.
        let shader = View::Shader(0x1234, None);
        let kept = views.entry_kept(shader.clone(), 4);
        for n in 10..40 {
            views.entry_kept(placed(n), 4);
        }
        assert!(named(&views, kept, &shader));
        assert!(views.list.len() <= 6);
        // 0 keeps them all.
        for n in 40..60 {
            views.entry_kept(placed(n), 0);
        }
        assert_eq!(views.index.len(), 1 + 4 + 20);
    }

    /// The leak of run 65: a title that writes constant buffer views and
    /// views of placed memory at ever new addresses adds nothing to the
    /// table; one shape stands for all the placed views alike.
    #[test]
    fn views_at_new_addresses_add_no_entries() {
        if !inline() {
            return;
        }
        let null = std::ptr::null();
        let slots = [0u64; 4];
        let at = slots.as_ptr() as u64;
        // A resource description and a view no other test uses.
        // (Its padding words, 1 and 13, are zero: placed views leave them out.)
        let resource: [u32; 14] = std::array::from_fn(|k| if k == 1 || k == 13 { 0 } else { 0x5ee0_0000 + k as u32 });
        let mut desc: [u32; 16] = std::array::from_fn(|k| 0x7a70_0000 + k as u32);
        let mine = |v: &View| match v {
            View::Constant(address, _) => (0x6_0000_0000..0x6_1000_0000).contains(address),
            View::Placed(_, r, _) => *r == resource,
            _ => false,
        };
        let entries = || VIEWS.read().unwrap().list.iter().filter(|(v, _)| mine(v)).count();
        let shapes = || VIEWS.read().unwrap().shapes.iter().filter(|s| s.1 == resource).count();
        for n in 0..20_000u64 {
            let address = 0x6_0000_0000 + n * 0x100;
            create_cbv(null, &[address as u32, (address >> 32) as u32, 512, 0], at);
            assert_eq!(view(at), Some(View::Constant(address, 512)));
            let placed = 0xd_4940_0000 + n * 0x1_0000;
            (desc[10], desc[11]) = (placed as u32, (placed >> 32) as u32);
            create_placed_srv(null, &resource, &desc, at);
            assert_eq!(view(at), Some(View::Placed(false, resource, desc)));
        }
        assert_eq!(entries(), 0, "no table entries");
        assert_eq!(shapes(), 1, "one shape for 20,000 placed views");
        // The same shape, unordered: another shape.
        create_placed_uav(null, &resource, &desc, at);
        assert_eq!(view(at), Some(View::Placed(true, resource, desc)));
        assert_eq!(shapes(), 2);
        // Placed memory past 1 TiB (not title memory) is a table entry.
        (desc[10], desc[11]) = (0, 0x100);
        create_placed_srv(null, &resource, &desc, at);
        assert_eq!(view(at), Some(View::Placed(false, resource, desc)));
        assert_eq!(entries(), 1);
        // A record's words are the same for the same view (the host knows
        // a slot's view by them), and differ with the address.
        // SAFETY: both handles are slots of 32 bytes above.
        let words = |handle: u64| unsafe { *(handle as *const [u32; 3]) };
        let other = [0u64; 4];
        create_cbv(null, &[0x40, 6, 512, 0], at);
        create_cbv(null, &[0x40, 6, 512, 0], other.as_ptr() as u64);
        assert_eq!(words(at), words(other.as_ptr() as u64));
        create_cbv(null, &[0x140, 6, 512, 0], other.as_ptr() as u64);
        assert_ne!(words(at), words(other.as_ptr() as u64));
    }

    /// A released resource's views go stale; the same view of a new
    /// resource at the same address is a new record.
    #[test]
    fn views_die_with_their_resource() {
        let null = std::ptr::null();
        let slots = [0u64; 2 * 4];
        let (a, b) = (slots.as_ptr() as u64, slots.as_ptr() as u64 + 32);
        let resource = 0x7777_0000usize as This;
        create_srv(null, resource, std::ptr::null(), a);
        assert_eq!(view(a), Some(View::Shader(0x7777_0000, None)));
        forget(0x7777_0000);
        assert_eq!(view(a), None, "stale after release");
        create_srv(null, resource, std::ptr::null(), b);
        assert_eq!(view(b), Some(View::Shader(0x7777_0000, None)));
        // SAFETY: both slots are in `slots`.
        let (old, new) = unsafe { (*(a as *const [u32; 3]), *(b as *const [u32; 3])) };
        assert_ne!(old, new, "a new generation");
    }

    /// The title places its shader-visible heap in its own memory
    /// and checks that the heap starts there.
    #[test]
    fn placed_heaps_start_at_the_title_memory() {
        let mut memory = vec![0u8; 64 * 32];
        let address = memory.as_mut_ptr() as u64;
        let desc = HeapDesc {
            kind: 0,
            count: 64,
            flags: 1,
            nodes: 1,
        };
        let mut h = std::ptr::null();
        assert_eq!(
            create(
                desc,
                Placement {
                    address,
                    size: 64 * 32
                },
                &mut h
            ),
            S_OK
        );
        assert_eq!(slot::<extern "system" fn(This) -> u64>(h, 0x48)(h), address);
        assert_eq!(slot::<extern "system" fn(This) -> u64>(h, 0x50)(h), address);
        // A size that does not match the count is not a placement.
        assert_eq!(create(desc, Placement { address, size: 7 }, &mut h), S_OK);
        assert_ne!(slot::<extern "system" fn(This) -> u64>(h, 0x48)(h), address);
    }

    /// Load test (`cargo test -p victory-gpu --lib --locked placed_view_shapes_load -- --ignored --nocapture`).
    /// A million placed views whose descriptions differ only in padding words
    /// make one shape; a million that really differ make a million. Then each
    /// description word is varied alone, to list the words that tell shapes apart.
    #[cfg(windows)]
    #[test]
    #[ignore = "load test: about a million table inserts"]
    fn placed_view_shapes_load() {
        use crate::loadprobe::{mix, private_bytes};
        const N: u64 = 1_000_000;
        let shapes = || VIEWS.read().unwrap().shapes.len();
        let mut slot = [0u32; 8];
        let handle = slot.as_mut_ptr() as u64;
        let null = std::ptr::null();
        // A 2D texture SRV (dimension 4): the union's words 4-7 are used; 3, 8 and 9 are padding.
        let resource = [3, 0, 0, 0, 0x4321, 0, 0x4321, 0x10001, 28, 1, 0, 0, 0, 0];
        let desc = [28, 4, 0x1688, 0, 0, 1, 0, 0, 0, 0, 0x2000, 0x6, 0, 0, 0, 0];
        let before = (shapes(), private_bytes());
        for n in 0..N {
            let mut r = resource;
            r[1] = mix(n);
            r[13] = mix(n + N);
            let mut d = desc;
            (d[3], d[8], d[9]) = (mix(n + 2 * N), mix(n + 3 * N), mix(n + 4 * N));
            let address = 0xd_0000_0000 + n * 0x1_0000;
            (d[10], d[11]) = (address as u32, (address >> 32) as u32);
            create_placed_srv(null, &r, &d, handle);
        }
        let padding = (shapes() - before.0, private_bytes().saturating_sub(before.1));
        eprintln!("padding only: {N} views -> {} new shapes, private memory +{} KiB", padding.0, padding.1 >> 10);
        // Control: the mip level (word 4) really differs.
        let mid = (shapes(), private_bytes());
        for n in 0..N / 10 {
            let mut d = desc;
            (d[4], d[5]) = (n as u32 + 1, 0);
            create_placed_srv(null, &resource, &d, handle);
        }
        let control = (shapes() - mid.0, private_bytes().saturating_sub(mid.1));
        eprintln!("control: {} really different views -> {} new shapes, private memory +{} KiB", N / 10, control.0, control.1 >> 10);
        assert_eq!(control.0 as u64, N / 10, "views that differ are shapes");
        // One word at a time, 64 values each, from a base the other tests do not use.
        let mut identity = Vec::new();
        for (name, count) in [("resource", 14usize), ("view", 16)] {
            for w in 0..count {
                if name == "view" && (w == 10 || w == 11) {
                    continue; // the address
                }
                let base = shapes();
                for v in 0..64u32 {
                    let mut r = [3, 0, 0, 0, 0x7777, 0, 0x7777, 0x10001, 28, 1, 0, 0, 0, 0];
                    let mut d = [28, 4, 0x1688, 0, 0, 1, 0, 0, 0, 0, 0x2000, 0x6, 0, 0, 0, 0];
                    let v = 0x5000_0000 + v * 0x1111;
                    if name == "resource" { r[w] = v } else { d[w] = v }
                    create_placed_srv(null, &r, &d, handle);
                }
                let grew = shapes() - base;
                eprintln!("{name} word {w:2}: 64 values -> {grew} shapes");
                if grew > 2 {
                    identity.push(format!("{name}[{w}]"));
                }
            }
        }
        eprintln!("words that tell shapes apart: {identity:?}");
        assert!(padding.0 <= 4, "padding alone made {} shapes out of {N}", padding.0);
    }

    #[cfg(windows)]
    #[test]
    fn sync_rounds_coalesce_queued_submissions_and_early_rounds() {
        let mut r = super::SyncRounds::default();
        // Three queued submissions enqueued before any round starts share stamp 1.
        let s1 = r.stamp();
        let s2 = r.stamp();
        let s3 = r.stamp();
        assert_eq!((s1, s2, s3), (1, 1, 1));
        assert!(!r.has_asked());
        // First submission triggers round 1.
        assert_eq!(r.request(Some(s1)), (Some(1), true));
        assert!(r.has_asked());
        // Worker starts round 1; a submission enqueued *during* round 1 gets stamp 2.
        r.start_round();
        let s_during = r.stamp();
        assert_eq!(s_during, 2);
        // Round 1 finishes: submissions 2 and 3 need no round and no wait.
        r.finish_round();
        assert!(!r.has_asked());
        assert_eq!(r.request(Some(s2)), (None, false));
        assert_eq!(r.request(Some(s3)), (None, false));
        // An early round runs before `s_during` is submitted, covering stamp 2.
        r.start_round();
        r.finish_round();
        assert!(!r.has_asked());
        assert_eq!(r.request(Some(s_during)), (None, false));
        // Unstamped (synchronous) caller always demands a fresh round after `started`.
        assert_eq!(r.request(None), (Some(3), true));
        assert!(r.has_asked());
        r.start_round();
        r.finish_round();
        assert!(!r.has_asked());
        // Lazily creating a host heap during thin translation raises `floor`
        // so an earlier-stamped submission cannot skip syncing the new heap.
        r.note_heap();
        assert_eq!(r.request(Some(s2)), (Some(4), true));
        r.start_round();
        r.finish_round();
        assert_eq!(r.request(Some(s3)), (None, false));
    }
}
