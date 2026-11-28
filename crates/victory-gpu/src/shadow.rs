//! Host copies of title buffers (decision 0007, step 3 and "Copies of title
//! memory"). Title buffers are title memory, which the host GPU cannot
//! read, so a buffer a draw uses gets a persistent host buffer of its size
//! on first use (system memory shaders may also write), and the ranges each
//! submission reads (vertex, index and constant data, buffer views) are
//! brought up to date from title memory before it runs: the 64 KiB pages
//! the title wrote since they were last copied ([`crate::watch`]: write
//! watching, or read-only guards on physical-page views), all of it where
//! neither works. A title GPU address inside the buffer is the same offset
//! in its copy. Ranges shaders write go back to title memory after the
//! submission ([`Shadow::gpu_writes`]).
//!
//! shortcut: only the ranges draws are known to read are brought up to
//! date (vertex and index buffer views, constant buffers in tables, root
//! descriptors, buffer views in tables); buffers read through directly
//! indexed heaps keep what their first use copied.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock, Weak};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

use crate::host::{HostList, host};
use crate::watch::PAGE;

/// Per-page last copied write-watch sequence, paired with a 64-page-per-word
/// bitmask of uncopied pages so partially-initialized buffers and windows
/// still hit the O(1)/O(|written|) fast path on initialized subranges and skip
/// initialized regions 64 pages at a time when uncopied pages exist.
struct CopiedPages {
    at: Vec<u64>,
    /// Bit `i` (word `i / 64`, bit `i % 64`) is 1 iff `at[i] == 0`.
    /// Unused high bits in the final word are always 0.
    uncopied_bits: Vec<u64>,
    uncopied: usize,
}

impl CopiedPages {
    fn new(pages: usize) -> Self {
        let words = pages.div_ceil(64);
        let mut uncopied_bits = vec![u64::MAX; words];
        let rem = pages % 64;
        if rem != 0 {
            uncopied_bits[words - 1] = (1u64 << rem) - 1;
        }
        Self {
            at: vec![0; pages],
            uncopied_bits,
            uncopied: pages,
        }
    }

    fn set(&mut self, index: usize, now: u64) {
        if let Some(slot) = self.at.get_mut(index) {
            if *slot == 0 && now != 0 {
                self.uncopied -= 1;
                self.uncopied_bits[index / 64] &= !(1u64 << (index % 64));
            } else if *slot != 0 && now == 0 {
                self.uncopied += 1;
                self.uncopied_bits[index / 64] |= 1u64 << (index % 64);
            }
            *slot = now;
        }
    }

    #[cfg(test)]
    fn all_copied(&self) -> bool {
        self.uncopied == 0
    }

    /// Whether every page index in `lo..=hi` has been copied whole at least once.
    fn range_all_copied(&self, lo: usize, hi: usize) -> bool {
        if lo > hi || self.uncopied == 0 {
            return true;
        }
        let first_w = lo / 64;
        let last_w = hi / 64;
        let lo_mask = u64::MAX << (lo % 64);
        let hi_mask = u64::MAX >> (63 - (hi % 64));
        if first_w == last_w {
            return (self.uncopied_bits[first_w] & lo_mask & hi_mask) == 0;
        }
        if (self.uncopied_bits[first_w] & lo_mask) != 0 {
            return false;
        }
        for &word in &self.uncopied_bits[first_w + 1..last_w] {
            if word != 0 {
                return false;
            }
        }
        (self.uncopied_bits[last_w] & hi_mask) == 0
    }

    /// Lowest page index `>= from_idx` and `<= hi` that has never been copied whole.
    fn next_uncopied(&self, from_idx: usize, hi: usize) -> Option<usize> {
        if from_idx > hi || self.uncopied == 0 {
            return None;
        }
        let mut w = from_idx / 64;
        let last_w = hi / 64;
        let mut word = self.uncopied_bits[w] & (u64::MAX << (from_idx % 64));
        loop {
            if w == last_w {
                word &= u64::MAX >> (63 - (hi % 64));
            }
            if word != 0 {
                return Some(w * 64 + word.trailing_zeros() as usize);
            }
            if w == last_w {
                return None;
            }
            w += 1;
            word = self.uncopied_bits[w];
        }
    }

    fn written_start(written: &[(usize, u64)], first_page: usize) -> usize {
        if written.len() > 8 {
            written.partition_point(|&(p, _)| p < first_page)
        } else {
            let mut i = 0;
            while i < written.len() && written[i].0 < first_page {
                i += 1;
            }
            i
        }
    }

    /// Plans the coalesced byte runs `(run_from, run_to, known)` inside
    /// `[from, to)` that must be copied from title memory, together with
    /// `(page, taken)` for pages taken before and written since, while updating
    /// whole-page copied timestamps to `now`. Runs purely on metadata so
    /// callers can release the `copied` mutex before invoking memory copies.
    #[allow(clippy::too_many_arguments)]
    fn plan_stale(
        &mut self,
        base: usize,
        start: usize,
        end: usize,
        from: usize,
        to: usize,
        now: u64,
        ranged: u64,
        written: &[(usize, u64)],
    ) -> (Vec<(usize, usize, bool)>, Vec<(usize, u64)>) {
        let mut runs: Vec<(usize, usize, bool)> = Vec::new();
        let mut again_at: Vec<(usize, u64)> = Vec::new();
        if from >= to {
            return (runs, again_at);
        }
        let first_page = from / PAGE;
        let last = (to - 1) / PAGE;
        let lo = first_page - base;
        let hi = last - base;

        let push_page = |runs: &mut Vec<(usize, usize, bool)>,
                         run: &mut Option<(usize, usize, bool)>,
                         page: usize,
                         again: bool| {
            match *run {
                Some((_, run_end, _)) if page < run_end => {}
                Some((run_start, run_end, known)) if page == run_end && known == again => {
                    *run = Some((run_start, page + 1, known));
                }
                Some((run_start, run_end, known)) => {
                    runs.push(((run_start * PAGE).max(from), (run_end * PAGE).min(to), known));
                    *run = Some((page, page + 1, again));
                }
                None => *run = Some((page, page + 1, again)),
            }
        };

        let mut run: Option<(usize, usize, bool)> = None;
        let w_start = Self::written_start(written, first_page);

        // Fast path: every page in `[from, to)` has been copied at least once
        // (whole or via a covering `ranged` capture).
        if ranged != 0 || self.range_all_copied(lo, hi) {
            if w_start >= written.len() || written[w_start].0 > last {
                return (runs, again_at);
            }
            for &(page, write) in &written[w_start..] {
                if page > last {
                    break;
                }
                let idx = page - base;
                let taken = self.at[idx].max(ranged);
                if write <= taken {
                    continue;
                }
                if let Some((_, run_end, _)) = run
                    && page < run_end
                {
                    continue;
                }
                // A page only partly in the range is copied in part: it stays
                // stale for the rest.
                if (page * PAGE).max(start) >= from && ((page + 1) * PAGE).min(end) <= to {
                    self.set(idx, now);
                }
                again_at.push((page, taken));
                push_page(&mut runs, &mut run, page, true);
            }
            if let Some((run_start, run_end, known)) = run {
                runs.push(((run_start * PAGE).max(from), (run_end * PAGE).min(to), known));
            }
            return (runs, again_at);
        }

        // Sparse merge of uncopied pages (via 64-page bitmask) and dirty
        // `written` entries in `[first_page, last]`, skipping already-clean
        // initialized spans without touching `self.at`.
        let mut wi = w_start;
        let mut next_u = self.next_uncopied(lo, hi);
        loop {
            while wi < written.len() && written[wi].0 <= last {
                let (wp, write) = written[wi];
                let idx = wp - base;
                if self.at[idx] == 0 || write > self.at[idx] {
                    break;
                }
                wi += 1;
            }
            let next_w = (wi < written.len() && written[wi].0 <= last).then(|| written[wi].0 - base);
            let (idx, taken) = match (next_u, next_w) {
                (Some(u), Some(w)) if u < w => {
                    next_u = self.next_uncopied(u + 1, hi);
                    (u, 0)
                }
                (Some(u), Some(w)) if w < u => {
                    let page = base + w;
                    while wi < written.len() && written[wi].0 == page {
                        wi += 1;
                    }
                    (w, self.at[w])
                }
                (Some(u), Some(_)) => {
                    let page = base + u;
                    next_u = self.next_uncopied(u + 1, hi);
                    while wi < written.len() && written[wi].0 == page {
                        wi += 1;
                    }
                    (u, 0)
                }
                (Some(u), None) => {
                    next_u = self.next_uncopied(u + 1, hi);
                    (u, 0)
                }
                (None, Some(w)) => {
                    let page = base + w;
                    while wi < written.len() && written[wi].0 == page {
                        wi += 1;
                    }
                    (w, self.at[w])
                }
                (None, None) => break,
            };
            let page = base + idx;
            if (page * PAGE).max(start) >= from && ((page + 1) * PAGE).min(end) <= to {
                self.set(idx, now);
            }
            let again = taken != 0;
            if again {
                again_at.push((page, taken));
            }
            push_page(&mut runs, &mut run, page, again);
        }
        // Cut to the range: next to it shaders may be writing the copy.
        if let Some((run_start, run_end, known)) = run {
            runs.push(((run_start * PAGE).max(from), (run_end * PAGE).min(to), known));
        }
        (runs, again_at)
    }

    /// Pages of `[a, e)` within this copy that were never taken or were written
    /// after they were last taken, in ascending page order.
    fn stale_pages_in(&self, base: usize, a: usize, e: usize, written: &[(usize, u64)]) -> Vec<usize> {
        let mut out = Vec::new();
        if a >= e {
            return out;
        }
        let first = (a / PAGE).max(base);
        let last_excl = e.div_ceil(PAGE).min(base.saturating_add(self.at.len()));
        if first >= last_excl {
            return out;
        }
        let last = last_excl - 1;
        let lo = first - base;
        let hi = last - base;
        let w_start = Self::written_start(written, first);
        if self.range_all_copied(lo, hi) {
            for &(page, write) in &written[w_start..] {
                if page > last {
                    break;
                }
                if write > self.at[page - base] && out.last() != Some(&page) {
                    out.push(page);
                }
            }
            return out;
        }
        let mut wi = w_start;
        let mut next_u = self.next_uncopied(lo, hi);
        loop {
            while wi < written.len() && written[wi].0 <= last {
                let (wp, write) = written[wi];
                let idx = wp - base;
                if self.at[idx] == 0 || write > self.at[idx] {
                    break;
                }
                wi += 1;
            }
            let next_w = (wi < written.len() && written[wi].0 <= last).then(|| written[wi].0 - base);
            let idx = match (next_u, next_w) {
                (Some(u), Some(w)) if u < w => {
                    next_u = self.next_uncopied(u + 1, hi);
                    u
                }
                (Some(u), Some(w)) if w < u => {
                    let page = base + w;
                    while wi < written.len() && written[wi].0 == page {
                        wi += 1;
                    }
                    w
                }
                (Some(u), Some(_)) => {
                    let page = base + u;
                    next_u = self.next_uncopied(u + 1, hi);
                    while wi < written.len() && written[wi].0 == page {
                        wi += 1;
                    }
                    u
                }
                (Some(u), None) => {
                    next_u = self.next_uncopied(u + 1, hi);
                    u
                }
                (None, Some(w)) => {
                    let page = base + w;
                    while wi < written.len() && written[wi].0 == page {
                        wi += 1;
                    }
                    w
                }
                (None, None) => break,
            };
            out.push(base + idx);
        }
        out
    }

    /// Allocation-free check for whether [`Self::stale_pages_in`] is non-empty.
    fn has_stale_in(&self, base: usize, a: usize, e: usize, written: &[(usize, u64)]) -> bool {
        if a >= e {
            return false;
        }
        let first = (a / PAGE).max(base);
        let last_excl = e.div_ceil(PAGE).min(base.saturating_add(self.at.len()));
        if first >= last_excl {
            return false;
        }
        let last = last_excl - 1;
        let lo = first - base;
        let hi = last - base;
        if !self.range_all_copied(lo, hi) {
            return true;
        }
        let w_start = Self::written_start(written, first);
        for &(page, write) in &written[w_start..] {
            if page > last {
                break;
            }
            if write > self.at[page - base] {
                return true;
            }
        }
        false
    }

    #[cfg(any(test, feature = "diagnostics"))]
    fn copied_count(&self) -> usize {
        self.at.len() - self.uncopied
    }
}

/// A title buffer's host copy.
pub(crate) struct Shadow {
    /// Title memory's pages of 4 KiB (from the one `start` lies in on), a bit
    /// each, that what shaders wrote was put into since a refresh last looked
    /// at them ([`Shadow::take_landed`]); empty until then.
    landings: Mutex<Vec<u64>>,
    /// Whether any is.
    landing: AtomicBool,
    /// The bytes of `buffer`, once asked.
    width: std::sync::OnceLock<u64>,
    pub(crate) buffer: ID3D12Resource,
    data: *mut u8,
    gpu: u64,
    /// Where the title buffer starts and how long it is.
    start: u64,
    len: u64,
    /// Per page of title memory from `start`'s on: the write-watch asking
    /// after which it was last copied (0: never), indexed with a 64-page
    /// uncopied bitmask for O(dirty) `copy_stale`.
    copied: Mutex<CopiedPages>,
    /// Two ranges of title memory taken whole, the last one and the longest,
    /// and the asking after which each was ([`Shadow::copy_stale`]). A page
    /// a range holds in part cannot say so in `copied`. Locked inside
    /// `copied`'s lock.
    ranged: Mutex<[(usize, usize, u64); 2]>,
    /// Whether the memory is write-watched: unknown, yes, no.
    watched: AtomicU8,
    /// Ranges shaders of a submission not yet copied back may be writing:
    /// title memory there is older than the copy.
    pending: Mutex<Vec<(usize, usize)>>,
    /// Lock-free mirror of `!pending.is_empty()` so read-only windows and
    /// buffers skip acquiring `pending` on refreshes and flight checks.
    has_pending: AtomicBool,
    /// With [`relay`]: how many submissions not yet copied back declared
    /// each range of `pending` ([`Flights`]). Locked inside `pending`'s
    /// lock.
    flights: Mutex<Flights>,
    /// The copy's bytes before the submission ran, for the ranges its
    /// shaders write (`pending`), each at its offset in the copy: what the
    /// GPU wrote differs. Room for the whole copy; only those ranges' bytes
    /// are set (and read).
    before: Mutex<Option<Vec<u8>>>,
    /// Held shared while pieces of the copy are landed (`crate::land`), each
    /// its own bytes of it on several threads, and alone while title bytes
    /// go in under shader writes ([`Shadow::merge_under_writes`]), which
    /// takes `before`'s lock too. Taken ahead of that one.
    merging: RwLock<()>,
    /// With snapshots kept ([`keeps`]): which 64 KiB pieces of the snapshot
    /// hold the copy's bytes as they are now, and whether the snapshot is
    /// counted in [`KEPT_BYTES`].
    kept: Mutex<(Vec<bool>, bool)>,
    /// When the copy was last asked for (`crate::host::moment`): a window
    /// outside buffers nothing asked for in a while, and nothing holds, is
    /// let go ([`let_go_idle`]).
    used: AtomicU64,
    /// The lowest and highest title address asked for in the copy (a
    /// window is 4 MiB whatever was asked: measured for the memory line).
    asked: [AtomicU64; 2],
    /// Asks for the copy that gave no length (root views: they read from
    /// their address to the copy's end, whatever lies there).
    unsized_asks: AtomicU64,
    /// Whether shaders were ever declared to write it.
    written: AtomicBool,
    /// A number no other copy has had. Its address is no name for a copy:
    /// one let go and another made may share it.
    id: u64,
    /// With [`cards`], for a big title buffer: `buffer` is in the card's
    /// own memory, and this is the rest of the copy.
    card: Option<Card>,
    /// A window of title memory outside buffers, not a title buffer's copy.
    window: bool,
    /// Of a copy with two sides: only the dwords the title changed go up,
    /// as with [`send_changed`] ([`by_address`]).
    changes_only: AtomicBool,
    /// A window in card memory the CPU maps ([`window_mapped_for`]).
    mapped: bool,
    /// With `crate::inplace`: not a copy at all. `buffer` is placed in a
    /// heap opened on the title memory itself, so shaders read and write
    /// the title's own bytes and nothing is copied either way. The heap,
    /// and the bytes from the buffer's first byte to `start` (the heap
    /// begins on a 64 KiB line; a title buffer need not).
    place: Option<(windows::Win32::Graphics::Direct3D12::ID3D12Heap, u64)>,
}

/// What a copy in the card's memory ([`cards`]) has beside the buffer
/// shaders see.
struct Card {
    /// The CPU's side, in the PC's memory: what `data` maps. Title memory
    /// is copied into it as into any copy, and what changed in it goes to
    /// the card's side before the next submission.
    staging: ID3D12Resource,
    /// The card side's bytes as they were when the submission's shaders
    /// started, in the card's memory too: made when shader writes are
    /// first declared for the copy, and compared with the card's side, on
    /// the card, after each submission that declares some.
    before: Mutex<Option<ID3D12Resource>>,
    /// Offsets of the copy whose bytes changed on the CPU's side and have
    /// not gone to the card's side yet.
    dirty: Mutex<Vec<(usize, usize)>>,
    /// Lock-free mirror of `!dirty.is_empty()`; cleared only after `sent` and
    /// `has_sent` have been updated so concurrent `card_merge_relay` and
    /// `land_card` readers observing `has_dirty == false` are guaranteed to
    /// observe `has_sent == true`.
    has_dirty: AtomicBool,
    /// With [`relay`]: the offsets that went from `dirty` to the card's
    /// side, each with the number of the send that took it
    /// ([`card_dirty`]), kept until a fetch sent after it has been merged.
    /// Locked inside `dirty`'s lock.
    sent: Mutex<SentRuns>,
    /// Lock-free mirror of `!sent.is_empty()` so clean carded copies skip
    /// locking `sent` on every submission and landing.
    has_sent: AtomicBool,
    /// The buffers' bytes (the title buffer's, rounded up to 64 KiB).
    size: u64,
}

/// Whether the host copies of big title buffers are in the card's own
/// memory (on by default since run 75 showed it holds: the BVH skinning
/// list went from 50.5 to 4.5 ms on the card; `VICTORY_CARD_BUFFERS=0`
/// keeps every copy in the PC's memory, as before. Read when a copy is
/// made, so it takes a fresh start).
///
/// Every host copy was a buffer in the PC's memory that the card reads
/// and writes over the bus, so that our code can fill it and compare it.
/// The title's shaders read almost all their buffer data from a few big
/// title buffers (in the gameplay frame of run 69, 190-330 MB of distinct
/// ranges of the 174 MB heap per list against 0-13 MB elsewhere), and a
/// skinning dispatch of ONE group of 32 threads took 1.8 ms on the card
/// (run 70): the time of some 170 loads one after the other, not of the
/// arithmetic.
///
/// With the switch, a title buffer of [`card_limits`]' size or more has two
/// sides. Shaders see `Shadow::buffer`, in the card's memory. The CPU's
/// side (`Card::staging`, what `Shadow::data` maps) takes title memory as
/// before, and the ranges that changed in it are copied to the card's side
/// before each submission (`Host::card_before`). What shaders wrote is
/// found on the card: after the submission's lists a compute pass of ours
/// compares each declared range, by 64 KiB piece, with the card side's
/// bytes from before (`Card::before`), and only the pieces that differ are
/// fetched and merged into title memory, byte by byte as
/// [`Shadow::copy_back`] does (`Host::card_back`, [`Shadow::card_merge`]).
/// No snapshot is taken and nothing is compared on the CPU for such a copy.
///
/// What it assumes is what [`keeps`] assumes: shaders write a copy only in
/// ranges declared as written. A write outside them is seen by later
/// shaders, as before, and goes back to the title the next time a declared
/// range holds its piece (before: never).
pub(crate) fn cards() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CARD_BUFFERS").is_none_or(|v| v != "0"))
}

/// Default `(min_mb, max_mb)` for [`card_limits`] chosen from the adapter's
/// local VRAM budget (`DXGI_QUERY_VIDEO_MEMORY_INFO.Budget`), in bytes.
///
/// - Unknown (`None`, e.g. WARP) or `<= 12 GiB` (8–12 GB cards, whose WDDM
///   budget is below 12 GiB): `(16, 512)` MiB, matching the 12 GB baseline.
/// - `> 12 GiB` and `< 20 GiB` (16 GB cards, whose WDDM budget is typically
///   ~13.5–15.2 GiB): `(4, 2048)` MiB.
/// - `>= 20 GiB` (24–32+ GB cards): `(1, (budget_mb / 4).min(8192))` MiB,
///   capping at 25% of local budget (at most 8 GiB) so "before" twins (up to 2x)
///   still leave headroom for textures and render targets.
fn default_card_limits_mb(local_vram_budget: Option<u64>) -> (u64, u64) {
    let Some(budget_bytes) = local_vram_budget else {
        return (16, 512);
    };
    let budget_mb = budget_bytes >> 20;
    const TWELVE_GIB_MB: u64 = 12 * 1024;
    const TWENTY_GIB_MB: u64 = 20 * 1024;
    if budget_mb <= TWELVE_GIB_MB {
        (16, 512)
    } else if budget_mb < TWENTY_GIB_MB {
        (4, 2048)
    } else {
        (1, (budget_mb / 4).min(8192))
    }
}

/// Resolve `(min_bytes, max_bytes)` from the adapter's local VRAM budget and
/// optional explicit MiB overrides from `VICTORY_CARD_BUFFERS_MIN_MB` /
/// `VICTORY_CARD_BUFFERS_MB`.
fn resolve_card_limits(
    local_vram_budget: Option<u64>,
    min_mb_override: Option<u64>,
    max_mb_override: Option<u64>,
) -> (u64, u64) {
    const MIB: u64 = 1 << 20;
    let (default_min_mb, default_max_mb) = default_card_limits_mb(local_vram_budget);
    let min_bytes = min_mb_override.unwrap_or(default_min_mb).saturating_mul(MIB).max(MIB);
    let max_bytes = max_mb_override.unwrap_or(default_max_mb).saturating_mul(MIB);
    (min_bytes, max_bytes)
}

/// The smallest title buffer whose copy goes to the card's memory, and the
/// most bytes of such copies (their "before" twins not counted: a copy
/// shaders write takes its size twice), both in bytes:
/// `VICTORY_CARD_BUFFERS_MIN_MB` and `VICTORY_CARD_BUFFERS_MB`.
/// Defaults scale with the adapter's local VRAM budget (16 MiB / 512 MiB on
/// `<= 12 GiB` budgets, 4 MiB / 2048 MiB on 16 GB cards, 1 MiB / up to 8192 MiB
/// on `>= 20 GiB` budgets), and explicit env vars always override.
fn card_limits() -> (u64, u64) {
    static LIMITS: std::sync::OnceLock<(u64, u64)> = std::sync::OnceLock::new();
    *LIMITS.get_or_init(|| {
        let vram_budget = crate::host::host().and_then(|h| h.video_memory()).map(|[budget, ..]| budget);
        let (mut min_bytes, max_bytes) = resolve_card_limits(
            vram_budget,
            std::env::var("VICTORY_CARD_BUFFERS_MIN_MB").ok().as_deref().and_then(parse_mib),
            explicit_card_buffers_mb(),
        );
        // `VICTORY_CARD_BUFFERS_MIN_KB` (after fxceless c36971b2) says the
        // least size in KiB, so that it can go under 1 MiB: 64 is every title
        // buffer. Copies under 1 MiB are nearly all there are and few bytes;
        // in the PC's memory every load, store and atomic a shader does on
        // them crosses the bus.
        if let Some(kib) = std::env::var("VICTORY_CARD_BUFFERS_MIN_KB").ok().and_then(|v| v.trim().parse::<u64>().ok())
            && kib > 0
        {
            min_bytes = kib << 10;
        }
        if card_all() {
            min_bytes = 0;
        }
        tracing::info!(
            vram_budget_mb = vram_budget.map(|b| b >> 20),
            card_buffers_min_kb = min_bytes >> 10,
            card_buffers_max_mb = max_bytes >> 20,
            "card buffer limits"
        );
        (min_bytes, max_bytes)
    })
}

/// Use the same parsed override for allocation and reservation.
fn explicit_card_buffers_mb() -> Option<u64> {
    static MB: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *MB.get_or_init(|| std::env::var("VICTORY_CARD_BUFFERS_MB").ok().as_deref().and_then(parse_mib))
}

fn parse_mib(value: &str) -> Option<u64> {
    value.trim().parse().ok()
}

/// Bytes alive of copies' card sides, and of their "before" twins.
static CARD_BYTES: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Bytes of title-memory-by-address (RT BVH) arenas placed in local VRAM.
static BY_ADDRESS_CARD_BYTES: AtomicU64 = AtomicU64::new(0);

/// Local-VRAM buffer breakdown for the HUD: `[card_buffers_and_windows, bvh_arenas]`.
#[cfg(feature = "diagnostics")]
pub(crate) fn vram_breakdown() -> [u64; 2] {
    let bvh = BY_ADDRESS_CARD_BYTES.load(Ordering::Relaxed);
    let win = WINDOW_CARD_BYTES.load(Ordering::Relaxed).saturating_sub(bvh);
    let buf = CARD_BYTES[0].load(Ordering::Relaxed)
        + CARD_BYTES[1].load(Ordering::Relaxed)
        + win
        + WINDOW_MAPPED_BYTES.load(Ordering::Relaxed);
    [buf, bvh]
}

/// Whether the texture queue leaves room on the card for the card copies
/// of title buffers that are not made yet. The reservation is enabled by
/// default. `VICTORY_CARD_ROOM_FOR_BUFFERS=0` disables it.
///
/// The copies are made in one burst when the world loads (run 189: 94 of
/// its 95 within two seconds, right before the first world frame). Without
/// the reservation, [`card_for`] allows copies up to 80% of the reported
/// local budget. That held as long as the textures picked by index filled the
/// card after the burst. Since the texture queue keeps packed neighbours
/// (52b8060) it fills the card while the world loads: 85% and more at the
/// burst in runs 190 to 192, so one copy of 95 was made, the CPU looked
/// for shader writes in 600 to 720 MB a frame where it had been 30, and
/// the frame went from about 240 to 370-470 ms.
///
/// With the switch, the texture queue's rule (`resource::room`) counts the
/// part of [`card_limits`]' second number that no copy has taken yet as
/// used. After the burst, only the unused part of that limit is reserved
/// (87 MB of 1,024 in run 189).
///
/// On by default from the build after run 199
/// (`VICTORY_CARD_ROOM_FOR_BUFFERS=0`: off, as before). Run 193 had it on and still made 31 copies of 95: the copies
/// and the world's render targets (+1,603 MB) arrive in the same first
/// world frame and raced for the 448 MB left under the line, and a copy
/// turned down is not asked for again (`lookup` asks once), while its room
/// went on being kept (616 MB to the end of the run). So now: the texture
/// queue also keeps room for the targets (`resource::targets_room_kept`),
/// a copy is made up to 97 hundredths of the budget (its room was kept),
/// and the room of a copy turned down is given up ([`CARD_REFUSED`]).
fn card_room_kept_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CARD_ROOM_FOR_BUFFERS").is_none_or(|v| v != "0"))
}

/// The 4070 profile opts into reserving the configured buffer and window budgets.
fn reserve_card_budgets() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CARD_RESERVE_BUDGETS").is_some_and(|v| v == "1"))
}

/// Title buffers big enough for a copy in the card's memory that were
/// turned down for want of room, since the start, and their bytes: their
/// copies stay in the PC's memory for as long as the title keeps the
/// buffers, so no room is kept for them any more.
static CARD_REFUSED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// The bytes of the card's memory the texture queue leaves free for card
/// copies of title buffers not made yet ([`card_room_kept_on`]): 0 with
/// the switch off or without card copies.
pub(crate) fn card_room_kept() -> u64 {
    if !card_room_kept_on() || !cards() {
        return 0;
    }
    // Not more than such copies came to in any session, whatever they may
    // come to: on a card of 24 GB a quarter of its budget was kept for them,
    // 5 GB that never came (941 MB in 97 copies after fifteen minutes), and
    // no texture was made any more with 5 GB of the card free (session
    // play-1125-0127: 1,453 passes refused, the ground and people black).
    let (buffers, windows) = reservation_limits(
        card_limits().1, explicit_card_buffers_mb().is_some(), window_card_most(), reserve_card_budgets(),
    );
    remaining_card_room(
        buffers,
        CARD_BYTES[0].load(Ordering::Relaxed),
        CARD_REFUSED[1].load(Ordering::Relaxed),
        windows,
        WINDOW_CARD_BYTES.load(Ordering::Relaxed),
    )
}

fn reservation_limits(buffers: u64, explicit_buffers: bool, windows: u64, reserve: bool) -> (u64, u64) {
    const SEEN: u64 = 1 << 30;
    if reserve {
        (if explicit_buffers { buffers } else { buffers.min(SEEN) }, windows)
    } else {
        (buffers.min(SEEN), 0)
    }
}

fn remaining_card_room(buffers: u64, used: u64, refused: u64, windows: u64, window_used: u64) -> u64 {
    buffers.saturating_sub(used.saturating_add(refused))
        .saturating_add(windows.saturating_sub(window_used))
}

/// The copies in the card's memory.
static CARDED: Mutex<Vec<std::sync::Weak<Shadow>>> = Mutex::new(Vec::new());

/// Since the last frame line: bytes copied to card sides and the ranges
/// they were; pieces compared on the card, pieces fetched and their
/// bytes; the waits for the card the fetches added and their microseconds.
pub(crate) static CARD_STATS: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];

pub(crate) fn live_span_bytes() -> u64 {
    CARD_BYTES[0].load(Ordering::Relaxed)
        + WINDOW_CARD_BYTES.load(Ordering::Relaxed)
        + WINDOW_MAPPED_BYTES.load(Ordering::Relaxed)
}

/// [`CARD_STATS`], taken; then the copies in the card's memory, their
/// bytes and their "before" twins' bytes.
pub(crate) fn card_stats() -> [u64; 10] {
    let mut out = [0u64; 10];
    for (i, stat) in CARD_STATS.iter().enumerate() {
        out[i] = stat.swap(0, Ordering::Relaxed);
    }
    out[7] = CARDED.lock().unwrap_or_else(|e| e.into_inner()).iter().filter(|w| w.strong_count() > 0).count() as u64;
    out[8] = CARD_BYTES[0].load(Ordering::Relaxed)
        + WINDOW_CARD_BYTES.load(Ordering::Relaxed)
        + WINDOW_MAPPED_BYTES.load(Ordering::Relaxed);
    out[9] = CARD_BYTES[1].load(Ordering::Relaxed);
    out
}

/// The bytes of the hashes kept on the card for a copy of `size` bytes: 16
/// for each 64 KiB piece (a copy that does not start on a piece boundary
/// has one piece more).
pub(crate) fn card_hash_bytes(size: u64) -> u64 {
    (size / PIECE as u64 + 2) * 16
}

/// A buffer of `size` bytes in the card's memory for a title buffer's
/// copy, if the copy is to be there: [`cards`] is on, the size is within
/// [`card_limits`] and below 1 GiB (the compare counts in 32 bits), and the
/// host has what it takes (`Host::card_ready`).
fn card_for(size: u64) -> Option<ID3D12Resource> {
    let there = card_room_for(size);
    if there.is_some() || !card_all() || size >= 1 << 30 {
        return there;
    }
    // Two sides all the same, the card's in the PC's memory.
    let host = host()?;
    if !cards() || !host.card_ready() || !crate::residency::allow_optional_buffer_growth(size) {
        return None;
    }
    let (buffer, _) = host.shared_buffer(size)?;
    CARD_BYTES[0].fetch_add(size, Ordering::Relaxed);
    Some(buffer)
}

/// Whether every title buffer's copy has two sides, however small
/// (`VICTORY_CARD_ALL=1`; off by default).
///
/// A copy with one side, in the PC's memory, is what the card reads while
/// our refreshes write it: a submission that is prepared changes it under
/// the ones in the air ahead of it. The title's CPU does not write what they
/// still read, but its own buffer copies are done by us at preparing
/// (`crate::host::send` has the order of those right for copies with two
/// sides), and so are the bytes put in beside shaders' writes.
fn card_all() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CARD_ALL").is_some_and(|v| v == "1"))
}

/// [`card_for`] in the card's own memory.
fn card_room_for(size: u64) -> Option<ID3D12Resource> {
    if !cards() {
        return None;
    }
    let (least, most) = card_limits();
    if size < least || size >= 1 << 30 {
        return None;
    }
    let host = host()?;
    if !host.card_ready() {
        return None;
    }
    // Copies are made when the world loads, before the textures fill the
    // card. The texture queue's own rule (`resource::room`) counts
    // them, so the card's use stays where it was and the textures' share
    // shrinks by the copies' bytes, at most `card_limits`' second number.
    // With room reserved for buffers, local usage including the new copy
    // may reach 97% of the reported budget. Disabling the reservation uses 80%.
    let hundredths = if card_room_kept_on() { 97 } else { 80 };
    if let Some([budget, used, ..]) = host.video_memory()
        && used + size > budget / 100 * hundredths
    {
        // Said, with the numbers it was decided by: run 193 made 31 copies
        // of 95 and no line named the others. The first 64, then every
        // 32nd.
        let n = CARD_REFUSED[0].fetch_add(1, Ordering::Relaxed) + 1;
        let bytes = CARD_REFUSED[1].fetch_add(size, Ordering::Relaxed) + size;
        if n <= 64 || n.is_multiple_of(32) {
            tracing::info!(
                "no card copy for a title buffer of {} KiB: the card has {} MB in use of a budget of {} MB and a copy is made up to \
                 {} MB; the copies made hold {} MB of at most {} MB; {} buffers turned down so far ({} MB), frame {}",
                size >> 10,
                used >> 20,
                budget >> 20,
                (budget / 100 * hundredths) >> 20,
                CARD_BYTES[0].load(Ordering::Relaxed) >> 20,
                most >> 20,
                n,
                bytes >> 20,
                crate::frame::current()
            );
        }
        return None;
    }
    if CARD_BYTES[0].fetch_add(size, Ordering::Relaxed) + size > most {
        CARD_BYTES[0].fetch_sub(size, Ordering::Relaxed);
        return None;
    }
    let buffer = host.uav_buffer(size);
    if buffer.is_none() {
        CARD_BYTES[0].fetch_sub(size, Ordering::Relaxed);
    }
    buffer
}

/// The most bytes of windows outside buffers whose copies go to the card's
/// memory (`VICTORY_CARD_WINDOWS_MB`; 0, the default, none).
///
/// Windows are two thirds of the PC memory the card shares (run 104: 861
/// windows, 3,552 MB of 5,400), and every draw that takes vertices, indices
/// or constants from the title's own allocator reads them over the bus.
/// Shaders write none of them (run 91: "0.0 MB compared in 0 ranges of
/// windows"), so a window on the card needs only what a card copy has
/// already: the CPU's side takes title memory as before, and what changed
/// in it goes up before each submission. No hashes, no compare, no fetch,
/// unless a shader does declare a write to it one day.
///
/// A window goes to the card while there is room under this number and
/// the occupancy limit (80% by default, 97% with its budget reserved).
/// Textures keep their room, and once they have filled the
/// card later windows stay where they are. The CPU's side stays in the PC's
/// memory, so the memory shared with the card does not shrink by this
/// alone; what moves is what shaders read.
fn window_card_most() -> u64 {
    static MOST: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MOST.get_or_init(|| {
        std::env::var("VICTORY_CARD_WINDOWS_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(0) << 20
    })
}

/// Bytes alive of windows' card sides ([`window_card_most`]).
static WINDOW_CARD_BYTES: AtomicU64 = AtomicU64::new(0);

/// A buffer in the card's memory for a window of `size` bytes, if it is to
/// be there ([`window_card_most`]).
fn window_card_for(size: u64) -> Option<ID3D12Resource> {
    let there = window_room_for(size);
    if there.is_some() || !windows_own() || !cards() {
        return there;
    }
    // Two sides all the same, the card's in the PC's memory.
    let host = host()?;
    if !host.card_ready() || !crate::residency::allow_optional_buffer_growth(size) {
        return None;
    }
    let (buffer, _) = host.shared_buffer(size)?;
    WINDOW_CARD_BYTES.fetch_add(size, Ordering::Relaxed);
    Some(buffer)
}

/// The most bytes of windows outside buffers that lie in the card's memory
/// and are filled by the CPU through the bus (`VICTORY_WINDOWS_BAR_MB`; 0, the
/// default, none): one buffer (`D3D12_HEAP_TYPE_GPU_UPLOAD`), no CPU's side, no
/// copy to send and no list of changes, unlike [`window_card_most`]. The CPU
/// only writes them, as no shader is declared to write a window; if one were,
/// its snapshot and compare would read through the bus, right but slow. In a
/// captured frame the lists take 26 ms of the card instead of 29, the frame 32
/// instead of 36; filling them costs the CPU what it did.
///
/// Windows go there while a fifth of the card's budget is free, as with
/// [`window_room_for`], which this comes before.
fn window_mapped_most() -> u64 {
    static MOST: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MOST.get_or_init(|| {
        std::env::var("VICTORY_WINDOWS_BAR_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(0) << 20
    })
}

/// Bytes alive of windows in card memory the CPU maps.
static WINDOW_MAPPED_BYTES: AtomicU64 = AtomicU64::new(0);

/// A buffer for a window of `size` bytes in card memory the CPU maps, if it is
/// to be there ([`window_mapped_most`]).
fn window_mapped_for(size: u64) -> Option<(ID3D12Resource, *mut u8)> {
    let most = window_mapped_most();
    if most == 0 {
        return None;
    }
    let host = host()?;
    if let Some([budget, used, ..]) = host.video_memory()
        && used + size > budget / 10 * 8
    {
        return None;
    }
    if WINDOW_MAPPED_BYTES.fetch_add(size, Ordering::Relaxed) + size > most {
        WINDOW_MAPPED_BYTES.fetch_sub(size, Ordering::Relaxed);
        return None;
    }
    let made = host.mapped_card_buffer(size, true);
    if made.is_none() {
        WINDOW_MAPPED_BYTES.fetch_sub(size, Ordering::Relaxed);
    }
    made
}

/// Whether windows outside buffers have two sides too, so that submissions
/// in the air read the bytes they were prepared with (`VICTORY_WINDOWS_OWN=1`;
/// off by default: it takes the windows' bytes again, 3.3 GB in capture
/// replay2, and what changes in them under submissions in the air is what the
/// title's CPU writes, which it does not while they read it).
fn windows_own() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WINDOWS_OWN").is_some_and(|v| v == "1"))
}

/// [`window_card_for`] in the card's own memory.
fn window_room_for(size: u64) -> Option<ID3D12Resource> {
    let most = window_card_most();
    if most == 0 || !cards() {
        return None;
    }
    let host = host()?;
    if !host.card_ready() {
        return None;
    }
    if let Some([budget, used, ..]) = host.video_memory()
        && used.saturating_add(size) > window_usage_limit(budget, reserve_card_budgets() && card_room_kept_on())
    {
        return None;
    }
    if WINDOW_CARD_BYTES.fetch_add(size, Ordering::Relaxed) + size > most {
        WINDOW_CARD_BYTES.fetch_sub(size, Ordering::Relaxed);
        return None;
    }
    let buffer = host.uav_buffer(size);
    if buffer.is_none() {
        WINDOW_CARD_BYTES.fetch_sub(size, Ordering::Relaxed);
    }
    buffer
}

fn window_usage_limit(budget: u64, reserved: bool) -> u64 {
    if reserved { budget / 100 * 97 } else { budget / 10 * 8 }
}

/// What went up, by send: its number, and the offsets, in order and joined
/// (a refresh sends thousands of runs of a few words).
#[derive(Default)]
struct SentRuns(Vec<(u64, Vec<(usize, usize)>)>);

impl SentRuns {
    #[cfg(test)]
    fn note(&mut self, number: u64, ranges: Vec<(usize, usize)>) {
        self.note_joined(number, joined(ranges));
    }

    /// `ranges` are in order and joined.
    fn note_joined(&mut self, number: u64, ranges: Vec<(usize, usize)>) {
        if ranges.is_empty() {
            return;
        }
        match self.0.iter_mut().find(|s| s.0 == number) {
            Some((_, all)) => *all = joined([std::mem::take(all), ranges].concat()),
            None => self.0.push((number, ranges)),
        }
    }

    /// Sends up to `number` are of no use any more.
    fn forget(&mut self, number: u64) {
        self.0.retain(|s| s.0 > number);
    }

    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Adds to `out` what sends after `number` took of `[from, to)`.
    fn near(&self, number: u64, from: usize, to: usize, out: &mut Vec<(usize, usize)>) {
        for (_, ranges) in self.0.iter().filter(|s| s.0 > number) {
            let first = ranges.partition_point(|r| r.1 <= from);
            out.extend(ranges[first..].iter().take_while(|r| r.0 < to).copied());
        }
    }

    #[cfg(test)]
    fn of(all: &[(usize, usize, u64)]) -> SentRuns {
        let mut sent = SentRuns::default();
        for &(a, b, number) in all {
            sent.note(number, vec![(a, b)]);
        }
        sent
    }
}

/// The copies in the card's memory whose CPU side changed since they were
/// last asked: each with the changed offsets, in order and joined where
/// they touch. The ranges are taken: the caller copies them to the card's
/// side (`Host::card_before`).
///
/// With [`relay`], `sent_as` is the number of the send that takes them
/// (`crate::host`'s order of the queue): each range is noted with it, in
/// the same lock that took it, so that a merge of pieces fetched before
/// that send knows the card did not hold these bytes yet
/// ([`Shadow::card_merge_relay`]). Notes of sends that every fetch still to
/// be merged came after are dropped.
pub(crate) fn card_dirty(sent_as: u64) -> Vec<(Arc<Shadow>, Vec<(usize, usize)>)> {
    let mut out = Vec::new();
    let mut carded = CARDED.lock().unwrap_or_else(|e| e.into_inner());
    let fetched = RELAY_FETCHED.load(Ordering::Acquire);
    let relay_on = relay();
    carded.retain(|weak| {
        let Some(shadow) = weak.upgrade() else {
            return false;
        };
        let Some(c) = shadow.card.as_ref() else {
            return true;
        };
        if !c.has_dirty.load(Ordering::Acquire) {
            return true;
        }
        let ranges = {
            let mut dirty = c.dirty.lock().unwrap_or_else(|e| e.into_inner());
            let ranges = joined(std::mem::take(&mut *dirty));
            if relay_on {
                let mut sent = c.sent.lock().unwrap_or_else(|e| e.into_inner());
                sent.forget(fetched);
                sent.note_joined(sent_as, ranges.clone());
                // Publish the sent ranges before clearing the dirty state.
                c.has_sent.store(!sent.is_empty(), Ordering::Release);
            }
            c.has_dirty.store(false, Ordering::Release);
            ranges
        };
        if !ranges.is_empty() {
            out.push((shadow, ranges));
        }
        true
    });
    out
}

/// Diagnostics (`crate::host::upcheck`, nothing in the air): copies whose
/// pages the write watch and the guards hold for unchanged since they were
/// taken, compared with title memory, `most` bytes a call in turn. `say` gets
/// the copy, the title address of a page that is not the same, the words that
/// differ and the first of them (its address, the copy's value, title
/// memory's). Returns the bytes compared.
pub(crate) fn stale_pages(most: usize, say: &mut dyn FnMut(&Shadow, usize, usize, (usize, u32, u32))) -> usize {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let mut all: Vec<Arc<Shadow>> = buffers_read().values().filter_map(|e| e.shadow.clone()).collect();
    all.extend(LOOSE.read().unwrap_or_else(|e| e.into_inner()).values().cloned());
    all.retain(|s| s.place.is_none() && s.watched.load(Ordering::Relaxed) == 1);
    all.sort_by_key(|s| s.id);
    let next = NEXT.load(Ordering::Relaxed);
    let turn = all.partition_point(|s| s.id < next);
    let mut done = 0;
    let mut title = vec![0u8; PAGE];
    for shadow in all[turn..].iter().chain(&all[..turn]) {
        if done >= most {
            NEXT.store(shadow.id, Ordering::Relaxed);
            break;
        }
        let (start, end) = (shadow.start as usize, (shadow.start + shadow.len) as usize);
        // (What it finds written stays noted for the next refresh.)
        let Some((_, written)) = crate::watch::writes(start, end, true) else {
            continue;
        };
        let written: std::collections::HashMap<usize, u64> = written.into_iter().collect();
        let copied = shadow.copied.lock().unwrap_or_else(|e| e.into_inner()).at.clone();
        for (i, &at) in copied.iter().enumerate() {
            let page = start / PAGE + i;
            if at == 0 || written.get(&page).is_some_and(|&w| w > at) {
                continue;
            }
            let (lo, hi) = ((page * PAGE).max(start), ((page + 1) * PAGE).min(end));
            let title = &mut title[..hi - lo];
            if !crate::host::read_into(lo as *const u8, title) {
                continue;
            }
            done += hi - lo;
            // SAFETY: inside the mapping; nothing is in the air and the
            // submission lock is held.
            let held = unsafe { std::slice::from_raw_parts(shadow.data.add(lo - start).cast_const(), hi - lo) };
            if held == title {
                continue;
            }
            let word = |b: &[u8], k: usize| u32::from_le_bytes([b[k], b[k + 1], b[k + 2], b[k + 3]]);
            let differ: Vec<usize> = (0..(hi - lo) & !3).step_by(4).filter(|&k| word(held, k) != word(title, k)).collect();
            // Written meanwhile: not stale, and the next refresh sees it.
            let again = crate::watch::writes(lo, hi, true).is_none_or(|(_, w)| w.iter().any(|&(p, w)| p == page && w > at));
            if let Some(&k) = differ.first()
                && !again
            {
                say(shadow, lo, differ.len(), (lo + k, word(held, k), word(title, k)));
            }
        }
    }
    done
}

/// Copies of memory (from, to, bytes: apart), on several threads if they are
/// many.
pub(crate) fn copy_all(copies: Vec<(usize, usize, usize)>) {
    let bytes: usize = copies.iter().map(|c| c.2).sum();
    if bytes > 4 << 20 {
        wide(copy_jobs(copies));
        return;
    }
    for (src, dst, len) in copies {
        // SAFETY: the caller's ranges.
        unsafe { std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, len) };
    }
}

/// A title buffer copy was done on the CPU into `[dst, dst + len)`: all of it
/// goes up to the copies in the card's memory that hold it, whether their
/// CPU side held the same before or not. Left out is what shaders in the air
/// may be writing (such a copy waits for them, or is done on the card).
pub(crate) fn send_copied(dst: usize, len: usize) {
    for shadow in carded_copies() {
        let (start, end) = (shadow.start as usize, (shadow.start + shadow.len) as usize);
        let (from, to) = (dst.max(start), (dst + len).min(end));
        if from >= to {
            continue;
        }
        if !shadow.has_pending.load(Ordering::Acquire) {
            shadow.copy(from, to);
            continue;
        }
        let busy: Vec<(usize, usize)> =
            shadow.pending.lock().unwrap_or_else(|e| e.into_inner()).iter().copied().filter(|&(a, b)| a < to && b > from).collect();
        for (a, b) in outside(from, to, busy) {
            shadow.copy(a, b);
        }
    }
}

/// The copies in the card's memory (diagnostics: `crate::host::upcheck`).
pub(crate) fn carded_copies() -> Vec<Arc<Shadow>> {
    CARDED.lock().unwrap_or_else(|e| e.into_inner()).iter().filter_map(std::sync::Weak::upgrade).collect()
}

/// Ranges in order, those that overlap or touch as one.
fn joined(mut ranges: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    if ranges.len() < 2 {
        return ranges;
    }
    // In order and apart already (what one stretch of them is): as they are,
    // without the sort and the pass that joins.
    if ranges.windows(2).all(|w| w[0].1 < w[1].0) {
        return ranges;
    }
    // (The sort that finds what is in order already: they come as a few
    // stretches of tens of thousands in order, put behind one another.)
    ranges.sort();
    let mut last = 0;
    for index in 1..ranges.len() {
        let (from, to) = ranges[index];
        if from <= ranges[last].1 {
            ranges[last].1 = ranges[last].1.max(to);
        } else {
            last += 1;
            ranges[last] = (from, to);
        }
    }
    ranges.truncate(last + 1);
    ranges
}

/// Runs of pieces to hash again (`Host::card_before`), those that overlap
/// or touch as one, in order: every piece of `runs` is in the result once
/// and no other piece is. Ranges sent up lie close together (514 a frame
/// for 87 MB in run 102), and each had a pass of its own over its pieces,
/// so a piece that held several ranges was hashed once for each.
pub(crate) fn joined_piece_runs(mut runs: Vec<CardSpan>) -> Vec<CardSpan> {
    runs.sort_unstable_by_key(|r| r.first);
    let mut out: Vec<CardSpan> = Vec::new();
    for run in runs {
        match out.last_mut() {
            Some(last) if run.first <= last.first + last.pieces => {
                last.pieces = last.pieces.max(run.first + run.pieces - last.first);
                last.to = last.to.max(run.to);
            }
            _ => out.push(run),
        }
    }
    out
}

/// A declared range of a copy in the card's memory, as the compare on the
/// card takes it: its offsets in the copy, its first 64 KiB piece (pieces
/// are cut at multiples of 64 KiB of title memory; piece 0 holds offset
/// 0), how many pieces hold it, and where the copy starts in piece 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CardSpan {
    pub(crate) from: u32,
    pub(crate) to: u32,
    pub(crate) first: u32,
    pub(crate) pieces: u32,
    pub(crate) skew: u32,
}

/// [`CardSpan`] for offsets `[from, to)` (`from < to`) of a copy of title
/// memory from `start` on.
fn card_span_of(start: usize, from: usize, to: usize) -> CardSpan {
    let pieces = pieces_of(start, from, to);
    CardSpan {
        from: from as u32,
        to: to as u32,
        first: pieces.start as u32,
        pieces: pieces.len() as u32,
        skew: (start % PIECE) as u32,
    }
}

/// The offsets of piece `i` of `span` that lie in the span, for a copy of
/// `len` bytes of title memory from `start` on: what is fetched and merged
/// of a piece the compare found differing.
fn card_piece_of(start: usize, len: usize, span: &CardSpan, i: u32) -> (usize, usize) {
    let (lo, hi) = piece_at(start, len, (span.first + i) as usize);
    (lo.max(span.from as usize), hi.min(span.to as usize))
}

/// The offsets [`Shadow::card_hashed`] gives, for a card's buffer of `size`
/// bytes for title memory from `start` on.
fn card_hashed_of(start: usize, size: usize, piece: usize) -> (usize, usize) {
    let (lo, hi) = piece_at(start, size, piece);
    (lo & !15, hi.next_multiple_of(16))
}

#[cfg(test)]
mod card_tests {
    use super::{CardSpan, PIECE, card_hashed_of, card_piece_of, card_span_of, joined, joined_piece_runs};

    /// Joined runs hold every piece of the runs given, once, and no other.
    #[test]
    fn piece_runs_join_without_losing_or_adding_pieces() {
        let run = |first: u32, pieces: u32| CardSpan {
            from: first * 65536,
            to: (first + pieces) * 65536,
            first,
            pieces,
            skew: 7,
        };
        let runs = vec![run(40, 1), run(3, 2), run(4, 1), run(5, 3), run(40, 1), run(9, 1), run(41, 2), run(100, 1)];
        let mut wanted: Vec<u32> = runs.iter().flat_map(|r| r.first..r.first + r.pieces).collect();
        wanted.sort_unstable();
        wanted.dedup();
        let joined = joined_piece_runs(runs);
        let got: Vec<u32> = joined.iter().flat_map(|r| r.first..r.first + r.pieces).collect();
        assert_eq!(got, wanted);
        assert_eq!(joined.iter().map(|r| (r.first, r.pieces)).collect::<Vec<_>>(), [(3, 5), (9, 1), (40, 3), (100, 1)]);
        assert!(joined.iter().all(|r| r.skew == 7));
        assert!(joined_piece_runs(Vec::new()).is_empty());
    }

    /// The offsets the compare's shader takes for a piece (`CARD_COMPARE`
    /// in host.rs), before it widens them to 16 bytes.
    fn shader_piece(span: &CardSpan, i: u32) -> (u32, u32) {
        let piece = span.first + i;
        let lo = piece * 65536;
        let lo = if lo > span.skew { lo - span.skew } else { 0 };
        let hi = (piece + 1) * 65536 - span.skew;
        (lo.max(span.from), hi.min(span.to))
    }

    /// The pieces of a span cover it without gaps, and the shader's
    /// arithmetic names the same bytes as ours.
    #[test]
    fn the_shader_and_the_merge_cut_the_same_pieces() {
        for start in [0usize, 4096, PIECE - 16, PIECE, 3 * PIECE + 0x1f00, 0xd_b790_df00] {
            let len = 5 * PIECE + 4096;
            for (from, to) in [(0, len), (0, 1), (16, PIECE), (PIECE - 4, PIECE + 4), (2 * PIECE, len), (12345, 4 * PIECE + 7)] {
                let span = card_span_of(start, from, to);
                let mut at = from;
                for i in 0..span.pieces {
                    let (lo, hi) = card_piece_of(start, len, &span, i);
                    assert_eq!(lo, at, "{start:#x} {from} {to} piece {i}");
                    assert!(hi > lo && hi - lo <= PIECE);
                    assert_eq!(shader_piece(&span, i), (lo as u32, hi as u32), "{start:#x} {from} {to} piece {i}");
                    at = hi;
                }
                assert_eq!(at, to, "{start:#x} {from} {to}");
            }
        }
    }

    /// Changed ranges are joined where they overlap or touch, and only there.
    /// What is fetched of a piece to hash it is what the shader hashes.
    #[test]
    fn a_piece_is_hashed_here_as_on_the_card() {
        for start in [0usize, 0x10_0000, 0xd_5edb_c780, 0x12_3456_7008] {
            for size in [1usize << 16, 5 << 16, 0x20_0000] {
                let span = card_span_of(start, 0, size);
                for i in 0..span.pieces {
                    // As `CARD_COMPARE` has it.
                    let (piece, skew, size) = (span.first + i, span.skew, size as u32);
                    let lo = (piece * 65536).saturating_sub(skew);
                    let hi = ((piece + 1) * 65536 - skew).min(size);
                    let shader = (lo & !15, (hi + 15) & !15);
                    let here = card_hashed_of(start, size as usize, piece as usize);
                    assert_eq!((here.0 as u32, here.1 as u32), shader, "start {start:#x} size {size:#x} piece {piece}");
                }
            }
        }
    }

    #[test]
    fn ranges_join_where_they_touch() {
        assert_eq!(joined(vec![(8, 12), (0, 4), (4, 6), (10, 20), (30, 31)]), [(0, 6), (8, 20), (30, 31)]);
        assert!(joined(Vec::new()).is_empty());
        // In order and apart: as they are. In order and touching, or out of
        // order: joined as before.
        assert_eq!(joined(vec![(0, 10), (20, 30), (40, 50)]), [(0, 10), (20, 30), (40, 50)]);
        assert_eq!(joined(vec![(0, 10), (10, 20), (30, 40)]), [(0, 20), (30, 40)]);
        assert_eq!(joined(vec![(0, 10), (5, 8), (30, 40)]), [(0, 10), (30, 40)]);
        assert_eq!(joined(vec![(20, 30), (0, 10), (5, 25)]), [(0, 30)]);
    }
}

/// Whether the copy-back of what shaders wrote runs on a thread of its own,
/// beside the submitting thread and outside the submission lock
/// (`VICTORY_RELAY=1` or `2`; off by default; not with kept snapshots,
/// [`keeps`]). `crate::host::relay` says what it does; here it changes
/// what a copy keeps:
///
/// - the ranges shaders may be writing are counted per submission
///   ([`Flights`]): a range two flights declared stays closed to refreshes
///   until both have been copied back;
/// - the merge of what the card wrote never stores into memory the card
///   may be writing ([`merge_piece`]): what it learns goes into title
///   memory and into the copy's reference (the snapshot, or the CPU's side
///   of a copy in the card's memory), under that reference's lock;
/// - what goes up to a copy in the card's memory is numbered
///   ([`card_dirty`]), so that a merge can tell the title's bytes the card
///   had not received from bytes the card wrote.
pub(crate) fn relay() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // On by default (stage 2, `crate::host::relay_exact`); `VICTORY_RELAY=0`
    // turns it off, `=1` asks for the first stage.
    // The crate's own tests run without it unless they ask: several of
    // them look at title memory right after a submission, which the
    // relay's landing thread may still be writing.
    *ON.get_or_init(|| std::env::var("VICTORY_RELAY").map_or(!cfg!(test), |v| v == "1" || v == "2") && !keeps())
}

/// The submission being prepared (its number among those started): what
/// [`Shadow::writing`] counts a declared range for ([`Flights`]).
static PREPARED: AtomicU64 = AtomicU64::new(0);

/// A submission's preparing begins (`crate::host`, under the submission
/// lock).
pub(crate) fn preparing(number: u64) {
    PREPARED.store(number, Ordering::Relaxed);
}

/// With [`relay`]: the number of the last send of changed ranges to the
/// card (`crate::host`'s order of the queue) that every fetch still to be
/// merged came after. Notes of sends up to it are of no use any more.
static RELAY_FETCHED: AtomicU64 = AtomicU64::new(0);

/// A fetch sent after send `number` is the oldest one that may still be
/// merged ([`RELAY_FETCHED`]).
pub(crate) fn relay_fetched(number: u64) {
    RELAY_FETCHED.fetch_max(number, Ordering::AcqRel);
}

/// With [`relay`]: how many submissions that are not copied back yet
/// declared each range shaders may be writing (range, count, the last
/// submission that declared it).
///
/// Without it a range declared by two flights was one entry, and the end
/// of the first flight's copy-back removed it while the second was still
/// writing: the next refresh would have put title bytes over what the
/// second flight's shaders were writing.
#[derive(Default)]
struct Flights(std::collections::HashMap<(usize, usize), (u32, u64)>);

impl Flights {
    /// Submission `number` declares `range`: counted once for it, however
    /// often it declares it.
    fn declare(&mut self, range: (usize, usize), number: u64) {
        let f = self.0.entry(range).or_insert((0, number));
        if f.0 == 0 || f.1 != number {
            *f = (f.0 + 1, number);
        }
    }

    /// One submission that declared `ranges` (each once) is copied back:
    /// the ranges no other submission still holds, in order.
    fn done(&mut self, ranges: &[(usize, usize)]) -> Vec<(usize, usize)> {
        let mut free = Vec::new();
        for range in ranges {
            match self.0.get_mut(range) {
                Some(f) if f.0 > 1 => f.0 -= 1,
                Some(_) => {
                    self.0.remove(range);
                    free.push(*range);
                }
                // Not counted (declared before anything counted): free.
                None => free.push(*range),
            }
        }
        free.sort_unstable();
        free
    }
}

/// The relay's merge of a stretch of a copy ([`relay`]). `now` is what the
/// card's memory holds, read once; `held` is the copy's reference for the
/// stretch (its snapshot, or the CPU's side of a copy in the card's
/// memory): what the card's memory held when it was last looked at;
/// `title` is title memory as just read. Every byte the card changed
/// (`now` differs from `held`) goes into `title` and into `held`; every
/// other byte of both stays as it is, so a byte the title wrote and the
/// card did not keeps the title's value. Whether `title` changed.
///
/// Nothing is stored into `now`'s source: the card may be writing it for a
/// later flight at this moment, and a store there, even of the value just
/// read, could undo that write. (The merge of a waited-for submission
/// stores every byte of a differing stretch back into the copy.)
fn merge_piece(now: &[u8], held: &mut [u8], title: &mut [u8]) -> bool {
    let mut differs = 0u8;
    for ((n, h), t) in now.iter().zip(held.iter_mut()).zip(title.iter_mut()) {
        let wrote = if *n != *h { 0xffu8 } else { 0 };
        let merged = (*t & !wrote) | (*n & wrote);
        differs |= merged ^ *t;
        *t = merged;
        *h = (*h & !wrote) | (*n & wrote);
    }
    differs != 0
}

/// `[from, to)` without the ranges of `skip`, in order.
fn outside(from: usize, to: usize, mut skip: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    skip.sort_unstable();
    let mut out = Vec::new();
    for_outside(from, to, skip, |a, b| out.push((a, b)));
    out
}

/// Visits the parts of `[from, to)` outside ranges sorted by their start.
fn for_outside(
    from: usize,
    to: usize,
    skip: impl IntoIterator<Item = (usize, usize)>,
    mut part: impl FnMut(usize, usize),
) {
    let mut at = from;
    for (a, b) in skip {
        if b <= at {
            continue;
        }
        if a >= to {
            break;
        }
        if a > at {
            part(at, a);
        }
        at = at.max(b);
    }
    if at < to {
        part(at, to);
    }
}

#[cfg(test)]
mod relay_tests {
    use super::{Flights, changed_runs, merge_piece, outside};

    /// A range two flights declared is free only when both are done, in
    /// either order; a submission that declares it many times counts once.
    #[test]
    fn a_range_stays_closed_until_every_flight_that_declared_it_is_done() {
        let (a, b) = ((0x1000, 0x2000), (0x1800, 0x3000));
        let mut flights = Flights::default();
        for _ in 0..3 {
            flights.declare(a, 7);
        }
        flights.declare(b, 7);
        flights.declare(a, 8);
        // Flight 7 lands: `a` is still flight 8's, `b` was only flight 7's.
        assert_eq!(flights.done(&[a, b]), vec![b]);
        // Declared again meanwhile by a third.
        flights.declare(a, 9);
        assert_eq!(flights.done(&[a]), vec![]);
        assert_eq!(flights.done(&[a]), vec![a]);
        assert!(flights.0.is_empty());
        // The later flight first: the same.
        flights.declare(a, 1);
        flights.declare(a, 2);
        assert_eq!(flights.done(&[a]), vec![]);
        assert_eq!(flights.done(&[a]), vec![a]);
        // A range never counted is free at once.
        assert_eq!(flights.done(&[b]), vec![b]);
    }

    /// The merge moves exactly the bytes the card changed, into title
    /// memory and into the reference, and stores nothing else.
    #[test]
    fn the_merge_stores_only_what_the_card_changed() {
        let held0: Vec<u8> = (0..200u32).map(|i| (i * 5 + 1) as u8).collect();
        let mut now = held0.clone();
        for at in [0usize, 3, 4, 5, 64, 199] {
            now[at] ^= 0x33;
        }
        // The title wrote other bytes meanwhile, and one the card wrote too.
        let mut title = held0.clone();
        for at in [1usize, 5, 100] {
            title[at] = 0xee;
        }
        let title0 = title.clone();
        let mut held = held0.clone();
        assert!(merge_piece(&now, &mut held, &mut title));
        for at in 0..200 {
            if now[at] != held0[at] {
                assert_eq!((title[at], held[at]), (now[at], now[at]), "the card's byte at {at}");
            } else {
                assert_eq!((title[at], held[at]), (title0[at], held0[at]), "untouched at {at}");
            }
        }
        // Again with nothing new: nothing changes and nothing is reported.
        let (title1, held1) = (title.clone(), held.clone());
        assert!(!merge_piece(&now, &mut held, &mut title));
        assert_eq!((title, held), (title1, held1));
    }

    #[test]
    fn ranges_are_cut_out() {
        assert_eq!(outside(10, 50, vec![]), vec![(10, 50)]);
        assert_eq!(outside(10, 50, vec![(0, 10), (50, 60)]), vec![(10, 50)]);
        assert_eq!(outside(10, 50, vec![(30, 35), (0, 12), (34, 40), (48, 99)]), vec![(12, 30), (40, 48)]);
        assert_eq!(outside(10, 50, vec![(0, 99)]), vec![]);
    }

    /// A model of a copy in the PC's memory under the relay: the card
    /// writes `copy` directly; `snap` is the snapshot; `title` is title
    /// memory. Flight A and flight B declare overlapping ranges and both
    /// are in the air; the title's CPU writes in between and its bytes are
    /// put in beside the flights' (`merge_under_writes`: into the copy and
    /// the snapshot); A lands while B is still writing; then B lands.
    #[test]
    fn two_flights_overlapping_ranges_and_a_cpu_write_in_between() {
        const N: usize = 64;
        let base: Vec<u8> = (0..N as u32).map(|i| i as u8).collect();
        let (mut copy, mut title) = (base.clone(), base.clone());
        // A declares [0, 40), B declares [24, 64): each byte's snapshot is
        // taken when the first range that holds it is declared.
        let (a, b) = ((0usize, 40usize), (24usize, 64usize));
        let mut snap = base.clone();
        // The card runs A: writes 4..12 and 30..34.
        for at in (4..12).chain(30..34) {
            copy[at] = 0xa0;
        }
        // The title's CPU writes 8..10 (over A's bytes), 20 and 50, after
        // A's refresh: the next submission's preparing finds them against
        // the snapshot and puts them in, into the copy and the snapshot.
        for at in [8usize, 9, 20, 50] {
            title[at] = 0x77;
        }
        for (at, n) in changed_runs(&title, &snap) {
            copy[at..at + n].copy_from_slice(&title[at..at + n]);
            snap[at..at + n].copy_from_slice(&title[at..at + n]);
        }
        // The card starts B: it has written 32..36 so far, over A's 32..34.
        for at in 32..36 {
            copy[at] = 0xb0;
        }
        // A lands (B still on the card): its range is merged. The copy is
        // read once and not written.
        let seen = copy.clone();
        {
            let mut t = title[a.0..a.1].to_vec();
            assert!(merge_piece(&seen[a.0..a.1], &mut snap[a.0..a.1], &mut t));
            title[a.0..a.1].copy_from_slice(&t);
        }
        assert_eq!(copy, seen, "the merge stores nothing into the copy");
        // A's bytes are in title memory, except where the title wrote
        // after A's shaders did (8, 9: the title's); B's first bytes came
        // along; nothing of B's range beyond A's was touched.
        for at in (4..8).chain(10..12).chain(30..32) {
            assert_eq!(title[at], 0xa0, "A's byte at {at}");
        }
        assert_eq!((title[8], title[9], title[20], title[50]), (0x77, 0x77, 0x77, 0x77));
        for at in 32..36 {
            assert_eq!(title[at], 0xb0, "B's byte so far at {at}");
        }
        // The card finishes B: 36..44.
        for at in 36..44 {
            copy[at] = 0xb1;
        }
        // The title's CPU writes 40 after B's shader did: found against
        // the snapshot at the next preparing, which is before B lands.
        title[40] = 0x55;
        for (at, n) in changed_runs(&title[b.0..b.1], &snap[b.0..b.1]) {
            let at = at + b.0;
            copy[at..at + n].copy_from_slice(&title[at..at + n]);
            snap[at..at + n].copy_from_slice(&title[at..at + n]);
        }
        // B lands.
        let seen = copy.clone();
        {
            let mut t = title[b.0..b.1].to_vec();
            assert!(merge_piece(&seen[b.0..b.1], &mut snap[b.0..b.1], &mut t));
            title[b.0..b.1].copy_from_slice(&t);
        }
        // B's results are there, never replaced by A's older ones.
        for at in 32..36 {
            assert_eq!(title[at], 0xb0);
        }
        for at in (36..40).chain(41..44) {
            assert_eq!(title[at], 0xb1, "B's byte at {at}");
        }
        assert_eq!(title[40], 0x55, "the title wrote it last");
        // In the declared ranges the copy, the snapshot and title memory
        // now agree byte for byte: nothing is left to hand back, and
        // nothing was lost.
        assert_eq!(copy[a.0..b.1], title[a.0..b.1]);
        assert_eq!(snap[a.0..b.1], title[a.0..b.1]);
        // A third merge finds nothing.
        let mut t = title.clone();
        assert!(!merge_piece(&copy, &mut snap, &mut t));
    }

    /// A model of a copy in the card's memory under the relay: `card` is
    /// the card's side, `cpu` the CPU's side, `title` title memory. The
    /// title's changed bytes go into the CPU's side at once and up to the
    /// card with a later send; pieces fetched before that send must not
    /// take the card's older bytes there for something the card wrote.
    #[test]
    fn title_bytes_not_yet_on_the_card_are_not_taken_for_shader_writes() {
        const N: usize = 48;
        let base: Vec<u8> = (0..N as u32).map(|i| (i + 100) as u8).collect();
        let (mut card, mut cpu, mut title) = (base.clone(), base.clone(), base.clone());
        // Flight A's shaders write 10..14 on the card.
        for at in 10..14 {
            card[at] = 0xa0;
        }
        // The title's CPU writes 12 (over A's) and 30; the preparing of
        // the next submission puts both into the CPU's side and notes them
        // to go up (send 5). They are not on the card yet.
        title[12] = 0x77;
        title[30] = 0x78;
        let mut noted = Vec::new();
        for (at, n) in changed_runs(&title, &cpu) {
            cpu[at..at + n].copy_from_slice(&title[at..at + n]);
            noted.push((at, at + n, 5u64));
        }
        // A's pieces are fetched after send 4, before send 5: the fetch
        // holds A's bytes and the card's old bytes at 12 and 30.
        let fetched = card.clone();
        let fetched_after = 4u64;
        // Send 5 goes up behind it.
        for &(a, b, _) in &noted {
            card[a..b].copy_from_slice(&cpu[a..b]);
        }
        // The merge leaves out what went up after the fetch.
        let skip: Vec<(usize, usize)> = noted.iter().filter(|s| s.2 > fetched_after).map(|s| (s.0, s.1)).collect();
        let mut changed = false;
        for (a, b) in outside(0, N, skip) {
            let mut t = title[a..b].to_vec();
            changed |= merge_piece(&fetched[a..b], &mut cpu[a..b], &mut t);
            title[a..b].copy_from_slice(&t);
        }
        assert!(changed);
        // A's bytes arrived; the title's two writes stand, in title memory
        // and on both sides of the copy.
        for at in [10usize, 11, 13] {
            assert_eq!((title[at], cpu[at]), (0xa0, 0xa0));
        }
        assert_eq!((title[12], cpu[12], card[12]), (0x77, 0x77, 0x77));
        assert_eq!((title[30], cpu[30], card[30]), (0x78, 0x78, 0x78));
        assert_eq!(card, cpu);
        assert_eq!(card, title);
        // Without the leaving out, the card's old byte at 30 would have
        // been taken for a shader write and put over the title's.
        let mut wrong_title = title.clone();
        let mut wrong_cpu = cpu.clone();
        merge_piece(&fetched, &mut wrong_cpu, &mut wrong_title);
        assert_eq!(wrong_title[30], base[30]);
    }
}

/// Snapshot memory kept for reuse (already touched, so no page faults).
static POOL: Mutex<SnapshotPool> = Mutex::new(SnapshotPool { buffers: Vec::new(), bytes: 0 });

#[derive(Default)]
struct SnapshotPool {
    /// Descending capacity order, preserved on insertion and removal.
    buffers: Vec<Vec<u8>>,
    bytes: usize,
}

impl SnapshotPool {
    fn take(&mut self, n: usize) -> Option<Vec<u8>> {
        let most = n.saturating_mul(4).max(1 << 20);
        let i = self.buffers.partition_point(|v| v.capacity() >= n).checked_sub(1)?;
        if self.buffers[i].capacity() > most {
            return None;
        }
        let v = self.buffers.remove(i);
        self.bytes -= v.capacity();
        Some(v)
    }

    fn put(&mut self, v: Vec<u8>) {
        let capacity = v.capacity();
        let i = self.buffers.partition_point(|b| b.capacity() >= capacity);
        self.buffers.insert(i, v);
        self.bytes += capacity;
    }

    /// Normal returns keep the largest buffers, including one oversized buffer.
    fn keep_largest(&mut self, count: usize, most: usize) -> Vec<Vec<u8>> {
        if self.buffers.len() <= count && self.bytes <= most {
            return Vec::new();
        }
        let (mut keep, mut bytes) = (0, 0);
        for v in self.buffers.iter().take(count) {
            let next = bytes + v.capacity();
            if keep > 0 && next > most {
                break;
            }
            keep += 1;
            bytes = next;
        }
        self.bytes = bytes;
        self.buffers.split_off(keep)
    }

    /// Pressure sweeps keep the smallest buffers. An idle sweep takes the whole pool.
    fn keep_smallest(&mut self, most: usize) -> Vec<Vec<u8>> {
        if most == 0 {
            self.bytes = 0;
            return std::mem::take(&mut self.buffers);
        }
        if self.bytes <= most {
            return Vec::new();
        }
        let (mut keep, mut bytes) = (0, 0);
        for v in self.buffers.iter().rev() {
            let next = bytes + v.capacity();
            if next > most {
                break;
            }
            keep += 1;
            bytes = next;
        }
        self.bytes = bytes;
        let kept = self.buffers.split_off(self.buffers.len() - keep);
        std::mem::replace(&mut self.buffers, kept)
    }
}

/// When a snapshot buffer was last taken from or given back to [`POOL`]
/// ([`crate::host::moment`]), for [`let_go_idle_snapshots`]. Read and
/// updated while holding the pool lock so a sweep sees the latest use.
static POOL_USED: AtomicU64 = AtomicU64::new(0);

/// An empty buffer with room for `n` bytes, from the pool if it has one
/// of about that size: a small copy's snapshot took the 174 MB buffer kept
/// for the big heap's, which then got another, until eight of them were
/// reserved (1.4 GB in run 24).
fn pooled(n: usize) -> Vec<u8> {
    let reused = {
        let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
        POOL_USED.store(crate::host::moment(), Ordering::Relaxed);
        pool.take(n)
    };
    let mut v = reused.unwrap_or_else(|| Vec::with_capacity(n));
    v.clear();
    v
}

/// How many snapshot buffers are kept (`VICTORY_SNAPSHOT_POOL`; `8`: as
/// before).
///
/// A frame gives some 330 ranges of copies in the PC's memory to shaders, and
/// most are of copies of their own; with two or three frames in the air eight
/// buffers were the eight largest, and every other snapshot was memory fresh
/// from the kernel and given back to it: 11 % of the submitting thread in the
/// opening (profile cpu-1: `NtAllocateVirtualMemory` under
/// [`Shadow::writing`]), and the page faults of filling it.
fn snapshots_kept() -> usize {
    static MOST: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MOST.get_or_init(|| std::env::var("VICTORY_SNAPSHOT_POOL").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(1024))
}

/// Returns a snapshot buffer to the pool: at most [`snapshots_kept`] kept,
/// the largest, and of those the largest one and the next ones up to `KEPT`
/// bytes in all.
fn give_back(v: Vec<u8>) {
    const KEPT: usize = 512 << 20;
    let count = snapshots_kept();
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    POOL_USED.store(crate::host::moment(), Ordering::Relaxed);
    pool.put(v);
    // (Given back to the kernel without the lock: milliseconds.)
    let gone = pool.keep_largest(count, KEPT);
    drop(pool);
    drop(gone);
}

/// What [`POOL`] keeps after a sweep while the host is short of memory
/// ([`crate::residency::pressured`]): the smallest buffers, up to this many
/// bytes in all. They are the ones asked for most (most of a frame's 330
/// ranges are small), and the large ones are what fills the 512 MiB.
const POOL_PRESSURED_KEPT: usize = 32 << 20;

/// Lets go of the snapshot buffers [`POOL`] keeps when nobody wants them.
/// Only [`give_back`] ever trimmed the pool, and only when another buffer
/// came back, so once snapshots stopped the last 512 MiB stayed for the
/// rest of the run, low memory or not (a load test: 524,123 KiB kept after
/// five idle seconds; 13 MiB private once emptied by hand). Now
/// - idle for the time the other host caches wait ([`crate::host::ram_idle_ms`])
///   and four frames, all of it goes;
/// - while the host is short of memory, all but [`POOL_PRESSURED_KEPT`]
///   goes, the largest first, whatever the size of the largest. Returns can
///   refill the pool between sweeps.
///
/// The next snapshot is then memory fresh from the kernel, as it was for
/// every snapshot past the pool before. Called by the once-a-second sweep.
fn let_go_idle_snapshots() {
    const IDLE_FRAMES: u64 = 4;
    let idle_ms = crate::host::ram_idle_ms(crate::host::available_physical_ram());
    let pressured = crate::residency::pressured();
    let gone = {
        let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
        let idle = crate::host::long_ago(POOL_USED.load(Ordering::Relaxed), idle_ms, IDLE_FRAMES);
        let Some(most) = snapshots_to_keep(idle, pressured) else { return };
        pool.keep_smallest(most)
    };
    // (Given back to the kernel without the lock.)
    drop(gone);
}

/// How many bytes of snapshot buffers [`POOL`] may keep: none once idle,
/// [`POOL_PRESSURED_KEPT`] while the host is short of memory, otherwise
/// what [`give_back`] keeps (`None`: nothing to do).
fn snapshots_to_keep(idle: bool, pressured: bool) -> Option<usize> {
    if idle {
        Some(0)
    } else if pressured {
        Some(POOL_PRESSURED_KEPT)
    } else {
        None
    }
}

#[cfg(test)]
mod pool_tests {
    use super::*;

    fn sizes(pool: &[Vec<u8>]) -> Vec<usize> {
        pool.iter().map(Vec::capacity).collect()
    }

    fn pool_with_sizes(mib: &[usize]) -> SnapshotPool {
        let mut pool = SnapshotPool::default();
        for &n in mib {
            pool.put(Vec::with_capacity(n << 20));
        }
        pool
    }

    #[test]
    fn reuse_takes_the_smallest_fit_and_tracks_capacity() {
        let mut pool = pool_with_sizes(&[8, 1, 4, 2, 4]);
        assert!(pool.take(9 << 20).is_none());
        assert_eq!(pool.take(3 << 20).unwrap().capacity(), 4 << 20);
        assert_eq!(pool.take(5 << 20).unwrap().capacity(), 8 << 20);
        assert_eq!(sizes(&pool.buffers), [4 << 20, 2 << 20, 1 << 20]);
        assert_eq!(pool.bytes, 7 << 20);
        assert_eq!(pool.take(1).unwrap().capacity(), 1 << 20);
        assert!(pool.take(1).is_none());
        assert_eq!(pool.bytes, 6 << 20);
    }

    #[test]
    fn normal_returns_obey_size_and_count_limits() {
        let mut pool = pool_with_sizes(&[8, 2, 4, 1]);
        assert!(pool.keep_largest(4, 16 << 20).is_empty());
        assert_eq!(pool.bytes, 15 << 20);
        assert_eq!(sizes(&pool.keep_largest(2, 16 << 20)), [2 << 20, 1 << 20]);
        assert_eq!(pool.bytes, 12 << 20);
        assert_eq!(sizes(&pool.keep_largest(2, 7 << 20)), [4 << 20]);
        assert_eq!(pool.bytes, 8 << 20);
        assert_eq!(sizes(&pool.keep_largest(0, 7 << 20)), [8 << 20]);
        assert!(pool.buffers.is_empty());
        assert_eq!(pool.bytes, 0);
    }

    #[test]
    fn trimming_keeps_the_smallest_up_to_the_limit() {
        let mut pool = pool_with_sizes(&[8, 1, 4, 2]);
        let gone = pool.keep_smallest(7 << 20);
        assert_eq!(sizes(&pool.buffers), [4 << 20, 2 << 20, 1 << 20]);
        assert_eq!(sizes(&gone), [8 << 20]);
        assert_eq!(pool.bytes, 7 << 20);
        assert!(pool.keep_smallest(7 << 20).is_empty());
    }

    #[test]
    fn trimming_to_nothing_empties_the_pool() {
        let mut pool = pool_with_sizes(&[3, 1]);
        let gone = pool.keep_smallest(0);
        assert!(pool.buffers.is_empty());
        assert_eq!(pool.bytes, 0);
        assert_eq!(sizes(&gone), [3 << 20, 1 << 20]);
    }

    #[test]
    fn a_buffer_past_the_limit_goes_even_alone() {
        // `give_back` always keeps its largest buffer; under pressure that
        // one goes too.
        let mut pool = pool_with_sizes(&[600]);
        let gone = pool.keep_smallest(POOL_PRESSURED_KEPT);
        assert!(pool.buffers.is_empty());
        assert_eq!(pool.bytes, 0);
        assert_eq!(gone.len(), 1);
    }

    #[test]
    fn an_idle_pool_goes_and_a_pressed_one_shrinks() {
        assert_eq!(snapshots_to_keep(true, false), Some(0));
        assert_eq!(snapshots_to_keep(true, true), Some(0));
        assert_eq!(snapshots_to_keep(false, true), Some(POOL_PRESSURED_KEPT));
        assert_eq!(snapshots_to_keep(false, false), None);
    }

    /// A full pool used by eight snapshot workers. Run alone in a fresh process;
    /// reports wall-clock time for 80,000 acquire/return pairs in this process.
    #[test]
    #[ignore]
    fn pool_contention_load() {
        const WORKERS: usize = 8;
        const PAIRS_PER_WORKER: usize = 10_000;
        for i in 0..384 {
            give_back(Vec::with_capacity((16 + i % 64) << 10));
        }
        let barrier = std::sync::Barrier::new(WORKERS + 1);
        let elapsed = std::thread::scope(|scope| {
            let workers: Vec<_> = (0..WORKERS)
                .map(|worker| {
                    let barrier = &barrier;
                    scope.spawn(move || {
                        barrier.wait();
                        for i in 0..PAIRS_PER_WORKER {
                            let size = (16 + (i * 31 + worker * 7) % 64) << 10;
                            let mut buffer = pooled(size);
                            buffer.push(0);
                            give_back(buffer);
                        }
                    })
                })
                .collect();
            let start = std::time::Instant::now();
            barrier.wait();
            for worker in workers {
                worker.join().unwrap();
            }
            start.elapsed()
        });
        eprintln!("snapshot pool: {} pairs, {:.3} ms", WORKERS * PAIRS_PER_WORKER, elapsed.as_secs_f64() * 1000.0);
    }

    /// The pool after snapshots stop, in a process's private bytes: before,
    /// 512 MiB stayed however long nothing asked. Run alone (the pool is
    /// global): `cargo test -p victory-gpu --lib
    /// shadow::pool_tests::pool_memory_load -- --ignored --nocapture --test-threads=1`
    #[test]
    #[ignore]
    fn pool_memory_load() {
        fn private_kib() -> usize {
            use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS_EX};
            use windows_sys::Win32::System::Threading::GetCurrentProcess;
            // SAFETY: plain data the call fills in, for this process.
            let mut c: PROCESS_MEMORY_COUNTERS_EX = unsafe { std::mem::zeroed() };
            c.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32;
            unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), (&raw mut c).cast(), c.cb) };
            c.PrivateUsage >> 10
        }
        let start = private_kib();
        for i in 0..256usize {
            let mut v = pooled((1 + i % 32) << 20);
            v.resize(v.capacity(), 1);
            give_back(v);
        }
        let kept = POOL.lock().unwrap().bytes;
        let full = private_kib();
        let gone = POOL.lock().unwrap().keep_smallest(POOL_PRESSURED_KEPT);
        drop(gone);
        let pressed = private_kib();
        let gone = POOL.lock().unwrap().keep_smallest(0);
        drop(gone);
        let idle = private_kib();
        eprintln!(
            "pool kept {} KiB; private KiB over the start: full {}, pressed {}, idle {}",
            kept >> 10,
            full.saturating_sub(start),
            pressed.saturating_sub(start),
            idle.saturating_sub(start)
        );
        assert!(pressed.saturating_sub(start) <= (POOL_PRESSURED_KEPT >> 10) + 8 * 1024);
        assert!(POOL.lock().unwrap().buffers.is_empty());
    }
}

const NOT_WATCHED: u8 = 2;
/// Watched, not by the kernel but by the guards, when last asked about.
const GUARDED: u8 = 3;

// Fixed-size equality compiles to two SIMD comparisons on Windows x64.
// Only the first or last unequal chunk needs a byte-wise search.
const DIFF_CHUNK: usize = 32;

/// Where `a` and `b` first differ within their common length.
fn first_difference(a: &[u8], b: &[u8]) -> Option<usize> {
    let len = a.len().min(b.len());
    let (a_chunks, a_tail) = a[..len].as_chunks::<DIFF_CHUNK>();
    let (b_chunks, b_tail) = b[..len].as_chunks::<DIFF_CHUNK>();
    let (offset, a, b) = match a_chunks.iter().zip(b_chunks).position(|(x, y)| x != y) {
        Some(i) => (i * DIFF_CHUNK, &a_chunks[i][..], &b_chunks[i][..]),
        None => (len - a_tail.len(), a_tail, b_tail),
    };
    a.iter().zip(b).position(|(x, y)| x != y).map(|i| offset + i)
}

/// Where `a` and `b` last differ within their common length.
fn last_difference(a: &[u8], b: &[u8]) -> Option<usize> {
    let len = a.len().min(b.len());
    let (a_chunks, a_tail) = a[..len].as_chunks::<DIFF_CHUNK>();
    let (b_chunks, b_tail) = b[..len].as_chunks::<DIFF_CHUNK>();
    if let Some(i) = a_tail.iter().zip(b_tail).rposition(|(x, y)| x != y) {
        return Some(len - a_tail.len() + i);
    }
    let chunk = a_chunks.iter().zip(b_chunks).rposition(|(x, y)| x != y)?;
    a_chunks[chunk]
        .iter()
        .zip(&b_chunks[chunk])
        .rposition(|(x, y)| x != y)
        .map(|i| chunk * DIFF_CHUNK + i)
}

#[cfg(test)]
mod difference_tests {
    use super::{PIECE, Shadow, first_difference, last_difference};

    #[test]
    fn parallel_compare_includes_the_first_partial_piece() {
        const LEN: usize = 48 << 20;
        const START: u64 = 0x0012_3456_8000;
        let before = vec![0u8; LEN];
        let mut data = before.clone();
        let changed = [0, PIECE - 1, 3 << 20, 20 << 20, 40 << 20, LEN - 1];
        for at in changed {
            data[at] = 1;
        }

        // The compare reads the copy directly, so it needs no D3D12 device.
        let shadow = Shadow::unbacked(START, data.as_mut_ptr(), LEN as u64, false);
        let found = shadow.differing(0, LEN, &before);
        for at in changed {
            assert!(
                found.iter().any(|&(start, len)| (start..start + len).contains(&at)),
                "missed change at {at:#x}"
            );
        }
        for (start, len) in found {
            assert!(changed.iter().any(|at| (start..start + len).contains(at)));
            let end = start + len;
            assert!(end == LEN || (START as usize + end).is_multiple_of(PIECE));
        }
    }

    /// The chunk searches agree with a byte-wise scan across chunk boundaries.
    #[test]
    fn differences_are_found_as_byte_by_byte() {
        let base: Vec<u8> = (0..98u32).map(|i| (i * 7 + 3) as u8).collect();
        for len in [0, 1, 31, 32, 33, 63, 64, 65, 97] {
            let a = &base[1..1 + len];
            let same = a.to_vec();
            assert_eq!(first_difference(&same, a), None);
            assert_eq!(last_difference(&same, a), None);
            for at in 0..len {
                for other in at..len {
                    let mut b = a.to_vec();
                    b[at] ^= 0x80;
                    b[other] ^= 0x01;
                    let first = a.iter().zip(&b).position(|(x, y)| x != y);
                    let last = a.iter().zip(&b).rposition(|(x, y)| x != y);
                    assert_eq!(first_difference(a, &b), first, "{len} {at} {other}");
                    assert_eq!(last_difference(a, &b), last, "{len} {at} {other}");
                }
            }
        }
    }

    #[test]
    fn differences_ignore_bytes_past_the_shorter_slice() {
        let a = [0; 65];
        let mut b = [0; 67];
        b[65] = 1;
        assert_eq!(first_difference(&a, &b), None);
        assert_eq!(last_difference(&b, &a), None);
        b[32] = 1;
        b[64] = 1;
        assert_eq!(first_difference(&b, &a), Some(32));
        assert_eq!(last_difference(&a, &b), Some(64));
    }
}

/// The unit shader writes are looked for and put back in.
const PIECE: usize = 64 << 10;

/// Calls `f` with `[from, to)` cut at multiples of [`PIECE`].
fn for_pieces_of(from: usize, to: usize, mut f: impl FnMut(usize, usize)) {
    let mut at = from;
    while at < to {
        let next = ((at / PIECE + 1) * PIECE).min(to);
        f(at, next);
        at = next;
    }
}

/// Copies `len` bytes, on several threads if they are many: a snapshot of
/// the world's big heap is 100-170 MB per submission, 2-4 GB a frame, and
/// one thread copies some 8 GB/s.
///
/// # Safety
/// Both ranges are valid for `len` bytes and do not overlap.
unsafe fn copy_wide(src: *const u8, dst: *mut u8, len: usize) {
    const PER_THREAD: usize = 16 << 20;
    let threads = (len / PER_THREAD).clamp(1, 6);
    if threads == 1 {
        // SAFETY: the caller's.
        unsafe { std::ptr::copy_nonoverlapping(src, dst, len) };
        return;
    }
    let (src, dst, each) = (src as usize, dst as usize, len.div_ceil(threads));
    std::thread::scope(|s| {
        for i in 0..threads {
            let at = (i * each).min(len);
            let n = each.min(len - at);
            s.spawn(move || {
                // SAFETY: a part of the caller's ranges.
                unsafe { std::ptr::copy_nonoverlapping((src + at) as *const u8, (dst + at) as *mut u8, n) };
            });
        }
    });
}

/// Whether the snapshots of a submission's shader-written ranges, and the
/// compare that finds what the shaders wrote, run on several threads
/// across ranges (`VICTORY_WIDE_BACK=1`; off by default).
///
/// Both ran range by range on the submitting thread, a range on several
/// threads only if it was 16 MiB or more. The ranges left on the CPU's side
/// since the big buffers went to the card are small and many: in run 86,
/// 7,024 declared ranges a frame, 375 MB of them compared after the card
/// ran in 563 pieces of work, none of 16 MiB: 51 ms a frame of snapshots
/// and 35 ms of comparing, one thread, bound by one core's memory speed.
///
/// With the switch a submission's snapshots are noted as it is prepared
/// and taken together when its refreshes are done, just before it goes to
/// the card ([`take_snapshots`]), and the compares of everything declared
/// run together after the card ([`find_all`]), both cut into pieces of
/// work of about 1 MiB for a few threads that are kept ([`wide`]). The
/// bytes taken and the bytes compared are the same ones: a range's bytes
/// cannot change between the moment it is declared and the end of the
/// preparing (a declared range is left alone by every refresh, the title's
/// own copies write title memory and not the copy, and the card runs the
/// lists only afterwards).
pub(crate) fn wide_back() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // On by default; `VICTORY_WIDE_BACK=0` turns it off.
    *ON.get_or_init(|| std::env::var_os("VICTORY_WIDE_BACK").is_none_or(|v| v != "0"))
}

type Job = Box<dyn FnOnce() + Send>;

/// Threads kept for [`wide`]: what they are to do, and the word that there
/// is something.
struct Workers {
    jobs: Mutex<std::collections::VecDeque<Job>>,
    more: std::sync::Condvar,
    thread_count: usize,
    /// How many jobs there are, for [`spin`].
    there: std::sync::atomic::AtomicUsize,
}

/// Waits awake until `there` says so, `VICTORY_WIDE_SPIN` microseconds at
/// most (0, the default: not at all): asleep, a thread is woken in 50 us and
/// more, 95 times a frame.
fn spin(there: impl Fn() -> bool) {
    static MOST: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let most = *MOST.get_or_init(|| std::env::var("VICTORY_WIDE_SPIN").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(0));
    if most == 0 {
        return;
    }
    let began = std::time::Instant::now();
    while !there() && (began.elapsed().as_micros() as u64) < most {
        for _ in 0..16 {
            std::hint::spin_loop();
        }
    }
}

/// How many threads are kept for [`wide`].
fn wide_threads() -> usize {
    std::thread::available_parallelism().map_or(4, |n| n.get() / 4).clamp(2, 6)
}

fn workers() -> &'static Workers {
    static WORKERS: std::sync::OnceLock<Workers> = std::sync::OnceLock::new();
    static STARTED: std::sync::Once = std::sync::Once::new();
    let workers = WORKERS.get_or_init(|| Workers {
        jobs: Mutex::new(std::collections::VecDeque::new()),
        more: std::sync::Condvar::new(),
        thread_count: wide_threads(),
        there: std::sync::atomic::AtomicUsize::new(0),
    });
    STARTED.call_once(|| {
        for i in 0..workers.thread_count {
            let _ = std::thread::Builder::new().name(format!("victory-wide-{i}")).spawn(move || {
                crate::cores::keep_helper_on_fast_cores("victory-wide");
                let mut priority = crate::cores::Priority::new();
                let workers = WORKERS.get().expect("made above");
                loop {
                    let job = {
                        // A little while awake first: the next work comes
                        // within microseconds more often than not.
                        spin(|| workers.there.load(Ordering::Acquire) > 0);
                        let mut jobs = workers.jobs.lock().unwrap_or_else(|e| e.into_inner());
                        loop {
                            if let Some(job) = jobs.pop_front() {
                                workers.there.fetch_sub(1, Ordering::AcqRel);
                                break job;
                            }
                            jobs = workers.more.wait(jobs).unwrap_or_else(|e| e.into_inner());
                        }
                    };
                    priority.keep();
                    let _working = crate::timeline::span("a job for another thread");
                    job();
                }
            });
        }
    });
    workers
}

/// Runs `jobs` on the kept threads and on this one, and returns when all
/// of them are done. The jobs copy and compare memory; they do not panic.
fn wide(jobs: Vec<Job>) {
    if jobs.len() < 2 {
        for job in jobs {
            job();
        }
        return;
    }
    let workers = workers();
    let _wide = crate::timeline::span("work spread over threads");
    let job_count = jobs.len();
    let left = Arc::new((Mutex::new(job_count), std::sync::Condvar::new()));
    {
        let mut queue = workers.jobs.lock().unwrap_or_else(|e| e.into_inner());
        for job in jobs {
            let left = left.clone();
            workers.there.fetch_add(1, Ordering::AcqRel);
            queue.push_back(Box::new(move || {
                job();
                let mut n = left.0.lock().unwrap_or_else(|e| e.into_inner());
                *n -= 1;
                if *n == 0 {
                    left.1.notify_all();
                }
            }));
        }
    }
    // The calling thread also drains the queue. Wake helpers only for the
    // jobs it can share; a busy helper checks for more work before sleeping.
    let helpers = job_count - 1;
    if helpers >= workers.thread_count {
        workers.more.notify_all();
    } else {
        for _ in 0..helpers {
            workers.more.notify_one();
        }
    }
    loop {
        let job = workers.jobs.lock().unwrap_or_else(|e| e.into_inner()).pop_front();
        match job {
            Some(job) => {
                workers.there.fetch_sub(1, Ordering::AcqRel);
                job()
            }
            None => break,
        }
    }
    spin(|| left.0.try_lock().is_ok_and(|n| *n == 0));
    let mut n = left.0.lock().unwrap_or_else(|e| e.into_inner());
    while *n > 0 {
        n = left.1.wait(n).unwrap_or_else(|e| e.into_inner());
    }
}

/// [`wide`], for the host's own work on host copies (a submission's
/// refreshes, the merge of what shaders wrote to card copies).
pub(crate) fn wide_run(jobs: Vec<Box<dyn FnOnce() + Send>>) {
    wide(jobs);
}

/// Calls `work` with every number under `n`, on the kept threads and on
/// this one, the next number to whichever is free: for thousands of small
/// pieces of work, where a job each ([`wide`]) is mostly the jobs' queue.
pub(crate) fn wide_each(n: usize, work: impl Fn(usize) + Send + Sync + 'static) {
    let next = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let work = Arc::new(work);
    let threads = (wide_threads() + 1).min(n);
    let jobs = (0..threads)
        .map(|_| {
            let (next, work) = (next.clone(), work.clone());
            Box::new(move || {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= n {
                        break;
                    }
                    work(i);
                }
            }) as Job
        })
        .collect();
    wide(jobs);
}

/// About this many bytes to a piece of work for [`wide`].
const WIDE_JOB: usize = 1 << 20;

/// With [`wide_back`]: the snapshots noted while a submission is prepared
/// (from, to, bytes), taken by [`take_snapshots`].
static SNAPS: Mutex<Vec<(usize, usize, usize)>> = Mutex::new(Vec::new());

/// With [`wide_back`], when a submission's refreshes are done and before
/// its lists go to the card: takes the snapshots [`Shadow::writing`] noted.
/// Nothing to do without the switch.
pub(crate) fn take_snapshots() {
    let snaps = std::mem::take(&mut *SNAPS.lock().unwrap_or_else(|e| e.into_inner()));
    if snaps.is_empty() {
        return;
    }
    wide(copy_jobs(snaps));
}

/// `copies` (from, to, bytes) as pieces of work of about [`WIDE_JOB`]
/// bytes each: long ones cut, short ones together.
fn copy_jobs(copies: Vec<(usize, usize, usize)>) -> Vec<Job> {
    let mut jobs: Vec<Job> = Vec::new();
    let mut group: Vec<(usize, usize, usize)> = Vec::new();
    let mut bytes = 0;
    let flush = |group: &mut Vec<(usize, usize, usize)>, jobs: &mut Vec<Job>| {
        let group = std::mem::take(group);
        if !group.is_empty() {
            jobs.push(Box::new(move || {
                for (src, dst, len) in group {
                    // SAFETY: the caller's ranges: a host copy's mapping and
                    // its snapshot, which do not overlap.
                    unsafe { std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, len) };
                }
            }));
        }
    };
    for (src, dst, len) in copies {
        let mut at = 0;
        while at < len {
            let n = (WIDE_JOB - bytes).min(len - at);
            group.push((src + at, dst + at, n));
            at += n;
            bytes += n;
            if bytes >= WIDE_JOB {
                flush(&mut group, &mut jobs);
                bytes = 0;
            }
        }
    }
    flush(&mut group, &mut jobs);
    jobs
}

/// With [`wide_back`]: for each of `work` (a copy not in the card's memory
/// and a range `[address, end)` its shaders may have written, declared and
/// not yet done), the 64 KiB pieces that differ from the snapshot, as
/// [`Shadow::copy_back_with`] takes them; found on several threads.
pub(crate) fn find_all(work: &[(Arc<Shadow>, usize, usize)]) -> Vec<Vec<(usize, usize)>> {
    let scans = work
        .iter()
        .map(|(shadow, address, end)| {
            let (from, to) = shadow.inside(*address, end.saturating_sub(*address));
            // The snapshot's memory stays until the range is done with
            // (`copy_back_done`), which comes after this.
            match shadow.before.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
                Some(before) => (shadow.data as usize, before.as_ptr() as usize, shadow.start as usize, from, to),
                None => (0, 0, 0, 0, 0),
            }
        })
        .collect();
    find_pieces(scans)
}

/// For each of `scans` (where a copy's bytes are, where its snapshot's
/// are, the title address the copy starts at, offsets from and to): the
/// 64 KiB pieces (cut at multiples of 64 KiB of title memory, as
/// [`Shadow::differing`] cuts them) of `[from, to)` whose bytes differ, as
/// (offset, bytes) in order. On several threads ([`wide`]), in pieces of
/// work of about [`WIDE_JOB`] bytes that hold whole pieces.
fn find_pieces(scans: Vec<(usize, usize, usize, usize, usize)>) -> Vec<Vec<(usize, usize)>> {
    // What a job finds: (which scan, its place among that scan's jobs, the
    // pieces).
    type Found = Vec<(usize, usize, Vec<(usize, usize)>)>;
    // A job's part of a scan: the scan, then which and place.
    type Part = (usize, usize, usize, usize, usize, usize, usize);
    let found: Arc<Mutex<Found>> = Arc::default();
    let mut jobs: Vec<Job> = Vec::new();
    let mut group: Vec<Part> = Vec::new();
    let mut bytes = 0;
    let flush = |group: &mut Vec<Part>, jobs: &mut Vec<Job>| {
        let group = std::mem::take(group);
        if group.is_empty() {
            return;
        }
        let found = found.clone();
        jobs.push(Box::new(move || {
            let mut mine: Found = Vec::with_capacity(group.len());
            for (data, before, start, from, to, which, place) in group {
                let mut pieces = Vec::new();
                let mut at = from;
                while at < to {
                    let n = (PIECE - (start + at) % PIECE).min(to - at);
                    // SAFETY: inside the copy's mapping and its snapshot
                    // (which holds the copy's bytes at the copy's offsets);
                    // nothing writes either while a submission's results
                    // are compared.
                    let (now, old) = unsafe {
                        (
                            std::slice::from_raw_parts((data + at) as *const u8, n),
                            std::slice::from_raw_parts((before + at) as *const u8, n),
                        )
                    };
                    if now != old {
                        pieces.push((at, n));
                    }
                    at += n;
                }
                mine.push((which, place, pieces));
            }
            found.lock().unwrap_or_else(|e| e.into_inner()).extend(mine);
        }));
    };
    let count = scans.len();
    for (which, (data, before, start, from, to)) in scans.into_iter().enumerate() {
        let (mut at, mut place) = (from, 0);
        while at < to {
            // Cut where pieces end, so that every job sees whole pieces.
            let room = (WIDE_JOB - bytes).max(PIECE);
            let mut next = (at + room).min(to);
            if next < to {
                next -= (start + next) % PIECE;
                if next <= at {
                    next = (at + PIECE - (start + at) % PIECE).min(to);
                }
            }
            group.push((data, before, start, at, next, which, place));
            bytes += next - at;
            place += 1;
            at = next;
            if bytes >= WIDE_JOB {
                flush(&mut group, &mut jobs);
                bytes = 0;
            }
        }
    }
    flush(&mut group, &mut jobs);
    wide(jobs);
    let mut found = std::mem::take(&mut *found.lock().unwrap_or_else(|e| e.into_inner()));
    found.sort_unstable_by_key(|f| (f.0, f.1));
    let mut out = vec![Vec::new(); count];
    for (which, _, pieces) in found {
        out[which].extend(pieces);
    }
    out
}

/// Since the last frame-rate line, of the shader-written ranges looked for
/// on the CPU (copies not in the card's memory): bytes snapshotted in
/// windows outside buffers and in title buffers' copies; bytes compared in
/// windows, in buffers' copies, and of those in copies of 1 MiB or more;
/// ranges compared in windows and in buffers' copies.
static BACK_CLASS: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];

/// For the frame-rate line: where the CPU's search for shader writes goes
/// (per frame). A copy of a title buffer can go to the card's memory,
/// where the card does the search (`VICTORY_CARD_BUFFERS_MIN_MB`); a
/// window outside buffers cannot as the code stands.
pub(crate) fn back_line(frames: f64) {
    let c: Vec<f64> = BACK_CLASS.iter().map(|c| c.swap(0, Ordering::Relaxed) as f64 / frames).collect();
    if c.iter().all(|&v| v == 0.0) {
        return;
    }
    let mb = |v: f64| v / (1 << 20) as f64;
    tracing::info!(
        "shader writes looked for on the CPU per frame: {:.1} MB compared in {:.0} ranges of windows outside buffers ({:.1} MB \
         snapshotted), {:.1} MB in {:.0} ranges of title buffers' copies ({:.1} MB snapshotted), of them {:.1} MB in copies \
         of 1 MiB or more. VICTORY_CARD_BUFFERS_MIN_MB=1 sets only the lower size limit. Local copies must also be below \
         1 GiB and fit the buffer cap and the allowed share of any reported VRAM budget",
        mb(c[2]),
        c[5],
        mb(c[0]),
        mb(c[3]),
        c[6],
        mb(c[1]),
        mb(c[4]),
    );
}

#[cfg(test)]
mod wide_tests {
    use super::{Job, PIECE, WIDE_JOB, copy_jobs, find_pieces, wide};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Every job runs once, however many there are, and `wide` returns
    /// only when all have.
    #[test]
    fn all_jobs_run_before_wide_returns() {
        for n in [0usize, 1, 2, 7, 300] {
            let ran = Arc::new(AtomicUsize::new(0));
            let jobs: Vec<Job> = (0..n)
                .map(|i| {
                    let ran = ran.clone();
                    Box::new(move || {
                        ran.fetch_add(i + 1, Ordering::Relaxed);
                    }) as Job
                })
                .collect();
            wide(jobs);
            assert_eq!(ran.load(Ordering::Relaxed), n * (n + 1) / 2, "{n} jobs");
        }
    }

    /// Copies cut into jobs copy every byte once: long ones, short ones
    /// and empty ones.
    #[test]
    fn copies_cut_into_jobs_copy_the_same_bytes() {
        let len = 3 * WIDE_JOB + 12345;
        let src: Vec<u8> = (0..len).map(|i| (i * 31 + i / 977) as u8).collect();
        let mut dst = vec![0u8; len];
        let (s, d) = (src.as_ptr() as usize, dst.as_mut_ptr() as usize);
        // Ranges of the buffer, each to the same offsets of the other.
        let cuts = [0, 1, 1, 4096, 70_000, WIDE_JOB + 3, 2 * WIDE_JOB + 3, 2 * WIDE_JOB + 9, len];
        let copies: Vec<(usize, usize, usize)> = cuts.windows(2).map(|w| (s + w[0], d + w[0], w[1] - w[0])).collect();
        let jobs = copy_jobs(copies);
        assert!(jobs.len() >= 3);
        wide(jobs);
        assert!(src == dst);
    }

    /// The pieces found on several threads are the ones a walk piece by
    /// piece finds, for copies that start anywhere in a 64 KiB piece of
    /// title memory and ranges that start and end anywhere.
    #[test]
    fn pieces_found_wide_are_those_found_one_by_one() {
        let len = 5 * WIDE_JOB / 2 + 777;
        let old: Vec<u8> = (0..len).map(|i| (i * 7 + i / 513) as u8).collect();
        let mut now = old.clone();
        // Changes: single bytes at piece edges, a run across two pieces,
        // the last byte, and nothing in between.
        for at in [0usize, 1, PIECE - 1, PIECE, 3 * PIECE + 5, WIDE_JOB - 1, WIDE_JOB, WIDE_JOB + PIECE / 2, 2 * WIDE_JOB + 9, len - 1] {
            now[at] ^= 0x5a;
        }
        for byte in &mut now[9 * PIECE - 40..9 * PIECE + 40] {
            *byte ^= 0x01;
        }
        let (data, before) = (now.as_ptr() as usize, old.as_ptr() as usize);
        let mut scans = Vec::new();
        let mut wanted = Vec::new();
        for start in [0usize, 4096, PIECE - 16, 0xd_b790_df00] {
            for (from, to) in [(0, len), (0, 1), (1, 2), (PIECE - 1, PIECE + 1), (3 * PIECE, 3 * PIECE), (12345, len - 1), (WIDE_JOB - 3, 2 * WIDE_JOB + 10)] {
                scans.push((data, before, start, from, to));
                let mut pieces = Vec::new();
                let mut at = from;
                while at < to {
                    let n = (PIECE - (start + at) % PIECE).min(to - at);
                    if now[at..at + n] != old[at..at + n] {
                        pieces.push((at, n));
                    }
                    at += n;
                }
                wanted.push(pieces);
            }
        }
        // A scan of nothing (a range whose snapshot is gone).
        scans.push((0, 0, 0, 0, 0));
        wanted.push(Vec::new());
        assert_eq!(find_pieces(scans), wanted);
    }
}

/// What [`Shadow::copy_back`] does with a piece that differs from the
/// snapshot: `host` is the copy's bytes there, `old` the snapshot's. From
/// the first byte that differs to the last, title memory is read into
/// `title` (scratch, sized to that stretch) by `read` (the stretch's offset
/// in the piece, where to), the GPU's bytes are put in and `host` takes the
/// result. The stretch's offset in the piece and its length if that changed
/// title memory; none if nothing differs or title memory is not there.
/// `runs` takes the runs of the stretch (offsets into `title`, lengths)
/// whose bytes are the GPU's and differ from the title's as read: only
/// those are to be written, so a title write that lands between the read
/// and the write elsewhere in the stretch stays (upstream VIctory #2009).
/// `mask` is scratch.
fn back_piece(
    host: &mut [u8],
    old: &[u8],
    title: &mut Vec<u8>,
    mask: &mut Vec<u8>,
    runs: &mut Vec<(usize, usize)>,
    read: impl FnOnce(usize, &mut [u8]) -> bool,
) -> Option<(usize, usize)> {
    runs.clear();
    // From the first byte that differs to the last: title memory is read
    // and written there only (a shader that sets a few words of a piece
    // cost a read, a walk and a write of all 64 KiB of it).
    let first = first_difference(host, old)?;
    let last = last_difference(host, old).unwrap_or(first);
    let n = last + 1 - first;
    let (host, old) = (&mut host[first..=last], &old[first..=last]);
    title.clear();
    title.resize(n, 0);
    if !read(first, title) {
        return None;
    }
    // Title memory with the GPU's bytes put in; the copy takes the title's
    // bytes in turn.
    // Without a branch, so the loop runs many bytes at a time: the GPU's
    // byte where the copy's byte is not what it was before the submission
    // and the title left its byte as it was, else the title's. (With the
    // two searches above this was 44 ms a frame for 970 pieces.)
    // Where the shader and the title both changed a byte while the GPU ran,
    // the title's stays: the shader computed its byte from the copy's older
    // data (one cleared a bit of a record the title had already replaced,
    // and copying it back broke the title's object: an access violation
    // through 0xffffffff00000000; PalmVIDev e0b70476, VIctory f39c6c2).
    mask.clear();
    mask.resize(n, 0);
    let mut differs = 0u8;
    for (((t, h), o), m) in title.iter_mut().zip(host.iter_mut()).zip(old).zip(mask.iter_mut()) {
        let gpu = if (*h != *o) & (*t == *o) { 0xffu8 } else { 0 };
        let merged = (*t & !gpu) | (*h & gpu);
        *m = merged ^ *t;
        differs |= *m;
        *t = merged;
        *h = merged;
    }
    if differs == 0 {
        return None;
    }
    let mut i = 0;
    while i < n {
        while i < n && mask[i] == 0 {
            i += 1;
        }
        let start = i;
        while i < n && mask[i] != 0 {
            i += 1;
        }
        if i > start {
            runs.push((start, i - start));
        }
    }
    Some((first, n))
}

/// Since the last frame-rate line, of [`Shadow::merge_under_writes`]: pages
/// compared, changed runs put in, and their bytes.
static MERGED: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// Why [`Shadow::merge_under_writes`] last left the caller to wait.
static STALE_WHY: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn note_stale(why: usize) {
    STALE_WHY.store(why, Ordering::Relaxed);
}

/// [`STALE_WHY`] in words (diagnostics).
#[cfg(feature = "diagnostics")]
pub(crate) fn stale_why() -> &'static str {
    const WHY: [&str; 4] = [
        "the memory is not watched",
        "the journal does not say what was written",
        "more changed runs than a submission sends up",
        "no snapshot",
    ];
    WHY[STALE_WHY.load(Ordering::Relaxed).min(3)]
}

/// Since the last frame-rate line, where [`Shadow::merge_under_writes`]'s
/// time went, in ns: 0 asking which pages were written, 1 finding the
/// stale ones, 2 waiting for the locks, 3 the pages compared (the wall
/// time), 4 noting the runs; summed over threads: 5 reading title memory,
/// 6 comparing and putting in. Then 7 (unused), 8 stale pages, 9 calls.
pub(crate) static UNDER_NS: [AtomicU64; 10] = [const { AtomicU64::new(0) }; 10];

fn under_ns(i: usize, since: std::time::Instant) -> std::time::Instant {
    let now = std::time::Instant::now();
    UNDER_NS[i].fetch_add((now - since).as_nanos() as u64, Ordering::Relaxed);
    now
}

/// [`UNDER_NS`], taken.
pub(crate) fn under_stats() -> [u64; 10] {
    std::array::from_fn(|i| UNDER_NS[i].swap(0, Ordering::Relaxed))
}

/// [`MERGED`], taken.
pub(crate) fn merged_stats() -> [u64; 3] {
    [0, 1, 2].map(|i| MERGED[i].swap(0, Ordering::Relaxed))
}

/// Adds the time since it was made to [`UNDER_NS`]`[6]` when dropped.
struct Compared(std::time::Instant);

impl Drop for Compared {
    fn drop(&mut self) {
        under_ns(6, self.0);
    }
}

/// The runs of bytes where `now` differs from `was` (of one length), as
/// (offset, bytes) in order: every differing byte is in a run and no equal
/// byte is.
fn changed_runs(now: &[u8], was: &[u8]) -> Vec<(usize, usize)> {
    let n = now.len().min(was.len());
    let mut runs = Vec::new();
    let mut i = 0;
    while i < n {
        // Equal bytes, many at a time while they last (a page looked at
        // under a flight's writes hardly ever differs at all).
        while i + 512 <= n && now[i..i + 512] == was[i..i + 512] {
            i += 512;
        }
        while i + 8 <= n && now[i..i + 8] == was[i..i + 8] {
            i += 8;
        }
        while i < n && now[i] == was[i] {
            i += 1;
        }
        if i >= n {
            break;
        }
        let first = i;
        while i < n && now[i] != was[i] {
            i += 1;
        }
        runs.push((first, i - first));
    }
    runs
}

/// The kernel's page.
const KERNEL_PAGE: usize = 4096;

/// Whether a refresh looks if what counts as written is only what a landing
/// put there, and then sends nothing of it up (`VICTORY_LAND_ECHO=0`: it copies
/// and sends it, as before).
///
/// What shaders wrote comes down, goes into title memory and into the CPU's
/// side of the copy alike. In memory the kernel watches that is a write like
/// any other (it cannot be told as the guards' blocks can, `crate::land`), and
/// the next refresh of it copied the pages and sent them up again: in the dorm
/// 14.4 MB a frame of one copy of 90 MB whose 263 pieces were all fetched and
/// all sent, 28 of 72 MB over the bus in all (run submit-2). Only pages a
/// landing was noted in are compared, 4 KiB at a time: a page that is not the
/// same goes up whole, as before, and so does every other.
fn land_echo() -> bool {
    crate::toggled!("VICTORY_LAND_ECHO")
}

/// Since the last frame-rate line: bytes found to be a landing's alone, and
/// bytes a landing was noted in that were the title's too.
pub(crate) static ECHO: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Whether a refresh of a copy in the card's memory takes, and sends up,
/// the dwords that title memory changed and no others
/// (`VICTORY_SEND_CHANGED=1`; off: the 64 KiB pages written since it last
/// took them, whole). Off since run tile1-0: the room lost its indirect light
/// and geometry exploded; run bis1-0, the same but for this, was right.
///
/// A page that counts as written is mostly not the title's doing, or not all
/// of it: a word the title wrote, and around it what shaders wrote and the
/// title never touched; or only what the relay put into title memory, which
/// the card and the CPU's side hold already ([`Shadow::card_merge_relay`]).
/// Sent up whole, such a page goes over whatever the card holds there that
/// did not come back yet, or never will (a write through a view nobody
/// declared): what stood on the card, newer, was gone.
///
/// What it rests on: the CPU's side is what the card's side was last known
/// to hold, for every range no shader may be writing (no refresh touches
/// the others). So a dword goes up only if title memory holds another value
/// of it than the CPU's side, which then holds it too; title memory is read
/// once, what is compared is what is stored and sent. Dwords, not bytes:
/// no shader writes less, and the title's bytes beside a changed one in its
/// dword are those the CPU's side holds.
///
/// A page never taken goes up whole as long as no shader's write was ever
/// declared for the copy. After that it is compared as any other: the CPU's
/// side and the card's began as zeros, both.
///
/// What it cannot know: that the title wrote a value over the same value.
/// Where a shader wrote in between, through a view nobody declared, the
/// shader's stays.
fn send_changed() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        #[cfg(feature = "diagnostics")]
        if crate::copystat::mode() != 0 {
            return false;
        }
        std::env::var_os("VICTORY_SEND_CHANGED").is_some_and(|v| v != "0")
    })
}

/// Since the last frame-rate line, of pages taken into copies in the
/// card's memory ([`send_changed`]): bytes compared, those of pages that
/// held the same, bytes to go up, and the runs they are.
static SEND: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// [`SEND`], taken.
#[cfg(feature = "diagnostics")]
pub(crate) fn send_stats() -> [u64; 4] {
    [0, 1, 2, 3].map(|i| SEND[i].swap(0, Ordering::Relaxed))
}

/// Whether what shaders wrote goes into title memory run by run
/// ([`put_changed`]; on by default, `VICTORY_EXACT_BACK=0` writes the whole
/// stretch from the first changed byte to the last, as before).
pub(crate) fn exact_back() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_EXACT_BACK").is_none_or(|v| v != "0"))
}

/// Since the last frame-rate line: pieces put back by [`put_changed`], the
/// writes they took, the bytes written, and the pieces whose runs were
/// joined over bytes that had not changed.
pub(crate) static PUT: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// The dwords (of title memory from `at` on) where `merged` differs from
/// `title`, as (offset, bytes) in order, joined where at most `gap` bytes
/// lie between.
fn dword_runs(at: usize, title: &[u8], merged: &[u8], gap: usize) -> Vec<(usize, usize)> {
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for (from, n) in changed_runs(merged, title) {
        // Shaders write dwords: the whole of each.
        let lo = ((at + from) & !3).max(at) - at;
        let hi = ((at + from + n + 3) & !3).min(at + title.len()) - at;
        match runs.last_mut() {
            Some((a, len)) if lo <= *a + *len + gap => *len = (*len).max(hi - *a),
            _ => runs.push((lo, hi - lo)),
        }
    }
    runs
}

/// [`dword_runs`] with no gap, the fast way where it can be: a frame compares
/// 12 MB this way and finds 80,000 runs.
fn word_runs(at: usize, title: &[u8], held: &[u8]) -> Vec<(usize, usize)> {
    #[cfg(target_arch = "x86_64")]
    if title.len() == held.len() && std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: the processor has AVX2, and the lengths are the same.
        return unsafe { word_runs_avx2(at, title, held) };
    }
    dword_runs(at, title, held, 0)
}

/// # Safety
/// AVX2, and two slices of one length.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn word_runs_avx2(at: usize, title: &[u8], held: &[u8]) -> Vec<(usize, usize)> {
    use std::arch::x86_64::*;
    debug_assert_eq!(title.len(), held.len());
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut push = |from: usize, len: usize| match runs.last_mut() {
        Some((a, n)) if *a + *n == from => *n += len,
        _ => runs.push((from, len)),
    };
    let n = title.len();
    let prefix = ((4 - (at & 3)) & 3).min(n);
    if prefix > 0 && title[..prefix] != held[..prefix] {
        push(0, prefix);
    }
    let aligned_end = prefix + ((n - prefix) & !3);
    let mut i = prefix;
    while i + 256 <= aligned_end {
        let mut differ = 0u64;
        for k in 0..8 {
            // SAFETY: inside both slices (`i + 256 <= aligned_end <= n`).
            let same = unsafe {
                _mm256_cmpeq_epi32(
                    _mm256_loadu_si256(title.as_ptr().add(i + 32 * k).cast()),
                    _mm256_loadu_si256(held.as_ptr().add(i + 32 * k).cast()),
                )
            };
            differ |= u64::from(!_mm256_movemask_ps(_mm256_castsi256_ps(same)) as u8) << (8 * k);
        }
        while differ != 0 {
            let first = differ.trailing_zeros();
            let count = (differ >> first).trailing_ones();
            push(i + 4 * first as usize, 4 * count as usize);
            if first + count >= 64 {
                break;
            }
            differ &= u64::MAX << (first + count);
        }
        i += 256;
    }
    while i + 32 <= aligned_end {
        // SAFETY: inside both slices (`i + 32 <= aligned_end <= n`).
        let same = unsafe {
            _mm256_cmpeq_epi32(
                _mm256_loadu_si256(title.as_ptr().add(i).cast()),
                _mm256_loadu_si256(held.as_ptr().add(i).cast()),
            )
        };
        let mut differ = !_mm256_movemask_ps(_mm256_castsi256_ps(same)) as u8;
        while differ != 0 {
            let first = differ.trailing_zeros();
            let count = (differ >> first).trailing_ones();
            push(i + 4 * first as usize, 4 * count as usize);
            if first + count >= 8 {
                break;
            }
            differ &= u8::MAX << (first + count);
        }
        i += 32;
    }
    while i + 4 <= aligned_end {
        if title[i..i + 4] != held[i..i + 4] {
            push(i, 4);
        }
        i += 4;
    }
    if aligned_end < n && title[aligned_end..n] != held[aligned_end..n] {
        push(aligned_end, n - aligned_end);
    }
    runs
}

/// Whether [`Shadow::merge_under_writes`] puts the title's bytes into the
/// CPU's side of a copy in the card's memory as it compares, a block of 32
/// bytes where one of its words differs (`VICTORY_UNDER_FUSED=0`: the runs
/// are found first and copied one by one afterwards, as before). The same
/// bytes end up there: the words of a block that do not differ are equal.
/// In session play-1125-1724 a frame had 190,000 runs in 400 pages, each
/// a copy of a few bytes of its own.
fn under_fused() -> bool {
    crate::toggled!("VICTORY_UNDER_FUSED")
}

/// [`word_runs`], and every 32-byte block of `held` in which a word differs
/// takes `title`'s bytes (the words of the tail that differ, one by one):
/// `held` is `title` afterwards.
fn word_runs_put(at: usize, title: &[u8], held: &mut [u8]) -> Vec<(usize, usize)> {
    #[cfg(target_arch = "x86_64")]
    if title.len() == held.len() && std::arch::is_x86_feature_detected!("avx2") {
        // SAFETY: the processor has AVX2, and the lengths are the same.
        return unsafe { word_runs_put_avx2(at, title, held) };
    }
    let runs = dword_runs(at, title, held, 0);
    for &(r, m) in &runs {
        held[r..r + m].copy_from_slice(&title[r..r + m]);
    }
    runs
}

/// # Safety
/// AVX2, and two slices of one length.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn word_runs_put_avx2(at: usize, title: &[u8], held: &mut [u8]) -> Vec<(usize, usize)> {
    use std::arch::x86_64::*;
    debug_assert_eq!(title.len(), held.len());
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut push = |from: usize, len: usize| match runs.last_mut() {
        Some((a, n)) if *a + *n == from => *n += len,
        _ => runs.push((from, len)),
    };
    let n = title.len();
    let prefix = ((4 - (at & 3)) & 3).min(n);
    if prefix > 0 && title[..prefix] != held[..prefix] {
        held[..prefix].copy_from_slice(&title[..prefix]);
        push(0, prefix);
    }
    let aligned_end = prefix + ((n - prefix) & !3);
    let mut i = prefix;
    while i + 256 <= aligned_end {
        let mut differ = 0u64;
        for k in 0..8 {
            let off = i + 32 * k;
            // SAFETY: inside both slices (`i + 256 <= aligned_end <= n`).
            unsafe {
                let now = _mm256_loadu_si256(title.as_ptr().add(off).cast());
                let old = _mm256_loadu_si256(held.as_ptr().add(off).cast());
                let block = !_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(now, old))) as u8;
                if block != 0 {
                    _mm256_storeu_si256(held.as_mut_ptr().add(off).cast(), now);
                }
                differ |= u64::from(block) << (8 * k);
            }
        }
        while differ != 0 {
            let first = differ.trailing_zeros();
            let count = (differ >> first).trailing_ones();
            push(i + 4 * first as usize, 4 * count as usize);
            if first + count >= 64 {
                break;
            }
            differ &= u64::MAX << (first + count);
        }
        i += 256;
    }
    while i + 32 <= aligned_end {
        // SAFETY: inside both slices (`i + 32 <= aligned_end <= n`).
        let mut differ = unsafe {
            let now = _mm256_loadu_si256(title.as_ptr().add(i).cast());
            let old = _mm256_loadu_si256(held.as_ptr().add(i).cast());
            let block = !_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(now, old))) as u8;
            if block != 0 {
                _mm256_storeu_si256(held.as_mut_ptr().add(i).cast(), now);
            }
            block
        };
        while differ != 0 {
            let first = differ.trailing_zeros();
            let count = (differ >> first).trailing_ones();
            push(i + 4 * first as usize, 4 * count as usize);
            if first + count >= 8 {
                break;
            }
            differ &= u8::MAX << (first + count);
        }
        i += 32;
    }
    while i + 4 <= aligned_end {
        if title[i..i + 4] != held[i..i + 4] {
            held[i..i + 4].copy_from_slice(&title[i..i + 4]);
            push(i, 4);
        }
        i += 4;
    }
    if aligned_end < n && title[aligned_end..n] != held[aligned_end..n] {
        held[aligned_end..n].copy_from_slice(&title[aligned_end..n]);
        push(aligned_end, n - aligned_end);
    }
    runs
}

/// Applies one page chunk of [`Shadow::take`]: copies changed dwords from
/// `title` into `held` (or the whole slice when `whole`), returning the
/// changed byte runs relative to `0..title.len()`.
fn take_chunk(at: usize, title: &[u8], held: &mut [u8], whole: bool) -> Vec<(usize, usize)> {
    if whole {
        held.copy_from_slice(title);
        return if title.is_empty() { Vec::new() } else { vec![(0, title.len())] };
    }
    word_runs_put(at, title, held)
}

/// Whether the pages of [`Shadow::merge_under_writes`] are compared on several
/// threads (`VICTORY_UNDER_WIDE=0`: on the submitting one).
fn under_wide() -> bool {
    crate::toggled!("VICTORY_UNDER_WIDE")
}

/// What shaders wrote goes into title memory from `at` on: `title` is what
/// was read there, `merged` the same with their bytes put in. Only the
/// dwords that differ are written. The whole stretch from the first changed
/// byte to the last was: between the read and the write the title's own
/// threads, or data streamed in, may have written the bytes between (a heap
/// holds what shaders write beside what the CPU does), and those writes were
/// undone. Beside running title threads (the relay) that is a piece of
/// memory going back in time now and then: runs pool1, blame1 and seriess2
/// ended in the title decoding garbage.
///
/// shortcut: a piece with very many runs has them joined over small gaps,
/// where the old race is left.
fn put_changed(at: usize, title: &[u8], merged: &[u8]) -> usize {
    crate::watch::landing(at, title.len());
    if !exact_back() {
        crate::host::write_mapped(at as *mut u8, merged.as_ptr(), merged.len());
        return merged.len();
    }
    if crate::safecopy::on() {
        PUT[0].fetch_add(1, Ordering::Relaxed);
        PUT[2].fetch_add(merged.len() as u64, Ordering::Relaxed);
        let since = crate::watch::changes();
        // SAFETY: title memory; a page of it that is not there fails the
        // put, and the rest is put page by page.
        if unsafe { crate::safecopy::put_changed(at as *mut u8, title, merged) } {
            return merged.len();
        }
        let mut from = 0;
        while from < merged.len() {
            let n = (4096 - (at + from) % 4096).min(merged.len() - from);
            let (was, now) = (&title[from..from + n], &merged[from..from + n]);
            // SAFETY: as above.
            let put = || unsafe { crate::safecopy::put_changed((at + from) as *mut u8, was, now) };
            // A page that is away only while the runtime changes a mapping
            // beside it gets its bytes when that is done.
            if !put() && !crate::watch::again_after_change(since, put) {
                crate::watch::lost("what shaders wrote", at + from, n);
            }
            from += n;
        }
        return merged.len();
    }
    let mut runs = dword_runs(at, title, merged, 0);
    if runs.len() > 64 {
        runs = dword_runs(at, title, merged, 32);
        PUT[3].fetch_add(1, Ordering::Relaxed);
    }
    let mut written = 0;
    for &(from, n) in &runs {
        crate::host::write_mapped((at + from) as *mut u8, merged[from..].as_ptr(), n);
        written += n;
    }
    PUT[0].fetch_add(1, Ordering::Relaxed);
    PUT[1].fetch_add(runs.len() as u64, Ordering::Relaxed);
    PUT[2].fetch_add(written as u64, Ordering::Relaxed);
    written
}

/// For `crate::land`'s measurement: a piece as [`Shadow::card_merge_relay`]
/// lands it without `VICTORY_LAND`: `n` bytes at `now`, `held` what the CPU's
/// side held, both the caller's alone.
#[cfg(test)]
pub(crate) fn land_as_before(start: usize, now: usize, held: usize, n: usize, title: &mut Vec<u8>) {
    // SAFETY: the caller's.
    let (gpu, held) = unsafe { (std::slice::from_raw_parts(now as *const u8, n), std::slice::from_raw_parts_mut(held as *mut u8, n)) };
    let Some(first) = first_difference(gpu, held) else {
        return;
    };
    let last = last_difference(gpu, held).unwrap_or(first);
    let n = last + 1 - first;
    title.clear();
    title.resize(n, 0);
    assert!(crate::host::read_into((start + first) as *const u8, title));
    let was = title.clone();
    if merge_piece(&gpu[first..=last], &mut held[first..=last], title) {
        put_changed(start + first, &was, title);
    }
}

#[cfg(test)]
mod merge_tests {
    use super::{PAGE, PIECE, back_piece, changed_runs, dword_runs, put_changed, word_runs, word_runs_put};

    /// A piece put back takes the GPU's bytes where the copy changed and
    /// the title's everywhere else, reads and writes title memory only from
    /// the first change to the last, and its scratch holds that stretch and
    /// no more: not the 64 KiB per range it once was.
    #[test]
    fn a_piece_put_back_reads_only_its_stretch() {
        let old: Vec<u8> = (0..PIECE).map(|i| (i * 7 + i / 251) as u8).collect();
        // The title wrote 7 and 40,000 since the snapshot; the GPU wrote 5,
        // 9 and 40,000 (the GPU's byte wins where both wrote).
        let mut title_memory = old.clone();
        title_memory[7] ^= 0x11;
        title_memory[40_000] ^= 0x22;
        let mut host = old.clone();
        for at in [5, 9] {
            host[at] ^= 0x33;
        }
        let gpu = host.clone();
        let mut scratch = Vec::new();
        let (mut mask, mut runs) = (Vec::new(), Vec::new());
        let mut asked = None;
        let read = |first: usize, into: &mut [u8]| {
            asked = Some((first, into.len()));
            into.copy_from_slice(&title_memory[first..first + into.len()]);
            true
        };
        assert_eq!(back_piece(&mut host, &old, &mut scratch, &mut mask, &mut runs, read), Some((5, 5)));
        assert_eq!(asked, Some((5, 5)));
        let mut wanted = title_memory[5..10].to_vec();
        wanted[0] = gpu[5];
        wanted[4] = gpu[9];
        assert_eq!(scratch, wanted);
        assert_eq!(host[5..10], wanted);
        // Only the GPU's two bytes are written back (upstream #2009).
        assert_eq!(runs, [(0, 1), (4, 1)]);
        // The copy outside the stretch is left as it was.
        assert!(host[..5] == gpu[..5] && host[10..] == gpu[10..]);
        assert!(scratch.capacity() < PIECE / 2, "scratch of {} bytes for a stretch of 5", scratch.capacity());

        // Nothing differs: title memory is not read.
        let mut same = old.clone();
        assert_eq!(back_piece(&mut same, &old, &mut scratch, &mut mask, &mut runs, |_, _| unreachable!()), None);
        // Title memory not there: nothing to write, the copy unchanged.
        let mut host = gpu.clone();
        assert_eq!(back_piece(&mut host, &old, &mut scratch, &mut mask, &mut runs, |_, _| false), None);
        assert!(host == gpu);
        // The GPU wrote what the title holds already: nothing to write.
        let mut title_memory = old.clone();
        title_memory[5] = gpu[5];
        title_memory[9] = gpu[9];
        let mut host = gpu.clone();
        let read = |first: usize, into: &mut [u8]| {
            into.copy_from_slice(&title_memory[first..first + into.len()]);
            true
        };
        assert_eq!(back_piece(&mut host, &old, &mut scratch, &mut mask, &mut runs, read), None);
        // A whole piece changed: the scratch grows to it.
        let mut host: Vec<u8> = old.iter().map(|b| !b).collect();
        assert_eq!(back_piece(&mut host, &old, &mut scratch, &mut mask, &mut runs, |_, into| into.len() == PIECE), Some((0, PIECE)));
        assert_eq!(scratch.len(), PIECE);
    }

    /// A byte the shader and the title both changed while the GPU ran keeps
    /// the title's (the shader computed it from the copy's older data); a
    /// byte only the shader changed goes back, and only such bytes are in
    /// the runs to write (PalmVIDev e0b70476, upstream #2009).
    #[test]
    fn a_byte_both_changed_keeps_the_titles_and_only_gpu_bytes_are_written() {
        let old = vec![1u8, 2, 3, 4, 5, 6];
        let mut host = old.clone();
        host[1] = 20; // the GPU alone
        host[3] = 40; // both
        host[4] = 50; // the GPU alone
        let mut title_memory = old.clone();
        title_memory[3] = 44;
        title_memory[5] = 66; // the title alone
        let (mut scratch, mut mask, mut runs) = (Vec::new(), Vec::new(), Vec::new());
        let read = |first: usize, into: &mut [u8]| {
            into.copy_from_slice(&title_memory[first..first + into.len()]);
            true
        };
        assert_eq!(back_piece(&mut host, &old, &mut scratch, &mut mask, &mut runs, read), Some((1, 4)));
        assert_eq!(scratch, [20, 3, 44, 50], "the title's byte where both changed");
        assert_eq!(host[1..5], [20, 3, 44, 50], "the copy takes the result");
        assert_eq!(runs, [(0, 1), (3, 1)], "the GPU's bytes only");
    }


    /// What shaders wrote reaches a page that is away only while the runtime
    /// changes a mapping; a page that stays away loses it, and nothing else.
    #[test]
    fn a_put_waits_for_a_change_of_mapping() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        // SAFETY: a reservation of ours, nothing of it committed.
        let at = unsafe { VirtualAlloc(std::ptr::null(), 1 << 16, MEM_RESERVE, PAGE_NOACCESS) } as usize;
        assert_ne!(at, 0);
        put_changed(at, &[0; 64], &[3; 64]);
        let (began, wait) = std::sync::mpsc::channel();
        let change = std::thread::spawn(move || {
            let _changing = crate::watch::title_memory_remapping();
            began.send(()).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(50));
            // SAFETY: the reservation's first page.
            assert!(!unsafe { VirtualAlloc(at as _, 4096, MEM_COMMIT, PAGE_READWRITE) }.is_null());
        });
        wait.recv().unwrap();
        put_changed(at, &[0; 64], &[1; 64]);
        change.join().unwrap();
        // SAFETY: committed by now.
        assert_eq!(unsafe { std::slice::from_raw_parts(at as *const u8, 64) }, [1; 64]);
        // SAFETY: our reservation.
        unsafe { VirtualFree(at as _, 0, MEM_RELEASE) };
    }

    /// Whole dwords around every changed byte and no other, whatever the
    /// alignment; small gaps joined only when asked.
    #[test]
    fn dword_runs_hold_the_dwords_that_changed() {
        let title: Vec<u8> = (0..200u32).map(|i| (i * 7 + 3) as u8).collect();
        for at in [0x1000usize, 0x1001, 0x1002, 0x1003] {
            for changes in [vec![0], vec![199], vec![5, 6], vec![8, 12, 40, 41, 43, 120]] {
                let mut merged = title.clone();
                for &c in &changes {
                    merged[c] ^= 0x55;
                }
                let runs = dword_runs(at, &title, &merged, 0);
                let held: Vec<usize> = runs.iter().flat_map(|&(a, n)| a..a + n).collect();
                let mut expect: Vec<usize> = (0..200).filter(|i| changes.iter().any(|c| (at + c) / 4 == (at + i) / 4)).collect();
                expect.dedup();
                assert_eq!(held, expect, "{at:#x} {changes:?}");
                let joined = dword_runs(at, &title, &merged, 32);
                assert!(joined.len() <= runs.len());
                assert!(runs.iter().all(|r| joined.iter().any(|j| j.0 <= r.0 && r.0 + r.1 <= j.0 + j.1)));
            }
        }
        assert!(dword_runs(0, &title, &title, 0).is_empty());
    }

    /// The runs hold exactly the bytes that differ: putting them into the
    /// old bytes gives the new ones, and no run holds an equal byte.
    /// What looking at a page that did not change costs, against the look
    /// eight bytes at a time that it was (a frame has some 250 of them, on
    /// the submitting thread). Run with `--ignored --nocapture`.
    #[test]
    #[ignore]
    fn runs_of_equal_pages() {
        let (a, b) = (vec![7u8; 64 << 10], vec![7u8; 64 << 10]);
        let eights = || {
            let (now, was) = (std::hint::black_box(&a), std::hint::black_box(&b));
            let mut i = 0;
            while i + 8 <= now.len() && now[i..i + 8] == was[i..i + 8] {
                i += 8;
            }
            std::hint::black_box(i);
        };
        let best = |f: &dyn Fn()| {
            (0..500)
                .map(|_| {
                    let t = std::time::Instant::now();
                    f();
                    t.elapsed().as_secs_f64() * 1e6
                })
                .fold(f64::MAX, f64::min)
        };
        let runs = || drop(std::hint::black_box(changed_runs(std::hint::black_box(&a), std::hint::black_box(&b))));
        println!("64 KiB that are the same: {:.1} us by eights, {:.1} us now", best(&eights), best(&runs));
    }

    #[test]
    fn word_runs_put_finds_the_word_runs_and_puts_the_title_in() {
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for round in 0..300 {
            let n = [PAGE, 4096, 256, 260, 4, 0, 1000][round % 7];
            let held: Vec<u8> = (0..n).map(|_| next() as u8).collect();
            let mut title = held.clone();
            // Sparse words, dense stretches, single bytes of words.
            let changes = (next() % 600) as usize;
            for _ in 0..changes.min(n) {
                let at = (next() as usize) % n.max(1);
                let len = [1, 3, 4, 40, 300][(next() % 5) as usize].min(n - at);
                for b in &mut title[at..at + len] {
                    *b = b.wrapping_add(1 + (next() % 255) as u8);
                }
            }
            let want = word_runs(0, &title, &held);
            let mut merged = held.clone();
            let runs = word_runs_put(0, &title, &mut merged);
            assert_eq!(runs, want, "round {round}");
            assert_eq!(merged, title, "round {round}");
            // At most this many runs of more than 256 bytes (the page's
            // copies of their own, `merge_page_under_writes`).
            assert!(runs.iter().filter(|r| r.1 > 256).count() <= n / 260 + 1, "round {round}");
        }
    }

    #[test]
    fn word_runs_are_the_dword_runs() {
        let mut seed = 7u32;
        let mut random = move || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            seed >> 8
        };
        for round in 0..200 {
            let n = 4 * (random() as usize % 400);
            let was: Vec<u8> = (0..n).map(|_| random() as u8).collect();
            let mut now = was.clone();
            for _ in 0..random() % 200 {
                if n == 0 {
                    break;
                }
                let at = random() as usize % n;
                let len = (random() as usize % 24).min(n - at);
                for b in &mut now[at..at + len] {
                    *b = random() as u8;
                }
            }
            assert_eq!(word_runs(0x1000, &now, &was), dword_runs(0x1000, &now, &was, 0), "round {round}");
        }
    }

    #[test]
    fn changed_runs_are_exactly_the_bytes_that_differ() {
        let was: Vec<u8> = (0..300u32).map(|i| (i * 13 + 5) as u8).collect();
        for changes in [vec![], vec![0], vec![299], vec![7, 8, 9], vec![7, 9], vec![15, 16, 17, 23, 24, 64, 65, 66, 67, 68, 69, 70, 71, 72, 298, 299]] {
            let mut now = was.clone();
            for &at in &changes {
                now[at] ^= 0x41;
            }
            let runs = changed_runs(&now, &was);
            let mut put = was.clone();
            let mut held = Vec::new();
            for &(at, n) in &runs {
                assert!(n > 0);
                put[at..at + n].copy_from_slice(&now[at..at + n]);
                held.extend(at..at + n);
            }
            assert_eq!(put, now, "{changes:?}");
            assert_eq!(held, changes, "{changes:?}");
            // Runs do not touch: two that did would be one.
            assert!(runs.windows(2).all(|w| w[0].0 + w[0].1 < w[1].0), "{changes:?}");
        }
        assert!(changed_runs(&[], &[]).is_empty());
    }
}

/// Host copies alive (also those only a list, a descriptor pass or a
/// snapshot still holds) and the bytes of title memory they copy.
static LIVE: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

impl Drop for Shadow {
    fn drop(&mut self) {
        if self.place.is_some() {
            // Never counted with the copies (`Shadow::placed`).
            crate::inplace::gone(self.window, self.len);
            return;
        }
        LIVE[0].fetch_sub(1, Ordering::Relaxed);
        LIVE[1].fetch_sub(self.len, Ordering::Relaxed);
        if self.kept.get_mut().unwrap_or_else(|e| e.into_inner()).1 {
            KEPT_BYTES.fetch_sub(self.len, Ordering::Relaxed);
        }
        let window = self.window;
        if self.mapped {
            WINDOW_MAPPED_BYTES.fetch_sub(self.len, Ordering::Relaxed);
        }
        if let Some(card) = &mut self.card {
            if window {
                WINDOW_CARD_BYTES.fetch_sub(card.size, Ordering::Relaxed);
            } else {
                CARD_BYTES[0].fetch_sub(card.size, Ordering::Relaxed);
            }
            if card.before.get_mut().unwrap_or_else(|e| e.into_inner()).is_some() {
                CARD_BYTES[1].fetch_sub(card_hash_bytes(card.size), Ordering::Relaxed);
            }
        }
    }
}

/// Whether a copy's snapshot is kept from one submission to the next and
/// only the pieces that changed in between are taken again
/// (`VICTORY_KEEP_SNAPSHOTS=1`; off by default).
///
/// A snapshot is the copy's bytes before a submission, to find what its
/// shaders wrote. Taken anew for every submission it is 1,050 MB of copying
/// a frame (72-86 ms in gameplay, runs 61 and 62) for buffers of which
/// 100 MB changed. Kept, a piece of it is taken again only after something
/// changed the copy there: a refresh from title memory ([`Shadow::copy`]),
/// shader writes found after a submission, or the bytes put back then
/// ([`Shadow::copy_back`]).
///
/// What it assumes, and why it is not on by default: that nothing else
/// writes a copy. A shader that writes a buffer through a view we did not
/// declare as written (we declare the unordered-access views of shaders
/// that can store) changes the copy unseen; with a fresh snapshot that
/// write is simply never put back, with a kept one it would be taken for a
/// write of a later submission and put back then, over whatever the title
/// wrote there since.
fn keeps() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_KEEP_SNAPSHOTS").is_some_and(|v| v == "1"))
}

/// Bytes of the snapshots kept, and the most that may be.
static KEPT_BYTES: AtomicU64 = AtomicU64::new(0);
/// `VICTORY_KEEP_SNAPSHOTS_MB`, 256 unless set (it was 768: run 80, with
/// kept snapshots beside copies in the card's memory, ran the PC out of
/// memory in 27 s; kept snapshots are the PC's memory, all of it resident).
fn kept_most() -> u64 {
    static MOST: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MOST.get_or_init(|| {
        std::env::var("VICTORY_KEEP_SNAPSHOTS_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(256) << 20
    })
}

/// The 64 KiB pieces, cut at multiples of 64 KiB of title memory, that
/// hold offsets `[from, to)` of a copy of memory from `start` on
/// (`from < to`); piece 0 holds offset 0.
fn pieces_of(start: usize, from: usize, to: usize) -> std::ops::Range<usize> {
    let first = start / PIECE;
    (start + from) / PIECE - first..(start + to - 1) / PIECE - first + 1
}

/// The offsets that piece `piece` holds of a copy of `len` bytes of memory
/// from `start` on.
fn piece_at(start: usize, len: usize, piece: usize) -> (usize, usize) {
    let first = start / PIECE;
    let lo = ((first + piece) * PIECE).saturating_sub(start);
    let hi = ((first + piece + 1) * PIECE - start).min(len);
    (lo, hi)
}

#[cfg(test)]
mod piece_tests {
    use super::{PIECE, piece_at, pieces_of};

    /// The pieces cover a copy without gaps, each cut where the compare
    /// cuts (at multiples of 64 KiB of title memory).
    #[test]
    fn pieces_cover_the_copy_as_the_compare_cuts_it() {
        for start in [0usize, 1, 4096, PIECE - 1, PIECE, PIECE + 77, 5 * PIECE + 12345] {
            for len in [1usize, 100, PIECE - 1, PIECE, PIECE + 1, 3 * PIECE + 500] {
                let count = pieces_of(start, 0, len).end;
                let mut at = 0;
                for piece in 0..count {
                    let (lo, hi) = piece_at(start, len, piece);
                    assert_eq!(lo, at, "{start} {len} {piece}");
                    // As `Shadow::differing` cuts: to the next multiple.
                    let n = (PIECE - (start + at) % PIECE).min(len - at);
                    assert_eq!(hi, at + n, "{start} {len} {piece}");
                    assert_eq!(pieces_of(start, lo, hi), piece..piece + 1);
                    assert_eq!(pieces_of(start, hi - 1, hi), piece..piece + 1);
                    at = hi;
                }
                assert_eq!(at, len, "{start} {len}");
                assert_eq!(pieces_of(start, 0, len), 0..count);
            }
        }
    }
}

/// The pages of `[a, e)` (title addresses, `a < e`) that a copy whose
/// first page is `base` must take from title memory before it holds the
/// title's bytes: those it never took (`copied` 0), whether or not the
/// title wrote them since tracking began, and those `written` (as
/// `crate::watch::writes` gives it, by page in order) shows written after
/// the copy took them. As [`Shadow::copy_stale`] decides it.
///
/// [`Shadow::stale_under_writes`] and [`Shadow::merge_under_writes`] looked
/// at `written` alone, so a page of a range shaders may be writing that the
/// copy never took, and the title had not written lately, was neither
/// merged nor a reason to settle; the refresh leaves such ranges alone, and
/// shaders read zeros there (fxceless/victory-fork#206: missing geometry
/// with `VICTORY_LATE_WAIT`/`VICTORY_RELAY`).
#[cfg(test)]
fn stale_pages_in(base: usize, copied: &[u64], a: usize, e: usize, written: &[(usize, u64)]) -> Vec<usize> {
    (a / PAGE..e.div_ceil(PAGE))
        .filter(|&page| page >= base)
        .filter(|&page| match copied.get(page - base) {
            None => false,
            Some(0) => true,
            Some(&at) => written
                .binary_search_by_key(&page, |&(p, _)| p)
                .is_ok_and(|i| written[i].1 > at),
        })
        .collect()
}

/// `runs` (from, to, whether the copy held it before), cut in those of pages
/// held before (`pages`: page number and asking, in order) to the parts of
/// each page that were written since (`parts`: a bit for each 4 KiB).
fn by_parts(runs: Vec<(usize, usize, bool)>, pages: &[(usize, u64)], parts: &[u16]) -> Vec<(usize, usize, bool)> {
    const PART: usize = PAGE / 16;
    if parts.iter().all(|&p| p == u16::MAX) {
        return runs;
    }
    let mut out: Vec<(usize, usize, bool)> = Vec::with_capacity(runs.len() + pages.len());
    let mut pages = pages.iter().zip(parts).peekable();
    for (from, to, known) in runs {
        if !known {
            out.push((from, to, known));
            continue;
        }
        while let Some((&(page, _), &parts)) = pages.next_if(|((page, _), _)| page * PAGE < to) {
            for k in (0..16).filter(|k| parts >> k & 1 != 0) {
                let (a, b) = ((page * PAGE + k * PART).max(from), (page * PAGE + (k + 1) * PART).min(to));
                if a >= b {
                    continue;
                }
                match out.last_mut() {
                    Some((_, end, true)) if *end == a => *end = b,
                    _ => out.push((a, b, true)),
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod landing_tests {
    use super::{KERNEL_PAGE, Shadow};

    /// A page that only a landing wrote is not sent up again; one the title
    /// wrote too is, and so is every page no landing was noted in.
    #[test]
    fn what_a_landing_wrote_is_not_sent_up_again() {
        const PAGES: usize = 6;
        let room = vec![7u8; (PAGES + 1) * KERNEL_PAGE];
        let title = room.as_ptr() as usize;
        let title = title.next_multiple_of(KERNEL_PAGE);
        let mut side = vec![7u8; PAGES * KERNEL_PAGE];
        let copy = Shadow::unbacked(title as u64, side.as_mut_ptr(), side.len() as u64, true);
        let write = |page: usize| {
            // SAFETY: in `room`.
            unsafe { ((title + page * KERNEL_PAGE + 9) as *mut u8).write_volatile(8) }
        };
        let page = |from: usize, to: usize| (title + from * KERNEL_PAGE, title + to * KERNEL_PAGE);
        copy.note_landing(KERNEL_PAGE, 4 * KERNEL_PAGE);
        // Page 2: the title's too.
        write(2);
        let (from, to) = page(0, 6);
        copy.take_landed(from, to);
        let mut dirty = copy.take_dirty();
        dirty.sort_unstable();
        let part = |a: usize, b: usize| (a * KERNEL_PAGE, b * KERNEL_PAGE);
        assert_eq!(super::joined(dirty), [part(0, 1), part(2, 3), part(4, 6)]);
        assert_eq!(side[2 * KERNEL_PAGE + 9], 8);
        // Looked at: from here on as any other.
        let (from, to) = page(1, 2);
        copy.take_landed(from, to);
        assert_eq!(copy.take_dirty(), [part(1, 2)]);
        // In part: looked at again when the rest is.
        copy.note_landing(3 * KERNEL_PAGE, 3 * KERNEL_PAGE + 1);
        let (from, to) = page(3, 4);
        copy.take_landed(from, from + 100);
        copy.take_landed(from, to);
        assert_eq!(copy.take_dirty(), []);
        drop(room);
    }
}

#[cfg(test)]
mod parts_tests {
    use super::{PAGE, by_parts};

    #[test]
    fn runs_are_cut_to_the_parts_written() {
        const P: usize = PAGE / 16;
        let at = |page: usize, part: usize| page * PAGE + part * P;
        // Pages 10 and 11 again (from the middle of 10's part 1 on), 12
        // never taken, 14 again.
        let runs = vec![
            (at(10, 1) + 100, at(12, 0), true),
            (at(12, 0), at(13, 0), false),
            (at(14, 0), at(14, 9) + 7, true),
        ];
        let pages = [(10, 5), (11, 5), (14, 6)];
        assert_eq!(by_parts(runs.clone(), &pages, &[u16::MAX; 3]), runs);
        assert_eq!(
            by_parts(runs, &pages, &[0b1000_0000_0000_0011, 0b11, 0b10_0000_0000]),
            [
                (at(10, 1) + 100, at(10, 2), true),
                (at(10, 15), at(11, 2), true),
                (at(12, 0), at(13, 0), false),
                (at(14, 9), at(14, 9) + 7, true),
            ]
        );
    }
}

#[cfg(test)]
mod stale_tests {
    use super::*;

    #[test]
    fn pages_never_taken_are_stale_written_or_not() {
        let base = 0x100;
        let copied = [5, 0, 5, 5];
        let at = |page: usize| page * PAGE;
        // Page 0x101 was never taken; page 0x102 was written after it was
        // taken, page 0x103 before.
        let written = [(0x102, 6), (0x103, 4)];
        assert_eq!(stale_pages_in(base, &copied, at(0x100), at(0x104), &written), [0x101, 0x102]);
        // Part of a page counts as the page; pages outside the copy do not.
        assert_eq!(stale_pages_in(base, &copied, at(0x101) + 8, at(0x101) + 16, &[]), [0x101]);
        assert!(stale_pages_in(base, &copied, at(0x104), at(0x106), &[(0x104, 9)]).is_empty());
    }
}

/// Copies of windows outside buffers let go since the start, and their
/// bytes.
static LET_GO: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// For the memory line: the windows let go since the start and their
/// bytes, and of the bytes of the windows alive, how many lie between the
/// lowest and the highest address ever asked for in each.
#[cfg(feature = "diagnostics")]
pub(crate) fn windows() -> [u64; 3] {
    let asked: u64 = {
        let loose = LOOSE.read().unwrap_or_else(|e| e.into_inner());
        loose
            .values()
            .map(|s| {
                let low = s.asked[0].load(Ordering::Relaxed);
                let high = s.asked[1].load(Ordering::Relaxed);
                high.saturating_sub(low)
            })
            .sum()
    };
    [LET_GO[0].load(Ordering::Relaxed), LET_GO[1].load(Ordering::Relaxed), asked]
}

/// For the memory line, of the windows alive: the bytes of their pages
/// ever copied from title memory (the rest of each window is memory the
/// host zeroed and nothing read: what a window made no bigger than what is
/// read would not hold); the windows a descriptor slot shows and their
/// bytes (they stay until the title writes the slot again); the windows a
/// list or a pass over the heap holds and their bytes; the windows asked
/// for without a length (root views, which keep the 4 MiB grid) and their
/// bytes; and the window grid in bytes ([`window`]).
#[cfg(feature = "diagnostics")]
pub(crate) fn window_use() -> [u64; 8] {
    let mut out = [0u64; 8];
    let loose = LOOSE.read().unwrap_or_else(|e| e.into_inner());
    for s in loose.values() {
        let copied = s.copied.lock().unwrap_or_else(|e| e.into_inner());
        let pages = copied.copied_count() as u64;
        out[0] += (pages * PAGE as u64).min(s.len);
        if crate::host::references(&s.buffer) > 1 {
            out[1] += 1;
            out[2] += s.len;
        }
        if Arc::strong_count(s) > 1 {
            out[3] += 1;
            out[4] += s.len;
        }
        if s.unsized_asks.load(Ordering::Relaxed) > 0 {
            out[5] += 1;
            out[6] += s.len;
        }
    }
    out[7] = window();
    out
}

/// Optional byte cap for loose title-window copies, in MiB. Unset or `0`
/// keeps today's sweep. A positive value caps the copies only in the Watch
/// and High RAM bands; Calm is unchanged. The cap is on every loose copy's
/// bytes, held or idle (#450); only idle ones can go to meet it.
fn window_budget() -> Option<u64> {
    static MB: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *MB.get_or_init(|| {
        std::env::var("VICTORY_WINDOW_BUDGET_MB")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .filter(|&v| v > 0)
            .map(|v| v.saturating_mul(1 << 20))
    })
}

/// The bytes of title memory a loose copy holds a copy of: none for one in
/// place (`crate::inplace`), which is the title's own memory.
fn copy_bytes(shadow: &Shadow) -> u64 {
    if shadow.in_place() { 0 } else { shadow.len }
}

/// The unique bytes of every loose copy, eligible to go or not
/// ([`copy_bytes`]); one copy standing under several keys counts once.
fn loose_bytes<'a>(copies: impl Iterator<Item = &'a Arc<Shadow>>) -> u64 {
    let mut seen = std::collections::HashSet::new();
    copies.filter(|s| seen.insert(Arc::as_ptr(s) as usize)).map(|s| copy_bytes(s)).sum()
}

/// The eligible copies that go, by index. Entries are `(bytes, unique copy
/// identity, due)`, sorted oldest first. A `due` copy has been idle for as
/// long as the sweep asks without a budget and goes whatever the budget: a
/// budget never keeps more than none. The others go, oldest first, while
/// `total` (the unique bytes of every loose copy, [`loose_bytes`]) is over
/// `keep`. A copy standing under several keys goes under all of them and
/// counts once.
fn window_victims(entries: &[(u64, usize, bool)], total: u64, keep: u64) -> Vec<usize> {
    let mut left = total;
    let mut gone = std::collections::HashSet::new();
    for &(bytes, id, due) in entries {
        if due && gone.insert(id) {
            left = left.saturating_sub(bytes);
        }
    }
    for &(bytes, id, _) in entries {
        if left <= keep {
            break;
        }
        if bytes > 0 && gone.insert(id) {
            left = left.saturating_sub(bytes);
        }
    }
    (0..entries.len()).filter(|&i| gone.contains(&entries[i].1)).collect()
}

/// Kept between sweeps so one sample near a watermark does not toggle the
/// policy on the next submission.
static WINDOW_BAND: std::sync::Mutex<crate::host::RamBand> =
    std::sync::Mutex::new(crate::host::RamBand::Calm);

/// Lets go of the copies of windows outside buffers that nothing names any
/// more. They were never freed, and the world's streaming keeps asking for
/// new ones: 648 windows (2.8 GB) when the cutscene began, 974 (4.3 GB)
/// eight minutes later (run 24). A window goes when
/// - only the table of windows holds its copy: no list not yet reset reads
///   or writes it, no pass over the heap keeps its range, no submission is
///   comparing it;
/// - only the copy holds its host buffer: no descriptor slot views it (a
///   slot keeps the buffer it views, `HostHeap::kept`), so no draw can read
///   it through a slot made earlier;
/// - nothing asked for it for `IDLE_MS` and `IDLE_FRAMES` frames.
///
/// The next read of that memory makes a new copy and copies the window
/// again. Called before a submission starts, by the thread that submits.
pub(crate) fn let_go_idle() {
    const IDLE_FRAMES: u64 = 4;
    // Before the windows: their table may be empty while the pool is not.
    let_go_idle_snapshots();
    let budget = window_budget();
    // Without a budget (or in Calm) copies idle this long go; under a
    // budget they still do, and others idle for less go to meet it.
    let due_ms = crate::host::ram_idle_ms(crate::host::available_physical_ram());
    let mut idle_ms = due_ms;
    let mut keep = u64::MAX;
    if let Some(bytes) = budget
        && let Some((working, private, free, commit, total, limit)) = crate::host::ram_sample()
    {
        let mut band = WINDOW_BAND.lock().unwrap_or_else(|e| e.into_inner());
        *band = crate::host::ram_band(*band, working, private, free, commit, total, limit);
        match *band {
            crate::host::RamBand::Calm => {}
            crate::host::RamBand::Watch => {
                idle_ms = idle_ms.min(10_000);
                keep = bytes;
            }
            crate::host::RamBand::High => {
                idle_ms = idle_ms.min(2_000);
                keep = bytes / 2;
            }
        }
    }
    let gone: Vec<Arc<Shadow>> = {
        // Held from the look to the removal: `loose` hands a copy out
        // under the same lock, so none is handed out in between.
        let mut map = LOOSE.write().unwrap_or_else(|e| e.into_inner());
        if map.is_empty() {
            return;
        }
        let mut eligible: Vec<(u64, (u64, u64), usize, u64, bool)> = map
            .iter()
            .filter(|(_, s)| {
                Arc::strong_count(s) == 1
                    && crate::host::long_ago(s.used.load(Ordering::Relaxed), idle_ms, IDLE_FRAMES)
                    && crate::host::references(&s.buffer) == 1
            })
            .map(|(&key, s)| {
                let used = s.used.load(Ordering::Relaxed);
                let due = crate::host::long_ago(used, due_ms, IDLE_FRAMES);
                (used, key, Arc::as_ptr(s) as usize, copy_bytes(s), due)
            })
            .collect();
        eligible.sort_unstable_by_key(|&(used, ..)| used);
        let victims = if keep == u64::MAX {
            (0..eligible.len()).collect()
        } else {
            let entries: Vec<(u64, usize, bool)> = eligible
                .iter()
                .map(|&(_, _, id, bytes, due)| (bytes, id, due))
                .collect();
            window_victims(&entries, loose_bytes(map.values()), keep)
        };
        victims.into_iter().filter_map(|i| map.remove(&eligible[i].1)).collect()
    };
    if !gone.is_empty() {
        let (went, bytes) = (gone.len(), gone.iter().map(|s| s.len).sum::<u64>());
        crate::host::trail(|| {
            format!("sweep: {went} host copies of windows of title memory let go ({} KiB), held by no list and no slot", bytes >> 10)
        });
    }
    for shadow in gone {
        LET_GO[0].fetch_add(1, Ordering::Relaxed);
        LET_GO[1].fetch_add(shadow.len, Ordering::Relaxed);
        // Nothing runs now and nothing names it: freed at once unless a
        // submission is running after all.
        crate::host::retire(shadow);
    }
}

/// For the memory line: host copies alive and their bytes, of them the
/// windows of memory outside buffers and their bytes, and the snapshot
/// buffers kept for reuse and the bytes they reserve.
#[cfg(feature = "diagnostics")]
pub(crate) fn memory() -> [u64; 6] {
    let (windows, window_bytes) = {
        let loose = LOOSE.read().unwrap_or_else(|e| e.into_inner());
        // Each copy once: a window may stand under several keys.
        let mut seen = std::collections::HashSet::new();
        let bytes: u64 = loose
            .values()
            .filter(|s| seen.insert(Arc::as_ptr(s) as usize))
            .map(|s| s.len)
            .sum();
        (seen.len() as u64, bytes)
    };
    let (kept, kept_bytes) = {
        let pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
        (pool.buffers.len() as u64, pool.bytes as u64)
    };
    [
        LIVE[0].load(Ordering::Relaxed),
        LIVE[1].load(Ordering::Relaxed),
        windows,
        window_bytes,
        kept,
        kept_bytes,
    ]
}

// SAFETY: D3D12 resources are free-threaded; `data` is a persistent mapping
// written only on the submitting thread.
unsafe impl Send for Shadow {}
// SAFETY: as above.
unsafe impl Sync for Shadow {}

impl Shadow {
    /// The host GPU address of a title address inside the buffer.
    pub(crate) fn gpu(&self, address: u64) -> u64 {
        self.gpu + (address - self.start)
    }

    /// The title memory the copy is of: its start and length.
    pub(crate) fn span(&self) -> (u64, u64) {
        (self.start, self.len)
    }

    /// The offset of a title address in the copy, if it lies in it.
    pub(crate) fn offset_in(&self, address: u64) -> Option<u64> {
        address.checked_sub(self.start).filter(|&at| at < self.len)
    }

    /// Bytes from a title address to the buffer's end.
    pub(crate) fn left(&self, address: u64) -> u64 {
        (self.start + self.len).saturating_sub(address)
    }

    /// Whether the memory behind the copy turned out not to be write-watched:
    /// every refresh then copies its whole range.
    pub(crate) fn unwatched(&self) -> bool {
        self.watched.load(Ordering::Relaxed) == NOT_WATCHED
    }

    /// Whether a submission asks the kernel about the copy's ranges ahead:
    /// not of memory that is not watched, nor of view memory, which the
    /// kernel would refuse (and with it the ranges asked about together).
    pub(crate) fn asked_ahead(&self) -> bool {
        let watched = self.watched.load(Ordering::Relaxed);
        watched != NOT_WATCHED && (watched != GUARDED || !crate::watch::light())
    }

    /// The copy's own number (see the field).
    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    /// Whether it is no copy but title memory in place (`crate::inplace`).
    pub(crate) fn in_place(&self) -> bool {
        self.place.is_some()
    }

    /// Offset of a title address from the buffer's start.
    pub(crate) fn offset(&self, address: u64) -> u64 {
        address - self.start + self.place.as_ref().map_or(0, |place| place.1)
    }

    /// Title memory `[start, start + len)` in place: `placed` is a buffer
    /// over it in a heap opened on it. Shaped as a copy so that everything
    /// that binds, views or keeps a copy works on it unchanged; what
    /// copies does nothing ([`Shadow::update`], [`Shadow::gpu_writes`]).
    fn placed(placed: crate::inplace::Placed, start: u64, len: u64, window: bool) -> Shadow {
        crate::host::name(&placed.buffer, || format!("title memory {start:#x}+{len:#x} in place"));
        let mut shadow = Shadow::new(placed.buffer, start as *mut u8, placed.gpu, start, len);
        // Not a copy: not counted with them, and nothing asks what the
        // title wrote there.
        LIVE[0].fetch_sub(1, Ordering::Relaxed);
        LIVE[1].fetch_sub(len, Ordering::Relaxed);
        *shadow.watched.get_mut() = NOT_WATCHED;
        shadow.window = window;
        shadow.place = Some((placed.heap, placed.bias));
        shadow
    }

    /// Brings `len` bytes at `address` up to date when `list` is submitted,
    /// and keeps the copy alive until the list is reset.
    pub(crate) fn refresh(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.start + self.len - address);
        list.defer_refresh(self, address as usize, len as usize);
    }

    /// The bytes of the host buffer.
    pub(crate) fn width(&self) -> u64 {
        // SAFETY: a live buffer.
        *self.width.get_or_init(|| unsafe { self.buffer.GetDesc() }.Width)
    }

    fn new(buffer: ID3D12Resource, data: *mut u8, gpu: u64, start: u64, len: u64) -> Shadow {
        let pages = ((start + len).div_ceil(PAGE as u64) - start / PAGE as u64) as usize;
        LIVE[0].fetch_add(1, Ordering::Relaxed);
        LIVE[1].fetch_add(len, Ordering::Relaxed);
        Shadow {
            landings: Mutex::default(),
            landing: Default::default(),
            width: Default::default(),
            buffer,
            data,
            gpu,
            start,
            len,
            copied: Mutex::new(CopiedPages::new(pages)),
            ranged: Mutex::default(),
            watched: AtomicU8::new(0),
            pending: Mutex::default(),
            has_pending: AtomicBool::new(false),
            flights: Mutex::default(),
            before: Mutex::default(),
            merging: RwLock::default(),
            kept: Mutex::default(),
            used: AtomicU64::new(crate::host::moment()),
            asked: [AtomicU64::new(u64::MAX), AtomicU64::new(0)],
            unsized_asks: AtomicU64::new(0),
            written: Default::default(),
            id: {
                static NEXT: AtomicU64 = AtomicU64::new(1);
                NEXT.fetch_add(1, Ordering::Relaxed)
            },
            card: None,
            window: false,
            changes_only: Default::default(),
            mapped: false,
            place: None,
        }
    }

    /// `[address, address + len)` cut to the buffer, as offsets in the copy.
    fn inside(&self, address: usize, len: usize) -> (usize, usize) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let from = address.clamp(start, end);
        (from - start, (address + len).clamp(from, end) - start)
    }

    /// Shaders of `list` may write `[address, address + len)` of the copy
    /// (unordered access): after the list's submission has run, the range
    /// is copied back into title memory, where the title reads it; until
    /// then no refresh copies title memory over it. Call after the
    /// [`Shadow::refresh`] that gives the shaders the title's data.
    ///
    /// shortcut: what shaders wrote is found by keeping the range's bytes
    /// before the submission (a copy per declared range and submission);
    /// a GPU-side dirty mask would save that if big ranges make it slow.
    pub(crate) fn gpu_writes(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        // In place the shaders write title memory itself: no snapshot, no
        // compare, nothing to copy back, and no wait on its account.
        // shortcut: another host copy of the same bytes (a title buffer
        // inside this one with a copy of its own, a texture in it) is not
        // told: the card's writes are not seen by write watching.
        if self.place.is_some() {
            list.forget_placed_constants(address as usize, len.min(self.left(address)) as usize);
            return;
        }
        self.written.store(true, Ordering::Relaxed);
        let len = len.min(self.left(address));
        list.defer_gpu_write(self, address as usize, len as usize);
    }

    /// At submission, after the range's refresh: a submitted list's shaders
    /// write this range. Its bytes are kept to tell later what the GPU wrote:
    /// one snapshot per copy and submission, each byte taken once, when the
    /// first range holding it is declared (the world declares overlapping
    /// views of its 174 MB heap dozens of times, from different offsets: a
    /// snapshot grown to cover each was copied whole every time), in reused
    /// memory (fresh snapshots cost ~2.3 million page faults per world
    /// frame).
    pub(crate) fn writing(&self, address: usize, len: usize) {
        if self.place.is_some() {
            return;
        }
        let range = (address, address + len);
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        // With the copy-back on its own thread: counted for this submission
        // (once), whether or not an earlier one still holds the range.
        if relay() {
            self.flights.lock().unwrap_or_else(|e| e.into_inner()).declare(range, PREPARED.load(Ordering::Relaxed));
        }
        // Dispatches declare the same ranges thousands of times.
        if pending.contains(&range) {
            return;
        }
        // In the card's memory: no snapshot. What shaders wrote is found on
        // the card (`Host::card_before` reads the declared ranges from the
        // lists).
        if self.card.is_some() {
            pending.push(range);
            self.has_pending.store(true, Ordering::Release);
            return;
        }
        // The parts no declared range holds yet: their bytes are taken now,
        // just after the range's refresh (a part declared earlier has not
        // been refreshed since, and keeps the bytes taken then).
        let mut parts = vec![self.inside(address, len)];
        for &(a, e) in pending.iter() {
            let (a, e) = self.inside(a, e - a);
            let mut rest = Vec::with_capacity(parts.len() + 1);
            for (from, to) in parts {
                if e <= from || a >= to {
                    rest.push((from, to));
                    continue;
                }
                if from < a {
                    rest.push((from, a));
                }
                if e < to {
                    rest.push((e, to));
                }
            }
            parts = rest;
        }
        pending.push(range);
        self.has_pending.store(true, Ordering::Release);
        let taken: usize = parts.iter().map(|&(from, to)| to.saturating_sub(from)).sum();
        BACK_CLASS[usize::from(!self.window)].fetch_add(taken as u64, Ordering::Relaxed);
        let total = self.len as usize;
        let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let snapshot = before.get_or_insert_with(|| pooled(total));
        if keeps() {
            // Piece by piece, and only the pieces not known current. A
            // piece the part holds whole is current from now on; of a
            // piece it holds partly only the part's bytes are taken, as
            // without the switch, and it is taken again next time (the
            // rest of it may belong to a range declared earlier, whose
            // bytes stay as they were taken).
            let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
            if kept.0.is_empty() {
                kept.0 = vec![false; self.pieces(0, total.max(1)).end];
            }
            for (from, to) in parts {
                if from >= to {
                    continue;
                }
                for piece in self.pieces(from, to) {
                    if kept.0[piece] {
                        continue;
                    }
                    let (lo, hi) = self.piece(piece);
                    let whole = from <= lo && hi <= to;
                    let (lo, hi) = (lo.max(from), hi.min(to));
                    // SAFETY: as below; `hi <= total`.
                    unsafe {
                        std::ptr::copy_nonoverlapping(self.data.add(lo), snapshot.as_mut_ptr().add(lo), hi - lo);
                    }
                    kept.0[piece] = whole;
                }
            }
            return;
        }
        if wide_back() {
            // Noted, and taken with the submission's other snapshots when
            // its refreshes are done (`take_snapshots`): the range is
            // declared from here on, so nothing changes its bytes before
            // then. The snapshot's memory stays where it is until the
            // range is done with (`copy_back_done`).
            let mut snaps = SNAPS.lock().unwrap_or_else(|e| e.into_inner());
            let (data, snapshot) = (self.data as usize, snapshot.as_mut_ptr() as usize);
            snaps.extend(parts.into_iter().filter(|&(from, to)| from < to).map(|(from, to)| (data + from, snapshot + from, to - from)));
            return;
        }
        for (from, to) in parts {
            if from < to {
                // SAFETY: the copy's mapping and the snapshot's allocation
                // (`pooled`) both hold `total` bytes, and `to <= total`.
                unsafe {
                    copy_wide(self.data.add(from), snapshot.as_mut_ptr().add(from), to - from);
                }
            }
        }
    }

    /// After the submission ran: copies the bytes its shaders wrote back
    /// into title memory (unmapped pages left alone), and only those: the
    /// title's threads write the same buffers while the GPU runs (its
    /// command memory shares the big GPU heap with what shaders write), and
    /// putting back the copy's older bytes there broke the title's own
    /// records (a fence list read as -1, title+0x4e9b54a). Returns the bytes
    /// written. The range must lie in ranges declared with
    /// [`Shadow::writing`] and not yet done: `Host::execute` passes the
    /// union of a submission's ranges, each byte once, then calls
    /// [`Shadow::copy_back_done`].
    pub(crate) fn copy_back(&self, address: usize, len: usize) -> usize {
        self.copy_back_with(address, len, None)
    }

    /// [`Shadow::copy_back`]; `found`, if given, is the pieces of the range
    /// that differ from the snapshot, found beforehand ([`find_all`]).
    pub(crate) fn copy_back_with(&self, address: usize, len: usize, found: Option<Vec<(usize, usize)>>) -> usize {
        if self.place.is_some() {
            return 0;
        }
        let snapshot = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let mut written = 0;
        let (from, to) = self.inside(address, len);
        let start = self.start as usize;
        self.compared(from, to);
        if let Some(before) = &*snapshot
            && from < to
        {
            // SAFETY: `writing` set the snapshot's bytes of every declared
            // range, at the copy's offsets.
            let before = unsafe { std::slice::from_raw_parts(before.as_ptr().add(from), to - from) };
            // By 64 KiB piece: the copy compared with its bytes before the
            // submission (on several threads: gigabytes a frame, bound by
            // memory speed), title memory read only where the GPU wrote.
            #[cfg(feature = "diagnostics")]
            let scanning = std::time::Instant::now();
            let differing = match found {
                Some(found) => found,
                None => self.differing(from, to, before),
            };
            #[cfg(feature = "diagnostics")]
            let syncing = {
                use crate::copystat::{C, add};
                add(C::BackScanUs, scanning.elapsed().as_micros() as u64);
                add(C::BackDiffer, differing.len() as u64);
                std::time::Instant::now()
            };
            // Where the copy changes (the GPU wrote, and below the title's
            // bytes come in), a kept snapshot is no longer current.
            for &(at, n) in &differing {
                self.changed(at, at + n);
            }
            // Sized to each stretch read (`back_piece`): a zeroed 64 KiB per
            // range, most of whose ranges differ nowhere, was most of the
            // work for small ranges.
            let (mut title, mut mask, mut runs) = (Vec::new(), Vec::new(), Vec::new());
            for (at, n) in differing {
                // SAFETY: inside the mapping; only the submitting thread
                // writes it.
                let host = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), n) };
                let old = &before[at - from..at - from + n];
                let read = |first: usize, title: &mut [u8]| crate::host::read_into((start + at + first) as *const u8, title);
                let Some((first, n)) = back_piece(host, old, &mut title, &mut mask, &mut runs, read) else {
                    continue;
                };
                // Title memory as read, beside it with the card's bytes put
                // in: only the dwords that differ are written.
                let was: Vec<u8> = title.iter().zip(&mask).map(|(now, changed)| now ^ changed).collect();
                if !runs.is_empty() {
                    written += put_changed(start + at + first, &was[..n], &title[..n]);
                }
            }
            // Our write shows as a title write at the copy's next refresh,
            // which copies those pages once more (some dozens of MB a
            // frame). Asking about the whole range here, to note them
            // current, cost more: 19-31 askings a frame over 110-174 MB of
            // view memory, each a walk of its guards.
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::BackSyncUs, syncing.elapsed().as_micros() as u64);
        }
        written
    }

    /// Offsets `[from, to)` of the copy are compared with the snapshot: for
    /// the frame line's counts.
    fn compared(&self, from: usize, to: usize) {
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::BackCompared, (to - from) as u64);
        let (bytes, ranges) = if self.window { (2, 5) } else { (3, 6) };
        BACK_CLASS[bytes].fetch_add((to - from) as u64, Ordering::Relaxed);
        BACK_CLASS[ranges].fetch_add(1, Ordering::Relaxed);
        if !self.window && self.len >= 1 << 20 {
            BACK_CLASS[4].fetch_add((to - from) as u64, Ordering::Relaxed);
        }
    }

    /// [`relay`]: the declared range `[address, address + len)` of a copy
    /// in the PC's memory is about to be looked through
    /// (`crate::shadow::find_all`) by the thread that settles: counted as
    /// [`Shadow::copy_back_with`] counts it.
    pub(crate) fn relay_compared(&self, address: usize, len: usize) {
        let (from, to) = self.inside(address, len);
        self.compared(from, to);
    }

    /// [`relay`], for a copy in the PC's memory: offsets `[at, at + n)`
    /// are a 64 KiB piece of a declared range that differed from the
    /// snapshot when it was looked through. What the card wrote there
    /// since the snapshot was last brought up goes into title memory, from
    /// the first to the last such byte, and into the snapshot, so that the
    /// next flight's merge finds only what the card writes from now on.
    /// The copy is read once and never written: a later flight's shaders
    /// may be writing it at this moment. Under the snapshot's lock, which
    /// [`Shadow::merge_under_writes`] holds while it reads title memory
    /// and puts the title's bytes into the copy and the snapshot: the two
    /// never see each other half done. `now` and `title` are scratch
    /// memory. Returns the bytes written to title memory.
    pub(crate) fn relay_merge(&self, at: usize, n: usize, now: &mut Vec<u8>, title: &mut Vec<u8>) -> usize {
        let start = self.start as usize;
        if self.card.is_some() || n == 0 || at + n > self.len as usize {
            return 0;
        }
        let mut snapshot = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let Some(before) = snapshot.as_mut() else {
            return 0;
        };
        now.clear();
        now.reserve(n);
        // SAFETY: inside the copy's mapping (checked above), read into
        // room just reserved; the card may write the mapping meanwhile, and
        // each byte read is one it held at some moment.
        unsafe {
            std::ptr::copy_nonoverlapping(self.data.add(at), now.as_mut_ptr(), n);
            now.set_len(n);
        }
        // SAFETY: the snapshot's allocation holds the copy's length
        // (`writing`), and these offsets are in a range declared and not
        // done with, whose bytes it holds; nothing else touches them
        // without the lock held here.
        let held = unsafe { std::slice::from_raw_parts_mut(before.as_mut_ptr().add(at), n) };
        let Some(first) = first_difference(now, held) else {
            return 0;
        };
        let last = last_difference(now, held).unwrap_or(first);
        let (at, n) = (at + first, last + 1 - first);
        let (now, held) = (&now[first..=last], &mut held[first..=last]);
        title.clear();
        title.resize(n, 0);
        // Title memory not there: looked at again when the next flight
        // that declared the range lands (the snapshot stays as it is).
        if !crate::host::read_into((start + at) as *const u8, title) {
            return 0;
        }
        let was = title.clone();
        if !merge_piece(now, held, title) {
            return 0;
        }
        put_changed(start + at, &was, title)
    }

    /// The 64 KiB pieces (cut at multiples of 64 KiB of title memory) of
    /// `[from, to)` of the copy that differ from `before` (the bytes from
    /// `from` on), as (offset, bytes) in order; compared on several threads
    /// when the range is big.
    fn differing(&self, from: usize, to: usize, before: &[u8]) -> Vec<(usize, usize)> {
        let (start, data) = (self.start as usize, self.data as usize);
        let scan = |a: usize, b: usize| -> Vec<(usize, usize)> {
            let mut found = Vec::new();
            let mut at = a;
            while at < b {
                let n = (PIECE - (start + at) % PIECE).min(b - at);
                // SAFETY: inside the mapping, which nothing writes while a
                // submission's results are compared.
                let host = unsafe { std::slice::from_raw_parts((data + at) as *const u8, n) };
                if host != &before[at - from..at - from + n] {
                    found.push((at, n));
                }
                at += n;
            }
            found
        };
        const PER_THREAD: usize = 16 << 20;
        let threads = ((to - from) / PER_THREAD).clamp(1, 8);
        if threads == 1 {
            return scan(from, to);
        }
        // Cut on piece boundaries, so that every thread sees whole pieces.
        let cut = |i: usize| {
            if i == threads {
                return to;
            }
            let at = from + (to - from) / threads * i;
            // The first piece can begin before the copy.
            at.saturating_sub((start + at) % PIECE).max(from)
        };
        let scan = &scan;
        std::thread::scope(|s| {
            let parts: Vec<_> = (0..threads)
                .map(|i| {
                    let (a, b) = (cut(i), cut(i + 1));
                    s.spawn(move || scan(a, b))
                })
                .collect();
            parts
                .into_iter()
                .flat_map(|part| match part.join() {
                    Ok(found) => found,
                    Err(panic) => std::panic::resume_unwind(panic),
                })
                .collect()
        })
    }

    /// After [`Shadow::copy_back`] of a submission's ranges: every submitted
    /// step that declared one of `ranges` (`(address, end)`, as declared) is
    /// done with it (one submission runs at a time), and the snapshot with
    /// the last range.
    ///
    /// With [`relay`] several submissions may hold a range: `ranges` are
    /// those of one of them (each once), and a range is done with when the
    /// last submission that declared it is ([`Flights`]).
    pub(crate) fn copy_back_done(&self, ranges: &[(usize, usize)]) {
        if self.place.is_some() {
            return;
        }
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        if relay() {
            let free = self.flights.lock().unwrap_or_else(|e| e.into_inner()).done(ranges);
            pending.retain(|r| free.binary_search(r).is_err());
        } else {
            pending.retain(|r| !ranges.contains(r));
        }
        self.has_pending.store(!pending.is_empty(), Ordering::Release);
        if !pending.is_empty() || self.card.is_some() {
            return;
        }
        if keeps() {
            // Kept for the next submission, if it is worth it (a megabyte
            // or more) and there is room.
            let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
            if !kept.1 && self.len >= 1 << 20 {
                if KEPT_BYTES.fetch_add(self.len, Ordering::Relaxed) + self.len <= kept_most() {
                    kept.1 = true;
                } else {
                    KEPT_BYTES.fetch_sub(self.len, Ordering::Relaxed);
                }
            }
            if kept.1 {
                return;
            }
            // Given back: nothing of it is current next time. (The lock
            // on the pieces ends here: `copy_back` takes it while it holds
            // the snapshot's.)
            kept.0.clear();
        }
        if let Some(bytes) = self.before.lock().unwrap_or_else(|e| e.into_inner()).take() {
            give_back(bytes);
        }
    }

    /// The 64 KiB pieces (cut at multiples of 64 KiB of title memory, as
    /// [`Shadow::differing`] cuts them) that hold offsets `[from, to)` of
    /// the copy; `from < to`.
    fn pieces(&self, from: usize, to: usize) -> std::ops::Range<usize> {
        pieces_of(self.start as usize, from, to)
    }

    /// The offsets in the copy that piece `piece` holds.
    fn piece(&self, piece: usize) -> (usize, usize) {
        piece_at(self.start as usize, self.len as usize, piece)
    }

    /// The copy's bytes at offsets `[from, to)` change: a kept snapshot is
    /// not current there any more.
    fn changed(&self, from: usize, to: usize) {
        if !keeps() || from >= to {
            return;
        }
        let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
        if kept.0.is_empty() {
            return;
        }
        for piece in self.pieces(from, to) {
            kept.0[piece] = false;
        }
    }

    /// Whether shaders of a submission not yet copied back may be writing
    /// the copy.
    pub(crate) fn writes_pending(&self) -> bool {
        self.has_pending.load(Ordering::Acquire)
    }

    /// Whether one of those submissions may be writing in title memory
    /// `[from, to)`.
    pub(crate) fn writes_pending_in(&self, from: usize, to: usize) -> bool {
        self.has_pending.load(Ordering::Acquire)
            && self.pending.lock().unwrap_or_else(|e| e.into_inner()).iter().any(|&(a, e)| a < to && e > from)
    }

    /// Whether [`Shadow::update`] of `[address, address + len)` would leave
    /// title bytes behind: a range shaders may be writing holds a page the
    /// title wrote since the copy last took it. The answer comes from the
    /// write-watch journal alone, as the submission's one asking left it
    /// (`crate::watch::asked`); where the journal cannot say (memory not
    /// write-watched, a range not asked about) it is yes. A page the copy
    /// never took whole counts as written if the title ever wrote it.
    ///
    /// For submissions sent to the card without a wait
    /// (`crate::host::late`): the ranges are those of flights in the air,
    /// which are then settled before this refresh.
    pub(crate) fn stale_under_writes(&self, address: usize, len: usize) -> bool {
        if !self.has_pending.load(Ordering::Acquire) {
            return false;
        }
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return false;
        }
        let busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter(|&&(a, e)| a < to && e > from)
            .map(|&(a, e)| (a.max(from), e.min(to)))
            .collect();
        if busy.is_empty() {
            return false;
        }
        if self.watched.load(Ordering::Relaxed) == NOT_WATCHED {
            return true;
        }
        let base = start / PAGE;
        let copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        busy.into_iter().any(|(a, e)| match crate::watch::asked(a, e) {
            Some((_, written)) => copied.has_stale_in(base, a, e, &written),
            None => true,
        })
    }

    /// Where [`Shadow::stale_under_writes`] says yes: puts the bytes the
    /// title changed into the copy, in the ranges shaders may be writing,
    /// and only those bytes, so that what the shaders write beside them
    /// stays (`crate::host::late_exact`). For each page of such a range
    /// that the title wrote since the copy last took it, title memory as
    /// it is now is compared with what the copy held when the range was
    /// given to the shaders: the copy's CPU side for a copy in the card's
    /// memory (the changed runs go up before the next submission, each a
    /// copy of its own bytes, `runs_left` of them at most), the snapshot
    /// for the others (the changed bytes go into the copy and into the
    /// snapshot, so the compare after the card ran takes them for the
    /// title's and not the shaders').
    ///
    /// Whether the caller must still settle the flights: the journal
    /// cannot say what was written, a snapshot is missing, or `runs_left`
    /// ran out. Pages done before that stay done.
    pub(crate) fn merge_under_writes(&self, address: usize, len: usize, runs_left: &mut usize) -> bool {
        if !self.has_pending.load(Ordering::Acquire) {
            return false;
        }
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return false;
        }
        let busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter(|&&(a, e)| a < to && e > from)
            .map(|&(a, e)| (a.max(from), e.min(to)))
            .collect();
        if busy.is_empty() {
            return false;
        }
        if self.watched.load(Ordering::Relaxed) == NOT_WATCHED {
            note_stale(0);
            return true;
        }
        // The declared ranges overlap (views of the title's whole heap from
        // dozens of offsets): as one range each page is asked about and
        // compared once. Every byte of it is in a declared range, and of a
        // copy in the PC's memory in the snapshot.
        let busy = if crate::watch::light() { joined(busy) } else { busy };
        let base = start / PAGE;
        UNDER_NS[9].fetch_add(1, Ordering::Relaxed);
        for (a, e) in busy {
            let t = std::time::Instant::now();
            // As the refresh itself asks (`update_free`): the journal if
            // the submission's asking covered the range, else the kernel,
            // else the guards. The journal alone said nothing for copies
            // of view memory, which is guarded and not write-watched, and
            // those were all of run 99's cases: the flights were settled
            // 4-5 times a frame for "0 pages looked at".
            let Some((now, written)) = crate::watch::writes(a, e, true) else {
                note_stale(1);
                return true;
            };
            let t = under_ns(0, t);
            let stale: Vec<usize> = {
                let copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
                copied.stale_pages_in(base, a, e, &written)
            };
            UNDER_NS[8].fetch_add(stale.len() as u64, Ordering::Relaxed);
            let t = under_ns(1, t);
            if stale.is_empty() {
                continue;
            }
            // The lock of the copy's reference (the snapshot, or the
            // CPU's side of a copy in the card's memory), from before
            // title memory is read until the title's bytes are in: with
            // the copy-back on its own thread ([`relay`]) that thread
            // puts what the card wrote into title memory and into the
            // reference under the same lock, and a page read before
            // such a merge and compared after it would take the card's
            // bytes for the title's.
            let merging = self.merging.write().unwrap_or_else(|e| e.into_inner());
            let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
            let held = if self.card.is_some() {
                self.data as usize
            } else {
                before.as_mut().map_or(0, |snapshot| snapshot.as_mut_ptr() as usize)
            };
            if held == 0 {
                note_stale(3);
                return true;
            }
            let t = under_ns(2, t);
            // A page's part in the range, compared and put in: the runs it
            // changed; none for a page that is not there (left as it is, and
            // looked at again next time) or that had too many.
            type Done = Option<Vec<(usize, usize)>>;
            let left = std::sync::atomic::AtomicI64::new(*runs_left as i64);
            let over = std::sync::atomic::AtomicBool::new(false);
            let done: Vec<Mutex<Done>> = stale.iter().map(|_| Mutex::new(None)).collect();
            let page_of = |i: usize| -> (usize, usize) { ((stale[i] * PAGE).max(a), ((stale[i] + 1) * PAGE).min(e)) };
            let one = |i: usize| {
                let (lo, hi) = page_of(i);
                if lo < hi && !over.load(Ordering::Relaxed) {
                    // SAFETY: the reference holds the copy's bytes at the
                    // copy's offsets, and these are in a declared range; the
                    // locks held above keep out what else writes them, and
                    // no two pages are the same.
                    let runs = unsafe { self.merge_page_under_writes(lo, hi - lo, held as *mut u8, &left) };
                    match runs {
                        Ok(runs) => *done[i].lock().unwrap_or_else(|e| e.into_inner()) = runs,
                        Err(()) => over.store(true, Ordering::Relaxed),
                    }
                }
            };
            // (Not on the threads of `wide`: the thread that waits there
            // takes whatever work comes, and a landing's takes the lock.)
            if stale.len() > 2 && under_wide() {
                crate::land::each_under(stale.len(), &one);
            } else {
                (0..stale.len()).for_each(one);
            }
            let t = under_ns(3, t);
            *runs_left = left.into_inner().max(0) as usize;
            let mut taken = Vec::new();
            // The runs of all the pages, noted at once (`card.dirty` is
            // sorted and joined when it is taken).
            let mut changed: Vec<(usize, usize)> = Vec::new();
            let (mut runs_in, mut bytes_in) = (0u64, 0u64);
            for (i, runs) in done.into_iter().enumerate() {
                let Some(mut runs) = runs.into_inner().unwrap_or_else(|e| e.into_inner()) else {
                    continue;
                };
                let (lo, hi) = ((stale[i] * PAGE).max(a), ((stale[i] + 1) * PAGE).min(e));
                runs_in += runs.len() as u64;
                bytes_in += runs.iter().map(|r| r.1 as u64).sum::<u64>();
                if self.card.is_some() {
                    let page_offset = lo - start;
                    for (from, to) in &mut runs {
                        *from += page_offset;
                        *to += *from;
                    }
                    if changed.is_empty() {
                        changed = runs;
                    } else {
                        changed.append(&mut runs);
                    }
                }
                // Taken, if the range holds all of the page that is in the
                // copy; else the rest of it is the refresh's, which notes
                // only whole pages, and the page is looked at again.
                if lo < hi && (stale[i] * PAGE).max(start) >= a && ((stale[i] + 1) * PAGE).min(end) <= e {
                    taken.push(stale[i]);
                }
            }
            MERGED[1].fetch_add(runs_in, Ordering::Relaxed);
            MERGED[2].fetch_add(bytes_in, Ordering::Relaxed);
            self.card_changed_runs(changed.into_iter());
            drop(before);
            drop(merging);
            {
                let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
                for page in taken {
                    copied.set(page - base, now);
                }
            }
            under_ns(4, t);
            if over.into_inner() {
                note_stale(2);
                return true;
            }
        }
        false
    }

    /// Of [`Shadow::merge_under_writes`]: title memory `[lo, lo + n)`, in one
    /// page, compared with the reference, which begins at `held`, and put in.
    /// The runs that changed, none if it cannot be read, and an error if
    /// there were more to copy one by one than `left`.
    ///
    /// # Safety
    /// The reference holds the copy's length, and nothing else writes these
    /// bytes of it during the call.
    unsafe fn merge_page_under_writes(
        &self,
        lo: usize,
        n: usize,
        held: *mut u8,
        left: &std::sync::atomic::AtomicI64,
    ) -> Result<Option<Vec<(usize, usize)>>, ()> {
        thread_local! {
            static TITLE: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
        }
        TITLE.with_borrow_mut(|title| {
            title.resize(PAGE, 0);
            let title = &mut title[..n];
            let t = std::time::Instant::now();
            if !crate::host::read_into(lo as *const u8, title) {
                return Ok(None);
            }
            let t = under_ns(5, t);
            let _compared = Compared(t);
            MERGED[0].fetch_add(1, Ordering::Relaxed);
            let at = lo - self.start as usize;
            // SAFETY: the caller's.
            let held = unsafe { std::slice::from_raw_parts_mut(held.add(at), n) };
            if self.card.is_some() {
                // Whole words go up as pairs, all of them in one
                // dispatch: only what is a copy of its own counts.
                // Compared and put in, in one pass ([`under_fused`]), where
                // the most copies of their own the page can have are left
                // to take: they are taken first, and what is not used is
                // given back. (Whole words at word offsets of the copy: a
                // copy of its own is a run of more than 256 bytes.)
                let most = (n / 260 + 1) as i64;
                let fused = crate::host::words_under()
                    && under_fused()
                    && !crate::host::upcheck::on()
                    && at.is_multiple_of(4)
                    && lo.is_multiple_of(4)
                    && n.is_multiple_of(4)
                    && {
                        if left.fetch_sub(most, Ordering::Relaxed) >= most {
                            true
                        } else {
                            left.fetch_add(most, Ordering::Relaxed);
                            false
                        }
                    };
                let runs = if fused {
                    word_runs_put(lo, title, held)
                } else if crate::host::words_under() {
                    word_runs(lo, title, held)
                } else {
                    changed_runs(title, held)
                };
                let own = |&&(r, m): &&(usize, usize)| !crate::host::words_under() || !crate::host::goes_as_words(at + r, m);
                let copies = runs.iter().filter(own).count() as i64;
                if fused {
                    debug_assert!(copies <= most);
                    left.fetch_add(most - copies, Ordering::Relaxed);
                    return Ok(Some(runs));
                }
                if copies > 0 && left.fetch_sub(copies, Ordering::Relaxed) < copies {
                    return Err(());
                }
                let was = (!runs.is_empty() && crate::host::upcheck::on()).then(|| held.to_vec());
                if !fused {
                    for &(r, m) in &runs {
                        held[r..r + m].copy_from_slice(&title[r..r + m]);
                    }
                }
                if let Some(was) = was {
                    crate::host::upcheck::note(self.id, at, &was, held);
                }
                return Ok(Some(runs));
            }
            let runs = changed_runs(title, held);
            for &(r, m) in &runs {
                // SAFETY: inside the copy's mapping. The card may be
                // writing the copy beside these bytes; they are
                // written here and nothing around them.
                unsafe { std::ptr::copy_nonoverlapping(title.as_ptr().add(r), self.data.add(at + r), m) };
                held[r..r + m].copy_from_slice(&title[r..r + m]);
            }
            Ok(Some(runs))
        })
    }

    /// At submission: copies what changed of `[address, address + len)`
    /// since it was last copied (all of it if the memory is not
    /// write-watched). Returns the bytes copied.
    pub(crate) fn update(&self, address: usize, len: usize) -> usize {
        // In place: the card reads title memory itself.
        if self.place.is_some() {
            return 0;
        }
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return 0;
        }
        if !self.has_pending.load(Ordering::Acquire) {
            return self.update_free(from, to);
        }
        let mut busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .copied()
            .filter(|&(a, e)| a < to && e > from)
            .collect();
        if !busy.is_empty() {
            // Around ranges shaders are writing: the gaps, each as below
            // (in the world most submissions have such ranges; copying the
            // gaps whole cost ~1 s per frame).
            busy.sort_unstable();
            let (mut at, mut bytes) = (from, 0);
            for (a, e) in busy {
                if a > at {
                    bytes += self.update_free(at, a);
                }
                at = at.max(e);
            }
            if at < to {
                bytes += self.update_free(at, to);
            }
            return bytes;
        }
        self.update_free(from, to)
    }

    /// Whether [`Shadow::update`] of `[address, address + len)` would copy
    /// nothing, and knows it without asking anything
    /// (`crate::watch::nothing_new`). `false` if it would have to ask.
    pub(crate) fn nothing_new(&self, address: usize, len: usize) -> bool {
        if self.place.is_some() {
            return true;
        }
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return true;
        }
        if !matches!(self.watched.load(Ordering::Relaxed), 1 | GUARDED) {
            return false;
        }
        // Around ranges shaders are writing the refresh asks gap by gap.
        if self.has_pending.load(Ordering::Acquire)
            && self.pending.lock().unwrap_or_else(|e| e.into_inner()).iter().any(|&(a, e)| a < to && e > from)
        {
            return false;
        }
        let copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        let ranged = self.ranged(from, to);
        let base = start / PAGE;
        let same = (ranged != 0 || copied.range_all_copied(from / PAGE - base, (to - 1) / PAGE - base))
            && crate::watch::nothing_new(from, to, base, &copied.at, ranged);
        #[cfg(feature = "diagnostics")]
        if same {
            use crate::copystat::{C, add};
            add(C::Checked, (to - from) as u64);
            add(C::Unchanged, (to - from) as u64);
        }
        same
    }

    /// The latest asking after which a range that holds `[from, to)` was
    /// taken whole (0: none is known). The caller holds `copied`'s lock.
    fn ranged(&self, from: usize, to: usize) -> u64 {
        let ranged = self.ranged.lock().unwrap_or_else(|e| e.into_inner());
        ranged.iter().filter(|r| r.0 <= from && to <= r.1).map(|r| r.2).max().unwrap_or(0)
    }

    /// [`Shadow::update`] of `[from, to)` (inside the buffer) where no
    /// shader writes.
    fn update_free(&self, from: usize, to: usize) -> usize {
        if self.watched.load(Ordering::Relaxed) != NOT_WATCHED {
            if let Some((now, written)) = crate::watch::writes(from, to, true) {
                // (A range whose blocks rest after a write has no guard now.)
                let guarded = crate::watch::guarded(from, to) || self.watched.load(Ordering::Relaxed) == GUARDED;
                self.watched.store(if guarded { GUARDED } else { 1 }, Ordering::Relaxed);
                let copied = self.copy_stale(from, to, now, &written);
                self.refresh_check(from, to);
                #[cfg(feature = "diagnostics")]
                {
                    use crate::copystat::{C, add};
                    add(C::Checked, (to - from) as u64);
                    add(C::Unchanged, (to - from).saturating_sub(copied) as u64);
                }
                return copied;
            }
            self.watched.store(NOT_WATCHED, Ordering::Relaxed);
        }
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::WholeBytes, (to - from) as u64);
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        self.copy(from, to);
        #[cfg(feature = "diagnostics")]
        crate::copystat::whole(from, to - from, started.elapsed().as_micros() as u64);
        to - from
    }

    /// Diagnostic (`VICTORY_REFRESH_CHECK=1`; off by default): after a
    /// refresh of `[from, to)` that trusted the write watch or the guards,
    /// the first 256 bytes of every 64 KiB block of it are read from title
    /// memory and compared with the copy. A block that differs on three
    /// refreshes in a row was changed with nothing to tell of it (a write
    /// through another view of its pages, as DirectStorage's page
    /// destinations make) and the card reads old bytes there: said with
    /// the copy and the address. One that differs once is a write of the
    /// title's that arrived between the refresh and the look.
    fn refresh_check(&self, from: usize, to: usize) {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*ON.get_or_init(|| std::env::var_os("VICTORY_REFRESH_CHECK").is_some_and(|v| v == "1")) {
            return;
        }
        const SAMPLE: usize = 256;
        // Blocks found differing: how many refreshes in a row.
        static DIFFERING: Mutex<Option<std::collections::HashMap<usize, u32>>> = Mutex::new(None);
        static SAID: AtomicU64 = AtomicU64::new(0);
        let start = self.start as usize;
        let mut title = [0u8; SAMPLE];
        let mut at = from;
        while at < to {
            let n = SAMPLE.min(to - at);
            if crate::host::read_into(at as *const u8, &mut title[..n]) {
                // SAFETY: `[from, to)` lies inside the buffer (`update`),
                // so these bytes are inside the copy's mapping.
                let copy = unsafe { std::slice::from_raw_parts(self.data.add(at - start).cast_const(), n) };
                let mut map = DIFFERING.lock().unwrap_or_else(|e| e.into_inner());
                let map = map.get_or_insert_with(Default::default);
                if copy == &title[..n] {
                    map.remove(&at);
                } else {
                    let times = map.entry(at).or_insert(0);
                    *times += 1;
                    if *times == 3 {
                        let said = SAID.fetch_add(1, Ordering::Relaxed) + 1;
                        if said <= 64 || said.is_power_of_two() {
                            tracing::warn!(
                                said,
                                copy = self.id,
                                copy_start = format_args!("{start:#x}"),
                                copy_bytes = self.len,
                                address = format_args!("{at:#x}"),
                                refreshed = format_args!("{from:#x}..{to:#x}"),
                                on_the_card = self.card.is_some(),
                                window = self.window,
                                "a host copy of title memory is stale after its refresh, three times in a row:                                  the memory changed with nothing to tell of it (VICTORY_REFRESH_CHECK=1)"
                            );
                        }
                    }
                }
            }
            // The first bytes of the next 64 KiB block.
            at = (at / PAGE + 1) * PAGE;
        }
    }

    /// Copies the pages of `[from, to)` written after their last copy
    /// (`written`: last writes by page, in page order; `now`: this
    /// asking), whole pages within the buffer. Keeps the page state locked
    /// until the planned copies finish.
    fn copy_stale(&self, from: usize, to: usize, now: u64, written: &[(usize, u64)]) -> usize {
        let base = self.start as usize / PAGE;
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let light = crate::watch::light();
        let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        // Pages held in part are copied on every refresh. Whole, unwritten
        // pages and a range already covered by ranged can be left alone.
        let ranged = if light { self.ranged(from, to) } else { 0 };
        let (runs, again_at) = copied.plan_stale(base, start, end, from, to, now, ranged, written);
        let runs = if runs.is_empty() || again_at.is_empty() {
            runs
        } else {
            by_parts(runs, &again_at, &crate::watch::parts_written(&again_at))
        };
        let bytes = self.take_all(runs);
        if light {
            // Publish the completed range with copied -> ranged lock order.
            let mut ranged = self.ranged.lock().unwrap_or_else(|e| e.into_inner());
            ranged[0] = (from, to, now);
            if to - from >= ranged[1].1 - ranged[1].0 {
                ranged[1] = (from, to, now);
            }
        }
        drop(copied);
        bytes
    }

    /// [`Shadow::take`] of `runs` (from, to, known), on several threads if
    /// they are long: the big heap's copy has some 4 MB of them in each of
    /// 14 refreshes a frame, on one thread while the others had nothing.
    fn take_all(&self, runs: Vec<(usize, usize, bool)>) -> usize {
        const LONG: usize = 1 << 20;
        const PART: usize = 256 << 10;
        let bytes = runs.iter().map(|r| r.1 - r.0).sum();
        if bytes < LONG || !crate::watch::light() {
            for (from, to, known) in runs {
                self.take(from, to, known);
            }
            return bytes;
        }
        let mut parts = Vec::with_capacity(bytes / PART + runs.len());
        for (from, to, known) in runs {
            let mut at = from;
            while at < to {
                // Whole pages to a part.
                let next = ((at / PAGE) * PAGE + PART).min(to);
                parts.push((at, next, known));
                at = next;
            }
        }
        let this = std::ptr::from_ref(self) as usize;
        wide_each(parts.len(), move |i| {
            // SAFETY: `wide_each` returns when every part is done, before
            // the borrow of `self` ends. The parts lie apart.
            let this = unsafe { &*(this as *const Shadow) };
            let (from, to, known) = parts[i];
            this.take(from, to, known);
        });
        bytes
    }

    /// What shaders wrote in `[from, to)` (offsets) was put into title memory.
    fn note_landing(&self, from: usize, to: usize) {
        let first = self.start as usize / KERNEL_PAGE;
        let pages = (self.start as usize + from) / KERNEL_PAGE - first..(self.start as usize + to).div_ceil(KERNEL_PAGE) - first;
        let mut bits = self.landings.lock().unwrap_or_else(|e| e.into_inner());
        if bits.is_empty() {
            let all = (self.start + self.len).div_ceil(KERNEL_PAGE as u64) as usize - first;
            bits.resize(all.div_ceil(64), 0);
        }
        for page in pages {
            bits[page / 64] |= 1 << (page % 64);
        }
        self.landing.store(true, Ordering::Release);
    }

    /// [`Shadow::take`] of `[from, to)`, which the copy held before and which
    /// is written since: where it may be only the landing that wrote it, the
    /// 4 KiB that title memory and the CPU's side have alike are left as they
    /// are, and nothing of them is sent ([`land_echo`]).
    fn take_landed(&self, from: usize, to: usize) -> usize {
        thread_local! {
            static TITLE: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
        }
        let start = self.start as usize;
        let first = start / KERNEL_PAGE;
        // Runs of pages noted, and not noted (any more).
        let mut runs: Vec<(usize, usize, bool)> = Vec::new();
        {
            let mut bits = self.landings.lock().unwrap_or_else(|e| e.into_inner());
            let mut at = from;
            while at < to {
                let next = ((at / KERNEL_PAGE + 1) * KERNEL_PAGE).min(to);
                let page = at / KERNEL_PAGE - first;
                let noted = bits.get(page / 64).is_some_and(|word| word >> (page % 64) & 1 != 0);
                // (A page looked at in part is looked at again.)
                if noted && next - at == KERNEL_PAGE {
                    bits[page / 64] &= !(1 << (page % 64));
                }
                match runs.last_mut() {
                    Some(last) if last.2 == noted && next - last.0 <= PAGE => last.1 = next,
                    _ => runs.push((at, next, noted)),
                }
                at = next;
            }
        }
        TITLE.with_borrow_mut(|title| {
            for (from, to, noted) in runs {
                if !noted {
                    self.copy(from, to);
                    continue;
                }
                title.resize(to - from, 0);
                // (Pages that are not there read as zeros.)
                crate::host::copy_mapped(title.as_mut_ptr(), from as *const u8, to - from, false);
                let mut at = from;
                while at < to {
                    let next = ((at / KERNEL_PAGE + 1) * KERNEL_PAGE).min(to);
                    let title = &title[at - from..next - from];
                    // SAFETY: inside the mapping of the copy's CPU side, which
                    // nothing else reads or writes there: no shader's range
                    // holds it, and the parts of a refresh lie apart.
                    let held = unsafe { std::slice::from_raw_parts_mut(self.data.add(at - start), next - at) };
                    if title == held {
                        ECHO[0].fetch_add((next - at) as u64, Ordering::Relaxed);
                    } else {
                        ECHO[1].fetch_add((next - at) as u64, Ordering::Relaxed);
                        held.copy_from_slice(title);
                        self.changed(at - start, next - start);
                        self.card_changed(at - start, next - start);
                    }
                    at = next;
                }
            }
        });
        to - from
    }

    /// Copies title memory `[from, to)` (inside the buffer) into the copy;
    /// `known`: the copy held it before ([`send_changed`]).
    fn take(&self, from: usize, to: usize, known: bool) -> usize {
        thread_local! {
            /// A page of title memory, as read.
            static TITLE: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
        }
        let Some(card) = self.card.as_ref().filter(|_| self.changes_only.load(Ordering::Relaxed) || send_changed()) else {
            if known && self.card.is_some() && self.landing.load(Ordering::Acquire) && land_echo() && !crate::host::upcheck::on() {
                return self.take_landed(from, to);
            }
            return self.copy(from, to);
        };
        let whole = !known && card.before.lock().unwrap_or_else(|e| e.into_inner()).is_none();
        let start = self.start as usize;
        TITLE.with_borrow_mut(|title| {
            title.resize(PAGE, 0);
            let mut at = from;
            while at < to {
                let n = (PAGE - at % PAGE).min(to - at);
                let title = &mut title[..n];
                // (Pages that are not there read as zeros.)
                crate::host::copy_mapped(title.as_mut_ptr(), at as *const u8, n, false);
                // SAFETY: inside the mapping of the copy's CPU side, which
                // nothing else reads or writes there (the card copies from
                // it): no shader's range holds it, and the parts of a
                // refresh lie apart.
                let held = unsafe { std::slice::from_raw_parts_mut(self.data.add(at - start), n) };
                let was = crate::host::upcheck::on().then(|| held.to_vec());
                let runs = take_chunk(at, title, held, whole);
                if keeps() {
                    for &(a, len) in &runs {
                        self.changed(at - start + a, at - start + a + len);
                    }
                }
                self.card_changed_runs(runs.iter().map(|&(a, len)| (at - start + a, at - start + a + len)));
                if let Some(was) = was {
                    crate::host::upcheck::note(self.id, at - start, &was, held);
                }
                SEND[0].fetch_add(n as u64, Ordering::Relaxed);
                if runs.is_empty() {
                    SEND[1].fetch_add(n as u64, Ordering::Relaxed);
                }
                SEND[2].fetch_add(runs.iter().map(|r| r.1 as u64).sum(), Ordering::Relaxed);
                SEND[3].fetch_add(runs.len() as u64, Ordering::Relaxed);
                at += n;
            }
        });
        to - from
    }

    /// Copies title memory `[from, to)`, cut to the buffer, into the copy.
    fn copy(&self, from: usize, to: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        // SAFETY: inside the mapping (`from` and `to` are within the buffer).
        let at = unsafe { self.data.add(from - start) };
        if self.card.is_some() && from < to && crate::host::upcheck::on() {
            for_pieces_of(from - start, to - start, |a, b| {
                let was = self.card_held(a, b - a);
                // SAFETY: inside the mapping, as above.
                crate::host::copy_mapped(unsafe { self.data.add(a) }, (start + a) as *const u8, b - a, false);
                crate::host::upcheck::note(self.id, a, &was, &self.card_held(a, b - a));
            });
        } else {
            crate::host::copy_mapped(at, from as *const u8, to - from, false);
        }
        // After the bytes are in: a snapshot taken meanwhile is not current.
        self.changed(from - start, to - start);
        self.card_changed(from - start, to - start);
        to - from
    }

    /// Whether shaders read and write this copy in the card's own memory
    /// ([`cards`]).
    pub(crate) fn carded(&self) -> bool {
        self.card.is_some()
    }

    /// Shaders store in it what is not declared: from here on only the dwords
    /// the title changes go up ([`by_address`]).
    pub(crate) fn keep_stores(&self) {
        self.changes_only.store(true, Ordering::Relaxed);
    }

    /// Offsets `[from, to)` go up again whole, as the CPU's side has them.
    pub(crate) fn resend(&self, from: usize, to: usize) {
        self.card_changed(from, to);
    }

    /// The CPU side's bytes at offsets `[from, to)` changed: they go to the
    /// card's side before the next submission.
    fn card_changed(&self, from: usize, to: usize) {
        self.card_changed_runs(std::iter::once((from, to)));
    }

    /// [`Shadow::card_changed`] of each of `runs`.
    fn card_changed_runs(&self, runs: impl Iterator<Item = (usize, usize)>) {
        let Some(card) = &self.card else {
            return;
        };
        let most = runs.size_hint().0;
        let mut runs = runs.filter(|r| r.0 < r.1).peekable();
        if runs.peek().is_none() {
            return;
        }
        let mut dirty = card.dirty.lock().unwrap_or_else(|e| e.into_inner());
        dirty.reserve(most);
        for (from, to) in runs {
            match dirty.last_mut() {
                Some(last) if from <= last.1 && to >= last.0 => *last = (last.0.min(from), last.1.max(to)),
                _ => dirty.push((from, to)),
            }
        }
        // (Under the lock: what takes the list sees it or finds it set.)
        card.has_dirty.store(true, Ordering::Release);
    }

    /// Whether shaders were ever declared to write it.
    pub(crate) fn written(&self) -> bool {
        self.written.load(Ordering::Relaxed)
    }

    /// `bytes` were put into title memory at `address`, and the submission
    /// being prepared copies the same to the copy at that copy's place in its
    /// list: the reference (the CPU's side of a copy in the card's memory, the
    /// snapshot) takes them and nothing is sent. What the card makes of them
    /// is then what differs from them when the submission has run. The range
    /// is declared written by it, and its sends are taken.
    pub(crate) fn placed_copy(&self, address: usize, bytes: &[u8]) {
        let (from, to) = self.inside(address, bytes.len());
        if from >= to || self.place.is_some() {
            return;
        }
        let bytes = &bytes[self.start as usize + from - address..][..to - from];
        let _merging = self.merging.write().unwrap_or_else(|e| e.into_inner());
        if self.card.is_some() {
            // SAFETY: inside the mapping of the CPU's side, in a range no
            // refresh writes; the lock keeps the landing out.
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.data.add(from), bytes.len()) };
        } else if let Some(snapshot) = self.before.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
            // SAFETY: the snapshot's allocation holds the copy's length.
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), snapshot.as_mut_ptr().add(from), bytes.len()) };
        }
        self.changed(from, to);
    }

    /// Of a copy in the card's memory: where its CPU side is.
    pub(crate) fn card_data(&self) -> *const u8 {
        self.data
    }

    /// Diagnostics: of a copy in the card's memory, what its CPU side holds
    /// at offsets `[at, at + len)` (cut to the copy).
    pub(crate) fn card_held(&self, at: usize, len: usize) -> Vec<u8> {
        let to = (at + len).min(self.len as usize);
        if self.card.is_none() || at >= to {
            return Vec::new();
        }
        // SAFETY: inside the mapping; the caller holds the submission lock,
        // under which it is written but for what is being landed.
        unsafe { std::slice::from_raw_parts(self.data.add(at).cast_const(), to - at) }.to_vec()
    }

    /// Diagnostics: the offsets whose bytes have not gone to the card's side
    /// yet.
    pub(crate) fn card_dirty_now(&self) -> Vec<(usize, usize)> {
        self.card.as_ref().map_or_else(Vec::new, |c| {
            if !c.has_dirty.load(Ordering::Acquire) {
                return Vec::new();
            }
            c.dirty.lock().unwrap_or_else(|e| e.into_inner()).clone()
        })
    }

    /// Of a copy in the card's memory: the CPU's side (PC memory), the
    /// card side's bytes from before if made yet, and their size.
    pub(crate) fn card_sides(&self) -> Option<(&ID3D12Resource, Option<ID3D12Resource>, u64)> {
        let card = self.card.as_ref()?;
        let before = card.before.lock().unwrap_or_else(|e| e.into_inner()).clone();
        Some((&card.staging, before, card.size))
    }

    /// Shader writes are declared for a copy in the card's memory: makes
    /// its "before" twin with `make` (bytes to buffer) if it has none.
    /// Whether one was made now: the caller fills it, whole, from the
    /// card's side before the submission's lists run. Without one (no
    /// memory) every piece of a declared range is fetched.
    pub(crate) fn card_make_before(&self, make: impl FnOnce(u64) -> Option<ID3D12Resource>) -> bool {
        let Some(card) = &self.card else {
            return false;
        };
        let mut before = card.before.lock().unwrap_or_else(|e| e.into_inner());
        if before.is_some() {
            return false;
        }
        *before = make(card.size);
        if before.is_some() {
            CARD_BYTES[1].fetch_add(card_hash_bytes(card.size), Ordering::Relaxed);
        }
        before.is_some()
    }

    /// The declared range `[address, end)` as the compare on the card
    /// takes it; `None` if none of it lies in the copy.
    pub(crate) fn card_span(&self, address: usize, end: usize) -> Option<CardSpan> {
        let (from, to) = self.inside(address, end.saturating_sub(address));
        (from < to).then(|| card_span_of(self.start as usize, from, to))
    }

    /// The pieces that hold offsets `[from, to)` of the copy (cut to it),
    /// as the card's hashing takes them.
    pub(crate) fn card_run(&self, from: usize, to: usize) -> CardSpan {
        let to = to.min(self.len as usize);
        card_span_of(self.start as usize, from.min(to.saturating_sub(1)), to)
    }

    /// The offsets of piece `i` of `span` that lie in the span.
    pub(crate) fn card_piece(&self, span: &CardSpan, i: u32) -> (usize, usize) {
        card_piece_of(self.start as usize, self.len as usize, span, i)
    }

    /// The piece that holds offset `at`.
    pub(crate) fn card_piece_of(&self, at: usize) -> u32 {
        self.pieces(at, at + 1).start as u32
    }

    /// The offsets the compare on the card hashes of a piece: what lies of
    /// it in the card's buffer (which may be longer than the copy), widened
    /// to 16 bytes.
    pub(crate) fn card_hashed(&self, piece: u32) -> (usize, usize) {
        let size = self.card_hash_size() as usize;
        card_hashed_of(self.start as usize, size, piece as usize)
    }

    /// The bytes of the copy the card's hashes cover: the title buffer's own,
    /// rounded up to the hash's 16-byte step. The card's buffer is the title
    /// buffer rounded up to 64 KiB, of which a title buffer of 288 bytes can
    /// give nothing back but those (after fxceless c36971b2).
    /// `VICTORY_CARD_HASH_LEN=0`: the whole buffer.
    pub(crate) fn card_hash_size(&self) -> u64 {
        static OWN: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let size = self.card.as_ref().map_or(self.len, |c| c.size);
        if *OWN.get_or_init(|| std::env::var_os("VICTORY_CARD_HASH_LEN").is_none_or(|v| v != "0")) {
            self.len.next_multiple_of(16).min(size)
        } else {
            size
        }
    }

    /// The walls trace (`crate::host::walls_trace`): a piece of `fetched`
    /// bytes of this copy came back from the card and `written` of them
    /// went into title memory. Summed per buffer and logged once a frame:
    /// which shader-written buffers the title's CPU could see fresh in
    /// that frame, and how much of each changed.
    pub(crate) fn walls_merged(&self, fetched: usize, written: usize) {
        if !crate::host::walls_trace() {
            return;
        }
        static SEEN: Mutex<(u64, BTreeMap<u64, (u64, u32, u64, u64)>)> = Mutex::new((0, BTreeMap::new()));
        let frame = crate::frame::current();
        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        if seen.0 != frame {
            let buffers: Vec<String> = seen
                .1
                .iter()
                .map(|(at, v)| format!("{at:#x}+{:#x}: {} pieces, {} fetched, {} written", v.0, v.1, v.2, v.3))
                .take(32)
                .collect();
            if !buffers.is_empty() {
                tracing::info!(
                    frame = seen.0,
                    count = seen.1.len(),
                    buffers = ?buffers,
                    "walls trace: shader-written card buffers merged into title memory in this frame"
                );
            }
            seen.1.clear();
            seen.0 = frame;
        }
        let entry = seen.1.entry(self.start).or_insert((self.len, 0, 0, 0));
        entry.1 += 1;
        entry.2 += fetched as u64;
        entry.3 += written as u64;
    }

    /// After a submission, for a copy in the card's memory: `gpu` is what
    /// the card's side holds at offset `at` on, in a piece the compare
    /// found differing. The bytes shaders wrote (those that differ from the
    /// CPU's side, which is what the card's side held when they started)
    /// go into title memory, from the first to the last of them, as
    /// [`Shadow::copy_back`] puts them; the CPU's side takes the result and
    /// sends it up before the next submission. `title` is scratch memory.
    /// Returns the bytes written to title memory.
    pub(crate) fn card_merge(&self, at: usize, gpu: &[u8], title: &mut Vec<u8>) -> usize {
        let start = self.start as usize;
        if self.card.is_none() || at + gpu.len() > self.len as usize {
            return 0;
        }
        // SAFETY: inside the mapping (checked above); only the submitting
        // thread writes it.
        let old = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), gpu.len()) };
        let Some(first) = first_difference(gpu, old) else {
            return 0;
        };
        let last = last_difference(gpu, old).unwrap_or(first);
        let (at, n) = (at + first, last + 1 - first);
        let (gpu, old) = (&gpu[first..=last], &mut old[first..=last]);
        title.clear();
        title.resize(n, 0);
        let mut written = 0;
        if crate::host::read_into((start + at) as *const u8, title) {
            // The stretch as read: what the card did not change is the
            // title's, and may have been written again since.
            // As in `copy_back`: the card's byte where it is not what it
            // was before the submission, else the title's.
            let mut differs = 0u8;
            let was = title.clone();
            for ((t, g), o) in title.iter_mut().zip(gpu).zip(old.iter_mut()) {
                let wrote = if *g != *o { 0xffu8 } else { 0 };
                let merged = (*t & !wrote) | (*g & wrote);
                differs |= merged ^ *t;
                *t = merged;
                *o = merged;
            }
            if differs != 0 {
                written = put_changed(start + at, &was, title);
            }
        } else {
            // Title memory is not mapped there: the copy keeps what the
            // card wrote, as before.
            old.copy_from_slice(gpu);
        }
        // The card's side and its "before" take the result.
        self.card_changed(at, at + n);
        written
    }

    /// [`relay`], for a copy in the card's memory: `gpu` is what the
    /// card's side held at offset `at` on when a fetch sent after send
    /// `fetched_after` (`crate::host`'s order of the queue) ran, in a piece
    /// the compare found differing. The bytes that differ from the CPU's
    /// side are what shaders wrote; they go into title memory, from the
    /// first to the last of them, and into the CPU's side. Nothing is sent
    /// up afterwards (the card holds these bytes or newer ones: sending
    /// them up could undo a later flight's write), and the title's other
    /// bytes are left to the refresh.
    ///
    /// Left out: bytes of the CPU's side that hold title bytes the card
    /// had not received when the fetch ran, those still to go up and those
    /// that went up with a later send. There the card's side is the older
    /// one, and taking it for a shader's write would put it over what the
    /// title wrote. Under the CPU side's lock (the snapshot's, which such
    /// a copy has no other use for), as [`Shadow::merge_under_writes`].
    /// `title` is scratch memory. Returns the bytes written to title
    /// memory.
    pub(crate) fn card_merge_relay(&self, at: usize, gpu: &[u8], fetched_after: u64, title: &mut Vec<u8>) -> usize {
        let start = self.start as usize;
        let Some(card) = &self.card else {
            return 0;
        };
        if at + gpu.len() > self.len as usize {
            return 0;
        }
        let _side = self.before.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: inside the mapping (checked above). The offsets are in a
        // range declared and not done with, which no refresh writes; what
        // does write it (the title's bytes put in under shader writes)
        // holds the lock held here.
        let held = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), gpu.len()) };
        let Some(first) = first_difference(gpu, held) else {
            return 0;
        };
        let last = last_difference(gpu, held).unwrap_or(first);
        let (at, n) = (at + first, last + 1 - first);
        let (gpu, held) = (&gpu[first..=last], &mut held[first..=last]);
        // Check `has_dirty` FIRST, then `has_sent`: `card_dirty` publishes
        // `has_sent` (Release) before clearing `has_dirty` (Release), so
        // observing `has_dirty == false` guarantees observing `has_sent == true`
        // for any range transitioning from `dirty` to `sent`.
        let skip: Vec<(usize, usize)> =
            if !card.has_dirty.load(Ordering::Acquire) && !card.has_sent.load(Ordering::Acquire) {
                Vec::new()
            } else {
                let dirty = card.dirty.lock().unwrap_or_else(|e| e.into_inner());
                let mut skip: Vec<(usize, usize)> = dirty.iter().copied().filter(|&(a, b)| a < at + n && b > at).collect();
                if card.has_sent.load(Ordering::Acquire) {
                    let sent = card.sent.lock().unwrap_or_else(|e| e.into_inner());
                    sent.near(fetched_after, at, at + n, &mut skip);
                }
                skip
            };
        title.clear();
        title.resize(n, 0);
        let mapped = crate::host::read_into((start + at) as *const u8, title);
        let was = title.clone();
        let mut changed = false;
        for (a, b) in outside(at, at + n, skip) {
            let (a, b) = (a - at, b - at);
            // Title memory that is not mapped: the CPU's side keeps what
            // the card wrote, as before; `title` is scratch then.
            changed |= merge_piece(&gpu[a..b], &mut held[a..b], &mut title[a..b]);
        }
        if !mapped || !changed {
            return 0;
        }
        put_changed(start + at, &was, title)
    }
}

/// For tests: copies with no host buffer behind them.
#[cfg(test)]
impl Shadow {
    /// A copy at `data` of title memory `[start, start + len)`, with a card
    /// side if `card`. It is never let go: there is no buffer to release.
    pub(crate) fn unbacked(start: u64, data: *mut u8, len: u64, card: bool) -> Arc<Shadow> {
        // SAFETY: an interface pointer nothing calls.
        let none = || unsafe { std::mem::transmute::<usize, ID3D12Resource>(8) };
        let mut shadow = Shadow::new(none(), data, 0, start, len);
        if card {
            shadow.card = Some(Card {
                staging: none(),
                before: Mutex::default(),
                dirty: Mutex::default(),
                has_dirty: AtomicBool::new(false),
                sent: Mutex::default(),
                has_sent: AtomicBool::new(false),
                size: len.next_multiple_of(PIECE as u64),
            });
        }
        let shadow = Arc::new(shadow);
        std::mem::forget(shadow.clone());
        shadow
    }

    /// Includes a synthetic card copy in submission-time updates for a test.
    pub(crate) fn track_carded_test(self: &Arc<Self>) -> impl Drop {
        struct Tracked(std::sync::Weak<Shadow>);
        impl Drop for Tracked {
            fn drop(&mut self) {
                CARDED.lock().unwrap().retain(|weak| !weak.ptr_eq(&self.0));
            }
        }
        let weak = Arc::downgrade(self);
        CARDED.lock().unwrap().push(weak.clone());
        Tracked(weak)
    }

    /// What shaders wrote goes into title memory ([`put_changed`]).
    pub(crate) fn put(at: usize, title: &[u8], merged: &[u8]) {
        put_changed(at, title, merged);
    }

    /// What is to go up to the card, noted as sent with `number`, as
    /// [`card_dirty`] does; notes of sends up to `merged` go.
    pub(crate) fn send_up(&self, number: u64, merged: u64) -> Vec<(usize, usize)> {
        let card = self.card.as_ref().unwrap();
        let mut dirty = card.dirty.lock().unwrap();
        let ranges = std::mem::take(&mut *dirty);
        let mut sent = card.sent.lock().unwrap();
        sent.forget(merged);
        sent.note(number, ranges.clone());
        card.has_sent.store(!sent.is_empty(), Ordering::Release);
        card.has_dirty.store(false, Ordering::Release);
        ranges
    }

    /// What is to go up to the card, taken.
    pub(crate) fn take_dirty(&self) -> Vec<(usize, usize)> {
        let card = self.card.as_ref().unwrap();
        let mut dirty = card.dirty.lock().unwrap();
        let ranges = std::mem::take(&mut *dirty);
        card.has_dirty.store(false, Ordering::Release);
        ranges
    }
}

impl Shadow {
    /// [`Shadow::card_merge_relay`] of several pieces, each within one
    /// 64 KiB block of title memory: from and to (offsets in the copy), and
    /// where the fetched bytes are. On any thread: several may land pieces
    /// of one copy, no two the same bytes. Returns the bytes put into title
    /// memory.
    ///
    /// What is left out is as there, asked once for all of them: the notes
    /// of bytes that went up (or are to go up) behind the fetch only move
    /// from `dirty` to `sent`, with a later number, and grow only by
    /// [`Shadow::merge_under_writes`], which the lock held here keeps out.
    ///
    /// # Safety
    /// The third of each piece is the address of its bytes, which stay as
    /// they are during the call.
    pub(crate) unsafe fn land_card(
        &self,
        pieces: impl Iterator<Item = (usize, usize, usize)> + Clone,
        fetched_after: u64,
        notes: &mut crate::land::Notes,
    ) -> u64 {
        let Some(card) = &self.card else {
            return 0;
        };
        let waiting = std::time::Instant::now();
        let _merging = self.merging.read().unwrap_or_else(|e| e.into_inner());
        crate::land::spent(crate::land::stat::LOCK, waiting);
        let (from, to) = pieces.clone().fold((usize::MAX, 0), |(a, b), p| (a.min(p.0), b.max(p.1)));
        let skip: Vec<(usize, usize)> =
            if !card.has_dirty.load(Ordering::Acquire) && !card.has_sent.load(Ordering::Acquire) {
                Vec::new()
            } else {
                let dirty = card.dirty.lock().unwrap_or_else(|e| e.into_inner());
                let mut skip: Vec<(usize, usize)> = dirty.iter().copied().filter(|&(a, b)| a < to && b > from).collect();
                if card.has_sent.load(Ordering::Acquire) {
                    let sent = card.sent.lock().unwrap_or_else(|e| e.into_inner());
                    sent.near(fetched_after, from, to, &mut skip);
                }
                skip
            };
        // Both sources of protected bytes can overlap. Join them once so
        // each piece can find its first overlapping range by end address.
        let skip = if skip.len() > 1 { joined(skip) } else { skip };
        let mut partial = Vec::new();
        let mut put = 0;
        for (lo, hi, now) in pieces {
            if lo >= hi || hi > self.len as usize {
                continue;
            }
            let near = &skip[skip.partition_point(|&(_, end)| end <= lo)..];
            let whole = [(0, hi - lo)];
            let parts: &[(usize, usize)] = if near.first().is_none_or(|&(start, _)| start >= hi) {
                &whole
            } else {
                partial.clear();
                for_outside(lo, hi, near.iter().copied(), |a, b| partial.push((a - lo, b - lo)));
                &partial
            };
            // SAFETY: inside the mapping (checked above), in a range
            // declared and not done with, which no refresh writes; the
            // title's bytes are kept out by the lock; the caller's.
            // Where title memory is not mapped the CPU's side keeps what
            // the card wrote, as before.
            let merged =
                unsafe { crate::land::piece(self.start as usize + lo, now as *const u8, self.data.add(lo), parts, true, notes, self) };
            self.walls_merged(hi - lo, merged as usize);
            if merged > 0 {
                self.note_landing(lo, hi);
            }
            put += merged;
        }
        put
    }

    /// [`Shadow::relay_merge`] of a piece `[at, at + n)` (within one 64 KiB
    /// block of title memory) of a copy in the PC's memory; on any thread,
    /// as [`Shadow::land_card`]. The copy is read once and never written.
    /// Returns the bytes put into title memory.
    pub(crate) fn land_back(&self, at: usize, n: usize, notes: &mut crate::land::Notes) -> u64 {
        if self.card.is_some() || n == 0 || at + n > self.len as usize {
            return 0;
        }
        let waiting = std::time::Instant::now();
        let _merging = self.merging.read().unwrap_or_else(|e| e.into_inner());
        crate::land::spent(crate::land::stat::LOCK, waiting);
        // The snapshot's memory stays where it is until the range is done
        // with (`copy_back_done`), which comes after this.
        let Some(held) = self.before.lock().unwrap_or_else(|e| e.into_inner()).as_mut().map(Vec::as_mut_ptr) else {
            return 0;
        };
        // SAFETY: inside the copy's mapping (checked above), which only the
        // card writes there, and inside the snapshot's allocation, which
        // holds the copy's length (`writing`): these offsets are in a range
        // declared and not done with, whose bytes it holds, and the lock
        // keeps out what else writes them. Title memory that is not there
        // is looked at again when the next flight that declared the range
        // lands.
        unsafe { crate::land::piece(self.start as usize + at, self.data.add(at), held.add(at), &[(0, n)], false, notes, self) }
    }

    /// What shaders wrote in 64 KiB block `block` (of title memory) of this
    /// copy was put into title memory beside the guard, which was noted as a
    /// write with number `now`, `before` the number of the block's last write
    /// before. If the copy held title memory's bytes of the block ahead of
    /// that it holds them still, and is noted to ([`crate::land`] says why);
    /// whether it was.
    ///
    /// That it did: it took the whole block after that write, and no other
    /// came that is not noted yet. Looked at after the note, whatever comes
    /// later has a number above it: a write of the title's takes the block's
    /// bit, until an asking has noted it.
    pub(crate) fn landed(&self, block: usize, before: u64, now: u64) -> bool {
        let Some(page) = block.checked_sub(self.start as usize / PAGE) else {
            return false;
        };
        let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        let Some(&at) = copied.at.get(page) else {
            return false;
        };
        let held = at != 0 && at >= before && crate::watch::block_published(block);
        if held {
            copied.set(page, at.max(now));
        }
        held
    }
}

#[cfg(test)]
mod land_tests {
    use super::{Card, PIECE, Shadow};
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

    /// A copy in the card's memory of title memory at `title` (its card side
    /// is never looked at here), all bytes 0 on the CPU's side.
    fn copy_of(title: usize, len: usize) -> Option<Shadow> {
        let (buffer, data) = crate::host::host()?.shared_buffer(len as u64)?;
        // SAFETY: the mapping, `len` bytes.
        unsafe { std::ptr::write_bytes(data, 0, len) };
        let mut shadow = Shadow::new(buffer.clone(), data, 0, title as u64, len as u64);
        shadow.card = Some(Card {
            staging: buffer,
            before: Mutex::default(),
            dirty: Mutex::default(),
            has_dirty: AtomicBool::new(false),
            sent: Mutex::default(),
            has_sent: AtomicBool::new(false),
            size: len as u64,
        });
        Some(shadow)
    }

    /// Memory that stays (`crate::watch` knows its addresses): private, or a
    /// view noted as the runtime notes the title's.
    fn memory(len: usize, view: bool) -> usize {
        use windows_sys::Win32::System::Memory as mem;
        // SAFETY: plain allocations of our own.
        unsafe {
            if !view {
                return mem::VirtualAlloc(std::ptr::null(), len, mem::MEM_RESERVE | mem::MEM_COMMIT, mem::PAGE_READWRITE) as usize;
            }
            let section = mem::CreateFileMappingW(-1isize as _, std::ptr::null(), mem::PAGE_READWRITE, 0, len as u32, std::ptr::null());
            let at = mem::MapViewOfFile(section, mem::FILE_MAP_WRITE, 0, 0, len).Value as usize;
            crate::inplace::title_memory_section(section as usize, len as u64, true);
            crate::title_memory_remapped(at, len);
            crate::inplace::title_memory_view(at, len, section as usize, 0);
            at
        }
    }

    fn random(seed: &mut u64) -> usize {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        (*seed >> 8) as usize
    }

    /// [`Shadow::land_card`] leaves title memory and the CPU's side as
    /// [`Shadow::card_merge_relay`] does, with title bytes that went up
    /// behind the fetch, or have not gone up, left out.
    #[test]
    fn pieces_land_as_the_merge_one_by_one_lands_them() {
        assert!(crate::safecopy::on());
        const LEN: usize = 3 * PIECE;
        let mut seed = 0x1234_5678_9abcu64;
        for view in [false, true] {
            let (a, b) = (memory(LEN, view), memory(LEN, view));
            let (Some(one), Some(many)) = (copy_of(a, LEN), copy_of(b, LEN)) else {
                return; // no host GPU
            };
            // SAFETY: the two memories and the two mappings.
            let bytes = |at: usize| unsafe { std::slice::from_raw_parts_mut(at as *mut u8, LEN) };
            for round in 0..40u32 {
                let base: Vec<u8> = (0..LEN).map(|_| random(&mut seed) as u8).collect();
                let (mut held, mut title, mut now) = (base.clone(), base.clone(), base);
                let (mut dirty, mut sent) = (Vec::new(), Vec::new());
                for k in 0..120 {
                    let at = random(&mut seed) % LEN / 4 * 4;
                    let n = (random(&mut seed) % 900 / 4 * 4).min(LEN - at);
                    match k % 4 {
                        0 | 1 => now[at..at + n].fill(random(&mut seed) as u8),
                        2 => title[at..at + n].fill(random(&mut seed) as u8),
                        // Title bytes put into the CPU's side meanwhile.
                        _ => {
                            let byte = random(&mut seed) as u8;
                            held[at..at + n].fill(byte);
                            title[at..at + n].fill(byte);
                            if k % 8 == 3 {
                                dirty.push((at, at + n));
                            } else {
                                sent.push((at, at + n, u64::from(round % 3) + (k as u64 / 16)));
                            }
                        }
                    }
                }
                let after = 4u64;
                for (shadow, at) in [(&one, a), (&many, b)] {
                    crate::watch::writing(at, LEN);
                    bytes(at).copy_from_slice(&title);
                    bytes(shadow.data as usize).copy_from_slice(&held);
                    let card = shadow.card.as_ref().expect("made with one");
                    *card.dirty.lock().unwrap() = dirty.clone();
                    card.has_dirty.store(!dirty.is_empty(), Ordering::Release);
                    let sent_runs = super::SentRuns::of(&sent);
                    let has_sent = !sent_runs.is_empty();
                    *card.sent.lock().unwrap() = sent_runs;
                    card.has_sent.store(has_sent, Ordering::Release);
                }
                let pieces: Vec<(usize, usize)> = (0..3).map(|p| (p * PIECE + 16, (p + 1) * PIECE - 4)).collect();
                let mut scratch = Vec::new();
                for &(lo, hi) in &pieces {
                    one.card_merge_relay(lo, &now[lo..hi], after, &mut scratch);
                }
                let addressed = pieces.iter().map(|&(lo, hi)| (lo, hi, now.as_ptr() as usize + lo));
                let mut notes = crate::land::Notes::default();
                // SAFETY: the fetched bytes are `now`'s.
                unsafe { many.land_card(addressed, after, &mut notes) };
                notes.flush();
                assert!(bytes(a) == bytes(b), "title memory, round {round}");
                assert!(bytes(one.data as usize) == bytes(many.data as usize), "the CPU's side, round {round}");
            }
        }
    }

    /// Landings of one copy on several threads, the title writing dwords of
    /// the same blocks, and the title's bytes put into the CPU's side under
    /// the shaders' writes ([`Shadow::merge_under_writes`]), all at once. The
    /// card has never received the title's bytes: what is fetched holds older
    /// ones there, which must not be taken for shaders' writes.
    #[test]
    fn landings_and_title_bytes_going_in_keep_each_others_bytes() {
        assert!(crate::safecopy::on());
        const BLOCKS: usize = 16;
        const LEN: usize = BLOCKS * PIECE;
        let title = memory(LEN, true);
        let Some(shadow) = copy_of(title, LEN) else {
            return; // no host GPU
        };
        shadow.pending.lock().unwrap().push((title, title + LEN));
        shadow.has_pending.store(true, Ordering::Release);
        // SAFETY: committed, aligned; touched as atomics or by the landing's
        // stores of whole dwords.
        let dword = |i: usize| unsafe { &*((title + i * 4) as *const AtomicU32) };
        let stop = AtomicBool::new(false);
        std::thread::scope(|s| {
            let writer = s.spawn(|| {
                let mut value = 1u32;
                while !stop.load(Ordering::Relaxed) {
                    for i in (1..LEN / 4).step_by(4 * 331) {
                        assert_eq!(dword(i).swap(value, Ordering::Relaxed), value - 1, "dword {i}: undone");
                    }
                    value += 1;
                    std::thread::sleep(std::time::Duration::from_micros(200));
                }
                value - 1
            });
            s.spawn(|| {
                while !stop.load(Ordering::Relaxed) {
                    let mut runs = usize::MAX;
                    assert!(!shadow.merge_under_writes(title, LEN, &mut runs));
                    // Sent up, as the preparing of a submission would: noted
                    // with a number behind the fetches'.
                    let card = shadow.card.as_ref().expect("made with one");
                    let mut dirty = card.dirty.lock().unwrap();
                    let mut sent = card.sent.lock().unwrap();
                    sent.note(9, std::mem::take(&mut *dirty));
                    card.has_sent.store(!sent.is_empty(), Ordering::Release);
                    card.has_dirty.store(false, Ordering::Release);
                }
            });
            let mut now = vec![0u8; LEN];
            for round in 1..200u32 {
                for block in (0..BLOCKS).filter(|b| !(*b as u32 * 5 + round).is_multiple_of(3)) {
                    for i in (block * PIECE / 4 + 3..(block + 1) * PIECE / 4).step_by(4) {
                        now[i * 4..i * 4 + 4].copy_from_slice(&(round << 20 | i as u32).to_le_bytes());
                    }
                }
                let at = now.as_ptr() as usize;
                crate::land::each(BLOCKS, &|b| {
                    let mut notes = crate::land::Notes::default();
                    let (lo, hi) = (b * PIECE, (b + 1) * PIECE);
                    // SAFETY: the fetched bytes are `now`'s; each block once.
                    unsafe { shadow.land_card(std::iter::once((lo, hi, at + lo)), 8, &mut notes) };
                    notes.flush();
                });
                for i in (3..LEN / 4).step_by(4 * 53) {
                    let expect = u32::from_le_bytes(now[i * 4..i * 4 + 4].try_into().expect("four"));
                    assert_eq!(dword(i).load(Ordering::Relaxed), expect, "round {round} dword {i}");
                }
            }
            stop.store(true, Ordering::Relaxed);
            let last = writer.join().expect("nothing undone");
            assert!((1..LEN / 4).step_by(4 * 331).all(|i| dword(i).load(Ordering::Relaxed) == last));
        });
    }
}

struct Entry {
    end: u64,
    shadow: Option<Arc<Shadow>>,
}

/// Title buffers by (start address, resource), with their copy once made.
///
/// Read by every view a draw names, on all recording threads at once, and
/// written when a buffer comes, goes or gets its copy: readers share it.
static BUFFERS: std::sync::RwLock<BTreeMap<(u64, usize), Entry>> = std::sync::RwLock::new(BTreeMap::new());

fn buffers() -> std::sync::RwLockWriteGuard<'static, BTreeMap<(u64, usize), Entry>> {
    let map = BUFFERS.write().unwrap_or_else(|e| e.into_inner());
    // What threads remember of it ([`FOUND`]) is of before.
    BUFFERS_CHANGED.fetch_add(1, Ordering::SeqCst);
    map
}

/// How often [`BUFFERS`] was taken to be changed.
static BUFFERS_CHANGED: AtomicU64 = AtomicU64::new(1);

/// Answers a thread remembers.
const FOUND_MOST: usize = 1024;

/// An ask (address, bytes), the count of changes it was answered at, and the
/// answer.
type Found = (u64, u64, u64, std::sync::Weak<Shadow>);

thread_local! {
    /// The last answers of [`lookup`] out of [`BUFFERS`], by a hash of the
    /// ask: the same buffers are asked about draw after draw, from five
    /// threads, each time through the map's lock and a dozen of its nodes
    /// (7 % of the recording threads' time, profile waits-1).
    static FOUND: std::cell::RefCell<Vec<Found>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Whether [`FOUND`] is kept (`VICTORY_QUICK_FIND=0`: not).
fn quick_find() -> bool {
    crate::toggled!("VICTORY_QUICK_FIND")
}

fn found_at(address: u64, len: u64) -> usize {
    let h = (address ^ len.rotate_left(32)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (h >> 54) as usize % FOUND_MOST
}

/// What this thread remembers of an ask, if nothing changed since.
fn found(address: u64, len: u64) -> Option<Arc<Shadow>> {
    let changed = BUFFERS_CHANGED.load(Ordering::SeqCst);
    let shadow = FOUND.with(|f| {
        let f = f.borrow();
        let (a, n, at, shadow) = f.get(found_at(address, len))?;
        ((*a, *n, *at) == (address, len, changed)).then(|| shadow.upgrade()).flatten()
    })?;
    // Whoever takes the map to let a copy go that nobody holds finds it held
    // now, or had counted before: then it is not handed out.
    std::sync::atomic::fence(Ordering::SeqCst);
    (BUFFERS_CHANGED.load(Ordering::SeqCst) == changed).then_some(shadow)
}

/// Remembers an answer given while `changed` was the count.
fn remember(address: u64, len: u64, changed: u64, shadow: &Arc<Shadow>) {
    FOUND.with(|f| {
        let mut f = f.borrow_mut();
        if f.is_empty() {
            f.resize_with(FOUND_MOST, || (0, 0, 0, std::sync::Weak::new()));
        }
        f[found_at(address, len)] = (address, len, changed, Arc::downgrade(shadow));
    });
}

fn buffers_read() -> std::sync::RwLockReadGuard<'static, BTreeMap<(u64, usize), Entry>> {
    BUFFERS.read().unwrap_or_else(|e| e.into_inner())
}

/// A title buffer exists at `start` (`len` bytes).
pub(crate) fn register(resource: usize, start: u64, len: u64) {
    buffers().insert(
        (start, resource),
        Entry {
            end: start + len,
            shadow: None,
        },
    );
}

/// The title released it; lists that still hold its copy keep it alive.
pub(crate) fn unregister(resource: usize, start: u64) {
    buffers().remove(&(start, resource));
}

/// The title buffer holding `address`, as (start, length), without making
/// its copy (diagnostics: which buffer a dropped program's root view names).
#[cfg(feature = "diagnostics")]
pub(crate) fn extent(address: u64) -> Option<(u64, u64)> {
    buffers()
        .range(..=(address, usize::MAX))
        .rev()
        .take(64)
        .find(|(_, e)| e.end > address)
        .map(|(k, e)| (k.0, e.end - k.0))
}

/// Title memory `[at, at + len)` is about to show something else or be
/// given up: what is in place there (`crate::inplace`) is taken out of the
/// tables, so that the next ask makes a copy or opens the memory anew, and
/// goes when no list and no slot holds it any more. Returns how many.
pub(crate) fn in_place_let_go(at: u64, len: u64) -> u64 {
    let end = at.saturating_add(len);
    let mut gone: Vec<Arc<Shadow>> = Vec::new();
    {
        let mut map = LOOSE.write().unwrap_or_else(|e| e.into_inner());
        let keys: Vec<(u64, u64)> = map
            .range((at.saturating_sub(WINDOW), 0)..(end, 0))
            .filter(|(key, shadow)| shadow.place.is_some() && key.0 + key.1 > at)
            .map(|(key, _)| *key)
            .collect();
        gone.extend(keys.into_iter().filter_map(|key| map.remove(&key)));
    }
    {
        // As `lookup` finds a buffer: among the 256 that start below.
        let mut map = buffers();
        for (_, entry) in map.range_mut(..(end, 0)).rev().take(256) {
            if entry.end > at && entry.shadow.as_ref().is_some_and(|s| s.place.is_some()) {
                gone.extend(entry.shadow.take());
            }
        }
    }
    let n = gone.len() as u64;
    for shadow in gone {
        crate::host::retire(shadow);
    }
    n
}

/// For the in-place census (`crate::inplace`): the title buffers (start,
/// bytes, false) and the windows outside buffers that have a copy (start,
/// bytes, true).
pub(crate) fn spans() -> Vec<(u64, u64, bool)> {
    let mut spans: Vec<(u64, u64, bool)> = buffers_read().iter().map(|(key, entry)| (key.0, entry.end - key.0, false)).collect();
    let loose = LOOSE.read().unwrap_or_else(|e| e.into_inner());
    spans.extend(loose.keys().map(|&(start, len)| (start, len, true)));
    spans
}

/// The copy of the innermost title buffer holding `[address, address +
/// len)`, made on first use. A `len` of 0 or 1 is an ask without a length
/// (a root view: the caller reads on to the copy's end).
pub(crate) fn find(address: u64, len: u64) -> Option<Arc<Shadow>> {
    lookup(address, len, false)
}

/// As [`find`], for an ask that is done with the copy when its list has
/// run (a draw's vertex and index buffers, indirect arguments): outside
/// buffer objects it takes a window that is there already and holds the
/// range ([`holder`]), and makes one of its own only if none does.
///
/// [`find`] names a window by the ask alone, which a descriptor slot needs
/// (the slot shows the window its view found when it was made, and the
/// pass that brings it up to date must find the same one). A draw needs no
/// such thing: it binds the copy it found and brings that copy up to date.
/// Named by the ask alone, every draw's range was a window of its own
/// wherever the windows are cut finer than 4 MiB: draws that bind
/// different ranges of the same memory then each copied their range, and
/// windows nothing asked for again for ten seconds were let go and made
/// (and copied) anew. Run 71, on a 256 KiB grid: 350 MB copied per frame
/// against 120 MB, 7,087 write-watch askings against 2,600, and 5,263
/// windows (4.5 GB) let go in two minutes against 393 in a whole run.
pub(crate) fn find_any(address: u64, len: u64) -> Option<Arc<Shadow>> {
    // With whole windows (the default) there is one window to a cell and
    // nothing to choose: exactly `find`.
    if !fine() {
        return find(address, len);
    }
    lookup(address, len, true)
}

/// The copy [`find`] gives for a range inside a title buffer, if it was made.
pub(crate) fn peek(address: u64, len: u64) -> Option<Arc<Shadow>> {
    let end = address.checked_add(len.max(1))?;
    buffers_read().range(..=(address, usize::MAX)).rev().take(64).find(|(_, e)| e.end >= end).and_then(|(_, e)| e.shadow.clone())
}

fn lookup(address: u64, len: u64, any: bool) -> Option<Arc<Shadow>> {
    let quick = quick_find();
    if quick && let Some(shadow) = found(address, len) {
        return Some(shadow);
    }
    let end = address.checked_add(len.max(1))?;
    {
        let map = buffers_read();
        match map.range(..=(address, usize::MAX)).rev().take(64).find(|(_, e)| e.end >= end) {
            // The copy is made already (nearly always).
            Some((_, Entry { shadow: Some(shadow), .. })) => {
                if quick {
                    // (Nobody changes the map while this lock is held.)
                    remember(address, len, BUFFERS_CHANGED.load(Ordering::SeqCst), shadow);
                }
                return Some(shadow.clone());
            }
            Some(_) => {}
            None => {
                drop(map);
                return loose(address, end, len <= 1, any);
            }
        }
    }
    let mut map = buffers();
    let key = map
        .range(..=(address, usize::MAX))
        .rev()
        .take(64)
        .find(|(_, e)| e.end >= end)
        .map(|(k, _)| *k);
    let Some(key) = key else {
        drop(map);
        return loose(address, end, len <= 1, any);
    };
    let entry = map.get_mut(&key)?;
    if entry.shadow.is_none() {
        let len = entry.end - key.0;
        // Rounded up so that views ending in the buffer's last 256 bytes fit.
        // Shaders write some buffers (unordered access), so not upload memory.
        let size = len.next_multiple_of(1 << 16);
        // In place, if asked for and the memory can be opened
        // (`crate::inplace::place_for`): the buffer is the title's own
        // memory. Not the big ones `cards` puts in the card's memory: those
        // are read too often to stay in the PC's.
        let big = cards() && (size >= card_limits().0 || card_all());
        // Renderer port 1 (`crate::port`, `VICTORY_PORT_CONSTANTS=1`): the
        // title's 32-bit graphics heap in place, big or not.
        if let Some(placed) = crate::port::place_for(key.0, len, false) {
            entry.shadow = Some(Arc::new(Shadow::placed(placed, key.0, len, false)));
            return entry.shadow.clone();
        }
        if !big && let Some(placed) = crate::inplace::place_for(key.0, len, false) {
            entry.shadow = Some(Arc::new(Shadow::placed(placed, key.0, len, false)));
            return entry.shadow.clone();
        }
        let (buffer, data) = host()?.title_shared_buffer(size)?;
        crate::host::name(&buffer, || format!("title buffer {:#x}+{len:#x}", key.0));
        // A big one, with `cards`: shaders get a buffer in the card's own
        // memory, and the one just made is the CPU's side.
        let (buffer, card) = match card_for(size) {
            Some(on_card) => {
                crate::host::name(&on_card, || format!("title buffer {:#x}+{len:#x} on the card", key.0));
                tracing::info!(
                    start = format_args!("{:#x}", key.0),
                    len,
                    "a buffer's host copy is in the card's memory"
                );
                let card = Card {
                    staging: buffer,
                    before: Mutex::default(),
                    dirty: Mutex::default(),
                    has_dirty: AtomicBool::new(false),
                    sent: Mutex::default(),
                    has_sent: AtomicBool::new(false),
                    size,
                };
                (on_card, Some(card))
            }
            None => (buffer, None),
        };
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        if len >= 16 << 20 {
            tracing::info!(
                start = format_args!("{:#x}", key.0),
                len,
                "large buffer copied to the host"
            );
        } else {
            tracing::debug!(
                start = format_args!("{:#x}", key.0),
                len,
                "buffer copied to the host"
            );
        }
        let mut shadow = Shadow::new(buffer, data, gpu, key.0, len);
        shadow.card = card;
        let shadow = Arc::new(shadow);
        if shadow.card.is_some() {
            CARDED.lock().unwrap_or_else(|e| e.into_inner()).push(Arc::downgrade(&shadow));
        }
        entry.shadow = Some(shadow);
    }
    entry.shadow.clone()
}

/// Titles also use memory outside any buffer object as GPU memory (
/// the title's interface vertices, from its own allocator). Such memory
/// gets copies of the aligned windows it touches.
/// A window nothing names any more is let go ([`let_go_idle`]).
/// shortcut: a window is 4 MiB however little of it is read (the memory
/// line says how much was ever asked for, and how much was ever copied);
/// [`window`] cuts the windows of asks that give a length on a finer grid.
const WINDOW: u64 = 4 << 20;

/// The grid the windows of asks with a length are cut on, in bytes: 4 MiB
/// unless `VICTORY_WINDOW_KB` names a smaller power of two (64 at least).
/// A window is the PC's memory, all of it, however little of it is read:
/// 803 windows held 3.3 GB in run 60's gameplay, and all that was ever
/// asked for in them lay within 1.6 GB. On a grid of 1 MiB or 256 KiB a
/// window ends nearer to what is read. An ask without a length keeps the
/// 4 MiB grid whatever is set: a root view reads from its address to its
/// window's end, and mesh streams bound that way reach megabytes.
/// shortcut: the default stays 4 MiB until a run with the memory line's
/// window counts has shown what the finer grids give; the same memory
/// asked for with different ranges then lies in more than one window.
fn window() -> u64 {
    static GRID: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *GRID.get_or_init(|| {
        let kb = std::env::var("VICTORY_WINDOW_KB").ok().and_then(|v| v.trim().parse::<u64>().ok());
        grid(kb)
    })
}

/// The window grid in bytes for a setting in KiB: a power of two from
/// 64 KiB to 4 MiB, 4 MiB for anything else.
fn grid(kb: Option<u64>) -> u64 {
    match kb {
        Some(kb) if kb.is_power_of_two() && (64..=4096).contains(&kb) => kb << 10,
        _ => WINDOW,
    }
}

/// The window (start, bytes) of title memory for an ask of `[address,
/// end)` on a grid of `grid` bytes.
fn window_of(address: u64, end: u64, grid: u64) -> (u64, u64) {
    let start = address & !(grid - 1);
    (start, end.next_multiple_of(grid) - start)
}

/// Whether the window of an ask without a length begins at the ask instead
/// of at the 4 MiB line below it. Such an ask (a root view) reads from its
/// address to the end of its 4 MiB cell; what lies below the lowest
/// address ever asked for in the cell is memory the host zeroed and nothing
/// reads (513 of run 65's 782 windows were asked for that way, 2,052 MB;
/// of all windows' 3,236 MB, 1,988 MB were ever copied). With this on, the
/// window is the cell's tail from the ask's step on ([`TAIL_STEP`]); an ask
/// lower in the cell later makes a longer tail, which every ask from then
/// on takes, and the shorter one goes when nothing holds it any more.
/// shortcut: off unless `VICTORY_WINDOW_TAILS=1`. A longer tail is another
/// host buffer than the shorter one: what was bound to the shorter stays
/// right only because every list brings up to date the copy it bound
/// (`draw.rs` keeps a root view's copy with its slot). It pays only
/// together with `VICTORY_WINDOW_KB` below 4096: on the 4 MiB grid an ask
/// with a length in the same cell makes the whole cell's window, and the
/// tail is that.
fn tails() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WINDOW_TAILS").is_some_and(|v| v == "1"))
}

/// Tails begin on multiples of this. A longer tail is a new host buffer,
/// filled from title memory from its first byte to its last: begun on the
/// 64 KiB page of each ask, a cell whose asks came from the top down made
/// up to 63 of them, one after the other (part of run 71's copies). On
/// this step it is 15 at most, and a tail holds up to 192 KiB it did not
/// need.
const TAIL_STEP: u64 = 256 << 10;

/// The window (start, bytes) a tail ask at `address` makes: from the ask's
/// step ([`TAIL_STEP`]) to the end of its 4 MiB cell.
fn tail_of(address: u64, end: u64) -> (u64, u64) {
    let (cell, len) = window_of(address, end, WINDOW);
    let start = address & !(TAIL_STEP - 1);
    (start, cell + len - start)
}

/// Of `windows` (start, bytes), the one an ask of `[address, end)` that
/// may take any window takes ([`find_any`]): the longest that holds the
/// whole range.
fn holder(windows: impl Iterator<Item = (u64, u64)>, address: u64, end: u64) -> Option<(u64, u64)> {
    windows
        .filter(|&(start, bytes)| start <= address && start + bytes >= end)
        .max_by_key(|&(start, bytes)| (bytes, std::cmp::Reverse(start)))
}

/// Whether windows may be smaller than their 4 MiB cell (a finer grid, or
/// tails): only then is there more than one window to choose from.
fn fine() -> bool {
    tails() || window() < WINDOW
}

/// For the memory line: the bytes of title memory the windows alive lie
/// over (each byte once), beside the windows' own bytes in
/// [`memory`]: the difference is memory that is in more than one window,
/// which is copied once for each.
#[cfg(feature = "diagnostics")]
pub(crate) fn window_cover() -> u64 {
    let loose = LOOSE.read().unwrap_or_else(|e| e.into_inner());
    cover(loose.keys().copied())
}

/// The bytes `windows` (start, bytes; in order of start) lie over, each
/// byte once.
#[cfg_attr(not(any(test, feature = "diagnostics")), allow(dead_code))]
fn cover(windows: impl Iterator<Item = (u64, u64)>) -> u64 {
    let (mut covered, mut reached) = (0, 0);
    for (start, bytes) in windows {
        let end = start + bytes;
        if end > reached {
            covered += end - start.max(reached);
            reached = end;
        }
    }
    covered
}

/// Of `windows` (start, bytes; in order), the longest tail a tail ask at
/// `address` can take: the first that begins in the ask's cell at or below
/// the ask's page and ends where the cell ends.
fn longest_tail(windows: impl Iterator<Item = (u64, u64)>, address: u64, end: u64) -> Option<(u64, u64)> {
    let (cell, len) = window_of(address, end, WINDOW);
    let step = address & !(TAIL_STEP - 1);
    windows
        .filter(|&(start, bytes)| start >= cell && start <= step && start + bytes == cell + len)
        .min()
}

/// Whether `[address, address + len)` lies in one title buffer.
pub(crate) fn in_buffer(address: u64, len: u64) -> Option<(u64, u64)> {
    let end = address.checked_add(len)?;
    let map = buffers_read();
    let found = map.range(..=(address, usize::MAX)).rev().take(64).find(|(_, e)| e.end >= end);
    found.map(|(k, e)| (k.0, e.end - k.0))
}

/// Whether a title buffer of at most `most` bytes lies in `[address, address
/// + len)`, all of it or in part.
pub(crate) fn small_buffer_in(address: u64, len: u64, most: u64) -> bool {
    let map = buffers_read();
    map.range((address.saturating_sub(most), 0)..(address + len, 0))
        .any(|(key, entry)| entry.end > address && entry.end - key.0 <= most)
}

/// A copy of the memory `[start, start + len)` (an arena of
/// [`crate::memory`]'s, its directory, title memory around a small buffer)
/// for the programs that find memory by address, on the card: the structures
/// rays are traced through, read with every step of every ray.
///
/// What shaders write there by address is not brought back, so the copy is
/// the only place that has it. The CPU's side holds the bytes as they were
/// last taken, and only dwords that changed since go up.
pub(crate) fn by_address(start: u64, len: u64) -> Option<Arc<Shadow>> {
    let host = host()?;
    if !cards() || !host.card_ready() {
        return None;
    }
    // At most so much on the card's own memory (4 GiB unless
    // `VICTORY_MEMORY_CARD_MB` says otherwise), and while it has room; past
    // that the side shaders see is in the PC's memory.
    static MOST: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    static USED: AtomicU64 = AtomicU64::new(0);
    let most = *MOST.get_or_init(|| {
        let mb = std::env::var("VICTORY_MEMORY_CARD_MB").ok().and_then(|v| v.trim().parse::<u64>().ok());
        mb.unwrap_or(4096) << 20
    });
    let room = host.video_memory().is_none_or(|[budget, used, ..]| used + len < budget / 100 * 97);
    let (staging, data) = host.shared_buffer(len)?;
    let on_card = if room && USED.fetch_add(len, Ordering::Relaxed) + len <= most {
        host.uav_buffer(len)
    } else {
        None
    };
    let buffer = match on_card {
        Some(buffer) => {
            BY_ADDRESS_CARD_BYTES.fetch_add(len, Ordering::Relaxed);
            buffer
        }
        None => host.shared_buffer(len)?.0,
    };
    crate::host::name(&staging, || format!("title memory {start:#x}+{len:#x} by address, the CPU's side"));
    crate::host::name(&buffer, || format!("title memory {start:#x}+{len:#x} by address"));
    // (Counted down with the windows' on the card when it goes.)
    WINDOW_CARD_BYTES.fetch_add(len, Ordering::Relaxed);
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    let mut shadow = Shadow::new(buffer, data, gpu, start, len);
    shadow.window = true;
    shadow.changes_only.store(true, Ordering::Relaxed);
    shadow.card = Some(Card {
        staging,
        before: Mutex::default(),
        dirty: Mutex::default(),
        has_dirty: AtomicBool::new(false),
        sent: Mutex::default(),
        has_sent: AtomicBool::new(false),
        size: len,
    });
    let shadow = Arc::new(shadow);
    CARDED.lock().unwrap_or_else(|e| e.into_inner()).push(Arc::downgrade(&shadow));
    Some(shadow)
}

type LooseTable = BTreeMap<(u64, u64), Arc<Shadow>>;
type PendingTickets = Mutex<BTreeMap<(u64, u64), Weak<Mutex<()>>>>;

/// Copies of title memory outside buffers, by (start, length).
static LOOSE: std::sync::RwLock<LooseTable> = std::sync::RwLock::new(BTreeMap::new());

/// Per-key constructor locks for pending loose-window allocations so
/// concurrent misses on the same key do not allocate duplicate backing while
/// unrelated lookups, allocations and sweeps proceed without waiting on `LOOSE`.
static LOOSE_BUILDING: PendingTickets = Mutex::new(BTreeMap::new());

fn pending_ticket(table: &PendingTickets, key: (u64, u64)) -> Arc<Mutex<()>> {
    let mut map = table.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(existing) = map.get(&key).and_then(Weak::upgrade) {
        return existing;
    }
    if map.len() >= 64 {
        map.retain(|_, weak| weak.strong_count() > 0);
    }
    let ticket = Arc::new(Mutex::new(()));
    map.insert(key, Arc::downgrade(&ticket));
    ticket
}

enum LooseBacking {
    Mapped {
        buffer: ID3D12Resource,
        data: *mut u8,
    },
    Shared {
        buffer: ID3D12Resource,
        data: *mut u8,
        card: Option<ID3D12Resource>,
    },
}

fn make_loose_backing(len: u64) -> Option<LooseBacking> {
    // One buffer in the card's memory, filled by the CPU through the bus.
    if let Some((buffer, data)) = window_mapped_for(len) {
        return Some(LooseBacking::Mapped { buffer, data });
    }
    let (buffer, data) = host()?.shared_buffer(len)?;
    // With room for windows on the card (`window_card_most`): shaders get a
    // buffer there, and the one just made is the CPU's side.
    let card = window_card_for(len);
    Some(LooseBacking::Shared { buffer, data, card })
}

fn build_loose_shadow(start: u64, len: u64, backing: LooseBacking) -> Arc<Shadow> {
    match backing {
        LooseBacking::Mapped { buffer, data } => {
            crate::host::name(&buffer, || format!("title memory {start:#x}+{len:#x} mapped on the card"));
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            let mut shadow = Shadow::new(buffer, data, gpu, start, len);
            shadow.window = true;
            shadow.mapped = true;
            Arc::new(shadow)
        }
        LooseBacking::Shared { buffer, data, card } => {
            crate::host::name(&buffer, || format!("title memory {start:#x}+{len:#x}"));
            let (buffer, card) = match card {
                Some(on_card) => {
                    crate::host::name(&on_card, || format!("title memory {start:#x}+{len:#x} on the card"));
                    let card = Card {
                        staging: buffer,
                        before: Mutex::default(),
                        dirty: Mutex::default(),
                        has_dirty: AtomicBool::new(false),
                        sent: Mutex::default(),
                        has_sent: AtomicBool::new(false),
                        size: len,
                    };
                    (on_card, Some(card))
                }
                None => (buffer, None),
            };
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            tracing::debug!(
                start = format_args!("{start:#x}"),
                len,
                "loose memory copied to the host"
            );
            let mut shadow = Shadow::new(buffer, data, gpu, start, len);
            shadow.window = true;
            shadow.card = card;
            Arc::new(shadow)
        }
    }
}

#[derive(Clone, Copy)]
struct LooseQuery {
    address: u64,
    end: u64,
    unsized_ask: bool,
    use_tails: bool,
    use_holder: bool,
    grid: u64,
}

fn loose_key_with(map: &LooseTable, cell: u64, named: (u64, u64), q: LooseQuery) -> (u64, u64) {
    if q.use_tails {
        let there = map.range((cell, 0)..=(q.address, u64::MAX)).map(|(key, _)| *key);
        longest_tail(there, q.address, q.end).unwrap_or_else(|| tail_of(q.address, q.end))
    } else if q.use_holder {
        // The windows that begin in the ask's cell at or below the ask (one
        // that begins in a cell below and reaches this far is passed over:
        // the ask then makes its own).
        let there = map.range((cell, 0)..=(q.address, u64::MAX)).map(|(key, _)| *key);
        holder(there, q.address, q.end).unwrap_or(named)
    } else {
        named
    }
}

fn loose(address: u64, end: u64, unsized_ask: bool, any: bool) -> Option<Arc<Shadow>> {
    loose_with(
        LooseQuery {
            address,
            end,
            unsized_ask,
            use_tails: unsized_ask && tails(),
            use_holder: any && fine() && !unsized_ask,
            grid: if unsized_ask { WINDOW } else { window() },
        },
        make_loose_backing,
    )
}

fn loose_with(
    q: LooseQuery,
    make: impl Fn(u64) -> Option<LooseBacking>,
) -> Option<Arc<Shadow>> {
    let (cell, _) = window_of(q.address, q.end, WINDOW);
    let named = window_of(q.address, q.end, q.grid);
    // Under the lock, with the handing out: see `let_go_idle`.
    let note = |shadow: &Shadow| {
        // (Read first: the lines are shared by all recording threads.)
        let now = crate::host::moment();
        if shadow.used.load(Ordering::Relaxed) != now {
            shadow.used.store(now, Ordering::Relaxed);
        }
        if shadow.asked[0].load(Ordering::Relaxed) > q.address {
            shadow.asked[0].fetch_min(q.address, Ordering::Relaxed);
        }
        if shadow.asked[1].load(Ordering::Relaxed) < q.end {
            shadow.asked[1].fetch_max(q.end, Ordering::Relaxed);
        }
        if q.unsized_ask {
            shadow.unsized_asks.fetch_add(1, Ordering::Relaxed);
        }
    };
    loop {
        // The window is there already (nearly always): readers share the lock,
        // including tail and fine-grid holder lookups.
        let key = {
            let map = LOOSE.read().unwrap_or_else(|e| e.into_inner());
            let key = loose_key_with(&map, cell, named, q);
            if let Some(shadow) = map.get(&key) {
                note(shadow);
                return Some(shadow.clone());
            }
            key
        };
        // Keep the original transaction for optional imported/placed memory.
        // The outside-lock path below only manufactures independent backing.
        if crate::port::on() || crate::inplace::level() >= 1 {
            let mut map = LOOSE.write().unwrap_or_else(|e| e.into_inner());
            let (start, len) = loose_key_with(&map, cell, named, q);
            if let Some(shadow) = map.get(&(start, len)) {
                note(shadow);
                return Some(shadow.clone());
            }
            // In place, if asked for and the window's memory can be opened: the
            // window is the title's own memory, with nothing to copy.
            if let Some(placed) = crate::port::place_for(start, len, true).or_else(|| crate::inplace::place_for(start, len, true)) {
                let shadow = Arc::new(Shadow::placed(placed, start, len, true));
                note(&shadow);
                map.insert((start, len), shadow.clone());
                return Some(shadow);
            }
            let shadow = build_loose_shadow(start, len, make(len)?);
            if shadow.card.is_some() {
                CARDED.lock().unwrap_or_else(|e| e.into_inner()).push(Arc::downgrade(&shadow));
            }
            note(&shadow);
            map.insert((start, len), shadow.clone());
            return Some(shadow);
        }
        // Serialize concurrent creators of the same key without holding the
        // global LOOSE table lock across D3D12 creation, residency confirmation
        // and CPU mapping.
        let ticket = pending_ticket(&LOOSE_BUILDING, key);
        let _only = ticket.lock().unwrap_or_else(|e| e.into_inner());
        {
            let map = LOOSE.read().unwrap_or_else(|e| e.into_inner());
            let selected = loose_key_with(&map, cell, named, q);
            if let Some(shadow) = map.get(&selected) {
                note(shadow);
                return Some(shadow.clone());
            }
            if selected != key {
                continue;
            }
        }
        let candidate = build_loose_shadow(key.0, key.1, make(key.1)?);
        let mut map = LOOSE.write().unwrap_or_else(|e| e.into_inner());
        let selected = loose_key_with(&map, cell, named, q);
        if let Some(existing) = map.get(&selected) {
            // Another creator published this key or a wider holder/tail while
            // we were allocating. Drop the unpublished candidate outside LOOSE.
            note(existing);
            let existing = existing.clone();
            drop(map);
            drop(_only);
            drop(candidate);
            return Some(existing);
        }
        if selected != key {
            drop(map);
            drop(_only);
            drop(candidate);
            continue;
        }
        if candidate.card.is_some() {
            CARDED.lock().unwrap_or_else(|e| e.into_inner()).push(Arc::downgrade(&candidate));
        }
        note(&candidate);
        map.insert(key, candidate.clone());
        return Some(candidate);
    }
}

type ExactTable = BTreeMap<(u64, u64), (Arc<Shadow>, AtomicU64)>;

/// Copies of single buffer views' own bytes ([`exact`]), by (start,
/// bytes), each with the asking that last took it.
static EXACT: RwLock<ExactTable> = RwLock::new(BTreeMap::new());

/// Askings for single buffer views' own copies so far.
static EXACT_TICK: AtomicU64 = AtomicU64::new(0);

/// Per-key constructor locks for pending single-view allocations.
static EXACT_BUILDING: PendingTickets = Mutex::new(BTreeMap::new());

/// Since the last frame line: askings for such copies, and copies made.
static EXACT_STATS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// A copy of the title memory `[address, address + len)` that starts at
/// `address` itself: for a buffer view whose bytes do not start on an
/// element of the copy its buffer has (`descriptor/host.rs` `view_home`).
/// A host view counts its first element from its buffer's start; a console
/// view starts at any address, and the title places views at ring
/// positions inside its big GPU heap. Such a view was refused and its slot
/// read zeros ("buffer view not on an element of its host copy", in every
/// run; upstream VIctory 86a5ed6 found its meshlet culling constants and
/// colour-bake parameter blocks among them and gives each a copy of its
/// own bytes, as here).
///
/// The copy is brought up to date and written back like any other (a list
/// refreshes the copy its view's slot shows, `buffer_copy`). Kept by
/// (start, bytes) for as long as a list or a slot holds it; of the others,
/// those not asked for lately go once there are many (ring positions
/// change every frame).
pub(crate) fn exact(address: u64, len: u64) -> Option<Arc<Shadow>> {
    exact_with(address, len, |size| host()?.shared_buffer(size))
}

fn exact_with(
    address: u64,
    len: u64,
    make: impl Fn(u64) -> Option<(ID3D12Resource, *mut u8)>,
) -> Option<Arc<Shadow>> {
    /// Copies kept before the idle ones are looked for, and the askings an
    /// idle one stays for.
    const MANY: usize = 256;
    const LATELY: u64 = 512;
    exact_with_limits(address, len, MANY, LATELY, make)
}

fn exact_with_limits(
    address: u64,
    len: u64,
    many: usize,
    lately: u64,
    make: impl Fn(u64) -> Option<(ID3D12Resource, *mut u8)>,
) -> Option<Arc<Shadow>> {
    /// The largest, and no view the title made on purpose: off its dwords.
    const MOST: u64 = 64 << 20;
    if len == 0 || len > MOST || !address.is_multiple_of(4) {
        return None;
    }
    EXACT_STATS[0].fetch_add(1, Ordering::Relaxed);
    let now = EXACT_TICK.fetch_add(1, Ordering::Relaxed) + 1;
    let key = (address, len);
    let touch = |shadow: &Shadow, used: &AtomicU64, tick: u64| {
        used.fetch_max(tick, Ordering::Relaxed);
        let moment = crate::host::moment();
        if shadow.used.load(Ordering::Relaxed) != moment {
            shadow.used.store(moment, Ordering::Relaxed);
        }
    };
    {
        let map = EXACT.read().unwrap_or_else(|e| e.into_inner());
        if let Some((shadow, used)) = map.get(&key) {
            touch(shadow, used, now);
            return Some(shadow.clone());
        }
    }
    let ticket = pending_ticket(&EXACT_BUILDING, key);
    let _only = ticket.lock().unwrap_or_else(|e| e.into_inner());
    {
        let map = EXACT.read().unwrap_or_else(|e| e.into_inner());
        if let Some((shadow, used)) = map.get(&key) {
            touch(shadow, used, EXACT_TICK.load(Ordering::Relaxed).max(now));
            return Some(shadow.clone());
        }
    }
    let (buffer, data) = make(len.next_multiple_of(256))?;
    crate::host::name(&buffer, || format!("title view {address:#x}+{len:#x}"));
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    let candidate = Arc::new(Shadow::new(buffer, data, gpu, address, len));
    let mut evicted = Vec::new();
    let result = {
        let mut map = EXACT.write().unwrap_or_else(|e| e.into_inner());
        let published_now = EXACT_TICK.load(Ordering::Relaxed).max(now);
        if let Some((existing, used)) = map.get(&key) {
            touch(existing, used, published_now);
            let existing = existing.clone();
            drop(map);
            drop(_only);
            drop(candidate);
            return Some(existing);
        }
        if map.len() >= many {
            // Held by the map alone (no list), its buffer by the copy alone (no
            // slot shows it), and not asked for lately.
            let drop_keys: Vec<(u64, u64)> = map
                .iter()
                .filter(|(_, (shadow, used))| {
                    Arc::strong_count(shadow) <= 1
                        && crate::host::references(&shadow.buffer) <= 1
                        && used.load(Ordering::Relaxed).saturating_add(lately) <= published_now
                })
                .map(|(&k, _)| k)
                .collect();
            for k in drop_keys {
                if let Some((shadow, _)) = map.remove(&k) {
                    evicted.push(shadow);
                }
            }
        }
        candidate.used.store(crate::host::moment(), Ordering::Relaxed);
        map.insert(key, (candidate.clone(), AtomicU64::new(published_now)));
        EXACT_STATS[1].fetch_add(1, Ordering::Relaxed);
        OWN_MADE.fetch_add(1, Ordering::Relaxed);
        candidate
    };
    drop(_only);
    drop(evicted);
    Some(result)
}

/// Copies of single views' bytes made since the start.
pub(crate) static OWN_MADE: AtomicU64 = AtomicU64::new(0);

/// With the frame line: the buffer views that got a copy of their own
/// bytes ([`exact`]), per frame over `frames`, and the copies kept now.
/// Nothing while there are none.
pub(crate) fn exact_line(frames: f64) {
    let [asked, made] = [0, 1].map(|i| EXACT_STATS[i].swap(0, Ordering::Relaxed));
    if asked == 0 {
        return;
    }
    let kept = EXACT.read().unwrap_or_else(|e| e.into_inner()).len();
    tracing::info!(
        "buffer views off their copy's elements (they were refused and read zeros; VICTORY_VIEW_BYTES=0 refuses them again): \
         {:.1} asked for per frame, {:.1} copies of their own bytes made per frame, {kept} kept",
        asked as f64 / frames,
        made as f64 / frames
    );
}

#[cfg(test)]
mod loose_allocation_tests;

#[cfg(test)]
mod window_tests {
    use super::{
        WINDOW, cover, default_card_limits_mb, grid, holder, longest_tail, resolve_card_limits,
        tail_of, window_of, window_victims,
    };

    #[test]
    fn only_the_explicit_profile_reserves_full_budgets() {
        use super::{parse_mib, reservation_limits, window_usage_limit};
        const MIB: u64 = 1 << 20;
        assert_eq!(reservation_limits(1536 * MIB, true, 1024 * MIB, false), (1024 * MIB, 0));
        assert_eq!(reservation_limits(1536 * MIB, true, 1024 * MIB, true), (1536 * MIB, 1024 * MIB));
        // A missing or invalid override uses the adapter default, with the 1 GiB reserve cap.
        assert_eq!(parse_mib(" 1536 "), Some(1536));
        let invalid = parse_mib("not a number");
        assert_eq!(invalid, None);
        assert_eq!(reservation_limits(2048 * MIB, invalid.is_some(), 1024 * MIB, true), (1024 * MIB, 1024 * MIB));
        assert_eq!(reservation_limits(512 * MIB, false, 0, false), (512 * MIB, 0));
        assert_eq!(window_usage_limit(1_099, false), 872);
        assert_eq!(window_usage_limit(1_099, true), 970);
    }

    #[test]
    fn texture_admission_reserves_buffers_and_windows_independently() {
        use super::remaining_card_room;
        const MIB: u64 = 1 << 20;
        assert_eq!(remaining_card_room(1536 * MIB, 512 * MIB, 0, 1024 * MIB, 256 * MIB), 1792 * MIB);
        assert_eq!(remaining_card_room(512 * MIB, 256 * MIB, 1024 * MIB, 1024 * MIB, 256 * MIB), 768 * MIB);
        assert_eq!(remaining_card_room(0, u64::MAX, 1, 0, 0), 0);
    }

    #[test]
    fn window_budget_evicts_oldest_only_until_it_fits() {
        const M: u64 = 1 << 20;
        // Oldest first: 10, 20 and 30 MiB. Keeping 50 MiB must free only the
        // oldest copy (60 MiB left before, 50 MiB after).
        let entries = [(10 * M, 1usize, false), (20 * M, 2, false), (30 * M, 3, false)];
        assert_eq!(window_victims(&entries, 60 * M, 50 * M), vec![0]);
        // A tighter cap takes the newest as well.
        assert_eq!(window_victims(&entries, 60 * M, 20 * M), vec![0, 1, 2]);
    }

    #[test]
    fn window_budget_counts_one_copy_under_several_keys_once() {
        const M: u64 = 1 << 20;
        // The first two entries are one copy; 50 MiB is live, not 80.
        // Without dedupe both copies would be picked to reach a 40 MiB cap.
        // The copy goes under both its keys: under one alone it would stay.
        let entries = [(30 * M, 7usize, false), (30 * M, 7, false), (20 * M, 8, false)];
        assert_eq!(window_victims(&entries, 50 * M, 40 * M), vec![0, 1]);
    }

    /// #450 (the issue's row): copies held by a list or a slot, or used
    /// lately, count against the budget too. 3,072 MiB held and 1,536 MiB
    /// idle against 2,048 MiB: every idle copy goes, the held ones stay.
    /// Counting the idle ones alone (1,536 under 2,048) let none go.
    #[test]
    fn window_budget_counts_the_copies_that_cannot_go() {
        const M: u64 = 1 << 20;
        let held = 3_072 * M;
        let idle = [(512 * M, 1usize, false), (512 * M, 2, false), (512 * M, 3, false)];
        let total = held + idle.iter().map(|e| e.0).sum::<u64>();
        assert_eq!(total, 4_608 * M);
        assert_eq!(window_victims(&idle, total, 2_048 * M), vec![0, 1, 2]);
        // Only as many as it takes: with 4,096 MiB kept, the oldest one.
        assert_eq!(window_victims(&idle, total, 4_096 * M), vec![0]);
        // Under the budget, nothing that is not due goes.
        assert!(window_victims(&idle, total, 8_192 * M).is_empty());
    }

    /// #450: a budget never keeps more copies than none. Those idle for the
    /// time the sweep asks without one go even when everything fits.
    #[test]
    fn window_budget_never_keeps_what_the_sweep_lets_go_without_one() {
        const M: u64 = 1 << 20;
        let entries = [(100 * M, 1usize, true), (100 * M, 2, false), (100 * M, 3, true)];
        assert_eq!(window_victims(&entries, 300 * M, 2_048 * M), vec![0, 2]);
        // Over the budget, the others go too, oldest first.
        assert_eq!(window_victims(&entries, 300 * M, 50 * M), vec![0, 1, 2]);
    }

    /// In place a copy holds no bytes of its own: it neither counts against
    /// the budget nor goes to meet it (it still goes once due).
    #[test]
    fn window_budget_passes_over_copies_in_place() {
        const M: u64 = 1 << 20;
        let entries = [(0, 1usize, false), (64 * M, 2, false), (0, 3, true)];
        assert_eq!(window_victims(&entries, 64 * M, 0), vec![1, 2]);
    }

    #[test]
    fn card_limits_scale_with_local_vram_budget() {
        const GIB: u64 = 1 << 30;
        const MIB: u64 = 1 << 20;

        // Unknown (WARP / query failure) and <= 12 GiB keep the 12 GB baseline.
        assert_eq!(default_card_limits_mb(None), (16, 512));
        assert_eq!(default_card_limits_mb(Some(8 * GIB)), (16, 512));
        assert_eq!(default_card_limits_mb(Some(11 * GIB)), (16, 512));
        assert_eq!(default_card_limits_mb(Some(12 * GIB)), (16, 512));

        // 16 GB tier (> 12 GiB, < 20 GiB), including WDDM budgets around 13.5–15 GiB.
        assert_eq!(default_card_limits_mb(Some(13 * GIB)), (4, 2048));
        assert_eq!(default_card_limits_mb(Some(14 * GIB)), (4, 2048));
        assert_eq!(default_card_limits_mb(Some(15 * GIB)), (4, 2048));
        assert_eq!(default_card_limits_mb(Some(16 * GIB)), (4, 2048));

        // 24–32+ GB tier (>= 20 GiB): 1 MiB min, 25% of budget capped at 8192 MiB.
        assert_eq!(default_card_limits_mb(Some(20 * GIB)), (1, 5120));
        assert_eq!(default_card_limits_mb(Some(24 * GIB)), (1, 6144));
        assert_eq!(default_card_limits_mb(Some(32 * GIB)), (1, 8192));
        assert_eq!(default_card_limits_mb(Some(48 * GIB)), (1, 8192));

        // Explicit env overrides always win, clamp min to >= 1 MiB, and saturate on overflow.
        assert_eq!(resolve_card_limits(Some(32 * GIB), Some(8), Some(4096)), (8 * MIB, 4096 * MIB));
        assert_eq!(resolve_card_limits(Some(12 * GIB), Some(0), Some(1024)), (MIB, 1024 * MIB));
        assert_eq!(resolve_card_limits(None, Some(u64::MAX), Some(u64::MAX)), (u64::MAX, u64::MAX));
        assert_eq!(resolve_card_limits(None, None, None), (16 * MIB, 512 * MIB));
    }

    /// An ask that may take any window takes the longest that holds all of
    /// it, and none that holds only a part.
    #[test]
    fn an_ask_takes_the_longest_window_that_holds_it() {
        let cell = 0xd_1200_0000u64;
        let (ask, end) = (cell + 0x9_0000, cell + 0xb_0000);
        assert_eq!(holder([].into_iter(), ask, end), None);
        let there = [
            (cell, 0x4_0000),            // ends below the ask
            (cell + 0x8_0000, 0x2_0000), // holds its first half only
            (cell + 0x8_0000, 0x4_0000), // holds it
            (cell + 0x4_0000, 0x10_0000), // holds it, longer
            (cell + 0xa_0000, 0x40_0000), // begins above it
        ];
        assert_eq!(holder(there.into_iter(), ask, end), Some((cell + 0x4_0000, 0x10_0000)));
        // Of two as long, the one that begins lower.
        let two = [(cell + 0x8_0000, 0x8_0000), (cell + 0x4_0000, 0x8_0000)];
        assert_eq!(holder(two.into_iter(), ask, end), Some((cell + 0x4_0000, 0x8_0000)));
    }

    /// What windows lie over counts each byte once.
    #[test]
    fn overlapping_windows_cover_their_memory_once() {
        assert_eq!(cover([].into_iter()), 0);
        let apart = [(0x10_0000u64, 0x4_0000u64), (0x20_0000, 0x4_0000)];
        assert_eq!(cover(apart.into_iter()), 0x8_0000);
        // One inside another, one overlapping its end, one touching.
        let over = [
            (0x10_0000u64, 0x10_0000u64),
            (0x14_0000, 0x4_0000),
            (0x1c_0000, 0x8_0000),
            (0x24_0000, 0x4_0000),
        ];
        assert_eq!(cover(over.into_iter()), 0x18_0000);
    }

    /// A tail runs from the ask's page to the end of its 4 MiB cell, and an
    /// ask takes the longest tail there is that holds it.
    #[test]
    fn tails_run_to_the_cells_end_and_the_longest_is_taken() {
        let cell = 0xd_1200_0000u64;
        let ask = cell + 0x12_3456;
        assert_eq!(tail_of(ask, ask + 1), (cell + 0x10_0000, WINDOW - 0x10_0000));
        // The cell's first page: the whole cell, the window there always was.
        assert_eq!(tail_of(cell + 0xffff, cell + 0x1_0000), (cell, WINDOW));
        let end = cell + WINDOW;
        let tail = |from: u64| (cell + from, WINDOW - from);
        // Nothing there, or only tails that begin above the ask.
        assert_eq!(longest_tail([].into_iter(), ask, ask + 1), None);
        assert_eq!(longest_tail([tail(0x20_0000)].into_iter(), ask, ask + 1), None);
        // The longest of those that hold the ask; windows that do not reach
        // the cell's end (cut on a finer grid for asks with a length) and
        // windows of other cells are not tails.
        let there = [
            (cell - WINDOW, WINDOW),
            (cell, 0x4_0000),
            tail(0x8_0000),
            tail(0x10_0000),
            (cell + 0x10_0000, 0x4_0000),
            tail(0x20_0000),
        ];
        assert_eq!(longest_tail(there.into_iter(), ask, ask + 1), Some(tail(0x8_0000)));
        let (start, bytes) = tail(0x8_0000);
        assert!(start <= ask && start + bytes == end);
        // The whole cell's window is the longest tail of all.
        let with_whole = [(cell, WINDOW), tail(0x8_0000)];
        assert_eq!(longest_tail(with_whole.into_iter(), ask, ask + 1), Some((cell, WINDOW)));
    }

    /// The setting takes powers of two from 64 KiB to 4 MiB; anything else
    /// leaves the grid as it was.
    #[test]
    fn the_grid_is_a_power_of_two_or_the_default() {
        assert_eq!(grid(None), WINDOW);
        assert_eq!(grid(Some(4096)), WINDOW);
        assert_eq!(grid(Some(1024)), 1 << 20);
        assert_eq!(grid(Some(256)), 256 << 10);
        assert_eq!(grid(Some(64)), 64 << 10);
        for bad in [0, 32, 100, 1000, 8192] {
            assert_eq!(grid(Some(bad)), WINDOW, "{bad}");
        }
    }

    /// A window holds its ask and is cut on the grid; on the default grid
    /// it is the window there always was.
    #[test]
    fn windows_hold_their_ask_on_the_grid() {
        let (address, end) = (0xd_1234_5678u64, 0xd_1234_5678u64 + 0x9000);
        assert_eq!(window_of(address, end, WINDOW), (0xd_1200_0000, 4 << 20));
        assert_eq!(window_of(address, end, 1 << 20), (0xd_1230_0000, 1 << 20));
        assert_eq!(window_of(address, end, 256 << 10), (0xd_1234_0000, 256 << 10));
        // An ask across a grid line takes both cells.
        let end = address + 0x4_0000;
        assert_eq!(window_of(address, end, 256 << 10), (0xd_1234_0000, 512 << 10));
        for g in [64u64 << 10, 256 << 10, 1 << 20, WINDOW] {
            let (start, len) = window_of(address, end, g);
            assert!(start <= address && start + len >= end);
            assert!(start % g == 0 && len % g == 0);
        }
    }
}

/// For device-removed reports: the copy of title memory whose host range
/// starts closest below host GPU address `va`, and where `va` is in it.
pub(crate) fn near(va: u64) -> String {
    let mut all: Vec<(u64, u64, u64)> = buffers()
        .values()
        .filter_map(|e| e.shadow.as_deref().map(|s| (s.gpu, s.start, s.len)))
        .collect();
    all.extend(
        LOOSE
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .map(|s| (s.gpu, s.start, s.len)),
    );
    let below = all.iter().filter(|c| c.0 <= va).max();
    let above = all.iter().filter(|c| c.0 > va).min();
    let mut text = match below {
        Some(&(gpu, start, len)) => format!(
            "; the nearest title memory copy below it: {start:#x}+{len:#x} at host {gpu:#x}, offset {:#x}",
            va - gpu
        ),
        None => "; no title memory copy below it".into(),
    };
    if let Some(&(gpu, start, len)) = above {
        text.push_str(&format!(
            "; the nearest above: {start:#x}+{len:#x} at host {gpu:#x} ({} copies in all)",
            all.len()
        ));
    }
    text
}

#[cfg(test)]
mod copied_and_take_tests {
    use super::*;

    fn assert_copied_invariants(cp: &CopiedPages, oracle: &[u64]) {
        assert_eq!(cp.at, oracle);
        let expected_uncopied = oracle.iter().filter(|&&v| v == 0).count();
        assert_eq!(cp.uncopied, expected_uncopied);
        assert_eq!(cp.all_copied(), expected_uncopied == 0);
        assert_eq!(cp.copied_count(), oracle.len() - expected_uncopied);
        for (i, &val) in oracle.iter().enumerate() {
            let bit = (cp.uncopied_bits[i / 64] >> (i % 64)) & 1;
            assert_eq!(bit == 1, val == 0, "bitmask mismatch at page {i}");
        }
        if let Some(&last_word) = cp.uncopied_bits.last() {
            let rem = oracle.len() % 64;
            if rem != 0 {
                assert_eq!(last_word >> rem, 0, "unused high bits must stay zero");
            }
        }
    }

    #[test]
    fn copied_pages_bitmask_matches_dense_oracle_across_word_boundaries() {
        for pages in [0usize, 1, 63, 64, 65, 127, 128, 129, 257] {
            let mut cp = CopiedPages::new(pages);
            let mut oracle = vec![0u64; pages];
            assert_copied_invariants(&cp, &oracle);

            // Out-of-bounds set is a no-op.
            cp.set(pages, 42);
            cp.set(pages.saturating_add(100), 99);
            assert_copied_invariants(&cp, &oracle);

            // Mark selected pages copied across 64-page word boundaries.
            for idx in 0..pages {
                if idx % 3 != 0 || idx % 64 == 0 || idx % 64 == 63 {
                    let seq = (idx as u64 % 7) + 1;
                    cp.set(idx, seq);
                    oracle[idx] = seq;
                }
            }
            assert_copied_invariants(&cp, &oracle);

            // Reset a subset back to 0 and update existing non-zero entries.
            for idx in (0..pages).step_by(11) {
                cp.set(idx, 0);
                oracle[idx] = 0;
            }
            for idx in (1..pages).step_by(13) {
                let seq = 100 + idx as u64;
                cp.set(idx, seq);
                oracle[idx] = seq;
            }
            assert_copied_invariants(&cp, &oracle);

            // Compare `range_all_copied` and `next_uncopied` against dense scans.
            let probes: Vec<usize> = [0, 1, 2, 62, 63, 64, 65, 126, 127, 128, 129, pages.saturating_sub(1)]
                .into_iter()
                .filter(|&i| i < pages)
                .collect();
            for &lo in &probes {
                for &hi in &probes {
                    if lo > hi {
                        continue;
                    }
                    let expected_all = oracle[lo..=hi].iter().all(|&v| v != 0);
                    assert_eq!(cp.range_all_copied(lo, hi), expected_all, "pages={pages} {lo}..={hi}");
                    let expected_next = (lo..=hi).find(|&i| oracle[i] == 0);
                    assert_eq!(cp.next_uncopied(lo, hi), expected_next, "pages={pages} {lo}..={hi}");
                }
            }

            // Fill remaining pages and confirm fast-path completion.
            for idx in 0..pages {
                cp.set(idx, 500 + idx as u64);
                oracle[idx] = 500 + idx as u64;
            }
            assert_copied_invariants(&cp, &oracle);
            if pages > 0 {
                assert!(cp.range_all_copied(0, pages - 1));
                assert_eq!(cp.next_uncopied(0, pages - 1), None);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn plan_stale_oracle(
        oracle: &mut [u64],
        base: usize,
        start: usize,
        end: usize,
        from: usize,
        to: usize,
        now: u64,
        ranged: u64,
        written: &[(usize, u64)],
    ) -> (Vec<(usize, usize, bool)>, Vec<(usize, u64)>) {
        let mut runs = Vec::new();
        let mut again_at = Vec::new();
        if from >= to {
            return (runs, again_at);
        }
        let mut run: Option<(usize, bool)> = None;
        let last = (to - 1) / PAGE;
        for page in from / PAGE..=last {
            let write = written
                .binary_search_by_key(&page, |&(p, _)| p)
                .map_or(0, |i| written[i].1);
            let at = &mut oracle[page - base];
            let taken = (*at).max(ranged);
            if taken == 0 || write > taken {
                if (page * PAGE).max(start) >= from && ((page + 1) * PAGE).min(end) <= to {
                    *at = now;
                }
                let again = taken != 0;
                if again {
                    again_at.push((page, taken));
                }
                if let Some((first, known)) = run
                    && known != again
                {
                    runs.push(((first * PAGE).max(from), (page * PAGE).min(to), known));
                    run = None;
                }
                run.get_or_insert((page, again));
            } else if let Some((first, known)) = run.take() {
                runs.push(((first * PAGE).max(from), (page * PAGE).min(to), known));
            }
        }
        if let Some((first, known)) = run {
            runs.push(((first * PAGE).max(from), ((last + 1) * PAGE).min(to), known));
        }
        (runs, again_at)
    }

    #[test]
    fn plan_stale_matches_dense_oracle_with_ranged_and_partial_init() {
        const PAGES: usize = 160;
        let base = 0x400usize;
        let start = base * PAGE + 123;
        let end = (base + PAGES) * PAGE - 77;

        let mut oracle = vec![0u64; PAGES];
        let mut fast = CopiedPages::new(PAGES);

        // Partial initialization spanning 64-page word boundaries.
        for idx in (0..PAGES).filter(|i| i % 3 != 0 || *i == 63 || *i == 64 || *i == 127 || *i == 128) {
            let seq = (idx as u64 % 5) + 1;
            oracle[idx] = seq;
            fast.set(idx, seq);
        }

        let written = [
            (base - 2, 99),
            (base + 1, 10),
            (base + 2, 10),
            (base + 63, 2),
            (base + 64, 12),
            (base + 65, 12),
            (base + 66, 12),
            (base + 127, 15),
            (base + 128, 1),
            (base + 159, 20),
            (base + PAGES + 5, 99),
        ];

        // Mix unaligned ranges, partial pages, `ranged == 0`, and `ranged > 0`.
        let queries = [
            (start, start + 16, 0u64),
            (start, start + 16, 5u64),
            (start, (base + 4) * PAGE, 0u64),
            ((base + 62) * PAGE + 10, (base + 66) * PAGE - 10, 0u64),
            ((base + 62) * PAGE + 10, (base + 66) * PAGE - 10, 11u64),
            ((base + 62) * PAGE + 10, (base + 66) * PAGE - 10, 25u64),
            ((base + 126) * PAGE, (base + 130) * PAGE, 0u64),
            (start, end, 0u64),
            // Once fully initialized, verify clean and dirty fast paths.
            (start, end, 0u64),
            (start, end, 50u64),
        ];

        for (step, &(from, to, ranged)) in queries.iter().enumerate() {
            let now = 30 + step as u64;
            let expected = plan_stale_oracle(&mut oracle, base, start, end, from, to, now, ranged, &written);
            let got = fast.plan_stale(base, start, end, from, to, now, ranged, &written);
            assert_eq!(got, expected, "step {step} [{from:#x}..{to:#x}) ranged={ranged}");
            assert_copied_invariants(&fast, &oracle);
        }
    }

    #[test]
    fn stale_pages_in_and_has_stale_in_match_dense_oracle() {
        const PAGES: usize = 140;
        let base = 0x200usize;
        let mut oracle = vec![0u64; PAGES];
        let mut fast = CopiedPages::new(PAGES);
        for idx in (0..PAGES).filter(|i| i % 4 != 0) {
            let seq = (idx as u64 % 6) + 1;
            oracle[idx] = seq;
            fast.set(idx, seq);
        }
        let written = [
            (base - 3, 50),
            (base + 1, 10),
            (base + 63, 2),
            (base + 64, 10),
            (base + 127, 10),
            (base + 128, 1),
            (base + 139, 10),
            (base + PAGES + 2, 50),
        ];
        let at = |p: usize| p * PAGE;
        let ranges = [
            (at(base), at(base)),
            (at(base - 4), at(base)),
            (at(base + PAGES), at(base + PAGES + 4)),
            (at(base) + 17, at(base + 3) + 5),
            (at(base + 62), at(base + 67)),
            (at(base + 125), at(base + 131)),
            (at(base - 2), at(base + PAGES + 3)),
        ];
        for &(a, e) in &ranges {
            let expected = stale_pages_in(base, &oracle, a, e, &written);
            assert_eq!(fast.stale_pages_in(base, a, e, &written), expected, "[{a:#x}..{e:#x})");
            assert_eq!(fast.has_stale_in(base, a, e, &written), !expected.is_empty(), "[{a:#x}..{e:#x})");
        }
        // Mark all pages copied and re-verify fast path.
        for idx in 0..PAGES {
            oracle[idx] = 5;
            fast.set(idx, 5);
        }
        for &(a, e) in &ranges {
            let expected = stale_pages_in(base, &oracle, a, e, &written);
            assert_eq!(fast.stale_pages_in(base, a, e, &written), expected, "all-copied [{a:#x}..{e:#x})");
            assert_eq!(fast.has_stale_in(base, a, e, &written), !expected.is_empty(), "all-copied [{a:#x}..{e:#x})");
        }
    }

    #[test]
    fn lock_free_pending_mirror_matches_pending_state() {
        for card in [false, true] {
            let mut buf = vec![0x5Au8; 2 * PAGE];
            let start = buf.as_ptr() as usize;
            let shadow = Shadow::unbacked(start as u64, buf.as_mut_ptr(), buf.len() as u64, card);
            assert!(!shadow.writes_pending());
            assert!(!shadow.has_pending.load(Ordering::Acquire));
            assert!(!shadow.writes_pending_in(start, start + PAGE));
            assert!(!shadow.stale_under_writes(start, PAGE));
            let mut runs = 16;
            assert!(!shadow.merge_under_writes(start, PAGE, &mut runs));

            let r1 = (start, start + 4096);
            let r2 = (start + PAGE, start + PAGE + 4096);
            shadow.writing(r1.0, r1.1 - r1.0);
            if !card {
                take_snapshots();
            }
            assert!(shadow.writes_pending());
            assert!(shadow.has_pending.load(Ordering::Acquire));
            assert!(shadow.writes_pending_in(r1.0, r1.1));
            assert!(!shadow.writes_pending_in(r2.0, r2.1));

            // Duplicate declaration keeps mirror set.
            shadow.writing(r1.0, r1.1 - r1.0);
            shadow.writing(r2.0, r2.1 - r2.0);
            if !card {
                take_snapshots();
            }
            assert!(shadow.writes_pending());
            assert!(shadow.writes_pending_in(r2.0, r2.1));

            // Completing one range leaves mirror true while the other remains pending.
            shadow.copy_back_done(&[r1]);
            assert!(shadow.writes_pending());
            assert!(shadow.has_pending.load(Ordering::Acquire));
            assert!(!shadow.writes_pending_in(r1.0, r1.1));
            assert!(shadow.writes_pending_in(r2.0, r2.1));

            // Completing the last range clears the mirror.
            shadow.copy_back_done(&[r2]);
            assert!(!shadow.writes_pending());
            assert!(!shadow.has_pending.load(Ordering::Acquire));
            assert!(!shadow.writes_pending_in(start, start + 2 * PAGE));
        }
    }

    #[test]
    fn lock_free_card_dirty_and_sent_mirrors_match_transitions() {
        let mut title = vec![0x11u8; PAGE];
        let mut side = vec![0x11u8; PAGE];
        let start = title.as_ptr() as usize;
        let shadow = Shadow::unbacked(start as u64, side.as_mut_ptr(), side.len() as u64, true);
        let _tracked = shadow.track_carded_test();
        let card = shadow.card.as_ref().expect("carded copy");

        assert!(!card.has_dirty.load(Ordering::Acquire));
        assert!(!card.has_sent.load(Ordering::Acquire));
        assert!(shadow.card_dirty_now().is_empty());

        // Empty range does not mark dirty.
        shadow.card_changed(64, 64);
        assert!(!card.has_dirty.load(Ordering::Acquire));

        // Recording runs sets `has_dirty` and exposes them via `card_dirty_now`.
        shadow.card_changed_runs([(16, 32), (32, 48), (128, 160)].into_iter());
        assert!(card.has_dirty.load(Ordering::Acquire));
        assert_eq!(shadow.card_dirty_now(), [(16, 48), (128, 160)]);

        // `card_dirty` collects and clears `has_dirty` (and populates `sent` if relay is on).
        let taken = card_dirty(10);
        let mine = taken
            .iter()
            .find(|(s, _)| Arc::ptr_eq(s, &shadow))
            .map(|(_, r)| r.clone())
            .unwrap_or_default();
        assert_eq!(mine, [(16, 48), (128, 160)]);
        assert!(!card.has_dirty.load(Ordering::Acquire));
        assert!(shadow.card_dirty_now().is_empty());
        assert_eq!(card.has_sent.load(Ordering::Acquire), relay());

        // `send_up` transitions dirty -> sent and clears `has_sent` once pruned.
        shadow.card_changed(256, 320);
        assert!(card.has_dirty.load(Ordering::Acquire));
        assert_eq!(shadow.send_up(20, 0), [(256, 320)]);
        assert!(!card.has_dirty.load(Ordering::Acquire));
        assert!(card.has_sent.load(Ordering::Acquire));
        assert!(shadow.send_up(21, 20).is_empty());
        assert!(!card.has_sent.load(Ordering::Acquire));

        // `take_dirty` clears `has_dirty`.
        shadow.card_changed(400, 464);
        assert!(card.has_dirty.load(Ordering::Acquire));
        assert_eq!(shadow.take_dirty(), [(400, 464)]);
        assert!(!card.has_dirty.load(Ordering::Acquire));

        // Verify `card_merge_relay` across all 4 `(has_dirty, has_sent)` mirror combinations.
        for &(with_dirty, with_sent) in &[(false, false), (true, false), (false, true), (true, true)] {
            title.fill(0x20);
            side.fill(0x20);
            let mut gpu = vec![0x20u8; 256];
            // GPU wrote [0..16), [64..80), [128..144).
            gpu[0..16].fill(0xA1);
            gpu[64..80].fill(0xA2);
            gpu[128..144].fill(0xA3);
            if with_dirty {
                title[64..80].fill(0x71);
                side[64..80].fill(0x71);
                shadow.card_changed(64, 80);
            }
            if with_sent {
                title[128..144].fill(0x72);
                side[128..144].fill(0x72);
                shadow.card_changed(128, 144);
                shadow.send_up(30, 0);
                // Re-apply dirty after `send_up` cleared it when testing both together.
                if with_dirty {
                    shadow.card_changed(64, 80);
                }
            }
            assert_eq!(card.has_dirty.load(Ordering::Acquire), with_dirty);
            assert_eq!(card.has_sent.load(Ordering::Acquire), with_sent);

            let mut scratch = Vec::new();
            let _ = shadow.card_merge_relay(0, &gpu, 25, &mut scratch);
            assert_eq!(&title[0..16], &[0xA1; 16], "dirty={with_dirty} sent={with_sent}");
            assert_eq!(
                &title[64..80],
                if with_dirty { &[0x71; 16] } else { &[0xA2; 16] },
                "dirty={with_dirty} sent={with_sent}"
            );
            assert_eq!(
                &title[128..144],
                if with_sent { &[0x72; 16] } else { &[0xA3; 16] },
                "dirty={with_dirty} sent={with_sent}"
            );

            // Reset dirty and sent before the next case.
            shadow.take_dirty();
            shadow.send_up(31, 30);
            assert!(!card.has_dirty.load(Ordering::Acquire));
            assert!(!card.has_sent.load(Ordering::Acquire));
        }
    }

    #[test]
    fn word_runs_and_put_match_scalar_across_alignments_and_tails() {
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };

        let lengths = [
            0usize, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 31, 32, 33, 35, 36, 63, 64, 65, 252, 255, 256,
            257, 259, 260, 287, 288, 289, 511, 512, 515, 4096, 4099,
        ];
        for at in [0x1000usize, 0x1001, 0x1002, 0x1003] {
            for &len in &lengths {
                let held_base: Vec<u8> = (0..len).map(|i| ((i * 19 + 5) & 0xff) as u8).collect();

                // Check identical, all-differing, boundary-byte, and mixed sparse/dense patterns.
                for pattern in 0..4 {
                    let mut title = held_base.clone();
                    match pattern {
                        0 => {}
                        1 => {
                            for b in &mut title {
                                *b ^= 0xFF;
                            }
                        }
                        2 => {
                            for &pos in &[0usize, 1, 2, 3, 4, 31, 32, 33, 255, 256, 257, len.saturating_sub(1)] {
                                if pos < len {
                                    title[pos] ^= 0x5A;
                                }
                            }
                        }
                        _ => {
                            let edits = ((next() as usize) % 24).min(len);
                            for _ in 0..edits {
                                let pos = (next() as usize) % len;
                                let span = [1usize, 2, 4, 12, 40][(next() as usize) % 5].min(len - pos);
                                for b in &mut title[pos..pos + span] {
                                    *b ^= 0x3C;
                                }
                            }
                        }
                    }

                    let expected = dword_runs(at, &held_base, &title, 0);
                    assert_eq!(
                        word_runs(at, &title, &held_base),
                        expected,
                        "word_runs at={at:#x} len={len} pattern={pattern}"
                    );

                    let mut held_put = held_base.clone();
                    assert_eq!(
                        word_runs_put(at, &title, &mut held_put),
                        expected,
                        "word_runs_put at={at:#x} len={len} pattern={pattern}"
                    );
                    assert_eq!(held_put, title, "word_runs_put bytes at={at:#x} len={len} pattern={pattern}");

                    let mut held_chunk = held_base.clone();
                    assert_eq!(
                        take_chunk(at, &title, &mut held_chunk, false),
                        expected,
                        "take_chunk(!whole) at={at:#x} len={len} pattern={pattern}"
                    );
                    assert_eq!(held_chunk, title);

                    let mut held_whole = held_base.clone();
                    let expected_whole = if len == 0 { Vec::new() } else { vec![(0, len)] };
                    assert_eq!(take_chunk(at, &title, &mut held_whole, true), expected_whole);
                    assert_eq!(held_whole, title);
                }
            }
        }
    }

    #[test]
    fn shadow_take_matches_scalar_oracle_across_alignments_and_pages() {
        let total = 2 * PAGE + 515;
        let mut title_mem: Vec<u8> = (0..total).map(|i| ((i * 29 + 7) & 0xff) as u8).collect();
        let initial_side = title_mem.clone();

        // Mutate bytes spanning prefix, 32B/256B boundaries, cross-page boundary, and unaligned tail.
        for &pos in &[
            1usize,
            2,
            19,
            255,
            256,
            257,
            4095,
            4096,
            PAGE - 3,
            PAGE - 1,
            PAGE,
            PAGE + 3,
            PAGE + 300,
            2 * PAGE + 514,
        ] {
            title_mem[pos] ^= 0x6B;
        }

        let start = title_mem.as_ptr() as usize;
        for &(from_off, to_off) in &[
            (0usize, total),
            (1, total - 2),
            (3, PAGE + 37),
            (PAGE - 5, PAGE + 261),
            (PAGE + 2, PAGE + 291),
        ] {
            let mut side = initial_side.clone();
            let copy = Shadow::unbacked(start as u64, side.as_mut_ptr(), total as u64, true);
            copy.keep_stores();

            // `known = true`: compare and update changed dwords only.
            let taken = copy.take(start + from_off, start + to_off, true);
            assert_eq!(taken, to_off - from_off);
            let got_dirty = joined(copy.take_dirty());

            // Scalar oracle across the exact same page-split chunks as `Shadow::take`.
            let mut oracle_side = initial_side.clone();
            let mut oracle_dirty = Vec::new();
            let mut at = start + from_off;
            let to = start + to_off;
            while at < to {
                let n = (PAGE - at % PAGE).min(to - at);
                let off = at - start;
                let runs = dword_runs(at, &oracle_side[off..off + n], &title_mem[off..off + n], 0);
                for (a, len) in runs {
                    oracle_side[off + a..off + a + len].copy_from_slice(&title_mem[off + a..off + a + len]);
                    oracle_dirty.push((off + a, off + a + len));
                }
                at += n;
            }
            assert_eq!(side, oracle_side, "bytes [{from_off}..{to_off})");
            assert_eq!(got_dirty, joined(oracle_dirty), "dirty [{from_off}..{to_off})");

            // `known = false` with no `card.before`: takes the whole range.
            let mut side_whole = vec![0u8; total];
            let copy_whole = Shadow::unbacked(start as u64, side_whole.as_mut_ptr(), total as u64, true);
            copy_whole.keep_stores();
            assert_eq!(copy_whole.take(start + from_off, start + to_off, false), to_off - from_off);
            assert_eq!(&side_whole[from_off..to_off], &title_mem[from_off..to_off]);
            assert_eq!(joined(copy_whole.take_dirty()), [(from_off, to_off)]);
        }
    }

    #[test]
    #[ignore = "manual benchmark: cargo test -p victory-gpu --release bench_copied_pages_and_word_runs -- --ignored --nocapture"]
    fn bench_copied_pages_and_word_runs() {
        use std::time::Instant;
        const MIB: usize = 1 << 20;

        {
            const PAGES: usize = 2560;
            let base = 0x1000usize;
            let mut cp = CopiedPages::new(PAGES);
            for i in 0..PAGES {
                cp.set(i, 1);
            }
            let written = [(base + 10, 2), (base + 500, 2), (base + 1200, 2), (base + 2500, 2)];
            let (start, end) = (base * PAGE, (base + PAGES) * PAGE);
            let iters = 20_000u32;
            let t0 = Instant::now();
            let mut runs = 0usize;
            for i in 0..iters {
                let (r, _) = cp.plan_stale(base, start, end, start, end, 3 + u64::from(i), 0, &written);
                runs += r.len();
            }
            std::hint::black_box(runs);
            let ns = t0.elapsed().as_nanos() as f64 / f64::from(iters);
            eprintln!("plan_stale (160 MiB, 4 dirty): {ns:8.1} ns/call");
        }

        {
            const LEN: usize = 64 * MIB;
            let held_init: Vec<u8> = (0..LEN).map(|i| (i.wrapping_mul(17) & 0xff) as u8).collect();
            let mut title = held_init.clone();
            for off in (0..LEN).step_by(4096) {
                title[off] ^= 0x55;
                title[off + 64] ^= 0x33;
            }

            let t0 = Instant::now();
            let mut scalar_runs = 0usize;
            for c in 0..(LEN / PAGE) {
                let lo = c * PAGE;
                scalar_runs += dword_runs(lo, &held_init[lo..lo + PAGE], &title[lo..lo + PAGE], 0).len();
            }
            std::hint::black_box(scalar_runs);
            let scalar_ms = t0.elapsed().as_secs_f64() * 1e3;

            let mut held_avx2 = held_init.clone();
            let t0 = Instant::now();
            let mut avx2_runs = 0usize;
            for c in 0..(LEN / PAGE) {
                let lo = c * PAGE;
                avx2_runs += word_runs_put(lo, &title[lo..lo + PAGE], &mut held_avx2[lo..lo + PAGE]).len();
            }
            std::hint::black_box(avx2_runs);
            let avx2_ms = t0.elapsed().as_secs_f64() * 1e3;
            assert_eq!(avx2_runs, scalar_runs);
            assert_eq!(held_avx2, title);

            eprintln!("64 MiB Shadow::take diff: scalar {scalar_ms:.2} ms | AVX2 word_runs_put {avx2_ms:.2} ms");
        }
    }
}
