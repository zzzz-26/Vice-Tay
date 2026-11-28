//! The title's buffer copies, done on the card at their place in the list
//! (`VICTORY_COPIES_IN_PLACE`, on by default; `0`: all at submission, on the
//! CPU, as before).
//!
//! A buffer copy was done in title memory when its list was submitted, before
//! any draw of the submission ran, and the host copies refreshed after it.
//! That is another order than the list's: draws ahead of the copy read what
//! it put; of two copies to the same bytes with a draw behind each, both
//! draws read the second; a source that a dispatch of the same submission
//! writes first was read as it was before it; and a destination the
//! submissions in the air still read changed under them (where its host copy
//! has one side only, `crate::shadow::card_all`).
//!
//! Now the list copies, in the destination's host copy: out of the source's
//! host copy where shaders write that one, else out of upload memory of the
//! list that takes the source's bytes when the list is submitted (a source
//! shaders never wrote is the title's CPU's, or a copy's put there earlier in
//! the submission). What is written there is a shader's write from then on:
//! the bytes are declared, and what the card made of them is put into title
//! memory before the title is told of the submission.
//!
//! Title memory takes the copy at submission as well, as it did: fills of
//! upload memory, textures and host copies that lie over the same bytes read
//! it there. The reference of the destination's host copy takes the same
//! bytes ([`Shadow::placed_copy`]), so what the landing finds is what the card
//! made of them afterwards (a counter copied to zero and counted up; a source
//! that a shader wrote in between), and nothing is sent up over it. Where
//! flights in the air may write the source or the destination this is left to
//! the landing: what they wrote is not in title memory yet, and what they
//! will put there must not stand over this copy's bytes. If a step of the
//! submission reads the destination on the CPU, it waits for them instead
//! ([`Placed::waits`]).
//!
//! Copies to memory that has no host copy yet are done as before: nothing on
//! the card has read or written it. So are those of more than 8 MiB out of
//! memory shaders do not write (streaming).
//!
//! With `VICTORY_PLACED_CONSTANTS=1`, a placed copy can supply a constant
//! buffer that it covers in the same recording. Keep the copy at its
//! original position and snapshot its result on the GPU when the table is
//! made, so the CPU does not have to wait for the title memory to land.
//! Identical requests from the same placed writes share a snapshot. Adjacent
//! writes can supply one buffer when they cover it in the same host copy.
//! For partial coverage, refresh the missing slices of that host copy before
//! reading it on the GPU. A write invalidates any cached snapshot it overlaps.

use std::sync::Arc;
#[cfg(feature = "diagnostics")]
use std::sync::atomic::AtomicU64;

use super::*;
use crate::shadow::Shadow;

pub(super) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_COPIES_IN_PLACE").is_none_or(|v| v != "0"))
}

/// The most bytes copied out of memory shaders do not write.
const MOST: usize = 8 << 20;

/// Since the last frame-rate line: copies at their place, their bytes, those
/// out of a host copy, and texture copies out of one.
pub(super) static STATS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

#[cfg(feature = "diagnostics")]
static CONSTANT_ALLOCATED: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "diagnostics")]
static CONSTANT_REUSED: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "diagnostics")]
static CONSTANT_PARTIAL: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "diagnostics")]
static CONSTANT_PARTIAL_REUSED: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "diagnostics")]
static CONSTANT_ALLOCATION_FAILURES: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "diagnostics")]
static CONSTANT_ALLOCATION_NS: AtomicU64 = AtomicU64::new(0);

#[cfg(feature = "diagnostics")]
pub(super) fn constant_snapshots_line(frames: f64) {
    let copies = CONSTANT_ALLOCATED.swap(0, Ordering::Relaxed);
    let reused = CONSTANT_REUSED.swap(0, Ordering::Relaxed);
    let partial = CONSTANT_PARTIAL.swap(0, Ordering::Relaxed);
    let partial_reused = CONSTANT_PARTIAL_REUSED.swap(0, Ordering::Relaxed);
    let failures = CONSTANT_ALLOCATION_FAILURES.swap(0, Ordering::Relaxed);
    let allocation_ns = CONSTANT_ALLOCATION_NS.swap(0, Ordering::Relaxed);
    if copies != 0 || reused != 0 || failures != 0 {
        tracing::info!(
            copies_per_frame = copies as f64 / frames,
            reused_per_frame = reused as f64 / frames,
            partial_copies_per_frame = partial as f64 / frames,
            partial_reused_per_frame = partial_reused as f64 / frames,
            allocation_failures_per_frame = failures as f64 / frames,
            allocation_thread_ms_per_frame = allocation_ns as f64 / 1e6 / frames,
            "placed constant snapshots"
        );
    }
}

fn placed_constants_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_PLACED_CONSTANTS"))
}

/// A placed copy in the current Deferred recording. Reset drops the sources.
pub(super) struct ConstantSource {
    shadow: Arc<Shadow>,
    span: (usize, usize),
}

pub(super) struct ConstantSnapshot {
    index: usize,
    gpu: u64,
    shadow_id: u64,
}

const CONSTANT_BUFFER_BYTES: usize = 64 << 10;

fn covered(span: (usize, usize), from: usize, len: usize) -> bool {
    len != 0 && span.0 <= from && from.checked_add(len).is_some_and(|end| end <= span.1)
}

impl Deferred {
    /// One host copy with placed writes covering the whole range.
    fn constant_coverage(&self, address: usize, len: usize) -> Option<Arc<Shadow>> {
        let at = self.placed_constants
            .partition_point(|source| source.span.0 <= address)
            .checked_sub(1)?;
        let first = &self.placed_constants[at];
        if len == 0 || first.span.1 <= address {
            return None;
        }
        if covered(first.span, address, len) {
            return Some(first.shadow.clone());
        }
        let end = address.checked_add(len)?;
        let mut through = first.span.1;
        for source in &self.placed_constants[at + 1..] {
            if source.span.0 != through || !Arc::ptr_eq(&source.shadow, &first.shadow) {
                return None;
            }
            through = source.span.1;
            if through >= end {
                return Some(first.shadow.clone());
            }
        }
        None
    }

    /// The overlapping writes must use one host copy. Returns the parts to refresh.
    fn partial_constant_source(&self, address: usize, end: usize) -> Option<(Arc<Shadow>, Vec<(usize, usize)>)> {
        let first = self.placed_constants.partition_point(|source| source.span.1 <= address);
        let sources = &self.placed_constants[first..];
        let source = sources.first()?;
        if source.span.0 >= end {
            return None;
        }
        let mut gaps = Vec::new();
        let mut from = address;
        for part in sources.iter().take_while(|part| part.span.0 < end) {
            if !Arc::ptr_eq(&source.shadow, &part.shadow) {
                return None;
            }
            if from < part.span.0 {
                gaps.push((from, part.span.0));
            }
            from = part.span.1.min(end);
        }
        if from < end {
            gaps.push((from, end));
        }
        Some((source.shadow.clone(), gaps))
    }

    pub(super) fn forget_placed_constants(&mut self, from: usize, len: usize) {
        if len == 0 || (self.placed_constants.is_empty() && self.constant_values.is_empty()) {
            return;
        }
        self.constant_write_version = self.constant_write_version.wrapping_add(1);
        let to = from.saturating_add(len);
        let first = self.placed_constants.partition_point(|source| source.span.1 <= from);
        let last = self.placed_constants.partition_point(|source| source.span.0 < to);
        if first < last {
            drop(self.placed_constants.drain(first..last));
        }

        // A constant buffer starting more than 64 KiB earlier cannot reach the write.
        let first = (from.saturating_sub(CONSTANT_BUFFER_BYTES), 0);
        let affected: Vec<_> = self.constant_values
            .range(first..(to, 0))
            .filter_map(|(&key, _)| (key.0 + key.1 > from).then_some(key))
            .collect();
        for key in affected {
            self.constant_values.remove(&key);
        }
    }
}

/// What the CPU does at submission for a copy at its place in a list.
pub(super) struct Placed {
    pub(super) dst: usize,
    pub(super) src: usize,
    pub(super) len: usize,
    /// The destination's host copy.
    pub(super) shadow: std::sync::Arc<Shadow>,
    /// The exact source bound to the GPU copy, when it does not read upload memory.
    pub(super) source: Option<std::sync::Arc<Shadow>>,
    /// Upload memory the list copies from, to be given the source's bytes.
    from: Option<usize>,
}

impl Placed {
    /// Whether this placed copy reads from CPU upload memory instead of a
    /// source host copy on the card.
    pub(super) fn reads_upload(&self) -> bool {
        self.from.is_some()
    }

    /// Whether flights that write `written` are waited for.
    pub(super) fn waits(&self, written: &[(usize, usize)]) -> Waits {
        let source = touches(written, self.src, self.src + self.len);
        if source && self.from.is_some() {
            return Waits::Yes;
        }
        match source || touches(written, self.dst, self.dst + self.len) {
            true => Waits::ForReaders,
            false => Waits::No,
        }
    }

    /// At submission, in the order of the steps: the list's upload memory
    /// takes the source's bytes and, if `free` (no flight in the air may write
    /// source or destination), title memory at the destination does. Gives
    /// the bytes if it took them.
    pub(super) fn prepare(&self, free: bool) -> Option<Vec<u8>> {
        if !free && self.from.is_none() {
            return None;
        }
        let mut bytes = vec![0u8; self.len];
        // (What is not there reads as zeros, as any copy's source.)
        copy_mapped(bytes.as_mut_ptr(), self.src as *const u8, self.len, false);
        if let Some(to) = self.from {
            // SAFETY: the list's own upload memory of that many bytes.
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), to as *mut u8, self.len) };
        }
        if !free {
            return None;
        }
        copy_mapped(self.dst as *mut u8, bytes.as_ptr(), self.len, true);
        crate::watch::wrote(self.dst, self.len);
        // Title memory that is not all there was left alone.
        let mut there = vec![0u8; self.len];
        (read_into(self.dst as *const u8, &mut there) && there == bytes).then_some(bytes)
    }
}

/// Whether a copy waits for flights before its submission is prepared: not;
/// yes, as it takes a source on the CPU that they may write; or only if
/// another step of the submission reads its destination on the CPU.
pub(super) enum Waits {
    No,
    ForReaders,
    Yes,
}

/// Whether `VICTORY_PLACED_READERS=0` also counts readers ahead of a copy.
pub(super) fn readers_before_copy() -> bool {
    static ALL: OnceLock<bool> = OnceLock::new();
    *ALL.get_or_init(|| std::env::var_os("VICTORY_PLACED_READERS").is_some_and(|v| v == "0"))
}

/// Whether a CPU step reads the destination of a placed copy which still
/// depends on a flight. CPU readers ahead of the copy can use the old bytes.
pub(super) fn read_by(runs: &[&HostList], written: &[(usize, usize)]) -> bool {
    let all = readers_before_copy();
    let mut ranges: Vec<(usize, usize)> = Vec::new();
    let waits = |p: &Placed| matches!(p.waits(written), Waits::ForReaders);
    if all {
        for list in runs {
            let steps = &list.deferred().copies;
            let placed = steps.iter().filter_map(|s| if let Step::Placed(p) = s { Some(p) } else { None });
            let placed: Vec<_> = placed.filter(|p| waits(p)).map(|p| (p.dst, p.dst + p.len)).collect();
            join_ranges(&mut ranges, placed);
        }
    }
    runs.iter().any(|list| {
        list.deferred().copies.iter().any(|step| {
            // What reads, by kind ([`READERS`]).
            let (read, kind) = match step {
                Step::Placed(p) => {
                    let read = p.from.is_some() && touches(&ranges, p.src, p.src + p.len);
                    if !all && waits(p) {
                        join_ranges(&mut ranges, [(p.dst, p.dst + p.len)]);
                    }
                    (read, 0)
                }
                _ if ranges.is_empty() => (false, 0),
                &Step::Copy(_, src, len) => (touches(&ranges, src, src + len), 1),
                &Step::Fill(_, src, len, _) => (touches(&ranges, src, src + len), 2),
                Step::Texture(staging) => {
                    let (at, len) = staging.memory();
                    (touches(&ranges, at, at + len), 3)
                }
                &Step::Plane((at, len), _) => (touches(&ranges, at, at.saturating_add(len)), 4),
                Step::Host(_, (from, to)) => (touches(&ranges, *from, *to), 4),
                _ => (false, 0),
            };
            #[cfg(feature = "diagnostics")]
            if read {
                READERS[kind].fetch_add(1, Ordering::Relaxed);
            }
            let _ = kind;
            read
        })
    })
}

/// Diagnostics: the steps found reading what a copy at its place writes
/// ([`read_by`]) since the last frame line: another such copy, a buffer copy,
/// a fill, a texture, a step of ours.
#[cfg(feature = "diagnostics")]
pub(super) static READERS: [std::sync::atomic::AtomicU64; 5] = [const { std::sync::atomic::AtomicU64::new(0) }; 5];

/// Everything recorded so far is done, and seen, before what follows.
///
/// # Safety
/// An open list.
unsafe fn wait_for_all(host: &Host, list: &ID3D12GraphicsCommandList) {
    static ENHANCED: OnceLock<bool> = OnceLock::new();
    let enhanced = *ENHANCED.get_or_init(|| {
        let mut o = D3D12_FEATURE_DATA_D3D12_OPTIONS12::default();
        // SAFETY: a feature structure of its size.
        let known = unsafe {
            host.device.CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS12,
                (&raw mut o).cast(),
                size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS12>() as u32,
            )
        };
        known.is_ok() && o.EnhancedBarriersSupported.as_bool()
    });
    // SAFETY: the caller's.
    unsafe {
        if enhanced && let Ok(list) = list.cast::<ID3D12GraphicsCommandList7>() {
            let all = [D3D12_GLOBAL_BARRIER {
                SyncBefore: D3D12_BARRIER_SYNC_ALL,
                SyncAfter: D3D12_BARRIER_SYNC_ALL,
                AccessBefore: D3D12_BARRIER_ACCESS_COMMON,
                AccessAfter: D3D12_BARRIER_ACCESS_COMMON,
            }];
            list.Barrier(&[D3D12_BARRIER_GROUP {
                Type: D3D12_BARRIER_TYPE_GLOBAL,
                NumBarriers: 1,
                Anonymous: D3D12_BARRIER_GROUP_0 {
                    pGlobalBarriers: all.as_ptr(),
                },
            }]);
            return;
        }
        let alias = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_ALIASING,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Aliasing: std::mem::ManuallyDrop::new(D3D12_RESOURCE_ALIASING_BARRIER {
                    pResourceBefore: std::mem::ManuallyDrop::new(None),
                    pResourceAfter: std::mem::ManuallyDrop::new(None),
                }),
            },
        };
        let uav = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                    pResource: std::mem::ManuallyDrop::new(None),
                }),
            },
        };
        list.ResourceBarrier(&[alias, uav]);
    }
}

/// The host copy of `[src, src + len)` if shaders write it: what it holds at
/// a place in a list is not what title memory holds at submission.
fn shaders_own(src: usize, len: usize) -> Option<std::sync::Arc<Shadow>> {
    crate::shadow::peek(src as u64, len as u64).filter(|s| s.written() && !s.in_place())
}

impl HostList {
    /// Copies `len` bytes of title memory at `src` to `dst`, at this place of
    /// the list (see the file). `false` if it is to be done at submission.
    pub(crate) fn place_copy(&self, dst: usize, src: usize, len: usize) -> bool {
        if !on() {
            return false;
        }
        let Some(host) = host() else {
            return false;
        };
        let Some(to) = crate::shadow::peek(dst as u64, len as u64).filter(|s| !s.in_place()) else {
            return false;
        };
        let own = shaders_own(src, len);
        let scratch = match &own {
            Some(_) => None,
            None if len <= MOST => match self.scratch_at(len as u64) {
                Some(scratch) => Some(scratch),
                None => return false,
            },
            None => return false,
        };
        let at = to.offset(dst as u64);
        // What the title's CPU wrote around and under it so far goes first.
        to.refresh(self, dst as u64, len as u64);
        // SAFETY (both): an open list; live buffers (the list holds the
        // copies and its scratch memory), ranges inside them.
        unsafe { wait_for_all(host, &self.list) };
        match (&own, &scratch) {
            (Some(from), _) => {
                from.refresh(self, src as u64, len as u64);
                let offset = from.offset(src as u64);
                if from.buffer.as_raw() == to.buffer.as_raw() {
                    let Some(between) = self.between(host, len as u64) else {
                        return false;
                    };
                    unsafe {
                        self.list.CopyBufferRegion(&between, 0, &from.buffer, offset, len as u64);
                        self.list.ResourceBarrier(&[transition(&between, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
                        wait_for_all(host, &self.list);
                        self.list.CopyBufferRegion(&to.buffer, at, &between, 0, len as u64);
                    }
                } else {
                    unsafe { self.list.CopyBufferRegion(&to.buffer, at, &from.buffer, offset, len as u64) };
                }
                STATS[2].fetch_add(1, Ordering::Relaxed);
            }
            (None, Some((from, offset, _))) => unsafe {
                self.list.CopyBufferRegion(&to.buffer, at, from, *offset, len as u64);
            },
            (None, None) => return false,
        }
        unsafe { wait_for_all(host, &self.list) };
        STATS[0].fetch_add(1, Ordering::Relaxed);
        STATS[1].fetch_add(len as u64, Ordering::Relaxed);
        to.gpu_writes(self, dst as u64, len as u64);
        let mut deferred = self.deferred();
        deferred.title_written(Some((dst, len)));
        self.changed_something();
        if placed_constants_on() && len != 0 && let Some(end) = dst.checked_add(len) {
            let at = deferred.placed_constants.partition_point(|source| source.span.0 < dst);
            deferred.placed_constants.insert(at, ConstantSource {
                shadow: to.clone(),
                span: (dst, end),
            });
        }
        deferred.copies.push(Step::Placed(Placed {
            dst,
            src,
            len,
            shadow: to,
            source: own,
            from: scratch.map(|s| s.2 as usize),
        }));
        true
    }

    /// Invalidates a candidate when another GPU write overlaps it, including
    /// writes through a view which needs no CPU copy-back.
    pub(crate) fn forget_placed_constants(&self, from: usize, len: usize) {
        if placed_constants_on() {
            self.deferred().forget_placed_constants(from, len);
        }
    }

    /// Copies a constant buffer after the recorded writes it depends on.
    /// Each recording retains earlier snapshots for the draws that use them.
    pub(crate) fn snapshot_placed_constants(&self, address: u64, size: u32) -> Option<(ID3D12Resource, u64)> {
        if !placed_constants_on() || size == 0 || size as usize > CONSTANT_BUFFER_BYTES {
            return None;
        }
        let address = usize::try_from(address).ok()?;
        let len = size as usize;
        let end = address.checked_add(len)?;
        let key = (address, len);
        let (source, version, cached, gaps) = {
            let deferred = self.deferred();
            let (source, gaps) = match deferred.constant_coverage(address, len) {
                Some(source) => (source, Vec::new()),
                None => deferred.partial_constant_source(address, end)?,
            };
            let cached = deferred.constant_values.get(&key)
                .filter(|snapshot| snapshot.shadow_id == source.id())
                .map(|snapshot| (deferred.constant_snapshots[snapshot.index].clone(), snapshot.gpu));
            (source, deferred.constant_write_version, cached, gaps)
        };
        let block = address / (64 << 10) * (64 << 10);
        if !crate::inplace::shown_elsewhere(block, end).is_empty() {
            return None;
        }
        if !gaps.is_empty() && !shaders_own(address, len).is_some_and(|owner| Arc::ptr_eq(&owner, &source)) {
            return None;
        }
        if let Some(cached) = cached {
            #[cfg(feature = "diagnostics")]
            {
                CONSTANT_REUSED.fetch_add(1, Ordering::Relaxed);
                if !gaps.is_empty() {
                    CONSTANT_PARTIAL_REUSED.fetch_add(1, Ordering::Relaxed);
                }
            }
            return Some(cached);
        }
        let host = host()?;
        #[cfg(feature = "diagnostics")]
        let allocating = std::time::Instant::now();
        let allocation = host.uav_buffer(u64::from(size).next_multiple_of(256));
        #[cfg(feature = "diagnostics")]
        {
            let elapsed = allocating.elapsed().as_nanos() as u64;
            CONSTANT_ALLOCATION_NS.fetch_add(elapsed, Ordering::Relaxed);
            if allocation.is_some() {
                CONSTANT_ALLOCATED.fetch_add(1, Ordering::Relaxed);
            } else {
                CONSTANT_ALLOCATION_FAILURES.fetch_add(1, Ordering::Relaxed);
            }
        }
        let buffer = allocation?;
        #[cfg(feature = "diagnostics")]
        if !gaps.is_empty() {
            CONSTANT_PARTIAL.fetch_add(1, Ordering::Relaxed);
        }
        for (from, to) in gaps {
            source.refresh(self, from as u64, (to - from) as u64);
        }
        // SAFETY: the list is still recording. The placed copy and list
        // hold the source. Its range was checked above and the snapshot is
        // held until all uses of this recording finish.
        let gpu = unsafe {
            wait_for_all(host, &self.list);
            self.list.CopyBufferRegion(
                &buffer,
                0,
                &source.buffer,
                source.offset(address as u64),
                u64::from(size),
            );
            self.list.ResourceBarrier(&[transition(
                &buffer,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            )]);
            buffer.GetGPUVirtualAddress()
        };
        {
            let mut deferred = self.deferred();
            let index = deferred.constant_snapshots.len();
            deferred.constant_snapshots.push(buffer.clone());
            // Keep the resource for the draw, even if an intervening write
            // made the range unsuitable for reuse.
            if deferred.constant_write_version == version {
                deferred.constant_values.insert(key, ConstantSnapshot { index, gpu, shadow_id: source.id() });
            }
        }
        static FIRST: std::sync::Once = std::sync::Once::new();
        FIRST.call_once(|| tracing::info!("copied constants after a placed copy on the GPU"));
        Some((buffer, gpu))
    }

    pub(super) fn close_constant_snapshots(&self) {
        if !placed_constants_on() {
            return;
        }
        let barriers: Vec<_> = self
            .deferred()
            .constant_snapshots
            .iter()
            .map(|buffer| transition(buffer, D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_STATE_COMMON))
            .collect();
        if !barriers.is_empty() {
            // SAFETY: the list owns the snapshots and is about to close.
            // A recording can occur twice in one ExecuteCommandLists call,
            // so the next use must start in COMMON without relying on decay.
            unsafe { self.list.ResourceBarrier(&barriers) };
        }
    }

    /// For a copy of `[src, src + len)` into a texture: a buffer (readable by
    /// copies) that holds what shaders wrote there, at this place of the list.
    /// `None` where shaders do not write it.
    pub(crate) fn place_source(&self, src: usize, len: usize) -> Option<ID3D12Resource> {
        if !on() {
            return None;
        }
        let host = host()?;
        let from = shaders_own(src, len)?;
        let between = self.between(host, len as u64)?;
        from.refresh(self, src as u64, len as u64);
        STATS[3].fetch_add(1, Ordering::Relaxed);
        // SAFETY: an open list; live buffers, ranges inside them.
        unsafe {
            wait_for_all(host, &self.list);
            self.list.CopyBufferRegion(&between, 0, &from.buffer, from.offset(src as u64), len as u64);
            self.list.ResourceBarrier(&[transition(&between, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            wait_for_all(host, &self.list);
        }
        Some(between)
    }

    /// A buffer on the card for this recording alone.
    fn between(&self, host: &Host, len: u64) -> Option<ID3D12Resource> {
        let buffer = host.uav_buffer(len.next_multiple_of(1 << 16))?;
        name(&buffer, || "bytes between copies".into());
        self.keep(&buffer);
        Some(buffer)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_placed_copy_must_cover_the_whole_constant() {
        let span = (0x1000, 0x2000);
        assert!(super::covered(span, 0x1000, 0x1000));
        assert!(super::covered(span, 0x1100, 0x100));
        assert!(!super::covered(span, 0x0fff, 0x200));
        assert!(!super::covered(span, 0x1f00, 0x101));
        assert!(!super::covered(span, 0x1000, 0));
        assert!(!super::covered((0x1000, usize::MAX), usize::MAX - 5, 6));
    }

    /// What copies with two sides cost to make (ignored: a measurement for
    /// `VICTORY_CARD_ALL`): buffers of 64 KiB on the card and in the PC's
    /// memory.
    #[test]
    #[ignore]
    fn small_buffers_cost() {
        let Some(host) = super::host() else {
            return;
        };
        const N: usize = 1000;
        let used = || host.video_memory().map_or(0, |m| m[1] >> 20);
        let before = used();
        let t = std::time::Instant::now();
        let card: Vec<_> = (0..N).map(|_| host.uav_buffer(1 << 16)).collect();
        let card_us = t.elapsed().as_micros() as f64 / N as f64;
        let after = used();
        let t = std::time::Instant::now();
        let pc: Vec<_> = (0..N).map(|_| host.shared_buffer(1 << 16)).collect();
        let pc_us = t.elapsed().as_micros() as f64 / N as f64;
        println!(
            "{N} buffers of 64 KiB: on the card {card_us:.0} us each, {} MB of its memory; in the PC's memory {pc_us:.0} us each",
            after.saturating_sub(before)
        );
        drop((card, pc));
    }
}
