//! Host-memory admission for speculative texture work and reusable upload storage.
use std::sync::{Mutex, OnceLock};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};

const MIB: u64 = 1 << 20;
const GIB: u64 = 1 << 30;
static PRESSURED: AtomicBool = AtomicBool::new(false);
static HEADROOM: AtomicU64 = AtomicU64::new(u64::MAX);

fn memory_status() -> Option<MEMORYSTATUSEX> {
    // SAFETY: the API initializes the fields of a correctly sized structure.
    let mut status: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    status.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    (unsafe { GlobalMemoryStatusEx(&mut status) } != 0).then_some(status)
}

// The texture sweep may run for seconds inside a single submitted frame.
static NEXT_COMMIT_SAMPLE: AtomicU64 = AtomicU64::new(0);
static AVAILABLE_COMMIT: AtomicU64 = AtomicU64::new(u64::MAX);
static RETRY_UPLOAD_AT: AtomicU64 = AtomicU64::new(0);
pub(crate) const UPLOAD_RETRY_MS: u64 = 500;

pub(crate) fn allocation_error(error: &windows::core::Error) {
    if error.code() == windows::Win32::Foundation::E_OUTOFMEMORY {
        RETRY_UPLOAD_AT.fetch_max(crate::host::coarse_ms().saturating_add(UPLOAD_RETRY_MS), Ordering::Relaxed);
    }
}

thread_local! {
    static UPLOAD_DEFERRALS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

pub(crate) fn note_upload_deferred() {
    UPLOAD_DEFERRALS.set(UPLOAD_DEFERRALS.get().wrapping_add(1));
}

/// Detects upload deferral during this operation. Nested checks can observe
/// the same deferral without consuming the enclosing operation's signal.
pub(crate) fn upload_deferred_during<T>(work: impl FnOnce() -> T) -> (T, bool) {
    let before = UPLOAD_DEFERRALS.get();
    let result = work();
    (result, UPLOAD_DEFERRALS.get() != before)
}

pub(crate) fn upload_refused() -> bool {
    crate::host::coarse_ms() < RETRY_UPLOAD_AT.load(Ordering::Relaxed)
}

fn sample_commit(now: u64) -> u64 {
    let next = NEXT_COMMIT_SAMPLE.load(Ordering::Relaxed);
    if now >= next
        && NEXT_COMMIT_SAMPLE
            .compare_exchange(next, now.saturating_add(32), Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
        && let Some(status) = memory_status()
    {
        AVAILABLE_COMMIT.store(status.ullAvailPageFile, Ordering::Relaxed);
    }
    AVAILABLE_COMMIT.load(Ordering::Relaxed)
}

fn has_upload_reserve(commit: u64, bake: bool) -> bool {
    // Keep a protective floor against Rust/game OOM aborts without starving
    // bindless world textures or one-shot ped decoration bakes when steady-state
    // commit headroom sits in the 1-4 GiB range on 32 GiB hosts.
    commit >= if bake { 256 * MIB } else { 512 * MIB }
}

fn has_growth_reserve(commit: u64, bytes: u64, bake: bool) -> bool {
    // Growing a fresh 64 MiB upload block for speculative bindless prefetch
    // requires more headroom than reusing slices already committed inside the
    // staging pool.
    commit.saturating_sub(bytes) >= if bake { 256 * MIB } else { 1024 * MIB }
}

/// Leave commit for game allocations and submissions already in flight while
/// still letting bindless textures and one-shot material bakes upload.
pub(crate) fn admit_texture_pass(bake: bool) -> bool {
    let now = crate::host::coarse_ms();
    if now < RETRY_UPLOAD_AT.load(Ordering::Relaxed) {
        return false;
    }
    has_upload_reserve(sample_commit(now), bake)
}

/// A look-ahead hint, not an allocation permit. The allocation site reserves
/// its actual size with `reserve_growth`; existing staging slices stay usable.
pub(crate) fn allow_staging_growth(bake: bool) -> bool {
    if upload_refused() { return false; }
    let mut growth = GROWTH.lock().unwrap_or_else(|e| e.into_inner());
    growth.sample(crate::host::coarse_ms());
    growth.fits(64 * MIB, bake)
}

/// Optional CPU-backed allocations share a budget, including allocations
/// currently inside the driver and completed ones since the last OS sample.
/// A fresh sample may already include part of a pending allocation: keeping
/// that reservation too is deliberately conservative until it completes.
#[derive(Default)]
struct Growth {
    available: Option<(u64, u64)>, // commit, physical
    next_sample: u64,
    pending: u64,
    charged: u64,
}

impl Growth {
    fn sample(&mut self, now: u64) {
        if now >= self.next_sample {
            self.next_sample = now.saturating_add(32);
            if let Some(status) = memory_status() {
                self.update(status.ullAvailPageFile, status.ullAvailPhys);
            }
        }
    }

    fn update(&mut self, commit: u64, physical: u64) {
        self.available = Some((commit, physical));
        self.charged = 0;
    }

    fn fits(&self, bytes: u64, bake: bool) -> bool {
        let Some((commit, physical)) = self.available else { return false };
        let used = self.pending.saturating_add(self.charged);
        has_growth_reserve(commit.saturating_sub(used), bytes, bake)
            && (bake || physical.saturating_sub(used).saturating_sub(bytes) >= GIB)
    }

    fn reserve(&mut self, bytes: u64, bake: bool) -> bool {
        if !self.fits(bytes, bake) { return false; }
        self.pending += bytes;
        true
    }

    fn finish(&mut self, bytes: u64, committed: bool) {
        self.pending -= bytes;
        if committed { self.charged = self.charged.saturating_add(bytes); }
    }
}

static GROWTH: Mutex<Growth> = Mutex::new(Growth { available: None, next_sample: 0, pending: 0, charged: 0 });

/// Releases the pending charge on failure/unwind; successful allocations stay
/// charged until a fresh OS sample includes their committed memory.
pub(crate) struct GrowthReservation {
    bytes: u64,
    committed: bool,
}

impl GrowthReservation {
    pub(crate) fn complete(mut self) { self.committed = true; }
}

impl Drop for GrowthReservation {
    fn drop(&mut self) {
        GROWTH.lock().unwrap_or_else(|e| e.into_inner()).finish(self.bytes, self.committed);
    }
}

/// Only optional growth uses this gate. Required table-bound uploads and
/// material bakes retain their existing path, even when physical RAM is low.
pub(crate) fn reserve_growth(bytes: u64, bake: bool) -> Option<GrowthReservation> {
    if upload_refused() { return None; }
    // A committed D3D buffer occupies at least a 64 KiB heap allocation.
    let bytes = bytes.max(1).checked_next_multiple_of(64 << 10)?;
    let mut growth = GROWTH.lock().unwrap_or_else(|e| e.into_inner());
    growth.sample(crate::host::coarse_ms());
    growth.reserve(bytes, bake).then(|| GrowthReservation { bytes, committed: false })
}

/// A PC with 32 GiB of RAM or less, or `VICTORY_RESIDENCY_LIMITED=1` (the
/// name avatarmanz/VIctory-7900XTX gives it): the budgets of a smaller PC.
pub(crate) fn limited_host_ram() -> bool {
    static LIMITED: OnceLock<bool> = OnceLock::new();
    *LIMITED.get_or_init(|| {
        std::env::var_os("VICTORY_RESIDENCY_LIMITED").is_some_and(|v| v == "1")
            || memory_status().is_some_and(|s| s.ullTotalPhys <= 32 * GIB)
    })
}

fn next_pressure(was: bool, physical: u64, commit: u64) -> bool {
    if was {
        physical < 5 * GIB || commit < 6 * GIB
    } else {
        physical <= 3 * GIB || commit <= 4 * GIB
    }
}

/// Called by the once-per-second submission sweep. A failed sample keeps the last good state.
pub(crate) fn refresh() {
    let Some(status) = memory_status() else { return };
    let (physical, commit) = (status.ullAvailPhys, status.ullAvailPageFile);
    HEADROOM.store(physical.min(commit), Ordering::Relaxed);
    let was = pressured();
    let pressure = next_pressure(was, physical, commit);
    if pressure != was {
        PRESSURED.store(pressure, Ordering::Relaxed);
        if pressure {
            tracing::warn!(
                pid = std::process::id(),
                frame = crate::frame::current(),
                physical_free_mib = physical >> 20,
                available_commit_mib = commit >> 20,
                buffer_mirror_span_mib = crate::shadow::live_span_bytes() >> 20,
                texture_staging_mib = crate::texture::live_staging_bytes() >> 20,
                spare_scratch_mib = crate::host::ALIVE[4].load(Ordering::Relaxed) >> 20,
                game_filled_gpu_texture_mib = crate::resource::KINDS[1].load(Ordering::Relaxed) >> 20,
                "host memory pressure: limiting texture prefetch and idle upload storage"
            );
        } else {
            tracing::info!(physical_free_mib = physical >> 20, available_commit_mib = commit >> 20, "host memory pressure cleared");
        }
    }
}

pub(crate) fn headroom() -> Option<u64> {
    let bytes = HEADROOM.load(Ordering::Relaxed);
    (bytes != u64::MAX).then_some(bytes)
}

pub(crate) fn pressured() -> bool {
    PRESSURED.load(Ordering::Relaxed)
}

/// True when OS commit headroom is critically low (< 1.5 GiB) or an allocation
/// just failed with `E_OUTOFMEMORY`, requiring emergency throttling.
pub(crate) fn critical_commit() -> bool {
    let now = crate::host::coarse_ms();
    now < RETRY_UPLOAD_AT.load(Ordering::Relaxed) || sample_commit(now) < 1536 * MIB
}

/// A soft per-frame admission limit. A single texture can cross it. The
/// queue retains unserved slots so they can be retried on later frames.
pub(crate) fn prefetch_limit(bytes: u64, pressure: bool, critical: bool, limited: bool, staging: u64) -> u64 {
    if critical {
        bytes.min(16 << 20)
    } else if pressure || (limited && staging >= GIB) {
        bytes.min(64 << 20)
    } else {
        bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn growth_reservations_cover_pending_and_completed_allocations() {
        let mut budget = Growth::default();
        budget.update(4 * GIB, GIB + 128 * MIB);
        assert!(budget.reserve(64 * MIB, false));
        assert!(budget.reserve(64 * MIB, false));
        assert!(!budget.reserve(64 * MIB, false));
        budget.finish(64 * MIB, true);
        assert!(!budget.reserve(64 * MIB, false), "completion must not refund committed memory before a sample");
        budget.finish(64 * MIB, false);
        assert!(budget.reserve(64 * MIB, false), "failed allocation refunds its reservation");
        budget.finish(64 * MIB, false);
        budget.update(4 * GIB, GIB + 64 * MIB);
        assert!(budget.reserve(64 * MIB, false), "new sample replaces completed charges");
    }

    #[test]
    fn sampling_during_allocation_preserves_pending_reservations() {
        let mut budget = Growth::default();
        assert!(!budget.reserve(1, false), "no valid OS sample");
        budget.update(2 * GIB, 2 * GIB);
        assert!(budget.reserve(512 * MIB, false));
        budget.update(GIB + 512 * MIB, GIB + 512 * MIB);
        assert!(!budget.reserve(1, false));
        budget.finish(512 * MIB, false);
        assert!(budget.reserve(512 * MIB, false));
    }

    #[test]
    fn optional_growth_checks_both_physical_and_commit_reserves() {
        let mut budget = Growth::default();
        budget.update(16 * GIB, GIB);
        assert!(!budget.reserve(64 * MIB, false));
        budget.update(GIB, 16 * GIB);
        assert!(!budget.reserve(64 * MIB, false));
        budget.update(GIB + 64 * MIB, GIB + 64 * MIB);
        assert!(budget.reserve(64 * MIB, false));
    }

    #[test]
    fn concurrent_allocators_cannot_spend_the_same_headroom() {
        let mut budget = Growth::default();
        budget.update(4 * GIB, GIB + 4 * 64 * MIB);
        let budget = Mutex::new(budget);
        let barrier = std::sync::Barrier::new(8);
        let granted = std::sync::atomic::AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(|| {
                    barrier.wait();
                    let reserved = budget.lock().unwrap().reserve(64 * MIB, false);
                    if reserved { granted.fetch_add(1, Ordering::Relaxed); }
                    // Keep all admitted allocations pending until every worker
                    // has attempted admission against the same OS sample.
                    barrier.wait();
                    if reserved { budget.lock().unwrap().finish(64 * MIB, true); }
                });
            }
        });
        assert_eq!(granted.load(Ordering::Relaxed), 4);
        let budget = budget.lock().unwrap();
        assert_eq!(budget.pending, 0);
        assert_eq!(budget.charged, 256 * MIB);
        assert!(!budget.fits(1, false));
    }

    #[test]
    fn an_upload_deferral_reaches_the_enclosing_operation() {
        note_upload_deferred();
        assert!(!upload_deferred_during(|| ()).1);
        let (_, deferred) = upload_deferred_during(|| {
            assert!(upload_deferred_during(note_upload_deferred).1);
            assert!(!upload_deferred_during(|| ()).1);
        });
        assert!(deferred);
    }

    #[test]
    fn bake_and_lookahead_leave_room_for_live_game_allocations() {
        for (commit, bake, lookahead, growth) in [
            (128 * MIB, false, false, false),
            (256 * MIB, true, false, false),
            (512 * MIB, true, true, false),
            (768 * MIB, true, true, false),
            (1024 * MIB, true, true, true),
        ] {
            assert_eq!(has_upload_reserve(commit, true), bake);
            assert_eq!(has_upload_reserve(commit, false), lookahead);
            assert_eq!(has_growth_reserve(commit, 0, false), growth);
        }
    }

    #[test]
    fn a_second_buffer_counts_against_the_growth_reserve() {
        assert!(!has_growth_reserve(GIB, 64 * MIB, false));
        assert!(has_growth_reserve(GIB + 64 * MIB, 64 * MIB, false));
        assert!(!has_growth_reserve(512 * MIB, GIB, false));
    }

    #[test]
    fn either_constraint_enters_pressure_and_both_must_recover() {
        assert!(next_pressure(false, 12 * GIB, 4 * GIB));
        assert!(next_pressure(false, 3 * GIB, 16 * GIB));
        assert!(next_pressure(true, 4 * GIB, 7 * GIB));
        assert!(next_pressure(true, 6 * GIB, 5 * GIB));
        assert!(!next_pressure(true, 5 * GIB, 6 * GIB));
        assert!(!next_pressure(false, 4 * GIB, 5 * GIB));
    }

    #[test]
    fn prefetch_respects_existing_budgets_and_resumes_after_recovery() {
        assert_eq!(prefetch_limit(512 << 20, false, false, true, GIB), 64 << 20);
        assert_eq!(prefetch_limit(512 << 20, false, false, true, GIB - 1), 512 << 20);
        assert_eq!(prefetch_limit(512 << 20, false, false, false, 2 * GIB), 512 << 20);
        assert_eq!(prefetch_limit(0, true, true, false, 0), 0);
        assert_eq!(prefetch_limit(8 << 20, true, true, false, 0), 8 << 20);
        assert_eq!(prefetch_limit(512 << 20, true, false, false, 0), 64 << 20);
        assert_eq!(prefetch_limit(512 << 20, true, true, false, 0), 16 << 20);
        assert_eq!(prefetch_limit(192 << 20, false, false, false, 0), 192 << 20);
    }
}
