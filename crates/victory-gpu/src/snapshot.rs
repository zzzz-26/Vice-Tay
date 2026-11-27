//! Stable-offset CPU preimages. Large snapshots can reserve address space
//! without committing the unwritten gaps. Never decommit a live snapshot:
//! deferred copies and landing workers borrow its addresses.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};
use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
};
use windows_sys::Win32::System::SystemInformation::{GetSystemInfo, SYSTEM_INFO};

static ACTIVE: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
static CACHE: Mutex<Vec<(u64, Pages)>> = Mutex::new(Vec::new());

fn page_size() -> usize {
    static PAGE: OnceLock<usize> = OnceLock::new();
    *PAGE.get_or_init(|| {
        // SAFETY: fills a correctly sized output structure.
        let mut info: SYSTEM_INFO = unsafe { std::mem::zeroed() };
        unsafe { GetSystemInfo(&mut info) };
        info.dwPageSize as usize
    })
}

/// 0: original pooled vectors, 1: sparse large snapshots, otherwise adaptive
/// (the default): sparse only for new large snapshots under RAM pressure.
fn enabled(n: usize) -> bool {
    static MODE: OnceLock<String> = OnceLock::new();
    let mode = MODE.get_or_init(|| std::env::var("VICTORY_SPARSE_SNAPSHOTS").unwrap_or_default());
    n >= 8 << 20
        && match mode.as_str() {
            "0" => false,
            "1" => true,
            _ => crate::residency::pressured(),
        }
}

struct Pages {
    base: usize,
    len: usize,
    committed: Vec<bool>,
    bytes: usize,
}

impl Pages {
    fn new(n: usize) -> Option<Self> {
        let len = n.checked_next_multiple_of(page_size())?;
        // SAFETY: reserves private address space; no page is accessed yet.
        let base =
            unsafe { VirtualAlloc(std::ptr::null(), len, MEM_RESERVE, PAGE_READWRITE) } as usize;
        (base != 0).then(|| Self {
            base,
            len,
            committed: vec![false; len / page_size()],
            bytes: 0,
        })
    }

    fn prepare(&mut self, from: usize, to: usize) {
        if from == to {
            return;
        }
        assert!(from < to && to <= self.len);
        let page = page_size();
        let end = to.div_ceil(page);
        let mut at = from / page;
        while at < end {
            if self.committed[at] {
                at += 1;
                continue;
            }
            let first = at;
            while at < end && !self.committed[at] {
                at += 1;
            }
            let bytes = (at - first) * page;
            // SAFETY: page-aligned subrange of this exclusively owned reservation.
            let made = unsafe {
                VirtualAlloc(
                    (self.base + first * page) as *const _,
                    bytes,
                    MEM_COMMIT,
                    PAGE_READWRITE,
                )
            };
            if made.is_null() {
                // As with Vec allocation failure: never publish unbacked pointers
                // or silently omit a required preimage.
                std::alloc::handle_alloc_error(
                    std::alloc::Layout::from_size_align(bytes, page).unwrap(),
                );
            }
            self.committed[first..at].fill(true);
            self.bytes += bytes;
        }
    }
}

impl Drop for Pages {
    fn drop(&mut self) {
        // SAFETY: original reservation base, with no remaining snapshot users.
        unsafe { VirtualFree(self.base as *mut _, 0, MEM_RELEASE) };
    }
}

enum Storage {
    Dense(Vec<u8>),
    Sparse(Pages),
}

pub(crate) struct Snapshot {
    storage: Option<Storage>,
    len: usize,
    capacity: usize,
    bytes: usize,
    /// Sorted disjoint declared ranges; excludes page rounding and gaps.
    /// Deferred copies may not have filled the bytes yet.
    ranges: Vec<(usize, usize)>,
    covered: usize,
}

impl Snapshot {
    pub(crate) fn new(n: usize, dense: impl FnOnce() -> Vec<u8>) -> Self {
        let sparse = if enabled(n) {
            let reused = {
                let mut cache = CACHE.lock().unwrap_or_else(|e| e.into_inner());
                let index = cache
                    .iter()
                    .enumerate()
                    .filter(|(_, (_, p))| p.len >= n && p.len <= n.saturating_mul(2))
                    .min_by_key(|(_, (_, p))| p.len)
                    .map(|(i, _)| i);
                index.map(|i| cache.swap_remove(i).1)
            };
            reused.or_else(|| Pages::new(n))
        } else {
            None
        };
        let storage = sparse
            .map(Storage::Sparse)
            .unwrap_or_else(|| Storage::Dense(dense()));
        let (capacity, bytes) = match &storage {
            Storage::Dense(v) => (v.capacity(), v.capacity()),
            Storage::Sparse(p) => (p.len, p.bytes),
        };
        assert!(capacity >= n);
        ACTIVE[0].fetch_add(1, Relaxed);
        ACTIVE[1].fetch_add(capacity as u64, Relaxed);
        ACTIVE[2].fetch_add(bytes as u64, Relaxed);
        Self {
            storage: Some(storage),
            len: n,
            capacity,
            bytes,
            ranges: Vec::new(),
            covered: 0,
        }
    }

    /// Back memory before handing any address to a deferred writer. This does
    /// not copy bytes or overwrite an earlier preimage in overlapping ranges.
    pub(crate) fn prepare(&mut self, from: usize, to: usize) {
        assert!(from <= to && to <= self.len);
        if from == to {
            return;
        }
        let containing = self.ranges.partition_point(|&(start, _)| start <= from);
        if containing > 0 && self.ranges[containing - 1].1 >= to {
            return;
        }
        if let Some(Storage::Sparse(p)) = &mut self.storage {
            p.prepare(from, to);
            ACTIVE[2].fetch_add((p.bytes - self.bytes) as u64, Relaxed);
            self.bytes = p.bytes;
        }
        let (mut lo, mut hi) = (from, to);
        let first = self.ranges.partition_point(|&(_, end)| end < lo);
        let mut last = first;
        let mut replaced = 0;
        while last < self.ranges.len() && self.ranges[last].0 <= hi {
            let (a, b) = self.ranges[last];
            lo = lo.min(a);
            hi = hi.max(b);
            replaced += b - a;
            last += 1;
        }
        self.ranges.splice(first..last, [(lo, hi)]);
        let added = hi - lo - replaced;
        self.covered += added;
        ACTIVE[3].fetch_add(added as u64, Relaxed);
    }

    pub(crate) fn as_ptr(&self) -> *const u8 {
        match self.storage.as_ref().unwrap() {
            Storage::Dense(v) => v.as_ptr(),
            Storage::Sparse(p) => p.base as *const u8,
        }
    }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut u8 {
        self.as_ptr().cast_mut()
    }

    /// Called only at the existing completion point. Dense storage returns to
    /// the original pool; sparse pages have a separate committed-byte cap.
    pub(crate) fn recycle(mut self) -> Option<Vec<u8>> {
        match self.storage.take().unwrap() {
            Storage::Dense(v) => Some(v),
            Storage::Sparse(p) => {
                let mut cache = CACHE.lock().unwrap_or_else(|e| e.into_inner());
                let bytes: usize = cache.iter().map(|(_, p)| p.bytes).sum();
                if !crate::residency::pressured()
                    && cache.len() < 8
                    && bytes.saturating_add(p.bytes) <= 128 << 20
                {
                    cache.push((crate::host::now_ms(), p));
                } else {
                    drop(cache);
                    drop(p);
                }
                None
            }
        }
    }
}

impl Drop for Snapshot {
    fn drop(&mut self) {
        for (i, value) in [1, self.capacity, self.bytes, self.covered]
            .into_iter()
            .enumerate()
        {
            ACTIVE[i].fetch_sub(value as u64, Relaxed);
        }
    }
}

pub(crate) fn trim() {
    let now = crate::host::now_ms();
    let pressure = crate::residency::pressured();
    let gone = {
        let mut cache = CACHE.lock().unwrap_or_else(|e| e.into_inner());
        let mut gone = Vec::new();
        let mut i = 0;
        while i < cache.len() {
            if pressure || now.saturating_sub(cache[i].0) >= 30_000 {
                gone.push(cache.swap_remove(i));
            } else {
                i += 1;
            }
        }
        gone
    };
    drop(gone);
}

#[cfg(feature = "diagnostics")]
pub(crate) fn stats() -> [u64; 6] {
    let cache = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    [
        ACTIVE[0].load(Relaxed),
        ACTIVE[1].load(Relaxed),
        ACTIVE[2].load(Relaxed),
        ACTIVE[3].load(Relaxed),
        cache.len() as u64,
        cache.iter().map(|(_, p)| p.bytes as u64).sum(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_pages_commit_only_written_ranges_and_preserve_overlaps() {
        let page = page_size();
        let mut p = Pages::new(page * 1024).unwrap();
        assert_eq!(p.bytes, 0);
        p.prepare(page - 1, page + 1);
        assert_eq!(p.bytes, 2 * page);
        // SAFETY: both bytes are in committed pages owned by p.
        unsafe {
            (p.base as *mut u8).add(page).write(73);
        }
        let address = p.base;
        p.prepare(page, page * 3);
        p.prepare(page * 100, page * 100 + 1);
        assert_eq!(p.bytes, 4 * page);
        assert_eq!(p.base, address);
        assert_eq!(unsafe { (p.base as *const u8).add(page).read() }, 73);
        assert!(!p.committed[50]);
        use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
        let state = |offset: usize| {
            // SAFETY: VirtualQuery writes this output and never dereferences
            // the queried reserved address as application data.
            let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            assert_ne!(
                unsafe {
                    VirtualQuery((p.base + offset) as *const _, &mut info, size_of_val(&info))
                },
                0
            );
            info.State
        };
        assert_eq!(state(page), MEM_COMMIT);
        assert_eq!(state(50 * page), MEM_RESERVE);
    }

    #[test]
    fn coverage_counts_union_without_touching_or_replacing_preimages() {
        let mut s = Snapshot::new(1024, || Vec::with_capacity(1024));
        for (a, b) in [
            (100, 200),
            (150, 250),
            (0, 20),
            (20, 100),
            (500, 501),
            (0, 250),
        ] {
            s.prepare(a, b);
        }
        assert_eq!(s.covered, 251);
        assert_eq!(s.ranges, [(0, 250), (500, 501)]);
        assert_eq!(s.recycle().unwrap().capacity(), 1024);
    }
}
