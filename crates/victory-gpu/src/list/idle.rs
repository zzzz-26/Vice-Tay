//! Reuse completed objects without retaining a loading spike forever.
//! Callers put only objects that no recording or submission can still use.
//! Evictions return ownership: driver destruction happens outside pool locks.

pub(super) fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // VICTORY_LIST_TRIM=0 restores the old retention policy for comparisons.
    *ON.get_or_init(|| std::env::var_os("VICTORY_LIST_TRIM").is_none_or(|v| v != "0"))
}

struct Entry<T> {
    value: T,
    bytes: u64,
    returned: u64,
}

pub(super) struct Pool<T> {
    entries: Vec<Entry<T>>,
    bytes: u64,
    hits: u64,
    misses: u64,
    released: u64,
}

impl<T> Pool<T> {
    pub(super) const fn new() -> Self {
        Self { entries: Vec::new(), bytes: 0, hits: 0, misses: 0, released: 0 }
    }

    pub(super) fn len(&self) -> usize { self.entries.len() }

    pub(super) fn take(&mut self) -> Option<T> {
        if let Some(entry) = self.entries.pop() {
            self.bytes -= entry.bytes;
            self.hits += 1;
            Some(entry.value)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Rejected objects are returned, never destroyed under the caller's lock.
    pub(super) fn put(&mut self, value: T, bytes: u64, now: u64, count: usize, budget: u64) -> Option<T> {
        if self.entries.len() >= count || bytes > budget.saturating_sub(self.bytes) {
            self.released += 1;
            return Some(value);
        }
        self.bytes += bytes;
        self.entries.push(Entry { value, bytes, returned: now });
        None
    }

    /// Keep recent bursts and a warm floor. Limit driver frees per sweep.
    pub(super) fn trim(&mut self, now: u64, pressure: bool) -> Vec<T> {
        let age = if pressure { 2_000 } else { 30_000 };
        let mut gone = Vec::new();
        let mut i = 0;
        while i < self.entries.len() && self.entries.len() > 32 && gone.len() < 32 {
            if now.saturating_sub(self.entries[i].returned) >= age {
                // Preserve LIFO order: a cold entry must not displace the
                // most recently used allocator from the top of the stack.
                let entry = self.entries.remove(i);
                self.bytes -= entry.bytes;
                self.released += 1;
                gone.push(entry.value);
            } else {
                i += 1;
            }
        }
        gone
    }

    #[cfg(feature = "diagnostics")]
    pub(super) fn stats(&self) -> [u64; 5] {
        [self.entries.len() as u64, self.bytes, self.hits, self.misses, self.released]
    }
}

#[cfg(test)]
mod tests {
    use super::Pool;
    use std::sync::Arc;

    #[test]
    fn trimming_never_touches_checked_out_or_recent_objects() {
        let mut pool = Pool::new();
        for _ in 0..100 {
            assert!(pool.put(Arc::new(()), 4096, 0, 512, u64::MAX).is_none());
        }
        let checked_out = pool.take().unwrap();
        let live = Arc::downgrade(&checked_out);
        let recent = Arc::new(());
        let recent_weak = Arc::downgrade(&recent);
        assert!(pool.put(recent, 8192, 2_000, 512, u64::MAX).is_none());
        let gone = pool.trim(2_000, true);
        assert_eq!(gone.len(), 32);
        assert_eq!(pool.len(), 68);
        drop(gone);
        assert!(live.upgrade().is_some());
        assert!(recent_weak.upgrade().is_some());
        assert!(Arc::ptr_eq(&pool.take().unwrap(), &recent_weak.upgrade().unwrap()));
        assert_eq!(pool.bytes, 67 * 4096);
    }

    #[test]
    fn limits_return_ownership_and_allow_reuse_after_pressure_clears() {
        let mut pool = Pool::new();
        assert!(pool.put(1, 8, 0, 2, 10).is_none());
        assert_eq!(pool.put(2, 3, 0, 2, 10), Some(2));
        assert_eq!(pool.put(3, 0, 0, 1, 10), Some(3));
        assert_eq!(pool.take(), Some(1));
        assert_eq!(pool.bytes, 0);
        assert!(pool.put(4, 10, 0, 2, 10).is_none());
        assert_eq!(pool.take(), Some(4));
        assert_eq!(pool.take(), None);
    }

    #[test]
    fn cold_burst_drains_in_bounded_batches_and_keeps_a_warm_floor() {
        let mut pool = Pool::new();
        for i in 0..100 { assert!(pool.put(i, 1, 0, 512, 512).is_none()); }
        assert!(pool.trim(1_999, true).is_empty());
        assert!(pool.trim(2_000, false).is_empty());
        assert_eq!(pool.trim(30_000, false).len(), 32);
        assert_eq!(pool.trim(30_000, false).len(), 32);
        assert_eq!(pool.trim(30_000, false).len(), 4);
        assert!(pool.trim(30_000, false).is_empty());
        assert_eq!(pool.bytes, 32);
    }

    /// Synthetic touched allocations, no title data or GPU: verifies actual
    /// ownership release after a burst, without touching checked-out objects.
    #[test]
    fn a_loading_spike_releases_touched_buffers() {
        let mut pool = Pool::new();
        let weak: Vec<_> = (0..48).map(|_| {
            let bytes = Arc::new(vec![0x5au8; 1 << 20]);
            let weak = Arc::downgrade(&bytes);
            assert!(pool.put(bytes, 1 << 20, 0, 512, 256 << 20).is_none());
            weak
        }).collect();
        let live = pool.take().unwrap();
        drop(pool.trim(2_000, true));
        assert_eq!(weak.iter().filter(|w| w.strong_count() != 0).count(), 33);
        assert_eq!(pool.bytes, 32 << 20);
        assert_eq!(live[0], 0x5a);
        drop(pool);
        assert_eq!(weak.iter().filter(|w| w.strong_count() != 0).count(), 1);
    }
}
