//! Storage returned only after its owner has finished using it. Destruction is
//! left to the caller, outside its pool lock (driver releases can be expensive).
use std::sync::OnceLock;

pub(crate) fn trimming() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_IDLE_TRIM").is_none_or(|v| v != "0"))
}

struct Entry<T> {
    value: T,
    bytes: u64,
    returned: u64,
}

pub(crate) struct Pool<T> {
    entries: Vec<Entry<T>>,
    bytes: u64,
    hits: u64,
    misses: u64,
    released: u64,
}

impl<T> Pool<T> {
    pub(crate) const fn new() -> Self {
        Self { entries: Vec::new(), bytes: 0, hits: 0, misses: 0, released: 0 }
    }

    pub(crate) fn take(&mut self) -> Option<T> {
        self.take_at(self.entries.len().checked_sub(1))
    }

    pub(crate) fn full(&self, limit: usize) -> bool {
        self.entries.len() >= limit
    }

    pub(crate) fn best_fit(&mut self, bytes: u64) -> Option<T> {
        let fit = self.entries.iter().enumerate()
            .filter(|(_, entry)| entry.bytes >= bytes)
            // Reuse the newest of equal-sized buffers. Cycling through every
            // equal fit would refresh cold entries forever after a load spike.
            .min_by_key(|(i, entry)| (entry.bytes, std::cmp::Reverse(*i))).map(|(i, _)| i);
        self.take_at(fit)
    }

    fn take_at(&mut self, index: Option<usize>) -> Option<T> {
        match index {
            Some(i) => {
                self.hits += 1;
                Some(self.remove(i))
            }
            None => {
                self.misses += 1;
                None
            }
        }
    }

    fn remove(&mut self, index: usize) -> T {
        // Preserve return order: a sweep must not move an old entry to the
        // hot end of the pool and evict a more recently used one next time.
        let entry = self.entries.remove(index);
        self.bytes -= entry.bytes;
        entry.value
    }

    pub(crate) fn put(&mut self, value: T, bytes: u64, now: u64, count_limit: usize, byte_limit: u64) -> Option<T> {
        if self.entries.len() >= count_limit || bytes > byte_limit.saturating_sub(self.bytes) {
            self.released += 1;
            return Some(value);
        }
        self.entries.push(Entry { value, bytes, returned: now });
        self.bytes += bytes;
        None
    }

    /// Keep the existing upload-cache policy of shedding its largest buffer
    /// first. Unlike dropping inside `retain`, this returns ownership.
    pub(crate) fn cap_bytes(&mut self, limit: u64) -> Vec<T> {
        let mut gone = Vec::new();
        while self.bytes > limit {
            let Some(i) = self.entries.iter().enumerate().max_by_key(|(_, e)| e.bytes).map(|(i, _)| i) else { break };
            gone.push(self.remove(i));
        }
        self.released += gone.len() as u64;
        gone
    }

    /// Recently returned storage stays warm even under pressure. Bound the
    /// destruction work in each sweep; checked-out objects are never here.
    pub(crate) fn trim(&mut self, now: u64, pressure: bool, warm: usize) -> Vec<T> {
        let age = if pressure { 2_000 } else { 30_000 };
        let mut gone = Vec::new();
        let mut i = 0;
        while self.entries.len() > warm && i < self.entries.len() && gone.len() < 32 {
            if now.saturating_sub(self.entries[i].returned) >= age {
                gone.push(self.remove(i));
            } else {
                i += 1;
            }
        }
        self.released += gone.len() as u64;
        gone
    }

    #[cfg(any(test, feature = "diagnostics"))]
    pub(crate) fn stats(&self) -> [u64; 5] {
        [self.entries.len() as u64, self.bytes, self.hits, self.misses, self.released]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn sweep_releases_only_old_idle_owners_and_preserves_reuse_order() {
        let mut pool = Pool::new();
        let owned: Vec<_> = (0..8).map(Arc::new).collect();
        for (i, value) in owned.iter().enumerate() {
            assert!(pool.put(value.clone(), 10, if i < 6 { 0 } else { 1_500 }, 8, 80).is_none());
        }
        let checked_out = pool.take().unwrap();
        let gone = pool.trim(2_000, true, 2);
        assert_eq!(gone.len(), 5);
        assert_eq!(Arc::strong_count(&owned[0]), 2);
        drop(gone);
        assert_eq!(Arc::strong_count(&owned[0]), 1);
        assert_eq!(*checked_out, 7);
        assert_eq!(*pool.take().unwrap(), 6);
        assert_eq!(*pool.take().unwrap(), 5);
        assert_eq!(pool.stats()[1], 0);
    }

    #[test]
    fn bounds_and_best_fit_keep_byte_accounting_exact() {
        let mut pool = Pool::new();
        for bytes in [8, 16, 32] {
            assert!(pool.put(bytes, bytes, 0, 3, 56).is_none());
        }
        assert_eq!(pool.put(64, 64, 0, 3, 56), Some(64));
        assert_eq!(pool.best_fit(12), Some(16));
        assert_eq!(pool.best_fit(64), None);
        assert_eq!(pool.cap_bytes(8), vec![32]);
        assert_eq!(pool.stats(), [1, 8, 1, 1, 2]);
    }

    #[test]
    fn normal_expiry_is_slower_and_each_sweep_is_bounded() {
        let mut pool = Pool::new();
        for i in 0..100 { assert!(pool.put(i, 1, 0, 100, 100).is_none()); }
        assert!(pool.trim(2_000, false, 32).is_empty());
        assert_eq!(pool.trim(30_000, false, 32).len(), 32);
        assert_eq!(pool.trim(30_000, false, 32).len(), 32);
        assert_eq!(pool.trim(30_000, false, 32).len(), 4);
        assert!(pool.trim(30_000, false, 32).is_empty());
    }

    #[test]
    fn equal_sized_uploads_keep_one_hot_buffer_without_refreshing_cold_ones() {
        let mut pool = Pool::new();
        for i in 0..4 { assert!(pool.put(i, 8, 0, 4, 32).is_none()); }
        let hot = pool.best_fit(8).unwrap();
        assert_eq!(hot, 3);
        assert!(pool.put(hot, 8, 1_500, 4, 32).is_none());
        assert_eq!(pool.trim(2_000, true, 1), vec![0, 1, 2]);
        assert_eq!(pool.best_fit(8), Some(3));
    }
}
