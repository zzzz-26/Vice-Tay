use super::{Host, Prelude, host, lazy_prelude};
use std::sync::{Mutex, OnceLock};

const IDLE_LIMIT: usize = 128;

struct IdlePrelude(Prelude);

// SAFETY: D3D12 objects are free-threaded. No list owns an idle prelude.
unsafe impl Send for IdlePrelude {}

pub(super) struct Pool(Mutex<crate::idle_pool::Pool<IdlePrelude>>);

pub(super) static IDLE: Pool = Pool(Mutex::new(crate::idle_pool::Pool::new()));

fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("VICTORY_PRELUDE_POOL").is_some_and(|v| v == "1"))
}

impl Pool {
    fn take(&self) -> Option<Prelude> {
        let idle = self.0.lock().unwrap_or_else(|e| e.into_inner()).take();
        if let Some(IdlePrelude(prelude)) = idle {
            static SAID: std::sync::Once = std::sync::Once::new();
            SAID.call_once(|| tracing::info!("reused a state prelude (VICTORY_PRELUDE_POOL)"));
            Some(prelude)
        } else {
            host().and_then(Host::make_prelude)
        }
    }

    fn give(&self, prelude: Prelude) {
        let rejected = self.0.lock().unwrap_or_else(|e| e.into_inner())
            .put(IdlePrelude(prelude), 0, super::coarse_ms(), IDLE_LIMIT, u64::MAX);
        drop(rejected);
    }
}

pub(super) fn trim_idle() {
    if !crate::idle_pool::trimming() { return; }
    let gone = IDLE.0.try_lock().ok().map(|mut idle| idle.trim(super::coarse_ms(), crate::residency::pressured(), 16));
    drop(gone);
}

#[cfg(feature = "diagnostics")]
pub(super) fn idle_stats() -> [u64; 5] {
    IDLE.0.lock().unwrap_or_else(|e| e.into_inner()).stats()
}

pub(super) enum Slot {
    Owned(OnceLock<Option<Prelude>>),
    Pooled(Mutex<Option<Prelude>>),
}

impl Slot {
    pub(super) fn new(owner: &Host) -> Self {
        let lazy = lazy_prelude();
        if lazy && enabled() {
            return Self::Pooled(Mutex::new(None));
        }
        let owned = OnceLock::new();
        if !lazy {
            let _ = owned.set(owner.make_prelude());
        }
        Self::Owned(owned)
    }

    pub(super) fn record<T>(&self, pool: &Pool, record: impl FnOnce(&Prelude) -> T) -> Option<T> {
        match self {
            Self::Owned(owned) => owned
                .get_or_init(|| host().and_then(Host::make_prelude))
                .as_ref()
                .map(record),
            Self::Pooled(held) => {
                let mut held = held.lock().unwrap_or_else(|e| e.into_inner());
                if held.is_none() {
                    *held = pool.take();
                }
                held.as_ref().map(record)
            }
        }
    }

    /// Called after the list's previous submission has finished and it has reset.
    pub(super) fn recycle(&self, pool: &Pool) {
        if let Self::Pooled(held) = self {
            let prelude = held.lock().unwrap_or_else(|e| e.into_inner()).take();
            if let Some(prelude) = prelude {
                pool.give(prelude);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::core::Interface;

    #[test]
    fn a_held_prelude_becomes_available_only_after_recycling() {
        if host().is_none() {
            return;
        }
        let pool = Pool(Mutex::new(crate::idle_pool::Pool::new()));
        let first = Slot::Pooled(Mutex::new(None));
        let second = Slot::Pooled(Mutex::new(None));
        let allocator = |(allocator, _): &Prelude| allocator.as_raw() as usize;
        let held = first.record(&pool, allocator).unwrap();
        assert_eq!(first.record(&pool, allocator), Some(held));
        let other = second.record(&pool, allocator).unwrap();
        assert_ne!(other, held);
        first.recycle(&pool);
        let third = Slot::Pooled(Mutex::new(None));
        assert_eq!(third.record(&pool, allocator), Some(held));
    }
}
