//! Fresh 64 KiB title shadow buffers prepared before the first world.
//! Taking an item transfers ownership permanently: this is not a recycle pool.

use super::{Host, ID3D12Resource};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};
use windows::Win32::Graphics::Direct3D12::ID3D12Pageable;
use windows::core::Interface;

const BYTES: u64 = 64 << 10;
const BATCH: usize = 32;

fn count() -> usize {
    static COUNT: OnceLock<usize> = OnceLock::new();
    *COUNT.get_or_init(|| {
        std::env::var("VICTORY_TITLE_PREWARM_COUNT")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .unwrap_or(0)
            .min(3584)
    })
}

struct Fresh(ID3D12Resource, *mut u8);
// SAFETY: the resource owns the persistent map. No thread accesses its
// bytes until the only owner removes it from stock.
unsafe impl Send for Fresh {}

#[derive(Default)]
pub(super) struct Stock {
    started: AtomicBool,
    stopped: AtomicBool,
    ready: AtomicUsize,
    buffers: Mutex<Vec<Fresh>>,
}

fn room(memory: Option<[u64; 4]>) -> bool {
    memory.is_some_and(|[_, _, budget, used]| {
        budget.saturating_sub(used) >= 128 << 20 && used < budget / 10 * 9
    })
}

impl Stock {
    fn take(&self) -> Option<Fresh> {
        if self.ready.load(Relaxed) == 0 {
            return None;
        }
        let mut slots = self.buffers.lock().unwrap_or_else(|e| e.into_inner());
        let fresh = slots.pop()?;
        self.ready.store(slots.len(), Relaxed);
        Some(fresh)
    }

    fn stop(&self) {
        let unused = {
            let mut slots = self.buffers.lock().unwrap_or_else(|e| e.into_inner());
            self.stopped.store(true, Relaxed);
            self.ready.store(0, Relaxed);
            std::mem::take(&mut *slots)
        };
        drop(unused);
    }
}

impl Host {
    fn fresh_title_batch(&self, n: usize) -> windows::core::Result<Vec<Fresh>> {
        let buffers = (0..n)
            .map(|_| self.create_shared_buffer(BYTES))
            .collect::<windows::core::Result<Vec<_>>>()?;
        let objects = buffers
            .iter()
            .map(|b| b.cast::<ID3D12Pageable>().map(Some))
            .collect::<windows::core::Result<Vec<_>>>()?;
        // SAFETY: all pageables are fresh committed resources on this device.
        unsafe {
            self.device.MakeResident(&objects)?;
        }
        // One mapping fence covers the batch. No descriptor can reference
        // any of these buffers until all allocations and maps succeed.
        super::mapped(&self.device)?;
        buffers
            .into_iter()
            .map(|b| self.map_shared_buffer(b).map(|(b, p)| Fresh(b, p)))
            .collect()
    }

    pub(super) fn start_title_stock(&self) {
        let count = count();
        if count == 0 || !super::wait_resident() || self.title_stock.started.load(Relaxed) {
            return;
        }
        let Some(owner) = super::HOST
            .get()
            .and_then(Option::as_ref)
            .filter(|h| std::ptr::eq(*h, self))
        else {
            return;
        };
        if self.title_stock.started.swap(true, Relaxed) {
            return;
        }
        let start = Instant::now();
        if let Err(e) = std::thread::Builder::new()
            .name("title-prewarm".into())
            .spawn(move || {
                let stock = &owner.title_stock;
                let mut made = 0;
                let reason = loop {
                    if made == count {
                        break "cap";
                    }
                    if start.elapsed() >= Duration::from_secs(8) {
                        break "deadline";
                    }
                    if stock.stopped.load(Relaxed) {
                        break "pressure";
                    }
                    if !room(owner.video_memory())
                        || crate::residency::pressured()
                        || crate::residency::critical_commit()
                        || super::available_physical_ram().is_none_or(|n| n < 3 << 30)
                    {
                        stock.stop();
                        break "pressure";
                    }
                    let batch = match owner.fresh_title_batch(BATCH.min(count - made)) {
                        Ok(batch) => batch,
                        Err(error) => {
                            tracing::warn!(%error, "title prewarm batch failed");
                            stock.stop();
                            break "allocation failed";
                        }
                    };
                    let mut slots = stock.buffers.lock().unwrap_or_else(|e| e.into_inner());
                    if stock.stopped.load(Relaxed) {
                        break "pressure";
                    }
                    made += batch.len();
                    slots.extend(batch);
                    stock.ready.store(slots.len(), Relaxed);
                };
                tracing::info!(
                    count,
                    made,
                    reason,
                    elapsed_ms = start.elapsed().as_millis() as u64,
                    "title prewarm finished"
                );
            })
        {
            tracing::warn!(%e, "could not start title prewarm");
        }
    }

    pub(crate) fn title_shared_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        if size == BYTES
            && let Some(Fresh(b, p)) = self.title_stock.take()
        {
            return Some((b, p));
        }
        self.shared_buffer(size)
    }

    pub(super) fn trim_title_stock(&self, memory: Option<[u64; 4]>) {
        if self.title_stock.started.load(Relaxed)
            && !self.title_stock.stopped.load(Relaxed)
            && (!room(memory)
                || crate::residency::pressured()
                || crate::residency::critical_commit()
                || super::available_physical_ram().is_none_or(|n| n < 3 << 30))
        {
            self.title_stock.stop();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prewarm_requires_known_shared_budget_and_slack() {
        assert!(!room(None));
        assert!(!room(Some([0, 0, 0, 0])));
        assert!(!room(Some([0, 0, 127 << 20, 0])));
        assert!(!room(Some([0, 0, 8 << 30, 15 << 29])));
        assert!(room(Some([0, 0, 8 << 30, 4 << 30])));
    }

    #[test]
    fn stopping_stock_keeps_issued_buffers_independent_and_mapped() {
        let Some(host) = super::super::host() else {
            return;
        };
        if !super::super::wait_resident() {
            return;
        }
        let stock = Stock::default();
        *stock.buffers.lock().unwrap() = host.fresh_title_batch(3).unwrap();
        stock.ready.store(3, Relaxed);
        let a = stock.take().unwrap();
        let b = stock.take().unwrap();
        stock.stop();
        assert!(stock.take().is_none());
        // SAFETY: the two fresh maps are live and have not been submitted.
        unsafe {
            assert_ne!(a.0.GetGPUVirtualAddress(), b.0.GetGPUVirtualAddress());
            assert_eq!(a.1.read_volatile(), 0);
            a.1.write_volatile(0xa5);
            assert_eq!(b.1.read_volatile(), 0);
            drop(b);
            assert_eq!(a.1.read_volatile(), 0xa5);
        }
    }
}
