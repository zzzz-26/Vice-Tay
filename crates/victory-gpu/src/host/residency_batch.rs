//! Coalesce concurrent readiness requests, without timers or publishing an
//! allocation before its request succeeds. The existing mapping confirmation
//! still runs after every batch of objects that needs it.
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{HRESULT, Interface};

type Done = Arc<Mutex<Option<Result<(), HRESULT>>>>;
struct Request<T> {
    key: usize,
    item: T,
    done: Done,
}
struct State<T> {
    running: bool,
    queued: VecDeque<Request<T>>,
}
struct Batch<T> {
    state: Mutex<State<T>>,
    changed: Condvar,
}

static COUNTS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("VICTORY_BATCH_RESIDENCY").is_some_and(|v| v == "1"))
}

impl<T> Batch<T> {
    const fn new() -> Self {
        Self {
            state: Mutex::new(State {
                running: false,
                queued: VecDeque::new(),
            }),
            changed: Condvar::new(),
        }
    }

    fn run(
        &self,
        key: usize,
        item: T,
        mut execute: impl FnMut(&[T]) -> Result<(), HRESULT>,
    ) -> Result<(), HRESULT> {
        let done: Done = Arc::new(Mutex::new(None));
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.queued.push_back(Request {
            key,
            item,
            done: done.clone(),
        });
        COUNTS[0].fetch_add(1, Relaxed);
        self.changed.notify_all();
        loop {
            let result = *done.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(result) = result {
                return result;
            }
            if state.running {
                state = self.changed.wait(state).unwrap_or_else(|e| e.into_inner());
                continue;
            }
            state.running = true;
            let mut items = Vec::new();
            let mut completions = Vec::new();
            let mut i = 0;
            while i < state.queued.len() && items.len() < 32 {
                if state.queued[i].key == key {
                    let request = state.queued.remove(i).unwrap();
                    items.push(request.item);
                    completions.push(request.done);
                } else {
                    i += 1;
                }
            }
            drop(state);
            COUNTS[1].fetch_add(1, Relaxed);
            COUNTS[2].fetch_max(items.len() as u64, Relaxed);
            // Unwinding must not leave other callers waiting forever.
            let outcome =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| execute(&items)));
            let result = outcome
                .as_ref()
                .copied()
                .unwrap_or(Err(HRESULT(0x80004005u32 as i32)));
            if result.is_err() {
                COUNTS[3].fetch_add(completions.len() as u64, Relaxed);
            }
            state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            for completion in completions {
                *completion.lock().unwrap_or_else(|e| e.into_inner()) = Some(result);
            }
            state.running = false;
            self.changed.notify_all();
            if let Err(panic) = outcome {
                drop(state);
                std::panic::resume_unwind(panic);
            }
        }
    }
}

pub(super) fn resident(device: &ID3D12Device, object: ID3D12Pageable) -> windows::core::Result<()> {
    if !enabled() {
        // SAFETY: a live object belonging to this device.
        return unsafe { device.MakeResident(&[Some(object)]) };
    }
    static BATCH: Batch<ID3D12Pageable> = Batch::new();
    BATCH
        .run(device.as_raw() as usize, object, |objects| {
            let objects: Vec<_> = objects.iter().cloned().map(Some).collect();
            // SAFETY: the batch keeps the objects alive, all from this device.
            unsafe { device.MakeResident(&objects) }.map_err(|e| e.code())
        })
        .map_err(windows::core::Error::from_hresult)
}

pub(super) fn mapped(device: &ID3D12Device) -> windows::core::Result<()> {
    if !enabled() {
        return super::mapped_one(device);
    }
    static BATCH: Batch<()> = Batch::new();
    BATCH
        .run(device.as_raw() as usize, (), |_| {
            super::mapped_one(device).map_err(|e| e.code())
        })
        .map_err(windows::core::Error::from_hresult)
}

#[cfg(feature = "diagnostics")]
pub(crate) fn stats() -> [u64; 4] {
    std::array::from_fn(|i| COUNTS[i].load(Relaxed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concurrent_waiters_are_batched_only_after_the_first_operation_completes() {
        let batch = Arc::new(Batch::new());
        let (entered, entering) = std::sync::mpsc::channel();
        let (release, released) = std::sync::mpsc::channel();
        let first = batch.clone();
        let leader = std::thread::spawn(move || {
            first.run(1, 0, |items| {
                assert_eq!(items, [0]);
                entered.send(()).unwrap();
                released.recv().unwrap();
                Ok(())
            })
        });
        entering.recv().unwrap();
        let calls = Arc::new(Mutex::new(Vec::new()));
        let followers: Vec<_> = (1..=8)
            .map(|item| {
                let batch = batch.clone();
                let calls = calls.clone();
                std::thread::spawn(move || {
                    batch.run(1, item, |items| {
                        calls.lock().unwrap().push(items.to_vec());
                        Err(HRESULT(123))
                    })
                })
            })
            .collect();
        let guard = batch.state.lock().unwrap();
        let (guard, timeout) = batch
            .changed
            .wait_timeout_while(guard, std::time::Duration::from_secs(10), |s| {
                s.queued.len() != 8
            })
            .unwrap();
        let queued = guard.queued.len();
        drop(guard);
        release.send(()).unwrap();
        assert!(!timeout.timed_out());
        assert_eq!(queued, 8);
        assert_eq!(leader.join().unwrap(), Ok(()));
        for follower in followers {
            assert_eq!(follower.join().unwrap(), Err(HRESULT(123)));
        }
        assert_eq!(calls.lock().unwrap().len(), 1);
        assert_eq!(calls.lock().unwrap()[0].len(), 8);
    }

    #[test]
    fn panic_does_not_leave_the_batch_running() {
        let batch = Batch::new();
        assert!(std::panic::catch_unwind(|| batch.run(7, (), |_| panic!("test failure"))).is_err());
        assert_eq!(batch.run(8, (), |_| Ok(())), Ok(()));
    }
}
