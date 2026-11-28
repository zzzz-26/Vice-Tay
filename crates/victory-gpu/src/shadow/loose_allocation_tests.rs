use super::*;
use std::sync::atomic::AtomicUsize;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::{Duration, Instant};
use windows::Win32::Graphics::Direct3D12::D3D12_RANGE;
use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READWRITE, VirtualAlloc, VirtualFree,
};

const K64: usize = 64 << 10;

/// Reserves `count + 1` 4 MiB spans so `count` aligned 4 MiB cells lie
/// completely inside the reservation, committing only the first 64 KiB page
/// of each cell. Removes any test keys from `LOOSE` and `EXACT` before freeing.
struct CellFixture {
    base: usize,
    cells: Vec<u64>,
    end: u64,
}

impl CellFixture {
    fn new(count: usize) -> Self {
        let reserve = (count + 1) * WINDOW as usize;
        // SAFETY: reserves private address space owned by this fixture.
        let raw = unsafe { VirtualAlloc(std::ptr::null(), reserve, MEM_RESERVE, PAGE_NOACCESS) };
        assert!(!raw.is_null());
        let base = raw as usize;
        let first = (base as u64).next_multiple_of(WINDOW);
        let mut cells = Vec::with_capacity(count);
        for i in 0..count {
            let cell = first + i as u64 * WINDOW;
            // SAFETY: commits one 64 KiB page inside our reservation.
            let committed = unsafe {
                VirtualAlloc(cell as *mut _, K64, MEM_COMMIT, PAGE_READWRITE)
            };
            assert!(!committed.is_null());
            cells.push(cell);
        }
        let end = first + count as u64 * WINDOW;
        // Another test may have left a window or view on addresses it freed
        // and Windows has just handed to us; nothing of ours is there yet.
        forget_overlapping(first, end);
        Self { base, end, cells }
    }

    fn commit_at(&self, addr: u64, len: usize) {
        assert!(addr >= self.cells[0] && addr + len as u64 <= self.end);
        // SAFETY: inside our reservation.
        let committed = unsafe {
            VirtualAlloc(addr as *mut _, len, MEM_COMMIT, PAGE_READWRITE)
        };
        assert!(!committed.is_null());
    }

    fn fill_at(&self, addr: u64, len: usize, value: u8) {
        assert!(addr >= self.cells[0] && addr + len as u64 <= self.end);
        // SAFETY: caller committed `[addr, addr + len)` inside this fixture.
        unsafe { (addr as *mut u8).write_bytes(value, len) };
    }
}

/// Takes every loose window and exact view that overlaps `[lo, hi)` out of
/// the tables (keys are `(start, len)`), and drops them outside the locks.
fn forget_overlapping(lo: u64, hi: u64) {
    let overlaps = |&(start, len): &(u64, u64)| start < hi && start.saturating_add(len.max(1)) > lo;
    let dropped_loose: Vec<Arc<Shadow>> = {
        let mut map = LOOSE.write().unwrap_or_else(|e| e.into_inner());
        let keys: Vec<(u64, u64)> = map.range(..(hi, 0)).map(|(&k, _)| k).filter(overlaps).collect();
        keys.into_iter().filter_map(|k| map.remove(&k)).collect()
    };
    let dropped_exact: Vec<Arc<Shadow>> = {
        let mut map = EXACT.write().unwrap_or_else(|e| e.into_inner());
        let keys: Vec<(u64, u64)> = map.range(..(hi, 0)).map(|(&k, _)| k).filter(overlaps).collect();
        keys.into_iter()
            .filter_map(|k| map.remove(&k).map(|(s, _)| s))
            .collect()
    };
    drop(dropped_loose);
    drop(dropped_exact);
}

impl Drop for CellFixture {
    fn drop(&mut self) {
        forget_overlapping(self.cells[0], self.end);
        // SAFETY: releases our private reservation after all table entries are gone.
        unsafe { VirtualFree(self.base as *mut _, 0, MEM_RELEASE) };
    }
}

fn default_query(address: u64, len: u64) -> LooseQuery {
    LooseQuery {
        address,
        end: address + len,
        unsized_ask: false,
        use_tails: false,
        use_holder: false,
        grid: WINDOW,
    }
}

fn shared_only(len: u64) -> Option<LooseBacking> {
    let (buffer, data) = host()?.shared_buffer(len)?;
    Some(LooseBacking::Shared {
        buffer,
        data,
        card: None,
    })
}

fn shared_with_card(len: u64) -> Option<LooseBacking> {
    let host = host()?;
    let (buffer, data) = host.shared_buffer(len)?;
    let card = host.uav_buffer(len)?;
    WINDOW_CARD_BYTES.fetch_add(len, Ordering::Relaxed);
    Some(LooseBacking::Shared {
        buffer,
        data,
        card: Some(card),
    })
}

fn delayed_loose(
    ready: &Sender<()>,
    go: &Receiver<()>,
    len: u64,
    make: impl FnOnce(u64) -> Option<LooseBacking>,
) -> Option<LooseBacking> {
    let made = make(len);
    ready.send(()).unwrap();
    go.recv().unwrap();
    made
}

fn delayed_exact(ready: &Sender<()>, go: &Receiver<()>, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
    let made = host()?.shared_buffer(size);
    ready.send(()).unwrap();
    go.recv().unwrap();
    made
}

fn probe_while_pending<T>(access: impl Fn() -> Option<T>) -> Option<T> {
    // Allow unrelated tests a short window to release the same table lock.
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        if let Some(value) = access() {
            return Some(value);
        }
        if Instant::now() >= deadline {
            return None;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn readback_bytes(shadow: &Arc<Shadow>, from: u64, len: usize) -> Vec<u8> {
    let host = host().expect("D3D12 host");
    let readback = host.read_back_buffer(len as u64).unwrap();
    let list = host.new_list().unwrap();
    shadow.refresh(&list, from, len as u64);
    // SAFETY: `[from, from + len)` lies inside `shadow` and `readback` is a
    // live COPY_DEST buffer of `len` bytes.
    unsafe {
        list.list.CopyBufferRegion(
            &readback,
            0,
            &shadow.buffer,
            from - shadow.start,
            len as u64,
        );
    }
    list.close();
    host.execute_sync_internal(&list);
    let mut ptr = std::ptr::null_mut();
    let read = D3D12_RANGE { Begin: 0, End: len };
    let written = D3D12_RANGE { Begin: 0, End: 0 };
    // SAFETY: `execute_sync_internal` waited for the GPU copy to complete.
    let bytes = unsafe {
        readback.Map(0, Some(&read), Some(&mut ptr)).unwrap();
        assert!(!ptr.is_null());
        let out = std::slice::from_raw_parts(ptr.cast::<u8>(), len).to_vec();
        readback.Unmap(0, Some(&written));
        out
    };
    assert!(
        host.log_messages().is_none(),
        "D3D12 reported validation errors"
    );
    bytes
}

#[test]
fn unrelated_loose_lookups_and_parallel_allocations_proceed_while_a_window_is_pending() {
    if host().is_none() || crate::port::on() || crate::inplace::level() >= 1 {
        return;
    }
    let fixture = CellFixture::new(3);
    let (cell_a, cell_b, cell_c) = (fixture.cells[0], fixture.cells[1], fixture.cells[2]);
    let existing = loose_with(default_query(cell_b, K64 as u64), shared_only).unwrap();
    let follower_made = AtomicUsize::new(0);
    let parallel_made = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        let (ready, waiting) = channel();
        let (go, resume) = channel();
        let first = scope.spawn(move || {
            loose_with(default_query(cell_a, K64 as u64), |len| {
                delayed_loose(&ready, &resume, len, shared_only)
            })
            .unwrap()
        });
        waiting.recv().unwrap();

        // The per-key ticket for `cell_a` is held while `LOOSE` stays free.
        assert!(
            pending_ticket(&LOOSE_BUILDING, (cell_a, WINDOW))
                .try_lock()
                .is_err()
        );

        // Unrelated reads and read-only table walks proceed without waiting.
        let unrelated = probe_while_pending(|| {
            LOOSE
                .try_read()
                .ok()
                .and_then(|map| map.get(&(cell_b, WINDOW)).cloned())
        });
        assert!(
            unrelated.as_ref().is_some_and(|s| Arc::ptr_eq(&existing, s)),
            "unrelated loose window must remain readable during pending allocation"
        );
        let hit = loose_with(default_query(cell_b, K64 as u64), |_| {
            panic!("already-published window must not allocate")
        })
        .unwrap();
        assert!(Arc::ptr_eq(&existing, &hit));
        assert!(spans().iter().any(|&(s, l, w)| w && s == cell_b && l == WINDOW));

        // An unrelated missing window allocates and publishes in parallel
        // BEFORE `first` is released.
        let parallel_count = &parallel_made;
        let parallel = scope.spawn(move || {
            loose_with(default_query(cell_c, K64 as u64), |len| {
                parallel_count.fetch_add(1, Ordering::Relaxed);
                shared_only(len)
            })
            .unwrap()
        });
        let c_shadow = parallel.join().unwrap();
        assert_eq!(c_shadow.start, cell_c);
        assert_eq!(parallel_made.load(Ordering::Relaxed), 1);

        // A follower for the SAME pending key waits on the per-key ticket and
        // reuses the winner without allocating a duplicate.
        let follower_count = &follower_made;
        let follower = scope.spawn(move || {
            loose_with(default_query(cell_a, K64 as u64), |len| {
                follower_count.fetch_add(1, Ordering::Relaxed);
                shared_only(len)
            })
            .unwrap()
        });
        std::thread::sleep(Duration::from_millis(15));

        go.send(()).unwrap();
        let winner = first.join().unwrap();
        let followed = follower.join().unwrap();
        assert!(Arc::ptr_eq(&winner, &followed));
        assert_eq!(follower_made.load(Ordering::Relaxed), 0);
    });
}

#[test]
fn loose_reselects_concurrent_wider_tail_or_holder_and_gpu_readback_sees_live_bytes() {
    if host().is_none() || crate::port::on() || crate::inplace::level() >= 1 {
        return;
    }
    const STEP: u64 = 256 << 10;
    // Exercise both tail-mode (`use_tails=true`) and fine-grid holder-mode
    // (`use_holder=true`) concurrent wider-window publication, including
    // card-backed candidate budget accounting on discard.
    for use_tails in [true, false] {
        let fixture = CellFixture::new(1);
        let cell = fixture.cells[0];
        let ask_addr = cell + 2 * STEP;
        let wider_addr = cell + STEP;
        fixture.commit_at(ask_addr, K64);
        fixture.fill_at(ask_addr, K64, 0x22);

        let q = LooseQuery {
            address: ask_addr,
            end: ask_addr + if use_tails { 1 } else { K64 as u64 },
            unsized_ask: use_tails,
            use_tails,
            use_holder: !use_tails,
            grid: if use_tails { WINDOW } else { STEP },
        };
        let card_bytes_before = WINDOW_CARD_BYTES.load(Ordering::Relaxed);

        std::thread::scope(|scope| {
            let (ready, waiting) = channel();
            let (go, resume) = channel();
            let worker = scope.spawn(move || {
                loose_with(q, |len| delayed_loose(&ready, &resume, len, shared_with_card))
                    .unwrap()
            });
            waiting.recv().unwrap();

            // `LOOSE` must remain writable while `worker` allocates outside the lock.
            if probe_while_pending(|| LOOSE.try_write().ok().map(drop)).is_none() {
                go.send(()).unwrap();
                worker.join().unwrap();
                panic!("LOOSE writer must not be held during allocation");
            }

            // Concurrently publish a wider window covering `ask_addr`:
            // - tail mode: a longer tail starting at `wider_addr` to cell end;
            // - holder mode: a 1 MiB window `[cell, cell + 1 MiB)` covering `ask_addr`.
            let wider = if use_tails {
                loose_with(
                    LooseQuery {
                        address: wider_addr,
                        end: wider_addr + 1,
                        unsized_ask: true,
                        use_tails: true,
                        use_holder: false,
                        grid: WINDOW,
                    },
                    shared_only,
                )
                .unwrap()
            } else {
                let backing = shared_only(1 << 20).unwrap();
                let shadow = build_loose_shadow(cell, 1 << 20, backing);
                LOOSE
                    .write()
                    .unwrap_or_else(|e| e.into_inner())
                    .insert((cell, 1 << 20), shadow.clone());
                shadow
            };

            // Update title memory before resuming `worker`.
            fixture.fill_at(ask_addr, K64, 0x6d);
            go.send(()).unwrap();

            let chosen = worker.join().unwrap();
            assert!(
                Arc::ptr_eq(&chosen, &wider),
                "worker must adopt the concurrently published wider window (use_tails={use_tails})"
            );
            // Discarding the unpublished card-backed candidate must restore
            // `WINDOW_CARD_BYTES` cleanly without leaking budget.
            assert_eq!(
                WINDOW_CARD_BYTES.load(Ordering::Relaxed),
                card_bytes_before,
                "discarded candidate must release card budget (use_tails={use_tails})"
            );

            // Verify hit path succeeds under `LOOSE.read()` even for tails/holder.
            let again = loose_with(q, |_| panic!("tail/holder hit must not re-allocate")).unwrap();
            assert!(Arc::ptr_eq(&again, &wider));

            // Prove byte-exact GPU refresh + readback from the published window.
            let got = readback_bytes(&chosen, ask_addr, K64);
            assert!(
                got.iter().all(|&b| b == 0x6d),
                "GPU readback did not match live title bytes (use_tails={use_tails})"
            );
        });
    }
}

#[test]
fn loose_allocation_failure_leaves_window_open_to_retry() {
    if host().is_none() || crate::port::on() || crate::inplace::level() >= 1 {
        return;
    }
    let fixture = CellFixture::new(1);
    let cell = fixture.cells[0];
    let q = default_query(cell, K64 as u64);
    assert!(loose_with(q, |_| None).is_none());
    assert!(loose_with(q, shared_only).is_some());
}

#[test]
fn exact_lookups_and_parallel_allocations_proceed_while_pending_with_gpu_readback() {
    if host().is_none() {
        return;
    }
    let fixture = CellFixture::new(1);
    let base = fixture.cells[0];
    let (addr_a, addr_b, addr_c) = (base, base + 4096, base + 8192);
    fixture.fill_at(addr_a, 4096, 0x9e);

    let existing = exact_with(addr_b, 4096, |size| host()?.shared_buffer(size)).unwrap();
    let follower_made = AtomicUsize::new(0);
    let parallel_made = AtomicUsize::new(0);

    std::thread::scope(|scope| {
        let (ready, waiting) = channel();
        let (go, resume) = channel();
        let first = scope.spawn(move || {
            exact_with(addr_a, 4096, |size| delayed_exact(&ready, &resume, size)).unwrap()
        });
        waiting.recv().unwrap();

        // The per-key ticket for `addr_a` is held while `EXACT` stays free.
        assert!(
            pending_ticket(&EXACT_BUILDING, (addr_a, 4096))
                .try_lock()
                .is_err()
        );

        // Existing exact view remains readable without waiting on `first`.
        let unrelated = probe_while_pending(|| {
            EXACT
                .try_read()
                .ok()
                .and_then(|map| map.get(&(addr_b, 4096)).map(|(s, _)| s.clone()))
        });
        assert!(
            unrelated.as_ref().is_some_and(|s| Arc::ptr_eq(&existing, s)),
            "unrelated exact view must remain readable during pending allocation"
        );
        let hit = exact_with(addr_b, 4096, |_| panic!("cached exact view must not allocate"))
            .unwrap();
        assert!(Arc::ptr_eq(&existing, &hit));

        // Unrelated missing exact view allocates and publishes in parallel
        // BEFORE `first` is released.
        let parallel_count = &parallel_made;
        let parallel = scope.spawn(move || {
            exact_with(addr_c, 4096, |size| {
                parallel_count.fetch_add(1, Ordering::Relaxed);
                host()?.shared_buffer(size)
            })
            .unwrap()
        });
        let c_shadow = parallel.join().unwrap();
        assert_eq!(c_shadow.start, addr_c);
        assert_eq!(parallel_made.load(Ordering::Relaxed), 1);

        // Follower on the same key waits on the per-key ticket and reuses `first`.
        let follower_count = &follower_made;
        let follower = scope.spawn(move || {
            exact_with(addr_a, 4096, |size| {
                follower_count.fetch_add(1, Ordering::Relaxed);
                host()?.shared_buffer(size)
            })
            .unwrap()
        });
        std::thread::sleep(Duration::from_millis(15));

        go.send(()).unwrap();
        let winner = first.join().unwrap();
        let followed = follower.join().unwrap();
        assert!(Arc::ptr_eq(&winner, &followed));
        assert_eq!(follower_made.load(Ordering::Relaxed), 0);

        // Prove byte-exact GPU refresh + readback on the exact-view shadow.
        let got = readback_bytes(&winner, addr_a, 4096);
        assert!(
            got.iter().all(|&b| b == 0x9e),
            "GPU readback did not match live title bytes for exact view"
        );
    });

    // Allocation failure leaves the key open to retry.
    let addr_d = base + 12288;
    assert!(exact_with(addr_d, 4096, |_| None).is_none());
    assert!(exact_with(addr_d, 4096, |size| host()?.shared_buffer(size)).is_some());
}

#[test]
fn exact_eviction_keeps_held_entries_and_uses_publish_time_tick() {
    if host().is_none() {
        return;
    }
    let fixture = CellFixture::new(1);
    let base = fixture.cells[0];
    let (idle_addr, held_addr, slow_addr) = (base, base + 256, base + 512);
    let make = |size: u64| host()?.shared_buffer(size);

    let idle = exact_with_limits(idle_addr, 256, 2, 2, make).unwrap();
    drop(idle);
    let held = exact_with_limits(held_addr, 256, 2, 2, make).unwrap();

    // Age the idle entry while the slow allocation is blocked.
    let (slow, before_publish) = std::thread::scope(|scope| {
        let (ready, waiting) = channel();
        let (go, resume) = channel();
        let worker = scope.spawn(move || {
            exact_with_limits(slow_addr, 256, 2, 2, |size| {
                delayed_exact(&ready, &resume, size)
            })
            .unwrap()
        });
        waiting.recv().unwrap();
        for _ in 0..4 {
            let _ = exact_with_limits(held_addr, 256, 2, 2, make).unwrap();
        }
        let before_publish = EXACT_TICK.load(Ordering::Relaxed);
        go.send(()).unwrap();
        (worker.join().unwrap(), before_publish)
    });

    // Check the published stamp directly. Other tests also advance the shared clock.
    let map = EXACT.read().unwrap_or_else(|e| e.into_inner());
    assert!(
        !map.contains_key(&(idle_addr, 256)),
        "idle exact entry must be evicted"
    );
    let (held_in_cache, _) = map.get(&(held_addr, 256)).expect("held exact entry");
    assert!(Arc::ptr_eq(&held, held_in_cache));
    let (slow_in_cache, used) = map.get(&(slow_addr, 256)).expect("fresh exact entry");
    assert!(Arc::ptr_eq(&slow, slow_in_cache));
    assert!(
        used.load(Ordering::Relaxed) >= before_publish,
        "the slow allocation must use a publish-time tick"
    );
}
