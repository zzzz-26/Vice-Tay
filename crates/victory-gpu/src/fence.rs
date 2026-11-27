//! Fences: a 64-bit value and the events waiting for it. The title and our
//! queues can both signal them.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::c_void;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(not(windows))]
use crate::object::E_NOTIMPL;
use crate::object::{self, FENCE, HRESULT, Iid, Object, S_OK, Table, This, guid};
use crate::resource::PAGEABLE_IIDS;

pub(crate) const FENCE_IIDS: [Iid; 5] = [
    guid(
        0x0a753dcf,
        0xc4d8,
        0x4b91,
        [0xad, 0xf6, 0xbe, 0x5a, 0x60, 0xd9, 0x5a, 0x76],
    ),
    // ID3D12Fence1.
    guid(
        0x433685fe,
        0xe22b,
        0x4ca0,
        [0xa8, 0xdb, 0xb5, 0xb4, 0xf4, 0xdd, 0x0e, 0x4a],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
struct Fence {
    object: Object,
    /// Titles also read it straight from memory, at the fence's GPU
    /// address.
    value: AtomicU64,
    flags: u32,
    /// Events to set once the value reaches theirs (handles as integers).
    waiters: Mutex<Vec<(u64, usize)>>,
}

fn fence<'a>(this: This) -> &'a Fence {
    // SAFETY: called from fence methods only.
    unsafe { &*this.cast::<Fence>() }
}

/// Keeps a fence alive until the queue finishes using it.
#[cfg(windows)]
#[derive(PartialEq, Eq)]
pub(crate) struct FenceRef(usize);

#[cfg(windows)]
impl FenceRef {
    pub(crate) fn hold(this: This) -> Self {
        object::add_ref(this);
        Self(this as usize)
    }

    pub(crate) fn as_this(&self) -> This {
        self.0 as This
    }
}

#[cfg(windows)]
impl Clone for FenceRef {
    fn clone(&self) -> Self {
        Self::hold(self.as_this())
    }
}

#[cfg(windows)]
impl Drop for FenceRef {
    fn drop(&mut self) {
        release(self.as_this());
    }
}

pub(crate) extern "system" fn create(
    _: This,
    initial: u64,
    flags: u32,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    tracing::debug!(initial, flags, "fence");
    let fence = Box::new(Fence {
        object: Object::new(&FUNCTIONS),
        value: AtomicU64::new(initial),
        flags,
        waiters: Mutex::new(Vec::new()),
    });
    object::hand_out(FENCE, fence, iid, &FENCE_IIDS, out)
}

fn set_event(event: usize) {
    #[cfg(windows)]
    // SAFETY: the title passed an event handle.
    unsafe {
        windows_sys::Win32::System::Threading::SetEvent(event as *mut c_void)
    };
    #[cfg(not(windows))]
    let _ = event;
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(
        FENCE,
        iid,
        object::is(iid, &FENCE_IIDS).then_some(this),
        out,
    )
}
extern "system" fn release(this: This) -> u32 {
    object::release_boxed::<Fence>(this)
}
pub(crate) extern "system" fn completed(this: This) -> u64 {
    fence(this).value.load(Ordering::Acquire)
}
/// The title's `GetCompletedValue`.
extern "system" fn completed_called(this: This) -> u64 {
    let value = completed(this);
    #[cfg(all(windows, feature = "diagnostics"))]
    crate::timeline::polled(this as usize, value);
    value
}
extern "system" fn set_event_on_completion(this: This, value: u64, event: *mut c_void) -> HRESULT {
    let fence = fence(this);
    let mut waiters = fence.waiters.lock().unwrap_or_else(|e| e.into_inner());
    if fence.value.load(Ordering::Acquire) >= value {
        if !event.is_null() {
            set_event(event as usize);
        }
        return S_OK;
    }
    #[cfg(all(windows, feature = "diagnostics"))]
    crate::timeline::told("an event asked for", this as usize, value);
    if event.is_null() {
        drop(waiters);
        #[cfg(windows)]
        return wait_for_value(fence, value);
        #[cfg(not(windows))]
        return E_NOTIMPL;
    }
    waiters.push((value, event as usize));
    S_OK
}

#[cfg(windows)]
fn wait_for_value(fence: &Fence, value: u64) -> HRESULT {
    use std::time::{Duration, Instant};
    use windows_sys::Win32::System::Threading::WaitOnAddress;

    // A queued signal can be behind GPU work whose completion has been deferred.
    if crate::host::late() {
        crate::queue::before_sync();
        crate::host::settle_for_fence();
    }
    const REPORT_EVERY: Duration = Duration::from_secs(2);
    let mut next_report = Instant::now() + REPORT_EVERY;
    loop {
        let current = fence.value.load(Ordering::Acquire);
        if current >= value {
            return S_OK;
        }
        let now = Instant::now();
        if now >= next_report {
            // SAFETY: querying a live host device for its removal reason.
            if let Some(host) = crate::host::made_host()
                && let Err(error) = unsafe { host.device.GetDeviceRemovedReason() }
            {
                return error.code().0;
            }
            tracing::debug!(current, value, "waiting for a fence value");
            next_report = now + REPORT_EVERY;
        }
        // Raw writes to the fence's GPU address can change it without waking
        // any waiters, so reread the value even when there is no notification.
        // SAFETY: both pointers refer to aligned, live u64 values.
        unsafe {
            WaitOnAddress(
                (&raw const fence.value).cast(),
                (&raw const current).cast(),
                size_of::<u64>(),
                10,
            );
        }
        if fence.value.load(Ordering::Acquire) < value {
            crate::queue::release_held(false);
        }
    }
}

/// Moves the fence to `value` and sets the events that waited for it.
pub(crate) fn set(this: This, value: u64) {
    let fence = fence(this);
    #[cfg(all(windows, feature = "diagnostics"))]
    crate::timeline::told("a fence set", this as usize, value);
    let mut waiters = fence.waiters.lock().unwrap_or_else(|e| e.into_inner());
    fence.value.store(value, Ordering::Release);
    waiters.retain(|&(wanted, event)| {
        let reached = wanted <= value;
        if reached {
            set_event(event);
        }
        !reached
    });
    drop(waiters);
    #[cfg(windows)]
    // SAFETY: wake any guest thread polling/blocking via WaitOnAddress on the fence's GPU VA.
    unsafe {
        windows_sys::Win32::System::Threading::WakeByAddressAll((&raw const fence.value).cast());
    }
}
pub(crate) extern "system" fn signal(this: This, value: u64) -> HRESULT {
    let _called = crate::timeline::span("a fence set");
    set(this, value);
    // A queue may have waited for it: what it held runs now.
    #[cfg(windows)]
    crate::queue::release_held(false);
    S_OK
}

extern "system" fn creation_flags(this: This) -> u32 {
    fence(this).flags
}
/// Where the value lives (GPU addresses are CPU addresses).
extern "system" fn gpu_address(this: This) -> u64 {
    (&raw const fence(this).value) as u64
}

/// IUnknown 3, object 4, device child 1, fence 3, fence 1 (2, a console
/// one among them).
static FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<FENCE, 24>());
    f[0] = query as *const ();
    f[2] = release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    f[8] = completed_called as *const (); // 0x40
    f[9] = set_event_on_completion as *const (); // 0x48
    f[10] = signal as *const (); // 0x50
    f[11] = creation_flags as *const (); // 0x58
    f[12] = gpu_address as *const (); // 0x60
    Table(f)
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::slot;

    #[cfg(windows)]
    #[test]
    fn queued_references_keep_the_fence_alive_after_release() {
        let mut f = std::ptr::null();
        assert_eq!(create(std::ptr::null(), 1, 0, &FENCE_IIDS[0], &mut f), S_OK);
        let queued = FenceRef::hold(f);
        let waiting = queued.clone();
        assert_eq!(release(f), 2);
        drop(waiting);
        set(queued.as_this(), 5);
        assert_eq!(completed(queued.as_this()), 5);
    }

    #[cfg(windows)]
    #[test]
    fn signals_set_waiting_events() {
        use windows_sys::Win32::Foundation::{CloseHandle, WAIT_OBJECT_0};
        use windows_sys::Win32::System::Threading::{CreateEventW, WaitForSingleObject};

        let mut f = std::ptr::null();
        assert_eq!(create(std::ptr::null(), 1, 0, &FENCE_IIDS[0], &mut f), S_OK);
        assert_eq!(slot::<extern "system" fn(This) -> u64>(f, 0x40)(f), 1);
        // SAFETY: plain auto-reset event.
        let event = unsafe { CreateEventW(std::ptr::null(), 0, 0, std::ptr::null()) };
        assert!(!event.is_null());
        let wait = slot::<extern "system" fn(This, u64, *mut c_void) -> HRESULT>(f, 0x48);
        let signal = slot::<extern "system" fn(This, u64) -> HRESULT>(f, 0x50);
        assert_eq!(wait(f, 5, event), S_OK);
        // SAFETY: a valid event.
        assert_ne!(unsafe { WaitForSingleObject(event, 0) }, WAIT_OBJECT_0);
        assert_eq!(signal(f, 5), S_OK);
        // SAFETY: as above.
        assert_eq!(unsafe { WaitForSingleObject(event, 0) }, WAIT_OBJECT_0);
        assert_eq!(wait(f, 5, event), S_OK, "already reached");
        // SAFETY: as above.
        assert_eq!(unsafe { WaitForSingleObject(event, 0) }, WAIT_OBJECT_0);
        assert_eq!(wait(f, 5, std::ptr::null_mut()), S_OK);
        // The title reads the value at the fence's address.
        let at = slot::<extern "system" fn(This) -> u64>(f, 0x60)(f);
        // SAFETY: the fence's value.
        assert_eq!(unsafe { *(at as *const u64) }, 5);
        // SAFETY: ours.
        unsafe { CloseHandle(event) };
        assert_eq!(slot::<extern "system" fn(This) -> u32>(f, 0x10)(f), 0);
    }

    #[cfg(windows)]
    #[test]
    fn null_event_waits_for_a_signal_past_the_old_timeout() {
        let mut f = std::ptr::null();
        assert_eq!(create(std::ptr::null(), 1, 0, &FENCE_IIDS[0], &mut f), S_OK);
        let pending = FenceRef::hold(f);
        std::thread::scope(|scope| {
            scope.spawn(move || {
                std::thread::sleep(std::time::Duration::from_millis(400));
                signal(pending.as_this(), 5);
            });
            let wait = slot::<extern "system" fn(This, u64, *mut c_void) -> HRESULT>(f, 0x48);
            assert_eq!(wait(f, 5, std::ptr::null_mut()), S_OK);
        });
        assert_eq!(completed(f), 5);
        assert_eq!(release(f), 0);
    }

    #[test]
    fn fence1_interface_is_created_and_queried() {
        const FENCE1_IID: Iid = guid(
            0x433685fe,
            0xe22b,
            0x4ca0,
            [0xa8, 0xdb, 0xb5, 0xb4, 0xf4, 0xdd, 0x0e, 0x4a],
        );
        type Query = extern "system" fn(This, *const Iid, *mut This) -> HRESULT;
        type Flags = extern "system" fn(This) -> u32;
        type Release = extern "system" fn(This) -> u32;

        // Direct creation as ID3D12Fence1, and GetCreationFlags at slot 0x58.
        let mut f1 = std::ptr::null();
        assert_eq!(create(std::ptr::null(), 3, 0x5, &FENCE1_IID, &mut f1), S_OK);
        assert!(!f1.is_null());
        assert_eq!(slot::<Flags>(f1, 0x58)(f1), 0x5);
        assert_eq!(slot::<Release>(f1, 0x10)(f1), 0);

        // QueryInterface from ID3D12Fence to ID3D12Fence1 returns the same object.
        let mut f = std::ptr::null();
        assert_eq!(
            create(std::ptr::null(), 0, 0x2, &FENCE_IIDS[0], &mut f),
            S_OK
        );
        let mut queried = std::ptr::null();
        assert_eq!(slot::<Query>(f, 0x00)(f, &FENCE1_IID, &mut queried), S_OK);
        assert_eq!(queried, f);
        assert_eq!(slot::<Flags>(queried, 0x58)(queried), 0x2);
        assert_eq!(slot::<Release>(queried, 0x10)(queried), 1);
        assert_eq!(slot::<Release>(f, 0x10)(f), 0);
    }
}
