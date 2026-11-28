//! Confirm the queue that marks new memory usable. A device-removed fence
//! value or a failed Win32 event wait is not a successful mapping.

use super::{ID3D12CommandQueue, ID3D12Device, ID3D12Fence};
use windows::Win32::Foundation::{
    E_UNEXPECTED, ERROR_TIMEOUT, HANDLE, WAIT_FAILED, WAIT_OBJECT_0, WAIT_TIMEOUT,
};
use windows::Win32::Graphics::Dxgi::DXGI_ERROR_DEVICE_REMOVED;
use windows::Win32::System::Threading::{INFINITE, WaitForSingleObject};
use windows::core::{Error, HRESULT, Result};

const MAPPING_TIMEOUT: HRESULT = HRESULT::from_win32(ERROR_TIMEOUT.0);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Attempt {
    Initial,
    Retry,
    AfterTimeout,
}

#[derive(Default)]
pub(super) struct Progress {
    value: u64,
    pending: Option<Attempt>,
}

impl Progress {
    pub(super) fn run(
        &mut self,
        submit: impl FnOnce(),
        mut confirm: impl FnMut(u64, Attempt) -> Result<()>,
    ) -> Result<()> {
        if let Some(attempt) = self.pending {
            // Confirm the previous marker before using the reusable list again.
            self.check(attempt, &mut confirm)?;
        }
        self.value = self
            .value
            .checked_add(1)
            .filter(|v| *v != u64::MAX)
            .ok_or_else(|| Error::from_hresult(E_UNEXPECTED))?;
        self.pending = Some(Attempt::Retry);
        submit();
        self.check(Attempt::Initial, &mut confirm)
    }

    fn check(
        &mut self,
        attempt: Attempt,
        confirm: &mut impl FnMut(u64, Attempt) -> Result<()>,
    ) -> Result<()> {
        let result = confirm(self.value, attempt);
        self.pending = match &result {
            Ok(()) => None,
            Err(error) if attempt == Attempt::AfterTimeout || error.code() == MAPPING_TIMEOUT => {
                Some(Attempt::AfterTimeout)
            }
            Err(_) => Some(Attempt::Retry),
        };
        result
    }
}

fn reached(value: u64, completed: u64, removed: impl FnOnce() -> Result<()>) -> Result<bool> {
    if completed == u64::MAX {
        removed()?;
        // The sentinel is not a valid success even if the runtime no longer
        // has a more specific reason to report.
        return Err(Error::from_hresult(DXGI_ERROR_DEVICE_REMOVED));
    }
    Ok(completed >= value)
}

fn wait_until_complete(
    complete: impl Fn() -> Result<bool>,
    arm: impl FnOnce() -> Result<()>,
    mut wait: impl FnMut() -> Result<()>,
) -> Result<()> {
    if complete()? {
        return Ok(());
    }
    arm()?;
    loop {
        wait()?;
        // A wake might belong to a previous wait, or signal removal.
        if complete()? {
            return Ok(());
        }
    }
}

fn mapping_wait_ms() -> u32 {
    static LIMIT: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| {
        std::env::var("VICTORY_MAPPING_WAIT_MS")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .filter(|&ms| ms != 0)
            .unwrap_or(INFINITE)
    })
}

fn wait_event(event: HANDLE, millis: u32) -> Result<()> {
    // SAFETY: the caller owns the event for the duration of the wait.
    match unsafe { WaitForSingleObject(event, millis) } {
        WAIT_OBJECT_0 => Ok(()),
        WAIT_TIMEOUT => Err(Error::from_hresult(MAPPING_TIMEOUT)),
        WAIT_FAILED => Err(Error::from_thread()),
        _ => Err(Error::from_hresult(E_UNEXPECTED)),
    }
}

pub(super) fn confirm(
    device: &ID3D12Device,
    queue: &ID3D12CommandQueue,
    fence: &ID3D12Fence,
    event: HANDLE,
    value: u64,
    attempt: Attempt,
) -> Result<()> {
    // SAFETY: these objects belong to the serialized mapping queue, whose
    // caller retains the event, command list, and resources during recovery.
    unsafe {
        let complete = || {
            reached(value, fence.GetCompletedValue(), || {
                device.GetDeviceRemovedReason()
            })
        };
        if attempt != Attempt::Initial && complete()? {
            return Ok(());
        }
        if attempt == Attempt::AfterTimeout {
            // The previous wait already queued and armed this marker. Leave
            // both in place and let other mapping callers return promptly.
            device.GetDeviceRemovedReason()?;
            return Err(Error::from_hresult(MAPPING_TIMEOUT));
        }
        // Retrying a marker after an API error cannot overtake that submission.
        queue.Signal(fence, value)?;
        let limit = mapping_wait_ms();
        let started = (limit != INFINITE).then(std::time::Instant::now);
        let result = wait_until_complete(
            complete,
            || fence.SetEventOnCompletion(value, event),
            || {
                let remaining = started.map_or(INFINITE, |started| {
                    limit.saturating_sub(
                        u32::try_from(started.elapsed().as_millis()).unwrap_or(u32::MAX),
                    )
                });
                if remaining == 0 {
                    return Err(Error::from_hresult(MAPPING_TIMEOUT));
                }
                wait_event(event, remaining)
            },
        );
        if result
            .as_ref()
            .is_err_and(|error| error.code() == MAPPING_TIMEOUT)
        {
            if complete()? {
                return Ok(());
            }
            device.GetDeviceRemovedReason()?;
            tracing::warn!(value, limit, "mapping fence wait timed out");
        }
        result
    }
}

#[cfg(test)]
mod tests;
