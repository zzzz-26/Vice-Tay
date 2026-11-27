//! Confirm the queue that marks new memory usable. A device-removed fence
//! value or a failed Win32 event wait is not a successful mapping.

use super::{ID3D12CommandQueue, ID3D12Device, ID3D12Fence};
use windows::Win32::Foundation::{E_UNEXPECTED, HANDLE, WAIT_FAILED, WAIT_OBJECT_0};
use windows::Win32::Graphics::Dxgi::DXGI_ERROR_DEVICE_REMOVED;
use windows::Win32::System::Threading::{INFINITE, WaitForSingleObject};
use windows::core::{Error, Result};

#[derive(Default)]
pub(super) struct Progress {
    value: u64,
    pending: bool,
}

impl Progress {
    pub(super) fn run(
        &mut self,
        submit: impl FnOnce(),
        mut confirm: impl FnMut(u64, bool) -> Result<()>,
    ) -> Result<()> {
        if self.pending {
            // An earlier call could not confirm its submission. Do not resubmit
            // the reusable list while it could still be in flight. Even if the
            // old fence is complete, this call still needs its own marker.
            confirm(self.value, true)?;
        }
        self.value = self
            .value
            .checked_add(1)
            .filter(|v| *v != u64::MAX)
            .ok_or_else(|| Error::from_hresult(E_UNEXPECTED))?;
        self.pending = true;
        submit();
        confirm(self.value, false)?;
        self.pending = false;
        Ok(())
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

fn wait_event(event: HANDLE) -> Result<()> {
    // SAFETY: the caller owns the event for the duration of the wait.
    match unsafe { WaitForSingleObject(event, INFINITE) } {
        WAIT_OBJECT_0 => Ok(()),
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
    recovering: bool,
) -> Result<()> {
    // SAFETY: these objects belong to the serialized mapping queue, whose
    // caller retains the event, command list, and resources during recovery.
    unsafe {
        let complete = || {
            reached(value, fence.GetCompletedValue(), || {
                device.GetDeviceRemovedReason()
            })
        };
        if recovering && complete()? {
            return Ok(());
        }
        // On recovery, retrying the same marker cannot order ahead of the
        // earlier submission. Only this path signals this queue's fence.
        queue.Signal(fence, value)?;
        wait_until_complete(
            complete,
            || fence.SetEventOnCompletion(value, event),
            || wait_event(event),
        )
    }
}

#[cfg(test)]
mod tests;
