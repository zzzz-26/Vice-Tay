//! Windows Raw Input for the title's GameInput mouse.

use std::io;
use std::ptr::null_mut;
use std::sync::{LazyLock, Mutex};

use windows_sys::Win32::Foundation::{ERROR_INSUFFICIENT_BUFFER, HWND, LPARAM, WPARAM};
use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
    GetAsyncKeyState, VK_LBUTTON, VK_MBUTTON, VK_RBUTTON, VK_XBUTTON1, VK_XBUTTON2,
};
use windows_sys::Win32::UI::Input::{
    GetRawInputData, GetRegisteredRawInputDevices, MOUSE_MOVE_ABSOLUTE, RAWINPUT, RAWINPUTDEVICE,
    RAWINPUTHEADER, RAWMOUSE, RID_INPUT, RIDEV_PAGEONLY, RIDEV_REMOVE, RIM_TYPEMOUSE,
    RegisterRawInputDevices,
};
use windows_sys::Win32::UI::WindowsAndMessaging::{
    GetForegroundWindow, RI_MOUSE_HWHEEL, RI_MOUSE_WHEEL, RIM_INPUT,
};

use super::{MouseState, lock};

// HID Generic Desktop page, Mouse usage.
const USAGE_PAGE: u16 = 1;
const USAGE: u16 = 2;
const DEVICE_SIZE: u32 = size_of::<RAWINPUTDEVICE>() as u32;
const HEADER_SIZE: u32 = size_of::<RAWINPUTHEADER>() as u32;

#[derive(Default)]
struct Capture {
    window: usize,
    state: MouseState,
}

static ENABLED: LazyLock<bool> =
    LazyLock::new(|| std::env::var_os("VICTORY_GAMEPAD_MOUSE").is_some_and(|v| v == "1"));
static CAPTURE: LazyLock<Mutex<Capture>> = LazyLock::new(|| Mutex::new(Capture::default()));

/// Raw Input is process-wide. Inspect the registrations before adding ours.
fn registered_mouse() -> io::Result<Option<RAWINPUTDEVICE>> {
    let mut slots = 0;
    // SAFETY: the null pointer asks for the required size in `slots`.
    let result = unsafe { GetRegisteredRawInputDevices(null_mut(), &mut slots, DEVICE_SIZE) };
    if result == u32::MAX {
        let error = io::Error::last_os_error();
        if error.raw_os_error() != Some(ERROR_INSUFFICIENT_BUFFER as i32) {
            return Err(error);
        }
    }
    if slots == 0 {
        return Ok(None);
    }
    let mut devices = vec![RAWINPUTDEVICE::default(); slots as usize];
    // SAFETY: the buffer holds `slots` device registrations.
    let count =
        unsafe { GetRegisteredRawInputDevices(devices.as_mut_ptr(), &mut slots, DEVICE_SIZE) };
    if count == u32::MAX {
        return Err(io::Error::last_os_error());
    }
    Ok(devices.into_iter().take(count as usize).find(|d| {
        d.usUsagePage == USAGE_PAGE && (d.usUsage == USAGE || d.dwFlags & RIDEV_PAGEONLY != 0)
    }))
}

fn register(window: HWND, flags: u32) -> io::Result<()> {
    let device = RAWINPUTDEVICE {
        usUsagePage: USAGE_PAGE,
        usUsage: USAGE,
        dwFlags: flags,
        hwndTarget: window,
    };
    // SAFETY: one initialized device registration; removal uses a null window.
    if unsafe { RegisterRawInputDevices(&device, 1, DEVICE_SIZE) } == 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

pub(in crate::modules) fn attach(window: HWND) {
    if !*ENABLED {
        return;
    }
    match registered_mouse() {
        Ok(None) => {}
        Ok(Some(_)) => {
            tracing::warn!(
                "the process already receives mouse Raw Input; keeping its registration"
            );
            return;
        }
        Err(error) => {
            tracing::warn!(%error, "could not check existing Raw Input registrations");
            return;
        }
    }
    // Keep normal mouse messages and let Windows limit Raw Input to the foreground.
    if let Err(error) = register(window, 0) {
        tracing::warn!(%error, "could not register mouse Raw Input for the title window");
        return;
    }
    lock(&CAPTURE).window = window as usize;
}

pub(in crate::modules) fn detach(window: HWND) {
    if !*ENABLED {
        return;
    }
    {
        let mut capture = lock(&CAPTURE);
        if capture.window != window as usize {
            return;
        }
        // Keep the cumulative axes continuous if the title recreates its window.
        capture.window = 0;
    }
    if registered_mouse().is_ok_and(|device| {
        device.is_some_and(|d| d.hwndTarget == window && d.usUsage == USAGE && d.dwFlags == 0)
    }) && let Err(error) = register(null_mut(), RIDEV_REMOVE)
    {
        tracing::warn!(%error, "could not remove mouse Raw Input registration");
    }
}

/// Observe a foreground mouse packet without taking it from the title's procedure.
pub(in crate::modules) fn receive(window: HWND, wparam: WPARAM, lparam: LPARAM) {
    if !*ENABLED || wparam & 0xff != RIM_INPUT as usize {
        return;
    }
    let mut capture = lock(&CAPTURE);
    if capture.window != window as usize {
        return;
    }
    let mut packet = RAWINPUT::default();
    let mut buffer_size = size_of::<RAWINPUT>() as u32;
    // SAFETY: the window system provides the packet handle; the buffer is aligned and sized.
    let copied = unsafe {
        GetRawInputData(
            lparam as _,
            RID_INPUT,
            (&raw mut packet).cast(),
            &mut buffer_size,
            HEADER_SIZE,
        )
    };
    if copied == u32::MAX
        || copied < HEADER_SIZE + size_of::<RAWMOUSE>() as u32
        || packet.header.dwType != RIM_TYPEMOUSE
    {
        return;
    }
    // SAFETY: plain query of the foreground window. A packet can be queued before focus changes.
    if unsafe { GetForegroundWindow() } == window {
        // SAFETY: the packet header identifies the initialized union member as a mouse.
        let mouse = unsafe { packet.data.mouse };
        accumulate(&mut capture.state, mouse);
        #[cfg(feature = "diagnostics")]
        if mouse.usFlags & MOUSE_MOVE_ABSOLUTE == 0 && (mouse.lLastX != 0 || mouse.lLastY != 0) {
            static REPORTED: std::sync::Once = std::sync::Once::new();
            REPORTED.call_once(|| {
                tracing::info!("received relative mouse movement in the title window")
            });
        }
    }
}

fn accumulate(state: &mut MouseState, mouse: RAWMOUSE) {
    // Absolute packets (such as remote pointers) are screen coordinates in a different scale.
    if mouse.usFlags & MOUSE_MOVE_ABSOLUTE == 0 {
        state.position_x = state.position_x.saturating_add(i64::from(mouse.lLastX));
        state.position_y = state.position_y.saturating_add(i64::from(mouse.lLastY));
    }
    // SAFETY: this is the documented split of the mouse's button data word.
    let buttons = unsafe { mouse.Anonymous.Anonymous };
    let flags = u32::from(buttons.usButtonFlags);
    let delta = i64::from(buttons.usButtonData as i16);
    if flags & RI_MOUSE_HWHEEL != 0 {
        state.wheel_x = state.wheel_x.saturating_add(delta);
    }
    if flags & RI_MOUSE_WHEEL != 0 {
        state.wheel_y = state.wheel_y.saturating_add(delta);
    }
}

pub(super) fn sample() -> Option<MouseState> {
    if !*ENABLED {
        return None;
    }
    // SAFETY: plain foreground window and physical button queries.
    let window = unsafe { GetForegroundWindow() } as usize;
    let mut state = {
        let capture = lock(&CAPTURE);
        if window == 0 || window != capture.window {
            return None;
        }
        capture.state
    };
    // Poll the held state so a release while unfocused cannot leave a button stuck.
    state.buttons = [VK_LBUTTON, VK_RBUTTON, VK_MBUTTON, VK_XBUTTON1, VK_XBUTTON2]
        .into_iter()
        .enumerate()
        // SAFETY: each value is a mouse button key code.
        .filter(|&(_, button)| unsafe { GetAsyncKeyState(i32::from(button)) } < 0)
        .fold(0, |bits, (index, _)| bits | 1 << index);
    // SAFETY: plain query. Discard the sample if focus changed during the button reads.
    (unsafe { GetForegroundWindow() } as usize == window).then_some(state)
}

#[cfg(test)]
mod tests {
    use windows_sys::Win32::UI::Input::{RAWMOUSE_0, RAWMOUSE_0_0};

    use super::*;

    #[test]
    fn mouse_accumulates_relative_motion_and_signed_wheels() {
        let mut state = MouseState::default();
        for (position, movement, wheel) in [
            (0, (20, -6), (RI_MOUSE_WHEEL, -120_i16)),
            (0, (-8, 3), (RI_MOUSE_HWHEEL, 120)),
            (MOUSE_MOVE_ABSOLUTE, (65535, 65535), (RI_MOUSE_WHEEL, 60)),
        ] {
            accumulate(
                &mut state,
                RAWMOUSE {
                    usFlags: position,
                    lLastX: movement.0,
                    lLastY: movement.1,
                    Anonymous: RAWMOUSE_0 {
                        Anonymous: RAWMOUSE_0_0 {
                            usButtonFlags: wheel.0 as u16,
                            usButtonData: wheel.1 as u16,
                        },
                    },
                    ..Default::default()
                },
            );
        }
        assert_eq!((state.position_x, state.position_y), (12, -3));
        assert_eq!((state.wheel_x, state.wheel_y), (120, -60));
    }
}
