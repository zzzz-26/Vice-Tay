//! The title's life cycle. A console title is never told it is closing:
//! quitting it or launching a new game first constrains it (its
//! constrained-change routines run), then suspends it (its suspend routines
//! run, and it has a second to save) and then ends the process. Closing the
//! title's window does the same here; a new launch then starts `victory`
//! again, which waits for this process to be gone.
#![allow(non_snake_case)]
#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::too_many_arguments)]

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::time::Duration;

use windows_sys::Win32::Foundation::{HINSTANCE, HWND, LPARAM, LRESULT, RECT, WPARAM};
use windows_sys::Win32::System::Threading::{GetCurrentProcess, TerminateProcess};
use windows_sys::Win32::UI::WindowsAndMessaging::{self as wm, HMENU};
use windows_sys::core::{PCSTR, PCWSTR, w};

use crate::{HleModule, hle_export};

/// A suspend routine (`quiesced`) or a constrained-change routine
/// (`constrained`): both take a flag and the title's context.
type Routine = extern "system" fn(flag: u8, context: *mut c_void);

/// Registration id, routine and context.
type Registration = (usize, Routine, usize);

/// Each suspend routine.
static ROUTINES: Mutex<Vec<Registration>> = Mutex::new(Vec::new());

/// Each constrained-change routine.
static CONSTRAINED: Mutex<Vec<Registration>> = Mutex::new(Vec::new());

fn routines() -> MutexGuard<'static, Vec<Registration>> {
    ROUTINES.lock().unwrap_or_else(|e| e.into_inner())
}

fn constrained() -> MutexGuard<'static, Vec<Registration>> {
    CONSTRAINED.lock().unwrap_or_else(|e| e.into_inner())
}

/// How long the console lets suspend routines run before it ends the title.
const SUSPEND_DEADLINE: Duration = Duration::from_secs(1);

const ERROR_INVALID_PARAMETER: u32 = 87;

/// Adds `routine` to `list` and writes its registration id.
fn register(
    mut list: MutexGuard<'static, Vec<Registration>>,
    routine: Option<Routine>,
    context: *mut c_void,
    registration: *mut usize,
) -> u32 {
    let Some(routine) = routine else {
        return ERROR_INVALID_PARAMETER;
    };
    if registration.is_null() {
        return ERROR_INVALID_PARAMETER;
    }
    static NEXT: AtomicUsize = AtomicUsize::new(1);
    let id = NEXT.fetch_add(1, Ordering::Relaxed);
    list.push((id, routine, context as usize));
    // SAFETY: checked non-null; the title passes room for its registration.
    unsafe { registration.write(id) };
    0
}

pub extern "system" fn RegisterAppStateChangeNotification(
    routine: Option<Routine>,
    context: *mut c_void,
    registration: *mut usize,
) -> u32 {
    register(routines(), routine, context, registration)
}

pub extern "system" fn UnregisterAppStateChangeNotification(registration: usize) {
    routines().retain(|&(id, ..)| id != registration);
}

/// The host's own version never calls the routine (a desktop app is never
/// constrained), so a title waiting for it would never suspend.
pub extern "system" fn RegisterAppConstrainedChangeNotification(
    routine: Option<Routine>,
    context: *mut c_void,
    registration: *mut usize,
) -> u32 {
    register(constrained(), routine, context, registration)
}

pub extern "system" fn UnregisterAppConstrainedChangeNotification(registration: usize) {
    constrained().retain(|&(id, ..)| id != registration);
}

/// The window procedure a title window had before ours.
const TITLE_PROC: PCWSTR = w!("victory.wndproc");

pub extern "system" fn CreateWindowExW(
    ex_style: u32,
    class: PCWSTR,
    name: PCWSTR,
    style: u32,
    x: i32,
    y: i32,
    width: i32,
    height: i32,
    parent: HWND,
    menu: HMENU,
    instance: HINSTANCE,
    param: *const c_void,
) -> HWND {
    let ((ex_style, style, (x, y, width, height)), state) =
        place(ex_style, style, parent, x, y, width, height);
    // SAFETY: the title's own arguments, forwarded (the size fitted, or the
    // window made full screen).
    let hwnd = unsafe {
        wm::CreateWindowExW(
            ex_style, class, name, style, x, y, width, height, parent, menu, instance, param,
        )
    };
    hook(hwnd, state);
    hwnd
}

/// See [`CreateWindowExW`].
pub extern "system" fn CreateWindowExA(
    ex_style: u32,
    class: PCSTR,
    name: PCSTR,
    style: u32,
    x: i32,
    y: i32,
    width: i32,
    height: i32,
    parent: HWND,
    menu: HMENU,
    instance: HINSTANCE,
    param: *const c_void,
) -> HWND {
    let ((ex_style, style, (x, y, width, height)), state) =
        place(ex_style, style, parent, x, y, width, height);
    // SAFETY: the title's own arguments, forwarded (the size fitted, or the
    // window made full screen).
    let hwnd = unsafe {
        wm::CreateWindowExA(
            ex_style, class, name, style, x, y, width, height, parent, menu, instance, param,
        )
    };
    hook(hwnd, state);
    hwnd
}

/// Where a window the title makes goes: full screen ([`full_screen`]), or
/// fitted to the screen ([`fit`]); with the styles it gets.
fn place(
    ex_style: u32,
    style: u32,
    parent: HWND,
    x: i32,
    y: i32,
    width: i32,
    height: i32,
) -> (Windowed, Option<WindowState>) {
    if !placeable(ex_style, style, !parent.is_null()) || zero_sized(style, width, height) {
        return ((ex_style, style, (x, y, width, height)), None);
    }
    // Use the chosen monitor's work area for the windowed size and its
    // whole area for full screen.
    let chosen = chosen_monitor();
    let windowed = (ex_style, style, fit(x, y, width, height, chosen.map(|(work, _)| work)));
    let (origin, screen) = match chosen.map(|(_, whole)| whole) {
        Some(m) => ((m.left, m.top), (m.right - m.left, m.bottom - m.top)),
        // SAFETY: plain queries of the primary monitor's size.
        None => ((0, 0), unsafe { (wm::GetSystemMetrics(wm::SM_CXSCREEN), wm::GetSystemMetrics(wm::SM_CYSCREEN)) }),
    };
    let Some(full) = full_screen(ex_style, style, !parent.is_null(), screen)
        .map(|(e, s, (_, _, w, h))| (e, s, (origin.0, origin.1, w, h)))
    else {
        return (windowed, None);
    };
    let full_screen = full_screen_on();
    if full_screen {
        tracing::info!(width = screen.0, height = screen.1, left = origin.0, top = origin.1, "title window made full screen");
    }
    (if full_screen { full } else { windowed }, Some(WindowState { windowed, full_screen }))
}

/// Whether a window asks for no area of its own. Win32 supplies a default
/// size to overlapped windows, and uses zero size for a pop-up or child
/// with a default width.
fn zero_sized(style: u32, width: i32, height: i32) -> bool {
    match width {
        wm::CW_USEDEFAULT => style & (wm::WS_POPUP | wm::WS_CHILD) != 0,
        0 => true,
        _ => height == 0,
    }
}

/// The styles and place the title's top-level window had (or would have
/// had) as a window, for going back from full screen.
type WindowRect = (i32, i32, i32, i32);
type Windowed = (u32, u32, WindowRect);

#[derive(Clone, Copy)]
struct WindowState {
    windowed: Windowed,
    full_screen: bool,
}

static WINDOWS: Mutex<BTreeMap<usize, WindowState>> = Mutex::new(BTreeMap::new());

/// F11 or Alt+Enter on the title's window: full screen (frameless, over the
/// monitor it is on) or back to the window it was made as, whichever it is
/// not now. The swap chain stretches to either.
fn toggle_full_screen(hwnd: HWND) {
    let Some(mut state) = WINDOWS.lock().unwrap_or_else(|e| e.into_inner()).get(&(hwnd as usize)).copied() else {
        return;
    };
    // SAFETY: queries of the title's window and the monitor it is on.
    let monitor = unsafe {
        let mut info: windows_sys::Win32::Graphics::Gdi::MONITORINFO = std::mem::zeroed();
        info.cbSize = std::mem::size_of_val(&info) as u32;
        let on = windows_sys::Win32::Graphics::Gdi::MonitorFromWindow(hwnd, windows_sys::Win32::Graphics::Gdi::MONITOR_DEFAULTTONEAREST);
        (windows_sys::Win32::Graphics::Gdi::GetMonitorInfoW(on, &mut info) != 0).then_some(info)
    };
    let Some(monitor) = monitor else { return };
    if !state.full_screen {
        let mut rect = RECT::default();
        // SAFETY: a query of the actual window while it still has its frame.
        if unsafe { wm::GetWindowRect(hwnd, &mut rect) } != 0 {
            state.windowed.2 = (rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top);
        }
    } else {
        state.windowed.2 = restore_rect(state.windowed.2, monitor.rcWork);
    }
    let (ex_style, style, (x, y, w, h)) = toggled(state.full_screen, state.windowed, monitor.rcMonitor);
    // SAFETY: the title's own window, on its own thread (this procedure).
    unsafe {
        wm::SetWindowLongPtrW(hwnd, wm::GWL_STYLE, style as isize);
        wm::SetWindowLongPtrW(hwnd, wm::GWL_EXSTYLE, ex_style as isize);
        wm::SetWindowPos(hwnd, std::ptr::null_mut(), x, y, w, h, wm::SWP_FRAMECHANGED | wm::SWP_NOZORDER | wm::SWP_SHOWWINDOW);
    }
    state.full_screen = !state.full_screen;
    if let Some(saved) = WINDOWS.lock().unwrap_or_else(|e| e.into_inner()).get_mut(&(hwnd as usize)) {
        *saved = state;
    }
    tracing::info!(full_screen = state.full_screen, "the title's window switched");
}

/// A window created full screen never gave Win32 a chance to resolve its
/// default windowed coordinates. Center a usable rectangle in the work area.
fn restore_rect((x, y, width, height): WindowRect, area: RECT) -> WindowRect {
    let available_width = (area.right - area.left).max(1);
    let available_height = (area.bottom - area.top).max(1);
    let width_default = width == wm::CW_USEDEFAULT;
    let width = if width <= 0 { (available_width * 3 / 4).max(1) } else { width };
    let height = if width_default || height <= 0 { (available_height * 3 / 4).max(1) } else { height };
    let x_default = x == wm::CW_USEDEFAULT;
    let left = if x_default { area.left + (available_width - width) / 2 } else { x };
    let top = if x_default || y == wm::CW_USEDEFAULT { area.top + (available_height - height) / 2 } else { y };
    (left, top, width, height)
}

/// The styles and place after a switch from full screen (`full`) or to it:
/// back to `windowed`, or a frameless popup over `monitor`. Visible either way.
fn toggled(full: bool, windowed: Windowed, monitor: RECT) -> Windowed {
    if full {
        let (ex_style, style, place) = windowed;
        return (ex_style, style | wm::WS_VISIBLE, place);
    }
    let (ex_style, style, _) = windowed;
    let (w, h) = (monitor.right - monitor.left, monitor.bottom - monitor.top);
    let (ex_style, style, _) =
        full_screen(ex_style, style | wm::WS_VISIBLE, false, (w, h)).unwrap_or((ex_style, style, (0, 0, w, h)));
    (ex_style, style, (monitor.left, monitor.top, w, h))
}

/// A key that switches full screen: F11, or Enter with Alt.
fn switches_full_screen(message: u32, wparam: WPARAM, lparam: LPARAM) -> bool {
    const VK_RETURN: usize = 0x0D;
    const VK_F11: usize = 0x7A;
    let alt = (lparam >> 29) & 1 != 0;
    let repeated = (lparam >> 30) & 1 != 0;
    !repeated && ((message == wm::WM_KEYDOWN && wparam == VK_F11)
        || (message == wm::WM_SYSKEYDOWN && wparam == VK_RETURN && alt))
}

/// `VICTORY_FULLSCREEN=1`: the title's windows fill the whole primary monitor
/// with no frame, as a console fills its screen (the swap chain stretches
/// to the window).
fn full_screen_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_FULLSCREEN").is_some_and(|v| v == "1"))
}

/// Whether a window the title makes is its own top-level window, the kind that
/// goes full screen or onto the chosen monitor: not one with a parent or
/// owner, not a child, not a tool window.
fn placeable(ex_style: u32, style: u32, parented: bool) -> bool {
    !parented && style & wm::WS_CHILD == 0 && ex_style & wm::WS_EX_TOOLWINDOW == 0
}

/// The styles and place of a full-screen window on a screen of `screen`
/// pixels: a frameless popup over all of it. None for a window that keeps
/// its own (one with a parent or owner, a child, a tool window).
fn full_screen(ex_style: u32, style: u32, parented: bool, screen: (i32, i32)) -> Option<(u32, u32, (i32, i32, i32, i32))> {
    if !placeable(ex_style, style, parented) || screen.0 <= 0 || screen.1 <= 0 {
        return None;
    }
    let kept = wm::WS_VISIBLE | wm::WS_DISABLED | wm::WS_CLIPCHILDREN | wm::WS_CLIPSIBLINGS;
    let frame = wm::WS_EX_WINDOWEDGE | wm::WS_EX_CLIENTEDGE | wm::WS_EX_DLGMODALFRAME | wm::WS_EX_STATICEDGE;
    Some((ex_style & !frame, wm::WS_POPUP | (style & kept), (0, 0, screen.0, screen.1)))
}

/// A console title sizes its window to its render target (
/// the title: 3840x2160 on a Series X), which can exceed the host's screen.
/// A top-level window larger than the work area is shrunk to fit, keeping
/// its proportions, and centred; what it shows is scaled to it (the swap
/// chain stretches). `VICTORY_WINDOW_MAX=WIDTHxHEIGHT` applies a smaller
/// limit without changing the title's render targets.
fn fit(
    x: i32,
    y: i32,
    width: i32,
    height: i32,
    chosen: Option<RECT>,
) -> (i32, i32, i32, i32) {
    // A monitor named by `VICTORY_MONITOR` (its work area, `chosen`): the
    // title's own top-level window goes there, fitted and centred.
    if let Some(area) = chosen
        && let Some(placed) = on_monitor(&area, width, height, window_limit())
    {
        tracing::info!(width, height, to = ?placed, "title window put on the monitor VICTORY_MONITOR names");
        return placed;
    }
    let mut area = RECT::default();
    // SAFETY: `area` receives the primary monitor's work area.
    let known =
        unsafe { wm::SystemParametersInfoW(wm::SPI_GETWORKAREA, 0, (&raw mut area).cast(), 0) };
    let limit = window_limit();
    let fitted = fit_with_limit(&area, width, height, limit);
    if known == 0 || fitted == (width, height) {
        return (x, y, width, height);
    }
    tracing::info!(width, height, to = ?fitted, ?limit, "title window fitted to the screen");
    let (w, h) = fitted;
    let left = area.left + (area.right - area.left - w) / 2;
    let top = area.top + (area.bottom - area.top - h) / 2;
    (left, top, w, h)
}

/// Where a window of `width` x `height` goes on the work area `area` of the
/// chosen monitor: a size the title left to Windows (`CW_USEDEFAULT`) is
/// first made three quarters of the area (as the restore path does), then
/// the size is fitted to the area and to `limit` (`VICTORY_WINDOW_MAX`) and
/// centred. `None` for a window with no size of its own (a zero-sized
/// helper keeps what it asked for).
fn on_monitor(area: &RECT, width: i32, height: i32, limit: Option<(i32, i32)>) -> Option<(i32, i32, i32, i32)> {
    let (aw, ah) = (area.right - area.left, area.bottom - area.top);
    if aw <= 0 || ah <= 0 {
        return None;
    }
    let (width, height) = if width == wm::CW_USEDEFAULT { ((aw * 3 / 4).max(1), (ah * 3 / 4).max(1)) } else { (width, height) };
    if width <= 0 || height <= 0 {
        return None;
    }
    let (w, h) = fit_with_limit(area, width, height, limit);
    Some((area.left + (aw - w) / 2, area.top + (ah - h) / 2, w, h))
}

/// The monitor `VICTORY_MONITOR` names: its device name (`DISPLAY3` or
/// `\\.\DISPLAY3`), or its 1-based place in the order Windows lists the
/// monitors; its work area and its whole area. Unset or not found: none
/// (the primary monitor, as before). (Ours, 31790ae7.)
fn chosen_monitor() -> Option<(RECT, RECT)> {
    use windows_sys::Win32::Graphics::Gdi::{EnumDisplayMonitors, GetMonitorInfoW, HDC, HMONITOR, MONITORINFO, MONITORINFOEXW};
    let want = std::env::var("VICTORY_MONITOR").ok()?;
    let want = want.trim().trim_start_matches(r"\\.\").to_ascii_uppercase();
    if want.is_empty() {
        return None;
    }
    unsafe extern "system" fn each(monitor: HMONITOR, _: HDC, _: *mut RECT, data: LPARAM) -> windows_sys::core::BOOL {
        // SAFETY: `data` is the list below, alive for the enumeration.
        let list = unsafe { &mut *(data as *mut Vec<(String, RECT, RECT)>) };
        // SAFETY: a plain struct of numbers.
        let mut info: MONITORINFOEXW = unsafe { std::mem::zeroed() };
        info.monitorInfo.cbSize = size_of::<MONITORINFOEXW>() as u32;
        // SAFETY: a monitor handle from the enumeration and a sized struct.
        if unsafe { GetMonitorInfoW(monitor, (&raw mut info).cast::<MONITORINFO>()) } != 0 {
            let end = info.szDevice.iter().position(|&c| c == 0).unwrap_or(info.szDevice.len());
            list.push((String::from_utf16_lossy(&info.szDevice[..end]), info.monitorInfo.rcWork, info.monitorInfo.rcMonitor));
        }
        1
    }
    let mut list: Vec<(String, RECT, RECT)> = Vec::new();
    // SAFETY: the callback only touches `list`, which outlives the call.
    unsafe { EnumDisplayMonitors(std::ptr::null_mut(), std::ptr::null(), Some(each), (&raw mut list) as LPARAM) };
    let found = pick_monitor(&list, &want);
    if found.is_none() {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            let names: Vec<&str> = list.iter().map(|m| m.0.as_str()).collect();
            tracing::warn!(want, ?names, "VICTORY_MONITOR names no monitor: the primary one is used");
        });
    }
    found
}

/// The entry of `list` (device name, work area, whole area) that `want`
/// (upper case, without `\\.\`) names: a 1-based place, or a device name.
fn pick_monitor(list: &[(String, RECT, RECT)], want: &str) -> Option<(RECT, RECT)> {
    match want.parse::<usize>() {
        Ok(n) if n >= 1 => list.get(n - 1).map(|m| (m.1, m.2)),
        _ => list
            .iter()
            .find(|(name, _, _)| name.trim_start_matches(r"\\.\").eq_ignore_ascii_case(want))
            .map(|m| (m.1, m.2)),
    }
}

fn parse_window_limit(value: &str) -> Option<(i32, i32)> {
    let (width, height) = value.trim().split_once(['x', 'X'])?;
    let (width, height) = (width.trim().parse().ok()?, height.trim().parse().ok()?);
    (width > 0 && height > 0).then_some((width, height))
}

fn window_limit() -> Option<(i32, i32)> {
    static LIMIT: std::sync::OnceLock<Option<(i32, i32)>> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| {
        let value = std::env::var_os("VICTORY_WINDOW_MAX")?;
        let parsed = value.to_str().and_then(parse_window_limit);
        if parsed.is_none() {
            tracing::warn!(
                ?value,
                "ignoring invalid VICTORY_WINDOW_MAX; expected positive WIDTHxHEIGHT"
            );
        }
        parsed
    })
}

fn fit_with_limit(
    area: &RECT,
    width: i32,
    height: i32,
    limit: Option<(i32, i32)>,
) -> (i32, i32) {
    let Some((max_width, max_height)) = limit else {
        return fit_into(area, width, height);
    };
    let bounded = RECT {
        left: 0,
        top: 0,
        right: (area.right - area.left).min(max_width),
        bottom: (area.bottom - area.top).min(max_height),
    };
    fit_into(&bounded, width, height)
}

/// `width` x `height` scaled down to fit `area`, or unchanged if it fits
/// (or is `CW_USEDEFAULT`).
fn fit_into(area: &RECT, width: i32, height: i32) -> (i32, i32) {
    let (aw, ah) = (area.right - area.left, area.bottom - area.top);
    if width <= 0 || height <= 0 || aw <= 0 || ah <= 0 || (width <= aw && height <= ah) {
        return (width, height);
    }
    let scale = f64::min(
        f64::from(aw) / f64::from(width),
        f64::from(ah) / f64::from(height),
    );
    (
        (f64::from(width) * scale) as i32,
        (f64::from(height) * scale) as i32,
    )
}

/// Puts our procedure in front of the title's.
fn hook(hwnd: HWND, state: Option<WindowState>) {
    if hwnd.is_null() {
        return;
    }
    if let Some(state) = state {
        WINDOWS.lock().unwrap_or_else(|e| e.into_inner()).insert(hwnd as usize, state);
    }
    // SAFETY: a window this thread just created. The title's procedure is
    // stored before ours can run.
    unsafe {
        let title = wm::GetWindowLongPtrW(hwnd, wm::GWLP_WNDPROC);
        wm::SetPropW(hwnd, TITLE_PROC, title as *mut c_void);
        wm::SetWindowLongPtrW(hwnd, wm::GWLP_WNDPROC, window_proc as *const () as isize);
    }
    note_shown(hwnd);
}

/// The title's window: the first of its top-level windows (no owner) seen
/// visible. A close sent to it ends the title; one sent to another window
/// of the title's (a helper it shows too) closes only that window.
static TITLE_WINDOW: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Takes `hwnd` as [`TITLE_WINDOW`] if there is none yet and it is a
/// visible top-level window.
fn note_shown(hwnd: HWND) {
    use std::sync::atomic::Ordering;
    if TITLE_WINDOW.load(Ordering::Relaxed) != 0 {
        return;
    }
    // SAFETY: both queries borrow a window of this thread's.
    let top = unsafe { wm::IsWindowVisible(hwnd) != 0 && wm::GetAncestor(hwnd, wm::GA_ROOTOWNER) == hwnd };
    if top
        && TITLE_WINDOW
            .compare_exchange(0, hwnd as usize, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
    {
        super::gameinput::mouse::attach(hwnd);
    }
}

extern "system" fn window_proc(
    hwnd: HWND,
    message: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    // A user close (close button, Alt+F4, window or taskbar menu); a
    // console title never sees one, nor a WM_CLOSE from it.
    if message == wm::WM_SYSCOMMAND && wparam & 0xFFF0 == wm::SC_CLOSE as usize {
        note_close("SC_CLOSE");
        end_title("window closed", Box::new(|| {}));
        return 0;
    }
    // Automation and other applications can send WM_CLOSE directly. Only the
    // title's window ends the title; owned dialogs, hidden helper windows and
    // the title's other top-level windows handle their own close messages.
    note_shown(hwnd);
    let title_window = TITLE_WINDOW.load(std::sync::atomic::Ordering::Relaxed) == hwnd as usize;
    if message == wm::WM_CLOSE && title_window {
        note_close("WM_CLOSE to the title's window");
        end_title("window closed", Box::new(|| {}));
        return 0;
    }
    if title_window && switches_full_screen(message, wparam, lparam) {
        toggle_full_screen(hwnd);
        return 0;
    }
    if title_window && message == wm::WM_INPUT {
        super::gameinput::mouse::receive(hwnd, wparam, lparam);
    }
    if message == wm::WM_NCDESTROY {
        WINDOWS.lock().unwrap_or_else(|e| e.into_inner()).remove(&(hwnd as usize));
        if title_window {
            super::gameinput::mouse::detach(hwnd);
            // a window the title makes again is its window then
            TITLE_WINDOW.store(0, std::sync::atomic::Ordering::Relaxed);
        }
    }
    // Alt or F10 puts a window into menu mode: the system runs a modal menu
    // loop on the thread that owns the window, which is the title's. The
    // title then stops (no frames, no input) until a key ends the loop, and
    // nothing shows it, since the window has no menu bar;
    // the title aborts itself about half a minute later. A console has no window
    // menu. The mouse still opens it from the title bar.
    if message == wm::WM_SYSCOMMAND && wparam & 0xFFF0 == wm::SC_KEYMENU as usize {
        return 0;
    }
    // SAFETY: the procedure `hook` stored, called the way the system would.
    unsafe {
        let title = wm::GetPropW(hwnd, TITLE_PROC);
        if message == wm::WM_NCDESTROY {
            wm::RemovePropW(hwnd, TITLE_PROC);
        }
        let title = std::mem::transmute::<*mut c_void, wm::WNDPROC>(title);
        wm::CallWindowProcW(title, hwnd, message, wparam, lparam)
    }
}

/// Says how a close arrived: sent from another thread (and whether that
/// thread waits for the answer) or posted, and which process owns the
/// window in front. Test sessions were closed minutes in by a close no one
/// meant to send; this names the likely sender next time.
fn note_close(how: &str) {
    use windows_sys::Win32::System::Threading::GetCurrentProcessId;
    // SAFETY: plain queries about this thread's message and the desktop.
    let (flags, front, owner) = unsafe {
        let flags = wm::InSendMessageEx(std::ptr::null());
        let front = wm::GetForegroundWindow();
        let mut owner = 0u32;
        if !front.is_null() {
            wm::GetWindowThreadProcessId(front, &mut owner);
        }
        (flags, front, owner)
    };
    let sent = match flags & (wm::ISMEX_SEND | wm::ISMEX_NOTIFY | wm::ISMEX_CALLBACK) {
        0 => "posted, or sent by this thread",
        f if f & wm::ISMEX_NOTIFY != 0 => "sent by another thread, which does not wait",
        f if f & wm::ISMEX_CALLBACK != 0 => "sent by another thread, with a callback",
        _ => "sent by another thread, which waits",
    };
    // SAFETY: a plain query.
    let ours = unsafe { GetCurrentProcessId() };
    tracing::info!(
        how,
        sent,
        front = format_args!("{:#x}", front as usize),
        front_process = owner,
        this_process = ours,
        "close received"
    );
}

/// Suspends the title as the console does when it ends it, runs `then`
/// (once), then ends the process. Returns at once; later calls do nothing.
pub(crate) fn end_title(why: &'static str, then: Box<dyn FnOnce() + Send>) {
    static ENDING: AtomicBool = AtomicBool::new(false);
    if ENDING.swap(true, Ordering::SeqCst) {
        return;
    }
    let constrain = constrained().clone();
    let suspend = routines().clone();
    tracing::info!(
        constrained = constrain.len(),
        routines = suspend.len(),
        "{why}: suspending the title"
    );
    // Whoever comes first (the routines or the deadline) runs `then`; the
    // other waits for it, so the process never ends halfway through.
    let then = std::sync::Arc::new(Mutex::new(Some(then)));
    let finish = move |how: &str| {
        let mut then = then.lock().unwrap_or_else(|e| e.into_inner());
        tracing::info!("{how}; ending the title");
        if let Some(then) = then.take() {
            then();
        }
        end();
    };
    let late = finish.clone();
    // Not on the calling thread: titles hand the suspend to their window
    // thread and wait for it inside the routine.
    std::thread::spawn(move || {
        constrain_then_suspend(&constrain, &suspend);
        finish("title suspended");
    });
    std::thread::spawn(move || {
        std::thread::sleep(SUSPEND_DEADLINE);
        late("title did not finish suspending in time");
    });
}

/// Runs the constrained-change routines (constrained), then the suspend
/// routines (quiesced), as the console does when it suspends a title.
/// Grand Theft Auto VI's suspend routine waits for a state that only its
/// constrained routine brings about; called alone it never returns.
fn constrain_then_suspend(constrain: &[Registration], suspend: &[Registration]) {
    for &(_, routine, context) in constrain.iter().chain(suspend) {
        routine(1, context as *mut c_void);
    }
}

/// Set in the environment of a relaunched `victory`: an inherited handle
/// of the process it replaces.
const RELAUNCHED_FROM: &str = "VICTORY_RELAUNCHED_FROM";

/// Starts `victory` again with `args` (after the title has suspended); the
/// new process waits until this one has ended.
pub(crate) fn relaunch(args: Vec<std::ffi::OsString>) {
    use windows_sys::Win32::Foundation::{DUPLICATE_SAME_ACCESS, DuplicateHandle, HANDLE};
    let mut me: HANDLE = std::ptr::null_mut();
    // SAFETY: an inheritable duplicate of our own process handle.
    let duplicated = unsafe {
        DuplicateHandle(
            GetCurrentProcess(),
            GetCurrentProcess(),
            GetCurrentProcess(),
            &mut me,
            0,
            1,
            DUPLICATE_SAME_ACCESS,
        )
    } != 0;
    let spawned = std::env::current_exe().and_then(|exe| {
        let mut command = std::process::Command::new(exe);
        command.args(&args);
        if duplicated {
            command.env(RELAUNCHED_FROM, (me as usize).to_string());
        }
        command.spawn()
    });
    match spawned {
        Ok(child) => tracing::info!(pid = child.id(), ?args, "relaunched"),
        Err(e) => tracing::warn!(%e, "relaunch failed; the title just ends"),
    }
}

/// In a relaunched `victory`: waits (30 s at most) for the process it
/// replaces to be gone, so the two never share the title's files.
pub fn wait_for_previous() {
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
    use windows_sys::Win32::System::Threading::WaitForSingleObject;
    let Some(handle) = std::env::var(RELAUNCHED_FROM)
        .ok()
        .and_then(|h| h.parse::<usize>().ok())
    else {
        return;
    };
    // SAFETY: removing our own variable before any other thread reads it.
    unsafe { std::env::remove_var(RELAUNCHED_FROM) };
    let handle = handle as HANDLE;
    // SAFETY: the inherited process handle the previous process gave us.
    unsafe {
        WaitForSingleObject(handle, 30_000);
        CloseHandle(handle);
    }
}

fn end() {
    // Compiled host pipelines kept for the next run.
    #[cfg(windows)]
    victory_gpu::pipeline::save_pipeline_cache();
    // DLSS's features and NGX, before the device goes with the process.
    #[cfg(windows)]
    crate::modules::fidelityfx::shut_down();
    // SAFETY: ends this process, as the console ends a suspended title.
    unsafe { TerminateProcess(GetCurrentProcess(), 0) };
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "lifecycle",
        dlls: &[
            "api-ms-win-core-psm-appnotify-*",
            "ext-ms-win-rtcore-ntuser-window-*",
        ],
        exports: vec![
            hle_export!(RegisterAppStateChangeNotification),
            hle_export!(UnregisterAppStateChangeNotification),
            hle_export!(RegisterAppConstrainedChangeNotification),
            hle_export!(UnregisterAppConstrainedChangeNotification),
            hle_export!(CreateWindowExW),
            hle_export!(CreateWindowExA),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A window on a monitor left of and above the primary (negative origin),
    /// whose size the title left to Windows, with a 1280x720 cap: a default
    /// size resolved on that monitor first, then capped and centred; that is
    /// also the rectangle the first exit from full screen restores.
    #[test]
    fn a_default_sized_window_on_the_chosen_monitor_is_resolved_then_capped() {
        let area = RECT { left: -2560, top: -200, right: 0, bottom: 1240 };
        let default = wm::CW_USEDEFAULT;
        let placed = on_monitor(&area, default, default, Some((1280, 720))).unwrap();
        assert_eq!(placed, (-1920, 160, 1280, 720));
        // Without the cap: three quarters of the work area, centred.
        assert_eq!(on_monitor(&area, default, default, None), Some((-2240, -20, 1920, 1080)));
        // A size of its own, larger than the cap: fitted, centred.
        assert_eq!(on_monitor(&area, 2560, 1440, Some((1280, 720))), Some((-1920, 160, 1280, 720)));
        // A zero-sized helper keeps what it asked for.
        assert_eq!(on_monitor(&area, 0, 0, None), None);
        // The first exit from full screen restores that rectangle.
        let windowed = (wm::WS_EX_WINDOWEDGE, wm::WS_OVERLAPPEDWINDOW, placed);
        assert_eq!(toggled(true, windowed, area).2, placed);
    }

    /// Only the title's own top-level windows are moved or made full screen.
    #[test]
    fn tool_child_and_owned_windows_are_not_placed() {
        assert!(placeable(wm::WS_EX_APPWINDOW, wm::WS_OVERLAPPEDWINDOW, false));
        assert!(!placeable(wm::WS_EX_TOOLWINDOW, wm::WS_POPUP, false));
        assert!(!placeable(0, wm::WS_CHILD, false));
        assert!(!placeable(0, wm::WS_OVERLAPPEDWINDOW, true));

        let window = (25, 60, 0, 0);
        let parent = std::ptr::null_mut();
        let (placed, state) = place(0, wm::WS_POPUP, parent, window.0, window.1, window.2, window.3);
        assert_eq!(placed, (0, wm::WS_POPUP, window));
        assert!(state.is_none(), "the zero-sized helper is not eligible for full screen");

        let (placed, state) = place(wm::WS_EX_TOOLWINDOW, wm::WS_POPUP, parent, 25, 60, 3000, 2000);
        assert_eq!(placed, (wm::WS_EX_TOOLWINDOW, wm::WS_POPUP, (25, 60, 3000, 2000)));
        assert!(state.is_none(), "the tool window keeps its placement");
    }

    /// A default-sized window (`CW_USEDEFAULT` width; Windows ignores the height, here 0) is
    /// a normal window: placed, and kept for full screen and its toggle; a width of 0 is not.
    #[test]
    fn a_default_width_with_zero_height_is_not_a_zero_sized_helper() {
        let default = wm::CW_USEDEFAULT;
        let style = wm::WS_OVERLAPPEDWINDOW;
        assert!(!zero_sized(style, default, 0));
        assert!(!zero_sized(style, default, default));
        assert!(zero_sized(style, 0, 0));
        assert!(zero_sized(style, 0, 600));
        assert!(zero_sized(style, 800, 0));
        let (_, state) = place(
            wm::WS_EX_APPWINDOW, style, std::ptr::null_mut(), default, 0, default, 0,
        );
        assert!(state.is_some(), "the default-sized window keeps its full-screen state");
    }

    #[test]
    fn a_popup_with_default_width_keeps_its_native_zero_size() {
        let default = wm::CW_USEDEFAULT;
        let style = wm::WS_POPUP;
        assert!(zero_sized(style, default, 0));
        assert!(zero_sized(style, default, 600));
        assert!(zero_sized(wm::WS_CHILD, default, 600));
        assert!(!zero_sized(style, 800, 600));
        let (placed, state) = place(0, style, std::ptr::null_mut(), 10, 20, default, 0);
        assert_eq!(placed, (0, style, (10, 20, default, 0)));
        assert!(state.is_none());
    }

    /// `VICTORY_MONITOR`: a device name with or without its prefix, any
    /// case, or a 1-based place; anything else names none.
    #[test]
    fn a_monitor_is_picked_by_name_or_place() {
        let rect = |l: i32, w: i32| RECT { left: l, top: 0, right: l + w, bottom: 1440 };
        let list = vec![
            (r"\\.\DISPLAY1".to_owned(), rect(0, 3440), rect(0, 3440)),
            (r"\\.\DISPLAY3".to_owned(), rect(3440, 2560), rect(3440, 2560)),
        ];
        assert_eq!(pick_monitor(&list, "DISPLAY3").map(|m| m.1.left), Some(3440));
        assert_eq!(pick_monitor(&list, "display1").map(|m| m.1.left), Some(0));
        assert_eq!(pick_monitor(&list, "2").map(|m| m.1.left), Some(3440));
        assert!(pick_monitor(&list, "3").is_none());
        assert!(pick_monitor(&list, "0").is_none());
        assert!(pick_monitor(&list, "DISPLAY2").is_none());
    }

    extern "system" fn noop(_: u8, _: *mut c_void) {}

    #[test]
    fn full_screen_switches_both_ways_on_the_window_s_monitor() {
        let windowed = (wm::WS_EX_WINDOWEDGE, wm::WS_OVERLAPPEDWINDOW, (100, 50, 1600, 900));
        let monitor = RECT { left: 3840, top: 0, right: 3840 + 2560, bottom: 1440 };
        // to full screen: a frameless popup over the monitor the window is on
        let (ex, style, place) = toggled(false, windowed, monitor);
        assert_eq!(place, (3840, 0, 2560, 1440));
        assert_eq!(style, wm::WS_POPUP | wm::WS_VISIBLE);
        assert_eq!(ex & wm::WS_EX_WINDOWEDGE, 0);
        // and back: the window it was made as, shown
        assert_eq!(toggled(true, windowed, monitor), (wm::WS_EX_WINDOWEDGE, wm::WS_OVERLAPPEDWINDOW | wm::WS_VISIBLE, (100, 50, 1600, 900)));
        // the keys: F11, and Enter only with Alt
        let alt = 1 << 29;
        assert!(switches_full_screen(wm::WM_KEYDOWN, 0x7A, 0));
        assert!(switches_full_screen(wm::WM_SYSKEYDOWN, 0x0D, alt));
        assert!(!switches_full_screen(wm::WM_KEYDOWN, 0x0D, 0));
        assert!(!switches_full_screen(wm::WM_SYSKEYDOWN, 0x0D, 0));
        let repeat = 1 << 30;
        assert!(!switches_full_screen(wm::WM_KEYDOWN, 0x7A, repeat));
        assert!(!switches_full_screen(wm::WM_SYSKEYDOWN, 0x0D, alt | repeat));
    }

    #[test]
    fn restores_default_window_geometry_on_a_monitor_with_negative_coordinates() {
        let area = RECT { left: -1920, top: 0, right: 0, bottom: 1040 };
        let default = wm::CW_USEDEFAULT;
        assert_eq!(restore_rect((default, 5, default, 0), area), (-1680, 130, 1440, 780));
        let explicit = (-1800, -20, 900, 600);
        assert_eq!(restore_rect(explicit, area), explicit);
    }

    #[test]
    fn a_full_screen_window_is_a_frameless_popup_over_the_screen() {
        let framed = wm::WS_OVERLAPPEDWINDOW | wm::WS_VISIBLE;
        let (ex, style, place) =
            full_screen(wm::WS_EX_APPWINDOW | wm::WS_EX_WINDOWEDGE, framed, false, (3840, 2160)).unwrap();
        assert_eq!(place, (0, 0, 3840, 2160));
        assert_eq!(style, wm::WS_POPUP | wm::WS_VISIBLE);
        assert_eq!(ex, wm::WS_EX_APPWINDOW);
        // windows that keep their own place
        assert!(full_screen(0, framed, true, (3840, 2160)).is_none());
        assert!(full_screen(0, wm::WS_CHILD | wm::WS_VISIBLE, false, (3840, 2160)).is_none());
        assert!(full_screen(wm::WS_EX_TOOLWINDOW, wm::WS_POPUP, false, (3840, 2160)).is_none());
    }

    #[test]
    fn registrations_come_and_go() {
        let (mut a, mut b) = (0, 0);
        assert_eq!(
            RegisterAppStateChangeNotification(Some(noop), std::ptr::null_mut(), &mut a),
            0
        );
        assert_eq!(
            RegisterAppStateChangeNotification(Some(noop), std::ptr::null_mut(), &mut b),
            0
        );
        UnregisterAppStateChangeNotification(a);
        let left: Vec<_> = routines().iter().map(|r| r.0).collect();
        assert_eq!(left, [b]);
    }

    #[test]
    fn constrained_registrations_come_and_go() {
        let (mut a, mut b) = (0, 0);
        assert_eq!(
            RegisterAppConstrainedChangeNotification(Some(noop), std::ptr::null_mut(), &mut a),
            0
        );
        assert_eq!(
            RegisterAppConstrainedChangeNotification(Some(noop), std::ptr::null_mut(), &mut b),
            0
        );
        assert_eq!(
            RegisterAppConstrainedChangeNotification(None, std::ptr::null_mut(), &mut a),
            ERROR_INVALID_PARAMETER
        );
        UnregisterAppConstrainedChangeNotification(a);
        let left: Vec<_> = constrained().iter().map(|r| r.0).collect();
        assert_eq!(left, [b]);
    }

    /// What `recording_routine` saw: (context, flag) per call, in order.
    static CALLS: Mutex<Vec<(usize, u8)>> = Mutex::new(Vec::new());

    extern "system" fn recording_routine(flag: u8, context: *mut c_void) {
        CALLS.lock().unwrap().push((context as usize, flag));
    }

    #[test]
    fn constrained_routines_run_before_suspend_routines() {
        let r: Routine = recording_routine;
        constrain_then_suspend(&[(1, r, 10), (2, r, 11)], &[(3, r, 20), (4, r, 21)]);
        assert_eq!(*CALLS.lock().unwrap(), [(10, 1), (11, 1), (20, 1), (21, 1)]);
    }

    #[test]
    fn oversized_windows_shrink_to_the_screen() {
        let area = RECT {
            left: 0,
            top: 0,
            right: 1920,
            bottom: 1040,
        };
        assert_eq!(fit_into(&area, 3840, 2160), (1848, 1040));
        assert_eq!(fit_into(&area, 1280, 720), (1280, 720));
        assert_eq!(
            fit_into(&area, wm::CW_USEDEFAULT, 0),
            (wm::CW_USEDEFAULT, 0)
        );
    }

    #[test]
    fn configured_window_limit_fits_4k_into_1080p() {
        let area = RECT {
            left: 100,
            top: 50,
            right: 2660,
            bottom: 1490,
        };
        assert_eq!(fit_with_limit(&area, 3840, 2160, Some((1920, 1080))), (1920, 1080));
        assert_eq!(fit_with_limit(&area, 640, 480, Some((1920, 1080))), (640, 480));
        assert_eq!(fit_with_limit(&area, 3840, 2160, None), (2560, 1440));
    }

    #[test]
    fn configured_window_limit_still_respects_the_work_area() {
        let area = RECT {
            left: 0,
            top: 0,
            right: 1920,
            bottom: 1040,
        };
        assert_eq!(fit_with_limit(&area, 3840, 2160, Some((2560, 1440))), (1848, 1040));
        assert_eq!(
            fit_with_limit(&area, wm::CW_USEDEFAULT, 0, Some((1920, 1080))),
            (wm::CW_USEDEFAULT, 0)
        );
    }

    #[test]
    fn configured_window_limit_preserves_non_widescreen_proportions() {
        let area = RECT {
            left: 0,
            top: 0,
            right: 2560,
            bottom: 1440,
        };
        assert_eq!(fit_with_limit(&area, 1600, 1200, Some((1920, 1080))), (1440, 1080));
    }

    #[test]
    fn window_limit_requires_two_positive_dimensions() {
        assert_eq!(parse_window_limit("1920x1080"), Some((1920, 1080)));
        assert_eq!(parse_window_limit(" 1920 X 1080 "), Some((1920, 1080)));
        for bad in [
            "",
            "1920",
            "0x1080",
            "1920x0",
            "-1x1080",
            "1920x-1",
            "1920x1080x2",
            "2147483648x1080",
        ] {
            assert_eq!(parse_window_limit(bad), None, "{bad}");
        }
    }

    #[test]
    fn hooked_windows_still_reach_the_title_procedure() {
        let null = std::ptr::null_mut();
        let hwnd = CreateWindowExW(
            0,
            w!("STATIC"),
            w!("before"),
            0,
            0,
            0,
            10,
            10,
            null,
            null,
            null,
            std::ptr::null(),
        );
        assert!(!hwnd.is_null());
        let mut text = [0u16; 16];
        // SAFETY: our window; the buffer holds 16 characters.
        unsafe {
            wm::SetWindowTextW(hwnd, w!("after"));
            let n = wm::GetWindowTextW(hwnd, text.as_mut_ptr(), 16);
            assert_eq!(String::from_utf16_lossy(&text[..n as usize]), "after");
            assert!(!wm::GetPropW(hwnd, TITLE_PROC).is_null());
            wm::DestroyWindow(hwnd);
        }
    }

    #[test]
    fn direct_close_suspends_and_exits() {
        const CHILD: &str = "VICTORY_TEST_DIRECT_CLOSE_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "modules::lifecycle::tests::direct_close_suspends_and_exits",
                    "--nocapture",
                ])
                .env(CHILD, "1")
                .output()
                .unwrap();
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(output.status.success(), "close test child failed: {stderr}");
            assert!(stderr.contains("suspend routine ran"), "{stderr}");
            return;
        }

        extern "system" fn record_suspend(_: u8, _: *mut c_void) {
            eprintln!("suspend routine ran");
        }

        let null = std::ptr::null_mut();
        let create = |owner| {
            CreateWindowExW(
                wm::WS_EX_NOACTIVATE | wm::WS_EX_TOOLWINDOW,
                w!("STATIC"),
                w!("VIctory close test"),
                wm::WS_VISIBLE | wm::WS_POPUP,
                -10_000,
                -10_000,
                10,
                10,
                owner,
                null,
                null,
                std::ptr::null(),
            )
        };
        let title = create(null);
        assert!(!title.is_null());
        let dialog = create(title);
        assert!(!dialog.is_null());
        // a second visible top-level window of the title's, a helper
        let helper = create(null);
        assert!(!helper.is_null());
        // SAFETY: the windows belong to this test thread. Closing the dialog
        // or the helper goes to its original procedure and leaves the title
        // window alive.
        unsafe {
            wm::SendMessageW(dialog, wm::WM_CLOSE, 0, 0);
            assert_eq!(wm::IsWindow(dialog), 0);
            wm::SendMessageW(helper, wm::WM_CLOSE, 0, 0);
            assert_eq!(wm::IsWindow(helper), 0);
            assert_ne!(wm::IsWindow(title), 0);
        }
        let mut registration = 0;
        assert_eq!(
            RegisterAppStateChangeNotification(Some(record_suspend), null, &mut registration),
            0
        );
        // SAFETY: this thread owns the title window.
        unsafe { wm::SendMessageW(title, wm::WM_CLOSE, 0, 0) };
        std::thread::sleep(SUSPEND_DEADLINE + Duration::from_secs(1));
        panic!("direct close did not exit the title process");
    }

    thread_local! {
        /// The system commands `recording` received, low four bits cleared.
        static COMMANDS: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
    }

    /// Stands in for a title's procedure: notes system commands and acts
    /// on none of them (no menu loop, no minimising).
    extern "system" fn recording(
        hwnd: HWND,
        message: u32,
        wparam: WPARAM,
        lparam: LPARAM,
    ) -> LRESULT {
        if message == wm::WM_SYSCOMMAND {
            COMMANDS.with(|c| c.borrow_mut().push(wparam & 0xFFF0));
            return 0;
        }
        // SAFETY: the system's default procedure, called as the system would.
        unsafe { wm::DefWindowProcW(hwnd, message, wparam, lparam) }
    }

    #[test]
    fn the_keyboard_menu_never_reaches_the_title() {
        let null = std::ptr::null_mut();
        let hwnd = CreateWindowExW(
            0,
            w!("STATIC"),
            w!("menu"),
            0,
            0,
            0,
            10,
            10,
            null,
            null,
            null,
            std::ptr::null(),
        );
        assert!(!hwnd.is_null());
        // SAFETY: our window, owned by this thread, so each message is
        // handled before SendMessageW returns. `recording` stands in for the
        // title's procedure and the original is put back before the window
        // is destroyed.
        unsafe {
            let title = wm::GetPropW(hwnd, TITLE_PROC);
            wm::SetPropW(hwnd, TITLE_PROC, recording as *const () as *mut c_void);
            let command = |c: u32, key: u8| {
                wm::SendMessageW(hwnd, wm::WM_SYSCOMMAND, c as usize, LPARAM::from(key))
            };
            command(wm::SC_KEYMENU, 0); // Alt or F10
            command(wm::SC_KEYMENU, b' '); // Alt+Space
            command(wm::SC_KEYMENU | 1, 0); // the low bits are the system's
            command(wm::SC_MINIMIZE, 0);
            wm::SetPropW(hwnd, TITLE_PROC, title);
            wm::DestroyWindow(hwnd);
        }
        let seen = COMMANDS.with(|c| c.borrow().clone());
        assert_eq!(seen, [wm::SC_MINIMIZE as usize]);
    }
}
