//! DualSense, DualSense Edge and USB DualShock 4 pads read straight from HID:
//! they are not XInput devices, so without a remapper (Steam Input,
//! DS4Windows) XInput never lists them. Reports can be USB or Bluetooth. A scan
//! thread looks for new pads every [`SCAN`]; each pad has a reader thread
//! that keeps its latest state until the pad goes away. By default XInput
//! has priority over HID, since a remapper can present the same pad twice.

use super::{
    A, B, DOWN, LEFT, LEFT_SHOULDER, LEFT_STICK, MENU, PadState, RIGHT, RIGHT_SHOULDER,
    RIGHT_STICK, UP, VIEW, X, Y,
};

#[cfg(windows)]
const SONY: u16 = 0x054C;
/// DualSense, DualSense Edge.
#[cfg(windows)]
const PRODUCTS: [u16; 2] = [0x0CE6, 0x0DF2];
#[cfg(windows)]
const DUALSHOCK4_PRODUCTS: [u16; 2] = [0x05C4, 0x09CC];

#[derive(Clone, Copy)]
enum ReportKind {
    DualSenseUsb,
    DualSenseBluetooth,
    DualShock4Usb,
}

/// The D-pad hat (low nibble of the first button byte): 0 up, clockwise in
/// eighths, 8 and above centred.
fn hat(value: u8) -> u32 {
    match value & 0xF {
        0 => UP,
        1 => UP | RIGHT,
        2 => RIGHT,
        3 => RIGHT | DOWN,
        4 => DOWN,
        5 => DOWN | LEFT,
        6 => LEFT,
        7 => LEFT | UP,
        _ => 0,
    }
}

/// Face buttons in the first button byte, the rest in the second and third
/// (the PS button and the mute button have no console pad button).
const BUTTONS: [(usize, u8, u32); 11] = [
    (0, 0x10, X), // square
    (0, 0x20, A), // cross
    (0, 0x40, B), // circle
    (0, 0x80, Y), // triangle
    (1, 0x01, LEFT_SHOULDER),
    (1, 0x02, RIGHT_SHOULDER),
    (1, 0x10, VIEW), // create
    (1, 0x20, MENU), // options
    (1, 0x40, LEFT_STICK),
    (1, 0x80, RIGHT_STICK),
    (2, 0x02, VIEW), // touchpad click
];

/// The pad in one input report, `None` for a report of another kind.
fn parse(report: &[u8], kind: ReportKind) -> Option<PadState> {
    // Offsets of the sticks (LX LY RX RY), the triggers (L2 R2) and the
    // three button bytes.
    let (sticks, triggers, buttons) = match (report.first()?, kind) {
        (0x01, ReportKind::DualSenseUsb) => (1, 5, 8),
        (0x31, ReportKind::DualSenseBluetooth) => (2, 6, 9),
        (0x01, ReportKind::DualSenseBluetooth | ReportKind::DualShock4Usb) => (1, 8, 5),
        _ => return None,
    };
    if report.len() < sticks + 4 || report.len() < triggers + 2 || report.len() < buttons + 3 {
        return None;
    }
    let bytes = &report[buttons..buttons + 3];
    // 0 to 255 with 128 centred; down is positive on the pad, up here.
    let axis = |v: u8| ((f32::from(v) - 127.5) / 127.5).clamp(-1.0, 1.0);
    Some(PadState {
        buttons: BUTTONS
            .iter()
            .filter(|&&(byte, mask, _)| bytes[byte] & mask != 0)
            .fold(hat(bytes[0]), |all, &(_, _, bit)| all | bit),
        left_trigger: f32::from(report[triggers]) / 255.0,
        right_trigger: f32::from(report[triggers + 1]) / 255.0,
        left_x: axis(report[sticks]),
        left_y: -axis(report[sticks + 1]),
        right_x: axis(report[sticks + 2]),
        right_y: -axis(report[sticks + 3]),
    })
}

/// The Bluetooth address in a pairing feature report (0x09, the address at
/// 1..7 on both transports), `None` when it is too short or all zeros.
fn mac(pairing: &[u8]) -> Option<[u8; 6]> {
    let mac: [u8; 6] = pairing.get(1..7)?.try_into().ok()?;
    (mac != [0; 6]).then_some(mac)
}

#[cfg(windows)]
pub(super) use host::pads;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Mode {
    Off,
    Auto,
    Both,
}

#[cfg(windows)]
pub(super) fn mode() -> Mode {
    static MODE: std::sync::LazyLock<Mode> =
        std::sync::LazyLock::new(|| match std::env::var("VICTORY_DUALSENSE").as_deref() {
            Ok("0") => Mode::Off,
            Ok("1") => Mode::Both,
            _ => Mode::Auto,
        });
    *MODE
}

/// XInput does not identify the corresponding physical HID device. Prefer
/// it in automatic mode. If both sources are requested, retain the slot of
/// every connected HID pad unless that slot has been claimed by XInput.
pub(super) fn slots(
    mode: Mode,
    mut xinput: [Option<PadState>; super::PADS],
    assigned: &mut [Option<u64>; super::PADS],
    hid: impl FnOnce() -> Vec<(u64, PadState)>,
) -> [Option<PadState>; super::PADS] {
    if mode == Mode::Off || mode == Mode::Auto && xinput.iter().any(Option::is_some) {
        assigned.fill(None);
        return xinput;
    }
    let pads = hid();
    for (slot, owner) in xinput.iter_mut().zip(assigned.iter_mut()) {
        if slot.is_none()
            && let Some((_, state)) = pads.iter().find(|(id, _)| Some(*id) == *owner)
        {
            *slot = Some(*state);
        } else {
            *owner = None;
        }
    }
    for (id, state) in pads {
        if assigned.contains(&Some(id)) {
            continue;
        }
        let Some(slot) = xinput.iter().position(Option::is_none) else {
            break;
        };
        xinput[slot] = Some(state);
        assigned[slot] = Some(id);
    }
    xinput
}

#[cfg(windows)]
mod host {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{LazyLock, Mutex, PoisonError};
    use std::time::Duration;

    use windows_sys::Win32::Devices::DeviceAndDriverInstallation::{
        DIGCF_DEVICEINTERFACE, DIGCF_PRESENT, SP_DEVICE_INTERFACE_DATA,
        SP_DEVICE_INTERFACE_DETAIL_DATA_W, SetupDiDestroyDeviceInfoList,
        SetupDiEnumDeviceInterfaces, SetupDiGetClassDevsW, SetupDiGetDeviceInterfaceDetailW,
    };
    use windows_sys::Win32::Devices::HumanInterfaceDevice::{
        HIDD_ATTRIBUTES, HIDP_CAPS, HIDP_STATUS_SUCCESS, HidD_FreePreparsedData,
        HidD_GetAttributes, HidD_GetFeature, HidD_GetHidGuid, HidD_GetPreparsedData, HidP_GetCaps,
    };
    use windows_sys::Win32::Foundation::{
        CloseHandle, ERROR_IO_PENDING, GENERIC_READ, GENERIC_WRITE, GetLastError, HANDLE,
        INVALID_HANDLE_VALUE, WAIT_TIMEOUT,
    };
    use windows_sys::Win32::Storage::FileSystem::{
        CreateFileW, FILE_FLAG_OVERLAPPED, FILE_SHARE_READ, FILE_SHARE_WRITE, OPEN_EXISTING,
        ReadFile,
    };
    use windows_sys::Win32::System::IO::{
        CancelIoEx, GetOverlappedResult, GetOverlappedResultEx, OVERLAPPED,
    };
    use windows_sys::Win32::System::Threading::CreateEventW;

    use super::{DUALSHOCK4_PRODUCTS, PRODUCTS, PadState, ReportKind, SONY, mac, parse};

    /// How often to look for newly connected pads.
    const SCAN: Duration = Duration::from_secs(2);

    /// A pad in full-report mode sends reports all the time (hundreds a
    /// second); this long without one and we ask it whether it is still
    /// there. Over Bluetooth Windows can keep the device after the pad is
    /// switched off or out of range, and a read then waits for ever.
    const QUIET_MS: u32 = 500;

    struct Pad {
        /// This connection's: a pad that reconnects gets a new one.
        id: u64,
        path: Vec<u16>,
        /// The pad's Bluetooth address, the same over USB and Bluetooth:
        /// one pad seen through both is opened once.
        mac: Option<[u8; 6]>,
        /// `None` until the first report, and while the pad is silent.
        state: Option<PadState>,
    }

    /// The open pads, in the order they were found.
    static PADS: Mutex<Vec<Pad>> = Mutex::new(Vec::new());

    fn lock() -> std::sync::MutexGuard<'static, Vec<Pad>> {
        PADS.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Paths found to be a pad already open over its other transport, with
    /// its address: not opened again on every scan (two feature reports
    /// and a log line each time) while that pad stays open.
    static TWINS: Mutex<Vec<(Vec<u16>, [u8; 6])>> = Mutex::new(Vec::new());

    /// The open pads' latest states with their connection ids, oldest
    /// connection first, without the pads that are silent or yet to report;
    /// the first call starts looking for them.
    pub(in super::super) fn pads() -> Vec<(u64, PadState)> {
        static STARTED: LazyLock<()> = LazyLock::new(|| {
            std::thread::spawn(|| {
                loop {
                    scan();
                    std::thread::sleep(SCAN);
                }
            });
        });
        LazyLock::force(&STARTED);
        lock()
            .iter()
            .filter_map(|pad| Some((pad.id, pad.state?)))
            .collect()
    }

    /// Windows sizes each report buffer for its top-level HID collection.
    /// The calibration and pairing reports use the same feature buffer.
    struct Reports {
        input: Vec<u8>,
        feature: Vec<u8>,
    }

    impl Reports {
        fn for_device(handle: HANDLE, kind: ReportKind) -> Option<Self> {
            let mut data = 0;
            // SAFETY: the device is open and the output is writable.
            if !unsafe { HidD_GetPreparsedData(handle, &mut data) } {
                return None;
            }
            let mut caps = HIDP_CAPS::default();
            // SAFETY: `data` was obtained above and is released after its last use.
            let status = unsafe { HidP_GetCaps(data, &mut caps) };
            unsafe { HidD_FreePreparsedData(data) };
            if status != HIDP_STATUS_SUCCESS {
                return None;
            }
            Self::from_caps(&caps, kind)
        }

        fn from_caps(caps: &HIDP_CAPS, kind: ReportKind) -> Option<Self> {
            let dualshock4 = matches!(kind, ReportKind::DualShock4Usb);
            if caps.InputReportByteLength == 0
                || dualshock4 && (caps.UsagePage != 1 || caps.Usage != 5)
                || !dualshock4 && caps.FeatureReportByteLength == 0
            {
                return None;
            }
            Some(Self {
                input: vec![0; usize::from(caps.InputReportByteLength)],
                feature: vec![0; usize::from(caps.FeatureReportByteLength)],
            })
        }
    }

    #[test]
    fn dualshock4_can_open_an_input_only_gamepad_collection() {
        let mut caps = HIDP_CAPS {
            UsagePage: 1,
            Usage: 5,
            InputReportByteLength: 64,
            ..Default::default()
        };
        let reports = Reports::from_caps(&caps, ReportKind::DualShock4Usb).unwrap();
        assert_eq!(reports.input.len(), 64);
        assert!(reports.feature.is_empty());
        assert!(Reports::from_caps(&caps, ReportKind::DualSenseUsb).is_none());

        caps.Usage = 6;
        assert!(Reports::from_caps(&caps, ReportKind::DualShock4Usb).is_none());
    }

    /// Feature report `id` into `buffer` (its first byte), `false` when the
    /// pad did not answer. A round trip to the pad, so over Bluetooth it also
    /// tells whether the pad is still there.
    fn feature(handle: HANDLE, id: u8, buffer: &mut [u8]) -> bool {
        buffer.fill(0);
        buffer[0] = id;
        // SAFETY: the buffer is as long as we say.
        unsafe { HidD_GetFeature(handle, buffer.as_mut_ptr().cast(), buffer.len() as u32) }
    }

    /// Asks for the calibration report: it switches a pad on Bluetooth to
    /// full reports (0x31), again after it reconnected; over USB it changes
    /// nothing. `false` when the pad did not answer.
    fn full_reports(handle: HANDLE, buffer: &mut [u8]) -> bool {
        feature(handle, 0x05, buffer)
    }

    /// Opens the pads not open yet, each with a reader thread.
    fn scan() {
        let paths = hid_paths();
        {
            let pads = lock();
            // A twin whose pad went away is opened again: over USB, say,
            // after the Bluetooth link dropped with the cable still in.
            TWINS
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .retain(|(path, mac)| {
                    paths.contains(path) && pads.iter().any(|pad| pad.mac == Some(*mac))
                });
        }
        for path in paths {
            // Cheap filter before opening anything: Sony's vendor id is in
            // the path, as `vid_054c` (USB) or `vid&0002054c` (Bluetooth).
            let lower = String::from_utf16_lossy(&path).to_ascii_lowercase();
            if !lower.contains("054c") {
                continue;
            }
            if lock().iter().any(|pad| pad.path == path)
                || TWINS
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .iter()
                    .any(|(twin, _)| *twin == path)
            {
                continue;
            }
            let bluetooth = lower.contains("{00001124-");
            let Some((handle, mut reports, kind)) = open(&path, bluetooth) else {
                continue;
            };
            // A paired pad that is switched off can still be listed over
            // Bluetooth: it does not answer.
            if !matches!(kind, ReportKind::DualShock4Usb)
                && !full_reports(handle, &mut reports.feature)
            {
                tracing::debug!(path = %lower, "DualSense listed but not answering");
                // SAFETY: just opened, nothing else has it.
                unsafe { CloseHandle(handle) };
                continue;
            }
            let mac = if !matches!(kind, ReportKind::DualShock4Usb)
                && feature(handle, 0x09, &mut reports.feature)
            {
                mac(&reports.feature)
            } else {
                None
            };
            static NEXT_ID: AtomicU64 = AtomicU64::new(0);
            let id = {
                let mut pads = lock();
                let twin = mac.is_some() && pads.iter().any(|pad| pad.mac == mac);
                (!twin).then(|| {
                    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
                    pads.push(Pad {
                        id,
                        path: path.clone(),
                        mac,
                        state: None,
                    });
                    id
                })
            };
            let Some(id) = id else {
                tracing::info!(path = %lower, bluetooth, "DualSense already open over the other transport");
                if let Some(mac) = mac {
                    TWINS
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner)
                        .push((path, mac));
                }
                // SAFETY: just opened, nothing else has it.
                unsafe { CloseHandle(handle) };
                continue;
            };
            tracing::info!(path = %lower, bluetooth, "PlayStation gamepad connected");
            let raw = handle as usize;
            if let Err(error) = std::thread::Builder::new()
                .name("PlayStation gamepad reader".into())
                .spawn(move || read(raw as HANDLE, id, kind, reports))
            {
                lock().retain(|pad| pad.id != id);
                // SAFETY: just opened, and no reader has it.
                unsafe { CloseHandle(handle) };
                tracing::warn!(%error, "could not start a PlayStation gamepad reader");
            }
        }
    }

    /// Reads reports until the pad goes away, then forgets it.
    fn read(handle: HANDLE, id: u64, kind: ReportKind, mut reports: Reports) {
        let bluetooth = matches!(kind, ReportKind::DualSenseBluetooth);
        let set = |state: Option<PadState>| {
            if let Some(pad) = lock().iter_mut().find(|pad| pad.id == id) {
                pad.state = state;
            }
        };
        // SAFETY: an unnamed manual-reset event; no security attributes.
        let event = unsafe { CreateEventW(std::ptr::null(), 1, 0, std::ptr::null()) };
        let why = 'pad: loop {
            if event.is_null() {
                break 'pad "no event";
            }
            // SAFETY: plain data; only the event is set.
            let mut overlapped: OVERLAPPED = unsafe { std::mem::zeroed() };
            overlapped.hEvent = event;
            let mut got = 0;
            // SAFETY: the buffer and `overlapped` outlive the read: every way
            // out below waits for it to end first.
            let started = unsafe {
                ReadFile(
                    handle,
                    reports.input.as_mut_ptr(),
                    reports.input.len() as u32,
                    std::ptr::null_mut(),
                    &mut overlapped,
                )
            };
            // SAFETY: plain query.
            if started == 0 && unsafe { GetLastError() } != ERROR_IO_PENDING {
                break 'pad "read failed";
            }
            loop {
                // SAFETY: the read started above.
                let done =
                    unsafe { GetOverlappedResultEx(handle, &overlapped, &mut got, QUIET_MS, 0) };
                if done != 0 {
                    if got == 0 {
                        break 'pad "read nothing";
                    }
                    break;
                }
                // SAFETY: plain query.
                if unsafe { GetLastError() } != WAIT_TIMEOUT {
                    break 'pad "read failed";
                }
                // Silent: whatever was held is not held any more, and the
                // pad may be gone without the read failing.
                set(None);
                if !matches!(kind, ReportKind::DualShock4Usb)
                    && !full_reports(handle, &mut reports.feature)
                {
                    // SAFETY: cancel the read and wait for it to end, so
                    // nothing writes into the input buffer once we return.
                    unsafe {
                        CancelIoEx(handle, &overlapped);
                        GetOverlappedResult(handle, &overlapped, &mut got, 1);
                    }
                    break 'pad "not answering";
                }
            }
            if let Some(state) = parse(&reports.input[..got as usize], kind) {
                set(Some(state));
            }
        };
        tracing::info!(why, bluetooth, "PlayStation gamepad disconnected");
        lock().retain(|pad| pad.id != id);
        // SAFETY: ours, and nothing uses them any more.
        unsafe {
            if !event.is_null() {
                CloseHandle(event);
            }
            CloseHandle(handle);
        }
    }

    /// The pad at `path` opened for reading, `None` when it is something else
    /// or cannot be opened.
    fn open(path: &[u16], bluetooth: bool) -> Option<(HANDLE, Reports, ReportKind)> {
        let mut name = path.to_vec();
        name.push(0);
        // SAFETY: a terminated path; no security attributes or template.
        let handle = unsafe {
            CreateFileW(
                name.as_ptr(),
                GENERIC_READ | GENERIC_WRITE,
                FILE_SHARE_READ | FILE_SHARE_WRITE,
                std::ptr::null(),
                OPEN_EXISTING,
                FILE_FLAG_OVERLAPPED,
                std::ptr::null_mut(),
            )
        };
        if handle == INVALID_HANDLE_VALUE {
            return None;
        }
        let mut attributes = HIDD_ATTRIBUTES {
            Size: size_of::<HIDD_ATTRIBUTES>() as u32,
            ..Default::default()
        };
        // SAFETY: an open device and room for its attributes.
        if unsafe { HidD_GetAttributes(handle, &mut attributes) } && attributes.VendorID == SONY {
            let kind = match (attributes.ProductID, bluetooth) {
                (product, false) if DUALSHOCK4_PRODUCTS.contains(&product) => {
                    Some(ReportKind::DualShock4Usb)
                }
                (product, false) if PRODUCTS.contains(&product) => Some(ReportKind::DualSenseUsb),
                (product, true) if PRODUCTS.contains(&product) => {
                    Some(ReportKind::DualSenseBluetooth)
                }
                _ => None,
            };
            if let Some(kind) = kind
                && let Some(reports) = Reports::for_device(handle, kind)
            {
                return Some((handle, reports, kind));
            }
        }
        // SAFETY: just opened.
        unsafe { CloseHandle(handle) };
        None
    }

    /// The device paths of the HID devices present, without terminators.
    fn hid_paths() -> Vec<Vec<u16>> {
        let mut paths = Vec::new();
        // SAFETY: plain queries into buffers we own; the list is destroyed
        // before returning.
        unsafe {
            let mut guid = std::mem::zeroed();
            HidD_GetHidGuid(&mut guid);
            let set = SetupDiGetClassDevsW(
                &guid,
                std::ptr::null(),
                std::ptr::null_mut(),
                DIGCF_PRESENT | DIGCF_DEVICEINTERFACE,
            );
            if set == INVALID_HANDLE_VALUE as isize {
                return paths;
            }
            for index in 0.. {
                let mut interface = SP_DEVICE_INTERFACE_DATA {
                    cbSize: size_of::<SP_DEVICE_INTERFACE_DATA>() as u32,
                    ..std::mem::zeroed()
                };
                if SetupDiEnumDeviceInterfaces(set, std::ptr::null(), &guid, index, &mut interface)
                    == 0
                {
                    break;
                }
                let mut size = 0;
                SetupDiGetDeviceInterfaceDetailW(
                    set,
                    &interface,
                    std::ptr::null_mut(),
                    0,
                    &mut size,
                    std::ptr::null_mut(),
                );
                if size == 0 {
                    continue;
                }
                // u64s keep the detail's alignment.
                let mut buffer = vec![0u64; (size as usize).div_ceil(8)];
                let detail = buffer
                    .as_mut_ptr()
                    .cast::<SP_DEVICE_INTERFACE_DETAIL_DATA_W>();
                (*detail).cbSize = size_of::<SP_DEVICE_INTERFACE_DETAIL_DATA_W>() as u32;
                if SetupDiGetDeviceInterfaceDetailW(
                    set,
                    &interface,
                    detail,
                    size,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                ) == 0
                {
                    continue;
                }
                let start = std::ptr::addr_of!((*detail).DevicePath).cast::<u16>();
                let room = (size as usize
                    - std::mem::offset_of!(SP_DEVICE_INTERFACE_DETAIL_DATA_W, DevicePath))
                    / 2;
                let wide = std::slice::from_raw_parts(start, room);
                let end = wide.iter().position(|&c| c == 0).unwrap_or(room);
                paths.push(wide[..end].to_vec());
            }
            SetupDiDestroyDeviceInfoList(set);
        }
        paths
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn automatic_input_prefers_the_remapper_in_any_slot_and_falls_back_to_hid() {
        let physical = PadState {
            buttons: A,
            ..PadState::default()
        };
        let remapped = PadState {
            buttons: Y,
            ..PadState::default()
        };
        let mut assigned = [None; super::super::PADS];
        let none = [None; super::super::PADS];
        let xinput = [None, None, Some(remapped), None];

        assert_eq!(
            slots(Mode::Auto, none, &mut assigned, || vec![(1, physical)])[0],
            Some(physical)
        );
        assert_eq!(
            slots(Mode::Auto, xinput, &mut assigned, || panic!(
                "HID not needed"
            )),
            xinput
        );
        assert_eq!(
            slots(Mode::Off, none, &mut assigned, || panic!("HID disabled")),
            none
        );
        assert_eq!(
            slots(Mode::Auto, none, &mut assigned, || vec![(2, physical)])[0],
            Some(physical)
        );
    }

    #[test]
    fn a_surviving_hid_pad_keeps_its_slot_through_disconnect_and_reconnect() {
        let a = PadState {
            buttons: A,
            ..PadState::default()
        };
        let b = PadState {
            buttons: B,
            ..PadState::default()
        };
        let x = PadState {
            buttons: X,
            ..PadState::default()
        };
        let xinput = [None, Some(x), None, None];
        let mut assigned = [None; super::super::PADS];
        assert_eq!(
            slots(Mode::Both, xinput, &mut assigned, || vec![(7, a), (8, b)]),
            [Some(a), Some(x), Some(b), None]
        );
        assert_eq!(
            slots(Mode::Both, xinput, &mut assigned, || vec![(8, b)]),
            [None, Some(x), Some(b), None]
        );
        assert_eq!(
            slots(Mode::Both, xinput, &mut assigned, || vec![(8, b), (9, a)]),
            [Some(a), Some(x), Some(b), None]
        );
    }

    #[test]
    fn the_pairing_report_holds_the_address() {
        let mut pairing = [0u8; 20];
        pairing[0] = 0x09;
        pairing[1..7].copy_from_slice(&[1, 2, 3, 4, 5, 6]);
        assert_eq!(mac(&pairing), Some([1, 2, 3, 4, 5, 6]));
        // a pad that answered with nothing cannot be told apart
        assert_eq!(mac(&[0x09; 1]), None);
        assert_eq!(mac(&[0x09, 0, 0, 0, 0, 0, 0]), None);
    }

    #[test]
    fn a_usb_report_reads_as_a_pad() {
        let mut report = [0u8; 64];
        report[0] = 0x01;
        report[1..5].copy_from_slice(&[0, 255, 128, 128]); // left stick: left, down
        report[5] = 255; // L2
        report[8] = 0x20 | 2; // cross, D-pad right
        report[9] = 0x20 | 0x01; // options, L1
        let pad = parse(&report, ReportKind::DualSenseUsb).unwrap();
        assert_eq!(pad.buttons, A | RIGHT | MENU | LEFT_SHOULDER);
        assert_eq!(
            (pad.left_x, pad.left_y, pad.left_trigger),
            (-1.0, -1.0, 1.0)
        );
        assert!(pad.right_x.abs() < 0.01 && pad.right_y.abs() < 0.01);
    }

    #[test]
    fn bluetooth_reports_read_one_byte_further_in() {
        let mut report = [0u8; 78];
        report[0] = 0x31;
        report[2..6].copy_from_slice(&[128, 0, 255, 128]); // left up, right right
        report[7] = 255; // R2
        report[9] = 0x80 | 8; // triangle, D-pad centred
        let pad = parse(&report, ReportKind::DualSenseBluetooth).unwrap();
        assert_eq!(pad.buttons, Y);
        assert_eq!(
            (pad.left_y, pad.right_x, pad.right_trigger),
            (1.0, 1.0, 1.0)
        );
        // the same bytes over USB are not a pad report
        assert_eq!(parse(&report, ReportKind::DualSenseUsb), None);
    }

    #[test]
    fn the_short_bluetooth_report_reads_too() {
        let mut report = [0u8; 78];
        report[0] = 0x01;
        report[1..5].copy_from_slice(&[128; 4]);
        report[5] = 0x40 | 4; // circle, D-pad down
        report[7] = 0x02; // touchpad click
        report[8] = 128; // L2 halfway
        let pad = parse(&report, ReportKind::DualSenseBluetooth).unwrap();
        assert_eq!(pad.buttons, B | DOWN | VIEW);
        assert!((pad.left_trigger - 0.5).abs() < 0.01);
    }

    #[test]
    fn dualshock4_usb_reads_sticks_buttons_and_triggers() {
        let mut report = [0u8; 64];
        report[..10].copy_from_slice(&[1, 0, 255, 255, 0, 0x21, 0x20, 0x02, 255, 128]);
        let pad = parse(&report, ReportKind::DualShock4Usb).unwrap();
        assert_eq!(pad.buttons, A | UP | RIGHT | MENU | VIEW);
        assert_eq!(
            (pad.left_x, pad.left_y, pad.right_x, pad.right_y),
            (-1.0, -1.0, 1.0, 1.0)
        );
        assert_eq!(pad.left_trigger, 1.0);
        assert!((pad.right_trigger - 0.5).abs() < 0.01);
        assert_eq!(parse(&report[..9], ReportKind::DualShock4Usb), None);
    }
}
