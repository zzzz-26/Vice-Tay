//! Rumble for a DualSense read over USB HID: output report 0x02 with the
//! title's two motor levels. `VICTORY_DUALSENSE_RUMBLE=0` leaves the motors
//! alone.
//!
//! The report layout and the firmware check are ported from Asem912's
//! Marvisak/VIctory#2337 as developed in aleksesipenko/VIctory-RTX
//! (`gameinput/dualsense/protocol.rs`, GPL-3.0-or-later); they follow Linux
//! `hid-playstation.c`. Only USB is driven: Bluetooth (0x31, with a CRC-32)
//! and the DualShock 4 have no hardware coverage yet.

// Only the Windows host writes reports; the tests run everywhere.
#![cfg_attr(not(windows), allow(dead_code))]

/// USB output report 0x02: 48 bytes, or what the device asks for.
pub(super) const USB_OUTPUT_ID: u8 = 0x02;
pub(super) const USB_OUTPUT_LEN: usize = 48;
/// Feature report 0x20: firmware. From update version 0x0224 on, the motors
/// take the second compatible vibration mode.
pub(super) const FEATURE_FIRMWARE_ID: u8 = 0x20;
const FEATURE_FIRMWARE_LEN: usize = 64;
const FIRMWARE_VIBRATION2_MIN: u16 = 0x0224;

const FLAG0_COMPATIBLE_VIBRATION: u8 = 0x01;
const FLAG0_HAPTICS_SELECT: u8 = 0x02;
const FLAG2_COMPATIBLE_VIBRATION2: u8 = 0x04;

/// Heavy (left, low frequency) and light (right, high frequency) motors.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(super) struct Motors {
    pub(super) heavy: u8,
    pub(super) light: u8,
}

/// The firmware update version in feature report 0x20, `None` when it is
/// not there.
pub(super) fn firmware_version(report: &[u8]) -> Option<u16> {
    if report.len() < FEATURE_FIRMWARE_LEN || report[0] != FEATURE_FIRMWARE_ID {
        return None;
    }
    let version = u16::from_le_bytes([report[44], report[45]]);
    (version != 0).then_some(version)
}

/// A USB output report, `len` bytes (at least 48), that sets the motors and
/// leaves the triggers, lights and audio alone.
pub(super) fn usb(motors: Motors, firmware: u16, len: usize) -> Vec<u8> {
    let mut buf = vec![0u8; len.max(USB_OUTPUT_LEN)];
    buf[0] = USB_OUTPUT_ID;
    // The 47 bytes after the id: flags at 0, 1 and 38; motors at 2 and 3.
    let common = &mut buf[1..USB_OUTPUT_LEN];
    common[0] = FLAG0_HAPTICS_SELECT;
    if firmware >= FIRMWARE_VIBRATION2_MIN {
        common[38] = FLAG2_COMPATIBLE_VIBRATION2;
    } else {
        common[0] |= FLAG0_COMPATIBLE_VIBRATION;
    }
    common[2] = motors.light;
    common[3] = motors.heavy;
    buf
}

/// Whether a write that completed with `written` bytes took the whole
/// report of `len`: all of it, or all but the id byte, which Wine's HID
/// stack leaves out of the count (48 bytes complete as 47 under Proton).
pub(super) fn whole(written: u32, len: usize) -> bool {
    let written = written as usize;
    written == len || written + 1 == len
}

/// Whether the motors are driven (`VICTORY_DUALSENSE_RUMBLE=0`: not).
pub(super) fn rumble_on() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var_os("VICTORY_DUALSENSE_RUMBLE").is_none_or(|v| v != "0")
    });
    *ON
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usb_report_sets_the_motors_only() {
        let motors = Motors {
            heavy: 200,
            light: 50,
        };
        let new = usb(motors, 0x0630, 48);
        assert_eq!(new.len(), 48);
        assert_eq!(new[..5], [USB_OUTPUT_ID, FLAG0_HAPTICS_SELECT, 0, 50, 200]);
        assert_eq!(new[1 + 38], FLAG2_COMPATIBLE_VIBRATION2);
        // Trigger blocks and lights untouched.
        assert!(new[11..33].iter().all(|&b| b == 0));
        assert!(new[40..].iter().all(|&b| b == 0));
        let old = usb(motors, 0x0100, 64);
        assert_eq!(old.len(), 64);
        assert_eq!(old[1], FLAG0_HAPTICS_SELECT | FLAG0_COMPATIBLE_VIBRATION);
        assert_eq!(old[1 + 38], 0);
    }

    #[test]
    fn a_write_is_whole_with_or_without_the_id_byte() {
        assert!(whole(48, 48));
        assert!(whole(47, 48));
        assert!(!whole(0, 48));
        assert!(!whole(46, 48));
        assert!(!whole(49, 48));
    }

    #[test]
    fn firmware_version_is_read() {
        let mut report = [0u8; 64];
        assert_eq!(firmware_version(&report), None);
        report[0] = FEATURE_FIRMWARE_ID;
        assert_eq!(firmware_version(&report), None);
        report[44..46].copy_from_slice(&0x0630u16.to_le_bytes());
        assert_eq!(firmware_version(&report), Some(0x0630));
        assert_eq!(firmware_version(&report[..63]), None);
    }
}
