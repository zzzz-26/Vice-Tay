//! Fit dynamic interface viewboxes using uniform stretch mode when
//! rendering single-plane at a reduced resolution (`VICTORY_UI_VIEWBOX=1`).
//! Adapted from avatarmanz/VIctory-7900XTX PR #295.

use super::Site;
use std::ffi::OsStr;

pub(super) fn requested(size: (u32, u32), setting: Option<&OsStr>) -> bool {
    size != (3840, 2160) && setting == Some(OsStr::new("1"))
}

const MODES: [&[u8]; 4] = [
    b"STRETCH_TYPE_FILL",
    b"STRETCH_TYPE_UNIFORM",
    b"STRETCH_TYPE_UNIFORM_TO_FILL",
    b"STRETCH_TYPE_DYNAMIC",
];
const UNIFORM: u8 = 2;
const DYNAMIC: u8 = 4;
const STRIDE: usize = 12;

const fn name_hash(name: &[u8]) -> u32 {
    let mut hash = 0u32;
    let mut i = 0;
    while i < name.len() {
        hash = hash.wrapping_add(name[i] as u32);
        hash = hash.wrapping_add(hash << 10);
        hash ^= hash >> 6;
        i += 1;
    }
    hash = hash.wrapping_add(hash << 3);
    hash ^= hash >> 11;
    hash.wrapping_add(hash << 15)
}

const fn table(dynamic_mode: u8) -> [u8; STRIDE * MODES.len()] {
    let mut table = [0; STRIDE * MODES.len()];
    let mut i = 0;
    while i < MODES.len() {
        let offset = i * STRIDE;
        let hash = name_hash(MODES[i]).to_le_bytes();
        let mut j = 0;
        while j < hash.len() {
            table[offset + j] = hash[j];
            j += 1;
        }
        table[offset + 4] = if i == 3 { dynamic_mode } else { i as u8 + 1 };
        i += 1;
    }
    table
}

pub(super) const SITE: Site = Site {
    at: 0x630ff1c,
    what: "fit UI viewboxes to output resolution",
    bytes: &table(DYNAMIC),
};
pub(super) const FITTED: [u8; STRIDE * MODES.len()] = table(UNIFORM);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resolution::{Patched, matches};

    #[test]
    fn applies_to_non_4k_when_requested() {
        let on = Some(OsStr::new("1"));
        assert!(requested((1920, 1080), on));
        assert!(requested((1280, 720), on));
        assert!(requested((2560, 1440), on));
        assert!(!requested((3840, 2160), on));
        assert!(!requested((1920, 1080), None));
        assert!(!requested((1920, 1080), Some(OsStr::new("0"))));
    }

    #[test]
    fn only_changes_dynamic_stretch_entry() {
        let mut data = table(DYNAMIC);
        let index = 3 * STRIDE + 4;
        assert_eq!(data[index], DYNAMIC);
        data[index] = UNIFORM;
        assert_eq!(data, FITTED);

        let local = Site { at: 0, ..SITE };
        let mut original = table(DYNAMIC);
        let check = |buf: &[u8]| unsafe { matches(buf.as_ptr() as usize, buf.len(), &local) };
        assert_eq!(check(&original), Patched::Done);
        original[STRIDE + 4] += 1;
        assert_eq!(check(&original), Patched::Differs);
        assert!(matches!(check(&original[..STRIDE]), Patched::Failed(_)));
    }
}
