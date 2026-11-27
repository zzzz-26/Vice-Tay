//! Keep the title's interface at its native 3840x2160 layout on a separate
//! present plane when outputting 1920x1080 or 1280x720 on Series X. The GPU
//! compositor downsamples the 4K UI plane onto the lower-resolution scene so
//! subtitles, credits, HUD elements, and pause menus keep 4K proportions.
//! Adapted from avatarmanz/VIctory-7900XTX PR #295 for 720p and extended to
//! 1080p.

use super::{OUTPUT, Site};
use std::ffi::OsStr;

/// The separate 3840x2160 interface plane requires an exact integer multiple
/// scale factor (2x for 1080p, 3x for 720p) and the split-display pool.
pub(super) fn supports(output: (u32, u32)) -> bool {
    matches!(output, (1920, 1080) | (1280, 720))
}

/// Enabled for `(1920, 1080)` whenever the two-plane compositor is active unless
/// `VICTORY_UI_PLANE=0`; for `(1280, 720)` when `VICTORY_UI_PLANE=1` or
/// `VICTORY_UI_PLANE_720=1`.
pub(super) fn requested(
    output: (u32, u32),
    ui_plane: Option<&OsStr>,
    ui_plane_720: Option<&OsStr>,
    planes: Option<&OsStr>,
) -> bool {
    if !supports(output) || planes == Some(OsStr::new("0")) {
        return false;
    }
    let one = Some(OsStr::new("1"));
    let zero = Some(OsStr::new("0"));
    match output {
        (1920, 1080) => ui_plane != zero,
        (1280, 720) => ui_plane == one || ui_plane_720 == one,
        _ => false,
    }
}

// Series X initializes a 64 MiB display pool for a single plane, while
// Series S uses 82 MiB for split scene + 3840x2160 UI planes (+0x247ad45).
const SINGLE_POOL: [u8; 4] = (64u32 << 20).to_le_bytes();
pub(super) const SPLIT_POOL: [u8; 4] = (82u32 << 20).to_le_bytes();
pub(super) const POOL: Site = Site {
    at: 0x247ad4b,
    what: "separate interface plane display pool",
    bytes: &SINGLE_POOL,
};

/// Patch bytes for `OUTPUT` setting `resolutionFullscreen = w x h`,
/// `uiResolution = 3840 x 2160`, and `uiOverlay = true`.
pub(super) fn output_code(width: u32, height: u32) -> Option<Vec<u8>> {
    // Both 1920x1080 (2x) and 1280x720 (3x) derive packed 3840x2160
    // with one 4-byte `lea rdx, ...` so the 35-byte prefix fits before `+0x224354e`.
    let scale_op: [u8; 4] = match (width, height) {
        (1920, 1080) => [0x48, 0x8d, 0x14, 0x00], // lea rdx, [rax + rax]   -> 3840x2160
        (1280, 720) => [0x48, 0x8d, 0x14, 0x40],  // lea rdx, [rax + 2*rax] -> 3840x2160
        _ => return None,
    };
    let mut prefix = vec![0x48, 0x8d, 0x8e]; // lea rcx, [rsi + uiOverlay(0x1e8)]
    prefix.extend(0x1e8u32.to_le_bytes());
    prefix.extend([0x48, 0xb8]); // mov rax, packed (width | (height << 32))
    prefix.extend(width.to_le_bytes());
    prefix.extend(height.to_le_bytes());
    prefix.extend([0x48, 0x89, 0x41, 0x1c]); // resolutionFullscreen (+0x204) = rax
    prefix.extend(scale_op); // rdx = 3840x2160
    prefix.extend([0x48, 0x89, 0x51, 0x04]); // uiResolution (+0x1ec) = rdx
    prefix.extend([0xc6, 0x01, 0x01]); // uiOverlay (+0x1e8) = true
    prefix.extend([0x8b, 0x79, 0x20]); // mov edi, [rcx + 0x20] (output height)

    let mut code = OUTPUT.bytes.to_vec();
    debug_assert_eq!(prefix.len(), 35);
    code[..prefix.len()].copy_from_slice(&prefix);
    Some(code)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn activation_rules_for_1080p_and_720p() {
        let on = Some(OsStr::new("1"));
        let off = Some(OsStr::new("0"));
        // 1080p defaults on when compositor is enabled, opt-out via VICTORY_UI_PLANE=0 or VICTORY_PLANES=0.
        assert!(requested((1920, 1080), None, None, None));
        assert!(requested((1920, 1080), on, None, on));
        assert!(!requested((1920, 1080), off, None, None));
        assert!(!requested((1920, 1080), on, None, off));

        // 720p stays opt-in via VICTORY_UI_PLANE=1 or VICTORY_UI_PLANE_720=1.
        assert!(!requested((1280, 720), None, None, None));
        assert!(requested((1280, 720), None, on, None));
        assert!(requested((1280, 720), on, None, None));
        assert!(!requested((1280, 720), on, on, off));

        // Unsupported sizes stay false.
        assert!(!requested((3840, 2160), on, on, on));
        assert!(!requested((2560, 1440), on, on, on));
        assert!(!requested((1664, 936), on, on, on));
    }

    #[test]
    fn generates_exact_61_byte_patches_for_1080p_and_720p() {
        for (w, h, scale_byte) in [(1920, 1080, 0x00u8), (1280, 720, 0x40u8)] {
            let code = output_code(w, h).expect("supported mode");
            let packed = u64::from_le_bytes(code[9..17].try_into().unwrap());
            assert_eq!((packed as u32, (packed >> 32) as u32), (w, h));
            assert_eq!(code[21..25], [0x48, 0x8d, 0x14, scale_byte]);
            let factor = if scale_byte == 0x00 { 2 } else { 3 };
            let ui = packed * factor;
            assert_eq!((ui as u32, (ui >> 32) as u32), (3840, 2160));
            assert_eq!(code.len(), OUTPUT.bytes.len());
            assert_eq!(code[35..], OUTPUT.bytes[35..]);
        }
        assert!(output_code(2560, 1440).is_none());
        assert_eq!(u32::from_le_bytes(SPLIT_POOL) - u32::from_le_bytes(SINGLE_POOL), 18 << 20);
    }
}
