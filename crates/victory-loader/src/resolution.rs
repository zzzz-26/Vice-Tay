//! The title's output size and render scale (`VICTORY_OUTPUT`,
//! `VICTORY_RENDER_SCALE`, `VICTORY_FFX_SCALE`, `VICTORY_UPSCALER`; each off
//! without it): places
//! in the title's code changed at load, each compared byte for byte first
//! and left alone, said so, if it differs (another build of the title).
//!
//! What the title does (game_scarlett_master.exe 1.0.2.1, image offsets):
//!
//! - **Output size.** Its settings (`settings.xml` in its storage,
//!   `video/resolutionFullscreen`, 3840x2160 as a Series X wrote it; read
//!   into the live settings at +0x204/+0x208, the defaults' 1920x1080 at
//!   +0x5b37a20 overwritten) are applied by +0x22434a0: from +0x224352b it
//!   hands them to +0x2478130, the one writer of the screen size
//!   (+0xa9438cc/+0xa9438d0), and when the height changes calls +0xd8c3a0.
//!   The back buffers (+0x247ad10, by +0x501f4a0 and +0xcc2780) take that
//!   size through the copy +0x5eb24f0 makes (+0x75e6fe0). The display mode
//!   list it matches against is its own (+0x5a38910: 1280x720, 1664x936,
//!   1920x1080, 2560x1440, 3840x2160 at 60 Hz, table +0x615c0c0), not the
//!   DXGI output's, which is why `VICTORY_DISPLAY_MODE` changes nothing.
//!   Here that block becomes: write the size asked for into the settings
//!   (+0x204/+0x208), then the title's own else-path (screen size set,
//!   +0xd8c3a0 if the height changed). Everything that reads the settings
//!   after it (the device parameters +0x1e0fb30, the render size +0x7b4e10)
//!   sees the same size. The back buffers' pool (64 MB on Series X) is
//!   checked to be used up but for 2 MB (+0x247af25, an `int3`); for another
//!   size that check is jumped over.
//! - **Render scale.** The upscaler the title picks (+0x7e93f0: 6, its own,
//!   except on Series S with a flag, 7, AMD FidelityFX) gives the render
//!   size from the output size (+0x7e92f0 → the upscaler's +0x50 method):
//!   its own (+0x7e9440) multiplies by a scale at +0x90 of it, set to 2/3 at
//!   +0x28f1e78 (so 2560x1440 for 4K, 1280x720 for 1080p);
//!   `VICTORY_RENDER_SCALE` replaces that constant (Series X; run res-4 at
//!   1.0). FidelityFX's (+0x28ef680, Series S) multiplies by 1.0 on Series X,
//!   2/3 on Series S; `VICTORY_FFX_SCALE` points its two loads at a constant
//!   in its padding (as pull request #53 did). That one is not safe: at 0.5
//!   on Series S (run res-6) FidelityFX's jitter phase count came back 0 and
//!   the title divided by it (+0x28f03e8) 49 s in.
//! - **Upscaler.** `VICTORY_UPSCALER=<kind>` makes the picker return
//!   `kind` (pull request #53): 6 the title's own, 7 FidelityFX.

use crate::frame_rate::Patched;

#[cfg(any(windows, test))]
mod ui_plane;
#[cfg(any(windows, test))]
mod viewbox;

/// Sizes the title has a display mode for (its own list, +0x615c0c0).
pub const MODES: [(u32, u32); 5] = [(1280, 720), (1664, 936), (1920, 1080), (2560, 1440), (3840, 2160)];

/// `VICTORY_OUTPUT=<w>x<h>`: one of [`MODES`]. `None` if unset or not one
/// (said why).
pub fn asked_output() -> Option<(u32, u32)> {
    let spec = std::env::var("VICTORY_OUTPUT").ok()?;
    let size = parse_output(&spec);
    if size.is_none() {
        eprintln!("VIctory: VICTORY_OUTPUT={spec}: expected one of the title's modes: 1920x1080, 2560x1440, 1280x720, 1664x936, 3840x2160");
    }
    size
}

pub fn parse_output(spec: &str) -> Option<(u32, u32)> {
    let (w, h) = spec.trim().split_once(['x', 'X'])?;
    let size = (w.trim().parse().ok()?, h.trim().parse().ok()?);
    MODES.contains(&size).then_some(size)
}

/// `VICTORY_RENDER_SCALE=<s>`, 0.5 to 1.0 (0.5: half the output's width and
/// height): the title's own upscaler's scale (Series X; 2/3 without it).
/// Tested at 1.0 (run res-4); below 2/3 the title's largest render size
/// (+0x94, max(2/3, scale)) stays at 2/3. `None` if unset or out of range
/// (said why).
pub fn asked_render_scale() -> Option<f32> {
    let spec = std::env::var("VICTORY_RENDER_SCALE").ok()?;
    let scale = parse_render_scale(&spec);
    if scale.is_none() {
        eprintln!("VIctory: VICTORY_RENDER_SCALE={spec}: expected 0.5 to 1.0");
    }
    scale
}

pub fn parse_render_scale(spec: &str) -> Option<f32> {
    spec.trim().parse::<f32>().ok().filter(|s| (0.5..=1.0).contains(s))
}

/// `VICTORY_FFX_SCALE=<s>`, 0.5 to 1.0: FidelityFX's render scale (Series
/// S). Crashed at 0.5 (run res-6); see the module's notes.
pub fn asked_ffx_scale() -> Option<f32> {
    let spec = std::env::var("VICTORY_FFX_SCALE").ok()?;
    let scale = parse_render_scale(&spec);
    if scale.is_none() {
        eprintln!("VIctory: VICTORY_FFX_SCALE={spec}: expected 0.5 to 1.0");
    }
    scale
}

/// `VICTORY_UPSCALER=<kind>`: 6 (the title's own) or 7 (FidelityFX).
pub fn asked_upscaler() -> Option<u32> {
    let spec = std::env::var("VICTORY_UPSCALER").ok()?;
    let kind = spec.trim().parse::<u32>().ok().filter(|k| matches!(k, 6 | 7));
    if kind.is_none() {
        eprintln!("VIctory: VICTORY_UPSCALER={spec}: expected 6 (the title's own) or 7 (FidelityFX)");
    }
    kind
}

/// A place to change: its offset in the image and the bytes there
/// (compared first; the replacement is at most as long).
pub struct Site {
    pub at: usize,
    pub what: &'static str,
    pub bytes: &'static [u8],
}

/// The screen size setter, its screen height, and what +0x22434a0 calls when
/// the height changed.
pub const SET_SCREEN_SIZE: usize = 0x2478130;
pub const SCREEN_HEIGHT: usize = 0xa9438d0;
pub const HEIGHT_CHANGED: usize = 0xd8c3a0;

/// +0x22434a0 from +0x224352b to +0x2243568: the settings' size handed to
/// the screen size.
pub const OUTPUT: Site = Site {
    at: 0x224352b,
    what: "settings' output size applied",
    bytes: &[
        0x8b, 0x0d, 0x9b, 0x03, 0x70, 0x08, // mov ecx, [+0xa9438cc]
        0x8b, 0x86, 0x04, 0x02, 0x00, 0x00, // mov eax, [rsi+204h]
        0x39, 0xc1, 0x75, 0x0d, // cmp ecx, eax ; jne
        0x8b, 0x96, 0x08, 0x02, 0x00, 0x00, // mov edx, [rsi+208h]
        0xe8, 0xea, 0x4b, 0x23, 0x00, 0xeb, 0x20, // call +0x2478130 ; jmp +0x2243568
        0x8b, 0xbe, 0x08, 0x02, 0x00, 0x00, // mov edi, [rsi+208h]
        0x8b, 0x2d, 0x7c, 0x03, 0x70, 0x08, // mov ebp, [+0xa9438d0]
        0x89, 0xc1, 0x89, 0xfa, // mov ecx, eax ; mov edx, edi
        0xe8, 0xd3, 0x4b, 0x23, 0x00, // call +0x2478130
        0x39, 0xfd, 0x74, 0x07, 0xb1, 0x01, // cmp ebp, edi ; je ; mov cl, 1
        0xe8, 0x38, 0x8e, 0xb4, 0xfe, // call +0xd8c3a0
    ],
};

/// +0x247ad10, after the back buffers are made in their pool (64 MB on
/// Series X, 82 MB on Series S, from +0x247ad45): `int3` unless less than
/// 2 MB of the pool is left over. Sized for 3840x2160; at 1920x1080 49 MB
/// are left (run res-1 stopped here). With another output size the check
/// becomes a jump over the `int3`; the pool stays as large.
pub const POOL_CHECK: Site = Site {
    at: 0x247af25,
    what: "back buffer pool's left-over check",
    bytes: &[0x48, 0x3d, 0xff, 0xff, 0x1f, 0x00, 0x76, 0x01, 0xcc], // cmp rax, 1FFFFFh ; jbe +1 ; int3
};

/// What goes at [`POOL_CHECK`]: `jmp +1` in place of `jbe +1`.
pub fn pool_check_code() -> Vec<u8> {
    let mut c = POOL_CHECK.bytes[..8].to_vec();
    c[6] = 0xeb;
    c
}

/// The title's own upscaler, its method +0x28f1e70: `mov dword [rcx+90h],
/// 2/3` (the scale +0x7e9440 renders at).
pub const OWN_SCALE: Site = Site {
    at: 0x28f1e78,
    what: "the title's own upscaler's render scale",
    bytes: &[0xc7, 0x81, 0x90, 0x00, 0x00, 0x00, 0xab, 0xaa, 0x2a, 0x3f],
};

/// FidelityFX's render size method, +0x28ef680, and its padding: the scale
/// is 1.0 (+0x5fc6ed8) or, on Series S, 2/3 (+0x615d95c), loaded at +0x28
/// and +0x32.
pub const FFX_SCALE: Site = Site {
    at: 0x28ef680,
    what: "FidelityFX's render scale",
    bytes: &[
        0x41, 0x56, 0x56, 0x57, 0x53, 0x48, 0x83, 0xec, 0x28, 0x44, 0x89, 0xcb, 0x4c, 0x89, 0xc6, 0x48, 0x89, 0xd7, 0x49, 0x89,
        0xce, 0xe8, 0xa6, 0x5a, 0x78, 0xfd, 0x0f, 0xb6, 0xc0, 0x41, 0x89, 0x86, 0x90, 0x00, 0x00, 0x00, 0x84, 0xc0, 0x75, 0x0a,
        0xc5, 0xfa, 0x10, 0x05, 0x28, 0x78, 0x6d, 0x03, 0xeb, 0x08, 0xc5, 0xfa, 0x10, 0x05, 0xa2, 0xe2, 0x86, 0x03, 0xc5, 0xf2,
        0x2a, 0x54, 0x24, 0x70, 0xc5, 0xf2, 0x2a, 0xcb, 0xc5, 0xfa, 0x59, 0xc9, 0xc5, 0xfa, 0x2c, 0xc1, 0x89, 0x07, 0xc5, 0xfa,
        0x59, 0xc2, 0xc5, 0xfa, 0x2c, 0xc0, 0x89, 0x06, 0x48, 0x83, 0xc4, 0x28, 0x5b, 0x5f, 0x5e, 0x41, 0x5e, 0xc3, 0xcc, 0xcc,
        0xcc, 0xcc, 0xcc, 0xcc,
    ],
};

/// The upscaler picker, +0x7e93f0, to its first `ret`.
pub const PICKER: Site = Site {
    at: 0x7e93f0,
    what: "upscaler picker",
    bytes: &[
        0x48, 0x83, 0xec, 0x28, 0x80, 0x3d, 0x6d, 0xf0, 0x3a, 0x07, 0x00, 0x75, 0x09, 0x83, 0x3d, 0x68, 0xf0, 0x3a, 0x07, 0x00,
        0x74, 0x28, 0xe8, 0x35, 0xbd, 0x88, 0xff, 0x8a, 0x0d, 0x5b, 0xf0, 0x3a, 0x07, 0x80, 0xe1, 0x40, 0x80, 0xf9, 0x01, 0xb9,
        0x07, 0x00, 0x00, 0x00, 0x83, 0xd9, 0x00, 0x84, 0xc0, 0xb8, 0x06, 0x00, 0x00, 0x00, 0x0f, 0x45, 0xc1, 0x48, 0x83, 0xc4,
        0x28, 0xc3,
    ],
};

/// `rel32` from the end of an instruction at image offset `next` to `to`.
fn rel(to: usize, next: usize) -> [u8; 4] {
    ((to as i64 - next as i64) as i32).to_le_bytes()
}

/// What goes at [`OUTPUT`] for `width` x `height` (the same length).
pub fn output_code(width: u32, height: u32) -> Vec<u8> {
    let at = OUTPUT.at;
    let mut c = Vec::with_capacity(OUTPUT.bytes.len());
    c.push(0xb8); // mov eax, width
    c.extend(width.to_le_bytes());
    c.extend([0x89, 0x86, 0x04, 0x02, 0x00, 0x00]); // mov [rsi+204h], eax
    c.push(0xbf); // mov edi, height
    c.extend(height.to_le_bytes());
    c.extend([0x89, 0xbe, 0x08, 0x02, 0x00, 0x00]); // mov [rsi+208h], edi
    c.extend([0x8b, 0x2d]); // mov ebp, [screen height]
    c.extend(rel(SCREEN_HEIGHT, at + c.len() + 4));
    c.extend([0x89, 0xc1, 0x89, 0xfa]); // mov ecx, eax ; mov edx, edi
    c.push(0xe8); // call the setter
    c.extend(rel(SET_SCREEN_SIZE, at + c.len() + 4));
    let end = OUTPUT.bytes.len();
    c.extend([0x39, 0xfd, 0x74, (end - (c.len() + 4)) as u8]); // cmp ebp, edi ; je end
    c.extend([0xb1, 0x01, 0xe8]); // mov cl, 1 ; call
    c.extend(rel(HEIGHT_CHANGED, at + c.len() + 4));
    c.extend([0xeb, (end - (c.len() + 2)) as u8]); // jmp end
    c.resize(end, 0xcc);
    c
}

/// What goes at [`OWN_SCALE`] for `scale`.
pub fn own_scale_code(scale: f32) -> Vec<u8> {
    let mut c = OWN_SCALE.bytes[..6].to_vec();
    c.extend(scale.to_le_bytes());
    c
}

/// What goes at [`FFX_SCALE`] for `scale`: both loads read `scale`, written
/// into the padding at +0x64.
pub fn ffx_scale_code(scale: f32) -> Vec<u8> {
    const LOADS: [usize; 2] = [0x28, 0x32];
    const SLOT: usize = 0x64;
    let mut c = FFX_SCALE.bytes.to_vec();
    c[SLOT..SLOT + 4].copy_from_slice(&scale.to_le_bytes());
    for l in LOADS {
        c[l + 4..l + 8].copy_from_slice(&((SLOT - (l + 8)) as i32).to_le_bytes());
    }
    c
}

/// `mov eax, kind ; ret`.
pub fn picker_code(kind: u32) -> Vec<u8> {
    let mut c = vec![0xb8];
    c.extend(kind.to_le_bytes());
    c.push(0xc3);
    c
}

/// Compares `site`'s bytes in the image of `size` bytes at `base`.
///
/// # Safety
/// The image is mapped.
pub unsafe fn matches(base: usize, size: usize, site: &Site) -> Patched {
    if site.at.checked_add(site.bytes.len()).is_none_or(|end| end > size) {
        return Patched::Failed("outside the image");
    }
    // SAFETY: inside the mapped image (checked above).
    let there = unsafe { std::slice::from_raw_parts((base + site.at) as *const u8, site.bytes.len()) };
    if there == site.bytes { Patched::Done } else { Patched::Differs }
}

/// Writes `code` over `site` if its bytes are the expected ones.
///
/// # Safety
/// The image stays mapped and no thread runs its code yet.
#[cfg(windows)]
pub unsafe fn write(base: usize, size: usize, site: &Site, code: &[u8]) -> Patched {
    use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
    use windows_sys::Win32::System::Memory::{PAGE_EXECUTE_READWRITE, VirtualProtect};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    if code.len() > site.bytes.len() {
        return Patched::Failed("replacement longer than the bytes compared");
    }
    // SAFETY: the caller's promise.
    let found = unsafe { matches(base, size, site) };
    if found != Patched::Done {
        return found;
    }
    let at = base + site.at;
    let mut old = 0;
    // SAFETY: inside the image (checked); made writable for this and put
    // back; nothing runs it yet.
    unsafe {
        if VirtualProtect(at as *const _, code.len(), PAGE_EXECUTE_READWRITE, &mut old) == 0 {
            return Patched::Failed("Windows would not make it writable");
        }
        std::ptr::copy_nonoverlapping(code.as_ptr(), at as *mut u8, code.len());
        VirtualProtect(at as *const _, code.len(), old, &mut old);
        FlushInstructionCache(GetCurrentProcess(), at as *const _, code.len());
    }
    Patched::Done
}

/// Applies what the switches ask for; a line each in the log and on
/// standard output.
///
/// # Safety
/// The image of `size` bytes at `base` stays mapped and no thread runs its
/// code yet.
#[cfg(windows)]
pub unsafe fn apply_asked(base: usize, size: usize) {
    let mut asked: Vec<(&Site, Vec<u8>, String)> = Vec::new();
    if let Some((w, h)) = asked_output() {
        let split_code = ui_plane::requested(
            (w, h),
            std::env::var_os("VICTORY_UI_PLANE").as_deref(),
            std::env::var_os("VICTORY_UI_PLANE_720").as_deref(),
            std::env::var_os("VICTORY_PLANES").as_deref(),
        )
        .then(|| ui_plane::output_code(w, h))
        .flatten()
        .filter(|_| unsafe {
            [&OUTPUT, &POOL_CHECK, &ui_plane::POOL]
                .iter()
                .all(|site| matches(base, size, site) == Patched::Done)
        });
        if let Some(code) = split_code {
            let value = format!("{w}x{h} scene, 3840x2160 interface");
            asked.push((&POOL_CHECK, pool_check_code(), value.clone()));
            asked.push((&ui_plane::POOL, ui_plane::SPLIT_POOL.to_vec(), value.clone()));
            asked.push((&OUTPUT, code, value));
        } else {
            asked.push((&OUTPUT, output_code(w, h), format!("{w}x{h}")));
            if (w, h) != (3840, 2160) {
                asked.push((&POOL_CHECK, pool_check_code(), format!("{w}x{h}")));
            }
            if viewbox::requested((w, h), std::env::var_os("VICTORY_UI_VIEWBOX").as_deref()) {
                asked.push((&viewbox::SITE, viewbox::FITTED.to_vec(), format!("{w}x{h}")));
            }
        }
    }
    if let Some(s) = asked_render_scale() {
        asked.push((&OWN_SCALE, own_scale_code(s), format!("{s}")));
    }
    if let Some(s) = asked_ffx_scale() {
        asked.push((&FFX_SCALE, ffx_scale_code(s), format!("{s}")));
    }
    if let Some(kind) = asked_upscaler() {
        asked.push((&PICKER, picker_code(kind), format!("returns {kind}")));
    }
    for (site, code, value) in asked {
        // SAFETY: the caller's promise.
        let done = unsafe { write(base, size, site, &code) };
        let offset = format!("{:#x}", site.at);
        match done {
            Patched::Done => tracing::info!(site = %offset, what = site.what, value, "resolution: patched"),
            Patched::Differs => tracing::warn!(site = %offset, what = site.what, "resolution: not patched, the title's bytes there are not the ones expected (another build?)"),
            Patched::Failed(why) => tracing::warn!(site = %offset, what = site.what, why, "resolution: not patched"),
        }
        println!("resolution: {} ({value}) at title+{offset}: {done:?}", site.what);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn switches_parse() {
        assert_eq!(parse_output("1920x1080"), Some((1920, 1080)));
        assert_eq!(parse_output(" 2560X1440 "), Some((2560, 1440)));
        assert_eq!(parse_output("1600x900"), None);
        assert_eq!(parse_output("1080p"), None);
        assert_eq!(parse_render_scale("0.67"), Some(0.67));
        assert_eq!(parse_render_scale("0.5"), Some(0.5));
        assert_eq!(parse_render_scale("1"), Some(1.0));
        assert_eq!(parse_render_scale("0.4"), None);
        assert_eq!(parse_render_scale("1.5"), None);
    }

    /// The calls and loads in the bytes compared are the ones the
    /// replacement keeps.
    #[test]
    fn the_output_block_is_the_titles() {
        let b = OUTPUT.bytes;
        assert_eq!(b.len(), 61);
        let target = |op_end: usize| (OUTPUT.at + op_end) as i64 + i64::from(i32::from_le_bytes(b[op_end - 4..op_end].try_into().unwrap()));
        assert_eq!(target(27), SET_SCREEN_SIZE as i64);
        assert_eq!(target(50), SET_SCREEN_SIZE as i64);
        assert_eq!(target(61), HEIGHT_CHANGED as i64);
        assert_eq!(target(41), SCREEN_HEIGHT as i64);
        assert_eq!(target(6), (SCREEN_HEIGHT - 4) as i64, "the width before it");
        // jne to the else path at 29; jmp +0x20 and je +7 both land at the end.
        assert_eq!((b[14], 16 + b[15] as usize), (0x75, 29));
        assert_eq!((b[27], 29 + b[28] as usize), (0xeb, 61));
        assert_eq!((b[52], 54 + b[53] as usize), (0x74, 61));
    }

    #[test]
    fn the_output_code_keeps_the_titles_else_path() {
        let c = output_code(1920, 1080);
        assert_eq!(c.len(), OUTPUT.bytes.len());
        assert_eq!(c[0], 0xb8);
        assert_eq!(c[1..5], 1920u32.to_le_bytes());
        assert_eq!(c[5..11], [0x89, 0x86, 0x04, 0x02, 0x00, 0x00]);
        assert_eq!(c[12..16], 1080u32.to_le_bytes());
        assert_eq!(c[16..22], [0x89, 0xbe, 0x08, 0x02, 0x00, 0x00]);
        let target = |op_end: usize| (OUTPUT.at + op_end) as i64 + i64::from(i32::from_le_bytes(c[op_end - 4..op_end].try_into().unwrap()));
        assert_eq!(c[22..24], [0x8b, 0x2d]);
        assert_eq!(target(28), SCREEN_HEIGHT as i64);
        assert_eq!(c[32], 0xe8);
        assert_eq!(target(37), SET_SCREEN_SIZE as i64);
        assert_eq!(c[37..40], [0x39, 0xfd, 0x74]);
        assert_eq!(41 + c[40] as usize, 61, "je lands at the end");
        assert_eq!(c[43], 0xe8);
        assert_eq!(target(48), HEIGHT_CHANGED as i64);
        assert_eq!(c[48], 0xeb);
        assert_eq!(50 + c[49] as usize, 61, "jmp lands at the end");
        assert!(c[50..].iter().all(|&x| x == 0xcc));
    }

    #[test]
    fn the_scale_codes() {
        let own = own_scale_code(0.5);
        assert_eq!(own[..6], OWN_SCALE.bytes[..6]);
        assert_eq!(own[6..], 0.5f32.to_le_bytes());
        assert_eq!(OWN_SCALE.bytes[6..], (2.0f32 / 3.0).to_le_bytes());
        let ffx = ffx_scale_code(0.75);
        assert_eq!(ffx.len(), FFX_SCALE.bytes.len());
        assert_eq!(ffx[0x64..0x68], 0.75f32.to_le_bytes());
        for at in [0x28, 0x32] {
            assert_eq!(ffx[at..at + 4], [0xc5, 0xfa, 0x10, 0x05], "vmovss xmm0, [rip+d]");
            let disp = i32::from_le_bytes(ffx[at + 4..at + 8].try_into().unwrap());
            assert_eq!(at as i32 + 8 + disp, 0x64);
        }
        // Only the two displacements and the padding change.
        let changed: Vec<usize> = (0..ffx.len()).filter(|&i| ffx[i] != FFX_SCALE.bytes[i]).collect();
        assert!(changed.iter().all(|&i| (0x2c..0x30).contains(&i) || (0x36..0x3a).contains(&i) || (0x64..0x68).contains(&i)));
        assert_eq!(picker_code(7), [0xb8, 7, 0, 0, 0, 0xc3]);
        assert_eq!(pool_check_code(), [0x48, 0x3d, 0xff, 0xff, 0x1f, 0x00, 0xeb, 0x01]);
    }

    #[cfg(windows)]
    #[test]
    fn a_site_is_written_only_over_its_bytes() {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_EXECUTE_READ, VirtualAlloc, VirtualFree};
        // SAFETY: fresh memory of our own, read-only to code as the image's is.
        unsafe {
            let place = VirtualAlloc(std::ptr::null(), 0x1000, MEM_COMMIT | MEM_RESERVE, 0x40) as *mut u8;
            assert!(!place.is_null());
            let site = Site { at: 0x100, what: "test", bytes: OWN_SCALE.bytes };
            std::ptr::copy_nonoverlapping(site.bytes.as_ptr(), place.add(0x100), site.bytes.len());
            let mut old = 0;
            windows_sys::Win32::System::Memory::VirtualProtect(place.cast(), 0x1000, PAGE_EXECUTE_READ, &mut old);
            assert_eq!(write(place as usize, 0x1000, &site, &own_scale_code(0.5)), Patched::Done);
            assert_eq!(std::slice::from_raw_parts(place.add(0x106), 4), 0.5f32.to_le_bytes());
            // Already changed: its bytes differ now, nothing is written again.
            assert_eq!(write(place as usize, 0x1000, &site, &own_scale_code(0.75)), Patched::Differs);
            assert_eq!(std::slice::from_raw_parts(place.add(0x106), 4), 0.5f32.to_le_bytes());
            let far = Site { at: 0xffa, what: "test", bytes: OWN_SCALE.bytes };
            assert!(matches!(write(place as usize, 0x1000, &far, &own_scale_code(0.5)), Patched::Failed(_)));
            VirtualFree(place.cast(), 0, MEM_RELEASE);
        }
    }
}
