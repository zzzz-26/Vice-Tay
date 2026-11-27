//! NVIDIA DLSS Super Resolution behind the title's FidelityFX upscaler
//! calls (`VICTORY_DLSS_4K`): the same context, queries and dispatch, run
//! by NGX on the host GPU.
//!
//! Nothing of NVIDIA's SDK is linked or kept here. The driver's NGX core
//! (`_nvngx.dll`, in the folder named by the registry values the SDK's own
//! loader reads) is loaded at run time and its C exports are called by
//! name. NGX loads `nvngx_dlss.dll` from the folder in the feature path
//! list (`VICTORY_DLSS_DIR`, `[paths].dlss_dir`), else the driver's copy.
//! Parameters go through NGX's parameter object, a C++ class: the core
//! exports no C setters, so its virtual methods are called through its
//! table. MSVC lays a group of overloads out in reverse order of
//! declaration; a round trip of a probe value settles which order the
//! loaded build has.
//!
//! The title's FidelityFX descriptions map onto DLSS one to one:
//! - creation flags: HDR colour, inverted depth, motion vectors with the
//!   jitter in them, motion vectors at display size (else DLSS's low-res
//!   flag), auto exposure. Infinite depth has no DLSS counterpart.
//! - colour, depth, motion vectors, the output, the exposure texture when
//!   the title has DLSS not find the exposure itself, and the reactive mask
//!   as DLSS's mask biasing towards the current colour (both weigh the
//!   current frame over the history where they are 1). FidelityFX's
//!   transparency and composition mask has no DLSS input: NGX reserves
//!   `TransparencyMask` and reads nothing from it.
//! - the motion-vector scale as given: FidelityFX's (like NGX's `MV.Scale`)
//!   turns the title's vectors into render pixels, current to previous.
//! - jitter, in render pixels, as given: FidelityFX's `jitterOffset` is the
//!   offset the projection was moved by, which is what NGX's helpers take
//!   ("in input/render pixel space"; Unreal and NVIDIA's Donut samples hand
//!   the same value to both). `VICTORY_DLSS_JITTER_SIGN` flips it for A/B.
//! - reset, frame time, pre-exposure and the render size; the FidelityFX
//!   dispatch has no render offset, so every subrect starts at 0, 0.
//!
//! The queries the title makes of FidelityFX (render size per quality mode,
//! jitter sequence, memory) are answered with FidelityFX's formulas, so the
//! title renders at the sizes of the FSR profile it was set up for.
//!
//! When NGX fails a context (its feature cannot be made or evaluated, or
//! the device is removed), DLSS stops for that context, once, and the
//! host's FidelityFX takes it over if its folder was given; else the
//! context's dispatches fail. Features are released once the GPU is past
//! their last use, and NGX is shut down when the process ends in order.
#![allow(non_snake_case)]

use std::collections::BTreeMap;
use std::ffi::{CStr, c_char, c_void};
use std::path::{Path, PathBuf};
use std::ptr::{null, null_mut};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use victory_gpu::object::Object;
use windows::Win32::Graphics::Direct3D12::{ID3D12Device, ID3D12Resource};
use windows::core::Interface;

use super::{
    Binding, CREATE_UPSCALE, CreateUpscale, DISPATCH_UPSCALE, DispatchUpscale, GET_PROVIDER_VERSION,
    GET_RENDER_RESOLUTION, GET_UPSCALE_RATIO, GET_VERSIONS, GetVersions, Header,
    MEMORY_USAGE_WITH_DEVICE, MemoryUsageWithDevice, NO_PROVIDER, OK, PARAMETER, ProviderVersion,
    RatioQuery, RenderResolution, SetupError, UNKNOWN_TYPE, dispatch_output, host_state,
    minimum_extent, with_resource_states,
};

// FidelityFX queries the bridge answers itself.
const GET_JITTER_PHASE_COUNT: u64 = 0x1_0004;
const GET_JITTER_OFFSET: u64 = 0x1_0005;
const GPU_MEMORY_USAGE: u64 = 0x1_0008;

// FidelityFX creation flags.
const FFX_HDR: u32 = 1 << 0;
const FFX_DISPLAY_MOTION_VECTORS: u32 = 1 << 1;
const FFX_JITTERED_MOTION_VECTORS: u32 = 1 << 2;
const FFX_DEPTH_INVERTED: u32 = 1 << 3;
const FFX_AUTO_EXPOSURE: u32 = 1 << 5;

// NGX (nvsdk_ngx_defs.h): API version, feature, results, DLSS flags.
const SDK_VERSION: u32 = 0x15;
const SUPER_SAMPLING: u32 = 1;
const FAIL: u32 = 0xbad0_0000;
const DLSS_HDR: i32 = 1 << 0;
const DLSS_MV_LOW_RES: i32 = 1 << 1;
const DLSS_MV_JITTERED: i32 = 1 << 2;
const DLSS_DEPTH_INVERTED: i32 = 1 << 3;
const DLSS_AUTO_EXPOSURE: i32 = 1 << 6;
/// The GUID-like project id NGX takes from a custom engine in place of an
/// application id NVIDIA assigns.
const PROJECT_ID: &CStr = c"bd3b6220-d949-43a1-8a15-92bf6c75b590";
const ENGINE_CUSTOM: u32 = 0;
const ENGINE_VERSION: &CStr = c"1.0";
/// `NVSDK_NGX_LOGGING_LEVEL_ON`: creation and errors, nothing per frame.
const LOG_LEVEL: u32 = 1;

// Render presets: K is DLSS 4's transformer; L and M are DLSS 4.5's
// second-generation transformer, L tuned for ultra performance.
const PRESET_DEFAULT: u32 = 0;
const PRESET_J: u32 = 10;
const PRESET_K: u32 = 11;
const PRESET_L: u32 = 12;
const PRESET_M: u32 = 13;

/// What the provider calls itself, to FidelityFX version queries.
const PROVIDER_ID: u64 = 0x4e56_444c_5353;
const PROVIDER_NAME: &CStr = c"NVIDIA DLSS Super Resolution (NGX)";
/// Answered to the title's memory query until the feature reports its own.
const MEMORY_ESTIMATE: u64 = 256 << 20;

// The states DLSS reads its inputs and writes its output in.
const COMPUTE_READ: u32 = 0x40;
const UNORDERED_ACCESS: u32 = 0x8;

const DLL: &str = "nvngx_dlss.dll";

/// NGX succeeded (`NVSDK_NGX_SUCCEED`).
fn succeeded(result: u32) -> bool {
    result & 0xfff0_0000 != FAIL
}

/// An NGX result by its name in `NVSDK_NGX_Result`.
fn result_name(result: u32) -> String {
    if succeeded(result) {
        return "Success".into();
    }
    const NAMES: [&str; 19] = [
        "Fail",
        "FeatureNotSupported",
        "PlatformError",
        "FeatureAlreadyExists",
        "FeatureNotFound",
        "InvalidParameter",
        "ScratchBufferTooSmall",
        "NotInitialized",
        "UnsupportedInputFormat",
        "RWFlagMissing",
        "MissingInput",
        "UnableToInitializeFeature",
        "OutOfDate",
        "OutOfGPUMemory",
        "UnsupportedFormat",
        "UnableToWriteToAppDataPath",
        "UnsupportedParameter",
        "Denied",
        "NotImplemented",
    ];
    match NAMES.get((result & 0xf_ffff) as usize) {
        Some(name) => format!("{name} ({result:#x})"),
        None => format!("{result:#x}"),
    }
}

/// DLSS creation flags for a FidelityFX upscaler's.
fn create_flags(ffx: u32) -> i32 {
    let mut flags = 0;
    for (from, to) in [
        (FFX_HDR, DLSS_HDR),
        (FFX_JITTERED_MOTION_VECTORS, DLSS_MV_JITTERED),
        (FFX_DEPTH_INVERTED, DLSS_DEPTH_INVERTED),
        (FFX_AUTO_EXPOSURE, DLSS_AUTO_EXPOSURE),
    ] {
        if ffx & from != 0 {
            flags |= to;
        }
    }
    if ffx & FFX_DISPLAY_MOTION_VECTORS == 0 {
        flags |= DLSS_MV_LOW_RES;
    }
    flags
}

/// The model of each quality mode (`DLSS.Hint.Render.Preset.*`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Presets {
    dlaa: u32,
    ultra_quality: u32,
    quality: u32,
    balanced: u32,
    performance: u32,
    ultra_performance: u32,
}

impl Presets {
    /// DLSS 4.5 in every mode: M, and L where NVIDIA tuned it (ultra
    /// performance). Without hints the DLL picks DLSS 4's K for quality and
    /// balanced.
    const DLSS_4_5: Self = Self {
        dlaa: PRESET_M,
        ultra_quality: PRESET_M,
        quality: PRESET_M,
        balanced: PRESET_M,
        performance: PRESET_M,
        ultra_performance: PRESET_L,
    };

    fn every_mode(preset: u32) -> Self {
        Self {
            dlaa: preset,
            ultra_quality: preset,
            quality: preset,
            balanced: preset,
            performance: preset,
            ultra_performance: preset,
        }
    }

    /// `VICTORY_DLSS_PRESET`'s choice: unset for [`Self::DLSS_4_5`], or one
    /// model for every mode (J, K, L, M, or `default`: the DLL's own).
    fn chosen(name: Option<&str>) -> Option<Self> {
        let Some(name) = name else {
            return Some(Self::DLSS_4_5);
        };
        let preset = match name.trim().to_ascii_lowercase().as_str() {
            "default" => PRESET_DEFAULT,
            "j" => PRESET_J,
            "k" => PRESET_K,
            "l" => PRESET_L,
            "m" => PRESET_M,
            _ => return None,
        };
        Some(Self::every_mode(preset))
    }

    fn hints(&self) -> [(&'static CStr, u32); 6] {
        [
            (c"DLSS.Hint.Render.Preset.DLAA", self.dlaa),
            (c"DLSS.Hint.Render.Preset.UltraQuality", self.ultra_quality),
            (c"DLSS.Hint.Render.Preset.Quality", self.quality),
            (c"DLSS.Hint.Render.Preset.Balanced", self.balanced),
            (c"DLSS.Hint.Render.Preset.Performance", self.performance),
            (c"DLSS.Hint.Render.Preset.UltraPerformance", self.ultra_performance),
        ]
    }

    /// For the log: the letters, quality to ultra performance.
    fn letters(&self) -> String {
        let letter = |p: u32| match p {
            PRESET_DEFAULT => '-',
            5..=15 => char::from(b'A' + p as u8 - 1),
            _ => '?',
        };
        [self.quality, self.balanced, self.performance, self.ultra_performance]
            .map(letter)
            .iter()
            .collect()
    }
}

/// `VICTORY_DLSS_JITTER_SIGN`: the factor on FidelityFX's jitter per axis,
/// `1` (the default), `-1`, or `x,y` such as `1,-1`.
fn jitter_sign(value: Option<&str>) -> Option<[f32; 2]> {
    let Some(value) = value else {
        return Some([1.0, 1.0]);
    };
    let sign = |s: &str| match s.trim() {
        "1" | "+1" => Some(1.0),
        "-1" => Some(-1.0),
        _ => None,
    };
    match value.split_once(',') {
        Some((x, y)) => Some([sign(x)?, sign(y)?]),
        None => sign(value).map(|s| [s, s]),
    }
}

/// FidelityFX's upscale ratio for a quality mode (native, quality,
/// balanced, performance, ultra performance).
fn upscale_ratio(mode: u32) -> Option<f32> {
    [1.0, 1.5, 1.7, 2.0, 3.0].get(mode as usize).copied()
}

/// FidelityFX's jitter sequence length for these widths.
fn phase_count(render: u32, display: u32) -> i32 {
    if render == 0 || display == 0 {
        return 0;
    }
    ((8.0 * (display as f32 / render as f32).powi(2)) as i32).max(1)
}

fn halton(mut index: i32, base: i32) -> f32 {
    let (mut fraction, mut result) = (1.0, 0.0);
    while index > 0 {
        fraction /= base as f32;
        result += fraction * (index % base) as f32;
        index /= base;
    }
    result
}

/// FidelityFX's jitter offset, in render pixels, for one phase.
fn jitter_offset(index: i32, phases: i32) -> [f32; 2] {
    let phase = index.rem_euclid(phases.max(1)) + 1;
    [halton(phase, 2) - 0.5, halton(phase, 3) - 0.5]
}

/// What one FidelityFX dispatch asks of a DLSS evaluation.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Evaluation {
    jitter: [f32; 2],
    motion_scale: [f32; 2],
    render: [u32; 2],
    reset: bool,
    frame_ms: f32,
    pre_exposure: f32,
}

impl Evaluation {
    /// The settings words after the sizes are sharpening, sharpness, frame
    /// time (ms), pre-exposure, reset, camera, flags; booleans in the low
    /// byte. NGX's helpers turn a zero scale or pre-exposure into 1.
    fn of(dispatch: &DispatchUpscale, sign: [f32; 2]) -> Self {
        let one_if_zero = |v: f32| if v == 0.0 { 1.0 } else { v };
        Self {
            jitter: [dispatch.jitter[0] * sign[0], dispatch.jitter[1] * sign[1]],
            motion_scale: dispatch.motion_scale.map(one_if_zero),
            render: dispatch.render_size,
            reset: dispatch.settings[4] & 0xff != 0,
            frame_ms: f32::from_bits(dispatch.settings[2]),
            pre_exposure: one_if_zero(f32::from_bits(dispatch.settings[3])),
        }
    }
}

/// The folders the SDK's loader looks for the NGX core in, from the
/// registry values it reads (in its order), as DLL paths.
fn core_candidates(folders: &[Option<String>]) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    for folder in folders.iter().flatten() {
        let folder = folder.trim_end_matches(['\0', '\\', '/']);
        if folder.is_empty() {
            continue;
        }
        for name in ["_nvngx.dll", "nvngx.dll"] {
            let path = Path::new(folder).join(name);
            if !out.contains(&path) {
                out.push(path);
            }
        }
    }
    out
}

/// The registry values the SDK's loader reads for the NGX core's folder
/// (after asking the display driver, which writes the first of them).
const CORE_KEYS: [(&str, &str); 3] = [
    (
        r"System\CurrentControlSet\Services\nvlddmkm\Parameters\NGXCore",
        "NGXPath",
    ),
    (r"System\CurrentControlSet\Services\nvlddmkm\NGXCore", "NGXPath"),
    (r"SOFTWARE\NVIDIA Corporation\Global\NGXCore", "FullPath"),
];

fn wide(text: &str) -> Vec<u16> {
    text.encode_utf16().chain([0]).collect()
}

fn wide_path(path: &Path) -> Vec<u16> {
    use std::os::windows::ffi::OsStrExt;
    path.as_os_str().encode_wide().chain([0]).collect()
}

fn registry_string(key: &str, value: &str) -> Option<String> {
    use windows_sys::Win32::System::Registry::{HKEY_LOCAL_MACHINE, RRF_RT_REG_SZ, RegGetValueW};
    let (key, value) = (wide(key), wide(value));
    let mut buffer = vec![0u16; 1024];
    let mut bytes = (buffer.len() * 2) as u32;
    // SAFETY: NUL-terminated names and a buffer of `bytes` bytes.
    let status = unsafe {
        RegGetValueW(
            HKEY_LOCAL_MACHINE,
            key.as_ptr(),
            value.as_ptr(),
            RRF_RT_REG_SZ,
            null_mut(),
            buffer.as_mut_ptr().cast(),
            &mut bytes,
        )
    };
    if status != 0 {
        return None;
    }
    buffer.truncate(bytes as usize / 2);
    Some(String::from_utf16_lossy(&buffer))
}

/// A DLL's file version (`major.minor.build.revision`).
fn file_version(path: &Path) -> Option<String> {
    use windows_sys::Win32::Storage::FileSystem::{
        GetFileVersionInfoSizeW, GetFileVersionInfoW, VS_FIXEDFILEINFO, VerQueryValueW,
    };
    let name = wide_path(path);
    // SAFETY: a NUL-terminated path; the handle argument is unused.
    let size = unsafe { GetFileVersionInfoSizeW(name.as_ptr(), null_mut()) };
    if size == 0 {
        return None;
    }
    let mut block = vec![0u8; size as usize];
    let root = wide("\\");
    let mut info: *mut c_void = null_mut();
    let mut length = 0u32;
    // SAFETY: a buffer of the size asked for; the fixed info lies in it.
    let fixed = unsafe {
        if GetFileVersionInfoW(name.as_ptr(), 0, size, block.as_mut_ptr().cast()) == 0
            || VerQueryValueW(block.as_ptr().cast(), root.as_ptr(), &mut info, &mut length) == 0
            || info.is_null()
            || (length as usize) < size_of::<VS_FIXEDFILEINFO>()
        {
            return None;
        }
        info.cast::<VS_FIXEDFILEINFO>().read_unaligned()
    };
    let (ms, ls) = (fixed.dwFileVersionMS, fixed.dwFileVersionLS);
    Some(format!("{}.{}.{}.{}", ms >> 16, ms & 0xffff, ls >> 16, ls & 0xffff))
}

// The parameter object's table (nvsdk_ngx_params.h) holds eight `Set`
// overloads (unsigned long long, float, double, unsigned int, int, D3D11
// resource, D3D12 resource, void pointer), the eight matching `Get`s, then
// `Reset`.
#[derive(Clone, Copy)]
enum Kind {
    U64 = 0,
    F32 = 1,
    U32 = 3,
    I32 = 4,
    D3D12 = 6,
    Pointer = 7,
}

/// The order of each overload group in the table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Layout {
    /// MSVC's: reversed, `Set(void*)` first.
    Msvc,
    /// As declared, `Set(unsigned long long)` first.
    Declared,
}

impl Layout {
    fn set(self, kind: Kind) -> usize {
        match self {
            Layout::Declared => kind as usize,
            Layout::Msvc => 7 - kind as usize,
        }
    }

    fn get(self, kind: Kind) -> usize {
        8 + self.set(kind)
    }
}

#[repr(C)]
struct NgxParameter {
    table: *const *const c_void,
}

/// NGX's parameter object, through its table.
struct Parameters {
    object: *mut NgxParameter,
    layout: Layout,
}

impl Parameters {
    fn method(&self, index: usize) -> *const c_void {
        // SAFETY: a live NGX object, whose table has 17 methods.
        unsafe { *(*self.object).table.add(index) }
    }

    fn set<T: Copy>(&self, kind: Kind, name: &CStr, value: T) {
        type Set<T> = unsafe extern "C" fn(*mut NgxParameter, *const c_char, T);
        // SAFETY: the overload of `kind`, whose value argument is a `T`.
        unsafe {
            let set = std::mem::transmute_copy::<*const c_void, Set<T>>(
                &self.method(self.layout.set(kind)),
            );
            set(self.object, name.as_ptr(), value);
        }
    }

    /// The value as 8 bytes, whatever width the overload writes.
    fn get(&self, kind: Kind, name: &CStr) -> Option<u64> {
        type Get = unsafe extern "C" fn(*mut NgxParameter, *const c_char, *mut u64) -> u32;
        let mut out = 0u64;
        // SAFETY: a getter writing at most 8 bytes through its pointer.
        let result = unsafe {
            let get = std::mem::transmute::<*const c_void, Get>(self.method(self.layout.get(kind)));
            get(self.object, name.as_ptr(), &mut out)
        };
        succeeded(result).then_some(out)
    }

    fn set_u32(&self, name: &CStr, value: u32) {
        self.set(Kind::U32, name, value);
    }

    fn set_i32(&self, name: &CStr, value: i32) {
        self.set(Kind::I32, name, value);
    }

    fn set_f32(&self, name: &CStr, value: f32) {
        self.set(Kind::F32, name, value);
    }

    fn set_resource(&self, name: &CStr, value: *mut c_void) {
        self.set(Kind::D3D12, name, value);
    }

    fn get_i32(&self, name: &CStr) -> Option<i32> {
        self.get(Kind::I32, name).map(|v| v as u32 as i32)
    }

    /// The feature's video memory, from DLSS's statistics callback.
    fn video_memory(&self) -> Option<u64> {
        type Stats = unsafe extern "C" fn(*mut NgxParameter) -> u32;
        let callback = self.get(Kind::Pointer, c"DLSSGetStatsCallback")?;
        if callback == 0 {
            return None;
        }
        // SAFETY: NGX's PFN_NVSDK_NGX_DLSS_GetStatsCallback.
        let result = unsafe { std::mem::transmute::<usize, Stats>(callback as usize)(self.object) };
        if !succeeded(result) {
            return None;
        }
        self.get(Kind::U64, c"SizeInBytes")
    }
}

/// Which table order round-trips a float. MSVC's is tried first; in the
/// other order its float slots are the resource setter and getter, which
/// keep and return 8 bytes without reading them, so a wrong guess fails
/// harmlessly.
fn probe(object: *mut NgxParameter) -> Option<Layout> {
    [
        (Layout::Msvc, c"Victory.Probe.Msvc"),
        (Layout::Declared, c"Victory.Probe.Declared"),
    ]
    .into_iter()
    .find(|&(layout, name)| {
        let parameters = Parameters { object, layout };
        parameters.set_f32(name, 0.375);
        parameters
            .get(Kind::F32, name)
            .is_some_and(|v| f32::from_bits(v as u32) == 0.375)
    })
    .map(|(layout, _)| layout)
}

#[repr(C)]
struct PathList {
    paths: *const *const u16,
    count: u32,
}

#[repr(C)]
struct Logging {
    callback: Option<extern "C" fn(*const c_char, u32, u32)>,
    level: u32,
    other_sinks_off: bool,
}

/// `NVSDK_NGX_FeatureCommonInfo`.
#[repr(C)]
struct FeatureInfo {
    paths: PathList,
    internal: *mut c_void,
    logging: Logging,
}

const _: () = assert!(size_of::<FeatureInfo>() == 40);

/// NGX's log lines, into ours. NGX may call it from any thread.
extern "C" fn log_line(message: *const c_char, level: u32, _feature: u32) {
    if message.is_null() {
        return;
    }
    // SAFETY: a NUL-terminated line, valid for the call.
    let text = unsafe { CStr::from_ptr(message) }.to_string_lossy();
    let text = text.trim_end();
    if level > LOG_LEVEL {
        tracing::debug!(target: "ngx", "{text}");
    } else {
        tracing::info!(target: "ngx", "{text}");
    }
}

/// The core's `NVSDK_NGX_D3D12_Init_ProjectID`: the SDK's
/// `Init_with_ProjectID`, but with the API version ahead of the feature
/// info (in the SDK's order the core reads the version as the pointer).
type Init = unsafe extern "C" fn(
    *const c_char,
    u32,
    *const c_char,
    *const u16,
    *mut c_void,
    u32,
    *const FeatureInfo,
) -> u32;
type Capabilities = unsafe extern "C" fn(*mut *mut NgxParameter) -> u32;
type Create = unsafe extern "C" fn(*mut c_void, u32, *mut NgxParameter, *mut *mut c_void) -> u32;
type Evaluate = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut NgxParameter, usize) -> u32;
type Release = unsafe extern "C" fn(*mut c_void) -> u32;
type Shutdown = unsafe extern "C" fn(*mut c_void) -> u32;

struct Ngx {
    create: Create,
    evaluate: Evaluate,
    release: Release,
    shutdown: Shutdown,
    /// Shared by every context; NGX reads it during each call.
    parameters: Mutex<Parameters>,
    quality: i32,
    ffx_quality: Option<u32>,
    presets: Presets,
    jitter_sign: [f32; 2],
    /// Whether the title's reactive mask goes to DLSS.
    reactive: bool,
    /// The feature folder asked for, to check the DLL NGX loaded against.
    feature_dir: Option<PathBuf>,
    device: ID3D12Device,
}

// SAFETY: NGX's calls are free-threaded for distinct features; the shared
// parameter object is behind its lock; D3D12 devices are free-threaded.
unsafe impl Send for Ngx {}
// SAFETY: as above.
unsafe impl Sync for Ngx {}

static NGX: OnceLock<Ngx> = OnceLock::new();
/// The shutdown began: nothing is made or evaluated any more.
static SHUT_DOWN: AtomicBool = AtomicBool::new(false);
/// `Shutdown1` ran: NGX released what was left, and nothing is released
/// any more.
static NGX_GONE: AtomicBool = AtomicBool::new(false);
/// The last feature's video memory, for the title's memory query.
static VIDEO_MEMORY: AtomicU64 = AtomicU64::new(MEMORY_ESTIMATE);

/// What NGX's initialization reads: the feature folder list, the log
/// folder and the feature info. Kept for the process: nothing says NGX
/// copies them rather than keeping the pointers.
struct InitData {
    folders: Vec<Vec<u16>>,
    pointers: Vec<*const u16>,
    logs: Vec<u16>,
    info: FeatureInfo,
}

/// Whether DLSS serves the title's FidelityFX calls.
pub(super) fn active() -> bool {
    NGX.get().is_some()
}

/// Starts NGX on the host device for DLSS at `quality` (an
/// `NVSDK_NGX_PerfQuality_Value`), with `nvngx_dlss.dll` from `dir` if
/// given and NGX's logs in `logs`. A summary of the setup, or why DLSS
/// cannot run.
pub(super) fn start(
    dir: Option<&Path>,
    logs: &Path,
    quality: i32,
    ffx_quality: Option<u32>,
) -> Result<String, String> {
    use windows_sys::Win32::System::LibraryLoader::{
        GetProcAddress, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR, LOAD_LIBRARY_SEARCH_SYSTEM32,
        LoadLibraryExW,
    };
    // VICTORY_DLSS_PRESET: one model for every mode (j, k, l, m, or default
    // for the DLL's choice); without it DLSS 4.5's (see `Presets`).
    let preset = std::env::var("VICTORY_DLSS_PRESET").ok();
    let presets = Presets::chosen(preset.as_deref()).ok_or_else(|| {
        format!("VICTORY_DLSS_PRESET={preset:?}: expected j, k, l, m or default")
    })?;
    // VICTORY_DLSS_JITTER_SIGN: see `jitter_sign`.
    let sign = std::env::var("VICTORY_DLSS_JITTER_SIGN").ok();
    let jitter_sign = jitter_sign(sign.as_deref())
        .ok_or_else(|| format!("VICTORY_DLSS_JITTER_SIGN={sign:?}: expected 1, -1 or x,y"))?;
    // VICTORY_DLSS_REACTIVE=0: the title's reactive mask stays out of DLSS,
    // for an A/B (values made for FSR may keep aliasing DLSS would smooth).
    let reactive = !std::env::var("VICTORY_DLSS_REACTIVE").is_ok_and(|v| v == "0");
    // The feature folder, searched before NGX's defaults.
    let feature_dir = dir
        .map(std::path::absolute)
        .transpose()
        .map_err(|e| e.to_string())?;
    // Without an application copy of the DLL, NGX's own (the driver's, or
    // one its updater fetched) has crashed inside `Init_ProjectID` itself
    // (access violation in `_nvngx.dll`, drivers 610.88 and 617.42, #176),
    // where no result can be checked. So DLSS starts only with a copy in
    // the feature folder or beside the executable, unless
    // VICTORY_DLSS_TRUST_NGX=1 asks for NGX's own anyway.
    let trust_ngx = std::env::var("VICTORY_DLSS_TRUST_NGX").is_ok_and(|v| v == "1");
    if !trust_ngx {
        let exe = std::env::current_exe().ok();
        let beside_exe = exe.as_deref().and_then(Path::parent);
        if app_snippet(feature_dir.as_deref(), beside_exe).is_none() {
            return Err(format!(
                "no {DLL} in the DLSS folder ({}) or beside the executable; NGX's own copy has crashed \
                 inside NVSDK_NGX_D3D12_Init_ProjectID on some drivers. Point VICTORY_DLSS_DIR (or \
                 [paths].dlss_dir) at a folder with NVIDIA's {DLL}, or set VICTORY_DLSS_TRUST_NGX=1 to \
                 try NGX's own copy",
                feature_dir.as_ref().map_or_else(|| "not set".to_owned(), |d| d.display().to_string()),
            ));
        }
    }
    if let Some(dir) = &feature_dir
        && !dir.join(DLL).is_file()
    {
        tracing::warn!(folder = %dir.display(), "DLSS: no {DLL} there; NGX uses another copy");
    }

    // VICTORY_DLSS_NO_UPDATE=1: NGX neither starts its updater nor prefers
    // a model it downloaded, so the run uses the DLL from the feature
    // folder. The core reads `__NGX_DISABLE_UPDATER` (not in the SDK's
    // headers; the core's log says the updater is off); it goes into the
    // environment before the core is loaded, so the core's C runtime
    // starts with it.
    let no_update = std::env::var("VICTORY_DLSS_NO_UPDATE").is_ok_and(|v| v == "1");
    if no_update {
        // SAFETY: Windows' environment block is locked by the system; no
        // title code runs yet.
        unsafe { std::env::set_var("__NGX_DISABLE_UPDATER", "1") };
    }

    let folders: Vec<Option<String>> = CORE_KEYS
        .iter()
        .map(|&(key, value)| registry_string(key, value))
        .collect();
    let search = LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32;
    // The core is never unloaded, not even when DLSS does not start: its
    // updater may still run on a thread of its own.
    let (core, module) = core_candidates(&folders)
        .into_iter()
        .filter(|path| path.is_file())
        .find_map(|path| {
            let name = wide_path(&path);
            // SAFETY: a NUL-terminated full path to the driver's NGX core.
            let module = unsafe { LoadLibraryExW(name.as_ptr(), null_mut(), search) };
            (!module.is_null()).then_some((path, module))
        })
        .ok_or_else(|| {
            format!(
                "no NGX core (_nvngx.dll) in the driver folders {:?}; DLSS needs an NVIDIA RTX driver",
                folders.iter().flatten().collect::<Vec<_>>()
            )
        })?;
    let find = |name: &CStr| {
        // SAFETY: a loaded module and a NUL-terminated name.
        unsafe { GetProcAddress(module, name.as_ptr().cast()) }
            .map(|f| f as *const c_void)
            .ok_or_else(|| format!("{} has no {}", core.display(), name.to_string_lossy()))
    };
    // SAFETY: the core's exports, with nvsdk_ngx.h's signatures (`Init`'s
    // as the core has it).
    let (init, capabilities, create, evaluate, release, shutdown) = unsafe {
        (
            std::mem::transmute::<*const c_void, Init>(find(c"NVSDK_NGX_D3D12_Init_ProjectID")?),
            std::mem::transmute::<*const c_void, Capabilities>(find(
                c"NVSDK_NGX_D3D12_GetCapabilityParameters",
            )?),
            std::mem::transmute::<*const c_void, Create>(find(c"NVSDK_NGX_D3D12_CreateFeature")?),
            std::mem::transmute::<*const c_void, Evaluate>(find(
                c"NVSDK_NGX_D3D12_EvaluateFeature",
            )?),
            std::mem::transmute::<*const c_void, Release>(find(c"NVSDK_NGX_D3D12_ReleaseFeature")?),
            std::mem::transmute::<*const c_void, Shutdown>(find(c"NVSDK_NGX_D3D12_Shutdown1")?),
        )
    };
    let device = victory_gpu::external::host_device().ok_or("there is no host GPU")?;

    std::fs::create_dir_all(logs).map_err(|e| format!("{}: {e}", logs.display()))?;
    let logs = std::path::absolute(logs).map_err(|e| e.to_string())?;
    let data: &'static mut InitData = Box::leak(Box::new(InitData {
        folders: feature_dir.iter().map(|d| wide_path(d)).collect(),
        pointers: Vec::new(),
        logs: wide_path(&logs),
        info: FeatureInfo {
            paths: PathList {
                paths: null(),
                count: 0,
            },
            internal: null_mut(),
            logging: Logging {
                callback: Some(log_line),
                level: LOG_LEVEL,
                other_sinks_off: false,
            },
        },
    }));
    data.pointers = data.folders.iter().map(|p| p.as_ptr()).collect();
    if !data.pointers.is_empty() {
        data.info.paths = PathList {
            paths: data.pointers.as_ptr(),
            count: data.pointers.len() as u32,
        };
    }
    // SAFETY: NGX's initialization on the host device, with data that lives
    // for the process.
    let result = unsafe {
        init(
            PROJECT_ID.as_ptr(),
            ENGINE_CUSTOM,
            ENGINE_VERSION.as_ptr(),
            data.logs.as_ptr(),
            device.as_raw(),
            SDK_VERSION,
            &data.info,
        )
    };
    if !succeeded(result) {
        return Err(format!("NVSDK_NGX_D3D12_Init_ProjectID failed: {}", result_name(result)));
    }
    let fail = |message: String| {
        // SAFETY: NGX was initialized on this device and holds no feature.
        unsafe { shutdown(device.as_raw()) };
        Err(message)
    };
    let mut object = null_mut();
    // SAFETY: an out pointer for NGX's capability parameters.
    let result = unsafe { capabilities(&mut object) };
    if !succeeded(result) || object.is_null() {
        return fail(format!(
            "NVSDK_NGX_D3D12_GetCapabilityParameters failed: {}",
            result_name(result)
        ));
    }
    let Some(layout) = probe(object) else {
        return fail("NGX's parameter object has a method table of an unknown order".into());
    };
    let parameters = Parameters { object, layout };
    if parameters.get_i32(c"SuperSampling.Available") != Some(1) {
        let driver = parameters.get_i32(c"SuperSampling.NeedsUpdatedDriver");
        let major = parameters.get_i32(c"SuperSampling.MinDriverVersionMajor");
        let minor = parameters.get_i32(c"SuperSampling.MinDriverVersionMinor");
        let init = parameters
            .get_i32(c"SuperSampling.FeatureInitResult")
            .map(|r| result_name(r as u32));
        return fail(format!(
            "DLSS Super Resolution is unavailable on this GPU (needs a newer driver: {driver:?}, \
             minimum {major:?}.{minor:?}, feature initialization {init:?})"
        ));
    }
    let summary = format!(
        "core {}, {}, parameter table {layout:?}, presets {} (quality, balanced, \
         performance, ultra performance), NGX quality {quality}, jitter sign {jitter_sign:?}, \
         reactive mask {}, updater {}",
        core.display(),
        Snippet::loaded().map_or_else(|| format!("{DLL} not loaded yet"), |s| s.to_string()),
        presets.letters(),
        if reactive { "in" } else { "left out" },
        if no_update { "off" } else { "NGX's default" },
    );
    NGX.set(Ngx {
        create,
        evaluate,
        release,
        shutdown,
        parameters: Mutex::new(parameters),
        quality,
        ffx_quality,
        presets,
        jitter_sign,
        reactive,
        feature_dir,
        device,
    })
    .map_err(|_| String::from("DLSS already initialized"))?;
    Ok(summary)
}

/// The DLSS DLL in this process, as NGX loaded it.
struct Snippet {
    path: PathBuf,
    version: Option<String>,
}

impl Snippet {
    fn loaded() -> Option<Self> {
        let path = loaded_snippet()?;
        let version = file_version(&path);
        Some(Self { path, version })
    }
}

impl std::fmt::Display for Snippet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let version = self.version.as_deref().unwrap_or("of an unknown version");
        write!(f, "DLSS {version} from {}", self.path.display())
    }
}

/// The application copy of the DLSS DLL NGX would load first: in the
/// feature folder, else beside the executable.
fn app_snippet(feature_dir: Option<&Path>, beside_exe: Option<&Path>) -> Option<PathBuf> {
    feature_dir
        .into_iter()
        .chain(beside_exe)
        .map(|dir| dir.join(DLL))
        .find(|path| path.is_file())
}

/// Whether a module of this path is a DLSS Super Resolution DLL: the one an
/// application ships, or one NGX's updater downloaded into its models.
fn is_dlss_snippet(path: &Path) -> bool {
    let named = path
        .file_name()
        .is_some_and(|name| name.eq_ignore_ascii_case(DLL));
    named || folded(path).contains(r"\ngx\models\dlss\")
}

/// A path for comparing: lower case, backslashes only (NGX loads
/// `folder/nvngx_dlss.dll`), no trailing one.
fn folded(path: &Path) -> String {
    let text = path.to_string_lossy().to_ascii_lowercase().replace('/', r"\");
    text.trim_end_matches('\\').to_owned()
}

/// Whether the DLL at `path` is the one in the feature folder `dir`.
fn from_folder(path: &Path, dir: &Path) -> bool {
    path.parent().is_some_and(|parent| folded(parent) == folded(dir))
}

/// The path of the DLSS DLL loaded in this process, if any.
fn loaded_snippet() -> Option<PathBuf> {
    use std::os::windows::ffi::OsStringExt;
    use windows_sys::Win32::System::LibraryLoader::GetModuleFileNameW;
    use windows_sys::Win32::System::ProcessStatus::K32EnumProcessModules;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut modules = vec![null_mut(); 512];
    loop {
        let room = (modules.len() * size_of::<*mut c_void>()) as u32;
        let mut needed = 0u32;
        // SAFETY: room for `room` bytes of module handles.
        let listed = unsafe {
            K32EnumProcessModules(GetCurrentProcess(), modules.as_mut_ptr(), room, &mut needed)
        };
        if listed == 0 {
            return None;
        }
        if needed <= room {
            modules.truncate(needed as usize / size_of::<*mut c_void>());
            break;
        }
        modules.resize(needed as usize / size_of::<*mut c_void>() + 16, null_mut());
    }
    let mut name = vec![0u16; 1024];
    modules.into_iter().find_map(|module| {
        // SAFETY: a module of this process and a buffer of `name.len()` units.
        let length = unsafe { GetModuleFileNameW(module, name.as_mut_ptr(), name.len() as u32) };
        let path = PathBuf::from(std::ffi::OsString::from_wide(&name[..length as usize]));
        is_dlss_snippet(&path).then_some(path)
    })
}

// Locks, always taken in this order: a context's `feature`, NGX's
// `parameters`, a host list's own state (taken inside `Recording::keep`
// and when a list is reset), `PENDING`. `PENDING` is held for no call
// into NGX or anything else, so no order can close a cycle; every NGX
// call (make, evaluate, release, shut down) is made under `parameters`,
// so no two overlap.

/// A DLSS feature NGX made. When the last reference goes (the context's,
/// or that of a host list that recorded it, held until the list is reset
/// after its last submission ran), the handle waits in [`PENDING`] for the
/// next release under the parameter lock: the last reference may go on
/// any thread, inside a list's lock.
struct Handle(*mut c_void);

// SAFETY: NGX's feature handles are not tied to a thread; dropping one
// only queues it.
unsafe impl Send for Handle {}
// SAFETY: as above; a shared handle is only read.
unsafe impl Sync for Handle {}

/// Features no one refers to any more, for [`release_pending`].
static PENDING: Mutex<Vec<usize>> = Mutex::new(Vec::new());

impl Drop for Handle {
    fn drop(&mut self) {
        if !self.0.is_null() {
            PENDING
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(self.0 as usize);
        }
    }
}

/// Releases the features in [`PENDING`]. The caller holds the parameter
/// lock (`_held`), so a release overlaps no other NGX call; after
/// `Shutdown1`, NGX has released them itself.
fn release_pending(ngx: &Ngx, _held: &Parameters) {
    if NGX_GONE.load(Ordering::Acquire) {
        return;
    }
    let pending = std::mem::take(&mut *PENDING.lock().unwrap_or_else(|e| e.into_inner()));
    for handle in pending {
        // SAFETY: a feature NGX made, released once, in no list left to run.
        let result = unsafe { (ngx.release)(handle as *mut c_void) };
        if !succeeded(result) {
            tracing::warn!(result = result_name(result), "DLSS: releasing a feature failed");
        }
    }
}

/// Who upscales a context's frames.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Dlss,
    /// DLSS failed; the host's FidelityFX upscales, with this context.
    Fsr(usize),
    /// DLSS failed and nothing else can: dispatches fail.
    Stopped,
}

/// One context's DLSS feature, made at its first dispatch (NGX records its
/// setup on a command list), again when the output size changes.
struct Feature {
    handle: Option<Arc<Handle>>,
    output: [u32; 2],
    mode: Mode,
    live: bool,
}

/// FidelityFX's `ffxCreateContextDescUpscaleVersion`: the API version the
/// title was built against, by which a provider reads its descriptions.
const UPSCALE_VERSION: u64 = 0x1_000b;

/// As ffx_upscale.h (FidelityFX SDK 2.4.0) declares it: the header, then a
/// `uint32_t` version (`FFX_UPSCALER_MAKE_VERSION`; 4.1.1 is 0x0100_1001).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct UpscaleVersion {
    header: Header,
    version: u32,
}

const _: () = assert!(size_of::<UpscaleVersion>() == 24);

/// The version description in a creation chain from `first` on, if any,
/// on its own.
///
/// # Safety
/// `first` is null or the start of a chain of FidelityFX descriptions.
unsafe fn version_in(first: *const Header) -> Option<UpscaleVersion> {
    let mut at = first;
    // A bound against a chain that loops.
    for _ in 0..32 {
        // SAFETY: a description of the chain, by the caller.
        let header = unsafe { at.as_ref()? };
        if header.kind == UPSCALE_VERSION {
            // SAFETY: a version description, by its type.
            let mut version = unsafe { at.cast::<UpscaleVersion>().read() };
            version.header.next = null_mut();
            return Some(version);
        }
        at = header.next;
    }
    None
}

struct Context {
    /// The title's description, alone, for an FSR context should DLSS fail.
    upscale: CreateUpscale,
    /// The version description of the title's chain, which the FSR
    /// context's chain carries again; FidelityFX may read it until the
    /// context is destroyed, so it lives as long as this.
    version: Option<Box<UpscaleVersion>>,
    feature: Mutex<Feature>,
    frames: AtomicU64,
    /// Dispatches refused in a row; at [`REFUSALS`], DLSS gives up.
    refusals: AtomicU32,
}

/// Consecutive refused dispatches (resources DLSS cannot read, say) after
/// which a context goes to the fallback as if NGX had failed.
const REFUSALS: u32 = 8;

// SAFETY: the feature is only used under its lock; the title's message
// callback in `upscale` is a function, callable from any thread; the
// version description is only read.
unsafe impl Send for Context {}
// SAFETY: as above.
unsafe impl Sync for Context {}

static CONTEXTS: Mutex<BTreeMap<usize, Arc<Context>>> = Mutex::new(BTreeMap::new());
/// Our opaque context values (the title never reads through them).
static NEXT_CONTEXT: AtomicU64 = AtomicU64::new(0x0d15_5000);

fn context_of(context: *mut *mut c_void) -> Option<Arc<Context>> {
    if context.is_null() {
        return None;
    }
    // SAFETY: the title passes its context slot.
    let handle = unsafe { *context } as usize;
    CONTEXTS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&handle)
        .cloned()
}

/// The context's FSR fallback, if DLSS handed it over.
fn fallback_of(ctx: &Context) -> Option<(&'static super::Host, *mut c_void)> {
    let feature = ctx.feature.lock().unwrap_or_else(|e| e.into_inner());
    match feature.mode {
        Mode::Fsr(context) => super::HOST.get().map(|host| (host, context as *mut c_void)),
        _ => None,
    }
}

pub(super) fn create_context(context: *mut *mut c_void, desc: *mut Header) -> u32 {
    if context.is_null() || desc.is_null() {
        return PARAMETER;
    }
    // SAFETY: the title passes a chain of descriptions.
    let kind = unsafe { (*desc).kind };
    if kind != CREATE_UPSCALE {
        once!(
            warn,
            kind = format_args!("{kind:#x}"),
            "DLSS: only the upscaler runs on the host; another effect was refused"
        );
        return NO_PROVIDER;
    }
    // SAFETY: an upscaler description, by its type, ahead of the chain.
    let (mut upscale, version) = unsafe {
        let upscale = desc.cast::<CreateUpscale>().read();
        (upscale, version_in(upscale.header.next))
    };
    upscale.header.next = null_mut();
    let handle = NEXT_CONTEXT.fetch_add(0x10, Ordering::Relaxed) as usize;
    CONTEXTS.lock().unwrap_or_else(|e| e.into_inner()).insert(
        handle,
        Arc::new(Context {
            upscale,
            version: version.map(Box::new),
            feature: Mutex::new(Feature {
                handle: None,
                output: [0, 0],
                mode: Mode::Dlss,
                live: true,
            }),
            frames: AtomicU64::new(0),
            refusals: AtomicU32::new(0),
        }),
    );
    // SAFETY: the title's context slot.
    unsafe { *context = handle as *mut c_void };
    tracing::info!(
        render = ?upscale.max_render,
        output = ?upscale.max_output,
        flags = format_args!("{:#x}", upscale.flags),
        api_version = ?version.map(|v| format!("{:#x}", v.version)),
        "DLSS: the title's upscaler context (its feature is made at the first dispatch)"
    );
    OK
}

pub(super) fn destroy_context(context: *mut *mut c_void) -> u32 {
    let Some(ctx) = context_of(context) else {
        return PARAMETER;
    };
    let mut feature = ctx.feature.lock().unwrap_or_else(|e| e.into_inner());
    if !feature.live {
        return PARAMETER;
    }
    if let (Mode::Fsr(fallback), Some(host)) = (feature.mode, super::HOST.get()) {
        let mut fallback = fallback as *mut c_void;
        super::destroy_on_host(host, &mut fallback, null());
    }
    // Lists that recorded the feature keep it until they have run; if
    // none does, it is released now.
    feature.handle = None;
    feature.live = false;
    if let Some(ngx) = NGX.get() {
        release_pending(ngx, &ngx.parameters.lock().unwrap_or_else(|e| e.into_inner()));
    }
    drop(feature);
    // SAFETY: the title's context slot; it reads as released afterwards,
    // as FidelityFX leaves it.
    let handle = unsafe { std::mem::replace(&mut *context, null_mut()) } as usize;
    CONTEXTS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .remove(&handle);
    tracing::info!(
        frames = ctx.frames.load(Ordering::Relaxed),
        "DLSS: the title released its upscaler context"
    );
    OK
}

/// Shuts NGX down once nothing evaluates any more and the GPU has run
/// what was submitted: the features lists let go of while it waits are
/// released first, `Shutdown1` takes the rest. Later calls do nothing. (A
/// list recorded before and never submitted is not run before the process
/// ends.)
pub(super) fn shut_down() {
    let Some(ngx) = NGX.get() else { return };
    {
        // Under the lock every evaluation holds and checks the flag under:
        // none is under way, and none starts after.
        let _evaluations = ngx.parameters.lock().unwrap_or_else(|e| e.into_inner());
        if SHUT_DOWN.swap(true, Ordering::AcqRel) {
            return;
        }
    }
    victory_gpu::external::finish_submitted();
    let parameters = ngx.parameters.lock().unwrap_or_else(|e| e.into_inner());
    release_pending(ngx, &parameters);
    // SAFETY: NGX was started on this device; nothing of it is used after.
    let result = unsafe { (ngx.shutdown)(ngx.device.as_raw()) };
    NGX_GONE.store(true, Ordering::Release);
    drop(parameters);
    tracing::info!(result = result_name(result), "DLSS: NGX shut down");
}

#[repr(C)]
struct PhaseCount {
    header: Header,
    render_width: u32,
    output_width: u32,
    count: *mut i32,
}

#[repr(C)]
struct JitterOffset {
    header: Header,
    index: i32,
    phases: i32,
    x: *mut f32,
    y: *mut f32,
}

#[repr(C)]
struct MemoryUsage {
    header: Header,
    usage: *mut [u64; 2],
}

/// Writes through one of the title's output pointers, if it gave one.
fn put<T>(to: *mut T, value: T) {
    if !to.is_null() {
        // SAFETY: the title's output field for this query.
        unsafe { to.write(value) };
    }
}

/// Settings of a context FSR took over go to FSR; DLSS has no counterpart
/// to FidelityFX's keys (debug views, tuning).
pub(super) fn configure(context: *mut *mut c_void, desc: *mut Header) -> u32 {
    match context_of(context).as_deref().and_then(fallback_of) {
        Some((host, mut fallback)) => {
            // SAFETY: the FSR context made for this one, and the title's
            // descriptions.
            unsafe { (host.configure)(&mut fallback, desc) }
        }
        None => OK,
    }
}

pub(super) fn query(context: *mut *mut c_void, desc: *mut Header) -> u32 {
    if desc.is_null() {
        return PARAMETER;
    }
    if let Some((host, mut fallback)) = context_of(context).as_deref().and_then(fallback_of) {
        return super::query_on_host(host, &mut fallback, desc);
    }
    query_dlss(desc, NGX.get().and_then(|ngx| ngx.ffx_quality))
}

fn query_dlss(desc: *mut Header, quality: Option<u32>) -> u32 {
    // SAFETY: the title passes a description; each type has its layout.
    unsafe {
        match (*desc).kind {
            GET_VERSIONS => {
                // The count is the arrays' room on entry, the number on exit.
                let q = &*desc.cast::<GetVersions>();
                if q.count.is_null() {
                    return PARAMETER;
                }
                if *q.count > 0 {
                    put(q.ids, PROVIDER_ID);
                    put(q.names, PROVIDER_NAME.as_ptr());
                }
                *q.count = 1;
            }
            GET_PROVIDER_VERSION => {
                let q = &mut *desc.cast::<ProviderVersion>();
                q.id = PROVIDER_ID;
                q.name = PROVIDER_NAME.as_ptr();
            }
            GET_UPSCALE_RATIO => {
                let q = &*desc.cast::<RatioQuery>();
                let Some(ratio) = upscale_ratio(quality.unwrap_or(q.quality)) else {
                    return PARAMETER;
                };
                put(q.ratio, ratio);
            }
            GET_RENDER_RESOLUTION => {
                let q = &*desc.cast::<RenderResolution>();
                let Some(ratio) = upscale_ratio(quality.unwrap_or(q.quality)) else {
                    return PARAMETER;
                };
                put(q.width, (q.display[0] as f32 / ratio) as u32);
                put(q.height, (q.display[1] as f32 / ratio) as u32);
            }
            GET_JITTER_PHASE_COUNT => {
                let q = &*desc.cast::<PhaseCount>();
                put(q.count, phase_count(q.render_width, q.output_width));
            }
            GET_JITTER_OFFSET => {
                let q = &*desc.cast::<JitterOffset>();
                let [x, y] = jitter_offset(q.index, q.phases);
                put(q.x, x);
                put(q.y, y);
            }
            GPU_MEMORY_USAGE => {
                let bytes = VIDEO_MEMORY.load(Ordering::Relaxed);
                put((*desc.cast::<MemoryUsage>()).usage, [bytes, 0]);
            }
            MEMORY_USAGE_WITH_DEVICE => {
                let bytes = VIDEO_MEMORY.load(Ordering::Relaxed);
                let usage = (*desc.cast::<MemoryUsageWithDevice>()).usage;
                put(usage.cast::<[u64; 2]>(), [bytes, 0]);
            }
            kind => {
                once!(
                    warn,
                    kind = format_args!("{kind:#x}"),
                    "DLSS: a FidelityFX query it does not answer"
                );
                return UNKNOWN_TYPE;
            }
        }
    }
    OK
}

/// Why a dispatch failed on the title's list.
enum DispatchError {
    /// The dispatch itself was unusable (logged once, with the detail): the
    /// code for the title and what was wrong. The next may do.
    Refused(u32, &'static str),
    /// NGX failed: DLSS stops for this context.
    Failed(String),
}

impl From<SetupError> for DispatchError {
    fn from(error: SetupError) -> Self {
        match error {
            SetupError::Aliased(index) => {
                once!(warn, index, "DLSS: aliased arguments require conflicting resource states");
                Self::Refused(PARAMETER, "conflicting states for one texture")
            }
            SetupError::Input(index) => {
                once!(warn, index, "DLSS: an input could not be prepared on the title's list");
                Self::Refused(PARAMETER, "an input not prepared")
            }
        }
    }
}

type Outcome = Result<(), DispatchError>;

/// A refused dispatch, counted: at [`REFUSALS`] in a row, DLSS stops for
/// the context and the fallback (if any) takes this dispatch already.
fn refused(ctx: &Context, desc: *mut Header, code: u32, why: &str) -> u32 {
    // Refusals while NGX shuts down or the context goes away say nothing
    // about DLSS, and need no fallback.
    let ending = SHUT_DOWN.load(Ordering::Acquire)
        || !ctx.feature.lock().unwrap_or_else(|e| e.into_inner()).live;
    if ending || ctx.refusals.fetch_add(1, Ordering::Relaxed) + 1 < REFUSALS {
        return code;
    }
    let why = format!("{REFUSALS} dispatches in a row refused, the last as {why}");
    match stop(ctx, &why, false) {
        Mode::Fsr(_) => dispatch_fallback(ctx, desc),
        _ => code,
    }
}

pub(super) fn dispatch(context: *mut *mut c_void, desc: *mut Header) -> u32 {
    let Some(ngx) = NGX.get() else {
        return NO_PROVIDER;
    };
    if desc.is_null() {
        return PARAMETER;
    }
    // SAFETY: the title passes a description.
    let kind = unsafe { (*desc).kind };
    if kind != DISPATCH_UPSCALE {
        once!(
            warn,
            kind = format_args!("{kind:#x}"),
            "DLSS: only upscaling runs on the host; a dispatch was refused"
        );
        return UNKNOWN_TYPE;
    }
    let Some(ctx) = context_of(context) else {
        once!(warn, "DLSS: the dispatch names an unregistered context");
        return PARAMETER;
    };
    let mode = ctx.feature.lock().unwrap_or_else(|e| e.into_inner()).mode;
    match mode {
        Mode::Dlss if SHUT_DOWN.load(Ordering::Acquire) => return NO_PROVIDER,
        Mode::Dlss => {}
        Mode::Fsr(_) => return dispatch_fallback(&ctx, desc),
        Mode::Stopped => return NO_PROVIDER,
    }
    // SAFETY: a plain query; after a device removal nothing more is made
    // or evaluated, and nothing can run what was.
    if let Err(error) = unsafe { ngx.device.GetDeviceRemovedReason() } {
        stop(&ctx, &format!("the device was removed ({error})"), true);
        return NO_PROVIDER;
    }
    // SAFETY: an upscaling, by its type.
    let dispatch = unsafe { desc.cast::<DispatchUpscale>().read() };
    if let Some(index) = [0, 1, 2, 6]
        .into_iter()
        .find(|&i| dispatch.resources[i].resource.is_null())
    {
        once!(warn, index, "DLSS: the dispatch is missing a required texture");
        return refused(&ctx, desc, PARAMETER, "a missing texture");
    }
    let Some(output_size) = dispatch_output(
        dispatch.render_size,
        dispatch.upscale_size,
        ctx.upscale.max_render,
        ctx.upscale.max_output,
    ) else {
        once!(warn, render = ?dispatch.render_size, output = ?dispatch.upscale_size,
            max_render = ?ctx.upscale.max_render, max_output = ?ctx.upscale.max_output,
            "DLSS: dispatch dimensions outside the context's");
        return refused(&ctx, desc, PARAMETER, "sizes outside the context's");
    };
    let list = dispatch.list.cast::<Object>().cast_const();
    let outcome = victory_gpu::external::record(list, |list| {
        record_dispatch(list, ngx, &ctx, &dispatch, output_size)
    });
    match outcome {
        Some(Ok(())) => {
            ctx.refusals.store(0, Ordering::Relaxed);
            OK
        }
        Some(Err(DispatchError::Refused(code, why))) => refused(&ctx, desc, code, why),
        Some(Err(DispatchError::Failed(why))) => match stop(&ctx, &why, false) {
            // This frame too: FSR records on the same list after us.
            Mode::Fsr(_) => dispatch_fallback(&ctx, desc),
            _ => NO_PROVIDER,
        },
        None => {
            once!(warn, "DLSS: could not record to the title's command list");
            refused(&ctx, desc, PARAMETER, "no title list to record on")
        }
    }
}

fn dispatch_fallback(ctx: &Context, desc: *mut Header) -> u32 {
    match fallback_of(ctx) {
        Some((host, mut fallback)) => super::dispatch_on_host(host, &mut fallback, desc),
        None => NO_PROVIDER,
    }
}

/// Stops DLSS for a context, once: its feature goes (when the lists that
/// recorded it have run), and FSR takes the context over if its folder was
/// given, unless the device was removed. What upscales from now on.
fn stop(ctx: &Context, why: &str, device_removed: bool) -> Mode {
    let mut feature = ctx.feature.lock().unwrap_or_else(|e| e.into_inner());
    if feature.mode != Mode::Dlss {
        return feature.mode;
    }
    feature.handle = None;
    let host = super::HOST.get().filter(|_| !device_removed);
    feature.mode = host.map_or(Mode::Stopped, |host| {
        // The title's description and its API version, as the title chained
        // them (the console's backend aside).
        let mut upscale = ctx.upscale;
        upscale.header.next = ctx
            .version
            .as_deref()
            .map_or(null_mut(), |v| (&raw const v.header).cast_mut());
        let mut fallback = null_mut();
        let code = super::create_on_host(host, &mut fallback, &mut upscale.header, null());
        if code == OK && !fallback.is_null() {
            Mode::Fsr(fallback as usize)
        } else {
            tracing::error!(code, "DLSS: the FSR fallback refused the title's upscaler");
            Mode::Stopped
        }
    });
    let fallback_version = match feature.mode {
        Mode::Fsr(_) => host.map(|h| h.version),
        _ => None,
    };
    tracing::error!(
        reason = why,
        ?fallback_version,
        frames = ctx.frames.load(Ordering::Relaxed),
        "DLSS: stopped for this upscaler context"
    );
    feature.mode
}

/// The state DLSS gets a resource the title declared in `declared` (host
/// state): as it is when that already allows DLSS's access.
fn state_for(output: bool, declared: Option<u32>) -> u32 {
    match (output, declared) {
        (true, _) => UNORDERED_ACCESS,
        (false, Some(state)) if state & COMPUTE_READ != 0 => state,
        (false, _) => COMPUTE_READ,
    }
}

/// The FidelityFX resources DLSS reads, then its output: colour, depth,
/// motion vectors, the exposure texture unless DLSS finds the exposure
/// itself, the reactive mask (DLSS's bias towards the current colour) if
/// `reactive`. FidelityFX's transparency and composition mask has no DLSS
/// input (NGX reserves `TransparencyMask`).
fn inputs(
    dispatch: &DispatchUpscale,
    flags: u32,
    reactive: bool,
) -> impl Iterator<Item = usize> + '_ {
    [0, 1, 2, 3, 4, 6].into_iter().filter(move |&index| match index {
        3 => flags & FFX_AUTO_EXPOSURE == 0 && !dispatch.resources[3].resource.is_null(),
        4 => reactive && !dispatch.resources[4].resource.is_null(),
        _ => true,
    })
}

fn record_dispatch(
    list: &victory_gpu::external::Recording<'_>,
    ngx: &Ngx,
    ctx: &Context,
    dispatch: &DispatchUpscale,
    output_size: [u32; 2],
) -> Outcome {
    let mut feature = ctx.feature.lock().unwrap_or_else(|e| e.into_inner());
    if !feature.live || feature.mode != Mode::Dlss {
        return Err(DispatchError::Refused(PARAMETER, "a context DLSS no longer serves"));
    }
    let flags = ctx.upscale.flags;
    let mut bindings = [None; 7];
    // Two title arguments can refer to the same host texture. Validate all
    // bindings before recording any transitions: one backing cannot be both
    // an input and the output, or be returned in two different states.
    for index in inputs(dispatch, flags, ngx.reactive) {
        let resource = dispatch.resources[index].resource.cast::<Object>().cast_const();
        let declared = host_state(dispatch.resources[index].state);
        let used = state_for(index == 6, declared);
        let Some(before) = list.texture_binding(resource) else {
            once!(warn, index, "DLSS: a resource has no host texture; the upscaling did not run");
            return Err(DispatchError::Refused(PARAMETER, "a resource without a host texture"));
        };
        bindings[index] = Some(Binding {
            before,
            declared: declared.unwrap_or(used),
            used,
            on_failure: declared.or(before.1),
        });
    }
    let outcome = with_resource_states(
        bindings,
        |index| {
            list.prepare_input(dispatch.resources[index].resource.cast::<Object>())
                .is_some()
        },
        |index, used| {
            // Preparation may replace a mip-trimmed backing with a full one.
            let resource = dispatch.resources[index].resource.cast::<Object>();
            let Some((texture, state)) = list.texture(resource, used) else {
                return Err(DispatchError::Refused(
                    PARAMETER,
                    "a resource without a host texture",
                ));
            };
            if state != used {
                once!(
                    warn,
                    index,
                    state,
                    used,
                    "DLSS: a resource cannot be in the state DLSS uses"
                );
                return Err(DispatchError::Refused(
                    PARAMETER,
                    "a resource in a state DLSS cannot use",
                ));
            }
            // SAFETY: a live texture.
            let desc = unsafe { texture.GetDesc() };
            let needed = minimum_extent(index, flags, dispatch.render_size, output_size);
            if desc.Width < u64::from(needed[0]) || desc.Height < needed[1] {
                once!(
                    warn,
                    index,
                    ?needed,
                    width = desc.Width,
                    height = desc.Height,
                    "DLSS: a resource cannot cover the dispatch rectangle"
                );
                return Err(DispatchError::Refused(
                    PARAMETER,
                    "a resource smaller than the dispatch",
                ));
            }
            Ok(texture)
        },
        |index| list.existing_texture_binding(dispatch.resources[index].resource.cast::<Object>()),
        |index, state| {
            let _ = list.texture(dispatch.resources[index].resource.cast::<Object>(), state);
        },
        |textures| {
            evaluate(
                list,
                ngx,
                ctx,
                &mut feature,
                dispatch,
                output_size,
                textures,
            )
        },
    );
    if outcome.is_ok() {
        victory_gpu::external::output_written(dispatch.resources[6].resource.cast::<Object>());
    }
    outcome
}

fn evaluate(
    list: &victory_gpu::external::Recording<'_>,
    ngx: &Ngx,
    ctx: &Context,
    feature: &mut Feature,
    dispatch: &DispatchUpscale,
    output_size: [u32; 2],
    textures: &[Option<ID3D12Resource>; 7],
) -> Outcome {
    let raw = |index: usize| textures[index].as_ref().map_or(null_mut(), |t| t.as_raw());
    // Flushes the transitions above before NGX records.
    let commands = list.commands().as_raw();
    let parameters = ngx.parameters.lock().unwrap_or_else(|e| e.into_inner());
    // Checked under the lock the shutdown takes: nothing is made or
    // evaluated after it.
    if SHUT_DOWN.load(Ordering::Acquire) {
        return Err(DispatchError::Refused(NO_PROVIDER, "a dispatch after NGX shut down"));
    }
    // Features the lists have let go of since the last dispatch.
    release_pending(ngx, &parameters);
    if feature.output != output_size && feature.handle.is_some() {
        tracing::info!(output = ?output_size, before = ?feature.output,
            "DLSS: the output size changed; making a new feature");
        // Lists that recorded the old one keep it until they have run.
        feature.handle = None;
    }
    if feature.handle.is_none() {
        let flags = create_flags(ctx.upscale.flags);
        parameters.set_u32(c"Width", ctx.upscale.max_render[0]);
        parameters.set_u32(c"Height", ctx.upscale.max_render[1]);
        parameters.set_u32(c"OutWidth", output_size[0]);
        parameters.set_u32(c"OutHeight", output_size[1]);
        parameters.set_i32(c"PerfQualityValue", ngx.quality);
        // Older DLLs still read it (NGX_DLSS_GET_OPTIMAL_SETTINGS).
        parameters.set_i32(c"RTXValue", 0);
        parameters.set_i32(c"DLSS.Feature.Create.Flags", flags);
        parameters.set_i32(c"DLSS.Enable.Output.Subrects", 0);
        parameters.set_u32(c"CreationNodeMask", 1);
        parameters.set_u32(c"VisibilityNodeMask", 1);
        for (hint, preset) in ngx.presets.hints() {
            parameters.set_u32(hint, preset);
        }
        let mut handle = null_mut();
        // SAFETY: an open host list of the device NGX was started on.
        let result =
            unsafe { (ngx.create)(commands, SUPER_SAMPLING, parameters.object, &mut handle) };
        if !succeeded(result) || handle.is_null() {
            return Err(DispatchError::Failed(format!(
                "NVSDK_NGX_D3D12_CreateFeature failed: {} (render {:?}, output {output_size:?}, \
                 DLSS flags {flags:#x})",
                result_name(result),
                ctx.upscale.max_render,
            )));
        }
        feature.handle = Some(Arc::new(Handle(handle)));
        feature.output = output_size;
        let memory = parameters.video_memory();
        if let Some(bytes) = memory {
            VIDEO_MEMORY.store(bytes, Ordering::Relaxed);
        }
        let snippet = Snippet::loaded();
        tracing::info!(
            dll = snippet.as_ref().map_or_else(|| String::from("not found"), |s| s.to_string()),
            presets = ngx.presets.letters(),
            quality = ngx.quality,
            render = ?ctx.upscale.max_render,
            output = ?output_size,
            ffx_flags = format_args!("{:#x}", ctx.upscale.flags),
            dlss_flags = format_args!("{flags:#x}"),
            video_memory = ?memory,
            "DLSS: feature created"
        );
        if let Some(dir) = &ngx.feature_dir
            && !snippet.as_ref().is_some_and(|s| from_folder(&s.path, dir))
        {
            once!(warn, folder = %dir.display(),
                "DLSS: NGX runs another DLL than the feature folder's (VICTORY_DLSS_NO_UPDATE=1 keeps that one)");
        }
    }
    let Some(shared) = feature.handle.clone() else {
        return Err(DispatchError::Refused(PARAMETER, "no feature"));
    };
    let handle = shared.0;
    // The list holds the feature until it has run, whatever becomes of the
    // context (a new feature, a fallback, its destruction).
    list.keep(shared);
    let values = Evaluation::of(dispatch, ngx.jitter_sign);
    parameters.set_resource(c"Color", raw(0));
    parameters.set_resource(c"Depth", raw(1));
    parameters.set_resource(c"MotionVectors", raw(2));
    parameters.set_resource(c"ExposureTexture", raw(3));
    parameters.set_resource(c"DLSS.Input.Bias.Current.Color.Mask", raw(4));
    parameters.set_resource(c"Output", raw(6));
    parameters.set_f32(c"Jitter.Offset.X", values.jitter[0]);
    parameters.set_f32(c"Jitter.Offset.Y", values.jitter[1]);
    parameters.set_f32(c"MV.Scale.X", values.motion_scale[0]);
    parameters.set_f32(c"MV.Scale.Y", values.motion_scale[1]);
    parameters.set_i32(c"Reset", i32::from(values.reset));
    parameters.set_f32(c"Sharpness", 0.0);
    parameters.set_u32(c"DLSS.Render.Subrect.Dimensions.Width", values.render[0]);
    parameters.set_u32(c"DLSS.Render.Subrect.Dimensions.Height", values.render[1]);
    parameters.set_f32(c"DLSS.Pre.Exposure", values.pre_exposure);
    parameters.set_f32(c"DLSS.Exposure.Scale", 1.0);
    parameters.set_f32(c"FrameTimeDeltaInMsec", values.frame_ms);
    // SAFETY: the feature, its inputs in the states set above, an open list.
    let result = unsafe { (ngx.evaluate)(commands, handle, parameters.object, 0) };
    drop(parameters);
    if !succeeded(result) {
        return Err(DispatchError::Failed(format!(
            "NVSDK_NGX_D3D12_EvaluateFeature failed: {}",
            result_name(result)
        )));
    }
    if victory_gpu::framegen::requested()
        && let (Some(depth), Some(motion)) = (&textures[1], &textures[2])
        && depth.as_raw() != motion.as_raw()
    {
        let read_state = |index: usize| {
            match state_for(false, host_state(dispatch.resources[index].state)) {
                0xc0 => 0xc, // FidelityFX compute and pixel read.
                _ => 0x4,    // FidelityFX compute read.
            }
        };
        list.prepare_framegen(victory_gpu::framegen::Inputs {
            depth,
            motion,
            depth_desc: dispatch.resources[1].description,
            motion_desc: dispatch.resources[2].description,
            depth_state: read_state(1),
            motion_state: read_state(2),
            max_render: ctx.upscale.max_render,
            render_size: values.render,
            upscale_flags: ctx.upscale.flags,
            jitter: dispatch.jitter,
            motion_scale: dispatch.motion_scale,
            frame_ms: values.frame_ms,
            reset: values.reset,
            camera: std::array::from_fn(|index| f32::from_bits(dispatch.settings[5 + index])),
        });
    }
    let frames = ctx.frames.fetch_add(1, Ordering::Relaxed) + 1;
    if frames == 1 {
        tracing::info!(
            render = ?values.render,
            output = ?output_size,
            jitter = ?values.jitter,
            motion_vector_scale = ?values.motion_scale,
            pre_exposure = values.pre_exposure,
            frame_ms = values.frame_ms,
            reset = values.reset,
            exposure_texture = textures[3].is_some(),
            reactive_mask = textures[4].is_some(),
            "DLSS: evaluated the title's first frame"
        );
    } else if frames == 1000 {
        tracing::info!("DLSS: 1000 frames evaluated");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn application_snippet_prefers_the_configured_folder() {
        let tmp = tempfile::tempdir().unwrap();
        let feature = tmp.path().join("feature");
        let executable = tmp.path().join("app");
        std::fs::create_dir(&feature).unwrap();
        std::fs::create_dir(&executable).unwrap();

        assert_eq!(app_snippet(Some(&feature), Some(&executable)), None);

        let beside_exe = executable.join(DLL);
        std::fs::write(&beside_exe, b"fixture").unwrap();
        assert_eq!(
            app_snippet(Some(&feature), Some(&executable)),
            Some(beside_exe)
        );

        let configured = feature.join(DLL);
        std::fs::write(&configured, b"fixture").unwrap();
        assert_eq!(
            app_snippet(Some(&feature), Some(&executable)),
            Some(configured)
        );
    }
    use crate::modules::fidelityfx::Resource;

    #[test]
    fn the_titles_flags_become_dlss_flags() {
        // The title's: HDR, inverted depth, auto exposure, motion vectors at
        // render size without the jitter in them.
        assert_eq!(
            create_flags(0x29),
            DLSS_HDR | DLSS_DEPTH_INVERTED | DLSS_AUTO_EXPOSURE | DLSS_MV_LOW_RES
        );
        assert_eq!(create_flags(0x2 | 0x4), DLSS_MV_JITTERED);
        // Infinite depth has no DLSS flag.
        assert_eq!(create_flags(0x10), DLSS_MV_LOW_RES);
    }

    #[test]
    fn dlss_4_5_presets_unless_chosen() {
        let default = Presets::chosen(None).unwrap();
        assert_eq!(default.letters(), "MMML");
        assert_eq!(default.dlaa, PRESET_M);
        assert_eq!(Presets::chosen(Some(" k ")).unwrap().letters(), "KKKK");
        assert_eq!(Presets::chosen(Some("default")).unwrap().letters(), "----");
        assert_eq!(Presets::chosen(Some("a")), None);
        let hints = default.hints();
        assert_eq!(hints[2], (c"DLSS.Hint.Render.Preset.Quality", 13));
        assert_eq!(hints[5], (c"DLSS.Hint.Render.Preset.UltraPerformance", 12));
    }

    #[test]
    fn jitter_signs() {
        assert_eq!(jitter_sign(None), Some([1.0, 1.0]));
        assert_eq!(jitter_sign(Some("-1")), Some([-1.0, -1.0]));
        assert_eq!(jitter_sign(Some("1,-1")), Some([1.0, -1.0]));
        assert_eq!(jitter_sign(Some("2")), None);
    }

    fn dispatch_of(
        jitter: [f32; 2],
        scale: [f32; 2],
        frame_ms: f32,
        exposure: f32,
        reset: u32,
    ) -> DispatchUpscale {
        let none = Resource {
            resource: null_mut(),
            description: [0; 8],
            state: 0,
        };
        let mut settings = [0; 10];
        settings[2] = frame_ms.to_bits();
        settings[3] = exposure.to_bits();
        settings[4] = reset;
        DispatchUpscale {
            header: Header {
                kind: DISPATCH_UPSCALE,
                next: null_mut(),
            },
            list: null_mut(),
            resources: [none; 7],
            jitter,
            motion_scale: scale,
            render_size: [2560, 1440],
            upscale_size: [3840, 2160],
            settings,
        }
    }

    #[test]
    fn a_dispatch_as_dlss_values() {
        // The jitter passes as given, or flipped per axis for A/B.
        let d = dispatch_of([-0.25, 0.125], [2560.0, -1440.0], 16.6, 0.0, 1);
        let values = Evaluation::of(&d, jitter_sign(None).unwrap());
        assert_eq!(values.jitter, [-0.25, 0.125]);
        assert_eq!(Evaluation::of(&d, [1.0, -1.0]).jitter, [-0.25, -0.125]);
        // The scale passes with its sign: FidelityFX's is in render pixels too.
        assert_eq!(values.motion_scale, [2560.0, -1440.0]);
        assert_eq!(values.render, [2560, 1440]);
        assert!(values.reset);
        assert_eq!(values.frame_ms, 16.6);
        assert_eq!(values.pre_exposure, 1.0);
        // A bool in the low byte; the padding bytes may hold anything.
        let d = dispatch_of([0.0; 2], [0.0; 2], 0.0, 2.0, 0xabcd_ef00);
        let values = Evaluation::of(&d, [1.0, 1.0]);
        assert!(!values.reset);
        assert_eq!(values.motion_scale, [1.0, 1.0]);
        assert_eq!(values.pre_exposure, 2.0);
    }

    #[test]
    fn the_masks_dlss_reads() {
        let mut d = dispatch_of([0.0; 2], [1.0; 2], 0.0, 1.0, 0);
        let listed = |d: &DispatchUpscale, flags| inputs(d, flags, true).collect::<Vec<_>>();
        assert_eq!(listed(&d, 0x29), [0, 1, 2, 6]);
        // An exposure texture counts only without auto exposure; the
        // reactive mask whenever the title gives one (unless left out);
        // the transparency and composition mask never.
        for i in [3, 4, 5] {
            d.resources[i].resource = 0x1000 as *mut c_void;
        }
        assert_eq!(listed(&d, 0x29), [0, 1, 2, 4, 6]);
        assert_eq!(listed(&d, 0x09), [0, 1, 2, 3, 4, 6]);
        assert_eq!(inputs(&d, 0x29, false).collect::<Vec<_>>(), [0, 1, 2, 6]);
    }

    /// The title's chain: upscaler, console backend, version. The version
    /// is found past the backend and kept on its own.
    #[test]
    fn the_titles_api_version_is_kept_for_the_fallback() {
        let mut version = UpscaleVersion {
            header: Header {
                kind: UPSCALE_VERSION,
                next: 0x1234 as *mut Header,
            },
            version: 0x0100_1001,
        };
        let mut console = Header {
            kind: crate::modules::fidelityfx::CONSOLE_BACKEND,
            next: &raw mut version.header,
        };
        // SAFETY: a chain of live descriptions, ending past the version.
        let found = unsafe { version_in(&raw mut console) }.unwrap();
        assert_eq!(found.version, 0x0100_1001);
        assert!(found.header.next.is_null());
        console.next = null_mut();
        // SAFETY: as above.
        assert_eq!(unsafe { version_in(&raw mut console) }, None);
        // SAFETY: an empty chain.
        assert_eq!(unsafe { version_in(null()) }, None);
    }

    #[test]
    fn the_dll_in_the_process_and_its_folder() {
        let shipped = Path::new(r"C:\Games\dlss-310.9.16/nvngx_dlss.dll");
        assert!(is_dlss_snippet(shipped));
        assert!(is_dlss_snippet(Path::new(
            r"C:\ProgramData\NVIDIA\NGX\models\dlss\versions\20316673\files\160_E658703.bin"
        )));
        assert!(!is_dlss_snippet(Path::new(r"C:\Games\nvngx_dlssg.dll")));
        assert!(!is_dlss_snippet(Path::new(
            r"C:\ProgramData\NVIDIA\NGX\models\dlssg\versions\1\files\x.bin"
        )));
        assert!(from_folder(shipped, Path::new(r"c:\games\DLSS-310.9.16\")));
        assert!(!from_folder(shipped, Path::new(r"C:\Games")));
    }

    #[test]
    fn dlss_reads_inputs_in_compute_read_states() {
        // The title's "read" (pixel and compute) suits DLSS as it is.
        assert_eq!(state_for(false, Some(0xc0)), 0xc0);
        assert_eq!(state_for(false, Some(0x4)), COMPUTE_READ);
        assert_eq!(state_for(false, None), COMPUTE_READ);
        assert_eq!(state_for(true, Some(0x8)), UNORDERED_ACCESS);
        assert_eq!(state_for(true, Some(0xc0)), UNORDERED_ACCESS);
    }

    #[test]
    fn dlss_restores_bindings_on_preparation_binding_and_creation_failures() {
        use std::cell::Cell;

        let binding = |backing, declared, used| {
            Some(Binding {
                before: (backing, Some(declared)),
                declared,
                used,
                on_failure: Some(declared),
            })
        };
        let mut bindings = [None; 7];
        bindings[0] = binding(1, 0x4, COMPUTE_READ);
        bindings[1] = binding(2, 0xc0, 0xc0);
        bindings[6] = binding(3, 0x80, UNORDERED_ACCESS);

        for (fail_prepare, fail_bind, fail_create) in [
            (true, false, false),
            (false, true, false),
            (false, false, true),
            (false, false, false),
        ] {
            let state = [const { Cell::new(0) }; 7];
            state[0].set(0x4);
            state[1].set(0xc0);
            state[6].set(0x80);
            let called = Cell::new(false);
            let result = with_resource_states(
                bindings,
                |i| {
                    assert_ne!(i, 6, "the output is never prepared as an input");
                    // Preparing an alias can leave another binding in COPY_SOURCE.
                    state[1 - i].set(0x800);
                    !(i == 1 && fail_prepare)
                },
                |i, used| {
                    state[i].set(used);
                    if i == 1 && fail_bind {
                        // A texture with incompatible flags got a different legal state.
                        state[i].set(0);
                        return Err(DispatchError::Refused(PARAMETER, "too small"));
                    }
                    Ok(i)
                },
                |i| Some((bindings[i].unwrap().before.0, Some(state[i].get()))),
                |i, declared| state[i].set(declared),
                |textures| {
                    called.set(true);
                    assert_eq!(textures[6], Some(6));
                    assert_eq!(
                        [state[0].get(), state[1].get(), state[6].get()],
                        [COMPUTE_READ, 0xc0, UNORDERED_ACCESS]
                    );
                    if fail_create {
                        Err(DispatchError::Failed("CreateFeature".into()))
                    } else {
                        Ok(())
                    }
                },
            );
            assert_eq!(called.get(), !fail_prepare && !fail_bind);
            assert_eq!(
                matches!(result, Err(DispatchError::Refused(..))),
                fail_prepare || fail_bind
            );
            assert_eq!(matches!(result, Err(DispatchError::Failed(_))), fail_create);
            assert_eq!(
                [state[0].get(), state[1].get(), state[6].get()],
                [0x4, 0xc0, 0x80]
            );
        }
    }

    #[test]
    fn dlss_checks_aliased_arguments_before_recording() {
        let input = Binding {
            before: (1, Some(0x4)),
            declared: 0x4,
            used: COMPUTE_READ,
            on_failure: Some(0x4),
        };
        for (index, declared, used, allowed) in [
            (1, 0x4, COMPUTE_READ, true),
            (1, 0x80, COMPUTE_READ, false),
            (6, 0x4, COMPUTE_READ, false),
            (6, 0x4, UNORDERED_ACCESS, false),
        ] {
            let mut bindings = [None; 7];
            bindings[0] = Some(input);
            bindings[index] = Some(Binding {
                declared,
                used,
                ..input
            });
            let result: Outcome = with_resource_states(
                bindings,
                |_| {
                    assert!(allowed);
                    true
                },
                |_, _| {
                    assert!(allowed);
                    Ok(())
                },
                |_| panic!("a rejected alias cannot need rollback"),
                |_, _| assert!(allowed),
                |_| Ok(()),
            );
            assert_eq!(result.is_ok(), allowed);
        }
    }

    #[test]
    fn dlss_does_not_invent_a_state_when_a_later_binding_was_not_reached() {
        use std::cell::Cell;

        assert_eq!(host_state(0x14), None);
        for (previous, current, expected_restore) in [
            (None, None, None),
            (Some(0x400), Some(0x400), None),
            (Some(0x400), Some(0x800), Some(0x400)),
            (None, Some(0x800), None),
        ] {
            let mut bindings = [None; 7];
            bindings[0] = Some(Binding {
                before: (1, None),
                declared: COMPUTE_READ,
                used: COMPUTE_READ,
                on_failure: Some(COMPUTE_READ),
            });
            bindings[6] = Some(Binding {
                before: (2, previous),
                declared: UNORDERED_ACCESS,
                used: UNORDERED_ACCESS,
                on_failure: previous,
            });
            let restored = Cell::new(None);
            let result = with_resource_states(
                bindings,
                |_| true,
                |_, _| Err::<(), _>(DispatchError::Refused(PARAMETER, "bad input")),
                |i| if i == 6 { Some((2, current)) } else { None },
                |i, state| {
                    if i == 6 {
                        restored.set(Some(state));
                    }
                },
                |_| panic!("a refused dispatch cannot be evaluated"),
            );
            assert!(matches!(result, Err(DispatchError::Refused(..))));
            assert_eq!(restored.get(), expected_restore);
        }
    }

    #[test]
    fn series_s_dlss_answers_with_the_selected_fidelityfx_quality() {
        let (mut width, mut height) = (0, 0);
        let mut render = RenderResolution {
            header: Header {
                kind: GET_RENDER_RESOLUTION,
                next: null_mut(),
            },
            display: [1920, 1080],
            quality: 1,
            width: &mut width,
            height: &mut height,
        };
        assert_eq!(query_dlss(&mut render.header, None), OK);
        assert_eq!([width, height], [1280, 720]);
        assert_eq!(query_dlss(&mut render.header, Some(3)), OK);
        assert_eq!([width, height], [960, 540]);
        assert_eq!(render.quality, 1);

        let mut ratio = 0.0;
        let mut request = RatioQuery {
            header: Header {
                kind: GET_UPSCALE_RATIO,
                next: null_mut(),
            },
            quality: 1,
            ratio: &mut ratio,
        };
        assert_eq!(query_dlss(&mut request.header, Some(3)), OK);
        assert_eq!(ratio, 2.0);
    }

    #[test]
    fn fidelityfx_render_sizes_and_jitter() {
        let size = |mode| upscale_ratio(mode).map(|r| [3840, 2160].map(|v| (v as f32 / r) as u32));
        assert_eq!(size(1), Some([2560, 1440]));
        assert_eq!(size(2), Some([2258, 1270]));
        assert_eq!(size(3), Some([1920, 1080]));
        assert_eq!(size(9), None);
        assert_eq!(phase_count(2560, 3840), 18);
        assert_eq!(phase_count(1920, 3840), 32);
        assert_eq!(phase_count(0, 3840), 0);
        assert_eq!(phase_count(1920, 0), 0);
        assert_eq!(phase_count(1920, 1), 1);
        // Halton (2, 3) from its second point, centred on the pixel.
        assert_eq!(jitter_offset(0, 18), [0.0, 1.0 / 3.0 - 0.5]);
        assert_eq!(jitter_offset(1, 18), [-0.25, 2.0 / 3.0 - 0.5]);
        assert_eq!(jitter_offset(18, 18), jitter_offset(0, 18));
        for i in 0..64 {
            let [x, y] = jitter_offset(i, 32);
            assert!(x.abs() <= 0.5 && y.abs() <= 0.5);
        }
    }

    #[test]
    fn parameter_table_slots() {
        // Set(float) is the second overload declared, Get(int*) the fifth.
        assert_eq!(Layout::Declared.set(Kind::F32), 1);
        assert_eq!(Layout::Msvc.set(Kind::F32), 6);
        assert_eq!(Layout::Msvc.set(Kind::D3D12), 1);
        assert_eq!(Layout::Msvc.set(Kind::Pointer), 0);
        assert_eq!(Layout::Declared.get(Kind::I32), 12);
        assert_eq!(Layout::Msvc.get(Kind::I32), 11);
        assert_eq!(Layout::Msvc.get(Kind::U64), 15);
    }

    #[test]
    fn the_core_is_looked_for_where_the_loader_looks() {
        let found = core_candidates(&[
            None,
            Some(r"C:\Driver\nv_dispi".into()),
            Some("C:\\Driver\\nv_dispi\\\0".into()),
            Some(String::new()),
        ]);
        assert_eq!(
            found,
            [
                PathBuf::from(r"C:\Driver\nv_dispi\_nvngx.dll"),
                PathBuf::from(r"C:\Driver\nv_dispi\nvngx.dll"),
            ]
        );
    }

    #[test]
    fn results_by_name() {
        assert!(succeeded(1));
        assert!(!succeeded(0xbad0_0005));
        assert_eq!(result_name(0xbad0_0005), "InvalidParameter (0xbad00005)");
        assert_eq!(result_name(0xbad0_0001), "FeatureNotSupported (0xbad00001)");
        assert_eq!(result_name(1), "Success");
    }
}
