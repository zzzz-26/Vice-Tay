//! `amd_fidelityfx_loader_xbox.dll`: AMD FidelityFX's API, served by the
//! host's PC build of it on the host GPU when the user names a folder that
//! holds one (`VICTORY_FIDELITYFX_DIR`) and selects a PC upscaler.
//! Other profiles keep the title's own console DLLs.
//!
//! Both builds have the same API (the FidelityFX SDK's, MIT): five functions
//! over chains of descriptions, each starting with a type id and the next
//! one's address; contexts are opaque pointers. The console-specific parts
//! are translated: the backend description names the console device (type
//! 0x0100_0002; on PC type 2 with the host device), and a dispatch names the
//! title's command list and resources (their host list and host copies, in
//! the states the title gives, see `victory_gpu::external`). The rest passes
//! through unchanged: sizes, flags, jitter, motion-vector scale, camera
//! values, the resource descriptions, the context. Only the upscaler: other
//! effects (frame generation) are refused.
//!
//! With `VICTORY_DLSS_4K` the same calls are served by NVIDIA DLSS instead
//! (`dlss`), on an RTX card.
#![allow(non_snake_case)]
// Title code calls these through its import table.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::collections::BTreeMap;
use std::ffi::{CStr, c_char, c_void};
use std::path::Path;
use std::ptr::{null, null_mut};
use std::sync::{Arc, Mutex, OnceLock};

use victory_gpu::object::Object;
use windows::Win32::Graphics::Direct3D12::{ID3D12Device, ID3D12Resource};
use windows::core::Interface;

use crate::{HleModule, hle_export};

/// The PC build's upscaler, which has the five functions itself.
const DLL: &str = "amd_fidelityfx_upscaler_dx12.dll";

// Return codes.
const OK: u32 = 0;
const UNKNOWN_TYPE: u32 = 2;
const NO_PROVIDER: u32 = 4;
const PARAMETER: u32 = 6;

// Description types.
const CREATE_UPSCALE: u64 = 0x1_0000;
const DISPATCH_UPSCALE: u64 = 0x1_0001;
const CONSOLE_BACKEND: u64 = 0x100_0002;
const HOST_BACKEND: u64 = 2;
const GET_VERSIONS: u64 = 4;
const OVERRIDE_VERSION: u64 = 5;
const GET_PROVIDER_VERSION: u64 = 6;
const MEMORY_USAGE_WITH_DEVICE: u64 = 0x1_0009;
const GET_RENDER_RESOLUTION: u64 = 0x1_0003;
const GET_UPSCALE_RATIO: u64 = 0x1_0002;
const DISPLAY_RESOLUTION_MOTION_VECTORS: u32 = 1 << 1;
const FSR3_VERSION: &str = "3.1.6";
const FSR4_VERSION: &str = "4.1.2";

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Header {
    kind: u64,
    next: *mut Header,
}

/// An upscaler context: flags, largest render and output sizes, a message
/// callback.
#[repr(C)]
#[derive(Clone, Copy)]
struct CreateUpscale {
    header: Header,
    flags: u32,
    max_render: [u32; 2],
    max_output: [u32; 2],
    message: usize,
}

/// The backend description: a device.
#[repr(C)]
struct Backend {
    header: Header,
    device: *mut c_void,
}

/// A resource, its description (type, format, sizes, mips, flags, usage)
/// and the state it is in.
#[repr(C)]
#[derive(Clone, Copy)]
struct Resource {
    resource: *mut c_void,
    description: [u32; 8],
    state: u32,
}

/// One upscaling: the command list, then colour, depth, motion vectors,
/// exposure, reactive mask, transparency mask and output; then jitter,
/// motion-vector scale, sizes, sharpening, frame time, pre-exposure, reset,
/// camera and flags, which pass through.
#[repr(C)]
#[derive(Clone, Copy)]
struct DispatchUpscale {
    header: Header,
    list: *mut c_void,
    resources: [Resource; 7],
    jitter: [f32; 2],
    motion_scale: [f32; 2],
    render_size: [u32; 2],
    upscale_size: [u32; 2],
    settings: [u32; 10],
}

const _: () = assert!(size_of::<CreateUpscale>() == 0x30);
const _: () = assert!(size_of::<Resource>() == 0x30);
const _: () = assert!(size_of::<DispatchUpscale>() == 0x1b0);

/// The versions a backend offers: count in and out, ids, names.
#[repr(C)]
#[derive(Clone, Copy)]
struct GetVersions {
    header: Header,
    create_kind: u64,
    device: *mut c_void,
    count: *mut u64,
    ids: *mut u64,
    names: *mut *const c_char,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MemoryUsageWithDevice {
    header: Header,
    device: *mut c_void,
    max_render: [u32; 2],
    max_output: [u32; 2],
    flags: u32,
    usage: *mut c_void,
}

#[repr(C)]
struct ProviderVersion {
    header: Header,
    id: u64,
    name: *const c_char,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RatioQuery {
    header: Header,
    quality: u32,
    ratio: *mut f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RenderResolution {
    header: Header,
    display: [u32; 2],
    quality: u32,
    width: *mut u32,
    height: *mut u32,
}

#[repr(C)]
struct OverrideVersion {
    header: Header,
    id: u64,
}

type Create = unsafe extern "C" fn(*mut *mut c_void, *mut Header, *const c_void) -> u32;
type Destroy = unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> u32;
/// Configure, query and dispatch.
type Call = unsafe extern "C" fn(*mut *mut c_void, *mut Header) -> u32;

struct Host {
    _library: Library,
    create: Create,
    destroy: Destroy,
    configure: Call,
    query: Call,
    dispatch: Call,
    device: ID3D12Device,
    provider_id: u64,
    version: &'static str,
    quality: Option<u32>,
}

struct Library(windows_sys::Win32::Foundation::HMODULE);

impl Drop for Library {
    fn drop(&mut self) {
        // SAFETY: this owns the reference acquired by LoadLibraryExW.
        unsafe {
            windows_sys::Win32::Foundation::FreeLibrary(self.0);
        }
    }
}

// SAFETY: D3D12 devices are free-threaded; the rest is code.
unsafe impl Send for Host {}
// SAFETY: as above.
unsafe impl Sync for Host {}

static HOST: OnceLock<Host> = OnceLock::new();
struct Context {
    // Serializes recording and destruction for this particular temporal history.
    live: Mutex<bool>,
    first_dispatch: std::sync::Once,
    // The FFX API requires pointers in the creation descriptor to remain
    // live through destroy, including our replacement backend.
    creation: CreateChain,
}

struct CreateChain {
    description: Box<CreateUpscale>,
    _backend: Box<Backend>,
    _version: Box<OverrideVersion>,
}

// SAFETY: after Create, the boxed descriptors are retained immutably.
// The device is held by Host; the remaining extension pointers belong
// to the title under the FFX context lifetime contract.
unsafe impl Send for CreateChain {}
// SAFETY: as above, no thread mutates the retained descriptors.
unsafe impl Sync for CreateChain {}

static CONTEXTS: Mutex<BTreeMap<usize, Arc<Context>>> = Mutex::new(BTreeMap::new());

fn fits(extent: [u32; 2], maximum: [u32; 2]) -> bool {
    extent[0] != 0 && extent[1] != 0 && extent[0] <= maximum[0] && extent[1] <= maximum[1]
}

fn dispatch_output(
    render: [u32; 2],
    output: [u32; 2],
    max_render: [u32; 2],
    max_output: [u32; 2],
) -> Option<[u32; 2]> {
    let output = if output == [0, 0] { max_output } else { output };
    (fits(render, max_render) && fits(output, max_output) && fits(render, output)).then_some(output)
}

fn minimum_extent(index: usize, flags: u32, render: [u32; 2], output: [u32; 2]) -> [u32; 2] {
    match index {
        6 => output,
        2 if flags & DISPLAY_RESOLUTION_MOTION_VECTORS != 0 => output,
        3 => [1, 1], // Optional exposure texture.
        _ => render,
    }
}

macro_rules! once {
    ($level:ident, $($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::$level!($($arg)*));
    }};
}

mod dlss;
pub mod native_framegen;
mod resource_states;

use resource_states::{Binding, SetupError, with_resource_states};

pub use victory_gpu::framegen::{requested as framegen_requested, set_dir as set_framegen_dir};

/// Select the best supported PC provider before the title's imports are bound.
pub fn use_host_dir(dir: &Path) -> Result<&'static str, String> {
    use_host_version(dir, &[FSR4_VERSION, FSR3_VERSION], None)
}

/// Select FSR 3.1.6 with the Series S profile's requested quality mode.
pub fn use_host_fsr3(dir: &Path, quality: u32) -> Result<(), String> {
    use_host_version(dir, &[FSR3_VERSION], Some(quality)).map(|_| ())
}

fn use_host_version(
    dir: &Path,
    versions: &[&'static str],
    quality: Option<u32>,
) -> Result<&'static str, String> {
    let (host, name) = load(dir, versions, quality)?;
    let version = host.version;
    HOST.set(host)
        .map_err(|_| String::from("FidelityFX provider already initialized"))?;
    tracing::info!(folder = %dir.display(), provider = name, "FidelityFX: selected PC upscaler");
    Ok(version)
}

/// Select NVIDIA DLSS as the provider, before the title's import registry
/// is initialized: `nvngx_dlss.dll` from `dir` if given (else the
/// driver's), NGX's logs in `logs`, `quality` an NGX
/// `NVSDK_NGX_PerfQuality_Value`. Why it cannot run, on failure.
/// [`use_host_dir`] afterwards keeps FSR as the fallback of a context
/// whose DLSS feature fails.
pub fn use_dlss(dir: Option<&Path>, logs: &Path, quality: i32) -> Result<(), String> {
    start_dlss(dir, logs, quality, None)
}

/// Select DLSS with a fixed FidelityFX quality mode for Series S size queries.
pub fn use_dlss_series_s(
    dir: Option<&Path>,
    logs: &Path,
    quality: i32,
    ffx_quality: u32,
) -> Result<(), String> {
    start_dlss(dir, logs, quality, Some(ffx_quality))
}

fn start_dlss(
    dir: Option<&Path>,
    logs: &Path,
    quality: i32,
    ffx_quality: Option<u32>,
) -> Result<(), String> {
    if HOST.get().is_some() {
        return Err("FidelityFX provider already initialized".into());
    }
    let setup = dlss::start(dir, logs, quality, ffx_quality)?;
    tracing::info!(%setup, "FidelityFX: NVIDIA DLSS serves the title's upscaler");
    Ok(())
}

/// Whether the host's FidelityFX, or DLSS, serves the title's.
pub fn serving() -> bool {
    HOST.get().is_some() || dlss::active()
}

/// Releases presentation and upscaling resources before the process ends.
pub fn shut_down() {
    victory_gpu::framegen::shut_down();
    dlss::shut_down();
}

fn load(
    dir: &Path,
    versions: &[&'static str],
    quality: Option<u32>,
) -> Result<(Host, String), String> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::System::LibraryLoader::{
        GetProcAddress, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR, LOAD_LIBRARY_SEARCH_SYSTEM32,
        LoadLibraryExW,
    };
    // The search flags below want a full path.
    let path = std::path::absolute(dir.join(DLL)).map_err(|e| e.to_string())?;
    if !path.is_file() {
        return Err(format!("no {DLL} there"));
    }
    let wide: Vec<u16> = path.as_os_str().encode_wide().chain([0]).collect();
    // Its own folder and System32 only, never the current folder.
    let search = LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32;
    // SAFETY: a NUL-terminated path to the user's FidelityFX build.
    let module = unsafe { LoadLibraryExW(wide.as_ptr(), null_mut(), search) };
    if module.is_null() {
        return Err(format!(
            "{DLL} did not load ({})",
            std::io::Error::last_os_error()
        ));
    }
    let library = Library(module);
    let find = |name: &CStr| {
        // SAFETY: a loaded module and a NUL-terminated name.
        unsafe { GetProcAddress(module, name.as_ptr().cast()) }
            .map(|f| f as *const c_void)
            .ok_or_else(|| format!("{DLL} has no {}", name.to_string_lossy()))
    };
    let device = victory_gpu::external::host_device().ok_or("there is no host GPU")?;
    // SAFETY: the SDK's ffxQuery export, with its declared signature.
    let query = unsafe { std::mem::transmute::<*const c_void, Call>(find(c"ffxQuery")?) };
    let (provider_id, version, name) = select_version(query, device.as_raw(), versions)?;
    // SAFETY: the API's exports, with its signatures.
    let host = unsafe {
        Host {
            create: std::mem::transmute::<*const c_void, Create>(find(c"ffxCreateContext")?),
            destroy: std::mem::transmute::<*const c_void, Destroy>(find(c"ffxDestroyContext")?),
            configure: std::mem::transmute::<*const c_void, Call>(find(c"ffxConfigure")?),
            query,
            dispatch: std::mem::transmute::<*const c_void, Call>(find(c"ffxDispatch")?),
            _library: library,
            device,
            provider_id,
            version,
            quality,
        }
    };
    Ok((host, name))
}

fn is_version(name: &str, version: &str) -> bool {
    let name = name.trim();
    name.strip_prefix("FSR ")
        .unwrap_or(name)
        .strip_prefix(version)
        .is_some_and(|suffix| {
            suffix.is_empty() || suffix.starts_with(|c: char| c.is_whitespace() || c == '*')
        })
}

/// The SDK supplies the version id to put in the context creation chain.
fn select_version(
    query_fn: Call,
    device: *mut c_void,
    versions: &[&'static str],
) -> Result<(u64, &'static str, String), String> {
    let mut count = 0u64;
    let mut query = GetVersions {
        header: Header {
            kind: GET_VERSIONS,
            next: null_mut(),
        },
        create_kind: CREATE_UPSCALE,
        device,
        count: &mut count,
        ids: null_mut(),
        names: null_mut(),
    };
    // SAFETY: a version query naming the host device.
    let code = unsafe { query_fn(null_mut(), &mut query.header) };
    if code != OK || count == 0 {
        return Err(format!(
            "it offers no upscaler for the host GPU (code {code})"
        ));
    }
    let mut names = vec![null::<c_char>(); count as usize];
    let mut ids = vec![0u64; names.len()];
    query.names = names.as_mut_ptr();
    query.ids = ids.as_mut_ptr();
    // SAFETY: as above, with room for `count` names.
    let code = unsafe { query_fn(null_mut(), &mut query.header) };
    if code != OK {
        return Err(format!(
            "upscaler version names could not be queried (code {code})"
        ));
    }
    let available: Vec<_> = ids
        .into_iter()
        .zip(names)
        .take(count as usize)
        .filter(|(_, name)| !name.is_null())
        .map(|(id, name)| {
            // SAFETY: NUL-terminated name owned by the SDK.
            (id, unsafe { CStr::from_ptr(name) }.to_string_lossy())
        })
        .collect();
    for &version in versions {
        if let Some((id, name)) = available.iter().find(|(_, name)| is_version(name, version)) {
            return Ok((*id, version, name.to_string()));
        }
    }
    Err(format!(
        "no FSR {} provider for this adapter (available: {})",
        versions.join(" or "),
        available.iter().map(|(_, name)| name.as_ref()).collect::<Vec<_>>().join(", ")
    ))
}

/// Puts the host backend after the title's upscaler description, in place
/// of the console's backend that follows it (where the title puts it);
/// anywhere else the console's is left for the host DLL to ignore, ours
/// coming first.
fn link(upscale: &mut CreateUpscale, backend: &mut Backend) {
    let after = upscale.header.next;
    // SAFETY: the title's chain.
    backend.header.next = match unsafe { after.as_ref() } {
        Some(h) if h.kind == CONSOLE_BACKEND => h.next,
        _ => after,
    };
    backend.header.kind = HOST_BACKEND;
    upscale.header.next = &raw mut backend.header;
}

/// A FidelityFX resource state as the host resource state it means (as the
/// PC backend reads it); `None` for generic read and indirect arguments,
/// which a host texture cannot be in.
fn host_state(state: u32) -> Option<u32> {
    Some(match state {
        0x1 | 0x80 => 0,  // common, present
        0x2 => 0x8,       // unordered access
        0x4 => 0x40,      // compute read
        0x8 => 0x80,      // pixel read
        0xc => 0xc0,      // both
        0x10 => 0x800,    // copy source
        0x20 => 0x400,    // copy destination
        0x100 => 0x4,     // render target
        0x200 => 0x10,    // depth write
        _ => return None, // generic read, indirect arguments
    })
}

extern "C" fn ffxCreateContext(
    context: *mut *mut c_void,
    desc: *mut Header,
    memory: *const c_void,
) -> u32 {
    if dlss::active() {
        return dlss::create_context(context, desc);
    }
    HOST.get()
        .map_or(NO_PROVIDER, |host| create_on_host(host, context, desc, memory))
}

/// The upscaler context on the host's FidelityFX (also DLSS's fallback).
fn create_on_host(
    host: &Host,
    context: *mut *mut c_void,
    desc: *mut Header,
    memory: *const c_void,
) -> u32 {
    if context.is_null() || desc.is_null() {
        return PARAMETER;
    }
    // SAFETY: the title passes a chain of descriptions.
    let kind = unsafe { (*desc).kind };
    if kind != CREATE_UPSCALE {
        once!(
            warn,
            kind = format_args!("{kind:#x}"),
            "FidelityFX: only the upscaler runs on the host; another effect was refused"
        );
        return NO_PROVIDER;
    }
    // SAFETY: an upscaler description, by its type.
    let mut upscale = Box::new(unsafe { desc.cast::<CreateUpscale>().read() });
    let mut backend = Box::new(Backend {
        header: Header {
            kind: HOST_BACKEND,
            next: null_mut(),
        },
        device: host.device.as_raw(),
    });
    link(&mut upscale, &mut backend);
    let mut requested = Box::new(OverrideVersion {
        header: Header {
            kind: OVERRIDE_VERSION,
            next: backend.header.next,
        },
        id: host.provider_id,
    });
    backend.header.next = &raw mut requested.header;
    // SAFETY: a PC chain; the title's later descriptions live for the call.
    let code = unsafe { (host.create)(context, &mut upscale.header, memory) };
    if code != OK {
        tracing::error!(code, render = ?upscale.max_render, output = ?upscale.max_output, flags = format_args!("{:#x}", upscale.flags), "FidelityFX: the host refused the title's upscaler");
        return code;
    }
    let mut version = ProviderVersion {
        header: Header {
            kind: GET_PROVIDER_VERSION,
            next: null_mut(),
        },
        id: 0,
        name: null(),
    };
    // SAFETY: a provider query on the context just made.
    let query_code = unsafe { (host.query)(context, &mut version.header) };
    // SAFETY: a NUL-terminated name the DLL keeps, if any.
    let name = (!version.name.is_null())
        .then(|| unsafe { CStr::from_ptr(version.name) }.to_string_lossy());
    if query_code != OK
        || version.id != host.provider_id
        || !name.as_deref().is_some_and(|name| is_version(name, host.version))
    {
        tracing::warn!(
            query_code,
            provider = name.as_deref().unwrap_or("?"),
            requested = host.version,
            "FidelityFX created a different provider than requested"
        );
        // SAFETY: the PC provider just created the context with this allocator.
        unsafe { (host.destroy)(context, memory) };
        return NO_PROVIDER;
    }
    // SAFETY: a successful SDK creation wrote an opaque context into this slot.
    tracing::info!(
        provider = name.as_deref().unwrap_or("?"),
        id = format_args!("{:#x}", version.id),
        render = ?upscale.max_render,
        output = ?upscale.max_output,
        flags = format_args!("{:#x}", upscale.flags),
        "FidelityFX: upscaler context on the host"
    );
    CONTEXTS.lock().unwrap_or_else(|e| e.into_inner()).insert(
        unsafe { *context } as usize,
        Arc::new(Context {
            live: Mutex::new(true),
            first_dispatch: std::sync::Once::new(),
            creation: CreateChain {
                description: upscale,
                _backend: backend,
                _version: requested,
            },
        }),
    );
    OK
}

extern "C" fn ffxDestroyContext(context: *mut *mut c_void, memory: *const c_void) -> u32 {
    if dlss::active() {
        return dlss::destroy_context(context);
    }
    HOST.get()
        .map_or(NO_PROVIDER, |host| destroy_on_host(host, context, memory))
}

fn destroy_on_host(host: &Host, context: *mut *mut c_void, memory: *const c_void) -> u32 {
    if context.is_null() {
        return PARAMETER;
    }
    // SAFETY: the title passes its context. Capture its value before destroy clears it.
    let handle = unsafe { *context } as usize;
    let ctx = CONTEXTS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&handle)
        .cloned();
    let Some(ctx) = ctx else { return PARAMETER };
    let mut live = ctx.live.lock().unwrap_or_else(|e| e.into_inner());
    if !*live {
        return PARAMETER;
    }
    // The guest releases a context after submitting its frames. The host
    // may still be executing their lists, even after guest Execute returns.
    victory_gpu::external::finish_submitted();
    let code = unsafe { (host.destroy)(context, memory) };
    if code == OK {
        *live = false;
        let mut contexts = CONTEXTS.lock().unwrap_or_else(|e| e.into_inner());
        // A concurrent Create may already have reused the released SDK address.
        if contexts
            .get(&handle)
            .is_some_and(|current| Arc::ptr_eq(current, &ctx))
        {
            contexts.remove(&handle);
        }
    }
    code
}

/// Settings (key-value pairs, debug messages) name nothing console-specific.
extern "C" fn ffxConfigure(context: *mut *mut c_void, desc: *mut Header) -> u32 {
    if dlss::active() {
        return dlss::configure(context, desc);
    }
    // SAFETY: the title passes its context and descriptions.
    HOST.get()
        .map_or(NO_PROVIDER, |h| unsafe { (h.configure)(context, desc) })
}

extern "C" fn ffxQuery(context: *mut *mut c_void, desc: *mut Header) -> u32 {
    if dlss::active() {
        return dlss::query(context, desc);
    }
    HOST.get()
        .map_or(NO_PROVIDER, |host| query_on_host(host, context, desc))
}

fn query_on_host(host: &Host, context: *mut *mut c_void, desc: *mut Header) -> u32 {
    if desc.is_null() {
        return PARAMETER;
    }
    // SAFETY: the title passes a description.
    let kind = unsafe { (*desc).kind };
    // Substitute the native device; the SDK returns values through the
    // title's original output pointers in both request structures.
    if kind == GET_VERSIONS {
        let mut query = unsafe { *desc.cast::<GetVersions>() };
        query.device = host.device.as_raw();
        return unsafe { (host.query)(context, &mut query.header) };
    }
    if kind == MEMORY_USAGE_WITH_DEVICE {
        let mut query = unsafe { *desc.cast::<MemoryUsageWithDevice>() };
        query.device = host.device.as_raw();
        return unsafe { (host.query)(context, &mut query.header) };
    }
    if let Some(quality) = host.quality {
        if kind == GET_UPSCALE_RATIO {
            // SAFETY: the title passed the ratio-query descriptor identified above.
            let mut query = unsafe { *desc.cast::<RatioQuery>() };
            query.quality = quality;
            // SAFETY: the copy retains the title's output pointer.
            return unsafe { (host.query)(context, &mut query.header) };
        }
        if kind == GET_RENDER_RESOLUTION {
            // SAFETY: the title passed the render-resolution descriptor identified above.
            let mut query = unsafe { *desc.cast::<RenderResolution>() };
            query.quality = quality;
            // SAFETY: the copy retains the title's output pointers.
            return unsafe { (host.query)(context, &mut query.header) };
        }
    }
    // SAFETY: as above.
    let code = unsafe { (host.query)(context, desc) };
    if kind == GET_RENDER_RESOLUTION && code != OK {
        tracing::warn!(
            code,
            "FidelityFX provider refused the render-resolution query"
        );
    }
    code
}

extern "C" fn ffxDispatch(context: *mut *mut c_void, desc: *mut Header) -> u32 {
    if dlss::active() {
        return dlss::dispatch(context, desc);
    }
    HOST.get()
        .map_or(NO_PROVIDER, |host| dispatch_on_host(host, context, desc))
}

fn dispatch_on_host(host: &Host, context: *mut *mut c_void, desc: *mut Header) -> u32 {
    if desc.is_null() {
        return PARAMETER;
    }
    // SAFETY: the title passes a description.
    let kind = unsafe { (*desc).kind };
    if kind != DISPATCH_UPSCALE {
        once!(
            warn,
            kind = format_args!("{kind:#x}"),
            "FidelityFX: only upscaling runs on the host; a dispatch was refused"
        );
        return UNKNOWN_TYPE;
    }
    // SAFETY: an upscaling, by its type.
    let dispatch = unsafe { desc.cast::<DispatchUpscale>().read() };
    if context.is_null() {
        once!(warn, "FidelityFX dispatch arrived without a context");
        return PARAMETER;
    }
    if let Some(index) = [0, 1, 2, 6]
        .into_iter()
        .find(|&i| dispatch.resources[i].resource.is_null())
    {
        once!(
            warn,
            index,
            "FidelityFX dispatch is missing a required texture"
        );
        return PARAMETER;
    }
    // SAFETY: the title passes the same opaque context slot as for creation.
    let handle = unsafe { *context } as usize;
    let ctx = CONTEXTS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&handle)
        .cloned();
    let Some(ctx) = ctx else {
        once!(
            warn,
            "FidelityFX dispatch referenced an unregistered context"
        );
        return PARAMETER;
    };
    let live = ctx.live.lock().unwrap_or_else(|e| e.into_inner());
    if !*live {
        return PARAMETER;
    }
    let create = &ctx.creation.description;
    let Some(output_size) = dispatch_output(
        dispatch.render_size,
        dispatch.upscale_size,
        create.max_render,
        create.max_output,
    ) else {
        once!(warn, render = ?dispatch.render_size, output = ?dispatch.upscale_size,
            max_render = ?create.max_render, max_output = ?create.max_output,
            "FidelityFX rejected dispatch dimensions outside the context's allocations");
        return PARAMETER;
    };
    let list = dispatch.list.cast::<Object>().cast_const();
    victory_gpu::external::record(list, |list| {
        record_dispatch(list, host, context, &ctx, &dispatch, output_size)
    })
    .unwrap_or_else(|| {
        once!(
            warn,
            "FidelityFX could not record to the title command list"
        );
        PARAMETER
    })
}

struct DispatchError(u32);

impl From<SetupError> for DispatchError {
    fn from(error: SetupError) -> Self {
        match error {
            SetupError::Aliased(index) => {
                once!(warn, index, "FidelityFX: aliased arguments require conflicting resource states");
            }
            SetupError::Input(index) => {
                once!(warn, index, "FidelityFX: an input could not be prepared on the title's list");
            }
        }
        Self(PARAMETER)
    }
}

fn record_dispatch(
    list: &victory_gpu::external::Recording<'_>,
    host: &Host,
    context: *mut *mut c_void,
    ctx: &Context,
    dispatch: &DispatchUpscale,
    output_size: [u32; 2],
) -> u32 {
    let flags = ctx.creation.description.flags;
    let resource = |index: usize| dispatch.resources[index].resource.cast::<Object>();
    let mut bindings = [None; 7];
    for (index, r) in dispatch.resources.iter().enumerate() {
        if r.resource.is_null() {
            continue;
        }
        let Some((wanted, before)) = host_state(r.state).and_then(|wanted| {
            list.texture_binding(resource(index))
                .map(|before| (wanted, before))
        })
        else {
            once!(
                warn,
                index,
                state = r.state,
                "FidelityFX: a resource has no host texture in its state; the upscaling did not run"
            );
            return PARAMETER;
        };
        bindings[index] = Some(Binding {
            before,
            declared: wanted,
            used: wanted,
            on_failure: before.1.or(Some(wanted)),
        });
    }

    let outcome = with_resource_states(
        bindings,
        |index| list.prepare_input(resource(index)).is_some(),
        |index, wanted| {
            let Some((texture, _)) = list
                .texture(resource(index), wanted)
                .filter(|&(_, state)| state == wanted)
            else {
                once!(warn, index, wanted, "FidelityFX: a texture cannot be in the declared state");
                return Err(DispatchError(PARAMETER));
            };
            // SAFETY: a live texture.
            let desc = unsafe { texture.GetDesc() };
            let needed = minimum_extent(index, flags, dispatch.render_size, output_size);
            if desc.Width < u64::from(needed[0]) || desc.Height < needed[1] {
                once!(warn, index, ?needed, width = desc.Width, height = desc.Height,
                    "FidelityFX resource cannot cover the active dispatch rectangle");
                return Err(DispatchError(PARAMETER));
            }
            Ok((texture, desc))
        },
        |index| list.existing_texture_binding(resource(index)),
        |index, state| {
            let _ = list.texture(resource(index), state);
        },
        |textures| {
            let mut translated = *dispatch;
            for (r, texture) in translated.resources.iter_mut().zip(textures) {
                if let Some((texture, _)) = texture {
                    r.resource = texture.as_raw();
                }
            }
            translated.list = list.commands().as_raw();
            // SAFETY: a PC dispatch on host objects. The context is live and locked.
            let code = unsafe { (host.dispatch)(context, &raw mut translated.header) };
            if code != OK {
                once!(warn, code, "FidelityFX provider rejected the title dispatch");
                return Err(DispatchError(code));
            }
            if victory_gpu::framegen::requested()
                && let (Some((depth, _)), Some((motion, _))) = (&textures[1], &textures[2])
            {
                record_framegen(
                    list,
                    dispatch,
                    ctx.creation.description.max_render,
                    flags,
                    [depth, motion],
                    [dispatch.resources[1].state, dispatch.resources[2].state],
                );
            }
            victory_gpu::external::output_written(resource(6));
            ctx.first_dispatch.call_once(|| {
                let backing_extents = textures
                    .each_ref()
                    .map(|t| t.as_ref().map(|(_, d)| (d.Width, d.Height)));
                let formats = textures.each_ref().map(|t| t.as_ref().map(|(_, d)| d.Format.0));
                tracing::info!(render = ?dispatch.render_size, output = ?output_size, ?backing_extents,
                    ?formats, jitter = ?dispatch.jitter, motion_vector_scale = ?dispatch.motion_scale,
                    provider = host.version, "FidelityFX recorded the title's temporal dispatch");
            });
            Ok(())
        },
    );
    outcome.err().map_or(OK, |DispatchError(code)| code)
}

fn record_framegen(
    list: &victory_gpu::external::Recording<'_>,
    dispatch: &DispatchUpscale,
    max_render: [u32; 2],
    upscale_flags: u32,
    [depth, motion]: [&ID3D12Resource; 2],
    [depth_state, motion_state]: [u32; 2],
) {
    if depth.as_raw() == motion.as_raw()
        || !matches!(depth_state, 0x4 | 0xc)
        || !matches!(motion_state, 0x4 | 0xc)
    {
        return;
    }
    list.prepare_framegen(victory_gpu::framegen::Inputs {
        depth,
        motion,
        depth_desc: dispatch.resources[1].description,
        motion_desc: dispatch.resources[2].description,
        depth_state,
        motion_state,
        max_render,
        render_size: dispatch.render_size,
        upscale_flags,
        jitter: dispatch.jitter,
        motion_scale: dispatch.motion_scale,
        frame_ms: f32::from_bits(dispatch.settings[2]),
        reset: dispatch.settings[4] & 0xff != 0,
        camera: std::array::from_fn(|i| f32::from_bits(dispatch.settings[5 + i])),
    });
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "fidelityfx",
        dlls: &["amd_fidelityfx_loader_xbox.dll"],
        exports: vec![
            hle_export!(ffxConfigure),
            hle_export!(ffxCreateContext),
            hle_export!(ffxDestroyContext),
            hle_export!(ffxDispatch),
            hle_export!(ffxQuery),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_logical_rectangles_against_the_current_context() {
        for display in [[1920, 1080], [2560, 1440], [3840, 2160]] {
            for divisor in [1.0_f32, 1.5, 1.7, 2.0, 3.0] {
                let render = display.map(|v| (v as f32 / divisor) as u32);
                assert_eq!(
                    dispatch_output(render, display, render, display),
                    Some(display)
                );
                assert_eq!(
                    dispatch_output(render, [0, 0], render, display),
                    Some(display)
                );
                assert_eq!(minimum_extent(2, 0x29, render, display), render);
                assert_eq!(minimum_extent(2, 0x2b, render, display), display);
                assert_eq!(minimum_extent(6, 0x29, render, display), display);
                assert_eq!(minimum_extent(3, 0x29, render, display), [1, 1]);
            }
        }
        assert_eq!(
            dispatch_output([0, 720], [0, 0], [1280, 720], [2560, 1440]),
            None
        );
        assert_eq!(
            dispatch_output([1281, 720], [0, 0], [1280, 720], [2560, 1440]),
            None
        );
        assert_eq!(
            dispatch_output([1280, 720], [0, 1440], [1280, 720], [2560, 1440]),
            None
        );
        assert_eq!(
            dispatch_output([1280, 720], [3000, 1440], [1280, 720], [2560, 1440]),
            None
        );
        assert_eq!(
            dispatch_output([1280, 720], [960, 540], [1280, 720], [2560, 1440]),
            None
        );
    }

    fn header(kind: u64, next: *mut Header) -> Header {
        Header { kind, next }
    }

    /// The title's chain: upscaler, console backend, version.
    #[test]
    fn the_console_backend_becomes_the_hosts() {
        let mut version = header(0x1_000b, null_mut());
        let mut console = Backend {
            header: header(CONSOLE_BACKEND, &raw mut version),
            device: 0x1234 as *mut c_void,
        };
        let mut upscale = CreateUpscale {
            header: header(CREATE_UPSCALE, &raw mut console.header),
            flags: 0x29,
            max_render: [1280, 720],
            max_output: [1920, 1080],
            message: 0,
        };
        let mut backend = Backend {
            header: header(0, null_mut()),
            device: 0x5678 as *mut c_void,
        };
        link(&mut upscale, &mut backend);
        assert_eq!(upscale.header.next, &raw mut backend.header);
        assert_eq!(backend.header.kind, HOST_BACKEND);
        assert_eq!(backend.header.next, &raw mut version);
        assert_eq!(backend.device, 0x5678 as *mut c_void);
        // Without a console backend right after it, ours goes in front.
        let mut upscale = CreateUpscale {
            header: header(CREATE_UPSCALE, &raw mut version),
            ..upscale
        };
        link(&mut upscale, &mut backend);
        assert_eq!(backend.header.next, &raw mut version);
    }

    /// Without the PC build the title's own DLL serves its imports.
    #[test]
    fn without_the_dll_the_titles_own_runs() {
        let dir = tempfile::tempdir().unwrap();
        assert!(load(dir.path(), &[FSR4_VERSION], None).err().unwrap().contains(DLL));
        assert!(!serving());
        let symbol = victory_pe::ImportSymbol::Name {
            hint: 0,
            name: "ffxCreateContext".into(),
        };
        let resolver = crate::HleResolver::new(crate::HostPolicy::none());
        assert_eq!(
            resolver.classify("amd_fidelityfx_loader_xbox.dll", &symbol),
            None
        );
    }

    #[test]
    fn upscaler_version_follows_the_adapters_reported_providers() {
        unsafe extern "C" fn versions(_: *mut *mut c_void, header: *mut Header) -> u32 {
            // SAFETY: select_version sends a GetVersions and keeps its arrays alive.
            let q = unsafe { &mut *header.cast::<GetVersions>() };
            let names = [c"3.1.6", c"2.3.4", c"FSR 4.1.2 *"];
            let count = if q.device.is_null() { 2 } else { names.len() };
            unsafe {
                if !q.names.is_null() && *q.count >= count as u64 {
                    for (i, name) in names[..count].iter().enumerate() {
                        q.names.add(i).write(name.as_ptr());
                        q.ids.add(i).write(i as u64 + 1);
                    }
                }
                *q.count = count as u64;
            }
            OK
        }

        let preferred = &[FSR4_VERSION, FSR3_VERSION];
        let device = std::ptr::dangling_mut::<c_void>();
        assert_eq!(select_version(versions, device, preferred).unwrap().0, 3);
        let (id, selected, _) = select_version(versions, null_mut(), preferred).unwrap();
        assert_eq!((id, selected), (1, FSR3_VERSION));
        assert!(select_version(versions, null_mut(), &[FSR4_VERSION]).is_err());
        assert!(!is_version("4.1.20", FSR4_VERSION));
    }

    #[test]
    fn resource_states_read_as_the_pc_backend_reads_them() {
        assert_eq!(host_state(0xc), Some(0xc0));
        assert_eq!(host_state(0x2), Some(0x8));
        assert_eq!(host_state(0x200), Some(0x10));
        assert_eq!(host_state(0x14), None);
    }
}
