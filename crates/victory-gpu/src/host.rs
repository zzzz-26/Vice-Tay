//! The host GPU: one D3D12 device and direct queue (decision 0007), and a
//! window to present on.
//!
//! shortcut: every submission is waited for, so CPU and GPU never overlap
//! (unless `VICTORY_LATE_WAIT=1`, see [`late`]);
//! overlap them with per-submission fences once more than clears run.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};

use windows::Win32::Foundation::{HANDLE, HWND, LPARAM, LRESULT, RECT, WPARAM};
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::Win32::UI::WindowsAndMessaging::*;
use windows::core::{Interface, PCWSTR, w};

/// Resource states without the console's own bits (compression hints from
/// bit 25 up), which the host rejects.
pub(crate) fn pc_state(state: u32) -> u32 {
    state & 0x01ff_ffff
}

/// A console state as a host texture state: PC bits only, and none of the
/// buffer-only ones the console allows on textures (vertex, constant and
/// index buffer, stream out, indirect arguments, as in "generic read").
pub(crate) fn texture_state(state: u32) -> u32 {
    const BUFFER_ONLY: u32 = 0x1 | 0x2 | 0x100 | 0x200;
    pc_state(state) & !BUFFER_ONLY
}

/// Render target views live only until their command is recorded, so a
/// ring of host descriptors is enough.
const RTV_RING: u32 = 1024;

pub(crate) struct Host {
    pub(crate) device: ID3D12Device,
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    /// The last value signalled on `fence`, behind the submission lock.
    submitted: Mutex<u64>,
    event: HANDLE,
    rtvs: ID3D12DescriptorHeap,
    rtv_size: u32,
    /// A ring of depth views, used as `rtvs`.
    dsvs: ID3D12DescriptorHeap,
    dsv_size: u32,
    next_dsv: AtomicU32,
    /// With `VICTORY_D3D12_DEBUG` set (diagnostics builds): the debug
    /// layer's messages.
    messages: Option<ID3D12InfoQueue>,
    next_rtv: AtomicU32,
    presenter: Mutex<Option<Presenter>>,
    title_stock: title_stock::Stock,
    /// The adapter the device is on, for its memory budget
    /// ([`Host::video_memory`]); `None` if the system does not name it.
    adapter: Option<IDXGIAdapter3>,
    /// 1-descriptor shader-visible fallback heaps bound when a command list
    /// omits a view or sampler heap while using directly-indexed root signatures.
    fallback_view: OnceLock<Option<ID3D12DescriptorHeap>>,
    fallback_sampler: OnceLock<Option<ID3D12DescriptorHeap>>,
    texture_readbacks: ReadbackPool<ID3D12Resource>,
}

// SAFETY: D3D12 devices, queues, fences and descriptor heaps are
// free-threaded; the event handle is only waited on under `submitted`.
unsafe impl Send for Host {}
// SAFETY: as above.
unsafe impl Sync for Host {}

/// Host objects the title let go of while a submission may still use them
/// (a texture its descriptors name, a descriptor slot's old resource), by
/// the submission count they wait for. The title's descriptor writes and
/// releases reach the host at once, from its own threads, while the host
/// runs a submission: freeing there is a GPU page fault on the freed
/// texture (a device hang).
/// In parts, a thread always in the same one: the list threads and those of
/// the heaps' records retire some 10,000 views a frame, and stood 12 ms a frame
/// for one lock of them all (run city1).
static RETIRED: [Mutex<std::collections::VecDeque<(u64, Retired)>>; 16] =
    [const { Mutex::new(std::collections::VecDeque::new()) }; 16];

/// The calling thread's part of [`RETIRED`].
fn retired() -> std::sync::MutexGuard<'static, std::collections::VecDeque<(u64, Retired)>> {
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    thread_local! {
        static PART: usize = NEXT.fetch_add(1, Ordering::Relaxed) % RETIRED.len();
    }
    // (A thread that ends may come here after its locals are gone.)
    let part = PART.try_with(|p| *p).unwrap_or(0);
    RETIRED[part].lock().unwrap_or_else(|e| e.into_inner())
}

/// Submissions started and finished (waited for).
static STARTED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static FINISHED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// A retired object, dropped on whatever thread finishes the submission.
#[allow(dead_code)]
enum Retired {
    Any(Box<dyn std::any::Any>),
    /// What a descriptor slot showed: 20,000 a frame, without a box each.
    Resource(ID3D12Resource),
}
// SAFETY: what is retired is D3D12 objects (free-threaded) and the
// reference-counted wrappers holding them, dropped once.
unsafe impl Send for Retired {}

/// Milliseconds since the first call.
pub(crate) fn now_ms() -> u64 {
    static EPOCH: OnceLock<std::time::Instant> = OnceLock::new();
    EPOCH.get_or_init(std::time::Instant::now).elapsed().as_millis() as u64
}

/// [`now_ms`] to the system's tick (16 ms): a few nanoseconds, where the
/// performance counter's are 25. For what is noted tens of thousands of
/// times a frame and read in tenths of a second.
pub(crate) fn coarse_ms() -> u64 {
    use windows_sys::Win32::System::SystemInformation::GetTickCount64;
    // The same beginning as `now_ms`.
    static BEGAN: OnceLock<u64> = OnceLock::new();
    // SAFETY: a plain query.
    let now = unsafe { GetTickCount64() };
    now.saturating_sub(*BEGAN.get_or_init(|| now.saturating_sub(now_ms())))
}

/// This moment, for telling later how long ago it was ([`long_ago`]): the
/// frames begun so far (high half, `crate::frame::current`) and tenths of a
/// second since the start. Host copies of title memory, staging buffers and
/// host textures note when they were last used with it.
pub(crate) fn moment() -> u64 {
    (crate::frame::current() << 32) | ((coarse_ms() / 100) & 0xffff_ffff)
}

/// Whether `moment` lies at least `ms` milliseconds and `frames` frames
/// back. Both: at seconds per frame, what every frame uses is not touched
/// for seconds, and at many frames per second not for many frames. (A title
/// that never waits for frames has no frame count: nothing is then idle.)
pub(crate) fn long_ago(moment: u64, ms: u64, frames: u64) -> bool {
    let now = self::moment();
    let tenths = (now & 0xffff_ffff).saturating_sub(moment & 0xffff_ffff);
    (now >> 32).saturating_sub(moment >> 32) >= frames && tenths * 100 >= ms
}

/// Available physical RAM on the host in bytes, from `GlobalMemoryStatusEx`.
pub(crate) fn available_physical_ram() -> Option<u64> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    // SAFETY: plain data the call fills in.
    let mut m: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    m.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    // SAFETY: a sized structure.
    let ok = unsafe { GlobalMemoryStatusEx(&mut m) } != 0;
    ok.then_some(m.ullAvailPhys)
}

/// How long host copies of textures the title fills must have been idle to
/// be let go, in milliseconds, and how many bytes of them go at most (the
/// longest idle first): derived from [`crate::resource::room_percent`] so
/// eviction always begins just below the upload cutoff (2 points below,
/// trimming down to 5 points below; urgent eviction at 5 points above),
/// avoiding any dead zone when `VICTORY_TEXTURE_ROOM_PERCENT` is lowered.
/// At the default 90 %, eviction starts at 88 %, trims to 85 %, and turns
/// urgent at 95 %. `VICTORY_TEXTURE_KEEP=0`: [`vram_idle_ms`]' older rule,
/// all that are idle; `=1`: that rule with its two longer times.
///
/// (By those times everything idle for 2.5 s went from 75 % of the budget
/// on: 61,000 of them in the first 300 s of session play-1125-0203, at 78 to
/// 80 %, of which 13,000 were made and uploaded again.)
pub(crate) fn vram_keep(budget: u64, used: u64) -> (u64, u64) {
    vram_keep_for(budget, used, crate::resource::room_percent())
}

pub(crate) fn vram_keep_for(budget: u64, used: u64, room_pct: u64) -> (u64, u64) {
    const KEPT_MS: u64 = 600_000;
    const IDLE_MS: u64 = 5_000;
    const SHORT_MS: u64 = 1_500;
    if budget == 0 {
        return (IDLE_MS, u64::MAX);
    }
    let room = room_pct.clamp(50, 97);
    let start = room.saturating_sub(2);
    let target = room.saturating_sub(5);
    let urgent = (room + 5).min(98);
    let share = |percent: u64| budget / 100 * percent;
    if used < share(start) {
        (KEPT_MS, u64::MAX)
    } else if used < share(urgent) {
        (IDLE_MS, used.saturating_sub(share(target)))
    } else {
        (SHORT_MS, used.saturating_sub(share(target)))
    }
}

/// Effective idle timeout (in milliseconds) for VRAM-backed host textures,
/// scaled down under VRAM pressure. Never returns less than 1,000 ms so
/// handles not yet recorded into a list or slot remain protected.
///
/// With room on the card nothing has to go: in session play-1125-0127 88,000
/// of them (31 GB) went after 5 s with 7 GB of the card free, and 24,700
/// (12 GB) had to be made and uploaded again.
/// `VICTORY_TEXTURE_KEEP=0`: 5 s there too, as before.
pub(crate) fn vram_idle_ms(video_memory: Option<[u64; 4]>) -> u64 {
    static KEEP: OnceLock<bool> = OnceLock::new();
    let keep = *KEEP.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_KEEP").is_none_or(|v| v != "0"));
    vram_idle_for(video_memory, keep)
}

fn vram_idle_for(video_memory: Option<[u64; 4]>, keep: bool) -> u64 {
    const KEPT_MS: u64 = 600_000;
    const LONG_MS: u64 = 30_000;
    const NORMAL_MS: u64 = 5_000;
    const MODERATE_MS: u64 = 2_500;
    const HIGH_MS: u64 = 1_000;
    let Some([budget, used, ..]) = video_memory else {
        return NORMAL_MS;
    };
    if budget == 0 {
        return NORMAL_MS;
    }
    let used_pct = (u128::from(used) * 100) / u128::from(budget);
    if used_pct >= 85 {
        HIGH_MS
    } else if used_pct >= 75 {
        MODERATE_MS
    } else if !keep {
        NORMAL_MS
    } else if used_pct >= 68 {
        LONG_MS
    } else {
        KEPT_MS
    }
}

/// Whether new memory is made resident, and waited for, before anything can
/// name it (on by default; `VICTORY_WAIT_RESIDENT=0`: as before). A heap or
/// resource of the default kind is resident once it is made, but its
/// mapping into the card's address space is work of the system's that runs
/// later: only lists sent after it wait for it. Lists sent before run on,
/// and read our descriptor heaps as they are: a slot that shows a texture
/// in a new block is a page fault for them until the block is mapped. The
/// device losses of the night of 2026-11-25 were that (run n08: the fault
/// lay in a texture made in the same tenth of a second, in a new block).
fn wait_resident() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WAIT_RESIDENT").is_none_or(|v| v != "0"))
}

/// The flag memory is made with that [`resident`] waits for.
pub(crate) fn not_resident() -> D3D12_HEAP_FLAGS {
    if wait_resident() { D3D12_HEAP_FLAG_CREATE_NOT_RESIDENT } else { D3D12_HEAP_FLAG_NONE }
}

/// A queue of its own that new memory is waited for with ([`mapped`]).
struct Aside {
    queue: ID3D12CommandQueue,
    list: ID3D12CommandList,
    fence: ID3D12Fence,
    event: windows::Win32::Foundation::HANDLE,
    progress: mapping_fence::Progress,
    _keep: (ID3D12CommandAllocator, [ID3D12Resource; 2]),
}
// SAFETY: D3D12 objects are free-threaded, the event is waited on by one
// thread at a time (the lock).
unsafe impl Send for Aside {}

/// New memory of the card is mapped into its address space when this
/// returns: a copy of a few bytes is sent to a queue of ours after the
/// memory was made, which the system starts only when all mappings asked
/// for so far are done, and waited for. (`MakeResident` came back before
/// that: run n11 was lost as n08 was with it alone.) Best done off a hot
/// path: a tenth of a millisecond to some.
pub(crate) fn mapped(device: &ID3D12Device) -> windows::core::Result<()> {
    static ASIDE: Mutex<Option<Aside>> = Mutex::new(None);
    if !wait_resident() {
        return Ok(());
    }
    let started = std::time::Instant::now();
    let mut aside = ASIDE.lock().unwrap_or_else(|e| e.into_inner());
    // SAFETY: plain object creation; the objects the list names are kept.
    unsafe {
        if aside.is_none() {
            let buffer = |kind: D3D12_HEAP_TYPE, state: D3D12_RESOURCE_STATES| -> windows::core::Result<ID3D12Resource> {
                let mut made: Option<ID3D12Resource> = None;
                device.CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: kind, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                        Width: 256,
                        Height: 1,
                        DepthOrArraySize: 1,
                        MipLevels: 1,
                        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                        ..Default::default()
                    },
                    state,
                    None,
                    &mut made,
                )?;
                made.ok_or_else(windows::core::Error::empty)
            };
            let from = buffer(D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ)?;
            let to = buffer(D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST)?;
            let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_COPY,
                ..Default::default()
            })?;
            let allocator: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY)?;
            let list: ID3D12GraphicsCommandList = device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, &allocator, None)?;
            list.CopyBufferRegion(&to, 0, &from, 0, 16);
            list.Close()?;
            *aside = Some(Aside {
                queue,
                list: list.cast()?,
                fence: device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?,
                event: CreateEventW(None, false, false, None)?,
                progress: Default::default(),
                _keep: (allocator, [from, to]),
            });
        }
        let Some(a) = aside.as_mut() else { return Ok(()) };
        a.progress.run(
            || a.queue.ExecuteCommandLists(&[Some(a.list.clone())]),
            |value, attempt| mapping_fence::confirm(device, &a.queue, &a.fence, a.event, value, attempt),
        )?;
    }
    let us = started.elapsed().as_micros() as u64;
    MAPPED[0].fetch_add(1, Ordering::Relaxed);
    MAPPED[1].fetch_add(us, Ordering::Relaxed);
    MAPPED[2].fetch_max(us, Ordering::Relaxed);
    let n = MAPPED[0].load(Ordering::Relaxed);
    if n == 1 || n.is_multiple_of(200) {
        tracing::info!(
            "new card memory waited for until it is mapped (VICTORY_WAIT_RESIDENT=0: not), since the start: {n} times, {:.1} ms, the longest {:.1} ms",
            MAPPED[1].load(Ordering::Relaxed) as f64 / 1000.0,
            MAPPED[2].load(Ordering::Relaxed) as f64 / 1000.0
        );
    }
    Ok(())
}

/// Times, microseconds and the longest of [`mapped`].
static MAPPED: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// New memory (made with [`not_resident`]) is resident when this returns.
pub(crate) fn resident(device: &ID3D12Device, object: &impl Interface) -> windows::core::Result<()> {
    if !wait_resident() {
        return Ok(());
    }
    let started = std::time::Instant::now();
    let object: ID3D12Pageable = object.cast()?;
    // SAFETY: a live heap or resource of the device's.
    let made = unsafe { device.MakeResident(&[Some(object)]) };
    let us = started.elapsed().as_micros() as u64;
    RESIDENT[0].fetch_add(1, Ordering::Relaxed);
    RESIDENT[1].fetch_add(us, Ordering::Relaxed);
    RESIDENT[2].fetch_max(us, Ordering::Relaxed);
    static SAID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = RESIDENT[0].load(Ordering::Relaxed);
    if n >= 1000 * (SAID.load(Ordering::Relaxed) + 1) {
        SAID.store(n / 1000, Ordering::Relaxed);
        tracing::info!(
            "new memory waited for (VICTORY_WAIT_RESIDENT=0: not), since the start: {n} times, {:.1} ms, the longest {:.1} ms",
            RESIDENT[1].load(Ordering::Relaxed) as f64 / 1000.0,
            RESIDENT[2].load(Ordering::Relaxed) as f64 / 1000.0
        );
    }
    made
}

/// Times, microseconds and the longest of [`resident`].
static RESIDENT: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// Whether memory the card was given stays with it while there is room (on
/// by default; `VICTORY_KEEP_FREED=0`: it is given back as before). A shader
/// that reads past a view (a raw load through a typed, structured or
/// constant buffer view has no end on AMD: `indirect::tests::raw_loads_past_views`)
/// or through a slot that shows wrong data reads whatever lies there, and
/// nothing happens; where memory was given back it is a page fault, and the
/// device is lost. Both losses of 2026-11-25 were reads in memory given back
/// seconds before (a block of textures; a list's scratch and a window of
/// title memory). Kept are: empty blocks of textures ([`crate::pool`]) below
/// 85 % of the card's budget, windows of title memory and staging buffers
/// for five minutes with 12 GiB of the PC's memory free, and 1 GiB of
/// scratch chunks.
pub(crate) fn keep_freed() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_KEEP_FREED").is_none_or(|v| v != "0"))
}

/// Effective idle timeout (in milliseconds) for system-RAM caches (staging
/// buffers, loose shadows, and the mapped pipeline library), scaled down
/// when free physical RAM is low. Never returns less than 2,000 ms.
/// With plenty of it five minutes ([`keep_freed`]).
pub(crate) fn ram_idle_ms(avail_phys: Option<u64>) -> u64 {
    let effective = match (avail_phys, crate::residency::headroom()) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    };
    ram_idle_for(effective, keep_freed() && !crate::residency::limited_host_ram())
}

fn ram_idle_for(avail_phys: Option<u64>, keep: bool) -> u64 {
    const GIB: u64 = 1024 * 1024 * 1024;
    const KEPT_MS: u64 = 300_000;
    if keep && avail_phys.is_some_and(|avail| avail >= 12 * GIB) {
        return KEPT_MS;
    }
    const NORMAL_MS: u64 = 10_000;
    const MODERATE_MS: u64 = 5_000;
    const HIGH_MS: u64 = 2_000;
    let Some(avail) = avail_phys else {
        return NORMAL_MS;
    };
    if avail < 2 * GIB {
        HIGH_MS
    } else if avail < 4 * GIB {
        MODERATE_MS
    } else {
        NORMAL_MS
    }
}

/// System-RAM pressure band for optional caches. Resident working set and
/// private bytes describe this process; free physical and commit headroom
/// guard the PC and other processes. Windows API queries never change the
/// pagefile.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RamBand {
    Calm,
    Watch,
    High,
}

/// Band for one sample. Entering Watch or High needs a higher watermark than
/// leaving it, so a value near a threshold does not toggle the policy every
/// sample.
pub(crate) fn ram_band(
    previous: RamBand,
    working: u64,
    private: u64,
    free_physical: u64,
    commit_headroom: u64,
    total_physical: u64,
    commit_limit: u64,
) -> RamBand {
    const GIB: u64 = 1024 * 1024 * 1024;
    // Fractions of the machine, so a 64 GB host is not treated like the
    // 32 GB one. The free and headroom floors stay absolute: pressure is
    // pressure however big the machine is.
    let high = private * 5 >= commit_limit * 4
        || working * 2 >= total_physical
        || free_physical < 2 * GIB
        || commit_headroom < GIB;
    let watch = private * 10 >= commit_limit * 7
        || working * 4 >= total_physical
        || free_physical < 4 * GIB
        || commit_headroom < 2 * GIB;
    let wanted = if high {
        RamBand::High
    } else if watch {
        RamBand::Watch
    } else {
        RamBand::Calm
    };
    let safe_high = private * 20 <= commit_limit * 13
        && working * 4 <= total_physical
        && free_physical >= 4 * GIB
        && commit_headroom >= 3 * GIB;
    let safe_watch = private * 5 <= commit_limit * 3
        && working * 5 <= total_physical
        && free_physical >= 6 * GIB
        && commit_headroom >= 3 * GIB;
    match (previous, wanted) {
        (RamBand::High, RamBand::Calm) if !safe_high => RamBand::Watch,
        (RamBand::Watch, RamBand::Calm) if !safe_watch => RamBand::Watch,
        _ => wanted,
    }
}

/// The six numbers [`ram_band`] needs, from Windows. `None` if a query fails,
/// so a caller can keep its old band instead of guessing.
#[cfg(windows)]
pub(crate) fn ram_sample() -> Option<(u64, u64, u64, u64, u64, u64)> {
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut m: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    m.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    // SAFETY: plain data the call fills in.
    if unsafe { GlobalMemoryStatusEx(&mut m) } == 0 {
        return None;
    }
    let mut c = PROCESS_MEMORY_COUNTERS {
        cb: std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    if unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) } == 0 {
        return None;
    }
    Some((
        c.WorkingSetSize as u64,
        c.PagefileUsage as u64,
        m.ullAvailPhys,
        m.ullAvailPageFile,
        m.ullTotalPhys,
        m.ullTotalPageFile,
    ))
}

/// How many references are held on a host object (its COM count): ours
/// all, since the host's views and commands hold none. A buffer or texture
/// with one is held by its owner alone: no descriptor slot keeps it
/// (`HostHeap::kept`) and no list (`HostList::keep`).
pub(crate) fn references(object: &impl Interface) -> u32 {
    type Method = unsafe extern "system" fn(*mut std::ffi::c_void) -> u32;
    let raw = object.as_raw();
    // SAFETY: a live COM object, whose first three methods are those of
    // `IUnknown` (query, add a reference, release one); the pair leaves the
    // count as it was, and the release returns it. The first is not called.
    unsafe {
        let methods = *raw.cast::<*const [Method; 3]>();
        ((*methods)[1])(raw);
        ((*methods)[2])(raw)
    }
}

/// Once a second, before a submission starts: lets go of what nothing
/// names any more (copies of windows of title memory, staging buffers,
/// host copies of textures the title fills).
/// `VICTORY_SWEEP=0` keeps everything, as before.
fn sweep() {
    // The in-place probe and census, if asked for (`VICTORY_INPLACE_PROBE`).
    crate::inplace::tick();
    static ON: OnceLock<bool> = OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("VICTORY_SWEEP").is_none_or(|v| v != "0")) {
        return;
    }
    static LAST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let now = now_ms();
    let last = LAST.load(Ordering::Relaxed);
    if now < last + 1000
        || LAST
            .compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
    {
        return;
    }
    #[cfg(feature = "diagnostics")]
    crate::book::check();
    crate::residency::refresh();
    if crate::residency::pressured() {
        trim_spare_chunks();
    }
    let host = host();
    let memory = host.and_then(|h| h.video_memory());
    if let Some(host) = host { host.trim_title_stock(memory); }
    crate::pool::room(memory);
    crate::pipeline::trim_library();
    crate::shadow::let_go_idle();
    crate::texture::let_go_idle();
    crate::resource::let_go_idle();
}

/// The number of the newest submission begun, and of the newest of which all
/// before it have finished too.
pub(crate) fn submissions() -> (u64, u64) {
    (STARTED.load(Ordering::Acquire), FINISHED.load(Ordering::Acquire))
}

fn insert_retired(queue: &mut std::collections::VecDeque<(u64, Retired)>, entry: (u64, Retired)) {
    if queue.back().is_none_or(|(at, _)| *at <= entry.0) {
        queue.push_back(entry);
    } else {
        let index = queue.partition_point(|(at, _)| *at <= entry.0);
        queue.insert(index, entry);
    }
}

/// Keep each shard ordered by the original submission stamp. A worker can
/// hand off an older batch after that shard has already received newer entries.
fn merge_retired_batch(queue: &mut std::collections::VecDeque<(u64, Retired)>, batch: &mut Vec<(u64, Retired)>) {
    if batch.is_empty() {
        return;
    }
    batch.sort_by_key(|(at, _)| *at);
    let oldest = batch[0].0;
    if queue.back().is_none_or(|(at, _)| *at <= oldest) {
        queue.extend(batch.drain(..));
        return;
    }

    let index = queue.partition_point(|(at, _)| *at <= oldest);
    let mut tail = queue.split_off(index).into_iter().peekable();
    for entry in batch.drain(..) {
        while tail.peek().is_some_and(|(at, _)| *at <= entry.0) {
            queue.push_back(tail.next().unwrap());
        }
        queue.push_back(entry);
    }
    queue.extend(tail);
}

/// Drops `object` once every submission running now has finished: at once
/// if none is.
pub(crate) fn retire<T: 'static>(object: T) {
    // (Those the title made and the host has not begun get the next numbers.)
    let queued = crate::queue::queued();
    let started = STARTED.load(Ordering::Acquire) + queued;
    if bury_lag() == 0 && started <= FINISHED.load(Ordering::Acquire) {
        RETIREMENTS[0].fetch_add(1, Ordering::Relaxed);
        drop(object);
        return;
    }
    RETIREMENTS[1].fetch_add(1, Ordering::Relaxed);
    insert_retired(&mut retired(), (started, Retired::Any(Box::new(object))));
}

/// Whether list-translation worker threads accumulate retired `ID3D12Resource`s
/// in a thread-local batch (`VICTORY_BATCH_RETIRE=1`, opt-in, off by default;
/// `VICTORY_AB` can flip it live in `diagnostics` builds).
fn batch_retire_enabled() -> bool {
    #[cfg(feature = "diagnostics")]
    if let Some(on) = crate::ab::pick("VICTORY_BATCH_RETIRE") {
        return on;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| parse_batch_retire_switch(std::env::var("VICTORY_BATCH_RETIRE").ok().as_deref()))
}

fn parse_batch_retire_switch(raw: Option<&str>) -> bool {
    raw.is_some_and(|v| v.trim() == "1")
}

/// [`retire`] for a resource.
pub(crate) fn retire_resource(object: ID3D12Resource) {
    retire_resource_with(object, batch_retire_enabled());
}

fn retire_resource_with(object: ID3D12Resource, batch_on: bool) {
    let queued = crate::queue::queued();
    let started = STARTED.load(Ordering::Acquire) + queued;
    if bury_lag() == 0 && started <= FINISHED.load(Ordering::Acquire) {
        RETIREMENTS[0].fetch_add(1, Ordering::Relaxed);
        drop(object);
        return;
    }
    RETIREMENTS[1].fetch_add(1, Ordering::Relaxed);
    let mut entry = Some((started, Retired::Resource(object)));
    if !batch_on {
        // Flush any buffered entries left over if toggled off mid-run.
        flush_retired();
        if let Some(back) = entry {
            insert_retired(&mut retired(), back);
        }
        return;
    }
    let flush = BATCH.try_with(|b| {
        let taken = entry.take().expect("entry present");
        let mut guard = b.borrow_mut();
        let target = guard.eligible.then_some(&mut guard.entries);
        into_batch(target, taken, BATCH_MOST)
    });
    match flush {
        Ok(Err(back)) => insert_retired(&mut retired(), back),
        Ok(Ok(true)) => flush_retired(),
        Ok(Ok(false)) => {}
        Err(_) => {
            if let Some(back) = entry {
                insert_retired(&mut retired(), back);
            }
        }
    }
}

/// Pushes `entry` into the thread's batch if enabled; returns `Ok(true)` once
/// the batch reaches `most` and should be handed over, or `Err(entry)` if the
/// thread is not batching.
fn into_batch<T>(batch: Option<&mut Vec<T>>, entry: T, most: usize) -> Result<bool, T> {
    match batch {
        Some(batch) => {
            batch.push(entry);
            Ok(batch.len() >= most)
        }
        None => Err(entry),
    }
}

/// Number of retired resources a worker thread accumulates before handing them
/// over to [`RETIRED`] in one lock acquisition (after voidopcode/vilocity
/// `4315c77`).
const BATCH_MOST: usize = 256;

/// Thread-local retired resource batch that flushes any remaining entries on
/// thread exit so retired D3D12 objects are never dropped prematurely.
struct RetiredBatch {
    eligible: bool,
    entries: Vec<(u64, Retired)>,
}

impl Drop for RetiredBatch {
    fn drop(&mut self) {
        if !self.entries.is_empty() {
            merge_retired_batch(&mut retired(), &mut self.entries);
        }
    }
}

thread_local! {
    /// Resources retired on this thread and not yet handed to [`RETIRED`]
    /// ([`retire_in_batches`]).
    static BATCH: std::cell::RefCell<RetiredBatch> = const {
        std::cell::RefCell::new(RetiredBatch { eligible: false, entries: Vec::new() })
    };
}

/// Marks the calling thread as eligible for opt-in thread-local batching in
/// [`retire_resource`] (`VICTORY_BATCH_RETIRE=1`). Idempotent: calling it
/// repeatedly never discards already-buffered entries.
///
/// Worker threads translating command lists retire thousands of replaced view
/// resources per frame; accumulating up to [`BATCH_MOST`] items and flushing
/// at recording boundaries amortizes lock acquisitions on [`RETIRED`] while
/// preserving the stamp that determines when each resource can be released.
pub(crate) fn retire_in_batches() {
    let _ = BATCH.try_with(|b| {
        let mut guard = b.borrow_mut();
        if !guard.eligible {
            guard.eligible = true;
            if guard.entries.capacity() < BATCH_MOST {
                let need = BATCH_MOST.saturating_sub(guard.entries.len());
                guard.entries.reserve(need);
            }
        }
    });
}

/// Hands over every resource retired on this thread since the last flush
/// ([`retire_in_batches`]).
pub(crate) fn flush_retired() {
    let _ = BATCH.try_with(|b| {
        let mut guard = b.borrow_mut();
        if !guard.entries.is_empty() {
            merge_retired_batch(&mut retired(), &mut guard.entries);
        }
    });
}

#[cfg(test)]
mod retire_batch_tests {
    use super::*;

    /// Serializes tests that advance `STARTED` and call `bury()` when `cargo test`
    /// runs default unit tests in parallel.
    static RETIRE_TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn a_late_batch_does_not_hide_eligible_retirements() {
        let item = |stamp| (stamp, Retired::Any(Box::new(())));
        let mut queue = std::collections::VecDeque::new();
        insert_retired(&mut queue, item(10));
        insert_retired(&mut queue, item(14));
        insert_retired(&mut queue, item(9));

        let mut pending = vec![item(13), item(8), item(12)];
        merge_retired_batch(&mut queue, &mut pending);

        assert!(pending.is_empty());
        let stamps = queue.iter().map(|(stamp, _)| *stamp).collect::<Vec<_>>();
        assert_eq!(stamps, [8, 9, 10, 12, 13, 14]);
        assert_eq!(queue.iter().take_while(|(stamp, _)| *stamp <= 10).count(), 3);

        pending.push(item(16));
        merge_retired_batch(&mut queue, &mut pending);
        assert_eq!(queue.back().unwrap().0, 16);
    }

    #[test]
    fn batch_retire_switch_defaults_off_and_accepts_only_one() {
        assert!(!parse_batch_retire_switch(None));
        assert!(!parse_batch_retire_switch(Some("0")));
        assert!(!parse_batch_retire_switch(Some("")));
        assert!(!parse_batch_retire_switch(Some("false")));
        assert!(!parse_batch_retire_switch(Some("2")));
        assert!(parse_batch_retire_switch(Some("1")));
        assert!(parse_batch_retire_switch(Some(" 1 ")));
    }

    #[test]
    fn a_thread_without_a_batch_keeps_nothing_and_a_full_batch_signals_flush() {
        assert_eq!(into_batch(None, 7, 2), Err(7));
        let mut batch = Vec::new();
        assert_eq!(into_batch(Some(&mut batch), 7, 2), Ok(false));
        assert_eq!(into_batch(Some(&mut batch), 8, 2), Ok(true));
        assert_eq!(batch, [7, 8]);
    }

    #[test]
    fn batched_retirement_off_hands_over_immediately_and_flushes_leftovers() {
        let Some(host) = host() else {
            return;
        };
        let _lock = RETIRE_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _ = host.log_messages();
        let _ = STARTED.fetch_add(1, Ordering::AcqRel);
        std::thread::scope(|s| {
            s.spawn(|| {
                retire_in_batches();
                let (b1, _) = host.upload_buffer(256).expect("upload buffer");
                retire_resource_with(b1, false);
                assert_eq!(BATCH.with(|b| b.borrow().entries.len()), 0, "off mode never buffers");

                // Buffer 3 entries while on, then toggle off: leftovers must flush immediately.
                for _ in 0..3 {
                    let (b, _) = host.upload_buffer(256).expect("upload buffer");
                    retire_resource_with(b, true);
                }
                assert_eq!(BATCH.with(|b| b.borrow().entries.len()), 3);
                let (b4, _) = host.upload_buffer(256).expect("upload buffer");
                retire_resource_with(b4, false);
                assert_eq!(BATCH.with(|b| b.borrow().entries.len()), 0, "toggling off flushes buffered entries");
            });
        });
        let end_stamp = STARTED.load(Ordering::Acquire) + crate::queue::queued();
        bury(end_stamp.saturating_add(bury_lag()));
        assert_eq!(host.log_messages(), None);
    }

    #[test]
    fn retire_in_batches_is_idempotent_with_buffered_entries() {
        let Some(host) = host() else {
            return;
        };
        let _lock = RETIRE_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _ = host.log_messages();
        let _ = STARTED.fetch_add(1, Ordering::AcqRel);
        std::thread::scope(|s| {
            s.spawn(|| {
                retire_in_batches();
                for _ in 0..5 {
                    let (buf, _) = host.upload_buffer(256).expect("upload buffer");
                    retire_resource_with(buf, true);
                }
                assert_eq!(BATCH.with(|b| b.borrow().entries.len()), 5);
                // Calling retire_in_batches again must keep buffered entries intact.
                retire_in_batches();
                assert_eq!(BATCH.with(|b| b.borrow().entries.len()), 5);
                flush_retired();
                assert_eq!(BATCH.with(|b| b.borrow().entries.len()), 0);
            });
        });
        let end_stamp = STARTED.load(Ordering::Acquire) + crate::queue::queued();
        bury(end_stamp.saturating_add(bury_lag()));
        assert_eq!(host.log_messages(), None);
    }

    #[test]
    fn batched_resource_retirement_flushes_concurrently_without_losing_entries() {
        let Some(host) = host() else {
            return;
        };
        let _lock = RETIRE_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _ = host.log_messages();
        const THREADS: usize = 4;
        const PER_THREAD: usize = 300; // exceeds BATCH_MOST (256) to exercise auto-flush + tail flush
        let before = RETIREMENTS[1].load(Ordering::Relaxed);
        let _ = STARTED.fetch_add(1, Ordering::AcqRel);
        std::thread::scope(|s| {
            // Joined one by one: the scope's own wait can end before a thread's
            // thread-locals are dropped, and with them `RetiredBatch::drop`.
            let workers: Vec<_> = (0..THREADS)
                .map(|t| {
                    s.spawn(move || {
                        retire_in_batches();
                        for _ in 0..PER_THREAD {
                            let (buf, _) = host.upload_buffer(256).expect("upload buffer");
                            retire_resource_with(buf, true);
                        }
                        // Explicitly flush half the threads; let the others exercise RetiredBatch::drop.
                        if t % 2 == 0 {
                            flush_retired();
                        }
                    })
                })
                .collect();
            for worker in workers {
                worker.join().expect("retirement worker");
            }
        });
        let queued_now = RETIREMENTS[1].load(Ordering::Relaxed) - before;
        assert!(
            queued_now >= (THREADS * PER_THREAD) as u64,
            "expected at least {} queued retirements, got {queued_now}",
            THREADS * PER_THREAD
        );
        let end_stamp = STARTED.load(Ordering::Acquire) + crate::queue::queued();
        bury(end_stamp.saturating_add(bury_lag()));
        assert_eq!(host.log_messages(), None);
    }

    #[test]
    fn batched_retirement_preserves_resource_lifetime_until_bury_on_gpu() {
        struct SendResource(ID3D12Resource);
        // SAFETY: D3D12 resources are free-threaded (same invariant as `Retired`).
        unsafe impl Send for SendResource {}

        let Some(host) = host() else {
            return;
        };
        let _lock = RETIRE_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _ = host.log_messages();
        let stamp_before = STARTED.fetch_add(1, Ordering::AcqRel) + 1 + crate::queue::queued();
        let (buf_explicit, _) = host.upload_buffer(256).expect("upload buffer");
        let probe_explicit = buf_explicit.clone();
        let (buf_drop, _) = host.upload_buffer(256).expect("upload buffer");
        let probe_drop = buf_drop.clone();
        assert_eq!(references(&probe_explicit), 2);
        assert_eq!(references(&probe_drop), 2);
        let send_explicit = SendResource(buf_explicit);
        let send_drop = SendResource(buf_drop);

        std::thread::scope(|s| {
            // Joined explicitly: the scope's own wait can end before the second
            // thread's thread-locals are dropped, and with them `RetiredBatch::drop`.
            let explicit = s.spawn(move || {
                let SendResource(buf) = send_explicit;
                let probe = buf.clone();
                retire_in_batches();
                retire_resource_with(buf, true);
                assert_eq!(BATCH.with(|b| b.borrow().entries.len()), 1);
                assert!(references(&probe) >= 3, "buffered entry dropped early");
                drop(probe);
                flush_retired();
                assert_eq!(BATCH.with(|b| b.borrow().entries.len()), 0);
            });
            let dropped = s.spawn(move || {
                let SendResource(buf) = send_drop;
                retire_in_batches();
                retire_resource_with(buf, true);
                assert_eq!(BATCH.with(|b| b.borrow().entries.len()), 1);
                // Intentionally exit without flush_retired() to exercise RetiredBatch::drop.
            });
            explicit.join().expect("explicit flush worker");
            dropped.join().expect("drop flush worker");
        });
        let stamp_after = STARTED.load(Ordering::Acquire) + crate::queue::queued();

        // Both resources must remain alive in RETIRED before their stamped submission finishes.
        bury(stamp_before.saturating_sub(1).saturating_add(bury_lag()));
        assert!(references(&probe_explicit) >= 2, "explicitly flushed resource dropped before bury");
        assert!(references(&probe_drop) >= 2, "drop-flushed resource dropped before bury");

        // Once the stamped submission plus bury_lag finishes, both must be released.
        bury(stamp_after.saturating_add(bury_lag()));
        assert_eq!(references(&probe_explicit), 1, "explicitly flushed resource leaked after bury");
        assert_eq!(references(&probe_drop), 1, "drop-flushed resource leaked after bury");
        assert_eq!(host.log_messages(), None);
    }

    #[test]
    fn mixed_batched_and_direct_retirement_releases_older_resource_at_its_own_deadline_on_gpu() {
        struct SendResource(ID3D12Resource);
        // SAFETY: D3D12 resources are free-threaded (same invariant as `Retired`).
        unsafe impl Send for SendResource {}

        let Some(host) = host() else {
            return;
        };
        let _lock = RETIRE_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _ = host.log_messages();
        let lag = bury_lag();

        for (batch_on, explicit_flush) in [(false, true), (true, true), (true, false)] {
            let (buf_n, _) = host.upload_buffer(256).expect("upload buffer");
            let probe_n = buf_n.clone();
            let (buf_n1, _) = host.upload_buffer(256).expect("upload buffer");
            let probe_n1 = buf_n1.clone();
            assert_eq!(references(&probe_n), 2);
            assert_eq!(references(&probe_n1), 2);
            let send_n = SendResource(buf_n);
            let send_n1 = SendResource(buf_n1);

            let (stamp_before_n, stamp_after_n, stamp_after_n1) = std::thread::scope(|s| {
                s.spawn(move || {
                    let SendResource(buf_n) = send_n;
                    let SendResource(buf_n1) = send_n1;
                    retire_in_batches();

                    let before_n = STARTED.fetch_add(1, Ordering::AcqRel) + 1 + crate::queue::queued();
                    retire_resource_with(buf_n, batch_on);
                    let after_n = STARTED.load(Ordering::Acquire) + crate::queue::queued();

                    // Advance stamp strictly past `after_n`, then push direct retirements on the
                    // same thread before handing over the older batched resource.
                    STARTED.fetch_add(1, Ordering::AcqRel);
                    retire(());
                    retire(buf_n1);
                    let after_n1 = STARTED.load(Ordering::Acquire) + crate::queue::queued();

                    if explicit_flush {
                        flush_retired();
                    }
                    // When `explicit_flush == false`, thread exit exercises `RetiredBatch::drop`.
                    (before_n, after_n, after_n1)
                })
                .join()
                .expect("mixed retirement worker")
            });

            // Neither resource may drop before `buf_n`'s fence deadline.
            bury(stamp_before_n.saturating_sub(1).saturating_add(lag));
            assert!(
                references(&probe_n) >= 2,
                "batch_on={batch_on} explicit={explicit_flush}: older resource dropped before its deadline"
            );
            assert!(
                references(&probe_n1) >= 2,
                "batch_on={batch_on} explicit={explicit_flush}: newer resource dropped before older deadline"
            );

            // At `buf_n`'s own fence deadline, `buf_n` must drop to refcount 1 immediately
            // (not stalled behind the newer direct `retire(())`/`buf_n1` entries), while
            // `buf_n1` remains alive until its own deadline.
            bury(stamp_after_n.saturating_add(lag));
            assert_eq!(
                references(&probe_n),
                1,
                "batch_on={batch_on} explicit={explicit_flush}: older resource stuck behind newer direct entry"
            );
            assert!(
                references(&probe_n1) >= 2,
                "batch_on={batch_on} explicit={explicit_flush}: newer resource dropped at older deadline"
            );

            // Once `buf_n1`'s deadline passes, it releases cleanly too.
            bury(stamp_after_n1.saturating_add(lag));
            assert_eq!(
                references(&probe_n1),
                1,
                "batch_on={batch_on} explicit={explicit_flush}: newer resource leaked after its deadline"
            );
        }
        assert_eq!(host.log_messages(), None);
    }
}

/// How many submissions later than it could (the last one that may use it
/// has finished) an object let go is dropped, and a host list that has run
/// is reset, about two frames: the device losses of the night of 2026-11-25
/// were page faults in memory nothing of ours named any more, at no command
/// in particular. Objects let go while nothing runs wait as well: lists
/// recorded and not yet sent may name them. `VICTORY_BURY_LAG=0`: not.
fn bury_lag() -> u64 {
    static LAG: OnceLock<u64> = OnceLock::new();
    *LAG.get_or_init(|| {
        let default = if crate::residency::limited_host_ram() { 16 } else { 64 };
        std::env::var("VICTORY_BURY_LAG").ok().and_then(|v| v.parse().ok()).unwrap_or(default)
    })
}

/// Drops the eligible prefix of each stamp-ordered retirement shard.
fn bury(finished: u64) {
    let finished = finished.saturating_sub(bury_lag());
    for part in &RETIRED {
        let done: Vec<(u64, Retired)> = {
            let mut retired = part.lock().unwrap_or_else(|e| e.into_inner());
            let n = retired.iter().take_while(|(at, _)| *at <= finished).count();
            if n == 0 {
                continue;
            }
            retired.drain(..n).collect()
        };
        drop(done);
    }
}

/// Objects [`retire`] dropped at once (no submission was running) and
/// kept until a running submission finished, since the start.
static RETIREMENTS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// What the layer did to the host's objects lately, newest last (sweeps
/// that let host copies go, pipeline library calls, saves and trims): said
/// once when the host device is removed ([`log_removal_context`]), with
/// what the submitting thread and the pipeline workers were doing at that
/// moment. A device the driver removes "internally" names no command, and
/// the pass that happens to notice first names only the view it was at:
/// the three such losses of runs 72, 73 and 82 each came within 5 ms of a
/// pipeline library call that no line of the log tied to them.
/// `VICTORY_TRAIL=0`: nothing is kept and nothing is said.
static TRAIL: Mutex<std::collections::VecDeque<(u64, u32, String)>> = Mutex::new(std::collections::VecDeque::new());
/// The entries kept.
const TRAIL_MOST: usize = 96;

fn trail_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TRAIL").is_none_or(|v| v != "0"))
}

/// Microseconds since the first call.
pub(crate) fn now_us() -> u64 {
    static EPOCH: OnceLock<std::time::Instant> = OnceLock::new();
    EPOCH.get_or_init(std::time::Instant::now).elapsed().as_micros() as u64
}

/// The calling thread, as the log's other lines name threads.
pub(crate) fn thread_id() -> u32 {
    // SAFETY: a plain query.
    unsafe { windows::Win32::System::Threading::GetCurrentThreadId() }
}

/// Notes something done to the host's objects ([`TRAIL`]); `what` is only
/// run when the trail is kept.
pub(crate) fn trail(what: impl FnOnce() -> String) {
    if !trail_on() {
        return;
    }
    let entry = (now_us(), thread_id(), what());
    let mut trail = TRAIL.lock().unwrap_or_else(|e| e.into_inner());
    if trail.len() >= TRAIL_MOST {
        trail.pop_front();
    }
    trail.push_back(entry);
}

/// What the submitting thread is doing (0 nothing: between submissions,
/// 1 preparing one, 2 waiting for the card to run it, 3 copying back what
/// it wrote), the lists of that submission, and since when ([`now_us`]).
static SUBMITTING: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

fn submitting(phase: u64, lists: usize) {
    SUBMITTING[1].store(lists as u64, Ordering::Relaxed);
    SUBMITTING[2].store(now_us(), Ordering::Relaxed);
    SUBMITTING[0].store(phase, Ordering::Relaxed);
}

/// What was going on when the host device went: whether a submission was
/// on the card, the pipeline library's state and the calls running in it,
/// and the trail.
pub(crate) fn removal_context() -> String {
    use std::fmt::Write;
    let now = now_us();
    let ms = |at: u64| now.saturating_sub(at) as f64 / 1000.0;
    let phase = match SUBMITTING[0].load(Ordering::Relaxed) {
        0 => "between submissions (nothing of ours was running on the card)",
        1 => "preparing a submission (nothing of ours was running on the card)",
        2 => "waiting for the card to run a submission",
        _ => "copying back what a submission wrote (nothing of ours was running on the card)",
    };
    let mut text = format!(
        "the submitting thread was {phase}, of {} lists, for {:.1} ms; {}; objects let go at once {} and after a running submission {}; submissions started {} finished {}",
        SUBMITTING[1].load(Ordering::Relaxed),
        ms(SUBMITTING[2].load(Ordering::Relaxed)),
        crate::pipeline::library_removal_note(),
        RETIREMENTS[0].load(Ordering::Relaxed),
        RETIREMENTS[1].load(Ordering::Relaxed),
        STARTED.load(Ordering::Relaxed),
        FINISHED.load(Ordering::Relaxed),
    );
    let trail = TRAIL.lock().unwrap_or_else(|e| e.into_inner());
    let _ = write!(text, "; the last {} things done, newest last:", trail.len());
    for (at, thread, what) in trail.iter() {
        let _ = write!(text, "\n  {:.1} ms before, thread {thread}: {what}", ms(*at));
    }
    text
}

/// Says [`removal_context`] in the log, once: called by whoever finds the
/// host device removed.
pub(crate) fn log_removal_context() {
    static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !(trail_on() || dred_any()) || SAID.swap(true, Ordering::Relaxed) {
        return;
    }
    tracing::error!("when the host device went: {}", removal_context());
    if dred_any() {
        let data = host().map(|h| h.removal_data(&[])).unwrap_or_default();
        tracing::error!("device-removed data{}", if data.is_empty() { "; none" } else { &data });
    }
}

/// Time spent in submissions since the last frame-rate line, in
/// microseconds: preparing them on the CPU (title memory copies, texture
/// detiling), then waiting for the GPU.
static SUBMISSION_US: [std::sync::atomic::AtomicU64; 2] =
    [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// After submissions: shader-written ranges compared and copied back into
/// title memory (microseconds, bytes that differed), since the last
/// frame-rate line.
static COPY_BACK: [std::sync::atomic::AtomicU64; 2] =
    [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// The card's clock, for the frame line of every run, in microseconds since
/// the last one: how long the card had nothing of ours queued; how much of
/// that while the thread that holds the submission lock was working (the
/// card idle behind us: preparing and copying back are done with the card
/// standing); and how long that thread stood waiting for the card.
///
/// Every send of lists to the queue is followed by a signal on a fence of
/// the clock's own ([`Host::card_sent`]). A thread of the clock waits for
/// the newest signal and notes the moment the card reached it with nothing
/// newer sent; the next send ends that idle stretch.
static CARD_CLOCK_US: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// Of the thread that holds the submission lock: the microseconds it has
/// worked so far (preparing, copying back; not its waits for the card), and
/// since when it works now ([`now_us`] plus one; 0: it does not).
static WORKING: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// The thread that holds the submission lock starts (`on`) or stops
/// working; whether it worked until now.
fn working(on: bool) -> bool {
    let now = now_us() + 1;
    let since = WORKING[1].swap(if on { now } else { 0 }, Ordering::Relaxed);
    if since != 0 {
        WORKING[0].fetch_add(now.saturating_sub(since), Ordering::Relaxed);
    }
    since != 0
}

/// [`WORKING`] as of now: microseconds worked so far.
fn worked() -> u64 {
    let since = WORKING[1].load(Ordering::Relaxed);
    let now = if since != 0 { (now_us() + 1).saturating_sub(since) } else { 0 };
    WORKING[0].load(Ordering::Relaxed) + now
}

/// The thread that holds the submission lock stood waiting for the card
/// from `since` until now ([`CARD_CLOCK_US`]).
fn stood(since: std::time::Instant) {
    CARD_CLOCK_US[2].fetch_add(since.elapsed().as_micros() as u64, Ordering::Relaxed);
}

/// What [`CARD_CLOCK_US`] is measured with.
struct CardClock {
    fence: ID3D12Fence,
    /// Signals sent so far; and, while the card has reached the last of
    /// them, since when and what [`worked`] said then.
    state: Mutex<(u64, Option<(std::time::Instant, u64)>)>,
    sent: std::sync::Condvar,
}

// SAFETY: a fence (free-threaded) behind a lock.
unsafe impl Send for CardClock {}
// SAFETY: as above.
unsafe impl Sync for CardClock {}

/// The clock's thread: waits for the newest signal sent and notes when the
/// card got there with nothing newer behind it.
fn card_clock_watch(clock: &'static CardClock) {
    // SAFETY: a plain event of our own.
    let Ok(event) = (unsafe { CreateEventW(None, false, false, None) }) else {
        return;
    };
    let mut seen = 0;
    loop {
        let target = {
            let mut state = clock.state.lock().unwrap_or_else(|e| e.into_inner());
            while state.0 == seen {
                state = clock.sent.wait(state).unwrap_or_else(|e| e.into_inner());
            }
            state.0
        };
        // SAFETY: the clock's fence and this thread's event.
        unsafe {
            if clock.fence.SetEventOnCompletion(target, event).is_ok() {
                WaitForSingleObject(event, INFINITE);
            }
        }
        seen = target;
        let mut state = clock.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.0 == target && state.1.is_none() {
            state.1 = Some((std::time::Instant::now(), worked()));
        }
    }
}

/// For the frame-rate line: [`CARD_CLOCK_US`] per frame, `ms` being the
/// frame's time.
fn card_clock_line(frames: f64, ms: f64) {
    let [idle, behind, stood] = [0, 1, 2].map(|i| CARD_CLOCK_US[i].swap(0, Ordering::Relaxed) as f64 / 1000.0 / frames);
    tracing::info!(
        "the card's clock per frame: nothing of ours queued {idle:.1} ms (the card about {:.0}% busy by this count), \
         {behind:.1} ms of that while the submitting thread was working (the card idle behind us), the rest while the title \
         had not submitted; the submitting thread stood waiting for the card {stood:.1} ms",
        if ms > 0.0 { (100.0 * (1.0 - idle / ms)).clamp(0.0, 100.0) } else { 0.0 }
    );
}

/// Of the preparing: title memory copies (microseconds, bytes) and texture
/// refreshes (microseconds, bytes hashed), since the last frame-rate line.
pub(crate) static PREPARING: [std::sync::atomic::AtomicU64; 4] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

/// Host buffers made since the start, as (count, bytes): in the upload heap
/// (staging, list scratch), shared with the processor (host copies of title
/// memory) and the graphics card's own (indirect arguments, twins' copies).
/// For the memory line: against what is alive, they show what was let go.
pub(crate) static MADE: [std::sync::atomic::AtomicU64; 6] = [const { std::sync::atomic::AtomicU64::new(0) }; 6];

/// What of that is alive now, for the memory line (`MADE` counts what was
/// ever made: 42,607 upload buffers, 77.9 GB, in five minutes of run 60,
/// and nothing said how many were alive): host lists; the lists' scratch
/// chunks and their bytes; of those, the chunks no list holds (kept for the
/// next list, [`Spare`]) and their bytes; chunks taken from those kept
/// instead of made, since the start; and the lists' other upload buffers
/// (texture data copied from buffers, scratch too big for a chunk) and
/// their bytes.
pub(crate) static ALIVE: [std::sync::atomic::AtomicU64; 8] = [const { std::sync::atomic::AtomicU64::new(0) }; 8];

/// About host lists, for the memory line: lists made by the host since the
/// start; preludes made (a list's second allocator and list, made when it
/// first needs one, [`lazy_prelude`]); and, in diagnostics builds, the
/// sum of process-wide private-byte deltas around list creation. Samples
/// include other threads' work and can overlap across concurrent calls.
/// Then the lists alive that were never reset since they were made: the
/// host sides of title lists the title made and has not used (a host list
/// cost 0.68 MB of private memory in run 69: 431 MB for the 632 the title
/// makes at its start).
pub(crate) static LISTS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// The process's private bytes (diagnostics: around the making of lists).
#[cfg(feature = "diagnostics")]
fn private_bytes() -> u64 {
    use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut c = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    c.PagefileUsage as u64
}

/// What the runtime does when the host GPU fails for good: report it like a
/// crash and end the run. The console has no such state, and carrying on
/// draws nothing at seconds per frame.
static ON_FAILURE: OnceLock<fn(&str) -> !> = OnceLock::new();

/// Sets what a host GPU failure does (the runtime's crash report).
pub fn on_failure(report: fn(&str) -> !) {
    let _ = ON_FAILURE.set(report);
}

/// Ends the run on a host GPU failure: what failed, the removal reason and
/// the debug layer's first error, if any (`VICTORY_D3D12_DEBUG=1` in a
/// diagnostics build names the command).
pub(crate) fn fail(what: std::fmt::Arguments) -> ! {
    let mut text = what.to_string();
    if let Some(host) = host() {
        // SAFETY: a plain query.
        if let Err(e) = unsafe { host.device.GetDeviceRemovedReason() } {
            text.push_str(&format!("; device removed: {e}"));
            log_removal_context();
        }
        match host.log_messages() {
            Some(first) => text.push_str(&format!("; debug layer: {first}")),
            None if host.messages.is_none() => text.push_str(
                "; a diagnostics build with VICTORY_D3D12_DEBUG=1 names the command",
            ),
            None => {}
        }
    }
    tracing::error!("{text}");
    // Tests have no log: the reason goes to their output.
    #[cfg(test)]
    eprintln!("host GPU failure: {text}");
    match ON_FAILURE.get() {
        Some(report) => report(&text),
        None => std::process::abort(),
    }
}

static HOST: OnceLock<Option<Host>> = OnceLock::new();

/// The host GPU if it is made already.
pub(crate) fn made_host() -> Option<&'static Host> {
    HOST.get()?.as_ref()
}

/// The host GPU, created on first use; `None` without one (a run then
/// behaves as before: nothing executes).
pub(crate) fn host() -> Option<&'static Host> {
    HOST.get_or_init(|| match create() {
        Ok(host) => Some(host),
        Err(e) => {
            tracing::warn!(error = %e, "no host D3D12 device; command lists will not run");
            None
        }
    })
    .as_ref()
}

fn create() -> windows::core::Result<Host> {
    // SAFETY: plain D3D12 object creation with valid descriptions.
    unsafe {
        // The debug layer only in builds with diagnostics.
        let debug = cfg!(feature = "diagnostics") && crate::switch::on("VICTORY_D3D12_DEBUG");
        if debug {
            let mut layer: Option<ID3D12Debug> = None;
            match D3D12GetDebugInterface(&mut layer) {
                Ok(()) => layer.iter().for_each(|l| l.EnableDebugLayer()),
                Err(e) => tracing::warn!(error = %e, "no D3D12 debug layer"),
            }
            // GPU-based validation (`VICTORY_D3D12_GBV`): out-of-range
            // descriptor heap indices and the like become messages naming
            // the command instead of page faults. Without state tracking
            // (directly indexed textures are not transitioned).
            if crate::switch::on("VICTORY_D3D12_GBV")
                && let Some(l) = &layer
            {
                if let Ok(d1) = l.cast::<ID3D12Debug1>() {
                    d1.SetEnableGPUBasedValidation(true);
                }
                if let Ok(d2) = l.cast::<ID3D12Debug2>() {
                    d2.SetGPUBasedValidationFlags(
                        D3D12_GPU_BASED_VALIDATION_FLAGS_DISABLE_STATE_TRACKING,
                    );
                }
                tracing::info!("GPU-based validation on");
            }
        }
        // Diagnostics: device-removed data (the command that faulted, the
        // faulting address), read when the device is removed.
        if dred_any() {
            let mut dred: Option<ID3D12DeviceRemovedExtendedDataSettings> = None;
            if D3D12GetDebugInterface(&mut dred).is_ok()
                && let Some(dred) = dred
            {
                if dred_on() || dred_light() {
                    dred.SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
                }
                dred.SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
                tracing::info!(
                    breadcrumbs = dred_on() || dred_light(),
                    markers = dred_on(),
                    "crash tracing on (device-removed data)"
                );
            } else {
                tracing::warn!("crash tracing asked for and not there: no device-removed data settings");
            }
        }
        let mut device: Option<ID3D12Device> = None;
        // Diagnostics: the software adapter, on which a shader's wild access
        // is an access violation in this process (a crash report with the
        // address and instruction) instead of a GPU fault and driver reset.
        if cfg!(feature = "diagnostics") && crate::switch::on("VICTORY_D3D12_WARP") {
            let adapter = warp_adapter()?;
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device)?;
            tracing::info!("host device on the software adapter (WARP)");
        } else {
            device = Some(victory_shader::emit::host::create_preferred_device()?);
            // Whether a GPU hang ends the run or freezes the PC (#1054).
            crate::tdr::log();
        }
        let device = device.ok_or_else(windows::core::Error::empty)?;
        // Before anything is made with it (`VICTORY_RECORD`).
        #[cfg(feature = "diagnostics")]
        crate::replay::record::from_env(&device);
        // `VICTORY_D3D12_BACKGROUND=0`: the runtime's background work (the
        // driver compiling shaders again on the "D3D Background Thread"s,
        // for minutes after pipelines were made or loaded) is switched off.
        // An experiment: while those threads ran, the same shot took
        // 1,000-1,200 ms a frame, and 830-950 ms from the moment they
        // stopped (run 54, 08:32:01 to 08:32:06). Left as it is by default:
        // whether the card is slower with the shaders it then keeps is not
        // known.
        if std::env::var_os("VICTORY_D3D12_BACKGROUND").is_some_and(|v| v == "0")
            && let Ok(device6) = device.cast::<ID3D12Device6>()
        {
            let set = device6.SetBackgroundProcessingMode(
                D3D12_BACKGROUND_PROCESSING_MODE_DISABLE_BACKGROUND_WORK,
                D3D12_MEASUREMENTS_ACTION_KEEP_ALL,
                None,
                None,
            );
            tracing::info!(ok = set.is_ok(), "the host's background shader work is switched off (VICTORY_D3D12_BACKGROUND=0)");
        }
        // the widest wave the GPU runs: 32 (NVIDIA) gets console wave64 compute programs whose
        // cross-lane code stays in 32-lane halves translated for waves of 32
        let mut o1 = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
        if device
            .CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, (&raw mut o1).cast(), size_of_val(&o1) as u32)
            .is_ok()
        {
            tracing::info!(min = o1.WaveLaneCountMin, max = o1.WaveLaneCountMax, "host wave lanes");
            victory_shader::emit::set_host_max_wave(o1.WaveLaneCountMax);
            // (RDNA has waves of 32 and 64, the vendor's older cards of 64 alone.)
            let amd = crate::indirect::amd(&device);
            victory_shader::emit::set_host_rdna(amd && o1.WaveLaneCountMin == 32 && o1.WaveLaneCountMax == 64);
        }
        let messages = debug
            .then(|| device.cast::<ID3D12InfoQueue>().ok())
            .flatten();
        let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
            ..Default::default()
        })?;
        let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
        let event = CreateEventW(None, false, false, None)?;
        let rtvs: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                NumDescriptors: RTV_RING,
                ..Default::default()
            })?;
        let rtv_size = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        let dsvs: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
                NumDescriptors: RTV_RING,
                ..Default::default()
            })?;
        let dsv_size = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
        let adapter = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
            .and_then(|factory| factory.EnumAdapterByLuid::<IDXGIAdapter3>(device.GetAdapterLuid()))
            .ok();
        if let Some(desc) = adapter.as_ref().and_then(|a| a.GetDesc1().ok()) {
            let name = String::from_utf16_lossy(&desc.Description);
            tracing::info!(
                adapter = name.trim_end_matches('\0'),
                vendor = format_args!("{:#06x}", desc.VendorId),
                device = format_args!("{:#06x}", desc.DeviceId),
                vram_mb = desc.DedicatedVideoMemory >> 20,
                "host D3D12 adapter"
            );
        }
        tracing::info!("host D3D12 device ready");
        Ok(Host {
            device,
            queue,
            fence,
            submitted: Mutex::new(0),
            event,
            rtvs,
            rtv_size,
            dsvs,
            dsv_size,
            next_dsv: AtomicU32::new(0),
            messages,
            next_rtv: AtomicU32::new(0),
            presenter: Mutex::new(None),
            title_stock: Default::default(),
            adapter,
            fallback_view: OnceLock::new(),
            fallback_sampler: OnceLock::new(),
            texture_readbacks: ReadbackPool::new(),
        })
    }
}

impl Host {
    /// The driver's version, as the system counts it (0: not known).
    pub(crate) fn driver(&self) -> i64 {
        let adapter = self.adapter.as_ref();
        // SAFETY: a plain query.
        adapter.map_or(0, |a| unsafe { a.CheckInterfaceSupport(&IDXGIDevice::IID) }.unwrap_or(0))
    }
}

/// DXGI may return hardware for `EnumWarpAdapter` (notably under Proton).
/// Reject it before creating a device in diagnostics' software-only mode.
pub(crate) fn warp_adapter() -> windows::core::Result<IDXGIAdapter> {
    // SAFETY: plain DXGI adapter queries.
    unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
        let adapter: IDXGIAdapter = factory.EnumWarpAdapter()?;
        let desc = adapter.cast::<IDXGIAdapter1>()?.GetDesc1()?;
        if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0 {
            tracing::warn!(
                vendor = format_args!("{:#x}", desc.VendorId),
                flags = format_args!("{:#x}", desc.Flags),
                "DXGI did not identify the WARP adapter as software; refusing device"
            );
            return Err(windows::core::Error::from_hresult(
                windows::Win32::Foundation::E_FAIL,
            ));
        }
        Ok(adapter)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_host_list_can_reset_an_abandoned_recording() {
        let Some(host) = super::host() else { return };
        let list = host.new_list().unwrap();
        list.defer_host(|| panic!("the discarded step reached a submission"));
        list.reset();
        list.close();
        host.execute(&[&list]);
    }

    /// A list of ours is found by its name, with the debug layer's note or not.
    #[cfg(feature = "diagnostics")]
    #[test]
    fn a_list_is_found_by_the_number_in_its_name() {
        assert_eq!(super::list_number("victory host list 7"), Some(7));
        assert_eq!(super::list_number("victory host list 158 (debug layer indirect)"), Some(158));
        assert_eq!(super::list_number("victory host list 7x"), None);
        assert_eq!(super::list_number("victory host list 7 x"), None);
        assert_eq!(super::list_number("another list 7"), None);
        assert_eq!(super::list_number(""), None);
    }

    /// What is there of a range some of whose pages are not: those bytes,
    /// counted, and zeros for the rest, whatever the buffer held.
    #[test]
    fn what_is_there_is_read_and_the_rest_is_zeros() {
        use windows_sys::Win32::System::Memory::*;
        const SIZE: usize = 0x8_0000;
        // SAFETY: a reservation of ours; only committed pages are written.
        unsafe {
            let at = VirtualAlloc(std::ptr::null(), SIZE, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
            assert!(!at.is_null());
            // 0-0x1_1000 and 0x3_0000-0x4_0000 are there.
            for (from, len) in [(0, 0x1_1000), (0x3_0000, 0x1_0000)] {
                assert!(!VirtualAlloc(at.add(from).cast(), len, MEM_COMMIT, PAGE_READWRITE).is_null());
                at.add(from).write_bytes(0x5a, len);
            }
            // From an address inside a page, as textures are packed.
            let mut out = vec![0x77u8; SIZE - 0x300];
            assert_eq!(super::read_there(at.add(0x300), &mut out), 0x1_1000 - 0x300 + 0x1_0000);
            for (i, &b) in out.iter().enumerate() {
                let from = i + 0x300;
                let there = from < 0x1_1000 || (0x3_0000..0x4_0000).contains(&from);
                assert_eq!(b, if there { 0x5a } else { 0 }, "byte {from:#x}");
            }
            // All there: one read.
            let mut out = vec![0x77u8; 0x1_0000];
            assert_eq!(super::read_there(at.add(0x1000), &mut out), 0x1_0000);
            assert!(out.iter().all(|&b| b == 0x5a));
            // Nothing there.
            let mut out = vec![0x77u8; 0x1_0000];
            assert_eq!(super::read_there(at.add(0x5_0000), &mut out), 0);
            assert!(out.iter().all(|&b| b == 0));
            assert_ne!(VirtualFree(at.cast(), 0, MEM_RELEASE), 0);
        }
    }

    /// Zeros for a title destination that went meanwhile are no fault.
    #[test]
    fn zeros_where_nothing_is_are_lost() {
        use windows_sys::Win32::System::Memory::*;
        // SAFETY: a reservation of ours, its first page committed.
        unsafe {
            let at = VirtualAlloc(std::ptr::null(), 1 << 16, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
            assert!(!VirtualAlloc(at.cast(), 4096, MEM_COMMIT, PAGE_READWRITE).is_null());
            at.write_bytes(7, 4096);
            super::zero(at, 4096);
            assert!(std::slice::from_raw_parts(at, 4096).iter().all(|&b| b == 0));
            super::zero(at.add(4096), 4096);
            VirtualFree(at.cast(), 0, MEM_RELEASE);
        }
    }

    #[test]
    fn sparse_copies_preserve_islands_and_observe_remaps() {
        use windows_sys::Win32::System::Memory::*;
        const N: usize = 5 << 16;
        // SAFETY: we reserve a private region, commit selected pages inside it,
        // and only directly access the pages while they are writable.
        unsafe {
            let base = VirtualAlloc(std::ptr::null(), N, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
            assert!(!base.is_null());
            let mut expected = vec![0; N];
            for (offset, value) in [(0, 9), (2 << 16, 37), (N - 4096, 85)] {
                let at = base.add(offset);
                assert_eq!(VirtualAlloc(at.cast(), 4096, MEM_COMMIT, PAGE_READWRITE), at.cast());
                at.write_bytes(value, 4096);
                expected[offset..offset + 4096].fill(value);
            }
            for remap in [false, true] {
                if remap {
                    let mut old = 0;
                    assert_ne!(VirtualProtect(base.cast(), 4096, PAGE_NOACCESS, &mut old), 0);
                    expected[..4096].fill(0);
                    let at = base.add(1 << 16);
                    assert_eq!(VirtualAlloc(at.cast(), 4096, MEM_COMMIT, PAGE_READWRITE), at.cast());
                    at.write_bytes(0xc7, 4096);
                    expected[1 << 16..(1 << 16) + 4096].fill(0xc7);
                }
                let mut out = vec![0xee; N];
                super::copy_mapped(out.as_mut_ptr().add(27), base.add(27), N - 54, false);
                assert_eq!(out[27..N - 27], expected[27..N - 27]);
                assert!(out[..27].iter().chain(&out[N - 27..]).all(|&v| v == 0xee));
            }
            assert_ne!(VirtualFree(base.cast(), 0, MEM_RELEASE), 0);
        }
    }

    use super::*;

    #[test]
    fn warp_adapter_accepts_only_software() {
        // This probes DXGI without creating a D3D12 device or running a title.
        // SAFETY: plain DXGI adapter queries.
        let raw = unsafe {
            let Ok(factory) = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
            else {
                return;
            };
            factory.EnumWarpAdapter::<IDXGIAdapter>()
        };
        let Ok(raw) = raw else { return };
        // SAFETY: EnumWarpAdapter returns a DXGI adapter, which implements
        // IDXGIAdapter1; `raw` owns that interface.
        let desc = unsafe { raw.cast::<IDXGIAdapter1>().unwrap().GetDesc1().unwrap() };
        eprintln!(
            "EnumWarpAdapter: vendor={:#x}, flags={:#x}",
            desc.VendorId, desc.Flags
        );
        assert_eq!(
            warp_adapter().is_ok(),
            desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0,
        );
    }
}

/// The batches of clamped records in one scratch buffer
/// ([`HostList::batch_place`]).
const BATCHES: u32 = 8;

/// How a host list that was open is closed ([`Recording::close`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Closing {
    /// With what was recorded into it: the title's Close, a cut at a wait,
    /// a list of ours.
    Recorded,
    /// Taken for a submission while open (`HostList::close_if_open`).
    Taken,
    /// Still open where the lists go to the queue ([`closed_for_execution`]).
    AtExecution,
}

/// What became of the host lists that were left open
/// ([`Recording::left_open`]), and of lists that came to a submission open.
#[derive(Default)]
pub(crate) struct OpenLists {
    /// Left open: handed to a title's list in place of the recording a
    /// submission took (`crate::list::detach`).
    pub(crate) left: std::sync::atomic::AtomicU64,
    /// Of those: closed by the title's next Reset of its list.
    pub(crate) by_reset: std::sync::atomic::AtomicU64,
    /// Of those: closed when the title's list was submitted again with no
    /// Reset in between.
    pub(crate) taken_again: std::sync::atomic::AtomicU64,
    /// Of those: closed with something recorded and no Reset before it.
    pub(crate) otherwise: std::sync::atomic::AtomicU64,
    /// Recordings begun by a Reset that the title submitted without a
    /// Close: closed when taken.
    pub(crate) unclosed: std::sync::atomic::AtomicU64,
    /// Open where the lists go to the queue, and closed there. No path is
    /// known to lead here.
    pub(crate) at_execution: std::sync::atomic::AtomicU64,
}

impl OpenLists {
    const fn new() -> Self {
        OpenLists {
            left: std::sync::atomic::AtomicU64::new(0),
            by_reset: std::sync::atomic::AtomicU64::new(0),
            taken_again: std::sync::atomic::AtomicU64::new(0),
            otherwise: std::sync::atomic::AtomicU64::new(0),
            unclosed: std::sync::atomic::AtomicU64::new(0),
            at_execution: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// The six counts, in the order of the fields.
    #[cfg(any(test, feature = "diagnostics"))]
    pub(crate) fn counts(&self) -> [u64; 6] {
        [&self.left, &self.by_reset, &self.taken_again, &self.otherwise, &self.unclosed, &self.at_execution]
            .map(|count| count.load(Ordering::Relaxed))
    }
}

/// Of every host list of the process.
pub(crate) static OPEN_LISTS: OpenLists = OpenLists::new();

/// For the lines every five seconds: what became of the host lists left
/// open, since the start.
#[cfg(feature = "diagnostics")]
fn open_lists_line() {
    let [left, by_reset, taken_again, otherwise, unclosed, at_execution] = OPEN_LISTS.counts();
    tracing::info!(
        "host lists left open when their recording was taken, since the start: {left}; closed by the title's next Reset \
         {by_reset}, when submitted again without a Reset {taken_again}, with something recorded and no Reset {otherwise}; \
         recordings submitted without a Close and closed when taken {unclosed}; found open where lists go to the queue \
         {at_execution} (none is expected)"
    );
}

/// Whether a host list is open for recording, apart from the D3D12 calls:
/// [`HostList`] asks it what to do with its native list and tells it what
/// it did. A list is open when made and after a reset, and closed by a
/// close; only a closed list may go to a queue.
pub(crate) struct Recording(std::sync::atomic::AtomicU8);

impl Recording {
    const CLOSED: u8 = 0;
    const OPEN: u8 = 1;
    /// Open, and handed to a title's list in place of the recording a
    /// submission took: the title has not reset it yet.
    const LEFT_OPEN: u8 = 2;

    /// Of a list just made or reset.
    pub(crate) const fn open() -> Self {
        Recording(std::sync::atomic::AtomicU8::new(Self::OPEN))
    }

    pub(crate) fn is_closed(&self) -> bool {
        self.0.load(Ordering::Relaxed) == Self::CLOSED
    }

    /// The open list is handed to a title's list whose recording was taken,
    /// and is not closed for it.
    pub(crate) fn left_open(&self, lists: &OpenLists) {
        if self.0.compare_exchange(Self::OPEN, Self::LEFT_OPEN, Ordering::Relaxed, Ordering::Relaxed).is_ok() {
            lists.left.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// The list is to be closed: whether it was open, so that its native
    /// list has to be closed. It counts as closed from here on.
    pub(crate) fn close(&self, how: Closing, lists: &OpenLists) -> bool {
        let count = match (self.0.swap(Self::CLOSED, Ordering::Relaxed), how) {
            (Self::CLOSED, _) => return false,
            (_, Closing::AtExecution) => &lists.at_execution,
            (Self::LEFT_OPEN, Closing::Taken) => &lists.taken_again,
            (Self::LEFT_OPEN, Closing::Recorded) => &lists.otherwise,
            (_, Closing::Taken) => &lists.unclosed,
            (_, Closing::Recorded) => return true,
        };
        count.fetch_add(1, Ordering::Relaxed);
        true
    }

    /// The list is reset: whether it was open, so that its native list has
    /// to be closed first, as a recording that is discarded. It is open
    /// from here on.
    pub(crate) fn reset(&self, lists: &OpenLists) -> bool {
        match self.0.swap(Self::OPEN, Ordering::Relaxed) {
            Self::CLOSED => false,
            Self::LEFT_OPEN => {
                lists.by_reset.fetch_add(1, Ordering::Relaxed);
                true
            }
            _ => true,
        }
    }
}

#[cfg(test)]
mod recording_tests {
    use super::{Closing, OPEN_LISTS, OpenLists, Recording, host};

    /// A title's list through a submission, its next Reset and a second
    /// submission: the host list it was left open is closed by the Reset
    /// (its native list has to be, before its allocator is reset), records,
    /// is closed by the title, and has nothing left to close where the
    /// lists go to the queue.
    #[test]
    fn a_list_left_open_is_closed_by_the_next_reset_and_closed_when_submitted() {
        let lists = OpenLists::default();
        // The first recording: closed by the title, taken by the submission.
        let first = Recording::open();
        assert!(first.close(Closing::Recorded, &lists));
        assert!(first.is_closed());
        // In its place a new host list, left open.
        let second = Recording::open();
        second.left_open(&lists);
        assert!(!second.is_closed());
        // The title's Reset.
        assert!(second.reset(&lists), "the open native list must be closed before the reset");
        assert!(!second.is_closed());
        // Recorded into, closed by the title, taken by the second submission.
        assert!(second.close(Closing::Recorded, &lists));
        assert!(second.is_closed());
        for taken in [&first, &second] {
            assert!(!taken.close(Closing::AtExecution, &lists), "open where the lists go to the queue");
        }
        assert_eq!(lists.counts(), [1, 1, 0, 0, 0, 0]);
    }

    /// A title's list submitted twice with no Reset in between: the second
    /// submission takes the host list that was left open, which is closed
    /// when it is taken.
    #[test]
    fn a_list_submitted_twice_without_a_reset_is_closed_when_taken() {
        let lists = OpenLists::default();
        let first = Recording::open();
        assert!(first.close(Closing::Recorded, &lists));
        let second = Recording::open();
        second.left_open(&lists);
        // (`HostList::close_if_open`: open, so it is closed as taken.)
        assert!(!second.is_closed());
        assert!(second.close(Closing::Taken, &lists));
        assert!(second.is_closed());
        // The title's list has a third, left open; that one is not submitted.
        let third = Recording::open();
        third.left_open(&lists);
        for taken in [&first, &second] {
            assert!(!taken.close(Closing::AtExecution, &lists), "open where the lists go to the queue");
        }
        assert_eq!(lists.counts(), [2, 0, 1, 0, 0, 0]);
    }

    /// An open list where the lists go to the queue is closed there and
    /// counted, once.
    #[test]
    fn a_list_open_at_the_queue_is_closed_there_and_counted() {
        let lists = OpenLists::default();
        for left in [false, true] {
            let list = Recording::open();
            if left {
                list.left_open(&lists);
            }
            assert!(list.close(Closing::AtExecution, &lists));
            assert!(list.is_closed());
            assert!(!list.close(Closing::AtExecution, &lists));
        }
        assert_eq!(lists.counts(), [1, 0, 0, 0, 0, 2]);
    }

    /// Every list left open is counted once, by what closed it; a recording
    /// the title discards or closes itself is none of them.
    #[test]
    fn what_closes_an_open_list_is_counted_once() {
        let lists = OpenLists::default();
        // Left open, then closed with something recorded and no Reset.
        let list = Recording::open();
        list.left_open(&lists);
        list.left_open(&lists);
        assert!(list.close(Closing::Recorded, &lists));
        assert!(!list.close(Closing::Recorded, &lists));
        assert_eq!(lists.counts(), [1, 0, 0, 1, 0, 0]);
        // A closed list is reset: nothing to close first. Its recording is
        // discarded by another Reset: closed first, not counted.
        assert!(!list.reset(&lists));
        assert!(list.reset(&lists));
        // Submitted without a Close: closed when taken.
        assert!(list.close(Closing::Taken, &lists));
        assert!(!list.close(Closing::Taken, &lists));
        assert_eq!(lists.counts(), [1, 0, 0, 1, 1, 0]);
        // A closed list is not left open.
        list.left_open(&lists);
        assert!(list.is_closed());
        assert_eq!(lists.counts(), [1, 0, 0, 1, 1, 0]);
    }

    /// The check where lists go to the queue, with a real list: one given
    /// to a submission open is closed there, counted, and runs.
    #[test]
    fn an_open_host_list_given_to_a_submission_is_closed_and_runs() {
        let Some(host) = host() else {
            eprintln!("skipping: no D3D12 device");
            return;
        };
        let list = host.new_list().unwrap();
        assert!(!list.is_closed());
        let before = OPEN_LISTS.counts()[5];
        host.execute(&[&list]);
        assert!(list.is_closed());
        assert!(OPEN_LISTS.counts()[5] > before);
        // SAFETY: a plain query of the device.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }
}

/// Where host lists go to a queue: only closed lists may. Every path closes
/// them before (the title's Close, `HostList::close_if_open` when a list is
/// taken open); one that is open still is closed here, with what it holds,
/// counted and logged (the first four, then every one that doubles the
/// count). How many were open.
pub(crate) fn closed_for_execution(runs: &[&HostList]) -> usize {
    let mut found = 0;
    for list in runs {
        if !list.recording.is_closed() {
            list.close_as(Closing::AtExecution);
            found += 1;
        }
    }
    if found != 0 {
        let so_far = OPEN_LISTS.at_execution.load(Ordering::Relaxed);
        if (so_far - found as u64..so_far).any(|n| n < 4 || (n + 1).is_power_of_two()) {
            tracing::warn!(
                found,
                so_far,
                lists = runs.len(),
                "host lists were still open where they go to the queue: closed there"
            );
        }
    }
    found
}

/// A host command list and its allocator, always used together.
pub(crate) struct HostList {
    allocator: ID3D12CommandAllocator,
    /// With barriers held and joined (`rec`).
    pub(crate) list: rec::Rec,
    /// Whether the underlying D3D12 list is open for recording.
    recording: Recording,
    /// Work for submission time, in recorded order.
    deferred: Mutex<Deferred>,
    /// Upload memory for data the list's draws make (expanded vertices),
    /// reused after a reset.
    scratch: Mutex<Scratch>,
    /// Host texture states the list assumes, by host copy: at its start and
    /// at its end. Lists are recorded on many threads and run in submission
    /// order, so the transitions from a texture's real state are made at
    /// submission, in `prelude`. Holding the copies keeps every texture the
    /// list names alive until it is reset, even if the title releases the
    /// resource first.
    states: Mutex<States>,
    /// What the list does with depth textures that have a colour twin, by
    /// twin ([`TwinUse`]).
    twins: Mutex<std::collections::HashMap<usize, TwinUse>>,
    /// The list that runs those transitions just before this one, made
    /// when the list first has such transitions ([`lazy_prelude`]).
    prelude: prelude_pool::Slot,
    /// The pipelines the list binds, for reports of slow submissions.
    pipelines: std::sync::Arc<Mutex<std::collections::BTreeSet<String>>>,
    /// What its indirect executions ran, for the report of a removed device.
    #[cfg(feature = "diagnostics")]
    indirects: indirects::Shared,
    /// The work its indirect executions may still do (`indirect`): a
    /// shader-writable word, zeroed when the list is submitted, and after
    /// it what the clamp notes about mesh records. In memory the CPU maps.
    budget: OnceLock<Option<(ID3D12Resource, usize)>>,
    /// The same words in the card's own memory ([`card_budget`]), which the
    /// clamp uses instead when there is one: zeroed in the list before its
    /// first clamp, its notes copied to `budget` when the list closes.
    card_budget: OnceLock<Option<ID3D12Resource>>,
    /// Device-local scratch buffer for clamped indirect arguments (`indirect`),
    /// grown to the largest indirect execution recorded on the list.
    indirect_args: Mutex<Option<(ID3D12Resource, u64)>>,
    /// Scratch buffers of the batched clamps (`crate::indirect::batch`), and
    /// how many batches of this recording have a place in them.
    batch_args: Mutex<(Vec<ID3D12Resource>, u32)>,
    /// Refreshes of host copies that a thread of ours is still finding, and
    /// where among the list's steps they belong ([`aside`]).
    coming: Mutex<Option<(usize, std::sync::Arc<aside::Coming>)>>,
    /// Another number each time something changes that what a draw reads
    /// depends on ([`HostList::changed_something`]); no two lists have the
    /// same.
    epoch: std::sync::atomic::AtomicU64,
    /// Counts the list in [`ALIVE`] while it lives.
    _alive: Counted,
    /// The texts of the markers `note_op` set, in order (for device-removed
    /// data without marker texts), with the host GPU range a copy reads.
    #[cfg(feature = "diagnostics")]
    notes: std::sync::Arc<Mutex<Vec<Note>>>,
    /// With `VICTORY_GPU_TIMING`: a GPU timestamp before each noted command
    /// and at the end, and where they are read back.
    #[cfg(feature = "diagnostics")]
    stamps: OnceLock<Option<(ID3D12QueryHeap, ID3D12Resource)>>,
}

/// A list's prelude: an allocator and a closed list.
type Prelude = (ID3D12CommandAllocator, ID3D12GraphicsCommandList);

/// Whether a host list's prelude is made when the list first needs one. A
/// prelude runs the texture transitions and twin copies a list needs before
/// it; lists that name no texture never have one, and every host list was
/// made with one all the same: twice the host objects per list, for the
/// title's 630 lists and for every list made anew.
/// `VICTORY_LAZY_PRELUDE=0`: made with the list, as before.
fn lazy_prelude() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LAZY_PRELUDE").is_none_or(|v| v != "0"))
}

/// Timestamps per list (diagnostics); commands past them are not timed.
#[cfg(feature = "diagnostics")]
const STAMPS: u32 = 4096;

/// A marker's text and the host range its command reads (diagnostics).
#[cfg(feature = "diagnostics")]
type Note = (String, Option<(u64, u64)>);

type States = std::collections::HashMap<
    usize,
    (u32, u32, std::sync::Arc<crate::resource::HostTexture>),
>;

/// What a list does with a depth texture that has a colour twin
/// ([`crate::resource::Twin`]). Lists are recorded on many threads and run
/// in submission order, so a list cannot know which side an earlier list
/// leaves behind: it notes which sides it uses as it finds them, and its
/// prelude copies one side into the other at submission if that side is
/// behind then. From its first use of a side on, the list knows.
struct TwinUse {
    /// The depth texture (the title resource's host copy) and its twin.
    depth: std::sync::Arc<crate::resource::HostTexture>,
    twin: std::sync::Arc<crate::resource::Twin>,
    /// The sides (depth texture, twin) the list uses as it finds them.
    needs: [bool; 2],
    /// Whether each side lacks what the other was written last, where the
    /// list is now; `None`: as the list found it.
    behind: [Option<bool>; 2],
    /// The draw or dispatch being prepared writes the twin.
    writing: bool,
    /// The depth plane's twin holds what the plane lacks although the list
    /// copied the twins since: the plane's pass waits for the list's close.
    late: bool,
}

/// The pass that draws the twin of a depth plane into the plane: one
/// triangle over each mip of each slice, its pixel shader writing the
/// twin's texel as the depth, with the depth test "always" and no stencil
/// (the stencil plane is copied through a buffer and not touched here).
///
/// A 16-bit twin holds whole steps of 1/65535, and the depth written is
/// the step times the float nearest 1/65535 (0x37800080). That is how
/// The title's tiled lighting computes the depth of the
/// rectangles it then draws with the test "equal", one class of materials
/// a draw (all six of its vertex shaders multiply the class number by that
/// constant), so the pass and the draw give the host the same float, and
/// the host turns the same float into the same 24 bits. The step divided
/// by 65535 is another float for 512 of the 65,536 steps (none under 257),
/// and lands in another 24-bit step for 127 of them (none under 385).
const PLANE_CORNERS: &str = r#"
float4 corner(uint i : SV_VertexID) : SV_Position {
    return float4(float(i & 1) * 4.0 - 1.0, float(i >> 1) * 4.0 - 1.0, 0.0, 1.0);
}
"#;
const PLANE_STEPS: &str = r#"
#define RS "DescriptorTable(SRV(t0), visibility = SHADER_VISIBILITY_PIXEL)"
Texture2DArray<uint> twin : register(t0);
[RootSignature(RS)]
float plane(float4 p : SV_Position) : SV_Depth {
    return float(twin.Load(int4(int2(p.xy), 0, 0))) * asfloat(0x37800080);
}
"#;
const PLANE_FLOATS: &str = r#"
#define RS "DescriptorTable(SRV(t0), visibility = SHADER_VISIBILITY_PIXEL)"
Texture2DArray<float> twin : register(t0);
[RootSignature(RS)]
float plane(float4 p : SV_Position) : SV_Depth {
    return twin.Load(int4(int2(p.xy), 0, 0));
}
"#;

/// Cached pipeline objects. Shader-visible descriptors belong to the
/// individual recorded pass instead: a command list may be submitted late
/// or more than once, so a global descriptor ring cannot safely wrap.
struct PlanePass {
    made: Mutex<Vec<(bool, i32, Option<PlanePipeline>)>>,
}

/// The pass's root signature and pipeline for one kind of twin and one
/// depth format.
type PlanePipeline = (ID3D12RootSignature, ID3D12PipelineState);

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for PlanePass {}
// SAFETY: as above.
unsafe impl Sync for PlanePass {}

fn plane_pass() -> Option<&'static PlanePass> {
    static PASS: OnceLock<PlanePass> = OnceLock::new();
    host()?;
    Some(PASS.get_or_init(|| PlanePass { made: Mutex::default() }))
}

/// Compiles function `entry` of `source` for `target` with the system's
/// compiler, as the indirect-argument clamp is (`crate::indirect`).
pub(crate) fn compile(
    source: &str,
    entry: windows::core::PCSTR,
    target: windows::core::PCSTR,
) -> windows::core::Result<Vec<u8>> {
    use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
    use windows::Win32::Graphics::Direct3D::ID3DBlob;
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            windows::core::s!("victory"),
            None,
            None,
            entry,
            target,
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            // SAFETY: the blob's own bytes.
            let text = unsafe {
                std::slice::from_raw_parts(
                    errors.GetBufferPointer().cast::<u8>(),
                    errors.GetBufferSize(),
                )
            };
            tracing::warn!("a shader of ours: {}", String::from_utf8_lossy(text));
        }
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: the blob's own bytes.
    let bytes = unsafe {
        std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize())
    };
    Ok(bytes.to_vec())
}

impl PlanePass {
    /// The root signature and pipeline for a twin of 16-bit steps (`steps`)
    /// or of floats and the depth format `format`.
    fn pipeline(&self, steps: bool, format: DXGI_FORMAT) -> Option<PlanePipeline> {
        let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((_, _, pipeline)) = made.iter().find(|m| (m.0, m.1) == (steps, format.0)) {
            return pipeline.clone();
        }
        let pipeline = match Self::make(steps, format) {
            Ok(pipeline) => Some(pipeline),
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    steps,
                    format = format.0,
                    "no pipeline to draw a depth plane from its twin: such planes stay as they are"
                );
                None
            }
        };
        made.push((steps, format.0, pipeline.clone()));
        pipeline
    }

    fn make(steps: bool, format: DXGI_FORMAT) -> windows::core::Result<PlanePipeline> {
        use windows::core::s;
        let device = &host().ok_or_else(windows::core::Error::empty)?.device;
        let vs = compile(PLANE_CORNERS, s!("corner"), s!("vs_5_1"))?;
        let source = if steps { PLANE_STEPS } else { PLANE_FLOATS };
        let ps = compile(source, s!("plane"), s!("ps_5_1"))?;
        let code = |bytes: &[u8]| D3D12_SHADER_BYTECODE {
            pShaderBytecode: bytes.as_ptr().cast(),
            BytecodeLength: bytes.len(),
        };
        // SAFETY: the root signature is inside the pixel shader; the
        // description points at live data and never releases its pointer.
        unsafe {
            let root: ID3D12RootSignature = device.CreateRootSignature(0, &ps[..])?;
            let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                VS: code(&vs[..]),
                PS: code(&ps[..]),
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC {
                    FillMode: D3D12_FILL_MODE_SOLID,
                    CullMode: D3D12_CULL_MODE_NONE,
                    DepthClipEnable: true.into(),
                    ..Default::default()
                },
                // Every texel is written; stencil stays off.
                DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
                    DepthEnable: true.into(),
                    DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL,
                    DepthFunc: D3D12_COMPARISON_FUNC_ALWAYS,
                    ..Default::default()
                },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                DSVFormat: format,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                ..Default::default()
            };
            let pipeline: ID3D12PipelineState = device.CreateGraphicsPipelineState(&desc)?;
            Ok((root, pipeline))
        }
    }
}

/// Builds the immutable table once for the lifetime of a depth-plane twin.
fn plane_pass_views(
    host: &Host,
    plane: &ID3D12Resource,
    views: &Mutex<Option<ID3D12DescriptorHeap>>,
    shape: D3D12_RESOURCE_DESC,
    format: DXGI_FORMAT,
) -> Option<ID3D12DescriptorHeap> {
    let mut cached = views.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(heap) = &*cached {
        return Some(heap.clone());
    }
    let (slices, mips) = (u32::from(shape.DepthOrArraySize), u32::from(shape.MipLevels));
    let count = slices * mips;
    if count == 0 {
        return None;
    }
    let kind = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    // SAFETY: plain heap creation.
    let heap: ID3D12DescriptorHeap = match unsafe {
        host.device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
            Type: kind,
            NumDescriptors: count,
            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
            NodeMask: 0,
        })
    } {
        Ok(heap) => heap,
        Err(e) => {
            tracing::error!(error = %e, count, "no descriptor heap for depth plane pass");
            return None;
        }
    };
    // SAFETY: live texture and heap. Each slice/mip has its own slot, and
    // no command list can see the heap until all descriptors are published.
    unsafe {
        let base = heap.GetCPUDescriptorHandleForHeapStart().ptr;
        let step = host.device.GetDescriptorHandleIncrementSize(kind) as usize;
        for slice in 0..slices {
            for mip in 0..mips {
                let view = D3D12_SHADER_RESOURCE_VIEW_DESC {
                    Format: format,
                    ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
                    Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                    Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                        Texture2DArray: D3D12_TEX2D_ARRAY_SRV {
                            MostDetailedMip: mip,
                            MipLevels: 1,
                            FirstArraySlice: slice,
                            ArraySize: 1,
                            ..Default::default()
                        },
                    },
                };
                let handle = D3D12_CPU_DESCRIPTOR_HANDLE { ptr: base + (slice * mips + mip) as usize * step };
                host.device.CreateShaderResourceView(plane, Some(&view), handle);
            }
        }
    }
    *cached = Some(heap.clone());
    Some(heap)
}

/// Records into `list` the pass that draws `plane`, the twin of the depth
/// plane of `depth`, into that plane: `depth` in the depth-write state,
/// `plane` readable by pixel shaders. The pass sets the list's descriptor
/// heap, graphics root signature, pipeline, depth target, viewport,
/// scissor and topology and puts none of them back: it is for a list that
/// records nothing of the title's after it.
fn record_plane_pass(
    owner: &HostList,
    depth: &ID3D12Resource,
    plane: &ID3D12Resource,
    cache: &Mutex<Option<ID3D12DescriptorHeap>>,
) {
    let list = &owner.list;
    let (Some(host), Some(pass)) = (host(), plane_pass()) else {
        return;
    };
    // SAFETY: live resources.
    let (of, twin) = unsafe { (depth.GetDesc(), plane.GetDesc()) };
    // An R16 typeless twin (53) holds steps, read as R16_UINT (57); an R32
    // typeless one floats, read as R32_FLOAT (41). The depth view names
    // the depth-stencil format of the texture's family (44: 45, 19: 20).
    let steps = twin.Format.0 == 53;
    let format = DXGI_FORMAT(match of.Format.0 {
        44 => 45,
        19 => 20,
        typed => typed,
    });
    let Some((root, pipeline)) = pass.pipeline(steps, format) else {
        return;
    };
    let Some(views) = plane_pass_views(host, plane, cache, of, DXGI_FORMAT(if steps { 57 } else { 41 })) else {
        return;
    };
    // The recording must outlive the title's ownership of the twin.
    owner.keep(&views);
    owner.keep(depth);
    owner.keep(plane);
    // SAFETY: live heap, open list.
    let (base_gpu, step) = unsafe {
        list.SetDescriptorHeaps(&[Some(views.clone())]);
        (
            views.GetGPUDescriptorHandleForHeapStart().ptr,
            host.device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV),
        )
    };
    for slice in 0..u32::from(of.DepthOrArraySize) {
        for mip in 0..u32::from(of.MipLevels) {
            let slot = slice * u32::from(of.MipLevels) + mip;
            let target = D3D12_DEPTH_STENCIL_VIEW_DESC {
                Format: format,
                ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2DARRAY,
                Anonymous: D3D12_DEPTH_STENCIL_VIEW_DESC_0 {
                    Texture2DArray: D3D12_TEX2D_ARRAY_DSV {
                        MipSlice: mip,
                        FirstArraySlice: slice,
                        ArraySize: 1,
                    },
                },
                ..Default::default()
            };
            let target = host.dsv(depth, Some(&target));
            let (width, height) = ((of.Width >> mip).max(1) as u32, (of.Height >> mip).max(1));
            let gpu = D3D12_GPU_DESCRIPTOR_HANDLE {
                ptr: base_gpu + u64::from(slot) * u64::from(step),
            };
            // SAFETY: an open list; a slot of our heap; live resources and
            // objects (the pass's own live as long as the process).
            unsafe {
                list.SetGraphicsRootSignature(&root);
                list.SetPipelineState(&pipeline);
                list.SetGraphicsRootDescriptorTable(0, gpu);
                list.OMSetRenderTargets(0, None, false, Some(&target));
                list.RSSetViewports(&[D3D12_VIEWPORT {
                    TopLeftX: 0.0,
                    TopLeftY: 0.0,
                    Width: width as f32,
                    Height: height as f32,
                    MinDepth: 0.0,
                    MaxDepth: 1.0,
                }]);
                list.RSSetScissorRects(&[RECT {
                    left: 0,
                    top: 0,
                    right: width as i32,
                    bottom: height as i32,
                }]);
                list.IASetPrimitiveTopology(
                    windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY(4),
                );
                list.DrawInstanced(3, 1, 0, 0);
            }
            crate::resource::PLANES[0].fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Records into `list` the copy of a depth texture into its colour twin
/// (`colour`) or of the twin into the depth texture, both already in their
/// copy states: whole textures of one size in one format family, which the
/// host copies as they are; or, for the twin of a stencil plane, that plane
/// alone, through the twin's buffer.
fn record_twin_copy(
    list: &ID3D12GraphicsCommandList,
    depth: &ID3D12Resource,
    twin: &crate::resource::Twin,
    colour: bool,
) {
    let (from, to) = if colour {
        (depth, &twin.colour.texture)
    } else {
        (&twin.colour.texture, depth)
    };
    crate::resource::TWINS[2 + usize::from(colour)].fetch_add(1, Ordering::Relaxed);
    // shortcut: the depth plane's twin is not given the plane (that is a
    // pass the other way: the plane read by a shader, the twin as its
    // target). The twin then lacks what draws and clears wrote to the
    // plane since shaders last wrote the twin. In the title
    // nothing writes that plane but the twin's pass (its 192 draws against
    // it only test, run 29), so the twin lacks nothing. Counted.
    if colour && twin.plane.is_some() {
        crate::resource::PLANES[1].fetch_add(1, Ordering::Relaxed);
    }
    let Some(through) = &twin.through else {
        // SAFETY: an open list and two live textures.
        unsafe { list.CopyResource(to, from) };
        return;
    };
    // A stencil plane and its twin: each mip and slice of the plane goes
    // into the buffer, laid out as the host lays the plane out for copies,
    // and from the buffer into the other texture, whole subresources both
    // times. In the depth texture the stencil plane's subresources follow
    // the depth plane's; the twin has the one plane.
    let Some(host) = host() else { return };
    // SAFETY: a live texture.
    let of = unsafe { depth.GetDesc() };
    let planes = u32::from(of.MipLevels) * u32::from(of.DepthOrArraySize);
    let mut rows = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); planes as usize];
    // SAFETY: a plain query writing `planes` footprints.
    unsafe {
        host.device.GetCopyableFootprints(
            &of,
            planes,
            planes,
            0,
            Some(rows.as_mut_ptr()),
            None,
            None,
            None,
        );
    }
    let (first_from, first_to) = if colour { (planes, 0) } else { (0, planes) };
    let subresource = |texture: &ID3D12Resource, index: u32| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: a borrowed pointer the location never releases.
        pResource: unsafe { std::mem::transmute_copy(texture) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: index,
        },
    };
    let placed = |row: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: as above.
        pResource: unsafe { std::mem::transmute_copy(through) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: *row,
        },
    };
    let (common, source, dest) = (
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );
    // SAFETY: an open list and live resources; the buffer was made as big
    // as these footprints, and rests in the common state between copies.
    unsafe {
        list.ResourceBarrier(&[transition(through, common, dest)]);
        for (i, row) in rows.iter().enumerate() {
            let index = first_from + i as u32;
            list.CopyTextureRegion(&placed(row), 0, 0, 0, &subresource(from, index), None);
        }
        list.ResourceBarrier(&[transition(through, dest, source)]);
        for (i, row) in rows.iter().enumerate() {
            let index = first_to + i as u32;
            list.CopyTextureRegion(&subresource(to, index), 0, 0, 0, &placed(row), None);
        }
        list.ResourceBarrier(&[transition(through, source, common)]);
    }
}

/// Whether device-removed data is on (`VICTORY_D3D12_DRED`, diagnostics
/// builds).
pub(crate) fn dred_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| cfg!(feature = "diagnostics") && crate::switch::on("VICTORY_D3D12_DRED") && !dred_faults() && !dred_light())
}

/// Breadcrumbs and the faulting address, but no markers and no texts
/// (`VICTORY_D3D12_DRED=light`): which list a removed device stopped in, at
/// which of its commands, and the pipelines the list binds; formatting a text
/// for every command is what makes the whole of it slow.
pub(crate) fn dred_light() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        cfg!(feature = "diagnostics") && std::env::var_os("VICTORY_D3D12_DRED").is_some_and(|v| v == "light")
    })
}

/// Any of the three.
pub(crate) fn dred_any() -> bool {
    dred_on() || dred_light() || dred_faults()
}

/// Whether only the faulting address of a removed device is kept
/// (`VICTORY_D3D12_DRED=faults`): no breadcrumbs, no markers and no texts, so
/// it costs next to nothing and can stay on while the title is played.
pub(crate) fn dred_faults() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        cfg!(feature = "diagnostics") && std::env::var_os("VICTORY_D3D12_DRED").is_some_and(|v| v == "faults")
    })
}

/// The lists alive while device-removed data is on, by the number their name
/// ends in, with their marker texts and the pipelines they bind: a report made
/// on another thread than the one that submits has no lists of its own.
#[cfg(feature = "diagnostics")]
#[allow(clippy::type_complexity)]
static LIST_NOTES: Mutex<
    std::collections::BTreeMap<
        u32,
        (
            std::sync::Weak<Mutex<Vec<Note>>>,
            std::sync::Weak<Mutex<std::collections::BTreeSet<String>>>,
            std::sync::Weak<Mutex<indirects::Indirects>>,
        ),
    >,
> = Mutex::new(std::collections::BTreeMap::new());

/// The number in the name of a list of ours ("victory host list 7", which the
/// debug layer ends with a note of its own in brackets).
#[cfg(feature = "diagnostics")]
fn list_number(name: &str) -> Option<u32> {
    let rest = name.strip_prefix("victory host list ")?;
    let digits = rest.split(' ').next()?;
    (rest.len() == digits.len() || rest[digits.len()..].starts_with(" (")).then(|| digits.parse().ok()).flatten()
}

/// Indirect mesh dispatches since the last frame line: the largest group
/// count a record asked for, and the groups asked in all (from the lists'
/// budget buffers, a run late).
static MESH_GROUPS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Indirect records that ran empty for want of their list's budget, and the
/// largest cost of one (told from 2^22 on), likewise.
static BUDGET_NOTES: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Whether a list's indirect-work budget lives in the card's own memory
/// (`VICTORY_INDIRECT_CARD_BUDGET=0` keeps it in memory the CPU maps, as
/// before). The clamp charges it with atomics, at least two in a row for
/// every clamp; in mapped memory on a card of its own each of them
/// crosses the bus.
fn card_budget() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_INDIRECT_CARD_BUDGET").is_none_or(|v| v != "0"))
}

/// Whether each list of a submission runs and is timed on its own, slow
/// ones logged with their commands (`VICTORY_GPU_TIMING`, diagnostics
/// builds; the commands are noted with `VICTORY_D3D12_DRED` on).
#[cfg(feature = "diagnostics")]
fn timing() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_GPU_TIMING"))
}

/// Names a host object for the device-removed report, which names the
/// allocations a page fault hit (only with device-removed data on), or for
/// the frame recorder (`VICTORY_RECORD`).
pub(crate) fn name(object: &impl Interface, text: impl FnOnce() -> String) {
    #[cfg(feature = "diagnostics")]
    if crate::book::on()
        && let Ok(resource) = object.cast::<ID3D12Resource>()
    {
        let text = text();
        crate::book::made(&resource, || text.clone());
        return name_it(object, || text);
    }
    name_it(object, text);
}

fn name_it(object: &impl Interface, text: impl FnOnce() -> String) {
    #[cfg(feature = "diagnostics")]
    if crate::replay::record::on() {
        let text = text();
        crate::replay::record::name(object, &text);
        if dred_any()
            && let Ok(object) = object.cast::<ID3D12Object>()
        {
            // SAFETY: a plain call on a live object.
            let _ = unsafe { object.SetName(&windows::core::HSTRING::from(text)) };
        }
        return;
    }
    if dred_any()
        && let Ok(object) = object.cast::<ID3D12Object>()
    {
        // SAFETY: a plain call on a live object.
        let _ = unsafe { object.SetName(&windows::core::HSTRING::from(text())) };
    }
}

/// One in [`ALIVE`]`[0]` while it lives (a host list), and one in
/// [`LISTS`]`[3]` until it is first reset.
struct Counted {
    /// Not reset since it was made.
    fresh: std::sync::atomic::AtomicBool,
}

impl Counted {
    fn new() -> Counted {
        ALIVE[0].fetch_add(1, Ordering::Relaxed);
        LISTS[3].fetch_add(1, Ordering::Relaxed);
        Counted {
            fresh: std::sync::atomic::AtomicBool::new(true),
        }
    }

    /// The list is reset: it is in use.
    fn used(&self) {
        if self.fresh.swap(false, Ordering::Relaxed) {
            LISTS[3].fetch_sub(1, Ordering::Relaxed);
        }
    }
}

impl Drop for Counted {
    fn drop(&mut self) {
        ALIVE[0].fetch_sub(1, Ordering::Relaxed);
        self.used();
    }
}

/// A chunk of upload memory for lists' scratch data: the buffer, its
/// mapping.
struct Chunk {
    buffer: ID3D12Resource,
    data: usize,
}

// SAFETY: a D3D12 resource (free-threaded) and its persistent mapping,
// written only by the list that holds the chunk.
unsafe impl Send for Chunk {}

/// Upload chunks handed out front to back: the chunks (each with its
/// bytes), the one in use and the bytes used in it.
#[derive(Default)]
struct Scratch {
    chunks: Vec<(Chunk, u64)>,
    current: usize,
    used: u64,
}

// SAFETY: the chunk pointers are persistent mappings of upload buffers.
unsafe impl Send for Scratch {}

impl Drop for Scratch {
    fn drop(&mut self) {
        give_chunks(std::mem::take(&mut self.chunks));
    }
}

const SCRATCH_CHUNK: u64 = 4 << 20;
/// A list's first chunk when the chunks are shared ([`scratch_pool`]): most
/// lists put a few constants or a rectangle's corners there, and every
/// list that did took 4 MiB of the PC's memory for it.
const SCRATCH_FIRST: u64 = 256 << 10;
/// The most bytes of chunks kept while no list holds them.
const SPARE_MOST: u64 = 128 << 20;
/// The same with [`keep_freed`].
const SPARE_KEPT: u64 = 1 << 30;

/// Whether lists share their scratch chunks: a list takes a chunk when it
/// first needs one, a small one first, and gives all of them back when it
/// is reset or dropped, for the next list. Before, a list kept every chunk
/// it ever took (4 MiB each) for as long as it lived, used or not, and the
/// host lists of held submissions, which live for one submission, each
/// made theirs anew: some 57 upload buffers of 4 MiB made, zeroed and
/// freed per frame of gameplay (run 60).
/// `VICTORY_SCRATCH_POOL=0`: chunks of 4 MiB kept by their list, as before.
fn scratch_pool() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SCRATCH_POOL").is_none_or(|v| v != "0"))
}

/// Things of some sizes that nothing holds now, kept for whoever next asks
/// for one of the same size.
struct Spare<T> {
    kept: Vec<(T, u64)>,
    bytes: u64,
}

impl<T> Spare<T> {
    const fn new() -> Self {
        Spare { kept: Vec::new(), bytes: 0 }
    }

    /// A kept thing of `size` bytes, the one given last.
    fn take(&mut self, size: u64) -> Option<T> {
        let at = self.kept.iter().rposition(|(_, s)| *s == size)?;
        self.bytes -= size;
        Some(self.kept.remove(at).0)
    }

    /// Keeps `thing` (`size` bytes) unless that takes the kept bytes past
    /// `most`: then it comes back.
    fn give(&mut self, thing: T, size: u64, most: u64) -> Option<T> {
        if self.bytes + size > most {
            return Some(thing);
        }
        self.bytes += size;
        self.kept.push((thing, size));
        None
    }
}

/// Texture readback buffers are kept apart from the mapped upload chunks.
/// Pool only common sizes and count buffer widths, not driver residency.
fn texture_readback_idle_limit() -> u64 {
    static LIMIT: OnceLock<u64> = OnceLock::new();
    *LIMIT.get_or_init(|| {
        if !std::env::var_os("VICTORY_READBACK_POOL").is_some_and(|value| value == "1") {
            return 0;
        }
        std::env::var("VICTORY_READBACK_POOL_MB")
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
            .and_then(|mib| mib.checked_mul(1 << 20))
            .unwrap_or(256 << 20)
    })
}

#[derive(Clone, Copy, Default)]
pub(crate) struct ReadbackStats {
    pub(crate) created: u64,
    pub(crate) reused: u64,
    pub(crate) allocation_failures: u64,
    pub(crate) allocation_us: u64,
    pub(crate) peak_allocation_us: u64,
    pub(crate) active_buffers: u64,
    pub(crate) active_bytes: u64,
    pub(crate) peak_active_bytes: u64,
    pub(crate) idle_bytes: u64,
    pub(crate) peak_idle_bytes: u64,
}

struct ReadbackPool<T>(Mutex<(Spare<T>, ReadbackStats)>);

impl<T> ReadbackPool<T> {
    fn new() -> Self {
        Self(Mutex::new((Spare::new(), ReadbackStats::default())))
    }

    fn take(
        &self,
        size: u64,
        idle_limit: u64,
        mut allocate: impl FnMut(u64) -> Option<T>,
    ) -> Option<ReadbackLease<'_, T>> {
        let size = size.max(1);
        let class = size
            .max(64 << 10)
            .checked_next_power_of_two()
            .filter(|&bytes| bytes <= (16 << 20) && bytes <= idle_limit);
        if let Some(bytes) = class {
            let (idle, stats) = &mut *self.0.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(buffer) = idle.take(bytes) {
                stats.reused += 1;
                stats.idle_bytes = idle.bytes;
                return Some(self.lease(buffer, bytes, idle_limit, stats));
            }
        }

        // Do not hold the pool lock across a driver allocation. If rounding
        // up was refused, make the old, exact-sized allocation once.
        let capacity = class.unwrap_or(size);
        let mut allocation_us = 0;
        let mut peak_allocation_us = 0;
        let mut measured_allocate = |bytes| {
            let started = std::time::Instant::now();
            let buffer = allocate(bytes);
            let us = started.elapsed().as_micros() as u64;
            allocation_us += us;
            peak_allocation_us = peak_allocation_us.max(us);
            buffer
        };
        let (buffer, bytes, retain, failed_first) = match measured_allocate(capacity) {
            Some(buffer) => (Some(buffer), capacity, class.map_or(0, |_| idle_limit), 0),
            None if capacity != size => (measured_allocate(size), size, 0, 1),
            None => (None, size, 0, 0),
        };
        let (_, stats) = &mut *self.0.lock().unwrap_or_else(|e| e.into_inner());
        stats.allocation_failures += failed_first + u64::from(buffer.is_none());
        stats.allocation_us += allocation_us;
        stats.peak_allocation_us = stats.peak_allocation_us.max(peak_allocation_us);
        let buffer = buffer?;
        stats.created += 1;
        Some(self.lease(buffer, bytes, retain, stats))
    }

    fn lease(
        &self,
        buffer: T,
        bytes: u64,
        idle_limit: u64,
        stats: &mut ReadbackStats,
    ) -> ReadbackLease<'_, T> {
        stats.active_buffers += 1;
        stats.active_bytes += bytes;
        stats.peak_active_bytes = stats.peak_active_bytes.max(stats.active_bytes);
        ReadbackLease { pool: self, buffer: Some(buffer), bytes, idle_limit }
    }
}

pub(crate) type TextureReadback = ReadbackLease<'static, ID3D12Resource>;

/// An exclusive buffer owned by a list's recorded texture readback. It stays
/// checked out while that recording exists, even after one execution lands.
pub(crate) struct ReadbackLease<'a, T> {
    pool: &'a ReadbackPool<T>,
    buffer: Option<T>,
    bytes: u64,
    idle_limit: u64,
}

impl<T> ReadbackLease<'_, T> {
    pub(crate) fn resource(&self) -> &T {
        self.buffer.as_ref().expect("a live readback has its buffer")
    }

    pub(crate) fn stats(&self) -> ReadbackStats {
        self.pool.0.lock().unwrap_or_else(|e| e.into_inner()).1
    }
}

impl<T> Drop for ReadbackLease<'_, T> {
    fn drop(&mut self) {
        let Some(buffer) = self.buffer.take() else { return };
        let released = {
            let (idle, stats) = &mut *self.pool.0.lock().unwrap_or_else(|e| e.into_inner());
            stats.active_buffers -= 1;
            stats.active_bytes -= self.bytes;
            let released = idle.give(buffer, self.bytes, self.idle_limit);
            stats.idle_bytes = idle.bytes;
            stats.peak_idle_bytes = stats.peak_idle_bytes.max(stats.idle_bytes);
            released
        };
        // Drop a committed D3D12 resource outside the lock.
        drop(released);
    }
}

fn texture_readback_fence_error(
    completed: u64,
    submitted: u64,
    report_removed: impl FnOnce() -> String,
) -> Option<String> {
    // D3D12 uses UINT64_MAX for a removed device.
    if completed == u64::MAX {
        Some(report_removed())
    } else if completed < submitted {
        Some(format!("texture readback pool cannot retire before fence {submitted} (completed {completed})"))
    } else {
        None
    }
}

#[cfg(test)]
mod readback_pool_tests {
    use super::*;

    #[test]
    fn readback_fence_preserves_device_removal_details() {
        let describe = || "device removed with breadcrumbs".to_owned();
        assert_eq!(
            texture_readback_fence_error(u64::MAX, 8, describe).as_deref(),
            Some("device removed with breadcrumbs")
        );
        assert!(texture_readback_fence_error(7, 8, || panic!("no device removal"))
            .unwrap().contains("before fence 8 (completed 7)"));
        assert!(texture_readback_fence_error(8, 8, || panic!("fence complete")).is_none());
    }

    #[test]
    fn readback_pool_leases_are_exclusive_until_retired_and_idle_bytes_are_bounded() {
        const CLASS: u64 = 64 << 10;
        let pool = ReadbackPool::new();
        let mut created = 0;
        let mut allocate = |bytes| {
            assert_eq!(bytes, CLASS);
            created += 1;
            Some(created)
        };
        let first = pool.take(32 << 10, CLASS, &mut allocate).unwrap();
        let second = pool.take(32 << 10, CLASS, &mut allocate).unwrap();
        assert_eq!(*first.resource(), 1);
        assert_eq!(*second.resource(), 2);
        // Inspecting the same recording again (as a repeated submission
        // does at landing) cannot make its still-owned buffer available.
        assert_eq!(*first.resource(), 1);
        let stats = first.stats();
        assert_eq!((stats.created, stats.reused, stats.active_buffers), (2, 0, 2));
        assert_eq!((stats.active_bytes, stats.peak_active_bytes, stats.idle_bytes), (2 * CLASS, 2 * CLASS, 0));

        drop(first);
        let reused = pool.take(32 << 10, CLASS, &mut allocate).unwrap();
        assert_eq!(*reused.resource(), 1);
        assert_eq!(*second.resource(), 2);
        drop(reused);
        drop(second);
        let stats = pool.0.lock().unwrap().1;
        assert_eq!((stats.created, stats.reused, stats.active_buffers, stats.active_bytes), (2, 1, 0, 0));
        assert_eq!((stats.idle_bytes, stats.peak_idle_bytes), (CLASS, CLASS));
    }

    #[test]
    fn readback_pool_uses_exact_unshared_allocations_when_disabled_or_over_the_class_limit() {
        let pool = ReadbackPool::new();
        for (size, limit) in [(32 << 10, 0), (32 << 10, 16 << 10), ((16 << 20) + 1, 256 << 20)] {
            let lease = pool.take(size, limit, Some).unwrap();
            assert_eq!(*lease.resource(), size);
        }
        let stats = pool.0.lock().unwrap().1;
        assert_eq!((stats.created, stats.reused, stats.idle_bytes, stats.active_buffers), (3, 0, 0, 0));
    }

    #[test]
    fn readback_pool_tries_original_size_after_a_rounded_allocation_fails() {
        let pool = ReadbackPool::new();
        let mut attempts = Vec::new();
        let lease = pool
            .take(32 << 10, 256 << 20, |size| {
                attempts.push(size);
                (size == (32 << 10)).then_some(size)
            })
            .unwrap();
        assert_eq!(attempts, [64 << 10, 32 << 10]);
        assert_eq!(lease.stats().allocation_failures, 1);
        assert_eq!(*lease.resource(), 32 << 10);
        drop(lease);
        assert!(pool.take(32 << 10, 256 << 20, |_| None).is_none());
        let stats = pool.0.lock().unwrap().1;
        assert_eq!((stats.created, stats.allocation_failures, stats.idle_bytes), (1, 3, 0));
    }

    #[test]
    fn readback_pool_keeps_closed_recordings_exclusive_until_reset() {
        const ROW: u64 = 256;
        const CLASS: u64 = 64 << 10;
        let host = host().expect("D3D12 is needed for the readback recording test");
        // SAFETY: a private queue and fence on the same device as the test lists.
        let (queue, fence): (ID3D12CommandQueue, ID3D12Fence) = unsafe {
            (
                host.device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                }).unwrap(),
                host.device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap(),
            )
        };
        let submit = |lists: &[Option<ID3D12CommandList>], target| {
            // SAFETY: lists are closed and their resources are kept through completion.
            let signal = unsafe {
                queue.ExecuteCommandLists(lists);
                queue.Signal(&fence, target)
            };
            if signal.is_err() {
                return false;
            }
            let end = std::time::Instant::now() + std::time::Duration::from_secs(5);
            loop {
                // SAFETY: a plain query of the test's private fence.
                let completed = unsafe { fence.GetCompletedValue() };
                if completed == u64::MAX || std::time::Instant::now() >= end {
                    return false;
                }
                if completed >= target {
                    return true;
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        };
        let take = || -> TextureReadback {
            host.texture_readbacks
                .take(ROW, CLASS, |bytes| host.read_back_buffer(bytes))
                .expect("a pooled readback buffer")
        };
        let record = |source: &ID3D12Resource, lease: TextureReadback, dst: &mut [u8; ROW as usize]| {
            let list = host.new_list().unwrap();
            // SAFETY: a full initialized upload row, an open list, and a
            // live readback resource in COPY_DEST.
            unsafe { list.list.CopyBufferRegion(lease.resource(), 0, source, 0, ROW) };
            list.deferred().texture_backs.push(crate::texback::buffer_back_for_test(lease, dst));
            list.close();
            list
        };
        let upload = |value: u8| {
            let (buffer, data) = host.upload_buffer(ROW).expect("a mapped upload row");
            assert!(!data.is_null());
            // SAFETY: the upload row is mapped and has no GPU users yet.
            unsafe { data.write_bytes(value, ROW as usize) };
            (buffer, data)
        };
        let (source_a, data_a) = upload(0x1a);
        let mut dst_a = [0; ROW as usize];
        let lease_a = take();
        let address_a = lease_a.resource().as_raw();
        let list_a = record(&source_a, lease_a, &mut dst_a);
        if !submit(&[Some(list_a.list.cast().unwrap())], 1) {
            // Keep potential GPU uses alive on the failure path.
            std::mem::forget((list_a, source_a, queue.clone(), fence.clone()));
            panic!("first readback fence did not complete");
        }
        let mut scratch = crate::texback::Scratch::default();
        read_backs(&list_a, &mut scratch);
        assert_eq!(&dst_a[..4], &[0x1a; 4]);

        // The first fence did not retire the closed recording. A second
        // list must get a different destination before both run together.
        let lease_b = take();
        assert_ne!(lease_b.resource().as_raw(), address_a);
        let (source_b, _) = upload(0x2b);
        let mut dst_b = [0; ROW as usize];
        let list_b = record(&source_b, lease_b, &mut dst_b);
        // SAFETY: the previous GPU read of source_a has finished.
        unsafe { data_a.write_bytes(0x3c, ROW as usize) };
        let lists = [Some(list_a.list.cast().unwrap()), Some(list_b.list.cast().unwrap())];
        if !submit(&lists, 2) {
            std::mem::forget((list_a, list_b, source_a, source_b, queue.clone(), fence.clone()));
            panic!("second readback fence did not complete");
        }
        drop(lists);
        read_backs(&list_a, &mut scratch);
        read_backs(&list_b, &mut scratch);
        assert_eq!(&dst_a[..4], &[0x3c; 4]);
        assert_eq!(&dst_b[..4], &[0x2b; 4]);

        list_a.reset();
        let reassigned = take();
        assert_eq!(reassigned.resource().as_raw(), address_a);
    }
}

/// Scratch chunks no list holds.
static SPARE: Mutex<Spare<Chunk>> = Mutex::new(Spare::new());

/// A scratch chunk of `size` bytes: one kept ([`SPARE`]) or a new one.
fn take_chunk(host: &Host, size: u64) -> Option<Chunk> {
    if scratch_pool()
        && let Some(chunk) = SPARE.lock().unwrap_or_else(|e| e.into_inner()).take(size)
    {
        ALIVE[3].fetch_sub(1, Ordering::Relaxed);
        ALIVE[4].fetch_sub(size, Ordering::Relaxed);
        ALIVE[5].fetch_add(1, Ordering::Relaxed);
        return Some(chunk);
    }
    let (buffer, data) = host.upload_buffer(size)?;
    name(&buffer, || "list scratch".into());
    ALIVE[1].fetch_add(1, Ordering::Relaxed);
    ALIVE[2].fetch_add(size, Ordering::Relaxed);
    Some(Chunk {
        buffer,
        data: data as usize,
    })
}

/// A list is done with `chunks` (it was reset after its submissions ran,
/// or dropped): kept for the next list, or freed past [`SPARE_MOST`] and
/// without the pool.
fn give_chunks(chunks: Vec<(Chunk, u64)>) {
    if chunks.is_empty() {
        return;
    }
    let mut freed = Vec::new();
    {
        let mut spare = SPARE.lock().unwrap_or_else(|e| e.into_inner());
        for (chunk, size) in chunks {
            let back = if scratch_pool() {
                let limit = if keep_freed() && !crate::residency::limited_host_ram() {
                    SPARE_KEPT
                } else {
                    SPARE_MOST
                };
                spare.give(chunk, size, limit)
            } else {
                Some(chunk)
            };
            match back {
                None => {
                    ALIVE[3].fetch_add(1, Ordering::Relaxed);
                    ALIVE[4].fetch_add(size, Ordering::Relaxed);
                }
                Some(chunk) => {
                    ALIVE[1].fetch_sub(1, Ordering::Relaxed);
                    ALIVE[2].fetch_sub(size, Ordering::Relaxed);
                    freed.push(chunk);
                }
            }
        }
    }
    // Freed outside the lock.
    drop(freed);
}

/// Drops spare scratch chunks kept idle when host memory is pressured.
fn trim_spare_chunks() {
    let freed: Vec<(Chunk, u64)> = {
        let mut spare = SPARE.lock().unwrap_or_else(|e| e.into_inner());
        if spare.kept.is_empty() {
            return;
        }
        spare.bytes = 0;
        std::mem::take(&mut spare.kept)
    };
    let count = freed.len() as u64;
    let bytes: u64 = freed.iter().map(|(_, s)| *s).sum();
    ALIVE[1].fetch_sub(count, Ordering::Relaxed);
    ALIVE[2].fetch_sub(bytes, Ordering::Relaxed);
    ALIVE[3].fetch_sub(count, Ordering::Relaxed);
    ALIVE[4].fetch_sub(bytes, Ordering::Relaxed);
    drop(freed);
}

/// What the host keeps for copies in the card's own memory
/// ([`crate::shadow::cards`]): three lists of ours and the compare.
struct CardWork {
    /// Before a submission's lists: what changed on the copies' CPU side
    /// goes to their card side. Those not in use: a plan takes one and
    /// gives it back when its submission has run (`Host::card_done`), so
    /// that submissions sent ahead ([`late`]) each have their own.
    sync: Vec<Prelude>,
    /// After them, in the same submission: the compare of the declared
    /// ranges with their bytes from before. As `sync`.
    check: Vec<Prelude>,
    /// After it ran: the pieces that differ come down.
    fetch: Prelude,
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    /// One dword per piece compared, odd where the piece differs: PC
    /// memory, mapped (buffer, mapping, bytes).
    differs: (ID3D12Resource, usize, u64),
    /// Where the pieces that differ are fetched to: PC memory, mapped.
    fetched: Option<(ID3D12Resource, usize, u64)>,
    /// The dwords of `differs` taken by plans whose results were not read
    /// yet.
    taken: u32,
    /// Plans made and not done with (`Host::card_done`).
    out: u32,
}

// SAFETY: D3D12 objects (free-threaded) and persistent mappings, used only
// under the submission lock.
unsafe impl Send for CardWork {}

static CARD_WORK: Mutex<Option<CardWork>> = Mutex::new(None);

/// The compare of a declared range of a copy in the card's memory with
/// its bytes from before, one group per 64 KiB piece (cut at multiples of
/// 64 KiB of title memory, as `Shadow::card_piece` cuts them): `differs`
/// gets a dword per piece, odd where a byte differs. The offsets are widened
/// to 16 bytes; buffers are multiples of 64 KiB, and a piece flagged for a
/// byte just outside the range costs a fetch that finds nothing.
///
/// A piece's 64 threads walk it side by side, each every 64th 16 bytes, and
/// one of them hashes their 64 hashes: 64 steps after one another. One thread
/// walked the piece before, 4,096 steps; the passes took 24.8 ms of a frame's
/// 82 on the card (capture replay1, 405 of them), most of it waiting for
/// those steps.
pub(crate) const CARD_COMPARE: &str = r#"
#define RS "RootConstants(num32BitConstants=6, b0), SRV(t0), UAV(u0), UAV(u1)"
cbuffer Span : register(b0) {
    uint first;    // the first piece
    uint skew;     // where the copy starts in its piece 0
    uint size;     // the copy's buffer, in bytes (a multiple of 64 KiB)
    uint pieces;
    uint out_at;   // the first piece's dword in `differs`, in bytes
    uint compare;  // 0: the hashes are stored; else compared with those stored (2: and then stored)
};
ByteAddressBuffer now : register(t0);
// 16 bytes per piece of the copy: the hash of its bytes when they were
// last sent up (or hashed whole, the first time).
RWByteAddressBuffer hashes : register(u0);
RWByteAddressBuffer differs : register(u1);
groupshared uint4 parts[64];
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_GroupID, uint thread : SV_GroupIndex) {
    uint piece = first + id.x;
    uint lo = piece * 65536u;
    lo = lo > skew ? lo - skew : 0u;
    uint hi = min((piece + 1u) * 65536u - skew, size);
    lo = lo & ~15u;
    hi = (hi + 15u) & ~15u;
    uint4 h = uint4(0x9E3779B9u, 0x85EBCA6Bu, 0xC2B2AE35u, 0x27D4EB2Fu) + thread;
    uint at = lo + thread * 16u;
    // Eight loads at once: one after the other, each waits for memory.
    [loop]
    for (; at + 7168u < hi; at += 8192u) {
        uint4 v0 = now.Load4(at + 0u);
        uint4 v1 = now.Load4(at + 1024u);
        uint4 v2 = now.Load4(at + 2048u);
        uint4 v3 = now.Load4(at + 3072u);
        uint4 v4 = now.Load4(at + 4096u);
        uint4 v5 = now.Load4(at + 5120u);
        uint4 v6 = now.Load4(at + 6144u);
        uint4 v7 = now.Load4(at + 7168u);
        h = (h ^ v0) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
        h = (h ^ v1) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
        h = (h ^ v2) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
        h = (h ^ v3) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
        h = (h ^ v4) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
        h = (h ^ v5) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
        h = (h ^ v6) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
        h = (h ^ v7) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
    }
    [loop]
    for (; at < hi; at += 1024u) {
        h = (h ^ now.Load4(at)) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
    }
    parts[thread] = h;
    GroupMemoryBarrierWithGroupSync();
    if (thread != 0u) return;
    h = parts[0];
    [loop]
    for (uint i = 1u; i < 64u; ++i) {
        h = (h ^ parts[i]) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
    }
    if (compare != 0u) {
        // The flag, and above it 31 bits of the hash (`torn`).
        uint folded = h.x ^ (h.y * 3u) ^ (h.z * 5u) ^ (h.w * 7u);
        differs.Store(out_at + id.x * 4u, (folded << 1u) | (any(h != hashes.Load4(piece * 16u)) ? 1u : 0u));
        // With the copy-back on its own thread: the next compare of the
        // piece finds what is written from here on.
        if (compare == 2u) hashes.Store4(piece * 16u, h);
    } else {
        hashes.Store4(piece * 16u, h);
    }
}
"#;

/// The hash [`CARD_COMPARE`] gives of a piece with its flag.
#[cfg(test)]
pub(crate) fn piece_hash(bytes: &[u8]) -> u32 {
    torn::hash(bytes)
}

/// One pass of [`CARD_COMPARE`] over a span's pieces, into `differs` from
/// dword `at` on (`mode` as the shader's `compare`): a group per piece, in
/// parts a dispatch can hold.
///
/// # Safety
/// The list is open with the compare's root signature, pipeline and buffers
/// set, and the pieces lie in the buffer.
unsafe fn card_pass(list: &ID3D12GraphicsCommandList, span: &crate::shadow::CardSpan, size: u32, at: u32, mode: u32) {
    const MOST: u32 = D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION;
    let mut done = 0;
    while done < span.pieces {
        let n = (span.pieces - done).min(MOST);
        let constants = [span.first + done, span.skew, size, n, (at + done) * 4, mode];
        // SAFETY: the caller's.
        unsafe {
            list.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
            list.Dispatch(n, 1, 1);
        }
        done += n;
    }
}

/// Whether the pieces hashed again after ranges were sent up to a copy in
/// the card's memory are hashed in runs of pieces that touch, each piece
/// once (`VICTORY_CARD_REHASH_RUNS=1`; off by default), and not in a pass
/// per range sent up.
///
/// These passes are ours, they run on the card before every submission's
/// lists: a pass takes the time of one piece however few its pieces are.
/// Run 102 sent up 514 ranges a frame: 514 such passes. The same pieces are hashed either way;
/// a piece hashed when it need not be, or not hashed when it should, costs
/// a fetch that finds nothing and loses no shader write.
fn rehash_runs() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CARD_REHASH_RUNS").is_some_and(|v| v == "1"))
}

/// Since the last frame-rate line, of our own passes over copies in the
/// card's memory: hash passes after ranges were sent up, the pieces they
/// hashed, and compare passes after the lists.
static CARD_PASSES: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// Whether what shaders wrote to copies in the card's memory is merged
/// into title memory on several threads (`VICTORY_WIDE_MERGE=1`; off by
/// default). The pieces the card's compare found differing come down in
/// one fetch and were then merged one after the other on the thread that
/// settles: about 630 pieces and 40 MB a frame (run 104), each compared
/// with the copy's CPU side byte by byte and put into title memory through
/// the kernel. A piece is its own bytes of its copy and of title memory,
/// so eight threads take an eighth of them each. (Not with [`relay`], whose
/// landings do not come here: `crate::land` has threads of its own.)
fn wide_merge() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WIDE_MERGE").is_some_and(|v| v == "1"))
}

/// One declared range of a copy in the card's memory, in a submission.
struct CardRange {
    shadow: std::sync::Arc<crate::shadow::Shadow>,
    /// As declared (title addresses), for the statistics.
    declared: (usize, usize),
    span: crate::shadow::CardSpan,
    /// Its first piece's dword in the compare's results.
    at: u32,
    /// Whether the card compares it; else every piece is fetched.
    compared: bool,
    /// Looked at by `crate::picked`, and not declared: what is found changed
    /// there was written through a view nobody declared. Not landed.
    picked: bool,
}

impl CardPlan {
    /// `crate::picked` is told where what it looked at changed, by the
    /// compare's results.
    fn picked_found(&self, differs: &[u32]) {
        for r in self.ranges.iter().filter(|r| r.picked) {
            let changed = (0..r.span.pieces).find(|p| differs.get((r.at + p) as usize).is_some_and(|&d| torn::flagged(d)));
            if let Some(p) = changed {
                crate::picked::found(&r.shadow, r.shadow.card_piece(&r.span, p).0);
            }
        }
    }
}

/// What a submission does for copies in the card's memory.
struct CardPlan {
    ranges: Vec<CardRange>,
    /// The compare, run after the submission's lists.
    check: Option<ID3D12CommandList>,
    /// The copies whose card side was brought up to date, alive until the
    /// submission has run.
    _synced: Vec<std::sync::Arc<crate::shadow::Shadow>>,
    /// The lists of ours the plan recorded (whether the compare, the
    /// list), given back by `Host::card_done`.
    lists: Vec<(bool, Prelude)>,
    /// With [`relay`]: the number of the send to copies in the card's memory
    /// that the submission ran behind ([`QUEUE_ORDER`]).
    sent_as: u64,
    /// With [`upcheck`]: the look at what it sent up.
    looks: Option<upcheck::Looks>,
    /// With [`send`]: the memory its sends come from.
    sends: Option<send::Memory>,
    /// With [`upq`]: the copy list and buffer what went up came through.
    up: Option<upq::Slot>,
}

/// One piece of submission-time work, in recorded order.
enum Step {
    /// A copy into title memory (destination, source, bytes): the title's
    /// buffer copies, which later refreshes must see.
    Copy(usize, usize, usize),
    /// A copy of title memory into the list's own upload memory, and how
    /// the card could bring the bytes instead.
    Fill(usize, usize, usize, Option<card_steps::FillTo>),
    /// Brings a range (address, bytes) of a buffer's host copy up to date:
    /// once per submission, whichever lists ask, until a title copy intervenes.
    Refresh(std::sync::Arc<crate::shadow::Shadow>, usize, usize),
    /// A submission-time title copy whose placed route did not apply.
    RawCopy { dst: usize, src: usize, len: usize },
    /// Work that writes host memory only, and the title memory it reads
    /// (from, to; empty for none).
    Host(Box<dyn Fn() + Send>, (usize, usize)),
    /// Shaders write a range (address, bytes) of a buffer's host copy.
    GpuWrite(std::sync::Arc<crate::shadow::Shadow>, usize, usize),
    /// Brings a texture's staging buffer up to date: once per submission,
    /// whichever lists sample it, until a title copy intervenes.
    Texture(std::sync::Arc<crate::texture::Staging>),
    /// Reads a CPU-written linear movie plane into a per-list upload buffer.
    Plane((usize, usize), Box<dyn Fn() + Send>),
    /// A title buffer copy that the list does on the card ([`place`]): what
    /// is done on the CPU for it.
    Placed(place::Placed),
}

/// What a list does on the CPU when it is submitted: copies between host
/// memory regions (title buffers, upload buffers and buffer copies), and
/// the upload buffers its GPU commands read, kept until the list is reset.
#[derive(Default)]
struct Deferred {
    /// Destination, source, bytes.
    copies: Vec<Step>,
    uploads: Vec<ID3D12Resource>,
    /// Standalone upload footprints kept through this list's fence (movie planes).
    plane_uploads: Vec<(Chunk, u64)>,
    /// The bytes of `uploads` (counted in [`ALIVE`] while they are held).
    upload_bytes: u64,
    /// Host objects the list's commands name (pipelines, root signatures,
    /// descriptor heaps): the title may release its objects before it
    /// submits the list.
    kept: Vec<windows::core::IUnknown>,
    /// What a host library's commands in the list use (a DLSS feature):
    /// [`HostList::keep_shared`].
    shared: Vec<std::sync::Arc<dyn std::any::Any + Send + Sync>>,
    frame_preparation: Vec<crate::framegen::Preparation>,
    /// The pipelines [`HostList::bind_pipeline`] kept, by address.
    bound: std::collections::HashSet<usize, std::hash::BuildHasherDefault<AddressHash>>,
    /// Buffer ranges refreshed since the last title copy (host copy,
    /// address, bytes): refreshing them again would find the same bytes.
    refreshed: std::collections::HashSet<(usize, usize, usize), std::hash::BuildHasherDefault<AddressHash>>,
    /// Ranges declared written by shaders since the list last wrote title
    /// memory (host copy, address, bytes): declaring them again would
    /// change nothing.
    gpu_written: std::collections::HashSet<(usize, usize, usize), std::hash::BuildHasherDefault<AddressHash>>,
    /// Scratch memory [`HostList::fill_scratch`] fills since the list last
    /// wrote title memory (source, bytes): its host address.
    filled: std::collections::HashMap<(usize, usize), u64, std::hash::BuildHasherDefault<AddressHash>>,
    /// Disjoint placed copies that can supply constants in this recording, sorted by address.
    placed_constants: Vec<place::ConstantSource>,
    /// Reusable snapshots by title address and length.
    constant_values: std::collections::BTreeMap<(usize, usize), place::ConstantSnapshot>,
    /// Changes when a recorded write or reset can invalidate a pending snapshot.
    constant_write_version: u64,
    /// GPU snapshots of constant data, held through every use of the list.
    constant_snapshots: Vec<ID3D12Resource>,
    /// Textures whose contents the list brings up to date.
    textures: std::collections::HashSet<usize>,
    /// Last GPU source copied into each aliased host texture in this list.
    texture_aliases: std::collections::HashMap<usize, usize>,
    /// Bottom-of-pipe writes, done once the list has run: address, what,
    /// bytes.
    writes: Vec<(usize, crate::queue::Written, usize)>,
    /// Textures read back (texture to buffer copies), done once the list
    /// has run: the host buffer the list copies the texels into, the title
    /// memory they go to, the bytes of a row, the rows, the title's row
    /// pitch.
    backs: Vec<(ID3D12Resource, usize, [usize; 3])>,
    /// Title textures of the kinds copied back ([`crate::texback`]) that
    /// the list's shaders write, copied when the list closes.
    back_wanted: Vec<crate::texback::Target>,
    /// Those copies, whose texels go to title memory once the list has run.
    texture_backs: Vec<crate::texback::Back>,
    /// Whether the indirect-work budget is zeroed for this recording (a
    /// step on the CPU, or in the list for the card's copy).
    budget_zeroed: bool,
    /// The indirect scratch buffer (`HostList::indirect_args`, by its
    /// interface pointer) that a clamp in this recording left readable as
    /// indirect arguments; 0 if none. It stays readable until the next
    /// clamp into it ([`HostList::indirect_state`]).
    indirect_readable: usize,
}

/// Mark preparations only once their lists reached the graphics queue.
fn mark_frame_preparations(runs: &[&HostList]) {
    if crate::framegen::requested() {
        for run in runs {
            for preparation in &run.deferred().frame_preparation {
                preparation.submitted();
            }
        }
    }
}

impl Deferred {
    /// Title memory updated when these texture readbacks land.
    fn readback_ranges(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.backs
            .iter()
            .map(|&(_, at, [row, rows, pitch])| (at, at + rows.saturating_sub(1) * pitch + row))
            .chain(self.texture_backs.iter().map(crate::texback::Back::range))
    }

    /// Holds `buffer` (`size` bytes of upload memory) until the list is
    /// reset.
    fn hold_upload(&mut self, buffer: ID3D12Resource, size: u64) {
        ALIVE[6].fetch_add(1, Ordering::Relaxed);
        ALIVE[7].fetch_add(size, Ordering::Relaxed);
        self.upload_bytes += size;
        self.uploads.push(buffer);
    }

    /// The steps recorded last write title memory. A copy already recorded on the GPU
    /// only replaces sources it overlaps. Deferred CPU writes invalidate them all.
    fn title_written(&mut self, placed: Option<(usize, usize)>) {
        self.refreshed.clear();
        self.gpu_written.clear();
        self.filled.clear();
        if let Some((address, len)) = placed {
            self.forget_placed_constants(address, len);
        } else if !self.placed_constants.is_empty() || !self.constant_values.is_empty() {
            self.constant_write_version = self.constant_write_version.wrapping_add(1);
            self.placed_constants.clear();
            self.constant_values.clear();
        }
    }
}

impl Drop for Deferred {
    fn drop(&mut self) {
        give_chunks(std::mem::take(&mut self.plane_uploads));
        ALIVE[6].fetch_sub(self.uploads.len() as u64, Ordering::Relaxed);
        ALIVE[7].fetch_sub(self.upload_bytes, Ordering::Relaxed);
    }
}

/// Whether a queue's submissions go to the card without a wait for each
/// (`VICTORY_LATE_WAIT=1`; off by default, and off in the timing run, which
/// waits for every list on its own).
///
/// Every submission was prepared, sent, waited for and copied back before
/// the next one was looked at, so the card stood idle while we prepared
/// and we stood idle while it ran: in run 86 a frame of 446 ms is 134 ms of
/// preparing, 194 ms of waiting and 78 ms of copying back, one after the
/// other, on the one thread that submits.
///
/// With the switch a queue's submission is sent and joins a line of
/// "flights" ([`Flight`]); the thread goes on to the next one while the
/// card runs. The flights are settled together ([`Host::settle`]: one wait
/// for the card, what their shaders wrote copied back once per byte, then
/// their bottom-of-pipe writes and the fence signals behind them, in
/// order) at the next point where the title, or our own preparing, can
/// observe what they do. Those points, exactly:
///
/// 1. a synchronous submission of ours ([`Host::execute`]): a present, a
///    read-back or map of a texture, a clear or upload run at once;
/// 2. the next submission reads title memory that a flight's shaders may
///    write: a title buffer copy, a fill of upload memory, a texture's
///    memory, any step that runs code of ours on title memory
///    ([`late_blocked`]);
/// 3. the next submission refreshes a range of a host copy that a flight's
///    shaders may write, and the title wrote a page of it since it was last
///    copied ([`stale_under_flights`]): the refresh leaves such a range
///    alone until the flight's writes are back, as it always did within a
///    submission;
/// 4. a queue stands at a wait for a memory value that no flight's
///    bottom-of-pipe write is known to satisfy, at an address a flight may
///    write ([`late_met`]);
/// 5. the title blocks on a fence without an event;
/// 6. the card has run every flight (a thread of ours waits for that:
///    [`late_helper`]), or the flights hold [`late_most`] host lists.
///
/// What the title is told (bottom-of-pipe writes, fence values) is told
/// when the card has really run the work, later than before and never
/// earlier; a fence signalled on a queue while flights are in the air is
/// set behind them. A wait for a memory value that a flight's
/// bottom-of-pipe write satisfies counts as met: the waiting part goes to
/// the card behind that flight, which is the order the wait asks for.
///
/// What it assumes beyond today: memory the title rewrites between two of
/// its submissions, without waiting for the first, may be seen by the
/// first's shaders (as on the console, where the card reads title memory
/// when it runs, not when the list is submitted).
pub(crate) fn late() -> bool {
    late_stage() > 0
}

/// `VICTORY_LATE_WAIT`: 0 (off), 1, or 2 ([`late_exact`]).
fn late_stage() -> u8 {
    static STAGE: OnceLock<u8> = OnceLock::new();
    *STAGE.get_or_init(|| {
        if std::env::var_os("VICTORY_GPU_TIMING").is_some() {
            return 0;
        }
        match std::env::var("VICTORY_LATE_WAIT").as_deref() {
            Ok("1") => 1,
            Ok("2") => 2,
            Ok("3") => 3,
            // The relay alone: on top of stage 3.
            Err(_) if crate::shadow::relay() => 3,
            _ => 0,
        }
    })
}

/// Stage 2 of [`late`] (`VICTORY_LATE_WAIT=2`): point 3 of its list is no
/// longer a point. Where the title wrote a page that lies in a range a
/// flight's shaders may write, the bytes the title changed are found (the
/// page as it is now against what the copy held when the flight was
/// prepared) and put into the copy, and only those: what the flight's
/// shaders write beside them stays (`Shadow::merge_under_writes`). The
/// flights are settled for it only where that cannot be done: memory that
/// is not write-watched, or a copy in the card's memory with more changed
/// runs than a submission should send up one by one.
///
/// In run 92 (stage 1) this point settled the flights five times a frame
/// of twelve, each time with the card's work still to wait for: 172 ms a
/// frame of waiting were left of 200.
///
/// What differs from settling first: a byte both the title and a flight's
/// shader wrote is the title's if the shader wrote it first, the shader's
/// if it wrote it afterwards; settled first it was always the shader's.
/// The title writing a byte a shader of its own is writing is a race on
/// the console too.
fn late_exact() -> bool {
    late_stage() > 1
}

/// Stage 3 of [`late`] (`VICTORY_LATE_WAIT=3`: stage 2, and the present is
/// no longer a point where flights are settled). The copy to the back
/// buffer is sent as a flight of its own (`Host::execute_own`, a list of
/// its own for each present in the air) and the swap chain is told to
/// present behind it; the queue's call returns, and the card finishes the
/// frame while the title records the next one. What the title is told
/// about the frame (bottom-of-pipe writes, fences) is told when the card
/// has run it, by the settling thread.
///
/// What is left out with it: the look whether the presented frame is black
/// (diagnostics; it reads the frame back at once). A frame that is saved
/// or listed is presented the old way. Once every second and a half a
/// submission settles the flights all the same, for the sweep that lets
/// idle host objects go (it needs nothing in the air).
pub(crate) fn late_free_present() -> bool {
    late_stage() > 2
}

/// The relay's first stage (`VICTORY_RELAY=1`; off by default; it turns
/// [`late`] on at stage 3 unless `VICTORY_LATE_WAIT` says another stage,
/// and is off with it, in the timing run and with kept snapshots).
///
/// With [`late`] alone every settle does three things on the thread that
/// holds the submission lock: it waits for everything in the air, copies
/// back with the card standing, and only then goes on preparing. The frame
/// stays a relay: the card works about 242 ms of it and one thread
/// prepares, waits and copies back for about 220.
///
/// With the switch a thread of its own ([`relay_worker`]) lands flights as
/// soon as the card has run them, in the order they were sent and without
/// the submission lock: it fetches and merges what their shaders wrote to
/// copies in the card's memory, compares and merges the other copies,
/// frees the flights' ranges, and then does their read-backs, their
/// bottom-of-pipe writes and the fence signals behind them, in the line's
/// order. The submitting thread meanwhile prepares and sends the next
/// submissions. [`Host::settle`] no longer settles anything: it waits until
/// that thread has landed everything in the air, at the same points as
/// before (the list in [`late`]), so whoever can observe title memory, a
/// fence or a present still does so only after the flights before it are
/// merged.
///
/// What makes that safe is in `crate::shadow::relay` (ranges counted per
/// flight; a merge that never stores into memory the card may be writing;
/// sends to card copies numbered) and here: flights land in the order of
/// [`LINE`] by one thread, so a later flight's results never reach title
/// memory before an earlier flight's; the compare on the card stores the
/// hash it made, so each flight's compare flags exactly what changed since
/// the compare before it, however late the pieces are fetched; the fetch
/// goes to the queue under [`QUEUE_ORDER`], which says which sends of
/// title bytes the card had received when it ran.
pub(crate) fn relay() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| late_stage() > 0 && crate::shadow::relay())
}

/// The relay's second stage (`VICTORY_RELAY=2`): a submission that reads
/// title memory a flight's shaders may write (a title buffer copy, a fill
/// of upload memory, a texture) waits only until the flights that declared
/// that memory have landed, not for everything in the air; and at
/// [`late_most`] lists in the air it waits until there are fewer. At `1`
/// both wait for everything.
fn relay_exact() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    // Stage 2 unless `VICTORY_RELAY=1` asks for the first one (the default
    // since the walls of the first room were found drawn in every run with
    // this set and in fewer than half without it).
    *ON.get_or_init(|| relay() && std::env::var_os("VICTORY_RELAY").is_none_or(|v| v == "2"))
}

fn relay_mode() -> u8 {
    u8::from(relay()) + u8::from(relay_exact())
}

/// Whether the pieces the relay fetches from the card's memory come over the
/// card's copy queue (`VICTORY_RELAY_COPY=1`) and not over the queue that
/// draws.
///
/// On the drawing queue a fetch runs behind whatever was sent meanwhile,
/// and with the relay that is most of a frame: the thread that lands flights
/// stood 48 ms of a 120 ms frame waiting for seven fetches of 94 MB in all,
/// a few milliseconds of copying, and everything that waits for a landing
/// (the title's fences, reads of memory flights may write) waited with it.
///
/// What changes: the fetch runs when its flights are done and beside later
/// submissions, so of the sends of title bytes it is only known to have run
/// behind those its own flights ran behind (`CardPlan::sent_as`). Bytes sent
/// later are left out of the merge (`Shadow::card_merge_relay`): they are
/// the title's, newer than what the flights wrote. A later flight's bytes
/// may be in what is fetched, as before.
fn relay_copy() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_RELAY_COPY").is_some_and(|v| v == "1"))
}

/// Told when something joins [`LINE`], for the thread that lands flights
/// ([`relay`]); [`LINE_MOVED`] stays the word for the thread that looks at
/// what queues hold.
static RELAY_MOVED: std::sync::Condvar = std::sync::Condvar::new();

/// Told when flights have landed ([`relay`]): who waits for that looks
/// again.
static LINE_LANDED: std::sync::Condvar = std::sync::Condvar::new();

/// With [`relay`], the order of what we put on the queue: the number of
/// sends of changed ranges to copies in the card's memory so far
/// (`Host::card_before`). Held from the taking of the ranges until their
/// list is on the queue, and while a fetch of pieces goes to the queue: a
/// fetch sent with the count at N ran behind sends 1..N and ahead of every
/// later one.
static QUEUE_ORDER: Mutex<u64> = Mutex::new(0);

/// Since the last frame-rate line, of the thread that lands flights
/// ([`relay`]): landings, flights landed, microseconds it waited for the
/// card with nothing to land, microseconds it waited for fetches of card
/// copies' pieces, microseconds of landing in all.
static RELAY_STATS: [std::sync::atomic::AtomicU64; 5] = [const { std::sync::atomic::AtomicU64::new(0) }; 5];

/// Of the line's things in order (a flight: the value signalled on the
/// host's fence behind it; a write or a signal behind a flight: `None`):
/// how many, from the front, can be landed when the card has reached
/// `completed`: everything ahead of the first flight it has not run. A
/// later flight never lands before an earlier one, and what follows a
/// flight (its bottom-of-pipe writes, signals) never before the flight.
fn landable(line: &[Option<u64>], completed: u64) -> usize {
    line.iter().position(|f| f.is_some_and(|f| f > completed)).unwrap_or(line.len())
}

/// For one plan of copies in the card's memory, with [`relay`]: `flags`
/// holds, for every piece of every range of the plan, the copy, the
/// piece's number in the copy and whether the compare flagged it there (or
/// did not run). The pieces to fetch, under whichever range, in order: all
/// of them flagged under any. Two ranges of a plan may hold parts of the same
/// 64 KiB piece; the compare hashes the whole piece for each and stores
/// the hash, so the second of them may find it stored already and not
/// flag it.
fn flagged_pieces(flags: impl Iterator<Item = (usize, u32, bool)>) -> Vec<(usize, u32)> {
    let mut pieces: Vec<_> = flags
        .filter_map(|(copy, piece, wanted)| wanted.then_some((copy, piece)))
        .collect();
    pieces.sort_unstable();
    pieces.dedup();
    pieces
}

/// Indices within one range, in the same order as scanning its pieces.
fn flagged_in_range(
    pieces: &[(usize, u32)],
    copy: usize,
    first: u32,
    count: u32,
) -> impl Iterator<Item = u32> + '_ {
    let start = pieces.partition_point(|&key| key < (copy, first));
    pieces[start..]
        .iter()
        .take_while(move |&&(owner, piece)| owner == copy && piece - first < count)
        .map(move |&(_, piece)| piece - first)
}

#[cfg(test)]
mod relay_tests {
    use super::{flagged_in_range, flagged_pieces, landable};

    /// Two flights and what follows each: nothing of the second lands
    /// before the card has run it, and nothing behind a flight lands
    /// before the flight.
    #[test]
    fn flights_land_in_the_order_they_were_sent() {
        // Flight (fence 5), its write, a signal, flight (fence 6), its write.
        let line = [Some(5), None, None, Some(6), None];
        assert_eq!(landable(&line, 4), 0);
        assert_eq!(landable(&line, 5), 3);
        assert_eq!(landable(&line, 6), 5);
        assert_eq!(landable(&line, u64::MAX), 5);
        // Signals with no flight ahead land at once; behind one they wait.
        assert_eq!(landable(&[None, None], 0), 2);
        assert_eq!(landable(&[None, Some(9), None], 8), 1);
        assert_eq!(landable(&[], 3), 0);
        // After the first three landed, the rest when its turn comes.
        assert_eq!(landable(&line[3..], 5), 0);
        assert_eq!(landable(&line[3..], 6), 2);
    }

    /// A piece flagged under one range of a plan is fetched under every
    /// range that holds a part of it; pieces of other copies are not.
    #[test]
    fn a_piece_flagged_under_one_range_is_fetched_under_all() {
        let (a, b) = (0x1000usize, 0x2000usize);
        // Copy a: piece 3 under two ranges, flagged under the first only;
        // piece 4 not flagged; copy b: piece 3 not flagged, piece 9 flagged.
        let flags = [(a, 3, true), (a, 4, false), (a, 3, false), (b, 3, false), (b, 9, true)];
        let fetch = flagged_pieces(flags.into_iter());
        assert!(fetch.contains(&(a, 3)));
        assert!(!fetch.contains(&(a, 4)));
        assert!(!fetch.contains(&(b, 3)));
        assert!(fetch.contains(&(b, 9)));
        assert_eq!(fetch.len(), 2);
    }

    #[test]
    fn visiting_flagged_pieces_matches_scanning_each_range() {
        let flags = [
            (2, 5, true),
            (1, 34, true),
            (1, 2, true),
            (1, 7, false),
            (1, 34, false),
            (1, 8, true),
            (1, 34, true),
            (1, 6, true),
            (2, u32::MAX, true),
        ];
        let fetch = flagged_pieces(flags.into_iter());
        for (copy, first, count) in [
            (0, 0, 40),
            (1, 0, 40),
            (1, 6, 29),
            (1, 8, 26),
            (1, 34, 0),
            (1, 35, 2),
            (2, 4, 3),
            (2, u32::MAX - 1, 2),
            (3, 0, 40),
        ] {
            let scanned: Vec<_> = (0..count)
                .filter(|&p| fetch.binary_search(&(copy, first + p)).is_ok())
                .collect();
            let selected: Vec<_> = flagged_in_range(&fetch, copy, first, count).collect();
            assert_eq!(selected, scanned);
        }
    }
}

/// The most host lists in the air before the flights are settled
/// (`VICTORY_LATE_WAIT_LISTS`, 256; `=128` is the value before: with the
/// relay it stopped the submitting thread twice a frame for 92-103 ms in
/// all while the card stood idle 64-113 ms a frame, runs 158, 159 and 162;
/// 32 settled the flights three times a
/// frame in run 92 and 64 still 1.3-1.7 times in run 99, with 12,919 tables
/// asked and none refused; watch "tables asked, N refused"): their dynamic
/// descriptor slots, upload
/// memory and scratch chunks are held until then, and a frame's lists held
/// all at once ran the slots out before (run 45).
fn late_most() -> usize {
    static MOST: OnceLock<usize> = OnceLock::new();
    *MOST.get_or_init(|| {
        std::env::var("VICTORY_LATE_WAIT_LISTS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(256).max(1)
    })
}

/// Whether a submission with bottom-of-pipe writes may fly
/// (`VICTORY_LATE_WAIT_WRITES=0`: it is settled as soon as it is sent, so
/// the title is told at once, as before; for telling a title that polls
/// such a value from one that does not).
fn late_writes() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LATE_WAIT_WRITES").is_none_or(|v| v != "0"))
}

/// Per host copy, the ranges a submission's shaders may write, as declared
/// (address, end).
type Declared = (std::sync::Arc<crate::shadow::Shadow>, Vec<(usize, usize)>);

/// What [`Host::submit`] hands back for a submission it sent and did not
/// wait for.
struct Sent {
    /// Its number among the submissions started.
    number: u64,
    card: Option<CardPlan>,
    back: Vec<Declared>,
    /// What was taken off it to be done on the card.
    own: Option<card_steps::Taken>,
    /// Title-memory ranges written when its shader textures are copied back.
    texture_back_ranges: Vec<(usize, usize)>,
    /// Its bottom-of-pipe writes.
    writes: Vec<(usize, crate::queue::Written, usize)>,
}

/// The queue fence behind a submission and, if it was sent ahead, the
/// work to hand to the landing thread while still holding the submit lock.
struct Submission<'a> {
    fence: u64,
    ahead: Option<(std::sync::MutexGuard<'a, u64>, Sent)>,
}

/// Keep each joined part's queue call and start a final call at the buffer
/// comparison, including any timestamp lists that follow it.
fn submission_batches<'a, T>(
    lists: &'a [T],
    firsts: &'a [usize],
    parts: &'a [usize],
    compare_at: Option<usize>,
) -> impl Iterator<Item = &'a [T]> {
    let ends = parts.iter().filter_map(move |&part| firsts.get(part).copied());
    let mut from = 0;
    ends.chain(compare_at).chain([lists.len()]).filter_map(move |to| {
        let batch = &lists[from..to];
        from = to;
        (!batch.is_empty()).then_some(batch)
    })
}

#[cfg(test)]
mod submission_tests {
    use super::submission_batches;

    #[test]
    fn joined_parts_run_before_the_comparison_and_end_markers() {
        let lists = [
            "span start", "timeline start", "first", "prelude", "second", "compare", "timeline end", "span end",
        ];
        let batches: Vec<_> = submission_batches(&lists, &[2, 3], &[1], Some(5)).collect();
        assert_eq!(batches, [&lists[..3], &lists[3..5], &lists[5..]]);
    }

    #[test]
    fn a_submission_without_comparison_keeps_the_end_markers_with_the_last_part() {
        let lists = ["timeline start", "first", "second", "timeline end"];
        let batches: Vec<_> = submission_batches(&lists, &[1, 2], &[1], None).collect();
        assert_eq!(batches, [&lists[..2], &lists[2..]]);
    }
}

/// A submission on the card that has not been waited for ([`late`]): its
/// host lists, taken from the title's (which it may reset meanwhile), and
/// what is to be done once it has run.
struct Flight {
    number: u64,
    /// The value signalled on the host's fence behind it.
    fence: u64,
    lists: Vec<crate::list::Detached>,
    order: Vec<usize>,
    card: Option<CardPlan>,
    back: Vec<Declared>,
    /// The list of ours that ran ahead of them ([`card_steps`]).
    own: Option<card_steps::Taken>,
    texture_back_ranges: Vec<(usize, usize)>,
}

/// One thing in the line of what waits for the card, in order. (A flight
/// is large; the line is short and moves often, so it is not boxed.)
#[allow(clippy::large_enum_variant)]
enum Late {
    Run(Flight),
    /// A bottom-of-pipe write of the flight before it: address, what, bytes.
    Write(usize, crate::queue::Written, usize),
    /// A fence signalled behind the flights before it.
    Signal(crate::fence::FenceRef, u64),
}

struct Line {
    line: Vec<Late>,
    /// The title memory the flights' shaders and texture readbacks may write
    /// (address, end), in order and joined.
    written: Vec<(usize, usize)>,
    /// Host lists in the flights.
    lists: usize,
    /// The value signalled on the host's fence behind the last flight.
    target: u64,
    /// A settle has taken the line and is not done: a signal still goes
    /// behind it.
    settling: bool,
    /// Something was settled away from a queue's own calls: the helper
    /// thread looks at what queues hold.
    poke: bool,
    /// Host lists of settled flights, reset and given back outside the
    /// submission lock ([`land`]: a settle may run under a caller's own
    /// locks).
    landed: Vec<(Vec<crate::list::Detached>, Vec<usize>)>,
    /// With [`relay`]: the bottom-of-pipe writes of the flights being
    /// landed (taken out of `line`, not done yet), for [`late_met`].
    in_hand: Vec<(usize, crate::queue::Written, usize)>,
    /// With [`card_steps`]: what the flights being landed write.
    landing: Vec<card_steps::Writer>,
}

// SAFETY: D3D12 objects (free-threaded) and host lists nothing records
// into; touched under the lock.
unsafe impl Send for Line {}

static LINE: Mutex<Line> = Mutex::new(Line {
    line: Vec::new(),
    written: Vec::new(),
    lists: 0,
    target: 0,
    settling: false,
    poke: false,
    landed: Vec::new(),
    in_hand: Vec::new(),
    landing: Vec::new(),
});
/// Told when a flight joins [`LINE`] or it is poked.
static LINE_MOVED: std::sync::Condvar = std::sync::Condvar::new();

fn line() -> std::sync::MutexGuard<'static, Line> {
    LINE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Why flights were settled ([`LATE_STATS`]' places from 2 on).
const WHY_SYNC: usize = 2;
const WHY_CARD_DONE: usize = 3;
const WHY_VALUE: usize = 4;
const WHY_STALE: usize = 5;
const WHY_READ: usize = 6;
const WHY_RUN: usize = 7;
const WHY_MOST: usize = 8;
const WHY_WRITES: usize = 9;
const WHY_FENCE: usize = 10;

/// Since the last frame-rate line: submissions sent without a wait;
/// settles; settles by reason ([`WHY_SYNC`]..); flights settled;
/// microseconds settles waited for the card; value waits met by a flight's
/// bottom-of-pipe write; fence signals set behind flights.
/// Then: the settles for title memory read, by what read it (a buffer
/// copy, a fill of upload memory, a texture).
static LATE_STATS: [std::sync::atomic::AtomicU64; 18] = [const { std::sync::atomic::AtomicU64::new(0) }; 18];

/// Since the last frame-rate line: the microseconds settles waited for the
/// card, by why they settled (the `WHY_` places).
static LATE_WAITED: [std::sync::atomic::AtomicU64; 11] = [const { std::sync::atomic::AtomicU64::new(0) }; 11];

/// Whether `[from, to)` touches one of `ranges` (in order, joined).
fn touches(ranges: &[(usize, usize)], from: usize, to: usize) -> bool {
    if from >= to {
        return false;
    }
    let at = ranges.partition_point(|r| r.1 <= from);
    ranges.get(at).is_some_and(|r| r.0 < to)
}

/// `ranges` with `more`, in order, joined where they overlap or touch.
fn join_ranges(ranges: &mut Vec<(usize, usize)>, more: impl IntoIterator<Item = (usize, usize)>) {
    ranges.extend(more.into_iter().filter(|r| r.0 < r.1));
    ranges.sort_unstable();
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
    for &(from, to) in ranges.iter() {
        match out.last_mut() {
            Some(last) if from <= last.1 => last.1 = last.1.max(to),
            _ => out.push((from, to)),
        }
    }
    *ranges = out;
}

/// Whether anything is in the air.
fn flying() -> bool {
    if !late() {
        return false;
    }
    let line = line();
    // (With the relay: also while flights are being landed.)
    line.settling || line.line.iter().any(|l| matches!(l, Late::Run(_)))
}

/// With flights in the air ([`late`]): why `runs` cannot be prepared
/// before they are settled, if so. Preparing reads title memory on the
/// CPU (the title's buffer copies, fills of upload memory, textures) and
/// runs steps of ours that may read or write it anywhere; what the
/// flights' shaders write is not there yet.
fn late_blocked(runs: &[&HostList]) -> Option<usize> {
    let (written, copies) = {
        let line = line();
        if line.written.is_empty() {
            return None;
        }
        (line.written.clone(), card_steps::copies(&line))
    };
    blocked_by(&written, runs, true).or_else(|| card_steps::copied_read(&copies, runs).then_some(WHY_READ))
}

/// Whether a small fill of upload memory, or copy of a title buffer, goes on without waiting for
/// flights when the title itself wrote the memory since they were given it
/// (`VICTORY_FILL_TITLE=1`), or only for the second reason below (`=2`); off
/// by default.
///
/// Measured (`=2` against off, A/B by 10 s): with the title's threads
/// submitting (run fills-ab3, standing) mean 88.8 ms a frame against 100.5;
/// with submissions on our thread (`VICTORY_THIN=1`, run fills-ab4, walking)
/// 60-71 against 64-65: about one submission a frame then waits 8-15 ms for
/// landings, "what a copy at its place writes" ([`place::read_by`]), which
/// with the fills blocked the card's steps took instead.
///
/// The title's 174 MB heap is declared written as a whole (a root view has
/// no size), and the title keeps a ring of constants in it: one fill of 480
/// bytes a frame, at a new place each time, waited for every flight's
/// copy-back, some 30 ms of a 130 ms frame (run blame1: 38 of the 40 waits
/// named). The title writes those bytes just before the draw that reads
/// them, which takes the block's guard off ([`crate::watch::written_since_taken`]).
///
/// Or no shader's write went into that page in the last 600 frames
/// ([`crate::watch::landed_lately`]): the first alone left two thirds of
/// the wait, for constants at the end of a 64 KiB page that the title wrote
/// before the page was taken.
///
/// What can go wrong: a constant a flight's shader really writes, in a
/// block that the title (or an earlier flight's copy-back) also wrote, or
/// for the first time, is read as it was before that flight.
fn fill_is_titles(src: usize, len: usize) -> bool {
    const SMALL: usize = 64 << 10;
    let mode = fill_title_mode();
    #[cfg(feature = "diagnostics")]
    let mode = crate::ab::pick("VICTORY_FILL_TITLE").map_or(mode, |on| if on { 2 } else { 0 });
    if mode == 0 || len > SMALL {
        return false;
    }
    !crate::watch::landed_lately(src, len, 600) || mode == 1 && crate::watch::written_since_taken(src, len)
}

fn fill_title_mode() -> u32 {
    static MODE: OnceLock<u32> = OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_FILL_TITLE").ok().as_deref() {
        Some("1") => 1,
        Some("2") => 2,
        _ => 0,
    })
}

/// Diagnostics: the waits for flights, by what waited, why it was not done
/// on the card ([`card_steps`]), its bytes, and the declared write it lies
/// in (start, bytes).
#[cfg(feature = "diagnostics")]
type Blamed = (&'static str, &'static str, usize, usize, usize);
/// How often and how long, and the address the last one read.
#[cfg(feature = "diagnostics")]
static BLAME: Mutex<Option<std::collections::HashMap<Blamed, (u64, u64, usize)>>> = Mutex::new(None);
/// What the wait that comes next is for, and why not on the card.
#[cfg(feature = "diagnostics")]
static BLAMED: Mutex<(Option<(Blamed, usize)>, &'static str)> = Mutex::new((None, ""));

/// Diagnostics: shader writes declared over a megabyte or more: the range
/// (from, to) and what declares it ([`wide_write`]): how often, and its
/// shaders.
#[cfg(feature = "diagnostics")]
static WIDE_WRITES: Mutex<Option<std::collections::HashMap<(usize, usize, usize), (u64, String)>>> = Mutex::new(None);

/// Diagnostics: a draw or dispatch declares that it writes `len` bytes at
/// `address`, through a root view of `pipeline`, a view in a table (0) or a
/// raw address (1).
#[cfg(feature = "diagnostics")]
pub(crate) fn wide_write(address: u64, len: u64, pipeline: usize) {
    if len < 1 << 20 {
        return;
    }
    let mut all = WIDE_WRITES.lock().unwrap_or_else(|e| e.into_inner());
    let e = all
        .get_or_insert_with(Default::default)
        .entry((address as usize, (address + len) as usize, pipeline))
        .or_insert_with(|| {
            let shaders = match pipeline {
                0 => "a view in a table".into(),
                1 => "a raw address".into(),
                p => crate::pipeline::host_pipeline(p as crate::object::This).map_or_else(String::new, |p| {
                    p.stages.iter().map(|s| s.shader.hash.to_string()).collect::<Vec<_>>().join("+")
                }),
            };
            (0, shaders)
        });
    e.0 += 1;
}

#[cfg(feature = "diagnostics")]
fn blame(kind: &'static str, written: &[(usize, usize)], at: usize, len: usize) {
    let i = written.partition_point(|r| r.1 <= at);
    let (from, to) = written.get(i).copied().unwrap_or((0, 0));
    let mut blamed = BLAMED.lock().unwrap_or_else(|e| e.into_inner());
    let why = blamed.1;
    blamed.0 = Some(((kind, why, len, from, to - from), at));
}

/// Why [`card_steps`] left the submission to the wait that comes next.
#[cfg(feature = "diagnostics")]
fn refused(why: &'static str) {
    *BLAMED.lock().unwrap_or_else(|e| e.into_inner()) = (None, why);
}

/// The wait that comes next is for these ranges ([`stale_under_flights`]).
#[cfg(feature = "diagnostics")]
fn blame_stale(stale: &[(std::sync::Arc<crate::shadow::Shadow>, usize, usize)]) {
    if let Some((shadow, from, _)) = stale.first() {
        let (start, len) = shadow.span();
        let key = ("title memory changed under shader writes", crate::shadow::stale_why(), 0, start as usize, len as usize);
        BLAMED.lock().unwrap_or_else(|e| e.into_inner()).0 = Some((key, *from));
    }
}

/// The wait for what [`blame`] last named, or a wait of another reason than
/// title memory read, took `us`.
#[cfg(feature = "diagnostics")]
fn blamed(why: usize, us: u64) {
    let (named, refused) = std::mem::take(&mut *BLAMED.lock().unwrap_or_else(|e| e.into_inner()));
    let (key, at) = match (why, named) {
        (WHY_READ | WHY_STALE, Some(named)) => named,
        (WHY_READ, None) => (("a host copy of what a copy done on the card wrote", refused, 0, 0, 0), 0),
        _ => {
            let kind = match why {
                WHY_SYNC => "a synchronous submission, a present or the sweep",
                WHY_VALUE => "a memory value",
                WHY_STALE => "title memory changed under shader writes",
                WHY_RUN => "a step of ours",
                WHY_MOST => "lists in the air",
                WHY_FENCE => "a blocking fence wait",
                _ => "another reason",
            };
            ((kind, "", 0, 0, 0), 0)
        }
    };
    let mut all = BLAME.lock().unwrap_or_else(|e| e.into_inner());
    let e = all.get_or_insert_with(Default::default).entry(key).or_default();
    *e = (e.0 + 1, e.1 + us, at);
}

/// For the frame-rate line: the five that waited longest.
#[cfg(feature = "diagnostics")]
fn log_blame(frames: f64) {
    let all = BLAME.lock().unwrap_or_else(|e| e.into_inner()).take().unwrap_or_default();
    let wide = WIDE_WRITES.lock().unwrap_or_else(|e| e.into_inner()).take().unwrap_or_default();
    let mut all: Vec<_> = all.into_iter().collect();
    all.sort_unstable_by_key(|(_, (_, us, _))| std::cmp::Reverse(*us));
    for ((kind, why, len, from, size), (n, us, at)) in all.into_iter().take(5) {
        // The passes that declare it written, the ones that do so most.
        let mut passes: std::collections::BTreeMap<&str, u64> = std::collections::BTreeMap::new();
        for ((a, b, _), (n, shaders)) in &wide {
            if *a < at + len && at < *b {
                *passes.entry(shaders).or_default() += n;
            }
        }
        let mut passes: Vec<(&str, u64)> = passes.into_iter().collect();
        passes.sort_unstable_by_key(|p| std::cmp::Reverse(p.1));
        let passes: Vec<String> =
            passes.into_iter().take(5).map(|(shaders, n)| format!("{shaders} {:.1}x", n as f64 / frames)).collect();
        tracing::info!(
            "waited for flights, per frame: {:.1} ms in {:.2} waits for {kind} +{len:#x} (last at {at:#x}), in shader writes declared \
             over {from:#x}+{size:#x} (declared per frame over a megabyte or more there, by shader: {}){}{why}",
            us as f64 / 1e3 / frames,
            n as f64 / frames,
            passes.join(", "),
            match (why.is_empty(), kind) {
                (true, _) => "",
                (_, "title memory changed under shader writes") => "; not put in under them: ",
                _ => "; not done on the card: ",
            },
        );
    }
}

/// Whether flights that write `written` stop a read of `len` bytes at `src`.
fn read_blocked(written: &[(usize, usize)], src: usize, len: usize) -> bool {
    touches(written, src, src + len) && !fill_is_titles(src, len)
}

/// Whether they stop a title buffer copy, and if so whether by its source.
fn copy_blocked(written: &[(usize, usize)], dst: usize, src: usize, len: usize) -> Option<bool> {
    let read = read_blocked(written, src, len);
    (read || touches(written, dst, dst + len)).then_some(read)
}

/// [`late_blocked`] against `written`, the title memory flights' shaders
/// may write (in order, joined); `count` whether the kind of read is
/// counted for the frame line.
fn blocked_by(written: &[(usize, usize)], runs: &[&HostList], count: bool) -> Option<usize> {
    if written.is_empty() {
        return None;
    }
    // Whether a copy at its place waits for flights if a step reads it.
    let mut placed = false;
    for list in runs {
        for step in &list.deferred().copies {
            match step {
                &Step::Copy(dst, src, len) => {
                    if let Some(_read) = copy_blocked(written, dst, src, len) {
                        if count {
                            LATE_STATS[15].fetch_add(1, Ordering::Relaxed);
                            #[cfg(feature = "diagnostics")]
                            if _read {
                                blame("a buffer copy's source", written, src, len);
                            } else {
                                blame("a buffer copy's destination", written, dst, len);
                            }
                        }
                        return Some(WHY_READ);
                    }
                }
                &Step::Fill(_, src, len, _) => {
                    if read_blocked(written, src, len) {
                        if count {
                            LATE_STATS[16].fetch_add(1, Ordering::Relaxed);
                            #[cfg(feature = "diagnostics")]
                            blame("a fill of upload memory", written, src, len);
                        }
                        return Some(WHY_READ);
                    }
                }
                Step::Texture(staging) => {
                    let (at, len) = staging.memory();
                    if touches(written, at, at + len) {
                        if count {
                            LATE_STATS[17].fetch_add(1, Ordering::Relaxed);
                            #[cfg(feature = "diagnostics")]
                            blame("a texture", written, at, len);
                        }
                        return Some(WHY_READ);
                    }
                }
                &Step::Plane((at, len), _) => {
                    if touches(written, at, at.saturating_add(len)) {
                        if count {
                            LATE_STATS[17].fetch_add(1, Ordering::Relaxed);
                            #[cfg(feature = "diagnostics")]
                            blame("a movie plane", written, at, len);
                        }
                        return Some(WHY_READ);
                    }
                }
                Step::RawCopy { .. } => return Some(WHY_RUN),
                Step::Placed(p) => match p.waits(written) {
                    place::Waits::No => {}
                    place::Waits::ForReaders => placed = true,
                    place::Waits::Yes => {
                        if count {
                            LATE_STATS[15].fetch_add(1, Ordering::Relaxed);
                            #[cfg(feature = "diagnostics")]
                            blame("a buffer copy at its place", written, p.src, p.len);
                        }
                        return Some(WHY_READ);
                    }
                },
                Step::Refresh(..) | Step::GpuWrite(..) | Step::Host(..) => {}
            }
        }
    }
    if placed && place::read_by(runs, written) {
        #[cfg(feature = "diagnostics")]
        if count {
            blame("what a copy at its place writes", &[], 0, 0);
        }
        Some(WHY_READ)
    } else {
        None
    }
}

/// Since the last frame-rate line, ns of [`stale_under_flights`] looking
/// through the lists' refreshes for copies shaders may be writing.
static UNDER_SCAN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// With flights in the air, after the submission's write-watch asking:
/// whether `runs` refresh a range of a host copy that a flight's shaders
/// may write and that the title wrote since it was last copied. The
/// refresh leaves ranges shaders are writing alone (`Shadow::update`), so
/// the title's bytes there would reach the card a submission late.
fn stale_under_flights(runs: &[&HostList]) -> Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)> {
    use std::sync::Arc;
    // Per copy, the span of what is refreshed (a flush brings that much
    // up to date at most).
    let mut spans: Vec<(Arc<crate::shadow::Shadow>, usize, usize)> = Vec::new();
    // Where each copy is in it, if shaders may be writing it (none begins to
    // while this runs): a frame has 20,000 refreshes of some 4,500 copies.
    let mut places: ByCopy<Option<usize>> = ByCopy::default();
    let scanning = std::time::Instant::now();
    for list in runs {
        for step in &list.deferred().copies {
            if let Step::Refresh(shadow, address, len) = step {
                let place = places.entry(Arc::as_ptr(shadow) as usize).or_insert_with(|| {
                    shadow.writes_pending().then(|| {
                        spans.push((shadow.clone(), *address, *address + *len));
                        spans.len() - 1
                    })
                });
                if let Some(i) = *place {
                    let s = &mut spans[i];
                    (s.1, s.2) = (s.1.min(*address), s.2.max(*address + *len));
                }
            }
        }
    }
    if late_exact() {
        // The title's bytes are put in beside the flights' writes; a
        // settle is still due where that could not be done. At most this
        // many changed runs go up to copies in the card's memory, each a
        // copy of its own there. With [`card_steps`] the pages of the
        // title's ring of constants come here too, which were waited for
        // before: hundreds of short runs each, at about 1.5 us a run.
        let mut runs_left = if card_steps::stage() > 0 { 4096 } else { 512 };
        UNDER_SCAN.fetch_add(scanning.elapsed().as_nanos() as u64, Ordering::Relaxed);
        spans.retain(|(shadow, from, to)| shadow.merge_under_writes(*from, *to - *from, &mut runs_left));
        return spans;
    }
    spans.retain(|(shadow, from, to)| shadow.stale_under_writes(*from, *to - *from));
    spans
}

/// How many parts of submissions, and lists, the line's thread gives to the
/// host as one (`VICTORY_JOIN_PARTS=<n>`; off by default, 1; `VICTORY_JOIN_LISTS`,
/// 48).
///
/// The idea is fxceless/victory-fork's, commit 39731795 (GPL-3.0-or-later, as
/// this): the title's lists wait for memory values, which cuts a frame into
/// some 29 parts, each prepared on its own: the write watch asked, the heaps'
/// records, the title's bytes under the flights' writes, a list that sends
/// what changed and one that compares. (Run submit-1, the dorm, 12 of 28 parts
/// joined, eight at most: the thread works 51.4 ms a frame instead of 54.8, the
/// frame is 0.6 to 3 ms shorter.) What differs here: the parts are taken
/// off one queue's entries of the line (`crate::queue`), still go to the queue
/// in calls of their own, and everything they refresh is brought up to date in
/// front of the first ([`refresh_front`], without which a later part's title
/// bytes under an earlier part's declared writes would go missing).
pub(crate) fn joins() -> (usize, usize) {
    if ahead_of_writes() { joins_asked() } else { (1, 1) }
}

fn joins_asked() -> (usize, usize) {
    static MOST: OnceLock<(usize, usize)> = OnceLock::new();
    let of = |name: &str| std::env::var(name).ok().and_then(|v| v.trim().parse::<usize>().ok());
    let (parts, lists) = *MOST.get_or_init(|| (of("VICTORY_JOIN_PARTS").unwrap_or(1).max(1), of("VICTORY_JOIN_LISTS").unwrap_or(48).max(1)));
    #[cfg(feature = "diagnostics")]
    if let Some(on) = crate::ab::pick("VICTORY_JOIN_PARTS") {
        return (if on { 8 } else { 1 }, lists);
    }
    (parts, lists)
}

#[derive(Clone, Copy)]
pub(crate) enum JoinStop {
    QueueBoundary,
    FoldedWait,
    Disabled,
    BoundaryJob,
    ValueWait,
    FenceWait,
    PartLimit,
    ListLimit,
    CpuCopy,
    EarlierWrite,
}

/// Why the part `next` cannot be prepared together with `ahead`, which goes
/// before it and has not been sent: nothing done on the CPU for it reads what
/// those give their shaders, write back or write when they are done, and it
/// changes no title memory that they read ([`Step::Copy`], [`Step::RawCopy`]).
pub(crate) fn join_blocked<'a>(ahead: impl Iterator<Item = &'a HostList>, next: &[&HostList]) -> Option<JoinStop> {
    if next.iter().any(|l| l.deferred().copies.iter().any(|s| matches!(s, Step::Copy(..) | Step::RawCopy { .. }))) {
        return Some(JoinStop::CpuCopy);
    }
    let ahead: Vec<&HostList> = ahead.collect();
    let mut written: Vec<(usize, usize)> = declared_writes(&ahead).into_iter().flat_map(|(_, ranges)| ranges).collect();
    for list in &ahead {
        let deferred = list.deferred();
        written.extend(deferred.writes.iter().map(|&(at, _, bytes)| (at, at + bytes)));
        written.extend(deferred.readback_ranges());
    }
    let mut joined = Vec::new();
    join_ranges(&mut joined, written);
    blocked_by(&joined, next, false).map(|_| JoinStop::EarlierWrite)
}

/// Whether a bottom-of-pipe write of `list` meets `wait`.
pub(crate) fn will_meet(list: &HostList, wait: &crate::list::ValueWait) -> bool {
    let (address, bytes) = wait.place();
    // The last write there says.
    let deferred = list.deferred();
    let last = deferred.writes.iter().rev().find(|(at, _, n)| *at < address + bytes && address < *at + *n);
    matches!(last, Some((at, crate::queue::Written::Value(v), n)) if *at == address && *n >= bytes && wait.holds_for(*v))
}

/// Lists and the order they run in.
type Run<T> = (Vec<T>, Vec<usize>);

/// `order`'s first `cut` places and the rest, each with the lists they name;
/// as it was if a list is named on both sides.
fn split<T>(lists: Vec<T>, order: Vec<usize>, cut: usize) -> Result<[Run<T>; 2], Run<T>> {
    let (first, second) = order.split_at(cut);
    if first.is_empty() || second.is_empty() || first.iter().any(|i| second.contains(i)) {
        return Err((lists, order));
    }
    let behind: Vec<bool> = (0..lists.len()).map(|i| second.contains(&i)).collect();
    let place = |i: usize| behind[..i].iter().filter(|&&b| b == behind[i]).count();
    let orders = [first, second].map(|half| half.iter().map(|&i| place(i)).collect::<Vec<_>>());
    let mut halves: [Vec<T>; 2] = [Vec::new(), Vec::new()];
    for (list, behind) in lists.into_iter().zip(&behind) {
        halves[usize::from(*behind)].push(list);
    }
    let [a, b] = halves;
    let [x, y] = orders;
    Ok([(a, x), (b, y)])
}

/// A fence signal on a queue ([`late`]): goes behind the flights in the
/// air, if there are any, and is set when they are settled. Whether it
/// did; if not the caller sets the fence now.
pub(crate) fn late_signal(fence: crate::object::This, value: u64) -> bool {
    if !late() {
        return false;
    }
    let mut line = line();
    if line.line.is_empty() && !line.settling {
        return false;
    }
    line.line.push(Late::Signal(crate::fence::FenceRef::hold(fence), value));
    LATE_STATS[14].fetch_add(1, Ordering::Relaxed);
    drop(line);
    // A signal alone (behind a settle under way) has no flight to wake
    // the helper thread for it.
    LINE_MOVED.notify_one();
    RELAY_MOVED.notify_one();
    true
}

/// A queue stands at `wait`, which memory does not meet now ([`late`]):
/// whether it is met once the flights in the air have run. A flight's
/// bottom-of-pipe write of a value that meets it says yes without a wait
/// for the card: the part behind the wait goes to the card behind that
/// flight, which is the order the wait asks for. If the flights may write
/// the value in a way not known here (shaders, a timestamp, another
/// width), they are settled and memory is asked again.
pub(crate) fn late_met(wait: &crate::list::ValueWait) -> bool {
    if let Some(met) = late_met_known(wait) {
        return met;
    }
    settle_now(WHY_VALUE);
    wait.met()
}

/// [`late_met`] without a wait for the card; `None`: it has to be waited for.
pub(crate) fn late_met_known(wait: &crate::list::ValueWait) -> Option<bool> {
    let (address, bytes) = wait.place();
    {
        let line = line();
        if line.line.is_empty() && line.in_hand.is_empty() {
            return Some(false);
        }
        let mut unsure = touches(&line.written, address, address + bytes);
        // (With the relay: first the writes of flights being landed.)
        let writes = line.in_hand.iter().map(|(at, what, n)| (at, what, n)).chain(line.line.iter().filter_map(|item| match item {
            Late::Write(at, what, n) => Some((at, what, n)),
            _ => None,
        }));
        for (at, what, n) in writes {
            if *at >= address + bytes || address >= *at + *n {
                continue;
            }
            match what {
                crate::queue::Written::Value(v) if *at == address && *n >= bytes => {
                    if wait.holds_for(*v) {
                        LATE_STATS[13].fetch_add(1, Ordering::Relaxed);
                        return Some(true);
                    }
                }
                _ => unsure = true,
            }
        }
        if !unsure {
            return Some(false);
        }
    }
    None
}

/// Settles the flights in the air, if any ([`Host::settle`]), for reason
/// `why` (one of the `WHY_` places). Not from inside a submission.
pub(crate) fn settle_now(why: usize) {
    if !late() {
        return;
    }
    let Some(host) = host() else {
        return;
    };
    let mut submitted = host.submitted.lock().unwrap_or_else(|e| e.into_inner());
    host.settle(&mut submitted, why);
}

/// [`settle_now`] at a present, and for a fence the title blocks on.
pub(crate) fn settle_for_present() {
    settle_now(WHY_SYNC);
}
pub(crate) fn settle_for_fence() {
    settle_now(WHY_FENCE);
}

/// Diagnostics for landed-list recycling and dynamic descriptor slots waiting
/// for retirement: `[0]` µs spent in `land()` on the submitting thread, `[1]`
/// µs spent in `land()` on `victory-late-wait`, `[2]` peak `line.landed.len()`
/// observed, `[3]` peak unretired dynamic descriptor slots observed in
/// `line.landed` before `past_lag`, `[4]` peak dynamic descriptor slots held
/// by a batch of settled flights right before `Detached::release_tables`.
#[cfg(feature = "diagnostics")]
static LAND_RETIRE_STATS: [std::sync::atomic::AtomicU64; 5] = [const { std::sync::atomic::AtomicU64::new(0) }; 5];

/// Resets the host lists of settled flights and keeps them for reuse, as
/// a queue does for the lists it ran (`crate::list::recycle`), [`bury_lag`]
/// flights after they landed. `settle` already releases each flight's dynamic
/// descriptor slots (`Detached::release_tables`) before pushing it to
/// `line.landed`; [`past_lag`] still calls `crate::list::ran` as an idempotent
/// backstop.
fn land() {
    // One at a time, so that the flights kept go back in front of the ones
    // that landed meanwhile. A thread that finds another landing leaves it
    // to that one.
    static LANDING: Mutex<()> = Mutex::new(());
    let old = {
        let _landing = match LANDING.try_lock() {
            Ok(guard) => guard,
            Err(std::sync::TryLockError::Poisoned(e)) => e.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => return,
        };
        // Taken out of the line: giving slots back takes the heap's lock,
        // which recording threads hold while they search.
        let mut landed = std::mem::take(&mut line().landed);
        #[cfg(feature = "diagnostics")]
        {
            LAND_RETIRE_STATS[2].fetch_max(landed.len() as u64, Ordering::Relaxed);
            let pending: u64 = landed
                .iter()
                .flat_map(|(lists, _)| lists.iter())
                .map(crate::list::Detached::pending_dynamic_slots)
                .sum();
            LAND_RETIRE_STATS[3].fetch_max(pending, Ordering::Relaxed);
        }
        let old = past_lag(&mut landed, bury_lag() as usize, |(lists, _)| crate::list::ran(lists));
        let mut line = line();
        landed.append(&mut line.landed);
        line.landed = landed;
        old
    };
    for (lists, order) in old {
        crate::list::recycle(lists, &order);
    }
}

/// Of `landed`, the lists of flights that landed, oldest first: `ran` on
/// each (the card is done with them; the ones kept from an earlier call are
/// given to it again, and it must then do nothing), and the ones past the
/// newest `lag` taken out, to be reset.
fn past_lag<T>(landed: &mut Vec<T>, lag: usize, ran: impl FnMut(&mut T)) -> Vec<T> {
    landed.iter_mut().for_each(ran);
    let n = landed.len().saturating_sub(lag);
    landed.drain(..n).collect()
}

#[cfg(test)]
mod landed_tests {
    use super::past_lag;
    use crate::descriptor::host::HostHeap;

    /// Every flight that landed is told the card ran it, the ones kept from
    /// before again; the newest `lag` stay in the order they landed and the
    /// rest go to be reset, oldest first.
    #[test]
    fn landed_flights_are_told_at_once_and_reset_past_the_lag() {
        let mut landed: Vec<(u32, u32)> = (0..5).map(|flight| (flight, 0)).collect();
        let old = past_lag(&mut landed, 3, |(_, told)| *told += 1);
        assert_eq!(old, [(0, 1), (1, 1)]);
        assert_eq!(landed, [(2, 1), (3, 1), (4, 1)]);
        landed.push((5, 0));
        let old = past_lag(&mut landed, 3, |(_, told)| *told += 1);
        assert_eq!(old, [(2, 2)]);
        assert_eq!(landed, [(3, 2), (4, 2), (5, 1)]);
        // Fewer than the lag: all kept.
        assert!(past_lag(&mut landed, 64, |_| {}).is_empty());
        assert_eq!(landed.len(), 3);
    }

    /// The dynamic descriptor slots of lists the card has run are taken
    /// again by new tables while the lists themselves are still kept, and
    /// runs given back once are not given back again.
    #[test]
    fn slots_of_landed_lists_are_taken_again_before_the_lists_are_reset() {
        const RUN: u32 = 4096;
        let Some(heap) = HostHeap::new(false, 4) else {
            return; // no host GPU
        };
        // Three flights take every dynamic slot between them.
        let mut flights: Vec<Vec<u32>> = vec![Vec::new(); 3];
        let mut taken = 0;
        while let Some(first) = heap.dynamic(RUN) {
            flights[taken % 3].push(first);
            taken += 1;
        }
        assert!(taken >= 3);
        assert_eq!(heap.dynamic(RUN), None);
        // They land; the lag keeps all three. Their lists give their runs
        // back as `crate::draw::State::tables_done` does: once.
        let mut ran = |runs: &mut Vec<u32>| {
            for first in runs.drain(..) {
                heap.release(heap.id(), first, RUN);
            }
        };
        assert!(past_lag(&mut flights, 64, &mut ran).is_empty());
        assert_eq!(flights.len(), 3);
        // The next frame's tables find slots: all of them.
        let again: Vec<u32> = std::iter::from_fn(|| heap.dynamic(RUN)).collect();
        assert_eq!(again.len(), taken);
        // Landing again changes nothing: what the new tables took stays taken.
        assert!(past_lag(&mut flights, 64, &mut ran).is_empty());
        assert_eq!(heap.dynamic(RUN), None);
        for first in again {
            heap.release(heap.id(), first, RUN);
        }
    }
}

/// Whether `victory-late-wait` is alive and serving `line.poke` requests so
/// `send_late` can offload `land()` off the submitting thread.
static LATE_HELPER_LIVE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The thread that settles the flights once the card has run them all
/// ([`late`]): without it a bottom-of-pipe write would reach the title only
/// at the next call that settles, and a title thread that polls such a
/// value while the submitting thread waits for that thread would never
/// see it.
fn late_helper() {
    crate::cores::keep_on_fast_cores("victory-late-wait");
    let Some(host) = host() else {
        return;
    };
    // SAFETY: a plain event of our own.
    let Ok(event) = (unsafe { CreateEventW(None, false, false, None) }) else {
        tracing::warn!("no event for the thread that settles submissions sent ahead");
        return;
    };
    LATE_HELPER_LIVE.store(true, Ordering::Release);
    loop {
        let target = {
            let mut line = line();
            loop {
                if line.poke {
                    line.poke = false;
                    break None;
                }
                // (With the relay the flights are another thread's.)
                if !relay() && !line.line.is_empty() && !line.settling {
                    break Some(line.target);
                }
                line = LINE_MOVED.wait(line).unwrap_or_else(|e| e.into_inner());
            }
        };
        let Some(target) = target else {
            let mut helping = crate::timeline::span("helper: landed lists reset and given back");
            #[cfg(feature = "diagnostics")]
            let started = std::time::Instant::now();
            land();
            #[cfg(feature = "diagnostics")]
            LAND_RETIRE_STATS[1].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
            helping.next("helper: looking at what queues hold");
            crate::queue::release_held(false);
            continue;
        };
        // The card first, without the submission lock: the submitting
        // thread prepares and sends meanwhile.
        // SAFETY: our fence and event.
        unsafe {
            if host.fence.SetEventOnCompletion(target, event).is_ok() {
                #[cfg(feature = "diagnostics")]
                let _card = crate::timeline::card_wait(Some(target));
                WaitForSingleObject(event, INFINITE);
            }
        }
        {
            let mut submitted = host.submitted.lock().unwrap_or_else(|e| e.into_inner());
            // Something newer went up meanwhile: the card is waited for
            // again, the submitting thread is not held up for it.
            if line().target != target {
                continue;
            }
            host.settle(&mut submitted, WHY_CARD_DONE);
        }
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        land();
        #[cfg(feature = "diagnostics")]
        LAND_RETIRE_STATS[1].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
        crate::queue::release_held(false);
    }
}

/// Starts the threads of [`late`], once: the one that settles flights when
/// the card has run them and looks at what queues hold, and with [`relay`]
/// the one that lands flights.
fn late_threads() {
    static HELPER: std::sync::Once = std::sync::Once::new();
    HELPER.call_once(|| {
        if let Err(e) = std::thread::Builder::new().name("victory-late-wait".into()).spawn(late_helper) {
            tracing::warn!(error = %e, "no thread to settle submissions sent ahead: they are settled by the calls that follow");
        }
        tracing::info!(
            most = late_most(),
            writes = late_writes(),
            relay_mode = relay_mode(),
            "submissions are sent to the card without a wait for each (VICTORY_LATE_WAIT=1)"
        );
        if relay() {
            // Without it nothing would ever land.
            if let Err(e) = std::thread::Builder::new().name("victory-relay".into()).spawn(relay_worker) {
                fail(format_args!("no relay thread to land submissions sent ahead: {e}"));
            }
            tracing::info!(
                exact = relay_exact(),
                "what shaders wrote is copied back by a thread of its own, beside the submitting thread (relay)"
            );
        }
    });
}

/// With [`relay`]: waits until `done` says so of the line, looked at each
/// time flights have landed. Counted as a settle for `why`, and its time
/// as waiting for the card. The caller may hold the submission lock: the
/// thread that lands flights never takes it.
fn relay_wait(why: usize, done: impl Fn(&Line) -> bool) {
    let mut line = line();
    if done(&line) {
        return;
    }
    LATE_STATS[1].fetch_add(1, Ordering::Relaxed);
    LATE_STATS[why].fetch_add(1, Ordering::Relaxed);
    let was_working = working(false);
    let waiting = std::time::Instant::now();
    let _waiting = crate::timeline::span(match why {
        WHY_SYNC => "waiting for landings: everything (a present, the sweep)",
        WHY_VALUE => "waiting for landings: a memory value",
        WHY_STALE => "waiting for landings: title memory changed under shader writes",
        WHY_READ => "waiting for landings: title memory read",
        WHY_MOST => "waiting for landings: lists in the air",
        WHY_FENCE => "waiting for landings: a blocking fence wait",
        _ => "waiting for landings: another reason",
    });
    while !done(&line) {
        line = LINE_LANDED.wait(line).unwrap_or_else(|e| e.into_inner());
    }
    drop(line);
    stood(waiting);
    working(was_working);
    let us = waiting.elapsed().as_micros() as u64;
    LATE_STATS[12].fetch_add(us, Ordering::Relaxed);
    LATE_WAITED[why.min(10)].fetch_add(us, Ordering::Relaxed);
    SUBMISSION_US[1].fetch_add(us, Ordering::Relaxed);
    #[cfg(feature = "diagnostics")]
    blamed(why, us);
}

/// What the thread that lands flights fetches card copies' pieces with
/// ([`relay`]): a list, a fence and an event of its own (the host's are
/// behind the submission lock), and the memory the pieces come down to.
struct RelayFetch {
    list: Prelude,
    /// With [`relay_copy`]: the card's copy queue, which `list` is for.
    queue: Option<ID3D12CommandQueue>,
    fence: ID3D12Fence,
    event: HANDLE,
    /// The last value signalled on `fence`.
    sent: u64,
    /// Buffer, mapping, bytes.
    buffer: Option<(ID3D12Resource, usize, u64)>,
    /// Pieces caught while a later flight was writing them.
    suspects: torn::Suspects,
    /// What earlier fetches brought of flights not landed yet.
    had: crate::land::Had,
}

/// The thread that lands flights ([`relay`]): as soon as the card has run
/// the first flights of the line it takes them, with what follows them,
/// and does what they left to do ([`Host::relay_land`]); else it waits for
/// the card, or for a flight. It takes neither the submission lock nor
/// anything a caller of [`Host::settle`] may hold: resetting the landed
/// lists and looking at what queues hold are left to [`late_helper`].
fn relay_worker() {
    crate::cores::keep_on_fast_cores("victory-relay");
    let Some(host) = host() else {
        return;
    };
    // SAFETY: a plain event of our own.
    let Ok(event) = (unsafe { CreateEventW(None, false, false, None) }) else {
        fail(format_args!("no event for the relay thread that lands submissions sent ahead"));
    };
    let mut fetch: Option<RelayFetch> = None;
    let mut texture_scratch = crate::texback::Scratch::default();
    loop {
        let next = {
            let mut line = line();
            loop {
                if line.line.is_empty() {
                    line = RELAY_MOVED.wait(line).unwrap_or_else(|e| e.into_inner());
                    continue;
                }
                let fences: Vec<Option<u64>> = line
                    .line
                    .iter()
                    .map(|l| match l {
                        Late::Run(flight) => Some(flight.fence),
                        _ => None,
                    })
                    .collect();
                // SAFETY: a plain query.
                let completed = unsafe { host.fence.GetCompletedValue() };
                if texture_readback_idle_limit() != 0 && completed == u64::MAX {
                    drop(line);
                    fail(format_args!("texture readback pool cannot retire after device removal"));
                }
                let n = landable(&fences, completed);
                if n == 0 {
                    // The first thing is a flight the card has not run.
                    break Err(fences[0].unwrap_or(0));
                }
                line.settling = true;
                let items: Vec<Late> = line.line.drain(..n).collect();
                line.in_hand = items
                    .iter()
                    .filter_map(|l| match l {
                        Late::Write(at, what, bytes) => Some((*at, *what, *bytes)),
                        _ => None,
                    })
                    .collect();
                if card_steps::stage() > 0 {
                    line.landing = card_steps::writers(&items);
                }
                break Ok(items);
            }
        };
        match next {
            Ok(items) => host.relay_land(items, &mut fetch, &mut texture_scratch),
            Err(target) => {
                let waiting = std::time::Instant::now();
                #[cfg(feature = "diagnostics")]
                let _card = crate::timeline::card_wait(Some(target));
                // SAFETY: the host's fence and this thread's event.
                let waited = unsafe {
                    host.fence.SetEventOnCompletion(target, event).map(|()| {
                        WaitForSingleObject(event, INFINITE);
                    })
                };
                if waited.is_err() {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                RELAY_STATS[2].fetch_add(waiting.elapsed().as_micros() as u64, Ordering::Relaxed);
            }
        }
    }
}

/// [`relay`]: what shaders wrote to the declared ranges `back` of copies
/// in the PC's memory goes into title memory. As [`hand_back`], each byte
/// once, the pieces that differ found on several threads; but the merge is
/// `Shadow::relay_merge`, which does not write the copy, and the ranges
/// are not done with here (each flight's are, afterwards). In two steps:
/// what is to be looked through and what of it differs ([`RelayBack::look`],
/// which needs nothing that the fetch from the card's memory brings), and
/// the merge.
struct RelayBack {
    work: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)>,
    found: Option<Vec<Vec<(usize, usize)>>>,
}

fn relay_hand_back(back: &[Declared]) -> RelayBack {
    let mut work: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)> = Vec::new();
    for (shadow, declared) in back {
        if shadow.carded() {
            continue;
        }
        let mut spans: Vec<(usize, usize)> = Vec::new();
        join_ranges(&mut spans, declared.iter().copied());
        for &(from, to) in &spans {
            shadow.relay_compared(from, to - from);
        }
        work.extend(spans.into_iter().map(|(from, to)| (shadow.clone(), from, to)));
    }
    RelayBack { work, found: None }
}

impl RelayBack {
    /// Looks the ranges through, once; its time counts as `stat`.
    fn look(&mut self, stat: usize) {
        if self.work.is_empty() || self.found.is_some() {
            return;
        }
        let scanning = std::time::Instant::now();
        // Looked through without a lock: a piece that changes meanwhile (a
        // later flight writing, the title's bytes going in) is found now or
        // by that flight's landing, and the merge looks again under the
        // lock.
        self.found = Some(crate::shadow::find_all(&self.work));
        crate::land::spent(stat, scanning);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::BackScanUs, scanning.elapsed().as_micros() as u64);
    }

    /// What differs goes into title memory.
    fn merge(mut self) {
        use crate::land;
        self.look(land::stat::SCAN);
        let Some(found) = self.found else {
            return;
        };
        let merging = std::time::Instant::now();
        let mut written: Vec<u64> = Vec::new();
        if land::on() {
            // Pieces in the same block of title memory in one job, in order.
            let pieces: Vec<(usize, usize, usize)> =
                found.iter().enumerate().flat_map(|(i, found)| found.iter().map(move |&(at, n)| (i, at, n))).collect();
            let start = |i: usize| self.work[i].0.span().0 as usize;
            let mut order: Vec<usize> = (0..pieces.len()).collect();
            order.sort_by_key(|&p| (start(pieces[p].0) + pieces[p].1) / crate::land::BLOCK);
            let cut: Vec<(usize, usize)> = order.iter().map(|&p| (start(pieces[p].0) + pieces[p].1, pieces[p].2)).collect();
            let jobs = land::jobs(&cut);
            let put: Vec<std::sync::atomic::AtomicU64> = self.work.iter().map(|_| Default::default()).collect();
            land::each(jobs.len(), &|j| {
                let working = std::time::Instant::now();
                let mut notes = land::Notes::default();
                for &p in &order[jobs[j].clone()] {
                    let (i, at, n) = pieces[p];
                    put[i].fetch_add(self.work[i].0.land_back(at, n, &mut notes), Ordering::Relaxed);
                }
                notes.flush();
                land::spent(land::stat::BACK_ALL, working);
            });
            land::count(land::stat::JOBS, jobs.len() as u64);
            written.extend(put.iter().map(|p| p.load(Ordering::Relaxed)));
        } else {
            let (mut now, mut title) = (Vec::new(), Vec::new());
            for ((shadow, ..), found) in self.work.iter().zip(found) {
                let merge = |(at, n)| shadow.relay_merge(at, n, &mut now, &mut title) as u64;
                written.push(found.into_iter().map(merge).sum());
            }
        }
        land::spent(land::stat::BACK, merging);
        for ((_, from, to), written) in self.work.iter().zip(written) {
            COPY_BACK[1].fetch_add(written, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            crate::copystat::compared(*from, to - from, written as usize);
            let _ = (from, to);
        }
    }
}

/// For the frame-rate line, with [`late`]: what the line of flights did
/// per frame since the last one.
fn late_line(frames: f64) {
    if !late() {
        return;
    }
    let s: Vec<f64> = LATE_STATS.iter().map(|s| s.swap(0, Ordering::Relaxed) as f64 / frames).collect();
    tracing::info!(
        "submissions sent ahead per frame: {:.1} in {:.1} settles ({:.1} flights each time), {:.1} ms waiting for the card in them; \
         settled for: a synchronous submission or present {:.1}, the card had run them {:.1}, a wait for a memory value {:.1}, \
         title memory changed under shader writes {:.1}, title memory read {:.1} (by a buffer copy {:.1}, a fill of upload memory {:.1}, \
         a texture {:.1}), a step of ours {:.1}, {} lists in the air {:.1}, \
         bottom-of-pipe writes (VICTORY_LATE_WAIT_WRITES=0) {:.1}, a blocking fence wait {:.1}; \
         {:.1} waits for memory values met by a flight's bottom-of-pipe write, {:.1} fence signals set behind flights",
        s[0],
        s[1],
        if s[1] > 0.0 { s[11] / s[1] } else { 0.0 },
        s[12] / 1000.0,
        s[WHY_SYNC],
        s[WHY_CARD_DONE],
        s[WHY_VALUE],
        s[WHY_STALE],
        s[WHY_READ],
        s[15],
        s[16],
        s[17],
        s[WHY_RUN],
        late_most(),
        s[WHY_MOST],
        s[WHY_WRITES],
        s[WHY_FENCE],
        s[13],
        s[14],
    );
    {
        let w: Vec<f64> = LATE_WAITED.iter().map(|w| w.swap(0, Ordering::Relaxed) as f64 / 1000.0 / frames).collect();
        tracing::info!(
            "of the waiting for the card in settles, per frame, by what settled: a synchronous submission or present {:.1} ms, \
             the card had run them {:.1}, a wait for a memory value {:.1}, title memory changed under shader writes {:.1}, \
             title memory read {:.1}, a step of ours {:.1}, lists in the air {:.1}, bottom-of-pipe writes {:.1}, a blocking fence wait {:.1}",
            w[WHY_SYNC],
            w[WHY_CARD_DONE],
            w[WHY_VALUE],
            w[WHY_STALE],
            w[WHY_READ],
            w[WHY_RUN],
            w[WHY_MOST],
            w[WHY_WRITES],
            w[WHY_FENCE],
        );
    }
    if relay() {
        let r: Vec<f64> = RELAY_STATS.iter().map(|r| r.swap(0, Ordering::Relaxed) as f64 / frames).collect();
        tracing::info!(
            "the relay per frame: {:.1} flights landed in {:.1} landings by the thread that copies back, \
             beside the submitting thread; that thread waited {:.1} ms for the card with nothing to land and {:.1} ms for \
             fetches of card copies' pieces, and was {:.1} ms landing in all (the frame line's \"copying back\" is its time; \
             \"settles\" above are waits of other threads for it)",
            r[1],
            r[0],
            r[2] / 1000.0,
            r[3] / 1000.0,
            r[4] / 1000.0,
        );
        if send::own() {
            let t: Vec<f64> = send::STATS.iter().map(|r| r.swap(0, Ordering::Relaxed) as f64 / frames).collect();
            tracing::info!(
                "bytes of submissions' own sent up per frame (VICTORY_SEND_OWN): {:.1} MB in {:.0} runs, {:.0} words in scatters; {:.2} ms \
                 taking them out of the CPU sides",
                t[0] / (1 << 20) as f64,
                t[1],
                t[2],
                t[3] / 1000.0,
            );
        }
        let t: Vec<f64> = torn::STATS.iter().map(|r| r.swap(0, Ordering::Relaxed) as f64 / frames).collect();
        if t[0] > 0.0 {
            tracing::info!(
                "fetched pieces checked per frame (VICTORY_FETCH_CHECK): {:.1}, of which {:.2} were not what their flight ended                  with (caught while a later flight was at them, or title bytes went up meanwhile); {:.2} were found whole when                  fetched again",
                t[0],
                t[1],
                t[2],
            );
        }
    }
    #[cfg(feature = "diagnostics")]
    {
        let submit_us = LAND_RETIRE_STATS[0].swap(0, Ordering::Relaxed) as f64;
        let helper_us = LAND_RETIRE_STATS[1].swap(0, Ordering::Relaxed) as f64;
        let peak_landed_flights = LAND_RETIRE_STATS[2].swap(0, Ordering::Relaxed);
        let peak_landed_slots = LAND_RETIRE_STATS[3].swap(0, Ordering::Relaxed);
        let peak_settled_slots = LAND_RETIRE_STATS[4].swap(0, Ordering::Relaxed);
        let (in_flight_slots, landed_slots, landed_flights) = {
            let line = line();
            let in_flight: u64 = line
                .line
                .iter()
                .filter_map(|item| match item {
                    Late::Run(flight) => Some(&flight.lists),
                    _ => None,
                })
                .flat_map(|lists| lists.iter())
                .map(crate::list::Detached::pending_dynamic_slots)
                .sum();
            let landed: u64 = line
                .landed
                .iter()
                .flat_map(|(lists, _)| lists.iter())
                .map(crate::list::Detached::pending_dynamic_slots)
                .sum();
            (in_flight, landed, line.landed.len())
        };
        tracing::info!(
            "landed list recycling per frame: {:.2} ms on submitting thread, {:.2} ms on helper thread; \
             dynamic slots waiting for retirement: {} in flight (peak {} retired at settle), {} in landed (peak {} in landed; {} landed flights queued, peak {})",
            submit_us / 1000.0 / frames,
            helper_us / 1000.0 / frames,
            in_flight_slots,
            peak_settled_slots,
            landed_slots,
            peak_landed_slots.max(landed_slots),
            landed_flights,
            peak_landed_flights.max(landed_flights as u64),
        );
    }
    crate::land::log(frames);
    #[cfg(feature = "diagnostics")]
    place::constant_snapshots_line(frames);
    if place::on() {
        let t: Vec<f64> = place::STATS.iter().map(|r| r.swap(0, Ordering::Relaxed) as f64 / frames).collect();
        tracing::info!(
            "title buffer copies at their place in the lists, per frame (VICTORY_COPIES_IN_PLACE): {:.1} of {:.1} KB, {:.1} of them out              of what shaders wrote; {:.2} copies into textures out of what shaders wrote",
            t[0],
            t[1] / 1024.0,
            t[2],
            t[3],
        );
    }
    card_steps::log(frames);
    crate::port::log(frames);
    crate::port_cbv::log(frames);
    crate::watch::port_guard_line(frames);
    crate::port_journal::line(frames);
    crate::picked::log(frames);
    #[cfg(feature = "diagnostics")]
    log_blame(frames);
    if late_exact() {
        let [pages, runs, bytes] = crate::shadow::merged_stats().map(|n| n as f64 / frames);
        tracing::info!(
            "title bytes put in under flights' shader writes per frame (VICTORY_LATE_WAIT=2): {pages:.0} pages looked at, \
             {runs:.0} changed runs, {:.2} MB",
            bytes / (1 << 20) as f64
        );
        let u = crate::shadow::under_stats().map(|n| n as f64 / frames);
        let ms = |i: usize| u[i] / 1e6;
        tracing::info!(
            "of those, ms a frame: {:.2} finding copies in the lists' refreshes, {:.0} calls: {:.2} asking what was written, {:.2} finding the pages, {:.2} waiting for the locks, {:.2} comparing pages (threads together: {:.2} reading title memory, {:.2} comparing and putting in), {:.2} noting the runs; {:.0} stale pages",
            UNDER_SCAN.swap(0, Ordering::Relaxed) as f64 / 1e6 / frames,
            u[9],
            ms(0),
            ms(1),
            ms(2),
            ms(3),
            ms(5),
            ms(6),
            ms(4),
            u[8],
        );
    }
}

pub(crate) mod aside;
pub(crate) mod awake;
mod mapping_fence;
mod prelude_pool;
mod title_stock;
mod card_steps;
pub(crate) mod indirects;
#[cfg(feature = "diagnostics")]
mod panel;
mod place;
pub(crate) mod rec;
mod send;
mod upq;
pub(crate) use send::{goes_as_words, words_under};
#[cfg(feature = "diagnostics")]
pub(crate) mod spans;
mod torn;
#[cfg(feature = "diagnostics")]
mod traffic;
pub(crate) mod upcheck;
#[cfg(feature = "diagnostics")]
mod stamps;
#[cfg(feature = "diagnostics")]
pub(crate) use stamps::{timeline_ready, timeline_restart, timeline_stamps};

#[cfg(test)]
mod late_tests {
    use super::{continues, join_ranges, touches};

    #[test]
    fn a_late_texture_readback_blocks_a_following_cpu_copy() {
        use super::{INFINITE, WHY_READ, WaitForSingleObject, blocked_by, host, read_backs};

        let Some(host) = host() else {
            eprintln!("no D3D12 device; texture readback dependency not tested");
            return;
        };
        // Submitting a list clears the shared write-watch query.
        let _one = crate::watch::one_asking_test();
        let mut title = [0u8; 64];
        let producer = host.new_list().unwrap();
        let (upload, upload_bytes) = host.upload_buffer(512).unwrap();
        let readback = host.read_back_buffer(512).unwrap();
        // SAFETY: the live upload mapping holds 512 bytes; the producer is
        // open and both buffers hold the full transfer range.
        unsafe {
            upload_bytes.write_bytes(0x5a, 512);
            producer.list.CopyBufferRegion(&readback, 0, &upload, 0, 512);
        }
        producer.defer_read_back(readback, title.as_mut_ptr(), [16, 2, 32]);
        producer.close();

        let (submitted, sent) = host.submit(&[&producer], &[], true, None, None).ahead.unwrap();
        // SAFETY: the submission lock is held and the event and fence are
        // ours. The test keeps the producer, upload and destination alive.
        unsafe {
            host.fence.SetEventOnCompletion(*submitted, host.event).unwrap();
            WaitForSingleObject(host.event, INFINITE);
        }
        if let Some(plan) = sent.card {
            host.card_done(plan);
        }
        if let Some(own) = sent.own {
            own.landed();
        }
        drop(submitted);

        let consumer = host.new_list().unwrap();
        let mut taken = [0u8; 16];
        consumer.defer_copy(taken.as_mut_ptr(), title[32..].as_ptr(), taken.len());
        assert_eq!(blocked_by(&sent.texture_back_ranges, &[&consumer], false), Some(WHY_READ));
        assert_eq!(sent.texture_back_ranges, [(title.as_ptr() as usize, title.as_ptr() as usize + 48)]);
        read_backs(&producer, &mut crate::texback::Scratch::default());
        assert_eq!(title[..16], [0x5a; 16]);
        assert_eq!(title[16..32], [0; 16]);
        assert_eq!(title[32..48], [0x5a; 16]);
        consumer.close();
        eprintln!("the pending texture readback blocked a later CPU copy");
    }

    /// Copies join only when both ends continue and the whole does not
    /// overlap itself.
    #[test]
    fn copies_join_only_end_to_end_and_apart() {
        // (destination, source, bytes)
        assert!(continues((1000, 5000, 16), (1016, 5016, 32)));
        assert!(continues((5000, 1000, 16), (5016, 1016, 32)));
        // A gap, or only one side continues.
        assert!(!continues((1000, 5000, 16), (1020, 5016, 32)));
        assert!(!continues((1000, 5000, 16), (1016, 5020, 32)));
        assert!(!continues((1000, 5000, 16), (1000, 5000, 16)));
        // Together they would overlap: the second reads what the first wrote.
        assert!(!continues((1000, 1016, 16), (1016, 1032, 16)));
        assert!(!continues((1016, 1000, 16), (1032, 1016, 16)));
        // They touch without overlapping.
        assert!(continues((1000, 1032, 16), (1016, 1048, 16)));
        assert!(!continues((usize::MAX - 8, 0, 8), (0, 8, 8)));
    }

    /// Ranges are found by what they hold, not by where they start, and a
    /// range that only touches another's end does not touch it.
    #[test]
    fn ranges_touch_where_they_overlap() {
        let mut ranges = Vec::new();
        join_ranges(&mut ranges, [(40, 50), (10, 20), (20, 25), (45, 60), (7, 7)]);
        assert_eq!(ranges, [(10, 25), (40, 60)]);
        join_ranges(&mut ranges, [(25, 40)]);
        assert_eq!(ranges, [(10, 60)]);
        let ranges = [(10usize, 25usize), (40, 60), (100, 101)];
        for (from, to, hit) in [
            (0, 10, false),
            (0, 11, true),
            (24, 25, true),
            (25, 40, false),
            (25, 41, true),
            (59, 200, true),
            (60, 100, false),
            (100, 101, true),
            (101, 500, false),
            (30, 30, false),
            (50, 50, false),
        ] {
            assert_eq!(touches(&ranges, from, to), hit, "{from}..{to}");
        }
        assert!(!touches(&[], 0, usize::MAX));
    }
}

/// Whether a title buffer copy that continues the one recorded before it
/// in its list becomes one step with it (`VICTORY_JOIN_COPIES=1`; off by
/// default).
///
/// Each copy is done at submission through the kernel, which reports a
/// page that is not there instead of faulting (`copy_mapped`), tells the
/// write watch and the guards, and ends the refreshes gathered so far: a
/// call into the kernel for some 480 bytes, 14,042 times a frame in run 86
/// (6.8 MB; "moves 19 ms" with the refreshes' own reads). How many of them
/// continue the one before is counted either way and said on its own line
/// with the frame rate; the switch is worth that many kernel calls.
///
/// Joined, the same bytes move: [`continues`] joins only copies whose
/// ranges, taken together, do not overlap, and nothing is recorded between
/// two steps that follow each other. What differs: a page that is not
/// mapped now fails the joined copy as a whole (it then goes page by page,
/// zeros for what cannot be read, as a single copy does), where before
/// only the copy that held it did.
fn join_copies() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_JOIN_COPIES").is_some_and(|v| v == "1"))
}

/// Since the last frame-rate line: title buffer copies recorded, and those
/// that continue the one before ([`join_copies`]).
static COPIES_JOINED: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Whether the copy `next` (destination, source, bytes) goes on exactly
/// where `last` ended, source and destination both, and the two as one do
/// not overlap (source over destination): then one copy of both moves the
/// bytes the two would move in turn.
fn continues(last: (usize, usize, usize), next: (usize, usize, usize)) -> bool {
    let ((d, s, n), (dst, src, len)) = (last, next);
    let Some(all) = n.checked_add(len) else {
        return false;
    };
    d.checked_add(n) == Some(dst)
        && s.checked_add(n) == Some(src)
        && (d.checked_add(all).is_some_and(|end| end <= s) || s.checked_add(all).is_some_and(|end| end <= d))
}

/// See [`HostList::defer_gpu_write`].
fn declare_once() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_DECLARE_ONCE").map_or(true, |v| v != "0"))
}

/// The ranges the shaders of `runs` may write, per host copy, as declared.
/// One list's lock at a time: a submission may name a list twice.
fn declared_writes(runs: &[&HostList]) -> Vec<Declared> {
    let mut back: Vec<Declared> = Vec::new();
    // Where each copy is in it, and the ranges seen: dispatches declare the
    // same ranges thousands of times a frame.
    let mut places: ByCopy<usize> = ByCopy::default();
    let mut seen = std::collections::HashSet::<(usize, usize, usize), std::hash::BuildHasherDefault<AddressHash>>::default();
    for list in runs {
        for step in &list.deferred().copies {
            if let Step::GpuWrite(shadow, address, len) = step {
                let key = std::sync::Arc::as_ptr(shadow) as usize;
                if !seen.insert((key, *address, *len)) {
                    continue;
                }
                let i = *places.entry(key).or_insert_with(|| {
                    back.push((shadow.clone(), Vec::new()));
                    back.len() - 1
                });
                back[i].1.push((*address, *address + *len));
            }
        }
    }
    back
}

/// The walls trace (`VICTORY_WALLS_TRACE=1` or `2`; off by default): log
/// lines that say what the title's CPU was given while it decided whether
/// the camera is in a room. The values of the small read-backs
/// ([`read_backs`]), the frame's main pass by scissor rectangle with its
/// culling dispatches (`draw.rs`, `walls_census`), the shader-written
/// buffers merged into title memory (`shadow.rs`, `walls_merged`), host
/// copies found stale after their refresh (`shadow.rs`; only with
/// `VICTORY_REFRESH_CHECK=1` as well: it costs about 100 ms a frame), and, in the
/// runtime, the read requests that fail (with 2: every request).
pub(crate) fn walls_trace() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WALLS_TRACE").is_some_and(|v| v == "1" || v == "2"))
}

/// After `list` has run: the texels its read-backs copied into host
/// buffers of ours go to the title's memory (`HostList::defer_read_back`).
///
/// A submission that is waited for did this itself. One sent ahead
/// ([`late`]) did not, at any stage: the flights' lists were settled
/// without it, so the title read a buffer nobody wrote, as before read-backs
/// were made (the exposure metering's two small float targets, every frame).
fn read_backs(list: &HostList, scratch: &mut crate::texback::Scratch) {
    let deferred = list.deferred();
    // Textures shaders wrote that the title reads on the CPU
    // ([`crate::texback`]).
    if !deferred.texture_backs.is_empty() {
        #[cfg(feature = "diagnostics")]
        let began = std::time::Instant::now();
        for back in &deferred.texture_backs {
            crate::texback::land(back, scratch);
        }
        #[cfg(feature = "diagnostics")]
        {
            crate::land::spent(crate::land::stat::TEXTURE_READBACK_US, began);
            crate::land::count(crate::land::stat::TEXTURE_READBACKS, deferred.texture_backs.len() as u64);
        }
    }
    #[cfg(feature = "diagnostics")]
    let began = (!deferred.backs.is_empty()).then(std::time::Instant::now);
    for (buffer, at, [row, rows, pitch]) in &deferred.backs {
        let mut data = std::ptr::null_mut();
        // SAFETY: a read-back buffer of ours, written by the list that has
        // run; the title's destination holds every row (checked when the
        // copy was recorded).
        unsafe {
            if buffer.Map(0, None, Some(&mut data)).is_ok() && !data.is_null() {
                let from = data.cast::<u8>().cast_const();
                for line in 0..*rows {
                    std::ptr::copy_nonoverlapping(from.add(line * row.next_multiple_of(256)), (*at as *mut u8).add(line * pitch), *row);
                }
                // The walls trace: what the title is handed (one row of at
                // most 24 values: its 5x1 and 1x1 float targets).
                if walls_trace() && *rows == 1 && *row <= 96 {
                    let mut values = [0f32; 24];
                    std::ptr::copy_nonoverlapping(from, values.as_mut_ptr().cast::<u8>(), *row & !3);
                    tracing::info!(
                        frame = crate::frame::current(),
                        to = format_args!("{:#x}", *at),
                        bytes = *row,
                        values = ?&values[..*row / 4],
                        "walls trace: a read-back lands in title memory"
                    );
                }
                buffer.Unmap(0, None);
            }
        }
    }
    #[cfg(feature = "diagnostics")]
    if let Some(began) = began {
        crate::land::spent(crate::land::stat::BUFFER_READBACK_US, began);
        crate::land::count(crate::land::stat::BUFFER_READBACKS, deferred.backs.len() as u64);
    }
}

/// After the card ran what declared `back`: what shaders wrote there goes
/// into title memory, and the ranges are done with.
fn hand_back(back: &[Declared]) {
    if crate::shadow::wide_back() {
        // Every range first (the pieces that differ, found on several
        // threads), then the merges, in order, as below.
        let mut work: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)> = Vec::new();
        for (shadow, declared) in back {
            if shadow.carded() {
                continue;
            }
            let mut sorted = declared.clone();
            sorted.sort_unstable();
            let mut spans: Vec<(usize, usize)> = Vec::new();
            for (from, to) in sorted {
                match spans.last_mut() {
                    Some(last) if from <= last.1 => last.1 = last.1.max(to),
                    _ => spans.push((from, to)),
                }
            }
            work.extend(spans.into_iter().map(|(from, to)| (shadow.clone(), from, to)));
        }
        #[cfg(feature = "diagnostics")]
        let scanning = std::time::Instant::now();
        let found = crate::shadow::find_all(&work);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::BackScanUs, scanning.elapsed().as_micros() as u64);
        for ((shadow, from, to), found) in work.iter().zip(found) {
            let written = shadow.copy_back_with(*from, to - from, Some(found));
            COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            crate::copystat::compared(*from, to - from, written);
        }
        for (shadow, declared) in back {
            shadow.copy_back_done(declared);
        }
        return;
    }
    for (shadow, declared) in back {
        if shadow.carded() {
            // Done before (`Host::card_back`).
            shadow.copy_back_done(declared);
            continue;
        }
        // Each byte compared once: the world declares dozens of views
        // of its 174 MB heap that overlap almost entirely, and comparing
        // every one whole was gigabytes a frame (0.3-0.5 s) for the few
        // MB shaders wrote. Only ranges that overlap or touch are
        // joined: what lies between two was not given to shaders (the
        // snapshot does not hold it).
        let mut sorted = declared.clone();
        sorted.sort_unstable();
        let mut spans: Vec<(usize, usize)> = Vec::new();
        for (from, to) in sorted {
            match spans.last_mut() {
                Some(last) if from <= last.1 => last.1 = last.1.max(to),
                _ => spans.push((from, to)),
            }
        }
        for (from, to) in spans {
            let written = shadow.copy_back(from, to - from);
            COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            crate::copystat::compared(from, to - from, written);
        }
        shadow.copy_back_done(declared);
    }
}

impl Host {
    /// Whether the host runs mesh shaders at shader model 6.6 (translated
    /// mesh shaders need both); logged once.
    pub(crate) fn mesh_shaders(&self) -> bool {
        static MESH: OnceLock<bool> = OnceLock::new();
        *MESH.get_or_init(|| {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
            let mut model = D3D12_FEATURE_DATA_SHADER_MODEL {
                HighestShaderModel: D3D_SHADER_MODEL_6_6,
            };
            // SAFETY: each query writes its own structure of the given size.
            let ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS7,
                        (&raw mut options).cast(),
                        size_of_val(&options) as u32,
                    )
                    .is_ok()
                    && self
                        .device
                        .CheckFeatureSupport(
                            D3D12_FEATURE_SHADER_MODEL,
                            (&raw mut model).cast(),
                            size_of_val(&model) as u32,
                        )
                        .is_ok()
            };
            let mesh = ok
                && options.MeshShaderTier != D3D12_MESH_SHADER_TIER_NOT_SUPPORTED
                && model.HighestShaderModel.0 >= D3D_SHADER_MODEL_6_6.0;
            tracing::info!(mesh, "host mesh shader support");
            mesh
        })
    }

    /// Whether the host GPU supports the depth bounds test; logged once.
    pub(crate) fn depth_bounds(&self) -> bool {
        static BOUNDS: OnceLock<bool> = OnceLock::new();
        *BOUNDS.get_or_init(|| {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS2::default();
            // SAFETY: writes `options` of the given size.
            let ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS2,
                        (&raw mut options).cast(),
                        size_of_val(&options) as u32,
                    )
                    .is_ok()
            };
            let supported = ok && options.DepthBoundsTestSupported.as_bool();
            tracing::info!(supported, "host depth bounds support");
            supported
        })
    }

    /// A new list, open for recording.
    pub(crate) fn new_list(&self) -> windows::core::Result<HostList> {
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        let before = private_bytes();
        let list = self.make_list();
        LISTS[0].fetch_add(1, Ordering::Relaxed);
        // Keep the sign of each process-wide change in the sum.
        #[cfg(feature = "diagnostics")]
        LISTS[2].fetch_add(private_bytes().wrapping_sub(before), Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(C::HostLists, 1);
            add(C::HostListNs, started.elapsed().as_nanos() as u64);
        }
        list
    }

    /// An allocator and a closed list for a prelude.
    fn make_prelude(&self) -> Option<Prelude> {
        // SAFETY: plain object creation.
        let made: windows::core::Result<Prelude> = unsafe {
            (|| {
                let allocator: ID3D12CommandAllocator =
                    self.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
                let list: ID3D12GraphicsCommandList =
                    self.device
                        .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)?;
                list.Close()?;
                Ok((allocator, list))
            })()
        };
        LISTS[1].fetch_add(1, Ordering::Relaxed);
        match made {
            Ok(prelude) => Some(prelude),
            Err(e) => {
                tracing::warn!(error = %e, "no host list for a prelude");
                None
            }
        }
    }

    fn make_list(&self) -> windows::core::Result<HostList> {
        // SAFETY: plain object creation.
        unsafe {
            let allocator: ID3D12CommandAllocator = self
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList = self.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &allocator,
                None,
            )?;
            let prelude = prelude_pool::Slot::new(self);
            #[cfg(feature = "diagnostics")]
            let notes = std::sync::Arc::<Mutex<Vec<Note>>>::default();
            let pipelines = std::sync::Arc::<Mutex<std::collections::BTreeSet<String>>>::default();
            #[cfg(feature = "diagnostics")]
            let indirects = indirects::Shared::default();
            #[cfg(feature = "diagnostics")]
            if dred_on() || dred_light() {
                static NUMBER: AtomicU32 = AtomicU32::new(0);
                let number = NUMBER.fetch_add(1, Ordering::Relaxed);
                name(&list, || format!("victory host list {number}"));
                let mut all = LIST_NOTES.lock().unwrap_or_else(|e| e.into_inner());
                if number.is_multiple_of(256) {
                    all.retain(|_, l| l.0.strong_count() > 0);
                }
                all.insert(
                    number,
                    (
                        std::sync::Arc::downgrade(&notes),
                        std::sync::Arc::downgrade(&pipelines),
                        std::sync::Arc::downgrade(&indirects),
                    ),
                );
            }
            Ok(HostList {
                allocator,
                list: rec::Rec::new(list),
                recording: Recording::open(),
                deferred: Mutex::default(),
                scratch: Mutex::default(),
                states: Mutex::default(),
                twins: Mutex::default(),
                prelude,
                pipelines,
                #[cfg(feature = "diagnostics")]
                indirects,
                budget: OnceLock::new(),
                card_budget: OnceLock::new(),
                indirect_args: Mutex::default(),
                batch_args: Mutex::default(),
                coming: Mutex::default(),
                epoch: std::sync::atomic::AtomicU64::new(next_epoch()),
                _alive: Counted::new(),
                #[cfg(feature = "diagnostics")]
                notes,
                #[cfg(feature = "diagnostics")]
                stamps: OnceLock::new(),
            })
        }
    }

    /// The card's clock ([`CARD_CLOCK_US`]), made at the first send.
    fn card_clock(&self) -> Option<&'static CardClock> {
        static CLOCK: OnceLock<Option<CardClock>> = OnceLock::new();
        static WATCH: std::sync::Once = std::sync::Once::new();
        let clock = CLOCK
            .get_or_init(|| {
                // SAFETY: a plain fence on our device.
                let fence = unsafe { self.device.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE) }.ok()?;
                Some(CardClock {
                    fence,
                    state: Mutex::new((0, None)),
                    sent: std::sync::Condvar::new(),
                })
            })
            .as_ref()?;
        WATCH.call_once(|| {
            if let Err(e) = std::thread::Builder::new().name("victory-card-clock".into()).spawn(move || card_clock_watch(clock)) {
                tracing::warn!(error = %e, "no thread for the card's clock: the frame line's idle times stay 0");
            }
        });
        Some(clock)
    }

    /// Lists of ours just went to the queue: the clock's signal goes
    /// behind them, and an idle stretch of the card, if one was under way,
    /// ends here ([`CARD_CLOCK_US`]).
    fn card_sent(&self) {
        let Some(clock) = self.card_clock() else {
            return;
        };
        let mut state = clock.state.lock().unwrap_or_else(|e| e.into_inner());
        state.0 += 1;
        // SAFETY: our queue and the clock's fence.
        let _ = unsafe { self.queue.Signal(&clock.fence, state.0) };
        if let Some((since, worked_then)) = state.1.take() {
            CARD_CLOCK_US[0].fetch_add(since.elapsed().as_micros() as u64, Ordering::Relaxed);
            CARD_CLOCK_US[1].fetch_add(worked().saturating_sub(worked_then), Ordering::Relaxed);
        }
        drop(state);
        clock.sent.notify_one();
    }

    /// Whether the host has what copies in the card's memory need (three
    /// lists and the compare); made at the first ask. A copy goes to the
    /// card's memory only if it has.
    pub(crate) fn card_ready(&self) -> bool {
        static READY: OnceLock<bool> = OnceLock::new();
        *READY.get_or_init(|| match self.make_card_work() {
            Ok(work) => {
                *CARD_WORK.lock().unwrap_or_else(|e| e.into_inner()) = Some(work);
                true
            }
            Err(e) => {
                tracing::warn!(error = %e, "no host copies in the card's memory: the compare could not be made");
                false
            }
        })
    }

    fn make_card_work(&self) -> windows::core::Result<CardWork> {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        let mut code: Option<ID3DBlob> = None;
        // SAFETY: the source and names live through the call; the out
        // pointer is ours.
        unsafe {
            D3DCompile(
                CARD_COMPARE.as_ptr().cast(),
                CARD_COMPARE.len(),
                windows::core::s!("card compare"),
                None,
                None,
                windows::core::s!("main"),
                windows::core::s!("cs_5_1"),
                0,
                0,
                &mut code,
                None,
            )?;
        }
        let code = code.ok_or_else(windows::core::Error::empty)?;
        let list = || self.make_prelude().ok_or_else(windows::core::Error::empty);
        // A megabyte of results: 262,144 pieces, 16 GB of declared ranges
        // in one submission.
        const RESULTS: u64 = 1 << 20;
        let (differs, data) = self.shared_buffer(RESULTS).ok_or_else(windows::core::Error::empty)?;
        name(&differs, || "card compare results".into());
        // SAFETY: the blob's own bytes; the root signature is inside the
        // shader.
        unsafe {
            let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
            let root: ID3D12RootSignature = self.device.CreateRootSignature(0, bytes)?;
            let pso: ID3D12PipelineState = self.device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: bytes.as_ptr().cast(),
                    BytecodeLength: bytes.len(),
                },
                ..Default::default()
            })?;
            #[cfg(feature = "diagnostics")]
            name(&pso, || crate::replay::play::COMPARE.into());
            Ok(CardWork {
                sync: vec![list()?],
                check: vec![list()?],
                fetch: list()?,
                root,
                pso,
                differs: (differs, data as usize, RESULTS),
                fetched: None,
                taken: 0,
                out: 0,
            })
        }
    }

    /// Before a submission's lists run, for copies in the card's memory
    /// ([`crate::shadow::cards`]): what changed on their CPU side since
    /// the last submission goes to their card side (a list of ours, run
    /// now; the submission's lists follow it on the queue), and the ranges
    /// this submission's shaders may write get a compare to run after its
    /// lists. `None` if there is nothing to do.
    ///
    /// `flown`: the submission is sent ahead with [`relay`]. Its compare
    /// then stores the hashes it makes (see [`flagged_pieces`] and
    /// `relay`).
    /// `picked`: ranges that may be written too, and ranges to look at
    /// (`crate::picked`).
    fn card_before(&self, declared: &[Declared], picked: &crate::picked::Ranges, flown: bool) -> Option<CardPlan> {
        use std::sync::Arc;
        if !crate::shadow::cards() {
            return None;
        }
        // The ranges this submission's shaders may write, per copy.
        let declared: Vec<Declared> = declared.iter().filter(|(shadow, _)| shadow.carded()).cloned().collect();
        let declared = if crate::picked::lands() { crate::picked::with(declared, &picked.written) } else { declared };
        // (Of a copy with a declared range here, that says it.)
        let looked: Vec<&crate::picked::Range> =
            picked.looked.iter().filter(|l| !declared.iter().any(|(s, _)| Arc::ptr_eq(s, &l.0))).collect();
        // With the relay: the order of the queue is held from here until
        // what is taken is on the queue, and the taking is numbered.
        let mut order = relay().then(|| QUEUE_ORDER.lock().unwrap_or_else(|e| e.into_inner()));
        let sent_as = match order.as_mut() {
            Some(order) => {
                **order += 1;
                **order
            }
            None => 0,
        };
        let gathering = crate::timeline::span("copies in the card's memory: what changed");
        let mut dirty = crate::shadow::card_dirty(sent_as);
        gathering.end();
        if declared.is_empty() && dirty.is_empty() && looked.is_empty() {
            return None;
        }
        let mut work = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
        let work = work.as_mut()?;
        // A copy whose shader writes are declared for the first time gets
        // its bytes from before, whole.
        let mut whole: Vec<Arc<crate::shadow::Shadow>> = Vec::new();
        for shadow in declared.iter().map(|d| &d.0).chain(looked.iter().map(|l| &l.0)) {
            if shadow.card_make_before(|size| {
                let before = self.uav_buffer(crate::shadow::card_hash_bytes(size))?;
                name(&before, || "title buffer on the card, hashes of its pieces".into());
                Some(before)
            }) {
                whole.push(shadow.clone());
                if !dirty.iter().any(|(s, _)| Arc::ptr_eq(s, shadow)) {
                    dirty.push((shadow.clone(), Vec::new()));
                }
            }
        }
        let mut used: Vec<(bool, Prelude)> = Vec::new();
        let mut looks = None;
        let mut up = None;
        // The bytes as they are now, whatever later submissions put there.
        let taking = crate::timeline::span("copies in the card's memory: taken");
        let own = if send::own() { send::take(self, &dirty) } else { None };
        taking.end();
        if !dirty.is_empty() {
            let Some(sync) = work.sync.pop().or_else(|| self.make_prelude()) else {
                fail(format_args!("no host list to bring host copies in the card's memory up to date"));
            };
            let _recording = crate::timeline::span("copies in the card's memory: the list");
            let (allocator, list) = &sync;
            // SAFETY: the list's last run finished (it ran ahead of a
            // submission that was waited for, and came back after that);
            // live buffers, and the ranges lie in them.
            let recorded = unsafe {
                (|| -> windows::core::Result<()> {
                    allocator.Reset()?;
                    list.Reset(allocator, None)?;
                    if upcheck::on() {
                        looks = upcheck::record(self, list, &dirty);
                    }
                    // Every copy first, then one barrier call, then every
                    // pass: a wait between each copy and its pass had the
                    // card do them one after the other.
                    let mut sent = Vec::new();
                    let mut transitioned = Vec::new();
                    if let Some(own) = &own {
                        // With `upq`: the submission's own memory (the PC's)
                        // comes whole into a buffer of ours in the card's on
                        // the copy queue; the sends come from there.
                        if looks.is_none() && upq::on() {
                            let (memory, used) = own.source();
                            up = upq::send(self, &self.queue, &[(memory, 0, memory, used)], |_, _, _| {});
                        }
                        let from = up.as_ref().and_then(upq::Slot::bounce);
                        own.record(self, list, &dirty, |shadow| looks.as_ref().is_some_and(|l| l.hold(shadow)), from);
                    }
                    // With `upq`: the PC's bytes come into a buffer of ours
                    // on the copy queue, and this list copies them from
                    // there (the card's memory), after a wait for that.
                    if own.is_none() && looks.is_none() && upq::on() {
                        let sides: Vec<_> = dirty.iter().map(|(shadow, _)| shadow.card_sides()).collect();
                        let mut ranges: Vec<upq::Range<'_>> = Vec::new();
                        for ((shadow, rs), side) in dirty.iter().zip(&sides) {
                            if let Some((staging, _, _)) = side {
                                ranges.extend(rs.iter().map(|&(a, b)| (&shadow.buffer, a as u64, *staging, (b - a) as u64)));
                            }
                        }
                        up = upq::send(self, &self.queue, &ranges, |i, bounce, at| {
                            let (card, offset, _, bytes) = ranges[i];
                            list.CopyBufferRegion(card, offset, bounce, at, bytes);
                        });
                    }
                    for (shadow, ranges) in &dirty {
                        let Some((staging, before, _)) = shadow.card_sides() else {
                            continue;
                        };
                        if own.is_none() && up.is_none() {
                            for &(a, b) in ranges {
                                list.CopyBufferRegion(&shadow.buffer, a as u64, staging, a as u64, (b - a) as u64);
                            }
                        }
                        let bytes: u64 = ranges.iter().map(|&(a, b)| (b - a) as u64).sum();
                        crate::shadow::CARD_STATS[0].fetch_add(bytes, Ordering::Relaxed);
                        crate::shadow::CARD_STATS[1].fetch_add(ranges.len() as u64, Ordering::Relaxed);
                        #[cfg(feature = "diagnostics")]
                        traffic::moved(traffic::UP, shadow.span(), ranges.iter().copied());
                        if own.is_none() && before.is_some() && !ranges.is_empty() {
                            let raw = shadow.buffer.as_raw();
                            if !transitioned.contains(&raw) {
                                transitioned.push(raw);
                                sent.push(transition(
                                    &shadow.buffer,
                                    D3D12_RESOURCE_STATE_COPY_DEST,
                                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                ));
                            }
                        }
                    }
                    if !sent.is_empty() {
                        list.ResourceBarrier(&sent);
                    }
                    list.SetComputeRootSignature(&work.root);
                    list.SetPipelineState(&work.pso);
                    list.SetComputeRootUnorderedAccessView(3, work.differs.0.GetGPUVirtualAddress());
                    for (shadow, ranges) in &dirty {
                        let (Some((_, Some(before), _)), size) = (shadow.card_sides(), shadow.card_hash_size()) else {
                            continue;
                        };
                        // No twin (it cost the card the copy's size again,
                        // 399 MB in run 75): `before` holds a 128-bit hash
                        // of each 64 KiB piece. The pieces just sent up are
                        // hashed again, all of them the first time; the
                        // compare after the lists hashes the declared
                        // pieces and flags those whose hash is another.
                        list.SetComputeRootShaderResourceView(1, shadow.buffer.GetGPUVirtualAddress());
                        list.SetComputeRootUnorderedAccessView(2, before.GetGPUVirtualAddress());
                        let joins = rehash_runs() || own.is_some();
                        let runs: Vec<crate::shadow::CardSpan> = if whole.iter().any(|s| Arc::ptr_eq(s, shadow)) {
                            vec![shadow.card_run(0, usize::MAX)]
                        } else {
                            // (In order: the next of the same piece, of the
                            // tens of thousands there are, says nothing.)
                            let mut runs: Vec<crate::shadow::CardSpan> = Vec::new();
                            for run in ranges.iter().map(|&(a, b)| shadow.card_run(a, b)) {
                                match runs.last_mut() {
                                    Some(last) if joins && (last.first, last.pieces) == (run.first, run.pieces) => last.to = last.to.max(run.to),
                                    _ => runs.push(run),
                                }
                            }
                            runs
                        };
                        // The same pieces in fewer passes (`rehash_runs`).
                        let runs = if joins { crate::shadow::joined_piece_runs(runs) } else { runs };
                        CARD_PASSES[0].fetch_add(runs.len() as u64, Ordering::Relaxed);
                        CARD_PASSES[1].fetch_add(runs.iter().map(|r| u64::from(r.pieces)).sum::<u64>(), Ordering::Relaxed);
                        for run in runs {
                            card_pass(list, &run, size as u32, 0, 0);
                        }
                    }
                    list.Close()?;
                    self.queue.ExecuteCommandLists(&[Some(list.cast()?)]);
                    Ok(())
                })()
            };
            if let Err(e) = recorded {
                fail(format_args!("host copies in the card's memory could not be brought up to date ({e})"));
            }
            used.push((false, sync));
        }
        drop(order);
        let mut plan = CardPlan {
            ranges: Vec::new(),
            check: None,
            _synced: dirty.into_iter().map(|(shadow, _)| shadow).collect(),
            lists: used,
            sent_as,
            looks,
            sends: own.map(send::Own::memory),
            up,
        };
        // Each byte once, as the copy-back joins the declared ranges. The
        // results go behind those of plans still in the air.
        let mut at = work.taken;
        for (shadow, ranges) in &declared {
            let mut sorted = ranges.clone();
            sorted.sort_unstable();
            let mut spans: Vec<(usize, usize)> = Vec::new();
            for (from, to) in sorted {
                match spans.last_mut() {
                    Some(last) if from <= last.1 => last.1 = last.1.max(to),
                    _ => spans.push((from, to)),
                }
            }
            let twin = shadow.card_sides().is_some_and(|(_, before, _)| before.is_some());
            for (from, to) in spans {
                let Some(span) = shadow.card_span(from, to) else {
                    continue;
                };
                let compared = twin && u64::from(at + span.pieces) * 4 <= work.differs.2;
                plan.ranges.push(CardRange {
                    shadow: shadow.clone(),
                    declared: (from, to),
                    span,
                    at,
                    compared,
                    picked: false,
                });
                if compared {
                    at += span.pieces;
                }
            }
        }
        for (shadow, from, len) in looked {
            let twin = shadow.card_sides().is_some_and(|(_, before, _)| before.is_some());
            if let Some(span) = shadow.card_span(*from, from + len)
                && twin
                && u64::from(at + span.pieces) * 4 <= work.differs.2
            {
                plan.ranges.push(CardRange {
                    shadow: shadow.clone(),
                    declared: (*from, from + len),
                    span,
                    at,
                    compared: true,
                    picked: true,
                });
                at += span.pieces;
            }
        }
        work.taken = at;
        if plan.ranges.iter().any(|r| r.compared) {
            let Some(check) = work.check.pop().or_else(|| self.make_prelude()) else {
                fail(format_args!("no host list for the compare of host copies in the card's memory"));
            };
            let (allocator, list) = &check;
            // SAFETY: the list's last run finished (its submission was
            // waited for, and it came back after that); live buffers; the
            // compare's own root signature.
            let recorded = unsafe {
                (|| -> windows::core::Result<ID3D12CommandList> {
                    allocator.Reset()?;
                    list.Reset(allocator, None)?;
                    // What the submission's shaders wrote, first.
                    list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
                        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                        Anonymous: D3D12_RESOURCE_BARRIER_0 {
                            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                                pResource: std::mem::ManuallyDrop::new(None),
                            }),
                        },
                    }]);
                    list.SetComputeRootSignature(&work.root);
                    list.SetPipelineState(&work.pso);
                    list.SetComputeRootUnorderedAccessView(3, work.differs.0.GetGPUVirtualAddress());
                    for r in plan.ranges.iter().filter(|r| r.compared) {
                        let (Some((_, Some(before), _)), size) = (r.shadow.card_sides(), r.shadow.card_hash_size()) else {
                            continue;
                        };
                        list.SetComputeRootShaderResourceView(1, r.shadow.buffer.GetGPUVirtualAddress());
                        list.SetComputeRootUnorderedAccessView(2, before.GetGPUVirtualAddress());
                        // (What is only looked at keeps its hashes: what
                        // changed there is the next declared range's to land.)
                        card_pass(list, &r.span, size as u32, r.at, if flown && !r.picked { 2 } else { 1 });
                        CARD_PASSES[2].fetch_add(1, Ordering::Relaxed);
                    }
                    list.Close()?;
                    list.cast()
                })()
            };
            match recorded {
                Ok(list) => plan.check = Some(list),
                Err(e) => fail(format_args!("the compare of host copies in the card's memory could not be recorded ({e})")),
            }
            plan.lists.push((true, check));
        }
        work.out += 1;
        Some(plan)
    }

    /// With [`relay`]: whether the compare's result area is half taken by
    /// plans in the air. It starts from the beginning only when no plan is
    /// out, so the caller lets the flights land first.
    fn card_results_low(&self) -> bool {
        CARD_WORK
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .is_some_and(|work| u64::from(work.taken) * 8 > work.differs.2)
    }

    /// The plan's submission has run and its results were read
    /// ([`Host::card_back`]): its lists are free for the next plan. Called
    /// when nothing of ours is on the card, so the compare's results start
    /// from the beginning again.
    fn card_done(&self, plan: CardPlan) {
        if let Some(looks) = plan.looks {
            looks.read();
        }
        if let Some(memory) = plan.sends {
            send::give_back(memory);
        }
        if let Some(slot) = plan.up {
            upq::give_back(slot);
        }
        let mut work = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
        let Some(work) = work.as_mut() else {
            return;
        };
        for (check, list) in plan.lists {
            if check { work.check.push(list) } else { work.sync.push(list) }
        }
        work.out = work.out.saturating_sub(1);
        // With the relay other plans may be on the card, their results
        // not read yet: the area starts again only when none is out.
        if !relay() || work.out == 0 {
            work.taken = 0;
        }
    }

    /// The relay's own list, fence, event for [`Host::relay_card_back`].
    fn make_relay_fetch(&self) -> Option<RelayFetch> {
        // SAFETY: plain object creation.
        let copying = relay_copy().then(|| unsafe {
            (|| -> windows::core::Result<(Prelude, ID3D12CommandQueue)> {
                let allocator: ID3D12CommandAllocator = self.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY)?;
                let list: ID3D12GraphicsCommandList =
                    self.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, &allocator, None)?;
                list.Close()?;
                let queue: ID3D12CommandQueue = self.device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_COPY,
                    Priority: D3D12_COMMAND_QUEUE_PRIORITY_HIGH.0,
                    ..Default::default()
                })?;
                Ok(((allocator, list), queue))
            })()
        });
        let (list, queue) = match copying {
            Some(Ok((list, queue))) => (list, Some(queue)),
            Some(Err(e)) => {
                tracing::warn!(error = %e, "no copy queue: what shaders wrote to the card's memory is fetched on the drawing queue");
                (self.make_prelude()?, None)
            }
            None => (self.make_prelude()?, None),
        };
        // SAFETY: plain objects of our own.
        unsafe {
            let fence: ID3D12Fence = self.device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
            let event = CreateEventW(None, false, false, None).ok()?;
            Some(RelayFetch {
                list,
                queue,
                fence,
                event,
                sent: 0,
                buffer: None,
                suspects: torn::Suspects::default(),
                had: crate::land::Had::default(),
            })
        }
    }

    /// [`relay`], on the thread that lands flights: for the plans of the
    /// flights being landed (the card has run them), the pieces their
    /// compares flagged are fetched, once each, with a list and a wait of
    /// this thread's own, and what shaders wrote in them goes into title
    /// memory (`Shadow::card_merge_relay`). As [`Host::card_back`], but:
    /// no submission lock; a piece flagged under one range of a plan is
    /// fetched under all ([`flagged_pieces`]); the fetch runs behind
    /// whatever was sent meanwhile, so it may bring a later flight's
    /// bytes, which that flight's landing then finds merged already; and
    /// nothing is sent back up. `plans` come with the value signalled behind
    /// their flight. `meanwhile` is called while the fetch is on its way, if
    /// there is one.
    fn relay_card_back(&self, plans: &[(&CardPlan, u64)], fetch: &mut Option<RelayFetch>, meanwhile: &mut dyn FnMut()) {
        use crate::land;
        let gathering = std::time::Instant::now();
        let differs = {
            let work = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
            work.as_ref().map(|work| (work.differs.1, (work.differs.2 / 4) as usize))
        };
        let mut pieces: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize, u64)> = Vec::new();
        let mut total = 0u64;
        if let (Some(fetch), Some(first)) = (fetch.as_mut(), plans.iter().map(|p| p.1).min()) {
            fetch.had.landing(first);
        }
        // SAFETY: the mapping of the results. The compares of these plans
        // have run, and their dwords are given to no other plan before these
        // are done with (`card_done`).
        let differs = differs.map(|(differs, dwords)| unsafe { std::slice::from_raw_parts(differs as *const u32, dwords) });
        if let Some(differs) = differs {
            // Pieces caught while a later flight was at them (`torn`).
            let suspects = fetch.as_ref().map(|f| &f.suspects);
            let mut seen = std::collections::HashSet::new();
            for &(plan, behind) in plans {
                plan.picked_found(differs);
                let flags = plan
                    .ranges
                    .iter()
                    .filter(|r| !r.picked)
                    .flat_map(|r| {
                        (0..r.span.pieces).map(move |p| {
                            let flagged = !r.compared || differs.get((r.at + p) as usize).is_none_or(|&d| torn::flagged(d));
                            let wanted = suspects.map_or(flagged, |s| s.wanted((r.shadow.id(), r.span.first + p), flagged));
                            (std::sync::Arc::as_ptr(&r.shadow) as usize, r.span.first + p, wanted)
                        })
                    });
                let fetched = flagged_pieces(flags);
                for r in plan.ranges.iter().filter(|r| !r.picked) {
                    let copy = std::sync::Arc::as_ptr(&r.shadow) as usize;
                    if upcheck::on() {
                        upcheck::declared(r.shadow.id(), r.span.from as usize, r.span.to as usize);
                    }
                    for p in flagged_in_range(&fetched, copy, r.span.first, r.span.pieces) {
                        let (lo, hi) = r.shadow.card_piece(&r.span, p);
                        if lo < hi && !seen.contains(&(copy, lo, hi)) {
                            let key = (r.shadow.id(), r.shadow.span().0 as usize + lo, hi - lo);
                            if fetch.as_ref().is_some_and(|f| f.had.holds(behind, key)) {
                                land::count(land::stat::HAD, 1);
                                land::count(land::stat::HAD_BYTES, (hi - lo) as u64);
                                if land::skip_on() {
                                    continue;
                                }
                            }
                            if upcheck::on() {
                                upcheck::fetched(r.shadow.id(), lo);
                            }
                            seen.insert((copy, lo, hi));
                            pieces.push((r.shadow.clone(), lo, hi, total));
                            total += (hi - lo) as u64;
                        }
                    }
                    crate::shadow::CARD_STATS[2].fetch_add(u64::from(r.span.pieces), Ordering::Relaxed);
                }
            }
        }
        if land::on() {
            // In the order they are landed, each byte of a copy once.
            let keys: Vec<land::Piece> =
                pieces.iter().map(|(shadow, lo, hi, _)| (shadow.id(), shadow.span().0 as usize + lo, hi - lo)).collect();
            total = 0;
            pieces = land::ordered(&keys)
                .into_iter()
                .map(|(i, title, len)| {
                    let shadow = &pieces[i].0;
                    let lo = title - shadow.span().0 as usize;
                    let at = total;
                    total += len as u64;
                    (shadow.clone(), lo, lo + len, at)
                })
                .collect();
        }
        land::spent(land::stat::GATHER, gathering);
        if fetch.is_none() {
            *fetch = self.make_relay_fetch();
        }
        // On the copy queue a later flight may be at a piece while it is
        // fetched: the pieces come down whole, to be hashed.
        let check = (torn::on() && fetch.as_ref().is_some_and(|f| f.queue.is_some()))
            .then(|| torn::Check::new(plans.iter().map(|p| p.0), differs.unwrap_or_default(), &mut pieces));
        if let Some(check) = &check {
            total = check.total;
        }
        #[cfg(feature = "diagnostics")]
        for (shadow, lo, hi, _) in &pieces {
            traffic::moved(traffic::FETCHED, shadow.span(), std::iter::once((*lo, *hi)));
        }
        // The sends these flights ran behind, all a fetch on the copy queue
        // is known to have run behind.
        let behind = plans.iter().map(|p| p.0.sent_as).max().unwrap_or(0);
        if pieces.is_empty() {
            // Nothing to fetch: every send so far is older than the next
            // fetch, if that goes to the queue the sends went to.
            let after = match fetch.as_ref().is_some_and(|f| f.queue.is_some()) {
                true => behind,
                false => *QUEUE_ORDER.lock().unwrap_or_else(|e| e.into_inner()),
            };
            crate::shadow::relay_fetched(after);
            return;
        }
        let Some(f) = fetch.as_mut() else {
            fail(format_args!("no list, fence or event for the relay's GPU readback"));
        };
        if f.buffer.as_ref().is_none_or(|b| b.2 < total) {
            let size = total.next_multiple_of(16 << 20);
            f.buffer = self.shared_buffer(size).map(|(buffer, data)| (buffer, data as usize, size));
        }
        let Some((buffer, data, _)) = f.buffer.clone() else {
            fail(format_args!("no memory to fetch what shaders wrote to the card's memory ({total} bytes, relay)"));
        };
        f.sent += 1;
        let ((allocator, list), fence, event, sent) = (f.list.clone(), f.fence.clone(), f.event, f.sent);
        let copy_queue = f.queue.clone();
        let waiting = std::time::Instant::now();
        // The flights the fetch runs behind: those in the line by now have
        // been given to the queue. On the copy queue, those landed here.
        let ahead = match copy_queue {
            Some(_) => 0,
            None => line().target,
        };
        // SAFETY: the list's last run was waited for by this thread; live
        // buffers (the copies are held by the plans); the fence and the
        // event are this thread's.
        let ran = unsafe {
            (|| -> windows::core::Result<u64> {
                allocator.Reset()?;
                list.Reset(&allocator, None)?;
                for (shadow, lo, hi, at) in check.as_ref().map_or(&pieces, |c| &c.copies) {
                    list.CopyBufferRegion(&buffer, *at, &shadow.buffer, *lo as u64, (*hi - *lo) as u64);
                }
                list.Close()?;
                let lists = [Some(list.cast::<ID3D12CommandList>()?)];
                let _fetching = crate::timeline::span("landing: fetching pieces");
                if let Some(queue) = &copy_queue {
                    queue.ExecuteCommandLists(&lists);
                    queue.Signal(&fence, sent)?;
                    meanwhile();
                    fence.SetEventOnCompletion(sent, event)?;
                    #[cfg(feature = "diagnostics")]
                    let _card = crate::timeline::card_wait(None);
                    WaitForSingleObject(event, INFINITE);
                    return Ok(behind);
                }
                // In the order of the queue: the sends of title bytes so
                // far are ahead of this fetch, every later one behind it.
                let order = QUEUE_ORDER.lock().unwrap_or_else(|e| e.into_inner());
                self.queue.ExecuteCommandLists(&lists);
                let after = *order;
                drop(order);
                self.card_sent();
                self.queue.Signal(&fence, sent)?;
                meanwhile();
                fence.SetEventOnCompletion(sent, event)?;
                #[cfg(feature = "diagnostics")]
                let _card = crate::timeline::card_wait(None);
                WaitForSingleObject(event, INFINITE);
                Ok(after)
            })()
        };
        let after = match ran {
            Ok(after) => after,
            Err(e) => fail(format_args!("what shaders wrote to the card's memory could not be fetched ({e}, relay)")),
        };
        let us = waiting.elapsed().as_micros() as u64;
        RELAY_STATS[3].fetch_add(us, Ordering::Relaxed);
        {
            use crate::shadow::CARD_STATS;
            CARD_STATS[3].fetch_add(pieces.len() as u64, Ordering::Relaxed);
            CARD_STATS[4].fetch_add(total, Ordering::Relaxed);
            CARD_STATS[5].fetch_add(1, Ordering::Relaxed);
            CARD_STATS[6].fetch_add(us, Ordering::Relaxed);
        }
        let merging = std::time::Instant::now();
        let _merging = crate::timeline::span("landing: merging fetched pieces");
        let mut written = 0;
        if land::on() {
            let titles: Vec<(usize, usize)> =
                pieces.iter().map(|(shadow, lo, hi, _)| (shadow.span().0 as usize + lo, hi - lo)).collect();
            let jobs = land::jobs(&titles);
            let put = std::sync::atomic::AtomicU64::new(0);
            land::each(jobs.len(), &|j| {
                let working = std::time::Instant::now();
                if let Some(check) = &check {
                    for i in jobs[j].clone() {
                        // SAFETY: as below.
                        unsafe { check.look(i, data) };
                    }
                }
                let mut notes = land::Notes::default();
                let mut merged = 0;
                for run in pieces[jobs[j].clone()].chunk_by(|a, b| std::sync::Arc::ptr_eq(&a.0, &b.0)) {
                    let addressed = run.iter().map(|(_, lo, hi, at)| (*lo, *hi, data + *at as usize));
                    // SAFETY: the mapping of the fetched bytes, `total` of
                    // them; the copy into it has run, and the next waits
                    // for this call.
                    merged += unsafe { run[0].0.land_card(addressed, after, &mut notes) };
                }
                notes.flush();
                put.fetch_add(merged, Ordering::Relaxed);
                land::spent(land::stat::CARD_ALL, working);
            });
            land::count(land::stat::JOBS, jobs.len() as u64);
            written = put.into_inner();
        } else {
            let mut title = Vec::new();
            for (i, (shadow, lo, hi, at)) in pieces.iter().enumerate() {
                if let Some(check) = &check {
                    // SAFETY: as below.
                    unsafe { check.look(i, data) };
                }
                // SAFETY: the mapping of the fetched bytes, `total` of them;
                // the copy into it has run.
                let gpu = unsafe { std::slice::from_raw_parts((data + *at as usize) as *const u8, hi - lo) };
                let merged = shadow.card_merge_relay(*lo, gpu, after, &mut title);
                shadow.walls_merged(gpu.len(), merged);
                written += merged as u64;
            }
        }
        land::spent(land::stat::CARD, merging);
        COPY_BACK[1].fetch_add(written, Ordering::Relaxed);
        if let Some(f) = fetch.as_mut() {
            if let Some(check) = &check {
                check.done(&mut f.suspects);
            }
            f.had.fetched(
                ahead,
                pieces.iter().map(|(shadow, lo, hi, _)| (shadow.id(), shadow.span().0 as usize + lo, hi - lo)),
            );
        }
        // Sends up to `after` are of no use to any later merge.
        crate::shadow::relay_fetched(after);
    }

    /// Whether any GPU write copied back to title memory by `flights` overlaps
    /// a placed shader-visible descriptor heap, so subsequent stamped
    /// submissions wait for a heap-sync round begun after the landing completes.
    fn landed_touches_shader_heap(flights: &[&Flight]) -> bool {
        crate::descriptor::any_in_shader_heap(flights.iter().flat_map(|flight| {
            let back = flight.back.iter().flat_map(|(_, ranges)| ranges.iter().copied());
            let card = flight
                .card
                .iter()
                .flat_map(|card| card.ranges.iter().filter(|r| !r.picked).map(|r| r.declared));
            back.chain(card)
        }))
    }

    /// [`relay`], on the thread that lands flights: `items` are the first
    /// things of the line, flights the card has run and what follows them.
    /// What their shaders wrote goes into title memory (copies in the
    /// card's memory first, then the others, each byte once whichever
    /// flights declared it); then every flight's ranges are done with;
    /// then, in the line's order, each flight's read-backs, its
    /// bottom-of-pipe writes and the fence signals behind it. Title memory
    /// is complete before the title is told anything.
    fn relay_land(&self, items: Vec<Late>, fetch: &mut Option<RelayFetch>, texture_scratch: &mut crate::texback::Scratch) {
        let landing = std::time::Instant::now();
        let mut stage = crate::timeline::span("landing: what is to be looked through");
        {
            let flights: Vec<&Flight> = items
                .iter()
                .filter_map(|l| match l {
                    Late::Run(flight) => Some(flight),
                    _ => None,
                })
                .collect();
            if !flights.is_empty() {
                // SAFETY: a plain query.
                if unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
                    let runs: Vec<&HostList> = flights.iter().flat_map(|f| f.order.iter().map(|&i| f.lists[i].host())).collect();
                    fail(format_args!(
                        "host GPU device removed after {} submissions sent ahead (effective relay mode {}) of {} lists{}{}",
                        flights.len(),
                        relay_mode(),
                        runs.len(),
                        self.removal_data(&runs),
                        last_notes(&runs)
                    ));
                }
                let plans: Vec<(&CardPlan, u64)> = flights.iter().filter_map(|f| Some((f.card.as_ref()?, f.fence))).collect();
                // (Each byte once: the ranges are joined.)
                let mut back: Vec<Declared> = Vec::new();
                for flight in &flights {
                    for (shadow, ranges) in flight.back.iter().filter(|(shadow, _)| !shadow.carded()) {
                        match back.iter_mut().find(|(s, _)| std::sync::Arc::ptr_eq(s, shadow)) {
                            Some((_, all)) => all.extend_from_slice(ranges),
                            None => back.push((shadow.clone(), ranges.clone())),
                        }
                    }
                }
                let mut hand_back = relay_hand_back(&back);
                stage.next("landing: copies in the card's memory");
                self.relay_card_back(&plans, fetch, &mut || {
                    if crate::land::on() {
                        let _looking = crate::timeline::span("landing: looking through copies, beside the fetch");
                        hand_back.look(crate::land::stat::SCAN_BESIDE);
                    }
                });
                stage.next("landing: copies in the PC's memory");
                hand_back.merge();
                if Self::landed_touches_shader_heap(&flights) {
                    crate::port_journal::prep_wrote();
                }
                stage.next("landing: ranges done with");
                let finishing = std::time::Instant::now();
                // Only now, and flight by flight: a range another flight
                // in the air declared too stays closed to refreshes.
                for flight in &flights {
                    for (shadow, ranges) in &flight.back {
                        shadow.copy_back_done(ranges);
                    }
                }
                crate::land::spent(crate::land::stat::DONE, finishing);
                COPY_BACK[0].fetch_add(landing.elapsed().as_micros() as u64, Ordering::Relaxed);
                RELAY_STATS[0].fetch_add(1, Ordering::Relaxed);
                RELAY_STATS[1].fetch_add(flights.len() as u64, Ordering::Relaxed);
            }
        }
        // In the order given: what tells the title the card is done.
        stage.next("landing: read-backs, telling the title");
        let mut last = 0;
        let mut landed = Vec::new();
        #[cfg(feature = "diagnostics")]
        let mut retiring_slots = 0u64;
        let mut telling = std::time::Instant::now();
        for item in items {
            telling = match item {
                Late::Run(mut flight) => {
                    last = flight.number;
                    for &i in &flight.order {
                        read_backs(flight.lists[i].host(), texture_scratch);
                    }
                    if let Some(card) = flight.card {
                        self.card_done(card);
                    }
                    if let Some(own) = flight.own {
                        own.landed();
                    }
                    for detached in &mut flight.lists {
                        #[cfg(feature = "diagnostics")]
                        {
                            retiring_slots += detached.pending_dynamic_slots();
                        }
                        detached.release_tables();
                    }
                    landed.push((flight.lists, flight.order));
                    crate::land::spent(crate::land::stat::READ_BACKS, telling)
                }
                Late::Write(at, what, bytes) => {
                    what.write(at as *mut u8, bytes);
                    crate::land::spent(crate::land::stat::TOLD, telling)
                }
                Late::Signal(fence, value) => {
                    crate::fence::set(fence.as_this(), value);
                    crate::land::spent(crate::land::stat::TOLD, telling)
                }
            };
        }
        #[cfg(feature = "diagnostics")]
        LAND_RETIRE_STATS[4].fetch_max(retiring_slots, Ordering::Relaxed);
        if last > 0 {
            FINISHED.store(last, Ordering::Release);
            bury(last);
        }
        stage.next("landing: the rest");
        self.log_messages();
        crate::land::spent(crate::land::stat::REST, telling);
        RELAY_STATS[4].fetch_add(landing.elapsed().as_micros() as u64, Ordering::Relaxed);
        {
            let mut line = line();
            let Line {
                line: items,
                written,
                lists,
                settling,
                poke,
                landed: kept,
                in_hand,
                landing,
                ..
            } = &mut *line;
            kept.extend(landed);
            in_hand.clear();
            landing.clear();
            // What is still in the air, from the flights that are.
            written.clear();
            *lists = 0;
            let mut ranges = Vec::new();
            for item in items.iter() {
                if let Late::Run(flight) = item {
                    *lists += flight.order.len().max(1);
                    ranges.extend(flight.back.iter().flat_map(|(_, r)| r.iter().copied()));
                    ranges.extend(flight.texture_back_ranges.iter().copied());
                }
            }
            join_ranges(written, ranges);
            *settling = false;
            // The landed lists are reset, and what queues hold looked at,
            // by the helper thread (they take locks a waiter may hold).
            *poke = true;
        }
        LINE_MOVED.notify_all();
        LINE_LANDED.notify_all();
    }

    /// After a submission ran, for copies in the card's memory: the pieces
    /// of the declared ranges that the compare found differing (all of a
    /// range it did not compare) are fetched, with one more wait for the
    /// card, and what shaders wrote in them goes into title memory.
    /// `submitted` is the last value signalled on the fence. `seen` holds
    /// the pieces fetched already since the card was last waited for
    /// (several submissions settled together name the same ones).
    fn card_back(
        &self,
        plan: &CardPlan,
        submitted: &mut u64,
        seen: &mut std::collections::HashSet<(usize, usize, usize)>,
    ) {
        if plan.ranges.is_empty() {
            return;
        }
        let mut work = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
        let Some(work) = work.as_mut() else {
            return;
        };
        // SAFETY: the mapping of the results, which the compare (it has
        // run) wrote.
        let differs = unsafe { std::slice::from_raw_parts(work.differs.1 as *const u32, (work.differs.2 / 4) as usize) };
        // The pieces to fetch: the range, the offsets in its copy, and
        // where the bytes come down.
        let mut pieces: Vec<(usize, usize, usize, u64)> = Vec::new();
        let mut total = 0u64;
        plan.picked_found(differs);
        for (i, r) in plan.ranges.iter().enumerate().filter(|(_, r)| !r.picked) {
            for p in 0..r.span.pieces {
                if r.compared && !torn::flagged(differs[(r.at + p) as usize]) {
                    continue;
                }
                let (lo, hi) = r.shadow.card_piece(&r.span, p);
                if lo < hi && seen.insert((std::sync::Arc::as_ptr(&r.shadow) as usize, lo, hi)) {
                    pieces.push((i, lo, hi, total));
                    total += (hi - lo) as u64;
                }
            }
            crate::shadow::CARD_STATS[2].fetch_add(u64::from(r.span.pieces), Ordering::Relaxed);
        }
        let mut written = vec![0usize; plan.ranges.len()];
        if !pieces.is_empty() {
            if work.fetched.as_ref().is_none_or(|f| f.2 < total) {
                let size = total.next_multiple_of(16 << 20);
                work.fetched = self.shared_buffer(size).map(|(buffer, data)| (buffer, data as usize, size));
            }
            let Some((fetched, data, _)) = &work.fetched else {
                fail(format_args!("no memory to fetch what shaders wrote to the card's memory ({total} bytes)"));
            };
            let waiting = std::time::Instant::now();
            let (allocator, list) = &work.fetch;
            // SAFETY: the list's last run was waited for; live buffers; the
            // fence and event are ours, under the submission lock.
            let ran = unsafe {
                (|| -> windows::core::Result<()> {
                    allocator.Reset()?;
                    list.Reset(allocator, None)?;
                    for &(i, lo, hi, at) in &pieces {
                        list.CopyBufferRegion(fetched, at, &plan.ranges[i].shadow.buffer, lo as u64, (hi - lo) as u64);
                    }
                    list.Close()?;
                    self.queue.ExecuteCommandLists(&[Some(list.cast()?)]);
                    self.card_sent();
                    *submitted += 1;
                    self.queue.Signal(&self.fence, *submitted)?;
                    self.fence.SetEventOnCompletion(*submitted, self.event)?;
                    let was_working = working(false);
                    WaitForSingleObject(self.event, INFINITE);
                    stood(waiting);
                    working(was_working);
                    Ok(())
                })()
            };
            if let Err(e) = ran {
                fail(format_args!("what shaders wrote to the card's memory could not be fetched ({e})"));
            }
            {
                use crate::shadow::CARD_STATS;
                CARD_STATS[3].fetch_add(pieces.len() as u64, Ordering::Relaxed);
                CARD_STATS[4].fetch_add(total, Ordering::Relaxed);
                CARD_STATS[5].fetch_add(1, Ordering::Relaxed);
                CARD_STATS[6].fetch_add(waiting.elapsed().as_micros() as u64, Ordering::Relaxed);
            }
            if wide_merge() && pieces.len() >= 8 {
                // The pieces are apart from one another (each its own bytes
                // of its copy and of title memory): several at a time.
                let fetched = *data;
                let counts: std::sync::Arc<Vec<std::sync::atomic::AtomicUsize>> =
                    std::sync::Arc::new((0..plan.ranges.len()).map(|_| std::sync::atomic::AtomicUsize::new(0)).collect());
                let each = pieces.len().div_ceil(8);
                let jobs: Vec<Box<dyn FnOnce() + Send>> = pieces
                    .chunks(each)
                    .map(|chunk| {
                        let chunk: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize, usize, u64)> =
                            chunk.iter().map(|&(i, lo, hi, at)| (plan.ranges[i].shadow.clone(), i, lo, hi, at)).collect();
                        let counts = counts.clone();
                        Box::new(move || {
                            let mut title = Vec::new();
                            for (shadow, i, lo, hi, at) in chunk {
                                // SAFETY: the mapping of the fetched bytes;
                                // the copy into it has run, and the caller
                                // waits for this job.
                                let gpu = unsafe { std::slice::from_raw_parts((fetched + at as usize) as *const u8, hi - lo) };
                                let merged = shadow.card_merge(lo, gpu, &mut title);
                                shadow.walls_merged(gpu.len(), merged);
                                counts[i].fetch_add(merged, Ordering::Relaxed);
                            }
                        }) as Box<dyn FnOnce() + Send>
                    })
                    .collect();
                crate::shadow::wide_run(jobs);
                for (written, count) in written.iter_mut().zip(counts.iter()) {
                    *written += count.load(Ordering::Relaxed);
                }
            } else {
                let mut title = Vec::new();
                for &(i, lo, hi, at) in &pieces {
                    // SAFETY: the mapping of the fetched bytes, `total` of
                    // them; the copy into it has run.
                    let gpu = unsafe { std::slice::from_raw_parts((*data + at as usize) as *const u8, hi - lo) };
                    let merged = plan.ranges[i].shadow.card_merge(lo, gpu, &mut title);
                    plan.ranges[i].shadow.walls_merged(gpu.len(), merged);
                    written[i] += merged;
                }
            }
        }
        for (r, written) in plan.ranges.iter().zip(written) {
            COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
            let (from, to) = r.declared;
            #[cfg(feature = "diagnostics")]
            crate::copystat::compared(from, to - from, written);
            let _ = (from, to);
        }
    }

    /// Runs closed lists and waits for them, and with [`late`] for
    /// everything sent to the card before them, which is settled first.
    /// Returns the queue fence value covering these lists.
    pub(crate) fn execute(&self, runs: &[&HostList]) -> u64 {
        // Behind what the title's queues were given before it.
        crate::queue::before_sync();
        self.submit(runs, &[], false, None, None).fence
    }

    /// Runs a closed internal upload-only host list synchronously without
    /// draining title queues (`queue::before_sync`), so it is deadlock-safe
    /// when invoked from `VICTORY_THIN=1` worker threads. Settles in-flight
    /// GPU work and waits for the upload list's fence before returning.
    pub(crate) fn execute_sync_internal(&self, list: &HostList) {
        let _ = self.submit(&[list], &[], false, None, None);
    }

    /// A queue's submission with [`late`]: the host lists `lists`, taken
    /// from the title's, go to the card in `order` and are not waited for.
    /// They join the line of flights and are settled, with the others in
    /// the air, at the next point where what they do can be observed
    /// ([`late`] lists those points); their host lists are reset and kept
    /// for reuse then, as a queue does for the lists it ran.
    pub(crate) fn execute_late(&self, lists: Vec<crate::list::Detached>, order: Vec<usize>) {
        self.execute_parts_stamped(lists, order, &[], None);
    }

    /// [`Host::execute_late`] of parts of submissions in one: another begins
    /// at each place of `order` in `parts` ([`joins`]). `heap_sync` is an
    /// optional enqueue-time heap-sync stamp (`descriptor::heap_sync_stamp`)
    /// so back-to-back queued submissions share a single sync round.
    pub(crate) fn execute_parts_stamped(
        &self,
        lists: Vec<crate::list::Detached>,
        order: Vec<usize>,
        parts: &[usize],
        heap_sync: Option<u64>,
    ) {
        if order.is_empty() {
            crate::list::recycle(lists, &order);
            return;
        }
        late_threads();
        let preplan = {
            let runs: Vec<&HostList> = order.iter().map(|&i| lists[i].host()).collect();
            // (Before the lists' steps are looked at for a cut; `submit`
            // checks again, which costs a load for each list.)
            closed_for_execution(&runs);
            match card_steps::cut_or_plan(&runs) {
                card_steps::CutOrPlan::Cut(decision) => {
                    let cut = decision.at;
                    match split(lists, order, cut) {
                        Ok([(lists, order), (more, behind)]) => {
                            #[cfg(feature = "diagnostics")]
                            decision.applied();
                            let first: Vec<usize> = parts.iter().copied().filter(|&p| p < cut).collect();
                            let second: Vec<usize> = parts.iter().filter(|&&p| p > cut).map(|p| p - cut).collect();
                            self.execute_parts_stamped(lists, order, &first, heap_sync);
                            self.execute_parts_stamped(more, behind, &second, heap_sync);
                        }
                        Err((lists, order)) => self.send_late(lists, order, parts, heap_sync, None),
                    }
                    return;
                }
                card_steps::CutOrPlan::Plan(plan) => Some(plan),
                card_steps::CutOrPlan::None => None,
            }
        };
        self.send_late(lists, order, parts, heap_sync, preplan);
    }

    fn send_late(
        &self,
        lists: Vec<crate::list::Detached>,
        order: Vec<usize>,
        parts: &[usize],
        heap_sync: Option<u64>,
        preplan: Option<card_steps::Preplan>,
    ) {
        let sent = {
            let runs: Vec<&HostList> = order.iter().map(|&i| lists[i].host()).collect();
            self.submit(&runs, parts, true, heap_sync, preplan).ahead
        };
        let Some((mut submitted, sent)) = sent else {
            // (Not reached: a submission sent ahead always comes back.)
            crate::list::recycle(lists, &order);
            return;
        };
        let told = !sent.writes.is_empty();
        let sizes = crate::memory::asked_sizes(order.iter().map(|&i| lists[i].host()));
        let most = {
            let mut line = line();
            let Line { line: items, written, lists: flying, target, .. } = &mut *line;
            join_ranges(
                written,
                sent.back
                    .iter()
                    .flat_map(|(_, ranges)| ranges.iter().copied())
                    .chain(sent.texture_back_ranges.iter().copied()),
            );
            *flying += order.len();
            *target = *submitted;
            items.push(Late::Run(Flight {
                number: sent.number,
                fence: *target,
                lists,
                order,
                card: sent.card,
                back: sent.back,
                own: sent.own,
                texture_back_ranges: sent.texture_back_ranges,
            }));
            items.extend(sent.writes.into_iter().map(|(at, what, bytes)| Late::Write(at, what, bytes)));
            *flying >= late_most()
        };
        LATE_STATS[0].fetch_add(1, Ordering::Relaxed);
        RELAY_MOVED.notify_one();
        if told && !late_writes() {
            self.settle(&mut submitted, WHY_WRITES);
        } else if sizes {
            self.settle(&mut submitted, WHY_READ);
        } else if most && relay_exact() {
            // Until there are fewer, not until there are none.
            relay_wait(WHY_MOST, |line| line.lists < late_most());
        } else if most {
            self.settle(&mut submitted, WHY_MOST);
        } else {
            LINE_MOVED.notify_one();
        }
        drop(submitted);
        // Every push to `line.landed` (`relay_land` and `settle`) already sets
        // `line.poke` or runs `land()` directly on `victory-late-wait`, so
        // `send_late` only falls back to inline `land()` if the helper thread
        // failed to start.
        if !LATE_HELPER_LIVE.load(Ordering::Acquire) {
            let _landing = crate::timeline::span("landed lists reset and given back");
            #[cfg(feature = "diagnostics")]
            let started = std::time::Instant::now();
            land();
            #[cfg(feature = "diagnostics")]
            LAND_RETIRE_STATS[0].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
        }
    }

    /// Sends `list`, a closed list of ours (the present's copy), as a
    /// flight ([`late_free_present`]): not waited for, settled with the
    /// others. The caller keeps the list and does not reset it before the
    /// submission whose number comes back has finished ([`FINISHED`]); 0 if
    /// it ran and was waited for after all. The second value is the queue
    /// fence covering the list in either case.
    fn execute_own(&self, list: &HostList) -> (u64, u64) {
        late_threads();
        let Submission { fence, ahead: flight } = self.submit(&[list], &[], true, None, None);
        let Some((submitted, sent)) = flight else {
            return (0, fence);
        };
        let number = sent.number;
        {
            let mut line = line();
            let Line { line: items, written, lists: flying, target, .. } = &mut *line;
            join_ranges(
                written,
                sent.back
                    .iter()
                    .flat_map(|(_, ranges)| ranges.iter().copied())
                    .chain(sent.texture_back_ranges.iter().copied()),
            );
            *flying += 1;
            *target = *submitted;
            items.push(Late::Run(Flight {
                number,
                fence: *target,
                lists: Vec::new(),
                order: Vec::new(),
                card: sent.card,
                back: sent.back,
                own: sent.own,
                texture_back_ranges: sent.texture_back_ranges,
            }));
            items.extend(sent.writes.into_iter().map(|(at, what, bytes)| Late::Write(at, what, bytes)));
        }
        LATE_STATS[0].fetch_add(1, Ordering::Relaxed);
        LINE_MOVED.notify_one();
        RELAY_MOVED.notify_one();
        drop(submitted);
        (number, fence)
    }

    /// Waits for the card to run every flight in the air ([`late`]) and
    /// does what they left to do, as [`Host::execute`] does for one
    /// submission after its wait: the pieces their shaders wrote in copies
    /// in the card's memory are fetched and merged (each once), the ranges
    /// they declared in the other copies are compared and copied back
    /// (each byte once, whichever flights declared it), and then, in the
    /// order they were given: their bottom-of-pipe writes and the fence
    /// signals behind them. `submitted` is the submission lock's value;
    /// `why` one of the `WHY_` places. Nothing happens if nothing flies.
    ///
    /// Everything in the air is settled, never the first flights alone:
    /// the merge writes the copies' CPU side, which a flight still on the
    /// card may be writing too.
    fn settle(&self, submitted: &mut u64, why: usize) {
        if !late() {
            return;
        }
        if relay() {
            // Nothing is settled here: the thread that lands flights does
            // it, in order, and this waits until nothing is in the air and
            // nothing is being landed. Every point that settled before
            // comes here, so each still sees title memory, fences and
            // bottom-of-pipe values as after a settle.
            relay_wait(why, |line| line.line.is_empty() && !line.settling);
            return;
        }
        let was_working = working(true);
        let _settling = crate::timeline::span("settling flights");
        let mut counted = false;
        let mut texture_scratch = crate::texback::Scratch::default();
        loop {
            let items = {
                let mut line = line();
                if line.line.is_empty() {
                    line.settling = false;
                    break;
                }
                line.settling = true;
                line.written.clear();
                line.lists = 0;
                std::mem::take(&mut line.line)
            };
            let flights = items.iter().filter(|i| matches!(i, Late::Run(_))).count();
            if !counted {
                counted = true;
                LATE_STATS[1].fetch_add(1, Ordering::Relaxed);
                LATE_STATS[why].fetch_add(1, Ordering::Relaxed);
            }
            LATE_STATS[11].fetch_add(flights as u64, Ordering::Relaxed);
            if flights > 0 {
                submitting(2, flights);
                let waiting = std::time::Instant::now();
                // SAFETY: the fence and event are ours, under the
                // submission lock; `submitted` is the last value signalled.
                unsafe {
                    match self.fence.SetEventOnCompletion(*submitted, self.event) {
                        Ok(()) => {
                            working(false);
                            #[cfg(feature = "diagnostics")]
                            let _card = crate::timeline::card_wait(Some(*submitted));
                            WaitForSingleObject(self.event, INFINITE);
                            stood(waiting);
                            working(true);
                        }
                        Err(e) => tracing::warn!(error = %e, "submissions sent ahead not waited for"),
                    }
                }
                let report_removed = || {
                    let runs: Vec<&HostList> = items
                        .iter()
                        .filter_map(|i| match i {
                            Late::Run(f) => Some(f),
                            _ => None,
                        })
                        .flat_map(|f| f.order.iter().map(|&i| f.lists[i].host()))
                        .collect();
                    format!(
                        "host GPU device removed after {flights} submissions sent ahead (VICTORY_LATE_WAIT=1) of {} lists{}{}",
                        runs.len(), self.removal_data(&runs), last_notes(&runs)
                    )
                };
                self.verify_texture_readback_fence(*submitted, report_removed);
                let gpu = waiting.elapsed();
                LATE_STATS[12].fetch_add(gpu.as_micros() as u64, Ordering::Relaxed);
                LATE_WAITED[why.min(10)].fetch_add(gpu.as_micros() as u64, Ordering::Relaxed);
                SUBMISSION_US[1].fetch_add(gpu.as_micros() as u64, Ordering::Relaxed);
                #[cfg(feature = "diagnostics")]
                crate::copystat::card_waited(gpu.as_micros() as u64, || {
                    format!("{flights} submissions sent ahead, settled together")
                });
                // SAFETY: a plain query.
                if unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
                    fail(format_args!("{}", report_removed()));
                }
                submitting(3, flights);
                let copying = std::time::Instant::now();
                let mut seen = std::collections::HashSet::new();
                let mut back: Vec<Declared> = Vec::new();
                for item in &items {
                    let Late::Run(flight) = item else {
                        continue;
                    };
                    if let Some(card) = &flight.card {
                        self.card_back(card, submitted, &mut seen);
                    }
                    for (shadow, ranges) in &flight.back {
                        match back.iter_mut().find(|(s, _)| std::sync::Arc::ptr_eq(s, shadow)) {
                            Some((_, all)) => {
                                for range in ranges {
                                    if !all.contains(range) {
                                        all.push(*range);
                                    }
                                }
                            }
                            None => back.push((shadow.clone(), ranges.clone())),
                        }
                    }
                }
                hand_back(&back);
                let flights_ref: Vec<&Flight> = items
                    .iter()
                    .filter_map(|i| match i {
                        Late::Run(f) => Some(f),
                        _ => None,
                    })
                    .collect();
                if Self::landed_touches_shader_heap(&flights_ref) {
                    crate::port_journal::prep_wrote();
                }
                COPY_BACK[0].fetch_add(copying.elapsed().as_micros() as u64, Ordering::Relaxed);
            }
            // In the order given: what tells the title the card is done.
            let mut last = 0;
            let mut landed = Vec::new();
            #[cfg(feature = "diagnostics")]
            let mut retiring_slots = 0u64;
            for item in items {
                match item {
                    Late::Run(mut flight) => {
                        last = flight.number;
                        for &i in &flight.order {
                            read_backs(flight.lists[i].host(), &mut texture_scratch);
                        }
                        if let Some(card) = flight.card {
                            self.card_done(card);
                        }
                        for detached in &mut flight.lists {
                            #[cfg(feature = "diagnostics")]
                            {
                                retiring_slots += detached.pending_dynamic_slots();
                            }
                            detached.release_tables();
                        }
                        landed.push((flight.lists, flight.order));
                    }
                    Late::Write(at, what, bytes) => what.write(at as *mut u8, bytes),
                    Late::Signal(fence, value) => crate::fence::set(fence.as_this(), value),
                }
            }
            #[cfg(feature = "diagnostics")]
            LAND_RETIRE_STATS[4].fetch_max(retiring_slots, Ordering::Relaxed);
            if last > 0 {
                FINISHED.store(last, Ordering::Release);
                bury(last);
            }
            line().landed.extend(landed);
        }
        if counted {
            self.log_messages();
            submitting(0, 0);
            // What was settled may be what a queue stands at, and the
            // caller may not be one that looks (a read-back, a present).
            if why != WHY_CARD_DONE {
                line().poke = true;
                LINE_MOVED.notify_one();
            }
        }
        working(was_working);
    }

    /// For a submission of `runs` that refreshes the ranges `stale` of host
    /// copies, which flights may be writing and the title wrote too
    /// ([`stale_under_flights`]): the flights land first. `ranges` are
    /// those of the submission's one asking, which is made again.
    ///
    /// With [`card_steps`] only the flights that declared them: a range of
    /// a copy nobody declared any more is refreshed like any other. What
    /// they landed is title memory written, so everything is asked and
    /// looked at again; what is left after three times waits as before.
    fn settle_stale(
        &self,
        submitted: &mut u64,
        runs: &[&HostList],
        mut stale: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)>,
        ranges: &[(usize, usize)],
    ) {
        if relay_exact() && card_steps::stage() > 0 {
            for _ in 0..3 {
                #[cfg(feature = "diagnostics")]
                blame_stale(&stale);
                relay_wait(WHY_STALE, |_| stale.iter().all(|(shadow, from, to)| !shadow.writes_pending_in(*from, *to)));
                crate::watch::ask_once(ranges);
                stale = stale_under_flights(runs);
                if stale.is_empty() {
                    return;
                }
            }
        }
        #[cfg(feature = "diagnostics")]
        blame_stale(&stale);
        self.settle(submitted, WHY_STALE);
        crate::watch::ask_once(ranges);
    }

    /// [`Host::execute`], or with `late` the first half of
    /// [`Host::execute_late`]: the lists are prepared and sent, and what is
    /// to be done once they have run comes back with the submission lock
    /// still held.
    ///
    /// `parts`: the lists of `runs` at which another part begins
    /// ([`joins`]): those go to the queue in calls of their own, one after the
    /// other, as they would have.
    fn submit<'a>(
        &'a self,
        runs: &[&HostList],
        parts: &[usize],
        late: bool,
        heap_sync: Option<u64>,
        preplan: Option<card_steps::Preplan>,
    ) -> Submission<'a> {
        // A list is whole only when it is closed.
        closed_for_execution(runs);
        #[cfg(feature = "diagnostics")]
        let called = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        crate::copystat::submission_begins();
        // One submission at a time from preparation to copy-back: snapshots
        // and the ranges shaders write are per host copy, not per submission.
        let mut stage = crate::timeline::span("submit: the submission lock");
        let mut submitted = self.submitted.lock().unwrap_or_else(|e| e.into_inner());
        let was_working = working(true);
        submitting(1, runs.len());
        #[cfg(feature = "diagnostics")]
        let ahead = Split::begin();
        #[cfg(feature = "diagnostics")]
        SPLIT[11].fetch_add(called.elapsed().as_nanos() as u64, Ordering::Relaxed);
        stage.next("submit: steps given to the card");
        // Flights in the air ([`late`]): a submission that is waited for
        // settles them first, one sent ahead only if preparing it reads
        // what they write.
        // Or the reads are done on the card, behind the flights.
        #[cfg(feature = "diagnostics")]
        let mut redirect_wait = None;
        let own = if late {
            self.card_steps(
                runs, preplan,
                #[cfg(feature = "diagnostics")]
                &mut redirect_wait,
            )
        } else {
            None
        };
        #[cfg(feature = "diagnostics")]
        ahead.part(12);
        stage.next("submit: waiting for flights it reads from");
        if !late {
            self.settle(&mut submitted, WHY_SYNC);
        } else if own.is_none()
            && let Some(why) = late_blocked(runs)
        {
            #[cfg(feature = "diagnostics")]
            let timer = redirect_wait.map(|wait| (wait, std::time::Instant::now()));
            if relay_exact() && why == WHY_READ {
                // Only until the flights that may write what is read have
                // landed: their copy-back's end, not the card's other work.
                relay_wait(why, |line| {
                    blocked_by(&line.written, runs, false).is_none() && !card_steps::copied_read(&card_steps::copies(line), runs)
                });
            } else {
                self.settle(&mut submitted, why);
            }
            #[cfg(feature = "diagnostics")]
            if let Some((wait, began)) = timer {
                wait.finished(began.elapsed());
            }
        }
        #[cfg(feature = "diagnostics")]
        ahead.part(13);
        // Descriptors the title copied into placed heaps directly, before
        // command lists that index those slots are submitted.
        // Renderer port #7 (from renderer patch #4): the round asked for when
        // the title submitted it, if that is known (`crate::port_journal`),
        // otherwise the enqueue-time `heap_sync` stamp (`SyncRounds`).
        let heaps_on = crate::port_journal::on();
        let heaps_synced = match crate::port_journal::needed() {
            0 => crate::descriptor::begin_sync_heaps_for(heap_sync),
            mark => crate::descriptor::begin_sync_heaps_after(mark),
        };
        // That list runs first, and is one of the submission's from here on.
        let with: Vec<&HostList>;
        let runs = match &own {
            Some(own) => {
                with = std::iter::once(own.list()).chain(runs.iter().copied()).collect();
                &with[..]
            }
            None => runs,
        };
        let parts: Vec<usize> = parts.iter().map(|&n| n + usize::from(own.is_some())).collect();
        stage.next("submit: the sweep");
        let behind = late && flying();
        // Before this submission counts as started: what the sweep lets go
        // of is named by no list, and nothing runs now (behind flights the
        // sweep waits for a submission that finds none in the air: there is
        // one at every present).
        static SWEPT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        if !behind {
            if upcheck::on() {
                upcheck::at_rest(self);
            }
            sweep();
            SWEPT.store(now_ms(), Ordering::Relaxed);
        } else if late_free_present() && now_ms() > SWEPT.load(Ordering::Relaxed) + 1500 {
            // With the present sent ahead nothing else brings a moment
            // with nothing in the air: one is made for the sweep.
            self.settle(&mut submitted, WHY_SYNC);
            if upcheck::on() {
                upcheck::at_rest(self);
            }
            sweep();
            SWEPT.store(now_ms(), Ordering::Relaxed);
        }
        let behind = late && flying();
        stage.next("submit: gathering what to ask about");
        let this_submission = STARTED.fetch_add(1, Ordering::AcqRel) + 1;
        // The ranges shaders write are counted for it (with the relay).
        crate::shadow::preparing(this_submission);
        let started = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        ahead.part(14);
        #[cfg(feature = "diagnostics")]
        let lock = (started - called).as_micros() as u64;
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::Submissions, 1);
        let mut lists: Vec<Option<ID3D12CommandList>> = Vec::with_capacity(2 * runs.len());
        // Where each run's lists start in `lists`.
        let mut firsts = Vec::with_capacity(runs.len());
        // Diagnostics: where each list of `lists` starts and what it is, for
        // a timing run to time each on its own (`VICTORY_GPU_TIMING`).
        #[cfg(feature = "diagnostics")]
        let mut marks: Vec<(usize, crate::copystat::Timed)> = Vec::with_capacity(2 * runs.len() + 3);
        let mut batch = Batch::default();
        #[cfg(feature = "diagnostics")]
        let split = Split::begin();
        // The time a settle of flights takes inside this preparing: the
        // wait for the card and the copy-back have their own places on the
        // frame line (run 92 showed them twice, "preparing" 250 ms for 135).
        let mut settled = std::time::Duration::ZERO;
        // One write-watch asking for everything the submission refreshes
        // (`VICTORY_ASK_ONCE`): per copy, the span of its refreshed ranges.
        if crate::watch::once() {
            let mut spans: ByCopy<(usize, usize)> = ByCopy::default();
            for list in runs {
                for step in &list.deferred().copies {
                    if let Step::Refresh(shadow, address, len) = step
                        && shadow.asked_ahead()
                    {
                        let span = spans
                            .entry(std::sync::Arc::as_ptr(shadow) as usize)
                            .or_insert((*address, *address + *len));
                        *span = (span.0.min(*address), span.1.max(*address + *len));
                    }
                }
            }
            let ranges: Vec<(usize, usize)> = spans.into_values().collect();
            #[cfg(feature = "diagnostics")]
            SPLIT[8].fetch_add(split.so_far(), Ordering::Relaxed);
            stage.next("submit: asking the write watch");
            crate::watch::ask_once(&ranges);
            #[cfg(feature = "diagnostics")]
            split.part(0);
            stage.next("submit: title bytes under flights' writes");
            // Behind flights: the title wrote memory that a flight's shaders
            // may write too, and this submission refreshes it. The flights
            // are settled, and the asking made again: the copy-back wrote
            // title memory, which other copies of it must see.
            if upcheck::on() {
                upcheck::order(runs);
                if behind {
                    upcheck::reads(runs);
                }
            }
            let stale = if behind { stale_under_flights(runs) } else { Vec::new() };
            if !stale.is_empty() {
                let settling = std::time::Instant::now();
                self.settle_stale(&mut submitted, runs, stale, &ranges);
                settled += settling.elapsed();
            }
        } else if behind && !stale_under_flights(runs).is_empty() {
            let settling = std::time::Instant::now();
            self.settle(&mut submitted, WHY_STALE);
            settled += settling.elapsed();
        }
        #[cfg(feature = "diagnostics")]
        split.part(1);
        // What flights in the air may write.
        let air = line().written.clone();
        stage.next("submit: refreshes ahead of writes");
        if refresh_front() {
            batch.refresh_front(runs, &parts);
        } else if ahead_of_writes() {
            batch.refresh_ahead(runs, |list, i| own.as_ref().is_some_and(|own| own.takes(list, i)));
        }
        stage.next("submit: the lists' steps");
        #[cfg_attr(not(feature = "diagnostics"), allow(clippy::unused_enumerate_index))]
        for (_run, list) in runs.iter().enumerate() {
            list.run_deferred(&mut batch, &air, |i| own.as_ref().is_some_and(|own| own.takes(list, i)));
            firsts.push(lists.len());
            if let Some(prelude) = list.prelude() {
                #[cfg(feature = "diagnostics")]
                marks.push((lists.len(), crate::copystat::Timed::Prelude));
                lists.push(Some(prelude));
            }
            #[cfg(feature = "diagnostics")]
            marks.push((
                lists.len(),
                match &own {
                    Some(own) if std::ptr::eq(own.list(), *list) => crate::copystat::Timed::Steps,
                    _ => crate::copystat::Timed::Title(_run),
                },
            ));
            lists.push(list.list.cast().ok());
        }
        batch.fill();
        stage.next("submit: the last refreshes");
        batch.flush(None);
        crate::watch::ask_once_end();
        #[cfg(feature = "diagnostics")]
        split.part(2);
        stage.next("submit: snapshots");
        // The snapshots of the ranges shaders may write, if they were left
        // for now (`crate::shadow::wide_back`): the refreshes are done, the
        // lists have not gone to the card.
        {
            #[cfg(feature = "diagnostics")]
            let snapshots = std::time::Instant::now();
            crate::shadow::take_snapshots();
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::SnapshotUs, snapshots.elapsed().as_micros() as u64);
        }
        #[cfg(feature = "diagnostics")]
        split.part(3);
        stage.next("submit: copies in the card's memory");
        // Copies in the card's memory: what changed on their CPU side goes
        // up ahead of the lists, and a compare of what the shaders may
        // write follows them in the same submission.
        let flown = late && relay();
        if flown && self.card_results_low() {
            let settling = std::time::Instant::now();
            self.settle(&mut submitted, WHY_MOST);
            settled += settling.elapsed();
        }
        // What shaders may write through views they pick: last, so that no
        // refresh of this submission leaves it alone.
        let picked = crate::picked::declare(runs);
        for (shadow, at, len) in &picked.written {
            shadow.writing(*at, *len);
        }
        let declared = declared_writes(runs);
        let card = self.card_before(&declared, &picked, flown);
        // The sends are taken: from here on the references hold what the
        // copies in the lists put.
        for (shadow, at, bytes) in batch.placed.drain(..) {
            shadow.placed_copy(at, &bytes);
        }
        // Buffers decay to COMMON between ExecuteCommandLists calls. The
        // compare must start a new call in case a title list promoted one
        // to COPY_DEST. The trailing GPU timestamps still cover it.
        let compare_at = card.as_ref().and_then(|c| c.check.clone()).map(|check| {
            let at = lists.len();
            #[cfg(feature = "diagnostics")]
            marks.push((at, crate::copystat::Timed::Compare));
            lists.push(Some(check));
            at
        });
        #[cfg(feature = "diagnostics")]
        split.part(4);
        stage.next("submit: waiting for the heaps' records");
        let heaps_began = std::time::Instant::now();
        drop(heaps_synced);
        crate::port_journal::heaps_waited(heaps_began.elapsed().as_nanos() as u64, heaps_on);
        // The slots that waited for a texture's host copy, before the
        // lists that may read them run.
        crate::descriptor::make_waiting();
        #[cfg(feature = "diagnostics")]
        split.part(15);
        let prepared = std::time::Instant::now();
        stage.next("submit: sending");
        // When it runs on the card, for the timeline.
        #[cfg(feature = "diagnostics")]
        let stamp = stamps::take();
        #[cfg(feature = "diagnostics")]
        let compare_at = if let Some((ahead, behind)) = stamp.and_then(stamps::lists) {
            lists.insert(0, Some(ahead));
            marks.iter_mut().for_each(|m| m.0 += 1);
            marks.insert(0, (0, crate::copystat::Timed::Other));
            marks.push((lists.len(), crate::copystat::Timed::Other));
            lists.push(Some(behind));
            firsts.iter_mut().for_each(|first| *first += 1);
            compare_at.map(|at| at + 1)
        } else {
            compare_at
        };
        #[cfg(feature = "diagnostics")]
        let sending = crate::timeline::ticks();
        if late {
            *submitted += 1;
            #[cfg(feature = "diagnostics")]
            let compare_at = if let Some([front, behind]) = self.span_lists() {
                lists.insert(0, Some(front));
                lists.push(Some(behind));
                firsts.iter_mut().for_each(|first| *first += 1);
                compare_at.map(|at| at + 1)
            } else {
                compare_at
            };
            // SAFETY: closed lists; the fence is ours.
            let signalled = unsafe {
                for batch in submission_batches(&lists, &firsts, &parts, compare_at) {
                    self.queue.ExecuteCommandLists(batch);
                }
                self.queue.Signal(&self.fence, *submitted)
            };
            awake::sent(&self.device);
            #[cfg(feature = "diagnostics")]
            crate::timeline::sent(sending, *submitted, stamp);
            self.card_sent();
            stage.next("submit: noting what is in the air");
            if let Err(e) = signalled {
                fail(format_args!("a submission sent ahead could not be followed by a signal ({e})"));
            }
            mark_frame_preparations(runs);
            #[cfg(feature = "diagnostics")]
            split.part(5);
            let back = crate::picked::with(declared, &picked.written);
            let mut texture_back_ranges = Vec::new();
            for list in runs {
                texture_back_ranges.extend(list.deferred().readback_ranges());
            }
            let writes = runs.iter().flat_map(|l| l.deferred().writes.clone()).collect();
            SUBMISSION_US[0].fetch_add((prepared - started).saturating_sub(settled).as_micros() as u64, Ordering::Relaxed);
            submitting(0, 0);
            #[cfg(feature = "diagnostics")]
            crate::copystat::submission_ends(lock);
            let sent = Sent {
                number: this_submission,
                card,
                back,
                own,
                texture_back_ranges,
                writes,
            };
            working(was_working);
            return Submission { fence: *submitted, ahead: Some((submitted, sent)) };
        }
        let (title, comparison) = lists.split_at(compare_at.unwrap_or(lists.len()));
        // Diagnostics: each list on its own, timed (`VICTORY_GPU_TIMING`).
        #[cfg(feature = "diagnostics")]
        // (Each list on its own: a run's prelude, which brings its copies up
        // to date, apart from the run's commands; the compare apart from the
        // last run.)
        let (groups, comparison): (Vec<_>, &[_]) = if timing() {
            let timed = marks
                .iter()
                .zip(marks.iter().skip(1).map(|m| m.0).chain([lists.len()]))
                .filter(|&(&(a, _), b)| a < b)
                .map(|(&(a, kind), b)| (&lists[a..b], kind))
                .collect();
            (timed, &[])
        } else {
            (vec![(title, crate::copystat::Timed::Other)], comparison)
        };
        #[cfg(not(feature = "diagnostics"))]
        let groups = [(title, ())];
        let report_removed = || {
            format!(
                "host GPU device removed after a submission of {} lists{}{}",
                runs.len(), self.removal_data(runs), last_notes(runs)
            )
        };
        submitting(2, runs.len());
        for (_i, &(group, _kind)) in groups.iter().enumerate() {
            *submitted += 1;
            let started = std::time::Instant::now();
            // SAFETY: closed lists; the fence and event are ours.
            unsafe {
                self.queue.ExecuteCommandLists(group);
                if _i + 1 == groups.len() && !comparison.is_empty() {
                    self.queue.ExecuteCommandLists(comparison);
                }
                self.card_sent();
                let waited = self
                    .queue
                    .Signal(&self.fence, *submitted)
                    .and_then(|()| self.fence.SetEventOnCompletion(*submitted, self.event));
                #[cfg(feature = "diagnostics")]
                crate::timeline::sent(sending, *submitted, stamp);
                match waited {
                    Ok(()) => {
                        working(false);
                        #[cfg(feature = "diagnostics")]
                        let _card = crate::timeline::card_wait(Some(*submitted));
                        WaitForSingleObject(self.event, INFINITE);
                        stood(started);
                        working(true);
                    }
                    Err(e) => tracing::warn!(error = %e, "host submission not waited for"),
                }
            }
            self.verify_texture_readback_fence(*submitted, report_removed);
            // With timing on, record each list's time to signal separately.
            // For title lists, also add their commands' times by pipeline.
            #[cfg(feature = "diagnostics")]
            if timing() {
                let wall = started.elapsed().as_micros() as u64;
                let mut noted = 0.0;
                if let crate::copystat::Timed::Title(run) = _kind {
                    // SAFETY: a plain query.
                    let frequency = unsafe { self.queue.GetTimestampFrequency() }.unwrap_or(1);
                    for (ms, op) in runs[run].slowest(frequency, usize::MAX) {
                        crate::copystat::command_took(&op, ms);
                        noted += ms;
                    }
                }
                crate::copystat::list_took(_kind, wall, (noted * 1e3) as u64);
            }
            // A submission of one list was never named here (`groups.len()
            // > 1` was the test): the two 50 ms lists of the gameplay
            // frame, each a submission of its own, printed nothing.
            #[cfg(feature = "diagnostics")]
            if let (true, crate::copystat::Timed::Title(i)) = (timing() && started.elapsed() > std::time::Duration::from_millis(50), _kind) {
                // SAFETY: a plain query.
                let frequency = unsafe { self.queue.GetTimestampFrequency() }.unwrap_or(1);
                let commands = runs[i].notes.lock().unwrap_or_else(|e| e.into_inner()).len();
                tracing::warn!(ms = started.elapsed().as_millis(), commands, "slow host list");
                for (ms, op) in runs[i].slowest(frequency, 6) {
                    tracing::warn!("slow list command, {ms:.1} ms: {op}");
                }
            }
        }
        mark_frame_preparations(runs);
        let _ = &firsts;
        let fence = *submitted;
        FINISHED.store(this_submission, Ordering::Release);
        bury(this_submission);
        let gpu = prepared.elapsed();
        SUBMISSION_US[0].fetch_add((prepared - started).as_micros() as u64, Ordering::Relaxed);
        SUBMISSION_US[1].fetch_add(gpu.as_micros() as u64, Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        crate::copystat::card_waited(gpu.as_micros() as u64, || {
            let pipelines: std::collections::BTreeSet<String> = runs
                .iter()
                .flat_map(|l| l.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clone())
                .collect();
            let text = format!("{} lists, pipelines {pipelines:?}", runs.len());
            text.chars().take(1200).collect()
        });
        // The host resets a device whose submission runs for about 2 s.
        if gpu > std::time::Duration::from_millis(250) {
            let pipelines: std::collections::BTreeSet<String> = runs
                .iter()
                .flat_map(|l| l.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clone())
                .collect();
            tracing::warn!(ms = gpu.as_millis(), lists = runs.len(), ?pipelines, "slow host submission");
        }
        // SAFETY: a plain query.
        if unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
            fail(format_args!("{}", report_removed()));
        }
        submitting(3, runs.len());
        stage.next("submit: copying back");
        // What shaders wrote goes back before the bottom-of-pipe writes,
        // which tell the title the GPU is done; each range once per
        // submission, however many dispatches declared it (the world's
        // passes declare the same big ranges dozens of times).
        let copying = std::time::Instant::now();
        // For copies in the card's memory the compare has run there: the
        // pieces that differ are fetched and merged.
        if let Some(card) = &card {
            self.card_back(card, &mut submitted, &mut std::collections::HashSet::new());
        }
        let back = crate::picked::with(declared, &picked.written);
        hand_back(&back);
        if let Some(card) = card {
            self.card_done(card);
        }
        #[cfg(feature = "diagnostics")]
        if !back.is_empty() {
            crate::copystat::add(crate::copystat::C::BackSubmissions, 1);
        }
        // Nothing of this submission goes back to the title: no range of
        // shader writes, no value written when it is done. The wait for
        // such a one is only for order and for what the card still holds.
        #[cfg(feature = "diagnostics")]
        if back.is_empty() && runs.iter().all(|l| l.deferred().writes.is_empty()) {
            use crate::copystat::{C, add};
            add(C::NothingBack, 1);
            add(C::NothingBackUs, gpu.as_micros() as u64);
        }
        COPY_BACK[0].fetch_add(copying.elapsed().as_micros() as u64, Ordering::Relaxed);
        let mut texture_scratch = crate::texback::Scratch::default();
        for list in runs {
            // Read backs first: a value written at the bottom of the pipe
            // may be what tells the title that they are there.
            read_backs(list, &mut texture_scratch);
            for &(at, what, bytes) in &list.deferred().writes {
                what.write(at as *mut u8, bytes);
            }
        }
        self.log_messages();
        submitting(0, 0);
        #[cfg(feature = "diagnostics")]
        crate::copystat::submission_ends(lock);
        // With the relay: nothing is in the air now, so no fetch still to
        // be merged is older than the sends to card copies so far, and
        // their notes can go.
        if relay() {
            crate::shadow::relay_fetched(*QUEUE_ORDER.lock().unwrap_or_else(|e| e.into_inner()));
        }
        working(was_working);
        Submission { fence, ahead: None }
    }

    /// With `VICTORY_D3D12_DRED`: the faulting address and the allocations
    /// it lies in (by the names we give them) or the title memory copy just
    /// below it; for each list that did not finish, the command it stopped
    /// at and what the draws and dispatches in flight there run (or, in
    /// lists that never started, the pipelines they bind).
    fn removal_data(&self, runs: &[&HostList]) -> String {
        let Ok(dred) = self.device.cast::<ID3D12DeviceRemovedExtendedData1>() else {
            return String::new();
        };
        let mut text = String::new();
        let mut fault_va = None;
        // SAFETY: the outputs are the runtime's, valid while the device
        // lives; nodes are walked as linked, histories have their counts.
        unsafe {
            if let Ok(fault) = dred.GetPageFaultAllocationOutput1()
                && fault.PageFaultVA != 0
            {
                let va = fault.PageFaultVA;
                fault_va = Some(va);
                text.push_str(&format!("; page fault at {va:#x}"));
                for (what, head) in [
                    ("inside", fault.pHeadExistingAllocationNode),
                    ("inside freed", fault.pHeadRecentFreedAllocationNode),
                ] {
                    let mut names = Vec::new();
                    let mut node = head;
                    while let Some(n) = node.as_ref()
                        && names.len() < 8
                    {
                        let name = if n.ObjectNameW.is_null() {
                            "unnamed".to_owned()
                        } else {
                            n.ObjectNameW.to_string().unwrap_or_default()
                        };
                        names.push(format!("{name} (type {})", n.AllocationType.0));
                        node = n.pNext;
                    }
                    if !names.is_empty() {
                        text.push_str(&format!("; {what} {names:?}"));
                    }
                }
                text.push_str(&crate::shadow::near(va));
                text.push_str(&crate::descriptor::host::near_heaps(va));
                #[cfg(feature = "diagnostics")]
                text.push_str(&crate::book::at_loss(va));
            }
            // The plain output: the extended one (with marker texts) came
            // back without stopped lists on this host.
            if dred_faults() {
                return text;
            }
            let crumbs = match dred.GetAutoBreadcrumbsOutput() {
                Ok(crumbs) => crumbs,
                Err(e) => {
                    text.push_str(&format!("; no breadcrumbs ({e})"));
                    return text;
                }
            };
            // Every list's progress (done/count), in the runtime's order.
            let mut progress = Vec::new();
            let mut node = crumbs.pHeadAutoBreadcrumbNode;
            while let Some(p) = node.as_ref() {
                let done = p.pLastBreadcrumbValue.as_ref().copied().unwrap_or(0);
                let ours = p.pCommandList.as_ref().map(|l| l.as_raw());
                let index = runs.iter().position(|r| Some(r.list.as_raw()) == ours);
                progress.push(format!("{done}/{} {index:?}", p.BreadcrumbCount));
                node = p.pNext;
            }
            text.push_str(&format!("; breadcrumbs (done/count, list) {progress:?}"));
            let mut plain = crumbs.pHeadAutoBreadcrumbNode;
            while let Some(p) = plain.as_ref() {
                let done = p.pLastBreadcrumbValue.as_ref().copied().unwrap_or(0);
                if done < p.BreadcrumbCount && !p.pCommandHistory.is_null() {
                    let history =
                        std::slice::from_raw_parts(p.pCommandHistory, p.BreadcrumbCount as usize);
                    let Some(line) = stopped_command(history, done, p.BreadcrumbCount) else {
                        plain = p.pNext;
                        continue;
                    };
                    text.push_str(&line);
                    let list = p.pCommandList.as_ref().map(|l| l.as_raw());
                    let run = runs.iter().find(|r| Some(r.list.as_raw()) == list);
                    let named = [
                        (!p.pCommandListDebugNameW.is_null()).then(|| p.pCommandListDebugNameW.to_string().unwrap_or_default()),
                        (!p.pCommandListDebugNameA.is_null()).then(|| {
                            std::ffi::CStr::from_ptr(p.pCommandListDebugNameA.cast()).to_string_lossy().into_owned()
                        }),
                    ]
                    .into_iter()
                    .flatten()
                    .next()
                    .unwrap_or_default();
                    text.push_str(&format!(" of \"{named}\""));
                    // Not one of the caller's: by the name.
                    #[cfg(feature = "diagnostics")]
                    let by_name = list_number(&named).and_then(|n| {
                        let all = LIST_NOTES.lock().unwrap_or_else(|e| e.into_inner());
                        all.get(&n).map(|(notes, pipelines, _)| (notes.upgrade(), pipelines.upgrade()))
                    });
                    #[cfg(feature = "diagnostics")]
                    let indirects = run.map(|r| r.indirects.clone()).or_else(|| {
                        let n = list_number(&named)?;
                        LIST_NOTES.lock().unwrap_or_else(|e| e.into_inner()).get(&n)?.2.upgrade()
                    });
                    #[cfg(not(feature = "diagnostics"))]
                    #[allow(clippy::type_complexity)]
                    let by_name: Option<(Option<()>, Option<std::sync::Arc<Mutex<std::collections::BTreeSet<String>>>>)> = None;
                    #[cfg(feature = "diagnostics")]
                    let noted = run.map(|r| r.notes.clone()).or_else(|| by_name.as_ref().and_then(|b| b.0.clone()));
                    let bound = run.map(|r| r.pipelines.clone()).or_else(|| by_name.as_ref().and_then(|b| b.1.clone()));
                    // The k-th marker command is the k-th note.
                    #[cfg(feature = "diagnostics")]
                    let markers: Vec<(u32, String)> = noted.map_or_else(Vec::new, |notes| {
                        let notes = notes.lock().unwrap_or_else(|e| e.into_inner());
                        history
                            .iter()
                            .enumerate()
                            .filter(|(_, op)| **op == D3D12_AUTO_BREADCRUMB_OP_SETMARKER)
                            .zip(notes.iter())
                            .map(|((at, _), (note, range))| {
                                let place = fault_va.zip(*range).map(|(va, r)| fault_in(va, r));
                                (at as u32, format!("{note}{}", place.unwrap_or_default()))
                            })
                            .collect()
                    });
                    #[cfg(not(feature = "diagnostics"))]
                    let markers: Vec<(u32, String)> = Vec::new();
                    #[cfg(not(feature = "diagnostics"))]
                    let _ = fault_va;
                    if done > 0 {
                        text.push_str(&in_flight(history, &markers, done as usize));
                    }
                    #[cfg(feature = "diagnostics")]
                    if let Some(indirects) = indirects {
                        text.push_str(&indirects_around(history, done as usize, &indirects));
                    }
                    // (Without markers a list that stopped midway says what
                    // it may have been at only by what it binds.)
                    if (done == 0 || markers.is_empty())
                        && let Some(bound) = bound
                    {
                        let pipelines = bound.lock().unwrap_or_else(|e| e.into_inner());
                        text.push_str(&format!(", its pipelines {pipelines:?}"));
                    }
                }
                plain = p.pNext;
            }
        }
        text
    }

    /// Moves the debug layer's messages into our log; returns the first
    /// error among them.
    pub(crate) fn log_messages(&self) -> Option<String> {
        let Some(queue) = &self.messages else {
            return None;
        };
        let mut first_error = None;
        // SAFETY: each message is read into a buffer of the size the queue
        // reports, then the queue is emptied.
        unsafe {
            for i in 0..queue.GetNumStoredMessages() {
                let mut size = 0;
                if queue.GetMessage(i, None, &mut size).is_err() || size == 0 {
                    continue;
                }
                let mut buffer = vec![0u64; size.div_ceil(8)];
                let message = buffer.as_mut_ptr().cast::<D3D12_MESSAGE>();
                if queue.GetMessage(i, Some(message), &mut size).is_ok() {
                    let m = &*message;
                    let text = std::slice::from_raw_parts(m.pDescription, m.DescriptionByteLength);
                    let text = String::from_utf8_lossy(text);
                    let text = text.trim_end_matches('\0');
                    tracing::warn!(id = m.ID.0, severity = m.Severity.0, "D3D12: {text}");
                    if m.Severity.0 <= D3D12_MESSAGE_SEVERITY_ERROR.0 && first_error.is_none() {
                        first_error = Some(text.to_owned());
                    }
                }
            }
            queue.ClearStoredMessages();
        }
        first_error
    }

    /// A host render target view of `resource`, valid until it is recorded.
    pub(crate) fn rtv(
        &self,
        resource: Option<&ID3D12Resource>,
        desc: Option<&D3D12_RENDER_TARGET_VIEW_DESC>,
    ) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_rtv.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a slot of our heap and a live resource, or none with a
        // description (a typed null view).
        unsafe {
            let mut handle = self.rtvs.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (slot * self.rtv_size) as usize;
            self.device
                .CreateRenderTargetView(resource, desc.map(|d| d as *const _), handle);
            handle
        }
    }

    /// A host depth view of `resource`, valid until it is recorded.
    pub(crate) fn dsv(
        &self,
        resource: &ID3D12Resource,
        desc: Option<&D3D12_DEPTH_STENCIL_VIEW_DESC>,
    ) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_dsv.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a slot of our heap and a live resource.
        unsafe {
            let mut handle = self.dsvs.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (slot * self.dsv_size) as usize;
            self.device
                .CreateDepthStencilView(resource, desc.map(|d| d as *const _), handle);
            handle
        }
    }

    /// 1-descriptor shader-visible CBV/SRV/UAV fallback heap for directly-indexed root signatures.
    pub(crate) fn fallback_view_heap(&self) -> Option<&ID3D12DescriptorHeap> {
        self.fallback_view
            .get_or_init(|| {
                // SAFETY: plain 1-descriptor shader-visible heap + null SRV initialization.
                unsafe {
                    let heap: ID3D12DescriptorHeap = self
                        .device
                        .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                            Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                            NumDescriptors: 1,
                            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                            NodeMask: 0,
                        })
                        .ok()?;
                    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                        Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_UINT,
                        ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                            Buffer: D3D12_BUFFER_SRV::default(),
                        },
                    };
                    self.device.CreateShaderResourceView(
                        None,
                        Some(&desc),
                        heap.GetCPUDescriptorHandleForHeapStart(),
                    );
                    Some(heap)
                }
            })
            .as_ref()
    }

    /// 1-descriptor shader-visible Sampler fallback heap for directly-indexed root signatures.
    pub(crate) fn fallback_sampler_heap(&self) -> Option<&ID3D12DescriptorHeap> {
        self.fallback_sampler
            .get_or_init(|| {
                // SAFETY: plain 1-descriptor shader-visible sampler heap initialization.
                unsafe {
                    let heap: ID3D12DescriptorHeap = self
                        .device
                        .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                            Type: D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
                            NumDescriptors: 1,
                            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                            NodeMask: 0,
                        })
                        .ok()?;
                    let desc = D3D12_SAMPLER_DESC {
                        Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
                        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                        MaxLOD: f32::MAX,
                        ..Default::default()
                    };
                    self.device
                        .CreateSampler(&desc, heap.GetCPUDescriptorHandleForHeapStart());
                    Some(heap)
                }
            })
            .as_ref()
    }
}

impl HostList {
    /// The timestamp heap and its read-back buffer (diagnostics, timing on).
    #[cfg(feature = "diagnostics")]
    fn stamps(&self) -> Option<&(ID3D12QueryHeap, ID3D12Resource)> {
        if !timing() {
            return None;
        }
        self.stamps
            .get_or_init(|| {
                let device = &host()?.device;
                let mut heap: Option<ID3D12QueryHeap> = None;
                let mut back: Option<ID3D12Resource> = None;
                let readback = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: u64::from(STAMPS) * 8,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    ..Default::default()
                };
                // SAFETY: plain object creation.
                unsafe {
                    device
                        .CreateQueryHeap(
                            &D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: STAMPS, NodeMask: 0 },
                            &mut heap,
                        )
                        .ok()?;
                    device
                        .CreateCommittedResource(
                            &readback,
                            D3D12_HEAP_FLAG_NONE,
                            &desc,
                            D3D12_RESOURCE_STATE_COPY_DEST,
                            None,
                            &mut back,
                        )
                        .ok()?;
                }
                Some((heap?, back?))
            })
            .as_ref()
    }

    /// The noted commands that took longest on the GPU, longest first, as
    /// (milliseconds, text) (diagnostics, timing on, after the list ran).
    #[cfg(feature = "diagnostics")]
    fn slowest(&self, frequency: u64, n: usize) -> Vec<(f64, String)> {
        let notes = self.notes.lock().unwrap_or_else(|e| e.into_inner());
        let (Some((_, back)), count) = (self.stamps(), notes.len().min(STAMPS as usize - 1)) else {
            return Vec::new();
        };
        let mut data = std::ptr::null_mut();
        let range = D3D12_RANGE { Begin: 0, End: (count + 1) * 8 };
        // SAFETY: a read-back buffer the finished list resolved into.
        let stamps: Vec<u64> = unsafe {
            if back.Map(0, Some(&range), Some(&mut data)).is_err() {
                return Vec::new();
            }
            let s = std::slice::from_raw_parts(data.cast::<u64>(), count + 1).to_vec();
            back.Unmap(0, None);
            s
        };
        let mut times: Vec<(f64, String)> = (0..count)
            .map(|i| {
                let ticks = stamps[i + 1].saturating_sub(stamps[i]);
                (ticks as f64 * 1e3 / frequency.max(1) as f64, notes[i].0.chars().take(300).collect())
            })
            .collect();
        times.sort_by(|a, b| b.0.total_cmp(&a.0));
        times.truncate(n);
        times
    }

    pub(crate) fn close(&self) {
        self.close_as(Closing::Recorded);
    }

    /// Whether the list is closed: only then may it go to a queue.
    #[cfg(test)]
    pub(crate) fn is_closed(&self) -> bool {
        self.recording.is_closed()
    }

    /// The list, new or reset and open, is handed to a title's list whose
    /// recording a submission took, and stays open (`crate::list::detach`).
    pub(crate) fn left_open(&self) {
        self.recording.left_open(&OPEN_LISTS);
    }

    fn close_as(&self, how: Closing) {
        let _closing = crate::timeline::span("closing a host list");
        self.twins_to_depth();
        self.textures_back();
        let scratch = self.indirect_args.lock().unwrap_or_else(|e| e.into_inner()).clone();
        if let Some((scratch, _)) = scratch {
            self.indirect_common(&scratch);
        }
        self.batches_common();
        self.card_budget_out();
        self.take_coming();
        self.close_constant_snapshots();
        #[cfg(feature = "diagnostics")]
        if let Some((heap, back)) = self.stamps() {
            let count = (self.notes.lock().unwrap_or_else(|e| e.into_inner()).len() as u32).min(STAMPS - 1);
            // SAFETY: an open list, our heap and read-back buffer.
            unsafe {
                self.list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, count);
                self.list.ResolveQueryData(heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, count + 1, back, 0);
            }
        }
        // SAFETY: our list.
        if let Err(e) = unsafe { self.list.Close() } {
            fail(format_args!(
                "a host command list did not close ({e}): a command recorded into it was invalid"
            ));
        }
        self.recording.close(how, &OPEN_LISTS);
    }

    /// Closes a list that is open still, before it goes to a queue: the one
    /// a title's list was given when its recording was taken
    /// (`crate::list::detach`), submitted again with no reset in between.
    pub(crate) fn close_if_open(&self) {
        if !self.recording.is_closed() {
            self.close_as(Closing::Taken);
        }
    }

    /// `work` finds, on a thread of ours, refreshes that belong at this
    /// place among the list's steps; they are there when the list is closed.
    pub(crate) fn refreshes_coming(&self, work: impl FnOnce() -> Vec<aside::Refresh> + Send + 'static) {
        self.take_coming();
        let at = self.deferred().copies.len();
        let coming = aside::begin(work);
        *self.coming.lock().unwrap_or_else(|e| e.into_inner()) = Some((at, coming));
    }

    /// The refreshes that were still to come are put among the steps.
    fn take_coming(&self) {
        let Some((at, coming)) = self.coming.lock().unwrap_or_else(|e| e.into_inner()).take() else {
            return;
        };
        let found = coming.wait();
        let mut deferred = self.deferred();
        let at = at.min(deferred.copies.len());
        deferred
            .copies
            .splice(at..at, found.into_iter().map(|(shadow, from, to)| Step::Refresh(shadow, from as usize, (to - from) as usize)));
    }

    /// Only after the list's last submission finished (every one is waited
    /// for).
    pub(crate) fn reset(&self) {
        self.changed_something();
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        let _resetting = crate::timeline::span("resetting a host list");
        self._alive.used();
        if self.recording.reset(&OPEN_LISTS) {
            // SAFETY: the title is discarding an unsubmitted recording, or
            // resets the list it was left open (`crate::list::detach`). Its
            // native list must close before its allocator can be reset.
            if let Err(e) = unsafe { self.list.close_discarded() } {
                fail(format_args!("a discarded host command list did not close ({e})"));
            }
        }
        *self.coming.lock().unwrap_or_else(|e| e.into_inner()) = None;
        let texture_backs = {
            let mut deferred = self.deferred();
            let texture_backs = std::mem::take(&mut deferred.texture_backs);
            let constant_write_version = deferred.constant_write_version.wrapping_add(1);
            *deferred = Deferred::default();
            deferred.constant_write_version = constant_write_version;
            texture_backs
        };
        self.states.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.twins.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clear();
        #[cfg(feature = "diagnostics")]
        self.indirects.lock().unwrap_or_else(|e| e.into_inner()).clear();
        #[cfg(feature = "diagnostics")]
        self.notes.lock().unwrap_or_else(|e| e.into_inner()).clear();
        {
            let mut scratch = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
            (scratch.current, scratch.used) = (0, 0);
            // Every submission of the list has run: nothing reads its
            // scratch memory any more, and the next list to ask takes it.
            if scratch_pool() {
                give_chunks(std::mem::take(&mut scratch.chunks));
            }
        }
        // SAFETY: the list is not executing.
        let reset = unsafe {
            self.allocator
                .Reset()
                .and_then(|()| self.list.Reset(&self.allocator, None))
        };
        if let Err(e) = reset {
            fail(format_args!("a host command list did not reset ({e})"));
        }
        // Reuse only after the old native commands have been discarded.
        drop(texture_backs);
        self.prelude.recycle(&prelude_pool::IDLE);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::ResetNs, started.elapsed().as_nanos() as u64);
    }

    /// Something changed in this list that bringing a draw's textures and
    /// buffers up to date depends on: a texture's state, a twin, what was
    /// refreshed.
    pub(crate) fn changed_something(&self) {
        self.epoch.store(next_epoch(), Ordering::Relaxed);
    }

    /// See [`HostList::changed_something`].
    pub(crate) fn epoch(&self) -> u64 {
        self.epoch.load(Ordering::Relaxed)
    }

    /// Whether any deferred steps or texture state transitions have been
    /// recorded into this list since creation or its last [`HostList::reset`].
    pub(crate) fn has_recorded_work(&self) -> bool {
        !self.deferred().copies.is_empty()
            || !self.states.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
    }

    /// The backing's identity and the last state this list recorded for it.
    pub(crate) fn texture_binding(
        &self,
        copy: &std::sync::Arc<crate::resource::HostTexture>,
    ) -> (usize, Option<u32>) {
        let state = self
            .states
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&(std::sync::Arc::as_ptr(copy) as usize))
            .map(|(_, last, _)| *last);
        (copy.texture.as_raw() as usize, state)
    }

    /// Moves a title texture's host copy to `wanted` (a host state) in this
    /// list: the host texture and the state it leaves, if that is a change.
    /// The list's first use of a texture records nothing: its prelude brings
    /// the texture there at submission.
    pub(crate) fn transition(
        &self,
        resource: crate::object::This,
        wanted: u32,
    ) -> Option<(&'static ID3D12Resource, u32)> {
        let copy = crate::resource::host_copy(resource)?;
        // SAFETY: as in `crate::resource::host_resource`; the list's states
        // hold the copy from here on.
        let texture: &'static ID3D12Resource = unsafe { &*std::ptr::from_ref(&copy.texture) };
        let wanted = legal_for(copy.flags(), wanted);
        let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        match states.entry(std::sync::Arc::as_ptr(&copy) as usize) {
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert((wanted, wanted, copy));
                self.changed_something();
                None
            }
            std::collections::hash_map::Entry::Occupied(mut o) => {
                let before = std::mem::replace(&mut o.get_mut().1, wanted);
                if before != wanted {
                    self.changed_something();
                }
                (before != wanted).then_some((texture, before))
            }
        }
    }

    /// Moves host texture `copy` to `wanted` (a host state) in this list,
    /// as [`HostList::transition`] does a title resource's host copy, and
    /// records the barrier: for textures that are not a title resource's
    /// host copy (colour twins) and for the copies between the two.
    fn settle(&self, copy: &std::sync::Arc<crate::resource::HostTexture>, wanted: u32) {
        let wanted = legal_for(copy.flags(), wanted);
        self.changed_something();
        let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        let before = match states.entry(std::sync::Arc::as_ptr(copy) as usize) {
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert((wanted, wanted, copy.clone()));
                None
            }
            std::collections::hash_map::Entry::Occupied(mut o) => {
                let before = std::mem::replace(&mut o.get_mut().1, wanted);
                (before != wanted).then_some(before)
            }
        };
        drop(states);
        if let Some(before) = before {
            let barrier = transition(
                &copy.texture,
                D3D12_RESOURCE_STATES(before as i32),
                D3D12_RESOURCE_STATES(wanted as i32),
            );
            // SAFETY: an open list and a live texture.
            unsafe { self.list.ResourceBarrier(&[barrier]) };
        }
    }

    /// A use of title texture `resource` about to be recorded, if it is a
    /// depth texture with a colour twin (false if it is not): of the twin
    /// (`colour`: unordered access) or of the depth texture (depth target,
    /// clear, shader read, copy, resolve), writing it or only reading. The
    /// side used is first brought up to date: by a copy recorded here if
    /// this list wrote the other side since, by the list's prelude if the
    /// list uses the side as it finds it. The twin is left in the
    /// unordered access state; the caller moves the depth texture to the
    /// state its use needs, as for any texture. A twin still ahead when the
    /// list closes is copied then (`twins_to_depth`).
    /// shortcut: every depth target bound for a draw counts as written
    /// (also when the draw only tests), and every unordered access view a
    /// shader names as written; views a shader picks from the heap by a
    /// computed index are not seen here at all.
    pub(crate) fn twin_use(
        &self,
        resource: crate::object::This,
        colour: bool,
        write: bool,
    ) -> bool {
        const UNORDERED_ACCESS: u32 = 0x8;
        // Asked for every texture a draw reads: most have no twin.
        let Some(twin) = crate::resource::twin(resource) else {
            return false;
        };
        let Some(depth) = crate::resource::host_copy(resource) else {
            return false;
        };
        let side = usize::from(colour);
        let mut twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        let u = twins
            .entry(std::sync::Arc::as_ptr(&twin) as usize)
            .or_insert_with(|| TwinUse {
                depth: depth.clone(),
                twin: twin.clone(),
                needs: [false; 2],
                behind: [None; 2],
                writing: false,
                late: false,
            });
        let was = (u.needs, u.behind, u.writing);
        let copy = match u.behind[side] {
            Some(behind) => behind,
            None => {
                u.needs[side] = true;
                false
            }
        };
        u.behind[side] = Some(false);
        match (write, colour) {
            // Shaders write the twin when the draw or dispatch runs: what
            // the same one reads of the depth texture is prepared before
            // (`twins_written`).
            (true, true) => u.writing = true,
            (true, false) => u.behind[1] = Some(true),
            (false, _) => {}
        }
        if was != (u.needs, u.behind, u.writing) || copy || colour {
            self.changed_something();
        }
        // shortcut: the depth plane's twin is not drawn into the plane
        // here: the pass sets a pipeline and a target of ours, which a list
        // still recording the title's commands cannot take. This use sees
        // the plane as it was; the pass runs when the list closes
        // (`twins_to_depth`). Counted.
        if copy && !colour && twin.plane.is_some() {
            u.late = true;
            crate::resource::PLANES[2].fetch_add(1, Ordering::Relaxed);
        }
        drop(twins);
        if copy {
            self.twin_copy(&depth, &twin, colour);
        } else if colour {
            self.settle(&twin.colour, UNORDERED_ACCESS);
        }
        // The depth plane's twin is written through unordered access too.
        if colour && let Some(plane) = &twin.plane {
            self.settle(plane, UNORDERED_ACCESS);
        }
        true
    }

    /// Records the pass that draws the twin of the depth plane of `depth`
    /// into the plane, if `twin` has one, with the moves to the states the
    /// pass needs; the twin is left in the unordered access state. Only
    /// for a list that records nothing of the title's afterwards
    /// (`record_plane_pass`).
    fn pass_plane(
        &self,
        depth: &std::sync::Arc<crate::resource::HostTexture>,
        twin: &crate::resource::Twin,
    ) {
        const UNORDERED_ACCESS: u32 = 0x8;
        const DEPTH_WRITE: u32 = 0x10;
        const PIXEL_SHADER_RESOURCE: u32 = 0x80;
        let Some(plane) = &twin.plane else { return };
        self.settle(depth, DEPTH_WRITE);
        self.settle(plane, PIXEL_SHADER_RESOURCE);
        record_plane_pass(self, &depth.texture, &plane.texture, &twin.plane_views);
        self.settle(plane, UNORDERED_ACCESS);
    }

    /// Records the copy of a depth texture into its colour twin (`colour`)
    /// or of the twin into the depth texture, with the moves to the copy
    /// states. The twin is left in the unordered access state.
    fn twin_copy(
        &self,
        depth: &std::sync::Arc<crate::resource::HostTexture>,
        twin: &crate::resource::Twin,
        colour: bool,
    ) {
        const UNORDERED_ACCESS: u32 = 0x8;
        const COPY_DEST: u32 = 0x400;
        const COPY_SOURCE: u32 = 0x800;
        let (from, to) = if colour {
            (depth, &twin.colour)
        } else {
            (&twin.colour, depth)
        };
        self.settle(from, COPY_SOURCE);
        self.settle(to, COPY_DEST);
        record_twin_copy(&self.list, &depth.texture, twin, colour);
        self.settle(&twin.colour, UNORDERED_ACCESS);
    }

    /// Before the list closes: the depth textures whose twins it wrote last
    /// are given what was written, and left readable by shaders. Shaders
    /// that pick a view from the heap by a computed index are not seen by
    /// `twin_use` and read the depth texture in lists that never name it
    /// (in a captured frame of the title no table names a shadow
    /// map for reading but the pass that fills the next one).
    fn twins_to_depth(&self) {
        const SHADER_RESOURCE: u32 = 0x40 | 0x80;
        let mut twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        let written: Vec<_> = twins
            .values_mut()
            .filter(|u| u.behind[0] == Some(true) || u.late)
            .map(|u| {
                // Not copied again if the list copied since the last write
                // and only the depth plane's pass is left.
                let copy = u.behind[0] == Some(true);
                u.behind[0] = Some(false);
                u.late = false;
                (u.depth.clone(), u.twin.clone(), copy)
            })
            .collect();
        drop(twins);
        for (depth, twin, copy) in &written {
            if *copy {
                self.twin_copy(depth, twin, false);
            }
            // Nothing of the title's is recorded after this: the pass may
            // leave its own pipeline and target set.
            self.pass_plane(depth, twin);
            self.settle(depth, SHADER_RESOURCE);
        }
    }

    /// A draw or dispatch being prepared writes title texture `resource`
    /// through unordered access: if it is of a kind copied back
    /// ([`crate::texback`]), the list copies it when it closes.
    pub(crate) fn texture_written(&self, resource: crate::object::This) {
        if !crate::texback::wanted(resource) {
            return;
        }
        let mut deferred = self.deferred();
        if !deferred.back_wanted.iter().any(|target| target.owner() == resource as usize)
            && let Some(target) = crate::texback::Target::new(resource)
        {
            deferred.back_wanted.push(target);
        }
    }

    /// Before the list closes: the textures its shaders wrote that are copied
    /// back go into read-back buffers ([`crate::texback::record`]).
    fn textures_back(&self) {
        const COPY_SOURCE: u32 = 0x800;
        let wanted = std::mem::take(&mut self.deferred().back_wanted);
        for target in wanted {
            self.settle(&target.copy, COPY_SOURCE);
            if let Some(back) = crate::texback::record(&self.list, target) {
                self.deferred().texture_backs.push(back);
            }
        }
    }

    /// The draw or dispatch just prepared is recorded: the twins it writes
    /// now hold what their depth textures lack.
    pub(crate) fn twins_written(&self) {
        let mut twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        for u in twins.values_mut().filter(|u| u.writing) {
            u.writing = false;
            u.behind[0] = Some(true);
            self.changed_something();
        }
    }

    /// At submission: the list that moves each texture this list uses from
    /// its real state to the one the list starts with; the textures' real
    /// states become those the list ends with. For depth textures with a
    /// colour twin it first copies the side the list uses as it finds it
    /// from the other, if a list submitted earlier left it behind.
    fn prelude(&self) -> Option<ID3D12CommandList> {
        const COPY_DEST: u32 = 0x400;
        const COPY_SOURCE: u32 = 0x800;
        let host_state = |state: u32| D3D12_RESOURCE_STATES(state as i32);
        let states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        // The copies (depth texture, twin, whether into the twin); the
        // sides are then as this list leaves them.
        let mut syncs = Vec::new();
        let twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        for u in twins.values() {
            let real = u.twin.behind.load(Ordering::Relaxed);
            let mut behind = [real & 1 != 0, real & 2 != 0];
            for side in [0, 1] {
                if u.needs[side] && behind[side] {
                    syncs.push((u.depth.clone(), u.twin.clone(), side == 1));
                    behind = [false; 2];
                }
            }
            let left = [0, 1].map(|side| u.behind[side].unwrap_or(behind[side]));
            let left = u32::from(left[0]) | (u32::from(left[1]) << 1);
            u.twin.behind.store(left, Ordering::Relaxed);
        }
        drop(twins);
        // A copy runs with its textures in the copy states: each moves
        // there from its real state, and from there to the state the list
        // starts with (back to its real state if the list does not use it).
        let mut to_copy = Vec::new();
        let mut barriers = Vec::new();
        let mut copying = std::collections::HashMap::new();
        for (depth, twin, colour) in &syncs {
            let (from, to) = if *colour {
                (depth, &twin.colour)
            } else {
                (&twin.colour, depth)
            };
            for (texture, state) in [(from, COPY_SOURCE), (to, COPY_DEST)] {
                let key = std::sync::Arc::as_ptr(texture) as usize;
                let real = texture.state.load(Ordering::Relaxed);
                copying.insert(key, state);
                if real != state {
                    to_copy.push(transition(
                        &texture.texture,
                        host_state(real),
                        host_state(state),
                    ));
                    if !states.contains_key(&key) {
                        barriers.push(transition(
                            &texture.texture,
                            host_state(state),
                            host_state(real),
                        ));
                    }
                }
            }
        }
        barriers.extend(states.iter().filter_map(|(key, (first, last, copy))| {
            let real = copy.state.swap(*last, Ordering::Relaxed);
            let real = copying.get(key).copied().unwrap_or(real);
            (real != *first)
                .then(|| transition(&copy.texture, host_state(real), host_state(*first)))
        }));
        if barriers.is_empty() && syncs.is_empty() {
            return None;
        }
        self.prelude
            .record(&prelude_pool::IDLE, |(allocator, list)| {
                // SAFETY: the prelude is not executing and its textures are live.
                let recorded = unsafe {
                    allocator.Reset().and_then(|()| {
                        list.Reset(allocator, None)?;
                        if !to_copy.is_empty() {
                            list.ResourceBarrier(&to_copy);
                        }
                        for (depth, twin, colour) in &syncs {
                            record_twin_copy(list, &depth.texture, twin, *colour);
                        }
                        if !barriers.is_empty() {
                            list.ResourceBarrier(&barriers);
                        }
                        list.Close()
                    })
                };
                if let Err(e) = recorded {
                    fail(format_args!("a host state prelude did not record ({e})"));
                }
                list.cast().ok()
            })
            .unwrap_or_else(|| fail(format_args!("no host list for a state prelude")))
    }

    /// Notes what the command about to be recorded does, as a marker
    /// command whose text the list keeps (diagnostics, with device-removed
    /// data on): the report pairs the breadcrumbs' marker commands with the
    /// texts in order and names the commands a fault stopped among.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_op(&self, what: impl FnOnce() -> String) {
        self.note(what, None);
    }

    /// As [`HostList::note_op`], for a copy that reads the host GPU range
    /// `(address, bytes)`: the report places the faulting address in it.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_read(&self, what: impl FnOnce() -> String, range: (u64, u64)) {
        self.note(what, Some(range));
    }

    #[cfg(feature = "diagnostics")]
    fn note(&self, what: impl FnOnce() -> String, range: Option<(u64, u64)>) {
        if !dred_on() {
            crate::capture::op(std::ptr::from_ref(self) as usize, what);
            return;
        }
        {
            let started = std::time::Instant::now();
            let at = self.notes.lock().unwrap_or_else(|e| e.into_inner()).len() as u32;
            if let Some((heap, _)) = self.stamps() && at < STAMPS - 1 {
                // SAFETY: an open list and our query heap.
                unsafe { self.list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, at) };
            }
            let what = what();
            crate::capture::op(std::ptr::from_ref(self) as usize, || what.clone());
            let text: Vec<u16> = what.encode_utf16().chain([0]).collect();
            const UNICODE: u32 = 0;
            // SAFETY: an open list; the marker is copied when recorded.
            unsafe {
                self.list.SetMarker(UNICODE, Some(text.as_ptr().cast()), 2 * text.len() as u32)
            };
            self.notes
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push((what, range));
            crate::copystat::add(crate::copystat::C::NoteNs, started.elapsed().as_nanos() as u64);
        }
    }

    /// Whether nothing recorded since the list was opened does any work:
    /// no draw or dispatch (the first keeps its pipeline), no copy, clear,
    /// upload or resolve (they defer a step or move a texture's state), no
    /// bottom-of-pipe write. State set on the list does not count.
    pub(crate) fn untouched(&self) -> bool {
        {
            let deferred = self.deferred();
            if !(deferred.copies.is_empty()
                && deferred.uploads.is_empty()
                && deferred.kept.is_empty()
                && deferred.shared.is_empty()
                && deferred.frame_preparation.is_empty()
                && deferred.textures.is_empty()
                && deferred.writes.is_empty()
                && deferred.backs.is_empty()
                && deferred.texture_backs.is_empty())
            {
                return false;
            }
        }
        self.states.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
            && self.twins.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
    }

    /// Notes an indirect execution, while device-removed data is kept
    /// ([`indirects`]).
    #[cfg(feature = "diagnostics")]
    pub(crate) fn ran_indirect(&self, pipeline: usize, name: &str, kind: u32, count: u32, user: &[u32]) {
        if dred_on() || dred_light() {
            self.indirects.lock().unwrap_or_else(|e| e.into_inner()).note(pipeline, name, kind, count, user);
        }
    }

    /// Notes a pipeline the list binds.
    pub(crate) fn used_pipeline(&self, name: &str) {
        let mut pipelines = self.pipelines.lock().unwrap_or_else(|e| e.into_inner());
        if !pipelines.contains(name) {
            pipelines.insert(name.to_owned());
        }
    }

    /// The list binds a pipeline: it and its root signature are kept until
    /// the list is reset, and its name noted, the first time only (a kept
    /// pipeline's address names no other).
    pub(crate) fn bind_pipeline(&self, name: &str, pso: &ID3D12PipelineState, root: &ID3D12RootSignature) {
        {
            let mut deferred = self.deferred();
            if !deferred.bound.insert(pso.as_raw() as usize) {
                return;
            }
            deferred.kept.push(pso.clone().into());
            deferred.kept.push(root.clone().into());
        }
        self.used_pipeline(name);
    }

    fn deferred(&self) -> std::sync::MutexGuard<'_, Deferred> {
        self.deferred.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Copies `len` bytes from `src` to `dst` when the list is submitted.
    pub(crate) fn defer_copy(&self, dst: *mut u8, src: *const u8, len: usize) {
        let mut deferred = self.deferred();
        COPIES_JOINED[0].fetch_add(1, Ordering::Relaxed);
        // A copy that goes on where the one recorded just before it ended,
        // source and destination both: with `join_copies` the two are one
        // step.
        if let Some(Step::Copy(d, s, n)) = deferred.copies.last_mut()
            && continues((*d, *s, *n), (dst as usize, src as usize, len))
        {
            COPIES_JOINED[1].fetch_add(1, Ordering::Relaxed);
            if join_copies() {
                *n += len;
                return;
            }
        }
        deferred
            .copies
            .push(Step::Copy(dst as usize, src as usize, len));
        deferred.title_written(None);
        self.changed_something();
    }

    /// Copies `len` bytes of title memory at `src` into the list's upload
    /// memory at `dst` when the list is submitted.
    pub(crate) fn defer_fill(&self, dst: *mut u8, src: *const u8, len: usize) {
        self.deferred()
            .copies
            .push(Step::Fill(dst as usize, src as usize, len, None));
    }

    /// The host address of `len` bytes of scratch memory that hold the title
    /// memory at `src` as of the list's submission: one fill for every ask
    /// until the list next writes title memory.
    pub(crate) fn fill_scratch(&self, src: *const u8, len: u64) -> Option<u64> {
        let key = (src as usize, len as usize);
        let known = self.deferred().filled.get(&key).copied();
        if known.is_some() {
            return known;
        }
        let (gpu, to) = self.scratch(len)?;
        self.defer_fill(to, src, len as usize);
        self.deferred().filled.insert(key, gpu);
        Some(gpu)
    }

    /// As [`HostList::defer_fill`], for upload memory that only `slot` (a
    /// title heap's address, and the slot) shows.
    pub(crate) fn defer_constants(&self, dst: *mut u8, src: *const u8, len: usize, slot: (usize, u32)) {
        self.deferred()
            .copies
            .push(Step::Fill(dst as usize, src as usize, len, Some(card_steps::FillTo::Slot(slot.0, slot.1))));
    }

    /// As [`HostList::defer_fill`], for the beginning of a buffer
    /// [`HostList::upload`] made.
    pub(crate) fn defer_fill_buffer(&self, buffer: &ID3D12Resource, dst: *mut u8, src: *const u8, len: usize) {
        let to = (card_steps::stage() > 1).then(|| card_steps::FillTo::Buffer(buffer.clone()));
        self.deferred().copies.push(Step::Fill(dst as usize, src as usize, len, to));
    }

    /// Brings `[address, address + len)` of a buffer's host copy up to date
    /// when the list is submitted (once per range until a title copy
    /// intervenes), keeping the copy alive until the list is reset.
    pub(crate) fn defer_refresh(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
    ) {
        let mut deferred = self.deferred();
        let key = (std::sync::Arc::as_ptr(shadow) as usize, address, len);
        if deferred.refreshed.insert(key) {
            deferred
                .copies
                .push(Step::Refresh(shadow.clone(), address, len));
        }
    }

    /// Shaders of the list write `[address, address + len)` of a buffer's
    /// host copy: copied back into title memory once the list has run.
    /// Declared once until the list next writes title memory (but with
    /// `VICTORY_DECLARE_ONCE=0`): the dispatches of a frame declared the
    /// same ranges thousands of times, each a step of its own.
    pub(crate) fn defer_gpu_write(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
    ) {
        let mut deferred = self.deferred();
        deferred.forget_placed_constants(address, len);
        let key = (std::sync::Arc::as_ptr(shadow) as usize, address, len);
        if deferred.gpu_written.insert(key) || !declare_once() {
            deferred.copies.push(Step::GpuWrite(shadow.clone(), address, len));
        }
    }

    /// A read back: once the list has run, `rows` rows of `row` bytes go
    /// from `buffer` (rows 256-byte pitched, see `PlacedFootprint::host_rows`)
    /// to title memory at `dst`, pitched `pitch`. The buffer lives until
    /// the list is reset.
    pub(crate) fn defer_read_back(&self, buffer: ID3D12Resource, dst: *mut u8, [row, rows, pitch]: [usize; 3]) {
        self.deferred().backs.push((buffer, dst as usize, [row, rows, pitch]));
    }

    /// Stores `what` (`bytes` of it) at `dst` once the list has run.
    pub(crate) fn defer_write(&self, dst: *mut u8, what: crate::queue::Written, bytes: usize) {
        self.deferred().writes.push((dst as usize, what, bytes));
    }

    /// A mapped upload buffer of `size` bytes, alive until the list is
    /// reset.
    ///
    /// shortcut: one buffer per copy; pool them once uploads are frequent.
    pub(crate) fn upload(&self, host: &Host, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        // With [`card_steps`] the card may be the one that fills it.
        let (buffer, data) = if card_steps::stage() > 1 { host.shared_buffer(size)? } else { host.upload_buffer(size)? };
        name(&buffer, || format!("list upload {size:#x}"));
        self.deferred().hold_upload(buffer.clone(), size);
        Some((buffer, data))
    }

    /// `len` bytes of upload memory (256-byte aligned) until the list is
    /// reset: the host GPU address and the CPU pointer.
    pub(crate) fn scratch(&self, len: u64) -> Option<(u64, *mut u8)> {
        let (buffer, at, data) = self.scratch_at(len)?;
        // SAFETY: a live buffer.
        Some((unsafe { buffer.GetGPUVirtualAddress() } + at, data))
    }

    /// [`HostList::scratch`] and the buffer it lies in, for a descriptor
    /// slot to hold: the slot outlives the list's use of it.
    pub(crate) fn scratch_held(&self, len: u64) -> Option<(ID3D12Resource, u64, *mut u8)> {
        let (buffer, at, data) = self.scratch_at(len)?;
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() } + at;
        Some((buffer, gpu, data))
    }

    /// [`HostList::scratch`]: the buffer, the offset in it and the CPU
    /// pointer.
    fn scratch_at(&self, len: u64) -> Option<(ID3D12Resource, u64, *mut u8)> {
        let host = host()?;
        let mut s = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        let len = len.max(1).next_multiple_of(256);
        if len > SCRATCH_CHUNK {
            let (buffer, data) = host.upload_buffer(len)?;
            name(&buffer, || format!("list scratch {len:#x}"));
            self.deferred().hold_upload(buffer.clone(), len);
            return Some((buffer, 0, data));
        }
        // The chunk in use while `len` fits in what is left of it, else the
        // next one (a list's first chunk may be a small one).
        while s.current < s.chunks.len() && s.used + len > s.chunks[s.current].1 {
            (s.current, s.used) = (s.current + 1, 0);
        }
        if s.current == s.chunks.len() {
            let size = if scratch_pool() && s.chunks.is_empty() && len <= SCRATCH_FIRST {
                SCRATCH_FIRST
            } else {
                SCRATCH_CHUNK
            };
            let chunk = take_chunk(host, size)?;
            s.chunks.push((chunk, size));
            s.used = 0;
        }
        let (data, buffer) = (s.chunks[s.current].0.data, s.chunks[s.current].0.buffer.clone());
        let at = s.used;
        s.used += len;
        Some((buffer, at, (data + at as usize) as *mut u8))
    }

    /// The bytes of scratch chunks the list holds now (after a reset:
    /// none while lists share their chunks).
    pub(crate) fn scratch_kept(&self) -> u64 {
        let scratch = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        scratch.chunks.iter().map(|c| c.1).sum()
    }

    /// The host address of the list's indirect-work budget, zeroed at each
    /// submission of the list from the first use after a reset on.
    pub(crate) fn budget(&self) -> Option<u64> {
        let (buffer, data) = self
            .budget
            .get_or_init(|| {
                let (b, d) = host()?.shared_buffer(256)?;
                Some((b, d as usize))
            })
            .as_ref()?;
        let card = self.card_budget.get_or_init(|| self.make_card_budget()).as_ref();
        let mut deferred = self.deferred();
        if !deferred.budget_zeroed {
            deferred.budget_zeroed = true;
            let at = *data;
            // SAFETY: the list's own mapped words, written before it runs.
            // The words after the budget hold what the clamp noted about the
            // list's last run (mesh dispatches: the largest group count
            // asked for, the groups asked in all; the records refused, the
            // largest cost; with the card's copy, as
            // the list's end copied them here): taken, then zeroed.
            let notes = Box::new(move || unsafe {
                let words = at as *mut u32;
                let (largest, all) = (words.add(1).read_volatile(), words.add(2).read_volatile());
                MESH_GROUPS[0].fetch_max(u64::from(largest), Ordering::Relaxed);
                MESH_GROUPS[1].fetch_add(u64::from(all), Ordering::Relaxed);
                let (refused, biggest) = (words.add(3).read_volatile(), words.add(4).read_volatile());
                BUDGET_NOTES[0].fetch_add(u64::from(refused), Ordering::Relaxed);
                BUDGET_NOTES[1].fetch_max(u64::from(biggest), Ordering::Relaxed);
                for k in 0..5 {
                    words.add(k).write_volatile(0);
                }
            });
            deferred.copies.push(Step::Host(notes, (0, 0)));
            drop(deferred);
            if let Some(card) = card {
                self.zero_card_budget(card);
            }
        }
        // SAFETY: a live buffer.
        Some(unsafe { card.unwrap_or(buffer).GetGPUVirtualAddress() })
    }

    /// The budget's words in the card's own memory ([`card_budget`]), if
    /// the list can zero them itself (`WriteBufferImmediate`).
    fn make_card_budget(&self) -> Option<ID3D12Resource> {
        if !card_budget() {
            return None;
        }
        self.list.cast::<ID3D12GraphicsCommandList2>().ok()?;
        let buffer = host()?.uav_buffer(256)?;
        name(&buffer, || "indirect budget".to_owned());
        Some(buffer)
    }

    /// Records, before the list's first clamp, the zeroing of the card's
    /// copy of the budget: common, written as a copy, then left for
    /// unordered access until the list closes ([`HostList::card_budget_out`]).
    fn zero_card_budget(&self, card: &ID3D12Resource) {
        let Ok(list2) = self.list.cast::<ID3D12GraphicsCommandList2>() else {
            return;
        };
        // SAFETY: a live buffer.
        let at = unsafe { card.GetGPUVirtualAddress() };
        let zeros = [0u64, 4, 8, 12, 16].map(|offset| D3D12_WRITEBUFFERIMMEDIATE_PARAMETER {
            Dest: at + offset,
            Value: 0,
        });
        // SAFETY: an open list; the buffer is the list's own and in the
        // common state at the start of every recording.
        unsafe {
            self.list.ResourceBarrier(&[transition(
                card,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_DEST,
            )]);
            list2.WriteBufferImmediate(zeros.len() as u32, zeros.as_ptr(), None);
            self.list.ResourceBarrier(&[transition(
                card,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )]);
        }
    }

    /// Before the list closes, if this recording charged the card's copy
    /// of the budget: its notes on mesh records go to the mapped words
    /// (read at the next submission), and it goes back to common.
    fn card_budget_out(&self) {
        if !self.deferred().budget_zeroed {
            return;
        }
        let (Some(Some(card)), Some(Some((mapped, _)))) =
            (self.card_budget.get(), self.budget.get())
        else {
            return;
        };
        // SAFETY: an open list; both buffers are the list's own. The
        // mapped one is promoted from common to the copy destination.
        unsafe {
            self.list.ResourceBarrier(&[transition(
                card,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            self.list.CopyBufferRegion(mapped, 4, card, 4, 16);
            self.list.ResourceBarrier(&[transition(
                card,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_COMMON,
            )]);
        }
    }

    /// A device-local unordered-access buffer of at least `size` bytes for
    /// clamped indirect arguments, reused across executions and resets of the
    /// list. Smaller buffers replaced mid-list stay alive until
    /// [`HostList::reset`].
    pub(crate) fn indirect_scratch(&self, size: u64) -> Option<ID3D12Resource> {
        let needed = size.max(64 << 10);
        let mut slot = self.indirect_args.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((buffer, cap)) = &*slot
            && *cap >= needed
        {
            return Some(buffer.clone());
        }
        let prev = slot.as_ref().map_or(0, |(_, cap)| *cap);
        let alloc = prev
            .saturating_mul(2)
            .max(needed)
            .next_multiple_of(64 << 10);
        let buffer = host()?.uav_buffer(alloc)?;
        name(&buffer, || format!("indirect arguments {alloc:#x}"));
        let old = slot.replace((buffer.clone(), alloc));
        drop(slot);
        if let Some((old_buf, _)) = old {
            // It ends the list common, as every scratch buffer does.
            self.indirect_common(&old_buf);
            self.keep(&old_buf);
        }
        Some(buffer)
    }

    /// Where the next batch of clamped records goes: a device-local buffer
    /// the list keeps from recording to recording, the batch's place in it
    /// and the state the buffer is in. Every batch of a recording has a place
    /// of its own: the executions before it may still read theirs.
    pub(crate) fn batch_place(&self) -> Option<(ID3D12Resource, u32, D3D12_RESOURCE_STATES)> {
        use crate::indirect::batch::{JOBS, SLOT};
        let mut args = self.batch_args.lock().unwrap_or_else(|e| e.into_inner());
        let (buffer, i) = ((args.1 / BATCHES) as usize, args.1 % BATCHES);
        if buffer == args.0.len() {
            let size = u64::from(BATCHES * JOBS * SLOT);
            let made = host()?.uav_buffer(size)?;
            name(&made, || format!("indirect batches {size:#x}"));
            args.0.push(made);
        }
        args.1 += 1;
        // (Its first batch finds it as the recording before left it:
        // `batches_common`.)
        let state = if i == 0 { D3D12_RESOURCE_STATE_COMMON } else { D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT };
        Some((args.0[buffer].clone(), i * JOBS * SLOT, state))
    }

    /// Before the list closes: the scratch buffers of the batched clamps go
    /// back to the common state, where the next recording expects them.
    fn batches_common(&self) {
        let mut args = self.batch_args.lock().unwrap_or_else(|e| e.into_inner());
        let used = std::mem::take(&mut args.1).div_ceil(BATCHES) as usize;
        let barriers: Vec<D3D12_RESOURCE_BARRIER> = args.0[..used]
            .iter()
            .map(|b| transition(b, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COMMON))
            .collect();
        if !barriers.is_empty() {
            // SAFETY: an open list and live buffers this recording left in
            // `INDIRECT_ARGUMENT`.
            unsafe { self.list.ResourceBarrier(&barriers) };
        }
    }

    /// Moves `scratch` back to the common state if a clamp left it readable
    /// as indirect arguments ([`HostList::indirect_readable`]): where the
    /// list's first clamp expects it. Done before the list closes and before
    /// a bigger scratch buffer replaces it: one barrier per list (where
    /// there was one per indirect execution), and the list ends as it
    /// begins however it is submitted.
    fn indirect_common(&self, scratch: &ID3D12Resource) {
        {
            let mut deferred = self.deferred();
            if deferred.indirect_readable != scratch.as_raw() as usize {
                return;
            }
            deferred.indirect_readable = 0;
        }
        let barrier = transition(
            scratch,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            D3D12_RESOURCE_STATE_COMMON,
        );
        // SAFETY: an open list and a live buffer this recording left in
        // `INDIRECT_ARGUMENT`.
        unsafe { self.list.ResourceBarrier(&[barrier]) };
    }

    /// The state the list's indirect scratch buffer `scratch` is in at this
    /// point of the recording: indirect arguments if a clamp in this
    /// recording left it so ([`HostList::indirect_readable`]), otherwise
    /// common: the state a buffer is made in, and the one a recording
    /// leaves it in when it closes or replaces it ([`HostList::indirect_common`]).
    pub(crate) fn indirect_state(&self, scratch: &ID3D12Resource) -> D3D12_RESOURCE_STATES {
        if self.deferred().indirect_readable == scratch.as_raw() as usize {
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT
        } else {
            D3D12_RESOURCE_STATE_COMMON
        }
    }

    /// A clamp recorded in this list left `scratch` readable as indirect
    /// arguments. It stays so for every execution that reads it, until the
    /// next clamp moves it back to unordered access.
    pub(crate) fn indirect_readable(&self, scratch: &ID3D12Resource) {
        self.deferred().indirect_readable = scratch.as_raw() as usize;
    }

    /// Keeps `object` alive until the list is reset.
    pub(crate) fn keep(&self, object: &impl Interface) {
        if let Ok(object) = object.cast::<windows::core::IUnknown>() {
            self.deferred().kept.push(object);
        }
    }

    /// Holds a reference to `object` until the list is reset, which is
    /// after its last submission has run.
    pub(crate) fn keep_shared(&self, object: std::sync::Arc<dyn std::any::Any + Send + Sync>) {
        self.deferred().shared.push(object);
    }

    pub(crate) fn keep_frame_preparation(&self, preparation: crate::framegen::Preparation) {
        self.deferred().frame_preparation.push(preparation);
    }

    #[cfg(test)]
    pub(crate) fn kept_shader_heaps(&self) -> Vec<ID3D12DescriptorHeap> {
        self.deferred()
            .kept
            .iter()
            .filter_map(|object| object.cast::<ID3D12DescriptorHeap>().ok())
            .filter(|heap| {
                // SAFETY: the interface owns a live descriptor heap.
                unsafe { heap.GetDesc() }.Flags == D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE
            })
            .collect()
    }

    /// Whether this is the first time since the reset that the list asks
    /// about `key` (a texture to bring up to date).
    pub(crate) fn first_use(&self, key: usize) -> bool {
        self.deferred().textures.insert(key)
    }

    /// Whether `key` was asked about ([`HostList::first_use`]) since the
    /// reset.
    pub(crate) fn used(&self, key: usize) -> bool {
        self.deferred().textures.contains(&key)
    }

    /// Marks the texture used and reports whether its GPU copy has a new source.
    pub(crate) fn alias_source_changed(&self, key: usize, view: &ID3D12Resource, source: &ID3D12Resource) -> bool {
        let source = source.as_raw() as usize;
        let mut deferred = self.deferred();
        deferred.textures.insert(key);
        deferred.texture_aliases.insert(view.as_raw() as usize, source) != Some(source)
    }

    /// Records a raw title-memory copy in the order of the list's steps.
    pub(crate) fn defer_raw_copy(&self, dst: usize, src: usize, len: usize) {
        if len == 0 {
            return;
        }
        let mut deferred = self.deferred();
        deferred.copies.push(Step::RawCopy { dst, src, len });
        deferred.title_written(None);
        drop(deferred);
        self.changed_something();
    }

    /// Runs work that writes only host memory when the list is submitted.
    pub(crate) fn defer_host(&self, work: impl Fn() + Send + 'static) {
        self.deferred().copies.push(Step::Host(Box::new(work), (0, 0)));
    }

    /// As [`HostList::defer_host`], for work that reads `len` bytes of title
    /// memory at `from`.
    pub(crate) fn defer_host_read(&self, from: usize, len: usize, work: impl Fn() + Send + 'static) {
        self.deferred().copies.push(Step::Host(Box::new(work), (from, from + len)));
    }

    /// Brings a texture's staging buffer up to date when the list is
    /// submitted, keeping it alive until the list is reset.
    pub(crate) fn defer_texture(&self, staging: std::sync::Arc<crate::texture::Staging>) {
        self.deferred().copies.push(Step::Texture(staging));
    }

    /// A copy source owned exclusively by this list until reset after its
    /// last execution. A different list can prepare its movie plane concurrently.
    pub(crate) fn plane_upload(
        &self,
        size: u64,
        read: (usize, usize),
        fill: impl Fn(usize) + Send + 'static,
    ) -> Option<ID3D12Resource> {
        let chunk = take_chunk(host()?, size)?;
        let source = chunk.buffer.clone();
        let destination = chunk.data;
        let mut deferred = self.deferred();
        deferred.plane_uploads.push((chunk, size));
        deferred.copies.push(Step::Plane(read, Box::new(move || fill(destination))));
        Some(source)
    }

    /// Runs the list's submission-time steps; refreshes go into `batch`
    /// (the title's big buffers, a 174 MB one among them, are read by most
    /// lists of a submission, through many ranges). Copies and fills that
    /// `taken` names (by their place) are left out: they are done on the
    /// card ([`card_steps`]).
    fn run_deferred(&self, batch: &mut Batch, air: &[(usize, usize)], taken: impl Fn(usize) -> bool) {
        #[cfg(feature = "diagnostics")]
        use crate::copystat::{C, add};
        // The time and the bytes of the copies and fills, some 13,000 a
        // frame between 20,000 refreshes: summed here and added once.
        let (mut us, mut bytes) = (std::time::Duration::ZERO, 0u64);
        #[cfg(feature = "diagnostics")]
        let (mut copies, mut refreshes, mut refreshed) = (0u64, 0u64, 0u64);
        for (i, step) in self.deferred().copies.iter().enumerate() {
            if matches!(step, Step::Copy(..) | Step::Fill(..)) && taken(i) {
                continue;
            }
            let (started, copied) = match step {
                &Step::Copy(dst, src, len) => {
                    let started = std::time::Instant::now();
                    #[cfg(feature = "diagnostics")]
                    {
                        copies += 1;
                    }
                    batch.fill();
                    batch.title_written(TitleWrite::Copy);
                    crate::port::copied(dst, src, len);
                    copy_mapped(dst as *mut u8, src as *const u8, len, true);
                    // Raise the heap-sync floor *after* the copy lands in
                    // title memory so any round satisfying it sees the bytes.
                    if crate::descriptor::in_shader_heap(dst, len) {
                        crate::port_journal::prep_wrote();
                    }
                    // With the submission's one asking: the journal learns it.
                    crate::watch::wrote(dst, len);
                    // What the card holds there may be another value although
                    // ours did not change.
                    if send::own() {
                        crate::shadow::send_copied(dst, len);
                    }
                    (started, len)
                }
                &Step::Fill(dst, src, len, _) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        copies += 1;
                    }
                    crate::port::filled(src, len);
                    if wide_fills() {
                        batch.fills.push((dst, src, len));
                        bytes += len as u64;
                        continue;
                    }
                    let started = std::time::Instant::now();
                    copy_mapped(dst as *mut u8, src as *const u8, len, false);
                    (started, len)
                }
                Step::Placed(p) => {
                    let started = std::time::Instant::now();
                    let free = !touches(air, p.src, p.src + p.len) && !touches(air, p.dst, p.dst + p.len);
                    batch.fill();
                    if free {
                        batch.title_written(TitleWrite::PlacedCopy);
                    }
                    if let Some(bytes) = p.prepare(free) {
                        batch.placed.push((p.shadow.clone(), p.dst, bytes));
                    }
                    // Raise the floor after `prepare` wrote title memory (non-free
                    // placed copies land via GPU copy-back and raise the floor there).
                    if free && crate::descriptor::in_shader_heap(p.dst, p.len) {
                        crate::port_journal::prep_wrote();
                    }
                    (started, p.len)
                }
                Step::Refresh(shadow, address, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        refreshes += 1;
                        refreshed += *len as u64;
                    }
                    batch.add(shadow, *address, *len);
                    continue;
                }
                &Step::RawCopy { dst, src, len } => {
                    batch.fill();
                    #[cfg(feature = "diagnostics")]
                    let started = std::time::Instant::now();
                    batch.title_written(TitleWrite::Deferred);
                    let _step = crate::timeline::span("a step of ours");
                    copy_raw_title(dst, src, len);
                    // The destination can alias a heap's descriptor records;
                    // raise the floor after the bytes are in title memory.
                    crate::port_journal::prep_wrote();
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Runs, 1);
                        add(C::RunUs, started.elapsed().as_micros() as u64);
                    }
                    continue;
                }
                Step::Host(work, _) => {
                    batch.fill();
                    #[cfg(feature = "diagnostics")]
                    let started = std::time::Instant::now();
                    let _step = crate::timeline::span("a step of ours");
                    work();
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Runs, 1);
                        add(C::RunUs, started.elapsed().as_micros() as u64);
                    }
                    continue;
                }
                Step::GpuWrite(shadow, address, len) => {
                    // The shaders start from the title's data: this copy's
                    // pending refresh first. The other copies' wait for
                    // the next flush of all (flushing all of them at every
                    // range a dispatch writes made thousands of write-watch
                    // askings a frame).
                    batch.flush_one(shadow);
                    #[cfg(feature = "diagnostics")]
                    let snapshot = std::time::Instant::now();
                    shadow.writing(*address, *len);
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::GpuWrites, 1);
                        add(C::SnapshotUs, snapshot.elapsed().as_micros() as u64);
                    }
                    continue;
                }
                Step::Texture(staging) => {
                    if batch.texture_due(staging) {
                        batch.due.push(staging.clone());
                        if !wide_textures() {
                            batch.refresh_textures();
                        }
                    }
                    continue;
                }
                Step::Plane((_, len), fill) => {
                    let started = std::time::Instant::now();
                    fill();
                    PREPARING[2].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
                    PREPARING[3].fetch_add(*len as u64, Ordering::Relaxed);
                    continue;
                }
            };
            us += started.elapsed();
            bytes += copied as u64;
        }
        PREPARING[0].fetch_add(us.as_micros() as u64, Ordering::Relaxed);
        PREPARING[1].fetch_add(bytes, Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        {
            add(C::Copies, copies);
            add(C::CopyBytes, bytes);
            add(C::Refreshes, refreshes);
            add(C::RefreshBytes, refreshed);
            SPLIT[7].fetch_add(us.as_nanos() as u64, Ordering::Relaxed);
        }
    }
}

/// The numbers of [`HostList::epoch`].
fn next_epoch() -> u64 {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

/// A table by a copy's address.
pub(crate) type ByCopy<T> = std::collections::HashMap<usize, T, std::hash::BuildHasherDefault<AddressHash>>;

/// A hash for addresses of ours: the bits that tell them apart, spread. The
/// tables it serves are asked 40,000 times a frame, and hold no one's keys
/// but our own.
#[derive(Default)]
pub(crate) struct AddressHash(u64);

impl std::hash::Hasher for AddressHash {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_usize(usize::from(b));
        }
    }
    fn write_usize(&mut self, n: usize) {
        let h = (self.0.rotate_left(5) ^ n as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        self.0 = h ^ (h >> 32);
    }
    fn write_u64(&mut self, n: u64) {
        self.write_usize(n as usize);
    }
    fn write_u32(&mut self, n: u32) {
        self.write_usize(n as usize);
    }
}

/// Since the last frame-rate line, nanoseconds of the submitting thread's
/// preparing: the write-watch asking for a submission; the title's bytes put
/// in under flights' writes; the lists' steps, the refreshes among them; the
/// snapshots; what goes to copies in the card's memory; the lists sent; and
/// of the steps: the refreshes, and the copies and fills (those with the
/// refreshes a copy of title memory flushes). Of the asking, which ranges it
/// is of. Of the refreshes, those before ranges are given to shaders, one
/// copy at a time: nanoseconds, and how many. Ahead of all that: the wait for
/// the submission lock, the steps done on the card planned ([`card_steps`]),
/// the flights waited for, the heaps' records and what is swept; and, at the
/// end, the wait for the heaps' records.
#[cfg(feature = "diagnostics")]
static SPLIT: [std::sync::atomic::AtomicU64; 16] = [const { std::sync::atomic::AtomicU64::new(0) }; 16];

/// Since the last frame-rate line: flushes of refreshes, their refreshes,
/// those known to copy nothing, and those done on several threads.
#[cfg(feature = "diagnostics")]
static FLUSHES: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// The list step that needs to see pending refreshes completed.
enum TitleWrite {
    Copy,
    PlacedCopy,
    Deferred,
}

/// Nonempty refresh flushes before each kind of title write, and their nanoseconds.
#[cfg(feature = "diagnostics")]
static TITLE_WRITE_FLUSHES: [[std::sync::atomic::AtomicU64; 2]; 3] =
    [const { [const { std::sync::atomic::AtomicU64::new(0) }; 2] }; 3];

/// Counts into [`SPLIT`].
#[cfg(feature = "diagnostics")]
struct Split(std::cell::Cell<std::time::Instant>);

#[cfg(feature = "diagnostics")]
impl Split {
    fn begin() -> Split {
        Split(std::cell::Cell::new(std::time::Instant::now()))
    }

    /// Nanoseconds since the last part.
    fn so_far(&self) -> u64 {
        self.0.get().elapsed().as_nanos() as u64
    }

    /// What came since the last part was part `i`.
    fn part(&self, i: usize) {
        let now = std::time::Instant::now();
        SPLIT[i].fetch_add((now - self.0.replace(now)).as_nanos() as u64, Ordering::Relaxed);
    }
}

/// The split of the preparing, per frame, for the frame-rate line.
#[cfg(feature = "diagnostics")]
fn split_line(frames: f64) {
    let ms: Vec<f64> = SPLIT.iter().map(|n| n.swap(0, Ordering::Relaxed) as f64 / 1e6 / frames).collect();
    let f: Vec<f64> = FLUSHES.iter().map(|n| n.swap(0, Ordering::Relaxed) as f64 / frames).collect();
    let e: Vec<f64> = crate::watch::early_asks().iter().map(|&n| n as f64 / frames).collect();
    let a: Vec<f64> = crate::watch::asks().iter().map(|&n| n as f64 / frames).collect();
    let h: Vec<f64> = crate::draw::HEAP_SPLIT.iter().map(|n| n.swap(0, Ordering::Relaxed) as f64 / 1e6 / frames).collect();
    let s: Vec<f64> = crate::shadow::send_stats().iter().map(|&n| n as f64 / frames).collect();
    tracing::info!(
        "preparing per frame, the submitting thread: {:.1} ms asking the write watch ({:.1} of them finding what to ask), {:.1} putting the title's bytes under flights' \
         writes, {:.1} in the lists' steps (refreshes {:.1} in {:.0} flushes of {:.0}, {:.0} of them known to copy nothing, {:.0} on \
         several threads, {:.1} one at a time in {:.0} before shader writes; copies and fills {:.1}, the refreshes before copies among them), {:.1} snapshots, {:.1} for copies in the \
         card's memory, {:.1} sending the lists (VICTORY_LIGHT_REFRESH is {}); ahead of these {:.1} waiting for the submission lock, {:.1} \
         planning steps done on the card, {:.1} waiting for flights, {:.1} the heaps' records and what is swept; at the end {:.1} waiting \
         for the heaps' records; asked early by a thread of its own {:.0} pieces of 2 MiB, {:.1} ms (VICTORY_ASK_EARLY)",
        ms[0],
        ms[8],
        ms[1],
        ms[2],
        ms[6],
        f[0],
        f[1],
        f[2],
        f[3],
        ms[9],
        ms[10] * 1e6,
        ms[7],
        ms[3],
        ms[4],
        ms[5],
        if crate::watch::light() { "on" } else { "off" },
        ms[11],
        ms[12],
        ms[13],
        ms[14],
        ms[15],
        e[0],
        e[1] / 1e3,
    );
    let writes = TITLE_WRITE_FLUSHES.each_ref().map(|[count, nanos]| {
        (count.swap(0, Ordering::Relaxed) as f64 / frames, nanos.swap(0, Ordering::Relaxed) as f64 / 1e6 / frames)
    });
    if writes.iter().any(|&(count, _)| count > 0.0) {
        let [(copy, copy_ms), (placed, placed_ms), (deferred, deferred_ms)] = writes;
        tracing::info!(
            "refresh flushes before title writes per frame: copies {copy:.1} in {copy_ms:.2} ms, placed copies {placed:.1} in \
             {placed_ms:.2} ms, deferred writes {deferred:.1} in {deferred_ms:.2} ms"
        );
    }
    tracing::info!(
        "asking about title memory per frame, all threads: {:.0} kernel askings over {:.0} MB ({:.1} ms, {:.0} pages of 4 KiB found \
         written, {:.0} askings refused); {:.0} questions about holes; view ranges: {:.0} answered by their \
         bits, {:.0} by the guards whole, {:.0} block by block; {:.0} blocks made read-only, {:.0} opened by ourselves; {:.1} ms \
         waiting for the guards, {:.1} ms changing protections; pages taken for the card: {:.1} MB compared, {:.1} MB of them \
         as they were, {:.1} MB to go up in {:.0} runs (VICTORY_SEND_CHANGED)",
        a[0],
        a[1] / 1024.0,
        a[11] / 1e6,
        a[12],
        a[10],
        a[9],
        a[2],
        a[3],
        a[4],
        a[5],
        a[6],
        a[7] / 1e6,
        a[8] / 1e6,
        s[0] / 1048576.0,
        s[1] / 1048576.0,
        s[2] / 1048576.0,
        s[3],
    );
    tracing::info!(
        "the pass over the heap's views per frame: {:.1} ms looking at slots, {:.1} their copies' spans and refreshes, {:.1} textures; the \
         first two beside the recording (VICTORY_HEAP_PASS_ASIDE), {:.1} ms waited for at the lists' close",
        h[0],
        h[1],
        h[2],
        aside::WAITED.swap(0, Ordering::Relaxed) as f64 / 1e3 / frames,
    );
}

/// Whether a submission's refreshes of host copies run on several threads,
/// a copy to a thread (`VICTORY_WIDE_REFRESH=0` disables it; `=1` forces it).
///
/// A submission brings some sixty copies up to date, one after the other
/// on the submitting thread: for each, the pages the title wrote are asked
/// for and read through the kernel into the copy. That is the frame line's
/// "copies", 45-50 ms a frame, and the copies have nothing to do with one
/// another: each refresh writes its own copy and its own notes, and reads
/// title memory. With the switch the copies of a flush (four or more) are
/// spread over the threads kept for the snapshots (`crate::shadow::wide`).
/// The write-watch journal and the guards are behind their own locks, as
/// they are for the title's threads.
fn wide_refresh() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        wide_refresh_enabled(
            std::env::var("VICTORY_WIDE_REFRESH").ok().as_deref(),
            std::thread::available_parallelism().ok().map(|n| n.get()),
        )
    })
}

fn wide_refresh_enabled(setting: Option<&str>, logical_cpus: Option<usize>) -> bool {
    match setting {
        Some("1") => true,
        Some("0") => false,
        _ => logical_cpus.is_some_and(|n| n >= 4),
    }
}

#[cfg(test)]
mod wide_refresh_tests {
    use super::wide_refresh_enabled;

    #[test]
    fn parallel_refresh_is_default_on_multicore_and_can_be_overridden() {
        assert!(wide_refresh_enabled(None, Some(4)));
        assert!(wide_refresh_enabled(None, Some(16)));
        assert!(!wide_refresh_enabled(None, Some(2)));
        assert!(!wide_refresh_enabled(None, None));
        assert!(wide_refresh_enabled(Some("1"), Some(1)));
        assert!(!wide_refresh_enabled(Some("0"), Some(16)));
        assert!(wide_refresh_enabled(Some("auto"), Some(4)));
    }
}

/// A submission's refreshes not done yet, merged per buffer copy into the
/// range that covers them: done before a step that writes title memory or
/// gives a range to shaders, and before the lists run.
///
/// Also the textures brought up to date in the submission so far: most
/// lists of a world submission sample the same textures, and a texture in
/// physical-page memory is hashed whole each time it is checked. Checked
/// once, it is checked again only after a step that writes title memory.
#[derive(Default)]
struct Batch {
    refreshes: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)>,
    /// Where each copy (by address) is in `refreshes`: the list that makes
    /// the heap's buffers ready adds some 4,100 copies, and searching them
    /// for every refresh of a submission was tens of millions of
    /// comparisons a frame.
    index: ByCopy<usize>,
    /// Staging buffers (by address) refreshed since the last title write,
    /// or due to be.
    textures: std::collections::HashSet<usize>,
    /// Those due ([`Batch::refresh_textures`]).
    due: Vec<std::sync::Arc<crate::texture::Staging>>,
    /// What copies at their place in the submission's lists put into title
    /// memory ([`place`]): the host copy, the title address, the bytes.
    placed: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, Vec<u8>)>,
    /// Fills not done yet (destination, source, bytes): [`wide_fills`].
    fills: Vec<(usize, usize, usize)>,
    /// Copies (by address) brought up to date ahead of the steps, and the
    /// range (from, to) ([`Batch::refresh_ahead`]).
    ahead: ByCopy<(usize, usize)>,
    /// `ahead` holds everything the submission refreshes
    /// ([`Batch::refresh_front`]).
    front: bool,
}

/// Whether the copies that a submission's shaders write are brought up to
/// date together, on several threads, ahead of its steps
/// (`VICTORY_REFRESH_AHEAD=0`: each before the step that gives it to them).
///
/// 580 copies a frame, one after the other on the submitting thread, 3.4 to
/// 6.5 ms (session play-1125-0203).
fn ahead_of_writes() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_REFRESH_AHEAD").is_none_or(|v| v != "0") && wide_refresh() && crate::watch::light())
}

/// Whether everything a submission refreshes is brought up to date in one go,
/// on several threads, ahead of its steps (`VICTORY_REFRESH_FRONT=1`, and with
/// `VICTORY_JOIN_PARTS`, which needs it; else in groups where the steps ask
/// for it). Measured on its own at nothing (run submit-1: refreshes 17.1 ms a
/// frame without, 18.3 with).
///
/// A submission's refreshes were done in up to five groups one after the other
/// (the copies its shaders write, those between its steps, what is left),
/// each ending with the threads waiting for the last copy; and the submitting
/// thread went over all the steps once for each question it has (what to ask
/// about, what flights may write too, what shaders write). A refresh may come
/// earlier than its step: the title's bytes are the same until a step writes
/// title memory, and behind such a step everything is asked for again, as it
/// was.
fn refresh_front() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    ahead_of_writes() && (joins_asked().0 > 1 || *ON.get_or_init(|| crate::switch::on("VICTORY_REFRESH_FRONT")))
}

/// Per copy what is refreshed in front of a submission's steps, as
/// [`written_spans`]'s but of all copies, up to the copy's first write by
/// shaders in each part of the submission (`Touch::Part`: another begins).
/// What a part refreshes of a copy it has given to its shaders stays where it
/// stands: around what they write. What a later part refreshes is the
/// title's bytes of before either.
fn front_spans(steps: impl Iterator<Item = Touch>) -> Vec<(usize, usize, usize)> {
    let mut at: ByCopy<usize> = ByCopy::default();
    // (Copy, from, to), in the order they come, and the part that gave it to
    // shaders last.
    let mut spans: Vec<(usize, usize, usize)> = Vec::new();
    let mut given: Vec<u32> = Vec::new();
    let (mut part, mut last) = (1, usize::MAX);
    for step in steps {
        match step {
            Touch::Refresh(copy, from, to) => {
                // (Most come several times on end.)
                let i = if spans.get(last).is_some_and(|s| s.0 == copy) {
                    last
                } else {
                    *at.entry(copy).or_insert_with(|| {
                        spans.push((copy, usize::MAX, 0));
                        given.push(0);
                        spans.len() - 1
                    })
                };
                last = i;
                if given[i] != part {
                    let span = &mut spans[i];
                    (span.1, span.2) = (span.1.min(from), span.2.max(to));
                }
            }
            Touch::Write(copy) => {
                let i = *at.entry(copy).or_insert_with(|| {
                    spans.push((copy, usize::MAX, 0));
                    given.push(0);
                    spans.len() - 1
                });
                given[i] = part;
            }
            Touch::Part => part += 1,
            Touch::Title | Touch::Nothing => {}
        }
    }
    spans.retain(|s| s.1 < s.2);
    spans
}

/// Steps of a submission's lists from which [`Batch::refresh_front`] makes
/// each list's part beside the submitting thread (renderer port #6).
const WIDE_PLAN_STEPS: usize = 4096;

/// One list's part of [`front_spans`] (renderer port #6, [`list_front`]).
#[derive(Default)]
struct ListFront {
    /// The copies it touches, in the order of their first touch, each with
    /// the span its refreshes cover before the list's first write of it by
    /// shaders (`usize::MAX, 0`: none) and whether it writes it: what
    /// [`join_fronts`] takes.
    copies: Vec<(usize, usize, usize, bool)>,
    /// A copy the list refreshes, by address.
    shadows: Vec<(usize, std::sync::Arc<crate::shadow::Shadow>)>,
    /// The textures to bring up to date, in order.
    textures: Vec<std::sync::Arc<crate::texture::Staging>>,
}

/// A list's steps as [`front_spans`] takes them, folded per copy
/// ([`ListFront`]).
fn list_front(steps: &[Step], textures: bool) -> ListFront {
    let mut shadows: Vec<(usize, std::sync::Arc<crate::shadow::Shadow>)> = Vec::new();
    let mut staged: Vec<std::sync::Arc<crate::texture::Staging>> = Vec::new();
    let mut refreshed: ByCopy<()> = ByCopy::default();
    let touches = steps.iter().filter_map(|step| match step {
        Step::Refresh(shadow, from, len) => {
            let key = std::sync::Arc::as_ptr(shadow) as usize;
            if refreshed.insert(key, ()).is_none() {
                shadows.push((key, shadow.clone()));
            }
            Some(Touch::Refresh(key, *from, *from + *len))
        }
        Step::GpuWrite(shadow, ..) => Some(Touch::Write(std::sync::Arc::as_ptr(shadow) as usize)),
        Step::Texture(staging) if textures => {
            staged.push(staging.clone());
            None
        }
        _ => None,
    });
    let copies = fold_touches(touches);
    ListFront { copies, shadows, textures: staged }
}

/// Touches of one list, per copy in the order of their first touch: the span
/// of its refreshes before its first write, and whether it is written.
fn fold_touches(touches: impl Iterator<Item = Touch>) -> Vec<(usize, usize, usize, bool)> {
    let mut at: ByCopy<usize> = ByCopy::default();
    let mut copies: Vec<(usize, usize, usize, bool)> = Vec::new();
    let mut last = usize::MAX;
    for touch in touches {
        let (key, range) = match touch {
            Touch::Refresh(key, from, to) => (key, Some((from, to))),
            Touch::Write(key) => (key, None),
            _ => continue,
        };
        // (Most come several times on end.)
        let i = if copies.get(last).is_some_and(|c| c.0 == key) {
            last
        } else {
            *at.entry(key).or_insert_with(|| {
                copies.push((key, usize::MAX, 0, false));
                copies.len() - 1
            })
        };
        last = i;
        let c = &mut copies[i];
        match range {
            Some((from, to)) if !c.3 => (c.1, c.2) = (c.1.min(from), c.2.max(to)),
            Some(_) => {}
            None => c.3 = true,
        }
    }
    copies
}

/// [`front_spans`] of the lists whose parts are `lists` ([`fold_touches`]), a
/// new part of the submission beginning at each of `parts`: the same spans,
/// in the same order (`front_tests`).
fn join_keys<'a>(lists: impl Iterator<Item = &'a [(usize, usize, usize, bool)]>, parts: &[usize]) -> Vec<(usize, usize, usize)> {
    let mut at: ByCopy<usize> = ByCopy::default();
    let mut spans: Vec<(usize, usize, usize)> = Vec::new();
    let mut given: Vec<u32> = Vec::new();
    let mut part = 1;
    for (n, copies) in lists.enumerate() {
        if parts.contains(&n) {
            part += 1;
        }
        for &(key, from, to, written) in copies {
            let i = *at.entry(key).or_insert_with(|| {
                spans.push((key, usize::MAX, 0));
                given.push(0);
                spans.len() - 1
            });
            // What the list refreshes before it writes the copy counts unless
            // an earlier list of this part gave the copy to shaders.
            if given[i] != part && from < to {
                (spans[i].1, spans[i].2) = (spans[i].1.min(from), spans[i].2.max(to));
            }
            if written {
                given[i] = part;
            }
        }
    }
    spans.retain(|s| s.1 < s.2);
    spans
}

/// [`join_keys`] of the lists' parts, with their copies.
fn join_fronts(fronts: &[ListFront], parts: &[usize]) -> Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)> {
    let mut shadows: ByCopy<std::sync::Arc<crate::shadow::Shadow>> = ByCopy::default();
    for front in fronts {
        for (key, shadow) in &front.shadows {
            shadows.entry(*key).or_insert_with(|| shadow.clone());
        }
    }
    join_keys(fronts.iter().map(|f| f.copies.as_slice()), parts)
        .into_iter()
        .filter_map(|(key, from, to)| Some((shadows.remove(&key)?, from, to)))
        .collect()
}

#[cfg(test)]
mod front_tests {
    use super::{Touch, fold_touches, front_spans, join_keys};

    /// Random submissions of lists (refreshes and shaders' writes of a few
    /// copies, new parts now and then): each list folded on its own and the
    /// folds joined in order give what the whole submission's touches give,
    /// span for span, in the same order (renderer port #6's planning).
    #[test]
    fn lists_folded_apart_and_joined_give_the_submissions_spans() {
        let mut seed = 0x9e37_79b9_u64;
        let mut next = |n: usize| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as usize % n
        };
        for _ in 0..3000 {
            let lists = 1 + next(8);
            let copies = 1 + next(6);
            let mut parts = Vec::new();
            let mut whole: Vec<Touch> = Vec::new();
            let mut folds = Vec::new();
            for n in 0..lists {
                if n > 0 && next(4) == 0 {
                    parts.push(n);
                    whole.push(Touch::Part);
                }
                let mut mine: Vec<Touch> = Vec::new();
                for _ in 0..next(12) {
                    let key = 0x1000 * (1 + next(copies));
                    if next(5) == 0 {
                        mine.push(Touch::Write(key));
                        whole.push(Touch::Write(key));
                    } else {
                        let from = next(100);
                        let to = from + 1 + next(50);
                        mine.push(Touch::Refresh(key, from, to));
                        whole.push(Touch::Refresh(key, from, to));
                    }
                }
                folds.push(fold_touches(mine.into_iter()));
            }
            let want = front_spans(whole.into_iter());
            let got = join_keys(folds.iter().map(|f| f.as_slice()), &parts);
            assert_eq!(got, want, "parts {parts:?}");
        }
    }
}

/// What a step is to the copies' refreshes, for [`written_spans`].
enum Touch {
    /// A refresh of a copy (by address), from, to.
    Refresh(usize, usize, usize),
    /// The copy is given to shaders.
    Write(usize),
    /// Title memory is written.
    #[cfg_attr(not(test), allow(dead_code))]
    Title,
    Nothing,
    /// Another part of the submission begins.
    Part,
}

/// Of `steps`, up to the first that writes title memory: the copies given to
/// shaders, each with the range its refreshes before that cover.
fn written_spans(steps: impl Iterator<Item = Touch>) -> Vec<(usize, usize, usize)> {
    // (From, to, given to shaders.)
    let mut spans: ByCopy<(usize, usize, bool)> = ByCopy::default();
    for step in steps {
        match step {
            Touch::Refresh(copy, from, to) => {
                let span = spans.entry(copy).or_insert((from, to, false));
                if !span.2 {
                    (span.0, span.1) = (span.0.min(from), span.1.max(to));
                }
            }
            // (Not refreshed before: nothing of it is due.)
            Touch::Write(copy) => {
                if let Some(span) = spans.get_mut(&copy) {
                    span.2 = true;
                }
            }
            Touch::Title => break,
            Touch::Nothing | Touch::Part => {}
        }
    }
    spans.into_iter().filter(|(_, s)| s.2).map(|(copy, (from, to, _))| (copy, from, to)).collect()
}

#[cfg(test)]
mod written_spans_tests {
    use super::{Touch, front_spans, split, written_spans};

    #[test]
    fn everything_is_refreshed_in_front_but_what_a_part_gave_its_shaders() {
        use Touch::{Nothing, Part, Refresh, Title, Write};
        let found = front_spans(
            [
                Refresh(1, 10, 20),
                Refresh(1, 12, 25),
                Refresh(2, 0, 5),
                Nothing,
                Write(1),
                // (Around what the part's own shaders write.)
                Refresh(1, 0, 100),
                Write(3),
                Refresh(3, 1, 2),
                Title,
                Refresh(4, 7, 9),
                Part,
                // The next part's bytes under the first one's writes.
                Refresh(1, 30, 40),
                Write(4),
                Refresh(4, 0, 9),
            ]
            .into_iter(),
        );
        assert_eq!(found, [(1, 10, 40), (2, 0, 5), (4, 7, 9)]);
    }

    #[test]
    fn a_submission_is_cut_between_its_lists() {
        // Lists c and e are in neither half's order: they go with the first.
        let Ok([first, second]) = split(vec!['a', 'b', 'c', 'd', 'e'], vec![1, 3, 0], 1) else {
            panic!("no list is named twice");
        };
        let run = |(lists, order): &(Vec<char>, Vec<usize>)| order.iter().map(|&i| lists[i]).collect::<String>();
        assert_eq!((run(&first), run(&second)), ("b".into(), "da".into()));
        assert_eq!(first.0.len() + second.0.len(), 5);
        assert!(split(vec!['a', 'b'], vec![0, 1, 0], 2).is_err());
        assert!(split(vec!['a', 'b'], vec![0, 1], 0).is_err());
    }

    #[test]
    fn what_is_written_is_refreshed_ahead_up_to_its_write_and_to_title_writes() {
        use Touch::{Nothing, Refresh, Title, Write};
        let mut found = written_spans(
            [
                Refresh(1, 10, 20),
                Refresh(2, 0, 5),
                Nothing,
                Refresh(1, 30, 40),
                Write(1),
                Refresh(1, 0, 100),
                Write(3),
                Refresh(3, 1, 2),
                Refresh(4, 7, 9),
                Write(4),
                Title,
                Write(2),
            ]
            .into_iter(),
        );
        found.sort_unstable();
        assert_eq!(found, [(1, 10, 40), (4, 7, 9)]);
    }
}

/// Whether the fills of a submission's upload memory are done together on
/// several threads (`VICTORY_WIDE_FILLS=0`: one after the other, where the
/// lists name them).
///
/// They are 7,000 to 12,000 a frame, some 500 bytes each, 5 ms of the
/// submitting thread. Each reads title memory and writes memory of its own
/// that the card alone reads, when the lists run; they are done before the
/// next step that writes title memory or runs code of ours, and before the
/// submission goes on from its steps.
fn wide_fills() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WIDE_FILLS").is_none_or(|v| v != "0") && wide_refresh())
}

/// Whether a submission's textures are brought up to date on several
/// threads (`VICTORY_WIDE_TEXTURES=0`: one after the other, where the lists
/// name them, as before).
///
/// They are read, hashed and, if they changed, detiled into their staging
/// buffers: 5 ms a frame of the submitting thread, for some 16 MB. Each has
/// its own buffer, notes and lock, and is refreshed by the title's own
/// threads as it is. They are done before the next step that writes title
/// memory, as the copies' refreshes are, and before the lists go to the
/// card. What is new: the note of a list that its upload takes what the
/// buffer then holds (`texture.rs`, `dirty`) may come ahead of the refresh
/// an earlier list of the submission asked for, not behind it. The texture
/// is then uploaded by a later list too, once more than needed.
fn wide_textures() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WIDE_TEXTURES").is_none_or(|v| v != "0") && wide_refresh())
}

impl Batch {
    fn add(&mut self, shadow: &std::sync::Arc<crate::shadow::Shadow>, address: usize, len: usize) {
        let end = address + len;
        let key = std::sync::Arc::as_ptr(shadow) as usize;
        if self.ahead.get(&key).is_some_and(|&(from, to)| from <= address && end <= to) {
            return;
        }
        match self.index.get(&key) {
            Some(&at) => {
                let r = &mut self.refreshes[at];
                (r.1, r.2) = (r.1.min(address), r.2.max(end));
            }
            None => {
                self.index.insert(key, self.refreshes.len());
                self.refreshes.push((shadow.clone(), address, end));
            }
        }
    }

    /// Before shaders are given a range of `shadow`'s copy: its pending
    /// refresh is done, the others stay pending.
    fn flush_one(&mut self, shadow: &std::sync::Arc<crate::shadow::Shadow>) {
        // From here on it is refreshed around what the shaders write. (What
        // is in front of all the steps was, up to here.)
        if !self.front {
            self.ahead.remove(&(std::sync::Arc::as_ptr(shadow) as usize));
        }
        let Some(at) = self.index.remove(&(std::sync::Arc::as_ptr(shadow) as usize)) else {
            return;
        };
        let (shadow, from, to) = self.refreshes.swap_remove(at);
        // The last entry took its place.
        if let Some(moved) = self.refreshes.get(at) {
            self.index.insert(std::sync::Arc::as_ptr(&moved.0) as usize, at);
        }
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        if !(crate::watch::light() && shadow.nothing_new(from, to - from)) {
            Self::update(&shadow, from, to);
        }
        #[cfg(feature = "diagnostics")]
        {
            let ns = started.elapsed().as_nanos() as u64;
            SPLIT[6].fetch_add(ns, Ordering::Relaxed);
            SPLIT[9].fetch_add(ns, Ordering::Relaxed);
            SPLIT[10].fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Brings `[from, to)` of a copy up to date, timed.
    fn update(shadow: &crate::shadow::Shadow, from: usize, to: usize) {
        let started = std::time::Instant::now();
        let _one = crate::timeline::span("a copy's refresh");
        let copied = shadow.update(from, to - from);
        let us = started.elapsed().as_micros() as u64;
        #[cfg(feature = "diagnostics")]
        crate::copystat::range(from, to - from, us, 0);
        PREPARING[0].fetch_add(us, Ordering::Relaxed);
        PREPARING[1].fetch_add(copied as u64, Ordering::Relaxed);
    }

    /// Whether `staging` is still to be refreshed in this submission (and
    /// notes it refreshed).
    fn texture_due<T>(&mut self, staging: &std::sync::Arc<T>) -> bool {
        self.textures.insert(std::sync::Arc::as_ptr(staging) as *const () as usize)
    }

    /// Before a step that writes title memory: pending refreshes are done
    /// first, and textures are checked again after it.
    fn title_written(&mut self, write: TitleWrite) {
        self.flush(Some(write));
        self.ahead.clear();
        self.textures.clear();
    }

    /// Before the steps of `runs` ([`refresh_front`]), a new part of the
    /// submission beginning at each of `parts`.
    fn refresh_front(&mut self, runs: &[&HostList], parts: &[usize]) {
        let mut planning = crate::timeline::span("refreshes in front: the steps gathered");
        let mut due = Vec::new();
        // Renderer port #6: each list's part made on the threads beside, and
        // joined here in order (`list_front`, `join_fronts`): one submission
        // a frame has most of the frame's steps, 3-6 ms of this alone (run
        // guard-4).
        let steps: usize = runs.iter().map(|list| list.deferred().copies.len()).sum();
        if crate::watch::port_guard_plan() && runs.len() >= 2 && steps >= WIDE_PLAN_STEPS {
            #[derive(Clone, Copy)]
            struct Lists(*const *const HostList, usize);
            // SAFETY: the lists outlive the call, which waits for every job.
            unsafe impl Send for Lists {}
            unsafe impl Sync for Lists {}
            let lists: Vec<*const HostList> = runs.iter().map(|&list| std::ptr::from_ref(list)).collect();
            let at = Lists(lists.as_ptr(), lists.len());
            let fronts: std::sync::Arc<Vec<std::sync::Mutex<Option<ListFront>>>> =
                std::sync::Arc::new((0..runs.len()).map(|_| std::sync::Mutex::new(None)).collect());
            let out = fronts.clone();
            let textures = wide_textures();
            crate::shadow::wide_each(runs.len(), move |n| {
                let at = at;
                // SAFETY: as above; a list's deferred steps are behind its lock.
                let list = unsafe { &**at.0.add(n) };
                debug_assert!(n < at.1);
                let front = list_front(&list.deferred().copies, textures);
                *out[n].lock().unwrap_or_else(|e| e.into_inner()) = Some(front);
            });
            drop(lists);
            planning.next("refreshes in front: the spans");
            let fronts: Vec<ListFront> = fronts.iter().map(|m| m.lock().unwrap_or_else(|e| e.into_inner()).take().unwrap_or_default()).collect();
            for front in &fronts {
                for staging in &front.textures {
                    if self.texture_due(staging) {
                        due.push(staging.clone());
                    }
                }
            }
            let spans = join_fronts(&fronts, parts);
            self.refresh_spans(spans, due, planning);
            return;
        }
        let mut copies: ByCopy<std::sync::Arc<crate::shadow::Shadow>> = ByCopy::default();
        let mut touches = Vec::new();
        for (n, list) in runs.iter().enumerate() {
            if parts.contains(&n) {
                touches.push(Touch::Part);
            }
            let mut last = 0;
            for step in &list.deferred().copies {
                match step {
                    Step::Refresh(shadow, at, len) => {
                        let key = std::sync::Arc::as_ptr(shadow) as usize;
                        if key != last {
                            copies.entry(key).or_insert_with(|| shadow.clone());
                            last = key;
                        }
                        touches.push(Touch::Refresh(key, *at, *at + *len));
                    }
                    Step::GpuWrite(shadow, ..) => touches.push(Touch::Write(std::sync::Arc::as_ptr(shadow) as usize)),
                    Step::Texture(staging) if wide_textures() => {
                        if self.texture_due(staging) {
                            due.push(staging.clone());
                        }
                    }
                    _ => {}
                }
            }
        }
        planning.next("refreshes in front: the spans");
        let spans: Vec<_> = front_spans(touches.into_iter())
            .into_iter()
            .filter_map(|(key, from, to)| Some((copies.remove(&key)?, from, to)))
            .collect();
        self.refresh_spans(spans, due, planning);
    }

    /// The rest of [`Batch::refresh_front`]: `spans` brought up to date in
    /// front, and the textures `due`.
    fn refresh_spans(
        &mut self,
        spans: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)>,
        due: Vec<std::sync::Arc<crate::texture::Staging>>,
        planning: crate::timeline::Span,
    ) {
        planning.end();
        self.front = true;
        for (shadow, from, to) in &spans {
            self.ahead.insert(std::sync::Arc::as_ptr(shadow) as usize, (*from, *to));
        }
        #[cfg(feature = "diagnostics")]
        {
            FLUSHES[0].fetch_add(1, Ordering::Relaxed);
            FLUSHES[1].fetch_add(spans.len() as u64, Ordering::Relaxed);
        }
        let _flush = crate::timeline::span("refreshes of copies");
        let started = std::time::Instant::now();
        let before = [0, 2].map(|i| PREPARING[i].load(Ordering::Relaxed));
        // Those known to copy nothing are said so here, unless they are very
        // many: then beside the others.
        const FEW: usize = 256;
        let mut spans = spans;
        // (Renderer port #6: on the threads beside, whatever their number.)
        let many = spans.len() > FEW || crate::watch::port_guard_plan() && spans.len() + due.len() >= 4;
        if !many {
            let _known = crate::timeline::span("refreshes in front: those known to copy nothing");
            #[cfg(feature = "diagnostics")]
            let all = spans.len();
            spans.retain(|(shadow, from, to)| !shadow.nothing_new(*from, *to - *from));
            #[cfg(feature = "diagnostics")]
            FLUSHES[2].fetch_add((all - spans.len()) as u64, Ordering::Relaxed);
        }
        let ahead = spans.len() > 1;
        if ahead {
            let ranges = spans.iter().filter(|r| r.0.asked_ahead()).map(|r| (r.1, r.2));
            crate::watch::ask_ahead(ranges);
        }
        #[cfg(feature = "diagnostics")]
        FLUSHES[3].fetch_add(spans.len() as u64, Ordering::Relaxed);
        crate::shadow::wide_each(spans.len() + due.len(), move |i| {
            // The textures first: they are the longer pieces.
            let Some(i) = i.checked_sub(due.len()) else {
                due[i].refresh();
                return;
            };
            let (shadow, from, to) = &spans[i];
            if many && shadow.nothing_new(*from, *to - *from) {
                #[cfg(feature = "diagnostics")]
                FLUSHES[2].fetch_add(1, Ordering::Relaxed);
                return;
            }
            Self::update(shadow, *from, *to);
        });
        if ahead {
            crate::watch::ask_ahead_done();
        }
        // The time counted is the time it took, not the threads' sum.
        let took = started.elapsed().as_micros() as u64;
        let summed = [0, 2].map(|i| PREPARING[i].load(Ordering::Relaxed)).iter().zip(before).map(|(now, was)| now.saturating_sub(was)).collect::<Vec<_>>();
        PREPARING[2].fetch_sub(summed[1], Ordering::Relaxed);
        PREPARING[0].fetch_sub(summed[0], Ordering::Relaxed);
        PREPARING[0].fetch_add(took, Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        SPLIT[6].fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    /// Before the steps of `runs` ([`ahead_of_writes`]); copies and fills
    /// that `taken` names are not done on the CPU.
    fn refresh_ahead(&mut self, runs: &[&HostList], taken: impl Fn(&HostList, usize) -> bool) {
        let mut copies: ByCopy<std::sync::Arc<crate::shadow::Shadow>> = ByCopy::default();
        let mut touches = Vec::new();
        'steps: for list in runs {
            for (i, step) in list.deferred().copies.iter().enumerate() {
                touches.push(match step {
                    Step::Refresh(shadow, at, len) => {
                        let key = std::sync::Arc::as_ptr(shadow) as usize;
                        copies.entry(key).or_insert_with(|| shadow.clone());
                        Touch::Refresh(key, *at, *at + *len)
                    }
                    Step::GpuWrite(shadow, ..) => Touch::Write(std::sync::Arc::as_ptr(shadow) as usize),
                    Step::Copy(..) if taken(list, i) => Touch::Nothing,
                    Step::Copy(..) | Step::RawCopy { .. } | Step::Placed(_) => break 'steps,
                    _ => Touch::Nothing,
                });
            }
        }
        let due: Vec<_> = written_spans(touches.into_iter())
            .into_iter()
            .filter_map(|(key, from, to)| Some((copies.get(&key)?.clone(), from, to)))
            .collect();
        for (shadow, from, to) in &due {
            self.ahead.insert(std::sync::Arc::as_ptr(shadow) as usize, (*from, *to));
        }
        #[cfg(feature = "diagnostics")]
        let began = std::time::Instant::now();
        let started = std::time::Instant::now();
        let before = PREPARING[0].load(Ordering::Relaxed);
        crate::shadow::wide_each(due.len(), move |i| {
            let (shadow, from, to) = &due[i];
            if !shadow.nothing_new(*from, *to - *from) {
                Self::update(shadow, *from, *to);
            }
        });
        // The time counted is the time it took, not the threads' sum.
        let summed = PREPARING[0].load(Ordering::Relaxed).saturating_sub(before);
        PREPARING[0].fetch_sub(summed.saturating_sub(started.elapsed().as_micros() as u64), Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        {
            let ns = began.elapsed().as_nanos() as u64;
            SPLIT[6].fetch_add(ns, Ordering::Relaxed);
            SPLIT[9].fetch_add(ns, Ordering::Relaxed);
        }
    }

    /// The fills noted are done.
    fn fill(&mut self) {
        // Below this many the threads cost more than they save.
        const FEW: usize = 96;
        const PART: usize = 64;
        if self.fills.is_empty() {
            return;
        }
        let started = std::time::Instant::now();
        if self.fills.len() < FEW {
            for (dst, src, len) in self.fills.drain(..) {
                copy_mapped(dst as *mut u8, src as *const u8, len, false);
            }
        } else {
            let fills = std::mem::take(&mut self.fills);
            crate::shadow::wide_each(fills.len().div_ceil(PART), move |i| {
                for &(dst, src, len) in fills.iter().skip(i * PART).take(PART) {
                    copy_mapped(dst as *mut u8, src as *const u8, len, false);
                }
            });
        }
        let took = started.elapsed();
        PREPARING[0].fetch_add(took.as_micros() as u64, Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        SPLIT[7].fetch_add(took.as_nanos() as u64, Ordering::Relaxed);
    }

    /// The textures due are brought up to date.
    fn refresh_textures(&mut self) {
        if self.due.len() < 2 || !wide_textures() {
            self.due.drain(..).for_each(|staging| staging.refresh());
            return;
        }
        let _textures = crate::timeline::span("textures brought up to date");
        // The time counted is the time it took, not the threads' sum.
        let started = std::time::Instant::now();
        let before = PREPARING[2].load(Ordering::Relaxed);
        let due = std::mem::take(&mut self.due);
        crate::shadow::wide_each(due.len(), move |i| due[i].refresh());
        let summed = PREPARING[2].load(Ordering::Relaxed).saturating_sub(before);
        PREPARING[2].fetch_sub(summed.saturating_sub(started.elapsed().as_micros() as u64), Ordering::Relaxed);
    }

    fn flush(&mut self, _write: Option<TitleWrite>) {
        self.refresh_textures();
        self.index.clear();
        if self.refreshes.is_empty() {
            return;
        }
        #[cfg(feature = "diagnostics")]
        let (began, all) = (std::time::Instant::now(), self.refreshes.len());
        let _flush = crate::timeline::span("refreshes of copies");
        self.flush_all();
        #[cfg(feature = "diagnostics")]
        {
            let nanos = began.elapsed().as_nanos() as u64;
            SPLIT[6].fetch_add(nanos, Ordering::Relaxed);
            FLUSHES[0].fetch_add(1, Ordering::Relaxed);
            FLUSHES[1].fetch_add(all as u64, Ordering::Relaxed);
            if let Some(write) = _write {
                let [count, time] = &TITLE_WRITE_FLUSHES[write as usize];
                count.fetch_add(1, Ordering::Relaxed);
                time.fetch_add(nanos, Ordering::Relaxed);
            }
        }
    }

    fn flush_all(&mut self) {
        // Most of a flush's copies are as they were: they are told so here,
        // from what the submission's asking found, before anything is asked
        // and any thread woken. (Thousands of them, the heap's buffer views
        // once a frame: on the threads, below.)
        const FEW: usize = 256;
        let light = crate::watch::light();
        // Renderer port #6: looked at on the threads beside, below.
        let beside = light && crate::watch::port_guard_plan() && wide_refresh() && self.refreshes.len() >= 4;
        if light && self.refreshes.len() <= FEW && !beside {
            let started = std::time::Instant::now();
            #[cfg(feature = "diagnostics")]
            let all = self.refreshes.len();
            self.refreshes.retain(|(shadow, from, to)| !shadow.nothing_new(*from, *to - *from));
            PREPARING[0].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            FLUSHES[2].fetch_add((all - self.refreshes.len()) as u64, Ordering::Relaxed);
        }
        // One write-watch asking for the copies that lie close together,
        // not one per copy.
        let ahead = self.refreshes.len() > 1;
        if ahead {
            let started = std::time::Instant::now();
            let ranges = self.refreshes.iter().filter(|r| r.0.asked_ahead()).map(|r| (r.1, r.2));
            crate::watch::ask_ahead(ranges);
            PREPARING[0].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
        }
        if wide_refresh() && self.refreshes.len() >= 4 {
            // Each copy on its own, several at a time: a copy's refresh
            // writes that copy and reads title memory, nothing of another's.
            // The time counted is the time it took, not the threads' sum.
            let started = std::time::Instant::now();
            let before = PREPARING[0].load(Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            FLUSHES[3].fetch_add(self.refreshes.len() as u64, Ordering::Relaxed);
            if light {
                // A number to a free thread, not a job to each copy.
                let refreshes = std::mem::take(&mut self.refreshes);
                crate::shadow::wide_each(refreshes.len(), move |i| {
                    let (shadow, from, to) = &refreshes[i];
                    if (refreshes.len() > FEW || beside) && shadow.nothing_new(*from, *to - *from) {
                        #[cfg(feature = "diagnostics")]
                        FLUSHES[2].fetch_add(1, Ordering::Relaxed);
                        return;
                    }
                    Self::update(shadow, *from, *to);
                });
            } else {
                let jobs: Vec<Box<dyn FnOnce() + Send>> = self
                    .refreshes
                    .drain(..)
                    .map(|(shadow, from, to)| Box::new(move || Self::update(&shadow, from, to)) as Box<dyn FnOnce() + Send>)
                    .collect();
                crate::shadow::wide_run(jobs);
            }
            let summed = PREPARING[0].load(Ordering::Relaxed).saturating_sub(before);
            PREPARING[0].fetch_sub(summed.saturating_sub(started.elapsed().as_micros() as u64), Ordering::Relaxed);
        } else {
            for (shadow, from, to) in self.refreshes.drain(..) {
                Self::update(&shadow, from, to);
            }
        }
        if ahead {
            crate::watch::ask_ahead_done();
        }
    }
}

/// Copies mapped title memory, preserving overlap and the standing
/// write-watch query for a successful, covered private write.
fn copy_raw_title(dst: usize, src: usize, len: usize) {
    copy_raw_title_using_mapping(dst, src, len, crate::watch::title_memory_reading_now);
}

fn copy_raw_title_using_mapping(
    dst: usize,
    src: usize,
    len: usize,
    read_mapping: impl FnOnce() -> Option<crate::watch::Reading>,
) {
    if len == 0 || dst.checked_add(len).is_none() || src.checked_add(len).is_none() {
        return;
    }
    let reading = if len <= crate::watch::RAW_COPY_WATCH_LIMIT {
        read_mapping()
    } else {
        None
    };
    let note = reading.as_ref().and_then(|reading| crate::watch::prepare_raw_copy_note(reading, dst, len));
    let reading = reading.filter(|_| note.is_some());
    if !crate::texture::committed(src as *const u8, len) || !crate::texture::committed(dst as *const u8, len) {
        crate::watch::ask_once_end();
        return;
    }
    let written = {
        let _writing = if crate::safecopy::on() {
            crate::watch::writing(dst, len);
            None
        } else {
            Some(crate::watch::kernel_writing(dst, len))
        };
        copy_raw_spans(dst, src, len)
    };
    let keep = written[0].len() == len
        && note.zip(reading.as_ref()).is_some_and(|(note, reading)| note.written(reading));
    drop(reading);
    if !keep {
        crate::watch::ask_once_end();
    }
    for range in written.into_iter().filter(|r| !r.is_empty()) {
        crate::port::copied(range.start, src + (range.start - dst), range.len());
        // Shader writes may make the card's bytes differ even when the CPU's
        // old and new bytes are equal. Only send what the copy actually wrote.
        if send::own() {
            crate::shadow::send_copied(range.start, range.len());
        }
    }
}

/// Copies title bytes even if a mapping changes after it was checked. A
/// fault may leave a partial write. With backward traversal that write can
/// be separate from the fully copied suffix.
fn copy_raw_spans(dst: usize, src: usize, len: usize) -> [std::ops::Range<usize>; 2] {
    if dst == src || dst + len <= src || src + len <= dst {
        let n = read_count(src as *const u8, dst as *mut u8, len);
        return [dst..dst + n, 0..0];
    }
    let backward = dst > src;
    let mut scratch = [0u8; 4096];
    let mut copied = 0;
    let mut partial = 0..0;
    while copied < len {
        let n = (len - copied).min(scratch.len());
        let offset = if backward { len - copied - n } else { copied };
        if !read_into((src + offset) as *const u8, &mut scratch[..n]) {
            break;
        }
        let to = dst + offset;
        let written = read_count(scratch.as_ptr(), to as *mut u8, n);
        if written != n {
            partial = to..to + written;
            break;
        }
        copied += n;
    }
    let start = if backward { dst + len - copied } else { dst };
    [start..start + copied, partial]
}

/// Copies title memory at submission. The ranges were checked against their
/// resources when recorded, but the title may since have unmapped pages of
/// them (tiled memory, page mappings), which read as zeros here; a title
/// destination (`title_dst`) that is not all mapped is left alone.
///
/// The source is read through the kernel (`ReadProcessMemory`), which
/// reports an unmapped page instead of faulting. Asking `VirtualQuery` first
/// cost about 2 ms per copy in the title's big pools (it walks the page
/// tables to the region's end), which made copies run at 100-200 MB/s.
pub(crate) fn copy_mapped(dst: *mut u8, src: *const u8, len: usize, title_dst: bool) {
    #[cfg(feature = "diagnostics")]
    match crate::copystat::mode() {
        1 => {
            // SAFETY: timing only: the caller vouches for the source.
            unsafe { std::ptr::copy(src, dst, len) };
            return;
        }
        2 => return,
        _ => {}
    }
    if len == 0 {
        return;
    }
    // A title destination that is not all mapped is left alone. Asking
    // Windows first (`VirtualQuery`) walks the page tables to the end of
    // the destination's region: in the title's big blocks that is
    // milliseconds a copy, and it was a seventh of the submitting thread's
    // time in gameplay (run 60: 71 of 500 looks). The kernel's copy below
    // says the same when it fails, so the question is only asked then.
    // `VICTORY_COPY_ASK=1` asks first, as before.
    static ASK: OnceLock<bool> = OnceLock::new();
    let ask_first = title_dst
        && *ASK.get_or_init(|| std::env::var_os("VICTORY_COPY_ASK").is_some_and(|v| v == "1"));
    if ask_first && !crate::texture::committed(dst, len) {
        return;
    }
    if title_dst {
        // The kernel writes the destination: it must not be guarded.
        crate::watch::writing(dst as usize, len);
    }
    #[cfg(feature = "diagnostics")]
    let moved = std::time::Instant::now();
    let since = crate::watch::changes();
    // (A title destination that is away only while the runtime changes a
    // mapping beside it is written when that is done.)
    let whole = read(src, dst, len) || title_dst && crate::watch::again_after_change(since, || read(src, dst, len));
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::MoveUs, moved.elapsed().as_micros() as u64);
    if whole {
        return;
    }
    // The copy failed: a source page is not there, or the destination is
    // not all mapped. Then the destination stays as it is (the kernel may
    // have written its first pages).
    if title_dst && !ask_first && !crate::texture::committed(dst, len) {
        crate::watch::lost("a copy of the title's", dst as usize, len);
        return;
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::Fallbacks, 1);
    #[cfg(feature = "diagnostics")]
    let sparse = std::time::Instant::now();
    if !title_dst && len >= 64 << 10 {
        read_sparse(src, dst, len);
    } else {
        read_pieces(src, dst, len);
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::SparseUs, sparse.elapsed().as_micros() as u64);
}

/// After a failed bulk read into host memory, skip contiguous holes. Never
/// cache the query across copies: the title may remap the same address.
fn read_sparse(src: *const u8, dst: *mut u8, len: usize) {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_NOACCESS, VirtualQuery,
    };
    let mut offset = 0;
    while offset < len {
        let at = src.wrapping_add(offset);
        let to = dst.wrapping_add(offset);
        let remaining = len - offset;
        // SAFETY: the query inspects an address and fills a local structure.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        let size = unsafe { VirtualQuery(at.cast(), &mut info, size_of_val(&info)) };
        let n = (info.BaseAddress as usize).saturating_add(info.RegionSize)
            .saturating_sub(at as usize).min(remaining);
        if size != size_of_val(&info) || n == 0 {
            read_pieces(at, to, remaining);
            break;
        }
        if info.State != MEM_COMMIT || info.Protect == PAGE_NOACCESS {
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::Unmapped, n as u64);
            // SAFETY: this path is only used with a writable host destination.
            unsafe { to.write_bytes(0, n) };
        } else if !read(at, to, n) {
            // A mapping may change between query and read.
            read_pieces(at, to, n);
        }
        offset += n;
    }
}

fn read_pieces(src: *const u8, dst: *mut u8, len: usize) {
    // Something is unmapped: page by page (64 KiB, then 4 KiB in a piece
    // that fails), zeros for what cannot be read.
    for_pieces(src as usize, len, 64 << 10, |at, n| {
        let to = dst.wrapping_add(at - src as usize);
        if !read(at as *const u8, to, n) {
            for_pieces(at, n, 4 << 10, |at, n| {
                let to = dst.wrapping_add(at - src as usize);
                if !read(at as *const u8, to, n) {
                    #[cfg(feature = "diagnostics")]
                    crate::copystat::add(crate::copystat::C::Unmapped, n as u64);
                    zero(to, n);
                }
            });
        }
    });
}

/// Zeros at `to`, in one 4 KiB page or two. It may be title memory (the
/// destination of a title's copy), which can go between the question whether
/// it is there and this.
fn zero(to: *mut u8, n: usize) {
    static ZEROS: [u8; 4096] = [0; 4096];
    if n <= ZEROS.len() && crate::safecopy::on() {
        // SAFETY: a destination that is not there fails the copy.
        if !unsafe { crate::safecopy::copy(to, ZEROS.as_ptr(), n) } {
            crate::watch::lost("zeros", to as usize, n);
        }
        return;
    }
    // SAFETY: inside the destination, which is mapped.
    unsafe { to.write_bytes(0, n) };
}

/// Reads title memory at `src` into `dst`; false if a page of it is not
/// readable.
pub(crate) fn read_into(src: *const u8, dst: &mut [u8]) -> bool {
    read(src, dst.as_mut_ptr(), dst.len())
}

/// Reads what is there of the `dst.len()` bytes of title memory at `src`,
/// zeros for the rest; returns how many bytes were there. Safe while the
/// runtime remaps title memory (the kernel, or a copy that takes the fault,
/// reads). What the runtime's record says is mapped is read in one go each,
/// and by pieces where that fails; memory it has no record of by pieces.
pub(crate) fn read_there(src: *const u8, dst: &mut [u8]) -> usize {
    dst.fill(0);
    let (start, len) = (src as usize, dst.len());
    let mut got = 0;
    let mut runs = crate::inplace::shown_runs(start, len);
    if runs.is_empty() {
        runs.push((start, len));
    }
    for (at, n) in runs {
        let piece = &mut dst[at - start..at - start + n];
        if read_into(at as *const u8, piece) {
            got += n;
            continue;
        }
        // (What failed may have written its first pages.)
        piece.fill(0);
        for_pieces(at, n, 64 << 10, |from, n| {
            let piece = &mut dst[from - start..from - start + n];
            if read_into(from as *const u8, piece) {
                got += n;
                return;
            }
            piece.fill(0);
            for_pieces(from, n, 4 << 10, |from, n| {
                let piece = &mut dst[from - start..from - start + n];
                if read_into(from as *const u8, piece) {
                    got += n;
                } else {
                    piece.fill(0);
                }
            });
        });
    }
    got
}

/// Copies `len` bytes from `src` to `dst` unless a source page is not
/// readable.
fn read(src: *const u8, dst: *mut u8, len: usize) -> bool {
    read_count(src, dst, len) == len
}

/// Number of bytes transferred before a failed title-memory access.
fn read_count(src: *const u8, dst: *mut u8, len: usize) -> usize {
    use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    if crate::safecopy::on() {
        // SAFETY: the host side, if any, is valid for `len` bytes. Title
        // pages can be inaccessible and fail the copy. Spans are disjoint
        // or the same, except for callers with their own ordering contract.
        return unsafe { crate::safecopy::copy_count(dst, src, len) };
    }
    let mut done = 0;
    // SAFETY: the host side, if any, is valid for `len` bytes. The kernel
    // reports the transfer and any inaccessible source or destination.
    unsafe { ReadProcessMemory(GetCurrentProcess(), src.cast(), dst.cast(), len, &mut done) };
    done.min(len)
}

#[link(name = "ntdll", kind = "raw-dylib")]
unsafe extern "system" {
    /// The kernel's write into a process's memory, `(process, address,
    /// bytes, length, written out)`: an NTSTATUS, negative when a page of
    /// the range is not there or not writable. It changes no protection.
    fn NtWriteVirtualMemory(
        process: *mut std::ffi::c_void,
        base: *mut std::ffi::c_void,
        buffer: *const std::ffi::c_void,
        size: usize,
        written: *mut usize,
    ) -> i32;
}

/// Whether title memory is written with the kernel's own call
/// (`NtWriteVirtualMemory`) and not `WriteProcessMemory`, which first asks
/// what the pages are (as `VirtualQuery` does: about 2 ms a call in the
/// title's big pools, see [`copy_mapped`]): putting back what shaders wrote
/// cost 1.2-1.9 s a frame for 430-800 pieces, 2 ms each whatever their size
/// (run 24). On by default; `VICTORY_DIRECT_WRITES=0` writes as before.
fn direct_writes() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_DIRECT_WRITES").is_none_or(|v| v != "0"))
}

/// Copies host memory into title memory after a submission (what shaders
/// wrote), leaving pages that are not mapped alone.
pub(crate) fn write_mapped(dst: *mut u8, src: *const u8, len: usize) {
    use windows_sys::Win32::System::Diagnostics::Debug::WriteProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let direct = direct_writes();
    let since = crate::watch::changes();
    let write = |at: usize, n: usize| {
        let from = src.wrapping_add(at - dst as usize);
        if crate::safecopy::on() {
            // SAFETY: the source is ours; a destination page that is not
            // there, or not writable, fails the copy.
            return unsafe { crate::safecopy::copy(at as *mut u8, from, n) };
        }
        let mut done = 0;
        if direct {
            // SAFETY: as below; the kernel checks the destination and
            // writes nothing it cannot (the caller opened its guards:
            // `crate::watch::writing`).
            return unsafe {
                NtWriteVirtualMemory(GetCurrentProcess(), at as _, from.cast(), n, &mut done) >= 0
            };
        }
        // SAFETY: the source is our mapped host copy; the kernel checks the
        // destination.
        unsafe { WriteProcessMemory(GetCurrentProcess(), at as _, from.cast(), n, &mut done) != 0 }
    };
    if len == 0 || write(dst as usize, len) {
        return;
    }
    for_pieces(dst as usize, len, 64 << 10, |at, n| {
        if !write(at, n) {
            for_pieces(at, n, 4 << 10, |at, n| {
                if write(at, n) {
                    return;
                }
                // The kernel's write (`VICTORY_SAFE_COPY=0`) fails where a
                // refresh guarded the block again since the caller opened
                // it; and any where the runtime was changing a mapping.
                crate::watch::writing(at, n);
                if !write(at, n) && !crate::watch::again_after_change(since, || write(at, n)) {
                    crate::watch::lost("a host copy's bytes", at, n);
                }
            });
        }
    });
}

/// Calls `f` with the pieces of `[start, start + len)` cut at multiples of
/// `step`.
fn for_pieces(start: usize, len: usize, step: usize, mut f: impl FnMut(usize, usize)) {
    let end = start + len;
    let mut at = start;
    while at < end {
        let next = ((at / step + 1) * step).min(end);
        f(at, next - at);
        at = next;
    }
}

impl Host {
    pub(crate) fn release_framegen_presenter(&self) {
        let mut presenter = self.presenter.lock().unwrap_or_else(|e| e.into_inner());
        if presenter.as_ref().is_some_and(|p| p.framegen) {
            crate::framegen::release_swap_chain(|| drop(presenter.take()));
        }
    }

    /// The queue everything of ours runs on, for a call that must go to it
    /// in order with the lists (`crate::tiles`: a texture's levels are
    /// given memory ahead of the list that uploads them).
    pub(crate) fn queue(&self) -> &ID3D12CommandQueue {
        &self.queue
    }

    pub(crate) fn texture_readback_buffer(&'static self, size: u64) -> Option<TextureReadback> {
        self.texture_readbacks.take(size, texture_readback_idle_limit(), |bytes| {
            self.read_back_buffer(bytes)
        })
    }

    /// The synchronous wait paths can warn and proceed when setting the
    /// event fails. A pooled buffer cannot be retired on an unverified wait.
    /// The relay already selects flights by their completed fence value.
    fn verify_texture_readback_fence(&self, submitted: u64, report_removed: impl FnOnce() -> String) {
        if texture_readback_idle_limit() != 0 {
            // SAFETY: a plain query.
            let completed = unsafe { self.fence.GetCompletedValue() };
            if let Some(error) = texture_readback_fence_error(completed, submitted, report_removed) {
                fail(format_args!("{error}"));
            }
        }
    }

    /// A buffer of `size` bytes the card writes and we read (a read back),
    /// in the state copies into it need.
    pub(crate) fn read_back_buffer(&self, size: u64) -> Option<ID3D12Resource> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a read-back buffer.
        let made = unsafe {
            self.device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut buffer,
            )
        };
        match made {
            Ok(()) => buffer,
            Err(e) => {
                tracing::warn!(error = %e, size, "no read-back buffer");
                None
            }
        }
    }

    /// A buffer in the upload heap, mapped for writing.
    pub(crate) fn upload_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        MADE[0].fetch_add(1, Ordering::Relaxed);
        MADE[1].fetch_add(size, Ordering::Relaxed);
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_UPLOAD,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of an upload buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    not_resident(),
                    &desc,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    resident(&self.device, b)?;
                    b.Map(0, None, Some(&mut data))
                })
        };
        #[cfg(feature = "diagnostics")]
        if made.is_ok()
            && let Some(b) = &buffer
        {
            crate::book::made(b, || "an upload buffer".into());
        }
        match made {
            Ok(()) => Some((buffer?, data.cast())),
            Err(e) => {
                crate::residency::allocation_error(&e);
                tracing::warn!(error = %e, size, "no upload buffer");
                None
            }
        }
    }
}

/// The swap chain formats a presented texture can be copied into: its own
/// family's.
fn swap_format(format: DXGI_FORMAT) -> Option<DXGI_FORMAT> {
    Some(match format.0 {
        27..=32 => DXGI_FORMAT_R8G8B8A8_UNORM,
        87 | 90 | 91 => DXGI_FORMAT_B8G8R8A8_UNORM,
        23..=25 => DXGI_FORMAT_R10G10B10A2_UNORM,
        9..=14 => DXGI_FORMAT_R16G16B16A16_FLOAT,
        _ => return None,
    })
}

/// Use a direct copy for a single plane when possible. A format DXGI cannot
/// scan out directly can still be sampled into the composer's SDR target.
fn display_format(
    native: Option<DXGI_FORMAT>,
    sampleable: bool,
    compose: bool,
) -> Option<DXGI_FORMAT> {
    if sampleable && (compose || native.is_none()) {
        Some(crate::compose::UI_FORMAT)
    } else {
        native
    }
}

#[cfg(test)]
mod display_format_tests {
    use super::*;

    #[test]
    fn single_planes_use_the_composer_when_a_direct_copy_is_unavailable() {
        for format in [DXGI_FORMAT_R11G11B10_FLOAT, DXGI_FORMAT_R9G9B9E5_SHAREDEXP] {
            let mut desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Format: format,
                DepthOrArraySize: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                ..Default::default()
            };
            let native = swap_format(format);
            assert!(native.is_none());
            assert_eq!(
                display_format(native, crate::compose::view(&desc).is_some(), false),
                Some(crate::compose::UI_FORMAT),
            );
            desc.Flags = D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE;
            assert_eq!(
                display_format(native, crate::compose::view(&desc).is_some(), false),
                None,
            );
        }
        let bgra = swap_format(DXGI_FORMAT_B8G8R8A8_UNORM);
        assert_eq!(display_format(bgra, true, false), bgra);
        assert_eq!(display_format(bgra, true, true), Some(crate::compose::UI_FORMAT));
    }
}

const BUFFERS: u32 = 2;

struct Presenter {
    swap_chain: IDXGISwapChain3,
    framegen: bool,
    width: u32,
    height: u32,
    format: DXGI_FORMAT,
    copy: HostList,
    /// With presents sent ahead ([`late_free_present`]): a list for each
    /// present in the air, and the submission it last flew as (it is used
    /// again once that one has finished).
    ahead: Vec<(HostList, u64)>,
    hwnd: HWND,
    meter: Meter,
    composer: Option<crate::compose::Composer>,
    game_ui: Option<crate::compose::UiFrames>,
    ui_slot: usize,
    #[cfg(feature = "diagnostics")]
    panel: panel::Panel,
}

/// The frame rate, shown in the window's caption every second and logged
/// every few seconds.
/// Makes the frame meter's lines of every 5 s: on a thread of ours
/// ("victory-meter"), so presenting goes on (`VICTORY_METER_THREAD`, on
/// unless `=0`: by the presenting thread, as before). Their counters are
/// taken a moment later than the frames they are divided by; they are only
/// logged. Never waits: the thread takes them in order.
fn meter_lines(lines: impl FnOnce() + Send + 'static) {
    // Read here: whether the meter's lines of every 5 s leave the presenting thread.
    static ON: OnceLock<bool> = OnceLock::new();
    let on = *ON.get_or_init(|| std::env::var("VICTORY_METER_THREAD").as_deref() != Ok("0"));
    #[cfg(feature = "diagnostics")]
    let on = crate::ab::pick("VICTORY_METER_THREAD").unwrap_or(on);
    type Lines = Box<dyn FnOnce() + Send>;
    static TO: OnceLock<Option<Mutex<std::sync::mpsc::Sender<Lines>>>> = OnceLock::new();
    let to = on.then(|| {
        TO.get_or_init(|| {
            let (to, from) = std::sync::mpsc::channel::<Lines>();
            std::thread::Builder::new()
                .name("victory-meter".into())
                .spawn(move || {
                    for lines in from {
                        lines();
                    }
                })
                .ok()
                .map(|_| Mutex::new(to))
        })
        .as_ref()
    });
    match to.flatten() {
        Some(to) => {
            let lines: Lines = Box::new(lines);
            if let Err(back) = to.lock().unwrap_or_else(|e| e.into_inner()).send(lines) {
                (back.0)();
            }
        }
        None => lines(),
    }
}

struct Meter {
    last_present: windows::core::HRESULT,
    since: std::time::Instant,
    frames: u32,
    #[cfg(feature = "diagnostics")]
    copy_stats_frames: u64,
    drawn: u64,
    logged: std::time::Instant,
    /// The caption the window had, once read.
    caption: Option<Vec<u16>>,
}

impl Meter {
    fn new() -> Self {
        let now = std::time::Instant::now();
        Meter {
            last_present: windows::Win32::Foundation::S_OK,
            since: now,
            frames: 0,
            #[cfg(feature = "diagnostics")]
            copy_stats_frames: 0,
            drawn: crate::draw::drawn(),
            logged: now,
            caption: None,
        }
    }

    /// Whether the next frame counted makes the frame-rate line.
    #[cfg(feature = "diagnostics")]
    fn line_due(&self) -> bool {
        self.since.elapsed() >= std::time::Duration::from_secs(1)
    }

    /// Record a frame once the swap chain has accepted it.
    fn record_present(&mut self, hwnd: HWND, status: windows::core::HRESULT) -> bool {
        use windows::Win32::Foundation::S_OK;
        if status != S_OK && status != self.last_present {
            tracing::warn!(?status, "swap chain did not present the frame");
        }
        self.last_present = status;
        if status != S_OK {
            return false;
        }
        self.frame(hwnd);
        true
    }

    /// Counts a presented frame.
    fn frame(&mut self, hwnd: HWND) {
        #[cfg(feature = "diagnostics")]
        tracing::debug!(target: "victory_gpu::frame", token = crate::frame::current(), "present");
        #[cfg(feature = "diagnostics")]
        crate::copystat::presented();
        self.frames += 1;
        #[cfg(feature = "diagnostics")]
        {
            self.copy_stats_frames += 1;
        }
        let elapsed = self.since.elapsed();
        if elapsed < std::time::Duration::from_secs(1) {
            return;
        }
        let drawn = crate::draw::drawn();
        let frames = f64::from(self.frames);
        let fps = frames / elapsed.as_secs_f64();
        let ms = elapsed.as_secs_f64() * 1000.0 / frames;
        let draws = (drawn - self.drawn) as f64 / frames;
        (self.since, self.frames, self.drawn) = (std::time::Instant::now(), 0, drawn);
        // The lines (every second, and more every 5 s: sorting, system
        // calls, some forty lines written) are made by a thread of ours
        // ([`meter_lines`]), not by the presenting thread, the lane: about
        // 3 ms each second there (run ask-1), and 30 ms each 5 s (lane2-0).
        // Their counters are taken there, a moment after the frames counted.
        let five = self.logged.elapsed() >= std::time::Duration::from_secs(5);
        if five {
            self.logged = std::time::Instant::now();
        }
        #[cfg(feature = "diagnostics")]
        let copy_stats_frames = if five { std::mem::take(&mut self.copy_stats_frames) as f64 } else { 0.0 };
        meter_lines(move || {
            // Of a frame, ms preparing submissions and ms waiting for the GPU.
            let [cpu, gpu] = [0, 1].map(|i| {
                SUBMISSION_US[i].swap(0, Ordering::Relaxed) as f64 / 1000.0 / frames
            });
            let [copy_ms, copy_mb, texture_ms, texture_mb] = [0, 1, 2, 3].map(|i| {
                let scale = if i % 2 == 0 { 1e3 } else { 1e6 };
                PREPARING[i].swap(0, Ordering::Relaxed) as f64 / scale / frames
            });
            let [back_ms, back_mb] = [0, 1].map(|i| {
                COPY_BACK[i].swap(0, Ordering::Relaxed) as f64 / [1e3, 1e6][i] / frames
            });
            let dropped = crate::draw::take_dropped() as f64 / frames;
            let indirect = crate::list::take_indirect() as f64 / frames;
            let mesh_largest = MESH_GROUPS[0].swap(0, Ordering::Relaxed);
            let mesh_groups = MESH_GROUPS[1].swap(0, Ordering::Relaxed) as f64 / frames;
            let refused = BUDGET_NOTES[0].swap(0, Ordering::Relaxed) as f64 / frames;
            let biggest = BUDGET_NOTES[1].swap(0, Ordering::Relaxed);
            tracing::info!(
                "indirect mesh dispatches: the largest group count asked for {mesh_largest} (run: at most 65535), {mesh_groups:.0} groups asked per frame; {refused:.2} indirect records a frame ran empty for want of their list's budget (2^27), the largest cost of one {biggest} (0: under 2^22)"
            );
            crate::shadow::exact_line(frames);
            crate::descriptor::sync_line(frames);
            late_line(frames);
            crate::queue::behind_line(frames);
            card_clock_line(frames, ms);
            upq::line(frames);
            rec::line(frames);
            #[cfg(feature = "diagnostics")]
            spans::line(frames);
            crate::shadow::back_line(frames);
            #[cfg(feature = "diagnostics")]
            split_line(frames);
            {
                let [passes, pieces, compares] = [0, 1, 2].map(|i| CARD_PASSES[i].swap(0, Ordering::Relaxed) as f64 / frames);
                if passes + compares > 0.0 {
                    tracing::info!(
                        "our passes over copies in the card's memory per frame: {passes:.0} hash passes over {pieces:.0} pieces after \
                         ranges were sent up (VICTORY_CARD_REHASH_RUNS=1 joins them: now {}), {compares:.0} compare passes",
                        if rehash_runs() { "on" } else { "off" }
                    );
                }
            }
            {
                let [own, asking, still, all] =
                    [0, 1, 2, 3].map(|i| crate::texture::REFRESH_WAITS[i].swap(0, Ordering::Relaxed) as f64 / 1000.0 / frames);
                if all >= 1.0 {
                    tracing::info!(
                        "texture refreshes per frame, all threads: {all:.1} ms, of them {own:.1} ms waiting for a texture another thread \
                         is refreshing, {asking:.1} ms asking the write watch and the guards, {still:.1} ms waiting for title memory to \
                         hold still (the runtime remapping it); the rest is reading, hashing and detiling"
                    );
                }
                let [through, left, filled, short] =
                    [0, 1, 2, 3].map(|i| crate::texture::UNLOCKED[i].swap(0, Ordering::Relaxed) as f64 / frames);
                if through + left + filled > 0.0 {
                    tracing::info!(
                        "texture refreshes that did not wait for title memory to hold still, per frame (VICTORY_TEXTURE_NO_WAIT=1): \
                         {through:.1} read through the kernel, {left:.1} left for the next refresh (a page was not there), \
                         {filled:.1} first fillings from the pages there (VICTORY_TEXTURE_FILL=0: left empty), {short:.1} of them \
                         with less than was mapped (looked at again)"
                    );
                }
            }
            {
                let [copies, joined] = [0, 1].map(|i| COPIES_JOINED[i].swap(0, Ordering::Relaxed) as f64 / frames);
                if copies > 0.0 {
                    tracing::info!(
                        "title buffer copies recorded per frame: {copies:.0}, of them {joined:.0} continue the one before \
                         (VICTORY_JOIN_COPIES=1 makes those one copy with it: now {})",
                        if join_copies() { "on" } else { "off" }
                    );
                }
            }
            // Passes over the heap's buffer views for shaders that pick them by index.
            let picked = crate::draw::HEAP_BUFFER_PASSES.swap(0, Ordering::Relaxed) as f64 / frames;
            let picked_ms =
                crate::draw::HEAP_BUFFER_MICROS.swap(0, Ordering::Relaxed) as f64 / 1e3 / frames;
            // Copies between depth textures and their colour twins, each way.
            let [twin_to_depth, twin_to_colour] =
                [2, 3].map(|i| crate::resource::TWINS[i].swap(0, Ordering::Relaxed) as f64 / frames);
            // Depth planes drawn from their twins.
            let planes = crate::resource::PLANES[0].swap(0, Ordering::Relaxed) as f64 / frames;
            // Copies in the card's memory (`crate::shadow::cards`).
            let card = crate::shadow::card_stats();
            #[cfg(feature = "diagnostics")]
            let slow = crate::watch::slow_asks();
            // Over the same frames as the counts above.
            #[cfg(feature = "diagnostics")]
            let drops = crate::pipeline::take_drops();
            if !five {
                return;
            }
            #[cfg(feature = "diagnostics")]
            open_lists_line();
                tracing::info!(
                    "placed texture views: {} writes of another view of the texture a slot showed (it kept the old view until the new was made)",
                    crate::descriptor::host::KEPT_VIEWS.load(Ordering::Relaxed)
                );
                crate::tiles::status();
                {
                    let [queued, due, away, partly, oldest] =
                        [0, 1, 2, 3, 4].map(|i| crate::draw::HEAP_TEXTURES_QUEUE[i].load(Ordering::Relaxed));
                    let [same, parts, tails] =
                        [0, 1, 2].map(|i| crate::draw::HEAP_TEXTURES_PARTLY[i].load(Ordering::Relaxed));
                    tracing::info!(
                        "textures picked by index, the queue: {queued} slots, {due} of them due (the oldest written {oldest} frames ago), \
                         {away} waiting for their memory, {partly} uploaded in part; since the start: {} slots found uploaded already by \
                         the draws that read them through tables, {} writes of tables' slots (due some frames later), {} slots taken out \
                         as read through a table, {parts} uploads of textures not all mapped ({tails} of them without their first page), \
                         {same} looks that found no more mapped; {} slots made stale by a newer view of other memory there, \
                         {} ranges of memory mapped, unmapped or written under texture views ({} compacted), {} slots made due by them, \
                         {} made slots found unmapped and made null",
                        crate::draw::HEAP_TEXTURES_TABLED.load(Ordering::Relaxed),
                        crate::descriptor::host::TABLED_WRITES[0].load(Ordering::Relaxed),
                        crate::descriptor::host::TABLED_WRITES[1].load(Ordering::Relaxed),
                        crate::descriptor::host::STALE[0].load(Ordering::Relaxed),
                        crate::descriptor::host::STALE[1].load(Ordering::Relaxed),
                        crate::descriptor::host::STALE[2].load(Ordering::Relaxed),
                        crate::descriptor::host::STALE[3].load(Ordering::Relaxed),
                        crate::descriptor::host::STALE[4].load(Ordering::Relaxed)
                    );
                }
                tracing::info!(
                    "slots of textures made while lists sent before them ran, shown once those had run (VICTORY_SHOW_NEW_LATER={}), since the start: {}",
                    u8::from(crate::descriptor::host::show_new_later()),
                    crate::descriptor::host::NEW_LATER.load(Ordering::Relaxed)
                );
                tracing::info!("textures shown, since the start (crate::texlate): {}", crate::texlate::line());
                if let Some(line) = crate::texstore::line() {
                    tracing::info!("{line}");
                }
                if crate::resource::held_only() {
                    let [copies, bytes, grown, whole] = [0, 1, 2, 3].map(|i| crate::resource::HELD[i].load(Ordering::Relaxed));
                    let [flipped, missed] = [0, 1].map(|i| crate::descriptor::host::FLIPPED[i].load(Ordering::Relaxed));
                    tracing::info!(
                        "textures from the first level viewed (VICTORY_HELD_LEVELS=0: whole), since the start: {copies} host copies of the \
                         levels viewed alone ({} MB on the card, {} MB of title memory whole), {grown} replaced by one with more levels, \
                         {flipped} slots made from their new view after its upload had run ({missed} written again first); {} uploads left \
                         out as the host copy held those bytes (VICTORY_TEXTURE_SAME=0: made); {} MB of texture blocks free; \
                         {} looks for textures to let go, {:.2} ms each, the longest {:.1}",
                        bytes >> 20,
                        whole >> 20,
                        crate::draw::HEAP_TEXTURES_SAME.load(Ordering::Relaxed),
                        crate::pool::texture_room() >> 20,
                        crate::resource::SWEPT_US[0].load(Ordering::Relaxed),
                        crate::resource::SWEPT_US[1].load(Ordering::Relaxed) as f64
                            / 1e3
                            / crate::resource::SWEPT_US[0].load(Ordering::Relaxed).max(1) as f64,
                        crate::resource::SWEPT_US[2].load(Ordering::Relaxed) as f64 / 1e3
                    );
                }
                {
                    let mode = crate::watch::page_writes();
                    let [writes, bytes, guarded, plain] = [0, 1, 2, 3].map(|i| crate::watch::WRITTEN[i].load(Ordering::Relaxed));
                    let [kept, compacted, slots, uploaded] =
                        [0, 1, 2, 3].map(|i| crate::descriptor::host::PAGE_WRITES[i].load(Ordering::Relaxed));
                    tracing::info!(
                        "writes behind the guards, since the start: {writes} DirectStorage writes into physical pages at addresses the                      title had mapped ({} MB); {guarded} guarded 64 KiB blocks of host copies lay under them ({}); {plain} reads to                      plain addresses finished ({}); {kept} ranges under texture views kept for the texture pass ({compacted} compacted);                      {slots} slots of placed textures held their bytes, {uploaded} of them uploaded before the write ({})                      (VICTORY_PAGE_WRITES: unset or guards opens the guarded blocks, 0 counts only, 1, slots, all)",
                        bytes >> 20,
                        if mode & 1 != 0 { "opened, taken again" } else { "left as they were" },
                        if mode & 4 != 0 { "told as well" } else { "counted only" },
                        if mode & 2 != 0 { "made due again" } else { "left as they were" }
                    );
                }
                #[cfg(feature = "diagnostics")]
                tracing::info!(
                    "write-watch askings of a millisecond or more: {:.1} a frame, {:.1} ms a frame; the longest asking {:.1} ms",
                    slow[0] as f64 / frames,
                    slow[1] as f64 / 1e3 / frames,
                    slow[2] as f64 / 1e3
                );
                if crate::shadow::cards() {
                    let per_frame = |i: usize| card[i] as f64 / frames;
                    tracing::info!(
                        "host copies in the card's memory: {} copies {} MB, their bytes from before {} MB; per frame: \
                         {:.1} MB sent up in {:.0} ranges, {:.0} pieces compared on the card, {:.0} of them fetched \
                         ({:.1} MB) with {:.1} more waits for the card ({:.1} ms)",
                        card[7],
                        card[8] >> 20,
                        card[9] >> 20,
                        per_frame(0) / 1e6,
                        per_frame(1),
                        per_frame(2),
                        per_frame(3),
                        per_frame(4) / 1e6,
                        per_frame(5),
                        per_frame(6) / 1e3
                    );
                }
                tracing::info!(
                    "{fps:.1} fps, {ms:.1} ms/frame ({cpu:.1} preparing: copies {copy_ms:.1} ms \
                     {copy_mb:.1} MB, textures {texture_ms:.1} ms {texture_mb:.1} MB; {gpu:.1} on the \
                     GPU; {back_ms:.1} copying back {back_mb:.1} MB), {draws:.0} draws/frame, \
                     dropped {dropped:.0} without a host pipeline ({making} being made) and \
                     {indirect:.0} indirect; heap buffers {picked:.1} passes {picked_ms:.1} ms;                  texture slots written so far: {placed} placed, {objects} of objects; \
                     textures picked by index: {picked_textures} uploaded so far ({picked_mb} MB), \
                     {picked_waiting} waiting, {picked_away} passed over unmapped; depth textures with                  unordered access: {du0} made, bound as depth {du1}x, cleared as depth {du2}x,                  unordered views refused {du3}x; buffer views left null as the host would refuse them: {refused}",
                    picked_textures = crate::draw::HEAP_TEXTURES.load(Ordering::Relaxed),
                    picked_mb = crate::draw::HEAP_TEXTURE_BYTES.load(Ordering::Relaxed) >> 20,
                    picked_waiting = crate::draw::HEAP_TEXTURES_WAITING.load(Ordering::Relaxed),
                    picked_away = crate::draw::HEAP_TEXTURES_AWAY.load(Ordering::Relaxed),
                    du0 = crate::resource::DEPTH_UNORDERED[0].load(Ordering::Relaxed),
                    du1 = crate::resource::DEPTH_UNORDERED[1].load(Ordering::Relaxed),
                    du2 = crate::resource::DEPTH_UNORDERED[2].load(Ordering::Relaxed),
                    du3 = crate::resource::DEPTH_UNORDERED[3].load(Ordering::Relaxed),
                    refused = format!(
                        "{} ({} copies made of views off their buffer's elements); depth twins: {} alive, {} depth textures left without one; twin copies per frame: {twin_to_depth:.1} into depth textures, {twin_to_colour:.1} into twins; depth planes: {planes:.1} drawn from their twins per frame, {} times not given back to a twin, {} times used before the pass",
                        crate::descriptor::host::REFUSED_VIEWS.load(Ordering::Relaxed),
                        crate::shadow::OWN_MADE.load(Ordering::Relaxed),
                        crate::resource::TWINS[0].load(Ordering::Relaxed),
                        crate::resource::TWINS[1].load(Ordering::Relaxed),
                        crate::resource::PLANES[1].load(Ordering::Relaxed),
                        crate::resource::PLANES[2].load(Ordering::Relaxed)
                    ),
                    making = crate::pipeline::making(),
                    placed = crate::descriptor::host::PLACED_TEXTURE_SLOTS.load(Ordering::Relaxed),
                    objects = crate::descriptor::host::OBJECT_TEXTURE_SLOTS.load(Ordering::Relaxed)
                );
                #[cfg(feature = "diagnostics")]
                {
                    let put = [0, 1, 2, 3].map(|i| crate::shadow::PUT[i].swap(0, Ordering::Relaxed) as f64 / frames);
                    tracing::info!(
                        "what shaders wrote put into title memory per frame (VICTORY_EXACT_BACK): {:.0} pieces in {:.0} writes, {:.1} MB; {:.0} pieces joined over gaps",
                        put[0],
                        put[1],
                        put[2] / 1e6,
                        put[3]
                    );
                    traffic::line(copy_stats_frames);
                    crate::copystat::log(copy_stats_frames);
                    for (why, n) in drops.iter().take(40) {
                        tracing::info!(
                            "dropped {:.1}/frame without a host pipeline: {why}",
                            *n as f64 / frames
                        );
                    }
                }
        });
        if self.caption.is_none() {
            // The window's own text, read without a message.
            let mut buffer = vec![0u16; 512];
            // SAFETY: a window handle and a buffer of that length.
            let n = unsafe { InternalGetWindowText(hwnd, &mut buffer) };
            buffer.truncate(n.max(0) as usize);
            self.caption = Some(buffer);
        }
        let Some(caption) = &self.caption else { return };
        let text = format!(" | VIctory {fps:.0} fps {ms:.1} ms/frame {draws:.0} draws/frame");
        let mut text: Vec<u16> = caption.iter().copied().chain(text.encode_utf16()).collect();
        text.push(0);
        // Setting the text waits for the window's thread (a title thread,
        // which may not pump for a while), so a thread of ours does it and
        // presenting never waits; it takes the newest text when free.
        static PENDING: Mutex<Option<(usize, Vec<u16>)>> = Mutex::new(None);
        static SETTER: std::sync::Once = std::sync::Once::new();
        *PENDING.lock().unwrap_or_else(|e| e.into_inner()) = Some((hwnd.0 as usize, text));
        SETTER.call_once(|| {
            let _ = std::thread::Builder::new()
                .name("victory-caption".into())
                .spawn(|| {
                    loop {
                        let next = PENDING.lock().unwrap_or_else(|e| e.into_inner()).take();
                        match next {
                            // SAFETY: a window handle and a NUL-terminated text.
                            Some((hwnd, text)) => unsafe {
                                let _ = SetWindowTextW(HWND(hwnd as *mut _), PCWSTR(text.as_ptr()));
                            },
                            None => std::thread::sleep(std::time::Duration::from_millis(100)),
                        }
                    }
                });
        });
    }
}

#[cfg(test)]
mod present_status_tests {
    use super::{HWND, Meter};
    use windows::Win32::Foundation::{DXGI_STATUS_OCCLUDED, E_FAIL, S_OK};

    #[test]
    fn the_meter_counts_successful_presents_across_a_failure() {
        let mut meter = Meter::new();
        let hwnd = HWND::default();
        assert!(meter.record_present(hwnd, S_OK));
        assert!(!meter.record_present(hwnd, DXGI_STATUS_OCCLUDED));
        assert!(!meter.record_present(hwnd, E_FAIL));
        assert!(!meter.record_present(hwnd, E_FAIL));
        assert!(meter.record_present(hwnd, S_OK));
        assert_eq!(meter.frames, 2);
        assert_eq!(meter.last_present, S_OK);
    }
}

impl Host {
    /// Shows `texture`, which is in `state` (a host state) and stays in it,
    /// as frame `frame` (saved if captures are on). `None` if nothing was
    /// shown; else, in diagnostics builds, whether the frame was black.
    #[allow(dead_code)]
    pub(crate) fn present(&self, texture: &ID3D12Resource, state: u32, frame: u64) -> Option<bool> {
        self.present_planes(texture, state, None, frame, None)
    }

    /// Shows `texture`, optionally alpha-compositing a second `overlay` plane
    /// over it (Series S two-plane presentation), and returns them to their states.
    pub(crate) fn present_planes(
        &self,
        texture: &ID3D12Resource,
        state: u32,
        overlay: Option<(&ID3D12Resource, u32)>,
        frame: u64,
        token: Option<u64>,
    ) -> Option<bool> {
        // SAFETY: a live resource.
        let desc = unsafe { texture.GetDesc() };
        let base_view = crate::compose::view(&desc);
        let planes = overlay.and_then(|(over, over_state)| {
            // SAFETY: a live overlay resource.
            let over_desc = unsafe { over.GetDesc() };
            Some((
                over,
                D3D12_RESOURCE_STATES(over_state as i32),
                base_view?,
                crate::compose::view(&over_desc)?,
            ))
        });
        let native_format = swap_format(desc.Format);
        let Some(format) = display_format(native_format, base_view.is_some(), planes.is_some()) else {
            tracing::warn!(format = desc.Format.0, "cannot present this format");
            return None;
        };
        // Keep a single SDR format as Series S changes between UI and movie
        // frames. The composer converts single-plane sources when needed.
        let fg_format = if crate::console::current() == crate::console::Console::SeriesS && base_view.is_some() {
            DXGI_FORMAT_R8G8B8A8_UNORM
        } else {
            format
        };
        let (width, height) = (desc.Width as u32, desc.Height);
        let mut presenter = self.presenter.lock().unwrap_or_else(|e| e.into_inner());
        if presenter.is_none() {
            *presenter = match self.presenter(width, height, format, fg_format, frame) {
                Ok(p) => Some(p),
                Err(e) => {
                    tracing::warn!(error = %e, "no swap chain");
                    return None;
                }
            };
        }
        let state_u32 = state;
        let p = presenter.as_mut()?;
        let format = if p.framegen { fg_format } else { format };
        if (p.width, p.height, p.format) != (width, height, format) {
            // Presents sent ahead hold back buffers until they have run.
            settle_now(WHY_SYNC);
            let resize_buffers = || {
                // SAFETY: no back buffer is held; every submission was waited for.
                unsafe {
                    p.swap_chain
                        .ResizeBuffers(BUFFERS, width, height, format, DXGI_SWAP_CHAIN_FLAG(0))
                }
            };
            let resized = if p.framegen {
                crate::framegen::resize(frame, width, height, format.0, resize_buffers)
            } else {
                resize_buffers()
            };
            match resized {
                Ok(()) => {
                    if p.format != format {
                        p.composer = None;
                    }
                    p.game_ui = None;
                    (p.width, p.height, p.format) = (width, height, format);
                    tracing::info!(
                        width,
                        height,
                        format = format.0,
                        frame,
                        "the swap chain's buffers were made again for another size or format"
                    );
                    self.log_video_memory("right after the swap chain's buffers were made again");
                }
                Err(e) => {
                    tracing::warn!(error = %e, width, height, "swap chain not resized");
                    if p.framegen {
                        // The optional proxy should not prevent DXGI from
                        // presenting after a failed resize.
                        if !crate::framegen::release_swap_chain(|| drop(presenter.take())) {
                            return None;
                        }
                        drop(presenter);
                        return self.present_planes(texture, state_u32, overlay, frame, token);
                    }
                    return None;
                }
            }
        }
        let draw_planes = planes.is_some() || native_format != Some(format);
        if draw_planes && p.composer.is_none() {
            match crate::compose::Composer::new(self, format) {
                Ok(c) => p.composer = Some(c),
                Err(e) => {
                    tracing::warn!(error = %e, "display composer not made");
                    drop(presenter);
                    return planes.and_then(|_| self.present_planes(texture, state_u32, None, frame, token));
                }
            }
        }
        let state = D3D12_RESOURCE_STATES(state_u32 as i32);
        // SAFETY: our swap chain, list and live source texture(s).
        unsafe {
            let index = p.swap_chain.GetCurrentBackBufferIndex();
            let Ok(back) = p.swap_chain.GetBuffer::<ID3D12Resource>(index) else {
                return None;
            };
            // Sent ahead ([`late_free_present`]) unless the frame is to be
            // read back at once (a saved or listed frame): then a list of
            // its own, one whose last flight has finished or a new one.
            #[cfg(feature = "diagnostics")]
            let (save_frame, capture_scene, read_now) = {
                let listed = crate::capture::listed(frame);
                let (save_frame, capture_scene) = crate::capture::requested(frame, listed, p.framegen);
                (save_frame, capture_scene, save_frame || listed)
            };
            #[cfg(not(feature = "diagnostics"))]
            let read_now = false;
            #[cfg(not(feature = "diagnostics"))]
            let capture_scene = false;
            let game_ui = if p.framegen && planes.is_some() && (!read_now || capture_scene) {
                if p.game_ui.as_ref().is_none_or(|ui| ui.size != (width, height)) {
                    p.game_ui = crate::compose::UiFrames::new(self, (width, height));
                }
                p.game_ui.as_ref().map(|ui| ui.images[p.ui_slot].clone())
                    .filter(|ui| crate::framegen::set_ui(Some(ui)))
            } else {
                None
            };
            let slot = if late_free_present() && !read_now {
                let finished = FINISHED.load(Ordering::Acquire);
                p.ahead.iter().position(|(_, flown)| *flown <= finished).or_else(|| {
                    let list = self.new_list().ok()?;
                    list.close();
                    p.ahead.push((list, 0));
                    Some(p.ahead.len() - 1)
                })
            } else {
                None
            };
            let copy = match slot {
                Some(i) => &p.ahead[i].0,
                None => &p.copy,
            };
            copy.reset();
            copy.keep(texture);
            let list = &copy.list;
            let composed = if let Some(composer) = p.composer.as_mut().filter(|_| draw_planes) {
                use crate::compose::{Picture, UI_READ};
                let mut pre = Vec::with_capacity(4);
                if state != D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE {
                    pre.push(transition(texture, state, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE));
                }
                if let Some((over, over_state, ..)) = planes {
                    copy.keep(over);
                    if over_state != D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE {
                        pre.push(transition(over, over_state, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE));
                    }
                }
                pre.push(transition(&back, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET));
                if let Some(ui) = &game_ui {
                    copy.keep(ui);
                    pre.push(transition(ui, UI_READ, D3D12_RESOURCE_STATE_RENDER_TARGET));
                }
                list.ResourceBarrier(&pre);
                let compose_slot = 2 * slot.map_or(0, |i| i + 1);
                let base = (texture, base_view?);
                let picture = match planes {
                    Some((over, _, _, over_fmt)) if game_ui.is_none() => Picture::Composed(base, (over, over_fmt)),
                    _ => Picture::Scene(base),
                };
                let drawn = composer.record(self, compose_slot, list, picture, &back, (width, height))
                    .and_then(|()| {
                        if let (Some(ui), Some((over, _, _, over_fmt))) = (&game_ui, planes) {
                            composer.record(self, compose_slot + 1, list, Picture::Overlay((over, over_fmt)), ui, (width, height))?;
                        }
                        Ok(())
                    });
                if let Err(err) = drawn {
                    copy.close();
                    drop(back);
                    drop(presenter);
                    tracing::warn!(%err, "could not record the display planes");
                    return planes.and_then(|_| self.present_planes(texture, state_u32, None, frame, token));
                }
                static ONCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                if let Some((_, _, base_fmt, over_fmt)) = planes
                    && !ONCE.swap(true, Ordering::Relaxed)
                {
                    tracing::info!(
                        frame,
                        width,
                        height,
                        base_format = base_fmt.0,
                        overlay_format = over_fmt.0,
                        "display output: compositing two planes (overlay active)"
                    );
                }
                let mut post = Vec::with_capacity(4);
                if state != D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE {
                    post.push(transition(texture, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, state));
                }
                if let Some((over, over_state, ..)) = planes
                    && over_state != D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
                {
                        post.push(transition(over, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, over_state));
                }
                if let Some(ui) = &game_ui {
                    post.push(transition(ui, D3D12_RESOURCE_STATE_RENDER_TARGET, UI_READ));
                }
                post.push(transition(
                    &back,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                ));
                list.ResourceBarrier(&post);
                true
            } else {
                false
            };
            if !composed {
                let mut pre = Vec::with_capacity(2);
                if state != D3D12_RESOURCE_STATE_COPY_SOURCE {
                    pre.push(transition(texture, state, D3D12_RESOURCE_STATE_COPY_SOURCE));
                }
                pre.push(transition(
                    &back,
                    D3D12_RESOURCE_STATE_PRESENT,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                ));
                list.ResourceBarrier(&pre);
                list.CopyResource(&back, texture);
            }
            #[cfg(feature = "diagnostics")]
            let scene_available = !capture_scene || planes.is_none() || game_ui.is_some();
            #[cfg(feature = "diagnostics")]
            if capture_scene && !scene_available {
                tracing::warn!(frame, "could not separate the scene for capture before frame generation");
            }
            #[cfg(feature = "diagnostics")]
            let capture = (save_frame && scene_available)
                .then(|| {
                    if capture_scene {
                        tracing::info!(
                            frame,
                            title_ui_separate = game_ui.is_some(),
                            "capturing the scene before frame generation"
                        );
                    }
                    if composed {
                        list.ResourceBarrier(&[transition(
                            &back,
                            D3D12_RESOURCE_STATE_COPY_DEST,
                            D3D12_RESOURCE_STATE_COPY_SOURCE,
                        )]);
                        let res = crate::capture::read_back(&self.device, list, &back);
                        list.ResourceBarrier(&[transition(
                            &back,
                            D3D12_RESOURCE_STATE_COPY_SOURCE,
                            D3D12_RESOURCE_STATE_COPY_DEST,
                        )]);
                        res
                    } else {
                        crate::capture::read_back(&self.device, list, texture)
                    }
                })
                .flatten();
            #[cfg(feature = "diagnostics")]
            let probe = if slot.is_some() || composed {
                None
            } else {
                crate::capture::probe(&self.device, list, texture)
            };
            #[cfg(feature = "diagnostics")]
            let watched = if composed {
                None
            } else {
                crate::popwatch::take(&self.device, list, texture)
            };
            #[cfg(feature = "diagnostics")]
            let panel_copy = {
                let target = match game_ui.as_ref() {
                    Some(ui) => panel::Target::GameUi(ui),
                    None if p.framegen && planes.is_none() => panel::Target::SeparateUi,
                    None => panel::Target::BackBuffer,
                };
                p.panel.draw(self, copy, &back, (width, height), format, target)
            };
            #[cfg(not(feature = "diagnostics"))]
            let _ = frame;
            if composed {
                list.ResourceBarrier(&[transition(
                    &back,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_PRESENT,
                )]);
            } else {
                let mut post = Vec::with_capacity(2);
                if state != D3D12_RESOURCE_STATE_COPY_SOURCE {
                    post.push(transition(texture, D3D12_RESOURCE_STATE_COPY_SOURCE, state));
                }
                post.push(transition(
                    &back,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_PRESENT,
                ));
                list.ResourceBarrier(&post);
            }
            copy.close();
            let (flown, fence) = match slot {
                Some(i) => {
                    let (flown, fence) = self.execute_own(copy);
                    p.ahead[i].1 = flown;
                    (flown, fence)
                }
                None => (0, self.execute(&[copy])),
            };
            #[cfg(feature = "diagnostics")]
            p.panel.submitted(panel_copy.as_ref(), fence);
            #[cfg(feature = "diagnostics")]
            if let Some(taken) = watched {
                crate::popwatch::sent(taken, flown, frame);
            }
            #[cfg(not(feature = "diagnostics"))]
            let _ = (flown, fence);
            #[cfg(feature = "diagnostics")]
            if let Some((buffer, footprint)) = capture {
                crate::capture::save_read_back(frame, capture_scene, &buffer, &footprint);
            }
            #[cfg(feature = "diagnostics")]
            crate::replay::record::present(texture, state);
            let mut showing = crate::timeline::span("the host's Present");
            let status = if p.framegen {
                #[cfg(feature = "diagnostics")]
                let ui = panel_copy.as_ref().and_then(|panel| panel.ui.as_ref());
                #[cfg(not(feature = "diagnostics"))]
                let ui = None;
                let registered = game_ui.is_some() || crate::framegen::set_ui(ui);
                if !registered && !crate::framegen::set_ui(None) {
                    // The proxy still owns an older UI. Finish this list
                    // before retiring the presenter, then render both planes.
                    drop(back);
                    settle_now(WHY_SYNC);
                    if !crate::framegen::release_swap_chain(|| drop(presenter.take())) {
                        return None;
                    }
                    drop(presenter);
                    return self.present_planes(texture, state_u32, overlay, frame, token);
                }
                let generation = registered && (planes.is_none() || game_ui.is_some());
                let status = crate::framegen::present(&p.swap_chain, frame, token, generation);
                if game_ui.is_some() {
                    p.ui_slot ^= 1;
                }
                status
            } else {
                p.swap_chain.Present(0, DXGI_PRESENT(0))
            };
            showing.next("the frame meter");
            #[cfg(feature = "diagnostics")]
            if status == windows::Win32::Foundation::S_OK {
                p.panel.frame(p.meter.line_due());
            }
            let presented = p.meter.record_present(p.hwnd, status);
            showing.end();
            if !presented { return None; }
            self.start_title_stock();
            #[cfg(feature = "diagnostics")]
            return Some(probe.is_some_and(|b| crate::capture::black(&b)));
            #[cfg(not(feature = "diagnostics"))]
            Some(false)
        }
    }

    fn presenter(
        &self,
        width: u32,
        height: u32,
        format: DXGI_FORMAT,
        fg_format: DXGI_FORMAT,
        frame: u64,
    ) -> windows::core::Result<Presenter> {
        let title = title_window();
        let hwnd = match title {
            Some(hwnd) => hwnd,
            None => window(width, height)?,
        };
        self.log_video_memory("right before the swap chain is made");
        // SAFETY: plain object creation for our window.
        unsafe {
            // Finish fallible list creation before ownership crosses into AMD.
            let copy = self.new_list()?;
            copy.close();
            let factory: IDXGIFactory2 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
            let desc = DXGI_SWAP_CHAIN_DESC1 {
                Width: width,
                Height: height,
                Format: format,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
                BufferCount: BUFFERS,
                SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
                ..Default::default()
            };
            let fg_desc = DXGI_SWAP_CHAIN_DESC1 { Format: fg_format, ..desc };
            let generated = crate::framegen::requested()
                .then(|| crate::framegen::swap_chain(hwnd, &fg_desc, &factory, &self.queue, &self.device, frame))
                .flatten();
            let framegen = generated.is_some();
            let swap_chain = match generated {
                Some(swap_chain) => swap_chain,
                None => factory
                    .CreateSwapChainForHwnd(&self.queue, hwnd, &desc, None, None)?
                    .cast::<IDXGISwapChain3>()?,
            };
            // Alt+Enter is the title window's own switch to and from full
            // screen (frameless, the swap chain stretched); DXGI's would make
            // it exclusive full screen, whose buffers nothing resizes.
            if title.is_some() {
                let _ = factory.MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER);
            }
            tracing::info!(width, height, format = format.0, "presenting to a window");
            self.log_video_memory("right after the swap chain is made");
            Ok(Presenter {
                swap_chain,
                framegen,
                width,
                height,
                format: if framegen { fg_format } else { format },
                copy,
                ahead: Vec::new(),
                hwnd,
                meter: Meter::new(),
                composer: None,
                game_ui: None,
                ui_slot: 0,
                #[cfg(feature = "diagnostics")]
                panel: panel::Panel::new(),
            })
        }
    }
}

/// A visible top-level window of this process: the one the title made, if
/// it made one.
fn title_window() -> Option<HWND> {
    extern "system" fn visit(hwnd: HWND, found: LPARAM) -> windows::core::BOOL {
        let mut process = 0;
        // SAFETY: a window handle from the enumeration; `found` is ours.
        unsafe {
            GetWindowThreadProcessId(hwnd, Some(&mut process));
            if process == windows::Win32::System::Threading::GetCurrentProcessId()
                && IsWindowVisible(hwnd).as_bool()
            {
                *(found.0 as *mut Option<HWND>) = Some(hwnd);
                return false.into();
            }
        }
        true.into()
    }
    let mut found: Option<HWND> = None;
    // SAFETY: the callback only writes `found`, which outlives the call.
    let _ = unsafe { EnumWindows(Some(visit), LPARAM(&raw mut found as isize)) };
    found
}

/// `state` without the bits the host texture's flags forbid: the console
/// lets the title name states its own resources allow (unordered access on
/// a depth target, whose host copy cannot have both).
pub(crate) fn legal_state(texture: &ID3D12Resource, state: u32) -> u32 {
    // SAFETY: a plain query.
    legal_for(unsafe { texture.GetDesc() }.Flags.0 as u32, state)
}

/// [`legal_state`] for a texture with resource flags `flags`.
pub(crate) fn legal_for(flags: u32, state: u32) -> u32 {
    // (flag, set or clear forbids, state bits): render target, unordered
    // access, depth, denied shader resource.
    const FORBIDDEN: [(u32, bool, u32); 4] = [
        (0x1, false, 0x4),
        (0x4, false, 0x8),
        (0x2, false, 0x10 | 0x20),
        (0x8, true, 0x40 | 0x80),
    ];
    FORBIDDEN
        .iter()
        .filter(|&&(flag, deny, _)| (flags & flag != 0) == deny)
        .fold(state, |s, &(_, _, bits)| s & !bits)
}

impl Host {
    /// PCI vendor ID of the adapter running the host device.
    pub(crate) fn adapter_vendor_id(&self) -> Option<u32> {
        // SAFETY: a plain query on the adapter kept with this device.
        unsafe { self.adapter.as_ref()?.GetDesc1() }.ok().map(|desc| desc.VendorId)
    }

    /// The card's memory as the system accounts it to this process, in
    /// bytes: the budget and the use of the card's own memory, then of the
    /// PC's memory shared with the card (where the system moves what does
    /// not fit on the card).
    pub(crate) fn video_memory(&self) -> Option<[u64; 4]> {
        let adapter = self.adapter.as_ref()?;
        let mut own = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        let mut shared = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        // SAFETY: plain queries that fill our structures.
        unsafe {
            adapter
                .QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut own)
                .ok()?;
            adapter
                .QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &mut shared)
                .ok()?;
        }
        // The card's budget moved by 256 MB or more since the last look:
        // said at once, with what was in use at the last look, so the log
        // has the moment. (It fell from 11,242 MB to some 10,300 MB with
        // the first world frame in runs 179 to 185, and the memory line,
        // seconds later, was the first to show it. We reserve nothing
        // and make the swap chain once, at the start.)
        static LAST: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];
        let now = [own.Budget, own.CurrentUsage, shared.CurrentUsage];
        let last = [0, 1, 2].map(|i| LAST[i].swap(now[i], Ordering::Relaxed));
        if last[0] != 0 && last[0].abs_diff(own.Budget) >= (256u64 << 20) {
            tracing::info!(
                "the system changed the budget of the card's memory from {} MB to {} MB (frame {}); at the last look {} MB of the card's and {} MB of the PC's shared memory were in use",
                last[0] >> 20,
                own.Budget >> 20,
                crate::frame::current(),
                last[1] >> 20,
                last[2] >> 20
            );
            self.log_video_memory("right after that change");
        }
        Some([own.Budget, own.CurrentUsage, shared.Budget, shared.CurrentUsage])
    }

    /// Both of the system's memory groups for this process, in bytes: for
    /// the card's own memory and then for the PC's memory shared with the
    /// card, the budget, the use, what could still be reserved and what
    /// is reserved (the layer reserves none).
    pub(crate) fn video_memory_groups(&self) -> Option<[[u64; 4]; 2]> {
        let adapter = self.adapter.as_ref()?;
        let mut own = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        let mut shared = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        // SAFETY: plain queries that fill our structures.
        unsafe {
            adapter
                .QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut own)
                .ok()?;
            adapter
                .QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &mut shared)
                .ok()?;
        }
        let all = |m: &DXGI_QUERY_VIDEO_MEMORY_INFO| {
            [m.Budget, m.CurrentUsage, m.AvailableForReservation, m.CurrentReservation]
        };
        Some([all(&own), all(&shared)])
    }

    /// Logs [`Self::video_memory_groups`], with `when` it is said.
    pub(crate) fn log_video_memory(&self, when: &str) {
        let Some([own, shared]) = self.video_memory_groups() else {
            return;
        };
        tracing::info!(
            "video memory as the system counts it, {}: the card's own: budget {} MB, in use {} MB, {} MB could still be reserved, {} MB reserved; the PC's shared with the card: budget {} MB, in use {} MB, {} MB could still be reserved, {} MB reserved (the layer reserves none)",
            when,
            own[0] >> 20,
            own[1] >> 20,
            own[2] >> 20,
            own[3] >> 20,
            shared[0] >> 20,
            shared[1] >> 20,
            shared[2] >> 20,
            shared[3] >> 20
        );
    }

    /// A buffer in system memory that the CPU maps (write-back, so reading
    /// it is fast too) and shaders read and write: for host copies of title
    /// memory that shaders write, which upload memory cannot be. Starts in
    /// the common state (buffers move out of it implicitly).
    pub(crate) fn shared_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let made = self.create_shared_buffer(size).and_then(|buffer| {
            resident(&self.device, &buffer)?;
            mapped(&self.device)?;
            self.map_shared_buffer(buffer)
        });
        match made {
            Ok(pair) => Some(pair),
            Err(e) => {
                tracing::warn!(error = %e, size, "no shared buffer");
                None
            }
        }
    }

    fn map_shared_buffer(&self, buffer: ID3D12Resource) -> windows::core::Result<(ID3D12Resource, *mut u8)> {
        let mut data = std::ptr::null_mut();
        // SAFETY: CPU-visible L0 memory. The owner keeps the map alive.
        unsafe { buffer.Map(0, None, Some(&mut data))?; }
        #[cfg(feature = "diagnostics")]
        crate::book::made(&buffer, || "a shared buffer".into());
        Ok((buffer, data.cast()))
    }

    fn create_shared_buffer(&self, size: u64) -> windows::core::Result<ID3D12Resource> {
        MADE[2].fetch_add(1, Ordering::Relaxed);
        MADE[3].fetch_add(size, Ordering::Relaxed);
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        // SAFETY: plain creation with a live device and valid descriptor.
        unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    not_resident(),
                    &desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut buffer,
                )?;
        }
        buffer.ok_or_else(windows::core::Error::empty)
    }

    /// Whether buffers can lie in the card's memory and be mapped by the CPU
    /// (`D3D12_HEAP_TYPE_GPU_UPLOAD`: a resizable BAR, and a system and driver
    /// that offer it); logged once.
    pub(crate) fn mapped_card_memory(&self) -> bool {
        static THERE: OnceLock<bool> = OnceLock::new();
        *THERE.get_or_init(|| {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS16::default();
            // SAFETY: writes `options` of the given size.
            let ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS16,
                        (&raw mut options).cast(),
                        size_of_val(&options) as u32,
                    )
                    .is_ok()
            };
            let supported = ok && options.GPUUploadHeapSupported.as_bool();
            tracing::info!(supported, "card memory the CPU maps");
            supported
        })
    }

    /// A buffer in the card's own memory that the CPU maps, shaders read and
    /// (with `written`) may write. The CPU should only write it: a read of the
    /// mapping crosses the bus a few bytes at a time.
    pub(crate) fn mapped_card_buffer(&self, size: u64, written: bool) -> Option<(ID3D12Resource, *mut u8)> {
        if !self.mapped_card_memory() {
            return None;
        }
        MADE[4].fetch_add(1, Ordering::Relaxed);
        MADE[5].fetch_add(size, Ordering::Relaxed);
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_GPU_UPLOAD,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: if written {
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
            } else {
                D3D12_RESOURCE_FLAG_NONE
            },
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of a CPU-visible buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    not_resident(),
                    &desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    resident(&self.device, b)?;
                    mapped(&self.device)?;
                    b.Map(0, None, Some(&mut data))
                })
        };
        match made {
            Ok(()) => Some((buffer?, data.cast())),
            Err(e) => {
                tracing::warn!(error = %e, size, "no buffer in card memory the CPU maps");
                None
            }
        }
    }

    /// A device-local buffer shaders may write (unordered access) and indirect
    /// execution may read, starting in the common state.
    pub(crate) fn uav_buffer(&self, size: u64) -> Option<ID3D12Resource> {
        MADE[4].fetch_add(1, Ordering::Relaxed);
        MADE[5].fetch_add(size, Ordering::Relaxed);
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a device-local buffer.
        let made = unsafe {
            self.device.CreateCommittedResource(
                &heap,
                not_resident(),
                &desc,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut buffer,
            )
        };
        #[cfg(feature = "diagnostics")]
        if let Some(b) = &buffer {
            crate::book::made(b, || "an unordered access buffer".into());
        }
        let made = made.and_then(|()| buffer.as_ref().map_or(Ok(()), |b| resident(&self.device, b))).and_then(|()| mapped(&self.device));
        match made {
            Ok(()) => buffer,
            Err(e) => {
                tracing::warn!(error = %e, size, "no unordered-access buffer");
                None
            }
        }
    }
}

/// For adapters without device-removed data (the software one): the last
/// commands each list of the failed submission noted (diagnostics, with
/// `VICTORY_D3D12_DRED`).
fn last_notes(runs: &[&HostList]) -> String {
    #[cfg(feature = "diagnostics")]
    {
        let mut text = String::new();
        for (i, r) in runs.iter().enumerate() {
            let notes = r.notes.lock().unwrap_or_else(|e| e.into_inner());
            for (n, _) in notes.iter().rev().take(6).rev() {
                text.push_str(&format!("; list {i} of {}, of its {}: {n}", runs.len(), notes.len()));
            }
        }
        text
    }
    #[cfg(not(feature = "diagnostics"))]
    {
        let _ = runs;
        String::new()
    }
}

/// Where the faulting address `va` lies against a range `(start, bytes)` a
/// command reads (diagnostics).
#[cfg(feature = "diagnostics")]
fn fault_in(va: u64, (start, bytes): (u64, u64)) -> String {
    let end = start + bytes;
    if va < start {
        format!(" (fault {:#x} below its source)", start - va)
    } else if va < end {
        format!(" (fault at source +{:#x})", va - start)
    } else {
        format!(" (fault {:#x} past its source's end)", va - end)
    }
}

/// The commands that may have been running where a list stopped at
/// command `done`: those between the barriers around it (work between
/// barriers overlaps on the GPU), by kind, and what the markers just
/// before them say (`HostList::note_op`: diagnostics builds with
/// device-removed data on).
/// For a list that stopped at command `done`: what the indirect executions
/// there ran, from the one before it to the next barrier (four at most), and
/// what the heap slots their user data names hold now.
#[cfg(feature = "diagnostics")]
fn indirects_around(history: &[D3D12_AUTO_BREADCRUMB_OP], done: usize, all: &indirects::Shared) -> String {
    let indirect = |op: &D3D12_AUTO_BREADCRUMB_OP| *op == D3D12_AUTO_BREADCRUMB_OP_EXECUTEINDIRECT;
    let all = all.lock().unwrap_or_else(|e| e.into_inner());
    let mut text = String::new();
    let mut shown = 0;
    let mesh = |op: &D3D12_AUTO_BREADCRUMB_OP| *op == D3D12_AUTO_BREADCRUMB_OP_DISPATCHMESH;
    for (at, op) in history.iter().enumerate().skip(done.saturating_sub(1)) {
        if shown == 4 || (at > done && *op == D3D12_AUTO_BREADCRUMB_OP_RESOURCEBARRIER) {
            break;
        }
        if !indirect(op) && !mesh(op) {
            continue;
        }
        shown += 1;
        let direct = mesh(op);
        let nth = if direct { indirects::before(history, at, mesh) } else { indirects::before(history, at, indirect) };
        let Some((what, words)) = all.nth(nth, direct) else {
            continue;
        };
        text.push_str(&format!("; command {at} ran {what}"));
        for (word, slot, what) in crate::descriptor::host::describe_slots(words) {
            text.push_str(&format!(", word {word} slot {slot}: {what}"));
        }
    }
    text
}

/// The stop line for a list with `done` breadcrumbs completed. Microsoft
/// documents `pLastBreadcrumbValue` as the *number* of completed ops: the
/// last one that finished is at `done - 1`, and `history[done]` is the
/// first that had not completed. The old "stopped at command {done} of
/// {count}" read as one-based numbering and was taken to name the last
/// completed command, so say both explicitly, with zero-based indices.
/// `None` when the runtime reports no such command in `history`.
fn stopped_command(
    history: &[D3D12_AUTO_BREADCRUMB_OP],
    done: u32,
    count: u32,
) -> Option<String> {
    if done >= count || done as usize >= history.len() {
        return None;
    }
    let op = history[done as usize].0;
    Some(if done == 0 {
        format!("; no command completed of {count}; the first uncompleted is command index 0 (op {op})")
    } else {
        let last = done - 1;
        let last_op = history[last as usize].0;
        format!(
            "; {done} of {count} commands completed, last at index {last} (op {last_op}); the first uncompleted is command index {done} (op {op})"
        )
    })
}

#[cfg(test)]
mod breadcrumbs {
    use super::stopped_command;
    use windows::Win32::Graphics::Direct3D12::D3D12_AUTO_BREADCRUMB_OP;

    /// The op named is the first one that had *not* completed, at a
    /// zero-based index. This holds when none ran and when the last
    /// breadcrumb is the one that stopped the list.
    #[test]
    fn the_named_op_is_the_first_command_that_had_not_completed() {
        let op = D3D12_AUTO_BREADCRUMB_OP;
        let history = [op(46), op(3), op(15), op(6)];

        assert_eq!(
            stopped_command(&history, 0, 4).as_deref(),
            Some("; no command completed of 4; the first uncompleted is command index 0 (op 46)")
        );
        assert_eq!(
            stopped_command(&history, 1, 4).as_deref(),
            Some("; 1 of 4 commands completed, last at index 0 (op 46); the first uncompleted is command index 1 (op 3)")
        );
        assert_eq!(
            stopped_command(&history, 3, 4).as_deref(),
            Some("; 3 of 4 commands completed, last at index 2 (op 15); the first uncompleted is command index 3 (op 6)")
        );

        // A finished list has no stop command; nor does a driver that
        // reports more breadcrumbs than it handed us, or u32::MAX.
        assert_eq!(stopped_command(&history, 4, 4), None);
        assert_eq!(stopped_command(&history, 4, 8), None);
        assert_eq!(stopped_command(&history, u32::MAX, 4), None);
    }
}

fn in_flight(history: &[D3D12_AUTO_BREADCRUMB_OP], markers: &[(u32, String)], done: usize) -> String {
    let barrier = |op: &D3D12_AUTO_BREADCRUMB_OP| *op == D3D12_AUTO_BREADCRUMB_OP_RESOURCEBARRIER;
    let from = history[..done].iter().rposition(barrier).map_or(0, |b| b + 1);
    let to = history[done..]
        .iter()
        .position(barrier)
        .map_or(history.len(), |b| (done + b).max(done + 1));
    let mut text = format!(", commands {from}..{to} between its barriers:");
    let mut shown = 0;
    // From just before the command the GPU had not finished: the first
    // twelve after the barrier named draws long done when a list of 79
    // mesh draws stopped at its 45th (run 41).
    for i in from.max(done.saturating_sub(4))..to {
        let op = history[i];
        if op == D3D12_AUTO_BREADCRUMB_OP_SETMARKER || shown == 12 {
            continue;
        }
        shown += 1;
        let marker = i
            .checked_sub(1)
            .filter(|&m| history[m] == D3D12_AUTO_BREADCRUMB_OP_SETMARKER)
            .and_then(|m| markers.iter().find(|(at, _)| *at as usize == m))
            .map_or("no marker", |(_, s)| s.as_str());
        text.push_str(&format!("; command {i} (op {}): {marker}", op.0));
    }
    text
}

pub(crate) fn transition(
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: a borrowed pointer; the barrier never releases it.
                pResource: unsafe { std::mem::transmute_copy(resource) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

/// A window of the given client size, owned by its own message thread.
fn window(width: u32, height: u32) -> windows::core::Result<HWND> {
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name("victory-window".into())
        .spawn(move || {
            let made = make_window(width, height);
            let ok = made.is_ok();
            let _ = sender.send(made.map(|h| h.0 as usize));
            if ok {
                pump();
            }
        })
        .map_err(|_| windows::core::Error::empty())?;
    let hwnd = receiver
        .recv()
        .map_err(|_| windows::core::Error::empty())??;
    Ok(HWND(hwnd as *mut _))
}

extern "system" fn window_proc(
    hwnd: HWND,
    message: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    // SAFETY: the default handling for our window.
    unsafe { DefWindowProcW(hwnd, message, wparam, lparam) }
}

fn make_window(width: u32, height: u32) -> windows::core::Result<HWND> {
    const CLASS: PCWSTR = w!("VIctory");
    // SAFETY: registers our class once and creates a window of it.
    unsafe {
        let instance = GetModuleHandleW(None)?;
        let class = WNDCLASSEXW {
            cbSize: size_of::<WNDCLASSEXW>() as u32,
            lpfnWndProc: Some(window_proc),
            hInstance: instance.into(),
            lpszClassName: CLASS,
            hCursor: LoadCursorW(None, IDC_ARROW)?,
            ..Default::default()
        };
        RegisterClassExW(&class);
        let mut rect = RECT {
            left: 0,
            top: 0,
            right: width as i32,
            bottom: height as i32,
        };
        AdjustWindowRect(&mut rect, WS_OVERLAPPEDWINDOW, false)?;
        CreateWindowExW(
            WINDOW_EX_STYLE(0),
            CLASS,
            w!("VIctory"),
            WS_OVERLAPPEDWINDOW | WS_VISIBLE,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            rect.right - rect.left,
            rect.bottom - rect.top,
            None,
            None,
            Some(instance.into()),
            None,
        )
    }
}

fn pump() {
    let mut message = MSG::default();
    // SAFETY: the standard message loop of this thread's window.
    unsafe {
        while GetMessageW(&mut message, None, 0, 0).as_bool() {
            let _ = TranslateMessage(&message);
            DispatchMessageW(&message);
        }
    }
}

#[cfg(test)]
mod trail {
    /// The trail keeps the newest things, and the report made when the
    /// host device is removed says them and the pipeline library's state
    /// (no card is asked for any of it).
    #[test]
    fn the_removal_report_says_the_newest_things_done() {
        for n in 0..super::TRAIL_MOST + 5 {
            super::trail(|| format!("thing {n} done"));
        }
        let text = super::removal_context();
        assert!(text.contains("the submitting thread was"));
        assert!(text.contains("pipeline library: no call running"));
        assert!(text.contains(&format!("thing {} done", super::TRAIL_MOST + 4)));
        assert!(!text.contains("thing 4 done"), "the oldest entries are dropped");
    }
}

#[cfg(test)]
mod spare {
    use super::Spare;

    /// Kept things come back by size, the last given first, and nothing is
    /// kept past the limit.
    #[test]
    fn kept_things_come_back_by_size_within_the_limit() {
        let mut spare: Spare<u32> = Spare::new();
        assert_eq!(spare.take(4), None);
        assert_eq!(spare.give(1, 4, 10), None);
        assert_eq!(spare.give(2, 1, 10), None);
        assert_eq!(spare.give(3, 4, 10), None);
        assert_eq!(spare.bytes, 9);
        // Past the limit: handed back, nothing changes.
        assert_eq!(spare.give(4, 4, 10), Some(4));
        assert_eq!(spare.bytes, 9);
        assert_eq!(spare.take(2), None, "no thing of that size");
        assert_eq!(spare.take(4), Some(3));
        assert_eq!(spare.take(4), Some(1));
        assert_eq!(spare.take(4), None);
        assert_eq!(spare.take(1), Some(2));
        assert_eq!(spare.bytes, 0);
        assert!(spare.kept.is_empty());
    }
}

#[cfg(test)]
mod refresh_tests {
    use super::Batch;
    use crate::shadow::Shadow;
    use std::sync::Arc;
    use windows_sys::Win32::System::Memory::{
        CreateFileMappingW, FILE_MAP_WRITE, MEM_COMMIT, MEM_RESERVE, MapViewOfFile, PAGE_READWRITE,
        VirtualAlloc,
    };

    const PAGE: usize = 64 << 10;
    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    /// Title memory for the tests (never given back: copies of it stay in
    /// the guards): write-watched, or a view of a section.
    fn title(len: usize, view: bool) -> usize {
        // SAFETY: plain allocations.
        let at = unsafe {
            if view {
                let section =
                    CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE, 0, len as u32, std::ptr::null());
                let at = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, len).Value as usize;
                // As the runtime tells of it.
                crate::title_memory_section(section as usize, len as u64, true);
                crate::title_memory_remapped(at, len);
                crate::title_memory_view(at, len, section as usize, 0);
                at
            } else {
                VirtualAlloc(std::ptr::null(), len, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) as usize
            }
        };
        assert_ne!(at, 0);
        at
    }

    /// Title memory of `blocks` pages in a placeholder, one view to a page,
    /// and every fourth page not there: as a title that streams has it.
    fn with_holes(blocks: usize) -> usize {
        use windows_sys::Win32::System::Memory::{
            MEM_PRESERVE_PLACEHOLDER, MEM_RELEASE, MEM_REPLACE_PLACEHOLDER, MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS,
            VirtualAlloc2, VirtualFree,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: a placeholder and a section of our own.
        unsafe {
            let at = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                blocks * PAGE,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(at, 0);
            let len = blocks * PAGE;
            let section =
                CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE, (len >> 32) as u32, len as u32, std::ptr::null());
            for block in (0..blocks).filter(|b| b % 4 != 3) {
                let to = at + block * PAGE;
                VirtualFree(to as *mut _, PAGE, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER);
                let view = MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    to as *const _,
                    (block * PAGE) as u64,
                    PAGE,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                );
                assert_eq!(view.Value as usize, to);
                crate::title_memory_remapped(to, PAGE);
            }
            at
        }
    }

    fn memory(len: usize) -> *mut u8 {
        // SAFETY: a plain allocation.
        unsafe { VirtualAlloc(std::ptr::null(), len, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE).cast() }
    }

    #[test]
    fn raw_title_copy_preserves_overlap_in_either_direction() {
        let _one = crate::watch::one_asking_test();
        let input: Vec<u8> = (0..12_555).map(|i| (i % 251) as u8).collect();
        for (src, dst) in [(0, 13), (13, 0)] {
            let mut expected = input.clone();
            expected.copy_within(src..src + 12_000, dst);
            let mut actual = input.clone();
            let at = actual.as_mut_ptr() as usize;
            super::copy_raw_title(at + dst, at + src, 12_000);
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn raw_title_copy_keeps_write_watch_and_resends_equal_bytes() {
        let _one = crate::watch::one_asking_test();
        let dst = title(PAGE, false);
        let value = [0u8; 16];
        let held = memory(PAGE);
        assert!(!held.is_null());
        let shadow = Shadow::unbacked(dst as u64, held, PAGE as u64, true);
        let _tracked = shadow.track_carded_test();
        crate::watch::ask_once(&[(dst, dst + PAGE)]);
        let before = crate::watch::asked(dst, dst + PAGE).unwrap().0;
        // This case exercises a covered copy with a stable mapping.
        // Other GPU fixtures can be remapping unrelated pages concurrently.
        super::copy_raw_title_using_mapping(dst, value.as_ptr() as usize, value.len(), || {
            Some(crate::watch::title_memory_reading())
        });
        let after = crate::watch::asked(dst, dst + PAGE);
        crate::watch::ask_once_end();
        let (asking, notes) = after.expect("the copy preserves the query");
        assert!(asking > before);
        assert_eq!(notes, [(dst / PAGE, asking)]);
        if super::send::own() {
            assert_eq!(shadow.send_up(1, 0), [(0, value.len())]);
        }
    }

    #[test]
    fn raw_title_copy_tolerates_decommit_after_a_mapping_check() {
        use super::{copy_raw_spans, copy_raw_title};
        use windows_sys::Win32::System::Memory::{MEM_DECOMMIT, MEM_RELEASE, VirtualFree};
        let _one = crate::watch::one_asking_test();
        let len = 2 * crate::watch::RAW_COPY_WATCH_LIMIT;
        let src = memory(len) as usize;
        assert_ne!(src, 0);
        let mut dst = vec![0xCCu8; len];
        let dst_at = dst.as_mut_ptr() as usize;
        let second = src + len / 2;
        assert!(crate::texture::committed(src as *const u8, len));
        // SAFETY: these pages are in the test's allocation.
        assert_ne!(unsafe { VirtualFree(second as *mut _, len / 2, MEM_DECOMMIT) }, 0);
        copy_raw_title(dst_at, src, len);
        assert!(dst.iter().all(|&b| b == 0xCC));
        let regions = copy_raw_spans(dst_at, src, len);
        let copied = regions[0].len();
        assert!(copied <= len / 2 && regions[1].is_empty());
        if crate::safecopy::on() {
            assert_eq!(copied, len / 2);
        }
        assert!(dst[..copied].iter().all(|&b| b == 0));
        assert!(dst[copied..].iter().all(|&b| b == 0xCC));

        // Repeated decommits outside the mapping lock can race the entire
        // raw-copy path, including preparation of a still-mapped span.
        let start = std::sync::Barrier::new(2);
        std::thread::scope(|scope| {
            scope.spawn(|| {
                start.wait();
                for _ in 0..64 {
                    // SAFETY: commit and decommit one part of this reservation.
                    unsafe {
                        assert_eq!(VirtualAlloc(second as *const _, len / 2, MEM_COMMIT, PAGE_READWRITE) as usize, second);
                        assert_ne!(VirtualFree(second as *mut _, len / 2, MEM_DECOMMIT), 0);
                    }
                }
            });
            start.wait();
            for _ in 0..64 {
                copy_raw_title(dst_at, src, len);
            }
        });
        // SAFETY: all accesses, including the other thread's, are finished.
        assert_ne!(unsafe { VirtualFree(src as *mut _, 0, MEM_RELEASE) }, 0);
    }

    struct Random(u64);

    impl Random {
        fn below(&mut self, n: usize) -> usize {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as usize % n
        }
    }

    /// Prepares as a submission does: one asking for all the ranges, the
    /// refreshes in several flushes, the asking's end.
    fn submission(ranges: &[(Arc<Shadow>, usize, usize)], flushes: usize) {
        // One at a time, as submissions are.
        static ONE: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _one = ONE.lock().unwrap_or_else(|e| e.into_inner());
        let asked: Vec<(usize, usize)> = ranges.iter().filter(|r| !r.0.unwatched()).map(|r| (r.1, r.1 + r.2)).collect();
        crate::watch::ask_once(&asked);
        let mut batch = Batch::default();
        for part in ranges.chunks(ranges.len().div_ceil(flushes).max(1)) {
            for (shadow, at, len) in part {
                batch.add(shadow, *at, *len);
            }
            batch.flush(None);
        }
        crate::watch::ask_once_end();
    }

    /// Whatever is written where, by the title or by us, and whichever
    /// copies (of both kinds of memory, over one another, none on a page
    /// line) are refreshed in which ranges: a refreshed range holds what
    /// title memory holds.
    #[test]
    fn refreshed_ranges_hold_title_memory() {
        let _one = crate::watch::one_asking_test();
        // SAFETY (all blocks): title memory and copies of the test's own,
        // written by this thread and, in `Batch::flush`, by the threads that
        // refresh, which are done when it returns.
        let mut random = Random(0x1234_5678_9ABC_DEF1);
        const LEN: usize = 24 * PAGE;
        let regions = [title(LEN, false), title(LEN, true)];
        let mut copies = Vec::new();
        for &region in &regions {
            for (from, to) in [(0, LEN), (100, LEN - 5000), (3 * PAGE + 777, 9 * PAGE + 10), (8 * PAGE, 8 * PAGE + 300), (PAGE, 20 * PAGE)] {
                let data = memory(to - from);
                copies.push((Shadow::unbacked((region + from) as u64, data, (to - from) as u64, false), data));
            }
        }
        let byte = |at: usize| unsafe { (at as *const u8).read_volatile() };
        for round in 0..600 {
            for _ in 0..random.below(4) {
                let at = regions[random.below(2)] + random.below(LEN - 64);
                let n = 1 + random.below(60);
                if random.below(3) == 0 {
                    // What shaders wrote.
                    let was: Vec<u8> = (0..n).map(|i| byte(at + i)).collect();
                    let now: Vec<u8> = was.iter().map(|b| if random.below(2) == 0 { *b } else { random.below(256) as u8 }).collect();
                    Shadow::put(at, &was, &now);
                    assert!(now.iter().enumerate().all(|(i, &b)| byte(at + i) == b), "not put");
                    continue;
                }
                for i in 0..n {
                    unsafe { ((at + i) as *mut u8).write_volatile(random.below(256) as u8) };
                }
            }
            let mut ranges = Vec::new();
            for (shadow, _) in &copies {
                if random.below(3) == 0 {
                    continue;
                }
                let (start, len) = shadow.span();
                let (start, len) = (start as usize, len as usize);
                // Often the range of the time before, as a list's is.
                let from = if random.below(2) == 0 { 0 } else { random.below(len) };
                let n = if random.below(2) == 0 { len - from } else { 1 + random.below(len - from) };
                ranges.push((shadow.clone(), start + from, n));
            }
            if ranges.is_empty() {
                continue;
            }
            let flushes = 1 + random.below(3);
            submission(&ranges, flushes);
            for (shadow, at, n) in &ranges {
                let data = copies.iter().find(|c| Arc::ptr_eq(&c.0, shadow)).unwrap().1;
                let start = shadow.span().0 as usize;
                let same = unsafe {
                    std::slice::from_raw_parts(*at as *const u8, *n) == std::slice::from_raw_parts(data.add(at - start), *n)
                };
                assert!(same, "round {round}: {at:#x}+{n:#x} of a copy at {start:#x}; first byte {}", byte(*at));
            }
        }
    }

    /// A copy in the card's memory, with the card's side and the sends to it
    /// played by the test.
    struct Carded {
        shadow: Arc<Shadow>,
        data: usize,
        start: usize,
        len: usize,
        /// The last send, the last send every fetch to come is behind, and
        /// the card's bytes: the queue's order is the order it is locked in.
        card: std::sync::Mutex<(u64, u64, Vec<u8>)>,
        /// Whether a flight that writes it has not landed.
        flying: std::sync::atomic::AtomicBool,
        /// The dword of each eight its shaders write.
        class: usize,
    }

    impl Carded {
        fn new(start: usize, len: usize, class: usize) -> Carded {
            let data = memory(len);
            Carded {
                shadow: Shadow::unbacked(start as u64, data, len as u64, true),
                data: data as usize,
                start,
                len,
                card: std::sync::Mutex::new((0, 0, vec![0; len])),
                flying: std::sync::atomic::AtomicBool::new(false),
                class,
            }
        }

        /// What changed on the CPU's side goes up.
        fn send(&self) {
            let mut card = self.card.lock().unwrap();
            card.0 += 1;
            let (number, merged) = (card.0, card.1);
            for (a, b) in self.shadow.send_up(number, merged) {
                // SAFETY: the copy's bytes; nothing writes those that are to
                // go up.
                unsafe { std::ptr::copy_nonoverlapping((self.data + a) as *const u8, card.2[a..b].as_mut_ptr(), b - a) };
            }
        }
    }

    /// Everything at once: the title writes its memory on two threads, the
    /// submitting thread puts its bytes in under shader writes
    /// (`Shadow::merge_under_writes`), refreshes copies that overlap (two in
    /// the card's memory, one not), sends up what changed and declares what
    /// the next flight's shaders write, and what flights wrote is landed
    /// meanwhile (`crate::land`, beside the guards). When all of it stands
    /// still and the copies are refreshed once more, each of them, and the
    /// card's side of those that have one, hold what title memory holds, and
    /// title memory what was written last.
    ///
    /// Into the first copy shaders write besides through a view nobody
    /// declared, words the title does not write: nothing that goes up ever
    /// puts an older value over those.
    ///
    /// (It fails in the first rounds if `landed_beside` notes nothing, or
    /// only takes the block out of the guards' clean ones and leaves its bit,
    /// or if a fault leaves the bit; in some thousand rounds,
    /// `VICTORY_TEST_ROUNDS`, if what is landed is noted before it is in.)
    #[test]
    fn landings_title_writes_and_light_refreshes_at_once() {
        let _one = crate::watch::one_asking_test();
        use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
        assert!(crate::safecopy::on());
        const LEN: usize = 24 * PAGE;
        let rounds: usize = std::env::var("VICTORY_TEST_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(120);
        let region = title(LEN, true);
        let dword = |at: usize| unsafe { &*((region + at) as *const AtomicU32) };
        let carded = [Carded::new(region, LEN, 3), Carded::new(region + 4 * PAGE + 1000, LEN - 6 * PAGE - 4, 5)];
        let plain = {
            let (start, len) = (region + 700, LEN - 9000);
            let data = memory(len);
            (Shadow::unbacked(start as u64, data, len as u64, false), start, len, data as usize)
        };
        // What was last written so, by offset.
        let mut undeclared = std::collections::HashMap::new();
        let exact = std::env::var_os("VICTORY_SEND_CHANGED").is_some_and(|v| v != "0");
        let stop = AtomicBool::new(false);
        // Also when an assertion fails on one of the threads.
        struct Stop<'a>(&'a AtomicBool);
        impl Drop for Stop<'_> {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Relaxed);
            }
        }
        // Held shared by who writes title memory, alone while it is looked at.
        let still = std::sync::RwLock::new(());
        let (fly, flights) = std::sync::mpsc::channel::<(usize, Vec<(usize, usize)>, u32)>();
        std::thread::scope(|s| {
            for t in 0..2usize {
                let (stop, still) = (&stop, &still);
                s.spawn(move || {
                    let mut random = Random(0x7777_1234_0000_0001 + t as u64 * 0x1111);
                    let mut value = 0u32;
                    while !stop.load(Ordering::Relaxed) {
                        {
                            let _moving = still.read().unwrap();
                            for _ in 0..3 {
                                value += 1;
                                // Of each eight dwords, one a thread.
                                let at = random.below(LEN / 32) * 32 + 4 * (1 + t);
                                dword(at).store(value, Ordering::Relaxed);
                            }
                        }
                        std::thread::sleep(std::time::Duration::from_micros(20));
                    }
                });
            }
            // Lands flights.
            {
                let (carded, stop) = (&carded, &stop);
                s.spawn(move || {
                    let _stop = Stop(stop);
                    while let Ok((which, ranges, value)) = flights.recv() {
                        let c = &carded[which];
                        // Fetched: the card's bytes, behind every send so far.
                        let (after, now) = {
                            let card = c.card.lock().unwrap();
                            (card.0, card.2.clone())
                        };
                        // By piece: 64 KiB of title memory.
                        let mut pieces = Vec::new();
                        for &(from, to) in &ranges {
                            let mut at = from;
                            while at < to {
                                let next = ((c.start + at) / PAGE * PAGE + PAGE - c.start).min(to);
                                pieces.push((at, next, now.as_ptr() as usize + at));
                                at = next;
                            }
                        }
                        let parts: Vec<_> = pieces.chunks(3).collect();
                        crate::land::each(parts.len(), &|i| {
                            let mut notes = crate::land::Notes::default();
                            // SAFETY: the bytes of `now`; each piece once.
                            unsafe { c.shadow.land_card(parts[i].iter().copied(), after, &mut notes) };
                            notes.flush();
                        });
                        for &(from, to) in &ranges {
                            for at in (from..to).filter(|at| (c.start - region + at) % 32 == 4 * c.class) {
                                // (The next fetch may have brought more.)
                                assert!(dword(c.start - region + at).load(Ordering::Relaxed) >= value, "not landed");
                            }
                        }
                        let done: Vec<_> = ranges.iter().map(|&(a, b)| (c.start + a, c.start + b)).collect();
                        c.shadow.copy_back_done(&done);
                        c.card.lock().unwrap().1 = after;
                        c.flying.store(false, Ordering::Release);
                    }
                });
            }
            let _stop = Stop(&stop);
            let mut random = Random(0xABCD_EF01_2345_6789);
            let check = |round: usize, random: &mut Random, quiet: bool| {
                for c in &carded {
                    if c.shadow.writes_pending() {
                        let mut runs = usize::MAX;
                        assert!(!c.shadow.merge_under_writes(c.start, c.len, &mut runs), "round {round}");
                    }
                }
                let mut ranges = Vec::new();
                for (shadow, start, len) in
                    carded.iter().map(|c| (&c.shadow, c.start, c.len)).chain([(&plain.0, plain.1, plain.2)])
                {
                    // All of it, or a part; often the same part.
                    let (from, n) = match random.below(3) {
                        _ if quiet => (0, len),
                        0 => (0, len),
                        1 => (len / 3 / 4 * 4, len / 2 / 4 * 4),
                        _ => {
                            let from = random.below(len / 4) * 4;
                            (from, (4 + random.below((len - from) / 4) * 4).min(len - from))
                        }
                    };
                    ranges.push((shadow.clone(), start + from, n));
                }
                submission(&ranges, 1 + random.below(3));
                carded.iter().for_each(Carded::send);
            };
            let mut value = 1u32;
            for round in 0..rounds {
                check(round, &mut random, false);
                if exact {
                    let c = &carded[0];
                    let mut card = c.card.lock().unwrap();
                    for _ in 0..40 {
                        value += 1;
                        let at = random.below(LEN / 32) * 32 + 4 * 6;
                        card.2[at..at + 4].copy_from_slice(&value.to_le_bytes());
                        undeclared.insert(at, value);
                    }
                }
                // The flight of this submission: its shaders' writes.
                for (which, c) in carded.iter().enumerate() {
                    if c.flying.swap(true, Ordering::Acquire) {
                        continue;
                    }
                    value += 1;
                    let mut ranges: Vec<(usize, usize)> = Vec::new();
                    let first = random.below(6) * PAGE / 2;
                    for k in 0..2 {
                        let from = first + k * 9 * PAGE + 4 * random.below(9000);
                        let to = (from + 4 * (1 + random.below(7 * PAGE / 4))).min(c.len);
                        if from < to {
                            ranges.push((from, to));
                        }
                    }
                    for &(from, to) in &ranges {
                        c.shadow.writing(c.start + from, to - from);
                    }
                    {
                        let mut card = c.card.lock().unwrap();
                        for &(from, to) in &ranges {
                            for at in (from..to).filter(|at| (c.start - region + at) % 32 == 4 * c.class) {
                                card.2[at..at + 4].copy_from_slice(&value.to_le_bytes());
                            }
                        }
                    }
                    fly.send((which, ranges, value)).unwrap();
                }
                if round % 20 != 19 {
                    continue;
                }
                // Still: no flight in the air, no title write.
                let _still = still.write().unwrap();
                while carded.iter().any(|c| c.flying.load(Ordering::Acquire)) {
                    assert!(!stop.load(Ordering::Relaxed), "the landing thread failed");
                    std::thread::yield_now();
                }
                check(round, &mut random, true);
                // SAFETY: the test's own memory, written by nobody now.
                let bytes = |at: usize, len: usize| unsafe { std::slice::from_raw_parts(at as *const u8, len) };
                let first = |a: &[u8], b: &[u8]| a.iter().zip(b).position(|(x, y)| x != y);
                for c in &carded {
                    let (held, there) = (bytes(c.data, c.len), bytes(c.start, c.len));
                    assert_eq!(first(held, there), None, "round {round}: a card copy's CPU side (class {})", c.class);
                    let card = c.card.lock().unwrap();
                    let mut side = card.2.clone();
                    if c.class == carded[0].class {
                        // (They may have been landed meanwhile, by a flight
                        // that declared their range, and may not.)
                        for (&at, &value) in &undeclared {
                            let on_card = u32::from_le_bytes(side[at..at + 4].try_into().unwrap());
                            assert_eq!(on_card, value, "round {round}: an older value at {at:#x}");
                            side[at..at + 4].copy_from_slice(&there[at..at + 4]);
                        }
                    }
                    assert_eq!(first(&side, there), None, "round {round}: a card copy's card side (class {})", c.class);
                }
                assert_eq!(first(bytes(plain.3, plain.2), bytes(plain.1, plain.2)), None, "round {round}: the other copy");
            }
            drop(fly);
        });
    }

    /// The title's bytes put in under shader writes, for a copy of 174 MB
    /// whose shader writes were declared as 40 ranges that overlap, with 12
    /// pages written since. Run with `--ignored --nocapture`, and with
    /// `VICTORY_LIGHT_REFRESH=0` for the way it was.
    #[test]
    #[ignore]
    fn bytes_under_overlapping_ranges() {
        const BIG: usize = 174 << 20;
        let big = title(BIG, true);
        // SAFETY (both): the test's own memory.
        unsafe { std::ptr::write_bytes(big as *mut u8, 1, BIG) };
        let copy = Shadow::unbacked(big as u64, memory(BIG), BIG as u64, true);
        copy.update(big, BIG);
        copy.take_dirty();
        for i in 0..40 {
            copy.writing(big + i * 5920, BIG - i * 5920);
        }
        let mut best = f64::MAX;
        for round in 0..30 {
            for i in 0..12 {
                unsafe { ((big + (i * 211 + round) % 2700 * PAGE + 64) as *mut u8).write_volatile(round as u8) };
            }
            let started = std::time::Instant::now();
            let mut runs = usize::MAX;
            assert!(!copy.merge_under_writes(big, BIG, &mut runs));
            best = best.min(started.elapsed().as_secs_f64() * 1e3);
            copy.take_dirty();
        }
        println!("VICTORY_LIGHT_REFRESH {}: best {best:.3} ms", crate::watch::light());
    }

    /// A range nothing wrote is known to be as it was, at its ends too, and
    /// is so no more once something wrote it.
    #[test]
    fn unchanged_ranges_are_known() {
        let _one = crate::watch::one_asking_test();
        for view in [false, true] {
            let region = title(8 * PAGE, view);
            // (A page read for the first time counts as written.)
            // SAFETY: the test's own memory.
            unsafe { std::ptr::write_bytes(region as *mut u8, 1, 8 * PAGE) };
            let data = memory(8 * PAGE);
            let copy = Shadow::unbacked(region as u64, data, (8 * PAGE) as u64, false);
            // Ends in the middle of pages.
            let (at, len) = (region + 500, 5 * PAGE);
            let look = |expected: bool, what: &str| {
                // The submission's one asking, which takes two ranges. (Again
                // if a submission of another test ended it.)
                let mut known = false;
                for _ in 0..100 {
                    crate::watch::ask_once(&[(at, at + len), (region + 7 * PAGE, region + 8 * PAGE)]);
                    known = copy.nothing_new(at, len);
                    if known || !expected {
                        break;
                    }
                }
                assert_eq!(known, expected, "view {view}: {what}");
                let copied = copy.update(at, len);
                crate::watch::ask_once_end();
                copied
            };
            assert_eq!(look(false, "never taken"), len);
            assert_eq!(look(true, "taken"), 0);
            assert_eq!(look(true, "taken"), 0);
            // SAFETY: the test's own memory.
            unsafe { ((region + 600) as *mut u8).write_volatile(1) };
            // (The kernel names the 4 KiB, the guards only the block.)
            assert_eq!(look(false, "written at its end"), if view { PAGE } else { 4096 } - 500);
            assert_eq!(look(true, "taken again"), 0);
            // Outside the range, on its last page.
            // SAFETY: as above.
            unsafe { ((region + 5 * PAGE + 600) as *mut u8).write_volatile(1) };
            assert_eq!(look(false, "written on its last page"), 500);
            // A longer range was never taken whole.
            assert!(!copy.nothing_new(region, 6 * PAGE));
        }
    }

    /// Of a copy in the card's memory only the dwords title memory changed go
    /// up, and none beside them, not even between two in 64 bytes; a page
    /// never taken goes up whole.
    #[test]
    fn only_what_differs_goes_up() {
        let region = title(4 * PAGE, true);
        let data = memory(4 * PAGE);
        let copy = Shadow::unbacked(region as u64, data, (4 * PAGE) as u64, true);
        copy.update(region, 4 * PAGE);
        assert_eq!(copy.take_dirty(), [(0, 4 * PAGE)], "all of it, though zeros are no change");
        // What the relay does: shader writes into title memory, and into the
        // CPU's side.
        crate::watch::landing(region + PAGE + 640, 64);
        // SAFETY: the test's own memory.
        unsafe {
            std::ptr::write_bytes((region + PAGE + 640) as *mut u8, 7, 64);
            std::ptr::write_bytes(data.add(PAGE + 640), 7, 64);
        }
        copy.update(region, 4 * PAGE);
        let again = copy.take_dirty();
        // SAFETY: as above.
        unsafe {
            ((region + 2 * PAGE + 129) as *mut u8).write_volatile(9);
            ((region + 2 * PAGE + 136) as *mut u32).write_volatile(9);
            ((region + 2 * PAGE + 140) as *mut u32).write_volatile(9);
            // The value that was there.
            ((region + 2 * PAGE + 132) as *mut u32).write_volatile(0);
        }
        copy.update(region, 4 * PAGE);
        let written = copy.take_dirty();
        if std::env::var_os("VICTORY_SEND_CHANGED").is_some_and(|v| v != "0") {
            assert_eq!(again, []);
            assert_eq!(written, [(2 * PAGE + 128, 2 * PAGE + 132), (2 * PAGE + 136, 2 * PAGE + 144)]);
        } else {
            assert_eq!(again, [(PAGE, 2 * PAGE)]);
            assert_eq!(written, [(2 * PAGE, 3 * PAGE)]);
        }
        // SAFETY: as above.
        assert_eq!(unsafe { data.add(2 * PAGE + 129).read() }, 9);
    }

    /// The refreshes of a frame as run record-1 had them: 30 submissions, the
    /// first with a copy for each of 4,400 ranges, the others with 60 and a
    /// big copy that 55 landed pieces and 9 title writes opened, while a
    /// thread lands what shaders wrote. Run with `--ignored --nocapture`,
    /// and with `VICTORY_LIGHT_REFRESH=0` for the way it was.
    #[test]
    #[ignore]
    fn frame_of_refreshes() {
        let _one = crate::watch::one_asking_test();
        let mut random = Random(0x9876_5432_1234_5671);
        const EACH: usize = 224 << 10;
        const BIG: usize = 174 << 20;
        let watched = title(3000 * EACH, false);
        // `VICTORY_BENCH_HOLES=0`: views without holes.
        let holes = std::env::var_os("VICTORY_BENCH_HOLES").is_none_or(|v| v != "0");
        let views = if holes { with_holes(1400 * EACH / PAGE) } else { title(1400 * EACH, true) };
        let there = |at: usize| !holes || (at - views) / PAGE % 4 != 3;
        let big = title(BIG, true);
        // SAFETY: the test's own memory.
        unsafe {
            std::ptr::write_bytes(watched as *mut u8, 1, 3000 * EACH);
            for at in (views..views + 1400 * EACH).step_by(PAGE).filter(|&at| there(at)) {
                std::ptr::write_bytes(at as *mut u8, 1, PAGE);
            }
            std::ptr::write_bytes(big as *mut u8, 1, BIG);
        }
        let mut small = Vec::new();
        for i in 0..4400 {
            let (start, at) = if i < 3000 { (watched, i * EACH) } else { (views, (i - 3000) * EACH) };
            let data = memory(EACH);
            // The range views read, not all of the copy.
            small.push((Shadow::unbacked((start + at) as u64, data, EACH as u64, false), start + at + 4000, EACH - 9000));
        }
        let heap = (Shadow::unbacked(big as u64, memory(BIG), BIG as u64, true), big, BIG);
        // The same memory, as another copy: what is landed is landed through
        // it, and the refreshes are of the first.
        let lands = Shadow::unbacked(big as u64, memory(BIG), BIG as u64, true);
        let (mut best, mut asking) = ([f64::MAX; 2], [0f64; 2]);
        let (mut sum, mut landing) = (0.0, 0.0);
        const FRAMES: usize = 12;
        for frame in 0..FRAMES {
            let mut took = [0f64; 2];
            for i in 0..30 {
                // The title's writes, and the relay's beside the submission.
                for _ in 0..9 {
                    let at = views + random.below(1400 * EACH / 8) * 8;
                    if there(at) {
                        // SAFETY: the test's own memory.
                        unsafe { (at as *mut u64).write_volatile(frame as u64) };
                    }
                }
                for _ in 0..60 {
                    let at = watched + random.below(3000 * EACH / 8) * 8;
                    // SAFETY: the test's own memory.
                    unsafe { (at as *mut u64).write_volatile(frame as u64) };
                }
                let first = random.below(BIG / PAGE - 60);
                let lands = lands.clone();
                let land = std::thread::spawn(move || {
                    let now = vec![frame as u8 ^ i as u8 ^ 0x55; PAGE];
                    let started = std::time::Instant::now();
                    for pieces in (first..first + 55).collect::<Vec<_>>().chunks(4) {
                        let pieces: Vec<_> =
                            pieces.iter().map(|p| (p * PAGE + 8, (p + 1) * PAGE - 8, now.as_ptr() as usize)).collect();
                        let mut notes = crate::land::Notes::default();
                        // SAFETY: the bytes of `now`; nothing else lands here.
                        unsafe { lands.land_card(pieces.iter().copied(), 0, &mut notes) };
                        notes.flush();
                    }
                    started.elapsed().as_secs_f64() * 1e3
                });
                let ranges: Vec<(Arc<Shadow>, usize, usize)> = if i == 0 {
                    small.iter().cloned().chain([heap.clone()]).collect()
                } else {
                    (0..60).map(|_| small[random.below(4400)].clone()).chain([heap.clone()]).collect()
                };
                let started = std::time::Instant::now();
                submission(&ranges, 3);
                took[usize::from(i != 0)] += started.elapsed().as_secs_f64() * 1e3;
                let put = land.join().unwrap();
                if frame >= 2 {
                    landing += put;
                }
                heap.0.take_dirty();
            }
            if frame >= 2 {
                sum += took[0] + took[1];
                for k in 0..2 {
                    best[k] = best[k].min(took[k]);
                    asking[k] += took[k];
                }
            }
        }
        println!(
            "VICTORY_LIGHT_REFRESH {}: a frame's refreshes {:.2} ms in the mean; the submission of all copies {:.2} (best {:.2}), the 29 \
             others {:.2} (best {:.2}); the landing thread's puts {:.2}",
            crate::watch::light(),
            sum / (FRAMES - 2) as f64,
            asking[0] / (FRAMES - 2) as f64,
            best[0],
            asking[1] / (FRAMES - 2) as f64,
            best[1],
            landing / (FRAMES - 2) as f64,
        );
        #[cfg(feature = "diagnostics")]
        println!("per frame, as `crate::watch::asks`: {:?}", crate::watch::asks().map(|n| n / FRAMES as u64));
    }
}

#[cfg(test)]
mod batch {   // Batch::texture_due lives here; `mod tests` is already taken at the top of this file
    use super::{Batch, RamBand, TitleWrite, ram_band};
    use std::sync::Arc;

    #[test]
    fn ram_band_tracks_resident_and_commit_pressure() {
        const M: u64 = 1 << 20;
        // The measured local Series S baseline stayed calm (32 GB machine,
        // roughly 11 GB commit limit).
        assert_eq!(
            ram_band(RamBand::Calm, 3857 * M, 7500 * M, 15900 * M, 3962 * M, 32 * 1024 * M, 11 * 1024 * M),
            RamBand::Calm
        );
        // Private commit alone crosses the watch watermark.
        assert_eq!(
            ram_band(RamBand::Calm, 3857 * M, 8200 * M, 15900 * M, 3962 * M, 32 * 1024 * M, 11 * 1024 * M),
            RamBand::Watch
        );
        // The measured Series X commit failure was High.
        assert_eq!(
            ram_band(RamBand::Calm, 4452 * M, 10191 * M, 16047 * M, 154 * M, 32 * 1024 * M, 11 * 1024 * M),
            RamBand::High
        );
    }

    #[test]
    fn ram_band_exits_only_below_lower_watermarks() {
        const M: u64 = 1 << 20;
        assert_eq!(
            ram_band(RamBand::High, 4452 * M, 8900 * M, 16047 * M, 1200 * M, 32 * 1024 * M, 11 * 1024 * M),
            RamBand::Watch,
            "just under the entry mark does not drop straight to calm"
        );
        assert_eq!(
            ram_band(RamBand::Watch, 3857 * M, 6600 * M, 15900 * M, 3962 * M, 32 * 1024 * M, 11 * 1024 * M),
            RamBand::Calm,
            "all lower watermarks met"
        );
    }

    #[test]
    fn ram_band_does_not_toggle_below_the_entry_watermark() {
        const M: u64 = 1 << 20;
        let mut band = RamBand::Calm;
        for private in [7500 * M, 8200 * M, 8200 * M, 8200 * M] {
            band = ram_band(band, 3857 * M, private, 15900 * M, 3962 * M, 32 * 1024 * M, 11 * 1024 * M);
        }
        assert_eq!(band, RamBand::Watch);
        band = ram_band(band, 3857 * M, 7800 * M, 15900 * M, 3962 * M, 32 * 1024 * M, 11 * 1024 * M);
        assert_eq!(band, RamBand::Watch);
    }

    #[test]
    fn a_texture_is_checked_once_per_submission() {
        let mut batch = Batch::default();
        let (a, b) = (Arc::new([0u8; 16]), Arc::new([0u8; 16]));
        // The first list sampling each texture checks it ...
        assert!(batch.texture_due(&a));
        assert!(batch.texture_due(&b));
        // ... later lists of the same submission do not.
        assert!(!batch.texture_due(&a));
        assert!(!batch.texture_due(&a.clone()));
        assert!(!batch.texture_due(&b));
    }

    #[test]
    fn title_writes_make_textures_due_again() {
        let mut batch = Batch::default();
        let a = Arc::new(1u32);
        assert!(batch.texture_due(&a));
        assert!(!batch.texture_due(&a));
        // A copy into title memory or work that may write it may have
        // changed the texture's memory.
        batch.title_written(TitleWrite::Copy);
        assert!(batch.texture_due(&a));
        assert!(!batch.texture_due(&a));
        // A fresh submission starts over.
        let mut next = Batch::default();
        assert!(next.texture_due(&a));
    }

    /// The hashing a submission does for textures in physical-page memory,
    /// with every list checking the textures it samples (before) and with
    /// each checked once (after). Run with
    /// `cargo test --release -p victory-gpu -- --ignored --nocapture texture_checks`.
    #[test]
    #[ignore = "benchmark"]
    fn texture_checks_per_submission() {
        const TEXTURES: usize = 48;
        const SIZE: usize = 4 << 20;
        const LISTS: usize = 12;
        let textures: Vec<Arc<Vec<u8>>> = (0..TEXTURES)
            .map(|t| Arc::new((0..SIZE).map(|i| (i * 31 + t) as u8).collect()))
            .collect();
        let submit = |dedupe: bool| {
            let mut batch = Batch::default();
            let mut hashed = 0usize;
            let started = std::time::Instant::now();
            for _ in 0..LISTS {
                for t in &textures {
                    if !dedupe || batch.texture_due(t) {
                        std::hint::black_box(blake3::hash(t));
                        hashed += 1;
                    }
                }
            }
            (started.elapsed(), hashed)
        };
        submit(false); // warm up
        let (before, n_before) = submit(false);
        let (after, n_after) = submit(true);
        assert_eq!((n_before, n_after), (TEXTURES * LISTS, TEXTURES));
        println!(
            "{LISTS} lists sampling {TEXTURES} textures of {} MiB: \
             before {n_before} hashes in {before:?}, after {n_after} in {after:?} ({:.1}x)",
            SIZE >> 20,
            before.as_secs_f64() / after.as_secs_f64()
        );
    }

    #[test]
    fn vram_idle_ms_scales_with_pressure() {
        let vram_idle_ms = |memory| super::vram_idle_for(memory, false);
        let keeps = |memory| super::vram_idle_for(memory, true);
        let gib = 1024 * 1024 * 1024;
        assert_eq!(keeps(Some([100, 67, 0, 0])), 600_000);
        assert_eq!(keeps(Some([100, 68, 0, 0])), 30_000);
        assert_eq!(keeps(Some([100, 75, 0, 0])), 2_500);
        assert_eq!(keeps(Some([100, 85, 0, 0])), 1_000);
        assert_eq!(keeps(None), 5_000);
        // Normal (< 75%)
        assert_eq!(vram_idle_ms(Some([10 * gib, 0, 0, 0])), 5_000);
        assert_eq!(vram_idle_ms(Some([10 * gib, 7 * gib, 0, 0])), 5_000);
        assert_eq!(vram_idle_ms(Some([100, 74, 0, 0])), 5_000);
        // Moderate (75%..85%)
        assert_eq!(vram_idle_ms(Some([100, 75, 0, 0])), 2_500);
        assert_eq!(vram_idle_ms(Some([100, 84, 0, 0])), 2_500);
        // High (>= 85%, including over-budget)
        assert_eq!(vram_idle_ms(Some([100, 85, 0, 0])), 1_000);
        assert_eq!(vram_idle_ms(Some([10 * gib, 12 * gib, 0, 0])), 1_000);
        // Missing or zero budget falls back to normal
        assert_eq!(vram_idle_ms(None), 5_000);
        assert_eq!(vram_idle_ms(Some([0, 100, 0, 0])), 5_000);
    }

    #[test]
    fn textures_are_kept_while_the_card_has_room() {
        use super::{vram_keep, vram_keep_for};
        let gib = 1 << 30;
        assert_eq!(vram_keep(100 * gib, 87 * gib).0, 600_000);
        // Past it: the longest idle ones, enough to come down to 85 %.
        let (ms, bytes) = vram_keep(100 * gib, 90 * gib);
        assert_eq!((ms, bytes), (5_000, 5 * gib));
        assert_eq!(vram_keep(100 * gib, 120 * gib), (1_500, 35 * gib));
        assert_eq!(vram_keep(0, 7).0, 5_000);
        // Lower room_percent (e.g. 80 %) must evict before the upload cutoff
        // instead of stalling in a dead zone between 80 % and 88 %.
        assert_eq!(vram_keep_for(100 * gib, 77 * gib, 80).0, 600_000);
        assert_eq!(vram_keep_for(100 * gib, 78 * gib, 80), (5_000, 3 * gib));
        assert_eq!(vram_keep_for(100 * gib, 80 * gib, 80), (5_000, 5 * gib));
        assert_eq!(vram_keep_for(100 * gib, 85 * gib, 80), (1_500, 10 * gib));
        // Higher room_percent (e.g. 94 %) scales cleanly too.
        assert_eq!(vram_keep_for(100 * gib, 91 * gib, 94).0, 600_000);
        assert_eq!(vram_keep_for(100 * gib, 92 * gib, 94), (5_000, 3 * gib));
    }

    #[test]
    fn ram_idle_ms_scales_with_pressure() {
        let ram_idle_ms = |avail| super::ram_idle_for(avail, false);
        assert_eq!(super::ram_idle_for(Some(12 << 30), true), 300_000);
        assert_eq!(super::ram_idle_for(Some((12 << 30) - 1), true), 10_000);
        assert_eq!(super::ram_idle_for(None, true), 10_000);
        let gib = 1024 * 1024 * 1024;
        // Normal (>= 4 GiB)
        assert_eq!(ram_idle_ms(Some(8 * gib)), 10_000);
        assert_eq!(ram_idle_ms(Some(4 * gib)), 10_000);
        // Moderate (2 GiB..4 GiB)
        assert_eq!(ram_idle_ms(Some(4 * gib - 1)), 5_000);
        assert_eq!(ram_idle_ms(Some(2 * gib)), 5_000);
        // High (< 2 GiB)
        assert_eq!(ram_idle_ms(Some(2 * gib - 1)), 2_000);
        assert_eq!(ram_idle_ms(Some(0)), 2_000);
        // Missing falls back to normal
        assert_eq!(ram_idle_ms(None), 10_000);
    }
}
