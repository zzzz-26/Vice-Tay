//! Exercise DLSS Quality behind the title's FidelityFX descriptions and
//! console objects, then read back the upscaled colour. Requires an NVIDIA
//! RTX GPU and `VICTORY_DLSS_TEST=1`; `VICTORY_DLSS_DIR` names the folder
//! of the `nvngx_dlss.dll` to use (else the driver's).
#![cfg(windows)]

use std::ffi::c_void;
use std::ptr::{null, null_mut};

use victory_gpu::object::{Iid, Object};
use victory_loader::{ImportResolver, Resolution};
use victory_pe::ImportSymbol;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::{GUID, Interface};

type This = *const Object;

/// A method of one of our console objects, by table offset.
fn slot<F: Copy>(object: This, offset: usize) -> F {
    // SAFETY: our objects start with a pointer to their table, which has
    // the slot.
    unsafe {
        let table = *object.cast::<*const *const ()>();
        std::mem::transmute_copy(&*table.byte_add(offset))
    }
}

fn iid(guid: &GUID) -> Iid {
    // SAFETY: the same 16 bytes.
    unsafe { std::mem::transmute_copy(guid) }
}

/// The console's resource description: the host's.
fn texture(
    width: u64,
    height: u32,
    format: DXGI_FORMAT,
    flags: D3D12_RESOURCE_FLAGS,
) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: width,
        Height: height,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: format,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Flags: flags,
        ..Default::default()
    }
}

/// The title's descriptions (its layouts, from its code).
#[repr(C)]
struct Header {
    kind: u64,
    next: *mut Header,
}

#[repr(C)]
struct CreateUpscale {
    header: Header,
    flags: u32,
    max_render: [u32; 2],
    max_output: [u32; 2],
    message: usize,
}

#[repr(C)]
struct Device {
    header: Header,
    device: This,
}

#[repr(C)]
struct Version {
    header: Header,
    version: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Resource {
    resource: This,
    description: [u32; 8],
    state: u32,
}

#[repr(C)]
struct DispatchUpscale {
    header: Header,
    list: This,
    resources: [Resource; 7],
    jitter: [f32; 2],
    motion_scale: [f32; 2],
    render: [u32; 2],
    output: [u32; 2],
    sharpen: u32,
    sharpness: f32,
    frame_ms: f32,
    pre_exposure: f32,
    reset: u32,
    near: f32,
    far: f32,
    fov: f32,
    meters: f32,
    flags: u32,
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

fn header(kind: u64) -> Header {
    Header {
        kind,
        next: null_mut(),
    }
}

fn half(bits: u16) -> f32 {
    let (sign, exp, frac) = (
        (bits >> 15) as u32,
        ((bits >> 10) & 0x1f) as i32,
        (bits & 0x3ff) as f32,
    );
    let value = match exp {
        0 => frac * 2f32.powi(-24),
        31 => f32::NAN,
        e => (1.0 + frac / 1024.0) * 2f32.powi(e - 15),
    };
    if sign == 1 { -value } else { value }
}

#[test]
fn dlss_runs_the_titles_upscaler() {
    if std::env::var_os("VICTORY_DLSS_TEST").is_none() {
        eprintln!("VICTORY_DLSS_TEST not set; skipped");
        return;
    }
    let dir = std::env::var_os("VICTORY_DLSS_DIR").map(std::path::PathBuf::from);
    // The debug layer, before anything makes the host device: it checks
    // the states FidelityFX is told against the ones the list tracks.
    // SAFETY: plain calls; the layer may be missing.
    unsafe {
        let mut debug: Option<ID3D12Debug> = None;
        if D3D12GetDebugInterface(&mut debug).is_ok() {
            debug.iter().for_each(|d| d.EnableDebugLayer());
        }
    }
    let logs = tempfile::tempdir().unwrap();
    if let Err(error) = victory_hle::modules::fidelityfx::use_dlss(dir.as_deref(), logs.path(), 2) {
        panic!("DLSS did not start: {error}");
    }
    let fallback = std::env::var_os("VICTORY_FIDELITYFX_DIR").map(|dir| {
        victory_hle::modules::fidelityfx::use_host_dir(std::path::Path::new(&dir)).unwrap()
    });
    let host = victory_gpu::external::host_device().unwrap();

    // The title's import binding.
    let resolver = victory_hle::HleResolver::new(victory_hle::HostPolicy::none());
    let export = |name: &str| {
        let symbol = ImportSymbol::Name {
            hint: 0,
            name: name.into(),
        };
        match resolver.resolve("amd_fidelityfx_loader_xbox.dll", &symbol) {
            Resolution::Found { address, .. } => address,
            Resolution::Missing => panic!("{name} is not served (no upscaler for this GPU?)"),
        }
    };
    type Create = extern "C" fn(*mut *mut c_void, *mut Header, *const c_void) -> u32;
    type Destroy = extern "C" fn(*mut *mut c_void, *const c_void) -> u32;
    type Call = extern "C" fn(*mut *mut c_void, *mut Header) -> u32;
    // SAFETY: our exports, with the API's signatures.
    let (create, destroy, query, dispatch) = unsafe {
        (
            std::mem::transmute::<usize, Create>(export("ffxCreateContext")),
            std::mem::transmute::<usize, Destroy>(export("ffxDestroyContext")),
            std::mem::transmute::<usize, Call>(export("ffxQuery")),
            std::mem::transmute::<usize, Call>(export("ffxDispatch")),
        )
    };

    // The console device and objects, through its table.
    let parameters = [victory_gpu::device::CHECKED_VERSION, 0, 0, 0];
    let mut device = null();
    assert_eq!(
        victory_gpu::device::create(
            null(),
            parameters.as_ptr(),
            &victory_gpu::device::IID,
            &mut device
        ),
        0
    );
    type MakeQueue = extern "system" fn(This, *const u32, *const Iid, *mut This) -> i32;
    type MakeAllocator = extern "system" fn(This, u32, *const Iid, *mut This) -> i32;
    type MakeList = extern "system" fn(This, u32, u32, This, This, *const Iid, *mut This) -> i32;
    type MakeTexture = extern "system" fn(
        This,
        *const D3D12_HEAP_PROPERTIES,
        u32,
        *const D3D12_RESOURCE_DESC,
        u32,
        *const c_void,
        *const Iid,
        *mut This,
    ) -> i32;
    let (mut queue, mut allocator, mut list) = (null(), null(), null());
    let queue_desc = [0u32; 4];
    assert_eq!(
        slot::<MakeQueue>(device, 0x40)(
            device,
            queue_desc.as_ptr(),
            &iid(&ID3D12CommandQueue::IID),
            &mut queue
        ),
        0
    );
    assert_eq!(
        slot::<MakeAllocator>(device, 0x48)(
            device,
            0,
            &iid(&ID3D12CommandAllocator::IID),
            &mut allocator
        ),
        0
    );
    assert_eq!(
        slot::<MakeList>(device, 0x60)(
            device,
            0,
            0,
            allocator,
            null(),
            &iid(&ID3D12GraphicsCommandList::IID),
            &mut list
        ),
        0
    );
    let make = |desc: D3D12_RESOURCE_DESC| {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let mut out = null();
        const READ: u32 = 0xc0;
        assert_eq!(
            slot::<MakeTexture>(device, 0xd8)(
                device,
                &heap,
                0,
                &desc,
                READ,
                null(),
                &iid(&ID3D12Resource::IID),
                &mut out
            ),
            0
        );
        out
    };
    // Quality at a 4K output resolution.
    let render = [2560u32, 1440];
    let output = [3840u32, 2160];
    let at = |size: [u32; 2], format, flags| make(texture(size[0].into(), size[1], format, flags));
    let target = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    let color = at(render, DXGI_FORMAT_R16G16B16A16_FLOAT, target);
    let depth = at(
        render,
        DXGI_FORMAT_D32_FLOAT,
        D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
    );
    let motion = at(render, DXGI_FORMAT_R16G16_FLOAT, target);
    let upscaled = at(
        output,
        DXGI_FORMAT_R16G16B16A16_FLOAT,
        target | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
    );

    // A flat colour in, another colour where the output should be written.
    // SAFETY: plain host objects.
    let targets: ID3D12DescriptorHeap = unsafe {
        host.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
            Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
            NumDescriptors: 2,
            ..Default::default()
        })
    }
    .unwrap();
    // SAFETY: as above.
    let (first, step) = unsafe {
        (
            targets.GetCPUDescriptorHandleForHeapStart(),
            host.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV) as usize,
        )
    };
    const RENDER_TARGET: u32 = 0x4;
    const COPY_SOURCE: u32 = 0x800;
    let flat = [0.25f32, 0.5, 0.75, 1.0];
    for (i, (resource, value)) in [(color, flat), (upscaled, [0.0, 1.0, 0.0, 1.0])]
        .into_iter()
        .enumerate()
    {
        let at = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: first.ptr + i * step,
        };
        victory_gpu::external::record(list, |l| {
            let (texture, state) = l.texture(resource, RENDER_TARGET).unwrap();
            assert_eq!(state, RENDER_TARGET);
            // SAFETY: our heap, a live texture, and an open list.
            unsafe {
                host.CreateRenderTargetView(&texture, None, at);
                l.commands().ClearRenderTargetView(at, &value, None);
            }
        })
        .unwrap();
    }

    // The title's context: HDR, inverted depth, auto exposure; its device;
    // the API version it was built against (4.1.1).
    let mut version = Version {
        header: header(0x1_000b),
        version: 0x0100_1001,
    };
    let mut console = Device {
        header: Header {
            kind: 0x100_0002,
            next: &raw mut version.header,
        },
        device,
    };
    let mut upscale = CreateUpscale {
        header: Header {
            kind: 0x1_0000,
            next: &raw mut console.header,
        },
        flags: 0x29,
        max_render: render,
        max_output: output,
        message: 0,
    };
    let mut context = null_mut();
    assert_eq!(create(&mut context, &mut upscale.header, null()), 0);
    assert!(!context.is_null());

    #[repr(C)]
    struct RatioQuery {
        header: Header,
        quality: u32,
        ratio: *mut f32,
    }
    let mut ratio = 0.0;
    let mut ratio_query = RatioQuery {
        header: header(0x1_0002),
        quality: 0,
        ratio: &mut ratio,
    };
    assert_eq!(query(&mut context, &mut ratio_query.header), 0);
    assert_eq!(ratio, 1.0);
    ratio_query.quality = 1;
    assert_eq!(query(&mut context, &mut ratio_query.header), 0);
    assert_eq!(ratio, 1.5);

    #[repr(C)]
    struct RenderResolution {
        header: Header,
        display: [u32; 2],
        quality: u32,
        width: *mut u32,
        height: *mut u32,
    }
    let (mut render_width, mut render_height) = (0, 0);
    let mut resolution = RenderResolution {
        header: header(0x1_0003),
        display: output,
        quality: 1,
        width: &mut render_width,
        height: &mut render_height,
    };
    for (mode, expected) in [(1, [2560, 1440]), (2, [2258, 1270]), (3, [1920, 1080])] {
        resolution.quality = mode;
        assert_eq!(query(&mut context, &mut resolution.header), 0);
        assert_eq!([render_width, render_height], expected);
    }

    // Its queries: phase count, the first jitter offset, memory.
    let mut phases = 0;
    let mut count = PhaseCount {
        header: header(0x1_0004),
        render_width: render[0],
        output_width: output[0],
        count: &mut phases,
    };
    assert_eq!(query(&mut context, &mut count.header), 0);
    assert!(phases > 0, "{phases} phases");
    let (mut x, mut y) = (f32::NAN, f32::NAN);
    let mut offset = JitterOffset {
        header: header(0x1_0005),
        index: 0,
        phases,
        x: &mut x,
        y: &mut y,
    };
    assert_eq!(query(&mut context, &mut offset.header), 0);
    assert!(x.abs() <= 0.5 && y.abs() <= 0.5, "jitter {x} {y}");
    let mut usage = [0u64; 2];
    let mut memory = MemoryUsage {
        header: header(0x1_0008),
        usage: &mut usage,
    };
    assert_eq!(query(&mut context, &mut memory.header), 0);

    // Its dispatch: its resource descriptions (type, format, size, depth,
    // mips, flags, usage) in the states it gives (read 0xc, written 0x2).
    let described = |resource, format, size: [u32; 2], usage, state| Resource {
        resource,
        description: [2, format, size[0], size[1], 1, 1, 0, usage],
        state,
    };
    let none = Resource {
        resource: null(),
        description: [0; 8],
        state: 0,
    };
    let mut upscaling = DispatchUpscale {
        header: header(0x1_0001),
        list,
        resources: [
            described(color, 4, render, 0, 0xc),
            described(depth, 28, render, 4, 0xc),
            described(motion, 18, render, 0, 0xc),
            none,
            none,
            none,
            described(upscaled, 4, output, 2, 0x2),
        ],
        jitter: [-x, -y],
        motion_scale: [1.0, 1.0],
        render,
        output,
        sharpen: 0,
        sharpness: 0.0,
        frame_ms: 33.3,
        pre_exposure: 1.0,
        reset: 1,
        // Swapped for inverted depth, as the title passes them.
        near: 1000.0,
        far: 0.1,
        fov: 1.0,
        meters: 0.0,
        flags: 0,
    };
    #[repr(C)]
    struct Provider {
        header: Header,
        id: u64,
        name: *const std::ffi::c_char,
    }
    let mut provider = Provider {
        header: header(6),
        id: 0,
        name: null(),
    };
    assert_eq!(query(&mut context, &mut provider.header), 0);
    // SAFETY: a name the DLL keeps.
    eprintln!("provider {:#x} {:?}", provider.id, unsafe {
        std::ffi::CStr::from_ptr(provider.name)
    });
    assert!(
        unsafe { std::ffi::CStr::from_ptr(provider.name) }
            .to_bytes()
            .starts_with(b"NVIDIA DLSS")
    );
    // Frames: the first resets the history.
    for reset in [1, 0, 0, 0] {
        upscaling.reset = reset;
        assert_eq!(dispatch(&mut context, &mut upscaling.header), 0);
    }

    // Read the output back in the same list, run it as the title would.
    let (pitch, rows) = (
        u64::from(output[0]).next_multiple_of(32) * 8,
        u64::from(output[1]),
    );
    // SAFETY: plain creation of a read-back buffer.
    let readback: ID3D12Resource = unsafe {
        let mut buffer = None;
        host.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_READBACK,
                ..Default::default()
            },
            D3D12_HEAP_FLAG_NONE,
            &D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: pitch * rows,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            },
            D3D12_RESOURCE_STATE_COPY_DEST,
            None,
            &mut buffer,
        )
        .unwrap();
        buffer.unwrap()
    };
    victory_gpu::external::record(list, |l| {
        let (texture, state) = l.texture(upscaled, COPY_SOURCE).unwrap();
        assert_eq!(state, COPY_SOURCE);
        let to = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: borrowed; the location never releases it.
            pResource: unsafe { std::mem::transmute_copy(&readback) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: 0,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: DXGI_FORMAT_R16G16B16A16_FLOAT,
                        Width: output[0],
                        Height: output[1],
                        Depth: 1,
                        RowPitch: pitch as u32,
                    },
                },
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(&texture) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        // SAFETY: an open list and live resources.
        unsafe { l.commands().CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    })
    .unwrap();
    slot::<extern "system" fn(This) -> i32>(list, 0x48)(list);
    slot::<extern "system" fn(This, u32, *const This)>(queue, 0x50)(queue, 1, &list);

    // This build normally submits ahead. The fixture requested the older
    // queue interface and can wait for its real guest-visible fence here.
    let mut fence = null();
    let make_fence =
        slot::<extern "system" fn(This, u64, u32, *const Iid, *mut This) -> i32>(device, 0x120);
    assert_eq!(
        make_fence(device, 0, 0, &iid(&ID3D12Fence::IID), &mut fence),
        0
    );
    assert_eq!(
        slot::<extern "system" fn(This, This, u64) -> i32>(queue, 0x70)(queue, fence, 1),
        0
    );
    let completed = slot::<extern "system" fn(This) -> u64>(fence, 0x40);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    while completed(fence) < 1 {
        assert!(
            std::time::Instant::now() < deadline,
            "timed out waiting for GPU readback"
        );
        std::thread::sleep(std::time::Duration::from_millis(1));
    }

    // SAFETY: the list ran (submissions are waited for).
    let pixels = unsafe {
        let mut mapped = null_mut();
        readback.Map(0, None, Some(&mut mapped)).unwrap();
        std::slice::from_raw_parts(mapped.cast::<u16>(), (pitch * rows / 2) as usize).to_vec()
    };
    // No state the debug layer found wrong (if it is installed).
    if let Ok(messages) = host.cast::<ID3D12InfoQueue>() {
        // SAFETY: plain queries of the stored messages.
        let errors: Vec<String> = unsafe {
            (0..messages.GetNumStoredMessages())
                .filter_map(|i| {
                    let mut size = 0;
                    messages.GetMessage(i, None, &mut size).ok()?;
                    let mut bytes = vec![0u64; size.div_ceil(8)];
                    let message = bytes.as_mut_ptr().cast::<D3D12_MESSAGE>();
                    messages.GetMessage(i, Some(message), &mut size).ok()?;
                    let m = &*message;
                    let text = std::ffi::CStr::from_ptr(m.pDescription.cast()).to_string_lossy();
                    eprintln!("debug layer {}: {text}", m.Severity.0);
                    (m.Severity.0 <= D3D12_MESSAGE_SEVERITY_ERROR.0).then(|| text.into_owned())
                })
                .collect()
        };
        assert!(errors.is_empty(), "debug layer: {errors:#?}");
    }
    // SAFETY: plain query.
    assert!(unsafe { host.GetDeviceRemovedReason() }.is_ok());
    eprintln!(
        "first pixels: {:?}",
        pixels[..16].iter().map(|&h| half(h)).collect::<Vec<_>>()
    );
    for (i, pixel) in pixels.chunks(4).enumerate() {
        for (c, (&channel, &want)) in pixel[..3].iter().zip(&flat).enumerate() {
            let got = half(channel);
            assert!(
                (got - want).abs() < 0.05,
                "pixel {i} channel {c}: {got}, not {want}"
            );
        }
    }
    if let Some(version) = fallback {
        // Refused dispatches hand off this context to its FSR standby.
        upscaling.resources[6].resource = null();
        let mut active = String::new();
        for _ in 0..16 {
            assert_eq!(dispatch(&mut context, &mut upscaling.header), 6);
            assert_eq!(query(&mut context, &mut provider.header), 0);
            // SAFETY: the provider owns the returned name.
            active = unsafe { std::ffi::CStr::from_ptr(provider.name) }
                .to_string_lossy()
                .into_owned();
            if active.trim_start_matches("FSR ").starts_with(version) {
                break;
            }
        }
        assert!(active.trim_start_matches("FSR ").starts_with(version), "{active}");
        eprintln!("standby provider after DLSS refused dispatches: {active}");
    }
    assert_eq!(destroy(&mut context, null()), 0);
    // Exercise NGX teardown and verify repeated shutdown idempotence.
    victory_hle::modules::fidelityfx::shut_down();
    victory_hle::modules::fidelityfx::shut_down();
}
