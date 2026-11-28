//! DXIL emission (decision 0003, step 3): a console vertex, pixel, compute
//! or mesh shader as a PC DXIL container (SM 6.6) that host D3D12 accepts.
//! Compute shaders keep the console's thread-group size; one that depends
//! on the lane layout (cross-lane operations) declares the console's wave
//! size (`WaveSize`), so the host device must support it.
//!
//! What the host binds for a translated shader ([`HostShader`]):
//!
//! * **User data** at `b0, space 0` (32-bit root constants or a constant
//!   buffer, per stage): dword `index` holds a root parameter's value, as
//!   [`UserData`] lists. A root-constant dword is the constant. A
//!   descriptor table is the byte offset of its first descriptor in its
//!   heap: the descriptor index times 32 for CBV/SRV/UAV tables, times 16
//!   for sampler tables (the console's descriptor sizes; the code adds
//!   offsets in those units). The heap bases of dynamic resources are 0
//!   and not listed.
//! * **Descriptors** through SM 6.6 `ResourceDescriptorHeap[]` and
//!   `SamplerDescriptorHeap[]`, at the index the console address gives, so
//!   the host heaps keep the title's descriptor order. The root signature
//!   needs `CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED` and
//!   `SAMPLER_HEAP_DIRECTLY_INDEXED`.
//! * **Immediate samplers** as static samplers `s<n>, space 0`, `n` the
//!   position in [`HostShader::samplers`] (the console sampler words) and
//!   [`HostShader::static_samplers`] (the same decoded into a D3D12 static
//!   sampler description). Comparison sampling (added 2026-11-23): an
//!   immediate sampler it uses has a comparison filter there (bit 0x80);
//!   a heap sampler it uses must be a comparison sampler on the host.
//! * **Root CBVs and root SRVs** (raw buffers) as root descriptors
//!   `b<i>` / `t<i>` in [`ROOT_CBV_SPACE`], `i` the title's root parameter
//!   index ([`HostShader::root_cbvs`], [`HostShader::root_srvs`]); the
//!   address the title sets is the buffer's.
//! * **Vertex fetch** (added 2026-11-23). The title's mesh and
//!   vertex programs fetch vertices themselves with typed-format loads
//!   (`tbuffer_load_format_*`): the format (8_8_8_8 unorm, 16_16 snorm,
//!   2_10_10_10 unorm, 8_8_8_8 and 16_16_16_16 uint...) is in the
//!   instruction, so the translation reads raw bytes and converts them
//!   itself. Their descriptors are built around a root SRV (stride 1, no
//!   record limit): the host binds the root SRV `t<slot>` above with the
//!   address the title sets, nothing more. The vertex index, the stride
//!   and the attribute offset are the shader's own arithmetic on data in
//!   that buffer (per-stream headers), not descriptor fields.
//! * **Buffer views in the heap** are declared by how the shader reads
//!   them: raw (ByteAddressBuffer) for dword loads and stores, atomics and
//!   typed-format loads without an index; structured with the used-resource
//!   entry's stride when indexed; typed with four 32-bit float channels for
//!   `buffer_load_format_*` and `buffer_store_format_*`. The host's view at
//!   that index must be of that kind and start at the console descriptor's
//!   base address (raw: `R32_TYPELESS` with the raw flag), whatever kind the
//!   title's view description says. Typed-format loads from heap
//!   descriptors (1,024 world compute programs, ray-tracing material
//!   resolves) read raw views this way.
//!
//! * **Texture atomics** use a scalar integer UAV. The host binds R32_UINT
//!   for 32-bit operations and R32G32_UINT for 64-bit operations. The latter
//!   require SM 6.6 atomic support on typed and descriptor-heap resources.
//!
//! Entry points: [`translate`] (container bytes as the title passes them
//! at pipeline creation → [`HostShader`]), [`emit`] (the same with the
//! earlier stages already run). The root constants of each stage must
//! cover [`HostShader::user_data_dwords`]. Stability: these types and the
//! binding scheme are what pipeline creation builds on; changes are
//! announced.
//! * **Signatures** as the console container has them (same semantics,
//!   registers and masks), so the input layout and stage linking are the
//!   title's; pixel outputs are plain 32-bit values.
//! * **Mesh shaders** ([`HwStage::Ms`], `ms_6_6`): user data and
//!   descriptors as above (the root constants visible to the mesh stage);
//!   the pipeline is a mesh pipeline (pipeline-state stream, no input
//!   layout) with the title's pixel shader. The thread group is the
//!   console's in lanes (`[numthreads]` in the DXIL), the vertex and
//!   primitive maxima its own, triangles only, no wave size required. The
//!   console numbers groups in x only: the draw path passes the title's
//!   count as `DispatchMesh(x, 1, 1)`. Per-primitive outputs (`PSG1`) are
//!   the console's, without `SV_PrimitiveID` and `SV_CullPrimitive` (the
//!   console never exports them).
//! * **Rectangle lists** ([`rect_list_gs`], `gs_6_6`): the console draws a
//!   rectangle from three vertices; the host has no such topology. A
//!   pipeline the title makes for rectangles (topology type 19) gets a
//!   geometry shader of ours between the translated vertex and pixel
//!   shaders and draws triangle lists with the title's own vertex counts.
//!   It binds nothing.

mod bitcode;
mod container;
#[cfg(all(test, windows))]
mod cube_lod_tests;
#[cfg(all(test, windows))]
mod diff_tests;
#[cfg(all(test, windows))]
mod intersect_tests;
#[cfg(all(windows, any(test, feature = "host")))]
pub mod host;
mod loose_lds;
mod lower;
mod own_bits;
#[cfg(all(test, windows))]
mod permlane_tests;
mod rect;
mod sampler;
mod shader;
mod threading;
#[cfg(windows)]
pub mod validator;
#[cfg(all(test, windows))]
mod probe_tests;
#[cfg(all(test, windows))]
mod size_tests;
#[cfg(all(test, windows))]
mod valu_tests;
#[cfg(all(test, windows))]
mod waterfall_tests;
#[cfg(all(test, windows))]
mod wave64_tests;
mod wave;

use crate::dxbc::{Container, FourCC};
use crate::native::{HwStage, NativeProgram, RootType};
use crate::{ShaderError, io, ir, resource, structure};

pub use bitcode::census;
pub use sampler::StaticSampler;
pub use lower::{host_settings, set_host_max_wave, set_host_rdna, set_thread_max_wave};
pub use rect::rect_list_gs;
pub use shader::{
    MEMORY_FAULTS, MEMORY_PAGES, MEMORY_READ_ONLY, ROOT_CBV_SPACE, SAMPLER_SPACE, SLOT_FAULTS, SLOT_MEMORY, SLOT_STRIDES,
    SLOT_STRIDES_BYTES, SLOT_STRIDES_COUNT, SLOT_TABLE_STRIDES, VIEW_ADDRESSES, USER_DATA_SPACE, VIEW_INTEGER, VIEW_SIGNED, VIEW_STRIDE_BITS, USER_DISPATCH_THREADS_X, USER_MEMORY_BASE, USER_ROOT_ADDRESS, USER_VIEW_ADDRESS, USER_VIEW_BIAS,
    constants_raw, raw_views, set_raw_views, set_view_heap_limit, view_heap_limit,
};

/// The loop limit without an override: 2^20.
pub const DEFAULT_LOOP_LIMIT: u32 = lower::LOOP_LIMIT;

/// The iteration count after which a translated loop is left, per lane and
/// per entry into the loop: [`DEFAULT_LOOP_LIMIT`].
///
/// With the `diagnostics` feature, `VICTORY_SHADER_LOOP_LIMIT` (decimal or
/// `0x` hex, at least 1) sets it instead. A loop that hits the limit in
/// every lane of a full-screen pass still runs for seconds (2^20 times a
/// 4K screen is about 10^13 iterations); a much lower limit shows whether
/// such loops are what makes a frame slow. The value is part of the
/// translation, so the shader cache must key on it (`victory-gpu`'s does).
pub fn loop_limit() -> u32 {
    static LIMIT: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| {
        if cfg!(feature = "diagnostics") {
            parse_loop_limit(std::env::var("VICTORY_SHADER_LOOP_LIMIT").ok().as_deref())
        } else {
            lower::LOOP_LIMIT
        }
    })
}

/// Whether a pixel shader's helper lane reads its own value where the console reads the first
/// lane's (`lower.rs` `first_lane`); on unless `VICTORY_HELPER_OWN=0`. Part of the translation:
/// the shader cache keys on it.
pub fn helper_own() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_HELPER_OWN").is_none_or(|v| v != "0"))
}

/// Whether pixel shaders that request early depth testing can use the host's
/// early depth-stencil test. Off unless `VICTORY_EARLY_DEPTH=1`; the shader
/// cache keys on this setting.
pub fn early_depth_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_EARLY_DEPTH").is_some_and(|v| v == "1"))
}

/// Which of a translated program's group barriers wait for the whole
/// group (`GroupMemoryBarrierWithGroupSync`); the others keep their memory
/// fence and lose the wait. A wait that some thread of the group never
/// reaches is undefined on the host, and NVIDIA's wait is for every
/// thread: the group stands until Windows resets the card. (The console's
/// barrier is the wave's, whatever lanes are live, and a wave that ended
/// is not waited for.) `VICTORY_GROUP_SYNC`:
/// - unset or `2`: every barrier waits, as before.
/// - `1`: the title's `s_barrier` waits only in a block every lane runs,
///   in a program no lane can be cut out of at the iteration limit.
/// - `0`: no barrier waits, neither the title's nor the ones the
///   translation adds (wave64 programs in two host waves exchange through
///   them, so such programs compute wrong values). For one run that tells
///   whether a barrier is what stops the card: with it no run can stand in
///   a barrier.
///
/// Part of the translation: the shader cache keys on it (`victory-gpu`'s
/// does).
pub fn group_sync() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_GROUP_SYNC").ok().as_deref().map(str::trim) {
        Some("0") => 0,
        Some("1") => 1,
        _ => 2,
    })
}

/// Whether buffer atomics of the ray-tracing structure builder stand between
/// two fences (`VICTORY_BVH_FENCES=0`: they do not). Part of the
/// translation: the shader cache keys on it.
pub fn builder_fences() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_BVH_FENCES").is_none_or(|v| v != "0"))
}

/// Whether the structure builder's accesses through its root views stay near
/// them, and a size in a structure's header that cannot be one counts as none
/// (`VICTORY_BVH_BOUNDS=0`: neither). Part of the translation: the shader
/// cache keys on it.
pub fn builder_bounds() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_BVH_BOUNDS").is_none_or(|v| v != "0"))
}

/// The clean-up passes that run over a translated program before it is
/// written (`bitcode::opt`): a set of that module's pass bits. Every pass
/// is exact, so the switch is for telling a translator fault from a
/// clean-up fault, and for measuring what the clean-up is worth.
/// `VICTORY_SHADER_OPT`:
/// - unset: every pass but the last one below (63).
/// - `0`: none (the module as the lowering made it).
/// - a number (decimal or `0x` hex): those passes; 1 dead code, 2 constants
///   and identities, 4 one instruction for one value, 8 what a branch taken
///   says about its condition, 16 one read for one place within a block,
///   32 a first-lane read of two host waves without the round that asks
///   which half has a lane (where every lane has one).
///   64 the dwords of one console load read by one call for every four
///   (not on unless asked: `127` is every pass with it).
///
/// Part of the translation: the shader cache keys on it (`victory-gpu`'s
/// does).
pub fn shader_opt() -> u32 {
    #[cfg(test)]
    if let Some(passes) = TEST_SHADER_OPT.get() {
        return passes;
    }
    static PASSES: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *PASSES.get_or_init(|| parse_shader_opt(std::env::var("VICTORY_SHADER_OPT").ok().as_deref()))
}

#[cfg(test)]
thread_local! {
    /// For a test's thread alone: [`shader_opt`].
    pub(super) static TEST_SHADER_OPT: std::cell::Cell<Option<u32>> = const { std::cell::Cell::new(None) };
}

/// On a host whose shaders run on the console's own kind of hardware (AMD
/// RDNA, [`set_host_rdna`]): which of the console's float instructions are
/// written as the one host operation the driver makes the same instruction
/// of, instead of spelled out in operations every host agrees on.
/// `VICTORY_RDNA_FLOAT`: a set of [`rdna`]'s bits (decimal or `0x` hex); all
/// of them when it is unset, `0` none. Nothing on other hosts.
///
/// Measured on an RX 7900 XTX (a frame of the yard, replayed turn by turn with
/// and without): the title's dispatches and draws 18.4 -> 17.6 ms, 0.4 of it
/// without [`rdna::LEGACY`]. Tried and worth nothing there: `v_mad_f32` as one
/// multiply-add, a fast reciprocal, scalar buffer loads of four dwords at once,
/// a wave size for every compute program (32, 64, the console's).
///
/// Part of the translation: the shader cache keys on it ([`host_settings`]).
pub fn rdna_float() -> u32 {
    static BITS: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *BITS.get_or_init(|| {
        let value = std::env::var("VICTORY_RDNA_FLOAT").ok();
        parse_set(value.as_deref(), rdna::ALL, rdna::ALL)
    })
}

/// The bits of [`rdna_float`].
pub mod rdna {
    /// `v_max_f32` and the like: the host's FMax and FMin, without the code
    /// that orders -0 below +0.
    pub const MAX_MIN: u32 = 1;
    /// `v_mul_legacy_f32` and the like (0 times anything is 0): a plain product
    /// where no factor can be an infinity or no number where the other is 0,
    /// and one comparison where it can. A zero may have the other sign.
    pub const LEGACY: u32 = 2;
    /// `v_med3_f32` of a value between two ordered constants: a minimum of a
    /// maximum.
    pub const MED3: u32 = 4;
    /// `v_fract_f32`: the host's Frc as it is.
    pub const FRACT: u32 = 8;
    pub const ALL: u32 = 15;
}

/// A switch's set of bits; `unset` when it is unset or unreadable. Pure, for
/// testing.
fn parse_set(value: Option<&str>, unset: u32, all: u32) -> u32 {
    let parsed = value
        .map(str::trim)
        .and_then(|v| match v.strip_prefix("0x") {
            Some(hex) => u32::from_str_radix(hex, 16).ok(),
            None => v.parse().ok(),
        });
    parsed.map_or(unset, |n| n & all)
}

/// Everything a translation depends on besides the container: the
/// translator's own source (hashed at build time), the value of every
/// `VICTORY_*` switch it reads, the host facts the device hands it (widest
/// wave, view heap limit) and the settings above. The shader cache keys on
/// this instead of on the whole executable, so a rebuild that leaves the
/// translator alone keeps its translations. Call it after the device is made.
pub fn translation_fingerprint() -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(env!("VICTORY_SHADER_SOURCE_HASH").as_bytes());
    out.push(u8::from(cfg!(feature = "diagnostics")));
    out.push(u8::from(cfg!(debug_assertions)));
    for name in env!("VICTORY_SHADER_SWITCHES").split(',').filter(|n| !n.is_empty()) {
        out.extend_from_slice(name.as_bytes());
        out.push(b'=');
        if let Some(value) = std::env::var_os(name) {
            out.extend_from_slice(value.as_encoded_bytes());
        }
        out.push(0);
    }
    out.extend_from_slice(&lower::host_max_wave().to_le_bytes());
    out.extend_from_slice(&lower::host_rdna().to_le_bytes());
    out.extend_from_slice(&shader::view_heap_limit().to_le_bytes());
    out.extend_from_slice(&shader::raw_views().to_le_bytes());
    out.extend_from_slice(&loop_limit().to_le_bytes());
    out.push(group_sync());
    out.extend_from_slice(&shader_opt().to_le_bytes());
    out.push(u8::from(wave_pairs_on()));
    out
}

/// `VICTORY_SHADER_OPT`'s value; every pass when it is unset or
/// unreadable. Pure, for testing.
fn parse_shader_opt(value: Option<&str>) -> u32 {
    let parsed = value
        .map(str::trim)
        .and_then(|v| match v.strip_prefix("0x") {
            Some(hex) => u32::from_str_radix(hex, 16).ok(),
            None => v.parse().ok(),
        });
    parsed.map_or(bitcode::opt::DEFAULT, |n| n & bitcode::opt::ALL)
}

/// `VICTORY_SHADER_LOOP_LIMIT`'s value; the default when it is unset,
/// unreadable or 0. Pure, for testing.
fn parse_loop_limit(value: Option<&str>) -> u32 {
    let parsed = value
        .map(str::trim)
        .and_then(|v| match v.strip_prefix("0x") {
            Some(hex) => u32::from_str_radix(hex, 16).ok(),
            None => v.parse().ok(),
        });
    match parsed {
        Some(n) if n > 0 => n,
        _ => lower::LOOP_LIMIT,
    }
}

/// A translated shader and what the host binds for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostShader {
    pub stage: HwStage,
    /// The DXIL container, hashed as D3D12 expects.
    pub dxil: Vec<u8>,
    /// The user-data dwords the shader reads.
    pub user_data: Vec<UserData>,
    /// Immediate samplers: the four sampler words of `s<n>`.
    pub samplers: Vec<[u32; 4]>,
    /// The same samplers decoded, `s<n>` at position n: the static sampler
    /// descriptions the host's root signature declares.
    pub static_samplers: Vec<StaticSampler>,
    /// Root CBVs the shader reads: root parameter indices, each bound as
    /// `b<index>` in [`ROOT_CBV_SPACE`].
    pub root_cbvs: Vec<u8>,
    /// Root SRVs (raw buffers) the shader reads: `t<index>` in
    /// [`ROOT_CBV_SPACE`].
    pub root_srvs: Vec<u8>,
    /// Root UAVs (raw buffers) the shader reads or writes: `u<index>` in
    /// [`ROOT_CBV_SPACE`].
    pub root_uavs: Vec<u8>,
    /// Heap descriptors that can be selected from root state at bind time.
    pub root_indexed_views: Vec<crate::resource::RootIndexedView>,
}

impl HostShader {
    /// Size of the user-data buffer the shader declares, in dwords (a
    /// multiple of 4; 0 when it reads none). Root constants must cover it.
    pub fn user_data_dwords(&self) -> u32 {
        let n = self
            .user_data
            .iter()
            .map(|u| u.index + 1)
            .max()
            .unwrap_or(0);
        n.next_multiple_of(4)
    }
}

/// One dword of user data and the root parameter that fills it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UserData {
    /// Dword in the user-data buffer (the console's user-data index).
    pub index: u32,
    /// Root parameter index in the title's root signature.
    pub slot: u8,
    /// `Constants` or `Table` (see the module notes for the value).
    pub root: RootType,
    /// Dword within the parameter.
    pub dword: u32,
}

/// Translates the vertex, pixel, compute or mesh program of a console
/// shader container.
///
/// A program refused because the first half of an interpolation is read as
/// data is translated once more with first halves evaluated as the pair
/// they start (`lower::with_first_half_pairs`), and only such a program:
/// with that on for every program, 8,551 pixel programs of the title that
/// already translated came out changed (an extra evaluation in each), for
/// the sake of 9 that did not translate.
///
/// A program refused at a lane mask with lanes elsewhere read as data is
/// translated once more with mask operations on numbers read as the
/// numbers they compute (`lower::with_mask_ops_on_numbers`), and only such
/// a program: with that on for every program, 69 compute programs that
/// already translated came out changed (`CS_RaytracingResolve`, ...).
pub fn translate(container: &[u8]) -> Result<HostShader, ShaderError> {
    match translate_plain(container) {
        Err(ShaderError::Unsupported { what, .. }) if what == lower::FIRST_HALF => {
            lower::with_first_half_pairs(|| translate_plain(container))
        }
        Err(ShaderError::Unsupported { what, .. }) if what == lower::MASK_ELSEWHERE => {
            lower::with_mask_ops_on_numbers(|| translate_plain(container))
        }
        other => other,
    }
}

fn translate_plain(container: &[u8]) -> Result<HostShader, ShaderError> {
    let c = Container::parse(container)?;
    let dxil = c
        .part(FourCC::DXIL)
        .or(c.part(FourCC::ACCC))
        .ok_or(ShaderError::Malformed("no DXIL part"))?;
    let native = crate::native_programs(dxil.data)?
        .into_iter()
        .find(|p| {
            matches!(
                p.stage,
                HwStage::Vs | HwStage::Ps | HwStage::Cs | HwStage::Ms
            )
        })
        .ok_or(ShaderError::Unsupported {
            offset: 0,
            what: "a container without a vertex, pixel, compute or mesh program",
        })?;
    let p = crate::spill::build(&native)?;
    // Structurisation first: tracing per-lane descriptors needs it (its
    // failure is reported after the earlier stages').
    let st = structure::structure(&p);
    let res = resource::track(&p, &native.layout, native.stage, st.as_ref().ok())?;
    let io = io::raise(&p, &native)?;
    let st = st?;
    emit(&c, &native, &p, &st, &io, &res)
}

/// [`emit`] with the same second tries as [`translate`] (first halves of
/// interpolations as pairs, mask operations on numbers), for tools that
/// run the stages themselves.
pub fn emit_with_retries(
    c: &Container,
    native: &NativeProgram,
    p: &ir::Program,
    st: &structure::Structured,
    io: &io::Io,
    res: &resource::Resources,
) -> Result<HostShader, ShaderError> {
    match emit(c, native, p, st, io, res) {
        Err(ShaderError::Unsupported { what, .. }) if what == lower::FIRST_HALF => {
            lower::with_first_half_pairs(|| emit(c, native, p, st, io, res))
        }
        Err(ShaderError::Unsupported { what, .. }) if what == lower::MASK_ELSEWHERE => {
            lower::with_mask_ops_on_numbers(|| emit(c, native, p, st, io, res))
        }
        other => other,
    }
}

thread_local! {
    /// [`with_wave_pairs`]'s choice, for this thread's translations.
    static WAVE_PAIRS_HERE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether a nonempty list of shader log hashes names this container (`VICTORY_WAVE_PAIRS_ONLY`).
fn container_listed(list: Option<&std::ffi::OsStr>, input: &[u8]) -> bool {
    let Some(list) = list.filter(|s| !s.is_empty()) else {
        return true;
    };
    let Some(list) = list.to_str() else {
        return false;
    };
    let hash = blake3::hash(input).to_hex();
    list.split(',').any(|name| name.trim().eq_ignore_ascii_case(&hash[..16]))
}

/// Shaders whose wave-wide form was exercised by the NVIDIA launcher profile.
const DEFAULT_WAVE_WIDE_HASHES: [&str; 4] = [
    "6f225b641c5b0708",
    "5dc3b25986622949",
    "369cbe906ea72764",
    "2cc4c197ee29e291",
];

fn wave_wide_hash_matches(list: &std::ffi::OsStr, hash: &str) -> bool {
    list.to_str().is_some_and(|list| {
        list.split(',').any(|name| name.trim().eq_ignore_ascii_case(hash))
    })
}

/// Per-container override of the global wave-wide setting.
fn wave_wide_override(
    setting: Option<&std::ffi::OsStr>,
    list: Option<&std::ffi::OsStr>,
    input: &[u8],
) -> Option<bool> {
    let list = list.filter(|list| !list.is_empty());
    match setting {
        None => {
            let digest = blake3::hash(input).to_hex();
            let hash = &digest[..16];
            (DEFAULT_WAVE_WIDE_HASHES.contains(&hash)
                || list.is_some_and(|list| wave_wide_hash_matches(list, hash)))
            .then_some(true)
        }
        Some(value) if value == "1" => {
            let list = list?;
            let digest = blake3::hash(input).to_hex();
            (!wave_wide_hash_matches(list, &digest[..16])).then_some(false)
        }
        Some(_) => None,
    }
}

/// Changes the cache key only where the container's wave-wide policy differs
/// from the global setting. The fallback is only used on a 32-lane host.
pub fn wave_wide_cache_override(input: &[u8]) -> Option<bool> {
    if lower::host_max_wave() != 32 {
        return None;
    }
    wave_wide_override(
        std::env::var_os("VICTORY_WAVE_WIDE").as_deref(),
        std::env::var_os("VICTORY_WAVE_WIDE_ONLY").as_deref(),
        input,
    )
}

/// Returns whether the wave-wide path and the driver loop are allowed, respectively.
fn wave_wide_routes(
    setting: Option<&std::ffi::OsStr>,
    list: Option<&std::ffi::OsStr>,
    input: &[u8],
    scene_kernel: bool,
) -> (bool, bool) {
    let picked = wave_wide_override(setting, list, input);
    match setting {
        None => (picked.unwrap_or(scene_kernel), picked.is_none()),
        Some(value) if value == "1" => {
            let enabled = picked.unwrap_or(true);
            (enabled, !enabled)
        }
        Some(_) => (false, false),
    }
}

/// Whether `VICTORY_WAVE_PAIRS_ONLY` (16-hex-digit shader hashes as the log names them, comma
/// separated; unset or empty: none) names this container: it is tried in the paired wave64
/// form first (fxceless/victory-fork #1397; for 5dc3b25986622949 and 6f225b641c5b0708).
fn wave_pairs_listed(input: &[u8]) -> bool {
    match std::env::var_os("VICTORY_WAVE_PAIRS_ONLY") {
        Some(list) if list == "none" || list == "0" => false,
        Some(list) if !list.is_empty() => container_listed(Some(&list), input),
        _ => paired_by_default(input),
    }
}

/// The compute programs that take the paired wave64 form unless `VICTORY_WAVE_PAIRS_ONLY`
/// names others (`none`: no program). On hosts of 32-lane waves these programs' wave-wide
/// sums, first-lane reads and LDS shared within a console wave are exact only in the paired
/// form; the other forms left the picture visibly wrong, each checked in game:
/// - the shadow mask, `CS_ShadowRenderTiled*` (18 programs): whole 16x16 tiles black or lit
///   (stair-stepped shadows on the ground and squares on characters);
/// - the water's transmission volumetrics, `CS_TransmissionVolumetrics_Tiled`,
///   `_Tiled_LocalLights`, `_GenerateTiles` and `_Submerged`: dark blocks on open water and a broken view under
///   water;
/// - the atmospheric diffusion and its depth scan, `CS_GatherDiffusion`, `CS_ScanMip0..3`, with
///   `CS_DeferredFallbacksAlpha_SkyProbe` and `CS_TemporalResolve2_Opaque_FrameStatistics`: a
///   halo around dark silhouettes against the sky.
///
/// A timing run on an RTX 4090 measured about 0.1 ms per frame for the first 28 programs.
const DEFAULT_WAVE_PAIRS_HASHES: [&str; 29] = [
    // shadow mask
    "191ecaed66eea3b1", "1d44a723bdebbf27", "20b78ee12ecdb57d", "240cb516c1af0bc3",
    "31d703207b51549a", "466ec41d3f9d24c0", "56f3bf2d9388e5a7", "65de113e23fa4fdd",
    "804b342cb4b9d1e9", "922e1d0ee1e19a85", "ab5455fc6abb804f", "abd7a22106898c66",
    "b4b0427232707598", "be6ce5d849d51127", "ce79e9eba7ae0348", "d7db57731ada919b",
    "f133e35efed2a48a", "fa0bd10a35bfef3c",
    // water
    "5dc3b25986622949", "6f225b641c5b0708", "926c46bab3b86bd6", "1aa247ed3653b616",
    // diffusion / sky
    "ce2378e5e8711aad", "b7953a05b189dfc1", "71883e40620a567a",
    "3c196d81eca7ff15", "acf14a70f37fb012", "f79824724d8e2daf", "f0aa2b1e79275592",
];

/// Whether `input` is one of [`DEFAULT_WAVE_PAIRS_HASHES`].
fn paired_by_default(input: &[u8]) -> bool {
    let digest = blake3::hash(input).to_hex();
    DEFAULT_WAVE_PAIRS_HASHES.contains(&&digest[..16])
}

thread_local! {
    /// The program being translated is paired by default ([`paired_by_default`]): its FP32
    /// reductions replay with the confirmed guest mode unless `VICTORY_WAVE_PAIRS_FP32` says
    /// otherwise (`lower::paired_fp32`).
    static PAIRED_BY_DEFAULT_HERE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// See [`PAIRED_BY_DEFAULT_HERE`].
pub(crate) fn paired_by_default_here() -> bool {
    PAIRED_BY_DEFAULT_HERE.get()
}

/// Translations inside `f` on this thread try the paired wave64 form first when `on`.
pub fn with_wave_pairs<R>(on: bool, f: impl FnOnce() -> R) -> R {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            WAVE_PAIRS_HERE.set(self.0);
        }
    }
    let _restore = Restore(WAVE_PAIRS_HERE.replace(on));
    f()
}

#[cfg(test)]
#[test]
fn listed_wave_pairs_scope_is_restored() {
    assert!(!WAVE_PAIRS_HERE.get());
    with_wave_pairs(true, || {
        assert!(WAVE_PAIRS_HERE.get());
        with_wave_pairs(false, || assert!(!WAVE_PAIRS_HERE.get()));
        assert!(WAVE_PAIRS_HERE.get());
    });
    assert!(!WAVE_PAIRS_HERE.get());
}

/// Emits a program whose earlier stages already ran.
pub fn emit(
    c: &Container,
    native: &NativeProgram,
    p: &ir::Program,
    st: &structure::Structured,
    io: &io::Io,
    res: &resource::Resources,
) -> Result<HostShader, ShaderError> {
    // (what an earlier program's lowering on this thread left behind)
    lower::take_needs();
    let wave32_compute =
        native.stage == HwStage::Cs && native.wave64 && lower::host_max_wave() == 32;
    let listed = WAVE_PAIRS_HERE.get() || wave_pairs_listed(c.input);
    if listed && wave32_compute {
        let by_default = std::env::var_os("VICTORY_WAVE_PAIRS_ONLY").is_none() && paired_by_default(c.input);
        let was = PAIRED_BY_DEFAULT_HERE.replace(by_default);
        let tried = paired(c, native, p);
        PAIRED_BY_DEFAULT_HERE.set(was);
        match tried {
            Ok((shader, Some(32))) => {
                eprintln!("WAVE64-PAIRS-ONLY translated with paired Wave64");
                return Ok(shader);
            }
            Ok((_, wave)) => eprintln!("WAVE64-PAIRS-ONLY unexpected wave {wave:?}; falling back"),
            Err(e) => eprintln!("WAVE64-PAIRS-ONLY refused: {e}; falling back"),
        }
        lower::take_needs();
    }
    let original = emit_program(c, native, p, st, io, res);
    // A console wave64 compute program on a host of 32-lane waves that
    // still needs 64 lanes, or whose cross-lane code sits where not every
    // lane runs: once more with its early-outs left out
    // ([`ir::without_early_outs`]), which is what usually keeps the whole
    // body from being a block every lane runs.
    let stuck = match &original {
        Ok((_, wave)) => *wave == Some(64),
        Err(ShaderError::Unsupported { what, .. }) => lower::needs_every_lane(what),
        Err(_) => false,
    };
    let pairs_on = wave_pairs_on() && !listed;
    if wave32_compute
        && pairs_on
        && !stuck
        && (matches!(native.io, crate::native::StageIo::Cs(cs) if cs.lds_bytes() > 0)
            || lower::crosses_halves(p, st)
                && (original.is_err() || lower::wave_local(p, st).is_err()))
    {
        let again = paired(c, native, p);
        if std::env::var_os("VICTORY_WAVE_WHY").is_some() {
            match &again {
                Ok((_, wave)) => eprintln!("WAVE64-RETRY paired Wave64: wave {wave:?}"),
                Err(e) => eprintln!("WAVE64-RETRY paired Wave64: {e}"),
            }
        }
        if let Ok((shader, Some(32))) = again {
            return Ok(shader);
        }
    }
    // (a mask with lanes elsewhere is only helped by the exact forms: not by each half-wave
    // on its own)
    let exact_only = matches!(&original, Err(ShaderError::Unsupported { what, .. }) if *what == lower::MASK_ELSEWHERE);
    if stuck && wave32_compute {
        let why = std::env::var_os("VICTORY_WAVE_WHY").is_some();
        let report = |what: &str, r: &Result<(HostShader, Option<u32>), ShaderError>| {
            if why {
                match r {
                    Ok((_, wave)) => eprintln!("WAVE64-RETRY {what}: wave {wave:?}"),
                    Err(e) => eprintln!("WAVE64-RETRY {what}: {e}"),
                }
            }
        };
        let done = |r: &Result<(HostShader, Option<u32>), ShaderError>| matches!(r, Ok((_, wave)) if *wave != Some(64));
        // Each host wave as a wave of its own (`lower::with_local_waves`),
        // for programs still stopped where not every lane runs.
        let local_on = std::env::var_os("VICTORY_WAVE_LOCAL").is_none_or(|v| v != "0");
        // Where the wave operations are the allocation idiom
        // ([`lower::wave_local`]) that is exact. Other programs are taken
        // the same way last of all, each half-wave giving its own results
        // (approximate: a tile-wide result becomes two half-tile ones);
        // `VICTORY_WAVE_LOCAL=idiom` refuses those instead.
        let strict = std::env::var_os("VICTORY_WAVE_LOCAL").is_some_and(|v| v == "idiom");
        let idiom = |p: &ir::Program, st: &structure::Structured| match lower::wave_local(p, st) {
            Ok(()) => true,
            Err((offset, what)) => {
                if why {
                    eprintln!("WAVE64-RETRY local waves: not the allocation idiom at {offset:#x}: {what}");
                }
                false
            }
        };
        // The program in two other forms: without its early-outs, and
        // without any branch that only skips code no lane is left for.
        let form = |label: &str, p: Option<ir::Program>| {
            let Some(p) = p else {
                if why {
                    eprintln!("WAVE64-FORM {label}: ir returned None");
                }
                return None;
            };
            let st = match structure::structure(&p) {
                Ok(st) => st,
                Err(e) => {
                    if why {
                        eprintln!("WAVE64-FORM {label}: structure failed: {e}");
                    }
                    return None;
                }
            };
            let io = match io::raise(&p, native) {
                Ok(io) => io,
                Err(e) => {
                    if why {
                        eprintln!("WAVE64-FORM {label}: io::raise failed: {e}");
                    }
                    return None;
                }
            };
            let res = match resource::track(&p, &native.layout, native.stage, Some(&st)) {
                Ok(res) => res,
                Err(e) => {
                    if why {
                        eprintln!("WAVE64-FORM {label}: resource::track failed: {e}");
                    }
                    return None;
                }
            };
            Some((p, st, io, res))
        };
        let forms = [
            ("without early-outs", form("without early-outs", ir::without_early_outs(p))),
            ("without skips", form("without skips", ir::without_skips(p, false))),
        ];
        for (name, f) in &forms {
            if let Some((p, st, io, res)) = f {
                let again = emit_program(c, native, p, st, io, res);
                report(name, &again);
                if done(&again) {
                    return again.map(|(shader, _)| shader);
                }
            }
        }
        // The full fallback is restricted to selected shaders by default: this form
        // puts group barriers inside branches and loops, and runs 121-124
        // lost the card in a submission of two programs that had just moved to it
        // (`ptfx_CS_lightAtlas_Sprite` and `_Trail`). The 32-lane programs' form below, which
        // has no barrier, stays on.
        // VICTORY_WAVE_WIDE_ONLY selects additional containers, or limits an explicit opt-in.
        let wide_option = std::env::var_os("VICTORY_WAVE_WIDE");
        // Both archived kernels use one logical wave per group and have
        // validated wave32 PSOs on RTX. Explicit policy wins.
        let scene_kernel = matches!(native.io, crate::native::StageIo::Cs(cs) if cs.threads.iter().product::<u32>() == 64)
            && matches!(
                c.digest,
                [0x58, 0x1d, 0xab, 0x2c, 0xd8, 0xbc, 0xf2, 0x2e, 0x5c, 0x96, 0x4b, 0x98, 0xe9, 0x03, 0x48, 0xd1]
                    | [0x24, 0x31, 0xbb, 0xe7, 0xdc, 0x15, 0x4a, 0xd7, 0x59, 0xd2, 0x0b, 0xc9, 0x54, 0x82, 0x9c, 0x1c]
            );
        let (wide_on, driver_loop_allowed) = wave_wide_routes(
            wide_option.as_deref(),
            std::env::var_os("VICTORY_WAVE_WIDE_ONLY").as_deref(),
            c.input,
            scene_kernel,
        );
        // A single guest wave can iterate an active-lane loop until both host
        // halves have finished. The precise latch and convergence conditions
        // for the driver compute route are checked by widen().
        let driver_loop = driver_loop_allowed
            && c.part(FourCC::DXIL).is_none()
            && c.part(FourCC::ACCC).is_some()
            && matches!(&original, Err(ShaderError::Unsupported { what, .. })
                if *what == lower::DIVERGENT_MASK_EXCHANGE);
        // exact first: each host wave on its own where that is exact (the idiom, every
        // form; no barriers), then the other exact forms; then approximate
        for exact in [true, false] {
            if !exact {
                if pairs_on {
                    let again = paired(c, native, p);
                    report("paired Wave64", &again);
                    if done(&again) {
                        return again.map(|(shader, _)| shader);
                    }
                }
                // Before anything approximate, the exact forms that cost group barriers.
                // Every skip left out that the code after it cannot see
                // ([`ir::without_skips`], `deep`): most of the body is then code every
                // lane runs, where the two host waves can exchange.
                // (`VICTORY_WAVE_DEEP_SKIPS=0` and `VICTORY_WAVE_WIDE=0` turn the two off.)
                let deep_on = std::env::var_os("VICTORY_WAVE_DEEP_SKIPS").is_none_or(|v| v != "0");
                let deep = if deep_on { form("without skips, scalars followed", ir::without_skips(p, true)) } else { None };
                if let Some((p, st, io, res)) = &deep {
                    let again = emit_program(c, native, p, st, io, res);
                    report("without skips, scalars followed", &again);
                    if done(&again) {
                        return again.map(|(shader, _)| shader);
                    }
                }
                // Still stopped where not every lane runs: the per-lane branches around
                // those places taken as the console takes them, by the whole wave at once
                // ([`widen`]), at two group barriers per such branch.
                if wide_on || driver_loop {
                    let bases = [
                        deep.as_ref().map(|f| &f.0),
                        forms[1].1.as_ref().map(|f| &f.0),
                        forms[0].1.as_ref().map(|f| &f.0),
                        Some(p),
                    ];
                    if let Some(base) = bases.into_iter().flatten().next() {
                        let again = widen(c, native, base, why, driver_loop);
                        report("wave-wide branches", &again);
                        if done(&again) {
                            return again.map(|(shader, _)| shader);
                        }
                    }
                }
                if strict {
                    break;
                }
            }
            if !local_on || exact_only {
                continue;
            }
            for (name, f) in &forms {
                if let Some((p, st, io, res)) = f
                    && idiom(p, st) == exact
                {
                    let local = lower::with_local_waves(|| emit_program(c, native, p, st, io, res));
                    report(&format!("local waves ({}), {name}", if exact { "idiom" } else { "APPROXIMATE" }), &local);
                    if done(&local) {
                        return local.map(|(shader, _)| shader);
                    }
                }
            }
            if idiom(p, st) == exact {
                let local = lower::with_local_waves(|| emit_program(c, native, p, st, io, res));
                report(if exact { "local waves (idiom)" } else { "local waves (APPROXIMATE)" }, &local);
                if done(&local) {
                    return local.map(|(shader, _)| shader);
                }
            }
        }
        if local_on && !exact_only {
            // Last: every mask and lane read per host wave.
            if !strict {
                for (name, f) in &forms {
                    if let Some((p, st, io, res)) = f {
                        let local = lower::with_local_waves_any(|| emit_program(c, native, p, st, io, res));
                        report(&format!("local waves (APPROXIMATE, any mask), {name}"), &local);
                        if done(&local) {
                            return local.map(|(shader, _)| shader);
                        }
                    }
                }
                let local = lower::with_local_waves_any(|| emit_program(c, native, p, st, io, res));
                report("local waves (APPROXIMATE, any mask)", &local);
                if done(&local) {
                    return local.map(|(shader, _)| shader);
                }
            }
        }
        // Last of all (`VICTORY_WAVE_LDS_LOOSE=0` turns it off): LDS read and written inside
        // one lane-branch region, ordered between the two host waves only where the region
        // is entered and left (`loose_lds::with`, approximate). A 32-lane host refuses
        // the 64-lane form outright, which drops every dispatch of the program.
        let loose_on = std::env::var_os("VICTORY_WAVE_LDS_LOOSE").is_none_or(|v| v != "0");
        if loose_on && !strict {
            let bases = [forms[0].1.as_ref(), forms[1].1.as_ref()];
            for (name, f) in [("without early-outs", bases[0]), ("without skips", bases[1])] {
                if let Some((p, st, io, res)) = f {
                    let again = loose_lds::with(|| emit_program(c, native, p, st, io, res));
                    report(&format!("loose LDS (APPROXIMATE), {name}"), &again);
                    if done(&again) {
                        return again.map(|(shader, _)| shader);
                    }
                }
            }
            let again = loose_lds::with(|| emit_program(c, native, p, st, io, res));
            report("loose LDS (APPROXIMATE)", &again);
            if done(&again) {
                return again.map(|(shader, _)| shader);
            }
            if local_on && !exact_only {
                let local = loose_lds::with(|| lower::with_local_waves_any(|| emit_program(c, native, p, st, io, res)));
                report("loose LDS + local waves (APPROXIMATE, any mask)", &local);
                if done(&local) {
                    return local.map(|(shader, _)| shader);
                }
            }
        }
    }
    // SCC of a mask operation joined with a scalar test behind a per-lane skip (the 32-lane
    // trace programs): right once the wave takes that skip together ([`widen`]; the
    // lowering has named the skipped blocks).
    let original = match original {
        Err(ShaderError::Unsupported { what, .. })
            if native.stage == HwStage::Cs
                && what == lower::LANE_MASK
                && std::env::var_os("VICTORY_WAVE_WIDE").is_none_or(|v| v != "0") =>
        {
            let again = widen(c, native, p, std::env::var_os("VICTORY_WAVE_WHY").is_some(), false);
            match again {
                Ok(_) => again,
                Err(_) => emit_program(c, native, p, st, io, res),
            }
        }
        other => other,
    };
    let supported = |(shader, wave): (HostShader, Option<u32>)| {
        if wave == Some(64) && lower::host_max_wave() == 32 {
            Err(ShaderError::Unsupported {
                offset: 0,
                what: "no wave32 lowering for this wave64 program",
            })
        } else {
            Ok(shader)
        }
    };
    let original = original.and_then(supported);
    // Retry unsupported programs only. Threading can separate an unknown
    // register value from its apparent use on a mutually exclusive path.
    if !matches!(original, Err(ShaderError::Unsupported { .. })) {
        return original;
    }
    let edges = threading::constant_edges(p, st);
    if edges.is_empty() {
        return original;
    }
    let p = ir::thread_edges(p, &edges)?;
    let st = structure::structure(&p)?;
    let io = io::raise(&p, native)?;
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    emit_program(c, native, &p, &st, &io, &res).and_then(supported)
}

/// `p` (a console wave64 compute program on a host of 32-lane waves) with the per-lane
/// branches around its cross-lane code taken wave-wide ([`structure::structure_wide`]): the
/// lowering names the blocks where the two host waves must meet and could not
/// (`lower::take_needs`), the branches whose regions hold them are widened, and so on until
/// it translates or a branch cannot be widened. Only where a thread group is one console
/// wave: with several, the waves of a group would decide differently and a barrier inside
/// such a branch would wait for threads that never come.
fn widen(
    c: &Container,
    native: &NativeProgram,
    p: &ir::Program,
    why: bool,
    active_loop_only: bool,
) -> Result<(HostShader, Option<u32>), ShaderError> {
    const NOT: ShaderError = ShaderError::Unsupported {
        offset: 0,
        what: "wave-wide branches in a thread group of several console waves",
    };
    match native.io {
        // (a program whose waves are the host's meets no one: no barrier, any group)
        crate::native::StageIo::Cs(_) if !(native.wave64 && lower::host_max_wave() == 32) => {}
        crate::native::StageIo::Cs(cs) if cs.threads.iter().product::<u32>() == 64 => {}
        crate::native::StageIo::Cs(cs) => {
            if why {
                eprintln!("WAVE64-WIDE not tried: thread groups of {:?}", cs.threads);
            }
            return Err(NOT);
        }
        _ => return Err(NOT),
    }
    let mut wide = std::collections::HashSet::new();
    loop {
        let st = structure::structure_wide(p, &wide)?;
        let io = io::raise(p, native)?;
        let res = resource::track(p, &native.layout, native.stage, Some(&st))?;
        lower::take_needs();
        let r = emit_program(c, native, p, &st, &io, &res);
        let needs = lower::take_needs();
        // (or translated for 64-lane waves still: shared memory read and written where not
        // every lane runs, whose blocks the lowering names too)
        let stuck = matches!(&r, Err(ShaderError::Unsupported { what, .. })
                if lower::needs_every_lane(what) || *what == lower::LANE_MASK && !needs.is_empty())
            || matches!(&r, Ok((_, Some(64)))) && !needs.is_empty();
        if !stuck {
            if why {
                let order = ir::reverse_postorder(&p.blocks);
                let mut rpo = vec![0usize; p.blocks.len()];
                for (i, &b) in order.iter().enumerate() {
                    rpo[b] = i;
                }
                let mut sorted: Vec<usize> = wide.iter().copied().collect();
                sorted.sort_unstable();
                let desc: Vec<String> = sorted
                    .iter()
                    .map(|&b| {
                        let back = matches!(p.blocks[b].exit, ir::Exit::Branch { taken, not_taken } if rpo[taken] <= rpo[b] || rpo[not_taken] <= rpo[b]);
                        let nm = p.blocks[b].insts.last().map_or("?", |i| i.name());
                        format!("B{b}({nm}{})", if back { ",BACK" } else { "" })
                    })
                    .collect();
                eprintln!("WAVE64-WIDE {} branches taken wave-wide: [{}]", wide.len(), desc.join(", "));
            }
            return r;
        }
        let more: Vec<usize> = lower::branches_around_with_loop_exits(&st, &needs, native.wave64 && lower::host_max_wave() == 32 && lower::loop_exits_on())
            .unwrap_or_default()
            .into_iter()
            .filter(|b| !wide.contains(b))
            .collect();
        if more.is_empty() {
            if why {
                eprintln!("WAVE64-WIDE stopped with {} branches wave-wide: a branch around it cannot be", wide.len());
            }
            return r;
        }
        if active_loop_only {
            // Only a single backward branch on the SCC just set by
            // s_andn2_wrexec_b64. This instruction restricts EXEC to the
            // unfinished lanes. The guest repeats the loop if *either*
            // half still has work, while individual vector writes retain
            // their EXEC predicates.
            let [latch] = more.as_slice() else {
                return r;
            };
            let block = &p.blocks[*latch];
            let [.., update, branch] = block.insts.as_slice() else {
                return r;
            };
            if !wide.is_empty()
                || update.name() != "s_andn2_wrexec_b64"
                || branch.name() != "s_cbranch_scc1"
                || !matches!(
                    block.exit,
                    ir::Exit::Branch { taken, not_taken } if taken <= *latch && not_taken > *latch
                )
            {
                return r;
            }
        }
        wide.extend(more);
    }
}

fn wave_pairs_on() -> bool {
    std::env::var_os("VICTORY_WAVE_PAIRS").is_some_and(|v| v == "1")
}

// Paired lanes from PR #854 (PiloteCode), including the per-wave CFG used by 840243e2.
fn paired(
    c: &Container,
    native: &NativeProgram,
    p: &ir::Program,
) -> Result<(HostShader, Option<u32>), ShaderError> {
    let crate::native::StageIo::Cs(cs) = native.io else {
        return Err(ShaderError::Unsupported {
            offset: 0,
            what: "paired lanes outside compute",
        });
    };
    if crate::spill::is_spilled(native) {
        return Err(ShaderError::Unsupported {
            offset: 0,
            what: "paired Wave64 for spilled DispatchX programs is not validated",
        });
    }
    let threads = cs.threads.iter().product::<u32>();
    if threads == 0 || threads > 1024 || threads % 64 != 0 {
        return Err(ShaderError::Unsupported {
            offset: 0,
            what: "paired lanes in a thread group that does not fill its console waves",
        });
    }
    if p.blocks.first().is_some_and(|b| !b.preds.is_empty()) {
        return Err(ShaderError::Unsupported {
            offset: 0,
            what: "paired Wave64 requires a single-entry wave prologue",
        });
    }
    let lds = cs.lds_bytes();
    if lds + 4 * u32::from(threads > 64) > 32 * 1024 {
        return Err(ShaderError::Unsupported {
            offset: 0,
            what: "paired Wave64 exceeds the compute group shared-memory limit",
        });
    }
    let insts = || p.blocks.iter().flat_map(|b| &b.insts);
    let accesses_lds = insts().any(lower::accesses_lds);
    if accesses_lds
        && (lds == 0 || insts().filter(|i| lower::accesses_lds(i)).any(|i| {
            !lower::supports_paired_lds(i.name())
        }))
    {
        return Err(ShaderError::Unsupported {
            offset: 0,
            what: "an LDS operation not supported by paired Wave64",
        });
    }
    let waits = accesses_lds || insts().any(|i| i.name() == "s_barrier");
    // Independently progressing waves cannot rendezvous after a divergent exit.
    if (threads > 64 && p.blocks.len() > 1 && waits)
        || (group_sync() == 0 && (threads > 64 || waits))
    {
        return Err(ShaderError::Unsupported {
            offset: 0,
            what: "paired Wave64 would require an unavailable group rendezvous",
        });
    }
    // every per-lane branch the wave's; again while that leaves another branch per lane
    let mut wide = std::collections::HashSet::new();
    let st = loop {
        let st = structure::structure_wide(p, &wide)?;
        let lanes: Vec<usize> = (0..st.exits.len())
            .filter(|b| !wide.contains(b))
            .filter(|&b| {
                matches!(st.exits[b], structure::LaneExit::Branch { when, .. }
                    if !st.bools.support(when).iter().all(|a| matches!(a, structure::Atom::Uniform(_))))
            })
            .collect();
        if lanes.is_empty() {
            break st;
        }
        wide.extend(lanes);
    };
    let io = io::raise(p, native)?;
    let res = resource::track(p, &native.layout, native.stage, Some(&st))?;
    lower::with_paired_lanes(|| emit_program(c, native, p, &st, &io, &res))
}

/// The translated program, and the wave size it requires of the host.
fn emit_program(
    c: &Container,
    native: &NativeProgram,
    p: &ir::Program,
    st: &structure::Structured,
    io: &io::Io,
    res: &resource::Resources,
) -> Result<(HostShader, Option<u32>), ShaderError> {
    let sig = |fourcc| -> Result<Vec<container::SigElement>, ShaderError> {
        match c.part(fourcc) {
            Some(part) => signature(part.data),
            None => Ok(Vec::new()),
        }
    };
    let pixel = native.stage == HwStage::Ps;
    let mut inputs = sig(FourCC::ISG1)?;
    let mut outputs = sig(FourCC::OSG1)?;
    // Interpolation as DXC declares it by default; pixel inputs that are
    // read get the mode their barycentrics say.
    let interpolated = if pixel { &mut inputs } else { &mut outputs };
    for e in interpolated {
        e.interp = match (e.kind, e.comp) {
            (3, _) => 4,
            // Constant, as DXC declares it (the AMD compiler crashed on a
            // front face without a mode).
            (13, _) => 1,
            // Until a read says which mode (`lower`).
            (28, _) => 2,
            (0 | 6 | 7, 9) => 2,
            (0, _) => 1,
            _ => 0,
        };
    }
    // (DexNom/VIctory 5f4c3df4.)
    if native.stage == HwStage::Vs
        && let Some(psv) = c.part(FourCC::PSV0)
    {
        vertex_output_interpolation(psv.data, &mut outputs)?;
    }
    // With a conservative depth output (SV_DepthLessEqual, GreaterEqual) DXIL
    // wants the position centroid (or per sample), as DXC declares it then.
    if pixel && outputs.iter().any(|e| matches!(e.kind, 18 | 19)) {
        for e in inputs.iter_mut().filter(|e| e.kind == 3) {
            e.interp = 5;
        }
    }
    // Per-primitive outputs are constant. The console never exports a
    // primitive id or cull flag a mesh shader writes (it culls with the
    // primitive export), so those rows are left out.
    let mut prims = sig(FourCC::PSG1)?;
    prims.retain(|e| !matches!(e.kind, 10 | 30));
    for e in &mut prims {
        e.interp = 1;
    }
    // A mesh group's id arrives multiplied by max(threads, vertices,
    // primitives) (the code divides by that); the thread count is in PSV0
    // from runtime info version 2 on.
    // shortcut: older containers fall back to the group's lane count, right
    // when the HLSL group fills its waves (the title's do).
    let mesh_stride = match &native.io {
        crate::native::StageIo::Ms(ms) => {
            let psv = c.part(FourCC::PSV0).map(|p| p.data);
            let threads = psv
                .filter(|d| crate::dxbc::u32_at(d, 0).is_ok_and(|n| n >= 48))
                .and_then(|d| crate::dxbc::u32_at(d, 4 + 36).ok());
            threads.map_or(ms.threads, |t| {
                t.max(ms.max_vertices).max(ms.max_primitives)
            })
        }
        _ => 0,
    };
    let l = lower::lower(&lower::Parts {
        native,
        p,
        st,
        io,
        res,
        inputs,
        outputs,
        prims,
        mesh_stride,
    })?;
    let wave_size = l.wave_size;
    let shader = HostShader {
        stage: native.stage,
        dxil: l.dxil,
        user_data: l.user_data,
        static_samplers: l
            .samplers
            .iter()
            .enumerate()
            .map(|(n, &w)| {
                let mut s = StaticSampler::from_words(w);
                if l.cmp_samplers >> n & 1 != 0 {
                    s.make_comparison();
                }
                s
            })
            .collect(),
        samplers: l.samplers,
        root_cbvs: l.root_cbvs,
        root_srvs: l.root_srvs,
        root_uavs: l.root_uavs,
        root_indexed_views: res.root_indexed_views.clone(),
    };
    Ok((shader, wave_size))
}

/// The title's vertex output interpolation lives in PSV0. A float packed
/// beside an integer can be explicitly constant, as can a float on its own.
/// OSG1 supplies the physical rows and columns used to match both tables.
fn vertex_output_interpolation(
    mut data: &[u8],
    outputs: &mut [container::SigElement],
) -> Result<(), ShaderError> {
    const TRUNCATED: ShaderError =
        ShaderError::Malformed("truncated vertex pipeline validation data");
    fn take<'a>(data: &mut &'a [u8], len: usize) -> Result<&'a [u8], ShaderError> {
        let (head, tail) = data.split_at_checked(len).ok_or(TRUNCATED)?;
        *data = tail;
        Ok(head)
    }
    fn size(data: &mut &[u8]) -> Result<usize, ShaderError> {
        Ok(u32::from_le_bytes(take(data, 4)?.try_into().unwrap()) as usize)
    }
    let info_size = size(&mut data)?;
    let info = take(&mut data, info_size)?;
    // PSVRuntimeInfo0 has no signature. PSVRuntimeInfo1 and later share
    // these stage and signature count fields.
    if info.len() < 36 || info[24] != 1 || info[29] == 0 {
        return Ok(());
    }
    let resources = size(&mut data)?;
    if resources != 0 {
        let stride = size(&mut data)?;
        take(&mut data, resources.checked_mul(stride).ok_or(TRUNCATED)?)?;
    }
    let strings = size(&mut data)?;
    take(&mut data, strings)?;
    let indices = size(&mut data)?;
    take(&mut data, indices.checked_mul(4).ok_or(TRUNCATED)?)?;
    let stride = size(&mut data)?;
    if stride < 16 {
        return Err(TRUNCATED);
    }
    let count = usize::from(info[28]) + usize::from(info[29]);
    let elements = take(&mut data, count.checked_mul(stride).ok_or(TRUNCATED)?)?;
    for record in elements.chunks_exact(stride).skip(usize::from(info[28])) {
        let packed = record[10];
        if packed & 0x40 == 0 {
            continue;
        }
        if record[13] > 7 {
            return Err(ShaderError::Malformed("invalid vertex output interpolation"));
        }
        let first = u32::from(record[9]);
        let rows = first..first + u32::from(record[8]);
        for output in outputs.iter_mut().filter(|e| {
            e.row.is_some_and(|r| rows.contains(&r))
                && e.col == ((packed >> 4) & 3)
                && e.cols == (packed & 15)
                && e.kind == record[11]
        }) {
            output.interp = record[13];
        }
    }
    Ok(())
}

/// The rows of an `ISG1`/`OSG1` part.
fn signature(part: &[u8]) -> Result<Vec<container::SigElement>, ShaderError> {
    use crate::dxbc::u32_at;
    let count = u32_at(part, 0)? as usize;
    let base = u32_at(part, 4)? as usize;
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let at = base + 32 * i;
        let name_at = u32_at(part, at + 4)? as usize;
        let name = part
            .get(name_at..)
            .and_then(|s| s.split(|&b| b == 0).next())
            .ok_or(ShaderError::Malformed("signature name out of bounds"))?;
        let mask = *part
            .get(at + 24)
            .ok_or(ShaderError::Malformed("truncated"))?;
        if u32_at(part, at + 28)? != 0 {
            return Err(ShaderError::Unsupported {
                offset: 0,
                what: "a minimum-precision signature element",
            });
        }
        if u32_at(part, at)? != 0 {
            return Err(ShaderError::Unsupported {
                offset: 0,
                what: "a signature element of another stream",
            });
        }
        let kind = match u32_at(part, at + 12)? {
            0 => 0,
            1 => 3,
            2 => 6,
            3 => 7,
            4 => 4,
            5 => 5,
            6 => 1,
            7 => 10,
            8 => 2,
            9 => 13,
            10 => 12,
            64 => 16,
            65 => 17,
            66 => 14,
            67 => 19,
            68 => 18,
            69 => 20,
            70 => 15,
            23 => 28, // barycentrics
            25 => 30, // cull primitive
            _ => {
                return Err(ShaderError::Unsupported {
                    offset: 0,
                    what: "a system value in the signature",
                });
            }
        };
        // 16-bit elements are declared 32-bit; reading or writing one
        // converts (16-bit values travel packed on the console).
        let raw = u32_at(part, at + 16)?;
        let comp = match (kind, raw) {
            // DXIL declares the front face a uint (the validator insists).
            (13, _) => 5,
            (_, 1 | 4) => 5,
            (_, 2 | 5) => 4,
            (_, 3 | 6) => 9,
            (_, c) => {
                return Err(ShaderError::Unsupported {
                    offset: 0,
                    what: match c {
                        0 => "a signature element of unknown type",
                        _ => "a 64-bit signature element",
                    },
                });
            }
        };
        let register = u32_at(part, at + 20)?;
        out.push(container::SigElement {
            name: String::from_utf8_lossy(name).into_owned(),
            index: u32_at(part, at + 8)?,
            kind,
            comp,
            interp: 0,
            row: (register != u32::MAX).then_some(register),
            col: mask.trailing_zeros().min(3) as u8,
            cols: mask.count_ones().max(1) as u8,
            usage: 0,
            half: (4..=6).contains(&raw),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::bitcode::Attr;
    use super::lower::LOOP_LIMIT;
    use super::parse_loop_limit;

    #[test]
    fn explicit_wave_wide_list_matches_only_named_containers() {
        use std::ffi::OsStr;

        let water = b"selected compute container";
        let other = b"another compute container";
        let hash = blake3::hash(water).to_hex();
        let list = format!("invalid, {}", hash[..16].to_ascii_uppercase());
        assert!(super::wave_wide_hash_matches(OsStr::new(&list), &hash[..16]));
        let hash = blake3::hash(other).to_hex();
        assert!(!super::wave_wide_hash_matches(OsStr::new(&list), &hash[..16]));
        assert!(!super::wave_wide_hash_matches(OsStr::new("invalid"), &hash[..16]));
    }

    #[test]
    fn wave_wide_allowlist_keeps_the_default_driver_loop_for_unlisted_shaders() {
        use std::ffi::OsStr;

        let listed = b"listed shader";
        let unlisted = b"unlisted shader";
        let hash = blake3::hash(listed).to_hex();
        let allowlist = Some(OsStr::new(&hash[..16]));
        let on = Some(OsStr::new("1"));
        let off = Some(OsStr::new("0"));
        let routes = super::wave_wide_routes;

        assert_eq!(routes(None, allowlist, unlisted, false), (false, true));
        assert_eq!(routes(None, allowlist, listed, false), (true, false));
        assert_eq!(routes(None, allowlist, unlisted, true), (true, true));
        assert_eq!(routes(on, allowlist, listed, false), (true, false));
        assert_eq!(routes(on, allowlist, unlisted, false), (false, true));
        assert_eq!(routes(on, None, unlisted, false), (true, false));
        assert_eq!(routes(on, Some(OsStr::new("")), unlisted, false), (true, false));
        assert_eq!(routes(off, allowlist, unlisted, false), (false, false));
        assert_eq!(routes(off, allowlist, listed, true), (false, false));

        let key = super::wave_wide_override;
        assert_eq!(key(None, allowlist, listed), Some(true));
        assert_eq!(key(None, allowlist, unlisted), key(None, None, unlisted));
        assert_eq!(key(on, allowlist, listed), key(on, None, listed));
        assert_eq!(key(on, allowlist, unlisted), Some(false));
        assert_eq!(key(off, allowlist, listed), key(off, None, listed));
    }

    /// The LLVM text `dxc -dumpbin` prints for a DXIL container, when dxc
    /// is installed.
    #[cfg(windows)]
    fn dump_dxil(container: &[u8]) -> Option<String> {
        let dxc = match std::env::var_os("VICTORY_PC_DXC") {
            Some(p) => std::path::PathBuf::from(p),
            None => super::validator::Validator::default_path()?.with_file_name("dxc.exe"),
        };
        if !dxc.exists() {
            return None;
        }
        let file = std::env::temp_dir().join(format!("victory-interp-{}.dxil", std::process::id()));
        std::fs::write(&file, container).ok()?;
        let out = std::process::Command::new(dxc)
            .arg("-dumpbin")
            .arg(&file)
            .output()
            .ok()?;
        let _ = std::fs::remove_file(&file);
        assert!(
            out.status.success(),
            "dxc -dumpbin failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        Some(String::from_utf8_lossy(&out.stdout).into_owned())
    }

    #[cfg(windows)]
    #[test]
    fn interpolated_f16_writes_the_op_sel_half_with_modifiers() {
        // p1_f32 v2, v0, attr0.x; then a p2_f16 completing it into v3, which holds a value
        // (v2's own is no number here: a first half is evaluated with its second).
        let p2 = |dst_high: bool, omod: u32, clamp: bool| {
            let vgpr = |n: u32| 256 + n;
            let mut w0 = 0x35u32 << 26 | 0x35au32 << 16 | 3;
            if dst_high {
                w0 |= 1 << 14; // op_sel[3]: the high destination half.
            }
            if clamp {
                w0 |= 1 << 15;
            }
            let w1 = vgpr(1) << 9 | vgpr(2) << 18;
            [w0, w1 | omod << 27]
        };
        let dump = |dst_high: bool, omod: u32, clamp: bool| {
            let [w0, w1] = p2(dst_high, omod, clamp);
            let words = [
                0x7E06_0301u32, // v_mov_b32 v3, v1
                0xC808_0000, // v_interp_p1_f32 v2, v0, attr0.x
                w0,
                w1, // v_interp_p2_f16 v3, attr0, v1, v2
                0xF800_180F,
                0x0303_0303, // exp mrt0, v3, v3, v3, v3 done vm
                0xBF81_0000,
            ];
            let native = super::NativeProgram {
                stage: super::HwStage::Ps,
                wave64: false,
                vgprs: 4,
                layout: vec![],
                used: vec![],
                io: crate::native::StageIo::Ps {
                    inputs: vec![crate::native::PsInput {
                        usage: crate::native::Usage::Generic,
                        row: 0,
                        mask: 0b1111,
                        attr: 0,
                        flat: false,
                        fp16: true,
                        pack2: false,
                        row2: 0,
                    }],
                    input_addr: 2,
                    db_shader_control: 0,
                    patches: vec![],
                },
                code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
            };
            let el = |name: &str, kind, usage, half| SigElement {
                name: name.into(),
                index: 0,
                kind,
                comp: 9,
                interp: 0,
                row: Some(0),
                col: 0,
                cols: 4,
                usage,
                half,
            };
            let isg1 = super::container::signature(&[el("TEXCOORD", 0, 0, true)], true);
            let osg1 = super::container::signature(&[el("SV_Target", 16, 15, false)], false);
            let bytes =
                crate::dxbc::build(&[(super::FourCC::ISG1, &isg1), (super::FourCC::OSG1, &osg1)]);
            let c = super::Container::parse(&bytes).unwrap();
            let p = crate::ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
            let st = crate::structure::structure(&p).unwrap();
            let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
            let io = crate::io::raise(&p, &native).unwrap();
            super::emit(&c, &native, &p, &st, &io, &res).unwrap().dxil
        };
        let (Some(low), Some(high)) = (
            dump_dxil(&dump(false, 0, false)),
            dump_dxil(&dump(true, 1, true)),
        ) else {
            eprintln!("no dxc.exe; DXIL check skipped");
            return;
        };
        // The low-half write keeps the destination's high half.
        assert!(low.contains("-65536"), "low write keeps the high half");
        // The high-half write shifts the result up by 16.
        assert!(high.contains("shl i32"), "op_sel[3] writes the high half");
        // omod mul:2 and clamp (dx.op.unary.f32, opcode 7) are applied.
        assert!(high.contains("2.000000e+00"), "omod multiplies by two");
        assert!(
            high.contains("dx.op.unary.f32(i32 7"),
            "clamp saturates the result"
        );
        assert!(!low.contains("dx.op.unary.f32(i32 7"));
    }

    #[test]
    fn the_clean_up_switch_parses_or_runs_every_pass() {
        let (all, usual) = (super::bitcode::opt::ALL, super::bitcode::opt::DEFAULT);
        assert_eq!(usual, 63, "the passes of the builds before the wide loads");
        assert_eq!(parse_shader_opt(None), usual);
        assert_eq!(parse_shader_opt(Some("0")), 0);
        assert_eq!(parse_shader_opt(Some(" 1 ")), 1);
        assert_eq!(parse_shader_opt(Some("0x1")), 1);
        assert_eq!(parse_shader_opt(Some("127")), all);
        assert_eq!(parse_shader_opt(Some("0xffffffff")), all);
        assert_eq!(parse_shader_opt(Some("on")), usual);
    }

    #[test]
    fn the_loop_limit_switch_parses_or_falls_back() {
        assert_eq!(parse_loop_limit(None), LOOP_LIMIT);
        assert_eq!(parse_loop_limit(Some("4096")), 4096);
        assert_eq!(parse_loop_limit(Some(" 0x1000 ")), 4096);
        assert_eq!(parse_loop_limit(Some("1")), 1);
        for bad in ["", "0", "-5", "lots", "0x", "4294967296"] {
            assert_eq!(parse_loop_limit(Some(bad)), LOOP_LIMIT, "{bad:?}");
        }
    }
    use super::container::SigElement;
    use super::shader::Shader;
    use super::*;
    use crate::native::{ComputeIo, StageIo};

    /// A pixel shader writing SV_StencilRef (channel y of `exp mrtz`).
    /// DXC sets ShaderFlags.StencilRef (bit 11) for a pixel shader whose
    /// output signature has SV_StencilRef, and the feature bit
    /// D3D_SHADER_FEATURE_STENCIL_REF (0x200) in SFI0; the validator checks
    /// both against the module.
    #[test]
    fn a_stencil_export_declares_the_stencil_ref_feature() {
        let words = [
            0x7E00_02F2u32, // v_mov_b32 v0, 1.0
            0x7E02_0285,    // v_mov_b32 v1, 5
            0xF800_1883,
            0x0000_0100, // exp mrtz v0, v1, off, off done vm
            0xBF81_0000,
        ];
        let native = NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            vgprs: 2,
            layout: vec![],
            used: vec![],
            io: StageIo::Ps {
                inputs: vec![],
                input_addr: 0,
                db_shader_control: 0,
                patches: vec![],
            },
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let el = |name: &str, kind, comp| SigElement {
            name: name.into(),
            index: 0,
            kind,
            comp,
            interp: 0,
            row: None,
            col: 0,
            cols: 1,
            usage: 1,
            half: false,
        };
        let osg1 =
            container::signature(&[el("SV_Depth", 17, 9), el("SV_StencilRef", 20, 5)], false);
        let bytes = crate::dxbc::build(&[(FourCC::OSG1, &osg1)]);
        let c = Container::parse(&bytes).unwrap();
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let out = emit(&c, &native, &p, &st, &io, &res).unwrap();
        // dxil.dll, where there is one, checks the flags against the module.
        #[cfg(windows)]
        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut out.dxil.clone()).unwrap();
        }
        let host = Container::parse(&out.dxil).unwrap();
        let sfi0 = host.part(FourCC::SFI0).unwrap().data;
        let features = u64::from_le_bytes(sfi0.try_into().unwrap());
        assert_ne!(features & 0x200, 0, "SFI0 {features:#x}: no StencilRef");
    }

    /// A level query (`image_get_lod`) of a 2D array passes no array slice:
    /// DXIL's `CalculateLOD` takes the spatial coordinates only and the
    /// validator refuses a defined third one ("out of bound coord must be
    /// undef"), whatever the program left in that register.
    #[test]
    fn a_level_query_of_an_array_takes_no_slice() {
        use crate::native::{ResourceType, RootParam, UsedResource};
        let mut words = vec![
            0xBE81_0380u32,
            0xBE83_0380, // s1, s3 = 0 (upper halves of the tables)
            0xF40C_0400,
            0xFA00_0000, // 2D array SRV to s[16:23] from s[0:1]
            0xF408_0601,
            0xFA00_0000, // sampler to s[24:27] from s[2:3]
        ];
        for (reg, value) in [0.25f32, 0.5, 7.0].iter().enumerate() {
            words.extend([0x7E00_02FF | ((reg as u32) << 17), value.to_bits()]);
        }
        words.extend([
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            // image_get_lod v[10:11], v[0:2], s[16:23], s[24:27] dmask:0x3 dim:2D_ARRAY
            0xF000_0328 | (0x60 << 18),
            0x00C4_0A00,
            0xBF8C_3F70, // s_waitcnt vmcnt(0)
            0xF800_180F,
            0x0B0A_0B0A, // exp mrt0, v10, v11, v10, v11 done vm
            0xBF81_0000,
        ]);
        let table = |slot, user_data| RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data,
            slot,
        };
        let range = |kind, words, slot, user_data| UsedResource {
            kind,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: words,
            user_data,
            slot,
        };
        let native = NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            vgprs: 12,
            layout: vec![table(0, 0), table(1, 2)],
            used: vec![range(ResourceType::Srv, 8, 0, 0), range(ResourceType::Sampler, 4, 1, 2)],
            io: StageIo::Ps {
                inputs: vec![],
                input_addr: 0,
                db_shader_control: 0,
                patches: vec![],
            },
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let target = SigElement {
            name: "SV_Target".into(),
            index: 0,
            kind: 16,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 0xF,
            half: false,
        };
        let osg1 = container::signature(&[target], false);
        let bytes = crate::dxbc::build(&[(FourCC::OSG1, &osg1)]);
        let c = Container::parse(&bytes).unwrap();
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let out = emit(&c, &native, &p, &st, &io, &res).unwrap();
        #[cfg(windows)]
        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut out.dxil.clone()).expect("the level query validates");
        }
    }

    /// A pixel program with an ordered section, as the console compiler
    /// makes one of rasterizer ordered views (the title's hair cards keep
    /// their nearest layers that way): the wait for overlapped waves is
    /// skipped, the views inside the section are the host's ordered ones,
    /// the one after it is not.
    #[test]
    fn an_ordered_section_uses_rasterizer_ordered_views() {
        let section = [
            0xBE82_0300u32, // s_mov_b32 s2, s0
            0xBE83_0380,    // s_mov_b32 s3, 0
            0xF40C_0201,    // s_load_dwordx8 s[8:15], s[2:3], 0
            0xFA00_0000,
            0x9390_FF01, // s_bfe_u32 s16, s1, 0x2001c
            0x0002_001C,
            0x9391_FF01, // s_bfe_u32 s17, s1, 0xa0000
            0x000A_0000,
            0x8F10_8110, // s_lshl_b32 s16, s16, 1
            0x9392_FF01, // s_bfe_u32 s18, s1, 0xa0010
            0x000A_0010,
            0xBE91_0711, // s_not_b32 s17, s17
            0x9013_9F01, // s_lshr_b32 s19, s1, 31
            0x8810_8110, // s_or_b32 s16, s16, 1
            0x8112_1112, // s_add_i32 s18, s18, s17
            0xBF06_8013, // s_cmp_eq_u32 s19, 0
            0xB990_1019, // s_setreg_b32 hwreg(HW_REG_POPS_PACKER, 0, 3), s16
            0x8514_807E, // s_cselect_b32 s20, exec_lo, 0
            0x8115_11EF, // s_add_i32 s21, src_pops_exiting_wave_id, s17
            0xBF08_1215, // s_cmp_gt_u32 s21, s18
            0x8516_807E, // s_cselect_b32 s22, exec_lo, 0
            0x8814_1614, // s_or_b32 s20, s20, s22
            0x876A_147E, // s_and_b32 vcc_lo, exec_lo, s20
            0xBF87_0004, // s_cbranch_vccnz past the wait
            0xBF8E_FFFF, // s_sleep 0xffff
            0x8115_11EF, // s_add_i32 s21, src_pops_exiting_wave_id, s17
            0xBF0A_1512, // s_cmp_lt_u32 s18, s21
            0xBF84_FFFC, // s_cbranch_scc0 back to the sleep
            0x7E00_0280, // v_mov_b32 v0, 0
            0x7E02_0280, // v_mov_b32 v1, 0
            0xF000_3F88, // image_load v[2:5], v[0:1], s[8:15] dmask:0xf 2D glc dlc
            0x0002_0200,
            0x0604_04F2, // v_add_f32 v2, 1.0, v2
            0xF020_3F08, // image_store v[2:5], v[0:1], s[8:15] dmask:0xf 2D glc
            0x0002_0200,
            0xBEFC_0380, // s_mov_b32 m0, 0
            0xBF90_0007, // s_sendmsg sendmsg(MSG_ORDERED_PS_DONE)
        ];
        let end = [
            0xF800_180Fu32, // exp mrt0 v2, v2, v2, v2 done vm
            0x0202_0202,
            0xBF81_0000, // s_endpgm
        ];
        // A store after the section, through an ordinary view.
        let after = [0xF020_3F08u32, 0x0002_0200];
        let lower = |words: Vec<u32>| {
            let native = NativeProgram {
                stage: HwStage::Ps,
                wave64: false,
                vgprs: 8,
                layout: vec![crate::native::RootParam {
                    root: RootType::Table,
                    table_offset_dwords: 0,
                    sgprs: 1,
                    base_register: 0,
                    user_data: 0,
                    slot: 0,
                }],
                used: vec![crate::native::UsedResource {
                    kind: crate::native::ResourceType::Uav,
                    root: RootType::Table,
                    table_offset_dwords: 0,
                    lower_bound: 0,
                    stride: 4,
                    dwords: 8,
                    user_data: 0,
                    slot: 0,
                }],
                io: StageIo::Ps {
                    inputs: vec![],
                    input_addr: 0,
                    db_shader_control: 0,
                    patches: vec![],
                },
                code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
            };
            let target = SigElement {
                name: "SV_Target".into(),
                index: 0,
                kind: 16,
                comp: 9,
                interp: 0,
                row: Some(0),
                col: 0,
                cols: 4,
                usage: 15,
                half: false,
            };
            let osg1 = container::signature(&[target], false);
            let bytes = crate::dxbc::build(&[(FourCC::OSG1, &osg1)]);
            let c = Container::parse(&bytes).unwrap();
            let p = ir::build(&native.patched_code(), native.wave64, native.vgprs)?;
            let st = structure::structure(&p)?;
            let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
            let io = io::raise(&p, &native)?;
            let out = emit(&c, &native, &p, &st, &io, &res)?;
            #[cfg(windows)]
            if let Ok(mut v) = super::validator::Validator::load(
                super::validator::Validator::default_path().as_deref(),
            ) {
                v.validate(&mut out.dxil.clone()).unwrap();
            }
            let host = Container::parse(&out.dxil).unwrap();
            let sfi0 = host.part(FourCC::SFI0).unwrap().data;
            Ok::<u64, ShaderError>(u64::from_le_bytes(sfi0.try_into().unwrap()))
        };
        // D3D_SHADER_REQUIRES_ROVS
        let features = lower([&section[..], &end].concat()).unwrap();
        assert_ne!(features & 0x1000, 0, "SFI0 {features:#x}: no ROVs");
        lower([&section[..], &after, &end].concat()).unwrap();
        // The same accesses outside a section: ordinary views, no feature.
        let plain = [&section[..4], &section[28..35], &end].concat();
        let features = lower(plain).unwrap();
        assert_eq!(features & 0x1000, 0, "SFI0 {features:#x}: ROVs");
        // Without the section's start the wait stays unknown.
        let mut unmarked = [&section[..], &end].concat();
        unmarked.remove(16);
        assert!(lower(unmarked).is_err());
    }

    #[test]
    fn immediate_sampler_slots_distinguish_comparison_uses() {
        let words = [
            0xBE81_0380u32, // s_mov_b32 s1, 0
            0xF40C_0000,
            0xFA00_0000, // s_load_dwordx8 s[0:7], s[0:1], null
            0xBE88_0380, // s_mov_b32 s8, 0
            0xBE89_0380, // s_mov_b32 s9, 0
            0xBE8A_0380, // s_mov_b32 s10, 0
            0xBE8B_0380, // s_mov_b32 s11, 0
            0x7E00_02F0, // v_mov_b32 v0, 0.5 (comparison reference)
            0x7E02_02F0, // v_mov_b32 v1, 0.5 (u)
            0x7E04_02F0, // v_mov_b32 v2, 0.5 (v)
            0xBF8C_0070, // s_waitcnt vmcnt(0) lgkmcnt(0)
            0xF080_0108,
            0x0040_0301, // image_sample v3, v[1:2], s[0:7], s[8:11]
            0xF0A0_0108,
            0x0040_0400, // image_sample_c v4, v[0:2], s[0:7], s[8:11]
            0xF0BC_0108,
            0x0040_0500, // image_sample_c_lz v5, v[0:2], s[0:7], s[8:11]
            0xF080_0108,
            0x0040_0601, // image_sample v6, v[1:2], s[0:7], s[8:11]
            0xBF8C_3F70, // s_waitcnt vmcnt(0)
            0xF800_180F,
            0x0605_0403, // exp mrt0, v3, v4, v5, v6 done vm
            0xBF81_0000,
        ];
        let native = NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            vgprs: 7,
            layout: vec![crate::native::RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 0,
                slot: 0,
            }],
            used: vec![crate::native::UsedResource {
                kind: crate::native::ResourceType::Srv,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 8,
                user_data: 0,
                slot: 0,
            }],
            io: StageIo::Ps {
                inputs: vec![],
                input_addr: 0,
                db_shader_control: 0,
                patches: vec![],
            },
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let output = SigElement {
            name: "SV_Target".into(),
            index: 0,
            kind: 16,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 15,
            half: false,
        };
        let osg1 = container::signature(&[output], false);
        let bytes = crate::dxbc::build(&[(FourCC::OSG1, &osg1)]);
        let c = Container::parse(&bytes).unwrap();
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let out = emit(&c, &native, &p, &st, &io, &res).unwrap();

        let sampler = StaticSampler::from_words([0; 4]);
        let mut comparison = sampler;
        comparison.make_comparison();
        assert_eq!(out.samplers, [[0; 4]; 2]);
        assert_eq!(out.static_samplers, [sampler, comparison]);
        #[cfg(windows)]
        if let Some(path) = super::validator::Validator::default_path() {
            let mut v = super::validator::Validator::load(Some(&path)).unwrap();
            v.validate(&mut out.dxil.clone()).unwrap();
        }
    }

    /// A wave64 compute program that writes only EXEC's high dword
    /// (`s_mov_b32 exec_hi, 0`) before a vector instruction. The new mask was
    /// opaque and keyed by the unchanged low dword, so building its lane bit
    /// needed the mask's own value: lowering recursed until the stack
    /// overflowed, which takes the whole process down (found by
    /// random-input testing). Now that low dword, seen with two high
    /// dwords, has no mask, and the instruction fails like any other that
    /// cannot be expressed: its result is missing, and only a side effect
    /// that uses it would stop the translation.
    #[test]
    fn an_exec_high_write_does_not_overflow_the_stack() {
        // s_mov_b32 exec_hi, 0; v_mov_b32 v1, v0; s_endpgm
        let words = [0xBEFF_0380u32, 0x7E02_0300, 0xBF81_0000];
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: true,
            vgprs: 2,
            layout: vec![],
            used: vec![],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 0,
                threads: [64, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let container = crate::dxbc::build(&[]);
        let c = Container::parse(&container).unwrap();
        if let Err(e) = emit(&c, &native, &p, &st, &io, &res) {
            assert!(e.to_string().contains("its own value"), "{e}");
        }
    }

    /// A wave64 compute program: `body`, then a store of v1 to a UAV.
    fn stores_v1(body: &[u32]) -> Result<(), ShaderError> {
        use crate::native::RootParam;
        use crate::{ResourceType, RootType, UsedResource};
        // s_mov_b32 s1, 0; s_load_dwordx4 s[8:11], s[0:1], 0
        let mut words = vec![0xBE81_0380, 0xF408_0200, 0xFA00_0000];
        words.extend_from_slice(body);
        // v_lshlrev_b32 v3, 2, v0; buffer_store_dword v1, v3, s[8:11], 0 offen;
        // s_endpgm
        words.extend_from_slice(&[0x3406_0082, 0xE070_1000, 0x8002_0103, 0xBF81_0000]);
        let native = NativeProgram {
            stage: HwStage::Cs,
            wave64: true,
            vgprs: 4,
            layout: vec![RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 0,
                slot: 0,
            }],
            used: vec![UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 0,
                slot: 0,
            }],
            io: StageIo::Cs(ComputeIo {
                rsrc2: 2,
                threads: [64, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, None).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let st = structure::structure(&p).unwrap();
        let container = crate::dxbc::build(&[]);
        let c = Container::parse(&container).unwrap();
        emit(&c, &native, &p, &st, &io, &res).map(|_| ())
    }

    /// The high dword of a wave64 mask rewritten alone (`s_mov_b32 vcc_hi`
    /// or `exec_hi`): lanes 32-63 took the mask the low dword was first
    /// written with (the compare's, EXEC's on entry). Now the store refuses;
    /// without the write it translates.
    #[test]
    fn a_mask_whose_high_dword_is_rewritten_is_refused() {
        // v_cmp_eq_u32 vcc, 0, v0; v_mov_b32 v2, 11; v_cndmask_b32 v1, 0, v2, vcc
        let select = [0x7D84_0080, 0x7E04_028B, 0x0202_0480];
        // The same with v_add_co_ci_u32 v1, vcc, 0, v2, vcc.
        let carry = [0x7D84_0080, 0x7E04_028B, 0x5002_0480];
        // v_mov_b32 v1, 7
        let mov = [0x7E02_0287];
        for body in [&select[..], &carry, &mov] {
            stores_v1(body).unwrap();
        }
        // s_mov_b32 vcc_hi, 0 after the compare.
        let select_hi = [0x7D84_0080, 0xBEEB_0380, 0x7E04_028B, 0x0202_0480];
        let carry_hi = [0x7D84_0080, 0xBEEB_0380, 0x7E04_028B, 0x5002_0480];
        // s_mov_b32 exec_hi, 0; v_mov_b32 v1, 7, then the same after
        // s_mov_b64 s[2:3], exec.
        let exec_hi = [0xBEFF_0380, 0x7E02_0287];
        let exec_hi_saved = [0xBE82_047E, 0xBEFF_0380, 0x7E02_0287];
        for body in [&select_hi[..], &carry_hi, &exec_hi, &exec_hi_saved] {
            assert!(stores_v1(body).is_err(), "{body:x?}");
        }
    }

    /// `ds_bpermute_b32` stores nothing: each lane reads the lane its address
    /// names. It stopped as "LDS the program does not allocate" (sticky lips)
    /// or "an LDS operation" (the checkerboard shadow masks).
    #[test]
    fn a_backward_permute_reads_the_lane_its_address_names() {
        // v_mov_b32 v2, 11; ds_bpermute_b32 v1, v0, v2
        stores_v1(&[0x7E04_028B, 0xDACC_0000, 0x0100_0200]).unwrap();
    }

    /// A carry-in reading a join of compares after a divergent branch. A
    /// lane that skipped the inner compare has VCC clear there (VOPC clears
    /// inactive lanes), not the first compare's bit, so the join is no
    /// per-lane phi: lowering refuses it.
    #[test]
    fn a_carry_in_from_a_divergent_join_is_refused() {
        let body = [
            0x7E04_028B, // v_mov_b32 v2, 11
            0x7D84_0080, // v_cmp_eq_u32 vcc, 0, v0
            0xBE82_376A, // s_andn1_saveexec_b64 s[2:3], vcc
            0xBF88_0001, // s_cbranch_execz 1
            0x7D84_0081, // v_cmp_eq_u32 vcc, 1, v0
            0x88FE_027E, // s_or_b64 exec, exec, s[2:3]
            0x5002_0480, // v_add_co_ci_u32 v1, vcc, 0, v2, vcc
        ];
        let e = stores_v1(&body).unwrap_err().to_string();
        assert!(e.contains("a carry-in from an untracked mask"), "{e}");
    }

    /// A pixel shader writing a constant colour.
    #[cfg(windows)]
    fn constant_color() -> Vec<u8> {
        let mut s = Shader::new(true);
        s.outputs.push(SigElement {
            name: "SV_Target".into(),
            index: 0,
            kind: 16,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 15,
            half: false,
        });
        let b = s.entry;
        for (c, v) in [1.0f32, 0.5, 0.25, 1.0].into_iter().enumerate() {
            let args = [
                s.m.i32(0),
                s.m.i32(0),
                s.m.i8(c as u8),
                s.m.f32(v.to_bits()),
            ];
            s.op(b, "storeOutput.f32", Attr::NoUnwind, s.t.void, 5, &args);
        }
        s.m.terminate(b, super::bitcode::Term::Ret);
        s.finish()
    }

    #[test]
    #[cfg(windows)]
    fn stub_pipeline_is_created() {
        let host = match super::host::Host::new(false) {
            Ok(host) => host,
            Err(e) => {
                eprintln!("no SM 6.6 device ({e}); skipped");
                return;
            }
        };
        let ps = super::HostShader {
            stage: crate::HwStage::Ps,
            dxil: constant_color(),
            user_data: Vec::new(),
            samplers: Vec::new(),
            static_samplers: Vec::new(),
            root_cbvs: Vec::new(),
            root_srvs: Vec::new(),
            root_uavs: Vec::new(),
            root_indexed_views: Vec::new(),
        };
        let file = |var: &str, stage| {
            std::env::var(var).ok().map(|p| super::HostShader {
                stage,
                dxil: std::fs::read(p).unwrap(),
                user_data: Vec::new(),
                samplers: Vec::new(),
                static_samplers: Vec::new(),
                root_cbvs: Vec::new(),
                root_srvs: Vec::new(),
                root_uavs: Vec::new(),
                root_indexed_views: Vec::new(),
            })
        };
        let ps = file("VICTORY_REF_PS", crate::HwStage::Ps).unwrap_or(ps);
        let vs = file("VICTORY_REF_VS", crate::HwStage::Vs)
            .unwrap_or_else(|| super::host::stub_vs(&ps).unwrap());
        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut vs.dxil.clone()).unwrap();
        }
        host.pipeline(&vs, &ps).unwrap();
        // The runtime checks the container hash: no hash or a wrong one is
        // refused. (Recent runtimes also take sixteen 0x01 bytes, a bypass
        // value; ours works with older ones too.)
        let mut wrong = ps.dxil[4..20].to_vec();
        wrong[0] ^= 1;
        for digest in [&[0u8; 16][..], &wrong] {
            let mut other = ps.clone();
            other.dxil[4..20].copy_from_slice(digest);
            assert!(
                host.pipeline(&vs, &other).is_err(),
                "digest {digest:02x?} accepted"
            );
        }
    }

    #[test]
    #[cfg(windows)]
    fn constant_color_validates() {
        let mut c = constant_color();
        let path = super::validator::Validator::default_path();
        let Ok(mut v) = super::validator::Validator::load(path.as_deref()) else {
            eprintln!("no dxil.dll; skipped");
            return;
        };
        let ours = c[4..20].to_vec();
        v.validate(&mut c).unwrap();
        assert_eq!(c[4..20], ours[..], "our hash differs from the validator's");
    }

    /// A pixel program of `words` with no user data.
    fn emit_native(words: &[u32], input_addr: u32) -> NativeProgram {
        NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            vgprs: 4,
            layout: vec![],
            used: vec![],
            io: StageIo::Ps {
                inputs: vec![],
                input_addr,
                db_shader_control: 0,
                patches: vec![],
            },
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        }
    }

    /// A pixel program that takes its neighbour's position across its quad
    /// (`ds_swizzle_b32`, as the console's compiler makes derivatives) and
    /// exports the differences in x and y: 1 and -1 in every pixel of a
    /// triangle, those at its edges too, whose neighbours are helpers.
    #[test]
    #[cfg(windows)]
    fn quad_neighbours_at_a_triangles_edge_are_read() {
        use super::host::{Draw, Host, Root};
        use windows::Win32::Graphics::Direct3D12::{
            D3D12_SHADER_VISIBILITY_MESH, D3D12_SHADER_VISIBILITY_PIXEL,
        };
        let _one = super::valu_tests::ONE_AT_A_TIME
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        // v2, v3: position x, y (input address: POS_X and POS_Y after the
        // two perspective barycentrics).
        let swizzle = |dst: u32, src: u32, pattern: u32| {
            [
                0xD800_0000 | 0x35 << 18 | 0x8000 | pattern,
                dst << 24 | src,
            ]
        };
        let mut words = vec![0xBEFC_0380u32]; // s_mov_b32 m0, 0
        words.extend(swizzle(4, 2, 0xB1)); // lanes 1 0 3 2
        words.extend(swizzle(5, 3, 0x4E)); // lanes 2 3 0 1
        words.extend(swizzle(6, 2, 0xA0)); // lanes 0 0 2 2
        words.extend([
            0x0808_0504,  // v_sub_f32 v4, v4, v2
            0x080A_0705,  // v_sub_f32 v5, v5, v3
            0x080C_0506,  // v_sub_f32 v6, v6, v2
            0xF800_180F,
            0x0302_0504, // exp mrt0 v4, v5, v2, v3 done vm
            0xBF81_0000,
        ]);
        let _ = 6;
        let mut native = emit_native(&words, 1 << 1 | 1 << 8 | 1 << 9);
        native.vgprs = 8;
        let out = SigElement {
            name: "SV_Target".into(),
            index: 0,
            kind: 16,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 15,
            half: false,
        };
        let position = SigElement {
            name: "SV_Position".into(),
            index: 0,
            kind: 3,
            comp: 9,
            interp: 4,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 3,
            half: false,
        };
        let isg1 = container::signature(&[position], true);
        let osg1 = container::signature(&[out], false);
        let bytes = crate::dxbc::build(&[(FourCC::ISG1, &isg1), (FourCC::OSG1, &osg1)]);
        let c = Container::parse(&bytes).unwrap();
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let ps = emit(&c, &native, &p, &st, &io, &res).unwrap();
        if let Some(path) = super::validator::Validator::default_path() {
            let mut v = super::validator::Validator::load(Some(&path)).unwrap();
            v.validate(&mut ps.dxil.clone()).unwrap();
        }
        let ms = triangle_mesh_to(TRIANGLE, 1.3).unwrap();
        for (label, warp) in [("hardware", false), ("warp", true)] {
            let Ok(host) = Host::new(warp) else {
                continue;
            };
            let Ok(root) = host.root_signature_for(&[
                (&ms, D3D12_SHADER_VISIBILITY_MESH),
                (&ps, D3D12_SHADER_VISIBILITY_PIXEL),
            ]) else {
                continue;
            };
            let Ok(pso) = host.mesh_pipeline_with(&root, &ms.dxil, &ps.dxil) else {
                eprintln!("no mesh pipeline on {label}; skipped");
                continue;
            };
            let texels = host
                .draw(
                    &root,
                    &pso,
                    &Draw {
                        roots: vec![(0, Root::Constants(vec![0])), (1, Root::Constants(vec![0]))],
                        textures: &[],
                        samplers: &[],
                        vertices: &[],
                        stride: 0,
                        size: (16, 16),
                        mesh_groups: 1,
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: draw: {e}"));
            let lit: Vec<(usize, &[f32; 4])> = texels.iter().enumerate().filter(|t| t.1[2] != 0.0).collect();
            assert!((20..240).contains(&lit.len()), "{label}: {} texels lit", lit.len());
            let bad: Vec<_> = lit
                .iter()
                .filter(|(n, t)| {
                    let x = if n % 2 == 0 { 1.0 } else { -1.0 };
                    let y = if n / 16 % 2 == 0 { 1.0 } else { -1.0 };
                    (t[0], t[1]) != (x, y)
                })
                .collect();
            assert!(bad.is_empty(), "{label}: {} of {} texels: {:?}", bad.len(), lit.len(), bad[0]);
        }
    }

    /// Triangle (0, 1, 2), each corner's 9-bit index in its 10-bit field.
    const TRIANGLE: u32 = 1 << 10 | 2 << 20;
    /// The edge flags of the three corners (bits 9, 19 and 29).
    const EDGE_FLAGS: u32 = 1 << 9 | 1 << 19 | 1 << 29;
    /// The culling bit.
    const CULL: u32 = 1 << 31;

    /// A one-wave (32-lane) triangle mesh program whose lanes all export
    /// primitive `prim` and, as vertex `lane`, the position
    /// (lane & 1 ? 3 : -1, lane & 2 ? 3 : -1, 0.5, 1): triangle (0, 1, 2)
    /// covers the whole target.
    fn triangle_mesh(prim: u32) -> Result<HostShader, ShaderError> {
        triangle_mesh_to(prim, 4.0)
    }

    /// The triangle with its far corners at `reach` - 1.
    fn triangle_mesh_to(prim: u32, reach: f32) -> Result<HostShader, ShaderError> {
        // (the second corner's y comes out as 2 x its lane bit: half of it)
        let [x, y] = [reach.to_bits(), (reach / 2.0).to_bits()];
        let words = [
            0xD765_0001u32, // v_mbcnt_lo_u32_b32 v1, -1, 0
            0x0001_00C1,
            0x3604_0281, // v_and_b32 v2, 1, v1
            0x3606_0282, // v_and_b32 v3, 2, v1
            0x7E04_0D02, // v_cvt_f32_u32 v2, v2
            0x7E06_0D03, // v_cvt_f32_u32 v3, v3
            0x1008_04FF, // v_mul_f32 v4, reach, v2
            x,
            0x0608_08F3, // v_add_f32 v4, -1.0, v4
            0x100A_06FF, // v_mul_f32 v5, reach / 2, v3
            y,
            0x060A_0AF3, // v_add_f32 v5, -1.0, v5
            0x7E0C_02F0, // v_mov_b32 v6, 0.5
            0x7E0E_02F2, // v_mov_b32 v7, 1.0
            0x7E12_02FF, // v_mov_b32 v9, prim
            prim,
            0xF800_0941, // exp prim v9, off, off, off done
            0x0000_0009,
            0xF800_08CF, // exp pos0 v4, v5, v6, v7 done
            0x0706_0504,
            0xBF81_0000, // s_endpgm
        ];
        let native = NativeProgram {
            stage: HwStage::Ms,
            wave64: false,
            vgprs: 10,
            layout: vec![],
            used: vec![],
            io: StageIo::Ms(crate::native::MeshIo {
                outputs: vec![],
                clip_cntl: 0,
                threads: 32,
                max_vertices: 32,
                max_primitives: 32,
                topology: 2,
                lds_bytes: 0,
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let position = SigElement {
            name: "SV_Position".into(),
            index: 0,
            kind: 3,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 0,
            cols: 4,
            usage: 0xF,
            half: false,
        };
        let osg1 = container::signature(&[position], false);
        let bytes = crate::dxbc::build(&[(FourCC::OSG1, &osg1)]);
        let c = Container::parse(&bytes)?;
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs)?;
        let st = structure::structure(&p)?;
        let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
        let io = io::raise(&p, &native)?;
        emit(&c, &native, &p, &st, &io, &res)
    }

    /// A primitive export's corners are 9-bit indices: the edge flag above
    /// each (bits 9, 19, 29) is no part of it. Masking ten bits made a
    /// flagged corner index 512 or more, past any mesh program's vertices
    /// (the draw test below shows the triangle). Flagged, unflagged and
    /// culled exports translate, and dxil.dll, where there is one, takes
    /// each module.
    #[test]
    fn primitive_export_indices_drop_the_edge_flags() {
        #[cfg(windows)]
        let mut validator = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        )
        .ok();
        for prim in [
            TRIANGLE,
            TRIANGLE | EDGE_FLAGS,
            TRIANGLE | EDGE_FLAGS | CULL,
        ] {
            let ms = triangle_mesh(prim).unwrap_or_else(|e| panic!("{prim:#x}: {e}"));
            #[cfg(windows)]
            if let Some(v) = validator.as_mut() {
                v.validate(&mut ms.dxil.clone())
                    .unwrap_or_else(|e| panic!("{prim:#x}: {e}"));
            }
        }
    }

    /// A triangle mesh program on the host GPU and on WARP (where they have
    /// mesh shaders): triangle (0, 1, 2) covers the target with its corners'
    /// edge flags set as without them; with the culling bit, nothing is lit.
    #[test]
    #[cfg(windows)]
    fn edge_flagged_triangles_draw_as_unflagged_ones() {
        use super::host::{Draw, Host, Root};
        use windows::Win32::Graphics::Direct3D12::{
            D3D12_FEATURE_D3D12_OPTIONS7, D3D12_FEATURE_DATA_D3D12_OPTIONS7,
            D3D12_MESH_SHADER_TIER_NOT_SUPPORTED, D3D12_SHADER_VISIBILITY_MESH,
        };
        let _one = super::valu_tests::ONE_AT_A_TIME
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let ps = constant_color();
        let mut ran = 0;
        for (label, warp) in [("hardware", false), ("warp", true)] {
            let Ok(host) = Host::new(warp) else {
                eprintln!("no SM 6.6 {label} device; skipped");
                continue;
            };
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
            // SAFETY: the query writes the structure of the given size.
            let queried = unsafe {
                host.device.CheckFeatureSupport(
                    D3D12_FEATURE_D3D12_OPTIONS7,
                    (&raw mut options).cast(),
                    size_of_val(&options) as u32,
                )
            };
            if queried.is_err() || options.MeshShaderTier == D3D12_MESH_SHADER_TIER_NOT_SUPPORTED {
                eprintln!("no mesh shaders on the {label} device; skipped");
                continue;
            }
            for (prim, culled) in [
                (TRIANGLE, false),
                (TRIANGLE | EDGE_FLAGS, false),
                (TRIANGLE | EDGE_FLAGS | CULL, true),
            ] {
                let ms = triangle_mesh(prim).unwrap();
                let root = host
                    .root_signature_for(&[(&ms, D3D12_SHADER_VISIBILITY_MESH)])
                    .unwrap();
                let pso = host
                    .mesh_pipeline_with(&root, &ms.dxil, &ps)
                    .unwrap_or_else(|e| panic!("{label}: mesh pipeline: {e}"));
                let texels = host
                    .draw(
                        &root,
                        &pso,
                        &Draw {
                            roots: vec![(0, Root::Constants(vec![0]))],
                            textures: &[],
                            samplers: &[],
                            vertices: &[],
                            stride: 0,
                            size: (16, 16),
                            mesh_groups: 1,
                        },
                    )
                    .unwrap_or_else(|e| panic!("{label}: draw: {e}"));
                let lit = texels.iter().filter(|t| t[3] != 0.0).count();
                if culled {
                    assert_eq!(lit, 0, "{label}, {prim:#x}: a culled triangle lit texels");
                    continue;
                }
                assert!(
                    texels.iter().all(|t| *t == [1.0, 0.5, 0.25, 1.0]),
                    "{label}, {prim:#x}: {lit} of 256 texels lit"
                );
            }
            eprintln!("{label}: edge-flagged triangles drew");
            ran += 1;
        }
        if ran == 0 {
            eprintln!("no device with mesh shaders; skipped");
        }
    }

    /// A pixel shader interpolating at sample positions (PERSP_SAMPLE
    /// barycentrics, `SPI_PS_INPUT_ADDR` bit 0) runs at sample frequency:
    /// DXC's PSV0 says so in PSInfo.SampleFrequency (runtime info byte 1)
    /// whenever an input is interpolated per sample.
    #[test]
    fn sample_interpolated_inputs_set_psv_sample_frequency() {
        let words = [
            0xC808_0000u32, // v_interp_p1_f32 v2, v0, attr0.x
            0xC809_0001,    // v_interp_p2_f32 v2, v1, attr0.x
            0xF800_180F,
            0x0202_0202, // exp mrt0, v2, v2, v2, v2 done vm
            0xBF81_0000,
        ];
        let psv_of = |input_addr: u32| {
            let native = super::NativeProgram {
                stage: super::HwStage::Ps,
                wave64: false,
                vgprs: 4,
                layout: vec![],
                used: vec![],
                io: crate::native::StageIo::Ps {
                    inputs: vec![crate::native::PsInput {
                        usage: crate::native::Usage::Generic,
                        row: 0,
                        mask: 0b1111,
                        attr: 0,
                        flat: false,
                        fp16: false,
                        pack2: false,
                        row2: 0,
                    }],
                    input_addr,
                    db_shader_control: 0,
                    patches: vec![],
                },
                code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
            };
            let el = |name: &str, kind, usage| SigElement {
                name: name.into(),
                index: 0,
                kind,
                comp: 9,
                interp: 0,
                row: Some(0),
                col: 0,
                cols: 4,
                usage,
                half: false,
            };
            let isg1 = super::container::signature(&[el("TEXCOORD", 0, 0)], true);
            let osg1 = super::container::signature(&[el("SV_Target", 16, 15)], false);
            let bytes =
                crate::dxbc::build(&[(super::FourCC::ISG1, &isg1), (super::FourCC::OSG1, &osg1)]);
            let c = super::Container::parse(&bytes).unwrap();
            let p = crate::ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
            let st = crate::structure::structure(&p).unwrap();
            let res =
                crate::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
            let io = crate::io::raise(&p, &native).unwrap();
            let out = super::emit(&c, &native, &p, &st, &io, &res).unwrap();
            // dxil.dll, where there is one, checks PSV0 against the module.
            #[cfg(windows)]
            if let Ok(mut v) = super::validator::Validator::load(
                super::validator::Validator::default_path().as_deref(),
            ) {
                v.validate(&mut out.dxil.clone()).unwrap();
            }
            let host = super::Container::parse(&out.dxil).unwrap();
            host.part(super::FourCC::PSV0).unwrap().data.to_vec()
        };
        // Runtime info starts after its size; PSInfo is {DepthOutput, SampleFrequency}.
        let center = psv_of(1 << 1);
        assert_eq!(center[4 + 1], 0, "center interpolation is per pixel");
        let sample = psv_of(1 << 0);
        assert_eq!(sample[4 + 1], 1, "PSInfo.SampleFrequency");
    }

    /// A pixel shader reading channel z of an input whose element starts at
    /// w. Looking the channel up subtracted the element's first channel
    /// before checking the channel lies in it, so in a build with overflow
    /// checks (tests, debug) any read below an element that starts at z or
    /// w panicked, whichever row it was in (found by random-input testing).
    #[test]
    fn reading_a_channel_below_an_input_element_does_not_underflow() {
        // v_interp_mov_f32 v3, p0, attr0.z; s_endpgm
        let words = [0xC80E_0202u32, 0xBF81_0000];
        let native = super::NativeProgram {
            stage: super::HwStage::Ps,
            wave64: false,
            vgprs: 4,
            layout: vec![],
            used: vec![],
            io: crate::native::StageIo::Ps {
                inputs: vec![crate::native::PsInput {
                    usage: crate::native::Usage::Generic,
                    row: 0,
                    mask: 0b1000,
                    attr: 0,
                    flat: false,
                    fp16: false,
                    pack2: false,
                    row2: 0,
                }],
                input_addr: 0,
                db_shader_control: 0,
                patches: vec![],
            },
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let w_only = SigElement {
            name: "TEXCOORD".into(),
            index: 0,
            kind: 0,
            comp: 9,
            interp: 0,
            row: Some(0),
            col: 3,
            cols: 1,
            usage: 0,
            half: false,
        };
        let isg1 = super::container::signature(&[w_only], true);
        let bytes = crate::dxbc::build(&[(super::FourCC::ISG1, &isg1)]);
        let c = super::Container::parse(&bytes).unwrap();
        let p = crate::ir::build(&native.patched_code(), native.wave64, native.vgprs).unwrap();
        let st = crate::structure::structure(&p).unwrap();
        let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = crate::io::raise(&p, &native).unwrap();
        // Translated or refused with a reason; it must not panic.
        let _ = super::emit(&c, &native, &p, &st, &io, &res);
    }

    /// `s_buffer_load_dwordx4`, `x8` and `x16` from a raw SRV read the dwords
    /// the console's range-checked load reads, a call a dword or
    /// (`VICTORY_SHADER_OPT` with 64) a call for every four: also where a
    /// call's four dwords lie across the view's end (the dwords out of it
    /// are 0, the others are read). On the software adapter only: run the
    /// test once plain and once with `VICTORY_SHADER_OPT=127`.
    #[test]
    #[cfg(windows)]
    fn scalar_buffer_loads_read_the_same_dwords_one_call_or_four() {
        // (first register, dwords) of the three loads, in order.
        const LOADS: [(u32, u32); 3] = [(16, 8), (24, 4), (32, 16)];
        let mut words: Vec<u32> = vec![
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0 (the SRV)
            0xF408_0302,
            0xFA00_0020, // s_load_dwordx4 s[12:15], s[4:5], 0x20 (the UAV)
            0xF42C_0404,
            0x0000_0000, // s_buffer_load_dwordx8 s[16:23], s[8:11], s0
            0xF428_0604,
            0xFA00_0008, // s_buffer_load_dwordx4 s[24:27], s[8:11], 0x8
            0xF430_0804,
            0x0200_0004, // s_buffer_load_dwordx16 s[32:47], s[8:11], s1, 0x4
        ];
        let mut out_dword = 0u32;
        for (first, n) in LOADS {
            for k in 0..n {
                // v_mov_b32 v1, s<first + k>
                words.push(0x7E00_0000 | 1 << 17 | 0x001 << 9 | (first + k));
                // buffer_store_dword v1, s[12:15], offset 4 * out_dword
                words.push(0x38 << 26 | 0x1C << 18 | (4 * out_dword));
                words.push(0x80 << 24 | (12 >> 2) << 16 | 1 << 8);
                out_dword += 1;
            }
        }
        words.push(0xBF81_0000); // s_endpgm
        let table = |kind, at| crate::UsedResource {
            kind,
            root: crate::RootType::Table,
            table_offset_dwords: at,
            lower_bound: 0,
            stride: 4,
            dwords: 4,
            user_data: 4,
            slot: 1,
        };
        let native = super::NativeProgram {
            stage: super::HwStage::Cs,
            wave64: false,
            vgprs: 4,
            layout: vec![
                crate::RootParam {
                    root: crate::RootType::Constants,
                    table_offset_dwords: 0,
                    sgprs: 4,
                    base_register: 0,
                    user_data: 0,
                    slot: 0,
                },
                crate::RootParam {
                    root: crate::RootType::Table,
                    table_offset_dwords: 0,
                    sgprs: 1,
                    base_register: 0,
                    user_data: 4,
                    slot: 1,
                },
            ],
            used: vec![table(crate::ResourceType::Srv, 0), table(crate::ResourceType::Uav, 8)],
            io: crate::native::StageIo::Cs(crate::native::ComputeIo {
                rsrc2: 5 << 1,
                threads: [1, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = crate::ir::build(&native.code, native.wave64, native.vgprs).unwrap();
        let res = crate::resource::track(&p, &native.layout, native.stage, None).unwrap();
        let io = crate::io::raise(&p, &native).unwrap();
        let st = crate::structure::structure(&p).unwrap();
        let l = super::lower::lower(&super::lower::Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: Vec::new(),
            outputs: Vec::new(),
            prims: Vec::new(),
            mesh_stride: 0,
        })
        .unwrap();
        // The calls that read the SRV: 28 a dword, 7 with four a call.
        let wide = shader_opt() & super::bitcode::opt::WIDE != 0;
        let mut dxil = l.dxil;
        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut dxil).unwrap();
        }
        let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        let Ok(host) = super::host::Host::new(true) else {
            eprintln!("no software adapter; the dispatch is skipped");
            return;
        };
        let cs = super::HostShader {
            stage: super::HwStage::Cs,
            dxil,
            user_data: l.user_data,
            samplers: Vec::new(),
            static_samplers: Vec::new(),
            root_cbvs: l.root_cbvs,
            root_srvs: l.root_srvs,
            root_uavs: l.root_uavs,
            root_indexed_views: Vec::new(),
        };
        let (root, pso) = host.compute_pipeline(&cs).unwrap();
        // Twelve dwords: the loads below run across the view's end.
        let data: Vec<u32> = (0..12u32).map(|i| 0x1111_1111u32.wrapping_mul(i + 1) ^ 0x8000_0000).collect();
        let dword = |byte: u64| -> u32 {
            if byte.is_multiple_of(4) && byte + 4 <= 4 * data.len() as u64 { data[(byte / 4) as usize] } else { 0 }
        };
        for (s0, s1) in [(0u32, 0u32), (16, 0), (24, 4), (40, 28), (48, 44), (8, 12)] {
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &super::host::Dispatch {
                        roots: vec![(0, super::host::Root::Constants(vec![s0, s1, 0, 0, 0]))],
                        views: &[
                            super::host::View::Buffer {
                                data: &data,
                                stride: super::host::Stride::Raw,
                            },
                            super::host::View::RawUav(32),
                        ],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap();
            let starts = [u64::from(s0), 8, u64::from(s1) + 4];
            let mut want = Vec::new();
            for (&(_, n), start) in LOADS.iter().zip(starts) {
                want.extend((0..u64::from(n)).map(|k| dword(start + 4 * k)));
            }
            eprintln!("offsets {s0} and {s1}, four dwords a call: {wide}: {:x?}", &out[0][..28]);
            assert_eq!(out[0][..28], want[..], "offsets {s0} and {s1}, four dwords a call: {wide}");
        }
    }

    /// Renderer port 5 (`VICTORY_PORT_CBV=1`): a table's constant buffer is
    /// read as raw bytes through the raw twin of its slot, at a constant and
    /// at a computed offset, and gives the dwords the constant buffer read
    /// gives without the switch; the slot itself (a constant buffer view, of
    /// other bytes here) is not read.
    #[test]
    #[cfg(windows)]
    fn table_constant_buffers_read_as_raw_bytes_through_their_twins() {
        let mut words: Vec<u32> = vec![
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0 (the constant buffer)
            0xF408_0302,
            0xFA00_0020, // s_load_dwordx4 s[12:15], s[4:5], 0x20 (the UAV)
            0xF428_0404,
            0xFA00_0010, // s_buffer_load_dwordx4 s[16:19], s[8:11], 0x10
            0xF420_0504,
            0x0000_0000, // s_buffer_load_dword s20, s[8:11], s0
        ];
        for (out, s) in (16..21u32).enumerate() {
            // v_mov_b32 v1, s<s>; buffer_store_dword v1, s[12:15], offset 4 * out
            words.push(0x7E00_0000 | 1 << 17 | 0x001 << 9 | s);
            words.push(0x38 << 26 | 0x1C << 18 | (4 * out as u32));
            words.push(0x80 << 24 | (12 >> 2) << 16 | 1 << 8);
        }
        words.push(0xBF81_0000); // s_endpgm
        let table = |kind, at| crate::UsedResource {
            kind,
            root: crate::RootType::Table,
            table_offset_dwords: at,
            lower_bound: 0,
            stride: 4,
            dwords: 4,
            user_data: 4,
            slot: 1,
        };
        let native = super::NativeProgram {
            stage: super::HwStage::Cs,
            wave64: false,
            vgprs: 4,
            layout: vec![
                crate::RootParam { root: crate::RootType::Constants, table_offset_dwords: 0, sgprs: 4, base_register: 0, user_data: 0, slot: 0 },
                crate::RootParam { root: crate::RootType::Table, table_offset_dwords: 0, sgprs: 1, base_register: 0, user_data: 4, slot: 1 },
            ],
            used: vec![table(crate::ResourceType::Cbv, 0), table(crate::ResourceType::Uav, 8)],
            io: crate::native::StageIo::Cs(crate::native::ComputeIo { rsrc2: 5 << 1, threads: [1, 1, 1] }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        // The twins start at slot 2: the constant buffer's (slot 0) is 2, the UAV's (1) 3.
        const TWINS: u32 = 2;
        let lowered = |raw: bool| {
            super::shader::CONSTANTS_RAW_HERE.set(Some(raw));
            super::shader::RAW_VIEWS_HERE.set(Some(if raw { TWINS } else { 0 }));
            let p = crate::ir::build(&native.code, native.wave64, native.vgprs).unwrap();
            let res = crate::resource::track(&p, &native.layout, native.stage, None).unwrap();
            let io = crate::io::raise(&p, &native).unwrap();
            let st = crate::structure::structure(&p).unwrap();
            let l = super::lower::lower(&super::lower::Parts {
                native: &native,
                p: &p,
                st: &st,
                io: &io,
                res: &res,
                inputs: Vec::new(),
                outputs: Vec::new(),
                prims: Vec::new(),
                mesh_stride: 0,
            })
            .unwrap();
            super::shader::CONSTANTS_RAW_HERE.set(None);
            super::shader::RAW_VIEWS_HERE.set(None);
            l
        };
        let (before, after) = (lowered(false), lowered(true));
        assert_ne!(before.dxil, after.dxil, "the switch changes the translation");
        let data: Vec<u32> = (0..16u32).map(|i| 0x0101_0101u32.wrapping_mul(i + 3) ^ 0x4000_0000).collect();
        let other: Vec<u32> = vec![0xdead_beef; 64];
        let s0 = 8u32;
        let want: Vec<u32> = vec![data[4], data[5], data[6], data[7], data[(s0 / 4) as usize]];
        let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        for (raw, l) in [(false, before), (true, after)] {
            let mut dxil = l.dxil;
            if let Ok(mut v) = super::validator::Validator::load(super::validator::Validator::default_path().as_deref()) {
                v.validate(&mut dxil).unwrap();
            }
            let Ok(host) = super::host::Host::new(true) else {
                eprintln!("no software adapter; the dispatch is skipped");
                return;
            };
            let cs = super::HostShader {
                stage: super::HwStage::Cs,
                dxil,
                user_data: l.user_data,
                samplers: Vec::new(),
                static_samplers: Vec::new(),
                root_cbvs: l.root_cbvs,
                root_srvs: l.root_srvs,
                root_uavs: l.root_uavs,
                root_indexed_views: Vec::new(),
            };
            let (root, pso) = host.compute_pipeline(&cs).unwrap();
            let views: Vec<super::host::View> = if raw {
                vec![
                    super::host::View::Cbv(&other),
                    super::host::View::RawUav(8),
                    super::host::View::Buffer { data: &data, stride: super::host::Stride::Raw },
                    super::host::View::RawUav(8),
                ]
            } else {
                vec![super::host::View::Cbv(&data), super::host::View::RawUav(8)]
            };
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &super::host::Dispatch {
                        roots: vec![(0, super::host::Root::Constants(vec![s0, 0, 0, 0, 0]))],
                        views: &views,
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap();
            // The UAV is written where the program reads it: its twin with the switch.
            let got = out.iter().find(|o| o[..5].iter().any(|&w| w != 0)).map(|o| o[..5].to_vec());
            assert_eq!(got.as_deref(), Some(&want[..]), "switch {raw}: {out:x?}");
        }
    }

    #[test]
    fn valu_div_f32_sequence_executes_on_gpu() {
        let words: &[u32] = &[
            0xBE85_0380, // s_mov_b32 s5, 0
            0xF408_0202,
            0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], 0x0
            0xF408_0302,
            0xFA00_0020, // s_load_dwordx4 s[12:15], s[4:5], 0x20
            0x340E_0083, // v_lshlrev_b32 v7, 3, v0
            0xE034_1000,
            0x8002_0107, // buffer_load_dwordx2 v[1:2], v7, s[8:11], 0 offen
            0xD56D_6A03,
            0x0406_0502, // v_div_scale_f32 v3, vcc_lo, v2, v2, v1
            0xD56D_1008,
            0x0406_0501, // v_div_scale_f32 v8, s16, v1, v2, v1
            0x7E08_5503, // v_rcp_f32 v4, v3
            0xD54B_0005,
            0x23CA_0903, // v_fma_f32 v5, -v3, v4, 1.0
            0xD54B_0004,
            0x0412_0905, // v_fma_f32 v4, v5, v4, v4
            0x100A_0908, // v_mul_f32 v5, v8, v4
            0xD54B_0006,
            0x2422_0B03, // v_fma_f32 v6, -v3, v5, v8
            0xD56F_0004,
            0x0416_0906, // v_div_fmas_f32 v4, v6, v4, v5
            0xD55F_0001,
            0x0406_0504, // v_div_fixup_f32 v1, v4, v2, v1
            0x340E_0082, // v_lshlrev_b32 v7, 2, v0
            0xE070_1000,
            0x8003_0107, // buffer_store_dword v1, v7, s[12:15], 0 offen
            0xBF81_0000, // s_endpgm
        ];
        let native = super::NativeProgram {
            stage: super::HwStage::Cs,
            wave64: false,
            vgprs: 9,
            layout: vec![
                crate::RootParam {
                    root: crate::RootType::Constants,
                    table_offset_dwords: 0,
                    sgprs: 4,
                    base_register: 0,
                    user_data: 0,
                    slot: 0,
                },
                crate::RootParam {
                    root: crate::RootType::Table,
                    table_offset_dwords: 0,
                    sgprs: 1,
                    base_register: 0,
                    user_data: 4,
                    slot: 1,
                },
            ],
            used: vec![
                crate::UsedResource {
                    kind: crate::ResourceType::Srv,
                    root: crate::RootType::Table,
                    table_offset_dwords: 0,
                    lower_bound: 0,
                    stride: 4,
                    dwords: 4,
                    user_data: 4,
                    slot: 1,
                },
                crate::UsedResource {
                    kind: crate::ResourceType::Uav,
                    root: crate::RootType::Table,
                    table_offset_dwords: 8,
                    lower_bound: 0,
                    stride: 4,
                    dwords: 4,
                    user_data: 4,
                    slot: 1,
                },
            ],
            io: crate::native::StageIo::Cs(crate::native::ComputeIo {
                rsrc2: 5 << 1,
                threads: [7, 1, 1],
            }),
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let p = crate::ir::build(&native.code, native.wave64, native.vgprs).unwrap();
        let res = crate::resource::track(&p, &native.layout, native.stage, None).unwrap();
        let io = crate::io::raise(&p, &native).unwrap();
        let st = crate::structure::structure(&p).unwrap();
        let l = super::lower::lower(&super::lower::Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: Vec::new(),
            outputs: Vec::new(),
            prims: Vec::new(),
            mesh_stride: 0,
        })
        .unwrap();
        let mut dxil = l.dxil;
        #[cfg(windows)]
        if let Ok(mut v) = super::validator::Validator::load(
            super::validator::Validator::default_path().as_deref(),
        ) {
            v.validate(&mut dxil).unwrap();
        }
        #[cfg(windows)]
        if let Ok(host) = super::host::Host::new(false) {
            let cs = super::HostShader {
                stage: super::HwStage::Cs,
                dxil,
                user_data: l.user_data,
                samplers: Vec::new(),
                static_samplers: Vec::new(),
                root_cbvs: l.root_cbvs,
                root_srvs: l.root_srvs,
                root_uavs: l.root_uavs,
                root_indexed_views: Vec::new(),
            };
            let (root, pso) = host.compute_pipeline(&cs).unwrap();
            let inputs: Vec<u32> = [
                7.0f32, 3.0, -1.0, 1e-40, 1.0, 1e35, 1e-35, 1e-36, 0.0, 0.0, -1.0, 0.0, -1e35, 1e35,
            ]
            .into_iter()
            .map(f32::to_bits)
            .collect();
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &super::host::Dispatch {
                        roots: vec![(0, super::host::Root::Constants(vec![0, 0, 0, 0, 0]))],
                        views: &[
                            super::host::View::Buffer {
                                data: &inputs,
                                stride: super::host::Stride::Raw,
                            },
                            super::host::View::RawUav(64),
                        ],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap();
            let got: Vec<f32> = out[0][..7].iter().copied().map(f32::from_bits).collect();
            assert!((got[0] - (7.0f32 / 3.0)).abs() < 1e-6, "got {:?}", got);
            assert_eq!(out[0][1], (-1.0f32 / 1e-40).to_bits());
            assert!(
                (got[2] - 1e-35).abs() <= 1e-40,
                "lane 2 got {:?} ({:#x})",
                got[2],
                out[0][2]
            );
            assert!(
                (got[3] - 10.0).abs() < 1e-5,
                "lane 3 got {:?} ({:#x})",
                got[3],
                out[0][3]
            );
            assert_eq!(out[0][4], 0xFFC0_0000);
            assert_eq!(out[0][5], (-1.0f32 / 0.0).to_bits());
            assert!(
                (got[6] - (-1.0)).abs() < 1e-6,
                "lane 6 got {:?} ({:#x})",
                got[6],
                out[0][6]
            );
        }
    }
}

#[cfg(test)]
mod atomic_tests {
    use super::*;
    use crate::native::{ComputeIo, RootParam, StageIo};

    #[test]
    fn heap_texture_atomic_widths() {
        // A synthetic compute shader loads a 2D UAV from a root table,
        // compares or adds at (0, 0), then uses the returned register(s).
        for (opcode, mask, wide) in [
            (16u32, 3, false),
            (16, 15, true),
            (17, 1, false),
            (17, 3, true),
        ] {
            let words = [
                0xBE81_0380, // s_mov_b32 s1, 0 (table address high)
                0xF40C_0200,
                0xFA00_0000, // s_load_dwordx8 s[8:15], s[0:1], 0
                0x7E00_0280,
                0x7E02_0280, // v0, v1 = coordinates
                0x7E04_0281,
                0x7E06_0282,
                0x7E08_0283,
                0x7E0A_0284, // v2..v5 = 1..4
                0xF000_2008 | opcode << 18 | mask << 8,
                0x0002_0200,
                // Use the old value in a second atomic at the same location.
                0xF044_0008 | (if wide { 3 } else { 1 }) << 8,
                0x0002_0200,
                0xBF81_0000,
            ];
            let native = NativeProgram {
                stage: HwStage::Cs,
                wave64: false,
                vgprs: 6,
                layout: vec![RootParam {
                    root: RootType::Table,
                    table_offset_dwords: 0,
                    sgprs: 1,
                    base_register: 0,
                    user_data: 0,
                    slot: 0,
                }],
                used: vec![],
                io: StageIo::Cs(ComputeIo {
                    rsrc2: 2,
                    threads: [32, 1, 1],
                }),
                code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
            };
            let p = ir::build(&native.code, false, native.vgprs).unwrap();
            let bytes = crate::dxbc::build(&[]);
            let shader = emit(
                &Container::parse(&bytes).unwrap(),
                &native,
                &p,
                &structure::structure(&p).unwrap(),
                &io::raise(&p, &native).unwrap(),
                &resource::track(&p, &native.layout, native.stage, None).unwrap(),
            )
            .unwrap();
            let container = Container::parse(&shader.dxil).unwrap();
            let features = u64::from_le_bytes(
                container
                    .part(FourCC::SFI0)
                    .unwrap()
                    .data
                    .try_into()
                    .unwrap(),
            );
            let required = 0x1040_8000; // heap atomics, typed atomics, Int64Ops
            assert_eq!(features & required, if wide { required } else { 0 });
            #[cfg(windows)]
            if let Ok(mut v) =
                validator::Validator::load(validator::Validator::default_path().as_deref())
            {
                v.validate(&mut shader.dxil.clone()).unwrap();
            }
        }
    }
}

#[cfg(test)]
mod barycentric_tests {
    use super::container::SigElement;
    use super::*;
    use crate::native::{PsInput, StageIo, Usage};

    /// `SPI_PS_INPUT_ADDR` bits: PERSP_CENTER, LINEAR_CENTER.
    const PERSP_CENTER: u32 = 1 << 1;
    const LINEAR_CENTER: u32 = 1 << 5;

    /// `exp mrt0, v<r>, v<r>, v<r>, v<r> done vm`.
    fn exp_mrt0(r: u32) -> [u32; 2] {
        [0xF800_180F, r * 0x0101_0101]
    }

    /// VOP2 `v_add_f32 v<dst>, v<a>, v<b>`.
    fn v_add_f32(dst: u32, a: u32, b: u32) -> u32 {
        3 << 25 | dst << 17 | b << 9 | (256 + a)
    }

    fn element(name: &str, kind: u8, row: u32) -> SigElement {
        SigElement {
            name: name.into(),
            index: 0,
            kind,
            comp: 9,
            interp: 0,
            row: Some(row),
            col: 0,
            cols: 4,
            usage: 0,
            half: false,
        }
    }

    /// Translates a pixel program with the input VGPRs `input_addr`
    /// enables and one float4 TEXCOORD0 attribute (attr 0, row 0).
    fn pixel(words: &[u32], input_addr: u32) -> Result<HostShader, ShaderError> {
        pixel_with_inputs(words, input_addr, &[element("TEXCOORD", 0, 0)])
    }

    fn pixel_with_inputs(words: &[u32], input_addr: u32, inputs: &[container::SigElement]) -> Result<HostShader, ShaderError> {
        let native = NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            vgprs: 8,
            layout: vec![],
            used: vec![],
            io: StageIo::Ps {
                inputs: vec![PsInput {
                    usage: Usage::Generic,
                    row: 0,
                    mask: 0b1111,
                    attr: 0,
                    flat: false,
                    fp16: false,
                    pack2: false,
                    row2: 0,
                }],
                input_addr,
                db_shader_control: 0,
                patches: vec![],
            },
            code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
        };
        let mut target = element("SV_Target", 16, 0);
        target.usage = 15;
        let isg1 = container::signature(inputs, true);
        let osg1 = container::signature(&[target], false);
        let bytes = crate::dxbc::build(&[(FourCC::ISG1, &isg1), (FourCC::OSG1, &osg1)]);
        let c = Container::parse(&bytes).unwrap();
        let p = ir::build(&native.patched_code(), native.wave64, native.vgprs)?;
        let res = resource::track(&p, &native.layout, native.stage, None)?;
        let io = io::raise(&p, &native)?;
        let st = structure::structure(&p)?;
        emit(&c, &native, &p, &st, &io, &res)
    }

    #[test]
    fn an_unused_packed_input_matches_its_provoking_vertex_sibling() {
        let [e0, e1] = exp_mrt0(3);
        // v_interp_mov_f32 v3, p0, attr0.z followed by exp mrt0.
        let words = [0xC80E_0202, e0, e1, 0xBF81_0000];
        let xy = container::SigElement { cols: 2, ..element("TEXCOORD", 0, 0) };
        let zw = container::SigElement { col: 2, cols: 2, ..element("NORMAL", 0, 0) };
        let shader = pixel_with_inputs(&words, PERSP_CENTER, &[xy, zw]).unwrap();
        let c = Container::parse(&shader.dxil).unwrap();
        let psv = c.part(FourCC::PSV0).unwrap().data;
        for columns in [0x42, 0x62] {
            let record = [1, 0, columns, 0, 3, 1];
            assert!(psv.windows(record.len()).any(|w| w == record));
        }
        let inputs = c.part(FourCC::ISG1).unwrap().data;
        assert_eq!((inputs[8 + 25], inputs[8 + 32 + 25]), (0, 4));
        validate(&shader);
    }

    /// A pixel program's first-lane loop (no table entry in it, so not a waterfall the tracker
    /// knows) asks for helper lanes, which is valid DXIL (upstream #946's program).
    #[test]
    fn a_pixel_first_lane_loop_serves_helper_lanes() {
        let shader = pixel(
            &[
                0xD765_0000, 0x0001_00C1, // v_mbcnt_lo_u32_b32 v0, -1, 0
                0x7E02_0280, // v_mov_b32 v1, 0
                0xBE97_037E, // s_mov_b32 s23, exec_lo
                0x7E18_0500, // v_readfirstlane_b32 s12, v0
                0x7D84_000C, // v_cmp_eq_u32 vcc, s12, v0
                0xBE95_3C6A, // s_and_saveexec_b32 s21, vcc_lo
                0xBF80_0000, // s_nop 0
                0x7E02_02FF, 0x3F80_0000, // v_mov_b32 v1, 1.0
                0x897E_157E, // s_xor_b32 exec_lo, exec_lo, s21
                0xBF89_FFF8, // s_cbranch_execnz loop
                0xBEFE_0317, // s_mov_b32 exec_lo, s23
                0xF800_180F, 0x0101_0101, // exp mrt0 v1 v1 v1 v1 done vm
                0xBF81_0000, // s_endpgm
            ],
            0,
        )
        .unwrap();
        validate(&shader);
    }

    /// The translation's `SFI0` features and `ISG1` rows.
    fn parts(shader: &HostShader) -> (u64, Vec<SigElement>) {
        let c = Container::parse(&shader.dxil).unwrap();
        let sfi0 = c.part(FourCC::SFI0).unwrap().data.try_into().unwrap();
        (
            u64::from_le_bytes(sfi0),
            signature(c.part(FourCC::ISG1).unwrap().data).unwrap(),
        )
    }

    fn validate(shader: &HostShader) {
        #[cfg(windows)]
        if let Ok(mut v) =
            validator::Validator::load(validator::Validator::default_path().as_deref())
        {
            v.validate(&mut shader.dxil.clone()).unwrap();
        }
        let _ = shader;
    }

    /// Custom interpolation reads I and J (v0, v1 with PERSP_CENTER) as
    /// plain floats: they are channels y and z of `SV_Barycentrics`, the
    /// weights of vertices 1 and 2.
    #[test]
    fn barycentrics_read_as_values_are_sv_barycentrics() {
        let [e0, e1] = exp_mrt0(2);
        let words = [v_add_f32(2, 0, 1), e0, e1, 0xBF81_0000];
        let shader = pixel(&words, PERSP_CENTER).unwrap();
        let (features, inputs) = parts(&shader);
        assert_ne!(features & 0x2_0000, 0, "no Barycentrics feature");
        let bary: Vec<_> = inputs.iter().filter(|e| e.kind == 28).collect();
        assert_eq!(bary.len(), 1, "{inputs:?}");
        assert_eq!((bary[0].index, bary[0].col, bary[0].cols), (0, 0, 3));
        // Not packed: the validator wants packing location -1.
        assert_eq!(bary[0].row, None);
        // Only y and z (I and J) are read.
        let isg1 = Container::parse(&shader.dxil).unwrap();
        let isg1 = isg1.part(FourCC::ISG1).unwrap().data;
        let rw: Vec<u8> = (0..2).map(|k| isg1[8 + 32 * k + 25]).collect();
        assert!(rw.contains(&0b110), "{rw:?}");
        validate(&shader);
    }

    /// Two kinds of barycentrics read directly: one element per mode.
    #[test]
    fn two_barycentric_modes_get_two_elements() {
        // v0-v1 perspective i, j; v2-v3 linear i, j.
        let [e0, e1] = exp_mrt0(4);
        let words = [v_add_f32(4, 0, 3), e0, e1, 0xBF81_0000];
        let shader = pixel(&words, PERSP_CENTER | LINEAR_CENTER).unwrap();
        let (_, inputs) = parts(&shader);
        let mut indices: Vec<u32> = inputs
            .iter()
            .filter(|e| e.kind == 28)
            .map(|e| e.index)
            .collect();
        indices.sort();
        assert_eq!(indices, [0, 1], "{inputs:?}");
        validate(&shader);
    }

    /// I and J only fed to `v_interp_*` stay host interpolation: no
    /// barycentrics are declared.
    #[test]
    fn interpolation_alone_declares_no_barycentrics() {
        let [e0, e1] = exp_mrt0(2);
        let words = [
            0xC808_0000, // v_interp_p1_f32 v2, v0, attr0.x
            0xC809_0001, // v_interp_p2_f32 v2, v1, attr0.x
            e0,
            e1,
            0xBF81_0000,
        ];
        let shader = pixel(&words, PERSP_CENTER).unwrap();
        let (features, inputs) = parts(&shader);
        assert_eq!(features & 0x2_0000, 0);
        assert!(inputs.iter().all(|e| e.kind != 28), "{inputs:?}");
        validate(&shader);
    }

    #[test]
    fn a_standalone_first_half_reads_the_vertices() {
        let shader = first_half_pixel(false);
        let (_, inputs) = parts(&shader);
        assert_eq!(inputs.iter().filter(|e| e.kind == 28).count(), 1);
        assert_eq!(texcoord_mode(&shader), 1);
        validate(&shader);
    }

    #[test]
    fn a_first_half_read_beside_its_pair_keeps_its_partial_value() {
        let shader = first_half_pixel(true);
        assert_eq!(texcoord_mode(&shader), 1);
        validate(&shader);
    }

    fn first_half_pixel(paired: bool) -> HostShader {
        let [e0, e1] = exp_mrt0(2);
        let mut words = vec![0xC808_0000]; // v_interp_p1_f32 v2, v0, attr0.x
        if paired {
            words.extend([
                0x7E06_0302, // v_mov_b32 v3, v2 (keep the partial value)
                0xC809_0001, // v_interp_p2_f32 v2, v1, attr0.x
                v_add_f32(2, 2, 3),
            ]);
        }
        words.extend([e0, e1, 0xBF81_0000]);
        pixel(&words, PERSP_CENTER).unwrap()
    }

    #[test]
    #[cfg(windows)]
    fn first_half_values_render_on_hardware_and_warp() {
        use super::bitcode::{Attr, Term};
        use super::host::{Draw, Host};
        use super::shader::Shader;
        let _one = super::valu_tests::ONE_AT_A_TIME
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        // Pass POSITION and TEXCOORD through, with TEXCOORD in the same
        // output row as the translated pixel shader's input signature.
        let mut s = Shader::new(false);
        s.inputs = vec![element("POSITION", 0, 0), element("TEXCOORD", 0, 1)];
        for e in &mut s.inputs {
            e.usage = 15;
        }
        s.outputs = vec![element("TEXCOORD", 0, 0), element("SV_Position", 3, 1)];
        for (i, e) in s.outputs.iter_mut().enumerate() {
            e.usage = 15;
            e.interp = if i == 1 { 4 } else { 2 };
        }
        for row in 0..2 {
            for col in 0..4 {
                let args = [s.m.i32(row), s.m.i32(0), s.m.i8(col), s.m.undef(s.t.i32)];
                let v = s.op(s.entry, "loadInput.f32", Attr::ReadNone, s.t.f32, 4, &args);
                let args = [s.m.i32(1 - row), s.m.i32(0), s.m.i8(col), v];
                s.op(
                    s.entry,
                    "storeOutput.f32",
                    Attr::NoUnwind,
                    s.t.void,
                    5,
                    &args,
                );
            }
        }
        s.m.terminate(s.entry, Term::Ret);
        let vs = HostShader {
            stage: HwStage::Vs,
            dxil: s.finish(),
            user_data: Vec::new(),
            samplers: Vec::new(),
            static_samplers: Vec::new(),
            root_cbvs: Vec::new(),
            root_srvs: Vec::new(),
            root_uavs: Vec::new(),
            root_indexed_views: Vec::new(),
        };
        validate(&vs);
        // One triangle covers an 8x8 target, with attribute values 2, 6, 10.
        let vertices = [
            -1.0, 1.0, 0.5, 1.0, 2.0, 0.0, 0.0, 0.0, -1.0, -3.0, 0.5, 1.0, 6.0, 0.0, 0.0, 0.0, 3.0,
            1.0, 0.5, 1.0, 10.0, 0.0, 0.0, 0.0,
        ];
        let mut ran = 0;
        for (backend, warp) in [("hardware", false), ("WARP", true)] {
            let Ok(host) = Host::new(warp) else { continue };
            let mut options =
                windows::Win32::Graphics::Direct3D12::D3D12_FEATURE_DATA_D3D12_OPTIONS3::default();
            // SAFETY: a correctly sized output for the feature query.
            unsafe {
                host.device
                    .CheckFeatureSupport(
                        windows::Win32::Graphics::Direct3D12::D3D12_FEATURE_D3D12_OPTIONS3,
                        (&raw mut options).cast(),
                        size_of_val(&options) as u32,
                    )
                    .unwrap();
            }
            if !options.BarycentricsSupported.as_bool() {
                eprintln!("no barycentrics on {backend}; skipped");
                continue;
            }
            for paired in [false, true] {
                let ps = first_half_pixel(paired);
                validate(&ps);
                let (root, pso) = host.pipeline(&vs, &ps).expect("interpolation pipeline");
                let out = host
                    .draw(
                        &root,
                        &pso,
                        &Draw {
                            roots: Vec::new(),
                            textures: &[],
                            samplers: &[],
                            vertices: &vertices,
                            stride: 32,
                            size: (8, 8),
                            mesh_groups: 0,
                        },
                    )
                    .expect("draw partial interpolation");
                for (n, texel) in out.iter().enumerate() {
                    let i = (n / 8) as f32 / 16.0 + 0.5 / 16.0;
                    let j = (n % 8) as f32 / 16.0 + 0.5 / 16.0;
                    let partial = 2.0 + i * 4.0;
                    let want = if paired {
                        partial + 2.0 + i * 4.0 + j * 8.0
                    } else {
                        partial
                    };
                    for &got in texel {
                        assert!(
                            (got - want).abs() < 1e-5,
                            "{backend} paired={paired} pixel={n}: got {got}, want {want}"
                        );
                    }
                }
            }
            ran += 1;
            eprintln!("partial interpolation renders correctly on {backend}");
        }
        if ran == 0 {
            eprintln!("no SM 6.6 host with barycentrics; GPU checks skipped");
        }
    }

    /// Both: the attribute is interpolated and J is used as a value.
    #[test]
    fn interpolation_and_a_barycentric_value_together() {
        let [e0, e1] = exp_mrt0(3);
        let words = [
            0xC808_0000,        // v_interp_p1_f32 v2, v0, attr0.x
            0xC809_0001,        // v_interp_p2_f32 v2, v1, attr0.x
            v_add_f32(3, 2, 1), // v3 = attribute + j
            e0,
            e1,
            0xBF81_0000,
        ];
        let shader = pixel(&words, PERSP_CENTER).unwrap();
        let (features, inputs) = parts(&shader);
        assert_ne!(features & 0x2_0000, 0);
        assert_eq!(inputs.iter().filter(|e| e.kind == 28).count(), 1);
        validate(&shader);
    }
    /// The `PSV0` interpolation mode of TEXCOORD0 (row 0, x-w, float).
    fn texcoord_mode(shader: &HostShader) -> u8 {
        let c = Container::parse(&shader.dxil).unwrap();
        let psv = c.part(FourCC::PSV0).unwrap().data;
        let row = [1, 0, 0x44, 0, 3];
        let at = psv.windows(5).position(|w| w == row).unwrap();
        psv[at + 5]
    }

    const S_ENDPGM: u32 = 0xBF81_0000;

    /// An interpolation whose i is computed (`v_mul_f32 v3, 0.5, v0`) reads
    /// the attribute's vertices: TEXCOORD0 becomes constant, read with
    /// `attributeAtVertex`; i and j are `SV_Barycentrics` values.
    #[test]
    fn computed_barycentrics_read_the_vertices() {
        let [e0, e1] = exp_mrt0(2);
        let words = [
            0x1006_00F0, // v_mul_f32 v3, 0.5, v0
            0xC808_0003, // v_interp_p1_f32 v2, v3, attr0.x
            0xC809_0001, // v_interp_p2_f32 v2, v1, attr0.x
            e0,
            e1,
            S_ENDPGM,
        ];
        let shader = pixel(&words, PERSP_CENTER).unwrap();
        let (features, inputs) = parts(&shader);
        assert_ne!(features & 0x2_0000, 0, "no Barycentrics feature");
        assert_eq!(inputs.iter().filter(|e| e.kind == 28).count(), 1);
        assert_eq!(texcoord_mode(&shader), 1);
        validate(&shader);
    }

    /// Weights from constants alone: the vertices, no `SV_Barycentrics`.
    #[test]
    fn constant_weights_need_no_barycentrics() {
        let [e0, e1] = exp_mrt0(2);
        let words = [
            0x7E06_02F0, // v_mov_b32 v3, 0.5
            0xC808_0003, // v_interp_p1_f32 v2, v3, attr0.x
            0xC809_0003, // v_interp_p2_f32 v2, v3, attr0.x
            e0,
            e1,
            S_ENDPGM,
        ];
        let shader = pixel(&words, PERSP_CENTER).unwrap();
        let (features, inputs) = parts(&shader);
        assert_ne!(features & 0x2_0000, 0, "no Barycentrics feature");
        assert!(inputs.iter().all(|e| e.kind != 28), "{inputs:?}");
        assert_eq!(texcoord_mode(&shader), 1);
        validate(&shader);
    }

    /// The same attribute interpolated with the loaded i and j and with a
    /// computed j: both from its vertices, the first weighed with
    /// `SV_Barycentrics`.
    #[test]
    fn computed_and_loaded_barycentrics_on_one_attribute() {
        let [e0, e1] = exp_mrt0(2);
        let words = [
            0xC810_0100, // v_interp_p1_f32 v4, v0, attr0.y
            0xC811_0101, // v_interp_p2_f32 v4, v1, attr0.y
            0x1006_02F0, // v_mul_f32 v3, 0.5, v1
            0xC808_0000, // v_interp_p1_f32 v2, v0, attr0.x
            0xC809_0003, // v_interp_p2_f32 v2, v3, attr0.x
            v_add_f32(2, 2, 4),
            e0,
            e1,
            S_ENDPGM,
        ];
        let shader = pixel(&words, PERSP_CENTER).unwrap();
        let (_, inputs) = parts(&shader);
        assert_eq!(inputs.iter().filter(|e| e.kind == 28).count(), 1);
        assert_eq!(texcoord_mode(&shader), 1);
        validate(&shader);
    }

    /// P10 of an attribute the program interpolates is P1 - P0.
    #[test]
    fn p10_of_an_interpolated_attribute() {
        let [e0, e1] = exp_mrt0(2);
        let words = [
            0xC808_0000, // v_interp_p1_f32 v2, v0, attr0.x
            0xC809_0001, // v_interp_p2_f32 v2, v1, attr0.x
            0xC812_0000, // v_interp_mov_f32 v4, p10, attr0.x
            v_add_f32(2, 2, 4),
            e0,
            e1,
            S_ENDPGM,
        ];
        let shader = pixel(&words, PERSP_CENTER).unwrap();
        assert_eq!(texcoord_mode(&shader), 1);
        validate(&shader);
        // Alone, what the hardware holds is not known.
        let [e0, e1] = exp_mrt0(4);
        let words = [0xC812_0000, e0, e1, S_ENDPGM];
        assert!(pixel(&words, PERSP_CENTER).is_err());
    }
}
