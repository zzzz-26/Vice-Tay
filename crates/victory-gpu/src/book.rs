//! Who has what memory of the card, and who had it (diagnostics).
//!
//! A device loss names a faulting address and, from a short list, what was
//! freed there. A texture gone for minutes is in "nothing named", and which
//! descriptor still names it is not told at all. Here every buffer, texture
//! and texture block of ours stands with its address from when it is made,
//! and stays with the time it went ([`made`], [`block`]); a texture's address
//! is the base in the descriptor of its first view, which the card's heaps
//! hold as plain bytes (measured: `indirect::tests::descriptors_written_under_readers`).
//!
//! - At a device loss ([`at_loss`]): who owns the faulting address and who
//!   owned it, newest first, and every slot of the view heap whose
//!   descriptor reaches it.
//! - `VICTORY_SLOT_CHECK=1`: a thread walks the view heap and says which
//!   slots name memory that is gone, while nothing is lost yet ([`check`]).
//!
//! The book is kept with device-removed data on (`VICTORY_D3D12_DRED`) or the
//! check; `VICTORY_BOOK=0` keeps none.
//!
//! What the descriptors' bits mean is AMD's (RDNA 2 and 3); on another card
//! nothing is decoded as an address we know, and nothing is said.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
use windows::core::{GUID, HRESULT, IUnknown, Interface};

/// Entries of what is gone that are kept.
const GONE: usize = 400_000;
/// Longer than anything of ours: how far below an address its owner starts.
const LONGEST: u64 = 2 << 30;
/// Entries longer than this are few and looked through one by one.
const LONG: u64 = 64 << 20;

#[derive(Clone, Debug)]
struct Entry {
    start: u64,
    size: u64,
    what: String,
    made: u64,
    /// What was told of it since ([`note`]).
    notes: String,
}

#[derive(Default)]
struct Book {
    next: u64,
    live: HashMap<u64, Entry>,
    by_start: BTreeMap<(u64, u64), ()>,
    long: Vec<u64>,
    /// By the object's address, for a name given after it was made.
    by_object: HashMap<usize, u64>,
    gone: VecDeque<(Entry, u64)>,
    /// Objects whose address could not be told, by what they are.
    unknown: BTreeMap<String, u64>,
}

static BOOK: Mutex<Option<Book>> = Mutex::new(None);
/// Root descriptors given by address lately: time, address, bytes.
static ROOTS: Mutex<VecDeque<(u64, u64, u64)>> = Mutex::new(VecDeque::new());

/// Tests that want the book.
#[cfg(test)]
static FORCED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub(crate) fn on() -> bool {
    #[cfg(test)]
    if FORCED.load(Ordering::Relaxed) {
        return true;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("VICTORY_BOOK").is_none_or(|v| v != "0")
            && !cfg!(test)
            && (crate::host::dred_any() || checks())
    })
}

/// `VICTORY_SLOT_CHECK=1`.
pub(crate) fn checks() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SLOT_CHECK").is_some_and(|v| v != "0"))
}

fn with<T>(f: impl FnOnce(&mut Book) -> T) -> T {
    f(BOOK.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(Book::default))
}

impl Book {
    fn add(&mut self, object: usize, start: u64, size: u64, what: String, now: u64) -> u64 {
        self.next += 1;
        let id = self.next;
        self.by_start.insert((start, id), ());
        if size > LONG {
            self.long.push(id);
        }
        self.live.insert(id, Entry { start, size, what, made: now, notes: String::new() });
        if object != 0 {
            self.by_object.insert(object, id);
        }
        id
    }

    fn remove(&mut self, id: u64, object: usize, now: u64) {
        if let Some(entry) = self.live.remove(&id) {
            self.by_start.remove(&(entry.start, id));
            if entry.size > LONG {
                self.long.retain(|&l| l != id);
            }
            if self.gone.len() >= GONE {
                self.gone.pop_front();
            }
            self.gone.push_back((entry, now));
        }
        if self.by_object.get(&object) == Some(&id) {
            self.by_object.remove(&object);
        }
    }

    /// Whether something alive holds `at`.
    fn alive(&self, at: u64) -> bool {
        let holds = |id: &u64| self.live.get(id).is_some_and(|e| at >= e.start && at - e.start < e.size);
        self.by_start
            .range((at.saturating_sub(LONG), 0)..=(at, u64::MAX))
            .rev()
            .any(|((_, id), ())| holds(id))
            || self.long.iter().any(holds)
    }

    /// The live entries that hold `at`, the one that starts last first.
    fn holders(&self, at: u64) -> Vec<&Entry> {
        let from = at.saturating_sub(LONGEST);
        self.by_start
            .range((from, 0)..=(at, u64::MAX))
            .rev()
            .filter_map(|((_, id), ())| self.live.get(id))
            .filter(|e| at - e.start < e.size)
            .take(8)
            .collect()
    }

    /// The live entry that starts last at or below `at`, and the next above.
    fn neighbours(&self, at: u64) -> [Option<&Entry>; 2] {
        let below = self.by_start.range(..=(at, u64::MAX)).next_back();
        let above = self.by_start.range((at + 1, 0)..).next();
        [below, above].map(|e| e.and_then(|((_, id), ())| self.live.get(id)))
    }

    /// What held `at` and is gone, newest first.
    fn gone_at(&self, at: u64, most: usize) -> Vec<&(Entry, u64)> {
        self.gone.iter().rev().filter(|(e, _)| at >= e.start && at - e.start < e.size).take(most).collect()
    }
}

#[repr(C)]
struct Notice {
    methods: &'static [*const (); 3],
    references: AtomicU32,
    id: u64,
    object: usize,
}

struct Methods([*const (); 3]);
// SAFETY: function addresses only.
unsafe impl Sync for Methods {}
static NOTICE: Methods = Methods([query as *const (), add_ref as *const (), release as *const ()]);

unsafe extern "system" fn query(this: *mut c_void, iid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    // SAFETY: COM's arguments.
    unsafe {
        if out.is_null() {
            return HRESULT(0x8000_4003_u32 as i32);
        }
        if iid.as_ref().is_some_and(|iid| *iid == IUnknown::IID) {
            add_ref(this);
            *out = this;
            return HRESULT(0);
        }
        *out = std::ptr::null_mut();
    }
    HRESULT(0x8000_4002_u32 as i32)
}

unsafe extern "system" fn add_ref(this: *mut c_void) -> u32 {
    // SAFETY: one of ours, alive while a reference to it is.
    unsafe { (*this.cast::<Notice>()).references.fetch_add(1, Ordering::Relaxed) + 1 }
}

unsafe extern "system" fn release(this: *mut c_void) -> u32 {
    // SAFETY: as above; the last reference frees the box it was made as.
    unsafe {
        let left = (*this.cast::<Notice>()).references.fetch_sub(1, Ordering::AcqRel) - 1;
        if left == 0 {
            let gone = Box::from_raw(this.cast::<Notice>());
            let now = crate::host::now_ms();
            with(|b| b.remove(gone.id, gone.object, now));
        }
        left
    }
}

/// Has the book told when `object` goes.
fn watch(object: &ID3D12Object, id: u64) {
    const KEY: GUID = GUID::from_u128(0x2c7d1a55_9e0b_46f2_8c11_5b8e70a4d913);
    let raw = object.as_raw() as usize;
    let notice = Box::into_raw(Box::new(Notice { methods: &NOTICE.0, references: AtomicU32::new(1), id, object: raw }));
    // SAFETY: a COM object of ours with one reference, which this drops; the
    // object takes its own.
    unsafe {
        let notice = IUnknown::from_raw(notice.cast());
        if object.SetPrivateDataInterface(&KEY, &notice).is_err() {
            std::mem::forget(notice);
        }
    }
}

/// A view format for a typeless one (its first with a plain type).
fn viewable(format: i32) -> Option<i32> {
    Some(match format {
        1 => 2,
        5 => 6,
        9 => 11,
        15 => 16,
        19 => 21,
        23 => 24,
        27 => 28,
        33 => 35,
        39 => 41,
        44 => 46,
        48 => 49,
        53 => 56,
        60 => 61,
        70 | 73 | 76 | 79 | 82 | 94 | 97 => format + 1,
        90 => 87,
        92 => 88,
        20 | 40 | 45 | 55 => return None, // typed depth: no shader view
        f => f,
    })
}

/// The address of `texture` as its descriptor has it (its first level's),
/// if the book is kept.
pub(crate) fn address(texture: &ID3D12Resource) -> Option<u64> {
    if !on() {
        return None;
    }
    // SAFETY: a live resource.
    let desc = unsafe { texture.GetDesc() };
    texture_address(texture, &desc)
}

/// The address of a texture: the base its descriptor has.
fn texture_address(texture: &ID3D12Resource, desc: &D3D12_RESOURCE_DESC) -> Option<u64> {
    // The heap is held, not read: it keeps the slot's descriptors alive.
    struct Slot(#[allow(dead_code)] ID3D12DescriptorHeap, usize);
    // SAFETY: a descriptor heap, used under the lock.
    unsafe impl Send for Slot {}
    static SLOT: Mutex<Option<Slot>> = Mutex::new(None);
    if desc.Flags.0 & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE.0 != 0 {
        return None;
    }
    let host = crate::host::made_host()?;
    let mut slot = SLOT.lock().unwrap_or_else(|e| e.into_inner());
    if slot.is_none() {
        // SAFETY: plain heap creation.
        let heap: ID3D12DescriptorHeap = unsafe {
            host.device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 1,
                ..Default::default()
            })
        }
        .ok()?;
        // SAFETY: a live heap.
        let at = unsafe { heap.GetCPUDescriptorHandleForHeapStart().ptr };
        *slot = Some(Slot(heap, at));
    }
    let at = slot.as_ref()?.1;
    let format = viewable(desc.Format.0)?;
    let mut view = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT(format),
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        ..Default::default()
    };
    match desc.Dimension {
        D3D12_RESOURCE_DIMENSION_TEXTURE1D => {
            view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE1D;
            view.Anonymous.Texture1D.MipLevels = 1;
        }
        D3D12_RESOURCE_DIMENSION_TEXTURE3D => {
            view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE3D;
            view.Anonymous.Texture3D.MipLevels = 1;
        }
        _ if desc.SampleDesc.Count > 1 => view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2DMS,
        _ => view.Anonymous.Texture2D.MipLevels = 1,
    }
    let mut words = [0u8; 32];
    // SAFETY: a slot of ours, a live texture; the handle is the descriptor's
    // memory or the read fails.
    unsafe {
        host.device
            .CreateShaderResourceView(texture, Some(&view), D3D12_CPU_DESCRIPTOR_HANDLE { ptr: at });
    }
    if !crate::host::read_into(at as *const u8, &mut words) {
        return None;
    }
    match decode(&words) {
        Names::Texture { base, .. } => Some(base),
        _ => None,
    }
}

/// `resource` was made; `what` it is (a name given later replaces it).
pub(crate) fn made(resource: &ID3D12Resource, what: impl FnOnce() -> String) {
    if !on() {
        return;
    }
    let Ok(object) = resource.cast::<ID3D12Object>() else { return };
    let raw = object.as_raw() as usize;
    if with(|b| b.by_object.contains_key(&raw)) {
        return named(&object, what);
    }
    // SAFETY: a live resource.
    let desc = unsafe { resource.GetDesc() };
    let (start, size) = if desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER {
        // SAFETY: as above.
        (Some(unsafe { resource.GetGPUVirtualAddress() }), desc.Width)
    } else {
        let size = crate::host::made_host()
            // SAFETY: a plain query.
            .map_or(0, |h| unsafe { h.device.GetResourceAllocationInfo(0, &[desc]) }.SizeInBytes);
        (texture_address(resource, &desc), if size == u64::MAX { 0 } else { size })
    };
    let Some(start) = start.filter(|&s| s != 0) else {
        return with(|b| *b.unknown.entry(format!("f{} flags {:#x}", desc.Format.0, desc.Flags.0)).or_default() += 1);
    };
    let what = what();
    let now = crate::host::now_ms();
    let id = with(|b| b.add(raw, start, size.max(1), what, now));
    watch(&object, id);
}

/// A block of texture memory, found by the texture `resource` at `offset` of it.
pub(crate) fn block(heap: &ID3D12Heap, resource: &ID3D12Resource, offset: u64, bytes: u64) {
    if !on() {
        return;
    }
    let (Ok(heap), Ok(object)) = (heap.cast::<ID3D12Object>(), resource.cast::<ID3D12Object>()) else {
        return;
    };
    let raw = heap.as_raw() as usize;
    let now = crate::host::now_ms();
    let id = with(|b| {
        let at = b.by_object.get(&(object.as_raw() as usize)).and_then(|id| b.live.get(id))?.start;
        // Known, unless the entry is of what had the pointer before.
        if b.by_object
            .get(&raw)
            .and_then(|id| b.live.get(id))
            .is_some_and(|e| at >= e.start && at - e.start < e.size)
        {
            return None;
        }
        Some(b.add(raw, at.checked_sub(offset)?, bytes, "a block of textures".into(), now))
    });
    if let Some(id) = id {
        watch(&heap, id);
    }
}

/// Something to say of the object at `raw` if it is in the book.
pub(crate) fn note(raw: usize, what: impl FnOnce() -> String) {
    if !on() {
        return;
    }
    let what = what();
    with(|b| {
        if let Some(id) = b.by_object.get(&raw).copied()
            && let Some(e) = b.live.get_mut(&id)
            && e.notes.len() < 600
        {
            e.notes.push_str(&what);
        }
    });
}

/// A name for something made before.
pub(crate) fn named(object: &ID3D12Object, what: impl FnOnce() -> String) {
    let raw = object.as_raw() as usize;
    if with(|b| b.by_object.contains_key(&raw)) {
        let what = what();
        with(|b| {
            if let Some(id) = b.by_object.get(&raw).copied()
                && let Some(e) = b.live.get_mut(&id)
            {
                e.what = what;
            }
        });
    }
}

/// A root descriptor was set by address.
pub(crate) fn root(gpu: u64, bytes: u64) {
    if on() {
        let mut roots = ROOTS.lock().unwrap_or_else(|e| e.into_inner());
        if roots.len() >= 16_384 {
            roots.pop_front();
        }
        roots.push_back((crate::host::now_ms(), gpu, bytes));
    }
}

/// What a descriptor's 32 bytes name.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Names {
    Nothing,
    Texture { base: u64, meta: Option<u64>, kind: u32, width: u32, height: u32 },
    Buffer { base: u64, stride: u32, bytes: u64 },
}

/// The descriptor `bytes` hold, as AMD's cards have them since RDNA 2.
pub(crate) fn decode(bytes: &[u8; 32]) -> Names {
    let w = |i: usize| u32::from_le_bytes([bytes[4 * i], bytes[4 * i + 1], bytes[4 * i + 2], bytes[4 * i + 3]]);
    let kind = w(3) >> 28;
    if kind >= 8 {
        let base = (u64::from(w(0)) | u64::from(w(1) & 0xff) << 32) << 8;
        if base == 0 {
            return Names::Nothing;
        }
        // Compressed (dword 6, bit 21): where the metadata lies.
        let meta = (w(6) >> 21 & 1 != 0 && w(7) != 0).then(|| u64::from(w(7)) << 16 | u64::from(w(6) >> 24) << 8);
        let width = ((w(1) >> 30) | (w(2) & 0xfff) << 2) + 1;
        let height = (w(2) >> 14 & 0x3fff) + 1;
        return Names::Texture { base, meta, kind, width, height };
    }
    let base = u64::from(w(0)) | u64::from(w(1) & 0xffff) << 32;
    let stride = w(1) >> 16 & 0x3fff;
    let bytes = u64::from(w(2)) * u64::from(stride.max(1));
    if base == 0 || bytes == 0 {
        Names::Nothing
    } else {
        Names::Buffer { base, stride, bytes }
    }
}

/// What the book says of an address: alive, gone or never there.
fn fate(at: u64) -> Fate {
    with(|b| {
        if b.alive(at) {
            return Fate::Alive;
        }
        match b.gone_at(at, 1).first() {
            Some((e, gone)) => Fate::Gone(format!(
                "{} at {:#x}+{:#x} made at {:.1} s, gone at {:.1} s",
                e.what,
                e.start,
                e.size,
                e.made as f64 / 1e3,
                *gone as f64 / 1e3
            )),
            None => Fate::Never,
        }
    })
}

enum Fate {
    Alive,
    Gone(String),
    Never,
}

/// The addresses a descriptor reads first.
fn bases(names: Names) -> Vec<(&'static str, u64)> {
    match names {
        Names::Nothing => Vec::new(),
        Names::Texture { base, meta, .. } => {
            let mut out = vec![("texture", base)];
            out.extend(meta.map(|m| ("metadata", m)));
            out
        }
        Names::Buffer { base, bytes, .. } => vec![("buffer", base), ("buffer's end", base + bytes - 1)],
    }
}

/// Starts the thread of `VICTORY_SLOT_CHECK=1`, once.
pub(crate) fn check() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    if !checks() || !on() {
        return;
    }
    ONCE.call_once(|| {
        let _ = std::thread::Builder::new().name("slot check".into()).spawn(walk);
    });
}

/// Slots found, since the start: looked at, naming nothing, textures,
/// buffers, naming what is gone, naming what was never there.
static FOUND: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

fn walk() {
    use Ordering::Relaxed;
    const CHUNK: u32 = 4096;
    let mut at = 0u32;
    // Slots said already, by slot and address.
    let mut said: HashMap<(u32, u64), u64> = HashMap::new();
    let mut suspects: Vec<(u32, [u8; 32], u64)> = Vec::new();
    let mut round = std::time::Instant::now();
    let mut never_said = 0;
    let mut gone_said = 0;
    loop {
        let read = crate::descriptor::host::shown_bytes(at, CHUNK);
        let Some((first, bytes, end)) = read else {
            std::thread::sleep(std::time::Duration::from_millis(500));
            continue;
        };
        for (i, d) in bytes.chunks_exact(32).enumerate() {
            let d: [u8; 32] = d.try_into().unwrap_or([0; 32]);
            let names = decode(&d);
            FOUND[0].fetch_add(1, Relaxed);
            FOUND[match names {
                Names::Nothing => 1,
                Names::Texture { .. } => 2,
                Names::Buffer { .. } => 3,
            }]
            .fetch_add(1, Relaxed);
            if bases(names).iter().any(|&(_, base)| !matches!(fate(base), Fate::Alive)) {
                suspects.push((first + i as u32, d, crate::host::now_ms()));
            }
        }
        // Again a moment later: a slot between two states says nothing.
        let now = crate::host::now_ms();
        let due: Vec<(u32, [u8; 32], u64)> = {
            let (due, later) = suspects.into_iter().partition(|s| now >= s.2 + 400);
            suspects = later;
            due
        };
        for (slot, was, since) in due {
            let Some((_, again, _)) = crate::descriptor::host::shown_bytes(slot, 1) else { continue };
            if again != was {
                continue;
            }
            for (part, base) in bases(decode(&was)) {
                let (what, text) = match fate(base) {
                    Fate::Alive => continue,
                    Fate::Gone(text) => (4, text),
                    Fate::Never => (5, "nothing the book ever had".to_owned()),
                };
                FOUND[what].fetch_add(1, Relaxed);
                let times = said.entry((slot, base)).or_insert(0);
                *times += 1;
                let quota = if what == 4 { &mut gone_said } else { &mut never_said };
                if *times > 1 || *quota >= if what == 4 { 400 } else { 60 } {
                    continue;
                }
                *quota += 1;
                tracing::warn!(
                    "slot check: view slot {slot} names {part} {base:#x} ({:?}), {text}; seen since {:.1} s; {}",
                    decode(&was),
                    since as f64 / 1e3,
                    crate::descriptor::host::slot_story(slot)
                );
            }
        }
        at = first + (bytes.len() / 32) as u32;
        if at >= end {
            at = 0;
            let f = [0, 1, 2, 3, 4, 5].map(|i| FOUND[i].swap(0, Relaxed));
            let (live, gone, unknown) = with(|b| (b.live.len(), b.gone.len(), b.unknown.clone()));
            tracing::info!(
                "slot check: a round of the view heap in {:.1} s: {} slots, {} null, {} textures, {} buffers; {} name what is gone, {} what the book never had; the book has {live} things alive and {gone} gone; textures without an address {unknown:?}",
                round.elapsed().as_secs_f64(),
                f[0], f[1], f[2], f[3], f[4], f[5]
            );
            round = std::time::Instant::now();
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
    }
}

/// For the report of a lost device: who has and had the faulting address,
/// and what names it.
pub(crate) fn at_loss(va: u64) -> String {
    if !on() {
        return String::new();
    }
    let mut text = with(|b| {
        let one =
            |e: &Entry| format!("{} {:#x}+{:#x} made at {:.3} s{}", e.what, e.start, e.size, e.made as f64 / 1e3, e.notes);
        let mut text = String::from("; BOOK: alive there:");
        for e in b.holders(va) {
            text.push_str(&format!(" [{}]", one(e)));
        }
        for (name, e) in ["below", "above"].iter().zip(b.neighbours(va)) {
            if let Some(e) = e {
                text.push_str(&format!(" {name} [{}, {:#x} away]", one(e), e.start.abs_diff(va)));
            }
        }
        text.push_str("; gone there, newest first:");
        for (e, gone) in b.gone_at(va, 12) {
            text.push_str(&format!(" [{} gone at {:.3} s]", one(e), *gone as f64 / 1e3));
        }
        text.push_str(&format!("; now {:.3} s, submissions (begun, finished) {:?}, {} alive, {} gone", crate::host::now_ms() as f64 / 1e3, crate::host::submissions(), b.live.len(), b.gone.len()));
        text
    });
    {
        let roots = ROOTS.lock().unwrap_or_else(|e| e.into_inner());
        let mut near: Vec<(u64, u64, u64)> = roots.iter().filter(|r| r.1 <= va && va - r.1 < 4 << 30).copied().collect();
        near.sort_by_key(|r| va - r.1);
        near.dedup_by_key(|r| r.1);
        for (at, gpu, bytes) in near.into_iter().take(4) {
            text.push_str(&format!("; root descriptor {gpu:#x}+{bytes:#x} ({:#x} below) at {:.1} s", va - gpu, at as f64 / 1e3));
        }
    }
    // Every slot that reaches the address, and the nearest below it.
    let mut reach: Vec<(u32, Names)> = Vec::new();
    let mut near: Vec<(u64, u32, Names)> = Vec::new();
    let mut at = 0;
    let began = std::time::Instant::now();
    while let Some((first, bytes, end)) = crate::descriptor::host::shown_bytes(at, 16_384) {
        for (i, d) in bytes.chunks_exact(32).enumerate() {
            let slot = first + i as u32;
            let names = decode(&d.try_into().unwrap_or([0; 32]));
            let (base, len) = match names {
                Names::Nothing => continue,
                Names::Texture { base, meta, .. } => {
                    if meta.is_some_and(|m| va >= m && va - m < 32 << 20) {
                        reach.push((slot, names));
                    }
                    let len = with(|b| {
                        b.holders(base)
                            .iter()
                            .map(|e| e.start + e.size - base)
                            .min()
                            .or_else(|| b.gone_at(base, 1).first().map(|(e, _)| e.start + e.size - base))
                    });
                    (base, len.unwrap_or(64 << 20) + (64 << 10))
                }
                Names::Buffer { base, bytes, .. } => (base, bytes),
            };
            if va >= base && va - base < len {
                reach.push((slot, names));
            } else if va >= base && va - base < 4 << 30 {
                near.push((va - base, slot, names));
                if near.len() > 4096 {
                    near.sort_by_key(|n| n.0);
                    near.truncate(16);
                }
            }
        }
        at = first + (bytes.len() / 32) as u32;
        if at >= end || began.elapsed().as_secs() > 20 {
            break;
        }
    }
    text.push_str(&format!("; {} slots reach it (heap read in {:.1} s)", reach.len(), began.elapsed().as_secs_f64()));
    for (slot, names) in reach.iter().take(24) {
        let (Names::Texture { base, .. } | Names::Buffer { base, .. }) = *names else { continue };
        let fate = match fate(base) {
            Fate::Alive => "alive".to_owned(),
            Fate::Gone(text) => format!("GONE: {text}"),
            Fate::Never => "never in the book".to_owned(),
        };
        text.push_str(&format!("; slot {slot} {names:?} {fate} {}", crate::descriptor::host::slot_story(*slot)));
    }
    near.sort_by_key(|n| n.0);
    for (away, slot, names) in near.iter().take(8) {
        text.push_str(&format!("; near: slot {slot} {names:?} {away:#x} below {}", crate::descriptor::host::slot_story(*slot)));
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two descriptors of the card that were read (a texture of 64x64 and
    /// one of 2048x2048, RX 7900 XTX) and a raw buffer's.
    #[test]
    fn descriptors_are_read_as_the_card_has_them() {
        let mut d = [0u8; 32];
        d[..16].copy_from_slice(&[0, 0xb4, 0, 2, 0, 0, 0xa0, 0xc2, 0x0f, 0xc0, 0x0f, 0, 0xac, 0x0f, 0x60, 0x91]);
        assert_eq!(decode(&d), Names::Texture { base: 0x2_00b4_0000, meta: None, kind: 9, width: 64, height: 64 });
        d[..16].copy_from_slice(&[0, 0xb4, 2, 2, 0, 0, 0xa0, 0xc2, 0xff, 0xc1, 0xff, 1, 0xac, 0x0f, 0xb0, 0x91]);
        assert_eq!(decode(&d), Names::Texture { base: 0x2_02b4_0000, meta: None, kind: 9, width: 2048, height: 2048 });
        d = [0; 32];
        assert_eq!(decode(&d), Names::Nothing);
        d[..8].copy_from_slice(&0x1_2345_6780u64.to_le_bytes());
        d[8..12].copy_from_slice(&400u32.to_le_bytes());
        assert_eq!(decode(&d), Names::Buffer { base: 0x1_2345_6780, stride: 0, bytes: 400 });
        d[6] = 24; // a stride
        assert_eq!(decode(&d), Names::Buffer { base: 0x1_2345_6780, stride: 24, bytes: 9600 });
    }

    /// Who has an address, and who had it, newest first.
    /// A texture of the pool and its block stand in the book while they
    /// live, and not after.
    #[test]
    fn a_pooled_texture_and_its_block_are_in_the_book() {
        let Some(host) = crate::host::host() else { return };
        FORCED.store(true, Ordering::Relaxed);
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 128,
            Height: 256,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT(83),
            SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        // SAFETY: a valid description.
        let texture = unsafe { crate::pool::texture(&host.device, &desc, D3D12_RESOURCE_STATE_COMMON, true) };
        if let Some(texture) = texture {
            let at = texture_address(&texture, &desc);
            println!("address {at:x?}");
            if let Some(at) = at {
                let names = || with(|b| b.holders(at).iter().map(|e| e.what.clone()).collect::<Vec<_>>());
                let found = names();
                println!("held by {found:?}");
                assert!(found.iter().any(|n| n.contains("a texture in a block")), "{found:?}");
                assert!(found.iter().any(|n| n.contains("a block of textures")), "{found:?}");
                drop(texture);
                assert!(!names().iter().any(|n| n.contains("a texture in a block")));
            }
        }
        FORCED.store(false, Ordering::Relaxed);
    }

    #[test]
    fn the_book_knows_what_is_gone() {
        let mut b = Book::default();
        let block = b.add(1, 0x1000_0000, 64 << 20, "a block".into(), 5);
        let one = b.add(2, 0x1000_0000, 0x8000, "one".into(), 6);
        let two = b.add(3, 0x1000_8000, 0x8000, "two".into(), 7);
        let names = |b: &Book, at| b.holders(at).iter().map(|e| e.what.clone()).collect::<Vec<_>>();
        assert_eq!(names(&b, 0x1000_0100), ["one", "a block"]);
        assert_eq!(names(&b, 0x1000_8100), ["two", "a block"]);
        assert!(b.alive(0x1000_8100) && !b.alive(0x1000_0000 + (64 << 20)) && !b.alive(0xfff_ffff));
        assert!(names(&b, 0x1000_0000 + (64 << 20)).is_empty());
        b.remove(one, 2, 10);
        b.remove(block, 1, 12);
        assert!(names(&b, 0x1000_0100).is_empty());
        let gone: Vec<(String, u64)> = b.gone_at(0x1000_0100, 9).iter().map(|(e, at)| (e.what.clone(), *at)).collect();
        assert_eq!(gone, [("a block".to_owned(), 12), ("one".to_owned(), 10)]);
        assert_eq!(b.neighbours(0x1000_0100)[1].map(|e| e.what.as_str()), Some("two"));
        b.remove(two, 3, 13);
        assert!(b.neighbours(u64::MAX - 1)[0].is_none() && b.by_object.is_empty());
    }
}
