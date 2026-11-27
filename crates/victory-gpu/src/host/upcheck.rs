//! Diagnostics (`VICTORY_UPCOPY_CHECK=1`; off by default): whether what goes
//! up to a copy in the card's memory is older than what the card holds.
//!
//! What every send up rests on: outside ranges shaders may be writing, and
//! but for what the title wrote since, a copy's CPU side holds what its card
//! side holds. Two looks at that, both on the card:
//!
//! - **at each send**: a refresh tells what the CPU's side held before
//!   ([`note`]) and what it took. Ahead of the send, in its place on the
//!   queue, the card's words are compared with the former. A word the title
//!   did not change (both are equal) and the card holds another value of is
//!   one the send puts an older value over;
//! - **at rest**: every few seconds, when the flights were settled for the
//!   sweep and nothing is in the air, every copy is compared whole with its
//!   CPU side. A word that differs is one a later send would undo. The log
//!   line says which buffers, what stands on the card, on the CPU's side and
//!   in title memory, and whether the range was declared as written at all
//!   (if not: a shader writes through a view nobody declared) and fetched
//!   lately (if so: what came down was not what the card ended with).
//!
//! The looks cost memory and time (the bytes sent twice more, and each
//! copy read over the bus at rest): the check is not for runs that are timed.

use std::collections::{BTreeMap, HashMap};

use windows::Win32::Foundation::CloseHandle;

use super::*;
use crate::shadow::Shadow;

pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_UPCOPY_CHECK").is_some_and(|v| v == "1"))
}

/// One dispatch looks at up to a piece: `words` words of the card's buffer
/// from `at`, beside `was` (what the CPU's side held when both were known to
/// be alike) and `new` (what the title has there now). Where the two are
/// equal and the card's word is another one, it is counted, and the first
/// found is kept: its offset, the card's value, the CPU's.
const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=5, b0), SRV(t0), SRV(t1), UAV(u0)"
cbuffer Look : register(b0) { uint at; uint words; uint was; uint now; uint found_at; };
ByteAddressBuffer card : register(t0);
ByteAddressBuffer cpu : register(t1);
RWByteAddressBuffer found : register(u0);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= words) return;
    uint k = id.x * 4u, n;
    uint c = card.Load(at + k), w = cpu.Load(was + k);
    if (c == w || cpu.Load(now + k) != w) return;
    found.InterlockedAdd(found_at, 1u, n);
    if (n == 0u) found.Store3(found_at + 4u, uint3(at + k, c, w));
}
"#;

/// Bytes of an answer.
const ANSWER: u64 = 16;
const PIECE: usize = 64 << 10;

struct Pass {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
}

fn pass(host: &Host) -> Option<&'static Pass> {
    static PASS: OnceLock<Option<Pass>> = OnceLock::new();
    PASS.get_or_init(|| {
        let code = compile(SOURCE, windows::core::s!("main"), windows::core::s!("cs_5_1")).ok()?;
        // SAFETY: the code lives through the calls; the root signature is
        // inside it.
        unsafe {
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, &code).ok()?;
            let pso: ID3D12PipelineState = host
                .device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: code.as_ptr().cast(),
                        BytecodeLength: code.len(),
                    },
                    ..Default::default()
                })
                .ok()?;
            Some(Pass { root, pso })
        }
    })
    .as_ref()
}

/// A piece a refresh took: its offset in the copy, what the CPU's side held
/// and what it holds now.
pub(crate) struct Taken {
    at: usize,
    was: Vec<u8>,
    new: Vec<u8>,
}

/// Refreshes since the last send, per copy (by its number).
static NOTED: Mutex<BTreeMap<u64, Vec<Taken>>> = Mutex::new(BTreeMap::new());

/// At most this much is looked at per submission.
const MOST: usize = 128 << 20;

/// Offsets `[at, at + was.len())` of `copy`'s CPU side held `was` and hold
/// `new` from here on. (Whole words: a word only partly in it is left out.)
pub(crate) fn note(copy: u64, at: usize, was: &[u8], new: &[u8]) {
    let skip = at.wrapping_neg() & 3;
    let n = was.len().min(new.len()).saturating_sub(skip) & !3;
    if n == 0 {
        return;
    }
    let at = at + skip;
    let (mut was, new) = (was[skip..skip + n].to_vec(), new[skip..skip + n].to_vec());
    let mut noted = NOTED.lock().unwrap_or_else(|e| e.into_inner());
    let noted = noted.entry(copy).or_default();
    // What was noted before in this submission has not gone up yet: the card
    // knows what stood there before that. And what it took is not what goes
    // up where this took it again.
    for t in noted.iter_mut() {
        let (a, b) = (at.max(t.at), (at + n).min(t.at + t.was.len()));
        if a < b {
            was[a - at..b - at].copy_from_slice(&t.was[a - t.at..b - t.at]);
            t.new[a - t.at..b - t.at].copy_from_slice(&new[a - at..b - at]);
        }
    }
    noted.push(Taken { at, was, new });
}

/// A submission's look at what it sends up, until it has run.
pub(super) struct Looks {
    found: (ID3D12Resource, usize),
    _bytes: ID3D12Resource,
    /// Per answer: the copy, and the bytes looked at.
    of: Vec<(std::sync::Arc<Shadow>, usize)>,
}

/// Per copy, since the last line: sends looked at, their bytes, words an
/// older value went over, and the first few of them.
#[derive(Default)]
struct Stale {
    looked: u64,
    bytes: u64,
    sends: u64,
    words: u64,
    some: Vec<String>,
}

static STALE: Mutex<BTreeMap<(u64, u64), Stale>> = Mutex::new(BTreeMap::new());

/// Records the looks at what `dirty` sends up into `list`, ahead of the
/// sends.
pub(super) fn record(
    host: &Host,
    list: &ID3D12GraphicsCommandList,
    dirty: &[(std::sync::Arc<Shadow>, Vec<(usize, usize)>)],
) -> Option<Looks> {
    let mut noted = std::mem::take(&mut *NOTED.lock().unwrap_or_else(|e| e.into_inner()));
    let mut all: Vec<(std::sync::Arc<Shadow>, Taken)> = Vec::new();
    let mut total = 0;
    for (shadow, _) in dirty {
        for taken in noted.remove(&shadow.id()).unwrap_or_default() {
            total += 2 * taken.was.len();
            if total <= MOST {
                all.push((shadow.clone(), taken));
            }
        }
    }
    if all.is_empty() {
        return None;
    }
    let pass = pass(host)?;
    let (bytes, to) = host.shared_buffer(total.min(MOST) as u64)?;
    let size = (all.len() as u64 * ANSWER).next_multiple_of(256);
    let (found, answers) = host.shared_buffer(size)?;
    // SAFETY: mappings of the sizes asked for; an open list, whose compute
    // state is set whole by what follows; live buffers, the ranges in them.
    unsafe {
        std::ptr::write_bytes(answers, 0, size as usize);
        list.SetComputeRootSignature(&pass.root);
        list.SetPipelineState(&pass.pso);
        list.SetComputeRootShaderResourceView(2, bytes.GetGPUVirtualAddress());
        list.SetComputeRootUnorderedAccessView(3, found.GetGPUVirtualAddress());
        let mut at = 0usize;
        for (i, (shadow, taken)) in all.iter().enumerate() {
            let n = taken.was.len();
            std::ptr::copy_nonoverlapping(taken.was.as_ptr(), to.add(at), n);
            std::ptr::copy_nonoverlapping(taken.new.as_ptr(), to.add(at + n), n);
            list.SetComputeRootShaderResourceView(1, shadow.buffer.GetGPUVirtualAddress());
            let constants = [taken.at as u32, (n / 4) as u32, at as u32, (at + n) as u32, (i as u64 * ANSWER) as u32];
            list.SetComputeRoot32BitConstants(0, 5, constants.as_ptr().cast(), 0);
            list.Dispatch((n / 4).div_ceil(64) as u32, 1, 1);
            at += 2 * n;
        }
        // The sends come after the looks.
        let mut seen = std::collections::HashSet::new();
        let done: Vec<D3D12_RESOURCE_BARRIER> = all
            .iter()
            .filter(|(s, _)| seen.insert(std::sync::Arc::as_ptr(s) as usize))
            .map(|(s, _)| transition(&s.buffer, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST))
            .collect();
        list.ResourceBarrier(&done);
    }
    Some(Looks {
        found: (found, answers as usize),
        _bytes: bytes,
        of: all.into_iter().map(|(s, t)| (s, t.was.len())).collect(),
    })
}

impl Looks {
    /// Whether `shadow` was looked at: it is left in `COPY_DEST`.
    pub(super) fn hold(&self, shadow: &Shadow) -> bool {
        self.of.iter().any(|(s, _)| s.id() == shadow.id())
    }

    /// The submission has run: what it found is counted.
    pub(super) fn read(self) {
        // SAFETY: the mapping holds an answer for each; the card has run.
        let answers = unsafe { std::slice::from_raw_parts(self.found.1 as *const [u32; 4], self.of.len()) };
        let mut stale = STALE.lock().unwrap_or_else(|e| e.into_inner());
        for ((shadow, bytes), &[words, at, card, cpu]) in self.of.iter().zip(answers) {
            let (start, len) = shadow.span();
            let e = stale.entry((start, len)).or_default();
            e.looked += 1;
            e.bytes += *bytes as u64;
            if words > 0 {
                e.sends += 1;
                e.words += u64::from(words);
                if e.some.len() < 4 {
                    e.some.push(format!("{words} words from +{at:#x} (card {card:#x}, sent {cpu:#x}; {})", known(shadow.id(), at as usize)));
                }
            }
        }
        line(&mut stale);
    }
}

/// Once a second: what was found since.
fn line(stale: &mut BTreeMap<(u64, u64), Stale>) {
    static SAID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let now = now_ms();
    let said = SAID.load(Ordering::Relaxed);
    if now < said + 1000 || SAID.compare_exchange(said, now, Ordering::Relaxed, Ordering::Relaxed).is_err() {
        return;
    }
    let all = std::mem::take(stale);
    let sum = |f: fn(&Stale) -> u64| all.values().map(f).sum::<u64>();
    tracing::info!(
        frame = crate::frame::current(),
        "up-copy check: {} pieces ({} KB) sent up again in {} buffers; {} of them put an older value over {} words the card had written",
        sum(|s| s.looked),
        sum(|s| s.bytes) >> 10,
        all.len(),
        sum(|s| s.sends),
        sum(|s| s.words),
    );
    let mut worst: Vec<_> = all.into_iter().filter(|(_, s)| s.words > 0).collect();
    worst.sort_by_key(|(_, s)| std::cmp::Reverse(s.words));
    for ((start, len), s) in worst.into_iter().take(12) {
        tracing::warn!(
            "up-copy check: older values sent up to {start:#x}+{len:#x}: {} words in {} pieces: {}",
            s.words,
            s.sends,
            s.some.join(", ")
        );
    }
}

/// Of a copy: the ranges shaders were declared to write (joined), and the
/// frame each piece was last fetched in.
#[derive(Default)]
struct Known {
    declared: Vec<(usize, usize)>,
    fetched: HashMap<usize, u64>,
}

static KNOWN: Mutex<BTreeMap<u64, Known>> = Mutex::new(BTreeMap::new());

/// Offsets `[from, to)` of a copy were declared as written.
pub(super) fn declared(copy: u64, from: usize, to: usize) {
    let mut known = KNOWN.lock().unwrap_or_else(|e| e.into_inner());
    let known = known.entry(copy).or_default();
    if !known.declared.iter().any(|r| r.0 <= from && to <= r.1) {
        join_ranges(&mut known.declared, [(from, to)]);
    }
}

/// A piece from offset `at` on of a copy was fetched.
pub(super) fn fetched(copy: u64, at: usize) {
    let mut known = KNOWN.lock().unwrap_or_else(|e| e.into_inner());
    known.entry(copy).or_default().fetched.insert(at / PIECE, crate::frame::current());
}

fn known(copy: u64, at: usize) -> String {
    let all = KNOWN.lock().unwrap_or_else(|e| e.into_inner());
    let Some(known) = all.get(&copy) else {
        return "no shader was declared to write the buffer".into();
    };
    if !touches(&known.declared, at, at + 4) {
        return "never declared as written there".into();
    }
    // (Fetched pieces begin where title memory's do; the copy may begin
    // inside one.)
    let last = [at / PIECE, (at / PIECE).saturating_sub(1)].iter().filter_map(|p| known.fetched.get(p)).max().copied();
    match last {
        Some(frame) => format!("declared; last fetched {} frames ago", crate::frame::current().saturating_sub(frame)),
        None => "declared; never fetched".into(),
    }
}

/// Reads of title memory through one copy while a flight in the air may
/// write it through another: (reader, writer) by their spans: how often, and
/// the last range they share.
type Crossed = BTreeMap<((u64, u64), (u64, u64)), (u64, usize, usize)>;
static CROSSED: Mutex<(u64, Crossed)> = Mutex::new((0, BTreeMap::new()));

/// Flights are in the air and `runs` are about to be prepared: which
/// refreshes of theirs read title memory that a flight may be writing through
/// another copy of it. Such a refresh takes what was there before the flight
/// (nothing waits for it), and no later one knows that it was too early but
/// for the landing's own write.
pub(super) fn reads(runs: &[&HostList]) {
    let mut writes: Vec<(std::sync::Arc<Shadow>, usize, usize)> = Vec::new();
    let mut all = Vec::new();
    for item in &super::line().line {
        if let Late::Run(flight) = item {
            for (shadow, ranges) in &flight.back {
                writes.extend(ranges.iter().map(|&(a, b)| (shadow.clone(), a, b)));
                join_ranges(&mut all, ranges.iter().copied());
            }
        }
    }
    if writes.is_empty() {
        return;
    }
    let mut crossed = CROSSED.lock().unwrap_or_else(|e| e.into_inner());
    for list in runs {
        for step in &list.deferred().copies {
            let Step::Refresh(shadow, at, len) = step else {
                continue;
            };
            let (from, to) = (*at, *at + *len);
            if !touches(&all, from, to) {
                continue;
            }
            for (writer, a, b) in &writes {
                if *a < to && from < *b && !std::sync::Arc::ptr_eq(writer, shadow) {
                    let e = crossed.1.entry((shadow.span(), writer.span())).or_default();
                    *e = (e.0 + 1, (*a).max(from), (*b).min(to));
                }
            }
        }
    }
    let now = now_ms();
    if now < crossed.0 + 4000 {
        return;
    }
    crossed.0 = now;
    let mut all: Vec<_> = std::mem::take(&mut crossed.1).into_iter().collect();
    all.sort_by_key(|(_, e)| std::cmp::Reverse(e.0));
    for (((r, rn), (w, wn)), (n, a, b)) in all.into_iter().take(12) {
        tracing::warn!(
            "up-copy check: {n} refreshes of {r:#x}+{rn:#x} read title memory that flights in the air may write through              {w:#x}+{wn:#x} (the last at {a:#x}..{b:#x})"
        );
    }
}

/// The title's buffer copies that are out of their place in a submission:
/// by what is wrong, bytes and destination: how often, the last source, and
/// (for shader writes) how much was declared.
type Misplaced = BTreeMap<(&'static str, usize, usize), (u64, usize, usize)>;
static MISPLACED: Mutex<(u64, Misplaced)> = Mutex::new((0, BTreeMap::new()));

/// A submission is about to be prepared: its title buffer copies are done on
/// the CPU then, all of them before the first of its draws runs. Where that
/// is not the order of its lists: what an earlier draw of it reads of the
/// destination, or shaders of it write there, or an earlier copy of it put
/// there for what lies between; and a source that shaders of it write first.
pub(super) fn order(runs: &[&HostList]) {
    let (mut read, mut written, mut copied): (Vec<(usize, usize)>, Vec<(usize, usize)>, Vec<(usize, usize)>) =
        (Vec::new(), Vec::new(), Vec::new());
    let meets = |all: &[(usize, usize)], from: usize, to: usize| all.iter().rev().find(|r| r.0 < to && from < r.1).copied();
    let mut misplaced = MISPLACED.lock().unwrap_or_else(|e| e.into_inner());
    for list in runs {
        for step in &list.deferred().copies {
            match step {
                Step::Refresh(_, at, len) => read.push((*at, *at + *len)),
                Step::GpuWrite(_, at, len) => written.push((*at, *at + *len)),
                &Step::Copy(dst, src, len) => {
                    let mut say = |what: &'static str, range: (usize, usize)| {
                        let e = misplaced.1.entry((what, len, dst)).or_default();
                        *e = (e.0 + 1, src, range.1 - range.0);
                    };
                    if let Some(r) = meets(&written, src, src + len) {
                        say("its source is declared written earlier in the submission", r);
                    }
                    if let Some(r) = meets(&written, dst, dst + len) {
                        say("its destination is declared written earlier in the submission", r);
                    }
                    if let Some(r) = meets(&copied, dst, dst + len) {
                        say("its destination is copied to earlier in the submission", r);
                    } else if let Some(r) = meets(&read, dst, dst + len) {
                        say("its destination is read earlier in the submission", r);
                    }
                    copied.push((dst, dst + len));
                }
                _ => {}
            }
        }
    }
    let now = now_ms();
    if now < misplaced.0 + 4000 {
        return;
    }
    misplaced.0 = now;
    let mut all: Vec<_> = std::mem::take(&mut misplaced.1).into_iter().collect();
    all.sort_by_key(|(_, e)| std::cmp::Reverse(e.0));
    let mut kinds: BTreeMap<&str, (u64, u64)> = BTreeMap::new();
    for ((what, len, _), (n, ..)) in &all {
        let e = kinds.entry(what).or_default();
        *e = (e.0 + n, e.1 + n * *len as u64);
    }
    for (what, (n, bytes)) in kinds {
        tracing::warn!("up-copy check, copies out of place: {n} ({bytes} bytes): {what}");
    }
    for ((what, len, dst), (n, src, declared)) in all.into_iter().take(16) {
        tracing::warn!("up-copy check, a copy out of place: {n} times {len:#x} bytes from {src:#x} to {dst:#x}: {what} ({declared:#x} bytes there)");
    }
}

/// A run of words that differ at rest.
struct Differs {
    shadow: std::sync::Arc<Shadow>,
    at: usize,
    len: usize,
    words: u32,
}

/// Nothing is in the air, and nothing is sent meanwhile (the submission
/// lock is held): compares every copy in the card's memory with its CPU
/// side, every few seconds.
pub(super) fn at_rest(host: &Host) {
    static LAST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let now = now_ms();
    if now < LAST.load(Ordering::Relaxed) + 4000 {
        return;
    }
    let began = std::time::Instant::now();
    if look(host).is_none() {
        tracing::warn!("up-copy check: the look at rest could not be made");
    }
    look_for_stale();
    tracing::info!(ms = began.elapsed().as_millis(), "up-copy check: the look at rest is done");
    LAST.store(now_ms(), Ordering::Relaxed);
}

/// At rest: copies of title memory, whatever reads them (meshes too), that
/// are not what title memory holds although nothing told that it changed: a
/// refresh will not bring them up to date.
fn look_for_stale() {
    const MOST: usize = 1 << 30;
    let mut by: BTreeMap<(u64, u64), (bool, u64, usize, Vec<String>)> = BTreeMap::new();
    let compared = crate::shadow::stale_pages(MOST, &mut |shadow, page, words, (at, copy, title)| {
        let e = by.entry(shadow.span()).or_insert((shadow.carded(), 0, 0, Vec::new()));
        e.1 += words as u64;
        e.2 += 1;
        if e.3.len() < 4 {
            e.3.push(format!("{words} words in the 64 KiB at {page:#x} (the first at {at:#x}: the copy {copy:#x}, title memory {title:#x})"));
        }
    });
    tracing::info!(
        "up-copy check at rest: {} MB of copies held for unchanged compared with title memory, {} copies are stale",
        compared >> 20,
        by.len()
    );
    let mut by: Vec<_> = by.into_iter().collect();
    by.sort_by_key(|(_, e)| std::cmp::Reverse(e.1));
    for ((start, len), (carded, words, pages, some)) in by.into_iter().take(16) {
        tracing::warn!(
            "up-copy check at rest, stale: {start:#x}+{len:#x} ({}): {words} words in {pages} pages are not title memory's, and nothing \
             told of it: {}",
            if carded { "on the card" } else { "in PC memory" },
            some.join("; ")
        );
    }
}

/// Sends `list` (recorded by `f`) and waits for it.
fn run(host: &Host, f: impl FnOnce(&ID3D12GraphicsCommandList)) -> Option<()> {
    let (allocator, list) = host.make_prelude()?;
    // SAFETY: objects of our own, waited for before they go.
    unsafe {
        allocator.Reset().ok()?;
        list.Reset(&allocator, None).ok()?;
        f(&list);
        list.Close().ok()?;
        let fence: ID3D12Fence = host.device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
        let event = CreateEventW(None, false, false, None).ok()?;
        host.queue.ExecuteCommandLists(&[Some(list.cast().ok()?)]);
        let done = host.queue.Signal(&fence, 1).and_then(|()| fence.SetEventOnCompletion(1, event));
        if done.is_ok() {
            WaitForSingleObject(event, INFINITE);
        }
        let _ = CloseHandle(event);
        done.ok()
    }
}

/// The words that differ.
fn look(host: &Host) -> Option<u64> {
    let pass = pass(host)?;
    let copies = crate::shadow::carded_copies();
    // (copy, offset, bytes)
    let mut pieces: Vec<(std::sync::Arc<Shadow>, usize, usize)> = Vec::new();
    for shadow in &copies {
        // What has not gone up yet differs rightly.
        let dirty = shadow.card_dirty_now();
        let len = shadow.span().1 as usize & !3;
        for at in (0..len).step_by(PIECE) {
            let n = PIECE.min(len - at);
            if !dirty.iter().any(|&(a, b)| a < at + n && b > at) {
                pieces.push((shadow.clone(), at, n));
            }
        }
    }
    if pieces.is_empty() {
        return Some(0);
    }
    let size = (pieces.len() as u64 * ANSWER).next_multiple_of(256);
    let (found, answers) = host.shared_buffer(size)?;
    // SAFETY: a mapping of that size.
    unsafe { std::ptr::write_bytes(answers, 0, size as usize) };
    run(host, |list| {
        // SAFETY: an open list; live buffers (held by `pieces`).
        unsafe {
            list.SetComputeRootSignature(&pass.root);
            list.SetPipelineState(&pass.pso);
            list.SetComputeRootUnorderedAccessView(3, found.GetGPUVirtualAddress());
            let mut bound = 0;
            for (i, (shadow, at, n)) in pieces.iter().enumerate() {
                let Some((staging, ..)) = shadow.card_sides() else {
                    continue;
                };
                if bound != shadow.id() {
                    bound = shadow.id();
                    list.SetComputeRootShaderResourceView(1, shadow.buffer.GetGPUVirtualAddress());
                    list.SetComputeRootShaderResourceView(2, staging.GetGPUVirtualAddress());
                }
                let at = *at as u32;
                let constants = [at, (*n / 4) as u32, at, at, (i as u64 * ANSWER) as u32];
                list.SetComputeRoot32BitConstants(0, 5, constants.as_ptr().cast(), 0);
                list.Dispatch((*n / 4).div_ceil(64) as u32, 1, 1);
            }
        }
    })?;
    // SAFETY: the mapping holds an answer for each; the card has run.
    let answers = unsafe { std::slice::from_raw_parts(answers as *const [u32; 4], pieces.len()) };
    let differs: Vec<Differs> = pieces
        .iter()
        .zip(answers)
        .filter(|(_, a)| a[0] > 0)
        .map(|((shadow, at, len), a)| Differs {
            shadow: shadow.clone(),
            at: *at,
            len: *len,
            words: a[0],
        })
        .collect();
    let bytes: u64 = pieces.iter().map(|p| p.2 as u64).sum();
    let words: u64 = differs.iter().map(|d| u64::from(d.words)).sum();
    tracing::info!(
        frame = crate::frame::current(),
        "up-copy check at rest: {} copies, {} MB; {words} words in {} pieces of them differ from the CPU's side",
        copies.len(),
        bytes >> 20,
        differs.len()
    );
    if differs.is_empty() {
        return Some(0);
    }
    // Per copy, what stands where, of a few pieces.
    const SHOWN: usize = 64;
    let shown = &differs[..differs.len().min(SHOWN)];
    let (held, data) = host.shared_buffer((SHOWN * PIECE) as u64)?;
    run(host, |list| {
        for (i, d) in shown.iter().enumerate() {
            // SAFETY: an open list; live buffers, the ranges in them.
            unsafe { list.CopyBufferRegion(&held, (i * PIECE) as u64, &d.shadow.buffer, d.at as u64, d.len as u64) };
        }
    })?;
    let mut by: BTreeMap<(u64, u64), (u64, usize, Vec<String>)> = BTreeMap::new();
    for d in &differs {
        let e = by.entry(d.shadow.span()).or_default();
        e.0 += u64::from(d.words);
        e.1 += 1;
    }
    for (i, d) in shown.iter().enumerate() {
        let (start, _) = d.shadow.span();
        // SAFETY: the mapping; the copy into it has run.
        let card = unsafe { std::slice::from_raw_parts(data.add(i * PIECE).cast_const(), d.len) };
        let cpu = d.shadow.card_held(d.at, d.len);
        let mut title = vec![0u8; d.len];
        let mapped = read_into((start as usize + d.at) as *const u8, &mut title);
        // Of the words that differ: title memory holds the CPU side's
        // value, the card's, another one.
        let (mut as_cpu, mut as_card, mut other) = (0, 0, 0);
        let mut first = None;
        let (mut low, mut high) = (usize::MAX, 0);
        for at in (0..d.len & !3).step_by(4) {
            let word = |b: &[u8]| u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]]);
            let (c, h, t) = (word(card), word(&cpu), word(&title));
            if c == h {
                continue;
            }
            (low, high) = (low.min(at), at);
            first.get_or_insert((at, c, h, t));
            if t == h {
                as_cpu += 1;
            } else if t == c {
                as_card += 1;
            } else {
                other += 1;
            }
        }
        let Some((at, c, h, t)) = first else {
            continue;
        };
        if let Some(e) = by.get_mut(&d.shadow.span())
            && e.2.len() < 6
        {
            e.2.push(format!(
                "{} words in +{:#x}..{:#x} (title memory: {as_cpu} as the CPU's side, {as_card} as the card, {other} neither{}; first: \
                 card {c:#x}, CPU {h:#x}, title {t:#x}; {})",
                d.words,
                d.at + low,
                d.at + high + 4,
                if mapped { "" } else { ", not all mapped" },
                known(d.shadow.id(), d.at + at)
            ));
        }
    }
    let mut by: Vec<_> = by.into_iter().collect();
    by.sort_by_key(|(_, e)| std::cmp::Reverse(e.0));
    for ((start, len), (words, pieces, some)) in by.into_iter().take(16) {
        tracing::warn!(
            "up-copy check at rest: {start:#x}+{len:#x}: {words} words in {pieces} pieces are not the CPU side's: {}",
            some.join("; ")
        );
    }
    Some(words)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn notes_are_cut_to_whole_words() {
        // Bytes 5..18 of a copy: the words at 8 and 12.
        let (was, new): (Vec<u8>, Vec<u8>) = ((5..18).collect(), (105..118).collect());
        note(u64::MAX, 5, &was, &new);
        let noted = NOTED.lock().unwrap_or_else(|e| e.into_inner()).remove(&u64::MAX).expect("noted");
        let [t] = &noted[..] else { panic!("one piece") };
        assert_eq!((t.at, &t.was[..], &t.new[..]), (8, &was[3..11], &new[3..11]));
    }

    /// The tests below want processes of their own with the relay on: each
    /// `VICTORY_RELAY=2 VICTORY_RELAY_COPY=1 VICTORY_UPCOPY_CHECK=1 cargo
    /// test -p victory-gpu --lib upcheck -- --ignored --test-threads=1
    /// <name>`.
    /// The card-copy tests request a 1 MiB minimum so this 4 MiB buffer
    /// goes to the card even when its default minimum is 16 MiB.
    const LEN: usize = 4 << 20;

    /// Title memory of our own (write-watched, or a view of a section, which
    /// is guarded), and the card copy of the title buffer it is.
    fn title(resource: usize, view: bool) -> (&'static mut [u8], usize, std::sync::Arc<Shadow>) {
        let all = title_of(resource, view, LEN);
        assert!(all.2.carded(), "the copy is in the card's memory");
        all
    }

    fn title_of(resource: usize, view: bool, len: usize) -> (&'static mut [u8], usize, std::sync::Arc<Shadow>) {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEM_COMMIT, MEM_RESERVE, MapViewOfFile, PAGE_READWRITE, VirtualAlloc,
        };
        const WRITE_WATCH: u32 = 0x20_0000;
        // SAFETY: memory of our own, never freed.
        let base = unsafe {
            if view {
                let section = CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE, 0, len as u32, std::ptr::null());
                assert!(!section.is_null());
                MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, len).Value.cast::<u8>()
            } else {
                VirtualAlloc(std::ptr::null(), len, MEM_RESERVE | MEM_COMMIT | WRITE_WATCH, PAGE_READWRITE).cast::<u8>()
            }
        };
        assert!(!base.is_null());
        crate::shadow::register(resource, base as u64, len as u64);
        let shadow = crate::shadow::find(base as u64, 64).expect("a copy");
        assert!(super::super::relay_exact() && super::super::relay_copy(), "set VICTORY_RELAY=2 VICTORY_RELAY_COPY=1");
        // SAFETY: `len` bytes just committed.
        (unsafe { std::slice::from_raw_parts_mut(base, len) }, base as usize, shadow)
    }

    /// Work for the card that takes some milliseconds.
    struct Slow(ID3D12Resource, ID3D12Resource);

    impl Slow {
        fn new(host: &Host) -> Slow {
            Slow(host.uav_buffer(64 << 20).unwrap(), host.uav_buffer(64 << 20).unwrap())
        }

        fn record(&self, list: &HostList, times: usize) {
            // SAFETY: an open list; live buffers.
            unsafe {
                for _ in 0..times {
                    list.list.CopyBufferRegion(&self.1, 0, &self.0, 0, 64 << 20);
                    list.list.ResourceBarrier(&[
                        transition(&self.1, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON),
                    ]);
                }
            }
        }
    }

    /// A shader's stand-in: `bytes` go to offset `at` of the copy.
    fn write(host: &Host, list: &HostList, shadow: &Shadow, at: usize, bytes: &[u8]) {
        let (upload, data) = host.upload_buffer(bytes.len() as u64).unwrap();
        // SAFETY: the mapping; an open list and live buffers.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), data, bytes.len());
            list.list.CopyBufferRegion(&shadow.buffer, at as u64, &upload, 0, bytes.len() as u64);
            list.list
                .ResourceBarrier(&[transition(&shadow.buffer, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON)]);
        }
        list.keep(&upload);
    }

    /// What the card's side holds, nothing being in the air.
    fn card(host: &Host, shadow: &Shadow) -> Vec<u8> {
        let (held, data) = host.shared_buffer(LEN as u64).unwrap();
        // SAFETY: an open list; live buffers.
        run(host, |list| unsafe { list.CopyBufferRegion(&held, 0, &shadow.buffer, 0, LEN as u64) }).unwrap();
        // SAFETY: the mapping; the copy has run.
        unsafe { std::slice::from_raw_parts(data, LEN) }.to_vec()
    }

    /// (e) of the list in `Host::relay_card_back`'s fetch: a flight's result
    /// is fetched while the next flight is at the same bytes, which ends with
    /// what it found. Without `VICTORY_FETCH_CHECK` title memory keeps what
    /// was in between, and the next refresh of the 64 KiB sends it up.
    #[test]
    #[ignore = "wants VICTORY_RELAY=2 VICTORY_RELAY_COPY=1 VICTORY_FETCH_CHECK=1 VICTORY_CARD_BUFFERS_MIN_MB=1 and a process of its own"]
    fn a_result_caught_in_between_is_put_right() {
        let Some(host) = host() else {
            return;
        };
        let (title, start, shadow) = title(0x3333, false);
        let slow = Slow::new(host);
        let (mut wrong, mut undone) = (0, 0);
        let mut lists = Vec::new();
        const ROUNDS: usize = 12;
        for round in 0..ROUNDS {
            let result: Vec<u8> = (0..256).map(|i| (round * 17 + i + 1) as u8).collect();
            let between = vec![0u8; 256];
            let a = host.new_list().unwrap();
            shadow.refresh(&a, start as u64, LEN as u64);
            shadow.gpu_writes(&a, start as u64, LEN as u64);
            write(host, &a, &shadow, 1024, &result);
            a.close();
            let b = host.new_list().unwrap();
            shadow.gpu_writes(&b, start as u64, LEN as u64);
            write(host, &b, &shadow, 1024, &between);
            slow.record(&b, 60);
            write(host, &b, &shadow, 1024, &result);
            b.close();
            host.execute_own(&a);
            host.execute_own(&b);
            settle_now(WHY_SYNC);
            wrong += usize::from(title[1024..1280] != result[..]);
            // The title writes beside it: the refresh takes the 64 KiB.
            title[4096] = round as u8 + 1;
            let c = host.new_list().unwrap();
            shadow.refresh(&c, start as u64, LEN as u64);
            c.close();
            host.execute_own(&c);
            settle_now(WHY_SYNC);
            let holds = card(host, &shadow);
            assert_eq!(holds[4096], round as u8 + 1, "the title's write is on the card");
            undone += usize::from(holds[1024..1280] != result[..]);
            lists.extend([a, b, c]);
        }
        eprintln!(
            "{wrong} of {ROUNDS} results wrong in title memory, {undone} undone on the card; pieces checked, suspect, put right: {:?}",
            super::torn::STATS.iter().map(|s| s.load(Ordering::Relaxed)).collect::<Vec<_>>()
        );
        assert_eq!(look(host), Some(0));
        assert_eq!((wrong, undone), (0, 0), "without VICTORY_FETCH_CHECK=1 most are wrong");
    }

    /// What nobody declared: a shader writes a piece through a view that was
    /// not known. The title writes another byte of the piece: it goes up
    /// whole, with title memory's older bytes, and the check says so.
    #[test]
    #[ignore = "wants VICTORY_RELAY=2 VICTORY_RELAY_COPY=1 VICTORY_UPCOPY_CHECK=1 VICTORY_CARD_BUFFERS_MIN_MB=1 and a process of its own"]
    fn the_check_finds_an_undeclared_write_undone() {
        let Some(host) = host() else {
            return;
        };
        assert!(on(), "set VICTORY_UPCOPY_CHECK=1");
        let (title, start, shadow) = title(0x4444, false);
        let a = host.new_list().unwrap();
        shadow.refresh(&a, start as u64, LEN as u64);
        shadow.gpu_writes(&a, start as u64, 1 << 16);
        write(host, &a, &shadow, (5 << 16) + 512, &[0x77; 64]);
        a.close();
        host.execute_own(&a);
        settle_now(WHY_SYNC);
        assert_eq!(card(host, &shadow)[(5 << 16) + 512], 0x77);
        assert!(title[(5 << 16) + 512] == 0, "it never comes back");
        assert_eq!(look(host), Some(16), "the look at rest finds it");
        title[(5 << 16) + 9000] = 1;
        let b = host.new_list().unwrap();
        shadow.refresh(&b, start as u64, LEN as u64);
        b.close();
        host.execute_own(&b);
        settle_now(WHY_SYNC);
        let undone = card(host, &shadow)[(5 << 16) + 512] == 0;
        let stale = STALE.lock().unwrap_or_else(|e| e.into_inner());
        let found = stale.get(&shadow.span()).map_or(0, |s| s.words);
        eprintln!("undone on the card: {undone}; the check found {found} words");
        assert_eq!((undone, found), (true, 16));
    }

    /// A shader that writes through views it picks: its buffer is found when
    /// it comes to it in turn, and what it writes from then on is landed.
    #[test]
    #[ignore = "wants VICTORY_RELAY=2 VICTORY_RELAY_COPY=1 VICTORY_PICKED_WRITES=2 VICTORY_PICKED_MB=1 VICTORY_CARD_BUFFERS_MIN_MB=1 and a process of its own"]
    fn writes_through_picked_views_come_back() {
        let Some(host) = host() else {
            return;
        };
        assert!(crate::picked::on(), "set VICTORY_PICKED_WRITES=2");
        let (title, start, shadow) = title(0x6666, true);
        crate::picked::shown(vec![(shadow.clone(), start as u64, (start + LEN) as u64)], true);
        let at = (37 << 16) + 512;
        let mut lists = Vec::new();
        let mut back = Vec::new();
        for round in 0..12u8 {
            // Turns go by frames.
            crate::frame::test_advance();
            title[(37 << 16) + 9000] = round + 1;
            let list = host.new_list().unwrap();
            shadow.refresh(&list, start as u64, LEN as u64);
            crate::picked::runs_in(&list);
            write(host, &list, &shadow, at, &[0x40 + round; 64]);
            list.close();
            host.execute_own(&list);
            settle_now(WHY_SYNC);
            back.push(title[at..at + 64] == [0x40 + round; 64]);
            // The title's own write beside it is on the card, and the
            // shader's stays, once its buffer is known.
            let next = host.new_list().unwrap();
            shadow.refresh(&next, start as u64, LEN as u64);
            next.close();
            host.execute_own(&next);
            settle_now(WHY_SYNC);
            if *back.last().unwrap() {
                let holds = card(host, &shadow);
                assert_eq!((holds[(37 << 16) + 9000], holds[at]), (round + 1, 0x40 + round));
                assert_eq!(look(host), Some(0));
            }
            lists.extend([list, next]);
        }
        eprintln!("came back, by round: {back:?}");
        let first = back.iter().position(|b| *b).expect("it is found");
        assert!(first <= 6 && back[first..].iter().all(|b| *b), "{back:?}");
    }

    /// The title copies another value to the same bytes in each of three
    /// submissions, and each has a draw (a copy here) that reads them. The
    /// card is busy meanwhile, so all three are prepared before the first
    /// runs: each must read its own.
    fn each_reads_its_own(len: usize, resource: usize, carded: bool) {
        let Some(host) = host() else {
            return;
        };
        let (title, start, shadow) = title_of(resource, true, len);
        assert_eq!(shadow.carded(), carded, "where the copy is");
        let slow = Slow::new(host);
        let (out, out_data) = host.shared_buffer(1 << 16).unwrap();
        let mut lists = Vec::new();
        const AT: usize = 8192;
        // All there once.
        let all = host.new_list().unwrap();
        shadow.refresh(&all, start as u64, len as u64);
        all.close();
        host.execute_own(&all);
        settle_now(WHY_SYNC);
        lists.push(all);
        let mut wrong = Vec::new();
        for round in 0..6u8 {
            for v in 0..3u8 {
                title[65_536 + usize::from(v) * 256..][..256].fill(round * 16 + v + 1);
            }
            let busy = host.new_list().unwrap();
            slow.record(&busy, 40);
            busy.close();
            host.execute_own(&busy);
            lists.push(busy);
            for v in 0..3usize {
                let list = host.new_list().unwrap();
                list.defer_copy((start + AT) as *mut u8, (start + 65_536 + v * 256) as *const u8, 256);
                shadow.refresh(&list, (start + AT) as u64, 256);
                // SAFETY: an open list; live buffers.
                unsafe {
                    list.list.CopyBufferRegion(&out, (v * 256) as u64, &shadow.buffer, AT as u64, 256);
                }
                list.close();
                host.execute_own(&list);
                lists.push(list);
            }
            settle_now(WHY_SYNC);
            // SAFETY: a mapping of 64 KiB; the copies into it have run.
            let read = unsafe { std::slice::from_raw_parts(out_data, 768) };
            for v in 0..3u8 {
                let got = read[usize::from(v) * 256];
                if got != round * 16 + v + 1 {
                    wrong.push((round, v, got - round * 16));
                }
            }
        }
        eprintln!("read another submission's value (round, submission, whose + 1): {wrong:?}");
        let stale = STALE.lock().unwrap_or_else(|e| e.into_inner());
        let found = stale.get(&shadow.span()).map_or(0, |s| s.words);
        eprintln!("the check at sends found {found} words");
        assert!(wrong.is_empty());
        assert_eq!(found, 0);
        // With the debug layer (`VICTORY_D3D12_DEBUG=1`, diagnostics).
        assert_eq!(host.log_messages(), None);
    }

    #[test]
    #[ignore = "wants VICTORY_RELAY=2 VICTORY_RELAY_COPY=1 VICTORY_CARD_BUFFERS_MIN_MB=1 and a process of its own"]
    fn each_submission_reads_its_own_copy_on_the_card() {
        each_reads_its_own(LEN, 0x7777, true);
    }

    #[test]
    #[ignore = "wants VICTORY_RELAY=2 VICTORY_RELAY_COPY=1 VICTORY_CARD_ALL=1 and a process of its own"]
    fn each_submission_reads_its_own_copy_in_pc_memory() {
        each_reads_its_own(256 << 10, 0x8888, std::env::var_os("VICTORY_CARD_ALL").is_some_and(|v| v == "1"));
    }

    /// What a draw of `list` reads at `at`: comes to `out`.
    fn read_out(list: &HostList, shadow: &std::sync::Arc<Shadow>, at: usize, n: usize, out: (&ID3D12Resource, u64)) {
        shadow.refresh(list, shadow.span().0 + at as u64, n as u64);
        // SAFETY: an open list; live buffers.
        unsafe {
            list.list.CopyBufferRegion(out.0, out.1, &shadow.buffer, at as u64, n as u64);
            list.list
                .ResourceBarrier(&[transition(&shadow.buffer, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON)]);
        }
    }

    /// Copies, shader writes and reads in random order, within lists and
    /// across submissions in the air, the title's CPU writing sources in
    /// between: every read sees what stands before it, and in the end title
    /// memory and the copies hold the last of everything.
    fn in_order(resource: usize, sizes: [usize; 3]) {
        let Some(host) = host() else {
            return;
        };
        let buffers: Vec<_> = sizes.iter().enumerate().map(|(i, &n)| title_of(resource + i, i == 0, n)).collect();
        let slow = Slow::new(host);
        let (out, out_data) = host.shared_buffer(1 << 20).unwrap();
        let mut seed = 0x9e37_79b9_7f4a_7c15u64 ^ resource as u64;
        let mut next = move |n: usize| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as usize % n
        };
        // Of each buffer: the first 4 KiB are the title's own (sources), the
        // next 12 are written by copies and shaders.
        const OWN: usize = 4096;
        const ALL: usize = 16384;
        // (Of the third only the first 4 KiB: shaders never write it.)
        let mut model = [vec![0u8; ALL], vec![0u8; ALL], vec![0u8; ALL]];
        let mut titles: Vec<&'static mut [u8]> = Vec::new();
        let mut made = Vec::new();
        for (title, start, shadow) in buffers {
            titles.push(title);
            made.push((start, shadow));
        }
        let mut lists = Vec::new();
        for (start, shadow) in &made {
            let all = host.new_list().unwrap();
            shadow.refresh(&all, *start as u64, ALL as u64);
            all.close();
            host.execute_own(&all);
            lists.push(all);
        }
        settle_now(WHY_SYNC);
        for round in 0..150 {
            // (offset in `out`, what was expected, where from)
            let mut expected: Vec<(usize, Vec<u8>, String)> = Vec::new();
            let mut told = String::new();
            for submission in 0..1 + next(5) {
                for _ in 0..next(6) {
                    let (b, at) = (next(3), next(OWN / 64) * 64);
                    let value = next(255) as u8 + 1;
                    // (A copy with one side is what submissions in the air
                    // read: the title's CPU waits for them, as it does.)
                    if !made[b].1.carded() {
                        settle_now(WHY_SYNC);
                    }
                    titles[b][at..at + 64].fill(value);
                    model[b][at..at + 64].fill(value);
                }
                let list = host.new_list().unwrap();
                if next(3) == 0 {
                    slow.record(&list, 4 + next(12));
                }
                for op in 0..1 + next(10) {
                    let word = |next: &mut dyn FnMut(usize) -> usize| OWN + next((ALL - OWN - 256) / 4) * 4;
                    match next(5) {
                        0 | 1 => {
                            let (d, s) = (next(2), next(3));
                            let n = 4 * (1 + next(64));
                            let to = word(&mut next);
                            let from = if s == 2 || next(2) == 0 { next((OWN - 256) / 4) * 4 } else { word(&mut next) };
                            if d == s && from < to + n && to < from + n {
                                continue;
                            }
                            assert!(list.place_copy(made[d].0 + to, made[s].0 + from, n), "done in the list");
                            let bytes = model[s][from..from + n].to_vec();
                            model[d][to..to + n].copy_from_slice(&bytes);
                            told += &format!("\n  s{submission} op{op}: copy {d}+{to:#x} <- {s}+{from:#x} {n:#x}");
                        }
                        2 => {
                            let (b, at, n) = (next(2), word(&mut next), 4 * (1 + next(32)));
                            let value = next(255) as u8 + 1;
                            let (start, shadow) = &made[b];
                            shadow.refresh(&list, (*start + at) as u64, n as u64);
                            shadow.gpu_writes(&list, (*start + at) as u64, n as u64);
                            write(host, &list, shadow, at, &vec![value; n]);
                            model[b][at..at + n].fill(value);
                            told += &format!("\n  s{submission} op{op}: shader {b}+{at:#x} {n:#x} = {value:#x}");
                        }
                        _ => {
                            let (b, n) = (next(2), 256);
                            let at = if next(4) == 0 { next((OWN - 256) / 4) * 4 } else { word(&mut next) };
                            let place = expected.len() * 256;
                            read_out(&list, &made[b].1, at, n, (&out, place as u64));
                            expected.push((place, model[b][at..at + n].to_vec(), format!("s{submission} op{op}: read {b}+{at:#x}")));
                        }
                    }
                }
                list.close();
                host.execute_own(&list);
                lists.push(list);
            }
            settle_now(WHY_SYNC);
            for (at, bytes, what) in expected {
                // SAFETY: inside the mapping; the copies into it have run.
                let got = unsafe { std::slice::from_raw_parts(out_data.add(at), bytes.len()) };
                if let Some(k) = (0..bytes.len()).find(|&k| got[k] != bytes[k]) {
                    panic!("round {round}, {what}: at +{k:#x} {:#x}, not {:#x}{told}", got[k], bytes[k]);
                }
            }
            for (b, title) in titles.iter().enumerate() {
                if let Some(k) = (0..ALL).find(|&k| title[k] != model[b][k]) {
                    panic!("round {round}: title memory {b}+{k:#x} is {:#x}, not {:#x}{told}", title[k], model[b][k]);
                }
            }
            // Brought up to date by the next, the copies hold it too.
            let list = host.new_list().unwrap();
            for (b, (_, shadow)) in made.iter().enumerate() {
                read_out(&list, shadow, 0, ALL, (&out, (b * ALL) as u64));
            }
            list.close();
            host.execute_own(&list);
            lists.push(list);
            settle_now(WHY_SYNC);
            for (b, model) in model.iter().enumerate() {
                // SAFETY: as above.
                let got = unsafe { std::slice::from_raw_parts(out_data.add(b * ALL), ALL) };
                if let Some(k) = (0..ALL).find(|&k| got[k] != model[k]) {
                    panic!("round {round}: the copy {b}+{k:#x} is {:#x}, not {:#x}{told}", got[k], model[k]);
                }
            }
            if on() {
                assert_eq!(look(host), Some(0), "round {round}");
            }
        }
        eprintln!(
            "copies at their place, bytes, out of host copies, into textures: {:?}",
            super::place::STATS.iter().map(|s| s.load(Ordering::Relaxed)).collect::<Vec<_>>()
        );
    }

    /// Two buffers with two sides.
    #[test]
    #[ignore = "wants VICTORY_RELAY=2 VICTORY_RELAY_COPY=1 and a process of its own"]
    fn copies_stand_where_the_list_has_them_on_the_card() {
        in_order(0x9100, [LEN, LEN, LEN]);
    }

    /// One in the card's memory, one in the PC's (one side without
    /// `VICTORY_CARD_ALL`).
    #[test]
    #[ignore = "wants VICTORY_RELAY=2 VICTORY_RELAY_COPY=1 and a process of its own"]
    fn copies_stand_where_the_list_has_them_mixed() {
        in_order(0x9200, [LEN, 256 << 10, 256 << 10]);
    }

    #[test]
    #[ignore = "wants VICTORY_RELAY=2 VICTORY_RELAY_COPY=1 and a process of its own"]
    fn copies_stand_where_the_list_has_them_in_pc_memory() {
        in_order(0x9300, [256 << 10, 256 << 10, LEN]);
    }

    /// As the title records it: copy, draw, copy, draw in one list; a source
    /// a shader writes first; a counter copied to zero and counted up.
    #[test]
    #[ignore = "wants VICTORY_RELAY=2 VICTORY_RELAY_COPY=1 and a process of its own"]
    fn copy_draw_copy_draw() {
        let Some(host) = host() else {
            return;
        };
        let (title, start, shadow) = title_of(0x9400, true, LEN);
        let (out, out_data) = host.shared_buffer(1 << 16).unwrap();
        let all = host.new_list().unwrap();
        shadow.refresh(&all, start as u64, LEN as u64);
        all.close();
        host.execute_own(&all);
        settle_now(WHY_SYNC);
        title[0..64].fill(1);
        title[64..128].fill(2);
        title[128..192].fill(0);
        let list = host.new_list().unwrap();
        const D: usize = 0x10000;
        const S: usize = 0x20000;
        assert!(list.place_copy(start + D, start, 64));
        read_out(&list, &shadow, D, 64, (&out, 0));
        assert!(list.place_copy(start + D, start + 64, 64));
        read_out(&list, &shadow, D, 64, (&out, 64));
        // A shader writes S, and it is copied on.
        shadow.gpu_writes(&list, (start + S) as u64, 64);
        write(host, &list, &shadow, S, &[7; 64]);
        assert!(list.place_copy(start + D + 64, start + S, 64));
        read_out(&list, &shadow, D + 64, 64, (&out, 128));
        // Zero, counted up, taken away.
        assert!(list.place_copy(start + D + 128, start + 128, 64));
        shadow.gpu_writes(&list, (start + D + 128) as u64, 64);
        write(host, &list, &shadow, D + 128, &[5; 64]);
        assert!(list.place_copy(start + D + 192, start + D + 128, 64));
        read_out(&list, &shadow, D + 192, 64, (&out, 192));
        // What a copy into a texture takes of S here.
        let between = list.place_source(start + S, 64).expect("shaders write it");
        shadow.gpu_writes(&list, (start + S) as u64, 64);
        write(host, &list, &shadow, S, &[9; 64]);
        // SAFETY: an open list; live buffers.
        unsafe { list.list.CopyBufferRegion(&out, 256, &between, 0, 64) };
        assert!(list.place_source(start + (1 << 20), 64).is_some(), "the buffer is written");
        // And zero for the next frame.
        assert!(list.place_copy(start + D + 128, start + 128, 64));
        list.close();
        host.execute_own(&list);
        settle_now(WHY_SYNC);
        // SAFETY: the mapping; the copies into it have run.
        let read = unsafe { std::slice::from_raw_parts(out_data, 320) };
        assert_eq!(read[256], 7, "what a texture takes");
        assert_eq!([read[0], read[64], read[128], read[192]], [1, 2, 7, 5], "what the draws read");
        assert_eq!(
            [title[D], title[D + 64], title[D + 128], title[D + 192], title[S]],
            [2, 7, 0, 5, 9],
            "title memory"
        );
        let holds = card(host, &shadow);
        assert_eq!([holds[D], holds[D + 64], holds[D + 128], holds[D + 192], holds[S]], [2, 7, 0, 5, 9], "the card");
        assert_eq!(look(host), Some(0));
        // With the debug layer.
        if let Some(said) = host.log_messages() {
            eprintln!("{said}");
        }
    }

    /// Declared writes beside the title's own, in the same 64 KiB pieces and
    /// dword by dword, flights landing while later ones are in the air: in
    /// the end title memory and the card hold what each wrote last. (A byte
    /// is written by flights or by the title, not both at a time.)
    #[test]
    #[ignore = "wants VICTORY_RELAY=2 VICTORY_RELAY_COPY=1 VICTORY_CARD_BUFFERS_MIN_MB=1 and a process of its own"]
    fn declared_writes_beside_the_titles_own_stay() {
        let Some(host) = host() else {
            return;
        };
        for view in [false, true] {
            beside(host, 0x5555 + usize::from(view), view);
        }
    }

    fn beside(host: &Host, resource: usize, view: bool) {
        let (title, start, shadow) = title(resource, view);
        let slow = Slow::new(host);
        // A small generator: the runs are the same each time.
        let mut seed = 0x1234_5678_9abc_def1u64;
        let mut next = move |n: usize| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as usize % n
        };
        // What memory should hold: pieces 0..8 are used, words of them by
        // turns the title's (even) and the shaders' (odd).
        const USED: usize = 8 << 16;
        let mut model = vec![0u8; USED];
        let mut lists = Vec::new();
        for round in 0..800usize {
            // The title writes some of its words, flights in the air or not.
            for _ in 0..next(24) {
                let at = next(USED / 8) * 8;
                let value = ((round as u32) << 8 | next(256) as u32).to_le_bytes();
                title[at..at + 4].copy_from_slice(&value);
                model[at..at + 4].copy_from_slice(&value);
            }
            // A submission: reads all or a part, writes a declared part.
            let list = host.new_list().unwrap();
            let (from, to) = match next(4) {
                0 => (0, LEN),
                1 => {
                    let a = next(8) << 16;
                    (a, a + (1 << 16))
                }
                _ => {
                    let a = next(USED - 4096) & !255;
                    (a, a + 256 * (1 + next(16)))
                }
            };
            shadow.refresh(&list, (start + from) as u64, (to - from) as u64);
            if next(5) > 0 {
                shadow.gpu_writes(&list, (start + from) as u64, (to - from) as u64);
                if next(3) == 0 {
                    slow.record(&list, 1 + next(3));
                }
                for _ in 0..1 + next(6) {
                    let words = (to.min(USED) - from) / 8;
                    let at = from + next(words) * 8 + 4;
                    // Now and then something in between first.
                    if next(4) == 0 {
                        write(host, &list, &shadow, at, &[0xEE; 4]);
                        slow.record(&list, 1);
                    }
                    let value = if next(3) == 0 { [7, 7, 7, 7] } else { (0x8000_0000 | (round as u32) << 8 | next(256) as u32).to_le_bytes() };
                    write(host, &list, &shadow, at, &value);
                    model[at..at + 4].copy_from_slice(&value);
                }
            }
            list.close();
            host.execute_own(&list);
            lists.push(list);
            if round % 40 == 39 {
                // Everything refreshed, landed and looked at.
                let all = host.new_list().unwrap();
                shadow.refresh(&all, start as u64, LEN as u64);
                all.close();
                host.execute_own(&all);
                settle_now(WHY_SYNC);
                let all = host.new_list().unwrap();
                shadow.refresh(&all, start as u64, LEN as u64);
                all.close();
                host.execute_own(&all);
                settle_now(WHY_SYNC);
                lists.push(all);
                assert_eq!(look(host), Some(0), "round {round}: the card's side is the CPU's side");
                let holds = card(host, &shadow);
                let wrong = |held: &[u8]| (0..USED / 4).filter(|w| held[w * 4..w * 4 + 4] != model[w * 4..w * 4 + 4]).collect::<Vec<_>>();
                let (on_card, in_title) = (wrong(&holds), wrong(title));
                assert!(
                    on_card.is_empty() && in_title.is_empty(),
                    "round {round}: words {on_card:x?} on the card and {in_title:x?} in title memory are not what was written last"
                );
                assert_eq!(host.log_messages(), None);
            }
        }
    }
}
