//! Every pipeline the title has ever created, kept on disk
//! (`VICTORY_PIPELINE_BOOK=1`), and their host pipelines made ahead of a run
//! (`victory warm`), so that the driver's own cache on disk has them before
//! a draw asks for one.
//!
//! The title creates 84,000 pipelines while it loads and draws with some
//! 2,000 of them; which ones is known from earlier runs (`library`), and
//! those are made during loading. One met for the first time (a place never
//! seen, or everything after a change of the translator or of the driver) is
//! compiled when a draw wants it. Making all of them in the running title
//! (`VICTORY_EARLY_ALL=1`) ended it inside the driver's compiler; made here,
//! a compiler that crashes or hangs ends a child process, costs that pipeline
//! alone and is named in `warm-bad.txt`.
//!
//! Two files beside the library, both only ever added to (a torn end is cut
//! off): `known-shaders.bin` (stage, hash, length, container) and
//! `known-pipelines.bin` (length, then the key, the shaders' stages and
//! hashes, and the graphics state). They hold the title's own code: they stay
//! in the cache folder.

use std::collections::{HashMap, HashSet};
use std::ffi::CString;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::PCSTR;

use super::Shader;
use super::host::{GraphicsState, HostPipeline};

mod writer;

const SHADERS: &str = "known-shaders.bin";
const PIPELINES: &str = "known-pipelines.bin";
/// Stages by their number in the files.
const STAGES: [&str; 8] =
    ["vertex", "pixel", "domain", "hull", "geometry", "compute", "amplification", "mesh"];

/// `VICTORY_PIPELINE_BOOK=1`: the pipelines the title creates are kept.
fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PIPELINE_BOOK").is_some_and(|v| v == "1"))
}

struct Book {
    dir: PathBuf,
    keys: HashSet<[u8; 16]>,
    shaders: HashSet<[u8; 32]>,
    /// Not yet written: shaders, then pipelines.
    new: [Vec<u8>; 2],
    added: usize,
}

struct Recorder {
    book: Mutex<Book>,
    writer: writer::Writer,
}

static BOOK: OnceLock<Option<Recorder>> = OnceLock::new();

fn book() -> Option<&'static Recorder> {
    BOOK.get_or_init(|| {
        let dir = super::library::folder().filter(|_| on())?;
        std::fs::create_dir_all(&dir).ok()?;
        let mut keys = HashSet::new();
        let mut shaders = HashSet::new();
        let ends = [
            walk_shaders(&dir, |_, hash, _, _| {
                shaders.insert(hash);
            }),
            walk_pipelines(&dir, |_, bytes| {
                keys.extend(bytes.first_chunk::<16>().copied());
            }),
        ];
        for (file, end) in [SHADERS, PIPELINES].iter().zip(ends) {
            let path = dir.join(file);
            if std::fs::metadata(&path).is_ok_and(|m| m.len() > end)
                && let Ok(f) = std::fs::OpenOptions::new().write(true).open(&path)
            {
                let _ = f.set_len(end);
            }
        }
        tracing::info!(pipelines = keys.len(), shaders = shaders.len(), "title pipelines known from earlier runs");
        Some(Recorder {
            writer: writer::Writer::start(flush).ok()?,
            book: Mutex::new(Book { dir, keys, shaders, new: Default::default(), added: 0 }),
        })
    })
    .as_ref()
}

/// The title created a pipeline.
pub(super) fn note(key: &str, shaders: &[Arc<Shader>], graphics: Option<&GraphicsState>) {
    let Some(recorder) = book() else { return };
    let mut id = [0u8; 16];
    for (at, b) in id.iter_mut().enumerate() {
        *b = key.get(2 * at..2 * at + 2).and_then(|x| u8::from_str_radix(x, 16).ok()).unwrap_or(0);
    }
    let mut book = recorder.book.lock().unwrap_or_else(|e| e.into_inner());
    if shaders.is_empty() || !book.keys.insert(id) {
        return;
    }
    let mut entry = id.to_vec();
    entry.push(shaders.len() as u8);
    for s in shaders {
        let hash = *blake3::hash(&s.container).as_bytes();
        let stage = STAGES.iter().position(|&x| x == s.stage).unwrap_or(0) as u8;
        entry.push(stage);
        entry.extend(hash);
        if book.shaders.insert(hash) {
            let new = &mut book.new[0];
            new.push(stage);
            new.extend(hash);
            new.extend((s.container.len() as u32).to_le_bytes());
            new.extend(&s.container);
        }
    }
    entry.push(u8::from(graphics.is_some()));
    if let Some(g) = graphics {
        g.put(&mut entry);
    }
    book.new[1].extend((entry.len() as u32).to_le_bytes());
    book.new[1].extend(entry);
    book.added += 1;
}

/// Appends pipelines noted since the last successful write, if the book is active.
pub fn flush() {
    let Some(Some(recorder)) = BOOK.get() else { return };
    flush_book(&recorder.book);
}

/// Gives the existing writer a short chance to save after the fatal GPU report.
pub fn flush_on_gpu_failure() {
    if let Some(Some(recorder)) = BOOK.get() {
        recorder.writer.request_flush(std::time::Duration::from_millis(250));
    }
}

fn flush_book(book: &Mutex<Book>) {
    // The timer and shutdown may both flush. Keep their writes in order while
    // allowing the title to note more pipelines during disk I/O.
    static WRITER: Mutex<()> = Mutex::new(());
    let _writer = WRITER.lock().unwrap_or_else(|e| e.into_inner());
    let (dir, mut new, added) = {
        let mut book = book.lock().unwrap_or_else(|e| e.into_inner());
        if book.added == 0 {
            return;
        }
        (book.dir.clone(), std::mem::take(&mut book.new), std::mem::take(&mut book.added))
    };
    for (file, bytes) in [SHADERS, PIPELINES].into_iter().zip(&mut new) {
        if bytes.is_empty() {
            continue;
        }
        let written = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(dir.join(file))
            .and_then(|f| append_bytes(f, bytes));
        if let Err(e) = written {
            let mut book = book.lock().unwrap_or_else(|e| e.into_inner());
            for (mut older, newer) in new.into_iter().zip(&mut book.new) {
                if !older.is_empty() {
                    older.append(newer);
                    *newer = older;
                }
            }
            book.added += added;
            drop(book);
            tracing::warn!(error = %e, file, "keeping unwritten title pipelines for retry");
            return;
        }
    }
    tracing::info!(pipelines = added, "title pipelines written down");
}

/// A failed append may already have written a prefix of a record.
fn append_bytes(mut out: impl Write, bytes: &mut Vec<u8>) -> std::io::Result<()> {
    use std::io::{Error, ErrorKind};
    let mut written = 0;
    let result = loop {
        if written == bytes.len() {
            break Ok(());
        }
        match out.write(&bytes[written..]) {
            Ok(0) => break Err(Error::from(ErrorKind::WriteZero)),
            Ok(n) => written += n,
            Err(e) if e.kind() == ErrorKind::Interrupted => continue,
            Err(e) => break Err(e),
        }
    };
    bytes.drain(..written);
    result
}

/// Calls `f` with each container's stage, hash, place and length; where the
/// last whole one ends.
fn walk_shaders(dir: &Path, mut f: impl FnMut(u8, [u8; 32], u64, u32)) -> u64 {
    let Ok(file) = std::fs::File::open(dir.join(SHADERS)) else { return 0 };
    let len = file.metadata().map_or(0, |m| m.len());
    let mut file = std::io::BufReader::new(file);
    let mut at = 0;
    let mut head = [0u8; 37];
    while file.read_exact(&mut head).is_ok() {
        let size = u32::from_le_bytes([head[33], head[34], head[35], head[36]]);
        if at + 37 + u64::from(size) > len || file.seek_relative(i64::from(size)).is_err() {
            break;
        }
        let mut hash = [0; 32];
        hash.copy_from_slice(&head[1..33]);
        f(head[0], hash, at + 37, size);
        at += 37 + u64::from(size);
    }
    at
}

/// Calls `f` with the offset of each length prefix and the entry that follows.
/// Returns where the last whole entry ends.
fn walk_pipelines(dir: &Path, mut f: impl FnMut(u64, &[u8])) -> u64 {
    let bytes = std::fs::read(dir.join(PIPELINES)).unwrap_or_default();
    let mut at = 0;
    while let Some(head) = bytes.get(at..at + 4) {
        let size = u32::from_le_bytes([head[0], head[1], head[2], head[3]]) as usize;
        let Some(entry) = bytes.get(at + 4..at + 4 + size) else { break };
        f(at as u64, entry);
        at += 4 + size;
    }
    at as u64
}

/// How many pipelines are known in `dir`.
pub fn count(dir: &Path) -> usize {
    let mut n = 0;
    walk_pipelines(dir, |_, _| n += 1);
    n
}

fn recipe_key(entry: &[u8]) -> String {
    entry.get(..16).unwrap_or_default().iter().map(|b| format!("{b:02x}")).collect()
}

/// The key of each pipeline known in `dir`, as the early list has it.
pub fn keys(dir: &Path) -> Vec<String> {
    let mut all = Vec::new();
    walk_pipelines(dir, |_, entry| all.push(recipe_key(entry)));
    all
}

/// A recorded pipeline and whether the configured shader skip list excludes it.
pub struct WarmEntry {
    pub key: String,
    pub skipped: bool,
    pub offset: u64,
}

/// Entries in their recorded order, so worker indices still name the same pipelines.
pub fn warm_entries(dir: &Path) -> Vec<WarmEntry> {
    let skips = super::skip_hashes();
    let mut entries = Vec::new();
    walk_pipelines(dir, |offset, entry| {
        entries.push(WarmEntry { key: recipe_key(entry), skipped: matches_skip_list(entry, skips), offset });
    });
    entries
}

fn matches_skip_list(entry: &[u8], skips: &[String]) -> bool {
    if skips.is_empty() {
        return false;
    }
    let Some((&count, rest)) = entry.get(16..).and_then(|bytes| bytes.split_first()) else {
        return false;
    };
    let Some(shaders) = rest.get(..usize::from(count) * 33) else {
        return false;
    };
    shaders.chunks_exact(33).any(|shader| {
        let hash = blake3::Hash::from_bytes(shader[1..].try_into().expect("recorded shader hash"));
        let hex = hash.to_hex();
        skips.iter().any(|skip| skip == &hex[..16])
    })
}

/// What a driver's cached pipelines depend on besides the title's: the
/// translator with its settings, and the driver.
pub fn fingerprint() -> String {
    let mut h = blake3::Hasher::new();
    h.update(&crate::shader_cache::translation());
    if let Some(host) = crate::host::host() {
        h.update(&host.driver().to_le_bytes());
    }
    h.finalize().to_hex()[..16].to_owned()
}

/// The translations of earlier runs are in `dir` (the pipeline library is
/// not used: what is made here is the driver's to keep).
pub fn use_translations(dir: PathBuf) {
    crate::shader_cache::use_shader_cache(dir);
}

/// A recipe selected by the warm command, in the book it scanned.
pub struct WarmRecipe<'a> {
    pub index: usize,
    pub offset: u64,
    pub key: &'a str,
}

/// Load only this worker's recipes, in request order. Check the keys before
/// any work is reported so a changed book is a setup error, not a bad shader.
fn requested_pipelines(dir: &Path, which: &[WarmRecipe<'_>]) -> io::Result<Vec<Vec<u8>>> {
    let mut book = std::fs::File::open(dir.join(PIPELINES))?;
    let len = book.metadata()?.len();
    which.iter().map(|recipe| {
        let Some(body) = recipe.offset.checked_add(4).filter(|&at| at <= len) else {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "pipeline recipe offset past end of book"));
        };
        book.seek(SeekFrom::Start(recipe.offset))?;
        let mut head = [0; 4];
        book.read_exact(&mut head)?;
        let size = u32::from_le_bytes(head);
        if u64::from(size) > len - body {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "incomplete pipeline recipe"));
        }
        let mut entry = vec![0; size as usize];
        book.read_exact(&mut entry)?;
        if recipe_key(&entry) != recipe.key {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "pipeline book changed before the worker read it"));
        }
        Ok(entry)
    }).collect()
}

/// Makes the selected host pipelines and lets each go again: the driver's
/// cache on disk keeps them. `each` receives each recipe's index before
/// compiling, and afterward with elapsed milliseconds and success.
pub fn make(
    dir: &Path, which: &[WarmRecipe<'_>], mut each: impl FnMut(usize, Option<(f64, bool)>),
) -> io::Result<()> {
    let entries = requested_pipelines(dir, which)?;
    let mut places = HashMap::new();
    walk_shaders(dir, |stage, hash, at, len| {
        places.insert(hash, (stage, at, len));
    });
    let mut file = std::fs::File::open(dir.join(SHADERS))?;
    let mut read = |hash: &[u8; 32]| {
        let &(_, at, len) = places.get(hash)?;
        let mut bytes = vec![0; len as usize];
        file.seek(SeekFrom::Start(at)).ok()?;
        file.read_exact(&mut bytes).ok()?;
        Some(bytes)
    };
    for (recipe, entry) in which.iter().zip(&entries) {
        each(recipe.index, None);
        let began = std::time::Instant::now();
        let made = parse(entry, &mut read)
            .and_then(|(shaders, g)| HostPipeline::new(&shaders, g.as_deref()))
            .is_some();
        each(recipe.index, Some((began.elapsed().as_secs_f64() * 1e3, made)));
    }
    Ok(())
}

#[allow(clippy::type_complexity)]
fn parse(
    entry: &[u8],
    read: &mut impl FnMut(&[u8; 32]) -> Option<Vec<u8>>,
) -> Option<(Vec<Arc<Shader>>, Option<Box<GraphicsState>>)> {
    let mut rest = entry.get(16..)?;
    let (&count, after) = rest.split_first()?;
    rest = after;
    let mut shaders = Vec::new();
    for _ in 0..count {
        let (head, after) = rest.split_first_chunk::<33>()?;
        rest = after;
        let mut hash = [0; 32];
        hash.copy_from_slice(&head[1..]);
        shaders.push(super::shader(STAGES.get(usize::from(head[0]))?, &read(&hash)?));
    }
    let (&graphics, rest) = rest.split_first()?;
    let g = match graphics {
        0 => None,
        _ => Some(Box::new(GraphicsState::get(rest)?)),
    };
    Some((shaders, g))
}

/// Calls `$f` with each number of a graphics state but its input elements',
/// as a place of 32 bits, in one order.
macro_rules! numbers {
    ($g:expr, $f:ident) => {{
        $f!($g.blend.AlphaToCoverageEnable.0);
        $f!($g.blend.IndependentBlendEnable.0);
        for t in &mut $g.blend.RenderTarget {
            $f!(t.BlendEnable.0);
            $f!(t.LogicOpEnable.0);
            $f!(t.SrcBlend.0);
            $f!(t.DestBlend.0);
            $f!(t.BlendOp.0);
            $f!(t.SrcBlendAlpha.0);
            $f!(t.DestBlendAlpha.0);
            $f!(t.BlendOpAlpha.0);
            $f!(t.LogicOp.0);
            $f!(t.RenderTargetWriteMask);
        }
        $f!($g.sample_mask);
        $f!($g.rasterizer.FillMode.0);
        $f!($g.rasterizer.CullMode.0);
        $f!($g.rasterizer.FrontCounterClockwise.0);
        $f!($g.rasterizer.DepthBias);
        $f!($g.rasterizer.DepthClipEnable.0);
        $f!($g.rasterizer.MultisampleEnable.0);
        $f!($g.rasterizer.AntialiasedLineEnable.0);
        $f!($g.rasterizer.ForcedSampleCount);
        $f!($g.rasterizer.ConservativeRaster.0);
        $f!($g.depth.DepthEnable.0);
        $f!($g.depth.DepthWriteMask.0);
        $f!($g.depth.DepthFunc.0);
        $f!($g.depth.StencilEnable.0);
        $f!($g.depth.StencilReadMask);
        $f!($g.depth.StencilWriteMask);
        for face in [&mut $g.depth.FrontFace, &mut $g.depth.BackFace] {
            $f!(face.StencilFailOp.0);
            $f!(face.StencilDepthFailOp.0);
            $f!(face.StencilPassOp.0);
            $f!(face.StencilFunc.0);
        }
        $f!($g.strip_cut.0);
        $f!($g.topology.0);
        $f!($g.targets.NumRenderTargets);
        for format in &mut $g.targets.RTFormats {
            $f!(format.0);
        }
        $f!($g.depth_format.0);
        $f!($g.samples.Count);
        $f!($g.samples.Quality);
    }};
}

impl GraphicsState {
    /// The state as bytes.
    fn put(&self, out: &mut Vec<u8>) {
        out.push(u8::from(self.depth_bounds));
        for f in [self.rasterizer.DepthBiasClamp, self.rasterizer.SlopeScaledDepthBias] {
            out.extend(f.to_le_bytes());
        }
        let mut plain = GraphicsState {
            blend: self.blend,
            sample_mask: self.sample_mask,
            rasterizer: self.rasterizer,
            depth: self.depth,
            strip_cut: self.strip_cut,
            topology: self.topology,
            targets: self.targets,
            depth_format: self.depth_format,
            samples: self.samples,
            ..Default::default()
        };
        macro_rules! write {
            ($x:expr) => {
                out.extend(($x as u32).to_le_bytes())
            };
        }
        numbers!(plain, write);
        out.push(self.inputs.len() as u8);
        for (e, name) in self.inputs.iter().zip(&self.names) {
            let name = name.as_bytes();
            out.push(name.len() as u8);
            out.extend(name);
            for v in [
                e.SemanticIndex,
                e.Format.0 as u32,
                e.InputSlot,
                e.AlignedByteOffset,
                e.InputSlotClass.0 as u32,
                e.InstanceDataStepRate,
            ] {
                out.extend(v.to_le_bytes());
            }
        }
    }

    fn get(bytes: &[u8]) -> Option<GraphicsState> {
        let mut g = GraphicsState::default();
        let (&bounds, mut rest) = bytes.split_first()?;
        g.depth_bounds = bounds != 0;
        let mut next = || {
            let (head, after) = rest.split_first_chunk::<4>()?;
            rest = after;
            Some(u32::from_le_bytes(*head))
        };
        g.rasterizer.DepthBiasClamp = f32::from_bits(next()?);
        g.rasterizer.SlopeScaledDepthBias = f32::from_bits(next()?);
        macro_rules! read {
            ($x:expr) => {
                $x = next()? as _
            };
        }
        numbers!(g, read);
        let (&count, mut rest) = rest.split_first()?;
        let mut elements = Vec::new();
        for _ in 0..count {
            let (&len, after) = rest.split_first()?;
            let (name, after) = after.split_at_checked(usize::from(len))?;
            let (numbers, after) = after.split_first_chunk::<24>()?;
            rest = after;
            let mut v = [0u32; 6];
            for (v, b) in v.iter_mut().zip(numbers.chunks_exact(4)) {
                *v = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            }
            elements.push((CString::new(name).ok()?, v));
        }
        g.names = elements.iter().map(|e| e.0.clone()).collect();
        g.inputs = g
            .names
            .iter()
            .zip(&elements)
            .map(|(name, (_, v))| D3D12_INPUT_ELEMENT_DESC {
                SemanticName: PCSTR(name.as_ptr().cast()),
                SemanticIndex: v[0],
                Format: DXGI_FORMAT(v[1] as i32),
                InputSlot: v[2],
                AlignedByteOffset: v[3],
                InputSlotClass: D3D12_INPUT_CLASSIFICATION(v[4] as i32),
                InstanceDataStepRate: v[5],
            })
            .collect();
        Some(g)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failed_open_retries_before_later_records() {
        for blocked in [SHADERS, PIPELINES] {
            let tmp = tempfile::tempdir().unwrap();
            let dir = tmp.path();
            let book = Mutex::new(Book {
                dir: dir.into(),
                keys: HashSet::new(),
                shaders: HashSet::new(),
                new: [vec![1], vec![11]],
                added: 1,
            });
            std::fs::create_dir(dir.join(blocked)).unwrap();
            flush_book(&book);
            std::fs::remove_dir(dir.join(blocked)).unwrap();
            {
                let mut b = book.lock().unwrap();
                assert_eq!(b.added, 1);
                b.new[0].push(2);
                b.new[1].push(12);
                b.added += 1;
            }
            flush_book(&book);
            flush_book(&book);
            assert_eq!(std::fs::read(dir.join(SHADERS)).unwrap(), [1, 2]);
            assert_eq!(std::fs::read(dir.join(PIPELINES)).unwrap(), [11, 12]);
            assert_eq!(book.lock().unwrap().added, 0);
        }
    }

    #[test]
    fn a_short_write_keeps_only_the_unwritten_suffix() {
        let mut pending = vec![1, 2, 3, 4, 5];
        let mut first = [0; 2];
        // A fixed-size output accepts the prefix, then returns Ok(0).
        let err = append_bytes(&mut first[..], &mut pending).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::WriteZero);
        assert_eq!(first, [1, 2]);
        assert_eq!(pending, [3, 4, 5]);
        let mut rest = Vec::new();
        append_bytes(&mut rest, &mut pending).unwrap();
        assert!(pending.is_empty());
        assert_eq!(rest, [3, 4, 5]);
    }

    #[test]
    fn a_state_comes_back_as_it_was() {
        let mut g = GraphicsState { sample_mask: 7, depth_bounds: true, ..Default::default() };
        g.rasterizer.DepthBias = -3;
        g.rasterizer.SlopeScaledDepthBias = 1.25;
        g.blend.RenderTarget[3].RenderTargetWriteMask = 5;
        g.depth.StencilReadMask = 9;
        g.targets.NumRenderTargets = 2;
        g.targets.RTFormats[1] = DXGI_FORMAT_R8_UNORM;
        g.depth_format = DXGI_FORMAT_D32_FLOAT;
        g.names = vec![CString::new("POSITION").unwrap(), CString::new("T").unwrap()];
        g.inputs = g
            .names
            .iter()
            .enumerate()
            .map(|(n, name)| D3D12_INPUT_ELEMENT_DESC {
                SemanticName: PCSTR(name.as_ptr().cast()),
                SemanticIndex: n as u32,
                Format: DXGI_FORMAT_R32G32_FLOAT,
                InputSlot: 2,
                AlignedByteOffset: D3D12_APPEND_ALIGNED_ELEMENT,
                InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA,
                InstanceDataStepRate: 4,
            })
            .collect();
        let mut bytes = Vec::new();
        g.put(&mut bytes);
        let back = GraphicsState::get(&bytes).expect("whole");
        assert_eq!(back.describe(), g.describe());
        assert!(GraphicsState::get(&bytes[..bytes.len() - 1]).is_none());
        assert!(GraphicsState::get(&bytes[..9]).is_none());
    }

    #[test]
    fn a_warm_skip_can_match_any_shader_in_a_recipe() {
        let mut entry = vec![0xff; 16];
        entry.extend([2, 0]);
        entry.extend([0x11; 32]);
        entry.push(1);
        entry.extend([0x22; 32]);
        entry.push(0);

        let skips = vec!["22".repeat(8)];
        assert!(matches_skip_list(&entry, &skips));
        assert!(!matches_skip_list(&entry, &["33".repeat(8)]));
        assert!(!matches_skip_list(&entry, &[]));
        assert!(!matches_skip_list(&entry[..entry.len() - 2], &skips));
    }

    #[test]
    fn a_warm_worker_reads_selected_recipes_in_request_order() {
        let dir = tempfile::tempdir().unwrap();
        let mut bytes = Vec::new();
        for (byte, size) in [(1, 18u32), (2, 240), (3, 51)] {
            bytes.extend(size.to_le_bytes());
            bytes.extend(vec![byte; size as usize]);
        }
        bytes.extend(100u32.to_le_bytes()); // interrupted append
        std::fs::write(dir.path().join(PIPELINES), &bytes).unwrap();
        let entries = warm_entries(dir.path());
        assert_eq!(entries.len(), 3);
        let request = |index: usize| WarmRecipe {
            index, offset: entries[index].offset, key: &entries[index].key,
        };
        let which = [request(2), request(0), request(2)];
        assert_eq!(requested_pipelines(dir.path(), &which).unwrap(), [vec![3; 51], vec![1; 18], vec![3; 51]]);

        let mut stale = [request(0)];
        stale[0].offset = entries[1].offset;
        assert_eq!(requested_pipelines(dir.path(), &stale).unwrap_err().kind(), io::ErrorKind::InvalidData);
        stale[0].offset = (bytes.len() - 4) as u64;
        assert_eq!(requested_pipelines(dir.path(), &stale).unwrap_err().kind(), io::ErrorKind::UnexpectedEof);
    }

    #[test]
    fn a_torn_end_is_not_read() {
        let dir = std::env::temp_dir().join(format!("victory-known-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut bytes = Vec::new();
        for n in 0..3u8 {
            bytes.extend(20u32.to_le_bytes());
            bytes.extend([n; 20]);
        }
        std::fs::write(dir.join(PIPELINES), &bytes[..bytes.len() - 5]).unwrap();
        assert_eq!(count(&dir), 2);
        assert_eq!(keys(&dir)[1], "01".repeat(16));
        assert_eq!(walk_pipelines(&dir, |_, _| ()), 48);
        let mut shader = vec![5u8];
        shader.extend([9; 32]);
        shader.extend(4u32.to_le_bytes());
        shader.extend([1, 2, 3, 4]);
        let one = shader.len() as u64;
        shader.extend_from_slice(&shader.clone()[..20]);
        std::fs::write(dir.join(SHADERS), &shader).unwrap();
        let mut seen = Vec::new();
        assert_eq!(walk_shaders(&dir, |stage, hash, at, len| seen.push((stage, hash[0], at, len))), one);
        assert_eq!(seen, [(5, 9, 37, 4)]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
