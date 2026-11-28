//! Presented frames written to PNG files, to see what reaches the screen:
//! with `VICTORY_CAPTURE=<folder>`, every frame whose number is a power of
//! two (the frames the log reports) is saved as `frame-<n>.png`, at half
//! size. Only in builds with diagnostics.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

/// The folder frames go to, if capturing.
pub(crate) fn folder() -> Option<&'static PathBuf> {
    static FOLDER: std::sync::OnceLock<Option<PathBuf>> = std::sync::OnceLock::new();
    FOLDER
        .get_or_init(|| std::env::var_os("VICTORY_CAPTURE").map(PathBuf::from))
        .as_ref()
}

/// Frames listed in `VICTORY_CAPTURE_FRAMES` (comma-separated): saved
/// too, and their draws traced at debug level.
pub(crate) fn listed(n: u64) -> bool {
    if TRACE_FRAME.load(Ordering::Acquire) == n {
        return true;
    }
    static LIST: std::sync::OnceLock<Vec<u64>> = std::sync::OnceLock::new();
    let named = LIST
        .get_or_init(|| {
            let list = std::env::var("VICTORY_CAPTURE_FRAMES").unwrap_or_default();
            list.split(',')
                .filter_map(|f| f.trim().parse().ok())
                .collect()
        })
        .contains(&n);
    // `VICTORY_CAPTURE_WORLD`: also the 2nd and the 6th frame after the first
    // one that ran 500 draws or more (a title's first 3D frames: their
    // numbers depend on how long the menus were shown), and the 30th and
    // the 80th (the title opens with a cutscene made in the
    // engine: later shots of it, in case the first is dark by design).
    //
    // `VICTORY_CAPTURE_WORLD_ALSO` (comma-separated): more frames counted
    // from that first one (the title's blue patches show from
    // about the 118th frame on, when the window's model has arrived and
    // the reflections hold values: none of the four above has them).
    static ALSO: std::sync::OnceLock<Vec<u64>> = std::sync::OnceLock::new();
    let also = ALSO.get_or_init(|| {
        let list = std::env::var("VICTORY_CAPTURE_WORLD_ALSO").unwrap_or_default();
        list.split(',')
            .filter_map(|f| f.trim().parse().ok())
            .collect()
    });
    let first = WORLD.load(std::sync::atomic::Ordering::Relaxed);
    let after = n.wrapping_sub(first);
    named || first != u64::MAX && ([2, 6, 30, 80].contains(&after) || also.contains(&after))
}

static TRACE_FRAME: AtomicU64 = AtomicU64::new(u64::MAX);

fn scene_capture_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::switch::on("VICTORY_CAPTURE_SCENE_BEFORE_FRAMEGEN"))
}

/// The title has reached a present. A trace request names the frame
/// after it. The caller drains translation at both edges while operations
/// are still numbered with the frame they belong to.
pub(crate) fn trace_boundary(present: u64) -> bool {
    if TRACE_FRAME.load(Ordering::Acquire) == present {
        return true;
    }
    if !scene_capture_on() {
        return false;
    }
    let (Some(folder), Some(frame)) = (folder(), present.checked_add(1)) else {
        return false;
    };
    static TRIGGER: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    if std::fs::remove_file(TRIGGER.get_or_init(|| folder.join("scene-trace"))).is_err() {
        return false;
    }
    TRACE_FRAME.store(frame, Ordering::Release);
    tracing::info!(frame, "scene trace requested");
    true
}

/// With a capture folder, for the frames [`listed`]: `what` is added to
/// the frame's `frame-<n>-ops.txt`, one line per draw, dispatch, copy or
/// drop in the order recorded (the title records on several threads, so
/// lines of different lists interleave; each names its list). The file
/// says which pass writes which target, to find the pass whose output
/// stays empty.
#[cfg(feature = "diagnostics")]
pub(crate) fn op(list: usize, what: impl FnOnce() -> String) {
    use std::io::Write;
    let frame = crate::queue::presented() + 1;
    if !listed(frame) {
        // Before the world: the last lines are kept, and written out when
        // it starts (`before-world-ops.txt`): what the title drew just
        // before (the title shows a title card there, whose text
        // did not show).
        if world_on() && WORLD.load(std::sync::atomic::Ordering::Relaxed) == u64::MAX {
            // At most 300 lines of a frame: the world's first frame alone
            // (thousands of operations, recorded before it is known to be
            // the world's) pushed everything before it out.
            static IN_FRAME: std::sync::Mutex<(u64, u32)> = std::sync::Mutex::new((0, 0));
            {
                let mut in_frame = IN_FRAME.lock().unwrap_or_else(|e| e.into_inner());
                if in_frame.0 != frame {
                    *in_frame = (frame, 0);
                }
                in_frame.1 += 1;
                if in_frame.1 > 300 {
                    return;
                }
            }
            let what = what();
            let mut ring = RING.lock().unwrap_or_else(|e| e.into_inner());
            if ring.len() >= RING_LINES {
                ring.pop_front();
            }
            ring.push_back(format!("frame {frame} list {list:#x} {what}"));
        }
        return;
    }
    let Some(folder) = folder() else {
        return;
    };
    static FILE: std::sync::Mutex<Option<(u64, u64, std::fs::File)>> = std::sync::Mutex::new(None);
    let what = what();
    let mut file = FILE.lock().unwrap_or_else(|e| e.into_inner());
    if file.as_ref().is_none_or(|f| f.0 != frame) {
        let Ok(made) = std::fs::File::create(folder.join(format!("frame-{frame}-ops.txt"))) else {
            return;
        };
        *file = Some((frame, 0, made));
    }
    if let Some((_, n, f)) = file.as_mut() {
        *n += 1;
        let _ = writeln!(f, "{n} list {list:#x} {what}");
    }
}

/// The last lines of [`op`] before the world.
#[cfg(feature = "diagnostics")]
static RING: std::sync::Mutex<std::collections::VecDeque<String>> =
    std::sync::Mutex::new(std::collections::VecDeque::new());
#[cfg(feature = "diagnostics")]
const RING_LINES: usize = 6000;

/// Whether `VICTORY_CAPTURE_WORLD` is set.
fn world_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CAPTURE_WORLD").is_some())
}

/// The first frame that ran 500 draws or more (`VICTORY_CAPTURE_WORLD`).
static WORLD: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(u64::MAX);

/// At the present of `frame`: the draws run on the host so far.
pub(crate) fn frame_drew(frame: u64, drawn: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    static LAST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let before = LAST.swap(drawn, Relaxed);
    if world_on()
        && drawn.saturating_sub(before) >= 500
        && WORLD.compare_exchange(u64::MAX, frame, Relaxed, Relaxed).is_ok()
    {
        tracing::info!(frame, "the first frame of 500 draws or more: frames {} and {} are captured", frame + 2, frame + 6);
        #[cfg(feature = "diagnostics")]
        if let Some(folder) = folder() {
            let lines: Vec<String> = std::mem::take(&mut *RING.lock().unwrap_or_else(|e| e.into_inner())).into();
            let _ = std::fs::write(folder.join("before-world-ops.txt"), lines.join("
"));
        }
    }
}

/// Whether to save this present, and whether to keep its UI separate.
pub(crate) fn requested(n: u64, listed: bool, framegen: bool) -> (bool, bool) {
    let Some(folder) = folder() else { return (false, false) };
    if scene_capture_on() {
        static TRIGGER: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
        let scene = framegen && (
            std::fs::remove_file(TRIGGER.get_or_init(|| folder.join("scene-next"))).is_ok() || listed
        );
        return (listed || scene, scene);
    }
    // Before the world, with `VICTORY_CAPTURE_WORLD`: every 16th frame too
    // (small pictures of menus, loading screens and title cards).
    let before_world = world_on() && n.is_multiple_of(16) && WORLD.load(std::sync::atomic::Ordering::Relaxed) == u64::MAX;
    // In the world: every 8th frame's picture, a timeline of the opening
    // cutscene (its shots do not fall on the same frame numbers from run to
    // run, so single frames cannot be compared between runs).
    let first = WORLD.load(std::sync::atomic::Ordering::Relaxed);
    let in_world = world_on() && first != u64::MAX && n > first && (n - first).is_multiple_of(8);
    (n.is_power_of_two() || listed || before_world || in_world, false)
}

/// Saves 8-bit RGBA or BGRA rows as `<name>.png`, sampling both axes at `step`.
fn save_as(name: &str, width: u32, height: u32, pitch: usize, bgra: bool, rows: &[u8], step: usize) {
    let Some(folder) = folder() else { return };
    let (w, h, raw) = pack_rgba_rows(width, height, pitch, bgra, rows, step);
    write_aside(folder.join(format!("{name}.png")), w, h, raw);
}

/// Copies only the sampled pixels into an owned PNG scanline buffer.
fn pack_rgba_rows(
    width: u32,
    height: u32,
    pitch: usize,
    bgra: bool,
    rows: &[u8],
    step: usize,
) -> (u32, u32, Vec<u8>) {
    let (w, h) = ((width as usize / step).max(1), (height as usize / step).max(1));
    let mut raw = Vec::with_capacity((w * 3 + 1) * h);
    for y in 0..h {
        raw.push(0);
        let row = &rows[step * y * pitch..];
        for x in 0..w {
            let p = &row[step * x * 4..][..3];
            if bgra {
                raw.extend([p[2], p[1], p[0]]);
            } else {
                raw.extend_from_slice(p);
            }
        }
    }
    (w as u32, h as u32, raw)
}

/// A picture to pack and write: its file, width, height and filtered rows.
type Picture = (PathBuf, u32, u32, Vec<u8>);

/// A thread that takes pictures from a bounded line, and what is needed to
/// end it with nothing left behind ([`Aside::finish`]).
struct Aside {
    line: std::sync::mpsc::SyncSender<Picture>,
    /// Let go by the thread when the line is closed and empty and the last
    /// picture is done.
    done: std::sync::mpsc::Receiver<()>,
    /// Pictures in the line or being worked on.
    waiting: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

impl Aside {
    /// A thread that hands each picture of its line to `work`, or none if
    /// the thread does not start.
    fn start(mut work: impl FnMut(Picture) + Send + 'static) -> Option<Self> {
        use std::sync::atomic::Ordering::AcqRel;
        let (line, pictures) = std::sync::mpsc::sync_channel::<Picture>(16);
        let (ended, done) = std::sync::mpsc::channel::<()>();
        let waiting = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let left = waiting.clone();
        std::thread::Builder::new()
            .name("victory-capture".to_owned())
            .spawn(move || {
                for picture in pictures {
                    work(picture);
                    left.fetch_sub(1, AcqRel);
                }
                drop(ended);
            })
            .ok()
            .map(|_| Aside { line, done, waiting })
    }

    /// Puts a picture in the line, or gives it back if the line is full (the
    /// disk is slower than the frames) or the thread is gone.
    fn give(&self, picture: Picture) -> Result<(), Picture> {
        use std::sync::atomic::Ordering::AcqRel;
        use std::sync::mpsc::TrySendError;
        self.waiting.fetch_add(1, AcqRel);
        self.line.try_send(picture).map_err(|e| {
            self.waiting.fetch_sub(1, AcqRel);
            match e {
                TrySendError::Full(picture) | TrySendError::Disconnected(picture) => picture,
            }
        })
    }

    /// Closes the line and waits, `bound` at most, until the thread has
    /// done every picture that was in it: how many it has not. (Not for
    /// the thread's own end: a thread ends under the loader's lock, which
    /// the thread that ends the process may hold.)
    fn finish(self, bound: std::time::Duration) -> usize {
        let Aside { line, done, waiting } = self;
        drop(line);
        // Nothing is ever sent: the wait ends when the thread lets go of its
        // end, or at the bound.
        let _ = done.recv_timeout(bound);
        waiting.load(std::sync::atomic::Ordering::Acquire)
    }
}

/// Where pictures go to be packed and written.
enum Line {
    /// No picture yet.
    Unused,
    Aside(Aside),
    /// On the caller: the thread did not start, or the process is ending
    /// ([`finish`]).
    Caller,
}

static LINE: std::sync::Mutex<Line> = std::sync::Mutex::new(Line::Unused);

/// Packs and writes a picture on a thread of its own: whether it went
/// there. The caller is the thread that prepares the submissions (a present
/// stands in its line), the one a frame waits for: with
/// `VICTORY_CAPTURE_WORLD` (every 8th frame) packing took 23% of its samples
/// (a profile of the Series S prison scene at 22 fps), in the very runs the
/// pictures are there to judge. What is waiting when the process ends is
/// written by [`finish`].
fn write_aside(path: PathBuf, w: u32, h: u32, raw: Vec<u8>) -> bool {
    let picture = (path, w, h, raw);
    let given = {
        let mut line = LINE.lock().unwrap_or_else(|e| e.into_inner());
        if matches!(*line, Line::Unused) {
            *line = Aside::start(|(path, w, h, raw)| write(&path, w, h, &raw)).map_or(Line::Caller, Line::Aside);
        }
        match &*line {
            Line::Aside(aside) => aside.give(picture),
            _ => Err(picture),
        }
    };
    // No thread, or the line is full: here, as before.
    match given {
        Ok(()) => true,
        Err((path, w, h, raw)) => {
            write(&path, w, h, &raw);
            false
        }
    }
}

/// How long the end of the process waits for the pictures in the line: a
/// disk that does not answer must not keep the process.
const FINISH_BOUND: std::time::Duration = std::time::Duration::from_secs(3);

/// The process is ending: the pictures in the line are written before this
/// returns, or [`FINISH_BOUND`] has passed (logged). A picture saved after
/// this is written by its caller.
pub(crate) fn finish() {
    // Window close and title exit may overlap. Both wait for one bounded drain.
    static DRAIN: std::sync::Once = std::sync::Once::new();
    DRAIN.call_once(|| {
        let left = finish_within(FINISH_BOUND);
        if left != 0 {
            tracing::warn!(
                left,
                seconds = FINISH_BOUND.as_secs(),
                "capture pictures were still to be written when the process ended"
            );
        }
    });
}

/// [`finish`] with its bound: how many pictures were not written in time.
fn finish_within(bound: std::time::Duration) -> usize {
    let line = std::mem::replace(&mut *LINE.lock().unwrap_or_else(|e| e.into_inner()), Line::Caller);
    match line {
        Line::Aside(aside) => aside.finish(bound),
        Line::Unused | Line::Caller => 0,
    }
}

fn write(path: &std::path::Path, w: u32, h: u32, raw: &[u8]) {
    let png = encode(w, h, raw);
    match std::fs::write(path, png) {
        Ok(()) => tracing::info!(path = %path.display(), "frame saved"),
        Err(e) => tracing::warn!(error = %e, path = %path.display(), "frame not saved"),
    }
}

/// A PNG of `width * height` pixels of 8-bit RGB.
pub(crate) fn png(width: u32, height: u32, rgb: &[u8]) -> Vec<u8> {
    let row = width as usize * 3;
    let mut raw = Vec::with_capacity((row + 1) * height as usize);
    for line in rgb.chunks_exact(row.max(1)) {
        raw.push(0);
        raw.extend_from_slice(line);
    }
    encode(width, height, &raw)
}

/// A PNG of rows of RGB, each behind its filter byte.
fn encode(w: u32, h: u32, raw: &[u8]) -> Vec<u8> {
    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut header = Vec::new();
    header.extend(w.to_be_bytes());
    header.extend(h.to_be_bytes());
    header.extend([8, 2, 0, 0, 0]); // 8-bit RGB
    chunk(&mut png, b"IHDR", &header);
    let mut compressed = vec![0u8; zlib_rs::compress_bound(raw.len())];
    let (idat, rc) = zlib_rs::compress_slice(&mut compressed, raw, zlib_rs::DeflateConfig::new(3));
    debug_assert_eq!(rc, zlib_rs::ReturnCode::Ok);
    chunk(&mut png, b"IDAT", idat);
    chunk(&mut png, b"IEND", &[]);
    png
}

fn chunk(png: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    png.extend((data.len() as u32).to_be_bytes());
    let start = png.len();
    png.extend(kind);
    png.extend(data);
    let crc = crc32(&png[start..]);
    png.extend(crc.to_be_bytes());
}

/// The check values of the 256 bytes: a byte a step instead of a bit (the
/// bit-by-bit loop was 5% of the thread that saved a picture).
const CRC_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u32;
        let mut bit = 0;
        while bit < 8 {
            crc = (crc >> 1) ^ (0xedb8_8320 & (crc & 1).wrapping_neg());
            bit += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
};

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &b in bytes {
        crc = CRC_TABLE[((crc ^ u32::from(b)) & 0xff) as usize] ^ (crc >> 8);
    }
    !crc
}

/// Records a copy of `texture` (in the copy-source state) into a
/// readback buffer.
pub(crate) fn read_back(
    device: &ID3D12Device,
    list: &ID3D12GraphicsCommandList,
    texture: &ID3D12Resource,
) -> Option<(ID3D12Resource, D3D12_PLACED_SUBRESOURCE_FOOTPRINT)> {
    read_back_of(device, list, texture, 0)
}

/// As [`read_back`], of subresource `sub` (the stencil plane of a
/// one-level, one-slice depth-stencil texture is subresource 1).
fn read_back_of(
    device: &ID3D12Device,
    list: &ID3D12GraphicsCommandList,
    texture: &ID3D12Resource,
    sub: u32,
) -> Option<(ID3D12Resource, D3D12_PLACED_SUBRESOURCE_FOOTPRINT)> {
    let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
    let mut total = 0;
    // SAFETY: a live texture; outputs for one subresource.
    unsafe {
        let desc = texture.GetDesc();
        device.GetCopyableFootprints(
            &desc,
            sub,
            1,
            0,
            Some(&mut footprint),
            None,
            None,
            Some(&mut total),
        );
    }
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_READBACK,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: total,
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
    // SAFETY: plain creation; an open list; locations naming live resources.
    unsafe {
        device
            .CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut buffer,
            )
            .ok()?;
        let buffer = buffer?;
        let to = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::transmute_copy(&buffer),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: footprint,
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::transmute_copy(texture),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: sub,
            },
        };
        list.CopyTextureRegion(&to, 0, 0, 0, &from, None);
        Some((buffer, footprint))
    }
}

/// Saves a frame read back by [`read_back`] (8-bit color only).
pub(crate) fn save_read_back(
    frame: u64,
    scene: bool,
    buffer: &ID3D12Resource,
    footprint: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
) {
    let f = footprint.Footprint;
    let bgra = match f.Format {
        DXGI_FORMAT_R8G8B8A8_UNORM | DXGI_FORMAT_R8G8B8A8_UNORM_SRGB => false,
        DXGI_FORMAT_B8G8R8A8_UNORM | DXGI_FORMAT_B8G8R8A8_UNORM_SRGB => true,
        other => {
            tracing::warn!(format = other.0, "frame not saved: not 8-bit color");
            return;
        }
    };
    let mut data = std::ptr::null_mut();
    // SAFETY: the copy finished (the capture forces a synchronous submission).
    // The footprint is checked against the buffer before slicing its mapping.
    unsafe {
        let Some(range) = rgba_readback_range(
            footprint.Offset,
            f.Width,
            f.Height,
            f.RowPitch,
            buffer.GetDesc().Width,
        ) else {
            tracing::warn!(frame, "frame not saved: invalid readback footprint");
            return;
        };
        if buffer.Map(0, None, Some(&mut data)).is_err() {
            return;
        }
        let pitch = f.RowPitch as usize;
        let rows = std::slice::from_raw_parts(data.cast::<u8>().add(range.start), range.len());
        let (name, step) = if scene {
            (format!("scene-{frame}"), 1)
        } else {
            (format!("frame-{frame}"), 2)
        };
        save_as(&name, f.Width, f.Height, pitch, bgra, rows, step);
        buffer.Unmap(0, None);
    }
}

fn rgba_readback_range(
    offset: u64,
    width: u32,
    height: u32,
    pitch: u32,
    buffer_size: u64,
) -> Option<std::ops::Range<usize>> {
    let row_bytes = u64::from(width) * 4;
    if row_bytes == 0 || row_bytes > u64::from(pitch) {
        return None;
    }
    // GetCopyableFootprints does not require padding after the final row.
    let rows_before_last = u64::from(height.checked_sub(1)?);
    let end = offset
        .checked_add(rows_before_last * u64::from(pitch))?
        .checked_add(row_bytes)?;
    if end > buffer_size || end > isize::MAX as u64 {
        return None;
    }
    Some(offset as usize..end as usize)
}

/// With `VICTORY_DUMP_TARGETS` set: title textures the frames listed in
/// `VICTORY_CAPTURE_FRAMES` render to or write, in first-use order, saved
/// at that frame's present as `frame-<n>-<i>-<resource>-<w>x<h>-f<format>.png`
/// (what each holds at the end of the frame; formats normalized to show
/// structure: floats scaled by their largest value, depth by its range).
/// The host copies are held: the title may release a target before the
/// frame is presented.
static TARGETS: std::sync::Mutex<Vec<(usize, std::sync::Arc<crate::resource::HostTexture>)>> =
    std::sync::Mutex::new(Vec::new());

fn dump_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| folder().is_some() && crate::switch::on("VICTORY_DUMP_TARGETS"))
}

/// Notes a title texture the frame being built renders to or writes.
pub(crate) fn note_target(resource: crate::object::This) {
    if resource.is_null() || !dump_on() || !listed(crate::queue::presented() + 1) {
        return;
    }
    let mut targets = TARGETS.lock().unwrap_or_else(|e| e.into_inner());
    if !targets.iter().any(|&(r, _)| r == resource as usize)
        && let Some(copy) = crate::resource::host_copy(resource)
    {
        targets.push((resource as usize, copy));
    }
}

/// At the present of frame `frame`: saves the noted targets.
pub(crate) fn dump_targets(frame: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    let targets = std::mem::take(&mut *TARGETS.lock().unwrap_or_else(|e| e.into_inner()));
    let Some(host) = crate::host::host() else { return };
    for (i, (resource, copy)) in targets.iter().enumerate() {
        let texture = &copy.texture;
        // SAFETY: a live resource.
        let d = unsafe { texture.GetDesc() };
        if d.SampleDesc.Count > 1 || d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D {
            continue;
        }
        // Every submission has run; the dump's list moves it there and back.
        let state = copy.state.load(Relaxed);
        dump_one(host, texture, state, frame, i, *resource as crate::object::This, &d);
    }
}

fn dump_one(
    host: &crate::host::Host,
    texture: &ID3D12Resource,
    state: u32,
    frame: u64,
    i: usize,
    resource: crate::object::This,
    d: &D3D12_RESOURCE_DESC,
) {
    let Ok(list) = host.new_list() else { return };
    let state = D3D12_RESOURCE_STATES(crate::host::legal_state(texture, state) as i32);
    let copy = D3D12_RESOURCE_STATE_COPY_SOURCE;
    // SAFETY: an open list of ours and a live texture.
    let read = unsafe {
        if state != copy {
            list.list
                .ResourceBarrier(&[crate::host::transition(texture, state, copy)]);
        }
        let read = read_back(&host.device, &list.list, texture);
        // The stencil plane of a depth-stencil target, which the title's
        // lighting reads (material class in the low bits; 0x08 "inside a
        // room", which the lamps of a room are multiplied by): the plane
        // after the depth plane of a one-level, one-slice texture.
        let stencil = (matches!(d.Format.0, 19..=22 | 44..=47)
            && d.MipLevels == 1
            && d.DepthOrArraySize == 1)
            .then(|| read_back_of(&host.device, &list.list, texture, 1))
            .flatten();
        if state != copy {
            list.list
                .ResourceBarrier(&[crate::host::transition(texture, copy, state)]);
        }
        (read, stencil)
    };
    let (read, stencil) = read;
    list.close();
    host.execute(&[&list]);
    if let Some((buffer, footprint)) = stencil {
        stencil_values(&buffer, &footprint, frame, resource);
    }
    let Some((buffer, footprint)) = read else { return };
    let f = footprint.Footprint;
    let mut data = std::ptr::null_mut();
    // SAFETY: the copy finished; the mapping spans the footprint's rows.
    unsafe {
        if buffer.Map(0, None, Some(&mut data)).is_err() {
            return;
        }
        let pitch = f.RowPitch as usize;
        let rows = std::slice::from_raw_parts(data.cast::<u8>(), pitch * f.Height as usize);
        // A 24-bit depth plane comes back as 32-bit words: which depth values
        // (low 24 bits) and which top bytes it holds, the most common first.
        if matches!(d.Format.0, 44..=47) {
            let mut depths = std::collections::HashMap::<u32, u32>::new();
            let mut tops = std::collections::HashMap::<u32, u32>::new();
            for y in 0..f.Height as usize {
                for x in 0..f.Width as usize {
                    let at = y * pitch + x * 4;
                    let w = u32::from_le_bytes(rows[at..at + 4].try_into().unwrap_or([0; 4]));
                    *depths.entry(w & 0xff_ffff).or_default() += 1;
                    *tops.entry(w >> 24).or_default() += 1;
                }
            }
            let most = |m: std::collections::HashMap<u32, u32>| {
                let mut v: Vec<(u32, u32)> = m.into_iter().collect();
                v.sort_by_key(|e| std::cmp::Reverse(e.1));
                (v.len(), v.into_iter().take(12).collect::<Vec<_>>())
            };
            tracing::info!(?resource, depths = ?most(depths), top_bytes = ?most(tops), "24-bit depth plane values (value, texels)");
        }
        if let Some(rgba) = to_rgba(f.Format.0 as u32, f.Width as usize, f.Height as usize, pitch, rows) {
            let name = format!(
                "frame-{frame}-{i:02}-{resource:?}-{}x{}-f{}",
                d.Width, d.Height, d.Format.0
            );
            save_as(&name, f.Width, f.Height, f.Width as usize * 4, false, &rgba, 2);
        } else {
            tracing::info!(format = f.Format.0, ?resource, "target not dumped: format");
        }
        buffer.Unmap(0, None);
    }
}

/// Logs which values a stencil plane read back by [`read_back_of`] holds
/// (the most common first), how many texels have each of its bits, and the
/// box of the texels with bit 3 (0x08).
fn stencil_values(
    buffer: &ID3D12Resource,
    footprint: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    frame: u64,
    resource: crate::object::This,
) {
    let f = footprint.Footprint;
    let (w, h, pitch) = (f.Width as usize, f.Height as usize, f.RowPitch as usize);
    let mut data = std::ptr::null_mut();
    // SAFETY: the copy finished; the mapping spans the footprint's rows of
    // one byte a texel, the last row without its padding.
    unsafe {
        if h == 0 || w == 0 || buffer.Map(0, None, Some(&mut data)).is_err() {
            return;
        }
        let rows = std::slice::from_raw_parts(data.cast::<u8>(), pitch * (h - 1) + w);
        let mut values = [0u32; 256];
        let mut bits = [0u32; 8];
        // Left, top, right, bottom of the texels with bit 3.
        let mut inside = [u32::MAX, u32::MAX, 0, 0];
        for y in 0..h {
            for (x, &v) in rows[y * pitch..y * pitch + w].iter().enumerate() {
                values[usize::from(v)] += 1;
                for (bit, count) in bits.iter_mut().enumerate() {
                    *count += u32::from(v >> bit & 1);
                }
                if v & 8 != 0 {
                    inside = [
                        inside[0].min(x as u32),
                        inside[1].min(y as u32),
                        inside[2].max(x as u32),
                        inside[3].max(y as u32),
                    ];
                }
            }
        }
        let mut most: Vec<(usize, u32)> = values
            .iter()
            .copied()
            .enumerate()
            .filter(|v| v.1 != 0)
            .collect();
        most.sort_by_key(|v| std::cmp::Reverse(v.1));
        most.truncate(16);
        tracing::info!(
            frame,
            ?resource,
            width = w,
            height = h,
            values = ?most,
            texels_with_bit = ?bits,
            bit3_box = ?inside,
            "stencil plane values (value, texels)"
        );
        buffer.Unmap(0, None);
    }
}

fn half(h: u16) -> f32 {
    let (sign, exp, man) = ((h >> 15) as u32, ((h >> 10) & 0x1f) as i32, (h & 0x3ff) as f32);
    let v = match exp {
        0 => man * 2f32.powi(-24),
        31 => f32::INFINITY,
        e => (1.0 + man / 1024.0) * 2f32.powi(e - 15),
    };
    if sign == 1 { -v } else { v }
}

/// A float packed in `bits` with a 5-bit exponent and `man` mantissa bits
/// (the 11- and 10-bit floats of R11G11B10).
fn small_float(bits: u32, man: u32) -> f32 {
    let (exp, m) = ((bits >> man) & 0x1f, (bits & ((1 << man) - 1)) as f32);
    match exp {
        0 => m / (1 << man) as f32 * 2f32.powi(-14),
        31 => f32::INFINITY,
        e => (1.0 + m / (1 << man) as f32) * 2f32.powi(e as i32 - 15),
    }
}

/// Converts host-format rows to opaque 8-bit RGBA for the RGB dump. Byte
/// colors and packed 16-bit BGR use their full channel range. Other formats
/// scale to the highest finite value, with single channels shown in grey
/// from their lowest to highest value. Returns `None` for unhandled formats.
fn to_rgba(format: u32, w: usize, h: usize, pitch: usize, rows: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(w * h * 4);
    let px = |x: usize, y: usize, n: usize| &rows[y * pitch + x * n..][..n];
    let u16_at = |b: &[u8], i: usize| u16::from_le_bytes([b[2 * i], b[2 * i + 1]]);
    let u32_at = |b: &[u8], i: usize| u32::from_le_bytes(b[4 * i..4 * i + 4].try_into().unwrap_or([0; 4]));
    // Channel values as floats, then scaled.
    type Decode = Box<dyn Fn(&[u8]) -> [f32; 3]>;
    let (n, decode): (usize, Decode) = match format {
        27..=32 => {
            for y in 0..h {
                for x in 0..w {
                    let p = px(x, y, 4);
                    out.extend([p[0], p[1], p[2], 255]);
                }
            }
            return Some(out);
        }
        // B8G8R8A8 and B8G8R8X8 (89 is R10G10B10_XR_BIAS_A2, below).
        87 | 88 | 90..=93 => {
            for y in 0..h {
                for x in 0..w {
                    let p = px(x, y, 4);
                    out.extend([p[2], p[1], p[0], 255]);
                }
            }
            return Some(out);
        }
        // B5G6R5, B5G5R5A1 and B4G4R4A4 store blue in the low bits.
        85 | 86 | 115 => {
            let (rb_bits, green_bits) = match format {
                85 => (5, 6),
                86 => (5, 5),
                _ => (4, 4),
            };
            let unorm8 = |value: u16, bits: u32| {
                let max = (1u16 << bits) - 1;
                (((value & max) * 255 + max / 2) / max) as u8
            };
            for y in 0..h {
                for x in 0..w {
                    let word = u16_at(px(x, y, 2), 0);
                    out.extend([
                        unorm8(word >> (rb_bits + green_bits), rb_bits),
                        unorm8(word >> rb_bits, green_bits),
                        unorm8(word, rb_bits),
                        255,
                    ]);
                }
            }
            return Some(out);
        }
        23..=25 | 89 => (4, Box::new(move |b| {
            let v = u32_at(b, 0);
            [(v & 0x3ff) as f32, (v >> 10 & 0x3ff) as f32, (v >> 20 & 0x3ff) as f32]
        })),
        26 => (4, Box::new(move |b| {
            let v = u32_at(b, 0);
            [small_float(v & 0x7ff, 6), small_float(v >> 11 & 0x7ff, 6), small_float(v >> 22, 5)]
        })),
        9..=10 => (8, Box::new(move |b| [half(u16_at(b, 0)), half(u16_at(b, 1)), half(u16_at(b, 2))])),
        11..=14 => (8, Box::new(move |b| [0, 1, 2].map(|i| f32::from(u16_at(b, i))))),
        33..=34 => (4, Box::new(move |b| [half(u16_at(b, 0)), half(u16_at(b, 1)), 0.0])),
        1..=2 => (16, Box::new(move |b| {
            [0, 1, 2].map(|i| f32::from_bits(u32_at(b, i)))
        })),
        15..=16 => (8, Box::new(move |b| [f32::from_bits(u32_at(b, 0)), f32::from_bits(u32_at(b, 1)), 0.0])),
        35..=38 => (4, Box::new(move |b| [f32::from(u16_at(b, 0)), f32::from(u16_at(b, 1)), 0.0])),
        48..=52 => (2, Box::new(move |b| [f32::from(b[0]), f32::from(b[1]), 0.0])),
        // Single channels (and depth planes): shown as grey.
        39..=43 => (4, Box::new(move |b| [f32::from_bits(u32_at(b, 0)); 3])),
        19..=22 => (8, Box::new(move |b| [f32::from_bits(u32_at(b, 0)); 3])),
        44..=47 => (4, Box::new(move |b| [(u32_at(b, 0) & 0xff_ffff) as f32; 3])),
        53..=54 => (2, Box::new(move |b| [half(u16_at(b, 0)); 3])),
        55..=59 => (2, Box::new(move |b| [f32::from(u16_at(b, 0)); 3])),
        60..=65 => (1, Box::new(move |b| [f32::from(b[0]); 3])),
        _ => return None,
    };
    let values: Vec<[f32; 3]> = (0..h)
        .flat_map(|y| (0..w).map(move |x| (x, y)))
        .map(|(x, y)| decode(px(x, y, n)))
        .collect();
    let grey = matches!(format, 19..=22 | 39..=47 | 53..=65);
    let finite = values.iter().flatten().copied().filter(|v| v.is_finite());
    let (low, high) = if grey {
        finite.fold((f32::MAX, f32::MIN), |(l, h), v| (l.min(v), h.max(v)))
    } else {
        (0.0, finite.fold(0.0f32, f32::max))
    };
    let range = (high - low).max(1e-20);
    // How much of the target holds anything: a lit scene and one bright
    // lamp on black have the same highest value.
    let above = values.iter().filter(|v| v.iter().any(|c| c.is_finite() && *c > low)).count();
    let nonzero_percent = above as f32 * 100.0 / values.len().max(1) as f32;
    tracing::info!(format, low, high, nonzero_percent, "target values");
    // Small float colour targets (the 1x1 luminance and exposure targets):
    // every texel's four channels as numbers. The picture has no alpha and
    // eight bits a channel.
    if matches!(format, 1..=2 | 9..=10) && w * h <= 16 {
        for y in 0..h {
            for x in 0..w {
                let b = px(x, y, n);
                let texel: [f32; 4] = if n == 16 {
                    [0, 1, 2, 3].map(|i| f32::from_bits(u32_at(b, i)))
                } else {
                    [0, 1, 2, 3].map(|i| half(u16_at(b, i)))
                };
                tracing::info!(format, x, y, ?texel, "target texel");
            }
        }
    }
    // R32_UINT targets hold packed light (the title reads them back as
    // R9G9B9E5): as floats their low, high and share above the lowest say
    // nothing. How many texels hold any bit, and the highest channel value
    // of the packed reading.
    if format == 42 {
        let (mut set, mut packed_high) = (0usize, 0f32);
        for y in 0..h {
            for x in 0..w {
                let v = u32_at(px(x, y, 4), 0);
                set += usize::from(v != 0);
                let scale = 2f32.powi((v >> 27) as i32 - 24);
                for k in 0..3 {
                    packed_high = packed_high.max((v >> (9 * k) & 0x1ff) as f32 * scale);
                }
            }
        }
        let nonzero_bits_percent = set as f32 * 100.0 / (w * h).max(1) as f32;
        tracing::info!(format, nonzero_bits_percent, packed_high, "target values as packed light");
        // The same words as plain numbers, by class. Tables of indices and
        // masks say nothing as floats or as packed light: in the title's
        // shadow-mask table 0 is a block in full shadow, a word up to 255
        // one value for the block, above that a block of the atlas, and all
        // ones a texel never written.
        let mut classes = [0usize; 6];
        let mut highest_below_ones = 0u32;
        for y in 0..h {
            for x in 0..w {
                let v = u32_at(px(x, y, 4), 0);
                classes[match v {
                    0 => 0,
                    1..=15 => 1,
                    16..=255 => 2,
                    256..=0x00ff_ffff => 3,
                    u32::MAX => 5,
                    _ => 4,
                }] += 1;
                if v != u32::MAX {
                    highest_below_ones = highest_below_ones.max(v);
                }
            }
        }
        tracing::info!(
            format,
            width = w,
            height = h,
            zero = classes[0],
            one_to_15 = classes[1],
            to_255 = classes[2],
            to_24_bits = classes[3],
            larger = classes[4],
            all_ones = classes[5],
            highest_below_ones,
            "target values as words"
        );
    }
    // Float colours (scene light) go through a cube root: scaled linearly,
    // a night scene under one lamp of 50 units came out all black.
    let curved = matches!(format, 1..=2 | 9..=10 | 26);
    for v in values {
        let c = v.map(|c| {
            let t = ((c - low) / range).clamp(0.0, 1.0);
            ((if curved { t.cbrt() } else { t }) * 255.0) as u8
        });
        out.extend([c[0], c[1], c[2], 255]);
    }
    Some(out)
}

/// Pixels per side of the grid [`probe`] reads.
const GRID: u32 = 4;
/// Bytes between the grid's pixels in the readback buffer (placement
/// alignment).
const SLOT: u64 = 512;

/// Records copies of a grid of pixels of `texture` (8-bit color, in the
/// copy-source state) into a readback buffer, to tell black frames.
pub(crate) fn probe(
    device: &ID3D12Device,
    list: &ID3D12GraphicsCommandList,
    texture: &ID3D12Resource,
) -> Option<ID3D12Resource> {
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    if !matches!(d.Format.0, 27..=29 | 87 | 90 | 91) {
        return None;
    }
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: SLOT * u64::from(GRID * GRID),
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
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_READBACK,
        ..Default::default()
    };
    let mut buffer: Option<ID3D12Resource> = None;
    // SAFETY: plain creation; an open list; locations naming live resources.
    unsafe {
        device
            .CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut buffer,
            )
            .ok()?;
        let buffer = buffer?;
        for i in 0..GRID * GRID {
            let (x, y) = (
                (2 * (i % GRID) + 1) * d.Width as u32 / (2 * GRID),
                (2 * (i / GRID) + 1) * d.Height / (2 * GRID),
            );
            let to = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&buffer),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: SLOT * u64::from(i),
                        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                            Format: d.Format,
                            Width: 1,
                            Height: 1,
                            Depth: 1,
                            RowPitch: 256,
                        },
                    },
                },
            };
            let from = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(texture),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: 0,
                },
            };
            let pixel = D3D12_BOX {
                left: x,
                top: y,
                front: 0,
                right: x + 1,
                bottom: y + 1,
                back: 1,
            };
            list.CopyTextureRegion(&to, 0, 0, 0, &from, Some(&pixel));
        }
        Some(buffer)
    }
}

/// Whether every pixel [`probe`] read is (near) black.
pub(crate) fn black(buffer: &ID3D12Resource) -> bool {
    let mut data = std::ptr::null_mut();
    // SAFETY: the copies finished (every submission is waited for).
    unsafe {
        if buffer.Map(0, None, Some(&mut data)).is_err() {
            return false;
        }
        let bytes =
            std::slice::from_raw_parts(data.cast::<u8>(), (SLOT * u64::from(GRID * GRID)) as usize);
        let level = bytes
            .chunks(SLOT as usize)
            .flat_map(|pixel| pixel[..3].iter().copied())
            .max()
            .unwrap_or(0);
        buffer.Unmap(0, None);
        let mut levels = LEVELS.lock().unwrap_or_else(|e| e.into_inner());
        if levels.len() == 12 {
            levels.pop_front();
        }
        levels.push_back(level);
        level < 8
    }
}

/// The brightest probed channel of the last frames, oldest first: a fade
/// darkens over several, a glitch drops at once.
static LEVELS: std::sync::Mutex<std::collections::VecDeque<u8>> =
    std::sync::Mutex::new(std::collections::VecDeque::new());

pub(crate) fn recent_levels() -> Vec<u8> {
    let levels = LEVELS.lock().unwrap_or_else(|e| e.into_inner());
    levels.iter().copied().collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn rgba_capture_respects_the_placement_and_unpadded_last_row() {
        let range = super::rgba_readback_range;
        assert_eq!(range(512, 3, 2, 256, 780), Some(512..780));
        assert_eq!(range(512, 3, 2, 256, 779), None);
        assert_eq!(range(0, 1, 1, 256, 4), Some(0..4));
        assert_eq!(range(0, 3, 0, 256, 780), None);
        assert_eq!(range(0, 3, 2, 8, 780), None);
        assert_eq!(range(u64::MAX, 1, 1, 256, u64::MAX), None);
    }

    #[test]
    fn scene_pixels_are_owned_and_preserve_native_resolution() {
        let mut readback = [
            255, 255, 255, 255, // space before the footprint
            3, 2, 1, 0, 6, 5, 4, 0, 255, 255, 255, 255, // first row and padding
            9, 8, 7, 0, 12, 11, 10, 0, // last row, without trailing padding
        ];
        let range = super::rgba_readback_range(4, 2, 2, 12, readback.len() as u64).unwrap();
        let scene = super::pack_rgba_rows(2, 2, 12, true, &readback[range.clone()], 1);
        let preview = super::pack_rgba_rows(2, 2, 12, true, &readback[range], 2);
        readback.fill(0);
        assert_eq!(scene, (2, 2, vec![0, 1, 2, 3, 4, 5, 6, 0, 7, 8, 9, 10, 11, 12]));
        assert_eq!(preview, (1, 1, vec![0, 1, 2, 3]));
    }

    #[test]
    fn crc_matches_the_png_check_value() {
        assert_eq!(super::crc32(b"IEND"), 0xae42_6082);
        // The check value every CRC-32 implementation is tested with.
        assert_eq!(super::crc32(b"123456789"), 0xcbf4_3926);
    }

    #[test]
    fn packed_bgr_targets_preserve_color_and_skip_row_padding() {
        // Each 2x2 target has red, green, blue and a mixed color. The first
        // row has a padding word, and alpha bits do not affect the RGB dump.
        for (format, words, mixed) in [
            (85, [0xf800u16, 0x07e0, 0x001f, 0x8408], [132, 130, 66, 255]),
            (86, [0x7c00, 0x03e0, 0x801f, 0xc208], [132, 132, 66, 255]),
            (115, [0x0f00, 0x00f0, 0xf00f, 0xf841], [136, 68, 17, 255]),
        ] {
            let [red, green, blue, blend] = words;
            let rows = [red, green, 0xcafe, blue, blend]
                .map(u16::to_le_bytes)
                .concat();
            let expected = [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255], mixed].concat();
            assert_eq!(
                super::to_rgba(format, 2, 2, 6, &rows),
                Some(expected),
                "format {format}"
            );
        }
    }

    /// R16G16B16A16_UNORM is integers, not half floats; 89 is 10:10:10:2,
    /// not B8G8R8A8.
    #[test]
    fn unorm16_and_xr_bias_targets_decode_as_their_formats() {
        // Red 0x8000 (a half-float -0), green 0x4000, blue 0.
        let unorm16 = [0x00, 0x80, 0x00, 0x40, 0, 0, 0xff, 0xff];
        assert_eq!(super::to_rgba(11, 1, 1, 8, &unorm16), Some(vec![255, 127, 0, 255]));
        // Red 0x3ff, green 0x1ff, blue 0, alpha 3.
        let xr = (0x3ffu32 | 0x1ff << 10 | 3 << 30).to_le_bytes();
        assert_eq!(super::to_rgba(89, 1, 1, 4, &xr), Some(vec![255, 127, 0, 255]));
        // B8G8R8A8 still swaps to RGBA.
        assert_eq!(super::to_rgba(87, 1, 1, 4, &[1, 2, 3, 4]), Some(vec![3, 2, 1, 255]));
    }
    /// A CRC-32 made a bit at a time: not the table the writer uses.
    fn crc_by_bits(bytes: &[u8]) -> u32 {
        let mut crc = !0u32;
        for &b in bytes {
            crc ^= u32::from(b);
            for _ in 0..8 {
                crc = (crc >> 1) ^ (0xedb8_8320 & (crc & 1).wrapping_neg());
            }
        }
        !crc
    }

    /// Reads a PNG as a decoder would: the signature, every chunk's length
    /// and check value, the header's fields, the packed rows unpacked, the
    /// end chunk last. Gives the size and the rows, each behind its filter
    /// byte.
    fn unpacked(png: &[u8]) -> (u32, u32, Vec<u8>) {
        assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n", "signature");
        let (mut at, mut header, mut packed, mut ended) = (8, None, Vec::new(), false);
        while at < png.len() {
            assert!(!ended, "a chunk behind the end chunk");
            let len = u32::from_be_bytes(png[at..at + 4].try_into().unwrap()) as usize;
            let kind = &png[at + 4..at + 8];
            let data = &png[at + 8..at + 8 + len];
            let crc = u32::from_be_bytes(png[at + 8 + len..at + 12 + len].try_into().unwrap());
            assert_eq!(crc, crc_by_bits(&png[at + 4..at + 8 + len]), "check value of {kind:?}");
            match kind {
                b"IHDR" => {
                    assert!(header.is_none() && at == 8, "the header comes first, once");
                    assert_eq!(len, 13);
                    // 8 bits a channel, RGB, deflate, filter method 0, not interlaced.
                    assert_eq!(&data[8..], &[8, 2, 0, 0, 0]);
                    header = Some((
                        u32::from_be_bytes(data[..4].try_into().unwrap()),
                        u32::from_be_bytes(data[4..8].try_into().unwrap()),
                    ));
                }
                b"IDAT" => packed.extend_from_slice(data),
                b"IEND" => {
                    assert_eq!(len, 0);
                    ended = true;
                }
                other => panic!("an unexpected chunk {other:?}"),
            }
            at += 12 + len;
        }
        assert!(ended, "no end chunk");
        let (w, h) = header.expect("no header");
        let mut rows = vec![0u8; (w as usize * 3 + 1) * h as usize];
        let (out, rc) = zlib_rs::decompress_slice(&mut rows, &packed, zlib_rs::InflateConfig::default());
        assert_eq!(rc, zlib_rs::ReturnCode::Ok, "the packed rows do not unpack");
        assert_eq!(out.len(), (w as usize * 3 + 1) * h as usize, "rows for the header's size");
        assert!(rows.chunks_exact(w as usize * 3 + 1).all(|row| row[0] == 0), "filter type 0 in every row");
        (w, h, rows)
    }

    /// Rows of a picture no two of which are alike, each behind its filter
    /// byte, as `save_as` makes them.
    fn rows(w: u32, h: u32, seed: u32) -> Vec<u8> {
        let mut raw = Vec::new();
        for y in 0..h {
            raw.push(0);
            for x in 0..w {
                let v = (x * 7 + y * 13 + seed * 31).wrapping_mul(2_654_435_761);
                raw.extend([(v >> 24) as u8, (v >> 16) as u8, (v >> 8) as u8]);
            }
        }
        raw
    }

    /// Pictures handed to the capture thread are on disk, whole and valid,
    /// when the end of the process has waited for it: through the process's
    /// own line, as a frame's picture goes. (The one test that uses that
    /// line: it is closed afterwards.)
    #[test]
    fn pictures_in_the_line_are_valid_png_files_when_finish_returns() {
        use std::sync::mpsc::{RecvTimeoutError, channel, sync_channel};
        use std::time::{Duration, Instant};

        let dir = tempfile::tempdir().unwrap();
        let (w, h) = (37, 21);
        let (release, held) = std::sync::mpsc::channel::<()>();
        let mut held = Some(held);
        let worker = super::Aside::start(move |(path, w, h, raw)| {
            if let Some(held) = held.take() {
                let _ = held.recv();
            }
            super::write(&path, w, h, &raw);
        })
        .unwrap();
        {
            let mut line = super::LINE.lock().unwrap();
            assert!(matches!(*line, super::Line::Unused));
            *line = super::Line::Aside(worker);
        }
        let pictures: Vec<_> = (0..8).map(|i| (dir.path().join(format!("frame-{i}.png")), rows(w, h, i))).collect();
        for (path, raw) in &pictures {
            assert!(super::write_aside(path.clone(), w, h, raw.clone()), "the picture did not go to the thread");
        }
        std::thread::scope(|scope| {
            scope.spawn(super::finish);
            let deadline = Instant::now() + Duration::from_secs(2);
            while !matches!(*super::LINE.lock().unwrap(), super::Line::Caller) {
                assert!(Instant::now() < deadline, "the first exit did not close the line");
                std::thread::yield_now();
            }

            let (started, start) = sync_channel(0);
            let (returned, done) = channel();
            scope.spawn(move || {
                started.send(()).unwrap();
                super::finish();
                returned.send(()).unwrap();
            });
            start.recv_timeout(Duration::from_secs(2)).unwrap();
            let early = done.recv_timeout(Duration::from_millis(100));
            drop(release);
            assert_eq!(early, Err(RecvTimeoutError::Timeout), "the second exit bypassed the drain");
            done.recv_timeout(Duration::from_secs(3)).unwrap();
        });
        for (path, raw) in &pictures {
            let png = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            assert_eq!(unpacked(&png), (w, h, raw.clone()), "{}", path.display());
        }
        // After that a picture is written by its caller, at once.
        let late = dir.path().join("late.png");
        assert!(!super::write_aside(late.clone(), w, h, rows(w, h, 99)));
        assert_eq!(unpacked(&std::fs::read(&late).unwrap()), (w, h, rows(w, h, 99)));
        assert_eq!(super::finish_within(std::time::Duration::ZERO), 0);
    }

    /// Closing the line does not drop what is in it: the thread works off
    /// every picture that was waiting, in order, before the wait ends.
    #[test]
    fn a_closed_line_is_worked_off_before_the_wait_ends() {
        let (open, gate) = std::sync::mpsc::channel::<()>();
        let (seen, order) = std::sync::mpsc::channel::<u32>();
        let mut first = true;
        let aside = super::Aside::start(move |(_, n, _, _)| {
            // The first picture waits: the others are still in the line when
            // it is closed.
            if std::mem::take(&mut first) {
                let _ = gate.recv();
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
            seen.send(n).unwrap();
        })
        .unwrap();
        for n in 0..10 {
            assert!(aside.give((std::path::PathBuf::new(), n, 0, Vec::new())).is_ok());
        }
        open.send(()).unwrap();
        assert_eq!(aside.finish(std::time::Duration::from_secs(60)), 0);
        assert_eq!(order.try_iter().collect::<Vec<_>>(), (0..10).collect::<Vec<_>>());
    }

    /// A thread that does not come back (a disk that does not answer) holds
    /// the end of the process for the bound and no longer, and what it had
    /// not written is counted.
    #[test]
    fn the_wait_for_the_line_is_bounded_and_counts_what_is_left() {
        let (release, held) = std::sync::mpsc::channel::<()>();
        let aside = super::Aside::start(move |_| {
            let _ = held.recv();
        })
        .unwrap();
        for _ in 0..3 {
            assert!(aside.give((std::path::PathBuf::new(), 0, 0, Vec::new())).is_ok());
        }
        let started = std::time::Instant::now();
        assert_eq!(aside.finish(std::time::Duration::from_millis(100)), 3);
        let waited = started.elapsed();
        assert!(waited >= std::time::Duration::from_millis(100), "{waited:?}");
        assert!(waited < std::time::Duration::from_secs(20), "{waited:?}");
        drop(release);
    }
}
