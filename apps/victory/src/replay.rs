//! `victory replay`: a captured frame run again (docs/replay.md).

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Args, ValueEnum};
use victory_hle::replay::play::{self, Options};

#[derive(Clone, Copy, ValueEnum)]
enum Timing {
    /// Each list.
    Lists,
    /// Each command (adds some time of its own).
    Detailed,
}

#[derive(Clone, Copy, ValueEnum)]
enum Buffers {
    AsRecorded,
    /// All in the card's memory.
    Vram,
    /// All in the PC's memory.
    Sysmem,
}

#[derive(Args)]
pub(crate) struct Replay {
    /// The folder given as VICTORY_RECORD (its last capture) or a capture.
    folder: PathBuf,
    /// How often the frame is run.
    #[arg(long, default_value_t = 5)]
    loops: u32,
    /// Do not put back, between runs, what a run changed.
    #[arg(long)]
    no_restore: bool,
    #[arg(long, value_enum, default_value_t = Timing::Lists)]
    timing: Timing,
    /// Pipelines listed with detailed timing.
    #[arg(long, default_value_t = 20)]
    top: usize,
    /// Where the picture goes (default: replay.png in the capture).
    #[arg(long)]
    image: Option<PathBuf>,
    /// Run the frame unchanged first, and print both.
    #[arg(long)]
    compare: bool,
    /// Leave out unordered-access barriers.
    #[arg(long)]
    no_barriers: bool,
    /// Join barrier calls with no work between them.
    #[arg(long)]
    merge_barriers: bool,
    /// Leave out the argument clamps: what they bound is kept from a first run.
    #[arg(long)]
    skip_clamps: bool,
    /// The argument clamps in batches, as VICTORY_INDIRECT_BATCH does (2:
    /// a wait before every indirect dispatch).
    #[arg(long, num_args = 0..=1, default_missing_value = "1", default_value_t = 0)]
    batch_clamps: u32,
    /// Leave out the hash and compare passes over card buffers.
    #[arg(long)]
    skip_hash_passes: bool,
    /// Where buffers lie.
    #[arg(long, value_enum, default_value_t = Buffers::AsRecorded)]
    buffers: Buffers,
    /// With --buffers, only the buffers all of these match: `name=<text>`,
    /// `min=<bytes>`, `max=<bytes>`, `seen=<No, Cached, Combined>`,
    /// `id=<a>..<b>`, or `!` and one of those (may be given several times).
    #[arg(long)]
    only_buffers: Vec<String>,
    /// One ExecuteCommandLists for the whole frame.
    #[arg(long)]
    one_submit: bool,
    /// Let copies of what the CPU wrote also overwrite what the card wrote in
    /// the replay (as the recorded commands say).
    #[arg(long)]
    no_merge: bool,
    /// A full wait after every draw, dispatch, copy and clear.
    #[arg(long)]
    wait_all: bool,
    /// Each list sent on its own, and waited for.
    #[arg(long)]
    wait_lists: bool,
    /// Milliseconds to wait after sending each submission: the card stands
    /// idle between them, as it does while the title prepares the next.
    #[arg(long, default_value_t = 0.0)]
    pace: f64,
    /// Keep the card's clocks up meanwhile (VICTORY_CARD_AWAKE=1).
    #[arg(long)]
    awake: bool,
    /// The system's breadcrumbs after every command (VICTORY_D3D12_DRED=light).
    #[arg(long)]
    breadcrumbs: bool,
    /// Run only these lists, by their place in the frame (`3..7`, or `5`);
    /// of the others only barriers.
    #[arg(long)]
    only: Option<String>,
    /// Leave out what runs a pipeline whose name or DXIL hash has this in it
    /// (may be given several times).
    #[arg(long = "skip-shader")]
    skip_shader: Vec<String>,
    /// Find the commands that make identical runs differ: each list is run
    /// this often from the same memory.
    #[arg(long, num_args = 0..=1, default_missing_value = "3")]
    diverge: Option<u32>,
    /// Find where the picture's variants are decided: from each list on, the
    /// rest of the frame is run this often from the same memory.
    #[arg(long, num_args = 0..=1, default_missing_value = "5")]
    diverge_picture: Option<u32>,
    /// Run each draw of one list (--lists) whose pipeline's name has this in
    /// it alone, --diverge times: how many outcomes it has.
    #[arg(long)]
    draws: Option<String>,
    /// Say which lists (with --lists: commands) change words of buffers:
    /// `<resource>:<offset>:<bytes>` (hexadecimal; repeatable).
    #[arg(long)]
    watch: Vec<String>,
    /// With --diverge-picture and one list in --lists: its commands (`0..900`)
    /// and how many are stepped over at a time.
    #[arg(long)]
    commands: Option<String>,
    #[arg(long, default_value_t = 1)]
    step: usize,
    /// With --diverge and --diverge-picture: only these lists (`3..7`).
    #[arg(long)]
    lists: Option<String>,
    /// With --diverge: the most commands named of one list.
    #[arg(long, default_value_t = 4)]
    roots: u32,
    /// Name resources the frame changes that the replay does not put back.
    #[arg(long)]
    check_written: bool,
    /// Run the frame twice and say which words the copies of the CPU's sides
    /// of card buffers bring that the card holds another value of, and what
    /// shaders wrote that was not declared as written.
    #[arg(long)]
    upcopies: bool,
    /// Name what the heap slots hold that a command's constants name:
    /// `<list>:<command>`.
    #[arg(long)]
    slots: Option<String>,
    /// Only write the DXIL of pipelines with this in their name to the
    /// capture's `shaders` folder.
    #[arg(long)]
    dump: Option<String>,
    /// Only write the frame's buffers and the commands of these lists (`3..7`).
    #[arg(long)]
    listing: Option<String>,
    /// Only write what the dispatches of these lists have bound (`3..7`).
    #[arg(long)]
    flow: Option<String>,
    /// With --flow: draws too.
    #[arg(long)]
    flow_draws: bool,
    /// With --flow: heap slots shown from each place named.
    #[arg(long, default_value_t = 40)]
    flow_slots: u32,
    /// Run the frame once and save textures as pictures in this folder, with
    /// what values they hold (see --texture, --until).
    #[arg(long)]
    images: Option<PathBuf>,
    /// With --images: `<resource>[.<subresource>]` (repeatable); none: all
    /// that the frame's shaders write.
    #[arg(long)]
    texture: Vec<String>,
    /// With --images and no --texture: from this many texels on.
    #[arg(long, default_value_t = 4096)]
    least: u64,
    /// With --images: the widest picture.
    #[arg(long, default_value_t = 960)]
    width: u32,
    /// With --images: the values too.
    #[arg(long)]
    raw: bool,
    /// The last command that runs, `<list>:<command>`; after it only barriers.
    #[arg(long)]
    until: Option<String>,
    /// Say which commands change the pixel: `<x>,<y>` of the presented
    /// picture, or `<resource>[.<subresource>]:<x>,<y>` of a texture
    /// (repeatable). With --lists, only those are looked into.
    #[arg(long)]
    pixel: Vec<String>,
    /// With --pixel: the size of the picture the points are of (`1280x720`),
    /// if it is another than the presented one's.
    #[arg(long)]
    of: Option<String>,
    /// With --pixel: depth targets too.
    #[arg(long)]
    depth: bool,
    /// With --pixel: leave out textures that shaders may write (unordered
    /// access).
    #[arg(long)]
    targets_only: bool,
    /// With --pixel: indirect calls whose draws are run one by one:
    /// `<list>:<command>` (repeatable).
    #[arg(long)]
    split: Vec<String>,
    /// Take shaders from this folder where it has them (`<hash>.dxil`, by the
    /// hash the runtime names the title's shader by).
    #[arg(long)]
    shaders: Option<PathBuf>,
    /// With --compare: the folder the second run's shaders are from.
    #[arg(long)]
    shaders_b: Option<PathBuf>,
    /// Only translate the capture's shaders, into this folder (for --shaders),
    /// with the translator's settings of this run. Containers are looked up in
    /// `containers` beside the capture, and first collected there from the
    /// title's files.
    #[arg(long)]
    retranslate: Option<PathBuf>,
    /// With --retranslate: the view heap's null slot, and where raw twins of
    /// views start, as the capture's shaders have them (0: none).
    #[arg(long, default_value_t = 0)]
    view_limit: u32,
    #[arg(long, default_value_t = 0)]
    raw_views: u32,
    /// Use the D3D12 debug layer (slow).
    #[arg(long)]
    debug_layer: bool,
}

/// `3..7` (both included) or `5`.
fn range(text: &str) -> Option<(u32, u32)> {
    match text.split_once("..") {
        Some((a, b)) => Some((a.trim().parse().ok()?, b.trim().parse().ok()?)),
        None => text.trim().parse().ok().map(|n| (n, n)),
    }
}

/// Two numbers with `by` between them.
fn pair(text: &str, by: char) -> Option<(u32, u32)> {
    let (a, b) = text.split_once(by)?;
    Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
}

pub(crate) fn run(r: Replay) -> Result<()> {
    if let Some(text) = &r.dump {
        for line in play::dump(&r.folder, text)? {
            println!("{line}");
        }
        return Ok(());
    }
    if let Some(text) = &r.listing {
        let lists = range(text).with_context(|| format!("--listing {text}"))?;
        return play::listing(&r.folder, lists, &mut |line| println!("{line}"));
    }
    if let Some(text) = &r.flow {
        let lists = range(text).with_context(|| format!("--flow {text}"))?;
        return play::look::flow(&r.folder, lists, r.flow_draws, r.flow_slots, &mut |line| println!("{line}"));
    }
    if let Some(into) = &r.retranslate {
        return retranslate(&r.folder, into, r.view_limit, r.raw_views);
    }
    let until = match &r.until {
        Some(text) => Some(
            text.split_once(':')
                .and_then(|(l, c)| Some((l.parse().ok()?, c.parse().ok()?)))
                .with_context(|| format!("--until {text}: not <list>:<command>"))?,
        ),
        None => None,
    };
    let only = match &r.only {
        Some(text) => Some(
            range(text)
                .with_context(|| format!("--only {text}: not a list or a range of lists (3..7)"))?,
        ),
        None => None,
    };
    let options = Options {
        loops: r.loops.max(1),
        restore: !r.no_restore,
        detailed: matches!(r.timing, Timing::Detailed),
        top: r.top,
        image: r.image,
        buffers: match r.buffers {
            Buffers::AsRecorded => play::Buffers::AsRecorded,
            Buffers::Vram => play::Buffers::Card,
            Buffers::Sysmem => play::Buffers::Pc,
        },
        only_buffers: r.only_buffers,
        no_barriers: r.no_barriers,
        merge_barriers: r.merge_barriers,
        skip_clamps: r.skip_clamps,
        batch_clamps: r.batch_clamps,
        skip_hash_passes: r.skip_hash_passes,
        one_submit: r.one_submit,
        merge: !r.no_merge,
        wait_all: r.wait_all,
        wait_lists: r.wait_lists,
        pace: r.pace,
        awake: r.awake,
        only,
        until,
        skip_shaders: r.skip_shader,
        shaders: r.shaders,
        shaders_b: r.shaders_b,
        compare: r.compare,
        debug: r.debug_layer,
        breadcrumbs: r.breadcrumbs,
    };
    if let Some(text) = &r.slots {
        let (list, command) = text
            .split_once(':')
            .and_then(|(l, c)| Some((l.parse().ok()?, c.parse().ok()?)))
            .with_context(|| format!("--slots {text}: not <list>:<command>"))?;
        let player = play::Player::load(&r.folder, &options)?;
        return player.slots(list, command, &mut |line| println!("{line}"));
    }
    if let Some(folder) = &r.images {
        let mut which = Vec::new();
        for text in &r.texture {
            let (id, sub) = text.split_once('.').unwrap_or((text, "0"));
            let bad = || format!("--texture {text}");
            which.push((id.parse().with_context(bad)?, sub.parse().with_context(bad)?));
        }
        let images = play::look::Images {
            which,
            least: r.least,
            folder: folder.clone(),
            width: r.width,
            raw: r.raw,
        };
        let mut player = play::Player::load(&r.folder, &options)?;
        return player.images(&options, &images, &mut |line| println!("{line}"));
    }
    if !r.pixel.is_empty() {
        let mut player = play::Player::load(&r.folder, &options)?;
        let mut pixels = play::pixel::Pixels {
            depth: r.depth,
            written: !r.targets_only,
            ..Default::default()
        };
        if let Some(text) = &r.lists {
            pixels.lists = Some(range(text).with_context(|| format!("--lists {text}"))?);
        }
        let of = match &r.of {
            Some(text) => Some(pair(text, 'x').with_context(|| format!("--of {text}"))?),
            None => None,
        };
        for text in &r.pixel {
            let bad = || format!("--pixel {text}");
            match text.split_once(':') {
                Some((what, at)) => {
                    let (id, sub) = what.split_once('.').unwrap_or((what, "0"));
                    let (x, y) = pair(at, ',').with_context(bad)?;
                    let id = id.parse().with_context(bad)?;
                    let sub = sub.parse().with_context(bad)?;
                    pixels.texels.push((id, sub, x, y));
                }
                None => {
                    let (mut x, mut y) = pair(text, ',').with_context(bad)?;
                    if let (Some((w, h)), Some((pw, ph))) = (of, player.picture_size()) {
                        x = x * pw / w.max(1);
                        y = y * ph / h.max(1);
                    }
                    pixels.points.push((x, y));
                }
            }
        }
        for text in &r.split {
            pixels
                .split
                .push(pair(text, ':').with_context(|| format!("--split {text}"))?);
        }
        return player.pixel(&options, &pixels, &mut |line| println!("{line}"));
    }
    if r.check_written {
        let mut player = play::Player::load(&r.folder, &options)?;
        return player.check_written(&mut |line| println!("{line}"));
    }
    if r.upcopies {
        let mut player = play::Player::load(&r.folder, &options)?;
        return player.upcopies(&options, &mut |line| println!("{line}"));
    }
    let mut watched = Vec::new();
    for text in &r.watch {
        let mut n = text
            .split(':')
            .map(|p| u64::from_str_radix(p.trim_start_matches("0x"), 16));
        let mut next = || {
            n.next()
                .and_then(Result::ok)
                .with_context(|| format!("--watch {text}"))
        };
        let id = text
            .split(':')
            .next()
            .and_then(|p| p.parse().ok())
            .with_context(|| format!("--watch {text}"))?;
        let _ = next();
        watched.push((id, next()?, next()?));
    }
    if let Some(runs) = r
        .diverge
        .or(r.diverge_picture)
        .or((!watched.is_empty()).then_some(2))
    {
        let lists = match &r.lists {
            Some(text) => Some(range(text).with_context(|| format!("--lists {text}"))?),
            None => None,
        };
        let search = play::diverge::Search {
            runs: runs.max(2),
            lists,
            roots: r.roots,
            wait_all: options.wait_all,
            options: options.clone(),
        };
        let mut player = play::Player::load(&r.folder, &options)?;
        if !watched.is_empty() {
            return player.watch(&search, &watched, &mut |line| println!("{line}"));
        }
        if let Some(text) = &r.draws {
            return player.diverge_draws(&search, text, &mut |line| println!("{line}"));
        }
        if r.diverge_picture.is_some() {
            let commands = match &r.commands {
                Some(text) => Some(range(text).with_context(|| format!("--commands {text}"))?),
                None => None,
            };
            return player
                .diverge_picture(&search, commands, r.step, &mut |line| println!("{line}"));
        }
        return player.diverge(&search, &mut |line| println!("{line}"));
    }
    let reports = play::replay(&r.folder, &options)?;
    for (what, report) in &reports {
        println!("{}", report.text(what));
    }
    if let [(_, before), (_, after)] = &reports[..] {
        println!("{:+.2} ms", after.median_ms() - before.median_ms());
    }
    Ok(())
}

/// Translates the capture's shaders into `into`.
fn retranslate(folder: &std::path::Path, into: &std::path::Path, view_limit: u32, raw_views: u32) -> Result<()> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let wanted = play::title_shaders(folder)?;
    let kept = play::find(folder)?.with_file_name("containers");
    std::fs::create_dir_all(&kept)?;
    std::fs::create_dir_all(into)?;
    let mut missing: std::collections::HashSet<&str> = wanted
        .iter()
        .map(|w| w.1.as_str())
        .filter(|h| !kept.join(format!("{h}.cdxil")).exists())
        .collect();
    if !missing.is_empty() {
        println!("{} of {} containers are looked for in the title's files", missing.len(), wanted.len());
        'scan: for root in crate::default_paths(Vec::new())? {
            for entry in walkdir::WalkDir::new(root).into_iter().filter_map(std::result::Result::ok) {
                let size = entry.metadata().map_or(0, |m| m.len());
                if !entry.file_type().is_file() || size > crate::shader::MAX_FILE {
                    continue;
                }
                let Ok(data) = std::fs::read(entry.path()) else { continue };
                for source in crate::shader::sources(&data) {
                    let Some(bytes) = source.bytes() else { continue };
                    let hash = blake3::hash(&bytes).to_hex()[..16].to_owned();
                    if missing.remove(hash.as_str()) {
                        std::fs::write(kept.join(format!("{hash}.cdxil")), &bytes[..])?;
                        if missing.is_empty() {
                            break 'scan;
                        }
                    }
                }
            }
        }
    }
    println!("{} containers not found", missing.len());
    victory_shader::emit::set_host_max_wave(crate::requested_host_wave().unwrap_or(64));
    victory_shader::emit::set_host_rdna(std::env::var_os("VICTORY_HOST_RDNA").is_none_or(|value| value == "1"));
    victory_shader::emit::set_view_heap_limit(view_limit);
    victory_shader::emit::set_raw_views(raw_views);
    let all: Vec<_> = wanted.iter().collect();
    let (next, done) = (AtomicUsize::new(0), AtomicUsize::new(0));
    let started = std::time::Instant::now();
    std::thread::scope(|s| {
        for _ in 0..std::thread::available_parallelism().map_or(4, |n| n.get()) {
            s.spawn(|| {
                while let Some((stage, hash)) = all.get(next.fetch_add(1, Ordering::Relaxed)) {
                    let Ok(bytes) = std::fs::read(kept.join(format!("{hash}.cdxil"))) else { continue };
                    match victory_shader::emit::translate(&bytes) {
                        Ok(t) => {
                            if std::fs::write(into.join(format!("{hash}.dxil")), &t.dxil).is_ok() {
                                done.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                        Err(e) => println!("{stage} {hash}: {e}"),
                    }
                }
            });
        }
    });
    println!("{} of {} shaders translated, {:.1} s", done.into_inner(), all.len(), started.elapsed().as_secs_f64());
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn ranges() {
        assert_eq!(super::range("3..7"), Some((3, 7)));
        assert_eq!(super::range("5"), Some((5, 5)));
        assert_eq!(super::range("x"), None);
        assert_eq!(super::pair("12,7", ','), Some((12, 7)));
        assert_eq!(super::pair("12", ','), None);
        assert_eq!(super::range("1..x"), None);
    }
}
