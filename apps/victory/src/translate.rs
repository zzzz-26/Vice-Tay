//! The offline batch: every shader found goes through the translator, and
//! what it cannot handle yet is tallied (decision 0003).

pub(crate) mod host_wave;

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;
use victory_shader::emit::host::{self, Host};
use victory_shader::emit::{self, validator::Validator};
use victory_shader::structure::{self, Atom, LaneExit};
use victory_shader::{
    Container, FourCC, HwStage, NativeProgram, RootType, ShaderError, ir, native_programs,
};
use victory_shader::{io, resource};

use crate::shader::{MAX_FILE, Source, scan::FileScan, sources};

#[derive(Default)]
struct Tally {
    containers: usize,
    duplicates: usize,
    programs: usize,
    translated: usize,
    instructions: usize,
    structured: usize,
    vector: usize,
    guarded: usize,
    /// Guards that read a mask the pass cannot see into.
    opaque: usize,
    lane_branches: usize,
    uniform_branches: usize,
    tracked: usize,
    loaded: usize,
    built: usize,
    constant: usize,
    /// Descriptors and scalar accesses with a computed address term.
    dynamic: usize,
    scalar: usize,
    used: usize,
    /// Used-resource entries whose root parameter no descriptor traces
    /// to, by resource and root type.
    untraced_used: BTreeMap<String, usize>,
    /// Root parameters descriptors trace to that no used entry names.
    unlisted: usize,
    /// Vertex and pixel programs, and those whose I/O was raised.
    io_programs: usize,
    io_raised: usize,
    interpolated: usize,
    vertex_values: usize,
    /// Exports by target kind.
    exports: BTreeMap<&'static str, usize>,
    /// Exports still packed as f16 pairs after the export patch.
    compressed: usize,
    /// Vertex and pixel programs every earlier stage handled, and how far
    /// emission got with them.
    emit_programs: usize,
    emitted: usize,
    validated: usize,
    pipelines: usize,
    /// The same for compute programs.
    cs_programs: usize,
    cs_emitted: usize,
    cs_validated: usize,
    cs_pipelines: usize,
    /// The same for mesh programs.
    ms_programs: usize,
    ms_emitted: usize,
    ms_validated: usize,
    ms_pipelines: usize,
    /// Reason → programs stopped by it, and the first one.
    failed: BTreeMap<String, (usize, String)>,
    /// With `--list`: one line per program and outcome.
    list: bool,
    lines: Vec<String>,
}

impl Tally {
    fn fail(&mut self, reason: String, example: impl FnOnce() -> String) {
        if self.list {
            let e = example();
            self.lines.push(format!("{e}  {reason}"));
            self.failed.entry(reason).or_insert((0, e)).0 += 1;
            return;
        }
        self.failed
            .entry(reason)
            .or_insert_with(|| (0, example()))
            .0 += 1;
    }

    fn merge(&mut self, other: Tally) {
        self.containers += other.containers;
        self.duplicates += other.duplicates;
        self.programs += other.programs;
        self.translated += other.translated;
        self.instructions += other.instructions;
        self.structured += other.structured;
        self.vector += other.vector;
        self.guarded += other.guarded;
        self.opaque += other.opaque;
        self.lane_branches += other.lane_branches;
        self.uniform_branches += other.uniform_branches;
        self.tracked += other.tracked;
        self.loaded += other.loaded;
        self.built += other.built;
        self.constant += other.constant;
        self.dynamic += other.dynamic;
        self.scalar += other.scalar;
        self.used += other.used;
        for (k, n) in other.untraced_used {
            *self.untraced_used.entry(k).or_default() += n;
        }
        self.unlisted += other.unlisted;
        self.io_programs += other.io_programs;
        self.io_raised += other.io_raised;
        self.interpolated += other.interpolated;
        self.vertex_values += other.vertex_values;
        self.compressed += other.compressed;
        self.emit_programs += other.emit_programs;
        self.emitted += other.emitted;
        self.validated += other.validated;
        self.pipelines += other.pipelines;
        self.cs_programs += other.cs_programs;
        self.cs_emitted += other.cs_emitted;
        self.cs_validated += other.cs_validated;
        self.cs_pipelines += other.cs_pipelines;
        self.ms_programs += other.ms_programs;
        self.ms_emitted += other.ms_emitted;
        self.ms_validated += other.ms_validated;
        self.ms_pipelines += other.ms_pipelines;
        for (k, n) in other.exports {
            *self.exports.entry(k).or_default() += n;
        }
        self.lines.extend(other.lines);
        for (reason, (n, example)) in other.failed {
            self.failed.entry(reason).or_insert((0, example)).0 += n;
        }
    }
}

/// Emitted shaders by label, shared between threads.
type Kept = std::sync::Arc<Mutex<Vec<(String, emit::HostShader)>>>;

/// The optional checks after emission, one set per thread.
struct Checks {
    /// Shaders named in `VICTORY_PAIRS` (`vs+ps,...`, label substrings),
    /// kept to build those pipelines after the walk.
    keep: Option<Kept>,
    validation_requested: bool,
    validator: Option<Validator>,
    host: Option<(Host, emit::HostShader)>,
}

/// `translate --warm <file>`: the containers whose hash the file names are
/// translated into the run's shader cache, on `jobs` threads; nothing is
/// made on the GPU. A run of the title then finds them there instead of
/// translating each at its first use.
pub fn warm(paths: Vec<PathBuf>, file: &Path, jobs: Option<usize>, wave: Option<u32>) -> Result<()> {
    // `--warm all`: every container found, whatever its hash.
    let all = file.as_os_str() == "all";
    let mut wanted: HashSet<String> = HashSet::new();
    if !all {
        let text = std::fs::read_to_string(file)?;
        for line in text.lines() {
            for word in line.split(|c: char| !c.is_ascii_hexdigit()) {
                if word.len() == 16 {
                    wanted.insert(word.to_ascii_lowercase());
                }
            }
        }
        anyhow::ensure!(!wanted.is_empty(), "no 16-digit shader hash in {}", file.display());
    }
    // `VICTORY_WARM_SECONDS=<n>`: stop taking new shaders after n seconds
    // (what is done stays in the cache; the next warm-up goes on from there).
    let limit = std::env::var("VICTORY_WARM_SECONDS").ok().and_then(|v| v.trim().parse::<u64>().ok());
    let cfg = victory_core::Config::load()?;
    victory_hle::modules::d3d12x::use_shader_cache(cfg.cache_subdir("shaders")?);
    let paths = crate::default_paths(paths)?;
    let threads = jobs.unwrap_or(4).max(1);
    let began = std::time::Instant::now();
    let done = Mutex::new(HashSet::new());
    let counts = [const { AtomicUsize::new(0) }; 5];
    for root in &paths {
        for entry in walkdir::WalkDir::new(root).into_iter().filter_map(Result::ok) {
            if !entry.file_type().is_file() || entry.metadata().map_or(0, |m| m.len()) > MAX_FILE {
                continue;
            }
            let Ok(data) = std::fs::read(entry.path()) else {
                continue;
            };
            let jobs = sources(&data);
            let next = AtomicUsize::new(0);
            std::thread::scope(|s| {
                for _ in 0..threads.min(jobs.len()) {
                    host_wave::spawn(s, wave, || {
                        while let Some(job) = jobs.get(next.fetch_add(1, Ordering::Relaxed)) {
                            let Some(bytes) = job.bytes() else {
                                continue;
                            };
                            let hash = blake3::hash(&bytes).to_hex()[..16].to_owned();
                            if limit.is_some_and(|n| began.elapsed().as_secs() >= n) {
                                counts[3].fetch_add(1, Ordering::Relaxed);
                                break;
                            }
                            if (!all && !wanted.contains(&hash)) || !lock(&done).insert(hash) {
                                continue;
                            }
                            let n = counts[4].fetch_add(1, Ordering::Relaxed) + 1;
                            if n.is_multiple_of(1000) {
                                println!("warm-up      {n} shaders so far, {:.0} s", began.elapsed().as_secs_f64());
                            }
                            let which = match victory_hle::modules::d3d12x::warm_shader(&bytes) {
                                Ok(true) => 0,
                                Ok(false) => 1,
                                Err(_) => 2,
                            };
                            counts[which].fetch_add(1, Ordering::Relaxed);
                        }
                    });
                }
            });
        }
    }
    let [translated, there, failed] = [0, 1, 2].map(|i| counts[i].load(Ordering::Relaxed));
    let asked = if all { "all".to_owned() } else { wanted.len().to_string() };
    let stopped = if counts[3].load(Ordering::Relaxed) > 0 { " (stopped at the time limit)" } else { "" };
    println!(
        "warm-up      {asked} hashes asked, {} found: {translated} translated now, {there} in the cache already, {failed} do not translate; {:.1} s on {threads} threads{stopped}",
        translated + there + failed,
        began.elapsed().as_secs_f64()
    );
    Ok(())
}

pub fn run(
    paths: Vec<PathBuf>,
    validate: bool,
    pipelines: bool,
    list: bool,
    jobs: Option<usize>,
    wave: Option<u32>,
) -> Result<()> {
    // Dumped shaders and validator reports are game-derived (rule #1).
    if let Some(dir) = std::env::var_os("VICTORY_EMIT_DUMP") {
        victory_core::ensure_outside_repo(Path::new(&dir))?;
    }
    let paths = crate::default_paths(paths)?;
    let host_caps = if pipelines {
        let host = Host::new(false).map_err(anyhow::Error::msg)?;
        if let Some(name) = host.adapter_name() {
            println!("pipeline adapter: {name}");
        }
        host.translation_caps()
    } else {
        None
    };
    // Without VICTORY_HOST_WAVE / VICTORY_HOST_MAX_WAVE (`wave`, set per
    // worker), translate as for the adapter the pipelines are made on.
    if wave.is_none()
        && let Some((lanes, _)) = host_caps
    {
        victory_shader::emit::set_host_max_wave(lanes);
    }
    // `VICTORY_HOST_RDNA=1`: as for an AMD RDNA card (`VICTORY_RDNA_FLOAT`).
    victory_shader::emit::set_host_rdna(std::env::var_os("VICTORY_HOST_RDNA").map_or_else(
        || host_caps.is_some_and(|(_, rdna)| rdna),
        |v| v == "1",
    ));
    let dll = validate.then(Validator::default_path).flatten();
    if validate {
        Validator::load(dll.as_deref()).map_err(anyhow::Error::msg)?;
        println!(
            "validator    {}",
            dll.as_ref()
                .map_or("dxil.dll".into(), |p| p.display().to_string())
        );
    }
    let pairs: Vec<(String, String)> = std::env::var("VICTORY_PAIRS")
        .unwrap_or_default()
        .split(',')
        .filter_map(|p| p.split_once('+'))
        .map(|(v, p)| (v.trim().to_owned(), p.trim().to_owned()))
        .collect();
    let kept = std::sync::Arc::new(Mutex::new(Vec::new()));
    let threads =
        jobs.unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get()));
    let seen = Mutex::new(HashSet::new());
    let total = Mutex::new(Tally::default());
    for root in &paths {
        for entry in walkdir::WalkDir::new(root)
            .into_iter()
            .filter_map(Result::ok)
        {
            if !entry.file_type().is_file() || entry.metadata().map_or(0, |m| m.len()) > MAX_FILE {
                continue;
            }
            let Ok(data) = std::fs::read(entry.path()) else {
                continue;
            };
            let jobs = sources(&data);
            let next = AtomicUsize::new(0);
            std::thread::scope(|s| {
                for _ in 0..threads.min(jobs.len()) {
                    host_wave::spawn(s, wave, || {
                        let mut t = Tally {
                            list,
                            ..Tally::default()
                        };
                        let mut checks = Checks {
                            keep: (!pairs.is_empty()).then(|| kept.clone()),
                            validation_requested: validate,
                            validator: validate
                                .then(|| Validator::load(dll.as_deref()).ok())
                                .flatten(),
                            host: pipelines
                                .then(|| Host::new(false).ok().map(|h| (h, host::stub_ps())))
                                .flatten(),
                        };
                        while let Some(job) = jobs.get(next.fetch_add(1, Ordering::Relaxed)) {
                            translate(job, entry.path(), &seen, &mut t, &mut checks);
                        }
                        lock(&total).merge(t);
                    });
                }
            });
        }
    }
    let t = total.into_inner().unwrap_or_else(|e| e.into_inner());
    println!(
        "containers   {} ({} duplicates skipped)",
        t.containers, t.duplicates
    );
    println!(
        "programs     {} translated of {} ({} IR instructions)",
        t.translated, t.programs, t.instructions
    );
    println!(
        "structured   {} of {} ({} vector instructions, {} guarded, {} on opaque masks; \
         branches {} per lane, {} uniform)",
        t.structured,
        t.translated,
        t.vector,
        t.guarded,
        t.opaque,
        t.lane_branches,
        t.uniform_branches
    );
    println!(
        "resources    {} of {} tracked (descriptors {} loaded, {} built, {} constant; \
         {} scalar accesses; {} with a computed term)",
        t.tracked, t.translated, t.loaded, t.built, t.constant, t.scalar, t.dynamic
    );
    println!(
        "  used-resource entries {}: {} not traced {:?}; {} traced parameters not listed",
        t.used,
        t.untraced_used.values().sum::<usize>(),
        t.untraced_used,
        t.unlisted
    );
    println!(
        "I/O          {} of {} vertex and pixel programs raised ({} interpolations, \
         {} vertex values; exports {:?}, {} still f16 pairs)",
        t.io_raised, t.io_programs, t.interpolated, t.vertex_values, t.exports, t.compressed
    );
    println!(
        "DXIL         {} emitted of {} vertex and pixel programs; {} validated, {} pipelines",
        t.emitted, t.emit_programs, t.validated, t.pipelines
    );
    println!(
        "             {} emitted of {} compute programs; {} validated, {} pipelines",
        t.cs_emitted, t.cs_programs, t.cs_validated, t.cs_pipelines
    );
    println!(
        "             {} emitted of {} mesh programs; {} validated, {} pipelines",
        t.ms_emitted, t.ms_programs, t.ms_validated, t.ms_pipelines
    );
    if !pairs.is_empty() {
        let host = Host::new(false).map_err(anyhow::Error::msg)?;
        let kept = lock(&kept);
        let find = |name: &str| kept.iter().find(|(l, _)| l.contains(name));
        for (v, p) in &pairs {
            let made = match (find(v), find(p)) {
                (Some((_, vs)), Some((_, ps))) => if vs.stage == HwStage::Ms {
                    host.mesh_pipeline(vs, ps)
                } else {
                    host.pipeline(vs, ps)
                }
                .map(|_| "pipeline created".to_owned())
                .unwrap_or_else(|e| e),
                _ => "not emitted".to_owned(),
            };
            println!("pair {v} + {p}: {made}");
        }
    }
    let mut lines = t.lines;
    lines.sort();
    for l in lines {
        println!("{l}");
    }
    let mut failed: Vec<_> = t.failed.into_iter().collect();
    failed.sort_by_key(|(_, (n, _))| std::cmp::Reverse(*n));
    for (reason, (n, example)) in failed {
        println!("  {n:>7}  {reason}  (e.g. {example})");
    }
    Ok(())
}

/// Every shader found into the cache runs translate from.
pub fn precompile(paths: Vec<PathBuf>, jobs: Option<usize>, list: Option<PathBuf>) -> Result<()> {
    let cfg = victory_core::Config::load()?;
    victory_hle::modules::d3d12x::use_shader_cache(cfg.cache_subdir("shaders")?);
    let paths = crate::default_paths(paths)?;
    let threads =
        jobs.unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get()));
    let started = std::time::Instant::now();
    let seen = Mutex::new(HashSet::new());
    let done = AtomicUsize::new(0);
    let failed = Mutex::new(BTreeMap::<String, (usize, String)>::new());
    let lines = Mutex::new(Vec::<String>::new());
    let mut scan = FileScan::new(MAX_FILE, false);
    for root in &paths {
        for (path, data) in scan.files(root) {
            let jobs = sources(&data);
            let next = AtomicUsize::new(0);
            std::thread::scope(|s| {
                for _ in 0..threads.min(jobs.len()) {
                    s.spawn(|| {
                        while let Some(job) = jobs.get(next.fetch_add(1, Ordering::Relaxed)) {
                            let Some(bytes) = job.bytes() else { continue };
                            let hash = blake3::hash(&bytes);
                            if !lock(&seen).insert(hash) {
                                continue;
                            }
                            match victory_hle::modules::d3d12x::precompile(&bytes) {
                                Ok(_) => {
                                    let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                                    if n.is_multiple_of(1000) {
                                        println!("{n} shaders in the cache, {:.0} s", started.elapsed().as_secs_f64());
                                    }
                                }
                                Err(why) => {
                                    let hex = hash.to_hex()[..16].to_owned();
                                    if list.is_some() {
                                        let line = format!("{hex}	{}{}	{why}", path.display(), job.label());
                                        lock(&lines).push(line);
                                    }
                                    let mut failed = lock(&failed);
                                    failed.entry(why).or_insert((0, hex)).0 += 1;
                                }
                            }
                        }
                    });
                }
            });
        }
    }
    let failed = failed.into_inner().unwrap_or_else(|e| e.into_inner());
    if let Some(list) = &list {
        let mut lines = lines.into_inner().unwrap_or_else(|e| e.into_inner());
        lines.sort();
        std::fs::write(list, lines.join("
") + "
")?;
    }
    println!(
        "{} shaders in the cache, {} do not translate, {:.0} s",
        done.load(Ordering::Relaxed),
        failed.values().map(|f| f.0).sum::<usize>(),
        started.elapsed().as_secs_f64()
    );
    let (stale, refreshed, changed) = victory_hle::modules::d3d12x::stale_shaders();
    if stale > 0 {
        println!(
            "{refreshed} of {stale} translations an earlier build of the translator had cached translated again, {changed} came out different{}",
            if changed > 0 {
                " (a change of the translator's output: victory_shader::TRANSLATOR_VERSION should have been bumped)"
            } else {
                ""
            }
        );
    }
    for (why, (n, hash)) in failed {
        println!("  {n:>5}  {why}  (e.g. {hash})");
    }
    println!("{scan}");
    Ok(())
}

fn translate(
    source: &Source,
    path: &Path,
    seen: &Mutex<HashSet<blake3::Hash>>,
    t: &mut Tally,
    checks: &mut Checks,
) {
    let example = || format!("{}{}", path.display(), source.label());
    let Some(bytes) = source.bytes() else {
        return t.fail("corrupt record".into(), example);
    };
    // The hash a run's log names the shader by (`hash=` in "first bind",
    // "shader does not translate", "host pipeline refused").
    let hash = blake3::hash(&bytes).to_hex()[..16].to_owned();
    // Diagnostics: `VICTORY_TRANSLATE_ONLY=<hash prefix>,...` translates
    // those containers alone.
    static ONLY: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    let only = ONLY.get_or_init(|| {
        std::env::var("VICTORY_TRANSLATE_ONLY")
            .unwrap_or_default()
            .split(',')
            .map(|h| h.trim().to_lowercase())
            .filter(|h| !h.is_empty())
            .collect()
    });
    if !only.is_empty() && !only.iter().any(|o| hash.starts_with(o.as_str())) {
        return;
    }
    tracing::debug!(shader_hash = %hash, "reading offline shader container");
    let example = || format!("{}{} #{hash}", path.display(), source.label());
    t.containers += 1;
    let Ok(container) = Container::parse(&bytes) else {
        return t.fail("no DXIL part (SM5 or older)".into(), example);
    };
    let Some(dxil) = container.part(FourCC::DXIL).or(container.part(FourCC::ACCC)).map(|p| p.data) else {
        return t.fail("no DXIL part (SM5 or older)".into(), example);
    };
    // `~`: the hash of the code alone; containers that differ only around
    // it are translated once, the others listed as duplicates of it.
    let code = blake3::hash(dxil);
    let code8 = code.to_hex()[..8].to_owned();
    let example = || format!("{} ~{code8}", example());
    // `VICTORY_DUMP_MATCH=<dir>;<text>|<text>...`: the containers whose
    // line holds one of the texts (a name, `#hash`), as the title has them.
    if let Ok(spec) = std::env::var("VICTORY_DUMP_MATCH")
        && let Some((dir, texts)) = spec.split_once(';')
    {
        let line = example();
        if texts.split('|').any(|text| !text.is_empty() && line.contains(text)) {
            let name: String = source
                .label()
                .chars()
                .skip_while(|c| *c != ' ')
                .filter(|c| c.is_ascii_alphanumeric() || *c == '_')
                .take(70)
                .collect();
            let _ = std::fs::create_dir_all(dir);
            let _ = std::fs::write(std::path::Path::new(dir).join(format!("{name}.{hash}.cdxil")), &bytes[..]);
        }
    }
    if !lock(seen).insert(code) {
        t.duplicates += 1;
        if t.list {
            t.lines.push(format!("{}  duplicate", example()));
        }
        return;
    }
    let programs = match native_programs(dxil) {
        Ok(p) => p,
        Err(e) => return t.fail(e.to_string(), example),
    };
    // `VICTORY_FAIL_DUMP=<dir>`: the first 3 containers of each failure reason, as the title
    // has them (`.cdxil`), to work on a translator gap offline
    let dump = std::env::var_os("VICTORY_FAIL_DUMP");
    let before: Option<BTreeMap<String, usize>> = dump.as_ref().map(|_| t.failed.iter().map(|(k, v)| (k.clone(), v.0)).collect());
    translate_programs(&container, programs, &hash, &example, t, checks);
    if let (Some(dir), Some(before)) = (dump, before) {
        for (reason, (n, _)) in &t.failed {
            let was = before.get(reason).copied().unwrap_or(0);
            if *n > was && was < 3 {
                let stem: String = reason.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).take(60).collect();
                let _ = std::fs::create_dir_all(&dir);
                let _ = std::fs::write(std::path::Path::new(&dir).join(format!("{stem}_{was}.cdxil")), &bytes[..]);
            }
        }
    }
}

fn translate_programs(
    container: &Container,
    programs: Vec<NativeProgram>,
    hash: &str,
    example: &dyn Fn() -> String,
    t: &mut Tally,
    checks: &mut Checks,
) {
    for p in programs {
        t.programs += 1;
        match victory_shader::spill::build(&p) {
            Ok(program) => {
                t.translated += 1;
                t.instructions += program.blocks.iter().map(|b| b.insts.len()).sum::<usize>();
                // Per-lane descriptors are traced on the per-lane form; its
                // failure is listed after the resources' and I/O's.
                let structured = structure::structure(&program);
                let lanes = structured.as_ref().ok();
                let res = match resource::track(&program, &p.layout, p.stage, lanes) {
                    Ok(r) => {
                        tally_resources(&p, &r, t);
                        Some(r)
                    }
                    Err(ShaderError::Unsupported { offset, what }) => {
                        t.fail(format!("resources: {what}"), || {
                            format!("{} {:?} at {offset:#x}", example(), p.stage)
                        });
                        None
                    }
                    Err(e) => {
                        t.fail(e.to_string(), || format!("{} {:?}", example(), p.stage));
                        None
                    }
                };
                let graphics = matches!(p.stage, HwStage::Vs | HwStage::Ps);
                let compute = p.stage == HwStage::Cs;
                let mut raised = None;
                if graphics || compute || p.stage == HwStage::Ms {
                    t.io_programs += usize::from(graphics);
                    match io::raise(&program, &p) {
                        Ok(r) => {
                            if graphics {
                                tally_io(&r, t);
                            }
                            raised = Some(r);
                        }
                        Err(ShaderError::Unsupported { offset, what }) => t
                            .fail(format!("I/O: {what}"), || {
                                format!("{} {:?} at {offset:#x}", example(), p.stage)
                            }),
                        Err(e) => t.fail(e.to_string(), || format!("{} {:?}", example(), p.stage)),
                    }
                }
                let structured = match structured {
                    Ok(s) => {
                        tally_structured(&program, &s, t);
                        Some(s)
                    }
                    Err(e) => {
                        let reason = match e {
                            ShaderError::Unsupported { what, .. } => format!("structure: {what}"),
                            ShaderError::Isa { reason, .. } => format!("structure: {reason}"),
                            e => e.to_string(),
                        };
                        t.fail(reason, || format!("{} {:?}", example(), p.stage));
                        None
                    }
                };
                if let (Some(r), Some(i), Some(s)) = (res, raised, structured) {
                    let label = || format!("{} {:?}", example(), p.stage);
                    *counter(t, p.stage, 0) += 1;
                    tracing::debug!(shader_hash = %hash, stage = ?p.stage, "emitting offline DXIL");
                    match emit::emit_with_retries(container, &p, &program, &s, &i, &r) {
                        Ok(shader) => {
                            census_line(&label(), &p, &program, &shader);
                            *counter(t, p.stage, 1) += 1;
                            check(shader, t, checks, hash, label);
                        }
                        Err(ShaderError::Unsupported { offset, what }) => t
                            .fail(format!("emit: {what}"), || {
                                format!("{} at {offset:#x}", label())
                            }),
                        Err(e) => t.fail(format!("emit: {e}"), label),
                    }
                }
            }
            Err(e) => {
                let reason = match e {
                    ShaderError::Unsupported { what, .. } => format!("unsupported: {what}"),
                    e => e.to_string(),
                };
                t.fail(reason, || format!("{} {:?}", example(), p.stage));
            }
        }
    }
}

/// `VICTORY_SHADER_CENSUS=<file>`: one line per emitted program, appended:
/// its label, the console program's instructions by encoding class (a
/// static count of the whole program, not of one thread's path), and the
/// host module's instructions by kind (`emit::census::KINDS`).
fn census_line(label: &str, native: &NativeProgram, p: &ir::Program, shader: &emit::HostShader) {
    use std::io::Write;
    use victory_shader::gfx10::Fields;
    static OUT: std::sync::OnceLock<Option<Mutex<std::fs::File>>> = std::sync::OnceLock::new();
    let Some(out) = OUT.get_or_init(|| {
        let path = std::env::var_os("VICTORY_SHADER_CENSUS")?;
        let file = std::fs::OpenOptions::new().create(true).append(true).open(path).ok()?;
        Some(Mutex::new(file))
    }) else {
        return;
    };
    let Some(census) = emit::census::take(&shader.dxil) else {
        return;
    };
    // total, vector, scalar, descriptor loads, buffer and memory, image, LDS, control, other
    let mut console = [0u32; 9];
    for inst in p.blocks.iter().flat_map(|b| &b.insts) {
        console[0] += 1;
        let k = match &inst.fields {
            Fields::Valu(_) => 1,
            Fields::Sop { .. } => 2,
            Fields::Smem { .. } => 3,
            Fields::Buffer { .. } | Fields::Flat { .. } => 4,
            Fields::Image { .. } => 5,
            Fields::Ds { .. } => 6,
            Fields::Sopp { .. } => 7,
            _ => 8,
        };
        console[k] += 1;
    }
    let mut line = format!("{label}\twave64={}", u8::from(native.wave64));
    for (name, n) in ["c_total", "c_valu", "c_sop", "c_smem", "c_buffer", "c_image", "c_ds", "c_sopp", "c_other"].iter().zip(console) {
        line.push_str(&format!("\t{name}={n}"));
    }
    for (name, n) in emit::census::KINDS.iter().zip(census.counts) {
        line.push_str(&format!("\t{name}={n}"));
    }
    line.push('\n');
    let _ = lock(out).write_all(line.as_bytes());
}

/// Validates an emitted shader before any requested pipeline creation.
fn check(
    mut shader: emit::HostShader,
    t: &mut Tally,
    checks: &mut Checks,
    hash: &str,
    label: impl Fn() -> String,
) {
    // With `--list`: a hash of the module and of what the host binds for it,
    // to show which shaders a change to the translator touches.
    if t.list {
        let hash = blake3::hash(format!("{shader:?}").as_bytes()).to_hex();
        t.lines.push(format!("{}  emitted {}", label(), &hash[..16]));
    }
    // `VICTORY_EMIT_DUMP` with `--list`: every emitted shader as a file.
    if t.list
        && let Ok(dir) = std::env::var("VICTORY_EMIT_DUMP")
    {
        let name: String = label()
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect();
        let _ = std::fs::write(
            format!("{dir}/{}.dxil", &name[name.len().saturating_sub(100)..]),
            &shader.dxil,
        );
    }
    if checks.validation_requested {
        let Some(validator) = checks.validator.as_mut() else {
            t.fail("validator: unavailable in the translation worker".to_owned(), &label);
            return;
        };
        tracing::debug!(shader_hash = %hash, stage = ?shader.stage, "validating offline DXIL");
        if let Err(text) = validator.validate(&mut shader.dxil) {
            let first = text
                .lines()
                .find(|l| l.contains("error"))
                .unwrap_or(text.lines().next().unwrap_or(""))
                .to_owned();
            if let Ok(dir) = std::env::var("VICTORY_EMIT_DUMP") {
                let n = t
                    .failed
                    .get(&format!("validator: {first}"))
                    .map_or(0, |f| f.0);
                if n < 3 {
                    let name = format!(
                        "{dir}/{}_{n}",
                        first
                            .chars()
                            .filter(char::is_ascii_alphanumeric)
                            .take(40)
                            .collect::<String>()
                    );
                    let _ = std::fs::write(format!("{name}.dxil"), &shader.dxil);
                    let _ = std::fs::write(format!("{name}.txt"), format!("{}\n{text}", label()));
                }
            }
            t.fail(format!("validator: {first}"), &label);
            return;
        }
        *counter(t, shader.stage, 2) += 1;
        if t.list {
            t.lines.push(format!("{}  validated", label()));
        }
    }
    if let Some(keep) = &checks.keep {
        lock(keep).push((label(), shader.clone()));
    }
    if let Some((h, stub_ps)) = &checks.host {
        tracing::debug!(shader_hash = %hash, stage = ?shader.stage, "creating offline D3D12 pipeline");
        let made = match shader.stage {
            HwStage::Vs => h.pipeline(&shader, stub_ps).map(drop),
            HwStage::Cs => h.compute_pipeline(&shader).map(drop),
            HwStage::Ms => h.mesh_pipeline(&shader, stub_ps).map(drop),
            _ => host::stub_vs(&shader).and_then(|vs| h.pipeline(&vs, &shader).map(drop)),
        };
        match made {
            Ok(()) => {
                *counter(t, shader.stage, 3) += 1;
                if t.list {
                    t.lines.push(format!("{}  pipeline created", label()));
                }
            }
            Err(e) => t.fail(format!("pipeline: {e}"), &label),
        }
    } else if t.list && !checks.validation_requested {
        t.lines.push(format!("{}  emitted", label()));
    }
}

/// Emission counts of `stage`'s kind: 0 programs, 1 emitted, 2 validated,
/// 3 pipelines.
fn counter(t: &mut Tally, stage: HwStage, k: usize) -> &mut usize {
    let [a, b, c, d] = match stage {
        HwStage::Cs => [
            &mut t.cs_programs,
            &mut t.cs_emitted,
            &mut t.cs_validated,
            &mut t.cs_pipelines,
        ],
        HwStage::Ms => [
            &mut t.ms_programs,
            &mut t.ms_emitted,
            &mut t.ms_validated,
            &mut t.ms_pipelines,
        ],
        _ => [
            &mut t.emit_programs,
            &mut t.emitted,
            &mut t.validated,
            &mut t.pipelines,
        ],
    };
    [a, b, c, d].into_iter().nth(k).expect("0 to 3")
}

fn tally_io(r: &io::Io, t: &mut Tally) {
    t.io_raised += 1;
    for (_, i) in &r.interps {
        match i {
            io::Interp::Interpolated { .. } => t.interpolated += 1,
            io::Interp::Vertex { .. } => t.vertex_values += 1,
            io::Interp::Weighted { .. } => t.interpolated += 1,
        }
    }
    for e in &r.exports {
        let kind = match e.target {
            io::Target::Color(_) => "color",
            io::Target::Depth => "depth",
            io::Target::Null => "null",
            io::Target::Position(_) => "position",
            io::Target::Param { .. } => "param",
            io::Target::Unread => "unread param",
            io::Target::Primitive => "primitive",
            io::Target::PrimitiveParam { .. } => "primitive param",
        };
        *t.exports.entry(kind).or_default() += 1;
        t.compressed += usize::from(e.compressed);
    }
}

fn tally_resources(p: &NativeProgram, r: &resource::Resources, t: &mut Tally) {
    t.tracked += 1;
    t.scalar += r.scalar.len();
    t.dynamic += r.scalar.iter().filter(|s| s.1.dynamic.is_some()).count();
    let address = |load| r.scalar.iter().find(|s| s.0 == load).map(|s| s.1);
    let mut traced = HashSet::new();
    for leaf in r.descriptors.iter().flat_map(|d| d.source.leaves()) {
        let a = match *leaf {
            resource::Source::Loaded { load, .. } | resource::Source::Fetched { load, .. } => {
                t.loaded += 1;
                address(load)
            }
            resource::Source::Built { address, .. } => {
                t.built += 1;
                t.dynamic += usize::from(address.dynamic.is_some());
                Some(address)
            }
            resource::Source::Constant(_) => {
                t.constant += 1;
                None
            }
            resource::Source::Phi { .. }
            | resource::Source::Select { .. }
            | resource::Source::Lanes { .. }
            | resource::Source::LanePhi { .. } => {
                unreachable!("leaves are no phis, selects or per-lane choices")
            }
        };
        // Heaps hold dynamic resources, which no used entry lists; code
        // tables are no root parameter.
        if let Some(a) = a
            && a.param != resource::CODE
            && !matches!(
                p.layout[a.param].root,
                RootType::ResourceHeap | RootType::SamplerHeap
            )
        {
            traced.insert(p.layout[a.param].user_data);
        }
    }
    // Root constants are listed as a CBV, read straight from the SGPRs.
    traced.extend(
        r.params
            .iter()
            .map(|&(_, param, _)| &p.layout[param])
            .filter(|l| l.root == RootType::Constants)
            .map(|l| l.user_data),
    );
    let listed: HashSet<u8> = p.used.iter().map(|u| u.user_data).collect();
    t.used += p.used.len();
    // Immediate samplers are literals in the code: the distinct sampler
    // values (decoded) are matched against the listed entries, which do not
    // say which value is whose.
    let mut values: Vec<emit::StaticSampler> = Vec::new();
    for d in r
        .descriptors
        .iter()
        .filter(|d| d.role == resource::Role::Sampler)
    {
        for leaf in d.source.leaves() {
            if let resource::Source::Constant(words) = leaf
                && let Ok(w) = <[u32; 4]>::try_from(&words[..words.len().min(4)])
            {
                let s = emit::StaticSampler::from_words(w);
                if !values.contains(&s) {
                    values.push(s);
                }
            }
        }
    }
    // Engines often declare equal samplers at several registers, so equal
    // values may stand for several entries, but never more values than
    // entries.
    let immediate = p
        .used
        .iter()
        .filter(|u| u.root == RootType::ImmediateSampler)
        .count();
    let samplers_match = !values.is_empty() && values.len() <= immediate;
    let matched = |u: &&victory_shader::UsedResource| {
        if u.root == RootType::ImmediateSampler {
            samplers_match
        } else {
            traced.contains(&u.user_data)
        }
    };
    for u in p.used.iter().filter(|u| !matched(u)) {
        *t.untraced_used
            .entry(format!("{:?} via {:?}", u.kind, u.root))
            .or_default() += 1;
    }
    t.unlisted += traced.difference(&listed).count();
}

fn tally_structured(program: &ir::Program, s: &structure::Structured, t: &mut Tally) {
    t.structured += 1;
    for (block, guards) in program.blocks.iter().zip(&s.guards) {
        t.vector += block
            .insts
            .iter()
            .filter(|i| i.read(ir::Reg::S(ir::EXEC)).is_some())
            .count();
        for &g in guards.iter().flatten() {
            t.guarded += 1;
            let support = s.bools.support(g);
            t.opaque += usize::from(support.iter().any(|a| matches!(a, Atom::Opaque(_))));
        }
    }
    for exit in &s.exits {
        if let LaneExit::Branch { when, .. } = *exit {
            if s.bools
                .support(when)
                .iter()
                .all(|a| matches!(a, Atom::Uniform(_)))
            {
                t.uniform_branches += 1;
            } else {
                t.lane_branches += 1;
            }
        }
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_missing_requested_validator_keeps_shaders_out_of_later_pipeline_checks() {
        let kept = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut checks = super::Checks {
            keep: Some(kept.clone()),
            validation_requested: true,
            validator: None,
            host: None,
        };
        let mut tally = super::Tally::default();
        let shader = super::host::stub_ps();
        super::check(shader.clone(), &mut tally, &mut checks, "synthetic", || "synthetic".into());
        assert!(super::lock(&kept).is_empty());
        assert!(tally.failed.contains_key("validator: unavailable in the translation worker"));

        checks.validation_requested = false;
        super::check(shader, &mut tally, &mut checks, "synthetic", || "synthetic".into());
        assert_eq!(super::lock(&kept).len(), 1);
    }

    /// Shader dumps are game-derived, so a dump directory in the repo is refused.
    #[test]
    fn emit_dump_inside_repo_is_refused() {
        // SAFETY: no other test in this binary touches the environment.
        unsafe { std::env::set_var("VICTORY_EMIT_DUMP", env!("CARGO_MANIFEST_DIR")) };
        let missing = std::env::temp_dir().join("victory-translate-no-such-dir");
        let err = super::run(vec![missing], false, false, false, Some(1), None).unwrap_err();
        // SAFETY: as above; unset so a later test's `run` is not refused.
        unsafe { std::env::remove_var("VICTORY_EMIT_DUMP") };
        assert!(
            matches!(
                err.downcast_ref(),
                Some(victory_core::PolicyError::InsideRepo { .. })
            ),
            "{err}"
        );
    }
}
