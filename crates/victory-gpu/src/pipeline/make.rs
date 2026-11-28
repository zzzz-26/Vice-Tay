//! Host pipelines are made by workers, not by the title thread that first
//! records a draw with them. Translating a shader and the host driver's
//! compile take from milliseconds to seconds each (structurisation of the
//! biggest compute programs ~0.4 s, some driver compiles 3.4 s), and
//! the title's first world frame wants over 1,200 new pipelines: made
//! on its recording threads, that frame took more than 30 s and the
//! title's GPU watchdog ended the run (decision 0007, "Pipelines made off
//! the title's threads").
//!
//! A first use queues the pipeline and waits for it within a budget per
//! frame, so the cheap ones (menus, one-off passes such as a colour-grading
//! bake) are there at once as before; past the budget a draw or dispatch
//! whose pipeline is not ready is dropped, as an untranslated one is.
//!
//! The default budget is half a second per title frame. A pass the title
//! runs only once and whose pipeline takes longer is lost. Passes over the
//! screen (no vertex input, no depth) have workers and a longer budget of
//! their own ([`screen_budget`]): the title fills its lighting tables with
//! such draws once, at the world's first frame, and behind a world's worth of
//! pipelines they were dropped; that run had no lamps and no diffuse light to
//! its end (run textures-d2, the first after a new translation; Belobug
//! 18e57db8 found the same).

use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use super::Shader;
use super::host::{GraphicsState, HostPipeline};

/// What a host pipeline is made from, and the host pipeline once made.
pub(crate) struct Pending {
    shaders: Vec<Arc<Shader>>,
    pub(crate) graphics: Option<Box<GraphicsState>>,
    queue_state: AtomicU8,
    /// Taken by a worker from [`Queue::slow`].
    was_slow: AtomicBool,
    made: OnceLock<Option<HostPipeline>>,
    /// What identifies it across runs (shader containers and state), and
    /// whether a draw or dispatch has requested it yet.
    key: String,
    used: AtomicBool,
    /// A pass over the screen ([`GraphicsState::screen_pass`]) or a confirmed
    /// opening-scene / bake producer that must never sit behind cold RT shaders.
    screen: bool,
    /// Confirmed first-scene / bake producer prepared early even on a cold cache.
    scene_input: bool,
    /// The title dispatches this head blend-shape map only once.
    head_map: bool,
    wait_for_head_map: bool,
    /// Why this pipeline is intentionally skipped (e.g. RT resolve without BVH).
    left_out: Option<&'static str>,
}

fn named(shaders: &[Arc<Shader>], name: &[u8]) -> bool {
    shaders
        .iter()
        .any(|s| s.stage == "compute" && s.container.windows(name.len()).any(|w| w == name))
}

fn left_out(shaders: &[Arc<Shader>], skip_hashes: &[String]) -> Option<&'static str> {
    static SKIP_RT_RESOLVE: OnceLock<bool> = OnceLock::new();
    let skip_rt = *SKIP_RT_RESOLVE.get_or_init(|| {
        let builder = std::env::var_os("VICTORY_BVH_BUILDER").is_some_and(|v| v == "1");
        if builder {
            return false;
        }
        // Without BVH, ~1,000 CS_RaytracingResolve shaders shade no hits while
        // clogging the cold compiler queue; skip unless VICTORY_RT_RESOLVE=1.
        std::env::var_os("VICTORY_RT_RESOLVE").is_none_or(|v| v == "0")
    });
    if skip_rt && named(shaders, b"CS_RaytracingResolve") {
        return Some("shades ray-tracing hits without BVH, left out");
    }
    static SKIP_ACTIVATE_RAYS: OnceLock<bool> = OnceLock::new();
    let skip_activate = *SKIP_ACTIVATE_RAYS.get_or_init(|| {
        let builder = std::env::var_os("VICTORY_BVH_BUILDER").is_some_and(|v| v == "1");
        if builder {
            return std::env::var_os("VICTORY_ACTIVATE_RAYS").is_some_and(|v| v == "0");
        }
        std::env::var_os("VICTORY_ACTIVATE_RAYS").is_none_or(|v| v == "0")
    });
    if skip_activate && named(shaders, b"CS_ScreenProbes_ActivateRays") {
        return Some("ray activation shader without BVH, left out");
    }
    if super::skipped_in_list(shaders, skip_hashes).is_some() {
        return Some("skipped by VICTORY_SKIP_PIPELINES");
    }
    None
}

/// Confirmed first-scene producers. Prepare these at creation even on a cold
/// launch so cutscene lighting, tiling, and bakes never miss their first frame.
fn scene_input(shaders: &[Arc<Shader>]) -> bool {
    shaders.iter().any(|s| {
        matches!(
            s.hash.as_str(),
            "970c4fbb3b017c0e"
                | "068c90badc86f1fd"
                | "e4a7e088c7780ce8"
                | "ebe4e53f031f5249"
                | "2cc4c197ee29e291"
                | "369cbe906ea72764"
        ) || crate::draw::BAKE_PIXEL_SHADERS.contains(&s.hash.as_str())
    })
}

fn head_map_wait() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_HEAD_MAP_WAIT").is_none_or(|value| value != "0"))
}

// SAFETY: D3D12 objects are free-threaded; the graphics state (whose input
// elements point into its own names) is only read after creation.
unsafe impl Send for Pending {}
// SAFETY: as above.
unsafe impl Sync for Pending {}

impl Pending {
    pub(crate) fn new(shaders: Vec<Arc<Shader>>, graphics: Option<Box<GraphicsState>>) -> Pending {
        let protect_head_map = graphics.is_none() && head_map_wait();
        Self::new_with_policy(shaders, graphics, super::skip_hashes(), protect_head_map)
    }

    fn new_with_policy(
        shaders: Vec<Arc<Shader>>,
        graphics: Option<Box<GraphicsState>>,
        skip_hashes: &[String],
        protect_head_map: bool,
    ) -> Pending {
        let mut key = blake3::Hasher::new();
        for s in &shaders {
            key.update(s.stage.as_bytes());
            key.update(s.hash.as_bytes());
        }
        key.update(graphics.as_ref().map(|g| g.describe()).unwrap_or_default().as_bytes());
        // Losing this producer leaves eyes and teeth without the surrounding
        // skin. See avatarmanz/VIctory-7900XTX #166.
        let head_map = graphics.is_none()
            && matches!(&shaders[..], [shader] if shader.stage == "compute" && shader.hash == "61e54e90adccaf6a");
        let wait_for_head_map = head_map && protect_head_map;
        let scene_input = wait_for_head_map || scene_input(&shaders);
        let screen = (graphics.as_ref().is_some_and(|g| g.screen_pass())
            && shaders.iter().all(|s| s.stage != "mesh"))
            || scene_input;
        let left_out = left_out(&shaders, skip_hashes);
        Pending {
            screen,
            scene_input,
            head_map,
            wait_for_head_map,
            left_out,
            shaders,
            graphics,
            queue_state: AtomicU8::new(NOT_QUEUED),
            was_slow: AtomicBool::new(false),
            made: OnceLock::new(),
            key: key.finalize().to_hex()[..32].to_owned(),
            used: AtomicBool::new(false),
        }
    }

    pub(super) fn key(&self) -> &str {
        &self.key
    }

    fn make(&self) {
        self.make_with(|| {
            let _building = crate::timeline::span("pipeline: build a host pipeline");
            let began = Instant::now();
            let made = HostPipeline::new(&self.shaders, self.graphics.as_deref());
            super::library::note_time(&self.key, began.elapsed().as_millis() as u64);
            made
        });
    }

    fn make_with(&self, build: impl FnOnce() -> Option<HostPipeline>) {
        self.made.get_or_init(|| match catch_unwind(AssertUnwindSafe(build)) {
            Ok(pipeline) => pipeline,
            Err(_) => {
                tracing::warn!(key = self.key, shaders = ?self.shaders, "host pipeline build panicked");
                None
            }
        });
        let (lock, done) = &DONE;
        let _held = lock.lock().unwrap_or_else(|e| e.into_inner());
        done.notify_all();
    }
}

/// Reports why a host pipeline was left out, is pending, or failed to build.
/// Returns `None` when the host pipeline is ready.
#[cfg(feature = "diagnostics")]
pub(crate) fn missing(p: &Pending) -> Option<String> {
    match p.made.get() {
        Some(Some(_)) => None,
        None => Some(p.left_out.unwrap_or("being made").into()),
        Some(None) => {
            let failed: Vec<String> = p
                .shaders
                .iter()
                .filter(|s| matches!(s.translated.get(), Some(None)))
                .map(|s| format!("{} {}", s.stage, s.hash))
                .collect();
            Some(if failed.is_empty() {
                format!("refused {:?}", p.shaders)
            } else {
                failed.join(" + ")
            })
        }
    }
}

/// The hashes of the shaders `p` is made of.
pub(crate) fn hashes(p: &Pending) -> String {
    let hashes: Vec<&str> = p.shaders.iter().map(|s| s.hash.as_str()).collect();
    hashes.join("+")
}

/// Signalled whenever a worker has made a pipeline.
static DONE: (Mutex<()>, Condvar) = (Mutex::new(()), Condvar::new());

fn wait_for_made<T>(made: &OnceLock<T>, started: Instant, allowance: Duration) -> Option<&T> {
    if let Some(result) = made.get() {
        return Some(result);
    }
    if started.elapsed() >= allowance {
        return made.get();
    }
    let (lock, done) = &DONE;
    let mut held = lock.lock().unwrap_or_else(|e| e.into_inner());
    while made.get().is_none() {
        let left = allowance.saturating_sub(started.elapsed());
        if left.is_zero() {
            break;
        }
        held = done.wait_timeout(held, left).unwrap_or_else(|e| e.into_inner()).0;
    }
    made.get()
}

/// Waiting for the prequeued head map is normally a matter of milliseconds.
/// After a timeout, later dispatches only poll. A successful wait leaves
/// the budget available if the title creates another pipeline for the shader.
struct HeadMapWait {
    allowance: Duration,
    timed_out: AtomicBool,
}

impl HeadMapWait {
    const fn new(allowance: Duration) -> Self {
        Self { allowance, timed_out: AtomicBool::new(false) }
    }

    fn wait<'a>(&self, pipeline: &'a Pending) -> Option<&'a Option<HostPipeline>> {
        if self.timed_out.load(Ordering::Acquire) {
            return pipeline.made.get();
        }
        let made = wait_for_made(&pipeline.made, Instant::now(), self.allowance);
        if made.is_none() && !self.timed_out.swap(true, Ordering::AcqRel) {
            tracing::warn!(
                key = pipeline.key,
                frame = crate::frame::current(),
                budget_ms = self.allowance.as_millis() as u64,
                "head blend-shape map pipeline still pending at its dispatch deadline"
            );
        }
        made
    }
}

/// Waiting allowed per title frame, over all threads.
fn wait_budget() -> Duration {
    static BUDGET: OnceLock<Duration> = OnceLock::new();
    *BUDGET.get_or_init(|| {
        let millis = std::env::var("VICTORY_PIPELINE_WAIT_MS")
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
            .unwrap_or(500);
        tracing::info!(millis, "host pipeline wait budget per frame");
        Duration::from_millis(millis)
    })
}

/// As [`wait_budget`], for passes over the screen (`VICTORY_SCREEN_WAIT_MS`,
/// 10 s; 0: like the others). Below the title's 30 s watchdog.
fn screen_budget() -> Duration {
    static BUDGET: OnceLock<Duration> = OnceLock::new();
    *BUDGET.get_or_init(|| {
        let millis = std::env::var("VICTORY_SCREEN_WAIT_MS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(10_000);
        Duration::from_millis(millis)
    })
}

/// Whether passes over the screen go before the others.
fn screens_first() -> bool {
    !screen_budget().is_zero()
}

/// Pipelines queued and not made yet.
static WAITING: AtomicU64 = AtomicU64::new(0);

/// Pipelines queued and not made yet (for the frame line).
pub(crate) fn waiting() -> u64 {
    WAITING.load(Ordering::Relaxed)
}

/// Queue the host pipeline on first use and wait within the frame budget.
/// The one-shot head map has its own budget below the title's watchdog.
pub(crate) fn ready(p: &Arc<Pending>, dispatch: bool) -> Option<&HostPipeline> {
    static HEAD_MAP_WAIT: HeadMapWait = HeadMapWait::new(Duration::from_secs(2));
    ready_with(p, dispatch, &HEAD_MAP_WAIT, |p| wait(p, crate::frame::current()))
}

fn ready_with<'a>(
    p: &'a Arc<Pending>,
    dispatch: bool,
    head_wait: &HeadMapWait,
    wait_for_frame: impl FnOnce(&Pending),
) -> Option<&'a HostPipeline> {
    if p.left_out.is_some() {
        return None;
    }
    // A still-pending pipeline was requested by the title too. Prewarm it
    // on the next launch, even if this run never got to use it.
    if !p.used.load(Ordering::Relaxed) && !p.used.swap(true, Ordering::Relaxed) {
        super::library::note_used(&p.key);
    }
    let mut made = p.made.get();
    let already_ready = made.is_some();
    // The same shader can be dispatched for many characters or instances.
    // Record its first actual dispatch entry in both arms of the comparison.
    static HEAD_MAP_REPORTED: AtomicBool = AtomicBool::new(false);
    let first_head = dispatch && p.head_map && !HEAD_MAP_REPORTED.load(Ordering::Relaxed)
        && !HEAD_MAP_REPORTED.swap(true, Ordering::Relaxed);
    let began = first_head.then(|| (crate::frame::current(), Instant::now()));
    if !already_ready {
        start(p, REQUESTED);
        made = if dispatch && p.wait_for_head_map {
            // start has released the queue lock. The worker publishes both
            // failure and success through made and wakes all callers.
            head_wait.wait(p)
        } else {
            wait_for_frame(p);
            p.made.get()
        };
    }
    let host_pipeline = made.and_then(Option::as_ref);
    if let Some((frame, began)) = began {
        tracing::info!(
            frame,
            key = p.key,
            already_ready,
            host_pipeline = host_pipeline.is_some(),
            wait_enabled = p.wait_for_head_map,
            wait_ms = began.elapsed().as_secs_f64() * 1e3,
            "head blend-shape map pipeline at dispatch"
        );
    }
    host_pipeline
}

/// Makes `p` on this thread if it is not made yet (waiting for a worker
/// that is making it); whether it has a host pipeline.
pub(crate) fn now(p: &Arc<Pending>) -> bool {
    if p.left_out.is_some() {
        return false;
    }
    p.make();
    p.made.get().is_some_and(|made| made.is_some())
}

/// At the title's creation of a pipeline: starts it at once if an earlier
/// run drew with it (during loading, so the world's first frames find it),
/// or if it is a confirmed opening-scene / bake producer.
pub(crate) fn early(p: &Arc<Pending>) {
    if p.left_out.is_some() {
        return;
    }
    if !p.shaders.is_empty() && (p.scene_input || super::library::wanted_early(&p.key)) {
        super::library::EARLY.fetch_add(1, Ordering::Relaxed);
        start(p, if p.scene_input { REQUESTED } else { EARLY });
    }
}

/// Whether the frame's budget is one for all threads together: a thread
/// that waits takes what is left of it and gives back what it did not
/// use. Counted only when a wait had ended, the budget was one per thread:
/// each of the title's five recording threads began its first wait of a
/// frame with nothing counted yet and waited the whole half second, 2.5 s
/// of waiting in every frame in which pipelines were being made (runs
/// 34-49: "2504 ms waiting for pipelines" in each such frame).
/// On by default; `VICTORY_PIPELINE_WAIT_SHARED=0` counts as before.
fn shared_budget() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PIPELINE_WAIT_SHARED").is_none_or(|v| v != "0"))
}

/// Waits for `p` while frame `frame` has budget left.
fn wait(p: &Pending, frame: u64) {
    let screen = p.screen && screens_first();
    let budget = if screen { screen_budget() } else { wait_budget() };
    if budget.is_zero() {
        return;
    }
    // The frame the budget belongs to, and what was waited of it (or is
    // taken by a thread waiting now); of the passes over the screen too.
    static SPENT: [Mutex<(u64, Duration)>; 2] = [const { Mutex::new((u64::MAX, Duration::ZERO)) }; 2];
    let spent_of = &SPENT[usize::from(screen)];
    let shared = shared_budget();
    let left = {
        let mut spent = spent_of.lock().unwrap_or_else(|e| e.into_inner());
        if spent.0 != frame {
            *spent = (frame, Duration::ZERO);
        }
        let left = budget.saturating_sub(spent.1);
        if shared {
            spent.1 += left;
        }
        left
    };
    if left.is_zero() {
        return;
    }
    let waiting = crate::timeline::span("pipeline: first use waits for a host pipeline");
    let started = Instant::now();
    let _ = wait_for_made(&p.made, started, left);
    let waited = started.elapsed().min(left);
    waiting.end();
    if screen && waited > Duration::from_millis(500) {
        tracing::info!(
            ms = waited.as_millis() as u64,
            made = p.made.get().is_some(),
            shaders = ?p.shaders,
            "a pass over the screen was waited for (VICTORY_SCREEN_WAIT_MS=0: dropped past half a second a frame)"
        );
    }
    {
        let mut spent = spent_of.lock().unwrap_or_else(|e| e.into_inner());
        // A frame that began meanwhile has its own budget.
        if spent.0 == frame {
            if shared {
                spent.1 = spent.1.saturating_sub(left - waited);
            } else {
                spent.1 += waited;
            }
        }
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(
        crate::copystat::C::PipelineWaitUs,
        started.elapsed().as_micros() as u64,
    );
}

const NOT_QUEUED: u8 = 0;
const EARLY: u8 = 1;
const REQUESTED: u8 = 2;
const MAKING: u8 = 3;

/// How many workers may make pipelines at once that are known to take long
/// and that no draw has asked for: a quarter of them. On a fresh cache (a
/// change of the translator or the driver) the early list begins with
/// programs that take tens of seconds each: all 16 workers were in them
/// through the whole of the title's loading, 97 pipelines made in 20 s, and
/// the first minute and a half of the world ran at 0.5 to 1.6 s a frame,
/// dropping up to 2,900 draws a frame (run shaders-1).
/// `VICTORY_PIPELINE_SLOW=0`: in the order the title creates them, as before.
fn slow_limit(workers: usize) -> usize {
    (workers / 4).max(1)
}

struct Queue {
    /// Requested passes over the screen: before all others, and the only ones
    /// [`SCREEN_WORKERS`] take.
    screen: VecDeque<Arc<Pending>>,
    early: VecDeque<Arc<Pending>>,
    /// Those of the early ones known to take long, and how many of them are
    /// being made ([`slow_limit`]).
    slow: VecDeque<Arc<Pending>>,
    slow_now: usize,
    requested: VecDeque<Arc<Pending>>,
}

impl Queue {
    const fn new() -> Self {
        Self {
            screen: VecDeque::new(),
            early: VecDeque::new(),
            slow: VecDeque::new(),
            slow_now: 0,
            requested: VecDeque::new(),
        }
    }

    /// Queues an unrequested pipeline or promotes one queued at creation.
    /// Returns whether it is new to the queue, or `None` if nothing changed.
    fn push(&mut self, p: &Arc<Pending>, priority: u8) -> Option<bool> {
        static LIMIT_SLOW: OnceLock<bool> = OnceLock::new();
        let slow = priority == EARLY
            && *LIMIT_SLOW.get_or_init(|| std::env::var_os("VICTORY_PIPELINE_SLOW").is_none_or(|v| v != "0"))
            && super::library::slow(&p.key);
        self.push_as(p, priority, slow)
    }

    fn push_as(&mut self, p: &Arc<Pending>, priority: u8, slow: bool) -> Option<bool> {
        let prev = p.queue_state.load(Ordering::Relaxed);
        if prev >= priority {
            return None;
        }
        p.queue_state.store(priority, Ordering::Relaxed);
        if priority == REQUESTED && p.wait_for_head_map {
            self.screen.push_front(p.clone());
        } else if priority == REQUESTED && p.screen && screens_first() {
            self.screen.push_back(p.clone());
        } else if priority == REQUESTED {
            self.requested.push_back(p.clone());
        } else if slow {
            self.slow.push_back(p.clone());
        } else {
            self.early.push_back(p.clone());
        }
        Some(prev == NOT_QUEUED)
    }

    fn pop(&mut self, screens_only: bool, slow_limit: usize) -> Option<Arc<Pending>> {
        // An early entry can have been promoted and claimed already.
        let unclaimed = |list: &mut VecDeque<Arc<Pending>>| {
            std::iter::from_fn(|| list.pop_front()).find(|p| p.queue_state.load(Ordering::Relaxed) == EARLY)
        };
        let mut p = self.screen.pop_front();
        if p.is_none() && !screens_only {
            p = self.requested.pop_front().or_else(|| unclaimed(&mut self.early));
            if p.is_none() && self.slow_now < slow_limit {
                p = unclaimed(&mut self.slow);
                if let Some(p) = &p {
                    p.was_slow.store(true, Ordering::Relaxed);
                    self.slow_now += 1;
                }
            }
        }
        let p = p?;
        p.queue_state.store(MAKING, Ordering::Relaxed);
        Some(p)
    }
}

static QUEUE: Mutex<Queue> = Mutex::new(Queue::new());
static AVAILABLE: Condvar = Condvar::new();

/// Start or promote work for a draw without waiting behind the entire early
/// list. Queue changes and worker claims happen under the same lock.
fn start(p: &Arc<Pending>, priority: u8) {
    if p.queue_state.load(Ordering::Relaxed) >= priority {
        return;
    }
    if !workers() {
        if p.queue_state.swap(MAKING, Ordering::Relaxed) == NOT_QUEUED {
            if p.wait_for_head_map {
                // No worker could be started. Publish the failure instead of
                // moving this unbounded driver call onto the title's thread.
                p.make_with(|| None);
            } else {
                p.make();
            }
        }
        return;
    }
    let mut queue = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(new) = queue.push(p, priority) {
        if new {
            PEAK.fetch_max(WAITING.fetch_add(1, Ordering::Relaxed) + 1, Ordering::Relaxed);
        }
        // (One woken could be a worker that takes passes over the screen
        // alone: it would leave the pipeline to nobody.)
        AVAILABLE.notify_all();
    }
}

/// Workers that make passes over the screen alone, besides the others.
const SCREEN_WORKERS: usize = 2;

fn next(screens_only: bool, slow_limit: usize) -> Arc<Pending> {
    let mut queue = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
    loop {
        if let Some(p) = queue.pop(screens_only, slow_limit) {
            return p;
        }
        queue = AVAILABLE.wait(queue).unwrap_or_else(|e| e.into_inner());
    }
}

/// How many workers make pipelines: `asked` (`VICTORY_PIPELINE_WORKERS`, 1 to
/// 64), else two thirds of the logical processors, 2 to 16. They run below the
/// title's own threads, and a cold cache queues hundreds at once.
fn worker_count_for(asked: Option<usize>, processors: usize, limited_ram: bool) -> usize {
    match asked {
        Some(n) => n.clamp(1, 64),
        None if limited_ram => (processors * 3 / 4).clamp(2, 8),
        None => (processors * 2 / 3).clamp(2, 16),
    }
}

/// Starts the workers ([`worker_count_for`]) on first use.
fn workers() -> bool {
    static RUNNING: OnceLock<bool> = OnceLock::new();
    *RUNNING.get_or_init(|| {
        let mut running = false;
        let workers = worker_count_for(
            std::env::var("VICTORY_PIPELINE_WORKERS")
                .ok()
                .and_then(|v| v.trim().parse().ok()),
            std::thread::available_parallelism().map_or(2, |n| n.get()),
            crate::residency::limited_host_ram(),
        );
        tracing::info!(workers, "pipeline workers");
        let slow_limit = slow_limit(workers);
        let screens = if screens_first() { SCREEN_WORKERS } else { 0 };
        for i in 0..workers + screens {
            let screens_only = i >= workers;
            let spawned = std::thread::Builder::new()
                .name(format!("victory-pipelines-{i}"))
                .spawn(move || {
                    // Below the title's own threads (loading, recording),
                    // and the streaming one, which is below normal itself
                    // (`VICTORY_PIPELINE_LOWEST=0`: level with it, as before).
                    let lowest = !screens_only && std::env::var_os("VICTORY_PIPELINE_LOWEST").is_none_or(|v| v != "0");
                    // SAFETY: a plain call on our own thread.
                    unsafe {
                        use windows::Win32::System::Threading::*;
                        let priority = if lowest { THREAD_PRIORITY_LOWEST } else { THREAD_PRIORITY_BELOW_NORMAL };
                        let _ = SetThreadPriority(GetCurrentThread(), priority);
                    }
                    loop {
                        let p = next(screens_only, slow_limit);
                        p.make();
                        if p.was_slow.load(Ordering::Relaxed) {
                            let mut queue = QUEUE.lock().unwrap_or_else(|e| e.into_inner());
                            queue.slow_now -= 1;
                            AVAILABLE.notify_all();
                        }
                        if WAITING.fetch_sub(1, Ordering::Relaxed) == 1 {
                            drained();
                        } else {
                            super::library::save(false);
                        }
                    }
                });
            if let Err(e) = spawned {
                tracing::warn!(error = %e, "no pipeline worker");
            } else {
                running = true;
            }
        }
        running
    })
}

/// The most pipelines waiting at once since the workers last ran out.
static PEAK: AtomicU64 = AtomicU64::new(0);

/// The workers ran out of work: logged after a batch of 10 or more, and
/// the cache saved (at most every 20 s).
fn drained() {
    use super::library::{EARLY, LOADED, MADE};
    if PEAK.swap(0, Ordering::Relaxed) >= 10 {
        tracing::info!(
            made = MADE.load(Ordering::Relaxed),
            loaded = LOADED.load(Ordering::Relaxed),
            started_early = EARLY.load(Ordering::Relaxed),
            "host pipelines all made"
        );
    }
    super::library::save(false);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configured_pool_limits_slow_precompilation() {
        assert_eq!(slow_limit(worker_count_for(Some(3), 32, false)), 1);
        assert_eq!(slow_limit(worker_count_for(None, 32, true)), 2);
        assert_eq!(slow_limit(worker_count_for(None, 32, false)), 4);
        assert_eq!(slow_limit(worker_count_for(Some(64), 32, false)), 16);
    }

    #[test]
    fn a_panicking_pipeline_is_refused_and_the_worker_continues() {
        let failed = Arc::new(Pending::new(Vec::new(), None));
        let next = Arc::new(Pending::new(Vec::new(), None));
        let failed_job = Arc::clone(&failed);
        let next_job = Arc::clone(&next);
        std::thread::spawn(move || {
            failed_job.make_with(|| panic!("synthetic pipeline build panic"));
            next_job.make_with(|| None);
        })
        .join()
        .expect("the pipeline worker should survive");

        assert!(matches!(failed.made.get(), Some(None)));
        assert!(matches!(next.made.get(), Some(None)));
        let mut rebuilt = false;
        failed.make_with(|| {
            rebuilt = true;
            None
        });
        assert!(!rebuilt);
    }

    #[cfg(feature = "diagnostics")]
    #[test]
    fn missing_distinguishes_omitted_and_pending_pipelines() {
        let reason = "ray activation shader without BVH, left out";
        let mut omitted = Pending::new(Vec::new(), None);
        omitted.left_out = Some(reason);
        assert_eq!(missing(&omitted).as_deref(), Some(reason));

        let pending = Pending::new(Vec::new(), None);
        pending.queue_state.store(MAKING, Ordering::Relaxed);
        assert_eq!(missing(&pending).as_deref(), Some("being made"));
    }

    #[test]
    fn worker_count_follows_the_processors_and_the_switch() {
        assert_eq!(worker_count_for(None, 1, false), 2);
        assert_eq!(worker_count_for(None, 24, false), 16);
        assert_eq!(worker_count_for(None, 64, false), 16);
        assert_eq!(worker_count_for(Some(0), 24, false), 1);
        assert_eq!(worker_count_for(Some(20), 24, false), 20);
        assert_eq!(worker_count_for(None, 8, true), 6);
        assert_eq!(worker_count_for(None, 16, true), 8);
        assert_eq!(worker_count_for(None, 24, true), 8);
        assert_eq!(worker_count_for(Some(12), 8, true), 12);
    }

    /// A first use queues the pipeline and, within the budget, gets the
    /// worker's answer (here: no host pipeline for no shaders).
    #[test]
    fn first_use_is_made_by_a_worker() {
        let p = Arc::new(Pending::new(Vec::new(), None));
        assert!(ready(&p, false).is_none());
        assert_ne!(p.queue_state.load(Ordering::Relaxed), NOT_QUEUED);
        let started = Instant::now();
        while p.made.get().is_none() && started.elapsed() < Duration::from_secs(10) {
            std::thread::yield_now();
        }
        assert!(matches!(p.made.get(), Some(None)));
    }

    #[test]
    fn pending_pipeline_is_recorded_for_the_next_launch() {
        let mut p = Pending::new(Vec::new(), None);
        p.key = "pending-pipeline-test".to_owned();
        // Simulate a worker that has taken the job but cannot finish it yet.
        p.queue_state.store(MAKING, Ordering::Relaxed);
        let p = Arc::new(p);

        assert!(!super::super::library::wanted_early(&p.key));
        assert!(ready(&p, false).is_none());
        assert!(p.made.get().is_none());
        assert!(super::super::library::wanted_early(&p.key));
    }

    #[test]
    fn draws_promote_early_pipelines_without_claiming_them_twice() {
        let earlier = Arc::new(Pending::new(Vec::new(), None));
        let promoted = Arc::new(Pending::new(Vec::new(), None));
        let requested = Arc::new(Pending::new(Vec::new(), None));
        let mut queue = Queue::new();

        assert_eq!(queue.push(&earlier, EARLY), Some(true));
        assert_eq!(queue.push(&promoted, EARLY), Some(true));
        assert_eq!(queue.push(&requested, REQUESTED), Some(true));
        assert_eq!(queue.push(&promoted, REQUESTED), Some(false));
        assert_eq!(queue.push(&promoted, REQUESTED), None);

        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &requested));
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &promoted));
        assert_eq!(queue.push(&promoted, REQUESTED), None);
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &earlier));
        assert!(queue.pop(false, 1).is_none());
    }

    #[test]
    fn pipelines_known_to_take_long_leave_workers_for_the_others() {
        let new = || Arc::new(Pending::new(Vec::new(), None));
        let (long, longer, asked, quick) = (new(), new(), new(), new());
        let mut queue = Queue::new();
        for p in [&long, &longer, &asked] {
            assert_eq!(queue.push_as(p, EARLY, true), Some(true));
        }
        assert_eq!(queue.push_as(&quick, EARLY, false), Some(true));
        // The quick one first, then one long one at a time; one a draw asks
        // for waits for nobody.
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &quick));
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &long));
        assert!(queue.pop(false, 1).is_none());
        assert_eq!(queue.push_as(&asked, REQUESTED, true), Some(false));
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &asked));
        assert!(!asked.was_slow.load(Ordering::Relaxed));
        assert!(queue.pop(false, 1).is_none());
        queue.slow_now -= 1;
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &longer));
        assert!(queue.pop(false, 1).is_none());
    }

    /// Passes over the screen go before all others, and workers of their own
    /// take nothing else.
    #[test]
    fn passes_over_the_screen_go_first() {
        let of = |screen| {
            let mut p = Pending::new(Vec::new(), None);
            p.screen = screen;
            Arc::new(p)
        };
        let (world, screen) = (of(false), of(true));
        let mut queue = Queue::new();
        assert_eq!(queue.push(&world, REQUESTED), Some(true));
        assert_eq!(queue.push(&screen, REQUESTED), Some(true));
        assert!(Arc::ptr_eq(&queue.pop(true, 1).unwrap(), &screen));
        assert!(queue.pop(true, 1).is_none());
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &world));
    }

    #[test]
    fn a_head_map_is_next_for_a_screen_worker() {
        let mut screen = Pending::new(Vec::new(), None);
        screen.screen = true;
        let screen = Arc::new(screen);
        let head = Arc::new(head_map_pipeline(true));
        assert!(head.head_map && head.wait_for_head_map && head.scene_input);
        let control = Arc::new(head_map_pipeline(false));
        assert!(control.head_map && !control.wait_for_head_map && !control.scene_input);
        let mut queue = Queue::new();
        assert_eq!(queue.push(&screen, REQUESTED), Some(true));
        assert_eq!(queue.push(&control, REQUESTED), Some(true));
        assert_eq!(queue.push(&head, REQUESTED), Some(true));
        assert_eq!(queue.push(&head, REQUESTED), None);
        assert!(Arc::ptr_eq(&queue.pop(true, 1).unwrap(), &head));
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &screen));
        assert!(Arc::ptr_eq(&queue.pop(false, 1).unwrap(), &control));
    }

    #[test]
    fn head_map_dispatch_bypasses_the_frame_budget_and_wakes_on_driver_failure() {
        for wait in [true, false] {
            let head = Arc::new(head_map_pipeline(wait));
            head.queue_state.store(MAKING, Ordering::Relaxed);
            head.used.store(true, Ordering::Relaxed);
            let head_wait = HeadMapWait::new(Duration::from_secs(2));

            std::thread::scope(|scope| {
                let (entered, began) = std::sync::mpsc::channel();
                let (finished, result) = std::sync::mpsc::channel();
                let pending = &head;
                let head_wait = &head_wait;
                scope.spawn(move || {
                    entered.send(()).unwrap();
                    let mut used_budget = false;
                    let ready = ready_with(pending, true, head_wait, |_| used_budget = true);
                    assert!(ready.is_none());
                    finished.send((used_budget, pending.made.get().is_some())).unwrap();
                });
                began.recv().unwrap();
                let deadline = if wait { Duration::from_millis(25) } else { Duration::from_secs(1) };
                let first = result.recv_timeout(deadline);
                head.make_with(|| None);
                if wait {
                    assert_eq!(first, Err(std::sync::mpsc::RecvTimeoutError::Timeout));
                    assert_eq!(result.recv_timeout(Duration::from_secs(1)), Ok((false, true)));
                } else {
                    assert_eq!(first, Ok((true, false)));
                }
            });
        }
    }

    #[test]
    fn head_map_timeout_is_shared_and_leaves_the_worker_result_pending() {
        let budget = HeadMapWait::new(Duration::from_millis(25));
        let pending = head_map_pipeline(true);
        std::thread::scope(|scope| {
            let (finished, result) = std::sync::mpsc::channel();
            let budget = &budget;
            let pending = &pending;
            scope.spawn(move || finished.send(budget.wait(pending).is_none()).unwrap());
            let expired = result.recv_timeout(Duration::from_secs(2));
            if expired.is_err() {
                pending.make_with(|| None);
            }
            assert_eq!(expired, Ok(true));
        });
        assert!(budget.timed_out.load(Ordering::Acquire));
        assert!(pending.made.get().is_none());

        let other = head_map_pipeline(true);
        assert!(budget.wait(&other).is_none());

        pending.make_with(|| None);
        assert!(matches!(budget.wait(&pending), Some(None)));
    }

    #[test]
    fn a_pipeline_result_remains_visible_after_the_wait_deadline() {
        let made = OnceLock::new();
        let started = Instant::now();
        assert_eq!(wait_for_made(&made, started, Duration::ZERO), None);
        made.set(Some(7)).unwrap();
        assert_eq!(wait_for_made(&made, started, Duration::ZERO), Some(&Some(7)));
    }

    fn compute_shader(hash: &str) -> Arc<Shader> {
        Arc::new(Shader {
            stage: "compute",
            hash: hash.to_owned(),
            container: Vec::new(),
            ambient: None,
            probe: None,
            translated: OnceLock::from(None),
        })
    }

    fn head_map_pipeline(wait: bool) -> Pending {
        Pending::new_with_policy(vec![compute_shader("61e54e90adccaf6a")], None, &[], wait)
    }

    #[test]
    fn skipped_hash_leaves_pipeline_unqueued_and_unmatched_queues_normally() {
        let skips = super::super::parse_skip_hashes("32ed072b9f845048");

        let skipped = Arc::new(Pending::new_with_policy(
            vec![compute_shader("32ed072b9f845048")],
            None,
            &skips,
            true,
        ));
        assert_eq!(skipped.left_out, Some("skipped by VICTORY_SKIP_PIPELINES"));
        early(&skipped);
        assert!(ready(&skipped, false).is_none());
        assert_eq!(skipped.queue_state.load(Ordering::Relaxed), NOT_QUEUED);
        assert!(skipped.made.get().is_none());
        #[cfg(feature = "diagnostics")]
        assert_eq!(
            missing(&skipped).as_deref(),
            Some("skipped by VICTORY_SKIP_PIPELINES")
        );

        let normal = Arc::new(Pending::new_with_policy(
            vec![compute_shader("0011223344556677")],
            None,
            &skips,
            true,
        ));
        assert!(!normal.head_map && !normal.wait_for_head_map);
        assert_eq!(normal.left_out, None);
        assert!(ready(&normal, false).is_none());
        assert_ne!(normal.queue_state.load(Ordering::Relaxed), NOT_QUEUED);
        let started = Instant::now();
        while normal.made.get().is_none() && started.elapsed() < Duration::from_secs(10) {
            std::thread::yield_now();
        }
        assert!(matches!(normal.made.get(), Some(None)));
    }
}
