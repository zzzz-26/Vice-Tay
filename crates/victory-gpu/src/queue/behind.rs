//! Submissions prepared and sent by a thread of ours, in order, instead of by
//! the thread of the title that submits them (`VICTORY_ASYNC_SUBMIT=1`).
//!
//! The idea and the argument are fxceless/victory-fork's, commit f85101b6
//! (GPL-3.0-or-later, as this): a profile had the title's render thread inside
//! `queue::execute` for 202 of 292 ms of a frame while its recording threads
//! and the card stood idle, and the title does nothing with the result of that
//! work: it only has to be done before the lists run, in order. (Here, at
//! 77-79 ms a frame: 58 ms of it in the submit calls, the card idle for 37 ms
//! behind them.) What differs from that commit: one line for all queues, kept
//! in the order of the title's calls; the present stands in it too; nobody but
//! our thread runs what is in it.
//!
//! With the switch a submission's lists are taken from the title's lists (as
//! they are for every flight, `crate::list::detach`) and put in the [`Lane`],
//! with what they wait for, the queues' fence waits and signals, and the
//! presents. [`Lane::serve`], on a thread of ours, takes from it whatever is
//! free to run, the oldest first; within a queue the order is the title's,
//! always. What tells the title that work is done (fence values,
//! bottom-of-pipe writes) is told by the flights as before, when the card has
//! run them: later than the title's call by more than before, never earlier.
//!
//! What the title may do between its submit call and the moment our thread
//! prepares the lists, and why that is all right:
//!
//! - Reset or release the lists and allocators: their host sides, upload
//!   memory and draw state were taken out of them.
//! - Write upload memory, descriptors, constants for later lists: those of
//!   lists submitted are the card's to read until a fence behind them is
//!   reached, on the console as here; no fence behind them is reached before
//!   the lists have run ([`signal`] goes into the line behind them).
//! - Signal, wait, present on the queue: in the line behind them.
//! - Write a fence or a value in memory by the CPU: at once; our thread looks
//!   at what it stands at again ([`wake`], and every millisecond).
//! - Map pages (`CopyPageMappingsBatchX`): at once, as before, where the lists
//!   on the card may be reading them too; the bytes of uploads were read when
//!   the lists were recorded.
//! - Ask for something that has to see earlier submissions done (a read-back,
//!   a fence waited for without an event): [`drain`] first.
//! - Release a texture or a view's memory: what is retired waits for what is
//!   in the line too ([`queued`]).
//!
//! The line is bounded: a present returns when the one before it was shown
//! ([`frames_most`]), and a submission when at most [`runs_most`] are in the
//! line.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Condvar, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

use super::This;
use crate::fence::FenceRef;
use crate::host::JoinStop;

/// One thing in the line: what it stands at, if it does, and what it does.
pub(super) trait Work: Send {
    /// What has to be met before it goes.
    type Ask;
    fn ask(&self) -> Option<Self::Ask>;
    fn met(ask: &Self::Ask) -> bool;
    /// Whether it is a present: everything before it goes first, met or not.
    fn shows(&self) -> bool;
    /// Whether it is a submission.
    fn runs(&self) -> bool;
    /// Whether it stands for other work, which takes its place in the line
    /// once it is free to go ([`Work::unfold`]).
    fn folded(&self) -> bool {
        false
    }
    fn unfold(self) -> Vec<Self>
    where
        Self: Sized,
    {
        vec![self]
    }
    /// Waits until a folded one is free to go.
    fn settle(_: &Self::Ask) {}
    /// Takes `next`, what its queue was given behind it, along to be done with
    /// it; or gives it back. Whether it is a submission that was taken.
    fn take_along(&mut self, next: Self) -> Result<bool, Self>
    where
        Self: Sized,
    {
        Err(next)
    }
    fn join_stopped(&self, _reason: JoinStop) {}
    /// Does it. `forced`: its wait was not met.
    fn go(self, forced: bool);
    /// Renderer port #7 (`crate::port_journal`, from renderer patch #4): a mark taken when an entry is
    /// put in the line (by the thread that puts it, before the line's lock):
    /// the round of the heaps' records begun after that. 0: none.
    fn mark_now() -> u64 {
        0
    }
    /// A mark taken when a wait goes or is taken along (on our thread): a
    /// round that begins after.
    fn mark_later() -> u64 {
        0
    }
    /// Whether what its queue has behind this wait may have been written
    /// only just before the wait was met (by the CPU, which writes, then
    /// meets the wait): it was met in memory. A wait met by a bottom-of-pipe
    /// write of a flight or of the submission it is taken along with is the
    /// card's order, not the CPU's.
    fn raises(&self) -> bool {
        true
    }
    /// Told just before [`Work::go`] of a submission: the highest mark of
    /// what it is made of and of the waits its queue went past.
    fn marked(&mut self, _mark: u64) {}
}

struct Entry<W> {
    number: u64,
    queue: usize,
    what: W,
    /// [`Work::mark_now`] when it was put in the line.
    mark: u64,
}

struct State<W> {
    /// In the order of the title's calls, whatever queue they were made on.
    items: VecDeque<Entry<W>>,
    /// The number the next entry gets.
    next: u64,
    /// The queue whose entry is being done now (it is no longer in `items`;
    /// what that queue is given stays behind it all the same).
    running: Option<usize>,
    /// Times a look at the line was asked for, and the last of them that a
    /// whole look has followed.
    asked: u64,
    served: u64,
    /// Presents and submissions in the line or being done.
    presents: usize,
    runs: usize,
    /// Per queue, the highest [`Work::mark_later`] of the waits it went past:
    /// what is behind a wait may have been written only before the wait was
    /// met, not before it was put in the line.
    floors: Vec<(usize, u64)>,
}

impl<W> State<W> {
    fn floor(&self, queue: usize) -> u64 {
        self.floors.iter().find(|f| f.0 == queue).map_or(0, |f| f.1)
    }

    fn raise(&mut self, queue: usize, mark: u64) {
        match self.floors.iter_mut().find(|f| f.0 == queue) {
            Some(f) => f.1 = f.1.max(mark),
            None => self.floors.push((queue, mark)),
        }
    }
}

/// See the module's text.
pub(super) struct Lane<W: Work> {
    state: Mutex<State<W>>,
    /// Something to look at.
    work: Condvar,
    /// Something was done, or a look at the line ended.
    moved: Condvar,
}

impl<W: Work> Lane<W> {
    pub(super) const fn new() -> Self {
        Lane {
            state: Mutex::new(State {
                items: VecDeque::new(),
                next: 0,
                running: None,
                asked: 0,
                served: 0,
                presents: 0,
                runs: 0,
                floors: Vec::new(),
            }),
            work: Condvar::new(),
            moved: Condvar::new(),
        }
    }

    fn lock(&self) -> MutexGuard<'_, State<W>> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn put(state: &mut State<W>, queue: usize, what: W, mark: u64) {
        state.presents += usize::from(what.shows());
        state.runs += usize::from(what.runs());
        let number = state.next;
        state.next += 1;
        state.items.push_back(Entry { number, queue, what, mark });
    }

    /// Adds to the end of the line, and has it looked at.
    pub(super) fn push(&self, queue: usize, what: impl IntoIterator<Item = W>) {
        let what: Vec<W> = what.into_iter().collect();
        let mark = if what.iter().any(W::runs) { W::mark_now() } else { 0 };
        let mut state = self.lock();
        for what in what {
            Self::put(&mut state, queue, what, mark);
        }
        state.asked += 1;
        drop(state);
        self.work.notify_one();
    }

    /// Adds `what` if `queue` has anything in the line or being done (or
    /// `always`); else does `now`, with nothing of the queue's able to come
    /// in between.
    pub(super) fn push_behind(&self, queue: usize, always: bool, what: W, now: impl FnOnce()) {
        let mut state = self.lock();
        if always || state.running == Some(queue) || state.items.iter().any(|e| e.queue == queue) {
            Self::put(&mut state, queue, what, 0);
        } else {
            now();
        }
        state.asked += 1;
        drop(state);
        self.work.notify_one();
    }

    /// Has the line looked at: something an entry stands at may have moved.
    pub(super) fn wake(&self) {
        let mut state = self.lock();
        if state.items.is_empty() {
            return;
        }
        state.asked += 1;
        drop(state);
        self.work.notify_one();
    }

    /// Waits until everything in the line that is free to run has been done
    /// (what comes in after this call may not be).
    pub(super) fn drain(&self) {
        let folded: Vec<W::Ask> =
            self.lock().items.iter().filter(|e| e.what.folded()).filter_map(|e| e.what.ask()).collect();
        folded.iter().for_each(W::settle);
        let mut state = self.lock();
        if state.items.is_empty() && state.running.is_none() {
            return;
        }
        state.asked += 1;
        let ticket = state.asked;
        self.work.notify_one();
        while state.served < ticket {
            state = self.moved.wait(state).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// Waits until at most `most` presents are in the line.
    pub(super) fn wait_presents(&self, most: usize) {
        let mut state = self.lock();
        while state.presents > most {
            state = self.moved.wait(state).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// Waits until at most `most` submissions are in the line, or what is
    /// left of it stands at something.
    pub(super) fn wait_runs(&self, most: usize) {
        let mut state = self.lock();
        while state.runs > most && state.served < state.asked {
            state = self.moved.wait(state).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// Looks at the line for as long as the process runs.
    fn serve(&self) -> ! {
        loop {
            let (asked, limit) = {
                let mut state = self.lock();
                while state.asked == state.served {
                    if state.items.is_empty() {
                        state = self.work.wait(state).unwrap_or_else(|e| e.into_inner());
                    } else {
                        // It stands at something the CPU may write without
                        // a call of ours.
                        let (back, out) =
                            self.work.wait_timeout(state, Duration::from_millis(1)).unwrap_or_else(|e| e.into_inner());
                        state = back;
                        if out.timed_out() {
                            break;
                        }
                    }
                }
                (state.asked, state.next)
            };
            self.look(limit);
            let mut state = self.lock();
            state.served = state.served.max(asked);
            drop(state);
            self.moved.notify_all();
        }
    }

    /// Does what is free to run of the entries before number `limit`.
    pub(super) fn look(&self, limit: u64) {
        while self.step(limit) {}
    }

    /// The parts of `entry`, which stands for them, in place `at`.
    fn unfold(state: &mut State<W>, at: usize, entry: Entry<W>) {
        let (number, queue, runs, mark) = (entry.number, entry.queue, usize::from(entry.what.runs()), entry.mark);
        let parts = entry.what.unfold();
        state.runs -= runs;
        state.runs += parts.iter().filter(|p| p.runs()).count();
        for (i, what) in parts.into_iter().enumerate() {
            state.items.insert(at + i, Entry { number, queue, what, mark });
        }
    }

    /// Does one entry: the oldest that is the first of its queue and free to
    /// run. Whether there was one. Only one thread calls this.
    fn step(&self, limit: u64) -> bool {
        // Each queue's first entry, oldest first.
        let mut heads: Vec<(u64, usize, Option<W::Ask>, bool)> = Vec::new();
        // Whether one of them stands for others not known yet.
        let mut folded = false;
        {
            let state = self.lock();
            for e in state.items.iter().take_while(|e| e.number < limit) {
                if !heads.iter().any(|h| h.1 == e.queue) {
                    heads.push((e.number, e.queue, e.what.ask(), e.what.shows()));
                    folded |= e.what.folded();
                }
            }
        }
        let Some(&(first, ..)) = heads.first() else {
            return false;
        };
        let ready = heads.iter().find(|(number, _, ask, shows)| match ask {
            Some(ask) => W::met(ask),
            // A present comes behind everything given before it.
            None => !shows || *number == first || !super::forces(),
        });
        let (number, forced) = match ready {
            Some(head) => (head.0, false),
            None => {
                // Nothing is free to run. With a present in the line the
                // oldest goes on regardless: a wait is not kept across
                // frames (`super::present`). (Not while there is work whose
                // parts are not known: it may be what is waited for.)
                if folded {
                    return false;
                }
                let state = self.lock();
                if !super::forces() || !state.items.iter().any(|e| e.number < limit && e.what.shows()) {
                    return false;
                }
                (first, true)
            }
        };
        let mut entry = {
            let mut state = self.lock();
            let Some(at) = state.items.iter().position(|e| e.number == number) else {
                return false;
            };
            let Some(entry) = state.items.remove(at) else {
                return false;
            };
            state.running = Some(entry.queue);
            entry
        };
        let (shows, mut runs) = (entry.what.shows(), usize::from(entry.what.runs()));
        // The highest mark of what it is made of (renderer port #7).
        let mut mark = entry.mark;
        if entry.what.ask().is_some() && !entry.what.folded() && entry.what.raises() {
            // A wait that goes: what its queue has behind it was written
            // before it was met, which is now at the latest.
            let later = W::mark_later();
            self.lock().raise(entry.queue, later);
        }
        if entry.what.folded() {
            // Its parts take its place: behind everything before it, before
            // everything after it (they share its number; the first of them
            // is found first).
            let mut state = self.lock();
            state.running = None;
            let at = state.items.iter().position(|e| e.number > entry.number).unwrap_or(state.items.len());
            Self::unfold(&mut state, at, entry);
            drop(state);
            self.moved.notify_all();
            return true;
        }
        // What follows it in its queue, if that can go with it. (Nobody else
        // takes anything out, and what comes in goes to the end.)
        while runs > 0 && !forced {
            let next = {
                let mut state = self.lock();
                let at = state.items.iter().position(|e| e.queue == entry.queue && e.number < limit);
                at.and_then(|at| Some((at, state.items.remove(at)?)))
            };
            let Some((at, next)) = next else {
                entry.what.join_stopped(JoinStop::QueueBoundary);
                break;
            };
            // (Back where it was: parts of one submission share a number.)
            if next.what.folded() {
                let free = next.what.ask().is_none_or(|ask| W::met(&ask));
                let mut state = self.lock();
                if free {
                    Self::unfold(&mut state, at, next);
                    continue;
                }
                state.items.insert(at, next);
                entry.what.join_stopped(JoinStop::FoldedWait);
                break;
            }
            let Entry { number, queue, what, mark: theirs } = next;
            let waits = what.ask().is_some() && what.raises();
            match entry.what.take_along(what) {
                Ok(run) => {
                    runs += usize::from(run);
                    mark = mark.max(theirs);
                    if waits {
                        let later = W::mark_later();
                        self.lock().raise(queue, later);
                    }
                }
                Err(what) => {
                    self.lock().items.insert(at, Entry { number, queue, what, mark: theirs });
                    break;
                }
            }
        }
        if runs > 0 {
            let floor = self.lock().floor(entry.queue);
            entry.what.marked(mark.max(floor));
        }
        entry.what.go(forced);
        {
            let mut state = self.lock();
            state.running = None;
            state.presents -= usize::from(shows);
            state.runs -= runs;
        }
        self.moved.notify_all();
        true
    }
}

type Recordings = std::sync::Arc<[std::sync::Arc<crate::list::thin::Recording>]>;

/// What the title's queues put in the line.
enum Job {
    /// The queue waits for the fence to reach the value.
    Wait(FenceRef, u64),
    /// A part of a submission waits for a memory value.
    Value(crate::list::ValueWait),
    /// A submission, or a part of one: the lists' host sides, the order
    /// they run in, and the heap-sync stamp captured at enqueue.
    Run(usize, Vec<crate::list::Detached>, Vec<usize>, u64),
    /// A submission whose lists are being translated ([`crate::list::thin`]):
    /// the parts of it are known when they are. Carries the enqueue heap-sync
    /// stamp so unfolded runs inherit it.
    Lists(usize, Recordings, u64),
    /// Parts taken along with a submission ([`crate::host::joins`]): where they
    /// begin in the order, the signals given behind them, and the combined
    /// heap-sync stamp (`merge_stamp`, where `0` dominates to force a fresh
    /// round whenever any joined part is unstamped).
    Joined(usize, Vec<crate::list::Detached>, Vec<usize>, Vec<usize>, Vec<(FenceRef, u64)>, u64),
    /// A signal of the queue's.
    Signal(FenceRef, u64),
    /// A frame is shown: the base texture (and optional overlay plane) and the states their lists leave them in.
    Present(
        Option<(std::sync::Arc<crate::resource::HostTexture>, u32)>,
        Option<(std::sync::Arc<crate::resource::HostTexture>, u32)>,
        super::Shown,
    ),
}

// SAFETY: D3D12 objects are free-threaded; a job is touched by whoever put it
// in the line and then, once it is taken out, by our thread alone (the flights
// it becomes are handed to other threads the same way).
unsafe impl Send for Job {}

enum Ask {
    Wait(FenceRef, u64),
    Value(crate::list::ValueWait),
    Lists(Recordings),
}

impl Work for Job {
    type Ask = Ask;

    fn ask(&self) -> Option<Ask> {
        match self {
            Job::Wait(fence, value) => Some(Ask::Wait(fence.clone(), *value)),
            Job::Value(wait) => Some(Ask::Value(*wait)),
            Job::Lists(_, lists, _) => Some(Ask::Lists(lists.clone())),
            _ => None,
        }
    }

    fn folded(&self) -> bool {
        matches!(self, Job::Lists(..))
    }

    fn unfold(self) -> Vec<Self> {
        let Job::Lists(queue, recordings, stamp) = self else {
            return vec![self];
        };
        let lists: Vec<This> = recordings.iter().map(|r| r.back()).collect();
        let jobs = jobs_of(queue as This, &lists, stamp);
        // It stood for one submission not begun.
        let runs = jobs.iter().filter(|j| j.runs()).count() as u64;
        QUEUED.fetch_add(runs, Relaxed);
        QUEUED.fetch_sub(1, Relaxed);
        STATS[RUNS].fetch_add(runs, Relaxed);
        jobs
    }

    fn settle(ask: &Ask) {
        if let Ask::Lists(recordings) = ask {
            recordings.iter().for_each(|r| r.wait());
        }
    }

    fn mark_now() -> u64 {
        if crate::port_journal::on() { crate::descriptor::round_after_now() } else { 0 }
    }

    fn mark_later() -> u64 {
        // (Asked for now when `port_journal` is on: the round runs while the
        // lane does what is before the part behind the wait. Otherwise capture
        // the current `SyncRounds` stamp so any queued submission behind a
        // fence or memory-value wait still waits for a round begun after the
        // wait was met.)
        if crate::port_journal::on() {
            crate::descriptor::round_after_now()
        } else {
            crate::descriptor::heap_sync_stamp()
        }
    }

    fn raises(&self) -> bool {
        // Without `port_journal`, treat every wait conservatively without
        // re-reading title memory.
        if !crate::port_journal::on() {
            return true;
        }
        match self {
            Job::Value(wait) => wait.met(),
            _ => true,
        }
    }

    fn marked(&mut self, mark: u64) {
        if crate::port_journal::on() {
            crate::port_journal::set_mark(mark);
        } else if mark > 0 {
            match self {
                Job::Run(.., stamp) | Job::Joined(.., stamp) if *stamp > 0 => *stamp = (*stamp).max(mark),
                _ => {}
            }
        }
    }

    fn met(ask: &Ask) -> bool {
        match ask {
            Ask::Wait(fence, value) => crate::fence::completed(fence.as_this()) >= *value,
            Ask::Value(wait) => super::value_met(wait),
            Ask::Lists(recordings) => recordings.iter().all(|r| r.done()),
        }
    }

    fn shows(&self) -> bool {
        matches!(self, Job::Present(..))
    }

    fn runs(&self) -> bool {
        matches!(self, Job::Run(..) | Job::Joined(..) | Job::Lists(..))
    }

    fn join_stopped(&self, reason: JoinStop) {
        note_join_stop(reason);
    }

    fn take_along(&mut self, next: Job) -> Result<bool, Job> {
        let (most, most_lists) = crate::host::joins();
        if most < 2 {
            note_join_stop(JoinStop::Disabled);
            return Err(next);
        }
        // Nothing but waits and signals that a submission may follow.
        let (count, lists, signals) = match self {
            Job::Run(_, lists, order, _) => (1, (&*lists, &*order), &[][..]),
            Job::Joined(_, lists, order, parts, signals, _) => (1 + parts.len(), (&*lists, &*order), &signals[..]),
            _ => {
                note_join_stop(JoinStop::BoundaryJob);
                return Err(next);
            }
        };
        let hosts = || lists.1.iter().map(|&i| lists.0[i].host());
        match &next {
            Job::Value(wait) => {
                let met = wait.met()
                    || crate::host::late_met_known(wait) == Some(true)
                    || hosts().any(|list| crate::host::will_meet(list, wait));
                if !met {
                    note_join_stop(JoinStop::ValueWait);
                    return Err(next);
                }
            }
            Job::Wait(fence, value) => {
                if crate::fence::completed(fence.as_this()) < *value
                    && !signals.iter().any(|(f, v)| f == fence && v >= value)
                {
                    note_join_stop(JoinStop::FenceWait);
                    return Err(next);
                }
            }
            Job::Signal(..) => {}
            Job::Run(_, more, order, _) => {
                let runs: Vec<&crate::host::HostList> = order.iter().map(|&i| more[i].host()).collect();
                let stop = if count >= most {
                    Some(JoinStop::PartLimit)
                } else if lists.1.len() + runs.len() > most_lists {
                    Some(JoinStop::ListLimit)
                } else {
                    crate::host::join_blocked(hosts(), &runs)
                };
                if let Some(reason) = stop {
                    note_join_stop(reason);
                    return Err(next);
                }
            }
            Job::Joined(..) | Job::Present(..) | Job::Lists(..) => {
                note_join_stop(JoinStop::BoundaryJob);
                return Err(next);
            }
        }
        // (A signal held in the line keeps its fence: no placeholder job
        // can stand in for this one while it is taken apart.)
        if let Job::Run(queue, lists, order, stamp) = self {
            *self = Job::Joined(*queue, std::mem::take(lists), std::mem::take(order), Vec::new(), Vec::new(), *stamp);
        }
        let Job::Joined(_, lists, order, parts, signals, stamp) = self else {
            return Err(next);
        };
        let run = matches!(next, Job::Run(..));
        match next {
            Job::Run(_, more, then, next_stamp) => {
                parts.push(order.len());
                order.extend(then.into_iter().map(|i| i + lists.len()));
                lists.extend(more);
                *stamp = merge_stamp(*stamp, next_stamp);
                STATS[JOINED].fetch_add(1, Relaxed);
            }
            Job::Signal(fence, value) => signals.push((fence, value)),
            _ => {}
        }
        Ok(run)
    }

    fn go(self, forced: bool) {
        let began = Instant::now();
        let _doing = crate::timeline::span(match self {
            Job::Run(..) => "the line: a submission",
            Job::Joined(..) => "the line: submissions joined",
            Job::Present(..) => "the line: a present",
            Job::Signal(..) => "the line: a signal",
            _ => "the line: a wait that is met",
        });
        if forced {
            STATS[FORCED].fetch_add(1, Relaxed);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::Forced, 1);
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!("a queue still stood at a wait at a present: it was not met"));
        }
        match self {
            // (A folded one is unfolded, never done.)
            Job::Wait(..) | Job::Value(..) | Job::Lists(..) => {}
            Job::Run(queue, lists, order, stamp) => {
                let _ = queue;
                #[cfg(feature = "diagnostics")]
                crate::capture::op(queue, || format!("RUN submission of {} lists (its thread's)", order.len()));
                match crate::host::host() {
                    Some(host) => host.execute_parts_stamped(lists, order, &[], (stamp > 0).then_some(stamp)),
                    None => drop(lists),
                }
                QUEUED.fetch_sub(1, Relaxed);
            }
            Job::Joined(queue, lists, order, parts, signals, stamp) => {
                let _ = queue;
                #[cfg(feature = "diagnostics")]
                crate::capture::op(queue, || {
                    format!("RUN {} parts of submissions in one, {} lists (its thread's)", 1 + parts.len(), order.len())
                });
                match crate::host::host() {
                    Some(host) => host.execute_parts_stamped(lists, order, &parts, (stamp > 0).then_some(stamp)),
                    None => drop(lists),
                }
                // (A part that had no list is one all the same.)
                QUEUED.fetch_sub(1 + parts.len() as u64, Relaxed);
                for (fence, value) in signals {
                    if !crate::host::late_signal(fence.as_this(), value) {
                        crate::fence::set(fence.as_this(), value);
                    }
                }
            }
            Job::Signal(fence, value) => {
                if !crate::host::late_signal(fence.as_this(), value) {
                    crate::fence::set(fence.as_this(), value);
                }
            }
            Job::Present(texture, overlay, shown) => {
                if let Some((texture, state)) = texture {
                    // (Where its lists were not translated when the title
                    // presented: the state they left it in.)
                    let state = if state == u32::MAX { texture.state.load(Relaxed) } else { state };
                    let over = overlay.as_ref().map(|(ov, ov_state)| {
                        let st = if *ov_state == u32::MAX { ov.state.load(Relaxed) } else { *ov_state };
                        (&ov.texture, st)
                    });
                    super::show_planes(&texture.texture, state, over, shown);
                }
                QUEUED.fetch_sub(1, Relaxed);
            }
        }
        // (The mark was this submission's alone.)
        crate::port_journal::set_mark(0);
        STATS[BUSY].fetch_add(began.elapsed().as_micros() as u64, Relaxed);
    }
}

static LANE: Lane<Job> = Lane::new();

/// Submissions and presents in the line that have not been given to the host
/// yet.
static QUEUED: AtomicU64 = AtomicU64::new(0);

/// Since the last frame-rate line: submissions put in the line, presents, the
/// times they found things in it (the sum of their numbers) and the most; what
/// threads of the title waited, in microseconds, for room in the line, at a
/// present, and for a drain, and the drains; what our thread worked; waits
/// not met at a present.
static STATS: [AtomicU64; 11] = [const { AtomicU64::new(0) }; 11];
const RUNS: usize = 0;
const PRESENTS: usize = 1;
const DEPTH: usize = 2;
const DEEPEST: usize = 3;
const ROOM: usize = 4;
const AT_PRESENT: usize = 5;
const DRAINING: usize = 6;
const DRAINS: usize = 7;
const BUSY: usize = 8;
const FORCED: usize = 9;
/// Parts prepared together with the one before them.
const JOINED: usize = 10;

#[cfg(feature = "diagnostics")]
const JOIN_STOP_COUNT: usize = JoinStop::EarlierWrite as usize + 1;
#[cfg(feature = "diagnostics")]
static JOIN_STOPS: [AtomicU64; JOIN_STOP_COUNT] = [const { AtomicU64::new(0) }; JOIN_STOP_COUNT];

fn note_join_stop(reason: JoinStop) {
    #[cfg(feature = "diagnostics")]
    {
        static ON: OnceLock<bool> = OnceLock::new();
        if *ON.get_or_init(|| std::env::var_os("VICTORY_JOIN_REASONS").is_some_and(|v| v == "1")) {
            JOIN_STOPS[reason as usize].fetch_add(1, Relaxed);
        }
    }
    #[cfg(not(feature = "diagnostics"))]
    let _ = reason;
}

/// How many submissions may be in the line when a submit call returns
/// (`VICTORY_SUBMIT_DEPTH`; 64: about two frames of the title's).
fn runs_most() -> usize {
    static MOST: OnceLock<usize> = OnceLock::new();
    *MOST.get_or_init(|| number("VICTORY_SUBMIT_DEPTH").unwrap_or(64).max(1))
}

/// How many presents may be in the line when a present call returns
/// (`VICTORY_SUBMIT_FRAMES`; 1: the frame before is sent, this one may not be;
/// 0: both are).
fn frames_most() -> usize {
    static MOST: OnceLock<usize> = OnceLock::new();
    *MOST.get_or_init(|| number("VICTORY_SUBMIT_FRAMES").unwrap_or(1))
}

fn number(name: &str) -> Option<usize> {
    std::env::var(name).ok()?.trim().parse().ok()
}

thread_local! {
    /// This is the thread that does what is in the line.
    static OURS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether submissions are prepared by a thread of ours
/// (`VICTORY_ASYNC_SUBMIT=1`; needs flights with the present sent ahead, as
/// the relay has them, and lists cut at their waits).
pub(super) fn on() -> bool {
    // Diagnostics: on and off while the title runs (`crate::ab`).
    #[cfg(feature = "diagnostics")]
    if let Some(on) = crate::ab::pick("VICTORY_ASYNC_SUBMIT") {
        return on && started();
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        (std::env::var_os("VICTORY_ASYNC_SUBMIT").is_some_and(|v| v == "1") || crate::list::thin::wanted()) && started()
    })
}

/// The line is about to be left for the way without it: everything in it is
/// done first. (At a present, by the title's thread that submits.)
#[cfg(feature = "diagnostics")]
pub(super) fn leave() {
    LANE.wait_presents(0);
    LANE.drain();
}

/// Starts the thread, once: whether there is one.
fn started() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        if !crate::host::late_free_present() || !super::cuts() {
            tracing::warn!("VICTORY_ASYNC_SUBMIT=1 needs the relay and lists cut at their waits: submissions are prepared by the thread that makes them");
            return false;
        }
        let worker = || {
            use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_ABOVE_NORMAL};
            // SAFETY: this thread's own priority.
            unsafe { SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL) };
            crate::cores::keep_on_fast_cores("victory-submit");
            OURS.set(true);
            let ended = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| LANE.serve()));
            if ended.is_err() {
                // Nothing would ever be sent again, and the title's thread
                // would stand at its next present for good.
                crate::host::fail(format_args!("the thread that prepares and sends submissions panicked"));
            }
        };
        match std::thread::Builder::new().name("victory-submit".into()).spawn(worker) {
            Ok(_) => {
                tracing::info!(
                    frames = frames_most(),
                    runs = runs_most(),
                    "submissions are prepared and sent by a thread of ours, not by the thread that submits (VICTORY_ASYNC_SUBMIT=1)"
                );
                true
            }
            Err(e) => {
                tracing::warn!(error = %e, "no thread to prepare submissions: the threads that make them do");
                false
            }
        }
    })
}

/// Submissions in the line that the host has not begun
/// (`crate::host::retire` waits for them too).
pub(super) fn queued() -> u64 {
    QUEUED.load(Relaxed)
}

/// Something a queue may stand at moved.
pub(super) fn wake() {
    LANE.wake();
}

/// This thread does work the line waits for.
pub(super) fn no_drain_here() {
    OURS.set(true);
}

/// Before something that has to see the submissions made so far sent: they
/// are, as far as they are free to run. Not on our own thread, which is
/// inside one of them.
pub(super) fn drain() {
    if OURS.get() {
        return;
    }
    let began = Instant::now();
    LANE.drain();
    STATS[DRAINING].fetch_add(began.elapsed().as_micros() as u64, Relaxed);
    STATS[DRAINS].fetch_add(1, Relaxed);
}

/// On the upscaler's thread, wait for any prior Present to consume its
/// frame-generation buffers. A blocked title queue can still make progress.
pub(super) fn wait_before_framegen() {
    if OURS.get() {
        return;
    }
    let state = LANE.lock();
    let (_state, wait) = LANE
        .moved
        .wait_timeout_while(state, Duration::from_millis(250), |state| state.presents != 0)
        .unwrap_or_else(|e| e.into_inner());
    if wait.timed_out() {
        tracing::debug!(target: "victory_gpu::framegen", "frame generation: previous present still pending at upscale");
    }
}

/// A submission of `lists` to queue `this`.
pub(super) fn submit(this: This, lists: &[This]) {
    let queue = this as usize;
    let stamp = crate::descriptor::heap_sync_stamp();
    if crate::list::thin::on() {
        let recordings = crate::list::thin::submitted(lists);
        if recordings.is_empty() {
            return;
        }
        let before = QUEUED.fetch_add(1, Relaxed);
        STATS[DEPTH].fetch_add(before, Relaxed);
        STATS[DEEPEST].fetch_max(before + 1, Relaxed);
        LANE.push(queue, [Job::Lists(queue, recordings, stamp)]);
        room(before + 1);
        return;
    }
    let jobs = jobs_of(this, lists, stamp);
    let runs = jobs.iter().filter(|j| j.runs()).count() as u64;
    let before = QUEUED.fetch_add(runs, Relaxed);
    STATS[RUNS].fetch_add(runs, Relaxed);
    STATS[DEPTH].fetch_add(before, Relaxed);
    STATS[DEEPEST].fetch_max(before + runs, Relaxed);
    LANE.push(queue, jobs);
    room(before + runs);
}

/// Waits for room in the line, with `queued` submissions in it.
fn room(queued: u64) {
    if queued > runs_most() as u64 {
        let began = Instant::now();
        let _waiting = crate::timeline::span("a submission waits for room in the line");
        LANE.wait_runs(runs_most());
        STATS[ROOM].fetch_add(began.elapsed().as_micros() as u64, Relaxed);
    }
}

/// Combines enqueue heap-sync stamps when joining two submissions: `0` means
/// unstamped (must request a fresh round at preparation) and dominates any
/// stamped part.
fn merge_stamp(a: u64, b: u64) -> u64 {
    if a == 0 || b == 0 { 0 } else { a.max(b) }
}

/// What a submission of `lists` is in the line.
fn jobs_of(this: This, lists: &[This], stamp: u64) -> Vec<Job> {
    let queue = this as usize;
    // In one piece up to the first list that waits for a memory value, as
    // `super::submit_cut` runs them; the rest part by part.
    let first = lists.iter().position(|&l| crate::list::cut_list(l)).unwrap_or(lists.len());
    #[cfg(feature = "diagnostics")]
    crate::capture::op(queue, || {
        let named: Vec<String> = lists
            .iter()
            .map(|&l| match crate::list::host_list(l) {
                Some(h) => format!("{:#x}", std::ptr::from_ref(h) as usize),
                None => format!("title {l:?}"),
            })
            .collect();
        format!(
            "SUBMIT queue kind {} lists {named:?} to the line (the first {first} in one piece)",
            super::queue(this).kind
        )
    });
    let mut jobs: Vec<Job> = Vec::new();
    if first > 0 {
        let (taken, order) = crate::list::detach(&lists[..first]);
        jobs.push(Job::Run(queue, taken, order, stamp));
    }
    if first < lists.len() {
        // Parts behind a memory-value wait may depend on descriptor records
        // written after `submit` (before that wait is met), so they and every
        // subsequent part stay unstamped (`0` -> fresh round at preparation).
        let mut after_wait = false;
        for (waits, taken, order) in crate::list::detach_parts(&lists[first..]) {
            if !waits.is_empty() {
                after_wait = true;
            }
            jobs.extend(waits.into_iter().map(Job::Value));
            jobs.push(Job::Run(queue, taken, order, if after_wait { 0 } else { stamp }));
        }
    }
    jobs
}

/// A signal of queue `this`: behind what it has in the line; if that is
/// nothing, behind the flights in the air, or now.
pub(super) fn signal(this: This, fence: This, value: u64) {
    LANE.push_behind(this as usize, false, Job::Signal(FenceRef::hold(fence), value), || {
        if !crate::host::late_signal(fence, value) {
            crate::fence::set(fence, value);
        }
    });
}

/// Queue `this` waits for `fence`: nothing if it has nothing in the line and
/// the fence is there already.
pub(super) fn wait(this: This, fence: This, value: u64) {
    let always = crate::fence::completed(fence) < value;
    LANE.push_behind(this as usize, always, Job::Wait(FenceRef::hold(fence), value), || {});
}

/// A present on queue `this` of `texture`, in the state the lists recorded so
/// far leave it in.
pub(super) fn present(
    this: This,
    texture: Option<(std::sync::Arc<crate::resource::HostTexture>, u32)>,
    overlay: Option<(std::sync::Arc<crate::resource::HostTexture>, u32)>,
    shown: super::Shown,
) {
    QUEUED.fetch_add(1, Relaxed);
    STATS[PRESENTS].fetch_add(1, Relaxed);
    LANE.push(this as usize, [Job::Present(texture, overlay, shown)]);
    let began = Instant::now();
    let _waiting = crate::timeline::span("a present waits for the one before it to be sent");
    LANE.wait_presents(frames_most());
    STATS[AT_PRESENT].fetch_add(began.elapsed().as_micros() as u64, Relaxed);
}

/// For the frame-rate line.
pub(super) fn line(frames: f64) {
    let s: Vec<f64> = STATS.iter().map(|s| s.swap(0, Relaxed) as f64).collect();
    let ms = |i: usize| s[i] / 1000.0 / frames;
    tracing::info!(
        "submissions prepared by a thread of ours, per frame (VICTORY_ASYNC_SUBMIT): {:.1} in {:.1} presents, {:.1} in the line \
         when one was added (at most {:.0}); that thread worked {:.1} ms; threads of the title waited {:.1} ms for room in the \
         line, {:.1} ms at presents, {:.1} ms in {:.1} drains; waits not met at a present {:.2}; {:.1} of the \
         submissions were prepared together with the one before them (VICTORY_JOIN_PARTS)",
        s[RUNS] / frames,
        s[PRESENTS] / frames,
        s[DEPTH] / s[RUNS].max(1.0),
        s[DEEPEST],
        ms(BUSY),
        ms(ROOM),
        ms(AT_PRESENT),
        ms(DRAINING),
        s[DRAINS] / frames,
        s[FORCED] / frames,
        s[JOINED] / frames
    );
    #[cfg(feature = "diagnostics")]
    {
        let stops: [u64; JOIN_STOP_COUNT] = std::array::from_fn(|i| JOIN_STOPS[i].swap(0, Relaxed));
        if stops.iter().any(|&count| count != 0) {
            let per_frame = |reason: JoinStop| stops[reason as usize] as f64 / frames;
            tracing::info!(
                "submission join stops per frame (VICTORY_JOIN_REASONS): {:.1} no queued candidate, \
                 {:.1} folded wait, {:.1} joining disabled, {:.1} other job, {:.1} value wait, \
                 {:.1} fence wait, {:.1} part limit, {:.1} list limit, {:.1} CPU copy, {:.1} earlier write",
                per_frame(JoinStop::QueueBoundary),
                per_frame(JoinStop::FoldedWait),
                per_frame(JoinStop::Disabled),
                per_frame(JoinStop::BoundaryJob),
                per_frame(JoinStop::ValueWait),
                per_frame(JoinStop::FenceWait),
                per_frame(JoinStop::PartLimit),
                per_frame(JoinStop::ListLimit),
                per_frame(JoinStop::CpuCopy),
                per_frame(JoinStop::EarlierWrite),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    type Log = Arc<Mutex<Vec<String>>>;

    /// A job of a test: what it stands at, and what it does.
    enum T {
        /// Stands until the fence has the value.
        Wait(usize, u64),
        Run(&'static str, Log),
        /// Runs once the gate is open: a submission that takes long.
        Slow(&'static str, Log, Arc<(Mutex<bool>, Condvar)>),
        Signal(usize, u64, Log),
        Present(&'static str, Log),
        /// Takes others of its kind, waits that are met, and signals along.
        Group(Vec<&'static str>, Log, Vec<(usize, u64)>),
    }

    impl Work for T {
        type Ask = (usize, u64);
        fn ask(&self) -> Option<(usize, u64)> {
            match self {
                &T::Wait(fence, value) => Some((fence, value)),
                _ => None,
            }
        }
        fn met(ask: &(usize, u64)) -> bool {
            crate::fence::completed(ask.0 as This) >= ask.1
        }
        fn shows(&self) -> bool {
            matches!(self, T::Present(..))
        }
        fn runs(&self) -> bool {
            matches!(self, T::Run(..) | T::Slow(..) | T::Group(..))
        }
        fn take_along(&mut self, next: T) -> Result<bool, T> {
            let T::Group(names, _, signals) = self else {
                return Err(next);
            };
            match next {
                T::Group(more, ..) if names.len() < 3 => {
                    names.extend(more);
                    Ok(true)
                }
                T::Wait(f, v) if crate::fence::completed(f as This) >= v || signals.contains(&(f, v)) => Ok(false),
                T::Signal(f, v, _) => {
                    signals.push((f, v));
                    Ok(false)
                }
                other => Err(other),
            }
        }
        fn go(self, forced: bool) {
            match self {
                T::Wait(..) => {
                    if forced {
                        FORCED_HERE.with(|f| f.set(f.get() + 1));
                    }
                }
                T::Run(name, log) | T::Present(name, log) => log.lock().unwrap().push(name.into()),
                T::Slow(name, log, gate) => {
                    let mut open = gate.0.lock().unwrap();
                    while !*open {
                        open = gate.1.wait(open).unwrap();
                    }
                    log.lock().unwrap().push(name.into());
                }
                T::Signal(fence, value, log) => {
                    log.lock().unwrap().push(format!("signal {value}"));
                    crate::fence::set(fence as This, value);
                }
                T::Group(names, log, signals) => {
                    log.lock().unwrap().push(names.join("+"));
                    for (fence, value) in signals {
                        crate::fence::set(fence as This, value);
                    }
                }
            }
        }
        fn mark_now() -> u64 {
            NOW.get()
        }
        fn mark_later() -> u64 {
            LATER.get()
        }
        fn marked(&mut self, mark: u64) {
            let name = match self {
                T::Run(name, _) | T::Slow(name, ..) => name.to_string(),
                T::Group(names, ..) => names.join("+"),
                _ => "?".into(),
            };
            MARKS.with(|m| m.borrow_mut().push((name, mark)));
        }
    }

    thread_local! {
        static FORCED_HERE: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
        /// What [`Work::mark_now`] and [`Work::mark_later`] give in a test.
        static NOW: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
        static LATER: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
        /// The marks submissions were done with, by name.
        static MARKS: std::cell::RefCell<Vec<(String, u64)>> = const { std::cell::RefCell::new(Vec::new()) };
    }

    /// Renderer port #7: a submission is done with the mark taken when it
    /// was put in the line, or, behind a wait of its queue, at least the one
    /// taken when that wait went (what is behind a wait may be written only
    /// before it is met); one taken along with others, the highest of all.
    #[test]
    fn a_submission_needs_what_was_written_before_it_and_its_queues_waits() {
        let lane = Lane::new();
        let log: Log = Default::default();
        let f = fence();
        let run = |name| T::Run(name, log.clone());
        NOW.set(5);
        lane.push(1, [run("a")]);
        NOW.set(6);
        lane.push(1, [T::Wait(f, 1), run("b")]);
        look(&lane);
        assert_eq!(MARKS.take(), [("a".to_string(), 5)], "b stands at its wait");
        // The wait is met later; what b reads may have been written just
        // before that.
        LATER.set(9);
        crate::fence::set(f as This, 1);
        look(&lane);
        assert_eq!(MARKS.take(), [("b".to_string(), 9)]);
        // Another queue went past no wait; this one did, for good.
        NOW.set(7);
        lane.push(2, [run("c")]);
        NOW.set(3);
        lane.push(1, [run("d")]);
        look(&lane);
        assert_eq!(MARKS.take(), [("c".to_string(), 7), ("d".to_string(), 9)]);
        // Taken along: a met wait between two parts raises them both.
        let part = |name| T::Group(vec![name], log.clone(), Vec::new());
        NOW.set(10);
        LATER.set(12);
        lane.push(3, [part("x"), T::Wait(f, 1), part("y")]);
        look(&lane);
        assert_eq!(MARKS.take(), [("x+y".to_string(), 12)]);
        // A present is no submission: it is told no mark.
        lane.push(3, [T::Present("present", log.clone())]);
        look(&lane);
        assert!(MARKS.take().is_empty());
        assert_eq!(*log.lock().unwrap(), ["a", "b", "c", "d", "x+y", "present"]);
    }

    fn fence() -> usize {
        let mut f = std::ptr::null();
        assert_eq!(crate::fence::create(std::ptr::null(), 0, 0, &crate::fence::FENCE_IIDS[0], &mut f), super::super::S_OK);
        f as usize
    }

    fn value(f: usize) -> u64 {
        crate::fence::completed(f as This)
    }

    fn look(lane: &Lane<T>) {
        let limit = lane.lock().next;
        lane.look(limit);
    }

    /// A queue's order is kept, a wait for another queue's signal holds only
    /// what is behind it on its own queue, and the oldest goes first.
    #[test]
    fn queues_keep_their_order_and_wait_for_each_others_fences() {
        let lane = Lane::new();
        let log: Log = Default::default();
        let f = fence();
        let run = |name| T::Run(name, log.clone());
        // The compute queue waits for what the graphics queue signals later.
        lane.push(2, [T::Wait(f, 1), run("compute")]);
        lane.push(1, [run("draw 1")]);
        lane.push_behind(1, false, T::Signal(f, 1, log.clone()), || panic!("the queue is not empty"));
        lane.push(1, [run("draw 2")]);
        look(&lane);
        assert_eq!(*log.lock().unwrap(), ["draw 1", "signal 1", "compute", "draw 2"]);
        assert_eq!(value(f), 1);
        // Nothing in the line: a signal is at once.
        let mut done = false;
        lane.push_behind(1, false, T::Signal(f, 2, log.clone()), || done = true);
        assert!(done);
    }

    /// A submission takes what follows it in its queue along, up to what it
    /// cannot take, which stays at its place; the counts come out even.
    #[test]
    fn submissions_take_those_behind_them_along() {
        let lane = Lane::new();
        let log: Log = Default::default();
        let f = fence();
        let part = |name| T::Group(vec![name], log.clone(), Vec::new());
        lane.push(1, [part("a"), T::Wait(f, 0), part("b")]);
        lane.push(2, [part("other queue"), T::Wait(f, 1), part("behind a's signal")]);
        lane.push(1, [T::Signal(f, 1, log.clone()), T::Wait(f, 1), part("c"), part("d")]);
        lane.push(1, [T::Wait(f, 9), part("held"), T::Present("present", log.clone()), part("next frame")]);
        // What came after the look was asked for is not touched.
        let limit = lane.lock().next;
        lane.push(1, [part("later")]);
        lane.look(limit);
        assert_eq!(
            *log.lock().unwrap(),
            ["a+b+c", "other queue+behind a's signal", "d", "held", "present", "next frame"],
            "{:?}",
            lane.lock().items.len()
        );
        look(&lane);
        assert_eq!(log.lock().unwrap().last().unwrap(), "later");
        let state = lane.lock();
        assert_eq!((state.presents, state.runs, state.items.len()), (0, 0, 0));
    }

    /// A wait that nothing meets holds its queue until a present, which
    /// comes behind everything given before it and before what comes after.
    #[test]
    fn a_present_goes_behind_everything_before_it_met_or_not() {
        let lane = Lane::new();
        let log: Log = Default::default();
        let f = fence();
        let run = |name| T::Run(name, log.clone());
        lane.push(2, [T::Wait(f, 9), run("held")]);
        lane.push(1, [run("draw")]);
        look(&lane);
        assert_eq!(*log.lock().unwrap(), ["draw"], "the wait holds its queue and no other");
        lane.push(1, [T::Present("present", log.clone())]);
        lane.push(1, [run("next frame")]);
        lane.push(2, [T::Wait(f, 10), run("held again")]);
        look(&lane);
        assert_eq!(*log.lock().unwrap(), ["draw", "held", "present", "next frame"]);
        assert_eq!(FORCED_HERE.get(), 1);
        // The second wait is met by a signal of the CPU.
        crate::fence::set(f as This, 10);
        look(&lane);
        assert_eq!(log.lock().unwrap().last().unwrap(), "held again");
        assert_eq!(FORCED_HERE.get(), 1);
        let state = lane.lock();
        assert_eq!((state.presents, state.runs, state.items.len()), (0, 0, 0));
    }

    /// With the thread: a fence behind a submission is not reached before the
    /// submission has been done, presents hold the title back a frame later,
    /// and a drain comes back when everything was done.
    #[test]
    fn the_title_goes_on_and_is_told_in_order() {
        let lane: &'static Lane<T> = Box::leak(Box::new(Lane::new()));
        std::thread::spawn(|| lane.serve());
        let log: Log = Default::default();
        let f = fence();
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        lane.push(1, [T::Slow("slow", log.clone(), gate.clone())]);
        lane.push_behind(1, false, T::Signal(f, 1, log.clone()), || panic!("behind the submission"));
        lane.push(1, [T::Present("present 1", log.clone())]);
        // One present may be in the line: this returns with it there.
        lane.wait_presents(1);
        assert_eq!(value(f), 0);
        assert!(log.lock().unwrap().is_empty());
        // A fence behind a submission being done (no longer in the line).
        while lane.lock().running.is_none() {
            std::thread::yield_now();
        }
        lane.push(1, [T::Run("second", log.clone())]);
        lane.push(1, [T::Present("present 2", log.clone())]);
        let opener = {
            let gate = gate.clone();
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(20));
                *gate.0.lock().unwrap() = true;
                gate.1.notify_all();
            })
        };
        lane.wait_presents(1);
        assert!(value(f) == 1, "the first frame was done");
        lane.drain();
        assert_eq!(*log.lock().unwrap(), ["slow", "signal 1", "present 1", "second", "present 2"]);
        opener.join().unwrap();
        // A wait the CPU meets without telling anybody; a drain does not
        // stand at it.
        lane.push(1, [T::Wait(f, 5), T::Run("after the wait", log.clone())]);
        lane.drain();
        assert_eq!(log.lock().unwrap().len(), 5);
        lane.wait_runs(0);
        crate::fence::set(f as This, 5);
        while log.lock().unwrap().len() < 6 {
            std::thread::yield_now();
        }
    }

    #[test]
    fn merge_stamp_preserves_unstamped_parts() {
        assert_eq!(merge_stamp(3, 5), 5);
        assert_eq!(merge_stamp(7, 2), 7);
        assert_eq!(merge_stamp(0, 5), 0);
        assert_eq!(merge_stamp(5, 0), 0);
        assert_eq!(merge_stamp(0, 0), 0);
    }
}
