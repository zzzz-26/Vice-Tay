//! Work for a list that a thread of ours does while the title records on:
//! the pass over the heap's buffer views (`crate::draw`), which took 12 to 17
//! ms of one recording thread at the frame's first list that needs it, with
//! the submission of that list, and of those behind it, waiting.
//! `VICTORY_HEAP_PASS_ASIDE=0`: done where it is needed, as before.
//!
//! What it finds are refreshes of host copies, which are steps of the list
//! for its submission, and what it makes are descriptors, which the card
//! reads when the list runs: both are there when the list is closed
//! (`HostList::close` waits), and the refreshes stand at the place in the
//! list's steps where the pass was begun.

use super::*;

/// A host copy, and the title memory of it (from, to).
pub(crate) type Refresh = (std::sync::Arc<crate::shadow::Shadow>, u64, u64);

/// What is being found, until it is.
pub(crate) struct Coming(Mutex<Option<Vec<Refresh>>>, std::sync::Condvar);

/// Since the last frame line: microseconds lists waited at their close.
pub(crate) static WAITED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

impl Coming {
    /// What was found.
    pub(crate) fn wait(&self) -> Vec<Refresh> {
        let began = std::time::Instant::now();
        let _waiting = crate::timeline::span("a list's close waits for the heap's views");
        let mut found = self.0.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if let Some(found) = found.take() {
                WAITED.fetch_add(began.elapsed().as_micros() as u64, Ordering::Relaxed);
                return found;
            }
            found = self.1.wait(found).unwrap_or_else(|e| e.into_inner());
        }
    }

    fn found(&self, what: Vec<Refresh>) {
        *self.0.lock().unwrap_or_else(|e| e.into_inner()) = Some(what);
        self.1.notify_all();
    }
}

pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_HEAP_PASS_ASIDE").is_none_or(|v| v != "0"))
}

type Job = Box<dyn FnOnce() + Send>;

// Queue occupancy is distinct from lists waiting for already-completed work.
// Counters are cumulative except queued/active; diagnostics can difference
// samples without perturbing scheduling or introducing producer blocking.
static QUEUE: [std::sync::atomic::AtomicU64; 7] = [const { std::sync::atomic::AtomicU64::new(0) }; 7];

#[cfg(feature = "diagnostics")]
pub(crate) fn stats() -> [u64; 7] { std::array::from_fn(|i| QUEUE[i].load(Ordering::Relaxed)) }

/// `work` begins on the thread for it, one after another; here and now if
/// there is no such thread.
pub(crate) fn begin(work: impl FnOnce() -> Vec<Refresh> + Send + 'static) -> std::sync::Arc<Coming> {
    static THREAD: OnceLock<Option<Mutex<std::sync::mpsc::Sender<Job>>>> = OnceLock::new();
    let coming = std::sync::Arc::new(Coming(Mutex::new(None), std::sync::Condvar::new()));
    let told = coming.clone();
    let queued_at = std::time::Instant::now();
    let queued = QUEUE[0].fetch_add(1, Ordering::Relaxed) + 1;
    QUEUE[1].fetch_max(queued, Ordering::Relaxed);
    let job: Job = Box::new(move || {
        QUEUE[0].fetch_sub(1, Ordering::Relaxed);
        QUEUE[2].fetch_add(1, Ordering::Relaxed);
        let wait = queued_at.elapsed().as_micros() as u64;
        QUEUE[3].fetch_add(wait, Ordering::Relaxed);
        QUEUE[4].fetch_max(wait, Ordering::Relaxed);
        let started = std::time::Instant::now();
        // (However it ends, the list that waits goes on.)
        let _aside = crate::timeline::span("beside: the heap's views");
        let found = std::panic::catch_unwind(std::panic::AssertUnwindSafe(work)).unwrap_or_default();
        QUEUE[5].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
        QUEUE[6].fetch_add(1, Ordering::Relaxed);
        QUEUE[2].fetch_sub(1, Ordering::Relaxed);
        told.found(found);
    });
    let thread = THREAD.get_or_init(|| {
        let (send, jobs) = std::sync::mpsc::channel::<Job>();
        std::thread::Builder::new()
            .name("victory-aside".into())
            .spawn(move || {
                crate::cores::keep_on_fast_cores("victory-aside");
                // (The submitting thread waits for its pass at the lists' close.)
                let mut priority = crate::cores::Priority::new();
                jobs.into_iter().for_each(|job| {
                    priority.keep();
                    job()
                })
            })
            .ok()
            .map(|_| Mutex::new(send))
    });
    match thread {
        Some(send) => {
            // Never execute fallback work while holding the sender lock.
            let sent = send.lock().unwrap_or_else(|e| e.into_inner()).send(job);
            if let Err(job) = sent {
                (job.0)();
            }
        }
        None => job(),
    }
    coming
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn what_is_found_is_waited_for_and_a_panic_lets_the_list_go_on() {
        let (go, wait) = std::sync::mpsc::channel::<()>();
        let coming = begin(move || {
            let _ = wait.recv();
            Vec::new()
        });
        assert!(coming.0.lock().unwrap().is_none());
        go.send(()).unwrap();
        assert!(coming.wait().is_empty());
        assert!(begin(|| panic!("a panic of the work")).wait().is_empty());
        assert!(begin(Vec::new).wait().is_empty());
    }
}
