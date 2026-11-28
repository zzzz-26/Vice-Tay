//! `victory warm`: host pipelines made ahead of a run (`known_pipelines`), in
//! processes of their own: a compiler that crashes or hangs costs one pipeline.

use std::collections::HashSet;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::{Mutex, OnceLock, mpsc};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use victory_hle::known_pipelines as known;

/// Pipelines a process makes before another takes its place: the compiler's
/// and the translations' memory goes with it.
const AT_ONCE: usize = 40;
/// A pipeline that takes longer is taken for one the compiler hangs on.
const LONGEST: Duration = Duration::from_secs(180);

/// As the runs count one that takes long.
const SLOW_MS: f64 = 1000.0;

fn folder() -> Result<PathBuf> {
    let cfg = victory_core::Config::load()?;
    Ok(cfg.cache_subdir("shaders")?.with_file_name("pipelines"))
}

fn bad_keys<'a>(text: &'a str, fingerprint: &str) -> HashSet<&'a str> {
    let hex = |word: &str, len| word.len() == len && word.bytes().all(|b| b.is_ascii_hexdigit());
    text.lines()
        .filter_map(|line| {
            let mut words = line.split_whitespace();
            let first = words.next()?;
            let second = words.next();
            let (key, recorded) = match second {
                // Accept journals written by builds that put the fingerprint first.
                Some(key) if hex(first, 16) && hex(key, 32) => (key, Some(first)),
                _ => (first, second.and_then(|word| word.strip_prefix("fp=")).filter(|fp| hex(fp, 16))),
            };
            (hex(key, 32) && recorded.is_none_or(|fp| fp.eq_ignore_ascii_case(fingerprint))).then_some(key)
        })
        .collect()
}

fn completed_keys<'a>(text: &'a str, fingerprint: &str) -> HashSet<&'a str> {
    text.lines()
        .filter_map(|line| {
            // Older journals also counted refusals as completed.
            let (recorded, key) = line.strip_prefix("v2 ")?.split_once(' ')?;
            (recorded == fingerprint).then_some(key)
        })
        .collect()
}

fn record_completion(out: &mut impl Write, fingerprint: &str, key: &str, there: bool) -> Result<()> {
    if there {
        writeln!(out, "v2 {fingerprint} {key}").context("writing warm-done.txt")?;
    }
    Ok(())
}

pub fn run(
    jobs: Option<usize>,
    batch_size: Option<u16>,
    normal_priority: bool,
    limit: Option<usize>,
    again: bool,
    used: bool,
    these: Option<String>,
) -> Result<()> {
    let dir = folder()?;
    if let Some(these) = these {
        return make(&dir, &these);
    }
    if !dir.try_exists().with_context(|| format!("checking {}", dir.display()))? {
        println!("0 pipelines known. Run the title with VICTORY_PIPELINE_BOOK=1 to record pipelines before warming.");
        return Ok(());
    }
    let began = Instant::now();
    victory_hle::known_pipelines::use_translations(dir.with_file_name("shaders"));
    let fingerprint = known::fingerprint();
    let done_file = dir.join("warm-done.txt");
    let bad_file = dir.join("warm-bad.txt");
    let entries = known::warm_entries(&dir);
    let listed = std::fs::read_to_string(&done_file).unwrap_or_default();
    let done = completed_keys(&listed, &fingerprint);
    let listed_bad = if again || std::env::var_os("VICTORY_WARM_BAD").is_some_and(|v| v == "0") {
        String::new()
    } else {
        std::fs::read_to_string(&bad_file).unwrap_or_default()
    };
    let bad = bad_keys(&listed_bad, &fingerprint);
    let early = std::fs::read_to_string(dir.join("early-pipelines.txt")).unwrap_or_default();
    let early: HashSet<&str> = early.lines().collect();
    // Those earlier runs drew with first, the others in no order (by key).
    let mut all: Vec<usize> = (0..entries.len())
        .filter(|&n| {
            let entry = &entries[n];
            let key = entry.key.as_str();
            !entry.skipped && (again || !done.contains(key)) && !bad.contains(key) && (!used || early.contains(key))
        })
        .collect();
    all.sort_by_key(|&n| (!early.contains(entries[n].key.as_str()), &entries[n].key));
    all.truncate(limit.unwrap_or(usize::MAX));
    let cores = std::thread::available_parallelism().map_or(4, |n| n.get());
    let jobs = jobs.unwrap_or(cores * 2 / 3).max(1);
    let batch_size = batch_size.map_or(AT_ONCE, usize::from);
    let skipped = entries.iter().filter(|entry| entry.skipped).count();
    println!(
        "{} pipelines known, {} made before ({fingerprint}), {} bad, {skipped} skipped by VICTORY_SKIP_PIPELINES. {} to make, {jobs} processes",
        entries.len(),
        done.len(),
        bad.len(),
        all.len()
    );
    let state = Mutex::new((Vec::<f64>::new(), 0usize, 0usize));
    let done_out = Mutex::new(std::fs::OpenOptions::new().create(true).append(true).open(&done_file)?);
    // (Read back by the runs: these are made beside the others, not before.)
    let slow_out = Mutex::new(std::fs::OpenOptions::new().create(true).append(true).open(dir.join("slow-pipelines.txt"))?);
    let bad_out = Mutex::new(std::fs::OpenOptions::new().create(true).append(true).open(&bad_file)?);
    run_batches(
        &all,
        jobs,
        batch_size,
        |indices| start_worker(indices, &entries, normal_priority),
        |n, ms, there| {
            if ms >= SLOW_MS {
                writeln!(lock(&slow_out), "\n{} {ms:.0}", entries[n].key).context("writing slow-pipelines.txt")?;
            }
            let mut state = lock(&state);
            state.0.push(ms);
            state.1 += usize::from(!there);
            let count = state.0.len();
            if count.is_multiple_of(1000) {
                println!("{count} of {}, {:.0} s", all.len(), began.elapsed().as_secs_f64());
            }
            drop(state);
            record_completion(&mut *lock(&done_out), &fingerprint, &entries[n].key, there)
        },
        |n, why| {
            println!("pipeline {n} {}: {why}", entries[n].key);
            // Keep the key in column one for older readers of this journal.
            writeln!(lock(&bad_out), "{} fp={fingerprint} {why}", entries[n].key).context("writing warm-bad.txt")?;
            lock(&state).2 += 1;
            Ok(())
        },
    )?;
    let (mut times, refused, crashed) = state.into_inner().unwrap_or_else(|e| e.into_inner());
    times.sort_by(f64::total_cmp);
    let at = |q: f64| times.get((times.len() as f64 * q) as usize).copied().unwrap_or(0.0);
    println!(
        "{} made in {:.0} s ({refused} have no host pipeline, {crashed} ended their process); ms each: median {:.1}, p90 {:.1}, p99 {:.1}, most {:.0}, sum {:.0} s",
        times.len(),
        began.elapsed().as_secs_f64(),
        at(0.5),
        at(0.9),
        at(0.99),
        times.last().copied().unwrap_or(0.0),
        times.iter().sum::<f64>() / 1e3
    );
    Ok(())
}

fn run_batches(
    all: &[usize],
    jobs: usize,
    batch_size: usize,
    start: impl Fn(&[usize]) -> Result<Child> + Sync,
    each: impl Fn(usize, f64, bool) -> Result<()> + Sync,
    bad: impl Fn(usize, String) -> Result<()> + Sync,
) -> Result<()> {
    let chunks: Vec<&[usize]> = all.chunks(batch_size).collect();
    let next = AtomicUsize::new(0);
    // Stop launching workers after a setup failure. In-flight workers can still report.
    let failure = OnceLock::new();
    std::thread::scope(|s| {
        for _ in 0..jobs.min(chunks.len()) {
            s.spawn(|| {
                while let Some(&chunk) = chunks.get(next.fetch_add(1, Ordering::Relaxed)) {
                    let mut rest = chunk;
                    while !rest.is_empty() {
                        if failure.get().is_some() {
                            return;
                        }
                        let outcome = start(rest)
                            .context("could not start warm worker")
                            .and_then(|process| child(process, rest, &each));
                        match outcome {
                            Ok(Some((made, why))) => {
                                // child only returns a failure index after a validated start.
                                let (n, after) = rest[made..].split_first().expect("active pipeline");
                                if let Err(error) = bad(*n, why) {
                                    let _ = failure.set(error);
                                    return;
                                }
                                rest = after;
                            }
                            Ok(None) => break,
                            Err(error) => {
                                let _ = failure.set(error);
                                return;
                            }
                        }
                    }
                }
            });
        }
    });
    failure.into_inner().map_or(Ok(()), Err)
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn start_worker(these: &[usize], entries: &[known::WarmEntry], normal_priority: bool) -> Result<Child> {
    use std::os::windows::process::CommandExt;
    const IDLE_PRIORITY_CLASS: u32 = 0x40;
    const NORMAL_PRIORITY_CLASS: u32 = 0x20;
    let exe = std::env::current_exe().context("could not locate the warm worker executable")?;
    let list: Vec<String> = these.iter().map(|&index| {
        let recipe = &entries[index];
        format!("{index}:{}:{}", recipe.offset, recipe.key)
    }).collect();
    Command::new(&exe)
        .args(["warm", "--these", &list.join(",")])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .creation_flags(if normal_priority { NORMAL_PRIORITY_CLASS } else { IDLE_PRIORITY_CLASS })
        .spawn()
        .with_context(|| format!("could not spawn {}", exe.display()))
}

/// The worker reports a start and a completion for each requested index, in order.
struct Progress<'a> {
    expected: &'a [usize],
    made: usize,
    active: bool,
}

impl<'a> Progress<'a> {
    fn new(expected: &'a [usize]) -> Self {
        Self { expected, made: 0, active: false }
    }

    fn line(&mut self, line: &str) -> Result<Option<(usize, f64, bool)>> {
        let mut words = line.split_whitespace();
        match words.next() {
            Some("at") => {
                let n: usize = words.next().context("missing pipeline index")?.parse()?;
                anyhow::ensure!(
                    !self.active && self.expected.get(self.made) == Some(&n) && words.next().is_none(),
                    "unexpected pipeline start: {line}"
                );
                self.active = true;
                Ok(None)
            }
            Some("made") => {
                let n: usize = words.next().context("missing pipeline index")?.parse()?;
                let ms: f64 = words.next().context("missing pipeline duration")?.parse()?;
                let there = match words.next() {
                    Some("0") => false,
                    Some("1") => true,
                    _ => anyhow::bail!("invalid pipeline result: {line}"),
                };
                anyhow::ensure!(
                    self.active && self.expected.get(self.made) == Some(&n)
                        && ms.is_finite() && ms >= 0.0 && words.next().is_none(),
                    "unexpected pipeline completion: {line}"
                );
                self.made += 1;
                self.active = false;
                Ok(Some((n, ms, there)))
            }
            _ => Ok(None),
        }
    }

    fn finish(self, status: ExitStatus, timed_out: bool) -> Result<Option<(usize, String)>> {
        if status.success() && !timed_out {
            anyhow::ensure!(
                self.made == self.expected.len() && !self.active,
                "warm worker exited successfully after reporting only {} of {} pipelines",
                self.made, self.expected.len()
            );
            return Ok(None);
        }
        let why = if timed_out {
            format!("hung for {} s", LONGEST.as_secs())
        } else {
            status.to_string()
        };
        // Retain the existing compiler-hang policy, but only after a valid start.
        // For process exits, require a specific DXGI device-loss HRESULT, not
        // just a negative exit code (which includes unrelated NTSTATUS errors).
        if self.active && (timed_out || device_loss(status)) {
            return Ok(Some((self.made, why)));
        }
        anyhow::bail!("warm worker failed after {} of {} pipelines: {why}", self.made, self.expected.len())
    }
}

fn device_loss(status: ExitStatus) -> bool {
    // DXGI_ERROR_DEVICE_REMOVED, DXGI_ERROR_DEVICE_HUNG, DXGI_ERROR_DEVICE_RESET.
    matches!(status.code().map(|code| code as u32), Some(0x887A0005..=0x887A0007))
}

/// Only return a bad pipeline for a failure attributable to an active pipeline.
fn child(
    mut process: Child,
    these: &[usize],
    mut each: impl FnMut(usize, f64, bool) -> Result<()>,
) -> Result<Option<(usize, String)>> {
    let Some(out) = process.stdout.take() else {
        let _ = process.kill();
        process.wait().context("warm worker stdout unavailable, failed to reap worker")?;
        anyhow::bail!("warm worker stdout unavailable");
    };
    let process = Mutex::new(process);
    let heard = Mutex::new(Instant::now());
    let hung = AtomicBool::new(false);
    let mut progress = Progress::new(these);
    let status = std::thread::scope(|s| -> Result<ExitStatus> {
        let (finished, wake) = mpsc::channel::<()>();
        let (watch_heard, watch_hung, watch_process) = (&heard, &hung, &process);
        s.spawn(move || {
            while let Err(mpsc::RecvTimeoutError::Timeout) = wake.recv_timeout(Duration::from_millis(500)) {
                if lock(watch_heard).elapsed() > LONGEST {
                    let mut worker = lock(watch_process);
                    if matches!(worker.try_wait(), Ok(None)) && worker.kill().is_ok() {
                        watch_hung.store(true, Ordering::Relaxed);
                    }
                    return;
                }
            }
        });
        let result = (|| -> Result<ExitStatus> {
            for line in BufReader::new(out).lines() {
                let line = line.context("reading warm worker output")?;
                if let Some((n, ms, there)) = progress.line(&line)? {
                    each(n, ms, there)?;
                }
                *lock(&heard) = Instant::now();
            }
            // Keep the watchdog alive even if stdout closes before the process exits.
            loop {
                if let Some(status) = lock(&process).try_wait().context("waiting for warm worker")? {
                    return Ok(status);
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        })();
        if result.is_err() {
            let mut worker = lock(&process);
            let _ = worker.kill();
            let _ = worker.wait();
        }
        drop(finished);
        result
    })?;
    progress.finish(status, hung.load(Ordering::Relaxed))
}

/// What such a process does.
fn make(dir: &Path, these: &str) -> Result<()> {
    let these = these.split(',').map(|request| {
        let (index, details) = request.split_once(':').context("missing recipe offset in --these")?;
        let (offset, key) = details.split_once(':').context("missing recipe key in --these")?;
        Ok(known::WarmRecipe { index: index.parse()?, offset: offset.parse()?, key })
    }).collect::<Result<Vec<_>>>()?;
    let cfg = victory_core::Config::load()?;
    victory_hle::known_pipelines::use_translations(cfg.cache_subdir("shaders")?);
    let mut out = std::io::stdout();
    known::make(dir, &these, |n, came| {
        let told = match came {
            None => writeln!(out, "at {n}"),
            Some((ms, there)) => writeln!(out, "made {n} {ms:.2} {}", u8::from(there)),
        };
        if told.and_then(|()| out.flush()).is_err() {
            // Whoever asked is gone.
            std::process::exit(1);
        }
    }).context("reading the selected pipeline recipes")?;
    // Without what a run's end does (nothing to save, the host's threads).
    let _ = out.flush();
    std::process::exit(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::windows::process::CommandExt;

    #[test]
    fn warm_completion_keeps_refusals_and_ambiguous_legacy_records_pending() {
        let mut journal = b"fingerprint legacy\nv2 earlier old\n".to_vec();
        record_completion(&mut journal, "fingerprint", "refused", false).unwrap();
        record_completion(&mut journal, "fingerprint", "compiled", true).unwrap();
        assert_eq!(
            completed_keys(std::str::from_utf8(&journal).unwrap(), "fingerprint"),
            HashSet::from(["compiled"])
        );
    }

    #[test]
    fn quarantine_expires_with_the_fingerprint_and_keeps_legacy_keys() {
        let [current, stale, legacy, imported] = ['1', '2', '3', '4'].map(|c| c.to_string().repeat(32));
        let fp = "1234567890abcdef";
        let other = "ffffffffffffffff";
        let listed = format!(
            "# ignored\n\n{current} fp={fp} timed out\n{stale} fp={other}\n{legacy} manual entry\n{fp}\t{imported}\n"
        );
        assert_eq!(bad_keys(&listed, fp), HashSet::from([current.as_str(), legacy.as_str(), imported.as_str()]));
        assert_eq!(bad_keys(&listed, other), HashSet::from([stale.as_str(), legacy.as_str()]));
    }

    fn fake_child(script: &str, stdout: Stdio) -> Result<Child> {
        let cmd = PathBuf::from(std::env::var_os("SystemRoot").unwrap()).join(r"System32\cmd.exe");
        Ok(Command::new(cmd)
            .args(["/d", "/c"])
            .raw_arg(script)
            .stdin(Stdio::null())
            .stdout(stdout)
            .stderr(Stdio::null())
            .spawn()?)
    }

    #[test]
    fn missing_executable_leaves_the_batch_pending() {
        let missing = std::env::temp_dir().join(format!("victory-missing-worker-{}.exe", std::process::id()));
        assert!(!missing.try_exists().unwrap());
        let starts = AtomicUsize::new(0);
        let done = Mutex::new(Vec::new());
        let bad = Mutex::new(Vec::new());
        let result = run_batches(
            &[7, 8, 9],
            1,
            AT_ONCE,
            |_| {
                starts.fetch_add(1, Ordering::Relaxed);
                Ok(Command::new(&missing).spawn()?)
            },
            |n, _, _| { lock(&done).push(n); Ok(()) },
            |n, _| { lock(&bad).push(n); Ok(()) },
        );
        assert_eq!(starts.load(Ordering::Relaxed), 1);
        assert!(lock(&done).is_empty());
        assert!(lock(&bad).is_empty());
        let error = result.unwrap_err();
        assert!(error.to_string().contains("start warm worker"));
        assert_eq!(error.downcast_ref::<std::io::Error>().unwrap().kind(), std::io::ErrorKind::NotFound);
    }

    #[test]
    fn missing_stdout_is_a_startup_failure() {
        let error = run_batches(
            &[1, 2],
            1,
            AT_ONCE,
            |_| fake_child("exit /b 0", Stdio::null()),
            |_, _, _| panic!("unexpected completion"),
            |_, _| panic!("unexpected bad pipeline"),
        )
        .unwrap_err();
        assert!(format!("{error:#}").contains("stdout"));
    }

    #[test]
    fn device_loss_records_only_the_active_pipeline_and_resumes() {
        let done = Mutex::new(Vec::new());
        let bad = Mutex::new(Vec::new());
        run_batches(
            &[7, 8, 9],
            1,
            AT_ONCE,
            |rest| match rest {
                [7, 8, 9] => fake_child("echo at 7&echo made 7 3.5 1&echo at 8&exit /b -2005270523", Stdio::piped()),
                [9] => fake_child("echo at 9&echo made 9 2.0 0", Stdio::piped()),
                _ => panic!("unexpected batch {rest:?}"),
            },
            |n, ms, there| { lock(&done).push((n, ms, there)); Ok(()) },
            |n, why| { lock(&bad).push((n, why)); Ok(()) },
        )
        .unwrap();
        assert_eq!(*lock(&done), [(7, 3.5, true), (9, 2.0, false)]);
        let bad = lock(&bad);
        assert_eq!(bad.len(), 1);
        assert_eq!(bad[0].0, 8);
        assert!(!bad[0].1.is_empty());
    }

    #[test]
    fn batches_follow_the_requested_size() {
        let sizes = Mutex::new(Vec::new());
        run_batches(
            &[1, 2, 3],
            1,
            2,
            |batch| {
                lock(&sizes).push(batch.len());
                let lines: Vec<_> = batch.iter().map(|key| format!("echo at {key}&echo made {key} 1 1")).collect();
                fake_child(&lines.join("&"), Stdio::piped())
            },
            |_, _, _| Ok(()),
            |_, _| panic!("unexpected bad pipeline"),
        )
        .unwrap();
        assert_eq!(*lock(&sizes), [2, 1]);
    }

    #[test]
    fn startup_failure_preserves_another_workers_completion() {
        // Hold the failed start until its peer has reported the only key in its chunk.
        let keys: Vec<_> = (0..=AT_ONCE).collect();
        let (confirmed, received) = std::sync::mpsc::channel();
        let received = Mutex::new(received);
        let error = run_batches(
            &keys,
            2,
            AT_ONCE,
            |chunk| {
                if chunk[0] == 0 {
                    assert_eq!(lock(&received).recv_timeout(Duration::from_secs(5)).unwrap(), AT_ONCE);
                    anyhow::bail!("worker unavailable")
                }
                fake_child(&format!("echo at {AT_ONCE}&echo made {AT_ONCE} 1 1"), Stdio::piped())
            },
            |n, _, _| { confirmed.send(n).unwrap(); Ok(()) },
            |_, _| panic!("unexpected bad pipeline"),
        )
        .unwrap_err();
        assert!(format!("{error:#}").contains("worker unavailable"));
    }

    #[test]
    fn generic_exit_leaves_active_pipeline_pending() {
        for code in [7, -1073741510, -1073741819] {
            let done = Mutex::new(Vec::new());
            let error = run_batches(
                &[7, 8, 9],
                1,
                AT_ONCE,
                |_| fake_child(&format!("echo at 7&echo made 7 3.5 1&echo at 8&exit /b {code}"), Stdio::piped()),
                |n, _, _| { lock(&done).push(n); Ok(()) },
                |_, _| panic!("generic failures must not blacklist a pipeline"),
            ).unwrap_err();
            assert!(error.to_string().contains("after 1 of 3 pipelines"));
            assert_eq!(*lock(&done), [7]);
        }
    }

    #[test]
    fn incomplete_success_leaves_unreported_pipelines_pending() {
        for (script, completed) in [
            ("exit /b 0", vec![]),
            ("echo at 7&echo made 7 1 1&exit /b 0", vec![7]),
            ("echo at 7&exit /b 0", vec![]),
        ] {
            let done = Mutex::new(Vec::new());
            let error = run_batches(
                &[7, 8], 1, AT_ONCE,
                |_| fake_child(script, Stdio::piped()),
                |n, _, _| { lock(&done).push(n); Ok(()) },
                |_, _| panic!("incomplete success must not blacklist a pipeline"),
            ).unwrap_err();
            assert!(error.to_string().contains("exited successfully after reporting only"));
            assert_eq!(*lock(&done), completed);
        }
    }

    #[test]
    fn device_loss_and_timeout_require_an_active_pipeline() {
        use std::os::windows::process::ExitStatusExt;
        for code in [0x887A0005, 0x887A0006, 0x887A0007] {
            let status = ExitStatus::from_raw(code);
            assert!(Progress::new(&[7]).finish(status, false).is_err());
            let mut progress = Progress::new(&[7]);
            assert_eq!(progress.line("at 7").unwrap(), None);
            assert_eq!(progress.finish(status, false).unwrap().unwrap().0, 0);
        }
        let status = ExitStatus::from_raw(1);
        assert!(Progress::new(&[7]).finish(status, true).is_err());
        let mut progress = Progress::new(&[7]);
        assert_eq!(progress.line("at 7").unwrap(), None);
        assert_eq!(progress.finish(status, true).unwrap().unwrap().0, 0);
        let mut finished = Progress::new(&[7]);
        assert_eq!(finished.line("at 7").unwrap(), None);
        assert!(finished.line("made 7 0 1").unwrap().is_some());
        assert!(finished.finish(ExitStatus::from_raw(0x887A0005), false).is_err());
    }

    #[test]
    fn malformed_reports_do_not_complete_or_blacklist_pipelines() {
        for script in [
            "echo at 8",
            "echo made 7 1 1",
            "echo at 7&echo at 7",
            "echo at 7&echo made 8 1 1",
            "echo at 7&echo made 7 NaN 1",
            "echo at 7&echo made 7 1 2",
        ] {
            assert!(run_batches(
                &[7, 8], 1, AT_ONCE,
                |_| fake_child(script, Stdio::piped()),
                |_, _, _| panic!("invalid report"),
                |_, _| panic!("invalid report"),
            ).is_err());
        }
    }

    #[test]
    fn journal_failure_reaches_the_caller() {
        let error = run_batches(
            &[7, 8], 1, AT_ONCE,
            |_| fake_child("echo at 7&echo made 7 1 1", Stdio::piped()),
            |_, _, _| anyhow::bail!("journal unavailable"),
            |_, _| panic!("journal error must not blacklist a pipeline"),
        ).unwrap_err();
        assert_eq!(error.to_string(), "journal unavailable");
    }
}
