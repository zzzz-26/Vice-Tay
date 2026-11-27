use std::path::Path;
use std::process::{Command, Output};

const MODES: &[&[&str]] = &[
    &["translate"],
    &["translate", "--warm", "all"],
    &["shaders"],
    #[cfg(windows)]
    &["warm"],
];

fn run(args: &[&str], jobs: &[&str], missing_config: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_victory"))
        .args(args)
        .args(jobs)
        .env("VICTORY_CONFIG", missing_config)
        .env_remove("VICTORY_AUDIO_CAPTURE")
        .env_remove("VICTORY_EMIT_DUMP")
        .output()
        .unwrap()
}

#[test]
fn zero_workers_are_rejected_before_dispatch() {
    let dir = tempfile::tempdir().unwrap();
    let missing_config = dir.path().join("missing.toml");
    for args in MODES {
        for jobs in [&["--jobs", "0"][..], &["--jobs=0"][..]] {
            let output = run(args, jobs, &missing_config);
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert_eq!(output.status.code(), Some(2), "{args:?} {jobs:?}: {stderr}");
            assert!(
                stderr.contains("expected a worker count of 1 or more"),
                "{args:?} {jobs:?}: {stderr}"
            );
        }
    }
}

#[test]
fn omitted_and_positive_worker_counts_reach_dispatch() {
    let dir = tempfile::tempdir().unwrap();
    let missing_config = dir.path().join("missing.toml");
    for args in MODES {
        for jobs in [&[][..], &["--jobs", "1"][..], &["--jobs", "3"][..]] {
            let output = run(args, jobs, &missing_config);
            let stderr = String::from_utf8_lossy(&output.stderr);
            // Config fails inside the command, before any file scan or GPU probe.
            assert_eq!(output.status.code(), Some(1), "{args:?} {jobs:?}: {stderr}");
            assert!(
                stderr.contains("VICTORY_CONFIG points at"),
                "{args:?} {jobs:?}: {stderr}"
            );
        }
    }
}
