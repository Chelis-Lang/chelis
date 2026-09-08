#[path = "common/mod.rs"]
mod common;

use std::ffi::OsString;
use std::fs;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Duration;
use tempfile::tempdir;

#[test]
fn absent_shared_root_keeps_the_local_fixture_isolated() {
    assert_eq!(
        common::resolve_configured_shared_reef_home(None).unwrap(),
        None
    );
}

#[test]
fn configured_shared_root_must_be_absolute() {
    let error =
        common::resolve_configured_shared_reef_home(Some(OsString::from("relative/reef-home")))
            .unwrap_err();
    assert!(error.contains("absolute"), "{error}");
}

#[test]
fn concurrent_initializers_prepare_one_root_exactly_once() {
    let dir = tempdir().unwrap();
    let root = dir.path().join("reef-home");
    let starts = Arc::new(AtomicUsize::new(0));
    let barrier = Arc::new(Barrier::new(8));
    let mut workers = Vec::new();
    for _ in 0..8 {
        let root = root.clone();
        let starts = starts.clone();
        let barrier = barrier.clone();
        workers.push(thread::spawn(move || {
            barrier.wait();
            common::ensure_job_shared_reef_with(&root, |reef_home| {
                starts.fetch_add(1, Ordering::SeqCst);
                thread::sleep(Duration::from_millis(50));
                fs::write(reef_home.join("prepared"), b"yes").map_err(|error| error.to_string())
            })
        }));
    }
    for worker in workers {
        worker.join().unwrap().unwrap();
    }
    assert_eq!(starts.load(Ordering::SeqCst), 1);
    assert_eq!(fs::read(root.join("prepared")).unwrap(), b"yes");
}

#[test]
fn invalid_ready_sentinel_fails_loudly() {
    let dir = tempdir().unwrap();
    let root = dir.path().join("reef-home");
    fs::create_dir_all(&root).unwrap();
    fs::write(root.join(common::SHARED_REEF_READY_FILE), b"wrong contract").unwrap();
    let error = common::ensure_job_shared_reef_with(&root, |_| {
        panic!("invalid sentinel must not run the initializer")
    })
    .unwrap_err();
    assert!(error.contains("sentinel"), "{error}");
}

#[test]
fn partial_root_without_a_sentinel_fails_loudly() {
    let dir = tempdir().unwrap();
    let root = dir.path().join("reef-home");
    fs::create_dir_all(&root).unwrap();
    fs::write(root.join("partial"), b"stale").unwrap();
    let error =
        common::ensure_job_shared_reef_with(&root, |_| panic!("partial root must not be blessed"))
            .unwrap_err();
    assert!(error.contains("not empty"), "{error}");
}
