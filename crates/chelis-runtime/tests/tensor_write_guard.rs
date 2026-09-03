#![cfg(feature = "ownership-ledger")]

//! [05-OP-44] guarded tensor writes require unique runtime-owned storage.
//!
//! The positive child begins one guard, obtains the typed write view, ends the
//! guard, and then observes the write through a read view. The negative matrix
//! covers shared descriptors, caller-owned entry storage, and every descriptor
//! operation forbidden while a guard is active.

use std::fs;
use std::process::{Command, Output};

use chelis_runtime::{
    chelis_alloc, chelis_tensor_begin_write, chelis_tensor_end_write, chelis_tensor_entry_borrow,
    chelis_tensor_numel, chelis_tensor_rank, chelis_tensor_read_view, chelis_tensor_release,
    chelis_tensor_retain, chelis_tensor_shape, chelis_tensor_write_view, CHELIS_DTYPE_F32,
};

const CHILD_CASE: &str = "CHELIS_TENSOR_WRITE_CHILD_CASE";
const LEDGER_PATH: &str = "CHELIS_OWNERSHIP_LEDGER_PATH";

fn ledger_path(case: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "chelis-tensor-write-{}-{case}.jsonl",
        std::process::id()
    ))
}

fn run_child(case: &str, ledger: &std::path::Path) -> Output {
    let _ = fs::remove_file(ledger);
    Command::new(std::env::current_exe().expect("current test executable"))
        .args(["--exact", "tensor_write_contract_child", "--nocapture"])
        .env(CHILD_CASE, case)
        .env(LEDGER_PATH, ledger)
        .output()
        .unwrap_or_else(|error| panic!("run tensor-write child `{case}`: {error}"))
}

#[test]
fn tensor_write_contract_child() {
    let Ok(case) = std::env::var(CHILD_CASE) else {
        return;
    };
    unsafe {
        let shape = [2_i64];
        match case.as_str() {
            "balanced" => {
                let tensor = chelis_alloc(1, shape.as_ptr(), CHELIS_DTYPE_F32);
                let guard = chelis_tensor_begin_write(tensor);
                let view = chelis_tensor_write_view(guard);
                assert_eq!(view.count, 2);
                assert_eq!(view.dtype, CHELIS_DTYPE_F32);
                assert!(!view.data.is_null());
                *(view.data as *mut f32) = 3.5;
                assert_eq!(chelis_tensor_rank(tensor), 1);
                assert_eq!(chelis_tensor_shape(tensor, 0), 2);
                assert_eq!(chelis_tensor_numel(tensor), 2);
                chelis_tensor_end_write(guard);

                let read = chelis_tensor_read_view(tensor);
                assert_eq!(read.count, 2);
                assert_eq!(read.dtype, CHELIS_DTYPE_F32);
                assert_eq!(*(read.data as *const f32), 3.5);
                chelis_tensor_release(tensor);
            }
            "shared" => {
                let tensor = chelis_alloc(1, shape.as_ptr(), CHELIS_DTYPE_F32);
                chelis_tensor_retain(tensor);
                let _ = chelis_tensor_begin_write(tensor);
                panic!("begin_write accepted a shared descriptor");
            }
            "entry-borrow" => {
                let data = [0.0_f32; 2];
                let tensor = chelis_tensor_entry_borrow(
                    1,
                    shape.as_ptr(),
                    CHELIS_DTYPE_F32,
                    data.as_ptr().cast(),
                    8,
                );
                let _ = chelis_tensor_begin_write(tensor);
                panic!("begin_write accepted caller-owned storage");
            }
            "second-begin" => {
                let tensor = chelis_alloc(1, shape.as_ptr(), CHELIS_DTYPE_F32);
                let _guard = chelis_tensor_begin_write(tensor);
                let _ = chelis_tensor_begin_write(tensor);
                panic!("begin_write accepted a second live guard");
            }
            "read-during-write" => {
                let tensor = chelis_alloc(1, shape.as_ptr(), CHELIS_DTYPE_F32);
                let _guard = chelis_tensor_begin_write(tensor);
                let _ = chelis_tensor_read_view(tensor);
                panic!("read_view accepted a descriptor under a write guard");
            }
            "retain-during-write" => {
                let tensor = chelis_alloc(1, shape.as_ptr(), CHELIS_DTYPE_F32);
                let _guard = chelis_tensor_begin_write(tensor);
                chelis_tensor_retain(tensor);
                panic!("retain accepted a descriptor under a write guard");
            }
            "release-during-write" => {
                let tensor = chelis_alloc(1, shape.as_ptr(), CHELIS_DTYPE_F32);
                let _guard = chelis_tensor_begin_write(tensor);
                chelis_tensor_release(tensor);
                panic!("release accepted a descriptor under a write guard");
            }
            "view-after-end" => {
                let tensor = chelis_alloc(1, shape.as_ptr(), CHELIS_DTYPE_F32);
                let guard = chelis_tensor_begin_write(tensor);
                chelis_tensor_end_write(guard);
                let _ = chelis_tensor_write_view(guard);
                panic!("write_view accepted an ended guard");
            }
            "second-end" => {
                let tensor = chelis_alloc(1, shape.as_ptr(), CHELIS_DTYPE_F32);
                let guard = chelis_tensor_begin_write(tensor);
                chelis_tensor_end_write(guard);
                chelis_tensor_end_write(guard);
                panic!("end_write accepted an ended guard");
            }
            other => panic!("unknown tensor-write child case `{other}`"),
        }
    }
}

#[test]
fn unique_runtime_owned_tensor_accepts_balanced_write_guard() {
    let path = ledger_path("balanced");
    let output = run_child("balanced", &path);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let ledger = fs::read_to_string(&path).expect("tensor write ledger");
    assert!(ledger.contains(r#""kind":"Tensor""#), "{ledger}");
    assert!(ledger.contains(r#""kind":"TensorStorage""#), "{ledger}");
    assert!(ledger.contains(r#""live_owners":0"#), "{ledger}");
    assert!(ledger.contains(r#""live_bytes":0"#), "{ledger}");
    assert!(ledger.contains(r#""invalid_operations":0"#), "{ledger}");
    let _ = fs::remove_file(path);
}

#[test]
fn tensor_write_guard_refuses_nonunique_or_caller_owned_storage() {
    for (case, reason) in [("shared", "unique"), ("entry-borrow", "runtime-owned")] {
        let path = ledger_path(case);
        let output = run_child(case, &path);
        assert!(
            !output.status.success(),
            "invalid write case `{case}` returned"
        );
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("Domain:"), "{case}: {stderr}");
        assert!(stderr.contains(reason), "{case}: {stderr}");
        let _ = fs::remove_file(path);
    }
}

#[test]
fn live_write_guard_blocks_every_other_descriptor_operation() {
    for case in [
        "second-begin",
        "read-during-write",
        "retain-during-write",
        "release-during-write",
    ] {
        let path = ledger_path(case);
        let output = run_child(case, &path);
        assert!(
            !output.status.success(),
            "guarded operation `{case}` returned"
        );
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("Domain:"), "{case}: {stderr}");
        assert!(stderr.contains("write guard"), "{case}: {stderr}");
        let _ = fs::remove_file(path);
    }
}

#[test]
fn ended_write_guard_has_no_view_and_cannot_be_consumed_twice() {
    for case in ["view-after-end", "second-end"] {
        let path = ledger_path(case);
        let output = run_child(case, &path);
        assert!(
            !output.status.success(),
            "ended guard case `{case}` returned"
        );
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("Domain:"), "{case}: {stderr}");
        assert!(stderr.contains("has ended"), "{case}: {stderr}");
        let _ = fs::remove_file(path);
    }
}
