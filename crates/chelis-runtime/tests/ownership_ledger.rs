#![cfg(feature = "ownership-ledger")]

use std::ffi::CString;
use std::fs;
use std::process::Command;

use chelis_runtime::{
    chelis_alloc, chelis_string_from_cstr, chelis_string_release, chelis_string_retain,
    chelis_tensor_entry_borrow, chelis_tensor_release, CHELIS_DTYPE_F32,
};

const CHILD_MODE: &str = "CHELIS_OWNERSHIP_LEDGER_TEST_CHILD";
const LEDGER_PATH: &str = "CHELIS_OWNERSHIP_LEDGER_PATH";

fn child_path(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "chelis-ownership-ledger-{}-{name}.jsonl",
        std::process::id()
    ))
}

fn run_child(mode: &str, path: &std::path::Path) -> std::process::Output {
    let _ = fs::remove_file(path);
    Command::new(std::env::current_exe().expect("current test executable"))
        .args(["--exact", "ledger_process_probe", "--nocapture"])
        .env(CHILD_MODE, mode)
        .env(LEDGER_PATH, path)
        .output()
        .expect("spawn ownership-ledger child")
}

#[test]
fn ledger_process_probe() {
    let Ok(mode) = std::env::var(CHILD_MODE) else {
        return;
    };
    unsafe {
        match mode.as_str() {
            "balanced-string" => {
                let text = CString::new("hello").unwrap();
                let value = chelis_string_from_cstr(text.as_ptr());
                chelis_string_retain(value);
                chelis_string_release(value);
                chelis_string_release(value);
            }
            "tensor-bytes" => {
                let shape = [4_i64];
                let tensor = chelis_alloc(1, shape.as_ptr(), CHELIS_DTYPE_F32);
                chelis_tensor_release(tensor);
            }
            "borrowed-tensor-bytes" => {
                let shape = [4_i64];
                let data = [0.0_f32; 4];
                let tensor = chelis_tensor_entry_borrow(
                    1,
                    shape.as_ptr(),
                    CHELIS_DTYPE_F32,
                    data.as_ptr().cast(),
                    16,
                );
                chelis_tensor_release(tensor);
            }
            "invalid-release" => {
                let text = CString::new("bad").unwrap();
                let value = chelis_string_from_cstr(text.as_ptr());
                chelis_string_release(value);
                chelis_string_release(value);
            }
            other => panic!("unknown child mode {other}"),
        }
    }
}

#[test]
fn balanced_string_records_owner_transitions_and_summary() {
    let path = child_path("balanced-string");
    let output = run_child("balanced-string", &path);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let ledger = fs::read_to_string(&path).expect("ledger file");
    for required in [
        r#""schema":"compiled-value-ownership-ledger-v1""#,
        r#""event":"allocate""#,
        r#""kind":"String""#,
        r#""event":"retain""#,
        r#""owners_before":1,"owners_after":2"#,
        r#""event":"finalize""#,
        r#""event":"summary""#,
        r#""live_owners":0"#,
        r#""live_bytes":0"#,
        r#""invalid_operations":0"#,
    ] {
        assert!(ledger.contains(required), "missing {required:?}:\n{ledger}");
    }
    let _ = fs::remove_file(path);
}

#[test]
fn tensor_storage_uses_portable_payload_bytes() {
    let path = child_path("tensor-bytes");
    let output = run_child("tensor-bytes", &path);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let ledger = fs::read_to_string(&path).expect("ledger file");
    assert!(
        ledger.contains(r#""kind":"TensorStorage","bytes":16"#),
        "four f32 values own exactly sixteen logical bytes:\n{ledger}"
    );
    assert!(ledger.contains(r#""kind":"Tensor","bytes":0"#));
    let _ = fs::remove_file(path);

    // Negative parity: an entry borrow records the runtime-owned storage
    // wrapper, but never counts the caller's bytes as runtime ownership.
    let path = child_path("borrowed-tensor-bytes");
    let output = run_child("borrowed-tensor-bytes", &path);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let ledger = fs::read_to_string(&path).expect("ledger file");
    assert!(
        ledger.contains(r#""kind":"TensorStorage","bytes":0"#),
        "the runtime owns only the storage wrapper for an entry borrow:\n{ledger}"
    );
    assert!(ledger.contains(r#""event":"borrow""#), "{ledger}");
    assert!(ledger.contains(r#""live_bytes":0"#), "{ledger}");
    let _ = fs::remove_file(path);
}

#[test]
fn duplicate_release_records_invalid_operation_before_exit() {
    let path = child_path("invalid-release");
    let output = run_child("invalid-release", &path);
    assert!(!output.status.success(), "duplicate release must fail");
    let ledger = fs::read_to_string(&path).expect("ledger file survives failure");
    assert!(ledger.contains(r#""event":"invalid_release""#), "{ledger}");
    assert!(ledger.contains(r#""invalid_operations":1"#), "{ledger}");
    let _ = fs::remove_file(path);
}
