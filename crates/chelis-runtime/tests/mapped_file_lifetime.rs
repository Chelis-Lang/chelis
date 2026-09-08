#![cfg(feature = "ownership-ledger")]

//! [05-OP-44] mapped files share the unified heap retain/release authority.
//!
//! The positive child balances a real mapping. The negative child presents a
//! live List allocation to the mapped-file retain entry, proving that the
//! common header is kind-checked rather than merely assumed from the pointer
//! type.

use std::ffi::CString;
use std::fs;
use std::process::{Command, Output};

use chelis_runtime::{
    chelis_list_empty, chelis_mapped_file, chelis_mapped_file_release, chelis_mapped_file_retain,
    chelis_mmap_file, chelis_mmap_len, chelis_string_from_cstr, chelis_string_release,
};

const CHILD_CASE: &str = "CHELIS_MAPPED_FILE_CHILD_CASE";
const LEDGER_PATH: &str = "CHELIS_OWNERSHIP_LEDGER_PATH";
const MAPPED_PATH: &str = "CHELIS_MAPPED_FILE_FIXTURE_PATH";

fn temp_path(label: &str, extension: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "chelis-mapped-lifetime-{}-{label}.{extension}",
        std::process::id()
    ))
}

fn run_child(case: &str, ledger: &std::path::Path, mapped: &std::path::Path) -> Output {
    let _ = fs::remove_file(ledger);
    Command::new(std::env::current_exe().expect("current test executable"))
        .args(["--exact", "mapped_file_contract_child", "--nocapture"])
        .env(CHILD_CASE, case)
        .env(LEDGER_PATH, ledger)
        .env(MAPPED_PATH, mapped)
        .output()
        .unwrap_or_else(|error| panic!("run mapped-file child `{case}`: {error}"))
}

#[test]
fn mapped_file_contract_child() {
    let Ok(case) = std::env::var(CHILD_CASE) else {
        return;
    };
    unsafe {
        match case.as_str() {
            "balanced" => {
                let path_text = CString::new(std::env::var(MAPPED_PATH).unwrap()).unwrap();
                let path = chelis_string_from_cstr(path_text.as_ptr());
                let mapped = chelis_mmap_file(path);
                chelis_string_release(path);
                assert_eq!(chelis_mmap_len(mapped), 4);
                chelis_mapped_file_retain(mapped);
                chelis_mapped_file_release(mapped);
                chelis_mapped_file_release(mapped);
            }
            "live-wrong-kind" => {
                let list = chelis_list_empty();
                chelis_mapped_file_retain(list.cast::<chelis_mapped_file>());
                panic!("mapped-file retain accepted a live List handle");
            }
            "null" => {
                chelis_mapped_file_retain(std::ptr::null());
                panic!("mapped-file retain accepted a null handle");
            }
            other => panic!("unknown mapped-file child case `{other}`"),
        }
    }
}

#[test]
fn mapped_file_retain_release_is_balanced() {
    let ledger_path = temp_path("balanced", "jsonl");
    let mapped_path = temp_path("fixture", "bin");
    fs::write(&mapped_path, b"data").expect("write mapped fixture");
    let output = run_child("balanced", &ledger_path, &mapped_path);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let ledger = fs::read_to_string(&ledger_path).expect("mapped-file ledger");
    for required in [
        r#""kind":"MappedFile""#,
        r#""event":"retain""#,
        r#""owners_before":1,"owners_after":2"#,
        r#""event":"finalize""#,
        r#""live_owners":0"#,
        r#""live_bytes":0"#,
        r#""invalid_operations":0"#,
    ] {
        assert!(ledger.contains(required), "missing {required:?}:\n{ledger}");
    }
    let _ = fs::remove_file(ledger_path);
    let _ = fs::remove_file(mapped_path);
}

#[test]
fn mapped_file_lifetime_rejects_null_and_live_wrong_kind_handles() {
    let mapped_path = temp_path("unused", "bin");
    for (case, reason) in [("null", "null"), ("live-wrong-kind", "kind")] {
        let ledger_path = temp_path(case, "jsonl");
        let output = run_child(case, &ledger_path, &mapped_path);
        assert!(
            !output.status.success(),
            "invalid mapped case `{case}` returned"
        );
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("Domain:"), "{case}: {stderr}");
        assert!(stderr.contains(reason), "{case}: {stderr}");
        let _ = fs::remove_file(ledger_path);
    }
}
