#![cfg(feature = "ownership-ledger")]

//! [05-OP-44] exact heap-kind and tagged clone/release contract.
//!
//! The positive child constructs every public heap kind, clones its tagged
//! owner, and releases both owners. Tensor construction also observes the one
//! private `TensorStorage` kind. The paired negative children prove that clone
//! validates null and live wrong-kind payloads rather than accepting a
//! wildcard/default arm.

use std::collections::BTreeSet;
use std::ffi::CString;
use std::fs;
use std::process::{Command, Output};

use chelis_runtime::{
    chelis_adt_construct, chelis_dict_from_pairs, chelis_list_empty, chelis_list_release,
    chelis_mmap_file, chelis_option_none, chelis_string_from_cstr, chelis_string_release,
    chelis_tensor, chelis_tuple_from_values, chelis_value, chelis_value_clone,
    chelis_value_payload, chelis_value_release, chelis_value_take_adt, chelis_value_take_dict,
    chelis_value_take_list, chelis_value_take_mapped_file, chelis_value_take_option,
    chelis_value_take_string, chelis_value_take_tensor, chelis_value_take_tuple, CHELIS_DTYPE_F32,
    CHELIS_VALUE_LIST,
};

const CHILD_CASE: &str = "CHELIS_HEAP_KIND_CHILD_CASE";
const LEDGER_PATH: &str = "CHELIS_OWNERSHIP_LEDGER_PATH";
const MAPPED_PATH: &str = "CHELIS_HEAP_KIND_MAPPED_PATH";

fn temp_path(label: &str, extension: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "chelis-heap-kind-{}-{label}.{extension}",
        std::process::id()
    ))
}

fn run_child(case: &str, ledger: &std::path::Path, mapped: &std::path::Path) -> Output {
    let _ = fs::remove_file(ledger);
    Command::new(std::env::current_exe().expect("current test executable"))
        .args(["--exact", "heap_kind_contract_child", "--nocapture"])
        .env(CHILD_CASE, case)
        .env(LEDGER_PATH, ledger)
        .env(MAPPED_PATH, mapped)
        .output()
        .unwrap_or_else(|error| panic!("run heap-kind child `{case}`: {error}"))
}

fn event_kinds(ledger: &str, event: &str) -> BTreeSet<String> {
    let event_marker = format!(r#""event":"{event}""#);
    ledger
        .lines()
        .filter(|line| line.contains(&event_marker))
        .filter_map(|line| {
            line.split_once(r#""kind":""#)
                .and_then(|(_, suffix)| suffix.split_once('"'))
                .map(|(kind, _)| kind.to_owned())
        })
        .collect()
}

unsafe fn all_public_kind_values(mapped_path: &str) -> Vec<chelis_value> {
    let string = CString::new("heap-kind-string").unwrap();
    let string_value = chelis_value_take_string(chelis_string_from_cstr(string.as_ptr()));

    let shape = [2_i64];
    let tensor_value = chelis_value_take_tensor(chelis_runtime::chelis_alloc(
        1,
        shape.as_ptr(),
        CHELIS_DTYPE_F32,
    ));

    let list_value = chelis_value_take_list(chelis_list_empty());
    let tuple_value = chelis_value_take_tuple(chelis_tuple_from_values(std::ptr::null(), 0));

    let pairs = chelis_list_empty();
    let dict = chelis_dict_from_pairs(pairs);
    chelis_list_release(pairs);
    let dict_value = chelis_value_take_dict(dict);

    let ctor_text = CString::new("HeapKindProbe").unwrap();
    let ctor = chelis_string_from_cstr(ctor_text.as_ptr());
    let adt = chelis_adt_construct(ctor, std::ptr::null(), 0);
    chelis_string_release(ctor);
    let adt_value = chelis_value_take_adt(adt);

    let option_value = chelis_value_take_option(chelis_option_none());

    let path_text = CString::new(mapped_path).unwrap();
    let path = chelis_string_from_cstr(path_text.as_ptr());
    let mapped = chelis_mmap_file(path);
    chelis_string_release(path);
    let mapped_value = chelis_value_take_mapped_file(mapped);

    vec![
        string_value,
        tensor_value,
        list_value,
        tuple_value,
        dict_value,
        adt_value,
        option_value,
        mapped_value,
    ]
}

#[test]
fn heap_kind_contract_child() {
    let Ok(case) = std::env::var(CHILD_CASE) else {
        return;
    };
    unsafe {
        match case.as_str() {
            "all-kinds" => {
                let mapped = std::env::var(MAPPED_PATH).expect("mapped fixture path");
                for value in all_public_kind_values(&mapped) {
                    let clone = chelis_value_clone(value);
                    chelis_value_release(value);
                    chelis_value_release(clone);
                }
            }
            "null-heap-value" => {
                let mut payload: chelis_value_payload = std::mem::zeroed();
                payload.handle = std::ptr::null_mut();
                let value = chelis_value {
                    tag: CHELIS_VALUE_LIST,
                    reserved: [0; 7],
                    payload,
                };
                let _ = chelis_value_clone(value);
                panic!("clone accepted a null heap handle");
            }
            "live-wrong-kind" => {
                let shape = [1_i64];
                let tensor: *mut chelis_tensor =
                    chelis_runtime::chelis_alloc(1, shape.as_ptr(), CHELIS_DTYPE_F32);
                let mut value = chelis_value_take_tensor(tensor);
                value.tag = CHELIS_VALUE_LIST;
                let _ = chelis_value_clone(value);
                panic!("clone accepted a live Tensor handle tagged as List");
            }
            other => panic!("unknown heap-kind child case `{other}`"),
        }
    }
}

#[test]
fn heap_kind_universe_and_clone_release_are_exhaustive() {
    let ledger_path = temp_path("all", "jsonl");
    let mapped_path = temp_path("mapped", "bin");
    fs::write(&mapped_path, b"mapped-kind").expect("write mapped fixture");
    let output = run_child("all-kinds", &ledger_path, &mapped_path);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let ledger = fs::read_to_string(&ledger_path).expect("heap-kind ledger");
    let expected_all = BTreeSet::from([
        "Adt".to_owned(),
        "Dict".to_owned(),
        "List".to_owned(),
        "MappedFile".to_owned(),
        "Option".to_owned(),
        "String".to_owned(),
        "Tensor".to_owned(),
        "TensorStorage".to_owned(),
        "Tuple".to_owned(),
    ]);
    assert_eq!(event_kinds(&ledger, "allocate"), expected_all);
    assert_eq!(event_kinds(&ledger, "finalize"), expected_all);

    let expected_public = BTreeSet::from([
        "Adt".to_owned(),
        "Dict".to_owned(),
        "List".to_owned(),
        "MappedFile".to_owned(),
        "Option".to_owned(),
        "String".to_owned(),
        "Tensor".to_owned(),
        "Tuple".to_owned(),
    ]);
    assert_eq!(event_kinds(&ledger, "retain"), expected_public);
    assert!(ledger.contains(r#""live_owners":0"#), "{ledger}");
    assert!(ledger.contains(r#""live_bytes":0"#), "{ledger}");
    assert!(ledger.contains(r#""invalid_operations":0"#), "{ledger}");

    let _ = fs::remove_file(ledger_path);
    let _ = fs::remove_file(mapped_path);
}

#[test]
fn clone_rejects_null_and_live_wrong_kind_values() {
    let mapped_path = temp_path("unused", "bin");
    for (case, reason) in [("null-heap-value", "null"), ("live-wrong-kind", "kind")] {
        let ledger_path = temp_path(case, "jsonl");
        let output = run_child(case, &ledger_path, &mapped_path);
        assert!(!output.status.success(), "invalid `{case}` clone returned");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("Domain:"), "{case}: {stderr}");
        assert!(stderr.contains(reason), "{case}: {stderr}");
        let _ = fs::remove_file(ledger_path);
    }
}
