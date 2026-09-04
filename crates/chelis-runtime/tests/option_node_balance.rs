#![cfg(feature = "ownership-ledger")]

//! [05-OP-44] option nodes own zero or one cloned tagged child.
//!
//! The positive child checks nested option construction, unwrap cloning, and
//! final child release. The negative children exercise `None` unwrap and an
//! invalid tagged child, so an implementation cannot pass by returning a
//! zero/default carrier.

use std::fs;
use std::process::{Command, Output};

use chelis_runtime::{
    chelis_option_is_some, chelis_option_none, chelis_option_release, chelis_option_some,
    chelis_option_unwrap, chelis_scalar_from_bits, chelis_value, chelis_value_box_scalar,
    chelis_value_payload, chelis_value_release, chelis_value_take_list, chelis_value_take_option,
    CHELIS_DTYPE_I64, CHELIS_VALUE_LIST,
};

const CHILD_CASE: &str = "CHELIS_OPTION_NODE_CHILD_CASE";
const LEDGER_PATH: &str = "CHELIS_OWNERSHIP_LEDGER_PATH";

fn ledger_path(case: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "chelis-option-node-{}-{case}.jsonl",
        std::process::id()
    ))
}

fn run_child(case: &str, ledger: &std::path::Path) -> Output {
    let _ = fs::remove_file(ledger);
    Command::new(std::env::current_exe().expect("current test executable"))
        .args(["--exact", "option_node_contract_child", "--nocapture"])
        .env(CHILD_CASE, case)
        .env(LEDGER_PATH, ledger)
        .output()
        .unwrap_or_else(|error| panic!("run option child `{case}`: {error}"))
}

unsafe fn scalar_value(value: i64) -> chelis_value {
    chelis_value_box_scalar(chelis_scalar_from_bits(
        CHELIS_DTYPE_I64,
        u64::from_ne_bytes(value.to_ne_bytes()),
    ))
}

#[test]
fn option_node_contract_child() {
    let Ok(case) = std::env::var(CHILD_CASE) else {
        return;
    };
    unsafe {
        match case.as_str() {
            "balanced-nested" => {
                let inner = chelis_option_some(scalar_value(7));
                assert!(chelis_option_is_some(inner));
                let inner_value = chelis_value_take_option(inner);
                let outer = chelis_option_some(inner_value);
                chelis_value_release(inner_value);

                let unwrapped = chelis_option_unwrap(outer);
                chelis_value_release(unwrapped);
                chelis_option_release(outer);

                let none = chelis_option_none();
                assert!(!chelis_option_is_some(none));
                chelis_option_release(none);
            }
            "unwrap-none" => {
                let none = chelis_option_none();
                let _ = chelis_option_unwrap(none);
                panic!("unwrap returned a value for None");
            }
            "invalid-child" => {
                let mut payload: chelis_value_payload = std::mem::zeroed();
                payload.handle = std::ptr::null_mut();
                let invalid = chelis_value {
                    tag: CHELIS_VALUE_LIST,
                    reserved: [0; 7],
                    payload,
                };
                let _ = chelis_option_some(invalid);
                panic!("option constructor accepted an invalid tagged child");
            }
            "child-clone" => {
                let list = chelis_value_take_list(chelis_runtime::chelis_list_empty());
                let option = chelis_option_some(list);
                chelis_value_release(list);
                let child = chelis_option_unwrap(option);
                chelis_value_release(child);
                chelis_option_release(option);
            }
            other => panic!("unknown option child case `{other}`"),
        }
    }
}

#[test]
fn option_nodes_clone_once_and_release_once_at_each_edge() {
    for case in ["balanced-nested", "child-clone"] {
        let path = ledger_path(case);
        let output = run_child(case, &path);
        assert!(
            output.status.success(),
            "{case}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let ledger = fs::read_to_string(&path).expect("option ledger");
        assert!(ledger.contains(r#""kind":"Option""#), "{case}: {ledger}");
        assert!(ledger.contains(r#""event":"finalize""#), "{case}: {ledger}");
        assert!(ledger.contains(r#""live_owners":0"#), "{case}: {ledger}");
        assert!(ledger.contains(r#""live_bytes":0"#), "{case}: {ledger}");
        assert!(
            ledger.contains(r#""invalid_operations":0"#),
            "{case}: {ledger}"
        );
        let _ = fs::remove_file(path);
    }
}

#[test]
fn option_nodes_reject_none_unwrap_and_invalid_children() {
    for (case, reason) in [("unwrap-none", "None"), ("invalid-child", "null")] {
        let path = ledger_path(case);
        let output = run_child(case, &path);
        assert!(
            !output.status.success(),
            "invalid option case `{case}` returned"
        );
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("Domain:"), "{case}: {stderr}");
        assert!(stderr.contains(reason), "{case}: {stderr}");
        let _ = fs::remove_file(path);
    }
}
