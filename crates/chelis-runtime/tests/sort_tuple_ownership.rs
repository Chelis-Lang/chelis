#![cfg(feature = "ownership-ledger")]

//! chelis#3032: `chelis_tensor_sort` hands its `(values, indices)` tuple the
//! only owner of each tensor it allocates.
//!
//! The sort allocated both result tensors with one owner each and then built
//! the tuple through `chelis_tuple_from_values`, which clones its borrowed
//! items. Each tensor ended with two owners, the tuple released one, and a
//! compiled program leaked both tensors of every sort result.
//!
//! The positive cases release the tuple, every component read from it, and
//! the input, and require the ledger to end with no live owner, for a
//! populated and an empty operand (the empty one returns early). The negative
//! case leaves the tuple unreleased and requires the ledger to report exactly
//! the tuple and its two tensors live, each with one owner, so the oracle can
//! neither pass by losing track of the allocations nor miss a second owner.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use chelis_runtime::{
    chelis_alloc, chelis_tensor, chelis_tensor_begin_write, chelis_tensor_end_write,
    chelis_tensor_release, chelis_tensor_sort, chelis_tensor_write_view, chelis_tuple_get,
    chelis_tuple_release, chelis_value_release, CHELIS_DTYPE_I64,
};

const CHILD_CASE: &str = "CHELIS_SORT_TUPLE_CHILD_CASE";
const LEDGER_PATH: &str = "CHELIS_OWNERSHIP_LEDGER_PATH";

fn ledger_path(case: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "chelis-sort-tuple-{}-{case}.jsonl",
        std::process::id()
    ))
}

fn run_child(case: &str, ledger: &Path) -> Output {
    let _ = fs::remove_file(ledger);
    Command::new(std::env::current_exe().expect("current test executable"))
        .args(["--exact", "sort_tuple_contract_child", "--nocapture"])
        .env(CHILD_CASE, case)
        .env(LEDGER_PATH, ledger)
        .output()
        .unwrap_or_else(|error| panic!("run sort child `{case}`: {error}"))
}

/// The summary line's `live_owners` and `live_bytes`.
fn live_counts(case: &str, ledger: &str) -> (u64, u64) {
    let summary = ledger
        .lines()
        .find(|line| line.contains(r#""event":"summary""#))
        .unwrap_or_else(|| panic!("{case}: no summary in ledger:\n{ledger}"));
    assert!(
        summary.contains(r#""invalid_operations":0"#),
        "{case}: {summary}"
    );
    let field = |name: &str| -> u64 {
        let key = format!("\"{name}\":");
        let start = summary.find(&key).expect(name) + key.len();
        summary[start..]
            .chars()
            .take_while(char::is_ascii_digit)
            .collect::<String>()
            .parse()
            .expect(name)
    };
    (field("live_owners"), field("live_bytes"))
}

unsafe fn i64_tensor(values: &[i64]) -> *mut chelis_tensor {
    let shape = [values.len() as i64];
    let tensor = chelis_alloc(1, shape.as_ptr(), CHELIS_DTYPE_I64);
    let guard = chelis_tensor_begin_write(tensor);
    let view = chelis_tensor_write_view(guard);
    view.data
        .cast::<i64>()
        .copy_from(values.as_ptr(), values.len());
    chelis_tensor_end_write(guard);
    tensor
}

#[test]
fn sort_tuple_contract_child() {
    let Ok(case) = std::env::var(CHILD_CASE) else {
        return;
    };
    unsafe {
        let input = match case.as_str() {
            "populated-released" | "tuple-unreleased" => i64_tensor(&[3, 1, 2]),
            "empty-released" => i64_tensor(&[]),
            other => panic!("unknown sort child case `{other}`"),
        };
        let result = chelis_tensor_sort(input, 0);
        let values = chelis_tuple_get(result, 0);
        let indices = chelis_tuple_get(result, 1);
        chelis_value_release(values);
        chelis_value_release(indices);
        if case != "tuple-unreleased" {
            chelis_tuple_release(result);
        }
        chelis_tensor_release(input);
    }
}

#[test]
fn a_released_sort_result_leaves_no_live_owner() {
    for case in ["populated-released", "empty-released"] {
        let path = ledger_path(case);
        let output = run_child(case, &path);
        assert!(
            output.status.success(),
            "{case}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let ledger = fs::read_to_string(&path).expect("sort ledger");
        assert!(ledger.contains(r#""kind":"Tuple""#), "{case}: {ledger}");
        assert_eq!(live_counts(case, &ledger), (0, 0), "{case}: {ledger}");
        let _ = fs::remove_file(path);
    }
}

#[test]
fn an_unreleased_sort_tuple_keeps_one_owner_of_each_tensor_live() {
    let case = "tuple-unreleased";
    let path = ledger_path(case);
    let output = run_child(case, &path);
    assert!(
        output.status.success(),
        "{case}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let ledger = fs::read_to_string(&path).expect("sort ledger");
    let (live_owners, live_bytes) = live_counts(case, &ledger);
    // The tuple and one owner each of the two result tensors and their
    // storage; never the input.
    assert_eq!(live_owners, 5, "{case}: {ledger}");
    assert!(live_bytes > 0, "{case}: {ledger}");
    let _ = fs::remove_file(path);
}
