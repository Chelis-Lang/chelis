//! chelis#2508: the consuming accumulator entry points.
//!
//! `chelis_list_push_moved` takes the caller's owner of its value and
//! `chelis_list_extend_moved` takes the caller's owner of its source list:
//! the extend retains every source item into the destination, then releases
//! the one owner the caller gave up. It has no move-when-unique arm, so a
//! source some other holder still owns keeps its items and loses exactly one
//! owner. The in-place push accepts only a list with exactly one owner, which
//! is how these tests read an owner count through the public ABI; its refusal
//! terminates the process, so the negative control runs in a child.

use std::ffi::CStr;
use std::process::Command;

use chelis_runtime::{
    chelis_list_empty, chelis_list_extend_moved, chelis_list_index, chelis_list_len,
    chelis_list_push_moved, chelis_list_release, chelis_list_retain, chelis_string_borrow_value,
    chelis_string_data, chelis_string_from_cstr, chelis_value, chelis_value_release,
    chelis_value_take_string,
};

const CHILD_CASE_ENV: &str = "CHELIS_LIST_MOVED_ACCUMULATOR_CHILD_CASE";

unsafe fn text(value: &CStr) -> chelis_value {
    chelis_value_take_string(chelis_string_from_cstr(value.as_ptr()))
}

unsafe fn text_at(list: *const chelis_runtime::chelis_list, index: i64) -> String {
    let item = chelis_list_index(list, index);
    let rendered = CStr::from_ptr(chelis_string_data(chelis_string_borrow_value(item)))
        .to_string_lossy()
        .into_owned();
    chelis_value_release(item);
    rendered
}

/// A two-string source list with one owner.
unsafe fn source() -> *mut chelis_runtime::chelis_list {
    let src = chelis_list_empty();
    chelis_list_push_moved(src, text(c"left"));
    chelis_list_push_moved(src, text(c"right"));
    src
}

#[test]
fn extend_moved_leaves_a_shared_source_intact_with_one_owner_fewer() {
    unsafe {
        let src = source();
        // The second holder: the source must survive the extend for it.
        chelis_list_retain(src);
        let dst = chelis_list_empty();
        chelis_list_extend_moved(dst, src);
        chelis_list_release(dst);

        assert_eq!(chelis_list_len(src), 2, "the shared source keeps its items");
        assert_eq!(text_at(src, 0), "left");
        assert_eq!(text_at(src, 1), "right");
        // Exactly one owner left: the in-place push accepts it. A released
        // second owner would already have freed the list; an unreleased one
        // would make the push refuse.
        chelis_list_push_moved(src, text(c"third"));
        assert_eq!(chelis_list_len(src), 3);
        chelis_list_release(src);
    }
}

#[test]
fn extend_moved_shares_the_items_with_the_destination() {
    unsafe {
        let src = source();
        chelis_list_retain(src);
        let dst = chelis_list_empty();
        chelis_list_extend_moved(dst, src);
        // Releasing the source's last owner must not free the items the
        // destination retained.
        chelis_list_release(src);
        assert_eq!(chelis_list_len(dst), 2);
        assert_eq!(text_at(dst, 0), "left");
        assert_eq!(text_at(dst, 1), "right");
        chelis_list_release(dst);
    }
}

#[test]
fn push_moved_takes_the_owner_of_its_value() {
    unsafe {
        let list = chelis_list_empty();
        let item = text(c"only");
        chelis_list_push_moved(list, item);
        assert_eq!(text_at(list, 0), "only");
        // No release of `item` is owed; the list's finalizer releases it.
        chelis_list_release(list);
    }
}

/// The child half of the negative control: a source the extend did not
/// consume still has its second owner, so the probe push must refuse it.
#[test]
fn list_moved_accumulator_child() {
    let Ok(case) = std::env::var(CHILD_CASE_ENV) else {
        return;
    };
    assert_eq!(case, "unconsumed");
    unsafe {
        let src = source();
        chelis_list_retain(src);
        chelis_list_push_moved(src, text(c"third"));
    }
    println!("probe accepted a shared list");
}

#[test]
fn the_owner_probe_refuses_a_list_with_two_owners() {
    let output = Command::new(std::env::current_exe().expect("current test binary"))
        .args(["--exact", "list_moved_accumulator_child", "--nocapture"])
        .env(CHILD_CASE_ENV, "unconsumed")
        .output()
        .expect("run child");
    assert!(!output.status.success(), "the probe accepted a shared list");
    assert!(String::from_utf8_lossy(&output.stderr)
        .contains("chelis_list_push_moved requires exclusive ownership (refcount 1)"));
}

/// Under the ownership ledger the shared-source case returns every
/// allocation: the extend released exactly the owner it was given.
#[cfg(feature = "ownership-ledger")]
#[test]
fn a_shared_source_extend_balances_under_the_ledger() {
    let ledger = std::env::temp_dir().join(format!(
        "chelis-list-moved-accumulator-{}.jsonl",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&ledger);
    let output = Command::new(std::env::current_exe().expect("current test binary"))
        .args([
            "--exact",
            "extend_moved_leaves_a_shared_source_intact_with_one_owner_fewer",
            "--nocapture",
        ])
        .env("CHELIS_OWNERSHIP_LEDGER_PATH", &ledger)
        .output()
        .expect("run ledger child");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let rows = std::fs::read_to_string(&ledger).expect("ledger written");
    let summary = rows.lines().last().expect("summary row");
    for field in [
        r#""live_owners":0"#,
        r#""live_bytes":0"#,
        r#""invalid_operations":0"#,
    ] {
        assert!(summary.contains(field), "{summary}");
    }
    assert!(rows.contains(r#""kind":"String""#), "{summary}");
    let _ = std::fs::remove_file(ledger);
}
