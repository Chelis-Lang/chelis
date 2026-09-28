//! chelis#2334: the consuming list skip.
//!
//! `chelis_list_drop_owned` takes ownership of its list operand. A
//! uniquely owned list releases its leading elements and advances a
//! private offset, so it comes back as the same allocation and the call
//! costs the count rather than the length; a shared one is left
//! untouched and a fresh list is returned exactly as the cloning
//! `chelis_list_drop` builds it. The static half of the contract (no
//! un-retained reference survives) is the ownership verifier's; this
//! file locks the runtime half.
//!
//! The offset is what makes the element release worth testing
//! separately from the value. The cloning path never releases anything:
//! it retains the suffix and leaves the caller's list alone. The
//! consuming path owes exactly one release per element it retires, and
//! getting that wrong is invisible to every length and value assertion
//! here, which is why the ledger rows below count owners rather than
//! reading the list back.
//!
//! Evidentiary status: REGRESSION TEST for every row that names
//! `chelis_list_drop_owned`, which does not exist before this change
//! set. `skip_owned_agrees_with_the_cloning_skip` is additionally a LOCK
//! on the cloning entry point's results, which are unchanged.
use std::process::Command;

use chelis_runtime::{
    chelis_list_append_owned, chelis_list_borrow_value, chelis_list_concat_owned, chelis_list_drop,
    chelis_list_drop_owned, chelis_list_empty, chelis_list_index, chelis_list_len,
    chelis_list_push_moved, chelis_list_release, chelis_list_retain, chelis_scalar_from_bits,
    chelis_value, chelis_value_box_scalar, chelis_value_release, chelis_value_take_list,
    chelis_value_unbox_scalar, CHELIS_DTYPE_I64,
};

const CHILD_CASE_ENV: &str = "CHELIS_LIST_SKIP_OWNED_CHILD_CASE";

unsafe fn int_value(value: i64) -> chelis_value {
    chelis_value_box_scalar(chelis_scalar_from_bits(
        CHELIS_DTYPE_I64,
        u64::from_ne_bytes(value.to_ne_bytes()),
    ))
}

unsafe fn int_at(list: *const chelis_runtime::chelis_list, index: i64) -> i64 {
    let value = chelis_list_index(list, index);
    let bits = chelis_value_unbox_scalar(value).bits;
    chelis_value_release(value);
    i64::from_ne_bytes(bits.to_ne_bytes())
}

unsafe fn contents(list: *const chelis_runtime::chelis_list) -> Vec<i64> {
    (0..chelis_list_len(list))
        .map(|i| int_at(list, i))
        .collect()
}

unsafe fn list_of(values: &[i64]) -> *mut chelis_runtime::chelis_list {
    let list = chelis_list_empty();
    for &value in values {
        chelis_list_push_moved(list, int_value(value));
    }
    list
}

#[test]
fn unique_skip_owned_advances_in_place_and_returns_the_same_list() {
    unsafe {
        let list = list_of(&[1, 2, 3, 4]);
        let result = chelis_list_drop_owned(list, 2);
        assert!(
            std::ptr::eq(result, list),
            "a uniquely owned list is skipped in place"
        );
        assert_eq!(contents(result), vec![3, 4]);
        chelis_list_release(result);
    }
}

/// The offset is cumulative and every read still starts at the live
/// window. A `live()` that forgot the head would report four here.
#[test]
fn repeated_skips_compose_and_leave_the_list_readable() {
    unsafe {
        let list = list_of(&[10, 20, 30, 40, 50, 60]);
        let mut cursor = list;
        for expected_head in [20, 30, 40, 50, 60] {
            cursor = chelis_list_drop_owned(cursor, 1);
            assert_eq!(int_at(cursor, 0), expected_head);
        }
        assert_eq!(chelis_list_len(cursor), 1);
        cursor = chelis_list_drop_owned(cursor, 1);
        assert_eq!(chelis_list_len(cursor), 0, "a walked-out cursor is empty");
        chelis_list_release(cursor);
    }
}

#[test]
fn shared_skip_owned_clones_and_leaves_the_shared_view_untouched() {
    unsafe {
        let list = list_of(&[1, 2, 3]);
        // A second owner: the observable source list of the alias controls.
        chelis_list_retain(list);
        let result = chelis_list_drop_owned(list, 1);
        assert!(
            !std::ptr::eq(result, list),
            "a shared list is never skipped in place"
        );
        assert_eq!(contents(list), vec![1, 2, 3], "the shared view is intact");
        assert_eq!(contents(result), vec![2, 3]);
        // The consumed owner was released: exactly one strong owner is
        // left, so this release must free the source without a ledger
        // complaint.
        chelis_list_release(list);
        chelis_list_release(result);
    }
}

#[test]
fn skip_owned_agrees_with_the_cloning_skip_on_every_count() {
    unsafe {
        let values = [7_i64, 8, 9];
        for count in [0_i64, 1, 2, 3, 4, i64::MAX] {
            let borrowed = list_of(&values);
            let cloned = chelis_list_drop(borrowed, count);
            let owned = chelis_list_drop_owned(list_of(&values), count);
            assert_eq!(
                contents(cloned),
                contents(owned),
                "the two entry points disagree at count {count}"
            );
            assert_eq!(
                contents(borrowed),
                values.to_vec(),
                "the cloning entry point never touches its input"
            );
            chelis_list_release(borrowed);
            chelis_list_release(cloned);
            chelis_list_release(owned);
        }
    }
}

/// [05-OP-32]: a count at or above the length yields the empty List, and
/// the emptied list is still a list the other entry points accept.
#[test]
fn a_count_above_the_length_empties_the_list_without_breaking_it() {
    unsafe {
        let list = chelis_list_drop_owned(list_of(&[1, 2]), 9);
        assert_eq!(chelis_list_len(list), 0);
        let grown = chelis_list_append_owned(list, int_value(5));
        assert_eq!(contents(grown), vec![5]);
        chelis_list_release(grown);
    }
}

/// The push paths append at the buffer's end, so they stay correct at a
/// non-zero head. A `push` that wrote at `len - head` instead would pass
/// every zero-head test in the suite.
#[test]
fn append_and_concat_are_correct_after_a_skip() {
    unsafe {
        let skipped = chelis_list_drop_owned(list_of(&[1, 2, 3]), 2);
        let appended = chelis_list_append_owned(skipped, int_value(4));
        assert_eq!(contents(appended), vec![3, 4]);

        let rhs = list_of(&[5, 6]);
        let joined = chelis_list_concat_owned(appended, rhs);
        assert_eq!(contents(joined), vec![3, 4, 5, 6]);
        chelis_list_release(rhs);
        chelis_list_release(joined);

        // The right-hand side may itself carry a head.
        let lhs = list_of(&[1, 2]);
        let tail = chelis_list_drop_owned(list_of(&[7, 8, 9]), 1);
        let both = chelis_list_concat_owned(lhs, tail);
        assert_eq!(contents(both), vec![1, 2, 8, 9]);
        chelis_list_release(tail);
        chelis_list_release(both);
    }
}

/// An rhs that aliases the consumed lhs is a retained second owner, so
/// `concat_owned` clones. A skipped list is no different, and the clone
/// must carry the live window rather than the whole buffer.
#[test]
fn concat_owned_with_an_rhs_aliasing_a_skipped_lhs_clones_the_live_window() {
    unsafe {
        let skipped = chelis_list_drop_owned(list_of(&[1, 2, 3, 4]), 2);
        chelis_list_retain(skipped);
        let joined = chelis_list_concat_owned(skipped, skipped);
        assert!(!std::ptr::eq(joined, skipped), "an aliasing rhs clones");
        assert_eq!(contents(joined), vec![3, 4, 3, 4]);
        chelis_list_release(skipped);
        chelis_list_release(joined);
    }
}

#[test]
fn a_null_list_behaves_like_the_cloning_entry_point() {
    unsafe {
        let empty = chelis_list_drop_owned(std::ptr::null_mut(), 1);
        assert_eq!(chelis_list_len(empty), 0);
        chelis_list_release(empty);
    }
}

/// A skipped heap element leaves the outer list, and the outer list's
/// own finalizer must not release it a second time.
///
/// This row does **not** prove the skip released it. Nothing in process
/// can: a refcount is not observable through the public ABI, and the
/// list reads identically whether the release happened or not. The
/// release is proved by `a_cursor_walk_leaks_nothing_and_allocates_once`
/// below, which counts owners in the ledger. This row is here for the
/// half that is observable, and for the double-release the ledger would
/// also catch but which would crash here first.
#[test]
fn a_skipped_heap_element_leaves_the_list_without_a_second_release() {
    unsafe {
        let inner = list_of(&[42]);
        // The probe's own owner, beside the one the outer list takes.
        chelis_list_retain(inner);
        let outer = chelis_list_empty();
        chelis_list_push_moved(outer, chelis_value_take_list(inner));

        let skipped = chelis_list_drop_owned(outer, 1);
        assert_eq!(chelis_list_len(skipped), 0);
        chelis_list_release(skipped);

        // The probe's own owner survived the outer list's finalizer, so
        // that finalizer did not walk the retired slot.
        assert_eq!(chelis_list_len(inner), 1, "the probe's owner is still live");
        chelis_list_release(inner);
    }
}

fn assert_child_refuses(case: &str, diagnostic: &str) {
    let test_binary = std::env::current_exe().expect("current test binary");
    let output = Command::new(&test_binary)
        .args(["--exact", "skip_owned_invalid_input_child", "--nocapture"])
        .env(CHILD_CASE_ENV, case)
        .output()
        .unwrap_or_else(|error| panic!("run invalid-input child `{case}`: {error}"));
    assert!(
        !output.status.success(),
        "invalid-input case `{case}` returned success; its guard did not fire"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains(diagnostic),
        "invalid-input case `{case}` terminated without its branded diagnostic:\n{stderr}"
    );
}

/// The child half. `runtime_fail!` terminates the process, so the guard
/// cannot be observed in-process.
#[test]
fn skip_owned_invalid_input_child() {
    let Ok(case) = std::env::var(CHILD_CASE_ENV) else {
        return;
    };
    unsafe {
        match case.as_str() {
            "negative_count" => {
                chelis_list_drop_owned(list_of(&[1]), -1);
            }
            "negative_count_null" => {
                chelis_list_drop_owned(std::ptr::null_mut(), -1);
            }
            other => panic!("unknown child case {other}"),
        }
    }
    panic!("guarded child case `{case}` returned instead of terminating");
}

/// The diagnostic names `skip`, the operation the user wrote under
/// [05-OP-54], rather than this symbol's `drop` stem or [05-OP-67]'s
/// one-argument linearity consume. The null row proves the count is
/// checked before the pointer, so a null list cannot turn a rejected
/// count into an empty result.
#[test]
fn skip_owned_refuses_a_negative_count_before_it_reads_the_list() {
    assert_child_refuses("negative_count", "skip requires non-negative count");
    assert_child_refuses("negative_count_null", "skip requires non-negative count");
}

/// A borrowed list value handed back through the value ABI still reads
/// its live window, so nothing downstream of `chelis_value` observes the
/// offset.
#[test]
fn a_skipped_list_reads_correctly_through_the_value_abi() {
    unsafe {
        let skipped = chelis_list_drop_owned(list_of(&[1, 2, 3]), 1);
        let value = chelis_value_take_list(skipped);
        let borrowed = chelis_list_borrow_value(value);
        assert_eq!(contents(borrowed), vec![2, 3]);
        chelis_value_release(value);
    }
}

// ---------------------------------------------------------------------
// The ledger rows. Ownership is not observable in process: a refcount is
// not in the public ABI, and a list that leaked its skipped elements
// reads exactly like one that released them. The test-only allocation
// ledger is the instrument that can express the property, so the two
// rows below run a child under it and read its summary.
// ---------------------------------------------------------------------

#[cfg(feature = "ownership-ledger")]
const LEDGER_MODE_ENV: &str = "CHELIS_LIST_SKIP_OWNED_LEDGER_MODE";
#[cfg(feature = "ownership-ledger")]
const LEDGER_PATH_ENV: &str = "CHELIS_OWNERSHIP_LEDGER_PATH";
#[cfg(feature = "ownership-ledger")]
const WALK: i64 = 64;
#[cfg(feature = "ownership-ledger")]
const LARGE: i64 = 4096;

/// The child half of the ledger rows. Each arm builds a cursor walk and
/// releases everything it made, so a clean summary is the expected
/// result and any deviation is the defect.
#[cfg(feature = "ownership-ledger")]
#[test]
fn skip_owned_ledger_child() {
    let Ok(mode) = std::env::var(LEDGER_MODE_ENV) else {
        return;
    };
    unsafe {
        // Heap elements, so a missed release leaves a live owner rather
        // than an inert scalar the ledger never counted.
        let seed = chelis_list_empty();
        for value in 0..WALK {
            let inner = list_of(&[value]);
            chelis_list_push_moved(seed, chelis_value_take_list(inner));
        }
        match mode.as_str() {
            "owned" => {
                let mut cursor = seed;
                while chelis_list_len(cursor) > 0 {
                    cursor = chelis_list_drop_owned(cursor, 1);
                }
                chelis_list_release(cursor);
            }
            "cloning" => {
                let mut cursor = seed;
                while chelis_list_len(cursor) > 0 {
                    let next = chelis_list_drop(cursor, 1);
                    chelis_list_release(cursor);
                    cursor = next;
                }
                chelis_list_release(cursor);
            }
            "partial" => {
                // One element retired, sixty-three live: far below the
                // compaction threshold, so the list reaches its
                // finalizer still holding the retired slot. This is the only shape in
                // which the finalizer's choice of window is observable,
                // and a walk to the end is not it -- the last
                // compaction leaves that list empty.
                let cursor = chelis_list_drop_owned(seed, 1);
                chelis_list_release(cursor);
            }
            "large-skip" => {
                // Red-team round 1's witness, through the public entry
                // point: a single large skip whose result is retained.
                // The elements are scalars, so the only allocation that
                // matters is the list's own buffer.
                let big = chelis_list_empty();
                for value in 0..LARGE {
                    chelis_list_push_moved(big, int_value(value));
                }
                let tail = chelis_list_drop_owned(big, LARGE - 1);
                assert_eq!(chelis_list_len(tail), 1);
                chelis_list_release(tail);
                chelis_list_release(seed);
            }
            other => panic!("unknown ledger mode {other}"),
        }
    }
}

#[cfg(feature = "ownership-ledger")]
fn run_ledger_child(mode: &str) -> String {
    let path = std::env::temp_dir().join(format!(
        "chelis-list-skip-owned-{}-{mode}.jsonl",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&path);
    let output = Command::new(std::env::current_exe().expect("current test binary"))
        .args(["--exact", "skip_owned_ledger_child", "--nocapture"])
        .env(LEDGER_MODE_ENV, mode)
        .env(LEDGER_PATH_ENV, &path)
        .output()
        .expect("spawn ledger child");
    assert!(
        output.status.success(),
        "ledger child `{mode}` failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let ledger = std::fs::read_to_string(&path).expect("ledger file");
    let _ = std::fs::remove_file(&path);
    ledger
}

/// The property the value assertions cannot reach, and the receipt for
/// the whole change, in one instrument.
///
/// A cursor that walks `WALK` heap elements with the consuming skip must
/// end with no live owner, no live byte and no invalid operation, and
/// must allocate the lists the probe built and not one more. The cloning
/// control walks the same list through `chelis_list_drop` and allocates
/// one extra list per step, which is the quadratic shape chelis#2334
/// reports.
///
/// Evidentiary status: REGRESSION TEST. The allocation counts are the
/// measured difference between the two arms on this head; there is no
/// consuming arm before this change set.
#[cfg(feature = "ownership-ledger")]
#[test]
fn a_cursor_walk_leaks_nothing_and_allocates_once() {
    let owned = run_ledger_child("owned");
    for required in [
        r#""live_owners":0"#,
        r#""live_bytes":0"#,
        r#""invalid_operations":0"#,
    ] {
        assert!(
            owned.contains(required),
            "the consuming walk left {required:?} unsatisfied:\n{owned}"
        );
    }

    let list_allocations = |ledger: &str| -> usize {
        ledger
            .lines()
            .filter(|line| {
                line.contains(r#""event":"allocate""#) && line.contains(r#""kind":"List""#)
            })
            .count()
    };
    // The probe builds one seed plus `WALK` inner lists and nothing else.
    let built = WALK as usize + 1;
    assert_eq!(
        list_allocations(&owned),
        built,
        "the consuming walk allocates no list of its own:\n{owned}"
    );

    // The negative control: the same walk on the cloning entry point.
    // Without it, an arm that never ran would satisfy the count above.
    let cloning = run_ledger_child("cloning");
    assert_eq!(
        list_allocations(&cloning),
        built + WALK as usize,
        "the cloning walk allocates one list per step:\n{cloning}"
    );
    assert!(
        cloning.contains(r#""live_owners":0"#),
        "the cloning control is itself balanced:\n{cloning}"
    );
}

/// The finalizer walks the live window, not the allocation.
///
/// A list finalized while it still holds a retired prefix is the one
/// shape in which that choice is observable: `chelis_list_drop_owned`
/// has already released those elements, so a finalizer walking the
/// whole buffer releases each of them a second time against
/// [05-OP-44]'s "releases each stored child exactly once". A cursor
/// walked to the end cannot show it, because the last compaction leaves
/// the allocation empty and the two windows agree.
///
/// Evidentiary status: REGRESSION TEST, proven failing first. Adding a
/// whole-buffer accessor and pointing the finalizer at it leaves the
/// other fourteen rows in this file green and kills this one's child
/// with `Domain: chelis_value_release: heap kind mismatch (expected
/// List, got MappedFile)` -- the second release reading a freed and
/// reused allocation.
#[cfg(feature = "ownership-ledger")]
#[test]
fn a_list_finalized_with_a_retired_prefix_releases_each_child_once() {
    let ledger = run_ledger_child("partial");
    for required in [
        r#""live_owners":0"#,
        r#""live_bytes":0"#,
        r#""invalid_operations":0"#,
    ] {
        assert!(
            ledger.contains(required),
            "a finalize over a retired prefix left {required:?} unsatisfied:\n{ledger}"
        );
    }
}

/// A single large skip gives the retired capacity back.
///
/// Red-team round 1 measured the shape this locks: before the
/// compaction rebuilt its buffer, `skip(n - 1)` on a 1,000,000-element
/// list left a one-element list holding the whole 1,000,000-slot
/// allocation, 24 MB by the ledger's own formula, where the cloning
/// path it replaces allocated the suffix and freed the operand. The
/// bound was stated over the allocation and enforced over the length.
///
/// The ledger is the instrument because `buffer_capacity()` is private
/// to the runtime: the consuming skip records a `resize` on every
/// in-place return, so the last one for the list carries the figure the
/// allocation actually ends at.
///
/// Evidentiary status: REGRESSION TEST. With `Vec::drain` in place of
/// the rebuild, `bytes_after` is `24 * LARGE` and this row is red.
#[cfg(feature = "ownership-ledger")]
#[test]
fn a_large_skip_gives_the_retired_capacity_back() {
    let ledger = run_ledger_child("large-skip");
    let resizes: Vec<u64> = ledger
        .lines()
        .filter(|line| line.contains(r#""event":"resize""#) && line.contains(r#""kind":"List""#))
        .filter_map(|line| {
            let tail = line.split(r#""bytes_after":"#).nth(1)?;
            tail.split(|c: char| !c.is_ascii_digit())
                .next()?
                .parse::<u64>()
                .ok()
        })
        .collect();
    let last = *resizes
        .last()
        .unwrap_or_else(|| panic!("no list resize recorded:\n{ledger}"));
    // 24 bytes a slot. One live element, so one slot, and the pushes
    // that built the list resized upward before it.
    assert_eq!(
        last, 24,
        "a one-element result must not retain the {LARGE}-slot buffer; \
         the last recorded list resize was {last} bytes:\n{ledger}"
    );
}
