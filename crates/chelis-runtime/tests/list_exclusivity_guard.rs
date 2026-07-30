//! chelis#943: the in-place list mutators must refuse a shared list.
//!
//! `chelis_list_push` and `chelis_list_extend` are the whole reason this
//! change set is safe. `List[T]` is *specified* immutable
//! (`spec/04-type-system.md:695`, `chelis_project_plan.md:437`) and the
//! ABI encodes it — `chelis_list_append` takes `*const chelis_list` and
//! clones, so it cannot mutate. The new mutators take `*mut` and skip the
//! clone, which is only sound because the emitted code exclusively owns
//! the accumulator. `refcount == 1` is what enforces that, and it is
//! therefore the entire safety argument for the optimization.
//!
//! A guard with no test that it *fires* is the "belief without a test"
//! case `loud_unsupported.md` §C1 rule 4 rejects, and is how a guard rots
//! into a no-op. §C6.2 sets the pattern for exactly this shape — drive the
//! invalid input at the real FFI boundary in a subprocess and assert a
//! nonzero exit — because `runtime_fail!` terminates the process and so
//! cannot be observed in-process. PR #855 added three tests whose only job
//! was proving its boundary guard fires; these are the same for #943's.
//!
//! §C1.4's raise-or-prove applies whether or not the emitter can currently
//! produce a shared accumulator: the guard is load-bearing, so it is
//! tested directly rather than argued to be unreachable.

use std::process::Command;

use chelis_runtime::{
    chelis_list_empty, chelis_list_extend, chelis_list_push, chelis_list_retain,
    chelis_value_from_int64,
};

const CHILD_CASE_ENV: &str = "CHELIS_LIST_EXCLUSIVITY_CHILD_CASE";

/// The child half. Builds a list, takes a second reference so
/// `refcount == 2`, and drives the mutator that must refuse it. Each arm
/// must terminate; reaching the `panic!` below means the guard did not fire.
#[test]
fn shared_list_mutation_child() {
    let Ok(case) = std::env::var(CHILD_CASE_ENV) else {
        return;
    };
    unsafe {
        let list = chelis_list_empty();
        // The second owner. This is the condition the guard exists for:
        // pushing now would mutate a view someone else still holds.
        chelis_list_retain(list);

        match case.as_str() {
            "push" => {
                chelis_list_push(list, chelis_value_from_int64(1));
            }
            "extend" => {
                let src = chelis_list_empty();
                chelis_list_push(src, chelis_value_from_int64(2));
                chelis_list_extend(list, src);
            }
            other => panic!("unknown child case {other}"),
        }
    }
    panic!("shared-list case `{case}` returned instead of terminating");
}

#[test]
fn in_place_list_mutators_refuse_a_shared_list() {
    let test_binary = std::env::current_exe().expect("current test binary");
    for (case, symbol) in [
        ("push", "chelis_list_push"),
        ("extend", "chelis_list_extend"),
    ] {
        let output = Command::new(&test_binary)
            .args(["--exact", "shared_list_mutation_child", "--nocapture"])
            .env(CHILD_CASE_ENV, case)
            .output()
            .unwrap_or_else(|error| panic!("run shared-list child `{case}`: {error}"));

        assert!(
            !output.status.success(),
            "shared-list case `{case}` returned success; the exclusivity guard did not fire"
        );
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains(&format!(
                "{symbol} requires exclusive ownership (refcount 1)"
            )),
            "shared-list case `{case}` terminated without naming the guarantee \
             (§C1.3: a legitimate abort names what it protects):\n{stderr}"
        );
    }
}

/// The positive half, in-process: at `refcount == 1` the same calls are
/// accepted. Without this, a guard that rejected *everything* would pass
/// the negative test above and still be wrong.
#[test]
fn exclusively_owned_lists_accept_in_place_mutation() {
    unsafe {
        let list = chelis_list_empty();
        chelis_list_push(list, chelis_value_from_int64(1));
        chelis_list_push(list, chelis_value_from_int64(2));

        let src = chelis_list_empty();
        chelis_list_push(src, chelis_value_from_int64(3));
        chelis_list_extend(list, src);

        assert_eq!(
            chelis_runtime::chelis_list_len(list),
            3,
            "an exclusively owned list must accept push and extend"
        );
    }
}
