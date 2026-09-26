//! chelis#2522: releasing a deeply nested value must not recurse on the
//! native stack.
//!
//! [05-OP-44] runs each kind's finalizer exactly once when its last owner is
//! released, and a container's finalizer releases its children. The runtime
//! did that by recursion, one set of native frames per level, so releasing a
//! recursive data-type chain overflowed the stack between depth 6,000 and
//! 20,000 even with no Chelis call involved.
//!
//! Each case builds a chain 200,000 levels deep of one container kind whose
//! innermost level holds a probe list that the test also owns, then releases
//! the chain on a thread with a 256 KiB stack. Reclamation is read from the
//! probe: the in-place push accepts only a list with exactly one owner, so it
//! succeeds only if releasing the chain gave back the chain's owner of the
//! probe. The push refuses by terminating the process, so every case runs in a
//! child process.

use std::process::Command;

use chelis_runtime::{
    chelis_adt_construct, chelis_dict_insert, chelis_list_empty, chelis_list_push_moved,
    chelis_list_release, chelis_list_retain, chelis_option_some, chelis_scalar_from_bits,
    chelis_string_from_cstr, chelis_string_release, chelis_tuple_from_values, chelis_value,
    chelis_value_box_scalar, chelis_value_release, chelis_value_take_adt, chelis_value_take_dict,
    chelis_value_take_list, chelis_value_take_option, chelis_value_take_tuple, CHELIS_DTYPE_I64,
};

const CHILD_CASE_ENV: &str = "CHELIS_DEEP_RELEASE_CHILD_CASE";
const DEPTH: usize = 200_000;
const STACK_BYTES: usize = 256 * 1024;
const CASES: &[&str] = &["adt", "list", "option", "tuple", "dict"];

unsafe fn int_value(value: i64) -> chelis_value {
    chelis_value_box_scalar(chelis_scalar_from_bits(
        CHELIS_DTYPE_I64,
        u64::from_ne_bytes(value.to_ne_bytes()),
    ))
}

/// Wrap `inner` in one level of `case`'s container, consuming `inner`.
unsafe fn wrap(case: &str, inner: chelis_value) -> chelis_value {
    let outer = match case {
        "adt" => {
            let ctor = chelis_string_from_cstr(c"Link".as_ptr());
            let fields = [int_value(1), inner];
            let adt = chelis_value_take_adt(chelis_adt_construct(ctor, fields.as_ptr(), 2));
            chelis_string_release(ctor);
            adt
        }
        "list" => {
            let list = chelis_list_empty();
            chelis_list_push_moved(list, inner);
            return chelis_value_take_list(list);
        }
        "option" => chelis_value_take_option(chelis_option_some(inner)),
        "tuple" => {
            let items = [inner, int_value(1)];
            chelis_value_take_tuple(chelis_tuple_from_values(items.as_ptr(), 2))
        }
        "dict" => chelis_value_take_dict(chelis_dict_insert(std::ptr::null(), int_value(1), inner)),
        other => panic!("unknown deep release case `{other}`"),
    };
    // Every constructor above except the in-place push retains its children.
    chelis_value_release(inner);
    outer
}

/// The child half: build, release on a small stack, then prove reclamation.
#[test]
fn deep_release_child() {
    let Ok(case) = std::env::var(CHILD_CASE_ENV) else {
        return;
    };
    unsafe {
        let probe = chelis_list_empty();
        chelis_list_retain(probe);
        let mut chain = chelis_value_take_list(probe);
        for _ in 0..DEPTH {
            chain = wrap(&case, chain);
        }
        // Raw pointers are not `Send`; the value is moved to the thread as bits.
        let bits: [u8; std::mem::size_of::<chelis_value>()] = std::mem::transmute(chain);
        std::thread::Builder::new()
            .stack_size(STACK_BYTES)
            .spawn(move || {
                let chain: chelis_value = std::mem::transmute(bits);
                chelis_value_release(chain);
            })
            .expect("spawn release thread")
            .join()
            .expect("release thread");
        chelis_list_push_moved(probe, int_value(7));
        chelis_list_release(probe);
    }
    println!("deep release reclaimed {case}");
}

#[test]
fn a_deep_chain_of_every_container_kind_releases_in_bounded_stack() {
    let test_binary = std::env::current_exe().expect("current test binary");
    let mut failures = Vec::new();
    for case in CASES {
        let output = Command::new(&test_binary)
            .args(["--exact", "deep_release_child", "--nocapture"])
            .env(CHILD_CASE_ENV, case)
            .output()
            .unwrap_or_else(|error| panic!("run deep release child `{case}`: {error}"));
        let stdout = String::from_utf8_lossy(&output.stdout);
        if !output.status.success() || !stdout.contains(&format!("deep release reclaimed {case}")) {
            failures.push(format!(
                "{case}: {} {}",
                output.status,
                String::from_utf8_lossy(&output.stderr)
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The negative control: a chain whose release the test withholds leaves the
/// probe shared, and the push must refuse it. Without this, a probe that
/// accepted every list would pass the test above.
#[test]
fn deep_release_negative_child() {
    let Ok(case) = std::env::var(CHILD_CASE_ENV) else {
        return;
    };
    assert_eq!(case, "withheld");
    unsafe {
        let probe = chelis_list_empty();
        chelis_list_retain(probe);
        let chain = wrap("adt", chelis_value_take_list(probe));
        chelis_list_push_moved(probe, int_value(7));
        chelis_value_release(chain);
    }
    println!("withheld release accepted");
}

#[test]
fn the_reclamation_probe_refuses_a_list_the_chain_still_owns() {
    let test_binary = std::env::current_exe().expect("current test binary");
    let output = Command::new(&test_binary)
        .args(["--exact", "deep_release_negative_child", "--nocapture"])
        .env(CHILD_CASE_ENV, "withheld")
        .output()
        .expect("run negative child");
    assert!(!output.status.success(), "the probe accepted a shared list");
    assert!(String::from_utf8_lossy(&output.stderr)
        .contains("chelis_list_push_moved requires exclusive ownership (refcount 1)"));
}
