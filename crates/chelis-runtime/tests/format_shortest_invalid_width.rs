//! chelis#732 Phase 2 negative rows for `chelis_format_shortest`: every
//! contract violation must abort loudly BEFORE any text reaches the
//! caller's buffer - the loud_unsupported.md section C1 response,
//! mirroring `runtime_dtype_invalid_ffi.rs`'s subprocess pattern.
//!
//! Two contracts, checked separately because the diagnostics must name
//! which side is wrong:
//!
//! * **the dtype**: a non-float dtype id (integer/bool ids) or an
//!   unknown raw id. No default width may exist - a formatter that
//!   "helpfully" picked f64 for an integer id would recreate the census's
//!   everything-through-double funnel.
//! * **the buffer**: a NULL pointer, or a `cap` too small for the
//!   rendering plus its NUL. Added at PR #863 R2, which demonstrated that
//!   the previous `(double, int, char *)` signature could not express -
//!   let alone check - the buffer contract its own docs stated, and wrote
//!   past a four-byte logical buffer. Truncating instead of aborting would
//!   be a chelis#703 substitution at the byte level: a short write that
//!   parses back as a DIFFERENT value is precisely the unfaithful exit
//!   this plan exists to kill.

use std::os::raw::c_char;
use std::process::Command;

use chelis_runtime::{chelis_format_shortest, CHELIS_FORMAT_SHORTEST_BUF};

const CHILD_CASE_ENV: &str = "CHELIS_FORMAT_SHORTEST_INVALID_CHILD_CASE";
const BUFFER_CHILD_CASE_ENV: &str = "CHELIS_FORMAT_SHORTEST_BUFFER_CHILD_CASE";

#[test]
fn invalid_width_child() {
    let Ok(case) = std::env::var(CHILD_CASE_ENV) else {
        return;
    };
    let dtype: i32 = case.parse().unwrap_or_else(|_| panic!("bad case {case}"));
    let mut buf = [0u8; CHELIS_FORMAT_SHORTEST_BUF];
    unsafe {
        chelis_format_shortest(1.5, dtype, buf.as_mut_ptr() as *mut c_char, buf.len());
    }
    panic!("dtype {dtype} returned a rendering instead of terminating");
}

#[test]
fn invalid_buffer_child() {
    let Ok(case) = std::env::var(BUFFER_CHILD_CASE_ENV) else {
        return;
    };
    let width = chelis_vocab::RuntimeDType::F64.id();
    match case.as_str() {
        "null" => unsafe {
            chelis_format_shortest(1.5, width, std::ptr::null_mut(), CHELIS_FORMAT_SHORTEST_BUF);
        },
        // A real buffer whose DECLARED capacity is far smaller than the
        // rendering needs: `0.30000000000000004` is 19 bytes at F64. The
        // backing storage is deliberately full-size so the probe stays
        // defined if the abort ever regresses; the point under test is
        // that the routine refuses the declared cap, not that the bytes
        // happen to fit somewhere.
        "short" => unsafe {
            let mut buf = [0u8; CHELIS_FORMAT_SHORTEST_BUF];
            chelis_format_shortest(
                0.300_000_000_000_000_04,
                width,
                buf.as_mut_ptr() as *mut c_char,
                4,
            );
        },
        // Exactly one byte short: the rendering fits but its NUL does not.
        // `0.1` at F64 renders three characters, so a cap of 3 leaves no
        // room to terminate.
        "off-by-one" => unsafe {
            let mut buf = [0u8; CHELIS_FORMAT_SHORTEST_BUF];
            chelis_format_shortest(0.1, width, buf.as_mut_ptr() as *mut c_char, 3);
        },
        "zero" => unsafe {
            let mut buf = [0u8; CHELIS_FORMAT_SHORTEST_BUF];
            chelis_format_shortest(1.5, width, buf.as_mut_ptr() as *mut c_char, 0);
        },
        other => panic!("bad buffer case {other}"),
    }
    panic!("buffer case {case} returned instead of terminating");
}

fn run_child(env_key: &str, case: &str, child_test: &str) -> (bool, String) {
    let test_binary = std::env::current_exe().expect("current test binary");
    let output = Command::new(&test_binary)
        .args(["--exact", child_test, "--nocapture"])
        .env(env_key, case)
        .output()
        .unwrap_or_else(|error| panic!("run child `{case}`: {error}"));
    (
        output.status.success(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

#[test]
fn non_float_and_unknown_dtype_ids_abort_before_formatting() {
    // Integer and bool ids (real dtypes that must not funnel through the
    // float formatter), the first unused id, and the extrema.
    for (dtype, expected) in [
        (
            chelis_vocab::RuntimeDType::I64.id(),
            "is not a float dtype".to_string(),
        ),
        (
            chelis_vocab::RuntimeDType::Bool.id(),
            "is not a float dtype".to_string(),
        ),
        (9, "invalid Chelis runtime dtype id: 9".to_string()),
        (-1, "invalid Chelis runtime dtype id: -1".to_string()),
        (
            i32::MAX,
            format!("invalid Chelis runtime dtype id: {}", i32::MAX),
        ),
    ] {
        let (success, stderr) =
            run_child(CHILD_CASE_ENV, &dtype.to_string(), "invalid_width_child");
        assert!(!success, "dtype {dtype} returned success");
        assert!(
            stderr.contains(&expected),
            "dtype {dtype} lost its diagnostic (wanted `{expected}`):\n{stderr}"
        );
    }
}

#[test]
fn null_and_undersized_buffers_abort_instead_of_writing() {
    for (case, expected) in [
        ("null", "null output buffer"),
        ("short", "caller declared 4"),
        ("off-by-one", "caller declared 3"),
        ("zero", "caller declared 0"),
    ] {
        let (success, stderr) = run_child(BUFFER_CHILD_CASE_ENV, case, "invalid_buffer_child");
        assert!(!success, "buffer case {case} returned success");
        assert!(
            stderr.contains(expected),
            "buffer case {case} lost its diagnostic (wanted `{expected}`):\n{stderr}"
        );
    }
}

#[test]
fn a_sufficient_buffer_returns_the_written_length() {
    let mut buf = [0u8; CHELIS_FORMAT_SHORTEST_BUF];
    let written = unsafe {
        chelis_format_shortest(
            0.1,
            chelis_vocab::RuntimeDType::F64.id(),
            buf.as_mut_ptr() as *mut c_char,
            buf.len(),
        )
    };
    assert_eq!(written, 3, "`0.1` renders three bytes before the NUL");
    assert_eq!(&buf[..4], b"0.1\0", "the rendering is NUL-terminated");
    // The documented minimum is exactly sufficient, and one more than the
    // rendering needs is too: the boundary is `len + 1`, not `len`.
    let exact = unsafe {
        chelis_format_shortest(
            0.1,
            chelis_vocab::RuntimeDType::F64.id(),
            buf.as_mut_ptr() as *mut c_char,
            4,
        )
    };
    assert_eq!(exact, 3, "a cap of exactly len + 1 is sufficient");
}
