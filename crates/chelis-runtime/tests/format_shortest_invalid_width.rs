//! chelis#732 Phase 2 negative rows for `chelis_format_shortest`: a
//! non-float width_kind (integer/bool ids) or an unknown raw id must abort
//! loudly BEFORE any text is produced - the loud_unsupported.md section C1
//! response, mirroring `runtime_dtype_invalid_ffi.rs`'s subprocess
//! pattern. No default width may exist: a formatter that "helpfully"
//! picked f64 for an integer id would recreate the census's
//! everything-through-double funnel.

use std::os::raw::c_char;
use std::process::Command;

use chelis_runtime::{CHELIS_FORMAT_SHORTEST_BUF, chelis_format_shortest};

const CHILD_CASE_ENV: &str = "CHELIS_FORMAT_SHORTEST_INVALID_CHILD_CASE";

#[test]
fn invalid_width_child() {
    let Ok(case) = std::env::var(CHILD_CASE_ENV) else {
        return;
    };
    let width_kind: i32 = case.parse().unwrap_or_else(|_| panic!("bad case {case}"));
    let mut buf = [0u8; CHELIS_FORMAT_SHORTEST_BUF];
    unsafe {
        chelis_format_shortest(1.5, width_kind, buf.as_mut_ptr() as *mut c_char);
    }
    panic!("width_kind {width_kind} returned a rendering instead of terminating");
}

#[test]
fn non_float_and_unknown_width_kinds_abort_before_formatting() {
    let test_binary = std::env::current_exe().expect("current test binary");
    // Integer and bool ids (real dtypes that must not funnel through the
    // float formatter), the first unused id, and the extrema.
    for (width_kind, expected) in [
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
        let output = Command::new(&test_binary)
            .args(["--exact", "invalid_width_child", "--nocapture"])
            .env(CHILD_CASE_ENV, width_kind.to_string())
            .output()
            .unwrap_or_else(|error| panic!("run invalid-width child `{width_kind}`: {error}"));
        assert!(
            !output.status.success(),
            "width_kind {width_kind} returned success"
        );
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains(&expected),
            "width_kind {width_kind} lost its diagnostic (wanted `{expected}`):\n{stderr}"
        );
    }
}
