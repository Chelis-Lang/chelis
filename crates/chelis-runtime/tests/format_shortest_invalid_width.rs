//! Negative rows for the exact scalar-to-string boundary.
//!
//! Rendering owns its returned string and accepts every active scalar dtype.
//! Malformed foreign scalar carriers must terminate before a string handle is
//! returned.

use std::process::Command;

use chelis_runtime::{
    chelis_scalar, chelis_scalar_from_bits, chelis_string_data, chelis_string_from_scalar,
    chelis_string_release, CHELIS_DTYPE_BOOL, CHELIS_DTYPE_I64,
};

const CHILD_CASE_ENV: &str = "CHELIS_SCALAR_STRING_INVALID_CHILD_CASE";

#[test]
fn invalid_scalar_child() {
    let Ok(case) = std::env::var(CHILD_CASE_ENV) else {
        return;
    };
    let scalar = match case.as_str() {
        "invalid-dtype" => chelis_scalar {
            dtype: 9,
            reserved: [0; 7],
            bits: 0,
        },
        "reserved" => chelis_scalar {
            dtype: CHELIS_DTYPE_I64,
            reserved: [1, 0, 0, 0, 0, 0, 0],
            bits: 0,
        },
        "unused-bits" => chelis_scalar {
            dtype: CHELIS_DTYPE_BOOL,
            reserved: [0; 7],
            bits: 0x100,
        },
        "noncanonical-bool" => chelis_scalar {
            dtype: CHELIS_DTYPE_BOOL,
            reserved: [0; 7],
            bits: 2,
        },
        other => panic!("unknown child case {other}"),
    };
    let _ = chelis_string_from_scalar(scalar);
    panic!("malformed scalar case `{case}` returned instead of terminating");
}

fn run_child(case: &str) -> (bool, String) {
    let test_binary = std::env::current_exe().expect("current test binary");
    let output = Command::new(&test_binary)
        .args(["--exact", "invalid_scalar_child", "--nocapture"])
        .env(CHILD_CASE_ENV, case)
        .output()
        .unwrap_or_else(|error| panic!("run child `{case}`: {error}"));
    (
        output.status.success(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

#[test]
fn malformed_scalar_carriers_abort_before_rendering() {
    for (case, expected) in [
        ("invalid-dtype", "invalid Chelis runtime dtype id: 9"),
        ("reserved", "scalar reserved bytes must be zero"),
        ("unused-bits", "scalar has nonzero unused bits"),
        (
            "noncanonical-bool",
            "bool scalar payload must be zero or one",
        ),
    ] {
        let (success, stderr) = run_child(case);
        assert!(!success, "malformed scalar case `{case}` returned success");
        assert!(
            stderr.contains(expected),
            "malformed scalar case `{case}` lost `{expected}`:\n{stderr}"
        );
    }
}

#[test]
fn signed_integer_rendering_uses_the_exact_scalar_bits() {
    let scalar = chelis_scalar_from_bits(
        CHELIS_DTYPE_I64,
        u64::from_ne_bytes((-9_007_199_254_740_993_i64).to_ne_bytes()),
    );
    let rendered = chelis_string_from_scalar(scalar);
    unsafe {
        let text = std::ffi::CStr::from_ptr(chelis_string_data(rendered))
            .to_str()
            .expect("scalar rendering is UTF-8");
        assert_eq!(text, "-9007199254740993");
        chelis_string_release(rendered);
    }
}
