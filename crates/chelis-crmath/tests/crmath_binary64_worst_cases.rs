//! Per-PR binary64 worst cases (spec/design/correctly_rounded_math.md section 8, test 2).
//!
//! `fixtures/binary64_worst_cases.txt` is a deterministic sample of CORE-MATH's
//! shipped `.wc` corpora (inputs whose exact result lies closest to a rounding
//! boundary), each input with its negation, with MPFR-derived expected bits. These are
//! the inputs on which a nearly correct algorithm misrounds.

mod common;

use common::{FUNCTIONS, api_bits, mismatches, read_fixture};

const FIXTURE: &str = "binary64_worst_cases.txt";

#[test]
fn api_matches_mpfr_on_sampled_worst_cases() {
    let rows = read_fixture(FIXTURE);
    let bad = mismatches(&rows, api_bits);
    assert!(bad.is_empty(), "{} of {} worst cases differ:\n{}", bad.len(), rows.len(), bad.join("\n"));
}

#[test]
fn sample_covers_every_function_at_f64() {
    let rows = read_fixture(FIXTURE);
    for function in FUNCTIONS {
        let count = rows.iter().filter(|r| r.function == function && r.width == 64).count();
        assert_eq!(count, 256, "{function}: expected 128 worst cases and their negations");
    }
}

#[test]
fn planted_one_ulp_error_is_reported_on_every_worst_case() {
    let rows = read_fixture(FIXTURE);
    let bad = mismatches(&rows, |row| api_bits(row) ^ 1);
    assert_eq!(bad.len(), rows.len());
}
