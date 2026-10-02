//! Per-PR canary (spec/design/correctly_rounded_math.md section 8, test 1).
//!
//! `fixtures/canary.txt` holds MPFR-derived bit patterns for every [05-OP-46] special
//! case (signed zeros, infinities, NaNs with payload and sign, subnormal operands and
//! results, overflow and underflow thresholds) and the #2952, #2959, and #2971
//! witnesses, for each of the seven functions at f32 and f64. The API must match every
//! row bit for bit.
//!
//! Negative partners: the comparison reports a planted one-ULP change on every row,
//! and it reports the evaluator's former f32 route (compute at f64, narrow once) on
//! the double-rounding witnesses the fixture carries for it. Rust `std` is not used as
//! the bypass: its results are host-library dependent (glibc's `expf` is correctly
//! rounded, Apple's is not), so a canary that needed `std` to disagree would itself
//! vary by host, and the workspace lint forbids `std` transcendentals outside this
//! crate.

mod common;

use common::{FUNCTIONS, Row, api_bits, f64_kernel, mismatches, read_fixture};

const FIXTURE: &str = "canary.txt";
const FUNNEL_NOTE: &str = "f64 funnel double-rounds";

#[test]
fn api_matches_mpfr_on_every_canary_row() {
    let rows = read_fixture(FIXTURE);
    let bad = mismatches(&rows, api_bits);
    assert!(bad.is_empty(), "{} of {} canary rows differ:\n{}", bad.len(), rows.len(), bad.join("\n"));
}

#[test]
fn canary_covers_every_function_width_and_special_case() {
    let rows = read_fixture(FIXTURE);
    for function in FUNCTIONS {
        for width in [32, 64] {
            let notes: Vec<&str> = rows
                .iter()
                .filter(|r| r.function == function && r.width == width)
                .map(|r| r.note.as_str())
                .collect();
            for required in ["+0", "-0", "+inf", "-inf", "quiet NaN with payload", "negative quiet NaN", "signaling NaN", "smallest subnormal"] {
                assert!(notes.contains(&required), "{function} f{width}: no `{required}` row");
            }
        }
    }
    for witness in ["#2952", "#2959", "#2971"] {
        assert!(rows.iter().any(|r| r.note.contains(witness)), "no {witness} witness row");
    }
}

#[test]
fn planted_one_ulp_error_is_reported_on_every_row() {
    let rows = read_fixture(FIXTURE);
    let bad = mismatches(&rows, |row| api_bits(row) ^ 1);
    assert_eq!(bad.len(), rows.len(), "the comparison missed a planted one-ULP error");
}

/// The evaluator's former f32 route: the f64 function, narrowed once. Correctly
/// rounded f64 kernels make this deterministic, and it still differs from the
/// correctly rounded f32 result wherever narrowing double-rounds.
fn f64_funnel_bits(row: &Row) -> u64 {
    let input = f32::from_bits(u32::try_from(row.input).unwrap());
    #[allow(clippy::cast_possible_truncation)]
    let narrowed = f64_kernel(&row.function)(f64::from(input)) as f32;
    u64::from(narrowed.to_bits())
}

#[test]
fn f64_funnel_is_reported_on_its_witnesses() {
    let rows = read_fixture(FIXTURE);
    let witnesses: Vec<Row> = rows.into_iter().filter(|r| r.note.contains(FUNNEL_NOTE)).collect();
    assert!(!witnesses.is_empty(), "the canary carries no f64-funnel witness");
    let bad = mismatches(&witnesses, f64_funnel_bits);
    assert_eq!(bad.len(), witnesses.len(), "the f64 funnel matched a witness it should miss: {bad:?}");
}
