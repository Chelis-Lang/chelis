//! Exhaustive binary32 manual gate (spec/design/correctly_rounded_math.md section 8,
//! test 3; `docs/manual_gates.md`). Ignored by default: it evaluates all 2^32 inputs
//! of each of the seven functions.
//!
//! For each input `x`, the exact value `v = f(x)` lies within half an f64 ULP of the
//! correctly rounded f64 result `z`, so it lies between `z`'s two f64 neighbours.
//! Rounding to f32 is monotone, so when both neighbours round to the same f32, that
//! f32 is the correctly rounded result, and the f32 kernel must return it. A zero `z`
//! carries the sign of `v`, so its rounding is that signed zero. The few inputs whose
//! bracket straddles an f32 rounding boundary are written to a list and checked
//! against MPFR by `scripts/vendor_core_math.py resolve`, which needs `gmpy2` in the
//! checkout's `.venv`. This makes MPFR the authority on every input the binary64 kernel
//! cannot decide alone, and makes the gate depend on two independent CORE-MATH
//! kernels agreeing everywhere else.
//!
//! `CHELIS_CRMATH_THREADS` sets the worker count (default: half the available CPUs).

mod common;

use std::fmt::Write as _;
use std::path::Path;
use std::process::Command;

use common::{FUNCTIONS, f32_kernel, f64_kernel};

const CANONICAL_F32: u32 = 0x7fc0_0000;

#[derive(Default)]
struct Outcome {
    mismatches: Vec<String>,
    ambiguous: Vec<(u32, u32)>,
}

/// The correctly rounded f32 result when the f64 bracket decides it.
fn decided(z: f64) -> Option<u32> {
    if z.is_nan() {
        return Some(CANONICAL_F32);
    }
    #[allow(clippy::cast_possible_truncation)]
    let round = |v: f64| (v as f32).to_bits();
    if z == 0.0 {
        return Some(round(z));
    }
    let (low, high) = (round(z.next_down()), round(z.next_up()));
    (low == high).then_some(low)
}

fn check_range(function: &str, start: u64, end: u64) -> Outcome {
    let (k32, k64) = (f32_kernel(function), f64_kernel(function));
    let mut outcome = Outcome::default();
    for wide in start..end {
        let bits = u32::try_from(wide).unwrap();
        let x = f32::from_bits(bits);
        let got = k32(x).to_bits();
        match decided(k64(f64::from(x))) {
            Some(expected) if expected == got => {}
            Some(expected) => outcome.mismatches.push(format!(
                "{function} f32 input {bits:08x}: kernel {got:08x}, expected {expected:08x}"
            )),
            None => outcome.ambiguous.push((bits, got)),
        }
    }
    outcome
}

fn threads() -> u64 {
    std::env::var("CHELIS_CRMATH_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| {
            let available = std::thread::available_parallelism().map_or(2, std::num::NonZero::get);
            u64::try_from(available / 2).unwrap().max(1)
        })
}

#[test]
#[ignore = "manual gate: every binary32 input of the seven functions; see docs/manual_gates.md"]
fn every_binary32_input_is_correctly_rounded() {
    let workers = threads();
    let span = 1_u64 << 32;
    let mut mismatches = Vec::new();
    let mut ambiguous_list = String::from("# function width input kernel-result\n");
    for function in FUNCTIONS {
        let outcomes: Vec<Outcome> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..workers)
                .map(|w| {
                    scope.spawn(move || {
                        check_range(function, span * w / workers, span * (w + 1) / workers)
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let ambiguous: usize = outcomes.iter().map(|o| o.ambiguous.len()).sum();
        let wrong: usize = outcomes.iter().map(|o| o.mismatches.len()).sum();
        println!("{function}: 2^32 inputs, {wrong} mismatches, {ambiguous} left to MPFR");
        for outcome in outcomes {
            mismatches.extend(outcome.mismatches);
            for (input, result) in outcome.ambiguous {
                writeln!(ambiguous_list, "{function} f32 {input:08x} {result:08x}").unwrap();
            }
        }
    }
    assert!(
        mismatches.is_empty(),
        "{} mismatches:\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );

    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let list = std::env::temp_dir().join(format!(
        "chelis-crmath-ambiguous-{}.txt",
        std::process::id()
    ));
    std::fs::write(&list, ambiguous_list).unwrap();
    let python = root.join(".venv/bin/python");
    let status = Command::new(&python)
        .arg(root.join("scripts/vendor_core_math.py"))
        .arg("resolve")
        .arg(&list)
        .status()
        .unwrap_or_else(|e| panic!("cannot run {}: {e}", python.display()));
    assert!(
        status.success(),
        "MPFR resolution of {} failed ({status})",
        list.display()
    );
    std::fs::remove_file(&list).unwrap();
}

#[test]
fn bracket_decides_only_when_both_neighbours_agree() {
    // 1.0 is exact, so its neighbours both round to 1.0f32.
    assert_eq!(decided(1.0), Some(1.0_f32.to_bits()));
    // Exactly halfway between two f32 values: the neighbours round apart.
    let midpoint = f64::from(1.0_f32) + f64::from(f32::EPSILON) / 2.0;
    assert_eq!(decided(midpoint), None);
    // Zero keeps its sign; NaN is the canonical NaN; infinity stays infinite.
    assert_eq!(decided(-0.0), Some((-0.0_f32).to_bits()));
    assert_eq!(decided(f64::NAN), Some(CANONICAL_F32));
    assert_eq!(decided(f64::INFINITY), Some(f32::INFINITY.to_bits()));
}
