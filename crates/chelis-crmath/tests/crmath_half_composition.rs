//! f16 and bf16 composition, exhaustively (spec/design/correctly_rounded_math.md
//! section 8, test 4).
//!
//! [05-OP-46]: for f16 and bf16 the operand widens exactly to f32, the function is
//! correctly rounded at f32, and that f32 value is finalized to storage once. Every one
//! of the 65,536 inputs of each dtype is checked for each function.
//!
//! Negative partner: a planted direct-to-storage rounding (the f64 result rounded
//! once, straight to f16, which is what "correctly rounded at the storage width"
//! would mean) is reported by the same check. At bf16 the two readings agree on
//! every input of these nine kernels (measured exhaustively): the readings differ
//! only where the f32 result lands exactly on a bf16 tie that the exact value
//! does not, and with 16 more bits in f32 than in bf16 none of these kernels
//! produces one, so a bf16 planted mutant has no input to be reported on.

use chelis_crmath as cr;
use chelis_crmath::profile::{Output, storage_reference};
use half::{bf16, f16};

type Kernels = (
    fn(f16) -> f16,
    fn(bf16) -> bf16,
    fn(f32) -> f32,
    fn(f64) -> f64,
);

const KERNELS: [(&str, Kernels); 9] = [
    ("exp", (cr::exp_f16, cr::exp_bf16, cr::exp_f32, cr::exp_f64)),
    ("log", (cr::log_f16, cr::log_bf16, cr::log_f32, cr::log_f64)),
    ("sin", (cr::sin_f16, cr::sin_bf16, cr::sin_f32, cr::sin_f64)),
    ("cos", (cr::cos_f16, cr::cos_bf16, cr::cos_f32, cr::cos_f64)),
    ("tan", (cr::tan_f16, cr::tan_bf16, cr::tan_f32, cr::tan_f64)),
    (
        "atan",
        (cr::atan_f16, cr::atan_bf16, cr::atan_f32, cr::atan_f64),
    ),
    (
        "tanh",
        (cr::tanh_f16, cr::tanh_bf16, cr::tanh_f32, cr::tanh_f64),
    ),
    ("erf", (cr::erf_f16, cr::erf_bf16, cr::erf_f32, cr::erf_f64)),
    (
        "erfc",
        (cr::erfc_f16, cr::erfc_bf16, cr::erfc_f32, cr::erfc_f64),
    ),
];

fn composition_f16(k32: fn(f32) -> f32, x: f16) -> u16 {
    f16::from_f32(k32(x.to_f32())).to_bits()
}

fn composition_bf16(k32: fn(f32) -> f32, x: bf16) -> u16 {
    bf16::from_f32(k32(x.to_f32())).to_bits()
}

/// Inputs on which `candidate` disagrees with the composition.
fn reported_f16(k32: fn(f32) -> f32, candidate: impl Fn(f16) -> u16) -> Vec<u16> {
    (0..=u16::MAX)
        .filter(|&bits| {
            let x = f16::from_bits(bits);
            candidate(x) != composition_f16(k32, x)
        })
        .collect()
}

fn reported_bf16(k32: fn(f32) -> f32, candidate: impl Fn(bf16) -> u16) -> Vec<u16> {
    (0..=u16::MAX)
        .filter(|&bits| {
            let x = bf16::from_bits(bits);
            candidate(x) != composition_bf16(k32, x)
        })
        .collect()
}

#[test]
fn f16_api_is_the_f32_composition_on_every_input() {
    for (name, (k16, _, k32, _)) in KERNELS {
        let bad = reported_f16(k32, |x| k16(x).to_bits());
        assert!(
            bad.is_empty(),
            "{name}_f16 differs from the composition on {} inputs, e.g. {:#06x}",
            bad.len(),
            bad[0]
        );
    }
}

#[test]
fn bf16_api_is_the_f32_composition_on_every_input() {
    for (name, (_, kbf16, k32, _)) in KERNELS {
        let bad = reported_bf16(k32, |x| kbf16(x).to_bits());
        assert!(
            bad.is_empty(),
            "{name}_bf16 differs from the composition on {} inputs, e.g. {:#06x}",
            bad.len(),
            bad[0]
        );
    }
}

/// `x` rounded once, ties to even, to a binary format with `mantissa_bits`
/// fraction bits and minimum normal exponent `min_exponent`; the result is a
/// value of that format, exact in f64 (or an infinity on overflow). `half`'s
/// own `from_f64` cannot be the planted mutant: with F16C on x86_64 it
/// narrows through f32 first, which rounds twice and turns the direct
/// rounding back into the composition.
fn round_once(x: f64, mantissa_bits: i32, min_exponent: i32) -> f64 {
    if !x.is_finite() || x == 0.0 {
        return x;
    }
    let exponent = (((x.to_bits() >> 52) & 0x7ff) as i32 - 1023).max(min_exponent);
    let quantum = 2.0_f64.powi(exponent - mantissa_bits);
    // Scaling by a power of two is exact, and so is rounding the scaled
    // value to an integer.
    (x / quantum).round_ties_even() * quantum
}

#[test]
fn planted_direct_rounding_is_reported() {
    let mut f16_reports = 0;
    for (_, (_, _, k32, k64)) in KERNELS {
        // The rounded value is exactly representable (or infinite), so the
        // final conversion is exact.
        f16_reports += reported_f16(k32, |x| {
            let rounded = round_once(k64(x.to_f64()), 10, -14);
            storage_reference(rounded.to_bits(), 64, Output::F16)
        })
        .len();
    }
    assert!(
        f16_reports > 0,
        "no f16 input distinguishes direct rounding from the composition"
    );
}
