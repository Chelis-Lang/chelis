//! f16 and bf16 composition, exhaustively (spec/design/correctly_rounded_math.md
//! section 8, test 4).
//!
//! [05-OP-46]: for f16 and bf16 the operand widens exactly to f32, the function is
//! correctly rounded at f32, and that f32 value is finalized to storage once. Every one
//! of the 65,536 inputs of each dtype is checked for each function.
//!
//! Negative partner: a planted direct-to-storage rounding (the f64 result rounded
//! straight to f16 or bf16, which is what "correctly rounded at the storage width"
//! would mean) is reported by the same check at f16 and at bf16.

use chelis_crmath as cr;
use half::{bf16, f16};

type Kernels = (fn(f16) -> f16, fn(bf16) -> bf16, fn(f32) -> f32, fn(f64) -> f64);

const KERNELS: [(&str, Kernels); 7] = [
    ("exp", (cr::exp_f16, cr::exp_bf16, cr::exp_f32, cr::exp_f64)),
    ("log", (cr::log_f16, cr::log_bf16, cr::log_f32, cr::log_f64)),
    ("sin", (cr::sin_f16, cr::sin_bf16, cr::sin_f32, cr::sin_f64)),
    ("cos", (cr::cos_f16, cr::cos_bf16, cr::cos_f32, cr::cos_f64)),
    ("tan", (cr::tan_f16, cr::tan_bf16, cr::tan_f32, cr::tan_f64)),
    ("atan", (cr::atan_f16, cr::atan_bf16, cr::atan_f32, cr::atan_f64)),
    ("tanh", (cr::tanh_f16, cr::tanh_bf16, cr::tanh_f32, cr::tanh_f64)),
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
        assert!(bad.is_empty(), "{name}_f16 differs from the composition on {} inputs, e.g. {:#06x}", bad.len(), bad[0]);
    }
}

#[test]
fn bf16_api_is_the_f32_composition_on_every_input() {
    for (name, (_, kbf16, k32, _)) in KERNELS {
        let bad = reported_bf16(k32, |x| kbf16(x).to_bits());
        assert!(bad.is_empty(), "{name}_bf16 differs from the composition on {} inputs, e.g. {:#06x}", bad.len(), bad[0]);
    }
}

#[test]
fn planted_direct_rounding_is_reported() {
    let (mut f16_reports, mut bf16_reports) = (0, 0);
    for (_, (_, _, k32, k64)) in KERNELS {
        f16_reports += reported_f16(k32, |x| f16::from_f64(k64(x.to_f64())).to_bits()).len();
        bf16_reports += reported_bf16(k32, |x| bf16::from_f64(k64(x.to_f64())).to_bits()).len();
    }
    assert!(f16_reports > 0, "no f16 input distinguishes direct rounding from the composition");
    assert!(bf16_reports > 0, "no bf16 input distinguishes direct rounding from the composition");
}
