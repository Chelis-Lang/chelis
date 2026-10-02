//! NaN canonicalization at the API (design section 8, test 5).
//!
//! Every NaN result, including one from a payload-carrying, negative, or signaling NaN
//! operand, is [04-NUM-2]'s canonical quiet NaN. The negative partner calls the raw
//! upstream kernels, which skip the wrapper, and shows they return a non-canonical
//! NaN, so the wrapper is what canonicalizes.

use super::*;

mod raw {
    unsafe extern "C" {
        pub safe fn chelis_crmath_ffi_raw_expf(x: f32) -> f32;
        pub safe fn chelis_crmath_ffi_raw_logf(x: f32) -> f32;
        pub safe fn chelis_crmath_ffi_raw_sinf(x: f32) -> f32;
        pub safe fn chelis_crmath_ffi_raw_cosf(x: f32) -> f32;
        pub safe fn chelis_crmath_ffi_raw_tanf(x: f32) -> f32;
        pub safe fn chelis_crmath_ffi_raw_atanf(x: f32) -> f32;
        pub safe fn chelis_crmath_ffi_raw_tanhf(x: f32) -> f32;
        pub safe fn chelis_crmath_ffi_raw_exp(x: f64) -> f64;
        pub safe fn chelis_crmath_ffi_raw_log(x: f64) -> f64;
        pub safe fn chelis_crmath_ffi_raw_sin(x: f64) -> f64;
        pub safe fn chelis_crmath_ffi_raw_cos(x: f64) -> f64;
        pub safe fn chelis_crmath_ffi_raw_tan(x: f64) -> f64;
        pub safe fn chelis_crmath_ffi_raw_atan(x: f64) -> f64;
        pub safe fn chelis_crmath_ffi_raw_tanh(x: f64) -> f64;
    }
}

type F32Pair = (&'static str, fn(f32) -> f32, extern "C" fn(f32) -> f32);
type F64Pair = (&'static str, fn(f64) -> f64, extern "C" fn(f64) -> f64);

const F32_KERNELS: [F32Pair; 7] = [
    ("exp", exp_f32, raw::chelis_crmath_ffi_raw_expf),
    ("log", log_f32, raw::chelis_crmath_ffi_raw_logf),
    ("sin", sin_f32, raw::chelis_crmath_ffi_raw_sinf),
    ("cos", cos_f32, raw::chelis_crmath_ffi_raw_cosf),
    ("tan", tan_f32, raw::chelis_crmath_ffi_raw_tanf),
    ("atan", atan_f32, raw::chelis_crmath_ffi_raw_atanf),
    ("tanh", tanh_f32, raw::chelis_crmath_ffi_raw_tanhf),
];

const F64_KERNELS: [F64Pair; 7] = [
    ("exp", exp_f64, raw::chelis_crmath_ffi_raw_exp),
    ("log", log_f64, raw::chelis_crmath_ffi_raw_log),
    ("sin", sin_f64, raw::chelis_crmath_ffi_raw_sin),
    ("cos", cos_f64, raw::chelis_crmath_ffi_raw_cos),
    ("tan", tan_f64, raw::chelis_crmath_ffi_raw_tan),
    ("atan", atan_f64, raw::chelis_crmath_ffi_raw_atan),
    ("tanh", tanh_f64, raw::chelis_crmath_ffi_raw_tanh),
];

const CANONICAL_F32: u32 = 0x7fc0_0000;
const CANONICAL_F64: u64 = 0x7ff8_0000_0000_0000;
const CANONICAL_F16: u16 = 0x7e00;
const CANONICAL_BF16: u16 = 0x7fc0;

/// Non-canonical NaN operands: negative quiet, payload-carrying quiet, signaling,
/// and negative signaling with payload.
const NAN_OPERANDS_F32: [u32; 4] = [0xffc0_0000, 0x7fc1_2345, 0x7f80_0001, 0xffa0_0001];
const NAN_OPERANDS_F64: [u64; 4] = [
    0xfff8_0000_0000_0000,
    0x7ff8_0000_0001_2345,
    0x7ff0_0000_0000_0001,
    0xfff4_0000_0000_0001,
];
/// Operands outside a function's domain, whose result is a NaN from a non-NaN operand.
const DOMAIN_NAN_OPERANDS: [(&str, f64); 6] = [
    ("log", -1.0),
    ("log", f64::NEG_INFINITY),
    ("sin", f64::INFINITY),
    ("cos", f64::NEG_INFINITY),
    ("tan", f64::INFINITY),
    ("sin", f64::NEG_INFINITY),
];

#[test]
fn every_f32_nan_result_is_canonical() {
    for (name, wrapped, _) in F32_KERNELS {
        for bits in NAN_OPERANDS_F32 {
            let got = wrapped(f32::from_bits(bits)).to_bits();
            assert_eq!(got, CANONICAL_F32, "{name}_f32({bits:#010x}) gave {got:#010x}");
        }
    }
    for (name, x) in DOMAIN_NAN_OPERANDS {
        let (_, wrapped, _) = F32_KERNELS.iter().find(|k| k.0 == name).unwrap();
        let got = wrapped(x as f32).to_bits();
        assert_eq!(got, CANONICAL_F32, "{name}_f32({x}) gave {got:#010x}");
    }
}

#[test]
fn every_f64_nan_result_is_canonical() {
    for (name, wrapped, _) in F64_KERNELS {
        for bits in NAN_OPERANDS_F64 {
            let got = wrapped(f64::from_bits(bits)).to_bits();
            assert_eq!(got, CANONICAL_F64, "{name}_f64({bits:#018x}) gave {got:#018x}");
        }
    }
    for (name, x) in DOMAIN_NAN_OPERANDS {
        let (_, wrapped, _) = F64_KERNELS.iter().find(|k| k.0 == name).unwrap();
        let got = wrapped(x).to_bits();
        assert_eq!(got, CANONICAL_F64, "{name}_f64({x}) gave {got:#018x}");
    }
}

#[test]
fn every_half_nan_result_is_canonical() {
    type HalfKernels = (fn(f16) -> f16, fn(bf16) -> bf16);
    let kernels: [HalfKernels; 7] = [
        (exp_f16, exp_bf16),
        (log_f16, log_bf16),
        (sin_f16, sin_bf16),
        (cos_f16, cos_bf16),
        (tan_f16, tan_bf16),
        (atan_f16, atan_bf16),
        (tanh_f16, tanh_bf16),
    ];
    // Negative quiet, payload-carrying quiet, and signaling NaNs at each width.
    let f16_operands = [0xfe00_u16, 0x7e12, 0x7c01];
    let bf16_operands = [0xffc0_u16, 0x7fc5, 0x7f81];
    for (index, (k16, kbf16)) in kernels.iter().enumerate() {
        for bits in f16_operands {
            let got = k16(f16::from_bits(bits)).to_bits();
            assert_eq!(got, CANONICAL_F16, "kernel {index} f16({bits:#06x}) gave {got:#06x}");
        }
        for bits in bf16_operands {
            let got = kbf16(bf16::from_bits(bits)).to_bits();
            assert_eq!(got, CANONICAL_BF16, "kernel {index} bf16({bits:#06x}) gave {got:#06x}");
        }
    }
}

/// Negative partner: without the wrapper, each upstream kernel returns a NaN that is
/// not canonical for at least one of the operands above. If this ever fails, the
/// positive tests above no longer prove anything about the wrapper.
#[test]
fn raw_upstream_kernels_do_not_canonicalize() {
    for (name, _, raw) in F32_KERNELS {
        let witness = NAN_OPERANDS_F32
            .iter()
            .map(|&bits| raw(f32::from_bits(bits)).to_bits())
            .find(|&got| got != CANONICAL_F32);
        assert!(witness.is_some(), "raw {name}f canonicalized every NaN operand");
    }
    for (name, _, raw) in F64_KERNELS {
        let witness = NAN_OPERANDS_F64
            .iter()
            .map(|&bits| raw(f64::from_bits(bits)).to_bits())
            .find(|&got| got != CANONICAL_F64);
        assert!(witness.is_some(), "raw {name} canonicalized every NaN operand");
    }
}
