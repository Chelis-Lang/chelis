//! chelis#732 Phase 2: the byte-equality lock between the compiled lane's
//! exact scalar-to-string boundary and the reference renderer
//! `chelis_types::observation::format_element` (faithful_observation.md
//! section C3.3: "unit-tested against the Rust formatter over the
//! harness's value table - byte equality is the test").
//!
//! Coverage tiers:
//!
//! * the round-trip harness's FROZEN per-dtype value tables (mirrored
//!   verbatim from `observation_roundtrip_harness.rs`; append-only there,
//!   append here in the same change);
//! * the section 8.1 grammar edge set (thresholds, specials, signed zero,
//!   subnormals, extremes);
//! * EXHAUSTIVE sweeps over all 65536 bit patterns per half format;
//! * deterministic 65536-pattern bit sweeps for f32 and f64 (stratified:
//!   the generator visits every exponent byte), NaN classes included.
//!
//! Malformed scalar-carrier rows live in
//! `format_shortest_invalid_width.rs`.

use chelis_crmath::profile::{storage_reference, Output};
use chelis_runtime::{
    chelis_scalar, chelis_scalar_from_bits, chelis_string_data, chelis_string_from_scalar,
    chelis_string_release,
};
use chelis_types::types::Prim;
use chelis_types::{format_element, ElementRef};
use chelis_vocab::RuntimeDType;
fn scalar_from_f64(value: f64, dtype: RuntimeDType) -> chelis_scalar {
    let bits = match dtype {
        RuntimeDType::F64 => value.to_bits(),
        RuntimeDType::F32 => u64::from((value as f32).to_bits()),
        RuntimeDType::F16 => u64::from(storage_reference(value.to_bits(), 64, Output::F16)),
        RuntimeDType::Bf16 => u64::from(storage_reference(value.to_bits(), 64, Output::Bf16)),
        _ => panic!("floating reference helper received {}", dtype.name()),
    };
    chelis_scalar_from_bits(dtype.id() as u8, bits)
}

/// Call the exact public C ABI routine and decode its owned string.
fn c_format(value: f64, width: RuntimeDType) -> String {
    let rendered = chelis_string_from_scalar(scalar_from_f64(value, width));
    unsafe {
        let text = std::ffi::CStr::from_ptr(chelis_string_data(rendered))
            .to_str()
            .expect("scalar rendering is UTF-8")
            .to_string();
        chelis_string_release(rendered);
        text
    }
}

// ---------------------------------------------------------------------------
// The harness's frozen value tables (observation_roundtrip_harness.rs).
// ---------------------------------------------------------------------------

const F64_TABLE: &[f64] = &[
    f64::MAX,
    5e-324,
    2.2250738585072014e-308,
    -0.0,
    0.1,
    0.30000000000000004,
    9007199254740992.0,
    9007199254740994.0,
    9.999999980506448e19,
    -1.5,
];

const F32_TABLE: &[f32] = &[
    f32::MAX,
    1e-45,
    f32::MIN_POSITIVE,
    -0.0,
    0.1,
    16777216.0,
    16777218.0,
    2049.0,
    -1.5,
];

const F16_TABLE: &[f64] = &[
    65504.0,
    0.00006103515625,
    5.960464477539063e-8,
    -0.0,
    0.75,
    2048.0,
    2050.0,
    -1.5,
];

const BF16_TABLE: &[f64] = &[
    3.3895313892515355e38,
    1.1754943508222875e-38,
    9.183549615799121e-41,
    -0.0,
    0.75,
    256.0,
    258.0,
    -1.5,
];

/// The section 8.1 grammar edge set: threshold straddles, exponent forms,
/// integral decimals, specials.
const GRAMMAR_EDGES: &[f64] = &[
    1e-4,
    9.999999999999999e-5,
    1e-5,
    1e16,
    9999999999999998.0,
    1e20,
    1.5e-7,
    5e-324,
    2048.0,
    -2048.0,
    0.5,
    123.456,
    f64::INFINITY,
    f64::NEG_INFINITY,
    f64::NAN,
    -0.0,
    0.0,
];

#[test]
fn f64_table_and_edges_match_reference_bytes() {
    for &v in F64_TABLE.iter().chain(GRAMMAR_EDGES) {
        assert_eq!(
            c_format(v, RuntimeDType::F64),
            format_element(Prim::F64, ElementRef::F64(v)),
            "f64 value {v:?} ({:#018x})",
            v.to_bits()
        );
    }
}

#[test]
fn f32_table_and_edges_match_reference_bytes() {
    let edges = GRAMMAR_EDGES.iter().map(|&v| v as f32);
    for v in F32_TABLE.iter().copied().chain(edges) {
        assert_eq!(
            c_format(f64::from(v), RuntimeDType::F32),
            format_element(Prim::F32, ElementRef::F32(v)),
            "f32 value {v:?} ({:#010x})",
            v.to_bits()
        );
    }
}

#[test]
fn f16_table_matches_reference_bytes() {
    for &v in F16_TABLE {
        let h = half::f16::from_bits(storage_reference(v.to_bits(), 64, Output::F16));
        assert_eq!(
            c_format(f64::from(h), RuntimeDType::F16),
            format_element(Prim::F16, ElementRef::F16(h)),
            "f16 value {v:?}"
        );
    }
}

#[test]
fn bf16_table_matches_reference_bytes() {
    for &v in BF16_TABLE {
        let h = half::bf16::from_bits(storage_reference(v.to_bits(), 64, Output::Bf16));
        assert_eq!(
            c_format(f64::from(h), RuntimeDType::Bf16),
            format_element(Prim::Bf16, ElementRef::Bf16(h)),
            "bf16 value {v:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// Exhaustive half-format sweeps (the P1 open-question-1 discipline carried
// to the compiled lane: the narrow widths are fully enumerable, so no
// boundary-case debate survives).
// ---------------------------------------------------------------------------

#[test]
fn every_f16_bit_pattern_matches_reference_bytes() {
    for bits in 0..=u16::MAX {
        let h = half::f16::from_bits(bits);
        assert_eq!(
            c_format(f64::from(h), RuntimeDType::F16),
            format_element(Prim::F16, ElementRef::F16(h)),
            "f16 bits {bits:#06x}"
        );
    }
}

#[test]
fn every_bf16_bit_pattern_matches_reference_bytes() {
    for bits in 0..=u16::MAX {
        let h = half::bf16::from_bits(bits);
        assert_eq!(
            c_format(f64::from(h), RuntimeDType::Bf16),
            format_element(Prim::Bf16, ElementRef::Bf16(h)),
            "bf16 bits {bits:#06x}"
        );
    }
}

// ---------------------------------------------------------------------------
// Deterministic stratified bit sweeps for the wide widths. splitmix64 over
// a fixed seed; the exponent byte is forced through every value so
// subnormals, extremes, and every binade appear regardless of the stream.
// ---------------------------------------------------------------------------

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

#[test]
fn stratified_f32_bit_patterns_match_reference_bytes() {
    let mut state = 0x7332_0002u64;
    for i in 0..65536u32 {
        let raw = splitmix64(&mut state) as u32;
        // Force exponent stratum i % 256 into bits 23..31.
        let bits = (raw & 0x807F_FFFF) | ((i % 256) << 23);
        let v = f32::from_bits(bits);
        assert_eq!(
            c_format(f64::from(v), RuntimeDType::F32),
            format_element(Prim::F32, ElementRef::F32(v)),
            "f32 bits {bits:#010x}"
        );
    }
}

#[test]
fn stratified_f64_bit_patterns_match_reference_bytes() {
    let mut state = 0x7332_0001u64;
    for i in 0..65536u64 {
        let raw = splitmix64(&mut state);
        // Force exponent stratum i % 2048 into bits 52..62.
        let bits = (raw & 0x800F_FFFF_FFFF_FFFF) | ((i % 2048) << 52);
        let v = f64::from_bits(bits);
        assert_eq!(
            c_format(v, RuntimeDType::F64),
            format_element(Prim::F64, ElementRef::F64(v)),
            "f64 bits {bits:#018x}"
        );
    }
}
