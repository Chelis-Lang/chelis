//! [05-OP-23] `cast_saturate` and [05-OP-24] `cast_wrap` sealed-kernel
//! contract (chelis#759, the named lossy cast ladder).
//!
//! `cast_saturate` truncates a finite float toward zero and clamps the
//! integer to the target range, clamps an integer source directly, maps
//! `-inf`/`+inf` to the target minimum/maximum, traps `Domain` on NaN, and
//! never traps `Overflow`. `cast_wrap` returns the target-width two's
//! complement representative of a signed integer and never traps.
//!
//! The expected values come from an independent oracle over `i128`, so a
//! kernel that shares a mistake with the oracle is not the one under test.

use chelis_deep::NamedCastMode;
use chelis_types::dtype_semantics::{
    NumericTrap, RawScalar, RawTensor, named_cast_raw, named_cast_scalar, named_cast_tensor,
    scalar_from_f64, scalar_from_i64,
};
use chelis_types::types::Prim;

const INT_TARGETS: [Prim; 4] = [Prim::Int8, Prim::Int16, Prim::Int32, Prim::Int64];
const INT_SOURCES: [Prim; 4] = INT_TARGETS;
const FLOAT_SOURCES: [Prim; 4] = [Prim::F16, Prim::Bf16, Prim::F32, Prim::F64];

fn range(prim: Prim) -> (i128, i128) {
    let (min, max) = prim.integer_range().expect("an integer dtype");
    (i128::from(min), i128::from(max))
}

/// The edge values of an integer source: its extremes, their neighbours,
/// and every narrower target's extremes and their neighbours.
fn integer_edges(source: Prim) -> Vec<i64> {
    let (min, max) = range(source);
    let mut values: Vec<i128> = vec![min, min + 1, -1, 0, 1, max - 1, max];
    for target in INT_TARGETS {
        let (lo, hi) = range(target);
        values.extend([lo - 1, lo, lo + 1, hi - 1, hi, hi + 1]);
    }
    let mut values: Vec<i64> = values
        .into_iter()
        .filter(|value| (min..=max).contains(value))
        .map(|value| i64::try_from(value).expect("in the source range"))
        .collect();
    values.sort_unstable();
    values.dedup();
    values
}

fn saturate_oracle(value: i128, target: Prim) -> i64 {
    let (lo, hi) = range(target);
    i64::try_from(value.clamp(lo, hi)).expect("clamped into i64")
}

fn wrap_oracle(value: i128, target: Prim) -> i64 {
    let bits = match target {
        Prim::Int8 => 8,
        Prim::Int16 => 16,
        Prim::Int32 => 32,
        _ => 64,
    };
    let modulus = 1i128 << bits;
    let mut residue = value.rem_euclid(modulus);
    if residue >= modulus / 2 {
        residue -= modulus;
    }
    i64::try_from(residue).expect("a target-width representative")
}

fn exact(value: chelis_types::dtype_semantics::ScalarValue, target: Prim) -> i64 {
    assert_eq!(value.prim(), target, "the result is at the target dtype");
    value
        .as_i64_exact()
        .expect("an integer target stores an exact i64")
}

fn scalar_int(mode: NamedCastMode, source: Prim, value: i64, target: Prim) -> i64 {
    let sealed = scalar_from_i64("test_ingress", source, value).expect("in the source range");
    exact(
        named_cast_scalar(mode, sealed, target)
            .unwrap_or_else(|trap| panic!("{mode:?} {source:?}->{target:?} {value}: {trap}")),
        target,
    )
}

fn scalar_float(source: Prim, value: f64, target: Prim) -> Result<i64, NumericTrap> {
    let sealed = scalar_from_f64("test_ingress", source, value).expect("a float ingress");
    named_cast_scalar(NamedCastMode::Saturate, sealed, target).map(|v| exact(v, target))
}

// ---------------------------------------------------------------------
// cast_saturate: integer sources
// ---------------------------------------------------------------------

#[test]
fn saturate_clamps_every_integer_pair_at_its_edges() {
    for source in INT_SOURCES {
        for target in INT_TARGETS {
            for value in integer_edges(source) {
                assert_eq!(
                    scalar_int(NamedCastMode::Saturate, source, value, target),
                    saturate_oracle(i128::from(value), target),
                    "cast_saturate({value}: {source:?}, {target:?})"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------
// cast_saturate: float sources
// ---------------------------------------------------------------------

#[test]
fn saturate_truncates_a_finite_float_toward_zero_then_clamps() {
    // Every value is exact in f16, bf16, f32 and f64.
    let values: [f64; 13] = [
        -256.0, -129.0, -128.0, -127.5, -1.5, -0.5, -0.0, 0.0, 0.5, 1.5, 127.5, 128.0, 256.0,
    ];
    for source in FLOAT_SOURCES {
        for target in INT_TARGETS {
            for value in values {
                let want = saturate_oracle(value.trunc() as i128, target);
                assert_eq!(
                    scalar_float(source, value, target),
                    Ok(want),
                    "cast_saturate({value}: {source:?}, {target:?})"
                );
            }
        }
    }
}

#[test]
fn saturate_maps_values_beyond_every_target_to_its_extremes() {
    for target in INT_TARGETS {
        let (lo, hi) = range(target);
        let (lo, hi) = (i64::try_from(lo).unwrap(), i64::try_from(hi).unwrap());
        for source in [Prim::F32, Prim::F64, Prim::Bf16] {
            assert_eq!(scalar_float(source, 1e30, target), Ok(hi), "{source:?}");
            assert_eq!(scalar_float(source, -1e30, target), Ok(lo), "{source:?}");
        }
        // 2^63 is the first f64 above i64::MAX; -2^63 is i64::MIN exactly.
        assert_eq!(
            scalar_float(Prim::F64, 9223372036854775808.0, target),
            Ok(hi)
        );
        assert_eq!(
            scalar_float(Prim::F64, -9223372036854775808.0, target),
            Ok(lo)
        );
    }
    // The largest f64 below 2^63 is in range and is not clamped.
    assert_eq!(
        scalar_float(Prim::F64, 9223372036854774784.0, Prim::Int64),
        Ok(9223372036854774784)
    );
}

#[test]
fn saturate_maps_infinities_to_the_target_extremes() {
    for source in FLOAT_SOURCES {
        for target in INT_TARGETS {
            let (lo, hi) = range(target);
            assert_eq!(
                scalar_float(source, f64::INFINITY, target),
                Ok(i64::try_from(hi).unwrap())
            );
            assert_eq!(
                scalar_float(source, f64::NEG_INFINITY, target),
                Ok(i64::try_from(lo).unwrap())
            );
        }
    }
}

/// The negative twin of the clamping rows: NaN has no integer image, so it
/// traps `Domain` at the target dtype as operation `cast_saturate`.
#[test]
fn saturate_traps_domain_on_nan_and_never_overflow() {
    for source in FLOAT_SOURCES {
        for target in INT_TARGETS {
            let trap = scalar_float(source, f64::NAN, target).expect_err("NaN traps");
            assert_eq!(
                trap,
                NumericTrap::Domain {
                    op: "cast_saturate",
                    prim: target
                }
            );
        }
    }
    let trap = named_cast_raw(
        NamedCastMode::Saturate,
        RawScalar::Float(f64::NAN),
        Prim::Int8,
    )
    .expect_err("NaN traps on the raw surface too");
    assert_eq!(
        trap.to_string(),
        "numeric trap: domain in cast_saturate at i8"
    );
}

#[test]
fn saturate_tensor_traps_on_the_first_nan_and_clamps_the_rest() {
    let clamped = named_cast_tensor(
        NamedCastMode::Saturate,
        RawTensor::Float(vec![
            f64::NEG_INFINITY,
            -300.5,
            -1.5,
            1.5,
            300.5,
            f64::INFINITY,
        ]),
        Prim::Int8,
    )
    .expect("no NaN, so nothing traps");
    let got: Vec<i64> = (0..6)
        .map(|i| clamped.scalar_at(i).as_i64_exact().unwrap())
        .collect();
    assert_eq!(got, vec![-128, -128, -1, 1, 127, 127]);
    let trap = named_cast_tensor(
        NamedCastMode::Saturate,
        RawTensor::Float(vec![1e30, f64::NAN]),
        Prim::Int8,
    )
    .expect_err("the NaN element traps");
    assert_eq!(
        trap.to_string(),
        "numeric trap: domain in cast_saturate at i8"
    );
}

// ---------------------------------------------------------------------
// cast_wrap
// ---------------------------------------------------------------------

#[test]
fn wrap_returns_the_twos_complement_representative_for_every_pair() {
    for source in INT_SOURCES {
        for target in INT_TARGETS {
            for value in integer_edges(source) {
                assert_eq!(
                    scalar_int(NamedCastMode::Wrap, source, value, target),
                    wrap_oracle(i128::from(value), target),
                    "cast_wrap({value}: {source:?}, {target:?})"
                );
            }
        }
    }
}

#[test]
fn wrap_tensor_never_traps() {
    let wrapped = named_cast_tensor(
        NamedCastMode::Wrap,
        RawTensor::Int(vec![i64::MIN, -129, -128, 127, 128, 255, 256, i64::MAX]),
        Prim::Int8,
    )
    .expect("cast_wrap never traps");
    let got: Vec<i64> = (0..8)
        .map(|i| wrapped.scalar_at(i).as_i64_exact().unwrap())
        .collect();
    assert_eq!(got, vec![0, 127, -128, 127, -128, -1, 0, -1]);
}
