//! [05-OP-6] `cast_trunc` sealed-kernel contract
//! (`spec/05-risc-primitives.md` §3.8; the chelis#759 ladder's
//! float-to-integer rung).
//!
//! One rule, exercised at every declared width of [04-NUM-8]:
//!
//! * finite source -> integer part truncated TOWARD ZERO, finalized at
//!   the target width;
//! * truncated integer outside the target range -> `Overflow` trap
//!   (never wraps, never saturates);
//! * `NaN` / `+-inf` -> `Domain` trap.
//!
//! Every positive row below has its negative twin: for each target width
//! the test pins the largest value that fits AND the smallest one that
//! does not, on both signs, from both an f32 and an f64 source.

use chelis_types::dtype_semantics::{
    NUMERIC_TRAP_DOMAIN_KIND, NUMERIC_TRAP_OVERFLOW_KIND, NumericTrap, RawScalar, RawTensor,
    ScalarValue, cast_trunc_raw, cast_trunc_scalar, cast_trunc_tensor, scalar_from_f64,
};
use chelis_types::types::Prim;

const INT_TARGETS: [Prim; 4] = [Prim::Int8, Prim::Int16, Prim::Int32, Prim::Int64];
const FLOAT_SOURCES: [Prim; 2] = [Prim::F32, Prim::F64];

fn trunc_f64(value: f64, target: Prim) -> Result<i64, NumericTrap> {
    cast_trunc_raw("cast_trunc", RawScalar::Float(value), target).map(|v| {
        v.as_i64_exact()
            .expect("an integer target stores an exact i64")
    })
}

/// Route a value through the SEALED scalar surface at `source`, so the
/// f32 rows actually read an f32 image rather than an f64 one.
fn trunc_from(source: Prim, value: f64, target: Prim) -> Result<i64, NumericTrap> {
    let sealed: ScalarValue = scalar_from_f64("test_ingress", source, value)
        .expect("a float target finalizes without trapping");
    cast_trunc_scalar("cast_trunc", sealed, target).map(|v| {
        v.as_i64_exact()
            .expect("an integer target stores an exact i64")
    })
}

// ---------------------------------------------------------------------
// Truncation toward zero
// ---------------------------------------------------------------------

#[test]
fn fractional_values_truncate_toward_zero_not_toward_negative_infinity() {
    for target in INT_TARGETS {
        for source in FLOAT_SOURCES {
            assert_eq!(
                trunc_from(source, 0.9, target),
                Ok(0),
                "{source:?}->{target:?}: 0.9 truncates to 0"
            );
            assert_eq!(
                trunc_from(source, -0.9, target),
                Ok(0),
                "{source:?}->{target:?}: -0.9 truncates TOWARD ZERO (0), \
                 not toward -inf (which would be -1)"
            );
            assert_eq!(
                trunc_from(source, 1.9, target),
                Ok(1),
                "{source:?}->{target:?}: 1.9 truncates to 1"
            );
            assert_eq!(
                trunc_from(source, -1.9, target),
                Ok(-1),
                "{source:?}->{target:?}: -1.9 truncates to -1, not -2"
            );
        }
    }
}

#[test]
fn negative_zero_truncates_to_zero() {
    for target in INT_TARGETS {
        for source in FLOAT_SOURCES {
            assert_eq!(
                trunc_from(source, -0.0, target),
                Ok(0),
                "{source:?}->{target:?}: -0.0 is finite and truncates to 0"
            );
        }
    }
}

#[test]
fn exactly_integral_values_pass_through_unchanged() {
    for target in INT_TARGETS {
        for source in FLOAT_SOURCES {
            for value in [0.0, 1.0, -1.0, 7.0, -7.0] {
                assert_eq!(
                    trunc_from(source, value, target),
                    Ok(value as i64),
                    "{source:?}->{target:?}: an already-integral {value} is unchanged"
                );
            }
        }
    }
}

/// The rung agrees with the checked default wherever the default is
/// defined: an integral in-range value yields the same integer.
#[test]
fn agrees_with_the_checked_default_on_integral_in_range_values() {
    for target in INT_TARGETS {
        for value in [0.0, 1.0, -1.0, 100.0, -100.0] {
            let checked = chelis_types::cast_raw("cast", RawScalar::Float(value), target)
                .expect("integral in-range values do not trap under the checked default");
            let truncating = cast_trunc_raw("cast_trunc", RawScalar::Float(value), target)
                .expect("nor under the truncating rung");
            assert_eq!(
                checked.as_i64_exact(),
                truncating.as_i64_exact(),
                "{target:?}: the two rungs differ only on the fractional case"
            );
        }
    }
}

// ---------------------------------------------------------------------
// Range: just-inside / just-outside, both signs, every width
// ---------------------------------------------------------------------

#[test]
fn largest_in_range_value_succeeds_and_smallest_out_of_range_traps_overflow() {
    for target in INT_TARGETS {
        let (lo, hi) = target
            .integer_range()
            .expect("every integer target declares a range");

        // Just inside, both ends. `i64::MAX` is NOT representable in
        // f64 -- it rounds up to 2^63, which is out of range -- so the
        // i64 row uses the largest f64 strictly below 2^63 instead.
        // This is the same boundary the checked default enforces.
        let in_max: f64 = if target == Prim::Int64 {
            9_223_372_036_854_774_784.0
        } else {
            hi as f64
        };
        assert_eq!(
            trunc_f64(in_max, target),
            Ok(in_max as i64),
            "{target:?}: the largest representable in-range value is in range"
        );
        assert_eq!(
            trunc_f64(lo as f64, target),
            Ok(lo),
            "{target:?}: the minimum is in range (-2^63 is exact in f64)"
        );

        // A fraction ABOVE the largest in-range value still truncates
        // INTO range: this is the whole point of truncating before the
        // width check, and the compiled lane must agree (it truncates
        // first too). i64 has no representable fractional neighbour at
        // that magnitude, so it uses a fractional value well inside.
        let (frac_high, frac_high_expected) = if target == Prim::Int64 {
            (
                4_611_686_018_427_387_904.5_f64,
                4_611_686_018_427_387_904_i64,
            )
        } else {
            (hi as f64 + 0.5, hi)
        };
        assert_eq!(
            trunc_f64(frac_high, target),
            Ok(frac_high_expected),
            "{target:?}: a fractional value above an integer truncates down to it, \
             it does not overflow"
        );
        let (frac_low, frac_low_expected) = if target == Prim::Int64 {
            (
                -4_611_686_018_427_387_904.5_f64,
                -4_611_686_018_427_387_904_i64,
            )
        } else {
            (lo as f64 - 0.5, lo)
        };
        assert_eq!(
            trunc_f64(frac_low, target),
            Ok(frac_low_expected),
            "{target:?}: and toward zero on the negative side too"
        );

        // Just outside, both ends. i64's neighbours are not
        // representable in f64, so step by the f64 ulp at that
        // magnitude (2^63 exactly) rather than by 1.
        let (over, under) = if target == Prim::Int64 {
            (9223372036854775808.0_f64, -9223372036854777856.0_f64)
        } else {
            (hi as f64 + 1.0, lo as f64 - 1.0)
        };
        assert_eq!(
            trunc_f64(over, target),
            Err(NumericTrap::Overflow {
                op: "cast_trunc",
                prim: target
            }),
            "{target:?}: {over} is past the maximum and must trap Overflow, \
             never wrap and never saturate"
        );
        assert_eq!(
            trunc_f64(under, target),
            Err(NumericTrap::Overflow {
                op: "cast_trunc",
                prim: target
            }),
            "{target:?}: {under} is past the minimum and must trap Overflow"
        );
    }
}

#[test]
fn huge_magnitudes_trap_overflow_at_every_width() {
    for target in INT_TARGETS {
        for source in FLOAT_SOURCES {
            for value in [1e30, -1e30] {
                assert_eq!(
                    trunc_from(source, value, target),
                    Err(NumericTrap::Overflow {
                        op: "cast_trunc",
                        prim: target
                    }),
                    "{source:?}->{target:?}: {value} must Overflow-trap"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------
// Non-finite sources: Domain, never Overflow, never a silent value
// ---------------------------------------------------------------------

#[test]
fn nan_and_infinities_trap_domain_at_every_width() {
    for target in INT_TARGETS {
        for source in FLOAT_SOURCES {
            for (value, label) in [
                (f64::NAN, "NaN"),
                (f64::INFINITY, "+inf"),
                (f64::NEG_INFINITY, "-inf"),
            ] {
                assert_eq!(
                    trunc_from(source, value, target),
                    Err(NumericTrap::Domain {
                        op: "cast_trunc",
                        prim: target
                    }),
                    "{source:?}->{target:?}: {label} has no integer meaning \
                     and must trap Domain, not Overflow"
                );
            }
        }
    }
}

#[test]
fn trap_messages_carry_the_cast_trunc_brand() {
    let domain = trunc_f64(f64::NAN, Prim::Int32).expect_err("NaN traps");
    assert_eq!(
        domain.to_string(),
        format!("numeric trap: {NUMERIC_TRAP_DOMAIN_KIND} in cast_trunc at i32")
    );
    let overflow = trunc_f64(1e30, Prim::Int32).expect_err("1e30 traps");
    assert_eq!(
        overflow.to_string(),
        format!("numeric trap: {NUMERIC_TRAP_OVERFLOW_KIND} in cast_trunc at i32")
    );
}

// ---------------------------------------------------------------------
// Tensor surface: identical semantics, first offending element traps
// ---------------------------------------------------------------------

#[test]
fn tensor_surface_matches_the_scalar_surface_element_for_element() {
    let values = vec![0.9, -0.9, -0.0, 3.0, -3.0, 127.0];
    for target in INT_TARGETS {
        let storage = cast_trunc_tensor("cast_trunc", RawTensor::Float(values.clone()), target)
            .expect("every element is in range for every target");
        let actual = match storage.to_raw() {
            RawTensor::Int(v) => v,
            RawTensor::Float(_) => panic!("{target:?}: an integer target stores integers"),
        };
        let expected: Vec<i64> = values
            .iter()
            .map(|v| trunc_f64(*v, target).expect("scalar surface agrees"))
            .collect();
        assert_eq!(
            actual, expected,
            "{target:?}: tensor == scalar, elementwise"
        );
    }
}

#[test]
fn tensor_surface_traps_on_the_first_offending_element() {
    assert_eq!(
        cast_trunc_tensor(
            "cast_trunc",
            RawTensor::Float(vec![1.5, 1e30, 2.5]),
            Prim::Int32
        )
        .expect_err("the 1e30 element is out of range"),
        NumericTrap::Overflow {
            op: "cast_trunc",
            prim: Prim::Int32
        }
    );
    assert_eq!(
        cast_trunc_tensor(
            "cast_trunc",
            RawTensor::Float(vec![1.5, f64::NAN, 2.5]),
            Prim::Int32
        )
        .expect_err("the NaN element has no integer meaning"),
        NumericTrap::Domain {
            op: "cast_trunc",
            prim: Prim::Int32
        }
    );
}

/// chelis#759 MEDIUM-1: a buffer carrying BOTH an out-of-range and a
/// non-finite element must report the kind belonging to whichever comes
/// FIRST, because the compiled C lane is a per-element loop and
/// [05-OP-6] declares the lanes identical.
///
/// The narrow-width rows are the ones that actually bite: the bulk
/// `finalize_tensor` path domain-checks the whole buffer before it
/// width-checks any of it, so `[300.9, NaN] -> i8` reported `Domain`
/// there while C reports `Overflow` at element 0. i64 alone is immune
/// (both offenders are caught in the same pass), so testing only i32
/// or i64 would have missed the divergence.
#[test]
fn mixed_offender_buffers_report_the_first_offender_in_order() {
    let cases: [(&str, Vec<f64>, Prim, NumericTrap); 6] = [
        // Out-of-range first, then non-finite.
        (
            "[1e30, NaN] -> i32",
            vec![1e30, f64::NAN],
            Prim::Int32,
            NumericTrap::Overflow {
                op: "cast_trunc",
                prim: Prim::Int32,
            },
        ),
        (
            "[300.9, NaN] -> i8",
            vec![300.9, f64::NAN],
            Prim::Int8,
            NumericTrap::Overflow {
                op: "cast_trunc",
                prim: Prim::Int8,
            },
        ),
        (
            "[40000.5, inf] -> i16",
            vec![40000.5, f64::INFINITY],
            Prim::Int16,
            NumericTrap::Overflow {
                op: "cast_trunc",
                prim: Prim::Int16,
            },
        ),
        // Non-finite first, then out-of-range.
        (
            "[NaN, 1e30] -> i32",
            vec![f64::NAN, 1e30],
            Prim::Int32,
            NumericTrap::Domain {
                op: "cast_trunc",
                prim: Prim::Int32,
            },
        ),
        (
            "[NaN, 300.9] -> i8",
            vec![f64::NAN, 300.9],
            Prim::Int8,
            NumericTrap::Domain {
                op: "cast_trunc",
                prim: Prim::Int8,
            },
        ),
        (
            "[-inf, 40000.5] -> i16",
            vec![f64::NEG_INFINITY, 40000.5],
            Prim::Int16,
            NumericTrap::Domain {
                op: "cast_trunc",
                prim: Prim::Int16,
            },
        ),
    ];
    for (label, values, target, expected) in cases {
        assert_eq!(
            cast_trunc_tensor("cast_trunc", RawTensor::Float(values), target)
                .expect_err("both elements offend; one of them must trap"),
            expected,
            "{label}: the FIRST offending element decides the trap kind"
        );
    }
}

// ---------------------------------------------------------------------
// Contract violations the checker is supposed to make unreachable
// ---------------------------------------------------------------------

#[test]
#[should_panic(expected = "is not an integer target")]
fn a_float_target_is_not_a_truncating_cast() {
    let _ = cast_trunc_raw("cast_trunc", RawScalar::Float(1.9), Prim::F32);
}

#[test]
#[should_panic(expected = "is not an integer target")]
fn a_bool_target_is_not_a_truncating_cast() {
    let _ = cast_trunc_raw("cast_trunc", RawScalar::Float(1.0), Prim::Bool);
}

#[test]
#[should_panic(expected = "float-to-integer only")]
fn an_integer_source_has_no_truncating_cast() {
    let _ = cast_trunc_raw("cast_trunc", RawScalar::Int(3), Prim::Int32);
}

#[test]
#[should_panic(expected = "is not a float source")]
fn a_sealed_integer_scalar_has_no_truncating_cast() {
    let sealed = chelis_types::scalar_from_i64("test_ingress", Prim::Int32, 3)
        .expect("an in-range i32 ingress");
    let _ = cast_trunc_scalar("cast_trunc", sealed, Prim::Int64);
}
