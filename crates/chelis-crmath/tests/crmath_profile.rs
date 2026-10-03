//! The profile obligation table (`chelis_crmath::profile`) in the Rust lanes, its
//! coverage, and the canary generated from it (chelis#2957).
//!
//! The C canary that `chelis build` runs compiles the same rows; these tests check that
//! the table is complete by category, that its expected bits are what IEEE arithmetic
//! in Rust and this crate's kernels compute, that each row is sensitive to the rewrite
//! it witnesses, and that the canary reports every planted error.

use chelis_crmath::profile::{
    Class, Obligation, Output, PRIMITIVES, Row, canary_driver, canary_input, canary_mismatches,
    canary_rows, describe, rows, storage_midpoint_inputs, storage_reference,
};

const CANONICAL_F32: u32 = 0x7fc0_0000;
const CANONICAL_F64: u64 = 0x7ff8_0000_0000_0000;

fn f32_operand(row: &Row, index: usize) -> f32 {
    f32::from_bits(u32::try_from(row.operands[index]).expect("f32 operand fits 32 bits"))
}

fn f64_operand(row: &Row, index: usize) -> f64 {
    f64::from_bits(row.operands[index])
}

fn finalize_f32(value: f32) -> u64 {
    u64::from(if value.is_nan() {
        CANONICAL_F32
    } else {
        value.to_bits()
    })
}

fn finalize_f64(value: f64) -> u64 {
    if value.is_nan() {
        CANONICAL_F64
    } else {
        value.to_bits()
    }
}

/// The row's result in this crate's Rust lane, or `None` for a primitive whose
/// production Rust lane is the runtime's (the f16 and bf16 storage conversions, which
/// `chelis-runtime`'s tests check against these rows). `raw` skips NaN finalization.
// `a - a` and `a == a` are the expression shapes the rows witness.
#[allow(clippy::eq_op)]
fn rust_lane(row: &Row, raw: bool) -> Option<u64> {
    let name = row.primitive.name;
    let fin32 = |value: f32| {
        if raw {
            u64::from(value.to_bits())
        } else {
            finalize_f32(value)
        }
    };
    let fin64 = |value: f64| {
        if raw {
            value.to_bits()
        } else {
            finalize_f64(value)
        }
    };
    if matches!(name, "to_f16" | "to_bf16") {
        return None;
    }
    if row.primitive.width == 32 {
        let a = f32_operand(row, 0);
        let b = || f32_operand(row, 1);
        let c = || f32_operand(row, 2);
        Some(match name {
            "add" => fin32(a + b()),
            "sub" => fin32(a - b()),
            "mul" => fin32(a * b()),
            "div" => fin32(a / b()),
            "sqrt" => fin32(a.sqrt()),
            "fma" => fin32(a.mul_add(b(), c())),
            "eq" => u64::from(a == b()),
            "ne" => u64::from(a != b()),
            "lt" => u64::from(a < b()),
            "mul_add" => fin32(a * b() + c()),
            "add_sub" => fin32((a + b()) - a),
            "div_three" => fin32(a / 3.0),
            "add_zero" => fin32(a + 0.0),
            "sub_self" => fin32(a - a),
            "mul_zero" => fin32(a * 0.0),
            "self_eq" => u64::from(a == a),
            "gt_max" => u64::from(a > f32::MAX),
            "widen" => fin64(f64::from(a)),
            "exp" => u64::from(chelis_crmath::exp_f32(a).to_bits()),
            "log" => u64::from(chelis_crmath::log_f32(a).to_bits()),
            "sin" => u64::from(chelis_crmath::sin_f32(a).to_bits()),
            "cos" => u64::from(chelis_crmath::cos_f32(a).to_bits()),
            "tan" => u64::from(chelis_crmath::tan_f32(a).to_bits()),
            "atan" => u64::from(chelis_crmath::atan_f32(a).to_bits()),
            "tanh" => u64::from(chelis_crmath::tanh_f32(a).to_bits()),
            other => panic!("no Rust lane for {other} f32"),
        })
    } else {
        let a = f64_operand(row, 0);
        let b = || f64_operand(row, 1);
        let c = || f64_operand(row, 2);
        Some(match name {
            "add" => fin64(a + b()),
            "sub" => fin64(a - b()),
            "mul" => fin64(a * b()),
            "div" => fin64(a / b()),
            "sqrt" => fin64(a.sqrt()),
            "fma" => fin64(a.mul_add(b(), c())),
            "eq" => u64::from(a == b()),
            "ne" => u64::from(a != b()),
            "lt" => u64::from(a < b()),
            "mul_add" => fin64(a * b() + c()),
            "add_sub" => fin64((a + b()) - a),
            "div_three" => fin64(a / 3.0),
            "add_zero" => fin64(a + 0.0),
            "sub_self" => fin64(a - a),
            "mul_zero" => fin64(a * 0.0),
            "self_eq" => u64::from(a == a),
            "gt_max" => u64::from(a > f64::MAX),
            #[allow(clippy::cast_possible_truncation)]
            "narrow" => fin32(a as f32),
            "exp" => chelis_crmath::exp_f64(a).to_bits(),
            "log" => chelis_crmath::log_f64(a).to_bits(),
            "sin" => chelis_crmath::sin_f64(a).to_bits(),
            "cos" => chelis_crmath::cos_f64(a).to_bits(),
            "tan" => chelis_crmath::tan_f64(a).to_bits(),
            "atan" => chelis_crmath::atan_f64(a).to_bits(),
            "tanh" => chelis_crmath::tanh_f64(a).to_bits(),
            other => panic!("no Rust lane for {other} f64"),
        })
    }
}

#[test]
fn rust_lane_matches_every_row_it_computes() {
    let mut checked = 0;
    let bad: Vec<String> = rows()
        .iter()
        .filter_map(|row| {
            let got = rust_lane(row, false)?;
            checked += 1;
            (got != row.expected)
                .then(|| format!("{row}: Rust gives 0x{got:x}, table 0x{:x}", row.expected))
        })
        .collect();
    assert!(
        bad.is_empty(),
        "{} rows differ:\n{}",
        bad.len(),
        bad.join("\n")
    );
    assert!(checked > 2000, "only {checked} rows checked");
}

#[test]
fn every_primitive_covers_its_value_classes() {
    let mut missing = Vec::new();
    for primitive in PRIMITIVES {
        let own: Vec<&Row> = rows()
            .iter()
            .filter(|row| std::ptr::eq(row.primitive, primitive))
            .collect();
        if own.is_empty() {
            missing.push(format!("{primitive}: no rows"));
        }
        for &class in primitive.classes {
            if !own.iter().any(|row| row.covers(class)) {
                missing.push(format!("{primitive}: no {class:?} row"));
            }
        }
    }
    assert!(missing.is_empty(), "coverage gaps:\n{}", missing.join("\n"));
}

#[test]
fn every_obligation_has_a_canary_row() {
    for obligation in Obligation::ALL {
        assert!(
            canary_rows().any(|row| row.obligation == obligation),
            "no C canary row witnesses {obligation}"
        );
    }
}

/// Each row is sensitive to the failure it witnesses: computing it the way the
/// forbidden rewrite would gives different bits.
#[test]
fn rows_detect_the_rewrite_they_witness() {
    let mut checked = 0;
    for row in rows() {
        let rewritten = match (row.obligation, row.primitive.name) {
            // A NaN passed through, payload and sign intact.
            (Obligation::CanonicalNan, _) => rust_lane(row, true),
            // Contraction: one rounding of a*b + c.
            (Obligation::NoContraction, "mul_add") if row.primitive.width == 64 => Some(
                f64_operand(row, 0)
                    .mul_add(f64_operand(row, 1), f64_operand(row, 2))
                    .to_bits(),
            ),
            (Obligation::NoContraction, "mul_add") => Some(u64::from(
                f32_operand(row, 0)
                    .mul_add(f32_operand(row, 1), f32_operand(row, 2))
                    .to_bits(),
            )),
            // Reassociation: (a + b) - a becomes b.
            (Obligation::NoValueChangingOptimization, "add_sub") => Some(row.operands[1]),
            // Reciprocal: a / 3 becomes a * (1/3).
            (Obligation::NoValueChangingOptimization, "div_three") if row.primitive.width == 64 => {
                Some((f64_operand(row, 0) * (1.0 / 3.0)).to_bits())
            }
            (Obligation::NoValueChangingOptimization, "div_three") => {
                Some(u64::from((f32_operand(row, 0) * (1.0_f32 / 3.0)).to_bits()))
            }
            _ => None,
        };
        let Some(rewritten) = rewritten else { continue };
        checked += 1;
        assert_ne!(rewritten, row.expected, "{row} does not detect its rewrite");
    }
    assert!(checked >= 40, "only {checked} rows checked");
}

#[test]
fn canary_driver_dispatches_every_c_primitive() {
    let driver = canary_driver();
    for primitive in PRIMITIVES {
        let branch = format!(
            "strcmp(name, \"{}\") == 0 && width == {}",
            primitive.name, primitive.width
        );
        assert_eq!(
            driver.matches(&branch).count(),
            usize::from(primitive.c.is_some()),
            "{primitive}"
        );
        if let Some(expression) = primitive.c {
            assert!(driver.contains(expression), "{primitive}: `{expression}`");
        }
        assert_eq!(
            primitive.c.is_none(),
            matches!(primitive.result, Output::F16 | Output::Bf16),
            "{primitive}: only the storage conversions have no C lane"
        );
    }
    assert_eq!(canary_driver(), driver, "generation is deterministic");
    assert_eq!(canary_input().lines().count(), canary_rows().count());
}

fn expected_output() -> String {
    canary_rows()
        .map(|row| format!("{:x}\n", row.expected))
        .collect()
}

#[test]
fn canary_comparison_accepts_the_table_and_reports_each_planted_error() {
    assert!(canary_mismatches(&expected_output()).unwrap().is_empty());
    let count = canary_rows().count();
    for (index, row) in canary_rows().enumerate().step_by(97) {
        let planted: String = canary_rows()
            .enumerate()
            .map(|(other, row)| {
                let bits = if other == index {
                    row.expected ^ 1
                } else {
                    row.expected
                };
                format!("{bits:x}\n")
            })
            .collect();
        let reported = canary_mismatches(&planted).unwrap();
        assert_eq!(reported.len(), 1, "{row}");
        assert!(std::ptr::eq(reported[0].row, row));
        let description = describe(&reported);
        assert!(
            description.starts_with(row.obligation.name()),
            "{description}"
        );
        assert!(description.contains(row.obligation.spec()), "{description}");
    }
    let short: String = expected_output()
        .lines()
        .skip(1)
        .map(|line| format!("{line}\n"))
        .collect();
    assert!(
        canary_mismatches(&short).is_err(),
        "a missing line is an error"
    );
    assert!(canary_mismatches(&format!("{}zz\n", expected_output())).is_err());
    assert!(count > 2000, "{count} canary rows");
}

#[test]
fn covers_reads_classes_from_bits() {
    let row = rows()
        .iter()
        .find(|row| row.primitive.name == "sub" && row.note == "inf - inf is invalid")
        .expect("an inf - inf row");
    assert!(row.covers(Class::Invalid));
    assert!(row.covers(Class::InfiniteOperand));
    assert!(!row.covers(Class::NanOperand));
    assert!(!row.covers(Class::SubnormalOperand));
}

/// The integer storage reference the exhaustive narrowing gates use agrees with MPFR
/// on every storage-conversion row.
#[test]
fn storage_reference_matches_every_conversion_row() {
    let mut checked = 0;
    for row in rows().iter().filter(|row| matches!(row.primitive.result, Output::F16 | Output::Bf16)) {
        let got = storage_reference(row.operands[0], row.primitive.width, row.primitive.result);
        assert_eq!(u64::from(got), row.expected, "{row}");
        checked += 1;
    }
    assert!(checked >= 60, "only {checked} rows");
}

/// At every midpoint the reference ties to the even neighbour, and one f64 ulp either
/// side decides the direction.
#[test]
fn storage_reference_ties_to_even_and_breaks_ties_by_one_ulp() {
    for output in [Output::F16, Output::Bf16] {
        let inputs = storage_midpoint_inputs(output);
        assert_eq!(inputs.len() % 10, 0);
        for group in inputs.chunks(10) {
            // middle, +ulp, -ulp, +2^-30, -2^-30, each followed by its negation.
            let code = |index: usize| storage_reference(group[index], 64, output);
            let (middle, above, below) = (code(0), code(2), code(4));
            assert_eq!(above.wrapping_sub(below), 1, "{output:?} {:x}", group[0]);
            assert!(middle == above || middle == below);
            assert_eq!(middle & 1, 0, "{output:?} midpoint {:x} ties to odd", group[0]);
            assert_eq!(code(6), above);
            assert_eq!(code(8), below);
            for index in (0..10).step_by(2) {
                assert_eq!(code(index + 1), code(index) | 0x8000, "sign of {:x}", group[index]);
            }
        }
    }
}
