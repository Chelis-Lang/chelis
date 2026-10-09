//! [05-OP-79] `pow` against MPFR on CORE-MATH's hardest-to-round pairs, and its
//! f16/bf16 composition.
//!
//! `tests/fixtures/pow_worst_cases.txt` samples both widths' upstream worst-case
//! corpora, each pair also with its base negated, so the rows cover correct rounding
//! near rounding boundaries, the NaN of a negative base with a non-integer exponent,
//! and the signed power of a negative base with an integer exponent. The IEEE special
//! cases are rows of `profile_obligations.txt`, which `crmath_profile` checks.

use std::path::Path;

use half::{bf16, f16};

struct Row {
    width: u32,
    x: u64,
    y: u64,
    expected: u64,
    note: String,
}

fn rows() -> Vec<Row> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pow_worst_cases.txt");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    text.lines()
        .filter(|line| !line.trim().is_empty() && !line.starts_with('#'))
        .map(|line| {
            let (data, note) = line.split_once('#').unwrap_or((line, ""));
            let fields: Vec<&str> = data.split_whitespace().collect();
            assert_eq!(fields.len(), 5, "malformed fixture row: {line}");
            assert_eq!(fields[0], "pow", "malformed fixture row: {line}");
            let hex = |field: &str| u64::from_str_radix(field, 16).unwrap();
            Row {
                width: match fields[1] {
                    "f32" => 32,
                    "f64" => 64,
                    other => panic!("unknown width {other} in: {line}"),
                },
                x: hex(fields[2]),
                y: hex(fields[3]),
                expected: hex(fields[4]),
                note: note.trim().to_string(),
            }
        })
        .collect()
}

fn f32_bits(bits: u64) -> f32 {
    f32::from_bits(u32::try_from(bits).expect("f32 operand fits 32 bits"))
}

#[test]
fn pow_matches_mpfr_on_upstream_worst_cases() {
    let rows = rows();
    let mut bad = Vec::new();
    let (mut f32_rows, mut f64_rows, mut nan_rows, mut negative_rows) = (0, 0, 0, 0);
    for row in &rows {
        let got = if row.width == 32 {
            f32_rows += 1;
            u64::from(chelis_crmath::pow_f32(f32_bits(row.x), f32_bits(row.y)).to_bits())
        } else {
            f64_rows += 1;
            chelis_crmath::pow_f64(f64::from_bits(row.x), f64::from_bits(row.y)).to_bits()
        };
        let sign = 1u64 << (row.width - 1);
        if row.expected == if row.width == 32 { 0x7fc0_0000 } else { 0x7ff8_0000_0000_0000 } {
            nan_rows += 1;
        } else if row.expected & sign != 0 {
            negative_rows += 1;
        }
        if got != row.expected {
            bad.push(format!(
                "pow f{} ({:#x}, {:#x}): got {got:#x}, MPFR {:#x} ({})",
                row.width, row.x, row.y, row.expected, row.note
            ));
        }
    }
    assert!(bad.is_empty(), "{} rows differ:\n{}", bad.len(), bad.join("\n"));
    // The sample is not vacuous at either width, and it reaches both the invalid
    // negative-base case and a negative signed power.
    assert!(f32_rows >= 150 && f64_rows >= 150, "{f32_rows} f32 and {f64_rows} f64 rows");
    assert!(nan_rows > 0 && negative_rows > 0, "{nan_rows} NaN and {negative_rows} negative rows");
}

/// f16 and bf16 widen both operands exactly to f32, take the correctly rounded f32
/// power, and finalize once ([05-OP-79]); the composition is not a direct rounding to
/// the storage width.
#[test]
fn half_widths_compose_through_the_f32_power() {
    let operands = [
        (-3.0_f32, 3.0_f32),
        (-2.0, 0.5),
        (0.0, 0.0),
        (-0.0, -3.0),
        (1.5, 2.5),
        (0.1, 7.0),
        (65504.0, 0.5),
        (2.0, 20.0),
        (2.0, -30.0),
    ];
    for (x, y) in operands {
        let (hx, hy) = (f16::from_f32(x), f16::from_f32(y));
        let expect16 = f16::from_f32(chelis_crmath::pow_f32(hx.to_f32(), hy.to_f32()));
        let got16 = chelis_crmath::pow_f16(hx, hy);
        assert_eq!(got16.to_bits(), expect16.to_bits(), "f16 pow({x}, {y})");
        let (bx, by) = (bf16::from_f32(x), bf16::from_f32(y));
        let expect_bf = bf16::from_f32(chelis_crmath::pow_f32(bx.to_f32(), by.to_f32()));
        let got_bf = chelis_crmath::pow_bf16(bx, by);
        assert_eq!(got_bf.to_bits(), expect_bf.to_bits(), "bf16 pow({x}, {y})");
    }
    assert_eq!(chelis_crmath::pow_f16(f16::from_f32(-3.0), f16::from_f32(3.0)).to_f32(), -27.0);
    assert_eq!(chelis_crmath::pow_f16(f16::from_f32(2.0), f16::from_f32(20.0)), f16::INFINITY);
    assert!(chelis_crmath::pow_bf16(bf16::from_f32(-2.0), bf16::from_f32(0.5)).is_nan());
}

/// A signaling NaN is invalid at f32 and f64 even where a quiet NaN gives 1, while an
/// f16 or bf16 signaling NaN is quieted by its widening and so takes the quiet rule.
#[test]
fn signaling_nan_exceptions_follow_the_operand_width() {
    let snan32 = f32::from_bits(0x7f80_0001);
    let snan64 = f64::from_bits(0x7ff0_0000_0000_0001);
    assert!(chelis_crmath::pow_f32(snan32, 0.0).is_nan());
    assert!(chelis_crmath::pow_f32(1.0, snan32).is_nan());
    assert!(chelis_crmath::pow_f64(snan64, -0.0).is_nan());
    assert!(chelis_crmath::pow_f64(1.0, snan64).is_nan());
    assert_eq!(chelis_crmath::pow_f32(f32::NAN, 0.0), 1.0);
    assert_eq!(chelis_crmath::pow_f64(1.0, f64::NAN), 1.0);
    let snan16 = f16::from_bits(0x7c01);
    let snan_bf = bf16::from_bits(0x7f81);
    assert_eq!(chelis_crmath::pow_f16(snan16, f16::ZERO), f16::ONE);
    assert_eq!(chelis_crmath::pow_f16(f16::ONE, snan16), f16::ONE);
    assert_eq!(chelis_crmath::pow_bf16(snan_bf, bf16::ZERO), bf16::ONE);
    assert_eq!(chelis_crmath::pow_bf16(bf16::ONE, snan_bf), bf16::ONE);
}
