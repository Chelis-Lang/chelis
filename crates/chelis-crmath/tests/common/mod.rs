//! Shared fixture reading and dispatch for the chelis-crmath integration tests.

#![allow(dead_code)]

use std::path::Path;

pub const FUNCTIONS: [&str; 9] = [
    "exp", "log", "sin", "cos", "tan", "atan", "tanh", "erf", "erfc",
];

/// One MPFR-derived fixture row: `function width input expected # note`.
#[derive(Clone, Debug)]
pub struct Row {
    pub function: String,
    pub width: u32,
    pub input: u64,
    pub expected: u64,
    pub note: String,
}

pub fn read_fixture(name: &str) -> Vec<Row> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let rows: Vec<Row> = text
        .lines()
        .filter(|line| !line.trim().is_empty() && !line.starts_with('#'))
        .map(|line| {
            let (data, note) = line.split_once('#').unwrap_or((line, ""));
            let fields: Vec<&str> = data.split_whitespace().collect();
            assert_eq!(fields.len(), 4, "malformed fixture row: {line}");
            let width = match fields[1] {
                "f32" => 32,
                "f64" => 64,
                other => panic!("unknown width {other} in: {line}"),
            };
            Row {
                function: fields[0].to_string(),
                width,
                input: u64::from_str_radix(fields[2], 16).unwrap(),
                expected: u64::from_str_radix(fields[3], 16).unwrap(),
                note: note.trim().to_string(),
            }
        })
        .collect();
    assert!(!rows.is_empty(), "{} has no rows", path.display());
    rows
}

pub fn f32_kernel(function: &str) -> fn(f32) -> f32 {
    match function {
        "exp" => chelis_crmath::exp_f32,
        "log" => chelis_crmath::log_f32,
        "sin" => chelis_crmath::sin_f32,
        "cos" => chelis_crmath::cos_f32,
        "tan" => chelis_crmath::tan_f32,
        "atan" => chelis_crmath::atan_f32,
        "tanh" => chelis_crmath::tanh_f32,
        "erf" => chelis_crmath::erf_f32,
        "erfc" => chelis_crmath::erfc_f32,
        other => panic!("unknown function {other}"),
    }
}

pub fn f64_kernel(function: &str) -> fn(f64) -> f64 {
    match function {
        "exp" => chelis_crmath::exp_f64,
        "log" => chelis_crmath::log_f64,
        "sin" => chelis_crmath::sin_f64,
        "cos" => chelis_crmath::cos_f64,
        "tan" => chelis_crmath::tan_f64,
        "atan" => chelis_crmath::atan_f64,
        "tanh" => chelis_crmath::tanh_f64,
        "erf" => chelis_crmath::erf_f64,
        "erfc" => chelis_crmath::erfc_f64,
        other => panic!("unknown function {other}"),
    }
}

/// The API's result bits for a row's input.
pub fn api_bits(row: &Row) -> u64 {
    if row.width == 32 {
        let input = u32::try_from(row.input).expect("f32 input fits 32 bits");
        u64::from(f32_kernel(&row.function)(f32::from_bits(input)).to_bits())
    } else {
        f64_kernel(&row.function)(f64::from_bits(row.input)).to_bits()
    }
}

/// Rows where `compute` disagrees with the fixture, formatted for a failure message.
pub fn mismatches(rows: &[Row], compute: impl Fn(&Row) -> u64) -> Vec<String> {
    rows.iter()
        .filter_map(|row| {
            let got = compute(row);
            (got != row.expected).then(|| {
                format!(
                    "{} f{} input {:x}: got {got:x}, MPFR {:x} ({})",
                    row.function, row.width, row.input, row.expected, row.note
                )
            })
        })
        .collect()
}
