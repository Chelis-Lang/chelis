//! chelis#2870: `to_int` and `to_float` follow [05-OP-59] in both lanes.
//!
//! One program applies the parsers to every case below, reading the longest
//! spellings from files. The interpreter runs it, `chelis build` compiles and
//! runs it, and the two lanes' stdout must be byte-equal. Each lane's printed
//! result must also equal what the atom requires. Printing renders an f64 through the shortest round-trip channel
//! ([05-OBS-1]), so decoding the printed text recovers the exact f64: finite
//! values and the sign of zero are compared bit for bit, infinities by sign,
//! and NaN by class, because the atom fixes no NaN payload or sign.
//!
//! The expected images were computed by an independent correctly rounded
//! parser over the atom's grammar after trimming Unicode White_Space. The
//! cases cover what separates [05-OP-59] from [05-OP-31]'s scalar-carrier
//! parse, which compiled C used before: IEEE overflow to a signed infinity,
//! Unicode whitespace trimming, and the signed case-insensitive `inf`,
//! `infinity`, and `nan` spellings, together with underflow, rounding
//! boundaries, and malformed text in both parsers.

#[path = "common/mod.rs"]
mod common;

use assert_cmd::Command;
use common::{build_and_run, gcc_available, write_file};
use std::path::{Path, PathBuf};
use tempfile::tempdir;

/// 2^1024 - 2^970: exactly halfway between f64::MAX and 2^1024, so ties to
/// even rounds it up to infinity.
const OVERFLOW_MIDPOINT: &str = "179769313486231580793728971405303415079934132710037826936173778980444968292764750946649017977587207096330286416692887910946555547851940402630657488671505820681908902000708383676273854845817711531764475730270069855571366959622842914819860834936475292719074168444365510704342711559699508093042880177904174497792";
/// One below [`OVERFLOW_MIDPOINT`], which rounds down to f64::MAX.
const BELOW_OVERFLOW_MIDPOINT: &str = "179769313486231580793728971405303415079934132710037826936173778980444968292764750946649017977587207096330286416692887910946555547851940402630657488671505820681908902000708383676273854845817711531764475730270069855571366959622842914819860834936475292719074168444365510704342711559699508093042880177904174497791";

#[derive(Clone, Copy, PartialEq)]
enum Float {
    Bits(u64),
    Nan,
    Malformed,
}

impl std::fmt::Debug for Float {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Float::Bits(bits) => write!(formatter, "Bits({bits:#018x})"),
            Float::Nan => formatter.write_str("Nan"),
            Float::Malformed => formatter.write_str("Malformed"),
        }
    }
}
use Float::{Bits, Malformed, Nan};

const TO_FLOAT_CASES: &[(&str, Float)] = &[
    ("1.5", Bits(0x3ff8_0000_0000_0000)),
    ("-2.25", Bits(0xc002_0000_0000_0000)),
    ("+0.5", Bits(0x3fe0_0000_0000_0000)),
    (".5", Bits(0x3fe0_0000_0000_0000)),
    ("5.", Bits(0x4014_0000_0000_0000)),
    ("1e3", Bits(0x408f_4000_0000_0000)),
    ("1E-3", Bits(0x3f50_624d_d2f1_a9fc)),
    ("+.5e+2", Bits(0x4049_0000_0000_0000)),
    ("007.50", Bits(0x401e_0000_0000_0000)),
    ("0", Bits(0x0000_0000_0000_0000)),
    ("-0", Bits(0x8000_0000_0000_0000)),
    ("-0.0", Bits(0x8000_0000_0000_0000)),
    ("0.1", Bits(0x3fb9_9999_9999_999a)),
    (
        "123456789012345678901234567890",
        Bits(0x45f8_ee90_ff6c_373e),
    ),
    ("4.9e-324", Bits(0x0000_0000_0000_0001)),
    ("2.4703282292062328e-324", Bits(0x0000_0000_0000_0001)),
    ("2.4703282292062327e-324", Bits(0x0000_0000_0000_0000)),
    ("1.7976931348623157e308", Bits(0x7fef_ffff_ffff_ffff)),
    ("1.7976931348623158e308", Bits(0x7fef_ffff_ffff_ffff)),
    ("1.7976931348623159e308", Bits(0x7ff0_0000_0000_0000)),
    ("1e400", Bits(0x7ff0_0000_0000_0000)),
    ("-1e400", Bits(0xfff0_0000_0000_0000)),
    ("1e99999999999999999999", Bits(0x7ff0_0000_0000_0000)),
    ("1e-400", Bits(0x0000_0000_0000_0000)),
    ("-1e-400", Bits(0x8000_0000_0000_0000)),
    ("-1e-99999999999999999999", Bits(0x8000_0000_0000_0000)),
    (" 1.5 ", Bits(0x3ff8_0000_0000_0000)),
    ("\t2.5\t", Bits(0x4004_0000_0000_0000)),
    ("\n3.5\n", Bits(0x400c_0000_0000_0000)),
    ("\r\n4.5\r\n", Bits(0x4012_0000_0000_0000)),
    ("\u{b}6.5\u{c}", Bits(0x401a_0000_0000_0000)),
    ("\u{a0}7.5\u{a0}", Bits(0x401e_0000_0000_0000)),
    ("\u{2003}8.5\u{2003}", Bits(0x4021_0000_0000_0000)),
    ("\u{3000}9.5", Bits(0x4023_0000_0000_0000)),
    ("\u{2028}-1.5\u{2029}", Bits(0xbff8_0000_0000_0000)),
    ("\u{85} 2.5", Bits(0x4004_0000_0000_0000)),
    ("inf", Bits(0x7ff0_0000_0000_0000)),
    ("INF", Bits(0x7ff0_0000_0000_0000)),
    ("Inf", Bits(0x7ff0_0000_0000_0000)),
    ("+inf", Bits(0x7ff0_0000_0000_0000)),
    ("-inf", Bits(0xfff0_0000_0000_0000)),
    ("infinity", Bits(0x7ff0_0000_0000_0000)),
    ("Infinity", Bits(0x7ff0_0000_0000_0000)),
    ("-INFINITY", Bits(0xfff0_0000_0000_0000)),
    ("+Infinity", Bits(0x7ff0_0000_0000_0000)),
    ("nan", Nan),
    ("NaN", Nan),
    ("NAN", Nan),
    ("+nan", Nan),
    ("-NaN", Nan),
    (" nan ", Nan),
    ("\u{a0}-inf\u{a0}", Bits(0xfff0_0000_0000_0000)),
    ("", Malformed),
    ("  ", Malformed),
    ("abc", Malformed),
    ("1.2.3", Malformed),
    ("1e", Malformed),
    ("e5", Malformed),
    (".", Malformed),
    ("+", Malformed),
    ("-", Malformed),
    ("+-1", Malformed),
    ("--1", Malformed),
    ("1_000", Malformed),
    ("0x10", Malformed),
    ("1,5", Malformed),
    ("1 5", Malformed),
    ("infin", Malformed),
    ("infinit", Malformed),
    ("infinityy", Malformed),
    ("nan(1)", Malformed),
    ("in", Malformed),
    ("1e+", Malformed),
    ("1.5f", Malformed),
    ("1e5.5", Malformed),
    ("\u{661}", Malformed),
    ("\u{ff11}", Malformed),
    ("\u{feff}1", Malformed),
    ("\u{200b}1", Malformed),
    ("1\u{200b}", Malformed),
    (
        "1.000000000000000111022302462515654042363166809082031249999999999999",
        Bits(0x3ff0_0000_0000_0000),
    ),
    (
        "1.00000000000000011102230246251565404236316680908203125",
        Bits(0x3ff0_0000_0000_0000),
    ),
    (
        "1.000000000000000111022302462515654042363166809082031250000000000001",
        Bits(0x3ff0_0000_0000_0001),
    ),
    ("2.2250738585072011e-308", Bits(0x000f_ffff_ffff_ffff)),
    ("-1.8e308", Bits(0xfff0_0000_0000_0000)),
    ("1e-99999999999999999999", Bits(0x0000_0000_0000_0000)),
    ("1\u{0}", Malformed),
    ("\u{0}1", Malformed),
    ("\u{205f}1.5\u{1680}", Bits(0x3ff8_0000_0000_0000)),
    (OVERFLOW_MIDPOINT, Bits(0x7ff0_0000_0000_0000)),
    (BELOW_OVERFLOW_MIDPOINT, Bits(0x7fef_ffff_ffff_ffff)),
];

const TO_INT_CASES: &[(&str, Option<i64>)] = &[
    ("7", Some(7)),
    ("+7", Some(7)),
    ("-7", Some(-7)),
    ("007", Some(7)),
    ("9223372036854775807", Some(i64::MAX)),
    ("-9223372036854775808", Some(i64::MIN)),
    (" 7 ", Some(7)),
    ("\n7\n", Some(7)),
    ("\u{a0}7", Some(7)),
    ("\u{2003}-7\u{3000}", Some(-7)),
    ("\t+7\r\n", Some(7)),
    ("9223372036854775808", None),
    ("-9223372036854775809", None),
    ("", None),
    ("+", None),
    ("-", None),
    ("1.0", None),
    ("1e3", None),
    ("0x1", None),
    ("1_0", None),
    ("\u{661}", None),
    (" ", None),
    ("7 7", None),
    ("\u{feff}7", None),
];

/// A Surf string literal for `text`, escaping everything outside printable
/// ASCII so the program text carries the exact scalar values.
fn surf_string_literal(text: &str) -> String {
    let mut literal = String::from("\"");
    for character in text.chars() {
        match character {
            '"' => literal.push_str("\\\""),
            '\\' => literal.push_str("\\\\"),
            ' '..='~' => literal.push(character),
            other => literal.push_str(&format!("\\u{{{:x}}}", u32::from(other))),
        }
    }
    literal.push('"');
    literal
}

/// Spellings too long for a program literal, read from files at run time. An
/// exponent offset by hundreds of thousands of digits must not saturate, and
/// a nonzero digit far past the deciding prefix must still round up.
fn long_float_cases() -> Vec<(String, Float)> {
    let one = Bits(0x3ff0_0000_0000_0000);
    let midpoint_after_one = "1.00000000000000011102230246251565404236316680908203125";
    vec![
        (format!("1{}e-700000", "0".repeat(700_000)), one),
        (format!("0.{}1e700000", "0".repeat(699_999)), one),
        (format!("1{}e-655360", "0".repeat(655_360)), one),
        (
            format!("{midpoint_after_one}{}1", "0".repeat(2_000)),
            Bits(0x3ff0_0000_0000_0001),
        ),
    ]
}

/// Write each long spelling to its own file under `dir`.
fn write_long_cases(dir: &Path, cases: &[(String, Float)]) -> Vec<PathBuf> {
    cases
        .iter()
        .enumerate()
        .map(|(index, (text, _))| {
            let path = dir.join(format!("long-{index}.txt"));
            write_file(&path, text);
            path
        })
        .collect()
}

/// The paths are absolute, so the lanes' different working directories cannot
/// change what they read.
fn program(long_paths: &[PathBuf]) -> String {
    let mut source = String::from("module Issue2870.TextParsers\n");
    for (index, (text, _)) in TO_FLOAT_CASES.iter().enumerate() {
        source.push_str(&format!(
            "f{index:03} = print(to_float({}))\n",
            surf_string_literal(text)
        ));
    }
    for (index, (text, _)) in TO_INT_CASES.iter().enumerate() {
        source.push_str(&format!(
            "i{index:03} = print(to_int({}))\n",
            surf_string_literal(text)
        ));
    }
    for (index, path) in long_paths.iter().enumerate() {
        source.push_str(&format!(
            "l{index:03} = print(to_float(read_file({})))\n",
            surf_string_literal(path.to_str().expect("UTF-8 path"))
        ));
    }
    source
}

fn eval_stdout(source: &str, name: &str) -> String {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}.ch"));
    write_file(&path, source);
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("chelis eval should run");
    assert!(
        out.status.success(),
        "eval failed: {}\nstderr:\n{}",
        out.status,
        String::from_utf8_lossy(&out.stderr),
    );
    String::from_utf8(out.stdout).expect("utf-8 stdout")
}

/// The payload of a printed `Some(...)`, or `None` for a printed `None`.
fn printed_payload(line: &str) -> Option<&str> {
    if line == "None" {
        return None;
    }
    Some(
        line.strip_prefix("Some(")
            .and_then(|rest| rest.strip_suffix(')'))
            .unwrap_or_else(|| panic!("not a printed Option: {line:?}")),
    )
}

fn printed_float(line: &str) -> Float {
    match printed_payload(line) {
        None => Malformed,
        Some(text) => {
            let value = text
                .parse::<f64>()
                .unwrap_or_else(|_| panic!("not a printed f64: {line:?}"));
            if value.is_nan() {
                Nan
            } else {
                Bits(value.to_bits())
            }
        }
    }
}

fn printed_int(line: &str) -> Option<i64> {
    printed_payload(line).map(|text| {
        text.parse::<i64>()
            .unwrap_or_else(|_| panic!("not a printed i64: {line:?}"))
    })
}

/// Every case on which `stdout` disagrees with [05-OP-59].
fn atom_mismatches(stdout: &str, long_cases: &[(String, Float)]) -> Vec<String> {
    let lines: Vec<&str> = stdout.lines().collect();
    let count = TO_FLOAT_CASES.len() + TO_INT_CASES.len() + long_cases.len();
    assert!(
        lines.len() >= count,
        "expected at least {count} printed results, got:\n{stdout}"
    );
    let mut mismatches = Vec::new();
    for ((text, expected), line) in TO_FLOAT_CASES.iter().zip(&lines) {
        if printed_float(line) != *expected {
            mismatches.push(format!(
                "to_float({text:?}) printed {line}, expected {expected:?}"
            ));
        }
    }
    for ((text, expected), line) in TO_INT_CASES.iter().zip(&lines[TO_FLOAT_CASES.len()..]) {
        if printed_int(line) != *expected {
            mismatches.push(format!(
                "to_int({text:?}) printed {line}, expected {expected:?}"
            ));
        }
    }
    let long_lines = &lines[TO_FLOAT_CASES.len() + TO_INT_CASES.len()..];
    for ((text, expected), line) in long_cases.iter().zip(long_lines) {
        if printed_float(line) != *expected {
            mismatches.push(format!(
                "to_float of a {}-character spelling printed {line}, expected {expected:?}",
                text.len()
            ));
        }
    }
    mismatches
}

#[test]
fn to_int_and_to_float_follow_op59_in_both_lanes() {
    let inputs = tempdir().expect("tempdir");
    let long_cases = long_float_cases();
    let long_paths = write_long_cases(inputs.path(), &long_cases);
    let source = program(&long_paths);
    let name = "issue_2870_text_parsers";

    let eval = eval_stdout(&source, name);
    let eval_mismatches = atom_mismatches(&eval, &long_cases);

    assert!(
        gcc_available(),
        "this cross-lane oracle requires a C compiler"
    );
    let built = build_and_run(&source, name);
    let built_mismatches = atom_mismatches(&built, &long_cases);

    assert!(
        eval_mismatches.is_empty() && built_mismatches.is_empty(),
        "[05-OP-59] violations\neval ({}):\n{}\nbuild ({}):\n{}",
        eval_mismatches.len(),
        eval_mismatches.join("\n"),
        built_mismatches.len(),
        built_mismatches.join("\n"),
    );
    assert_eq!(built, eval, "build lane disagrees with eval lane");
}
