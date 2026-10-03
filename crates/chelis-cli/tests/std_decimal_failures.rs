//! chelis#2778: the `Std.Decimal` failure contract of [05-OP-76].
//!
//! Every failure is a `fail` whose message is exactly
//! `<function>: <kind>: <detail>`. When several checks fail, each argument's
//! own validity is reported left to right, then `RejectInexact`, then the
//! result's range, and every check runs before the arithmetic it protects, so
//! no primitive numeric trap escapes a call for any arguments. Each eval suite
//! runs as one `chelis test` invocation over a generated fixture whose every
//! test calls one failing (or extreme) expression; the runner reports each
//! call's failure message on its FAIL line. The compiled C lane builds one
//! program that selects a failing expression by the contents of `case.txt`
//! and runs it once per case. Opaque construction and inspection outside the
//! module, the removed names, and `grad` through `decimal_from_f64` are
//! checked through `chelis check`, `chelis eval`, and `chelis build`.
//!
//! Pull-request CI runs both message tests in full and a canary of the extreme
//! sweep; the complete sweep runs in the nightly workflow
//! (`.config/ci-test-targets.toml`).

#[path = "common/mod.rs"]
mod common;

use assert_cmd::Command;
use common::{gcc_available, make_app, write_file};
use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command as StdCommand;

const IMPORTS: &str = "import Std.Decimal (Decimal, decimal, try_decimal, decimal_to_string, decimal_to_fixed_string, decimal_from_i64, decimal_to_i64, try_decimal_to_i64, decimal_from_f64, try_decimal_from_f64, decimal_to_f64, decimal_to_f32, decimal_scale, decimal_add, decimal_sub, decimal_mul, decimal_round, decimal_div, try_decimal_div, decimal_lt, decimal_lte, decimal_gt, decimal_gte)\nimport Std.Rounding (Rounding, RoundTowardNegative, RoundTowardPositive, RoundTowardZero, RoundAwayFromZero, RoundTiesToEven, RoundTiesToAway, RejectInexact)";

const HELPERS: &str = "def i64_minimum() -> i64 = sub(-9223372036854775807i64, 1i64)\ndef zeros(count: i64) -> string = fold(fn (acc: string, unused: i64) -> string_concat(acc, \"0\"), \"\", range(0i64, count))\ndef reached[q](value: q) -> string = \"reached\"\n";

const MAX: &str = "99999999999999999999999999999999999999";

/// (test name, expression, exact failure message).
const EXACT_FAILURES: &[(&str, &str, &str)] = &[
    (
        "parse_plus_sign",
        "decimal(\"+1\")",
        "decimal: domain: malformed number text \"+1\"",
    ),
    (
        "parse_leading_space",
        "decimal(\" 1\")",
        "decimal: domain: malformed number text \" 1\"",
    ),
    (
        "parse_nan",
        "decimal(\"NaN\")",
        "decimal: domain: malformed number text \"NaN\"",
    ),
    (
        "parse_empty",
        "decimal(\"\")",
        "decimal: domain: malformed number text \"\"",
    ),
    (
        "parse_long_malformed",
        "decimal(\"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\")",
        "decimal: domain: malformed number text \"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx...\"",
    ),
    (
        "parse_over_long",
        "decimal(string_concat(\"1.\", zeros(999i64)))",
        "decimal: domain: number text has 1001 characters, more than 1000",
    ),
    (
        "parse_over_long_malformed",
        "decimal(string_concat(\"x\", zeros(1000i64)))",
        "decimal: domain: number text has 1001 characters, more than 1000",
    ),
    (
        "parse_above_range",
        "decimal(\"1e38\")",
        "decimal: domain: 1e38 is outside the decimal range",
    ),
    (
        "parse_below_scale",
        "decimal(\"1e-39\")",
        "decimal: domain: 1e-39 is outside the decimal range",
    ),
    (
        "parse_i64_minimum_exponent",
        "decimal(\"1e-9223372036854775808\")",
        "decimal: domain: 1e-9223372036854775808 is outside the decimal range",
    ),
    (
        "parse_long_out_of_range",
        "decimal(string_concat(\"1\", zeros(60i64)))",
        "decimal: domain: 1000000000000000000000000000000000000000... is outside the decimal range",
    ),
    (
        "fixed_string_digits",
        "decimal_to_fixed_string(decimal(\"1.25\"), 1i64)",
        "decimal_to_fixed_string: domain: 1.25 has 2 fractional digits, more than 1",
    ),
    (
        "fixed_string_scale",
        "decimal_to_fixed_string(decimal(\"1.25\"), 39i64)",
        "decimal_to_fixed_string: domain: scale 39 is outside 0..38",
    ),
    (
        "fixed_string_negative_scale",
        "decimal_to_fixed_string(decimal(\"1\"), -1i64)",
        "decimal_to_fixed_string: domain: scale -1 is outside 0..38",
    ),
    (
        "fixed_string_minimum_scale",
        "decimal_to_fixed_string(decimal(\"1.25\"), i64_minimum())",
        "decimal_to_fixed_string: domain: scale -9223372036854775808 is outside 0..38",
    ),
    (
        "to_i64_above",
        "decimal_to_i64(decimal(\"9223372036854775808\"), RoundTiesToEven)",
        "decimal_to_i64: overflow: 9223372036854775808 rounds to an integer outside i64",
    ),
    (
        "to_i64_below",
        "decimal_to_i64(decimal(\"-9223372036854775808.5\"), RoundTowardNegative)",
        "decimal_to_i64: overflow: -9223372036854775808.5 rounds to an integer outside i64",
    ),
    (
        "to_i64_rounded_up",
        "decimal_to_i64(decimal(\"9223372036854775807.1\"), RoundTowardPositive)",
        "decimal_to_i64: overflow: 9223372036854775807.1 rounds to an integer outside i64",
    ),
    (
        "to_i64_decimal_maximum",
        "decimal_to_i64(decimal(\"99999999999999999999999999999999999999\"), RoundTowardZero)",
        "decimal_to_i64: overflow: 99999999999999999999999999999999999999 rounds to an integer outside i64",
    ),
    (
        "to_i64_inexact",
        "decimal_to_i64(decimal(\"1.5\"), RejectInexact)",
        "decimal_to_i64: domain: 1.5 is not a multiple of 10^-0",
    ),
    (
        "to_i64_inexact_before_range",
        "decimal_to_i64(decimal(\"9223372036854775808.5\"), RejectInexact)",
        "decimal_to_i64: domain: 9223372036854775808.5 is not a multiple of 10^-0",
    ),
    (
        "from_f64_nan",
        "decimal_from_f64(div(0.0f64, 0.0f64), 2i64, RoundTiesToEven)",
        "decimal_from_f64: domain: NaN is not finite",
    ),
    (
        "from_f64_infinity",
        "decimal_from_f64(div(1.0f64, 0.0f64), 2i64, RoundTiesToEven)",
        "decimal_from_f64: domain: inf is not finite",
    ),
    (
        "from_f64_finiteness_before_scale",
        "decimal_from_f64(div(-1.0f64, 0.0f64), 99i64, RejectInexact)",
        "decimal_from_f64: domain: -inf is not finite",
    ),
    (
        "from_f64_scale",
        "decimal_from_f64(0.1f64, 39i64, RoundTiesToEven)",
        "decimal_from_f64: domain: scale 39 is outside 0..38",
    ),
    (
        "from_f64_negative_scale",
        "decimal_from_f64(0.1f64, -1i64, RoundTiesToEven)",
        "decimal_from_f64: domain: scale -1 is outside 0..38",
    ),
    (
        "from_f64_inexact",
        "decimal_from_f64(0.1f64, 2i64, RejectInexact)",
        "decimal_from_f64: domain: 0.1 is not a multiple of 10^-2",
    ),
    (
        "from_f64_range",
        "decimal_from_f64(2.0e38f64, 0i64, RoundTiesToEven)",
        "decimal_from_f64: domain: 2e38 is outside the decimal range",
    ),
    (
        "from_f64_double_above_range",
        "decimal_from_f64(neg(1.0000000000000002e38f64), 0i64, RoundTowardZero)",
        "decimal_from_f64: domain: -1.0000000000000002e38 is outside the decimal range",
    ),
    (
        "from_f64_largest_double",
        "decimal_from_f64(1.7976931348623157e308f64, 38i64, RejectInexact)",
        "decimal_from_f64: domain: 1.7976931348623157e308 is outside the decimal range",
    ),
    (
        "add_above_range",
        "decimal_add(decimal(\"99999999999999999999999999999999999999\"), decimal(\"1\"))",
        "decimal_add: overflow: 99999999999999999999999999999999999999 plus 1 is outside the decimal range",
    ),
    (
        "add_below_range",
        "decimal_add(decimal(\"-99999999999999999999999999999999999999\"), decimal(\"-0.5\"))",
        "decimal_add: overflow: -99999999999999999999999999999999999999 plus -0.5 is outside the decimal range",
    ),
    (
        "sub_below_range",
        "decimal_sub(decimal(\"-99999999999999999999999999999999999999\"), decimal(\"1\"))",
        "decimal_sub: overflow: -99999999999999999999999999999999999999 minus 1 is outside the decimal range",
    ),
    (
        "sub_too_many_digits",
        "decimal_sub(decimal(\"10\"), decimal(\"1e-38\"))",
        "decimal_sub: overflow: 10 minus 0.00000000000000000000000000000000000001 is outside the decimal range",
    ),
    (
        "mul_too_many_fraction_digits",
        "decimal_mul(decimal(\"1e-20\"), decimal(\"1e-19\"))",
        "decimal_mul: overflow: 0.00000000000000000001 times 0.0000000000000000001 is outside the decimal range",
    ),
    (
        "mul_above_range",
        "decimal_mul(decimal(\"99999999999999999999999999999999999999\"), decimal(\"-99999999999999999999999999999999999999\"))",
        "decimal_mul: overflow: 99999999999999999999999999999999999999 times -99999999999999999999999999999999999999 is outside the decimal range",
    ),
    (
        "round_scale",
        "decimal_round(decimal(\"1.25\"), 39i64, RoundTiesToEven)",
        "decimal_round: domain: scale 39 is outside 0..38",
    ),
    (
        "round_negative_scale",
        "decimal_round(decimal(\"1.25\"), -1i64, RoundTiesToEven)",
        "decimal_round: domain: scale -1 is outside 0..38",
    ),
    (
        "round_maximum_scale",
        "decimal_round(decimal(\"1.25\"), 9223372036854775807i64, RejectInexact)",
        "decimal_round: domain: scale 9223372036854775807 is outside 0..38",
    ),
    (
        "round_inexact",
        "decimal_round(decimal(\"1.25\"), 1i64, RejectInexact)",
        "decimal_round: domain: 1.25 is not a multiple of 10^-1",
    ),
    (
        "div_zero",
        "decimal_div(decimal(\"1\"), decimal(\"0\"), 2i64, RoundTiesToEven)",
        "decimal_div: domain: division by zero",
    ),
    (
        "div_zero_before_scale",
        "decimal_div(decimal(\"1\"), decimal(\"-0.0\"), 39i64, RejectInexact)",
        "decimal_div: domain: division by zero",
    ),
    (
        "div_scale",
        "decimal_div(decimal(\"1\"), decimal(\"3\"), 39i64, RoundTiesToEven)",
        "decimal_div: domain: scale 39 is outside 0..38",
    ),
    (
        "div_minimum_scale",
        "decimal_div(decimal(\"1\"), decimal(\"3\"), i64_minimum(), RoundTiesToEven)",
        "decimal_div: domain: scale -9223372036854775808 is outside 0..38",
    ),
    (
        "div_inexact",
        "decimal_div(decimal(\"1\"), decimal(\"3\"), 2i64, RejectInexact)",
        "decimal_div: domain: 1 divided by 3 is not a multiple of 10^-2",
    ),
    (
        "div_inexact_before_range",
        "decimal_div(decimal(\"99999999999999999999999999999999999999\"), decimal(\"0.7\"), 0i64, RejectInexact)",
        "decimal_div: domain: 99999999999999999999999999999999999999 divided by 0.7 is not a multiple of 10^-0",
    ),
    (
        "div_too_many_digits",
        "decimal_div(decimal(\"10\"), decimal(\"3\"), 38i64, RoundTiesToEven)",
        "decimal_div: overflow: 10 divided by 3 rounded to scale 38 is outside the decimal range",
    ),
    (
        "div_above_range",
        "decimal_div(decimal(\"99999999999999999999999999999999999999\"), decimal(\"0.1\"), 0i64, RoundTowardZero)",
        "decimal_div: overflow: 99999999999999999999999999999999999999 divided by 0.1 rounded to scale 0 is outside the decimal range",
    ),
    (
        "try_div_overflow",
        "try_decimal_div(decimal(\"10\"), decimal(\"3\"), 38i64, RoundTiesToEven)",
        "try_decimal_div: overflow: 10 divided by 3 rounded to scale 38 is outside the decimal range",
    ),
];

fn run_chelis(app_pkg: &Path, reef_home: &Path, args: &[&str]) -> (bool, String) {
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", reef_home)
        .current_dir(app_pkg)
        .args(args)
        .output()
        .expect("chelis runs");
    (
        output.status.success(),
        format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ),
    )
}

/// Runs one `chelis test` over a fixture with one test per expression and
/// returns each test's outcome: `None` for PASS, `Some(message)` for FAIL.
fn run_expression_suite(
    dir_name: &str,
    expressions: &[(String, String)],
) -> BTreeMap<String, Option<String>> {
    let (_dir, reef_home, app_pkg) = make_app(dir_name);
    let mut source = format!("module Demo.Tests.Decimal\n{IMPORTS}\n{HELPERS}");
    for (name, expression) in expressions {
        source.push_str(&format!(
            "def test_{name}() -> unit ! {{ Test }} = {{\n  _value = {expression}\n  ()\n}}\n"
        ));
    }
    write_file(
        &app_pkg.join("src/main.ch"),
        "module Demo.Main\nanchor = 0i64\n",
    );
    let path = app_pkg.join("tests/decimal_cases.ch");
    write_file(&path, &source);
    let (_, rendered) = run_chelis(
        &app_pkg,
        &reef_home,
        &["test", "--batch-mode", "file", path.to_str().unwrap()],
    );
    let mut outcomes = BTreeMap::new();
    for line in rendered.lines() {
        let Some(rest) = line.trim_start().strip_prefix("test_") else {
            continue;
        };
        let Some((name, verdict)) = rest.split_once(' ') else {
            continue;
        };
        let verdict = verdict.trim_start_matches(['.', ' ']);
        if verdict == "PASS" {
            outcomes.insert(name.to_string(), None);
        } else if let Some(message) = verdict
            .strip_prefix("FAIL (")
            .and_then(|tail| tail.strip_suffix(')'))
        {
            outcomes.insert(name.to_string(), Some(message.to_string()));
        }
    }
    assert_eq!(
        outcomes.len(),
        expressions.len(),
        "every generated test must report exactly one verdict:\n{rendered}"
    );
    outcomes
}

#[test]
fn std_decimal_failures_report_their_exact_message() {
    let expressions: Vec<(String, String)> = EXACT_FAILURES
        .iter()
        .map(|(name, expression, _)| (name.to_string(), expression.to_string()))
        .collect();
    let outcomes = run_expression_suite("decimal-failures-2778", &expressions);
    for (name, expression, expected) in EXACT_FAILURES {
        assert_eq!(
            outcomes.get(*name),
            Some(&Some(expected.to_string())),
            "{name}: `{expression}` must fail with exactly `{expected}`"
        );
    }
}

/// The compiled C lane reports the same messages: one program selects an
/// expression by the contents of `case.txt`, and each run must fail with that
/// case's exact message.
#[test]
fn std_decimal_compiled_failures_report_their_exact_message() {
    if !gcc_available() {
        return;
    }
    let (_dir, reef_home, app_pkg) = make_app("decimal-compiled-failures-2778");
    let mut chain = String::from("\"unselected\"");
    for (index, (_, expression, _)) in EXACT_FAILURES.iter().enumerate().rev() {
        chain = format!("if eq(selector, \"{index}\") then reached({expression}) else {chain}");
    }
    let source = format!(
        "module Demo.Main\n{IMPORTS}\n{HELPERS}def outcome_of(selector: string) -> string = {chain}\noutcome = print(outcome_of(string_trim(read_file(\"case.txt\"))))\n"
    );
    write_file(&app_pkg.join("src/main.ch"), &source);
    let out_dir = app_pkg.join("main-out");
    let (built, rendered) = run_chelis(
        &app_pkg,
        &reef_home,
        &[
            "build",
            app_pkg.join("src/main.ch").to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ],
    );
    assert!(
        built,
        "the failure-selection program must build:\n{rendered}"
    );
    for (index, (name, expression, expected)) in EXACT_FAILURES.iter().enumerate() {
        write_file(&app_pkg.join("case.txt"), &format!("{index}\n"));
        let run = StdCommand::new(out_dir.join("main"))
            .current_dir(&app_pkg)
            .output()
            .expect("compiled binary runs");
        let printed = format!(
            "{}{}",
            String::from_utf8_lossy(&run.stdout),
            String::from_utf8_lossy(&run.stderr)
        );
        assert!(
            !run.status.success() && printed.lines().any(|line| line.trim() == *expected),
            "{name}: compiled `{expression}` must fail with exactly `{expected}`:\n{printed}"
        );
    }
    write_file(&app_pkg.join("case.txt"), "none\n");
    let run = StdCommand::new(out_dir.join("main"))
        .current_dir(&app_pkg)
        .output()
        .expect("compiled binary runs");
    assert!(
        run.status.success() && String::from_utf8_lossy(&run.stdout).contains("unselected"),
        "an unselected case must run to completion: {run:?}"
    );
}

const VALUES: &[&str] = &[
    MAX,
    "-99999999999999999999999999999999999999",
    "0.00000000000000000000000000000000000001",
    "-0.00000000000000000000000000000000000001",
    "0.99999999999999999999999999999999999999",
    "9223372036854775807.5",
    "-9223372036854775808.5",
    "0",
];

const MODES: &[&str] = &[
    "RoundTowardNegative",
    "RoundTowardPositive",
    "RoundTowardZero",
    "RoundAwayFromZero",
    "RoundTiesToEven",
    "RoundTiesToAway",
    "RejectInexact",
];

const SCALES: &[&str] = &[
    "0i64",
    "37i64",
    "39i64",
    "9223372036854775807i64",
    "i64_minimum()",
];

const FLOATS: &[&str] = &[
    "0.0f64",
    "neg(0.0f64)",
    "5.0e-324f64",
    "neg(5.0e-324f64)",
    "2.2250738585072014e-308f64",
    "1.0e38f64",
    "neg(1.7976931348623157e308f64)",
    "1.7976931348623157e308f64",
    "div(0.0f64, 0.0f64)",
    "div(1.0f64, 0.0f64)",
    "div(-1.0f64, 0.0f64)",
    "0.1f64",
    "2.5f64",
];

/// Every call below passes, or fails `domain` or `overflow` under its own
/// callable's name; a primitive trap would surface under another message.
fn extreme_cases() -> Vec<(&'static str, String)> {
    let dec = |text: &str| format!("decimal(\"{text}\")");
    let mut cases: Vec<(&'static str, String)> = Vec::new();
    for text in [
        "1e9223372036854775807",
        "1e-9223372036854775808",
        "1e99999999999999999999999999999",
        "-0e-99999999999999999999999999999",
        "9.9999999999999999999999999999999999999e37",
        "99999999999999999999999999999999999999.5",
        "0.000000000000000000000000000000000000005",
        "1e0000000000000000000000000000000000000000038",
        "1E+37",
        "-1e-38",
        "9223372036854775807",
        "-9223372036854775808",
        "\u{0661}",
        "-",
        "1e",
    ] {
        cases.push(("decimal", dec(text)));
    }
    for (prefix, count) in [
        ("1.", 998),
        ("1.", 999),
        ("0.", 999),
        ("1", 999),
        ("x", 999),
    ] {
        cases.push((
            "decimal",
            format!("decimal(string_concat(\"{prefix}\", zeros({count}i64)))"),
        ));
    }
    for value in &VALUES[..4] {
        for scale in [
            "0i64",
            "38i64",
            "39i64",
            "i64_minimum()",
            "9223372036854775807i64",
        ] {
            cases.push((
                "decimal_to_fixed_string",
                format!("decimal_to_fixed_string({}, {scale})", dec(value)),
            ));
        }
    }
    for int in ["9223372036854775807i64", "i64_minimum()", "0i64"] {
        cases.push(("decimal_from_i64", format!("decimal_from_i64({int})")));
    }
    for value in VALUES {
        for mode in MODES {
            cases.push((
                "decimal_to_i64",
                format!("decimal_to_i64({}, {mode})", dec(value)),
            ));
        }
    }
    for (index, float) in FLOATS.iter().enumerate() {
        for scale in ["0i64", "38i64", "i64_minimum()"] {
            let mode = MODES[index % MODES.len()];
            cases.push((
                "decimal_from_f64",
                format!("decimal_from_f64({float}, {scale}, {mode})"),
            ));
        }
    }
    for value in VALUES {
        cases.push(("decimal_to_f64", format!("decimal_to_f64({})", dec(value))));
        cases.push(("decimal_to_f32", format!("decimal_to_f32({})", dec(value))));
    }
    let operands = [
        MAX,
        "-99999999999999999999999999999999999999",
        "0.00000000000000000000000000000000000001",
        "0.5",
    ];
    for left in operands {
        for right in operands {
            for function in ["decimal_add", "decimal_sub", "decimal_mul"] {
                cases.push((
                    function,
                    format!("{function}({}, {})", dec(left), dec(right)),
                ));
            }
        }
    }
    for value in [
        "9999999999999999999999999999999999999.5",
        "-0.99999999999999999999999999999999999999",
    ] {
        for scale in SCALES {
            for mode in MODES {
                cases.push((
                    "decimal_round",
                    format!("decimal_round({}, {scale}, {mode})", dec(value)),
                ));
            }
        }
    }
    for numerator in [MAX, "-0.00000000000000000000000000000000000001"] {
        for denominator in ["0.00000000000000000000000000000000000001", "0", "-3"] {
            for scale in ["0i64", "38i64", "39i64"] {
                for mode in ["RoundTiesToEven", "RejectInexact"] {
                    cases.push((
                        "decimal_div",
                        format!(
                            "decimal_div({}, {}, {scale}, {mode})",
                            dec(numerator),
                            dec(denominator)
                        ),
                    ));
                }
            }
        }
    }
    cases
}

/// One representative of each extreme class per callable, every one of them a
/// case of `extreme_cases`: the envelope's ends, scales outside 0..38 and at
/// the i64 limits, the i64 limits and their ties, subnormal and non-finite
/// floats, text at the length bound or with an exponent at the i64 limit, a
/// zero divisor, and a result past the envelope. A scale outside 0..38 comes
/// with a finite float and a nonzero divisor, since a non-finite float and a
/// zero divisor are rejected before the scale. A trap that only another
/// combination of arguments reaches is left to the complete sweep.
fn extreme_canary_cases() -> Vec<(&'static str, String)> {
    let dec = |text: &str| format!("decimal(\"{text}\")");
    let tiny = "0.00000000000000000000000000000000000001";
    let mut cases: Vec<(&'static str, String)> = Vec::new();
    for text in [
        "1e9223372036854775807",
        "99999999999999999999999999999999999999.5",
        "\u{0661}",
    ] {
        cases.push(("decimal", dec(text)));
    }
    cases.push((
        "decimal",
        "decimal(string_concat(\"1.\", zeros(998i64)))".to_string(),
    ));
    for (value, scale) in [(MAX, "38i64"), (tiny, "9223372036854775807i64")] {
        cases.push((
            "decimal_to_fixed_string",
            format!("decimal_to_fixed_string({}, {scale})", dec(value)),
        ));
    }
    cases.push((
        "decimal_from_i64",
        "decimal_from_i64(i64_minimum())".to_string(),
    ));
    for (value, mode) in [
        (MAX, "RoundTowardZero"),
        ("-9223372036854775808.5", "RoundTowardNegative"),
        (tiny, "RejectInexact"),
    ] {
        cases.push((
            "decimal_to_i64",
            format!("decimal_to_i64({}, {mode})", dec(value)),
        ));
    }
    for (float, scale, mode) in [
        ("5.0e-324f64", "38i64", "RoundTowardZero"),
        ("neg(1.7976931348623157e308f64)", "0i64", "RejectInexact"),
        ("div(1.0f64, 0.0f64)", "0i64", "RoundTowardZero"),
        ("0.1f64", "i64_minimum()", "RoundTiesToEven"),
    ] {
        cases.push((
            "decimal_from_f64",
            format!("decimal_from_f64({float}, {scale}, {mode})"),
        ));
    }
    cases.push(("decimal_to_f64", format!("decimal_to_f64({})", dec(tiny))));
    cases.push(("decimal_to_f32", format!("decimal_to_f32({})", dec(tiny))));
    for (function, left, right) in [
        ("decimal_add", MAX, MAX),
        ("decimal_sub", MAX, tiny),
        ("decimal_mul", tiny, tiny),
    ] {
        cases.push((
            function,
            format!("{function}({}, {})", dec(left), dec(right)),
        ));
    }
    let carry = "9999999999999999999999999999999999999.5";
    for (value, scale, mode) in [
        (carry, "0i64", "RoundAwayFromZero"),
        (
            "-0.99999999999999999999999999999999999999",
            "37i64",
            "RoundTiesToEven",
        ),
        (carry, "i64_minimum()", "RoundTowardZero"),
    ] {
        cases.push((
            "decimal_round",
            format!("decimal_round({}, {scale}, {mode})", dec(value)),
        ));
    }
    for (numerator, denominator, scale, mode) in [
        (MAX, tiny, "0i64", "RoundTiesToEven"),
        (MAX, "0", "0i64", "RejectInexact"),
        (MAX, "-3", "39i64", "RoundTiesToEven"),
        (
            "-0.00000000000000000000000000000000000001",
            "-3",
            "38i64",
            "RejectInexact",
        ),
    ] {
        cases.push((
            "decimal_div",
            format!(
                "decimal_div({}, {}, {scale}, {mode})",
                dec(numerator),
                dec(denominator)
            ),
        ));
    }
    cases
}

/// Runs `cases` as one suite: each call passes or fails `domain` or
/// `overflow` under its own callable's name, and the cases include both.
fn assert_no_primitive_trap(dir_name: &str, cases: &[(&'static str, String)]) {
    let expressions: Vec<(String, String)> = cases
        .iter()
        .enumerate()
        .map(|(index, (_, expression))| (format!("case_{index:04}"), expression.clone()))
        .collect();
    let outcomes = run_expression_suite(dir_name, &expressions);
    let mut failures = 0usize;
    for (index, (function, expression)) in cases.iter().enumerate() {
        let outcome = &outcomes[&format!("case_{index:04}")];
        if let Some(message) = outcome {
            failures += 1;
            assert!(
                message.starts_with(&format!("{function}: domain: "))
                    || message.starts_with(&format!("{function}: overflow: ")),
                "`{expression}` escaped the failure contract with `{message}`"
            );
        }
    }
    assert!(
        failures > 0 && failures < cases.len(),
        "the sweep must exercise both accepted and rejected extremes ({failures} of {})",
        cases.len()
    );
}

#[test]
fn std_decimal_extreme_argument_canary_raises_no_primitive_trap() {
    let canary = extreme_canary_cases();
    let sweep = extreme_cases();
    for case in &canary {
        assert!(
            sweep.contains(case),
            "canary case {case:?} is not in the complete sweep"
        );
    }
    assert_no_primitive_trap("decimal-extreme-canary-2778", &canary);
}

/// Runs in the nightly workflow, not in pull-request CI
/// (`.config/ci-test-targets.toml`).
#[test]
fn std_decimal_extreme_arguments_raise_no_primitive_trap() {
    assert_no_primitive_trap("decimal-extremes-2778", &extreme_cases());
}

/// [05-OP-76]: `Decimal` is opaque. Outside `Std.Decimal`, record
/// construction, a bare constructor reference, field access, and a record
/// pattern are `OpaqueTypeViolation`s, and Surf has no `cast` spelling into
/// an ADT at all.
#[test]
fn std_decimal_opaque_type_rejects_outside_construction_and_inspection() {
    let (_dir, reef_home, app_pkg) = make_app("decimal-opaque-2778");
    let path = app_pkg.join("src/main.ch");
    write_file(
        &path,
        &format!(
            "module Demo.Main\n{IMPORTS}\nforged = Decimal {{ negative: false, limb0: 15i64, limb1: 0i64, limb2: 0i64, limb3: 0i64, limb4: 0i64, scale: 1i64 }}\nnamed = Decimal\npeeked = decimal(\"1.5\").limb0\ndef unpacked(x: Decimal) -> i64 =\n  match x with {{\n    | Decimal {{ negative: n, limb0: a, limb1: b, limb2: c, limb3: d, limb4: e, scale: s }} => a\n  }}\n"
        ),
    );
    let (success, rendered) = run_chelis(&app_pkg, &reef_home, &["check", path.to_str().unwrap()]);
    assert!(
        !success,
        "forging or inspecting a Decimal must not check:\n{rendered}"
    );
    for action in [
        "record construction of",
        "constructor reference to",
        "field access on",
        "record pattern match on",
    ] {
        assert!(
            rendered.contains(&format!(
                "{action} opaque type `Decimal` outside its defining module `Std.Decimal`"
            )),
            "`{action}` was not rejected as an opaque violation:\n{rendered}"
        );
    }

    write_file(
        &path,
        &format!("module Demo.Main\n{IMPORTS}\ndef forged(x: i64) -> Decimal = cast(x, Decimal)\n"),
    );
    let (success, rendered) = run_chelis(&app_pkg, &reef_home, &["check", path.to_str().unwrap()]);
    assert!(
        !success && rendered.contains("expected identifier"),
        "a cast into Decimal must not parse:\n{rendered}"
    );

    write_file(
        &path,
        &format!("module Demo.Main\n{IMPORTS}\nshown = decimal_to_string(decimal(\"1.50\"))\n"),
    );
    let (success, rendered) = run_chelis(&app_pkg, &reef_home, &["check", path.to_str().unwrap()]);
    assert!(success, "the exported producers must check:\n{rendered}");
}

/// The old surface is gone: each removed name is an ordinary unexported
/// import.
#[test]
fn std_decimal_removed_names_are_not_exported() {
    let (_dir, reef_home, app_pkg) = make_app("decimal-removed-2778");
    let path = app_pkg.join("src/main.ch");
    for name in [
        "RoundingMode",
        "RoundHalfUp",
        "round_half_up",
        "decimal_from_int",
        "decimal_to_float",
        "decimal_eq",
    ] {
        write_file(
            &path,
            &format!("module Demo.Main\nimport Std.Decimal ({name})\nvalue = 1i64\n"),
        );
        let (success, rendered) =
            run_chelis(&app_pkg, &reef_home, &["check", path.to_str().unwrap()]);
        assert!(
            !success
                && rendered.contains(&format!(
                    "module `Std.Decimal` does not export `{name}` for import into Demo.Main"
                )),
            "importing the removed `{name}` must be the ordinary unexported-name error:\n{rendered}"
        );
    }
}

/// [05-OP-76] and [05-OP-6]: `decimal_from_f64` produces an exact,
/// non-differentiable value, so `grad` through it is rejected, never answered
/// with a zero cotangent. The same harness differentiates a float function in
/// eval, so the eval rejection belongs to `decimal_from_f64`. The C lane does
/// not lower this `grad` placement for the float control either, so the build
/// assertion only shows that no program is produced.
#[test]
fn std_decimal_grad_through_float_ingest_is_rejected() {
    let (_dir, reef_home, app_pkg) = make_app("decimal-grad-2778");
    let path = app_pkg.join("src/main.ch");
    let program = |body: &str| {
        format!(
            "module Demo.Main\n{IMPORTS}\ndef slope_of(model: f64 -> f64, x: f64) -> f64 = {{\n  target = fn (x_local: f64) -> model(x_local)\n  grad(target, wrt=x_local)(x)\n}}\ndef model(x: f64) -> f64 = {body}\nslope = slope_of(model, 1.5f64)\n"
        )
    };
    write_file(&path, &program("mul(x, 2.0f64)"));
    let (success, rendered) = run_chelis(
        &app_pkg,
        &reef_home,
        &["eval", "--file", path.to_str().unwrap()],
    );
    assert!(
        success && rendered.contains("slope = 2.0"),
        "the float control must differentiate in eval:\n{rendered}"
    );

    write_file(
        &path,
        &program("mul(x, cast(decimal_scale(decimal_from_f64(x, 2i64, RoundTiesToEven)), f64))"),
    );
    let (success, rendered) = run_chelis(
        &app_pkg,
        &reef_home,
        &["eval", "--file", path.to_str().unwrap()],
    );
    assert!(
        !success && rendered.contains("structurally rejects differentiation"),
        "grad through decimal_from_f64 must be a structural rejection in eval:\n{rendered}"
    );
    assert!(
        !rendered.contains("slope ="),
        "grad through decimal_from_f64 must not produce a value:\n{rendered}"
    );

    let out_dir = app_pkg.join("grad-out");
    let (built, rendered) = run_chelis(
        &app_pkg,
        &reef_home,
        &[
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ],
    );
    assert!(
        !built && !out_dir.join("main.c").exists(),
        "grad through decimal_from_f64 must not build:\n{rendered}"
    );
}
