// chelis#2442: a literal pattern against a rigid dtype-binder scrutinee was
// never checked against the binder's instantiations. `| 300 =>` under
// `p: Int`, `| 70000.0 =>` under `p: Float`, and `| 0 =>` under `p: Numeric`
// all scored 1, and the Numeric program evaluated to a wrong answer
// (`is_zero(0.0f32)` was `false`).
//
// spec/04-type-system.md [04-PAT-1] makes a literal pattern a typing
// constraint, [04-LIT-2] binds it at the scrutinee's primitive, and [04-INF-6]
// makes the binder denote every admissible instantiation. These tests drive
// the real CLI through `check`, `eval --file`, and `build` for the C lane:
// each rejected program must not score 1 and must not evaluate, and the
// repair its diagnostic names must, once spliced into the program, check
// clean and evaluate to the value "the scrutinee equals the literal" at float
// and integer instantiations in both lanes. That second half is what proves
// the repair is not advice that fails. A
// comparison repair is an `if` ahead of the match rather than a guard,
// because the eval and C lanes ignore a guard at run time (chelis#2445).
//
// Beside each rejection sit the controls that must stay legal: a pattern
// every member admits, the generic comparison the repairs are built from, a
// concrete scrutinee, and a flexible scrutinee resolved by local application.

use assert_cmd::Command;
use serde_json::Value;
use std::fs;
use std::path::Path;
use std::process::Command as StdCommand;
use tempfile::tempdir;

fn check(path: &Path) -> Value {
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["check", path.to_str().unwrap()])
        .output()
        .expect("run chelis check");
    serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "check output for {} is not JSON ({error}): stdout={} stderr={}",
            path.display(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        )
    })
}

fn eval(path: &Path) -> std::process::Output {
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("run chelis eval --file")
}

/// A binder-generic `f` with one literal arm, applied at each call.
struct Case {
    label: &'static str,
    binders: &'static str,
    pattern: &'static str,
    /// Fragments the rejection must contain besides [04-PAT-1].
    needles: &'static [&'static str],
    /// `(argument, 1 when the argument equals the literal, else 0)`.
    calls: &'static [(&'static str, i32)],
}

const REJECTED: &[Case] = &[
    Case {
        label: "issue program: out of range at i8",
        binders: "[p: Int]",
        pattern: "300",
        needles: &["`p: Int`", "at `i8`"],
        calls: &[
            ("44i8", 0),
            ("300i32", 1),
            ("300i64", 1),
            ("-5i16", 0),
            ("44i64", 0),
        ],
    },
    Case {
        label: "issue program: infinite at f16",
        binders: "[p: Float]",
        pattern: "70000.0",
        needles: &["`p: Float`", "at `f16`"],
        calls: &[
            ("cast(70000.0f32, f16)", 0),
            ("70000.0f32", 1),
            ("70000.0f64", 1),
            ("1.5f32", 0),
        ],
    },
    Case {
        label: "issue program: integer pattern under Numeric",
        binders: "[p: Numeric]",
        pattern: "0",
        needles: &["`p: Numeric`", "at `f32`"],
        calls: &[
            ("0.0f32", 1),
            ("0i64", 1),
            ("3i32", 0),
            ("2.5f64", 0),
            ("3i64", 0),
        ],
    },
    Case {
        label: "integer pattern under Float",
        binders: "[p: Float]",
        pattern: "0",
        needles: &["`p: Float`", "at `f32`"],
        calls: &[("0.0f32", 1), ("3.0f64", 0)],
    },
    Case {
        label: "integral float pattern under Int",
        binders: "[p: Int]",
        pattern: "1.0",
        needles: &["`p: Int`", "at `i8`"],
        calls: &[("1i8", 1), ("2i32", 0), ("1i64", 1), ("2i64", 0)],
    },
    Case {
        label: "integral float pattern under Numeric",
        binders: "[p: Numeric]",
        pattern: "-1.0",
        needles: &["`p: Numeric`", "at `i8`"],
        calls: &[
            ("-1i8", 1),
            ("-1.0f64", 1),
            ("1i32", 0),
            ("-1i64", 1),
            ("1i64", 0),
        ],
    },
    Case {
        label: "integer pattern bf16 rounds, under Float",
        binders: "[p: Float]",
        pattern: "257",
        needles: &["`p: Float`", "at `f32`"],
        calls: &[
            ("257.0f32", 1),
            ("257.0f64", 1),
            ("cast(257.0f32, bf16)", 0),
        ],
    },
    Case {
        label: "out-of-range integer pattern under Numeric",
        binders: "[p: Numeric]",
        pattern: "300",
        needles: &["`p: Numeric`", "at `i8`"],
        calls: &[
            ("300i32", 1),
            ("300.0f32", 1),
            ("44i8", 0),
            ("300i64", 1),
            ("44i64", 0),
        ],
    },
    Case {
        label: "fractional pattern under Numeric",
        binders: "[p: Numeric]",
        pattern: "0.5",
        needles: &["`p: Numeric`", "at `i8`"],
        calls: &[("0.5f32", 1), ("0.5f64", 1), ("3i32", 0), ("3i64", 0)],
    },
    Case {
        label: "fractional pattern under Int matches nothing",
        binders: "[p: Int]",
        pattern: "0.5",
        needles: &["`p: Int`", "at `i8`"],
        calls: &[("0i32", 0), ("1i8", 0), ("0i64", 0)],
    },
    Case {
        label: "unbounded binder",
        binders: "[p]",
        pattern: "1",
        needles: &["`p` declares no dtype-family bound"],
        calls: &[("1i32", 1), ("5i64", 0), ("1i64", 1)],
    },
    Case {
        label: "unbounded binder, out of range once bounded",
        binders: "[p]",
        pattern: "300",
        needles: &["`p` declares no dtype-family bound"],
        calls: &[("300i32", 1), ("44i8", 0), ("300i64", 1), ("44i64", 0)],
    },
];

/// `f` with `before_match` ahead of its `match` and `arm` above the wildcard,
/// applied at each call.
fn program(binders: &str, before_match: &str, arm: &str, calls: &[(&str, i32)]) -> String {
    let mut source = format!(
        "def f{binders}(x: p) -> i32 =\n  {before_match}match x with {{\n{arm}    | _ => 0\n  }}\n"
    );
    for (index, (argument, _)) in calls.iter().enumerate() {
        source.push_str(&format!("r{index} = f({argument})\n"));
    }
    source
}

fn literal_arm(pattern: &str) -> String {
    format!("    | {pattern} => 1\n")
}

/// The backticked spellings in `repair`.
fn spellings(repair: &str) -> impl Iterator<Item = &str> {
    repair.split('`').skip(1).step_by(2)
}

/// The program `repair` asks for, with the literal arm's body `1` and the
/// other arm `| _ => 0`: a deleted arm, a declared bound, a respelled
/// literal, or a comparison in an `if` ahead of the match, each applied when
/// the repair names it.
fn apply_repair(case: &Case, repair: &str, calls: &[(&str, i32)]) -> String {
    let binders = match repair.split_once("Declare `p: ") {
        Some((_, rest)) => {
            let family = rest.split('`').next().expect("a family name");
            format!("[p: {family}]")
        }
        None => case.binders.to_string(),
    };
    let literal = repair
        .starts_with("Write the literal as ")
        .then(|| spellings(repair).next())
        .flatten();
    let condition = spellings(repair).find_map(|span| {
        span.strip_prefix("if ")
            .and_then(|rest| rest.split_once(" then "))
            .map(|(condition, _)| condition)
    });
    let (before_match, arm) = if repair.contains("delete it") {
        (String::new(), String::new())
    } else if let Some(literal) = literal {
        (String::new(), literal_arm(literal))
    } else if let Some(condition) = condition {
        (format!("if {condition} then 1 else "), String::new())
    } else {
        (String::new(), literal_arm(case.pattern))
    };
    program(&binders, &before_match, &arm, calls)
}

/// Whether the C lane can lower a `match` over this argument's dtype. Host
/// lowering rejects a `match` over a narrower integer or float scrutinee
/// loudly today (chelis#2446), so the C leg runs the `i64`, `f32`, and `f64`
/// calls.
fn c_lane_lowers(argument: &str) -> bool {
    ["i64", "f32", "f64"]
        .iter()
        .any(|suffix| argument.ends_with(suffix))
}

/// Build `source` for the C lane, compile it, and return what it prints.
fn run_c_lane(dir: &Path, source: &str) -> String {
    let path = dir.join("lane.ch");
    fs::write(&path, source).expect("write");
    let out_dir = dir.join("lane-c");
    let build = Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "build",
            "--emit-c",
            path.to_str().unwrap(),
            "--target",
            "c",
            "-o",
            out_dir.to_str().unwrap(),
        ])
        .output()
        .expect("run chelis build");
    assert!(
        build.status.success(),
        "chelis build failed:\n{source}\nstderr={}",
        String::from_utf8_lossy(&build.stderr)
    );
    let binary = out_dir.join("lane-bin");
    let compile = StdCommand::new("cc")
        .args([
            "-std=c11",
            "-I",
            out_dir.to_str().unwrap(),
            out_dir.join("lane.c").to_str().unwrap(),
            out_dir.join("libchelis_runtime.a").to_str().unwrap(),
            "-lm",
            "-lpthread",
            "-o",
            binary.to_str().unwrap(),
        ])
        .output()
        .expect("compile emitted C");
    assert!(
        compile.status.success(),
        "C compilation failed:\n{source}\nstderr={}",
        String::from_utf8_lossy(&compile.stderr)
    );
    let run = StdCommand::new(&binary).output().expect("run compiled C");
    assert!(
        run.status.success(),
        "the compiled program failed:\n{source}\nstderr={}",
        String::from_utf8_lossy(&run.stderr)
    );
    String::from_utf8_lossy(&run.stdout).into_owned()
}

/// Assert that `stdout` prints `rK = expected` for every call.
fn assert_results(
    lane: &str,
    label: &str,
    repair: &str,
    source: &str,
    stdout: &str,
    calls: &[(&str, i32)],
) {
    for (index, (argument, expected)) in calls.iter().enumerate() {
        assert!(
            stdout
                .lines()
                .any(|line| line == format!("r{index} = {expected}")),
            "{label} ({lane}): after the repair `{repair}`, f({argument}) must be \
             {expected}:\n{source}\nstdout={stdout}"
        );
    }
}

/// REGRESSION TEST. Every case scored 1 before the fix, and the issue's
/// Numeric program evaluated to `false` for `0.0f32`.
#[test]
fn a_binder_scrutinee_pattern_is_refused_before_any_lane_runs() {
    for case in REJECTED {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("pattern.ch");
        fs::write(
            &path,
            program(case.binders, "", &literal_arm(case.pattern), case.calls),
        )
        .expect("write");

        let report = check(&path);
        assert!(
            report["score"].as_f64().expect("score") < 1.0,
            "{}: must not score 1: {report}",
            case.label
        );
        let errors = report["errors"].as_array().expect("errors");
        assert!(
            errors.iter().any(|error| {
                let message = error["message"].as_str().unwrap_or_default();
                message.contains("[04-PAT-1]")
                    && case.needles.iter().all(|needle| message.contains(needle))
            }),
            "{}: expected a [04-PAT-1] rejection naming {:?}, got {report}",
            case.label,
            case.needles
        );

        let output = eval(&path);
        assert!(
            !output.status.success() && !String::from_utf8_lossy(&output.stdout).contains("r0 ="),
            "{}: eval must refuse the program, stdout={}",
            case.label,
            String::from_utf8_lossy(&output.stdout)
        );
    }
}

/// The repair each rejection names checks clean once applied, and in both the
/// eval and C lanes evaluates to "the scrutinee equals the literal" at every
/// call, including a float and an integer instantiation where the family has
/// both.
#[test]
fn the_named_repair_checks_and_evaluates_correctly() {
    for case in REJECTED {
        let dir = tempdir().expect("tempdir");
        let rejected = dir.path().join("rejected.ch");
        fs::write(
            &rejected,
            program(case.binders, "", &literal_arm(case.pattern), case.calls),
        )
        .expect("write");
        let report = check(&rejected);
        let repair = report["errors"]
            .as_array()
            .expect("errors")
            .iter()
            .find(|error| {
                error["message"]
                    .as_str()
                    .is_some_and(|message| message.contains("[04-PAT-1]"))
            })
            .and_then(|error| error["suggestions"][0].as_str())
            .unwrap_or_else(|| panic!("{}: no repair in {report}", case.label))
            .to_string();

        let repaired = dir.path().join("repaired.ch");
        let source = apply_repair(case, &repair, case.calls);
        fs::write(&repaired, &source).expect("write");
        let report = check(&repaired);
        assert_eq!(
            report["score"].as_f64(),
            Some(1.0),
            "{}: the repair `{repair}` must check clean:\n{source}\n{report}",
            case.label
        );

        let output = eval(&repaired);
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success(),
            "{}: the repaired program must evaluate:\n{source}\nstderr={}",
            case.label,
            String::from_utf8_lossy(&output.stderr)
        );
        assert_results("eval", case.label, &repair, &source, &stdout, case.calls);

        let c_calls: Vec<(&str, i32)> = case
            .calls
            .iter()
            .copied()
            .filter(|(argument, _)| c_lane_lowers(argument))
            .collect();
        assert!(
            !c_calls.is_empty(),
            "{}: the C leg needs a call it can lower",
            case.label
        );
        let c_source = apply_repair(case, &repair, &c_calls);
        let c_stdout = run_c_lane(dir.path(), &c_source);
        assert_results("C", case.label, &repair, &c_source, &c_stdout, &c_calls);
    }
}

/// Programs that must keep checking clean and evaluating correctly.
#[test]
fn legal_controls_check_and_evaluate() {
    let controls: &[(&str, &str, &[&str])] = &[
        (
            "a pattern every Int member admits",
            "def f[p: Int](x: p) -> i32 =\n  match x with {\n    | 100 => 1\n    | _ => 0\n  }\n\
             r0 = f(100i8)\nr1 = f(5i64)\n",
            &["r0 = 1", "r1 = 0"],
        ),
        (
            "a pattern every Float member admits",
            "def f[p: Float](x: p) -> i32 =\n  match x with {\n    | 0.5 => 1\n    | _ => 0\n  }\n\
             r0 = f(0.5f32)\nr1 = f(1.0f64)\n",
            &["r0 = 1", "r1 = 0"],
        ),
        (
            "the generic comparison",
            "def is_zero[p: Numeric](x: p) -> bool = if eq(x, cast(0, p)) then true else false\n\
             a = is_zero(0.0f32)\nb = is_zero(0i64)\nc = is_zero(3i32)\n",
            &["a = true", "b = true", "c = false"],
        ),
        (
            "a concrete scrutinee",
            "def f(x: i8) -> i32 =\n  match x with {\n    | 100 => 1\n    | _ => 0\n  }\n\
             r0 = f(100i8)\n",
            &["r0 = 1"],
        ),
        (
            "a flexible scrutinee resolved by local application",
            "def pick(x: i32) -> i32 = {\n  k = fn (y) -> match y with {\n    | 300 => 1\n    | _ => 0\n  }\n  k(x)\n}\n\
             r0 = pick(300i32)\n",
            &["r0 = 1"],
        ),
    ];
    for (label, source, expected) in controls {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("control.ch");
        fs::write(&path, source).expect("write");
        let report = check(&path);
        assert_eq!(
            report["score"].as_f64(),
            Some(1.0),
            "{label}: must check clean: {report}"
        );
        let output = eval(&path);
        let stdout = String::from_utf8_lossy(&output.stdout);
        for line in *expected {
            assert!(
                stdout.lines().any(|printed| printed == *line),
                "{label}: expected `{line}`, stdout={stdout} stderr={}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }
}
