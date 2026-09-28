//! chelis#2413 step 3: the chelis-std random initialisers take an explicit
//! key ([05-OP-35] as recast for keys), through the published package.
//!
//! Kaiming's uniform is checked against `slice2_ref.py`'s transcription of
//! the [05-OP-35] graph over `key_ref.py`'s [05-RNG-2] words (f32 bounds 0
//! and 1, then `mul(sub(mul(2, u), 1), sqrt(div(6, fan_in)))` at f32).
//! `normal_like` is checked against its [05-OP-35] graph spelled in source:
//! `split_key` halves key u1 and u2. Both run in eval and in native C.
//!
//! [05-OP-35] also requires every random parameter to be validated before any
//! element is drawn, with a `Domain` trap on violation. The validation tests
//! below cover all six exported initialisers in three lanes: the host
//! interpreter (`chelis eval` of a plain call), the DAG evaluator (`chelis
//! eval` of the same call under `grad`, which lowers the body and evaluates
//! it in `chelis_ir::eval`; a plain call stays in the host interpreter), and
//! compiled C for both shapes.

#[path = "common/mod.rs"]
mod common;

use assert_cmd::Command;
use common::{build_and_run_app, link_generated, make_app, write_file};
use std::path::Path;
use std::process::Command as StdCommand;

fn eval_app_stdout(reef_home: &std::path::Path, app_pkg: &std::path::Path) -> String {
    let assert = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", reef_home)
        .current_dir(app_pkg)
        .args([
            "eval",
            "--file",
            app_pkg.join("src/main.ch").to_str().unwrap(),
        ])
        .assert()
        .success();
    String::from_utf8(assert.get_output().stdout.clone()).expect("utf-8 stdout")
}

/// The printed f32 elements of `bits`, in `chelis eval`'s shortest form.
fn f32_elements(bits: &[u32]) -> String {
    bits.iter()
        .map(|bits| format!("{:?}", f32::from_bits(*bits)))
        .collect::<Vec<_>>()
        .join(", ")
}

#[test]
fn std_initialisers_draw_from_the_key_they_are_given_in_eval_and_c() {
    let (_dir, reef_home, app_pkg) = make_app("key-std-initialisers");
    let template = "to_tensor([0.0f32, 0.0f32, 0.0f32, 0.0f32])";
    write_file(
        &app_pkg.join("src/main.ch"),
        &format!(
            "module Demo.Main\n\
             import Std.Init.Kaiming (kaiming_uniform)\n\
             import Std.Init.Random (normal_like)\n\
             def spelled_normal(k: key, mean: f32, std: f32) -> tensor[4, f32] = {{\n\
             \x20 (k1, k2) = split_key(k)\n\
             \x20 u1 = uniform_like(k1, {template}, 1e-7f32, 1.0f32)\n\
             \x20 u2 = uniform_like(k2, {template}, 0.0f32, 1.0f32)\n\
             \x20 radii = map(fn (x: f32) -> sqrt(mul(-2.0f32, log(x))), to_list(u1))\n\
             \x20 cos_terms = map(fn (x: f32) -> cos(mul(6.283185307179586f32, x)), to_list(u2))\n\
             \x20 z = mul(to_tensor(radii), to_tensor(cos_terms))\n\
             \x20 to_tensor(map(fn (x: f32) -> add(mean, mul(std, x)), to_list(z)))\n\
             }}\n\
             def main() = (kaiming_uniform(key_from_seed(11i64), {template}, 4.0f32), \
             normal_like(key_from_seed(42i64), {template}, 0.5f32, 2.0f32), \
             spelled_normal(key_from_seed(42i64), 0.5f32, 2.0f32))\n"
        ),
    );
    let eval = eval_app_stdout(&reef_home, &app_pkg);
    let kaiming = format!(
        "main.0 = tensor(shape=[4], data=[{}])",
        f32_elements(&[0xbf204b20, 0xbea7e74b, 0x3f3f8248, 0xbf6ffe83])
    );
    let lines = eval.lines().collect::<Vec<_>>();
    assert_eq!(lines.first().copied(), Some(kaiming.as_str()), "{eval}");
    let normal = |line: &str, index: usize| {
        line.strip_prefix(&format!("main.{index} = "))
            .unwrap_or_else(|| panic!("root {index}: {line}"))
            .to_string()
    };
    assert_eq!(normal(lines[1], 1), normal(lines[2], 2), "{eval}");
    let native = build_and_run_app(&reef_home, &app_pkg, "main");
    assert_eq!(native.trim_end(), eval.trim_end(), "eval and C disagree");
}

/// The validation's trap: each initialiser maps an invalid parameter to the
/// integer 2 and casts it to `bool`, which traps `Domain`.
const VALIDATION_TRAP: &str = "numeric trap: domain in cast at bool";

/// The trap of the later statement that consumes the draw: `1000` does not
/// fit `i8`. Every draw value `d` reaches it as `d * 0 + 1000`, so it traps
/// whenever it runs, and a non-finite draw traps `Domain` in the same cast.
const LATER_TRAP: &str = "numeric trap: overflow in cast at i8";
const LATER_CAST: &str = "in cast at i8";

/// One exported initialiser, its extra parameters, and an invalid actual that
/// only the initialiser's own validation rejects. For the three initialisers
/// that call `normal_like`, the invalid actual yields a valid `normal_like`
/// call (`std = sqrt(2 / inf) = 0`, or a finite `mean` and `std` with
/// `a > b`), so `normal_like`'s own validation cannot be what traps.
struct Initialiser {
    name: &'static str,
    import: &'static str,
    formals: &'static str,
    call: &'static str,
    wrt: &'static str,
    invalid: &'static str,
    valid: &'static str,
}

const INITIALISERS: &[Initialiser] = &[
    Initialiser {
        name: "normal_like",
        import: "Std.Init.Random (normal_like)",
        formals: "mean: f32, std: f32",
        call: "normal_like(k, template, mean, std)",
        wrt: "std",
        invalid: "0.0f32, -1.0f32",
        valid: "0.0f32, 1.0f32",
    },
    Initialiser {
        name: "kaiming_uniform",
        import: "Std.Init.Kaiming (kaiming_uniform)",
        formals: "fan_in: f32",
        call: "kaiming_uniform(k, template, fan_in)",
        wrt: "fan_in",
        invalid: "0.0f32",
        valid: "4.0f32",
    },
    Initialiser {
        name: "kaiming_normal",
        import: "Std.Init.Kaiming (kaiming_normal)",
        formals: "fan_in: f32",
        call: "kaiming_normal(k, template, fan_in)",
        wrt: "fan_in",
        invalid: "div(1.0f32, 0.0f32)",
        valid: "4.0f32",
    },
    Initialiser {
        name: "xavier_uniform",
        import: "Std.Init.XavierExt (xavier_uniform)",
        formals: "fan_in: f32, fan_out: f32",
        call: "xavier_uniform(k, template, fan_in, fan_out)",
        wrt: "fan_in",
        invalid: "1.0f32, -1.0f32",
        valid: "2.0f32, 2.0f32",
    },
    Initialiser {
        name: "xavier_normal",
        import: "Std.Init.XavierExt (xavier_normal)",
        formals: "fan_in: f32, fan_out: f32",
        call: "xavier_normal(k, template, fan_in, fan_out)",
        wrt: "fan_in",
        invalid: "div(1.0f32, 0.0f32), 1.0f32",
        valid: "2.0f32, 2.0f32",
    },
    Initialiser {
        name: "trunc_normal",
        import: "Std.Init.XavierExt (trunc_normal)",
        formals: "mean: f32, std: f32, a: f32, b: f32",
        call: "trunc_normal(k, template, mean, std, a, b)",
        wrt: "mean",
        invalid: "0.0f32, 1.0f32, 2.0f32, 1.0f32",
        valid: "0.0f32, 1.0f32, -2.0f32, 2.0f32",
    },
];

const TEMPLATE: &str = "to_tensor([0.0f32, 0.0f32, 0.0f32, 0.0f32])";
const LATER_STATEMENT: &str = "cast(add(mul(&drawn, to_tensor([0.0f32, 0.0f32, 0.0f32, 0.0f32])), to_tensor([1000.0f32, 1000.0f32, 1000.0f32, 1000.0f32])), i8)";

/// The host-interpreter shape: the draw, then a statement that consumes it.
fn plain_source(initialiser: &Initialiser, actuals: &str) -> String {
    format!(
        "module Demo.Main\n\
         import {import}\n\
         def consumed(k: key, template: tensor[4, f32], {formals}) -> tensor[4, i8] = {{\n\
         \x20 drawn = {call}\n\
         \x20 {LATER_STATEMENT}\n\
         }}\n\
         def main() -> tensor[4, i8] = consumed(key_from_seed(42i64), {TEMPLATE}, {actuals})\n",
        import = initialiser.import,
        formals = initialiser.formals,
        call = initialiser.call,
    )
}

/// The DAG-evaluator shape: the same draw and later statement, evaluated
/// through `grad`. The later statement's result is discarded, so it is a
/// potentially trapping observable root of the lowered graph.
fn grad_source(initialiser: &Initialiser, actuals: &str) -> String {
    format!(
        "module Demo.Main\n\
         import {import}\n\
         def total(k: key, template: tensor[4, f32], {formals}) -> f32 = {{\n\
         \x20 drawn = {call}\n\
         \x20 _ = {LATER_STATEMENT}\n\
         \x20 tensor_to_scalar(sum(drawn, 0i32))\n\
         }}\n\
         gradient = grad(total, wrt={wrt})(key_from_seed(42i64), {TEMPLATE}, {actuals})\n",
        import = initialiser.import,
        formals = initialiser.formals,
        call = initialiser.call,
        wrt = initialiser.wrt,
    )
}

/// Exit status and combined stdout and stderr of one lane's run.
struct LaneRun {
    success: bool,
    text: String,
}

fn lane_run(output: &std::process::Output) -> LaneRun {
    LaneRun {
        success: output.status.success(),
        text: format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ),
    }
}

/// `chelis eval --file`, with the style gate on: the sources are canonical.
fn eval_lane(reef_home: &Path, app_pkg: &Path) -> LaneRun {
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", reef_home)
        .current_dir(app_pkg)
        .args(["eval", "--file", "src/main.ch"])
        .output()
        .expect("chelis eval runs");
    lane_run(&output)
}

/// `chelis build --target c`, link, and run. A build or link failure is a
/// lane failure reported with its text, never a trap.
fn c_lane(reef_home: &Path, app_pkg: &Path) -> LaneRun {
    let out_dir = app_pkg.join("main-out");
    let built = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", reef_home)
        .current_dir(app_pkg)
        .args([
            "build",
            "src/main.ch",
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .output()
        .expect("chelis build runs");
    if !built.status.success() {
        let run = lane_run(&built);
        return LaneRun {
            success: false,
            text: format!("BUILD FAILED: {}", run.text),
        };
    }
    if !link_generated(&out_dir, "main.c", "main").success() {
        return LaneRun {
            success: false,
            text: "LINK FAILED".to_string(),
        };
    }
    let ran = StdCommand::new(out_dir.join("main"))
        .output()
        .expect("compiled binary runs");
    lane_run(&ran)
}

/// Run `source` in eval and C and record every lane whose outcome is not
/// `expected` trap text free of `forbidden`.
fn expect_trap(
    failures: &mut Vec<String>,
    label: &str,
    source: &str,
    expected: &str,
    forbidden: &str,
) {
    let (_dir, reef_home, app_pkg) = make_app("key-std-validation");
    write_file(&app_pkg.join("src/main.ch"), source);
    for (lane, run) in [
        ("eval", eval_lane(&reef_home, &app_pkg)),
        ("C", c_lane(&reef_home, &app_pkg)),
    ] {
        if run.success || !run.text.contains(expected) || run.text.contains(forbidden) {
            failures.push(format!(
                "{label} [{lane}]: expected a failure with `{expected}` and no `{forbidden}`, got success={}:\n{}",
                run.success, run.text
            ));
        }
    }
}

fn check_validation_order(shape: &str, source: fn(&Initialiser, &str) -> String) {
    let mut failures = Vec::new();
    for initialiser in INITIALISERS {
        expect_trap(
            &mut failures,
            &format!(
                "{shape} {} invalid ({})",
                initialiser.name, initialiser.invalid
            ),
            &source(initialiser, initialiser.invalid),
            VALIDATION_TRAP,
            LATER_CAST,
        );
        expect_trap(
            &mut failures,
            &format!("{shape} {} valid ({})", initialiser.name, initialiser.valid),
            &source(initialiser, initialiser.valid),
            LATER_TRAP,
            VALIDATION_TRAP,
        );
    }
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}

/// An invalid parameter traps the initialiser's validation before a later
/// statement that consumes the draw can run: that statement traps whenever it
/// runs, so its trap text in the transcript would mean the validation ran
/// late or not at all. The valid twin shows the later statement is live.
#[test]
fn std_initialisers_validate_before_the_draw_is_consumed_in_eval_and_c() {
    check_validation_order("plain", plain_source);
}

/// The same order through `grad`, the shape that runs the initialiser in the
/// DAG evaluator, and in C.
#[test]
fn std_initialisers_validate_before_the_draw_is_consumed_under_grad_in_eval_and_c() {
    check_validation_order("grad", grad_source);
}

/// The completing valid twins: every initialiser with its valid parameters
/// runs to a value, with no later statement, in the host interpreter and C,
/// and under `grad` in the DAG evaluator and C. Kaiming's uniform keeps the
/// case the first test checks against the reference bits.
#[test]
fn std_initialisers_with_valid_parameters_run_in_eval_and_c() {
    let calls = INITIALISERS
        .iter()
        .map(|initialiser| {
            let call = initialiser
                .call
                .split_once('(')
                .map(|(name, _)| name)
                .expect("a call");
            let actuals = if initialiser.name == "kaiming_uniform" {
                format!("key_from_seed(11i64), {TEMPLATE}, 4.0f32")
            } else {
                format!("key_from_seed(42i64), {TEMPLATE}, {}", initialiser.valid)
            };
            format!("{call}({actuals})")
        })
        .collect::<Vec<_>>();
    let imports = INITIALISERS
        .iter()
        .map(|initialiser| format!("import {}\n", initialiser.import))
        .collect::<String>();
    let plain = format!(
        "module Demo.Main\n{imports}def main() = ({})\n",
        calls.join(", ")
    );
    let totals = INITIALISERS
        .iter()
        .map(|initialiser| {
            format!(
                "def total_{name}(k: key, template: tensor[4, f32], {formals}) -> f32 = tensor_to_scalar(sum({call}, 0i32))\n",
                name = initialiser.name,
                formals = initialiser.formals,
                call = initialiser.call,
            )
        })
        .collect::<String>();
    let gradients = INITIALISERS
        .iter()
        .map(|initialiser| {
            format!(
                "gradient_{name} = grad(total_{name}, wrt={wrt})(key_from_seed(42i64), {TEMPLATE}, {valid})\n",
                name = initialiser.name,
                wrt = initialiser.wrt,
                valid = initialiser.valid,
            )
        })
        .collect::<String>();
    let graded = format!("module Demo.Main\n{imports}{totals}{gradients}");

    let kaiming = format!(
        "main.1 = tensor(shape=[4], data=[{}])",
        f32_elements(&[0xbf204b20, 0xbea7e74b, 0x3f3f8248, 0xbf6ffe83])
    );
    for (shape, source, rows) in [("plain", plain, 6), ("grad", graded, 6)] {
        let (_dir, reef_home, app_pkg) = make_app("key-std-valid");
        write_file(&app_pkg.join("src/main.ch"), &source);
        let eval = eval_lane(&reef_home, &app_pkg);
        assert!(
            eval.success,
            "{shape} eval failed:\n{}\n{source}",
            eval.text
        );
        assert_eq!(
            eval.text.lines().count(),
            rows,
            "{shape}: one row per initialiser:\n{}",
            eval.text
        );
        if shape == "plain" {
            assert!(
                eval.text.lines().any(|line| line == kaiming),
                "{}",
                eval.text
            );
        }
        let native = c_lane(&reef_home, &app_pkg);
        assert!(native.success, "{shape} C failed:\n{}", native.text);
        assert_eq!(
            native.text.trim_end(),
            eval.text.trim_end(),
            "{shape}: eval and C disagree"
        );
    }
}
