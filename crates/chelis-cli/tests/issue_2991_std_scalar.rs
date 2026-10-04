//! [05-OP-35]: exact special-value selection and abs' specified adjoint.
use assert_cmd::Command;

#[path = "common/mod.rs"]
mod common;
use common::{build_and_run_app, make_app, write_file};

fn eval(reef: &std::path::Path, app: &std::path::Path) -> String {
    let output = Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", reef)
        .current_dir(app)
        .args(["eval", "--file", app.join("src/main.ch").to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

#[test]
fn scalar_special_values_match_the_contract_in_eval_and_c() {
    for dtype in ["f16", "bf16", "f32", "f64"] {
        let (_dir, reef, app) = make_app("issue-2991-values");
        let source = r#"module Demo.Main
import Std.Scalar (abs, max, min)
a = abs(-0.0DTYPE)
b = abs(neg(div(1.0DTYPE, 0.0DTYPE)))
c = abs(-2.5DTYPE)
d = abs(div(0.0DTYPE, 0.0DTYPE))
e = max(div(0.0DTYPE, 0.0DTYPE), 1.0DTYPE)
f = max(1.0DTYPE, div(0.0DTYPE, 0.0DTYPE))
g = min(div(0.0DTYPE, 0.0DTYPE), 1.0DTYPE)
h = min(1.0DTYPE, div(0.0DTYPE, 0.0DTYPE))
i = max(0.0DTYPE, -0.0DTYPE)
j = max(-0.0DTYPE, 0.0DTYPE)
k = min(0.0DTYPE, -0.0DTYPE)
l = min(-0.0DTYPE, 0.0DTYPE)
m = max(-3.0DTYPE, 2.0DTYPE)
n = min(2.0DTYPE, -3.0DTYPE)
o = abs(0.0DTYPE)
"#
        .replace("DTYPE", dtype);
        write_file(&app.join("src/main.ch"), &source);
        let expected = "a = 0.0\nb = inf\nc = 2.5\nd = NaN\ne = NaN\nf = NaN\ng = NaN\nh = NaN\ni = 0.0\nj = -0.0\nk = 0.0\nl = -0.0\nm = 2.0\nn = -3.0\no = 0.0\n";
        assert_eq!(eval(&reef, &app), expected, "eval {dtype}");
        assert_eq!(
            build_and_run_app(&reef, &app, "main"),
            expected,
            "C {dtype}"
        );
    }
}

#[test]
fn scalar_adjoints_select_the_first_tie_and_zero_abs_at_both_zeros() {
    for dtype in ["f16", "bf16", "f32", "f64"] {
        let (_dir, reef, app) = make_app("issue-2991-grad");
        // #2995's reproducer is the loose-file import route. The package
        // route's separate scalar-transform refusal is outside this repair.
        std::fs::remove_file(app.join("reef.toml")).unwrap();
        write_file(
            &app.join("src/main.ch"),
            &r#"module Demo.Main
import Std.Scalar (abs, max, min)
def magnitude(x: DTYPE) -> DTYPE = abs(x)
def high(x: DTYPE, y: DTYPE) -> DTYPE = max(x, y)
def low(x: DTYPE, y: DTYPE) -> DTYPE = min(x, y)
a = grad(magnitude)(0.0DTYPE)
b = grad(magnitude)(-0.0DTYPE)
c = grad(magnitude)(-2.0DTYPE)
d = grad(magnitude)(2.0DTYPE)
e = grad(high)(1.0DTYPE, 1.0DTYPE)
f = grad(low)(1.0DTYPE, 1.0DTYPE)
g = grad(high)(1.0DTYPE, 2.0DTYPE)
h = grad(low)(1.0DTYPE, 2.0DTYPE)
i = grad(magnitude)(div(0.0DTYPE, 0.0DTYPE))
j = grad(magnitude)(div(1.0DTYPE, 0.0DTYPE))
k = grad(magnitude)(neg(div(1.0DTYPE, 0.0DTYPE)))
"#
            .replace("DTYPE", dtype),
        );
        let expected = "a = 0.0\nb = 0.0\nc = -1.0\nd = 1.0\ne.0 = 1.0\ne.1 = 0.0\nf.0 = 1.0\nf.1 = 0.0\ng.0 = 0.0\ng.1 = 1.0\nh.0 = 1.0\nh.1 = 0.0\ni = 0.0\nj = 1.0\nk = -1.0\n";
        assert_eq!(eval(&reef, &app), expected, "eval {dtype}");
        assert_eq!(
            build_and_run_app(&reef, &app, "main"),
            expected,
            "C {dtype}"
        );
    }
}

#[test]
fn integer_abs_checks_the_signed_minimum() {
    for (dtype, minimum) in [
        ("i8", "-128i64"),
        ("i16", "-32768i64"),
        ("i32", "-2147483648i64"),
        ("i64", "-9223372036854775808i64"),
    ] {
        let (_dir, reef, app) = make_app("issue-2991-int");
        write_file(
            &app.join("src/main.ch"),
            &format!(
                "module Demo.Main\nimport Std.Scalar (abs, max, min)\na = abs(cast(-3, {dtype}))\nb = max(cast(-3, {dtype}), cast(2, {dtype}))\nc = min(cast(2, {dtype}), cast(-3, {dtype}))\n"
            ),
        );
        let expected = "a = 3\nb = 2\nc = -3\n";
        assert_eq!(eval(&reef, &app), expected);
        assert_eq!(build_and_run_app(&reef, &app, "main"), expected);
        write_file(
            &app.join("src/main.ch"),
            &format!(
                "module Demo.Main\nimport Std.Scalar (abs)\na = abs(cast({minimum}, {dtype}))\n"
            ),
        );
        let result = Command::cargo_bin("chelis")
            .unwrap()
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .env("CHELIS_REEF_HOME", &reef)
            .current_dir(&app)
            .args(["eval", "--file", app.join("src/main.ch").to_str().unwrap()])
            .output()
            .unwrap();
        assert!(!result.status.success(), "{dtype} minimum did not trap");
        assert!(
            String::from_utf8_lossy(&result.stderr).contains("overflow"),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let out = app.join("overflow-out");
        Command::cargo_bin("chelis")
            .unwrap()
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .env("CHELIS_REEF_HOME", &reef)
            .args([
                "build",
                app.join("src/main.ch").to_str().unwrap(),
                "--target",
                "c",
                "--output",
                out.to_str().unwrap(),
            ])
            .assert()
            .success();
        let result = std::process::Command::new(out.join("main"))
            .output()
            .unwrap();
        assert!(!result.status.success(), "C {dtype} minimum did not trap");
        assert!(
            String::from_utf8_lossy(&result.stderr).contains("overflow"),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
    }
}

#[test]
fn composed_abs_matches_the_primitive_with_nonunit_cotangents() {
    for dtype in ["f16", "bf16", "f32", "f64"] {
        let (_dir, reef, app) = make_app("issue-2991-cotangents");
        std::fs::remove_file(app.join("reef.toml")).unwrap();
        let mut source = String::from("module Demo.Main\nimport Std.Scalar\n");
        for (i, coefficient) in ["-2.0", "2.0", "div(1.0, 0.0)"].iter().enumerate() {
            source.push_str(&format!("def wrapper_{i}(x: {dtype}) -> {dtype} = mul(cast({coefficient}, {dtype}), Std.Scalar.abs(x))\ndef core_{i}(x: {dtype}) -> {dtype} = mul(cast({coefficient}, {dtype}), abs(x))\n"));
            for (j, x) in ["-2.0", "2.0", "0.0", "-0.0", "div(0.0, 0.0)"]
                .iter()
                .enumerate()
            {
                source.push_str(&format!("wrapper_{i}_{j} = grad(wrapper_{i})(cast({x}, {dtype}))\ncore_{i}_{j} = grad(core_{i})(cast({x}, {dtype}))\n"));
            }
        }
        write_file(&app.join("src/main.ch"), &source);
        let check_pairs = |output: &str| {
            for i in 0..3 {
                for j in 0..5 {
                    let value = |prefix: String| {
                        output
                            .lines()
                            .find_map(|line| line.strip_prefix(&prefix))
                            .unwrap()
                    };
                    assert_eq!(
                        value(format!("wrapper_{i}_{j} = ")),
                        value(format!("core_{i}_{j} = ")),
                        "{dtype} coefficient {i} input {j}: {output}"
                    );
                }
            }
        };
        let evaluated = eval(&reef, &app);
        check_pairs(&evaluated);
        let compiled = build_and_run_app(&reef, &app, "main");
        check_pairs(&compiled);
        assert_eq!(compiled, evaluated, "{dtype}");
    }
}
