//! Command-line acceptance for the keyed `uniform_like` primitive.
//!
//! Chelis exposes keyed random primitives; School builds its initializer
//! functions from those primitives. These cases check the primitive's key,
//! bound-validation, evaluator, and generated-C behavior against [05-OP-8].

use std::path::Path;
use std::process::{Command as StdCommand, Output};

use assert_cmd::Command;

#[path = "common/mod.rs"]
mod common;

use common::{build_and_run_app, link_generated, make_app, parse_tensor_data, write_file};

const TEMPLATE: &str = "to_tensor([0.0f32, 0.0f32, 0.0f32, 0.0f32])";
const LATER_OVERFLOW: &str = "cast(add(mul(&drawn, \
    to_tensor([0.0f32, 0.0f32, 0.0f32, 0.0f32])), \
    to_tensor([1000.0f32, 1000.0f32, 1000.0f32, 1000.0f32])), i8)";

fn eval_app(reef_home: &Path, app_pkg: &Path) -> Output {
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", reef_home)
        .current_dir(app_pkg)
        .args(["eval", "--file", "src/main.ch"])
        .output()
        .expect("chelis eval runs")
}

fn c_app(reef_home: &Path, app_pkg: &Path, name: &str) -> Output {
    let out_dir = app_pkg.join(format!("{name}-out"));
    let built = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", reef_home)
        .current_dir(app_pkg)
        .args([
            "build",
            "--emit-c",
            "src/main.ch",
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .output()
        .expect("chelis build runs");
    if !built.status.success() {
        return built;
    }
    let linked = link_generated(&out_dir, &format!("{name}.c"), name);
    assert!(linked.success(), "generated C did not link: {linked}");
    StdCommand::new(out_dir.join(name))
        .output()
        .expect("compiled binary runs")
}

fn output_text(output: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

fn assert_draw_matches_reference(stdout: &str, seed: i64) {
    let actual = parse_tensor_data(stdout, "sampled")
        .into_iter()
        .map(|value| (value as f32).to_bits())
        .collect::<Vec<_>>();
    let expected = common::key_ref::uniform_f32(common::key_ref::key_from_seed(seed), 4, 2.0, 5.0)
        .into_iter()
        .map(f32::to_bits)
        .collect::<Vec<_>>();
    assert_eq!(actual, expected, "{stdout}");
}

fn source_with_validation_order(graded: bool, low: &str, high: &str) -> String {
    if graded {
        format!(
            "module Demo.Main\n\
             def total(k: key, template: tensor[4, f32], low: f32, high: f32) -> f32 = {{\n\
             \x20 drawn = uniform_like(k, template, low, high)\n\
             \x20 _ = {LATER_OVERFLOW}\n\
             \x20 tensor_to_scalar(sum(drawn, 0i32))\n\
             }}\n\
             gradient = grad(total, wrt=low)(key_from_seed(42i64), {TEMPLATE}, {low}, {high})\n"
        )
    } else {
        format!(
            "module Demo.Main\n\
             def consumed(k: key, template: tensor[4, f32], low: f32, high: f32) -> tensor[4, i8] = {{\n\
             \x20 drawn = uniform_like(k, template, low, high)\n\
             \x20 {LATER_OVERFLOW}\n\
             }}\n\
             result = consumed(key_from_seed(42i64), {TEMPLATE}, {low}, {high})\n"
        )
    }
}

fn assert_validation_precedes_later_trap(graded: bool, low: &str, high: &str) {
    let (_dir, reef_home, app_pkg) = make_app("key-random-validation-order");
    write_file(
        &app_pkg.join("src/main.ch"),
        &source_with_validation_order(graded, low, high),
    );

    for (lane, output) in [
        ("eval", eval_app(&reef_home, &app_pkg)),
        ("C", c_app(&reef_home, &app_pkg, "main")),
    ] {
        let text = output_text(&output);
        assert!(
            !output.status.success(),
            "{lane} accepted invalid uniform_like bounds:\n{text}"
        );
        assert!(
            text.to_ascii_lowercase().contains("domain") && text.contains("uniform_like"),
            "{lane} must report the uniform_like Domain trap:\n{text}"
        );
        assert!(
            !text.contains("overflow in cast at i8"),
            "{lane} reached the later cast before rejecting the draw:\n{text}"
        );
    }
}

#[test]
fn uniform_like_draws_from_its_key_in_eval_and_c() {
    let (_dir, reef_home, app_pkg) = make_app("key-random-reference");
    write_file(
        &app_pkg.join("src/main.ch"),
        &format!(
            "module Demo.Main\n\
             sampled = uniform_like(key_from_seed(7i64), {TEMPLATE}, 2.0f32, 5.0f32)\n"
        ),
    );

    let eval = eval_app(&reef_home, &app_pkg);
    assert!(
        eval.status.success(),
        "eval failed:\n{}",
        output_text(&eval)
    );
    let eval_text = output_text(&eval);
    assert_draw_matches_reference(&eval_text, 7);

    let native = build_and_run_app(&reef_home, &app_pkg, "main");
    assert_draw_matches_reference(&native, 7);
}

#[test]
fn uniform_like_rejects_invalid_bounds_before_later_work_in_eval_and_c() {
    assert_validation_precedes_later_trap(false, "2.0f32", "1.0f32");
}

#[test]
fn uniform_like_rejects_invalid_bounds_before_later_work_under_grad_in_eval_and_c() {
    assert_validation_precedes_later_trap(true, "2.0f32", "1.0f32");
}

#[test]
fn uniform_like_valid_bounds_run_in_eval_c_and_grad() {
    let (_dir, reef_home, app_pkg) = make_app("key-random-valid-bounds");
    write_file(
        &app_pkg.join("src/main.ch"),
        &format!(
            "module Demo.Main\n\
             def total(k: key, template: tensor[4, f32], low: f32, high: f32) -> f32 =\n\
             \x20 tensor_to_scalar(sum(uniform_like(k, template, low, high), 0i32))\n\
             sampled = uniform_like(key_from_seed(7i64), {TEMPLATE}, 2.0f32, 5.0f32)\n\
             gradient = grad(total, wrt=low)(key_from_seed(7i64), {TEMPLATE}, 2.0f32, 5.0f32)\n"
        ),
    );

    let eval = eval_app(&reef_home, &app_pkg);
    assert!(
        eval.status.success(),
        "eval failed:\n{}",
        output_text(&eval)
    );
    let eval_text = output_text(&eval);
    assert_draw_matches_reference(&eval_text, 7);
    let eval_gradient = scalar_root(&eval_text, "gradient");

    let native = build_and_run_app(&reef_home, &app_pkg, "main");
    assert_draw_matches_reference(&native, 7);
    let native_gradient = scalar_root(&native, "gradient");
    assert!(
        (eval_gradient - native_gradient).abs() <= 1e-6,
        "gradient differs between eval ({eval_gradient}) and C ({native_gradient}):\n\
         eval:\n{eval_text}\nC:\n{native}"
    );
}

fn scalar_root(stdout: &str, name: &str) -> f64 {
    stdout
        .lines()
        .find_map(|line| {
            line.strip_prefix(&format!("{name} = "))
                .and_then(|value| value.trim().parse::<f64>().ok())
        })
        .unwrap_or_else(|| panic!("output has no scalar `{name}` root:\n{stdout}"))
}
