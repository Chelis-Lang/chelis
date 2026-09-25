//! chelis#2413 step 3: the chelis-std random initialisers take an explicit
//! key ([05-OP-35] as recast for keys), through the published package.
//!
//! Kaiming's uniform is checked against `slice2_ref.py`'s transcription of
//! the [05-OP-35] graph over `key_ref.py`'s [05-RNG-2] words (f32 bounds 0
//! and 1, then `mul(sub(mul(2, u), 1), sqrt(div(6, fan_in)))` at f32).
//! `normal_like` is checked against its [05-OP-35] graph spelled in source:
//! `split_key` halves key u1 and u2. Both run in eval and in native C.

#[path = "common/mod.rs"]
mod common;

use assert_cmd::Command;
use common::{build_and_run_app, make_app, write_file};

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
