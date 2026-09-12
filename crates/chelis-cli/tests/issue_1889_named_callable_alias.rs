//! Statically known lexical aliases preserve the direct helper's callable
//! identity and authored shape claims (spec/04 §4.7 and §8.6, Surf P5).

use assert_cmd::Command;
use serde_json::json;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

const ALIGNED: &str = "def aligned[d](x: tensor[d, f32], gain: tensor[fixed, f32]) -> tensor[d, f32] = mul(x, gain)\n";
const ARGS: &str = "to_tensor([1.0f32, 2.0f32]), to_tensor([1.0f32, 2.0f32])";

fn assert_lanes(source: &str, shape: &[usize], bits: &[&str], native: &str) {
    assert!(
        common::gcc_available(),
        "this oracle requires native execution"
    );
    let dir = tempdir().unwrap();
    let path = dir.path().join("alias.ch");
    common::write_file(&path, source);
    Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .arg("check")
        .arg(&path)
        .assert()
        .success();
    let output = Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--json", "--timeout", "10", "--file"])
        .arg(&path)
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let value: serde_json::Value = serde_json::from_slice(&output).unwrap();
    let names = if source.contains("def use()") {
        vec!["use", "out"]
    } else {
        vec!["out"]
    };
    let roots = value["roots"].as_array().unwrap();
    assert_eq!(roots.len(), names.len(), "{value}");
    for (root, name) in roots.iter().zip(&names) {
        assert_eq!(root["name"], *name);
        assert_eq!(
            root["value"]["value"],
            json!({"shape": shape, "data": {"dtype": "f32", "bits": bits}}),
            "{source}"
        );
    }
    let expected = names
        .iter()
        .map(|name| format!("{name}{}", native.strip_prefix("out").unwrap()))
        .collect::<String>();
    assert_eq!(common::build_and_run(source, "alias"), expected, "{source}");
}

#[test]
fn direct_and_three_alias_forms_agree_on_native_values() {
    assert_lanes(
        include_str!("../../../examples/named_callable_alias.ch"),
        &[],
        &["40a00000"],
        "out = 5.0\n",
    );
    for body in [
        format!("sum(aligned({ARGS}), fixed)"),
        format!("{{ f = aligned\n sum(f({ARGS}), fixed) }}"),
        format!("{{ f = aligned\n sum(f({ARGS}), 0i32) }}"),
    ] {
        assert_lanes(
            &format!("{ALIGNED}out = {body}\n"),
            &[],
            &["40a00000"],
            "out = 5.0\n",
        );
    }
    for alias in ["f", "sum"] {
        assert_lanes(
            &format!("{ALIGNED}out = {{ {alias} = aligned\n {alias}({ARGS}) }}\n"),
            &[2],
            &["3f800000", "40800000"],
            "out = tensor(shape=[2], data=[1.0, 4.0])\n",
        );
    }
}

#[test]
fn chained_aliases_in_later_bindings_keep_independent_instantiations() {
    for operation in ["mul(x, gain)", "mul(gain, x)"] {
        let source = format!(
            "{}def use() = {{\n f = aligned\n g = f\n\
             a = sum(g({ARGS}), fixed)\n\
             b = sum(g(to_tensor([1.0f32, 2.0f32, 3.0f32]), to_tensor([1.0f32, 2.0f32, 3.0f32])), fixed)\n\
             add(a, b)\n}}\nout = use()\n",
            ALIGNED.replace("mul(x, gain)", operation)
        );
        assert_lanes(&source, &[], &["41980000"], "out = 19.0\n");
    }
}

#[test]
fn alias_captures_target_before_same_named_local_binding() {
    let source = format!(
        "{ALIGNED}def use() = {{\n f = aligned\n\
         aligned = fn (x: tensor[2, f32], gain: tensor[2, f32]) -> add(x, gain)\n\
         a = sum(f({ARGS}), fixed)\n b = sum(aligned({ARGS}), 0i32)\n\
         add(a, b)\n}}\nout = use()\n"
    );
    assert_lanes(&source, &[], &["41300000"], "out = 11.0\n");
}

#[test]
fn captured_definition_uses_its_global_helper_not_callers_alias() {
    let source = "def scale(x: tensor[2, f32]) -> tensor[2, f32] = mul(x, x)\n\
         def plus(x: tensor[2, f32]) -> tensor[2, f32] = add(x, x)\n\
         def wrapper(x: tensor[2, f32]) -> tensor[2, f32] = scale(x)\n\
         out = {\n f = wrapper\n scale = plus\n\
         a = sum(f(to_tensor([1.0f32, 2.0f32])), 0i32)\n\
         b = sum(scale(to_tensor([1.0f32, 2.0f32])), 0i32)\n add(a, b)\n}\n";
    assert_lanes(source, &[], &["41300000"], "out = 11.0\n");
}

#[test]
fn inner_alias_shadowing_restores_the_outer_callable() {
    let source = format!(
        "{ALIGNED}def plus(x: tensor[2, f32], gain: tensor[2, f32]) -> tensor[2, f32] = add(x, gain)\n\
         out = {{\n f = aligned\n a = {{ f = plus\n sum(f({ARGS}), 0i32) }}\n\
         add(a, sum(f({ARGS}), fixed))\n}}\n"
    );
    assert_lanes(&source, &[], &["41300000"], "out = 11.0\n");
    let source = format!(
        "{ALIGNED}out = {{ f = aligned\n\
         a = (fn (f: tensor[2, f32]) -> sum(f, 0i32))(to_tensor([2.0f32, 4.0f32]))\n\
         add(a, sum(f({ARGS}), fixed))\n}}\n"
    );
    assert_lanes(&source, &[], &["41300000"], "out = 11.0\n");
}

#[test]
fn alias_retains_authored_result_extent_guard_and_agreeing_control() {
    let source = |gain: &str| {
        format!(
            "def narrow[d](x: tensor[r, f32], gain: tensor[d, f32]) -> tensor[d, f32] = shrink(x, [[1i64, shape(x, 0i32)]])\n\
         out = {{ f = narrow\n f(to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32]), to_tensor([{gain}])) }}\n"
        )
    };
    assert_lanes(
        &source("1.0f32, 2.0f32, 3.0f32"),
        &[3],
        &["40000000", "40400000", "40800000"],
        "out = tensor(shape=[3], data=[2.0, 3.0, 4.0])\n",
    );
    let dir = tempdir().unwrap();
    let path = dir.path().join("mismatch.ch");
    let out = dir.path().join("c");
    common::write_file(&path, &source("1.0f32, 2.0f32"));
    let eval = Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--timeout", "10", "--file"])
        .arg(&path)
        .assert()
        .failure()
        .get_output()
        .clone();
    Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .arg("build")
        .arg(&path)
        .args(["--target", "c", "--output"])
        .arg(&out)
        .assert()
        .success();
    assert!(common::link_generated(&out, "mismatch.c", "mismatch").success());
    let native = std::process::Command::new(out.join("mismatch"))
        .output()
        .unwrap();
    assert!(!native.status.success());
    for output in [eval, native] {
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(text.contains("claimed = 2, shrink axis 0 = 3"), "{text}");
    }
}

#[test]
fn noncallable_shadow_and_unresolved_cycles_do_not_become_known_aliases() {
    for (body, kind) in [
        (
            "{ aligned = 1.0f32\n f = aligned\n f(to_tensor([1.0f32]), to_tensor([1.0f32])) }",
            "TypeMismatch",
        ),
        (
            "{ f = g\n g = f\n f(to_tensor([1.0f32]), to_tensor([1.0f32])) }",
            "UnboundVariable",
        ),
    ] {
        let dir = tempdir().unwrap();
        let path = dir.path().join("invalid.ch");
        common::write_file(&path, &format!("{ALIGNED}out = {body}\n"));
        let checked = Command::cargo_bin("chelis")
            .unwrap()
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .arg("check")
            .arg(&path)
            .assert()
            .failure()
            .get_output()
            .stdout
            .clone();
        let checked: serde_json::Value = serde_json::from_slice(&checked).unwrap();
        assert_eq!(checked["errors"][0]["kind"], kind, "{checked}");
        let out = dir.path().join("c");
        Command::cargo_bin("chelis")
            .unwrap()
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .arg("build")
            .arg(&path)
            .args(["--target", "c", "--output"])
            .arg(&out)
            .assert()
            .failure();
        assert!(!out.join("invalid.c").exists());
    }
}
