//! [05-OP-35], [04-INF-2/3], [04-DTYPE-1]: concrete recursive
//! specializations resolve cast targets without adopting the operand dtype.

mod common;

use std::{
    fs,
    path::Path,
    process::{Command, Output},
};
use tempfile::tempdir;

fn run(root: &Path, reef: Option<&Path>, args: &[&str]) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_chelis"));
    command
        .current_dir(root)
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(args);
    if let Some(reef) = reef {
        command.env("CHELIS_REEF_HOME", reef);
    }
    command.output().unwrap()
}

fn success(output: &Output) {
    assert!(
        output.status.success(),
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn parity(root: &Path, reef: Option<&Path>, file: &str, expected: &str) {
    let checked = run(root, reef, &["check", file]);
    success(&checked);
    let report: serde_json::Value = serde_json::from_slice(&checked.stdout).unwrap();
    assert_eq!(report["score"], 1.0);
    assert_eq!(report["errors"], serde_json::json!([]));
    let evaluated = run(root, reef, &["eval", "--file", file]);
    success(&evaluated);
    assert_eq!(String::from_utf8_lossy(&evaluated.stdout), expected);
    let built = run(
        root,
        reef,
        &["build", file, "--target", "c", "--output", "out"],
    );
    success(&built);
    let name = Path::new(file).file_stem().unwrap().to_str().unwrap();
    assert!(common::link_generated(&root.join("out"), &format!("{name}.c"), name).success());
    let native = Command::new(root.join("out").join(name)).output().unwrap();
    success(&native);
    assert_eq!(native.stdout, evaluated.stdout);
}

#[test]
fn recursive_casts_use_each_selected_integer_and_float_dtype() {
    let dir = tempdir().unwrap();
    fs::write(
        dir.path().join("probe.ch"),
        include_str!("../../../examples/recursive_cast_targets.ch"),
    )
    .unwrap();
    parity(
        dir.path(),
        None,
        "probe.ch",
        "i32 = 4\ni64 = 9007199254740996\nf32 = 6.5\nf64 = 6.5\n",
    );
}

#[test]
fn mutually_recursive_casts_keep_target_dtype_across_instantiations() {
    let dir = tempdir().unwrap();
    fs::write(dir.path().join("probe.ch"), r#"module MutualCasts
def ping[p: Float](x: p, n: int64) -> p = if eq(n, 0i64) then x else pong(add(x, cast(n, p)), sub(n, 1i64))
def pong[p: Float](x: p, n: int64) -> p = if eq(n, 0i64) then x else ping(add(x, cast(n, p)), sub(n, 1i64))
f32 = ping(0.5f32, 2i64)
f64 = ping(0.5f64, 2i64)
"#).unwrap();
    parity(dir.path(), None, "probe.ch", "f32 = 3.5\nf64 = 3.5\n");
}

#[test]
fn published_arange_and_linspace_match_eval_at_each_requested_width() {
    let (_dir, reef, app) = common::make_app("recursive-cast-targets");
    fs::write(
        app.join("src/main.ch"),
        r#"module Demo.Main
import Std.Tensor.Construct (arange, linspace)
integer32 = arange(cast(1, int32), cast(4, int32))
integer64 = arange(9007199254740993i64, 9007199254740996i64)
float32 = linspace(cast(0.0, f32), cast(1.0, f32), cast(3, int64))
float64 = linspace(0.0f64, 1.0f64, 3i64)
"#,
    )
    .unwrap();
    parity(
        &app,
        Some(&reef),
        "src/main.ch",
        "integer32 = tensor(shape=[3], data=[1, 2, 3])\ninteger64 = tensor(shape=[3], data=[9007199254740993, 9007199254740994, 9007199254740995])\nfloat32 = tensor(shape=[3], data=[0.0, 0.5, 1.0])\nfloat64 = tensor(shape=[3], data=[0.0, 0.5, 1.0])\n",
    );
}

#[test]
fn invalid_cast_targets_and_polymorphic_recursion_fail_for_the_right_reason() {
    for (source, extension, diagnostic) in [
        (
            "out = cast(1, imaginary_dtype)",
            "ch",
            "not a recognized primitive type",
        ),
        (
            "(def {} out (cast {} 1 (t-tuple {} (t-prim {} int32) (t-prim {} int32))))",
            "dp",
            "not a recognized primitive type",
        ),
        (
            "def f[p: Float](x: p) -> p = cast(1, p)\nout = f(1i32)",
            "ch",
            "bounded by dtype family `Float`",
        ),
        (
            "type Box[a] =\n  | Full {value: a}\ndef f[a](x: a, n: int32) -> int32 = if n <= 0 then 0 else f(Full {value: x}, n - 1)\nout = f(1i32, 2i32)",
            "ch",
            "[04-INF-3]",
        ),
    ] {
        let dir = tempdir().unwrap();
        let file = format!("probe.{extension}");
        fs::write(dir.path().join(&file), source).unwrap();
        for args in [
            &["check", &file][..],
            &["eval", "--file", &file],
            &["build", &file, "--target", "c", "--output", "out"],
        ] {
            let result = run(dir.path(), None, args);
            assert!(!result.status.success(), "{source}: {result:?}");
            let errors = format!(
                "{}{}",
                String::from_utf8_lossy(&result.stdout),
                String::from_utf8_lossy(&result.stderr)
            );
            assert!(errors.contains(diagnostic), "{source}: {errors}");
            assert!(!dir.path().join("out/probe.c").exists());
        }
    }
}

#[test]
fn unactualized_generic_target_is_not_replaced_by_the_operand_dtype() {
    let dir = tempdir().unwrap();
    fs::write(
        dir.path().join("probe.ch"),
        "def f[p: Float]() -> p = cast(1, p)\nout = f()\n",
    )
    .unwrap();
    success(&run(dir.path(), None, &["check", "probe.ch"]));
    for args in [
        &["eval", "--file", "probe.ch"][..],
        &["build", "probe.ch", "--target", "c", "--output", "out"],
    ] {
        let result = run(dir.path(), None, args);
        assert!(!result.status.success(), "{result:?}");
        let errors = String::from_utf8_lossy(&result.stderr);
        assert!(
            errors.contains("cast") && errors.contains("primitive"),
            "{errors}"
        );
        assert!(!dir.path().join("out/probe.c").exists());
    }
}
