//! chelis#3114 and chelis#3080: a bracket literal is a `List` unless a tensor
//! is declared for it, in both execution lanes and through `chelis surf`.
//!
//! `spec/02-surf-syntax.md` §P10b and `spec/04-type-system.md` §5.6: an
//! unannotated bracket literal is a `List` whatever its element spelling; a
//! tensor literal is `to_tensor([...])` or a bare literal under its own
//! declared tensor type; and the bracket-literal argument of a `to_tensor`
//! call in an adopting position takes that position's dtype.
#[path = "common/mod.rs"]
mod common;
use assert_cmd::Command;
use serde_json::Value;
use std::path::Path;
use tempfile::tempdir;

fn chelis(directory: &Path) -> Command {
    let mut command = Command::cargo_bin("chelis").expect("chelis binary");
    command
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(directory);
    command
}

fn eval_json(directory: &Path, file: &str) -> Value {
    let output = chelis(directory)
        .args(["eval", "--json", "--file", file])
        .output()
        .expect("chelis eval runs");
    assert!(
        output.status.success(),
        "{file} must evaluate: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("eval JSON")
}

fn eval_text(directory: &Path, file: &str) -> String {
    let output = chelis(directory)
        .args(["eval", "--file", file])
        .output()
        .expect("chelis eval runs");
    assert!(output.status.success(), "{output:?}");
    String::from_utf8(output.stdout).expect("utf-8 eval output")
}

fn root<'a>(report: &'a Value, name: &str) -> &'a Value {
    report["roots"]
        .as_array()
        .expect("roots")
        .iter()
        .find(|root| root["name"] == name)
        .unwrap_or_else(|| panic!("no root `{name}` in {report}"))
}

/// `chelis deep`, then `chelis surf`, into `resugared.ch` beside `file`.
fn resugar(directory: &Path, file: &str) -> String {
    let deep = chelis(directory)
        .args(["deep", file])
        .output()
        .expect("chelis deep runs");
    assert!(deep.status.success(), "{deep:?}");
    std::fs::write(directory.join("program.dp"), &deep.stdout).expect("write Deep");
    let surf = chelis(directory)
        .args(["surf", "program.dp"])
        .output()
        .expect("chelis surf runs");
    assert!(surf.status.success(), "{surf:?}");
    let text = String::from_utf8(surf.stdout).expect("utf-8 Surf");
    std::fs::write(directory.join("resugared.ch"), &text).expect("write Surf");
    text
}

fn f64_bits(value: &Value) -> Vec<String> {
    assert_eq!(value["type"], "tensor", "{value}");
    let data = &value["value"]["data"];
    assert_eq!(data["dtype"], "f64", "{value}");
    data["bits"]
        .as_array()
        .expect("bits")
        .iter()
        .map(|bits| bits.as_str().expect("hex bits").to_string())
        .collect()
}

#[test]
fn unannotated_bracket_literals_are_lists_in_every_lane_and_after_resugaring() {
    for (index, (literal, items)) in [
        ("[1.0, 2.0, 3.0]", 3),
        ("[1i64, 2i64]", 2),
        // The chelis#3114 reproduction: `neg(2.0f64)` resugars as `-2.0f64`.
        ("[1.0f64, neg(2.0f64)]", 2),
        ("[(1.0 : f64), 2.0f64]", 2),
        ("[half(), 2.0f64]", 2),
        // A List of Lists may be ragged; only a tensor literal is rejected.
        ("[[1i64], [2i64, 3i64]]", 2),
    ]
    .into_iter()
    .enumerate()
    {
        let directory = tempdir().expect("tempdir");
        let name = format!("lists{index}");
        let file = format!("{name}.ch");
        let source = format!(
            "macro half() = 1.0f64\n\
             values = {literal}\n\
             size = len(values)\n\
             def local() -> i64 = {{\n  inner = {literal}\n  len(inner)\n}}\n\
             local_size = local()\n"
        );
        common::write_file(&directory.path().join(&file), &source);
        let report = eval_json(directory.path(), &file);
        assert_eq!(
            root(&report, "values")["value"]["type"],
            "list",
            "{literal}"
        );
        for size in ["size", "local_size"] {
            assert_eq!(
                root(&report, size)["value"]["value"]["value"],
                items,
                "{literal}: {report}"
            );
        }
        let evaluated = eval_text(directory.path(), &file);
        assert_eq!(
            common::build_and_run(&source, &name),
            evaluated,
            "{literal}"
        );

        let resugared = resugar(directory.path(), &file);
        assert_eq!(
            eval_json(directory.path(), "resugared.ch")["roots"],
            report["roots"],
            "{literal} changed meaning through chelis surf:\n{resugared}"
        );
    }
}

#[test]
fn a_tensor_literal_in_an_adopting_position_keeps_its_bits_through_resugaring() {
    const EXACT: [&str; 2] = ["3ff199999999999a", "400199999999999a"];
    for (index, declarations) in [
        // chelis#3080: the cast silently rounded through f32.
        "values = cast(to_tensor([1.1, 2.2]), f64)\n",
        "def ident(x: tensor[2, f64]) -> tensor[2, f64] = x\nvalues = ident(to_tensor([1.1, 2.2]))\n",
        "def make() -> tensor[2, f64] = [1.1, 2.2]\nvalues = make()\n",
        "sig make: i32 -> tensor[2, f64]\ndef make(n) = [1.1, 2.2]\nvalues = make(1)\n",
        "values: tensor[2, f64] = [1.1, 2.2]\n",
        "values: tensor[2, f64] = to_tensor([1.1, 2.2])\n",
        "def make() -> tensor[2, f64] = {\n  inner: tensor[2, f64] = [1.1, 2.2]\n  inner\n}\nvalues = make()\n",
    ]
    .into_iter()
    .enumerate()
    {
        let directory = tempdir().expect("tempdir");
        let name = format!("adopting{index}");
        let file = format!("{name}.ch");
        common::write_file(&directory.path().join(&file), declarations);
        let report = eval_json(directory.path(), &file);
        assert_eq!(
            f64_bits(&root(&report, "values")["value"]),
            EXACT,
            "{declarations}"
        );
        let evaluated = eval_text(directory.path(), &file);
        assert_eq!(
            common::build_and_run(declarations, &name),
            evaluated,
            "{declarations}"
        );

        let resugared = resugar(directory.path(), &file);
        let after = eval_json(directory.path(), "resugared.ch");
        assert_eq!(
            f64_bits(&root(&after, "values")["value"]),
            EXACT,
            "{declarations} lost its dtype through chelis surf:\n{resugared}"
        );
    }
}

#[test]
fn a_cast_or_tensor_parameter_never_converts_a_bare_bracket_literal() {
    for (index, declarations) in [
        "values = cast([1.1, 2.2], f64)\n",
        "def ident(x: tensor[2, f64]) -> tensor[2, f64] = x\nvalues = ident([1.1, 2.2])\n",
    ]
    .into_iter()
    .enumerate()
    {
        let directory = tempdir().expect("tempdir");
        let file = format!("bare{index}.ch");
        common::write_file(&directory.path().join(&file), declarations);
        let output = chelis(directory.path())
            .args(["check", &file])
            .output()
            .expect("chelis check runs");
        let report: Value = serde_json::from_slice(&output.stdout).expect("check JSON");
        assert!(!output.status.success(), "{declarations}: {report}");
        let errors = report["errors"].as_array().expect("errors");
        assert!(
            errors.iter().any(|error| error["message"]
                .as_str()
                .is_some_and(|m| m.contains("List"))),
            "{declarations} must be rejected as a List: {report}"
        );
    }
}

const EXACT_F64: [&str; 2] = ["3ff199999999999a", "400199999999999a"];
const F32_WIDENED: [&str; 2] = ["3ff19999a0000000", "40019999a0000000"];

/// Round 1 of chelis#3145's review: a pipe spelling adopts exactly like the
/// nested call, a lexically rebound callee adopts nothing, and both lanes
/// agree.
#[test]
fn pipe_spellings_and_rebound_callees_keep_the_nested_call_values() {
    for (index, (source, expected)) in [
        (
            "def take_it(x: tensor[2, f64]) -> tensor[2, f64] = x\n\
             c1 = take_it(to_tensor([1.1, 2.2]))\n\
             c2 = cast(to_tensor([1.1, 2.2]), f64)\n\
             p1 = to_tensor([1.1, 2.2]) |> take_it\n\
             p2 = to_tensor([1.1, 2.2]) |> cast(f64)\n\
             p3 = [1.1, 2.2] |> to_tensor |> cast(f64)\n\
             p4 = cast([1.1, 2.2] |> to_tensor, f64)\n",
            &[
                ("c1", EXACT_F64),
                ("c2", EXACT_F64),
                ("p1", EXACT_F64),
                ("p2", EXACT_F64),
                ("p3", EXACT_F64),
                ("p4", EXACT_F64),
            ][..],
        ),
        (
            "def f(x: tensor[2, f64]) -> tensor[2, f64] = x\n\
             def g() -> tensor[2, f64] = {\n  f = fn (y) -> cast(y, f64)\n  f(to_tensor([1.1, 2.2]))\n}\n\
             r = g()\n",
            &[("r", F32_WIDENED)][..],
        ),
    ]
    .into_iter()
    .enumerate()
    {
        let directory = tempdir().expect("tempdir");
        let name = format!("pipes{index}");
        let file = format!("{name}.ch");
        common::write_file(&directory.path().join(&file), source);
        let report = eval_json(directory.path(), &file);
        for (root_name, bits) in expected {
            assert_eq!(
                f64_bits(&root(&report, root_name)["value"]),
                *bits,
                "{root_name} in\n{source}"
            );
        }
        let evaluated = eval_text(directory.path(), &file);
        assert_eq!(common::build_and_run(source, &name), evaluated, "{source}");
        let resugared = resugar(directory.path(), &file);
        assert_eq!(
            eval_json(directory.path(), "resugared.ch")["roots"],
            report["roots"],
            "{source} changed meaning through chelis surf:\n{resugared}"
        );
    }
}

#[test]
fn scalar_pipe_cast_adopts_like_the_nested_call() {
    let directory = tempdir().expect("tempdir");
    common::write_file(
        &directory.path().join("scalar.ch"),
        "s1 = cast(1.1, f64)\ns2 = 1.1 |> cast(f64)\n",
    );
    let report = eval_json(directory.path(), "scalar.ch");
    for name in ["s1", "s2"] {
        let value = &root(&report, name)["value"];
        assert_eq!(value["value"]["dtype"], "f64", "{name}: {value}");
        assert_eq!(value["value"]["bits"], EXACT_F64[0], "{name}: {value}");
    }
}

#[test]
fn a_parameter_shadowing_the_callee_keeps_its_own_type() {
    let directory = tempdir().expect("tempdir");
    let source = "def f(x: tensor[2, f64]) -> tensor[2, f64] = x\n\
                  def apply(f: tensor[2, f32] -> tensor[2, f32]) -> tensor[2, f32] = f(to_tensor([1.1, 2.2]))\n\
                  def ident(x: tensor[2, f32]) -> tensor[2, f32] = x\n\
                  r = apply(ident)\n";
    common::write_file(&directory.path().join("param.ch"), source);
    let report = eval_json(directory.path(), "param.ch");
    let value = &root(&report, "r")["value"];
    assert_eq!(value["type"], "tensor", "{value}");
    assert_eq!(value["value"]["data"]["dtype"], "f32", "{value}");
}

#[test]
fn a_tensor_literal_cast_to_a_dtype_binder_is_refused_not_rounded() {
    let directory = tempdir().expect("tempdir");
    let source = "def widen[p: Float](x: tensor[2, p]) -> tensor[2, p] = add(x, cast(to_tensor([1.1, 2.2]), p))\n\
                  def scalar_b[p: Float](x: p) -> p = add(x, cast(1.1, p))\n\
                  r1 = widen(to_tensor([0.0f64, 0.0f64]))\n\
                  r3 = scalar_b(0.0f64)\n";
    common::write_file(&directory.path().join("binder.ch"), source);
    let output = chelis(directory.path())
        .args(["check", "binder.ch"])
        .output()
        .expect("chelis check runs");
    let report: Value = serde_json::from_slice(&output.stdout).expect("check JSON");
    assert!(!output.status.success(), "{report}");
    assert!(
        report["errors"]
            .as_array()
            .expect("errors")
            .iter()
            .any(|error| {
                error["message"]
                    .as_str()
                    .is_some_and(|m| m.contains("to_tensor") && m.contains("cast(1.1, p)"))
            }),
        "a binder cast of a tensor literal must be refused with the per-element spelling: {report}"
    );
}

/// Round 1 verification of chelis#3145: a cast operand that did not adopt
/// (a macro's `neg(lit)`, an ascribed literal, a macro literal) keeps its value
/// through `chelis surf`, in the call and pipe spellings, at a primitive and at
/// a dtype binder.
#[test]
fn cast_operands_keep_their_values_through_resugaring() {
    for (index, source) in [
        "macro mh() = neg(1.1)\n\
         a = cast(mh(), f64)\n\
         b = mh() |> cast(f64)\n\
         c = (neg(1.1) : f32) |> cast(f64)\n\
         d = (1.1 : f32) |> cast(f64)\n\
         e = cast((1.1 : f32), f64)\n",
        "macro half() = 1.1\n\
         def b[p: Float](x: p) -> p = add(x, half() |> cast(p))\n\
         def c[p: Float](x: p) -> p = add(x, cast(half(), p))\n\
         rb = b(0.0f64)\n\
         rc = c(0.0f64)\n",
        "def b[p: Float](x: p) -> p = add(x, (1.1 : f32) |> cast(p))\n\
         def c[p: Float](x: p) -> p = add(x, cast((1.1 : f32), p))\n\
         rb = b(0.0f64)\n\
         rc = c(0.0f64)\n",
    ]
    .into_iter()
    .enumerate()
    {
        let directory = tempdir().expect("tempdir");
        let file = format!("operands{index}.ch");
        common::write_file(&directory.path().join(&file), source);
        let report = eval_json(directory.path(), &file);
        let resugared = resugar(directory.path(), &file);
        assert_eq!(
            eval_json(directory.path(), "resugared.ch")["roots"],
            report["roots"],
            "{source} changed value through chelis surf:\n{resugared}"
        );
    }
}
