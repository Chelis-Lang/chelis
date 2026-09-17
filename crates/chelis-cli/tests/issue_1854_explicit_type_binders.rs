//! chelis#1854 acceptance on the machine-facing CLI surfaces.

use assert_cmd::Command;
use std::{fs, path::Path};
use tempfile::tempdir;

const REJECTED_DTYPES: [(&str, Option<&str>); 7] = [
    ("float32", Some("f32")),
    ("fp32", Some("f32")),
    ("int33", Some("i32")),
    ("double", Some("f64")),
    ("half", Some("f16")),
    ("f8e4m3", None),
    ("f8e5m2", None),
];

fn run(root: &Path, args: &[&str]) -> std::process::Output {
    Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(root)
        .args(args)
        .output()
        .expect("run chelis")
}

fn text(output: &std::process::Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

fn check_json(root: &Path, path: &str) -> serde_json::Value {
    let output = run(root, &["check", path]);
    serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "`chelis check {path}` must emit JSON: {error}\n{}",
            text(&output)
        )
    })
}

fn error_messages(report: &serde_json::Value) -> Vec<&str> {
    report["errors"]
        .as_array()
        .expect("check report errors must be an array")
        .iter()
        .filter_map(|error| error["message"].as_str())
        .collect()
}

#[test]
fn check_rejects_unknown_surf_type_names_and_the_desugared_deep_agrees() {
    let dir = tempdir().expect("tempdir");
    for (name, nearest) in REJECTED_DTYPES {
        let source = format!("def ident(x: {name}) -> {name} = x\n");
        fs::write(dir.path().join("case.ch"), source).expect("write Surf fixture");

        let checked = check_json(dir.path(), "case.ch");
        let messages = error_messages(&checked);
        assert!(
            messages.len() == 1
                && messages[0].contains(&format!("`{name}`"))
                && nearest.is_none_or(|nearest| {
                    checked.to_string().contains(&format!("`{nearest}`"))
                }),
            "Surf check must give `{name}` one owner and any required suggestion: {checked}"
        );

        let deep = run(dir.path(), &["deep", "case.ch"]);
        assert!(
            deep.status.success(),
            "`chelis deep` must succeed: {deep:?}"
        );
        let printed = String::from_utf8_lossy(&deep.stdout).to_string();
        assert!(
            printed.contains(&format!("(t-prim {{}} {name})"))
                && !printed.contains(&format!("(t-var {{}} {name})")),
            "Surf must preserve unknown dtype intent in Deep: {printed}"
        );
        fs::write(dir.path().join("case.dp"), printed).expect("write Deep fixture");
        let deep_checked = check_json(dir.path(), "case.dp");
        let deep_messages = error_messages(&deep_checked);
        assert!(
            deep_messages.len() == 1
                && deep_messages[0].contains(&format!("`{name}`"))
                && nearest.is_none_or(|nearest| {
                    deep_checked.to_string().contains(&format!("`{nearest}`"))
                }),
            "Deep check must give `{name}` the same single owner: {deep_checked}"
        );
    }
}

#[test]
fn check_keeps_distinct_unknown_dtype_names_separately_owned() {
    let dir = tempdir().expect("tempdir");
    fs::write(
        dir.path().join("distinct.ch"),
        "def convert(x: float32) -> fp32 = x\n",
    )
    .expect("write Surf fixture");
    let report = check_json(dir.path(), "distinct.ch");
    let messages = error_messages(&report);
    assert_eq!(
        messages.len(),
        2,
        "distinct names need distinct owners: {report}"
    );
    for name in ["float32", "fp32"] {
        assert_eq!(
            messages
                .iter()
                .filter(|message| message.contains(&format!("`{name}`")))
                .count(),
            1,
            "`{name}` must report exactly once: {report}"
        );
    }
}

#[test]
fn cli_rejects_forbidden_dtype_names_at_surf_and_deep_binder_lists() {
    let dir = tempdir().expect("tempdir");
    for (index, name, surf_type, deep_type) in [
        (
            0,
            "f32",
            "tensor[f32, i32]",
            "(t-tensor {} (d-var {} f32) (t-prim {} i32))",
        ),
        (
            1,
            "f8e4m3",
            "tensor[f8e4m3, i32]",
            "(t-tensor {} (d-var {} f8e4m3) (t-prim {} i32))",
        ),
        (
            2,
            "i32",
            "tensor[..i32, f32]",
            "(t-tensor {} (d-rank {} i32) (t-prim {} f32))",
        ),
        (
            3,
            "f8e5m2",
            "tensor[..f8e5m2, f32]",
            "(t-tensor {} (d-rank {} f8e5m2) (t-prim {} f32))",
        ),
        (4, "bool", "i32", "(t-prim {} i32)"),
        (5, "complex64", "i32", "(t-prim {} i32)"),
    ] {
        let surf_path = format!("forbidden-{index}.ch");
        fs::write(
            dir.path().join(&surf_path),
            format!("def forbidden[{name}](x: {surf_type}) -> i32 = 0i32\n"),
        )
        .expect("write Surf fixture");
        let surf_report = check_json(dir.path(), &surf_path);
        let surf_messages = error_messages(&surf_report);
        assert!(
            surf_messages.len() == 1
                && surf_messages[0].contains(&format!("`{name}`"))
                && surf_messages[0].contains("cannot be a declaration binder"),
            "Surf binder list must own `{name}` rejection: {surf_report}"
        );

        let deep_path = format!("forbidden-{index}.dp");
        fs::write(
            dir.path().join(&deep_path),
            format!(
                "(defsig {{}} forbidden ({name}) (t-fn {{}} {deep_type} (t-prim {{}} i32)))\n\
                 (def {{}} forbidden\n\
                   (fn {{}} (params {{}} x) (lit {{type: (t-prim {{}} i32)}} 0)))\n"
            ),
        )
        .expect("write Deep fixture");
        let deep_report = check_json(dir.path(), &deep_path);
        let deep_messages = error_messages(&deep_report);
        assert!(
            deep_messages.len() == 1
                && deep_messages[0].contains(&format!("`{name}`"))
                && deep_messages[0].contains("cannot be a `defsig` binder"),
            "Deep binder list must own `{name}` rejection: {deep_report}"
        );
    }
}

#[test]
fn check_rejects_undeclared_surf_dimension_and_rank_variables() {
    let dir = tempdir().expect("tempdir");
    for (index, source, name, needle) in [
        (
            0,
            "sig shaped: tensor[n, f32] -> f32\ndef shaped(x) = 0.0f32\n",
            "n",
            "undeclared dimension variable",
        ),
        (
            1,
            "sig shaped: tensor[..r, f32] -> f32\ndef shaped(x) = 0.0f32\n",
            "r",
            "undeclared rank variable",
        ),
    ] {
        let path = format!("undeclared-{index}.ch");
        fs::write(dir.path().join(&path), source).expect("write Surf fixture");
        let checked = text(&run(dir.path(), &["check", &path]));
        assert!(
            !checked.contains("\"score\": 1,")
                && checked.contains(needle)
                && checked.contains(&format!("`{name}`")),
            "Surf check must reject undeclared `{name}`: {checked}"
        );
    }
}

#[test]
fn declared_surf_type_binders_and_active_primitives_score_one() {
    let dir = tempdir().expect("tempdir");
    for (index, source) in [
        "def ident[a](x: a) -> a = x\n",
        "sig ident[a]: a -> a\ndef ident(x) = x\n",
        "def constant[a]() -> i32 = 1i32\n",
        "sig constant[a]: i32 -> i32\ndef constant(x) = x\n",
        "def ident[n, p](x: tensor[n, p]) -> tensor[n, p] = x\n",
        "sig ident[n, p]: tensor[n, p] -> tensor[n, p]\n\
         def ident(x: tensor[n, p]) -> tensor[n, p] = x\n",
        "sig ident[r]: tensor[..r, f32] -> tensor[..r, f32]\ndef ident(x) = x\n",
        "def ident[float32](x: tensor[float32, f32]) -> tensor[float32, f32] = x\n",
        "def ident[float32](x: tensor[..float32, f32]) -> tensor[..float32, f32] = x\n",
        "def constant[float32]() -> i32 = 1i32\n",
        "def ident(x: f32) -> f32 = x\n",
    ]
    .into_iter()
    .enumerate()
    {
        let path = format!("control-{index}.ch");
        fs::write(dir.path().join(&path), source).expect("write control");
        let checked = text(&run(dir.path(), &["check", &path]));
        assert!(
            checked.contains("\"score\": 1,"),
            "declared binder/control must score 1: {source}\n{checked}"
        );
    }
}

#[test]
fn deep_defsig_wrong_arity_reaches_the_cli_contract_owner() {
    let dir = tempdir().expect("tempdir");
    for (index, source, actual) in [
        (0, "(defsig {} ident)\n", 1),
        (1, "(defsig {} ident (a) (t-var {} a) (t-prim {} f32))\n", 4),
    ] {
        let path = format!("arity-{index}.dp");
        fs::write(dir.path().join(&path), source).expect("write Deep fixture");
        for command in [["check", path.as_str()], ["surf", path.as_str()]] {
            let output = run(dir.path(), &command);
            let rendered = text(&output);
            assert!(
                !output.status.success()
                    && rendered.contains(&format!(
                        "wrong child count for `defsig`: expected Range(2, 3), got {actual}"
                    ))
                    && !rendered.contains("undecodable type head"),
                "`chelis {}` must report the declared arity owner: {rendered}",
                command[0]
            );
        }
    }
}

#[test]
fn check_rejects_a_declared_but_unused_bounded_binder() {
    let dir = tempdir().expect("tempdir");
    let source = "sig constant[a: Numeric]: i32 -> i32\ndef constant(x) = x\n";
    fs::write(dir.path().join("bounded-unused.ch"), source).expect("write control");
    let checked = text(&run(dir.path(), &["check", "bounded-unused.ch"]));
    assert!(
        !checked.contains("\"score\": 1,")
            && checked.contains("`a`")
            && checked.contains("does not occur"),
        "a bounded binder cannot be inert metadata: {checked}"
    );
}
