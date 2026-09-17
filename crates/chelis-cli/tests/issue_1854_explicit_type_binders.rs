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

#[test]
fn check_rejects_unknown_surf_type_names_and_the_desugared_deep_agrees() {
    let dir = tempdir().expect("tempdir");
    for (name, nearest) in REJECTED_DTYPES {
        let source = format!("def ident(x: {name}) = 0.0f32\n");
        fs::write(dir.path().join("case.ch"), source).expect("write Surf fixture");

        let checked = text(&run(dir.path(), &["check", "case.ch"]));
        assert!(
            !checked.contains("\"score\": 1,")
                && checked.contains(&format!("`{name}`"))
                && nearest.is_none_or(|nearest| checked.contains(&format!("`{nearest}`"))),
            "Surf check must reject `{name}` with any required suggestion: {checked}"
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
        let deep_checked = text(&run(dir.path(), &["check", "case.dp"]));
        assert!(
            !deep_checked.contains("\"score\": 1,")
                && deep_checked.contains(&format!("`{name}`"))
                && nearest.is_none_or(|nearest| deep_checked.contains(&format!("`{nearest}`"))),
            "Deep check must agree for `{name}`: {deep_checked}"
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
        "sig constant[a]: () -> i32\ndef constant() = 1i32\n",
        "def ident[n, p](x: tensor[n, p]) -> tensor[n, p] = x\n",
        "sig ident[n, p]: tensor[n, p] -> tensor[n, p]\n\
         def ident(x: tensor[n, p]) -> tensor[n, p] = x\n",
        "sig ident[r]: tensor[..r, f32] -> tensor[..r, f32]\ndef ident(x) = x\n",
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
fn check_rejects_a_declared_but_unused_bounded_binder() {
    let dir = tempdir().expect("tempdir");
    let source = "sig constant[a: Numeric]: () -> i32\ndef constant() = 1i32\n";
    fs::write(dir.path().join("bounded-unused.ch"), source).expect("write control");
    let checked = text(&run(dir.path(), &["check", "bounded-unused.ch"]));
    assert!(
        !checked.contains("\"score\": 1,")
            && checked.contains("`a`")
            && checked.contains("does not occur"),
        "a bounded binder cannot be inert metadata: {checked}"
    );
}
