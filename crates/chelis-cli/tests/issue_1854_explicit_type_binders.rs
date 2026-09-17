//! chelis#1854 acceptance on the machine-facing CLI surfaces.

use assert_cmd::Command;
use std::{fs, path::Path};
use tempfile::tempdir;

const UNKNOWN_DTYPES: [(&str, Option<&str>); 5] = [
    ("float32", Some("f32")),
    ("fp32", Some("f32")),
    ("int33", Some("i32")),
    ("double", Some("f64")),
    ("half", Some("f16")),
];
const RESERVED_DTYPES: [&str; 2] = ["f8e4m3", "f8e5m2"];

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
    for (name, nearest) in UNKNOWN_DTYPES {
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
    for name in RESERVED_DTYPES {
        let source = format!("def ident(x: {name}) -> {name} = x\n");
        fs::write(dir.path().join("case.ch"), source).expect("write Surf fixture");

        let checked = check_json(dir.path(), "case.ch");
        let messages = error_messages(&checked);
        assert!(
            messages.len() == 2
                && messages
                    .iter()
                    .all(|message| message.contains(&format!("`{name}`"))),
            "Surf check must give every authored `{name}` site an owner: {checked}"
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
            "Surf must preserve reserved dtype intent in Deep: {printed}"
        );
        fs::write(dir.path().join("case.dp"), printed).expect("write Deep fixture");
        let deep_checked = check_json(dir.path(), "case.dp");
        let deep_messages = error_messages(&deep_checked);
        assert!(
            deep_messages.len() == 2
                && deep_messages
                    .iter()
                    .all(|message| message.contains(&format!("`{name}`"))),
            "Deep check must retain every authored `{name}` owner: {deep_checked}"
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

fn assert_cli_unknown_spelling_order(root: &Path, path: &str, expected: &[&str]) {
    let report = check_json(root, path);
    let messages = error_messages(&report);
    let actual = messages
        .iter()
        .filter_map(|message| {
            expected
                .iter()
                .copied()
                .find(|name| message.contains(&format!("`{name}`")))
        })
        .collect::<Vec<_>>();
    assert_eq!(actual, expected, "wrong CLI owner order: {report}");
    assert_eq!(
        messages.len(),
        expected.len(),
        "CLI emitted an extra diagnostic: {report}"
    );
}

fn assert_cli_unknown_spelling_order_amid_other_errors(root: &Path, path: &str, expected: &[&str]) {
    let report = check_json(root, path);
    let messages = error_messages(&report);
    let actual = messages
        .iter()
        .filter(|message| message.contains("unknown primitive type"))
        .filter_map(|message| {
            expected
                .iter()
                .copied()
                .find(|name| message.contains(&format!("`{name}`")))
        })
        .collect::<Vec<_>>();
    assert_eq!(
        actual, expected,
        "CLI unknown-name owners must remain isolated across modules: {report}"
    );
}

#[test]
fn cli_shares_unknown_primitive_ownership_across_signature_and_definition() {
    let dir = tempdir().expect("tempdir");
    for (index, source) in [
        "sig ident: float32 -> float32\n\
         def ident(x: float32) -> float32 = x\n",
        "def ident(x: float32) -> float32 = x\n\
         sig ident: float32 -> float32\n",
        "sig ident[a]: float32 -> float32\n\
         def ident(x: float32) -> float32 = x\n",
        "def ident(x: float32) -> float32 = x\n\
         sig ident[a]: float32 -> float32\n",
    ]
    .into_iter()
    .enumerate()
    {
        let path = format!("standalone-inline-{index}.ch");
        fs::write(dir.path().join(&path), source).expect("write Surf fixture");
        assert_cli_unknown_spelling_order(dir.path(), &path, &["float32"]);
    }

    for (index, source) in [
        "(defsig {} ident
           (t-fn {} (t-prim {} float32) (t-prim {} float32)))
         (def {} ident
           (fn {} (params {} (x {type: (t-prim {} float32)}))
             (var {} x)))\n",
        "(def {} ident
           (fn {} (params {} (x {type: (t-prim {} float32)}))
             (var {} x)))
         (defsig {} ident
           (t-fn {} (t-prim {} float32) (t-prim {} float32)))\n",
    ]
    .into_iter()
    .enumerate()
    {
        let path = format!("standalone-inline-{index}.dp");
        fs::write(dir.path().join(&path), source).expect("write Deep fixture");
        assert_cli_unknown_spelling_order(dir.path(), &path, &["float32"]);
    }
}

#[test]
fn cli_keeps_reserved_primitive_diagnostics_per_use_with_explicit_binders() {
    let dir = tempdir().expect("tempdir");
    for (index, name) in RESERVED_DTYPES.into_iter().enumerate() {
        let path = format!("reserved-explicit-binder-{index}.ch");
        fs::write(
            dir.path().join(&path),
            format!(
                "sig ident[a]: {name} -> {name}\n\
                 def ident(x: {name}) -> {name} = x\n"
            ),
        )
        .expect("write Surf fixture");
        let report = check_json(dir.path(), &path);
        let messages = error_messages(&report);
        assert_eq!(
            messages.len(),
            3,
            "reserved `{name}` must retain one diagnostic per authored use: {report}"
        );
        assert!(
            messages
                .iter()
                .all(|message| message.contains(&format!("`{name}`"))),
            "every reserved-use diagnostic must name `{name}`: {report}"
        );
    }
}

#[test]
fn cli_preserves_unknown_spelling_declaration_order_and_module_scope() {
    let dir = tempdir().expect("tempdir");
    for (index, source, expected) in [
        (
            0,
            "sig convert: float32 -> float32\n\
             def convert(x: fp32) -> fp32 = x\n",
            ["float32", "fp32"],
        ),
        (
            1,
            "sig first: fp32 -> fp32\n\
             def first(x: fp32) -> fp32 = x\n\
             sig second: float32 -> float32\n\
             def second(x: float32) -> float32 = x\n",
            ["fp32", "float32"],
        ),
        (
            2,
            "sig second: float32 -> float32\n\
             def second(x: float32) -> float32 = x\n\
             sig first: fp32 -> fp32\n\
             def first(x: fp32) -> fp32 = x\n",
            ["float32", "fp32"],
        ),
    ] {
        let path = format!("ownership-order-{index}.ch");
        fs::write(dir.path().join(&path), source).expect("write Surf fixture");
        assert_cli_unknown_spelling_order(dir.path(), &path, &expected);
    }
    fs::write(
        dir.path().join("ownership-multiplicity.ch"),
        "sig first: float32 -> float32\n\
         def first(x: float32) -> float32 = x\n\
         sig second: float32 -> float32\n\
         def second(x: float32) -> float32 = x\n",
    )
    .expect("write same-spelling declaration fixture");
    assert_cli_unknown_spelling_order(
        dir.path(),
        "ownership-multiplicity.ch",
        &["float32", "float32"],
    );

    let modules = "(module {} Left
      (defsig {} ident
        (t-fn {} (t-prim {} float32) (t-prim {} float32)))
      (def {} ident
        (fn {} (params {} (x {type: (t-prim {} float32)}))
          (var {} x))))
    (module {} Right
      (defsig {} ident
        (t-fn {} (t-prim {} float32) (t-prim {} float32)))
      (def {} ident
        (fn {} (params {} (x {type: (t-prim {} float32)}))
          (var {} x))))\n";
    fs::write(dir.path().join("ownership-modules.dp"), modules).expect("write Deep fixture");
    assert_cli_unknown_spelling_order_amid_other_errors(
        dir.path(),
        "ownership-modules.dp",
        &["float32", "float32"],
    );
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
fn body_tensor_precision_binder_scores_one_through_surf_and_canonical_deep() {
    let dir = tempdir().expect("tempdir");
    let source = "def f[n, p](a: tensor[n, p]) -> tensor[n, p] = {\n\
                    b: tensor[n, p] = a\n\
                    b\n\
                  }\n\
                  def main() -> tensor[2, f32] = f(to_tensor([1.0f32, 2.0f32]))\n";
    fs::write(dir.path().join("body-precision.ch"), source).expect("write Surf fixture");

    let deep = run(dir.path(), &["deep", "body-precision.ch"]);
    assert!(
        deep.status.success(),
        "`chelis deep` failed: {}",
        text(&deep)
    );
    let rendered = String::from_utf8_lossy(&deep.stdout);
    assert!(
        rendered.contains("(t-var {} p)") && !rendered.contains("(t-prim {} p)"),
        "body tensor precision must lower through the declaration binder: {rendered}"
    );
    fs::write(dir.path().join("body-precision.dp"), &deep.stdout)
        .expect("write canonical Deep fixture");

    for path in ["body-precision.ch", "body-precision.dp"] {
        let checked = check_json(dir.path(), path);
        assert_eq!(
            checked["score"].as_f64(),
            Some(1.0),
            "body tensor precision binder must score 1 through {path}: {checked}"
        );
        assert_eq!(
            error_messages(&checked),
            Vec::<&str>::new(),
            "body tensor precision binder must have no diagnostic through {path}: {checked}"
        );
    }
}

#[test]
fn cli_preserves_and_checks_body_only_surf_binders() {
    let dir = tempdir().expect("tempdir");
    for (index, source, binder, annotation) in [
        (0, "def maker[p]() = fn (x: p) -> x\n", "p", "(t-var {} p)"),
        (
            1,
            "def maker[n]() = fn (x: tensor[n, f32]) -> x\n",
            "n",
            "(d-var {} n)",
        ),
        (
            2,
            "def maker[r]() = fn (x: tensor[..r, f32]) -> x\n",
            "r",
            "(d-rank {} r)",
        ),
    ] {
        let path = format!("body-only-{index}.ch");
        fs::write(dir.path().join(&path), source).expect("write Surf fixture");

        let deep = run(dir.path(), &["deep", &path]);
        assert!(
            deep.status.success(),
            "`chelis deep` failed: {}",
            text(&deep)
        );
        let rendered = String::from_utf8_lossy(&deep.stdout);
        assert!(
            rendered.contains(&format!("(defsig {{}} maker ({binder}) "))
                && rendered.contains(annotation),
            "`chelis deep` dropped a body-only declaration binder: {rendered}"
        );

        let checked = check_json(dir.path(), &path);
        assert_eq!(
            checked["score"].as_f64(),
            Some(1.0),
            "`chelis check` rejected a body-only binder: {checked}"
        );
    }
}

#[test]
fn cli_rejects_concrete_pins_of_body_only_dimension_and_rank_binders() {
    let dir = tempdir().expect("tempdir");
    for (index, source, declaration, binder, role) in [
        (
            0,
            "def narrowed_dimension[n]() = {\n\
               value: tensor[n, f32] = to_tensor([1.0f32, 2.0f32])\n\
               value\n\
             }\n",
            "narrowed_dimension",
            "n",
            "dimension",
        ),
        (
            1,
            "def narrowed_rank[r]() = {\n\
               value: tensor[..r, f32] = to_tensor([1.0f32, 2.0f32])\n\
               value\n\
             }\n",
            "narrowed_rank",
            "r",
            "rank",
        ),
        (
            2,
            "def recursive_dimension[n](stop: bool) -> i32 = {\n\
               value: tensor[n, f32] = to_tensor([1.0f32, 2.0f32])\n\
               if stop then 0i32 else recursive_dimension(true)\n\
             }\n",
            "recursive_dimension",
            "n",
            "dimension",
        ),
        (
            3,
            "def recursive_rank[r](stop: bool) -> i32 = {\n\
               value: tensor[..r, f32] = to_tensor([1.0f32, 2.0f32])\n\
               if stop then 0i32 else recursive_rank(true)\n\
             }\n",
            "recursive_rank",
            "r",
            "rank",
        ),
    ] {
        let surf_path = format!("body-only-rigid-{index}.ch");
        fs::write(dir.path().join(&surf_path), source).expect("write Surf fixture");

        let deep = run(dir.path(), &["deep", &surf_path]);
        assert!(
            deep.status.success(),
            "`chelis deep` failed: {}",
            text(&deep)
        );
        let deep_path = format!("body-only-rigid-{index}.dp");
        fs::write(dir.path().join(&deep_path), &deep.stdout).expect("write canonical Deep fixture");

        for path in [&surf_path, &deep_path] {
            let checked = check_json(dir.path(), path);
            let messages = error_messages(&checked);
            assert!(
                checked["score"].as_f64() != Some(1.0)
                    && messages.iter().any(|message| {
                        message.contains(declaration)
                            && message.contains(&format!("`{binder}`"))
                            && message.contains("[04-INF-6]")
                    }),
                "`chelis check` must reject {role} binder `{binder}` pinned through {path}: \
                 {checked}"
            );
        }
    }
}

#[test]
fn check_rejects_non_function_binder_narrowing_and_preserves_polymorphic_values() {
    let dir = tempdir().expect("tempdir");
    for (index, source, declaration, binders) in [
        (
            0,
            "sig type_value[p]: p\n\
             type_value = 1.0f32\n",
            "type_value",
            &["p"][..],
        ),
        (
            1,
            "sig dimension_value[n]: tensor[n, f32]\n\
             dimension_value = to_tensor([1.0f32, 2.0f32])\n",
            "dimension_value",
            &["n"][..],
        ),
        (
            2,
            "sig rank_value[r]: tensor[..r, f32]\n\
             rank_value = to_tensor([1.0f32])\n",
            "rank_value",
            &["r"][..],
        ),
        (
            3,
            "sig matrix_value[rows, cols]: tensor[rows, cols, f32]\n\
             matrix_value = to_tensor([\n\
               [1.0f32, 2.0f32],\n\
               [3.0f32, 4.0f32]\n\
             ])\n",
            "matrix_value",
            &["rows", "cols"][..],
        ),
    ] {
        let surf_path = format!("non-function-rigid-{index}.ch");
        fs::write(dir.path().join(&surf_path), source).expect("write non-function Surf fixture");
        let deep = run(dir.path(), &["deep", &surf_path]);
        assert!(
            deep.status.success(),
            "`chelis deep` failed for `{declaration}`: {}",
            text(&deep)
        );
        let deep_path = format!("non-function-rigid-{index}.dp");
        fs::write(dir.path().join(&deep_path), &deep.stdout)
            .expect("write canonical non-function Deep fixture");
        for path in [&surf_path, &deep_path] {
            let checked = check_json(dir.path(), path);
            let messages = error_messages(&checked);
            assert!(
                checked["score"].as_f64() != Some(1.0),
                "`chelis check` must reject concrete narrowing in `{declaration}` through \
                 {path}: {checked}"
            );
            for binder in binders {
                assert!(
                    messages.iter().any(|message| {
                        message.contains(declaration)
                            && message.contains(&format!("`{binder}`"))
                            && message.contains("[04-INF-6]")
                    }),
                    "`chelis check` must report narrowed binder `{binder}` in `{declaration}` \
                     through {path}: {checked}"
                );
            }
        }
    }

    fs::write(
        dir.path().join("polymorphic-value.ch"),
        "sig empty[p]: List[p]\nempty = []\n",
    )
    .expect("write polymorphic Surf value fixture");
    let deep = run(dir.path(), &["deep", "polymorphic-value.ch"]);
    assert!(
        deep.status.success(),
        "`chelis deep` failed for the polymorphic value: {}",
        text(&deep)
    );
    fs::write(dir.path().join("polymorphic-value.dp"), &deep.stdout)
        .expect("write canonical polymorphic Deep fixture");
    for path in ["polymorphic-value.ch", "polymorphic-value.dp"] {
        let accepted = check_json(dir.path(), path);
        assert_eq!(
            accepted["score"].as_f64(),
            Some(1.0),
            "`chelis check` must preserve an unconstrained polymorphic value through {path}: \
             {accepted}"
        );
    }
}

#[test]
fn cli_rejects_body_only_dimension_collapsed_into_return_only_at_both_surfaces() {
    let dir = tempdir().expect("tempdir");
    for (index, source, declaration) in [
        (
            0,
            "def mixed_return_only[n, m](values: List[f32]) -> tensor[n, f32] = {\n\
               result: tensor[m, f32] = to_tensor(values)\n\
               result\n\
             }\n",
            "mixed_return_only",
        ),
        (
            1,
            "def recursive_mixed_return_only[n, m](\n\
               values: List[f32],\n\
               stop: bool\n\
             ) -> tensor[n, f32] = {\n\
               result: tensor[m, f32] = to_tensor(values)\n\
               if stop then result else recursive_mixed_return_only(values, true)\n\
             }\n",
            "recursive_mixed_return_only",
        ),
    ] {
        let surf_path = format!("mixed-return-only-{index}.ch");
        fs::write(dir.path().join(&surf_path), source).expect("write Surf fixture");

        let deep = run(dir.path(), &["deep", &surf_path]);
        assert!(
            deep.status.success(),
            "`chelis deep` failed: {}",
            text(&deep)
        );
        let deep_path = format!("mixed-return-only-{index}.dp");
        fs::write(dir.path().join(&deep_path), &deep.stdout).expect("write canonical Deep fixture");

        for path in [&surf_path, &deep_path] {
            let checked = check_json(dir.path(), path);
            let messages = error_messages(&checked);
            assert!(
                checked["score"].as_f64() != Some(1.0)
                    && messages.iter().any(|message| {
                        message.contains(declaration)
                            && message.contains("`n`")
                            && message.contains("`m`")
                            && message.contains("[04-INF-6]")
                    }),
                "`chelis check` must reject rigid body-only `m` collapsed into return-only `n` \
                 through {path}: {checked}"
            );
        }
    }
}

#[test]
fn cli_preserves_return_only_to_return_only_output_inference() {
    let dir = tempdir().expect("tempdir");
    let source = "def shared_return_only[n, m](values: List[f32]) \
                  -> (tensor[n, f32], tensor[m, f32]) = {\n\
                    result = to_tensor(values)\n\
                    (result, result)\n\
                  }\n";
    fs::write(dir.path().join("shared-return-only.ch"), source).expect("write Surf fixture");

    let deep = run(dir.path(), &["deep", "shared-return-only.ch"]);
    assert!(
        deep.status.success(),
        "`chelis deep` failed: {}",
        text(&deep)
    );
    fs::write(dir.path().join("shared-return-only.dp"), &deep.stdout)
        .expect("write canonical Deep fixture");

    for path in ["shared-return-only.ch", "shared-return-only.dp"] {
        let checked = check_json(dir.path(), path);
        assert_eq!(
            checked["score"].as_f64(),
            Some(1.0),
            "two return-only binders may share one body-inferred output through {path}: {checked}"
        );
    }
}

#[test]
fn cli_checks_canonical_deep_body_only_binders_and_rejects_undeclared_neighbors() {
    let dir = tempdir().expect("tempdir");
    for (index, binder, annotation) in [
        (0, "p", "(t-var {} p)"),
        (1, "n", "(t-tensor {} (d-var {} n) (t-prim {} f32))"),
        (2, "r", "(t-tensor {} (d-rank {} r) (t-prim {} f32))"),
        (3, "p", "(t-tensor {} (d-lit {} 2) (t-var {} p))"),
    ] {
        let declared_path = format!("deep-body-only-{index}.dp");
        fs::write(
            dir.path().join(&declared_path),
            format!(
                "(defsig {{}} maker ({binder}) (t-fn {{}} (t-var {{}} _)))
                 (def {{}}
                   maker
                   (fn {{}}
                     (params {{}})
                     (fn {{}}
                       (params {{}} (x {{type: {annotation}}}))
                       (var {{}} x))))\n"
            ),
        )
        .expect("write declared Deep fixture");
        let declared = check_json(dir.path(), &declared_path);
        assert_eq!(
            declared["score"].as_f64(),
            Some(1.0),
            "canonical Deep body-only `{binder}` must check: {declared}"
        );

        let undeclared_path = format!("deep-body-undeclared-{index}.dp");
        fs::write(
            dir.path().join(&undeclared_path),
            format!(
                "(defsig {{}} maker (t-fn {{}} (t-var {{}} _)))
                 (def {{}}
                   maker
                   (fn {{}}
                     (params {{}})
                     (fn {{}}
                       (params {{}} (x {{type: {annotation}}}))
                       (var {{}} x))))\n"
            ),
        )
        .expect("write undeclared Deep fixture");
        let undeclared = check_json(dir.path(), &undeclared_path);
        let messages = error_messages(&undeclared);
        assert!(
            messages.len() == 1
                && messages[0].contains(&format!("`{binder}`"))
                && messages[0].contains("undeclared"),
            "an ordinary body annotation must not declare `{binder}`: {undeclared}"
        );

        let surf_path = format!("surf-body-undeclared-{index}.ch");
        let surf_annotation = match binder {
            "p" if index == 0 => "p",
            "p" => "tensor[2, p]",
            "n" => "tensor[n, f32]",
            "r" => "tensor[..r, f32]",
            _ => unreachable!("closed test table"),
        };
        fs::write(
            dir.path().join(&surf_path),
            format!("def maker() = fn (x: {surf_annotation}) -> x\n"),
        )
        .expect("write undeclared Surf fixture");
        let surf_undeclared = check_json(dir.path(), &surf_path);
        assert!(
            error_messages(&surf_undeclared)
                .iter()
                .any(|message| message.contains(&format!("`{binder}`"))),
            "Surf body annotation must not declare `{binder}`: {surf_undeclared}"
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

#[test]
fn cli_round_trips_and_checks_polymorphic_properties() {
    let dir = tempdir().expect("tempdir");
    let source = "@property accepts[p] forall(x: p):\n  true\n";
    fs::write(dir.path().join("property.ch"), source).expect("write Surf property");

    let checked = check_json(dir.path(), "property.ch");
    assert_eq!(
        checked["score"].as_f64(),
        Some(1.0),
        "the Surf property must check: {checked}"
    );

    let deep = run(dir.path(), &["deep", "property.ch"]);
    assert!(
        deep.status.success(),
        "`chelis deep` failed: {}",
        text(&deep)
    );
    let deep_text = String::from_utf8_lossy(&deep.stdout);
    assert!(
        deep_text.contains("(defsig {} accepts (p) (t-fn {} (t-var {} p) (t-prim {} bool)))"),
        "Deep must carry the property binder on its defsig: {deep_text}"
    );
    fs::write(dir.path().join("property.dp"), &deep.stdout).expect("write Deep property");

    let surf = run(dir.path(), &["surf", "property.dp"]);
    assert!(
        surf.status.success(),
        "`chelis surf` failed: {}",
        text(&surf)
    );
    assert_eq!(String::from_utf8_lossy(&surf.stdout), source);
    fs::write(dir.path().join("recovered.ch"), &surf.stdout).expect("write recovered Surf");
    let recovered = check_json(dir.path(), "recovered.ch");
    assert_eq!(
        recovered["score"].as_f64(),
        Some(1.0),
        "the recovered property must check: {recovered}"
    );
}
