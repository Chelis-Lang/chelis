//! [05-HOST-1], [05-OP-36], spec/06 sections 2.1 and 2.10.1.
//! Host selectors retain exact values and lexical bindings during numeric AD.
use chelis_compiler_api::compiler::eval;
use chelis_compiler_api::schema::{EvalRequest, ExecutionValue, SourceKind};
use std::collections::BTreeMap;

fn gradient(source: &str) -> Result<Vec<f64>, String> {
    let result = eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: source.into(),
        bindings: BTreeMap::new(),
    })
    .map_err(|e| format!("{e:?}"))?;
    match &result
        .roots
        .iter()
        .find(|r| r.name.as_deref() == Some("out"))
        .expect("out root")
        .value
    {
        ExecutionValue::Tensor { value } => {
            assert_eq!(value.shape, vec![2]);
            assert_eq!(value.data.prim(), chelis_types::types::Prim::F32);
            Ok(value.data.to_f64_lossy_vec())
        }
        other => panic!("expected tensor, got {other:?}"),
    }
}

fn selected(key: &str) -> String {
    format!(
        r#"
def pick[n](w: tensor[n, f32], name: string) -> tensor[n, f32] = if eq(name, "w") then w else neg(w)
def loss[n](w: tensor[n, f32]) -> tensor[f32] = sum(pick(w, "{key}"), 0i32)
out = grad(loss, wrt=w)(to_tensor([1.0f32, 2.0f32]))
"#
    )
}

#[test]
fn literal_selector_preserves_selected_gradient() {
    assert_eq!(gradient(&selected("w")).unwrap(), vec![1.0, 1.0]);
}

#[test]
fn wrong_key_selects_opposite_gradient() {
    assert_eq!(gradient(&selected("other")).unwrap(), vec![-1.0, -1.0]);
}

#[test]
fn host_selector_prunes_untaken_failure() {
    let source = r#"
def pick[n](w: tensor[n, f32], name: string) -> tensor[n, f32] = if eq(name, "w") then w else fail("wrong key")
def loss[n](w: tensor[n, f32]) -> tensor[f32] = sum(pick(w, "w"), 0i32)
out = grad(loss)(to_tensor([1.0f32, 2.0f32]))
"#;
    assert_eq!(gradient(source).unwrap(), vec![1.0, 1.0]);
    assert!(gradient(&source.replace("pick(w, \"w\")", "pick(w, \"other\")")).is_err());
}

#[test]
fn declaring_string_capture_survives_same_named_caller() {
    let source = r#"
name = "w"
def pick[n](w: tensor[n, f32]) -> tensor[n, f32] = if eq(name, "w") then w else neg(w)
def loss[n](w: tensor[n, f32], name: string) -> tensor[f32] = sum(pick(w), 0i32)
out = grad(loss, wrt=w)(to_tensor([1.0f32, 2.0f32]), "other")
"#;
    assert_eq!(gradient(source).unwrap(), vec![1.0, 1.0]);
}

#[test]
fn runtime_string_argument_is_discrete() {
    for (key, expected) in [("w", vec![1.0, 1.0]), ("other", vec![-1.0, -1.0])] {
        let source = format!(
            r#"
def loss[n](w: tensor[n, f32], name: string) -> tensor[f32] = if eq(name, "w") then sum(w, 0i32) else neg(sum(w, 0i32))
out = grad(loss, wrt=w)(to_tensor([1.0f32, 2.0f32]), "{key}")
"#
        );
        assert_eq!(gradient(&source).unwrap(), expected);
    }
}

#[test]
fn selected_string_target_is_a_type_error() {
    let source = r#"
def loss[n](w: tensor[n, f32], name: string) -> tensor[f32] = if eq(name, "w") then sum(w, 0i32) else neg(sum(w, 0i32))
out = grad(loss, wrt=name)(to_tensor([1.0f32, 2.0f32]), "w")
"#;
    assert!(gradient(source).is_err());
}

#[test]
fn lexical_string_capture_survives_nested_closure() {
    let source = r#"
def differentiated[n](w: tensor[n, f32], name: string) -> tensor[n, f32] = {
    target = fn (x: tensor[n, f32]) -> if eq(name, "w") then sum(x, 0i32) else neg(sum(x, 0i32))
    grad(target)(w)
}
out = differentiated(to_tensor([1.0f32, 2.0f32]), "w")
"#;
    assert_eq!(gradient(source).unwrap(), vec![1.0, 1.0]);
    assert_eq!(
        gradient(&source.replace("2.0f32]), \"w\"", "2.0f32]), \"other\"")).unwrap(),
        vec![-1.0, -1.0]
    );
}

#[test]
fn named_tensor_inside_tuple_retains_numeric_gradient() {
    let source = r#"
def loss[n](w: tensor[n, f32]) -> tensor[f32] = {
    column = ("w", w)
    value = if eq(column.0, "w") then column.1 else neg(column.1)
    sum(value, 0i32)
}
out = grad(loss)(to_tensor([1.0f32, 2.0f32]))
"#;
    assert_eq!(gradient(source).unwrap(), vec![1.0, 1.0]);
    assert_eq!(
        gradient(&source.replace("column = (\"w\"", "column = (\"other\"")).unwrap(),
        vec![-1.0, -1.0]
    );
}

#[test]
fn runtime_host_branch_stays_explicitly_unrepresentable() {
    let source = r#"
def loss[n](w: tensor[n, f32]) -> tensor[f32] = {
    name = if gt(tensor_to_scalar(sum(w, 0i32)), 0.0f32) then "w" else "other"
    if eq(name, "w") then sum(w, 0i32) else neg(sum(w, 0i32))
}
out = grad(loss)(to_tensor([1.0f32, 2.0f32]))
"#;
    let error = gradient(source).unwrap_err();
    assert!(
        error.contains("host control flow must execute in its source stage"),
        "{error}"
    );
}

#[test]
fn string_equality_preserves_unicode_and_inequality() {
    for (key, expected) in [
        ("w", vec![1.0, 1.0]),
        ("ｗ", vec![-1.0, -1.0]),
        ("", vec![-1.0, -1.0]),
    ] {
        assert_eq!(gradient(&selected(key)).unwrap(), expected);
        let neq = selected(key)
            .replace("if eq(name,", "if neq(name,")
            .replace("then w else neg(w)", "then neg(w) else w");
        assert_eq!(gradient(&neq).unwrap(), expected);
    }
}

fn native_output(source: &str) -> String {
    use chelis_compiler_api::compiler::compile_for_execution;
    use chelis_compiler_api::schema::{CompileRequest, CompileTarget};
    use std::{fs, process::Command};
    let result = compile_for_execution(CompileRequest {
        source_kind: SourceKind::Surf,
        source: source.into(),
        target: CompileTarget::C,
        entry_name: Some("host_gradient".into()),
    })
    .unwrap_or_else(|error| panic!("C build failed: {error:?}"));
    let dir = tempfile::tempdir().unwrap();
    for file in &result.compile_result.files {
        fs::write(dir.path().join(&file.path), &file.contents).unwrap();
    }
    let staged = chelis_runtime_bundle::stage(dir.path()).unwrap();
    let binary = dir.path().join("probe");
    let mut cc = Command::new(std::env::var("CC").unwrap_or_else(|_| "cc".into()));
    cc.args(["-std=c11", "-O0", "-Werror=incompatible-pointer-types"])
        .arg(dir.path().join("host_gradient.c"))
        .arg("-I")
        .arg(dir.path())
        .arg(staged.archive)
        .args(["-lm", "-lpthread"]);
    if cfg!(target_os = "macos") {
        cc.args([
            "-framework",
            "Accelerate",
            "-framework",
            "Security",
            "-framework",
            "CoreFoundation",
        ]);
    } else {
        cc.arg("-ldl");
    }
    let compiled = cc.arg("-o").arg(&binary).output().unwrap();
    assert!(
        compiled.status.success(),
        "{}",
        String::from_utf8_lossy(&compiled.stderr)
    );
    let run = Command::new(binary).output().unwrap();
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    String::from_utf8(run.stdout).unwrap()
}

#[test]
fn string_selected_gradient_executes_in_c() {
    for (key, expected) in [("w", "[1.0, 1.0]"), ("other", "[-1.0, -1.0]")] {
        let source = selected(key);
        let stdout = native_output(&source);
        assert!(
            stdout.contains(&format!("out = tensor(shape=[2], data={expected})")),
            "{stdout}"
        );
    }
}

#[test]
fn squared_loss_matches_issue_reproducer_in_both_lanes() {
    let source = selected("w").replace(
        "sum(pick(w, \"w\"), 0i32)",
        "{ v = pick(w, \"w\")\n sum(mul(v, v), 0i32) }",
    );
    assert_eq!(gradient(&source).unwrap(), vec![2.0, 4.0]);
    let stdout = native_output(&source);
    assert!(
        stdout.contains("out = tensor(shape=[2], data=[2.0, 4.0])"),
        "{stdout}"
    );
}

#[test]
fn mixed_host_tensor_argument_has_unit_host_cotangent() {
    let source = r#"
def loss[n](column: (string, tensor[n, f32])) -> tensor[f32] = if eq(column.0, "w") then sum(column.1, 0i32) else neg(sum(column.1, 0i32))
g = grad(loss)(("w", to_tensor([1.0f32, 2.0f32])))
out = g.1
"#;
    assert_eq!(gradient(source).unwrap(), vec![1.0, 1.0]);
    let result = eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: source.into(),
        bindings: BTreeMap::new(),
    })
    .unwrap();
    let unit = result
        .roots
        .iter()
        .find(|root| root.name.as_deref() == Some("g.0"))
        .unwrap_or_else(|| panic!("missing unit gradient in {:?}", result.roots));
    assert!(matches!(unit.value, ExecutionValue::Unit));
    assert_eq!(
        gradient(&source.replace("((\"w\"", "((\"other\"")).unwrap(),
        vec![-1.0, -1.0]
    );
}

#[test]
fn string_keyed_list_routes_gradient_to_selected_column() {
    let source = r#"
def lookup[n](columns: List[(string, tensor[n, f32])], name: string) -> tensor[n, f32] = match columns with {
    | Cons(pair, rest) => if eq(pair.0, name) then pair.1 else lookup(rest, name)
    | Nil => fail("missing column")
}
def loss[n](w: tensor[n, f32]) -> tensor[f32] = sum(lookup([("first", neg(w)), ("w", w)], "w"), 0i32)
out = grad(loss)(to_tensor([1.0f32, 2.0f32]))
"#;
    assert_eq!(gradient(source).unwrap(), vec![1.0, 1.0]);
    assert_eq!(
        gradient(&source.replace(", \"w\"), 0i32)", ", \"first\"), 0i32)")).unwrap(),
        vec![-1.0, -1.0]
    );
    let missing = source.replace(", \"w\"), 0i32)", ", \"missing\"), 0i32)");
    assert!(gradient(&missing).is_err());
    assert!(native_output(source).contains("out = tensor(shape=[2], data=[1.0, 1.0])"));
}
