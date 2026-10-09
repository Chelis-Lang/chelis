//! Borrow disposition is family-wide. A family read after an ordinary consume
//! is consuming fan-out that an inserted copy repairs, and after a `drop` it is
//! a use-after-consume (spec/04 section 8.3, spec/05 section 1.3.1).
#[path = "common/mod.rs"]
mod common;
use assert_cmd::Command;
use serde_json::Value;
use std::fs;
use tempfile::tempdir;
const COMPARISONS: &[&str] = &["cmplt", "lt", "gt", "lte", "gte", "eq", "neq"];

#[test]
fn intrinsic_borrowing_preserves_dropout_but_respects_shadowing() {
    let valid = check(
        "def noisy(x: tensor[4, f32]) -> tensor[4, f32] = dropout(key_from_seed(42i64), x, tensor_to_scalar(sum(x, 0)))\n",
    );
    assert!(valid["errors"].as_array().unwrap().is_empty(), "{valid}");
    assert_eq!(
        repairs(
            "def noisy(x: tensor[4, f32]) -> tensor[4, f32] = dropout(key_from_seed(42i64), x, tensor_to_scalar(sum(x, 0)))\n"
        ),
        Vec::<(String, String)>::new(),
        "the intrinsic borrows, so nothing is copied"
    );
    // A local `dropout` takes `y` by value, so the call consumes `b` and the
    // later read copies at it.
    let shadowed = "def bad(b: tensor[2, f32]) = { dropout = fn (k: key, y: tensor[2, f32], rate: f32) -> realize(y)\nfirst = dropout(key_from_seed(42i64), b, 0.5f32)\nadd(first, b) }\n";
    assert!(check(shadowed)["errors"].as_array().unwrap().is_empty());
    assert_eq!(
        repairs(shadowed),
        [("b".to_string(), "call to `dropout`".to_string())]
    );
    let consumed = "def bad(x: tensor[4, f32]) = { gone = realize(x)\ndropout(key_from_seed(42i64), x, 0.5f32) }\n";
    assert!(check(consumed)["errors"].as_array().unwrap().is_empty());
    assert_eq!(
        repairs(consumed),
        [("x".to_string(), "realize".to_string())]
    );
    let dropped = check(
        "def bad(x: tensor[4, f32]) = { gone = drop(x)\ndropout(key_from_seed(42i64), x, 0.5f32) }\n",
    );
    assert!(
        dropped["errors"]
            .as_array()
            .unwrap()
            .iter()
            .any(|e| e["kind"] == "UseAfterConsume"),
        "{dropped}"
    );
    let key_reuse = check(
        "def bad(x: tensor[4, f32]) = { key = key_from_seed(42i64)\nfirst = dropout(key, x, 0.5f32)\ndropout(key, first, 0.5f32) }\n",
    );
    assert!(
        key_reuse["errors"]
            .as_array()
            .unwrap()
            .iter()
            .any(|e| e["kind"] == "KeyReuse"),
        "{key_reuse}"
    );
}

/// `(binding, consumed_by)` of every copy repair `chelis cost --json` reports.
fn repairs(source: &str) -> Vec<(String, String)> {
    let dir = tempdir().unwrap();
    let path = dir.path().join("probe.ch");
    fs::write(&path, source).unwrap();
    let out = Command::cargo_bin("chelis")
        .unwrap()
        .arg("cost")
        .arg(path)
        .arg("--json")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let report: Value = serde_json::from_slice(&out.stdout).unwrap();
    report["copy_repairs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|repair| {
            (
                repair["binding"].as_str().unwrap().to_string(),
                repair["consumed_by"].as_str().unwrap().to_string(),
            )
        })
        .collect()
}

fn check(source: &str) -> Value {
    let dir = tempdir().unwrap();
    let path = dir.path().join("probe.ch");
    fs::write(&path, source).unwrap();
    let out = Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .arg("check")
        .arg(path)
        .output()
        .unwrap();
    serde_json::from_slice(&out.stdout).unwrap()
}

#[test]
fn read_only_families_leave_both_owners_live() {
    let mut failures = Vec::new();
    for op in COMPARISONS {
        let source = format!(
            "def pick(a: tensor[2, f32], b: tensor[2, f32]) -> tensor[2, f32] = where({op}(a, b), a, b)\n"
        );
        let report = check(&source);
        if !report["errors"].as_array().unwrap().is_empty() {
            failures.push(format!("{op}: {report}"));
        }
    }
    for op in ["max_elem", "min_elem"] {
        let report = check(&format!(
            "def pick(a: tensor[2, f32], b: tensor[2, f32]) -> tensor[2, f32] = add(add({op}(a, b), a), b)\n"
        ));
        if !report["errors"].as_array().unwrap().is_empty() {
            failures.push(format!("{op}: {report}"));
        }
    }
    for op in ["sigmoid", "tanh", "silu", "gelu"] {
        let report = check(&format!(
            "def use(x: tensor[2, f32]) -> tensor[2, f32] = add({op}(x), x)\n"
        ));
        if !report["errors"].as_array().unwrap().is_empty() {
            failures.push(format!("{op}: {report}"));
        }
    }
    for expression in [
        "where((a > b), a, b)",
        "where((b < a), a, b)",
        "where(lt(b, a), a, b)",
    ] {
        let report = check(&format!(
            "def pick(a: tensor[2, f32], b: tensor[2, f32]) -> tensor[2, f32] = {expression}\n"
        ));
        if !report["errors"].as_array().unwrap().is_empty() {
            failures.push(format!("{expression}: {report}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn later_family_reads_copy_after_a_consume_and_reject_after_a_drop() {
    for op in COMPARISONS.iter().copied().chain(["max_elem", "min_elem"]) {
        let result = if COMPARISONS.contains(&op) {
            "tensor[2, bool]"
        } else {
            "tensor[2, f32]"
        };
        let realized = format!(
            "def use(a: tensor[2, f32], b: tensor[2, f32]) -> {result} = {{ dead = realize(a)\n {op}(a, b) }}\n"
        );
        let report = check(&realized);
        assert!(
            report["errors"].as_array().unwrap().is_empty(),
            "{op}: {report}"
        );
        assert_eq!(
            repairs(&realized),
            [("a".to_string(), "realize".to_string())],
            "{op}"
        );
        let report = check(&format!(
            "def use(a: tensor[2, f32], b: tensor[2, f32]) -> {result} = {{ dead = drop(a)\n {op}(a, b) }}\n"
        ));
        assert!(
            report["errors"]
                .as_array()
                .unwrap()
                .iter()
                .any(|e| e["kind"] == "UseAfterConsume"
                    && e["message"].as_str().unwrap().contains("call to `drop`")),
            "{op}: {report}"
        );
    }
    for op in ["sigmoid", "tanh", "silu", "gelu"] {
        let realized = format!(
            "def use(x: tensor[2, f32]) -> tensor[2, f32] = {{ dead = realize(x)\n {op}(x) }}\n"
        );
        let report = check(&realized);
        assert!(
            report["errors"].as_array().unwrap().is_empty(),
            "{op}: {report}"
        );
        assert_eq!(
            repairs(&realized),
            [("x".to_string(), "realize".to_string())],
            "{op}"
        );
        let report = check(&format!(
            "def use(x: tensor[2, f32]) -> tensor[2, f32] = {{ dead = drop(x)\n {op}(x) }}\n"
        ));
        assert!(
            report["errors"]
                .as_array()
                .unwrap()
                .iter()
                .any(|e| e["kind"] == "UseAfterConsume"),
            "{op}: {report}"
        );
    }
}

#[test]
fn host_comparisons_match_eval_and_reject_bad_operands() {
    for dtype in ["f16", "bf16", "f32", "f64"] {
        let mut source = format!(
            "def lhs() -> tensor[2, {dtype}] = to_tensor([1.0{dtype}, 3.0{dtype}])\ndef rhs() -> tensor[2, {dtype}] = to_tensor([2.0{dtype}, 3.0{dtype}])\n"
        );
        for op in COMPARISONS {
            source.push_str(&format!("out_{op} = print({op}(lhs(), rhs()))\n"));
        }
        let dir = tempdir().unwrap();
        let path = dir.path().join("probe.ch");
        fs::write(&path, &source).unwrap();
        let out = Command::cargo_bin("chelis")
            .unwrap()
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .args(["eval", "--file"])
            .arg(path)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert_eq!(
            common::build_and_run(&source, "probe"),
            String::from_utf8(out.stdout).unwrap(),
            "{dtype}"
        );
    }
    for op in COMPARISONS {
        for rhs in ["tensor[3, f32]", "tensor[2, f64]"] {
            let report = check(&format!(
                "def bad(a: tensor[2, f32], b: {rhs}) = {op}(a, b)\n"
            ));
            assert!(
                !report["errors"].as_array().unwrap().is_empty(),
                "{op}: {report}"
            );
        }
    }
}

#[test]
fn consuming_user_bindings_do_not_inherit_builtin_borrows() {
    for name in COMPARISONS.iter().copied().chain([
        "min_elem", "max_elem", "sigmoid", "tanh", "silu", "gelu", "ordinary",
    ]) {
        for consume_rhs in [false, true] {
            let parameter = if consume_rhs { "rhs" } else { "lhs" };
            let argument = if consume_rhs { "b" } else { "a" };
            let source = format!(
                "def bad(a: tensor[2, f32], b: tensor[2, f32]) = {{ {name} = fn (lhs: tensor[2, f32], rhs: tensor[2, f32]) -> realize({parameter})\n first = {name}(a, b)\n add(first, {argument}) }}\n"
            );
            let report = check(&source);
            assert!(
                report["errors"].as_array().unwrap().is_empty(),
                "{name}/{parameter}: {report}"
            );
            // The local binding takes both operands by value, so the call
            // consumes the one read again later, and the copy sits at it.
            assert_eq!(
                repairs(&source),
                [(argument.to_string(), format!("call to `{name}`"))],
                "{name}/{parameter}"
            );
        }
    }
}

#[test]
fn explicitly_borrowed_user_bindings_leave_owner_live() {
    for name in ["tanh", "min_elem", "ordinary"] {
        let report = check(&format!(
            "def good(x: tensor[2, f32]) = {{ {name} = fn (a: &tensor[2, f32]) -> add(a, a)\n first = {name}(x)\n add(first, x) }}\n"
        ));
        assert!(
            report["errors"].as_array().unwrap().is_empty(),
            "{name}: {report}"
        );
    }
}
