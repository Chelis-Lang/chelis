//! Borrow disposition is family-wide; a genuine consume remains terminal.
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
    let shadowed = check(
        "def bad(b: tensor[2, f32]) = { dropout = fn (k: key, y: tensor[2, f32], rate: f32) -> realize(y)\nfirst = dropout(key_from_seed(42i64), b, 0.5f32)\nadd(first, b) }\n",
    );
    assert!(
        shadowed["errors"]
            .as_array()
            .unwrap()
            .iter()
            .any(|e| e["kind"] == "UseAfterConsume"),
        "{shadowed}"
    );
    let consumed = check(
        "def bad(x: tensor[4, f32]) = { gone = realize(x)\ndropout(key_from_seed(42i64), x, 0.5f32) }\n",
    );
    assert!(
        consumed["errors"]
            .as_array()
            .unwrap()
            .iter()
            .any(|e| e["kind"] == "UseAfterConsume"),
        "{consumed}"
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
fn actual_consumes_still_reject_later_family_reads() {
    for op in COMPARISONS.iter().copied().chain(["max_elem", "min_elem"]) {
        let result = if COMPARISONS.contains(&op) {
            "tensor[2, bool]"
        } else {
            "tensor[2, f32]"
        };
        let report = check(&format!(
            "def use(a: tensor[2, f32], b: tensor[2, f32]) -> {result} = {{ dead = realize(a)\n {op}(a, b) }}\n"
        ));
        assert!(
            report["errors"]
                .as_array()
                .unwrap()
                .iter()
                .any(|e| e["kind"] == "UseAfterConsume"
                    && e["message"].as_str().unwrap().contains("realize")),
            "{op}: {report}"
        );
    }
    for op in ["sigmoid", "tanh", "silu", "gelu"] {
        let report = check(&format!(
            "def use(x: tensor[2, f32]) -> tensor[2, f32] = {{ dead = realize(x)\n {op}(x) }}\n"
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
            let report = check(&format!(
                "def bad(a: tensor[2, f32], b: tensor[2, f32]) = {{ {name} = fn (lhs: tensor[2, f32], rhs: tensor[2, f32]) -> realize({parameter})\n first = {name}(a, b)\n add(first, {}) }}\n",
                if consume_rhs { "b" } else { "a" }
            ));
            assert!(
                report["errors"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|e| e["kind"] == "UseAfterConsume"),
                "{name}/{parameter}: {report}"
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
