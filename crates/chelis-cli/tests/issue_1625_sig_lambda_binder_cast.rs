//! chelis#1625: a sig-declared binder in a lambda-valued top-level binding
//! must reject the same way the `def` spelling does.
//!
//! `sig recast: p -> p` followed by `recast = fn (v) -> cast(v, p)` desugared
//! the cast target as `(t-prim {} p)` instead of `(t-var {} p)`, so the
//! [04-DTYPE-1] classifier in `chelis_deep::literal_source` (which keys on a
//! `t-var` target) never fired. `chelis check` scored 1.0 with no errors at
//! both ingresses; `chelis eval --file` failed later at IR lowering with
//! "cast target `p` ... does not name an active primitive dtype". The fix is
//! in `crates/chelis-surf/src/desugar.rs`'s `Decl::LetDef { ty: None, .. }`
//! arm, which now installs the same sig-binder scope `desugar_fun_def`
//! installs before desugaring the value.
//!
//! This is the sig-plus-lambda path adjacent to chelis#1544 (PR #1545) and
//! chelis#1558 (PR #1623), which covered the `def` spelling only.

use assert_cmd::Command;
use serde_json::Value;
use std::fs;
use std::path::Path;
use tempfile::tempdir;

#[test]
fn both_checker_apis_enforce_the_signature_binder() {
    for (source, accepted) in [
        ("sig recast: p -> p\nrecast = fn (v) -> cast(v, p)", false),
        ("sig recast: p -> p\nrecast = fn (v) -> cast(7, p)", false),
        (
            "sig recast[p: Float]: p -> p\nrecast = fn (v) -> cast(v, p)",
            true,
        ),
    ] {
        let declarations = chelis_surf::parser::parse_str(source).expect("parse fixture");
        let deep = chelis_surf::desugar::desugar_program(&declarations);
        let text = chelis_deep::printer::print_canonical_flat(&deep);
        let stamped = chelis_deep::parse_and_stamp_file(&text).expect("stamp fixture");
        for (name, result) in [
            ("ir", chelis_types::check_ir_program(&stamped)),
            ("typed", chelis_types::check_typed_program(&stamped)),
        ] {
            if accepted {
                assert!(result.is_ok(), "{name}: {source}: {result:?}");
            } else {
                let report = result.expect_err("unbounded cast must reject");
                assert!(
                    report.errors.iter().any(|error| {
                        error.message.contains("04-DTYPE-1")
                            && error.message.contains("cast target `p`")
                    }),
                    "{name}: {source}: {:?}",
                    report.errors
                );
            }
        }
    }
}

fn write_file(path: &Path, contents: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create parent");
    }
    fs::write(path, contents).expect("write file");
}

fn chelis(args: &[&str]) -> std::process::Output {
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(args)
        .output()
        .expect("run chelis")
}

fn run_check(path: &Path) -> Value {
    let output = chelis(&["check", path.to_str().unwrap()]);
    serde_json::from_slice(&output.stdout).expect("check output should be json")
}

fn fmt_inplace(path: &Path) {
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["fmt", "--inplace", path.to_str().unwrap()])
        .assert()
        .success();
}

fn error_messages(json: &Value) -> Vec<String> {
    json["errors"]
        .as_array()
        .expect("errors should be a json array")
        .iter()
        .map(|e| e["message"].as_str().unwrap_or("").to_string())
        .collect()
}

/// Write, canonicalize, then check the program at both ingresses (Surf and
/// Deep, via `chelis deep`). Mirrors
/// `issue_1558_binder_cast_variable_source.rs::check_both_ingresses`.
fn check_both_ingresses(dir: &Path, module: &str, body: &str) -> (Value, Value) {
    let source = dir.join(format!("{module}.ch"));
    write_file(&source, &format!("module {module}\n{body}"));
    fmt_inplace(&source);
    let surf = run_check(&source);

    let deep = chelis(&["deep", source.to_str().unwrap()]);
    assert!(
        deep.status.success(),
        "chelis deep must succeed for {module}: {}",
        String::from_utf8_lossy(&deep.stderr)
    );
    let lowered = source.with_extension("dp");
    fs::write(&lowered, &deep.stdout).expect("write .dp");
    (surf, run_check(&lowered))
}

/// chelis#1625 **regression test**: variable-source cast under an unbounded
/// `sig` binder, in a lambda-valued top-level binding. Red before the fix:
/// scored 1.0 with no errors at both ingresses.
#[test]
fn variable_source_cast_under_unbounded_sig_lambda_binder_is_rejected() {
    let dir = tempdir().expect("tempdir");
    let (surf, deep) = check_both_ingresses(
        dir.path(),
        "Issue1625Variable",
        "sig recast: p -> p\nrecast = fn (v) -> cast(v, p)\nout = recast(cast(7, i32))\n",
    );
    for (label, json) in [("surf", &surf), ("deep", &deep)] {
        assert_ne!(
            json["score"], 1.0,
            "must not check clean at the {label} ingress: {json}"
        );
        assert!(
            error_messages(json).iter().any(|m| m
                .contains("cast target `p` in `recast` does not name an active primitive")
                && m.contains("04-DTYPE-1")),
            "{label} ingress must carry the [04-DTYPE-1] rejection naming `p`; got {:?}",
            error_messages(json)
        );
    }
}

/// chelis#1625 **regression test**: literal-source variant from the issue
/// body. Red before the fix, identically to the variable-source case.
#[test]
fn literal_source_cast_under_unbounded_sig_lambda_binder_is_rejected() {
    let dir = tempdir().expect("tempdir");
    let (surf, deep) = check_both_ingresses(
        dir.path(),
        "Issue1625Literal",
        "sig recast: p -> p\nrecast = fn (v) -> cast(7, p)\nout = recast(cast(7, i32))\n",
    );
    for (label, json) in [("surf", &surf), ("deep", &deep)] {
        assert_ne!(
            json["score"], 1.0,
            "must not check clean at the {label} ingress: {json}"
        );
        assert!(
            error_messages(json).iter().any(|m| m
                .contains("cast target `p` in `recast` does not name an active primitive")
                && m.contains("04-DTYPE-1")),
            "{label} ingress must carry the [04-DTYPE-1] rejection naming `p`; got {:?}",
            error_messages(json)
        );
    }
}

/// chelis#1625 **disposition lock**: a concrete primitive target in the same
/// sig-plus-lambda shape must keep checking clean. Guards against a fix that
/// over-applies and starts rejecting ordinary primitive cast targets.
#[test]
fn concrete_primitive_target_in_sig_lambda_binding_still_checks_clean() {
    let dir = tempdir().expect("tempdir");
    let (surf, deep) = check_both_ingresses(
        dir.path(),
        "Issue1625Concrete",
        "sig widen: i32 -> f64\nwiden = fn (v) -> cast(v, f64)\nout = widen(cast(7, i32))\n",
    );
    for (label, json) in [("surf", &surf), ("deep", &deep)] {
        assert_eq!(
            json["score"], 1.0,
            "{label} ingress must check clean: {json}"
        );
        assert!(
            error_messages(json).is_empty(),
            "{label} ingress must report no errors; got {:?}",
            error_messages(json)
        );
    }
}

/// chelis#1625 **positive twin**: a bounded sig binder (spec §P4c spelling
/// `sig f[p: Float]: ...`) in the same lambda-valued shape keeps checking
/// clean, exactly as the bounded `def` form does.
#[test]
fn bounded_sig_lambda_binder_still_checks_clean() {
    let dir = tempdir().expect("tempdir");
    let (surf, deep) = check_both_ingresses(
        dir.path(),
        "Issue1625Bounded",
        "sig recast[p: Float]: p -> p\nrecast = fn (v) -> cast(v, p)\nout = recast(cast(1.5, f32))\n",
    );
    for (label, json) in [("surf", &surf), ("deep", &deep)] {
        assert_eq!(
            json["score"], 1.0,
            "{label} ingress must check clean: {json}"
        );
        assert!(
            error_messages(json).is_empty(),
            "{label} ingress must report no errors; got {:?}",
            error_messages(json)
        );
    }
}

/// chelis#1625 **scoping test**: an unrelated function that happens to
/// declare a *bounded* generic also named `p` must not affect, and must not
/// be affected by, the *unbounded* sig-plus-lambda binder `p` on `recast` in
/// the same module. Each name's binder scope is installed and restored
/// per-declaration (see `DesugarCtx::current_type_binders`), keyed by
/// `declared_type_binders`, which is itself keyed by declaration name, not a
/// single shared binder-name set — so the two `p`s must not cross-contaminate
/// in either direction.
#[test]
fn a_same_named_bounded_binder_in_another_function_does_not_leak_into_the_unbounded_one() {
    let dir = tempdir().expect("tempdir");
    let (surf, deep) = check_both_ingresses(
        dir.path(),
        "Issue1625Scoping",
        "def helper[p: Int](value: p) -> p = cast(value, p)\n\
         sig recast: p -> p\n\
         recast = fn (v) -> cast(v, p)\n\
         out = recast(cast(7, i32))\n\
         also = helper(cast(3, i32))\n",
    );
    for (label, json) in [("surf", &surf), ("deep", &deep)] {
        assert_ne!(
            json["score"], 1.0,
            "{label} ingress must still reject `recast`'s unbounded `p`: {json}"
        );
        let messages = error_messages(json);
        assert!(
            messages.iter().any(|m| m
                .contains("cast target `p` in `recast` does not name an active primitive")
                && m.contains("04-DTYPE-1")),
            "{label} ingress must reject `recast`'s unbounded `p`; got {messages:?}"
        );
        assert!(
            !messages
                .iter()
                .any(|m| m.contains("in `helper`") && m.contains("04-DTYPE-1")),
            "{label} ingress must not reject `helper`'s bounded `p`; got {messages:?}"
        );
    }
}

#[test]
fn an_explicit_primitive_sig_binder_stays_concrete_and_round_trips() {
    let dir = tempdir().expect("tempdir");
    let source = dir.path().join("primitive_binder.ch");
    let lowered = dir.path().join("primitive_binder.dp");
    write_file(
        &source,
        "module Issue1625PrimitiveBinder\n\
         sig recast[f32]: f32 -> f32\n\
         recast = fn (v) -> cast(v, f32)\n\
         out = recast(1.5f32)\n",
    );
    fmt_inplace(&source);

    let deep = chelis(&["deep", source.to_str().unwrap()]);
    assert!(
        deep.status.success(),
        "primitive-binder source must lower: {}",
        String::from_utf8_lossy(&deep.stderr)
    );
    let deep_text = String::from_utf8(deep.stdout.clone()).expect("Deep output is UTF-8");
    assert!(
        deep_text.contains("(cast {span:"),
        "fixture must retain its cast body:\n{deep_text}"
    );
    assert!(
        deep_text.contains("(t-prim {} f32)") && !deep_text.contains("(t-var {} f32)"),
        "an explicit binder list must not rebind active primitive `f32`:\n{deep_text}"
    );
    fs::write(&lowered, &deep.stdout).expect("write Deep fixture");

    for (label, json) in [("surf", run_check(&source)), ("deep", run_check(&lowered))] {
        assert_eq!(
            json["score"], 1.0,
            "{label} ingress must accept the concrete primitive: {json}"
        );
        assert!(
            error_messages(&json).is_empty(),
            "{label} ingress must report no errors; got {:?}",
            error_messages(&json)
        );
    }

    let recovered = chelis(&["surf", lowered.to_str().unwrap()]);
    assert!(
        recovered.status.success(),
        "Deep-to-Surf round-trip must succeed: {}",
        String::from_utf8_lossy(&recovered.stderr)
    );
}

#[test]
fn an_explicit_reserved_sig_binder_stays_on_the_dtype_rejection_path() {
    let dir = tempdir().expect("tempdir");
    let (surf, deep) = check_both_ingresses(
        dir.path(),
        "Issue1625ReservedBinder",
        "sig recast[u8]: u8 -> u8\n\
         recast = fn (v) -> cast(v, u8)\n\
         out = recast(1)\n",
    );
    for (label, json) in [("surf", &surf), ("deep", &deep)] {
        assert_ne!(
            json["score"], 1.0,
            "{label} ingress must reject reserved dtype `u8`: {json}"
        );
        let messages = error_messages(json);
        assert!(
            messages.iter().any(|message| {
                message.contains("cannot use `u8` as a scalar dtype")
                    && message.contains("unsigned integer types are deferred")
            }),
            "{label} ingress must use the reserved-dtype rejection; got {messages:?}"
        );
        assert!(
            !messages
                .iter()
                .any(|message| message.contains("undeclared type variable `u8`")),
            "{label} ingress must not misclassify `u8` as a binder; got {messages:?}"
        );
    }
}
