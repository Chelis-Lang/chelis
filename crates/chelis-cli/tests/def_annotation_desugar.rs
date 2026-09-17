//! WS-A6 acceptance tests: contextual desugar for def parameter annotations.
//!
//! Pin the bug surfaced during the WS-C v2 stdlib generalization:
//! `def forward[batch, seq, vocab, hidden, p](table: &tensor[vocab, hidden, p])`
//! was rejected because `p` reached the type checker as a `t-prim` in
//! the def's parameter annotation, even though the def's `[..]`
//! quantifier list explicitly named it. The WS-A5 contextual rule for
//! sigs is here extended to cover def parameter annotations per
//! `spec/02-surf-syntax.md` §P4b.
//!
//! Coverage:
//!
//! 1. Reproducer: a def with an explicit precision tvar in its
//!    quantifier list type-checks.
//! 2. Two independent precision tvars in one def's quantifier list
//!    can be instantiated independently at call sites.
//! 3. A precision name in a def parameter annotation that is NOT in
//!    an explicit def quantifier list still errors.
//! 4. An explicitly quantified unusual name stays one shared precision
//!    binder across its signature and checked metadata.
//! 5. Sig-only WS-A5 generalization still works (regression guard).

use assert_cmd::Command;
use serde_json::Value;
use std::fs;
use std::path::Path;
use tempfile::tempdir;

fn write_file(path: &Path, contents: &str) {
    fs::write(path, contents).expect("write file");
}

// Issue #207: `chelis check` now exits non-zero when the JSON
// `errors` array is non-empty. The negative cases below
// (`def_precision_name_not_in_quantifier_list_errors`) expect errors,
// so this helper captures stdout regardless of exit code.
fn run_json_check(path: &Path) -> Value {
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", path.to_str().unwrap()])
        .output()
        .expect("run chelis check");
    serde_json::from_slice(&output.stdout).expect("check output should be json")
}

fn run_annotated_deep(path: &Path) -> String {
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["deep", "--annotate", path.to_str().unwrap()])
        .output()
        .expect("run chelis deep --annotate");
    assert!(
        output.status.success(),
        "annotated desugaring should succeed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("annotated Deep should be UTF-8")
}

/// The WS-C v2 reproducer. Before WS-A6, the def's parameter annotation
/// `&tensor[vocab, hidden, p]` was desugared with an empty tvar scope,
/// so `p` became `(t-prim {} p)` and `validate_tensor_precisions_in_program`
/// rejected it as an unknown primitive. After WS-A6, `p` appears in
/// the def's `[..]` quantifier list, so the contextual rule emits
/// `(t-var {} p)` and the program type-checks.
#[test]
fn def_quantifier_precision_tvar_in_param_annotation_typechecks() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("def_quantifier_p.ch");
    write_file(
        &path,
        "def take_table[vocab, hidden, p](table: &tensor[vocab, hidden, p]) \
         -> &tensor[vocab, hidden, p] = table\n",
    );

    let json = run_json_check(&path);
    let errors = json["errors"].as_array().cloned().unwrap_or_default();
    assert!(
        errors.is_empty(),
        "def with explicit precision tvar in quantifier list should \
         type-check, got errors {errors:?}"
    );
    assert!(
        (json["score"].as_f64().unwrap_or(0.0) - 1.0).abs() < f64::EPSILON,
        "score must be 1.0 for the WS-A6 reproducer, got {json}"
    );
}

/// Two independent precision tvars in a def's quantifier list. Calling
/// with mismatched precisions across the two tvars is allowed (they're
/// independent); calling with mismatched precisions for the SAME tvar
/// errors at unification.
#[test]
fn def_two_independent_precision_tvars_accepted() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("two_tvars.ch");
    write_file(
        &path,
        "def two_args[p, q](x: &tensor[3, p], y: &tensor[3, q]) -> &tensor[3, p] = x\n\
         def caller(a: &tensor[3, f32], b: &tensor[3, i32]) -> &tensor[3, f32] = two_args(a, b)\n",
    );

    let json = run_json_check(&path);
    let errors = json["errors"].as_array().cloned().unwrap_or_default();
    assert!(
        errors.is_empty(),
        "two independent precision tvars in def quantifier list should \
         accept distinct instantiations, got {errors:?}"
    );
}

/// Negative case: a precision name that is NOT in the def's binder list
/// still errors. The def declares `[a, b]` but uses `p` in a precision slot.
/// `p` is not in `{a, b}`, so it remains a primitive request and the shared
/// resolver rejects it.
#[test]
fn def_precision_name_not_in_quantifier_list_errors() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("not_in_quantifier.ch");
    write_file(
        &path,
        "def f[a, b](x: &tensor[3, p]) -> &tensor[3, p] = x\n",
    );

    let json = run_json_check(&path);
    let errors = json["errors"].as_array().cloned().unwrap_or_default();
    let all_are_unknown_primitive = errors.iter().all(|e| {
        let kind = e.get("kind").and_then(|k| k.as_str()).unwrap_or("");
        let msg = e.get("message").and_then(|m| m.as_str()).unwrap_or("");
        kind == "TypeMismatch"
            && msg.contains("unknown primitive type `p`")
            && e.get("suggestions")
                .and_then(|suggestions| suggestions.as_array())
                .is_some_and(|suggestions| {
                    suggestions.iter().any(|suggestion| {
                        suggestion.as_str().is_some_and(|text| {
                            text.contains("declare `p` in the signature binder list")
                        })
                    })
                })
    });
    assert_eq!(
        errors.len(),
        2,
        "each undeclared primitive use must have one owning diagnostic: {errors:?}"
    );
    assert!(
        all_are_unknown_primitive,
        "precision name `p` not in def binder list `[a, b]` must be rejected \
         by the shared primitive resolver per spec/02-surf-syntax.md §P4b. \
         Got {errors:?}"
    );
}

/// Surf P4b applies the explicit binder list to the synthesized signature.
/// The parameter and result annotations share one `weirdname` binder, and
/// the checked parameter metadata retains that binder rather than degrading
/// it to an unsupported primitive.
#[test]
fn def_with_explicit_unusual_name_keeps_shared_precision_binder() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("no_quantifier.ch");
    write_file(
        &path,
        "def f[weirdname](x: &tensor[3, weirdname]) -> &tensor[3, weirdname] = x\n",
    );

    let json = run_json_check(&path);
    let errors = json["errors"].as_array().cloned().unwrap_or_default();
    assert!(
        errors.is_empty(),
        "explicit unusual precision binder should type-check, got {errors:?}"
    );
    assert_eq!(json["score"].as_f64(), Some(1.0), "{json:#?}");

    let deep = run_annotated_deep(&path);
    assert_eq!(
        deep.matches("(t-var {} weirdname)").count(),
        3,
        "the defsig input/result and parameter metadata must share \
         the authored precision binder:\n{deep}"
    );
    assert!(
        deep.contains(
            "(x {type: (t-ref {} (t-tensor {} (d-lit {} 3) \
             (t-var {} weirdname)))})"
        ),
        "checked parameter metadata must retain the synthesized precision binder:\n{deep}"
    );
    assert!(
        !deep.contains("(t-prim {} weirdname)"),
        "the explicit binder must never become an unsupported primitive:\n{deep}"
    );
}

/// Mixing dim-vars and precision-tvars in the same quantifier list:
/// the user's `[batch, seq, hidden, p]` declares four polymorphic
/// names. Three appear in dim slots (so they desugar to `d-var`); one
/// appears in the precision slot (so it desugars to `t-var`). The
/// position inside the tensor type is what determines kind; the
/// quantifier list is unkinded.
#[test]
fn def_mixed_dim_and_precision_quantifiers_accepted() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("mixed.ch");
    // Fixture renamed from `take` for chelis#353: `take` is a builtin
    // name and bare shadowing defs are now rejected at declaration time.
    write_file(
        &path,
        "def grab[batch, seq, hidden, p](x: &tensor[batch, seq, hidden, p]) \
         -> &tensor[batch, seq, hidden, p] = x\n",
    );

    let json = run_json_check(&path);
    let errors = json["errors"].as_array().cloned().unwrap_or_default();
    assert!(
        errors.is_empty(),
        "def with mixed dim and precision quantifiers should type-check, \
         got {errors:?}"
    );
}

/// Arithmetic-dtype matrix: a def with an explicit precision tvar in
/// its quantifier list must type-check when instantiated at every
/// active arithmetic dtype, not only the float/int spot checks above.
/// This loop was consolidated here from `stdlib_precision_generalization_followups.rs`
/// (formerly `wsa6_def_param_annotation_precision_quantifier_typechecks`)
/// in the e2e parsimony pass so the dtype-matrix coverage lives with
/// the invariant's owning file.
#[test]
fn def_quantifier_precision_tvar_typechecks_at_every_arithmetic_dtype() {
    const ARITHMETIC_DTYPES: &[&str] = &["f32", "f64", "bf16", "f16", "i8", "i16", "i32", "i64"];
    for dtype in ARITHMETIC_DTYPES {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("wsa6_dtype_matrix.ch");
        let src = format!(
            "def grab[n, p](xs: &tensor[n, p]) -> &tensor[n, p] = xs\n\
             def use_at_dtype(xs: &tensor[3, {dtype}]) -> &tensor[3, {dtype}] = grab(xs)\n"
        );
        write_file(&path, &src);
        let json = run_json_check(&path);
        let errors = json["errors"].as_array().cloned().unwrap_or_default();
        assert!(
            errors.is_empty(),
            "WS-A6 def param annotation at {dtype}: expected clean check, got {errors:?}"
        );
        assert!(
            (json["score"].as_f64().unwrap_or(0.0) - 1.0).abs() < f64::EPSILON,
            "WS-A6 def param annotation at {dtype}: expected score 1.0, got {json}"
        );
    }
}

/// Regression guard: sig-only generalization remains available through the
/// explicit `sig poly_id[d, p]: tensor[d, p] -> tensor[d, p]` spelling, and
/// `poly_id` remains a polymorphic reference.
#[test]
fn wsa5_sig_only_generalization_still_works() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("wsa5_regression.ch");
    write_file(
        &path,
        "sig poly_id[d, p]: tensor[d, p] -> tensor[d, p]\n\
         def poly_id(x) = x\n\
         def use_f32(x: tensor[3, f32]) -> tensor[3, f32] = poly_id(x)\n",
    );

    let json = run_json_check(&path);
    let errors = json["errors"].as_array().cloned().unwrap_or_default();
    assert!(
        errors.is_empty(),
        "WS-A5 sig-only generalization regression: got {errors:?}"
    );
}
