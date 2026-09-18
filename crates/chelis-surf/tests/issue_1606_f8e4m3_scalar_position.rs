//! chelis#1606: `f8e4m3` in a scalar type position desugars to
//! `(t-prim {} f8e4m3)`, never to an invented
//! `(t-var {} f8e4m3)`.
//!
//! `f8e4m3` is a real `Prim` variant, unlike the other `spec/04-type-system.md`
//! §1.1.1 reserved names, so it was excluded from `DEFERRED_DTYPE_NAMES` in
//! `crates/chelis-surf/src/desugar.rs` on the theory that the checker's
//! `Prim::parse_name` path would catch it downstream. It did not: excluding it
//! from BOTH `PRIMITIVES` and `DEFERRED_DTYPE_NAMES` made it a candidate
//! quantified type variable, so `def f(x: f8e4m3) -> f8e4m3 = x` typed as
//! `forall f8e4m3. f8e4m3 -> f8e4m3` and scored 1.0. Same class as chelis#1593,
//! different cause: not a missing primitive alias but a name that IS one.
//!
//! Mirrors `issue_1593_unsigned_scalar_position.rs`'s structure for the one
//! name that repair could not cover.

use chelis_deep::Span;
use chelis_deep::parser::parse_and_stamp_file;
use chelis_deep::printer::print_canonical_flat;
use chelis_surf::ast::{TensorPrecision, TypeExpr};
use chelis_surf::desugar::desugar_program;
use chelis_surf::format::format_source;
use chelis_surf::parser::parse_str;

fn deep_of(source: &str) -> String {
    let decls = parse_str(source).expect("parse");
    print_canonical_flat(&desugar_program(&decls))
}

fn prim_of(name: &str) -> String {
    format!("(t-prim {{}} {name})")
}

fn tvar_of(name: &str) -> String {
    format!("(t-var {{}} {name})")
}

fn assert_reaches_rejection(deep: &str, position: &str) {
    assert!(
        deep.contains(&prim_of("f8e4m3")),
        "`f8e4m3` in {position} must desugar to a `t-prim` so the §1.1.1 \
         rejection can fire: {deep}"
    );
    assert!(
        !deep.contains(&tvar_of("f8e4m3")),
        "`f8e4m3` in {position} must not be absorbed as a quantified type \
         variable: {deep}"
    );
}

fn assert_forbidden_binder(source: &str, position: &str) {
    let error = parse_str(source).expect_err("f8e4m3 cannot enter a declaration binder list");
    let rendered = error.to_string();
    assert!(
        rendered.contains("`f8e4m3`") && rendered.contains("cannot be a declaration binder"),
        "wrong forbidden-binder diagnostic for {position}: {rendered}"
    );
}

/// REGRESSION test. The named instance of chelis#1606: a scalar parameter
/// and return annotation.
#[test]
fn f8e4m3_in_a_scalar_parameter_and_return_reaches_the_rejection() {
    let deep = deep_of("module P.M\nexport (f)\ndef f(x: f8e4m3) -> f8e4m3 = x\n");
    assert_reaches_rejection(&deep, "a scalar parameter and return");
}

/// REGRESSION test. A standalone `sig`.
#[test]
fn f8e4m3_in_a_sig_reaches_the_rejection() {
    let deep = deep_of("module P.M\nexport (f)\nsig f: f8e4m3 -> f8e4m3\ndef f(x) = x\n");
    assert_reaches_rejection(&deep, "a sig");
}

/// REGRESSION test. An explicit `[..]` quantifier list naming `f8e4m3` must
/// reject it before either scalar or tensor use can be rebound.
#[test]
fn f8e4m3_explicit_binder_does_not_rebind_in_scalar_or_tensor_position() {
    assert_forbidden_binder(
        "module P.M\nexport (f)\ndef f[f8e4m3](x: f8e4m3) -> f8e4m3 = x\n",
        "an explicit binder in a scalar position",
    );
    assert_forbidden_binder(
        "module P.M\nexport (f)\n\
         def f[f8e4m3](x: tensor[3, f8e4m3]) -> tensor[3, f8e4m3] = x\n",
        "an explicit binder in a tensor precision slot",
    );
}

/// REGRESSION test. A `&f8e4m3` reference type.
#[test]
fn f8e4m3_behind_a_reference_reaches_the_rejection() {
    let deep = deep_of("module P.M\nexport (f)\ndef f(x: &f8e4m3) -> i32 = 0i32\n");
    assert_reaches_rejection(&deep, "a reference type");
}

/// REGRESSION test. A tuple element.
#[test]
fn f8e4m3_in_a_tuple_element_reaches_the_rejection() {
    let deep = deep_of("module P.M\nexport (f)\ndef f(x: (f8e4m3, i32)) -> i32 = 0i32\n");
    assert_reaches_rejection(&deep, "a tuple element");
}

/// REGRESSION test. An arrow (function-value) parameter.
#[test]
fn f8e4m3_in_an_arrow_parameter_reaches_the_rejection() {
    let deep = deep_of("module P.M\nexport (f)\ndef f(g: (f8e4m3) -> i32) -> i32 = 0i32\n");
    assert_reaches_rejection(&deep, "an arrow parameter");
}

/// REGRESSION test. A `List[f8e4m3]` element type.
#[test]
fn f8e4m3_in_a_list_element_reaches_the_rejection() {
    let deep = deep_of("module P.M\nexport (f)\ndef f(x: List[f8e4m3]) -> i32 = 0i32\n");
    assert_reaches_rejection(&deep, "a List element");
}

/// DISPOSITION LOCK. Green in both states. The tensor precision slot's
/// non-primitive fall-through was already `t-prim` outside an explicit
/// binder; this row confirms the ordinary sig-precision case is unaffected.
#[test]
fn the_tensor_precision_slot_still_emits_t_prim_for_f8e4m3() {
    let deep = deep_of(
        "module P.M\nexport (f)\nsig f[d]: tensor[d, f8e4m3] -> tensor[d, f8e4m3]\n\
         def f(x) = x\n",
    );
    assert_reaches_rejection(&deep, "a tensor precision slot");
}

/// DISPOSITION LOCK. Green in both states. The cast target already reached
/// the §1.1.1 diagnostic through its own path; it must keep desugaring to
/// `t-prim`.
#[test]
fn a_cast_target_still_emits_t_prim_for_f8e4m3() {
    let deep = deep_of("module P.M\nexport (f)\ndef f() -> i32 = cast(1i32, f8e4m3)\n");
    assert_reaches_rejection(&deep, "a cast target");
}

/// DISPOSITION LOCK. Positive control: a supported float dtype must still
/// desugar to `t-prim` and must still typecheck as itself, not be disturbed
/// by the reserved-name routing.
#[test]
fn a_supported_float_is_unaffected() {
    for name in ["f32", "f64", "f16", "bf16"] {
        let deep = deep_of(&format!(
            "module P.M\nexport (f)\ndef f(x: {name}) -> {name} = x\n"
        ));
        assert!(
            deep.contains(&prim_of(name)),
            "`{name}` must still desugar to `t-prim`: {deep}"
        );
    }
}

/// DISPOSITION LOCK. An ordinary explicitly listed lowercase binder remains a
/// type variable; the reserved-name routing must not widen.
#[test]
fn a_genuine_explicit_lowercase_binder_still_quantifies() {
    let deep = deep_of("module P.M\nexport (f)\ndef f[a](x: a) -> a = x\n");
    assert!(
        deep.contains(&tvar_of("a")) && !deep.contains(&prim_of("a")),
        "`a` must stay a quantified type variable: {deep}"
    );
}

/// REGRESSION test. `spec/02-surf-syntax.md` §0.1's round-trip law: a
/// hand-written Deep `(t-var {} f8e4m3)` has no Surf representation (since
/// Surf can never produce it from source any longer) and the decompiler must
/// fail closed rather than print Surf this same build rejects.
#[test]
fn the_decompiler_refuses_a_reserved_f8e4m3_type_variable() {
    const TEMPLATE: &str = "(module {surf_path: \"P.M\"}\n\
         p.m\n\
         (export {} f)\n\
         (defsig {} f (f8e4m3) (t-fn {} (t-var {} f8e4m3) (t-var {} f8e4m3)))\n\
         (def {} f (fn {} (params {} (x {type: (t-var {} f8e4m3)})) (var {} x))))";
    let deep = parse_and_stamp_file(TEMPLATE).expect("deep parse");
    let error = chelis_surf::decompile::try_decompile_program(&deep)
        .expect_err("`(t-var {} f8e4m3)` has no Surf representation and must fail closed");
    let rendered = error.to_string();
    assert!(
        rendered.contains("f8e4m3") && rendered.contains("type variable"),
        "the refusal must name `f8e4m3` and the role: {rendered}"
    );
}

/// DISPOSITION LOCK. Idempotent formatting: the formatter must leave a bare
/// `f8e4m3` annotation untouched (it names no active primitive to rewrite to).
#[test]
fn the_formatter_leaves_f8e4m3_unchanged() {
    let source = "module P.M\nexport (f)\ndef f(x: f8e4m3) -> f8e4m3 = x\n";
    let once = format_source(source).expect("format once");
    assert!(
        once.contains("f8e4m3"),
        "the formatter must preserve `f8e4m3`: {once}"
    );
    let twice = format_source(&once).expect("format twice");
    assert_eq!(once, twice, "formatter idempotence on `f8e4m3`");
}

#[test]
fn tensor_precision_serde_preserves_current_spans_and_accepts_legacy_strings() {
    for name in ["f8e4m3", "f8e5m2"] {
        let current = TypeExpr::Tensor(
            Vec::new(),
            TensorPrecision::new(name, Span::new(26, name.len())),
            Span::new(16, 17),
        );
        let encoded = serde_json::to_value(&current).expect("serialize current Surf AST");
        assert_eq!(encoded["Tensor"][1]["name"], name);
        assert_eq!(encoded["Tensor"][1]["span"]["offset"], 26);
        assert_eq!(encoded["Tensor"][1]["span"]["len"], name.len());
        let decoded: TypeExpr =
            serde_json::from_value(encoded).expect("deserialize current Surf AST");
        assert_eq!(decoded, current, "same-version spans must round-trip");

        let legacy = serde_json::json!({
            "Tensor": [
                [],
                name,
                {"offset": 16, "len": 17}
            ]
        });
        let decoded: TypeExpr =
            serde_json::from_value(legacy).expect("deserialize legacy string precision");
        let TypeExpr::Tensor(_, precision, span) = decoded else {
            panic!("legacy tensor AST decoded to the wrong variant");
        };
        assert_eq!(precision.as_str(), name);
        assert_eq!(precision.span(), Span::new(0, 0));
        assert_eq!(span, Span::new(16, 17));
    }
}
