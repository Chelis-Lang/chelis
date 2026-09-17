//! Unit tests for inference.

use super::*;

#[test]
fn child_stamp_roles_cover_the_canonical_deep_vocabulary() {
    // Since chelis#731 Phase 3 the ownership table is an exhaustive
    // match over `DeepTag`, so completeness is compile-time and an
    // unclassified tag is unrepresentable (the old string-keyed None
    // arm no longer exists). This smoke keeps every row executed.
    for tag in chelis_deep::DeepTag::ALL {
        for index in 0..3 {
            let _ = child_stamp_role(tag, index, 3);
        }
    }
}

#[test]
fn structural_child_stamp_roles_match_owner_positions() {
    use ChildStampRole::{
        Binder, EffectHandler, ExplicitInferenceBypass, RuntimeExpr, Selector, Syntax, Type,
    };
    let cases = [
        (DeepTag::Module, 0, 2, Binder),
        (DeepTag::Module, 1, 2, ExplicitInferenceBypass),
        (DeepTag::Def, 0, 2, Binder),
        (DeepTag::Def, 1, 2, RuntimeExpr),
        (DeepTag::Deftype, 0, 3, Binder),
        (DeepTag::Deftype, 1, 3, Syntax),
        (DeepTag::Deftype, 2, 3, Type),
        (DeepTag::Typealias, 1, 3, Syntax),
        (DeepTag::Typealias, 2, 3, Type),
        (DeepTag::Fn, 0, 2, Binder),
        (DeepTag::Fn, 1, 2, RuntimeExpr),
        (DeepTag::HandleEffect, 0, 2, EffectHandler),
        (DeepTag::HandleEffect, 1, 2, RuntimeExpr),
        (DeepTag::Bind, 0, 4, Binder),
        (DeepTag::Bind, 1, 4, RuntimeExpr),
        (DeepTag::Arm, 0, 3, ExplicitInferenceBypass),
        (DeepTag::Arm, 1, 3, RuntimeExpr),
        (DeepTag::Record, 0, 2, Type),
        (DeepTag::Record, 1, 2, ExplicitInferenceBypass),
        (DeepTag::Access, 1, 2, Selector),
        (DeepTag::TupleGet, 1, 2, Selector),
        (DeepTag::Cast, 1, 2, Type),
        (DeepTag::Grad, 1, 2, Selector),
        (DeepTag::Vmap, 1, 2, Selector),
        (DeepTag::PatAs, 0, 2, Binder),
        (DeepTag::PatAs, 1, 2, ExplicitInferenceBypass),
        (DeepTag::TTensor, 0, 2, Type),
    ];
    for (tag, index, arity, expected) in cases {
        assert_eq!(
            child_stamp_role(tag, index, arity),
            expected,
            "wrong child role for `{}` child {index}",
            tag.as_str()
        );
    }
}

/// chelis#873 / loud_unsupported.md section C1 rule 4, for decode-once's
/// third structural token. Stamping is positional and `children()` skips
/// elements 0-1, so no SOURCE program can put an `Atom::Tag` in
/// expression position - which is exactly why the arm needs a canary
/// rather than a comment. Built programmatically, the way the rule says
/// to prove a path you believe is dead.
///
/// The sibling arms (`Symbol`, `Keyword`) are covered from source by
/// `bare_atom_expression_position_scores_below_one` in the CLI corpus;
/// this is the one arm that cannot be reached that way.
#[test]
fn tag_atom_in_expression_position_is_a_loud_malformed_form() {
    let program = vec![node_expr(
        DeepTag::Def,
        vec![
            symbol_expr("x"),
            deep::Expr::Atom(deep::Atom::Tag(DeepTag::App), zero_span()),
        ],
    )];
    let result = infer_program(&program);
    assert!(
        result.errors.iter().any(|error| {
            matches!(error.kind, CheckErrorKind::MalformedForm)
                && error.message.contains("a decoded tag atom `app`")
                && error.message.contains("outside a list's tag position")
        }),
        "a tag atom in expression position must raise, not type as a value; got: {:?}",
        result.errors
    );
}

/// Negative parity: the same tag in its OWN position is ordinary
/// structure and must not trip the arm above. Without this, the canary
/// could pass for an over-broad reason.
#[test]
fn tag_atom_in_tag_position_is_not_a_malformed_form() {
    let program = vec![node_expr(DeepTag::App, vec![])];
    let result = infer_program(&program);
    assert!(
        !result
            .errors
            .iter()
            .any(|error| error.message.contains("outside a list's tag position")),
        "a stamped tag at element 0 is structure, not a bare atom; got: {:?}",
        result.errors
    );
}

fn check(src: &str) -> InferResult {
    let exprs = chelis_deep::parser::parse_str(src).unwrap();
    infer_program(&exprs)
}

fn checked_surf(src: &str) -> CheckedProgram {
    let decls = chelis_surf::parser::parse_str(src).expect("surf parse");
    let exprs = chelis_surf::desugar::desugar_program(&decls);
    check_ir_program(&exprs).expect("IR check")
}

#[test]
fn checked_program_preserves_alias_resolved_generic_adt_registry() {
    let checked = checked_surf(
        r#"
type Payload[a] = a
type Boxed[a] = | Boxed(Payload[a])
def identity[a](value: Boxed[a]) -> Boxed[a] = value
"#,
    );
    let assert_checker_metadata = |stage: &str, program: &CheckedProgram| {
        assert!(
            program.adt_registry().lookup("Boxed").is_some(),
            "{stage}: checker-owned ADT registry must survive"
        );
        let identity = program
            .signature_inference()
            .functions
            .get("identity")
            .unwrap_or_else(|| panic!("{stage}: identity signature must survive"));
        assert!(
            identity.authored_signature,
            "{stage}: authored-signature provenance must survive"
        );
        assert!(
            identity.authored_signature_type.is_some(),
            "{stage}: checker-decoded authored signature must survive"
        );
    };
    assert_checker_metadata("initial checking", &checked);
    let boxed = checked
        .adt_registry()
        .lookup("Boxed")
        .expect("checked registry must retain Boxed");
    assert_eq!(boxed.type_params, ["a"]);
    assert_eq!(boxed.variants.len(), 1);
    let field = &boxed.variants[0].fields[0].1;
    assert!(
        matches!(field, Type::Var(_)),
        "aliases must be checker-resolved and the stored payload retained: {field:?}"
    );

    let encoded = bincode::serialize(&checked).expect("CheckedProgram serializes");
    let decoded: CheckedProgram =
        bincode::deserialize(&encoded).expect("CheckedProgram deserializes");
    assert_checker_metadata("serialization", &decoded);
    assert_eq!(
        decoded
            .adt_registry()
            .lookup("Boxed")
            .expect("registry survives serialization")
            .variants[0]
            .fields[0]
            .1,
        *field
    );
    assert_eq!(
        decoded
            .signature_inference()
            .functions
            .get("identity")
            .and_then(|signature| signature.authored_signature_type.as_ref()),
        checked
            .signature_inference()
            .functions
            .get("identity")
            .and_then(|signature| signature.authored_signature_type.as_ref()),
        "checker-decoded authored signatures survive serialization"
    );
    let effects_reannotated = checked
        .try_with_effect_annotations(checked.annotated_exprs().to_vec())
        .expect("effects-only rewrite preserves checked metadata");
    assert_checker_metadata("effects reannotation", &effects_reannotated);
    let linearity_checked = crate::linearity::check_linearity(&effects_reannotated)
        .expect("linearity preserves checked metadata");
    assert_checker_metadata("linearity", &linearity_checked);
    assert!(
        CheckedProgram::compose(&checked, &effects_reannotated).is_none(),
        "independent checked programs must not compose"
    );

    let dimensional = checked_surf(
        r#"
type Column[n, a] = | Column(tensor[n, a])
def identity[n, a](column: Column[n, a]) -> Column[n, a] = column
"#,
    );
    let column = dimensional
        .adt_registry()
        .lookup("Column")
        .expect("checked registry must retain dimensional Column");
    let Type::Tensor(stored_dims, TensorPrec::Var(stored_precision)) =
        &column.variants[0].fields[0].1
    else {
        panic!("Column must retain its checker-owned polymorphic tensor field");
    };
    assert_eq!(
        column.param_kinds,
        vec![NominalParamKind::Dimension, NominalParamKind::Type],
        "the registry must preserve each source parameter's checker-owned kind"
    );
    let NominalArg::Dimension(Dim::Var(parameter_dimension)) = &column.param_args[0] else {
        panic!("Column's first parameter must retain its dimension variable");
    };
    let NominalArg::Type(Type::Var(parameter_precision)) = &column.param_args[1] else {
        panic!("Column's second parameter must retain its type variable");
    };
    let Some(Dim::Var(field_dimension)) = stored_dims.first() else {
        panic!("Column's field must retain its parameterized dimension");
    };
    assert_eq!(
        stored_precision, parameter_precision,
        "dtype parameter is the tensor's stored precision"
    );
    assert_eq!(
        field_dimension, parameter_dimension,
        "dimension parameter is the tensor's stored extent"
    );
    assert_eq!(
        column.param_vars,
        vec![*parameter_precision],
        "dimension parameters must not be reconstructed as stored dtypes"
    );
}

#[test]
fn generalized_unannotated_function_is_not_an_authored_signature() {
    let checked = checked_surf(
        r#"
def identity(x) = x
def use_int() = identity(1)
def use_bool() = identity(true)
"#,
    );
    let identity = checked
        .signature_inference()
        .functions
        .get("identity")
        .expect("generalized identity metadata");
    assert!(
        !identity.authored_signature,
        "body inference/generalization must not fabricate authored provenance"
    );
    assert!(
        identity.authored_signature_type.is_none(),
        "an unannotated function has no checker-decoded authored signature"
    );
    assert!(
        matches!(identity.checked_signature, Type::Fn(_, _)),
        "the function is still generalized and callable"
    );
}

/// Run the full Surf → desugar → infer pipeline and return the raw
/// `InferResult` (errors included) so a test can assert clean or assert
/// a specific failure mode end-to-end.
fn infer_surf(src: &str) -> InferResult {
    let decls = chelis_surf::parser::parse_str(src).expect("surf parse");
    let exprs = chelis_surf::desugar::desugar_program(&decls);
    infer_program(&exprs)
}

fn missing_shape_sensitive_app(expr: &deep::Expr) -> Option<String> {
    match expr {
        deep::Expr::List(list, _) => {
            if get_tag(list) == Some(DeepTag::App)
                && is_shape_sensitive_app(list)
                && !list
                    .elements
                    .get(1)
                    .and_then(|expr| match expr {
                        deep::Expr::Map(meta, _) => Some(meta),
                        _ => None,
                    })
                    .is_some_and(|meta| meta.ty().is_some())
            {
                return Some(
                    chelis_deep::printer::print_canonical(std::slice::from_ref(expr))
                        .replace('\n', " ")
                        .trim()
                        .to_string(),
                );
            }
            for child in &list.elements {
                if let Some(missing) = missing_shape_sensitive_app(child) {
                    return Some(missing);
                }
            }
            None
        }
        deep::Expr::Map(map, _) => map.find_expression(missing_shape_sensitive_app),
        deep::Expr::MetaExpr(meta, _) => missing_shape_sensitive_app(&meta.expr)
            .or_else(|| meta.metadata.find_expression(missing_shape_sensitive_app)),
        deep::Expr::Node(node, _) => {
            if let Some(missing) = node.meta().find_expression(missing_shape_sensitive_app) {
                return Some(missing);
            }
            for child in node.children_iter() {
                let child = match child {
                    chelis_deep::node::ChildRef::Expr(expr)
                    | chelis_deep::node::ChildRef::Syntax(expr)
                    | chelis_deep::node::ChildRef::Type(expr)
                    | chelis_deep::node::ChildRef::EffectHandler(expr)
                    | chelis_deep::node::ChildRef::Bypass(expr) => expr,
                    chelis_deep::node::ChildRef::Binder(_)
                    | chelis_deep::node::ChildRef::Selector(_) => continue,
                };
                if let Some(missing) = missing_shape_sensitive_app(child) {
                    return Some(missing);
                }
            }
            None
        }
        deep::Expr::BareList(elements, _) => elements.iter().find_map(missing_shape_sensitive_app),
        deep::Expr::UnknownForm(data) => data
            .meta
            .find_expression(missing_shape_sensitive_app)
            .or_else(|| data.children.iter().find_map(missing_shape_sensitive_app)),
        deep::Expr::Atom(_, _) => None,
    }
}

fn is_shape_sensitive_app(list: &deep::List) -> bool {
    get_tag(list) == Some(DeepTag::App)
        && ir_builtin_name(list).is_some_and(super::is_ir_shape_sensitive_builtin)
}

fn check_ok(src: &str) {
    let result = check(src);
    assert!(
        result.errors.is_empty(),
        "expected no errors, got: {:?}",
        result.errors
    );
}

fn check_err(src: &str, expected_kind: CheckErrorKind) {
    let result = check(src);
    assert!(
        !result.errors.is_empty(),
        "expected error {expected_kind:?}, got none"
    );
    assert!(
        result
            .errors
            .iter()
            .any(|e| std::mem::discriminant(&e.kind) == std::mem::discriminant(&expected_kind)),
        "expected {expected_kind:?}, got: {:?}",
        result.errors
    );
}

// ── WI-1 stack-budget guard (per-walker coverage) ────────────

/// Build a left-nested `app` chain of `depth` distinct names directly in
/// Deep, for the recursion-depth guard tests.
fn deep_app_chain_node(depth: usize) -> deep::Expr {
    let sym = |s: &str| deep::Expr::Atom(deep::Atom::Name(s.to_string()), Span::new(0, 0));
    let meta = || deep::Expr::Map(deep::Metadata::default(), Span::new(0, 0));
    let var = |n: &str| {
        deep::Expr::List(
            deep::List {
                elements: vec![sym("var"), meta(), sym(n)],
            },
            Span::new(0, 0),
        )
    };
    let mut e = var("f0");
    for i in 1..=depth {
        e = deep::Expr::List(
            deep::List {
                elements: vec![sym("app"), meta(), e, var(&format!("f{i}"))],
            },
            Span::new(0, 0),
        );
    }
    e
}

/// The sibling walker `walk_for_tensor_precision` is independently
/// guarded: run it (via `validate_tensor_precisions_in_program`) on a
/// chain deep enough to overflow an unguarded recursion, on a bounded
/// stack. gdb showed this walker -- not `infer_expr` -- is the SIGSEGV
/// site for this pass on deep input; reaching the assertion at all proves
/// the guard prevented the overflow, and the recorded `STACK_EXHAUSTED`
/// flag proves the bail surfaces.
#[test]
fn walk_for_tensor_precision_sibling_is_guarded_not_sigsegv() {
    // Built and dropped on the main thread: the chain's unguarded drop
    // glue overflows a 1 MiB stack on its own under rustc 1.97 codegen
    // (see `stack_exhaustion_drains_into_a_located_error`).
    let program = std::sync::Arc::new(vec![deep_app_chain_node(4000)]);
    let worker_program = std::sync::Arc::clone(&program);
    let flagged = std::thread::Builder::new()
        .name("sibling-guard-test".to_string())
        // 1 MiB: small enough that a 4000-deep chain trips the byte
        // budget early, well inside the guard, with no risk of overflow.
        .stack_size(1024 * 1024)
        .spawn(move || {
            let _scope = StackExhaustionScope::enter();
            let mut errors = Vec::new();
            // Drive ONLY the tensor-precision pass (whose deep walker is
            // walk_for_tensor_precision), isolating it from infer_expr.
            validate_tensor_precisions_in_program(&worker_program, &mut errors);
            STACK_EXHAUSTED.with(|cell| cell.borrow().clone())
        })
        .expect("spawn sibling-guard worker")
        .join()
        .expect("sibling walker aborted (stack overflow?) instead of returning");
    drop(program);

    let (site, _span) = flagged.expect(
        "walk_for_tensor_precision must record a stack-exhaustion bail on a \
         4000-deep chain run on a 1 MiB stack",
    );
    assert!(
        site.contains("walk_for_tensor_precision"),
        "the bail must be attributed to the precision walker, got site: {site}",
    );
}

/// The funnel is sound: a walker that bails with NO error vector (here the
/// tensor-precision pass is driven in isolation) still makes the flag set,
/// and `StackExhaustionScope::drain_into` turns it into a hard located
/// error -- never a silent empty result.
#[test]
fn stack_exhaustion_drains_into_a_located_error() {
    // The 4000-deep chain is BUILT and DROPPED on the test's main thread:
    // its derived drop glue recurses the full chain depth with no guard,
    // and under rustc 1.97 codegen those frames overflow the worker's
    // 1 MiB stack on their own (the pre-1.97 frames merely happened to
    // fit). The worker thread exists to exercise the GUARDED walker on a
    // small stack; the unguarded collateral must not share it.
    // (Production runs construction, walkers, and drops on the
    // `with_grown_stack` segment, so this is a test-harness concern.)
    let program = std::sync::Arc::new(vec![deep_app_chain_node(4000)]);
    let worker_program = std::sync::Arc::clone(&program);
    let errors = std::thread::Builder::new()
        .stack_size(1024 * 1024)
        .spawn(move || {
            let scope = StackExhaustionScope::enter();
            let mut errors = Vec::new();
            validate_tensor_precisions_in_program(&worker_program, &mut errors);
            // Before drain: the precision pass carries no error vector of
            // its own for the bail, so `errors` may be empty here ...
            scope.drain_into(&mut errors);
            // ... but after drain the exhaustion is a hard error.
            drop(worker_program);
            errors
        })
        .expect("spawn drain-test worker")
        .join()
        .expect("worker aborted instead of returning");
    drop(program);
    assert!(
        errors
            .iter()
            .any(|e| e.message.contains("stack budget exhausted")),
        "drain_into must surface a located stack-budget error; got: {:?}",
        errors.iter().map(|e| &e.message).collect::<Vec<_>>(),
    );
}

/// Negative: a moderate-depth chain does not trip the guard in the
/// precision pass either (no false positive on the sibling site).
#[test]
fn moderate_chain_does_not_trip_sibling_guard() {
    let flagged = std::thread::Builder::new()
        .stack_size(8 * 1024 * 1024)
        .spawn(|| {
            let program = vec![deep_app_chain_node(20)];
            let _scope = StackExhaustionScope::enter();
            let mut errors = Vec::new();
            validate_tensor_precisions_in_program(&program, &mut errors);
            STACK_EXHAUSTED.with(|cell| cell.borrow().is_some())
        })
        .expect("spawn worker")
        .join()
        .expect("worker aborted");
    assert!(
        !flagged,
        "a 20-deep chain must not trip the precision walker's stack guard",
    );
}

// ── Variable / literal tests ─────────────────────────────────

#[test]
fn lit_int32() {
    check_ok("(def {} x (lit {type: (t-prim {} i32)} 42))");
}

#[test]
fn lit_f32() {
    check_ok("(def {} x (lit {type: (t-prim {} f32)} 3.0))");
}

#[test]
fn lit_bool() {
    check_ok("(def {} x (lit {type: (t-prim {} bool)} true))");
}

#[test]
fn lit_string() {
    check_ok(r#"(def {} x (lit {type: (t-prim {} string)} "hello"))"#);
}

#[test]
fn scalar_add_is_allowed() {
    check_ok(
        "(def {} x
            (app {} (var {} add)
                (lit {type: (t-prim {} i64)} 2)
                (lit {type: (t-prim {} i64)} 3)))",
    );
}

#[test]
fn integer_mod_and_bitwise_builtins_are_allowed() {
    check_ok(
        "(def {} bits
            (app {} (var {} bitxor)
                (app {} (var {} bitand)
                    (lit {type: (t-prim {} i64)} 7)
                    (lit {type: (t-prim {} i64)} 3))
                (app {} (var {} shl)
                    (lit {type: (t-prim {} i64)} 1)
                    (lit {type: (t-prim {} i64)} 2))))
         (def {} rem
            (app {} (var {} mod)
                (lit {type: (t-prim {} i64)} 17)
                (lit {type: (t-prim {} i64)} 5)))
         (def {} shrunk
            (app {} (var {} shr)
                (lit {type: (t-prim {} i64)} 8)
                (lit {type: (t-prim {} i64)} 1)))",
    );
}

#[test]
fn string_len_builtin_is_allowed() {
    check_ok(
        r#"(def {} x
            (app {} (var {} string_len)
                (lit {type: (t-prim {} string)} "hé")))"#,
    );
}

#[test]
fn string_predicates_and_transforms_are_allowed() {
    check_ok(
        r#"(def {} ok
            (if {}
                (app {} (var {} and)
                    (app {} (var {} string_contains)
                        (lit {type: (t-prim {} string)} "ckpt-7.safetensors")
                        (lit {type: (t-prim {} string)} "ckpt"))
                    (app {} (var {} string_ends_with)
                        (app {} (var {} string_slice)
                            (app {} (var {} string_trim)
                                (lit {type: (t-prim {} string)} "  ckpt-7.safetensors  "))
                            (lit {type: (t-prim {} i64)} 7)
                            (lit {type: (t-prim {} i64)} 12))
                        (lit {type: (t-prim {} string)} ".safetensors")))
                (lit {type: (t-prim {} bool)} true)
                (lit {type: (t-prim {} bool)} false)))"#,
    );
}

#[test]
fn to_int_builtin_uses_prelude_option_without_local_deftype() {
    check_ok(
        r#"(def {} parsed
            (match {}
                (app {} (var {} to_int)
                    (lit {type: (t-prim {} string)} "42"))
                (arm {} (pat-ctor {} Some (pat-var {} n)) () (var {} n))
                (arm {} (pat-ctor {} None) () (lit {type: (t-prim {} i64)} 0))))"#,
    );
}

#[test]
fn to_float_builtin_uses_prelude_option_without_local_deftype() {
    check_ok(
        r#"(def {} parsed
            (match {}
                (app {} (var {} to_float)
                    (lit {type: (t-prim {} string)} "0.125"))
                (arm {} (pat-ctor {} Some (pat-var {} x)) () (var {} x))
                (arm {} (pat-ctor {} None) () (lit {type: (t-prim {} f64)} 1.0))))"#,
    );
}

#[test]
fn to_int_builtin_rejects_non_string_input() {
    check_err(
        r#"(def {} parsed
            (app {} (var {} to_int)
                (lit {type: (t-prim {} i32)} 7)))"#,
        CheckErrorKind::TypeMismatch,
    );
}

#[test]
fn mod_rejects_float_input() {
    check_err(
        r#"(def {} bad
            (app {} (var {} mod)
                (lit {type: (t-prim {} f64)} 7.0)
                (lit {type: (t-prim {} f64)} 3.0)))"#,
        CheckErrorKind::TypeMismatch,
    );
}

#[test]
fn bitand_rejects_mismatched_integer_widths() {
    check_err(
        r#"(def {} bad
            (app {} (var {} bitand)
                (lit {type: (t-prim {} i32)} 7)
                (lit {type: (t-prim {} i64)} 3)))"#,
        CheckErrorKind::TypeMismatch,
    );
}

#[test]
fn shl_rejects_non_integer_shift_amount() {
    check_err(
        r#"(def {} bad
            (app {} (var {} shl)
                (lit {type: (t-prim {} i64)} 1)
                (lit {type: (t-prim {} f64)} 2.0)))"#,
        CheckErrorKind::TypeMismatch,
    );
}

#[test]
fn string_slice_rejects_non_string_input() {
    check_err(
        r#"(def {} bad
            (app {} (var {} string_slice)
                (lit {type: (t-prim {} i32)} 7)
                (lit {type: (t-prim {} i64)} 0)
                (lit {type: (t-prim {} i64)} 1)))"#,
        CheckErrorKind::TypeMismatch,
    );
}

#[test]
fn rank_builtin_requires_tensor_input() {
    check_err(
        "(def {} x
            (app {} (var {} rank)
                (lit {type: (t-prim {} i64)} 2)))",
        CheckErrorKind::TypeMismatch,
    );
}

#[test]
fn shape_builtin_accepts_tensor_input() {
    check_ok(
        r#"(def {} x
            (lit {type: (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))} 0))
           (def {} dim
            (app {} (var {} shape)
                (var {type: (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))} x)
                (lit {type: (t-prim {} i32)} 1)))"#,
    );
}

#[test]
fn shape_builtin_rejects_negative_axis_when_rank_is_known() {
    check_err(
        r#"(def {} x
            (lit {type: (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))} 0))
           (def {} dim
            (app {} (var {} shape)
                (var {type: (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))} x)
                (lit {type: (t-prim {} i32)} -1)))"#,
        CheckErrorKind::DimensionMismatch,
    );
}

#[test]
fn lit_default_int() {
    check_ok("(def {} x (lit {} 42))");
}

#[test]
fn lit_default_float() {
    check_ok("(def {} x (lit {} 3.14))");
}

#[test]
fn unbound_variable() {
    check_err(
        "(def {} x (var {} unknown))",
        CheckErrorKind::UnboundVariable {
            identifier: "unknown".to_string(),
        },
    );
}

#[test]
fn var_lookup_defined() {
    check_ok(
        "(def {} x (lit {type: (t-prim {} i32)} 42))
         (def {} y (var {} x))",
    );
}

// ── Application tests ────────────────────────────────────────

#[test]
fn app_add_tensors() {
    check_ok(
        "(def {} a (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0))
         (def {} b (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0))
         (def {} c (app {} (var {} add) (var {} a) (var {} b)))",
    );
}

#[test]
fn app_precision_mismatch() {
    check_err(
        "(def {} a (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0))
         (def {} b (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} bf16))} 0))
         (def {} c (app {} (var {} add) (var {} a) (var {} b)))",
        CheckErrorKind::PrecisionMismatch,
    );
}

#[test]
fn app_not_a_function() {
    check_err(
        "(def {} x (lit {type: (t-prim {} i32)} 42))
         (def {} y (app {} (var {} x) (lit {type: (t-prim {} i32)} 1)))",
        CheckErrorKind::TypeMismatch,
    );
}

#[test]
fn app_dimension_mismatch() {
    check_err(
        "(def {} a (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0))
         (def {} b (lit {type: (t-tensor {} (d-name {} seq) (t-prim {} f32))} 0))
         (def {} c (app {} (var {} add) (var {} a) (var {} b)))",
        CheckErrorKind::DimensionMismatch,
    );
}

// ── Lambda tests ─────────────────────────────────────────────

#[test]
fn fn_identity() {
    check_ok("(def {} id (fn {} (params {} x) (var {} x)))");
}

#[test]
fn fn_two_params() {
    check_ok("(def {} f (fn {} (params {} x y) (var {} x)))");
}

#[test]
fn fn_with_body_app() {
    check_ok(
        "(def {} f (fn {} (params {} (x {type: (t-prim {} i32)}) (y {type: (t-prim {} i32)})) (app {} (var {} add) (var {} x) (var {} y))))",
    );
    check_err(
        "(def {} f (fn {} (params {} x y) (app {} (var {} add) (var {} x) (var {} y))))",
        CheckErrorKind::PrecisionMismatch,
    );
}

// ── Let tests ────────────────────────────────────────────────

#[test]
fn let_simple() {
    check_ok(
        "(def {} result
           (let {} (bind {} x (lit {type: (t-prim {} i32)} 42))
             (var {} x)))",
    );
}

#[test]
fn let_multiple_bindings() {
    check_ok(
        "(def {} result
           (let {} (bind {} x (lit {type: (t-prim {} i32)} 1) y (lit {type: (t-prim {} f32)} 2.0))
             (var {} x)))",
    );
}

#[test]
fn let_scoping() {
    // Variable defined in let should be usable in body
    check_ok(
        "(def {} result
           (let {} (bind {} x (lit {type: (t-prim {} i32)} 42))
             (var {} x)))",
    );
}

// ── If tests ─────────────────────────────────────────────────

#[test]
fn if_correct() {
    check_ok(
        "(def {} x (lit {type: (t-prim {} bool)} true))
         (def {} a (lit {type: (t-prim {} i32)} 1))
         (def {} b (lit {type: (t-prim {} i32)} 2))
         (def {} c (if {} (var {} x) (var {} a) (var {} b)))",
    );
}

#[test]
fn if_branch_mismatch() {
    check_err(
        "(def {} x (lit {type: (t-prim {} bool)} true))
         (def {} a (lit {type: (t-prim {} i32)} 1))
         (def {} b (lit {type: (t-prim {} f32)} 2.0))
         (def {} c (if {} (var {} x) (var {} a) (var {} b)))",
        CheckErrorKind::PrecisionMismatch,
    );
}

// ── Pipe tests ───────────────────────────────────────────────

#[test]
fn pipe_simple() {
    check_ok(
        "(defsig {} f (t-fn {} (t-prim {} f32) (t-prim {} f32)))
         (def {} f (fn {} (params {} x) (var {} x)))
         (def {} result (pipe {} (lit {type: (t-prim {} f32)} 1.0) (var {} f)))",
    );
}

#[test]
fn pipe_chain() {
    check_ok(
        "(defsig {} f (t-fn {} (t-prim {} f32) (t-prim {} f32)))
         (def {} f (fn {} (params {} x) (var {} x)))
         (def {} result (pipe {} (lit {type: (t-prim {} f32)} 1.0) (var {} f) (var {} f)))",
    );
}

// ── Tuple tests ──────────────────────────────────────────────

#[test]
fn tuple_creation() {
    check_ok(
        "(def {} t (tuple {} (lit {type: (t-prim {} i32)} 1) (lit {type: (t-prim {} f32)} 2.0)))",
    );
}

#[test]
fn tuple_get_valid() {
    check_ok(
        "(def {} t (tuple {} (lit {type: (t-prim {} i32)} 1) (lit {type: (t-prim {} f32)} 2.0)))
         (def {} x (tuple-get {} (var {} t) 0))",
    );
}

#[test]
fn tuple_get_out_of_bounds() {
    check_err(
        "(def {} t (tuple {} (lit {type: (t-prim {} i32)} 1)))
         (def {} x (tuple-get {} (var {} t) 5))",
        CheckErrorKind::TupleIndexOutOfBounds,
    );
}

#[test]
fn tuple_get_lit_node_index_valid() {
    // chelis#707: the Surf `.N` desugar emits the index as a `lit`
    // node `(lit {i32} N)`, not a bare `Int` atom. The bare-atom
    // form above always worked; this is the untested seam that made
    // every Surf-level projection type as `Type::Error`.
    check_ok(
        "(def {} t (tuple {} (lit {type: (t-prim {} i32)} 1) (lit {type: (t-prim {} f32)} 2.0)))
         (def {} x (tuple-get {} (var {} t) (lit {type: (t-prim {} i32)} 0)))",
    );
}

#[test]
fn tuple_get_lit_node_index_out_of_bounds() {
    // Negative parity for the seam: a `lit`-node index must reach the
    // bounds check. Before chelis#707 the index never parsed, so this
    // returned `Type::Error` with NO diagnostic (a swallowed reject);
    // now it fires `TupleIndexOutOfBounds`.
    check_err(
        "(def {} t (tuple {} (lit {type: (t-prim {} i32)} 1)))
         (def {} x (tuple-get {} (var {} t) (lit {type: (t-prim {} i32)} 5)))",
        CheckErrorKind::TupleIndexOutOfBounds,
    );
}

#[test]
fn tuple_get_index_reads_bare_atom_and_lit_node() {
    // Bare `Int` atom (hand-written Deep).
    let bare = deep::Expr::Atom(deep::Atom::Int(2), zero_span());
    assert_eq!(tuple_get_index(&bare), Some(2));
    // `lit` node wrapping an `Int` atom (the Surf `.N` desugar).
    let lit = node_expr(
        DeepTag::Lit,
        vec![deep::Expr::Atom(deep::Atom::Int(2), zero_span())],
    );
    assert_eq!(tuple_get_index(&lit), Some(2));
    // A negative literal is not a valid index.
    let negative = deep::Expr::Atom(deep::Atom::Int(-1), zero_span());
    assert_eq!(tuple_get_index(&negative), None);
    // A non-`Int` payload (symbol) is not an index.
    let symbolic = node_expr(DeepTag::Lit, vec![symbol_expr("nope")]);
    assert_eq!(tuple_get_index(&symbolic), None);
    // A non-`lit` list tag is not an index.
    let other = node_expr(DeepTag::Var, vec![symbol_expr("t")]);
    assert_eq!(tuple_get_index(&other), None);
}

// ── Cast tests ───────────────────────────────────────────────

#[test]
fn cast_tensor() {
    // Cast to a precision the Phase 0f tensor backend supports.
    // Reduced-float targets (bf16/f16/f64/f8e4m3) are rejected — see
    // `cast_tensor_rejects_unsupported_precision` below.
    check_ok(
        "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0))
         (def {} y (cast {} (var {} x) (t-prim {} i32)))",
    );
}

#[test]
fn cast_prim() {
    check_ok(
        "(def {} x (lit {type: (t-prim {} i32)} 42))
         (def {} y (cast {} (var {} x) (t-prim {} f32)))",
    );
}

#[test]
fn cast_accepts_alias_precision() {
    check_ok(
        "(typealias {} Floaty () (t-prim {} f32))
         (def {} x (lit {type: (t-prim {} i32)} 42))
         (def {} y (cast {} (var {} x) (t-adt {} Floaty)))",
    );
}

// ── Unsupported tensor precision tests ──────────────────────

#[test]
fn tensor_ascription_accepts_f64() {
    // v0.2.3: f64 tensors are now a first-class precision. The checker
    // accepts tensor[..., f64]; the C backend emits `double` arrays.
    check_ok("(def {} y (lit {type: (t-tensor {} (d-lit {} 4) (t-prim {} f64))} 0))");
}

#[test]
fn tensor_ascription_accepts_f16() {
    // WS-0 lock cc47e6d: f16 is in the active dtype set per
    // spec/04-type-system.md §1.1 and must be admitted as a tensor
    // element type at check time. Backend coverage is staged
    // separately (WS-A1/A2/A3).
    check_ok("(def {} y (lit {type: (t-tensor {} (d-lit {} 4) (t-prim {} f16))} 0))");
}

#[test]
fn tensor_ascription_accepts_bf16() {
    // WS-0 lock cc47e6d: bf16 is in the active dtype set per
    // spec/04-type-system.md §1.1 and must be admitted as a tensor
    // element type at check time. Backend coverage is staged
    // separately (WS-A1/A2/A3).
    check_ok("(def {} y (lit {type: (t-tensor {} (d-lit {} 4) (t-prim {} bf16))} 0))");
}

#[test]
fn tensor_ascription_rejects_f8e4m3() {
    // f8e4m3 is deferred per spec/04-type-system.md §1.1.1 and must
    // be rejected at check time.
    check_err(
        "(def {} y (lit {type: (t-tensor {} (d-lit {} 4) (t-prim {} f8e4m3))} 0))",
        CheckErrorKind::UnsupportedTensorPrecision,
    );
}

#[test]
fn tensor_ascription_accepts_int64() {
    // Integer tensor precisions remain valid.
    check_ok("(def {} y (lit {type: (t-tensor {} (d-lit {} 4) (t-prim {} i64))} 0))");
}

#[test]
fn tensor_ascription_accepts_bool() {
    check_ok("(def {} y (lit {type: (t-tensor {} (d-lit {} 4) (t-prim {} bool))} false))");
}

#[test]
fn cast_tensor_accepts_f64() {
    // v0.2.3: tensor-f64 is a valid cast target. The checker accepts
    // cast(tensor_f32, f64); the backend emits float→double conversion.
    check_ok(
        "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0))
         (def {} y (cast {} (var {} x) (t-prim {} f64)))",
    );
}

#[test]
fn cast_tensor_accepts_bf16() {
    // WS-0 lock cc47e6d: cast(x: tensor[..., f32], bf16) is permitted
    // because bf16 is in the active tensor element set per
    // spec/04-type-system.md §1.1.
    check_ok(
        "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0))
         (def {} y (cast {} (var {} x) (t-prim {} bf16)))",
    );
}

#[test]
fn cast_tensor_rejects_f8e4m3() {
    // f8e4m3 is deferred per spec/04-type-system.md §1.1.1; cast
    // targets must be rejected with the deferred-dtype diagnostic.
    check_err(
        "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0))
         (def {} y (cast {} (var {} x) (t-prim {} f8e4m3)))",
        CheckErrorKind::UnsupportedTensorPrecision,
    );
}

#[test]
fn cast_prim_to_f64_is_allowed() {
    // Host scalar f64 is still valid — only tensor-precision f64 is banned.
    check_ok(
        "(def {} x (lit {type: (t-prim {} i32)} 42))
         (def {} y (cast {} (var {} x) (t-prim {} f64)))",
    );
}

#[test]
fn surf_source_accepts_f64_tensor_ascription() {
    // v0.2.3: exercise the full surf → desugar → check pipeline to confirm
    // the user-facing syntax `(expr : tensor[4, f64])` is accepted.
    let decls =
        chelis_surf::parser::parse_str("y = (to_tensor([1.0, 2.0, 3.0, 4.0]) : tensor[4, f64])")
            .expect("surf parse");
    let exprs = chelis_surf::desugar::desugar_program(&decls);
    let result = infer_program(&exprs);
    assert!(
        !result
            .errors
            .iter()
            .any(|e| matches!(e.kind, CheckErrorKind::UnsupportedTensorPrecision)),
        "expected no UnsupportedTensorPrecision for f64 tensor ascription, got: {:?}",
        result.errors.iter().map(|e| &e.kind).collect::<Vec<_>>()
    );
}

#[test]
fn surf_source_accepts_cast_to_f64_tensor() {
    // v0.2.3: exercise the full pipeline for `cast(tensor, f64)`.
    let decls =
        chelis_surf::parser::parse_str("y = cast(to_tensor([1.5]), f64)").expect("surf parse");
    let exprs = chelis_surf::desugar::desugar_program(&decls);
    let result = infer_program(&exprs);
    assert!(
        !result
            .errors
            .iter()
            .any(|e| matches!(e.kind, CheckErrorKind::UnsupportedTensorPrecision)),
        "expected no UnsupportedTensorPrecision for cast(tensor, f64), got: {:?}",
        result.errors.iter().map(|e| &e.kind).collect::<Vec<_>>()
    );
}

#[test]
fn copy_accepts_tensor() {
    check_ok(
        "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0))
         (def {} y (copy {} (var {} x)))",
    );
}

#[test]
fn copy_rejects_scalar() {
    check_err(
        "(def {} x (lit {type: (t-prim {} i32)} 42))
         (def {} y (copy {} (var {} x)))",
        CheckErrorKind::TypeMismatch,
    );
}

#[test]
fn copy_rejects_unconstrained_generic() {
    check_err(
        "(def {} id (fn {} (params {} x) (copy {} (var {} x))))",
        CheckErrorKind::TypeMismatch,
    );
}

// ── Grad tests ───────────────────────────────────────────────

#[test]
fn grad_function() {
    check_ok(
        "(defsig {} loss (t-fn {} (t-prim {} f32) (t-prim {} f32)))
         (def {} loss (fn {} (params {} x) (var {} x)))
         (def {} g (grad {} (var {} loss)))",
    );
}

#[test]
fn grad_non_float_param_is_ignored_by_default() {
    let exprs = chelis_deep::parser::parse_str(
        "(defsig {} f (t-fn {} (t-prim {} bool) (t-prim {} f32)))
         (def {} f (fn {} (params {} x) (lit {type: (t-prim {} f32)} 1.0)))
         (def {} g (grad {} (var {} f)))",
    )
    .unwrap();
    let checked = check_ir_program(&exprs).expect("IR check");
    let ty = checked.type_env().get("g").expect("g type");
    assert_eq!(
        chelis_deep::printer::print_canonical_flat(std::slice::from_ref(ty))
            .trim()
            .to_string(),
        "(t-fn {} (t-prim {} bool) (t-unit {}))"
    );
}

#[test]
fn grad_over_all_float_field_adt_types_as_same_adt() {
    // chelis#520 D2 slice: an ADT whose fields are all float tensors
    // gets a field-wise gradient of the same constructor shape, so
    // `grad(f : Box -> f32) : Box -> Box`.
    let exprs = chelis_deep::parser::parse_str(
        "(deftype {} Box ()
            (variant {} Box
                (field {} t (t-tensor {} (d-lit {} 2) (t-prim {} f32)))))
         (defsig {} f (t-fn {} (t-adt {} Box) (t-prim {} f32)))
         (def {} f (fn {} (params {} p) (lit {type: (t-prim {} f32)} 1.0)))
         (def {} g (grad {} (var {} f)))",
    )
    .unwrap();
    let checked = check_ir_program(&exprs).expect("IR check");
    let ty = checked.type_env().get("g").expect("g type");
    let printed = chelis_deep::printer::print_canonical_flat(std::slice::from_ref(ty))
        .trim()
        .to_string();
    assert_eq!(printed, "(t-fn {} (t-adt {} Box) (t-adt {} Box))");
}

#[test]
fn grad_over_mixed_field_adt_preserves_the_nominal_cotangent_shape() {
    // spec/06 section 2.1: a mixed struct is differentiable when any
    // reachable field contains a float leaf. The checker preserves the
    // nominal constructor type; execution replaces the discrete field's
    // cotangent with unit in its original position. The adjacent pure-enum
    // and explicit-bool-wrt tests retain negative parity for all-unit targets.
    let exprs = chelis_deep::parser::parse_str(
        "(deftype {} Mixed ()
            (variant {} Mixed
                (field {} t (t-tensor {} (d-lit {} 2) (t-prim {} f32)))
                (field {} n (t-prim {} i32))))
         (defsig {} f (t-fn {} (t-adt {} Mixed) (t-prim {} f32)))
         (def {} f (fn {} (params {} p) (lit {type: (t-prim {} f32)} 1.0)))
         (def {} g (grad {} (var {} f)))",
    )
    .unwrap();
    let checked = check_ir_program(&exprs).expect("IR check");
    let ty = checked.type_env().get("g").expect("g type");
    let printed = chelis_deep::printer::print_canonical_flat(std::slice::from_ref(ty))
        .trim()
        .to_string();
    assert_eq!(printed, "(t-fn {} (t-adt {} Mixed) (t-adt {} Mixed))");
}

#[test]
fn grad_over_adt_plus_tensor_multi_target_returns_tuple() {
    // chelis#520 D2: an ADT gradient target is supported ALONGSIDE a
    // plain tensor argument (the closing bar). The default (no `wrt`)
    // gradient payload is the per-target tuple whose ADT slot is the
    // field-wise gradient struct and whose tensor slot is the bare
    // tensor gradient.
    let exprs = chelis_deep::parser::parse_str(
        "(deftype {} Box ()
            (variant {} Box
                (field {} t (t-tensor {} (d-lit {} 2) (t-prim {} f32)))))
         (defsig {} f (t-fn {}
            (t-adt {} Box)
            (t-tensor {} (d-lit {} 2) (t-prim {} f32))
            (t-prim {} f32)))
         (def {} f (fn {} (params {} p y) (lit {type: (t-prim {} f32)} 1.0)))
         (def {} g (grad {} (var {} f)))",
    )
    .unwrap();
    let checked = check_ir_program(&exprs).expect("IR check");
    let ty = checked.type_env().get("g").expect("g type");
    let printed = chelis_deep::printer::print_canonical_flat(std::slice::from_ref(ty))
        .trim()
        .to_string();
    assert_eq!(
        printed,
        "(t-fn {} (t-adt {} Box) (t-tensor {} (d-lit {} 2) (t-prim {} f32)) (t-tuple {} (t-adt {} Box) (t-tensor {} (d-lit {} 2) (t-prim {} f32))))"
    );
}

#[test]
fn grad_wrt_adt_in_multi_arg_call_returns_struct() {
    // chelis#520 D2: a `wrt`-restricted ADT target inside a
    // multi-argument function is supported; narrowing to the ADT alone
    // yields the bare field-wise gradient struct (a single target, so
    // no enclosing tuple).
    let exprs = chelis_deep::parser::parse_str(
        "(deftype {} Box ()
            (variant {} Box
                (field {} t (t-tensor {} (d-lit {} 2) (t-prim {} f32)))))
         (defsig {} f (t-fn {}
            (t-adt {} Box)
            (t-tensor {} (d-lit {} 2) (t-prim {} f32))
            (t-prim {} f32)))
         (def {} f (fn {} (params {} p y) (lit {type: (t-prim {} f32)} 1.0)))
         (def {} g (grad {} (var {} f) (lit {type: (t-prim {} i32)} 0)))",
    )
    .unwrap();
    let checked = check_ir_program(&exprs).expect("IR check");
    let ty = checked.type_env().get("g").expect("g type");
    let printed = chelis_deep::printer::print_canonical_flat(std::slice::from_ref(ty))
        .trim()
        .to_string();
    assert_eq!(
        printed,
        "(t-fn {} (t-adt {} Box) (t-tensor {} (d-lit {} 2) (t-prim {} f32)) (t-adt {} Box))"
    );
}

#[test]
fn grad_over_pure_enum_stays_unit() {
    // chelis#520 D2: a pure enum (no fields in any variant) carries
    // no continuous payload, so it stays non-differentiable and the
    // default gradient payload is unit, the pre-#520 typing.
    let exprs = chelis_deep::parser::parse_str(
        "(deftype {} Mode ()
            (variant {} ModeA)
            (variant {} ModeB))
         (defsig {} f (t-fn {} (t-adt {} Mode) (t-prim {} f32)))
         (def {} f (fn {} (params {} m) (lit {type: (t-prim {} f32)} 1.0)))
         (def {} g (grad {} (var {} f)))",
    )
    .unwrap();
    let checked = check_ir_program(&exprs).expect("IR check");
    let ty = checked.type_env().get("g").expect("g type");
    let printed = chelis_deep::printer::print_canonical_flat(std::slice::from_ref(ty))
        .trim()
        .to_string();
    assert_eq!(printed, "(t-fn {} (t-adt {} Mode) (t-unit {}))");
}

#[test]
fn grad_over_enum_plus_tensor_keeps_tensor_only_payload() {
    // chelis#520 D2 regression guard: a nullary-enum parameter next
    // to a tensor parameter must not trip the single-argument ADT
    // rejection; the enum is non-differentiable (skipped), so the
    // gradient payload is the tensor alone, the pre-#520 typing.
    let exprs = chelis_deep::parser::parse_str(
        "(deftype {} Mode ()
            (variant {} ModeA)
            (variant {} ModeB))
         (defsig {} f (t-fn {}
            (t-adt {} Mode)
            (t-tensor {} (d-lit {} 2) (t-prim {} f32))
            (t-prim {} f32)))
         (def {} f (fn {} (params {} m x) (lit {type: (t-prim {} f32)} 1.0)))
         (def {} g (grad {} (var {} f)))",
    )
    .unwrap();
    let checked = check_ir_program(&exprs).expect("IR check");
    let ty = checked.type_env().get("g").expect("g type");
    let printed = chelis_deep::printer::print_canonical_flat(std::slice::from_ref(ty))
        .trim()
        .to_string();
    assert_eq!(
        printed,
        "(t-fn {} (t-adt {} Mode) (t-tensor {} (d-lit {} 2) (t-prim {} f32)) (t-tensor {} (d-lit {} 2) (t-prim {} f32)))"
    );
}

#[test]
fn grad_rejects_non_scalar_output() {
    check_err(
        "(defsig {} f (t-fn {}
            (t-prim {} f32)
            (t-tensor {} (d-lit {} 2) (t-prim {} f32))))
         (def {} f
            (fn {} (params {} x)
              (lit {type: (t-tensor {} (d-lit {} 2) (t-prim {} f32))} 1.0)))
         (def {} g (grad {} (var {} f)))",
        CheckErrorKind::Other,
    );
}

#[test]
fn grad_with_explicit_wrt_returns_selected_gradient_only() {
    let exprs = chelis_deep::parser::parse_str(
        "(defsig {} loss
            (t-fn {}
                (t-tensor {} (d-name {} features) (t-prim {} f32))
                (t-tensor {} (d-name {} hidden) (t-prim {} f32))
                (t-prim {} f32)))
         (def {} loss
            (fn {} (params {} x w)
                (lit {type: (t-prim {} f32)} 1.0)))
         (def {} dw
            (grad {} (var {} loss) (lit {type: (t-prim {} i32)} 1)))",
    )
    .unwrap();
    let checked = check_ir_program(&exprs).expect("IR check");
    let ty = checked.type_env().get("dw").expect("dw type");
    assert_eq!(
        chelis_deep::printer::print_canonical_flat(std::slice::from_ref(ty))
            .trim()
            .to_string(),
        "(t-fn {} (t-tensor {} (d-name {} features) (t-prim {} f32)) (t-tensor {} (d-name {} hidden) (t-prim {} f32)) (t-tensor {} (d-name {} hidden) (t-prim {} f32)))"
    );
}

#[test]
fn grad_with_multiple_wrt_returns_flat_tuple() {
    let exprs = chelis_deep::parser::parse_str(
        "(defsig {} loss
            (t-fn {}
                (t-tensor {} (d-name {} features) (t-prim {} f32))
                (t-tensor {} (d-name {} hidden) (t-prim {} f32))
                (t-prim {} f32)))
         (def {} loss
            (fn {} (params {} x w)
                (lit {type: (t-prim {} f32)} 1.0)))
         (def {} grads
            (grad {} (var {} loss)
                (tuple {}
                    (lit {type: (t-prim {} i32)} 0)
                    (lit {type: (t-prim {} i32)} 1))))",
    )
    .unwrap();
    let checked = check_ir_program(&exprs).expect("IR check");
    let ty = checked.type_env().get("grads").expect("grads type");
    assert_eq!(
        chelis_deep::printer::print_canonical_flat(std::slice::from_ref(ty))
            .trim()
            .to_string(),
        "(t-fn {} (t-tensor {} (d-name {} features) (t-prim {} f32)) (t-tensor {} (d-name {} hidden) (t-prim {} f32)) (t-tuple {} (t-tensor {} (d-name {} features) (t-prim {} f32)) (t-tensor {} (d-name {} hidden) (t-prim {} f32))))"
    );
}

#[test]
fn grad_rejects_nondifferentiable_explicit_wrt_target() {
    check_err(
        "(defsig {} f (t-fn {} (t-prim {} bool) (t-prim {} f32)))
         (def {} f (fn {} (params {} x) (lit {type: (t-prim {} f32)} 1.0)))
         (def {} g (grad {} (var {} f) (lit {type: (t-prim {} i32)} 0)))",
        CheckErrorKind::TypeMismatch,
    );
}

#[test]
fn vmap_function_inserts_axis_zero_batch_dim() {
    check_ok(
        "(defsig {} process
            (t-fn {}
                (t-tensor {} (d-name {} features) (t-prim {} f32))
                (t-tensor {} (d-name {} features) (t-prim {} f32))))
         (def {} process
            (fn {} (params {} x)
                (var {} x)))
         (defsig {} batch_process
            (t-fn {}
                (t-tensor {} (d-name {} batch) (d-name {} features) (t-prim {} f32))
                (t-tensor {} (d-name {} batch) (d-name {} features) (t-prim {} f32))))
         (def {} batch_process
            (vmap {} (var {} process) (lit {type: (t-prim {} i32)} 0)))",
    );
}

#[test]
fn vmap_non_function_is_rejected() {
    check_err(
        "(def {} x (lit {type: (t-prim {} f32)} 1.0))
         (def {} y (vmap {} (var {} x) (lit {type: (t-prim {} i32)} 0)))",
        CheckErrorKind::TypeMismatch,
    );
}

#[test]
fn vmap_axis_out_of_bounds_is_rejected() {
    check_err(
        "(defsig {} process
            (t-fn {}
                (t-tensor {} (d-name {} features) (t-prim {} f32))
                (t-tensor {} (d-name {} features) (t-prim {} f32))))
         (def {} process
            (fn {} (params {} x)
                (var {} x)))
         (def {} batch_process
            (vmap {} (var {} process) (lit {type: (t-prim {} i32)} 2)))",
        CheckErrorKind::DimensionMismatch,
    );
}

#[test]
fn vmap_grad_single_tensor_param_type_checks() {
    // WS-A7: pre-fix, the bare-arg `x` in `def loss(x) = sum(x, 0)` was
    // never seeded with the declared sig type (`tensor[features, f32]`)
    // before body inference, so `sum`'s reduction-signature check
    // early-returned an unconstrained result tvar (the input was still a
    // free `Var(_)`). The post-body sig-unify could then pin that
    // unconstrained tvar to the declared `f32` even though `sum` on a
    // rank-1 tensor produces `tensor[, f32]` per the spec — masking the
    // missing `tensor_to_scalar` coercion. Seeding bare params with the
    // declared sig types now exposes the rank-0 result, so the fixture
    // wraps the reduction in `tensor_to_scalar` to match the declared
    // scalar return.
    check_ok(
        "(defsig {} loss
            (t-fn {}
                (t-tensor {} (d-name {} features) (t-prim {} f32))
                (t-prim {} f32)))
         (def {} loss
            (fn {} (params {} x)
                (app {type: (t-prim {} f32)} (var {} tensor_to_scalar)
                    (app {type: (t-tensor {} (t-prim {} f32))} (var {} sum)
                        (var {type: (t-tensor {} (d-name {} features) (t-prim {} f32))} x)
                        (lit {type: (t-prim {} i32)} 0))))
         )
         (defsig {} per_example_grad
            (t-fn {}
                (t-tensor {} (d-name {} batch) (d-name {} features) (t-prim {} f32))
                (t-tensor {} (d-name {} batch) (d-name {} features) (t-prim {} f32))))
         (def {} per_example_grad
            (vmap {} (grad {} (var {} loss)) (lit {type: (t-prim {} i32)} 0)))",
    );
}

#[test]
fn vmap_grad_multiple_params_type_checks_with_tuple_result() {
    // WS-A7: see vmap_grad_single_tensor_param_type_checks — the same
    // rank-0 vs `Prim(f32)` distinction applies; wrap `sum` in
    // `tensor_to_scalar` to match the declared scalar return.
    check_ok(
        "(defsig {} loss
            (t-fn {}
                (t-tensor {} (d-name {} features) (t-prim {} f32))
                (t-tensor {} (d-name {} features) (t-prim {} f32))
                (t-prim {} f32)))
         (def {} loss
            (fn {} (params {} x y)
                (app {type: (t-prim {} f32)} (var {} tensor_to_scalar)
                    (app {type: (t-tensor {} (t-prim {} f32))} (var {} sum)
                        (app {type: (t-tensor {} (d-name {} features) (t-prim {} f32))} (var {} add)
                            (var {type: (t-tensor {} (d-name {} features) (t-prim {} f32))} x)
                            (var {type: (t-tensor {} (d-name {} features) (t-prim {} f32))} y))
                        (lit {type: (t-prim {} i32)} 0))))
         )
         (def {} per_example_grad
            (vmap {} (grad {} (var {} loss)) (lit {type: (t-prim {} i32)} 0)))",
    );
}

#[test]
fn grad_of_vmap_is_rejected_for_non_scalar_output() {
    check_err(
        "(defsig {} loss
            (t-fn {}
                (t-tensor {} (d-name {} features) (t-prim {} f32))
                (t-prim {} f32)))
         (def {} loss
            (fn {} (params {} x)
                (lit {type: (t-prim {} f32)} 1.0)))
         (def {} g
            (grad {} (vmap {} (var {} loss) (lit {type: (t-prim {} i32)} 0))))",
        CheckErrorKind::Other,
    );
}

// ── ADT tests ────────────────────────────────────────────────

#[test]
fn adt_deftype_and_construct() {
    check_ok(
        "(deftype {} MyOpt (a) (variant {} MySome (t-var {} a)) (variant {} MyNone))
         (def {} x (app {} (var {} MySome) (lit {type: (t-prim {} i32)} 42)))",
    );
}

#[test]
fn adt_nullary_constructor() {
    check_ok(
        "(deftype {} MyOpt (a) (variant {} MySome (t-var {} a)) (variant {} MyNone))
         (def {} x (var {} MyNone))",
    );
}

#[test]
fn adt_no_type_params() {
    check_ok(
        "(deftype {} Color () (variant {} Red) (variant {} Green) (variant {} Blue))
         (def {} c (var {} Red))",
    );
}

// ── Duplicate type-definition rejection ──────────────────────
// The carrier-set in `linearity::compute_tensor_carrying_adts`
// keys on bare ADT names, so silent last-write-wins on duplicate
// `deftype`s would produce order-dependent borrow semantics. The
// type checker rejects collisions at declaration time
// (`CheckErrorKind::DuplicateDefinition`); the tests below pin
// both sides of that rule.

#[test]
fn duplicate_deftype_in_same_program_is_rejected() {
    check_err(
        "(deftype {} Dup () (variant {} A))
         (deftype {} Dup () (variant {} B))",
        CheckErrorKind::DuplicateDefinition,
    );
}

#[test]
fn distinct_deftypes_with_overlapping_variant_names_are_accepted() {
    // Two ADTs may share a variant name; only ADT-name collisions
    // are duplicates. This pins that the rejection is scoped to
    // the type name, not to constructor names.
    check_ok(
        "(deftype {} Lhs () (variant {} A))
         (deftype {} Rhs () (variant {} B))
         (def {} x (var {} A))
         (def {} y (var {} B))",
    );
}

#[test]
fn deftype_colliding_with_prelude_option_is_rejected() {
    // `Option[a]` is registered by `register_prelude_adts` before
    // `collect_declarations` runs. User code re-declaring it would
    // overwrite the prelude entry under `UnordMap::insert`.
    check_err(
        "(deftype {} Option (a) (variant {} Some (t-var {} a)) (variant {} None))",
        CheckErrorKind::DuplicateDefinition,
    );
}

#[test]
fn deftype_then_typealias_with_same_name_is_rejected() {
    // `deftype` and `typealias` share the same type-name namespace
    // (both live in `AdtRegistry`). A later `typealias Holder = ...`
    // would silently overwrite an earlier `deftype Holder`.
    check_err(
        "(deftype {} Holder () (variant {} V))
         (typealias {} Holder () (t-prim {} i32))",
        CheckErrorKind::DuplicateDefinition,
    );
}

#[test]
fn typealias_then_deftype_with_same_name_is_rejected() {
    check_err(
        "(typealias {} Holder () (t-prim {} i32))
         (deftype {} Holder () (variant {} V))",
        CheckErrorKind::DuplicateDefinition,
    );
}

#[test]
fn duplicate_typealias_is_rejected() {
    check_err(
        "(typealias {} Alias () (t-prim {} i32))
         (typealias {} Alias () (t-prim {} f32))",
        CheckErrorKind::DuplicateDefinition,
    );
}

// ── Duplicate def rejection (chelis#258) ─────────────────────
// A def's value binding is silent last-write-wins, and Chelis does
// not dispatch same-name defs by arg arity or tensor rank. Two
// same-name defs (e.g. rank-distinct "overloads") therefore leave
// one arm unreachable and surface a confusing DimensionMismatch at
// the other arm's call sites. `report_duplicate_defs` rejects them
// at the definition site instead. The tests below pin both sides.

#[test]
fn duplicate_def_in_same_program_is_rejected() {
    check_err(
        "(def {} f (fn {} (params {} x) (var {} x)))
         (def {} f (fn {} (params {} y) (var {} y)))",
        CheckErrorKind::DuplicateDefinition,
    );
}

#[test]
fn duplicate_value_def_in_same_program_is_rejected() {
    // The rule keys on the `def` tag, so duplicate value defs collide too.
    check_err(
        "(def {} x (lit {type: (t-prim {} i32)} 1))
         (def {} x (lit {type: (t-prim {} i32)} 2))",
        CheckErrorKind::DuplicateDefinition,
    );
}

#[test]
fn distinct_name_defs_are_accepted() {
    // Only same-name collisions are duplicates; distinct names are fine.
    check_ok(
        "(def {} f (fn {} (params {} x) (var {} x)))
         (def {} g (fn {} (params {} y) (var {} y)))",
    );
}

#[test]
fn sig_plus_def_same_name_is_not_a_duplicate() {
    // A `defsig` + a `def` for one name is the ordinary annotated-def
    // shape (and an inline-annotated def desugars to exactly that pair),
    // so it must not be flagged. Only two `def`s for one name collide.
    check_ok(
        "(defsig {} f (a) (t-fn {} (t-var {} a) (t-var {} a)))
         (def {} f (fn {} (params {} x) (var {} x)))",
    );
}

#[test]
fn duplicate_defsig_conflicting_order_a_is_rejected() {
    check_err(
        "(defsig {} f (t-fn {} (t-prim {} f32) (t-prim {} f32)))
         (defsig {} f (t-fn {} (t-prim {} bool) (t-prim {} f32)))
         (def {} f (fn {} (params {} x) (var {} x)))",
        CheckErrorKind::DuplicateDefinition,
    );
}

#[test]
fn duplicate_defsig_conflicting_order_b_is_rejected() {
    check_err(
        "(defsig {} f (t-fn {} (t-prim {} bool) (t-prim {} f32)))
         (defsig {} f (t-fn {} (t-prim {} f32) (t-prim {} f32)))
         (def {} f (fn {} (params {} x) (var {} x)))",
        CheckErrorKind::DuplicateDefinition,
    );
}

#[test]
fn duplicate_defsig_identical_signature_is_rejected() {
    check_err(
        "(defsig {} f (t-fn {} (t-prim {} f32) (t-prim {} f32)))
         (defsig {} f (t-fn {} (t-prim {} f32) (t-prim {} f32)))
         (def {} f (fn {} (params {} x) (var {} x)))",
        CheckErrorKind::DuplicateDefinition,
    );
}

// ── Match tests ──────────────────────────────────────────────

#[test]
fn match_simple_adt() {
    check_ok(
        "(deftype {} MyOpt (a) (variant {} MySome (t-var {} a)) (variant {} MyNone))
         (def {} x (app {} (var {} MySome) (lit {type: (t-prim {} i32)} 42)))
         (def {} result
           (match {} (var {} x)
             (arm {} (pat-ctor {} MySome (pat-var {} v)) () (var {} v))
             (arm {} (pat-ctor {} MyNone) () (lit {type: (t-prim {} i32)} 0))))",
    );
}

#[test]
fn match_non_exhaustive() {
    check_err(
        "(deftype {} MyOpt (a) (variant {} MySome (t-var {} a)) (variant {} MyNone))
         (def {} x (app {} (var {} MySome) (lit {type: (t-prim {} i32)} 42)))
         (def {} result
           (match {} (var {} x)
             (arm {} (pat-ctor {} MySome (pat-var {} v)) () (var {} v))))",
        CheckErrorKind::NonExhaustiveMatch,
    );
}

// ── Defsig tests ─────────────────────────────────────────────

#[test]
fn defsig_fn_signature() {
    check_ok(
        "(defsig {} double (t-fn {} (t-prim {} f32) (t-prim {} f32)))
         (def {} double (fn {} (params {} x) (var {} x)))",
    );
}

#[test]
fn defsig_tensor_signature() {
    check_ok(
        "(defsig {} normalize_fn
           (t-fn {} (t-tensor {} (d-name {} batch) (t-prim {} f32))
                    (t-tensor {} (d-name {} batch) (t-prim {} f32))))
         (def {} normalize_fn (fn {} (params {} x) (var {} x)))",
    );
}

#[test]
fn typealias_zero_param_resolves_in_defsig() {
    check_ok(
        "(typealias {} Scalar () (t-prim {} f32))
         (defsig {} id (t-fn {} (t-adt {} Scalar) (t-adt {} Scalar)))
         (def {} id (fn {} (params {} x) (var {} x)))",
    );
}

#[test]
fn typealias_parameterized_resolves_in_defsig() {
    check_ok(
        "(typealias {} Boxed (a) (t-tuple {} (t-var {} a)))
         (defsig {} wrap (t-fn {} (t-adt {} Boxed (t-prim {} f32)) (t-adt {} Boxed (t-prim {} f32))))
         (def {} wrap (fn {} (params {} x) (var {} x)))",
    );
}

#[test]
fn typealias_resolves_in_typed_param_metadata() {
    check_ok(
        "(typealias {} Scalar () (t-prim {} f32))
         (def {} id (fn {} (params {} (x {type: (t-adt {} Scalar)})) (var {} x)))",
    );
}

#[test]
fn typealias_resolves_in_literal_metadata() {
    check_ok(
        "(typealias {} Scalar () (t-prim {} f32))
         (def {} x (lit {type: (t-adt {} Scalar)} 1.0))",
    );
}

// ── Transparent-alias-in-constructor-field tests ──────────────
//
// Per spec/02-surf-syntax.md ("Aliases are transparent — expanded
// during desugaring"), a `deftype` field declared with a transparent
// alias must unify against the alias expansion. The Hull.Ast scenario
// that motivated this is `type EffectRow = List[Effect]` used in
// `type Type = ... | TArrow(Type, Type, EffectRow) | ...`; constructing
// `TArrow(a, b, [])` must not report `EffectRow vs List`.

/// Surf source -> desugar -> IR check. Returns the collected check
/// errors (empty on success). Parse failures panic — the source is
/// the test's own fixture, not user input under test.
fn surf_check_errors(src: &str) -> Vec<CheckError> {
    let decls = chelis_surf::parser::parse_str(src).expect("surf parse");
    let exprs = chelis_surf::desugar::desugar_program(&decls);
    match check_ir_program(&exprs) {
        Ok(_) => Vec::new(),
        Err(result) => result.errors,
    }
}

fn assert_surf_ok(src: &str) {
    let errors = surf_check_errors(src);
    assert!(errors.is_empty(), "expected no errors, got: {errors:?}");
}

#[test]
fn alias_typed_ctor_field_constructs_with_list_value() {
    // EffectRow = List[Effect] declared AFTER the deftype that uses it
    // (forward reference). Constructing TArrow(a, b, e) where the third
    // field is the alias must type-check: the field expands to
    // List[Effect] and unifies with the EffectRow-typed argument.
    assert_surf_ok(
        "module M\n\
         export (mk)\n\
         type Effect =\n\
           | Pure\n\
           | Impure\n\
         type Ty =\n\
           | TBase\n\
           | TArrow(Ty, Ty, EffectRow)\n\
         type EffectRow = List[Effect]\n\
         def mk(a: Ty, b: Ty, e: EffectRow) -> Ty = TArrow(a, b, e)\n",
    );
}

#[test]
fn alias_typed_ctor_field_constructs_with_empty_list_literal() {
    // Constructing TArrow(a, b, []) where the third field is the alias
    // must type-check: [] is List[Effect], the field expands to
    // List[Effect].
    assert_surf_ok(
        "module M\n\
         export (mk)\n\
         type Effect =\n\
           | Pure\n\
           | Impure\n\
         type Ty =\n\
           | TBase\n\
           | TArrow(Ty, Ty, EffectRow)\n\
         type EffectRow = List[Effect]\n\
         def mk(a: Ty, b: Ty) -> Ty = TArrow(a, b, [])\n",
    );
}

#[test]
fn def_returning_option_tuple_with_alias_field_type_checks() {
    // The spec §3 pattern `Some((Ctor(...), []))` returning
    // Option[(Ty, EffectRow)].
    assert_surf_ok(
        "module M\n\
         export (mk)\n\
         type Effect =\n\
           | Pure\n\
           | Impure\n\
         type Ty =\n\
           | TBase\n\
           | TArrow(Ty, Ty, EffectRow)\n\
         type EffectRow = List[Effect]\n\
         def mk(a: Ty, b: Ty) -> Option[(Ty, EffectRow)] = Some((TArrow(a, b, []), []))\n",
    );
}

#[test]
fn two_level_alias_in_ctor_field_resolves() {
    // Alias of an alias: Effects = EffectRow = List[Effect].
    assert_surf_ok(
        "module M\n\
         export (mk)\n\
         type Effect =\n\
           | Pure\n\
           | Impure\n\
         type Ty =\n\
           | TBase\n\
           | TArrow(Ty, Ty, Effects)\n\
         type EffectRow = List[Effect]\n\
         type Effects = EffectRow\n\
         def mk(a: Ty, b: Ty) -> Ty = TArrow(a, b, [])\n",
    );
}

#[test]
fn ctx_list_of_tuple_alias_used_in_value_position() {
    // Ctx = List[(String, Ty)] -- a tuple-bearing alias used as a
    // constructor field; constructing with an empty list must work.
    assert_surf_ok(
        "module M\n\
         export (mk)\n\
         type Ty =\n\
           | TBase\n\
           | TArrow(Ty, Ty)\n\
         type Ctx = List[(String, Ty)]\n\
         type Judgement =\n\
           | Judge(Ctx, Ty)\n\
         def mk(t: Ty) -> Judgement = Judge([], t)\n",
    );
}

#[test]
fn genuine_mismatch_against_expanded_alias_still_rejected() {
    // NEGATIVE PARITY: passing an Int where the expanded alias is
    // List[Effect] must STILL be a TypeMismatch. Alias transparency
    // must not weaken genuine error detection.
    let errors = surf_check_errors(
        "module M\n\
         export (mk)\n\
         type Effect =\n\
           | Pure\n\
           | Impure\n\
         type Ty =\n\
           | TBase\n\
           | TArrow(Ty, Ty, EffectRow)\n\
         type EffectRow = List[Effect]\n\
         def mk(a: Ty, b: Ty) -> Ty = TArrow(a, b, 5)\n",
    );
    assert!(
        errors
            .iter()
            .any(|e| matches!(e.kind, CheckErrorKind::TypeMismatch)),
        "expected a TypeMismatch for Int passed to a List[Effect] field, got: {errors:?}"
    );
}

#[test]
fn alias_field_rejects_wrong_list_element_type() {
    // NEGATIVE PARITY: List[Ty] where the field expands to
    // List[Effect] must still mismatch on the element type.
    let errors = surf_check_errors(
        "module M\n\
         export (mk)\n\
         type Effect =\n\
           | Pure\n\
           | Impure\n\
         type Ty =\n\
           | TBase\n\
           | TArrow(Ty, Ty, EffectRow)\n\
         type EffectRow = List[Effect]\n\
         def mk(a: Ty, b: Ty, ts: List[Ty]) -> Ty = TArrow(a, b, ts)\n",
    );
    assert!(
        errors
            .iter()
            .any(|e| matches!(e.kind, CheckErrorKind::TypeMismatch)),
        "expected a TypeMismatch for List[Ty] passed to a List[Effect] field, got: {errors:?}"
    );
}

// ── Partial inference tests ──────────────────────────────────

#[test]
fn partial_inference_continues_after_error() {
    // First def has an error, second should still be processed
    let result = check(
        "(def {} x (var {} nonexistent))
         (def {} y (lit {type: (t-prim {} i32)} 42))",
    );
    assert!(!result.errors.is_empty(), "expected at least one error");
    // y should still have been typed
    assert!(result.typed_nodes > 0, "expected some typed nodes");
}

#[test]
fn multiple_errors_collected() {
    let result = check(
        "(def {} x (var {} unknown1))
         (def {} y (var {} unknown2))",
    );
    assert!(
        result.errors.len() >= 2,
        "expected at least 2 errors, got {}",
        result.errors.len()
    );
}

// ── Builtin operation tests ──────────────────────────────────

#[test]
fn builtin_neg() {
    check_ok(
        "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0))
         (def {} y (app {} (var {} neg) (var {} x)))",
    );
}

#[test]
fn builtin_exp() {
    check_ok(
        "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0))
         (def {} y (app {} (var {} exp) (var {} x)))",
    );
}

#[test]
fn builtin_matmul() {
    check_ok(
        "(def {} a (lit {type: (t-tensor {} (d-name {} batch) (d-name {} hidden) (t-prim {} f32))} 0))
         (def {} b (lit {type: (t-tensor {} (d-name {} hidden) (d-name {} classes) (t-prim {} f32))} 0))
         (def {} c (app {type: (t-tensor {} (d-name {} batch) (d-name {} classes) (t-prim {} f32))}
             (var {} matmul) (var {} a) (var {} b)))",
    );
}

#[test]
fn builtin_batched_matmul_rank4() {
    check_ok(
        "(def {} q (lit {type: (t-tensor {} (d-name {} batch) (d-name {} head) (d-name {} seq) (d-name {} dim) (t-prim {} f32))} 0))
         (def {} k (lit {type: (t-tensor {} (d-name {} batch) (d-name {} head) (d-name {} dim) (d-name {} seq) (t-prim {} f32))} 0))
         (def {} scores (app {type: (t-tensor {} (d-name {} batch) (d-name {} head) (d-name {} seq) (d-name {} seq) (t-prim {} f32))}
             (var {} matmul) (var {} q) (var {} k)))",
    );
}

#[test]
fn builtin_batched_matmul_rejects_incompatible_leading_dim() {
    check_err(
        "(def {} q (lit {type: (t-tensor {} (d-name {} batch) (d-name {} head) (d-name {} seq) (d-name {} dim) (t-prim {} f32))} 0))
         (def {} k (lit {type: (t-tensor {} (d-name {} other_batch) (d-name {} head) (d-name {} dim) (d-name {} seq) (t-prim {} f32))} 0))
         (def {} scores (app {type: (t-tensor {} (d-name {} batch) (d-name {} head) (d-name {} seq) (d-name {} seq) (t-prim {} f32))}
             (var {} matmul) (var {} q) (var {} k)))",
        CheckErrorKind::DimensionMismatch,
    );
}

#[test]
fn builtin_layer_norm() {
    check_ok(
        "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (d-name {} hidden) (t-prim {} f32))} 0))
         (def {} gamma (lit {type: (t-tensor {} (d-name {} hidden) (t-prim {} f32))} 0))
         (def {} beta (lit {type: (t-tensor {} (d-name {} hidden) (t-prim {} f32))} 0))
         (def {} y (app {} (var {} layer_norm) (var {} x) (var {} gamma) (var {} beta) (lit {type: (t-prim {} f32)} 0.00001)))",
    );
}

#[test]
fn builtin_layer_norm_rejects_rank2_gamma() {
    check_err(
        "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (d-name {} hidden) (t-prim {} f32))} 0))
         (def {} gamma (lit {type: (t-tensor {} (d-name {} hidden) (d-name {} extra) (t-prim {} f32))} 0))
         (def {} beta (lit {type: (t-tensor {} (d-name {} hidden) (t-prim {} f32))} 0))
         (def {} y (app {} (var {} layer_norm) (var {} x) (var {} gamma) (var {} beta) (lit {type: (t-prim {} f32)} 0.00001)))",
        CheckErrorKind::DimensionMismatch,
    );
}

#[test]
fn builtin_layer_norm_rejects_precision_mismatch() {
    check_err(
        "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (d-name {} hidden) (t-prim {} f32))} 0))
         (def {} gamma (lit {type: (t-tensor {} (d-name {} hidden) (t-prim {} bf16))} 0))
         (def {} beta (lit {type: (t-tensor {} (d-name {} hidden) (t-prim {} f32))} 0))
         (def {} y (app {} (var {} layer_norm) (var {} x) (var {} gamma) (var {} beta) (lit {type: (t-prim {} f32)} 0.00001)))",
        CheckErrorKind::PrecisionMismatch,
    );
}

#[test]
fn builtin_conv_accepts_int_stride_padding() {
    check_ok(
        "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (d-name {} in_c) (d-name {} h) (d-name {} w) (t-prim {} f32))} 0))
         (def {} k (lit {type: (t-tensor {} (d-name {} out_c) (d-name {} in_c) (d-lit {} 3) (d-lit {} 3) (t-prim {} f32))} 0))
         (def {} y (app {} (var {} conv) (var {} x) (var {} k) (app {} (var {} Cons) (lit {type: (t-prim {} i64)} 1) (app {} (var {} Cons) (lit {type: (t-prim {} i64)} 1) (var {} Nil))) (app {} (var {} Cons) (tuple {} (lit {type: (t-prim {} i64)} 1) (lit {type: (t-prim {} i64)} 1)) (app {} (var {} Cons) (tuple {} (lit {type: (t-prim {} i64)} 1) (lit {type: (t-prim {} i64)} 1)) (var {} Nil)))))",
    );
}

#[test]
fn builtin_conv_rejects_channel_mismatch() {
    check_err(
        "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (d-name {} in_a) (d-name {} h) (d-name {} w) (t-prim {} f32))} 0))
         (def {} k (lit {type: (t-tensor {} (d-name {} out_c) (d-name {} in_b) (d-lit {} 3) (d-lit {} 3) (t-prim {} f32))} 0))
         (def {} y (app {} (var {} conv) (var {} x) (var {} k) (app {} (var {} Cons) (lit {type: (t-prim {} i64)} 1) (app {} (var {} Cons) (lit {type: (t-prim {} i64)} 1) (var {} Nil))) (app {} (var {} Cons) (tuple {} (lit {type: (t-prim {} i64)} 1) (lit {type: (t-prim {} i64)} 1)) (app {} (var {} Cons) (tuple {} (lit {type: (t-prim {} i64)} 1) (lit {type: (t-prim {} i64)} 1)) (var {} Nil)))))",
        CheckErrorKind::DimensionMismatch,
    );
}

#[test]
fn builtin_conv_rejects_kernel_precision_mismatch() {
    check_err(
        "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (d-name {} in_c) (d-name {} h) (d-name {} w) (t-prim {} f32))} 0))
         (def {} k (lit {type: (t-tensor {} (d-name {} out_c) (d-name {} in_c) (d-lit {} 3) (d-lit {} 3) (t-prim {} bf16))} 0))
         (def {} y (app {} (var {} conv) (var {} x) (var {} k) (app {} (var {} Cons) (lit {type: (t-prim {} i64)} 1) (app {} (var {} Cons) (lit {type: (t-prim {} i64)} 1) (var {} Nil))) (app {} (var {} Cons) (tuple {} (lit {type: (t-prim {} i64)} 1) (lit {type: (t-prim {} i64)} 1)) (app {} (var {} Cons) (tuple {} (lit {type: (t-prim {} i64)} 1) (lit {type: (t-prim {} i64)} 1)) (var {} Nil)))))",
        CheckErrorKind::PrecisionMismatch,
    );
}

mod issue_1316;
mod more;
mod post_app_ledger_key;
mod post_app_replay_precedence;
mod recursion_uniformity;
mod schedule_invariants;
