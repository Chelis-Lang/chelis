use super::host_ops::*;
use super::transforms::*;
use super::*;
use chelis_deep::DeepTag;

/// chelis#399: a reef-linked ADT value carries the internal
/// `Pkg__..__Ctor` constructor name; eval rendering (the human renderer
/// AND the `--json` `ExecutionValue` ABI surface) must show the bare,
/// user-facing name, matching the de-mangling already applied to
/// diagnostics. Bare / builtin constructors pass through unchanged.
#[test]
fn eval_renderer_demangles_reef_linked_ctor() {
    let mangled = RuntimeValue::Adt {
        ctor: "Pkg__kb__chelis__agent__KellyBenchAgent__Strategy__StrategyState".to_string(),
        fields: vec![RuntimeValue::Unit, RuntimeValue::Unit],
        field_names: None,
    };
    // human renderer: bare ctor with fields
    assert_eq!(render_value(&mangled), "StrategyState((), ())");
    // `--json` / ExecutionValue ABI surface: bare ctor
    match runtime_value_to_schema(&mangled).expect("schema") {
        crate::schema::ExecutionValue::Adt { ctor, .. } => assert_eq!(ctor, "StrategyState"),
        _ => panic!("expected ExecutionValue::Adt"),
    }
    // nullary reef-linked ctor de-mangles too
    let nullary = RuntimeValue::Adt {
        ctor: "Pkg__pkg__Mod__NoBet".to_string(),
        fields: vec![],
        field_names: None,
    };
    assert_eq!(render_value(&nullary), "NoBet");
    // bare / builtin constructor is unchanged (demangle_ident no-op)
    let bare = RuntimeValue::Adt {
        ctor: "None".to_string(),
        fields: vec![],
        field_names: None,
    };
    assert_eq!(render_value(&bare), "None");
}

fn checked_surf(source: &str) -> CheckedProgram {
    let decls = chelis_surf::parser::parse_str(source).expect("surf parse");
    let exprs = chelis_surf::desugar::desugar_program(&decls);
    chelis_types::check_ir_program(&exprs).expect("ir check")
}

fn manifest_entry_with_path(
    name: &str,
    def_name: &str,
    path: Vec<chelis_types::manifest::RootPathStep>,
) -> chelis_types::manifest::RootEntry {
    chelis_types::manifest::RootEntry {
        name: name.to_string(),
        path,
        def_name: def_name.to_string(),
        ty: Expr::Atom(
            chelis_deep::ast::Atom::Name("unknown".to_string()),
            chelis_deep::Span::new(0, 0),
        ),
        lane: chelis_types::types::Lane::Host,
        required_inputs: std::collections::BTreeSet::new(),
        reasons: Vec::new(),
    }
}

#[test]
fn manifest_root_lookup_follows_recursive_list_adt_path() {
    use chelis_types::manifest::RootPathStep::Adt;

    let entry = manifest_entry_with_path("items.1.0", "items", vec![Adt(1), Adt(0)]);
    let bindings = HashMap::from([(
        "items".to_string(),
        RuntimeValue::List(vec![RuntimeValue::int_lit(1), RuntimeValue::int_lit(2)]),
    )]);

    let value = lookup_runtime_value_for_manifest_root(&entry, &bindings, &HashMap::new())
        .expect("Cons tail/head path must select the second list item");
    assert_eq!(render_value(&value), "2");
}

#[test]
fn manifest_root_lookup_rejects_a_path_step_for_the_wrong_runtime_shape() {
    use chelis_types::manifest::RootPathStep::Tuple;

    let entry = manifest_entry_with_path("value.0", "value", vec![Tuple(0)]);
    let bindings = HashMap::from([("value".to_string(), RuntimeValue::Bool(true))]);

    assert!(
        lookup_runtime_value_for_manifest_root(&entry, &bindings, &HashMap::new()).is_none(),
        "an unavailable owed component must not fall back to the base value"
    );
}

#[test]
fn with_seed_uniform_like_evaluates_body() {
    let checked = checked_surf(
        r#"
x = with seed(7i64) {
  tensor_to_scalar(
uniform_like(
  trace(pad_sequences_to([[0.0]], cast(1, int64), cast(0.0, f32)), cast(0, int32), cast(1, int32)),
  0.0,
  1.0
)
  )
}
"#,
    );

    let outcome = evaluate_host_program(&checked, &HashMap::new())
        .expect("seeded host program should evaluate");
    let value = outcome.host_bindings.get("x").expect("x binding");
    match value.as_f64() {
        Some(v) => assert!((0.0..=1.0).contains(&v), "got {v}"),
        None => panic!("expected float result, got {value:?}"),
    }
}

/// chelis#771: `literal_seed_i64` reads a `with seed(...)` literal at full
/// i64 width, peeling the `(lit {type: (t-prim {} int32)} n)` wrapper the
/// desugarer attaches (desugar.rs:1564-1569) and ignoring the int32 default
/// meta — so the evaluator's effective u64 seed matches the C lane's
/// `(uint64_t)n` instead of `eval_lit`'s int32-narrowed value.
#[test]
fn literal_seed_read_at_full_i64_width() {
    use chelis_deep::Span;
    let sp = Span::new(0, 0);
    let node = |tag: &str, children: Vec<Expr>| {
        let mut elements = vec![
            Expr::Atom(Atom::Name(tag.to_string()), sp),
            Expr::Map(MetaMap::default(), sp),
        ];
        elements.extend(children);
        Expr::List(List { elements }, sp)
    };
    // (lit {type: (t-prim {} int32)} 4294967295) — the exact shape desugar
    // emits for `seed(4294967295)`.
    let int32_seed_lit = |n: i64| {
        let t_int32 = node(
            "t-prim",
            vec![Expr::Atom(Atom::Name("int32".to_string()), sp)],
        );
        Expr::List(
            List {
                elements: vec![
                    Expr::Atom(Atom::Tag(DeepTag::Lit), sp),
                    Expr::Map(
                        MetaMap {
                            entries: vec![("type".to_string(), t_int32)],
                        },
                        sp,
                    ),
                    Expr::Atom(Atom::Int(n), sp),
                ],
            },
            sp,
        )
    };

    // The peel reads the raw i64 atom regardless of the int32 meta.
    let lit = int32_seed_lit(4_294_967_295);
    assert_eq!(literal_seed_i64(&lit), Some(4_294_967_295));
    assert_eq!(literal_seed_i64(&lit).unwrap() as u64, 4_294_967_295_u64);
    // Guard against the pre-#771 bug: `eval_lit` would narrow
    // `4294967295 as i32` = -1, sign-extend, and seed 0xFFFF_FFFF_FFFF_FFFF.
    assert_ne!(
        literal_seed_i64(&lit).unwrap() as u64,
        0xFFFF_FFFF_FFFF_FFFF_u64,
    );
    // The exact 2^31 boundary and a bare (unwrapped) int atom both read full.
    assert_eq!(
        literal_seed_i64(&int32_seed_lit(2_147_483_648)),
        Some(2_147_483_648),
    );
    assert_eq!(
        literal_seed_i64(&Expr::Atom(Atom::Int(2_147_483_648), sp)),
        Some(2_147_483_648),
    );
    // Non-literal seed expressions return None, so the caller keeps the
    // dtype-narrowing `eval_expr` fallback for computed seeds.
    let var_seed = node("var", vec![Expr::Atom(Atom::Name("s".to_string()), sp)]);
    assert_eq!(literal_seed_i64(&var_seed), None);
}

#[test]
fn ownership_drop_is_unit_and_does_not_shadow_list_drop_runtime() {
    let checked = checked_surf(
        r#"
x = {
  t = to_tensor([cast(1.0, f32), cast(2.0, f32)])
  values = to_list(t)
  actual = index(values, cast(0, int64))
  _ = drop(t)
  actual
}
y = index(drop([cast(10, int64), cast(20, int64)], cast(1, int64)), cast(0, int64))
"#,
    );

    let outcome = evaluate_host_program(&checked, &HashMap::new())
        .expect("ownership drop and list drop should both evaluate");
    assert!(matches!(
        outcome.host_bindings.get("x").and_then(RuntimeValue::as_f64),
        Some(v) if (v - 1.0).abs() < f64::EPSILON
    ));
    assert_eq!(
        outcome
            .host_bindings
            .get("y")
            .and_then(RuntimeValue::as_i64),
        Some(20),
    );
}

// ----- Phase 3t.1: test_assert_* builtins -----

#[test]
fn test_assert_true_returns_unit() {
    let checked = checked_surf(r#"x = test_assert(true, "ok")"#);
    let outcome = evaluate_host_program(&checked, &HashMap::new())
        .expect("test_assert(true, ...) should evaluate to Ok");
    let value = outcome.host_bindings.get("x").expect("x binding");
    assert!(matches!(value, RuntimeValue::Unit), "got {value:?}");
}

#[test]
fn test_assert_false_returns_err_with_label() {
    let checked = checked_surf(r#"x = test_assert(false, "my-label")"#);
    let err = evaluate_host_program(&checked, &HashMap::new())
        .expect_err("test_assert(false, ...) should surface as host Err");
    assert!(
        err.contains("assert failed") && err.contains("my-label"),
        "expected 'assert failed' and 'my-label' in error, got: {err}"
    );
}

#[test]
fn test_assert_eq_f32_mismatch_includes_actual_and_expected() {
    let checked = checked_surf(r#"x = test_assert_eq_f32(1.0, 2.0, "label")"#);
    let err = evaluate_host_program(&checked, &HashMap::new())
        .expect_err("mismatched f32 assert should surface as host Err");
    assert!(err.contains("1") && err.contains("2"), "got: {err}");
    assert!(err.contains("label"), "expected label in error, got: {err}");
}

#[test]
fn test_assert_eq_f32_match_returns_unit() {
    let checked = checked_surf(r#"x = test_assert_eq_f32(1.5, 1.5, "same")"#);
    let outcome = evaluate_host_program(&checked, &HashMap::new())
        .expect("matched f32 assert should evaluate");
    let value = outcome.host_bindings.get("x").expect("x binding");
    assert!(matches!(value, RuntimeValue::Unit), "got {value:?}");
}

#[test]
fn test_assert_eq_int_match_and_mismatch() {
    let ok = checked_surf(r#"x = test_assert_eq_int(cast(3, int64), cast(3, int64), "i")"#);
    let outcome = evaluate_host_program(&ok, &HashMap::new()).expect("int match should eval");
    assert!(matches!(
        outcome.host_bindings.get("x"),
        Some(RuntimeValue::Unit)
    ));

    let bad = checked_surf(r#"x = test_assert_eq_int(cast(3, int64), cast(5, int64), "i")"#);
    let err =
        evaluate_host_program(&bad, &HashMap::new()).expect_err("int mismatch should surface Err");
    assert!(
        err.contains("3") && err.contains("5") && err.contains("i"),
        "got: {err}"
    );
}

#[test]
fn test_assert_eq_bool_match_and_mismatch() {
    let ok = checked_surf(r#"x = test_assert_eq_bool(true, true, "b")"#);
    evaluate_host_program(&ok, &HashMap::new()).expect("bool match should eval");

    let bad = checked_surf(r#"x = test_assert_eq_bool(true, false, "b")"#);
    let err =
        evaluate_host_program(&bad, &HashMap::new()).expect_err("bool mismatch should surface Err");
    assert!(
        err.contains("true") && err.contains("false") && err.contains("b"),
        "got: {err}"
    );
}

#[test]
fn test_assert_eq_string_match_and_mismatch() {
    let ok = checked_surf(r#"x = test_assert_eq_string("hi", "hi", "s")"#);
    evaluate_host_program(&ok, &HashMap::new()).expect("string match should eval");

    let bad = checked_surf(r#"x = test_assert_eq_string("foo", "bar", "s")"#);
    let err = evaluate_host_program(&bad, &HashMap::new())
        .expect_err("string mismatch should surface Err");
    assert!(
        err.contains("foo") && err.contains("bar") && err.contains("s"),
        "got: {err}"
    );
}

#[test]
fn test_assert_close_tensor_reports_first_mismatch_index() {
    // Use 4-element tensors [1.0, 2.0, 3.0, 4.0] vs [1.0, 2.0, 99.0, 4.0]:
    // index 2 is the first mismatch.
    let checked = checked_surf(
        r#"
actual: tensor[4, f32] = to_tensor([1.0, 2.0, 3.0, 4.0])
expected: tensor[4, f32] = to_tensor([1.0, 2.0, 99.0, 4.0])
x = test_assert_close_tensor(actual, expected, 0.001, "close")
"#,
    );
    let err = evaluate_host_program(&checked, &HashMap::new())
        .expect_err("close-tensor mismatch should surface Err");
    assert!(
        err.contains("at index 2") && err.contains("close"),
        "expected 'at index 2' and label, got: {err}"
    );
    assert!(err.contains("99") && err.contains('3'), "got: {err}");
}

#[test]
fn test_assert_close_tensor_match_returns_unit() {
    let checked = checked_surf(
        r#"
actual: tensor[3, f32] = to_tensor([1.0, 2.0, 3.0])
expected: tensor[3, f32] = to_tensor([1.001, 2.001, 3.001])
x = test_assert_close_tensor(actual, expected, 0.01, "close")
"#,
    );
    let outcome = evaluate_host_program(&checked, &HashMap::new())
        .expect("close-tensor within tol should evaluate to Unit");
    assert!(matches!(
        outcome.host_bindings.get("x"),
        Some(RuntimeValue::Unit)
    ));
}

#[test]
fn test_assert_close_tensor_zero_tol_passes_bit_exact() {
    // Regression: tol = 0 with identical data must pass, not report a false mismatch.
    let checked = checked_surf(
        r#"
actual: tensor[3, f32] = to_tensor([1.0, 2.0, 3.0])
expected: tensor[3, f32] = to_tensor([1.0, 2.0, 3.0])
x = test_assert_close_tensor(actual, expected, 0.0, "bit-exact")
"#,
    );
    let outcome = evaluate_host_program(&checked, &HashMap::new())
        .expect("zero-tol bit-exact equality should pass");
    assert!(matches!(
        outcome.host_bindings.get("x"),
        Some(RuntimeValue::Unit)
    ));
}

#[test]
fn test_assert_close_tensor_zero_tol_rejects_any_delta() {
    let checked = checked_surf(
        r#"
actual: tensor[2, f32] = to_tensor([1.0, 2.0])
expected: tensor[2, f32] = to_tensor([1.0, 2.00001])
x = test_assert_close_tensor(actual, expected, 0.0, "strict")
"#,
    );
    let err = evaluate_host_program(&checked, &HashMap::new())
        .expect_err("zero-tol with any delta must fail");
    assert!(
        err.contains("at index 1") && err.contains("strict"),
        "got: {err}"
    );
}

#[test]
fn test_assert_close_tensor_nan_actual_fails() {
    // Regression: NaN in actual must fail. (NaN - x).abs() is NaN, which silently
    // passes the old `>= tol` check. sqrt(-1.0) produces NaN.
    let checked = checked_surf(
        r#"
nan_val: f32 = sqrt(-1.0)
actual: tensor[2, f32] = to_tensor([1.0, nan_val])
expected: tensor[2, f32] = to_tensor([1.0, 2.0])
x = test_assert_close_tensor(actual, expected, 0.01, "nan-actual")
"#,
    );
    let err =
        evaluate_host_program(&checked, &HashMap::new()).expect_err("NaN in actual must fail");
    assert!(
        err.contains("at index 1") && err.contains("NaN"),
        "expected NaN-aware diagnostic, got: {err}"
    );
}

#[test]
fn test_assert_close_tensor_nan_expected_fails() {
    let checked = checked_surf(
        r#"
nan_val: f32 = sqrt(-1.0)
actual: tensor[2, f32] = to_tensor([1.0, 2.0])
expected: tensor[2, f32] = to_tensor([1.0, nan_val])
x = test_assert_close_tensor(actual, expected, 0.01, "nan-expected")
"#,
    );
    let err =
        evaluate_host_program(&checked, &HashMap::new()).expect_err("NaN in expected must fail");
    assert!(
        err.contains("at index 1") && err.contains("NaN"),
        "got: {err}"
    );
}

#[test]
fn test_assert_close_tensor_negative_tol_rejected() {
    let checked = checked_surf(
        r#"
actual: tensor[2, f32] = to_tensor([1.0, 2.0])
expected: tensor[2, f32] = to_tensor([1.0, 2.0])
x = test_assert_close_tensor(actual, expected, -0.001, "neg-tol")
"#,
    );
    let err = evaluate_host_program(&checked, &HashMap::new())
        .expect_err("negative tol must be rejected");
    assert!(
        err.contains("invalid tolerance") && err.contains("neg-tol"),
        "got: {err}"
    );
}

#[test]
fn test_assert_close_tensor_nan_tol_rejected() {
    let checked = checked_surf(
        r#"
nan_tol: f32 = sqrt(-1.0)
actual: tensor[2, f32] = to_tensor([1.0, 2.0])
expected: tensor[2, f32] = to_tensor([1.0, 4.0])
x = test_assert_close_tensor(actual, expected, nan_tol, "nan-tol")
"#,
    );
    let err =
        evaluate_host_program(&checked, &HashMap::new()).expect_err("NaN tol must be rejected");
    assert!(
        err.contains("invalid tolerance") && err.contains("nan-tol"),
        "got: {err}"
    );
}

// ----- N2 fix: matmul / permute / sum host evaluator coverage -----
//
// Pins the closure of the upstream-reported gap: native `chelis test`
// erroring with `unsupported builtin 'matmul'` / `'permute'` / `'sum'`
// when those primitives appear in a test's dependency graph.

fn first_tensor_data(outcome: &RuntimeOutcome, name: &str) -> Vec<f64> {
    match outcome.host_bindings.get(name) {
        Some(RuntimeValue::Tensor(t)) => t.value.to_f64_lossy_vec().clone(),
        other => panic!("expected tensor binding {name}, got {other:?}"),
    }
}

fn first_tensor_shape(outcome: &RuntimeOutcome, name: &str) -> Vec<usize> {
    match outcome.host_bindings.get(name) {
        Some(RuntimeValue::Tensor(t)) => t.value.shape.clone(),
        other => panic!("expected tensor binding {name}, got {other:?}"),
    }
}

#[test]
fn host_runtime_matmul_2x2_identity_passthrough() {
    let checked = checked_surf(
        r#"
a = pad_sequences_to([[cast(1.0, f32), cast(0.0, f32)], [cast(0.0, f32), cast(1.0, f32)]], cast(2, int64), cast(0.0, f32))
b = pad_sequences_to([[cast(3.0, f32), cast(5.0, f32)], [cast(7.0, f32), cast(11.0, f32)]], cast(2, int64), cast(0.0, f32))
y = matmul(a, b)
"#,
    );
    let outcome = evaluate_host_program(&checked, &HashMap::new())
        .expect("matmul should evaluate under host runtime");
    assert_eq!(first_tensor_shape(&outcome, "y"), vec![2, 2]);
    assert_eq!(first_tensor_data(&outcome, "y"), vec![3.0, 5.0, 7.0, 11.0]);
}

// IEEE-754 corner cases for `Div` and `Recip` on the runtime
// evaluator path. A `mul(a, exp(neg(log(b))))` decomposition
// would NaN on every non-positive operand below; these tests
// pin the IEEE-correct outputs and serve as regression guards.
#[test]
fn host_runtime_div_negative_divisor_returns_finite_value() {
    let checked = checked_surf(
        r#"
a = to_tensor([cast(5.0, f32)])
b = to_tensor([cast(-2.0, f32)])
y = div(a, b)
"#,
    );
    let outcome = evaluate_host_program(&checked, &HashMap::new())
        .expect("div with negative divisor should evaluate");
    // IEEE: 5 / -2 = -2.5 (an `exp(neg(log(-2)))` decomposition
    // would NaN here).
    assert_eq!(first_tensor_data(&outcome, "y"), vec![-2.5]);
}

#[test]
fn host_runtime_div_by_positive_zero_is_positive_infinity() {
    let checked = checked_surf(
        r#"
a = to_tensor([cast(1.0, f32)])
b = to_tensor([cast(0.0, f32)])
y = div(a, b)
"#,
    );
    let outcome =
        evaluate_host_program(&checked, &HashMap::new()).expect("div by zero should evaluate");
    let v = first_tensor_data(&outcome, "y");
    assert_eq!(v.len(), 1);
    assert!(
        v[0].is_infinite() && v[0] > 0.0,
        "expected +inf, got {}",
        v[0]
    );
}

#[test]
fn host_runtime_div_negative_one_by_zero_is_negative_infinity() {
    let checked = checked_surf(
        r#"
a = to_tensor([cast(-1.0, f32)])
b = to_tensor([cast(0.0, f32)])
y = div(a, b)
"#,
    );
    let outcome = evaluate_host_program(&checked, &HashMap::new()).expect("-1/0 should evaluate");
    let v = first_tensor_data(&outcome, "y");
    assert_eq!(v.len(), 1);
    assert!(
        v[0].is_infinite() && v[0] < 0.0,
        "expected -inf, got {}",
        v[0]
    );
}

#[test]
fn host_runtime_div_zero_by_zero_is_nan() {
    let checked = checked_surf(
        r#"
a = to_tensor([cast(0.0, f32)])
b = to_tensor([cast(0.0, f32)])
y = div(a, b)
"#,
    );
    let outcome =
        evaluate_host_program(&checked, &HashMap::new()).expect("0/0 should evaluate (NaN)");
    let v = first_tensor_data(&outcome, "y");
    assert_eq!(v.len(), 1);
    assert!(v[0].is_nan(), "expected NaN, got {}", v[0]);
}

// chelis#458 regression lock (eval/fold side). On chelis 0.9.0 a mixed
// `(f32, i32)` numeric op silently const-folded when both operands were
// literals: `div(1.0, 4)` evaluated to 0.25 via a `_ => Prim::F32`
// re-precisioning fallback, while the bound-variable form errored at
// type-check — a check↔eval inconsistency (the #458 defect). Per
// spec/04-type-system.md §5.1 (no implicit promotion) and §5.2 (casts are
// always explicit) the mixed pair is a type error, so the literal form
// must NOT fold to 0.25. These two tests pin both barriers: the
// type-checker rejects the literal pair before eval, AND the host scalar
// dispatch itself rejects a mixed pair rather than re-precisioning it.

/// The full check→eval pipeline rejects `div(1.0, 4)` at type-check, so
/// the const-fold to 0.25 is never reached. `4` is an `int32` literal
/// (§5.3), making this the same mixed `(f32, int32)` pair as the
/// bound-variable form.
#[test]
fn issue_458_div_f32_over_i32_literal_rejected_before_eval_not_folded() {
    let decls = chelis_surf::parser::parse_str("out = div(1.0, 4)").expect("surf parse");
    let exprs = chelis_surf::desugar::desugar_program(&decls);
    let err = chelis_types::check_ir_program(&exprs).expect_err(
        "chelis#458: div(1.0, 4) is a mixed (f32, int32) pair and must be a type error",
    );
    assert!(
        err.errors.iter().any(|e| matches!(
            e.kind,
            chelis_types::errors::CheckErrorKind::PrecisionMismatch
        )),
        "spec §5.1: mixed (f32, int32) div must surface a PrecisionMismatch; got: {:?}",
        err.errors
    );
}

/// Runtime backstop: even if a future type-checker hole let a mixed
/// `(f32, int32)` scalar pair reach the host evaluator, the closed kernel
/// dispatcher
/// must reject it rather than re-precisioning to a 0.25 float fold (the old
/// 0.9.0 `_ => Prim::F32` fallback). The mixed pair is neither int-int nor
/// float-float, so it falls to the `_ => Err(..)` catch-all.
#[test]
fn issue_458_closed_binop_dispatch_rejects_mixed_f32_i32_not_folds_to_quarter() {
    let lhs = RuntimeValue::scalar_like_float(chelis_types::types::Prim::F32, 1.0)
        .expect("f32 scalar 1.0");
    let rhs =
        RuntimeValue::scalar_like_int(chelis_types::types::Prim::Int32, 4).expect("int32 scalar 4");
    let result = numeric_binop(
        &[lhs, rhs],
        Some(chelis_types::IntBinOp::TruncDiv),
        Some(chelis_types::FloatBinOp::Div),
    );
    assert!(
        result.is_err(),
        "chelis#458 / spec §5.1: a mixed (f32, int32) scalar div must be rejected by the \
         host dispatch, NOT silently re-precisioned and folded to 0.25; got Ok({result:?})"
    );
}

#[test]
fn host_runtime_recip_negative_value_is_negative_reciprocal() {
    let checked = checked_surf(
        r#"
a = to_tensor([cast(-2.0, f32)])
y = recip(a)
"#,
    );
    let outcome =
        evaluate_host_program(&checked, &HashMap::new()).expect("recip(-2.0) should evaluate");
    // IEEE: 1 / -2 = -0.5 (an `exp(neg(log(-2)))` decomposition
    // would NaN here).
    assert_eq!(first_tensor_data(&outcome, "y"), vec![-0.5]);
}

#[test]
fn host_runtime_matmul_2x3_3x2_basic() {
    let checked = checked_surf(
        r#"
a = pad_sequences_to([[cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)], [cast(4.0, f32), cast(5.0, f32), cast(6.0, f32)]], cast(3, int64), cast(0.0, f32))
b = pad_sequences_to([[cast(1.0, f32), cast(0.0, f32)], [cast(0.0, f32), cast(1.0, f32)], [cast(1.0, f32), cast(1.0, f32)]], cast(2, int64), cast(0.0, f32))
y = matmul(a, b)
"#,
    );
    let outcome = evaluate_host_program(&checked, &HashMap::new())
        .expect("rectangular matmul should evaluate");
    assert_eq!(first_tensor_shape(&outcome, "y"), vec![2, 2]);
    // Row 0: [1*1+2*0+3*1, 1*0+2*1+3*1] = [4, 5]
    // Row 1: [4*1+5*0+6*1, 4*0+5*1+6*1] = [10, 11]
    assert_eq!(first_tensor_data(&outcome, "y"), vec![4.0, 5.0, 10.0, 11.0]);
}

#[test]
fn host_runtime_permute_2x2_transpose_swaps_off_diagonal() {
    let checked = checked_surf(
        r#"
a = pad_sequences_to([[cast(1.0, f32), cast(2.0, f32)], [cast(3.0, f32), cast(4.0, f32)]], cast(2, int64), cast(0.0, f32))
y = permute(a, 1, 0)
"#,
    );
    let outcome = evaluate_host_program(&checked, &HashMap::new())
        .expect("permute should evaluate under host runtime");
    // Row-major: original [[1,2],[3,4]] -> transpose [[1,3],[2,4]]
    assert_eq!(first_tensor_shape(&outcome, "y"), vec![2, 2]);
    assert_eq!(first_tensor_data(&outcome, "y"), vec![1.0, 3.0, 2.0, 4.0]);
}

#[test]
fn host_runtime_permute_2x3_transpose_to_3x2() {
    let checked = checked_surf(
        r#"
a = pad_sequences_to([[cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)], [cast(4.0, f32), cast(5.0, f32), cast(6.0, f32)]], cast(3, int64), cast(0.0, f32))
y = permute(a, 1, 0)
"#,
    );
    let outcome = evaluate_host_program(&checked, &HashMap::new())
        .expect("rectangular permute should evaluate");
    assert_eq!(first_tensor_shape(&outcome, "y"), vec![3, 2]);
    // [[1,2,3],[4,5,6]] -> [[1,4],[2,5],[3,6]]
    assert_eq!(
        first_tensor_data(&outcome, "y"),
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    );
}

#[test]
fn host_runtime_sum_axis1_reduces_2x3_to_2() {
    let checked = checked_surf(
        r#"
a = pad_sequences_to([[cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)], [cast(4.0, f32), cast(5.0, f32), cast(6.0, f32)]], cast(3, int64), cast(0.0, f32))
y = sum(a, cast(1, int32))
"#,
    );
    let outcome = evaluate_host_program(&checked, &HashMap::new())
        .expect("sum should evaluate under host runtime");
    assert_eq!(first_tensor_shape(&outcome, "y"), vec![2]);
    assert_eq!(first_tensor_data(&outcome, "y"), vec![6.0, 15.0]);
}

#[test]
fn host_runtime_sum_axis0_reduces_2x3_to_3() {
    let checked = checked_surf(
        r#"
a = pad_sequences_to([[cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)], [cast(4.0, f32), cast(5.0, f32), cast(6.0, f32)]], cast(3, int64), cast(0.0, f32))
y = sum(a, cast(0, int32))
"#,
    );
    let outcome =
        evaluate_host_program(&checked, &HashMap::new()).expect("sum on axis 0 should evaluate");
    assert_eq!(first_tensor_shape(&outcome, "y"), vec![3]);
    assert_eq!(first_tensor_data(&outcome, "y"), vec![5.0, 7.0, 9.0]);
}

// Issue Chelis-Lang/chelis#163: `sum` on f32 must use the stride-4 ILP
// cascade (torch's CPU `row_sum`) reduction order, not the previous
// strict left-fold. The 11-element reflected-pad sequence below is
// the issue's exact reproducer pattern: the same multiset summed in
// two different orderings produces the same result under stride-4
// (matches torch/numpy) but differs by 1 ULP under left-fold.
//
// Source values: torch.rand(6) with manual_seed(0); each f32 value
// is expressed as the f64 string that round-trips back to the same
// f32 bit pattern via the Surf `cast(_, f32)` path.
#[test]
fn host_runtime_sum_f32_uses_pairwise_order_for_issue_163_repro() {
    // right-pad: [v0, v1, v2, v3, v4, v5, v4, v3, v2, v1, v0]
    let checked_right = checked_surf(
        r#"
seq = to_tensor([
cast(0.49625658988952637, f32),
cast(0.7682217955589294, f32),
cast(0.08847743272781372, f32),
cast(0.13203048706054688, f32),
cast(0.30742114782333374, f32),
cast(0.6340786814689636, f32),
cast(0.30742114782333374, f32),
cast(0.13203048706054688, f32),
cast(0.08847743272781372, f32),
cast(0.7682217955589294, f32),
cast(0.49625658988952637, f32)
])
y = sum(seq, cast(0, int32))
"#,
    );
    let outcome_right = evaluate_host_program(&checked_right, &HashMap::new())
        .expect("right-pad reflected sum should evaluate");
    let right = first_tensor_data(&outcome_right, "y");
    assert_eq!(right.len(), 1);
    // Stride-4 ILP cascade f32 result. Matches torch's CPU
    // `row_sum` bit-exactly for n <= 16; coincides with
    // `numpy.sum` only because this specific 11-element multiset
    // happens to round the same way under both the stride-4
    // cascade and numpy's pairwise tree — the two algorithms
    // disagree in general (numpy uses a divide-and-conquer
    // pairwise tree with 128-element blocks). The old strict
    // left-fold would have produced 4.218894004821777 here — a
    // 1-ULP drift that the parity harness now no longer needs to
    // carve out (issue #163 acceptance criterion).
    // Bit patterns rather than f32 decimal literals: clippy's
    // `excessive_precision` lint would rewrite the source
    // literals to shorter decimals that round to the SAME bits
    // but obscure intent. This regression-lock IS about exact
    // bits, so encode them directly.
    let stride4 = 0x4087012d_u32; // = 4.218893527984619 -> f32 (stride-4 cascade)
    let left_fold = 0x4087012e_u32; // = 4.218894004821777_f32 (old left-fold)
    assert_eq!(
        (right[0] as f32).to_bits(),
        stride4,
        "expected stride-4 cascade result; got {}",
        right[0]
    );
    // Negative regression-lock: the test must also assert the OLD
    // left-fold result is NOT produced, so a future change that
    // accidentally reverts to a left-fold (or to a different
    // tree shape that lands on the old value) fails loudly here.
    assert_ne!(
        (right[0] as f32).to_bits(),
        left_fold,
        "regression: result matches the old left-fold value 4.218894004821777, \
         which the stride-4 cascade was supposed to replace"
    );

    // Same multiset, left-pad ordering. Both stride-4 and the old
    // left-fold happen to agree here — pinning to prove parity stays
    // intact across the algorithm change.
    let checked_left = checked_surf(
        r#"
seq = to_tensor([
cast(0.30742114782333374, f32),
cast(0.13203048706054688, f32),
cast(0.08847743272781372, f32),
cast(0.7682217955589294, f32),
cast(0.49625658988952637, f32),
cast(0.49625658988952637, f32),
cast(0.7682217955589294, f32),
cast(0.08847743272781372, f32),
cast(0.13203048706054688, f32),
cast(0.30742114782333374, f32),
cast(0.6340786814689636, f32)
])
y = sum(seq, cast(0, int32))
"#,
    );
    let outcome_left = evaluate_host_program(&checked_left, &HashMap::new())
        .expect("left-pad reflected sum should evaluate");
    let left = first_tensor_data(&outcome_left, "y");
    assert_eq!(left.len(), 1);
    assert_eq!(
        (left[0] as f32).to_bits(),
        stride4,
        "left-pad ordering must produce same result as right-pad under stride-4; got {}",
        left[0]
    );
}

/// Trace is diagonal followed by [05-OP-30]'s canonical adjacent-pair tree.
/// This cancellation sequence distinguishes it from both retired orders.
#[test]
fn host_runtime_trace_f32_uses_canonical_balanced_tree() {
    let diag = [1e20_f64, 1.0, -1e20_f64, 1.0, 1.0];
    let mut m = vec![0.0_f64; 5 * 5];
    for (i, &v) in diag.iter().enumerate() {
        m[i * 5 + i] = v;
    }
    let tensor = RuntimeTensorValue {
        value: IrTensorValue::from_vec(vec![5, 5], m),
        precision: Prim::F32,
    };
    let out = tensor_trace_value(&tensor, 0, 1).expect("trace must evaluate");
    assert_eq!(out.value.shape, Vec::<usize>::new(), "trace is a scalar");
    assert_eq!(
        (out.value.to_f64_lossy_vec()[0] as f32).to_bits(),
        1.0_f32.to_bits()
    );
}

/// The f64 host lane uses the same canonical tree at f64 arithmetic width.
#[test]
fn host_runtime_trace_f64_uses_canonical_balanced_tree() {
    let diag = [1e300_f64, 1.0, -1e300_f64, 1.0, 1.0];
    let mut m = vec![0.0_f64; 5 * 5];
    for (i, &v) in diag.iter().enumerate() {
        m[i * 5 + i] = v;
    }
    let tensor = RuntimeTensorValue {
        value: IrTensorValue::from_vec(vec![5, 5], m),
        precision: Prim::F64,
    };
    let out = tensor_trace_value(&tensor, 0, 1).expect("trace must evaluate");
    assert_eq!(out.value.shape, Vec::<usize>::new(), "trace is a scalar");
    assert_eq!(out.value.to_f64_lossy_vec()[0].to_bits(), 1.0_f64.to_bits());
}

/// #172 sibling (eval lane): windowed Max/Min DROP NaN — Rust `f64::max`/`min`
/// return the non-NaN operand — unlike `max_reduce`/`min_reduce`, which
/// PROPAGATE NaN (`tensor_reduce_host` `saw_nan` path). This pins the
/// documented drop-vs-propagate asymmetry on the eval side; the C lane is
/// pinned by `reduce_window_max_min_drop_nan` in `chelis-backend-c`. The two
/// windows `[NaN, 1.0]` and `[2.0, NaN]` must reduce to the finite operand in
/// either NaN position.
#[test]
fn host_runtime_reduce_window_max_min_drop_nan() {
    let tensor = RuntimeTensorValue {
        value: IrTensorValue::from_vec(vec![4], vec![f64::NAN, 1.0, 2.0, f64::NAN]),
        precision: Prim::F32,
    };
    let max = tensor_reduce_window_host(
        &tensor,
        &[2],
        &[2],
        ReduceWindowOp::Max,
        "reduce_window_max",
    )
    .expect("reduce_window max must evaluate");
    assert_eq!(max.value.shape, vec![2]);
    assert!(
        max.value.to_f64_lossy_vec().iter().all(|v| !v.is_nan()),
        "windowed max must DROP NaN (not propagate); got {:?}",
        max.value.to_f64_lossy_vec(),
    );
    assert_eq!(
        max.value.to_f64_lossy_vec(),
        vec![1.0, 2.0],
        "windowed max drops NaN -> the non-NaN operand",
    );
    let min = tensor_reduce_window_host(
        &tensor,
        &[2],
        &[2],
        ReduceWindowOp::Min,
        "reduce_window_min",
    )
    .expect("reduce_window min must evaluate");
    assert!(
        min.value.to_f64_lossy_vec().iter().all(|v| !v.is_nan()),
        "windowed min must DROP NaN (not propagate); got {:?}",
        min.value.to_f64_lossy_vec(),
    );
    assert_eq!(
        min.value.to_f64_lossy_vec(),
        vec![1.0, 2.0],
        "windowed min drops NaN -> the non-NaN operand",
    );
}

/// #170 DECISION-LOCK: f32 `matmul` deliberately keeps an f64 eval
/// accumulator and does NOT take the #163 `sum` cascade nor downcast to
/// strict f32. torch's CPU f32 matmul is a BLAS GEMM (strict-f32-left-fold
/// order, NOT the cascade), and the eval reference intentionally stays at
/// HIGHER precision (f64): it is the reference, the shipped C backend uses
/// `cblas_sgemm`, and the matmul eval-vs-C parity oracle uses a TOLERANCE,
/// not bit-identity, for exactly this expected gap. Downcasting eval to
/// strict f32 would lower precision, couple the reference to torch's BLAS
/// version, and still not win bit-identity — so it is rejected.
///
/// The absorption probe `[2^24, 1×40, -2^24] · [1×42]` pins this: under the
/// retained f64 accumulator the forty `1.0`s are preserved (`-> 40.0`);
/// under a strict-f32 fold they would be absorbed by `2^24` (`-> 0.0`).
/// This test FAILS if someone "fixes" matmul into strict f32 (or the
/// cascade), guarding the documented decision.
#[test]
fn host_runtime_matmul_f32_keeps_f64_accumulator_not_strict_f32() {
    let mut lhs_row = vec![16_777_216.0_f64];
    lhs_row.extend(std::iter::repeat_n(1.0_f64, 40));
    lhs_row.push(-16_777_216.0_f64);
    let k = lhs_row.len(); // 42
    let lhs = RuntimeTensorValue {
        value: IrTensorValue::from_vec(vec![1, k], lhs_row),
        precision: Prim::F32,
    };
    let rhs = RuntimeTensorValue {
        value: IrTensorValue::from_vec(vec![k, 1], vec![1.0_f64; k]),
        precision: Prim::F32,
    };
    let out = tensor_matmul_host(&lhs, &rhs).expect("matmul must evaluate");
    assert_eq!(
        out.value.to_f64_lossy_vec(),
        vec![40.0_f64],
        "f32 matmul keeps the higher-precision f64 eval accumulator (#170 decision): \
         the forty 1.0s survive (=> 40.0); a strict-f32 fold would absorb them (=> 0.0)"
    );
}

/// #170 DECISION-LOCK: f32 `einsum` shares matmul's disposition — f64 eval
/// accumulator retained, no cascade, no strict-f32 downcast. Same
/// absorption probe; FAILS if einsum is downcast to strict f32.
#[test]
fn host_runtime_einsum_f32_keeps_f64_accumulator_not_strict_f32() {
    let mut lhs_row = vec![16_777_216.0_f64];
    lhs_row.extend(std::iter::repeat_n(1.0_f64, 40));
    lhs_row.push(-16_777_216.0_f64);
    let k = lhs_row.len();
    let lhs = RuntimeTensorValue {
        value: IrTensorValue::from_vec(vec![1, k], lhs_row),
        precision: Prim::F32,
    };
    let rhs = RuntimeTensorValue {
        value: IrTensorValue::from_vec(vec![k, 1], vec![1.0_f64; k]),
        precision: Prim::F32,
    };
    let out = tensor_einsum_value("ik,kj->ij", &lhs, &rhs).expect("einsum must evaluate");
    assert_eq!(
        out.value.to_f64_lossy_vec(),
        vec![40.0_f64],
        "f32 einsum keeps the f64 eval accumulator (#170 decision); got {:?}",
        out.value.to_f64_lossy_vec()
    );
}

#[test]
fn host_runtime_einsum_accepts_the_legal_rank_zero_grammar() {
    let lhs = RuntimeTensorValue {
        value: IrTensorValue::from_vec(vec![], vec![2.0]),
        precision: Prim::F32,
    };
    let rhs = RuntimeTensorValue {
        value: IrTensorValue::from_vec(vec![], vec![3.0]),
        precision: Prim::F32,
    };
    let output = tensor_einsum_value(",->", &lhs, &rhs).expect("rank-zero einsum is legal");
    assert_eq!(output.value.shape, Vec::<usize>::new());
    assert_eq!(output.value.to_f64_lossy_vec(), vec![6.0]);
}

/// A zero extent means zero elements wherever the zero sits, so both shapes
/// below describe the same empty operand and the derived reduction count is
/// zero for both. A left-to-right checked fold reaches `BIG * BIG` first,
/// which is not an int64, and so accepted one permutation while rejecting the
/// other. The C runtime holds the same invariant in
/// `crates/chelis-runtime/tests/op33_int64_extent_domain.rs`.
#[test]
fn host_runtime_einsum_zero_extent_acceptance_does_not_depend_on_axis_order() {
    const BIG: usize = 4_000_000_000;
    // The zero's axis is the only difference between the two cases, and the
    // equation names it, so diagnostics identify the case by equation rather
    // than by Debug-printing the extents (faithful_observation.md B2.4: no
    // third formatter in an observation exit surface).
    for (shape, equation) in [
        (vec![BIG, 0, BIG], "abc,def->b"),
        (vec![BIG, BIG, 0], "abc,def->c"),
    ] {
        let operand = RuntimeTensorValue {
            value: IrTensorValue::from_vec(shape, Vec::new()),
            precision: Prim::F32,
        };
        let output = tensor_einsum_value(equation, &operand, &operand)
            .unwrap_or_else(|error| panic!("host einsum `{equation}` must evaluate: {error}"));
        assert_eq!(
            output.value.shape,
            vec![0],
            "host einsum `{equation}` must produce an empty result"
        );
    }
}

/// An empty operand's axis decomposition is never read, and computing it
/// anyway overflows `usize` on the prefix product or spins an empty loop. The
/// C runtime holds the same invariant in
/// `crates/chelis-runtime/tests/op33_empty_tensor_axis_decomposition.rs`; this
/// is the host lane's half. The extents are chosen so the prefix product is
/// `2^64` exactly.
#[test]
fn host_runtime_empty_operands_skip_their_axis_decomposition() {
    const BIG: usize = 1 << 32;
    let operand = RuntimeTensorValue {
        value: IrTensorValue::from_vec(vec![BIG, BIG, 0], Vec::new()),
        precision: Prim::F32,
    };

    let scanned = tensor_cumsum_value(&operand, 2).expect("empty cumsum must evaluate");
    assert_eq!(scanned.value.shape, vec![BIG, BIG, 0]);
    assert_eq!(scanned.value.len(), 0);

    let sorted = tensor_sort_value(&operand, 2).expect("empty sort must evaluate");
    // Diagnostics name the expectation rather than Debug-printing the value:
    // this module is an observation exit surface, and
    // faithful_observation.md B2.4 admits no third formatter.
    match sorted {
        RuntimeValue::Tuple(items) => {
            assert_eq!(items.len(), 2, "sort must return two results");
            for item in items {
                match item {
                    RuntimeValue::Tensor(tensor) => assert_eq!(tensor.value.len(), 0),
                    _ => panic!("sort must return tensors"),
                }
            }
        }
        _ => panic!("sort must return a tuple"),
    }
}

#[test]
fn host_runtime_einsum_rejects_non_lowercase_labels() {
    let operand = RuntimeTensorValue {
        value: IrTensorValue::from_vec(vec![1], vec![1.0]),
        precision: Prim::F32,
    };
    for equation in ["I,I->", "_,i->"] {
        assert!(
            tensor_einsum_value(equation, &operand, &operand).is_err(),
            "host einsum accepted non-lowercase equation `{equation}`"
        );
    }
}

#[test]
fn host_runtime_einsum_rejects_duplicate_output_labels() {
    let operand = RuntimeTensorValue {
        value: IrTensorValue::from_vec(vec![1], vec![1.0]),
        precision: Prim::F32,
    };
    assert!(
        tensor_einsum_value("i,i->ii", &operand, &operand).is_err(),
        "host einsum accepted a duplicate output label"
    );
}

// PR #168 review LOW #5: NaN/Inf/n<4 edge-case coverage for the
// stride-4 ILP cascade. Tail handling (n < 4 where the lane-fill
// doesn't complete a full cycle) and special-value propagation
// are load-bearing invariants of the cascade; without these
// tests, a future change to the tail loop or lane combine could
// silently regress them.

/// Stride-4 reference for n < 4: the lanes are assigned naturally
/// (acc0=x[0], acc1=x[1], acc2=x[2]); the combine is
/// `(acc0 + acc1) + (acc2 + 0.0)` which equals a left-fold for
/// n ≤ 3. The expected bit pattern is therefore the straight-
/// forward sum.
#[test]
fn host_runtime_sum_f32_n1_n2_n3_bit_exact() {
    for (n, expr, expected) in [
        (1, "to_tensor([cast(1.5, f32)])", 1.5_f32),
        (2, "to_tensor([cast(1.5, f32), cast(0.25, f32)])", 1.75_f32),
        (
            3,
            "to_tensor([cast(1.5, f32), cast(0.25, f32), cast(0.125, f32)])",
            1.875_f32,
        ),
    ] {
        let src = format!(
            "seq = {expr}\n\
             y = sum(seq, cast(0, int32))\n"
        );
        let checked = checked_surf(&src);
        let outcome = evaluate_host_program(&checked, &HashMap::new())
            .unwrap_or_else(|_| panic!("n={n} sum should evaluate"));
        let result = first_tensor_data(&outcome, "y");
        assert_eq!(result.len(), 1);
        assert_eq!(
            (result[0] as f32).to_bits(),
            expected.to_bits(),
            "n={n}: expected {expected}, got {}",
            result[0]
        );
    }
}

#[test]
fn host_runtime_sum_f32_propagates_nan() {
    // A single NaN anywhere in the input must propagate to the
    // final result. Pinned bit-exactly so a future change to the
    // lane combine that hides NaN through e.g. min/max can't slip
    // by.
    let checked = checked_surf(
        r#"
seq = to_tensor([
cast(1.0, f32),
cast(2.0, f32),
cast(0.0, f32) / cast(0.0, f32),
cast(4.0, f32),
cast(5.0, f32)
])
y = sum(seq, cast(0, int32))
"#,
    );
    let outcome =
        evaluate_host_program(&checked, &HashMap::new()).expect("nan-bearing sum should evaluate");
    let result = first_tensor_data(&outcome, "y");
    assert_eq!(result.len(), 1);
    assert!(
        (result[0] as f32).is_nan(),
        "sum with NaN must propagate NaN; got {}",
        result[0]
    );
}

#[test]
fn host_runtime_sum_f32_inf_plus_neg_inf_is_nan() {
    // +Inf + -Inf is IEEE-754 NaN. The stride-4 cascade must
    // produce this regardless of which lanes the two infinities
    // land in (`x[0]` and `x[1]` here land in acc0/acc1; under
    // stride-4 the cascade still adds them and the result is NaN).
    let checked = checked_surf(
        r#"
seq = to_tensor([
cast(1.0, f32) / cast(0.0, f32),
cast(-1.0, f32) / cast(0.0, f32),
cast(2.0, f32),
cast(3.0, f32)
])
y = sum(seq, cast(0, int32))
"#,
    );
    let outcome =
        evaluate_host_program(&checked, &HashMap::new()).expect("inf-pair sum should evaluate");
    let result = first_tensor_data(&outcome, "y");
    assert_eq!(result.len(), 1);
    assert!(
        (result[0] as f32).is_nan(),
        "sum with +Inf and -Inf must produce NaN; got {}",
        result[0]
    );
}

#[test]
fn host_runtime_matmul_shared_axis_mismatch_errors() {
    // Build a 2x3 and a 2x2 — shared axis is 3 vs 2, must fail.
    let checked = checked_surf(
        r#"
a = pad_sequences_to([[cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)], [cast(4.0, f32), cast(5.0, f32), cast(6.0, f32)]], cast(3, int64), cast(0.0, f32))
b = pad_sequences_to([[cast(1.0, f32), cast(0.0, f32)], [cast(0.0, f32), cast(1.0, f32)]], cast(2, int64), cast(0.0, f32))
y = matmul(a, b)
"#,
    );
    let err = evaluate_host_program(&checked, &HashMap::new())
        .expect_err("matmul shared-axis mismatch must fail");
    assert!(
        err.contains("matmul") && err.contains("mismatch"),
        "expected matmul shared-axis diagnostic, got: {err}"
    );
}

// ----- expand / softmax host evaluator coverage (#38 follow-up) -----
//
// Pins the closure of the second host-runtime gap from the N2 fix:
// `chelis test` / `chelis eval` erroring with `unsupported builtin
// 'expand'` / `'softmax'` when those primitives appear in a test's
// dependency graph (School.Nn.Linear, School.Nn.Attention, School.Loss.CrossEntropy).

#[test]
fn host_runtime_expand_inserts_new_leading_axis() {
    // Linear.forward calls `expand(b, 0, batch)` where `b` is a 1-D
    // bias [out_dim] and the output is [batch, out_dim]. Pin that.
    let checked = checked_surf(
        r#"
b = to_tensor([cast(10.0, f32), cast(100.0, f32)])
y = expand(b, cast(0, int32), cast(3, int64))
"#,
    );
    let outcome = evaluate_host_program(&checked, &HashMap::new())
        .expect("expand should evaluate under host runtime");
    assert_eq!(first_tensor_shape(&outcome, "y"), vec![3, 2]);
    // Three replicas of [10, 100].
    assert_eq!(
        first_tensor_data(&outcome, "y"),
        vec![10.0, 100.0, 10.0, 100.0, 10.0, 100.0]
    );
}

#[test]
fn host_runtime_expand_inserts_trailing_axis() {
    // axis == rank inserts a new last axis.
    let checked = checked_surf(
        r#"
b = to_tensor([cast(1.0, f32), cast(2.0, f32)])
y = expand(b, cast(1, int32), cast(2, int64))
"#,
    );
    let outcome = evaluate_host_program(&checked, &HashMap::new())
        .expect("expand at trailing axis should evaluate");
    assert_eq!(first_tensor_shape(&outcome, "y"), vec![2, 2]);
    // [1, 2] expanded along new last axis with count 2 -> [[1,1],[2,2]].
    assert_eq!(first_tensor_data(&outcome, "y"), vec![1.0, 1.0, 2.0, 2.0]);
}

#[test]
fn host_runtime_expand_singleton_input_inserts_not_replicates() {
    // Bucket 4a regression: `expand(b: tensor[1, f32], 0, count)` must
    // produce shape `[count, 1]` (INSERT semantics), matching the
    // typer's first-preference branch in
    // `chelis-types::infer::check_expand_signature`. Previously the
    // host runtime detected `in_shape[axis] == 1` and silently
    // replicated the singleton in-place, producing `[count]` and
    // diverging from `chelis check` on `examples/linreg.ch`.
    let checked = checked_surf(
        r#"
b = to_tensor([cast(7.0, f32)])
y = expand(b, cast(0, int32), cast(4, int64))
"#,
    );
    let outcome = evaluate_host_program(&checked, &HashMap::new())
        .expect("expand([1], 0, 4) should evaluate under host runtime");
    assert_eq!(
        first_tensor_shape(&outcome, "y"),
        vec![4, 1],
        "INSERT semantics: rank-1 [1] expand at axis 0 with count 4 must produce rank-2 [4, 1]",
    );
    assert_eq!(first_tensor_data(&outcome, "y"), vec![7.0, 7.0, 7.0, 7.0]);
}

#[test]
fn host_runtime_expand_negative_count_errors() {
    // PR #214 / red team round 3 sibling sweep: `infer_expand_app`
    // now extracts cast-wrapped int literals via `extract_int_for_dim`
    // and rejects `cast(0, int32)` at infer time. To keep this test
    // exercising the host-runtime arm (defense in depth for direct-DAG
    // callers and any non-literal size that evaluates to 0 at runtime),
    // the count is built from arithmetic that the infer-time literal
    // extractor cannot resolve.
    let checked = checked_surf(
        r#"
b = to_tensor([cast(1.0, f32), cast(2.0, f32)])
zero_count = sub(cast(0, int64), cast(0, int64))
y = expand(b, cast(0, int32), zero_count)
"#,
    );
    let err = evaluate_host_program(&checked, &HashMap::new())
        .expect_err("expand with non-positive count must fail");
    assert!(
        err.contains("expand") && err.contains("count"),
        "expected expand count diagnostic, got: {err}"
    );
}

// ----- to_tensor nested-list (Bucket 4b) -----

#[test]
fn host_runtime_to_tensor_accepts_2d_float_literal() {
    // Bucket 4b: previously rejected with
    // "to_tensor expects numeric or bool List elements, got List f32".
    let checked = checked_surf(
        r#"
y = to_tensor([[cast(1.0, f32), cast(2.0, f32)], [cast(3.0, f32), cast(4.0, f32)]])
"#,
    );
    let outcome = evaluate_host_program(&checked, &HashMap::new())
        .expect("to_tensor of [[1,2],[3,4]] must evaluate to a rank-2 tensor");
    assert_eq!(first_tensor_shape(&outcome, "y"), vec![2, 2]);
    assert_eq!(first_tensor_data(&outcome, "y"), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn host_runtime_to_tensor_accepts_3d_float_literal() {
    // 2x2x2 cube — exercises 3-deep recursion in
    // `nested_list_to_tensor_data`.
    let checked = checked_surf(
        r#"
y = to_tensor([
  [[cast(1.0, f32), cast(2.0, f32)], [cast(3.0, f32), cast(4.0, f32)]],
  [[cast(5.0, f32), cast(6.0, f32)], [cast(7.0, f32), cast(8.0, f32)]]
])
"#,
    );
    let outcome = evaluate_host_program(&checked, &HashMap::new())
        .expect("to_tensor of 2x2x2 nested list must evaluate to a rank-3 tensor");
    assert_eq!(first_tensor_shape(&outcome, "y"), vec![2, 2, 2]);
    assert_eq!(
        first_tensor_data(&outcome, "y"),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    );
}

#[test]
fn host_runtime_to_tensor_rejects_ragged_2d_literal() {
    // Negative parity for 4b: ragged inner-list shapes must error
    // out at the host runtime, not silently produce a malformed
    // tensor.
    let checked = checked_surf(
        r#"
y = to_tensor([[cast(1.0, f32), cast(2.0, f32)], [cast(3.0, f32)]])
"#,
    );
    let err = evaluate_host_program(&checked, &HashMap::new())
        .expect_err("ragged nested list must fail to_tensor");
    assert!(
        err.contains("uniform inner shape"),
        "expected ragged-shape diagnostic, got: {err}"
    );
}

#[test]
fn host_runtime_softmax_uniform_input_is_uniform_output() {
    // softmax of all-zeros along axis 0 of length 3 is [1/3, 1/3, 1/3].
    let checked = checked_surf(
        r#"
x = to_tensor([cast(0.0, f32), cast(0.0, f32), cast(0.0, f32)])
y = softmax(x, cast(0, int32))
"#,
    );
    let outcome = evaluate_host_program(&checked, &HashMap::new())
        .expect("softmax should evaluate under host runtime");
    assert_eq!(first_tensor_shape(&outcome, "y"), vec![3]);
    let data = first_tensor_data(&outcome, "y");
    for value in &data {
        // chelis#729 Phase 1: the f32-typed softmax result finalizes at
        // f32, so each element is exactly the f32 image of 1/3.
        assert_eq!(
            *value,
            (1.0f32 / 3.0f32) as f64,
            "uniform softmax element should be f32(1/3), got {value}"
        );
    }
}

#[test]
fn host_runtime_softmax_two_class_matches_reference() {
    // softmax([1.0, 0.0], 0) = [exp(1)/(exp(1)+1), 1/(exp(1)+1)]
    //                         ≈ [0.7310585, 0.2689414]
    let checked = checked_surf(
        r#"
x = to_tensor([cast(1.0, f32), cast(0.0, f32)])
y = softmax(x, cast(0, int32))
"#,
    );
    let outcome =
        evaluate_host_program(&checked, &HashMap::new()).expect("softmax 2-class should evaluate");
    assert_eq!(first_tensor_shape(&outcome, "y"), vec![2]);
    let data = first_tensor_data(&outcome, "y");
    let e = 1.0_f64.exp();
    let expected = [e / (e + 1.0), 1.0 / (e + 1.0)];
    for (got, want) in data.iter().zip(expected.iter()) {
        assert!(
            (*got - *want).abs() < 1e-6,
            "softmax 2-class mismatch: got {got}, want {want}"
        );
    }
}

#[test]
fn host_runtime_softmax_numerical_stability_handles_large_inputs() {
    // Without the max-subtraction trick, exp(1000) would overflow to
    // inf and produce NaN. The stable lowering must still produce
    // a normalized distribution.
    let checked = checked_surf(
        r#"
x = to_tensor([cast(1000.0, f32), cast(1000.0, f32)])
y = softmax(x, cast(0, int32))
"#,
    );
    let outcome = evaluate_host_program(&checked, &HashMap::new())
        .expect("softmax with large inputs should remain numerically stable");
    let data = first_tensor_data(&outcome, "y");
    assert_eq!(data.len(), 2);
    for value in &data {
        assert!(
            (*value - 0.5).abs() < 1e-9,
            "softmax([1000,1000]) should be uniform 0.5, got {value}"
        );
    }
}

#[test]
fn host_runtime_softmax_positive_infinity_yields_nan_like_torch() {
    // #173: `softmax([+Inf, 0, 0])` must return all-NaN, matching torch
    // (verified: torch 2.x CPU `torch.softmax([inf,1,2]) == [nan,nan,nan]`).
    // The standard max-shift formula computes `exp(+Inf - +Inf) = exp(NaN)
    // = NaN`, which is what the C backend, IR evaluator, and spec'd
    // lowering all already produce. The host runtime previously
    // special-cased +Inf to a `1/K` one-hot ("natural limit"), silently
    // diverging from torch and every other Chelis lane; that special-case
    // is removed.
    let inf = f64::INFINITY;
    let tensor = RuntimeTensorValue {
        value: IrTensorValue::from_vec(vec![3], vec![inf, 0.0, 0.0]),
        precision: Prim::F32,
    };
    let out = tensor_softmax_host(&tensor, 0).expect("+Inf softmax must not error in host runtime");
    assert_eq!(out.value.len(), 3);
    assert!(
        out.value.to_f64_lossy_vec().iter().all(|v| v.is_nan()),
        "softmax of a slice containing +Inf must be all-NaN (torch parity, #173); got {:?}",
        out.value.to_f64_lossy_vec()
    );
}

#[test]
fn host_runtime_softmax_two_positive_infinities_yield_nan_like_torch() {
    // #173: torch returns all-NaN for any +Inf-containing slice, including
    // multiple +Inf (`torch.softmax([inf,inf,2]) == [nan,nan,nan]`). The
    // prior `1/K`-split special-case is removed.
    let inf = f64::INFINITY;
    let tensor = RuntimeTensorValue {
        value: IrTensorValue::from_vec(vec![3], vec![inf, inf, 0.0]),
        precision: Prim::F32,
    };
    let out = tensor_softmax_host(&tensor, 0).expect("two +Inf softmax must not error");
    assert!(
        out.value.to_f64_lossy_vec().iter().all(|v| v.is_nan()),
        "softmax of a slice with multiple +Inf must be all-NaN (torch parity, #173); got {:?}",
        out.value.to_f64_lossy_vec()
    );
}

#[test]
fn host_runtime_softmax_all_negative_infinity_yields_nan_like_torch() {
    // #173: an all-`-Inf` slice yields `exp(-Inf - -Inf) = exp(NaN) = NaN`
    // under the standard formula, which is what torch returns
    // (`torch.softmax([-inf,-inf,-inf]) == [nan,nan,nan]`) and what the C
    // backend / IR evaluator already produce. The host runtime previously
    // special-cased this to uniform `1/N`; that special-case is removed.
    let neg_inf = f64::NEG_INFINITY;
    let tensor = RuntimeTensorValue {
        value: IrTensorValue::from_vec(vec![3], vec![neg_inf, neg_inf, neg_inf]),
        precision: Prim::F32,
    };
    let out = tensor_softmax_host(&tensor, 0).expect("all -Inf softmax must not error");
    assert!(
        out.value.to_f64_lossy_vec().iter().all(|v| v.is_nan()),
        "softmax of an all-(-Inf) slice must be all-NaN (torch parity, #173); got {:?}",
        out.value.to_f64_lossy_vec()
    );
}

#[test]
fn host_runtime_softmax_mixed_negative_infinity_is_finite_like_torch() {
    // #173 positive control: a slice with `-Inf` but NO `+Inf` has a
    // finite max, so `exp(-Inf - max) = 0` at the masked position and the
    // standard formula produces a valid (non-NaN) distribution. This is
    // the legitimate attention-masking case and must NOT regress to NaN.
    // torch: `torch.softmax([1, -inf, 2]) == [0.2689, 0.0, 0.7311]`.
    let neg_inf = f64::NEG_INFINITY;
    let tensor = RuntimeTensorValue {
        value: IrTensorValue::from_vec(vec![3], vec![1.0, neg_inf, 2.0]),
        precision: Prim::F32,
    };
    let out = tensor_softmax_host(&tensor, 0).expect("mixed -Inf softmax must not error");
    assert!(
        out.value.to_f64_lossy_vec().iter().all(|v| !v.is_nan()),
        "mixed -Inf (no +Inf) softmax must be finite, not NaN; got {:?}",
        out.value.to_f64_lossy_vec()
    );
    assert!(
        out.value.to_f64_lossy_vec()[1].abs() < 1e-9,
        "the -Inf position must be 0"
    );
    assert!(
        (out.value.to_f64_lossy_vec()[0] - 0.268_941_4).abs() < 1e-5
            && (out.value.to_f64_lossy_vec()[2] - 0.731_058_6).abs() < 1e-5,
        "mixed -Inf softmax must match torch [0.2689, 0, 0.7311]; got {:?}",
        out.value.to_f64_lossy_vec()
    );
}

#[test]
fn host_runtime_softmax_nan_input_propagates_nan() {
    // NaN is contagious by spec; matches PyTorch / NumPy / JAX behavior.
    // We pin this so a future refactor doesn't accidentally mask it.
    let nan = f64::NAN;
    let tensor = RuntimeTensorValue {
        value: IrTensorValue::from_vec(vec![3], vec![nan, 0.0, 0.0]),
        precision: Prim::F32,
    };
    let out = tensor_softmax_host(&tensor, 0).expect("NaN softmax does not error");
    assert!(
        out.value.to_f64_lossy_vec().iter().all(|v| v.is_nan()),
        "NaN must propagate to every output element; got {:?}",
        out.value.to_f64_lossy_vec()
    );
}

#[test]
fn host_runtime_softmax_axis_out_of_bounds_errors() {
    // Issue #216: cast-wrapped out-of-bounds softmax axis is now
    // caught at infer time (the checker peels the `cast(N, int32)`
    // wrapper via `extract_int_for_dim` and applies the rank-bounds
    // check). Pre-fix the cast hid the literal from
    // `extract_int_literal` and the rejection only fired in the
    // host-runtime defense-in-depth layer. The user-facing contract
    // is unchanged (the program is still rejected); only the layer
    // emitting the diagnostic moved upstream.
    let src = r#"
x = to_tensor([cast(1.0, f32), cast(2.0, f32)])
y = softmax(x, cast(5, int32))
"#;
    let res = chelis_types::check_ir_program(&chelis_surf::desugar::desugar_program(
        &chelis_surf::parser::parse_str(src).expect("surf parse"),
    ));
    let infer_err = res.expect_err("softmax with out-of-bounds axis must fail infer-time check");
    let joined = infer_err
        .errors
        .iter()
        .map(|e| e.message.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        joined.contains("softmax") && joined.contains("out of bounds"),
        "expected softmax axis-bounds diagnostic, got: {joined}"
    );
}

// ----------------------------------------------------------------
// #172: Max/Min reductions PROPAGATE NaN, matching torch
// (`torch.max`/`torch.min` of any NaN-containing slice return NaN, at
// every position). A naive `value > best` drops NaN, which made the C
// backend's SIMD `chelis_max_f32` position-dependent and diverged from
// torch on both lanes. The eval lane is the reference, so it must agree.
// ----------------------------------------------------------------

#[test]
fn host_runtime_max_reduce_propagates_nan_like_torch() {
    let nan = f64::NAN;
    // NaN at each position must still yield NaN (position-independent).
    for pos in 0..3 {
        let mut data = vec![1.0, 2.0, 3.0];
        data[pos] = nan;
        let tensor = RuntimeTensorValue {
            value: IrTensorValue::from_vec(vec![3], data),
            precision: Prim::F32,
        };
        let out = tensor_reduce_host(&tensor, 0, ReduceOp::Max).expect("max reduce");
        assert!(
            out.value.to_f64_lossy_vec()[0].is_nan(),
            "max_reduce of a slice with NaN@{pos} must be NaN (torch parity, #172); got {:?}",
            out.value.to_f64_lossy_vec()
        );
    }
}

#[test]
fn host_runtime_min_reduce_propagates_nan_like_torch() {
    let nan = f64::NAN;
    for pos in 0..3 {
        let mut data = vec![1.0, 2.0, 3.0];
        data[pos] = nan;
        let tensor = RuntimeTensorValue {
            value: IrTensorValue::from_vec(vec![3], data),
            precision: Prim::F32,
        };
        let out = tensor_reduce_host(&tensor, 0, ReduceOp::Min).expect("min reduce");
        assert!(
            out.value.to_f64_lossy_vec()[0].is_nan(),
            "min_reduce of a slice with NaN@{pos} must be NaN (torch parity, #172); got {:?}",
            out.value.to_f64_lossy_vec()
        );
    }
}

#[test]
fn host_runtime_max_reduce_no_nan_is_unchanged() {
    // POSITIVE control: a NaN-free slice still reduces normally — the
    // NaN-propagation fix must not perturb ordinary max/min.
    let tensor = RuntimeTensorValue {
        value: IrTensorValue::from_vec(vec![3], vec![1.0, 3.0, 2.0]),
        precision: Prim::F32,
    };
    let max = tensor_reduce_host(&tensor, 0, ReduceOp::Max).expect("max reduce");
    let min = tensor_reduce_host(&tensor, 0, ReduceOp::Min).expect("min reduce");
    assert_eq!(max.value.to_f64_lossy_vec(), vec![3.0]);
    assert_eq!(min.value.to_f64_lossy_vec(), vec![1.0]);
}

// ----------------------------------------------------------------
// WS-A0 acceptance test (c): the dtype-tagged Scalar payload must
// reject mismatched (dtype, bits) pairs at construction time. The
// public constructor `RuntimeValue::scalar` is the only checked
// path; the typed convenience constructors (`int_lit`, `float_lit`,
// `int64`, `float64`, `scalar_like_*`) are implementation-internal
// and statically-correct by construction.
// ----------------------------------------------------------------

#[test]
fn scalar_construction_is_sealed_and_width_checked() {
    // chelis#729 Phase 1 re-authoring: the old dtype/bits mismatch this
    // test enumerated is now UNREPRESENTABLE - the sealed ScalarValue's
    // storage variant IS the dtype, and the only constructors are the
    // dtype-semantics module's finalize/ingress chokepoints. What remains
    // testable at this level is the width discipline those constructors
    // enforce: out-of-width integers trap instead of wrapping, and every
    // constructed scalar reports the dtype it was finalized at.
    let trapped = RuntimeValue::scalar_like_int(Prim::Int8, 200)
        .expect_err("int8 cannot hold 200; finalize must trap, not wrap");
    assert!(
        trapped.contains("overflow"),
        "the trap must carry the branded overflow diagnostic, got: {trapped}"
    );
    let ok = RuntimeValue::scalar_like_int(Prim::Int8, 127).expect("127 fits int8");
    match ok {
        RuntimeValue::Scalar(payload) => assert_eq!(payload.dtype(), Prim::Int8),
        other => panic!("expected RuntimeValue::Scalar, got {other:?}"),
    }
}

// ----------------------------------------------------------------
// RT-1 finding (C1), history: the original RT-1 test exercised
// `RuntimeValue::Scalar { dtype: F16, bits: ScalarBits::F32(_) }`
// directly, then C1 sealed the payload so the literal no longer
// compiled, and chelis#729 Phase 1 removed the (dtype, bits) pair
// entirely - the storage variant IS the dtype.
// ----------------------------------------------------------------

/// RT-1 closed finding C1, strengthened by chelis#729 Phase 1: the
/// dtype/bits mismatch the pre-fix red team constructed is now
/// UNREPRESENTABLE, not merely rejected. `RuntimeValue::Scalar` wraps the
/// sealed `chelis_types::ScalarValue`, whose storage variant IS the
/// dtype; there is no (dtype, bits) pair to disagree, and the only
/// constructors are the dtype-semantics module's finalize/ingress
/// chokepoints. This pin documents the strengthening and asserts the
/// module constructor reports the dtype it stored at.
#[test]
fn scalar_payload_dtype_is_the_storage_variant() {
    let value =
        chelis_types::scalar_from_f64("test", Prim::F16, 1.5).expect("1.5 finalizes at f16");
    assert_eq!(value.prim(), Prim::F16);
    let payload = match RuntimeValue::from_scalar_value(value) {
        RuntimeValue::Scalar(payload) => payload,
        other => panic!("expected RuntimeValue::Scalar, got {other:?}"),
    };
    assert_eq!(payload.dtype(), Prim::F16);
    assert_eq!(payload.as_f64_lossy(), 1.5);
}

fn numeric_scalar(prim: Prim, integer: i64, float: f64) -> RuntimeValue {
    if prim.is_integer() {
        RuntimeValue::from_scalar_value(
            chelis_types::scalar_from_i64("test", prim, integer)
                .expect("test integer fits its dtype"),
        )
    } else {
        RuntimeValue::from_scalar_value(
            chelis_types::scalar_from_f64("test", prim, float)
                .expect("float finalization is total"),
        )
    }
}

/// chelis#729 Phase 1, dtype-semantics C3: scalar values nested in the
/// execution wire are dtype-tagged carriers too. Lists, tuples, and ADTs may
/// not silently widen all integer values to int64 or all floats to f64.
#[test]
fn execution_wire_nested_numeric_scalars_keep_their_dtype_tags() {
    let cases = [
        (Prim::Int8, "int8"),
        (Prim::Int16, "int16"),
        (Prim::Int32, "int32"),
        (Prim::Int64, "int64"),
        (Prim::F16, "float16"),
        (Prim::Bf16, "bfloat16"),
        (Prim::F32, "float32"),
        (Prim::F64, "float64"),
    ];

    for (prim, expected_tag) in cases {
        let scalar = numeric_scalar(prim, 7, 1.5);
        let containers = [
            RuntimeValue::List(vec![scalar.clone()]),
            RuntimeValue::Tuple(vec![scalar.clone()]),
            RuntimeValue::Dict(vec![(
                RuntimeValue::String("value".to_string()),
                scalar.clone(),
            )]),
            RuntimeValue::Adt {
                ctor: "Boxed".to_string(),
                fields: vec![scalar],
                field_names: Some(vec!["value".to_string()]),
            },
        ];

        for container in containers {
            let encoded = runtime_value_to_schema(&container).expect("wire encode");
            let json = serde_json::to_value(encoded).expect("serialize execution value");
            let nested = json
                .get("value")
                .and_then(|value| value.as_array())
                .and_then(|values| values.first())
                .or_else(|| {
                    json.get("fields")
                        .and_then(|value| value.as_array())
                        .and_then(|values| values.first())
                })
                .or_else(|| {
                    json.get("entries")
                        .and_then(|value| value.as_array())
                        .and_then(|entries| entries.first())
                        .and_then(|entry| entry.get("value"))
                })
                .expect("container has one nested scalar");
            assert_eq!(
                nested.get("type").and_then(|value| value.as_str()),
                Some(expected_tag),
                "nested numeric scalar must keep its own wire tag: {json}"
            );
        }
    }
}

/// Positive width matrix for the list bridges covered by Phase 1 eval
/// adoption. The checker derives the exact element dtype, so runtime
/// construction must preserve it for `to_tensor`, `pad_sequences`, and
/// `pad_sequences_to`.
#[test]
fn list_tensor_bridges_preserve_every_numeric_dtype() {
    let dtypes = [
        Prim::Int8,
        Prim::Int16,
        Prim::Int32,
        Prim::Int64,
        Prim::F16,
        Prim::Bf16,
        Prim::F32,
        Prim::F64,
    ];

    for prim in dtypes {
        let value = numeric_scalar(prim, 7, if prim == Prim::F64 { 1e100 } else { 1.5 });
        let expected_float = if prim == Prim::F64 { 1e100 } else { 1.5 };
        let (tensor_prim, _, tensor_data) =
            nested_list_to_tensor_data(std::slice::from_ref(&value))
                .expect("to_tensor list ingress");
        assert_eq!(tensor_prim, prim, "to_tensor must preserve the input dtype");
        match tensor_data {
            ListTensorData::Int(values) => assert_eq!(values, vec![7]),
            ListTensorData::Float(values) => assert_eq!(values, vec![expected_float]),
        }

        let sequences = [RuntimeValue::List(vec![value.clone()])];
        let (padded_prim, padded_data, _, _) =
            pad_sequences_value(&sequences, &value).expect("pad_sequences ingress");
        assert_eq!(
            padded_prim, prim,
            "pad_sequences must preserve the input dtype"
        );
        match padded_data {
            ListTensorData::Int(values) => assert_eq!(values, vec![7]),
            ListTensorData::Float(values) => assert_eq!(values, vec![expected_float]),
        }

        let (padded_to_prim, padded_to_data, _) =
            pad_sequences_to_value(&sequences, 2, &value).expect("pad_sequences_to ingress");
        assert_eq!(
            padded_to_prim, prim,
            "pad_sequences_to must preserve the input dtype"
        );
        match padded_to_data {
            ListTensorData::Int(values) => assert_eq!(values, vec![7, 7]),
            ListTensorData::Float(values) => {
                assert_eq!(values, vec![expected_float, expected_float])
            }
        }
    }
}

/// Negative-test parity for the exact list bridge: same-family dtypes are
/// still heterogeneous. Accepting int8 beside int16 or f64 beside f32 would
/// perform an implicit cast that the checker never authorized.
#[test]
fn list_tensor_bridges_reject_same_family_dtype_substitution() {
    let int8 = numeric_scalar(Prim::Int8, 7, 0.0);
    let int16 = numeric_scalar(Prim::Int16, 7, 0.0);
    assert!(
        nested_list_to_tensor_data(&[int8.clone(), int16.clone()]).is_err(),
        "to_tensor must reject heterogeneous integer widths"
    );
    assert!(
        pad_sequences_value(&[RuntimeValue::List(vec![int8])], &int16).is_err(),
        "pad_sequences must reject a different integer pad dtype"
    );

    let f64_value = numeric_scalar(Prim::F64, 0, 1e100);
    let f32_pad = numeric_scalar(Prim::F32, 0, 0.0);
    assert!(
        pad_sequences_to_value(&[RuntimeValue::List(vec![f64_value])], 2, &f32_pad).is_err(),
        "pad_sequences_to must reject a different float pad dtype"
    );
}

/// E2 (WS-A0 RT-1 fixup): `prim_from_name` must panic on the
/// `"f8e4m3"` token with the §1.1.1 message rather than producing
/// `Some(Prim::F8e4m3)` and letting the deferred dtype flow into
/// runtime classification.
#[test]
#[should_panic(expected = "f8e4m3 is deferred per spec/04-type-system.md §1.1.1")]
fn prim_from_name_panics_on_f8e4m3_per_spec_1_1_1() {
    let _ = prim_from_name("f8e4m3");
}

/// Negative-parity twin: the active dtype names still resolve.
#[test]
fn prim_from_name_resolves_active_dtype_names() {
    for (name, expected) in &[
        ("f32", Prim::F32),
        ("f64", Prim::F64),
        ("f16", Prim::F16),
        ("bf16", Prim::Bf16),
        ("int8", Prim::Int8),
        ("int16", Prim::Int16),
        ("int32", Prim::Int32),
        ("int64", Prim::Int64),
        ("bool", Prim::Bool),
    ] {
        assert_eq!(
            prim_from_name(name),
            Some(*expected),
            "active dtype name `{name}` must still resolve"
        );
    }
}

/// E1 (WS-A0 RT-1 fixup): the old float-binop fallback silently
/// re-precisioned mixed narrow floats to F32. The type checker must reject
/// `(Bf16, F16)` and `(F16, Bf16)` per spec/04-type-system.md §5.1, while
/// the closed dtype kernel is the runtime backstop.
#[test]
fn type_checker_rejects_mixed_narrow_float_binop_per_spec_5_1() {
    let src = r#"
def main() -> bf16 = add(cast(1.0, bf16), cast(1.0, f16))
"#;
    let res = chelis_types::check_ir_program(&chelis_surf::desugar::desugar_program(
        &chelis_surf::parser::parse_str(src).expect("surf parse"),
    ));
    assert!(
        res.is_err(),
        "spec §5.1 forbids implicit precision promotion; \
         `add(_:bf16, _:f16)` must be rejected by the type checker"
    );
}

#[test]
fn scalar_constructor_accepts_every_active_numeric_dtype() {
    // Sanity sibling (re-authored for chelis#729 Phase 1): every active
    // numeric dtype constructs through the module's ingress constructors
    // and reports itself back from the sealed storage.
    for (dtype, value) in [
        (Prim::Int8, 7.0),
        (Prim::Int16, 123.0),
        (Prim::Int32, -42.0),
        (Prim::Int64, 1_000_000.0),
        (Prim::F32, 1.5),
        (Prim::F64, 2.5),
        (Prim::F16, 0.25),
        (Prim::Bf16, 0.25),
    ] {
        let v = RuntimeValue::scalar_like_float(dtype, value)
            .unwrap_or_else(|e| panic!("{} must construct: {e}", dtype.name()));
        match v {
            RuntimeValue::Scalar(payload) => {
                assert_eq!(payload.dtype(), dtype);
                assert_eq!(payload.as_f64_lossy(), value);
            }
            other => panic!("expected RuntimeValue::Scalar, got {other:?}"),
        }
    }
}

#[test]
fn integer_spelled_f32_scalar_finalizes_without_f64_intermediate() {
    let checked = checked_surf(
        r#"
def probe() -> f32 = sub(
  cast(18014399583223809, f32),
  cast(18014398509481984, f32)
)
root = probe()
"#,
    );
    let outcome = evaluate_host_program(&checked, &HashMap::new())
        .expect("integer-spelled f32 scalar must evaluate");
    let root = outcome.host_bindings.get("root").expect("root binding");
    let RuntimeValue::Scalar(payload) = root else {
        panic!("expected scalar root");
    };
    assert_eq!(payload.dtype(), Prim::F32);
    assert_eq!(
        (payload.as_f64_lossy() as f32).to_bits(),
        0x4F00_0000,
        "single rounding keeps the low-side operand one f32 ulp above the control; double rounding produces zero"
    );
}

// ===========================================================================
// chelis#732 Phase 1: render_value's [05-OBS] behavior at the unit level.
// The end-to-end exits are locked by the CLI observation harness; these pin
// the renderer's edge policy directly.
// ===========================================================================

fn tensor_value(precision: Prim, shape: Vec<usize>, data: Vec<f64>) -> RuntimeValue {
    RuntimeValue::Tensor(
        RuntimeTensorValue::from_wide("test", precision, shape, data)
            .expect("test fixtures carry in-domain values"),
    )
}

/// Integer/bool tensor elements render per their tag's class
/// ([05-OBS-2]): integers lose the `.0`, bools print true/false.
#[test]
fn render_value_tensor_elements_follow_tag_class() {
    assert_eq!(
        render_value(&tensor_value(Prim::Int8, vec![3], vec![127.0, -127.0, 0.0])),
        "tensor(shape=[3], data=[127, -127, 0])"
    );
    assert_eq!(
        render_value(&tensor_value(Prim::Bool, vec![2], vec![1.0, 0.0])),
        "tensor(shape=[2], data=[true, false])"
    );
    // Float elements render at the stored f64 width (the chelis#717 width
    // note, spec/05 section 8.1).
    assert_eq!(
        render_value(&tensor_value(
            Prim::F64,
            vec![2],
            vec![0.30000000000000004, -0.0]
        )),
        "tensor(shape=[2], data=[0.30000000000000004, -0.0])"
    );
}

/// Tag-vs-bits disagreements print the BITS (spec/05 section 8.1): an
/// int64-tagged slot holding 187.5 (the live `mean`-of-int64 example,
/// chelis#724 territory) renders 187.5, never a truncated 187; a
/// bool-tagged slot holding 2.0 renders 2.0, never `true`; an
/// int8-tagged slot holding 400 renders 400.0's bits faithfully rather
/// than a wrapped/saturated lie.
#[test]
fn tag_storage_disagreement_is_unconstructible() {
    // chelis#729 Phase 1 re-authoring: this test used to pin FAITHFUL
    // MISREPORTING - an integer/bool-tagged buffer holding out-of-domain
    // f64 bits rendered the stored bits. Per-dtype sealed storage makes
    // that state unrepresentable: the construction chokepoint traps at
    // finalize instead, so nothing out-of-domain can reach the renderer.
    let trapped = RuntimeTensorValue::from_wide("test", Prim::Int64, vec![1], vec![187.5])
        .expect_err("a fractional value in an int64 buffer must Domain-trap");
    assert!(
        trapped.contains("domain"),
        "branded domain trap, got: {trapped}"
    );
    let trapped = RuntimeTensorValue::from_wide("test", Prim::Bool, vec![1], vec![2.0])
        .expect_err("2.0 in a bool buffer must Domain-trap");
    assert!(
        trapped.contains("domain"),
        "branded domain trap, got: {trapped}"
    );
    let trapped = RuntimeTensorValue::from_wide("test", Prim::Int8, vec![1], vec![400.0])
        .expect_err("400 in an int8 buffer must Overflow-trap");
    assert!(
        trapped.contains("overflow"),
        "branded overflow trap, got: {trapped}"
    );
}

/// [05-OBS-4]: a rank-0 tensor renders as its single element, bare, at
/// the renderer level - the wrapper is not an exit form.
#[test]
fn render_value_rank_zero_tensor_renders_bare() {
    assert_eq!(
        render_value(&tensor_value(Prim::F64, vec![], vec![0.1])),
        "0.1"
    );
    assert_eq!(
        render_value(&tensor_value(Prim::Int32, vec![], vec![7.0])),
        "7"
    );
    assert_eq!(
        render_value(&tensor_value(Prim::Bool, vec![], vec![1.0])),
        "true"
    );
}

/// [05-OBS-5]: tensor renders truncate after 32 elements with the
/// `, ...` marker; a 32-element tensor renders in full.
#[test]
fn render_value_truncates_at_32_with_marker() {
    let data: Vec<f64> = (1..=33).map(f64::from).collect();
    let rendered = render_value(&tensor_value(Prim::F64, vec![33], data));
    let visible: Vec<String> = (1..=32).map(|i| format!("{i}.0")).collect();
    assert_eq!(
        rendered,
        format!("tensor(shape=[33], data=[{}, ...])", visible.join(", "))
    );

    let full: Vec<f64> = (1..=32).map(f64::from).collect();
    let rendered = render_value(&tensor_value(Prim::F64, vec![32], full));
    assert!(
        !rendered.contains("..."),
        "a 32-element tensor renders in full: {rendered}"
    );
}

/// Scalar payloads render at their OWN width ([05-OBS-2]): the f32
/// scalar prints its shortest-at-f32 digits, not the f64 image.
#[test]
fn render_value_scalars_render_at_own_width() {
    let f32_scalar = RuntimeValue::scalar_like_float(Prim::F32, 0.1f32 as f64).expect("scalar");
    assert_eq!(render_value(&f32_scalar), "0.1");
    let f16_scalar = RuntimeValue::scalar_like_float(Prim::F16, 2048.0).expect("scalar");
    assert_eq!(render_value(&f16_scalar), "2048.0");
    let i64_scalar = RuntimeValue::scalar_like_int(Prim::Int64, 9007199254740993).expect("scalar");
    assert_eq!(render_value(&i64_scalar), "9007199254740993");
    let f64_scalar = RuntimeValue::scalar_like_float(Prim::F64, f64::MAX).expect("scalar");
    assert_eq!(render_value(&f64_scalar), "1.7976931348623157e308");
}

// ===========================================================================
// RT792 probes (PR #792 fresh-context red team, adopted): the tag-vs-bits
// arithmetic edges of render_tensor_element, locked at the boundaries.
// ===========================================================================

/// int_or_bits boundary sweep at +/-2^63 (red-team authored): stored
/// exactly -2^63 sits inside the exact-i64 guard and prints the integer;
/// stored exactly +2^63 (where f64 has no i64 twin) is outside and prints
/// the faithful f64 bits, never a saturated integer; the largest f64
/// below 2^63 prints exactly. Non-integral / non-finite stored values
/// under an integer tag print the f64 bits.
#[test]
fn rt792_render_value_int64_tag_pow63_boundaries() {
    // chelis#729 Phase 1 re-authoring: in-range integral values store and
    // render exactly; the out-of-range and fractional rows that used to
    // render as faithful f64 bits now trap at the construction chokepoint
    // (see tag_storage_disagreement_is_unconstructible).
    assert_eq!(
        render_value(&tensor_value(
            Prim::Int64,
            vec![1],
            vec![-9223372036854775808.0]
        )),
        "tensor(shape=[1], data=[-9223372036854775808])"
    );
    assert_eq!(
        render_value(&tensor_value(
            Prim::Int64,
            vec![1],
            vec![9223372036854774784.0]
        )),
        "tensor(shape=[1], data=[9223372036854774784])"
    );
}

/// A bool-tagged slot storing -0.0 prints the BITS (`-0.0`), never
/// `false`: the sign bit is stored state a boolean rendering would
/// launder, and the slot is source-reachable via
/// `print(neg(to_tensor([false, true])))`. Red-team finding F6, fixed;
/// this is the tightened lock (the red team's original test pinned the
/// laundering behavior as evidence). A bool-tagged 2.0 / 0.5 prints the
/// bits per the tag-vs-bits rule; +0.0 stays `false`; an int-tagged -0.0
/// prints `0` (integers have no signed zero - the sign bit there is an
/// f64-backing-store artifact, not integer state).
#[test]
fn rt792_render_value_bool_tag_negative_zero_is_false() {
    // chelis#729 Phase 1 re-authoring: -0.0 equals 0, a member of bool's
    // {0, 1} set, so it finalizes to false (the old faithful-misreport
    // "-0.0" row required a tag/storage disagreement that is now
    // unrepresentable; non-members like 2.0 and 0.5 trap at the
    // chokepoint instead - see tag_storage_disagreement_is_unconstructible).
    assert_eq!(
        render_value(&tensor_value(Prim::Bool, vec![1], vec![-0.0])),
        "tensor(shape=[1], data=[false])"
    );
    assert_eq!(
        render_value(&tensor_value(Prim::Bool, vec![1], vec![0.0])),
        "tensor(shape=[1], data=[false])"
    );
    assert_eq!(
        render_value(&tensor_value(Prim::Int32, vec![1], vec![-0.0])),
        "tensor(shape=[1], data=[0])"
    );
}

// ---------------------------------------------------------------------------
// Host-lane JSON I/O (chelis#890): full-pipeline coverage (surf parse ->
// desugar -> check -> host eval) for parse_json / accessors / constructors /
// to_json / round_to. The pure core is unit-tested in `runtime/json.rs`;
// these tests pin the registry wiring: env schemes, the prelude `Json` ADT,
// `check_json_builtin_signature`, and the eval dispatch arms.
// ---------------------------------------------------------------------------

/// End-to-end in-memory pipeline: parse a document, read scalars and a
/// number list through dot-paths, compute, round, assemble nested output,
/// and serialize. The final string is asserted byte-exactly (insertion
/// order, shortest-round-trip numbers).
#[test]
fn json_pipeline_parse_access_assemble_serialize() {
    let checked = checked_surf(
        r#"
doc = parse_json("{\"instrument\": \"generic\", \"quotes\": {\"mid\": 101.4568}, \"weights\": [0.25, 0.75]}")
name = json_str(doc, "instrument")
mid = json_f64(doc, "quotes.mid")
weights = json_f64s(doc, "weights")
out = jdict([("instrument", jstr(name))])
out2 = json_set(out, "results.mid_rounded", jnum(round_to(mid, 2)))
out3 = json_set(out2, "results.first_weight", jnum(index(weights, 0)))
text = to_json(out3)
"#,
    );
    let outcome =
        evaluate_host_program(&checked, &HashMap::new()).expect("pipeline should evaluate");
    let text = outcome.host_bindings.get("text").expect("text binding");
    match text {
        RuntimeValue::String(s) => assert_eq!(
            s,
            r#"{"instrument":"generic","results":{"mid_rounded":101.46,"first_weight":0.25}}"#
        ),
        other => panic!("expected string, got {other:?}"),
    }
}

/// `round_to` accepts a bare int literal for `places` (an int32 per §5.3;
/// the signature arm admits any integer precision) and preserves the f64
/// coming out of `json_f64`. 2.675 parses to 2.67499999999999982..., so
/// correct decimal rounding gives 2.67 (ties-to-even on the exact value,
/// matching Python's round), NOT 2.68.
#[test]
fn json_round_to_exact_binary_value_with_bare_int_places() {
    let checked = checked_surf(
        r#"
x = json_f64(parse_json("{\"v\": 2.675}"), "v")
rounded = round_to(x, 2)
"#,
    );
    let outcome = evaluate_host_program(&checked, &HashMap::new()).expect("should evaluate");
    let rounded = outcome.host_bindings.get("rounded").expect("rounded");
    assert_eq!(rounded.as_f64(), Some(2.67));
}

/// Power-user surface: matching on the prelude `Json` ADT constructors.
#[test]
fn json_adt_match_extracts_variants() {
    let checked = checked_surf(
        r#"
num_or_zero = match parse_json("2.5") with {
  | JNum(n) => n
  | _ => cast(0.0, f64)
}
flag = match parse_json("true") with {
  | JBool(b) => b
  | _ => false
}
"#,
    );
    let outcome = evaluate_host_program(&checked, &HashMap::new()).expect("should evaluate");
    assert_eq!(
        outcome
            .host_bindings
            .get("num_or_zero")
            .and_then(RuntimeValue::as_f64),
        Some(2.5)
    );
    match outcome.host_bindings.get("flag") {
        Some(RuntimeValue::Bool(true)) => {}
        other => panic!("expected true, got {other:?}"),
    }
}

/// Loud-failure contract at eval time: a missing key names the builtin,
/// the path, the segment, and the available keys; a type mismatch names
/// the actual node kind. No silent defaults.
#[test]
fn json_accessor_failures_are_loud() {
    let missing = checked_surf(
        r#"
x = json_f64(parse_json("{\"alpha\": 1.5}"), "beta")
"#,
    );
    let err =
        evaluate_host_program(&missing, &HashMap::new()).expect_err("missing key must fail eval");
    assert!(err.contains("json_f64"), "got `{err}`");
    assert!(err.contains("key `beta` not found"), "got `{err}`");
    assert!(err.contains("`alpha`"), "available keys listed: `{err}`");

    let mismatch = checked_surf(
        r#"
x = json_f64(parse_json("{\"alpha\": \"txt\"}"), "alpha")
"#,
    );
    let err = evaluate_host_program(&mismatch, &HashMap::new())
        .expect_err("type mismatch must fail eval");
    assert!(err.contains("expected a number, got string"), "got `{err}`");

    let malformed = checked_surf(
        r#"
x = parse_json("{\"alpha\": }")
"#,
    );
    let err = evaluate_host_program(&malformed, &HashMap::new())
        .expect_err("malformed JSON must fail eval");
    assert!(err.contains("parse_json"), "got `{err}`");
}

/// Check-time negative parity: the signature arm rejects non-Json /
/// non-string / non-numeric slots with named diagnostics.
#[test]
fn json_builtin_type_errors_reject_at_check() {
    for (source, fragment) in [
        (
            "x = json_f64(1.5, \"a\")\n",
            "json_f64 expects a Json first argument",
        ),
        (
            "x = parse_json(1.5)\n",
            "parse_json expects a string argument",
        ),
        ("x = to_json(\"raw\")\n", "to_json expects a Json argument"),
        (
            "x = round_to(\"s\", 2)\n",
            "round_to expects an f64 or f32 first argument",
        ),
        // [05-OP-1] authors decimal rounding for f64 and f32 only; an
        // f16 operand is rejected with the remediation, never silently
        // computed at another width ([04-NUM-8] has no exception
        // vocabulary).
        (
            "x = round_to(cast(1.5, f16), 2)\n",
            "round_to expects an f64 or f32 first argument",
        ),
        // Operand must be f64 (unsuffixed floats default to f32, spec/04
        // §5.3) so the case reaches the `places` slot it exercises.
        (
            "x = round_to(1.5f64, 2.0)\n",
            "round_to expects an integer `places` second argument",
        ),
        (
            "x = jnum(\"not a number\")\n",
            "jnum expects an f64 argument",
        ),
        // chelis#891 review finding 7: a bare float literal is f32 (§5.3)
        // and would quantize through the byte-exact serializer, so jnum
        // rejects it loudly with the suffix/cast guidance.
        ("x = jnum(0.1)\n", "jnum expects an f64 argument"),
        (
            "x = json_set(jdict([(\"a\", jnum(1.0))]), \"a\", 2.0)\n",
            "json_set expects a Json third argument",
        ),
        // Wrong arity is caught by the generic scheme unification before
        // the signature arm runs; the diagnostic is still loud and typed.
        ("x = parse_json(\"1\", \"2\")\n", "arity mismatch"),
    ] {
        let decls = chelis_surf::parser::parse_str(source).expect("surf parse");
        let exprs = chelis_surf::desugar::desugar_program(&decls);
        let err = chelis_types::check_ir_program(&exprs)
            .expect_err(&format!("{source:?} must be a check error"));
        assert!(
            err.errors.iter().any(|e| e.message.contains(fragment)),
            "{source:?}: expected a diagnostic containing `{fragment}`, got: {:?}",
            err.errors.iter().map(|e| &e.message).collect::<Vec<_>>()
        );
    }
}

/// Constructors compose with `json_list` element-relative access, and
/// serialization is deterministic (two runs, identical bytes).
#[test]
fn json_constructors_and_element_relative_access() {
    let checked = checked_surf(
        r#"
doc = parse_json("{\"rows\": [{\"px\": 1.5}, {\"px\": 2.5}]}")
rows = json_list(doc, "rows")
second_px = json_f64(index(rows, 1), "px")
text = to_json(jlist([jnum(second_px), JNull, JBool(true)]))
"#,
    );
    let run = |checked: &chelis_types::CheckedProgram| {
        let outcome = evaluate_host_program(checked, &HashMap::new()).expect("should evaluate");
        match outcome.host_bindings.get("text") {
            Some(RuntimeValue::String(s)) => s.clone(),
            other => panic!("expected string, got {other:?}"),
        }
    };
    let first = run(&checked);
    assert_eq!(first, "[2.5,null,true]");
    assert_eq!(run(&checked), first, "serialization must be deterministic");
}

/// chelis#891 review finding 6: the builtin contracts are enforced by
/// unification, so an un-annotated parameter flowing into a Json/float
/// slot is pinned to the expected type instead of leaving the contract
/// vacuous (`forall a. a -> f64`).
#[test]
fn json_builtin_slots_unify_unannotated_params() {
    // The lambda parameter unifies with Json; applying it to a float is
    // now a check error (previously it checked clean and failed at eval).
    let source = "f = fn (doc) -> json_f64(doc, \"a\")\nx = f(1.5)\n";
    let decls = chelis_surf::parser::parse_str(source).expect("surf parse");
    let exprs = chelis_surf::desugar::desugar_program(&decls);
    let err = chelis_types::check_ir_program(&exprs).expect_err("non-Json arg must be rejected");
    assert!(
        err.errors.iter().any(|e| e.message.contains("mismatch")),
        "expected a type mismatch, got: {:?}",
        err.errors.iter().map(|e| &e.message).collect::<Vec<_>>()
    );

    // round_to's operand slot pins an unresolved Var to f64, so checker
    // and eval agree on the result dtype; an f32 application is a check
    // error rather than a silent f32-in/f64-claimed disagreement.
    let source = "g = fn (v) -> round_to(v, 2)\nz = g(1.5)\n";
    let decls = chelis_surf::parser::parse_str(source).expect("surf parse");
    let exprs = chelis_surf::desugar::desugar_program(&decls);
    let err = chelis_types::check_ir_program(&exprs).expect_err("f32 arg must be rejected");
    assert!(
        err.errors.iter().any(|e| e.message.contains("mismatch")),
        "expected a precision mismatch, got: {:?}",
        err.errors.iter().map(|e| &e.message).collect::<Vec<_>>()
    );

    // The pinned path works end-to-end with a genuine f64.
    let checked = checked_surf(
        r#"
g = fn (v) -> round_to(v, 2)
z = g(json_f64(parse_json("{\"v\": 2.675}"), "v"))
"#,
    );
    let outcome = evaluate_host_program(&checked, &HashMap::new()).expect("should evaluate");
    assert_eq!(
        outcome
            .host_bindings
            .get("z")
            .and_then(RuntimeValue::as_f64),
        Some(2.67)
    );
}

// ---------------------------------------------------------------------------
// Host-lane CSV I/O (chelis#903): full-pipeline coverage (surf parse ->
// desugar -> check -> host eval) for parse_csv / csv_* accessors / to_csv.
// The pure core is unit-tested in `runtime/csv.rs`; these tests pin the
// registry wiring: env schemes, `check_csv_builtin_signature`, the eval
// dispatch arms, and the composition with the #890 JSON surface.
// ---------------------------------------------------------------------------

/// The QFBench-shaped loop in miniature: parse a CSV, pull numeric columns
/// through `csv_f64s`, compute in the tensor lane, round, and assemble a
/// nested JSON output. The bare `0`/`2` literals exercise the any-integer
/// slots. Asserted byte-exactly.
#[test]
fn csv_pipeline_parse_access_compute_assemble() {
    let checked = checked_surf(
        r#"
c = parse_csv("id,qty,px\nalpha,2,101.5\nbeta,4,99.25\n")
qty = csv_f64s(c, "qty")
px = csv_f64s(c, "px")
notional = tensor_to_scalar(sum(mul(to_tensor(qty), to_tensor(px)), 0))
out = jdict([("first_id", jstr(csv_str(c, 0, "id")))])
out2 = json_set(out, "results.notional", jnum(round_to(notional, 2)))
out3 = json_set(out2, "results.rows", jnum(cast(csv_nrows(c), f64)))
text = to_json(out3)
"#,
    );
    let outcome =
        evaluate_host_program(&checked, &HashMap::new()).expect("pipeline should evaluate");
    match outcome.host_bindings.get("text") {
        Some(RuntimeValue::String(s)) => assert_eq!(
            s,
            r#"{"first_id":"alpha","results":{"notional":600.0,"rows":2.0}}"#
        ),
        other => panic!("expected string, got {other:?}"),
    }
}

/// A Csv document is a Json value: the #890 accessors work on it directly
/// (`json_list` over `rows`, dot-path reads into row cells, `to_json` for
/// debugging); this is the payoff of riding the Json ADT instead of
/// adding a `Csv` prelude type.
#[test]
fn csv_document_composes_with_json_accessors() {
    let checked = checked_surf(
        r#"
c = parse_csv("id,px\nalpha,1.5\nbeta,2.5\n")
rows = json_list(c, "rows")
first_id = json_str(index(rows, 0), "id")
second_px_text = json_str(c, "rows.1.px")
doc_text = to_json(c)
"#,
    );
    let outcome = evaluate_host_program(&checked, &HashMap::new()).expect("should evaluate");
    match outcome.host_bindings.get("first_id") {
        Some(RuntimeValue::String(s)) => assert_eq!(s, "alpha"),
        other => panic!("expected string, got {other:?}"),
    }
    match outcome.host_bindings.get("second_px_text") {
        Some(RuntimeValue::String(s)) => assert_eq!(s, "2.5"),
        other => panic!("expected string, got {other:?}"),
    }
    match outcome.host_bindings.get("doc_text") {
        Some(RuntimeValue::String(s)) => assert_eq!(
            s,
            r#"{"columns":["id","px"],"rows":[{"id":"alpha","px":"1.5"},{"id":"beta","px":"2.5"}]}"#
        ),
        other => panic!("expected string, got {other:?}"),
    }
}

/// CSV output assembly with the #890 constructors (`jdict`/`jlist`/`jnum`/
/// `jstr`) feeding `to_csv`, deterministic across runs. `csv_f64` reads an
/// exact f64 out of the input document; `round_to` fixes the decimals.
#[test]
fn csv_output_assembly_via_to_csv() {
    let checked = checked_surf(
        r#"
c = parse_csv("id,px\nalpha,1.23456\nbeta,2.5\n")
row = jdict([("id", jstr(csv_str(c, 0, "id"))), ("pv", jnum(round_to(csv_f64(c, 0, "px"), 2)))])
out = jdict([("columns", jlist([jstr("id"), jstr("pv")])), ("rows", jlist([row]))])
text = to_csv(out)
"#,
    );
    let run = |checked: &chelis_types::CheckedProgram| {
        let outcome = evaluate_host_program(checked, &HashMap::new()).expect("should evaluate");
        match outcome.host_bindings.get("text") {
            Some(RuntimeValue::String(s)) => s.clone(),
            other => panic!("expected string, got {other:?}"),
        }
    };
    let first = run(&checked);
    assert_eq!(first, "id,pv\nalpha,1.23\n");
    assert_eq!(run(&checked), first, "to_csv must be deterministic");
}

/// Loud-failure contract at eval time: a missing column names the builtin,
/// the column, and the available columns; a non-numeric cell names the
/// column, the 0-based data row, and the offending text; a malformed file
/// names the 1-based row/column position. No silent NaN/defaults.
#[test]
fn csv_accessor_failures_are_loud() {
    let missing = checked_surf(
        r#"
xs = csv_f64s(parse_csv("date,mid\n2020-01-02,1.5\n"), "px")
"#,
    );
    let err = evaluate_host_program(&missing, &HashMap::new())
        .expect_err("missing column must fail eval");
    assert!(err.contains("csv_f64s"), "got `{err}`");
    assert!(err.contains("column `px` not found"), "got `{err}`");
    assert!(
        err.contains("available columns: `date`, `mid`"),
        "got `{err}`"
    );

    let non_numeric = checked_surf(
        r#"
xs = csv_f64s(parse_csv("id,px\nalpha,n/a\n"), "px")
"#,
    );
    let err = evaluate_host_program(&non_numeric, &HashMap::new())
        .expect_err("non-numeric cell must fail eval");
    assert!(err.contains("column `px`"), "got `{err}`");
    assert!(err.contains("data row 0"), "got `{err}`");
    assert!(err.contains("cell `n/a` is not a number"), "got `{err}`");

    let malformed = checked_surf(
        r#"
c = parse_csv("a,b\n1,\"oops\n")
"#,
    );
    let err = evaluate_host_program(&malformed, &HashMap::new())
        .expect_err("malformed CSV must fail eval");
    assert!(err.contains("parse_csv"), "got `{err}`");
    assert!(err.contains("row 2, column 2"), "got `{err}`");
    assert!(err.contains("unclosed quoted field"), "got `{err}`");

    // The json_set seam (chelis#903 review): augmenting a Csv document
    // with an extra top-level subtree is fine for reads, but to_csv
    // refuses to silently drop it.
    let augmented = checked_surf(
        r#"
c2 = json_set(parse_csv("id,px\nalpha,1.5\n"), "meta.note", jstr("x"))
n = csv_nrows(c2)
text = to_csv(c2)
"#,
    );
    let err = evaluate_host_program(&augmented, &HashMap::new())
        .expect_err("extra top-level key must fail to_csv");
    assert!(
        err.contains("unexpected top-level key `meta`"),
        "got `{err}`"
    );
    assert!(
        err.contains("refusing to silently drop data"),
        "got `{err}`"
    );
}

/// Check-time negative parity: `check_csv_builtin_signature` rejects
/// non-document / non-string / non-integer slots with named diagnostics.
#[test]
fn csv_builtin_type_errors_reject_at_check() {
    for (source, fragment) in [
        (
            "x = parse_csv(1.5)\n",
            "parse_csv expects a string argument",
        ),
        (
            "x = to_csv(\"raw\")\n",
            "to_csv expects a Csv document first argument",
        ),
        (
            "x = csv_f64s(1.5, \"a\")\n",
            "csv_f64s expects a Csv document first argument",
        ),
        (
            "x = csv_nrows(1.5)\n",
            "csv_nrows expects a Csv document first argument",
        ),
        (
            "x = csv_f64s(parse_csv(\"a\"), 2)\n",
            "csv_f64s expects a column-name string second argument",
        ),
        (
            "x = csv_f64(parse_csv(\"a\"), \"zero\", \"a\")\n",
            "csv_f64 expects an integer row index second argument",
        ),
        (
            "x = csv_str(parse_csv(\"a\"), 0, 1.5)\n",
            "csv_str expects a column-name string third argument",
        ),
        // Wrong arity is caught by the generic scheme unification before
        // the signature arm runs; the diagnostic is still loud and typed.
        ("x = parse_csv(\"a\", \"b\")\n", "arity mismatch"),
    ] {
        let decls = chelis_surf::parser::parse_str(source).expect("surf parse");
        let exprs = chelis_surf::desugar::desugar_program(&decls);
        let err = chelis_types::check_ir_program(&exprs)
            .expect_err(&format!("{source:?} must be a check error"));
        assert!(
            err.errors.iter().any(|e| e.message.contains(fragment)),
            "{source:?}: expected a diagnostic containing `{fragment}`, got: {:?}",
            err.errors.iter().map(|e| &e.message).collect::<Vec<_>>()
        );
    }
}

/// The integer accessors are exact end-to-end ([05-OP-3] / [04-NUM-11]):
/// an int64 ID above 2^53 survives parse_json -> json_int/json_ints ->
/// jint -> to_json bit-exactly, while the float accessors refuse nothing
/// silently -- json_f64 widens (the named lossy read) and json_ints
/// refuses a float element loudly.
#[test]
fn json_integer_accessors_are_exact_above_2_53() {
    let checked = checked_surf(
        r#"
doc = parse_json("{\"id\": 9007199254740993, \"ids\": [9007199254740993, 7]}")
exact = json_int(doc, "id")
exact_list = json_ints(doc, "ids")
widened = json_f64(doc, "id")
echoed = to_json(json_set(doc, "out", jint(json_int(doc, "id"))))
"#,
    );
    let outcome = evaluate_host_program(&checked, &HashMap::new()).expect("should evaluate");
    assert_eq!(
        outcome
            .host_bindings
            .get("exact")
            .and_then(RuntimeValue::as_i64),
        Some(9007199254740993)
    );
    match outcome.host_bindings.get("exact_list") {
        Some(RuntimeValue::List(items)) => {
            assert_eq!(items[0].as_i64(), Some(9007199254740993));
            assert_eq!(items[1].as_i64(), Some(7));
        }
        other => panic!("expected list, got {other:?}"),
    }
    // The named lossy widening: 2^53 + 1 is not representable in f64.
    assert_eq!(
        outcome
            .host_bindings
            .get("widened")
            .and_then(RuntimeValue::as_f64),
        Some(9007199254740992.0)
    );
    match outcome.host_bindings.get("echoed") {
        Some(RuntimeValue::String(s)) => assert!(
            s.contains("\"out\":9007199254740993"),
            "JInt must serialize exactly, got {s}"
        ),
        other => panic!("expected string, got {other:?}"),
    }

    // Negative parity: a float element in json_ints fails loudly, naming
    // the float accessor -- no silent truncation.
    let refused = checked_surf(
        r#"
xs = json_ints(parse_json("{\"v\": [1, 2.5]}"), "v")
"#,
    );
    let err =
        evaluate_host_program(&refused, &HashMap::new()).expect_err("float element must fail");
    assert!(err.contains("json_ints"), "got `{err}`");
    assert!(err.contains("element 1 is a float"), "got `{err}`");
    assert!(err.contains("json_f64s"), "names the remedy: `{err}`");
}

/// CSV integer columns are first-class ([05-OP-3]): `csv_ints`/`csv_int`
/// read int64 IDs above 2^53 exactly from cell text, refuse float cells
/// and float text, and round-trip exactly through jint -> to_csv ->
/// parse_csv -> csv_int. `csv_f64s` on the same column is the named
/// lossy widening, not an error.
#[test]
fn csv_integer_accessors_are_exact_above_2_53() {
    let checked = checked_surf(
        r#"
c = parse_csv("id,qty\n9007199254740993,2\n12,4\n")
ids = csv_ints(c, "id")
first = csv_int(c, 0, "id")
widened = csv_f64s(c, "id")
row = jdict([("id", jint(csv_int(c, 0, "id")))])
out = jdict([("columns", jlist([jstr("id")])), ("rows", jlist([row]))])
echoed = csv_int(parse_csv(to_csv(out)), 0, "id")
"#,
    );
    let outcome = evaluate_host_program(&checked, &HashMap::new()).expect("should evaluate");
    match outcome.host_bindings.get("ids") {
        Some(RuntimeValue::List(items)) => {
            assert_eq!(items[0].as_i64(), Some(9007199254740993));
            assert_eq!(items[1].as_i64(), Some(12));
        }
        other => panic!("expected list, got {other:?}"),
    }
    assert_eq!(
        outcome
            .host_bindings
            .get("first")
            .and_then(RuntimeValue::as_i64),
        Some(9007199254740993)
    );
    match outcome.host_bindings.get("widened") {
        Some(RuntimeValue::List(items)) => {
            assert_eq!(items[0].as_f64(), Some(9007199254740992.0));
        }
        other => panic!("expected list, got {other:?}"),
    }
    assert_eq!(
        outcome
            .host_bindings
            .get("echoed")
            .and_then(RuntimeValue::as_i64),
        Some(9007199254740993),
        "JInt cell must round-trip to_csv -> parse_csv -> csv_int exactly"
    );

    // Negative parity: float text refuses the integer read (naming the
    // float accessor), and int64 overflow is a loud trap-class error,
    // never an f64 fallback.
    let float_text = checked_surf(
        r#"
xs = csv_ints(parse_csv("id\n1.5\n"), "id")
"#,
    );
    let err =
        evaluate_host_program(&float_text, &HashMap::new()).expect_err("float text must fail");
    assert!(err.contains("not an integer"), "got `{err}`");
    assert!(
        err.contains("csv_f64/csv_f64s"),
        "names the remedy: `{err}`"
    );

    let overflow = checked_surf(
        r#"
xs = csv_ints(parse_csv("id\n99999999999999999999\n"), "id")
"#,
    );
    let err = evaluate_host_program(&overflow, &HashMap::new()).expect_err("overflow must fail");
    assert!(err.contains("overflows int64"), "got `{err}`");
    assert!(err.contains("Overflow"), "names the trap kind: `{err}`");
}

/// [05-OP-1]'s f32 lane: an f32 operand rounds at its OWN width and the
/// result stays f32 -- checker and eval agree on the dtype, and the value
/// is the correctly-rounded decimal rounding of the f32's exact binary
/// value. 2.675f32 is exactly 2.67499995231628417968750, so 2 places
/// rounds DOWN to 2.67 (then stored as the nearest f32).
#[test]
fn round_to_f32_lane_preserves_dtype_and_rounds_at_own_width() {
    let checked = checked_surf(
        r#"
x = round_to(2.675f32, 2)
y = round_to(2.675f64, 2)
"#,
    );
    let outcome = evaluate_host_program(&checked, &HashMap::new()).expect("should evaluate");
    match outcome.host_bindings.get("x") {
        Some(RuntimeValue::Scalar(payload)) => {
            assert_eq!(payload.dtype(), Prim::F32, "f32 in, f32 out");
            assert_eq!(payload.as_f64_lossy(), f64::from(2.67f32));
        }
        other => panic!("expected scalar, got {other:?}"),
    }
    match outcome.host_bindings.get("y") {
        Some(RuntimeValue::Scalar(payload)) => {
            assert_eq!(payload.dtype(), Prim::F64, "f64 in, f64 out");
            assert_eq!(payload.as_f64_lossy(), 2.67);
        }
        other => panic!("expected scalar, got {other:?}"),
    }
}

// ===========================================================================
// FO-DIAG: the diagnostic-rendering boundary (chelis#997,
// spec/design/faithful_observation.md §C1.6 / §B2.4).
//
// §C1.6: "a diagnostic must not launder what it reports." These pin the
// boundary beside `render_value` -- `describe_value`, `describe_argument`,
// `describe_fields` -- as the ONE way a runtime, JSON, or CSV payload
// becomes diagnostic text, and pin the negative half: the derived-`Debug`
// spellings must not reappear in any of it.
//
// Focused run:
//   cargo nextest run -p chelis-compiler-api --lib -E 'test(fo_diag)'
// ===========================================================================

/// Derived-`Debug` fingerprints. A diagnostic containing any of these is
/// rendering the Rust representation instead of the value.
const DEBUG_SPELLINGS: &[&str] = &[
    "Scalar(",
    "ScalarPayload",
    "ScalarValue",
    "bits:",
    "Bits::",
    "F16(",
    "Bf16(",
    "F32(",
    "F64(",
    "I32(",
    "I64(",
    "RuntimeValue::",
    "field_names",
    "Adt {",
    "IrTensorValue",
];

/// The measured pre-migration renderings of a stored f16/bf16 `0.1`:
/// `half`'s `Debug` forwards to `to_f32()`, so it reported `0.099975586`
/// and `0.100097656` for values every exit renders as `0.1`. These are
/// PREFIXES of those strings, so a `half` version bump that shifts the
/// trailing digits still trips the assertions, while the correct `0.1`
/// rendering can never contain either one.
const WIDENED_F16_IMAGE: &str = "0.0999755";
const WIDENED_BF16_IMAGE: &str = "0.1000976";

#[track_caller]
fn assert_no_debug_spelling(rendered: &str, what: &str) {
    for needle in DEBUG_SPELLINGS {
        assert!(
            !rendered.contains(needle),
            "{what} leaked the derived-Debug spelling `{needle}`: {rendered}"
        );
    }
}

fn scalar_of(dtype: Prim, value: f64) -> RuntimeValue {
    RuntimeValue::scalar_like_float(dtype, value).expect("in-domain scalar fixture")
}

fn int_scalar_of(dtype: Prim, value: i64) -> RuntimeValue {
    RuntimeValue::scalar_like_int(dtype, value).expect("in-domain scalar fixture")
}

fn adt(ctor: &str, fields: Vec<RuntimeValue>) -> RuntimeValue {
    RuntimeValue::Adt {
        ctor: ctor.to_string(),
        fields,
        field_names: None,
    }
}

/// A scalar renders as `<dtype> <canonical digits>`: the dtype comes from
/// the sealed storage variant, and the digits are `format_element`'s, so a
/// diagnostic and an exit channel agree on every payload.
#[test]
fn fo_diag_scalars_carry_their_dtype_and_own_width_digits() {
    // Exact int64 above 2^53 -- the [#723] case. 9007199254740993 has no
    // f64 twin, so a rendering that funnels through double reports
    // ...992.
    let big = int_scalar_of(Prim::Int64, 9_007_199_254_740_993);
    assert_eq!(describe_value(&big), "int64 9007199254740993");
    assert!(
        !describe_value(&big).contains("9007199254740992"),
        "the f64 image must not reach the diagnostic channel"
    );

    assert_eq!(describe_value(&int_scalar_of(Prim::Int32, -5)), "int32 -5");
    assert_eq!(describe_value(&int_scalar_of(Prim::Int8, 127)), "int8 127");

    // Own-width floats: the f32 prints its shortest-at-f32 digits, and the
    // f64 holding the same f32's image prints the wider, different string.
    assert_eq!(
        describe_value(&scalar_of(Prim::F32, 0.1f32 as f64)),
        "f32 0.1"
    );
    assert_eq!(
        describe_value(&scalar_of(Prim::F64, 0.1f32 as f64)),
        "f64 0.10000000149011612"
    );
    assert_eq!(
        describe_value(&scalar_of(Prim::F64, f64::MAX)),
        "f64 1.7976931348623157e308"
    );
    assert_eq!(describe_value(&scalar_of(Prim::F64, -0.0)), "f64 -0.0");

    for value in [&big, &scalar_of(Prim::F32, 0.1f32 as f64)] {
        assert_no_debug_spelling(&describe_value(value), "a scalar diagnostic");
    }
}

/// The half widths are where derived `Debug` actively lies: `half::f16`'s
/// `Debug` forwards to `to_f32()`, so it reports the WIDENED image rather
/// than the shortest string that parses back to the stored f16. The
/// pre-migration rendering of the first two fixtures below, measured:
///
/// ```text
/// [Scalar(ScalarPayload { value: ScalarValue { bits: F16(0.099975586) } }),
///  Scalar(ScalarPayload { value: ScalarValue { bits: Bf16(0.100097656) } })]
/// ```
///
/// One stored value, two answers: `0.099975586` in a diagnostic and `0.1`
/// at every exit. That is the §B2.4 harm itself, so these assertions pin
/// the own-width digits and the absence of the widened image.
#[test]
fn fo_diag_half_widths_are_not_reported_through_their_f32_image() {
    let f16 = scalar_of(Prim::F16, 0.1);
    assert_eq!(describe_value(&f16), "f16 0.1");
    // The needle is the shared PREFIX of the measured widened image
    // (`0.099975586`), not the whole string: a `half` bump that shifts the
    // trailing digits must still trip this, and the correct rendering
    // (`0.1`) can never contain it.
    assert!(
        !describe_value(&f16).contains(WIDENED_F16_IMAGE),
        "the f32 image of the stored f16 must not reach the diagnostic"
    );

    let bf16 = scalar_of(Prim::Bf16, 0.1);
    assert_eq!(describe_value(&bf16), "bf16 0.1");
    assert!(!describe_value(&bf16).contains(WIDENED_BF16_IMAGE));

    // And the same value inside a malformed ADT field list, which is how a
    // JSON/CSV shape diagnostic reports it.
    assert_eq!(describe_fields(&[f16, bf16]), "[f16 0.1, bf16 0.1]");
}

/// Bools, strings, and the nonnumeric controls: each names its kind, and
/// a string is quoted so an empty or control-carrying value is visible.
#[test]
fn fo_diag_bools_strings_and_nonnumeric_controls() {
    assert_eq!(describe_value(&RuntimeValue::Bool(true)), "bool true");
    assert_eq!(describe_value(&RuntimeValue::Bool(false)), "bool false");
    // A bool is never reported as a number ([05-OBS-3]).
    assert!(!describe_value(&RuntimeValue::Bool(true)).contains('1'));

    assert_eq!(
        describe_value(&RuntimeValue::String(String::new())),
        "string \"\""
    );
    assert_eq!(
        describe_value(&RuntimeValue::String("a\nb\"c".to_string())),
        "string \"a\\nb\\\"c\""
    );

    assert_eq!(describe_value(&RuntimeValue::Unit), "()");
    assert_eq!(
        describe_value(&RuntimeValue::MappedFile(vec![1, 2, 3])),
        "<mapped-file:3>"
    );
    assert_eq!(
        describe_value(&RuntimeValue::Closure {
            params: vec!["x".to_string()],
            param_types: vec![None],
            body: chelis_deep::ast::Expr::Atom(
                chelis_deep::ast::Atom::Bool(false),
                chelis_deep::Span::new(0, 0)
            ),
            env: HashMap::new(),
        }),
        "<closure>"
    );
}

/// Nested structure is `render_value`'s output verbatim -- the diagnostic
/// channel and the exit channel are the same grammar below the top-level
/// kind tag, so nested payloads cannot disagree.
#[test]
fn fo_diag_nested_structure_delegates_to_the_canonical_renderer() {
    let list = RuntimeValue::List(vec![
        int_scalar_of(Prim::Int64, 9_007_199_254_740_993),
        scalar_of(Prim::F32, 0.1f32 as f64),
        RuntimeValue::Bool(true),
    ]);
    assert_eq!(
        describe_value(&list),
        format!("list {}", render_value(&list))
    );
    assert_eq!(describe_value(&list), "list [9007199254740993, 0.1, true]");

    let tuple = RuntimeValue::Tuple(vec![int_scalar_of(Prim::Int32, 1), RuntimeValue::Unit]);
    assert_eq!(describe_value(&tuple), "tuple (1, ())");

    // Tensors, dicts, and ADTs already name their own shape, so they are
    // the canonical rendering unchanged.
    let tensor = tensor_value(
        Prim::F32,
        vec![2],
        vec![f64::from(0.1f32), f64::from(0.2f32)],
    );
    assert_eq!(describe_value(&tensor), render_value(&tensor));
    assert_eq!(
        describe_value(&tensor),
        "tensor(shape=[2], data=[0.1, 0.2])"
    );

    let nested = adt(
        "JList",
        vec![RuntimeValue::List(vec![
            adt(
                "JInt",
                vec![int_scalar_of(Prim::Int64, 9_007_199_254_740_993)],
            ),
            adt("JNum", vec![scalar_of(Prim::F64, 1e-7)]),
        ])],
    );
    assert_eq!(describe_value(&nested), render_value(&nested));
    assert_eq!(
        describe_value(&nested),
        "JList([JInt(9007199254740993), JNum(1e-7)])"
    );
    assert_no_debug_spelling(&describe_value(&nested), "a nested Json diagnostic");
}

/// `describe_fields` tags every field, because a malformed-shape
/// diagnostic has to say WHY the shape was rejected: `[int32 5]` under a
/// `JNum` names the reason an untagged `[5]` does not.
#[test]
fn fo_diag_field_lists_tag_every_field() {
    assert_eq!(describe_fields(&[]), "[]");
    assert_eq!(
        describe_fields(&[int_scalar_of(Prim::Int32, 5)]),
        "[int32 5]"
    );
    assert_eq!(
        describe_fields(&[
            scalar_of(Prim::F32, 0.1f32 as f64),
            RuntimeValue::Bool(true),
            RuntimeValue::String("x".to_string()),
        ]),
        "[f32 0.1, bool true, string \"x\"]"
    );
}

/// A missing argument slot reads as a missing argument, not as a Rust
/// `Option` spelling.
#[test]
fn fo_diag_argument_slots_name_an_absent_argument() {
    assert_eq!(describe_argument(None), "nothing");
    let value = int_scalar_of(Prim::Int32, 7);
    assert_eq!(describe_argument(Some(&value)), describe_value(&value));
    assert_no_debug_spelling(&describe_argument(None), "an absent argument");
}

/// Truncation is the boundary's, not the call site's: one cap, applied
/// once, on a char boundary, and stated rather than silent. A rendering
/// that fits is byte-identical to the untruncated form.
#[test]
fn fo_diag_truncation_is_owned_by_the_boundary() {
    let short = RuntimeValue::List(vec![int_scalar_of(Prim::Int32, 1)]);
    assert_eq!(describe_value(&short), "list [1]");
    assert!(!describe_value(&short).contains("elided"));

    let long = RuntimeValue::List(
        (0..200)
            .map(|i| int_scalar_of(Prim::Int32, i))
            .collect::<Vec<_>>(),
    );
    let rendered = describe_value(&long);
    assert!(
        rendered.starts_with("list [0, 1, 2, "),
        "the head of the payload survives: {rendered}"
    );
    assert!(
        rendered.ends_with(" more bytes elided)"),
        "the cut is stated, not silent: {rendered}"
    );

    // Multibyte: the cut lands on a char boundary (a panic here is the
    // failure), and a long multibyte string still truncates.
    let wide = RuntimeValue::String("\u{1F600}".repeat(200));
    let rendered = describe_value(&wide);
    assert!(rendered.ends_with(" more bytes elided)"));
    assert!(rendered.starts_with("string \"\u{1F600}"));
}

/// The JSON runtime's malformed-shape diagnostics report their payloads
/// through the boundary. Positive: the exact int64 and the own-width f16
/// survive. Negative: no derived-`Debug` spelling appears.
#[test]
fn fo_diag_json_shape_diagnostics_report_exact_payloads() {
    // A `JNum` whose field is an integer is malformed; the diagnostic
    // names the field's dtype, which is the whole reason it was rejected.
    let bad_num = adt(
        "JNum",
        vec![int_scalar_of(Prim::Int64, 9_007_199_254_740_993)],
    );
    let doc = adt(
        "JDict",
        vec![RuntimeValue::Dict(vec![(
            RuntimeValue::String("v".to_string()),
            bad_num,
        )])],
    );
    let err = super::json::json_f64_at(&doc, "v").expect_err("a malformed JNum must fail");
    assert!(
        err.contains("malformed JNum fields [int64 9007199254740993]"),
        "the exact stored integer must reach the diagnostic: {err}"
    );
    assert!(
        !err.contains("9007199254740992"),
        "no double funnel in the diagnostic channel: {err}"
    );
    assert_no_debug_spelling(&err, "the json_f64 shape diagnostic");

    // A `JInt` carrying an f16 is malformed for `json_int`; the f16 is
    // reported at its own width, not through its f32 image.
    let bad_int = adt("JInt", vec![scalar_of(Prim::F16, 0.1)]);
    let doc = adt(
        "JDict",
        vec![RuntimeValue::Dict(vec![(
            RuntimeValue::String("v".to_string()),
            bad_int,
        )])],
    );
    let err = super::json::json_int_at(&doc, "v").expect_err("a malformed JInt must fail");
    assert!(
        err.contains("malformed JInt fields [f16 0.1]"),
        "own-width f16 digits in the diagnostic: {err}"
    );
    assert!(!err.contains(WIDENED_F16_IMAGE), "no widened image: {err}");

    // A non-Json value reaching the serializer names what it actually is.
    let err = super::json::json_value_to_text(&RuntimeValue::Bool(true))
        .expect_err("a bare bool is not a Json value");
    assert!(err.contains("bool true"), "got: {err}");
    assert_no_debug_spelling(&err, "the to_json shape diagnostic");

    // A non-string JDict key is reported as the value it is.
    let bad_key = adt(
        "JDict",
        vec![RuntimeValue::Dict(vec![(
            int_scalar_of(Prim::Int32, 1),
            adt("JNull", vec![]),
        )])],
    );
    let err = super::json::ensure_json_value(&bad_key).expect_err("non-string key must fail");
    assert!(err.contains("got int32 1"), "got: {err}");
}

/// The CSV runtime's cell and document diagnostics use the same boundary.
#[test]
fn fo_diag_csv_shape_diagnostics_report_exact_payloads() {
    fn csv_doc(cell: RuntimeValue) -> RuntimeValue {
        let row = adt(
            "JDict",
            vec![RuntimeValue::Dict(vec![(
                RuntimeValue::String("v".to_string()),
                cell,
            )])],
        );
        adt(
            "JDict",
            vec![RuntimeValue::Dict(vec![
                (
                    RuntimeValue::String("columns".to_string()),
                    adt(
                        "JList",
                        vec![RuntimeValue::List(vec![adt(
                            "JStr",
                            vec![RuntimeValue::String("v".to_string())],
                        )])],
                    ),
                ),
                (
                    RuntimeValue::String("rows".to_string()),
                    adt("JList", vec![RuntimeValue::List(vec![row])]),
                ),
            ])],
        )
    }

    // A `JNum` cell holding an int64 above 2^53 is malformed; the exact
    // digits reach the diagnostic.
    let err = super::csv::csv_f64_at(
        &csv_doc(adt(
            "JNum",
            vec![int_scalar_of(Prim::Int64, 9_007_199_254_740_993)],
        )),
        0,
        "v",
    )
    .expect_err("a malformed JNum cell must fail");
    assert!(
        err.contains("malformed JNum cell [int64 9007199254740993]"),
        "got: {err}"
    );
    assert!(!err.contains("9007199254740992"), "got: {err}");
    assert_no_debug_spelling(&err, "the csv_f64 cell diagnostic");

    // A `JInt` cell holding an f16 is malformed for `csv_int`; own width.
    let err = super::csv::csv_int_at(
        &csv_doc(adt("JInt", vec![scalar_of(Prim::F16, 0.1)])),
        0,
        "v",
    )
    .expect_err("a malformed JInt cell must fail");
    assert!(err.contains("malformed JInt cell [f16 0.1]"), "got: {err}");
    assert!(!err.contains(WIDENED_F16_IMAGE), "no widened image: {err}");

    // A non-string top-level key in a document reaching `to_csv`.
    let bad_doc = adt(
        "JDict",
        vec![RuntimeValue::Dict(vec![(
            int_scalar_of(Prim::Int32, 1),
            adt("JNull", vec![]),
        )])],
    );
    let err = super::csv::csv_to_text(&bad_doc).expect_err("a non-string key must fail");
    assert_no_debug_spelling(&err, "the to_csv document diagnostic");
}

/// Evaluate one Deep expression against hand-supplied runtime bindings.
///
/// The seven eval dispatch arms this package migrated are guards behind
/// the checker: `check_json_builtin_signature` rejects a non-f64 `jnum`
/// operand, a non-int64 `jint` operand, a non-`(string, Json)` `jdict`
/// entry, and a non-f32/f64 `round_to` operand, so no CHECKED program
/// reaches them (pinned below by
/// `fo_diag_migrated_eval_arms_are_checker_front_run`). They exist for
/// dynamically-constructed calls, and their whole job is to name the value
/// they were handed - so the runtime library's own `eval_expr` entry, not
/// the checked pipeline, is where their text is observable.
fn eval_deep_with_bindings(
    surf_expr: &str,
    args: &[(&str, RuntimeValue)],
) -> Result<String, String> {
    let source = format!("probe = {surf_expr}\n");
    let decls = chelis_surf::parser::parse_str(&source).expect("surf parse");
    let exprs = chelis_surf::desugar::desugar_program(&decls);
    let Expr::List(def, _) = &exprs[0] else {
        panic!("desugaring a top-level binding yields one def form");
    };
    // `(def {} <name> <body>)`: the body is the fourth element.
    let body = def.elements[3].clone();

    let empty_tensors: HashMap<String, RuntimeTensorValue> = HashMap::new();
    let mut ctx = EvalContext {
        bindings: HashMap::new(),
        binding_types: HashMap::new(),
        named_axis_route_cache: HashMap::new(),
        named_axis_route_visiting: HashSet::new(),
        top_level_defs: HashMap::new(),
        type_env: HashMap::new(),
        adt_fields: HashMap::new(),
        adt_grad_rejections: HashMap::new(),
        tensor_bindings: &empty_tensors,
        transcript: Vec::new(),
        resolving_top_levels: Vec::new(),
        random_seed: None,
        random_counter: 0,
        cancel: None,
    };
    for (name, value) in args {
        ctx.bindings.insert((*name).to_string(), value.clone());
        ctx.binding_types.insert((*name).to_string(), None);
    }
    ctx.eval_expr(&body).map(|value| render_value(&value))
}

/// One migrated-arm probe: a Surf expression, the runtime bindings its
/// free names take, and the diagnostic fragment the arm must report.
type DispatchProbe<'a> = (&'a str, &'a [(&'a str, RuntimeValue)], &'a str);

/// The migrated eval dispatch arms name the argument they were handed, at
/// its own dtype and value.
///
/// Every expected fragment below is a rewrite, not an addition: the
/// pre-migration text rendered `payload.dtype()` through `Debug` (`F32`,
/// `Int32`, `F16`) or the whole `Option<&RuntimeValue>` through `Debug`
/// (`Some(Scalar(ScalarPayload { value: ScalarValue { bits: I32(5) } }))`),
/// so each `contains` here fails against the pre-migration sources.
#[test]
fn fo_diag_eval_dispatch_arms_report_the_argument_they_were_handed() {
    let f32_scalar = scalar_of(Prim::F32, f64::from(0.1f32));
    let int32_scalar = int_scalar_of(Prim::Int32, 5);
    let f16_scalar = scalar_of(Prim::F16, 0.1);
    let text = RuntimeValue::String("x".to_string());
    let bad_entry = RuntimeValue::List(vec![RuntimeValue::Tuple(vec![
        int_scalar_of(Prim::Int32, 1),
        adt("JNull", vec![]),
    ])]);

    let cases: &[DispatchProbe<'_>] = &[
        (
            "jnum(arg)",
            &[("arg", f32_scalar)],
            "jnum: expected an f64 value, got f32 0.1",
        ),
        (
            "jnum(arg)",
            &[("arg", int32_scalar.clone())],
            "expected f64 arg at index 0, got int32 5",
        ),
        (
            "jint(arg)",
            &[("arg", int32_scalar)],
            "jint: expected an int64 value, got int32 5",
        ),
        (
            "round_to(arg, 2)",
            &[("arg", f16_scalar)],
            "round_to: unsupported operand dtype f16",
        ),
        (
            "round_to(arg, 2)",
            &[("arg", text)],
            "expected float arg at index 0, got string \"x\"",
        ),
        (
            "jdict(arg)",
            &[("arg", bad_entry)],
            "jdict keys must be strings, got int32 1 at index 0",
        ),
    ];

    for (expr, bindings, expected) in cases {
        let err = eval_deep_with_bindings(expr, bindings)
            .expect_err("the dispatch guard must reject this argument");
        assert!(
            err.contains(expected),
            "`{expr}` should report `{expected}`, got: {err}"
        );
        assert_no_debug_spelling(&err, "a migrated eval dispatch diagnostic");
    }
}

/// Why the cell above drives the runtime library directly: every one of
/// those arguments is rejected at CHECK time, so the arm's text is not
/// reachable from a checked program today. If that ever stops being true,
/// this cell goes red and the arm's wording becomes user-facing output
/// that owes an end-to-end expectation.
#[test]
fn fo_diag_migrated_eval_arms_are_checker_front_run() {
    for source in [
        "x = jnum(0.1f32)\n",
        "x = jnum(1i64)\n",
        "x = jint(1i32)\n",
        "x = jint(1.0f64)\n",
        "x = round_to(cast(1.5f64, f16), 2)\n",
        "x = round_to(5i32, 2)\n",
        "x = jdict([(1i64, jnum(1.0f64))])\n",
    ] {
        let decls = chelis_surf::parser::parse_str(source).expect("surf parse");
        let exprs = chelis_surf::desugar::desugar_program(&decls);
        assert!(
            chelis_types::check_ir_program(&exprs).is_err(),
            "the checker must reject `{source}` before eval sees the argument"
        );
    }
}
