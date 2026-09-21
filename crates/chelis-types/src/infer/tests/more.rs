//! Additional unit tests for inference.

use super::*;

#[test]
fn builtin_relu() {
    check_ok(
        "(def {} x (lit {type: (t-tensor {} (d-name {} hidden) (t-prim {} f32))} 0))
         (def {} y (app {} (var {} relu) (var {} x)))",
    );
}

// ── Tensor type tests ────────────────────────────────────────

#[test]
fn tensor_2d() {
    check_ok(
        "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (d-name {} hidden) (t-prim {} f32))} 0))",
    );
}

#[test]
fn tensor_literal_dim() {
    check_ok("(def {} x (lit {type: (t-tensor {} (d-lit {} 512) (t-prim {} f32))} 0))");
}

// ── Edge cases ───────────────────────────────────────────────

#[test]
fn empty_program() {
    let result = check("");
    assert!(result.errors.is_empty());
    assert_eq!(result.typed_nodes, 0);
    assert_eq!(result.total_nodes, 0);
}

#[test]
fn def_with_fn_body() {
    check_ok(
        "(def {} double (fn {} (params {} (x {type: (t-prim {} i32)})) (app {} (var {} add) (var {} x) (var {} x))))",
    );
    check_err(
        "(def {} double (fn {} (params {} x) (app {} (var {} add) (var {} x) (var {} x))))",
        CheckErrorKind::PrecisionMismatch,
    );
}

#[test]
fn nested_let() {
    check_ok(
        "(def {} result
           (let {} (bind {} x (lit {type: (t-prim {} i32)} 1))
             (let {} (bind {} y (var {} x))
               (var {} y))))",
    );
}

#[test]
fn if_with_tensor_branches() {
    check_ok(
        "(def {} cond (lit {type: (t-prim {} bool)} true))
         (def {} a (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0))
         (def {} b (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0))
         (def {} c (if {} (var {} cond) (var {} a) (var {} b)))",
    );
}

#[test]
fn fn_applied_to_args() {
    check_ok(
        "(def {} f (fn {} (params {} x) (var {} x)))
         (def {} result (app {} (var {} f) (lit {type: (t-prim {} i32)} 42)))",
    );
}

#[test]
fn adt_with_fields() {
    check_ok(
        "(deftype {} Pair ()
           (variant {} MkPair
             (field {} fst (t-prim {} i32))
             (field {} snd (t-prim {} f32))))
         (def {} p
           (record {}
             MkPair
             (kv {} fst (lit {type: (t-prim {} i32)} 1))
             (kv {} snd (lit {type: (t-prim {} f32)} 2.0))))",
    );
}

#[test]
fn pipe_with_lambda() {
    check_ok(
        "(def {} result
           (pipe {} (lit {type: (t-prim {} f32)} 1.0)
                    (fn {} (params {} x) (var {} x))))",
    );
}

#[test]
fn total_nodes_counted() {
    let result = check("(def {} x (lit {type: (t-prim {} i32)} 42))");
    assert!(result.total_nodes > 0, "expected some total nodes");
}

#[test]
fn tuple_three_elems() {
    check_ok(
        "(def {} t (tuple {}
           (lit {type: (t-prim {} i32)} 1)
           (lit {type: (t-prim {} f32)} 2.0)
           (lit {type: (t-prim {} bool)} true)))",
    );
}

// ── Regression tests for bug fixes ──────────────────────────────

// Fix 1: scalar arithmetic accepts matching numeric scalars but still rejects bad mixes
#[test]
fn fix1_tensor_op_rejects_non_tensor_args() {
    check_err(
        "(def {} r (app {} (var {} add) (lit {type: (t-prim {} i32)} 1) (lit {type: (t-prim {} bool)} true)))",
        CheckErrorKind::PrecisionMismatch,
    );
}

// Fix 2: unsound generalization — fn x -> { y = x; (y 1, y true) } should fail
#[test]
fn fix2_unsound_generalization_rejected() {
    // x is a monomorphic param, y = x so y is also monomorphic.
    // Applying y to both i32 and bool should fail.
    check_err(
        "(def {} test \
           (fn {} (params {} x) \
             (let {} (bind {} y (var {} x)) \
               (tuple {} \
                 (app {} (var {} y) (lit {type: (t-prim {} i32)} 1)) \
                 (app {} (var {} y) (lit {type: (t-prim {} bool)} true))))))",
        CheckErrorKind::PrecisionMismatch,
    );
}

#[test]
fn level_generalization_quantifies_only_the_ignored_inner_argument() {
    // [04-INF-1] positive twin for `fix2_unsound_generalization_rejected`:
    // `x` belongs to the enclosing lambda and must remain monomorphic, while
    // the ignored `z` is created by the let RHS and may be generalized.
    check_ok(
        "(def {} test \
           (fn {} (params {} x) \
             (let {} (bind {} y (fn {} (params {} z) (var {} x))) \
               (tuple {} \
                 (app {} (var {} y) (lit {type: (t-prim {} i32)} 1)) \
                 (app {} (var {} y) (lit {type: (t-prim {} bool)} true))))))",
    );
}

#[cfg(feature = "generalize-sweep-oracle")]
#[test]
fn independent_binding_generalization_visits_zero_environment_bindings() {
    let mut source = String::new();
    for index in 0..128 {
        source.push_str(&format!(
            "(def {{}} independent_{index} \
               (fn {{}} (params {{}} value_{index}) (var {{}} value_{index})))\n"
        ));
    }

    crate::env::reset_generalize_sweep_env_visits();
    let result = crate::env::without_generalize_sweep_oracle(|| check(&source));
    assert!(
        result.errors.is_empty(),
        "generated independent-binding fixture must check: {:?}",
        result.errors
    );
    assert_eq!(
        crate::env::generalize_sweep_env_visits(),
        0,
        "the production level path must not enumerate environment bindings"
    );
}

// Fix 3: defsig not enforced — body must match declared signature
#[test]
fn fix3_defsig_enforced() {
    check_err(
        "(defsig {} f (t-fn {} (t-prim {} i32) (t-prim {} i32))) \
         (def {} f (lit {type: (t-prim {} bool)} true))",
        CheckErrorKind::TypeMismatch,
    );
}

// Fix 4: d-var names shared within a type — same d-var name maps to same DimVar
#[test]
fn fix4_dvar_names_shared() {
    // Declare a function requiring same dim 'a' in both args.
    // Call with tensor[batch,f32] and tensor[seq,f32] — should fail.
    check_err(
        "(defsig {} myfn (a) \
           (t-fn {} \
             (t-tensor {} (d-var {} a) (t-prim {} f32)) \
             (t-tensor {} (d-var {} a) (t-prim {} f32)) \
             (t-tensor {} (d-var {} a) (t-prim {} f32)))) \
         (def {} myfn (fn {} (params {} x y) (var {} x))) \
         (def {} result (app {} (var {} myfn) \
           (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0) \
           (lit {type: (t-tensor {} (d-name {} seq) (t-prim {} f32))} 0)))",
        CheckErrorKind::DimensionMismatch,
    );
}

// Fix 5: if condition must be bool
#[test]
fn fix5_if_condition_must_be_bool() {
    check_err(
        "(if {} (lit {type: (t-prim {} i32)} 0) \
                (lit {type: (t-prim {} i32)} 1) \
                (lit {type: (t-prim {} i32)} 2))",
        CheckErrorKind::TypeMismatch,
    );
}

// Fix 6a: wildcard satisfies exhaustiveness
#[test]
fn fix6a_wildcard_exhaustive() {
    check_ok(
        "(deftype {} MyOpt (a) (variant {} MySome (t-var {} a)) (variant {} MyNone)) \
         (def {} x (app {} (var {} MySome) (lit {type: (t-prim {} i32)} 42))) \
         (def {} result \
           (match {} (var {} x) \
             (arm {} (pat-wild {}) () (lit {type: (t-prim {} i32)} 0))))",
    );
}

// Fix 6b: pat-as binds name
#[test]
fn fix6b_pat_as_binds_name() {
    check_ok(
        "(deftype {} MyOpt (a) (variant {} MySome (t-var {} a)) (variant {} MyNone)) \
         (def {} x (app {} (var {} MySome) (lit {type: (t-prim {} i32)} 42))) \
         (def {} result \
           (match {} (var {} x) \
             (arm {} (pat-as {} whole (pat-wild {})) () (var {} whole))))",
    );
}

// Fix 7: fitness report includes unresolved names
#[test]
fn fix7_fitness_unresolved_names() {
    let exprs = chelis_deep::parser::parse_str("(def {} x (var {} unknown))").unwrap();
    let result = crate::infer::infer_program(&exprs);
    let report = crate::fitness::FitnessReport::from_infer_result(&result);
    assert!(
        report.unresolved_names.contains(&"unknown".to_string()),
        "expected 'unknown' in unresolved_names, got {:?}",
        report.unresolved_names
    );
    // Check severity is set
    assert!(report.errors.iter().all(|e| e.severity > 0.0));
}

#[test]
fn fitness_uses_the_structured_identifier_instead_of_rendered_prose() {
    let exprs = chelis_deep::parser::parse_str("(def {} x (var {} unknown))").unwrap();
    let mut result = crate::infer::infer_program(&exprs);
    let error = result
        .errors
        .iter_mut()
        .find(|error| matches!(error.kind, CheckErrorKind::UnboundVariable { .. }))
        .expect("unbound diagnostic");
    error.message = "localized name-resolution rendering".to_string();

    let report = crate::fitness::FitnessReport::from_infer_result(&result);
    assert_eq!(report.unresolved_names, vec!["unknown"]);
    assert!(report.components.names < 1.0);
}

// Fix 7b: suggestions populated for UnboundVariable
#[test]
fn fix7b_suggestions_for_unbound() {
    let result = check("(def {} x (var {} typo))");
    let unbound_err = result
        .errors
        .iter()
        .find(|e| matches!(e.kind, CheckErrorKind::UnboundVariable { .. }))
        .expect("expected UnboundVariable error");
    assert!(
        !unbound_err.suggestions.is_empty(),
        "expected suggestions for unbound variable"
    );
}

#[test]
fn ir_literal_dimension_mismatch_surfaces_error() {
    let decls = chelis_surf::parser::parse_str(
        "def want_2x2(a: tensor[2, 2, f32]) -> tensor[f32] = trace(a, 0, 1)\n\
         def main(a: tensor[3, 3, f32]) -> tensor[f32] = want_2x2(a)\n",
    )
    .expect("surf parse");
    let exprs = chelis_surf::desugar::desugar_program(&decls).expect("Surf fixture must desugar");

    let result = infer_ir_program(&exprs);
    assert!(
        result
            .errors
            .iter()
            .any(|error| matches!(error.kind, CheckErrorKind::DimensionMismatch)),
        "expected ir inference to preserve literal dimension mismatches, got {:?}",
        result.errors
    );
    assert!(
        !result
            .errors
            .iter()
            .any(|error| matches!(error.kind, CheckErrorKind::TypeMismatch)),
        "dimension mismatch fixture must not include an unrelated trace return TypeMismatch: {:?}",
        result.errors
    );
}

#[test]
fn ir_rejects_polymorphic_dims_pinned_by_body() {
    let decls = chelis_surf::parser::parse_str(
        "def want_2x2(a: tensor[2, 2, f32]) -> tensor[f32] = trace(a, 0, 1)\n\
         def bad_consumer[m, n](a: tensor[m, n, f32]) -> tensor[f32] = want_2x2(a)\n\
         def main() -> f32 = cast(0.0, f32)\n",
    )
    .expect("surf parse");
    let exprs = chelis_surf::desugar::desugar_program(&decls).expect("Surf fixture must desugar");

    let result = infer_ir_program(&exprs);
    assert!(
        result
            .errors
            .iter()
            .any(|error| matches!(error.kind, CheckErrorKind::DimensionMismatch)),
        "expected ir inference to reject polymorphic dims forced to literals by the body, got {:?}",
        result.errors
    );
    assert!(
        !result
            .errors
            .iter()
            .any(|error| matches!(error.kind, CheckErrorKind::TypeMismatch)),
        "polymorphic dimension fixture must not include an unrelated trace return TypeMismatch: {:?}",
        result.errors
    );
}

#[test]
fn ir_preserves_unresolved_name_errors() {
    let decls = chelis_surf::parser::parse_str(
        "def probe(x: f32) -> f32 = sub(x, frobnicate(x))\n\
         def main() -> f32 = probe(cast(1.0, f32))\n",
    )
    .expect("surf parse");
    let exprs = chelis_surf::desugar::desugar_program(&decls).expect("Surf fixture must desugar");

    let result = infer_ir_program(&exprs);
    assert!(
        result
            .errors
            .iter()
            .any(|error| matches!(error.kind, CheckErrorKind::UnboundVariable { .. })),
        "expected ir inference to preserve unresolved-name errors, got {:?}",
        result.errors
    );

    let report = crate::fitness::check_ir_program(&exprs);
    assert!(
        report.unresolved_names.contains(&"frobnicate".to_string()),
        "expected ir fitness report to include unresolved frobnicate, got {:?}",
        report.unresolved_names
    );
}

// chelis#317: an in-scope constructor resolves through its *exact*
// (consistently mangled) name. Before #317 this test fed a half-mangled
// program — a bare `deftype KVCache` plus a `Pkg__..__KVCache` reference —
// and relied on the registry's fuzzy terminal-segment fallback to bind the
// two. That fuzzy bind is exactly the cross-module mis-resolution #317
// removes, so the program the real reef pipeline produces (deftype AND
// reference carry the same mangled name) is the one that must resolve. Both
// the construction site (`None => KVCache([])`) and the type annotations
// resolve against the same key without any terminal-match fuzziness.
#[test]
fn ir_resolves_consistently_mangled_constructor_names() {
    let decls = chelis_surf::parser::parse_str(
        "type Pkg__chelis__std__Std__Nn__Generate__KVCache[a] = \
            | Pkg__chelis__std__Std__Nn__Generate__KVCache(List[a])\n\
         def keep_cache[p](cache: \
             Option[Pkg__chelis__std__Std__Nn__Generate__KVCache[p]]) -> \
             Pkg__chelis__std__Std__Nn__Generate__KVCache[p] =\n\
           match cache with {\n\
             | Some(value) => value\n\
             | None => Pkg__chelis__std__Std__Nn__Generate__KVCache([])\n\
           }\n",
    )
    .expect("surf parse");
    let exprs = chelis_surf::desugar::desugar_program(&decls).expect("Surf fixture must desugar");

    let result = infer_ir_program(&exprs);
    assert!(
        !result.errors.iter().any(|error| matches!(
            error.kind,
            CheckErrorKind::UnboundVariable { .. } | CheckErrorKind::UnknownConstructor { .. }
        )),
        "expected the consistently mangled constructor to resolve clean, got {:?}",
        result.errors
    );
}

// chelis#317 negative parity: a bare constructor referenced against a
// registry that only holds a *different* (mangled) same-terminal name is
// out of scope and must be an `UnknownConstructor` error — not a silent
// fuzzy bind to the foreign tag that defers to a runtime non-exhaustive
// match. This is the half-mangled state the old
// `ir_resolves_unique_terminal_constructor_names` test accepted.
#[test]
fn ir_rejects_out_of_scope_terminal_constructor_name() {
    let decls = chelis_surf::parser::parse_str(
        "type Pkg__chelis__std__Std__Nn__Generate__KVCache[a] = \
            | Pkg__chelis__std__Std__Nn__Generate__KVCache(List[a])\n\
         def make_cache[p]() -> \
             Pkg__chelis__std__Std__Nn__Generate__KVCache[p] = KVCache([])\n",
    )
    .expect("surf parse");
    let exprs = chelis_surf::desugar::desugar_program(&decls).expect("Surf fixture must desugar");

    let result = infer_ir_program(&exprs);
    assert!(
        result.errors.iter().any(|error| matches!(
            error.kind,
            CheckErrorKind::UnknownConstructor { .. }
        ) && error.message.contains("KVCache")),
        "expected an UnknownConstructor for the out-of-scope bare `KVCache`, got {:?}",
        result.errors
    );

    let diagnostic = result
        .errors
        .iter()
        .find(|error| matches!(error.kind, CheckErrorKind::UnknownConstructor { .. }))
        .expect("unknown constructor diagnostic");
    assert_eq!(diagnostic.kind.unresolved_identifier(), Some("KVCache"));
    let report = crate::fitness::FitnessReport::from_infer_result(&result);
    assert_eq!(report.unresolved_names, vec!["KVCache"]);
    assert!(report.components.names < 1.0);
}

// Fix 8: typed params in Deep
#[test]
fn fix8_typed_params() {
    // fn with typed param x: f32 — using x should give f32
    check_ok("(def {} f (fn {} (params {} (x {type: (t-prim {} f32)})) (var {} x)))");
}

// Fix 8b: typed param enforces type
#[test]
fn fix8b_typed_param_enforced() {
    // Param x is f32, so mixing it with a bool in arithmetic must fail.
    check_err(
        "(def {} f (fn {} (params {} (x {type: (t-prim {} f32)})) \
           (app {} (var {} add) (var {} x) (lit {type: (t-prim {} bool)} true))))",
        CheckErrorKind::PrecisionMismatch,
    );
}

// ── Round 3 regression tests ──────────────────────────────────

#[test]
fn fix9_logical_ops_reject_non_bool_tensors() {
    // and(tensor[batch, f32], tensor[batch, f32]) should fail — requires bool
    check_err(
        "(def {} a (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0)) \
         (def {} b (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0)) \
         (def {} r (app {} (var {} and) (var {} a) (var {} b)))",
        CheckErrorKind::TypeMismatch,
    );
}

#[test]
fn fix9b_logical_ops_reject_non_tensor() {
    // and(i32, i32) should fail — requires tensor
    check_err(
        "(def {} r (app {} (var {} and) (lit {type: (t-prim {} i32)} 1) (lit {type: (t-prim {} i32)} 2)))",
        CheckErrorKind::TypeMismatch,
    );
}

#[test]
fn fix9c_not_rejects_non_bool_tensor() {
    // not(tensor[batch, f32]) should fail — requires bool
    check_err(
        "(def {} a (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0)) \
         (def {} r (app {} (var {} not) (var {} a)))",
        CheckErrorKind::TypeMismatch,
    );
}

#[test]
fn fix9d_logical_ops_accept_bool_tensors() {
    // and(tensor[batch, bool], tensor[batch, bool]) should pass
    check_ok(
        "(def {} a (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} bool))} 0)) \
         (def {} b (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} bool))} 0)) \
         (def {} r (app {} (var {} and) (var {} a) (var {} b)))",
    );
}

#[test]
fn fix10_fitness_has_untyped_nodes() {
    let result = check(
        "(def {} good (lit {type: (t-prim {} i32)} 42)) \
                        (def {} bad (var {} nope))",
    );
    let report = crate::fitness::FitnessReport::from_infer_result(&result);
    assert!(report.untyped_nodes > 0, "expected untyped_nodes > 0");
    // Errors should have severity set
    assert!(
        report.errors.iter().all(|e| e.severity > 0.0),
        "all errors should have severity > 0"
    );
}

// ── Round 4 regression tests ──────────────────────────────────

#[test]
fn fix11_pat_record_rejects_unknown_field() {
    // Define Adam with field lr, then match on nonexistent field 'nope'
    check_err(
        "(deftype {} Optimizer () \
           (variant {} Adam (field {} lr (t-prim {} f32)))) \
         (def {} x (lit {type: (t-adt {} Optimizer)} 0)) \
         (def {} r \
           (match {} (var {} x) \
             (arm {} (pat-record {} Adam (kv {} nope (pat-var {} v))) () (var {} v))))",
        CheckErrorKind::TypeMismatch,
    );
}

#[test]
fn fix11b_pat_record_accepts_valid_field() {
    // Match on actual field lr — should pass
    check_ok(
        "(deftype {} Optimizer () \
           (variant {} Adam (field {} lr (t-prim {} f32)))) \
         (def {} x (lit {type: (t-adt {} Optimizer)} 0)) \
         (def {} r \
           (match {} (var {} x) \
             (arm {} (pat-record {} Adam (kv {} lr (pat-var {} v))) () (var {} v))))",
    );
}

#[test]
fn fix12_structure_score_measured() {
    // check_program runs tag validator — structure should be 1.0 for valid programs
    let exprs =
        chelis_deep::parser::parse_str("(def {} x (lit {type: (t-prim {} i32)} 42))").unwrap();
    let report = crate::fitness::check_program(&exprs);
    assert!(
        (report.components.structure - 1.0).abs() < 0.01,
        "valid program structure should be ~1.0, got {}",
        report.components.structure
    );
}

#[test]
fn checked_program_annotates_fn_bodies() {
    let exprs = chelis_deep::parser::parse_str(
        "(def {} f (fn {} (params {} (x {type: (t-prim {} f32)})) (var {} x)))",
    )
    .unwrap();
    let checked = check_ir_program(&exprs).expect("checked program");
    let text = chelis_deep::printer::print_canonical(checked.annotated_exprs());
    assert!(
        text.contains("(fn {type: (t-fn {} (t-prim {} f32) (t-prim {} f32))}"),
        "expected typed fn metadata, got:\n{text}"
    );
}

#[test]
fn checked_program_annotates_apps_and_updates_type_env() {
    let exprs = chelis_deep::parser::parse_str(
        "(def {} a (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0))
         (def {} b (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0))
         (def {} c (app {} (var {} add) (var {} a) (var {} b)))",
    )
    .unwrap();
    let checked = check_ir_program(&exprs).expect("checked program");
    let text = chelis_deep::printer::print_canonical(checked.annotated_exprs());
    assert!(
        text.contains("(app {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))}"),
        "expected typed app metadata, got:\n{text}"
    );
    assert!(checked.type_env().contains_key("c"));
}

#[test]
fn ir_accepts_symbolic_normalized_axis_for_layer_norm() {
    let exprs = chelis_deep::parser::parse_str(
        "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (d-name {} hidden) (t-prim {} f32))} 0))
         (def {} gamma (lit {type: (t-tensor {} (d-name {} hidden) (t-prim {} f32))} 0))
         (def {} beta (lit {type: (t-tensor {} (d-name {} hidden) (t-prim {} f32))} 0))
         (def {} y (app {} (var {} layer_norm) (var {} x) (var {} gamma) (var {} beta) (lit {type: (t-prim {} f32)} 0.00001)))",
    )
    .unwrap();
    check_ir_program(&exprs)
        .expect("symbolic normalized-axis metadata is legal at the language checker");
}

#[test]
fn checked_program_annotates_symbolic_expand_apps_from_surf() {
    let checked = checked_surf(
        r#"
def predict(
  x: tensor[batch, 64, f32],
  w: tensor[64, 1, f32],
  b: tensor[1, f32]
) -> tensor[batch, 1, f32] =
  add(matmul(x, w), insert(b, 0, batch))
"#,
    );
    let missing = checked
        .annotated_exprs()
        .iter()
        .find_map(missing_shape_sensitive_app);
    assert!(
        missing.is_none(),
        "expected all shape-sensitive apps to be annotated, missing: {:?}",
        missing
    );
}

#[test]
fn surf_permute_with_axis_arguments_type_checks() {
    let checked = checked_surf(
        r#"
def transpose(x: tensor[seq, hidden, f32]) -> tensor[hidden, seq, f32] =
  permute(x, 1, 0)
"#,
    );
    let missing = checked
        .annotated_exprs()
        .iter()
        .find_map(missing_shape_sensitive_app);
    assert!(
        missing.is_none(),
        "expected typed shape-sensitive apps after permute, missing: {:?}",
        missing
    );
}

#[test]
fn surf_list_builtins_type_check() {
    let checked = checked_surf(
        r#"
xs: List[f32] = [1.0, 2.0]
ys = append(xs, 3.0)
total = tensor_to_scalar(sum(to_tensor(ys), 0))
roundtrip = to_list(to_tensor(ys))
"#,
    );
    assert!(checked.annotated_exprs().len() >= 4);
}

#[test]
fn surf_pad_sequences_type_checks() {
    let checked = checked_surf(
        r#"
tokens: List[List[i64]] = [[cast(1, i64), cast(2, i64)], [cast(3, i64)]]
padded = pad_sequences(tokens, cast(0, i64))
"#,
    );
    assert!(checked.annotated_exprs().len() >= 2);
}

#[test]
fn surf_3g_io_and_exact_padding_builtins_type_check() {
    let checked = checked_surf(
        r#"
contents = read_file("dataset.txt")
lines = read_lines("dataset.txt")
bytes = read_bytes("dataset.txt")
exists = file_exists("dataset.txt")
names = list_dir(".")
mapped = mmap_file("dataset.txt")
mapped_len = mmap_len(mapped)
prefix = mmap_read(mapped, cast(0, i64), cast(4, i64))
padded = pad_sequences_to([[cast(1, i64)], [cast(2, i64), cast(3, i64)]], cast(4, i64), cast(0, i64))
"#,
    );
    assert!(checked.annotated_exprs().len() >= 9);
}

// #143: `pad_sequences_to`'s padded (axis-1) dimension equals its
// literal `width` argument. The result type carries `Dim::Lit(width)`
// for that axis (not `Dim::Wildcard`), so a declared return type with
// the matching concrete width type-checks and a mismatched one is
// rejected. The width arrives as `cast(N, i64)` in every caller.

#[test]
fn pad_sequences_to_literal_width_matches_declared_shape() {
    let decls = chelis_surf::parser::parse_str(
        "def f() -> tensor[1, 4, f32] = \
         pad_sequences_to([[cast(10.0, f32)]], cast(4, i64), cast(0.0, f32))\n",
    )
    .expect("surf parse");
    let exprs = chelis_surf::desugar::desugar_program(&decls).expect("Surf fixture must desugar");
    let result = infer_ir_program(&exprs);
    assert!(
        result.errors.is_empty(),
        "literal pad width 4 should match declared tensor[1, 4, f32], got {:?}",
        result.errors
    );
}

#[test]
fn pad_sequences_to_wrong_literal_width_is_rejected() {
    let decls = chelis_surf::parser::parse_str(
        "def f() -> tensor[1, 5, f32] = \
         pad_sequences_to([[cast(10.0, f32)]], cast(4, i64), cast(0.0, f32))\n",
    )
    .expect("surf parse");
    let exprs = chelis_surf::desugar::desugar_program(&decls).expect("Surf fixture must desugar");
    let result = infer_ir_program(&exprs);
    assert!(
        !result.errors.is_empty(),
        "pad width 4 declared as tensor[1, 5, f32] should be a type error"
    );
}

#[test]
fn pad_sequences_to_mismatched_width_through_shared_sig_dim_is_rejected() {
    // The two `pad_sequences_to` results flow into a function whose
    // sig declares the same dim variable `d` on both parameters.
    // Different literal widths (8 vs 5) must collide on `d`.
    let decls = chelis_surf::parser::parse_str(
        "sig demo_unify[s, d, p]: &tensor[s, d, p] -> &tensor[s, d, p] -> tensor[s, d, p]\n\
         def demo_unify(a, b) = a\n\
         def test_mismatch[s, d]() -> tensor[s, d, f32] = {\n\
           q = pad_sequences_to([[cast(0.0, f32)]], cast(8, i64), cast(0.0, f32))\n\
           k = pad_sequences_to([[cast(0.0, f32)]], cast(5, i64), cast(0.0, f32))\n\
           demo_unify(q, k)\n\
         }\n",
    )
    .expect("surf parse");
    let exprs = chelis_surf::desugar::desugar_program(&decls).expect("Surf fixture must desugar");
    let result = infer_ir_program(&exprs);
    assert!(
        result
            .errors
            .iter()
            .any(|error| matches!(error.kind, CheckErrorKind::DimensionMismatch)),
        "widths 8 and 5 sharing sig dim `d` should surface a DimensionMismatch, got {:?}",
        result.errors
    );
}

#[test]
fn surf_dict_and_iteration_builtins_type_check() {
    let checked = checked_surf(
        r#"
keys: List[string] = ["alpha", "beta"]
ids: List[i64] = [cast(1, i64), cast(2, i64)]
pairs = zip(keys, ids)
indexed = enumerate(keys)
vocab: Dict[string, i64] = dict_of(pairs)
found = dict_contains(vocab, "alpha")
id = dict_get(vocab, "beta")
only_keys = dict_keys(vocab)
only_values = dict_values(vocab)
roundtrip = dict_entries(vocab)
"#,
    );
    assert!(checked.annotated_exprs().len() >= 9);
}

#[test]
fn surf_3h_tensor_numeric_builtins_type_check() {
    let checked = checked_surf(
        r#"
def projection(
  x: tensor[batch, seq, hidden, f32],
  w: tensor[hidden, out_dim, f32],
  token_ids: tensor[batch, seq, i64],
  mask: tensor[batch, seq, out_dim, bool],
  table: tensor[vocab, out_dim, f32]
) -> tensor[batch, seq, out_dim, f32] = {
  logits = einsum("bsh,ho->bso", x, w)
  embed = gather(table, token_ids, 0)
  clipped = clamp(add(logits, embed), scalar_to_tensor(0.0), scalar_to_tensor(6.0))
  running = cumsum(clipped, 1)
  where(mask, running, clipped)
}
"#,
    );
    assert!(!checked.annotated_exprs().is_empty());
}

#[test]
fn surf_3h_structural_tensor_builtins_type_check() {
    let checked = checked_surf(
        r#"
def pack_heads(
  q: tensor[batch, seq, 2, f32],
  k: tensor[batch, seq, 2, f32]
) -> tensor[batch, seq, *, f32] = {
  packed = concat([q, k], 2)
  pieces = split(packed, 2, [2, 2])
  concat(pieces, 2)
}
"#,
    );
    assert!(!checked.annotated_exprs().is_empty());
}

// ── chelis#631: concat result typing (spec/04-type-system.md §4.5.4) ──

/// Axis-0 concat of two `[1, 2]` rows types `[2, 2]`: the concat axis
/// is the element extent times the statically-counted element count,
/// every other axis unchanged. Pre-chelis#631 the result kept the
/// element's `1` on axis 0 and wildcarded the LAST axis, so this
/// concrete return annotation was rejected.
#[test]
fn issue631_concat_axis0_literal_list_counts_elements() {
    let result = infer_surf(
        r#"
module Repro.ConcatCount
def stack_rows(a: tensor[1, 2, f32], b: tensor[1, 2, f32]) -> tensor[2, 2, f32] =
  concat([a, b], 0)
"#,
    );
    assert!(
        result.errors.is_empty(),
        "axis-0 concat of two [1, 2] rows must type [2, 2], got: {:?}",
        result.errors
    );
}

/// The chelis#631 reproducer shape: the list is LET-BOUND, so the
/// concat site sees a variable. The binding carries its literal
/// length to the concat (Env::list_literal_len).
#[test]
fn issue631_concat_axis0_let_bound_list_counts_elements() {
    let result = infer_surf(
        r#"
module Repro.ConcatCountLet
def stack_rows(a: tensor[1, 2, f32], b: tensor[1, 2, f32]) -> tensor[2, 2, f32] = {
  rows = [a, b]
  concat(rows, 0)
}
"#,
    );
    assert!(
        result.errors.is_empty(),
        "let-bound list concat must count elements through the binding, got: {:?}",
        result.errors
    );
}

/// Negative parity: a wrong concat-axis sum is rejected — the
/// computed `Lit(2)` is a real claim, not a permissive wildcard.
#[test]
fn issue631_concat_wrong_sum_annotation_rejected() {
    let result = infer_surf(
        r#"
module Repro.ConcatWrongSum
def stack_rows(a: tensor[1, 2, f32], b: tensor[1, 2, f32]) -> tensor[3, 2, f32] =
  concat([a, b], 0)
"#,
    );
    assert!(
        !result.errors.is_empty(),
        "an axis-0 concat of two [1, 2] rows must not satisfy [3, 2]"
    );
}

/// A non-last-axis concat sizes the CONCAT axis and preserves the
/// trailing axes. Pre-chelis#631 the last axis was wildcarded
/// unconditionally (the chelis#594 symptom).
#[test]
fn issue631_concat_mid_axis_preserves_last_axis() {
    let accepted = infer_surf(
        r#"
module Repro.ConcatMidAxis
def stack_mid(a: tensor[2, 1, 5, f32], b: tensor[2, 1, 5, f32]) -> tensor[2, 2, 5, f32] =
  concat([a, b], 1)
"#,
    );
    assert!(
        accepted.errors.is_empty(),
        "axis-1 concat must type [2, 2, 5], got: {:?}",
        accepted.errors
    );
    // The last axis is the element's `5`, not a wildcard: a wrong
    // trailing extent no longer slips through.
    let rejected = infer_surf(
        r#"
module Repro.ConcatMidAxisBad
def stack_mid(a: tensor[2, 1, 5, f32], b: tensor[2, 1, 5, f32]) -> tensor[2, 2, 6, f32] =
  concat([a, b], 1)
"#,
    );
    assert!(
        !rejected.errors.is_empty(),
        "the non-concat trailing axis must stay 5; [2, 2, 6] must be rejected"
    );
}

/// A non-enumerable list (a `List` parameter) wildcards the CONCAT
/// axis — and ONLY the concat axis: the trailing axis keeps the
/// element extent (pre-chelis#631 it was the wildcarded one).
#[test]
fn issue631_concat_param_list_wildcards_concat_axis_only() {
    let accepted = infer_surf(
        r#"
module Repro.ConcatParamList
def cat_all(xs: List[tensor[2, 3, f32]]) -> tensor[*, 3, f32] =
  concat(xs, 0)
"#,
    );
    assert!(
        accepted.errors.is_empty(),
        "param-list concat must type [*, 3], got: {:?}",
        accepted.errors
    );
    let rejected = infer_surf(
        r#"
module Repro.ConcatParamListBad
def cat_all(xs: List[tensor[2, 3, f32]]) -> tensor[*, 4, f32] =
  concat(xs, 0)
"#,
    );
    assert!(
        !rejected.errors.is_empty(),
        "the non-concat trailing axis must stay 3; [*, 4] must be rejected"
    );
}

/// A runtime (non-literal) concat axis wildcards EVERY axis at the
/// element rank: any axis may be the one that grows. An
/// axis-0-concat-shaped annotation must typecheck even though axis 0
/// carried a literal in the element type (pre-chelis#631 the element's
/// `2` was pinned and `[4, 3]` was rejected).
#[test]
fn issue631_concat_dynamic_axis_wildcards_all_axes() {
    let result = infer_surf(
        r#"
module Repro.ConcatDynAxis
def cat_dyn(a: tensor[2, 3, f32], b: tensor[2, 3, f32], ax: i32) -> tensor[4, 3, f32] =
  concat([a, b], ax)
"#,
    );
    assert!(
        result.errors.is_empty(),
        "runtime-axis concat must not pin any per-axis extent, got: {:?}",
        result.errors
    );
}

/// An out-of-bounds literal concat axis is a check-time error.
#[test]
fn issue631_concat_axis_out_of_bounds_rejected() {
    let result = infer_surf(
        r#"
module Repro.ConcatOob
def cat_oob(a: tensor[2, f32], b: tensor[2, f32]) -> tensor[*, f32] =
  concat([a, b], 5)
"#,
    );
    assert!(
        result
            .errors
            .iter()
            .any(|e| e.message.contains("concat axis 5 out of bounds for rank 1")),
        "a literal axis past the element rank must error, got: {:?}",
        result.errors
    );
}

/// A negative literal axis indexes from the end (`-1` = last axis),
/// matching every other axis-taking builtin.
#[test]
fn issue631_concat_negative_axis_indexes_from_end() {
    let accepted = infer_surf(
        r#"
module Repro.ConcatNegAxis
def cat_neg(a: tensor[1, 2, f32], b: tensor[1, 2, f32]) -> tensor[1, 4, f32] =
  concat([a, b], -1)
"#,
    );
    assert!(
        accepted.errors.is_empty(),
        "axis -1 concat of two [1, 2] rows must type [1, 4], got: {:?}",
        accepted.errors
    );
    let rejected = infer_surf(
        r#"
module Repro.ConcatNegAxisBad
def cat_neg(a: tensor[1, 2, f32], b: tensor[1, 2, f32]) -> tensor[1, 5, f32] =
  concat([a, b], -1)
"#,
    );
    assert!(
        !rejected.errors.is_empty(),
        "axis -1 concat sum is 4; [1, 5] must be rejected"
    );
}

/// chelis#594 CONSCIOUS FLIP of the original chelis#631 head-bias
/// boundary pin: a DIRECT literal list now sums PER-ELEMENT extents,
/// so `[tensor[2], tensor[*]]` no longer inherits the §4.5.2
/// head-biased join times the count (`Lit(4)`) — the wildcard
/// element makes the sum honestly unknown and the concat axis
/// wildcards (both `[4]` and `[5]` typecheck). The head bias remains
/// observable on the BINDING path, where only the joined element
/// type and the literal length survive.
#[test]
fn issue594_mixed_wildcard_element_wildcards_direct_concat_axis() {
    for claim in ["4", "5"] {
        let result = infer_surf(&format!(
            r#"
module Repro.ConcatHeadBias
def cat_bias(a: tensor[2, f32], b: tensor[*, f32]) -> tensor[{claim}, f32] =
  concat([a, b], 0)
"#
        ));
        assert!(
            result.errors.is_empty(),
            "a wildcard element makes the direct sum unknown; [{claim}] must typecheck, got: {:?}",
            result.errors
        );
    }
    // Binding path: the join is head-biased to the element `[2]` and
    // the length is 2, so the concat axis is `Lit(4)` — `[5]` rejects.
    let rejected = infer_surf(
        r#"
module Repro.ConcatHeadBiasBinding
def cat_bias(a: tensor[2, f32], b: tensor[*, f32]) -> tensor[5, f32] = {
  rows = [a, b]
  concat(rows, 0)
}
"#,
    );
    assert!(
        !rejected.errors.is_empty(),
        "the binding path keeps the head-biased join times count (Lit(4)); [5] must be rejected"
    );
}

// ── chelis#594: ragged direct-literal concat sums per-element extents ──

/// A DIRECT literal list with ragged extents sums them: `[2] + [3]`
/// on axis 0 types `[5]`. Pre-chelis#594 the §4.5.2 join widened the
/// mismatched literals to `*` before the concat rule saw them, so
/// the sum was unrecoverable and the axis stayed a wildcard.
#[test]
fn issue594_ragged_direct_literal_concat_sums_extents() {
    let accepted = infer_surf(
        r#"
module Repro.RaggedConcat
def cat(a: tensor[2, f32], b: tensor[3, f32]) -> tensor[5, f32] =
  concat([a, b], 0)
"#,
    );
    assert!(
        accepted.errors.is_empty(),
        "ragged [2] + [3] must type [5], got: {:?}",
        accepted.errors
    );
    let rejected = infer_surf(
        r#"
module Repro.RaggedConcatBad
def cat(a: tensor[2, f32], b: tensor[3, f32]) -> tensor[6, f32] =
  concat([a, b], 0)
"#,
    );
    assert!(
        !rejected.errors.is_empty(),
        "ragged sum is 5; [6] must be rejected"
    );
}

/// Three ragged elements and a non-zero concat axis: concatenating
/// `[4, 1]`, `[4, 2]`, and `[4, 3]` on axis 1 types `[4, 6]`, with
/// the non-concat axis preserved.
#[test]
fn issue594_ragged_three_element_mid_axis_concat_sums_extents() {
    let accepted = infer_surf(
        r#"
module Repro.RaggedConcat3
def cat(a: tensor[4, 1, f32], b: tensor[4, 2, f32], c: tensor[4, 3, f32]) -> tensor[4, 6, f32] =
  concat([a, b, c], 1)
"#,
    );
    assert!(
        accepted.errors.is_empty(),
        "ragged axis-1 sum must type [4, 6], got: {:?}",
        accepted.errors
    );
    let rejected = infer_surf(
        r#"
module Repro.RaggedConcat3Bad
def cat(a: tensor[4, 1, f32], b: tensor[4, 2, f32], c: tensor[4, 3, f32]) -> tensor[4, 7, f32] =
  concat([a, b, c], 1)
"#,
    );
    assert!(
        !rejected.errors.is_empty(),
        "ragged axis-1 sum is 6; [4, 7] must be rejected"
    );
}

/// Ragged extents are NOT recoverable through a binding — only the
/// literal length survives (`Env::list_literal_lens`), and the
/// §4.5.2 join has already widened the mismatched extents to `*`. A
/// let-bound ragged list therefore keeps the honest wildcard: a
/// claim the direct form would reject is accepted permissively.
/// Deliberate boundary, pinned so a future extension is conscious.
#[test]
fn issue594_ragged_let_bound_list_stays_wildcard() {
    let result = infer_surf(
        r#"
module Repro.RaggedConcatLet
def cat(a: tensor[2, f32], b: tensor[3, f32]) -> tensor[9, f32] = {
  rows = [a, b]
  concat(rows, 0)
}
"#,
    );
    assert!(
        result.errors.is_empty(),
        "a let-bound ragged list wildcards the concat axis (permissive), got: {:?}",
        result.errors
    );
}

/// Add-symmetric staleness pin (mirrors the size-provenance BLOCKER-B
/// discipline): a name re-bound from a list literal to a non-literal
/// list must NOT keep the stale literal length. With the stale length
/// the concat would type `Lit(2)` and reject `[7, 2]`; the cleared
/// binding honestly wildcards the concat axis.
#[test]
fn issue631_concat_rebound_list_clears_stale_length() {
    let result = infer_surf(
        r#"
module Repro.ConcatRebind
def cat_rebind(a: tensor[1, 2, f32], b: tensor[1, 2, f32], xs: List[tensor[1, 2, f32]]) -> tensor[7, 2, f32] = {
  rows = [a, b]
  rows = xs
  concat(rows, 0)
}
"#,
    );
    assert!(
        result.errors.is_empty(),
        "a re-bound list name must clear its stale literal length, got: {:?}",
        result.errors
    );
}

/// Shadow pin (mirrors size-provenance BLOCKER C): a `List` value
/// parameter that shadows an outer list-literal binding must not
/// inherit the outer literal length through the env clone.
#[test]
fn issue631_concat_param_shadow_clears_outer_length() {
    let result = infer_surf(
        r#"
module Repro.ConcatShadow
rows = [to_tensor([cast(1.0, f32), cast(2.0, f32)])]
def cat_shadow(rows: List[tensor[2, f32]]) -> tensor[9, f32] =
  concat(rows, 0)
"#,
    );
    assert!(
        result.errors.is_empty(),
        "a shadowing List param must clear the outer literal length, got: {:?}",
        result.errors
    );
}

#[test]
fn surf_3h_sort_and_trace_type_check() {
    let checked = checked_surf(
        r#"
def summarize(x: tensor[batch, hidden, hidden, f32]) -> (tensor[batch, hidden, f32], tensor[batch, hidden, i64], tensor[batch, f32]) = {
  diag = diagonal(x, 1, 2)
  sorted = sort(diag, 1)
  values = sorted.0
  indices = sorted.1
  total = trace(x, 1, 2)
  (values, indices, total)
}
"#,
    );
    assert!(!checked.annotated_exprs().is_empty());
}

#[test]
fn surf_einsum_rejects_ellipsis_in_3h() {
    let result = check_ir_program(
        &chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(
                r#"
def bad(
  x: tensor[batch, seq, hidden, f32],
  w: tensor[hidden, out_dim, f32]
) -> tensor[batch, seq, out_dim, f32] =
  einsum("...h,ho->...o", x, w)
"#,
            )
            .expect("surf parse"),
        )
        .expect("Surf fixture must desugar"),
    );
    let err = result.expect_err("ellipsis should be rejected in 3h einsum");
    assert!(
        err.errors.iter().any(|error| {
            error.message.contains("einsum")
                && (error.message.contains("ellipsis") || error.message.contains("..."))
        }),
        "expected einsum ellipsis rejection, got {:?}",
        err.errors
    );
}

#[test]
fn surf_where_rejects_non_bool_condition_tensor() {
    let result = check_ir_program(
        &chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(
                r#"
def bad(
  cond: tensor[batch, hidden, f32],
  x: tensor[batch, hidden, f32],
  y: tensor[batch, hidden, f32]
) -> tensor[batch, hidden, f32] =
  where(cond, x, y)
"#,
            )
            .expect("surf parse"),
        )
        .expect("Surf fixture must desugar"),
    );
    let err = result.expect_err("where should reject non-bool condition tensors");
    assert!(
        err.errors
            .iter()
            .any(|error| { error.message.contains("where") && error.message.contains("bool") }),
        "expected where bool mismatch, got {:?}",
        err.errors
    );
}

#[test]
fn surf_scatter_replace_rejects_unknown_mode() {
    let result = check_ir_program(
        &chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(
                r#"
def bad(
  base: tensor[seq, hidden, f32],
  ids: tensor[seq, i64],
  updates: tensor[seq, hidden, f32]
) -> tensor[seq, hidden, f32] =
  scatter(base, ids, updates, 0, "last")
"#,
            )
            .expect("surf parse"),
        )
        .expect("Surf fixture must desugar"),
    );
    let err = result.expect_err("scatter should reject unsupported mode");
    assert!(
        err.errors.iter().any(|error| {
            error.message.contains("scatter")
                && error.message.contains("replace")
                && error.message.contains("add")
        }),
        "expected scatter mode rejection, got {:?}",
        err.errors
    );
}

#[test]
fn surf_einsum_rejects_static_extent_mismatch() {
    let result = check_ir_program(
        &chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(
                r#"
a = pad_sequences([[1.0, 2.0], [3.0, 4.0]], 0.0)
b = pad_sequences([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]], 0.0)
out = einsum("ij,jk->ik", a, b)
"#,
            )
            .expect("surf parse"),
        )
        .expect("Surf fixture must desugar"),
    );
    let err = result.expect_err("static einsum extent mismatch should be rejected");
    assert!(
        err.errors.iter().any(|error| {
            error.message.contains("einsum") && error.message.contains("inconsistent extents")
        }),
        "expected einsum extent mismatch, got {:?}",
        err.errors
    );
}

#[test]
fn surf_scatter_replace_accepts_static_duplicate_indices() {
    let result = check_ir_program(
        &chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(
                r#"
base = pad_sequences([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], 0.0)
ids: List[i64] = [cast(1, i64), cast(1, i64)]
idx = to_tensor(ids)
updates = pad_sequences([[5.0, 5.0], [6.0, 6.0]], 0.0)
out = scatter(base, idx, updates, 0, "replace")
"#,
            )
            .expect("surf parse"),
        )
        .expect("Surf fixture must desugar"),
    );
    result.expect("static scatter duplicate indices follow deterministic last-write-wins");
}

#[test]
fn surf_map_filter_fold_type_check() {
    let checked = checked_surf(
        r#"
def inc(x: i64) -> i64 = add(x, cast(1, i64))
xs: List[i64] = [cast(1, i64), cast(2, i64), cast(3, i64)]
mapped = map(inc, xs)
filtered = filter(fn (x: i64) -> eq(mod(x, cast(2, i64)), cast(0, i64)), mapped)
total = fold(fn (acc: i64, x: i64) -> add(acc, x), cast(0, i64), filtered)
"#,
    );
    assert!(checked.annotated_exprs().len() >= 5);
}

#[test]
fn surf_append_rejects_wrong_element_type() {
    let result = check_ir_program(
        &chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(
                r#"
xs: List[i64] = [cast(1, i64)]
bad = append(xs, "oops")
"#,
            )
            .expect("surf parse"),
        )
        .expect("Surf fixture must desugar"),
    );
    let err = result.expect_err("append should reject mismatched element type");
    assert!(
        err.errors
            .iter()
            .any(|error| error.message.contains("List") || error.message.contains("string")),
        "expected list element mismatch, got {:?}",
        err.errors
    );
}

#[test]
fn surf_filter_rejects_non_bool_callback() {
    let result = check_ir_program(
        &chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(
                r#"
xs: List[i64] = [cast(1, i64)]
bad = filter(fn (x: i64) -> add(x, cast(1, i64)), xs)
"#,
            )
            .expect("surf parse"),
        )
        .expect("Surf fixture must desugar"),
    );
    let err = result.expect_err("filter should reject non-bool callback");
    assert!(
        err.errors
            .iter()
            .any(|error| error.message.contains("filter")
                && error.message.contains("bool")
                && error.message.contains("callback")),
        "expected bool callback mismatch, got {:?}",
        err.errors
    );
}

#[test]
fn surf_dict_entries_rejects_non_dict_input() {
    let result = check_ir_program(
        &chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(
                r#"
xs: List[i64] = [cast(1, i64)]
bad = dict_entries(xs)
"#,
            )
            .expect("surf parse"),
        )
        .expect("Surf fixture must desugar"),
    );
    let err = result.expect_err("dict_entries should reject non-dict input");
    assert!(
        err.errors
            .iter()
            .any(|error| error.message.contains("dict_entries") || error.message.contains("Dict")),
        "expected dict input mismatch, got {:?}",
        err.errors
    );
}

#[test]
fn surf_to_list_rejects_rank2_tensor() {
    let result = check_ir_program(
        &chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(
                r#"
def bad(x: tensor[2, 2, f32]) -> List[f32] = to_list(x)
"#,
            )
            .expect("surf parse"),
        )
        .expect("Surf fixture must desugar"),
    );
    let err = result.expect_err("to_list should reject rank-2 tensor input");
    assert!(
        err.errors.iter().any(|error| {
            error.message.contains("rank-1 tensor") || error.message.contains("to_list")
        }),
        "expected rank mismatch, got {:?}",
        err.errors
    );
}

#[test]
fn surf_fold_rejects_accumulator_mismatch() {
    let result = check_ir_program(
        &chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(
                r#"
xs: List[i64] = [cast(1, i64)]
bad = fold(fn (acc: string, x: i64) -> string_concat(acc, to_string(x)), cast(0, i64), xs)
"#,
            )
            .expect("surf parse"),
        )
        .expect("Surf fixture must desugar"),
    );
    let err = result.expect_err("fold should reject mismatched accumulator type");
    assert!(
        err.errors.iter().any(|error| error.message.contains("fold")
            && error.message.contains("accumulator")
            && error.message.contains("string")
            && error.message.contains("i64")),
        "expected accumulator mismatch, got {:?}",
        err.errors
    );
}

fn surf_tuple_fold_tensor_slot_program() -> Vec<deep::Expr> {
    chelis_surf::desugar::desugar_program(
        &chelis_surf::parser::parse_str(
            r#"
def f[n](xs: tensor[n, f32]) -> (tensor[n, f32], i64) = {
  idxs = range(cast(0, i64), numel(copy(xs)))
  state0 = (to_tensor(map(fn (x: f32) -> cast(0.0, f32), to_list(copy(xs)))), cast(0, i64))
  step = fn (state, i) -> {
acc = state.0
total = state.1
(acc, add(total, i))
  }
  fold(step, state0, idxs)
}

out = f(to_tensor([1.0, 2.0, 3.0]))
"#,
        )
        .expect("surf parse"),
    )
    .expect("Surf fixture must desugar")
}

#[test]
fn surf_polymorphic_tuple_fold_with_tensor_slot_infers_ir() {
    let result = infer_ir_program(&surf_tuple_fold_tensor_slot_program());
    assert!(
        result.errors.is_empty(),
        "ir inference should succeed without overflowing: {:?}",
        result.errors
    );
}

#[test]
fn surf_polymorphic_tuple_fold_with_tensor_slot_annotates_ir() {
    let program = surf_tuple_fold_tensor_slot_program();
    check_typed_program(&program).expect("annotation should succeed without overflowing");
}

#[test]
fn surf_polymorphic_tuple_fold_with_tensor_slot_type_checks() {
    let result = check_ir_program(&surf_tuple_fold_tensor_slot_program());
    result.expect("polymorphic tuple fold should type check without overflowing");
}

#[test]
fn surf_collection_helper_builtins_type_check() {
    let result = check_ir_program(
        &chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(
                r#"
xs: List[i64] = [cast(1, i64), cast(2, i64), cast(3, i64)]
prefix = take(xs, cast(2, i64))
suffix = skip(xs, cast(1, i64))
groups = chunk(xs, cast(2, i64))
scanned = scan(fn (acc: i64, x: i64) -> add(acc, x), cast(0, i64), xs)
buckets = partition(fn (x: i64) -> gt(x, cast(1, i64)), xs)
exploded = flat_map(fn (x: i64) -> [x, add(x, cast(10, i64))], xs)
flattened = flatten([[cast(1, i64)], [cast(2, i64), cast(3, i64)]])
base: Dict[string, i64] = dict_of([("alpha", cast(1, i64))])
extended = dict_insert(base, "beta", cast(2, i64))
merged = dict_merge(extended, dict_of([("beta", cast(20, i64)), ("gamma", cast(3, i64))]))
trimmed = dict_remove(merged, "gamma")
"#,
            )
            .expect("surf parse"),
        )
        .expect("Surf fixture must desugar"),
    );
    result.expect("collection helper builtins should type check");
}

#[test]
fn surf_take_rejects_non_integer_count() {
    let result = check_ir_program(
        &chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(
                r#"
xs: List[i64] = [cast(1, i64), cast(2, i64)]
bad = take(xs, "two")
"#,
            )
            .expect("surf parse"),
        )
        .expect("Surf fixture must desugar"),
    );
    let err = result.expect_err("take should reject non-integer count");
    assert!(
        err.errors
            .iter()
            .any(|error| error.message.contains("take") || error.message.contains("integer")),
        "expected integer count mismatch, got {:?}",
        err.errors
    );
}

#[test]
fn surf_dict_insert_rejects_value_type_mismatch() {
    let result = check_ir_program(
        &chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(
                r#"
base: Dict[string, i64] = dict_of([("alpha", cast(1, i64))])
bad = dict_insert(base, "beta", "two")
"#,
            )
            .expect("surf parse"),
        )
        .expect("Surf fixture must desugar"),
    );
    let err = result.expect_err("dict_insert should reject mismatched value type");
    assert!(
        err.errors
            .iter()
            .any(|error| error.message.contains("dict_insert")
                || error.message.contains("i64")
                || error.message.contains("string")),
        "expected dict value mismatch, got {:?}",
        err.errors
    );
}

#[test]
fn surf_dict_merge_rejects_mismatched_dict_value_types() {
    let result = check_ir_program(
        &chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(
                r#"
lhs: Dict[string, i64] = dict_of([("alpha", cast(1, i64))])
rhs: Dict[string, string] = dict_of([("beta", "two")])
bad = dict_merge(lhs, rhs)
"#,
            )
            .expect("surf parse"),
        )
        .expect("Surf fixture must desugar"),
    );
    let err = result.expect_err("dict_merge should reject mismatched dict value types");
    assert!(
        err.errors
            .iter()
            .any(|error| error.message.contains("dict_merge")
                || error.message.contains("Dict")
                || error.message.contains("i64")
                || error.message.contains("string")),
        "expected dict merge mismatch, got {:?}",
        err.errors
    );
}

#[test]
fn surf_scan_rejects_accumulator_mismatch() {
    let result = check_ir_program(
        &chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(
                r#"
xs: List[i64] = [cast(1, i64)]
bad = scan(fn (acc: string, x: i64) -> string_concat(acc, to_string(x)), cast(0, i64), xs)
"#,
            )
            .expect("surf parse"),
        )
        .expect("Surf fixture must desugar"),
    );
    let err = result.expect_err("scan should reject mismatched accumulator type");
    assert!(
        err.errors.iter().any(|error| error.message.contains("scan")
            && error.message.contains("accumulator")
            && error.message.contains("string")
            && error.message.contains("i64")),
        "expected scan accumulator mismatch, got {:?}",
        err.errors
    );
}

#[test]
fn surf_partition_rejects_non_bool_callback() {
    let result = check_ir_program(
        &chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(
                r#"
xs: List[i64] = [cast(1, i64)]
bad = partition(fn (x: i64) -> add(x, cast(1, i64)), xs)
"#,
            )
            .expect("surf parse"),
        )
        .expect("Surf fixture must desugar"),
    );
    let err = result.expect_err("partition should reject non-bool callback");
    assert!(
        err.errors
            .iter()
            .any(|error| error.message.contains("partition")
                && error.message.contains("bool")
                && error.message.contains("callback")),
        "expected partition callback mismatch, got {:?}",
        err.errors
    );
}

#[test]
fn surf_flat_map_rejects_non_list_callback() {
    let result = check_ir_program(
        &chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(
                r#"
xs: List[i64] = [cast(1, i64)]
bad = flat_map(fn (x: i64) -> add(x, cast(1, i64)), xs)
"#,
            )
            .expect("surf parse"),
        )
        .expect("Surf fixture must desugar"),
    );
    let err = result.expect_err("flat_map should reject non-list callback");
    assert!(
        err.errors
            .iter()
            .any(|error| error.message.contains("List") || error.message.contains("flat_map")),
        "expected flat_map callback mismatch, got {:?}",
        err.errors
    );
}

#[test]
fn surf_flatten_rejects_non_nested_list_input() {
    let result = check_ir_program(
        &chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(
                r#"
xs: List[i64] = [cast(1, i64)]
bad = flatten(xs)
"#,
            )
            .expect("surf parse"),
        )
        .expect("Surf fixture must desugar"),
    );
    let err = result.expect_err("flatten should reject non-nested list input");
    assert!(
        err.errors
            .iter()
            .any(|error| error.message.contains("flatten") || error.message.contains("List")),
        "expected flatten input mismatch, got {:?}",
        err.errors
    );
}

#[test]
fn surf_dict_remove_rejects_mismatched_key_type() {
    let result = check_ir_program(
        &chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(
                r#"
base: Dict[string, i64] = dict_of([("alpha", cast(1, i64))])
bad = dict_remove(base, cast(7, i64))
"#,
            )
            .expect("surf parse"),
        )
        .expect("Surf fixture must desugar"),
    );
    let err = result.expect_err("dict_remove should reject mismatched key type");
    assert!(
        err.errors.iter().any(|error| {
            error.message.contains("dict_remove")
                || (error.message.contains("string") && error.message.contains("i64"))
        }),
        "expected dict_remove key mismatch, got {:?}",
        err.errors
    );
}

#[test]
fn surf_chunk_rejects_non_integer_size() {
    let result = check_ir_program(
        &chelis_surf::desugar::desugar_program(
            &chelis_surf::parser::parse_str(
                r#"
xs: List[i64] = [cast(1, i64)]
bad = chunk(xs, "two")
"#,
            )
            .expect("surf parse"),
        )
        .expect("Surf fixture must desugar"),
    );
    let err = result.expect_err("chunk should reject non-integer size");
    assert!(
        err.errors
            .iter()
            .any(|error| error.message.contains("chunk") || error.message.contains("integer")),
        "expected chunk size mismatch, got {:?}",
        err.errors
    );
}

// ------------------------------------------------------------------
// Top-level binding cycle detection
// ------------------------------------------------------------------

fn ir_errors_from_surf(src: &str) -> Vec<CheckError> {
    let decls = chelis_surf::parser::parse_str(src).expect("surf parse");
    let exprs = chelis_surf::desugar::desugar_program(&decls).expect("Surf fixture must desugar");
    infer_ir_program(&exprs).errors
}

#[test]
fn nautilus_self_reference_is_allowed() {
    // The single-hop identity `x = (x : tensor[...])` is a pinned Nautilus
    // external-input pattern and must NOT be flagged as a binding cycle.
    let errors = ir_errors_from_surf("x = (x : tensor[4, f32])\n");
    assert!(
        !errors
            .iter()
            .any(|e| matches!(e.kind, CheckErrorKind::CycleDetected)),
        "Nautilus self-reference should not be flagged; got {errors:?}"
    );
}

#[test]
fn two_hop_binding_cycle_is_detected() {
    let errors = ir_errors_from_surf(
        "a = (b : tensor[4, f32])\n\
         b = (a : tensor[4, f32])\n",
    );
    let cycle_err = errors
        .iter()
        .find(|e| matches!(e.kind, CheckErrorKind::CycleDetected))
        .unwrap_or_else(|| {
            panic!("expected a CycleDetected error for a two-hop cycle; got {errors:?}")
        });
    assert!(
        cycle_err.message.contains("binding cycle"),
        "message should mention 'binding cycle'; got {:?}",
        cycle_err.message
    );
    assert!(
        cycle_err.message.contains("a -> b -> a"),
        "expected 'a -> b -> a' in message; got {:?}",
        cycle_err.message
    );
}

#[test]
fn three_hop_binding_cycle_is_detected() {
    let errors = ir_errors_from_surf(
        "a = (b : tensor[4, f32])\n\
         b = (c : tensor[4, f32])\n\
         c = (a : tensor[4, f32])\n",
    );
    let cycle_err = errors
        .iter()
        .find(|e| matches!(e.kind, CheckErrorKind::CycleDetected))
        .unwrap_or_else(|| {
            panic!("expected a CycleDetected error for a three-hop cycle; got {errors:?}")
        });
    assert!(
        cycle_err.message.contains("a -> b -> c -> a"),
        "expected 'a -> b -> c -> a' in message; got {:?}",
        cycle_err.message
    );
}

#[test]
fn unrelated_defs_do_not_trigger_cycle_false_positive() {
    // Sanity: multiple Nautilus self-references together should still pass.
    let errors = ir_errors_from_surf(
        "x = (x : tensor[4, f32])\n\
         y = (y : tensor[4, f32])\n",
    );
    assert!(
        !errors
            .iter()
            .any(|e| matches!(e.kind, CheckErrorKind::CycleDetected)),
        "multiple independent Nautilus inputs must not trigger a cycle error; got {errors:?}"
    );
}

// ── chelis#293: general type var through a function-typed parameter ──
//
// A def generic over a general type variable `P` declared in its
// explicit `[..]` quantifier list, where `P` is threaded through a
// function-typed parameter, must instantiate `P` to a fresh variable at
// each call site and unify it against the concrete callback argument.
// Before the fix, the Surf desugarer misclassified the uppercase `P` as
// a rigid ADT `(t-adt {} P)`, so every call site failed with
// `type mismatch: P vs tensor[..]`.

#[test]
fn issue_293_general_tvar_through_arrow_param_checks_clean() {
    // Positive: the reproducer must check clean — no TypeMismatch.
    let result = infer_surf(
        r#"
module Repro.GenericCallback
def apply_resid[n, P](x: tensor[n, f32], inner_p: P, f: tensor[n, f32] -> P -> tensor[n, f32]) -> tensor[n, f32] =
  add(x, f(x, inner_p))
def use_it(x: tensor[3, f32], w: tensor[3, f32]) -> tensor[3, f32] =
  apply_resid(x, w, fn (t, q) -> mul(t, q))
"#,
    );
    assert!(
        result.errors.is_empty(),
        "expected the general-tvar-through-arrow reproducer to check clean, got: {:?}",
        result.errors
    );
}

#[test]
fn issue_293_apply_resid_checks_clean_in_isolation() {
    // Control: the generic def alone already checked clean before the
    // fix; it must keep checking clean.
    let result = infer_surf(
        r#"
module Repro.GenericCallback
def apply_resid[n, P](x: tensor[n, f32], inner_p: P, f: tensor[n, f32] -> P -> tensor[n, f32]) -> tensor[n, f32] =
  add(x, f(x, inner_p))
"#,
    );
    assert!(
        result.errors.is_empty(),
        "apply_resid must check clean in isolation, got: {:?}",
        result.errors
    );
}

#[test]
fn issue_293_dim_var_callback_variant_checks_clean() {
    // Control (dim-var path): the same shape where the threaded
    // parameter is a *dim* var `m` rather than a general type var
    // already worked and must keep working.
    let result = infer_surf(
        r#"
module Repro.DimCallback
def apply_resid[n, m](x: tensor[n, f32], inner: tensor[m, f32], f: tensor[n, f32] -> tensor[m, f32] -> tensor[n, f32]) -> tensor[n, f32] =
  f(x, inner)
def use_it(x: tensor[3, f32], w: tensor[3, f32]) -> tensor[3, f32] =
  apply_resid(x, w, fn (t, q) -> add(t, q))
"#,
    );
    assert!(
        result.errors.is_empty(),
        "the dim-var callback control must check clean, got: {:?}",
        result.errors
    );
}

#[test]
fn issue_293_monomorphic_callback_variant_checks_clean() {
    // Control (monomorphic path): a fully concrete callback already
    // worked and must keep working.
    let result = infer_surf(
        r#"
module Repro.MonoCallback
def apply_resid[n](x: tensor[n, f32], inner: tensor[n, f32], f: tensor[n, f32] -> tensor[n, f32] -> tensor[n, f32]) -> tensor[n, f32] =
  add(x, f(x, inner))
def use_it(x: tensor[3, f32], w: tensor[3, f32]) -> tensor[3, f32] =
  apply_resid(x, w, fn (t, q) -> mul(t, q))
"#,
    );
    assert!(
        result.errors.is_empty(),
        "the monomorphic callback control must check clean, got: {:?}",
        result.errors
    );
}

#[test]
fn issue_293_incompatible_callback_still_rejected() {
    // Negative parity: the fix must NOT over-loosen unification. Here
    // the callback's second parameter `q` is multiplied with `t`
    // (a `tensor[n, f32]`), so `q` must be `tensor[n, f32]`. But the
    // `inner_p` argument supplied at the call site is a scalar `i32`,
    // which is bound to the same `P`. `P` cannot be both a tensor and a
    // scalar i32, so this must still produce a clear mismatch.
    let result = infer_surf(
        r#"
module Repro.BadCallback
def apply_resid[n, P](x: tensor[n, f32], inner_p: P, f: tensor[n, f32] -> P -> tensor[n, f32]) -> tensor[n, f32] =
  add(x, f(x, inner_p))
def use_it(x: tensor[3, f32]) -> tensor[3, f32] =
  apply_resid(x, 1, fn (t, q) -> mul(t, q))
"#,
    );
    assert!(
        result.errors.iter().any(|e| matches!(
            e.kind,
            CheckErrorKind::TypeMismatch
                | CheckErrorKind::PrecisionMismatch
                | CheckErrorKind::DimensionMismatch
        )),
        "an incompatible callback/argument combination must still be rejected with a \
         unification mismatch; unification must not have been over-loosened (chelis#293), \
         got: {:?}",
        result.errors
    );
}

// ── #39 / chelis#405 wildcard-narrowing param-bound-dvar gate ────

/// `narrow_wildcards_with` narrows a body `Wildcard` to a declared
/// `Dim::Var` ONLY when that var is bound by a parameter tensor
/// position. This is the `const_col[n](spots: tensor[n, ..]) ->
/// tensor[n, 1]` family: the return dim `n` is the same var as the
/// `spots` parameter axis-0, so the caller binds it. The narrow
/// reconnects the wildcard return to that input dim.
#[test]
fn narrow_substitutes_param_bound_dim_var_for_wildcard() {
    let n = DimVar(7);
    let decl = Type::Fn(
        vec![
            Type::Tensor(
                vec![Dim::Var(n), Dim::Lit(1)],
                TensorPrec::Concrete(Prim::F64),
            ),
            Type::Prim(Prim::F64),
        ],
        Box::new(Type::Tensor(
            vec![Dim::Var(n), Dim::Lit(1)],
            TensorPrec::Concrete(Prim::F64),
        )),
    );
    // The inferred body return is the shape-erased `tensor[*, 1]`.
    let body = Type::Fn(
        vec![
            Type::Tensor(
                vec![Dim::Var(n), Dim::Lit(1)],
                TensorPrec::Concrete(Prim::F64),
            ),
            Type::Prim(Prim::F64),
        ],
        Box::new(Type::Tensor(
            vec![Dim::Wildcard, Dim::Lit(1)],
            TensorPrec::Concrete(Prim::F64),
        )),
    );
    let param_dvars = param_bound_dims(&decl);
    assert!(
        param_dvars.contains(&Dim::Var(n)),
        "n appears in a parameter tensor position, so it is param-bound"
    );
    let narrowed = narrow_wildcards_with(&body, &decl, &param_dvars);
    let Type::Fn(_, ret) = &narrowed else {
        panic!("expected Fn, got {narrowed:?}");
    };
    assert_eq!(
        **ret,
        Type::Tensor(
            vec![Dim::Var(n), Dim::Lit(1)],
            TensorPrec::Concrete(Prim::F64)
        ),
        "the return wildcard must narrow to the param-bound dim var n, not stay `*`"
    );
}

#[test]
fn declared_named_dimension_requires_its_own_signature_parameter() {
    for has_parameter in [true, false] {
        let parameter_dim = Dim::Name(if has_parameter { "rows" } else { "cols" }.into());
        let declared = Type::Fn(
            vec![Type::Tensor(
                vec![parameter_dim],
                TensorPrec::Concrete(Prim::F32),
            )],
            Box::new(Type::Tensor(
                vec![Dim::Name("rows".into()), Dim::Lit(2)],
                TensorPrec::Concrete(Prim::F32),
            )),
        );
        let Type::Fn(params, _) = &declared else {
            unreachable!()
        };
        let body = Type::Fn(
            params.clone(),
            Box::new(Type::Tensor(
                vec![Dim::Wildcard, Dim::Lit(2)],
                TensorPrec::Concrete(Prim::F32),
            )),
        );
        let narrowed = narrow_wildcards_with(&body, &declared, &param_bound_dims(&declared));
        assert_eq!(narrowed, if has_parameter { declared } else { body });
    }
}

/// A *return-only* dim var (it appears in the declared return but in
/// NO parameter tensor position, e.g. `arange[n](start: i32, stop:
/// i32) -> tensor[n, i32]`) is NOT param-bound. The body wildcard
/// must stay `Wildcard`: narrowing it to the unbound var would leak a
/// free dim var into callers (the RT-39+44 soundness regression,
/// commit 8067c9ce).
#[test]
fn narrow_keeps_wildcard_for_return_only_dim_var() {
    let n = DimVar(11);
    let decl = Type::Fn(
        vec![Type::Prim(Prim::Int32), Type::Prim(Prim::Int32)],
        Box::new(Type::Tensor(
            vec![Dim::Var(n)],
            TensorPrec::Concrete(Prim::Int32),
        )),
    );
    let body = Type::Fn(
        vec![Type::Prim(Prim::Int32), Type::Prim(Prim::Int32)],
        Box::new(Type::Tensor(
            vec![Dim::Wildcard],
            TensorPrec::Concrete(Prim::Int32),
        )),
    );
    let param_dvars = param_bound_dims(&decl);
    assert!(
        !param_dvars.contains(&Dim::Var(n)),
        "n is return-only: it must not be in the param-bound set"
    );
    let narrowed = narrow_wildcards_with(&body, &decl, &param_dvars);
    let Type::Fn(_, ret) = &narrowed else {
        panic!("expected Fn, got {narrowed:?}");
    };
    assert_eq!(
        **ret,
        Type::Tensor(vec![Dim::Wildcard], TensorPrec::Concrete(Prim::Int32)),
        "a return-only dim var's body wildcard must stay `*` (no free-var leak)"
    );
}

/// `Wildcard` against a declared `Dim::Lit` is always narrowed
/// regardless of the param-bound set -- concrete literals are
/// self-contained (the original #39 behavior).
#[test]
fn narrow_substitutes_literal_for_wildcard_unconditionally() {
    let empty = Vec::new();
    let body = Type::Tensor(
        vec![Dim::Wildcard, Dim::Wildcard],
        TensorPrec::Concrete(Prim::F32),
    );
    let decl = Type::Tensor(
        vec![Dim::Lit(4), Dim::Lit(1)],
        TensorPrec::Concrete(Prim::F32),
    );
    let narrowed = narrow_wildcards_with(&body, &decl, &empty);
    assert_eq!(
        narrowed,
        Type::Tensor(
            vec![Dim::Lit(4), Dim::Lit(1)],
            TensorPrec::Concrete(Prim::F32)
        )
    );
}

/// End-to-end regression lock (chelis#405 / WS-3): the
/// `const_col`-style `shape -> to_tensor(map(range)) -> reshape` chain
/// must publish a scheme whose return batch dim is the param-bound
/// DECLARED dim, not the shape-erased `*`. Pre-fix the published scheme
/// was `tensor[*, 1]`, so a CALLER that forwards the result observed
/// `*` for its batch axis; that erased the link the C backend needs and
/// ICEd the §345 `symbolic_occurrences` guard at the `vmap` lane
/// kernel. The fix ties the return dim to the `spots` parameter dim
/// var, so a caller forwarding `const_col`'s result sees a declared
/// dim. (`type_env` records each def's body annotation; the call-site
/// result type in the *caller's* body annotation is what reflects the
/// narrowed published scheme, so the lock inspects the caller.)
#[test]
fn const_col_chain_propagates_declared_dim_to_callers() {
    let checked = checked_surf(
        r#"
def const_col[n](spots: tensor[n, f32], v: f64) -> tensor[n, 1, f64] = {
  nn = cast(shape(copy(spots), cast(0, i32)), i64)
  reshape(to_tensor(map(fn (i: i64) -> v, range(cast(0, i64), nn))), [nn, cast(1, i64)])
}
def caller[n](spots: tensor[n, f32]) -> tensor[n, 1, f64] = const_col(spots, cast(1.0, f64))
"#,
    );
    let caller_ty = checked
        .type_env()
        .get("caller")
        .expect("caller must be in the published type env");
    let rendered = chelis_deep::printer::print_canonical_flat(std::slice::from_ref(caller_ty));
    // The caller's body is `const_col(spots, ..)`; its annotated type
    // is the call-site result. A param-bound declared dim narrows the
    // wildcard, so the caller's batch axis is a `d-var`, never the
    // shape-erased `(d-name {} *)`.
    assert!(
        !rendered.contains("(d-name {} *)"),
        "const_col's result must not carry a wildcard `*` dim at the call \
         site; got {rendered}"
    );
    assert!(
        rendered.contains("d-var"),
        "the const_col call-site batch dim must be a declared dim var; \
         got {rendered}"
    );
}
