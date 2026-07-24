//! Explicit declaration-local binder scope must not leak between siblings,
//! sequential checks, worker threads, stacked contexts, or serialized caches.

use chelis_types::{TypeEnv, build_type_env_from_library, check_ir_program, check_ir_with_context};

const LEGAL: &str = r#"
(defsig {} legal
  (t-fn {}
    (t-tensor {} (d-var {} n) (t-prim {} f32))
    (t-tensor {} (d-var {} n) (t-prim {} f32))))
(def {} legal
  (fn {} (params {} x)
    (let {} (bind {} y
      (var {type: (t-tensor {} (d-var {} n) (t-prim {} f32))} x))
      (var {} y))))
"#;

const ROGUE: &str = r#"
(def {} rogue
  (lit {type: (t-tensor {} (d-var {} n) (t-prim {} f32))} 1))
"#;

fn parse(source: &str) -> Vec<chelis_deep::Expr> {
    chelis_deep::parser::parse_str(source).expect("Deep binder fixture must parse")
}

fn surf(source: &str) -> Vec<chelis_deep::Expr> {
    let parsed = chelis_surf::parser::parse_str(source).expect("Surf binder fixture must parse");
    chelis_surf::desugar::desugar_program(&parsed)
}

fn assert_legal() {
    let exprs = parse(LEGAL);
    check_ir_program(&exprs)
        .unwrap_or_else(|result| panic!("legal explicit binder must check: {:?}", result.errors));
}

fn assert_rogue_rejected(result: Result<chelis_types::CheckedProgram, chelis_types::InferResult>) {
    let errors = result
        .expect_err("a sibling without a signature must not inherit `n`")
        .errors;
    assert_eq!(
        errors.len(),
        1,
        "undeclared binder must report once: {errors:?}"
    );
    assert!(
        errors[0]
            .message
            .contains("undeclared dimension variable `n`"),
        "unexpected binder failure: {errors:?}"
    );
}

#[test]
fn sibling_definition_does_not_inherit_binders() {
    let exprs = parse(&format!("{LEGAL}\n{ROGUE}"));
    assert_rogue_rejected(check_ir_program(&exprs));
}

#[test]
fn sequential_checks_do_not_inherit_binders() {
    assert_legal();
    assert_rogue_rejected(check_ir_program(&parse(ROGUE)));
    assert_legal();
}

#[test]
fn parallel_checks_keep_binders_thread_local_by_construction() {
    let legal = std::thread::spawn(|| {
        for _ in 0..8 {
            assert_legal();
        }
    });
    let rogue = std::thread::spawn(|| {
        for _ in 0..8 {
            assert_rogue_rejected(check_ir_program(&parse(ROGUE)));
        }
    });
    legal.join().expect("legal checker worker panicked");
    rogue.join().expect("rogue checker worker panicked");
}

#[test]
fn stacked_context_does_not_export_a_body_binder() {
    let context = build_type_env_from_library(&parse(LEGAL)).expect("legal library must check");
    assert_rogue_rejected(check_ir_with_context(&context, &parse(ROGUE)));
}

#[test]
fn serialized_context_drops_transient_binder_scope() {
    let context = build_type_env_from_library(&parse(LEGAL)).expect("legal library must check");
    let bytes = bincode::serialize(&context).expect("TypeEnv cache serialization must succeed");
    let restored: TypeEnv =
        bincode::deserialize(&bytes).expect("TypeEnv cache deserialization must succeed");
    assert_rogue_rejected(check_ir_with_context(&restored, &parse(ROGUE)));
}

#[test]
fn malformed_parameter_is_rejected_once_by_the_binder_owner() {
    let exprs =
        parse("(def {} bad (fn {} (params {} (x {type: (t-prim {} f32 extra)})) (var {} x)))");
    let errors = check_ir_program(&exprs)
        .expect_err("malformed binder syntax must fail")
        .errors;
    assert_eq!(
        errors.len(),
        1,
        "malformed binder must report once: {errors:?}"
    );
    assert!(
        matches!(
            errors[0].kind,
            chelis_types::errors::CheckErrorKind::MalformedForm
        ) && errors[0].message.contains("t-prim"),
        "unexpected malformed-binder diagnostic: {errors:?}"
    );
}

#[test]
fn explicit_signature_binders_reach_generated_annotation_finalization() {
    let exprs = surf(
        r#"
sig row_argmax: &tensor[piece, classes, p] -> int64
def row_argmax[piece, classes, p](row: &tensor[piece, classes, p]) -> int64 = {
  pair = sort(row, cast(1, int32))
  cast(0, int64)
}
def call(xs: &tensor[1, 3, f32]) -> int64 = row_argmax(xs)
"#,
    );
    check_ir_program(&exprs).unwrap_or_else(|result| {
        panic!(
            "the declaration-owned `piece`/`classes`/`p` scope must survive finalization: {:?}",
            result.errors
        )
    });
}

#[test]
fn unknown_nominal_metadata_reports_once_without_finalizer_duplication() {
    let exprs = parse(
        "(def {} bad
           (fn {} (params {} (x {type: (t-adt {} Missing)})) (var {} x)))",
    );
    let errors = check_ir_program(&exprs)
        .expect_err("unknown nominal metadata must reject")
        .errors;
    assert_eq!(
        errors.len(),
        1,
        "the inference owner reports once and finalization must not retry it: {errors:?}"
    );
    assert!(errors[0].message.contains("Missing"), "{errors:?}");
}
