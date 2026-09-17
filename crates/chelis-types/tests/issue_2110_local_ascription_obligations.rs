use chelis_deep::printer::print_expr;
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str;
use chelis_types::errors::CheckErrorKind;
use chelis_types::types::{Dim, Type};
use chelis_types::{
    CheckedProgram, LocalTensorAscriptionOrigin, build_compiled_library_context, check_ir_program,
    check_ir_with_context, check_linearity,
};

fn check_surf(source: &str) -> Result<CheckedProgram, chelis_types::InferResult> {
    let decls = parse_str(source).expect("Surf parse");
    check_ir_program(&desugar_program(&decls))
}

fn checked_surf(source: &str) -> CheckedProgram {
    check_surf(source).unwrap_or_else(|result| panic!("type check failed: {:?}", result.errors))
}

fn check_deep(source: &str) -> Result<CheckedProgram, chelis_types::InferResult> {
    let program = chelis_deep::parser::parse_str(source).expect("Deep parse");
    check_ir_program(&program)
}

fn source_at(source: &str, span: chelis_deep::Span) -> &str {
    &source[span.offset..span.end()]
}

#[test]
fn runtime_dependent_local_ascription_retains_exact_authored_claim() {
    let source = r#"
def f(x: tensor[*, f32]) -> tensor[*, f32] = {
  y: tensor[9, f32] =
    pad(x, [[0i64, 0i64]], 0.0f32)
  y
}
"#;
    let checked = checked_surf(source);
    let [ascription] = checked.local_tensor_ascriptions() else {
        panic!(
            "expected one local tensor ascription, got {:?}",
            checked.local_tensor_ascriptions()
        );
    };

    assert_eq!(ascription.id().get(), 0);
    assert_eq!(
        ascription.origin(),
        LocalTensorAscriptionOrigin::SurfExplicit
    );
    assert_eq!(ascription.declaration_name(), Some("f"));
    assert_eq!(ascription.binding_name(), "y");
    assert_eq!(source_at(source, ascription.binding_span()), "y");
    assert_eq!(
        source_at(source, ascription.ascription_span()),
        "tensor[9, f32]"
    );
    let initializer_source = source_at(source, ascription.initializer_span());
    assert!(
        initializer_source.starts_with("pad("),
        "initializer span must name the producing expression, got {initializer_source:?}"
    );
    assert_eq!(
        ascription.declared_type(),
        &Type::Tensor(
            vec![Dim::Lit(9)],
            chelis_types::types::TensorPrec::Concrete(chelis_types::types::Prim::F32)
        )
    );
    assert!(
        print_expr(ascription.authored_type()).contains("(d-lit {} 9)"),
        "the exact authored Deep tensor type must survive"
    );
    let [claim] = ascription.outstanding_claims() else {
        panic!(
            "runtime-dependent RHS must retain one outstanding axis claim: {:?}",
            ascription.outstanding_claims()
        );
    };
    assert_eq!(claim.axis(), 0);
    assert_eq!(claim.required_extent(), &Dim::Lit(9));
}

const DEEP_RUNTIME_LOCAL_ASCRIPTION: &str = r#"
(defsig {}
  f
  (t-fn {}
    (t-tensor {} (d-name {} *) (t-prim {} f32))
    (t-tensor {} (d-name {} *) (t-prim {} f32))))
(def {}
  f
  (fn {}
    (params {} (x {type: (t-var {} _)}))
    (let {}
      (bind {}
        y
        (app {type: (t-tensor {} (d-lit {} 2) (t-prim {} f32))}
          (var {} pad)
          (var {} x)
          (app {}
            (var {} Cons)
            (app {}
              (var {} Cons)
              (lit {type: (t-prim {} i64)} 0)
              (app {}
                (var {} Cons)
                (lit {type: (t-prim {} i64)} 0)
                (var {} Nil)))
            (var {} Nil))
          (lit {type: (t-prim {} f32)} 0.0)))
      (var {} y))))
"#;

#[test]
fn hand_authored_deep_type_metadata_is_an_explicit_local_ascription() {
    let checked = check_deep(DEEP_RUNTIME_LOCAL_ASCRIPTION)
        .unwrap_or_else(|result| panic!("Deep type check failed: {:?}", result.errors));
    let [ascription] = checked.local_tensor_ascriptions() else {
        panic!(
            "hand-authored Deep metadata must create one obligation: {:?}",
            checked.local_tensor_ascriptions()
        );
    };
    assert_eq!(
        ascription.origin(),
        LocalTensorAscriptionOrigin::DeepTypeMetadata
    );
    assert_eq!(ascription.binding_name(), "y");
    assert_eq!(
        ascription.declared_type(),
        &Type::Tensor(
            vec![Dim::Lit(2)],
            chelis_types::types::TensorPrec::Concrete(chelis_types::types::Prim::F32)
        )
    );
    assert_eq!(ascription.outstanding_claims().len(), 1);
    assert_eq!(
        ascription.outstanding_claims()[0].required_extent(),
        &Dim::Lit(2)
    );
}

#[test]
fn hand_authored_deep_static_mismatch_is_dimension_mismatch() {
    let source = r#"
(def {}
  out
  (let {}
    (bind {}
      y
      (app {type: (t-tensor {} (d-lit {} 9) (t-prim {} f32))}
        (var {} pad)
        (app {}
          (var {} to_tensor)
          (app {}
            (var {} Cons)
            (lit {type: (t-prim {} f32)} 1.0)
            (var {} Nil)))
        (app {}
          (var {} Cons)
          (app {}
            (var {} Cons)
            (lit {type: (t-prim {} i64)} 1)
            (app {}
              (var {} Cons)
              (lit {type: (t-prim {} i64)} 0)
              (var {} Nil)))
          (var {} Nil))
        (lit {type: (t-prim {} f32)} 0.0)))
    (var {} y)))
"#;
    let result = check_deep(source).expect_err("static Deep mismatch must reject");
    assert!(
        result
            .errors
            .iter()
            .any(|error| matches!(error.kind, CheckErrorKind::DimensionMismatch)),
        "expected DimensionMismatch, got {:?}",
        result.errors
    );
}

#[test]
fn claimed_axes_remain_in_authored_axis_order() {
    let checked = checked_surf(
        r#"
def f(x: tensor[*, *, f32]) -> tensor[*, *, f32] = {
  y: tensor[7, 11, f32] =
    pad(x, [[0i64, 0i64], [0i64, 0i64]], 0.0f32)
  y
}
"#,
    );
    let [ascription] = checked.local_tensor_ascriptions() else {
        panic!("expected one local tensor ascription");
    };
    let claims = ascription.outstanding_claims();
    assert_eq!(claims.len(), 2);
    assert_eq!(claims[0].axis(), 0);
    assert_eq!(claims[0].required_extent(), &Dim::Lit(7));
    assert_eq!(claims[1].axis(), 1);
    assert_eq!(claims[1].required_extent(), &Dim::Lit(11));
}

#[test]
fn inferred_metadata_does_not_become_authored_ascription_provenance() {
    let checked = checked_surf(
        r#"
def f() -> f32 = {
  inferred = 1.0f32
  inferred
}
"#,
    );
    assert!(
        checked.local_tensor_ascriptions().is_empty(),
        "ordinary inferred `type` metadata is not an authored local tensor ascription"
    );
}

#[test]
fn authored_wildcard_is_preserved_without_manufacturing_a_claim() {
    let checked = checked_surf(
        r#"
def f(x: tensor[*, f32]) -> tensor[*, f32] = {
  y: tensor[*, f32] =
    pad(x, [[0i64, 0i64]], 0.0f32)
  y
}
"#,
    );
    let [ascription] = checked.local_tensor_ascriptions() else {
        panic!("expected one local tensor ascription");
    };
    assert_eq!(
        ascription.declared_type(),
        &Type::Tensor(
            vec![Dim::Wildcard],
            chelis_types::types::TensorPrec::Concrete(chelis_types::types::Prim::F32)
        )
    );
    assert!(ascription.outstanding_claims().is_empty());
    assert!(
        print_expr(ascription.authored_type()).contains("(d-name {} *)"),
        "the wildcard authored axis must survive even though it creates no claim"
    );
}

#[test]
fn authored_ascriptions_inside_match_arms_reach_the_checked_program() {
    let checked = checked_surf(
        r#"
type Choice = | First | Second

def f(choice: Choice, x: tensor[*, f32]) -> tensor[*, f32] =
  match choice with {
    | First => x
    | Second => {
        y: tensor[2, f32] = pad(x, [[0i64, 0i64]], 0.0f32)
        y
      }
  }
"#,
    );
    let [ascription] = checked.local_tensor_ascriptions() else {
        panic!(
            "the authored match-arm binding must create one checked obligation: {:?}",
            checked.local_tensor_ascriptions()
        );
    };
    assert_eq!(ascription.binding_name(), "y");
    assert_eq!(ascription.declaration_name(), Some("f"));
    assert_eq!(ascription.outstanding_claims().len(), 1);
    assert_eq!(
        ascription.outstanding_claims()[0].required_extent(),
        &Dim::Lit(2)
    );
}

#[test]
fn static_agreement_discharges_and_static_disagreement_stays_dimension_mismatch() {
    let agreeing = checked_surf(
        r#"
def f() -> tensor[*, f32] = {
  y: tensor[2, f32] =
    pad(to_tensor([1.0f32, 2.0f32]), [[0i64, 0i64]], 0.0f32)
  y
}
"#,
    );
    let [ascription] = agreeing.local_tensor_ascriptions() else {
        panic!("expected agreeing ascription provenance");
    };
    assert!(
        ascription.outstanding_claims().is_empty(),
        "literal agreement is checker-proved and needs no runtime claim"
    );

    let disagreement = check_surf(
        r#"
def f() -> tensor[*, f32] = {
  y: tensor[9, f32] =
    pad(to_tensor([1.0f32]), [[1i64, 0i64]], 0.0f32)
  y
}
"#,
    )
    .expect_err("literal disagreement must be rejected before a CheckedProgram exists");
    assert!(
        disagreement
            .errors
            .iter()
            .any(|error| matches!(error.kind, CheckErrorKind::DimensionMismatch)),
        "expected DimensionMismatch, got {:?}",
        disagreement.errors
    );
}

#[test]
fn checker_owned_ascription_survives_clone_serialization_and_checker_rewrites() {
    let checked = checked_surf(
        r#"
def f(x: tensor[*, f32]) -> tensor[*, f32] = {
  y: tensor[9, f32] =
    pad(x, [[0i64, 0i64]], 0.0f32)
  y
}
"#,
    );
    assert_eq!(
        checked.clone().local_tensor_ascriptions(),
        checked.local_tensor_ascriptions()
    );

    let bytes = bincode::serialize(&checked).expect("CheckedProgram serialization");
    let decoded: CheckedProgram =
        bincode::deserialize(&bytes).expect("CheckedProgram deserialization");
    assert_eq!(
        decoded.local_tensor_ascriptions(),
        checked.local_tensor_ascriptions()
    );

    let effects = checked
        .try_with_effect_annotations(checked.annotated_exprs().to_vec())
        .expect("effects-only rewrite");
    assert_eq!(
        effects.local_tensor_ascriptions(),
        checked.local_tensor_ascriptions()
    );
    let linearity = check_linearity(&effects).expect("linearity check");
    assert_eq!(
        linearity.local_tensor_ascriptions(),
        checked.local_tensor_ascriptions()
    );
}

#[test]
fn checked_library_composition_preserves_both_identity_domains() {
    let library_decls = parse_str(
        r#"
def library_f(x: tensor[*, f32]) -> tensor[*, f32] = {
  library_y: tensor[7, f32] =
    pad(x, [[0i64, 0i64]], 0.0f32)
  library_y
}
"#,
    )
    .expect("library Surf parse");
    let (context, library_checked) =
        build_compiled_library_context(&desugar_program(&library_decls))
            .expect("library type check");

    let application_decls = parse_str(
        r#"
def application_f(x: tensor[*, f32]) -> tensor[*, f32] = {
  application_y: tensor[11, f32] =
    pad(x, [[0i64, 0i64]], 0.0f32)
  application_y
}
"#,
    )
    .expect("application Surf parse");
    let application_checked = check_ir_with_context(&context, &desugar_program(&application_decls))
        .expect("application type check");
    let composed = CheckedProgram::compose(&library_checked, &application_checked)
        .expect("proof-bound checked programs compose");

    let ascriptions = composed.local_tensor_ascriptions();
    assert_eq!(ascriptions.len(), 2);
    assert_eq!(ascriptions[0].declaration_name(), Some("library_f"));
    assert_eq!(ascriptions[1].declaration_name(), Some("application_f"));
    assert_eq!(ascriptions[0].binding_name(), "library_y");
    assert_eq!(ascriptions[1].binding_name(), "application_y");
    assert_ne!(
        ascriptions[0].id(),
        ascriptions[1].id(),
        "composition must explicitly separate independently checked ID domains"
    );
}
