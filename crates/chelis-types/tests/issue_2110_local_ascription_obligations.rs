use chelis_deep::printer::print_expr;
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str;
use chelis_types::errors::CheckErrorKind;
use chelis_types::types::{Dim, Type};
use chelis_types::{
    CheckedProgram, LocalAscriptionOrigin, build_compiled_library_context, check_ir_program,
    check_ir_with_context, check_linearity,
};

fn check_surf(source: &str) -> Result<CheckedProgram, chelis_types::InferResult> {
    let decls = parse_str(source, &chelis_syntax_host::NativeStack).expect("Surf parse");
    check_ir_program(
        &desugar_program(&decls, &chelis_syntax_host::NativeStack)
            .expect("Surf fixture must desugar"),
    )
}

fn checked_surf(source: &str) -> CheckedProgram {
    check_surf(source).unwrap_or_else(|result| panic!("type check failed: {:?}", result.errors))
}

fn check_deep(source: &str) -> Result<CheckedProgram, chelis_types::InferResult> {
    let program = chelis_deep::parser::parse_str(source, &chelis_syntax_host::NativeStack)
        .expect("Deep parse");
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
    let [ascription] = checked.local_ascriptions() else {
        panic!(
            "expected one local tensor ascription, got {:?}",
            checked.local_ascriptions()
        );
    };

    assert_eq!(ascription.id().get(), 0);
    assert_eq!(ascription.origin(), LocalAscriptionOrigin::SurfExplicit);
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
            chelis_types::types::TensorPrec::Concrete(chelis_numeric_core::prim::Prim::F32)
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
    let [ascription] = checked.local_ascriptions() else {
        panic!(
            "hand-authored Deep metadata must create one obligation: {:?}",
            checked.local_ascriptions()
        );
    };
    assert_eq!(ascription.origin(), LocalAscriptionOrigin::DeepTypeMetadata);
    assert_eq!(ascription.binding_name(), "y");
    assert_eq!(
        ascription.declared_type(),
        &Type::Tensor(
            vec![Dim::Lit(2)],
            chelis_types::types::TensorPrec::Concrete(chelis_numeric_core::prim::Prim::F32)
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
    let [ascription] = checked.local_ascriptions() else {
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
        checked.local_ascriptions().is_empty(),
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
    let [ascription] = checked.local_ascriptions() else {
        panic!("expected one local tensor ascription");
    };
    assert_eq!(
        ascription.declared_type(),
        &Type::Tensor(
            vec![Dim::Wildcard],
            chelis_types::types::TensorPrec::Concrete(chelis_numeric_core::prim::Prim::F32)
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
    let [ascription] = checked.local_ascriptions() else {
        panic!(
            "the authored match-arm binding must create one checked obligation: {:?}",
            checked.local_ascriptions()
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
    let [ascription] = agreeing.local_ascriptions() else {
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
        checked.clone().local_ascriptions(),
        checked.local_ascriptions()
    );

    let bytes = bincode::serialize(&checked).expect("CheckedProgram serialization");
    let decoded: CheckedProgram =
        bincode::deserialize(&bytes).expect("CheckedProgram deserialization");
    assert_eq!(decoded.local_ascriptions(), checked.local_ascriptions());

    let effects = checked
        .try_with_effect_annotations(checked.annotated_exprs().to_vec())
        .expect("effects-only rewrite");
    assert_eq!(effects.local_ascriptions(), checked.local_ascriptions());
    let linearity = check_linearity(&effects).expect("linearity check");
    assert_eq!(linearity.local_ascriptions(), checked.local_ascriptions());
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
        &chelis_syntax_host::NativeStack,
    )
    .expect("library Surf parse");
    let (context, library_checked) = build_compiled_library_context(
        &desugar_program(&library_decls, &chelis_syntax_host::NativeStack)
            .expect("Surf fixture must desugar"),
    )
    .expect("library type check");

    let application_decls = parse_str(
        r#"
    def application_f(x: tensor[*, f32]) -> tensor[*, f32] = {
      application_y: tensor[11, f32] =
        pad(x, [[0i64, 0i64]], 0.0f32)
      application_y
    }
    "#,
        &chelis_syntax_host::NativeStack,
    )
    .expect("application Surf parse");
    let application_checked = check_ir_with_context(
        &context,
        &desugar_program(&application_decls, &chelis_syntax_host::NativeStack)
            .expect("Surf fixture must desugar"),
    )
    .expect("application type check");
    let composed = CheckedProgram::compose(&library_checked, &application_checked)
        .expect("proof-bound checked programs compose");

    let ascriptions = composed.local_ascriptions();
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

// [03-META-1], [04-INF-9], and spec/04 §4.7: callable bindings retain
// their complete authored claims in the same checker-owned source ledger.
fn callable_owner_source(name: &str, operand: &str) -> String {
    format!(
        "def {name}(flag: bool, x: tensor[*, f64]) -> tensor[*, bool] = {{\n f = if flag then is_nan else is_finite\n g: &tensor[2, f64] -> tensor[*, bool] = {operand}\n g(x)\n}}\n"
    )
}

fn callable_owner_origin(
    expr: &chelis_deep::Expr,
    origin: Option<chelis_deep::annotations::BindingTypeOrigin>,
    stamp_operand_abi: bool,
) -> chelis_deep::Expr {
    use chelis_deep::annotations::{MetadataKey, MetadataValue, Spanned};
    use chelis_deep::{DeepTag, Expr};
    let Expr::Node(node, span) = expr else {
        return expr.clone();
    };
    let mut children: Vec<_> = node
        .children_slice()
        .iter()
        .map(|child| callable_owner_origin(child, origin, stamp_operand_abi))
        .collect();
    if node.tag() == DeepTag::Bind {
        for pair in children.as_chunks_mut::<2>().0 {
            if matches!(&pair[0], Expr::Atom(chelis_deep::Atom::Name(name), _) if name == "g") {
                let Expr::Node(rhs, rhs_span) = &pair[1] else {
                    panic!("fixture RHS")
                };
                let mut metadata = rhs.meta().clone();
                metadata.remove(MetadataKey::SurfBindingType);
                if let Some(origin) = origin {
                    metadata.replace(MetadataValue::SurfBindingType(Spanned::new(
                        origin, *rhs_span,
                    )));
                }
                let mut rhs_children = rhs.children_slice().to_vec();
                if stamp_operand_abi {
                    let Expr::Node(operand, operand_span) = &rhs_children[0] else {
                        panic!("fixture operand")
                    };
                    let mut operand_meta = operand.meta().clone();
                    operand_meta.replace(MetadataValue::Type(
                        metadata.ty().expect("source type").clone(),
                    ));
                    operand_meta.remove(MetadataKey::SurfBindingType);
                    rhs_children[0] = Expr::node(
                        operand.tag(),
                        operand_meta,
                        operand.children_slice().to_vec(),
                        *operand_span,
                    );
                }
                pair[1] = Expr::node(rhs.tag(), metadata, rhs_children, *rhs_span);
            }
        }
    }
    Expr::node(node.tag(), node.meta().clone(), children, *span)
}

#[test]
fn callable_owner_distinguishes_deep_surf_and_inferred_binding_metadata() {
    use chelis_deep::annotations::BindingTypeOrigin;
    for origin in [
        None,
        Some(BindingTypeOrigin::Explicit),
        Some(BindingTypeOrigin::Inferred),
    ] {
        for operand in ["f", "copy(copy(f))"] {
            let source = callable_owner_source("run", operand);
            let source_deep = desugar_program(
                &parse_str(&source, &chelis_syntax_host::NativeStack).expect("Surf parse"),
                &chelis_syntax_host::NativeStack,
            )
            .expect("desugar");
            let deep: Vec<_> = source_deep
                .iter()
                .map(|expr| callable_owner_origin(expr, origin, false))
                .collect();
            let checked = check_ir_program(&deep).expect("checked callable source");
            let records = checked.local_ascriptions();
            if origin == Some(BindingTypeOrigin::Inferred) {
                assert!(
                    records.is_empty(),
                    "inferred ABI is not independent authorship: {records:?}"
                );
            } else {
                let [record] = records else {
                    panic!("one callable source owner, got {records:?}")
                };
                assert_eq!(record.binding_name(), "g");
                assert_eq!(record.declaration_name(), Some("run"));
                assert_eq!(
                    record.origin(),
                    if origin.is_none() {
                        LocalAscriptionOrigin::DeepTypeMetadata
                    } else {
                        LocalAscriptionOrigin::SurfExplicit
                    }
                );
                assert!(
                    matches!(record.declared_type(), Type::Fn(_, _)),
                    "complete Fn type: {record:?}"
                );
                assert!(print_expr(record.authored_type()).contains("(d-lit {} 2)"));
                assert!(
                    record.outstanding_claims().is_empty(),
                    "callable frame is not a tensor axis claim"
                );
            }
        }
    }
}

#[test]
fn callable_owner_does_not_promote_nonbinding_none_origin_abi_metadata() {
    use chelis_deep::annotations::BindingTypeOrigin;
    let source = callable_owner_source("run", "copy(f)");
    let deep = desugar_program(
        &parse_str(&source, &chelis_syntax_host::NativeStack).expect("Surf parse"),
        &chelis_syntax_host::NativeStack,
    )
    .expect("desugar");
    let deep: Vec<_> = deep
        .iter()
        .map(|expr| callable_owner_origin(expr, Some(BindingTypeOrigin::Inferred), true))
        .collect();
    let checked = check_ir_program(&deep).expect("checked expression ABI stamp");
    assert!(
        checked.local_ascriptions().is_empty(),
        "a None-origin stamp outside a binding owner cannot author a frame"
    );
}

#[test]
fn callable_owner_retains_result_only_and_nested_adapter_sources_once() {
    let checked = checked_surf(
        "def run(flag: bool, x: tensor[*, f64]) -> tensor[*, bool] = {\n f = if flag then is_nan else is_finite\n g: &tensor[*, f64] -> tensor[2, bool] = copy(f)\n h: &tensor[2, f64] -> tensor[*, bool] = copy(copy(g))\n h(x)\n}\n",
    );
    let records = checked.local_ascriptions();
    assert_eq!(
        records.len(),
        2,
        "one owner per independent adapter: {records:?}"
    );
    assert_eq!(
        records.iter().map(|r| r.binding_name()).collect::<Vec<_>>(),
        ["g", "h"]
    );
    assert_ne!(records[0].id(), records[1].id());
    assert!(
        matches!(records[0].declared_type(), Type::Fn(_, result) if matches!(result.as_ref(), Type::Tensor(dims, _) if dims == &[Dim::Lit(2)]))
    );
}

#[test]
fn callable_owner_recursive_inference_keeps_one_record_per_source_binding() {
    let source = "def run(flag: bool, x: tensor[*, f64]) -> tensor[*, bool] = {\n f = if flag then is_nan else is_finite\n g: &tensor[2, f64] -> tensor[*, bool] = copy(f)\n if flag then g(x) else run(true, x)\n}\n";
    for _ in 0..2 {
        let checked = checked_surf(source);
        let records = checked.local_ascriptions();
        assert_eq!(
            records.len(),
            1,
            "replay/close cannot append duplicate frames: {records:?}"
        );
        assert_eq!(records[0].binding_name(), "g");
        assert_eq!(records[0].id().get(), 0);
    }
}

#[test]
fn callable_owner_serialization_preserves_source_and_full_claims() {
    let checked = checked_surf(&callable_owner_source("run", "copy(copy(f))"));
    assert_eq!(
        checked.local_ascriptions().len(),
        1,
        "callable ledger must exist before serialization"
    );
    let encoded = bincode::serialize(&checked).expect("encode checked program");
    let decoded: CheckedProgram = bincode::deserialize(&encoded).expect("decode checked program");
    assert_eq!(decoded.local_ascriptions(), checked.local_ascriptions());
    assert_eq!(bincode::serialize(&decoded).unwrap(), encoded);
}

#[test]
fn callable_owner_proof_bound_composition_remaps_ids_and_replaces_declarations() {
    let library_deep = desugar_program(
        &parse_str(
            &callable_owner_source("helper", "copy(f)"),
            &chelis_syntax_host::NativeStack,
        )
        .unwrap(),
        &chelis_syntax_host::NativeStack,
    )
    .unwrap();
    let (context, library) = build_compiled_library_context(&library_deep).expect("library checks");
    assert_eq!(
        library.local_ascriptions().len(),
        1,
        "library callable owner"
    );
    let new_deep = desugar_program(
        &parse_str(
            &callable_owner_source("entry", "f"),
            &chelis_syntax_host::NativeStack,
        )
        .unwrap(),
        &chelis_syntax_host::NativeStack,
    )
    .unwrap();
    let entry = check_ir_with_context(&context, &new_deep).expect("entry checks");
    let composed = CheckedProgram::compose(&library, &entry).expect("proof-bound compose");
    let records = composed.local_ascriptions();
    assert_eq!(records.len(), 2);
    assert_eq!(
        records
            .iter()
            .map(|r| r.declaration_name())
            .collect::<Vec<_>>(),
        [Some("helper"), Some("entry")]
    );
    assert_ne!(
        records[0].id(),
        records[1].id(),
        "independent artifact ID domains"
    );
    fn owners(value: &serde_json::Value, output: &mut std::collections::BTreeSet<String>) {
        match value {
            serde_json::Value::Object(map) => {
                if let Some(serde_json::Value::String(token)) = map.get("checked_binding_owner") {
                    output.insert(token.clone());
                }
                for child in map.values() {
                    owners(child, output);
                }
            }
            serde_json::Value::Array(values) => {
                for child in values {
                    owners(child, output);
                }
            }
            _ => {}
        }
    }
    let mut tokens = std::collections::BTreeSet::new();
    owners(
        &serde_json::to_value(composed.annotated_exprs()).unwrap(),
        &mut tokens,
    );
    assert_eq!(
        tokens,
        records
            .iter()
            .map(|record| format!("local-ascription:{}", record.id().get()))
            .collect(),
        "AST tokens and records use the same composition offset"
    );

    let replacement_deep = desugar_program(
        &parse_str(
            &callable_owner_source("helper", "copy(copy(f))"),
            &chelis_syntax_host::NativeStack,
        )
        .unwrap(),
        &chelis_syntax_host::NativeStack,
    )
    .unwrap();
    let replacement =
        check_ir_with_context(&context, &replacement_deep).expect("replacement checks");
    let composed = CheckedProgram::compose(&library, &replacement).expect("replacement compose");
    let [record] = composed.local_ascriptions() else {
        panic!("shadowed source owner must be removed")
    };
    assert_eq!(record.declaration_name(), Some("helper"));
    assert_eq!(record.binding_name(), "g");
}

fn zero_callable_source_spans(expr: &chelis_deep::Expr) -> chelis_deep::Expr {
    use chelis_deep::{Expr, Span};
    let zero = Span::new(0, 0);
    fn metadata(meta: &chelis_deep::Metadata) -> chelis_deep::Metadata {
        let mut meta = meta.clone();
        meta.remove(chelis_deep::annotations::MetadataKey::Span);
        if let Some(ty) = meta.ty() {
            let syntax = zero_callable_source_spans(ty.expression());
            meta.replace(chelis_deep::annotations::MetadataValue::Type(
                chelis_deep::annotations::TypeSyntax::try_new(syntax).unwrap(),
            ));
        }
        if let Some(origin) = meta.surf_binding_type() {
            let origin = *origin.value();
            meta.replace(chelis_deep::annotations::MetadataValue::SurfBindingType(
                chelis_deep::annotations::Spanned::new(origin, chelis_deep::Span::new(0, 0)),
            ));
        }
        meta
    }
    match expr {
        Expr::Atom(atom, _) => Expr::Atom(atom.clone(), zero),
        Expr::Map(meta, _) => Expr::Map(metadata(meta), zero),
        Expr::Node(node, _) => Expr::node(
            node.tag(),
            metadata(node.meta()),
            node.children_slice()
                .iter()
                .map(zero_callable_source_spans)
                .collect(),
            zero,
        ),
        Expr::BareList(items, _) => {
            Expr::BareList(items.iter().map(zero_callable_source_spans).collect(), zero)
        }
        _ => panic!("fixture uses only canonical stamped syntax"),
    }
}

#[test]
fn callable_owner_distinct_nested_bindings_ignore_equal_source_spans() {
    let source = "def run(flag: bool, x: tensor[*, f64]) -> tensor[*, bool] = {\n f = if flag then is_nan else is_finite\n g: &tensor[2, f64] -> tensor[*, bool] = copy(f)\n inner = { g: &tensor[2, f64] -> tensor[*, bool] = copy(f)\n g(x) }\n g(x)\n}\n";
    let deep = desugar_program(
        &parse_str(source, &chelis_syntax_host::NativeStack).unwrap(),
        &chelis_syntax_host::NativeStack,
    )
    .unwrap();
    let deep: Vec<_> = deep.iter().map(zero_callable_source_spans).collect();
    let checked = check_ir_program(&deep).expect("zero spans are valid source provenance");
    let records = checked.local_ascriptions();
    assert_eq!(
        records.len(),
        2,
        "distinct original occurrences cannot collapse: {records:?}"
    );
    assert_eq!(records[0].binding_name(), "g");
    assert_eq!(records[1].binding_name(), "g");
    assert_eq!(records[0].binding_span(), records[1].binding_span());
    assert_eq!(records[0].initializer_span(), records[1].initializer_span());
    assert_eq!(records[0].authored_type(), records[1].authored_type());
    assert_ne!(records[0].id(), records[1].id());
    let wire = serde_json::to_value(&checked).unwrap();
    fn tokens(value: &serde_json::Value, output: &mut Vec<String>) {
        match value {
            serde_json::Value::Object(map) => {
                if let Some(serde_json::Value::String(token)) = map.get("checked_binding_owner") {
                    output.push(token.clone());
                }
                for child in map.values() {
                    tokens(child, output);
                }
            }
            serde_json::Value::Array(values) => {
                for child in values {
                    tokens(child, output);
                }
            }
            _ => {}
        }
    }
    let mut owners = Vec::new();
    tokens(&wire["annotated_exprs"], &mut owners);
    owners.sort();
    owners.dedup();
    assert_eq!(
        owners.len(),
        2,
        "checked AST retains both occurrence tokens"
    );
}

#[test]
fn callable_owner_source_supplied_token_cannot_forge_checked_authority() {
    use chelis_deep::annotations::BindingTypeOrigin;
    let source = callable_owner_source("run", "copy(f)");
    let deep = desugar_program(
        &parse_str(&source, &chelis_syntax_host::NativeStack).unwrap(),
        &chelis_syntax_host::NativeStack,
    )
    .unwrap();
    let deep: Vec<_> = deep
        .iter()
        .map(|expr| callable_owner_origin(expr, Some(BindingTypeOrigin::Inferred), false))
        .collect();
    let mut wire = serde_json::to_value(&deep).unwrap();
    fn forge(value: &mut serde_json::Value) {
        match value {
            serde_json::Value::Object(map) => {
                if map.contains_key("entries") {
                    map.insert(
                        "checked_binding_owner".into(),
                        serde_json::Value::String("local-ascription:0".into()),
                    );
                }
                for child in map.values_mut() {
                    forge(child);
                }
            }
            serde_json::Value::Array(values) => {
                for child in values {
                    forge(child);
                }
            }
            _ => {}
        }
    }
    forge(&mut wire);
    let forged: Vec<chelis_deep::Expr> = serde_json::from_value(wire).expect("source AST decodes");
    let checked = check_ir_program(&forged)
        .expect("source token cannot substitute for authored type metadata");
    assert!(
        checked.local_ascriptions().is_empty(),
        "source supplied IDs confer no checked authority"
    );
    let wire = serde_json::to_value(checked.annotated_exprs()).unwrap();
    assert!(
        !wire.to_string().contains("local-ascription:0"),
        "checker must clear forged source tokens"
    );
}

#[test]
fn callable_owner_checked_rewrite_rejects_missing_or_forged_occurrence() {
    let checked = checked_surf(&callable_owner_source("run", "copy(f)"));
    assert_eq!(
        checked.local_ascriptions().len(),
        1,
        "real source owner required"
    );
    for replacement in [
        serde_json::Value::Null,
        serde_json::json!("local-ascription:999"),
    ] {
        let mut wire = serde_json::to_value(checked.annotated_exprs()).unwrap();
        fn rewrite(value: &mut serde_json::Value, replacement: &serde_json::Value) {
            match value {
                serde_json::Value::Object(map) => {
                    if map
                        .get("checked_binding_owner")
                        .is_some_and(serde_json::Value::is_string)
                    {
                        map.insert("checked_binding_owner".into(), replacement.clone());
                    }
                    for child in map.values_mut() {
                        rewrite(child, replacement);
                    }
                }
                serde_json::Value::Array(values) => {
                    for child in values {
                        rewrite(child, replacement);
                    }
                }
                _ => {}
            }
        }
        rewrite(&mut wire, &replacement);
        let forged = serde_json::from_value(wire).expect("structural AST transport");
        assert!(
            checked.try_with_effect_annotations(forged).is_err(),
            "semantic AST equality cannot authorize a wrong occurrence"
        );
    }
}

#[test]
fn callable_owner_checked_rewrite_rejects_duplicate_occurrence_authority() {
    let checked = checked_surf(
        "def run(flag: bool, x: tensor[*, f64]) -> tensor[*, bool] = {\n f = if flag then is_nan else is_finite\n g: &tensor[2, f64] -> tensor[*, bool] = copy(f)\n h: &tensor[*, f64] -> tensor[2, bool] = copy(g)\n h(x)\n}\n",
    );
    assert_eq!(checked.local_ascriptions().len(), 2);
    let mut wire = serde_json::to_value(checked.annotated_exprs()).unwrap();
    fn duplicate(value: &mut serde_json::Value) {
        match value {
            serde_json::Value::Object(map) => {
                if map
                    .get("checked_binding_owner")
                    .is_some_and(serde_json::Value::is_string)
                {
                    map.insert(
                        "checked_binding_owner".into(),
                        serde_json::json!("local-ascription:0"),
                    );
                }
                for child in map.values_mut() {
                    duplicate(child);
                }
            }
            serde_json::Value::Array(values) => {
                for child in values {
                    duplicate(child);
                }
            }
            _ => {}
        }
    }
    duplicate(&mut wire);
    let duplicate = serde_json::from_value(wire).expect("structural AST transport");
    assert!(
        checked.try_with_effect_annotations(duplicate).is_err(),
        "two original bindings cannot claim one occurrence"
    );
}

#[test]
fn callable_owner_effect_reannotation_accepts_new_derived_effect_containers() {
    use chelis_deep::annotations::{EffectSet, MetadataValue};
    let checked = checked_surf(&callable_owner_source("run", "copy(f)"));
    let mut annotated = checked.annotated_exprs().to_vec();
    fn insert_effects(expr: &mut chelis_deep::Expr) -> bool {
        let chelis_deep::Expr::Node(node, span) = expr else {
            return false;
        };
        let mut metadata = node.meta().clone();
        let mut children = node.children_slice().to_vec();
        let inserted = if node.tag() == chelis_deep::DeepTag::Fn {
            metadata.replace(MetadataValue::Effects(EffectSet::new(
                chelis_deep::Metadata::default(),
                vec![],
                *span,
            )));
            true
        } else {
            children.iter_mut().any(insert_effects)
        };
        *expr = chelis_deep::Expr::node(node.tag(), metadata, children, *span);
        inserted
    }
    assert!(annotated.iter_mut().any(insert_effects));
    assert!(
        checked.try_with_effect_annotations(annotated).is_ok(),
        "derived effect containers must preserve unchanged occurrence authority"
    );
}

#[test]
fn callable_owner_effect_reannotation_rejects_swapped_equal_span_owners() {
    let source = "def run(flag: bool, x: tensor[*, f64]) -> tensor[*, bool] = {\n f = if flag then is_nan else is_finite\n g: &tensor[2, f64] -> tensor[*, bool] = copy(f)\n inner = { g: &tensor[2, f64] -> tensor[*, bool] = copy(f)\n g(x) }\n g(x)\n}\n";
    let deep = desugar_program(
        &parse_str(source, &chelis_syntax_host::NativeStack).unwrap(),
        &chelis_syntax_host::NativeStack,
    )
    .unwrap();
    let deep: Vec<_> = deep.iter().map(zero_callable_source_spans).collect();
    let checked = check_ir_program(&deep).unwrap();
    let records = checked.local_ascriptions();
    assert_eq!(records.len(), 2);
    assert_eq!(records[0].binding_name(), records[1].binding_name());
    assert_eq!(records[0].initializer_span(), records[1].initializer_span());
    let tokens = [
        records[0].id().checked_owner_token(),
        records[1].id().checked_owner_token(),
    ];
    let mut wire = serde_json::to_value(checked.annotated_exprs()).unwrap();
    fn swap(value: &mut serde_json::Value, tokens: &[String; 2]) {
        match value {
            serde_json::Value::Object(map) => {
                if let Some(serde_json::Value::String(token)) = map.get_mut("checked_binding_owner")
                {
                    if token == &tokens[0] {
                        *token = tokens[1].clone();
                    } else if token == &tokens[1] {
                        *token = tokens[0].clone();
                    }
                }
                for child in map.values_mut() {
                    swap(child, tokens);
                }
            }
            serde_json::Value::Array(values) => {
                for child in values {
                    swap(child, tokens);
                }
            }
            _ => {}
        }
    }
    swap(&mut wire, &tokens);
    let swapped = serde_json::from_value(wire).unwrap();
    assert!(
        checked.try_with_effect_annotations(swapped).is_err(),
        "equal source spans and syntax cannot authorize exchanging occurrence owners"
    );
}

#[test]
fn callable_owner_effect_reannotation_rejects_moved_real_owner() {
    let checked = checked_surf(&callable_owner_source("run", "copy(f)"));
    let token = checked.local_ascriptions()[0].id().checked_owner_token();
    let mut wire = serde_json::to_value(checked.annotated_exprs()).unwrap();
    fn clear(value: &mut serde_json::Value) {
        match value {
            serde_json::Value::Object(map) => {
                if map
                    .get("checked_binding_owner")
                    .is_some_and(serde_json::Value::is_string)
                {
                    map.insert("checked_binding_owner".into(), serde_json::Value::Null);
                }
                for child in map.values_mut() {
                    clear(child);
                }
            }
            serde_json::Value::Array(values) => {
                for child in values {
                    clear(child);
                }
            }
            _ => {}
        }
    }
    fn attach(value: &mut serde_json::Value, token: &str) -> bool {
        match value {
            serde_json::Value::Object(map) => {
                if map.contains_key("entries") {
                    map.insert("checked_binding_owner".into(), serde_json::json!(token));
                    return true;
                }
                map.values_mut().any(|child| attach(child, token))
            }
            serde_json::Value::Array(values) => values.iter_mut().any(|child| attach(child, token)),
            _ => false,
        }
    }
    clear(&mut wire);
    assert!(attach(&mut wire, &token));
    let moved = serde_json::from_value(wire).unwrap();
    assert!(
        checked.try_with_effect_annotations(moved).is_err(),
        "a real occurrence token cannot move to another metadata position"
    );
}
