//! Body evidence, ordered rows, and library ownership under spec/04 §4.5.3.
//! Test-first: these use existing APIs; no production summary API is assumed.

use chelis_deep::Expr;
use chelis_surf::{desugar::desugar_program, parser::parse_str};
use chelis_types::errors::CheckErrorKind;
use chelis_types::infer::{
    FunctionSignatureInference, ParamSignatureInference, SignatureInferenceMetadata,
};
use chelis_types::types::Type;
use chelis_types::{
    CheckedProgram, InferResult, TypeEnv, build_compiled_library_context,
    build_compiled_library_context_with_base, check_ir_program, check_ir_with_context,
    check_ir_with_signature_context, check_linearity, check_typed_program,
    install_linked_program_guard,
};

fn parse(source: &str) -> Vec<Expr> {
    desugar_program(&parse_str(source).expect("canonical Surf fixture parses"))
        .expect("fixture desugars")
}

fn clean(result: Result<CheckedProgram, InferResult>, label: &str) {
    if let Err(result) = result {
        panic!("{label}: {:?}", result.errors);
    }
}

fn reject(result: Result<CheckedProgram, InferResult>, label: &str) {
    let errors = result.expect_err(label).errors;
    assert!(!errors.is_empty(), "{label}: rejection has diagnostics");
    assert!(
        errors
            .iter()
            .all(|error| !matches!(error.kind, CheckErrorKind::UnboundVariable { .. })),
        "{label}: fixture must resolve its names: {errors:?}"
    );
}

fn both(source: &str, accept: bool) {
    let program = parse(source);
    for (label, result) in [
        ("IR checker", check_ir_program(&program)),
        ("typed checker", check_typed_program(&program)),
    ] {
        if accept {
            clean(result, label);
        } else {
            reject(result, label);
        }
    }
}

const LIBRARY: &str = "def negate[r, p: Float](x: &tensor[..r, p]) -> tensor[..r, p] = neg(x)\n";
const WRAPPER: &str =
    "def wrapped[s, q: Float](x: &tensor[..s, q]) -> tensor[..s, q] = negate(x)\n";

#[test]
fn direct_and_multi_hop_helpers_preserve_complete_parametric_rows() {
    for bound in ["Float", "{f32, f64}"] {
        both(
            &format!(
                "def outer[s, q: {bound}](x: &tensor[..s, q]) -> tensor[..s, q] = middle(x)\n\
                 def middle[t, v: {bound}](x: &tensor[..t, v]) -> tensor[..t, v] = negate(x)\n\
                 def negate[r, p: {bound}](x: &tensor[..r, p]) -> tensor[..r, p] = neg(x)\n"
            ),
            true,
        );
    }
}

#[test]
fn named_axis_helper_composes_under_the_same_unitary_anchors() {
    both(
        "def reduce_seq[pre, post, p: Float](x: &tensor[..pre, seq, ..post, p]) -> tensor[..pre, ..post, p] = sum(x, seq)\n\
         def wrapped[before, after, q: Float](x: &tensor[..before, seq, ..after, q]) -> tensor[..before, ..after, q] = reduce_seq(x)\n",
        true,
    );
}

#[test]
fn concrete_helpers_cannot_hide_positional_shape_rewrites() {
    for helper in [
        "def helper[a, b](x: &tensor[a, b, f32]) -> tensor[b, a, f32] = permute(x, 1i32, 0i32)\n",
        "def helper(x: &tensor[2, 3, f32]) -> tensor[3, 2, f32] = reshape(x, [3i64, 2i64])\n",
    ] {
        both(helper, true);
        both(
            &format!(
                "{helper}\
                 def outer[r](x: &tensor[..r, f32]) -> tensor[..r, f32] = helper(x)\n"
            ),
            false,
        );
    }
}

#[test]
fn shared_helper_row_does_not_unify_independent_caller_rows() {
    both(
        "def combine[r, p: Float](x: &tensor[..r, p], y: &tensor[..r, p]) -> tensor[..r, p] = add(x, y)\n\
         def outer[a, b, q: Float](x: &tensor[..a, q], y: &tensor[..b, q]) -> tensor[..a, q] = combine(x, y)\n",
        false,
    );
}

#[test]
fn declared_precision_bounds_are_not_body_evidence_exemptions() {
    both(
        &format!("{LIBRARY}{WRAPPER}out = wrapped(to_tensor([1i32, 2i32]))\n"),
        false,
    );
}

#[test]
fn effectful_body_can_have_row_evidence_without_erasing_its_declared_effects() {
    both(
        "def noisy[r](x: &tensor[..r, f32]) -> tensor[..r, f32] ! {IO} = { shown = print(\"called\")\n neg(x) }\n\
         def outer[s](x: &tensor[..s, f32]) -> tensor[..s, f32] ! {IO} = noisy(x)\n",
        true,
    );
}

#[test]
fn builtin_spelling_is_neither_evidence_nor_an_automatic_rejection() {
    // Package linking preserves the source spelling as a qualified declaration
    // identity. An unqualified top-level `relu` is independently forbidden by
    // spec/04 §8.6 and cannot test rank-body evidence.
    let _linked = install_linked_program_guard();
    both(
        "def pkg__library__Math__relu[r, p: Float](x: &tensor[..r, p]) -> tensor[..r, p] = neg(x)\n\
         def outer[s, q: Float](x: &tensor[..s, q]) -> tensor[..s, q] = pkg__library__Math__relu(x)\n",
        true,
    );
    both(
        "def pkg__library__Math__relu[a, b](x: &tensor[a, b, f32]) -> tensor[b, a, f32] = permute(x, 1i32, 0i32)\n\
         def outer[r](x: &tensor[..r, f32]) -> tensor[..r, f32] = pkg__library__Math__relu(x)\n",
        false,
    );
}

#[test]
fn lexical_shadowing_cannot_reuse_a_verified_declaration_summary() {
    for body in [
        "negate(x)",
        "{ local = negate\n local(x) }",
        "(if flag then negate else negate)(x)",
    ] {
        both(
            &format!(
                "{LIBRARY}def outer[r, p: Float](negate: &tensor[..r, p] -> tensor[..r, p], x: &tensor[..r, p], flag: bool) -> tensor[..r, p] = {body}\n"
            ),
            false,
        );
    }
}

#[test]
fn builtin_named_parameters_have_no_intrinsic_body_evidence() {
    for callee in ["neg", "abs", "relu"] {
        both(
            &format!(
                "def outer[r, p: Float]({callee}: &tensor[..r, p] -> tensor[..r, p], x: &tensor[..r, p]) -> tensor[..r, p] = {callee}(x)\n"
            ),
            false,
        );
    }
}

#[test]
fn jointly_checked_recursive_bodies_preserve_rows_without_a_termination_claim() {
    for bound in ["Float", "{f32, f64}"] {
        both(
            &format!(
                "def recur[r, p: {bound}](x: &tensor[..r, p], stop: bool) -> tensor[..r, p] = if stop then neg(x) else recur(x, true)\n"
            ),
            true,
        );
        both(
            &format!(
                "def left[r, p: {bound}](x: &tensor[..r, p], stop: bool) -> tensor[..r, p] = if stop then neg(x) else right(x, true)\n\
                 def right[s, q: {bound}](x: &tensor[..s, q], stop: bool) -> tensor[..s, q] = if stop then neg(x) else left(x, true)\n"
            ),
            true,
        );
    }
    // These programs are checked only. Divergence supplies no returned value
    // whose shape could violate the relation; it is not an execution oracle.
    for source in [
        "def loop[r](x: &tensor[..r, f32]) -> tensor[..r, f32] = loop(x)\n",
        "def left[r](x: &tensor[..r, f32]) -> tensor[..r, f32] = right(x)\n\
         def right[s](x: &tensor[..s, f32]) -> tensor[..s, f32] = left(x)\n",
    ] {
        both(source, true);
    }
}

#[test]
fn recursive_component_cannot_hide_an_unsafe_member_or_growing_result_row() {
    for source in [
        "def left[r](x: &tensor[..r, f32], stop: bool) -> tensor[..r, f32] = if stop then neg(x) else right(x, true)\n\
         def right[s](x: &tensor[..s, f32], stop: bool) -> tensor[..s, f32] = if stop then permute(x, 1i32, 0i32) else left(x, true)\n",
        "def grow[r](x: &tensor[..r, f32]) -> tensor[..r, f32] = grow(insert(x, extra, 1i64))\n",
        "def left[r, p: Float](x: &tensor[..r, p]) -> tensor[..r, p] = right(cast(x, f32))\n\
         def right[s, q: Float](x: &tensor[..s, q]) -> tensor[..s, q] = left(x)\n",
        "def left[r](x: &tensor[..r, f32], callback: &tensor[..r, f32] -> tensor[..r, f32]) -> tensor[..r, f32] = right(x, callback)\n\
         def right[s](x: &tensor[..s, f32], callback: &tensor[..s, f32] -> tensor[..s, f32]) -> tensor[..s, f32] = left(callback(x), callback)\n",
    ] {
        both(source, false);
    }
}

#[test]
fn recursive_component_evidence_survives_context_round_trip() {
    let library = parse(
        "def left[r, p: Float](x: &tensor[..r, p], stop: bool) -> tensor[..r, p] = if stop then neg(x) else right(x, true)\n\
         def right[s, q: Float](x: &tensor[..s, q], stop: bool) -> tensor[..s, q] = if stop then neg(x) else left(x, true)\n",
    );
    let (context, checked) =
        build_compiled_library_context(&library).expect("all members establish evidence together");
    assert!(context.matches_checked_program(&checked));
    let restored: TypeEnv =
        bincode::deserialize(&bincode::serialize(&context).expect("encode SCC"))
            .expect("decode SCC");
    for base in [context, restored] {
        clean(
            check_ir_with_context(
                &base,
                &parse(
                    "def outer[t, v: Float](x: &tensor[..t, v]) -> tensor[..t, v] = left(x, false)\n",
                ),
            ),
            "imported joint evidence",
        );
        let bad = parse(
            "def left[r](x: &tensor[..r, f32]) -> tensor[..r, f32] = right(x)\n\
             def right[s](x: &tensor[..s, f32]) -> tensor[..s, f32] = left(permute(x, 1i32, 0i32))\n",
        );
        assert!(
            build_compiled_library_context_with_base(&base, &bad).is_err(),
            "unsafe replacement SCC cannot inherit accepted base summaries"
        );
        clean(
            check_ir_with_context(
                &base,
                &parse(
                    "def again[t, v: Float](x: &tensor[..t, v]) -> tensor[..t, v] = right(x, true)\n",
                ),
            ),
            "failed component does not mutate its base",
        );
    }
}

#[test]
fn accepted_library_evidence_survives_layering_and_binary_round_trip() {
    let (base, checked_base) = build_compiled_library_context(&parse(LIBRARY))
        .expect("builtin-only library checks before helper composition");
    assert!(base.matches_checked_program(&checked_base));
    let restored: TypeEnv = bincode::deserialize(&bincode::serialize(&base).expect("encode base"))
        .expect("decode base");
    for context in [base, restored] {
        clean(
            check_ir_with_context(&context, &parse(WRAPPER)),
            "direct context",
        );
        let (layer, checked_layer) =
            build_compiled_library_context_with_base(&context, &parse(WRAPPER))
                .expect("layered helper composition checks");
        assert!(layer.matches_checked_program(&checked_layer));
        let layer: TypeEnv =
            bincode::deserialize(&bincode::serialize(&layer).expect("encode layer"))
                .expect("decode layer");
        clean(
            check_ir_with_context(
                &layer,
                &parse(
                    "def outer[t, p: Float](x: &tensor[..t, p]) -> tensor[..t, p] = wrapped(x)\n",
                ),
            ),
            "second checked hop after restored layer",
        );
    }
}

#[test]
fn replacement_owner_does_not_inherit_base_evidence_or_mutate_the_base() {
    let (base, _) = build_compiled_library_context(&parse(LIBRARY)).expect("library checks");
    let replacement = parse(
        "def negate[a, b](x: &tensor[a, b, f32]) -> tensor[b, a, f32] = permute(x, 1i32, 0i32)\n\
         def outer[r](x: &tensor[..r, f32]) -> tensor[..r, f32] = negate(x)\n",
    );
    reject(
        check_ir_with_context(&base, &replacement),
        "shadowed context rejects",
    );
    assert!(build_compiled_library_context_with_base(&base, &replacement).is_err());
    clean(
        check_ir_with_context(&base, &parse(WRAPPER)),
        "base remains unchanged",
    );
}

#[test]
fn a_function_signature_without_a_checked_body_supplies_no_evidence() {
    const EXTERNAL: &str = "pkg__dep__Api__external";
    // The existing linked-interface ingress (#850) admits this complete
    // signature without inventing an authored orphan or an eager value cycle.
    let (context, _) = {
        let _linked = install_linked_program_guard();
        build_compiled_library_context(&parse(&format!(
            "sig {EXTERNAL}[r, p: Float]: &tensor[..r, p] -> tensor[..r, p]\n"
        )))
        .expect("valid linked signature-only context")
    };
    let scheme = context
        .scheme(EXTERNAL)
        .expect("external signature is bound");
    assert_eq!(scheme.rvars.len(), 1);
    assert_eq!(scheme.tvars.len(), 1);
    assert_eq!(scheme.tvar_restrictions.len(), 1, "Float bound retained");
    let signature = scheme.body.clone();
    let Type::Fn(params, _) = &signature else {
        panic!("external signature must be callable");
    };
    assert_eq!(params.len(), 1);
    let mut signatures = SignatureInferenceMetadata::default();
    signatures.functions.insert(
        EXTERNAL.to_string(),
        FunctionSignatureInference {
            name: EXTERNAL.to_string(),
            authored_signature: true,
            authored_signature_type: Some(signature.clone()),
            recursive_cycle: false,
            checked_signature: signature.clone(),
            display_signature: signature.clone(),
            params: vec![ParamSignatureInference {
                index: 0,
                name: "x".to_string(),
                written: true,
                inferred_read_only: false,
                checked_type: params[0].clone(),
                display_type: params[0].clone(),
            }],
        },
    );
    // Concrete call typing needs the signature, not a row-body certificate.
    // This control makes no execution claim for the absent implementation.
    clean(
        check_ir_with_signature_context(
            &context,
            &signatures,
            &parse(&format!(
                "def concrete(x: &tensor[2, 3, f32]) -> tensor[2, 3, f32] = {EXTERNAL}(x)\n"
            )),
        ),
        "signature context resolves and instantiates the callable",
    );
    let result = check_ir_with_signature_context(
        &context,
        &signatures,
        &parse(&format!(
            "def outer[s, q: Float](x: &tensor[..s, q]) -> tensor[..s, q] = {EXTERNAL}(x)\n"
        )),
    );
    let errors = &result.as_ref().expect_err("body evidence is absent").errors;
    assert!(
        errors.iter().any(
            |error| matches!(error.kind, CheckErrorKind::DimensionMismatch)
                && error.message.contains(EXTERNAL)
                && error.message.contains("rank-polymorphic")
                && error.message.contains("body evidence")
        ),
        "expected the missing body-evidence rejection, not a binding failure: {errors:?}"
    );
    reject(result, "body-less callable has no rank-body authority");
}

#[test]
fn helper_evidence_does_not_allow_borrowed_values_to_escape() {
    let program = parse(
        "def borrowed[r](x: tensor[..r, f32]) -> &tensor[..r, f32] = &x\n\
         def outer[s](x: tensor[..s, f32]) -> &tensor[..s, f32] = borrowed(x)\n",
    );
    let checked = check_typed_program(&program).expect("row-safe body is independently typed");
    let errors = check_linearity(&checked).expect_err("borrowed function result cannot escape");
    assert!(
        errors
            .iter()
            .any(|error| matches!(error.kind, CheckErrorKind::InvalidBorrow))
    );
}
