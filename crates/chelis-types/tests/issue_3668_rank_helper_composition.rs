//! Body evidence, ordered rows, and library ownership under spec/04 §4.5.3.
//! Test-first: these use existing APIs; no production summary API is assumed.

use chelis_deep::Expr;
use chelis_surf::{desugar::desugar_program, parser::parse_str};
use chelis_types::errors::CheckErrorKind;
use chelis_types::{
    CheckedProgram, InferResult, TypeEnv, build_compiled_library_context,
    build_compiled_library_context_with_base, check_ir_program, check_ir_with_context,
    check_linearity, check_typed_program,
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
fn declared_precision_and_effect_bounds_are_not_body_evidence_exemptions() {
    both(
        &format!("{LIBRARY}{WRAPPER}out = wrapped(to_tensor([1i32, 2i32]))\n"),
        false,
    );
    both(
        "def noisy[r](x: &tensor[..r, f32]) -> tensor[..r, f32] ! {IO} = { print(\"called\")\n neg(x) }\n\
         def outer[s](x: &tensor[..s, f32]) -> tensor[..s, f32] ! {} = noisy(x)\n",
        false,
    );
    both(
        "def noisy[r](x: &tensor[..r, f32]) -> tensor[..r, f32] ! {IO} = { print(\"called\")\n neg(x) }\n\
         def outer[s](x: &tensor[..s, f32]) -> tensor[..s, f32] ! {IO} = noisy(x)\n",
        true,
    );
}

#[test]
fn builtin_spelling_is_neither_evidence_nor_an_automatic_rejection() {
    both(
        "def relu[r, p: Float](x: &tensor[..r, p]) -> tensor[..r, p] = neg(x)\n\
         def outer[s, q: Float](x: &tensor[..s, q]) -> tensor[..s, q] = relu(x)\n",
        true,
    );
    both(
        "def relu[a, b](x: &tensor[a, b, f32]) -> tensor[b, a, f32] = permute(x, 1i32, 0i32)\n\
         def outer[r](x: &tensor[..r, f32]) -> tensor[..r, f32] = relu(x)\n",
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
fn recursive_components_never_supply_provisional_body_evidence() {
    for source in [
        "def loop[r](x: &tensor[..r, f32]) -> tensor[..r, f32] = loop(x)\n",
        "def left[r](x: &tensor[..r, f32]) -> tensor[..r, f32] = right(x)\n\
         def right[s](x: &tensor[..s, f32]) -> tensor[..s, f32] = left(x)\n",
    ] {
        both(source, false);
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
    both(
        "sig foreign[r, p: Float]: &tensor[..r, p] -> tensor[..r, p]\n\
         foreign = foreign\n\
         def outer[s, q: Float](x: &tensor[..s, q]) -> tensor[..s, q] = foreign(x)\n",
        false,
    );
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
