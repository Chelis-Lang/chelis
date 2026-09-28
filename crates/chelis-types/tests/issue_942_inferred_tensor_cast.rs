//! chelis#942: tensor producers must carry enough shape evidence for `cast`
//! to consume them without an expected result annotation.
//!
//! The positive cases cover the two reported producers. The negative cases
//! preserve cast totality: a genuinely unresolved operand and a concrete
//! function operand remain errors rather than being guessed to be tensors.
//!
//! Twenty rows were removed with the two-candidate model (chelis#1277 S2b).
//! `spec/04-type-system.md` section 4.7.2 gives `expand` and `insert` one
//! result shape each, so consumer selection, the freeze default, and the
//! deferred-obligation rejection those rows asserted no longer exist. What
//! remains is chelis#942's own subject plus the reshape and cast rows, whose
//! `expand` operands all carry a unit extent at the axis and are therefore
//! same-rank broadcasts under the single meaning.

use chelis_deep::parse_and_stamp_file;
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as parse_surf;
use chelis_types::errors::{CheckError, CheckErrorKind};
use chelis_types::{check_ir_program, check_typed_program};

fn typecheck(source: &str) -> Vec<CheckError> {
    let decls = parse_surf(source).expect("surf parse should succeed");
    let deep = desugar_program(&decls).expect("Surf fixture must desugar");
    match check_typed_program(&deep) {
        Ok(_) => Vec::new(),
        Err(errors) => errors.errors,
    }
}

fn typecheck_ir(exprs: &[chelis_deep::Expr]) -> Vec<CheckError> {
    match check_ir_program(exprs) {
        Ok(_) => Vec::new(),
        Err(errors) => errors.errors,
    }
}

fn typecheck_stamped(exprs: &[chelis_deep::Expr]) -> Vec<CheckError> {
    match check_typed_program(exprs) {
        Ok(_) => Vec::new(),
        Err(errors) => errors.errors,
    }
}

fn summary(errors: &[CheckError]) -> String {
    errors
        .iter()
        .map(|error| format!("[{:?}] {}", error.kind, error.message))
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn otherwise_unconsumed_expand_materializes_its_context_free_default() {
    let decls =
        parse_surf("t = expand(to_tensor([0.5f32]), 0, 8i64)").expect("surf parse should succeed");
    let deep = desugar_program(&decls).expect("Surf fixture must desugar");
    let checked = check_typed_program(&deep).expect("bare expand binding should typecheck");
    let rendered = chelis_deep::printer::print_canonical(checked.annotated_exprs());
    assert!(
        !rendered.contains("(t-var"),
        "an otherwise-unconsumed expand must not escape with an unresolved type:\n{rendered}"
    );
    assert!(
        rendered.contains("(t-tensor {} (d-lit {} 8) (t-prim {} f32))"),
        "an existing-axis default must materialize same-rank replacement:\n{rendered}"
    );
}

#[test]
fn otherwise_unconsumed_expand_rejects_axis_beyond_trailing_position() {
    let errors = typecheck("t = expand(to_tensor([0.5f32]), 2, 8i64)");
    assert!(
        errors.iter().any(|error| {
            matches!(error.kind, CheckErrorKind::DimensionMismatch)
                && error
                    .message
                    .contains("expand axis 2 is out of bounds for rank 1 tensor")
        }),
        "an axis beyond the sole trailing insertion position must fail:\n{}",
        summary(&errors)
    );
}

#[test]
fn cast_accepts_unannotated_positional_expand_result() {
    let errors = typecheck(
        r#"
t = expand(to_tensor([0.5f32]), 0, 8i64)
c = cast(t, f64)
"#,
    );
    assert!(
        errors.is_empty(),
        "expand must expose its inferred tensor result to cast:\n{}",
        summary(&errors)
    );
}

#[test]
fn context_free_positional_expand_retains_same_rank_replacement() {
    let errors = typecheck(
        r#"
def require_rank_one(x: tensor[8, f32]) -> tensor[8, f32] = x
t = expand(to_tensor([0.5f32]), 0, 8i64)
y = require_rank_one(t)
"#,
    );
    assert!(
        errors.is_empty(),
        "context-free positional expand must replace the selected axis:\n{}",
        summary(&errors)
    );
}

#[test]
fn reshape_shape_read_selects_replacement_before_axis_validation_on_every_ingress() {
    for (borrow, source) in [
        (
            "unborrowed",
            "expanded = expand(to_tensor([0.0f32]), 0, 6i64)\n\
             result = reshape(expanded, [shape(expanded, 1), 6i64])",
        ),
        (
            "borrowed",
            "expanded = expand(to_tensor([0.0f32]), 0, 6i64)\n\
             result = reshape(expanded, [shape(&expanded, 1), 6i64])",
        ),
    ] {
        let decls = parse_surf(source).expect("surf parse should succeed");
        let lists = desugar_program(&decls).expect("Surf fixture must desugar");
        let canonical = chelis_deep::printer::print_canonical(&lists);
        let nodes = parse_and_stamp_file(&canonical).expect("canonical Deep should stamp");

        for (ingress, errors) in [
            ("list", typecheck_ir(&lists)),
            ("node", typecheck_stamped(&nodes)),
        ] {
            assert_eq!(
                errors.len(),
                1,
                "a {borrow} shape read through {ingress} ingress must reject exactly once:\n{}",
                summary(&errors)
            );
            assert!(
                matches!(errors[0].kind, CheckErrorKind::DimensionMismatch)
                    && errors[0]
                        .message
                        .contains("shape axis 1 is out of bounds for rank 1 tensor"),
                "a {borrow} shape read through {ingress} ingress must select same-rank \
                 replacement before checking its axis:\n{}",
                summary(&errors)
            );
        }
    }
}

#[test]
fn expand_replacement_and_reshape_output_agree() {
    let errors = typecheck(
        r#"
def require_replaced(x: tensor[3, 2, 1, f32]) -> tensor[3, 2, 1, f32] = x
def require_replaced_reshape(x: tensor[2, 1, 3, f32]) -> tensor[2, 1, 3, f32] = x
def f(bias: tensor[1, 2, 1, f32]) -> tensor[2, 1, 3, f32] = {
  expanded = expand(bias, 0, 3i64)
  reshaped = reshape(expanded, [shape(expanded, 1), shape(expanded, 2), 3i64])
  replaced = require_replaced(expanded)
  selected = require_replaced_reshape(reshaped)
  selected
}
"#,
    );
    assert!(
        errors.is_empty(),
        "expand replacement must agree with the immediately published reshape output:\n{}",
        summary(&errors)
    );
}

#[test]
fn reshape_output_rejects_a_contradictory_consumer() {
    let errors = typecheck(
        r#"
def require_replaced(x: tensor[3, 2, 1, f32]) -> tensor[3, 2, 1, f32] = x
def require_inserted_reshape(x: tensor[1, 2, 3, f32]) -> tensor[1, 2, 3, f32] = x
def f(bias: tensor[1, 2, 1, f32]) -> tensor[1, 2, 3, f32] = {
  expanded = expand(bias, 0, 3i64)
  reshaped = reshape(expanded, [shape(expanded, 1), shape(expanded, 2), 3i64])
  replaced = require_replaced(expanded)
  selected = require_inserted_reshape(reshaped)
  selected
}
"#,
    );
    assert!(
        errors
            .iter()
            .any(|error| matches!(error.kind, CheckErrorKind::DimensionMismatch)),
        "the published reshape output must reject a contradictory consumer:\n{}",
        summary(&errors)
    );
}

#[test]
fn chained_reshape_consumes_the_published_first_shape() {
    let errors = typecheck(
        r#"
def f(bias: tensor[1, 2, 1, f32]) -> tensor[6, f32] = {
  expanded = expand(bias, 0, 3i64)
  first = reshape(expanded, [shape(expanded, 1), shape(expanded, 2), 3i64])
  reshape(first, [6i64])
}
"#,
    );
    assert!(
        errors.is_empty(),
        "a second reshape must consume the first reshape's published shape:\n{}",
        summary(&errors)
    );
}

#[test]
fn chained_reshape_rejects_incompatible_static_numel() {
    let errors = typecheck(
        r#"
def f(bias: tensor[1, 2, 1, f32]) = {
  expanded = expand(bias, 0, 3i64)
  first = reshape(expanded, [shape(expanded, 1), shape(expanded, 2), 3i64])
  reshape(first, [5i64])
}
"#,
    );
    assert!(
        errors
            .iter()
            .any(|error| matches!(error.kind, CheckErrorKind::DimensionMismatch)),
        "a chained reshape with the wrong element count must fail:\n{}",
        summary(&errors)
    );
}

#[test]
fn reshape_rejects_negative_static_extents_even_when_signed_product_matches() {
    let errors = typecheck(
        r#"
def f(x: tensor[6, f32]) = reshape(x, [-2i64, -3i64])
"#,
    );
    assert!(
        errors.iter().any(|error| {
            matches!(error.kind, CheckErrorKind::DimensionMismatch)
                && error.message.contains("non-negative")
        }),
        "negative reshape extents must fail before signed-product comparison:\n{}",
        summary(&errors)
    );
}

#[test]
fn reshape_rejects_extreme_negative_extent_even_when_zero_masks_product() {
    let errors = typecheck(
        r#"
def f(x: tensor[0, f32]) =
  reshape(x, [-9223372036854775808i64, 0i64])
"#,
    );
    assert!(
        errors.iter().any(|error| {
            matches!(error.kind, CheckErrorKind::DimensionMismatch)
                && error.message.contains("non-negative")
        }),
        "zero product must not hide an invalid negative reshape extent:\n{}",
        summary(&errors)
    );
}

#[test]
fn scalar_reshape_rejects_negative_static_extent() {
    let errors = typecheck(
        r#"
def f(x: f32) = reshape(x, [-1i64])
"#,
    );
    assert!(
        errors.iter().any(|error| {
            matches!(error.kind, CheckErrorKind::DimensionMismatch)
                && error.message.contains("non-negative")
        }),
        "scalar reshape must reject a negative target extent:\n{}",
        summary(&errors)
    );
}

#[test]
fn deferred_expand_reshape_rejects_negative_static_extents() {
    let errors = typecheck(
        r#"
def f(bias: tensor[1, 2, 1, f32]) = {
  expanded = expand(bias, 0, 3i64)
  reshape(expanded, [-2i64, -3i64])
}
"#,
    );
    assert!(
        errors.iter().any(|error| {
            matches!(error.kind, CheckErrorKind::DimensionMismatch)
                && error.message.contains("non-negative")
        }),
        "a deferred expand relation must not admit a signed-product reshape:\n{}",
        summary(&errors)
    );
}

#[test]
fn chained_deferred_reshape_rejects_negative_static_extents() {
    let errors = typecheck(
        r#"
def f(bias: tensor[1, 2, 1, f32]) = {
  expanded = expand(bias, 0, 3i64)
  first = reshape(expanded, [shape(expanded, 1), shape(expanded, 2), 3i64])
  reshape(first, [-2i64, -3i64])
}
"#,
    );
    assert!(
        errors.iter().any(|error| {
            matches!(error.kind, CheckErrorKind::DimensionMismatch)
                && error.message.contains("non-negative")
        }),
        "a chained deferred reshape must reject negative target extents:\n{}",
        summary(&errors)
    );
}

#[test]
fn reshape_accepts_equal_static_products_beyond_i64() {
    let errors = typecheck(
        r#"
def f(x: tensor[9223372036854775807, 2, f32]) -> tensor[9223372036854775807, 2, f32] =
  reshape(x, [9223372036854775807i64, 2i64])
"#,
    );
    assert!(
        errors.is_empty(),
        "equal static products must compare exactly even when they exceed i64:\n{}",
        summary(&errors)
    );
}

#[test]
fn reshape_rejects_unequal_static_products_beyond_i64() {
    let errors = typecheck(
        r#"
def f(x: tensor[9223372036854775807, 2, f32]) -> tensor[9223372036854775807, f32] =
  reshape(x, [9223372036854775807i64])
"#,
    );
    assert!(
        errors
            .iter()
            .any(|error| matches!(error.kind, CheckErrorKind::DimensionMismatch)),
        "overflow must not turn a known static element-count mismatch into unknown:\n{}",
        summary(&errors)
    );
}

#[test]
fn declared_same_rank_result_selects_expand_replacement() {
    let errors = typecheck(
        r#"
def replace_axis(x: tensor[1, f32]) -> tensor[8, f32] = expand(&x, 0, 8i64)
"#,
    );
    assert!(
        errors.is_empty(),
        "a declared same-rank result must select positional replacement:\n{}",
        summary(&errors)
    );
}

#[test]
fn declared_rank_increasing_result_takes_insert() {
    let errors = typecheck(
        r#"
def insert_axis(x: tensor[1, f32]) -> tensor[8, 1, f32] = insert(&x, 0, 8i64)
"#,
    );
    assert!(
        errors.is_empty(),
        "a declared rank-plus-one result is `insert`'s shape:\n{}",
        summary(&errors)
    );
}

#[test]
fn cast_accepts_uniform_like_result_inferred_from_template() {
    let errors = typecheck(
        r#"
t = expand(to_tensor([0.0f32]), 0, 8i64)
u = uniform_like(key_from_seed(1i64), t, 0.0, 1.0)
c = cast(u, f64)
"#,
    );
    assert!(
        errors.is_empty(),
        "uniform_like must preserve the template tensor type for cast:\n{}",
        summary(&errors)
    );
}

#[test]
fn cast_rejects_genuinely_unresolved_operand() {
    let errors = typecheck("def f(x) = cast(x, f64)");
    assert!(
        errors.iter().any(|error| {
            matches!(error.kind, CheckErrorKind::CastNonTensor)
                && error.message.contains("cast requires tensor or prim type")
        }),
        "an unconstrained operand must not be guessed to be a tensor:\n{}",
        summary(&errors)
    );
}

#[test]
fn cast_rejects_concrete_function_operand() {
    let errors = typecheck(
        r#"
def identity(x) = x
c = cast(identity, f64)
"#,
    );
    assert!(
        errors.iter().any(|error| {
            matches!(error.kind, CheckErrorKind::CastNonTensor)
                && error.message.contains("cast requires tensor or prim type")
        }),
        "a function operand must remain outside the cast domain:\n{}",
        summary(&errors)
    );
}
