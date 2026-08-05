//! chelis#942: tensor producers must carry enough deferred shape evidence for
//! `cast` to consume them without an expected result annotation.
//!
//! The positive cases cover the two reported producers. The negative
//! cases preserve cast totality: a genuinely unresolved operand and a
//! concrete function operand remain errors rather than being guessed to be
//! tensors. The positional-expand rows also lock ordinary consumer selection,
//! scalar/trailing insertion, and rejection of a third output rank.

use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as parse_surf;
use chelis_types::errors::{CheckError, CheckErrorKind};
use chelis_types::{
    TypeEnv, build_type_env_from_library, check_ir_with_context, check_typed_program,
};

fn typecheck(source: &str) -> Vec<CheckError> {
    let decls = parse_surf(source).expect("surf parse should succeed");
    let deep = desugar_program(&decls);
    match check_typed_program(&deep) {
        Ok(_) => Vec::new(),
        Err(errors) => errors.errors,
    }
}

fn checked_render(source: &str) -> String {
    let decls = parse_surf(source).expect("surf parse should succeed");
    let deep = desugar_program(&decls);
    let checked = check_typed_program(&deep).expect("program should typecheck");
    chelis_deep::printer::print_canonical(checked.annotated_exprs())
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
    let deep = desugar_program(&decls);
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
fn otherwise_unconsumed_trailing_expand_materializes_insertion() {
    let decls =
        parse_surf("t = expand(to_tensor([0.5f32]), 1, 8i64)").expect("surf parse should succeed");
    let deep = desugar_program(&decls);
    let checked = check_typed_program(&deep).expect("trailing expand binding should typecheck");
    let rendered = chelis_deep::printer::print_canonical(checked.annotated_exprs());
    assert!(
        rendered.contains("(t-tensor {} (d-lit {} 1) (d-lit {} 8) (t-prim {} f32))"),
        "axis equal to rank must materialize trailing insertion:\n{rendered}"
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
                    .contains("expand insert axis 2 is out of bounds for rank 1 tensor")
        }),
        "an axis beyond the sole trailing insertion position must fail:\n{}",
        summary(&errors)
    );
}

#[test]
fn serialized_library_context_preserves_later_expand_shape_selection() {
    let library_decls = parse_surf(
        "bias = to_tensor([0.5f32, 1.5f32, 2.5f32])\n\
         expanded = expand(bias, 0, 2i64)",
    )
    .expect("library surf parse should succeed");
    let library = desugar_program(&library_decls);
    let context = build_type_env_from_library(&library).expect("library context should build");
    let bytes = bincode::serialize(&context).expect("type context should serialize");
    let restored: TypeEnv = bincode::deserialize(&bytes).expect("type context should deserialize");

    let consumer_decls = parse_surf(
        "def require_rank_two(x: tensor[2, 3, f32]) -> tensor[2, 3, f32] = x\n\
         y = require_rank_two(expanded)",
    )
    .expect("consumer surf parse should succeed");
    let consumer = desugar_program(&consumer_decls);
    check_ir_with_context(&restored, &consumer)
        .expect("a serialized reusable context must preserve rank-increasing selection");
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
fn later_tensor_consumer_selects_rank_increasing_expand() {
    let errors = typecheck(
        r#"
bias = to_tensor([0.5f32, 1.5f32, 2.5f32])
expanded = expand(bias, 0, 2i64)
matrix = to_tensor([[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]])
y = add(matrix, expanded)
"#,
    );
    assert!(
        errors.is_empty(),
        "a later tensor consumer must be able to select rank-increasing insertion:\n{}",
        summary(&errors)
    );
}

#[test]
fn reshape_binds_deferred_expand_precision_from_its_input() {
    let errors = typecheck(
        r#"
def flatten_expanded(bias: tensor[2, f32]) = {
  expanded = expand(bias, 0, cast(3, int64))
  reshape(expanded, [cast(6, int64)])
}
flat = flatten_expanded(to_tensor([1.0f32, 2.0f32]))
"#,
    );
    assert!(
        errors.is_empty(),
        "reshape must bind its unresolved input precision to the expand operand:\n{}",
        summary(&errors)
    );
}

#[test]
fn reshape_shape_read_selects_inserted_expand_axis() {
    let decls = parse_surf(
        r#"
def f(bias: tensor[2, f32]) = {
  expanded = expand(bias, 0, 3i64)
  reshape(expanded, [shape(expanded, 1), 3i64])
}
"#,
    )
    .expect("surf parse should succeed");
    let deep = desugar_program(&decls);
    let checked = check_typed_program(&deep).expect("the insertion case should typecheck");
    let rendered = chelis_deep::printer::print_canonical(checked.annotated_exprs());
    assert!(
        rendered.contains("(t-tensor {} (d-lit {} 3) (d-lit {} 2) (t-prim {} f32))"),
        "the inserted-axis read must select tensor[3, 2, f32]:\n{rendered}"
    );
}

#[test]
fn reshape_rejects_axis_outside_every_deferred_expand_candidate() {
    let errors = typecheck(
        r#"
def f(bias: tensor[2, f32]) = {
  expanded = expand(bias, 0, 3i64)
  reshape(expanded, [shape(expanded, 2)])
}
"#,
    );
    assert!(
        errors
            .iter()
            .any(|error| matches!(error.kind, CheckErrorKind::DimensionMismatch)),
        "axis two exists in neither legal expand shape:\n{}",
        summary(&errors)
    );
}

#[test]
fn reshape_rejects_replacement_after_shared_expand_selected_insertion() {
    let errors = typecheck(
        r#"
def f(bias: tensor[2, f32]) = {
  expanded = expand(bias, 0, 3i64)
  inserted = reshape(expanded, [6i64])
  reshape(expanded, [3i64])
}
"#,
    );
    assert!(
        errors.iter().any(|error| {
            matches!(error.kind, CheckErrorKind::DimensionMismatch)
                && error
                    .message
                    .contains("reshape target has 3 elements but input tensor has 6")
        }),
        "one deferred binding cannot select insertion then replacement:\n{}",
        summary(&errors)
    );
}

#[test]
fn reshape_rejects_insertion_after_shared_expand_selected_replacement() {
    let errors = typecheck(
        r#"
def f(bias: tensor[2, f32]) = {
  expanded = expand(bias, 0, 3i64)
  replaced = reshape(expanded, [3i64])
  reshape(expanded, [6i64])
}
"#,
    );
    assert!(
        errors.iter().any(|error| {
            matches!(error.kind, CheckErrorKind::DimensionMismatch)
                && error
                    .message
                    .contains("reshape target has 6 elements but input tensor has 3")
        }),
        "one deferred binding cannot select replacement then insertion:\n{}",
        summary(&errors)
    );
}

#[test]
fn ambiguous_reshape_before_consumer_leaves_deferred_expand_unresolved() {
    let rendered = checked_render(
        r#"
def require_inserted(x: tensor[3, 2, f32]) -> tensor[3, 2, f32] = x
def f(bias: tensor[2, f32], shape_source: tensor[6, f32]) = {
  expanded = expand(bias, 0, 3i64)
  reshaped = reshape(expanded, [shape(shape_source, 0)])
  inserted = require_inserted(expanded)
  reshaped
}
"#,
    );
    assert!(
        rendered.contains("(t-tensor {} (d-name {} *) (t-prim {} f32))"),
        "a reshape with unknown numel must retain its wildcard result before insertion:\n{rendered}"
    );
}

#[test]
fn ambiguous_reshape_after_consumer_preserves_selected_deferred_expand() {
    let rendered = checked_render(
        r#"
def require_inserted(x: tensor[3, 2, f32]) -> tensor[3, 2, f32] = x
def f(bias: tensor[2, f32], shape_source: tensor[6, f32]) = {
  expanded = expand(bias, 0, 3i64)
  inserted = require_inserted(expanded)
  reshaped = reshape(expanded, [shape(shape_source, 0)])
  reshaped
}
"#,
    );
    assert!(
        rendered.contains("(t-tensor {} (d-name {} *) (t-prim {} f32))"),
        "a reshape with unknown numel must retain its wildcard result after insertion:\n{rendered}"
    );
}

#[test]
fn ambiguous_reshape_before_replacement_keeps_wildcard_result() {
    let rendered = checked_render(
        r#"
def require_replaced(x: tensor[3, f32]) -> tensor[3, f32] = x
def f(bias: tensor[2, f32], shape_source: tensor[6, f32]) = {
  expanded = expand(bias, 0, 3i64)
  reshaped = reshape(expanded, [shape(shape_source, 0)])
  replaced = require_replaced(expanded)
  reshaped
}
"#,
    );
    assert!(
        rendered.contains("(t-tensor {} (d-name {} *) (t-prim {} f32))"),
        "a reshape with unknown numel must retain its wildcard result before replacement:\n{rendered}"
    );
}

#[test]
fn ambiguous_reshape_after_replacement_keeps_wildcard_result() {
    let rendered = checked_render(
        r#"
def require_replaced(x: tensor[3, f32]) -> tensor[3, f32] = x
def f(bias: tensor[2, f32], shape_source: tensor[6, f32]) = {
  expanded = expand(bias, 0, 3i64)
  replaced = require_replaced(expanded)
  reshaped = reshape(expanded, [shape(shape_source, 0)])
  reshaped
}
"#,
    );
    assert!(
        rendered.contains("(t-tensor {} (d-name {} *) (t-prim {} f32))"),
        "a reshape with unknown numel must retain its wildcard result after replacement:\n{rendered}"
    );
}

#[test]
fn reshape_rejects_a_concrete_precision_incompatible_with_expand() {
    let errors = typecheck(
        r#"
def require_f64(x: tensor[6, f64]) -> tensor[6, f64] = x
def flatten_expanded(bias: tensor[2, f32]) = {
  expanded = expand(bias, 0, cast(3, int64))
  reshape(expanded, [cast(6, int64)])
}
flat = flatten_expanded(to_tensor([1.0f32, 2.0f32]))
bad = require_f64(flat)
"#,
    );
    assert!(
        errors.iter().any(|error| {
            matches!(error.kind, CheckErrorKind::PrecisionMismatch)
                && error.message.contains("f32")
                && error.message.contains("f64")
        }),
        "a concrete precision mismatch must still fail:\n{}",
        summary(&errors)
    );
}

#[test]
fn elementwise_consumer_selects_common_shape_for_two_deferred_expands() {
    let errors = typecheck(
        r#"
def f(a: tensor[8, 16, f32], b: tensor[16, 4, f32]) -> tensor[8, 4, f32] = {
  ae = expand(a, 2, 4i64)
  be = expand(b, 0, 8i64)
  sum(mul(ae, be), 1)
}
"#,
    );
    assert!(
        errors.is_empty(),
        "an elementwise consumer must select the common insertion shape of two deferred expands:\n{}",
        summary(&errors)
    );
}

#[test]
fn elementwise_consumer_rejects_deferred_expands_without_a_common_shape() {
    let errors = typecheck(
        r#"
def f(a: tensor[8, 16, f32], b: tensor[16, 5, f32]) = {
  ae = expand(a, 2, 4i64)
  be = expand(b, 0, 8i64)
  mul(ae, be)
}
"#,
    );
    assert!(
        errors
            .iter()
            .any(|error| matches!(error.kind, CheckErrorKind::DimensionMismatch)),
        "deferred expands with no common elementwise shape must fail:\n{}",
        summary(&errors)
    );
}

#[test]
fn cast_accepts_trailing_axis_expand_result() {
    let errors = typecheck(
        r#"
t = expand(to_tensor([0.5f32]), 1, 8i64)
c = cast(t, f64)
"#,
    );
    assert!(
        errors.is_empty(),
        "axis equal to the input rank has only the trailing-insertion shape:\n{}",
        summary(&errors)
    );
}

#[test]
fn cast_accepts_scalar_expand_result() {
    let errors = typecheck(
        r#"
t = expand(scalar_to_tensor(0.5f32), 0, 8i64)
c = cast(t, f64)
"#,
    );
    assert!(
        errors.is_empty(),
        "rank-zero expand has only the insertion shape:\n{}",
        summary(&errors)
    );
}

#[test]
fn later_tensor_consumer_rejects_non_expand_result_rank() {
    let errors = typecheck(
        r#"
def require_rank_three(x: tensor[2, 3, 1, f32]) -> tensor[2, 3, 1, f32] = x
bias = to_tensor([0.5f32, 1.5f32, 2.5f32])
expanded = expand(bias, 0, 2i64)
y = require_rank_three(expanded)
"#,
    );
    assert!(
        errors.iter().any(|error| {
            matches!(error.kind, CheckErrorKind::DimensionMismatch)
                && error
                    .message
                    .contains("expand output rank 3 must equal input rank 1 or 2")
        }),
        "a consumer must not bind expand to an unrelated output rank:\n{}",
        summary(&errors)
    );
}

#[test]
fn one_expand_binding_cannot_choose_different_shapes_per_use() {
    let errors = typecheck(
        r#"
def require_rank_one(x: tensor[2, f32]) -> tensor[2, f32] = x
def require_rank_two(x: tensor[2, 3, f32]) -> tensor[2, 3, f32] = x
bias = to_tensor([0.5f32, 1.5f32, 2.5f32])
expanded = expand(bias, 0, 2i64)
replacement = require_rank_one(expanded)
insertion = require_rank_two(expanded)
"#,
    );
    assert!(
        errors
            .iter()
            .any(|error| matches!(error.kind, CheckErrorKind::DimensionMismatch)),
        "one produced value must stay monomorphic across consumers:\n{}",
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
fn declared_rank_increasing_result_selects_expand_insertion() {
    let errors = typecheck(
        r#"
def insert_axis(x: tensor[1, f32]) -> tensor[8, 1, f32] = expand(&x, 0, 8i64)
"#,
    );
    assert!(
        errors.is_empty(),
        "a declared rank-plus-one result must select positional insertion:\n{}",
        summary(&errors)
    );
}

#[test]
fn cast_accepts_uniform_like_result_inferred_from_template() {
    let errors = typecheck(
        r#"
t = expand(to_tensor([0.0f32]), 0, 8i64)
u = with seed(1i64) { uniform_like(t, 0.0, 1.0) }
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
