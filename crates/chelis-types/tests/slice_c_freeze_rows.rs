//! Slice C c2: the `Freeze` rows of `spec/04-type-system.md` §4.7.2.
//!
//! A freeze point is reached when a concrete tensor type is required and no
//! independent constraint has selected a candidate. §4.7.2 fixes what is
//! selected there: an axis within the input rank selects the same-rank
//! replacement form, `axis == rank(x)` selects trailing insertion, and
//! coupled results settle in source order so the verdict is a property of
//! the program text.
//!
//! These rows are a disposition lock, not a regression test. The behavior
//! they assert is already the behavior of `main` at `a5137d2c3`; they exist
//! so the `Constrain` half of the protocol cannot move a `Freeze` verdict
//! without a test saying so.

use chelis_deep::printer::print_canonical;
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as parse_surf;
use chelis_types::check_typed_program;
use chelis_types::errors::{CheckError, CheckErrorKind};

fn checked_render(source: &str) -> String {
    let decls = parse_surf(source).expect("surf parse should succeed");
    let deep = desugar_program(&decls);
    let checked = check_typed_program(&deep).expect("program should typecheck");
    print_canonical(checked.annotated_exprs())
}

fn typecheck(source: &str) -> Vec<CheckError> {
    let decls = parse_surf(source).expect("surf parse should succeed");
    let deep = desugar_program(&decls);
    match check_typed_program(&deep) {
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

const REPLACEMENT_F32: &str = "(t-tensor {} (d-lit {} 8) (t-prim {} f32))";
const INSERTION_F32: &str = "(t-tensor {} (d-lit {} 1) (d-lit {} 8) (t-prim {} f32))";

#[test]
fn cast_freezes_an_axis_within_the_input_rank_to_same_rank_replacement() {
    let rendered = checked_render(
        r#"
t = expand(to_tensor([0.5f32]), 0, 8i64)
c = cast(t, f64)
"#,
    );
    assert!(
        rendered.contains(REPLACEMENT_F32),
        "cast is shape neutral, so axis 0 of a rank-1 input must freeze to \
         same-rank replacement:\n{rendered}"
    );
    assert!(
        !rendered.contains(INSERTION_F32),
        "the insertion candidate must not be selected at a cast freeze \
         point:\n{rendered}"
    );
}

#[test]
fn cast_freezes_an_axis_equal_to_the_input_rank_to_trailing_insertion() {
    let rendered = checked_render(
        r#"
t = expand(to_tensor([0.5f32]), 1, 8i64)
c = cast(t, f64)
"#,
    );
    assert!(
        rendered.contains(INSERTION_F32),
        "axis equal to the input rank has no replacement form and must \
         freeze to trailing insertion:\n{rendered}"
    );
}

#[test]
fn shape_freezes_an_axis_within_the_input_rank_to_same_rank_replacement() {
    let rendered = checked_render(
        r#"
t = expand(to_tensor([0.5f32]), 0, 8i64)
n = shape(t, 0)
"#,
    );
    assert!(
        rendered.contains(REPLACEMENT_F32),
        "a shape read requires the tensor type and must freeze to same-rank \
         replacement for an axis within the input rank:\n{rendered}"
    );
    assert!(
        !rendered.contains(INSERTION_F32),
        "a shape read must not select the insertion candidate:\n{rendered}"
    );
}

#[test]
fn shape_freezes_an_axis_equal_to_the_input_rank_to_trailing_insertion() {
    let rendered = checked_render(
        r#"
t = expand(to_tensor([0.5f32]), 1, 8i64)
n = shape(t, 1)
"#,
    );
    assert!(
        rendered.contains(INSERTION_F32),
        "a shape read of a trailing-axis expand must freeze to insertion, \
         which makes axis 1 legal:\n{rendered}"
    );
}

#[test]
fn shape_rejects_an_axis_outside_the_frozen_form() {
    let errors = typecheck(
        r#"
t = expand(to_tensor([0.5f32]), 0, 8i64)
n = shape(t, 1)
"#,
    );
    assert!(
        errors
            .iter()
            .any(|error| matches!(error.kind, CheckErrorKind::DimensionMismatch)),
        "freezing to the rank-1 replacement form makes axis 1 a type error \
         rather than a reason to prefer the insertion candidate:\n{}",
        summary(&errors)
    );
}

#[test]
fn program_freeze_point_selects_replacement_for_an_axis_within_the_input_rank() {
    let rendered = checked_render("t = expand(to_tensor([0.5f32]), 0, 8i64)");
    assert!(
        rendered.contains(REPLACEMENT_F32),
        "an otherwise-unconsumed result freezes to same-rank \
         replacement:\n{rendered}"
    );
    assert!(
        !rendered.contains(INSERTION_F32),
        "the program freeze point must not select insertion for an axis \
         within the input rank:\n{rendered}"
    );
}

#[test]
fn program_freeze_point_selects_insertion_for_an_axis_equal_to_the_input_rank() {
    let rendered = checked_render("t = expand(to_tensor([0.5f32]), 1, 8i64)");
    assert!(
        rendered.contains(INSERTION_F32),
        "an otherwise-unconsumed trailing-axis result freezes to \
         insertion:\n{rendered}"
    );
}

#[test]
fn one_result_carrying_two_obligations_settles_to_the_first_shape_satisfying_both() {
    let rendered = checked_render(
        r#"
def pick[a](x: a, y: a) -> a = x
t = pick(expand(to_tensor([0.5f32]), 0, 3i64), expand(to_tensor([1.0f32, 2.0f32]), 0, 3i64))
"#,
    );
    assert!(
        rendered.contains("(t-tensor {} (d-lit {} 3) (t-prim {} f32))"),
        "unification identified the two results, so the earliest obligation's \
         first candidate that satisfies both is selected:\n{rendered}"
    );
    assert!(
        !rendered.contains("(d-lit {} 3) (d-lit {} 1)"),
        "the insertion candidate of the earlier obligation does not satisfy \
         the later one:\n{rendered}"
    );
}

#[test]
fn coupled_results_settle_to_one_verdict_under_operand_order() {
    let original = checked_render(
        "original = sub(reshape(expand(to_tensor([0.0f32]), 0, 6i64), [3i64, 2i64]), \
         expand(to_tensor([1.0f32, 2.0f32]), 0, 3i64))",
    );
    let mirrored = checked_render(
        "mirrored = sub(expand(to_tensor([1.0f32, 2.0f32]), 0, 3i64), \
         reshape(expand(to_tensor([0.0f32]), 0, 6i64), [3i64, 2i64]))",
    );
    for (name, rendered) in [("original", &original), ("mirrored", &mirrored)] {
        assert!(
            rendered.contains("(t-tensor {} (d-lit {} 3) (d-lit {} 2) (t-prim {} f32))"),
            "{name}: the reshape target fixes the sub operands at rank 2:\n{rendered}"
        );
        assert!(
            rendered.contains("(t-tensor {} (d-lit {} 6) (t-prim {} f32))"),
            "{name}: the reshape input's own choice stays ambiguous under the \
             element count and freezes to same-rank replacement:\n{rendered}"
        );
    }
}
