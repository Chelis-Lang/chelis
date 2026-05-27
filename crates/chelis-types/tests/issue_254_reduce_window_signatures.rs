//! Issue #254 — type-checker rules for the four `reduce_window_*`
//! builtins.
//!
//! Pins the spec §2.3.1 shape contract at check time:
//! - Output rank equals input rank.
//! - Trailing windowed-axis extent is
//!   `floor((input - window) / stride) + 1` under Valid padding.
//! - `window_shape` and `strides` must be int32 lists of equal
//!   non-empty length; entries must be positive.
//! - Window arity may not exceed input rank.
//! - `window > input_dim` is rejected as a `DimensionMismatch`.
//!
//! All four reducers share the same shape contract; we exercise
//! `reduce_window_max` for the positive path and check at least one
//! representative reducer per negative case so the four arms stay
//! aligned.

use chelis_types::check_ir_program;

fn check_surf(source: &str) -> Result<Vec<String>, Vec<String>> {
    let decls = chelis_surf::parser::parse_str(source).expect("surf parse");
    let exprs = chelis_macros::expand_program(
        &chelis_surf::desugar::desugar_program(&decls),
        &chelis_macros::ExpansionOptions::default(),
    )
    .expect("macro expand")
    .into_exprs();
    match check_ir_program(&exprs) {
        Ok(_) => Ok(Vec::new()),
        Err(r) => Err(r.errors.iter().map(|e| e.message.clone()).collect()),
    }
}

fn check_reducer(
    reducer_name: &str,
    input_shape: &[usize],
    window: &[usize],
    strides: &[usize],
) -> Result<Vec<String>, Vec<String>> {
    let shape_str = input_shape
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let window_str = window
        .iter()
        .map(|w| w.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let strides_str = strides
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let src = format!(
        r#"
sig run: tensor[{shape_str}, f32] -> tensor[{shape_str}, f32]
def run(x) = {reducer_name}(x, [{window_str}], [{strides_str}])
"#
    );
    check_surf(&src)
}

// -------- Positive coverage --------

/// The issue's acceptance shape: rank-4 [2, 3, 8, 8] input + window
/// [2, 2] + strides [1, 1] produces a rank-4 [2, 3, 7, 7] output.
#[test]
fn issue254_reduce_window_max_acceptance_shape_typechecks() {
    let src = r#"
sig run: tensor[2, 3, 8, 8, f32] -> tensor[2, 3, 7, 7, f32]
def run(x) = reduce_window_max(x, [2, 2], [1, 1])
"#;
    check_surf(src).expect("acceptance shape must typecheck");
}

/// Non-overlapping pool2d (stride == window) on the same canonical
/// rank-4 input: [2, 3, 8, 8] + [2, 2] + [2, 2] → [2, 3, 4, 4].
#[test]
fn issue254_reduce_window_max_nonoverlapping_pool2d_typechecks() {
    let src = r#"
sig run: tensor[2, 3, 8, 8, f32] -> tensor[2, 3, 4, 4, f32]
def run(x) = reduce_window_max(x, [2, 2], [2, 2])
"#;
    check_surf(src).expect("non-overlapping pool2d must typecheck");
}

/// All four reducers share the shape contract.
#[test]
fn issue254_all_four_reducers_typecheck_with_same_shape_contract() {
    for name in [
        "reduce_window_max",
        "reduce_window_min",
        "reduce_window_sum",
        "reduce_window_mean",
    ] {
        let src = format!(
            r#"
sig run: tensor[1, 1, 3, 3, f32] -> tensor[1, 1, 2, 2, f32]
def run(x) = {name}(x, [2, 2], [1, 1])
"#
        );
        check_surf(&src).unwrap_or_else(|errors| panic!("{name} must typecheck, got {errors:?}"));
    }
}

// -------- Negative coverage --------

#[test]
fn issue254_reduce_window_rejects_window_larger_than_input_dim_at_check() {
    let errors = check_reducer("reduce_window_max", &[1, 1, 2, 2], &[3, 3], &[1, 1])
        .expect_err("window > input dim must be a check-time error");
    assert!(
        errors
            .iter()
            .any(|e| e.contains("input dim") || e.contains("window_shape")),
        "expected a window-vs-input-dim diagnostic, got {errors:?}"
    );
}

#[test]
fn issue254_reduce_window_rejects_zero_stride_at_check() {
    let errors = check_reducer("reduce_window_max", &[1, 1, 3, 3], &[2, 2], &[1, 0])
        .expect_err("stride 0 must be a check-time error");
    assert!(
        errors.iter().any(|e| e.contains("strides")),
        "expected a strides positivity diagnostic, got {errors:?}"
    );
}

#[test]
fn issue254_reduce_window_rejects_zero_window_at_check() {
    let errors = check_reducer("reduce_window_max", &[1, 1, 3, 3], &[0, 2], &[1, 1])
        .expect_err("window 0 must be a check-time error");
    assert!(
        errors.iter().any(|e| e.contains("window_shape")),
        "expected a window-shape positivity diagnostic, got {errors:?}"
    );
}

#[test]
fn issue254_reduce_window_rejects_window_arity_exceeding_rank_at_check() {
    let errors = check_reducer("reduce_window_max", &[3, 3], &[2, 2, 2], &[1, 1, 1])
        .expect_err("window arity > rank must be a check-time error");
    assert!(
        errors
            .iter()
            .any(|e| e.contains("window arity") || e.contains("rank")),
        "expected a rank-vs-arity diagnostic, got {errors:?}"
    );
}

#[test]
fn issue254_reduce_window_rejects_window_strides_length_mismatch_at_check() {
    let errors = check_reducer("reduce_window_max", &[1, 1, 3, 3], &[2, 2], &[1, 1, 1])
        .expect_err("window vs strides length mismatch must be a check-time error");
    assert!(
        errors
            .iter()
            .any(|e| e.contains("window_shape") || e.contains("strides")),
        "expected a window/strides-len mismatch diagnostic, got {errors:?}"
    );
}

#[test]
fn issue254_reduce_window_rejects_wrong_declared_output_shape_at_check() {
    // Real output is [1,1,7,7] (input 8 - window 2 + 1 = 7); declared
    // is [1,1,8,8]. Type checker must reject.
    let src = r#"
sig run: tensor[1, 1, 8, 8, f32] -> tensor[1, 1, 8, 8, f32]
def run(x) = reduce_window_max(x, [2, 2], [1, 1])
"#;
    let errors = check_surf(src).expect_err("wrong declared output shape must NOT typecheck");
    assert!(
        !errors.is_empty(),
        "expected a shape-mismatch diagnostic, got an empty error list"
    );
}
