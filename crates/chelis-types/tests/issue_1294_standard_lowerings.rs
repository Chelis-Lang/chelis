//! [05-OP-51] and section 4: explicit typed graph parameters.
//!
//! This suite also owns the executable checker distinction left by the
//! chelis#1277 movement split: `expand` is same-rank and requires an existing
//! unit axis, while `insert` is the rank-raising operation. That contract is
//! behavioral; it does not depend on a hand-maintained inventory of source
//! spellings.

use chelis_types::check_ir_program;

fn check(source: &str) -> Result<(), String> {
    let program = chelis_deep::parse_and_stamp_file(source).expect("Deep fixture");
    check_ir_program(&program, chelis_types::ProgramOrigin::Unlinked)
        .map(|_| ())
        .map_err(|e| format!("{e:?}"))
}

#[test]
fn layer_norm_accepts_explicit_epsilon_at_each_float_width() {
    for dtype in ["f16", "bf16", "f32", "f64"] {
        let source =
            include_str!("fixtures/core/tests/issue_1294_standard_lowerings/dtype_template_0.deep")
                .replace("f32", dtype);
        check(&source).unwrap_or_else(|e| panic!("{source}: {e}"));
    }
}

#[test]
fn layer_norm_rejects_missing_or_mistyped_epsilon_and_invalid_operands() {
    for args in [
        "(var {} x) (var {} g) (var {} b)",
        "(var {} x) (var {} g) (var {} b) (lit {type: (t-prim {} i64)} 1)",
        "(var {} x) (var {} g) (var {} b) (lit {type: (t-prim {} f64)} 1.0)",
        "(var {} x) (var {} g) (var {} b) (lit {type: (t-prim {} bool)} true)",
        "(var {} x) (var {} g) (var {} b) (var {} g)",
        "(var {} x) (var {} g) (var {} b) (lit {type: (t-prim {} f32)} 1.0) (lit {type: (t-prim {} f32)} 1.0)",
    ] {
        let source =
            include_str!("fixtures/core/tests/issue_1294_standard_lowerings/layer_norm_args.deep")
                .replace("@args@", args);
        assert!(check(&source).is_err(), "accepted {source}");
    }
    for dtype in ["i8", "i16", "i32", "i64", "bool"] {
        let source =
            include_str!("fixtures/core/tests/issue_1294_standard_lowerings/dtype_template_8.deep")
                .replace("f32", dtype);
        assert!(check(&source).is_err(), "accepted {source}");
    }
}

#[test]
fn standard_axis_recipes_use_insert_for_rank_raising() {
    check(
        include_str!("fixtures/core/tests/issue_1294_standard_lowerings/standard_axis_recipes_use_insert_for_rank_raising_1.deep"),
    )
    .unwrap();
    check(
        include_str!("fixtures/core/tests/issue_1294_standard_lowerings/standard_axis_recipes_use_insert_for_rank_raising_2.deep"),
    )
    .unwrap();
}

#[test]
fn expand_accepts_same_rank_unit_axis_recipe() {
    check(
        include_str!("fixtures/core/tests/issue_1294_standard_lowerings/expand_accepts_same_rank_unit_axis_recipe_1.deep"),
    )
    .unwrap();
}

#[test]
fn expand_rejects_rank_raising_recipe() {
    let rank_raising = include_str!(
        "fixtures/core/tests/issue_1294_standard_lowerings/expand_rejects_rank_raising_recipe_1.deep"
    );
    let error = check(rank_raising).expect_err("expand must not add an axis");
    assert!(
        error.contains("expand output rank 4 must equal input rank 3")
            && error.contains("Use `insert` to add an axis"),
        "rank-raising expand should fail at its owning checker rule, got {error}"
    );
}

#[test]
fn expand_rejects_nonunit_axis_recipe() {
    let nonunit = include_str!(
        "fixtures/core/tests/issue_1294_standard_lowerings/expand_rejects_nonunit_axis_recipe_1.deep"
    );
    let error = check(nonunit).expect_err("expand must not replace a non-unit axis");
    assert!(
        error.contains("requires the operand's extent at axis 1 to be 1, got 3"),
        "non-unit expand should fail at its owning checker guard, got {error}"
    );
}

#[test]
fn expand_rejects_retired_shape_taking_recipe() {
    let obsolete_shape_form = include_str!(
        "fixtures/core/tests/issue_1294_standard_lowerings/expand_rejects_retired_shape_taking_recipe_1.deep"
    );
    let error = check(obsolete_shape_form).expect_err("retired expand form must be rejected");
    assert!(
        error.contains("kind: ArityMismatch")
            && error.contains(
                "expand expects (tensor, axis, size) or the named-axis form \
                 (tensor, name, size, anchor), got 2 arguments"
            ),
        "the retired shape-taking form must reach expand's owning arity route, got {error}"
    );
}

#[test]
fn attention_recipe_checks_explicit_batch_mask_and_permutation_axes() {
    check(include_str!("fixtures/core/tests/issue_1294_standard_lowerings/attention_recipe_checks_explicit_batch_mask_and_permutation_axes_1.deep")).unwrap();
    assert!(
        check(include_str!(
            "fixtures/core/tests/issue_1294_standard_lowerings/attention_invalid_0.deep"
        ))
        .is_err()
    );
    assert!(
        check(include_str!(
            "fixtures/core/tests/issue_1294_standard_lowerings/attention_invalid_1.deep"
        ))
        .is_err()
    );
    assert!(
        check(include_str!(
            "fixtures/core/tests/issue_1294_standard_lowerings/attention_invalid_2.deep"
        ))
        .is_err()
    );
}

#[test]
fn hosted_matmul_rejects_rank_dtype_and_batch_mismatches() {
    for source in [
        include_str!(
            "fixtures/core/tests/issue_1294_standard_lowerings/hosted_matmul_rejects_rank_dtype_and_batch_mismatches_1.deep"
        ),
        include_str!(
            "fixtures/core/tests/issue_1294_standard_lowerings/hosted_matmul_rejects_rank_dtype_and_batch_mismatches_2.deep"
        ),
        include_str!(
            "fixtures/core/tests/issue_1294_standard_lowerings/hosted_matmul_rejects_rank_dtype_and_batch_mismatches_3.deep"
        ),
        include_str!(
            "fixtures/core/tests/issue_1294_standard_lowerings/hosted_matmul_rejects_rank_dtype_and_batch_mismatches_4.deep"
        ),
        include_str!(
            "fixtures/core/tests/issue_1294_standard_lowerings/hosted_matmul_rejects_rank_dtype_and_batch_mismatches_5.deep"
        ),
    ] {
        assert!(check(source).is_err(), "{source}");
    }
}
