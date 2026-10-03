//! [05-OP-51] and section 4: explicit typed graph parameters.
//!
//! This suite also owns the executable checker distinction left by the
//! chelis#1277 movement split: `expand` is same-rank and requires an existing
//! unit axis, while `insert` is the rank-raising operation. That contract is
//! behavioral; it does not depend on a hand-maintained inventory of source
//! spellings.
use chelis_surf::{desugar::desugar_program, parser::parse_str};
use chelis_types::{InferResult, check_ir_program};

fn check(source: &str) -> Result<(), InferResult> {
    let parsed = parse_str(source).expect("valid Surf fixture");
    let expanded = chelis_macros::expand_program(
        &desugar_program(&parsed).expect("Surf fixture must desugar"),
        &chelis_macros::ExpansionOptions::default(),
    )
    .expect("valid macro expansion")
    .into_exprs();
    check_ir_program(&expanded).map(|_| ())
}

#[test]
fn layer_norm_accepts_explicit_epsilon_at_each_float_width() {
    for dtype in ["f16", "bf16", "f32", "f64"] {
        let source = format!(
            "def f(x: tensor[2,2,{dtype}], g: tensor[2,{dtype}], b: tensor[2,{dtype}], e: {dtype}) -> tensor[2,2,{dtype}] = layer_norm(x,g,b,e)\n"
        );
        check(&source).unwrap_or_else(|e| panic!("{source}: {e:?}"));
    }
}

#[test]
fn layer_norm_rejects_missing_or_mistyped_epsilon_and_invalid_operands() {
    for args in [
        "x,g,b",
        "x,g,b,1i64",
        "x,g,b,1.0f64",
        "x,g,b,true",
        "x,g,b,g",
        "x,g,b,1.0f32,1.0f32",
    ] {
        let source = format!(
            "def f(x: tensor[2,2,f32], g: tensor[2,f32], b: tensor[2,f32]) = layer_norm({args})\n"
        );
        assert!(check(&source).is_err(), "accepted {source}");
    }
    for dtype in ["i8", "i16", "i32", "i64", "bool"] {
        let source = format!(
            "def f(x: tensor[2,2,{dtype}], g: tensor[2,{dtype}], b: tensor[2,{dtype}], e: {dtype}) = layer_norm(x,g,b,e)\n"
        );
        assert!(check(&source).is_err(), "accepted {source}");
    }
}

#[test]
fn standard_axis_recipes_use_insert_for_rank_raising() {
    check(
        "def f(x: tensor[2,3,f32]) -> tensor[2,3,f32] = \
         insert(max_reduce(x,1i32),1i32,shape(x,1i32))\n",
    )
    .unwrap();
    check(
        "def f(x: tensor[1,2,3,f32]) -> tensor[4,1,2,3,f32] = \
         insert(x,0i32,4i64)\n",
    )
    .unwrap();
}

#[test]
fn expand_accepts_same_rank_unit_axis_recipe() {
    check(
        "def f(x: tensor[1,2,3,f32]) -> tensor[4,2,3,f32] = \
         expand(x,0i32,4i64)\n",
    )
    .unwrap();
}

#[test]
fn expand_rejects_rank_raising_recipe() {
    let rank_raising = "def f(x: tensor[1,2,3,f32]) -> tensor[4,1,2,3,f32] = \
         expand(x,0i32,4i64)\n";
    let error = check(rank_raising).expect_err("expand must not add an axis");
    assert!(
        error.errors.iter().any(|diagnostic| {
            diagnostic.kind.diagnostic_name() == "DimensionMismatch"
                && diagnostic.message.contains("expand")
                && diagnostic.expected.as_deref() == Some("3")
                && diagnostic.got.as_deref() == Some("4")
                && diagnostic.span_offset == rank_raising.find("expand(")
        }),
        "rank-raising expand should fail at its owning checker rule, got {error:?}"
    );
}

#[test]
fn expand_rejects_nonunit_axis_recipe() {
    let nonunit = "def f(x: tensor[2,3,f32]) -> tensor[2,4,f32] = expand(x,1i32,4i64)\n";
    let error = check(nonunit).expect_err("expand must not replace a non-unit axis");
    assert!(
        error.errors.iter().any(|diagnostic| {
            diagnostic.kind.diagnostic_name() == "DimensionMismatch"
                && diagnostic.message.contains("expand")
                && diagnostic.message.contains("axis 1")
                && diagnostic.expected.as_deref() == Some("1")
                && diagnostic.got.as_deref() == Some("3")
                && diagnostic.span_offset == nonunit.find("expand(")
        }),
        "non-unit expand should fail at its owning checker guard, got {error:?}"
    );
}

#[test]
fn expand_rejects_retired_shape_taking_recipe() {
    let obsolete_shape_form = "def f(x: tensor[2,3,f32]) = expand(x,[2i64,3i64,4i64])\n";
    let error = check(obsolete_shape_form).expect_err("retired expand form must be rejected");
    assert!(
        error.errors.iter().any(|diagnostic| {
            diagnostic.kind.diagnostic_name() == "ArityMismatch"
                && diagnostic.message.contains("expand")
                && diagnostic.expected.as_deref() == Some("3 arguments")
                && diagnostic.got.as_deref() == Some("2 arguments")
                && diagnostic.span_offset == obsolete_shape_form.find("expand(")
        }),
        "the retired shape-taking form must reach expand's owning arity route, got {error:?}"
    );
}

#[test]
fn expand_and_insert_reject_only_their_own_invalid_arities() {
    let four_arg_expand = "def f(x: tensor[1,2,f32]) = expand(x,0i32,4i64,1i32)\n";
    let error = check(four_arg_expand).expect_err("expand cannot take an anchor");
    assert!(
        error.errors.iter().any(|diagnostic| {
            diagnostic.kind.diagnostic_name() == "ArityMismatch"
                && diagnostic.expected.as_deref() == Some("3 arguments")
                && diagnostic.got.as_deref() == Some("4 arguments")
                && diagnostic.span_offset == four_arg_expand.find("expand(")
        }),
        "four-argument expand must reject at its call with its exact arity: {error:?}"
    );

    let short_insert = "def f(x: tensor[1,2,f32]) = insert(x,0i32)\n";
    let error = check(short_insert).expect_err("insert needs a size");
    assert!(
        error.errors.iter().any(|diagnostic| {
            diagnostic.kind.diagnostic_name() == "ArityMismatch"
                && diagnostic.expected.as_deref() == Some("3 or 4 arguments")
                && diagnostic.got.as_deref() == Some("2 arguments")
                && diagnostic.span_offset == short_insert.find("insert(")
        }),
        "insert must report both admitted arities: {error:?}"
    );
}

#[test]
fn attention_recipe_checks_explicit_batch_mask_and_permutation_axes() {
    let source = r#"def attention(q: tensor[1,2,2,1,f32], k: tensor[1,2,3,1,f32], v: tensor[1,2,3,2,f32], mask: tensor[1,2,2,3,bool], scale: f32) -> tensor[1,2,2,2,f32] = matmul(softmax(where(mask,mul(matmul(q,permute(k,0,1,3,2)),insert(insert(insert(insert(scalar_to_tensor(scale),0,1i64),1,2i64),2,2i64),3,3i64)),insert(insert(insert(insert(scalar_to_tensor(div(-1.0f32,0.0f32)),0,1i64),1,2i64),2,2i64),3,3i64)),3),v)
"#;
    check(source).unwrap();
    assert!(
        check(&source.replace("mask: tensor[1,2,2,3,bool]", "mask: tensor[1,1,2,3,bool]")).is_err()
    );
    assert!(check(&source.replace("permute(k,0,1,3,2)", "permute(k,[0,1,3,2])")).is_err());
    assert!(check(&source.replace("permute(k,0,1,3,2)", "permute(k,(0,1,3,2))")).is_err());
}

#[test]
fn hosted_matmul_rejects_rank_dtype_and_batch_mismatches() {
    for source in [
        "def f(a: tensor[2,f32], b: tensor[2,1,f32]) = matmul(a,b)\n",
        "def f(a: tensor[1,2,f32], b: tensor[2,1,f64]) = matmul(a,b)\n",
        "def f(a: tensor[1,2,i32], b: tensor[2,1,i32]) = matmul(a,b)\n",
        "def f(a: tensor[2,1,2,f32], b: tensor[3,2,1,f32]) = matmul(a,b)\n",
        "def f(a: tensor[1,2,f32], b: tensor[3,1,f32]) = matmul(a,b)\n",
    ] {
        assert!(check(source).is_err(), "{source}");
    }
}
