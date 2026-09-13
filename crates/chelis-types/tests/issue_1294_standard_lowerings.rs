//! [05-OP-51] and section 4: explicit typed graph parameters.
//!
//! This suite also owns the executable checker distinction left by the
//! chelis#1277 movement split: `expand` is same-rank and requires an existing
//! unit axis, while `insert` is the rank-raising operation. That contract is
//! behavioral; it does not depend on a hand-maintained inventory of source
//! spellings.
use chelis_surf::{desugar::desugar_program, parser::parse_str};
use chelis_types::check_ir_program;

fn check(source: &str) -> Result<(), String> {
    let parsed = parse_str(source).expect("valid Surf fixture");
    let expanded = chelis_macros::expand_program(
        &desugar_program(&parsed),
        &chelis_macros::ExpansionOptions::default(),
    )
    .expect("valid macro expansion")
    .into_exprs();
    check_ir_program(&expanded)
        .map(|_| ())
        .map_err(|e| format!("{e:?}"))
}

#[test]
fn layer_norm_accepts_explicit_epsilon_at_each_float_width() {
    for dtype in ["f16", "bf16", "f32", "f64"] {
        let source = format!(
            "def f(x: tensor[2,2,{dtype}], g: tensor[2,{dtype}], b: tensor[2,{dtype}], e: {dtype}) -> tensor[2,2,{dtype}] = layer_norm(x,g,b,e)\n"
        );
        check(&source).unwrap_or_else(|e| panic!("{source}: {e}"));
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
    for dtype in ["int8", "int16", "int32", "int64", "bool"] {
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
fn expand_rejects_rank_raising_recipe() {
    let rank_raising = "def f(x: tensor[1,2,3,f32]) -> tensor[4,1,2,3,f32] = \
         expand(x,0i32,4i64)\n";
    let error = check(rank_raising).expect_err("expand must not add an axis");
    assert!(
        error.contains("expand output rank 4 must equal input rank 3")
            && error.contains("Use `insert` to add an axis"),
        "rank-raising expand should fail at its owning checker rule, got {error}"
    );
}

#[test]
fn expand_rejects_nonunit_axis_recipe() {
    let nonunit = "def f(x: tensor[2,3,f32]) -> tensor[2,4,f32] = expand(x,1i32,4i64)\n";
    let error = check(nonunit).expect_err("expand must not replace a non-unit axis");
    assert!(
        error.contains("requires the operand's extent at axis 1 to be 1, got 3"),
        "non-unit expand should fail at its owning checker guard, got {error}"
    );
}

#[test]
fn expand_rejects_retired_shape_taking_recipe() {
    let obsolete_shape_form = "def f(x: tensor[2,3,f32]) = expand(x,[2i64,3i64,4i64])\n";
    assert!(
        check(obsolete_shape_form).is_err(),
        "the retired shape-taking expand recipe must stay rejected"
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
        "def f(a: tensor[1,2,int32], b: tensor[2,1,int32]) = matmul(a,b)\n",
        "def f(a: tensor[2,1,2,f32], b: tensor[3,2,1,f32]) = matmul(a,b)\n",
        "def f(a: tensor[1,2,f32], b: tensor[3,1,f32]) = matmul(a,b)\n",
    ] {
        assert!(check(source).is_err(), "{source}");
    }
}
