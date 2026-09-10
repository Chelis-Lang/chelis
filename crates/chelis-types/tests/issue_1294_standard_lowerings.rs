//! [05-OP-51] and section 4: explicit typed graph parameters.
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
fn standard_axis_recipes_use_insert_and_reject_shape_form_expand() {
    check("def f(x: tensor[2,3,f32]) -> tensor[2,3,f32] = insert(max_reduce(x,1),1,shape(x,1))\n")
        .unwrap();
    assert!(check("def f(x: tensor[2,3,f32]) = expand(x,[2i64,3i64,4i64])\n").is_err());
    assert!(check("def f(x: tensor[2,3,f32]) = expand(x,1,4i64)\n").is_err());
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
