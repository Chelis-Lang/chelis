//! [04-INF-6]: an authored binder has one meaning in ordinary and precision positions.

use chelis_types::{check_ir_program, check_typed_program};

fn diagnostics(source: &str) -> Vec<String> {
    let surf = chelis_surf::parser::parse_str(source).expect("Surf parse");
    let deep = chelis_macros::expand_program(
        &chelis_surf::desugar::desugar_program(&surf).expect("Surf desugar"),
        &chelis_macros::ExpansionOptions::default(),
    )
    .expect("expand")
    .into_exprs();
    let render = |errors: Vec<chelis_types::errors::CheckError>| {
        errors.into_iter().map(|e| e.message).collect::<Vec<_>>()
    };
    let ir = render(check_ir_program(&deep).err().map_or(vec![], |r| r.errors));
    let typed = render(
        check_typed_program(&deep)
            .err()
            .map_or(vec![], |r| r.errors),
    );
    assert_eq!(ir, typed, "checker entry disagreement");
    ir
}

#[test]
fn mixed_binder_positions_reject_tensor_in_scalar_position() {
    let errors = diagnostics(
        "def f[p](x: p, t: tensor[3, p]) -> p = x\n\
         t = to_tensor([1.0f32, 2.0f32, 3.0f32])\n\
         out = scalar_to_tensor(f(t, t))\n",
    );
    assert!(
        errors
            .iter()
            .any(|e| e.contains("tensor precision must be a primitive")),
        "the ill-typed call must reject at precision unification: {errors:?}"
    );
}

#[test]
fn mixed_binder_positions_accept_compatible_scalar_and_tensor() {
    let errors = diagnostics(
        "def f[p](x: p, t: tensor[3, p]) -> p = x\n\
         t = to_tensor([1.0f32, 2.0f32, 3.0f32])\n\
         out = f(7.0f32, t)\n",
    );
    assert!(
        errors.is_empty(),
        "valid precision polymorphism: {errors:?}"
    );
}
