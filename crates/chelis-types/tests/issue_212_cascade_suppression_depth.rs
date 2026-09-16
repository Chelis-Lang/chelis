//! Issue #212's cascade marker existed only to suppress repeated instances of
//! the old checker-level `conv` concrete-metadata refusal. PP9 relocates that
//! backend capability restriction to #730, so symbolic conv chains are legal
//! checker input and need no failed-derivation side channel.

use chelis_deep::Expr;
use chelis_surf::{desugar::desugar_program, parser::parse_str as parse_surf};
use chelis_types::{check_ir_program, check_typed_program};

fn surf_to_deep(source: &str) -> Vec<Expr> {
    let declarations = parse_surf(source).expect("Surf fixture");
    chelis_macros::expand_program(
        &desugar_program(&declarations),
        &chelis_macros::ExpansionOptions::default(),
    )
    .expect("macro expansion")
    .into_exprs()
}

#[test]
fn symbolic_conv_chain_is_accepted_without_cascade_state() {
    let source = r#"
def f(
  x: tensor[1, 3, h, 16, f32],
  k1: tensor[8, 3, 3, 3, f32],
  k2: tensor[16, 8, 3, 3, f32],
  k3: tensor[32, 16, 3, 3, f32]
) -> tensor[1, 32, out_h, out_w, f32] = {
  y1 = conv(&x, &k1, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)])
  y2 = relu(conv(&y1, &k2, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)]))
  conv(&y2, &k3, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)])
}
"#;
    let expressions = surf_to_deep(source);
    check_ir_program(&expressions).expect("IR checker accepts symbolic conv metadata");
    check_typed_program(&expressions).expect("typed checker accepts symbolic conv metadata");
}
