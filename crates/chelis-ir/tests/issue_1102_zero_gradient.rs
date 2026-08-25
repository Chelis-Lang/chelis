//! Issue #1102: a missing adjoint for a completely lowered constant
//! function is a proven zero cotangent, not a missing result root.
//!
//! The negative control is chelis#1095's higher-order shape. Its standalone
//! definition contains an unresolved function-valued parameter, so its
//! gradient is unknown until call-site specialization and must remain
//! rootless rather than being fabricated as zero.

use chelis_ir::dag::DimInfo;
use chelis_ir::lower::{LoweredLibrary, try_lower_program_to_library};
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as surf_parse;
use chelis_types::check_typed_program;
use chelis_types::types::Prim;

fn surf_to_library(source: &str) -> Result<LoweredLibrary, String> {
    let decls = surf_parse(source).map_err(|error| format!("surf parse: {error:?}"))?;
    let deep = desugar_program(&decls);
    let checked = check_typed_program(&deep)
        .map_err(|errors| format!("typecheck failed: {:?}", errors.errors))?;
    let checked = chelis_effects::check_program(&checked)
        .map_err(|errors| format!("effects failed: {errors:?}"))?;
    let checked = chelis_types::check_linearity(&checked)
        .map_err(|errors| format!("linearity failed: {errors:?}"))?;
    try_lower_program_to_library(&checked).map_err(|error| format!("lowering failed: {error:?}"))
}

#[test]
fn proven_shaped_zero_gradient_retains_a_typed_root() {
    let library = surf_to_library(
        r#"
def constish(x: tensor[3, f32]) -> f32 = cast(1.0, f32)
def g(theta: tensor[3, f32]) -> tensor[3, f32] = grad(constish, wrt=theta)(theta)
"#,
    )
    .expect("a proven zero gradient must lower");

    assert!(
        !library.rootless_defs().contains("g"),
        "a complete constant function has an exact zero cotangent, so `g` must own a root"
    );
    let root = library
        .symbol_table()
        .get("g")
        .copied()
        .expect("the shaped zero gradient must survive in the symbol table");
    let node = library
        .dag()
        .get(root)
        .expect("the gradient symbol must identify a live DAG node");
    assert_eq!(node.output_type.dims, vec![DimInfo::Lit(3)]);
    assert_eq!(node.output_type.precision, Prim::F32);
    assert!(
        library.dag().is_root(root),
        "the shaped zero must be a realized DAG root"
    );
}

#[test]
fn proven_rank_zero_gradient_retains_an_f32_root() {
    let library = surf_to_library(
        r#"
def const_rank_zero(x: tensor[f32]) -> f32 = cast(1.0, f32)
def rank_zero_grad(x: tensor[f32]) -> tensor[f32] = grad(const_rank_zero, wrt=x)(x)
"#,
    )
    .expect("a proven rank-zero gradient must lower");

    assert!(!library.rootless_defs().contains("rank_zero_grad"));
    let root = library.symbol_table()["rank_zero_grad"];
    let node = library.dag().get(root).expect("rank-zero gradient node");
    assert!(node.output_type.dims.is_empty());
    assert_eq!(node.output_type.precision, Prim::F32);
    assert!(library.dag().is_root(root));
}

#[test]
fn unresolved_callable_gradient_remains_rootless_until_specialized() {
    let library = surf_to_library(
        r#"
def sumsq(theta: tensor[3, f32]) -> f32 =
  tensor_to_scalar(sum(mul(theta, theta), 0))

def grad_sumsq(
  model: tensor[3, f32] -> f32,
  theta: tensor[3, f32]
) -> tensor[3, f32] = {
  target = fn (theta_local: tensor[3, f32]) -> model(theta_local)
  grad(target, wrt=theta_local)(theta)
}
"#,
    )
    .expect("the standalone higher-order definition must remain representable");

    assert!(
        library.rootless_defs().contains("grad_sumsq"),
        "an unresolved callable dependency is unknown, not a proven zero"
    );
    assert!(
        !library.symbol_table().contains_key("grad_sumsq"),
        "the unknown standalone gradient must not publish a fabricated value"
    );
}
