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
    let deep = desugar_program(&decls).expect("Surf fixture must desugar");
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

#[test]
fn unresolved_callable_forwarded_through_helpers_remains_rootless() {
    let library = surf_to_library(
        r#"
def apply(model: tensor[3, f32] -> f32, x: tensor[3, f32]) -> f32 = model(x)
def apply_pipe(model: tensor[3, f32] -> f32, x: tensor[3, f32]) -> f32 = x |> model

def indirect(model: tensor[3, f32] -> f32, x: tensor[3, f32]) -> tensor[3, f32] = {
  target = fn (v: tensor[3, f32]) -> apply(model, v)
  grad(target, wrt=v)(x)
}

def indirect_pipe(model: tensor[3, f32] -> f32, x: tensor[3, f32]) -> tensor[3, f32] = {
  target = fn (v: tensor[3, f32]) -> apply_pipe(model, v)
  grad(target, wrt=v)(x)
}
"#,
    )
    .expect("forwarding an unresolved callable must remain representable");

    for name in ["indirect", "indirect_pipe"] {
        assert!(
            library.rootless_defs().contains(name),
            "`{name}` depends on an unresolved callable and must remain rootless"
        );
        assert!(
            !library.symbol_table().contains_key(name),
            "`{name}` must not publish a fabricated zero gradient"
        );
    }
}

#[test]
fn unresolved_callable_only_blocks_a_gradient_when_its_result_reaches_the_output() {
    let library = surf_to_library(
        r#"
def sumsq(x: tensor[3, f32]) -> f32 =
  tensor_to_scalar(sum(mul(x, x), 0))

def dead_callable(
  model: tensor[3, f32] -> f32,
  x: tensor[3, f32]
) -> tensor[3, f32] = {
  target = fn (v: tensor[3, f32]) -> {
    dead = model(v)
    cast(1.0, f32)
  }
  grad(target, wrt=v)(x)
}

def live_callable(
  model: tensor[3, f32] -> f32,
  x: tensor[3, f32]
) -> tensor[3, f32] = {
  target = fn (v: tensor[3, f32]) -> model(v)
  grad(target, wrt=v)(x)
}

def dead_specialized(x: tensor[3, f32]) -> tensor[3, f32] =
  dead_callable(sumsq, x)

def live_specialized(x: tensor[3, f32]) -> tensor[3, f32] =
  live_callable(sumsq, x)
"#,
    )
    .expect("dead and live callable controls must remain representable");

    assert!(
        !library.rootless_defs().contains("dead_callable"),
        "an unresolved pure call whose result cannot reach the output cannot block the exact zero cotangent"
    );
    let dead_root = library
        .symbol_table()
        .get("dead_callable")
        .copied()
        .expect("the dead-call gradient must publish its shaped zero");
    let dead_node = library
        .dag()
        .get(dead_root)
        .expect("the dead-call gradient root must identify a live DAG node");
    assert_eq!(dead_node.output_type.dims, vec![DimInfo::Lit(3)]);
    assert_eq!(dead_node.output_type.precision, Prim::F32);
    assert!(library.dag().is_root(dead_root));

    assert!(
        library.rootless_defs().contains("live_callable"),
        "an unresolved call whose result is the output remains unknown until specialization"
    );
    assert!(
        !library.symbol_table().contains_key("live_callable"),
        "the live unresolved call must not publish a fabricated gradient"
    );

    for name in ["dead_specialized", "live_specialized"] {
        assert!(
            !library.rootless_defs().contains(name),
            "a concrete call-site specialization must realize `{name}`"
        );
        let root = library
            .symbol_table()
            .get(name)
            .copied()
            .unwrap_or_else(|| panic!("`{name}` must publish a specialized gradient"));
        let node = library
            .dag()
            .get(root)
            .unwrap_or_else(|| panic!("`{name}` must identify a live DAG node"));
        assert_eq!(node.output_type.dims, vec![DimInfo::Lit(3)]);
        assert_eq!(node.output_type.precision, Prim::F32);
        assert!(library.dag().is_root(root));
    }
}
