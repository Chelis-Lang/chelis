//! Spec-derived acceptance tests for the opt-in trace, not an AD certificate.
#![cfg(feature = "lowering-trace")]

use chelis_ir::dag::{Dag, RiscOp};
use chelis_ir::lower::try_lower_program_to_library;
use chelis_ir::lowering_trace::{
    BoundaryKind, ContextId, ContextKind, try_lower_program_to_library_with_trace,
};
use chelis_surf::{desugar::desugar_program, parser::parse_str};
use chelis_types::{CheckedProgram, check_typed_program};

const SQUARE: &str = r#"
def loss(x: tensor[3, f32]) -> f32 = tensor_to_scalar(sum(mul(x, x), 0))
def derivative(x: tensor[3, f32]) -> tensor[3, f32] = grad(loss)(x)
"#;

fn checked(source: &str) -> CheckedProgram {
    let deep = desugar_program(&parse_str(source).expect("parse fixture"));
    let program = check_typed_program(&deep).expect("check fixture types");
    let program = chelis_effects::check_program(&program).expect("check fixture effects");
    chelis_types::check_linearity(&program).expect("check fixture linearity")
}

// The existing DAG codec compares every stored bit and metadata field. This
// helper is not a new trace wire format and does not serialize LoweringTrace.
fn same_dag(left: &Dag, right: &Dag) {
    assert_eq!(
        bincode::serialize(left).unwrap(),
        bincode::serialize(right).unwrap()
    );
}

#[test]
fn trace_preserves_the_ordinary_library_and_actual_normalization() {
    let program = checked(&format!(
        r#"{SQUARE}
def pair(x: tensor[3, f32]) -> (tensor[3, f32], tensor[3, f32]) = (x, x)
def dead(x: tensor[3, f32]) -> tensor[3, f32] = {{
  unused = mul(x, x)
  x
}}
"#
    ));
    let ordinary = try_lower_program_to_library(&program).unwrap();
    let (observed, trace) = try_lower_program_to_library_with_trace(&program).unwrap();
    assert_eq!(
        bincode::serialize(&ordinary).unwrap(),
        bincode::serialize(&observed).unwrap()
    );
    let normalization = trace.normalization;
    same_dag(&normalization.after_drops, observed.dag());
    // These are genuinely different stages, not four copies of the result.
    assert!(
        normalization
            .after_copies
            .nodes()
            .iter()
            .any(|node| node.op == RiscOp::Copy)
    );
    assert!(
        !normalization
            .after_copies
            .nodes()
            .iter()
            .any(|node| node.op == RiscOp::Drop)
    );
    assert!(
        normalization
            .after_drops
            .nodes()
            .iter()
            .any(|node| node.op == RiscOp::Drop)
    );
    assert!(normalization.after_dce.len() < normalization.before_dce.len());
    assert_eq!(
        normalization.copy_remap.len(),
        normalization.after_dce.len()
    );
    for (old, new) in &normalization.dce_remap {
        let before = normalization.before_dce.get(*old).unwrap();
        let after = normalization.after_dce.get(*new).unwrap();
        assert_eq!(before.op, after.op);
        assert_eq!(before.output_type, after.output_type);
    }
}

#[test]
fn trace_captures_actual_ad_ports_roots_and_ordered_wrt() {
    let (_, trace) = try_lower_program_to_library_with_trace(&checked(SQUARE)).unwrap();
    assert_eq!(trace.gradients.len(), 1);
    let grad = &trace.gradients[0];
    assert_eq!(trace.contexts[grad.context.0].kind, ContextKind::Gradient);
    assert_eq!(trace.contexts[grad.context.0].parent, Some(ContextId(0)));
    assert_eq!(grad.wrt.len(), 1);
    assert!(
        matches!(&grad.forward.get(grad.wrt[0]).unwrap().op, RiscOp::Load { name } if name.as_str() == "x")
    );
    let mul = grad
        .forward
        .nodes()
        .iter()
        .find(|node| node.op == RiscOp::Mul)
        .unwrap();
    assert_eq!(mul.inputs, vec![grad.wrt[0], grad.wrt[0]]);
    assert_eq!(grad.forward.roots(), &[grad.output]);
    assert!(grad.backward.is_root(grad.backward_output));
    assert!(grad.backward.is_root(grad.gradients[&grad.wrt[0]]));
    assert!(grad.backward.len() > grad.forward.len());
    assert!(trace.boundaries.is_empty());
}

#[test]
fn empty_and_repeated_invocations_do_not_leak_capture_state() {
    for source in [SQUARE, "", SQUARE, ""] {
        let program = checked(source);
        let (library, trace) = try_lower_program_to_library_with_trace(&program).unwrap();
        let ordinary = try_lower_program_to_library(&program).unwrap();
        same_dag(library.dag(), ordinary.dag());
        assert_eq!(trace.contexts[0].kind, ContextKind::Library);
        assert_eq!(trace.contexts[0].parent, None);
        assert_eq!(trace.gradients.len(), usize::from(!source.is_empty()));
        assert_eq!(trace.contexts.len(), if source.is_empty() { 1 } else { 2 });
    }
}

#[test]
fn rejected_ad_keeps_its_diagnostic_instead_of_returning_a_trace() {
    let source = r#"
def loss(x: tensor[3, f32]) -> f32 = tensor_to_scalar(sum(floor(x), 0))
def derivative(x: tensor[3, f32]) -> tensor[3, f32] = grad(loss)(x)
"#;
    let program = checked(source);
    let ordinary = try_lower_program_to_library(&program).unwrap_err();
    let observed = try_lower_program_to_library_with_trace(&program).unwrap_err();
    assert_eq!(ordinary, observed);
    assert!(ordinary.to_string().contains("floor"));
    let (_, next) = try_lower_program_to_library_with_trace(&checked(SQUARE)).unwrap();
    assert_eq!(next.gradients.len(), 1);
    assert_eq!(next.contexts.len(), 2);
}

#[test]
fn multiple_wrt_inputs_keep_their_forward_order_and_distinct_results() {
    let source = r#"
def loss(x: tensor[3, f32], y: tensor[3, f32]) -> f32 = tensor_to_scalar(sum(mul(x, y), 0))
def derivative(x: tensor[3, f32], y: tensor[3, f32]) -> (tensor[3, f32], tensor[3, f32]) = grad(loss)(x, y)
"#;
    let (_, trace) = try_lower_program_to_library_with_trace(&checked(source)).unwrap();
    let grad = &trace.gradients[0];
    assert_eq!(grad.wrt.len(), 2);
    for (id, expected_name) in grad.wrt.iter().zip(["x", "y"]) {
        assert!(
            matches!(&grad.forward.get(*id).unwrap().op, RiscOp::Load { name } if name.as_str() == expected_name)
        );
        assert!(grad.backward.is_root(grad.gradients[id]));
    }
    assert_ne!(grad.gradients[&grad.wrt[0]], grad.gradients[&grad.wrt[1]]);
}

#[test]
fn nested_gradients_have_distinct_contexts_and_actual_parent_links() {
    let source = format!(
        r#"{SQUARE}
def first(x: tensor[3, f32]) -> f32 = tensor_to_scalar(sum(grad(loss)(x), 0))
def second(x: tensor[3, f32]) -> tensor[3, f32] = grad(first)(x)
"#
    );
    let (_, trace) = try_lower_program_to_library_with_trace(&checked(&source)).unwrap();
    assert_eq!(trace.gradients.len(), 4);
    let nested = trace
        .gradients
        .iter()
        .find(|grad| trace.contexts[grad.context.0].parent != Some(ContextId(0)))
        .unwrap();
    let parent = trace.contexts[nested.context.0].parent.unwrap();
    assert_ne!(parent, nested.context);
    assert!(trace.gradients.iter().any(|grad| grad.context == parent));
    assert_eq!(trace.contexts[parent.0].parent, Some(ContextId(0)));
}

#[test]
fn shaped_zero_materialization_is_not_fabricated_in_the_raw_ad_result() {
    let source = r#"
def constant(x: tensor[3, f32]) -> f32 = 1.0f32
def derivative(x: tensor[3, f32]) -> tensor[3, f32] = grad(constant)(x)
"#;
    let (library, trace) = try_lower_program_to_library_with_trace(&checked(source)).unwrap();
    let grad = &trace.gradients[0];
    assert_eq!(grad.wrt.len(), 1);
    assert!(
        grad.gradients.is_empty(),
        "the AD pass itself does not materialize this zero"
    );
    assert!(!library.rootless_defs().contains("derivative"));
    let root = library.symbol_table()["derivative"];
    assert!(
        library.dag().is_root(root),
        "source packing subsequently materializes it"
    );
    assert_eq!(
        library.dag().get(root).unwrap().output_type.dims,
        vec![chelis_ir::dag::DimInfo::Lit(3)]
    );
    same_dag(&trace.normalization.after_drops, library.dag());
}

#[test]
fn host_only_definitions_do_not_fabricate_graph_observations() {
    let source = r#"
def filled(x: tensor[n, f32]) -> tensor[n, f32] =
  expand(to_tensor([3.0f32]), 0, cast(shape(&x, 0), int64))
"#;
    let (library, trace) = try_lower_program_to_library_with_trace(&checked(source)).unwrap();
    assert!(library.dag().nodes().is_empty());
    assert!(trace.gradients.is_empty());
    assert_eq!(trace.boundaries.len(), 1);
    assert_eq!(trace.boundaries[0].kind, BoundaryKind::UnloweredDefinitions);
    same_dag(&trace.normalization.after_drops, library.dag());
}

#[test]
fn full_integer_constants_keep_bits_beyond_the_f64_exact_range() {
    let source = "def values() -> tensor[2, int64] = [9007199254740993, -9007199254740993]\n";
    let (library, trace) = try_lower_program_to_library_with_trace(&checked(source)).unwrap();
    for dag in [
        &trace.normalization.before_dce,
        &trace.normalization.after_dce,
        &trace.normalization.after_copies,
        &trace.normalization.after_drops,
        library.dag(),
    ] {
        let payload = dag
            .nodes()
            .iter()
            .find_map(|node| match &node.op {
                RiscOp::ConstTensor { data } => Some(data),
                _ => None,
            })
            .expect("nonuniform complete tensor literal");
        assert_eq!(
            payload.scalar_at(0),
            chelis_types::scalar_from_i64(
                "trace test",
                chelis_types::types::Prim::Int64,
                9_007_199_254_740_993
            )
            .unwrap()
        );
        assert_eq!(
            payload.scalar_at(1),
            chelis_types::scalar_from_i64(
                "trace test",
                chelis_types::types::Prim::Int64,
                -9_007_199_254_740_993
            )
            .unwrap()
        );
    }
}

#[test]
fn unresolved_and_vectorized_gradients_are_explicit_boundaries() {
    let source = r#"
def unknown(model: tensor[3, f32] -> f32, x: tensor[3, f32]) -> tensor[3, f32] = {
  target = fn (v: tensor[3, f32]) -> model(v)
  grad(target, wrt=v)(x)
}
"#;
    let (library, trace) = try_lower_program_to_library_with_trace(&checked(source)).unwrap();
    assert!(library.rootless_defs().contains("unknown"));
    assert!(trace.gradients.is_empty());
    assert_eq!(trace.boundaries.len(), 1);
    assert_eq!(
        trace.boundaries[0].kind,
        BoundaryKind::UnresolvedCallableGradient
    );
    assert_eq!(
        trace.contexts[trace.boundaries[0].context.0].kind,
        ContextKind::Gradient
    );

    for (transform, result_type, boundary, context) in [
        (
            "vmap(loss)",
            "tensor[2, f32]",
            BoundaryKind::Vmap,
            ContextKind::Vmap,
        ),
        (
            "vmap(grad(loss))",
            "tensor[2, 3, f32]",
            BoundaryKind::VmapGradient,
            ContextKind::VmapGradient,
        ),
    ] {
        let source = format!(
            r#"
def loss(x: tensor[3, f32]) -> f32 = tensor_to_scalar(sum(mul(x, x), 0))
def mapped(x: tensor[2, 3, f32]) -> {result_type} = x |> {transform}
"#
        );
        let program = checked(&source);
        let (library, trace) = try_lower_program_to_library_with_trace(&program).unwrap();
        same_dag(
            library.dag(),
            try_lower_program_to_library(&program).unwrap().dag(),
        );
        assert!(trace.gradients.is_empty());
        assert_eq!(trace.boundaries.len(), 1);
        assert_eq!(trace.boundaries[0].kind, boundary);
        assert_eq!(trace.contexts[1].kind, context);
    }
}
