//! Phase 3's Metal projection is a typed no-reuse boundary.
//!
//! These tests intentionally exercise the production projection and public
//! codegen entry instead of a backend-local eligibility helper.  Metal may
//! decline an unsupported graph, but it cannot accept a reusable-storage
//! capability or advertise an aborting placeholder as a successful artifact.

mod support;

use chelis_ir::dag::{Dag, DimInfo, RiscOp, RtDim, TensorType};
use chelis_types::types::Prim;
use chelis_types::unsupported::{RejectionAuthorityKind, Stage};

fn vector(n: usize) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(n)],
        precision: Prim::F32,
    }
}

#[test]
fn virtual_matmul_nodes_are_excluded_from_the_physical_plan() {
    let matrix = |rows, cols| TensorType {
        dims: vec![DimInfo::Lit(rows), DimInfo::Lit(cols)],
        precision: Prim::F32,
    };
    let cube = TensorType {
        dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
        precision: Prim::F32,
    };
    let mut dag = Dag::new();
    let left = dag.add_node(
        RiscOp::Load {
            name: "left".into(),
        },
        vec![],
        matrix(2, 3),
        None,
    );
    let right = dag.add_node(
        RiscOp::Load {
            name: "right".into(),
        },
        vec![],
        matrix(3, 4),
        None,
    );
    let expand_left = dag.add_node(
        RiscOp::Expand {
            axis: 2,
            size: RtDim::Lit(4),
        },
        vec![left],
        cube.clone(),
        None,
    );
    let expand_right = dag.add_node(
        RiscOp::Expand {
            axis: 0,
            size: RtDim::Lit(2),
        },
        vec![right],
        cube.clone(),
        None,
    );
    let multiply = dag.add_node(RiscOp::Mul, vec![expand_left, expand_right], cube, None);
    let output = dag.add_node(
        RiscOp::Sum {
            axis: 1,
            accumulator: Prim::F32,
        },
        vec![multiply],
        matrix(2, 4),
        None,
    );
    dag.add_root(output);

    let verified = support::verified_dag(&dag);
    let plan = chelis_backend_metal::plan_metal(verified);
    assert_eq!(plan.allocation_count(), 3);
    assert!(plan.allocation_for(left).is_some());
    assert!(plan.allocation_for(right).is_some());
    assert!(plan.allocation_for(output).is_some());
    assert!(plan.allocation_for(expand_left).is_none());
    assert!(plan.allocation_for(expand_right).is_none());
    assert!(plan.allocation_for(multiply).is_none());

    chelis_backend_metal::codegen_metal(plan, "matmul")
        .expect("the physical plan must be bijective with successful emission");
}

#[test]
fn reusable_hint_still_projects_distinct_metal_allocations() {
    let mut dag = Dag::new();
    let input = dag.add_node(RiscOp::Load { name: "x".into() }, vec![], vector(4), None);
    let scale = dag.add_node(RiscOp::synth_const(Prim::F32, 2.0), vec![], vector(4), None);
    let output = dag.add_node(RiscOp::Add, vec![input, scale], vector(4), None);
    dag.set_reusable_input(output, input);
    dag.add_root(output);

    let verified = support::verified_dag(&dag);
    let plan = chelis_backend_metal::plan_metal(verified);

    assert_ne!(
        plan.allocation_for(input),
        plan.allocation_for(output),
        "an unverified reusable_input hint must not collapse Metal allocations"
    );
    assert_ne!(plan.allocation_for(input), plan.allocation_for(scale));
    assert_ne!(plan.allocation_for(scale), plan.allocation_for(output));
    assert_eq!(plan.allocation_count(), 3);

    let generated = chelis_backend_metal::codegen_metal(plan, "hint_ignored")
        .expect("a reuse hint cannot make Metal codegen fail or alias storage");
    assert_eq!(
        generated.mm_source.matches("= chelis_metal_alloc(").count(),
        3
    );
}

#[test]
fn unsupported_lowering_is_a_typed_error_not_an_abort_artifact() {
    let mut dag = Dag::new();
    let rank_two = TensorType {
        dims: vec![DimInfo::Lit(2), DimInfo::Lit(2)],
        precision: Prim::F32,
    };
    let left = dag.add_node(
        RiscOp::Load {
            name: "left".into(),
        },
        vec![],
        rank_two.clone(),
        None,
    );
    let right = dag.add_node(
        RiscOp::Load {
            name: "right".into(),
        },
        vec![],
        rank_two.clone(),
        None,
    );
    let output = dag.add_node(RiscOp::Add, vec![left, right], rank_two, None);
    dag.add_root(output);

    let error = support::try_codegen_metal(&dag, "rank_two_add").unwrap_err();
    assert_eq!(error.stage, Stage::Codegen("metal"));
    assert_eq!(
        error.authority.kind(),
        RejectionAuthorityKind::Unimplemented
    );
    assert!(error.to_string().starts_with("unsupported:"), "{error}");
    assert!(error.to_string().contains("rank 2"), "{error}");
}

#[test]
fn production_source_has_no_local_reuse_or_abort_stub_path() {
    let lib = include_str!("../src/lib.rs");
    let emitter = include_str!("../src/emit.rs");

    for forbidden in [
        "fused_in_place_spec",
        "ReusableOwnedStorage",
        "reusable_input",
        "stub_mm_source_with_reason",
        "M1 fallback stub",
    ] {
        assert!(
            !lib.contains(forbidden) && !emitter.contains(forbidden),
            "Metal production source regained forbidden local reuse/stub state `{forbidden}`"
        );
    }

    let declaration = lib
        .split_once("pub struct MetalNeverReuse {")
        .expect("MetalNeverReuse declaration must remain present")
        .1
        .split_once('}')
        .expect("MetalNeverReuse declaration must remain closed")
        .0;
    assert!(
        declaration.contains("program: VerifiedDagProgram"),
        "{declaration}"
    );
    assert!(declaration.contains("allocations:"), "{declaration}");
    assert!(!declaration.contains("reuse"), "{declaration}");
    assert!(!declaration.contains("token"), "{declaration}");

    let plan_signature = lib
        .split_once("pub fn plan_metal(")
        .expect("plan_metal must remain public")
        .1
        .split_once('{')
        .expect("plan_metal must have a body")
        .0;
    assert!(
        plan_signature.contains("program: VerifiedDagProgram"),
        "{plan_signature}"
    );
    assert!(
        !plan_signature.contains("&VerifiedDagProgram"),
        "{plan_signature}"
    );

    let codegen_signature = lib
        .split_once("pub fn codegen_metal(")
        .expect("codegen_metal must remain public")
        .1
        .split_once('{')
        .expect("codegen_metal must have a body")
        .0;
    assert!(
        codegen_signature.contains("plan: MetalNeverReuse"),
        "{codegen_signature}"
    );
    assert!(
        !codegen_signature.contains("&MetalNeverReuse"),
        "{codegen_signature}"
    );

    assert_eq!(
        emitter.matches("= chelis_metal_alloc(").count(),
        emitter.matches("claim_allocation(node.id)?;").count(),
        "every physical Metal allocation must consume exactly one no-reuse identity"
    );
}
