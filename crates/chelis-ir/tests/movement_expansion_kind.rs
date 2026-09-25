//! [05-MOV-1]/[04-NUM-9]: rank-preserving expansion and insertion retain identity.
use chelis_ir::axis_sources::{ExpansionKind, expansion_kind, local_dim_guard_sites};
use chelis_ir::dag::{Dag, DimInfo, RiscOp, RtDim, TensorType};
use chelis_types::types::Prim;
#[test]
fn expansion_kind_and_local_guard_keep_the_primitive_identity() {
    for (out, expected, op) in [
        (vec![3], ExpansionKind::Expand, "expand"),
        (vec![3, 1], ExpansionKind::Insert, "insert"),
    ] {
        let mut dag = Dag::new();
        let x = dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            TensorType {
                dims: vec![DimInfo::Lit(1)],
                precision: Prim::F32,
            },
            None,
        );
        let size = dag.add_node(
            RiscOp::Load {
                name: "size".into(),
            },
            vec![],
            TensorType {
                dims: vec![],
                precision: Prim::Int64,
            },
            None,
        );
        let bound = dag.add_node(
            RiscOp::Add,
            vec![size, size],
            TensorType {
                dims: vec![],
                precision: Prim::Int64,
            },
            None,
        );
        let result = dag.add_node(
            RiscOp::Expand {
                axis: 0,
                size: RtDim::Node(1),
            },
            vec![x, bound],
            TensorType {
                dims: out.into_iter().map(DimInfo::Lit).collect(),
                precision: Prim::F32,
            },
            None,
        );
        assert_eq!(expansion_kind(&dag, result), Some(expected));
        let guards = local_dim_guard_sites(&dag).unwrap();
        assert!(
            guards
                .iter()
                .any(|((node, _), claim)| *node == result.0 && claim.op == op),
            "{guards:?}"
        );
        assert!(
            !guards
                .iter()
                .any(|((node, _), claim)| *node == result.0 && claim.op != op),
            "{guards:?}"
        );
        assert_eq!(expansion_kind(&dag, x), None);
        let invalid = dag.add_node(
            RiscOp::Expand {
                axis: 0,
                size: RtDim::Lit(1),
            },
            vec![x],
            TensorType {
                dims: vec![DimInfo::Lit(1); 3],
                precision: Prim::F32,
            },
            None,
        );
        assert_eq!(expansion_kind(&dag, invalid), None);
    }
}
