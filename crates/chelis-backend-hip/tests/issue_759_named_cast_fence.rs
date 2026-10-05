//! chelis#759: the HIP cast kernels are an unguarded device-side conversion,
//! so codegen rejects every named cast rung with a typed diagnostic naming
//! the rung instead of emitting it without its traps. A checked `cast` on
//! the same graph shape stays admitted.

mod support;

use chelis_ir::dag::{Dag, DimInfo, NamedCastMode, RiscOp, TensorType};
use chelis_types::types::Prim;
use chelis_types::unsupported::Unsupported;

fn vector(precision: Prim) -> TensorType {
    TensorType {
        dims: vec![DimInfo::Lit(4)],
        precision,
    }
}

/// `out = op(add(a, b))` from `source` to `target`.
fn cast_dag(op: RiscOp, source: Prim, target: Prim) -> Dag {
    let mut dag = Dag::new();
    let decl = dag.declare("test");
    let a = dag.add_node(
        decl,
        RiscOp::Load { name: "a".into() },
        vec![],
        vector(source),
        None,
    );
    let b = dag.add_node(
        decl,
        RiscOp::Load { name: "b".into() },
        vec![],
        vector(source),
        None,
    );
    let sum = dag.add_node(decl, RiscOp::Add, vec![a, b], vector(source), None);
    let cast = dag.add_node(decl, op, vec![sum], vector(target), None);
    let stored = dag.add_node(
        decl,
        RiscOp::Store { name: "out".into() },
        vec![cast],
        vector(target),
        None,
    );
    dag.add_root(stored);
    dag
}

fn codegen(dag: &Dag, name: &str) -> Result<(), Unsupported> {
    support::codegen_hip(dag, name).map(|_| ())
}

#[test]
fn every_named_cast_rung_is_rejected_with_a_typed_diagnostic() {
    for &mode in NamedCastMode::ALL {
        let source = match mode {
            NamedCastMode::Wrap => Prim::Int64,
            NamedCastMode::Trunc | NamedCastMode::Saturate => Prim::F32,
        };
        let op = RiscOp::NamedCast {
            mode,
            new_precision: Prim::Int32,
        };
        let name = mode.keyword();
        let rendered = codegen(&cast_dag(op, source, Prim::Int32), name)
            .expect_err(name)
            .to_string();
        assert!(
            rendered.contains(&format!("`{name}`")),
            "{name}: {rendered}"
        );
        assert!(rendered.contains("codegen:hip"), "{name}: {rendered}");
        assert!(rendered.contains("chelis#759"), "{name}: {rendered}");
    }
}

#[test]
fn the_checked_cast_on_the_same_graph_is_admitted() {
    let op = RiscOp::Cast {
        new_precision: Prim::F64,
    };
    codegen(&cast_dag(op, Prim::F32, Prim::F64), "cast").expect("the checked cast is admitted");
}
