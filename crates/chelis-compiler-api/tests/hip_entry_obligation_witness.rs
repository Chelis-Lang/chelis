//! chelis#1374/#1376: which `ExtentWitness` the HIP target refuses.
//!
//! [05-SHAPE-1] excludes the runtime `shape` VALUE read from the HIP device
//! lane, and `reject_unsupported_hip_ops` enforces that. A witness carrying a
//! `spec/04-type-system.md` section 4.7 named claim is not such a read: nothing
//! computes with its value, `retain_invocation_witnesses` keeps it alive
//! through a `shape_deps` edge alone, and the obligation is discharged in the
//! host prologue beside the other entry guards.
//!
//! The pair below is the same witness with and without one data edge, so the
//! boundary is measured rather than asserted.

use chelis_compiler_api::compiler::reject_unsupported_hip_ops;
use chelis_ir::dag::{
    Dag, DimInfo, ExtentClaim, ExtentWitnessSite, NodeId, RiscOp, RtAxis, TensorType,
};
use chelis_types::types::Prim;

fn tensor(dims: Vec<DimInfo>, precision: Prim) -> TensorType {
    TensorType { dims, precision }
}

fn load(dag: &mut Dag, name: &str) -> NodeId {
    dag.add_node(
        RiscOp::Load { name: name.into() },
        vec![],
        tensor(vec![DimInfo::Named("n".into(), None)], Prim::F32),
        None,
    )
}

fn witness(dag: &mut Dag, parameter: &str, input: NodeId, claims: Vec<ExtentClaim>) -> NodeId {
    dag.add_node(
        RiscOp::ExtentWitness {
            site: ExtentWitnessSite::Caller,
            parameter: parameter.into(),
            axis: RtAxis::Lit(0),
            requirements: Vec::new(),
            claims,
        },
        vec![input],
        tensor(vec![], Prim::Int64),
        None,
    )
}

/// `def f[n](x: tensor[n, f32], p: tensor[n, f32])`: `p`'s witness carries the
/// repeated binder's equality and nothing reads its value.
///
/// EVIDENTIARY STATUS: regression test. Measured red at `01c6e33a1`, where the
/// gate refused this graph under [05-SHAPE-1] and
/// `build_symbolic_matmul_succeeds_on_c_and_hip_targets` - an ordinary matmul
/// signature - stopped building on the HIP target.
#[test]
fn an_entry_obligation_witness_passes_the_hip_gate() {
    let mut dag = Dag::new();
    let x = load(&mut dag, "x");
    let p = load(&mut dag, "p");
    let declaring = witness(&mut dag, "x", x, Vec::new());
    let owed = witness(
        &mut dag,
        "p",
        p,
        vec![ExtentClaim {
            claim: "n".into(),
            requirement_declares: true,
        }],
    );
    dag.node_mut(owed).expect("witness").inputs.push(declaring);
    let result = dag.add_node(
        RiscOp::Copy,
        vec![x],
        tensor(vec![DimInfo::Named("n".into(), None)], Prim::F32),
        None,
    );
    dag.node_mut(result).expect("carrier").shape_deps = vec![owed];
    dag.add_root(result);

    reject_unsupported_hip_ops(&dag)
        .expect("a witness nothing reads is an entry obligation, not a device shape read");
}

/// The same witness with one data edge: a node now computes with the extent,
/// which is exactly the runtime `shape` value read [05-SHAPE-1] excludes.
///
/// EVIDENTIARY STATUS: negative twin of the row above, and the lock on the
/// relaxation's boundary. Measured by deleting the `Cast` consumer, which
/// turns this row green.
#[test]
fn a_witness_a_device_node_reads_still_draws_the_shape_refusal() {
    let mut dag = Dag::new();
    let x = load(&mut dag, "x");
    let p = load(&mut dag, "p");
    let declaring = witness(&mut dag, "x", x, Vec::new());
    let owed = witness(
        &mut dag,
        "p",
        p,
        vec![ExtentClaim {
            claim: "n".into(),
            requirement_declares: true,
        }],
    );
    dag.node_mut(owed).expect("witness").inputs.push(declaring);
    let read = dag.add_node(
        RiscOp::Cast {
            new_precision: Prim::F32,
        },
        vec![owed],
        tensor(vec![], Prim::F32),
        None,
    );
    dag.add_root(read);

    let error = reject_unsupported_hip_ops(&dag)
        .expect_err("a read extent is the runtime shape value read HIP excludes");
    let text = error
        .errors
        .iter()
        .map(|diagnostic| diagnostic.message.clone())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        text.contains("does not support the runtime `shape` value read"),
        "the refusal keeps its [05-SHAPE-1] wording: {text}"
    );
}
