//! Host policy for correctly rounded device operations.

use chelis_dag_core::dag::{Dag, DagNode, FusedStepOp, RiscOp};
use chelis_numeric_core::prim::Prim;

const fn fused_device_fenced_name(op: FusedStepOp) -> Option<&'static str> {
    match op {
        FusedStepOp::Pow => Some("pow"),
        FusedStepOp::Exp => Some("exp"),
        FusedStepOp::Log => Some("log"),
        FusedStepOp::Sin => Some("sin"),
        FusedStepOp::Cos => Some("cos"),
        FusedStepOp::Tan => Some("tan"),
        FusedStepOp::Atan => Some("atan"),
        FusedStepOp::Tanh => Some("tanh"),
        FusedStepOp::Erf => Some("erf"),
        FusedStepOp::Erfc => Some("erfc"),
        FusedStepOp::Sqrt => Some("sqrt"),
        FusedStepOp::Add
        | FusedStepOp::Sub
        | FusedStepOp::Mul
        | FusedStepOp::Div
        | FusedStepOp::FloorDiv
        | FusedStepOp::TruncDiv
        | FusedStepOp::MaxElem
        | FusedStepOp::MinElem
        | FusedStepOp::Neg
        | FusedStepOp::Recip
        | FusedStepOp::Abs
        | FusedStepOp::Floor
        | FusedStepOp::Ceil
        | FusedStepOp::Round => None,
    }
}

/// chelis#2957 GPU fence (`spec/design/correctly_rounded_math.md` §4.3): a
/// device lane has no correctly rounded kernels for [05-OP-46]'s
/// transcendentals, `sqrt`, or [05-OP-79]'s `pow`, so a DAG that computes one, directly or inside
/// a fused chain, is rejected through [05-UNS-1] rather than computed with a
/// vendor library.
pub fn reject_device_correctly_rounded_ops(
    nodes: &[DagNode],
    target: &'static str,
) -> Result<(), chelis_types::unsupported::Unsupported> {
    for node in nodes {
        let name = match &node.op {
            RiscOp::Exp => Some("exp"),
            RiscOp::Log => Some("log"),
            RiscOp::Sin => Some("sin"),
            RiscOp::Cos => Some("cos"),
            RiscOp::Tan => Some("tan"),
            RiscOp::Atan => Some("atan"),
            RiscOp::Tanh => Some("tanh"),
            RiscOp::Erf => Some("erf"),
            RiscOp::Erfc => Some("erfc"),
            RiscOp::Pow => Some("pow"),
            RiscOp::Sqrt => Some("sqrt"),
            RiscOp::FusedElem { ops } => ops
                .iter()
                .find_map(|step| fused_device_fenced_name(step.op)),
            _ => None,
        };
        if let Some(name) = name {
            return Err(device_correctly_rounded_rejection(name, target));
        }
    }
    Ok(())
}

/// The [05-UNS-1] rejection of a correctly rounded operation `name` on the
/// device lane `target`: one of [05-OP-46]'s transcendentals, `sqrt`, or
/// [05-OP-79]'s `pow`.
pub fn device_correctly_rounded_rejection(
    name: &str,
    target: &'static str,
) -> chelis_types::unsupported::Unsupported {
    let authority = if name == "pow" {
        chelis_types::deliberate_rejection!(
            "[05-OP-79]",
            "device pow is fenced until the device lane has a correctly rounded kernel"
        )
    } else {
        chelis_types::deliberate_rejection!(
            "[05-OP-46]",
            "device transcendentals and sqrt are fenced until the device lane has correctly rounded kernels"
        )
    };
    chelis_types::unsupported::Unsupported::new(
        chelis_types::unsupported::UnsupportedKind::Op(name.to_string()),
        format!(
            "`{name}` must be correctly rounded and the {target} device lane has no correctly rounded kernel for it; build for `--target c`"
        ),
        chelis_types::unsupported::Stage::Codegen(target),
        authority,
    )
}
/// IR validation pass for the Metal target's admissible dtype matrix.
///
/// Per spec/04-type-system.md §1.1.3 the Metal backend rejects f64
/// because Apple Silicon GPUs lack FP64 ALUs (software emulation is
/// explicitly out of scope). The spec names three rejection surfaces:
/// the CLI gate, the IR validation pass, and the codegen entry. This
/// function is the IR validation pass; the CLI gate is
/// `chelis-cli::reject_unsupported_metal_ops` and the codegen entry is
/// `chelis-backend-metal::emit::Emitter::require_metal_admissible`.
/// All three surfaces emit the same diagnostic text so a regression
/// in one is caught by the same string-match tests used by the
/// others.
///
/// Returns `Ok(())` if every node's tensor precision is admissible on
/// Metal, or `Err(message)` with the spec-pinned diagnostic for the
/// first f64 / unsupported-precision node encountered. The message
/// uses the same wording as the CLI gate and codegen entry so the
/// three surfaces speak with one voice.
pub fn validate_metal_admissible_precisions(dag: &Dag) -> Result<(), String> {
    for node in dag.nodes() {
        match node.output_type.precision {
            Prim::F32
            | Prim::F16
            | Prim::Bf16
            | Prim::Int8
            | Prim::Int16
            | Prim::Int32
            | Prim::Int64
            | Prim::Bool => {}
            Prim::F64 => {
                return Err(format!(
                    "IR validation rejects FP64 for `--target metal` (node {}): \
                     Apple Silicon GPUs lack FP64 ALUs; use `--target c` or \
                     `--target hip` for f64 workloads. \
                     See spec/04-type-system.md §1.1.3.",
                    node.id.0
                ));
            }
            other => {
                return Err(format!(
                    "IR validation rejects precision `{}` for `--target metal` \
                     (node {}). The Metal backend admits the active dtype set \
                     per spec/04-type-system.md §1.1.3 except f64; supported: \
                     f32/f16/bf16/i8/i16/i32/i64/bool.",
                    other.name(),
                    node.id.0
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chelis_dag_core::dag::{DimInfo, TensorType};

    fn tensor_ty(dims: &[usize], precision: Prim) -> TensorType {
        TensorType {
            dims: dims.iter().copied().map(DimInfo::Lit).collect(),
            precision,
        }
    }
    /// F4: the IR validation pass for Metal must accept admissible
    /// dtypes and reject f64 with the spec-pinned diagnostic.
    #[test]
    fn metal_validation_accepts_admissible_dtypes() {
        for prec in [
            Prim::F32,
            Prim::F16,
            Prim::Bf16,
            Prim::Int8,
            Prim::Int16,
            Prim::Int32,
            Prim::Int64,
            Prim::Bool,
        ] {
            let mut dag = Dag::new();
            let decl = dag.declare("test");
            dag.add_node(
                decl,
                RiscOp::Load { name: "x".into() },
                vec![],
                tensor_ty(&[4], prec),
                None,
            );
            assert!(
                validate_metal_admissible_precisions(&dag).is_ok(),
                "{prec:?} should be admissible on Metal"
            );
        }
    }

    #[test]
    fn metal_validation_rejects_f64_with_spec_diagnostic() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            tensor_ty(&[4], Prim::F64),
            None,
        );
        let err =
            validate_metal_admissible_precisions(&dag).expect_err("f64 must be rejected on Metal");
        assert!(
            err.contains("FP64") && err.contains("Apple Silicon"),
            "diagnostic must cite the spec hardware constraint: {err}"
        );
        assert!(
            err.contains("§1.1.3"),
            "diagnostic must cite spec section: {err}"
        );
        assert!(
            err.contains("--target c") || err.contains("--target hip"),
            "diagnostic must point at the alternate targets: {err}"
        );
    }
}
