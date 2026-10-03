//! HIP GPU code generation backend for the Chelis language.
//!
//! Generates C host code with embedded HIP kernel source strings.
//! At runtime, `hiprtc` JIT-compiles the kernels and dispatches them to GPU.

/// Primitive types the HIP backend's tensor-DAG path can realize.
/// Declared from hardware spec + local HIP environment verification.
/// Declaration-only in CI (no HIP toolchain) — verified in the field via
/// [05-UNS-1] wiring when the backend rejection path fires.
pub const TENSOR_CAPABLE_PRIMS: &[chelis_types::types::Prim] = &[
    chelis_types::types::Prim::F32,
    chelis_types::types::Prim::F64,
    chelis_types::types::Prim::Bool,
    chelis_types::types::Prim::Bf16,
    chelis_types::types::Prim::F16,
    chelis_types::types::Prim::Int32,
    chelis_types::types::Prim::Int64,
];

use chelis_unord::UnordMap;

use chelis_ir::dag::DimExpr;

pub mod blas;
mod emit;
pub(crate) mod fusion;
pub mod kernels;
pub mod launch;
pub mod memory;

/// Result of HIP code generation.
pub struct HipCodegenResult {
    /// Generated C host source (includes `#include "chelis_hip_runtime.h"` and kernel strings).
    pub c_source: String,
    /// Generated C header declaration for the function.
    pub h_header: String,
    /// Compiler flags required (e.g., passed to `hipcc`).
    pub compile_flags: Vec<String>,
    /// Linker flags required (e.g., `-lhiprtc`).
    pub link_flags: Vec<String>,
    /// Input slot labels in positional order.
    pub input_labels: Vec<String>,
    /// Output slot labels in positional order.
    pub output_labels: Vec<String>,
    /// Unresolved symbolic dimensions that the generated function binds from input metadata.
    pub symbolic_dims: Vec<String>,
    /// Human-readable peak device-memory formula from the slot plan plus inline staged-reduction scratch.
    pub peak_device_bytes_formula: String,
    /// Concrete peak device-memory estimate when every term is statically known.
    pub peak_device_bytes_estimate: Option<usize>,
    peak_device_bytes_terms: Vec<DimExpr>,
    peak_device_bytes_static_extra: usize,
}

/// One device translation unit used by a host-program wrapper.
pub struct HipHostTensorHelperCodegen {
    pub name: String,
    pub result: HipCodegenResult,
}

/// A scalar/container host wrapper plus every Count-bearing HIP helper it calls.
pub struct HipHostProgramCodegenResult {
    pub host: chelis_backend_c::CodegenResult,
    pub device_helpers: Vec<HipHostTensorHelperCodegen>,
}

impl HipCodegenResult {
    pub fn peak_device_bytes_at(
        &self,
        bindings: &UnordMap<String, usize>,
    ) -> Result<usize, String> {
        self.peak_device_bytes_terms
            .iter()
            .try_fold(self.peak_device_bytes_static_extra, |acc, term| {
                Ok(acc + term.evaluate(bindings)?)
            })
    }
}

/// Return the path to the HIP runtime directory (relative to the crate root).
pub fn runtime_dir() -> &'static str {
    "runtime"
}

/// Generate HIP GPU source code from a RISC DAG.
///
/// The generated code follows the same ABI as the C backend:
/// ```c
/// void func_name(chelis_tensor **inputs, int n_in,
///                chelis_tensor **outputs, int n_out);
/// ```
///
/// Inputs arrive as host tensors, are transferred to GPU, processed via
/// HIP kernels, and results are transferred back to host tensors in outputs.
/// Run [`prepare_dag_for_codegen`] before ownership lowering. Direct verified
/// callers must supply BLAS operands whose storage plan proves materialization;
/// a borrowed device input has no implicit contiguous-layout guarantee.
///
/// ```compile_fail
/// # use chelis_ir::dag::Dag;
/// fn bypass(raw: &Dag) {
///     let _ = chelis_backend_hip::codegen_hip(raw, "unchecked");
/// }
/// ```
pub fn codegen_hip(
    dag: chelis_ir::ownership::VerifiedDagProgram,
    func_name: &str,
) -> Result<HipCodegenResult, chelis_types::unsupported::Unsupported> {
    // chelis#1277 C4.1/C4.5: derived after the last rewrite, so this runs on
    // the specialized DAG the emitter actually consumes.
    dag.emission()
        .check_axis_sources(chelis_types::unsupported::Stage::Codegen("hip"))?;
    chelis_ir::dag::reject_device_correctly_rounded_ops(dag.emission().nodes(), "hip")?;
    let h_header = format!(
        "#include \"chelis_device_owner.h\"\nextern \"C\" void {func_name}(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);\nextern \"C\" void {func_name}_device(const chelis_device_tensor_owner *const *inputs, chelis_device_rank n_in, chelis_device_tensor_owner **outputs, chelis_device_rank n_out);"
    );
    let (input_labels, output_labels, symbolic_dims) = {
        let emission = dag.emission();
        (
            emit::HipEmitter::input_labels(emission),
            emit::HipEmitter::output_labels(emission),
            emission.symbolic_params(),
        )
    };
    let plan = chelis_ir::ownership::plan_hip_storage(dag).map_err(|error| {
        chelis_types::unsupported::Unsupported::new(
            chelis_types::unsupported::UnsupportedKind::Op("storage planning".to_string()),
            error.to_string(),
            chelis_types::unsupported::Stage::Codegen("hip"),
            chelis_types::deliberate_rejection!(
                "[04-SHAPE-1]",
                "HIP storage placement requires the verified exact-capacity plan"
            ),
        )
    })?;
    let (c_source, peak_device_bytes) = emit::HipEmitter::emit_dag(plan, func_name)?;
    let mut link_flags = vec!["-lhiprtc".to_string()];
    // WS-A3: bf16 / f16 matmul also routes through hipBLAS (via
    // `hipblasGemmEx`). Add `-lhipblas` whenever any hipblas wrapper
    // call appears in the generated source, not just the f32 ones.
    if c_source.contains("chelis_hipblas_sgemm_row_major(")
        || c_source.contains("chelis_hipblas_sgemm_batched_row_major(")
        || c_source.contains("chelis_hipblas_sgemm_strided_batched_row_major(")
        || c_source.contains("chelis_hipblas_dgemm_row_major(")
        || c_source.contains("chelis_hipblas_dgemm_batched_row_major(")
        || c_source.contains("chelis_hipblas_dgemm_strided_batched_row_major(")
        || c_source.contains("chelis_hipblas_bf16_gemm_f32_acc_row_major(")
        || c_source.contains("chelis_hipblas_f16_gemm_f32_acc_row_major(")
    {
        link_flags.push("-lhipblas".to_string());
    }
    Ok(HipCodegenResult {
        c_source,
        h_header,
        compile_flags: vec![],
        link_flags,
        input_labels,
        output_labels,
        symbolic_dims,
        peak_device_bytes_formula: peak_device_bytes.formula,
        peak_device_bytes_estimate: peak_device_bytes.estimate,
        peak_device_bytes_terms: peak_device_bytes.terms,
        peak_device_bytes_static_extra: peak_device_bytes.extra_bytes,
    })
}

/// Apply HIP's final BLAS selection and operand materialization before ownership
/// lowering. This pass is idempotent. Explicit Realize nodes make all temporary
/// bytes and lifetimes visible to the shared storage planner, including when a
/// device caller supplies noncontiguous inputs.
pub fn prepare_dag_for_codegen(dag: chelis_ir::dag::Dag) -> chelis_ir::dag::Dag {
    use chelis_ir::dag::{Dag, NodeId, RiscOp};
    let selected = chelis_ir::specialize::specialize_for_blas(&dag);
    let mut out = Dag::new();
    out.inherit_declarations(&selected);
    let mut remap: UnordMap<NodeId, NodeId> = UnordMap::new();
    let mut materialized: UnordMap<NodeId, NodeId> = UnordMap::new();
    for node in selected.nodes() {
        let mut inputs = node.inputs.iter().map(|id| remap[id]).collect::<Vec<_>>();
        if matches!(node.op, RiscOp::BlasMatmul { .. }) {
            for input in &mut inputs {
                if matches!(out.get(*input).unwrap().op, RiscOp::Realize) {
                    continue;
                }
                *input = *materialized.entry(*input).or_insert_with(|| {
                    let source = out.get(*input).unwrap().clone();
                    let realized = out.add_node(
                        source.owner,
                        RiscOp::Realize,
                        vec![*input],
                        source.output_type,
                        source.span_id,
                    );
                    out.node_mut(realized).unwrap().merged_spans = source.merged_spans;
                    realized
                });
            }
        }
        let reusable = node.reusable_input.map(|old| {
            node.inputs
                .iter()
                .position(|id| *id == old)
                .map(|index| inputs[index])
                .unwrap_or(remap[&old])
        });
        let id = out.add_node(
            node.owner.remap(&remap),
            node.op.clone(),
            inputs,
            node.output_type.clone(),
            node.span_id.clone(),
        );
        if let Some(source) = reusable {
            out.set_reusable_input(id, source);
        }
        out.node_mut(id).unwrap().merged_spans = node.merged_spans.clone();
        out.preserve_shape_deps(id, &node.shape_deps, &remap);
        out.preserve_result_claim_deps(id, &node.result_claim_deps, &remap);
        remap.insert(node.id, id);
    }
    for root in selected.roots() {
        out.add_root(remap[root]);
    }
    out
}

/// Compile a verified host program whose Count-bearing tensor helpers run
/// on the device.
///
/// `helpers` is the wrapper's helper manifest, read off the concrete program
/// by [`chelis_backend_c::host_tensor_helper_codegen`] before payload
/// selection and ownership lowering. Every helper whose DAG contains `Count`
/// becomes its own HIP translation unit: the C wrapper declares it as an
/// external symbol, and this function lowers the helper's source DAG through
/// the same preparation, ownership lowering, and verification a tensor entry
/// receives before emitting it with [`codegen_hip`]. Other tensor helpers
/// keep their C-host disposition. There is no C fallback for a Count helper:
/// the selection predicate below is the only place that decides.
pub fn codegen_hip_host_program(
    program: &chelis_ir::ownership::VerifiedHostProgram,
    func_name: &str,
    helpers: Vec<chelis_backend_c::HostTensorHelperCodegen>,
) -> Result<HipHostProgramCodegenResult, chelis_types::unsupported::Unsupported> {
    let count_helpers = helpers
        .into_iter()
        .filter(|helper| {
            helper
                .dag
                .nodes()
                .iter()
                .any(|node| matches!(node.op, chelis_ir::dag::RiscOp::Count { .. }))
        })
        .collect::<Vec<_>>();
    let external_names = count_helpers
        .iter()
        .map(|helper| helper.name.clone())
        .collect::<Vec<_>>();
    let host = chelis_backend_c::codegen_host_program_with_external_tensor_helpers(
        program,
        func_name,
        &external_names,
    )?;
    let device_helpers = count_helpers
        .into_iter()
        .map(|helper| {
            let verified = verified_host_helper_dag(&helper.name, helper.dag)?;
            Ok(HipHostTensorHelperCodegen {
                result: codegen_hip(verified, &helper.name)?,
                name: helper.name,
            })
        })
        .collect::<Result<Vec<_>, chelis_types::unsupported::Unsupported>>()?;
    Ok(HipHostProgramCodegenResult {
        host,
        device_helpers,
    })
}

/// Lower one externalized helper DAG exactly as a HIP tensor entry is
/// lowered: HIP payload selection, ownership lowering, then verification.
fn verified_host_helper_dag(
    helper_name: &str,
    dag: chelis_ir::dag::Dag,
) -> Result<chelis_ir::ownership::VerifiedDagProgram, chelis_types::unsupported::Unsupported> {
    let selected = prepare_dag_for_codegen(dag);
    let unsupported = |error: chelis_ir::ownership::OwnershipError| {
        chelis_types::unsupported::Unsupported::new(
            chelis_types::unsupported::UnsupportedKind::Construct(format!(
                "HIP device helper `{helper_name}` ownership lowering"
            )),
            error.to_string(),
            chelis_types::unsupported::Stage::Codegen("hip"),
            chelis_types::deliberate_rejection!(
                "[04-SHAPE-1]",
                "a device-emitted host tensor helper requires the same verified ownership plan as a tensor entry"
            ),
        )
    };
    chelis_ir::ownership::verify_ownership(
        chelis_ir::ownership::lower_dag_ownership(selected).map_err(unsupported)?,
    )
    .map_err(unsupported)
}

#[cfg(test)]
pub(crate) mod testing {
    use chelis_ir::ownership::{OwnershipError, VerifiedDagProgram};

    pub(crate) fn verified_dag(
        dag: &chelis_ir::dag::Dag,
    ) -> Result<VerifiedDagProgram, OwnershipError> {
        let selected = crate::prepare_dag_for_codegen(dag.clone());
        chelis_ir::ownership::verify_ownership(chelis_ir::ownership::lower_dag_ownership(selected)?)
    }
}

#[cfg(test)]
mod preparation_tests {
    use super::*;
    use chelis_ir::dag::{Dag, DimInfo, RiscOp, TensorType};
    use chelis_ir::ownership::{
        StoragePlacement, lower_dag_ownership, plan_hip_storage, verify_ownership,
    };
    use chelis_types::types::Prim;

    fn matrix_program(precision: Prim) -> Dag {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let ty = TensorType {
            dims: vec![DimInfo::Lit(2), DimInfo::Lit(2)],
            precision,
        };
        let a = dag.add_node(
            decl,
            RiscOp::Load { name: "a".into() },
            vec![],
            ty.clone(),
            None,
        );
        let b = dag.add_node(
            decl,
            RiscOp::Load { name: "b".into() },
            vec![],
            ty.clone(),
            None,
        );
        let output = dag.add_node(
            decl,
            RiscOp::BlasMatmul {
                batch_dims: vec![],
                m: DimExpr::Concrete(2),
                n: DimExpr::Concrete(2),
                k: DimExpr::Concrete(2),
                accumulator: if precision == Prim::F64 {
                    Prim::F64
                } else {
                    Prim::F32
                },
            },
            vec![a, b],
            ty,
            None,
        );
        dag.add_root(a);
        dag.add_root(output);
        dag
    }

    #[test]
    fn blas_preparation_is_idempotent_and_accounts_for_every_operand_materialization() {
        for precision in [Prim::F32, Prim::F64, Prim::Bf16, Prim::F16] {
            let prepared = prepare_dag_for_codegen(matrix_program(precision));
            let again = prepare_dag_for_codegen(prepared.clone());
            assert_eq!(
                serde_json::to_value(&prepared).unwrap(),
                serde_json::to_value(again).unwrap()
            );
            assert_eq!(prepared.roots().len(), 2);
            let realizes = prepared
                .nodes()
                .iter()
                .filter(|node| matches!(node.op, RiscOp::Realize))
                .map(|node| node.id)
                .collect::<Vec<_>>();
            assert_eq!(realizes.len(), 2);
            let verified = verify_ownership(lower_dag_ownership(prepared).unwrap()).unwrap();
            let plan = plan_hip_storage(verified).unwrap();
            for id in &realizes {
                assert!(matches!(
                    plan.placement(*id),
                    Some(StoragePlacement::OwnedSlot { .. })
                ));
                assert_eq!(
                    plan.slot(plan.slot_for_node(*id).unwrap())
                        .unwrap()
                        .capacity()
                        .allocation_bytes(),
                    Some((4 * precision.runtime_dtype().unwrap().byte_width()) as u64)
                );
            }
            assert_ne!(
                plan.slot_for_node(realizes[0]),
                plan.slot_for_node(realizes[1]),
                "both operands stay live through GEMM"
            );
            let source = codegen_hip(
                testing::verified_dag(&matrix_program(precision)).unwrap(),
                "prepared_blas",
            )
            .unwrap()
            .c_source;
            assert!(source.contains("kernel_realize_"));
        }
    }

    #[test]
    fn direct_codegen_rejects_blas_borrows_without_the_preparation_pass() {
        let verified =
            verify_ownership(lower_dag_ownership(matrix_program(Prim::F32)).unwrap()).unwrap();
        let error = match codegen_hip(verified, "unprepared_blas") {
            Ok(_) => panic!("unprepared borrowed BLAS operands must not reach source emission"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("prepare_dag_for_codegen"));
    }
}
