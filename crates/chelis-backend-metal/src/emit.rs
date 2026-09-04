//! Metal codegen entry point.
//!
//! M2 first cut: emits Objective-C++ host code with embedded MSL kernel
//! strings for **same-shape, contiguous, rank-1** elementwise + fused-elem
//! programs. Reductions land in M4, matmul in M5, and broadcasting/strided
//! layouts will be added incrementally.

use chelis_ir::dag::{DagNode, DimInfo, NodeId, RiscOp, RtDim, TensorType};
use chelis_ir::ownership::VerifiedDagView;
use chelis_types::ScalarValue;

/// chelis#616: the Metal lane requires compile-time movement bounds (it rejects
/// symbolic movement shapes via `require_movement_shape`). This converter
/// materializes the literal bound for the emitters and panics on a node-valued
/// (runtime) bound as a defensive backstop.
fn metal_bound_to_usize(b: &RtDim) -> usize {
    match b {
        RtDim::Lit(n) => *n,
        RtDim::ToEnd => chelis_ir::dag::SHRINK_TO_END,
        RtDim::Node(_) => panic!(
            "Metal backend reached a node-valued (runtime) movement bound; runtime-symbolic \
             movement bounds are not supported on the Metal lane (chelis#616)"
        ),
        RtDim::Sym(name) => panic!(
            "Metal backend reached a symbolic movement bound `{name}`; verify rejects \
             symbolic dims outside reshape targets (chelis#616)"
        ),
        RtDim::InputAxis { .. } => panic!(
            "Metal backend reached InputAxis on a movement owner; IR verification only admits \
             InputAxis for Expand and Reshape"
        ),
    }
}

#[cfg(test)]
mod rejection_authority_tests {
    use super::Emitter;
    use chelis_types::types::Prim;

    #[test]
    fn pad_fill_rejections_use_case_specific_authorities() {
        let fill = chelis_types::scalar_from_f64("pad", Prim::F64, 0.0).unwrap();
        let f64 = Emitter::host_scalar_literal(Prim::F64, fill).unwrap_err();

        assert!(f64.contains("deliberate [04-TGT-1]"), "{f64}");
    }

    #[test]
    fn int64_min_pad_fill_has_a_portable_host_literal() {
        let fill = chelis_types::scalar_from_i64("pad", Prim::Int64, i64::MIN).unwrap();
        assert_eq!(
            Emitter::host_scalar_literal(Prim::Int64, fill).unwrap(),
            "(-9223372036854775807LL - 1LL)"
        );
    }
}

fn metal_pairs_to_usize(bounds: &[(RtDim, RtDim)]) -> Vec<(usize, usize)> {
    bounds
        .iter()
        .map(|(s, e)| (metal_bound_to_usize(s), metal_bound_to_usize(e)))
        .collect()
}
use chelis_types::types::Prim;
use chelis_types::unsupported::{Stage, Unsupported, UnsupportedKind};

use crate::blas;
use crate::dtype;
use crate::kernels;
use chelis_unord::{UnordMap, UnordSet};

/// Distinct input labels in DAG order.
///
/// Mirrors `chelis_backend_hip::emit::HipEmitter::input_labels`.
pub fn input_labels(dag: VerifiedDagView<'_>) -> Vec<String> {
    let mut labels = Vec::new();
    let mut seen = chelis_unord::UnordSet::new();
    for node in dag.nodes() {
        if let RiscOp::Load { name } = &node.op
            && seen.insert(name.as_str().to_string())
        {
            labels.push(name.as_str().to_string());
        }
    }
    labels
}

/// Output specification for one positional output of the emitted function.
///
/// An output may come from an explicit `Store { name }` node in the DAG OR
/// from a root that has no downstream store (the CLI's regular lowering
/// path produces this shape — DAG roots without `Store`). Mirrors
/// `chelis_backend_hip::emit::HipEmitter::output_specs`.
#[derive(Debug, Clone)]
struct OutputSpec {
    id: NodeId,
    label: String,
    is_store: bool,
}

/// Compute output specs in stable order: Store-tagged nodes first, then any
/// remaining roots that aren't already covered.
fn output_specs(dag: VerifiedDagView<'_>) -> Vec<OutputSpec> {
    let mut specs = Vec::new();
    let mut seen = chelis_unord::UnordSet::new();

    for node in dag.nodes() {
        if let RiscOp::Store { name } = &node.op
            && seen.insert(node.id)
        {
            specs.push(OutputSpec {
                id: node.id,
                label: name.as_str().to_string(),
                is_store: true,
            });
        }
    }

    let roots: Vec<NodeId> = if dag.roots().is_empty() {
        dag.nodes().last().map(|n| vec![n.id]).unwrap_or_default()
    } else {
        dag.roots().to_vec()
    };
    for (index, root_id) in roots.into_iter().enumerate() {
        if seen.insert(root_id) {
            specs.push(OutputSpec {
                id: root_id,
                label: format!("root{index}"),
                is_store: false,
            });
        }
    }
    specs
}

/// Output labels in DAG order — `Store { name }` nodes plus any
/// no-Store roots, mirroring `output_specs`. The CLI's `cmd_build_metal`
/// passes this through to the result struct so users can map positional
/// outputs back to symbolic names.
pub fn output_labels(dag: VerifiedDagView<'_>) -> Vec<String> {
    output_specs(dag).into_iter().map(|s| s.label).collect()
}

/// Stub mm-source carrying a structured reason in its abort message.
/// Used when emit_dag returns Err so the runtime diagnostic explains
/// why the DAG was rejected (e.g., integer matmul per spec §5.7.2)
/// instead of the bare "not yet implemented" string. The hint is
/// embedded as a comment in the source AND printed at abort time.
pub fn stub_mm_source_with_reason(func_name: &str, hint: &str) -> String {
    let trimmed = hint.trim();
    let hint_comment = if trimmed.is_empty() {
        String::new()
    } else {
        // Sanitize so a producer-supplied error message cannot break
        // out of the C++ `// ...` comment context.
        let safe = chelis_ir::span_sanitize::sanitize_for_comment(trimmed);
        format!("// reason: {safe}\n")
    };
    let runtime_msg = if trimmed.is_empty() {
        format!("chelis Metal backend stub: codegen for `{func_name}` not yet implemented")
    } else {
        // The runtime diagnostic must survive a C string literal, so
        // escape any `\` and `\"`. Newlines are flattened to spaces so
        // the abort message stays single-line.
        let escaped = trimmed.replace('\\', "\\\\").replace('"', "\\\"");
        let oneline = escaped.replace('\n', " ");
        format!(
            "chelis Metal backend stub: codegen for `{func_name}` not yet implemented; \
             reason: {oneline}"
        )
    };
    format!(
        r#"// Generated by chelis --target metal (M1 fallback stub)
{hint_comment}#import "chelis_metal_runtime.h"
#include "chelis_runtime.h"
#include <stdlib.h>
#include <stdio.h>

extern "C" void {func_name}(chelis_tensor **inputs, int n_in,
                            chelis_tensor **outputs, int n_out) {{
    (void)inputs; (void)n_in; (void)outputs; (void)n_out;
    fprintf(stderr,
        "{runtime_msg}\n");
    abort();
}}
"#
    )
}

/// Inspect the DAG and the emit_dag error message to derive a more
/// precise hint than the bare error. Called by `codegen_metal` when
/// emit_dag returns Err.
///
/// Currently surfaces the integer-matmul §5.7.2 hint when the DAG
/// contains a Sum-rooted matmul subgraph at integer precision (the
/// shape that `blas::detect_matmul_pattern` rejects, falling through
/// to the stub). Additional reason categories slot in here as the
/// stub gains more failure modes.
pub fn stub_reason_hint(dag: VerifiedDagView<'_>, base_reason: &str) -> String {
    use chelis_types::types::Prim;
    // If any Sum node has integer operand precision and matches the
    // expand+mul+sum matmul shape, surface the §5.7.2 hint. The
    // detector itself rejects integer matmul so we cannot reuse it
    // here; instead, walk the Sum nodes directly and check for the
    // integer matmul shape.
    for node in dag.nodes() {
        if !matches!(node.op, RiscOp::Sum { .. }) {
            continue;
        }
        if node.inputs.len() != 1 {
            continue;
        }
        let mul = match dag.get(node.inputs[0]) {
            Some(n) if matches!(n.op, RiscOp::Mul) => n,
            _ => continue,
        };
        if mul.inputs.len() != 2 {
            continue;
        }
        let lhs_prec = mul.output_type.precision;
        let is_int = matches!(
            lhs_prec,
            Prim::Int8 | Prim::Int16 | Prim::Int32 | Prim::Int64
        );
        if !is_int {
            continue;
        }
        let lhs = match dag.get(mul.inputs[0]) {
            Some(n) => n,
            None => continue,
        };
        let rhs = match dag.get(mul.inputs[1]) {
            Some(n) => n,
            None => continue,
        };
        if matches!(lhs.op, RiscOp::Expand { .. }) && matches!(rhs.op, RiscOp::Expand { .. }) {
            return format!(
                "integer matmul not admitted per spec/04-type-system.md §5.7.2; \
                 use floats (f32/f16/bf16) or implement quantization explicitly. \
                 Original emit error: {base_reason}"
            );
        }
    }
    base_reason.to_string()
}

/// Result of an emit pass — source plus the materialized peak-device-bytes
/// total (sum of all per-op buffer allocations the emitted code performs).
pub struct EmitResult {
    pub mm_source: String,
    /// Total bytes of `chelis_metal_alloc()` calls the emitted host code
    /// will issue. On Apple Silicon (unified memory) this is also peak
    /// system RAM consumed for tensor storage. The CLI prefixes the
    /// formula with that note so users do not double-count against host
    /// allocations.
    pub peak_device_bytes: usize,
}

pub(crate) fn emit_verified_dag(
    dag: VerifiedDagView<'_>,
    func_name: &str,
) -> Result<EmitResult, String> {
    // chelis#1277 C4.1: the Metal codegen entry has no typed error channel
    // (`codegen_metal` falls back to an aborting stub), so a sourceless
    // mapping is rendered into the stub's reason here rather than returned
    // as a typed receipt. Metal device-path rows sit at their recorded
    // `lane_divergent` baseline until chelis#1383, per runtime_extents.md
    // C2.5.
    dag.check_axis_sources(Stage::Codegen("metal"))
        .map_err(|unsupported| unsupported.to_string())?;
    reject_integer_abs(dag)?;
    let mut e = Emitter::new(func_name);
    e.emit(dag)?;
    let peak_device_bytes = e.peak_device_bytes();
    Ok(EmitResult {
        mm_source: e.into_source(),
        peak_device_bytes,
    })
}

/// The Metal unary template maps `Abs` to `fabs`; reject integer inputs at
/// the public emission boundary until Phase 3 provides a typed, trapping
/// backend kernel (chelis#699).
fn reject_integer_abs(dag: VerifiedDagView<'_>) -> Result<(), String> {
    if let Some(node) = dag.first_integer_abs_node() {
        return Err(Unsupported::new(
            UnsupportedKind::Op("Abs".to_string()),
            format!("an integer tensor at Metal DAG node {}", node.0),
            Stage::Codegen("metal"),
            chelis_types::unimplemented_rejection!(
                729,
                "integer abs code generation waits for the typed, trapping Phase 3 \
                 kernel (chelis#699); use `chelis eval` for the Phase 2 reference lane"
            ),
        )
        .to_string());
    }
    Ok(())
}

/// Per-tensor metadata emitted alongside the host program.
#[derive(Clone)]
struct TensorPlan {
    /// Buffer variable name in the emitted .mm (e.g. `buf_2`).
    buf: String,
    /// Tensor element precision; the source of truth for MSL spelling
    /// (`dtype::msl_type`), per-element width (`dtype::metal_elem_size`),
    /// runtime tag (`dtype::runtime_dtype_tag`), and reduction
    /// accumulator (`dtype::sum_accumulator`).
    prec: Prim,
    /// Element count (must be statically known for M2).
    n: usize,
    /// Row-major shape (dimensions listed in declaration order).
    shape: Vec<usize>,
}

impl TensorPlan {
    fn msl_ty(&self) -> &'static str {
        dtype::msl_type(self.prec)
    }

    fn elem_size(&self) -> usize {
        dtype::metal_elem_size(self.prec)
    }
}

struct Emitter {
    func_name: String,
    body: Vec<String>,
    /// Collected MSL kernel sources, one per compute node, in emission order.
    /// `(c_var_for_pso, kernel_name, msl_source)`.
    kernels: Vec<(String, String, String)>,
    /// Per-DAG-node tensor plan — `None` if the node hasn't been materialized
    /// yet (e.g., metadata-only ops we will add later) or fell through to a
    /// rejection path.
    plans: Vec<Option<TensorPlan>>,
    /// DAG nodes that participate in a detected matmul subgraph as the
    /// Expand or Mul intermediate. The emitter skips these when walking
    /// the DAG; the matmul kernel emits at the Sum node instead.
    matmul_consumed: UnordSet<usize>,
    /// Sum-node-id → MatmulInfo. emit_node looks up the MatmulInfo here
    /// when reaching the Sum and emits a matmul kernel + dispatch.
    matmuls: UnordMap<usize, blas::MatmulInfo>,
}

impl Emitter {
    fn new(func_name: &str) -> Self {
        Self {
            func_name: func_name.to_string(),
            body: Vec::new(),
            kernels: Vec::new(),
            plans: Vec::new(),
            matmul_consumed: UnordSet::new(),
            matmuls: UnordMap::new(),
        }
    }

    /// Build the deduped, lex-sorted `// span:` comment block for a node.
    /// Returns a vector of comment strings (each one a single line, no
    /// indent prefix). Used both by host-side emission (via
    /// `push_span_comments`) and by per-node MSL kernel string emission
    /// (where the comments are prepended inside the embedded kernel source
    /// so they survive into the runtime-compiled MSL).
    ///
    /// Per `spec/design/chelis_span_survival.md` §2.4 (S4):
    ///   * canonical `span_id` first (if present),
    ///   * then `merged_spans` lex-sorted (deduped against `span_id`).
    ///
    /// No-op when both fields are empty.
    fn span_comment_block(node: &DagNode) -> Vec<String> {
        let mut out = Vec::new();
        if node.span_id.is_none() && node.merged_spans.is_empty() {
            return out;
        }
        if let Some(canonical) = node.span_id.as_deref() {
            let safe = chelis_ir::span_sanitize::sanitize_for_comment(canonical);
            out.push(format!("// span: {safe}"));
        }
        let mut merged: Vec<&str> = node
            .merged_spans
            .iter()
            .map(String::as_str)
            .filter(|s| node.span_id.as_deref() != Some(*s))
            .collect();
        merged.sort();
        merged.dedup();
        for span in merged {
            let safe = chelis_ir::span_sanitize::sanitize_for_comment(span);
            out.push(format!("// span: {safe}"));
        }
        out
    }

    /// Push host-side `// span:` comment lines for a node onto `self.body`.
    /// Called at the top of every per-node emit_* method so the launch
    /// site is preceded by the span block. No-op for span-free nodes.
    fn push_span_comments(&mut self, node: &DagNode) {
        for line in Self::span_comment_block(node) {
            self.body.push(line);
        }
    }

    /// Prepend `// span:` comment lines (followed by a newline) to an
    /// MSL kernel source string. Used for per-node kernels — every Metal
    /// kernel is per-node (`k_unary_<id>`, `k_binary_<id>`, …), so this
    /// applies uniformly. No-op when the node carries no spans.
    fn prepend_span_comments_to_kernel_source(node: &DagNode, source: String) -> String {
        let block = Self::span_comment_block(node);
        if block.is_empty() {
            return source;
        }
        let mut out = String::new();
        for line in block {
            out.push_str(&line);
            out.push('\n');
        }
        out.push_str(&source);
        out
    }

    fn emit(&mut self, dag: VerifiedDagView<'_>) -> Result<(), String> {
        self.plans.resize(dag.nodes().len(), None);
        let inputs = input_labels(dag);
        let specs = output_specs(dag);
        let outputs: Vec<String> = specs.iter().map(|s| s.label.clone()).collect();

        // Pre-walk: detect matmul subgraphs and mark Expand/Mul intermediates
        // as "consumed" so the per-node walk skips them. This is the same
        // shape HIP uses (see chelis_backend_hip::emit::HipEmitter pattern).
        for (sum_id, info) in blas::find_all_matmuls(dag) {
            // Mark the two Expand nodes and the Mul node as consumed.
            // (The Sum itself stays — emit_node dispatches it to the matmul
            // emitter via self.matmuls lookup.)
            if let Some(sum_node) = dag.get(sum_id)
                && sum_node.inputs.len() == 1
            {
                let mul_id = sum_node.inputs[0];
                self.matmul_consumed.insert(mul_id.0);
                if let Some(mul_node) = dag.get(mul_id) {
                    for input in &mul_node.inputs {
                        self.matmul_consumed.insert(input.0);
                    }
                }
            }
            self.matmuls.insert(sum_id.0, info);
        }

        for node in dag.nodes() {
            // Skip Expand/Mul nodes that were folded into a matmul subgraph
            // — they have no standalone kernel emission.
            if self.matmul_consumed.contains(&node.id.0) {
                continue;
            }
            self.emit_node(dag, node, &inputs, &outputs)?;
        }

        // Materialize any output that wasn't already written via an
        // explicit `Store` node. This is the path the CLI's regular
        // lowering uses: a DAG with roots but no Store. Without this,
        // the emitted function would compute results into a device
        // buffer and return without ever writing `outputs[idx]`.
        for (idx, spec) in specs.iter().enumerate() {
            if spec.is_store {
                continue;
            }
            self.emit_root_writeback(idx, spec)?;
        }
        Ok(())
    }

    fn emit_root_writeback(&mut self, idx: usize, spec: &OutputSpec) -> Result<(), String> {
        let plan = self
            .plan_of(spec.id)
            .ok_or_else(|| {
                format!(
                    "Metal emit: root output `{}` (node {}) not materialized",
                    spec.label, spec.id.0
                )
            })?
            .clone();
        let dtype = dtype::runtime_dtype_tag(plan.prec);
        // Use host-safe sizeof: MSL `half` and `bfloat` are not visible
        // to host C++; route via `host_sizeof_expr` so the emitted .mm
        // links cleanly under `clang++ -fobjc-arc`.
        let bytes = format!("{}u * {}", plan.n, dtype::host_sizeof_expr(plan.prec));
        // `spec.label` is producer-supplied (Store name from LoadStoreName,
        // or `root{N}` synthesized internally). Comment-context sanitize
        // for the LoadStoreName case (synthesized labels are clean ASCII).
        let safe_label = chelis_ir::span_sanitize::sanitize_for_comment(&spec.label);
        self.body.push(format!(
            "// root output {idx} = `{safe_label}` (node {})",
            spec.id.0
        ));
        if plan.shape.is_empty() {
            // Scalar output — chelis_alloc(0, NULL, dtype) per chelis_runtime.h.
            self.body
                .push(format!("outputs[{idx}] = chelis_alloc(0, NULL, {dtype});"));
        } else {
            let shape_lit = plan
                .shape
                .iter()
                .map(|d| format!("(int64_t){d}"))
                .collect::<Vec<_>>()
                .join(", ");
            self.body.push(format!(
                "{{ int64_t shape_{idx}[{}] = {{ {shape_lit} }};",
                plan.shape.len()
            ));
            self.body.push(format!(
                "  outputs[{idx}] = chelis_alloc({}, shape_{idx}, {dtype}); }}",
                plan.shape.len()
            ));
        }
        self.body.push(format!(
            "chelis_tensor_write *root_guard_{idx} = chelis_tensor_begin_write(outputs[{idx}]);"
        ));
        self.body.push(format!(
            "chelis_write_view root_view_{idx} = chelis_tensor_write_view(root_guard_{idx});"
        ));
        self.body.push(format!(
            "chelis_metal_device_to_host(root_view_{idx}.data, {}, {bytes});",
            plan.buf
        ));
        self.body
            .push(format!("chelis_tensor_end_write(root_guard_{idx});"));
        Ok(())
    }

    fn into_source(self) -> String {
        let mut header = String::new();
        header.push_str("// Generated by chelis --target metal (M2)\n");
        header.push_str("#import \"chelis_metal_runtime.h\"\n");
        header.push_str("#include \"chelis_runtime.h\"\n");
        header.push_str("#include <stdint.h>\n");
        header.push_str("#include <stdio.h>\n");
        header.push_str("#include <stdlib.h>\n\n");

        // Kernel source string declarations, one C++11 raw string literal per kernel.
        // The cache key for chelis_metal_get_pipeline is the (NSString*, name) pair,
        // so each kernel must have a stable static NSString *const handle.
        for (var, _name, src) in &self.kernels {
            header.push_str(&format!(
                "static NSString *const {var}_src = @R\"MSL(\n{src})MSL\";\n\n"
            ));
        }

        let mut out = header;
        out.push_str(&format!(
            "extern \"C\" void {}(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out) {{\n",
            self.func_name
        ));
        out.push_str("    (void)n_in; (void)n_out;\n");
        for line in &self.body {
            out.push_str("    ");
            out.push_str(line);
            out.push('\n');
        }
        out.push_str("}\n");
        out
    }

    fn emit_node(
        &mut self,
        dag: VerifiedDagView<'_>,
        node: &DagNode,
        inputs: &[String],
        outputs: &[String],
    ) -> Result<(), String> {
        let id = node.id.0;
        match &node.op {
            RiscOp::Load { name } => self.emit_load(node, name.as_str(), inputs),
            RiscOp::Store { name } => self.emit_store(dag, node, name.as_str(), outputs),
            RiscOp::Const { value } => self.emit_const(node, value.as_f64_lossy()),
            RiscOp::Copy | RiscOp::Drop => Ok(()),

            // Lowering represents f16/bf16 matmul's required f32
            // accumulation followed by an explicit downcast. The existing
            // matmul dispatch performs that downcast and records the
            // operand-precision buffer on the Sum node, so bind the verified
            // Cast identity to that exact buffer.
            RiscOp::Cast { new_precision }
                if node.inputs.len() == 1 && self.matmuls.contains_key(&node.inputs[0].0) =>
            {
                let source = self.plan_of(node.inputs[0]).cloned().ok_or_else(|| {
                    format!(
                        "Metal matmul downcast node {id}: source node {} was not materialized",
                        node.inputs[0].0
                    )
                })?;
                if source.prec != *new_precision {
                    return Err(format!(
                        "Metal matmul downcast node {id}: emitted precision `{}` does not match cast target `{}`",
                        source.prec.name(),
                        new_precision.name()
                    ));
                }
                self.plans[id] = Some(source);
                Ok(())
            }

            // Unary elementwise (M2 first cut).
            RiscOp::Neg
            | RiscOp::Exp
            | RiscOp::Log
            | RiscOp::Sin
            | RiscOp::Sqrt
            | RiscOp::Cos
            | RiscOp::Tan
            | RiscOp::Atan
            | RiscOp::Abs
            | RiscOp::Floor
            | RiscOp::Ceil
            | RiscOp::Round => self.emit_unary(dag, node),

            // Binary elementwise (M2 first cut: add, mul).
            RiscOp::Add | RiscOp::Mul => self.emit_binary(dag, node),

            // chelis#1306: these identities are rejected by the shared typed
            // Metal capability gate. Keep explicit backend arms so no new
            // operation can fall through the generic unsupported wildcard.
            RiscOp::Sub | RiscOp::MaxElem | RiscOp::MinElem | RiscOp::ExtremaAdjoint { .. } => {
                Err(format!(
                    "Metal direct arithmetic node {id} reached emission after the #1306 typed capability rejection"
                ))
            }
            RiscOp::FusedElem { ops }
                if ops.iter().any(|step| {
                    matches!(
                        step.op,
                        chelis_ir::dag::FusedStepOp::Sub
                            | chelis_ir::dag::FusedStepOp::MaxElem
                            | chelis_ir::dag::FusedStepOp::MinElem
                    )
                }) =>
            {
                Err(format!(
                    "Metal fused direct arithmetic node {id} reached emission after the #1306 typed capability rejection"
                ))
            }

            // Full-axis rank-1 reductions to scalar. WS-M1 widens the
            // accumulator admission per spec/04-type-system.md §5.7.1
            // (f16/bf16 → f32; i8/i16 → i32; i32/i64 → same; f32 → f32).
            // Reading the accumulator field (not destructure-`..`) is the
            // memory-rule-pinned discipline so silent precision downgrades
            // surface as kernel-template panics rather than wrong answers.
            RiscOp::Sum {
                axis: 0,
                accumulator,
            } => self.emit_reduce(dag, node, kernels::ReduceKind::Sum, Some(*accumulator)),
            RiscOp::MaxReduce { axis: 0 } => {
                self.emit_reduce(dag, node, kernels::ReduceKind::Max, None)
            }
            RiscOp::MinReduce { axis: 0 } => {
                self.emit_reduce(dag, node, kernels::ReduceKind::Min, None)
            }

            // Matmul (Sum{axis:1} head of expand+mul+sum subgraph; M5).
            // Read the accumulator field rather than destructure-`..` so
            // silent precision drift can be caught here in defense in
            // depth alongside the F1 guard in `emit_matmul`.
            RiscOp::Sum {
                axis: 1,
                accumulator: _,
            } if self.matmuls.contains_key(&node.id.0) => {
                let info = self.matmuls[&node.id.0].clone();
                self.emit_matmul(node, &info)
            }

            // WS-8A: pad / shrink movement ops. Typed per-output-element
            // kernels (one thread per output element) over contiguous
            // buffers, mirroring the HIP path. The GPU==eval numeric proof
            // is the manual `gpu_correctness` Mac gate.
            RiscOp::Pad { padding, fill } => {
                self.emit_pad(node, &metal_pairs_to_usize(padding), *fill)
            }
            RiscOp::Shrink { bounds } => self.emit_shrink(node, &metal_pairs_to_usize(bounds)),

            other => Err(format!(
                "Metal M4 emit: node {id} op {other:?} not yet supported \
                 (axis>0 reductions, fused-elem-into-reduction, matmul, \
                 broadcasts/strides land in M4.next/M5)"
            )),
        }
    }

    /// Reject f64 with the FP64-ALU diagnostic per spec/04-type-system.md
    /// §1.1.3. Defense in depth: the CLI gate
    /// (`reject_unsupported_metal_ops`) and the IR validation pass also
    /// reject this case, so reaching here represents a contract drift.
    /// All three surfaces share the same diagnostic text.
    fn require_metal_admissible(prec: Prim, ctx: &str) -> Result<(), String> {
        match prec {
            Prim::F32
            | Prim::F16
            | Prim::Bf16
            | Prim::Int8
            | Prim::Int16
            | Prim::Int32
            | Prim::Int64
            | Prim::Bool => Ok(()),
            Prim::F64 => Err(format!(
                "Metal emit ({ctx}): Apple Silicon GPUs lack FP64 ALUs; \
                 use `--target c` or `--target hip` for f64 workloads. \
                 See spec/04-type-system.md §1.1.3."
            )),
            other => Err(format!(
                "Metal emit ({ctx}): precision `{}` is not in the active \
                 per-backend dtype matrix (spec/04-type-system.md §1.1.3)",
                other.name()
            )),
        }
    }

    fn require_static_rank1(&self, ty: &TensorType, ctx: &str) -> Result<(usize, Prim), String> {
        Self::require_metal_admissible(ty.precision, ctx)?;
        if ty.dims.len() != 1 {
            return Err(format!(
                "Metal M2 emit ({ctx}): rank {} not yet supported; M2 first cut handles rank-1 only",
                ty.dims.len()
            ));
        }
        let n = match &ty.dims[0] {
            DimInfo::Lit(n) => *n,
            other => {
                return Err(format!(
                    "Metal M2 emit ({ctx}): dim {other:?} not yet supported; rank-1 with literal extent only"
                ));
            }
        };
        Ok((n, ty.precision))
    }

    /// Accept rank-1 OR rank-2 with literal extents and any active Metal
    /// precision (per spec/04-type-system.md §1.1.3). Used for ops that
    /// work on both ranks (Load, Store, matmul operands, matmul output).
    /// Returns `(total_n, shape, prec)`.
    fn require_static_shape(
        &self,
        ty: &TensorType,
        ctx: &str,
    ) -> Result<(usize, Vec<usize>, Prim), String> {
        Self::require_metal_admissible(ty.precision, ctx)?;
        if ty.dims.is_empty() {
            return Err(format!(
                "Metal emit ({ctx}): rank-0 not supported via this path; use scalar Const/reduce instead"
            ));
        }
        if ty.dims.len() > 2 {
            return Err(format!(
                "Metal emit ({ctx}): rank {} not yet supported; rank-1 or rank-2 only in this phase",
                ty.dims.len()
            ));
        }
        let mut shape = Vec::with_capacity(ty.dims.len());
        for d in &ty.dims {
            match d {
                DimInfo::Lit(n) => shape.push(*n),
                other => {
                    return Err(format!(
                        "Metal emit ({ctx}): symbolic dim {other:?} not yet supported"
                    ));
                }
            }
        }
        let n: usize = shape.iter().product();
        Ok((n, shape, ty.precision))
    }

    fn emit_load(&mut self, node: &DagNode, name: &str, inputs: &[String]) -> Result<(), String> {
        // Loads are the boundary at which rank-2 enters. Use the rank-1-or-2
        // helper so a rank-2 input materializes its plan correctly; downstream
        // ops that don't yet handle rank-2 (unary/binary elementwise without
        // matching shape) will reject via require_static_rank1.
        let (n, shape, prec) = self.require_static_shape(&node.output_type, "Load")?;
        let idx = inputs
            .iter()
            .position(|l| l == name)
            .ok_or_else(|| format!("Load `{name}` not registered in input_labels"))?;
        let buf = format!("buf_{}", node.id.0);
        // Host-safe sizeof: see `dtype::host_sizeof_expr` (MSL `half`/`bfloat`
        // are not visible to host C++).
        let bytes = format!("{n}u * {}", dtype::host_sizeof_expr(prec));
        self.push_span_comments(node);
        // Producer-supplied `name` (LoadStoreName) flows into a `// ...`
        // comment context. Even though LoadStoreName's constructor enforces
        // identifier-grammar (so newlines / NUL / DEL cannot reach here),
        // route through the shared comment-context sanitizer to lock the
        // architectural pattern from
        // spec/upstream-bugs/producer-string-sanitization.md: every
        // producer-supplied string into a comment context goes through
        // `sanitize_for_comment`.
        let safe_name = chelis_ir::span_sanitize::sanitize_for_comment(name);
        self.body
            .push(format!("// node {} = Load {safe_name}", node.id.0));
        self.body.push(format!(
            "id<MTLBuffer> {buf} = chelis_metal_alloc({bytes});"
        ));
        self.body.push(format!(
            "chelis_metal_host_to_device({buf}, chelis_tensor_read_view(inputs[{idx}]).data, {bytes});"
        ));
        self.plans[node.id.0] = Some(TensorPlan {
            buf,
            prec,
            n,
            shape,
        });
        Ok(())
    }

    fn emit_const(&mut self, node: &DagNode, value: f64) -> Result<(), String> {
        let (n, prec) = self.require_static_rank1(&node.output_type, "Const")?;
        let buf = format!("buf_{}", node.id.0);
        // Both the byte-count and the fill body route through host-safe
        // helpers in `dtype` (MSL `half`/`bfloat` are not visible to host
        // C++ and would fail to compile under `clang++ -fobjc-arc`). The
        // helper picks a typed `float*` for F32, `uint16_t*` + IEEE-754
        // bit-pattern literal for F16/Bf16, matching `intN_t*` for the
        // integer family, and `bool*` for Bool. See
        // `dtype::host_const_fill_body` for the per-dtype contract.
        let bytes = format!("{n}u * {}", dtype::host_sizeof_expr(prec));
        self.push_span_comments(node);
        self.body
            .push(format!("// node {} = Const {value}", node.id.0));
        self.body.push(format!(
            "id<MTLBuffer> {buf} = chelis_metal_alloc({bytes});"
        ));
        // Fill via a small CPU loop; small enough for M2's static-extent
        // first cut. M4 will switch to a fused fill MSL kernel when scale
        // matters.
        self.body
            .push(dtype::host_const_fill_body(prec, value, &buf, n));
        self.plans[node.id.0] = Some(TensorPlan {
            buf,
            prec,
            n,
            shape: vec![n],
        });
        Ok(())
    }

    fn emit_unary(&mut self, dag: VerifiedDagView<'_>, node: &DagNode) -> Result<(), String> {
        let in_id = *node
            .inputs
            .first()
            .ok_or_else(|| format!("unary node {} has no input", node.id.0))?;
        let in_plan = self
            .plan_of(in_id)
            .ok_or_else(|| {
                format!(
                    "unary node {} input {} not materialized",
                    node.id.0, in_id.0
                )
            })?
            .clone();
        let (n, prec) = self.require_static_rank1(&node.output_type, "unary")?;
        let msl_ty = dtype::msl_type(prec);
        if in_plan.n != n || in_plan.prec != prec {
            return Err(format!(
                "Metal M2 emit unary node {}: input shape {:?}/{} != output shape [{}]/{} \
                 (no implicit broadcast/cast in M2 first cut)",
                node.id.0,
                in_plan.shape,
                in_plan.msl_ty(),
                n,
                msl_ty,
            ));
        }
        // Transcendental ops on integer dtypes are spec-rejected at the
        // type checker (§5.4 — Transcendental requires float). Defense in
        // depth: surface a clear codegen error if one ever leaks through.
        let needs_float = matches!(
            &node.op,
            RiscOp::Exp
                | RiscOp::Log
                | RiscOp::Sin
                | RiscOp::Cos
                | RiscOp::Tan
                | RiscOp::Atan
                | RiscOp::Sqrt
        );
        if needs_float && !matches!(prec, Prim::F32 | Prim::F16 | Prim::Bf16) {
            return Err(format!(
                "Metal emit unary node {}: transcendental op {:?} requires float precision; \
                 got `{}`. See spec/04-type-system.md §5.4.",
                node.id.0,
                node.op,
                prec.name()
            ));
        }
        let suffix = dtype::kernel_suffix(prec);
        let kernel_name = format!("k_unary{suffix}_{}", node.id.0);
        let pso_var = format!("pso_{}", node.id.0);
        let body = match &node.op {
            RiscOp::Neg => "    out[tid] = -a[tid];".to_string(),
            other => {
                let f = kernels::unary_func(other).ok_or_else(|| {
                    format!(
                        "Metal M2 emit unary node {}: op {:?} unsupported",
                        node.id.0, other
                    )
                })?;
                format!("    out[tid] = {f}(a[tid]);")
            }
        };
        let params = vec![
            kernels::input_param(0, msl_ty, "a"),
            kernels::output_param(1, msl_ty, "out"),
        ];
        let src = kernels::elementwise_kernel_for(&kernel_name, &params, &body, &[prec]);
        let src = Self::prepend_span_comments_to_kernel_source(node, src);
        self.kernels
            .push((pso_var.clone(), kernel_name.clone(), src));

        let out_buf = format!("buf_{}", node.id.0);
        // Host-safe sizeof: see `dtype::host_sizeof_expr`.
        let bytes = format!("{n}u * {}", dtype::host_sizeof_expr(prec));
        self.push_span_comments(node);
        self.body
            .push(format!("// node {} = unary {:?}", node.id.0, node.op));
        self.body.push(format!(
            "id<MTLBuffer> {out_buf} = chelis_metal_alloc({bytes});"
        ));
        self.body.push(format!(
            "id<MTLComputePipelineState> {pso_var} = chelis_metal_get_pipeline({pso_var}_src, @\"{kernel_name}\");"
        ));
        self.body.push(format!("uint32_t n_{} = {n}u;", node.id.0));
        self.body.push(format!(
            "{{ __unsafe_unretained id<MTLBuffer> bufs[2] = {{ {}, {out_buf} }}; chelis_metal_launch({pso_var}, {n}u, MIN((NSUInteger){n}u, 256u), bufs, 2, &n_{}, sizeof(uint32_t)); }}",
            in_plan.buf,
            node.id.0
        ));
        self.plans[node.id.0] = Some(TensorPlan {
            buf: out_buf,
            prec,
            n,
            shape: vec![n],
        });
        let _ = dag;
        Ok(())
    }

    fn emit_binary(&mut self, dag: VerifiedDagView<'_>, node: &DagNode) -> Result<(), String> {
        if node.inputs.len() != 2 {
            return Err(format!(
                "binary node {} has {} inputs (expected 2)",
                node.id.0,
                node.inputs.len()
            ));
        }
        let a_id = node.inputs[0];
        let b_id = node.inputs[1];
        let a_plan = self
            .plan_of(a_id)
            .ok_or_else(|| format!("binary node {} lhs {} not materialized", node.id.0, a_id.0))?
            .clone();
        let b_plan = self
            .plan_of(b_id)
            .ok_or_else(|| format!("binary node {} rhs {} not materialized", node.id.0, b_id.0))?
            .clone();
        let (n, prec) = self.require_static_rank1(&node.output_type, "binary")?;
        let msl_ty = dtype::msl_type(prec);
        if a_plan.n != n || b_plan.n != n || a_plan.prec != prec || b_plan.prec != prec {
            return Err(format!(
                "Metal M2 emit binary node {}: shape/type mismatch (M2 first cut: same-shape contiguous only)",
                node.id.0
            ));
        }
        let op = kernels::binary_op(&node.op).ok_or_else(|| {
            format!(
                "Metal M2 emit binary node {}: op {:?} unsupported",
                node.id.0, node.op
            )
        })?;
        let suffix = dtype::kernel_suffix(prec);
        let kernel_name = format!("k_binary{suffix}_{}", node.id.0);
        let pso_var = format!("pso_{}", node.id.0);
        let params = vec![
            kernels::input_param(0, msl_ty, "a"),
            kernels::input_param(1, msl_ty, "b"),
            kernels::output_param(2, msl_ty, "out"),
        ];
        let body = format!("    out[tid] = a[tid] {op} b[tid];");
        let src = kernels::elementwise_kernel_for(&kernel_name, &params, &body, &[prec]);
        let src = Self::prepend_span_comments_to_kernel_source(node, src);
        self.kernels
            .push((pso_var.clone(), kernel_name.clone(), src));

        let out_buf = format!("buf_{}", node.id.0);
        // Host-safe sizeof: see `dtype::host_sizeof_expr`.
        let bytes = format!("{n}u * {}", dtype::host_sizeof_expr(prec));
        self.push_span_comments(node);
        self.body
            .push(format!("// node {} = binary {:?}", node.id.0, node.op));
        self.body.push(format!(
            "id<MTLBuffer> {out_buf} = chelis_metal_alloc({bytes});"
        ));
        self.body.push(format!(
            "id<MTLComputePipelineState> {pso_var} = chelis_metal_get_pipeline({pso_var}_src, @\"{kernel_name}\");"
        ));
        self.body.push(format!("uint32_t n_{} = {n}u;", node.id.0));
        self.body.push(format!(
            "{{ __unsafe_unretained id<MTLBuffer> bufs[3] = {{ {}, {}, {out_buf} }}; chelis_metal_launch({pso_var}, {n}u, MIN((NSUInteger){n}u, 256u), bufs, 3, &n_{}, sizeof(uint32_t)); }}",
            a_plan.buf,
            b_plan.buf,
            node.id.0
        ));
        self.plans[node.id.0] = Some(TensorPlan {
            buf: out_buf,
            prec,
            n,
            shape: vec![n],
        });
        let _ = dag;
        Ok(())
    }

    /// Emit a full-axis reduction kernel for a rank-1 input to a rank-0
    /// scalar output.
    ///
    /// `accumulator_override` carries the IR's pinned accumulator for
    /// `reduce_sum` (per spec §5.7.1). For `max`/`min` reductions the
    /// accumulator equals the operand precision (no upgrade) and
    /// `accumulator_override` is `None`.
    fn emit_reduce(
        &mut self,
        _dag: VerifiedDagView<'_>,
        node: &DagNode,
        kind: kernels::ReduceKind,
        accumulator_override: Option<Prim>,
    ) -> Result<(), String> {
        let in_id = *node
            .inputs
            .first()
            .ok_or_else(|| format!("reduce node {} has no input", node.id.0))?;
        let in_plan = self
            .plan_of(in_id)
            .ok_or_else(|| {
                format!(
                    "reduce node {} input {} not materialized",
                    node.id.0, in_id.0
                )
            })?
            .clone();
        // Full-axis reduce to scalar, rank-1 contiguous input.
        if in_plan.shape.len() != 1 {
            return Err(format!(
                "Metal M4 emit reduce node {}: rank {} input not yet supported \
                 (rank-1 first cut)",
                node.id.0,
                in_plan.shape.len()
            ));
        }
        Self::require_metal_admissible(in_plan.prec, "reduce input")?;
        // Output should be rank-0 with the spec-required accumulator
        // precision (per §5.7.1 for sums; same as operand for max/min).
        let expected_acc = match (kind, accumulator_override) {
            (kernels::ReduceKind::Sum, Some(acc)) => acc,
            (kernels::ReduceKind::Sum, None) => dtype::sum_accumulator(in_plan.prec),
            (kernels::ReduceKind::Max | kernels::ReduceKind::Min, _) => in_plan.prec,
        };
        Self::require_metal_admissible(expected_acc, "reduce accumulator")?;
        if !node.output_type.dims.is_empty() || node.output_type.precision != expected_acc {
            return Err(format!(
                "Metal M4 emit reduce node {}: output must be rank-0 `{}`, got {:?}",
                node.id.0,
                expected_acc.name(),
                node.output_type
            ));
        }
        let n = in_plan.n;
        // Single-threadgroup reduction with a fixed power-of-two TG_SIZE
        // (see kernels::REDUCE_TG_SIZE). The kernel's wrap loop strides
        // TG_SIZE elements per iteration, so every n up to TG_SIZE *
        // TG_SIZE = 65536 reduces in one pass. We cap at 4096 for now
        // because this M4 first cut hasn't been benchmarked above that;
        // two-pass reduction for larger inputs lands in M4.next. Reject
        // here so the oracle never silently runs a wrong reduction.
        const SINGLE_TG_LIMIT: usize = 4096;
        if n > SINGLE_TG_LIMIT {
            return Err(format!(
                "Metal M4 emit reduce node {}: n={} exceeds single-threadgroup limit {} \
                 (two-pass reduction is M4.next)",
                node.id.0, n, SINGLE_TG_LIMIT
            ));
        }
        let in_suffix = dtype::kernel_suffix(in_plan.prec);
        let acc_suffix = dtype::kernel_suffix(expected_acc);
        let kernel_name = format!(
            "k_reduce_{}{in_suffix}{acc_suffix}_{}",
            kind.label(),
            node.id.0
        );
        let pso_var = format!("pso_{}", node.id.0);
        let src = kernels::reduce_full_kernel_for(&kernel_name, kind, in_plan.prec, expected_acc);
        let src = Self::prepend_span_comments_to_kernel_source(node, src);
        self.kernels
            .push((pso_var.clone(), kernel_name.clone(), src));

        let out_buf = format!("buf_{}", node.id.0);
        // Output is one accumulator-typed scalar; host-safe sizeof per
        // `dtype::host_sizeof_expr` (panics on f64; host-safe for the
        // active per-backend dtype matrix).
        let bytes = format!("1u * {}", dtype::host_sizeof_expr(expected_acc));
        self.push_span_comments(node);
        self.body
            .push(format!("// node {} = reduce_{}", node.id.0, kind.label()));
        self.body.push(format!(
            "id<MTLBuffer> {out_buf} = chelis_metal_alloc({bytes});"
        ));
        self.body.push(format!(
            "id<MTLComputePipelineState> {pso_var} = chelis_metal_get_pipeline({pso_var}_src, @\"{kernel_name}\");"
        ));
        self.body.push(format!("uint32_t n_{} = {n}u;", node.id.0));
        // Always dispatch a single TG_SIZE-thread threadgroup. The kernel
        // strides over n internally; mismatching tg with n was the source
        // of a critical non-power-of-2 miscompile that the previous fix
        // (clamping tg = n.min(256)) shipped; fixed here by decoupling.
        let tg = kernels::REDUCE_TG_SIZE;
        self.body.push(format!(
            "{{ __unsafe_unretained id<MTLBuffer> bufs[2] = {{ {}, {out_buf} }}; chelis_metal_launch({pso_var}, {tg}u, {tg}u, bufs, 2, &n_{}, sizeof(uint32_t)); }}",
            in_plan.buf,
            node.id.0
        ));
        // Reduction returns a rank-0 scalar in IR semantics. Reflect
        // that in the plan so emit_root_writeback emits chelis_alloc(0,
        // NULL, <accumulator>) and the resulting tensor has ndim=0,
        // matching the IR's contract. The buffer itself still holds 1
        // accumulator-typed element on the device.
        self.plans[node.id.0] = Some(TensorPlan {
            buf: out_buf,
            prec: expected_acc,
            n: 1,
            shape: vec![],
        });
        Ok(())
    }

    fn emit_matmul(&mut self, node: &DagNode, info: &blas::MatmulInfo) -> Result<(), String> {
        // Operands must already be materialized (their Load/Const/etc.
        // emitted earlier in the topological walk; the Expand intermediates
        // were skipped because they're in matmul_consumed).
        let a_plan = self
            .plan_of(info.a)
            .ok_or_else(|| {
                format!(
                    "Metal M5 matmul node {}: lhs operand node {} not materialized",
                    node.id.0, info.a.0
                )
            })?
            .clone();
        let b_plan = self
            .plan_of(info.b)
            .ok_or_else(|| {
                format!(
                    "Metal M5 matmul node {}: rhs operand node {} not materialized",
                    node.id.0, info.b.0
                )
            })?
            .clone();
        if a_plan.shape.len() != 2 || b_plan.shape.len() != 2 {
            return Err(format!(
                "Metal M5 matmul node {}: operand ranks must be 2, got lhs={:?} rhs={:?}",
                node.id.0, a_plan.shape, b_plan.shape
            ));
        }
        if a_plan.prec != b_plan.prec || a_plan.prec != info.precision {
            return Err(format!(
                "Metal matmul node {}: operand precision mismatch (lhs `{}`, rhs `{}`, info `{}`); \
                 spec/04-type-system.md §5.4 forbids implicit precision promotion",
                node.id.0,
                a_plan.prec.name(),
                b_plan.prec.name(),
                info.precision.name()
            ));
        }
        // F1 tactical guard: integer matmul is rejected at type-check
        // (Wave-2-Fixups B6, spec/04-type-system.md §5.7.2). Defense in
        // depth here so a future regression that admits integer matmul
        // surfaces a structured codegen error instead of a kernel
        // template that silently truncates.
        if !matches!(info.precision, Prim::F32 | Prim::F16 | Prim::Bf16) {
            return Err(format!(
                "F1: Metal-backend BlasMatmul currently supports only float \
                 precisions (f32/f16/bf16); node {} has operand precision `{}`. \
                 Integer matmul is rejected at type-check per \
                 spec/04-type-system.md §5.7.2 and should never reach codegen.",
                node.id.0,
                info.precision.name()
            ));
        }

        let prec = info.precision;
        let elem_bytes = dtype::metal_elem_size(prec);
        let out_n = info.m * info.n;
        let out_buf = format!("buf_{}", node.id.0);
        self.push_span_comments(node);
        self.body.push(format!(
            "// node {} = matmul {}x{}*{}x{} ({})",
            node.id.0,
            info.m,
            info.k,
            info.k,
            info.n,
            prec.name()
        ));
        self.body.push(format!(
            "id<MTLBuffer> {out_buf} = chelis_metal_alloc({out_n}u * {elem_bytes}u);"
        ));

        match prec {
            Prim::F32 => {
                // MPSMatrixMultiplication for f32. Per the WS-M0 ARC
                // ownership pin, the helper constructs and releases all
                // MPS objects internally and wraps the dispatch in
                // `@autoreleasepool { ... }`.
                self.body.push(format!(
                    "chelis_metal_mps_gemm_f32({}, {}, {out_buf}, {}u, {}u, {}u);",
                    a_plan.buf, b_plan.buf, info.m, info.n, info.k
                ));
            }
            Prim::F16 => {
                self.body.push(format!(
                    "chelis_metal_mps_gemm_f16({}, {}, {out_buf}, {}u, {}u, {}u);",
                    a_plan.buf, b_plan.buf, info.m, info.n, info.k
                ));
            }
            Prim::Bf16 => {
                // bf16 falls back to the tiled MSL kernel because MPS
                // does not expose an Apple7+-only bfloat GEMM in the
                // public API on every shipping toolchain. The kernel
                // template gates on `__METAL_VERSION__ >= 320` so the
                // emitted source remains valid on every toolchain.
                //
                // Per spec/04-type-system.md §5.7.1, bf16 matmul
                // accumulates in f32: the kernel template selects
                // `float acc` and casts both tile operands to f32 at
                // the multiply (operand-precision multiplication
                // truncates before accumulation, even with an f32
                // running sum). `dtype::matmul_accumulator(prec)`
                // selects the spec-pinned accumulator from the operand
                // precision so a future MatmulInfo extension that
                // carries an explicit accumulator can flow in without
                // changing the kernel template.
                let suffix = dtype::kernel_suffix(prec);
                let kernel_name = format!("k_matmul{suffix}_{}", node.id.0);
                let pso_var = format!("pso_{}", node.id.0);
                let acc_prec = dtype::matmul_accumulator(prec);
                let src = kernels::matmul_tiled_kernel_with_acc(&kernel_name, prec, acc_prec);
                let src = Self::prepend_span_comments_to_kernel_source(node, src);
                self.kernels
                    .push((pso_var.clone(), kernel_name.clone(), src));
                // F3: bf16 dispatch sites must guard against pre-Apple7
                // GPU families per spec/04-type-system.md §1.1.3. The
                // helper aborts with the spec-pinned diagnostic at the
                // first dispatch (cached internally so subsequent
                // dispatches are O(1)).
                self.body.push(
                    "chelis_metal_require_bf16_capability(chelis_metal_device());".to_string(),
                );
                self.body.push(format!(
                    "id<MTLComputePipelineState> {pso_var} = chelis_metal_get_pipeline({pso_var}_src, @\"{kernel_name}\");"
                ));
                self.body.push(format!(
                    "{{ struct {{ uint32_t M, N, K; }} mm_uniforms_{} = {{ {}u, {}u, {}u }};",
                    node.id.0, info.m, info.n, info.k
                ));
                self.body.push(format!(
                    "  __unsafe_unretained id<MTLBuffer> bufs[3] = {{ {}, {}, {out_buf} }};",
                    a_plan.buf, b_plan.buf
                ));
                let tile = kernels::MATMUL_TILE;
                let grid_y = info.m.div_ceil(tile) * tile;
                let grid_x = info.n.div_ceil(tile) * tile;
                self.body.push(format!(
                    "  chelis_metal_launch2d({pso_var}, {grid_x}u, {grid_y}u, {tile}u, {tile}u, bufs, 3, &mm_uniforms_{}, sizeof(mm_uniforms_{})); }}",
                    node.id.0, node.id.0
                ));
            }
            other => {
                return Err(format!(
                    "Metal matmul node {}: unsupported precision `{}` reached the \
                     dispatch arm; the F1 guard upstream should have caught this",
                    node.id.0,
                    other.name()
                ));
            }
        }

        self.plans[node.id.0] = Some(TensorPlan {
            buf: out_buf,
            prec,
            n: out_n,
            shape: vec![info.m, info.n],
        });
        Ok(())
    }

    /// Resolve a tensor type to `(shape, prec)` for the movement-op path,
    /// admitting any rank up to `MOVEMENT_MAX_DIM` with literal extents.
    /// Wider than `require_static_shape` (which caps at rank 2) because
    /// pad/shrink kernels iterate per-axis at runtime.
    fn require_movement_shape(
        &self,
        ty: &TensorType,
        ctx: &str,
    ) -> Result<(Vec<usize>, Prim), String> {
        Self::require_metal_admissible(ty.precision, ctx)?;
        if ty.dims.is_empty() {
            return Err(format!(
                "Metal emit ({ctx}): rank-0 not supported on the movement-op path"
            ));
        }
        if ty.dims.len() > kernels::MOVEMENT_MAX_DIM {
            return Err(format!(
                "Metal emit ({ctx}): rank {} exceeds MOVEMENT_MAX_DIM ({})",
                ty.dims.len(),
                kernels::MOVEMENT_MAX_DIM
            ));
        }
        let mut shape = Vec::with_capacity(ty.dims.len());
        for d in &ty.dims {
            match d {
                DimInfo::Lit(n) => shape.push(*n),
                other => {
                    return Err(format!(
                        "Metal emit ({ctx}): symbolic dim {other:?} not yet supported on the \
                         movement-op path (literal extents only)"
                    ));
                }
            }
        }
        Ok((shape, ty.precision))
    }

    /// Emit the `ChelisMovementDims` uniform initializer for a pad/shrink
    /// node. `offset[d]` is the per-axis low padding (pad) or start bound
    /// (shrink); the arrays are zero-padded up to `MOVEMENT_MAX_DIM`.
    fn movement_dims_initializer(
        node_id: usize,
        src_shape: &[usize],
        out_shape: &[usize],
        offset: &[usize],
        total: usize,
    ) -> String {
        let max_dim = kernels::MOVEMENT_MAX_DIM;
        let pad_array = |v: &[usize]| -> String {
            (0..max_dim)
                .map(|i| format!("{}u", v.get(i).copied().unwrap_or(0)))
                .collect::<Vec<_>>()
                .join(", ")
        };
        format!(
            "struct {{ uint ndim; uint total; uint src_shape[{max_dim}]; uint out_shape[{max_dim}]; uint offset[{max_dim}]; }} mv_dims_{node_id} = {{ {ndim}u, {total}u, {{ {src} }}, {{ {out} }}, {{ {off} }} }}",
            ndim = src_shape.len(),
            src = pad_array(src_shape),
            out = pad_array(out_shape),
            off = pad_array(offset),
        )
    }

    fn emit_pad(
        &mut self,
        node: &DagNode,
        padding: &[(usize, usize)],
        fill: ScalarValue,
    ) -> Result<(), String> {
        let in_id = *node
            .inputs
            .first()
            .ok_or_else(|| format!("pad node {} has no input", node.id.0))?;
        let in_plan = self
            .plan_of(in_id)
            .ok_or_else(|| format!("pad node {} input {} not materialized", node.id.0, in_id.0))?
            .clone();
        let (out_shape, prec) = self.require_movement_shape(&node.output_type, "pad")?;
        if in_plan.prec != prec {
            return Err(format!(
                "Metal emit pad node {}: input precision `{}` != output `{}` (pad preserves precision)",
                node.id.0,
                in_plan.prec.name(),
                prec.name()
            ));
        }
        if padding.len() != out_shape.len() || in_plan.shape.len() != out_shape.len() {
            return Err(format!(
                "Metal emit pad node {}: rank mismatch (padding {}, in {:?}, out {:?})",
                node.id.0,
                padding.len(),
                in_plan.shape,
                out_shape
            ));
        }
        let lo: Vec<usize> = padding.iter().map(|&(before, _)| before).collect();
        let out_n: usize = out_shape.iter().product();
        let msl_ty = dtype::msl_type(prec);
        let suffix = dtype::kernel_suffix(prec);
        let kernel_name = format!("k_pad{suffix}_{}", node.id.0);
        let pso_var = format!("pso_{}", node.id.0);
        let src = kernels::pad_kernel(&kernel_name, prec);
        let src = Self::prepend_span_comments_to_kernel_source(node, src);
        self.kernels
            .push((pso_var.clone(), kernel_name.clone(), src));

        let out_buf = format!("buf_{}", node.id.0);
        let bytes = format!("{out_n}u * {}", dtype::host_sizeof_expr(prec));
        self.push_span_comments(node);
        self.body
            .push(format!("// node {} = pad {:?}", node.id.0, padding));
        self.body.push(format!(
            "id<MTLBuffer> {out_buf} = chelis_metal_alloc({bytes});"
        ));
        self.body.push(format!(
            "id<MTLComputePipelineState> {pso_var} = chelis_metal_get_pipeline({pso_var}_src, @\"{kernel_name}\");"
        ));
        self.body.push(format!(
            "{};",
            Self::movement_dims_initializer(node.id.0, &in_plan.shape, &out_shape, &lo, out_n)
        ));
        // Pad fill is bound as a separate one-element constant buffer so the
        // dtype matches the buffer element type exactly. Host-side the value
        // is the typed scalar; MSL reads `constant T& fill`.
        self.body.push(format!(
            "{msl_ty} pad_fill_{} = ({msl_ty}){};",
            node.id.0,
            Self::host_scalar_literal(prec, fill)?
        ));
        self.body.push(format!(
            "{{ __unsafe_unretained id<MTLBuffer> bufs[2] = {{ {}, {out_buf} }}; \
             chelis_metal_launch_two_uniforms({pso_var}, {out_n}u, MIN((NSUInteger){out_n}u, 256u), bufs, 2, &mv_dims_{}, sizeof(mv_dims_{}), &pad_fill_{}, sizeof({msl_ty})); }}",
            in_plan.buf, node.id.0, node.id.0, node.id.0,
        ));
        self.plans[node.id.0] = Some(TensorPlan {
            buf: out_buf,
            prec,
            n: out_n,
            shape: out_shape,
        });
        Ok(())
    }

    fn emit_shrink(&mut self, node: &DagNode, bounds: &[(usize, usize)]) -> Result<(), String> {
        let in_id = *node
            .inputs
            .first()
            .ok_or_else(|| format!("shrink node {} has no input", node.id.0))?;
        let in_plan = self
            .plan_of(in_id)
            .ok_or_else(|| {
                format!(
                    "shrink node {} input {} not materialized",
                    node.id.0, in_id.0
                )
            })?
            .clone();
        let (out_shape, prec) = self.require_movement_shape(&node.output_type, "shrink")?;
        if in_plan.prec != prec {
            return Err(format!(
                "Metal emit shrink node {}: input precision `{}` != output `{}` (shrink preserves precision)",
                node.id.0,
                in_plan.prec.name(),
                prec.name()
            ));
        }
        if bounds.len() != out_shape.len() || in_plan.shape.len() != out_shape.len() {
            return Err(format!(
                "Metal emit shrink node {}: rank mismatch (bounds {}, in {:?}, out {:?})",
                node.id.0,
                bounds.len(),
                in_plan.shape,
                out_shape
            ));
        }
        let start: Vec<usize> = bounds.iter().map(|&(s, _)| s).collect();
        let out_n: usize = out_shape.iter().product();
        let suffix = dtype::kernel_suffix(prec);
        let kernel_name = format!("k_shrink{suffix}_{}", node.id.0);
        let pso_var = format!("pso_{}", node.id.0);
        let src = kernels::shrink_kernel(&kernel_name, prec);
        let src = Self::prepend_span_comments_to_kernel_source(node, src);
        self.kernels
            .push((pso_var.clone(), kernel_name.clone(), src));

        let out_buf = format!("buf_{}", node.id.0);
        let bytes = format!("{out_n}u * {}", dtype::host_sizeof_expr(prec));
        self.push_span_comments(node);
        self.body
            .push(format!("// node {} = shrink {:?}", node.id.0, bounds));
        self.body.push(format!(
            "id<MTLBuffer> {out_buf} = chelis_metal_alloc({bytes});"
        ));
        self.body.push(format!(
            "id<MTLComputePipelineState> {pso_var} = chelis_metal_get_pipeline({pso_var}_src, @\"{kernel_name}\");"
        ));
        self.body.push(format!(
            "{};",
            Self::movement_dims_initializer(node.id.0, &in_plan.shape, &out_shape, &start, out_n)
        ));
        self.body.push(format!(
            "{{ __unsafe_unretained id<MTLBuffer> bufs[2] = {{ {}, {out_buf} }}; \
             chelis_metal_launch({pso_var}, {out_n}u, MIN((NSUInteger){out_n}u, 256u), bufs, 2, &mv_dims_{}, sizeof(mv_dims_{})); }}",
            in_plan.buf, node.id.0, node.id.0,
        ));
        self.plans[node.id.0] = Some(TensorPlan {
            buf: out_buf,
            prec,
            n: out_n,
            shape: out_shape,
        });
        Ok(())
    }

    /// Host-side literal spelling of an `f64` IR scalar for a typed Metal
    /// constant. Floats keep their decimal form (the M-phase tolerance
    /// model already accepts f32 fast-math drift; the exact-bit-pattern
    /// refinement is HIP-side via `chelis_f32_from_bits`); integers and
    /// bool cast directly.
    /// chelis#730 Phase 1 (census row 19, chelis#745): the former
    /// catch-all emitted `/* unsupported pad fill dtype */ 0` - a silent
    /// zero substituted into the pad constant. The remaining precisions
    /// (f64 behind the Metal gate, the deferred f8e4m3, string) are a
    /// section C2 diagnostic through this emitter's existing String error
    /// channel; exhaustive per section C4.1 - no wildcard arm.
    fn host_scalar_literal(prec: Prim, value: ScalarValue) -> Result<String, String> {
        if value.prim() != prec {
            return Err(format!(
                "Metal pad fill dtype `{}` does not match output `{}`",
                value.prim().name(),
                prec.name()
            ));
        }
        Ok(match prec {
            Prim::F32 => format!("{:?}f", value.as_f64_lossy()),
            Prim::F16 | Prim::Bf16 => format!("{:?}", value.as_f64_lossy()),
            Prim::Bool => (if value.as_bool_exact().unwrap_or(false) {
                "true"
            } else {
                "false"
            })
            .to_string(),
            Prim::Int8 | Prim::Int16 | Prim::Int32 => {
                value.as_i64_exact().expect("integer pad fill").to_string()
            }
            Prim::Int64 => {
                let value = value.as_i64_exact().expect("int64 pad fill");
                if value == i64::MIN {
                    "(-9223372036854775807LL - 1LL)".to_string()
                } else {
                    format!("{value}LL")
                }
            }
            Prim::F64 => {
                return Err(Unsupported::new(
                    UnsupportedKind::Dtype(prec.name().to_string()),
                    "a Metal pad-fill host scalar literal",
                    Stage::Codegen("metal"),
                    chelis_types::deliberate_rejection!(
                        "[04-TGT-1]",
                        "the Metal backend rejects f64 because Apple GPUs have no FP64 ALUs; use the C or HIP target"
                    ),
                )
                .to_string());
            }
            Prim::F8e4m3 => {
                return Err(Unsupported::new(
                    UnsupportedKind::Dtype(prec.name().to_string()),
                    "a Metal pad-fill host scalar literal",
                    Stage::Codegen("metal"),
                    chelis_types::deliberate_rejection!(
                        "[04-DTYPE-1]",
                        "f8e4m3 is a reserved but inactive dtype and must be rejected before emission"
                    ),
                )
                .to_string());
            }
            Prim::String => {
                return Err(Unsupported::new(
                    UnsupportedKind::Dtype(prec.name().to_string()),
                    "a Metal pad-fill host scalar literal",
                    Stage::Codegen("metal"),
                    chelis_types::unimplemented_rejection!(
                        729,
                        "the exhaustive target capability table has not implemented a Metal string storage cell"
                    ),
                )
                .to_string());
            }
        })
    }

    fn emit_store(
        &mut self,
        _dag: VerifiedDagView<'_>,
        node: &DagNode,
        name: &str,
        outputs: &[String],
    ) -> Result<(), String> {
        let in_id = *node
            .inputs
            .first()
            .ok_or_else(|| format!("Store `{name}` has no input"))?;
        let in_plan = self
            .plan_of(in_id)
            .ok_or_else(|| format!("Store `{name}`: input node {} not materialized", in_id.0))?
            .clone();
        let idx = outputs
            .iter()
            .position(|l| l == name)
            .ok_or_else(|| format!("Store `{name}` not registered in output_labels"))?;
        // Host-safe sizeof: see `dtype::host_sizeof_expr` (MSL `half`/`bfloat`
        // are not visible to host C++).
        let bytes = format!("{}u * {}", in_plan.n, dtype::host_sizeof_expr(in_plan.prec));
        let dtype = dtype::runtime_dtype_tag(in_plan.prec);
        self.push_span_comments(node);
        // Comment-context sanitization for producer-supplied Store name.
        // See `emit_load` for the architectural-pattern rationale.
        let safe_name = chelis_ir::span_sanitize::sanitize_for_comment(name);
        self.body
            .push(format!("// node {} = Store `{safe_name}`", node.id.0));
        if in_plan.shape.is_empty() {
            // Rank-0 scalar Store: chelis_alloc takes (0, NULL, dtype).
            self.body
                .push(format!("outputs[{idx}] = chelis_alloc(0, NULL, {dtype});"));
        } else {
            let shape_lit = in_plan
                .shape
                .iter()
                .map(|d| format!("(int64_t){d}"))
                .collect::<Vec<_>>()
                .join(", ");
            self.body.push(format!(
                "{{ int64_t shape_store_{idx}[{}] = {{ {shape_lit} }};",
                in_plan.shape.len()
            ));
            self.body.push(format!(
                "  outputs[{idx}] = chelis_alloc({}, shape_store_{idx}, {dtype}); }}",
                in_plan.shape.len()
            ));
        }
        self.body.push(format!(
            "chelis_tensor_write *store_guard_{idx} = chelis_tensor_begin_write(outputs[{idx}]);"
        ));
        self.body.push(format!(
            "chelis_write_view store_view_{idx} = chelis_tensor_write_view(store_guard_{idx});"
        ));
        self.body.push(format!(
            "chelis_metal_device_to_host(store_view_{idx}.data, {}, {bytes});",
            in_plan.buf
        ));
        self.body
            .push(format!("chelis_tensor_end_write(store_guard_{idx});"));
        Ok(())
    }

    /// Sum of per-buffer allocations the emitted host code will issue.
    ///
    /// Each materialized DAG node owns one device buffer (no slot reuse
    /// yet — that's M-phase planner work). Bytes per element follows
    /// the MSL type, matching what the emitter passes to
    /// `chelis_metal_alloc`. On Apple Silicon unified memory this is
    /// also the peak system RAM cost for tensor storage.
    fn peak_device_bytes(&self) -> usize {
        self.plans
            .iter()
            .filter_map(|p| p.as_ref())
            .map(|p| {
                // Honor the per-dtype element width via the central
                // helper so no caller hardcodes `sizeof(float)`
                // (RT-4-Fixups F2 lesson). Every alloc bumps
                // zero-length to >= 1 byte in the runtime, but we
                // report the requested size; n=0 reports 0 bytes,
                // consistent with the formula contract.
                p.n * p.elem_size()
            })
            .sum()
    }

    fn plan_of(&self, id: NodeId) -> Option<&TensorPlan> {
        self.plans.get(id.0).and_then(Option::as_ref)
    }
}
