//! C code generation backend for the Chelis language.

pub mod blas;
mod emit;
mod emitted_expr;
mod generated_header;
mod host_abi;
mod host_emit;
pub mod integer_float;
pub mod memory;
pub mod toolchain;

pub use generated_header::{GeneratedDeclaration, GeneratedHeader, GeneratedHeaderError};

/// Primitive types the C backend's tensor-DAG path can realize.
/// A def whose declared return type or intermediates use a prim NOT in this
/// set must route to the host lane. Verified by execution (issue #912 Task 1):
/// i32/i64 DO lower through the DAG path as general tensor ops, not only
/// as sparse indices. Only f64 is actually rejected.
pub const TENSOR_CAPABLE_PRIMS: &[chelis_types::types::Prim] = &[
    chelis_types::types::Prim::F32,
    chelis_types::types::Prim::Bool,
    chelis_types::types::Prim::Bf16,
    chelis_types::types::Prim::F16,
    chelis_types::types::Prim::Int32,
    chelis_types::types::Prim::Int64,
];

#[cfg(test)]
mod host_abi_tests;

/// Result of C code generation.
pub struct CodegenResult {
    /// The generated C source code (includes `#include "chelis_runtime.h"`).
    pub c_source: String,
    /// The generated C header declaration for the function.
    pub h_header: String,
    /// Platform-neutral codegen requirements resolved later by the native toolchain layer.
    pub requirements: toolchain::CodegenRequirements,
    /// Input slot labels in positional order. Repeated `Load(name)` nodes share one slot.
    pub input_labels: Vec<String>,
    /// Output slot labels in positional order.
    ///
    /// Phase 0f treats `Store(name)` as a named exported output. Non-store roots
    /// are appended afterward as `root{index}`.
    pub output_labels: Vec<String>,
    /// Unresolved symbolic dimensions that the generated function binds from input metadata.
    pub symbolic_dims: Vec<String>,
}

impl CodegenResult {
    /// Rebind the generated header to the current exact source bytes.
    ///
    /// Artifact assemblers call this after appending compiler-owned source such
    /// as the optional observation `main`; callers must not edit generated C
    /// without resealing the paired header.
    pub fn reseal_artifact(
        &mut self,
        program_identity: &str,
    ) -> Result<(), chelis_types::unsupported::Unsupported> {
        let (source, header) = generated_header::reseal_generated_artifact(
            program_identity,
            &self.c_source,
            &self.h_header,
        )
        .map_err(generated_artifact_error)?;
        self.c_source = source;
        self.h_header = header;
        Ok(())
    }
}

/// A tensor-helper DAG and the symbol a peer translation unit must define.
/// The generated host calls it through a private context adapter.
///
/// The manifest is read off the concrete host program before payload
/// selection and ownership lowering, so a device backend can lower a selected
/// helper through its own preparation pipeline while the C wrapper declares
/// that helper as an external symbol instead of embedding a C body.
#[derive(Debug, Clone)]
pub struct HostTensorHelperCodegen {
    pub name: String,
    pub dag: chelis_ir::dag::Dag,
}

/// Which vectorized math library is available for SIMD emission (Level 3b).
///
/// Detected at build time via `build.rs`; stored on [`CEmitter`] and threaded through
/// codegen so the emitter can choose the right include and macro set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MathLib {
    /// AVX2 + Sleef (Linux, detected via pkg-config at build time).
    Sleef,
    /// vForce via Accelerate.framework (macOS).
    VForce,
    /// No vectorized math library available; scalar fallback only.
    None,
}

impl MathLib {
    /// Select the appropriate variant based on compile-time feature flags.
    pub fn detect() -> Self {
        #[cfg(feature = "sleef")]
        return MathLib::Sleef;
        #[cfg(all(not(feature = "sleef"), target_os = "macos"))]
        return MathLib::VForce;
        #[cfg(all(not(feature = "sleef"), not(target_os = "macos")))]
        MathLib::None
    }
}

/// Optional backend features for C code generation.
#[derive(Debug, Clone, Copy, Default)]
pub struct CodegenOptions {
    /// Emit already-explicit backend BLAS nodes and their OpenBLAS link flags.
    /// Primitive contractions retain their specified arithmetic regardless of
    /// this option; recognizing a matrix shape does not authorize reassociation.
    pub use_blas: bool,
    /// Override the math library selection for SIMD emission (Level 3b).
    ///
    /// `None` means auto-detect via [`MathLib::detect`].  Set to `Some(MathLib::Sleef)`
    /// in tests to force the Sleef code-generation path without requiring the library to
    /// actually be installed on the build machine.
    pub math_lib_override: Option<MathLib>,
    /// Emit the generated entry function with `static` linkage.
    ///
    /// Set to `true` when the emitted DAG kernel is a TU-internal helper (e.g. a
    /// `HostTensorHelper` DAG embedded in a host `.c` file).  External callers must
    /// never see `static` on the exported entry point — leave this `false` (the
    /// default) for all standalone `codegen` / `codegen_with_options` calls.
    pub static_entry: bool,
}

/// Generate C source code from a RISC DAG.
///
/// Phase 0f codegen supports only `f32`/`bool` tensors.
///
/// Returns a [`CodegenResult`] containing the generated code plus the native-toolchain
/// requirements and positional input/output labels.
///
/// Repeated `Load(name)` nodes share one input slot, surfaced via `input_labels`.
/// `Store(name)` nodes are exported as named outputs in `output_labels`; any
/// remaining DAG roots are appended afterward as `root{index}`.
///
/// ```compile_fail
/// # use chelis_ir::dag::Dag;
/// fn bypass(raw: &Dag) {
///     let _ = chelis_backend_c::codegen(raw, "unchecked");
/// }
/// ```
pub fn codegen(
    dag: chelis_ir::ownership::VerifiedDagProgram,
    func_name: &str,
) -> Result<CodegenResult, chelis_types::unsupported::Unsupported> {
    codegen_with_options(dag, func_name, CodegenOptions::default())
}

/// Compile a sealed, verified host payload.
///
/// ```compile_fail
/// # use chelis_ir::host::ConcreteHostProgram;
/// fn bypass(raw: &ConcreteHostProgram) {
///     let _ = chelis_backend_c::codegen_host_program(raw, "unchecked");
/// }
/// ```
pub fn codegen_host_program(
    program: &chelis_ir::ownership::VerifiedHostProgram,
    func_name: &str,
) -> Result<CodegenResult, chelis_types::unsupported::Unsupported> {
    codegen_host_program_with_external_tensor_helpers(program, func_name, &[])
}

/// Compile a sealed, verified host payload while leaving the named tensor
/// helpers to peer translation units.
///
/// Every name in `external_helpers` must be an exact member of the wrapper's
/// helper manifest ([`host_tensor_helper_codegen`]); the wrapper then emits an
/// `extern` prototype for it instead of a `static` C body. An unknown name is
/// a typed rejection rather than a silent missing-definition link error.
pub fn codegen_host_program_with_external_tensor_helpers(
    program: &chelis_ir::ownership::VerifiedHostProgram,
    func_name: &str,
    external_helpers: &[String],
) -> Result<CodegenResult, chelis_types::unsupported::Unsupported> {
    // Resolve the backend capability boundary once.  All emission below is
    // over the private, fully-resolved ABI vocabulary; neither source nor
    // header generation can re-interpret logical types independently.
    let abi_program = host_abi::project_program(program.emission())?;
    let known_helpers = host_emit::tensor_helper_names(abi_program.program(), func_name)?;
    if let Some(unknown) = external_helpers
        .iter()
        .find(|name| !known_helpers.iter().any(|known| known == *name))
    {
        return Err(chelis_types::unsupported::Unsupported::new(
            chelis_types::unsupported::UnsupportedKind::Construct(format!(
                "unknown external host tensor helper `{unknown}`"
            )),
            "C host tensor-helper composition",
            chelis_types::unsupported::Stage::Codegen("c"),
            chelis_types::deliberate_rejection!(
                "[04-TOT-2]",
                "an external helper selection must name an exact manifest member; no missing-definition fallback is permitted"
            ),
        ));
    }
    let external_helpers = external_helpers
        .iter()
        .cloned()
        .collect::<chelis_unord::UnordSet<_>>();
    let c_source = host_emit::emit_host_abi_program(&abi_program, func_name, &external_helpers)?;
    let h_header = host_emit::emit_host_abi_header(&abi_program, func_name)?;
    let (c_source, h_header) = seal_generated_artifact(func_name, &c_source, &h_header)?;
    let needs_blas = c_source.contains("#include \"chelis_blas.h\"")
        || c_source.contains("cblas_sgemm(")
        || c_source.contains("cblas_dgemm(")
        || c_source.contains("chelis_blas_matmul");
    Ok(CodegenResult {
        c_source,
        h_header,
        requirements: toolchain::CodegenRequirements {
            wants_openmp: true,
            needs_blas,
        },
        input_labels: Vec::new(),
        output_labels: Vec::new(),
        symbolic_dims: Vec::new(),
    })
}

/// Return the exact helper symbols and source DAGs a host-program wrapper
/// will call, in wrapper emission order, handing the program back unchanged.
///
/// This is a by-value pre-verification step like
/// [`prepare_host_program_for_codegen`], never a borrowed emission edge: it
/// reads the concrete program before C payload selection and ownership
/// lowering, so a device backend that externalizes a helper lowers the
/// returned source DAG through its own preparation pipeline, exactly as it
/// would a tensor entry, rather than reusing the C lane's selected payload.
pub fn host_tensor_helper_codegen(
    program: chelis_ir::host::ConcreteHostProgram,
    func_name: &str,
) -> Result<
    (
        Vec<HostTensorHelperCodegen>,
        chelis_ir::host::ConcreteHostProgram,
    ),
    chelis_types::unsupported::Unsupported,
> {
    let helpers = host_emit::concrete_tensor_helper_codegen(&program, func_name)?
        .into_iter()
        .map(|(name, dag)| HostTensorHelperCodegen { name, dag })
        .collect();
    Ok((helpers, program))
}

/// Generate C source code from a RISC DAG with explicit backend options.
///
/// Fixed-control programs use `codegen_evaluation_with_options` instead:
/// ordinary graph ownership alone grants no forward/replay association.
pub fn codegen_with_options(
    dag: chelis_ir::ownership::VerifiedDagProgram,
    func_name: &str,
    options: CodegenOptions,
) -> Result<CodegenResult, chelis_types::unsupported::Unsupported> {
    let h_header = generated_header::render_declaration(
        func_name,
        func_name,
        &format!(
            "void {func_name}(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);"
        ),
    );
    let (needs_blas, input_labels, output_labels, symbolic_dims) = {
        let emission = dag.emission();
        (
            emission
                .nodes()
                .iter()
                .any(|node| matches!(node.op, chelis_ir::dag::RiscOp::BlasMatmul { .. })),
            emit::CEmitter::input_labels(emission),
            emit::CEmitter::output_labels(emission),
            emission.symbolic_params(),
        )
    };
    let c_source = emit::CEmitter::emit_dag_with_options(dag, func_name, options)?;
    let (c_source, h_header) = if options.static_entry {
        (c_source, String::new())
    } else {
        seal_generated_artifact(func_name, &c_source, &h_header)?
    };
    Ok(CodegenResult {
        c_source,
        h_header,
        requirements: toolchain::CodegenRequirements {
            wants_openmp: true,
            needs_blas,
        },
        input_labels,
        output_labels,
        symbolic_dims,
    })
}

fn seal_generated_artifact(
    program_identity: &str,
    source: &str,
    header: &str,
) -> Result<(String, String), chelis_types::unsupported::Unsupported> {
    generated_header::seal_generated_artifact(program_identity, source, header)
        .map_err(generated_artifact_error)
}

fn generated_artifact_error(
    error: generated_header::GeneratedHeaderError,
) -> chelis_types::unsupported::Unsupported {
    chelis_types::unsupported::Unsupported::new(
        chelis_types::unsupported::UnsupportedKind::Construct(
            "generated C artifact contract".to_string(),
        ),
        error.to_string(),
        chelis_types::unsupported::Stage::Codegen("c"),
        chelis_types::deliberate_rejection!(
            "[01-CID-1]",
            "generated C source and header must carry one exact program/export artifact envelope"
        ),
    )
}

/// Apply the C backend's payload-selection rewrites before ownership lowering.
pub fn prepare_dag_for_codegen(
    dag: chelis_ir::dag::Dag,
    options: CodegenOptions,
) -> chelis_ir::dag::Dag {
    let dag = if options.use_blas {
        chelis_ir::specialize::specialize_for_exact_arithmetic(&dag)
    } else {
        dag
    };
    // chelis#1788: split a dimension identity two independent scopes spell
    // before anything keys a declaration by name, so the anonymous pass
    // propagates whatever identity each scope ended up with.
    emit::CEmitter::rename_anonymous_dims(emit::CEmitter::rename_scoped_dims(dag))
}

/// Select the exact nested C helper DAGs before host ownership lowering.
pub fn prepare_host_program_for_codegen(
    mut program: chelis_ir::host::ConcreteHostProgram,
) -> Result<chelis_ir::host::ConcreteHostProgram, chelis_types::unsupported::Unsupported> {
    prepare_concrete_host_program_for_codegen(&mut program, true)?;
    Ok(program)
}

/// Check the nested C helper DAGs of a program the C execution lane selected
/// because a helper draws `dropout`. The helper graphs are emitted exactly as
/// lowered, without the payload rewrites of the ordinary host lane.
pub fn prepare_host_execution_plan_for_codegen(
    plan: chelis_ir::host::HostExecutionPlan,
) -> Result<chelis_ir::host::HostExecutionPlan, chelis_types::unsupported::Unsupported> {
    for helper in &plan.program().global_tensor_helpers {
        chelis_ir::check_axis_sources(&helper.dag, chelis_types::unsupported::Stage::Codegen("c"))?;
    }
    for function in &plan.program().functions {
        for helper in &function.tensor_helpers {
            chelis_ir::check_axis_sources(
                &helper.dag,
                chelis_types::unsupported::Stage::Codegen("c"),
            )?;
        }
    }
    Ok(plan)
}

fn prepare_concrete_host_program_for_codegen(
    program: &mut chelis_ir::host::ConcreteHostProgram,
    specialize_exact_arithmetic: bool,
) -> Result<(), chelis_types::unsupported::Unsupported> {
    fn prepare(
        helper: &mut chelis_ir::host::HostTensorHelper,
        specialize_exact_arithmetic: bool,
    ) -> Result<(), chelis_types::unsupported::Unsupported> {
        // A shape-derived BLAS summary is not proof of the primitive
        // contraction's arithmetic. Keep the source graph as the payload.
        if matches!(
            helper.specialization,
            Some(chelis_ir::host::HostTensorSpecialization::BlasMatmul(_))
        ) {
            helper.specialization = None;
        }
        let specialized = if specialize_exact_arithmetic {
            chelis_ir::specialize::specialize_for_exact_arithmetic(&helper.dag)
        } else {
            helper.dag.clone()
        };
        chelis_ir::check_axis_sources(
            &specialized,
            chelis_types::unsupported::Stage::Codegen("c"),
        )?;
        helper.dag = emit::CEmitter::rename_anonymous_dims(specialized);
        Ok(())
    }
    for helper in &mut program.global_tensor_helpers {
        prepare(helper, specialize_exact_arithmetic)?;
    }
    for function in &mut program.functions {
        // Wrapper propagation can lift a helper's shortcut into a function
        // summary; clear that route as well before ownership binds payloads.
        if matches!(
            function.specialization,
            Some(chelis_ir::host::HostFunctionSpecialization::BlasMatmul(_))
        ) {
            function.specialization = None;
        }
        for helper in &mut function.tensor_helpers {
            prepare(helper, specialize_exact_arithmetic)?;
        }
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod testing {
    use chelis_ir::ownership::{OwnershipError, VerifiedDagProgram};

    pub(crate) fn verified_dag(
        dag: &chelis_ir::dag::Dag,
        options: crate::CodegenOptions,
    ) -> Result<VerifiedDagProgram, OwnershipError> {
        let selected = crate::prepare_dag_for_codegen(dag.clone(), options);
        chelis_ir::ownership::verify_ownership(chelis_ir::ownership::lower_dag_ownership(selected)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chelis_ir::dag::{ComparisonKind, Dag, DimInfo, RiscOp, RtDim, TensorType};
    use chelis_ir::tier2;
    use chelis_types::types::Prim;
    use std::io::Write;
    use std::path::PathBuf;
    use std::process::Command;

    fn codegen(
        dag: &Dag,
        name: &str,
    ) -> Result<CodegenResult, chelis_types::unsupported::Unsupported> {
        codegen_with_options(dag, name, CodegenOptions::default())
    }

    fn codegen_with_options(
        dag: &Dag,
        name: &str,
        options: CodegenOptions,
    ) -> Result<CodegenResult, chelis_types::unsupported::Unsupported> {
        let verified = crate::testing::verified_dag(dag, options)
            .expect("C backend unit-test DAG must verify ownership");
        super::codegen_with_options(verified, name, options)
    }

    fn codegen_host_program(
        program: &chelis_ir::host::ConcreteHostProgram,
        name: &str,
    ) -> Result<CodegenResult, chelis_types::unsupported::Unsupported> {
        let source = program
            .functions
            .iter()
            .map(|function| {
                let params = function
                    .params
                    .iter()
                    .enumerate()
                    .map(|(index, _)| format!("p{index}: f64"))
                    .collect::<Vec<_>>();
                let body = if params.is_empty() { "0.0f64" } else { "p0" };
                format!(
                    "def {}({}) -> f64 = {body}",
                    function.name,
                    params.join(", ")
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        let declarations = chelis_surf::parser::parse_str(&source)
            .unwrap_or_else(|error| panic!("parse synthetic host signatures: {error:?}"));
        let deep = chelis_surf::desugar::desugar_program(&declarations)
            .expect("Surf fixture must desugar");
        let checked = chelis_types::check_typed_program(&deep).unwrap_or_else(|errors| {
            panic!("check synthetic host signatures: {:?}", errors.errors)
        });
        let checked = chelis_effects::check_program(&checked)
            .unwrap_or_else(|error| panic!("effects synthetic host signatures: {error:?}"));
        let checked = chelis_types::check_linearity(&checked)
            .unwrap_or_else(|error| panic!("linearity synthetic host signatures: {error:?}"));
        let manifested = chelis_types::manifest::ManifestedProgram::new(
            checked,
            chelis_types::manifest::RootManifest {
                entries: Vec::new(),
            },
            chelis_types::types::Target::C,
        );
        let selected = prepare_host_program_for_codegen(program.clone())?;
        let lowered = chelis_ir::ownership::lower_host_ownership(&manifested, selected)
            .expect("C backend unit-test host must lower ownership");
        let verified = chelis_ir::ownership::verify_ownership(lowered)
            .expect("C backend unit-test host must verify ownership");
        super::codegen_host_program(&verified, name)
    }

    fn authored_c_symbol(name: &str) -> String {
        let mut symbol = "chelis_fn_".to_string();
        for byte in name.bytes() {
            symbol.push_str(&format!("{byte:02x}"));
        }
        symbol
    }

    fn scalar_f32() -> TensorType {
        TensorType::scalar_f32()
    }

    fn vec_f32(n: usize) -> TensorType {
        TensorType {
            dims: vec![DimInfo::Lit(n)],
            precision: Prim::F32,
        }
    }

    fn mat_f32(r: usize, c: usize) -> TensorType {
        TensorType {
            dims: vec![DimInfo::Lit(r), DimInfo::Lit(c)],
            precision: Prim::F32,
        }
    }

    fn tensor_ty(dims: &[usize], precision: Prim) -> TensorType {
        TensorType {
            dims: dims.iter().copied().map(DimInfo::Lit).collect(),
            precision,
        }
    }

    /// Stage the runtime this test build carries, with its public headers, into
    /// `dst` (spec/08-backends.md §2.1). Nothing is looked up in a build
    /// directory: the bundle writes its own verified bytes.
    fn stage_runtime(dst: &std::path::Path) {
        chelis_runtime_bundle::stage(dst)
            .unwrap_or_else(|error| panic!("stage the carried runtime: {error}"));
    }

    /// Link the runtime `stage_runtime` wrote into `dir`, by exact path.
    fn add_runtime_link(cmd: &mut Command, dir: &std::path::Path) {
        cmd.arg(dir.join(chelis_runtime_bundle::ARCHIVE_FILE_NAME));
        cmd.args(
            crate::toolchain::runtime_toolchain(crate::toolchain::CodegenRequirements::default())
                .link_flags,
        );
    }

    // ---- Codegen API tests ----

    #[test]
    fn codegen_returns_source_and_header() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        let result = codegen(&dag, "my_func").unwrap();
        assert!(result.c_source.contains("void my_func("));
        assert!(result.h_header.contains("void my_func("));
        GeneratedHeader::parse(&result.h_header)
            .expect("generated direct declaration")
            .validate_source(&result.c_source)
            .expect("direct source/header export sets agree");
        assert_eq!(
            result.requirements,
            toolchain::CodegenRequirements {
                wants_openmp: true,
                needs_blas: false,
            }
        );
        assert!(result.input_labels.is_empty());
        assert_eq!(result.output_labels, vec!["root0"]);
    }

    #[test]
    fn codegen_header_is_declaration() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        let result = codegen(&dag, "test_fn").unwrap();
        assert!(result.h_header.ends_with(';'));
        assert!(!result.h_header.contains('{'));
    }

    /// chelis#593 memory-safety floor (negative parity): a `Pad` whose output
    /// extent on the concrete leading axis disagrees with `input + before +
    /// after` (the symbolic entry-wrapper concat mis-sizing) must be rejected
    /// loud at codegen, never emit the heap-corrupting copy loop.
    #[test]
    fn codegen_rejects_mis_sized_leading_axis_pad() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let sym = TensorType {
            dims: vec![DimInfo::Lit(2), DimInfo::Named("batch".into(), None)],
            precision: Prim::F32,
        };
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            sym.clone(),
            None,
        );
        // Padded leading axis by (2,0): output SHOULD be [4, batch] but is
        // MIS-SIZED to the operand extent [2, batch] (the #593 wrapper clobber).
        dag.add_node(
            decl,
            RiscOp::zero_pad(
                Prim::F32,
                vec![
                    (RtDim::Lit(2), RtDim::Lit(0)),
                    (RtDim::Lit(0), RtDim::Lit(0)),
                ],
            ),
            vec![x],
            sym,
            None,
        );
        let error = match chelis_ir::ownership::lower_dag_ownership(dag) {
            Err(error) => error,
            Ok(_) => panic!("a mis-sized pad must not become verified backend input"),
        };
        assert!(error.to_string().contains("expected 4"));
    }

    /// Positive parity: a CORRECTLY sized leading-axis Pad over a symbolic
    /// trailing dim must codegen fine (no over-rejection). Output [4, batch]
    /// == input [2, batch] + (2, 0).
    #[test]
    fn codegen_accepts_well_sized_leading_axis_pad() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let in_ty = TensorType {
            dims: vec![DimInfo::Lit(2), DimInfo::Named("batch".into(), None)],
            precision: Prim::F32,
        };
        let out_ty = TensorType {
            dims: vec![DimInfo::Lit(4), DimInfo::Named("batch".into(), None)],
            precision: Prim::F32,
        };
        let x = dag.add_node(decl, RiscOp::Load { name: "x".into() }, vec![], in_ty, None);
        dag.add_node(
            decl,
            RiscOp::zero_pad(
                Prim::F32,
                vec![
                    (RtDim::Lit(2), RtDim::Lit(0)),
                    (RtDim::Lit(0), RtDim::Lit(0)),
                ],
            ),
            vec![x],
            out_ty,
            None,
        );
        let result = codegen(&dag, "well_sized").unwrap();
        assert!(result.c_source.contains("void well_sized("));
    }

    // ---- Linkage invariant tests ----
    //
    // These tests enforce the PLT-avoidance contract: internal tensor helper
    // functions emitted into the `.c` file must carry `static` linkage so
    // that `-shared -fPIC` builds do not export them via PLT.  The exported
    // entry point must NOT be `static` so external callers can link against it.

    #[test]
    fn dag_codegen_exported_entry_has_no_static_prefix() {
        // The standalone DAG codegen path (used by `chelis build` for object-mode
        // compilation) must emit an extern-linkage entry function.  Making it
        // `static` would prevent external callers from linking against the symbol.
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        let result = codegen(&dag, "my_entry").unwrap();
        // The definition line must start with `void`, not `static void`.
        assert!(
            result.c_source.contains("\nvoid my_entry("),
            "exported entry must not be static; got:\n{}",
            &result.c_source[..result.c_source.find('{').unwrap_or(result.c_source.len())]
        );
        assert!(
            !result.c_source.contains("static void my_entry("),
            "exported entry must not carry static linkage"
        );
    }

    #[test]
    fn dag_codegen_with_static_entry_option_emits_static_prefix() {
        // When the caller explicitly requests static_entry (used internally for
        // HostTensorHelper DAG kernels), the function definition must be `static void`.
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        let options = CodegenOptions {
            static_entry: true,
            ..CodegenOptions::default()
        };
        let verified = crate::testing::verified_dag(&dag, options)
            .expect("static-entry test DAG must verify ownership");
        let result =
            emit::CEmitter::emit_dag_with_options(verified, "internal_helper", options).unwrap();
        assert!(
            result.contains("static void internal_helper("),
            "internal helper must be static; got source starting:\n{}",
            &result[..result.find('{').unwrap_or(result.len()).min(300)]
        );
    }

    #[test]
    fn host_program_tensor_helpers_are_static_entry_not_exported() {
        // For a HostProgram with a tensor helper, the emitted `.c` source must
        // mark the helper function as `static` (preventing PLT export) while
        // the user-facing HostFunction entry keeps external linkage.
        use chelis_ir::ConcreteHostType as HostType;
        use chelis_ir::host::{
            ConcreteHostExpr as HostExpr, ConcreteHostExprKind as HostExprKind,
            ConcreteHostFunction as HostFunction, ConcreteHostParam as HostParam,
            ConcreteHostProgram as HostProgram, HostTensorHelper,
        };

        // Build a simple 1-element scalar DAG for the helper.
        let mut helper_dag = Dag::new();
        let helper_dag_decl = helper_dag.declare("test");
        helper_dag.add_node(
            helper_dag_decl,
            RiscOp::synth_const(scalar_f32().precision, 2.0),
            vec![],
            scalar_f32(),
            None,
        );

        let helper = HostTensorHelper {
            name: "my_prog__my_fn__tensor_0".to_string(),
            dag: helper_dag,
            inputs: vec![],
            output: scalar_f32(),
            specialization: None,
            summary_rejection: None,
        };

        let func = HostFunction {
            helper_result_claim_axes: Vec::new(),
            entry_contract: Default::default(),
            name: "my_fn".to_string(),
            params: vec![HostParam {
                name: "x".to_string(),
                ty: HostType::Float64,
            }],
            ret_ty: HostType::Float64,
            // Body is just a float literal — does not actually call the tensor helper,
            // but the helper must still be emitted into the file.
            body: HostExpr::new(HostExprKind::Float(0.0)),
            tensor_helpers: vec![helper],
            origin: chelis_ir::host::HostFunctionOrigin::Authored,
            specialization: None,
            summary_rejections: Vec::new(),
        };

        let program = HostProgram {
            globals: vec![],
            global_tensor_helpers: vec![],
            functions: vec![func],
            summary_rejections: Vec::new(),
            adt_layouts: Vec::new(),
        };

        let result = codegen_host_program(&program, "my_prog").unwrap();
        let src = &result.c_source;

        // The tensor helper must be static (internal to the TU).
        // The context-carrying helper is private to this translation unit.
        let my_fn = authored_c_symbol("my_fn");
        assert!(
            src.contains(&format!("static void {my_fn}__tensor_0__private(")),
            "tensor helper must carry static linkage to avoid PLT export;\ngenerated source:\n{}",
            src
        );

        // The user-facing entry function must NOT be static (library mode: no globals).
        // Without globals, internal_linkage=false, so the function has external linkage.
        assert!(
            !src.contains(&format!("static void {my_fn}("))
                && !src.contains(&format!("static inline void {my_fn}(")),
            "exported entry function my_fn must not be static;\ngenerated source:\n{}",
            src
        );
        // Confirm the external-linkage definition is present.
        assert!(
            src.contains(&format!(" {my_fn}(")),
            "exported entry function my_fn must have an external-linkage definition;\ngenerated source:\n{}",
            src
        );
    }

    #[test]
    fn host_program_tensor_helper_blas_sets_toolchain_requirement() {
        use chelis_ir::ConcreteHostType as HostType;
        use chelis_ir::host::{
            ConcreteHostExpr as HostExpr, ConcreteHostExprKind as HostExprKind,
            ConcreteHostFunction as HostFunction, ConcreteHostParam as HostParam,
            ConcreteHostProgram as HostProgram, HostTensorHelper, HostTensorInput,
        };

        let a_ty = mat_f32(2, 3);
        let b_ty = mat_f32(3, 4);
        let out_ty = mat_f32(2, 4);
        let mut helper_dag = Dag::new();
        let helper_dag_decl = helper_dag.declare("test");
        let a = helper_dag.add_node(
            helper_dag_decl,
            RiscOp::Load { name: "a".into() },
            vec![],
            a_ty.clone(),
            None,
        );
        let b = helper_dag.add_node(
            helper_dag_decl,
            RiscOp::Load { name: "b".into() },
            vec![],
            b_ty.clone(),
            None,
        );
        let out = helper_dag.add_node(
            helper_dag_decl,
            RiscOp::BlasMatmul {
                batch_dims: vec![],
                m: chelis_ir::dag::DimExpr::Concrete(2),
                n: chelis_ir::dag::DimExpr::Concrete(4),
                k: chelis_ir::dag::DimExpr::Concrete(3),
                accumulator: chelis_types::types::Prim::F32,
            },
            vec![a, b],
            out_ty.clone(),
            None,
        );
        helper_dag.add_root(out);

        let helper = HostTensorHelper {
            name: "blas_helper".to_string(),
            dag: helper_dag,
            inputs: vec![
                HostTensorInput {
                    name: "a".to_string(),
                    ty: a_ty,
                },
                HostTensorInput {
                    name: "b".to_string(),
                    ty: b_ty,
                },
            ],
            output: out_ty,
            specialization: None,
            summary_rejection: None,
        };
        let func = HostFunction {
            helper_result_claim_axes: Vec::new(),
            entry_contract: Default::default(),
            name: "my_fn".to_string(),
            params: vec![HostParam {
                name: "x".to_string(),
                ty: HostType::Float64,
            }],
            ret_ty: HostType::Float64,
            body: HostExpr::new(HostExprKind::Float(0.0)),
            tensor_helpers: vec![helper],
            origin: chelis_ir::host::HostFunctionOrigin::Authored,
            specialization: None,
            summary_rejections: Vec::new(),
        };
        let program = HostProgram {
            globals: vec![],
            global_tensor_helpers: vec![],
            functions: vec![func],
            summary_rejections: Vec::new(),
            adt_layouts: Vec::new(),
        };

        let result = codegen_host_program(&program, "my_prog").unwrap();
        assert!(result.c_source.contains("#include \"chelis_blas.h\""));
        assert!(result.c_source.contains("cblas_sgemm("));
        assert!(
            result.requirements.needs_blas,
            "host-program codegen must surface BLAS link requirements when a tensor helper specializes"
        );
    }

    #[test]
    fn codegen_surfaces_store_output_labels() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(
            decl,
            RiscOp::Store { name: "out".into() },
            vec![a],
            scalar_f32(),
            None,
        );
        let result = codegen(&dag, "test_fn").unwrap();
        assert_eq!(result.output_labels, vec!["out"]);
    }

    #[test]
    fn codegen_surfaces_distinct_input_labels() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let x0 = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            scalar_f32(),
            None,
        );
        let x1 = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            scalar_f32(),
            None,
        );
        let y = dag.add_node(
            decl,
            RiscOp::Load { name: "y".into() },
            vec![],
            scalar_f32(),
            None,
        );
        let sum = dag.add_node(decl, RiscOp::Add, vec![x0, x1], scalar_f32(), None);
        dag.add_node(decl, RiscOp::Add, vec![sum, y], scalar_f32(), None);
        let result = codegen(&dag, "test_fn").unwrap();
        assert_eq!(result.input_labels, vec!["x", "y"]);
        assert_eq!(result.output_labels, vec!["root0"]);
    }

    #[test]
    fn codegen_does_not_surface_openblas_by_default() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(mat_f32(2, 3).precision, 1.0),
            vec![],
            mat_f32(2, 3),
            None,
        );
        let b = dag.add_node(
            decl,
            RiscOp::synth_const(mat_f32(3, 4).precision, 1.0),
            vec![],
            mat_f32(3, 4),
            None,
        );
        let ea = dag.add_node(
            decl,
            RiscOp::Expand {
                axis: 2,
                size: chelis_ir::dag::RtDim::Lit(4),
            },
            vec![a],
            TensorType {
                dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
                precision: Prim::F32,
            },
            None,
        );
        let eb = dag.add_node(
            decl,
            RiscOp::Expand {
                axis: 0,
                size: chelis_ir::dag::RtDim::Lit(2),
            },
            vec![b],
            TensorType {
                dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
                precision: Prim::F32,
            },
            None,
        );
        let mul = dag.add_node(
            decl,
            RiscOp::Mul,
            vec![ea, eb],
            TensorType {
                dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
                precision: Prim::F32,
            },
            None,
        );
        dag.add_node(
            decl,
            RiscOp::Sum {
                axis: 1,
                accumulator: chelis_types::types::Prim::F32,
            },
            vec![mul],
            mat_f32(2, 4),
            None,
        );
        let result = codegen(&dag, "test_fn").unwrap();
        assert!(!result.requirements.needs_blas);
        assert!(!result.c_source.contains("cblas_sgemm("));
    }

    #[test]
    fn blas_option_keeps_primitive_matmul_off_vendor_path() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(mat_f32(2, 3).precision, 1.0),
            vec![],
            mat_f32(2, 3),
            None,
        );
        let b = dag.add_node(
            decl,
            RiscOp::synth_const(mat_f32(3, 4).precision, 1.0),
            vec![],
            mat_f32(3, 4),
            None,
        );
        let ea = dag.add_node(
            decl,
            RiscOp::Expand {
                axis: 2,
                size: chelis_ir::dag::RtDim::Lit(4),
            },
            vec![a],
            TensorType {
                dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
                precision: Prim::F32,
            },
            None,
        );
        let eb = dag.add_node(
            decl,
            RiscOp::Expand {
                axis: 0,
                size: chelis_ir::dag::RtDim::Lit(2),
            },
            vec![b],
            TensorType {
                dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
                precision: Prim::F32,
            },
            None,
        );
        let mul = dag.add_node(
            decl,
            RiscOp::Mul,
            vec![ea, eb],
            TensorType {
                dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
                precision: Prim::F32,
            },
            None,
        );
        dag.add_node(
            decl,
            RiscOp::Sum {
                axis: 1,
                accumulator: chelis_types::types::Prim::F32,
            },
            vec![mul],
            mat_f32(2, 4),
            None,
        );
        let result = codegen_with_options(
            &dag,
            "test_fn",
            CodegenOptions {
                use_blas: true,
                ..CodegenOptions::default()
            },
        )
        .unwrap();
        assert!(!result.requirements.needs_blas);
        assert!(!result.c_source.contains("cblas_sgemm("));
        assert!(result.c_source.contains("__sum_level_"));
    }

    // ---- Compilation tests ----

    fn write_temp_file(dir: &std::path::Path, name: &str, content: &str) -> PathBuf {
        let path = dir.join(name);
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(content.as_bytes()).unwrap();
        path
    }

    fn c_test_extra_flags() -> Vec<String> {
        std::env::var("CHELIS_C_TEST_EXTRA_FLAGS")
            .ok()
            .map(|flags| {
                flags
                    .split_whitespace()
                    .map(|flag| flag.to_string())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default()
    }

    fn apply_c_test_flags(cmd: &mut Command) {
        let extra = c_test_extra_flags();
        if !extra.is_empty() {
            cmd.args(extra);
        }
    }

    /// Host-arch SIMD ISA flag for SIMD-exercising compile-run tests.
    ///
    /// `chelis_simd.h` and the generated math kernels are arch-aware:
    /// `#ifdef __AVX2__` on x86, `#elif defined(__ARM_NEON)` on ARM, with
    /// a scalar fallback. On x86_64 we pass `-mavx2` to guarantee the AVX2
    /// path is exercised regardless of the host's `-march=native` baseline.
    /// On aarch64 (e.g. Apple Silicon CI runners) NEON is part of the
    /// architecture baseline, so `__ARM_NEON` is already defined and the
    /// NEON path activates with no extra flag; passing `-mavx2` there is an
    /// `unsupported option` clang error. Returning `&[]` keeps the build
    /// portable and lets the kernel run via NEON. Other arches fall through
    /// to the scalar path with no ISA flag.
    fn simd_isa_test_flags() -> &'static [&'static str] {
        if cfg!(target_arch = "x86_64") {
            &["-mavx2"]
        } else {
            &[]
        }
    }

    fn test_toolchain(
        requirements: crate::toolchain::CodegenRequirements,
    ) -> crate::toolchain::NativeToolchain {
        crate::toolchain::test_toolchain(requirements)
    }

    #[test]
    fn runtime_compiles_standalone() {
        let tmp = tempfile::tempdir().unwrap();
        stage_runtime(tmp.path());
        write_temp_file(
            tmp.path(),
            "main.c",
            "#include \"chelis_runtime.h\"\nint main(void) { return 0; }\n",
        );
        let o_path = tmp.path().join("runtime_smoke");

        let mut cmd = Command::new(crate::toolchain::c_compiler());
        apply_c_test_flags(&mut cmd);
        cmd.args(["-O2"])
            .arg(tmp.path().join("main.c").to_str().unwrap());
        add_runtime_link(&mut cmd, tmp.path());
        cmd.arg("-lm").arg("-o").arg(o_path.to_str().unwrap());
        let out = cmd.output().unwrap();
        assert!(
            out.status.success(),
            "gcc failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(o_path.exists());
    }

    #[test]
    fn runtime_view_free_is_safe() {
        let tmp = tempfile::tempdir().unwrap();
        stage_runtime(tmp.path());
        let main_c = r#"
#include "chelis_runtime.h"
int main(void) {
    int64_t shape[2] = {2, 3};
    float data[6] = {0};
    data[4] = 7.0f;
    chelis_tensor *base = chelis_tensor_entry_borrow(
        2, shape, CHELIS_DTYPE_F32, data, sizeof(data)
    );
    chelis_tensor *view = chelis_contiguous(base);
    chelis_tensor_release(view);
    if (data[4] != 7.0f) return 2;
    chelis_tensor_release(base);
    return 0;
}
"#;
        write_temp_file(tmp.path(), "main.c", main_c);
        let bin_path = tmp.path().join("runtime_view");
        let mut cmd = Command::new(crate::toolchain::c_compiler());
        apply_c_test_flags(&mut cmd);
        cmd.args(["-O2", "-lm"])
            .arg(tmp.path().join("main.c").to_str().unwrap());
        add_runtime_link(&mut cmd, tmp.path());
        cmd.arg("-lm").arg("-o").arg(bin_path.to_str().unwrap());
        let out = cmd.output().unwrap();
        assert!(
            out.status.success(),
            "gcc failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let run = Command::new(bin_path).output().unwrap();
        assert!(
            run.status.success(),
            "runtime view free binary failed: {}",
            String::from_utf8_lossy(&run.stderr)
        );
    }

    #[test]
    fn simple_add_compiles() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        let b = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 2.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(decl, RiscOp::Add, vec![a, b], scalar_f32(), None);
        let result = codegen(&dag, "test_add").unwrap();

        let tmp = tempfile::tempdir().unwrap();
        stage_runtime(tmp.path());
        write_temp_file(tmp.path(), "model.c", &result.c_source);

        let main_c = r#"
#include "chelis_runtime.h"
void test_add(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);
int main() {
    chelis_tensor *outputs[1] = {0};
    test_add(NULL, 0, outputs, 1);
    printf("%.1f\n", ((const float*)chelis_tensor_read_view(outputs[0]).data)[0]);
    chelis_tensor_release(outputs[0]);
    return 0;
}
"#;
        write_temp_file(tmp.path(), "main.c", main_c);
        let bin_path = tmp.path().join("test_add");

        let mut cmd = Command::new(crate::toolchain::c_compiler());
        apply_c_test_flags(&mut cmd);
        cmd.args(["-O2", "-lm"])
            .arg(tmp.path().join("main.c").to_str().unwrap())
            .arg(tmp.path().join("model.c").to_str().unwrap());
        add_runtime_link(&mut cmd, tmp.path());
        cmd.arg("-lm").arg("-o").arg(bin_path.to_str().unwrap());
        let out = cmd.output().unwrap();
        assert!(
            out.status.success(),
            "gcc failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(bin_path.exists());
    }

    // ---- Numerical tests (compile + run + check output) ----

    /// Helper: build a DAG, generate C, compile with a main() wrapper, run, return stdout.
    fn compile_and_run(dag: &Dag, func_name: &str) -> String {
        compile_and_run_with_codegen_options(dag, func_name, CodegenOptions::default(), &[])
    }

    fn compile_and_run_with_flags(dag: &Dag, func_name: &str, extra_args: &[&str]) -> String {
        compile_and_run_with_codegen_options(dag, func_name, CodegenOptions::default(), extra_args)
    }

    fn compile_and_run_with_codegen_options(
        dag: &Dag,
        func_name: &str,
        options: CodegenOptions,
        extra_args: &[&str],
    ) -> String {
        let result = codegen_with_options(dag, func_name, options).unwrap();

        let tmp = tempfile::tempdir().unwrap();
        stage_runtime(tmp.path());
        write_temp_file(tmp.path(), "model.c", &result.c_source);

        let main_c = format!(
            r#"
#include "chelis_runtime.h"
static double chelis_test_element_as_f64(const chelis_tensor *tensor, int64_t index) {{
    chelis_read_view view = chelis_tensor_read_view(tensor);
    switch (view.dtype) {{
        case CHELIS_DTYPE_F32: return ((const float*)view.data)[index];
        case CHELIS_DTYPE_F64: return ((const double*)view.data)[index];
        case CHELIS_DTYPE_I8: return ((const int8_t*)view.data)[index];
        case CHELIS_DTYPE_I16: return ((const int16_t*)view.data)[index];
        case CHELIS_DTYPE_I32: return ((const int32_t*)view.data)[index];
        case CHELIS_DTYPE_I64: return (double)((const int64_t*)view.data)[index];
        case CHELIS_DTYPE_BF16: return chelis_bf16_to_f32(((const uint16_t*)view.data)[index]);
        case CHELIS_DTYPE_F16: return chelis_f16_to_f32(((const uint16_t*)view.data)[index]);
        case CHELIS_DTYPE_BOOL: {{
            uint8_t value = ((const uint8_t*)view.data)[index];
            if (value > UINT8_C(1)) abort();
            return value;
        }}
        default: abort();
    }}
}}
void {func_name}(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);
int main() {{
    chelis_tensor *outputs[1] = {{0}};
    {func_name}(NULL, 0, outputs, 1);
    for (int i = 0; i < chelis_tensor_numel(outputs[0]); i++) {{
        if (i > 0) printf(" ");
        printf("%.6f", chelis_test_element_as_f64(outputs[0], i));
    }}
    printf("\n");
    chelis_tensor_release(outputs[0]);
    return 0;
}}
"#
        );
        write_temp_file(tmp.path(), "main.c", &main_c);
        let bin_path = tmp.path().join("test_bin");

        let toolchain = test_toolchain(result.requirements);
        let mut compile_cmd = Command::new(&toolchain.compiler);
        apply_c_test_flags(&mut compile_cmd);
        compile_cmd.args(["-O2"]);
        compile_cmd.args(&toolchain.compile_flags);
        compile_cmd.args(extra_args);
        compile_cmd.arg(tmp.path().join("main.c").to_str().unwrap());
        compile_cmd.arg(tmp.path().join("model.c").to_str().unwrap());
        add_runtime_link(&mut compile_cmd, tmp.path());
        compile_cmd.args(&toolchain.link_flags);
        compile_cmd.arg("-o");
        compile_cmd.arg(bin_path.to_str().unwrap());
        let compile = compile_cmd.output().unwrap();
        assert!(
            compile.status.success(),
            "gcc failed:\nstderr: {}\nC source:\n{}",
            String::from_utf8_lossy(&compile.stderr),
            result.c_source
        );

        let run = Command::new(bin_path.to_str().unwrap()).output().unwrap();
        assert!(
            run.status.success(),
            "binary failed: {}",
            String::from_utf8_lossy(&run.stderr)
        );
        String::from_utf8(run.stdout).unwrap().trim().to_string()
    }

    #[derive(Clone)]
    struct TestInput {
        name: String,
        shape: Vec<usize>,
        data: Vec<f32>,
    }

    impl TestInput {
        fn new(name: &str, shape: &[usize], data: &[f32]) -> Self {
            Self {
                name: name.to_string(),
                shape: shape.to_vec(),
                data: data.to_vec(),
            }
        }
    }

    fn c_shape(shape: &[usize]) -> (usize, Vec<usize>) {
        (shape.len(), shape.to_vec())
    }

    /// Issue #252: build a single test-harness input-fill statement that
    /// reconstructs `value` from its exact f32 bit pattern via the
    /// `chelis_f32_from_bits` static inline helper (declared in the
    /// emitted `chelis_runtime.h`). Mirrors the production #189 / #248
    /// bit-pattern emission. The pre-fix `{:.8}f` format string printed
    /// decimal places after the point (not significant digits), so values
    /// like `0.1f32` or `1.0 / 3.0` could not round-trip to their exact
    /// f32 bits when the emitted C reparsed the literal.
    fn harness_input_fill_line(lhs: &str, value: f32) -> String {
        let bits = value.to_bits();
        let (tensor, index) = lhs
            .split_once('[')
            .and_then(|(tensor, index)| index.strip_suffix(']').map(|index| (tensor, index)))
            .expect("f32 harness destination must be a tensor data index");
        format!(
            "do {{ chelis_tensor_write *guard = chelis_tensor_begin_write({tensor}); \
             chelis_write_view view = chelis_tensor_write_view(guard); \
             ((float*)view.data)[{index}] = chelis_f32_from_bits(0x{bits:08x}u); \
             chelis_tensor_end_write(guard); }} while (0);"
        )
    }

    /// Issue #252: the test-harness fill must round-trip every f32 value
    /// to its exact bit pattern, including values that the pre-fix
    /// `{:.8}f` format string truncated or collapsed. This is a pure
    /// string-shape oracle: no gcc / run needed.
    #[test]
    fn issue_252_harness_input_fill_round_trips_exact_f32_bits() {
        // `0.1f32`, `1.0 / 3.0`, and a denormal all reparse to a
        // different bit pattern (or zero) under `%.8`.
        let cases: [f32; 4] = [
            0.1_f32,
            (1.0_f64 / 3.0_f64) as f32,
            1e-40_f32, // denormal: `%.8` -> `0.00000000f` (bits 0)
            // Arbitrary pinned bit pattern in the #189 small-magnitude
            // family; `%.8` cannot reproduce these exact bits.
            f32::from_bits(0x1234_5678),
        ];
        for value in cases {
            let line = harness_input_fill_line("dst[0]", value);
            let want_bits = value.to_bits();
            let needle = format!("chelis_f32_from_bits(0x{want_bits:08x}u)");
            assert!(
                line.contains(&needle),
                "fill for {value:e} must carry exact bits {want_bits:#010x}; got: {line}"
            );
            // Negative parity: the lossy decimal `f` literal must be gone.
            assert!(
                !line.contains(&format!("{value:.8}f")),
                "fill must not use the lossy `{{:.8}}f` form for {value:e}: {line}"
            );
        }
        // The denormal reproducer is the sharpest: `%.8` collapses it to
        // zero, but the bit pattern is nonzero and must survive.
        let denormal = 1e-40_f32;
        assert_ne!(denormal.to_bits(), 0);
        let line = harness_input_fill_line("dst[0]", denormal);
        assert!(
            !line.contains("0.00000000f"),
            "denormal must not collapse to `0.00000000f`: {line}"
        );
    }

    fn compile_and_run_input_cases(
        dag: &Dag,
        func_name: &str,
        options: CodegenOptions,
        cases: &[Vec<TestInput>],
    ) -> Vec<String> {
        let result = codegen_with_options(dag, func_name, options).unwrap();
        let n_out = result.output_labels.len();

        let mut case_blocks = Vec::new();
        for (case_idx, case) in cases.iter().enumerate() {
            for label in &result.input_labels {
                assert!(
                    case.iter().any(|input| &input.name == label),
                    "missing input '{label}' in case {case_idx}"
                );
            }

            let mut lines = Vec::new();
            lines.push(format!(
                "chelis_tensor *inputs_{case_idx}[{}] = {{0}};",
                result.input_labels.len()
            ));
            lines.push(format!(
                "chelis_tensor *outputs_{case_idx}[{n_out}] = {{0}};"
            ));

            for (slot, label) in result.input_labels.iter().enumerate() {
                let input = case
                    .iter()
                    .find(|candidate| &candidate.name == label)
                    .unwrap_or_else(|| panic!("missing input '{label}'"));
                let (ndim, c_dims) = c_shape(&input.shape);
                let shape_vals = c_dims
                    .iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                let shape_arg = if ndim == 0 {
                    "NULL".to_string()
                } else {
                    lines.push(format!(
                        "int64_t shape_{case_idx}_{slot}[{ndim}] = {{ {shape_vals} }};"
                    ));
                    format!("shape_{case_idx}_{slot}")
                };
                lines.push(format!(
                    "chelis_tensor *input_{case_idx}_{slot} = chelis_alloc({ndim}, {shape_arg}, CHELIS_DTYPE_F32);"
                ));
                for (i, value) in input.data.iter().enumerate() {
                    let lhs = format!("input_{case_idx}_{slot}[{i}]");
                    lines.push(harness_input_fill_line(&lhs, *value));
                }
                lines.push(format!(
                    "inputs_{case_idx}[{slot}] = input_{case_idx}_{slot};"
                ));
            }

            lines.push(format!(
                "{func_name}(inputs_{case_idx}, {}, outputs_{case_idx}, {n_out});",
                result.input_labels.len()
            ));
            lines.push(format!(
                "for (int out_idx = 0; out_idx < {n_out}; out_idx++) {{"
            ));
            lines.push(format!(
                "    for (int i = 0; i < chelis_tensor_numel(outputs_{case_idx}[out_idx]); i++) {{"
            ));
            lines.push("        if (out_idx > 0 || i > 0) printf(\" \");".to_string());
            lines.push(format!(
                "        printf(\"%.6f\", chelis_test_element_as_f64(outputs_{case_idx}[out_idx], i));"
            ));
            lines.push("    }".to_string());
            lines.push("    if (out_idx + 1 < n_out) printf(\" |\");".to_string());
            lines.push("}".to_string());
            lines.push("printf(\"\\n\");".to_string());
            for slot in 0..result.input_labels.len() {
                lines.push(format!("chelis_tensor_release(input_{case_idx}_{slot});"));
            }
            for slot in 0..n_out {
                lines.push(format!(
                    "chelis_tensor_release(outputs_{case_idx}[{slot}]);"
                ));
            }
            case_blocks.push(lines.join("\n    "));
        }

        let tmp = tempfile::tempdir().unwrap();
        stage_runtime(tmp.path());
        write_temp_file(tmp.path(), "model.c", &result.c_source);

        let main_c = format!(
            r#"
#include "chelis_runtime.h"
static double chelis_test_element_as_f64(const chelis_tensor *tensor, int64_t index) {{
    chelis_read_view view = chelis_tensor_read_view(tensor);
    switch (view.dtype) {{
        case CHELIS_DTYPE_F32: return ((const float*)view.data)[index];
        case CHELIS_DTYPE_F64: return ((const double*)view.data)[index];
        case CHELIS_DTYPE_I8: return ((const int8_t*)view.data)[index];
        case CHELIS_DTYPE_I16: return ((const int16_t*)view.data)[index];
        case CHELIS_DTYPE_I32: return ((const int32_t*)view.data)[index];
        case CHELIS_DTYPE_I64: return (double)((const int64_t*)view.data)[index];
        case CHELIS_DTYPE_BF16: return chelis_bf16_to_f32(((const uint16_t*)view.data)[index]);
        case CHELIS_DTYPE_F16: return chelis_f16_to_f32(((const uint16_t*)view.data)[index]);
        case CHELIS_DTYPE_BOOL: {{
            uint8_t value = ((const uint8_t*)view.data)[index];
            if (value > UINT8_C(1)) abort();
            return value;
        }}
        default: abort();
    }}
}}
void {func_name}(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);
int main(void) {{
    int n_out = {n_out};
    {cases}
    return 0;
}}
"#,
            cases = case_blocks.join("\n    ")
        );
        write_temp_file(tmp.path(), "main.c", &main_c);
        let bin_path = tmp.path().join("test_cases");

        let toolchain = test_toolchain(result.requirements);
        let mut compile_cmd = Command::new(&toolchain.compiler);
        apply_c_test_flags(&mut compile_cmd);
        compile_cmd.args(["-O2"]);
        compile_cmd.args(&toolchain.compile_flags);
        compile_cmd.arg(tmp.path().join("main.c").to_str().unwrap());
        compile_cmd.arg(tmp.path().join("model.c").to_str().unwrap());
        add_runtime_link(&mut compile_cmd, tmp.path());
        compile_cmd.args(&toolchain.link_flags);
        compile_cmd.arg("-o");
        compile_cmd.arg(bin_path.to_str().unwrap());
        let compile = compile_cmd.output().unwrap();
        assert!(
            compile.status.success(),
            "gcc failed:\nstderr: {}\nC source:\n{}",
            String::from_utf8_lossy(&compile.stderr),
            result.c_source
        );

        let run = Command::new(bin_path).output().unwrap();
        assert!(
            run.status.success(),
            "binary failed: {}",
            String::from_utf8_lossy(&run.stderr)
        );
        String::from_utf8(run.stdout)
            .unwrap()
            .lines()
            .map(|line| line.trim().to_string())
            .filter(|line| !line.is_empty())
            .collect()
    }

    fn assert_float_eq(actual: &str, expected: f32) {
        let val: f32 = actual.trim().parse().unwrap_or_else(|_| {
            panic!("could not parse '{actual}' as f32");
        });
        assert!(
            (val - expected).abs() < 1e-4,
            "expected {expected}, got {val}"
        );
    }

    #[test]
    fn numerical_tensor_add_is_elementwise_with_runtime_inputs() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            vec_f32(4),
            None,
        );
        let y = dag.add_node(
            decl,
            RiscOp::Load { name: "y".into() },
            vec![],
            vec_f32(4),
            None,
        );
        dag.add_node(decl, RiscOp::Add, vec![x, y], vec_f32(4), None);

        let outputs = compile_and_run_input_cases(
            &dag,
            "test_tensor_add",
            CodegenOptions::default(),
            &[vec![
                TestInput::new("x", &[4], &[1.0, 2.0, 3.0, 4.0]),
                TestInput::new("y", &[4], &[10.0, 20.0, 30.0, 40.0]),
            ]],
        );

        assert_eq!(outputs, vec!["11.000000 22.000000 33.000000 44.000000"]);
    }

    #[test]
    fn numerical_tensor_mul_is_elementwise_with_runtime_inputs() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            vec_f32(4),
            None,
        );
        let y = dag.add_node(
            decl,
            RiscOp::Load { name: "y".into() },
            vec![],
            vec_f32(4),
            None,
        );
        dag.add_node(decl, RiscOp::Mul, vec![x, y], vec_f32(4), None);

        let outputs = compile_and_run_input_cases(
            &dag,
            "test_tensor_mul",
            CodegenOptions::default(),
            &[vec![
                TestInput::new("x", &[4], &[1.5, 2.0, 3.0, 4.0]),
                TestInput::new("y", &[4], &[10.0, 20.0, 30.0, 40.0]),
            ]],
        );

        assert_eq!(outputs, vec!["15.000000 40.000000 90.000000 160.000000"]);
    }

    #[test]
    fn numerical_add_const() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        let b = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 2.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(decl, RiscOp::Add, vec![a, b], scalar_f32(), None);
        let out = compile_and_run(&dag, "test_add");
        assert_float_eq(&out, 3.0);
    }

    #[test]
    fn numerical_neg() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 5.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(decl, RiscOp::Neg, vec![a], scalar_f32(), None);
        let out = compile_and_run(&dag, "test_neg");
        assert_float_eq(&out, -5.0);
    }

    #[test]
    fn numerical_mul() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 3.0),
            vec![],
            scalar_f32(),
            None,
        );
        let b = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 4.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(decl, RiscOp::Mul, vec![a, b], scalar_f32(), None);
        let out = compile_and_run(&dag, "test_mul");
        assert_float_eq(&out, 12.0);
    }

    #[test]
    fn numerical_exp_zero() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 0.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(decl, RiscOp::Exp, vec![a], scalar_f32(), None);
        let out = compile_and_run(&dag, "test_exp");
        assert_float_eq(&out, 1.0);
    }

    #[test]
    fn numerical_sqrt() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 9.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(decl, RiscOp::Sqrt, vec![a], scalar_f32(), None);
        let out = compile_and_run(&dag, "test_sqrt");
        assert_float_eq(&out, 3.0);
    }

    #[test]
    fn numerical_log_e() {
        // log(e) = 1.0
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, std::f64::consts::E),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(decl, RiscOp::Log, vec![a], scalar_f32(), None);
        let out = compile_and_run(&dag, "test_log");
        assert_float_eq(&out, 1.0);
    }

    #[test]
    fn numerical_sin_zero() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 0.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(decl, RiscOp::Sin, vec![a], scalar_f32(), None);
        let out = compile_and_run(&dag, "test_sin");
        assert_float_eq(&out, 0.0);
    }

    #[test]
    fn numerical_cmplt_true() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        let b = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 2.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(
            decl,
            RiscOp::Compare(ComparisonKind::CmpLt),
            vec![a, b],
            TensorType {
                dims: vec![],
                precision: Prim::Bool,
            },
            None,
        );
        let out = compile_and_run(&dag, "test_cmplt");
        assert_float_eq(&out, 1.0);
    }

    #[test]
    fn numerical_cmplt_false() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 5.0),
            vec![],
            scalar_f32(),
            None,
        );
        let b = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 2.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(
            decl,
            RiscOp::Compare(ComparisonKind::CmpLt),
            vec![a, b],
            TensorType {
                dims: vec![],
                precision: Prim::Bool,
            },
            None,
        );
        let out = compile_and_run(&dag, "test_cmplt_f");
        assert_float_eq(&out, 0.0);
    }

    #[test]
    fn numerical_sum_vector() {
        // sum([2, 2, 2]) = 6
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(vec_f32(3).precision, 2.0),
            vec![],
            vec_f32(3),
            None,
        );
        dag.add_node(
            decl,
            RiscOp::Sum {
                axis: 0,
                accumulator: chelis_types::types::Prim::F32,
            },
            vec![a],
            scalar_f32(),
            None,
        );
        let out = compile_and_run(&dag, "test_sum");
        assert_float_eq(&out, 6.0);
    }

    #[test]
    fn numerical_add_then_mul() {
        // (1 + 2) * 4 = 12
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        let b = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 2.0),
            vec![],
            scalar_f32(),
            None,
        );
        let c = dag.add_node(decl, RiscOp::Add, vec![a, b], scalar_f32(), None);
        let d = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 4.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(decl, RiscOp::Mul, vec![c, d], scalar_f32(), None);
        let out = compile_and_run(&dag, "test_chain");
        assert_float_eq(&out, 12.0);
    }

    #[test]
    fn numerical_max_elem() {
        // max(3, 7) = 7
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 3.0),
            vec![],
            scalar_f32(),
            None,
        );
        let b = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 7.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(decl, RiscOp::MaxElem, vec![a, b], scalar_f32(), None);
        let out = compile_and_run(&dag, "test_maxe");
        assert_float_eq(&out, 7.0);
    }

    #[test]
    fn numerical_relu() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, -3.0),
            vec![],
            scalar_f32(),
            None,
        );
        let relu = tier2::lower_relu(decl.into(), &mut dag, x, &scalar_f32(), None);
        let out = compile_and_run(&dag, "test_relu");
        assert_float_eq(&out, 0.0);
        assert!(matches!(dag.get(relu).unwrap().op, RiscOp::Relu));
    }

    #[test]
    fn numerical_sigmoid_zero() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 0.0),
            vec![],
            scalar_f32(),
            None,
        );
        let _ = tier2::lower_sigmoid(decl.into(), &mut dag, x, &scalar_f32(), None);
        let out = compile_and_run(&dag, "test_sigmoid");
        assert_float_eq(&out, 0.5);
    }

    #[test]
    fn numerical_max_reduce() {
        // max_reduce([5, 5, 5]) over axis 0 = 5
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(vec_f32(3).precision, 5.0),
            vec![],
            vec_f32(3),
            None,
        );
        dag.add_node(
            decl,
            RiscOp::MaxReduce { axis: 0 },
            vec![a],
            scalar_f32(),
            None,
        );
        let out = compile_and_run(&dag, "test_maxr");
        assert_float_eq(&out, 5.0);
    }

    #[test]
    fn numerical_cast_identity() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 42.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(
            decl,
            RiscOp::Cast {
                new_precision: Prim::F32,
            },
            vec![a],
            scalar_f32(),
            None,
        );
        let out = compile_and_run(&dag, "test_cast");
        assert_float_eq(&out, 42.0);
    }

    #[test]
    fn numerical_add_then_mul_then_sum() {
        // vec of 3 ones + vec of 3 twos = [3,3,3], * vec of 3 threes = [9,9,9], sum = 27
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(vec_f32(3).precision, 1.0),
            vec![],
            vec_f32(3),
            None,
        );
        let b = dag.add_node(
            decl,
            RiscOp::synth_const(vec_f32(3).precision, 2.0),
            vec![],
            vec_f32(3),
            None,
        );
        let c = dag.add_node(decl, RiscOp::Add, vec![a, b], vec_f32(3), None);
        let d = dag.add_node(
            decl,
            RiscOp::synth_const(vec_f32(3).precision, 3.0),
            vec![],
            vec_f32(3),
            None,
        );
        let e = dag.add_node(decl, RiscOp::Mul, vec![c, d], vec_f32(3), None);
        dag.add_node(
            decl,
            RiscOp::Sum {
                axis: 0,
                accumulator: chelis_types::types::Prim::F32,
            },
            vec![e],
            scalar_f32(),
            None,
        );
        let out = compile_and_run(&dag, "test_pipe");
        assert_float_eq(&out, 27.0);
    }

    // ---- Movement op numerical tests ----

    fn assert_floats_eq(actual: &str, expected: &[f32]) {
        let vals: Vec<f32> = actual
            .split_whitespace()
            .map(|s| {
                s.parse::<f32>()
                    .unwrap_or_else(|_| panic!("could not parse '{s}' as f32"))
            })
            .collect();
        assert_eq!(
            vals.len(),
            expected.len(),
            "expected {} values, got {}",
            expected.len(),
            vals.len()
        );
        for (i, (a, e)) in vals.iter().zip(expected.iter()).enumerate() {
            assert!((a - e).abs() < 1e-4, "index {i}: expected {e}, got {a}");
        }
    }

    /// One generated DAG consumes exact runtime extents in both valid empty
    /// domains and overflow domains. UBSan makes the retired product fold an
    /// executable failure even where an optimizer could discard its comparison.
    #[test]
    fn checked_c_metadata_dag_reshape_executes_under_ubsan() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let input = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            TensorType {
                dims: vec![DimInfo::Named("n".into(), None)],
                precision: Prim::F64,
            },
            None,
        );
        let mut inputs = vec![input];
        for name in ["a", "b", "c"] {
            inputs.push(dag.add_node(
                decl,
                RiscOp::Load { name: name.into() },
                vec![],
                TensorType {
                    dims: vec![],
                    precision: Prim::Int64,
                },
                None,
            ));
        }
        let output = dag.add_node(
            decl,
            RiscOp::Reshape {
                new_shape: vec![RtDim::Node(1), RtDim::Node(2), RtDim::Node(3)],
            },
            inputs,
            TensorType {
                dims: ["r", "s", "t"]
                    .map(|name| DimInfo::Named(name.into(), None))
                    .to_vec(),
                precision: Prim::F64,
            },
            None,
        );
        dag.set_roots(vec![output]);
        let result = codegen(&dag, "checked_reshape").unwrap();
        let source = &result.c_source;
        assert!(!source.contains("chelis_tensor_stride("));
        assert!(source.contains("chelis_tensor_byte_count("));
        assert!(!source.contains("__stride_"));
        let check = source
            .find("chelis_tensor_check_reshape(")
            .expect("checked reshape entry");
        let allocation = source[check..]
            .find("chelis_alloc(")
            .expect("destination allocation");
        let copy = source[check..].find("memcpy(").expect("reshape copy");
        assert!(
            allocation < copy,
            "validation must precede allocation and copying"
        );
        let input_values = result
            .input_labels
            .iter()
            .map(|label| match label.as_str() {
                "x" => "x",
                "a" => "a",
                "b" => "b",
                "c" => "c",
                other => panic!("unexpected input {other}"),
            })
            .collect::<Vec<_>>()
            .join(", ");
        let driver = format!(
            r#"
#include "chelis_runtime.h"
void checked_reshape(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);
static chelis_tensor *extent(int64_t n) {{
    return chelis_scalar_tensor(chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)n));
}}
int main(int argc, char **argv) {{
    int mode = argc > 1 ? atoi(argv[1]) : 0;
    int64_t dims[3] = {{2, 3, 1}};
    int64_t count = 6;
    switch (mode) {{
        case 1: dims[0] = INT64_MAX; dims[1] = INT64_MAX; dims[2] = 0; count = 0; break;
        case 2: dims[0] = 0; dims[1] = 3; dims[2] = 4; count = 0; break;
        case 3: dims[0] = 2; dims[1] = 0; dims[2] = 4; count = 0; break;
        case 4: dims[0] = INT64_MAX; dims[1] = 2; break;
        case 5: dims[0] = INT64_MAX / 8 + 1; dims[1] = 1; break;
        case 6: dims[0] = 0; dims[1] = INT64_MAX; dims[2] = INT64_MAX; break;
        case 7: dims[0] = 5; dims[1] = 1; break;
    }}
    chelis_tensor *x = chelis_alloc(1, &count, CHELIS_DTYPE_F64);
    chelis_tensor_write *guard = chelis_tensor_begin_write(x);
    chelis_fill_scalar(guard, chelis_scalar_from_bits(CHELIS_DTYPE_F64, UINT64_C(0x7ff8123456789abc)));
    chelis_tensor_end_write(guard);
    chelis_tensor *a = extent(dims[0]), *b = extent(dims[1]), *c = extent(dims[2]);
    chelis_tensor *inputs[] = {{ {input_values} }};
    chelis_tensor *outputs[1] = {{NULL}};
    checked_reshape(inputs, 4, outputs, 1);
    if (chelis_tensor_rank(outputs[0]) != 3) return 2;
    for (int axis = 0; axis < 3; ++axis)
        if (chelis_tensor_shape(outputs[0], axis) != dims[axis]) return 3;
    chelis_read_view view = chelis_tensor_read_view(outputs[0]);
    if (view.count != count || (count == 0 && view.data != NULL)) return 4;
    for (int64_t i = 0; i < count; ++i) {{
        uint64_t bits; memcpy(&bits, (const unsigned char*)view.data + i * 8, 8);
        if (bits != UINT64_C(0x7ff8123456789abc)) return 5;
    }}
    chelis_tensor_release(outputs[0]);
    chelis_tensor_release(x); chelis_tensor_release(a);
    chelis_tensor_release(b); chelis_tensor_release(c);
    puts("CHECKED DAG PASS");
    return 0;
}}
"#
        );
        let tmp = tempfile::tempdir().unwrap();
        stage_runtime(tmp.path());
        write_temp_file(tmp.path(), "model.c", source);
        write_temp_file(tmp.path(), "main.c", &driver);
        let binary = tmp.path().join("probe");
        let toolchain = test_toolchain(result.requirements);
        let mut command = Command::new(&toolchain.compiler);
        apply_c_test_flags(&mut command);
        command.args([
            "-O2",
            "-fsanitize=undefined",
            "-fno-sanitize-recover=undefined",
        ]);
        command.args(&toolchain.compile_flags);
        command
            .arg(tmp.path().join("main.c"))
            .arg(tmp.path().join("model.c"));
        add_runtime_link(&mut command, tmp.path());
        command.args(&toolchain.link_flags).arg("-o").arg(&binary);
        let compiled = command.output().unwrap();
        assert!(
            compiled.status.success(),
            "{}\n{source}",
            String::from_utf8_lossy(&compiled.stderr)
        );
        for mode in 0..8 {
            let run = Command::new(&binary)
                .arg(mode.to_string())
                .output()
                .unwrap();
            let stderr = String::from_utf8_lossy(&run.stderr);
            if mode < 4 {
                assert!(run.status.success(), "mode {mode}: {stderr}\n{source}");
                assert_eq!(
                    String::from_utf8_lossy(&run.stdout).trim(),
                    "CHECKED DAG PASS"
                );
            } else {
                assert!(!run.status.success(), "invalid mode {mode} succeeded");
                let brand = if mode == 7 { "Domain:" } else { "Overflow:" };
                assert!(stderr.contains(brand), "mode {mode}: {stderr}");
            }
            assert!(!stderr.contains("runtime error:"), "mode {mode}: {stderr}");
        }
    }

    #[test]
    fn numerical_reshape_preserves_values() {
        // Create a 1D tensor [1,1,1,1,1,1] (const fills all), reshape to 2x3
        // We use const value 1.0 for a vec of 6, reshape to 2x3 -> still 6 ones
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(vec_f32(6).precision, 1.0),
            vec![],
            vec_f32(6),
            None,
        );
        dag.add_node(
            decl,
            RiscOp::Reshape {
                new_shape: vec![RtDim::Lit(2), RtDim::Lit(3)],
            },
            vec![a],
            mat_f32(2, 3),
            None,
        );
        let out = compile_and_run(&dag, "test_reshape");
        assert_floats_eq(&out, &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn numerical_reshape_then_add() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(vec_f32(6).precision, 1.0),
            vec![],
            vec_f32(6),
            None,
        );
        let reshaped = dag.add_node(
            decl,
            RiscOp::Reshape {
                new_shape: vec![RtDim::Lit(2), RtDim::Lit(3)],
            },
            vec![a],
            mat_f32(2, 3),
            None,
        );
        let b = dag.add_node(
            decl,
            RiscOp::synth_const(mat_f32(2, 3).precision, 2.0),
            vec![],
            mat_f32(2, 3),
            None,
        );
        dag.add_node(decl, RiscOp::Add, vec![reshaped, b], mat_f32(2, 3), None);
        let out = compile_and_run(&dag, "test_reshape_add");
        assert_floats_eq(&out, &[3.0, 3.0, 3.0, 3.0, 3.0, 3.0]);
    }

    #[test]
    fn numerical_expand_add_broadcast() {
        // Create 1D tensor [3.0] (size 1), expand to size 3 (stride 0 broadcast),
        // add with a 1D tensor [1.0, 1.0, 1.0] -> [4.0, 4.0, 4.0]
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(vec_f32(1).precision, 3.0),
            vec![],
            vec_f32(1),
            None,
        );
        let expanded = dag.add_node(
            decl,
            RiscOp::Expand {
                axis: 0,
                size: chelis_ir::dag::RtDim::Lit(3),
            },
            vec![a],
            vec_f32(3),
            None,
        );
        let b = dag.add_node(
            decl,
            RiscOp::synth_const(vec_f32(3).precision, 1.0),
            vec![],
            vec_f32(3),
            None,
        );
        dag.add_node(decl, RiscOp::Add, vec![expanded, b], vec_f32(3), None);
        let out = compile_and_run(&dag, "test_expand_add");
        assert_floats_eq(&out, &[4.0, 4.0, 4.0]);
    }

    #[test]
    fn numerical_permute_transpose() {
        // Create a 2x3 matrix (all 2.0), permute to 3x2 -> still all 2.0 but shape changes
        // Since const fills all elements with same value, we verify shape via size
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(mat_f32(2, 3).precision, 2.0),
            vec![],
            mat_f32(2, 3),
            None,
        );
        let permuted = dag.add_node(
            decl,
            RiscOp::Permute { axes: vec![1, 0] },
            vec![a],
            mat_f32(3, 2),
            None,
        );
        // Sum along axis 0 (3 rows) to get vec of 2: each column sums 3 x 2.0 = 6.0
        dag.add_node(
            decl,
            RiscOp::Sum {
                axis: 0,
                accumulator: chelis_types::types::Prim::F32,
            },
            vec![permuted],
            vec_f32(2),
            None,
        );
        let out = compile_and_run(&dag, "test_permute");
        assert_floats_eq(&out, &[6.0, 6.0]);
    }

    #[test]
    fn checked_cast_identity_materializes_a_noncontiguous_source() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let input = dag.add_node(
            decl,
            RiscOp::Load {
                name: "input".into(),
            },
            vec![],
            mat_f32(2, 3),
            None,
        );
        let permuted = dag.add_node(
            decl,
            RiscOp::Permute { axes: vec![1, 0] },
            vec![input],
            mat_f32(3, 2),
            None,
        );
        let cast = dag.add_node(
            decl,
            RiscOp::Cast {
                new_precision: Prim::F32,
            },
            vec![permuted],
            mat_f32(3, 2),
            None,
        );
        dag.add_root(cast);

        let output = compile_and_run_input_cases(
            &dag,
            "checked_cast_identity_noncontiguous",
            CodegenOptions::default(),
            &[vec![TestInput::new(
                "input",
                &[2, 3],
                &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            )]],
        );
        assert_eq!(
            output,
            ["1.000000 4.000000 2.000000 5.000000 3.000000 6.000000"]
        );
    }

    #[test]
    fn numerical_sparse_gather_and_scatter_add_with_duplicate_indices() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let values = dag.add_node(
            decl,
            RiscOp::Load {
                name: "values".into(),
            },
            vec![],
            tensor_ty(&[4, 2], Prim::F32),
            None,
        );
        let indices = dag.add_node(
            decl,
            RiscOp::Load {
                name: "indices".into(),
            },
            vec![],
            tensor_ty(&[3], Prim::Int32),
            None,
        );
        let gathered = dag.add_node(
            decl,
            RiscOp::Gather {
                axis: 0,
                batch_rank: 0,
            },
            vec![values, indices],
            tensor_ty(&[3, 2], Prim::F32),
            None,
        );
        let target = dag.add_node(
            decl,
            RiscOp::Load {
                name: "target".into(),
            },
            vec![],
            tensor_ty(&[4, 2], Prim::F32),
            None,
        );
        let updates = dag.add_node(
            decl,
            RiscOp::Load {
                name: "updates".into(),
            },
            vec![],
            tensor_ty(&[3, 2], Prim::F32),
            None,
        );
        let scattered = dag.add_node(
            decl,
            RiscOp::ScatterAdd {
                axis: 0,
                batch_rank: 0,
            },
            vec![target, indices, updates],
            tensor_ty(&[4, 2], Prim::F32),
            None,
        );
        dag.add_root(gathered);
        dag.add_root(scattered);

        let result = codegen(&dag, "test_sparse").unwrap();
        let tmp = tempfile::tempdir().unwrap();
        stage_runtime(tmp.path());
        write_temp_file(tmp.path(), "model.c", &result.c_source);
        let mut input_lines = Vec::new();
        input_lines.push(format!(
            "chelis_tensor *inputs[{}] = {{0}};",
            result.input_labels.len()
        ));
        for (slot, label) in result.input_labels.iter().enumerate() {
            match label.as_str() {
                "values" => input_lines.push(
                    r#"int64_t shape_values[2] = { 4, 2 };
    chelis_tensor *values = chelis_alloc(2, shape_values, CHELIS_DTYPE_F32);
    float values_data[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    chelis_tensor_write *values_guard = chelis_tensor_begin_write(values);
    chelis_write_view values_view = chelis_tensor_write_view(values_guard);
    for (int i = 0; i < 8; i++) ((float*)values_view.data)[i] = values_data[i];
    chelis_tensor_end_write(values_guard);
    inputs[SLOT] = values;"#
                        .replace("SLOT", &slot.to_string()),
                ),
                // #476: a CHELIS_DTYPE_I32 index tensor stores i32 values
                // bit-packed into the float-typed `data` buffer; they MUST be
                // written through an `(int32_t*)` cast, not as floats. The
                // pre-fix fixture wrote an f32 through an i32 tensor (the FLOAT
                // 2.0, whose i32 reinterpretation is 0x40000000), which only
                // round-tripped because the buggy reader did `(int)data[i]` and
                // truncated the float back. Writing the i32 value directly is
                // what real generated input code and the runtime do.
                "indices" => input_lines.push(
                    r#"int64_t shape_indices[1] = { 3 };
    chelis_tensor *indices = chelis_alloc(1, shape_indices, CHELIS_DTYPE_I32);
    chelis_tensor_write *indices_guard = chelis_tensor_begin_write(indices);
    chelis_write_view indices_view = chelis_tensor_write_view(indices_guard);
    int32_t *indices_i32 = (int32_t*)indices_view.data;
    indices_i32[0] = 0; indices_i32[1] = 2; indices_i32[2] = 0;
    chelis_tensor_end_write(indices_guard);
    inputs[SLOT] = indices;"#
                        .replace("SLOT", &slot.to_string()),
                ),
                "target" => input_lines.push(
                    r#"int64_t shape_target[2] = { 4, 2 };
    chelis_tensor *target = chelis_alloc(2, shape_target, CHELIS_DTYPE_F32);
    inputs[SLOT] = target;"#
                        .replace("SLOT", &slot.to_string()),
                ),
                "updates" => input_lines.push(
                    r#"int64_t shape_updates[2] = { 3, 2 };
    chelis_tensor *updates = chelis_alloc(2, shape_updates, CHELIS_DTYPE_F32);
    float updates_data[6] = { 1, 10, 2, 20, 3, 30 };
    chelis_tensor_write *updates_guard = chelis_tensor_begin_write(updates);
    chelis_write_view updates_view = chelis_tensor_write_view(updates_guard);
    for (int i = 0; i < 6; i++) ((float*)updates_view.data)[i] = updates_data[i];
    chelis_tensor_end_write(updates_guard);
    inputs[SLOT] = updates;"#
                        .replace("SLOT", &slot.to_string()),
                ),
                other => panic!("unexpected input label {other}"),
            }
        }
        let main_c = format!(
            r#"
#include "chelis_runtime.h"
void test_sparse(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);
int main(void) {{
    {inputs}
    chelis_tensor *outputs[2] = {{0}};
    test_sparse(inputs, {n_in}, outputs, 2);
    chelis_read_view output0_view = chelis_tensor_read_view(outputs[0]);
    chelis_read_view output1_view = chelis_tensor_read_view(outputs[1]);
    for (int i = 0; i < output0_view.count; i++) {{
        if (i > 0) printf(" ");
        printf("%.6f", ((const float*)output0_view.data)[i]);
    }}
    printf(" |");
    for (int i = 0; i < output1_view.count; i++) {{
        if (i > 0) printf(" ");
        printf("%.6f", ((const float*)output1_view.data)[i]);
    }}
    printf("\n");
    for (int i = 0; i < {n_in}; i++) chelis_tensor_release(inputs[i]);
    chelis_tensor_release(outputs[0]);
    chelis_tensor_release(outputs[1]);
    return 0;
}}
"#,
            inputs = input_lines.join("\n    "),
            n_in = result.input_labels.len()
        );
        write_temp_file(tmp.path(), "main.c", &main_c);
        let bin_path = tmp.path().join("test_sparse");
        let toolchain = test_toolchain(result.requirements);
        let mut compile_cmd = Command::new(&toolchain.compiler);
        apply_c_test_flags(&mut compile_cmd);
        compile_cmd.args(["-O2"]);
        compile_cmd.args(&toolchain.compile_flags);
        compile_cmd.arg(tmp.path().join("main.c").to_str().unwrap());
        compile_cmd.arg(tmp.path().join("model.c").to_str().unwrap());
        add_runtime_link(&mut compile_cmd, tmp.path());
        compile_cmd.args(&toolchain.link_flags);
        compile_cmd.arg("-o");
        compile_cmd.arg(bin_path.to_str().unwrap());
        let compile = compile_cmd.output().unwrap();
        assert!(
            compile.status.success(),
            "gcc failed:\nstderr: {}\nC source:\n{}",
            String::from_utf8_lossy(&compile.stderr),
            result.c_source
        );
        let run = Command::new(bin_path).output().unwrap();
        assert!(
            run.status.success(),
            "binary failed: {}",
            String::from_utf8_lossy(&run.stderr)
        );
        assert_eq!(
            String::from_utf8(run.stdout).unwrap().trim(),
            "1.000000 2.000000 5.000000 6.000000 1.000000 2.000000 |4.000000 40.000000 0.000000 0.000000 2.000000 20.000000 0.000000 0.000000"
        );
    }

    #[test]
    fn numerical_pad_with_runtime_input() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            vec_f32(3),
            None,
        );
        dag.add_node(
            decl,
            RiscOp::pad(
                vec![(RtDim::Lit(1), RtDim::Lit(2))],
                chelis_types::scalar_from_f64("pad", Prim::F32, -1.0).unwrap(),
            ),
            vec![x],
            vec_f32(6),
            None,
        );
        let lines = compile_and_run_input_cases(
            &dag,
            "test_pad_runtime",
            CodegenOptions::default(),
            &[vec![TestInput::new("x", &[3], &[1.0, 2.0, 3.0])]],
        );
        assert_eq!(
            lines,
            vec!["-1.000000 1.000000 2.000000 3.000000 -1.000000 -1.000000"]
        );
    }

    #[test]
    fn numerical_shrink_with_runtime_input() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            vec_f32(5),
            None,
        );
        dag.add_node(
            decl,
            RiscOp::Shrink {
                bounds: vec![(RtDim::Lit(1), RtDim::Lit(4))],
            },
            vec![x],
            vec_f32(3),
            None,
        );
        let lines = compile_and_run_input_cases(
            &dag,
            "test_shrink_runtime",
            CodegenOptions::default(),
            &[vec![TestInput::new("x", &[5], &[5.0, 6.0, 7.0, 8.0, 9.0])]],
        );
        assert_eq!(lines, vec!["6.000000 7.000000 8.000000"]);
    }

    #[test]
    fn numerical_stride_with_runtime_input() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            vec_f32(5),
            None,
        );
        dag.add_node(
            decl,
            RiscOp::Stride {
                strides: vec![RtDim::Lit(2)],
            },
            vec![x],
            vec_f32(3),
            None,
        );
        let lines = compile_and_run_input_cases(
            &dag,
            "test_stride_runtime",
            CodegenOptions::default(),
            &[vec![TestInput::new("x", &[5], &[1.0, 2.0, 3.0, 4.0, 5.0])]],
        );
        assert_eq!(lines, vec!["1.000000 3.000000 5.000000"]);
    }

    #[test]
    fn numerical_large_vector_add_then_sum() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(vec_f32(1024).precision, 1.0),
            vec![],
            vec_f32(1024),
            None,
        );
        let b = dag.add_node(
            decl,
            RiscOp::synth_const(vec_f32(1024).precision, 2.0),
            vec![],
            vec_f32(1024),
            None,
        );
        let c = dag.add_node(decl, RiscOp::Add, vec![a, b], vec_f32(1024), None);
        dag.add_node(
            decl,
            RiscOp::Sum {
                axis: 0,
                accumulator: chelis_types::types::Prim::F32,
            },
            vec![c],
            scalar_f32(),
            None,
        );
        let out = compile_and_run(&dag, "test_large_add_sum");
        assert_float_eq(&out, 3072.0);
    }

    #[test]
    fn parameterized_input_helper_supports_multiple_cases() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            scalar_f32(),
            None,
        );
        let one = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(decl, RiscOp::Add, vec![x, one], scalar_f32(), None);
        let lines = compile_and_run_input_cases(
            &dag,
            "test_cases",
            CodegenOptions::default(),
            &[
                vec![TestInput::new("x", &[], &[2.0])],
                vec![TestInput::new("x", &[], &[5.0])],
            ],
        );
        assert_eq!(lines, vec!["3.000000", "6.000000"]);
    }

    #[test]
    fn symbolic_batch_codegen_reuses_one_artifact_for_multiple_input_shapes() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let batch_vec = TensorType {
            dims: vec![DimInfo::Named("batch".to_string(), None)],
            precision: Prim::F32,
        };
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            batch_vec.clone(),
            None,
        );
        let y = dag.add_node(
            decl,
            RiscOp::Load { name: "y".into() },
            vec![],
            batch_vec.clone(),
            None,
        );
        dag.add_node(decl, RiscOp::Add, vec![x, y], batch_vec, None);

        let result = codegen(&dag, "test_symbolic_batch").unwrap();
        assert_eq!(result.symbolic_dims, vec!["batch"]);
        assert!(
            result
                .c_source
                .contains("int64_t batch = chelis_tensor_shape(inputs[0], 0);")
        );
        assert!(
            result
                .c_source
                .contains("chelis_tensor_shape(inputs[1], 0) != chelis_tensor_shape(inputs[0], 0)")
        );

        let lines = compile_and_run_input_cases(
            &dag,
            "test_symbolic_batch",
            CodegenOptions::default(),
            &[
                vec![
                    TestInput::new("x", &[2], &[1.0, 2.0]),
                    TestInput::new("y", &[2], &[3.0, 4.0]),
                ],
                vec![
                    TestInput::new("x", &[3], &[1.0, 2.0, 3.0]),
                    TestInput::new("y", &[3], &[4.0, 5.0, 6.0]),
                ],
            ],
        );
        assert_eq!(
            lines,
            vec!["4.000000 6.000000", "5.000000 7.000000 9.000000"]
        );
    }

    #[test]
    fn symbolic_matmul_codegen_reuses_one_artifact_for_multiple_batch_sizes() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a_ty = TensorType {
            dims: vec![DimInfo::Named("batch".to_string(), None), DimInfo::Lit(3)],
            precision: Prim::F32,
        };
        let b_ty = TensorType {
            dims: vec![DimInfo::Lit(3), DimInfo::Lit(2)],
            precision: Prim::F32,
        };
        let a = dag.add_node(
            decl,
            RiscOp::Load { name: "a".into() },
            vec![],
            a_ty.clone(),
            None,
        );
        let b = dag.add_node(
            decl,
            RiscOp::Load { name: "b".into() },
            vec![],
            b_ty.clone(),
            None,
        );
        let out = tier2::lower_matmul(decl.into(), &mut dag, a, b, &a_ty, &b_ty, None);
        dag.add_root(out);

        let result = codegen(&dag, "test_symbolic_matmul").unwrap();
        assert_eq!(result.symbolic_dims, vec!["batch"]);
        assert!(
            result
                .c_source
                .contains("int64_t batch = chelis_tensor_shape(inputs[0], 0);")
        );

        let lines = compile_and_run_input_cases(
            &dag,
            "test_symbolic_matmul",
            CodegenOptions::default(),
            &[
                vec![
                    TestInput::new("a", &[1, 3], &[1.0, 2.0, 3.0]),
                    TestInput::new("b", &[3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                ],
                vec![
                    TestInput::new("a", &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                    TestInput::new("b", &[3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                ],
            ],
        );
        assert_eq!(
            lines,
            vec![
                "22.000000 28.000000",
                "22.000000 28.000000 49.000000 64.000000"
            ]
        );
    }

    #[test]
    fn symbolic_batched_matmul_preserves_primitive_arithmetic() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a_ty = TensorType {
            dims: vec![
                DimInfo::Named("batch".to_string(), None),
                DimInfo::Named("heads".to_string(), None),
                DimInfo::Named("seq".to_string(), None),
                DimInfo::Lit(3),
            ],
            precision: Prim::F32,
        };
        let b_ty = TensorType {
            dims: vec![
                DimInfo::Named("batch".to_string(), None),
                DimInfo::Named("heads".to_string(), None),
                DimInfo::Lit(3),
                DimInfo::Lit(2),
            ],
            precision: Prim::F32,
        };
        let a = dag.add_node(
            decl,
            RiscOp::Load { name: "a".into() },
            vec![],
            a_ty.clone(),
            None,
        );
        let b = dag.add_node(
            decl,
            RiscOp::Load { name: "b".into() },
            vec![],
            b_ty.clone(),
            None,
        );
        let out = tier2::lower_matmul(decl.into(), &mut dag, a, b, &a_ty, &b_ty, None);
        dag.add_root(out);

        let result = codegen_with_options(
            &dag,
            "test_symbolic_batched_matmul",
            CodegenOptions {
                use_blas: true,
                ..CodegenOptions::default()
            },
        )
        .unwrap();
        assert!(!result.requirements.needs_blas);
        assert!(
            result
                .c_source
                .contains("int64_t seq = chelis_tensor_shape(inputs[0], 2);")
        );
        assert!(!result.c_source.contains("cblas_sgemm"));
        assert!(result.c_source.contains("__sum_level_"));

        let lines = compile_and_run_input_cases(
            &dag,
            "test_symbolic_batched_matmul",
            CodegenOptions {
                use_blas: true,
                ..CodegenOptions::default()
            },
            &[vec![
                TestInput::new(
                    "a",
                    &[1, 2, 2, 3],
                    &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0, 0.5, 2.0, 3.5, -2.0, 1.0],
                ),
                TestInput::new(
                    "b",
                    &[1, 2, 3, 2],
                    &[
                        1.0, 0.0, -1.0, 2.0, 0.5, 3.0, 2.0, -2.0, 1.0, 1.5, -0.5, 4.0,
                    ],
                ),
            ]],
        );
        assert_eq!(
            lines,
            vec!["0.500000 13.000000 2.000000 28.000000 -2.500000 10.750000 4.500000 -6.000000"]
        );
    }

    #[test]
    fn noncontiguous_matrix_slice_stays_on_generic_matmul_path() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let base_a = dag.add_node(
            decl,
            RiscOp::Load {
                name: "base_a".into(),
            },
            vec![],
            mat_f32(3, 2),
            None,
        );
        let a_ty = mat_f32(2, 3);
        let a = dag.add_node(
            decl,
            RiscOp::Permute { axes: vec![1, 0] },
            vec![base_a],
            a_ty.clone(),
            None,
        );
        let b_ty = mat_f32(3, 4);
        let b = dag.add_node(
            decl,
            RiscOp::Load { name: "b".into() },
            vec![],
            b_ty.clone(),
            None,
        );
        let out = tier2::lower_matmul(decl.into(), &mut dag, a, b, &a_ty, &b_ty, None);
        dag.add_root(out);

        let result = codegen_with_options(
            &dag,
            "test_noncontiguous_matmul",
            CodegenOptions {
                use_blas: true,
                ..CodegenOptions::default()
            },
        )
        .unwrap();
        assert!(
            !result.c_source.contains("cblas_sgemm("),
            "non-contiguous matrix slices must not bypass IR specialization \
             through the legacy codegen-time BLAS detector"
        );
        assert!(
            result.c_source.contains("parallel for"),
            "non-contiguous matmul should remain on the generic reduction path"
        );
    }

    #[test]
    fn symbolic_preamble_checks_every_non_canonical_occurrence() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let symbolic = TensorType {
            dims: vec![DimInfo::Named("batch".to_string(), None)],
            precision: Prim::F32,
        };
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            symbolic.clone(),
            None,
        );
        let y = dag.add_node(
            decl,
            RiscOp::Load { name: "y".into() },
            vec![],
            symbolic.clone(),
            None,
        );
        let z = dag.add_node(
            decl,
            RiscOp::Load { name: "z".into() },
            vec![],
            symbolic.clone(),
            None,
        );
        let xy = dag.add_node(decl, RiscOp::Add, vec![x, y], symbolic.clone(), None);
        let xyz = dag.add_node(decl, RiscOp::Add, vec![xy, z], symbolic, None);
        dag.add_root(xyz);

        let result = codegen(&dag, "test_symbolic_occurrences").unwrap();
        assert!(
            result
                .c_source
                .contains("int64_t batch = chelis_tensor_shape(inputs[0], 0);")
        );
        // The guard reads BOTH operands from the class's own witnesses rather
        // than comparing against the declared variable, so that a member
        // scoped to one signature is never compared with a variable the
        // occurrence walk declared for another (chelis#1277 C2.4). The
        // property this row names - every non-canonical occurrence is checked
        // against the canonical - is unchanged.
        let canonical = "chelis_tensor_shape(inputs[0], 0)";
        assert!(
            result
                .c_source
                .contains(&format!("chelis_tensor_shape(inputs[1], 0) != {canonical}"))
        );
        assert!(
            result
                .c_source
                .contains(&format!("chelis_tensor_shape(inputs[2], 0) != {canonical}"))
        );
    }

    #[test]
    fn multiple_roots_return_multiple_outputs() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        let b = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 2.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_root(a);
        dag.add_root(b);
        let result = codegen(&dag, "test_multi").unwrap();

        let tmp = tempfile::tempdir().unwrap();
        stage_runtime(tmp.path());
        write_temp_file(tmp.path(), "model.c", &result.c_source);
        let main_c = r#"
#include "chelis_runtime.h"
void test_multi(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);
int main(void) {
    chelis_tensor *outputs[2] = {0};
    test_multi(NULL, 0, outputs, 2);
    printf("%.1f %.1f\n",
           ((const float*)chelis_tensor_read_view(outputs[0]).data)[0],
           ((const float*)chelis_tensor_read_view(outputs[1]).data)[0]);
    chelis_tensor_release(outputs[0]);
    chelis_tensor_release(outputs[1]);
    return 0;
}
"#;
        write_temp_file(tmp.path(), "main.c", main_c);
        let bin_path = tmp.path().join("test_multi");
        let mut cmd = Command::new(crate::toolchain::c_compiler());
        apply_c_test_flags(&mut cmd);
        cmd.args(["-O2", "-lm"])
            .arg(tmp.path().join("main.c").to_str().unwrap())
            .arg(tmp.path().join("model.c").to_str().unwrap());
        add_runtime_link(&mut cmd, tmp.path());
        cmd.arg("-lm").arg("-o").arg(bin_path.to_str().unwrap());
        let out = cmd.output().unwrap();
        assert!(
            out.status.success(),
            "gcc failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let run = Command::new(bin_path).output().unwrap();
        assert!(run.status.success());
        assert_eq!(String::from_utf8(run.stdout).unwrap().trim(), "1.0 2.0");
    }

    #[test]
    fn output_from_load_is_materialized_not_borrowed() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            vec_f32(2),
            None,
        );
        let result = codegen(&dag, "test_load_copy").unwrap();

        let tmp = tempfile::tempdir().unwrap();
        stage_runtime(tmp.path());
        write_temp_file(tmp.path(), "model.c", &result.c_source);
        let main_c = r#"
#include "chelis_runtime.h"
void test_load_copy(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);
int main(void) {
    int64_t shape[1] = {2};
    chelis_tensor *input = chelis_alloc(1, shape, CHELIS_DTYPE_F32);
    chelis_tensor_write *guard = chelis_tensor_begin_write(input);
    chelis_write_view input_view = chelis_tensor_write_view(guard);
    ((float*)input_view.data)[0] = 3.0f;
    ((float*)input_view.data)[1] = 4.0f;
    chelis_tensor_end_write(guard);
    chelis_tensor *inputs[1] = {input};
    chelis_tensor *outputs[1] = {0};
    test_load_copy(inputs, 1, outputs, 1);
    printf("%d %d %.1f %.1f\n",
           outputs[0] == input,
           chelis_tensor_read_view(outputs[0]).data == chelis_tensor_read_view(input).data,
           ((const float*)chelis_tensor_read_view(outputs[0]).data)[0],
           ((const float*)chelis_tensor_read_view(outputs[0]).data)[1]);
    chelis_tensor_release(outputs[0]);
    chelis_tensor_release(input);
    return 0;
}
"#;
        write_temp_file(tmp.path(), "main.c", main_c);
        let bin_path = tmp.path().join("test_load_copy");
        let mut cmd = Command::new(crate::toolchain::c_compiler());
        apply_c_test_flags(&mut cmd);
        cmd.args(["-O2", "-lm"])
            .arg(tmp.path().join("main.c").to_str().unwrap())
            .arg(tmp.path().join("model.c").to_str().unwrap());
        add_runtime_link(&mut cmd, tmp.path());
        cmd.arg("-lm").arg("-o").arg(bin_path.to_str().unwrap());
        let out = cmd.output().unwrap();
        assert!(
            out.status.success(),
            "gcc failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let run = Command::new(bin_path).output().unwrap();
        assert!(
            run.status.success(),
            "materialized-load harness failed: {}",
            String::from_utf8_lossy(&run.stderr)
        );
        assert_eq!(String::from_utf8(run.stdout).unwrap().trim(), "0 0 3.0 4.0");
    }

    #[test]
    fn generated_code_compiles_with_platform_parallelism() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        let b = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 2.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(decl, RiscOp::Add, vec![a, b], scalar_f32(), None);
        let out = compile_and_run_with_flags(&dag, "test_platform_parallelism", &[]);
        assert_float_eq(&out, 3.0);
    }

    #[test]
    fn canonical_matmul_codegen_compiles_and_runs() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(mat_f32(2, 3).precision, 1.0),
            vec![],
            mat_f32(2, 3),
            None,
        );
        let b = dag.add_node(
            decl,
            RiscOp::synth_const(mat_f32(3, 4).precision, 1.0),
            vec![],
            mat_f32(3, 4),
            None,
        );
        let ea = dag.add_node(
            decl,
            RiscOp::Expand {
                axis: 2,
                size: chelis_ir::dag::RtDim::Lit(4),
            },
            vec![a],
            TensorType {
                dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
                precision: Prim::F32,
            },
            None,
        );
        let eb = dag.add_node(
            decl,
            RiscOp::Expand {
                axis: 0,
                size: chelis_ir::dag::RtDim::Lit(2),
            },
            vec![b],
            TensorType {
                dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
                precision: Prim::F32,
            },
            None,
        );
        let mul = dag.add_node(
            decl,
            RiscOp::Mul,
            vec![ea, eb],
            TensorType {
                dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
                precision: Prim::F32,
            },
            None,
        );
        dag.add_node(
            decl,
            RiscOp::Sum {
                axis: 1,
                accumulator: chelis_types::types::Prim::F32,
            },
            vec![mul],
            mat_f32(2, 4),
            None,
        );
        let result = codegen_with_options(
            &dag,
            "test_blas",
            CodegenOptions {
                use_blas: true,
                ..CodegenOptions::default()
            },
        )
        .unwrap();
        assert!(!result.requirements.needs_blas);
        assert!(!result.c_source.contains("cblas_sgemm("));

        let tmp = tempfile::tempdir().unwrap();
        stage_runtime(tmp.path());
        write_temp_file(tmp.path(), "model.c", &result.c_source);
        let main_c = r#"
#include "chelis_runtime.h"
void test_blas(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);
int main(void) {
    chelis_tensor *outputs[1] = {0};
    test_blas(NULL, 0, outputs, 1);
    const float *data = (const float*)chelis_tensor_read_view(outputs[0]).data;
    for (int i = 0; i < 8; i++) if (data[i] != 3.0f) return 1;
    chelis_tensor_release(outputs[0]);
    return 0;
}
"#;
        write_temp_file(tmp.path(), "main.c", main_c);
        let toolchain = test_toolchain(result.requirements);
        let mut cmd = Command::new(&toolchain.compiler);
        apply_c_test_flags(&mut cmd);
        cmd.args(["-O2"])
            .args(&toolchain.compile_flags)
            .arg(tmp.path().join("main.c").to_str().unwrap())
            .arg(tmp.path().join("model.c").to_str().unwrap());
        add_runtime_link(&mut cmd, tmp.path());
        cmd.args(&toolchain.link_flags)
            .arg("-o")
            .arg(tmp.path().join("test_blas").to_str().unwrap());
        let out = cmd.output().unwrap();
        assert!(
            out.status.success(),
            "gcc failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let run = Command::new(tmp.path().join("test_blas")).output().unwrap();
        assert!(run.status.success(), "canonical matmul output: {run:?}");
    }

    #[test]
    fn numerical_matmul_fallback() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(mat_f32(2, 3).precision, 2.0),
            vec![],
            mat_f32(2, 3),
            None,
        );
        let b = dag.add_node(
            decl,
            RiscOp::synth_const(mat_f32(3, 4).precision, 0.5),
            vec![],
            mat_f32(3, 4),
            None,
        );
        let ea = dag.add_node(
            decl,
            RiscOp::Expand {
                axis: 2,
                size: chelis_ir::dag::RtDim::Lit(4),
            },
            vec![a],
            TensorType {
                dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
                precision: Prim::F32,
            },
            None,
        );
        let eb = dag.add_node(
            decl,
            RiscOp::Expand {
                axis: 0,
                size: chelis_ir::dag::RtDim::Lit(2),
            },
            vec![b],
            TensorType {
                dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
                precision: Prim::F32,
            },
            None,
        );
        let mul = dag.add_node(
            decl,
            RiscOp::Mul,
            vec![ea, eb],
            TensorType {
                dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
                precision: Prim::F32,
            },
            None,
        );
        dag.add_node(
            decl,
            RiscOp::Sum {
                axis: 1,
                accumulator: chelis_types::types::Prim::F32,
            },
            vec![mul],
            mat_f32(2, 4),
            None,
        );
        let out = compile_and_run_with_codegen_options(
            &dag,
            "test_matmul_fallback",
            CodegenOptions::default(),
            &[],
        );
        assert_floats_eq(&out, &[3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]);
    }

    #[test]
    fn canonical_matmul_numerics_ignore_blas_hint() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(mat_f32(2, 3).precision, 2.0),
            vec![],
            mat_f32(2, 3),
            None,
        );
        let b = dag.add_node(
            decl,
            RiscOp::synth_const(mat_f32(3, 4).precision, 0.5),
            vec![],
            mat_f32(3, 4),
            None,
        );
        let ea = dag.add_node(
            decl,
            RiscOp::Expand {
                axis: 2,
                size: chelis_ir::dag::RtDim::Lit(4),
            },
            vec![a],
            TensorType {
                dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
                precision: Prim::F32,
            },
            None,
        );
        let eb = dag.add_node(
            decl,
            RiscOp::Expand {
                axis: 0,
                size: chelis_ir::dag::RtDim::Lit(2),
            },
            vec![b],
            TensorType {
                dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
                precision: Prim::F32,
            },
            None,
        );
        let mul = dag.add_node(
            decl,
            RiscOp::Mul,
            vec![ea, eb],
            TensorType {
                dims: vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)],
                precision: Prim::F32,
            },
            None,
        );
        dag.add_node(
            decl,
            RiscOp::Sum {
                axis: 1,
                accumulator: chelis_types::types::Prim::F32,
            },
            vec![mul],
            mat_f32(2, 4),
            None,
        );
        let out = compile_and_run_with_codegen_options(
            &dag,
            "test_matmul_blas_hint",
            CodegenOptions {
                use_blas: true,
                ..CodegenOptions::default()
            },
            &[],
        );
        assert_floats_eq(&out, &[3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]);
    }

    // ---- Level 3b SIMD math tests ----

    /// Build a fused exp(add(a, b)) DAG and return the fused version.
    fn build_fused_exp_add_dag(n: usize) -> Dag {
        use chelis_ir::fuse::fuse;
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::Load { name: "a".into() },
            vec![],
            vec_f32(n),
            None,
        );
        let b = dag.add_node(
            decl,
            RiscOp::Load { name: "b".into() },
            vec![],
            vec_f32(n),
            None,
        );
        let add = dag.add_node(decl, RiscOp::Add, vec![a, b], vec_f32(n), None);
        dag.add_node(decl, RiscOp::Exp, vec![add], vec_f32(n), None);
        fuse(&dag)
    }

    #[test]
    fn simd_math_fused_kernel_generates_sleef_path() {
        // Force MathLib::Sleef in codegen and verify the generated C contains the
        // expected Sleef macros and the scalar tail loop.
        let dag = build_fused_exp_add_dag(16);
        let result = codegen_with_options(
            &dag,
            "test_sleef_codegen",
            CodegenOptions {
                math_lib_override: Some(MathLib::Sleef),
                ..CodegenOptions::default()
            },
        )
        .unwrap();
        // The Sleef guard must be present.
        assert!(
            result.c_source.contains("#ifdef CHELIS_HAS_SLEEF"),
            "expected #ifdef CHELIS_HAS_SLEEF in generated C:\n{}",
            result.c_source
        );
        // The Sleef macro for exp must appear.
        assert!(
            result.c_source.contains("CHELIS_EXPF8("),
            "expected CHELIS_EXPF8( in generated C:\n{}",
            result.c_source
        );
        // The 8-wide loadu load must appear.
        assert!(
            result.c_source.contains("_mm256_loadu_ps("),
            "expected _mm256_loadu_ps( in generated C:\n{}",
            result.c_source
        );
        // The storeu store must appear.
        assert!(
            result.c_source.contains("_mm256_storeu_ps("),
            "expected _mm256_storeu_ps( in generated C:\n{}",
            result.c_source
        );
        // The scalar tail loop guard must appear (for n % 8 != 0 case).
        assert!(
            result.c_source.contains("for (; __i < t"),
            "expected scalar tail loop in generated C:\n{}",
            result.c_source
        );
        // The scalar fallback (Level-1 path) must also be present inside #else.
        assert!(
            result.c_source.contains("#else"),
            "expected #else guard in generated C:\n{}",
            result.c_source
        );
        // The math header include must appear.
        assert!(
            result.c_source.contains("#include \"chelis_math.h\""),
            "expected chelis_math.h include in generated C:\n{}",
            result.c_source
        );
    }

    #[test]
    fn simd_math_none_path_does_not_include_math_header() {
        // MathLib::None must NOT include chelis_math.h.
        let dag = build_fused_exp_add_dag(8);
        let result = codegen_with_options(
            &dag,
            "test_no_math_header",
            CodegenOptions {
                math_lib_override: Some(MathLib::None),
                ..CodegenOptions::default()
            },
        )
        .unwrap();
        assert!(
            !result.c_source.contains("#include \"chelis_math.h\""),
            "MathLib::None must not emit chelis_math.h include:\n{}",
            result.c_source
        );
        assert!(
            !result.c_source.contains("#ifdef CHELIS_HAS_SLEEF"),
            "MathLib::None must not emit Sleef guards:\n{}",
            result.c_source
        );
        // Positive: must still use Level-1 omp simd and scalar math functions.
        assert!(
            result.c_source.contains("#pragma omp parallel for simd"),
            "MathLib::None must emit Level-1 omp simd loop:\n{}",
            result.c_source
        );
        assert!(
            result.c_source.contains("expf("),
            "MathLib::None must emit scalar expf():\n{}",
            result.c_source
        );
    }

    #[test]
    fn simd_pure_arithmetic_fused_kernel_skips_sleef_path() {
        // A kernel with only Add (no math ops) must use the Level-1 path even when
        // MathLib::Sleef is forced, because has_math_ops() returns false.
        use chelis_ir::fuse::fuse;
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let a = dag.add_node(
            decl,
            RiscOp::Load { name: "a".into() },
            vec![],
            vec_f32(8),
            None,
        );
        let b = dag.add_node(
            decl,
            RiscOp::Load { name: "b".into() },
            vec![],
            vec_f32(8),
            None,
        );
        dag.add_node(decl, RiscOp::Add, vec![a, b], vec_f32(8), None);
        let dag = fuse(&dag);
        let result = codegen_with_options(
            &dag,
            "test_no_sleef_add",
            CodegenOptions {
                math_lib_override: Some(MathLib::Sleef),
                ..CodegenOptions::default()
            },
        )
        .unwrap();
        // Sleef guard must NOT appear — add-only kernel takes the Level-1 path.
        assert!(
            !result.c_source.contains("#ifdef CHELIS_HAS_SLEEF"),
            "pure-arithmetic kernel must not emit Sleef guard:\n{}",
            result.c_source
        );
        // The Level-1 omp simd loop must be present.
        assert!(
            result.c_source.contains("#pragma omp parallel for simd"),
            "pure-arithmetic kernel must use Level-1 omp simd loop:\n{}",
            result.c_source
        );
    }

    #[test]
    fn simd_math_fused_kernel_compiles_and_runs() {
        // Compile the Sleef-path C without -DCHELIS_HAS_SLEEF (so the #else Level-1
        // scalar path is compiled).  Run at n=7 (partial 8-wide block) and n=1024.
        for &n in &[7usize, 1024] {
            let dag = build_fused_exp_add_dag(n);
            // Generate C with Sleef path enabled so the generated source has the SIMD
            // block, but compile without -DCHELIS_HAS_SLEEF so the #else branch fires.
            let result = codegen_with_options(
                &dag,
                "test_simd_compile",
                CodegenOptions {
                    math_lib_override: Some(MathLib::Sleef),
                    ..CodegenOptions::default()
                },
            )
            .unwrap();

            let tmp = tempfile::tempdir().unwrap();
            stage_runtime(tmp.path());
            write_temp_file(tmp.path(), "model.c", &result.c_source);

            // Build the main harness: fill a with 0.5, b with 0.5, expect exp(1.0)=e^1.
            let main_c = format!(
                r#"
#include "chelis_runtime.h"
#include <math.h>
void test_simd_compile(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);
int main(void) {{
    int64_t shape[1] = {{ {n} }};
    chelis_tensor *ta = chelis_alloc(1, shape, CHELIS_DTYPE_F32);
    chelis_tensor *tb = chelis_alloc(1, shape, CHELIS_DTYPE_F32);
    chelis_tensor_write *ta_guard = chelis_tensor_begin_write(ta);
    chelis_tensor_write *tb_guard = chelis_tensor_begin_write(tb);
    chelis_write_view ta_view = chelis_tensor_write_view(ta_guard);
    chelis_write_view tb_view = chelis_tensor_write_view(tb_guard);
    for (int i = 0; i < {n}; i++) {{ ((float*)ta_view.data)[i] = 0.5f; ((float*)tb_view.data)[i] = 0.5f; }}
    chelis_tensor_end_write(ta_guard);
    chelis_tensor_end_write(tb_guard);
    chelis_tensor *inputs[2] = {{ta, tb}};
    chelis_tensor *outputs[1] = {{0}};
    test_simd_compile(inputs, 2, outputs, 1);
    float expected = expf(1.0f);
    for (int i = 0; i < {n}; i++) {{
        float diff = ((const float*)chelis_tensor_read_view(outputs[0]).data)[i] - expected;
        if (diff < 0) diff = -diff;
        if (diff > 1e-5f) {{ return 1; }}
    }}
    chelis_tensor_release(ta);
    chelis_tensor_release(tb);
    chelis_tensor_release(outputs[0]);
    return 0;
}}
"#
            );
            write_temp_file(tmp.path(), "main.c", &main_c);
            let bin_path = tmp.path().join("test_simd_compile");

            let toolchain = test_toolchain(result.requirements);
            let mut cmd = Command::new(&toolchain.compiler);
            apply_c_test_flags(&mut cmd);
            cmd.arg("-O2");
            cmd.args(simd_isa_test_flags());
            cmd.args(&toolchain.compile_flags);
            cmd.arg(tmp.path().join("main.c").to_str().unwrap());
            cmd.arg(tmp.path().join("model.c").to_str().unwrap());
            add_runtime_link(&mut cmd, tmp.path());
            cmd.args(&toolchain.link_flags);
            cmd.arg("-lm").arg("-o").arg(bin_path.to_str().unwrap());
            let compile_out = cmd.output().unwrap();
            assert!(
                compile_out.status.success(),
                "gcc failed at n={n}:\nstderr: {}\nC source:\n{}",
                String::from_utf8_lossy(&compile_out.stderr),
                result.c_source
            );

            let run_out = Command::new(bin_path.to_str().unwrap()).output().unwrap();
            assert!(
                run_out.status.success(),
                "binary failed at n={n}: {}",
                String::from_utf8_lossy(&run_out.stderr)
            );
        }
    }

    /// Acceptance oracle for Level 3b SIMD math.
    ///
    /// Runs exp(add(a, b)) via the Sleef-path C (compiled without CHELIS_HAS_SLEEF
    /// so the #else Level-1 scalar path executes) and verifies the result matches
    /// a pure scalar reference implementation within 1 ULP (< 2e-7 relative error).
    #[test]
    fn simd_math_fused_kernel_matches_scalar_output() {
        // Reference: compute exp(a+b) entirely in Rust.
        let n = 33usize; // Not a multiple of 8; exercises both the main block and the tail.
        let a_data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1f32 - 1.5f32).collect();
        let b_data: Vec<f32> = (0..n).map(|i| (i as f32) * -0.05f32 + 0.5f32).collect();
        let reference: Vec<f32> = a_data
            .iter()
            .zip(b_data.iter())
            .map(|(a, b)| (a + b).exp())
            .collect();

        let dag = build_fused_exp_add_dag(n);
        let result = codegen_with_options(
            &dag,
            "test_oracle",
            CodegenOptions {
                math_lib_override: Some(MathLib::Sleef),
                ..CodegenOptions::default()
            },
        )
        .unwrap();

        let tmp = tempfile::tempdir().unwrap();
        stage_runtime(tmp.path());
        write_temp_file(tmp.path(), "model.c", &result.c_source);

        // Build input initializers from the test vectors. Issue #252
        // sibling: reuse the exact-bits fill helper so the C program
        // computes its reference from byte-identical f32 inputs, not a
        // lossy `{:.8}f` truncation that could push the post-exp output
        // past the 1-ULP (`2e-7`) tolerance asserted below.
        let a_init: String = a_data
            .iter()
            .enumerate()
            .map(|(i, v)| harness_input_fill_line(&format!("ta[{i}]"), *v))
            .collect::<Vec<_>>()
            .join("\n    ");
        let b_init: String = b_data
            .iter()
            .enumerate()
            .map(|(i, v)| harness_input_fill_line(&format!("tb[{i}]"), *v))
            .collect::<Vec<_>>()
            .join("\n    ");

        let main_c = format!(
            r#"
#include "chelis_runtime.h"
#include <math.h>
#include <stdio.h>
void test_oracle(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);
int main(void) {{
    int64_t shape[1] = {{ {n} }};
    chelis_tensor *ta = chelis_alloc(1, shape, CHELIS_DTYPE_F32);
    chelis_tensor *tb = chelis_alloc(1, shape, CHELIS_DTYPE_F32);
    {a_init}
    {b_init}
    chelis_tensor *inputs[2] = {{ta, tb}};
    chelis_tensor *outputs[1] = {{0}};
    test_oracle(inputs, 2, outputs, 1);
    for (int i = 0; i < {n}; i++) {{
        printf("%.8f\n", ((const float*)chelis_tensor_read_view(outputs[0]).data)[i]);
    }}
    chelis_tensor_release(ta);
    chelis_tensor_release(tb);
    chelis_tensor_release(outputs[0]);
    return 0;
}}
"#
        );
        write_temp_file(tmp.path(), "main.c", &main_c);
        let bin_path = tmp.path().join("test_oracle");

        let toolchain = test_toolchain(result.requirements);
        let mut cmd = Command::new(&toolchain.compiler);
        apply_c_test_flags(&mut cmd);
        cmd.arg("-O2");
        cmd.args(simd_isa_test_flags());
        cmd.args(&toolchain.compile_flags);
        cmd.arg(tmp.path().join("main.c").to_str().unwrap());
        cmd.arg(tmp.path().join("model.c").to_str().unwrap());
        add_runtime_link(&mut cmd, tmp.path());
        cmd.args(&toolchain.link_flags);
        cmd.arg("-lm").arg("-o").arg(bin_path.to_str().unwrap());
        let compile_out = cmd.output().unwrap();
        assert!(
            compile_out.status.success(),
            "gcc failed:\nstderr: {}\nC source:\n{}",
            String::from_utf8_lossy(&compile_out.stderr),
            result.c_source
        );

        let run_out = Command::new(bin_path.to_str().unwrap()).output().unwrap();
        assert!(
            run_out.status.success(),
            "binary failed: {}",
            String::from_utf8_lossy(&run_out.stderr)
        );
        let stdout = String::from_utf8(run_out.stdout).unwrap();
        let actual: Vec<f32> = stdout
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| l.trim().parse::<f32>().expect("expected f32 output"))
            .collect();
        assert_eq!(
            actual.len(),
            n,
            "expected {n} output values, got {}",
            actual.len()
        );
        for (i, (got, expected)) in actual.iter().zip(reference.iter()).enumerate() {
            let rel_err = ((got - expected) / expected).abs();
            assert!(
                rel_err < 2e-7,
                "index {i}: expected {expected}, got {got}, relative error {rel_err:.2e} exceeds 2e-7"
            );
        }
    }

    #[test]
    fn simd_single_math_op_generates_sleef_path() {
        // A single Exp op (not fused) should emit the Sleef AVX2 path when MathLib::Sleef.
        // Previously this fell through to the omp-simd scalar path.
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let n: usize = 9; // 8 + 1 to exercise both SIMD and scalar tail
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            vec_f32(n),
            None,
        );
        dag.add_node(decl, RiscOp::Exp, vec![x], vec_f32(n), None);

        let result = codegen_with_options(
            &dag,
            "test_single_exp",
            CodegenOptions {
                math_lib_override: Some(MathLib::Sleef),
                ..CodegenOptions::default()
            },
        )
        .unwrap();
        let src = &result.c_source;
        assert!(
            src.contains("#ifdef CHELIS_HAS_SLEEF"),
            "single Exp with Sleef should emit #ifdef CHELIS_HAS_SLEEF:\n{src}"
        );
        assert!(
            src.contains("CHELIS_EXPF8("),
            "single Exp with Sleef should emit CHELIS_EXPF8:\n{src}"
        );
        assert!(
            src.contains("_mm256_loadu_ps("),
            "single Exp with Sleef should emit _mm256_loadu_ps:\n{src}"
        );
        assert!(
            src.contains("_mm256_storeu_ps("),
            "single Exp with Sleef should emit _mm256_storeu_ps:\n{src}"
        );
    }

    #[test]
    fn simd_single_math_op_sleef_compiles_and_runs() {
        let n: usize = 9; // exercises both 8-wide SIMD block and 1-element scalar tail
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            vec_f32(n),
            None,
        );
        dag.add_node(decl, RiscOp::Exp, vec![x], vec_f32(n), None);

        let result = codegen_with_options(
            &dag,
            "test_single_exp_run",
            CodegenOptions {
                math_lib_override: Some(MathLib::Sleef),
                ..CodegenOptions::default()
            },
        )
        .unwrap();

        let tmp = tempfile::tempdir().unwrap();
        stage_runtime(tmp.path());
        write_temp_file(tmp.path(), "model.c", &result.c_source);

        // Input: x[i] = 0.3 * i + 0.1 (non-trivial, spans SIMD + tail)
        let inputs: Vec<f32> = (0..n).map(|i| 0.3 * i as f32 + 0.1).collect();
        let reference: Vec<f32> = inputs.iter().map(|v| v.exp()).collect();
        // Issue #252 sibling: exact-bits fill so the C program's input is
        // byte-identical to the Rust `reference`, not `{:.8}f`-truncated.
        let x_init = inputs
            .iter()
            .enumerate()
            .map(|(i, v)| harness_input_fill_line(&format!("tx[{i}]"), *v))
            .collect::<Vec<_>>()
            .join("\n    ");

        let main_c = format!(
            r#"
#include "chelis_runtime.h"
#include <math.h>
#include <stdio.h>
void test_single_exp_run(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);
int main(void) {{
    int64_t shape[1] = {{ {n} }};
    chelis_tensor *tx = chelis_alloc(1, shape, CHELIS_DTYPE_F32);
    {x_init}
    chelis_tensor *inputs[1] = {{tx}};
    chelis_tensor *outputs[1] = {{0}};
    test_single_exp_run(inputs, 1, outputs, 1);
    for (int i = 0; i < {n}; i++) {{
        printf("%.8f\n", ((const float*)chelis_tensor_read_view(outputs[0]).data)[i]);
    }}
    chelis_tensor_release(tx);
    chelis_tensor_release(outputs[0]);
    return 0;
}}
"#
        );
        write_temp_file(tmp.path(), "main.c", &main_c);
        let bin_path = tmp.path().join("test_single_exp_run");

        let toolchain = test_toolchain(result.requirements);
        let mut cmd = Command::new(&toolchain.compiler);
        apply_c_test_flags(&mut cmd);
        cmd.arg("-O2");
        cmd.args(simd_isa_test_flags());
        cmd.args(&toolchain.compile_flags);
        cmd.arg(tmp.path().join("main.c").to_str().unwrap());
        cmd.arg(tmp.path().join("model.c").to_str().unwrap());
        add_runtime_link(&mut cmd, tmp.path());
        cmd.args(&toolchain.link_flags);
        cmd.arg("-lm").arg("-o").arg(bin_path.to_str().unwrap());
        let compile_out = cmd.output().unwrap();
        assert!(
            compile_out.status.success(),
            "gcc failed:\nstderr: {}\nC source:\n{}",
            String::from_utf8_lossy(&compile_out.stderr),
            result.c_source
        );

        let run_out = Command::new(bin_path.to_str().unwrap()).output().unwrap();
        assert!(
            run_out.status.success(),
            "binary failed: {}",
            String::from_utf8_lossy(&run_out.stderr)
        );
        let stdout = String::from_utf8(run_out.stdout).unwrap();
        let actual: Vec<f32> = stdout
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| l.trim().parse::<f32>().expect("f32"))
            .collect();
        assert_eq!(actual.len(), n);
        for (i, (got, expected)) in actual.iter().zip(reference.iter()).enumerate() {
            let rel_err = ((got - expected) / expected).abs();
            assert!(
                rel_err < 2e-7,
                "index {i}: expected {expected}, got {got}, rel_err {rel_err:.2e}"
            );
        }
    }

    // ---- ADVERSARIAL TESTS: static linkage, C compilation, and scalar builtin coverage ----

    /// E: Published authored functions retain external linkage when globals
    /// cause `main` emission. Compiler-owned tensor helpers remain `static
    /// void`, and monomorphized specializations remain translation-unit local.
    #[test]
    fn adv_host_program_with_globals_keeps_authored_exports_external() {
        use chelis_ir::ConcreteHostType as HostType;
        use chelis_ir::host::{
            ConcreteHostBinding as HostBinding, ConcreteHostExpr as HostExpr,
            ConcreteHostExprKind as HostExprKind, ConcreteHostFunction as HostFunction,
            ConcreteHostParam as HostParam, ConcreteHostProgram as HostProgram, HostTensorHelper,
        };

        let mut helper_dag = Dag::new();
        let helper_dag_decl = helper_dag.declare("test");
        helper_dag.add_node(
            helper_dag_decl,
            RiscOp::synth_const(scalar_f32().precision, 2.0),
            vec![],
            scalar_f32(),
            None,
        );

        let helper = HostTensorHelper {
            name: "prog__fn__tensor_0".to_string(),
            dag: helper_dag,
            inputs: vec![],
            output: scalar_f32(),
            specialization: None,
            summary_rejection: None,
        };

        let func = HostFunction {
            helper_result_claim_axes: Vec::new(),
            entry_contract: Default::default(),
            name: "my_func".to_string(),
            params: vec![HostParam {
                name: "x".to_string(),
                ty: HostType::Float64,
            }],
            ret_ty: HostType::Float64,
            body: HostExpr::new(HostExprKind::Float(0.0)),
            tensor_helpers: vec![helper],
            origin: chelis_ir::host::HostFunctionOrigin::Authored,
            specialization: None,
            summary_rejections: Vec::new(),
        };

        // Adding a global binding triggers `main` emission.
        let program = HostProgram {
            globals: vec![HostBinding {
                name: "__g".to_string(),
                display_name: None,
                display_roots: Vec::new(),
                ty: HostType::Int64,
                value: HostExpr::new(HostExprKind::Int(1)),
            }],
            global_tensor_helpers: vec![],
            functions: vec![func],
            summary_rejections: Vec::new(),
            adt_layouts: Vec::new(),
        };

        let result = codegen_host_program(&program, "prog").unwrap();
        let src = &result.c_source;
        let my_func = authored_c_symbol("my_func");

        // Tensor helper must be `static void` (never static inline — it uses the DAG kernel sig)
        assert!(
            src.contains(&format!("static void {my_func}__tensor_0__private(")),
            "tensor helper must be `static void` even in globals mode;\ngenerated source:\n{src}"
        );
        // The published header declares this authored function external, so
        // the definition must have matching external linkage even with main.
        assert!(
            src.contains(&format!("double {my_func}(double x)")),
            "authored export must keep an external definition;\ngenerated source:\n{src}"
        );
        assert!(
            !src.contains(&format!("static inline double {my_func}("))
                && !src.contains(&format!("static double {my_func}(")),
            "authored export must not become translation-unit local;\ngenerated source:\n{src}"
        );
    }

    /// F: Generate a host program with a tensor helper and compile it with -shared -fPIC.
    /// This confirms that `static` linkage on helpers doesn't break shared-library builds.
    #[test]
    fn adv_host_program_compiles_as_shared_library_with_fpic() {
        use chelis_ir::ConcreteHostType as HostType;
        use chelis_ir::host::{
            ConcreteHostExpr as HostExpr, ConcreteHostExprKind as HostExprKind,
            ConcreteHostFunction as HostFunction, ConcreteHostParam as HostParam,
            ConcreteHostProgram as HostProgram, HostTensorHelper,
        };

        let mut helper_dag = Dag::new();
        let helper_dag_decl = helper_dag.declare("test");
        helper_dag.add_node(
            helper_dag_decl,
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );

        let helper = HostTensorHelper {
            name: "lib__exported_fn__tensor_0".to_string(),
            dag: helper_dag,
            inputs: vec![],
            output: scalar_f32(),
            specialization: None,
            summary_rejection: None,
        };

        let func = HostFunction {
            helper_result_claim_axes: Vec::new(),
            entry_contract: Default::default(),
            name: "exported_fn".to_string(),
            params: vec![HostParam {
                name: "x".to_string(),
                ty: HostType::Float64,
            }],
            ret_ty: HostType::Float64,
            body: HostExpr::new(HostExprKind::Float(0.0)),
            tensor_helpers: vec![helper],
            origin: chelis_ir::host::HostFunctionOrigin::Authored,
            specialization: None,
            summary_rejections: Vec::new(),
        };

        // No globals → external linkage for functions (library mode)
        let program = HostProgram {
            globals: vec![],
            global_tensor_helpers: vec![],
            functions: vec![func],
            summary_rejections: Vec::new(),
            adt_layouts: Vec::new(),
        };

        let result = codegen_host_program(&program, "lib").unwrap();
        let src = &result.c_source;

        let tmp = tempfile::tempdir().unwrap();
        stage_runtime(tmp.path());
        write_temp_file(tmp.path(), "lib.c", src);

        let so_path = tmp.path().join("lib.so");
        let mut cmd = Command::new(crate::toolchain::c_compiler());
        apply_c_test_flags(&mut cmd);
        cmd.args(["-shared", "-fPIC", "-O2", "-std=c11"])
            .arg("-I")
            .arg(tmp.path())
            .arg(tmp.path().join("lib.c").to_str().unwrap())
            .arg("-o")
            .arg(so_path.to_str().unwrap())
            .arg("-lm");
        let out = cmd.output().unwrap();
        assert!(
            out.status.success(),
            "shared-library compilation with -shared -fPIC failed:\nstderr: {}\nC source:\n{src}",
            String::from_utf8_lossy(&out.stderr)
        );
        assert!(so_path.exists(), "shared library file not produced");
    }

    /// A (adversarial): Verify each new scalar builtin emits the correct C function name.
    /// Tests that `tan` emits `tanf` (not `tanhf` or anything else).
    #[test]
    fn adv_tan_emits_tanf_not_tanhf() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(decl, RiscOp::Tan, vec![x], scalar_f32(), None);
        let result = codegen(&dag, "test_tan").unwrap();
        let src = &result.c_source;
        assert!(src.contains("tanf("), "tan must emit `tanf(`; got:\n{src}");
        assert!(
            !src.contains("tanhf("),
            "tan must NOT emit `tanhf(` (hyperbolic); got:\n{src}"
        );
    }

    /// A (adversarial): Verify abs emits `fabsf` not `absf`.
    #[test]
    fn adv_abs_emits_fabsf_not_absf() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
            RiscOp::Load { name: "x".into() },
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(decl, RiscOp::Abs, vec![x], scalar_f32(), None);
        let result = codegen(&dag, "test_abs").unwrap();
        let src = &result.c_source;
        // `fabsf` must appear; plain `absf` (which doesn't exist in C) must not.
        assert!(
            src.contains("fabsf("),
            "abs must emit `fabsf(`; got:\n{src}"
        );
        // Check that the string `absf(` only appears as `fabsf(` (the 'f' prefix is required)
        let absf_count = src.matches("fabsf(").count();
        let abs_occurrences: Vec<_> = src.match_indices("absf(").collect();
        for (pos, _) in &abs_occurrences {
            if *pos == 0 || src.as_bytes()[pos - 1] != b'f' {
                panic!("found bare `absf(` at position {pos}. Should be `fabsf(`;\n{src}");
            }
        }
        assert!(absf_count >= 1, "no `fabsf(` in output:\n{src}");
    }

    /// B (adversarial): Numerically check cos at a non-trivial point: cos(π/4) ≈ 0.7071.
    #[test]
    fn adv_numerical_cos_at_pi_over_4() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, std::f64::consts::FRAC_PI_4),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(decl, RiscOp::Cos, vec![x], scalar_f32(), None);
        let out = compile_and_run(&dag, "test_cos_pi4");
        let val: f32 = out.trim().parse().expect("float output");
        assert!(
            (val - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-5,
            "cos(π/4) expected ≈ {}, got {val}",
            std::f32::consts::FRAC_1_SQRT_2
        );
    }

    /// B (adversarial): atan(1.0) should be π/4 ≈ 0.7854.
    #[test]
    fn adv_numerical_atan_at_1() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, 1.0),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(decl, RiscOp::Atan, vec![x], scalar_f32(), None);
        let out = compile_and_run(&dag, "test_atan_1");
        let val: f32 = out.trim().parse().expect("float output");
        let expected = std::f32::consts::FRAC_PI_4;
        assert!(
            (val - expected).abs() < 1e-5,
            "atan(1.0) expected ≈ {expected} (π/4), got {val}"
        );
    }

    /// B (adversarial): floor(-1.3) should be -2.0.
    #[test]
    fn adv_numerical_floor_negative() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, -1.3),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(decl, RiscOp::Floor, vec![x], scalar_f32(), None);
        let out = compile_and_run(&dag, "test_floor_neg");
        let val: f32 = out.trim().parse().expect("float output");
        assert!(
            (val - (-2.0f32)).abs() < 1e-6,
            "floor(-1.3) expected -2.0, got {val}"
        );
    }

    /// B (adversarial): ceil(-1.7) should be -1.0.
    #[test]
    fn adv_numerical_ceil_negative() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, -1.7),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(decl, RiscOp::Ceil, vec![x], scalar_f32(), None);
        let out = compile_and_run(&dag, "test_ceil_neg");
        let val: f32 = out.trim().parse().expect("float output");
        assert!(
            (val - (-1.0f32)).abs() < 1e-6,
            "ceil(-1.7) expected -1.0, got {val}"
        );
    }

    /// B (adversarial): abs(-3.14) should be 3.14.
    #[test]
    fn adv_numerical_abs_negative_pi() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f32().precision, -std::f64::consts::PI),
            vec![],
            scalar_f32(),
            None,
        );
        dag.add_node(decl, RiscOp::Abs, vec![x], scalar_f32(), None);
        let out = compile_and_run(&dag, "test_abs_neg_pi");
        let val: f32 = out.trim().parse().expect("float output");
        assert!(
            (val - std::f32::consts::PI).abs() < 1e-5,
            "abs(-π) expected ≈ {}, got {val}",
            std::f32::consts::PI
        );
    }

    // ---- f64 tensor tests (v0.2.3) ----
    //
    // These cover the new double-precision tensor dtype end-to-end: codegen,
    // gcc compile, and numerical correctness at the ~15-digit precision that
    // the host f64 path reaches. Reading outputs as `double*` via the shared
    // data pointer relies on chelis_alloc sizing the allocation by 8 bytes
    // for CHELIS_DTYPE_F64 (see chelis_alloc in chelis-runtime).

    fn vec_f64(n: usize) -> TensorType {
        TensorType {
            dims: vec![DimInfo::Lit(n)],
            precision: Prim::F64,
        }
    }

    fn scalar_f64() -> TensorType {
        TensorType {
            dims: vec![],
            precision: Prim::F64,
        }
    }

    /// Compile a DAG whose single root is an f64 tensor, drive it from a C main
    /// that reads the output view as `double*`, and return the printed
    /// line-separated values with 17 significant digits each.
    fn compile_and_run_f64(dag: &Dag, func_name: &str, expected_size: usize) -> Vec<f64> {
        let result = codegen(dag, func_name).unwrap();

        let tmp = tempfile::tempdir().unwrap();
        stage_runtime(tmp.path());
        write_temp_file(tmp.path(), "model.c", &result.c_source);

        let main_c = format!(
            r#"
#include "chelis_runtime.h"
#include <stdio.h>
void {func_name}(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);
int main(void) {{
    chelis_tensor *outputs[1] = {{0}};
    {func_name}(NULL, 0, outputs, 1);
    const double *d = (const double*)chelis_tensor_read_view(outputs[0]).data;
    for (int i = 0; i < chelis_tensor_numel(outputs[0]); i++) {{
        printf("%.17g\n", d[i]);
    }}
    chelis_tensor_release(outputs[0]);
    return 0;
}}
"#
        );
        write_temp_file(tmp.path(), "main.c", &main_c);
        let bin_path = tmp.path().join("test_f64_bin");

        let toolchain = test_toolchain(result.requirements);
        let mut cmd = Command::new(&toolchain.compiler);
        apply_c_test_flags(&mut cmd);
        cmd.args(["-O2"]);
        cmd.args(&toolchain.compile_flags);
        cmd.arg(tmp.path().join("main.c").to_str().unwrap());
        cmd.arg(tmp.path().join("model.c").to_str().unwrap());
        add_runtime_link(&mut cmd, tmp.path());
        cmd.args(&toolchain.link_flags);
        cmd.arg("-o").arg(bin_path.to_str().unwrap());
        let compile = cmd.output().unwrap();
        assert!(
            compile.status.success(),
            "gcc failed:\nstderr: {}\nC source:\n{}",
            String::from_utf8_lossy(&compile.stderr),
            result.c_source
        );

        let run = Command::new(bin_path.to_str().unwrap()).output().unwrap();
        assert!(
            run.status.success(),
            "binary failed: {}",
            String::from_utf8_lossy(&run.stderr)
        );
        let stdout = String::from_utf8(run.stdout).unwrap();
        let values: Vec<f64> = stdout
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| {
                line.trim()
                    .parse::<f64>()
                    .unwrap_or_else(|_| panic!("could not parse '{line}' as f64"))
            })
            .collect();
        assert_eq!(
            values.len(),
            expected_size,
            "expected {expected_size} output values, got {}",
            values.len()
        );
        values
    }

    /// End-to-end: build a 4-element f64 vector add of [1.1, 2.2, 3.3, 4.4] +
    /// [0.5, 0.5, 0.5, 0.5], compile, run, and verify the exact double-precision
    /// sums. This would fail at f32 because 1.1 + 0.5 = 1.6 is not representable
    /// exactly — the double values differ from their f32-downcast counterparts
    /// at ~1e-8, which `diff < 1e-14` detects.
    #[test]
    fn f64_tensor_const_and_add_produces_correct_output() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        // Emit four individual consts because RiscOp::Const fills the whole
        // tensor with one scalar; we build the vectors element-by-element via
        // scalar const tensors that we then broadcast-add pairwise. Simpler:
        // two 4-element constants whose per-element values differ — we can
        // still fall back to emitting one Const per scalar tensor and Add'ing
        // them. For this test we just use two scalar-fill constants covering
        // {1.1,2.2,3.3,4.4} by treating each lane as a separate const... but
        // Const fills uniformly. Use Add chain of 4 scalars summed pointwise
        // instead: construct via (1.1 + 0.5) scalar, tile to [4], add to
        // [0.0, 0.0, 0.0, 0.0]. Simpler: just test one scalar per lane.
        let a = dag.add_node(
            decl,
            RiscOp::synth_const(vec_f64(4).precision, 1.1),
            vec![],
            vec_f64(4),
            None,
        );
        let b = dag.add_node(
            decl,
            RiscOp::synth_const(vec_f64(4).precision, 0.5),
            vec![],
            vec_f64(4),
            None,
        );
        dag.add_node(decl, RiscOp::Add, vec![a, b], vec_f64(4), None);

        let out = compile_and_run_f64(&dag, "test_f64_add", 4);
        // Both inputs are uniform-fill, so every lane == 1.6 exactly in double.
        for (i, &v) in out.iter().enumerate() {
            assert!(
                (v - 1.6_f64).abs() < 1e-14,
                "f64 add lane {i}: expected 1.6, got {v}"
            );
        }
    }

    /// Numerical correctness: sin(π/4) at double precision ≈ 0.7071067811865475
    /// (15–16 digits). The f32 path bottoms out around 7 digits, so a tolerance
    /// of 1e-14 distinguishes the two backends.
    #[test]
    fn f64_tensor_sin_at_pi_over_4() {
        let mut dag = Dag::new();
        let decl = dag.declare("test");
        let x = dag.add_node(
            decl,
            RiscOp::synth_const(scalar_f64().precision, std::f64::consts::FRAC_PI_4),
            vec![],
            scalar_f64(),
            None,
        );
        dag.add_node(decl, RiscOp::Sin, vec![x], scalar_f64(), None);

        let out = compile_and_run_f64(&dag, "test_f64_sin_pi4", 1);
        let expected = (std::f64::consts::FRAC_PI_4).sin();
        assert!(
            (out[0] - expected).abs() < 1e-14,
            "sin(π/4) at f64: expected {expected}, got {} (diff {})",
            out[0],
            (out[0] - expected).abs()
        );
    }
}
