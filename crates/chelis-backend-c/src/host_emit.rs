use chelis_ir::host::{
    HostBlasMatmulSummary, HostCallback, HostCallbackKind, HostExpr, HostExprKind, HostFunction,
    HostFunctionSpecialization, HostMatchArm, HostParam, HostProgram, HostSparseOpSummary,
    HostTensorHelper, HostTensorSpecialization, HostType,
};

/// Sparse-op kind discriminator for the C summary-derived emission path.
/// Mirrors the three `HostTensorSpecialization` / `HostFunctionSpecialization`
/// sparse variants without re-importing them at every call site.
#[derive(Clone, Copy)]
enum SparseSummaryKind {
    Gather,
    ScatterAdd,
    ScatterReplace,
}

use crate::emit::CEmitter;
use chelis_ir::dag::{DimExpr, DimInfo, RiscOp, TensorType};
use chelis_types::types::Prim;
use std::collections::{HashMap, HashSet};

pub fn emit_host_program(program: &HostProgram, program_name: &str) -> String {
    // Emit helpers and functions into a body buffer first so we can detect which
    // runtime headers they transitively require (e.g. `chelis_math.h` on macOS
    // when a helper uses the vForce vvexpf/vvlogf path).  The preamble is then
    // assembled with the right includes and prepended.  Without this, the
    // include-stripping in `append_helper` silently drops the inner emitter's
    // `#include "chelis_math.h"` and the resulting `main.c` calls vvexpf with
    // no declaration in scope.
    let mut body: Vec<String> = Vec::new();
    let mut helper_requirements = HelperRequirements::default();
    append_tensor_reshape_helper(&mut body);
    body.push(String::new());
    append_tensor_print_helper(&mut body);
    body.push(String::new());
    append_uniform_sample_helper(&mut body);
    body.push(String::new());
    append_tensor_math_helpers(&mut body);
    body.push(String::new());
    // When the program is a self-contained binary (has globals → `main` is emitted in the
    // same translation unit), user-defined host functions can be marked `static inline` so
    // the compiler can inline scalar helpers across calls under `-O2` / `-fPIC` without
    // requiring the downstream shell to supply `-flto -Wl,-Bsymbolic`. Object-mode builds
    // (no main) keep external linkage so the exported symbols remain callable from the
    // linker.
    let internal_linkage = !program.globals.is_empty();
    let emitted_names = emitted_function_names(program, program_name);
    let function_specializations = function_specializations(program);
    let header = emit_host_header_with_linkage(program, program_name, internal_linkage);
    if !header.is_empty() {
        body.push(header);
        body.push(String::new());
    }

    // Issue #352: top-level bindings referenced inside a compiled host
    // function would otherwise dangle -- `main()` declares every binding as
    // a local, so a def body's `w` had no declaration in scope and the
    // native compiler rejected the TU. Hoist captured bindings to file
    // scope; `emit_main` assigns them in binding order instead of declaring
    // locals, so function bodies and `main()` resolve the same object
    // (mirroring eval's load-closure, which serves the binding's value at
    // call time). Check-time name resolution rejects forward references
    // from a use site to a later binding, so every hoisted binding is
    // initialized before the first user call that reads it.
    let captured_globals = captured_global_names(program);
    if !captured_globals.is_empty() {
        body.push("// Top-level bindings captured by compiled functions (issue #352):".to_string());
        for name in &captured_globals {
            let binding = program
                .globals
                .iter()
                .find(|binding| binding.name == *name)
                .expect("captured global name comes from program.globals");
            body.push(format!("static {};", c_decl(&binding.ty, name)));
        }
        body.push(String::new());
    }

    for (index, helper) in program.global_tensor_helpers.iter().enumerate() {
        helper_requirements.merge(append_helper(
            &mut body,
            helper,
            &format!("{program_name}__global__tensor_{index}"),
        ));
    }
    for function in &program.functions {
        for (index, helper) in function.tensor_helpers.iter().enumerate() {
            let function_name = emitted_names
                .get(&function.name)
                .expect("host function emitted name");
            helper_requirements.merge(append_helper(
                &mut body,
                helper,
                &format!("{function_name}__tensor_{index}"),
            ));
        }
    }

    for function in &program.functions {
        emit_function(
            &mut body,
            function,
            emitted_names
                .get(&function.name)
                .expect("host function emitted name"),
            &emitted_names,
            &function_specializations,
            internal_linkage,
        );
        body.push(String::new());
    }

    if !program.globals.is_empty() {
        let hoisted: HashSet<&str> = captured_globals.iter().map(String::as_str).collect();
        emit_main(&mut body, program_name, program, &hoisted);
    }

    let mut out: Vec<String> = vec![
        "#include \"chelis_runtime.h\"".to_string(),
        "#include <assert.h>".to_string(),
        "#include <math.h>".to_string(),
    ];
    if helper_requirements.needs_blas_header {
        out.push("#include \"chelis_blas.h\"".to_string());
    }
    if helper_requirements.needs_math_header {
        out.push("#include \"chelis_math.h\"".to_string());
    }
    out.push(String::new());
    out.extend(body);
    out.join("\n")
}

fn emitted_function_name(program_name: &str, function_name: &str) -> String {
    if function_name == "main" {
        format!("{program_name}__main")
    } else {
        function_name.to_string()
    }
}

fn emitted_function_names(program: &HostProgram, program_name: &str) -> HashMap<String, String> {
    program
        .functions
        .iter()
        .map(|function| {
            (
                function.name.clone(),
                emitted_function_name(program_name, &function.name),
            )
        })
        .collect()
}

fn function_specializations(program: &HostProgram) -> HashMap<String, HostFunctionSpecialization> {
    program
        .functions
        .iter()
        .filter_map(|function| {
            function
                .specialization
                .clone()
                .map(|summary| (function.name.clone(), summary))
        })
        .collect()
}

fn append_uniform_sample_helper(out: &mut Vec<String>) {
    out.push(
        "static inline float chelis_uniform_sample_f32(uint64_t seed, uint64_t index, float low, float high) {"
            .to_string(),
    );
    out.push("    uint64_t x = seed ^ (index * 0x9E3779B97F4A7C15ULL);".to_string());
    out.push("    x ^= x >> 30;".to_string());
    out.push("    x *= 0xBF58476D1CE4E5B9ULL;".to_string());
    out.push("    x ^= x >> 27;".to_string());
    out.push("    x *= 0x94D049BB133111EBULL;".to_string());
    out.push("    x ^= x >> 31;".to_string());
    out.push("    double unit = (double)(x >> 11) / (double)(1ULL << 53);".to_string());
    out.push("    return low + (high - low) * (float)unit;".to_string());
    out.push("}".to_string());
    out.push(
        "typedef struct { uint64_t seed; uint64_t counter; int active; } chelis_rng_state;"
            .to_string(),
    );
    out.push("static chelis_rng_state chelis_rng_current = {0ULL, 0ULL, 0};".to_string());
    out.push(
        "static inline uint64_t chelis_effective_uniform_seed(uint64_t baked_seed) {".to_string(),
    );
    out.push("    if (!chelis_rng_current.active) {".to_string());
    out.push("        return baked_seed;".to_string());
    out.push("    }".to_string());
    out.push("    uint64_t counter = chelis_rng_current.counter++;".to_string());
    out.push("    return chelis_rng_current.seed ^ (counter * 0x9E3779B97F4A7C15ULL);".to_string());
    out.push("}".to_string());
    out.push(
        "#define CHELIS_EFFECTIVE_UNIFORM_SEED(seed) chelis_effective_uniform_seed(seed)"
            .to_string(),
    );
}

fn append_tensor_math_helpers(out: &mut Vec<String>) {
    out.push("static inline float chelis_host_relu_f32(float x) {".to_string());
    out.push("    return fmaxf(0.0f, x);".to_string());
    out.push("}".to_string());
    out.push("static inline float chelis_host_sigmoid_f32(float x) {".to_string());
    out.push("    return 1.0f / (1.0f + expf(-x));".to_string());
    out.push("}".to_string());
    // Bucket 3 activation parity: `tanh`, `silu`, `gelu` mirror their
    // IR-evaluator counterparts in
    // `crates/chelis-compiler-api/src/runtime/host_ops.rs`. All math runs
    // through `float` so the two lanes agree byte-for-byte (modulo
    // documented float ulp tolerance).
    out.push("static inline float chelis_host_tanh_f32(float x) {".to_string());
    out.push("    return tanhf(x);".to_string());
    out.push("}".to_string());
    out.push("static inline float chelis_host_silu_f32(float x) {".to_string());
    out.push("    return x * chelis_host_sigmoid_f32(x);".to_string());
    out.push("}".to_string());
    // GELU tanh-approximation, matching `School.Nn.Gelu.gelu_scalar` and
    // `activation_gelu_f32` in chelis-compiler-api/src/runtime/host_ops.rs.
    out.push("static inline float chelis_host_gelu_f32(float x) {".to_string());
    out.push("    float c = 0.7978845608028654f;".to_string());
    out.push("    float k = 0.044715f;".to_string());
    out.push("    float inner = c * (x + k * x * x * x);".to_string());
    out.push("    return 0.5f * x * (1.0f + tanhf(inner));".to_string());
    out.push("}".to_string());
}

fn append_tensor_print_helper(out: &mut Vec<String>) {
    out.push("static void chelis_print_tensor_stdout(const chelis_tensor* t) {".to_string());
    out.push("    printf(\"tensor(shape=[\");".to_string());
    out.push("    for (int64_t d = 0; d < t->ndim; ++d) {".to_string());
    out.push("        if (d > 0) { printf(\", \"); }".to_string());
    out.push("        printf(\"%lld\", (long long)t->shape[d]);".to_string());
    out.push("    }".to_string());
    out.push("    printf(\"], data=[\");".to_string());
    // Limit raised from 10 to 32 (a 4x4 tensor previously rendered only 10
    // of 16 elements with no marker, indistinguishable from a true 10-element
    // tensor; red-team v0.2.6 MEDIUM). Also append "..." when truncated so
    // the trailing-data case is visually unambiguous; downstream parsers
    // must tolerate the `...` token.
    out.push("    int64_t limit = t->size < 32 ? t->size : 32;".to_string());
    out.push("    for (int64_t i = 0; i < limit; ++i) {".to_string());
    // RT-4 F1: read each slot at the correct dtype. The previous code
    // assumed `t->data` was always `float*` and silently read f64/i64
    // tensors as 4-byte slots, producing garbage when the runtime
    // (correctly) sized the buffer at 8 bytes/elem. The display
    // format stays float-style for parity with the evaluator's tensor
    // renderer (every tensor prints as `1.0, 2.0, ...`); int dtypes
    // are widened to double for the format step but stored at the
    // correct width.
    //
    // Mirrors PR #67's runtime-dtype dispatch (host reshape memcpy)
    // and PR #64's typed-cast pattern (DAG emit_cast). See
    // `docs/investigations/cbackend_print_tensor_f64_diagnosis.md`.
    out.push("        double value;".to_string());
    out.push("        switch (t->dtype) {".to_string());
    out.push(
        "            case CHELIS_F64: value = ((const double*)t->data)[i]; break;".to_string(),
    );
    out.push(
        "            case CHELIS_I64: value = (double)((const int64_t*)t->data)[i]; break;"
            .to_string(),
    );
    out.push(
        "            case CHELIS_I32: value = (double)((const int32_t*)t->data)[i]; break;"
            .to_string(),
    );
    out.push(
        "            case CHELIS_I16: value = (double)((const int16_t*)t->data)[i]; break;"
            .to_string(),
    );
    out.push(
        "            case CHELIS_I8:  value = (double)((const int8_t*)t->data)[i]; break;"
            .to_string(),
    );
    out.push("            case CHELIS_BOOL: value = (double)t->data[i]; break;".to_string());
    out.push("            default: value = (double)t->data[i]; break;".to_string());
    out.push("        }".to_string());
    out.push("        if (i > 0) { printf(\", \"); }".to_string());
    out.push("        if (fabs(value - round(value)) < 1e-9) {".to_string());
    out.push("            printf(\"%.1f\", value);".to_string());
    out.push("        } else {".to_string());
    out.push("            printf(\"%.16g\", value);".to_string());
    out.push("        }".to_string());
    out.push("    }".to_string());
    out.push("    if (t->size > limit) { printf(\", ...\"); }".to_string());
    out.push("    printf(\"])\");".to_string());
    out.push("}".to_string());
}

fn append_tensor_reshape_helper(out: &mut Vec<String>) {
    out.push(
        "static chelis_tensor* chelis_host_reshape_tensor(chelis_tensor* input, const chelis_list* shape_values) {"
            .to_string(),
    );
    out.push("    int64_t ndim64 = chelis_list_len(shape_values);".to_string());
    out.push("    if (ndim64 < 0 || ndim64 > CHELIS_MAX_DIM) {".to_string());
    out.push(
        "        fprintf(stderr, \"reshape expects between 0 and %d dims, got %lld\\n\", CHELIS_MAX_DIM, (long long)ndim64);"
            .to_string(),
    );
    out.push("        exit(1);".to_string());
    out.push("    }".to_string());
    out.push("    int ndim = (int)ndim64;".to_string());
    out.push("    int shape[CHELIS_MAX_DIM] = {0};".to_string());
    out.push("    int64_t expected = 1;".to_string());
    out.push("    for (int i = 0; i < ndim; ++i) {".to_string());
    out.push(
        "        int64_t dim = chelis_value_as_int64(chelis_list_index(shape_values, i));"
            .to_string(),
    );
    out.push("        if (dim < 0) {".to_string());
    out.push(
        "            fprintf(stderr, \"reshape expects non-negative sizes, got %lld\\n\", (long long)dim);"
            .to_string(),
    );
    out.push("            exit(1);".to_string());
    out.push("        }".to_string());
    out.push("        shape[i] = (int)dim;".to_string());
    out.push("        expected *= dim;".to_string());
    out.push("    }".to_string());
    out.push("    if (expected != input->size) {".to_string());
    out.push(
        "        fprintf(stderr, \"reshape expects %lld elements but tensor has %d\\n\", (long long)expected, input->size);"
            .to_string(),
    );
    out.push("        exit(1);".to_string());
    out.push("    }".to_string());
    out.push(
        "    chelis_tensor* out_tensor = chelis_alloc(ndim, shape, input->dtype);".to_string(),
    );
    // RT-4 F2: size the memcpy by the actual dtype element width via
    // chelis_dtype_size, not by hardcoded sizeof(float). Mirrors
    // `chelis_alloc`'s element sizing (crates/chelis-runtime/src/lib.rs::
    // tensor_elem_size), so f64/i64 reshape preserves all 8 bytes per
    // element and i8/i16 reshape don't overrun. The dtype-aware path
    // closes both CBackend-ReshapeMemcpy (HEAD; PR #67) and the
    // narrow-int extensions in this cycle. See
    // `docs/investigations/cbackend_reshape_memcpy_diagnosis.md`.
    out.push("    size_t elem_bytes = (size_t)chelis_dtype_size(input->dtype);".to_string());
    out.push(
        "    memcpy(out_tensor->data, input->data, (size_t)input->size * elem_bytes);".to_string(),
    );
    out.push("    return out_tensor;".to_string());
    out.push("}".to_string());
}

pub fn emit_host_header(program: &HostProgram, program_name: &str) -> String {
    emit_host_header_with_linkage(program, program_name, false)
}

fn emit_host_header_with_linkage(
    program: &HostProgram,
    program_name: &str,
    internal_linkage: bool,
) -> String {
    let prefix = if internal_linkage {
        "static inline "
    } else {
        ""
    };
    program
        .functions
        .iter()
        .map(|function| {
            let params = function
                .params
                .iter()
                .map(|param| c_decl(&param.ty, &param.name))
                .collect::<Vec<_>>()
                .join(", ");
            let emitted_name = emitted_function_name(program_name, &function.name);
            format!(
                "{prefix}{} {}({});",
                c_type(&function.ret_ty),
                emitted_name,
                params
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[derive(Debug, Clone, Copy, Default)]
struct HelperRequirements {
    needs_blas_header: bool,
    needs_math_header: bool,
}

impl HelperRequirements {
    fn merge(&mut self, other: Self) {
        self.needs_blas_header |= other.needs_blas_header;
        self.needs_math_header |= other.needs_math_header;
    }
}

/// Append a tensor helper to `out` and return the runtime headers required by
/// the inner emitter. The caller propagates these headers to the host preamble
/// so each one is emitted exactly once at file scope.
fn append_helper(
    out: &mut Vec<String>,
    helper: &HostTensorHelper,
    helper_name: &str,
) -> HelperRequirements {
    if let Some((_input_name, _input_ty)) = identity_helper_input(helper) {
        out.push(format!(
            "static void {}(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out) {{",
            helper_name,
        ));
        out.push("    (void)n_in;".to_string());
        out.push("    (void)n_out;".to_string());
        out.push("    outputs[0] = inputs[0];".to_string());
        out.push("}".to_string());
        out.push(String::new());
        return HelperRequirements::default();
    }

    // Tensor helpers are TU-internal: they are only called from within this
    // generated `.c` file and must never be exported symbols.  `static_entry`
    // ensures the kernel function itself gets `static` linkage so that when
    // compiled with `-shared -fPIC` the symbol is not exported via PLT.
    let specialized = chelis_ir::specialize::specialize_for_blas(&helper.dag);
    let uses_blas = specialized
        .nodes()
        .iter()
        .any(|node| matches!(node.op, RiscOp::BlasMatmul { .. }));
    let helper_src = CEmitter::emit_dag_with_options(
        &specialized,
        helper_name,
        crate::CodegenOptions {
            use_blas: uses_blas,
            static_entry: true,
            ..crate::CodegenOptions::default()
        },
    );
    // The CEmitter prepends a `static inline float chelis_uniform_sample_f32`
    // prelude to every DAG it emits so that a standalone-emitted kernel
    // stays self-contained. When multiple helpers get concatenated into a
    // single `main.c` that duplicates the definition and gcc rejects the
    // redefinition. We filter the prelude out here and rely on
    // `emit_host_program` to emit exactly one copy at file scope.
    let mut skipping_uniform_prelude = false;
    let mut requirements = HelperRequirements::default();
    for line in helper_src.lines() {
        if line.starts_with("#include ") {
            if line.contains("\"chelis_blas.h\"") {
                requirements.needs_blas_header = true;
            }
            if line.contains("\"chelis_math.h\"") {
                requirements.needs_math_header = true;
            }
            continue;
        }
        if line.contains("chelis_uniform_sample_f32(uint64_t seed") {
            skipping_uniform_prelude = true;
            continue;
        }
        if skipping_uniform_prelude {
            if line == "}" {
                skipping_uniform_prelude = false;
            }
            continue;
        }
        if line.is_empty() && out.last().is_some_and(|last| last.is_empty()) {
            continue;
        }
        out.push(line.to_string());
    }
    out.push(String::new());
    requirements
}

fn identity_helper_input(
    helper: &HostTensorHelper,
) -> Option<(String, chelis_ir::dag::TensorType)> {
    if helper.dag.roots().len() != 1 || helper.inputs.len() != 1 {
        return None;
    }
    let root = helper.dag.roots()[0];
    let node = helper.dag.get(root)?;
    match &node.op {
        RiscOp::Load { name } if node.output_type == helper.output => helper
            .inputs
            .iter()
            .find(|input| input.name == *name)
            .map(|input| (input.name.clone(), input.ty.clone())),
        _ => None,
    }
}

fn emit_function(
    out: &mut Vec<String>,
    function: &HostFunction,
    emitted_name: &str,
    emitted_names: &HashMap<String, String>,
    function_specializations: &HashMap<String, HostFunctionSpecialization>,
    internal_linkage: bool,
) {
    let params = function
        .params
        .iter()
        .map(|param| c_decl(&param.ty, &param.name))
        .collect::<Vec<_>>()
        .join(", ");
    let prefix = if internal_linkage {
        "static inline "
    } else {
        ""
    };
    out.push(format!(
        "{prefix}{} {}({}) {{",
        c_type(&function.ret_ty),
        emitted_name,
        params
    ));
    let mut emitter = HostEmitter::new(
        "    ".to_string(),
        emitted_name,
        emitted_names.clone(),
        function_specializations.clone(),
        &function.tensor_helpers,
    );
    emitter.emit_expr_to_var(&function.body, "__result", &function.ret_ty);
    out.extend(emitter.lines);
    out.push("    return __result;".to_string());
    out.push("}".to_string());
}

/// The base C indent inside the generated `main` body. Top-level `main`
/// locals are emitted at this indent; `track_owned_alloc` releases only
/// allocations at this depth (issue #406) so block-scoped temporaries in
/// nested loop/conditional bodies are not freed out of scope.
const BASE_MAIN_INDENT: &str = "    ";

fn emit_main(
    out: &mut Vec<String>,
    program_name: &str,
    program: &HostProgram,
    hoisted: &HashSet<&str>,
) {
    out.push("int main(void) {".to_string());
    let mut emitter = HostEmitter::new(
        BASE_MAIN_INDENT.to_string(),
        &format!("{program_name}__global"),
        HashMap::new(),
        function_specializations(program),
        &program.global_tensor_helpers,
    );
    // issue #406: `main` is the program root — it owns every heap value
    // it creates (globals plus the list/tuple/dict temporaries built to
    // construct them) and returns none, so enable scope-release tracking
    // and free each owned allocation before `return 0`. Without this the
    // generated binary leaks them for the whole process lifetime, which
    // a `valgrind --leak-check=full --error-exitcode=1` gate flags as
    // "definitely lost".
    emitter.scope_releases = Some(Vec::new());
    for (index, binding) in program.globals.iter().enumerate() {
        let binding_var = format!("__binding_{index}_value");
        emitter.emit_expr_to_var(&binding.value, &binding_var, &binding.ty);
        // The binding-value local owns its allocation regardless of how
        // it was produced (tensor kernel output, list/dict builtin,
        // literal). Track it here; the alias name (`theta`) is never
        // tracked, and dedup in `emit_scope_releases` collapses the case
        // where the binding value *is* a literal already tracked above.
        emitter.track_owned_alloc(&binding_var, &binding.ty);
        if hoisted.contains(binding.name.as_str()) {
            // Declared at file scope (issue #352); assign, don't shadow.
            emitter
                .lines
                .push(format!("    {} = __binding_{index}_value;", binding.name));
        } else {
            emitter.lines.push(format!(
                "    {} {} = __binding_{index}_value;",
                c_type(&binding.ty),
                binding.name
            ));
        }
    }
    for binding in &program.globals {
        if let Some(display_name) = binding.display_name.as_deref() {
            emitter.emit_labeled_root(display_name, binding.name.as_str(), &binding.ty);
        }
    }
    // issue #406: free everything `main` owns before returning. Emitted
    // after the labeled-root prints so the values are still live when
    // printed and reclaimed immediately after.
    emitter.emit_scope_releases();
    out.extend(emitter.lines);
    out.push("    return 0;".to_string());
    out.push("}".to_string());
}

/// Top-level bindings referenced by name inside at least one compiled host
/// function body (issue #352), in `program.globals` order, deduped.
///
/// Deliberately an over-approximation: the walk records every `Var` name
/// without subtracting binders (params, let names, match bindings).
/// Hoisting a binding that is shadowed inside a function body is harmless
/// in C -- the local declaration shadows the file-scope static -- while
/// missing a genuine capture reproduces the undeclared-identifier build
/// break this pass exists to prevent.
fn captured_global_names(program: &HostProgram) -> Vec<String> {
    let mut referenced: HashSet<String> = HashSet::new();
    for function in &program.functions {
        collect_var_names(&function.body, &mut referenced);
    }
    let mut seen: HashSet<&str> = HashSet::new();
    program
        .globals
        .iter()
        .filter(|binding| referenced.contains(&binding.name))
        .filter(|binding| seen.insert(binding.name.as_str()))
        .map(|binding| binding.name.clone())
        .collect()
}

/// Record every `Var` name referenced anywhere in `expr`, including
/// let-binding values, match arms, and inline-callback bodies. Exhaustive
/// over `HostExprKind` so a new variant forces this walk to be revisited.
fn collect_var_names(expr: &HostExpr, out: &mut HashSet<String>) {
    match &expr.kind {
        HostExprKind::Int(_)
        | HostExprKind::Float(_)
        | HostExprKind::Bool(_)
        | HostExprKind::String(_)
        | HostExprKind::Unit => {}
        HostExprKind::Var(name, _) => {
            out.insert(name.clone());
        }
        HostExprKind::List(items, _) | HostExprKind::Tuple(items, _) => {
            for item in items {
                collect_var_names(item, out);
            }
        }
        HostExprKind::Call { args, .. }
        | HostExprKind::Builtin { args, .. }
        | HostExprKind::TensorCall { args, .. } => {
            for arg in args {
                collect_var_names(arg, out);
            }
        }
        HostExprKind::AdtConstruct { fields, .. } => {
            for field in fields {
                collect_var_names(field, out);
            }
        }
        HostExprKind::AdtFieldAccess { base, .. } => collect_var_names(base, out),
        HostExprKind::If {
            cond,
            then_expr,
            else_expr,
            ..
        } => {
            collect_var_names(cond, out);
            collect_var_names(then_expr, out);
            collect_var_names(else_expr, out);
        }
        HostExprKind::MatchOption {
            scrutinee,
            some_expr,
            none_expr,
            ..
        } => {
            collect_var_names(scrutinee, out);
            collect_var_names(some_expr, out);
            collect_var_names(none_expr, out);
        }
        HostExprKind::MatchAdt {
            scrutinee,
            arms,
            default_expr,
            ..
        } => {
            collect_var_names(scrutinee, out);
            for arm in arms {
                collect_var_names(&arm.expr, out);
            }
            if let Some(default_expr) = default_expr {
                collect_var_names(default_expr, out);
            }
        }
        HostExprKind::Let { bindings, body, .. } => {
            for binding in bindings {
                collect_var_names(&binding.value, out);
            }
            collect_var_names(body, out);
        }
        HostExprKind::Map { callback, list, .. }
        | HostExprKind::Filter { callback, list, .. }
        | HostExprKind::Partition { callback, list, .. }
        | HostExprKind::FlatMap { callback, list, .. } => {
            collect_callback_var_names(callback, out);
            collect_var_names(list, out);
        }
        HostExprKind::Fold {
            callback,
            init,
            list,
            ..
        }
        | HostExprKind::Scan {
            callback,
            init,
            list,
            ..
        } => {
            collect_callback_var_names(callback, out);
            collect_var_names(init, out);
            collect_var_names(list, out);
        }
        HostExprKind::WithSeed { seed, body, .. } => {
            collect_var_names(seed, out);
            collect_var_names(body, out);
        }
    }
}

fn collect_callback_var_names(callback: &HostCallback, out: &mut HashSet<String>) {
    match &callback.kind {
        HostCallbackKind::Named { .. } => {}
        HostCallbackKind::Inline { body, .. } => collect_var_names(body, out),
    }
}

struct HostEmitter<'a> {
    lines: Vec<String>,
    indent: String,
    helper_prefix: String,
    emitted_names: HashMap<String, String>,
    function_specializations: HashMap<String, HostFunctionSpecialization>,
    tensor_helpers: &'a [HostTensorHelper],
    temp_counter: usize,
    /// When `Some`, every heap-owning allocation created in this emit
    /// scope is recorded as `(var, type)` so the scope can release it
    /// before returning. Set only for `emit_main` (issue #406): the
    /// generated `main` is the program root, owns every heap value it
    /// creates, and transfers none out, so each owned allocation must be
    /// freed at scope exit or it leaks for the process lifetime. `None`
    /// inside compiled functions, which already emit their own
    /// per-local `chelis_free` cleanup.
    scope_releases: Option<Vec<(String, HostType)>>,
    /// Stack of open release-tracking `let` blocks (issue #406, the
    /// function-body sibling of the `emit_main` leak). One entry per block;
    /// each records the slots the block transfers an owned reference into
    /// (`owned_destinations`: its result target plus each binding's value
    /// temp) and the heap bindings it frees at its close (`bindings`).
    ///
    /// A bare pointer-copy that aliases one of a block's `bindings` into an
    /// `owned_destinations` slot (emitted directly for a `Var` body or
    /// inside an `if`/`match` arm) is retained: that destination owns an
    /// independent reference and the retain cancels the eventual release so
    /// the caller keeps one reference. A transient read of a binding into an
    /// internal arg temp (e.g. `__arg0 = p` feeding `chelis_tuple_get`) is
    /// not an owned destination, so it is left untouched. Empty outside a
    /// tracked block (e.g. `emit_main`, whose alias handling is the distinct
    /// global-binding path above).
    let_scopes: Vec<LetReleaseScope>,
}

/// One open release-tracking `let` block; see `HostEmitter::let_scopes`.
struct LetReleaseScope {
    /// C variables this block transfers an owned reference into: the block's
    /// result target (the enclosing-scope variable it writes its result to)
    /// plus each binding's `__let_N` value temp. A bare pointer-copy that
    /// aliases one of this block's `bindings` into one of these destinations
    /// must be retained so the destination owns an independent reference —
    /// that destination is itself released later (a binding at this block's
    /// close, or by whatever owns the result target). A copy into any other
    /// temp (a transient arg fed to `chelis_tuple_get`, say) is a borrow and
    /// is not retained.
    owned_destinations: HashSet<String>,
    /// Heap binding names this block releases at its close.
    bindings: HashSet<String>,
}

impl<'a> HostEmitter<'a> {
    fn new(
        indent: String,
        helper_prefix: &str,
        emitted_names: HashMap<String, String>,
        function_specializations: HashMap<String, HostFunctionSpecialization>,
        tensor_helpers: &'a [HostTensorHelper],
    ) -> Self {
        Self {
            lines: Vec::new(),
            indent,
            helper_prefix: helper_prefix.to_string(),
            emitted_names,
            function_specializations,
            tensor_helpers,
            temp_counter: 0,
            scope_releases: None,
            let_scopes: Vec::new(),
        }
    }

    /// Record `var` (of `ty`) as a heap-owning allocation this scope must
    /// release before returning. No-op unless scope-release tracking is
    /// enabled (i.e. this is the `main` emitter, issue #406). Only the
    /// pointer-typed, heap-owning `HostType`s are tracked; scalars and
    /// borrowed views carry no ownership.
    fn track_owned_alloc(&mut self, var: &str, ty: &HostType) {
        // Only track allocations declared at the scope's own (base) indent
        // level. Temporaries created inside a nested C block -- a
        // `map` / `flat_map` / `filter` / `fold` loop body, or an `if` /
        // `match` arm -- are emitted at a deeper indent and are block-
        // scoped, so they are not visible at the function-level cleanup
        // and must not be released there (that would emit C referencing
        // an out-of-scope identifier). Those temporaries are already
        // freed where they are consumed by the surrounding helper.
        if self.indent.len() != BASE_MAIN_INDENT.len() {
            return;
        }
        if let Some(releases) = self.scope_releases.as_mut()
            && release_call(var, ty).is_some()
        {
            releases.push((var.to_string(), ty.clone()));
        }
    }

    /// Drain the recorded scope-owned allocations, emitting one release
    /// call per distinct variable (deduped: an alias such as `theta =
    /// __binding_0_value` is never recorded, only the underlying
    /// `__binding_N_value`, so each heap pointer is freed exactly once).
    /// Released in reverse creation order so a container is freed after
    /// any later-created value, mirroring C scope-exit destruction order.
    fn emit_scope_releases(&mut self) {
        let Some(releases) = self.scope_releases.take() else {
            return;
        };
        let mut seen: HashSet<String> = HashSet::new();
        for (var, ty) in releases.into_iter().rev() {
            if !seen.insert(var.clone()) {
                continue;
            }
            if let Some(call) = release_call(&var, &ty) {
                self.lines.push(format!("{}{call}", self.indent));
            }
        }
    }

    /// Retain `target` when a bare pointer-copy `target = source` moves a
    /// heap `let` binding into a slot that owns an independent reference
    /// (issue #406). Fires only when `source` is a binding some open block
    /// frees at its close *and* `target` is one of that block's
    /// `owned_destinations` (its result target or another binding's value
    /// temp). The retain cancels the eventual release of the destination so
    /// every owned slot — the escaping result, and any binding that aliases
    /// an earlier one — carries exactly one reference.
    ///
    /// A transient read of a binding into an internal arg temp (e.g.
    /// `__arg0 = p` feeding `chelis_tuple_get`) is not an owned destination,
    /// so it is left alone: `chelis_tuple_get` does its own element retain
    /// and the binding's single release still balances its construction. A
    /// transfer of a parameter or outer-scope value is likewise untouched —
    /// no open block frees it, so a retain would leak.
    fn retain_transferred_result(&mut self, target: &str, source: &str, ty: &HostType) {
        // `source` may be a binding of an outer block while `target` is an
        // owned slot of an inner one (a nested `let b = a in ...`), so test
        // the two conditions independently across all open scopes rather
        // than within a single scope.
        let target_is_owned = self
            .let_scopes
            .iter()
            .any(|scope| scope.owned_destinations.contains(target));
        let source_is_binding = self
            .let_scopes
            .iter()
            .any(|scope| scope.bindings.contains(source));
        if !(target_is_owned && source_is_binding) {
            return;
        }
        if let Some(call) = retain_call(target, ty) {
            self.lines.push(format!("{}{call}", self.indent));
        }
    }

    fn emit_expr_to_var(&mut self, expr: &HostExpr, target: &str, ty: &HostType) {
        self.lines
            .push(format!("{}{};", self.indent, c_decl(ty, target)));
        self.assign_expr(target, expr, ty);
    }

    /// Emit `// span:` comment lines for a `HostExpr`'s `span_id ∪ merged_spans`,
    /// per `spec/design/chelis_span_survival.md` §2.3 host-side table (host
    /// emit row) and §2.4 (host-path emission rule).
    ///
    /// Order: canonical `span_id` first (if present), then `merged_spans`
    /// lex-sorted and deduped against `span_id`. Same shape as the DAG-side
    /// helper at `chelis_backend_c::emit::CEmitter::emit_span_comments`
    /// (S4.1). Span IDs are sanitized via
    /// `chelis_ir::span_sanitize::sanitize_for_comment` before
    /// interpolation so a forbidden control byte cannot break out of the
    /// `// ` line comment, mirroring the DAG-side path.
    ///
    /// No-op when both fields are empty (the common case for hand-written
    /// Chelis or for span-free Deep input). This locks the
    /// backward-compatibility invariant: span-free programs emit zero
    /// `// span:` comments on the host path.
    fn emit_span_comments(&mut self, expr: &HostExpr) {
        if expr.span_id.is_none() && expr.merged_spans.is_empty() {
            return;
        }
        if let Some(canonical) = expr.span_id.as_deref() {
            let safe = chelis_ir::span_sanitize::sanitize_for_comment(canonical);
            self.lines.push(format!("{}// span: {safe}", self.indent));
        }
        let mut merged: Vec<&str> = expr
            .merged_spans
            .iter()
            .map(String::as_str)
            .filter(|s| expr.span_id.as_deref() != Some(*s))
            .collect();
        merged.sort();
        merged.dedup();
        for span in merged {
            let safe = chelis_ir::span_sanitize::sanitize_for_comment(span);
            self.lines.push(format!("{}// span: {safe}", self.indent));
        }
    }

    fn assign_expr(&mut self, target: &str, expr: &HostExpr, ty: &HostType) {
        self.emit_span_comments(expr);
        match &expr.kind {
            HostExprKind::Int(value) => self
                .lines
                .push(format!("{}{target} = {};", self.indent, value)),
            HostExprKind::Float(value) => self
                .lines
                .push(format!("{}{target} = {};", self.indent, value)),
            HostExprKind::Bool(value) => self.lines.push(format!(
                "{}{target} = {};",
                self.indent,
                if *value { "true" } else { "false" }
            )),
            HostExprKind::String(value) => self.lines.push(format!(
                "{}{target} = chelis_string_from_cstr({:?});",
                self.indent, value
            )),
            HostExprKind::List(items, expr_ty) => {
                let effective_ty = if !matches!(ty, HostType::Unknown) {
                    ty
                } else {
                    expr_ty
                };
                self.assign_list_literal(target, items, effective_ty);
            }
            HostExprKind::Tuple(items, expr_ty) => {
                let effective_ty = if !matches!(ty, HostType::Unknown) {
                    ty
                } else {
                    expr_ty
                };
                self.assign_tuple_literal(target, items, effective_ty);
            }
            HostExprKind::Var(name, var_ty) => {
                if name == "Nil" {
                    self.lines
                        .push(format!("{}{target} = chelis_list_empty();", self.indent));
                } else if name == "None" && matches!(ty, HostType::Option(_)) {
                    self.assign_option_none(target, ty);
                } else if name == "Nil" && matches!(var_ty, HostType::List(_)) {
                    self.lines
                        .push(format!("{}{target} = chelis_list_empty();", self.indent));
                } else {
                    // `target = name` is a bare pointer copy that does not
                    // bump the refcount. When `name` is a heap `let` binding
                    // freed at its block close (issue #406), retain the
                    // transferred result to keep the caller's reference
                    // alive; a parameter or outer-scope `name` is left
                    // untouched (the block does not free it).
                    self.lines
                        .push(format!("{}{target} = {name};", self.indent));
                    self.retain_transferred_result(target, name, ty);
                }
            }
            HostExprKind::Call {
                function,
                args,
                arg_tys,
                ty: call_ty,
            } => {
                self.assign_call(target, function, args, arg_tys, call_ty);
            }
            HostExprKind::Builtin {
                name,
                args,
                ty: expr_ty,
            } => {
                let effective_ty = if !matches!(ty, HostType::Unknown) {
                    ty
                } else {
                    expr_ty
                };
                self.assign_builtin(target, name, args, effective_ty);
            }
            HostExprKind::AdtConstruct {
                ctor,
                fields,
                ty: expr_ty,
            } => {
                let effective_ty = if !matches!(ty, HostType::Unknown) {
                    ty
                } else {
                    expr_ty
                };
                self.assign_adt_construct(target, ctor, fields, effective_ty);
            }
            HostExprKind::AdtFieldAccess {
                base,
                field_index,
                ty: expr_ty,
            } => {
                let effective_ty = if !matches!(ty, HostType::Unknown) {
                    ty
                } else {
                    expr_ty
                };
                self.assign_adt_field_access(target, base, *field_index, effective_ty);
            }
            HostExprKind::If {
                cond,
                then_expr,
                else_expr,
                ty: expr_ty,
            } => {
                let effective_ty = if !matches!(ty, HostType::Unknown) {
                    ty
                } else {
                    expr_ty
                };
                let cond_var = self.next_temp("cond");
                self.emit_expr_to_var(cond, &cond_var, &HostType::Bool);
                self.lines
                    .push(format!("{}if ({cond_var}) {{", self.indent));
                let nested_indent = format!("{}    ", self.indent);
                let previous = std::mem::replace(&mut self.indent, nested_indent);
                self.assign_expr(target, then_expr, effective_ty);
                self.indent = previous.clone();
                self.lines.push(format!("{}}} else {{", self.indent));
                let nested_indent = format!("{}    ", self.indent);
                let previous = std::mem::replace(&mut self.indent, nested_indent);
                self.assign_expr(target, else_expr, effective_ty);
                self.indent = previous;
                self.lines.push(format!("{}}}", self.indent));
            }
            HostExprKind::MatchOption {
                scrutinee,
                bind_name,
                some_expr,
                none_expr,
                ty: expr_ty,
            } => {
                let effective_ty = if !matches!(ty, HostType::Unknown) {
                    ty
                } else {
                    expr_ty
                };
                let option_var = self.next_temp("option");
                let option_ty = host_type(scrutinee);
                self.emit_expr_to_var(scrutinee, &option_var, &option_ty);
                self.lines
                    .push(format!("{}if ({}.is_some) {{", self.indent, option_var));
                let nested_indent = format!("{}    ", self.indent);
                let previous = std::mem::replace(&mut self.indent, nested_indent);
                let inner_ty = option_inner_type(option_ty.clone());
                match option_ty {
                    HostType::Option(inner)
                        if !matches!(inner.as_ref(), HostType::Int64 | HostType::Float64) =>
                    {
                        self.lines.push(format!(
                            "{}{} {};",
                            self.indent,
                            c_type(&inner_ty),
                            bind_name
                        ));
                        self.assign_unboxed_value(
                            bind_name,
                            &inner_ty,
                            &format!("{option_var}.value"),
                        );
                    }
                    _ => {
                        self.lines.push(format!(
                            "{}{} {} = {}.value;",
                            self.indent,
                            c_type(&inner_ty),
                            bind_name,
                            option_var
                        ));
                    }
                }
                self.assign_expr(target, some_expr, effective_ty);
                self.indent = previous.clone();
                self.lines.push(format!("{}}} else {{", self.indent));
                let nested_indent = format!("{}    ", self.indent);
                let previous = std::mem::replace(&mut self.indent, nested_indent);
                self.assign_expr(target, none_expr, effective_ty);
                self.indent = previous;
                self.lines.push(format!("{}}}", self.indent));
            }
            HostExprKind::MatchAdt {
                scrutinee,
                arms,
                default_expr,
                ty: expr_ty,
            } => {
                let effective_ty = if !matches!(ty, HostType::Unknown) {
                    ty
                } else {
                    expr_ty
                };
                self.assign_match_adt(
                    target,
                    scrutinee,
                    arms,
                    default_expr.as_deref(),
                    effective_ty,
                );
            }
            HostExprKind::Let {
                bindings,
                body,
                ty: expr_ty,
            } => {
                let effective_ty = if !matches!(ty, HostType::Unknown) {
                    ty
                } else {
                    expr_ty
                };
                self.lines.push(format!("{}{{", self.indent));
                let nested_indent = format!("{}    ", self.indent);
                let previous = std::mem::replace(&mut self.indent, nested_indent);
                // issue #406: this `let` introduces a nested C scope that
                // owns every heap binding it declares. The block body's
                // result is written to `target` in the *outer* scope, so
                // each heap binding must be released at block close or it
                // leaks (the function-body sibling of the `emit_main`
                // leak). Track this block's owned destinations + heap
                // bindings; a bare pointer-copy of a binding into an owned
                // destination (the escaping result, or another binding's
                // value temp that aliases an earlier binding) is retained so
                // each owned slot keeps exactly one reference.
                self.let_scopes.push(LetReleaseScope {
                    owned_destinations: HashSet::from([target.to_string()]),
                    bindings: HashSet::new(),
                });
                let mut heap_bindings: Vec<(String, HostType)> = Vec::new();
                for binding in bindings {
                    // Compute the value into a temp before declaring the binding name.
                    // If the compiler inlines a recursive call that reuses a binding
                    // name from the outer scope (e.g. two nested `let jtj_new = ...`),
                    // declaring the inner name first would shadow the outer variable
                    // before its value is read, yielding a NULL pointer at runtime.
                    let temp = self.next_temp("let");
                    // The value temp is an owned slot of this block: if the
                    // binding's value is a bare alias of an earlier binding
                    // (`b = a`), the copy into the temp must retain so `b`
                    // owns an independent reference and the two distinct
                    // block releases do not double-free the shared
                    // allocation. Register it before emitting the value.
                    if binding_release(&temp, &binding.ty).is_some()
                        && let Some(scope) = self.let_scopes.last_mut()
                    {
                        scope.owned_destinations.insert(temp.clone());
                    }
                    self.emit_expr_to_var(&binding.value, &temp, &binding.ty);
                    self.lines.push(format!(
                        "{}{};",
                        self.indent,
                        c_decl(&binding.ty, &binding.name)
                    ));
                    self.lines
                        .push(format!("{}{} = {};", self.indent, binding.name, temp));
                    // Track the binding name (not its `__let_N` temp: the
                    // two alias the same allocation, so releasing only the
                    // name frees it exactly once). Add it to the scope's
                    // binding set after its value is computed so a binding
                    // whose value reads an *earlier* binding still retains
                    // on that transfer.
                    if binding_release(&binding.name, &binding.ty).is_some() {
                        heap_bindings.push((binding.name.clone(), binding.ty.clone()));
                        if let Some(scope) = self.let_scopes.last_mut() {
                            scope.bindings.insert(binding.name.clone());
                        }
                    }
                }
                self.assign_expr(target, body, effective_ty);
                // Release the block's heap bindings in reverse declaration
                // order, before closing the C block while they are still in
                // scope. Last-declared shadows of a reused name win the C
                // lookup, mirroring C scope-exit destruction order.
                for (name, binding_ty) in heap_bindings.iter().rev() {
                    if let Some(call) = binding_release(name, binding_ty) {
                        self.lines.push(format!("{}{call}", self.indent));
                    }
                }
                self.let_scopes.pop();
                self.indent = previous;
                self.lines.push(format!("{}}}", self.indent));
            }
            HostExprKind::Map { callback, list, ty } => {
                self.assign_map(target, callback, list, ty);
            }
            HostExprKind::Filter { callback, list, ty } => {
                self.assign_filter(target, callback, list, ty);
            }
            HostExprKind::Fold {
                callback,
                init,
                list,
                ty,
            } => {
                self.assign_fold(target, callback, init, list, ty);
            }
            HostExprKind::Scan {
                callback,
                init,
                list,
                ty,
            } => {
                self.assign_scan(target, callback, init, list, ty);
            }
            HostExprKind::Partition { callback, list, ty } => {
                self.assign_partition(target, callback, list, ty);
            }
            HostExprKind::FlatMap { callback, list, ty } => {
                self.assign_flat_map(target, callback, list, ty);
            }
            HostExprKind::WithSeed { seed, body, ty } => {
                let seed_var = self.next_temp("seed");
                self.emit_expr_to_var(seed, &seed_var, &HostType::Int64);
                let saved_var = self.next_temp("rng_saved");
                self.lines.push(format!(
                    "{}chelis_rng_state {saved_var} = chelis_rng_current;",
                    self.indent
                ));
                self.lines.push(format!(
                    "{}chelis_rng_current.seed = (uint64_t){seed_var};",
                    self.indent
                ));
                self.lines
                    .push(format!("{}chelis_rng_current.counter = 0ULL;", self.indent));
                self.lines
                    .push(format!("{}chelis_rng_current.active = 1;", self.indent));
                self.assign_expr(target, body, ty);
                self.lines
                    .push(format!("{}chelis_rng_current = {saved_var};", self.indent));
            }
            HostExprKind::TensorCall { helper, args, ty } => {
                self.assign_tensor_call(target, *helper, args, ty);
            }
            HostExprKind::Unit => {
                self.lines.push(format!("{}{target} = 0;", self.indent));
            }
        }
    }

    fn assign_builtin(&mut self, target: &str, name: &str, args: &[HostExpr], ty: &HostType) {
        let arg_vars = args
            .iter()
            .enumerate()
            .map(|(index, arg)| {
                let arg_name = self.next_temp(&format!("arg{index}"));
                let inferred_ty = host_type(arg);
                let expected_ty = expected_builtin_arg_ty(name, ty, index);
                let arg_ty =
                    if has_unknown(&inferred_ty) && !matches!(expected_ty, HostType::Unknown) {
                        expected_ty
                    } else {
                        inferred_ty
                    };
                self.emit_expr_to_var(arg, &arg_name, &arg_ty);
                (arg_name, arg_ty)
            })
            .collect::<Vec<_>>();

        if let HostType::Tensor(_) = ty {
            match name {
                "add"
                    if matches!(
                        (&arg_vars[0].1, &arg_vars[1].1),
                        (HostType::Tensor(_), HostType::Tensor(_))
                    ) =>
                {
                    self.assign_tensor_binary_elementwise(
                        target,
                        &arg_vars[0].0,
                        &arg_vars[1].0,
                        "+",
                    );
                    return;
                }
                "sub"
                    if matches!(
                        (&arg_vars[0].1, &arg_vars[1].1),
                        (HostType::Tensor(_), HostType::Tensor(_))
                    ) =>
                {
                    self.assign_tensor_binary_elementwise(
                        target,
                        &arg_vars[0].0,
                        &arg_vars[1].0,
                        "-",
                    );
                    return;
                }
                "mul"
                    if matches!(
                        (&arg_vars[0].1, &arg_vars[1].1),
                        (HostType::Tensor(_), HostType::Tensor(_))
                    ) =>
                {
                    self.assign_tensor_binary_elementwise(
                        target,
                        &arg_vars[0].0,
                        &arg_vars[1].0,
                        "*",
                    );
                    return;
                }
                "div"
                    if matches!(
                        (&arg_vars[0].1, &arg_vars[1].1),
                        (HostType::Tensor(_), HostType::Tensor(_))
                    ) =>
                {
                    self.assign_tensor_binary_elementwise(
                        target,
                        &arg_vars[0].0,
                        &arg_vars[1].0,
                        "/",
                    );
                    return;
                }
                "max_elem"
                    if matches!(
                        (&arg_vars[0].1, &arg_vars[1].1),
                        (HostType::Tensor(_), HostType::Tensor(_))
                    ) =>
                {
                    self.assign_tensor_binary_func_elementwise(
                        target,
                        &arg_vars[0].0,
                        &arg_vars[1].0,
                        "fmaxf",
                    );
                    return;
                }
                "min_elem"
                    if matches!(
                        (&arg_vars[0].1, &arg_vars[1].1),
                        (HostType::Tensor(_), HostType::Tensor(_))
                    ) =>
                {
                    self.assign_tensor_binary_func_elementwise(
                        target,
                        &arg_vars[0].0,
                        &arg_vars[1].0,
                        "fminf",
                    );
                    return;
                }
                "neg" if matches!(&arg_vars[0].1, HostType::Tensor(_)) => {
                    self.assign_tensor_unary_elementwise(target, &arg_vars[0].0, "-");
                    return;
                }
                "not" if matches!(&arg_vars[0].1, HostType::Tensor(_)) => {
                    self.assign_tensor_unary_elementwise(target, &arg_vars[0].0, "!");
                    return;
                }
                "exp" if matches!(&arg_vars[0].1, HostType::Tensor(_)) => {
                    self.assign_tensor_unary_func_elementwise(target, &arg_vars[0].0, "expf");
                    return;
                }
                "log" if matches!(&arg_vars[0].1, HostType::Tensor(_)) => {
                    self.assign_tensor_unary_func_elementwise(target, &arg_vars[0].0, "logf");
                    return;
                }
                "sin" if matches!(&arg_vars[0].1, HostType::Tensor(_)) => {
                    self.assign_tensor_unary_func_elementwise(target, &arg_vars[0].0, "sinf");
                    return;
                }
                "sqrt" if matches!(&arg_vars[0].1, HostType::Tensor(_)) => {
                    self.assign_tensor_unary_func_elementwise(target, &arg_vars[0].0, "sqrtf");
                    return;
                }
                "relu" if matches!(&arg_vars[0].1, HostType::Tensor(_)) => {
                    self.assign_tensor_unary_func_elementwise(
                        target,
                        &arg_vars[0].0,
                        "chelis_host_relu_f32",
                    );
                    return;
                }
                "sigmoid" if matches!(&arg_vars[0].1, HostType::Tensor(_)) => {
                    self.assign_tensor_unary_func_elementwise(
                        target,
                        &arg_vars[0].0,
                        "chelis_host_sigmoid_f32",
                    );
                    return;
                }
                "tanh" if matches!(&arg_vars[0].1, HostType::Tensor(_)) => {
                    self.assign_tensor_unary_func_elementwise(
                        target,
                        &arg_vars[0].0,
                        "chelis_host_tanh_f32",
                    );
                    return;
                }
                "silu" if matches!(&arg_vars[0].1, HostType::Tensor(_)) => {
                    self.assign_tensor_unary_func_elementwise(
                        target,
                        &arg_vars[0].0,
                        "chelis_host_silu_f32",
                    );
                    return;
                }
                "gelu" if matches!(&arg_vars[0].1, HostType::Tensor(_)) => {
                    self.assign_tensor_unary_func_elementwise(
                        target,
                        &arg_vars[0].0,
                        "chelis_host_gelu_f32",
                    );
                    return;
                }
                _ => {}
            }
        }

        match name {
            "Some" => {
                self.assign_option_some(target, ty, &arg_vars[0].0, &arg_vars[0].1);
                return;
            }
            "None" => {
                self.assign_option_none(target, ty);
                return;
            }
            "cast" => {
                let expr = match (&arg_vars[0].1, ty) {
                    (HostType::Int64, HostType::Float64) => {
                        format!("(double){0}", arg_vars[0].0)
                    }
                    (HostType::Float64, HostType::Int64) => {
                        format!("(int64_t){0}", arg_vars[0].0)
                    }
                    (HostType::Bool, HostType::Int64) => {
                        format!("(int64_t){0}", arg_vars[0].0)
                    }
                    (HostType::Int64, HostType::Bool) => {
                        format!("((bool){})", arg_vars[0].0)
                    }
                    _ => arg_vars[0].0.clone(),
                };
                self.lines
                    .push(format!("{}{target} = {};", self.indent, expr));
                return;
            }
            "copy" => {
                self.lines
                    .push(format!("{}{target} = {};", self.indent, arg_vars[0].0));
                return;
            }
            "tuple-get" => {
                let value_var = self.next_temp("tuple_value");
                self.lines.push(format!(
                    "{}chelis_value {} = chelis_tuple_get({}, {});",
                    self.indent, value_var, arg_vars[0].0, arg_vars[1].0
                ));
                self.assign_unboxed_value(target, ty, &value_var);
                return;
            }
            "index" => {
                let value_var = self.next_temp("list_value");
                self.lines.push(format!(
                    "{}chelis_value {} = chelis_list_index({}, {});",
                    self.indent, value_var, arg_vars[0].0, arg_vars[1].0
                ));
                self.assign_unboxed_value(target, ty, &value_var);
                return;
            }
            "append" => {
                self.lines.push(format!(
                    "{}{target} = chelis_list_append({}, {});",
                    self.indent,
                    arg_vars[0].0,
                    self.box_value_expr(&arg_vars[1].0, &arg_vars[1].1)
                ));
                return;
            }
            "concat" => {
                if matches!(ty, HostType::Tensor(_)) {
                    self.lines.push(format!(
                        "{}{target} = chelis_tensor_concat({}, {});",
                        self.indent, arg_vars[0].0, arg_vars[1].0
                    ));
                } else {
                    self.lines.push(format!(
                        "{}{target} = chelis_list_concat({}, {});",
                        self.indent, arg_vars[0].0, arg_vars[1].0
                    ));
                }
                return;
            }
            "split" => {
                self.lines.push(format!(
                    "{}{target} = chelis_tensor_split({}, {}, {});",
                    self.indent, arg_vars[0].0, arg_vars[1].0, arg_vars[2].0
                ));
                return;
            }
            "gather" => {
                self.lines.push(format!(
                    "{}{target} = chelis_tensor_gather({}, {}, {});",
                    self.indent, arg_vars[0].0, arg_vars[1].0, arg_vars[2].0
                ));
                return;
            }
            "scatter" => {
                self.lines.push(format!(
                    "{}{target} = chelis_tensor_scatter({}, {}, {}, {}, {});",
                    self.indent,
                    arg_vars[0].0,
                    arg_vars[1].0,
                    arg_vars[2].0,
                    arg_vars[3].0,
                    arg_vars[4].0
                ));
                return;
            }
            "where" => {
                self.lines.push(format!(
                    "{}{target} = chelis_tensor_where({}, {}, {});",
                    self.indent, arg_vars[0].0, arg_vars[1].0, arg_vars[2].0
                ));
                return;
            }
            "cumsum" => {
                self.lines.push(format!(
                    "{}{target} = chelis_tensor_cumsum({}, {});",
                    self.indent, arg_vars[0].0, arg_vars[1].0
                ));
                return;
            }
            "sort" => {
                self.lines.push(format!(
                    "{}{target} = chelis_tensor_sort({}, {});",
                    self.indent, arg_vars[0].0, arg_vars[1].0
                ));
                return;
            }
            "diagonal" => {
                self.lines.push(format!(
                    "{}{target} = chelis_tensor_diagonal({}, {}, {});",
                    self.indent, arg_vars[0].0, arg_vars[1].0, arg_vars[2].0
                ));
                return;
            }
            "trace" => {
                self.lines.push(format!(
                    "{}{target} = chelis_tensor_trace({}, {}, {});",
                    self.indent, arg_vars[0].0, arg_vars[1].0, arg_vars[2].0
                ));
                return;
            }
            "clamp" => {
                self.lines.push(format!(
                    "{}{target} = chelis_tensor_clamp({}, {}, {});",
                    self.indent, arg_vars[0].0, arg_vars[1].0, arg_vars[2].0
                ));
                return;
            }
            "einsum" => {
                self.lines.push(format!(
                    "{}{target} = chelis_tensor_einsum({}, {}, {});",
                    self.indent, arg_vars[0].0, arg_vars[1].0, arg_vars[2].0
                ));
                return;
            }
            "take" => {
                self.lines.push(format!(
                    "{}{target} = chelis_list_take({}, {});",
                    self.indent, arg_vars[0].0, arg_vars[1].0
                ));
                return;
            }
            "drop" => {
                if arg_vars.len() == 1 {
                    self.lines.push(format!("{}{target} = 0;", self.indent));
                } else {
                    self.lines.push(format!(
                        "{}{target} = chelis_list_drop({}, {});",
                        self.indent, arg_vars[0].0, arg_vars[1].0
                    ));
                }
                return;
            }
            "chunk" => {
                self.lines.push(format!(
                    "{}{target} = chelis_list_chunk({}, {});",
                    self.indent, arg_vars[0].0, arg_vars[1].0
                ));
                return;
            }
            "flatten" => {
                self.lines.push(format!(
                    "{}{target} = chelis_list_flatten({});",
                    self.indent, arg_vars[0].0
                ));
                return;
            }
            "zip" => {
                self.lines.push(format!(
                    "{}{target} = chelis_list_zip({}, {});",
                    self.indent, arg_vars[0].0, arg_vars[1].0
                ));
                return;
            }
            "enumerate" => {
                self.lines.push(format!(
                    "{}{target} = chelis_list_enumerate({});",
                    self.indent, arg_vars[0].0
                ));
                return;
            }
            "dict_of" => {
                self.lines.push(format!(
                    "{}{target} = chelis_dict_from_pairs({});",
                    self.indent, arg_vars[0].0
                ));
                return;
            }
            "dict_get" => {
                let getter = match ty {
                    HostType::Option(inner) if matches!(inner.as_ref(), HostType::Int64) => {
                        "chelis_dict_get_i64"
                    }
                    HostType::Option(inner) if matches!(inner.as_ref(), HostType::Float64) => {
                        "chelis_dict_get_f64"
                    }
                    _ => "chelis_dict_get",
                };
                self.lines.push(format!(
                    "{}{target} = {}({}, {});",
                    self.indent,
                    getter,
                    arg_vars[0].0,
                    self.box_value_expr(&arg_vars[1].0, &arg_vars[1].1)
                ));
                return;
            }
            "dict_contains" => {
                self.lines.push(format!(
                    "{}{target} = chelis_dict_contains({}, {});",
                    self.indent,
                    arg_vars[0].0,
                    self.box_value_expr(&arg_vars[1].0, &arg_vars[1].1)
                ));
                return;
            }
            "dict_remove" => {
                self.lines.push(format!(
                    "{}{target} = chelis_dict_remove({}, {});",
                    self.indent,
                    arg_vars[0].0,
                    self.box_value_expr(&arg_vars[1].0, &arg_vars[1].1)
                ));
                return;
            }
            "dict_insert" => {
                self.lines.push(format!(
                    "{}{target} = chelis_dict_insert({}, {}, {});",
                    self.indent,
                    arg_vars[0].0,
                    self.box_value_expr(&arg_vars[1].0, &arg_vars[1].1),
                    self.box_value_expr(&arg_vars[2].0, &arg_vars[2].1)
                ));
                return;
            }
            "dict_merge" => {
                self.lines.push(format!(
                    "{}{target} = chelis_dict_merge({}, {});",
                    self.indent, arg_vars[0].0, arg_vars[1].0
                ));
                return;
            }
            "dict_keys" => {
                self.lines.push(format!(
                    "{}{target} = chelis_dict_keys({});",
                    self.indent, arg_vars[0].0
                ));
                return;
            }
            "dict_values" => {
                self.lines.push(format!(
                    "{}{target} = chelis_dict_values({});",
                    self.indent, arg_vars[0].0
                ));
                return;
            }
            "dict_entries" => {
                self.lines.push(format!(
                    "{}{target} = chelis_dict_entries({});",
                    self.indent, arg_vars[0].0
                ));
                return;
            }
            "to_tensor" => {
                // RT-4 F1: when the destination tensor's precision is
                // known at compile time, dispatch through the typed
                // runtime entry point so the storage width matches
                // the declared dtype. Without this hint the runtime
                // would deduce dtype from the value-tag of the first
                // leaf — but `chelis_value_from_f64` has the same tag
                // (CHELIS_VALUE_FLOAT64) for both f32 and f64 sources,
                // so a declared `tensor[3, f64] = [1.0, 2.0, 3.0]`
                // silently truncated to f32 storage.
                if let HostType::Tensor(t) = ty {
                    let dtype_macro = match t.precision {
                        chelis_types::types::Prim::F32 => "CHELIS_F32",
                        chelis_types::types::Prim::F64 => "CHELIS_F64",
                        chelis_types::types::Prim::Bool => "CHELIS_BOOL",
                        chelis_types::types::Prim::Int8 => "CHELIS_I8",
                        chelis_types::types::Prim::Int16 => "CHELIS_I16",
                        chelis_types::types::Prim::Int32 => "CHELIS_I32",
                        chelis_types::types::Prim::Int64 => "CHELIS_I64",
                        // WS-1: bf16 / f16 admitted on `--target c`;
                        // typed host-lane construction stamps the
                        // declared 16-bit pattern into the tensor
                        // buffer via the runtime's `CHELIS_BF16` /
                        // `CHELIS_F16` allocator path.
                        chelis_types::types::Prim::Bf16 => "CHELIS_BF16",
                        chelis_types::types::Prim::F16 => "CHELIS_F16",
                        // f8e4m3 host literals are rejected by the C
                        // backend's precision gate (deferred per spec
                        // §1.1.1); if we reach here, fall through to
                        // the legacy entry so the diagnostic surfaces
                        // consistently. String tensors and any other
                        // non-numeric precision class are also routed
                        // through the legacy path.
                        _ => "",
                    };
                    if !dtype_macro.is_empty() {
                        self.lines.push(format!(
                            "{}{target} = chelis_tensor_from_value_list_typed({}, {dtype_macro});",
                            self.indent, arg_vars[0].0
                        ));
                        return;
                    }
                }
                self.lines.push(format!(
                    "{}{target} = chelis_tensor_from_value_list({});",
                    self.indent, arg_vars[0].0
                ));
                return;
            }
            "to_list" => {
                self.lines.push(format!(
                    "{}{target} = chelis_list_from_tensor({});",
                    self.indent, arg_vars[0].0
                ));
                return;
            }
            "pad_sequences" => {
                self.lines.push(format!(
                    "{}{target} = chelis_pad_sequences({}, {});",
                    self.indent,
                    arg_vars[0].0,
                    self.box_value_expr(&arg_vars[1].0, &arg_vars[1].1)
                ));
                return;
            }
            "pad_sequences_to" => {
                self.lines.push(format!(
                    "{}{target} = chelis_pad_sequences_to({}, {}, {});",
                    self.indent,
                    arg_vars[0].0,
                    arg_vars[1].0,
                    self.box_value_expr(&arg_vars[2].0, &arg_vars[2].1)
                ));
                return;
            }
            "read_file" => {
                self.lines.push(format!(
                    "{}{target} = chelis_read_file({});",
                    self.indent, arg_vars[0].0
                ));
                return;
            }
            "write_file" => {
                self.lines.push(format!(
                    "{}chelis_write_file({}, {});",
                    self.indent, arg_vars[0].0, arg_vars[1].0
                ));
                self.lines.push(format!("{}{target} = 0;", self.indent));
                return;
            }
            "read_lines" => {
                self.lines.push(format!(
                    "{}{target} = chelis_read_lines({});",
                    self.indent, arg_vars[0].0
                ));
                return;
            }
            "read_bytes" => {
                self.lines.push(format!(
                    "{}{target} = chelis_read_bytes({});",
                    self.indent, arg_vars[0].0
                ));
                return;
            }
            "file_exists" => {
                self.lines.push(format!(
                    "{}{target} = chelis_file_exists({});",
                    self.indent, arg_vars[0].0
                ));
                return;
            }
            "list_dir" => {
                self.lines.push(format!(
                    "{}{target} = chelis_list_dir({});",
                    self.indent, arg_vars[0].0
                ));
                return;
            }
            "mmap_file" => {
                self.lines.push(format!(
                    "{}{target} = chelis_mmap_file({});",
                    self.indent, arg_vars[0].0
                ));
                return;
            }
            "mmap_read" => {
                self.lines.push(format!(
                    "{}{target} = chelis_mmap_read({}, {}, {});",
                    self.indent, arg_vars[0].0, arg_vars[1].0, arg_vars[2].0
                ));
                return;
            }
            "mmap_len" => {
                self.lines.push(format!(
                    "{}{target} = chelis_mmap_len({});",
                    self.indent, arg_vars[0].0
                ));
                return;
            }
            _ => {}
        }

        let expr = match name {
            "add" => format!("{} + {}", arg_vars[0].0, arg_vars[1].0),
            "sub" => format!("{} - {}", arg_vars[0].0, arg_vars[1].0),
            "mul" => format!("{} * {}", arg_vars[0].0, arg_vars[1].0),
            "div" => format!("{} / {}", arg_vars[0].0, arg_vars[1].0),
            "mod" => format!("{} % {}", arg_vars[0].0, arg_vars[1].0),
            "cmplt" if matches!(arg_vars[0].1, HostType::Tensor(_)) => {
                format!("chelis_tensor_cmplt({}, {})", arg_vars[0].0, arg_vars[1].0)
            }
            "cmplt" => format!("{} < {}", arg_vars[0].0, arg_vars[1].0),
            "lt" => format!("{} < {}", arg_vars[0].0, arg_vars[1].0),
            "gt" => format!("{} > {}", arg_vars[0].0, arg_vars[1].0),
            "gte" => format!("{} >= {}", arg_vars[0].0, arg_vars[1].0),
            "lte" => format!("{} <= {}", arg_vars[0].0, arg_vars[1].0),
            "eq" => match (&arg_vars[0].1, &arg_vars[1].1) {
                (HostType::String, HostType::String) => {
                    format!("chelis_string_eq({}, {})", arg_vars[0].0, arg_vars[1].0)
                }
                _ => format!("{} == {}", arg_vars[0].0, arg_vars[1].0),
            },
            "neq" => match (&arg_vars[0].1, &arg_vars[1].1) {
                (HostType::String, HostType::String) => {
                    format!("!chelis_string_eq({}, {})", arg_vars[0].0, arg_vars[1].0)
                }
                _ => format!("{} != {}", arg_vars[0].0, arg_vars[1].0),
            },
            "and" => format!("{} && {}", arg_vars[0].0, arg_vars[1].0),
            "or" => format!("{} || {}", arg_vars[0].0, arg_vars[1].0),
            "not" => format!("!{}", arg_vars[0].0),
            "neg" => format!("-({})", arg_vars[0].0),
            "string_concat" => {
                format!("chelis_string_concat({}, {})", arg_vars[0].0, arg_vars[1].0)
            }
            "string_trim" => format!("chelis_string_trim({})", arg_vars[0].0),
            "reshape" => {
                format!(
                    "chelis_host_reshape_tensor({}, {})",
                    arg_vars[0].0, arg_vars[1].0
                )
            }
            "string_slice" => format!(
                "chelis_string_slice({}, {}, {})",
                arg_vars[0].0, arg_vars[1].0, arg_vars[2].0
            ),
            "string_contains" => {
                format!(
                    "chelis_string_contains({}, {})",
                    arg_vars[0].0, arg_vars[1].0
                )
            }
            "string_starts_with" => format!(
                "chelis_string_starts_with({}, {})",
                arg_vars[0].0, arg_vars[1].0
            ),
            "string_ends_with" => format!(
                "chelis_string_ends_with({}, {})",
                arg_vars[0].0, arg_vars[1].0
            ),
            "string_len" => format!("chelis_string_len({})", arg_vars[0].0),
            "to_string" => match arg_vars[0].1 {
                HostType::Int64 => format!("chelis_string_from_int64({})", arg_vars[0].0),
                HostType::Float64 => format!("chelis_string_from_f64({})", arg_vars[0].0),
                HostType::Bool => format!("chelis_string_from_bool({})", arg_vars[0].0),
                HostType::String => arg_vars[0].0.clone(),
                _ => "chelis_string_from_cstr(\"<value>\")".to_string(),
            },
            "to_int" => format!("chelis_parse_int64({})", arg_vars[0].0),
            "to_float" => format!("chelis_parse_f64({})", arg_vars[0].0),
            "print" => {
                self.emit_print_value(&arg_vars[0].0, &arg_vars[0].1);
                "0".to_string()
            }
            "fail" => {
                self.lines
                    .push(format!("{}chelis_fail({});", self.indent, arg_vars[0].0));
                return;
            }
            "debug" => {
                self.emit_print_value(&arg_vars[0].0, &arg_vars[0].1);
                arg_vars[0].0.clone()
            }
            "tensor_to_scalar" => format!("chelis_tensor_to_f64({})", arg_vars[0].0),
            // Issue #300: dispatch on the *result* tensor precision, not just
            // the (coarse) argument host type. `scalar_to_tensor(cast(c,
            // f32))` must materialize an f32-backed rank-0 tensor: the f64
            // constructor stores 8 bytes, and an f32 consumer (e.g. a DAG
            // `expand` helper lowered at the operand's f32 precision) then
            // decodes the low 4 bytes -- 0.0 for an exactly-representable
            // value like 2.5. `Float64` host-classifies both f32 and f64, so
            // the argument type alone cannot distinguish them; the result
            // `ty` carries the real precision.
            "scalar_to_tensor" => match (&arg_vars[0].1, ty) {
                (HostType::Int64, _) => {
                    format!("chelis_scalar_tensor_from_i64({})", arg_vars[0].0)
                }
                (_, HostType::Tensor(tensor_ty)) if matches!(tensor_ty.precision, Prim::F64) => {
                    format!("chelis_scalar_tensor_from_f64({})", arg_vars[0].0)
                }
                // Default float storage is f32 (matches the IR's `Const`
                // f32 default and the DAG-helper operand precision).
                _ => format!("chelis_scalar_tensor_from_f32({})", arg_vars[0].0),
            },
            "len" => match arg_vars[0].1 {
                HostType::Dict(_, _) => format!("chelis_dict_len({})", arg_vars[0].0),
                _ => format!("chelis_list_len({})", arg_vars[0].0),
            },
            "range" => format!("chelis_range_i64({}, {})", arg_vars[0].0, arg_vars[1].0),
            "rank" => format!("chelis_tensor_rank({})", arg_vars[0].0),
            "shape" => format!("chelis_tensor_shape({}, {})", arg_vars[0].0, arg_vars[1].0),
            "numel" => format!("chelis_tensor_numel({})", arg_vars[0].0),
            // Scalar math — these run on host `double` values in lowered
            // closures (e.g. the per-element GELU / RMSNorm map bodies).
            // The RISC DAG variants of these ops are handled separately in
            // `emit.rs`, but when a Surf `def` body is routed through the
            // host interpreter, we need the libm names directly.
            "sqrt" => format!("sqrt({})", arg_vars[0].0),
            "exp" => format!("exp({})", arg_vars[0].0),
            "log" => format!("log({})", arg_vars[0].0),
            "sin" => format!("sin({})", arg_vars[0].0),
            "cos" => format!("cos({})", arg_vars[0].0),
            "tanh" => format!("tanh({})", arg_vars[0].0),
            "pow" => format!("pow({}, {})", arg_vars[0].0, arg_vars[1].0),
            "abs" => match arg_vars[0].1 {
                HostType::Int64 => format!("llabs({})", arg_vars[0].0),
                _ => format!("fabs({})", arg_vars[0].0),
            },
            "min" => format!("fmin({}, {})", arg_vars[0].0, arg_vars[1].0),
            "max" => format!("fmax({}, {})", arg_vars[0].0, arg_vars[1].0),
            other => format!("/* unsupported builtin {other} */ 0"),
        };
        self.lines
            .push(format!("{}{target} = {expr};", self.indent));
        if matches!(ty, HostType::Unit) {
            self.lines.push(format!("{}{target} = 0;", self.indent));
        }
    }

    // W2 PR 3 of the 0.7.8 compiler cleanup workstream
    // (`CRuntime-F32Coupling`).  The four `*_elementwise` helpers
    // below previously wrote `t->data[i]` directly.  `chelis_tensor`
    // declares `data` as `float *` in the public runtime header so
    // every such access decoded the buffer at the f32 4-byte stride
    // regardless of `(*t).dtype` -- the same bug class closed by
    // PR #64 (CastMemcpy), PR #67 (ReshapeMemcpy), and PR #72
    // (PrintTensorF64) on the storage side, and by PR #84/#86 on the
    // Rust runtime side.  These helpers now emit an outer
    // `switch (target->dtype)` and read/write through typed pointer
    // casts in every arm.
    //
    // Per `docs/design/compiler_cleanup_0_7_8_spec_lock.md` Contract
    // 2 the supported precisions are f32, f64, i32, i64, and bool.
    // CHELIS_I32 and CHELIS_BOOL storage is 4-byte f32-encoded today
    // (see `crates/chelis-runtime/src/lib.rs` `chelis_alloc` and the
    // f32-routed runtime accessors at L2178-L2192 / L2222-L2223);
    // their arms route through `(float*)t->data` to match the
    // runtime convention.  CHELIS_F64 uses `(double*)` and
    // CHELIS_I64 uses `(int64_t*)`.
    //
    // The four helpers split into two pairs:
    //
    //   * `assign_tensor_binary_elementwise` /
    //     `assign_tensor_unary_elementwise` take a raw C operator
    //     (`+`, `-`, `*`, `/`, `!`, unary `-`) and emit the operator
    //     for every supported dtype arm.  All arms are semantically
    //     well-defined for the supported operators.
    //
    //   * `assign_tensor_binary_func_elementwise` /
    //     `assign_tensor_unary_func_elementwise` take a libm-style
    //     function name (`fmaxf`, `expf`, `chelis_host_relu_f32`,
    //     ...).  Today these helper names are all f32-only.  Calling
    //     them on f64 data through a `(double*)` cast would
    //     auto-convert at the call site but introduces precision
    //     loss; calling them on `(int64_t*)` is meaningless.  The
    //     dtype switch therefore routes f32 / i32 / bool through
    //     `(float*)t->data` (the existing semantics) and emits a
    //     `runtime_fail`-style abort for f64 and i64.  A future PR
    //     can lift the precision domain into the IR layer and emit
    //     `fmax` / `exp` / per-precision custom helpers in the
    //     f64 / i64 arms; until that lands the abort is the
    //     correct-by-construction surface.
    fn assign_tensor_binary_elementwise(&mut self, target: &str, lhs: &str, rhs: &str, op: &str) {
        self.lines.push(format!(
            "{}{target} = chelis_alloc({lhs}->ndim, {lhs}->shape, {lhs}->dtype);",
            self.indent
        ));
        self.lines
            .push(format!("{}switch ({target}->dtype) {{", self.indent));
        for arm in DtypeArm::all_operator_arms() {
            self.emit_binary_elementwise_arm(target, lhs, rhs, op, *arm);
        }
        self.emit_default_runtime_fail_arm_for(target, "binary elementwise op");
        self.lines.push(format!("{}}}", self.indent));
    }

    fn assign_tensor_binary_func_elementwise(
        &mut self,
        target: &str,
        lhs: &str,
        rhs: &str,
        func: &str,
    ) {
        self.lines.push(format!(
            "{}{target} = chelis_alloc({lhs}->ndim, {lhs}->shape, {lhs}->dtype);",
            self.indent
        ));
        self.lines
            .push(format!("{}switch ({target}->dtype) {{", self.indent));
        for arm in DtypeArm::all_f32_only_func_arms() {
            self.emit_binary_func_elementwise_arm(target, lhs, rhs, func, *arm);
        }
        // CHELIS_F64 and CHELIS_I64 abort: the func names threaded
        // through this helper are all f32-only today.
        self.emit_dtype_fail_arms(
            &[DtypeArm::F64, DtypeArm::I64],
            &format!("binary func elementwise ({func})"),
        );
        self.emit_default_runtime_fail_arm_for(
            target,
            &format!("binary func elementwise ({func})"),
        );
        self.lines.push(format!("{}}}", self.indent));
    }

    fn assign_tensor_unary_elementwise(&mut self, target: &str, input: &str, op: &str) {
        self.lines.push(format!(
            "{}{target} = chelis_alloc({input}->ndim, {input}->shape, {input}->dtype);",
            self.indent
        ));
        self.lines
            .push(format!("{}switch ({target}->dtype) {{", self.indent));
        for arm in DtypeArm::all_operator_arms() {
            self.emit_unary_elementwise_arm(target, input, op, *arm);
        }
        self.emit_default_runtime_fail_arm_for(target, "unary elementwise op");
        self.lines.push(format!("{}}}", self.indent));
    }

    fn assign_tensor_unary_func_elementwise(&mut self, target: &str, input: &str, func: &str) {
        self.lines.push(format!(
            "{}{target} = chelis_alloc({input}->ndim, {input}->shape, {input}->dtype);",
            self.indent
        ));
        self.lines
            .push(format!("{}switch ({target}->dtype) {{", self.indent));
        for arm in DtypeArm::all_f32_only_func_arms() {
            self.emit_unary_func_elementwise_arm(target, input, func, *arm);
        }
        self.emit_dtype_fail_arms(
            &[DtypeArm::F64, DtypeArm::I64],
            &format!("unary func elementwise ({func})"),
        );
        self.emit_default_runtime_fail_arm_for(target, &format!("unary func elementwise ({func})"));
        self.lines.push(format!("{}}}", self.indent));
    }

    /// Emit one arm of the elementwise binary operator dispatch.
    fn emit_binary_elementwise_arm(
        &mut self,
        target: &str,
        lhs: &str,
        rhs: &str,
        op: &str,
        arm: DtypeArm,
    ) {
        let ind = &self.indent;
        let macro_name = arm.dtype_macro();
        let elem_t = arm.elem_t();
        self.lines.push(format!("{ind}    case {macro_name}: {{"));
        self.lines.push(format!(
            "{ind}        {elem_t} *__target_data = ({elem_t}*){target}->data;"
        ));
        self.lines.push(format!(
            "{ind}        const {elem_t} *__lhs_data = (const {elem_t}*){lhs}->data;"
        ));
        self.lines.push(format!(
            "{ind}        const {elem_t} *__rhs_data = (const {elem_t}*){rhs}->data;"
        ));
        self.lines.push(format!(
            "{ind}        for (int i = 0; i < {target}->size; i++) {{"
        ));
        self.lines
            .push(format!("{ind}            int indices[CHELIS_MAX_DIM];"));
        self.lines.push(format!(
            "{ind}            chelis_flat_to_indices(i, {target}->shape, {target}->ndim, indices);"
        ));
        self.lines.push(format!(
            "{ind}            int idx_lhs = chelis_indices_to_flat(indices, {lhs}->strides, {lhs}->ndim);"
        ));
        self.lines.push(format!(
            "{ind}            int idx_rhs = chelis_indices_to_flat(indices, {rhs}->strides, {rhs}->ndim);"
        ));
        self.lines.push(format!(
            "{ind}            __target_data[i] = __lhs_data[idx_lhs] {op} __rhs_data[idx_rhs];"
        ));
        self.lines.push(format!("{ind}        }}"));
        self.lines.push(format!("{ind}        break;"));
        self.lines.push(format!("{ind}    }}"));
    }

    /// Emit one arm of the elementwise binary func dispatch.
    fn emit_binary_func_elementwise_arm(
        &mut self,
        target: &str,
        lhs: &str,
        rhs: &str,
        func: &str,
        arm: DtypeArm,
    ) {
        let ind = &self.indent;
        let macro_name = arm.dtype_macro();
        let elem_t = arm.elem_t();
        self.lines.push(format!("{ind}    case {macro_name}: {{"));
        self.lines.push(format!(
            "{ind}        {elem_t} *__target_data = ({elem_t}*){target}->data;"
        ));
        self.lines.push(format!(
            "{ind}        const {elem_t} *__lhs_data = (const {elem_t}*){lhs}->data;"
        ));
        self.lines.push(format!(
            "{ind}        const {elem_t} *__rhs_data = (const {elem_t}*){rhs}->data;"
        ));
        self.lines.push(format!(
            "{ind}        for (int i = 0; i < {target}->size; i++) {{"
        ));
        self.lines
            .push(format!("{ind}            int indices[CHELIS_MAX_DIM];"));
        self.lines.push(format!(
            "{ind}            chelis_flat_to_indices(i, {target}->shape, {target}->ndim, indices);"
        ));
        self.lines.push(format!(
            "{ind}            int idx_lhs = chelis_indices_to_flat(indices, {lhs}->strides, {lhs}->ndim);"
        ));
        self.lines.push(format!(
            "{ind}            int idx_rhs = chelis_indices_to_flat(indices, {rhs}->strides, {rhs}->ndim);"
        ));
        self.lines.push(format!(
            "{ind}            __target_data[i] = {func}(__lhs_data[idx_lhs], __rhs_data[idx_rhs]);"
        ));
        self.lines.push(format!("{ind}        }}"));
        self.lines.push(format!("{ind}        break;"));
        self.lines.push(format!("{ind}    }}"));
    }

    /// Emit one arm of the elementwise unary operator dispatch.
    fn emit_unary_elementwise_arm(&mut self, target: &str, input: &str, op: &str, arm: DtypeArm) {
        let ind = &self.indent;
        let macro_name = arm.dtype_macro();
        let elem_t = arm.elem_t();
        self.lines.push(format!("{ind}    case {macro_name}: {{"));
        self.lines.push(format!(
            "{ind}        {elem_t} *__target_data = ({elem_t}*){target}->data;"
        ));
        self.lines.push(format!(
            "{ind}        const {elem_t} *__input_data = (const {elem_t}*){input}->data;"
        ));
        self.lines.push(format!(
            "{ind}        for (int i = 0; i < {target}->size; i++) {{"
        ));
        self.lines
            .push(format!("{ind}            int indices[CHELIS_MAX_DIM];"));
        self.lines.push(format!(
            "{ind}            chelis_flat_to_indices(i, {target}->shape, {target}->ndim, indices);"
        ));
        self.lines.push(format!(
            "{ind}            int idx = chelis_indices_to_flat(indices, {input}->strides, {input}->ndim);"
        ));
        self.lines.push(format!(
            "{ind}            __target_data[i] = {op}__input_data[idx];"
        ));
        self.lines.push(format!("{ind}        }}"));
        self.lines.push(format!("{ind}        break;"));
        self.lines.push(format!("{ind}    }}"));
    }

    /// Emit one arm of the elementwise unary func dispatch.
    fn emit_unary_func_elementwise_arm(
        &mut self,
        target: &str,
        input: &str,
        func: &str,
        arm: DtypeArm,
    ) {
        let ind = &self.indent;
        let macro_name = arm.dtype_macro();
        let elem_t = arm.elem_t();
        self.lines.push(format!("{ind}    case {macro_name}: {{"));
        self.lines.push(format!(
            "{ind}        {elem_t} *__target_data = ({elem_t}*){target}->data;"
        ));
        self.lines.push(format!(
            "{ind}        const {elem_t} *__input_data = (const {elem_t}*){input}->data;"
        ));
        self.lines.push(format!(
            "{ind}        for (int i = 0; i < {target}->size; i++) {{"
        ));
        self.lines
            .push(format!("{ind}            int indices[CHELIS_MAX_DIM];"));
        self.lines.push(format!(
            "{ind}            chelis_flat_to_indices(i, {target}->shape, {target}->ndim, indices);"
        ));
        self.lines.push(format!(
            "{ind}            int idx = chelis_indices_to_flat(indices, {input}->strides, {input}->ndim);"
        ));
        self.lines.push(format!(
            "{ind}            __target_data[i] = {func}(__input_data[idx]);"
        ));
        self.lines.push(format!("{ind}        }}"));
        self.lines.push(format!("{ind}        break;"));
        self.lines.push(format!("{ind}    }}"));
    }

    /// Emit one or more `case CHELIS_*: { ... abort(); break; }` arms
    /// for dtypes the surrounding switch cannot service.
    fn emit_dtype_fail_arms(&mut self, arms: &[DtypeArm], site_name: &str) {
        let ind = &self.indent;
        for arm in arms {
            let macro_name = arm.dtype_macro();
            self.lines.push(format!("{ind}    case {macro_name}: {{"));
            self.lines.push(format!(
                "{ind}        fprintf(stderr, \"{site_name} unsupported for dtype {macro_name}\\n\");"
            ));
            self.lines.push(format!("{ind}        abort();"));
            self.lines.push(format!("{ind}    }}"));
        }
    }

    /// Emit the `default:` arm for an elementwise dtype switch.
    /// `target` names the dispatched-on tensor so the stderr message
    /// can include its actual dtype value at runtime.
    fn emit_default_runtime_fail_arm_for(&mut self, target: &str, site_name: &str) {
        let ind = &self.indent;
        self.lines.push(format!("{ind}    default: {{"));
        self.lines.push(format!(
            "{ind}        fprintf(stderr, \"{site_name} unsupported dtype %d\\n\", (int){target}->dtype);"
        ));
        self.lines.push(format!("{ind}        abort();"));
        self.lines.push(format!("{ind}    }}"));
    }

    fn assign_tensor_call(
        &mut self,
        target: &str,
        helper: usize,
        args: &[HostExpr],
        ty: &HostType,
    ) {
        if let Some(host_helper) = self.tensor_helpers.get(helper) {
            match host_helper.specialization.as_ref() {
                Some(HostTensorSpecialization::BlasMatmul(summary)) => {
                    self.assign_blas_matmul_summary(target, summary, args, ty);
                    return;
                }
                Some(HostTensorSpecialization::SparseGather(summary)) => {
                    self.assign_sparse_summary(
                        target,
                        SparseSummaryKind::Gather,
                        summary,
                        args,
                        ty,
                    );
                    return;
                }
                Some(HostTensorSpecialization::SparseScatterAdd(summary)) => {
                    self.assign_sparse_summary(
                        target,
                        SparseSummaryKind::ScatterAdd,
                        summary,
                        args,
                        ty,
                    );
                    return;
                }
                Some(HostTensorSpecialization::SparseScatterReplace(summary)) => {
                    self.assign_sparse_summary(
                        target,
                        SparseSummaryKind::ScatterReplace,
                        summary,
                        args,
                        ty,
                    );
                    return;
                }
                None => {}
            }
        }

        let helper_name = format!("{}__tensor_{helper}", self.helper_prefix);
        let tensor_args = args
            .iter()
            .enumerate()
            .map(|(index, arg)| {
                let inferred_ty = host_type(arg);
                if matches!(inferred_ty, HostType::Tensor(_)) {
                    let arg_name = self.next_temp(&format!("tensor_arg{index}"));
                    self.emit_expr_to_var(arg, &arg_name, &inferred_ty);
                    (arg_name, None)
                } else {
                    let value_name = self.next_temp(&format!("tensor_scalar{index}"));
                    self.emit_expr_to_var(arg, &value_name, &inferred_ty);
                    let tensor_name = self.next_temp(&format!("tensor_arg{index}"));
                    self.lines
                        .push(format!("{}chelis_tensor* {};", self.indent, tensor_name));
                    // Scalar inputs to tensor helpers use true rank-0 tensors so
                    // tensor[f32] keeps shape=[] across generated host/DAG calls.
                    //
                    // W2 PR 3 (CRuntime-F32Coupling): every arm casts
                    // `->data` through a typed pointer before writing
                    // the scalar.  The legacy bool / f32 arms wrote
                    // through the public `float *data` declaration in
                    // the C runtime header; mirror the int64 arm's
                    // typed-cast pattern for the bool and f32 cases
                    // so the bug class closes uniformly.  Bool
                    // storage today is 4-byte f32-encoded (per
                    // `docs/investigations/c_runtime_dtype_accessors_diagnosis.md`),
                    // so the bool arm casts to `(float*)` and writes
                    // the 1.0f / 0.0f bit pattern.
                    let (dtype, store) = match inferred_ty {
                        HostType::Int64 => (
                            "CHELIS_I64",
                            format!("((int64_t*){tensor_name}->data)[0] = {value_name};"),
                        ),
                        HostType::Bool => (
                            "CHELIS_BOOL",
                            format!(
                                "((float*){tensor_name}->data)[0] = {value_name} ? 1.0f : 0.0f;"
                            ),
                        ),
                        _ => (
                            "CHELIS_F32",
                            format!("((float*){tensor_name}->data)[0] = (float)({value_name});"),
                        ),
                    };
                    self.lines.push(format!(
                        "{}{tensor_name} = chelis_alloc(0, NULL, {dtype});",
                        self.indent
                    ));
                    self.lines.push(format!("{}{store}", self.indent));
                    (tensor_name.clone(), Some(tensor_name))
                }
            })
            .collect::<Vec<_>>();
        let outputs_name = self.next_temp("outputs");
        // A constant-only tensor helper (e.g. `expand(scalar_to_tensor(c),
        // 0, n)`) has zero inputs. ISO C forbids a zero-length array
        // (`chelis_tensor *inputs[0];`), so pass a NULL inputs pointer with
        // count 0 instead; the helper's `n_in == 0` guard never dereferences
        // it (issue #300).
        let inputs_arg = if tensor_args.is_empty() {
            "NULL".to_string()
        } else {
            let inputs_name = self.next_temp("inputs");
            self.lines.push(format!(
                "{}chelis_tensor *{}[{}];",
                self.indent,
                inputs_name,
                tensor_args.len()
            ));
            for (index, (arg, _)) in tensor_args.iter().enumerate() {
                self.lines.push(format!(
                    "{}{}[{index}] = {};",
                    self.indent, inputs_name, arg
                ));
            }
            inputs_name
        };
        // Issue #309: a helper whose body has more than one DAG root
        // (the canonical case is a multi-`wrt` `grad`) writes one
        // tensor per root into `outputs[0..n_out]` and its emitted
        // wrapper asserts `n_out == roots().len()`. Size the output
        // array and the `n_out` argument from the helper's actual root
        // count; the prior hard-coded `[1]` / `n_out = 1` both crashed
        // the helper's arity guard for a multi-output grad and left the
        // downstream `.N` projection reading a single tensor as if it
        // were a tuple. When the call is tuple-typed, box each output
        // tensor and assemble a real `chelis_tuple` so the subsequent
        // `chelis_tuple_get` projection has a correctly-typed receiver.
        let root_count = self
            .tensor_helpers
            .get(helper)
            .map(|host_helper| host_helper.dag.roots().len().max(1))
            .unwrap_or(1);
        self.lines.push(format!(
            "{}chelis_tensor *{}[{}] = {{ NULL }};",
            self.indent, outputs_name, root_count
        ));
        self.lines.push(format!(
            "{}{}({}, {}, {}, {});",
            self.indent,
            helper_name,
            inputs_arg,
            tensor_args.len(),
            outputs_name,
            root_count
        ));
        if let HostType::Tuple(parts) = ty
            && root_count > 1
        {
            let values_name = self.next_temp("tuple_values");
            self.lines.push(format!(
                "{}chelis_value {}[{}];",
                self.indent, values_name, root_count
            ));
            for index in 0..root_count {
                // Each helper output slot is a `chelis_tensor*`; box it as
                // a tensor value regardless of the tuple part annotation
                // (a multi-root tensor helper only ever produces tensors).
                let elem_ty = parts
                    .get(index)
                    .filter(|part| matches!(part, HostType::Tensor(_)))
                    .cloned()
                    .unwrap_or(HostType::Tensor(TensorType {
                        dims: Vec::new(),
                        precision: Prim::F32,
                    }));
                let slot_expr = format!("{outputs_name}[{index}]");
                self.lines.push(format!(
                    "{}{}[{index}] = {};",
                    self.indent,
                    values_name,
                    self.box_value_expr(&slot_expr, &elem_ty)
                ));
            }
            self.lines.push(format!(
                "{}{target} = chelis_tuple_from_values({}, {});",
                self.indent, values_name, root_count
            ));
        } else {
            self.lines
                .push(format!("{}{target} = {}[0];", self.indent, outputs_name));
        }
        for (_, boxed) in tensor_args {
            if let Some(boxed) = boxed {
                self.lines
                    .push(format!("{}chelis_free({boxed});", self.indent));
            }
        }
    }

    fn assign_blas_matmul_summary(
        &mut self,
        target: &str,
        summary: &HostBlasMatmulSummary,
        args: &[HostExpr],
        _ty: &HostType,
    ) {
        assert_eq!(
            args.len(),
            summary.input_tys.len(),
            "BLAS summary argument count must match callsite argument count"
        );
        let tensor_args = args
            .iter()
            .enumerate()
            .map(|(index, arg)| {
                let arg_name = self.next_temp(&format!("blas_arg{index}"));
                let expected_ty = HostType::Tensor(
                    summary
                        .input_tys
                        .get(index)
                        .expect("summary input type")
                        .clone(),
                );
                self.emit_expr_to_var(arg, &arg_name, &expected_ty);
                arg_name
            })
            .collect::<Vec<_>>();
        self.emit_blas_summary_contract(summary, &tensor_args);

        let lhs = tensor_args
            .get(summary.lhs_input)
            .expect("summary lhs input index");
        let rhs = tensor_args
            .get(summary.rhs_input)
            .expect("summary rhs input index");
        let m_expr = self.summary_dim_expr(&summary.m, summary, &tensor_args);
        let n_expr = self.summary_dim_expr(&summary.n, summary, &tensor_args);
        let k_expr = self.summary_dim_expr(&summary.k, summary, &tensor_args);
        let output_dims = summary
            .batch_dims
            .iter()
            .chain([&summary.m, &summary.n])
            .map(|dim| self.summary_dim_expr(dim, summary, &tensor_args))
            .collect::<Vec<_>>();
        let shape_name = self.next_temp("blas_shape");
        self.lines.push(format!(
            "{}int {shape_name}[{}] = {{ {} }};",
            self.indent,
            output_dims.len(),
            output_dims.join(", ")
        ));
        self.lines.push(format!(
            "{}{target} = chelis_alloc({}, {shape_name}, CHELIS_F32);",
            self.indent,
            output_dims.len()
        ));

        let lhs_contig = self.next_temp("blas_lhs");
        let rhs_contig = self.next_temp("blas_rhs");
        self.lines.push(format!(
            "{}chelis_tensor *{lhs_contig} = {lhs};",
            self.indent
        ));
        self.lines.push(format!(
            "{}if (!({lhs_contig}->ndim >= 2 && {lhs_contig}->strides[{lhs_contig}->ndim - 1] == 1 && {lhs_contig}->strides[{lhs_contig}->ndim - 2] == {k_expr})) {{",
            self.indent
        ));
        self.lines.push(format!(
            "{}    {lhs_contig} = chelis_contiguous({lhs_contig});",
            self.indent
        ));
        self.lines.push(format!("{}}}", self.indent));
        self.lines.push(format!(
            "{}chelis_tensor *{rhs_contig} = {rhs};",
            self.indent
        ));
        self.lines.push(format!(
            "{}if (!({rhs_contig}->ndim >= 2 && {rhs_contig}->strides[{rhs_contig}->ndim - 1] == 1 && {rhs_contig}->strides[{rhs_contig}->ndim - 2] == {n_expr})) {{",
            self.indent
        ));
        self.lines.push(format!(
            "{}    {rhs_contig} = chelis_contiguous({rhs_contig});",
            self.indent
        ));
        self.lines.push(format!("{}}}", self.indent));

        if summary.batch_dims.is_empty() {
            self.lines.push(format!(
                "{}cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, {m_expr}, {n_expr}, {k_expr}, 1.0f, {lhs_contig}->data, {k_expr}, {rhs_contig}->data, {n_expr}, 0.0f, {target}->data, {n_expr});",
                self.indent
            ));
        } else {
            let batch_count = summary
                .batch_dims
                .iter()
                .map(|dim| self.summary_dim_expr(dim, summary, &tensor_args))
                .reduce(|lhs, rhs| format!("({lhs} * {rhs})"))
                .unwrap_or_else(|| "1".to_string());
            let batch = self.next_temp("blas_batch");
            let rem = self.next_temp("blas_rem");
            let lhs_offset = self.next_temp("blas_lhs_offset");
            let rhs_offset = self.next_temp("blas_rhs_offset");
            let out_offset = self.next_temp("blas_out_offset");
            self.lines.push(format!(
                "{}for (int {batch} = 0; {batch} < {batch_count}; {batch}++) {{",
                self.indent
            ));
            self.lines
                .push(format!("{}    int {rem} = {batch};", self.indent));
            self.lines
                .push(format!("{}    int {lhs_offset} = 0;", self.indent));
            self.lines
                .push(format!("{}    int {rhs_offset} = 0;", self.indent));
            self.lines
                .push(format!("{}    int {out_offset} = 0;", self.indent));
            for axis in (0..summary.batch_dims.len()).rev() {
                let dim_expr =
                    self.summary_dim_expr(&summary.batch_dims[axis], summary, &tensor_args);
                let coord = self.next_temp(&format!("blas_coord_{axis}"));
                self.lines.push(format!(
                    "{}    int {coord} = {rem} % ({dim_expr});",
                    self.indent
                ));
                self.lines
                    .push(format!("{}    {rem} /= ({dim_expr});", self.indent));
                self.lines.push(format!(
                    "{}    {lhs_offset} += {coord} * {lhs_contig}->strides[{axis}];",
                    self.indent
                ));
                self.lines.push(format!(
                    "{}    {rhs_offset} += {coord} * {rhs_contig}->strides[{axis}];",
                    self.indent
                ));
                self.lines.push(format!(
                    "{}    {out_offset} += {coord} * {target}->strides[{axis}];",
                    self.indent
                ));
            }
            self.lines.push(format!(
                "{}    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, {m_expr}, {n_expr}, {k_expr}, 1.0f, {lhs_contig}->data + {lhs_offset}, {k_expr}, {rhs_contig}->data + {rhs_offset}, {n_expr}, 0.0f, {target}->data + {out_offset}, {n_expr});",
                self.indent
            ));
            self.lines.push(format!("{}}}", self.indent));
        }
        self.lines.push(format!(
            "{}if ({lhs_contig} != {lhs}) chelis_free({lhs_contig});",
            self.indent
        ));
        self.lines.push(format!(
            "{}if ({rhs_contig} != {rhs}) chelis_free({rhs_contig});",
            self.indent
        ));
    }

    fn emit_blas_summary_contract(
        &mut self,
        summary: &HostBlasMatmulSummary,
        tensor_args: &[String],
    ) {
        let mut symbolic_first = HashMap::<String, String>::new();
        for (input_index, (arg, ty)) in tensor_args.iter().zip(summary.input_tys.iter()).enumerate()
        {
            self.lines
                .push(format!("{}if ({arg} == NULL) {{", self.indent));
            self.lines.push(format!(
                "{}    fprintf(stderr, \"specialized BLAS call input {input_index} is NULL\\n\");",
                self.indent
            ));
            self.lines.push(format!("{}    abort();", self.indent));
            self.lines.push(format!("{}}}", self.indent));
            self.lines
                .push(format!("{}if ({arg}->dtype != CHELIS_F32) {{", self.indent));
            self.lines.push(format!(
                "{}    fprintf(stderr, \"specialized BLAS call input {input_index} expected f32 tensor\\n\");",
                self.indent
            ));
            self.lines.push(format!("{}    abort();", self.indent));
            self.lines.push(format!("{}}}", self.indent));
            self.lines.push(format!(
                "{}if ({arg}->ndim != {}) {{",
                self.indent,
                ty.dims.len()
            ));
            self.lines.push(format!(
                "{}    fprintf(stderr, \"specialized BLAS call input {input_index} expected rank {}, got %d\\n\", {arg}->ndim);",
                self.indent,
                ty.dims.len()
            ));
            self.lines.push(format!("{}    abort();", self.indent));
            self.lines.push(format!("{}}}", self.indent));
            for (axis, dim) in ty.dims.iter().enumerate() {
                match dim {
                    DimInfo::Lit(size) | DimInfo::Named(_, Some(size)) => {
                        self.lines.push(format!(
                            "{}if ({arg}->shape[{axis}] != {size}) {{",
                            self.indent
                        ));
                        self.lines.push(format!(
                            "{}    fprintf(stderr, \"specialized BLAS call input {input_index} axis {axis} expected {size}, got %d\\n\", {arg}->shape[{axis}]);",
                            self.indent
                        ));
                        self.lines.push(format!("{}    abort();", self.indent));
                        self.lines.push(format!("{}}}", self.indent));
                    }
                    DimInfo::Named(name, None) => {
                        let expr = format!("{arg}->shape[{axis}]");
                        if let Some(first) = symbolic_first.get(name) {
                            self.lines
                                .push(format!("{}if ({expr} != {first}) {{", self.indent));
                            self.lines.push(format!(
                                "{}    fprintf(stderr, \"specialized BLAS call symbolic dimension mismatch\\n\");",
                                self.indent
                            ));
                            self.lines.push(format!("{}    abort();", self.indent));
                            self.lines.push(format!("{}}}", self.indent));
                        } else {
                            symbolic_first.insert(name.clone(), expr);
                        }
                    }
                }
            }
        }
    }

    fn summary_dim_expr(
        &self,
        dim: &DimExpr,
        summary: &HostBlasMatmulSummary,
        tensor_args: &[String],
    ) -> String {
        match dim {
            DimExpr::Concrete(value) => value.to_string(),
            DimExpr::Sym(name) => self
                .summary_symbol_expr(name, summary, tensor_args)
                .unwrap_or_else(|| panic!("BLAS summary symbol `{name}` has no input binding")),
            DimExpr::Mul(lhs, rhs) => format!(
                "({} * {})",
                self.summary_dim_expr(lhs, summary, tensor_args),
                self.summary_dim_expr(rhs, summary, tensor_args)
            ),
            DimExpr::Div(lhs, rhs) => format!(
                "({} / {})",
                self.summary_dim_expr(lhs, summary, tensor_args),
                self.summary_dim_expr(rhs, summary, tensor_args)
            ),
        }
    }

    fn summary_symbol_expr(
        &self,
        name: &str,
        summary: &HostBlasMatmulSummary,
        tensor_args: &[String],
    ) -> Option<String> {
        summary
            .input_tys
            .iter()
            .zip(tensor_args.iter())
            .find_map(|(ty, arg)| {
                ty.dims.iter().enumerate().find_map(|(axis, dim)| {
                    matches!(dim, DimInfo::Named(dim_name, None) if dim_name == name)
                        .then(|| format!("{arg}->shape[{axis}]"))
                })
            })
    }

    /// Emit a summary-derived inline sparse op (Gather / ScatterAdd /
    /// Scatter-replace). The loop body mirrors the direct-call C
    /// emission at `chelis_backend_c::emit::CEmitter::emit_sparse_*`
    /// so a user-`def` wrapper compiles to the same bounded sparse
    /// loop as `f(table, indices) = gather(table, indices, 0)`.
    ///
    /// Argument marshaling matches `assign_blas_matmul_summary`:
    /// callers' `args` are materialized into local tensor pointers
    /// (with their `summary.input_tys[i]` as the expected type), then
    /// contract assertions (nonnull, dtype, rank, dim) lock the
    /// summary's expectations.
    fn assign_sparse_summary(
        &mut self,
        target: &str,
        kind: SparseSummaryKind,
        summary: &HostSparseOpSummary,
        args: &[HostExpr],
        _ty: &HostType,
    ) {
        assert_eq!(
            args.len(),
            summary.input_tys.len(),
            "sparse summary argument count must match callsite argument count"
        );
        let tensor_args = args
            .iter()
            .enumerate()
            .map(|(index, arg)| {
                let arg_name = self.next_temp(&format!("sparse_arg{index}"));
                let expected_ty = HostType::Tensor(
                    summary
                        .input_tys
                        .get(index)
                        .expect("sparse summary input type")
                        .clone(),
                );
                self.emit_expr_to_var(arg, &arg_name, &expected_ty);
                arg_name
            })
            .collect::<Vec<_>>();
        self.emit_sparse_summary_contract(summary, &tensor_args);

        let target_dtype = sparse_dtype_macro(summary.output.precision);
        let target_elem_t = sparse_elem_type(summary.output.precision);
        let target_elem_size = format!("sizeof({target_elem_t})");

        // Build output shape from `summary.output.dims`. Symbolic dims
        // resolve via the caller's tensor-arg shape (the contract
        // assertions above already verified those are consistent).
        let output_dims = summary
            .output
            .dims
            .iter()
            .map(|dim| sparse_dim_info_expr(dim, summary, &tensor_args))
            .collect::<Vec<_>>();
        let shape_name = self.next_temp("sparse_shape");
        self.lines.push(format!(
            "{}int {shape_name}[{}] = {{ {} }};",
            self.indent,
            output_dims.len(),
            output_dims.join(", ")
        ));
        self.lines.push(format!(
            "{}{target} = chelis_alloc({}, {shape_name}, {target_dtype});",
            self.indent,
            output_dims.len()
        ));

        match kind {
            SparseSummaryKind::Gather => {
                self.emit_sparse_gather_summary_body(target, summary, &tensor_args, target_elem_t);
            }
            SparseSummaryKind::ScatterAdd => {
                self.emit_sparse_scatter_summary_body(
                    target,
                    summary,
                    &tensor_args,
                    target_elem_t,
                    &target_elem_size,
                    /* accumulate */ true,
                );
            }
            SparseSummaryKind::ScatterReplace => {
                self.emit_sparse_scatter_summary_body(
                    target,
                    summary,
                    &tensor_args,
                    target_elem_t,
                    &target_elem_size,
                    /* accumulate */ false,
                );
            }
        }
    }

    /// Emit input-contract assertions for a sparse summary: nonnull,
    /// dtype, rank, and per-axis dim checks. Mirrors
    /// `emit_blas_summary_contract` so summary-derived sparse
    /// codepaths fail loudly on the same shape mismatches the BLAS
    /// path catches.
    fn emit_sparse_summary_contract(
        &mut self,
        summary: &HostSparseOpSummary,
        tensor_args: &[String],
    ) {
        let mut symbolic_first = HashMap::<String, String>::new();
        for (input_index, (arg, ty)) in tensor_args.iter().zip(summary.input_tys.iter()).enumerate()
        {
            let expected_dtype = sparse_dtype_macro(ty.precision);
            self.lines
                .push(format!("{}if ({arg} == NULL) {{", self.indent));
            self.lines.push(format!(
                "{}    fprintf(stderr, \"specialized sparse call input {input_index} is NULL\\n\");",
                self.indent
            ));
            self.lines.push(format!("{}    abort();", self.indent));
            self.lines.push(format!("{}}}", self.indent));
            self.lines.push(format!(
                "{}if ({arg}->dtype != {expected_dtype}) {{",
                self.indent
            ));
            self.lines.push(format!(
                "{}    fprintf(stderr, \"specialized sparse call input {input_index} expected {expected_dtype} tensor\\n\");",
                self.indent
            ));
            self.lines.push(format!("{}    abort();", self.indent));
            self.lines.push(format!("{}}}", self.indent));
            self.lines.push(format!(
                "{}if ({arg}->ndim != {}) {{",
                self.indent,
                ty.dims.len()
            ));
            self.lines.push(format!(
                "{}    fprintf(stderr, \"specialized sparse call input {input_index} expected rank {}, got %d\\n\", {arg}->ndim);",
                self.indent,
                ty.dims.len()
            ));
            self.lines.push(format!("{}    abort();", self.indent));
            self.lines.push(format!("{}}}", self.indent));
            for (axis, dim) in ty.dims.iter().enumerate() {
                match dim {
                    DimInfo::Lit(size) | DimInfo::Named(_, Some(size)) => {
                        self.lines.push(format!(
                            "{}if ({arg}->shape[{axis}] != {size}) {{",
                            self.indent
                        ));
                        self.lines.push(format!(
                            "{}    fprintf(stderr, \"specialized sparse call input {input_index} axis {axis} expected {size}, got %d\\n\", {arg}->shape[{axis}]);",
                            self.indent
                        ));
                        self.lines.push(format!("{}    abort();", self.indent));
                        self.lines.push(format!("{}}}", self.indent));
                    }
                    DimInfo::Named(name, None) => {
                        let expr = format!("{arg}->shape[{axis}]");
                        if let Some(first) = symbolic_first.get(name) {
                            self.lines
                                .push(format!("{}if ({expr} != {first}) {{", self.indent));
                            self.lines.push(format!(
                                "{}    fprintf(stderr, \"specialized sparse call symbolic dimension `{name}` mismatch\\n\");",
                                self.indent
                            ));
                            self.lines.push(format!("{}    abort();", self.indent));
                            self.lines.push(format!("{}}}", self.indent));
                        } else {
                            symbolic_first.insert(name.clone(), expr);
                        }
                    }
                }
            }
        }
    }

    /// Emit the Gather loop body for a summary-derived callsite.
    /// Operands come from `summary.input_indices` referencing
    /// `tensor_args`: `[values, indices]`.
    ///
    /// The emitted shape mirrors `emit_sparse_gather` so structural
    /// tests can match the same `_g`, `_values_data`, `_indices_data`,
    /// `_out_data` markers and the same nested `(b, i, d)` loop
    /// ordering.
    fn emit_sparse_gather_summary_body(
        &mut self,
        target: &str,
        summary: &HostSparseOpSummary,
        tensor_args: &[String],
        target_elem_t: &str,
    ) {
        let values_arg = &tensor_args[summary.input_indices[0]];
        let indices_arg = &tensor_args[summary.input_indices[1]];
        let values_ty = &summary.input_tys[summary.input_indices[0]];
        let indices_ty = &summary.input_tys[summary.input_indices[1]];
        let values_elem_t = sparse_elem_type(values_ty.precision);
        let indices_elem_t = sparse_elem_type(indices_ty.precision);
        let before = sparse_dim_product(&values_ty.dims[..summary.axis], summary, tensor_args);
        let axis_size = sparse_dim_info_expr(&values_ty.dims[summary.axis], summary, tensor_args);
        let after = sparse_dim_product(&values_ty.dims[summary.axis + 1..], summary, tensor_args);

        let values_ct = self.next_temp("sparse_values");
        let indices_ct = self.next_temp("sparse_indices");
        self.lines.push(format!(
            "{}chelis_tensor *{values_ct} = chelis_contiguous({values_arg});",
            self.indent
        ));
        self.lines.push(format!(
            "{}chelis_tensor *{indices_ct} = chelis_contiguous({indices_arg});",
            self.indent
        ));
        self.lines.push(format!(
            "{}const {values_elem_t} *{values_ct}_data = (const {values_elem_t}*){values_ct}->data;",
            self.indent
        ));
        self.lines.push(format!(
            "{}const {indices_elem_t} *{indices_ct}_data = (const {indices_elem_t}*){indices_ct}->data;",
            self.indent
        ));
        self.lines.push(format!(
            "{}{target_elem_t} *{target}_out_data = ({target_elem_t}*){target}->data;",
            self.indent
        ));
        self.lines
            .push(format!("{}int {target}_before = {before};", self.indent));
        self.lines.push(format!(
            "{}int {target}_axis_size = {axis_size};",
            self.indent
        ));
        self.lines
            .push(format!("{}int {target}_after = {after};", self.indent));
        self.lines.push(format!(
            "{}int {target}_index_count = {indices_ct}->size;",
            self.indent
        ));
        self.lines.push(format!(
            "{}for (int {target}_b = 0; {target}_b < {target}_before; {target}_b++) {{",
            self.indent
        ));
        self.lines.push(format!(
            "{}    for (int {target}_i = 0; {target}_i < {target}_index_count; {target}_i++) {{",
            self.indent
        ));
        self.lines.push(format!(
            "{}        int {target}_g = ({indices_ct}->dtype == CHELIS_I64) ? (int)((const int64_t*){indices_ct}->data)[{target}_i] : (int)({indices_ct}_data)[{target}_i];",
            self.indent
        ));
        self.lines.push(format!(
            "{}        if ({target}_g < 0 || {target}_g >= {target}_axis_size) abort();",
            self.indent
        ));
        self.lines.push(format!(
            "{}        for (int {target}_d = 0; {target}_d < {target}_after; {target}_d++) {{",
            self.indent
        ));
        self.lines.push(format!(
            "{}            int {target}_out = (({target}_b * {target}_index_count + {target}_i) * {target}_after) + {target}_d;",
            self.indent
        ));
        self.lines.push(format!(
            "{}            int {target}_src = (({target}_b * {target}_axis_size + {target}_g) * {target}_after) + {target}_d;",
            self.indent
        ));
        self.lines.push(format!(
            "{}            {target}_out_data[{target}_out] = {values_ct}_data[{target}_src];",
            self.indent
        ));
        self.lines.push(format!("{}        }}", self.indent));
        self.lines.push(format!("{}    }}", self.indent));
        self.lines.push(format!("{}}}", self.indent));
        self.lines.push(format!(
            "{}if ({values_ct} != {values_arg}) chelis_free({values_ct});",
            self.indent
        ));
        self.lines.push(format!(
            "{}if ({indices_ct} != {indices_arg}) chelis_free({indices_ct});",
            self.indent
        ));
    }

    /// Emit the ScatterAdd or Scatter-replace loop body for a
    /// summary-derived callsite. Operands are `[target_in, indices,
    /// updates]`. `accumulate=true` selects `+=` (ScatterAdd);
    /// `accumulate=false` selects `=` (last-write-wins Scatter, single
    /// threaded to preserve the deterministic order documented in
    /// `spec/05-risc-primitives.md` §3.5).
    #[allow(clippy::too_many_arguments)]
    fn emit_sparse_scatter_summary_body(
        &mut self,
        target: &str,
        summary: &HostSparseOpSummary,
        tensor_args: &[String],
        target_elem_t: &str,
        target_elem_size: &str,
        accumulate: bool,
    ) {
        let target_arg = &tensor_args[summary.input_indices[0]];
        let indices_arg = &tensor_args[summary.input_indices[1]];
        let updates_arg = &tensor_args[summary.input_indices[2]];
        let target_ty = &summary.input_tys[summary.input_indices[0]];
        let indices_ty = &summary.input_tys[summary.input_indices[1]];
        let updates_ty = &summary.input_tys[summary.input_indices[2]];
        let indices_elem_t = sparse_elem_type(indices_ty.precision);
        let updates_elem_t = sparse_elem_type(updates_ty.precision);
        let before = sparse_dim_product(&target_ty.dims[..summary.axis], summary, tensor_args);
        let axis_size = sparse_dim_info_expr(&target_ty.dims[summary.axis], summary, tensor_args);
        let after = sparse_dim_product(&target_ty.dims[summary.axis + 1..], summary, tensor_args);

        let target_ct = self.next_temp("sparse_target");
        let indices_ct = self.next_temp("sparse_indices");
        let updates_ct = self.next_temp("sparse_updates");
        self.lines.push(format!(
            "{}chelis_tensor *{target_ct} = chelis_contiguous({target_arg});",
            self.indent
        ));
        self.lines.push(format!(
            "{}chelis_tensor *{indices_ct} = chelis_contiguous({indices_arg});",
            self.indent
        ));
        self.lines.push(format!(
            "{}chelis_tensor *{updates_ct} = chelis_contiguous({updates_arg});",
            self.indent
        ));
        self.lines.push(format!(
            "{}const {indices_elem_t} *{indices_ct}_data = (const {indices_elem_t}*){indices_ct}->data;",
            self.indent
        ));
        self.lines.push(format!(
            "{}const {updates_elem_t} *{updates_ct}_data = (const {updates_elem_t}*){updates_ct}->data;",
            self.indent
        ));
        self.lines.push(format!(
            "{}{target_elem_t} *{target}_out_data = ({target_elem_t}*){target}->data;",
            self.indent
        ));
        self.lines.push(format!(
            "{}memcpy({target}->data, {target_ct}->data, (size_t){target}->size * {target_elem_size});",
            self.indent
        ));
        self.lines
            .push(format!("{}int {target}_before = {before};", self.indent));
        self.lines.push(format!(
            "{}int {target}_axis_size = {axis_size};",
            self.indent
        ));
        self.lines
            .push(format!("{}int {target}_after = {after};", self.indent));
        self.lines.push(format!(
            "{}int {target}_index_count = {indices_ct}->size;",
            self.indent
        ));
        self.lines.push(format!(
            "{}for (int {target}_b = 0; {target}_b < {target}_before; {target}_b++) {{",
            self.indent
        ));
        self.lines.push(format!(
            "{}    for (int {target}_i = 0; {target}_i < {target}_index_count; {target}_i++) {{",
            self.indent
        ));
        self.lines.push(format!(
            "{}        int {target}_g = ({indices_ct}->dtype == CHELIS_I64) ? (int)((const int64_t*){indices_ct}->data)[{target}_i] : (int)({indices_ct}_data)[{target}_i];",
            self.indent
        ));
        self.lines.push(format!(
            "{}        if ({target}_g < 0 || {target}_g >= {target}_axis_size) abort();",
            self.indent
        ));
        self.lines.push(format!(
            "{}        for (int {target}_d = 0; {target}_d < {target}_after; {target}_d++) {{",
            self.indent
        ));
        self.lines.push(format!(
            "{}            int {target}_src = (({target}_b * {target}_index_count + {target}_i) * {target}_after) + {target}_d;",
            self.indent
        ));
        self.lines.push(format!(
            "{}            int {target}_out = (({target}_b * {target}_axis_size + {target}_g) * {target}_after) + {target}_d;",
            self.indent
        ));
        let op = if accumulate { "+=" } else { "=" };
        self.lines.push(format!(
            "{}            {target}_out_data[{target}_out] {op} {updates_ct}_data[{target}_src];",
            self.indent
        ));
        self.lines.push(format!("{}        }}", self.indent));
        self.lines.push(format!("{}    }}", self.indent));
        self.lines.push(format!("{}}}", self.indent));
        self.lines.push(format!(
            "{}if ({target_ct} != {target_arg}) chelis_free({target_ct});",
            self.indent
        ));
        self.lines.push(format!(
            "{}if ({indices_ct} != {indices_arg}) chelis_free({indices_ct});",
            self.indent
        ));
        self.lines.push(format!(
            "{}if ({updates_ct} != {updates_arg}) chelis_free({updates_ct});",
            self.indent
        ));
    }

    fn assign_call(
        &mut self,
        target: &str,
        function: &str,
        args: &[HostExpr],
        arg_tys: &[HostType],
        ty: &HostType,
    ) {
        if let Some(spec) = self.function_specializations.get(function).cloned() {
            match spec {
                HostFunctionSpecialization::BlasMatmul(summary) => {
                    self.assign_blas_matmul_summary(target, &summary, args, ty);
                    return;
                }
                HostFunctionSpecialization::SparseGather(summary) => {
                    self.assign_sparse_summary(
                        target,
                        SparseSummaryKind::Gather,
                        &summary,
                        args,
                        ty,
                    );
                    return;
                }
                HostFunctionSpecialization::SparseScatterAdd(summary) => {
                    self.assign_sparse_summary(
                        target,
                        SparseSummaryKind::ScatterAdd,
                        &summary,
                        args,
                        ty,
                    );
                    return;
                }
                HostFunctionSpecialization::SparseScatterReplace(summary) => {
                    self.assign_sparse_summary(
                        target,
                        SparseSummaryKind::ScatterReplace,
                        &summary,
                        args,
                        ty,
                    );
                    return;
                }
            }
        }

        let arg_vars = args
            .iter()
            .enumerate()
            .map(|(index, arg)| {
                let arg_name = self.next_temp(&format!("call_arg{index}"));
                let inferred_ty = host_type(arg);
                let expected_ty = arg_tys.get(index).cloned().unwrap_or(HostType::Unknown);
                let arg_ty = if !matches!(expected_ty, HostType::Unknown)
                    && (has_unknown(&inferred_ty) || inferred_ty != expected_ty)
                {
                    expected_ty
                } else {
                    inferred_ty
                };
                self.emit_expr_to_var(arg, &arg_name, &arg_ty);
                arg_name
            })
            .collect::<Vec<_>>();
        self.lines.push(format!(
            "{}{target} = {}({});",
            self.indent,
            self.emitted_names
                .get(function)
                .map(String::as_str)
                .unwrap_or(function),
            arg_vars.join(", ")
        ));
    }

    fn assign_adt_construct(
        &mut self,
        target: &str,
        ctor: &str,
        fields: &[HostExpr],
        _ty: &HostType,
    ) {
        // A nullary variant (e.g. `Nothing`, `True`) has no payload fields.
        // ISO C forbids a zero-length array (`chelis_value adt_fields[0];`),
        // so pass a NULL fields pointer with count 0 instead; the runtime
        // helper's `len <= 0` guard never dereferences it (issue #310).
        let fields_arg = if fields.is_empty() {
            "NULL".to_string()
        } else {
            let values_name = self.next_temp("adt_fields");
            self.lines.push(format!(
                "{}chelis_value {}[{}];",
                self.indent,
                values_name,
                fields.len()
            ));
            for (index, field) in fields.iter().enumerate() {
                let field_var = self.next_temp(&format!("adt_field{index}"));
                let field_ty = host_type(field);
                self.emit_expr_to_var(field, &field_var, &field_ty);
                self.lines.push(format!(
                    "{}{}[{index}] = {};",
                    self.indent,
                    values_name,
                    self.box_value_expr(&field_var, &field_ty)
                ));
            }
            values_name
        };
        self.lines.push(format!(
            "{}{target} = chelis_adt_construct(chelis_string_from_cstr({:?}), {}, {});",
            self.indent,
            ctor,
            fields_arg,
            fields.len()
        ));
    }

    fn assign_adt_field_access(
        &mut self,
        target: &str,
        base: &HostExpr,
        field_index: usize,
        ty: &HostType,
    ) {
        let base_var = self.next_temp("adt_base");
        self.emit_expr_to_var(base, &base_var, &host_type(base));
        let value_var = self.next_temp("adt_field");
        self.lines.push(format!(
            "{}chelis_value {} = chelis_adt_get_field({}, {});",
            self.indent, value_var, base_var, field_index
        ));
        self.assign_unboxed_value(target, ty, &value_var);
    }

    fn assign_match_adt(
        &mut self,
        target: &str,
        scrutinee: &HostExpr,
        arms: &[HostMatchArm],
        default_expr: Option<&HostExpr>,
        expr_ty: &HostType,
    ) {
        let scrutinee_var = self.next_temp("adt");
        self.emit_expr_to_var(scrutinee, &scrutinee_var, &host_type(scrutinee));
        let tag_var = self.next_temp("adt_tag");
        self.lines.push(format!(
            "{}chelis_string {} = chelis_adt_get_tag({});",
            self.indent, tag_var, scrutinee_var
        ));
        for (index, arm) in arms.iter().enumerate() {
            let prefix = if index == 0 { "if" } else { "else if" };
            self.lines.push(format!(
                "{}{prefix} (chelis_string_eq({}, chelis_string_from_cstr({:?}))) {{",
                self.indent, tag_var, arm.ctor
            ));
            let nested_indent = format!("{}    ", self.indent);
            let previous = std::mem::replace(&mut self.indent, nested_indent);
            for binding in &arm.bindings {
                let field_var = self.next_temp(&format!("{}_field", binding.name));
                self.lines.push(format!(
                    "{}chelis_value {} = chelis_adt_get_field({}, {});",
                    self.indent, field_var, scrutinee_var, binding.field_index
                ));
                self.lines.push(format!(
                    "{}{} {};",
                    self.indent,
                    c_type(&binding.ty),
                    binding.name
                ));
                self.assign_unboxed_value(&binding.name, &binding.ty, &field_var);
            }
            self.assign_expr(target, &arm.expr, expr_ty);
            self.indent = previous;
            self.lines.push(format!("{}}}", self.indent));
        }
        self.lines.push(format!("{}else {{", self.indent));
        let nested_indent = format!("{}    ", self.indent);
        let previous = std::mem::replace(&mut self.indent, nested_indent);
        if let Some(default_expr) = default_expr {
            self.assign_expr(target, default_expr, expr_ty);
        } else {
            self.lines.push(format!(
                "{}fprintf(stderr, \"non-exhaustive ADT match\\n\");",
                self.indent
            ));
            self.lines.push(format!("{}exit(1);", self.indent));
        }
        self.indent = previous;
        self.lines.push(format!("{}}}", self.indent));
    }

    fn assign_list_literal(&mut self, target: &str, items: &[HostExpr], ty: &HostType) {
        if items.is_empty() {
            self.lines
                .push(format!("{}{target} = chelis_list_empty();", self.indent));
            return;
        }
        let values_name = self.next_temp("list_values");
        self.lines.push(format!(
            "{}chelis_value {}[{}];",
            self.indent,
            values_name,
            items.len()
        ));
        for (index, item) in items.iter().enumerate() {
            let item_var = self.next_temp(&format!("list_item{index}"));
            let inferred_ty = host_type(item);
            let item_ty = if has_unknown(&inferred_ty) {
                match ty {
                    HostType::List(inner) => (**inner).clone(),
                    _ => inferred_ty,
                }
            } else {
                inferred_ty
            };
            self.emit_expr_to_var(item, &item_var, &item_ty);
            self.lines.push(format!(
                "{}{}[{index}] = {};",
                self.indent,
                values_name,
                self.box_value_expr(&item_var, &item_ty)
            ));
        }
        self.lines.push(format!(
            "{}{target} = chelis_list_from_values({}, {});",
            self.indent,
            values_name,
            items.len()
        ));
        // issue #406: a freshly-built list temporary in the program root
        // scope is owned by `main` and must be released at scope exit.
        let list_ty = match ty {
            HostType::List(_) => ty.clone(),
            _ => HostType::List(Box::new(HostType::Unknown)),
        };
        self.track_owned_alloc(target, &list_ty);
    }

    fn assign_tuple_literal(&mut self, target: &str, items: &[HostExpr], ty: &HostType) {
        // An empty tuple has no elements. ISO C forbids a zero-length array
        // (`chelis_value tuple_values[0];`), so pass a NULL items pointer with
        // count 0 instead; the runtime helper's `len <= 0` guard never
        // dereferences it (issue #310).
        let items_arg = if items.is_empty() {
            "NULL".to_string()
        } else {
            let values_name = self.next_temp("tuple_values");
            self.lines.push(format!(
                "{}chelis_value {}[{}];",
                self.indent,
                values_name,
                items.len()
            ));
            for (index, item) in items.iter().enumerate() {
                let item_var = self.next_temp(&format!("tuple_item{index}"));
                let inferred_ty = host_type(item);
                let item_ty = if has_unknown(&inferred_ty) {
                    match ty {
                        HostType::Tuple(item_tys) => {
                            item_tys.get(index).cloned().unwrap_or(inferred_ty)
                        }
                        _ => inferred_ty,
                    }
                } else {
                    inferred_ty
                };
                self.emit_expr_to_var(item, &item_var, &item_ty);
                self.lines.push(format!(
                    "{}{}[{index}] = {};",
                    self.indent,
                    values_name,
                    self.box_value_expr(&item_var, &item_ty)
                ));
            }
            values_name
        };
        self.lines.push(format!(
            "{}{target} = chelis_tuple_from_values({}, {});",
            self.indent,
            items_arg,
            items.len()
        ));
        // issue #406: a freshly-built tuple temporary in the program root
        // scope is owned by `main` and must be released at scope exit.
        let tuple_ty = match ty {
            HostType::Tuple(_) => ty.clone(),
            _ => HostType::Tuple(Vec::new()),
        };
        self.track_owned_alloc(target, &tuple_ty);
    }

    fn assign_map(
        &mut self,
        target: &str,
        callback: &HostCallback,
        list: &HostExpr,
        _ty: &HostType,
    ) {
        let list_var = self.next_temp("map_list");
        self.emit_expr_to_var(list, &list_var, &host_type(list));
        let len_var = self.next_temp("map_len");
        self.lines.push(format!(
            "{}int64_t {} = chelis_list_len({});",
            self.indent, len_var, list_var
        ));
        self.lines
            .push(format!("{}{target} = chelis_list_empty();", self.indent));
        self.lines.push(format!(
            "{}for (int64_t __i = 0; __i < {}; __i++) {{",
            self.indent, len_var
        ));
        let nested_indent = format!("{}    ", self.indent);
        let previous = std::mem::replace(&mut self.indent, nested_indent);
        let item_value = self.next_temp("map_item_value");
        self.lines.push(format!(
            "{}chelis_value {} = chelis_list_index({}, __i);",
            self.indent, item_value, list_var
        ));
        let result_var = self.next_temp("map_result");
        self.lines.push(format!(
            "{}{} {};",
            self.indent,
            c_type(&callback.ret_ty),
            result_var
        ));
        let param = callback_param(callback, 0);
        let arg_var = self.next_temp("map_item");
        self.lines
            .push(format!("{}{} {};", self.indent, c_type(&param.ty), arg_var));
        self.assign_unboxed_value(&arg_var, &param.ty, &item_value);
        self.emit_callback_assign(callback, std::slice::from_ref(&arg_var), &result_var);
        self.lines.push(format!(
            "{}{target} = chelis_list_append({target}, {});",
            self.indent,
            self.box_value_expr(&result_var, &callback.ret_ty)
        ));
        self.indent = previous;
        self.lines.push(format!("{}}}", self.indent));
    }

    fn assign_filter(
        &mut self,
        target: &str,
        callback: &HostCallback,
        list: &HostExpr,
        _ty: &HostType,
    ) {
        let list_var = self.next_temp("filter_list");
        self.emit_expr_to_var(list, &list_var, &host_type(list));
        let len_var = self.next_temp("filter_len");
        self.lines.push(format!(
            "{}int64_t {} = chelis_list_len({});",
            self.indent, len_var, list_var
        ));
        self.lines
            .push(format!("{}{target} = chelis_list_empty();", self.indent));
        self.lines.push(format!(
            "{}for (int64_t __i = 0; __i < {}; __i++) {{",
            self.indent, len_var
        ));
        let nested_indent = format!("{}    ", self.indent);
        let previous = std::mem::replace(&mut self.indent, nested_indent);
        let item_value = self.next_temp("filter_item_value");
        self.lines.push(format!(
            "{}chelis_value {} = chelis_list_index({}, __i);",
            self.indent, item_value, list_var
        ));
        let keep_var = self.next_temp("filter_keep");
        self.lines
            .push(format!("{}bool {};", self.indent, keep_var));
        let param = callback_param(callback, 0);
        let arg_var = self.next_temp("filter_item");
        self.lines
            .push(format!("{}{} {};", self.indent, c_type(&param.ty), arg_var));
        self.assign_unboxed_value(&arg_var, &param.ty, &item_value);
        self.emit_callback_assign(callback, std::slice::from_ref(&arg_var), &keep_var);
        self.lines
            .push(format!("{}if ({}) {{", self.indent, keep_var));
        let nested_indent = format!("{}    ", self.indent);
        let nested_previous = std::mem::replace(&mut self.indent, nested_indent);
        self.lines.push(format!(
            "{}{target} = chelis_list_append({target}, {});",
            self.indent, item_value
        ));
        self.indent = nested_previous;
        self.lines.push(format!("{}}}", self.indent));
        self.indent = previous;
        self.lines.push(format!("{}}}", self.indent));
    }

    fn assign_fold(
        &mut self,
        target: &str,
        callback: &HostCallback,
        init: &HostExpr,
        list: &HostExpr,
        ty: &HostType,
    ) {
        self.assign_expr(target, init, ty);
        let list_var = self.next_temp("fold_list");
        self.emit_expr_to_var(list, &list_var, &host_type(list));
        let len_var = self.next_temp("fold_len");
        self.lines.push(format!(
            "{}int64_t {} = chelis_list_len({});",
            self.indent, len_var, list_var
        ));
        self.lines.push(format!(
            "{}for (int64_t __i = 0; __i < {}; __i++) {{",
            self.indent, len_var
        ));
        let nested_indent = format!("{}    ", self.indent);
        let previous = std::mem::replace(&mut self.indent, nested_indent);
        let item_value = self.next_temp("fold_item_value");
        self.lines.push(format!(
            "{}chelis_value {} = chelis_list_index({}, __i);",
            self.indent, item_value, list_var
        ));
        let params = callback_params(callback);
        let acc_arg = self.next_temp("fold_acc");
        self.lines.push(format!(
            "{}{} {} = {target};",
            self.indent,
            c_type(&params[0].ty),
            acc_arg
        ));
        let item_arg = self.next_temp("fold_item");
        self.lines.push(format!(
            "{}{} {};",
            self.indent,
            c_type(&params[1].ty),
            item_arg
        ));
        self.assign_unboxed_value(&item_arg, &params[1].ty, &item_value);
        self.emit_callback_assign(callback, &[acc_arg, item_arg], target);
        self.indent = previous;
        self.lines.push(format!("{}}}", self.indent));
    }

    fn assign_scan(
        &mut self,
        target: &str,
        callback: &HostCallback,
        init: &HostExpr,
        list: &HostExpr,
        ty: &HostType,
    ) {
        let HostType::List(inner_ty) = ty else {
            self.lines
                .push(format!("{}{target} = chelis_list_empty();", self.indent));
            return;
        };
        let acc_ty = inner_ty.as_ref().clone();
        let acc_var = self.next_temp("scan_acc");
        self.emit_expr_to_var(init, &acc_var, &acc_ty);
        self.lines
            .push(format!("{}{target} = chelis_list_empty();", self.indent));
        let list_var = self.next_temp("scan_list");
        self.emit_expr_to_var(list, &list_var, &host_type(list));
        let len_var = self.next_temp("scan_len");
        self.lines.push(format!(
            "{}int64_t {} = chelis_list_len({});",
            self.indent, len_var, list_var
        ));
        self.lines.push(format!(
            "{}for (int64_t __i = 0; __i < {}; __i++) {{",
            self.indent, len_var
        ));
        let nested_indent = format!("{}    ", self.indent);
        let previous = std::mem::replace(&mut self.indent, nested_indent);
        let item_value = self.next_temp("scan_item_value");
        self.lines.push(format!(
            "{}chelis_value {} = chelis_list_index({}, __i);",
            self.indent, item_value, list_var
        ));
        let params = callback_params(callback);
        let acc_arg = self.next_temp("scan_acc_arg");
        self.lines.push(format!(
            "{}{} {} = {};",
            self.indent,
            c_type(&params[0].ty),
            acc_arg,
            acc_var
        ));
        let item_arg = self.next_temp("scan_item");
        self.lines.push(format!(
            "{}{} {};",
            self.indent,
            c_type(&params[1].ty),
            item_arg
        ));
        self.assign_unboxed_value(&item_arg, &params[1].ty, &item_value);
        self.emit_callback_assign(callback, &[acc_arg, item_arg], &acc_var);
        self.lines.push(format!(
            "{}{target} = chelis_list_append({target}, {});",
            self.indent,
            self.box_value_expr(&acc_var, &acc_ty)
        ));
        self.indent = previous;
        self.lines.push(format!("{}}}", self.indent));
    }

    fn assign_partition(
        &mut self,
        target: &str,
        callback: &HostCallback,
        list: &HostExpr,
        ty: &HostType,
    ) {
        let HostType::Tuple(parts) = ty else {
            self.lines
                .push(format!("{}/* unsupported partition type */", self.indent));
            return;
        };
        let pass_ty = parts.first().cloned().unwrap_or(HostType::Unknown);
        let fail_ty = parts.get(1).cloned().unwrap_or(HostType::Unknown);
        let pass_var = self.next_temp("partition_pass");
        let fail_var = self.next_temp("partition_fail");
        self.lines.push(format!(
            "{}{} {} = chelis_list_empty();",
            self.indent,
            c_type(&pass_ty),
            pass_var
        ));
        self.lines.push(format!(
            "{}{} {} = chelis_list_empty();",
            self.indent,
            c_type(&fail_ty),
            fail_var
        ));
        let list_var = self.next_temp("partition_list");
        self.emit_expr_to_var(list, &list_var, &host_type(list));
        let len_var = self.next_temp("partition_len");
        self.lines.push(format!(
            "{}int64_t {} = chelis_list_len({});",
            self.indent, len_var, list_var
        ));
        self.lines.push(format!(
            "{}for (int64_t __i = 0; __i < {}; __i++) {{",
            self.indent, len_var
        ));
        let nested_indent = format!("{}    ", self.indent);
        let previous = std::mem::replace(&mut self.indent, nested_indent);
        let item_value = self.next_temp("partition_item_value");
        self.lines.push(format!(
            "{}chelis_value {} = chelis_list_index({}, __i);",
            self.indent, item_value, list_var
        ));
        let keep_var = self.next_temp("partition_keep");
        self.lines
            .push(format!("{}bool {};", self.indent, keep_var));
        let param = callback_param(callback, 0);
        let arg_var = self.next_temp("partition_item");
        self.lines
            .push(format!("{}{} {};", self.indent, c_type(&param.ty), arg_var));
        self.assign_unboxed_value(&arg_var, &param.ty, &item_value);
        self.emit_callback_assign(callback, std::slice::from_ref(&arg_var), &keep_var);
        self.lines
            .push(format!("{}if ({}) {{", self.indent, keep_var));
        let then_indent = format!("{}    ", self.indent);
        let then_previous = std::mem::replace(&mut self.indent, then_indent);
        self.lines.push(format!(
            "{}{} = chelis_list_append({}, {});",
            self.indent, pass_var, pass_var, item_value
        ));
        self.indent = then_previous;
        self.lines.push(format!("{}}} else {{", self.indent));
        let else_indent = format!("{}    ", self.indent);
        let else_previous = std::mem::replace(&mut self.indent, else_indent);
        self.lines.push(format!(
            "{}{} = chelis_list_append({}, {});",
            self.indent, fail_var, fail_var, item_value
        ));
        self.indent = else_previous;
        self.lines.push(format!("{}}}", self.indent));
        self.indent = previous;
        self.lines.push(format!("{}}}", self.indent));
        let tuple_values = self.next_temp("partition_values");
        self.lines
            .push(format!("{}chelis_value {}[2];", self.indent, tuple_values));
        self.lines.push(format!(
            "{}{}[0] = chelis_value_from_list({});",
            self.indent, tuple_values, pass_var
        ));
        self.lines.push(format!(
            "{}{}[1] = chelis_value_from_list({});",
            self.indent, tuple_values, fail_var
        ));
        self.lines.push(format!(
            "{}{target} = chelis_tuple_from_values({}, 2);",
            self.indent, tuple_values
        ));
    }

    fn assign_flat_map(
        &mut self,
        target: &str,
        callback: &HostCallback,
        list: &HostExpr,
        _ty: &HostType,
    ) {
        let list_var = self.next_temp("flat_map_list");
        self.emit_expr_to_var(list, &list_var, &host_type(list));
        let len_var = self.next_temp("flat_map_len");
        self.lines.push(format!(
            "{}int64_t {} = chelis_list_len({});",
            self.indent, len_var, list_var
        ));
        self.lines
            .push(format!("{}{target} = chelis_list_empty();", self.indent));
        self.lines.push(format!(
            "{}for (int64_t __i = 0; __i < {}; __i++) {{",
            self.indent, len_var
        ));
        let nested_indent = format!("{}    ", self.indent);
        let previous = std::mem::replace(&mut self.indent, nested_indent);
        let item_value = self.next_temp("flat_map_item_value");
        self.lines.push(format!(
            "{}chelis_value {} = chelis_list_index({}, __i);",
            self.indent, item_value, list_var
        ));
        let result_var = self.next_temp("flat_map_result");
        self.lines.push(format!(
            "{}{} {};",
            self.indent,
            c_type(&callback.ret_ty),
            result_var
        ));
        let param = callback_param(callback, 0);
        let arg_var = self.next_temp("flat_map_item");
        self.lines
            .push(format!("{}{} {};", self.indent, c_type(&param.ty), arg_var));
        self.assign_unboxed_value(&arg_var, &param.ty, &item_value);
        self.emit_callback_assign(callback, std::slice::from_ref(&arg_var), &result_var);
        self.lines.push(format!(
            "{}{target} = chelis_list_concat({target}, {});",
            self.indent, result_var
        ));
        self.indent = previous;
        self.lines.push(format!("{}}}", self.indent));
    }

    fn emit_callback_assign(&mut self, callback: &HostCallback, arg_vars: &[String], target: &str) {
        match &callback.kind {
            HostCallbackKind::Named { function, .. } => {
                self.lines.push(format!(
                    "{}{target} = {}({});",
                    self.indent,
                    function,
                    arg_vars.join(", ")
                ));
            }
            HostCallbackKind::Inline { params, body } => {
                for (param, arg_var) in params.iter().zip(arg_vars.iter()) {
                    self.lines.push(format!(
                        "{}{} {} = {};",
                        self.indent,
                        c_type(&param.ty),
                        param.name,
                        arg_var
                    ));
                }
                self.assign_expr(target, body, &callback.ret_ty);
            }
        }
    }

    fn box_value_expr(&self, value: &str, ty: &HostType) -> String {
        match ty {
            HostType::Int64 => format!("chelis_value_from_int64({value})"),
            HostType::Float64 => format!("chelis_value_from_f64({value})"),
            HostType::Bool => format!("chelis_value_from_bool({value})"),
            HostType::String => format!("chelis_value_from_string({value})"),
            HostType::Adt(_, _) => format!("chelis_value_from_adt({value})"),
            HostType::Tensor(_) => format!("chelis_value_from_tensor({value})"),
            HostType::List(_) => format!("chelis_value_from_list({value})"),
            HostType::Tuple(_) => format!("chelis_value_from_tuple({value})"),
            HostType::Dict(_, _) => format!("chelis_value_from_dict({value})"),
            _ => "chelis_value_from_int64(0)".to_string(),
        }
    }

    fn assign_unboxed_value(&mut self, target: &str, ty: &HostType, value_expr: &str) {
        let expr = match ty {
            HostType::Int64 => format!("chelis_value_as_int64({value_expr})"),
            HostType::Float64 => format!("chelis_value_as_f64({value_expr})"),
            HostType::Bool => format!("chelis_value_as_bool({value_expr})"),
            HostType::String => format!("chelis_value_as_string({value_expr})"),
            HostType::Adt(_, _) => format!("chelis_value_as_adt({value_expr})"),
            HostType::Tensor(_) => format!("chelis_value_as_tensor({value_expr})"),
            HostType::List(_) => format!("chelis_value_as_list({value_expr})"),
            HostType::Tuple(_) => format!("chelis_value_as_tuple({value_expr})"),
            HostType::Dict(_, _) => format!("chelis_value_as_dict({value_expr})"),
            _ => "0".to_string(),
        };
        self.lines
            .push(format!("{}{target} = {expr};", self.indent));
    }

    fn emit_print_value(&mut self, value: &str, ty: &HostType) {
        match ty {
            HostType::String => self.lines.push(format!(
                "{}printf(\"%s\\n\", chelis_string_data({}));",
                self.indent, value
            )),
            HostType::Int64 => self.lines.push(format!(
                "{}printf(\"%lld\\n\", (long long){});",
                self.indent, value
            )),
            HostType::Float64 => self
                .lines
                .push(format!("{}printf(\"%.16g\\n\", {});", self.indent, value)),
            HostType::Bool => self.lines.push(format!(
                "{}printf(\"%s\\n\", {} ? \"true\" : \"false\");",
                self.indent, value
            )),
            HostType::Tensor(_) => self.lines.push(format!(
                "{}chelis_print_tensor_stdout({}); printf(\"\\n\");",
                self.indent, value
            )),
            HostType::Adt(_, _) => self.lines.push(format!(
                "{}chelis_print_adt({}); printf(\"\\n\");",
                self.indent, value
            )),
            HostType::List(_) => self.lines.push(format!(
                "{}chelis_print_list({}); printf(\"\\n\");",
                self.indent, value
            )),
            HostType::Tuple(_) => self.lines.push(format!(
                "{}chelis_print_tuple({}); printf(\"\\n\");",
                self.indent, value
            )),
            HostType::Dict(_, _) => self.lines.push(format!(
                "{}chelis_print_dict({}); printf(\"\\n\");",
                self.indent, value
            )),
            HostType::Unit => self
                .lines
                .push(format!("{}printf(\"()\\n\");", self.indent)),
            _ => self
                .lines
                .push(format!("{}printf(\"<value>\\n\");", self.indent)),
        }
    }

    fn emit_labeled_root(&mut self, name: &str, value: &str, ty: &HostType) {
        // Mirror eval's tuple-root expansion: a `Tuple([T0, T1, ...])`
        // top-level binding renders as `<name>.0 = ...`, `<name>.1 = ...`
        // (one labeled line per field). Eval produces this via
        // `extend_root_names_from_value`; the C backend reaches it here.
        if let HostType::Tuple(items) = ty {
            for (index, field_ty) in items.iter().enumerate() {
                let field_name = format!("{name}.{index}");
                let field_var = self.next_temp(&format!("root_field{index}"));
                self.lines.push(format!(
                    "{}chelis_value {} = chelis_tuple_get({}, {index});",
                    self.indent, field_var, value
                ));
                let field_value = self.next_temp(&format!("root_field_val{index}"));
                self.lines.push(format!(
                    "{}{};",
                    self.indent,
                    c_decl(field_ty, &field_value)
                ));
                self.assign_unboxed_value(&field_value, field_ty, &field_var);
                self.emit_labeled_root(&field_name, &field_value, field_ty);
                // issue #406: `chelis_tuple_get` retains the boxed element
                // it returns (a no-op for scalar fields). The labeled-root
                // printer only reads it, so release the retained handle once
                // the field has been printed — otherwise a tuple field that
                // is itself a heap container (a nested tuple/list/adt) leaks
                // that reference for the process lifetime.
                self.lines
                    .push(format!("{}chelis_value_release({field_var});", self.indent));
            }
            return;
        }
        // `name` is producer-supplied (HostProgram binding display_name).
        // It lands inside a `"..."` C string literal as a `printf %s`
        // RUNTIME argument. Even though %s substitution is itself safe
        // (the runtime never reinterprets the data as a format), the
        // SURROUNDING C string literal must lex correctly. Rust's `{:?}`
        // emits `\u{XX}` for forbidden bytes, which is NOT valid C —
        // route through the format-string sanitizer (which emits
        // C-compatible `\xNN`/`\\`/`\"` escapes) per
        // spec/upstream-bugs/producer-string-sanitization.md.
        let safe_name = chelis_ir::span_sanitize::sanitize_for_format_string(name);
        self.lines.push(format!(
            "{}printf(\"%s = \", \"{safe_name}\");",
            self.indent
        ));
        match ty {
            HostType::String => self.lines.push(format!(
                "{}printf(\"%s\", chelis_string_data({}));",
                self.indent, value
            )),
            HostType::Int64 => self.lines.push(format!(
                "{}printf(\"%lld\", (long long){});",
                self.indent, value
            )),
            HostType::Float64 => self
                .lines
                .push(format!("{}printf(\"%.16g\", {});", self.indent, value)),
            HostType::Bool => self.lines.push(format!(
                "{}printf(\"%s\", {} ? \"true\" : \"false\");",
                self.indent, value
            )),
            HostType::Tensor(_) => self.lines.push(format!(
                "{}chelis_print_tensor_stdout({});",
                self.indent, value
            )),
            HostType::Adt(_, _) => self
                .lines
                .push(format!("{}chelis_print_adt({});", self.indent, value)),
            HostType::List(_) => self
                .lines
                .push(format!("{}chelis_print_list({});", self.indent, value)),
            HostType::Tuple(_) => self
                .lines
                .push(format!("{}chelis_print_tuple({});", self.indent, value)),
            HostType::Dict(_, _) => self
                .lines
                .push(format!("{}chelis_print_dict({});", self.indent, value)),
            HostType::Unit => self.lines.push(format!("{}printf(\"()\");", self.indent)),
            _ => self
                .lines
                .push(format!("{}printf(\"<value>\");", self.indent)),
        }
        self.lines.push(format!("{}printf(\"\\n\");", self.indent));
    }

    fn next_temp(&mut self, prefix: &str) -> String {
        let name = format!("__{prefix}_{}", self.temp_counter);
        self.temp_counter += 1;
        name
    }

    fn assign_option_some(
        &mut self,
        target: &str,
        ty: &HostType,
        value_var: &str,
        value_ty: &HostType,
    ) {
        match ty {
            HostType::Option(inner) if matches!(inner.as_ref(), HostType::Int64) => {
                self.lines
                    .push(format!("{}{target}.is_some = true;", self.indent));
                self.lines
                    .push(format!("{}{target}.value = {value_var};", self.indent));
            }
            HostType::Option(inner) if matches!(inner.as_ref(), HostType::Float64) => {
                self.lines
                    .push(format!("{}{target}.is_some = true;", self.indent));
                self.lines
                    .push(format!("{}{target}.value = {value_var};", self.indent));
            }
            HostType::Option(_) => {
                self.lines
                    .push(format!("{}{target}.is_some = true;", self.indent));
                self.lines.push(format!(
                    "{}{target}.value = {};",
                    self.indent,
                    self.box_value_expr(value_var, value_ty)
                ));
            }
            _ => self
                .lines
                .push(format!("{}{target} = {value_var};", self.indent)),
        }
    }

    fn assign_option_none(&mut self, target: &str, ty: &HostType) {
        match ty {
            HostType::Option(inner) if matches!(inner.as_ref(), HostType::Int64) => {
                self.lines
                    .push(format!("{}{target}.is_some = false;", self.indent));
                self.lines
                    .push(format!("{}{target}.value = 0;", self.indent));
            }
            HostType::Option(inner) if matches!(inner.as_ref(), HostType::Float64) => {
                self.lines
                    .push(format!("{}{target}.is_some = false;", self.indent));
                self.lines
                    .push(format!("{}{target}.value = 0.0;", self.indent));
            }
            HostType::Option(_) => {
                self.lines
                    .push(format!("{}{target}.is_some = false;", self.indent));
                self.lines.push(format!(
                    "{}{target}.value = chelis_value_from_int64(0);",
                    self.indent
                ));
            }
            _ => self.lines.push(format!("{}{target} = 0;", self.indent)),
        }
    }
}

/// The runtime release call that frees the heap allocation a value of
/// `ty` held in `var` owns, or `None` for non-owning types (scalars,
/// borrowed views, function pointers). Used by `emit_main`'s scope-exit
/// cleanup (issue #406) so a `chelis build --target c` program frees the
/// list / tensor / tuple / dict / adt / string temporaries it allocates
/// instead of leaking them for the process lifetime. The runtime release
/// functions are refcounted, so releasing a container correctly
/// decrements any retained element without double-freeing it.
fn release_call(var: &str, ty: &HostType) -> Option<String> {
    match ty {
        HostType::Tensor(_) => Some(format!("chelis_free({var});")),
        HostType::List(_) => Some(format!("chelis_list_release({var});")),
        HostType::Tuple(_) => Some(format!("chelis_tuple_release({var});")),
        HostType::Dict(_, _) => Some(format!("chelis_dict_release({var});")),
        HostType::Adt(_, _) => Some(format!("chelis_adt_release({var});")),
        HostType::String => Some(format!("chelis_string_release({var});")),
        // Scalars (int/float/bool/unit), borrowed mapped files, function
        // pointers, and Option-of-scalar carry no owned heap allocation
        // for `main` to free. `Option` of a pointer type and `Unknown`
        // are deliberately not auto-freed here: their concrete ownership
        // is not recoverable from the host type alone, so freeing them
        // blindly would risk a double-free. They remain process-lifetime
        // until a future change threads precise ownership.
        _ => None,
    }
}

/// The runtime retain call that adds one reference to the heap allocation
/// a value of `ty` held in `var` owns, or `None` for non-refcounted types.
/// Used by the `let`-block scope release (issue #406): when a block result
/// is transferred to a `target` in the outer scope via a bare pointer copy
/// (`target = <heap var>` for a `Var` body or an `if`/`match` arm), the
/// target shares the source allocation's reference. Retaining at the
/// transfer leaf lets the block uniformly release every heap binding it
/// declared without freeing the value the caller now holds.
///
/// Only the refcounted host-value types are retainable. `Tensor` is
/// excluded deliberately: it is freed by the unconditional `chelis_free`,
/// has no refcount retain, and tensor locals are already cleaned up by the
/// tensor-helper lane — so tensor `let` bindings are never tracked for
/// block release in the first place (see `binding_release`).
fn retain_call(var: &str, ty: &HostType) -> Option<String> {
    match ty {
        HostType::List(_) => Some(format!("chelis_list_retain({var});")),
        HostType::Tuple(_) => Some(format!("chelis_tuple_retain({var});")),
        HostType::Dict(_, _) => Some(format!("chelis_dict_retain({var});")),
        HostType::Adt(_, _) => Some(format!("chelis_adt_retain({var});")),
        HostType::String => Some(format!("chelis_string_retain({var});")),
        _ => None,
    }
}

/// The release call for a `let` binding tracked by the block-scope cleanup
/// (issue #406), or `None` if the binding's type is not a refcounted
/// host-value (so it owns no heap allocation the block must reclaim, or it
/// is a `Tensor` reclaimed by the tensor-helper lane and has no matching
/// `retain_call` to pair the transfer-leaf retain against). Restricting
/// the tracked set to exactly the `retain_call` types keeps every
/// retain/release balanced regardless of how the block result is produced.
fn binding_release(var: &str, ty: &HostType) -> Option<String> {
    retain_call(var, ty)?;
    release_call(var, ty)
}

fn c_type(ty: &HostType) -> &'static str {
    match ty {
        HostType::Int64 => "int64_t",
        HostType::Float64 => "double",
        HostType::Bool => "bool",
        HostType::String => "chelis_string",
        HostType::Fn(_, _) => "void*",
        HostType::Adt(_, _) => "chelis_adt*",
        HostType::List(_) => "chelis_list*",
        HostType::Dict(_, _) => "chelis_dict*",
        HostType::Tuple(_) => "chelis_tuple*",
        HostType::Tensor(_) => "chelis_tensor*",
        HostType::MappedFile => "chelis_mapped_file*",
        HostType::Option(inner) => match inner.as_ref() {
            HostType::Int64 => "chelis_option_i64",
            HostType::Float64 => "chelis_option_f64",
            _ => "chelis_option_value",
        },
        HostType::Unit => "int",
        // Unresolved polymorphic type — use `void*` so callers passing
        // concrete pointer types (chelis_adt*, chelis_tensor*, etc.)
        // implicitly convert cleanly. Scalars (int64/f64/bool) require
        // explicit boxing at the callsite; Coral's HAMT / Nautilus's
        // `a`-valued defs pass only pointer types in practice.
        HostType::Unknown => "void*",
    }
}

fn c_decl(ty: &HostType, name: &str) -> String {
    match ty {
        HostType::Fn(params, ret) => {
            let args = if params.is_empty() {
                "void".to_string()
            } else {
                params.iter().map(c_type).collect::<Vec<_>>().join(", ")
            };
            format!("{} (*{})({})", c_type(ret), name, args)
        }
        _ => format!("{} {}", c_type(ty), name),
    }
}

fn host_type(expr: &HostExpr) -> HostType {
    match &expr.kind {
        HostExprKind::Int(_) => HostType::Int64,
        HostExprKind::Float(_) => HostType::Float64,
        HostExprKind::Bool(_) => HostType::Bool,
        HostExprKind::String(_) => HostType::String,
        HostExprKind::List(_, ty) => ty.clone(),
        HostExprKind::Tuple(_, ty) => ty.clone(),
        HostExprKind::Var(_, ty)
        | HostExprKind::Call { ty, .. }
        | HostExprKind::Builtin { ty, .. }
        | HostExprKind::AdtConstruct { ty, .. }
        | HostExprKind::AdtFieldAccess { ty, .. }
        | HostExprKind::If { ty, .. }
        | HostExprKind::MatchOption { ty, .. }
        | HostExprKind::MatchAdt { ty, .. }
        | HostExprKind::Let { ty, .. }
        | HostExprKind::Map { ty, .. }
        | HostExprKind::Filter { ty, .. }
        | HostExprKind::Fold { ty, .. }
        | HostExprKind::Scan { ty, .. }
        | HostExprKind::Partition { ty, .. }
        | HostExprKind::FlatMap { ty, .. }
        | HostExprKind::WithSeed { ty, .. }
        | HostExprKind::TensorCall { ty, .. } => ty.clone(),
        HostExprKind::Unit => HostType::Unit,
    }
}

fn option_inner_type(ty: HostType) -> HostType {
    match ty {
        HostType::Option(inner) => *inner,
        _ => HostType::Unknown,
    }
}

fn expected_builtin_arg_ty(name: &str, ty: &HostType, index: usize) -> HostType {
    match (name, ty, index) {
        ("Some", HostType::Option(inner), 0) => (**inner).clone(),
        ("dict_of", HostType::Dict(key, value), 0) => {
            HostType::List(Box::new(HostType::Tuple(vec![
                (**key).clone(),
                (**value).clone(),
            ])))
        }
        ("append", HostType::List(inner), 1) => (**inner).clone(),
        _ => HostType::Unknown,
    }
}

fn has_unknown(ty: &HostType) -> bool {
    match ty {
        HostType::Unknown => true,
        HostType::Fn(params, ret) => params.iter().any(has_unknown) || has_unknown(ret),
        HostType::List(inner) | HostType::Option(inner) => has_unknown(inner),
        HostType::Dict(key, value) => has_unknown(key) || has_unknown(value),
        HostType::Tuple(items) => items.iter().any(has_unknown),
        _ => false,
    }
}

fn callback_params(callback: &HostCallback) -> &[HostParam] {
    match &callback.kind {
        HostCallbackKind::Named { params, .. } | HostCallbackKind::Inline { params, .. } => params,
    }
}

fn callback_param(callback: &HostCallback, index: usize) -> &HostParam {
    callback_params(callback)
        .get(index)
        .expect("callback parameter should exist")
}

/// C element type for a `Prim` tensor payload, mirroring
/// `CEmitter::elem_type`. Kept as a module-level helper here so the
/// summary-derived sparse emission path can reuse the same dtype
/// table without depending on `CEmitter`'s `self`.
///
/// WS-1: the prior fallthrough default-arm silently downgraded
/// f64/i8/i16/bf16/f16 sparse payloads to single-precision storage,
/// which is exactly the destructure-default footgun the WS-A0 F1
/// guard targets. Replaced with explicit per-dtype arms plus a
/// panic on truly unsupported dtypes so a future dtype lift cannot
/// fall through to a quiet wrong-width read.
fn sparse_elem_type(prim: Prim) -> &'static str {
    match prim {
        Prim::F32 | Prim::Bool => "float",
        Prim::F64 => "double",
        Prim::Int8 => "int8_t",
        Prim::Int16 => "int16_t",
        Prim::Int32 => "int32_t",
        Prim::Int64 => "int64_t",
        // bf16/f16 sparse payloads share the `uint16_t` storage
        // contract with the dense path; per-element gather/scatter
        // moves the 2-byte slot verbatim. The HIP / Metal backends
        // route through their own bf16/f16 paths and do not depend
        // on this helper.
        Prim::Bf16 | Prim::F16 => "uint16_t",
        other => panic!(
            "C backend sparse path has no element type for `{}` (spec/04-type-system.md §1.1)",
            other.name()
        ),
    }
}

/// `CHELIS_<DTYPE>` macro selector for a `Prim`. Mirrors
/// `CEmitter::dtype_macro`. Used by the summary-derived sparse path
/// for both output allocation and contract assertions.
fn sparse_dtype_macro(prim: Prim) -> &'static str {
    match prim {
        Prim::F32 => "CHELIS_F32",
        Prim::F64 => "CHELIS_F64",
        Prim::Bool => "CHELIS_BOOL",
        Prim::Int8 => "CHELIS_I8",
        Prim::Int16 => "CHELIS_I16",
        Prim::Int32 => "CHELIS_I32",
        Prim::Int64 => "CHELIS_I64",
        // WS-1: bf16 / f16 routed via the runtime's matching tags.
        Prim::Bf16 => "CHELIS_BF16",
        Prim::F16 => "CHELIS_F16",
        other => panic!(
            "C backend sparse summary does not support {} tensors",
            other.name()
        ),
    }
}

/// One arm of the runtime-dtype dispatch emitted by the elementwise
/// host-emit helpers (`assign_tensor_*_elementwise`).  Each arm
/// names a `CHELIS_*` constant and the C element type used to read
/// and write the tensor's `data` buffer through a typed pointer.
///
/// The `Int32` and `Bool` arms reuse `float` as the element type
/// because CHELIS_I32 and CHELIS_BOOL tensor storage today is
/// 4-byte f32-encoded (see `crates/chelis-runtime/src/lib.rs`
/// `chelis_alloc` and the f32-routed runtime accessors at
/// L2178-L2192 / L2222-L2223 plus
/// `docs/investigations/c_runtime_dtype_accessors_diagnosis.md`).
/// `i32::data_ptr_unchecked` exists for future storage migration
/// but reading the current f32-encoded buffer through it would be
/// the wrong decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DtypeArm {
    F32,
    F64,
    I32,
    I64,
    Bool,
}

impl DtypeArm {
    fn dtype_macro(self) -> &'static str {
        match self {
            DtypeArm::F32 => "CHELIS_F32",
            DtypeArm::F64 => "CHELIS_F64",
            DtypeArm::I32 => "CHELIS_I32",
            DtypeArm::I64 => "CHELIS_I64",
            DtypeArm::Bool => "CHELIS_BOOL",
        }
    }

    fn elem_t(self) -> &'static str {
        match self {
            // f32 / int32 / bool storage is 4-byte f32-encoded
            // today.  Reading through `float*` matches the runtime
            // accessor convention; see the type doc-comment above.
            DtypeArm::F32 | DtypeArm::I32 | DtypeArm::Bool => "float",
            DtypeArm::F64 => "double",
            DtypeArm::I64 => "int64_t",
        }
    }

    /// Every supported precision the operator-form elementwise
    /// helpers emit a typed arm for.  Maps the runtime's currently-
    /// allocated dtypes onto the four C element types used by the
    /// runtime accessor pattern.
    fn all_operator_arms() -> &'static [DtypeArm] {
        &[
            DtypeArm::F32,
            DtypeArm::F64,
            DtypeArm::I32,
            DtypeArm::I64,
            DtypeArm::Bool,
        ]
    }

    /// Subset of the operator arms covered by the libm-f32 func
    /// form (`expf`, `sinf`, `fmaxf`, `chelis_host_relu_f32`, ...).
    /// CHELIS_F64 and CHELIS_I64 cannot be covered by these names
    /// without precision loss or type-mismatch; those arms emit a
    /// `runtime_fail`-style `abort()` until per-precision helper
    /// names land in a future PR.
    fn all_f32_only_func_arms() -> &'static [DtypeArm] {
        &[DtypeArm::F32, DtypeArm::I32, DtypeArm::Bool]
    }
}

/// Resolve a `DimInfo` to a C expression usable inside the summary's
/// inline emission. Concrete literals and named-with-binding dims
/// emit their literal value. Pure symbolic dims (`Named(name, None)`)
/// resolve to `arg->shape[axis]` against the first summary input that
/// carries the same symbol.
fn sparse_dim_info_expr(
    dim: &DimInfo,
    summary: &HostSparseOpSummary,
    tensor_args: &[String],
) -> String {
    match dim {
        DimInfo::Lit(n) => n.to_string(),
        DimInfo::Named(_, Some(n)) => n.to_string(),
        DimInfo::Named(name, None) => sparse_symbol_expr(name, summary, tensor_args)
            .unwrap_or_else(|| panic!("sparse summary symbol `{name}` has no input binding")),
    }
}

fn sparse_dim_product(
    dims: &[DimInfo],
    summary: &HostSparseOpSummary,
    tensor_args: &[String],
) -> String {
    dims.iter()
        .map(|d| sparse_dim_info_expr(d, summary, tensor_args))
        .reduce(|lhs, rhs| format!("({lhs} * {rhs})"))
        .unwrap_or_else(|| "1".to_string())
}

fn sparse_symbol_expr(
    name: &str,
    summary: &HostSparseOpSummary,
    tensor_args: &[String],
) -> Option<String> {
    summary
        .input_tys
        .iter()
        .zip(tensor_args.iter())
        .find_map(|(ty, arg)| {
            ty.dims.iter().enumerate().find_map(|(axis, dim)| {
                matches!(dim, DimInfo::Named(dim_name, None) if dim_name == name)
                    .then(|| format!("{arg}->shape[{axis}]"))
            })
        })
}
