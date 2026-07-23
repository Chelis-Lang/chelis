use chelis_ir::host::{
    ConcreteHostProgram, HostBlasMatmulSummary, HostFunctionSpecialization, HostSparseOpSummary,
    HostTensorHelper, HostTensorSpecialization,
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

/// Closed identity for the scalar/string C-expression dispatch table.
///
/// The source builtin name is open text, so it must cross this fallible
/// boundary before the exhaustive expression match. An unknown name has no
/// enum value and therefore cannot reach an `EmittedExpr` constructor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CExpressionBuiltin {
    Add,
    Sub,
    Mul,
    Div,
    TruncDiv,
    FloorDiv,
    Mod,
    CompareLess,
    Less,
    Greater,
    GreaterEqual,
    LessEqual,
    Equal,
    NotEqual,
    And,
    Or,
    Not,
    Neg,
    StringConcat,
    StringTrim,
    Reshape,
    StringSlice,
    StringContains,
    StringStartsWith,
    StringEndsWith,
    StringLen,
    ToString,
    ToInt,
    ToFloat,
    TensorToScalar,
    ScalarToTensor,
    Len,
    Range,
    Rank,
    Shape,
    Numel,
    Sqrt,
    Exp,
    Log,
    Sin,
    Cos,
    Tanh,
    Pow,
    Abs,
    Min,
    Max,
}

impl CExpressionBuiltin {
    fn decode(name: &str) -> Result<Self, Unsupported> {
        Ok(match name {
            "add" => Self::Add,
            "sub" => Self::Sub,
            "mul" => Self::Mul,
            "div" => Self::Div,
            "trunc_div" => Self::TruncDiv,
            "floor_div" => Self::FloorDiv,
            "mod" => Self::Mod,
            "cmplt" => Self::CompareLess,
            "lt" => Self::Less,
            "gt" => Self::Greater,
            "gte" => Self::GreaterEqual,
            "lte" => Self::LessEqual,
            "eq" => Self::Equal,
            "neq" => Self::NotEqual,
            "and" => Self::And,
            "or" => Self::Or,
            "not" => Self::Not,
            "neg" => Self::Neg,
            "string_concat" => Self::StringConcat,
            "string_trim" => Self::StringTrim,
            "reshape" => Self::Reshape,
            "string_slice" => Self::StringSlice,
            "string_contains" => Self::StringContains,
            "string_starts_with" => Self::StringStartsWith,
            "string_ends_with" => Self::StringEndsWith,
            "string_len" => Self::StringLen,
            "to_string" => Self::ToString,
            "to_int" => Self::ToInt,
            "to_float" => Self::ToFloat,
            "tensor_to_scalar" => Self::TensorToScalar,
            "scalar_to_tensor" => Self::ScalarToTensor,
            "len" => Self::Len,
            "range" => Self::Range,
            "rank" => Self::Rank,
            "shape" => Self::Shape,
            "numel" => Self::Numel,
            "sqrt" => Self::Sqrt,
            "exp" => Self::Exp,
            "log" => Self::Log,
            "sin" => Self::Sin,
            "cos" => Self::Cos,
            "tanh" => Self::Tanh,
            "pow" => Self::Pow,
            "abs" => Self::Abs,
            "min" => Self::Min,
            "max" => Self::Max,
            other => {
                return Err(Unsupported::new(
                    UnsupportedKind::Builtin(other.to_string()),
                    "`chelis build` host emission",
                    Stage::Codegen("c"),
                    "this builtin has no compiled-lane expression identity; the eval lane \
                     may support it (`chelis eval`). Silent-stub class: chelis#703; \
                     instances chelis#682/#704/#705/#715 ([05-UNS-1])",
                ));
            }
        })
    }
}

use crate::emit::CEmitter;
use crate::emitted_expr::{BinaryOperator, EmittedExpr, UnaryOperator};
use crate::host_abi::{
    HostAbiCallback as HostCallback, HostAbiCallbackKind as HostCallbackKind,
    HostAbiExpr as HostExpr, HostAbiExprKind as HostExprKind, HostAbiFunction as HostFunction,
    HostAbiMatchArm as HostMatchArm, HostAbiParam as HostParam, HostAbiProgram as HostProgram,
    HostAbiType, HostAbiType as HostType, project_program,
};
use chelis_ir::dag::{DimExpr, DimInfo, RiscOp, TensorType};
use chelis_types::types::Prim;
use chelis_types::unsupported::{Stage, Unsupported, UnsupportedKind};
use std::collections::{HashMap, HashSet};

/// The set of parameter indices a user function's result may alias
/// (issue #406 call-escape interprocedural summary). `Indices(s)` means
/// the result may *be* the allocation of one of the parameters in `s`
/// (and only those); an empty set means the result is always a fresh
/// allocation that does not escape any argument. `Any` is the
/// conservative top element: the result may alias *any* refcounted
/// pointer-typed argument. `Any` is used whenever the body contains a
/// shape the analysis does not precisely model (e.g. a call to a
/// function not yet in the summary, an unmodeled `HostExprKind`), so the
/// emit site over-retains rather than risking a use-after-free.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ReturnsArg {
    Indices(HashSet<usize>),
    Any,
}

impl ReturnsArg {
    fn empty() -> Self {
        ReturnsArg::Indices(HashSet::new())
    }

    /// Join two result-alias summaries (the `if`/`match`-arm union or the
    /// fixpoint widening). `Any` absorbs everything; otherwise the index
    /// sets are unioned.
    fn join(self, other: ReturnsArg) -> ReturnsArg {
        match (self, other) {
            (ReturnsArg::Any, _) | (_, ReturnsArg::Any) => ReturnsArg::Any,
            (ReturnsArg::Indices(mut a), ReturnsArg::Indices(b)) => {
                a.extend(b);
                ReturnsArg::Indices(a)
            }
        }
    }

    /// Does the result possibly alias parameter index `i`?
    fn may_return(&self, i: usize) -> bool {
        match self {
            ReturnsArg::Any => true,
            ReturnsArg::Indices(s) => s.contains(&i),
        }
    }
}

/// Compute, for every user function in the program, the set of parameter
/// indices its result may alias (issue #406 call-escape). Reaches a
/// least-fixpoint over the call graph so mutual recursion is handled: a
/// function that returns the result of calling another (or itself)
/// propagates that callee's parameter-alias set back through the matching
/// argument positions.
///
/// Soundness contract: the result for a function is only ever *widened*
/// across iterations, and any expression shape the walker does not
/// precisely model yields [`ReturnsArg::Any`] (top), so the summary is a
/// sound over-approximation of "may the result be this parameter's
/// allocation". The emit site uses it to decide which block-frame heap
/// bindings escape through a call and must be retained; an over-estimate
/// retains a binding that did not actually escape (a documented residual
/// leak), never frees one that did (which would be a use-after-free).
fn analyze_returns_arg(program: &HostProgram) -> HashMap<String, ReturnsArg> {
    let mut summary: HashMap<String, ReturnsArg> = program
        .functions
        .iter()
        .map(|f| (f.name.clone(), ReturnsArg::empty()))
        .collect();

    // Monotone fixpoint: re-evaluate each body until no summary widens.
    // Bounded by (function count x parameter count) widenings.
    loop {
        let mut changed = false;
        for function in &program.functions {
            let param_index: HashMap<&str, usize> = function
                .params
                .iter()
                .enumerate()
                .map(|(i, p)| (p.name.as_str(), i))
                .collect();
            let mut env: HashMap<String, ReturnsArg> = HashMap::new();
            let computed = result_alias_set(&function.body, &param_index, &summary, &mut env);
            let entry = summary
                .entry(function.name.clone())
                .or_insert_with(ReturnsArg::empty);
            let joined = entry.clone().join(computed);
            if &joined != entry {
                *entry = joined;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    summary
}

/// Evaluate the parameter-alias set of `expr`'s *result value* under the
/// current call-graph `summary` and the local `env` mapping in-scope
/// `let`-binding names to their own alias sets. `param_index` maps the
/// enclosing function's parameter names to their positions.
///
/// The result is the set of enclosing-function parameter indices the
/// value may alias, or [`ReturnsArg::Any`] when the value may alias an
/// argument the analysis cannot pin to a specific parameter (a call to a
/// not-yet-summarized or opaque function whose returned argument is
/// itself parameter-derived, etc.). Constructors, literals, builtins, and
/// tensor lanes produce fresh allocations and contribute the empty set.
fn result_alias_set(
    expr: &HostExpr,
    param_index: &HashMap<&str, usize>,
    summary: &HashMap<String, ReturnsArg>,
    env: &mut HashMap<String, ReturnsArg>,
) -> ReturnsArg {
    match &expr.kind {
        HostExprKind::Var(name, _) => {
            if let Some(&i) = param_index.get(name.as_str()) {
                ReturnsArg::Indices(HashSet::from([i]))
            } else if let Some(set) = env.get(name) {
                set.clone()
            } else {
                // An outer-scope / global name: not one of this
                // function's parameters, so it does not alias any
                // parameter. (A captured global is owned elsewhere.)
                ReturnsArg::empty()
            }
        }
        HostExprKind::Let { bindings, body, .. } => {
            for binding in bindings {
                let set = result_alias_set(&binding.value, param_index, summary, env);
                env.insert(binding.name.clone(), set);
            }
            result_alias_set(body, param_index, summary, env)
        }
        HostExprKind::If {
            then_expr,
            else_expr,
            ..
        } => {
            let t = result_alias_set(then_expr, param_index, summary, env);
            let e = result_alias_set(else_expr, param_index, summary, env);
            t.join(e)
        }
        HostExprKind::MatchOption {
            bind_name,
            some_expr,
            none_expr,
            ..
        } => {
            // `bind_name` names the Option's unwrapped inner value (a
            // fresh scalar/boxed extraction), not a parameter; shadow any
            // outer entry with the empty set for the `some` arm.
            let prev = env.insert(bind_name.clone(), ReturnsArg::empty());
            let s = result_alias_set(some_expr, param_index, summary, env);
            match prev {
                Some(set) => {
                    env.insert(bind_name.clone(), set);
                }
                None => {
                    env.remove(bind_name);
                }
            }
            let n = result_alias_set(none_expr, param_index, summary, env);
            s.join(n)
        }
        HostExprKind::MatchAdt {
            arms, default_expr, ..
        } => {
            let mut acc = ReturnsArg::empty();
            for arm in arms {
                // The arm's pattern bindings name freshly-accessed ADT
                // fields (a `chelis_adt_field` read returns an independent
                // retained handle), not the enclosing function's
                // parameters. Shadow any outer `env` entry of the same
                // name with the empty set for the arm body so a coincidental
                // name reuse cannot spuriously propagate a parameter alias.
                let saved: Vec<(String, Option<ReturnsArg>)> = arm
                    .bindings
                    .iter()
                    .map(|b| {
                        (
                            b.name.clone(),
                            env.insert(b.name.clone(), ReturnsArg::empty()),
                        )
                    })
                    .collect();
                acc = acc.join(result_alias_set(&arm.expr, param_index, summary, env));
                for (name, prev) in saved {
                    match prev {
                        Some(set) => {
                            env.insert(name, set);
                        }
                        None => {
                            env.remove(&name);
                        }
                    }
                }
            }
            if let Some(default) = default_expr {
                acc = acc.join(result_alias_set(default, param_index, summary, env));
            }
            acc
        }
        HostExprKind::Call { function, args, .. } => {
            // The call's result aliases this function's parameters only
            // through whichever arguments the callee returns. If the
            // callee is not yet summarized, treat it as may-return-any of
            // its arguments (conservative top): any argument that itself
            // aliases a parameter then propagates.
            let callee = summary.get(function);
            let mut acc = ReturnsArg::empty();
            for (i, arg) in args.iter().enumerate() {
                let returns_this = match callee {
                    Some(s) => s.may_return(i),
                    None => true,
                };
                if returns_this {
                    acc = acc.join(result_alias_set(arg, param_index, summary, env));
                }
            }
            acc
        }
        HostExprKind::WithSeed { body, .. } => result_alias_set(body, param_index, summary, env),
        // Constructors, literals, builtins, field access, the iterator
        // lanes, and tensor calls all build fresh allocations whose
        // result does not alias an incoming parameter pointer. (A builtin
        // like `id` is not a user function call; the few identity-shaped
        // builtins still hand back a retained/independent reference, so
        // treating them as fresh here is sound for the block-release
        // balance.)
        _ => ReturnsArg::empty(),
    }
}

pub fn emit_host_program(
    program: &ConcreteHostProgram,
    program_name: &str,
) -> Result<String, Unsupported> {
    let abi_program = project_program(program)?;
    emit_host_abi_program(&abi_program, program_name)
}

pub(crate) fn emit_host_abi_program(
    program: &HostProgram,
    program_name: &str,
) -> Result<String, Unsupported> {
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
    let returns_arg = analyze_returns_arg(program);
    let header = emit_host_header_with_linkage(program, program_name, internal_linkage)?;
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
            body.push(format!("static {};", c_decl(&binding.ty, name)?));
        }
        body.push(String::new());
    }

    for (index, helper) in program.global_tensor_helpers.iter().enumerate() {
        helper_requirements.merge(append_helper(
            &mut body,
            helper,
            &format!("{program_name}__global__tensor_{index}"),
        )?);
    }
    // chelis#730 Phase 1: an emission failure inside a function that the
    // program's globals can actually REACH is a hard build error; a
    // failure inside an exported-but-unreachable wrapper (e.g. a def
    // whose only use was inlined into a grad DAG, leaving the standalone
    // host wrapper dead) emits a branded self-naming ABORT stub instead -
    // the loud fallback shape the Metal rank-2 stub established. Object
    // mode (no globals, no main) has no reachability notion; every
    // function is live export surface there and fails hard.
    let reachable_functions = if program.globals.is_empty() {
        None
    } else {
        Some(host_functions_reachable_from_main(program))
    };
    let function_is_live = |name: &str| -> bool {
        reachable_functions
            .as_ref()
            .is_none_or(|reachable| reachable.contains(name))
    };

    let mut stubbed_functions: HashSet<String> = HashSet::new();
    let mut function_bodies: Vec<String> = Vec::new();
    for function in &program.functions {
        let emitted_name = emitted_names
            .get(&function.name)
            .expect("host function emitted name");
        let mut fn_buf: Vec<String> = Vec::new();
        match emit_function(
            &mut fn_buf,
            function,
            emitted_name,
            &emitted_names,
            &function_specializations,
            &returns_arg,
            internal_linkage,
        ) {
            Ok(()) => {
                function_bodies.extend(fn_buf);
                function_bodies.push(String::new());
            }
            Err(unsupported) if !function_is_live(&function.name) => {
                stubbed_functions.insert(function.name.clone());
                append_unreachable_fn_abort_stub(
                    &mut function_bodies,
                    function,
                    emitted_name,
                    internal_linkage,
                    &unsupported,
                )?;
                function_bodies.push(String::new());
            }
            Err(unsupported) => return Err(unsupported),
        }
    }

    for function in &program.functions {
        if stubbed_functions.contains(&function.name) {
            // A stubbed wrapper aborts before any helper call; skip its
            // (possibly unemittable) tensor helpers entirely.
            continue;
        }
        for (index, helper) in function.tensor_helpers.iter().enumerate() {
            let function_name = emitted_names
                .get(&function.name)
                .expect("host function emitted name");
            helper_requirements.merge(append_helper(
                &mut body,
                helper,
                &format!("{function_name}__tensor_{index}"),
            )?);
        }
    }

    body.extend(function_bodies);

    if !program.globals.is_empty() {
        let hoisted: HashSet<&str> = captured_globals.iter().map(String::as_str).collect();
        emit_main(&mut body, program_name, program, &returns_arg, &hoisted)?;
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
    Ok(out.join("\n"))
}

fn emitted_function_name(program_name: &str, function_name: &str) -> String {
    if function_name == "main" {
        format!("{program_name}__main")
    } else {
        // chelis#840: def names are C identifiers too. Route them through
        // the same #379 mapping as bindings and parameters so a def named
        // `double` declares, references, and prototypes consistently as
        // `chelis_user__double` instead of emitting a C keyword verbatim.
        c_ident(function_name).into_owned()
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
    // chelis#770: one explicit correctly-rounded FMA, flag-independent and
    // bit-identical to the host evaluator's `f32::mul_add`. Byte-identical to
    // the `emit.rs` copy (see the rationale there).
    out.push("    return fmaf(high - low, (float)unit, low);".to_string());
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
    out.push("            case CHELIS_F32: value = (double)t->data[i]; break;".to_string());
    // chelis#730 Phase 1 (census row 10 interim-hardening): a dtype this
    // helper cannot decode (f16/bf16 2-byte storage) aborts with the
    // dtype id instead of misreading the buffer as f32. The faithful
    // rendering is chelis#728/#732's work (their generated formatter
    // replaces this helper); until then the abort is the section C1
    // rule-4 response, mirroring the runtime `to_tensor` abort shape.
    out.push("            default:".to_string());
    out.push(
        "                fprintf(stderr, \"unsupported: tensor print of dtype id %d on \
         the emitted C print helper (runtime); f16/bf16 tensor rendering is tracked by \
         chelis#728\\n\", (int)t->dtype);"
            .to_string(),
    );
    out.push("                exit(1);".to_string());
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

pub fn emit_host_header(
    program: &ConcreteHostProgram,
    program_name: &str,
) -> Result<String, Unsupported> {
    let abi_program = project_program(program)?;
    emit_host_header_with_linkage(&abi_program, program_name, false)
}

pub(crate) fn emit_host_abi_header(
    program: &HostProgram,
    program_name: &str,
) -> Result<String, Unsupported> {
    emit_host_header_with_linkage(program, program_name, false)
}

fn emit_host_header_with_linkage(
    program: &HostProgram,
    program_name: &str,
    internal_linkage: bool,
) -> Result<String, Unsupported> {
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
                .collect::<Result<Vec<_>, _>>()?
                .join(", ");
            let emitted_name = emitted_function_name(program_name, &function.name);
            Ok(format!(
                "{prefix}{} {}({});",
                c_type(&function.ret_ty)?,
                emitted_name,
                params
            ))
        })
        .collect::<Result<Vec<_>, Unsupported>>()
        .map(|headers| headers.join("\n"))
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
) -> Result<HelperRequirements, Unsupported> {
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
        return Ok(HelperRequirements::default());
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
    )?;
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
    Ok(requirements)
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

/// chelis#730 Phase 1: the branded self-naming abort stub emitted for an
/// exported-but-unreachable host wrapper whose real body cannot be
/// emitted (see the reachability gate in `emit_host_program`). If an
/// external consumer links the object and calls it anyway, the failure
/// is loud at run time - the section C1 runtime-abort row, mirroring the
/// Metal rank-2 fallback stub - never a silently-wrong value.
fn append_unreachable_fn_abort_stub(
    out: &mut Vec<String>,
    function: &HostFunction,
    emitted_name: &str,
    internal_linkage: bool,
    unsupported: &Unsupported,
) -> Result<(), Unsupported> {
    let params = function
        .params
        .iter()
        .map(|param| c_decl(&param.ty, &param.name))
        .collect::<Result<Vec<_>, _>>()?
        .join(", ");
    let prefix = if internal_linkage {
        "static inline "
    } else {
        ""
    };
    out.push(format!(
        "{prefix}{} {}({}) {{",
        c_type(&function.ret_ty)?,
        emitted_name,
        params
    ));
    let rendered = unsupported.to_string();
    let safe = chelis_ir::span_sanitize::sanitize_for_format_string(&rendered);
    out.push(format!("    fprintf(stderr, \"%s\\n\", \"{safe}\");"));
    out.push("    abort();".to_string());
    out.push("}".to_string());
    Ok(())
}

fn emit_function(
    out: &mut Vec<String>,
    function: &HostFunction,
    emitted_name: &str,
    emitted_names: &HashMap<String, String>,
    function_specializations: &HashMap<String, HostFunctionSpecialization>,
    returns_arg: &HashMap<String, ReturnsArg>,
    internal_linkage: bool,
) -> Result<(), Unsupported> {
    let params = function
        .params
        .iter()
        .map(|param| c_decl(&param.ty, &param.name))
        .collect::<Result<Vec<_>, _>>()?
        .join(", ");
    let prefix = if internal_linkage {
        "static inline "
    } else {
        ""
    };
    out.push(format!(
        "{prefix}{} {}({}) {{",
        c_type(&function.ret_ty)?,
        emitted_name,
        params
    ));
    let mut emitter = HostEmitter::new(
        "    ".to_string(),
        emitted_name,
        emitted_names.clone(),
        function_specializations.clone(),
        returns_arg.clone(),
        &function.tensor_helpers,
    );
    emitter.emit_expr_to_var(&function.body, "__result", &function.ret_ty)?;
    out.extend(emitter.lines);
    out.push("    return __result;".to_string());
    out.push("}".to_string());
    Ok(())
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
    returns_arg: &HashMap<String, ReturnsArg>,
    hoisted: &HashSet<&str>,
) -> Result<(), Unsupported> {
    out.push("int main(void) {".to_string());
    // chelis#840: the globals emitter needs the same original-to-emitted
    // function-name map as function bodies, or a global calling a def
    // whose name was mangled (`double`) or renamed (`main`) emits the raw
    // name and the C cannot compile.
    let mut emitter = HostEmitter::new(
        BASE_MAIN_INDENT.to_string(),
        &format!("{program_name}__global"),
        emitted_function_names(program, program_name),
        function_specializations(program),
        returns_arg.clone(),
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
        emitter.emit_expr_to_var(&binding.value, &binding_var, &binding.ty)?;
        // The binding-value local owns its allocation regardless of how
        // it was produced (tensor kernel output, list/dict builtin,
        // literal). Track it here; the alias name (`theta`) is never
        // tracked, and dedup in `emit_scope_releases` collapses the case
        // where the binding value *is* a literal already tracked above.
        emitter.track_owned_alloc(&binding_var, &binding.ty);
        if hoisted.contains(binding.name.as_str()) {
            // Declared at file scope (issue #352); assign, don't shadow.
            // #379: reference the same mangled name the file-scope `static`
            // declaration used (both route through `c_ident`).
            emitter.lines.push(format!(
                "    {} = __binding_{index}_value;",
                c_ident(&binding.name)
            ));
        } else {
            emitter.lines.push(format!(
                "    {} = __binding_{index}_value;",
                c_decl(&binding.ty, &binding.name)?
            ));
        }
    }
    for binding in &program.globals {
        if let Some(display_name) = binding.display_name.as_deref() {
            // #379: the display label stays raw (it is a printed string);
            // the C value identifier routes through `c_ident` so it matches
            // the (possibly mangled) declaration above.
            emitter.emit_labeled_root(display_name, &c_ident(&binding.name), &binding.ty)?;
        }
    }
    // issue #406: free everything `main` owns before returning. Emitted
    // after the labeled-root prints so the values are still live when
    // printed and reclaimed immediately after.
    emitter.emit_scope_releases();
    out.extend(emitter.lines);
    out.push("    return 0;".to_string());
    out.push("}".to_string());
    Ok(())
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

/// Function names referenced from `expr` - `Call`/`Named`-callback
/// targets plus bare `Var` references (a def passed as a value). The
/// over-approximation direction is the safe one for the reachability
/// gate below: an over-counted reference makes a failing wrapper a hard
/// build error rather than a loud stub.
fn collect_referenced_fn_names(expr: &HostExpr, out: &mut HashSet<String>) {
    collect_var_names(expr, out);
    fn walk(expr: &HostExpr, out: &mut HashSet<String>) {
        match &expr.kind {
            HostExprKind::Call { function, args, .. } => {
                out.insert(function.clone());
                for arg in args {
                    walk(arg, out);
                }
            }
            HostExprKind::Builtin { args, .. } | HostExprKind::TensorCall { args, .. } => {
                for arg in args {
                    walk(arg, out);
                }
            }
            HostExprKind::List(items, _) | HostExprKind::Tuple(items, _) => {
                for item in items {
                    walk(item, out);
                }
            }
            HostExprKind::AdtConstruct { fields, .. } => {
                for field in fields {
                    walk(field, out);
                }
            }
            HostExprKind::AdtFieldAccess { base, .. } => walk(base, out),
            HostExprKind::If {
                cond,
                then_expr,
                else_expr,
                ..
            } => {
                walk(cond, out);
                walk(then_expr, out);
                walk(else_expr, out);
            }
            HostExprKind::MatchOption {
                scrutinee,
                some_expr,
                none_expr,
                ..
            } => {
                walk(scrutinee, out);
                walk(some_expr, out);
                walk(none_expr, out);
            }
            HostExprKind::MatchAdt {
                scrutinee,
                arms,
                default_expr,
                ..
            } => {
                walk(scrutinee, out);
                for arm in arms {
                    walk(&arm.expr, out);
                }
                if let Some(default_expr) = default_expr {
                    walk(default_expr, out);
                }
            }
            HostExprKind::Let { bindings, body, .. } => {
                for binding in bindings {
                    walk(&binding.value, out);
                }
                walk(body, out);
            }
            HostExprKind::Map { callback, list, .. }
            | HostExprKind::Filter { callback, list, .. }
            | HostExprKind::Partition { callback, list, .. }
            | HostExprKind::FlatMap { callback, list, .. } => {
                walk_callback(callback, out);
                walk(list, out);
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
                walk_callback(callback, out);
                walk(init, out);
                walk(list, out);
            }
            HostExprKind::WithSeed { seed, body, .. } => {
                walk(seed, out);
                walk(body, out);
            }
            HostExprKind::Int(_)
            | HostExprKind::Float(_)
            | HostExprKind::Bool(_)
            | HostExprKind::String(_)
            | HostExprKind::Unit
            | HostExprKind::Var(_, _) => {}
        }
    }
    fn walk_callback(callback: &HostCallback, out: &mut HashSet<String>) {
        match &callback.kind {
            HostCallbackKind::Named { function, .. } => {
                out.insert(function.clone());
            }
            HostCallbackKind::Inline { body, .. } => walk(body, out),
        }
    }
    walk(expr, out);
}

/// The set of host functions transitively reachable from the program's
/// global bindings (the emitted `main`). Used by `emit_host_program` to
/// decide whether an UNSUPPORTED emission failure inside a function is a
/// hard build error (the function is on the program's live surface) or a
/// loud abort stub (an exported-but-unreachable wrapper - e.g. a def
/// whose only use was inlined into a grad DAG; the abort keeps an
/// external caller loud at run time, the Metal rank-2 stub precedent).
fn host_functions_reachable_from_main(program: &HostProgram) -> HashSet<String> {
    let by_name: HashMap<&str, &HostFunction> = program
        .functions
        .iter()
        .map(|function| (function.name.as_str(), function))
        .collect();
    let mut seed = HashSet::new();
    for binding in &program.globals {
        collect_referenced_fn_names(&binding.value, &mut seed);
    }
    let mut reachable: HashSet<String> = HashSet::new();
    let mut stack: Vec<String> = seed
        .into_iter()
        .filter(|name| by_name.contains_key(name.as_str()))
        .collect();
    while let Some(name) = stack.pop() {
        if !reachable.insert(name.clone()) {
            continue;
        }
        if let Some(function) = by_name.get(name.as_str()) {
            let mut refs = HashSet::new();
            collect_referenced_fn_names(&function.body, &mut refs);
            for r in refs {
                if by_name.contains_key(r.as_str()) && !reachable.contains(&r) {
                    stack.push(r);
                }
            }
        }
    }
    reachable
}

struct HostEmitter<'a> {
    lines: Vec<String>,
    indent: String,
    helper_prefix: String,
    emitted_names: HashMap<String, String>,
    function_specializations: HashMap<String, HostFunctionSpecialization>,
    /// Interprocedural "result aliases parameter" summary (issue #406
    /// call-escape): maps a user function's name to the set of parameter
    /// indices whose allocation its result may *be* (rather than a fresh
    /// allocation). Computed once per program by [`analyze_returns_arg`].
    /// When a block result is `f(p, ...)` and `f` may return its first
    /// parameter, the block-frame heap binding `p` escapes through the
    /// call and must be retained so the block's release does not drop the
    /// reference the caller now holds. A callee absent from this map
    /// (recursion not yet at fixpoint, an unanalyzable builtin path, or a
    /// genuinely opaque call) is treated conservatively as may-return-any
    /// by the emit-site logic, which is use-after-free-safe (it may
    /// over-retain, never under-retain).
    returns_arg: HashMap<String, ReturnsArg>,
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
        returns_arg: HashMap<String, ReturnsArg>,
        tensor_helpers: &'a [HostTensorHelper],
    ) -> Self {
        Self {
            lines: Vec::new(),
            indent,
            helper_prefix: helper_prefix.to_string(),
            emitted_names,
            function_specializations,
            returns_arg,
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

    fn emit_expr_to_var(
        &mut self,
        expr: &HostExpr,
        target: &str,
        ty: &HostType,
    ) -> Result<(), Unsupported> {
        self.lines
            .push(format!("{}{};", self.indent, c_decl(ty, target)?));
        self.assign_expr(target, expr, ty)?;
        Ok(())
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

    fn assign_expr(
        &mut self,
        target: &str,
        expr: &HostExpr,
        ty: &HostType,
    ) -> Result<(), Unsupported> {
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
                require_same_abi_type(ty, expr_ty, "list expression")?;
                self.assign_list_literal(target, items, ty)?;
            }
            HostExprKind::Tuple(items, expr_ty) => {
                require_same_abi_type(ty, expr_ty, "tuple expression")?;
                self.assign_tuple_literal(target, items, ty)?;
            }
            HostExprKind::Var(name, var_ty) => {
                if name == "Nil" {
                    self.lines
                        .push(format!("{}{target} = chelis_list_empty();", self.indent));
                } else if name == "None" && matches!(ty, HostType::Option(_)) {
                    self.assign_option_none(target, ty)?;
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
                    //
                    // #379: route the referenced name through `c_ident` so a
                    // binding/param/let spelled like a C keyword resolves to
                    // the same mangled identifier its declaration used.
                    self.lines
                        .push(format!("{}{target} = {};", self.indent, c_ident(name)));
                    self.retain_transferred_result(target, name, ty);
                }
            }
            HostExprKind::Call {
                function,
                args,
                arg_tys,
                ty: call_ty,
            } => {
                require_same_abi_type(ty, call_ty, "call expression")?;
                self.assign_call(target, function, args, arg_tys, call_ty)?;
            }
            HostExprKind::Builtin {
                name,
                args,
                ty: expr_ty,
            } => {
                require_same_abi_type(ty, expr_ty, "builtin expression")?;
                self.assign_builtin(target, name, args, ty)?;
            }
            HostExprKind::AdtConstruct {
                ctor,
                fields,
                ty: expr_ty,
            } => {
                require_same_abi_type(ty, expr_ty, "ADT construction")?;
                self.assign_adt_construct(target, ctor, fields, ty)?;
            }
            HostExprKind::AdtFieldAccess {
                base,
                field_index,
                ty: expr_ty,
            } => {
                require_same_abi_type(ty, expr_ty, "ADT field access")?;
                self.assign_adt_field_access(target, base, *field_index, ty)?;
            }
            HostExprKind::If {
                cond,
                then_expr,
                else_expr,
                ty: expr_ty,
            } => {
                require_same_abi_type(ty, expr_ty, "if expression")?;
                let cond_var = self.next_temp("cond");
                self.emit_expr_to_var(cond, &cond_var, &HostType::Bool)?;
                self.lines
                    .push(format!("{}if ({cond_var}) {{", self.indent));
                let nested_indent = format!("{}    ", self.indent);
                let previous = std::mem::replace(&mut self.indent, nested_indent);
                self.assign_expr(target, then_expr, ty)?;
                self.indent = previous.clone();
                self.lines.push(format!("{}}} else {{", self.indent));
                let nested_indent = format!("{}    ", self.indent);
                let previous = std::mem::replace(&mut self.indent, nested_indent);
                self.assign_expr(target, else_expr, ty)?;
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
                require_same_abi_type(ty, expr_ty, "option match")?;
                let option_var = self.next_temp("option");
                let option_ty = host_type(scrutinee);
                self.emit_expr_to_var(scrutinee, &option_var, &option_ty)?;
                self.lines
                    .push(format!("{}if ({}.is_some) {{", self.indent, option_var));
                let nested_indent = format!("{}    ", self.indent);
                let previous = std::mem::replace(&mut self.indent, nested_indent);
                let inner_ty = option_inner_type(&option_ty)?;
                match option_ty {
                    HostType::Option(inner)
                        if !matches!(inner.as_ref(), HostType::Int64 | HostType::Float64) =>
                    {
                        self.lines.push(format!(
                            "{}{} {};",
                            self.indent,
                            c_type(&inner_ty)?,
                            bind_name
                        ));
                        self.assign_unboxed_value(
                            bind_name,
                            &inner_ty,
                            &format!("{option_var}.value"),
                        )?;
                    }
                    _ => {
                        self.lines.push(format!(
                            "{}{} {} = {}.value;",
                            self.indent,
                            c_type(&inner_ty)?,
                            bind_name,
                            option_var
                        ));
                    }
                }
                self.assign_expr(target, some_expr, ty)?;
                self.indent = previous.clone();
                self.lines.push(format!("{}}} else {{", self.indent));
                let nested_indent = format!("{}    ", self.indent);
                let previous = std::mem::replace(&mut self.indent, nested_indent);
                self.assign_expr(target, none_expr, ty)?;
                self.indent = previous;
                self.lines.push(format!("{}}}", self.indent));
            }
            HostExprKind::MatchAdt {
                scrutinee,
                arms,
                default_expr,
                ty: expr_ty,
            } => {
                require_same_abi_type(ty, expr_ty, "ADT match")?;
                self.assign_match_adt(target, scrutinee, arms, default_expr.as_deref(), ty)?;
            }
            HostExprKind::Let {
                bindings,
                body,
                ty: expr_ty,
            } => {
                require_same_abi_type(ty, expr_ty, "let expression")?;
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
                    self.emit_expr_to_var(&binding.value, &temp, &binding.ty)?;
                    self.lines.push(format!(
                        "{}{};",
                        self.indent,
                        c_decl(&binding.ty, &binding.name)?
                    ));
                    // #379: assign to the same mangled identifier the
                    // declaration used (both route through `c_ident`).
                    self.lines.push(format!(
                        "{}{} = {};",
                        self.indent,
                        c_ident(&binding.name),
                        temp
                    ));
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
                self.assign_expr(target, body, ty)?;
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
                self.assign_map(target, callback, list, ty)?;
            }
            HostExprKind::Filter { callback, list, ty } => {
                self.assign_filter(target, callback, list, ty)?;
            }
            HostExprKind::Fold {
                callback,
                init,
                list,
                ty,
            } => {
                self.assign_fold(target, callback, init, list, ty)?;
            }
            HostExprKind::Scan {
                callback,
                init,
                list,
                ty,
            } => {
                self.assign_scan(target, callback, init, list, ty)?;
            }
            HostExprKind::Partition { callback, list, ty } => {
                self.assign_partition(target, callback, list, ty)?;
            }
            HostExprKind::FlatMap { callback, list, ty } => {
                self.assign_flat_map(target, callback, list, ty)?;
            }
            HostExprKind::WithSeed { seed, body, ty } => {
                let seed_var = self.next_temp("seed");
                self.emit_expr_to_var(seed, &seed_var, &HostType::Int64)?;
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
                self.assign_expr(target, body, ty)?;
                self.lines
                    .push(format!("{}chelis_rng_current = {saved_var};", self.indent));
            }
            HostExprKind::TensorCall { helper, args, ty } => {
                self.assign_tensor_call(target, *helper, args, ty)?;
            }
            HostExprKind::Unit => {
                require_same_abi_type(ty, &HostType::Unit, "unit expression")?;
                self.lines.push(format!("{}{target} = 0;", self.indent));
            }
        }
        Ok(())
    }

    fn assign_builtin(
        &mut self,
        target: &str,
        name: &str,
        args: &[HostExpr],
        ty: &HostType,
    ) -> Result<(), Unsupported> {
        let mut arg_vars: Vec<(String, HostType)> = Vec::with_capacity(args.len());
        for (index, arg) in args.iter().enumerate() {
            let arg_name = self.next_temp(&format!("arg{index}"));
            let inferred_ty = host_type(arg);
            // A few builtins carry a type-directed argument (for example
            // `Some` and `append`).  Use that explicit contract when it is
            // available; otherwise the already-resolved expression type is
            // authoritative.  There is no catch-all ABI default.
            let arg_ty = expected_builtin_arg_ty(name, ty, index).unwrap_or(inferred_ty);
            self.emit_expr_to_var(arg, &arg_name, &arg_ty)?;
            arg_vars.push((arg_name, arg_ty));
        }

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
                    return Ok(());
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
                    return Ok(());
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
                    return Ok(());
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
                    return Ok(());
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
                    return Ok(());
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
                    return Ok(());
                }
                "neg" if matches!(&arg_vars[0].1, HostType::Tensor(_)) => {
                    self.assign_tensor_unary_elementwise(target, &arg_vars[0].0, "-");
                    return Ok(());
                }
                "not" if matches!(&arg_vars[0].1, HostType::Tensor(_)) => {
                    self.assign_tensor_unary_elementwise(target, &arg_vars[0].0, "!");
                    return Ok(());
                }
                "exp" if matches!(&arg_vars[0].1, HostType::Tensor(_)) => {
                    self.assign_tensor_unary_func_elementwise(target, &arg_vars[0].0, "expf");
                    return Ok(());
                }
                "log" if matches!(&arg_vars[0].1, HostType::Tensor(_)) => {
                    self.assign_tensor_unary_func_elementwise(target, &arg_vars[0].0, "logf");
                    return Ok(());
                }
                "sin" if matches!(&arg_vars[0].1, HostType::Tensor(_)) => {
                    self.assign_tensor_unary_func_elementwise(target, &arg_vars[0].0, "sinf");
                    return Ok(());
                }
                "sqrt" if matches!(&arg_vars[0].1, HostType::Tensor(_)) => {
                    self.assign_tensor_unary_func_elementwise(target, &arg_vars[0].0, "sqrtf");
                    return Ok(());
                }
                "relu" if matches!(&arg_vars[0].1, HostType::Tensor(_)) => {
                    self.assign_tensor_unary_func_elementwise(
                        target,
                        &arg_vars[0].0,
                        "chelis_host_relu_f32",
                    );
                    return Ok(());
                }
                "sigmoid" if matches!(&arg_vars[0].1, HostType::Tensor(_)) => {
                    self.assign_tensor_unary_func_elementwise(
                        target,
                        &arg_vars[0].0,
                        "chelis_host_sigmoid_f32",
                    );
                    return Ok(());
                }
                "tanh" if matches!(&arg_vars[0].1, HostType::Tensor(_)) => {
                    self.assign_tensor_unary_func_elementwise(
                        target,
                        &arg_vars[0].0,
                        "chelis_host_tanh_f32",
                    );
                    return Ok(());
                }
                "silu" if matches!(&arg_vars[0].1, HostType::Tensor(_)) => {
                    self.assign_tensor_unary_func_elementwise(
                        target,
                        &arg_vars[0].0,
                        "chelis_host_silu_f32",
                    );
                    return Ok(());
                }
                "gelu" if matches!(&arg_vars[0].1, HostType::Tensor(_)) => {
                    self.assign_tensor_unary_func_elementwise(
                        target,
                        &arg_vars[0].0,
                        "chelis_host_gelu_f32",
                    );
                    return Ok(());
                }
                _ => {}
            }
        }

        match name {
            "Some" => {
                self.assign_option_some(target, ty, &arg_vars[0].0, &arg_vars[0].1)?;
                return Ok(());
            }
            "None" => {
                self.assign_option_none(target, ty)?;
                return Ok(());
            }
            "cast" => {
                let expr = match (&arg_vars[0].1, ty) {
                    (source, target) if is_integer_abi(source) && is_integer_abi(target) => {
                        format!("({}){}", c_type(target)?, arg_vars[0].0)
                    }
                    (source, HostType::Float64) if is_integer_abi(source) => {
                        format!("(double){}", arg_vars[0].0)
                    }
                    (source, HostType::Float32) if is_integer_abi(source) => {
                        format!("(float){}", arg_vars[0].0)
                    }
                    (HostType::Float64 | HostType::Float32, target) if is_integer_abi(target) => {
                        format!("({}){}", c_type(target)?, arg_vars[0].0)
                    }
                    // WS-4: float entry params can now be `Float32`, so the
                    // int<->float casts must cover both float widths. The C
                    // numeric cast handles the narrowing/widening to the
                    // declared destination type.
                    (HostType::Int64, HostType::Float64) => {
                        format!("(double){0}", arg_vars[0].0)
                    }
                    (HostType::Int64, HostType::Float32) => {
                        format!("(float){0}", arg_vars[0].0)
                    }
                    (HostType::Float64 | HostType::Float32, HostType::Int64) => {
                        format!("(int64_t){0}", arg_vars[0].0)
                    }
                    (HostType::Float64, HostType::Float32) => {
                        format!("(float){0}", arg_vars[0].0)
                    }
                    (HostType::Float32, HostType::Float64) => {
                        format!("(double){0}", arg_vars[0].0)
                    }
                    (HostType::Bool, target) if is_integer_abi(target) => {
                        format!("({}){}", c_type(target)?, arg_vars[0].0)
                    }
                    (source, HostType::Bool) if is_integer_abi(source) => {
                        format!("((bool){})", arg_vars[0].0)
                    }
                    _ => arg_vars[0].0.clone(),
                };
                self.lines
                    .push(format!("{}{target} = {};", self.indent, expr));
                return Ok(());
            }
            "copy" => {
                self.lines
                    .push(format!("{}{target} = {};", self.indent, arg_vars[0].0));
                return Ok(());
            }
            "tuple-get" => {
                let value_var = self.next_temp("tuple_value");
                self.lines.push(format!(
                    "{}chelis_value {} = chelis_tuple_get({}, {});",
                    self.indent, value_var, arg_vars[0].0, arg_vars[1].0
                ));
                self.assign_unboxed_value(target, ty, &value_var)?;
                return Ok(());
            }
            "index" => {
                let value_var = self.next_temp("list_value");
                self.lines.push(format!(
                    "{}chelis_value {} = chelis_list_index({}, {});",
                    self.indent, value_var, arg_vars[0].0, arg_vars[1].0
                ));
                self.assign_unboxed_value(target, ty, &value_var)?;
                return Ok(());
            }
            "append" => {
                self.lines.push(format!(
                    "{}{target} = chelis_list_append({}, {});",
                    self.indent,
                    arg_vars[0].0,
                    self.box_value_expr(&arg_vars[1].0, &arg_vars[1].1)?
                ));
                return Ok(());
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
                return Ok(());
            }
            "split" => {
                self.lines.push(format!(
                    "{}{target} = chelis_tensor_split({}, {}, {});",
                    self.indent, arg_vars[0].0, arg_vars[1].0, arg_vars[2].0
                ));
                return Ok(());
            }
            "gather" => {
                self.lines.push(format!(
                    "{}{target} = chelis_tensor_gather({}, {}, {});",
                    self.indent, arg_vars[0].0, arg_vars[1].0, arg_vars[2].0
                ));
                return Ok(());
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
                return Ok(());
            }
            "where" => {
                self.lines.push(format!(
                    "{}{target} = chelis_tensor_where({}, {}, {});",
                    self.indent, arg_vars[0].0, arg_vars[1].0, arg_vars[2].0
                ));
                return Ok(());
            }
            "cumsum" => {
                self.lines.push(format!(
                    "{}{target} = chelis_tensor_cumsum({}, {});",
                    self.indent, arg_vars[0].0, arg_vars[1].0
                ));
                return Ok(());
            }
            "sort" => {
                self.lines.push(format!(
                    "{}{target} = chelis_tensor_sort({}, {});",
                    self.indent, arg_vars[0].0, arg_vars[1].0
                ));
                return Ok(());
            }
            "diagonal" => {
                self.lines.push(format!(
                    "{}{target} = chelis_tensor_diagonal({}, {}, {});",
                    self.indent, arg_vars[0].0, arg_vars[1].0, arg_vars[2].0
                ));
                return Ok(());
            }
            "trace" => {
                self.lines.push(format!(
                    "{}{target} = chelis_tensor_trace({}, {}, {});",
                    self.indent, arg_vars[0].0, arg_vars[1].0, arg_vars[2].0
                ));
                return Ok(());
            }
            "clamp" => {
                self.lines.push(format!(
                    "{}{target} = chelis_tensor_clamp({}, {}, {});",
                    self.indent, arg_vars[0].0, arg_vars[1].0, arg_vars[2].0
                ));
                return Ok(());
            }
            "einsum" => {
                self.lines.push(format!(
                    "{}{target} = chelis_tensor_einsum({}, {}, {});",
                    self.indent, arg_vars[0].0, arg_vars[1].0, arg_vars[2].0
                ));
                return Ok(());
            }
            "take" => {
                self.lines.push(format!(
                    "{}{target} = chelis_list_take({}, {});",
                    self.indent, arg_vars[0].0, arg_vars[1].0
                ));
                return Ok(());
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
                return Ok(());
            }
            "chunk" => {
                self.lines.push(format!(
                    "{}{target} = chelis_list_chunk({}, {});",
                    self.indent, arg_vars[0].0, arg_vars[1].0
                ));
                return Ok(());
            }
            "flatten" => {
                self.lines.push(format!(
                    "{}{target} = chelis_list_flatten({});",
                    self.indent, arg_vars[0].0
                ));
                return Ok(());
            }
            "zip" => {
                self.lines.push(format!(
                    "{}{target} = chelis_list_zip({}, {});",
                    self.indent, arg_vars[0].0, arg_vars[1].0
                ));
                return Ok(());
            }
            "enumerate" => {
                self.lines.push(format!(
                    "{}{target} = chelis_list_enumerate({});",
                    self.indent, arg_vars[0].0
                ));
                return Ok(());
            }
            "dict_of" => {
                self.lines.push(format!(
                    "{}{target} = chelis_dict_from_pairs({});",
                    self.indent, arg_vars[0].0
                ));
                return Ok(());
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
                    self.box_value_expr(&arg_vars[1].0, &arg_vars[1].1)?
                ));
                return Ok(());
            }
            "dict_contains" => {
                self.lines.push(format!(
                    "{}{target} = chelis_dict_contains({}, {});",
                    self.indent,
                    arg_vars[0].0,
                    self.box_value_expr(&arg_vars[1].0, &arg_vars[1].1)?
                ));
                return Ok(());
            }
            "dict_remove" => {
                self.lines.push(format!(
                    "{}{target} = chelis_dict_remove({}, {});",
                    self.indent,
                    arg_vars[0].0,
                    self.box_value_expr(&arg_vars[1].0, &arg_vars[1].1)?
                ));
                return Ok(());
            }
            "dict_insert" => {
                self.lines.push(format!(
                    "{}{target} = chelis_dict_insert({}, {}, {});",
                    self.indent,
                    arg_vars[0].0,
                    self.box_value_expr(&arg_vars[1].0, &arg_vars[1].1)?,
                    self.box_value_expr(&arg_vars[2].0, &arg_vars[2].1)?
                ));
                return Ok(());
            }
            "dict_merge" => {
                self.lines.push(format!(
                    "{}{target} = chelis_dict_merge({}, {});",
                    self.indent, arg_vars[0].0, arg_vars[1].0
                ));
                return Ok(());
            }
            "dict_keys" => {
                self.lines.push(format!(
                    "{}{target} = chelis_dict_keys({});",
                    self.indent, arg_vars[0].0
                ));
                return Ok(());
            }
            "dict_values" => {
                self.lines.push(format!(
                    "{}{target} = chelis_dict_values({});",
                    self.indent, arg_vars[0].0
                ));
                return Ok(());
            }
            "dict_entries" => {
                self.lines.push(format!(
                    "{}{target} = chelis_dict_entries({});",
                    self.indent, arg_vars[0].0
                ));
                return Ok(());
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
                if let HostType::Tensor(t) = ty
                    && let Ok(dtype) = t.precision.runtime_dtype()
                {
                    self.lines.push(format!(
                        "{}{target} = chelis_tensor_from_value_list_typed({}, {});",
                        self.indent,
                        arg_vars[0].0,
                        dtype.c_macro()
                    ));
                    return Ok(());
                }
                self.lines.push(format!(
                    "{}{target} = chelis_tensor_from_value_list({});",
                    self.indent, arg_vars[0].0
                ));
                return Ok(());
            }
            "to_list" => {
                self.lines.push(format!(
                    "{}{target} = chelis_list_from_tensor({});",
                    self.indent, arg_vars[0].0
                ));
                return Ok(());
            }
            "pad_sequences" => {
                self.lines.push(format!(
                    "{}{target} = chelis_pad_sequences({}, {});",
                    self.indent,
                    arg_vars[0].0,
                    self.box_value_expr(&arg_vars[1].0, &arg_vars[1].1)?
                ));
                return Ok(());
            }
            "pad_sequences_to" => {
                self.lines.push(format!(
                    "{}{target} = chelis_pad_sequences_to({}, {}, {});",
                    self.indent,
                    arg_vars[0].0,
                    arg_vars[1].0,
                    self.box_value_expr(&arg_vars[2].0, &arg_vars[2].1)?
                ));
                return Ok(());
            }
            "read_file" => {
                self.lines.push(format!(
                    "{}{target} = chelis_read_file({});",
                    self.indent, arg_vars[0].0
                ));
                return Ok(());
            }
            "write_file" => {
                self.lines.push(format!(
                    "{}chelis_write_file({}, {});",
                    self.indent, arg_vars[0].0, arg_vars[1].0
                ));
                self.lines.push(format!("{}{target} = 0;", self.indent));
                return Ok(());
            }
            "read_lines" => {
                self.lines.push(format!(
                    "{}{target} = chelis_read_lines({});",
                    self.indent, arg_vars[0].0
                ));
                return Ok(());
            }
            "read_bytes" => {
                self.lines.push(format!(
                    "{}{target} = chelis_read_bytes({});",
                    self.indent, arg_vars[0].0
                ));
                return Ok(());
            }
            "file_exists" => {
                self.lines.push(format!(
                    "{}{target} = chelis_file_exists({});",
                    self.indent, arg_vars[0].0
                ));
                return Ok(());
            }
            "list_dir" => {
                self.lines.push(format!(
                    "{}{target} = chelis_list_dir({});",
                    self.indent, arg_vars[0].0
                ));
                return Ok(());
            }
            "mmap_file" => {
                self.lines.push(format!(
                    "{}{target} = chelis_mmap_file({});",
                    self.indent, arg_vars[0].0
                ));
                return Ok(());
            }
            "mmap_read" => {
                self.lines.push(format!(
                    "{}{target} = chelis_mmap_read({}, {}, {});",
                    self.indent, arg_vars[0].0, arg_vars[1].0, arg_vars[2].0
                ));
                return Ok(());
            }
            "mmap_len" => {
                self.lines.push(format!(
                    "{}{target} = chelis_mmap_len({});",
                    self.indent, arg_vars[0].0
                ));
                return Ok(());
            }
            _ => {}
        }

        // Scalar numeric emission is exhaustive over the resolved ABI
        // vocabulary.  Unsupported narrow scalar types were rejected by
        // `project_program` and cannot reach this point ([05-UNS-1]).
        const SCALAR_NUMERIC_BUILTINS: &[&str] = &[
            "add",
            "sub",
            "mul",
            "div",
            "floor_div",
            "trunc_div",
            "mod",
            "neg",
            "cmplt",
            "lt",
            "gt",
            "gte",
            "lte",
            "sqrt",
            "exp",
            "log",
            "sin",
            "cos",
            "tanh",
            "pow",
            "abs",
            "min",
            "max",
        ];
        // A TENSOR operand reaching these scalar operator arms means the
        // op has no tensor emission arm (the tensor block above returned
        // early for every op that has one) - emitting `cos(ptr)` or
        // `a + b` over `chelis_tensor*` is garbage C that fails (or
        // corrupts) at the user's compiler. This is the loud terminal the
        // section C3 laundering rule requires for the recoverable
        // `lower_transcendental` raise: the speculative-probe fallback
        // lands here and errs instead of emitting.
        if SCALAR_NUMERIC_BUILTINS.contains(&name)
            && arg_vars
                .iter()
                .any(|(_, arg_ty)| matches!(arg_ty, HostType::Tensor(_)))
        {
            return Err(Unsupported::new(
                UnsupportedKind::Builtin(name.to_string()),
                "tensor operands in `chelis build` host emission (no tensor \
                 emission arm for this op)"
                    .to_string(),
                Stage::Codegen("c"),
                "this op has no compiled tensor arm yet; the eval lane may support \
                 it (chelis#703 class; the DAG lane owns the supported tensor ops)",
            ));
        }

        // These three builtins emit statements as part of their semantics,
        // so they are completed here rather than being admitted to the
        // expression vocabulary below.  All remaining open-set names must
        // cross `CExpressionBuiltin::decode` and the Result-typed expression
        // builder before they can be rendered.
        match name {
            "print" => {
                self.emit_print_value(&arg_vars[0].0, &arg_vars[0].1)?;
                self.lines.push(format!("{}{target} = 0;", self.indent));
                return Ok(());
            }
            "fail" => {
                self.lines
                    .push(format!("{}chelis_fail({});", self.indent, arg_vars[0].0));
                return Ok(());
            }
            "debug" => {
                self.emit_print_value(&arg_vars[0].0, &arg_vars[0].1)?;
                self.lines
                    .push(format!("{}{target} = {};", self.indent, arg_vars[0].0));
                return Ok(());
            }
            _ => {}
        }

        // chelis#730 Phase 2 (C3/C4.4): open-set dispatch constructs only
        // the closed C-expression AST. There is no raw-string node, and the
        // unmatched arm returns `Err(Unsupported)` before an expression can
        // exist.
        let build_expression = || -> Result<EmittedExpr, Unsupported> {
            let arg = |index: usize| EmittedExpr::identifier(arg_vars[index].0.clone());
            let binary = |operator, lhs, rhs| EmittedExpr::binary(operator, lhs, rhs);
            let unary = |operator, operand| EmittedExpr::unary(operator, operand);
            let expression_builtin = CExpressionBuiltin::decode(name)?;
            let expr = match expression_builtin {
                CExpressionBuiltin::Add => binary(BinaryOperator::Add, arg(0), arg(1)),
                CExpressionBuiltin::Sub => binary(BinaryOperator::Subtract, arg(0), arg(1)),
                CExpressionBuiltin::Mul => binary(BinaryOperator::Multiply, arg(0), arg(1)),
                // #387: integer scalar `div`/`mod` trap portably on a zero
                // divisor (ARM64 does not fault on integer div-by-zero), using the
                // same clean diagnostic the evaluator emits. `chelis_int_div_guard`
                // returns the (nonzero) divisor so it composes inline. Float `div`
                // is IEEE-754 and is never guarded; `mod` is integer-only.
                // chelis#178: integer `div` is a type error; this arm is dead
                // (the checker rejects it before host-emit) but kept as a
                // defensive guard. Float `div` is IEEE-754 and never guarded.
                CExpressionBuiltin::Div if is_integer_abi(&arg_vars[0].1) => binary(
                    BinaryOperator::Divide,
                    arg(0),
                    EmittedExpr::call("chelis_int_div_guard", [arg(1)]),
                ),
                CExpressionBuiltin::Div => binary(BinaryOperator::Divide, arg(0), arg(1)),
                // chelis#178: `trunc_div` is integer-only — the guarded C `/`
                // quotient (round toward zero).
                CExpressionBuiltin::TruncDiv => binary(
                    BinaryOperator::Divide,
                    arg(0),
                    EmittedExpr::call("chelis_int_div_guard", [arg(1)]),
                ),
                // chelis#178: `floor_div` rounds toward -inf. Integer (host
                // scalar) operands use the guarded `/` plus a remainder-sign
                // correction; float operands use `floor(a / b)`.
                CExpressionBuiltin::FloorDiv if is_integer_abi(&arg_vars[0].1) => {
                    let guarded_divisor = || EmittedExpr::call("chelis_int_div_guard", [arg(1)]);
                    let quotient = binary(BinaryOperator::Divide, arg(0), guarded_divisor());
                    let remainder = || binary(BinaryOperator::Remainder, arg(0), guarded_divisor());
                    let nonzero = binary(
                        BinaryOperator::NotEqual,
                        remainder(),
                        EmittedExpr::integer(0),
                    );
                    let sign_differs = binary(
                        BinaryOperator::NotEqual,
                        binary(BinaryOperator::Less, remainder(), EmittedExpr::integer(0)),
                        binary(BinaryOperator::Less, arg(1), EmittedExpr::integer(0)),
                    );
                    let correction = EmittedExpr::conditional(
                        binary(BinaryOperator::LogicalAnd, nonzero, sign_differs),
                        EmittedExpr::integer(1),
                        EmittedExpr::integer(0),
                    );
                    binary(BinaryOperator::Subtract, quotient, correction)
                }
                CExpressionBuiltin::FloorDiv => {
                    EmittedExpr::call("floor", [binary(BinaryOperator::Divide, arg(0), arg(1))])
                }
                CExpressionBuiltin::Mod => binary(
                    BinaryOperator::Remainder,
                    arg(0),
                    EmittedExpr::call("chelis_int_div_guard", [arg(1)]),
                ),
                CExpressionBuiltin::CompareLess if matches!(arg_vars[0].1, HostType::Tensor(_)) => {
                    EmittedExpr::call("chelis_tensor_cmplt", [arg(0), arg(1)])
                }
                CExpressionBuiltin::CompareLess | CExpressionBuiltin::Less => {
                    binary(BinaryOperator::Less, arg(0), arg(1))
                }
                CExpressionBuiltin::Greater => binary(BinaryOperator::Greater, arg(0), arg(1)),
                CExpressionBuiltin::GreaterEqual => {
                    binary(BinaryOperator::GreaterEqual, arg(0), arg(1))
                }
                CExpressionBuiltin::LessEqual => binary(BinaryOperator::LessEqual, arg(0), arg(1)),
                CExpressionBuiltin::Equal => match (&arg_vars[0].1, &arg_vars[1].1) {
                    (HostType::String, HostType::String) => {
                        EmittedExpr::call("chelis_string_eq", [arg(0), arg(1)])
                    }
                    _ => binary(BinaryOperator::Equal, arg(0), arg(1)),
                },
                CExpressionBuiltin::NotEqual => match (&arg_vars[0].1, &arg_vars[1].1) {
                    (HostType::String, HostType::String) => unary(
                        UnaryOperator::LogicalNot,
                        EmittedExpr::call("chelis_string_eq", [arg(0), arg(1)]),
                    ),
                    _ => binary(BinaryOperator::NotEqual, arg(0), arg(1)),
                },
                CExpressionBuiltin::And => binary(BinaryOperator::LogicalAnd, arg(0), arg(1)),
                CExpressionBuiltin::Or => binary(BinaryOperator::LogicalOr, arg(0), arg(1)),
                CExpressionBuiltin::Not => unary(UnaryOperator::LogicalNot, arg(0)),
                CExpressionBuiltin::Neg => unary(UnaryOperator::Negate, arg(0)),
                CExpressionBuiltin::StringConcat => {
                    EmittedExpr::call("chelis_string_concat", [arg(0), arg(1)])
                }
                CExpressionBuiltin::StringTrim => EmittedExpr::call("chelis_string_trim", [arg(0)]),
                CExpressionBuiltin::Reshape => {
                    EmittedExpr::call("chelis_host_reshape_tensor", [arg(0), arg(1)])
                }
                CExpressionBuiltin::StringSlice => {
                    EmittedExpr::call("chelis_string_slice", [arg(0), arg(1), arg(2)])
                }
                CExpressionBuiltin::StringContains => {
                    EmittedExpr::call("chelis_string_contains", [arg(0), arg(1)])
                }
                CExpressionBuiltin::StringStartsWith => {
                    EmittedExpr::call("chelis_string_starts_with", [arg(0), arg(1)])
                }
                CExpressionBuiltin::StringEndsWith => {
                    EmittedExpr::call("chelis_string_ends_with", [arg(0), arg(1)])
                }
                CExpressionBuiltin::StringLen => EmittedExpr::call("chelis_string_len", [arg(0)]),
                CExpressionBuiltin::ToString => match &arg_vars[0].1 {
                    HostType::Int8 | HostType::Int16 | HostType::Int32 | HostType::Int64 => {
                        EmittedExpr::call("chelis_string_from_int64", [arg(0)])
                    }
                    // f32 promotes to double for formatting (lossless); there is
                    // no separate f32 formatter in the runtime.
                    HostType::Float64 | HostType::Float32 => {
                        EmittedExpr::call("chelis_string_from_f64", [arg(0)])
                    }
                    HostType::Bool => EmittedExpr::call("chelis_string_from_bool", [arg(0)]),
                    HostType::String => arg(0),
                    // chelis#730 Phase 1 (census row 3, chelis#734): to_string
                    // of a tensor/list/other non-scalar has no C rendering yet;
                    // it previously compiled to the literal placeholder string
                    // `<value>`. Real rendering arrives with chelis#732's
                    // generated formatter.
                    other => {
                        return Err(Unsupported::new(
                            UnsupportedKind::Construct(format!(
                                "`to_string` of a `{other:?}`-typed value"
                            )),
                            "`chelis build` host emission",
                            Stage::Codegen("c"),
                            "the compiled lane stringifies int64/f32/f64/bool/string scalars \
                         only today; tensor/list rendering is tracked by chelis#732 \
                         (was the `<value>` placeholder, chelis#734)",
                        ));
                    }
                },
                CExpressionBuiltin::ToInt => EmittedExpr::call("chelis_parse_int64", [arg(0)]),
                CExpressionBuiltin::ToFloat => EmittedExpr::call("chelis_parse_f64", [arg(0)]),
                CExpressionBuiltin::TensorToScalar => {
                    EmittedExpr::call("chelis_tensor_to_f64", [arg(0)])
                }
                // Issue #300: dispatch on the *result* tensor precision, not just
                // the (coarse) argument host type. `scalar_to_tensor(cast(c,
                // f32))` must materialize an f32-backed rank-0 tensor: the f64
                // constructor stores 8 bytes, and an f32 consumer (e.g. a DAG
                // `expand` helper lowered at the operand's f32 precision) then
                // decodes the low 4 bytes -- 0.0 for an exactly-representable
                // value like 2.5. `Float64` host-classifies both f32 and f64, so
                // the argument type alone cannot distinguish them; the result
                // `ty` carries the real precision.
                CExpressionBuiltin::ScalarToTensor => match ty {
                    HostType::Tensor(tensor_ty) => match tensor_ty.precision {
                        Prim::Int64 => EmittedExpr::call("chelis_scalar_tensor_from_i64", [arg(0)]),
                        Prim::F64 => EmittedExpr::call("chelis_scalar_tensor_from_f64", [arg(0)]),
                        Prim::F32 => EmittedExpr::call("chelis_scalar_tensor_from_f32", [arg(0)]),
                        precision => {
                            return Err(Unsupported::new(
                                UnsupportedKind::HostType(format!(
                                    "scalar_to_tensor<{}>",
                                    precision.name()
                                )),
                                "`scalar_to_tensor` C host emission",
                                Stage::Codegen("c"),
                                "the resolved result dtype has no scalar-tensor constructor; \
                             implement the exact target capability instead of selecting f32 \
                             ([05-UNS-1]; chelis#714, chelis#729)",
                            ));
                        }
                    },
                    other => {
                        return Err(invalid_abi_shape(
                            format!("scalar_to_tensor carries non-tensor result type `{other:?}`"),
                            "scalar_to_tensor",
                        ));
                    }
                },
                CExpressionBuiltin::Len => match arg_vars[0].1 {
                    HostType::Dict(_, _) => EmittedExpr::call("chelis_dict_len", [arg(0)]),
                    HostType::List(_) => EmittedExpr::call("chelis_list_len", [arg(0)]),
                    ref other => {
                        return Err(invalid_abi_shape(
                            format!("len carries non-container argument type `{other:?}`"),
                            "len builtin",
                        ));
                    }
                },
                CExpressionBuiltin::Range => {
                    EmittedExpr::call("chelis_range_i64", [arg(0), arg(1)])
                }
                CExpressionBuiltin::Rank => EmittedExpr::call("chelis_tensor_rank", [arg(0)]),
                CExpressionBuiltin::Shape => {
                    EmittedExpr::call("chelis_tensor_shape", [arg(0), arg(1)])
                }
                CExpressionBuiltin::Numel => EmittedExpr::call("chelis_tensor_numel", [arg(0)]),
                // Scalar math — these run on host `double` values in lowered
                // closures (e.g. the per-element GELU / RMSNorm map bodies).
                // The RISC DAG variants of these ops are handled separately in
                // `emit.rs`, but when a Surf `def` body is routed through the
                // host interpreter, we need the libm names directly.
                CExpressionBuiltin::Sqrt => EmittedExpr::call("sqrt", [arg(0)]),
                CExpressionBuiltin::Exp => EmittedExpr::call("exp", [arg(0)]),
                CExpressionBuiltin::Log => EmittedExpr::call("log", [arg(0)]),
                CExpressionBuiltin::Sin => EmittedExpr::call("sin", [arg(0)]),
                CExpressionBuiltin::Cos => EmittedExpr::call("cos", [arg(0)]),
                CExpressionBuiltin::Tanh => EmittedExpr::call("tanh", [arg(0)]),
                CExpressionBuiltin::Pow => EmittedExpr::call("pow", [arg(0), arg(1)]),
                CExpressionBuiltin::Abs => match arg_vars[0].1 {
                    HostType::Int8 | HostType::Int16 | HostType::Int32 | HostType::Int64 => {
                        EmittedExpr::call("llabs", [arg(0)])
                    }
                    HostType::Float32 | HostType::Float64 => EmittedExpr::call("fabs", [arg(0)]),
                    ref other => {
                        return Err(invalid_abi_shape(
                            format!("abs carries non-numeric argument type `{other:?}`"),
                            "abs builtin",
                        ));
                    }
                },
                CExpressionBuiltin::Min => EmittedExpr::call("fmin", [arg(0), arg(1)]),
                CExpressionBuiltin::Max => EmittedExpr::call("fmax", [arg(0), arg(1)]),
            };
            Ok(expr)
        };
        let expr = build_expression()?;
        self.lines
            .push(format!("{}{target} = {};", self.indent, expr.as_c()));
        if matches!(ty, HostType::Unit) {
            self.lines.push(format!("{}{target} = 0;", self.indent));
        }
        Ok(())
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
    ) -> Result<(), Unsupported> {
        if let Some(host_helper) = self.tensor_helpers.get(helper) {
            match host_helper.specialization.as_ref() {
                Some(HostTensorSpecialization::BlasMatmul(summary)) => {
                    self.assign_blas_matmul_summary(target, summary, args, ty)?;
                    return Ok(());
                }
                Some(HostTensorSpecialization::SparseGather(summary)) => {
                    self.assign_sparse_summary(
                        target,
                        SparseSummaryKind::Gather,
                        summary,
                        args,
                        ty,
                    )?;
                    return Ok(());
                }
                Some(HostTensorSpecialization::SparseScatterAdd(summary)) => {
                    self.assign_sparse_summary(
                        target,
                        SparseSummaryKind::ScatterAdd,
                        summary,
                        args,
                        ty,
                    )?;
                    return Ok(());
                }
                Some(HostTensorSpecialization::SparseScatterReplace(summary)) => {
                    self.assign_sparse_summary(
                        target,
                        SparseSummaryKind::ScatterReplace,
                        summary,
                        args,
                        ty,
                    )?;
                    return Ok(());
                }
                None => {}
            }
        }

        let helper_name = format!("{}__tensor_{helper}", self.helper_prefix);
        let mut tensor_args: Vec<(String, Option<String>)> = Vec::with_capacity(args.len());
        for (index, arg) in args.iter().enumerate() {
            let inferred_ty = host_type(arg);
            let entry = if matches!(inferred_ty, HostType::Tensor(_)) {
                let arg_name = self.next_temp(&format!("tensor_arg{index}"));
                self.emit_expr_to_var(arg, &arg_name, &inferred_ty)?;
                (arg_name, None)
            } else {
                let value_name = self.next_temp(&format!("tensor_scalar{index}"));
                self.emit_expr_to_var(arg, &value_name, &inferred_ty)?;
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
                    HostType::Int8 => (
                        "CHELIS_I8",
                        format!("((int8_t*){tensor_name}->data)[0] = {value_name};"),
                    ),
                    HostType::Int16 => (
                        "CHELIS_I16",
                        format!("((int16_t*){tensor_name}->data)[0] = {value_name};"),
                    ),
                    HostType::Int64 => (
                        "CHELIS_I64",
                        format!("((int64_t*){tensor_name}->data)[0] = {value_name};"),
                    ),
                    HostType::Bool => (
                        "CHELIS_BOOL",
                        format!("((float*){tensor_name}->data)[0] = {value_name} ? 1.0f : 0.0f;"),
                    ),
                    // #381: an f64 captured scalar (e.g. `cast(1.1, f64)`)
                    // fed to a tensor helper via `scalar_to_tensor` must be
                    // packed into a `CHELIS_F64` rank-0 tensor and written
                    // through a `double*`. The pre-fix catch-all packed it
                    // as `CHELIS_F32` and stored only the low 4 bytes; the
                    // f64 kernel then read 8 bytes (the high 4 garbage),
                    // collapsing the value to ~0 and silently disagreeing
                    // with the evaluator. Float32 still uses the f32 arm.
                    HostType::Float64 => (
                        "CHELIS_F64",
                        format!("((double*){tensor_name}->data)[0] = (double)({value_name});"),
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
            };
            tensor_args.push(entry);
        }
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
                    self.box_value_expr(&slot_expr, &elem_ty)?
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
        Ok(())
    }

    fn assign_blas_matmul_summary(
        &mut self,
        target: &str,
        summary: &HostBlasMatmulSummary,
        args: &[HostExpr],
        _ty: &HostType,
    ) -> Result<(), Unsupported> {
        assert_eq!(
            args.len(),
            summary.input_tys.len(),
            "BLAS summary argument count must match callsite argument count"
        );
        let mut tensor_args: Vec<String> = Vec::with_capacity(args.len());
        for (index, arg) in args.iter().enumerate() {
            let arg_name = self.next_temp(&format!("blas_arg{index}"));
            let expected_ty = HostType::Tensor(
                summary
                    .input_tys
                    .get(index)
                    .expect("summary input type")
                    .clone(),
            );
            self.emit_expr_to_var(arg, &arg_name, &expected_ty)?;
            tensor_args.push(arg_name);
        }
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
        Ok(())
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
    ) -> Result<(), Unsupported> {
        assert_eq!(
            args.len(),
            summary.input_tys.len(),
            "sparse summary argument count must match callsite argument count"
        );
        let mut tensor_args: Vec<String> = Vec::with_capacity(args.len());
        for (index, arg) in args.iter().enumerate() {
            let arg_name = self.next_temp(&format!("sparse_arg{index}"));
            let expected_ty = HostType::Tensor(
                summary
                    .input_tys
                    .get(index)
                    .expect("sparse summary input type")
                    .clone(),
            );
            self.emit_expr_to_var(arg, &arg_name, &expected_ty)?;
            tensor_args.push(arg_name);
        }
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
        Ok(())
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
    ) -> Result<(), Unsupported> {
        if let Some(spec) = self.function_specializations.get(function).cloned() {
            match spec {
                HostFunctionSpecialization::BlasMatmul(summary) => {
                    self.assign_blas_matmul_summary(target, &summary, args, ty)?;
                    return Ok(());
                }
                HostFunctionSpecialization::SparseGather(summary) => {
                    self.assign_sparse_summary(
                        target,
                        SparseSummaryKind::Gather,
                        &summary,
                        args,
                        ty,
                    )?;
                    return Ok(());
                }
                HostFunctionSpecialization::SparseScatterAdd(summary) => {
                    self.assign_sparse_summary(
                        target,
                        SparseSummaryKind::ScatterAdd,
                        &summary,
                        args,
                        ty,
                    )?;
                    return Ok(());
                }
                HostFunctionSpecialization::SparseScatterReplace(summary) => {
                    self.assign_sparse_summary(
                        target,
                        SparseSummaryKind::ScatterReplace,
                        &summary,
                        args,
                        ty,
                    )?;
                    return Ok(());
                }
            }
        }

        let mut arg_vars: Vec<String> = Vec::with_capacity(args.len());
        for (index, arg) in args.iter().enumerate() {
            let arg_name = self.next_temp(&format!("call_arg{index}"));
            let expected_ty = arg_tys.get(index).ok_or_else(|| {
                invalid_abi_shape(
                    format!(
                        "call `{function}` has {} arguments but only {} ABI argument types",
                        args.len(),
                        arg_tys.len()
                    ),
                    "user-function call",
                )
            })?;
            self.emit_expr_to_var(arg, &arg_name, expected_ty)?;
            arg_vars.push(arg_name);
        }
        if arg_tys.len() != args.len() {
            return Err(invalid_abi_shape(
                format!(
                    "call `{function}` has {} arguments but {} ABI argument types",
                    args.len(),
                    arg_tys.len()
                ),
                "user-function call",
            ));
        }
        self.lines.push(format!(
            "{}{target} = {}({});",
            self.indent,
            self.emitted_names
                .get(function)
                .map(String::as_str)
                .unwrap_or(function),
            arg_vars.join(", ")
        ));
        self.retain_call_escaped_args(target, function, args, ty);
        Ok(())
    }

    /// Issue #406 (call-escape): when a block result is produced by a call
    /// whose return may alias one of its arguments, and that argument is a
    /// bare `Var` naming a heap binding some open `let` block frees at its
    /// close, the call's result `target` shares the binding's allocation
    /// and the block release would drop the reference the caller now
    /// holds. Retain `target` once per such escaping argument so the
    /// block's release leaves exactly one live reference (the same
    /// retain-cancels-release balance the bare-`Var` transfer arm uses).
    ///
    /// Precision: the per-function `returns_arg` summary
    /// ([`analyze_returns_arg`]) determines which argument positions the
    /// callee may return, so a call that demonstrably builds a fresh
    /// result (does not return the argument) retains nothing and does not
    /// over-retain. A callee absent from the summary (an opaque /
    /// not-user-defined call) is treated as may-return-any: the retain
    /// then fires, which is use-after-free-safe and at worst leaks one
    /// reference. Tensors and other non-refcounted types have no
    /// `retain_call` and are skipped, keeping them excluded as before.
    fn retain_call_escaped_args(
        &mut self,
        target: &str,
        function: &str,
        args: &[HostExpr],
        ty: &HostType,
    ) {
        // Only meaningful for a refcounted result with a retain primitive
        // and at least one open release-tracking `let` block.
        if retain_call(target, ty).is_none() || self.let_scopes.is_empty() {
            return;
        }
        let callee = self.returns_arg.get(function).cloned();
        let mut retained = false;
        for (index, arg) in args.iter().enumerate() {
            // Only a bare `Var` directly aliases a binding's allocation.
            // A more complex argument expression is materialized into a
            // fresh temp (and, if it transferred a binding, already
            // retained by the bare-`Var` arm or another call-escape
            // retain when it was built), so it does not need a retain
            // here.
            let HostExprKind::Var(name, _) = &arg.kind else {
                continue;
            };
            let source_is_binding = self
                .let_scopes
                .iter()
                .any(|scope| scope.bindings.contains(name));
            if !source_is_binding {
                continue;
            }
            let may_return = match &callee {
                Some(summary) => summary.may_return(index),
                // Opaque callee: conservatively assume the argument may
                // escape through the return (UAF-safe over-retain).
                None => true,
            };
            if may_return {
                retained = true;
            }
        }
        // Retain at most once: the result is a single pointer, and one
        // extra reference cancels the one block release that would
        // otherwise drop the escaping allocation. (Even if several
        // arguments alias the *same* binding, the block releases that
        // binding exactly once, so a single retain restores the balance.)
        if retained && let Some(call) = retain_call(target, ty) {
            self.lines.push(format!("{}{call}", self.indent));
        }
    }

    fn assign_adt_construct(
        &mut self,
        target: &str,
        ctor: &str,
        fields: &[HostExpr],
        _ty: &HostType,
    ) -> Result<(), Unsupported> {
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
                self.emit_expr_to_var(field, &field_var, &field_ty)?;
                self.lines.push(format!(
                    "{}{}[{index}] = {};",
                    self.indent,
                    values_name,
                    self.box_value_expr(&field_var, &field_ty)?
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
        Ok(())
    }

    fn assign_adt_field_access(
        &mut self,
        target: &str,
        base: &HostExpr,
        field_index: usize,
        ty: &HostType,
    ) -> Result<(), Unsupported> {
        let base_var = self.next_temp("adt_base");
        self.emit_expr_to_var(base, &base_var, &host_type(base))?;
        let value_var = self.next_temp("adt_field");
        self.lines.push(format!(
            "{}chelis_value {} = chelis_adt_get_field({}, {});",
            self.indent, value_var, base_var, field_index
        ));
        self.assign_unboxed_value(target, ty, &value_var)?;
        Ok(())
    }

    fn assign_match_adt(
        &mut self,
        target: &str,
        scrutinee: &HostExpr,
        arms: &[HostMatchArm],
        default_expr: Option<&HostExpr>,
        expr_ty: &HostType,
    ) -> Result<(), Unsupported> {
        let scrutinee_var = self.next_temp("adt");
        self.emit_expr_to_var(scrutinee, &scrutinee_var, &host_type(scrutinee))?;
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
                    c_type(&binding.ty)?,
                    binding.name
                ));
                self.assign_unboxed_value(&binding.name, &binding.ty, &field_var)?;
            }
            self.assign_expr(target, &arm.expr, expr_ty)?;
            self.indent = previous;
            self.lines.push(format!("{}}}", self.indent));
        }
        self.lines.push(format!("{}else {{", self.indent));
        let nested_indent = format!("{}    ", self.indent);
        let previous = std::mem::replace(&mut self.indent, nested_indent);
        if let Some(default_expr) = default_expr {
            self.assign_expr(target, default_expr, expr_ty)?;
        } else {
            self.lines.push(format!(
                "{}fprintf(stderr, \"non-exhaustive ADT match\\n\");",
                self.indent
            ));
            self.lines.push(format!("{}exit(1);", self.indent));
        }
        self.indent = previous;
        self.lines.push(format!("{}}}", self.indent));
        Ok(())
    }

    fn assign_list_literal(
        &mut self,
        target: &str,
        items: &[HostExpr],
        ty: &HostType,
    ) -> Result<(), Unsupported> {
        let HostType::List(item_ty) = ty else {
            return Err(invalid_abi_shape(
                format!("list literal carries non-list ABI type `{ty:?}`"),
                "list literal",
            ));
        };
        if items.is_empty() {
            self.lines
                .push(format!("{}{target} = chelis_list_empty();", self.indent));
            return Ok(());
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
            self.emit_expr_to_var(item, &item_var, item_ty)?;
            self.lines.push(format!(
                "{}{}[{index}] = {};",
                self.indent,
                values_name,
                self.box_value_expr(&item_var, item_ty)?
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
        self.track_owned_alloc(target, ty);
        Ok(())
    }

    fn assign_tuple_literal(
        &mut self,
        target: &str,
        items: &[HostExpr],
        ty: &HostType,
    ) -> Result<(), Unsupported> {
        let HostType::Tuple(item_tys) = ty else {
            return Err(invalid_abi_shape(
                format!("tuple literal carries non-tuple ABI type `{ty:?}`"),
                "tuple literal",
            ));
        };
        if item_tys.len() != items.len() {
            return Err(invalid_abi_shape(
                format!(
                    "tuple literal has {} items but its ABI type has {} fields",
                    items.len(),
                    item_tys.len()
                ),
                "tuple literal",
            ));
        }
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
                let item_ty = &item_tys[index];
                self.emit_expr_to_var(item, &item_var, item_ty)?;
                self.lines.push(format!(
                    "{}{}[{index}] = {};",
                    self.indent,
                    values_name,
                    self.box_value_expr(&item_var, item_ty)?
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
        self.track_owned_alloc(target, ty);
        Ok(())
    }

    fn assign_map(
        &mut self,
        target: &str,
        callback: &HostCallback,
        list: &HostExpr,
        _ty: &HostType,
    ) -> Result<(), Unsupported> {
        let list_var = self.next_temp("map_list");
        self.emit_expr_to_var(list, &list_var, &host_type(list))?;
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
            c_type(&callback.ret_ty)?,
            result_var
        ));
        let param = callback_param(callback, 0);
        let arg_var = self.next_temp("map_item");
        self.lines.push(format!(
            "{}{} {};",
            self.indent,
            c_type(&param.ty)?,
            arg_var
        ));
        self.assign_unboxed_value(&arg_var, &param.ty, &item_value)?;
        self.emit_callback_assign(callback, std::slice::from_ref(&arg_var), &result_var)?;
        self.lines.push(format!(
            "{}{target} = chelis_list_append({target}, {});",
            self.indent,
            self.box_value_expr(&result_var, &callback.ret_ty)?
        ));
        self.indent = previous;
        self.lines.push(format!("{}}}", self.indent));
        Ok(())
    }

    fn assign_filter(
        &mut self,
        target: &str,
        callback: &HostCallback,
        list: &HostExpr,
        _ty: &HostType,
    ) -> Result<(), Unsupported> {
        let list_var = self.next_temp("filter_list");
        self.emit_expr_to_var(list, &list_var, &host_type(list))?;
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
        self.lines.push(format!(
            "{}{} {};",
            self.indent,
            c_type(&param.ty)?,
            arg_var
        ));
        self.assign_unboxed_value(&arg_var, &param.ty, &item_value)?;
        self.emit_callback_assign(callback, std::slice::from_ref(&arg_var), &keep_var)?;
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
        Ok(())
    }

    fn assign_fold(
        &mut self,
        target: &str,
        callback: &HostCallback,
        init: &HostExpr,
        list: &HostExpr,
        ty: &HostType,
    ) -> Result<(), Unsupported> {
        self.assign_expr(target, init, ty)?;
        let list_var = self.next_temp("fold_list");
        self.emit_expr_to_var(list, &list_var, &host_type(list))?;
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
            c_type(&params[0].ty)?,
            acc_arg
        ));
        let item_arg = self.next_temp("fold_item");
        self.lines.push(format!(
            "{}{} {};",
            self.indent,
            c_type(&params[1].ty)?,
            item_arg
        ));
        self.assign_unboxed_value(&item_arg, &params[1].ty, &item_value)?;
        self.emit_callback_assign(callback, &[acc_arg, item_arg], target)?;
        self.indent = previous;
        self.lines.push(format!("{}}}", self.indent));
        Ok(())
    }

    fn assign_scan(
        &mut self,
        target: &str,
        callback: &HostCallback,
        init: &HostExpr,
        list: &HostExpr,
        ty: &HostType,
    ) -> Result<(), Unsupported> {
        let HostType::List(inner_ty) = ty else {
            self.lines
                .push(format!("{}{target} = chelis_list_empty();", self.indent));
            return Ok(());
        };
        let acc_ty = inner_ty.as_ref().clone();
        let acc_var = self.next_temp("scan_acc");
        self.emit_expr_to_var(init, &acc_var, &acc_ty)?;
        self.lines
            .push(format!("{}{target} = chelis_list_empty();", self.indent));
        let list_var = self.next_temp("scan_list");
        self.emit_expr_to_var(list, &list_var, &host_type(list))?;
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
            c_type(&params[0].ty)?,
            acc_arg,
            acc_var
        ));
        let item_arg = self.next_temp("scan_item");
        self.lines.push(format!(
            "{}{} {};",
            self.indent,
            c_type(&params[1].ty)?,
            item_arg
        ));
        self.assign_unboxed_value(&item_arg, &params[1].ty, &item_value)?;
        self.emit_callback_assign(callback, &[acc_arg, item_arg], &acc_var)?;
        self.lines.push(format!(
            "{}{target} = chelis_list_append({target}, {});",
            self.indent,
            self.box_value_expr(&acc_var, &acc_ty)?
        ));
        self.indent = previous;
        self.lines.push(format!("{}}}", self.indent));
        Ok(())
    }

    fn assign_partition(
        &mut self,
        target: &str,
        callback: &HostCallback,
        list: &HostExpr,
        ty: &HostType,
    ) -> Result<(), Unsupported> {
        let HostType::Tuple(parts) = ty else {
            // chelis#730 Phase 1 (census row 15, section C1.4
            // raise-or-prove): a `partition` whose result type is not a
            // tuple is an internal desync (the checker types partition as
            // a two-list tuple; see `partition_agrees_across_lanes` for
            // the reachable-surface clearance). Previously emitted a bare
            // C comment and NO assignment - garbage C downstream.
            return Err(Unsupported::new(
                UnsupportedKind::Construct(format!(
                    "a `partition` result typed `{ty:?}` instead of a tuple"
                )),
                "`chelis build` host emission",
                Stage::Codegen("c"),
                "internal desync: the checker guarantees a two-list tuple type for \
                 partition results (chelis#730 census row 15)",
            ));
        };
        let [pass_ty, fail_ty] = parts.as_slice() else {
            return Err(invalid_abi_shape(
                format!(
                    "partition result must have exactly two fields, found {}",
                    parts.len()
                ),
                "partition result",
            ));
        };
        let pass_var = self.next_temp("partition_pass");
        let fail_var = self.next_temp("partition_fail");
        self.lines.push(format!(
            "{}{} {} = chelis_list_empty();",
            self.indent,
            c_type(pass_ty)?,
            pass_var
        ));
        self.lines.push(format!(
            "{}{} {} = chelis_list_empty();",
            self.indent,
            c_type(fail_ty)?,
            fail_var
        ));
        let list_var = self.next_temp("partition_list");
        self.emit_expr_to_var(list, &list_var, &host_type(list))?;
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
        self.lines.push(format!(
            "{}{} {};",
            self.indent,
            c_type(&param.ty)?,
            arg_var
        ));
        self.assign_unboxed_value(&arg_var, &param.ty, &item_value)?;
        self.emit_callback_assign(callback, std::slice::from_ref(&arg_var), &keep_var)?;
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
        Ok(())
    }

    fn assign_flat_map(
        &mut self,
        target: &str,
        callback: &HostCallback,
        list: &HostExpr,
        _ty: &HostType,
    ) -> Result<(), Unsupported> {
        let list_var = self.next_temp("flat_map_list");
        self.emit_expr_to_var(list, &list_var, &host_type(list))?;
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
            c_type(&callback.ret_ty)?,
            result_var
        ));
        let param = callback_param(callback, 0);
        let arg_var = self.next_temp("flat_map_item");
        self.lines.push(format!(
            "{}{} {};",
            self.indent,
            c_type(&param.ty)?,
            arg_var
        ));
        self.assign_unboxed_value(&arg_var, &param.ty, &item_value)?;
        self.emit_callback_assign(callback, std::slice::from_ref(&arg_var), &result_var)?;
        self.lines.push(format!(
            "{}{target} = chelis_list_concat({target}, {});",
            self.indent, result_var
        ));
        self.indent = previous;
        self.lines.push(format!("{}}}", self.indent));
        Ok(())
    }

    fn emit_callback_assign(
        &mut self,
        callback: &HostCallback,
        arg_vars: &[String],
        target: &str,
    ) -> Result<(), Unsupported> {
        match &callback.kind {
            HostCallbackKind::Named { function, .. } => {
                self.lines.push(format!(
                    "{}{target} = {}({});",
                    self.indent,
                    // chelis#840: same original-to-emitted mapping as
                    // `assign_call`, so a mangled or renamed def is
                    // referenced consistently from callback position.
                    self.emitted_names
                        .get(function)
                        .map(String::as_str)
                        .unwrap_or(function),
                    arg_vars.join(", ")
                ));
            }
            HostCallbackKind::Inline { params, body } => {
                for (param, arg_var) in params.iter().zip(arg_vars.iter()) {
                    self.lines.push(format!(
                        "{}{} {} = {};",
                        self.indent,
                        c_type(&param.ty)?,
                        param.name,
                        arg_var
                    ));
                }
                self.assign_expr(target, body, &callback.ret_ty)?;
            }
        }
        Ok(())
    }

    fn box_value_expr(&self, value: &str, ty: &HostType) -> Result<String, Unsupported> {
        Ok(match ty {
            HostType::Int8 | HostType::Int16 | HostType::Int32 | HostType::Int64 => {
                format!("chelis_value_from_int64((int64_t){value})")
            }
            // The boxed value option has no separate f32 slot; an f32
            // promotes losslessly to the f64 box (WS-4).
            HostType::Float64 | HostType::Float32 => format!("chelis_value_from_f64({value})"),
            HostType::Bool => format!("chelis_value_from_bool({value})"),
            HostType::String => format!("chelis_value_from_string({value})"),
            HostType::Adt(_, _) => format!("chelis_value_from_adt({value})"),
            HostType::Tensor(_) => format!("chelis_value_from_tensor({value})"),
            HostType::List(_) => format!("chelis_value_from_list({value})"),
            HostType::Tuple(_) => format!("chelis_value_from_tuple({value})"),
            HostType::Dict(_, _) => format!("chelis_value_from_dict({value})"),
            HostType::Callback(_, _)
            | HostType::Option(_)
            | HostType::MappedFile
            | HostType::Unit => {
                return Err(unsupported_value_boxing(ty, "boxing a resolved host value"));
            }
        })
    }

    fn assign_unboxed_value(
        &mut self,
        target: &str,
        ty: &HostType,
        value_expr: &str,
    ) -> Result<(), Unsupported> {
        let expr = match ty {
            HostType::Int8 | HostType::Int16 | HostType::Int32 | HostType::Int64 => {
                format!("({})chelis_value_as_int64({value_expr})", c_type(ty)?)
            }
            // f32 is unboxed via the f64 accessor (the box stored it as
            // f64); the surrounding `c_decl` narrows back to `float` (WS-4).
            HostType::Float64 | HostType::Float32 => {
                format!("chelis_value_as_f64({value_expr})")
            }
            HostType::Bool => format!("chelis_value_as_bool({value_expr})"),
            HostType::String => format!("chelis_value_as_string({value_expr})"),
            HostType::Adt(_, _) => format!("chelis_value_as_adt({value_expr})"),
            HostType::Tensor(_) => format!("chelis_value_as_tensor({value_expr})"),
            HostType::List(_) => format!("chelis_value_as_list({value_expr})"),
            HostType::Tuple(_) => format!("chelis_value_as_tuple({value_expr})"),
            HostType::Dict(_, _) => format!("chelis_value_as_dict({value_expr})"),
            HostType::Callback(_, _)
            | HostType::Option(_)
            | HostType::MappedFile
            | HostType::Unit => {
                return Err(unsupported_value_boxing(
                    ty,
                    "unboxing a resolved host value",
                ));
            }
        };
        self.lines
            .push(format!("{}{target} = {expr};", self.indent));
        Ok(())
    }

    fn emit_print_value(&mut self, value: &str, ty: &HostType) -> Result<(), Unsupported> {
        match ty {
            HostType::String => self.lines.push(format!(
                "{}printf(\"%s\\n\", chelis_string_data({}));",
                self.indent, value
            )),
            HostType::Int8 | HostType::Int16 | HostType::Int32 | HostType::Int64 => {
                self.lines.push(format!(
                    "{}printf(\"%lld\\n\", (long long){});",
                    self.indent, value
                ))
            }
            HostType::Float64 | HostType::Float32 => self
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
            // chelis#730 Phase 1 (census row 4, chelis#714 symptom): an
            // unclassifiable value at a print site is a compiler bug
            // surfaced at emit time, never the literal `<value>` text.
            other => {
                return Err(Unsupported::new(
                    UnsupportedKind::HostType(format!("{other:?}")),
                    "a `print` site in `chelis build` host emission",
                    Stage::Codegen("c"),
                    "the value's host type never resolved to a printable representation \
                     (chelis#714's Unknown chain); previously this compiled to the \
                     literal `<value>` placeholder",
                ));
            }
        }
        Ok(())
    }

    fn emit_labeled_root(
        &mut self,
        name: &str,
        value: &str,
        ty: &HostType,
    ) -> Result<(), Unsupported> {
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
                    c_decl(field_ty, &field_value)?
                ));
                self.assign_unboxed_value(&field_value, field_ty, &field_var)?;
                self.emit_labeled_root(&field_name, &field_value, field_ty)?;
                // issue #406: `chelis_tuple_get` retains the boxed element
                // it returns (a no-op for scalar fields). The labeled-root
                // printer only reads it, so release the retained handle once
                // the field has been printed — otherwise a tuple field that
                // is itself a heap container (a nested tuple/list/adt) leaks
                // that reference for the process lifetime.
                self.lines
                    .push(format!("{}chelis_value_release({field_var});", self.indent));
            }
            return Ok(());
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
            HostType::Int8 | HostType::Int16 | HostType::Int32 | HostType::Int64 => {
                self.lines.push(format!(
                    "{}printf(\"%lld\", (long long){});",
                    self.indent, value
                ))
            }
            HostType::Float64 | HostType::Float32 => self
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
            // chelis#730 Phase 1 (census row 4): same contract as
            // `emit_print_value` - an unclassifiable labeled root is a
            // surfaced compiler bug, not a `<value>` placeholder.
            other => {
                return Err(Unsupported::new(
                    UnsupportedKind::HostType(format!("{other:?}")),
                    "a labeled-root print in `chelis build` host emission",
                    Stage::Codegen("c"),
                    "the value's host type never resolved to a printable representation \
                     (chelis#714's Unknown chain); previously this compiled to the \
                     literal `<value>` placeholder",
                ));
            }
        }
        self.lines.push(format!("{}printf(\"\\n\");", self.indent));
        Ok(())
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
    ) -> Result<(), Unsupported> {
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
                    self.box_value_expr(value_var, value_ty)?
                ));
            }
            other => {
                return Err(invalid_abi_shape(
                    format!("Some constructor carries non-option ABI type `{other:?}`"),
                    "Some constructor",
                ));
            }
        }
        Ok(())
    }

    fn assign_option_none(&mut self, target: &str, ty: &HostType) -> Result<(), Unsupported> {
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
            other => {
                return Err(invalid_abi_shape(
                    format!("None constructor carries non-option ABI type `{other:?}`"),
                    "None constructor",
                ));
            }
        }
        Ok(())
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
    // #379: the var may be a user binding/let name spelled like a C keyword;
    // route through `c_ident` so the free call names the same (possibly
    // mangled) identifier the declaration used. Compiler temps
    // (`__binding_N_value`, `__let_N`) pass through unchanged.
    let var = c_ident(var);
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
    // #379: mirror `release_call` — a user name spelled like a C keyword
    // routes through `c_ident`; compiler temps pass through unchanged.
    let var = c_ident(var);
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

fn c_type(ty: &HostAbiType) -> Result<&'static str, Unsupported> {
    ty.c_type_name().ok_or_else(|| {
        invalid_abi_shape(
            format!("callback type {ty:?} used where C requires a standalone value type"),
            "C host value type emission",
        )
    })
}

fn is_integer_abi(ty: &HostAbiType) -> bool {
    matches!(
        ty,
        HostAbiType::Int8 | HostAbiType::Int16 | HostAbiType::Int32 | HostAbiType::Int64
    )
}

/// The C / C++ reserved words a Chelis identifier must not collide with
/// when emitted verbatim. `chelis check` accepts user bindings, params,
/// and `let` names spelled like these (e.g. `register`, `static`, `int`,
/// or `main`), and emitting them raw produces a syntax error or a symbol
/// collision with the generated `int main(void)` (#379). The host C/HIP
/// lane shares this emit, so the list covers C11 keywords plus the C++
/// keywords hipcc rejects. `main` is included because the generated entry
/// point is `int main(void)`.
const C_RESERVED_WORDS: &[&str] = &[
    // C11 keywords
    "auto",
    "break",
    "case",
    "char",
    "const",
    "continue",
    "default",
    "do",
    "double",
    "else",
    "enum",
    "extern",
    "float",
    "for",
    "goto",
    "if",
    "inline",
    "int",
    "long",
    "register",
    "restrict",
    "return",
    "short",
    "signed",
    "sizeof",
    "static",
    "struct",
    "switch",
    "typedef",
    "union",
    "unsigned",
    "void",
    "volatile",
    "while",
    "_Alignas",
    "_Alignof",
    "_Atomic",
    "_Bool",
    "_Complex",
    "_Generic",
    "_Imaginary",
    "_Noreturn",
    "_Static_assert",
    "_Thread_local",
    // C++ keywords the shared HIP host lane (hipcc) also rejects
    "alignas",
    "alignof",
    "and",
    "asm",
    "bool",
    "catch",
    "class",
    "compl",
    "constexpr",
    "const_cast",
    "decltype",
    "delete",
    "dynamic_cast",
    "explicit",
    "export",
    "false",
    "friend",
    "mutable",
    "namespace",
    "new",
    "nullptr",
    "operator",
    "or",
    "private",
    "protected",
    "public",
    "reinterpret_cast",
    "static_cast",
    "template",
    "this",
    "throw",
    "true",
    "try",
    "typeid",
    "typename",
    "using",
    "virtual",
    "wchar_t",
    "xor",
    // The generated entry point
    "main",
    // Typedefs and macros the emitted translation unit includes via
    // stdint/stddef and the chelis runtime headers (chelis#840): a user
    // def, binding, or parameter spelled like one of these shadows or
    // redefines the typedef and the C cannot compile.
    "int8_t",
    "int16_t",
    "int32_t",
    "int64_t",
    "uint8_t",
    "uint16_t",
    "uint32_t",
    "uint64_t",
    "intmax_t",
    "uintmax_t",
    "intptr_t",
    "uintptr_t",
    "size_t",
    "ssize_t",
    "ptrdiff_t",
    "offsetof",
];

/// Prefix applied to a user identifier that would otherwise be illegal or
/// colliding in emitted C. The double underscore keeps it distinct from
/// any plausible user name and from the runtime's `chelis_*` symbols.
const C_USER_IDENT_PREFIX: &str = "chelis_user__";

/// Map a Chelis identifier to a legal, collision-free C identifier (#379).
///
/// Most names pass through byte-identical so the existing C/HIP corpus is
/// unchanged. A name is rewritten only when emitting it verbatim would
/// break compilation:
///   * it is a C/C++ reserved word (`register`, `static`, `main`, ...), or
///   * it collides with the compiler's emitted-helper naming scheme
///     (`{fn}__tensor_{n}`, `{prog}__global__...`), which a user binding
///     can only hit by literally containing `__tensor_` / `__global__`.
///
/// The emitter's OWN temporaries (`__binding_N_value`, `__arg...`,
/// `__result`, `__call_...`, `__let_...`) are generated internally, are
/// already legal C, and are NOT user-controlled, so they must pass through
/// untouched — `c_decl` is called with both user names and these temps.
/// They neither appear in `C_RESERVED_WORDS` nor contain `__tensor_` /
/// `__global__`, so the rules below leave them alone.
///
/// The same mapping must be applied at every site that turns a user name
/// into a C identifier (declaration AND reference) so the two stay
/// consistent; `c_decl` and the `Var`/binding/hoist emit paths all route
/// through here.
fn c_ident(name: &str) -> std::borrow::Cow<'_, str> {
    // Every emitter temporary (`__binding_N_value`, `__arg...`, `__result`,
    // `__call_...`, `__let_...`, `__tensor_argN_M`, `__host_tensor_arg_N`)
    // begins with `__`. A user identifier from Chelis source never does
    // (Surf/Deep identifiers cannot start with `__`), so a leading `__`
    // marks a name as compiler-internal and already-legal: leave it alone.
    // This is what keeps the helper-scheme check below from rewriting the
    // `__tensor_arg*` argument temps (which contain `__tensor_`).
    if name.starts_with("__") {
        return std::borrow::Cow::Borrowed(name);
    }
    // A user binding can only collide with the emitted-helper FUNCTION
    // naming scheme (`{fn}__tensor_{n}`, `{prog}__global__...`) by literally
    // containing those infixes; such names do not start with `__`.
    let collides_with_helper_scheme = name.contains("__tensor_") || name.contains("__global__");
    if C_RESERVED_WORDS.contains(&name) || collides_with_helper_scheme {
        std::borrow::Cow::Owned(format!("{C_USER_IDENT_PREFIX}{name}"))
    } else {
        std::borrow::Cow::Borrowed(name)
    }
}

fn c_decl(ty: &HostType, name: &str) -> Result<String, Unsupported> {
    let name = c_ident(name);
    match ty {
        HostType::Callback(params, ret) => {
            let args = if params.is_empty() {
                "void".to_string()
            } else {
                params
                    .iter()
                    .map(c_type)
                    .collect::<Result<Vec<_>, _>>()?
                    .join(", ")
            };
            Ok(format!("{} (*{})({})", c_type(ret)?, name, args))
        }
        _ => Ok(format!("{} {}", c_type(ty)?, name)),
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

fn invalid_abi_shape(detail: String, context: &'static str) -> Unsupported {
    Unsupported::new(
        UnsupportedKind::Construct(detail),
        context,
        Stage::Codegen("c"),
        "the resolved host IR and C ABI projection disagree; this is an internal \
         compiler error, never a request to select a fallback representation \
         (chelis#730; [05-UNS-1])",
    )
}

fn unsupported_value_boxing(ty: &HostType, context: &'static str) -> Unsupported {
    Unsupported::new(
        UnsupportedKind::HostType(format!("{ty:?}")),
        context,
        Stage::Codegen("c"),
        "the resolved C-host ABI has no chelis_value boxing representation for this \
         type; implement that representation explicitly or reject the containing \
         construct ([05-UNS-1]; chelis#730, chelis#729)",
    )
}

fn require_same_abi_type(
    expected: &HostType,
    actual: &HostType,
    context: &'static str,
) -> Result<(), Unsupported> {
    if expected == actual {
        Ok(())
    } else {
        Err(invalid_abi_shape(
            format!("expected ABI type `{expected:?}`, found `{actual:?}`"),
            context,
        ))
    }
}

fn option_inner_type(ty: &HostType) -> Result<HostType, Unsupported> {
    match ty {
        HostType::Option(inner) => Ok((**inner).clone()),
        other => Err(invalid_abi_shape(
            format!("option match scrutinee has non-option ABI type `{other:?}`"),
            "option match",
        )),
    }
}

fn expected_builtin_arg_ty(name: &str, ty: &HostType, index: usize) -> Option<HostType> {
    match (name, ty, index) {
        ("Some", HostType::Option(inner), 0) => Some((**inner).clone()),
        ("dict_of", HostType::Dict(key, value), 0) => {
            Some(HostType::List(Box::new(HostType::Tuple(vec![
                (**key).clone(),
                (**value).clone(),
            ]))))
        }
        ("append", HostType::List(inner), 1) => Some((**inner).clone()),
        _ => None,
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
    prim.runtime_dtype()
        .unwrap_or_else(|error| panic!("C backend sparse summary: {error}"))
        .c_macro()
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
        self.runtime_dtype().c_macro()
    }

    fn runtime_dtype(self) -> chelis_vocab::RuntimeDType {
        match self {
            DtypeArm::F32 => chelis_vocab::RuntimeDType::F32,
            DtypeArm::F64 => chelis_vocab::RuntimeDType::F64,
            DtypeArm::I32 => chelis_vocab::RuntimeDType::I32,
            DtypeArm::I64 => chelis_vocab::RuntimeDType::I64,
            DtypeArm::Bool => chelis_vocab::RuntimeDType::Bool,
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

#[cfg(test)]
mod expression_dispatch_tests {
    use super::*;

    #[test]
    fn open_builtin_name_must_decode_before_expression_construction() {
        assert_eq!(
            CExpressionBuiltin::decode("add"),
            Ok(CExpressionBuiltin::Add)
        );
        let error = CExpressionBuiltin::decode("future_unimplemented_builtin")
            .expect_err("an open-set name has no expression identity by default");
        assert_eq!(
            error.what,
            UnsupportedKind::Builtin("future_unimplemented_builtin".into())
        );
    }
}
