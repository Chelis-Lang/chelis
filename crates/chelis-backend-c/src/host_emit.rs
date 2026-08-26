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
    BitAnd,
    BitOr,
    BitXor,
    ShiftLeft,
    ShiftRight,
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
    Tan,
    Atan,
    Tanh,
    Relu,
    Sigmoid,
    Silu,
    Gelu,
    Floor,
    Ceil,
    Round,
    Recip,
    Pow,
    Abs,
    Min,
    Max,
    MinElem,
    MaxElem,
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
            "bitand" => Self::BitAnd,
            "bitor" => Self::BitOr,
            "bitxor" => Self::BitXor,
            "shl" => Self::ShiftLeft,
            "shr" => Self::ShiftRight,
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
            "tan" => Self::Tan,
            "atan" => Self::Atan,
            "tanh" => Self::Tanh,
            "relu" => Self::Relu,
            "sigmoid" => Self::Sigmoid,
            "silu" => Self::Silu,
            "gelu" => Self::Gelu,
            "floor" => Self::Floor,
            "ceil" => Self::Ceil,
            "round" => Self::Round,
            "recip" => Self::Recip,
            "pow" => Self::Pow,
            "abs" => Self::Abs,
            "min" => Self::Min,
            "max" => Self::Max,
            "min_elem" => Self::MinElem,
            "max_elem" => Self::MaxElem,
            other => {
                return Err(Unsupported::new(
                    UnsupportedKind::Builtin(other.to_string()),
                    "`chelis build` host emission",
                    Stage::Codegen("c"),
                    chelis_types::deliberate_rejection!(
                        "[04-TOT-2]",
                        "the checked builtin vocabulary and C expression vocabulary disagree; \
                         no fallback expression is permitted"
                    ),
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
use chelis_types::manifest::RootPathStep;
use chelis_types::types::Prim;
use chelis_types::unsupported::{Stage, Unsupported, UnsupportedKind};
use chelis_types::{CheckedCastKind, CheckedCastPlan, NumericTrap};
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
enum ParamAlias {
    Indices(HashSet<usize>),
    Any,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ReturnsArg {
    params: ParamAlias,
    /// chelis#1222: the result may be a value the caller never handed in --
    /// a top-level binding read as a free variable somewhere in the body
    /// (issue #352 hoists such a binding to file scope, so the body reads
    /// it by name). `main` already owns that allocation through the
    /// binding itself, so a caller that also claimed the call result would
    /// release one allocation twice. Widened exactly like `params`: any
    /// arm, any `let` body, or any callee that can hand back an enclosing
    /// scope's value sets it.
    outer: bool,
}

impl ReturnsArg {
    fn empty() -> Self {
        ReturnsArg {
            params: ParamAlias::Indices(HashSet::new()),
            outer: false,
        }
    }

    /// A result that is an enclosing scope's value rather than a fresh
    /// allocation or one of this function's own parameters (chelis#1222).
    fn outer() -> Self {
        ReturnsArg {
            params: ParamAlias::Indices(HashSet::new()),
            outer: true,
        }
    }

    /// Join two result-alias summaries (the `if`/`match`-arm union or the
    /// fixpoint widening). `Any` absorbs everything; otherwise the index
    /// sets are unioned and the outer-alias flags are or-ed.
    fn join(self, other: ReturnsArg) -> ReturnsArg {
        let params = match (self.params, other.params) {
            (ParamAlias::Any, _) | (_, ParamAlias::Any) => ParamAlias::Any,
            (ParamAlias::Indices(mut a), ParamAlias::Indices(b)) => {
                a.extend(b);
                ParamAlias::Indices(a)
            }
        };
        ReturnsArg {
            params,
            outer: self.outer || other.outer,
        }
    }

    /// Does the result possibly alias parameter index `i`?
    fn may_return(&self, i: usize) -> bool {
        match &self.params {
            ParamAlias::Any => true,
            ParamAlias::Indices(s) => s.contains(&i),
        }
    }

    /// Does the result possibly alias a value owned by an enclosing scope
    /// (chelis#1222)? A caller must not claim ownership of such a result.
    fn may_return_outer(&self) -> bool {
        self.outer
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
            let computed = result_alias_set(
                &function.body,
                &param_index,
                &summary,
                &mut env,
                &function.tensor_helpers,
            );
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
    helpers: &[HostTensorHelper],
) -> ReturnsArg {
    match &expr.kind {
        HostExprKind::Var(name, _) => {
            // chelis#1222: `env` first, `param_index` second. Parameters are
            // the function's outermost scope, so any binder currently in
            // `env` shadows one that reuses its name. Asking `param_index`
            // first made a `let` binder invisible to the analysis: for
            // `def f(p) = { p = g  p }` the body reported "returns parameter
            // 0" instead of `outer`, and the caller then claimed the
            // captured global `g` and released it a second time. The three
            // binder arms below already save and restore what they shadow,
            // so `env` is the authority on what a name means here.
            if let Some(set) = env.get(name) {
                set.clone()
            } else if let Some(&i) = param_index.get(name.as_str()) {
                ReturnsArg {
                    params: ParamAlias::Indices(HashSet::from([i])),
                    outer: false,
                }
            } else if name == "Nil" || name == "None" {
                // Emitted as a fresh empty list / `None` payload, not as a
                // read of an enclosing binding.
                ReturnsArg::empty()
            } else {
                // A free variable: a top-level binding this function
                // captured. The allocation is owned elsewhere (chelis#1222),
                // so the result is borrowed rather than fresh, and a caller
                // that released it would release it a second time.
                ReturnsArg::outer()
            }
        }
        HostExprKind::Let { bindings, body, .. } => {
            // chelis#1222: save what each binder shadows and put it back at
            // the end, the way the `MatchOption` and `MatchAdt` arms below
            // already do. Without this a `let` binder's meaning outlives its
            // block: a sibling branch reading the same NAME finds the inner
            // (fresh) set instead of falling through to `outer()`, the
            // summary reports `may_return_outer() == false` for a function
            // that does return an outer value, and the caller then claims a
            // borrowed result and releases it twice.
            //
            // The insert stays AFTER the value walk: the initializer is
            // evaluated in the enclosing scope and may read the outer
            // meaning of the very name being bound.
            let mut saved: Vec<(String, Option<ReturnsArg>)> = Vec::new();
            for binding in bindings {
                let set = result_alias_set(&binding.value, param_index, summary, env, helpers);
                saved.push((binding.name.clone(), env.insert(binding.name.clone(), set)));
            }
            let result = result_alias_set(body, param_index, summary, env, helpers);
            for (name, prev) in saved.into_iter().rev() {
                match prev {
                    Some(set) => {
                        env.insert(name, set);
                    }
                    None => {
                        env.remove(&name);
                    }
                }
            }
            result
        }
        HostExprKind::If {
            then_expr,
            else_expr,
            ..
        } => {
            let t = result_alias_set(then_expr, param_index, summary, env, helpers);
            let e = result_alias_set(else_expr, param_index, summary, env, helpers);
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
            let s = result_alias_set(some_expr, param_index, summary, env, helpers);
            match prev {
                Some(set) => {
                    env.insert(bind_name.clone(), set);
                }
                None => {
                    env.remove(bind_name);
                }
            }
            let n = result_alias_set(none_expr, param_index, summary, env, helpers);
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
                acc = acc.join(result_alias_set(
                    &arm.expr,
                    param_index,
                    summary,
                    env,
                    helpers,
                ));
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
                acc = acc.join(result_alias_set(
                    default,
                    param_index,
                    summary,
                    env,
                    helpers,
                ));
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
            // chelis#1222: an outer-scope value the callee hands back is
            // still an outer-scope value here. An unsummarized callee is
            // conservatively assumed to do so.
            let mut acc = match callee {
                Some(s) if !s.may_return_outer() => ReturnsArg::empty(),
                _ => ReturnsArg::outer(),
            };
            for (i, arg) in args.iter().enumerate() {
                let returns_this = match callee {
                    Some(s) => s.may_return(i),
                    None => true,
                };
                if returns_this {
                    acc = acc.join(result_alias_set(arg, param_index, summary, env, helpers));
                }
            }
            acc
        }
        HostExprKind::WithSeed { body, .. } => {
            result_alias_set(body, param_index, summary, env, helpers)
        }
        // chelis#1222: an identity tensor helper's whole body is
        // `outputs[0] = inputs[0];` (see `identity_helper_input`), so the
        // call hands back its argument's pointer rather than allocating.
        // Its provenance is the argument's. Every other helper writes a
        // freshly allocated `chelis_contiguous` output, which is why the
        // catch-all below reports a fresh result for the rest.
        HostExprKind::TensorCall { helper, args, .. } => {
            match (
                helpers.get(*helper).and_then(identity_helper_input),
                args.first(),
            ) {
                (Some(_), Some(arg)) => result_alias_set(arg, param_index, summary, env, helpers),
                _ => ReturnsArg::empty(),
            }
        }
        // Constructors, literals, builtins, field access, and the iterator
        // lanes all build fresh allocations whose result does not alias an
        // incoming parameter pointer. (A builtin
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
    append_scalar_conversion_helpers(&mut body);
    body.push(String::new());
    append_tensor_reshape_helper(&mut body);
    body.push(String::new());
    append_tensor_print_helper(&mut body);
    body.push(String::new());
    append_uniform_sample_helper(&mut body);
    body.push(String::new());
    append_tensor_math_helpers(&mut body);
    body.push(String::new());
    // Authored functions are published in the generated header with external
    // linkage. [05-OBS-11] can make the same translation unit executable by
    // adding `main`, but that observation driver must not contradict the
    // published ABI by turning those definitions `static`. Compiler-owned
    // monomorphized specializations remain translation-unit local below.
    let internal_linkage = false;
    let emitted_names = emitted_function_names(program, program_name);
    reject_duplicate_emitted_function_names(&emitted_names)?;
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
            internal_linkage || function.is_monomorphized_specialization(),
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
                    internal_linkage || function.is_monomorphized_specialization(),
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

    // Keep the JSON-only sorting machinery out of unrelated generated
    // translation units. Detect the structured call emitted above, then
    // prepend its definition so C never relies on an implicit declaration.
    let needs_json_canonical_object_helper = body
        .iter()
        .any(|line| line.contains(" = chelis_json_canonical_object_entries("));
    if needs_json_canonical_object_helper {
        let mut json_helpers = Vec::new();
        append_json_canonical_object_helpers(&mut json_helpers);
        json_helpers.push(String::new());
        json_helpers.extend(body);
        body = json_helpers;
    }

    let mut out: Vec<String> = vec![
        "#include \"chelis_runtime.h\"".to_string(),
        "#include <assert.h>".to_string(),
        "#include <math.h>".to_string(),
    ];
    if needs_json_canonical_object_helper {
        out.push("#include <stdlib.h>".to_string());
    }
    out.extend([
        String::new(),
        // chelis#943: emitter-internal accumulator ABI. Deliberately absent
        // from the published chelis_runtime.h (the capacity census governs
        // that surface, and these exist only for compiler-owned accumulators
        // whose refcount-1 exclusivity this emitter proves). The symbols are
        // exported by libchelis_runtime; only the declarations are private.
        "chelis_list *chelis_list_with_capacity(int64_t capacity);".to_string(),
        "void chelis_list_push(chelis_list *list, chelis_value value);".to_string(),
        "void chelis_list_extend(chelis_list *list, const chelis_list *src);".to_string(),
    ]);
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

/// Reject the program when two defs land on the same emitted C symbol
/// (chelis#840 review, finding 2): the `chelis_user__` mapping is not
/// collision-free, so `def double` next to `def chelis_user__double`
/// would otherwise emit a whole-TU symbol redefinition from a build that
/// reported success.
fn reject_duplicate_emitted_function_names(
    emitted_names: &HashMap<String, String>,
) -> Result<(), Unsupported> {
    let mut by_emitted: HashMap<&str, Vec<&str>> = HashMap::new();
    for (original, emitted) in emitted_names {
        by_emitted.entry(emitted).or_default().push(original);
    }
    let mut collisions: Vec<String> = by_emitted
        .into_iter()
        .filter(|(_, originals)| originals.len() > 1)
        .map(|(emitted, mut originals)| {
            originals.sort_unstable();
            format!("`{}` (from `{}`)", emitted, originals.join("`, `"))
        })
        .collect();
    if collisions.is_empty() {
        return Ok(());
    }
    collisions.sort();
    Err(Unsupported::new(
        UnsupportedKind::Construct(format!(
            "colliding emitted C symbol{} {}",
            if collisions.len() == 1 { "" } else { "s" },
            collisions.join(", ")
        )),
        "C host identifier emission",
        Stage::Codegen("c"),
        chelis_types::deliberate_rejection!(
            "[01-CID-1]",
            "the reserved-word mapping prefixes with `chelis_user__` and cannot disambiguate a \
             definition that literally spells the mangled name; rename one definition \
             (chelis#840)"
        ),
    ))
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
        "static inline double chelis_uniform_sample_f64(uint64_t seed, uint64_t index, double low, double high) {"
            .to_string(),
    );
    out.push("    uint64_t x = seed ^ (index * 0x9E3779B97F4A7C15ULL);".to_string());
    out.push("    x ^= x >> 30;".to_string());
    out.push("    x *= 0xBF58476D1CE4E5B9ULL;".to_string());
    out.push("    x ^= x >> 27;".to_string());
    out.push("    x *= 0x94D049BB133111EBULL;".to_string());
    out.push("    x ^= x >> 31;".to_string());
    out.push("    double unit = (double)(x >> 11) / (double)(1ULL << 53);".to_string());
    out.push("    return fma(high - low, unit, low);".to_string());
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

/// Translation-unit-local support for Std.Io.Json's canonical object
/// observation. Generic Dict iteration remains insertion ordered; this helper
/// sorts only the private JSON serializer boundary. Comparing one Unicode
/// scalar slice at a time handles prefixes and embedded U+0000, while UTF-8's
/// byte order preserves scalar-value order for every nonzero scalar.
fn append_json_canonical_object_helpers(out: &mut Vec<String>) {
    for line in [
        "static int chelis_json_compare_strings(chelis_string lhs, chelis_string rhs) {",
        "    int64_t lhs_len = chelis_string_len(lhs);",
        "    int64_t rhs_len = chelis_string_len(rhs);",
        "    int64_t common = lhs_len < rhs_len ? lhs_len : rhs_len;",
        "    for (int64_t index = 0; index < common; ++index) {",
        "        chelis_string lhs_scalar = chelis_string_slice(lhs, index, 1);",
        "        chelis_string rhs_scalar = chelis_string_slice(rhs, index, 1);",
        "        const unsigned char *lhs_bytes = (const unsigned char *)chelis_string_data(lhs_scalar);",
        "        const unsigned char *rhs_bytes = (const unsigned char *)chelis_string_data(rhs_scalar);",
        "        int result = 0;",
        "        int64_t byte = 0;",
        "        while (lhs_bytes[byte] != 0 && rhs_bytes[byte] != 0 && lhs_bytes[byte] == rhs_bytes[byte]) {",
        "            ++byte;",
        "        }",
        "        if (lhs_bytes[byte] < rhs_bytes[byte]) result = -1;",
        "        if (lhs_bytes[byte] > rhs_bytes[byte]) result = 1;",
        "        chelis_string_release(lhs_scalar);",
        "        chelis_string_release(rhs_scalar);",
        "        if (result != 0) return result;",
        "    }",
        "    return lhs_len < rhs_len ? -1 : (lhs_len > rhs_len ? 1 : 0);",
        "}",
        "",
        "static int chelis_json_compare_entry_keys(chelis_value lhs_entry, chelis_value rhs_entry) {",
        "    chelis_value lhs_key = chelis_tuple_get(chelis_value_as_tuple(lhs_entry), 0);",
        "    chelis_value rhs_key = chelis_tuple_get(chelis_value_as_tuple(rhs_entry), 0);",
        "    int result = chelis_json_compare_strings(chelis_value_as_string(lhs_key), chelis_value_as_string(rhs_key));",
        "    chelis_value_release(lhs_key);",
        "    chelis_value_release(rhs_key);",
        "    return result;",
        "}",
        "",
        "static chelis_list *chelis_json_canonical_object_entries(const chelis_dict *dict) {",
        "    chelis_list *source = chelis_dict_entries(dict);",
        "    int64_t len = chelis_list_len(source);",
        "    int64_t *order = len > 0 ? (int64_t *)malloc((size_t)len * sizeof(int64_t)) : NULL;",
        "    if (len > 0 && order == NULL) {",
        "        chelis_fail(chelis_string_from_cstr(\"JSON canonical object ordering allocation failed\"));",
        "    }",
        "    for (int64_t index = 0; index < len; ++index) {",
        "        order[index] = index;",
        "        int64_t cursor = index;",
        "        while (cursor > 0) {",
        "            chelis_value lhs = chelis_list_index(source, order[cursor - 1]);",
        "            chelis_value rhs = chelis_list_index(source, order[cursor]);",
        "            int comparison = chelis_json_compare_entry_keys(lhs, rhs);",
        "            chelis_value_release(lhs);",
        "            chelis_value_release(rhs);",
        "            if (comparison <= 0) break;",
        "            int64_t swap = order[cursor - 1];",
        "            order[cursor - 1] = order[cursor];",
        "            order[cursor] = swap;",
        "            --cursor;",
        "        }",
        "    }",
        "    chelis_list *result = chelis_list_with_capacity(len);",
        "    for (int64_t index = 0; index < len; ++index) {",
        "        chelis_value entry = chelis_list_index(source, order[index]);",
        "        chelis_list_push(result, entry);",
        "        chelis_value_release(entry);",
        "    }",
        "    free(order);",
        "    chelis_list_release(source);",
        "    return result;",
        "}",
    ] {
        out.push(line.to_string());
    }
}

/// Instantiate the scalar host-expression path at each concrete float ABI.
///
/// Tensor activations decompose through `chelis_ir::tier2`; scalar calls in a
/// Surf `def` reach this emitter after host-ABI projection instead. Reduced
/// floats need distinct per-node finalizers even though both compute as C
/// `float`, so one generated specialization cannot serve every source dtype.
fn append_activation_helpers(
    out: &mut Vec<String>,
    suffix: &str,
    c_type: &str,
    literal_suffix: &str,
    exp: &str,
    max: &str,
    finalizer: Option<&str>,
) {
    let literal = |value: &str| match suffix {
        "f16" => format!("chelis_f16_to_f32(chelis_host_f64_to_f16({value}))"),
        "bf16" => format!("chelis_bf16_to_f32(chelis_host_f64_to_bf16({value}))"),
        _ => format!("{value}{literal_suffix}"),
    };
    let finalize = |expr: String| match finalizer {
        Some(function) => format!("{function}({expr})"),
        None => expr,
    };

    out.push(format!(
        "static inline {c_type} chelis_host_relu_{suffix}({c_type} x) {{"
    ));
    out.push(format!(
        "    return {};",
        finalize(format!("{max}({}, x)", literal("0.0")))
    ));
    out.push("}".to_string());

    out.push(format!(
        "static inline {c_type} chelis_host_sigmoid_{suffix}({c_type} x) {{"
    ));
    out.push(format!("    {c_type} neg_x = {};", finalize("-x".into())));
    out.push(format!(
        "    {c_type} exp_neg_x = {};",
        finalize(format!("{exp}(neg_x)"))
    ));
    out.push(format!("    {c_type} one = {};", finalize(literal("1.0"))));
    out.push(format!(
        "    {c_type} denominator = {};",
        finalize("one + exp_neg_x".into())
    ));
    out.push(format!(
        "    return {};",
        finalize("one / denominator".into())
    ));
    out.push("}".to_string());

    out.push(format!(
        "static inline {c_type} chelis_host_tanh_{suffix}({c_type} x) {{"
    ));
    out.push(format!("    {c_type} two = {};", finalize(literal("2.0"))));
    out.push(format!(
        "    {c_type} two_x = {};",
        finalize("two * x".into())
    ));
    out.push(format!(
        "    {c_type} sigmoid = chelis_host_sigmoid_{suffix}(two_x);"
    ));
    out.push(format!(
        "    {c_type} two_again = {};",
        finalize(literal("2.0"))
    ));
    out.push(format!(
        "    {c_type} twice_sigmoid = {};",
        finalize("two_again * sigmoid".into())
    ));
    out.push(format!(
        "    {c_type} neg_one = {};",
        finalize(literal("-1.0"))
    ));
    out.push(format!(
        "    return {};",
        finalize("twice_sigmoid + neg_one".into())
    ));
    out.push("}".to_string());

    out.push(format!(
        "static inline {c_type} chelis_host_silu_{suffix}({c_type} x) {{"
    ));
    out.push(format!(
        "    {c_type} sigmoid = chelis_host_sigmoid_{suffix}(x);"
    ));
    out.push(format!("    return {};", finalize("x * sigmoid".into())));
    out.push("}".to_string());

    out.push(format!(
        "static inline {c_type} chelis_host_gelu_{suffix}({c_type} x) {{"
    ));
    out.push(format!(
        "    {c_type} c = {};",
        finalize(literal("0.7978845608028654"))
    ));
    out.push(format!(
        "    {c_type} k = {};",
        finalize(literal("0.044715"))
    ));
    out.push(format!(
        "    {c_type} x_squared = {};",
        finalize("x * x".into())
    ));
    out.push(format!(
        "    {c_type} x_cubed = {};",
        finalize("x_squared * x".into())
    ));
    out.push(format!(
        "    {c_type} scaled_cube = {};",
        finalize("k * x_cubed".into())
    ));
    out.push(format!(
        "    {c_type} sum_inner = {};",
        finalize("x + scaled_cube".into())
    ));
    out.push(format!(
        "    {c_type} inner = {};",
        finalize("c * sum_inner".into())
    ));
    out.push(format!(
        "    {c_type} tanh_inner = chelis_host_tanh_{suffix}(inner);"
    ));
    out.push(format!("    {c_type} one = {};", finalize(literal("1.0"))));
    out.push(format!(
        "    {c_type} one_plus_tanh = {};",
        finalize("one + tanh_inner".into())
    ));
    out.push(format!(
        "    {c_type} x_mul = {};",
        finalize("x * one_plus_tanh".into())
    ));
    out.push(format!("    {c_type} half = {};", finalize(literal("0.5"))));
    out.push(format!("    return {};", finalize("half * x_mul".into())));
    out.push("}".to_string());
}

fn append_tensor_math_helpers(out: &mut Vec<String>) {
    out.push("static inline float chelis_host_finalize_f16(float x) {".to_string());
    out.push("    return chelis_f16_to_f32(chelis_f32_to_f16(x));".to_string());
    out.push("}".to_string());
    out.push("static inline float chelis_host_finalize_bf16(float x) {".to_string());
    out.push("    return chelis_bf16_to_f32(chelis_f32_to_bf16(x));".to_string());
    out.push("}".to_string());
    append_activation_helpers(
        out,
        "f16",
        "float",
        "f",
        "expf",
        "fmaxf",
        Some("chelis_host_finalize_f16"),
    );
    append_activation_helpers(
        out,
        "bf16",
        "float",
        "f",
        "expf",
        "fmaxf",
        Some("chelis_host_finalize_bf16"),
    );
    append_activation_helpers(out, "f32", "float", "f", "expf", "fmaxf", None);
    append_activation_helpers(out, "f64", "double", "", "exp", "fmax", None);
}

/// Private scalar-cast helpers for the generated translation unit.
///
/// A C `(float)` intermediate is not a conforming f64 -> f16/bf16 cast:
/// values on the f32 rounding cell around a reduced-float midpoint can round
/// twice to the wrong neighbor. These helpers round the binary64 or exact
/// signed-integer significand directly to the destination's IEEE layout.
/// They stay TU-local so the published runtime ABI does not gain an untagged
/// numeric callable (dtype_semantics.md section C6).
pub(crate) fn append_checked_cast_conversion_helpers(out: &mut Vec<String>) {
    out.push("#ifndef CHELIS_PRIVATE_SCALAR_CONVERSION_HELPERS".to_string());
    out.push("#define CHELIS_PRIVATE_SCALAR_CONVERSION_HELPERS".to_string());
    out.extend(
        [
            "static uint64_t chelis_host_round_shift_even_u64(uint64_t value, int shift) {",
            "    if (shift <= 0) return value;",
            "    if (shift >= 64) return 0;",
            "    uint64_t quotient = value >> shift;",
            "    uint64_t remainder = value & ((UINT64_C(1) << shift) - UINT64_C(1));",
            "    uint64_t halfway = UINT64_C(1) << (shift - 1);",
            "    if (remainder > halfway || (remainder == halfway && (quotient & UINT64_C(1)) != 0)) quotient++;",
            "    return quotient;",
            "}",
            "",
            "static uint16_t chelis_host_f64_to_ieee16(double value, int exponent_bits, int mantissa_bits, int bias) {",
            "    uint64_t bits;",
            "    memcpy(&bits, &value, sizeof bits);",
            "    uint16_t sign = (uint16_t)((bits >> 48) & UINT64_C(0x8000));",
            "    uint32_t source_exponent = (uint32_t)((bits >> 52) & UINT64_C(0x7ff));",
            "    uint64_t source_mantissa = bits & UINT64_C(0x000fffffffffffff);",
            "    uint32_t target_exponent_max = (UINT32_C(1) << exponent_bits) - UINT32_C(1);",
            "    if (source_exponent == UINT32_C(0x7ff)) {",
            "        uint16_t target_exponent = (uint16_t)(target_exponent_max << mantissa_bits);",
            "        if (source_mantissa == 0) return (uint16_t)(sign | target_exponent);",
            "        return (uint16_t)(sign | target_exponent | (UINT16_C(1) << (mantissa_bits - 1)));",
            "    }",
            "    if (source_exponent == 0 && source_mantissa == 0) return sign;",
            "    int exponent;",
            "    uint64_t significand;",
            "    if (source_exponent == 0) {",
            "        exponent = -1022;",
            "        significand = source_mantissa;",
            "    } else {",
            "        exponent = (int)source_exponent - 1023;",
            "        significand = (UINT64_C(1) << 52) | source_mantissa;",
            "    }",
            "    int minimum_exponent = 1 - bias;",
            "    int maximum_exponent = (int)target_exponent_max - 1 - bias;",
            "    if (exponent > maximum_exponent) return (uint16_t)(sign | (uint16_t)(target_exponent_max << mantissa_bits));",
            "    uint64_t rounded;",
            "    if (exponent >= minimum_exponent) {",
            "        rounded = chelis_host_round_shift_even_u64(significand, 52 - mantissa_bits);",
            "        if (rounded == (UINT64_C(1) << (mantissa_bits + 1))) {",
            "            rounded >>= 1;",
            "            exponent++;",
            "            if (exponent > maximum_exponent) return (uint16_t)(sign | (uint16_t)(target_exponent_max << mantissa_bits));",
            "        }",
            "        uint16_t target_exponent = (uint16_t)((exponent + bias) << mantissa_bits);",
            "        uint16_t target_mantissa = (uint16_t)(rounded & ((UINT64_C(1) << mantissa_bits) - UINT64_C(1)));",
            "        return (uint16_t)(sign | target_exponent | target_mantissa);",
            "    }",
            "    int shift = (52 - mantissa_bits) + (minimum_exponent - exponent);",
            "    rounded = chelis_host_round_shift_even_u64(significand, shift);",
            "    return (uint16_t)(sign | (uint16_t)rounded);",
            "}",
            "",
            "static uint16_t chelis_host_i64_to_ieee16(int64_t value, int exponent_bits, int mantissa_bits, int bias) {",
            "    uint16_t sign = value < 0 ? UINT16_C(0x8000) : UINT16_C(0);",
            "    uint64_t magnitude = value < 0 ? (uint64_t)(-(value + 1)) + UINT64_C(1) : (uint64_t)value;",
            "    if (magnitude == 0) return sign;",
            "    int exponent = 0;",
            "    for (uint64_t probe = magnitude; probe > UINT64_C(1); probe >>= 1) exponent++;",
            "    uint32_t target_exponent_max = (UINT32_C(1) << exponent_bits) - UINT32_C(1);",
            "    int maximum_exponent = (int)target_exponent_max - 1 - bias;",
            "    if (exponent > maximum_exponent) return (uint16_t)(sign | (uint16_t)(target_exponent_max << mantissa_bits));",
            "    uint64_t rounded = exponent > mantissa_bits",
            "        ? chelis_host_round_shift_even_u64(magnitude, exponent - mantissa_bits)",
            "        : magnitude << (mantissa_bits - exponent);",
            "    if (rounded == (UINT64_C(1) << (mantissa_bits + 1))) {",
            "        rounded >>= 1;",
            "        exponent++;",
            "        if (exponent > maximum_exponent) return (uint16_t)(sign | (uint16_t)(target_exponent_max << mantissa_bits));",
            "    }",
            "    uint16_t target_exponent = (uint16_t)((exponent + bias) << mantissa_bits);",
            "    uint16_t target_mantissa = (uint16_t)(rounded & ((UINT64_C(1) << mantissa_bits) - UINT64_C(1)));",
            "    return (uint16_t)(sign | target_exponent | target_mantissa);",
            "}",
            "",
            "static uint16_t chelis_host_f64_to_f16(double value) { return chelis_host_f64_to_ieee16(value, 5, 10, 15); }",
            "static uint16_t chelis_host_f64_to_bf16(double value) { return chelis_host_f64_to_ieee16(value, 8, 7, 127); }",
            "static uint16_t chelis_host_i64_to_f16(int64_t value) { return chelis_host_i64_to_ieee16(value, 5, 10, 15); }",
            "static uint16_t chelis_host_i64_to_bf16(int64_t value) { return chelis_host_i64_to_ieee16(value, 8, 7, 127); }",
        ]
        .into_iter()
        .map(str::to_string),
    );
    out.push("#endif".to_string());
}

fn append_scalar_conversion_helpers(out: &mut Vec<String>) {
    append_checked_cast_conversion_helpers(out);
    out.push("#ifndef CHELIS_PRIVATE_SCALAR_TENSOR_HELPERS".to_string());
    out.push("#define CHELIS_PRIVATE_SCALAR_TENSOR_HELPERS".to_string());
    out.extend(
        [
            "static uint32_t chelis_host_f32_bits(float value) {",
            "    uint32_t bits; memcpy(&bits, &value, sizeof bits); return bits;",
            "}",
            "",
            "static uint64_t chelis_host_f64_bits(double value) {",
            "    uint64_t bits; memcpy(&bits, &value, sizeof bits); return bits;",
            "}",
            "",
            "static chelis_scalar chelis_host_scalar_from_i8(int8_t value) {",
            "    return chelis_scalar_from_bits(CHELIS_DTYPE_I8, (uint64_t)(uint8_t)value);",
            "}",
            "static chelis_scalar chelis_host_scalar_from_i16(int16_t value) {",
            "    return chelis_scalar_from_bits(CHELIS_DTYPE_I16, (uint64_t)(uint16_t)value);",
            "}",
            "static chelis_scalar chelis_host_scalar_from_i32(int32_t value) {",
            "    return chelis_scalar_from_bits(CHELIS_DTYPE_I32, (uint64_t)(uint32_t)value);",
            "}",
            "static chelis_scalar chelis_host_scalar_from_i64(int64_t value) {",
            "    return chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)value);",
            "}",
            "static chelis_scalar chelis_host_scalar_from_f64(double value) {",
            "    return chelis_scalar_from_bits(CHELIS_DTYPE_F64, chelis_host_f64_bits(value));",
            "}",
            "static chelis_scalar chelis_host_scalar_from_f32(float value) {",
            "    return chelis_scalar_from_bits(CHELIS_DTYPE_F32, (uint64_t)chelis_host_f32_bits(value));",
            "}",
            "static chelis_scalar chelis_host_scalar_from_bool(bool value) {",
            "    return chelis_scalar_from_bits(CHELIS_DTYPE_BOOL, value ? 1u : 0u);",
            "}",
            "",
            "static int64_t chelis_host_scalar_as_i64(chelis_scalar value, chelis_dtype expected) {",
            "    if (value.dtype != expected) { fprintf(stderr, \"scalar dtype mismatch\\n\"); exit(1); }",
            "    switch (expected) {",
            "        case CHELIS_DTYPE_I8: return (int8_t)(uint8_t)value.bits;",
            "        case CHELIS_DTYPE_I16: return (int16_t)(uint16_t)value.bits;",
            "        case CHELIS_DTYPE_I32: return (int32_t)(uint32_t)value.bits;",
            "        case CHELIS_DTYPE_I64: { int64_t out; memcpy(&out, &value.bits, sizeof out); return out; }",
            "        default: fprintf(stderr, \"expected signed integer scalar\\n\"); exit(1);",
            "    }",
            "}",
            "",
            "static uint64_t chelis_host_scalar_bits(chelis_scalar value, chelis_dtype expected) {",
            "    if (value.dtype != expected) { fprintf(stderr, \"scalar dtype mismatch\\n\"); exit(1); }",
            "    return value.bits;",
            "}",
            "",
            "static double chelis_host_scalar_as_float(chelis_scalar value, chelis_dtype expected) {",
            "    if (value.dtype != expected) { fprintf(stderr, \"scalar dtype mismatch\\n\"); exit(1); }",
            "    switch (expected) {",
            "        case CHELIS_DTYPE_F32: return (double)chelis_f32_from_bits((uint32_t)value.bits);",
            "        case CHELIS_DTYPE_F64: return chelis_f64_from_bits(value.bits);",
            "        case CHELIS_DTYPE_F16: return (double)chelis_f16_to_f32((uint16_t)value.bits);",
            "        case CHELIS_DTYPE_BF16: return (double)chelis_bf16_to_f32((uint16_t)value.bits);",
            "        default: fprintf(stderr, \"expected float scalar\\n\"); exit(1);",
            "    }",
            "}",
            "",
            "static bool chelis_host_scalar_as_bool(chelis_scalar value) {",
            "    if (value.dtype != CHELIS_DTYPE_BOOL) { fprintf(stderr, \"scalar dtype mismatch\\n\"); exit(1); }",
            "    return value.bits == 1;",
            "}",
            "",
            "static chelis_tensor *chelis_host_scalar_tensor_from_f16(uint16_t value) {",
            "    return chelis_scalar_tensor(chelis_scalar_from_bits(CHELIS_DTYPE_F16, (uint64_t)value));",
            "}",
            "",
            "static chelis_tensor *chelis_host_scalar_tensor_from_bf16(uint16_t value) {",
            "    return chelis_scalar_tensor(chelis_scalar_from_bits(CHELIS_DTYPE_BF16, (uint64_t)value));",
            "}",
        ]
        .into_iter()
        .map(str::to_string),
    );
    out.push("#endif".to_string());
}

/// C expression for an already-planned checked scalar conversion.
///
/// The host scalar emitter, host tensor emitter, and DAG emitter all consume
/// this projection so reduced-float direct rounding and trap helpers cannot
/// drift between C surfaces. Trapping callers may use the condition helpers
/// below to classify in parallel, then evaluate this expression only for
/// valid elements or for the selected lowest-index candidate.
pub(crate) fn checked_cast_c_expr(plan: CheckedCastPlan, value: &str) -> String {
    let target = plan.target();
    match plan.kind() {
        CheckedCastKind::Identity => value.to_string(),
        CheckedCastKind::ExactToInteger => {
            let overflow = NumericTrap::Overflow {
                op: "cast",
                prim: target,
            }
            .to_string();
            format!(
                "({})chelis_checked_int_cast((int64_t)({value}), {}, {overflow:?})",
                cast_prim_c_type(target),
                cast_integer_width(target)
            )
        }
        CheckedCastKind::FloatToInteger => {
            let domain = NumericTrap::Domain {
                op: "cast",
                prim: target,
            }
            .to_string();
            let overflow = NumericTrap::Overflow {
                op: "cast",
                prim: target,
            }
            .to_string();
            format!(
                "({})chelis_checked_float_to_int({}, {}, {domain:?}, {overflow:?})",
                cast_prim_c_type(target),
                cast_float_as_double(plan.source(), value),
                cast_integer_width(target)
            )
        }
        CheckedCastKind::ExactToFloat => match target {
            Prim::F64 => format!("(double)((int64_t)({value}))"),
            Prim::F32 => format!("(float)((int64_t)({value}))"),
            Prim::F16 => format!("chelis_host_i64_to_f16((int64_t)({value}))"),
            Prim::Bf16 => format!("chelis_host_i64_to_bf16((int64_t)({value}))"),
            Prim::F8e4m3
            | Prim::Int8
            | Prim::Int16
            | Prim::Int32
            | Prim::Int64
            | Prim::Bool
            | Prim::String => unreachable!("ExactToFloat plan has a float target"),
        },
        CheckedCastKind::FloatToFloat => match target {
            Prim::F64 => cast_float_as_double(plan.source(), value),
            Prim::F32 => format!("(float)({})", cast_float_as_double(plan.source(), value)),
            Prim::F16 => format!(
                "chelis_host_f64_to_f16({})",
                cast_float_as_double(plan.source(), value)
            ),
            Prim::Bf16 => format!(
                "chelis_host_f64_to_bf16({})",
                cast_float_as_double(plan.source(), value)
            ),
            Prim::F8e4m3
            | Prim::Int8
            | Prim::Int16
            | Prim::Int32
            | Prim::Int64
            | Prim::Bool
            | Prim::String => unreachable!("FloatToFloat plan has a float target"),
        },
        CheckedCastKind::ExactToBool => {
            let domain = NumericTrap::Domain {
                op: "cast",
                prim: Prim::Bool,
            }
            .to_string();
            format!("chelis_checked_bool_from_int((int64_t)({value}), {domain:?})")
        }
        CheckedCastKind::FloatToBool => {
            let domain = NumericTrap::Domain {
                op: "cast",
                prim: Prim::Bool,
            }
            .to_string();
            format!(
                "chelis_checked_bool_from_float({}, {domain:?})",
                cast_float_as_double(plan.source(), value)
            )
        }
    }
}

/// Non-trapping store expression after the caller has classified the element
/// with both condition helpers. This keeps aborting runtime helpers out of an
/// OpenMP worker while preserving the same representation conversion.
pub(crate) fn checked_cast_valid_c_expr(plan: CheckedCastPlan, value: &str) -> String {
    match plan.kind() {
        CheckedCastKind::ExactToInteger => {
            format!("({})((int64_t)({value}))", cast_prim_c_type(plan.target()))
        }
        CheckedCastKind::FloatToInteger => format!(
            "({})((int64_t)({}))",
            cast_prim_c_type(plan.target()),
            cast_float_as_double(plan.source(), value)
        ),
        CheckedCastKind::ExactToBool => format!("((int64_t)({value}) == 1)"),
        CheckedCastKind::FloatToBool => {
            format!("({} == 1.0)", cast_float_as_double(plan.source(), value))
        }
        CheckedCastKind::Identity
        | CheckedCastKind::ExactToFloat
        | CheckedCastKind::FloatToFloat => checked_cast_c_expr(plan, value),
    }
}

pub(crate) fn checked_cast_domain_condition(plan: CheckedCastPlan, value: &str) -> Option<String> {
    match plan.kind() {
        CheckedCastKind::FloatToInteger => {
            let value = cast_float_as_double(plan.source(), value);
            Some(format!("(!isfinite({value}) || trunc({value}) != {value})"))
        }
        CheckedCastKind::ExactToBool => Some(format!(
            "((int64_t)({value}) != 0 && (int64_t)({value}) != 1)"
        )),
        CheckedCastKind::FloatToBool => {
            let value = cast_float_as_double(plan.source(), value);
            Some(format!(
                "(!isfinite({value}) || ({value} != 0.0 && {value} != 1.0))"
            ))
        }
        CheckedCastKind::Identity
        | CheckedCastKind::ExactToInteger
        | CheckedCastKind::ExactToFloat
        | CheckedCastKind::FloatToFloat => None,
    }
}

pub(crate) fn checked_cast_overflow_condition(
    plan: CheckedCastPlan,
    value: &str,
) -> Option<String> {
    match plan.kind() {
        CheckedCastKind::ExactToInteger => {
            let (minimum, maximum) = cast_integer_bounds(plan.target());
            Some(format!(
                "((int64_t)({value}) < {minimum} || (int64_t)({value}) > {maximum})"
            ))
        }
        CheckedCastKind::FloatToInteger => {
            let value = cast_float_as_double(plan.source(), value);
            let condition = if plan.target() == Prim::Int64 {
                format!("({value} < -9223372036854775808.0 || {value} >= 9223372036854775808.0)")
            } else {
                let (minimum, maximum) = cast_integer_bounds(plan.target());
                format!("({value} < (double){minimum} || {value} > (double){maximum})")
            };
            Some(condition)
        }
        CheckedCastKind::Identity
        | CheckedCastKind::ExactToFloat
        | CheckedCastKind::FloatToFloat
        | CheckedCastKind::ExactToBool
        | CheckedCastKind::FloatToBool => None,
    }
}

fn cast_float_as_double(source: Prim, value: &str) -> String {
    match source {
        Prim::F64 | Prim::F32 => format!("(double)({value})"),
        Prim::F16 => format!("(double)chelis_f16_to_f32({value})"),
        Prim::Bf16 => format!("(double)chelis_bf16_to_f32({value})"),
        Prim::F8e4m3
        | Prim::Int8
        | Prim::Int16
        | Prim::Int32
        | Prim::Int64
        | Prim::Bool
        | Prim::String => unreachable!("float checked-cast action has a float source"),
    }
}

fn cast_integer_width(target: Prim) -> i64 {
    match target {
        Prim::Int8 => 8,
        Prim::Int16 => 16,
        Prim::Int32 => 32,
        Prim::Int64 => 64,
        Prim::F32
        | Prim::F64
        | Prim::F16
        | Prim::Bf16
        | Prim::F8e4m3
        | Prim::Bool
        | Prim::String => unreachable!("integer checked-cast action has an integer target"),
    }
}

fn cast_integer_bounds(target: Prim) -> (&'static str, &'static str) {
    match target {
        Prim::Int8 => ("INT8_MIN", "INT8_MAX"),
        Prim::Int16 => ("INT16_MIN", "INT16_MAX"),
        Prim::Int32 => ("INT32_MIN", "INT32_MAX"),
        Prim::Int64 => ("INT64_MIN", "INT64_MAX"),
        Prim::F32
        | Prim::F64
        | Prim::F16
        | Prim::Bf16
        | Prim::F8e4m3
        | Prim::Bool
        | Prim::String => unreachable!("integer checked-cast action has an integer target"),
    }
}

fn cast_prim_c_type(prim: Prim) -> &'static str {
    match prim {
        Prim::F64 => "double",
        Prim::F32 => "float",
        Prim::F16 | Prim::Bf16 => "uint16_t",
        Prim::Int8 => "int8_t",
        Prim::Int16 => "int16_t",
        Prim::Int32 => "int32_t",
        Prim::Int64 => "int64_t",
        Prim::Bool => "bool",
        Prim::F8e4m3 | Prim::String => {
            unreachable!("unsupported Prim cannot enter checked C cast emission")
        }
    }
}

/// Render a tensor element by first recovering the exact tagged scalar.
/// The runtime owns the exhaustive dtype dispatch and public text contract.
fn append_tensor_print_helper(out: &mut Vec<String>) {
    out.push("static chelis_string chelis_host_string_from_f16(uint16_t value) {".to_string());
    out.push(
        "    return chelis_string_from_scalar(chelis_scalar_from_bits(CHELIS_DTYPE_F16, (uint64_t)value));"
            .to_string(),
    );
    out.push("}".to_string());
    out.push(String::new());
    out.push("static chelis_string chelis_host_string_from_bf16(uint16_t value) {".to_string());
    out.push(
        "    return chelis_string_from_scalar(chelis_scalar_from_bits(CHELIS_DTYPE_BF16, (uint64_t)value));"
            .to_string(),
    );
    out.push("}".to_string());
    out.push(String::new());
    out.push(
        "static chelis_scalar chelis_host_tensor_scalar_at(const chelis_tensor* t, int64_t i) {"
            .to_string(),
    );
    out.push("    uint64_t bits = 0;".to_string());
    out.push("    int64_t width = chelis_dtype_size(t->dtype);".to_string());
    out.push("    memcpy(&bits, (const uint8_t*)t->data + i * width, (size_t)width);".to_string());
    out.push("    return chelis_scalar_from_bits(t->dtype, bits);".to_string());
    out.push("}".to_string());
    out.push(String::new());
    out.push(
        "static void chelis_print_tensor_elem_stdout(const chelis_tensor* t, int64_t i) {"
            .to_string(),
    );
    out.push(
        "    chelis_string text = chelis_string_from_scalar(chelis_host_tensor_scalar_at(t, i));"
            .to_string(),
    );
    out.push("    fputs(chelis_string_data(text), stdout);".to_string());
    out.push("    chelis_string_release(text);".to_string());
    out.push("}".to_string());
    out.push(String::new());
    out.push("static void chelis_print_tensor_stdout(const chelis_tensor* t) {".to_string());
    // [05-OBS-4]: a rank-0 tensor renders as its single element, bare -
    // the `tensor(shape=[], data=[..])` wrapper is not an exit form.
    out.push("    if (t->rank == 0) {".to_string());
    out.push("        chelis_print_tensor_elem_stdout(t, 0);".to_string());
    out.push("        return;".to_string());
    out.push("    }".to_string());
    out.push("    printf(\"tensor(shape=[\");".to_string());
    out.push("    for (int64_t d = 0; d < t->rank; ++d) {".to_string());
    out.push("        if (d > 0) { printf(\", \"); }".to_string());
    out.push("        printf(\"%lld\", (long long)t->shape[d]);".to_string());
    out.push("    }".to_string());
    out.push("    printf(\"], data=[\");".to_string());
    // [05-OBS-5]: every exit truncates tensor element rendering after 32
    // elements with the `, ...` marker; full-element fidelity is
    // to_list's and the wire's job, never print's.
    out.push("    int64_t limit = t->size < 32 ? t->size : 32;".to_string());
    out.push("    for (int64_t i = 0; i < limit; ++i) {".to_string());
    out.push("        if (i > 0) { printf(\", \"); }".to_string());
    out.push("        chelis_print_tensor_elem_stdout(t, i);".to_string());
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
    out.push("    if (ndim64 < 0 || ndim64 > INT32_MAX) {".to_string());
    out.push(
        "        fprintf(stderr, \"reshape rank is outside int32: %lld\\n\", (long long)ndim64);"
            .to_string(),
    );
    out.push("        exit(1);".to_string());
    out.push("    }".to_string());
    out.push("    int ndim = (int)ndim64;".to_string());
    out.push(
        "    int64_t *shape = (int64_t*)calloc((size_t)(ndim > 0 ? ndim : 1), sizeof(int64_t));"
            .to_string(),
    );
    out.push("    if (shape == NULL) { fprintf(stderr, \"reshape shape allocation failed\\n\"); exit(1); }".to_string());
    out.push("    int64_t expected = 1;".to_string());
    out.push("    for (int i = 0; i < ndim; ++i) {".to_string());
    out.push(
        "        int64_t dim = chelis_host_scalar_as_i64(chelis_value_as_scalar(chelis_list_index(shape_values, i)), CHELIS_DTYPE_I64);"
            .to_string(),
    );
    out.push("        if (dim < 0) {".to_string());
    out.push(
        "            fprintf(stderr, \"reshape expects non-negative sizes, got %lld\\n\", (long long)dim);"
            .to_string(),
    );
    out.push("            exit(1);".to_string());
    out.push("        }".to_string());
    out.push("        shape[i] = dim;".to_string());
    out.push("        expected *= dim;".to_string());
    out.push("    }".to_string());
    out.push("    if (expected != input->size) {".to_string());
    out.push(
        "        fprintf(stderr, \"reshape expects %lld elements but tensor has %lld\\n\", (long long)expected, (long long)input->size);"
            .to_string(),
    );
    out.push("        exit(1);".to_string());
    out.push("    }".to_string());
    out.push(
        "    chelis_tensor* out_tensor = chelis_alloc(ndim, shape, input->dtype);".to_string(),
    );
    out.push("    free(shape);".to_string());
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
    emit_host_declarations(&abi_program, program_name, false, false)
}

pub(crate) fn emit_host_abi_header(
    program: &HostProgram,
    program_name: &str,
) -> Result<String, Unsupported> {
    emit_host_declarations(program, program_name, false, false)
}

fn emit_host_header_with_linkage(
    program: &HostProgram,
    program_name: &str,
    internal_linkage: bool,
) -> Result<String, Unsupported> {
    // The in-`.c` prototype block: unfiltered, so a specialization may call
    // a definition emitted later in the translation unit.
    emit_host_declarations(program, program_name, internal_linkage, true)
}

/// Emit function declarations. The published `.h` legs pass
/// `include_specializations = false`: a monomorphized specialization is a
/// compiler-internal symbol whose name changes with the program's
/// instantiation set, and nothing outside the translation unit may call it
/// (harden-bounded-monomorphization D2).
fn emit_host_declarations(
    program: &HostProgram,
    program_name: &str,
    internal_linkage: bool,
    include_specializations: bool,
) -> Result<String, Unsupported> {
    program
        .functions
        .iter()
        .filter(|function| include_specializations || !function.is_monomorphized_specialization())
        .map(|function| {
            let prefix = if internal_linkage || function.is_monomorphized_specialization() {
                "static inline "
            } else {
                ""
            };
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
    // The CEmitter prepends dtype-specific uniform sampling helpers to
    // every DAG it emits so that a standalone-emitted kernel
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
        if line == "/* CHELIS_UNIFORM_HELPERS_BEGIN */" {
            skipping_uniform_prelude = true;
            continue;
        }
        if skipping_uniform_prelude {
            if line == "/* CHELIS_UNIFORM_HELPERS_END */" {
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
        // The binding-value local owns its allocation when the binding
        // built one (tensor kernel output, list/dict builtin, literal).
        // Track it here; the alias name (`theta`) is never tracked, and
        // dedup in `emit_scope_releases` collapses the case where the
        // binding value *is* a literal already tracked above.
        //
        // chelis#1222: a binding can instead be a second *name* for an
        // allocation an earlier binding already owns -- `rho = rho_base`,
        // an `if`/`match` whose arms are existing bindings, a call to a
        // function that returns one of its arguments or a captured
        // top-level binding, or an identity tensor helper. `main` frees
        // one pointer per tracked variable, so claiming such a binding
        // frees one allocation twice: a `chelis_free` double free for a
        // tensor, an unearned release for a refcounted container. Leave it
        // untracked; the owning binding's release reclaims it exactly once.
        if !emitter.scope_already_owns(&binding_var) {
            emitter.track_owned_alloc(&binding_var, &binding.ty);
        }
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
        // chelis#1222: the user-facing name is a second slot holding the
        // same pointer as the value temp. Record it so a later binding
        // that reads the name resolves back to the temp `main` tracks.
        if release_call(&binding.name, &binding.ty).is_some() {
            let name = c_ident(&binding.name).into_owned();
            emitter.record_alias(&name, &binding_var);
        }
    }
    for binding in &program.globals {
        if !binding.display_roots.is_empty() {
            for root in &binding.display_roots {
                emitter.emit_manifest_root(
                    &root.name,
                    &c_ident(&binding.name),
                    &binding.ty,
                    &root.path,
                )?;
            }
        } else if let Some(display_name) = binding.display_name.as_deref() {
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

/// chelis#1222: the spelling of a binder's alias-graph key.
///
/// A binder key shares one namespace with every other string
/// [`HostEmitter::alias_source`] is keyed on, and the other inhabitants of
/// that namespace are C identifiers: emitter temps, and -- through
/// [`HostEmitter::resolve_alias_key`]'s fallback -- the `c_ident` spelling
/// of any name no binder scope introduced (a hoisted top-level binding, a
/// compiled function's parameter).
///
/// The key must therefore be a string no source identifier can produce.
/// `#` is the discriminator: it is not a Chelis identifier character, so
/// `c_ident` can never return a name containing one, while the key itself
/// is only ever a `HashMap` key and never reaches emitted C.
///
/// The first cut spelled keys `__bind_N`, on `c_ident`'s premise that
/// "Surf/Deep identifiers cannot start with `__`". The lexer and checker
/// accept such identifiers, so `__bind_0 = to_tensor([1.0f32, 2.0f32])`
/// beside any `let` block overwrote the top-level binding's alias edge and
/// re-armed the chelis#1222 double free -- reachable only by spelling the
/// binding a particular way, which is exactly the alpha-dependence this
/// mechanism exists to remove.
const BINDER_KEY_PREFIX: &str = "#bind#";

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
    /// chelis#1222: C variables whose value is a bare pointer copy of
    /// another C variable's allocation, mapped to that source variable.
    ///
    /// A scope may only release what it allocated. Every emitted
    /// `target = source;` that hands one allocation to a second slot is
    /// recorded here, so a slot can be traced back to the variable that
    /// actually owns its pointer before the scope claims it. Chains are
    /// resolved by [`HostEmitter::alias_root`].
    ///
    /// Only heap-owning types are recorded: a scalar copy owns nothing, so
    /// tracking it would be noise. `let`-binding *names* are deliberately
    /// not recorded either -- the block-release ledger tracks the name, not
    /// its `__let_N` value temp, so a chain through the name would report a
    /// binding this block genuinely owns as borrowed.
    alias_source: HashMap<String, String>,
    /// chelis#1222: C variables holding a pointer this scope did not
    /// allocate and whose owner it cannot name -- the result of a call
    /// whose callee may hand back a value it read out of an enclosing
    /// scope (`may_return_outer`). There is no source variable to record,
    /// only the fact that claiming ownership would be wrong.
    foreign: HashSet<String>,
    /// chelis#1222: a stack of binder scopes, mapping a **raw source name**
    /// to the alias-graph key that currently means it.
    ///
    /// A source name is not a usable key on its own. C identifiers are
    /// scoped and reusable, so one name can have two live meanings, and a
    /// name-keyed graph silently conflates them: the outer meaning is the
    /// one a later reference needs, while the inner one is what the map
    /// holds. Every binder therefore gets its own [`BINDER_KEY_PREFIX`] key,
    /// and references resolve through this stack before touching the graph,
    /// so a bound name's *spelling* never reaches `alias_source` at all.
    ///
    /// That is what makes the ownership decision alpha-invariant: renaming
    /// a bound variable cannot change which allocations get released. The
    /// key spelling is part of that guarantee, not decoration -- a key a
    /// source identifier could also spell puts the two back in one slot.
    ///
    /// `result_alias_set`'s `env` is the analysis-side counterpart and now
    /// saves and restores shadowed names in all three of its binder arms.
    /// The `Let` arm did not until chelis#1222, so an earlier version of
    /// this comment cited a precedent that did not exist.
    binder_keys: Vec<HashMap<String, String>>,
    /// chelis#1222: counter for [`HostEmitter::bind_alias_key`].
    ///
    /// Deliberately NOT `temp_counter`. A binder key is a key in
    /// `alias_source` and never appears in emitted C, so drawing from the
    /// emitted-temp counter would renumber every later temp in the
    /// translation unit -- a corpus-wide textual diff that says nothing
    /// about behaviour and hides the diff that would. Measured: sharing
    /// the counter changed the emitted C of 21 of 76 corpus files with
    /// identical release counts in all 21.
    binder_key_counter: usize,
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
            alias_source: HashMap::new(),
            foreign: HashSet::new(),
            binder_keys: Vec::new(),
            binder_key_counter: 0,
        }
    }

    /// chelis#1222: record that `target` now holds `source`'s pointer.
    /// Self-aliases are dropped so [`HostEmitter::alias_root`] cannot spin.
    fn record_alias(&mut self, target: &str, source: &str) {
        if target == source {
            return;
        }
        self.alias_source
            .insert(target.to_string(), source.to_string());
    }

    /// chelis#1222: a builtin whose emitted form is `target = <arg temp>;`
    /// hands the argument's pointer straight through, so `target` owns
    /// nothing of its own and the receiving scope must trace it back before
    /// claiming it.
    ///
    /// `source` is an emitter temp, never a source name, so it needs no
    /// [`HostEmitter::resolve_alias_key`] pass. Only heap-owning types are
    /// recorded, matching the `Var` arm: a scalar copy owns nothing.
    ///
    /// Recording is right whether or not the argument was itself borrowed.
    /// A fresh argument's temp has no outgoing edge, so the chain ends at a
    /// variable this scope allocated and never tracked and `target` is still
    /// claimed; a borrowed one reaches its owner and is not. Without this,
    /// `b = debug(a)` freed `a`'s tensor twice -- the reported chelis#1222
    /// shape, through a builtin instead of a bare name.
    ///
    /// `arg` is the unlowered argument expression. When it is a bare `Var`,
    /// this is the same transfer the `Var` arm of [`HostEmitter::assign_expr`]
    /// performs, so it takes the same issue #406 escape retain: a block
    /// binding that reaches an owned destination through such a builtin is
    /// released at the block close like any other, and without the retain
    /// `b = { c = [1i64]  debug(c) }` released one allocation twice.
    fn record_pointer_copy(&mut self, target: &str, source: &str, arg: &HostExpr, ty: &HostType) {
        if let HostExprKind::Var(name, _) = &arg.kind {
            self.retain_transferred_result(target, name, ty);
        }
        if release_call(target, ty).is_some() {
            self.record_alias(target, source);
        }
    }

    /// chelis#1222: record that `target` holds a pointer from an enclosing
    /// scope that this emitter cannot attribute to a local variable.
    fn mark_foreign(&mut self, target: &str) {
        self.foreign.insert(target.to_string());
    }

    /// chelis#1222: the alias-graph key that currently means `name`.
    ///
    /// Walks the binder stack innermost-first, exactly as C name lookup
    /// does, and falls back to the `c_ident`-mapped name for anything no
    /// binder scope introduced -- a compiled function's parameter, a
    /// hoisted top-level binding, or a temp. Those are already unique
    /// within one emitted C function body, so the fallback needs no key of
    /// its own.
    fn resolve_alias_key(&self, name: &str) -> String {
        self.binder_keys
            .iter()
            .rev()
            .find_map(|frame| frame.get(name).cloned())
            .unwrap_or_else(|| c_ident(name).into_owned())
    }

    /// chelis#1222: give `name` its own alias-graph key inside the innermost
    /// binder scope.
    ///
    /// Call this only AFTER the binder's initializer has been emitted. The
    /// initializer is evaluated in the *enclosing* scope and may read the
    /// outer meaning of this very name (`a = a`); binding the name first
    /// would make that read resolve to the binder being defined. The
    /// analysis side already sequences it this way -- `result_alias_set`
    /// computes a binding's set before inserting the name -- and getting it
    /// backwards here is precisely the defect that produced a double free
    /// for `b = { a = a  a }` while `b = { z = a  z }` was correct.
    ///
    /// The key is spelled with [`BINDER_KEY_PREFIX`] so no source identifier
    /// can name one; see that constant for why a `__`-prefixed key was a
    /// double free waiting to be spelled.
    fn bind_alias_key(&mut self, name: &str) -> String {
        let key = format!("{BINDER_KEY_PREFIX}{}", self.binder_key_counter);
        self.binder_key_counter += 1;
        if let Some(frame) = self.binder_keys.last_mut() {
            frame.insert(name.to_string(), key.clone());
        }
        key
    }

    /// chelis#1222: follow `var` back through the recorded pointer copies,
    /// returning every variable that holds the same allocation, `var`
    /// first and the owner last. A single-element chain means nothing
    /// aliased into `var`, so its value is freshly allocated. The `seen`
    /// set makes the walk total even if a future emit path records a
    /// cycle.
    ///
    /// The whole chain matters, not just its end: a `let` block's result
    /// reaches an outer binding through the block's own binding name and
    /// value temp, and which of those links is a slot somebody already
    /// releases is exactly the ownership question.
    fn alias_chain(&self, var: &str) -> Vec<String> {
        let mut chain = vec![var.to_string()];
        let mut seen: HashSet<String> = HashSet::from([var.to_string()]);
        while let Some(next) = self.alias_source.get(chain.last().expect("non-empty")) {
            if !seen.insert(next.clone()) {
                break;
            }
            chain.push(next.clone());
        }
        chain
    }

    /// chelis#1222: the variable at the end of `var`'s alias chain.
    fn alias_root(&self, var: &str) -> String {
        self.alias_chain(var)
            .pop()
            .expect("alias chain is never empty")
    }

    /// chelis#1222: is `var`'s allocation already owned by another slot
    /// this scope releases, or by a scope outside this one?
    ///
    /// `main` frees one allocation per tracked variable, so tracking a
    /// second variable that holds the same pointer frees it twice -- for a
    /// tensor that is a hard `chelis_free` double free, and for a
    /// refcounted container a release the ledger never earned. Both are
    /// heap corruption; the answer here decides whether the value is
    /// claimed at all.
    fn scope_already_owns(&self, var: &str) -> bool {
        let chain = self.alias_chain(var);
        if chain.iter().any(|link| self.foreign.contains(link)) {
            return true;
        }
        self.scope_releases.as_ref().is_some_and(|tracked| {
            chain
                .iter()
                .any(|link| tracked.iter().any(|(name, _)| name == link))
        })
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
        // an out-of-scope identifier). The combinator loops accumulate
        // in place (`chelis_list_push`/`chelis_list_extend`, chelis#943)
        // so they create no per-iteration list generations; the `append`
        // builtin's per-call result inside a nested block is still
        // unreleased -- that remaining half of chelis#943 needs
        // consumption facts this emitter does not have yet.
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
        // chelis#1222: `bindings` holds alias keys, not spellings, so the
        // incoming source name resolves the same way a reference does.
        let source_key = self.resolve_alias_key(source);
        let source_is_binding = self
            .let_scopes
            .iter()
            .any(|scope| scope.bindings.contains(&source_key));
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
            HostExprKind::Float(value) => self.lines.push(format!(
                "{}{target} = chelis_f64_from_bits(UINT64_C(0x{:016x}));",
                self.indent,
                value.to_bits()
            )),
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
                    // chelis#1222: `target` now holds `name`'s pointer. Only
                    // a heap-owning type can be released twice, so only
                    // those are recorded -- and the link is to the key that
                    // currently means `name`, never to the spelling.
                    if release_call(target, ty).is_some() {
                        let source = self.resolve_alias_key(name);
                        self.record_alias(target, &source);
                    }
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
                    HostType::Option(inner) if is_scalar_abi(inner.as_ref()) => {
                        self.lines.push(format!(
                            "{}{} {} = {};",
                            self.indent,
                            c_type(&inner_ty)?,
                            bind_name,
                            scalar_carrier_value_expr(
                                &format!("{option_var}.value"),
                                inner.as_ref(),
                            )?
                        ));
                    }
                    HostType::Option(_) => {
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
                    other => {
                        return Err(invalid_abi_shape(
                            format!("option match scrutinee has non-option ABI type `{other:?}`"),
                            "option match",
                        ));
                    }
                }
                // chelis#1222: the binder shadows any enclosing name it
                // reuses. Its key carries no outgoing edge, because the
                // value is freshly extracted here rather than copied from
                // something this scope already owns -- which is also what
                // keeps emitted C unchanged for every program that does not
                // shadow: a reference to it dead-ends exactly as it does
                // today.
                self.binder_keys.push(HashMap::new());
                self.bind_alias_key(bind_name);
                self.assign_expr(target, some_expr, ty)?;
                self.binder_keys.pop();
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
                // chelis#1222: open a binder scope. It starts EMPTY on
                // purpose -- each name enters only after its own initializer
                // has been emitted, because that initializer runs in the
                // enclosing scope and may read the outer meaning of the very
                // name being bound.
                self.binder_keys.push(HashMap::new());
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
                    // chelis#1222: the binding is a second slot on the same
                    // pointer, and a block whose body is that name hands the
                    // allocation to the outer scope through it, so the chain
                    // has to cross it. Give it a key of its own now that the
                    // initializer has been emitted -- see `bind_alias_key`
                    // for why the ordering is the whole fix.
                    let binder_key = self.bind_alias_key(&binding.name);
                    if release_call(&binding.name, &binding.ty).is_some() {
                        self.record_alias(&binder_key, &temp);
                    }
                    // Track the binding name (not its `__let_N` temp: the
                    // two alias the same allocation, so releasing only the
                    // name frees it exactly once). Add it to the scope's
                    // binding set after its value is computed so a binding
                    // whose value reads an *earlier* binding still retains
                    // on that transfer.
                    //
                    // chelis#1222 deliberately does NOT gate this on whether
                    // the value looks borrowed. `emit_main`'s sibling rule
                    // ("cannot prove ownership, so do not claim") runs once
                    // per PROGRAM and its residual is bounded by the number
                    // of top-level bindings. The same rule here would run
                    // once per CALL, turning every unprovable case into a
                    // leak that grows with the call count -- measurably, in
                    // `Std.Io.Json` and `Std.Decimal`. Releasing a reference
                    // this block never acquired is still wrong, but the fix
                    // has to establish ownership positively rather than
                    // infer a borrow from missing evidence. Tracked as the
                    // block-scope follow-up in the PR.
                    if binding_release(&binding.name, &binding.ty).is_some() {
                        heap_bindings.push((binding.name.clone(), binding.ty.clone()));
                        if let Some(scope) = self.let_scopes.last_mut() {
                            // Keyed, not spelled: `retain_transferred_result`
                            // and `retain_call_escaped_args` resolve a
                            // reference before testing membership, so a
                            // shadowing binder elsewhere cannot match this
                            // block's binding by name alone (chelis#1222).
                            scope.bindings.insert(binder_key.clone());
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
                // The frame goes; the edges it recorded stay. A value that
                // escaped this block still reaches its owner through the
                // popped binder's key, which is why nothing has to be
                // collapsed or rewritten on the way out (chelis#1222).
                self.binder_keys.pop();
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
        // A checker-stamped float literal is represented as a cast around
        // its lexical f64 image. Materialize that literal directly at the
        // declared width: this both preserves the one-rounding contract and
        // leaves an own-width bit artifact in generated C. An explicit
        // nested source cast (for example `1.0f32` cast to f16) does not take
        // this fast path at the outer cast, so its two authored conversions
        // remain distinct.
        if name == "cast"
            && let [arg] = args
            && let HostExprKind::Float(value) = &arg.kind
        {
            let assignment = match ty {
                HostType::Float64 => Some(format!(
                    "chelis_f64_from_bits(UINT64_C(0x{:016x}))",
                    value.to_bits()
                )),
                HostType::Float32 => Some(format!(
                    "chelis_f32_from_bits(UINT32_C(0x{:08x}))",
                    (*value as f32).to_bits()
                )),
                HostType::Float16 => Some(format!(
                    "UINT16_C(0x{:04x})",
                    chelis_types::f16_from_f64_rne(*value).to_bits()
                )),
                HostType::BFloat16 => Some(format!(
                    "UINT16_C(0x{:04x})",
                    chelis_types::bf16_from_f64_rne(*value).to_bits()
                )),
                _ => None,
            };
            if let Some(assignment) = assignment {
                let target_prim = checked_cast_abi_scalar_prim(ty)?;
                let plan = CheckedCastPlan::new(Prim::F64, target_prim)
                    .map_err(|error| checked_cast_plan_error(error.to_string()))?;
                self.emit_span_comments(arg);
                self.lines.push(format!(
                    "{}/* checked cast plan: {} -> {} */",
                    self.indent,
                    plan.source().name(),
                    plan.target().name()
                ));
                if plan.kind() == CheckedCastKind::Identity {
                    self.lines
                        .push(format!("{}/* checked cast identity */", self.indent));
                }
                self.lines
                    .push(format!("{}{target} = {assignment};", self.indent));
                return Ok(());
            }
        }

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

        if name == "__json_canonical_object_entries" {
            if arg_vars.len() != 1 || !matches!(arg_vars[0].1, HostType::Dict(_, _)) {
                return Err(invalid_abi_shape(
                    "JSON canonical object ordering requires one Dict argument".to_string(),
                    "C host JSON serialization",
                ));
            }
            self.lines.push(format!(
                "{}{target} = chelis_json_canonical_object_entries({});",
                self.indent, arg_vars[0].0
            ));
            return Ok(());
        }

        if name == "cast" {
            let (source_prim, source_surface) = checked_cast_abi_axis(&arg_vars[0].1)?;
            let (target_prim, target_surface) = checked_cast_abi_axis(ty)?;
            if source_surface != target_surface {
                return Err(checked_cast_plan_error(format!(
                    "checked cast resolved across surfaces: {:?} -> {:?}",
                    arg_vars[0].1, ty
                )));
            }
            let plan = CheckedCastPlan::new(source_prim, target_prim)
                .map_err(|error| checked_cast_plan_error(error.to_string()))?;
            self.lines.push(format!(
                "{}/* checked cast plan: {} -> {} */",
                self.indent,
                source_prim.name(),
                target_prim.name()
            ));
            match source_surface {
                CheckedCastSurface::Scalar => {
                    if plan.kind() == CheckedCastKind::Identity {
                        self.lines
                            .push(format!("{}/* checked cast identity */", self.indent));
                    }
                    let expr = checked_cast_c_expr(plan, &arg_vars[0].0);
                    self.lines
                        .push(format!("{}{target} = {expr};", self.indent));
                }
                CheckedCastSurface::Tensor => {
                    self.assign_checked_tensor_cast(target, &arg_vars[0].0, plan);
                }
            }
            return Ok(());
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
                        BinaryElementwiseFunc::Max,
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
                        BinaryElementwiseFunc::Min,
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
            // [05-OP-6]. Unlike `cast`, this arm has NO identity
            // fallback: the only legal pair is float source to integer
            // target, and anything else must be a loud emission failure
            // rather than a silent un-truncated pass-through.
            "cast_trunc" => {
                let expr = match (&arg_vars[0].1, ty) {
                    (source, target) if is_float_abi(source) && is_integer_abi(target) => {
                        let prim = integer_abi_prim(target)?;
                        let domain = NumericTrap::Domain {
                            op: "cast_trunc",
                            prim,
                        }
                        .to_string();
                        let overflow = NumericTrap::Overflow {
                            op: "cast_trunc",
                            prim,
                        }
                        .to_string();
                        format!(
                            "({})chelis_trunc_float_to_int({}, {}, {domain:?}, {overflow:?})",
                            c_type(target)?,
                            host_float_as_double(&arg_vars[0].0, source),
                            integer_abi_width(target)?
                        )
                    }
                    (source, target) => {
                        return Err(invalid_abi_shape(
                            format!(
                                "`cast_trunc` resolved to {source:?} -> {target:?}; \
                                 [05-OP-6] is float-to-integer only"
                            ),
                            "C host cast_trunc emission",
                        ));
                    }
                };
                self.lines
                    .push(format!("{}{target} = {};", self.indent, expr));
                return Ok(());
            }
            "copy" => {
                self.lines
                    .push(format!("{}{target} = {};", self.indent, arg_vars[0].0));
                self.record_pointer_copy(target, &arg_vars[0].0, &args[0], ty);
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
                    "{}if (strcmp(chelis_string_data({}), \"replace\") == 0) {{",
                    self.indent, arg_vars[4].0
                ));
                self.lines.push(format!(
                    "{}    {target} = chelis_tensor_scatter_replace({}, {}, {}, {});",
                    self.indent, arg_vars[0].0, arg_vars[1].0, arg_vars[2].0, arg_vars[3].0
                ));
                self.lines.push(format!(
                    "{}}} else if (strcmp(chelis_string_data({}), \"add\") == 0) {{",
                    self.indent, arg_vars[4].0
                ));
                self.lines.push(format!(
                    "{}    {target} = chelis_tensor_scatter_add({}, {}, {}, {});",
                    self.indent, arg_vars[0].0, arg_vars[1].0, arg_vars[2].0, arg_vars[3].0
                ));
                self.lines.push(format!("{}}} else {{ fprintf(stderr, \"scatter mode must be replace or add\\n\"); exit(1); }}", self.indent));
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
                if arg_vars.len() != 3 {
                    return Err(invalid_abi_shape(
                        format!(
                            "einsum reached C emission with {} arguments; expected equation and two operands",
                            arg_vars.len()
                        ),
                        "einsum accumulator selection",
                    ));
                }
                let Some((_, HostType::Tensor(lhs_ty))) = arg_vars.get(1) else {
                    return Err(invalid_abi_shape(
                        "einsum lhs does not carry a resolved tensor ABI".to_string(),
                        "einsum accumulator selection",
                    ));
                };
                // spec/04-type-system.md section 5.7.1 and [05-OP-33]
                // define einsum's omitted accumulator with the reduce-sum
                // default table. Materialize that resolved dtype at the
                // exact runtime boundary.
                let accumulator = lhs_ty
                    .precision
                    .default_reduce_sum_accumulator()
                    .and_then(|precision| {
                        precision.runtime_dtype().map_err(|error| error.to_string())
                    })
                    .map_err(|error| invalid_abi_shape(error, "einsum accumulator selection"))?;
                self.lines.push(format!(
                    "{}{target} = chelis_tensor_einsum({}, {}, {}, {});",
                    self.indent,
                    arg_vars[0].0,
                    arg_vars[1].0,
                    arg_vars[2].0,
                    accumulator.c_macro()
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
                let HostType::Option(inner) = ty else {
                    return Err(invalid_abi_shape(
                        format!("dict_get result has non-option ABI type `{ty:?}`"),
                        "dict_get result",
                    ));
                };
                let key = self.box_value_expr(&arg_vars[1].0, &arg_vars[1].1)?;
                if is_scalar_abi(inner.as_ref()) {
                    self.lines.push(format!(
                        "{}{target} = chelis_dict_get_scalar({}, {}, {});",
                        self.indent,
                        arg_vars[0].0,
                        key,
                        scalar_dtype_macro(inner.as_ref())?
                    ));
                } else {
                    self.lines.push(format!(
                        "{}{target} = chelis_dict_get({}, {});",
                        self.indent, arg_vars[0].0, key
                    ));
                }
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
                // [05-OP-33]: the checked result dtype selects the exact
                // tagged list-ingress constructor. A non-tensor or deferred
                // result is an IR/ABI disagreement, never an untyped fallback.
                let HostType::Tensor(t) = ty else {
                    return Err(invalid_abi_shape(
                        format!("to_tensor result has non-tensor ABI type `{ty:?}`"),
                        "to_tensor list ingress",
                    ));
                };
                let dtype = t.precision.runtime_dtype().map_err(|error| {
                    invalid_abi_shape(error.to_string(), "to_tensor list ingress")
                })?;
                self.lines.push(format!(
                    "{}{target} = chelis_tensor_from_values({}, {});",
                    self.indent,
                    arg_vars[0].0,
                    dtype.c_macro()
                ));
                return Ok(());
            }
            "to_list" => {
                self.lines.push(format!(
                    "{}{target} = chelis_tensor_elements({});",
                    self.indent, arg_vars[0].0
                ));
                return Ok(());
            }
            "pad_sequences" => {
                self.lines.push(format!(
                    "{}{target} = chelis_pad_sequences({}, {});",
                    self.indent,
                    arg_vars[0].0,
                    scalar_carrier_expr(&arg_vars[1].0, &arg_vars[1].1)?
                ));
                return Ok(());
            }
            "pad_sequences_to" => {
                self.lines.push(format!(
                    "{}{target} = chelis_pad_sequences_to({}, {}, {});",
                    self.indent,
                    arg_vars[0].0,
                    arg_vars[1].0,
                    scalar_carrier_expr(&arg_vars[2].0, &arg_vars[2].1)?
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
            "bitand",
            "bitor",
            "bitxor",
            "shl",
            "shr",
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
            "tan",
            "atan",
            "tanh",
            "floor",
            "ceil",
            "round",
            "recip",
            "pow",
            "abs",
            "min",
            "max",
            "min_elem",
            "max_elem",
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
                chelis_types::deliberate_rejection!(
                    "[04-TOT-2]",
                    "a checked tensor operation must route through the typed DAG lane; the C \
                     host scalar lane has no fallback tensor expression"
                ),
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
                self.record_pointer_copy(target, &arg_vars[0].0, &args[0], ty);
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
            let numeric_arg =
                |index: usize| scalar_arithmetic_arg_expr(&arg_vars[index].0, &arg_vars[index].1);
            let binary = |operator, lhs, rhs| EmittedExpr::binary(operator, lhs, rhs);
            let unary = |operator, operand| EmittedExpr::unary(operator, operand);
            let expression_builtin = CExpressionBuiltin::decode(name)?;
            let expr = match expression_builtin {
                CExpressionBuiltin::Add if is_integer_abi(ty) => integer_checked_binary_expr(
                    "chelis_int_checked_add",
                    "add",
                    arg(0),
                    arg(1),
                    ty,
                )?,
                CExpressionBuiltin::Add => finalize_scalar_expr(
                    binary(BinaryOperator::Add, numeric_arg(0), numeric_arg(1)),
                    ty,
                ),
                CExpressionBuiltin::Sub if is_integer_abi(ty) => integer_checked_binary_expr(
                    "chelis_int_checked_sub",
                    "sub",
                    arg(0),
                    arg(1),
                    ty,
                )?,
                CExpressionBuiltin::Sub => finalize_scalar_expr(
                    binary(BinaryOperator::Subtract, numeric_arg(0), numeric_arg(1)),
                    ty,
                ),
                CExpressionBuiltin::Mul if is_integer_abi(ty) => integer_checked_binary_expr(
                    "chelis_int_checked_mul",
                    "mul",
                    arg(0),
                    arg(1),
                    ty,
                )?,
                CExpressionBuiltin::Mul => finalize_scalar_expr(
                    binary(BinaryOperator::Multiply, numeric_arg(0), numeric_arg(1)),
                    ty,
                ),
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
                CExpressionBuiltin::Div => finalize_scalar_expr(
                    binary(BinaryOperator::Divide, numeric_arg(0), numeric_arg(1)),
                    ty,
                ),
                // chelis#178: `trunc_div` is integer-only — the guarded C `/`
                // quotient (round toward zero).
                CExpressionBuiltin::TruncDiv => binary(
                    BinaryOperator::Divide,
                    arg(0),
                    checked_integer_divisor_expr("trunc_div", arg(0), arg(1), ty)?,
                ),
                // chelis#178: `floor_div` rounds toward -inf. Integer (host
                // scalar) operands use the guarded `/` plus a remainder-sign
                // correction; float operands use `floor(a / b)`.
                CExpressionBuiltin::FloorDiv if is_integer_abi(&arg_vars[0].1) => {
                    let guarded_divisor =
                        || checked_integer_divisor_expr("floor_div", arg(0), arg(1), ty);
                    let quotient = binary(BinaryOperator::Divide, arg(0), guarded_divisor()?);
                    let remainder = || {
                        Ok::<_, Unsupported>(binary(
                            BinaryOperator::Remainder,
                            arg(0),
                            guarded_divisor()?,
                        ))
                    };
                    let nonzero = binary(
                        BinaryOperator::NotEqual,
                        remainder()?,
                        EmittedExpr::integer(0),
                    );
                    let sign_differs = binary(
                        BinaryOperator::NotEqual,
                        binary(BinaryOperator::Less, remainder()?, EmittedExpr::integer(0)),
                        binary(BinaryOperator::Less, arg(1), EmittedExpr::integer(0)),
                    );
                    let correction = EmittedExpr::conditional(
                        binary(BinaryOperator::LogicalAnd, nonzero, sign_differs),
                        EmittedExpr::integer(1),
                        EmittedExpr::integer(0),
                    );
                    binary(BinaryOperator::Subtract, quotient, correction)
                }
                CExpressionBuiltin::FloorDiv => finalize_scalar_expr(
                    EmittedExpr::call(
                        float_math_function(ty, "floor", "floorf"),
                        [binary(
                            BinaryOperator::Divide,
                            numeric_arg(0),
                            numeric_arg(1),
                        )],
                    ),
                    ty,
                ),
                CExpressionBuiltin::Mod => EmittedExpr::conditional(
                    binary(BinaryOperator::Equal, arg(1), EmittedExpr::integer(-1)),
                    EmittedExpr::integer(0),
                    binary(
                        BinaryOperator::Remainder,
                        arg(0),
                        checked_integer_divisor_expr("mod", arg(0), arg(1), ty)?,
                    ),
                ),
                CExpressionBuiltin::BitAnd => binary(BinaryOperator::BitAnd, arg(0), arg(1)),
                CExpressionBuiltin::BitOr => binary(BinaryOperator::BitOr, arg(0), arg(1)),
                CExpressionBuiltin::BitXor => binary(BinaryOperator::BitXor, arg(0), arg(1)),
                // [04-NUM-13]: never emit raw signed C shifts. The runtime
                // helper implements declared-width two's-complement movement,
                // including negative-count traps and fully shifted-out values,
                // without C undefined or implementation-defined behavior.
                CExpressionBuiltin::ShiftLeft => EmittedExpr::call(
                    "chelis_int_shl",
                    [
                        arg(0),
                        arg(1),
                        EmittedExpr::integer(integer_abi_width(&arg_vars[0].1)?),
                    ],
                ),
                CExpressionBuiltin::ShiftRight => EmittedExpr::call(
                    "chelis_int_shr",
                    [
                        arg(0),
                        arg(1),
                        EmittedExpr::integer(integer_abi_width(&arg_vars[0].1)?),
                    ],
                ),
                CExpressionBuiltin::CompareLess if matches!(arg_vars[0].1, HostType::Tensor(_)) => {
                    EmittedExpr::call("chelis_tensor_cmplt", [arg(0), arg(1)])
                }
                CExpressionBuiltin::CompareLess | CExpressionBuiltin::Less => {
                    binary(BinaryOperator::Less, numeric_arg(0), numeric_arg(1))
                }
                CExpressionBuiltin::Greater => {
                    binary(BinaryOperator::Greater, numeric_arg(0), numeric_arg(1))
                }
                CExpressionBuiltin::GreaterEqual => {
                    binary(BinaryOperator::GreaterEqual, numeric_arg(0), numeric_arg(1))
                }
                CExpressionBuiltin::LessEqual => {
                    binary(BinaryOperator::LessEqual, numeric_arg(0), numeric_arg(1))
                }
                CExpressionBuiltin::Equal => match (&arg_vars[0].1, &arg_vars[1].1) {
                    (HostType::String, HostType::String) => {
                        EmittedExpr::call("chelis_string_eq", [arg(0), arg(1)])
                    }
                    _ => binary(BinaryOperator::Equal, numeric_arg(0), numeric_arg(1)),
                },
                CExpressionBuiltin::NotEqual => match (&arg_vars[0].1, &arg_vars[1].1) {
                    (HostType::String, HostType::String) => unary(
                        UnaryOperator::LogicalNot,
                        EmittedExpr::call("chelis_string_eq", [arg(0), arg(1)]),
                    ),
                    _ => binary(BinaryOperator::NotEqual, numeric_arg(0), numeric_arg(1)),
                },
                CExpressionBuiltin::And => binary(BinaryOperator::LogicalAnd, arg(0), arg(1)),
                CExpressionBuiltin::Or => binary(BinaryOperator::LogicalOr, arg(0), arg(1)),
                CExpressionBuiltin::Not => unary(UnaryOperator::LogicalNot, arg(0)),
                CExpressionBuiltin::Neg if is_integer_abi(ty) => EmittedExpr::call(
                    "chelis_int_checked_neg",
                    [
                        arg(0),
                        EmittedExpr::integer(integer_abi_width(ty)?),
                        EmittedExpr::string_literal(integer_trap_message(ty, "neg", true)?),
                    ],
                ),
                CExpressionBuiltin::Neg => {
                    finalize_scalar_expr(unary(UnaryOperator::Negate, numeric_arg(0)), ty)
                }
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
                    HostType::Int8 => EmittedExpr::call(
                        "chelis_string_from_scalar",
                        [EmittedExpr::call("chelis_host_scalar_from_i8", [arg(0)])],
                    ),
                    HostType::Int16 => EmittedExpr::call(
                        "chelis_string_from_scalar",
                        [EmittedExpr::call("chelis_host_scalar_from_i16", [arg(0)])],
                    ),
                    HostType::Int32 => EmittedExpr::call(
                        "chelis_string_from_scalar",
                        [EmittedExpr::call("chelis_host_scalar_from_i32", [arg(0)])],
                    ),
                    HostType::Int64 => EmittedExpr::call(
                        "chelis_string_from_scalar",
                        [EmittedExpr::call("chelis_host_scalar_from_i64", [arg(0)])],
                    ),
                    HostType::Float64 => EmittedExpr::call(
                        "chelis_string_from_scalar",
                        [EmittedExpr::call("chelis_host_scalar_from_f64", [arg(0)])],
                    ),
                    // to_string is an observation exit: the f32 scalar
                    // renders at ITS width through the runtime's own-width
                    // formatter ([05-OBS-2]; the former promote-to-double
                    // funnel carried f64-image digits and split this exit
                    // from `print` of the same stored value - PR #863
                    // round-1 F1).
                    HostType::Float32 => EmittedExpr::call(
                        "chelis_string_from_scalar",
                        [EmittedExpr::call("chelis_host_scalar_from_f32", [arg(0)])],
                    ),
                    HostType::Float16 => EmittedExpr::call("chelis_host_string_from_f16", [arg(0)]),
                    HostType::BFloat16 => {
                        EmittedExpr::call("chelis_host_string_from_bf16", [arg(0)])
                    }
                    HostType::Bool => EmittedExpr::call(
                        "chelis_string_from_scalar",
                        [EmittedExpr::call("chelis_host_scalar_from_bool", [arg(0)])],
                    ),
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
                            chelis_types::unimplemented_rejection!(
                                1059,
                                "the compiled lane stringifies admitted numeric/bool/string scalars \
                                 only today; chelis#1059 owns compiled tensor/list rendering \
                                 (the former `<value>` placeholder is chelis#734)"
                            ),
                        ));
                    }
                },
                CExpressionBuiltin::ToInt => EmittedExpr::call(
                    "chelis_parse_scalar",
                    [arg(0), EmittedExpr::identifier("CHELIS_DTYPE_I64")],
                ),
                CExpressionBuiltin::ToFloat => EmittedExpr::call(
                    "chelis_parse_scalar",
                    [arg(0), EmittedExpr::identifier("CHELIS_DTYPE_F64")],
                ),
                CExpressionBuiltin::TensorToScalar => match ty {
                    HostType::Float64 => EmittedExpr::call(
                        "chelis_host_scalar_as_float",
                        [
                            EmittedExpr::call("chelis_tensor_to_scalar", [arg(0)]),
                            EmittedExpr::identifier("CHELIS_DTYPE_F64"),
                        ],
                    ),
                    HostType::Float32 => EmittedExpr::call(
                        "chelis_host_scalar_as_float",
                        [
                            EmittedExpr::call("chelis_tensor_to_scalar", [arg(0)]),
                            EmittedExpr::identifier("CHELIS_DTYPE_F32"),
                        ],
                    ),
                    HostType::Float16 => EmittedExpr::call(
                        "chelis_host_scalar_bits",
                        [
                            EmittedExpr::call("chelis_tensor_to_scalar", [arg(0)]),
                            EmittedExpr::identifier("CHELIS_DTYPE_F16"),
                        ],
                    ),
                    HostType::BFloat16 => EmittedExpr::call(
                        "chelis_host_scalar_bits",
                        [
                            EmittedExpr::call("chelis_tensor_to_scalar", [arg(0)]),
                            EmittedExpr::identifier("CHELIS_DTYPE_BF16"),
                        ],
                    ),
                    HostType::Int8 => EmittedExpr::call(
                        "chelis_host_scalar_as_i64",
                        [
                            EmittedExpr::call("chelis_tensor_to_scalar", [arg(0)]),
                            EmittedExpr::identifier("CHELIS_DTYPE_I8"),
                        ],
                    ),
                    HostType::Int16 => EmittedExpr::call(
                        "chelis_host_scalar_as_i64",
                        [
                            EmittedExpr::call("chelis_tensor_to_scalar", [arg(0)]),
                            EmittedExpr::identifier("CHELIS_DTYPE_I16"),
                        ],
                    ),
                    HostType::Int32 => EmittedExpr::call(
                        "chelis_host_scalar_as_i64",
                        [
                            EmittedExpr::call("chelis_tensor_to_scalar", [arg(0)]),
                            EmittedExpr::identifier("CHELIS_DTYPE_I32"),
                        ],
                    ),
                    HostType::Int64 => EmittedExpr::call(
                        "chelis_host_scalar_as_i64",
                        [
                            EmittedExpr::call("chelis_tensor_to_scalar", [arg(0)]),
                            EmittedExpr::identifier("CHELIS_DTYPE_I64"),
                        ],
                    ),
                    HostType::Bool => EmittedExpr::call(
                        "chelis_host_scalar_as_bool",
                        [EmittedExpr::call("chelis_tensor_to_scalar", [arg(0)])],
                    ),
                    other => {
                        return Err(invalid_abi_shape(
                            format!("tensor_to_scalar carries non-scalar result type `{other:?}`"),
                            "tensor_to_scalar C emission",
                        ));
                    }
                },
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
                        Prim::Int8 => EmittedExpr::call(
                            "chelis_scalar_tensor",
                            [EmittedExpr::call("chelis_host_scalar_from_i8", [arg(0)])],
                        ),
                        Prim::Int16 => EmittedExpr::call(
                            "chelis_scalar_tensor",
                            [EmittedExpr::call("chelis_host_scalar_from_i16", [arg(0)])],
                        ),
                        Prim::Int32 => EmittedExpr::call(
                            "chelis_scalar_tensor",
                            [EmittedExpr::call("chelis_host_scalar_from_i32", [arg(0)])],
                        ),
                        Prim::Int64 => EmittedExpr::call(
                            "chelis_scalar_tensor",
                            [EmittedExpr::call("chelis_host_scalar_from_i64", [arg(0)])],
                        ),
                        Prim::F64 => EmittedExpr::call(
                            "chelis_scalar_tensor",
                            [EmittedExpr::call("chelis_host_scalar_from_f64", [arg(0)])],
                        ),
                        Prim::F32 => EmittedExpr::call(
                            "chelis_scalar_tensor",
                            [EmittedExpr::call("chelis_host_scalar_from_f32", [arg(0)])],
                        ),
                        Prim::F16 => {
                            EmittedExpr::call("chelis_host_scalar_tensor_from_f16", [arg(0)])
                        }
                        Prim::Bf16 => {
                            EmittedExpr::call("chelis_host_scalar_tensor_from_bf16", [arg(0)])
                        }
                        Prim::Bool => EmittedExpr::call(
                            "chelis_scalar_tensor",
                            [EmittedExpr::call("chelis_host_scalar_from_bool", [arg(0)])],
                        ),
                        precision => {
                            return Err(Unsupported::new(
                                UnsupportedKind::HostType(format!(
                                    "scalar_to_tensor<{}>",
                                    precision.name()
                                )),
                                "`scalar_to_tensor` C host emission",
                                Stage::Codegen("c"),
                                chelis_types::unimplemented_rejection!(
                                    729,
                                    "the resolved result dtype has no scalar-tensor constructor; \
                                     implement the exact target capability instead of selecting f32"
                                ),
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
                CExpressionBuiltin::Sqrt => finalize_scalar_expr(
                    EmittedExpr::call(float_math_function(ty, "sqrt", "sqrtf"), [numeric_arg(0)]),
                    ty,
                ),
                CExpressionBuiltin::Exp => finalize_scalar_expr(
                    EmittedExpr::call(float_math_function(ty, "exp", "expf"), [numeric_arg(0)]),
                    ty,
                ),
                CExpressionBuiltin::Log => finalize_scalar_expr(
                    EmittedExpr::call(float_math_function(ty, "log", "logf"), [numeric_arg(0)]),
                    ty,
                ),
                CExpressionBuiltin::Sin => finalize_scalar_expr(
                    EmittedExpr::call(float_math_function(ty, "sin", "sinf"), [numeric_arg(0)]),
                    ty,
                ),
                CExpressionBuiltin::Cos => finalize_scalar_expr(
                    EmittedExpr::call(float_math_function(ty, "cos", "cosf"), [numeric_arg(0)]),
                    ty,
                ),
                CExpressionBuiltin::Tan => finalize_scalar_expr(
                    EmittedExpr::call(float_math_function(ty, "tan", "tanf"), [numeric_arg(0)]),
                    ty,
                ),
                CExpressionBuiltin::Atan => finalize_scalar_expr(
                    EmittedExpr::call(float_math_function(ty, "atan", "atanf"), [numeric_arg(0)]),
                    ty,
                ),
                CExpressionBuiltin::Relu
                | CExpressionBuiltin::Sigmoid
                | CExpressionBuiltin::Tanh
                | CExpressionBuiltin::Silu
                | CExpressionBuiltin::Gelu
                    if !is_float_abi(ty) =>
                {
                    return Err(invalid_abi_shape(
                        format!(
                            "float activation `{name}` resolved to non-float result type `{ty:?}`"
                        ),
                        "C host scalar activation emission",
                    ));
                }
                CExpressionBuiltin::Relu => finalize_scalar_expr(
                    EmittedExpr::call(
                        activation_math_function(
                            ty,
                            "chelis_host_relu_f16",
                            "chelis_host_relu_bf16",
                            "chelis_host_relu_f32",
                            "chelis_host_relu_f64",
                        ),
                        [numeric_arg(0)],
                    ),
                    ty,
                ),
                CExpressionBuiltin::Sigmoid => finalize_scalar_expr(
                    EmittedExpr::call(
                        activation_math_function(
                            ty,
                            "chelis_host_sigmoid_f16",
                            "chelis_host_sigmoid_bf16",
                            "chelis_host_sigmoid_f32",
                            "chelis_host_sigmoid_f64",
                        ),
                        [numeric_arg(0)],
                    ),
                    ty,
                ),
                CExpressionBuiltin::Tanh => finalize_scalar_expr(
                    EmittedExpr::call(
                        activation_math_function(
                            ty,
                            "chelis_host_tanh_f16",
                            "chelis_host_tanh_bf16",
                            "chelis_host_tanh_f32",
                            "chelis_host_tanh_f64",
                        ),
                        [numeric_arg(0)],
                    ),
                    ty,
                ),
                CExpressionBuiltin::Silu => finalize_scalar_expr(
                    EmittedExpr::call(
                        activation_math_function(
                            ty,
                            "chelis_host_silu_f16",
                            "chelis_host_silu_bf16",
                            "chelis_host_silu_f32",
                            "chelis_host_silu_f64",
                        ),
                        [numeric_arg(0)],
                    ),
                    ty,
                ),
                CExpressionBuiltin::Gelu => finalize_scalar_expr(
                    EmittedExpr::call(
                        activation_math_function(
                            ty,
                            "chelis_host_gelu_f16",
                            "chelis_host_gelu_bf16",
                            "chelis_host_gelu_f32",
                            "chelis_host_gelu_f64",
                        ),
                        [numeric_arg(0)],
                    ),
                    ty,
                ),
                CExpressionBuiltin::Floor
                | CExpressionBuiltin::Ceil
                | CExpressionBuiltin::Round
                    if is_integer_abi(ty) =>
                {
                    arg(0)
                }
                CExpressionBuiltin::Floor => finalize_scalar_expr(
                    EmittedExpr::call(float_math_function(ty, "floor", "floorf"), [numeric_arg(0)]),
                    ty,
                ),
                CExpressionBuiltin::Ceil => finalize_scalar_expr(
                    EmittedExpr::call(float_math_function(ty, "ceil", "ceilf"), [numeric_arg(0)]),
                    ty,
                ),
                CExpressionBuiltin::Round => finalize_scalar_expr(
                    // spec/05 §2.2: `round` is IEEE roundTiesToEven. The C
                    // `round{,f}` family resolves half ties away from zero;
                    // `rint{,f}` under the default rounding mode matches the
                    // evaluator and the typed-DAG C emitter.
                    EmittedExpr::call(float_math_function(ty, "rint", "rintf"), [numeric_arg(0)]),
                    ty,
                ),
                CExpressionBuiltin::Recip => finalize_scalar_expr(
                    binary(
                        BinaryOperator::Divide,
                        EmittedExpr::integer(1),
                        numeric_arg(0),
                    ),
                    ty,
                ),
                CExpressionBuiltin::Pow => finalize_scalar_expr(
                    EmittedExpr::call(
                        float_math_function(ty, "pow", "powf"),
                        [numeric_arg(0), numeric_arg(1)],
                    ),
                    ty,
                ),
                CExpressionBuiltin::Abs => match arg_vars[0].1 {
                    HostType::Int8 | HostType::Int16 | HostType::Int32 | HostType::Int64 => {
                        let prim = match arg_vars[0].1 {
                            HostType::Int8 => Prim::Int8,
                            HostType::Int16 => Prim::Int16,
                            HostType::Int32 => Prim::Int32,
                            HostType::Int64 => Prim::Int64,
                            _ => unreachable!(),
                        };
                        let message = NumericTrap::Overflow { op: "abs", prim }.to_string();
                        EmittedExpr::call(
                            "chelis_int_abs_guard",
                            [
                                arg(0),
                                EmittedExpr::integer(integer_abi_width(&arg_vars[0].1)?),
                                EmittedExpr::string_literal(message),
                            ],
                        )
                    }
                    HostType::Float16
                    | HostType::BFloat16
                    | HostType::Float32
                    | HostType::Float64 => finalize_scalar_expr(
                        EmittedExpr::call(
                            float_math_function(ty, "fabs", "fabsf"),
                            [numeric_arg(0)],
                        ),
                        ty,
                    ),
                    ref other => {
                        return Err(invalid_abi_shape(
                            format!("abs carries non-numeric argument type `{other:?}`"),
                            "abs builtin",
                        ));
                    }
                },
                CExpressionBuiltin::Min => finalize_scalar_expr(
                    EmittedExpr::call(
                        float_math_function(ty, "fmin", "fminf"),
                        [numeric_arg(0), numeric_arg(1)],
                    ),
                    ty,
                ),
                CExpressionBuiltin::Max => finalize_scalar_expr(
                    EmittedExpr::call(
                        float_math_function(ty, "fmax", "fmaxf"),
                        [numeric_arg(0), numeric_arg(1)],
                    ),
                    ty,
                ),
                CExpressionBuiltin::MinElem if is_integer_abi(ty) => EmittedExpr::conditional(
                    binary(BinaryOperator::Less, arg(0), arg(1)),
                    arg(0),
                    arg(1),
                ),
                CExpressionBuiltin::MaxElem if is_integer_abi(ty) => EmittedExpr::conditional(
                    binary(BinaryOperator::Greater, arg(0), arg(1)),
                    arg(0),
                    arg(1),
                ),
                CExpressionBuiltin::MinElem => finalize_scalar_expr(
                    EmittedExpr::call(
                        float_math_function(ty, "fmin", "fminf"),
                        [numeric_arg(0), numeric_arg(1)],
                    ),
                    ty,
                ),
                CExpressionBuiltin::MaxElem => finalize_scalar_expr(
                    EmittedExpr::call(
                        float_math_function(ty, "fmax", "fmaxf"),
                        [numeric_arg(0), numeric_arg(1)],
                    ),
                    ty,
                ),
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
    // Each arm selects an element type that matches its representation.
    // CHELIS_DTYPE_BOOL uses the canonical one-byte `uint8_t` payload.
    // CHELIS_DTYPE_F32 uses `(float*)`, CHELIS_DTYPE_I32 uses `(int32_t*)`,
    // CHELIS_DTYPE_F64 uses `(double*)`, and CHELIS_DTYPE_I64 uses `(int64_t*)`.
    //
    // The four helpers split into two pairs:
    //
    //   * `assign_tensor_binary_elementwise` /
    //     `assign_tensor_unary_elementwise` take a raw C operator
    //     (`+`, `-`, `*`, `/`, `!`, unary `-`) and emit the operator
    //     for every supported dtype arm.  All arms are semantically
    //     well-defined for the supported operators.
    //
    //   * `assign_tensor_binary_func_elementwise` emits f32 libm calls
    //     for F32. Its I32 arm emits an exact integer comparison for max
    //     and min. F64, I64, and Bool abort.
    //
    //   * `assign_tensor_unary_func_elementwise` takes an f32-only helper
    //     name (`expf`, `chelis_host_relu_f32`, ...). It accepts F32.
    //     I32, F64, I64, and Bool abort rather than convert through
    //     binary32 or treat bool storage as a float payload.
    fn assign_checked_tensor_cast(&mut self, target: &str, input: &str, plan: CheckedCastPlan) {
        if plan.kind() == CheckedCastKind::Identity {
            self.lines
                .push(format!("{}/* checked cast identity */", self.indent));
            self.lines.push(format!(
                "{}{target} = chelis_contiguous({input});",
                self.indent
            ));
            return;
        }

        let source_prim = plan.source();
        let target_prim = plan.target();
        let source_type = sparse_elem_type(source_prim);
        let target_type = sparse_elem_type(target_prim);
        let source_data = format!("{target}_cast_source");
        let target_data = format!("{target}_cast_target");
        let flat_index = format!("{target}_cast_i");
        let indices = format!("{target}_cast_indices");
        let source_index = format!("{target}_cast_source_i");
        self.lines.push(format!(
            "{}{target} = chelis_alloc({input}->rank, {input}->shape, {});",
            self.indent,
            sparse_dtype_macro(target_prim)
        ));
        self.lines.push(format!(
            "{}if ({input}->dtype != {}) {{",
            self.indent,
            sparse_dtype_macro(source_prim)
        ));
        self.lines.push(format!(
            "{}    fprintf(stderr, \"checked cast source dtype contract mismatch\\n\");",
            self.indent
        ));
        self.lines.push(format!("{}    abort();", self.indent));
        self.lines.push(format!("{}}}", self.indent));
        self.lines.push(format!(
            "{}const {source_type} *{source_data} = (const {source_type} *){input}->data;",
            self.indent,
        ));
        self.lines.push(format!(
            "{}{target_type} *{target_data} = ({target_type} *){target}->data;",
            self.indent,
        ));
        self.lines.push(format!(
            "{}for (int64_t {flat_index} = 0; {flat_index} < {target}->size; {flat_index}++) {{",
            self.indent,
        ));
        self.lines.push(format!(
            "{}    int64_t {indices}[{target}->rank > 0 ? {target}->rank : 1];",
            self.indent
        ));
        self.lines.push(format!(
            "{}    chelis_flat_to_indices({flat_index}, {target}->shape, {target}->rank, {indices});",
            self.indent,
        ));
        self.lines.push(format!(
            "{}    int64_t {source_index} = chelis_indices_to_flat({indices}, {input}->strides, {input}->rank);",
            self.indent,
        ));
        let source_value = format!("{source_data}[{source_index}]");
        let expression = checked_cast_c_expr(plan, &source_value);
        self.lines.push(format!(
            "{}    {target_data}[{flat_index}] = {expression};",
            self.indent,
        ));
        self.lines.push(format!("{}}}", self.indent));
    }

    fn assign_tensor_binary_elementwise(&mut self, target: &str, lhs: &str, rhs: &str, op: &str) {
        self.lines.push(format!(
            "{}{target} = chelis_alloc({lhs}->rank, {lhs}->shape, {lhs}->dtype);",
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
        func: BinaryElementwiseFunc,
    ) {
        self.lines.push(format!(
            "{}{target} = chelis_alloc({lhs}->rank, {lhs}->shape, {lhs}->dtype);",
            self.indent
        ));
        self.lines
            .push(format!("{}switch ({target}->dtype) {{", self.indent));
        for arm in DtypeArm::f32_payload_func_arms() {
            self.emit_binary_func_elementwise_arm(target, lhs, rhs, func, *arm);
        }
        self.emit_binary_func_elementwise_arm(target, lhs, rhs, func, DtypeArm::I32);
        self.emit_dtype_fail_arms(
            &[DtypeArm::F64, DtypeArm::I64, DtypeArm::Bool],
            &format!("binary func elementwise ({})", func.f32_name()),
        );
        self.emit_default_runtime_fail_arm_for(
            target,
            &format!("binary func elementwise ({})", func.f32_name()),
        );
        self.lines.push(format!("{}}}", self.indent));
    }

    fn assign_tensor_unary_elementwise(&mut self, target: &str, input: &str, op: &str) {
        self.lines.push(format!(
            "{}{target} = chelis_alloc({input}->rank, {input}->shape, {input}->dtype);",
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
            "{}{target} = chelis_alloc({input}->rank, {input}->shape, {input}->dtype);",
            self.indent
        ));
        self.lines
            .push(format!("{}switch ({target}->dtype) {{", self.indent));
        for arm in DtypeArm::f32_payload_func_arms() {
            self.emit_unary_func_elementwise_arm(target, input, func, *arm);
        }
        self.emit_dtype_fail_arms(
            &[DtypeArm::F64, DtypeArm::I32, DtypeArm::I64, DtypeArm::Bool],
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
            "{ind}        for (int64_t i = 0; i < {target}->size; i++) {{"
        ));
        self.lines.push(format!(
            "{ind}            int64_t indices[{target}->rank > 0 ? {target}->rank : 1];"
        ));
        self.lines.push(format!(
            "{ind}            chelis_flat_to_indices(i, {target}->shape, {target}->rank, indices);"
        ));
        self.lines.push(format!(
            "{ind}            int64_t idx_lhs = chelis_indices_to_flat(indices, {lhs}->strides, {lhs}->rank);"
        ));
        self.lines.push(format!(
            "{ind}            int64_t idx_rhs = chelis_indices_to_flat(indices, {rhs}->strides, {rhs}->rank);"
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
        func: BinaryElementwiseFunc,
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
            "{ind}        for (int64_t i = 0; i < {target}->size; i++) {{"
        ));
        self.lines.push(format!(
            "{ind}            int64_t indices[{target}->rank > 0 ? {target}->rank : 1];"
        ));
        self.lines.push(format!(
            "{ind}            chelis_flat_to_indices(i, {target}->shape, {target}->rank, indices);"
        ));
        self.lines.push(format!(
            "{ind}            int64_t idx_lhs = chelis_indices_to_flat(indices, {lhs}->strides, {lhs}->rank);"
        ));
        self.lines.push(format!(
            "{ind}            int64_t idx_rhs = chelis_indices_to_flat(indices, {rhs}->strides, {rhs}->rank);"
        ));
        let expression = match arm {
            DtypeArm::F32 => format!(
                "{}(__lhs_data[idx_lhs], __rhs_data[idx_rhs])",
                func.f32_name()
            ),
            DtypeArm::I32 => format!(
                "__lhs_data[idx_lhs] {} __rhs_data[idx_rhs] ? __lhs_data[idx_lhs] : __rhs_data[idx_rhs]",
                func.i32_comparison()
            ),
            DtypeArm::F64 | DtypeArm::I64 | DtypeArm::Bool => {
                unreachable!("binary func arm must reject non-f32-function dtypes before emission")
            }
        };
        self.lines
            .push(format!("{ind}            __target_data[i] = {expression};"));
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
            "{ind}        for (int64_t i = 0; i < {target}->size; i++) {{"
        ));
        self.lines.push(format!(
            "{ind}            int64_t indices[{target}->rank > 0 ? {target}->rank : 1];"
        ));
        self.lines.push(format!(
            "{ind}            chelis_flat_to_indices(i, {target}->shape, {target}->rank, indices);"
        ));
        self.lines.push(format!(
            "{ind}            int64_t idx = chelis_indices_to_flat(indices, {input}->strides, {input}->rank);"
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
            "{ind}        for (int64_t i = 0; i < {target}->size; i++) {{"
        ));
        self.lines.push(format!(
            "{ind}            int64_t indices[{target}->rank > 0 ? {target}->rank : 1];"
        ));
        self.lines.push(format!(
            "{ind}            chelis_flat_to_indices(i, {target}->shape, {target}->rank, indices);"
        ));
        self.lines.push(format!(
            "{ind}            int64_t idx = chelis_indices_to_flat(indices, {input}->strides, {input}->rank);"
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
                // the declared dtype keeps shape=[] across generated host/DAG
                // calls. Every arm writes through its exact storage type;
                // Bool is the canonical one-byte Bool8 carrier.
                let (dtype, store) = match inferred_ty {
                    HostType::Int8 => (
                        "CHELIS_DTYPE_I8",
                        format!("((int8_t*){tensor_name}->data)[0] = {value_name};"),
                    ),
                    HostType::Int16 => (
                        "CHELIS_DTYPE_I16",
                        format!("((int16_t*){tensor_name}->data)[0] = {value_name};"),
                    ),
                    HostType::Int64 => (
                        "CHELIS_DTYPE_I64",
                        format!("((int64_t*){tensor_name}->data)[0] = {value_name};"),
                    ),
                    HostType::Bool => (
                        "CHELIS_DTYPE_BOOL",
                        format!(
                            "((uint8_t*){tensor_name}->data)[0] = {value_name} ? UINT8_C(1) : UINT8_C(0);"
                        ),
                    ),
                    // #381: an f64 captured scalar (e.g. `cast(1.1, f64)`)
                    // fed to a tensor helper via `scalar_to_tensor` must be
                    // packed into a `CHELIS_DTYPE_F64` rank-0 tensor and written
                    // through a `double*`. The pre-fix catch-all packed it
                    // as `CHELIS_DTYPE_F32` and stored only the low 4 bytes; the
                    // f64 kernel then read 8 bytes (the high 4 garbage),
                    // collapsing the value to ~0 and silently disagreeing
                    // with the evaluator. Float32 still uses the f32 arm.
                    HostType::Float64 => (
                        "CHELIS_DTYPE_F64",
                        format!("((double*){tensor_name}->data)[0] = (double)({value_name});"),
                    ),
                    _ => (
                        "CHELIS_DTYPE_F32",
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
        // chelis#1222: an identity helper's whole body is
        // `outputs[0] = inputs[0];` (see `identity_helper_input`), so its
        // result is its argument's pointer, not a fresh allocation.
        let identity_source = self
            .tensor_helpers
            .get(helper)
            .and_then(identity_helper_input)
            .and(tensor_args.first().map(|(name, _)| name.clone()));
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
            if let Some(source) = identity_source {
                self.record_alias(target, &source);
            }
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
            "{}int64_t {shape_name}[{}] = {{ {} }};",
            self.indent,
            output_dims.len(),
            output_dims.join(", ")
        ));
        self.lines.push(format!(
            "{}{target} = chelis_alloc({}, {shape_name}, CHELIS_DTYPE_F32);",
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
            "{}if (!({lhs_contig}->rank >= 2 && {lhs_contig}->strides[{lhs_contig}->rank - 1] == 1 && {lhs_contig}->strides[{lhs_contig}->rank - 2] == {k_expr})) {{",
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
            "{}if (!({rhs_contig}->rank >= 2 && {rhs_contig}->strides[{rhs_contig}->rank - 1] == 1 && {rhs_contig}->strides[{rhs_contig}->rank - 2] == {n_expr})) {{",
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
                "{}for (int64_t {batch} = 0; {batch} < {batch_count}; {batch}++) {{",
                self.indent
            ));
            self.lines
                .push(format!("{}    int64_t {rem} = {batch};", self.indent));
            self.lines
                .push(format!("{}    int64_t {lhs_offset} = 0;", self.indent));
            self.lines
                .push(format!("{}    int64_t {rhs_offset} = 0;", self.indent));
            self.lines
                .push(format!("{}    int64_t {out_offset} = 0;", self.indent));
            for axis in (0..summary.batch_dims.len()).rev() {
                let dim_expr =
                    self.summary_dim_expr(&summary.batch_dims[axis], summary, &tensor_args);
                let coord = self.next_temp(&format!("blas_coord_{axis}"));
                self.lines.push(format!(
                    "{}    int64_t {coord} = {rem} % ({dim_expr});",
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
            self.lines.push(format!(
                "{}if ({arg}->dtype != CHELIS_DTYPE_F32) {{",
                self.indent
            ));
            self.lines.push(format!(
                "{}    fprintf(stderr, \"specialized BLAS call input {input_index} expected f32 tensor\\n\");",
                self.indent
            ));
            self.lines.push(format!("{}    abort();", self.indent));
            self.lines.push(format!("{}}}", self.indent));
            self.lines.push(format!(
                "{}if ({arg}->rank != {}) {{",
                self.indent,
                ty.dims.len()
            ));
            self.lines.push(format!(
                "{}    fprintf(stderr, \"specialized BLAS call input {input_index} expected rank {}, got %d\\n\", {arg}->rank);",
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
                            "{}    fprintf(stderr, \"specialized BLAS call input {input_index} axis {axis} expected {size}, got %lld\\n\", (long long){arg}->shape[{axis}]);",
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
            "{}int64_t {shape_name}[{}] = {{ {} }};",
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
                "{}if ({arg}->rank != {}) {{",
                self.indent,
                ty.dims.len()
            ));
            self.lines.push(format!(
                "{}    fprintf(stderr, \"specialized sparse call input {input_index} expected rank {}, got %d\\n\", {arg}->rank);",
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
                            "{}    fprintf(stderr, \"specialized sparse call input {input_index} axis {axis} expected {size}, got %lld\\n\", (long long){arg}->shape[{axis}]);",
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
        self.lines.push(format!(
            "{}int64_t {target}_before = {before};",
            self.indent
        ));
        self.lines.push(format!(
            "{}int64_t {target}_axis_size = {axis_size};",
            self.indent
        ));
        self.lines
            .push(format!("{}int64_t {target}_after = {after};", self.indent));
        self.lines.push(format!(
            "{}int64_t {target}_index_count = {indices_ct}->size;",
            self.indent
        ));
        self.lines.push(format!(
            "{}for (int64_t {target}_b = 0; {target}_b < {target}_before; {target}_b++) {{",
            self.indent
        ));
        self.lines.push(format!(
            "{}    for (int64_t {target}_i = 0; {target}_i < {target}_index_count; {target}_i++) {{",
            self.indent
        ));
        self.lines.push(format!(
            "{}        int64_t {target}_g = ({indices_ct}->dtype == CHELIS_DTYPE_I64) ? (int64_t)((const int64_t*){indices_ct}->data)[{target}_i] : (int64_t)({indices_ct}_data)[{target}_i];",
            self.indent
        ));
        self.lines.push(format!(
            "{}        if ({target}_g < 0 || {target}_g >= {target}_axis_size) abort();",
            self.indent
        ));
        self.lines.push(format!(
            "{}        for (int64_t {target}_d = 0; {target}_d < {target}_after; {target}_d++) {{",
            self.indent
        ));
        self.lines.push(format!(
            "{}            int64_t {target}_out = (({target}_b * {target}_index_count + {target}_i) * {target}_after) + {target}_d;",
            self.indent
        ));
        self.lines.push(format!(
            "{}            int64_t {target}_src = (({target}_b * {target}_axis_size + {target}_g) * {target}_after) + {target}_d;",
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
        self.lines.push(format!(
            "{}int64_t {target}_before = {before};",
            self.indent
        ));
        self.lines.push(format!(
            "{}int64_t {target}_axis_size = {axis_size};",
            self.indent
        ));
        self.lines
            .push(format!("{}int64_t {target}_after = {after};", self.indent));
        self.lines.push(format!(
            "{}int64_t {target}_index_count = {indices_ct}->size;",
            self.indent
        ));
        self.lines.push(format!(
            "{}for (int64_t {target}_b = 0; {target}_b < {target}_before; {target}_b++) {{",
            self.indent
        ));
        self.lines.push(format!(
            "{}    for (int64_t {target}_i = 0; {target}_i < {target}_index_count; {target}_i++) {{",
            self.indent
        ));
        self.lines.push(format!(
            "{}        int64_t {target}_g = ({indices_ct}->dtype == CHELIS_DTYPE_I64) ? (int64_t)((const int64_t*){indices_ct}->data)[{target}_i] : (int64_t)({indices_ct}_data)[{target}_i];",
            self.indent
        ));
        self.lines.push(format!(
            "{}        if ({target}_g < 0 || {target}_g >= {target}_axis_size) abort();",
            self.indent
        ));
        self.lines.push(format!(
            "{}        for (int64_t {target}_d = 0; {target}_d < {target}_after; {target}_d++) {{",
            self.indent
        ));
        self.lines.push(format!(
            "{}            int64_t {target}_src = (({target}_b * {target}_index_count + {target}_i) * {target}_after) + {target}_d;",
            self.indent
        ));
        self.lines.push(format!(
            "{}            int64_t {target}_out = (({target}_b * {target}_axis_size + {target}_g) * {target}_after) + {target}_d;",
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
            // A def resolves through the emitted-name map; anything else
            // (a typed callback parameter, a local binding) was declared
            // through `c_ident`, so the reference must take the same
            // mapping or a reserved-word name diverges from its
            // declarator (chelis#840 review, finding 1).
            self.emitted_names
                .get(function)
                .map(|name| std::borrow::Cow::Borrowed(name.as_str()))
                .unwrap_or_else(|| c_ident(function)),
            arg_vars.join(", ")
        ));
        // chelis#1222: these two are halves of ONE judgement about the call
        // result and must not be decided independently. When the escape
        // retain fires, `target` owns a reference of its own and its scope
        // owes the matching release; recording a borrow provenance on top of
        // that would suppress the release and strand the retain, leaking one
        // reference per call. Only an un-retained result needs its
        // provenance traced.
        if !self.retain_call_escaped_args(target, function, args, ty) {
            self.record_call_result_provenance(target, function, &arg_vars, ty);
        }
        Ok(())
    }

    /// chelis#1222: record where a call's result pointer came from, so the
    /// receiving scope can tell an allocation the callee made from one it
    /// merely handed back.
    ///
    /// Two provenances are unsafe to claim. The callee may return one of
    /// its arguments, in which case the result is whatever that argument
    /// already aliased; or it may return a value read out of an enclosing
    /// scope (a captured top-level binding), in which case there is no
    /// local variable to name and the result is marked foreign outright.
    ///
    /// Precision comes from the same [`analyze_returns_arg`] summary the
    /// call-escape retain uses: a callee that demonstrably builds a fresh
    /// result records nothing and the caller claims it as usual. When more
    /// than one argument may be returned and more than one of them is
    /// itself an alias, the result is marked foreign rather than pinned to
    /// an arbitrary one of them: over-conservatism leaks at process exit,
    /// under-conservatism corrupts the heap.
    fn record_call_result_provenance(
        &mut self,
        target: &str,
        function: &str,
        arg_vars: &[String],
        ty: &HostType,
    ) {
        if release_call(target, ty).is_none() {
            return;
        }
        let callee = self.returns_arg.get(function).cloned();
        // An unsummarized callee (not a user function, or not yet in the
        // fixpoint) is treated as may-return-anything.
        if callee.as_ref().is_none_or(ReturnsArg::may_return_outer) {
            self.mark_foreign(target);
            return;
        }
        let mut aliased_roots: Vec<(String, String)> = Vec::new();
        for (index, arg_var) in arg_vars.iter().enumerate() {
            let may_return = callee.as_ref().is_none_or(|s| s.may_return(index));
            if !may_return {
                continue;
            }
            let root = self.alias_root(arg_var);
            // An argument temp that aliases nothing is left unrecorded, so
            // a result that is that same pointer is claimed here.
            //
            // That is right when the temp is genuinely untracked, and WRONG
            // when it is not: a fresh list literal built as an argument at
            // `main` scope IS tracked and does get its own release, so a
            // callee returning it leaves one allocation with two releases.
            // Reported as chelis#1356 with a repro; unchanged from the
            // parent commit, so it is not this change's regression, but do
            // not read the line above as a proof of anything.
            if root != *arg_var {
                // Keyed by root so two arguments that alias the SAME
                // allocation count once, but recorded as the argument
                // variable: `record_alias` must add a link to the chain,
                // never collapse it. The intermediate links are what a
                // chain walk reads ownership off, and jumping straight to
                // the root steps over them (chelis#1222).
                aliased_roots.push((root, arg_var.clone()));
            }
        }
        aliased_roots.sort();
        aliased_roots.dedup_by(|a, b| a.0 == b.0);
        match aliased_roots.as_slice() {
            [] => {}
            [(_, only)] => {
                let only = only.clone();
                self.record_alias(target, &only);
            }
            _ => self.mark_foreign(target),
        }
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
    ///
    /// Returns whether a retain was emitted, so the caller can keep the
    /// tracking decision consistent with it (chelis#1222).
    fn retain_call_escaped_args(
        &mut self,
        target: &str,
        function: &str,
        args: &[HostExpr],
        ty: &HostType,
    ) -> bool {
        // Only meaningful for a refcounted result with a retain primitive
        // and at least one open release-tracking `let` block.
        if retain_call(target, ty).is_none() || self.let_scopes.is_empty() {
            return false;
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
            // chelis#1222: resolve the reference before testing membership;
            // `bindings` holds alias keys, not spellings.
            let source_key = self.resolve_alias_key(name);
            let source_is_binding = self
                .let_scopes
                .iter()
                .any(|scope| scope.bindings.contains(&source_key));
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
            return true;
        }
        false
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
            // chelis#1222: one binder scope per arm. Each arm's pattern
            // bindings shadow any enclosing name they reuse, and their keys
            // carry no outgoing edge -- `chelis_adt_field` hands back an
            // independently retained handle, so the arm binding is not a
            // copy of anything this scope already owns.
            self.binder_keys.push(HashMap::new());
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
                self.bind_alias_key(&binding.name);
            }
            self.assign_expr(target, &arm.expr, expr_ty)?;
            self.binder_keys.pop();
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
        self.lines.push(format!(
            "{}{target} = chelis_list_with_capacity({});",
            self.indent, len_var
        ));
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
            "{}chelis_list_push({target}, {});",
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
        self.lines.push(format!(
            "{}{target} = chelis_list_with_capacity({});",
            self.indent, len_var
        ));
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
            "{}chelis_list_push({target}, {});",
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
        let list_var = self.next_temp("scan_list");
        self.emit_expr_to_var(list, &list_var, &host_type(list))?;
        let len_var = self.next_temp("scan_len");
        self.lines.push(format!(
            "{}int64_t {} = chelis_list_len({});",
            self.indent, len_var, list_var
        ));
        self.lines.push(format!(
            "{}{target} = chelis_list_with_capacity({});",
            self.indent, len_var
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
            "{}chelis_list_push({target}, {});",
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
                chelis_types::deliberate_rejection!(
                    "[04-TOT-2]",
                    "internal desync: the checker guarantees a two-list tuple type for \
                     partition results (chelis#730 census row 15)"
                ),
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
        let list_var = self.next_temp("partition_list");
        self.emit_expr_to_var(list, &list_var, &host_type(list))?;
        let len_var = self.next_temp("partition_len");
        self.lines.push(format!(
            "{}int64_t {} = chelis_list_len({});",
            self.indent, len_var, list_var
        ));
        self.lines.push(format!(
            "{}{} {} = chelis_list_with_capacity({});",
            self.indent,
            c_type(pass_ty)?,
            pass_var,
            len_var
        ));
        self.lines.push(format!(
            "{}{} {} = chelis_list_with_capacity({});",
            self.indent,
            c_type(fail_ty)?,
            fail_var,
            len_var
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
            "{}chelis_list_push({}, {});",
            self.indent, pass_var, item_value
        ));
        self.indent = then_previous;
        self.lines.push(format!("{}}} else {{", self.indent));
        let else_indent = format!("{}    ", self.indent);
        let else_previous = std::mem::replace(&mut self.indent, else_indent);
        self.lines.push(format!(
            "{}chelis_list_push({}, {});",
            self.indent, fail_var, item_value
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
        self.lines.push(format!(
            "{}{target} = chelis_list_with_capacity({});",
            self.indent, len_var
        ));
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
            "{}chelis_list_extend({target}, {});",
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
                    // `assign_call`, with the same `c_ident` fallback so a
                    // reserved-word callback PARAMETER referenced by name
                    // matches its mangled declarator.
                    self.emitted_names
                        .get(function)
                        .map(|name| std::borrow::Cow::Borrowed(name.as_str()))
                        .unwrap_or_else(|| c_ident(function)),
                    arg_vars.join(", ")
                ));
            }
            HostCallbackKind::Inline { params, body } => {
                // chelis#1222: a lambda parameter shadows any enclosing name
                // it reuses. Without a scope here, a parameter that happens
                // to reuse an outer binding's name made the emitter read the
                // OUTER binding's ownership facts for it -- which decided
                // whether a retain was emitted inside the loop, so the same
                // program leaked or did not depending on the parameter's
                // spelling. Edge-less, like the other extraction binders.
                self.binder_keys.push(HashMap::new());
                for (param, arg_var) in params.iter().zip(arg_vars.iter()) {
                    self.lines.push(format!(
                        "{}{} {} = {};",
                        self.indent,
                        c_type(&param.ty)?,
                        param.name,
                        arg_var
                    ));
                }
                for param in params {
                    self.bind_alias_key(&param.name);
                }
                self.assign_expr(target, body, &callback.ret_ty)?;
                self.binder_keys.pop();
            }
        }
        Ok(())
    }

    fn box_value_expr(&self, value: &str, ty: &HostType) -> Result<String, Unsupported> {
        Ok(match ty {
            HostType::Int8 => format!(
                "chelis_value_from_scalar(chelis_scalar_from_bits(CHELIS_DTYPE_I8, (uint64_t)(uint8_t)(int8_t){value}))"
            ),
            HostType::Int16 => format!(
                "chelis_value_from_scalar(chelis_scalar_from_bits(CHELIS_DTYPE_I16, (uint64_t)(uint16_t)(int16_t){value}))"
            ),
            HostType::Int32 => format!(
                "chelis_value_from_scalar(chelis_scalar_from_bits(CHELIS_DTYPE_I32, (uint64_t)(uint32_t)(int32_t){value}))"
            ),
            HostType::Int64 => format!(
                "chelis_value_from_scalar(chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)(int64_t){value}))"
            ),
            HostType::Float64 => format!(
                "chelis_value_from_scalar(chelis_scalar_from_bits(CHELIS_DTYPE_F64, chelis_host_f64_bits({value})))"
            ),
            HostType::Float32 => format!(
                "chelis_value_from_scalar(chelis_scalar_from_bits(CHELIS_DTYPE_F32, (uint64_t)chelis_host_f32_bits({value})))"
            ),
            HostType::Float16 => format!(
                "chelis_value_from_scalar(chelis_scalar_from_bits(CHELIS_DTYPE_F16, (uint64_t)(uint16_t){value}))"
            ),
            HostType::BFloat16 => format!(
                "chelis_value_from_scalar(chelis_scalar_from_bits(CHELIS_DTYPE_BF16, (uint64_t)(uint16_t){value}))"
            ),
            HostType::Bool => format!(
                "chelis_value_from_scalar(chelis_scalar_from_bits(CHELIS_DTYPE_BOOL, (uint64_t)({value} ? 1 : 0)))"
            ),
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
            HostType::Int8 => format!(
                "(int8_t)chelis_host_scalar_as_i64(chelis_value_as_scalar({value_expr}), CHELIS_DTYPE_I8)"
            ),
            HostType::Int16 => format!(
                "(int16_t)chelis_host_scalar_as_i64(chelis_value_as_scalar({value_expr}), CHELIS_DTYPE_I16)"
            ),
            HostType::Int32 => format!(
                "(int32_t)chelis_host_scalar_as_i64(chelis_value_as_scalar({value_expr}), CHELIS_DTYPE_I32)"
            ),
            HostType::Int64 => format!(
                "chelis_host_scalar_as_i64(chelis_value_as_scalar({value_expr}), CHELIS_DTYPE_I64)"
            ),
            HostType::Float64 => format!(
                "chelis_host_scalar_as_float(chelis_value_as_scalar({value_expr}), CHELIS_DTYPE_F64)"
            ),
            HostType::Float32 => format!(
                "(float)chelis_host_scalar_as_float(chelis_value_as_scalar({value_expr}), CHELIS_DTYPE_F32)"
            ),
            HostType::Float16 => format!(
                "(uint16_t)chelis_host_scalar_bits(chelis_value_as_scalar({value_expr}), CHELIS_DTYPE_F16)"
            ),
            HostType::BFloat16 => format!(
                "(uint16_t)chelis_host_scalar_bits(chelis_value_as_scalar({value_expr}), CHELIS_DTYPE_BF16)"
            ),
            HostType::Bool => {
                format!("chelis_host_scalar_as_bool(chelis_value_as_scalar({value_expr}))")
            }
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
            // chelis#732 Phase 2: scalar floats render through the
            // runtime's shortest-round-trip routine at their OWN width
            // (the f32 C value widens to its exact double image), never
            // through a fixed-precision printf (chelis#748).
            HostType::Float64 | HostType::Float32 | HostType::Float16 | HostType::BFloat16 => {
                let boxed = self.box_value_expr(value, ty)?;
                self.lines.push(format!(
                    "{}{{ chelis_value boxed = {boxed}; \
                     chelis_string text = chelis_string_from_scalar(chelis_value_as_scalar(boxed)); \
                     printf(\"%s\\n\", chelis_string_data(text)); \
                     chelis_string_release(text); }}",
                    self.indent
                ));
            }
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
                    chelis_types::deliberate_rejection!(
                        "[04-TOT-2]",
                        "the value's host type never resolved to a printable representation \
                         (chelis#714's Unknown chain); previously this compiled to the \
                         literal `<value>` placeholder"
                    ),
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
            // Same exact tagged-scalar path as `emit_print_value`.
            HostType::Float64 | HostType::Float32 | HostType::Float16 | HostType::BFloat16 => {
                let boxed = self.box_value_expr(value, ty)?;
                self.lines.push(format!(
                    "{}{{ chelis_value boxed = {boxed}; \
                     chelis_string text = chelis_string_from_scalar(chelis_value_as_scalar(boxed)); \
                     printf(\"%s\", chelis_string_data(text)); \
                     chelis_string_release(text); }}",
                    self.indent
                ));
            }
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
                    chelis_types::deliberate_rejection!(
                        "[04-TOT-2]",
                        "the value's host type never resolved to a printable representation \
                         (chelis#714's Unknown chain); previously this compiled to the \
                         literal `<value>` placeholder"
                    ),
                ));
            }
        }
        self.lines.push(format!("{}printf(\"\\n\");", self.indent));
        Ok(())
    }

    fn emit_manifest_root(
        &mut self,
        name: &str,
        value: &str,
        ty: &HostType,
        path: &[RootPathStep],
    ) -> Result<(), Unsupported> {
        if path.is_empty() {
            return self.emit_labeled_root(name, value, ty);
        }

        let mut current = value.to_string();
        let mut boxed_values = Vec::with_capacity(path.len());
        for (depth, step) in path.iter().enumerate() {
            let boxed = self.next_temp("manifest_root_field");
            let container = if depth == 0 {
                current.clone()
            } else {
                match step {
                    RootPathStep::Tuple(_) => format!("chelis_value_as_tuple({current})"),
                    RootPathStep::Adt(_) => format!("chelis_value_as_adt({current})"),
                }
            };
            let access = match step {
                RootPathStep::Tuple(index) => {
                    if depth == 0 && !matches!(ty, HostType::Tuple(_)) {
                        return Err(invalid_abi_shape(
                            format!("manifest tuple path starts at `{ty:?}`"),
                            "manifested root observation",
                        ));
                    }
                    format!("chelis_tuple_get({container}, {index})")
                }
                RootPathStep::Adt(index) => {
                    if depth == 0 && !matches!(ty, HostType::Adt(_, _)) {
                        return Err(invalid_abi_shape(
                            format!("manifest ADT path starts at `{ty:?}`"),
                            "manifested root observation",
                        ));
                    }
                    format!("chelis_adt_get_field({container}, {index})")
                }
            };
            self.lines
                .push(format!("{}chelis_value {boxed} = {access};", self.indent));
            current = boxed.clone();
            boxed_values.push(boxed);
        }

        self.emit_labeled_boxed_root(name, &current);
        for boxed in boxed_values.into_iter().rev() {
            self.lines
                .push(format!("{}chelis_value_release({boxed});", self.indent));
        }
        Ok(())
    }

    fn emit_labeled_boxed_root(&mut self, name: &str, value: &str) {
        let safe_name = chelis_ir::span_sanitize::sanitize_for_format_string(name);
        self.lines.push(format!(
            "{}printf(\"%s = \", \"{safe_name}\");",
            self.indent
        ));
        self.lines
            .push(format!("{}switch ({value}.tag) {{", self.indent));
        self.lines.push(format!(
            "{}case CHELIS_VALUE_SCALAR: {{ chelis_string text = \
             chelis_string_from_scalar(chelis_value_as_scalar({value})); \
             fputs(chelis_string_data(text), stdout); chelis_string_release(text); break; }}",
            self.indent
        ));
        self.lines.push(format!(
            "{}case CHELIS_VALUE_UNIT: printf(\"()\"); break;",
            self.indent
        ));
        self.lines.push(format!(
            "{}case CHELIS_VALUE_STRING: printf(\"%s\", chelis_string_data(chelis_value_as_string({value}))); break;",
            self.indent
        ));
        for (tag, printer, accessor) in [
            (
                "TENSOR",
                "chelis_print_tensor_stdout",
                "chelis_value_as_tensor",
            ),
            ("LIST", "chelis_print_list", "chelis_value_as_list"),
            ("TUPLE", "chelis_print_tuple", "chelis_value_as_tuple"),
            ("DICT", "chelis_print_dict", "chelis_value_as_dict"),
            ("ADT", "chelis_print_adt", "chelis_value_as_adt"),
        ] {
            self.lines.push(format!(
                "{}case CHELIS_VALUE_{tag}: {printer}({accessor}({value})); break;",
                self.indent
            ));
        }
        self.lines.push(format!(
            "{}default: fprintf(stderr, \"invalid chelis_value tag in manifested root\\n\"); abort();",
            self.indent
        ));
        self.lines.push(format!("{}}}", self.indent));
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
    ) -> Result<(), Unsupported> {
        let HostType::Option(inner) = ty else {
            return Err(invalid_abi_shape(
                format!("Some constructor carries non-option ABI type `{ty:?}`"),
                "Some constructor",
            ));
        };
        require_same_abi_type(inner.as_ref(), value_ty, "Some constructor payload")?;
        if is_scalar_abi(inner.as_ref()) {
            self.lines.push(format!(
                "{}{target} = (chelis_option_scalar){{ .is_some = 1, .reserved = {{0}}, .value = {} }};",
                self.indent,
                scalar_carrier_expr(value_var, value_ty)?
            ));
        } else {
            self.lines.push(format!(
                "{}{target} = (chelis_option_value){{ .is_some = 1, .reserved = {{0}}, .value = {} }};",
                self.indent,
                self.box_value_expr(value_var, value_ty)?
            ));
        }
        Ok(())
    }

    fn assign_option_none(&mut self, target: &str, ty: &HostType) -> Result<(), Unsupported> {
        let HostType::Option(inner) = ty else {
            return Err(invalid_abi_shape(
                format!("None constructor carries non-option ABI type `{ty:?}`"),
                "None constructor",
            ));
        };
        if is_scalar_abi(inner.as_ref()) {
            self.lines.push(format!(
                "{}{target} = (chelis_option_scalar){{ .is_some = 0, .reserved = {{0}}, .value = chelis_scalar_from_bits(CHELIS_DTYPE_F32, UINT64_C(0)) }};",
                self.indent
            ));
        } else {
            self.lines.push(format!(
                "{}{target} = (chelis_option_value){{ .is_some = 0, .reserved = {{0}}, .value = (chelis_value){{ .tag = CHELIS_VALUE_UNIT, .reserved = {{0}}, .payload.handle = NULL }} }};",
                self.indent
            ));
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

fn is_float_abi(ty: &HostAbiType) -> bool {
    matches!(
        ty,
        HostAbiType::Float16 | HostAbiType::BFloat16 | HostAbiType::Float32 | HostAbiType::Float64
    )
}

fn is_scalar_abi(ty: &HostAbiType) -> bool {
    matches!(
        ty,
        HostAbiType::Int8
            | HostAbiType::Int16
            | HostAbiType::Int32
            | HostAbiType::Int64
            | HostAbiType::Float16
            | HostAbiType::BFloat16
            | HostAbiType::Float32
            | HostAbiType::Float64
            | HostAbiType::Bool
    )
}

fn integer_abi_width(ty: &HostAbiType) -> Result<i64, Unsupported> {
    match ty {
        HostAbiType::Int8 => Ok(8),
        HostAbiType::Int16 => Ok(16),
        HostAbiType::Int32 => Ok(32),
        HostAbiType::Int64 => Ok(64),
        other => Err(invalid_abi_shape(
            format!("integer shift operand resolved to {other:?}"),
            "C host integer shift emission",
        )),
    }
}

fn integer_abi_prim(ty: &HostAbiType) -> Result<Prim, Unsupported> {
    match ty {
        HostAbiType::Int8 => Ok(Prim::Int8),
        HostAbiType::Int16 => Ok(Prim::Int16),
        HostAbiType::Int32 => Ok(Prim::Int32),
        HostAbiType::Int64 => Ok(Prim::Int64),
        other => Err(invalid_abi_shape(
            format!("integer kernel operand resolved to {other:?}"),
            "C host integer kernel emission",
        )),
    }
}

fn integer_trap_message(
    ty: &HostAbiType,
    op: &'static str,
    overflow: bool,
) -> Result<String, Unsupported> {
    let prim = integer_abi_prim(ty)?;
    Ok(if overflow {
        NumericTrap::Overflow { op, prim }
    } else {
        NumericTrap::DivZero { op, prim }
    }
    .to_string())
}

fn integer_checked_binary_expr(
    function: &'static str,
    op: &'static str,
    lhs: EmittedExpr,
    rhs: EmittedExpr,
    ty: &HostAbiType,
) -> Result<EmittedExpr, Unsupported> {
    Ok(EmittedExpr::call(
        function,
        [
            lhs,
            rhs,
            EmittedExpr::integer(integer_abi_width(ty)?),
            EmittedExpr::string_literal(integer_trap_message(ty, op, true)?),
        ],
    ))
}

fn checked_integer_divisor_expr(
    op: &'static str,
    dividend: EmittedExpr,
    divisor: EmittedExpr,
    ty: &HostAbiType,
) -> Result<EmittedExpr, Unsupported> {
    Ok(EmittedExpr::call(
        "chelis_int_checked_divisor",
        [
            dividend,
            divisor,
            EmittedExpr::integer(integer_abi_width(ty)?),
            EmittedExpr::string_literal(integer_trap_message(ty, op, false)?),
            EmittedExpr::string_literal(integer_trap_message(ty, op, true)?),
        ],
    ))
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
    // Typedefs and macros live in the emitted translation unit's include
    // set (stdbool/stdint/stdio/stdlib/string plus assert/math, with
    // stddef arriving transitively; `ssize_t` is POSIX) - chelis#840: a
    // user def, binding, or parameter spelled like one of these shadows
    // or redefines the typedef and the C cannot compile.
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
    "sig_atomic_t",
    "int_least8_t",
    "int_least16_t",
    "int_least32_t",
    "int_least64_t",
    "uint_least8_t",
    "uint_least16_t",
    "uint_least32_t",
    "uint_least64_t",
    "int_fast8_t",
    "int_fast16_t",
    "int_fast32_t",
    "int_fast64_t",
    "uint_fast8_t",
    "uint_fast16_t",
    "uint_fast32_t",
    "uint_fast64_t",
];

/// Prefix applied to a user identifier that would otherwise be illegal or
/// colliding in emitted C. The double underscore keeps it out of the
/// runtime's `chelis_*` symbol space, but the mapping is not
/// collision-free: a user name that literally spells `chelis_user__<kw>`
/// lands on the same emitted symbol as a mangled `<kw>`. Def-level
/// duplicates are detected and rejected loudly before emission
/// (chelis#840); parameter/binding-level duplicates remain a documented
/// #379 limit.
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
        chelis_types::deliberate_rejection!(
            "[04-TOT-2]",
            "the resolved host IR and C ABI projection disagree; this is an internal \
             compiler error, never a request to select a fallback representation \
             (chelis#730)"
        ),
    )
}

fn unsupported_value_boxing(ty: &HostType, context: &'static str) -> Unsupported {
    let authority = match ty {
        HostType::Callback(_, _) => chelis_types::unimplemented_rejection!(
            879,
            "the C host lane has no general first-class function-value box"
        ),
        HostType::Option(_) | HostType::MappedFile | HostType::Unit => {
            chelis_types::deliberate_rejection!(
                "[04-TOT-2]",
                "this checked host value cannot reach the generic boxing path; no fallback \
                 representation is permitted"
            )
        }
        _ => chelis_types::deliberate_rejection!(
            "[04-TOT-2]",
            "the resolved host type and boxing dispatcher disagree; no fallback \
             representation is permitted"
        ),
    };
    Unsupported::new(
        UnsupportedKind::HostType(format!("{ty:?}")),
        context,
        Stage::Codegen("c"),
        authority,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CheckedCastSurface {
    Scalar,
    Tensor,
}

fn checked_cast_abi_scalar_prim(ty: &HostType) -> Result<Prim, Unsupported> {
    let (prim, surface) = checked_cast_abi_axis(ty)?;
    if surface != CheckedCastSurface::Scalar {
        return Err(checked_cast_plan_error(format!(
            "checked scalar cast resolved to non-scalar ABI type {ty:?}"
        )));
    }
    Ok(prim)
}

/// Resolve an already-projected host ABI edge onto the closed checked-cast
/// axes. This match is exhaustive: container/function additions cannot
/// silently inherit numeric identity.
fn checked_cast_abi_axis(ty: &HostType) -> Result<(Prim, CheckedCastSurface), Unsupported> {
    match ty {
        HostType::Int8 => Ok((Prim::Int8, CheckedCastSurface::Scalar)),
        HostType::Int16 => Ok((Prim::Int16, CheckedCastSurface::Scalar)),
        HostType::Int32 => Ok((Prim::Int32, CheckedCastSurface::Scalar)),
        HostType::Int64 => Ok((Prim::Int64, CheckedCastSurface::Scalar)),
        HostType::Float16 => Ok((Prim::F16, CheckedCastSurface::Scalar)),
        HostType::BFloat16 => Ok((Prim::Bf16, CheckedCastSurface::Scalar)),
        HostType::Float32 => Ok((Prim::F32, CheckedCastSurface::Scalar)),
        HostType::Float64 => Ok((Prim::F64, CheckedCastSurface::Scalar)),
        HostType::Bool => Ok((Prim::Bool, CheckedCastSurface::Scalar)),
        HostType::Tensor(tensor) => Ok((tensor.precision, CheckedCastSurface::Tensor)),
        HostType::String
        | HostType::Callback(_, _)
        | HostType::Adt(_, _)
        | HostType::List(_)
        | HostType::Dict(_, _)
        | HostType::Tuple(_)
        | HostType::Option(_)
        | HostType::MappedFile
        | HostType::Unit => Err(checked_cast_plan_error(format!(
            "checked numeric cast resolved to non-numeric ABI type {ty:?}"
        ))),
    }
}

fn checked_cast_plan_error(detail: String) -> Unsupported {
    Unsupported::new(
        UnsupportedKind::HostAbi(detail),
        "C host checked-cast emission",
        Stage::Codegen("c"),
        chelis_types::deliberate_rejection!(
            "[04-NUM-14]",
            "checked cast requires an active numeric or bool source/target pair on one surface; \
             no identity fallback is permitted"
        ),
    )
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
        Prim::F32 => "float",
        Prim::Bool => "uint8_t",
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

/// Project a resolved host scalar to the exact tagged C carrier expected by
/// scalar-taking runtime calls. Generic `chelis_value` boxing is a distinct
/// container boundary and must not be used as a compatibility conversion.
fn scalar_carrier_expr(value: &str, ty: &HostType) -> Result<String, Unsupported> {
    let expr = match ty {
        HostType::Int8 => {
            format!("chelis_scalar_from_bits(CHELIS_DTYPE_I8, (uint64_t)(uint8_t)(int8_t){value})")
        }
        HostType::Int16 => format!(
            "chelis_scalar_from_bits(CHELIS_DTYPE_I16, (uint64_t)(uint16_t)(int16_t){value})"
        ),
        HostType::Int32 => format!(
            "chelis_scalar_from_bits(CHELIS_DTYPE_I32, (uint64_t)(uint32_t)(int32_t){value})"
        ),
        HostType::Int64 => {
            format!("chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)(int64_t){value})")
        }
        HostType::Float64 => {
            format!("chelis_scalar_from_bits(CHELIS_DTYPE_F64, chelis_host_f64_bits({value}))")
        }
        HostType::Float32 => format!(
            "chelis_scalar_from_bits(CHELIS_DTYPE_F32, (uint64_t)chelis_host_f32_bits({value}))"
        ),
        HostType::Float16 => {
            format!("chelis_scalar_from_bits(CHELIS_DTYPE_F16, (uint64_t)(uint16_t){value})")
        }
        HostType::BFloat16 => {
            format!("chelis_scalar_from_bits(CHELIS_DTYPE_BF16, (uint64_t)(uint16_t){value})")
        }
        HostType::Bool => {
            format!("chelis_scalar_from_bits(CHELIS_DTYPE_BOOL, (uint64_t)({value} ? 1 : 0))")
        }
        other => {
            return Err(invalid_abi_shape(
                format!("scalar runtime argument has non-scalar ABI type `{other:?}`"),
                "exact scalar runtime argument",
            ));
        }
    };
    Ok(expr)
}

/// Project the exact tagged runtime scalar back to the resolved host scalar
/// type. The expected dtype is always checked before reading the payload.
fn scalar_carrier_value_expr(value: &str, ty: &HostType) -> Result<String, Unsupported> {
    let expr = match ty {
        HostType::Int8 => {
            format!("(int8_t)chelis_host_scalar_as_i64({value}, CHELIS_DTYPE_I8)")
        }
        HostType::Int16 => {
            format!("(int16_t)chelis_host_scalar_as_i64({value}, CHELIS_DTYPE_I16)")
        }
        HostType::Int32 => {
            format!("(int32_t)chelis_host_scalar_as_i64({value}, CHELIS_DTYPE_I32)")
        }
        HostType::Int64 => {
            format!("chelis_host_scalar_as_i64({value}, CHELIS_DTYPE_I64)")
        }
        HostType::Float16 => {
            format!("(uint16_t)chelis_host_scalar_bits({value}, CHELIS_DTYPE_F16)")
        }
        HostType::BFloat16 => {
            format!("(uint16_t)chelis_host_scalar_bits({value}, CHELIS_DTYPE_BF16)")
        }
        HostType::Float32 => {
            format!("(float)chelis_host_scalar_as_float({value}, CHELIS_DTYPE_F32)")
        }
        HostType::Float64 => {
            format!("chelis_host_scalar_as_float({value}, CHELIS_DTYPE_F64)")
        }
        HostType::Bool => format!("chelis_host_scalar_as_bool({value})"),
        other => {
            return Err(invalid_abi_shape(
                format!("scalar runtime result has non-scalar ABI type `{other:?}`"),
                "exact scalar runtime result",
            ));
        }
    };
    Ok(expr)
}

fn scalar_dtype_macro(ty: &HostType) -> Result<&'static str, Unsupported> {
    Ok(match ty {
        HostType::Int8 => "CHELIS_DTYPE_I8",
        HostType::Int16 => "CHELIS_DTYPE_I16",
        HostType::Int32 => "CHELIS_DTYPE_I32",
        HostType::Int64 => "CHELIS_DTYPE_I64",
        HostType::Float16 => "CHELIS_DTYPE_F16",
        HostType::BFloat16 => "CHELIS_DTYPE_BF16",
        HostType::Float32 => "CHELIS_DTYPE_F32",
        HostType::Float64 => "CHELIS_DTYPE_F64",
        HostType::Bool => "CHELIS_DTYPE_BOOL",
        other => {
            return Err(invalid_abi_shape(
                format!("dtype selection has non-scalar ABI type `{other:?}`"),
                "exact scalar dtype selection",
            ));
        }
    })
}

fn host_float_as_double(value: &str, ty: &HostType) -> String {
    match ty {
        HostType::Float16 => format!("(double)chelis_f16_to_f32({value})"),
        HostType::BFloat16 => format!("(double)chelis_bf16_to_f32({value})"),
        HostType::Float32 | HostType::Float64 => format!("(double)({value})"),
        other => unreachable!("float conversion of non-float host type {other:?}"),
    }
}

fn scalar_arithmetic_arg_expr(value: &str, ty: &HostType) -> EmittedExpr {
    let value = EmittedExpr::identifier(value.to_string());
    match ty {
        HostType::Float16 => EmittedExpr::call("chelis_f16_to_f32", [value]),
        HostType::BFloat16 => EmittedExpr::call("chelis_bf16_to_f32", [value]),
        _ => value,
    }
}

fn finalize_scalar_expr(value: EmittedExpr, ty: &HostType) -> EmittedExpr {
    match ty {
        HostType::Float16 => EmittedExpr::call("chelis_f32_to_f16", [value]),
        HostType::BFloat16 => EmittedExpr::call("chelis_f32_to_bf16", [value]),
        _ => value,
    }
}

fn float_math_function(
    ty: &HostType,
    binary64: &'static str,
    binary32: &'static str,
) -> &'static str {
    match ty {
        HostType::Float16 | HostType::BFloat16 | HostType::Float32 => binary32,
        _ => binary64,
    }
}

fn activation_math_function(
    ty: &HostType,
    f16: &'static str,
    bf16: &'static str,
    f32: &'static str,
    f64: &'static str,
) -> &'static str {
    match ty {
        HostType::Float16 => f16,
        HostType::BFloat16 => bf16,
        HostType::Float32 => f32,
        HostType::Float64 => f64,
        other => unreachable!("activation helper selected for non-float host type {other:?}"),
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BinaryElementwiseFunc {
    Max,
    Min,
}

impl BinaryElementwiseFunc {
    fn f32_name(self) -> &'static str {
        match self {
            Self::Max => "fmaxf",
            Self::Min => "fminf",
        }
    }

    fn i32_comparison(self) -> &'static str {
        match self {
            Self::Max => ">=",
            Self::Min => "<=",
        }
    }
}

/// One arm of the runtime-dtype dispatch emitted by the elementwise
/// host-emit helpers (`assign_tensor_*_elementwise`). Each arm names a
/// `CHELIS_*` constant and the C element type for tensor buffer access.
///
/// Equal byte widths do not permit a shared element type. Bool uses its
/// canonical one-byte payload and never shares the f32 representation.
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
            DtypeArm::F32 => "float",
            DtypeArm::F64 => "double",
            DtypeArm::I32 => "int32_t",
            DtypeArm::I64 => "int64_t",
            DtypeArm::Bool => "uint8_t",
        }
    }

    /// Every supported precision the operator-form elementwise
    /// helpers emit a typed arm for. Maps each runtime dtype to a C element
    /// type that is compatible with its physical representation.
    fn all_operator_arms() -> &'static [DtypeArm] {
        &[
            DtypeArm::F32,
            DtypeArm::F64,
            DtypeArm::I32,
            DtypeArm::I64,
            DtypeArm::Bool,
        ]
    }

    /// Representations that can use f32-only helper functions directly.
    /// I32 is excluded because conversion to binary32 loses integer precision.
    fn f32_payload_func_arms() -> &'static [DtypeArm] {
        &[DtypeArm::F32]
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
    fn exact_scalar_argument_projection_never_boxes_through_chelis_value() {
        for (ty, dtype) in [
            (HostType::Int8, "CHELIS_DTYPE_I8"),
            (HostType::Int16, "CHELIS_DTYPE_I16"),
            (HostType::Int32, "CHELIS_DTYPE_I32"),
            (HostType::Int64, "CHELIS_DTYPE_I64"),
            (HostType::Float16, "CHELIS_DTYPE_F16"),
            (HostType::BFloat16, "CHELIS_DTYPE_BF16"),
            (HostType::Float32, "CHELIS_DTYPE_F32"),
            (HostType::Float64, "CHELIS_DTYPE_F64"),
            (HostType::Bool, "CHELIS_DTYPE_BOOL"),
        ] {
            let emitted = scalar_carrier_expr("value", &ty).unwrap();
            assert!(
                emitted.contains("chelis_scalar_from_bits") && emitted.contains(dtype),
                "{ty:?} did not project to its exact tagged scalar: {emitted}"
            );
            assert!(
                !emitted.contains("chelis_value_from_scalar"),
                "exact scalar argument was unnecessarily boxed: {emitted}"
            );
        }
        assert!(
            scalar_carrier_expr("value", &HostType::String).is_err(),
            "a non-scalar host value must not acquire a scalar ABI fallback"
        );
    }

    #[test]
    fn exact_scalar_result_projection_checks_every_active_dtype() {
        for (ty, dtype) in [
            (HostType::Int8, "CHELIS_DTYPE_I8"),
            (HostType::Int16, "CHELIS_DTYPE_I16"),
            (HostType::Int32, "CHELIS_DTYPE_I32"),
            (HostType::Int64, "CHELIS_DTYPE_I64"),
            (HostType::Float16, "CHELIS_DTYPE_F16"),
            (HostType::BFloat16, "CHELIS_DTYPE_BF16"),
            (HostType::Float32, "CHELIS_DTYPE_F32"),
            (HostType::Float64, "CHELIS_DTYPE_F64"),
            (HostType::Bool, "CHELIS_DTYPE_BOOL"),
        ] {
            let emitted = scalar_carrier_value_expr("value", &ty).unwrap();
            assert!(
                emitted.contains(dtype) || matches!(ty, HostType::Bool),
                "{ty:?} did not validate its exact tagged scalar dtype: {emitted}"
            );
            assert_eq!(scalar_dtype_macro(&ty).unwrap(), dtype);
            assert!(
                !emitted.contains("chelis_value_as_scalar"),
                "exact scalar result crossed the generic value carrier: {emitted}"
            );
        }
        assert!(scalar_carrier_value_expr("value", &HostType::String).is_err());
        assert!(scalar_dtype_macro(&HostType::String).is_err());
    }

    #[test]
    fn manifested_boxed_roots_use_only_the_exact_value_carrier() {
        let mut emitter = HostEmitter::new(
            "    ".to_string(),
            "manifest",
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            &[],
        );
        emitter.emit_labeled_boxed_root("root", "boxed");
        let emitted = emitter.lines.join("\n");

        for required in [
            "case CHELIS_VALUE_SCALAR:",
            "chelis_value_as_scalar(boxed)",
            "chelis_value_as_string(boxed)",
            "chelis_value_as_tensor(boxed)",
            "default:",
            "abort();",
        ] {
            assert!(
                emitted.contains(required),
                "boxed-root observation is missing `{required}`:\n{emitted}"
            );
        }
        for retired in [
            "CHELIS_VALUE_INT64",
            "CHELIS_VALUE_FLOAT64",
            "CHELIS_VALUE_BOOL",
            ".as.",
        ] {
            assert!(
                !emitted.contains(retired),
                "boxed-root observation restored retired value ABI `{retired}`:\n{emitted}"
            );
        }
    }

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

    #[test]
    fn scalar_activation_names_have_closed_expression_identities_and_all_width_helpers() {
        for (name, expected) in [
            ("relu", CExpressionBuiltin::Relu),
            ("sigmoid", CExpressionBuiltin::Sigmoid),
            ("tanh", CExpressionBuiltin::Tanh),
            ("silu", CExpressionBuiltin::Silu),
            ("gelu", CExpressionBuiltin::Gelu),
        ] {
            assert_eq!(CExpressionBuiltin::decode(name), Ok(expected));
        }

        let mut helpers = Vec::new();
        append_tensor_math_helpers(&mut helpers);
        let emitted = helpers.join("\n");
        for op in ["relu", "sigmoid", "tanh", "silu", "gelu"] {
            for width in ["f16", "bf16", "f32", "f64"] {
                assert!(
                    emitted.contains(&format!("chelis_host_{op}_{width}")),
                    "missing {width} helper for {op}:\n{emitted}"
                );
            }
        }
        assert!(emitted.contains("chelis_host_finalize_f16"));
        assert!(emitted.contains("chelis_host_finalize_bf16"));
    }

    /// chelis#1112: the emitted reshape helper stores the exact tagged
    /// int64 extent into a dynamically sized int64 shape buffer.
    ///
    /// This replaces `reshape_helper_traps_extent_above_int32_before_the_store`,
    /// which pinned the ordering of a trap against the `(int)` store it
    /// guarded. Both are gone: the trap existed only because the store was
    /// lossy, and rejecting a representable extent would now itself be the
    /// defect. Pinning the ABSENCE of the cast is what stops a later edit
    /// from quietly reintroducing the narrowing, so several assertions
    /// below are negative on purpose.
    #[test]
    fn reshape_helper_stores_the_extent_at_int64_with_no_truncating_cast() {
        let mut out = Vec::new();
        append_tensor_reshape_helper(&mut out);
        let text = out.join("\n");
        assert!(
            text.contains("int64_t *shape = (int64_t*)calloc("),
            "the shape buffer must be dynamically sized for the requested rank:\n{text}"
        );
        let read = text
            .find("int64_t dim = chelis_host_scalar_as_i64(chelis_value_as_scalar(")
            .expect("the extent is read from an exact tagged int64 scalar");
        let store = text
            .find("shape[i] = dim;")
            .expect("the extent is stored without a cast");
        assert!(
            read < store,
            "the extent must be read before it is stored; read at {read}, store at {store}"
        );
        assert!(
            !text.contains("shape[i] = (int)dim;"),
            "a truncating store into the shape buffer is the defect chelis#1112 removed:\n{text}"
        );
        assert!(
            !text.contains("2147483647LL"),
            "the int32 extent trap is dead with the cast it guarded:\n{text}"
        );
        assert!(
            !text.contains("CHELIS_MAX_DIM"),
            "reshape rank must not be capped by a fixed compatibility constant:\n{text}"
        );
        // The negative-extent guard is NOT dead: a negative dim is invalid
        // at every carrier width, so the widening must not have taken it
        // along with the truncation trap.
        assert!(
            text.contains("if (dim < 0) {"),
            "the negative-extent rejection survives the widening:\n{text}"
        );
    }
}
