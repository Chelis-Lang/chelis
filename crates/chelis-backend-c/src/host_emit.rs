use chelis_ir::host::{
    HostBlasMatmulSummary, HostFunctionSpecialization, HostSparseOpSummary, HostTensorHelper,
    HostTensorSpecialization,
};
mod entry;

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
    CharCode,
    CharFromCode,
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
            "char_code" => Self::CharCode,
            "char_from_code" => Self::CharFromCode,
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
                )
                .with_supported_alternative("run this program with `chelis eval`"));
            }
        })
    }
}

use crate::emit::CEmitter;
use crate::emitted_expr::{BinaryOperator, EmittedExpr, UnaryOperator};

/// Emit a C string literal whose bytes are unambiguous in every following
/// lexical context. Fixed-width three-digit octal escapes preserve embedded
/// NUL and cannot absorb an adjacent decimal or hexadecimal digit.
fn c_utf8_byte_literal(value: &str) -> String {
    let mut literal = String::from("\"");
    for byte in value.as_bytes() {
        literal.push_str(&format!("\\{byte:03o}"));
    }
    literal.push('"');
    literal
}

fn runtime_string_literal(value: &str) -> String {
    format!(
        "chelis_string_from_utf8((const uint8_t *){}, INT64_C({}))",
        c_utf8_byte_literal(value),
        value.len()
    )
}
use crate::host_abi::{
    HostAbiBinding as HostBinding, HostAbiCallback as HostCallback,
    HostAbiCallbackKind as HostCallbackKind, HostAbiExpr as HostExpr,
    HostAbiExprKind as HostExprKind, HostAbiFunction as HostFunction,
    HostAbiMatchArm as HostMatchArm, HostAbiParam as HostParam, HostAbiProgram as HostProgram,
    HostAbiType, HostAbiType as HostType, ProjectedHostProgram, ProjectedHostSite,
};
use chelis_ir::dag::{DimInfo, RiscOp, TensorType};
use chelis_ir::ownership::{
    HostSiteId, VerifiedApplyKind, VerifiedBlockId, VerifiedDagView, VerifiedEdgeView,
    VerifiedHostAction, VerifiedHostOperation, VerifiedHostTensorHelperView,
    VerifiedHostTerminator, VerifiedOperationId, VerifiedOwnerId, VerifiedOwnershipUse,
    VerifiedTerminalView,
};
use chelis_types::manifest::{RootManifest, RootPathStep};
use chelis_types::types::{Lane, Prim};
use chelis_types::unsupported::{Stage, Unsupported, UnsupportedKind};
use chelis_types::{CheckedCastKind, CheckedCastPlan, NumericTrap};
use chelis_unord::{UnordMap, UnordSet};

pub(crate) fn emit_host_abi_program(
    projected: &ProjectedHostProgram<'_>,
    program_name: &str,
    external_helpers: &UnordSet<String>,
) -> Result<String, Unsupported> {
    let program = projected.program();
    let _site_identity_count = projected.sites().len();
    // Emit helpers and functions into a body buffer first so we can detect which
    // runtime headers they transitively require (e.g. `chelis_math.h` on macOS
    // when a helper uses the vForce vvexpf/vvlogf path).  The preamble is then
    // assembled with the right includes and prepended.  Without this, the
    // include-stripping in `append_helper` silently drops the inner emitter's
    // `#include "chelis_math.h"` and the resulting `main.c` calls vvexpf with
    // no declaration in scope.
    let mut body: Vec<String> = Vec::new();
    let mut helper_requirements = HelperRequirements::default();
    #[cfg(feature = "native-random-observer")]
    let source_sites = crate::random_observer::source_sites(projected.source_emission());
    append_scalar_conversion_helpers(&mut body);
    body.push(String::new());
    append_tensor_abi_helpers(&mut body);
    body.push(String::new());
    append_host_result_claim_support(&mut body);
    append_tensor_reshape_helper(&mut body);
    body.push(String::new());
    append_tensor_print_helper(&mut body);
    body.push(String::new());
    append_uniform_sample_helper(&mut body);
    body.push(String::new());
    #[cfg(feature = "native-random-observer")]
    {
        crate::random_observer::append_support(&mut body);
        crate::random_observer::append_source_sites(&mut body, &source_sites);
        body.push(String::new());
    }
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
    let internal_names = internal_function_names(program, &emitted_names);
    let mut all_names = emitted_names.clone();
    for (original, internal) in internal_names.to_sorted() {
        if emitted_names
            .get(original)
            .is_some_and(|public| public != internal)
        {
            all_names.insert(format!("{original} (owned body)"), internal.clone());
        }
    }
    reject_duplicate_emitted_function_names(&all_names)?;
    let function_specializations = function_specializations(program);
    let header = emit_host_header_with_linkage(program, program_name, internal_linkage)?;
    if !header.is_empty() {
        body.push(header);
        body.push(String::new());
    }
    for function in &program.functions {
        if function.origin != chelis_ir::host::HostFunctionOrigin::Authored {
            continue;
        }
        let params = function
            .params
            .iter()
            .map(|param| c_decl(&param.ty, &param.name))
            .collect::<Result<Vec<_>, _>>()?
            .join(", ");
        body.push(format!(
            "static inline {} {}({});",
            c_type(&function.ret_ty)?,
            internal_names
                .get(&function.name)
                .expect("authored function has owned-body name"),
            private_host_function_params(&params)
        ));
    }
    if program
        .functions
        .iter()
        .any(|function| function.origin == chelis_ir::host::HostFunctionOrigin::Authored)
    {
        body.push(String::new());
    }

    // Issue #352: top-level bindings referenced inside a compiled host
    // function would otherwise dangle -- `main()` declares every binding as
    // a local, so a def body's `w` had no declaration in scope and the
    // native compiler rejected the TU. Hoist captured bindings to file
    // scope; `emit_main` assigns them in binding order instead of declaring
    // locals, so function bodies and `main()` resolve the same object
    // (mirroring eval's load-closure, which serves the binding's value at
    // call time). [04-INF-4] rejects a DIRECT forward reference from a
    // compiled function to a later binding at check time, annotated or not,
    // so a hoisted binding a function names is initialized before the first
    // user call that reads it. [04-INF-8] closes the INDIRECT shape too: when
    // an earlier binding's initializer calls a function that reaches a later
    // value, the checker rejects the initiating root before this emitter runs.
    // `main()` remains source ordered; the checker, not a reordered backend,
    // owns the top-level initialization frontier (chelis#1339).
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
            if matches!(binding.ty, HostType::Tensor(_)) {
                body.push(format!(
                    "static __chelis_host_result_origin {};",
                    result_origin_name(name)
                ));
            }
        }
        body.push(String::new());
    }

    let global_entry_coverage = entry::global_helper_coverage(program);
    for (index, helper) in program.global_tensor_helpers.iter().enumerate() {
        let helper_name = global_tensor_helper_name(program_name, index);
        if external_helpers.contains(&helper_name) {
            append_external_helper_declaration(&mut body, &helper_name);
        } else {
            for (variant, coverage) in global_entry_coverage.variants[index].iter().enumerate() {
                helper_requirements.merge(append_helper(
                    &mut body,
                    helper,
                    projected
                        .global_tensor_helper(index)
                        .expect("projected global helper retains verified child"),
                    &entry::variant_name(&helper_name, variant),
                    coverage,
                )?);
            }
        }
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

    let mut stubbed_functions: UnordSet<String> = UnordSet::new();
    let mut function_bodies: Vec<String> = Vec::new();
    for (function_index, function) in program.functions.iter().enumerate() {
        let emitted_name = emitted_names
            .get(&function.name)
            .expect("host function emitted name");
        let mut fn_buf: Vec<String> = Vec::new();
        let verified_helpers = (0..function.tensor_helpers.len())
            .map(|helper| {
                projected
                    .function_tensor_helper(function_index, helper)
                    .expect("projected function helper retains verified child")
            })
            .collect::<Vec<_>>();
        let helper_output_counts = verified_helpers
            .iter()
            .map(|verified| CEmitter::output_labels(verified.dag()).len().max(1))
            .collect::<Vec<_>>();
        match emit_function(
            &mut fn_buf,
            function,
            emitted_name,
            &internal_names,
            &function_specializations,
            internal_linkage || function.is_monomorphized_specialization(),
            projected
                .function_sites(function_index)
                .expect("projected function retains verified sites"),
            projected
                .function_owner_bindings(function_index)
                .expect("projected function retains verified body-owner bindings"),
            &helper_output_counts,
            &verified_helpers,
            external_helpers,
            #[cfg(feature = "native-random-observer")]
            &source_sites
                .iter()
                .filter(|s| s.verified.unit_word() == function_index + 1)
                .cloned()
                .collect::<Vec<_>>(),
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

    for (function_index, function) in program.functions.iter().enumerate() {
        if stubbed_functions.contains(&function.name) {
            // A stubbed wrapper aborts before any helper call; skip its
            // (possibly unemittable) tensor helpers entirely.
            continue;
        }
        let verified_helpers = (0..function.tensor_helpers.len())
            .map(|helper| {
                projected
                    .function_tensor_helper(function_index, helper)
                    .expect("projected function helper retains verified child")
            })
            .collect::<Vec<_>>();
        let entry_coverage = entry::helper_coverage_with_verified(function, &verified_helpers);
        for (index, helper) in function.tensor_helpers.iter().enumerate() {
            let function_name = emitted_names
                .get(&function.name)
                .expect("host function emitted name");
            let helper_name = function_tensor_helper_name(function_name, index);
            if external_helpers.contains(&helper_name) {
                append_external_helper_declaration(&mut body, &helper_name);
            } else {
                for (variant, coverage) in entry_coverage.variants[index].iter().enumerate() {
                    helper_requirements.merge(append_helper(
                        &mut body,
                        helper,
                        projected
                            .function_tensor_helper(function_index, index)
                            .expect("projected function helper retains verified child"),
                        &entry::variant_name(&helper_name, variant),
                        coverage,
                    )?);
                }
            }
        }
    }

    body.extend(function_bodies);

    if helper_requirements.needs_fixed_dropout_helpers {
        let mut fixed_helpers = Vec::new();
        append_fixed_dropout_helpers(&mut fixed_helpers);
        fixed_helpers.push(String::new());
        fixed_helpers.extend(body);
        body = fixed_helpers;
    }

    if !program.globals.is_empty() {
        let hoisted: UnordSet<&str> = captured_globals.iter().map(String::as_str).collect();
        let helper_output_counts = (0..program.global_tensor_helpers.len())
            .map(|helper| {
                let verified = projected
                    .global_tensor_helper(helper)
                    .expect("projected global helper retains verified child");
                CEmitter::output_labels(verified.dag()).len().max(1)
            })
            .collect::<Vec<_>>();
        let helper_result_origins = (0..program.global_tensor_helpers.len())
            .map(|helper| {
                projected
                    .global_tensor_helper(helper)
                    .map(verified_helper_result_origin)
                    .transpose()
                    .map(Option::flatten)
            })
            .collect::<Result<Vec<_>, _>>()?;
        emit_main(
            &mut body,
            program_name,
            program,
            projected.manifest(),
            &hoisted,
            projected.root_sites(),
            &internal_names,
            &helper_output_counts,
            helper_result_origins,
            external_helpers,
        )?;
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
        // chelis#2205: the consuming counterparts of the container builtins
        // that may take a same-kind operand the ownership verifier moved at
        // its scheduled last use. Each mutates in place only at strong-owner
        // count one and otherwise clones and releases the consumed input, so
        // a retained alias is never mutated. Private for the same reason as
        // the three above.
        "chelis_list *chelis_list_append_owned(chelis_list *list, chelis_value value);".to_string(),
        "chelis_list *chelis_list_concat_owned(chelis_list *lhs, const chelis_list *rhs);"
            .to_string(),
        "chelis_dict *chelis_dict_insert_owned(chelis_dict *dict, chelis_value key, chelis_value value);"
            .to_string(),
        "chelis_dict *chelis_dict_merge_owned(chelis_dict *lhs, const chelis_dict *rhs);"
            .to_string(),
        "chelis_dict *chelis_dict_remove_owned(chelis_dict *dict, chelis_value key);".to_string(),
        "chelis_string chelis_string_concat_owned(chelis_string lhs, chelis_string rhs);"
            .to_string(),
    ]);
    if helper_requirements.needs_blas_header {
        out.push("#include \"chelis_blas.h\"".to_string());
        out.push(CEmitter::blas_integer_support());
    }
    if helper_requirements.needs_math_header {
        out.push("#include \"chelis_math.h\"".to_string());
    }
    out.push(String::new());
    out.extend(body);
    Ok(out.join("\n"))
}

fn global_tensor_helper_name(program_name: &str, index: usize) -> String {
    format!("{program_name}__global__tensor_{index}")
}

fn function_tensor_helper_name(emitted_function_name: &str, index: usize) -> String {
    format!("{emitted_function_name}__tensor_{index}")
}

/// The exact helper symbols the wrapper for `program_name` emits, in wrapper
/// order: every global helper, then each function's helpers in function
/// order. One naming routine serves the wrapper, the projected manifest, and
/// the concrete manifest so the three can never disagree about a symbol.
fn tensor_helper_symbols<'a>(
    program_name: &str,
    global_helper_count: usize,
    functions: impl IntoIterator<Item = (&'a str, usize)>,
) -> Result<Vec<String>, Unsupported> {
    let functions = functions.into_iter().collect::<Vec<_>>();
    let emitted_names = functions
        .iter()
        .map(|(name, _)| {
            (
                (*name).to_string(),
                emitted_function_name(program_name, name),
            )
        })
        .collect::<UnordMap<String, String>>();
    reject_duplicate_emitted_function_names(&emitted_names)?;
    let mut symbols = (0..global_helper_count)
        .map(|index| global_tensor_helper_name(program_name, index))
        .collect::<Vec<_>>();
    for (name, helper_count) in functions {
        let function_name = emitted_names.get(name).expect("host function emitted name");
        symbols.extend(
            (0..helper_count).map(|index| function_tensor_helper_name(function_name, index)),
        );
    }
    Ok(symbols)
}

/// Helper symbols of the projected ABI program, in wrapper order.
pub(crate) fn tensor_helper_names(
    program: &HostProgram,
    program_name: &str,
) -> Result<Vec<String>, Unsupported> {
    tensor_helper_symbols(
        program_name,
        program.global_tensor_helpers.len(),
        program
            .functions
            .iter()
            .map(|function| (function.name.as_str(), function.tensor_helpers.len())),
    )
}

/// Helper symbols paired with their concrete, pre-lowering source DAGs.
pub(crate) fn concrete_tensor_helper_codegen(
    program: &chelis_ir::host::ConcreteHostProgram,
    program_name: &str,
) -> Result<Vec<(String, chelis_ir::dag::Dag)>, Unsupported> {
    let symbols = tensor_helper_symbols(
        program_name,
        program.global_tensor_helpers.len(),
        program
            .functions
            .iter()
            .map(|function| (function.name.as_str(), function.tensor_helpers.len())),
    )?;
    let dags = program
        .global_tensor_helpers
        .iter()
        .chain(
            program
                .functions
                .iter()
                .flat_map(|function| function.tensor_helpers.iter()),
        )
        .map(|helper| helper.dag.clone());
    Ok(symbols.into_iter().zip(dags).collect())
}

fn emitted_function_name(program_name: &str, function_name: &str) -> String {
    if function_name == "main" {
        format!("{program_name}__main")
    } else if chelis_types::is_linker_format_name(function_name)
        && chelis_types::demangle_ident(function_name) == "main"
    {
        function_name.to_string()
    } else {
        // Authored Chelis definitions are exported through the generated
        // header, but in a compiler-reserved C namespace rather than under a
        // source spelling. Encode their UTF-8 bytes injectively. That
        // structurally prevents both C keyword/typedef collisions (#840) and
        // platform-library collisions such as `read` from unistd.h (#1957);
        // a source name that resembles this prefix encodes to another symbol.
        let mut emitted = "chelis_fn_".to_string();
        for byte in function_name.bytes() {
            emitted.push_str(&format!("{byte:02x}"));
        }
        emitted
    }
}

/// Defense in depth for compiler-owned names. Authored definitions use the
/// injective `chelis_fn_<utf8-hex>` namespace above, so two source spellings
/// cannot collide; this remains a hard gate if a future internal naming path
/// accidentally overlaps an emitted public or owned-body symbol.
fn reject_duplicate_emitted_function_names(
    emitted_names: &UnordMap<String, String>,
) -> Result<(), Unsupported> {
    let mut by_emitted: UnordMap<&str, Vec<&str>> = UnordMap::new();
    for (original, emitted) in emitted_names.to_sorted() {
        by_emitted.entry(emitted).or_default().push(original);
    }
    let mut collisions: Vec<String> = by_emitted
        .into_sorted()
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
            "generated C symbols must remain one-to-one; an internal naming path overlapped \
             another emitted definition (chelis#840)"
        ),
    ))
}

fn emitted_function_names(program: &HostProgram, program_name: &str) -> UnordMap<String, String> {
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

fn owned_body_name(emitted_name: &str) -> String {
    format!("{emitted_name}__chelis_owned_body")
}

fn internal_function_names(
    program: &HostProgram,
    emitted_names: &UnordMap<String, String>,
) -> UnordMap<String, String> {
    program
        .functions
        .iter()
        .map(|function| {
            let public = emitted_names
                .get(&function.name)
                .expect("every host function has an emitted name");
            let internal = if function.origin == chelis_ir::host::HostFunctionOrigin::Authored {
                owned_body_name(public)
            } else {
                public.clone()
            };
            (function.name.clone(), internal)
        })
        .collect()
}

fn function_specializations(program: &HostProgram) -> UnordMap<String, HostFunctionSpecialization> {
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
    out.push(
        "static inline uint64_t chelis_effective_uniform_seed(chelis_rng_state *state, uint64_t baked_seed) {".to_string(),
    );
    // Advance a frame value and commit it as a whole. The private pointer
    // transports invocation state; it is not an element-storage view.
    out.push("    chelis_rng_state current = *state;".to_string());
    out.push("    if (!current.active) {".to_string());
    out.push("        return baked_seed;".to_string());
    out.push("    }".to_string());
    out.push("    uint64_t counter = current.counter++;".to_string());
    out.push("    *state = current;".to_string());
    out.push("    return current.seed ^ (counter * 0x9E3779B97F4A7C15ULL);".to_string());
    out.push("}".to_string());
    out.push(
        "#define CHELIS_EFFECTIVE_UNIFORM_SEED(seed) chelis_effective_uniform_seed(__chelis_rng, seed)"
            .to_string(),
    );
}

fn append_fixed_dropout_helpers(out: &mut Vec<String>) {
    out.extend(
        crate::emit::FIXED_DROPOUT_HELPERS
            .iter()
            .map(|line| line.to_string()),
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
        "    chelis_value lhs_key = chelis_tuple_get(chelis_tuple_borrow_value(lhs_entry), 0);",
        "    chelis_value rhs_key = chelis_tuple_get(chelis_tuple_borrow_value(rhs_entry), 0);",
        "    int result = chelis_json_compare_strings(chelis_string_borrow_value(lhs_key), chelis_string_borrow_value(rhs_key));",
        "    chelis_value_release(lhs_key);",
        "    chelis_value_release(rhs_key);",
        "    return result;",
        "}",
        "",
        "static chelis_list *chelis_json_canonical_object_entries(const chelis_dict *dict) {",
        "    chelis_list *source = chelis_dict_entries(dict);",
        "    int64_t len = chelis_list_len(source);",
        "    chelis_tensor *order_storage = chelis_alloc(1, &len, CHELIS_DTYPE_I64);",
        "    chelis_tensor_write *order_guard = chelis_tensor_begin_write(order_storage);",
        "    chelis_write_view order_view = chelis_tensor_write_view(order_guard);",
        "    int64_t *order = (int64_t *)order_view.data;",
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
        "    chelis_tensor_end_write(order_guard);",
        "    chelis_tensor_release(order_storage);",
        "    chelis_list_release(source);",
        "    return result;",
        "}",
    ] {
        out.push(line.to_string());
    }
}

/// Instantiate the scalar host-expression path at each concrete float ABI.
///
/// Tensor activations lower through `chelis_ir::tier2`; scalar calls in a
/// Surf `def` reach this emitter after host-ABI projection instead. Reduced
/// floats need distinct per-node finalizers even though both compute as C
/// `float`, so one generated specialization cannot serve every source dtype.
fn append_activation_helpers(
    out: &mut Vec<String>,
    suffix: &str,
    c_type: &str,
    literal_suffix: &str,
    exp: &str,
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
        "    return x < {} ? {} : x;",
        literal("0.0"),
        literal("0.0")
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
        Some("chelis_host_finalize_f16"),
    );
    append_activation_helpers(
        out,
        "bf16",
        "float",
        "f",
        "expf",
        Some("chelis_host_finalize_bf16"),
    );
    append_activation_helpers(out, "f32", "float", "f", "expf", None);
    append_activation_helpers(out, "f64", "double", "", "exp", None);
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

/// Private generated-C adapters over the opaque published tensor ABI.  Read
/// access always enters through `chelis_tensor_read_view`; output writers use
/// an explicit guard at the call site and therefore never route mutation
/// through these helpers.
fn append_tensor_abi_helpers(out: &mut Vec<String>) {
    out.push(
        "static chelis_dtype chelis_host_tensor_dtype(const chelis_tensor *tensor) { return chelis_tensor_read_view(tensor).dtype; }"
            .to_string(),
    );
    out.push(
        "static const void *chelis_host_tensor_data(const chelis_tensor *tensor) { return chelis_tensor_read_view(tensor).data; }"
            .to_string(),
    );
    out.push(
        "static int64_t chelis_host_tensor_stride(const chelis_tensor *tensor, int32_t axis) {"
            .to_string(),
    );
    out.push("    return chelis_tensor_stride(tensor, axis);".to_string());
    out.push("}".to_string());
    out.push(
        "static chelis_tensor *chelis_host_alloc_like(const chelis_tensor *input, chelis_dtype dtype) {"
            .to_string(),
    );
    out.push(
        "    return chelis_tensor_alloc_like(input, chelis_scalar_from_bits(dtype, UINT64_C(0)));"
            .to_string(),
    );
    out.push("}".to_string());
    out.push(
        "static void chelis_host_require_elementwise_agreement(const chelis_tensor *lhs, const chelis_tensor *rhs, const char *target_label, const char *lhs_label, const char *rhs_label) {"
            .to_string(),
    );
    out.push("    int32_t lhs_rank = chelis_tensor_rank(lhs);".to_string());
    out.push("    int32_t rhs_rank = chelis_tensor_rank(rhs);".to_string());
    out.push("    if (lhs_rank > 0 && rhs_rank > 0 && lhs_rank != rhs_rank) {".to_string());
    out.push(
        "        fprintf(stderr, \"chelis: elementwise operand rank mismatch at host value %s (%s vs %s): %d vs %d\\n\", target_label, lhs_label, rhs_label, lhs_rank, rhs_rank);"
            .to_string(),
    );
    out.push("        abort();".to_string());
    out.push("    }".to_string());
    out.push("    if (lhs_rank == rhs_rank) {".to_string());
    out.push("        for (int32_t axis = 0; axis < lhs_rank; ++axis) {".to_string());
    out.push(
        "            if (chelis_tensor_shape(lhs, axis) != chelis_tensor_shape(rhs, axis)) {"
            .to_string(),
    );
    out.push(
        "                fprintf(stderr, \"chelis: elementwise operand shape mismatch at host value %s (%s vs %s) axis %d\\n\", target_label, lhs_label, rhs_label, axis);"
            .to_string(),
    );
    out.push("                abort();".to_string());
    out.push("            }".to_string());
    out.push("        }".to_string());
    out.push("    }".to_string());
    out.push("}".to_string());
}

/// Render a tensor element by first recovering the exact tagged scalar.
/// The runtime owns the exhaustive dtype dispatch and public text contract.
fn append_tensor_print_helper(out: &mut Vec<String>) {
    out.push(
        "static bool chelis_host_string_eq_cstr(chelis_string lhs, const char *rhs) {".to_string(),
    );
    out.push("    chelis_string owned_rhs = chelis_string_from_cstr(rhs);".to_string());
    out.push("    bool equal = chelis_string_eq(lhs, owned_rhs);".to_string());
    out.push("    chelis_string_release(owned_rhs);".to_string());
    out.push("    return equal;".to_string());
    out.push("}".to_string());
    out.push(String::new());
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
    out.push("    chelis_read_view view = chelis_tensor_read_view(t);".to_string());
    out.push("    int64_t width = chelis_dtype_size(view.dtype);".to_string());
    out.push(
        "    memcpy(&bits, (const uint8_t*)view.data + i * width, (size_t)width);".to_string(),
    );
    out.push("    return chelis_scalar_from_bits(view.dtype, bits);".to_string());
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
    out.push("    chelis_print_string(text);".to_string());
    out.push("    chelis_string_release(text);".to_string());
    out.push("}".to_string());
    out.push(String::new());
    out.push("static void chelis_print_tensor_stdout(const chelis_tensor* t) {".to_string());
    // [05-OBS-4]: a rank-0 tensor renders as its single element, bare -
    // the `tensor(shape=[], data=[..])` wrapper is not an exit form.
    out.push("    int32_t rank = chelis_tensor_rank(t);".to_string());
    out.push("    if (rank == 0) {".to_string());
    out.push("        chelis_print_tensor_elem_stdout(t, 0);".to_string());
    out.push("        return;".to_string());
    out.push("    }".to_string());
    out.push("    printf(\"tensor(shape=[\");".to_string());
    out.push("    for (int32_t d = 0; d < rank; ++d) {".to_string());
    out.push("        if (d > 0) { printf(\", \"); }".to_string());
    out.push("        printf(\"%lld\", (long long)chelis_tensor_shape(t, d));".to_string());
    out.push("    }".to_string());
    out.push("    printf(\"], data=[\");".to_string());
    // [05-OBS-5]: every exit truncates tensor element rendering after 32
    // elements with the `, ...` marker; full-element fidelity is
    // to_list's and the wire's job, never print's.
    out.push("    int64_t size = chelis_tensor_numel(t);".to_string());
    out.push("    int64_t limit = size < 32 ? size : 32;".to_string());
    out.push("    for (int64_t i = 0; i < limit; ++i) {".to_string());
    out.push("        if (i > 0) { printf(\", \"); }".to_string());
    out.push("        chelis_print_tensor_elem_stdout(t, i);".to_string());
    out.push("    }".to_string());
    out.push("    if (size > limit) { printf(\", ...\"); }".to_string());
    out.push("    printf(\"])\");".to_string());
    out.push("}".to_string());
}

fn append_tensor_reshape_helper(out: &mut Vec<String>) {
    out.push(
        "static chelis_tensor* chelis_host_reshape_tensor(chelis_tensor* input, const chelis_list* shape_values) {"
            .to_string(),
    );
    out.push("    return chelis_tensor_reshape(input, shape_values);".to_string());
    out.push("}".to_string());
}

pub(crate) fn emit_host_abi_header(
    projected: &ProjectedHostProgram<'_>,
    program_name: &str,
) -> Result<String, Unsupported> {
    emit_host_declarations(projected.program(), program_name, false, false)
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
            let params = if function.is_monomorphized_specialization() {
                private_host_function_params(&params)
            } else {
                params
            };
            let declaration = format!(
                "{prefix}{} {}({});",
                c_type(&function.ret_ty)?,
                emitted_name,
                params
            );
            Ok(crate::generated_header::render_declaration(
                &function.name,
                &emitted_name,
                &declaration,
            ))
        })
        .collect::<Result<Vec<_>, Unsupported>>()
        .map(|headers| headers.join("\n"))
}

#[derive(Debug, Clone, Copy, Default)]
struct HelperRequirements {
    needs_blas_header: bool,
    needs_math_header: bool,
    needs_fixed_dropout_helpers: bool,
}

impl HelperRequirements {
    fn merge(&mut self, other: Self) {
        self.needs_blas_header |= other.needs_blas_header;
        self.needs_math_header |= other.needs_math_header;
        self.needs_fixed_dropout_helpers |= other.needs_fixed_dropout_helpers;
    }
}

/// Append a tensor helper to `out` and return the runtime headers required by
/// the inner emitter. The caller propagates these headers to the host preamble
/// so each one is emitted exactly once at file scope.
fn append_helper(
    out: &mut Vec<String>,
    helper: &HostTensorHelper,
    verified: VerifiedHostTensorHelperView<'_>,
    helper_name: &str,
    entry_coverage: &[chelis_ir::axis_sources::EntryExtentGuard],
) -> Result<HelperRequirements, Unsupported> {
    let helper_name = random_helper_name(helper_name);
    if verified.execution().is_none()
        && let Some((_input_name, _input_ty)) =
            verified_identity_helper_input(helper, verified.dag())
    {
        out.push(format!(
            "static void {}({}) {{",
            helper_name,
            private_host_params(
                "chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out"
            ),
        ));
        out.push("    (void)n_in;".to_string());
        out.push("    (void)n_out;".to_string());
        out.push("    (void)__chelis_rng;".to_string());
        #[cfg(feature = "native-random-observer")]
        out.push("    (void)__chelis_observer;".to_string());
        out.push("    __chelis_check_host_result_claims(__chelis_caller_result_claims, inputs[0], \"load\", \"numeric trap: domain in load at i64\");".to_string());
        out.push("    outputs[0] = inputs[0];".to_string());
        out.push("}".to_string());
        out.push(String::new());
        return Ok(HelperRequirements::default());
    }

    // Tensor helpers are TU-internal: they are only called from within this
    // generated `.c` file and must never be exported symbols.  `static_entry`
    // ensures the kernel function itself gets `static` linkage so that when
    // compiled with `-shared -fPIC` the symbol is not exported via PLT.
    let dag = verified.dag();
    let uses_blas = verified.execution().is_none()
        && dag
            .nodes()
            .iter()
            .any(|node| matches!(node.op, RiscOp::BlasMatmul { .. }));
    let options = crate::CodegenOptions {
        use_blas: uses_blas,
        static_entry: true,
        ..crate::CodegenOptions::default()
    };
    let helper_src = if let Some(execution) = verified.execution() {
        CEmitter::emit_verified_evaluation_with_options(
            dag,
            execution,
            &helper_name,
            options,
            entry_coverage,
            #[cfg(feature = "native-random-observer")]
            verified.source_location(),
        )?
    } else {
        CEmitter::emit_verified_dag_with_options(dag, &helper_name, options, entry_coverage)?
    };
    // The CEmitter prepends dtype-specific uniform sampling helpers to
    // every DAG it emits so that a standalone-emitted kernel
    // stays self-contained. When multiple helpers get concatenated into a
    // single `main.c` that duplicates the definition and gcc rejects the
    // redefinition. We filter the prelude out here and rely on
    // `emit_host_program` to emit exactly one copy at file scope.
    let mut skipping_helper_prelude = false;
    let mut requirements = HelperRequirements {
        needs_fixed_dropout_helpers: verified.execution().is_some(),
        ..HelperRequirements::default()
    };
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
        if matches!(
            line,
            "/* CHELIS_UNIFORM_HELPERS_BEGIN */" | "/* CHELIS_DROPOUT_HELPERS_BEGIN */"
        ) {
            skipping_helper_prelude = true;
            continue;
        }
        if skipping_helper_prelude {
            if matches!(
                line,
                "/* CHELIS_UNIFORM_HELPERS_END */" | "/* CHELIS_DROPOUT_HELPERS_END */"
            ) {
                skipping_helper_prelude = false;
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

/// Declare a helper that a peer translation unit defines. The prototype is
/// the shared tensor-helper ABI, so the wrapper's call site is unchanged
/// whether the body is the C emitter's `static` definition or a device
/// backend's exported entry.
fn append_external_helper_declaration(out: &mut Vec<String>, helper_name: &str) {
    out.push("#ifdef __cplusplus".to_string());
    out.push("extern \"C\"".to_string());
    out.push("#endif".to_string());
    out.push(format!(
        "void {helper_name}(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);"
    ));
    // Peer translation units keep their established ABI and baked-seed
    // behavior. The private adapter does not export Random state to a device.
    out.push(format!(
        "static void {}({}) {{",
        random_helper_name(helper_name),
        private_random_params(
            "chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out"
        )
    ));
    out.push("    (void)__chelis_rng;".to_string());
    #[cfg(feature = "native-random-observer")]
    out.push("    (void)__chelis_observer;".to_string());
    out.push(format!("    {helper_name}(inputs, n_in, outputs, n_out);"));
    out.push("}".to_string());
    out.push(String::new());
}

fn random_helper_name(name: &str) -> String {
    format!("{name}__with_rng")
}

#[cfg(not(feature = "native-random-observer"))]
fn private_random_params(params: &str) -> String {
    if params.is_empty() {
        "chelis_rng_state *__chelis_rng".to_string()
    } else {
        format!("{params}, chelis_rng_state *__chelis_rng")
    }
}

#[cfg(feature = "native-random-observer")]
fn private_random_params(params: &str) -> String {
    if params.is_empty() {
        crate::random_observer::PRIVATE_PARAM.to_string()
    } else {
        format!("{params}, {}", crate::random_observer::PRIVATE_PARAM)
    }
}

fn append_private_context_args(args: &mut Vec<String>) {
    #[cfg(not(feature = "native-random-observer"))]
    args.push("__chelis_rng".to_string());
    #[cfg(feature = "native-random-observer")]
    args.extend(
        crate::random_observer::PRIVATE_ARGS
            .iter()
            .map(|arg| (*arg).to_string()),
    );
}

fn append_invocation_random_context(out: &mut Vec<String>) {
    out.push("    chelis_rng_state __chelis_rng_local = {0ULL, 0ULL, 0};".to_string());
    out.push("    chelis_rng_state *__chelis_rng = &__chelis_rng_local;".to_string());
    #[cfg(feature = "native-random-observer")]
    crate::random_observer::append_inactive_context(out, "    ");
}

fn verified_identity_helper_input(
    helper: &HostTensorHelper,
    dag: VerifiedDagView<'_>,
) -> Option<(String, chelis_ir::dag::TensorType)> {
    if dag.roots().len() != 1 || helper.inputs.len() != 1 {
        return None;
    }
    let root = dag.roots()[0];
    let node = dag.get(root)?;
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
    let params = if function.is_monomorphized_specialization() {
        private_host_function_params(&params)
    } else {
        params
    };
    let declaration = format!(
        "{prefix}{} {}({params});",
        c_type(&function.ret_ty)?,
        emitted_name,
    );
    if !internal_linkage {
        out.push(crate::generated_header::render_authored_export_begin(
            &function.name,
            emitted_name,
            &declaration,
        ));
    }
    out.push(format!("{} {{", declaration.trim_end_matches(';')));
    let rendered = unsupported.to_string();
    let safe = chelis_ir::span_sanitize::sanitize_for_format_string(&rendered);
    out.push(format!("    fprintf(stderr, \"%s\\n\", \"{safe}\");"));
    out.push("    abort();".to_string());
    out.push("}".to_string());
    if !internal_linkage {
        out.push(crate::generated_header::render_authored_export_end(
            &function.name,
        ));
    }
    Ok(())
}

/// Resolve a returned alias outside its local let chain. Consuming bindings
/// backwards preserves the scope in which each initializer read its value.
fn host_tail_binding_name(body: &HostExpr) -> Option<&str> {
    let mut current = body;
    let mut visible: Vec<&HostBinding> = Vec::new();
    'value: loop {
        match &current.kind {
            HostExprKind::Var(name, _) => {
                while let Some(binding) = visible.pop() {
                    if binding.name == *name {
                        current = &binding.value;
                        continue 'value;
                    }
                }
                return Some(name.as_str());
            }
            HostExprKind::Let { bindings, body, .. } => {
                visible.extend(bindings);
                current = body;
            }
            _ => return None,
        }
    }
}

/// Select the original producing binding through sequential value aliases.
/// Guard placement follows this identity, not a later use of the same name.
fn host_result_binding_index(bindings: &[HostBinding], body: &HostExpr) -> Option<usize> {
    let mut name = host_tail_binding_name(body)?;
    let mut found = None;
    for (index, binding) in bindings.iter().enumerate().rev() {
        if binding.name == name {
            found = Some(index);
            match host_tail_binding_name(&binding.value) {
                Some(alias) => name = alias,
                None => break,
            }
        }
    }
    found
}

/// One declaration's ordered literal axes. Its invocation frame is forwarded
/// unchanged through branches and calls; the selected producer supplies `<op>`.
struct HostResultClaim {
    rank: usize,
    axes: Vec<(usize, usize)>,
}

impl HostResultClaim {
    fn from_tensor_type(ty: &TensorType) -> Self {
        Self {
            rank: ty.dims.len(),
            axes: ty
                .dims
                .iter()
                .enumerate()
                .filter_map(|(axis, dim)| match dim {
                    DimInfo::Lit(required) => Some((axis, *required)),
                    DimInfo::Named(_, _) => None,
                })
                .collect(),
        }
    }

    fn of(function: &HostFunction) -> Option<Self> {
        let HostAbiType::Tensor(ty) = &function.ret_ty else {
            return None;
        };
        let axes = ty
            .dims
            .iter()
            .enumerate()
            .filter_map(|(axis, dim)| match dim {
                DimInfo::Lit(required)
                    if !function.helper_result_claim_axes.contains(
                        &chelis_ir::dag::RtAxis::Lit(i32::try_from(axis).expect("rank fits i32")),
                    ) =>
                {
                    Some((axis, *required))
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        (!axes.is_empty()).then_some(Self {
            rank: ty.dims.len(),
            axes,
        })
    }

    fn frame_lines(
        &self,
        indent: &str,
        axes_name: &str,
        frame_name: &str,
        parent: &str,
        claims_name: Option<&str>,
    ) -> Vec<String> {
        let mut lines = vec![format!("{indent}const int64_t {axes_name}[][2] = {{")];
        for (axis, required) in &self.axes {
            lines.push(format!("{indent}    {{ {axis}, {required} }},"));
        }
        lines.push(format!("{indent}}};"));
        lines.push(format!(
            "{indent}const __chelis_host_result_claim {frame_name} = {{ {parent}, {}, {}, {axes_name} }};",
            self.rank,
            self.axes.len()
        ));
        if let Some(claims_name) = claims_name {
            lines.push(format!(
                "{indent}const __chelis_host_result_claim *{claims_name} = &{frame_name};"
            ));
        }
        lines
    }
}

/// This context is translation-unit private. Public wrappers retain their
/// authored signatures, while one owned callee accepts any caller's literals.
fn private_host_params(params: &str) -> String {
    format!(
        "{}, const __chelis_host_result_claim *__chelis_caller_result_claims",
        private_random_params(params)
    )
}

/// Owned host bodies additionally return private producer provenance. Public
/// wrappers pass NULL, so this never enters the generated ABI.
fn private_host_function_params(params: &str) -> String {
    format!(
        "{}, __chelis_host_result_origin *__chelis_result_origin_out",
        private_host_params(params)
    )
}

fn append_host_result_claim_support(out: &mut Vec<String>) {
    out.push(r#"typedef struct __chelis_host_result_claim {
    const struct __chelis_host_result_claim *next;
    int64_t rank;
    int64_t count;
    const int64_t (*axes)[2];
} __chelis_host_result_claim;

typedef struct __chelis_host_result_origin {
    const char *op;
    const char *trap;
} __chelis_host_result_origin;

static void __chelis_check_host_result_extent_claims(const __chelis_host_result_claim *claims, int64_t rank, const int64_t (*observations)[3], int64_t count, const char *op, const char *trap) {
    for (; claims != NULL; claims = claims->next) {
        if (rank != claims->rank) continue;
        for (int64_t i = 0; i < claims->count; ++i) {
            for (int64_t j = 0; j < count; ++j) {
                if (claims->axes[i][0] != observations[j][0]) continue;
                int64_t required = claims->axes[i][1];
                int64_t observed = observations[j][2];
                if (required != observed) {
                    fprintf(stderr, "extent `%lld`: claimed = %lld, %s axis %lld = %lld\n", (long long)required, (long long)required, op, (long long)observations[j][1], (long long)observed);
                    chelis_numeric_trap(trap);
                }
            }
        }
    }
}

static void __chelis_check_host_result_claims(const __chelis_host_result_claim *claims, const chelis_tensor *value, const char *op, const char *trap) {
    for (; claims != NULL; claims = claims->next) {
        if (chelis_tensor_rank(value) != claims->rank) continue;
        for (int64_t i = 0; i < claims->count; ++i) {
            int64_t axis = claims->axes[i][0];
            int64_t required = claims->axes[i][1];
            int64_t observed = chelis_tensor_shape(value, axis);
            if (observed != required) {
                fprintf(stderr, "extent `%lld`: claimed = %lld, %s axis %lld = %lld\n", (long long)required, (long long)required, op, (long long)axis, (long long)observed);
                chelis_numeric_trap(trap);
            }
        }
    }
}
"#.to_string());
}

/// The complete signature owns entry order; helper partitioning does not.
fn function_entry_plan(function: &HostFunction) -> chelis_ir::host::SignatureEntryPlan {
    chelis_ir::host::SignatureEntryPlan::new(function.params.iter().filter_map(|param| {
        let HostAbiType::Tensor(ty) = &param.ty else {
            return None;
        };
        Some(chelis_ir::host::HostTensorInput {
            name: param.name.clone(),
            ty: ty.clone(),
        })
    }))
}

fn signature_entry_lines(
    plan: &chelis_ir::host::SignatureEntryPlan,
    args: &[String],
    indent: &str,
    delegated: &[chelis_ir::axis_sources::EntryExtentGuard],
) -> Result<Vec<String>, Unsupported> {
    use chelis_ir::axis_sources::EntryExtentGuard;
    if args.len() != plan.observations().nodes().len() {
        return Err(invalid_abi_shape(
            "signature entry lost an input observation".into(),
            "signature entry",
        ));
    }
    let mut lines = Vec::new();
    // No extent read may obscure a malformed external input's rank/null
    // diagnostic. These metadata checks dominate the ordered comparisons.
    for (node, actual) in plan.observations().nodes().iter().zip(args) {
        let RiscOp::Load { name } = &node.op else {
            unreachable!("signature observation")
        };
        let label = chelis_ir::span_sanitize::sanitize_for_format_string(name.as_str());
        let rank = node.output_type.dims.len();
        lines.push(format!("{indent}if ({actual} == NULL) {{ fprintf(stderr, \"input `{label}` is NULL\\n\"); abort(); }}"));
        lines.push(format!("{indent}if (chelis_tensor_rank({actual}) != {rank}) {{ fprintf(stderr, \"input `{label}` expected rank {rank}, got %d\\n\", chelis_tensor_rank({actual})); abort(); }}"));
    }
    let read = |(load, axis): (chelis_ir::NodeId, usize)| {
        let node = plan.observations().get(load).expect("signature witness");
        let RiscOp::Load { name } = &node.op else {
            unreachable!("signature observation")
        };
        let label = chelis_ir::span_sanitize::sanitize_for_format_string(name.as_str());
        (
            format!("chelis_tensor_shape({}, {axis})", args[load.0]),
            label.into_owned(),
            axis,
        )
    };
    for guard in plan
        .guards()
        .iter()
        .filter(|guard| !delegated.contains(guard))
    {
        let (left, right, context) = match guard {
            EntryExtentGuard::Named {
                claim,
                canonical,
                observed,
            } => {
                let (left, first, first_axis) = read(*canonical);
                let (right, later, later_axis) = read(*observed);
                let claim = chelis_ir::span_sanitize::sanitize_for_format_string(claim);
                let context = format!(
                    "fprintf(stderr, \"extent `{claim}`: {first} axis {first_axis} = %lld, {later} axis {later_axis} = %lld\\n\", (long long)({left}), (long long)({right}));"
                );
                (left, right, context)
            }
            EntryExtentGuard::Literal { required, observed } => {
                let (right, label, axis) = read(*observed);
                let context = format!(
                    "fprintf(stderr, \"input `{label}` axis {axis} expected {required}, got %lld\\n\", (long long)({right}));"
                );
                (required.to_string(), right, context)
            }
        };
        lines.push(format!("{indent}if ({right} != {left}) {{"));
        lines.push(format!("{indent}    {context}"));
        lines.push(format!(
            "{indent}    chelis_numeric_trap(\"numeric trap: domain in load at i64\");"
        ));
        lines.push(format!("{indent}}}"));
    }
    Ok(lines)
}

#[allow(clippy::too_many_arguments)]
fn emit_function(
    out: &mut Vec<String>,
    function: &HostFunction,
    emitted_name: &str,
    internal_names: &UnordMap<String, String>,
    function_specializations: &UnordMap<String, HostFunctionSpecialization>,
    internal_linkage: bool,
    ownership_sites: &[ProjectedHostSite<'_>],
    owner_bindings: &[(VerifiedOwnerId, String)],
    helper_output_counts: &[usize],
    verified_helpers: &[VerifiedHostTensorHelperView<'_>],
    external_helpers: &UnordSet<String>,
    #[cfg(feature = "native-random-observer")]
    source_sites: &[crate::random_observer::SourceSite<'_>],
) -> Result<(), Unsupported> {
    let params = function
        .params
        .iter()
        .map(|param| c_decl(&param.ty, &param.name))
        .collect::<Result<Vec<_>, _>>()?
        .join(", ");
    let authored = function.origin == chelis_ir::host::HostFunctionOrigin::Authored;
    let body_name = internal_names
        .get(&function.name)
        .expect("verified function has an internal emitted name");
    let prefix = if internal_linkage || authored {
        "static inline "
    } else {
        ""
    };
    out.push(format!(
        "{prefix}{} {}({}) {{",
        c_type(&function.ret_ty)?,
        body_name,
        private_host_function_params(&params)
    ));
    out.push("    (void)__chelis_rng;".to_string());
    let mut emitter = HostEmitter::new(
        "    ".to_string(),
        emitted_name,
        internal_names.clone(),
        function_specializations.clone(),
        HostTensorHelpers {
            helpers: &function.tensor_helpers,
            output_counts: helper_output_counts,
            result_origins: verified_helpers
                .iter()
                .copied()
                .map(verified_helper_result_origin)
                .collect::<Result<Vec<_>, _>>()?,
        },
        ownership_sites,
    );
    emitter.entry_projection = entry::helper_coverage_with_verified(function, verified_helpers);
    emitter.external_helpers = external_helpers.clone();
    for param in &function.params {
        if matches!(param.ty, HostAbiType::Tensor(_)) {
            let origin = result_origin_name(&param.name);
            emitter.lines.push(format!(
                "{}__chelis_host_result_origin {origin} = {{ \"load\", \"numeric trap: domain in load at i64\" }};",
                emitter.indent
            ));
        }
    }
    #[cfg(feature = "native-random-observer")]
    {
        if !crate::random_observer::source_bijection(source_sites, &emitter.expression_sites) {
            return Err(invalid_abi_shape(
                "source identity sidecar is not the verified expression-site bijection".into(),
                "native Random source identity",
            ));
        }
        emitter.source_sites = source_sites.to_vec();
        let source = source_sites
            .first()
            .expect("function has a body expression");
        let unit = source.verified.unit_word();
        let admitted = usize::from(source.supported());
        out.push(format!("    __chelis_random_function_frame __chelis_source_function = __chelis_random_push_function(__chelis_observer, {unit}ULL, {admitted});"));
    }
    if owner_bindings.len() < function.params.len()
        || function
            .params
            .iter()
            .zip(owner_bindings)
            .any(|(param, (_, name))| param.name != *name)
    {
        return Err(invalid_abi_shape(
            format!(
                "verified function's {} payload parameters disagree with {} body-owner bindings",
                function.params.len(),
                owner_bindings.len()
            ),
            "verified C host ownership emission",
        ));
    }
    for (owner, name) in owner_bindings {
        if emitter
            .owner_vars
            .insert(*owner, c_ident(name).into_owned())
            .is_some()
        {
            return Err(invalid_abi_shape(
                format!("verified function repeats parameter owner {owner:?}"),
                "verified C host ownership emission",
            ));
        }
    }
    let entry = ownership_sites
        .iter()
        .find(|site| site.kind == chelis_ir::ownership::HostSiteKind::FunctionEntry)
        .ok_or_else(|| {
            invalid_abi_shape(
                "verified host function has no FunctionEntry site".to_string(),
                "verified C host ownership emission",
            )
        })?;
    let entry_plan = function_entry_plan(function);
    let entry_args = function
        .params
        .iter()
        .filter(|param| matches!(param.ty, HostAbiType::Tensor(_)))
        .map(|param| c_ident(&param.name).into_owned())
        .collect::<Vec<_>>();
    let delegated_entry_guards = entry::delegated_function_guards(function, verified_helpers);
    emitter.lines.extend(signature_entry_lines(
        &entry_plan,
        &entry_args,
        &emitter.indent,
        &delegated_entry_guards,
    )?);
    // Entry guards still read parameters the body does not use. Their
    // verified entry drops run only after those witness reads finish.
    emitter.emit_entry_terminals(entry, authored)?;
    // A frame belongs to this invocation, not to a selected callee name.
    // The expression spine forwards the frame; branch arms share its immutable
    // contents and arguments/sibling bindings never inherit it.
    match HostResultClaim::of(function) {
        Some(claim) => emitter.lines.extend(claim.frame_lines(
            &emitter.indent,
            "__chelis_result_axes",
            "__chelis_declared_result",
            "__chelis_caller_result_claims",
            Some("__chelis_result_claims"),
        )),
        None => emitter.lines.push(format!("{}const __chelis_host_result_claim *__chelis_result_claims = __chelis_caller_result_claims;", emitter.indent)),
    }
    emitter.result_claims = Some("__chelis_result_claims".to_string());
    emitter.claim_on_spine = true;
    emitter.emit_expr_to_var(&function.body, "__result", &function.ret_ty)?;
    let terminal = ownership_sites
        .iter()
        .find(|site| site.kind == chelis_ir::ownership::HostSiteKind::FunctionReturn)
        .ok_or_else(|| {
            invalid_abi_shape(
                "verified host function has no FunctionReturn site".to_string(),
                "verified C host ownership emission",
            )
        })?;
    emitter.emit_terminal_site(terminal, Some("__result"))?;
    if matches!(function.ret_ty, HostAbiType::Tensor(_)) {
        emitter.lines.push(format!(
            "{}if (__chelis_result_origin_out != NULL) *__chelis_result_origin_out = {};",
            emitter.indent,
            result_origin_name("__result")
        ));
    }
    emitter.finish_expression_sites()?;
    out.extend(emitter.lines);
    #[cfg(feature = "native-random-observer")]
    out.push(
        "    __chelis_random_pop_function(__chelis_observer, __chelis_source_function);".into(),
    );
    out.push("    return __result;".to_string());
    out.push("}".to_string());

    if authored {
        let entry_uses = authored_entry_uses(ownership_sites, function.params.len())?;
        let wrapper_params = function
            .params
            .iter()
            .map(|param| c_decl(&param.ty, &param.name))
            .collect::<Result<Vec<_>, _>>()?
            .join(", ");
        let declaration = format!(
            "{} {}({wrapper_params});",
            c_type(&function.ret_ty)?,
            emitted_name,
        );
        out.push(crate::generated_header::render_authored_export_begin(
            &function.name,
            emitted_name,
            &declaration,
        ));
        out.push(format!("{} {{", declaration.trim_end_matches(';')));
        append_invocation_random_context(out);
        let mut args = Vec::with_capacity(function.params.len());
        for (index, (param, use_)) in function.params.iter().zip(&entry_uses).enumerate() {
            if *use_ == VerifiedOwnershipUse::Move && retain_call(&param.name, &param.ty).is_some()
            {
                let owned = format!("__chelis_owned_arg_{index}");
                out.push(format!(
                    "    {} = {};",
                    c_decl(&param.ty, &owned)?,
                    c_ident(&param.name)
                ));
                out.push(format!(
                    "    {}",
                    retain_call(&owned, &param.ty).expect("heap retain")
                ));
                args.push(owned);
            } else {
                args.push(c_ident(&param.name).into_owned());
            }
        }
        append_private_context_args(&mut args);
        args.push("NULL".to_string());
        args.push("NULL".to_string());
        out.push(format!(
            "    {} __result = {}({});",
            c_type(&function.ret_ty)?,
            body_name,
            args.join(", ")
        ));
        out.push("    return __result;".to_string());
        out.push("}".to_string());
        out.push(crate::generated_header::render_authored_export_end(
            &function.name,
        ));

        #[cfg(feature = "native-random-observer")]
        {
            let observed_params = if wrapper_params.is_empty() {
                "__chelis_random_observer_sink __chelis_sink, void *__chelis_sink_context, uint64_t __chelis_invocation".to_string()
            } else {
                format!(
                    "{wrapper_params}, __chelis_random_observer_sink __chelis_sink, void *__chelis_sink_context, uint64_t __chelis_invocation"
                )
            };
            out.push(format!(
                "static {} __chelis_observed_{}({observed_params}) {{",
                c_type(&function.ret_ty)?,
                emitted_name
            ));
            out.push("    chelis_rng_state __chelis_rng_local = {0ULL, 0ULL, 0};".to_string());
            out.push("    chelis_rng_state *__chelis_rng = &__chelis_rng_local;".to_string());
            crate::random_observer::append_observed_context(out, "    ");
            let mut args = Vec::with_capacity(function.params.len() + 2);
            for (index, (param, use_)) in function.params.iter().zip(&entry_uses).enumerate() {
                if *use_ == VerifiedOwnershipUse::Move
                    && retain_call(&param.name, &param.ty).is_some()
                {
                    let owned = format!("__chelis_owned_arg_{index}");
                    out.push(format!(
                        "    {} = {};",
                        c_decl(&param.ty, &owned)?,
                        c_ident(&param.name)
                    ));
                    out.push(format!(
                        "    {}",
                        retain_call(&owned, &param.ty).expect("heap retain")
                    ));
                    args.push(owned);
                } else {
                    args.push(c_ident(&param.name).into_owned());
                }
            }
            append_private_context_args(&mut args);
            args.push("NULL".to_string());
            args.push("NULL".to_string());
            out.push(format!(
                "    {} __result = {}({});",
                c_type(&function.ret_ty)?,
                body_name,
                args.join(", ")
            ));
            out.push("    return __result;".to_string());
            out.push("}".to_string());
        }
    }
    Ok(())
}

fn authored_entry_uses(
    sites: &[ProjectedHostSite<'_>],
    param_count: usize,
) -> Result<Vec<VerifiedOwnershipUse>, Unsupported> {
    let entry = sites
        .iter()
        .find(|site| site.kind == chelis_ir::ownership::HostSiteKind::FunctionEntry)
        .ok_or_else(|| {
            invalid_abi_shape(
                "verified authored function has no FunctionEntry site".to_string(),
                "verified C host ownership emission",
            )
        })?;
    let args = entry.directives.iter().find_map(|action| match action {
        VerifiedHostAction::Terminator(VerifiedHostTerminator::Jump { edge, .. }) => {
            Some(edge.args())
        }
        _ => None,
    });
    let Some(args) = args else {
        return Err(invalid_abi_shape(
            "verified authored FunctionEntry has no consuming body edge".to_string(),
            "verified C host ownership emission",
        ));
    };
    if args.len() < param_count {
        return Err(invalid_abi_shape(
            format!(
                "verified authored FunctionEntry carries {} arguments for {param_count} parameters",
                args.len()
            ),
            "verified C host ownership emission",
        ));
    }
    Ok(args[..param_count].iter().map(|arg| arg.use_()).collect())
}

const BASE_MAIN_INDENT: &str = "    ";

#[allow(clippy::too_many_arguments)]
fn emit_main(
    out: &mut Vec<String>,
    program_name: &str,
    program: &HostProgram,
    manifest: &RootManifest,
    hoisted: &UnordSet<&str>,
    ownership_sites: &[ProjectedHostSite<'_>],
    internal_names: &UnordMap<String, String>,
    helper_output_counts: &[usize],
    helper_result_origins: Vec<Option<String>>,
    external_helpers: &UnordSet<String>,
) -> Result<(), Unsupported> {
    out.push("int main(void) {".to_string());
    append_invocation_random_context(out);
    // chelis#840: the globals emitter needs the same original-to-emitted
    // function-name map as function bodies, or a global calling a def
    // whose name was mangled (`double`) or renamed (`main`) emits the raw
    // name and the C cannot compile.
    let mut emitter = HostEmitter::new(
        BASE_MAIN_INDENT.to_string(),
        &format!("{program_name}__global"),
        internal_names.clone(),
        function_specializations(program),
        HostTensorHelpers {
            helpers: &program.global_tensor_helpers,
            output_counts: helper_output_counts,
            result_origins: helper_result_origins,
        },
        ownership_sites,
    );
    emitter.entry_projection = entry::global_helper_coverage(program);
    emitter.external_helpers = external_helpers.clone();
    for (index, binding) in program.globals.iter().enumerate() {
        let binding_var = format!("__binding_{index}_value");
        emitter.emit_expr_to_var(&binding.value, &binding_var, &binding.ty)?;
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
        if matches!(binding.ty, HostType::Tensor(_)) {
            let binding_origin = result_origin_name(&binding.name);
            let value_origin = result_origin_name(&binding_var);
            if hoisted.contains(binding.name.as_str()) {
                emitter
                    .lines
                    .push(format!("    {binding_origin} = {value_origin};"));
            } else {
                emitter.lines.push(format!(
                    "    __chelis_host_result_origin {binding_origin} = {value_origin};"
                ));
            }
        }
    }
    let root_sites = ownership_sites
        .iter()
        .filter(|site| site.kind == chelis_ir::ownership::HostSiteKind::ManifestRoot)
        .collect::<Vec<_>>();
    let mut consumed_root_sites = UnordSet::new();
    for binding in &program.globals {
        for display in &binding.display_roots {
            let mut entries = manifest.entries.iter().enumerate().filter(|(_, root)| {
                let short_def = if chelis_types::is_linker_format_name(&root.def_name) {
                    chelis_types::demangle_ident(&root.def_name)
                } else {
                    root.def_name.clone()
                };
                let suffix = root.name.strip_prefix(root.def_name.as_str()).unwrap_or("");
                let selected = binding.name == root.def_name
                    || matches!(
                        &binding.value.kind,
                        HostExprKind::Call { function, args, .. }
                            if function == &root.def_name && args.is_empty()
                    );
                root.lane == Lane::Host
                    && selected
                    && display.name == format!("{short_def}{suffix}")
                    && display.path == root.path
            });
            let manifest_index = entries.next().map(|(index, _)| index);
            if entries.next().is_some() {
                return Err(invalid_abi_shape(
                    format!(
                        "materialized root `{}` has duplicate manifest entries",
                        display.name
                    ),
                    "verified C host ownership emission",
                ));
            }
            let mut sites = root_sites.iter().copied().filter(|site| {
                let identity = site.directives.iter().any(|action| {
                    matches!(
                        action,
                        VerifiedHostAction::ManifestRoot {
                            manifest_index: candidate,
                            ..
                        } if *candidate == manifest_index
                    )
                });
                let label = manifest_index.is_some()
                    || site.directives.iter().any(|action| {
                        matches!(
                            action,
                            VerifiedHostAction::Operation(VerifiedHostOperation::RootConsume {
                                root,
                                ..
                            }) if *root == display.name
                        )
                    });
                identity && label
            });
            let site = sites.next().ok_or_else(|| {
                invalid_abi_shape(
                    format!("display root `{}` has no verified root site", display.name),
                    "verified C host ownership emission",
                )
            })?;
            if sites.next().is_some() || !consumed_root_sites.insert(site.id) {
                return Err(invalid_abi_shape(
                    format!(
                        "display root `{}` has duplicate verified root sites",
                        display.name
                    ),
                    "verified C host ownership emission",
                ));
            }
            emitter.prepare_manifest_root(site)?;
            emitter.emit_manifest_root(
                &display.name,
                &c_ident(&binding.name),
                &binding.ty,
                &display.path,
            )?;
            emitter.finish_manifest_root(site)?;
        }
        if binding.display_roots.is_empty()
            && let Some(display_name) = binding.display_name.as_deref()
        {
            let mut sites = root_sites.iter().copied().filter(|site| {
                let display = site.directives.iter().any(|action| {
                    matches!(
                        action,
                        VerifiedHostAction::ManifestRoot {
                            manifest_index: None,
                            ..
                        }
                    )
                });
                let consume = site.directives.iter().any(|action| {
                    matches!(
                        action,
                        VerifiedHostAction::Operation(VerifiedHostOperation::RootConsume {
                            root,
                            ..
                        }) if *root == display_name
                    )
                });
                display && consume
            });
            let site = sites.next().ok_or_else(|| {
                invalid_abi_shape(
                    format!("displayed root `{display_name}` has no verified root site"),
                    "verified C host ownership emission",
                )
            })?;
            if sites.next().is_some() || !consumed_root_sites.insert(site.id) {
                return Err(invalid_abi_shape(
                    format!("displayed root `{display_name}` has duplicate verified root sites"),
                    "verified C host ownership emission",
                ));
            }
            emitter.prepare_manifest_root(site)?;
            emitter.emit_labeled_root(display_name, &c_ident(&binding.name), &binding.ty)?;
            emitter.finish_manifest_root(site)?;
        }
    }
    if consumed_root_sites.len() != root_sites.len() {
        return Err(invalid_abi_shape(
            format!(
                "verified ownership root-site cursor consumed {} of {} roots",
                consumed_root_sites.len(),
                root_sites.len()
            ),
            "verified C host ownership emission",
        ));
    }
    let terminal = ownership_sites
        .iter()
        .find(|site| site.kind == chelis_ir::ownership::HostSiteKind::FunctionReturn)
        .ok_or_else(|| {
            invalid_abi_shape(
                "verified roots unit has no terminal site".to_string(),
                "verified C host ownership emission",
            )
        })?;
    emitter.emit_terminal_site(terminal, None)?;
    emitter.finish_expression_sites()?;
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
    let mut referenced: UnordSet<String> = UnordSet::new();
    for function in &program.functions {
        collect_var_names(&function.body, &mut referenced);
    }
    let mut seen: UnordSet<&str> = UnordSet::new();
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
fn collect_var_names(expr: &HostExpr, out: &mut UnordSet<String>) {
    match &expr.kind {
        HostExprKind::ResultClaimScope { body, .. } => collect_var_names(body, out),
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
        | HostExprKind::SignatureEntry { args, .. }
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

fn collect_callback_var_names(callback: &HostCallback, out: &mut UnordSet<String>) {
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
fn collect_referenced_fn_names(expr: &HostExpr, out: &mut UnordSet<String>) {
    collect_var_names(expr, out);
    fn walk(expr: &HostExpr, out: &mut UnordSet<String>) {
        match &expr.kind {
            HostExprKind::ResultClaimScope { body, .. } => walk(body, out),
            HostExprKind::Call { function, args, .. } => {
                out.insert(function.clone());
                for arg in args {
                    walk(arg, out);
                }
            }
            HostExprKind::Builtin { args, .. }
            | HostExprKind::TensorCall { args, .. }
            | HostExprKind::SignatureEntry { args, .. } => {
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
    fn walk_callback(callback: &HostCallback, out: &mut UnordSet<String>) {
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
fn host_functions_reachable_from_main(program: &HostProgram) -> UnordSet<String> {
    let by_name: UnordMap<&str, &HostFunction> = program
        .functions
        .iter()
        .map(|function| (function.name.as_str(), function))
        .collect();
    let mut seed = UnordSet::new();
    for binding in &program.globals {
        collect_referenced_fn_names(&binding.value, &mut seed);
    }
    let mut reachable: UnordSet<String> = UnordSet::new();
    let mut stack: Vec<String> = seed
        .into_sorted()
        .into_iter()
        .filter(|name| by_name.contains_key(name.as_str()))
        .collect();
    while let Some(name) = stack.pop() {
        if !reachable.insert(name.clone()) {
            continue;
        }
        if let Some(function) = by_name.get(name.as_str()) {
            let mut refs = UnordSet::new();
            collect_referenced_fn_names(&function.body, &mut refs);
            for r in refs.into_sorted() {
                if by_name.contains_key(r.as_str()) && !reachable.contains(&r) {
                    stack.push(r);
                }
            }
        }
    }
    reachable
}

fn verified_helper_result_origin(
    helper: VerifiedHostTensorHelperView<'_>,
) -> Result<Option<String>, Unsupported> {
    let dag = helper.dag();
    let [root] = dag.roots() else {
        return Ok(None);
    };
    let sites = dag.result_extent_sites(*root);
    let Some(operation) = sites.first().map(|site| site.operation()) else {
        return Ok(None);
    };
    if sites.iter().all(|site| site.operation() == operation) {
        Ok(Some(operation.to_string()))
    } else {
        Err(invalid_abi_shape(
            "one returned tensor root has conflicting per-axis producer operations".to_string(),
            "verified tensor-helper result provenance",
        ))
    }
}

struct HostEmitter<'a> {
    lines: Vec<String>,
    indent: String,
    helper_prefix: String,
    emitted_names: UnordMap<String, String>,
    function_specializations: UnordMap<String, HostFunctionSpecialization>,
    tensor_helpers: &'a [HostTensorHelper],
    tensor_helper_output_counts: &'a [usize],
    tensor_helper_result_origins: Vec<Option<String>>,
    expression_sites: Vec<ProjectedHostSite<'a>>,
    expression_site_index: usize,
    entry_projection: entry::Projection,
    external_helpers: UnordSet<String>,
    #[cfg(feature = "native-random-observer")]
    source_sites: Vec<crate::random_observer::SourceSite<'a>>,
    pre_emitted_clone_sites: UnordSet<HostSiteId>,
    pre_emitted_terminals: UnordSet<(HostSiteId, VerifiedOperationId)>,
    owner_vars: UnordMap<VerifiedOwnerId, String>,
    temp_counter: usize,
    /// Immutable invocation context. Only the expression on the returned-value
    /// spine receives it; nested arguments and sibling bindings get no context.
    result_claims: Option<String>,
    /// Taken at each expression entry and forwarded to its returned-value child.
    claim_on_spine: bool,
}

struct HostTensorHelpers<'a> {
    helpers: &'a [HostTensorHelper],
    output_counts: &'a [usize],
    result_origins: Vec<Option<String>>,
}

/// Whether the verified intrinsic application labelled `label` at this site
/// takes its container operand by Move (chelis#2205).
///
/// Every call site below hands `arg_vars[0]` to the consuming entry point,
/// so this answers a question about operand zero specifically, and it
/// refuses a move anywhere else rather than reporting it as a consumed
/// container. `chelis_ir::ownership`'s `CONTAINER_CONSUMERS` table chooses
/// which operand the scheduler upgrades and this crate cannot read it, so a
/// row naming a different operand is a disagreement only the refusal can
/// catch. Reporting "some operand moved" instead would route a
/// still-borrowed operand into an entry point that releases it, which is a
/// double release; ignoring the move entirely would merely leak. Neither is
/// acceptable, and the disagreement is a compiler defect, so it fails
/// closed. The site of a builtin expression carries exactly one intrinsic
/// application with that label; two would mean the emitter and the
/// ownership sites disagree about the tree, and that fails closed too.
fn container_operand_is_moved(
    site: &ProjectedHostSite<'_>,
    label: &str,
) -> Result<bool, Unsupported> {
    let mut found = None;
    for action in &site.directives {
        if let VerifiedHostAction::Operation(VerifiedHostOperation::Apply {
            kind: VerifiedApplyKind::Intrinsic,
            label: site_label,
            args,
            ..
        }) = action
            && *site_label == label
        {
            // Lowering borrows every builtin operand and the scheduler
            // upgrades at most the one operand its row names. This emitter
            // consumes operand zero, so exactly one moved operand at
            // position zero is the consuming shape, no moved operand is the
            // borrowing shape, and anything else is a disagreement between
            // the table and this emitter that must not reach generated code.
            let mut positions = args
                .iter()
                .enumerate()
                .filter(|(_, arg)| arg.use_() == VerifiedOwnershipUse::Move)
                .map(|(position, _)| position);
            let moved = positions.next();
            if positions.next().is_some() || matches!(moved, Some(position) if position != 0) {
                return Err(invalid_abi_shape(
                    format!(
                        "verified `{label}` application moves an operand this emitter \
                         does not consume; it consumes operand 0 only"
                    ),
                    "verified C host ownership emission",
                ));
            }
            if found.replace(moved.is_some()).is_some() {
                return Err(invalid_abi_shape(
                    format!("verified builtin site carries two `{label}` applications"),
                    "verified C host ownership emission",
                ));
            }
        }
    }
    found.ok_or_else(|| {
        invalid_abi_shape(
            format!("verified builtin site carries no `{label}` application"),
            "verified C host ownership emission",
        )
    })
}

/// Resolve the one verifier-authorized direct call at a host `Call` site.
///
/// A diagnostic label or an intrinsic/indirect application cannot authorize
/// consuming-call terminal motion. Missing or duplicated structural call
/// identity therefore fails closed before the emitter writes any pre-call
/// action.
pub(crate) fn direct_call_action_index(site: &ProjectedHostSite<'_>) -> Result<usize, Unsupported> {
    let mut direct_call = None;
    for (index, action) in site.directives.iter().enumerate() {
        if matches!(
            action,
            VerifiedHostAction::Operation(VerifiedHostOperation::Apply {
                kind: VerifiedApplyKind::DirectCall { .. },
                ..
            })
        ) && direct_call.replace(index).is_some()
        {
            return Err(invalid_abi_shape(
                "verified user-function call site contains multiple direct-call authorities"
                    .to_string(),
                "verified C host ownership emission",
            ));
        }
    }
    direct_call.ok_or_else(|| {
        invalid_abi_shape(
            "verified user-function call site has no direct-call authority".to_string(),
            "verified C host ownership emission",
        )
    })
}

impl<'a> HostEmitter<'a> {
    fn new(
        indent: String,
        helper_prefix: &str,
        emitted_names: UnordMap<String, String>,
        function_specializations: UnordMap<String, HostFunctionSpecialization>,
        tensor_helpers: HostTensorHelpers<'a>,
        ownership_sites: &[ProjectedHostSite<'a>],
    ) -> Self {
        Self {
            lines: Vec::new(),
            indent,
            helper_prefix: helper_prefix.to_string(),
            emitted_names,
            function_specializations,
            tensor_helpers: tensor_helpers.helpers,
            tensor_helper_output_counts: tensor_helpers.output_counts,
            tensor_helper_result_origins: tensor_helpers.result_origins,
            expression_sites: ownership_sites
                .iter()
                .filter(|site| site.kind == chelis_ir::ownership::HostSiteKind::Expression)
                .cloned()
                .collect(),
            expression_site_index: 0,
            entry_projection: entry::Projection::default(),
            external_helpers: UnordSet::new(),
            #[cfg(feature = "native-random-observer")]
            source_sites: Vec::new(),
            pre_emitted_clone_sites: UnordSet::new(),
            pre_emitted_terminals: UnordSet::new(),
            owner_vars: UnordMap::new(),
            temp_counter: 0,
            result_claims: None,
            claim_on_spine: false,
        }
    }

    /// chelis#2120: fill a freshly allocated tensor with a `[05-OP-8]`
    /// uniform draw in the C HOST lane.
    ///
    /// The tensor-DAG lane has its own arm (`emit::emit_uniform_like`) and
    /// bakes the handler seed into the kernel. The host lane cannot: its
    /// seed lives in `__chelis_rng`, installed by `HostExprKind::WithSeed`,
    /// and the draw ordinal is consumed at run time by
    /// `chelis_effective_uniform_seed`. This is the FIRST host-lane ordinal
    /// consumer, so the two rules below are what keep it in step with
    /// `chelis eval` (`chelis-compiler-api` `runtime/eval.rs` `"uniform_like"`):
    ///
    /// 1. **Exactly one ordinal per application, read after the arguments.**
    ///    `arg_vars` are already emitted when this runs, matching the
    ///    evaluator's left-to-right argument evaluation followed by its
    ///    `random_counter` read. `CHELIS_EFFECTIVE_UNIFORM_SEED` is invoked
    ///    once, into a temporary, and never inside the element loop.
    /// 2. **Bounds are re-folded from the structural `args`, not read from
    ///    `arg_vars`.** The checker already guarantees static literal bounds
    ///    (`infer::app_operand_dtype`, the chelis#776 gate), and this bakes
    ///    the same exact bit pattern the DAG lane bakes, so a template that
    ///    folds and one that does not sample identically.
    ///
    ///    This fold is value-based and does NOT model an intermediate
    ///    rounding: a bound spelled `cast(cast(x, f16), f32)` type-checks,
    ///    and the evaluator applies both roundings while `static_float_bound`
    ///    applies neither. The DAG lane's `lower::extract_f64_value` has the
    ///    identical behavior, so the two compiled lanes agree with each other
    ///    and both differ from `eval` on that spelling. The CLASS is
    ///    pre-existing and owned outside this change (chelis#2316); the
    ///    host-lane INSTANCE is new, because this lane previously refused to
    ///    build at all. Reproducing the DAG lane's exact behavior is
    ///    deliberate: correcting one lane alone would make template
    ///    foldability observable again, which is the defect chelis#2120
    ///    exists to remove.
    ///
    /// The template's element values are never read — only its shape and
    /// dtype reach the output through `chelis_host_alloc_like` — which is
    /// [05-OP-8]'s "the template values are not observed" made structural
    /// rather than incidental.
    ///
    /// Every active float dtype gets an arm. A runtime-derived template is
    /// exactly the case the DAG lane does NOT serve, so leaving f16/bf16 on
    /// the default arm would have built a program that aborts at run time
    /// where the previous rejection at least failed at build time.
    fn assign_uniform_like(
        &mut self,
        target: &str,
        args: &[HostExpr],
        arg_vars: &[(String, HostType)],
    ) -> Result<(), Unsupported> {
        let template = match arg_vars.first() {
            Some((var, HostType::Tensor(_))) => var.clone(),
            _ => {
                return Err(Unsupported::new(
                    UnsupportedKind::Builtin("uniform_like".to_string()),
                    "`chelis build` host emission",
                    Stage::Codegen("c"),
                    chelis_types::deliberate_rejection!(
                        "[04-TOT-2]",
                        "uniform_like's first operand must be a tensor template; the checker \
                         types it as one, so a non-tensor here is an internal desync"
                    ),
                ));
            }
        };
        let low = static_float_bound(args.get(1)).ok_or_else(|| unresolved_uniform_bound("low"))?;
        let high =
            static_float_bound(args.get(2)).ok_or_else(|| unresolved_uniform_bound("high"))?;

        // [05-OP-8] / chelis#248: the sampler sees the byte-identical f32
        // narrowing of each source bound, not a decimal round-trip. The f64
        // arm widens those SAME truncated images, exactly as
        // `emit::emit_uniform_like` and `host_ops::uniform_like_value` do.
        let low_f32 = low as f32;
        let high_f32 = high as f32;
        let low_f32_expr = format!(
            "chelis_f32_from_bits(UINT32_C(0x{:08x}))",
            low_f32.to_bits()
        );
        let high_f32_expr = format!(
            "chelis_f32_from_bits(UINT32_C(0x{:08x}))",
            high_f32.to_bits()
        );
        let low_f64_expr = format!(
            "chelis_f64_from_bits(UINT64_C(0x{:016x}))",
            f64::from(low_f32).to_bits()
        );
        let high_f64_expr = format!(
            "chelis_f64_from_bits(UINT64_C(0x{:016x}))",
            f64::from(high_f32).to_bits()
        );

        // An inactive scope is reachable, not an internal desync: a
        // top-level binding with an unhandled `Random` is a hard check error,
        // but an exported `def` carrying one is not, and its generated
        // wrapper initializes `__chelis_rng` inactive. Abort there rather
        // than let `chelis_effective_uniform_seed` silently return the baked
        // operand without advancing the counter, which would both return the
        // wrong value and desync every later draw in the scope.
        let seed = self.next_temp("uniform_seed");
        self.lines.push(format!(
            "{}if (__chelis_rng == NULL || !__chelis_rng->active) {{",
            self.indent
        ));
        self.lines.push(format!(
            "{}    fprintf(stderr, \"uniform_like requires an active host RNG scope\\n\");",
            self.indent
        ));
        self.lines.push(format!("{}    abort();", self.indent));
        self.lines.push(format!("{}}}", self.indent));
        self.lines.push(format!(
            "{}uint64_t {seed} = CHELIS_EFFECTIVE_UNIFORM_SEED(0ULL);",
            self.indent
        ));

        self.lines.push(format!(
            "{}{target} = chelis_host_alloc_like({template}, chelis_host_tensor_dtype({template}));",
            self.indent
        ));
        let (guard, view) = self.begin_tensor_write(target);
        self.lines
            .push(format!("{}switch ({view}.dtype) {{", self.indent));
        // One arm per active float dtype in [05-OP-8], which "admits every
        // active float template dtype `p`". f32 and f64 sample at their own
        // width. f16 and bf16 widen the stored bounds to f32, execute the one
        // fused multiply-add in f32, and narrow exactly once at the store —
        // the same shape `emit::emit_uniform_like` gives the DAG lane, so a
        // template that folds and one that does not agree element for
        // element. `chelis_f32_to_f16`/`_bf16` are `static inline` in
        // `chelis_runtime.h`, which every emitted translation unit includes.
        let sample_f32 = format!(
            "chelis_uniform_sample_f32({seed}, (uint64_t)i, {low_f32_expr}, {high_f32_expr})"
        );
        for (macro_name, elem_t, sampled) in [
            (
                chelis_vocab::RuntimeDType::F32.c_macro(),
                "float",
                sample_f32.clone(),
            ),
            (
                chelis_vocab::RuntimeDType::F64.c_macro(),
                "double",
                format!(
                    "chelis_uniform_sample_f64({seed}, (uint64_t)i, {low_f64_expr}, {high_f64_expr})"
                ),
            ),
            (
                chelis_vocab::RuntimeDType::F16.c_macro(),
                "uint16_t",
                format!("chelis_f32_to_f16({sample_f32})"),
            ),
            (
                chelis_vocab::RuntimeDType::Bf16.c_macro(),
                "uint16_t",
                format!("chelis_f32_to_bf16({sample_f32})"),
            ),
        ] {
            let ind = &self.indent;
            self.lines.push(format!("{ind}    case {macro_name}: {{"));
            self.lines.push(format!(
                "{ind}        {elem_t} *__target_data = ({elem_t}*){view}.data;"
            ));
            self.lines.push(format!(
                "{ind}        for (int64_t i = 0; i < {view}.count; i++) {{"
            ));
            self.lines
                .push(format!("{ind}            __target_data[i] = {sampled};"));
            self.lines.push(format!("{ind}        }}"));
            self.lines.push(format!("{ind}        break;"));
            self.lines.push(format!("{ind}    }}"));
        }
        self.emit_default_runtime_fail_arm_for(&format!("{view}.dtype"), "uniform_like");
        self.lines.push(format!("{}}}", self.indent));
        self.end_tensor_write(&guard);
        Ok(())
    }

    fn begin_tensor_write(&mut self, tensor: &str) -> (String, String) {
        let guard = self.next_temp("tensor_write_guard");
        let view = self.next_temp("tensor_write_view");
        self.lines.push(format!(
            "{}chelis_tensor_write *{guard} = chelis_tensor_begin_write({tensor});",
            self.indent
        ));
        self.lines.push(format!(
            "{}chelis_write_view {view} = chelis_tensor_write_view({guard});",
            self.indent
        ));
        (guard, view)
    }

    fn end_tensor_write(&mut self, guard: &str) {
        self.lines
            .push(format!("{}chelis_tensor_end_write({guard});", self.indent));
    }

    fn emit_expr_to_var(
        &mut self,
        expr: &HostExpr,
        target: &str,
        ty: &HostType,
    ) -> Result<(), Unsupported> {
        self.lines
            .push(format!("{}{};", self.indent, c_decl(ty, target)?));
        if matches!(ty, HostType::Tensor(_)) {
            self.lines.push(format!(
                "{}__chelis_host_result_origin {} = {{ NULL, NULL }};",
                self.indent,
                result_origin_name(target)
            ));
        }
        self.assign_expr(target, expr, ty)?;
        Ok(())
    }

    fn declare_result_origin(&mut self, value: &str, ty: &HostType, producer: Option<&str>) {
        if !matches!(ty, HostType::Tensor(_)) {
            return;
        }
        let origin = result_origin_name(value);
        let initializer = producer.map_or_else(
            || "{ NULL, NULL }".to_string(),
            |op| format!("{{ \"{op}\", \"numeric trap: domain in {op} at i64\" }}"),
        );
        self.lines.push(format!(
            "{}__chelis_host_result_origin {origin} = {initializer};",
            self.indent
        ));
    }

    fn next_expression_site(&mut self) -> Result<ProjectedHostSite<'a>, Unsupported> {
        let Some(site) = self
            .expression_sites
            .get(self.expression_site_index)
            .cloned()
        else {
            return Err(invalid_abi_shape(
                "verified ownership expression-site cursor exhausted before host payload"
                    .to_string(),
                "verified C host ownership emission",
            ));
        };
        self.expression_site_index += 1;
        Ok(site)
    }

    fn emit_expression_site(
        &mut self,
        site: &ProjectedHostSite<'a>,
        target: &str,
    ) -> Result<(), Unsupported> {
        self.emit_expression_site_excluding_block(site, target, None)
    }

    fn emit_expression_site_excluding_block(
        &mut self,
        site: &ProjectedHostSite<'a>,
        target: &str,
        excluded: Option<VerifiedBlockId>,
    ) -> Result<(), Unsupported> {
        let excluded = excluded.map_or_else(Vec::new, |block| vec![block]);
        self.emit_expression_site_excluding_blocks(site, target, &excluded)
    }

    fn emit_expression_site_excluding_blocks(
        &mut self,
        site: &ProjectedHostSite<'a>,
        target: &str,
        excluded: &[VerifiedBlockId],
    ) -> Result<(), Unsupported> {
        if site.actions.len() != site.directives.len() {
            return Err(invalid_abi_shape(
                format!(
                    "verified expression site carries {} classifications for {} directives",
                    site.actions.len(),
                    site.directives.len()
                ),
                "verified C host ownership emission",
            ));
        }
        for action in &site.directives {
            if excluded
                .iter()
                .any(|block| Self::action_is_in_block(action, *block))
            {
                continue;
            }
            match action {
                VerifiedHostAction::Operation(VerifiedHostOperation::Define { dest, .. }) => {
                    self.owner_vars.insert(dest.id(), target.to_string());
                }
                VerifiedHostAction::Operation(VerifiedHostOperation::Apply {
                    dest: Some(dest),
                    binding_name,
                    label,
                    ..
                }) => {
                    let value = binding_name
                        .as_ref()
                        .filter(|_| {
                            label.starts_with("option_payload")
                                || label.starts_with("adt_payload")
                                || *label == "loop_item"
                        })
                        .map_or_else(
                            || target.to_string(),
                            |name| c_ident(name.as_str()).into_owned(),
                        );
                    self.owner_vars.insert(dest.id(), value);
                }
                VerifiedHostAction::Operation(VerifiedHostOperation::Discard { .. }) => {}
                VerifiedHostAction::Operation(VerifiedHostOperation::Apply {
                    dest: None,
                    label: "builtin:copy" | "builtin:debug",
                    ..
                }) => {}
                VerifiedHostAction::Operation(VerifiedHostOperation::Clone {
                    dest,
                    source,
                    ..
                }) if !self.pre_emitted_clone_sites.contains(&site.id) => {
                    self.emit_clone_to(*dest, *source, None)?;
                }
                VerifiedHostAction::Operation(VerifiedHostOperation::Clone { .. }) => {}
                VerifiedHostAction::Operation(VerifiedHostOperation::Project {
                    source, ..
                }) => {
                    self.owner_vars
                        .insert(source.owner().id(), target.to_string());
                }
                VerifiedHostAction::Operation(VerifiedHostOperation::LoopItem { .. }) => {}
                VerifiedHostAction::Operation(VerifiedHostOperation::Drop {
                    operation,
                    owner,
                    ..
                }) if !self.pre_emitted_terminals.contains(&(site.id, *operation)) => {
                    self.emit_owner_drop(owner.owner())?;
                }
                VerifiedHostAction::Operation(VerifiedHostOperation::Drop { .. }) => {}
                VerifiedHostAction::Terminator(VerifiedHostTerminator::Jump { edge, .. }) => {
                    if edge.params().len() == 1 {
                        self.owner_vars
                            .insert(edge.params()[0].id(), target.to_string());
                    }
                }
                VerifiedHostAction::Terminator(VerifiedHostTerminator::Loop {
                    body_edge,
                    exit_edge,
                    ..
                }) => {
                    if body_edge.params().len() == 2 && exit_edge.params().len() == 2 {
                        // Scan carries `(state, output)` and binds those two
                        // payloads to distinct C variables in `assign_scan`.
                        continue;
                    }
                    for (edge, name) in [(body_edge, "body"), (exit_edge, "exit")] {
                        let [param] = edge.params() else {
                            return Err(invalid_abi_shape(
                                format!(
                                    "verified loop {name} edge carries {} parameters, expected one owned payload",
                                    edge.params().len()
                                ),
                                "verified C host ownership emission",
                            ));
                        };
                        self.owner_vars.insert(param.id(), target.to_string());
                    }
                }
                VerifiedHostAction::Terminator(VerifiedHostTerminator::Branch { .. })
                | VerifiedHostAction::Terminator(VerifiedHostTerminator::Match { .. }) => {}
                other => {
                    return Err(invalid_abi_shape(
                        format!("unexpected verified expression-site action {other:?}"),
                        "verified C host ownership emission",
                    ));
                }
            }
        }
        Ok(())
    }

    /// Emit argument-owner clones before the consuming call they feed.
    ///
    /// Ownership lowering records a call site's operand `Clone` operations
    /// before its result-producing `Apply`. Formatting necessarily visits the
    /// argument payloads first; this hook preserves that verified ordering
    /// instead of retaining an argument after the owned callee has consumed it.
    fn emit_pre_call_actions(&mut self, site: &ProjectedHostSite<'a>) -> Result<(), Unsupported> {
        let direct_call_index = direct_call_action_index(site)?;
        if !self.pre_emitted_clone_sites.insert(site.id) {
            return Err(invalid_abi_shape(
                "verified call site emitted its pre-call actions twice".to_string(),
                "verified C host ownership emission",
            ));
        }
        for (index, action) in site.directives.iter().enumerate() {
            match action {
                VerifiedHostAction::Operation(VerifiedHostOperation::Clone {
                    dest,
                    source,
                    ..
                }) => self.emit_clone_to(*dest, *source, None)?,
                VerifiedHostAction::Operation(VerifiedHostOperation::Drop {
                    operation,
                    owner,
                    ..
                }) if index < direct_call_index => {
                    self.emit_owner_drop(owner.owner())?;
                    self.pre_emitted_terminals.insert((site.id, *operation));
                }
                VerifiedHostAction::Operation(VerifiedHostOperation::Discard {
                    operation, ..
                }) if index < direct_call_index => {
                    self.pre_emitted_terminals.insert((site.id, *operation));
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn action_is_in_block(action: &VerifiedHostAction<'_>, expected: VerifiedBlockId) -> bool {
        let block = match action {
            VerifiedHostAction::Operation(VerifiedHostOperation::Define { block, .. })
            | VerifiedHostAction::Operation(VerifiedHostOperation::Apply { block, .. })
            | VerifiedHostAction::Operation(VerifiedHostOperation::Clone { block, .. })
            | VerifiedHostAction::Operation(VerifiedHostOperation::Project { block, .. })
            | VerifiedHostAction::Operation(VerifiedHostOperation::LoopItem { block, .. })
            | VerifiedHostAction::Operation(VerifiedHostOperation::Drop { block, .. })
            | VerifiedHostAction::Operation(VerifiedHostOperation::Discard { block, .. })
            | VerifiedHostAction::Operation(VerifiedHostOperation::RootConsume { block, .. })
            | VerifiedHostAction::Terminator(VerifiedHostTerminator::Return { block, .. })
            | VerifiedHostAction::Terminator(VerifiedHostTerminator::Jump { block, .. })
            | VerifiedHostAction::Terminator(VerifiedHostTerminator::Branch { block, .. })
            | VerifiedHostAction::Terminator(VerifiedHostTerminator::Match { block, .. })
            | VerifiedHostAction::Terminator(VerifiedHostTerminator::Loop { block, .. })
            | VerifiedHostAction::Terminator(VerifiedHostTerminator::Exit { block }) => *block,
            VerifiedHostAction::ControlEdge { source, .. } => *source,
            VerifiedHostAction::ManifestRoot { .. } => return false,
        };
        block == expected
    }

    fn join_completion_blocks(
        site: &ProjectedHostSite<'a>,
        expected: usize,
        context: &str,
    ) -> Result<Vec<VerifiedBlockId>, Unsupported> {
        let blocks = site
            .directives
            .iter()
            .filter_map(|action| match action {
                VerifiedHostAction::Terminator(VerifiedHostTerminator::Jump { block, .. }) => {
                    Some(*block)
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        if blocks.len() != expected {
            return Err(invalid_abi_shape(
                format!(
                    "verified {context} site has {} arm completions, expected {expected}",
                    blocks.len()
                ),
                "verified C host ownership emission",
            ));
        }
        Ok(blocks)
    }

    fn emit_expression_block_actions(
        &mut self,
        site: &ProjectedHostSite<'a>,
        block: VerifiedBlockId,
        target: &str,
    ) -> Result<(), Unsupported> {
        for action in &site.directives {
            match action {
                VerifiedHostAction::Operation(VerifiedHostOperation::Define {
                    block: owner_block,
                    dest,
                    ..
                }) if *owner_block == block => {
                    self.owner_vars.insert(dest.id(), target.to_string());
                }
                VerifiedHostAction::Operation(VerifiedHostOperation::Apply {
                    block: owner_block,
                    dest: Some(dest),
                    binding_name,
                    label,
                    ..
                }) if *owner_block == block => {
                    let value = binding_name
                        .as_ref()
                        .filter(|_| {
                            label.starts_with("option_payload")
                                || label.starts_with("adt_payload")
                                || *label == "loop_item"
                        })
                        .map_or_else(
                            || target.to_string(),
                            |name| c_ident(name.as_str()).into_owned(),
                        );
                    self.owner_vars.insert(dest.id(), value);
                }
                VerifiedHostAction::Operation(VerifiedHostOperation::Discard {
                    operation,
                    block: owner_block,
                    ..
                }) if *owner_block == block
                    && !self.pre_emitted_terminals.contains(&(site.id, *operation)) =>
                {
                    self.pre_emitted_terminals.insert((site.id, *operation));
                }
                VerifiedHostAction::Operation(VerifiedHostOperation::Discard { .. }) => {}
                VerifiedHostAction::Operation(VerifiedHostOperation::Clone {
                    block: owner_block,
                    dest,
                    source,
                    ..
                }) if *owner_block == block => self.emit_clone_to(*dest, *source, None)?,
                VerifiedHostAction::Operation(VerifiedHostOperation::Project {
                    block: owner_block,
                    source,
                    ..
                }) if *owner_block == block => {
                    self.owner_vars
                        .insert(source.owner().id(), target.to_string());
                }
                VerifiedHostAction::Operation(VerifiedHostOperation::LoopItem { .. }) => {}
                VerifiedHostAction::Operation(VerifiedHostOperation::Drop {
                    operation,
                    block: owner_block,
                    owner,
                    ..
                }) if *owner_block == block
                    && !self.pre_emitted_terminals.contains(&(site.id, *operation)) =>
                {
                    self.emit_owner_drop(owner.owner())?;
                    self.pre_emitted_terminals.insert((site.id, *operation));
                }
                VerifiedHostAction::Operation(VerifiedHostOperation::Drop { .. }) => {}
                VerifiedHostAction::Terminator(VerifiedHostTerminator::Jump {
                    block: owner_block,
                    edge,
                }) if *owner_block == block && edge.params().len() == 1 => {
                    self.owner_vars
                        .insert(edge.params()[0].id(), target.to_string());
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Emit one verified owner's scheduled terminal in a physical block.
    ///
    /// A nested source expression can move control to child blocks before the
    /// parent site's completion jump.  Its parent-owned terminals still need
    /// to run after that child expression on the selected path, even when the
    /// completion block differs from the arm's entry block.
    fn emit_expression_block_terminal_for_owner(
        &mut self,
        site: &ProjectedHostSite<'a>,
        block: VerifiedBlockId,
        expected_owner: VerifiedOwnerId,
    ) -> Result<(), Unsupported> {
        for action in &site.directives {
            match action {
                VerifiedHostAction::Operation(VerifiedHostOperation::Drop {
                    operation,
                    block: owner_block,
                    owner,
                    ..
                }) if *owner_block == block
                    && owner.owner().id() == expected_owner
                    && !self.pre_emitted_terminals.contains(&(site.id, *operation)) =>
                {
                    self.emit_owner_drop(owner.owner())?;
                    self.pre_emitted_terminals.insert((site.id, *operation));
                }
                VerifiedHostAction::Operation(VerifiedHostOperation::Discard {
                    operation,
                    block: owner_block,
                    owner,
                    ..
                }) if *owner_block == block && owner.id() == expected_owner => {
                    self.pre_emitted_terminals.insert((site.id, *operation));
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn bind_loop_item(
        &mut self,
        site: &ProjectedHostSite<'a>,
        emitted_var: &str,
    ) -> Result<(), Unsupported> {
        let mut items = site.directives.iter().filter_map(|action| match action {
            VerifiedHostAction::Operation(VerifiedHostOperation::LoopItem { dest, .. }) => {
                Some(*dest)
            }
            _ => None,
        });
        let Some(owner) = items.next() else {
            return Err(invalid_abi_shape(
                "verified list-loop site has no typed loop-item action".to_string(),
                "verified C host ownership emission",
            ));
        };
        if items.next().is_some() {
            return Err(invalid_abi_shape(
                "verified list-loop site has more than one typed loop-item action".to_string(),
                "verified C host ownership emission",
            ));
        }
        self.owner_vars.insert(owner.id(), emitted_var.to_string());
        Ok(())
    }

    /// Bind a verified pattern-payload owner to the C variable that holds the
    /// independently retained value extracted for that pattern.
    ///
    /// Payload owners can acquire further logical names while lowering nested
    /// lets.  Falling back to the last such name can point a later arm-terminal
    /// release at a C temporary whose lexical scope has already closed, so the
    /// match formatter consumes the exact entry-block payload action instead.
    fn bind_match_payload(
        &mut self,
        site: &ProjectedHostSite<'a>,
        block: VerifiedBlockId,
        label_prefix: &str,
        binding_name: &str,
    ) -> Result<(), Unsupported> {
        let mut owners = site.directives.iter().filter_map(|action| match action {
            VerifiedHostAction::Operation(VerifiedHostOperation::Apply {
                block: owner_block,
                dest: Some(dest),
                binding_name: projected_name,
                label,
                ..
            }) if *owner_block == block
                && label.starts_with(label_prefix)
                && projected_name
                    .as_ref()
                    .is_some_and(|name| name.as_str() == binding_name) =>
            {
                Some(*dest)
            }
            _ => None,
        });
        let Some(owner) = owners.next() else {
            return Err(invalid_abi_shape(
                format!(
                    "verified match block {:?} has no `{label_prefix}` payload owner for `{binding_name}`",
                    block
                ),
                "verified C host ownership emission",
            ));
        };
        if owners.next().is_some() {
            return Err(invalid_abi_shape(
                format!(
                    "verified match block {:?} has duplicate `{label_prefix}` payload owners for `{binding_name}`",
                    block
                ),
                "verified C host ownership emission",
            ));
        }
        self.owner_vars
            .insert(owner.id(), c_ident(binding_name).into_owned());
        Ok(())
    }

    /// Return the loop preheader and the block that completes one body
    /// iteration.
    ///
    /// The body entry is not necessarily its completion block: an inline
    /// callback may introduce branch/match blocks before the back-edge. Scope
    /// terminals recorded on the enclosing loop site belong immediately
    /// before that back-edge, while the callback locals are still in C scope.
    fn loop_blocks(
        site: &ProjectedHostSite<'a>,
    ) -> Result<(VerifiedBlockId, VerifiedBlockId), Unsupported> {
        let mut headers = site.directives.iter().filter_map(|action| match action {
            VerifiedHostAction::Terminator(VerifiedHostTerminator::Loop { block, .. }) => {
                Some(*block)
            }
            _ => None,
        });
        let Some(header) = headers.next() else {
            return Err(invalid_abi_shape(
                "verified list-loop site has no loop terminator".to_string(),
                "verified C host ownership emission",
            ));
        };
        if headers.next().is_some() {
            return Err(invalid_abi_shape(
                "verified list-loop site has more than one loop terminator".to_string(),
                "verified C host ownership emission",
            ));
        }

        let jumps =
            site.directives
                .iter()
                .filter_map(|action| match action {
                    VerifiedHostAction::Terminator(VerifiedHostTerminator::Jump {
                        block,
                        edge,
                    }) if edge.target() == header => Some(*block),
                    _ => None,
                })
                .collect::<Vec<_>>();
        let [preheader, completion] = jumps.as_slice() else {
            return Err(invalid_abi_shape(
                format!(
                    "verified list-loop site has {} jumps to its header, expected preheader and back-edge",
                    jumps.len()
                ),
                "verified C host ownership emission",
            ));
        };
        Ok((*preheader, *completion))
    }

    fn loop_edges(
        site: &ProjectedHostSite<'a>,
    ) -> Result<(VerifiedEdgeView<'a>, VerifiedEdgeView<'a>), Unsupported> {
        site.directives
            .iter()
            .find_map(|action| match action {
                VerifiedHostAction::Terminator(VerifiedHostTerminator::Loop {
                    body_edge,
                    exit_edge,
                    ..
                }) => Some((body_edge.clone(), exit_edge.clone())),
                _ => None,
            })
            .ok_or_else(|| {
                invalid_abi_shape(
                    "verified list-loop site has no loop terminator".to_string(),
                    "verified C host ownership emission",
                )
            })
    }

    fn finish_expression_sites(&self) -> Result<(), Unsupported> {
        if self.expression_site_index == self.expression_sites.len() {
            Ok(())
        } else {
            Err(invalid_abi_shape(
                format!(
                    "verified ownership expression-site cursor left {} of {} sites unconsumed",
                    self.expression_sites.len() - self.expression_site_index,
                    self.expression_sites.len()
                ),
                "verified C host ownership emission",
            ))
        }
    }

    fn owner_var(
        &self,
        owner: chelis_ir::ownership::VerifiedOwnerView<'_>,
    ) -> Result<String, Unsupported> {
        self.owner_vars.get(&owner.id()).cloned().ok_or_else(|| {
            invalid_abi_shape(
                format!(
                    "verified ownership action names emitted owner {:?} with no value binding",
                    owner.id()
                ),
                "verified C host ownership emission",
            )
        })
    }

    fn owner_abi_type(
        owner: chelis_ir::ownership::VerifiedOwnerView<'_>,
    ) -> Result<HostType, Unsupported> {
        HostAbiType::try_from_concrete(owner.ty())
    }

    fn emit_owner_drop(
        &mut self,
        owner: chelis_ir::ownership::VerifiedOwnerView<'_>,
    ) -> Result<(), Unsupported> {
        let var = self.owner_var(owner)?;
        let ty = Self::owner_abi_type(owner)?;
        if let Some(release) = release_call(&var, &ty) {
            self.lines.push(format!("{}{release}", self.indent));
        }
        Ok(())
    }

    fn emit_entry_terminals(
        &mut self,
        site: &ProjectedHostSite<'a>,
        authored: bool,
    ) -> Result<(), Unsupported> {
        // The ABI wrapper implements the authored entry's clones and jump.
        // Its target owners are bound to the body parameters above. Emit each
        // projected terminal once here: an edge terminal and its Operation
        // projection have the same identity, not two independent releases.
        // Internal specializations have no adapter edge, but can have drops
        // scheduled directly at block entry.
        for action in &site.directives {
            match action {
                VerifiedHostAction::Operation(VerifiedHostOperation::Drop { owner, .. }) => {
                    self.emit_owner_drop(owner.owner())?;
                }
                VerifiedHostAction::Operation(VerifiedHostOperation::Discard { .. }) => {}
                VerifiedHostAction::Operation(VerifiedHostOperation::Clone { .. })
                | VerifiedHostAction::Terminator(VerifiedHostTerminator::Jump { .. })
                    if authored => {}
                other => {
                    return Err(invalid_abi_shape(
                        format!("unexpected verified function-entry action {other:?}"),
                        "verified C host ownership emission",
                    ));
                }
            }
        }
        Ok(())
    }

    fn emit_edge_terminals(
        &mut self,
        site: HostSiteId,
        edge: &VerifiedEdgeView<'_>,
    ) -> Result<(), Unsupported> {
        for terminal in edge.terminals() {
            match terminal {
                VerifiedTerminalView::Drop { owner, .. } => self.emit_owner_drop(*owner)?,
                VerifiedTerminalView::Discard { .. } => {}
            }
            self.pre_emitted_terminals
                .insert((site, terminal.operation()));
        }
        Ok(())
    }

    fn emit_clone_to(
        &mut self,
        dest: chelis_ir::ownership::VerifiedOwnerView<'_>,
        source: chelis_ir::ownership::VerifiedOperandView<'_>,
        target: Option<&str>,
    ) -> Result<(), Unsupported> {
        let source_var = self.owner_var(source.owner())?;
        let dest_var = target.map_or_else(|| source_var.clone(), str::to_string);
        let ty = Self::owner_abi_type(dest)?;
        if let Some(retain) = retain_call(&dest_var, &ty) {
            self.lines.push(format!("{}{retain}", self.indent));
        }
        self.owner_vars.insert(dest.id(), dest_var);
        Ok(())
    }

    fn emit_terminal_site(
        &mut self,
        site: &ProjectedHostSite<'a>,
        result_target: Option<&str>,
    ) -> Result<(), Unsupported> {
        for action in &site.directives {
            match action {
                VerifiedHostAction::Operation(VerifiedHostOperation::Clone {
                    dest,
                    source,
                    ..
                }) => self.emit_clone_to(*dest, *source, result_target)?,
                VerifiedHostAction::Operation(VerifiedHostOperation::Drop { owner, .. }) => {
                    self.emit_owner_drop(owner.owner())?;
                }
                VerifiedHostAction::Operation(VerifiedHostOperation::Discard { .. }) => {}
                VerifiedHostAction::Terminator(VerifiedHostTerminator::Return { .. })
                | VerifiedHostAction::Terminator(VerifiedHostTerminator::Exit { .. }) => {}
                other => {
                    return Err(invalid_abi_shape(
                        format!("unexpected verified terminal-site action {other:?}"),
                        "verified C host ownership emission",
                    ));
                }
            }
        }
        Ok(())
    }

    fn prepare_manifest_root(&mut self, site: &ProjectedHostSite<'a>) -> Result<(), Unsupported> {
        let mut observed = false;
        let mut terminal_started = false;
        for action in &site.directives {
            match action {
                VerifiedHostAction::Operation(VerifiedHostOperation::Clone {
                    dest,
                    source,
                    ..
                }) if !observed && !terminal_started => {
                    self.emit_clone_to(*dest, *source, None)?;
                }
                VerifiedHostAction::Operation(VerifiedHostOperation::Drop {
                    operation,
                    owner,
                    ..
                }) if !observed => {
                    terminal_started = true;
                    self.emit_owner_drop(owner.owner())?;
                    self.pre_emitted_terminals.insert((site.id, *operation));
                }
                VerifiedHostAction::Operation(VerifiedHostOperation::Discard {
                    operation, ..
                }) if !observed => {
                    terminal_started = true;
                    self.pre_emitted_terminals.insert((site.id, *operation));
                }
                VerifiedHostAction::ManifestRoot { .. } if !observed => observed = true,
                VerifiedHostAction::Operation(VerifiedHostOperation::RootConsume { .. })
                    if observed => {}
                other => {
                    return Err(invalid_abi_shape(
                        format!("out-of-order verified manifest-root action {other:?}"),
                        "verified C host ownership emission",
                    ));
                }
            }
        }
        if observed {
            Ok(())
        } else {
            Err(invalid_abi_shape(
                "verified manifest-root site has no observation marker".to_string(),
                "verified C host ownership emission",
            ))
        }
    }

    fn finish_manifest_root(&mut self, site: &ProjectedHostSite<'a>) -> Result<(), Unsupported> {
        let mut consumed = 0usize;
        let mut observed = false;
        let mut terminal_started = false;
        for action in &site.directives {
            match action {
                VerifiedHostAction::Operation(VerifiedHostOperation::Clone { .. })
                    if !observed && !terminal_started => {}
                VerifiedHostAction::Operation(VerifiedHostOperation::Drop {
                    operation, ..
                })
                | VerifiedHostAction::Operation(VerifiedHostOperation::Discard {
                    operation, ..
                }) if !observed && self.pre_emitted_terminals.contains(&(site.id, *operation)) => {
                    terminal_started = true;
                }
                VerifiedHostAction::ManifestRoot { .. } if !observed => observed = true,
                VerifiedHostAction::Operation(VerifiedHostOperation::RootConsume {
                    owner, ..
                }) if observed => {
                    consumed += 1;
                    let var = self.owner_var(owner.owner())?;
                    let ty = Self::owner_abi_type(owner.owner())?;
                    if let Some(release) = release_call(&var, &ty) {
                        self.lines.push(format!("{}{release}", self.indent));
                    }
                }
                other => {
                    return Err(invalid_abi_shape(
                        format!("out-of-order verified manifest-root action {other:?}"),
                        "verified C host ownership emission",
                    ));
                }
            }
        }
        if consumed != 1 {
            return Err(invalid_abi_shape(
                format!("verified manifest-root site has {consumed} consuming actions"),
                "verified C host ownership emission",
            ));
        }
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
        // chelis#1771. Taken here rather than read, so exactly one expression
        // per forwarding step is on the result spine: a nested argument or a
        // sibling binding cannot inherit the flag.
        let on_result_spine = std::mem::take(&mut self.claim_on_spine);
        let site = self.next_expression_site()?;
        #[cfg(feature = "native-random-observer")]
        let call_frame = {
            let source = self
                .source_sites
                .iter()
                .find(|s| s.verified.site().id() == site.id);
            if source.is_some_and(|s| !s.matches(&site, expr)) {
                return Err(invalid_abi_shape(
                    "source identity disagrees with projected site/kind/target".into(),
                    "native Random source identity",
                ));
            }
            let pointer = source.map_or_else(|| "NULL".into(), |s| s.pointer());
            if matches!(
                expr.kind,
                HostExprKind::Call { .. } | HostExprKind::TensorCall { .. }
            ) {
                let name = self.next_temp("source_call");
                self.lines.extend(crate::random_observer::push_source_call(
                    &self.indent,
                    &name,
                    &pointer,
                ));
                Some(name)
            } else {
                None
            }
        };
        self.assign_expr_at_site(target, expr, ty, &site, on_result_spine)?;
        #[cfg(feature = "native-random-observer")]
        if let Some(frame) = call_frame {
            self.lines.push(crate::random_observer::pop_source_call(
                &self.indent,
                &frame,
            ));
        }
        Ok(())
    }

    fn assign_expr_at_site(
        &mut self,
        target: &str,
        expr: &HostExpr,
        ty: &HostType,
        site: &ProjectedHostSite<'a>,
        on_result_spine: bool,
    ) -> Result<(), Unsupported> {
        self.emit_span_comments(expr);
        let result_claims = on_result_spine
            .then(|| self.result_claims.clone())
            .flatten();
        match &expr.kind {
            HostExprKind::ResultClaimScope {
                plan,
                body,
                ty: scope_ty,
            } => {
                require_same_abi_type(ty, scope_ty, "result-claim scope")?;
                let result = plan.result();
                let axes = self.next_temp("result_claim_axes");
                let frame = self.next_temp("result_claim_frame");
                let parent = result_claims.as_deref().unwrap_or("NULL");
                self.lines
                    .extend(HostResultClaim::from_tensor_type(result).frame_lines(
                        &self.indent,
                        &axes,
                        &frame,
                        parent,
                        None,
                    ));
                let previous_claims = self.result_claims.replace(format!("&{frame}"));
                self.claim_on_spine = true;
                self.assign_expr(target, body, ty)?;
                self.result_claims = previous_claims;
                self.emit_expression_site(site, target)?;
                return Ok(());
            }
            HostExprKind::Int(value) => {
                // The positive magnitude of i64::MIN is not a signed C
                // decimal literal, even when preceded by unary minus.
                let literal = if *value == i64::MIN {
                    "INT64_MIN".to_string()
                } else {
                    value.to_string()
                };
                self.lines
                    .push(format!("{}{target} = {literal};", self.indent));
            }
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
                "{}{target} = {};",
                self.indent,
                runtime_string_literal(value)
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
                    // #379: route the referenced name through `c_ident` so a
                    // binding/param/let spelled like a C keyword resolves to
                    // the same mangled identifier its declaration used.
                    self.lines
                        .push(format!("{}{target} = {};", self.indent, c_ident(name)));
                    if matches!(ty, HostType::Tensor(_)) {
                        self.lines.push(format!(
                            "{}{} = {};",
                            self.indent,
                            result_origin_name(target),
                            result_origin_name(name)
                        ));
                    }
                }
                self.emit_result_claim_guard(target, ty, result_claims.as_deref());
            }
            HostExprKind::Call {
                function,
                args,
                arg_tys,
                ty: call_ty,
            } => {
                require_same_abi_type(ty, call_ty, "call expression")?;
                self.assign_call(
                    (target, call_ty),
                    function,
                    args,
                    arg_tys,
                    site,
                    result_claims.as_deref(),
                )?;
            }
            HostExprKind::Builtin {
                name,
                args,
                ty: expr_ty,
            } => {
                require_same_abi_type(ty, expr_ty, "builtin expression")?;
                self.assign_builtin(target, name, args, ty, site)?;
                self.stamp_result_origin(target, ty, name);
                self.emit_result_claim_guard(target, ty, result_claims.as_deref());
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
                let (then_edge, else_edge) = site
                    .directives
                    .iter()
                    .find_map(|action| match action {
                        VerifiedHostAction::Terminator(VerifiedHostTerminator::Branch {
                            then_edge,
                            else_edge,
                            ..
                        }) => Some((then_edge.clone(), else_edge.clone())),
                        _ => None,
                    })
                    .ok_or_else(|| {
                        invalid_abi_shape(
                            "verified if site has no branch terminator".to_string(),
                            "verified C host ownership emission",
                        )
                    })?;
                let completion = Self::join_completion_blocks(site, 2, "if")?;
                let (then_block, else_block) = (completion[0], completion[1]);
                let cond_var = self.next_temp("cond");
                self.emit_expr_to_var(cond, &cond_var, &HostType::Bool)?;
                self.lines
                    .push(format!("{}if ({cond_var}) {{", self.indent));
                let nested_indent = format!("{}    ", self.indent);
                let previous = std::mem::replace(&mut self.indent, nested_indent);
                self.emit_edge_terminals(site.id, &then_edge)?;
                self.claim_on_spine = on_result_spine;
                self.assign_expr(target, then_expr, ty)?;
                self.emit_expression_block_actions(site, then_block, target)?;
                self.indent = previous.clone();
                self.lines.push(format!("{}}} else {{", self.indent));
                let nested_indent = format!("{}    ", self.indent);
                let previous = std::mem::replace(&mut self.indent, nested_indent);
                self.emit_edge_terminals(site.id, &else_edge)?;
                self.claim_on_spine = on_result_spine;
                self.assign_expr(target, else_expr, ty)?;
                self.emit_expression_block_actions(site, else_block, target)?;
                self.indent = previous;
                self.lines.push(format!("{}}}", self.indent));
                return Ok(());
            }
            HostExprKind::MatchOption {
                scrutinee,
                bind_name,
                some_expr,
                none_expr,
                ty: expr_ty,
            } => {
                require_same_abi_type(ty, expr_ty, "option match")?;
                let (scrutinee_owner, some_edge, none_edge) = site
                    .directives
                    .iter()
                    .find_map(|action| match action {
                        VerifiedHostAction::Terminator(VerifiedHostTerminator::Match {
                            scrutinee,
                            arms,
                            ..
                        }) if arms.len() == 2 => {
                            Some((scrutinee.owner().id(), arms[0].clone(), arms[1].clone()))
                        }
                        _ => None,
                    })
                    .ok_or_else(|| {
                        invalid_abi_shape(
                            "verified option-match site has no two-arm terminator".to_string(),
                            "verified C host ownership emission",
                        )
                    })?;
                let completion = Self::join_completion_blocks(site, 2, "option-match")?;
                let arm_blocks = (completion[0], completion[1]);
                let option_var = self.next_temp("option");
                let option_ty = host_type(scrutinee);
                self.emit_expr_to_var(scrutinee, &option_var, &option_ty)?;
                self.lines.push(format!(
                    "{}if (chelis_option_is_some({option_var})) {{",
                    self.indent
                ));
                let nested_indent = format!("{}    ", self.indent);
                let previous = std::mem::replace(&mut self.indent, nested_indent);
                self.emit_edge_terminals(site.id, &some_edge)?;
                let inner_ty = option_inner_type(&option_ty)?;
                let boxed_inner = self.next_temp("option_value");
                self.lines.push(format!(
                    "{}chelis_value {boxed_inner} = chelis_option_unwrap({option_var});",
                    self.indent
                ));
                match option_ty {
                    HostType::Option(inner) if is_scalar_abi(inner.as_ref()) => {
                        self.lines.push(format!(
                            "{}{} {} = {};",
                            self.indent,
                            c_type(&inner_ty)?,
                            bind_name,
                            scalar_carrier_value_expr(
                                &format!("chelis_value_unbox_scalar({boxed_inner})"),
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
                        self.assign_unboxed_value(bind_name, &inner_ty, &boxed_inner)?;
                        self.declare_result_origin(bind_name, &inner_ty, Some("load"));
                    }
                    other => {
                        return Err(invalid_abi_shape(
                            format!("option match scrutinee has non-option ABI type `{other:?}`"),
                            "option match",
                        ));
                    }
                }
                self.bind_match_payload(site, some_edge.target(), "option_payload", bind_name)?;
                // chelis#1222: the binder shadows any enclosing name it
                // reuses. Its key carries no outgoing edge, because the
                // value is freshly extracted here rather than copied from
                // something this scope already owns -- which is also what
                // keeps emitted C unchanged for every program that does not
                // shadow: a reference to it dead-ends exactly as it does
                // today.
                self.claim_on_spine = on_result_spine;
                self.assign_expr(target, some_expr, ty)?;
                self.emit_expression_block_terminal_for_owner(
                    site,
                    some_edge.target(),
                    scrutinee_owner,
                )?;
                self.emit_expression_block_actions(site, arm_blocks.0, target)?;
                self.indent = previous.clone();
                self.lines.push(format!("{}}} else {{", self.indent));
                let nested_indent = format!("{}    ", self.indent);
                let previous = std::mem::replace(&mut self.indent, nested_indent);
                self.emit_edge_terminals(site.id, &none_edge)?;
                self.claim_on_spine = on_result_spine;
                self.assign_expr(target, none_expr, ty)?;
                self.emit_expression_block_terminal_for_owner(
                    site,
                    none_edge.target(),
                    scrutinee_owner,
                )?;
                self.emit_expression_block_actions(site, arm_blocks.1, target)?;
                self.indent = previous;
                self.lines.push(format!("{}}}", self.indent));
                return Ok(());
            }
            HostExprKind::MatchAdt {
                scrutinee,
                arms,
                default_expr,
                ty: expr_ty,
            } => {
                require_same_abi_type(ty, expr_ty, "ADT match")?;
                self.assign_match_adt(
                    (target, ty),
                    scrutinee,
                    arms,
                    default_expr.as_deref(),
                    site,
                    on_result_spine,
                )?;
                return Ok(());
            }
            HostExprKind::Let {
                bindings,
                body,
                ty: expr_ty,
            } => {
                require_same_abi_type(ty, expr_ty, "let expression")?;
                // chelis#1771: when this let is on the result spine, the guard
                // belongs at the binding whose value the let returns, not at
                // the let's own end, so an effect bound after that binding runs
                // only when the guard passes.
                let spine_binding = on_result_spine
                    .then(|| host_result_binding_index(bindings, body))
                    .flatten();
                self.lines.push(format!("{}{{", self.indent));
                let nested_indent = format!("{}    ", self.indent);
                let previous = std::mem::replace(&mut self.indent, nested_indent);
                for (index, binding) in bindings.iter().enumerate() {
                    // Compute the value into a temp before declaring the binding name.
                    // If the compiler inlines a recursive call that reuses a binding
                    // name from the outer scope (e.g. two nested `let jtj_new = ...`),
                    // declaring the inner name first would shadow the outer variable
                    // before its value is read, yielding a NULL pointer at runtime.
                    let temp = self.next_temp("let");
                    self.claim_on_spine = spine_binding == Some(index);
                    self.emit_expr_to_var(&binding.value, &temp, &binding.ty)?;
                    self.lines.push(format!(
                        "{}{};",
                        self.indent,
                        c_decl(&binding.ty, &binding.name)?
                    ));
                    if matches!(binding.ty, HostType::Tensor(_)) {
                        self.lines.push(format!(
                            "{}__chelis_host_result_origin {};",
                            self.indent,
                            result_origin_name(&binding.name)
                        ));
                    }
                    // #379: assign to the same mangled identifier the
                    // declaration used (both route through `c_ident`).
                    self.lines.push(format!(
                        "{}{} = {};",
                        self.indent,
                        c_ident(&binding.name),
                        temp
                    ));
                    if matches!(binding.ty, HostType::Tensor(_)) {
                        self.lines.push(format!(
                            "{}{} = {};",
                            self.indent,
                            result_origin_name(&binding.name),
                            result_origin_name(&temp)
                        ));
                    }
                }
                self.claim_on_spine = on_result_spine && spine_binding.is_none();
                self.assign_expr(target, body, ty)?;
                self.emit_expression_site(site, target)?;
                self.indent = previous;
                self.lines.push(format!("{}}}", self.indent));
                return Ok(());
            }
            HostExprKind::Map { callback, list, ty } => {
                let (_, body_block) = Self::loop_blocks(site)?;
                self.assign_map(target, callback, list, ty, site, body_block)?;
                self.emit_expression_site_excluding_block(site, target, Some(body_block))?;
                return Ok(());
            }
            HostExprKind::Filter { callback, list, ty } => {
                let (_, body_block) = Self::loop_blocks(site)?;
                self.assign_filter(target, callback, list, ty, site, body_block)?;
                self.emit_expression_site_excluding_block(site, target, Some(body_block))?;
                return Ok(());
            }
            HostExprKind::Fold {
                callback,
                init,
                list,
                ty,
            } => {
                let (preheader_block, body_block) = Self::loop_blocks(site)?;
                self.assign_fold(
                    target,
                    callback,
                    init,
                    list,
                    ty,
                    site,
                    preheader_block,
                    body_block,
                )?;
                self.emit_expression_site_excluding_blocks(
                    site,
                    target,
                    &[preheader_block, body_block],
                )?;
                return Ok(());
            }
            HostExprKind::Scan {
                callback,
                init,
                list,
                ty,
            } => {
                let (_, body_block) = Self::loop_blocks(site)?;
                self.assign_scan(target, callback, init, list, ty, site, body_block)?;
                self.emit_expression_site_excluding_block(site, target, Some(body_block))?;
                return Ok(());
            }
            HostExprKind::Partition { callback, list, ty } => {
                let (_, body_block) = Self::loop_blocks(site)?;
                self.assign_partition(target, callback, list, ty, site, body_block)?;
                self.emit_expression_site_excluding_block(site, target, Some(body_block))?;
                return Ok(());
            }
            HostExprKind::FlatMap { callback, list, ty } => {
                let (_, body_block) = Self::loop_blocks(site)?;
                self.assign_flat_map(target, callback, list, ty, site, body_block)?;
                self.emit_expression_site_excluding_block(site, target, Some(body_block))?;
                return Ok(());
            }
            HostExprKind::WithSeed { seed, body, ty } => {
                let seed_var = self.next_temp("seed");
                self.emit_expr_to_var(seed, &seed_var, &HostType::Int64)?;
                #[cfg(feature = "native-random-observer")]
                let source_frame = {
                    let pointer = self
                        .source_sites
                        .iter()
                        .find(|s| s.verified.site().id() == site.id)
                        .map_or_else(|| "NULL".into(), |s| s.pointer());
                    let frame = self.next_temp("source_host");
                    self.lines.push(format!("{}__chelis_random_host_frame {frame} = __chelis_random_push_host(__chelis_observer, {pointer}, (uint64_t){seed_var});", self.indent));
                    self.lines.push(format!(
                        "{}if (__chelis_observer != NULL) __chelis_observer->host = &{frame};",
                        self.indent
                    ));
                    frame
                };
                let saved_var = self.next_temp("rng_saved");
                let seeded_var = self.next_temp("rng_seeded");
                self.lines.push(format!(
                    "{}chelis_rng_state {saved_var} = *__chelis_rng;",
                    self.indent
                ));
                #[cfg(feature = "native-random-observer")]
                let observer_frame = self.next_temp("rng_observer_frame");
                #[cfg(feature = "native-random-observer")]
                self.lines.extend(crate::random_observer::push_frame(
                    &self.indent,
                    &observer_frame,
                    &saved_var,
                ));
                // Install the complete handler frame, just as exit restores
                // the complete saved frame, before evaluating its body.
                self.lines.push(format!(
                    "{}chelis_rng_state {seeded_var} = {{(uint64_t){seed_var}, 0ULL, 1}};",
                    self.indent
                ));
                self.lines
                    .push(format!("{}*__chelis_rng = {seeded_var};", self.indent));
                #[cfg(feature = "native-random-observer")]
                self.lines.push(crate::random_observer::record(
                    &self.indent,
                    "__CHELIS_RANDOM_OBSERVER_HOST_INSTALL",
                    "__CHELIS_RANDOM_OBSERVER_HOST_IDENTITY_UNSUPPORTED",
                    "NULL",
                    None,
                    None,
                    None,
                    "*__chelis_rng",
                    None,
                ));
                self.assign_expr(target, body, ty)?;
                self.lines
                    .push(format!("{}*__chelis_rng = {saved_var};", self.indent));
                #[cfg(feature = "native-random-observer")]
                {
                    self.lines.push(crate::random_observer::pop_frame(
                        &self.indent,
                        &observer_frame,
                    ));
                    self.lines.push(crate::random_observer::record(
                        &self.indent,
                        "__CHELIS_RANDOM_OBSERVER_HOST_RESTORE",
                        "__CHELIS_RANDOM_OBSERVER_HOST_IDENTITY_UNSUPPORTED",
                        "NULL",
                        None,
                        None,
                        None,
                        "*__chelis_rng",
                        None,
                    ));
                    self.lines.push(format!("{}if (__chelis_observer != NULL) __chelis_observer->host = {source_frame}.previous;", self.indent));
                }
            }
            HostExprKind::TensorCall { helper, args, ty } => {
                let expression = self
                    .expression_sites
                    .iter()
                    .position(|candidate| candidate.id == site.id)
                    .expect("current verified expression site");
                let variant = self
                    .entry_projection
                    .call_variant(expression, *helper)
                    .ok_or_else(|| {
                        invalid_abi_shape(
                            "tensor call lost its signature-entry schedule identity".into(),
                            "signature entry projection",
                        )
                    })?;
                self.assign_tensor_call(
                    (target, ty),
                    *helper,
                    variant,
                    args,
                    result_claims.as_deref(),
                )?;
            }
            HostExprKind::SignatureEntry { plan, args } => {
                require_same_abi_type(ty, &HostType::Unit, "signature entry")?;
                let mut actuals = Vec::with_capacity(args.len());
                for arg in args {
                    let temp = self.next_temp("entry_arg");
                    self.emit_expr_to_var(arg, &temp, &host_type(arg))?;
                    actuals.push(temp);
                }
                self.lines
                    .extend(signature_entry_lines(plan, &actuals, &self.indent, &[])?);
                self.lines.push(format!("{}{target} = 0;", self.indent));
            }
            HostExprKind::Unit => {
                require_same_abi_type(ty, &HostType::Unit, "unit expression")?;
                self.lines.push(format!("{}{target} = 0;", self.indent));
            }
        }
        self.emit_expression_site(site, target)
    }

    fn stamp_result_origin(&mut self, target: &str, ty: &HostType, op: &str) {
        if !matches!(ty, HostType::Tensor(_)) {
            return;
        }
        let op = chelis_ir::span_sanitize::sanitize_for_format_string(op);
        let origin = result_origin_name(target);
        self.lines.push(format!(
            "{}{origin}.op = \"{op}\"; {origin}.trap = \"numeric trap: domain in {op} at i64\";",
            self.indent
        ));
    }

    fn emit_result_claim_guard(&mut self, target: &str, ty: &HostType, claims: Option<&str>) {
        if let Some(claims) = claims
            && matches!(ty, HostType::Tensor(_))
        {
            let origin = result_origin_name(target);
            self.lines
                .push(format!("{}if ({claims} != NULL) {{", self.indent));
            self.lines.push(format!(
                "{}    if ({origin}.op == NULL || {origin}.trap == NULL) {{ fprintf(stderr, \"host runtime: pending result claim reached a tensor without producer provenance\\n\"); abort(); }}",
                self.indent
            ));
            self.lines.push(format!(
                "{}    __chelis_check_host_result_claims({claims}, {target}, {origin}.op, {origin}.trap);",
                self.indent
            ));
            self.lines.push(format!("{}}}", self.indent));
        }
    }

    fn assign_builtin(
        &mut self,
        target: &str,
        name: &str,
        args: &[HostExpr],
        ty: &HostType,
        site: &ProjectedHostSite<'a>,
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
                // The scalar literal is folded into its checked cast without
                // a temporary, but it remains an exact verified expression
                // site and must advance the ownership cursor.
                let literal_site = self.next_expression_site()?;
                self.emit_expression_site(&literal_site, target)?;
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
                self.assign_option_some(target, ty, &arg_vars[0].0, &arg_vars[0].1, &args[0])?;
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
                // chelis#2205: a container the verifier moved at its last use
                // is consumed by the owned entry point; a borrowed one still
                // goes through the cloning call.
                let entry = if container_operand_is_moved(site, "builtin:append")? {
                    "chelis_list_append_owned"
                } else {
                    "chelis_list_append"
                };
                self.lines.push(format!(
                    "{}{target} = {entry}({}, {});",
                    self.indent,
                    arg_vars[0].0,
                    self.box_value_expr(&arg_vars[1].0, &arg_vars[1].1)?
                ));
                return Ok(());
            }
            "concat" => {
                if matches!(ty, HostType::Tensor(_)) {
                    // Only a list-classed owner is ever upgraded to a move
                    // (chelis#2205); a moved tensor operand here would mean
                    // the scheduler and the emitter disagree about the class.
                    if container_operand_is_moved(site, "builtin:concat")? {
                        return Err(invalid_abi_shape(
                            "tensor `concat` received a moved container operand".to_string(),
                            "verified C host ownership emission",
                        ));
                    }
                    self.lines.push(format!(
                        "{}{target} = chelis_tensor_concat({}, {});",
                        self.indent, arg_vars[0].0, arg_vars[1].0
                    ));
                } else {
                    let entry = if container_operand_is_moved(site, "builtin:concat")? {
                        "chelis_list_concat_owned"
                    } else {
                        "chelis_list_concat"
                    };
                    self.lines.push(format!(
                        "{}{target} = {entry}({}, {});",
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
            "skip" => {
                // The published runtime symbol keeps its `chelis_list_drop`
                // spelling: it is already unambiguous behind the `list_`
                // prefix, and renaming a C ABI identity would retire a
                // capacity-census row without removing any ambiguity.
                self.lines.push(format!(
                    "{}{target} = chelis_list_drop({}, {});",
                    self.indent, arg_vars[0].0, arg_vars[1].0
                ));
                return Ok(());
            }
            "drop" => {
                // Same reason as the interpreter's arm: a lowering can
                // synthesize this node below the checker, and emitting `0`
                // for a two-argument call binds a null list tail that only
                // the runtime's own null check catches.
                if arg_vars.len() != 1 {
                    return Err(invalid_abi_shape(
                        format!(
                            "drop reached C emission with {} arguments; [05-OP-67] takes exactly one and the List slice is `skip` ([05-OP-54])",
                            arg_vars.len()
                        ),
                        "linearity consume emission",
                    ));
                }
                self.lines.push(format!("{}{target} = 0;", self.indent));
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
                // chelis#2205: a container the verifier moved at its last use
                // is consumed by the owned entry point; a borrowed one still
                // goes through the cloning call.
                let entry = if container_operand_is_moved(site, "builtin:dict_remove")? {
                    "chelis_dict_remove_owned"
                } else {
                    "chelis_dict_remove"
                };
                self.lines.push(format!(
                    "{}{target} = {entry}({}, {});",
                    self.indent,
                    arg_vars[0].0,
                    self.box_value_expr(&arg_vars[1].0, &arg_vars[1].1)?
                ));
                return Ok(());
            }
            "dict_insert" => {
                let entry = if container_operand_is_moved(site, "builtin:dict_insert")? {
                    "chelis_dict_insert_owned"
                } else {
                    "chelis_dict_insert"
                };
                self.lines.push(format!(
                    "{}{target} = {entry}({}, {}, {});",
                    self.indent,
                    arg_vars[0].0,
                    self.box_value_expr(&arg_vars[1].0, &arg_vars[1].1)?,
                    self.box_value_expr(&arg_vars[2].0, &arg_vars[2].1)?
                ));
                return Ok(());
            }
            "dict_merge" => {
                let entry = if container_operand_is_moved(site, "builtin:dict_merge")? {
                    "chelis_dict_merge_owned"
                } else {
                    "chelis_dict_merge"
                };
                self.lines.push(format!(
                    "{}{target} = {entry}({}, {});",
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
        // A TENSOR operand reaching these scalar operator arms normally means
        // the op has no tensor emission arm (the tensor block above returned
        // early for every such op). `cmplt` is the one legacy host-capable
        // exception: its expression arm below calls the dtype-dispatched,
        // shape-validating `chelis_tensor_cmplt` runtime entry. Emitting
        // `cos(ptr)` or `a + b` over `chelis_tensor*` is garbage C that fails
        // (or corrupts) at the user's compiler. This is the loud terminal the
        // section C3 laundering rule requires for the recoverable
        // `lower_transcendental` raise: the speculative-probe fallback lands
        // here and errs instead of emitting.
        let tensor_args = arg_vars
            .iter()
            .filter(|(_, arg_ty)| matches!(arg_ty, HostType::Tensor(_)))
            .count();
        let admitted_tensor_cmplt = name == "cmplt" && tensor_args == 2 && arg_vars.len() == 2;
        if SCALAR_NUMERIC_BUILTINS.contains(&name) && tensor_args != 0 && !admitted_tensor_cmplt {
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
                return Ok(());
            }
            // chelis#2120: the host lane had no `uniform_like` arm, so a draw
            // whose template was not constant-foldable — and which therefore
            // did not route through the tensor-DAG lane — fell into `decode`'s
            // [04-TOT-2] catch-all while `eval` answered correctly. It
            // allocates and fills a tensor, so it is completed here rather
            // than admitted to the expression vocabulary below.
            "uniform_like" => {
                self.assign_uniform_like(target, args, &arg_vars)?;
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
            let direct_extrema = |comparison| {
                let lhs = numeric_arg(0);
                let rhs = numeric_arg(1);
                let lhs_nan = EmittedExpr::call("isnan", [lhs.clone()]);
                let rhs_not_nan = EmittedExpr::unary(
                    UnaryOperator::LogicalNot,
                    EmittedExpr::call("isnan", [rhs.clone()]),
                );
                let ordered = binary(comparison, lhs, rhs);
                let select_left = binary(
                    BinaryOperator::LogicalOr,
                    lhs_nan,
                    binary(BinaryOperator::LogicalAnd, rhs_not_nan, ordered),
                );
                EmittedExpr::conditional(select_left, arg(0), arg(1))
            };
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
                    // chelis#2205: a string the verifier moved at its last
                    // use is consumed by the owned entry point; a borrowed
                    // one still goes through the cloning call.
                    let entry = if container_operand_is_moved(site, "builtin:string_concat")? {
                        "chelis_string_concat_owned"
                    } else {
                        "chelis_string_concat"
                    };
                    EmittedExpr::call(entry, [arg(0), arg(1)])
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
                CExpressionBuiltin::CharCode => EmittedExpr::call("chelis_char_code", [arg(0)]),
                CExpressionBuiltin::CharFromCode => {
                    EmittedExpr::call("chelis_char_from_code", [arg(0)])
                }
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
                CExpressionBuiltin::Relu => {
                    let helper = activation_math_function(
                        ty,
                        "chelis_host_relu_f16",
                        "chelis_host_relu_bf16",
                        "chelis_host_relu_f32",
                        "chelis_host_relu_f64",
                    );
                    match ty {
                        // Reduced-float host ABI values are their exact stored
                        // bits. Decode only for the comparison, then select
                        // the raw argument or raw +0 through the closed AST.
                        HostType::Float16 | HostType::BFloat16 => EmittedExpr::conditional(
                            binary(
                                BinaryOperator::Less,
                                numeric_arg(0),
                                EmittedExpr::integer(0),
                            ),
                            EmittedExpr::integer(0),
                            arg(0),
                        ),
                        HostType::Float32 | HostType::Float64 => {
                            EmittedExpr::call(helper, [numeric_arg(0)])
                        }
                        _ => unreachable!("non-float ReLU rejected above"),
                    }
                }
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
                    binary(BinaryOperator::LessEqual, arg(0), arg(1)),
                    arg(0),
                    arg(1),
                ),
                CExpressionBuiltin::MaxElem if is_integer_abi(ty) => EmittedExpr::conditional(
                    binary(BinaryOperator::GreaterEqual, arg(0), arg(1)),
                    arg(0),
                    arg(1),
                ),
                CExpressionBuiltin::MinElem => direct_extrema(BinaryOperator::LessEqual),
                CExpressionBuiltin::MaxElem => direct_extrema(BinaryOperator::GreaterEqual),
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
    //   * `assign_tensor_binary_func_elementwise` emits direct operand
    //     selection for every represented dtype. Float arms preserve the
    //     first NaN and every lhs equality bit-pattern; integer and Bool arms
    //     select the lhs on equality.
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
        let source_index = format!("{target}_cast_source_i");
        self.emit_elementwise_index_step(target, "cast", input, input);
        self.lines.push(format!(
            "{}{target} = chelis_host_alloc_like({input}, {});",
            self.indent,
            sparse_dtype_macro(target_prim)
        ));
        self.lines.push(format!(
            "{}if (chelis_host_tensor_dtype({input}) != {}) {{",
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
            "{}const {source_type} *{source_data} = (const {source_type} *)chelis_host_tensor_data({input});",
            self.indent,
        ));
        let (guard, view) = self.begin_tensor_write(target);
        self.lines.push(format!(
            "{}{target_type} *{target_data} = ({target_type} *){view}.data;",
            self.indent,
        ));
        self.lines.push(format!(
            "{}for (int64_t {flat_index} = 0; {flat_index} < {view}.count; {flat_index}++) {{",
            self.indent,
        ));
        self.lines.push(format!(
            "{}    int64_t {source_index} = {flat_index} * {target}_cast_step;",
            self.indent,
        ));
        let source_value = format!("{source_data}[{source_index}]");
        let expression = checked_cast_c_expr(plan, &source_value);
        self.lines.push(format!(
            "{}    {target_data}[{flat_index}] = {expression};",
            self.indent,
        ));
        self.lines.push(format!("{}}}", self.indent));
        self.end_tensor_write(&guard);
    }

    /// chelis#1484: the host-value lane's runtime operand-agreement guard for
    /// binary elementwise ops, and the sibling of `emit.rs`'s
    /// `emit_elementwise_operand_guard` (chelis#664, chelis#668).
    ///
    /// The emitters below allocate the result at the LHS rank and then read
    /// every operand through the TARGET's index vector applied to that
    /// operand's strides. An operand whose runtime rank or shape disagrees is
    /// therefore read partially or out of bounds, SILENTLY, where the
    /// evaluator rejects with "tensor shapes must match for elementwise op".
    /// An elementwise call one of whose operands carries an IO effect is
    /// lowered here rather than through the tensor DAG, so before this guard
    /// existed such a call reached codegen with no operand comparison in
    /// either lane.
    ///
    /// Scope, and the reason it is not simply "ranks must be equal":
    /// `spec/05-risc-primitives.md` §1.2 forbids implicit rank extension, so
    /// two positive ranks that differ are always a defect and abort. A rank-0
    /// operand is the backend's own scalar-input representation, not source
    /// broadcasting, and keeps its established meaning. At equal positive
    /// rank the shapes are compared axis by axis, which is what §2.4 already
    /// requires of the tensor-DAG lane.
    ///
    /// The message wording is the DAG guard's, so one grep over the emitted C
    /// finds either lane; the emitted result and operand variable names
    /// locate the site inside a generated translation unit.
    fn emit_elementwise_operand_guard(&mut self, target: &str, lhs: &str, rhs: &str) {
        self.lines.push(format!(
            "{}chelis_host_require_elementwise_agreement({lhs}, {rhs}, \"{target}\", \"{lhs}\", \"{rhs}\");",
            self.indent
        ));
    }

    fn emit_elementwise_index_step(
        &mut self,
        target: &str,
        label: &str,
        input: &str,
        domain: &str,
    ) {
        self.lines.push(format!(
            "{}const int64_t {target}_{label}_step = chelis_tensor_elementwise_index_step({input}, {domain});",
            self.indent,
        ));
    }

    fn assign_tensor_binary_elementwise(&mut self, target: &str, lhs: &str, rhs: &str, op: &str) {
        self.emit_elementwise_operand_guard(target, lhs, rhs);
        self.emit_elementwise_index_step(target, "lhs", lhs, lhs);
        self.emit_elementwise_index_step(target, "rhs", rhs, lhs);
        self.lines.push(format!(
            "{}{target} = chelis_host_alloc_like({lhs}, chelis_host_tensor_dtype({lhs}));",
            self.indent
        ));
        let (guard, view) = self.begin_tensor_write(target);
        self.lines
            .push(format!("{}switch ({view}.dtype) {{", self.indent));
        for arm in DtypeArm::all_operator_arms() {
            self.emit_binary_elementwise_arm(target, lhs, rhs, op, *arm, &view);
        }
        self.emit_default_runtime_fail_arm_for(&format!("{view}.dtype"), "binary elementwise op");
        self.lines.push(format!("{}}}", self.indent));
        self.end_tensor_write(&guard);
    }

    fn assign_tensor_binary_func_elementwise(
        &mut self,
        target: &str,
        lhs: &str,
        rhs: &str,
        func: BinaryElementwiseFunc,
    ) {
        self.emit_elementwise_operand_guard(target, lhs, rhs);
        self.emit_elementwise_index_step(target, "lhs", lhs, lhs);
        self.emit_elementwise_index_step(target, "rhs", rhs, lhs);
        self.lines.push(format!(
            "{}{target} = chelis_host_alloc_like({lhs}, chelis_host_tensor_dtype({lhs}));",
            self.indent
        ));
        let (guard, view) = self.begin_tensor_write(target);
        self.lines
            .push(format!("{}switch ({view}.dtype) {{", self.indent));
        for arm in DtypeArm::all_operator_arms() {
            self.emit_binary_func_elementwise_arm(target, lhs, rhs, func, *arm, &view);
        }
        self.emit_default_runtime_fail_arm_for(
            &format!("{view}.dtype"),
            &format!("binary func elementwise ({})", func.label()),
        );
        self.lines.push(format!("{}}}", self.indent));
        self.end_tensor_write(&guard);
    }

    fn assign_tensor_unary_elementwise(&mut self, target: &str, input: &str, op: &str) {
        self.emit_elementwise_index_step(target, "input", input, input);
        self.lines.push(format!(
            "{}{target} = chelis_host_alloc_like({input}, chelis_host_tensor_dtype({input}));",
            self.indent
        ));
        let (guard, view) = self.begin_tensor_write(target);
        self.lines
            .push(format!("{}switch ({view}.dtype) {{", self.indent));
        for arm in DtypeArm::all_operator_arms() {
            self.emit_unary_elementwise_arm(target, input, op, *arm, &view);
        }
        self.emit_default_runtime_fail_arm_for(&format!("{view}.dtype"), "unary elementwise op");
        self.lines.push(format!("{}}}", self.indent));
        self.end_tensor_write(&guard);
    }

    fn assign_tensor_unary_func_elementwise(&mut self, target: &str, input: &str, func: &str) {
        self.emit_elementwise_index_step(target, "input", input, input);
        self.lines.push(format!(
            "{}{target} = chelis_host_alloc_like({input}, chelis_host_tensor_dtype({input}));",
            self.indent
        ));
        let (guard, view) = self.begin_tensor_write(target);
        self.lines
            .push(format!("{}switch ({view}.dtype) {{", self.indent));
        for arm in DtypeArm::f32_payload_func_arms() {
            self.emit_unary_func_elementwise_arm(target, input, func, *arm, &view);
        }
        self.emit_dtype_fail_arms(
            &[DtypeArm::F64, DtypeArm::I32, DtypeArm::I64, DtypeArm::Bool],
            &format!("unary func elementwise ({func})"),
        );
        self.emit_default_runtime_fail_arm_for(
            &format!("{view}.dtype"),
            &format!("unary func elementwise ({func})"),
        );
        self.lines.push(format!("{}}}", self.indent));
        self.end_tensor_write(&guard);
    }

    /// Emit one arm of the elementwise binary operator dispatch.
    fn emit_binary_elementwise_arm(
        &mut self,
        target: &str,
        lhs: &str,
        rhs: &str,
        op: &str,
        arm: DtypeArm,
        target_view: &str,
    ) {
        let ind = &self.indent;
        let macro_name = arm.dtype_macro();
        let elem_t = arm.elem_t();
        self.lines.push(format!("{ind}    case {macro_name}: {{"));
        self.lines.push(format!(
            "{ind}        {elem_t} *__target_data = ({elem_t}*){target_view}.data;"
        ));
        self.lines.push(format!(
            "{ind}        const {elem_t} *__lhs_data = (const {elem_t}*)chelis_host_tensor_data({lhs});"
        ));
        self.lines.push(format!(
            "{ind}        const {elem_t} *__rhs_data = (const {elem_t}*)chelis_host_tensor_data({rhs});"
        ));
        self.lines.push(format!(
            "{ind}        for (int64_t i = 0; i < {target_view}.count; i++) {{"
        ));
        self.lines.push(format!(
            "{ind}            int64_t idx_lhs = i * {target}_lhs_step;"
        ));
        self.lines.push(format!(
            "{ind}            int64_t idx_rhs = i * {target}_rhs_step;"
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
        target_view: &str,
    ) {
        let ind = &self.indent;
        let macro_name = arm.dtype_macro();
        let elem_t = arm.elem_t();
        self.lines.push(format!("{ind}    case {macro_name}: {{"));
        self.lines.push(format!(
            "{ind}        {elem_t} *__target_data = ({elem_t}*){target_view}.data;"
        ));
        self.lines.push(format!(
            "{ind}        const {elem_t} *__lhs_data = (const {elem_t}*)chelis_host_tensor_data({lhs});"
        ));
        self.lines.push(format!(
            "{ind}        const {elem_t} *__rhs_data = (const {elem_t}*)chelis_host_tensor_data({rhs});"
        ));
        self.lines.push(format!(
            "{ind}        for (int64_t i = 0; i < {target_view}.count; i++) {{"
        ));
        self.lines.push(format!(
            "{ind}            int64_t idx_lhs = i * {target}_lhs_step;"
        ));
        self.lines.push(format!(
            "{ind}            int64_t idx_rhs = i * {target}_rhs_step;"
        ));
        let expression = match arm {
            DtypeArm::F32 | DtypeArm::F64 => format!(
                "isnan(__lhs_data[idx_lhs]) || (!isnan(__rhs_data[idx_rhs]) && __lhs_data[idx_lhs] {} __rhs_data[idx_rhs]) ? __lhs_data[idx_lhs] : __rhs_data[idx_rhs]",
                func.comparison()
            ),
            DtypeArm::Bool | DtypeArm::I32 | DtypeArm::I64 => format!(
                "__lhs_data[idx_lhs] {} __rhs_data[idx_rhs] ? __lhs_data[idx_lhs] : __rhs_data[idx_rhs]",
                func.comparison()
            ),
        };
        self.lines
            .push(format!("{ind}            __target_data[i] = {expression};"));
        self.lines.push(format!("{ind}        }}"));
        self.lines.push(format!("{ind}        break;"));
        self.lines.push(format!("{ind}    }}"));
    }

    /// Emit one arm of the elementwise unary operator dispatch.
    fn emit_unary_elementwise_arm(
        &mut self,
        target: &str,
        input: &str,
        op: &str,
        arm: DtypeArm,
        target_view: &str,
    ) {
        let ind = &self.indent;
        let macro_name = arm.dtype_macro();
        let elem_t = arm.elem_t();
        self.lines.push(format!("{ind}    case {macro_name}: {{"));
        self.lines.push(format!(
            "{ind}        {elem_t} *__target_data = ({elem_t}*){target_view}.data;"
        ));
        self.lines.push(format!(
            "{ind}        const {elem_t} *__input_data = (const {elem_t}*)chelis_host_tensor_data({input});"
        ));
        self.lines.push(format!(
            "{ind}        for (int64_t i = 0; i < {target_view}.count; i++) {{"
        ));
        self.lines.push(format!(
            "{ind}            int64_t idx = i * {target}_input_step;"
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
        target_view: &str,
    ) {
        let ind = &self.indent;
        let macro_name = arm.dtype_macro();
        let elem_t = arm.elem_t();
        self.lines.push(format!("{ind}    case {macro_name}: {{"));
        self.lines.push(format!(
            "{ind}        {elem_t} *__target_data = ({elem_t}*){target_view}.data;"
        ));
        self.lines.push(format!(
            "{ind}        const {elem_t} *__input_data = (const {elem_t}*)chelis_host_tensor_data({input});"
        ));
        self.lines.push(format!(
            "{ind}        for (int64_t i = 0; i < {target_view}.count; i++) {{"
        ));
        self.lines.push(format!(
            "{ind}            int64_t idx = i * {target}_input_step;"
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
    /// `dtype_expr` names the already-validated tagged dtype used by the
    /// surrounding switch.
    fn emit_default_runtime_fail_arm_for(&mut self, dtype_expr: &str, site_name: &str) {
        let ind = &self.indent;
        self.lines.push(format!("{ind}    default: {{"));
        self.lines.push(format!(
            "{ind}        fprintf(stderr, \"{site_name} unsupported dtype %d\\n\", (int)({dtype_expr}));"
        ));
        self.lines.push(format!("{ind}        abort();"));
        self.lines.push(format!("{ind}    }}"));
    }

    fn assign_tensor_call(
        &mut self,
        destination: (&str, &HostType),
        helper: usize,
        entry_variant: usize,
        args: &[HostExpr],
        result_claims: Option<&str>,
    ) -> Result<(), Unsupported> {
        let (target, ty) = destination;
        if let Some(host_helper) = self.tensor_helpers.get(helper) {
            match host_helper.specialization.as_ref() {
                Some(HostTensorSpecialization::BlasMatmul(summary)) => {
                    self.assign_blas_matmul_summary(target, summary, args, ty)?;
                    self.stamp_result_origin(target, ty, "matmul");
                    self.emit_result_claim_guard(target, ty, result_claims);
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
                    self.stamp_result_origin(target, ty, "gather");
                    self.emit_result_claim_guard(target, ty, result_claims);
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
                    self.stamp_result_origin(target, ty, "scatter_add");
                    self.emit_result_claim_guard(target, ty, result_claims);
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
                    self.stamp_result_origin(target, ty, "scatter");
                    self.emit_result_claim_guard(target, ty, result_claims);
                    return Ok(());
                }
                None => {}
            }
        }

        let base = format!("{}__tensor_{helper}", self.helper_prefix);
        let entry_variant = if self.external_helpers.contains(&base) {
            0
        } else {
            entry_variant
        };
        let helper_name = random_helper_name(&entry::variant_name(&base, entry_variant));
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
                // Preserve the declared dtype and stored bits through the
                // existing tagged scalar carrier; no scalar class falls back
                // to f32 or interprets f16/bf16 storage as an integer value.
                let scalar = scalar_carrier_expr(&value_name, &inferred_ty)?;
                self.lines.push(format!(
                    "{}{tensor_name} = chelis_scalar_tensor({scalar});",
                    self.indent
                ));
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
            .tensor_helper_output_counts
            .get(helper)
            .copied()
            .unwrap_or(1);
        self.lines.push(format!(
            "{}chelis_tensor *{}[{}] = {{ NULL }};",
            self.indent, outputs_name, root_count
        ));
        let mut helper_args = vec![
            inputs_arg.clone(),
            tensor_args.len().to_string(),
            outputs_name.clone(),
            root_count.to_string(),
        ];
        append_private_context_args(&mut helper_args);
        if !self.external_helpers.contains(&base) {
            helper_args.push(result_claims.unwrap_or("NULL").to_string());
        }
        self.lines.push(format!(
            "{}{}({});",
            self.indent,
            helper_name,
            helper_args.join(", ")
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
            // The tuple retains its fields; release each helper output's
            // temporary boxed owner, just as for an ordinary tuple literal.
            for index in 0..root_count {
                self.lines.push(format!(
                    "{}chelis_value_release({values_name}[{index}]);",
                    self.indent
                ));
            }
        } else {
            self.lines
                .push(format!("{}{target} = {}[0];", self.indent, outputs_name));
        }
        for (_, boxed) in tensor_args {
            if let Some(boxed) = boxed {
                self.lines
                    .push(format!("{}chelis_tensor_release({boxed});", self.indent));
            }
        }
        let producer = self
            .tensor_helper_result_origins
            .get(helper)
            .cloned()
            .flatten();
        if let Some(producer) = producer {
            self.stamp_result_origin(target, ty, &producer);
        }
        if self.external_helpers.contains(&base) {
            self.emit_result_claim_guard(target, ty, result_claims);
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
        let plan = self.next_temp("blas_plan");
        let index_type = sparse_elem_type(Prim::Int64);
        let element = sparse_elem_type(Prim::F32);
        self.lines.push(format!("{}chelis_matmul_plan *{plan} = chelis_tensor_matmul_plan({lhs}, {rhs}, chelis_scalar_from_bits(CHELIS_DTYPE_F32, UINT64_C(0)));", self.indent));
        // The summary's declared output remains an independent target claim.
        let output_dims = summary
            .output
            .dims
            .iter()
            .map(|dim| match dim {
                DimInfo::Lit(value) | DimInfo::Named(_, Some(value)) => value.to_string(),
                DimInfo::Named(name, None) => self
                    .summary_symbol_expr(name, summary, &tensor_args)
                    .unwrap_or_else(|| {
                        panic!("BLAS summary output symbol `{name}` has no input binding")
                    }),
            })
            .collect::<Vec<_>>();
        let rank = output_dims.len();
        let tagged = output_dims
            .iter()
            .map(|d| format!("chelis_scalar_from_bits(CHELIS_DTYPE_I64, {d})"))
            .collect::<Vec<_>>()
            .join(", ");
        self.lines.push(format!("{}chelis_matmul_check_target({plan}, chelis_scalar_from_bits(CHELIS_DTYPE_I64, {rank}), (chelis_scalar[]){{ {tagged} }});", self.indent));
        self.lines.push(format!("{}chelis_matmul_check_vendor({plan}, chelis_scalar_from_bits(CHELIS_DTYPE_I64, CHELIS_BLAS_MAXIMUM));", self.indent));
        let shape_name = self.next_temp("blas_shape");
        let shape = (0..rank).map(|axis| format!("chelis_matmul_extent({plan}, chelis_scalar_from_bits(CHELIS_DTYPE_I64, {axis}))")).collect::<Vec<_>>().join(", ");
        self.lines.push(format!(
            "{}{index_type} {shape_name}[{rank}] = {{ {shape} }};",
            self.indent
        ));
        let m = self.next_temp("blas_m");
        let n = self.next_temp("blas_n");
        let k = self.next_temp("blas_k");
        for (name, dimension) in [(&m, "ROWS"), (&n, "COLUMNS"), (&k, "REDUCTION")] {
            self.lines.push(format!("{}{index_type} {name} = chelis_matmul_dimension({plan}, CHELIS_MATMUL_{dimension});", self.indent));
        }
        let count = self.next_temp("blas_batches");
        self.lines.push(format!(
            "{}{index_type} {count} = chelis_matmul_batch_count({plan});",
            self.indent
        ));
        self.lines.push(format!(
            "{}{target} = chelis_alloc({rank}, {shape_name}, CHELIS_DTYPE_F32);",
            self.indent
        ));
        let (guard, view) = self.begin_tensor_write(target);
        self.lines.push(format!("{}if ({k} == 0) {{", self.indent));
        let bytes = self.next_temp("blas_bytes");
        self.lines.push(format!(
            "{}    {index_type} {bytes} = chelis_tensor_byte_count({target});",
            self.indent
        ));
        self.lines.push(format!(
            "{}    if ({bytes} != 0) memset({view}.data, 0, (size_t){bytes});",
            self.indent
        ));
        self.lines.push(format!("{}}} else {{", self.indent));
        let batch = self.next_temp("blas_batch");
        let lhs_offset = self.next_temp("blas_lhs_offset");
        let rhs_offset = self.next_temp("blas_rhs_offset");
        let out_offset = self.next_temp("blas_out_offset");
        self.lines.push(format!(
            "{}    for ({index_type} {batch} = 0; {batch} < {count}; ++{batch}) {{",
            self.indent
        ));
        for (name, part) in [
            (&lhs_offset, "LEFT"),
            (&rhs_offset, "RIGHT"),
            (&out_offset, "RESULT"),
        ] {
            self.lines.push(format!("{}        {index_type} {name} = chelis_matmul_index({plan}, CHELIS_MATMUL_{part}, chelis_scalar_from_bits(CHELIS_DTYPE_I64, {batch}), chelis_scalar_from_bits(CHELIS_DTYPE_I64, 0));", self.indent));
        }
        self.lines.push(format!("{}        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (chelis_blas_integer){m}, (chelis_blas_integer){n}, (chelis_blas_integer){k}, 1.0f, (const {element}*)chelis_host_tensor_data({lhs}) + {lhs_offset}, (chelis_blas_integer){k}, (const {element}*)chelis_host_tensor_data({rhs}) + {rhs_offset}, (chelis_blas_integer){n}, 0.0f, ({element}*){view}.data + {out_offset}, (chelis_blas_integer){n});", self.indent));
        self.lines.push(format!("{}    }}", self.indent));
        self.lines.push(format!("{}}}", self.indent));
        self.end_tensor_write(&guard);
        self.lines.push(format!(
            "{}chelis_matmul_plan_release({plan});",
            self.indent
        ));
        Ok(())
    }

    fn emit_blas_summary_contract(
        &mut self,
        summary: &HostBlasMatmulSummary,
        tensor_args: &[String],
    ) {
        let mut symbolic_first = UnordMap::<String, String>::new();
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
                "{}if (chelis_host_tensor_dtype({arg}) != CHELIS_DTYPE_F32) {{",
                self.indent
            ));
            self.lines.push(format!(
                "{}    fprintf(stderr, \"specialized BLAS call input {input_index} expected f32 tensor\\n\");",
                self.indent
            ));
            self.lines.push(format!("{}    abort();", self.indent));
            self.lines.push(format!("{}}}", self.indent));
            self.lines.push(format!(
                "{}if (chelis_tensor_rank({arg}) != {}) {{",
                self.indent,
                ty.dims.len()
            ));
            self.lines.push(format!(
                "{}    fprintf(stderr, \"specialized BLAS call input {input_index} expected rank {}, got %d\\n\", chelis_tensor_rank({arg}));",
                self.indent,
                ty.dims.len()
            ));
            self.lines.push(format!("{}    abort();", self.indent));
            self.lines.push(format!("{}}}", self.indent));
            for (axis, dim) in ty.dims.iter().enumerate() {
                match dim {
                    DimInfo::Lit(size) | DimInfo::Named(_, Some(size)) => {
                        self.lines.push(format!(
                            "{}if (chelis_tensor_shape({arg}, {axis}) != {size}) {{",
                            self.indent
                        ));
                        self.lines.push(format!(
                            "{}    fprintf(stderr, \"specialized BLAS call input {input_index} axis {axis} expected {size}, got %lld\\n\", (long long)chelis_tensor_shape({arg}, {axis}));",
                            self.indent
                        ));
                        self.lines.push(format!("{}    abort();", self.indent));
                        self.lines.push(format!("{}}}", self.indent));
                    }
                    DimInfo::Named(name, None) => {
                        let expr = format!("chelis_tensor_shape({arg}, {axis})");
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
                        .then(|| format!("chelis_tensor_shape({arg}, {axis})"))
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
        let base = &tensor_args[summary.input_indices[0]];
        let indices = &tensor_args[summary.input_indices[1]];
        let (operation, updates) = match kind {
            SparseSummaryKind::Gather => ("CHELIS_SPARSE_GATHER", "NULL"),
            SparseSummaryKind::ScatterAdd => (
                "CHELIS_SPARSE_ADD",
                tensor_args[summary.input_indices[2]].as_str(),
            ),
            SparseSummaryKind::ScatterReplace => (
                "CHELIS_SPARSE_REPLACE",
                tensor_args[summary.input_indices[2]].as_str(),
            ),
        };
        let plan = self.next_temp("sparse_plan");
        let count = self.next_temp("sparse_count");
        let index_type = sparse_elem_type(Prim::Int64);
        self.lines.push(format!("{}chelis_sparse_plan *{plan} = chelis_tensor_sparse_plan({base}, {indices}, {updates}, chelis_scalar_from_bits(CHELIS_DTYPE_I64, {}), {operation});", self.indent, summary.axis));
        let output_dims = summary
            .output
            .dims
            .iter()
            .map(|dim| sparse_dim_info_expr(dim, summary, &tensor_args))
            .collect::<Vec<_>>();
        let tagged = if output_dims.is_empty() {
            "NULL".to_owned()
        } else {
            format!(
                "(chelis_scalar[]){{ {} }}",
                output_dims
                    .iter()
                    .map(|d| format!("chelis_scalar_from_bits(CHELIS_DTYPE_I64, {d})"))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        let rank = output_dims.len();
        self.lines.push(format!("{}chelis_sparse_check_target({plan}, chelis_scalar_from_bits(CHELIS_DTYPE_I64, {rank}), {tagged});", self.indent));
        self.lines.push(format!(
            "{}{index_type} {count} = chelis_sparse_count({plan});",
            self.indent
        ));
        let shape_name = self.next_temp("sparse_shape");
        let shape = if rank == 0 {
            "0".to_owned()
        } else {
            output_dims.join(", ")
        };
        self.lines.push(format!(
            "{}{index_type} {shape_name}[{}] = {{ {shape} }};",
            self.indent,
            rank.max(1)
        ));
        self.lines.push(format!(
            "{}{target} = chelis_alloc({rank}, {shape_name}, {target_dtype});",
            self.indent
        ));
        self.emit_sparse_summary_body(target, kind, summary, &tensor_args, &plan, &count);
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
        let mut symbolic_first = UnordMap::<String, String>::new();
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
                "{}if (chelis_host_tensor_dtype({arg}) != {expected_dtype}) {{",
                self.indent
            ));
            self.lines.push(format!(
                "{}    fprintf(stderr, \"specialized sparse call input {input_index} expected {expected_dtype} tensor\\n\");",
                self.indent
            ));
            self.lines.push(format!("{}    abort();", self.indent));
            self.lines.push(format!("{}}}", self.indent));
            self.lines.push(format!(
                "{}if (chelis_tensor_rank({arg}) != {}) {{",
                self.indent,
                ty.dims.len()
            ));
            self.lines.push(format!(
                "{}    fprintf(stderr, \"specialized sparse call input {input_index} expected rank {}, got %d\\n\", chelis_tensor_rank({arg}));",
                self.indent,
                ty.dims.len()
            ));
            self.lines.push(format!("{}    abort();", self.indent));
            self.lines.push(format!("{}}}", self.indent));
            for (axis, dim) in ty.dims.iter().enumerate() {
                match dim {
                    DimInfo::Lit(size) | DimInfo::Named(_, Some(size)) => {
                        self.lines.push(format!(
                            "{}if (chelis_tensor_shape({arg}, {axis}) != {size}) {{",
                            self.indent
                        ));
                        self.lines.push(format!(
                            "{}    fprintf(stderr, \"specialized sparse call input {input_index} axis {axis} expected {size}, got %lld\\n\", (long long)chelis_tensor_shape({arg}, {axis}));",
                            self.indent
                        ));
                        self.lines.push(format!("{}    abort();", self.indent));
                        self.lines.push(format!("{}}}", self.indent));
                    }
                    DimInfo::Named(name, None) => {
                        let expr = format!("chelis_tensor_shape({arg}, {axis})");
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

    /// Both host sparse forms consume the same checked domain as DAG kernels.
    fn emit_sparse_summary_body(
        &mut self,
        target: &str,
        kind: SparseSummaryKind,
        summary: &HostSparseOpSummary,
        args: &[String],
        plan: &str,
        count: &str,
    ) {
        let base = &args[summary.input_indices[0]];
        let indices = &args[summary.input_indices[1]];
        let element = sparse_elem_type(summary.output.precision);
        let index_element = sparse_elem_type(summary.input_tys[summary.input_indices[1]].precision);
        let index_type = sparse_elem_type(Prim::Int64);
        let (guard, view) = self.begin_tensor_write(target);
        if !matches!(kind, SparseSummaryKind::Gather) {
            let bytes = self.next_temp("sparse_bytes");
            self.lines.push(format!(
                "{}{index_type} {bytes} = chelis_tensor_byte_count({target});",
                self.indent
            ));
            self.lines.push(format!("{}if ({bytes} != 0) memcpy({view}.data, chelis_host_tensor_data({base}), (size_t){bytes});", self.indent));
        }
        let linear = self.next_temp("sparse_linear");
        let slot = self.next_temp("sparse_slot");
        let selected = self.next_temp("sparse_selected");
        let offset = self.next_temp("sparse_offset");
        self.lines.push(format!(
            "{}for ({index_type} {linear} = 0; {linear} < {count}; ++{linear}) {{",
            self.indent
        ));
        self.lines.push(format!("{}    {index_type} {slot} = chelis_sparse_index_slot({plan}, chelis_scalar_from_bits(CHELIS_DTYPE_I64, {linear}));", self.indent));
        self.lines.push(format!("{}    {index_type} {selected} = ((const {index_element}*)chelis_host_tensor_data({indices}))[{slot}];", self.indent));
        self.lines.push(format!("{}    {index_type} {offset} = chelis_sparse_data_index({plan}, chelis_scalar_from_bits(CHELIS_DTYPE_I64, {linear}), chelis_scalar_from_bits(CHELIS_DTYPE_I64, {selected}));", self.indent));
        match kind {
            SparseSummaryKind::Gather => self.lines.push(format!("{}    (({element}*){view}.data)[{linear}] = ((const {element}*)chelis_host_tensor_data({base}))[{offset}];", self.indent)),
            SparseSummaryKind::ScatterAdd | SparseSummaryKind::ScatterReplace => {
                let updates = &args[summary.input_indices[2]];
                let op = match kind { SparseSummaryKind::ScatterAdd => "+=", SparseSummaryKind::ScatterReplace => "=", SparseSummaryKind::Gather => unreachable!() };
                self.lines.push(format!("{}    (({element}*){view}.data)[{offset}] {op} ((const {element}*)chelis_host_tensor_data({updates}))[{linear}];", self.indent));
            }
        }
        self.lines.push(format!("{}}}", self.indent));
        self.end_tensor_write(&guard);
        self.lines.push(format!(
            "{}chelis_sparse_plan_release({plan});",
            self.indent
        ));
    }

    fn assign_call(
        &mut self,
        destination: (&str, &HostType),
        function: &str,
        args: &[HostExpr],
        arg_tys: &[HostType],
        site: &ProjectedHostSite<'a>,
        result_claims: Option<&str>,
    ) -> Result<(), Unsupported> {
        let (target, ty) = destination;
        if let Some(spec) = self.function_specializations.get(function).cloned() {
            match spec {
                HostFunctionSpecialization::BlasMatmul(summary) => {
                    self.assign_blas_matmul_summary(target, &summary, args, ty)?;
                    self.stamp_result_origin(target, ty, "matmul");
                    self.emit_result_claim_guard(target, ty, result_claims);
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
                    self.stamp_result_origin(target, ty, "gather");
                    self.emit_result_claim_guard(target, ty, result_claims);
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
                    self.stamp_result_origin(target, ty, "scatter_add");
                    self.emit_result_claim_guard(target, ty, result_claims);
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
                    self.stamp_result_origin(target, ty, "scatter");
                    self.emit_result_claim_guard(target, ty, result_claims);
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
        self.emit_pre_call_actions(site)?;
        // Calls to declared functions use private bodies and inherit this
        // invocation. Callback parameters retain their authored C signature.
        if self.emitted_names.contains_key(function) {
            append_private_context_args(&mut arg_vars);
            arg_vars.push(result_claims.unwrap_or("NULL").to_string());
            arg_vars.push(if matches!(ty, HostType::Tensor(_)) {
                format!("&{}", result_origin_name(target))
            } else {
                "NULL".to_string()
            });
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
        if !self.emitted_names.contains_key(function) {
            self.emit_result_claim_guard(target, ty, result_claims);
        }
        Ok(())
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
        let (fields_arg, field_values) = if fields.is_empty() {
            ("NULL".to_string(), None)
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
                    self.box_aggregate_value_expr(&field_var, &field_ty, field)?
                ));
            }
            (values_name.clone(), Some(values_name))
        };
        let ctor_value = self.next_temp("adt_ctor");
        self.lines.push(format!(
            "{}chelis_string {ctor_value} = {};",
            self.indent,
            runtime_string_literal(ctor)
        ));
        self.lines.push(format!(
            "{}{target} = chelis_adt_construct({ctor_value}, {}, {});",
            self.indent,
            fields_arg,
            fields.len()
        ));
        self.lines.push(format!(
            "{}chelis_string_release({ctor_value});",
            self.indent
        ));
        if let Some(values_name) = field_values {
            for index in 0..fields.len() {
                self.lines.push(format!(
                    "{}chelis_value_release({values_name}[{index}]);",
                    self.indent
                ));
            }
        }
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
        destination: (&str, &HostType),
        scrutinee: &HostExpr,
        arms: &[HostMatchArm],
        default_expr: Option<&HostExpr>,
        site: &ProjectedHostSite<'a>,
        on_result_spine: bool,
    ) -> Result<(), Unsupported> {
        let (target, expr_ty) = destination;
        let (scrutinee_owner, arm_edges) = site
            .directives
            .iter()
            .find_map(|action| match action {
                VerifiedHostAction::Terminator(VerifiedHostTerminator::Match {
                    scrutinee,
                    arms,
                    ..
                }) => Some((scrutinee.owner().id(), arms.clone())),
                _ => None,
            })
            .ok_or_else(|| {
                invalid_abi_shape(
                    "verified ADT-match site has no match terminator".to_string(),
                    "verified C host ownership emission",
                )
            })?;
        let expected_arms = arms.len() + usize::from(default_expr.is_some());
        if arm_edges.len() != expected_arms {
            return Err(invalid_abi_shape(
                format!(
                    "verified ADT match has {} blocks for {expected_arms} emitted arms",
                    arm_edges.len()
                ),
                "verified C host ownership emission",
            ));
        }
        let arm_blocks = Self::join_completion_blocks(site, expected_arms, "ADT-match")?;
        let scrutinee_var = self.next_temp("adt");
        self.emit_expr_to_var(scrutinee, &scrutinee_var, &host_type(scrutinee))?;
        // Pattern bindings may shadow the authored name that originally held
        // the scrutinee. Keep the verified owner attached to this
        // compiler-generated temporary so an arm-completion drop cannot be
        // redirected to a same-spelled scalar or another payload binding.
        self.owner_vars
            .insert(scrutinee_owner, scrutinee_var.clone());
        let tag_var = self.next_temp("adt_tag");
        self.lines.push(format!(
            "{}chelis_string {} = chelis_adt_get_tag({});",
            self.indent, tag_var, scrutinee_var
        ));
        for (index, arm) in arms.iter().enumerate() {
            let prefix = if index == 0 { "if" } else { "else if" };
            self.lines.push(format!(
                "{}{prefix} (chelis_host_string_eq_cstr({}, {:?})) {{",
                self.indent, tag_var, arm.ctor
            ));
            let nested_indent = format!("{}    ", self.indent);
            let previous = std::mem::replace(&mut self.indent, nested_indent);
            self.emit_edge_terminals(site.id, &arm_edges[index])?;
            // chelis#1222: one binder scope per arm. Each arm's pattern
            // bindings shadow any enclosing name they reuse, and their keys
            // carry no outgoing edge -- `chelis_adt_field` hands back an
            // independently retained handle, so the arm binding is not a
            // copy of anything this scope already owns.
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
                self.declare_result_origin(&binding.name, &binding.ty, Some("load"));
                self.bind_match_payload(
                    site,
                    arm_edges[index].target(),
                    "adt_payload:",
                    &binding.name,
                )?;
            }
            self.claim_on_spine = on_result_spine;
            self.assign_expr(target, &arm.expr, expr_ty)?;
            self.emit_expression_block_actions(site, arm_blocks[index], target)?;
            self.indent = previous;
            self.lines.push(format!("{}}}", self.indent));
        }
        self.lines.push(format!("{}else {{", self.indent));
        let nested_indent = format!("{}    ", self.indent);
        let previous = std::mem::replace(&mut self.indent, nested_indent);
        if let Some(default_expr) = default_expr {
            self.emit_edge_terminals(site.id, &arm_edges[arms.len()])?;
            self.claim_on_spine = on_result_spine;
            self.assign_expr(target, default_expr, expr_ty)?;
            self.emit_expression_block_actions(site, arm_blocks[arms.len()], target)?;
        } else {
            self.lines.push(format!(
                "{}fprintf(stderr, \"non-exhaustive ADT match\\n\");",
                self.indent
            ));
            self.lines.push(format!("{}exit(1);", self.indent));
        }
        self.indent = previous;
        self.lines.push(format!("{}}}", self.indent));
        self.lines
            .push(format!("{}chelis_string_release({tag_var});", self.indent));
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
                self.box_aggregate_value_expr(&item_var, item_ty, item)?
            ));
        }
        self.lines.push(format!(
            "{}{target} = chelis_list_from_values({}, {});",
            self.indent,
            values_name,
            items.len()
        ));
        for index in 0..items.len() {
            self.lines.push(format!(
                "{}chelis_value_release({values_name}[{index}]);",
                self.indent
            ));
        }
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
                    self.box_aggregate_value_expr(&item_var, item_ty, item)?
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
        if !items.is_empty() {
            for index in 0..items.len() {
                self.lines.push(format!(
                    "{}chelis_value_release({items_arg}[{index}]);",
                    self.indent
                ));
            }
        }
        Ok(())
    }

    fn assign_map(
        &mut self,
        target: &str,
        callback: &HostCallback,
        list: &HostExpr,
        _ty: &HostType,
        site: &ProjectedHostSite<'a>,
        body_block: VerifiedBlockId,
    ) -> Result<(), Unsupported> {
        let (body_edge, exit_edge) = Self::loop_edges(site)?;
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
        self.emit_edge_terminals(site.id, &body_edge)?;
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
        self.declare_result_origin(&result_var, &callback.ret_ty, None);
        let param = callback_param(callback, 0);
        let arg_var = self.next_temp("map_item");
        self.lines.push(format!(
            "{}{} {};",
            self.indent,
            c_type(&param.ty)?,
            arg_var
        ));
        self.assign_unboxed_value(&arg_var, &param.ty, &item_value)?;
        self.declare_result_origin(&arg_var, &param.ty, Some("load"));
        self.bind_loop_item(site, &arg_var)?;
        self.emit_callback_assign(callback, std::slice::from_ref(&arg_var), &result_var)?;
        self.lines.push(format!(
            "{}chelis_list_push({target}, {});",
            self.indent,
            self.box_value_expr(&result_var, &callback.ret_ty)?
        ));
        self.emit_expression_block_actions(site, body_block, target)?;
        self.indent = previous;
        self.lines.push(format!("{}}}", self.indent));
        self.emit_edge_terminals(site.id, &exit_edge)?;
        Ok(())
    }

    fn assign_filter(
        &mut self,
        target: &str,
        callback: &HostCallback,
        list: &HostExpr,
        _ty: &HostType,
        site: &ProjectedHostSite<'a>,
        body_block: VerifiedBlockId,
    ) -> Result<(), Unsupported> {
        let (body_edge, exit_edge) = Self::loop_edges(site)?;
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
        self.emit_edge_terminals(site.id, &body_edge)?;
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
        self.declare_result_origin(&arg_var, &param.ty, Some("load"));
        self.bind_loop_item(site, &arg_var)?;
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
        self.emit_expression_block_actions(site, body_block, target)?;
        self.indent = previous;
        self.lines.push(format!("{}}}", self.indent));
        self.emit_edge_terminals(site.id, &exit_edge)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn assign_fold(
        &mut self,
        target: &str,
        callback: &HostCallback,
        init: &HostExpr,
        list: &HostExpr,
        ty: &HostType,
        site: &ProjectedHostSite<'a>,
        preheader_block: VerifiedBlockId,
        body_block: VerifiedBlockId,
    ) -> Result<(), Unsupported> {
        let (body_edge, exit_edge) = Self::loop_edges(site)?;
        self.assign_expr(target, init, ty)?;
        self.emit_expression_block_actions(site, preheader_block, target)?;
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
        let params = callback_params(callback);
        let acc_arg = self.next_temp("fold_acc");
        self.lines.push(format!(
            "{}{} {} = {target};",
            self.indent,
            c_type(&params[0].ty)?,
            acc_arg
        ));
        if matches!(params[0].ty, HostType::Tensor(_)) {
            self.lines.push(format!(
                "{}__chelis_host_result_origin {} = {};",
                self.indent,
                result_origin_name(&acc_arg),
                result_origin_name(target)
            ));
        }
        let [body_acc] = body_edge.params() else {
            return Err(invalid_abi_shape(
                format!(
                    "verified fold body edge carries {} parameters, expected one accumulator",
                    body_edge.params().len()
                ),
                "verified C host ownership emission",
            ));
        };
        let [exit_acc] = exit_edge.params() else {
            return Err(invalid_abi_shape(
                format!(
                    "verified fold exit edge carries {} parameters, expected one accumulator",
                    exit_edge.params().len()
                ),
                "verified C host ownership emission",
            ));
        };
        // The body consumes the per-iteration accumulator copy, while the exit
        // owner remains represented by the expression result.  Bind both from
        // the verified loop edges before the callback's ownership actions run;
        // a later name-derived recovery would reintroduce backend inference.
        self.owner_vars.insert(body_acc.id(), acc_arg.clone());
        self.owner_vars.insert(exit_acc.id(), target.to_string());
        let body_acc_dropped = body_edge.terminals().iter().any(|terminal| {
            matches!(
                terminal,
                VerifiedTerminalView::Drop { owner, .. } if owner.id() == body_acc.id()
            )
        });
        self.emit_edge_terminals(site.id, &body_edge)?;
        if body_acc_dropped {
            // An inline callback still has a physical parameter even when the
            // verified program proves that parameter dead on entry.  Do not
            // propagate a released pointer into that non-semantic C alias.
            self.lines
                .push(format!("{}{} = NULL;", self.indent, acc_arg));
        }
        let item_value = self.next_temp("fold_item_value");
        self.lines.push(format!(
            "{}chelis_value {} = chelis_list_index({}, __i);",
            self.indent, item_value, list_var
        ));
        let item_arg = self.next_temp("fold_item");
        self.lines.push(format!(
            "{}{} {};",
            self.indent,
            c_type(&params[1].ty)?,
            item_arg
        ));
        self.assign_unboxed_value(&item_arg, &params[1].ty, &item_value)?;
        self.declare_result_origin(&item_arg, &params[1].ty, Some("load"));
        self.bind_loop_item(site, &item_arg)?;
        self.emit_callback_assign(callback, &[acc_arg, item_arg], target)?;
        self.emit_expression_block_actions(site, body_block, target)?;
        self.indent = previous;
        self.lines.push(format!("{}}}", self.indent));
        self.emit_edge_terminals(site.id, &exit_edge)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn assign_scan(
        &mut self,
        target: &str,
        callback: &HostCallback,
        init: &HostExpr,
        list: &HostExpr,
        ty: &HostType,
        site: &ProjectedHostSite<'a>,
        body_block: VerifiedBlockId,
    ) -> Result<(), Unsupported> {
        let HostType::List(inner_ty) = ty else {
            self.lines
                .push(format!("{}{target} = chelis_list_empty();", self.indent));
            return Ok(());
        };
        let acc_ty = inner_ty.as_ref().clone();
        let acc_var = self.next_temp("scan_acc");
        self.emit_expr_to_var(init, &acc_var, &acc_ty)?;
        let (body_edge, exit_edge) = site
            .directives
            .iter()
            .find_map(|action| match action {
                VerifiedHostAction::Terminator(VerifiedHostTerminator::Loop {
                    body_edge,
                    exit_edge,
                    ..
                }) => Some((body_edge.clone(), exit_edge.clone())),
                _ => None,
            })
            .ok_or_else(|| {
                invalid_abi_shape(
                    "verified scan site has no loop terminator".to_string(),
                    "verified C host ownership emission",
                )
            })?;
        for (edge, name) in [(&body_edge, "body"), (&exit_edge, "exit")] {
            let [state, output] = edge.params() else {
                return Err(invalid_abi_shape(
                    format!(
                        "verified scan {name} edge carries {} parameters, expected state and output",
                        edge.params().len()
                    ),
                    "verified C host ownership emission",
                ));
            };
            self.owner_vars.insert(state.id(), acc_var.clone());
            self.owner_vars.insert(output.id(), target.to_string());
        }
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
        self.emit_edge_terminals(site.id, &body_edge)?;
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
        if matches!(params[0].ty, HostType::Tensor(_)) {
            self.lines.push(format!(
                "{}__chelis_host_result_origin {} = {};",
                self.indent,
                result_origin_name(&acc_arg),
                result_origin_name(&acc_var)
            ));
        }
        let item_arg = self.next_temp("scan_item");
        self.lines.push(format!(
            "{}{} {};",
            self.indent,
            c_type(&params[1].ty)?,
            item_arg
        ));
        self.assign_unboxed_value(&item_arg, &params[1].ty, &item_value)?;
        self.declare_result_origin(&item_arg, &params[1].ty, Some("load"));
        self.bind_loop_item(site, &item_arg)?;
        self.emit_callback_assign(callback, &[acc_arg, item_arg], &acc_var)?;
        self.lines.push(format!(
            "{}chelis_list_push({target}, {});",
            self.indent,
            self.box_value_expr(&acc_var, &acc_ty)?
        ));
        self.emit_expression_block_actions(site, body_block, target)?;
        self.indent = previous;
        self.lines.push(format!("{}}}", self.indent));
        self.emit_edge_terminals(site.id, &exit_edge)?;
        Ok(())
    }

    fn assign_partition(
        &mut self,
        target: &str,
        callback: &HostCallback,
        list: &HostExpr,
        ty: &HostType,
        site: &ProjectedHostSite<'a>,
        body_block: VerifiedBlockId,
    ) -> Result<(), Unsupported> {
        let (body_edge, exit_edge) = Self::loop_edges(site)?;
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
        self.emit_edge_terminals(site.id, &body_edge)?;
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
        self.declare_result_origin(&arg_var, &param.ty, Some("load"));
        self.bind_loop_item(site, &arg_var)?;
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
        self.emit_expression_block_actions(site, body_block, target)?;
        self.indent = previous;
        self.lines.push(format!("{}}}", self.indent));
        self.emit_edge_terminals(site.id, &exit_edge)?;
        let tuple_values = self.next_temp("partition_values");
        self.lines
            .push(format!("{}chelis_value {}[2];", self.indent, tuple_values));
        self.lines.push(format!(
            "{}{}[0] = chelis_value_take_list({});",
            self.indent, tuple_values, pass_var
        ));
        self.lines.push(format!(
            "{}{}[1] = chelis_value_take_list({});",
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
        site: &ProjectedHostSite<'a>,
        body_block: VerifiedBlockId,
    ) -> Result<(), Unsupported> {
        let (body_edge, exit_edge) = Self::loop_edges(site)?;
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
        self.emit_edge_terminals(site.id, &body_edge)?;
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
        self.declare_result_origin(&result_var, &callback.ret_ty, None);
        let param = callback_param(callback, 0);
        let arg_var = self.next_temp("flat_map_item");
        self.lines.push(format!(
            "{}{} {};",
            self.indent,
            c_type(&param.ty)?,
            arg_var
        ));
        self.assign_unboxed_value(&arg_var, &param.ty, &item_value)?;
        self.declare_result_origin(&arg_var, &param.ty, Some("load"));
        self.bind_loop_item(site, &arg_var)?;
        self.emit_callback_assign(callback, std::slice::from_ref(&arg_var), &result_var)?;
        self.lines.push(format!(
            "{}chelis_list_extend({target}, {});",
            self.indent, result_var
        ));
        self.emit_expression_block_actions(site, body_block, target)?;
        self.indent = previous;
        self.lines.push(format!("{}}}", self.indent));
        self.emit_edge_terminals(site.id, &exit_edge)?;
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
                let mut arg_vars = arg_vars.to_vec();
                if self.emitted_names.contains_key(function) {
                    append_private_context_args(&mut arg_vars);
                    arg_vars.push("NULL".to_string());
                    arg_vars.push(if matches!(callback.ret_ty, HostType::Tensor(_)) {
                        format!("&{}", result_origin_name(target))
                    } else {
                        "NULL".to_string()
                    });
                }
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
                for (param, arg_var) in params.iter().zip(arg_vars.iter()) {
                    self.lines.push(format!(
                        "{}{} {} = {};",
                        self.indent,
                        c_type(&param.ty)?,
                        param.name,
                        arg_var
                    ));
                    if matches!(param.ty, HostType::Tensor(_)) {
                        self.lines.push(format!(
                            "{}__chelis_host_result_origin {} = {};",
                            self.indent,
                            result_origin_name(&param.name),
                            result_origin_name(arg_var)
                        ));
                    }
                }
                self.assign_expr(target, body, &callback.ret_ty)?;
            }
        }
        Ok(())
    }

    fn box_value_expr(&self, value: &str, ty: &HostType) -> Result<String, Unsupported> {
        Ok(match ty {
            HostType::Int8 => format!(
                "chelis_value_box_scalar(chelis_scalar_from_bits(CHELIS_DTYPE_I8, (uint64_t)(uint8_t)(int8_t){value}))"
            ),
            HostType::Int16 => format!(
                "chelis_value_box_scalar(chelis_scalar_from_bits(CHELIS_DTYPE_I16, (uint64_t)(uint16_t)(int16_t){value}))"
            ),
            HostType::Int32 => format!(
                "chelis_value_box_scalar(chelis_scalar_from_bits(CHELIS_DTYPE_I32, (uint64_t)(uint32_t)(int32_t){value}))"
            ),
            HostType::Int64 => format!(
                "chelis_value_box_scalar(chelis_scalar_from_bits(CHELIS_DTYPE_I64, (uint64_t)(int64_t){value}))"
            ),
            HostType::Float64 => format!(
                "chelis_value_box_scalar(chelis_scalar_from_bits(CHELIS_DTYPE_F64, chelis_host_f64_bits({value})))"
            ),
            HostType::Float32 => format!(
                "chelis_value_box_scalar(chelis_scalar_from_bits(CHELIS_DTYPE_F32, (uint64_t)chelis_host_f32_bits({value})))"
            ),
            HostType::Float16 => format!(
                "chelis_value_box_scalar(chelis_scalar_from_bits(CHELIS_DTYPE_F16, (uint64_t)(uint16_t){value}))"
            ),
            HostType::BFloat16 => format!(
                "chelis_value_box_scalar(chelis_scalar_from_bits(CHELIS_DTYPE_BF16, (uint64_t)(uint16_t){value}))"
            ),
            HostType::Bool => format!(
                "chelis_value_box_scalar(chelis_scalar_from_bits(CHELIS_DTYPE_BOOL, (uint64_t)({value} ? 1 : 0)))"
            ),
            HostType::String => format!("chelis_value_take_string({value})"),
            HostType::Adt(_, _) => format!("chelis_value_take_adt({value})"),
            HostType::Tensor(_) => format!("chelis_value_take_tensor({value})"),
            HostType::List(_) => format!("chelis_value_take_list({value})"),
            HostType::Tuple(_) => format!("chelis_value_take_tuple({value})"),
            HostType::Dict(_, _) => format!("chelis_value_take_dict({value})"),
            HostType::Option(_) => format!("chelis_value_take_option({value})"),
            HostType::MappedFile => format!("chelis_value_take_mapped_file({value})"),
            // Unit has no payload and no dedicated public chelis_value tag.
            // Its canonical structural runtime image is the empty tuple,
            // which already renders as `()` and participates in the generic
            // ADT/list carriers without expanding the public C ABI.
            HostType::Unit => {
                "chelis_value_take_tuple(chelis_tuple_from_values(NULL, 0))".to_string()
            }
            HostType::Callback(_, _) => {
                return Err(unsupported_value_boxing(ty, "boxing a resolved host value"));
            }
        })
    }

    fn box_aggregate_value_expr(
        &self,
        value: &str,
        ty: &HostType,
        _source: &HostExpr,
    ) -> Result<String, Unsupported> {
        self.box_value_expr(value, ty)
    }

    fn assign_unboxed_value(
        &mut self,
        target: &str,
        ty: &HostType,
        value_expr: &str,
    ) -> Result<(), Unsupported> {
        let expr = match ty {
            HostType::Int8 => format!(
                "(int8_t)chelis_host_scalar_as_i64(chelis_value_unbox_scalar({value_expr}), CHELIS_DTYPE_I8)"
            ),
            HostType::Int16 => format!(
                "(int16_t)chelis_host_scalar_as_i64(chelis_value_unbox_scalar({value_expr}), CHELIS_DTYPE_I16)"
            ),
            HostType::Int32 => format!(
                "(int32_t)chelis_host_scalar_as_i64(chelis_value_unbox_scalar({value_expr}), CHELIS_DTYPE_I32)"
            ),
            HostType::Int64 => format!(
                "chelis_host_scalar_as_i64(chelis_value_unbox_scalar({value_expr}), CHELIS_DTYPE_I64)"
            ),
            HostType::Float64 => format!(
                "chelis_host_scalar_as_float(chelis_value_unbox_scalar({value_expr}), CHELIS_DTYPE_F64)"
            ),
            HostType::Float32 => format!(
                "(float)chelis_host_scalar_as_float(chelis_value_unbox_scalar({value_expr}), CHELIS_DTYPE_F32)"
            ),
            HostType::Float16 => format!(
                "(uint16_t)chelis_host_scalar_bits(chelis_value_unbox_scalar({value_expr}), CHELIS_DTYPE_F16)"
            ),
            HostType::BFloat16 => format!(
                "(uint16_t)chelis_host_scalar_bits(chelis_value_unbox_scalar({value_expr}), CHELIS_DTYPE_BF16)"
            ),
            HostType::Bool => {
                format!("chelis_host_scalar_as_bool(chelis_value_unbox_scalar({value_expr}))")
            }
            HostType::String => format!("chelis_string_take_value({value_expr})"),
            HostType::Adt(_, _) => format!("chelis_adt_take_value({value_expr})"),
            HostType::Tensor(_) => format!("chelis_tensor_take_value({value_expr})"),
            HostType::List(_) => format!("chelis_list_take_value({value_expr})"),
            HostType::Tuple(_) => format!("chelis_tuple_take_value({value_expr})"),
            HostType::Dict(_, _) => format!("chelis_dict_take_value({value_expr})"),
            HostType::Option(_) => format!("chelis_option_take_value({value_expr})"),
            HostType::MappedFile => format!("chelis_mapped_file_take_value({value_expr})"),
            // The empty tuple carrier above has no scalar payload to read.
            HostType::Unit => "0".to_string(),
            HostType::Callback(_, _) => {
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
                "{}chelis_print_string({}); printf(\"\\n\");",
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
                     chelis_string text = chelis_string_from_scalar(chelis_value_unbox_scalar(boxed)); \
                     chelis_print_string(text); printf(\"\\n\"); \
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
            HostType::String => self
                .lines
                .push(format!("{}chelis_print_string({});", self.indent, value)),
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
                     chelis_string text = chelis_string_from_scalar(chelis_value_unbox_scalar(boxed)); \
                     chelis_print_string(text); \
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
                    RootPathStep::Tuple(_) => format!("chelis_tuple_borrow_value({current})"),
                    RootPathStep::Adt(_) => format!("chelis_adt_borrow_value({current})"),
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
             chelis_string_from_scalar(chelis_value_unbox_scalar({value})); \
             chelis_print_string(text); chelis_string_release(text); break; }}",
            self.indent
        ));
        self.lines.push(format!(
            "{}case CHELIS_VALUE_UNIT: printf(\"()\"); break;",
            self.indent
        ));
        self.lines.push(format!(
            "{}case CHELIS_VALUE_STRING: chelis_print_string(chelis_string_borrow_value({value})); break;",
            self.indent
        ));
        for (tag, printer, accessor) in [
            (
                "TENSOR",
                "chelis_print_tensor_stdout",
                "chelis_tensor_borrow_value",
            ),
            ("LIST", "chelis_print_list", "chelis_list_borrow_value"),
            ("TUPLE", "chelis_print_tuple", "chelis_tuple_borrow_value"),
            ("DICT", "chelis_print_dict", "chelis_dict_borrow_value"),
            ("ADT", "chelis_print_adt", "chelis_adt_borrow_value"),
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
        source: &HostExpr,
    ) -> Result<(), Unsupported> {
        let HostType::Option(inner) = ty else {
            return Err(invalid_abi_shape(
                format!("Some constructor carries non-option ABI type `{ty:?}`"),
                "Some constructor",
            ));
        };
        require_same_abi_type(inner.as_ref(), value_ty, "Some constructor payload")?;
        let boxed = self.next_temp("option_payload");
        let boxed_expr = self.box_aggregate_value_expr(value_var, value_ty, source)?;
        self.lines.push(format!(
            "{}chelis_value {boxed} = {boxed_expr};",
            self.indent
        ));
        self.lines.push(format!(
            "{}{target} = chelis_option_some({boxed});",
            self.indent
        ));
        self.lines
            .push(format!("{}chelis_value_release({boxed});", self.indent));
        Ok(())
    }

    fn assign_option_none(&mut self, target: &str, ty: &HostType) -> Result<(), Unsupported> {
        let HostType::Option(_) = ty else {
            return Err(invalid_abi_shape(
                format!("None constructor carries non-option ABI type `{ty:?}`"),
                "None constructor",
            ));
        };
        self.lines
            .push(format!("{}{target} = chelis_option_none();", self.indent));
        Ok(())
    }
}

/// The runtime release call that frees the heap allocation a value of
/// `ty` held in `var` owns, or `None` for non-owning types. Called only
/// while formatting a verified `Drop` or `RootConsume` action.
fn release_call(var: &str, ty: &HostType) -> Option<String> {
    // #379: the var may be a user binding/let name spelled like a C keyword;
    // route through `c_ident` so the free call names the same (possibly
    // mangled) identifier the declaration used. Compiler temps
    // (`__binding_N_value`, `__let_N`) pass through unchanged.
    let var = c_ident(var);
    match ty {
        HostType::Tensor(_) => Some(format!("chelis_tensor_release({var});")),
        HostType::List(_) => Some(format!("chelis_list_release({var});")),
        HostType::Tuple(_) => Some(format!("chelis_tuple_release({var});")),
        HostType::Dict(_, _) => Some(format!("chelis_dict_release({var});")),
        HostType::Adt(_, _) => Some(format!("chelis_adt_release({var});")),
        HostType::String => Some(format!("chelis_string_release({var});")),
        HostType::Option(_) => Some(format!("chelis_option_release({var});")),
        HostType::MappedFile => Some(format!("chelis_mapped_file_release({var});")),
        // Scalars (int/float/bool/unit), function pointers, and `Unknown`
        // have no concrete heap owner that this Phase 1 emitter can release.
        _ => None,
    }
}

/// The runtime retain call that formats one verified `Clone` action.
fn retain_call(var: &str, ty: &HostType) -> Option<String> {
    // #379: mirror `release_call` — a user name spelled like a C keyword
    // routes through `c_ident`; compiler temps pass through unchanged.
    let var = c_ident(var);
    match ty {
        HostType::Tensor(_) => Some(format!("chelis_tensor_retain({var});")),
        HostType::List(_) => Some(format!("chelis_list_retain({var});")),
        HostType::Tuple(_) => Some(format!("chelis_tuple_retain({var});")),
        HostType::Dict(_, _) => Some(format!("chelis_dict_retain({var});")),
        HostType::Adt(_, _) => Some(format!("chelis_adt_retain({var});")),
        HostType::String => Some(format!("chelis_string_retain({var});")),
        HostType::Option(_) => Some(format!("chelis_option_retain({var});")),
        HostType::MappedFile => Some(format!("chelis_mapped_file_retain({var});")),
        _ => None,
    }
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

fn result_origin_name(value: &str) -> String {
    format!("__chelis_result_origin_{}", c_ident(value))
}

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
        | HostExprKind::TensorCall { ty, .. }
        | HostExprKind::ResultClaimScope { ty, .. } => ty.clone(),
        HostExprKind::Unit | HostExprKind::SignatureEntry { .. } => HostType::Unit,
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
    fn label(self) -> &'static str {
        match self {
            Self::Max => "max_elem",
            Self::Min => "min_elem",
        }
    }

    fn comparison(self) -> &'static str {
        match self {
            Self::Max => ">=",
            Self::Min => "<=",
        }
    }
}

/// chelis#2120: recover a `uniform_like` bound's exact source value from the
/// host expression tree.
///
/// The checker already restricts these bounds to static literals
/// (`chelis-types` `infer::app_operand_dtype`, whose message is the
/// chelis#776 gate), and this mirrors the shapes `is_static_numeric_bound`
/// admits there: a bare float or integer literal, a float-target `cast`
/// around one, and `neg` of either. Anything else is an internal desync
/// between that gate and this emitter, and is rejected loudly rather than
/// defaulted — silently substituting `[0, 1)` for an unreadable bound is the
/// exact chelis#776 failure this must not reintroduce.
fn static_float_bound(expr: Option<&HostExpr>) -> Option<f64> {
    match &expr?.kind {
        HostExprKind::Float(value) => Some(*value),
        HostExprKind::Int(value) => Some(*value as f64),
        HostExprKind::Builtin { name, args, .. } if name == "cast" => {
            static_float_bound(args.first())
        }
        HostExprKind::Builtin { name, args, .. } if name == "neg" => {
            static_float_bound(args.first()).map(|value| -value)
        }
        _ => None,
    }
}

/// The loud terminal for a `uniform_like` bound this emitter cannot fold.
fn unresolved_uniform_bound(which: &str) -> Unsupported {
    Unsupported::new(
        UnsupportedKind::Builtin("uniform_like".to_string()),
        "`chelis build` host emission",
        Stage::Codegen("c"),
        chelis_types::deliberate_rejection!(
            "[04-TOT-2]",
            "uniform_like's bounds must be static literals the emitter can narrow to f32 \
             exactly; the checker already rejects a runtime-computed bound, so an \
             unreadable one here is an internal desync"
        ),
    )
    .with_supported_alternative(match which {
        "low" => "give `uniform_like` a literal low bound",
        _ => "give `uniform_like` a literal high bound",
    })
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
                    .then(|| format!("chelis_tensor_shape({arg}, {axis})"))
            })
        })
}

#[cfg(test)]
mod expression_dispatch_tests {
    use super::*;

    #[test]
    fn source_main_keeps_its_module_abi_after_linker_qualification() {
        assert_eq!(emitted_function_name("demo", "main"), "demo__main");
        assert_eq!(
            emitted_function_name("ignored", "pkg__demo__Demo__Main__main"),
            "pkg__demo__Demo__Main__main"
        );
        assert_eq!(
            emitted_function_name("ignored", "pkg__demo__Demo__Main__almost__main"),
            "chelis_fn_706b675f5f64656d6f5f5f44656d6f5f5f4d61696e5f5f616c6d6f73745f5f6d61696e"
        );
    }

    #[test]
    fn verified_clone_and_drop_formatters_cover_all_eight_public_heap_payloads() {
        let scalar = HostType::Int64;
        let heap = [
            (HostType::String, "chelis_string"),
            (
                HostType::Tensor(TensorType {
                    dims: Vec::new(),
                    precision: Prim::F32,
                }),
                "chelis_tensor",
            ),
            (HostType::List(Box::new(scalar.clone())), "chelis_list"),
            (HostType::Tuple(vec![scalar.clone()]), "chelis_tuple"),
            (
                HostType::Dict(Box::new(scalar.clone()), Box::new(scalar.clone())),
                "chelis_dict",
            ),
            (HostType::Adt("Probe".into(), Vec::new()), "chelis_adt"),
            (HostType::Option(Box::new(scalar.clone())), "chelis_option"),
            (HostType::MappedFile, "chelis_mapped_file"),
        ];
        for (ty, prefix) in heap {
            assert_eq!(
                retain_call("owner", &ty),
                Some(format!("{prefix}_retain(owner);")),
                "verified Clone lost its {ty:?} runtime effect"
            );
            assert_eq!(
                release_call("owner", &ty),
                Some(format!("{prefix}_release(owner);")),
                "verified Drop lost its {ty:?} runtime effect"
            );
        }

        for ty in [
            HostType::Int64,
            HostType::Float32,
            HostType::Bool,
            HostType::Unit,
        ] {
            assert_eq!(retain_call("owner", &ty), None, "{ty:?}");
            assert_eq!(release_call("owner", &ty), None, "{ty:?}");
        }

        let source = include_str!("host_emit.rs");
        assert!(source.contains("VerifiedHostOperation::Clone"));
        assert!(source.contains("VerifiedHostOperation::Drop"));
        assert!(source.contains("self.emit_clone_to"));
        assert!(source.contains("release_call(&var, &ty)"));
    }

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
                !emitted.contains("chelis_value_box_scalar"),
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
                !emitted.contains("chelis_value_unbox_scalar"),
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
            UnordMap::new(),
            UnordMap::new(),
            HostTensorHelpers {
                helpers: &[],
                output_counts: &[],
                result_origins: Vec::new(),
            },
            &[],
        );
        emitter.emit_labeled_boxed_root("root", "boxed");
        let emitted = emitter.lines.join("\n");

        for required in [
            "case CHELIS_VALUE_SCALAR:",
            "chelis_value_unbox_scalar(boxed)",
            "chelis_string_borrow_value(boxed)",
            "chelis_tensor_borrow_value(boxed)",
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
            error.what.as_ref(),
            &UnsupportedKind::Builtin("future_unimplemented_builtin".into())
        );
    }

    #[test]
    fn runtime_string_literals_use_fixed_width_byte_escapes_for_every_admitted_class() {
        let value = "\"\\\n\t\r\0A1é";
        assert_eq!(
            c_utf8_byte_literal(value),
            r#""\042\134\012\011\015\000\101\061\303\251""#
        );
        assert_eq!(
            runtime_string_literal(value),
            r#"chelis_string_from_utf8((const uint8_t *)"\042\134\012\011\015\000\101\061\303\251", INT64_C(10))"#
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
        for width in ["f16", "bf16", "f32", "f64"] {
            let start = emitted
                .find(&format!("chelis_host_relu_{width}"))
                .expect("ReLU helper start");
            let body = &emitted[start..];
            let end = body.find("}\n").expect("ReLU helper end");
            let body = &body[..end];
            assert!(body.contains("return x <"), "{width}: {body}");
            assert!(!body.contains("fmax"), "{width}: {body}");
        }
        assert!(emitted.contains("chelis_host_finalize_f16"));
        assert!(emitted.contains("chelis_host_finalize_bf16"));
    }

    /// Extent decoding and checked allocation belong to the runtime owner.
    #[test]
    fn reshape_helper_delegates_exact_int64_metadata_to_runtime() {
        let mut out = Vec::new();
        append_tensor_reshape_helper(&mut out);
        assert_eq!(
            out,
            [
                "static chelis_tensor* chelis_host_reshape_tensor(chelis_tensor* input, const chelis_list* shape_values) {",
                "    return chelis_tensor_reshape(input, shape_values);",
                "}",
            ]
        );
    }
}
