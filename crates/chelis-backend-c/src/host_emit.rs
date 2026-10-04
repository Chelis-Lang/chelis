use chelis_ir::host::{
    HostBlasMatmulSummary, HostFunctionSpecialization, HostSparseOpSummary, HostTensorHelper,
    HostTensorSpecialization,
};
mod entry;
mod entry_walk;
use entry_walk::function_entry_plan;

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
    Abs,
    Min,
    Max,
    MinElem,
    MaxElem,
}

impl CExpressionBuiltin {
    /// Every builtin name the host scalar lane decodes, and so every
    /// expression it can emit.
    const NAMES: [(&'static str, Self); 64] = [
        ("add", Self::Add),
        ("sub", Self::Sub),
        ("mul", Self::Mul),
        ("div", Self::Div),
        ("trunc_div", Self::TruncDiv),
        ("floor_div", Self::FloorDiv),
        ("mod", Self::Mod),
        ("bitand", Self::BitAnd),
        ("bitor", Self::BitOr),
        ("bitxor", Self::BitXor),
        ("shl", Self::ShiftLeft),
        ("shr", Self::ShiftRight),
        ("cmplt", Self::CompareLess),
        ("lt", Self::Less),
        ("gt", Self::Greater),
        ("gte", Self::GreaterEqual),
        ("lte", Self::LessEqual),
        ("eq", Self::Equal),
        ("neq", Self::NotEqual),
        ("and", Self::And),
        ("or", Self::Or),
        ("not", Self::Not),
        ("neg", Self::Neg),
        ("string_concat", Self::StringConcat),
        ("string_trim", Self::StringTrim),
        ("reshape", Self::Reshape),
        ("string_slice", Self::StringSlice),
        ("string_contains", Self::StringContains),
        ("string_starts_with", Self::StringStartsWith),
        ("string_ends_with", Self::StringEndsWith),
        ("string_len", Self::StringLen),
        ("char_code", Self::CharCode),
        ("char_from_code", Self::CharFromCode),
        ("to_string", Self::ToString),
        ("to_int", Self::ToInt),
        ("to_float", Self::ToFloat),
        ("tensor_to_scalar", Self::TensorToScalar),
        ("scalar_to_tensor", Self::ScalarToTensor),
        ("len", Self::Len),
        ("range", Self::Range),
        ("rank", Self::Rank),
        ("shape", Self::Shape),
        ("numel", Self::Numel),
        ("sqrt", Self::Sqrt),
        ("exp", Self::Exp),
        ("log", Self::Log),
        ("sin", Self::Sin),
        ("cos", Self::Cos),
        ("tan", Self::Tan),
        ("atan", Self::Atan),
        ("tanh", Self::Tanh),
        ("relu", Self::Relu),
        ("sigmoid", Self::Sigmoid),
        ("silu", Self::Silu),
        ("gelu", Self::Gelu),
        ("floor", Self::Floor),
        ("ceil", Self::Ceil),
        ("round", Self::Round),
        ("recip", Self::Recip),
        ("abs", Self::Abs),
        ("min", Self::Min),
        ("max", Self::Max),
        ("min_elem", Self::MinElem),
        ("max_elem", Self::MaxElem),
    ];

    fn decode(name: &str) -> Result<Self, Unsupported> {
        match Self::NAMES.iter().find(|(candidate, _)| *candidate == name) {
            Some((_, builtin)) => Ok(*builtin),
            None => Err(Unsupported::new(
                UnsupportedKind::Builtin(name.to_string()),
                "`chelis build` host emission",
                Stage::Codegen("c"),
                chelis_types::deliberate_rejection!(
                    "[04-TOT-2]",
                    "the checked builtin vocabulary and C expression vocabulary disagree; \
                     no fallback expression is permitted"
                ),
            )
            .with_supported_alternative("run this program with `chelis eval`")),
        }
    }

    /// [04-NUM-2]'s NaN finalization of the builtin's float result, or `None`
    /// when it produces no float value of its own (integer, bool, string and
    /// shape builtins, and transport that carries stored bits). Exhaustive,
    /// so a new builtin does not compile until it is classified. `min` and
    /// `max` lower to C `fmin`/`fmax`, which no atom declares bit-preserving.
    fn nan_finalization(self) -> Option<crate::fp_env::NanFinalization> {
        use crate::fp_env::NanFinalization;
        match self {
            Self::Add
            | Self::Sub
            | Self::Mul
            | Self::Div
            | Self::FloorDiv
            | Self::Mod
            | Self::Neg
            | Self::Sqrt
            | Self::Exp
            | Self::Log
            | Self::Sin
            | Self::Cos
            | Self::Tan
            | Self::Atan
            | Self::Tanh
            | Self::Sigmoid
            | Self::Silu
            | Self::Gelu
            | Self::Floor
            | Self::Ceil
            | Self::Round
            | Self::Recip
            | Self::Abs
            | Self::Min
            | Self::Max => Some(NanFinalization::Canonical),
            Self::Relu | Self::MinElem | Self::MaxElem => Some(NanFinalization::BitPreserving),
            Self::TruncDiv
            | Self::BitAnd
            | Self::BitOr
            | Self::BitXor
            | Self::ShiftLeft
            | Self::ShiftRight
            | Self::CompareLess
            | Self::Less
            | Self::Greater
            | Self::GreaterEqual
            | Self::LessEqual
            | Self::Equal
            | Self::NotEqual
            | Self::And
            | Self::Or
            | Self::Not
            | Self::StringConcat
            | Self::StringTrim
            | Self::Reshape
            | Self::StringSlice
            | Self::StringContains
            | Self::StringStartsWith
            | Self::StringEndsWith
            | Self::StringLen
            | Self::CharCode
            | Self::CharFromCode
            | Self::ToString
            | Self::ToInt
            | Self::ToFloat
            | Self::TensorToScalar
            | Self::ScalarToTensor
            | Self::Len
            | Self::Range
            | Self::Rank
            | Self::Shape
            | Self::Numel => None,
        }
    }
}

/// The host scalar lane's NaN finalization inventory: every builtin name it
/// emits with the classification its finalization point applies. Oracles
/// derive their cases from this list.
pub fn host_builtin_nan_inventory() -> Vec<(&'static str, Option<crate::fp_env::NanFinalization>)> {
    CExpressionBuiltin::NAMES
        .iter()
        .map(|(name, builtin)| (*name, builtin.nan_finalization()))
        .collect()
}

use crate::emit::CEmitter;
use crate::emitted_expr::{BinaryOperator, EmittedExpr, UnaryOperator};

/// Emit a C string literal whose bytes are unambiguous in every following
/// lexical context. Fixed-width three-digit octal escapes preserve embedded
/// NUL and cannot absorb an adjacent decimal or hexadecimal digit.
pub(crate) fn c_utf8_byte_literal(value: &str) -> String {
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

/// The constructor name a compiled ADT of type `ty` stores, and the name
/// every tag test compares it against: the declared source spelling. A
/// package build gives each top-level name the reef linker's private
/// qualification (spec/04-type-system.md, "Reserved linker name format");
/// that identity stays inside the compiler, so every exit that renders the
/// stored name ([05-OP-32]) prints the constructor as `chelis eval` does
/// (chelis#2880). The spelling comes from removing the data type's own
/// module qualification, which keeps every authored `__` and therefore keeps
/// a type's distinct constructors distinct; a tag is only compared against
/// constructors of its value's own type. A constructor that does not carry
/// its type's qualification keeps the name it has.
fn stored_constructor_name(ty: &HostType, ctor: &str) -> String {
    let source = match ty {
        HostType::Adt(type_name, _) => {
            chelis_types::linked_constructor_source_name(type_name, ctor)
        }
        _ => None,
    };
    source.unwrap_or(ctor).to_string()
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
    HostSiteId, VerifiedApplyKind, VerifiedBlockId, VerifiedEdgeView, VerifiedHostAction,
    VerifiedHostOperation, VerifiedHostTensorHelperView, VerifiedHostTerminator,
    VerifiedOperationId, VerifiedOwnerId, VerifiedOwnershipUse, VerifiedTerminalView,
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
    // runtime headers they transitively require (`chelis_blas.h` when a helper
    // calls BLAS).  The preamble is then assembled with the right includes and
    // prepended.  Without this, the include-stripping in `append_helper`
    // silently drops the inner emitter's include and the resulting `main.c`
    // calls a function with no declaration in scope.
    let mut body: Vec<String> = Vec::new();
    let mut helper_requirements = HelperRequirements::default();
    append_scalar_conversion_helpers(&mut body);
    body.push(String::new());
    append_tensor_abi_helpers(&mut body);
    body.push(String::new());
    append_tensor_reshape_helper(&mut body);
    body.push(String::new());
    // The key helpers precede the tensor printer, which reads a key tensor's
    // elements through `chelis_key_at`.
    append_uniform_sample_helper(&mut body);
    body.push(String::new());
    let key_callable_helpers_index = body.len();
    append_tensor_print_helper(&mut body);
    body.push(String::new());
    append_option_print_helper(&mut body);
    body.push(String::new());
    let math_helpers_index = body.len();
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
    let captured_globals = chelis_ir::host::captured_global_names(program);
    if !captured_globals.is_empty() {
        body.push("// Top-level bindings captured by compiled functions (issue #352):".to_string());
        for name in &captured_globals {
            let source = chelis_ir::LoadStoreName::top_level_source_for_label(name)
                .map_err(|reason| invalid_abi_shape(reason, "captured global identity"))?
                .unwrap_or_else(|| name.clone());
            let binding = program
                .globals
                .iter()
                .find(|binding| binding.name == source)
                .expect("captured global name comes from program.globals");
            body.push(format!("static {};", c_decl(&binding.ty, name)?));
        }
        body.push(String::new());
    }

    let global_entry_coverage = entry::global_helper_coverage(program);
    for index in 0..program.global_tensor_helpers.len() {
        let helper_name = global_tensor_helper_name(program_name, index);
        if external_helpers.contains(&helper_name) {
            append_external_helper_declaration(&mut body, &helper_name);
        } else {
            for (variant, coverage) in global_entry_coverage.variants[index].iter().enumerate() {
                helper_requirements.merge(append_helper(
                    &mut body,
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

    // chelis#2506: every function's nested-value entry work, and the walkers
    // its exported entry calls, before any function that calls one.
    let mut entry_walkers = entry_walk::EntryWalkers::new(program)?;
    let entry_work = program
        .functions
        .iter()
        .map(|function| entry_walkers.entry_work(function))
        .collect::<Result<Vec<_>, _>>()?;
    entry_walkers.render(&mut body);
    let entry_groups = entry_receipt_groups(program);
    for (index, function) in program.functions.iter().enumerate() {
        if entry_groups.get(&function.name) == Some(&index) {
            body.push(format!(
                "static const char __chelis_entry_contract_token_{index} = 0;"
            ));
        }
    }

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
        let helper_output_types = verified_helpers
            .iter()
            .map(|verified| CEmitter::output_types(verified.dag()))
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
            &helper_output_types,
            &verified_helpers,
            external_helpers,
            &captured_globals,
            &entry_work[function_index],
            &entry_groups,
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
        for index in 0..function.tensor_helpers.len() {
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

    if !program.globals.is_empty() {
        let hoisted: UnordSet<&str> = captured_globals.iter().map(String::as_str).collect();
        let helper_output_types = (0..program.global_tensor_helpers.len())
            .map(|helper| {
                let verified = projected
                    .global_tensor_helper(helper)
                    .expect("projected global helper retains verified child");
                CEmitter::output_types(verified.dag())
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
            &helper_output_types,
            helper_result_origins,
            external_helpers,
        )?;
    }

    // The scalar activation helpers precede every authored body and are
    // emitted only when one calls them, so the unit carries only the
    // correctly rounded kernels it uses. This splice lies after the key
    // callback index, so it runs first and leaves that index valid.
    splice_tensor_math_helpers(&mut body, math_helpers_index);

    // Insert private key callback support only when a selected body needs
    // its closed carrier or function symbol, before any such declaration.
    if body.iter().any(|line| {
        line.trim().split_once(" = ").is_some_and(|(_, value)| {
            value == "(chelis_key_callable){ NULL };" || value.starts_with("__chelis_key_callable_")
        })
    }) {
        let mut key_helpers = Vec::new();
        append_key_callable_helpers(&mut key_helpers);
        key_helpers.push(String::new());
        body.splice(
            key_callable_helpers_index..key_callable_helpers_index,
            key_helpers,
        );
    }

    let mut result_claim_support = Vec::new();
    append_host_result_claim_support(&mut result_claim_support);
    append_host_result_claim_checks(&mut result_claim_support);
    result_claim_support.push(String::new());
    result_claim_support.extend(body);
    body = result_claim_support;

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
        "#include <stdlib.h>".to_string(),
        "#include <string.h>".to_string(),
    ];
    out.push(String::new());
    out.extend(c_linkage_declarations([
        // chelis#943: emitter-internal accumulator ABI. Deliberately absent
        // from the published chelis_runtime.h (the capacity census governs
        // that surface, and these exist only for compiler-owned accumulators
        // whose refcount-1 exclusivity this emitter proves). The symbols are
        // exported by libchelis_runtime; only the declarations are private.
        "chelis_list *chelis_list_with_capacity(int64_t capacity);".to_string(),
        // chelis#2508: both accumulator steps consume their operand, because
        // the ownership verifier moves every loop step's item into its
        // accumulator. No cloning push is declared, so a verified move
        // cannot be realized as a copy that leaves the moved owner live.
        "void chelis_list_push_moved(chelis_list *list, chelis_value value);".to_string(),
        "void chelis_list_extend_moved(chelis_list *list, chelis_list *src);".to_string(),
        // chelis#2205: the consuming counterparts of the container builtins
        // that may take a same-kind operand the ownership verifier moved at
        // its scheduled last use. Each mutates in place only at strong-owner
        // count one and otherwise clones and releases the consumed input, so
        // a retained alias is never mutated. Private for the same reason as
        // the three above.
        "chelis_list *chelis_list_append_owned(chelis_list *list, chelis_value value);".to_string(),
        "chelis_list *chelis_list_concat_owned(chelis_list *lhs, const chelis_list *rhs);"
            .to_string(),
        // chelis#2334: the consuming `skip`. Private for the same reason,
        // and it keeps the `chelis_list_drop` stem of the published
        // cloning symbol it pairs with.
        "chelis_list *chelis_list_drop_owned(chelis_list *list, int64_t count);".to_string(),
        "chelis_dict *chelis_dict_insert_owned(chelis_dict *dict, chelis_value key, chelis_value value);"
            .to_string(),
        "chelis_dict *chelis_dict_merge_owned(chelis_dict *lhs, const chelis_dict *rhs);"
            .to_string(),
        "chelis_dict *chelis_dict_remove_owned(chelis_dict *dict, chelis_value key);".to_string(),
        "chelis_string chelis_string_concat_owned(chelis_string lhs, chelis_string rhs);"
            .to_string(),
    ]));
    if helper_requirements.needs_blas_header {
        out.push("#include \"chelis_blas.h\"".to_string());
        out.push(CEmitter::blas_integer_support());
    }
    out.extend(crate::fp_env::helper_lines().map(str::to_string));
    out.push(String::new());
    out.extend(body);
    Ok(crate::fp_env::prune_unused_nan_helpers(&out.join("\n")))
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

/// The `[05-RNG-1]` stream and `[05-OP-8]` samplers, byte-identical to the
/// block `CEmitter` prepends to a standalone kernel (chelis#2408), then
/// `[05-RNG-2]`'s key derivation, which a kernel carries only when it derives
/// keys.
fn append_uniform_sample_helper(out: &mut Vec<String>) {
    for line in [
        "static inline uint64_t chelis_random_mix(uint64_t value) {",
        "    value += 0x9E3779B97F4A7C15ULL;",
        "    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;",
        "    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;",
        "    return value ^ (value >> 31);",
        "}",
        "static inline double chelis_random_unit(uint64_t key, uint64_t index) {",
        "    uint64_t element = chelis_random_mix(index);",
        "    uint64_t word = chelis_random_mix(key ^ ((element << 41) | (element >> 23)));",
        "    return (double)(word >> 11) / (double)(1ULL << 53);",
        "}",
        "static inline float chelis_uniform_sample_f32(uint64_t key, uint64_t index, float low, float high) {",
        "    return fmaf(high - low, (float)chelis_random_unit(key, index), low);",
        "}",
        "static inline double chelis_uniform_sample_f64(uint64_t key, uint64_t index, double low, double high) {",
        "    return fma(high - low, chelis_random_unit(key, index), low);",
        "}",
        "static inline uint64_t chelis_key_derive(uint64_t key, uint64_t index) {",
        "    uint64_t mixed = chelis_random_mix(index);",
        "    return chelis_random_mix(key ^ ((mixed << 29) | (mixed >> 35)));",
        "}",
    ] {
        out.push(line.to_string());
    }
    // A host scalar key ([05-OP-69]..[05-OP-72]), `chelis_runtime.h`'s
    // `chelis_key`, and its rank-0 key tensor, the form a kernel's key
    // `Load` reads and a boxed key takes.
    for line in [
        "static inline chelis_key chelis_key_from_seed_bits(long long seed) {",
        "    chelis_key key = { (unsigned long long)seed };",
        "    return key;",
        "}",
        "static inline chelis_key chelis_key_derive_value(chelis_key key, unsigned long long index) {",
        "    chelis_key derived = { chelis_key_derive(key.bits, index) };",
        "    return derived;",
        "}",
        "static inline chelis_key chelis_key_fold_in(chelis_key key, long long n) {",
        "    return chelis_key_derive_value(chelis_key_derive_value(key, 2ULL), (unsigned long long)n);",
        "}",
        "static inline chelis_tensor *chelis_key_tensor(chelis_key key) {",
        "    chelis_tensor *tensor = chelis_alloc(0, NULL, CHELIS_DTYPE_KEY);",
        "    chelis_tensor_write *guard = chelis_tensor_begin_write(tensor);",
        "    chelis_write_view view = chelis_tensor_write_view(guard);",
        "    ((unsigned long long *)view.data)[0] = key.bits;",
        "    chelis_tensor_end_write(guard);",
        "    return tensor;",
        "}",
        "static inline chelis_key chelis_key_of_tensor(const chelis_tensor *tensor) {",
        "    chelis_read_view view = chelis_tensor_read_view(tensor);",
        "    if (view.dtype != CHELIS_DTYPE_KEY || view.count != 1) {",
        "        fprintf(stderr, \"internal: a scalar key is not a rank-0 key tensor\\n\");",
        "        abort();",
        "    }",
        "    chelis_key key = { ((const unsigned long long *)view.data)[0] };",
        "    return key;",
        "}",
        "static inline chelis_key chelis_key_at(const chelis_tensor *tensor, int64_t index) {",
        "    chelis_read_view view = chelis_tensor_read_view(tensor);",
        "    if (view.dtype != CHELIS_DTYPE_KEY || index < 0 || index >= view.count) {",
        "        fprintf(stderr, \"internal: no key at this index of a key tensor\\n\");",
        "        abort();",
        "    }",
        "    chelis_key key = { ((const unsigned long long *)view.data)[index] };",
        "    return key;",
        "}",
        "static inline chelis_key chelis_key_take_value(chelis_value value) {",
        "    chelis_tensor *tensor = chelis_tensor_take_value(value);",
        "    chelis_key key = chelis_key_of_tensor(tensor);",
        "    chelis_tensor_release(tensor);",
        "    return key;",
        "}",
        "static inline chelis_tensor *chelis_split_keys_tensor(chelis_key key, long long count) {",
        "    if (count < 0) chelis_numeric_trap(\"numeric trap: domain in split_keys at i64\");",
        "    int64_t extent = (int64_t)count;",
        "    chelis_tensor *tensor = chelis_alloc(1, &extent, CHELIS_DTYPE_KEY);",
        "    chelis_tensor_write *guard = chelis_tensor_begin_write(tensor);",
        "    chelis_write_view view = chelis_tensor_write_view(guard);",
        "    for (long long j = 0; j < count; j++) {",
        "        ((unsigned long long *)view.data)[j] = chelis_key_fold_in(key, j).bits;",
        "    }",
        "    chelis_tensor_end_write(guard);",
        "    return tensor;",
        "}",
    ] {
        out.push(line.to_string());
    }
}

/// Capture-free C entry points for the four checked key operations when they
/// occur as values. Their names are compiler-private; host IR retains the
/// builtin identity and the checked function type selects one exact surface.
/// The tensor entries accept runtime rank and extents because rank-polymorphic
/// aliases may be specialized at more than one call site.
fn append_key_callable_helpers(out: &mut Vec<String>) {
    out.push(
        r#"
// The operation identity is in the checked host type. This private value is
// only an ABI witness; independent calls select concrete callback surfaces.
typedef struct { const void *zero; } chelis_key_callable;

static inline chelis_tuple *__chelis_key_pair(chelis_key left, chelis_key right) {
    chelis_value halves[2] = {
        chelis_value_take_tensor(chelis_key_tensor(left)),
        chelis_value_take_tensor(chelis_key_tensor(right))
    };
    chelis_tuple *pair = chelis_tuple_from_values(halves, 2);
    chelis_value_release(halves[0]);
    chelis_value_release(halves[1]);
    return pair;
}

// Shape scratch stays in a checked i64 tensor; rank never sizes a C stack
// array. Key storage is allocated by its own dtype, never a scalar exemplar.
static inline chelis_tensor *__chelis_key_callable_alloc_shape(
    const chelis_tensor *input, bool append_count, int64_t count) {
    int64_t rank = chelis_tensor_rank(input);
    if (append_count && rank >= INT32_MAX)
        chelis_numeric_trap("numeric trap: overflow in split_keys at i64");
    int64_t shape_count = rank + (append_count ? 1 : 0);
    chelis_tensor *shape_storage = chelis_alloc(1, &shape_count, CHELIS_DTYPE_I64);
    chelis_tensor_write *shape_guard = chelis_tensor_begin_write(shape_storage);
    chelis_write_view shape_view = chelis_tensor_write_view(shape_guard);
    int64_t *shape = (int64_t *)shape_view.data;
    for (int64_t axis = 0; axis < rank; ++axis) shape[axis] = chelis_tensor_shape(input, axis);
    if (append_count) shape[rank] = count;
    chelis_tensor_end_write(shape_guard);
    chelis_read_view checked_shape = chelis_tensor_read_view(shape_storage);
    chelis_tensor *out = chelis_alloc((int32_t)shape_count, (const int64_t *)checked_shape.data, CHELIS_DTYPE_KEY);
    chelis_tensor_release(shape_storage);
    return out;
}

static inline chelis_key __chelis_key_callable_seed_scalar(int64_t seed) {
    return chelis_key_from_seed_bits(seed);
}
static inline chelis_tuple *__chelis_key_callable_split_scalar(chelis_key key) {
    return __chelis_key_pair(
        chelis_key_derive_value(key, 0ULL),
        chelis_key_derive_value(key, 1ULL));
}
static inline chelis_tensor *__chelis_key_callable_children_scalar(chelis_key key, int64_t count) {
    return chelis_split_keys_tensor(key, count);
}
static inline chelis_key __chelis_key_callable_fold_scalar(chelis_key key, int64_t index) {
    return chelis_key_fold_in(key, index);
}
static inline chelis_tensor *__chelis_key_callable_seed_tensor(chelis_tensor *seeds) {
    chelis_read_view input = chelis_tensor_read_view(seeds);
    if (input.dtype != CHELIS_DTYPE_I64) abort();
    chelis_tensor *out = __chelis_key_callable_alloc_shape(seeds, false, 0);
    chelis_tensor_write *guard = chelis_tensor_begin_write(out);
    chelis_write_view view = chelis_tensor_write_view(guard);
    for (int64_t i = 0; i < input.count; ++i)
        ((uint64_t *)view.data)[i] = (uint64_t)((const int64_t *)input.data)[i];
    chelis_tensor_end_write(guard);
    chelis_tensor_release(seeds);
    return out;
}
static inline chelis_tuple *__chelis_key_callable_split_tensor(chelis_tensor *keys) {
    chelis_read_view input = chelis_tensor_read_view(keys);
    if (input.dtype != CHELIS_DTYPE_KEY) abort();
    chelis_tensor *left = __chelis_key_callable_alloc_shape(keys, false, 0);
    chelis_tensor *right = __chelis_key_callable_alloc_shape(keys, false, 0);
    chelis_tensor_write *left_guard = chelis_tensor_begin_write(left);
    chelis_tensor_write *right_guard = chelis_tensor_begin_write(right);
    chelis_write_view left_view = chelis_tensor_write_view(left_guard);
    chelis_write_view right_view = chelis_tensor_write_view(right_guard);
    for (int64_t i = 0; i < input.count; ++i) {
        uint64_t source = ((const uint64_t *)input.data)[i];
        ((uint64_t *)left_view.data)[i] = chelis_key_derive(source, 0ULL);
        ((uint64_t *)right_view.data)[i] = chelis_key_derive(source, 1ULL);
    }
    chelis_tensor_end_write(left_guard);
    chelis_tensor_end_write(right_guard);
    chelis_value halves[2] = { chelis_value_take_tensor(left), chelis_value_take_tensor(right) };
    chelis_tuple *pair = chelis_tuple_from_values(halves, 2);
    chelis_value_release(halves[0]);
    chelis_value_release(halves[1]);
    chelis_tensor_release(keys);
    return pair;
}
static inline chelis_tensor *__chelis_key_callable_children_tensor(chelis_tensor *keys, int64_t count) {
    if (count < 0) chelis_numeric_trap("numeric trap: domain in split_keys at i64");
    chelis_read_view input = chelis_tensor_read_view(keys);
    if (input.dtype != CHELIS_DTYPE_KEY) abort();
    chelis_tensor *out = __chelis_key_callable_alloc_shape(keys, true, count);
    chelis_tensor_write *guard = chelis_tensor_begin_write(out);
    chelis_write_view view = chelis_tensor_write_view(guard);
    for (int64_t i = 0; i < input.count; ++i) {
        uint64_t source = ((const uint64_t *)input.data)[i];
        for (int64_t j = 0; j < count; ++j)
            ((uint64_t *)view.data)[i * count + j] =
                chelis_key_derive(chelis_key_derive(source, 2ULL), (uint64_t)j);
    }
    chelis_tensor_end_write(guard);
    chelis_tensor_release(keys);
    return out;
}
static inline chelis_tensor *__chelis_key_callable_fold_tensor(chelis_tensor *keys, chelis_tensor *indices) {
    chelis_read_view input = chelis_tensor_read_view(keys);
    chelis_read_view index = chelis_tensor_read_view(indices);
    int64_t rank = chelis_tensor_rank(keys);
    if (input.dtype != CHELIS_DTYPE_KEY || index.dtype != CHELIS_DTYPE_I64
        || chelis_tensor_rank(indices) != rank) abort();
    for (int64_t axis = 0; axis < rank; ++axis) {
        if (chelis_tensor_shape(indices, axis) != chelis_tensor_shape(keys, axis))
            chelis_numeric_trap("numeric trap: domain in fold_in at i64");
    }
    chelis_tensor *out = __chelis_key_callable_alloc_shape(keys, false, 0);
    chelis_tensor_write *guard = chelis_tensor_begin_write(out);
    chelis_write_view view = chelis_tensor_write_view(guard);
    for (int64_t i = 0; i < input.count; ++i)
        ((uint64_t *)view.data)[i] = chelis_key_derive(
            chelis_key_derive(((const uint64_t *)input.data)[i], 2ULL),
            (uint64_t)((const int64_t *)index.data)[i]);
    chelis_tensor_end_write(guard);
    chelis_tensor_release(keys);
    chelis_tensor_release(indices);
    return out;
}
"#
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
        "        chelis_list_push_moved(result, entry);",
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
/// Surf `def` reach this emitter after host-ABI projection instead. Both
/// lanes take one definition: each composite helper body is emitted from the
/// very graph `tier2` builds for spec/05 §3.3, one finalized C statement per
/// primitive, so the scalar and tensor results cannot drift apart. `tanh` is
/// the [05-OP-46] primitive and `relu` is [05-OP-43]'s selection. Reduced
/// floats need distinct per-node finalizers even though both compute as C
/// `float`, so one generated specialization cannot serve every source dtype.
fn append_activation_helpers(
    out: &mut Vec<String>,
    prim: Prim,
    suffix: &str,
    finalizer: Option<&str>,
    used: &dyn Fn(&str) -> bool,
) {
    let c_type = if prim == Prim::F64 { "double" } else { "float" };
    let zero = if prim == Prim::F64 { "0.0" } else { "0.0f" };
    let tanh = if prim == Prim::F64 {
        "chelis_cr_tanh"
    } else {
        "chelis_cr_tanhf"
    };
    let finalize = |expr: String| match finalizer {
        Some(function) => format!("{function}({expr})"),
        None => expr,
    };

    if used(&format!("chelis_host_relu_{suffix}")) {
        out.push(format!(
            "static inline {c_type} chelis_host_relu_{suffix}({c_type} x) {{"
        ));
        out.push(format!("    return x < {zero} ? {zero} : x;"));
        out.push("}".to_string());
    }

    if used(&format!("chelis_host_tanh_{suffix}")) {
        out.push(format!(
            "static inline {c_type} chelis_host_tanh_{suffix}({c_type} x) {{"
        ));
        out.push(format!("    return {};", finalize(format!("{tanh}(x)"))));
        out.push("}".to_string());
    }

    let lowerings: [(&str, ActivationLowering); 3] = [
        ("sigmoid", chelis_ir::tier2::lower_sigmoid),
        ("silu", chelis_ir::tier2::lower_silu),
        ("gelu", chelis_ir::tier2::lower_gelu),
    ];
    for (name, lower) in lowerings {
        if !used(&format!("chelis_host_{name}_{suffix}")) {
            continue;
        }
        out.push(format!(
            "static inline {c_type} chelis_host_{name}_{suffix}({c_type} x) {{"
        ));
        out.extend(activation_body(prim, lower, &finalize));
        out.push("}".to_string());
    }
}

type ActivationLowering = fn(
    chelis_ir::dag::Owner,
    &mut chelis_ir::dag::Dag,
    chelis_ir::dag::NodeId,
    &TensorType,
    Option<&str>,
) -> chelis_ir::dag::NodeId;

/// The C statements of one activation helper: the `tier2` lowering of a
/// rank-0 `x` at `prim`, one finalized statement per primitive, in the
/// graph's construction order (every operand precedes its user).
fn activation_body(
    prim: Prim,
    lower: ActivationLowering,
    finalize: &dyn Fn(String) -> String,
) -> Vec<String> {
    let ty = chelis_ir::tier2::scalar_type(prim);
    let mut dag = chelis_ir::dag::Dag::new();
    let owner = chelis_ir::dag::Owner::from(dag.declare("activation"));
    let x = dag.add_node(
        owner,
        RiscOp::Load { name: "x".into() },
        vec![],
        ty.clone(),
        None,
    );
    let root = lower(owner, &mut dag, x, &ty, None);
    let c_type = if prim == Prim::F64 { "double" } else { "float" };
    let kernel = |f32_name: &str, f64_name: &str| {
        if prim == Prim::F64 {
            f64_name.to_string()
        } else {
            f32_name.to_string()
        }
    };
    let name = |id: chelis_ir::dag::NodeId| {
        if id == x {
            "x".to_string()
        } else {
            format!("v{}", id.0)
        }
    };
    let mut lines = Vec::new();
    for (index, node) in dag.nodes().iter().enumerate() {
        let id = chelis_ir::dag::NodeId(index);
        if id == x {
            continue;
        }
        let input = |position: usize| name(node.inputs[position]);
        let value = match &node.op {
            RiscOp::Const { value } => {
                // The constant is already finalized at `prim`; f16, bf16,
                // and f32 values are exact as C `float`.
                let wide = value.as_f64_lossy();
                if prim == Prim::F64 {
                    format!("chelis_f64_from_bits(UINT64_C(0x{:016x}))", wide.to_bits())
                } else {
                    format!(
                        "chelis_f32_from_bits(UINT32_C(0x{:08x}))",
                        exact_f64_to_f32(wide).to_bits()
                    )
                }
            }
            RiscOp::Neg => finalize(format!("-{}", input(0))),
            RiscOp::Exp => finalize(format!(
                "{}({})",
                kernel("chelis_cr_expf", "chelis_cr_exp"),
                input(0)
            )),
            RiscOp::Add => finalize(format!("{} + {}", input(0), input(1))),
            RiscOp::Mul => finalize(format!("{} * {}", input(0), input(1))),
            RiscOp::Recip => {
                let one = if prim == Prim::F64 { "1.0" } else { "1.0f" };
                finalize(format!("{one} / {}", input(0)))
            }
            other => unreachable!(
                "spec/05 §3.3 activation lowering produced `{other:?}`, which the \
                 host helper emitter has no C spelling for"
            ),
        };
        lines.push(format!("    {c_type} {} = {value};", name(id)));
    }
    lines.push(format!("    return {};", name(root)));
    lines
}

/// A finalized f16, bf16, or f32 constant's f64 image narrowed to C `float`.
/// The image is exactly representable, so the narrowing is exact.
fn exact_f64_to_f32(wide: f64) -> f32 {
    let narrow = wide as f32;
    debug_assert!(f64::from(narrow).to_bits() == wide.to_bits() || wide.is_nan());
    narrow
}

/// Insert at `index` the finalizers and every activation helper `body` calls.
fn splice_tensor_math_helpers(body: &mut Vec<String>, index: usize) {
    let mut helpers = Vec::new();
    append_tensor_math_helpers(&mut helpers, &|name| {
        let call = format!("{name}(");
        body.iter().any(|line| line.contains(&call))
    });
    helpers.push(String::new());
    body.splice(index..index, helpers);
}

/// Append the reduced-float finalizers and each activation helper `used`
/// selects by name.
fn append_tensor_math_helpers(out: &mut Vec<String>, used: &dyn Fn(&str) -> bool) {
    out.push("static inline float chelis_host_finalize_f16(float x) {".to_string());
    out.push("    return chelis_f16_to_f32(chelis_f32_to_f16(x));".to_string());
    out.push("}".to_string());
    out.push("static inline float chelis_host_finalize_bf16(float x) {".to_string());
    out.push("    return chelis_bf16_to_f32(chelis_f32_to_bf16(x));".to_string());
    out.push("}".to_string());
    append_activation_helpers(
        out,
        Prim::F16,
        "f16",
        Some("chelis_host_finalize_f16"),
        used,
    );
    append_activation_helpers(
        out,
        Prim::Bf16,
        "bf16",
        Some("chelis_host_finalize_bf16"),
        used,
    );
    append_activation_helpers(out, Prim::F32, "f32", None, used);
    append_activation_helpers(out, Prim::F64, "f64", None, used);
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
            "        return (uint16_t)(target_exponent | (UINT16_C(1) << (mantissa_bits - 1)));",
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
            | Prim::String
            | Prim::Key => unreachable!("ExactToFloat plan has a float target"),
        },
        // [04-NUM-2]: a numeric conversion that produces a NaN finalizes to
        // the target's canonical quiet NaN; the f16 and bf16 narrowing
        // helpers already do.
        CheckedCastKind::FloatToFloat => match target {
            Prim::F64 => crate::fp_env::finalize_float(
                &cast_float_as_double(plan.source(), value),
                true,
                crate::fp_env::NanFinalization::Canonical,
            ),
            Prim::F32 => crate::fp_env::finalize_float(
                &format!("(float)({})", cast_float_as_double(plan.source(), value)),
                false,
                crate::fp_env::NanFinalization::Canonical,
            ),
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
            | Prim::String
            | Prim::Key => unreachable!("FloatToFloat plan has a float target"),
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
        | Prim::String
        | Prim::Key => unreachable!("float checked-cast action has a float source"),
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
        | Prim::String
        | Prim::Key => unreachable!("integer checked-cast action has an integer target"),
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
        | Prim::String
        | Prim::Key => unreachable!("integer checked-cast action has an integer target"),
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
        Prim::F8e4m3 | Prim::String | Prim::Key => {
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

/// The only C callback signatures a resolved key operation can carry.
fn key_callable_symbol(name: &str, ty: &HostType) -> Option<&'static str> {
    let HostType::Callback(params, result) = ty else {
        return None;
    };
    let scalar = match (name, params.as_slice(), result.as_ref()) {
        ("key_from_seed", [HostType::Int64], HostType::Key) => {
            Some("__chelis_key_callable_seed_scalar")
        }
        ("split_key", [HostType::Key], HostType::Tuple(halves))
            if matches!(halves.as_slice(), [HostType::Key, HostType::Key]) =>
        {
            Some("__chelis_key_callable_split_scalar")
        }
        ("split_keys", [HostType::Key, HostType::Int64], HostType::Tensor(out))
            if out.precision == Prim::Key =>
        {
            Some("__chelis_key_callable_children_scalar")
        }
        ("fold_in", [HostType::Key, HostType::Int64], HostType::Key) => {
            Some("__chelis_key_callable_fold_scalar")
        }
        _ => None,
    };
    if scalar.is_some() {
        return scalar;
    }
    match (name, params.as_slice(), result.as_ref()) {
        ("key_from_seed", [HostType::Tensor(seeds)], HostType::Tensor(out))
            if seeds.precision == Prim::Int64 && out.precision == Prim::Key =>
        {
            Some("__chelis_key_callable_seed_tensor")
        }
        ("split_key", [HostType::Tensor(keys)], HostType::Tuple(halves))
            if keys.precision == Prim::Key
                && matches!(
                    halves.as_slice(),
                    [HostType::Tensor(left), HostType::Tensor(right)]
                        if left.precision == Prim::Key && right.precision == Prim::Key
                ) =>
        {
            Some("__chelis_key_callable_split_tensor")
        }
        ("split_keys", [HostType::Tensor(keys), HostType::Int64], HostType::Tensor(out))
            if keys.precision == Prim::Key && out.precision == Prim::Key =>
        {
            Some("__chelis_key_callable_children_tensor")
        }
        ("fold_in", [HostType::Tensor(keys), HostType::Tensor(indices)], HostType::Tensor(out))
            if keys.precision == Prim::Key
                && indices.precision == Prim::Int64
                && out.precision == Prim::Key =>
        {
            Some("__chelis_key_callable_fold_tensor")
        }
        _ => None,
    }
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
    // [05-OBS-2]: a key has no scalar carrier; its element renders through
    // the runtime's key text.
    out.push("    if (chelis_tensor_read_view(t).dtype == CHELIS_DTYPE_KEY) {".to_string());
    out.push(
        "        chelis_string key_text = chelis_string_from_key(chelis_key_at(t, i));".to_string(),
    );
    out.push("        chelis_print_string(key_text);".to_string());
    out.push("        chelis_string_release(key_text);".to_string());
    out.push("        return;".to_string());
    out.push("    }".to_string());
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

/// Render an option at a top-level exit (chelis#2597). [05-OP-32] spells an
/// option node exactly as its ADT rule spells a one-field `Some` constructor,
/// and `None` as the bare constructor name, so a present payload renders
/// through the runtime's own recursive ADT rendering: the bytes match an
/// option nested inside a list, tuple or data-type value.
fn append_option_print_helper(out: &mut Vec<String>) {
    out.extend(
        [
            "static void chelis_host_print_option(const chelis_option *option) {",
            "    if (!chelis_option_is_some(option)) {",
            "        printf(\"None\");",
            "        return;",
            "    }",
            "    chelis_value payload = chelis_option_unwrap(option);",
            "    chelis_string ctor = chelis_string_from_cstr(\"Some\");",
            "    chelis_adt *some = chelis_adt_construct(ctor, &payload, 1);",
            "    chelis_string_release(ctor);",
            "    chelis_value_release(payload);",
            "    chelis_print_adt(some);",
            "    chelis_adt_release(some);",
            "}",
        ]
        .map(str::to_string),
    );
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
}

impl HelperRequirements {
    fn merge(&mut self, other: Self) {
        self.needs_blas_header |= other.needs_blas_header;
    }
}

/// Append a tensor helper to `out` and return the runtime headers required by
/// the inner emitter. The caller propagates these headers to the host preamble
/// so each one is emitted exactly once at file scope.
fn append_helper(
    out: &mut Vec<String>,
    verified: VerifiedHostTensorHelperView<'_>,
    helper_name: &str,
    entry_coverage: &[chelis_ir::axis_sources::EntryExtentGuard],
) -> Result<HelperRequirements, Unsupported> {
    let helper_name = private_helper_name(helper_name);
    if verified.identity_input().is_some() {
        out.push(format!(
            "static void {}({}) {{",
            helper_name,
            private_host_params(
                "chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out"
            ),
        ));
        out.push("    (void)n_in;".to_string());
        out.push("    (void)n_out;".to_string());
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
    let uses_blas = dag
        .nodes()
        .iter()
        .any(|node| matches!(node.op, RiscOp::BlasMatmul { .. }));
    let options = crate::CodegenOptions {
        use_blas: uses_blas,
        static_entry: true,
    };
    let helper_src =
        CEmitter::emit_verified_dag_with_options(dag, &helper_name, options, entry_coverage)?;
    // The CEmitter prepends dtype-specific uniform sampling helpers to
    // every DAG it emits so that a standalone-emitted kernel
    // stays self-contained. When multiple helpers get concatenated into a
    // single `main.c` that duplicates the definition and gcc rejects the
    // redefinition. We filter the prelude out here and rely on
    // `emit_host_program` to emit exactly one copy at file scope.
    let mut skipping_helper_prelude = false;
    let mut requirements = HelperRequirements::default();
    for line in helper_src.lines() {
        if line.starts_with("#include ") {
            if line.contains("\"chelis_blas.h\"") {
                requirements.needs_blas_header = true;
            }
            continue;
        }
        if line == "/* CHELIS_UNIFORM_HELPERS_BEGIN */" || line == crate::fp_env::HELPERS_BEGIN {
            skipping_helper_prelude = true;
            continue;
        }
        if skipping_helper_prelude {
            if line == "/* CHELIS_UNIFORM_HELPERS_END */" || line == crate::fp_env::HELPERS_END {
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

/// The one path for a declaration of a symbol another translation unit
/// defines with C linkage: the runtime archive's emitter-private entries and
/// a peer helper's exported entry. The host program is also compiled as C++
/// (the Metal `.mm` and the HIP `.cpp` are this source), where a bare
/// prototype takes C++ linkage, is name-mangled, and does not link against
/// the C definition (chelis#2582). The guard leaves the C spelling unchanged.
/// Each declaration takes its own prefix `extern "C"` line rather than one
/// braced block: the prefix line is the linkage spelling the generated
/// artifact contract masks before parsing the source as C.
fn c_linkage_declarations(declarations: impl IntoIterator<Item = String>) -> Vec<String> {
    declarations
        .into_iter()
        .flat_map(|declaration| {
            [
                "#ifdef __cplusplus".to_string(),
                "extern \"C\"".to_string(),
                "#endif".to_string(),
                declaration,
            ]
        })
        .collect()
}

/// Declare a helper that a peer translation unit defines. The prototype is
/// the shared tensor-helper ABI, so the wrapper's call site is unchanged
/// whether the body is the C emitter's `static` definition or a device
/// backend's exported entry.
fn append_external_helper_declaration(out: &mut Vec<String>, helper_name: &str) {
    out.extend(c_linkage_declarations([format!(
        "void {helper_name}(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);"
    )]));
    // Peer translation units keep their established ABI. The private adapter
    // gives the helper the private name every host call site uses.
    out.push(format!(
        "static void {}(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out) {{",
        private_helper_name(helper_name),
    ));
    out.push(format!("    {helper_name}(inputs, n_in, outputs, n_out);"));
    out.push("}".to_string());
    out.push(String::new());
}

/// The private name of a tensor helper: the host-owned definition whose
/// signature carries the invocation context, or the adapter over a peer
/// translation unit's exported helper.
fn private_helper_name(name: &str) -> String {
    format!("{name}__private")
}

/// `params` followed by `rest`, with no leading separator when `params` is
/// empty.
fn join_params(params: &str, rest: &str) -> String {
    if params.is_empty() {
        rest.to_string()
    } else {
        format!("{params}, {rest}")
    }
}

fn append_private_host_context_args(args: &mut Vec<String>, entry_receipt: &str) {
    args.push("__chelis_origin_arena".to_string());
    args.push(entry_receipt.to_string());
}

fn append_invocation_origin_context(out: &mut Vec<String>) {
    out.push(
        "    __chelis_host_result_origin_arena __chelis_origin_arena_storage = { NULL, NULL };"
            .to_string(),
    );
    out.push(
        "    __chelis_host_result_origin_arena *__chelis_origin_arena = &__chelis_origin_arena_storage;"
            .to_string(),
    );
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

/// One declaration's ordered result axes. Its invocation frame is forwarded
/// unchanged through branches and calls; the selected producer supplies `<op>`.
struct HostResultClaim {
    rank: usize,
    axes: Vec<(usize, HostResultRequirement)>,
}

/// What one declared result axis requires of the returned value.
enum HostResultRequirement {
    Literal(usize),
    /// A binder the invocation's tensor parameter axis witnesses. Its value
    /// is read once, when the frame is built; the binder and the parameter
    /// axis are retained for the trap context (spec/04 section 4.7).
    Named {
        claim: String,
        parameter: String,
        prepared: String,
        axis: usize,
    },
    /// The first available witness may be a List element or a later direct
    /// parameter. An empty List does not bind the result axis.
    NamedList {
        state: usize,
    },
}

impl HostResultClaim {
    fn from_plan(plan: &chelis_ir::host::HostResultClaimPlan) -> Self {
        use chelis_ir::host::HostResultRequirementPlan;
        Self {
            rank: plan.result().dims.len(),
            axes: plan
                .axes()
                .iter()
                .map(|(axis, requirement)| {
                    let requirement = match requirement {
                        HostResultRequirementPlan::Literal(required) => {
                            HostResultRequirement::Literal(*required)
                        }
                        HostResultRequirementPlan::NamedDirect {
                            claim,
                            source,
                            prepared,
                            axis,
                        } => HostResultRequirement::Named {
                            claim: claim.clone(),
                            parameter: source.clone(),
                            prepared: prepared.clone(),
                            axis: *axis,
                        },
                        HostResultRequirementPlan::NamedList { state } => {
                            HostResultRequirement::NamedList { state: *state }
                        }
                    };
                    (*axis, requirement)
                })
                .collect(),
        }
    }

    /// A named axis resolves to the binder's first witness among the tensor
    /// parameters, in signature order: the canonical side of the entry plan's
    /// comparisons. A binder no tensor parameter declares adds nothing.
    fn of(function: &HostFunction) -> Option<Self> {
        let HostAbiType::Tensor(ty) = &function.ret_ty else {
            return None;
        };
        let named_lists = function.entry_contract.named_list_binders().to_vec();
        let witness = |binder: &str| {
            function.params.iter().find_map(|param| {
                let HostAbiType::Tensor(param_ty) = &param.ty else {
                    return None;
                };
                param_ty
                    .dims
                    .iter()
                    .position(|dim| matches!(dim, DimInfo::Named(name, _) if name == binder))
                    .map(|axis| (param.name.clone(), axis))
            })
        };
        let axes = ty
            .dims
            .iter()
            .enumerate()
            .filter(|(axis, _)| {
                !function
                    .helper_result_claim_axes
                    .contains(&chelis_ir::dag::RtAxis::Lit(
                        i32::try_from(*axis).expect("rank fits i32"),
                    ))
            })
            .filter_map(|(axis, dim)| match dim {
                DimInfo::Lit(required) => Some((axis, HostResultRequirement::Literal(*required))),
                DimInfo::Named(binder, _) if binder != "*" => {
                    if let Some(state) = named_lists.iter().position(|name| name == binder) {
                        return Some((axis, HostResultRequirement::NamedList { state }));
                    }
                    let (parameter, source_axis) = witness(binder)?;
                    Some((
                        axis,
                        HostResultRequirement::Named {
                            claim: chelis_ir::lower::extent_binder_label(binder),
                            prepared: parameter.clone(),
                            parameter,
                            axis: source_axis,
                        },
                    ))
                }
                DimInfo::Named(_, _) => None,
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
        outer_claims_first: bool,
    ) -> Vec<String> {
        if self
            .axes
            .iter()
            .any(|(_, requirement)| matches!(requirement, HostResultRequirement::NamedList { .. }))
        {
            let count = format!("{axes_name}_count");
            let mut lines = vec![
                format!(
                    "{indent}__chelis_host_result_axis {axes_name}[{}];",
                    self.axes.len()
                ),
                format!("{indent}int64_t {count} = 0;"),
            ];
            for (axis, requirement) in &self.axes {
                match requirement {
                    HostResultRequirement::Literal(required) => lines.push(format!(
                        "{indent}{axes_name}[{count}++] = (__chelis_host_result_axis){{ {axis}, {required}, NULL, NULL, 0 }};"
                    )),
                    HostResultRequirement::Named {
                        claim,
                        parameter,
                        prepared,
                        axis: source_axis,
                    } => lines.push(format!(
                        "{indent}{axes_name}[{count}++] = (__chelis_host_result_axis){{ {axis}, chelis_tensor_shape({}, {source_axis}), {}, {}, {source_axis} }};",
                        c_ident(prepared),
                        c_string_literal(claim),
                        c_string_literal(parameter),
                    )),
                    HostResultRequirement::NamedList { state } => {
                        let source = format!("__chelis_entry_named_states[{state}]");
                        lines.push(format!("{indent}if ({source}.seen) {{"));
                        lines.push(format!(
                            "{indent}    {axes_name}[{count}++] = (__chelis_host_result_axis){{ {axis}, {source}.value, {source}.claim, {source}.path, {source}.axis }};"
                        ));
                        lines.push(format!("{indent}}}"));
                    }
                }
            }
            lines.push(format!(
                "{indent}const __chelis_host_result_claim {frame_name} = {{ {parent}, {}, {count}, {axes_name}, {} }};",
                self.rank,
                i32::from(outer_claims_first),
            ));
            if let Some(claims_name) = claims_name {
                lines.push(format!(
                    "{indent}const __chelis_host_result_claim *{claims_name} = &{frame_name};"
                ));
            }
            return lines;
        }
        let mut lines = vec![format!(
            "{indent}const __chelis_host_result_axis {axes_name}[] = {{"
        )];
        for (axis, requirement) in &self.axes {
            lines.push(match requirement {
                HostResultRequirement::Literal(required) => {
                    format!("{indent}    {{ {axis}, {required}, NULL, NULL, 0 }},")
                }
                HostResultRequirement::Named {
                    claim,
                    parameter,
                    prepared,
                    axis: source_axis,
                } => format!(
                    "{indent}    {{ {axis}, chelis_tensor_shape({}, {source_axis}), {}, {}, {source_axis} }},",
                    c_ident(prepared),
                    c_string_literal(claim),
                    c_string_literal(parameter),
                ),
                HostResultRequirement::NamedList { .. } => unreachable!("dynamic frame handled above"),
            });
        }
        lines.push(format!("{indent}}};"));
        lines.push(format!(
            "{indent}const __chelis_host_result_claim {frame_name} = {{ {parent}, {}, {}, {axes_name}, {} }};",
            self.rank,
            self.axes.len(),
            i32::from(outer_claims_first),
        ));
        if let Some(claims_name) = claims_name {
            lines.push(format!(
                "{indent}const __chelis_host_result_claim *{claims_name} = &{frame_name};"
            ));
        }
        lines
    }
}

/// A C string literal spelling `text`. Binder labels and parameter names are
/// identifiers, but the escape keeps any byte a literal cannot hold verbatim.
fn c_string_literal(text: &str) -> String {
    let mut out = String::from("\"");
    for byte in text.bytes() {
        match byte {
            b'"' => out.push_str("\\\""),
            b'\\' => out.push_str("\\\\"),
            b' '..=b'~' => out.push(char::from(byte)),
            _ => out.push_str(&format!("\\{byte:03o}")),
        }
    }
    out.push('"');
    out
}

/// This context is translation-unit private. Public wrappers retain their
/// authored signatures, while one owned callee accepts any caller's literals.
fn private_host_params(params: &str) -> String {
    join_params(
        params,
        "const __chelis_host_result_claim *__chelis_caller_result_claims",
    )
}

/// Owned host bodies additionally return private producer provenance. Public
/// wrappers pass NULL, so this never enters the generated ABI.
/// This spelling deliberately lives outside `__chelis_result_origin_<value>`:
/// a legal source parameter such as `out` owns that exact sidecar namespace.
const PRIVATE_RESULT_ORIGIN_RETURN_SLOT: &str = "__chelis_private_result_origin_return";

fn private_host_function_params(params: &str) -> String {
    join_params(
        params,
        &format!(
            "__chelis_host_result_origin_arena *__chelis_origin_arena, const __chelis_entry_receipt *__chelis_caller_entry_receipt, const __chelis_host_result_claim *__chelis_caller_result_claims, const __chelis_host_result_origin **{PRIVATE_RESULT_ORIGIN_RETURN_SLOT}"
        ),
    )
}

fn append_host_result_claim_support(out: &mut Vec<String>) {
    out.push(r#"/* One declared result axis. `claim` is NULL for a literal requirement;
   otherwise it is the authored binder and `source` the parameter whose
   `source_axis` supplied `required`. */
typedef struct __chelis_host_result_axis {
    int64_t axis;
    int64_t required;
    const char *claim;
    const char *source;
    int64_t source_axis;
} __chelis_host_result_axis;

typedef struct __chelis_host_result_claim {
    const struct __chelis_host_result_claim *next;
    int64_t rank;
    int64_t count;
    const __chelis_host_result_axis *axes;
    int outer_claims_first;
} __chelis_host_result_claim;

typedef struct __chelis_host_result_origin {
    struct __chelis_host_result_origin *allocation_next;
    struct __chelis_host_result_origin *leaf_next;
    const char *op;
    const char *trap;
    int64_t child_count;
    /* A leaf that stands for every value nested in the one it labels: the
       interface `load` origin. Projecting a child of it yields itself. */
    int uniform;
    const struct __chelis_host_result_origin *const *child_view;
    const struct __chelis_host_result_origin *children[];
} __chelis_host_result_origin;

typedef struct __chelis_host_result_origin_arena {
    __chelis_host_result_origin *head;
    __chelis_host_result_origin *leaf_head;
} __chelis_host_result_origin_arena;

/* Private proof of one complete, successful entry contract. Only a verified
   direct call may forward it. The state bytes belong to the caller's stack
   frame and are copied at the callee entry before any body operation. */
typedef struct __chelis_entry_receipt {
    const void *contract;
    const void *named_states;
    size_t named_state_bytes;
} __chelis_entry_receipt;

static void __chelis_host_result_origin_arena_destroy(__chelis_host_result_origin_arena *arena) {
    __chelis_host_result_origin *node = arena->head;
    while (node != NULL) {
        __chelis_host_result_origin *next = node->allocation_next;
        free(node);
        node = next;
    }
    arena->head = NULL;
    arena->leaf_head = NULL;
}

static __chelis_host_result_origin *__chelis_host_result_origin_alloc(__chelis_host_result_origin_arena *arena, int64_t child_count) {
    if (arena == NULL || child_count < 0 || (uint64_t)child_count > (SIZE_MAX - sizeof(__chelis_host_result_origin)) / sizeof(const __chelis_host_result_origin *)) {
        fprintf(stderr, "host runtime: invalid result producer provenance allocation\n");
        abort();
    }
    size_t bytes = sizeof(__chelis_host_result_origin) + (size_t)child_count * sizeof(const __chelis_host_result_origin *);
    __chelis_host_result_origin *node = (__chelis_host_result_origin *)malloc(bytes);
    if (node == NULL) {
        fprintf(stderr, "host runtime: result producer provenance allocation failed\n");
        abort();
    }
    node->allocation_next = arena->head;
    node->leaf_next = NULL;
    node->op = NULL;
    node->trap = NULL;
    node->child_count = child_count;
    node->uniform = 0;
    node->child_view = NULL;
    arena->head = node;
    return node;
}

static const __chelis_host_result_origin *__chelis_host_result_origin_leaf(__chelis_host_result_origin_arena *arena, const char *op, const char *trap) {
    for (__chelis_host_result_origin *node = arena->leaf_head; node != NULL; node = node->leaf_next) {
        if (strcmp(node->op, op) == 0 && strcmp(node->trap, trap) == 0) return node;
    }
    __chelis_host_result_origin *node = __chelis_host_result_origin_alloc(arena, 0);
    node->op = op;
    node->trap = trap;
    node->child_count = -1;
    node->leaf_next = arena->leaf_head;
    arena->leaf_head = node;
    return node;
}

/* The origin of a list combinator's result. spec/04 section 4.7 makes the
   combinator the producer of every tensor nested in it, so one uniform leaf
   describes the whole value. */
static const __chelis_host_result_origin *__chelis_host_result_origin_uniform(__chelis_host_result_origin_arena *arena, const char *op, const char *trap) {
    const __chelis_host_result_origin *leaf = __chelis_host_result_origin_leaf(arena, op, trap);
    ((__chelis_host_result_origin *)leaf)->uniform = 1;
    return leaf;
}

/* The origin of a value that entered this invocation from its caller.
   Every tensor nested in an interface value is a `load`, so one uniform leaf
   describes the whole value without walking it (chelis#2522). */
static const __chelis_host_result_origin *__chelis_host_result_origin_load(__chelis_host_result_origin_arena *arena) {
    const __chelis_host_result_origin *leaf = __chelis_host_result_origin_leaf(arena, "load", "numeric trap: domain in load at i64");
    ((__chelis_host_result_origin *)leaf)->uniform = 1;
    return leaf;
}

static const __chelis_host_result_origin *__chelis_host_result_origin_aggregate(__chelis_host_result_origin_arena *arena, int64_t child_count, const __chelis_host_result_origin *const *children) {
    bool any = false;
    for (int64_t i = 0; i < child_count; ++i) {
        if (children[i] != NULL) any = true;
    }
    if (!any) return NULL;
    __chelis_host_result_origin *node = __chelis_host_result_origin_alloc(arena, child_count);
    for (int64_t i = 0; i < child_count; ++i) node->children[i] = children[i];
    node->child_view = node->children;
    return node;
}

static const __chelis_host_result_origin *__chelis_host_result_origin_child(const __chelis_host_result_origin *origin, int64_t index) {
    if (origin == NULL) return NULL;
    if (origin->uniform) return origin;
    if (origin->child_count < 0 || index < 0 || index >= origin->child_count || origin->child_view == NULL) {
        fprintf(stderr, "host runtime: aggregate result producer provenance does not match the projected value\n");
        abort();
    }
    return origin->child_view[index];
}

static const __chelis_host_result_origin *__chelis_host_result_origin_list_suffix(__chelis_host_result_origin_arena *arena, const __chelis_host_result_origin *origin, int64_t count) {
    if (origin == NULL || count == 0 || origin->uniform) return origin;
    // The runtime owns skip's negative-count diagnostic. Preserve that
    // ordering instead of replacing it with an internal metadata failure.
    if (count < 0) return origin;
    if (origin->child_count < 0 || origin->child_view == NULL) {
        fprintf(stderr, "host runtime: aggregate result producer provenance does not match the projected value\n");
        abort();
    }
    if (count >= origin->child_count) return NULL;
    // This is an immutable view over another node in the same invocation
    // arena. Flattening the view keeps repeated Cons-tail decomposition O(1).
    __chelis_host_result_origin *suffix = __chelis_host_result_origin_alloc(arena, 0);
    suffix->child_count = origin->child_count - count;
    suffix->child_view = origin->child_view + count;
    return suffix;
}

static const __chelis_host_result_origin *__chelis_host_result_origin_list_prefix(__chelis_host_result_origin_arena *arena, const __chelis_host_result_origin *origin, int64_t count) {
    if (origin == NULL || origin->uniform) return origin;
    // The runtime owns take's negative-count diagnostic.
    if (count < 0) return origin;
    if (origin->child_count < 0 || origin->child_view == NULL) {
        fprintf(stderr, "host runtime: aggregate result producer provenance does not match the projected value\n");
        abort();
    }
    if (count >= origin->child_count) return origin;
    if (count == 0) return NULL;
    __chelis_host_result_origin *prefix = __chelis_host_result_origin_alloc(arena, 0);
    prefix->child_count = count;
    prefix->child_view = origin->child_view;
    return prefix;
}

static const __chelis_host_result_origin **__chelis_host_result_origin_children(int64_t count) {
    if (count <= 0) return NULL;
    if ((uint64_t)count > SIZE_MAX / sizeof(const __chelis_host_result_origin *)) {
        fprintf(stderr, "host runtime: invalid result producer provenance child count\n");
        abort();
    }
    const __chelis_host_result_origin **children = (const __chelis_host_result_origin **)calloc((size_t)count, sizeof(*children));
    if (children == NULL) {
        fprintf(stderr, "host runtime: result producer provenance child allocation failed\n");
        abort();
    }
    return children;
}
"#.to_string());
}

fn append_host_result_claim_checks(out: &mut Vec<String>) {
    out.push(r#"
static void __chelis_host_result_claim_trap(const __chelis_host_result_axis *claim, const char *op, int64_t observed_axis, int64_t observed, const char *trap) {
    if (claim->claim == NULL) {
        fprintf(stderr, "extent `%lld`: claimed = %lld, %s axis %lld = %lld\n", (long long)claim->required, (long long)claim->required, op, (long long)observed_axis, (long long)observed);
    } else {
        fprintf(stderr, "extent `%s`: %s axis %lld = %lld, %s axis %lld = %lld\n", claim->claim, claim->source, (long long)claim->source_axis, (long long)claim->required, op, (long long)observed_axis, (long long)observed);
    }
    chelis_numeric_trap(trap);
}

static void __chelis_check_host_result_extent_claims(const __chelis_host_result_claim *claims, int64_t rank, const int64_t (*observations)[3], int64_t count, const char *op, const char *trap) {
    if (claims == NULL) return;
    if (claims->outer_claims_first) __chelis_check_host_result_extent_claims(claims->next, rank, observations, count, op, trap);
    {
        if (rank == claims->rank) for (int64_t i = 0; i < claims->count; ++i) {
            for (int64_t j = 0; j < count; ++j) {
                if (claims->axes[i].axis != observations[j][0]) continue;
                if (claims->axes[i].required != observations[j][2]) {
                    __chelis_host_result_claim_trap(&claims->axes[i], op, observations[j][1], observations[j][2], trap);
                }
            }
        }
    }
    if (!claims->outer_claims_first) __chelis_check_host_result_extent_claims(claims->next, rank, observations, count, op, trap);
}

static void __chelis_check_host_result_claims(const __chelis_host_result_claim *claims, const chelis_tensor *value, const char *op, const char *trap) {
    if (claims == NULL) return;
    if (claims->outer_claims_first) __chelis_check_host_result_claims(claims->next, value, op, trap);
    {
        if (chelis_tensor_rank(value) == claims->rank) for (int64_t i = 0; i < claims->count; ++i) {
            int64_t axis = claims->axes[i].axis;
            int64_t observed = chelis_tensor_shape(value, axis);
            if (observed != claims->axes[i].required) {
                __chelis_host_result_claim_trap(&claims->axes[i], op, axis, observed, trap);
            }
        }
    }
    if (!claims->outer_claims_first) __chelis_check_host_result_claims(claims->next, value, op, trap);
}
"#.to_string());
}

/// A signature entry's checks, in signature order: every observation's null,
/// dtype and rank checks, then the ordered extent comparisons.
///
/// `work` carries a function entry's nested values (chelis#2506): the reads
/// that fetch them run with their parameter's metadata checks, a walked
/// value's metadata pass runs after its parameter's observations, and its
/// literal-extent pass runs before the first comparison a later parameter
/// owes. A retained invocation supplies work when its formal carries a named
/// List.
fn entry_named_state_lines(names: &[String]) -> Vec<String> {
    if names.is_empty() {
        return Vec::new();
    }
    let mut lines = vec!["__chelis_entry_named_state __chelis_entry_named_states[] = {".into()];
    for name in names {
        lines.push(format!(
            "    {{ {}, {}, 0, 0, 0, {{0}} }},",
            c_string_literal(name),
            c_string_literal(&chelis_ir::lower::extent_binder_label(name))
        ));
    }
    lines.push("};".into());
    lines
}

fn validate_retained_entry_contract(
    contract: &chelis_ir::host::EntryContract<HostType>,
    plan: &chelis_ir::host::SignatureEntryPlan,
    positions: &[usize],
    lists: &[chelis_ir::host::HostListEntry<HostType>],
) -> Result<(), Unsupported> {
    use chelis_ir::host::EntryPattern;
    let mut represented = vec![false; contract.formals().len()];
    if positions.len() != plan.observations().nodes().len() {
        return Err(invalid_abi_shape(
            "retained entry lost a fixed tensor observation".into(),
            "signature entry",
        ));
    }
    for (node, position) in plan.observations().nodes().iter().zip(positions) {
        let Some(formal) = contract.formals().get(*position) else {
            return Err(invalid_abi_shape(
                "retained entry has an invalid tensor position".into(),
                "signature entry",
            ));
        };
        if represented[*position]
            || formal.name() != plan.label(node.id)
            || !entry_walk::entry_pattern_matches_type(
                formal.pattern(),
                &HostType::Tensor(node.output_type.clone()),
            )
        {
            return Err(invalid_abi_shape(
                "retained entry changed a tensor formal".into(),
                "signature entry",
            ));
        }
        represented[*position] = true;
    }
    let mut projected_binders = Vec::new();
    for entry in lists {
        let Some(formal) = contract.formals().get(entry.position) else {
            return Err(invalid_abi_shape(
                "retained entry has an invalid List position".into(),
                "signature entry",
            ));
        };
        if represented[entry.position]
            || formal.name() != entry.name
            || !matches!(formal.pattern(), EntryPattern::List(_))
            || !entry_walk::entry_pattern_matches_type(formal.pattern(), &entry.ty)
        {
            return Err(invalid_abi_shape(
                "retained entry changed a List formal".into(),
                "signature entry",
            ));
        }
        entry_walk::list_named_dims(&entry.ty, &mut projected_binders);
        represented[entry.position] = true;
    }
    if projected_binders != contract.named_list_binders() {
        return Err(invalid_abi_shape(
            "retained entry lost a List binder".into(),
            "signature entry",
        ));
    }
    for (formal, represented) in contract.formals().iter().zip(represented) {
        if entry_walk::entry_pattern_has_extent_claim(formal.pattern()) && !represented {
            return Err(invalid_abi_shape(
                "retained entry omitted a declared extent formal".into(),
                "signature entry",
            ));
        }
    }
    Ok(())
}

fn signature_entry_lines(
    plan: &chelis_ir::host::SignatureEntryPlan,
    args: &[String],
    indent: &str,
    delegated: &[chelis_ir::axis_sources::EntryExtentGuard],
    work: Option<&entry_walk::FunctionEntryWork>,
    declare_named_states: bool,
) -> Result<Vec<String>, Unsupported> {
    use chelis_ir::axis_sources::EntryExtentGuard;
    if args.len() != plan.observations().nodes().len()
        || work.is_some_and(|work| {
            work.owners.len() != args.len()
                || work.owners.iter().any(|owner| *owner >= work.params.len())
        })
    {
        return Err(invalid_abi_shape(
            "signature entry lost an input observation".into(),
            "signature entry",
        ));
    }
    let owners = work.map_or_else(|| (0..args.len()).collect(), |work| work.owners.clone());
    let params = work.map_or(&[][..], |work| work.params.as_slice());
    let param_count = work.map_or(args.len(), |work| work.params.len());
    if let Some(work) = work {
        let mut fixed_seen = vec![false; args.len()];
        for (owner, param) in work.params.iter().enumerate() {
            for step in &param.ordered_extents {
                let entry_walk::ExtentStep::Fixed(index) = step else {
                    continue;
                };
                if owners.get(*index) != Some(&owner) || fixed_seen[*index] {
                    return Err(invalid_abi_shape(
                        "signature entry changed an ordered tensor observation".into(),
                        "signature entry",
                    ));
                }
                fixed_seen[*index] = true;
            }
        }
        if fixed_seen.iter().any(|seen| !seen) {
            return Err(invalid_abi_shape(
                "signature entry omitted an ordered tensor observation".into(),
                "signature entry",
            ));
        }
    }
    let label_of = |node: &chelis_ir::dag::DagNode| {
        let RiscOp::Load { name } = &node.op else {
            unreachable!("signature observation")
        };
        entry_walk::TensorLabel::fixed(name.as_str())
    };
    let mut lines = Vec::new();
    let named_list_binders = work.map_or(&[][..], |work| work.named_list_binders.as_slice());
    if declare_named_states {
        lines.extend(entry_named_state_lines(named_list_binders));
    }
    // No extent read may obscure a malformed external input's null, dtype or
    // rank diagnostic. These metadata checks dominate the ordered comparisons.
    for param in 0..param_count {
        if let Some(work) = params.get(param) {
            lines.extend(work.fetch.iter().cloned());
        }
        for ((node, actual), _) in plan
            .observations()
            .nodes()
            .iter()
            .zip(args)
            .zip(&owners)
            .filter(|(_, owner)| **owner == param)
        {
            lines.extend(entry_walk::tensor_metadata_checks(
                actual,
                &label_of(node),
                &node.output_type,
            ));
        }
        if let Some(work) = params.get(param) {
            lines.extend(work.metadata.iter().cloned());
        }
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
    let guard_lines = |guard: &EntryExtentGuard| -> Vec<String> {
        match guard {
            EntryExtentGuard::Named {
                claim,
                canonical,
                observed,
            } => {
                let (left, first, first_axis) = read(*canonical);
                let (right, later, later_axis) = read(*observed);
                let claim = chelis_ir::span_sanitize::sanitize_for_format_string(claim);
                vec![
                    format!("if ({right} != {left}) {{"),
                    format!(
                        "    fprintf(stderr, \"extent `{claim}`: {first} axis {first_axis} = %lld, {later} axis {later_axis} = %lld\\n\", (long long)({left}), (long long)({right}));"
                    ),
                    "    chelis_numeric_trap(\"numeric trap: domain in load at i64\");".into(),
                    "}".into(),
                ]
            }
            EntryExtentGuard::Literal { required, observed } => {
                let node = plan
                    .observations()
                    .get(observed.0)
                    .expect("signature witness");
                entry_walk::literal_extent_check(
                    &args[observed.0.0],
                    &label_of(node),
                    observed.1,
                    *required,
                )
            }
        }
    };
    if work.is_some() {
        // The ordered work interleaves fixed tuple positions and aggregate
        // walks even when a List has only literal claims. An observation-only
        // pass would move a later tuple field ahead of an earlier List walk.
        for param in params {
            for step in &param.ordered_extents {
                match step {
                    entry_walk::ExtentStep::Walk(call) => lines.push(call.clone()),
                    entry_walk::ExtentStep::Fixed(index) => {
                        let node = &plan.observations().nodes()[*index];
                        let actual = &args[*index];
                        let label = plan.label(node.id);
                        for (axis, dim) in node.output_type.dims.iter().enumerate() {
                            if let DimInfo::Named(name, _) = dim
                                && named_list_binders.contains(name)
                            {
                                lines.push(format!(
                                    "__chelis_entry_named_observe(__chelis_entry_named_states, {}, {}, {}, {axis}, chelis_tensor_shape({actual}, {axis}));",
                                    named_list_binders.len(),
                                    c_string_literal(name),
                                    c_string_literal(label),
                                ));
                            }
                            for guard in plan.guards() {
                                let observed = match guard {
                                    EntryExtentGuard::Named { observed, .. }
                                    | EntryExtentGuard::Literal { observed, .. } => observed,
                                };
                                if *observed != (node.id, axis) || delegated.contains(guard) {
                                    continue;
                                }
                                if matches!(dim, DimInfo::Named(name, _) if named_list_binders.contains(name))
                                    && matches!(guard, EntryExtentGuard::Named { .. })
                                {
                                    continue;
                                }
                                lines.extend(guard_lines(guard));
                            }
                        }
                    }
                }
            }
        }
    } else {
        let mut walked = 0;
        for guard in plan
            .guards()
            .iter()
            .filter(|guard| !delegated.contains(guard))
        {
            let (EntryExtentGuard::Named { observed, .. }
            | EntryExtentGuard::Literal { observed, .. }) = guard;
            while walked < owners[observed.0.0].min(params.len()) {
                lines.extend(params[walked].extents.iter().cloned());
                walked += 1;
            }
            lines.extend(guard_lines(guard));
        }
        for work in &params[walked..] {
            lines.extend(work.extents.iter().cloned());
        }
    }
    if let Some(work) = work {
        lines.extend(work.release.iter().cloned());
    }
    Ok(lines
        .into_iter()
        .map(|line| format!("{indent}{line}"))
        .collect())
}

/// A receipt crosses a direct call only when both owned bodies check the
/// identical positional contract and ABI types. Display paths are part of
/// that equality, so replaying the first named witness keeps its label.
fn entry_receipt_groups(program: &HostProgram) -> UnordMap<String, usize> {
    let mut groups = UnordMap::new();
    for (index, function) in program.functions.iter().enumerate() {
        let representative = program.functions[..index]
            .iter()
            .position(|prior| {
                prior.entry_contract == function.entry_contract
                    && prior.params.len() == function.params.len()
                    && prior
                        .params
                        .iter()
                        .zip(&function.params)
                        .all(|(left, right)| left.name == right.name && left.ty == right.ty)
            })
            .unwrap_or(index);
        groups.insert(function.name.clone(), representative);
    }
    groups
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
    helper_output_types: &[Vec<TensorType>],
    verified_helpers: &[VerifiedHostTensorHelperView<'_>],
    external_helpers: &UnordSet<String>,
    captured_globals: &[String],
    entry_work: &entry_walk::EntryWork,
    entry_groups: &UnordMap<String, usize>,
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
    let mut emitter = HostEmitter::new(
        "    ".to_string(),
        emitted_name,
        internal_names.clone(),
        function_specializations.clone(),
        HostTensorHelpers {
            helpers: &function.tensor_helpers,
            output_types: helper_output_types,
            result_origins: verified_helpers
                .iter()
                .copied()
                .map(verified_helper_result_origin)
                .collect::<Result<Vec<_>, _>>()?,
        },
        ownership_sites,
    );
    emitter.entry_group = entry_groups.get(&function.name).copied();
    emitter.entry_groups = entry_groups.clone();
    if entry_work.extent_at_body {
        emitter.entry_proof_owners = entry_work
            .body
            .params
            .iter()
            .enumerate()
            .filter(|(_, work)| {
                !work.fetch.is_empty()
                    || !work.metadata.is_empty()
                    || !work.ordered_extents.is_empty()
            })
            .map(|(index, _)| {
                owner_bindings
                    .get(index)
                    .map(|(owner, _)| (index, *owner))
                    .ok_or_else(|| {
                        invalid_abi_shape(
                            "entry receipt lost a verified formal owner".into(),
                            "signature entry",
                        )
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
    }
    emitter.entry_projection = entry::helper_coverage_with_verified(function, verified_helpers);
    emitter.external_helpers = external_helpers.clone();
    emitter.interface_reload_names = captured_globals.iter().cloned().collect();
    for param in &function.params {
        emitter.interface_reload_names.remove(&param.name);
        emitter.declare_result_origin(&param.name, &param.ty, Some("load"));
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
    let delegated_entry_guards = entry::delegated_function_guards(function, verified_helpers);
    if entry_work.extent_at_body {
        let group = emitter
            .entry_group
            .expect("every emitted function has an entry group");
        let token = format!("&__chelis_entry_contract_token_{group}");
        emitter.lines.extend(
            entry_named_state_lines(&entry_work.body.named_list_binders)
                .into_iter()
                .map(|line| format!("{}{}", emitter.indent, line)),
        );
        let (states, bytes) = if entry_work.body.named_list_binders.is_empty() {
            ("NULL", "0")
        } else {
            (
                "__chelis_entry_named_states",
                "sizeof __chelis_entry_named_states",
            )
        };
        emitter.lines.push(format!(
            "{}if (__chelis_caller_entry_receipt != NULL && \
             __chelis_caller_entry_receipt->contract == {token} && \
             __chelis_caller_entry_receipt->named_state_bytes == {bytes} && \
             ({bytes} == 0 || __chelis_caller_entry_receipt->named_states != NULL)) {{",
            emitter.indent
        ));
        if !entry_work.body.named_list_binders.is_empty() {
            emitter.lines.push(format!(
                "{}    memcpy(__chelis_entry_named_states, \
                 __chelis_caller_entry_receipt->named_states, {bytes});",
                emitter.indent
            ));
        }
        emitter.lines.push(format!("{}}} else {{", emitter.indent));
        emitter.lines.extend(signature_entry_lines(
            &entry_plan,
            &entry_work.body.args,
            &format!("{}    ", emitter.indent),
            &delegated_entry_guards,
            Some(&entry_work.body),
            false,
        )?);
        emitter.lines.push(format!("{}}}", emitter.indent));
        emitter.lines.push(format!(
            "{}const __chelis_entry_receipt __chelis_entry_receipt_current = \
             {{ {token}, {states}, {bytes} }};",
            emitter.indent
        ));
    } else {
        emitter.lines.extend(signature_entry_lines(
            &entry_plan,
            &entry_work.body.args,
            &emitter.indent,
            &delegated_entry_guards,
            Some(&entry_work.body),
            true,
        )?);
    }
    // A frame belongs to this invocation, not to a selected callee name.
    // The expression spine forwards the frame; branch arms share its immutable
    // contents and arguments/sibling bindings never inherit it. A named axis
    // reads its witnessing parameter here, before entry drops can release it.
    //
    // The declared frame is checked first, then the frame of each
    // output-inferred binder a body site names, then the caller's. Which
    // binders a site names is known once the body is emitted, so the frames
    // are placed here afterwards.
    let claims_at = emitter.lines.len();
    emitter.first_site_frames = Some(FirstSiteFrames::of(
        function,
        if entry_work.extent_at_body {
            entry_work.body.named_list_binders.clone()
        } else {
            Vec::new()
        },
    ));
    // Entry guards and the frame still read parameters the body does not
    // use. Their verified entry drops run only after those reads finish.
    emitter.emit_entry_terminals(entry, authored)?;
    emitter.result_claims = Some("__chelis_result_claims".to_string());
    emitter.claim_on_spine = true;
    emitter.emit_expr_to_var(&function.body, "__result", &function.ret_ty)?;
    // The declared result is a later site of every binder the body bound,
    // including one whose first site ran after the returned value's producer,
    // where that producer's check found the frame empty.
    emitter.emit_late_first_site_checks("__result");
    let first_site_frames = emitter.first_site_frames.take().ok_or_else(|| {
        invalid_abi_shape(
            "a function body lost its first-site frames".to_string(),
            "verified C host first-site emission",
        )
    })?;
    let indent = emitter.indent.clone();
    let (mut claim_lines, claims_parent) =
        first_site_frames.declare(&indent, "__chelis_caller_result_claims");
    match HostResultClaim::of(function) {
        Some(claim) => claim_lines.extend(claim.frame_lines(
            &indent,
            "__chelis_result_axes",
            "__chelis_declared_result",
            &claims_parent,
            Some("__chelis_result_claims"),
            false,
        )),
        None => claim_lines.push(format!(
            "{indent}const __chelis_host_result_claim *__chelis_result_claims = {claims_parent};"
        )),
    }
    emitter.lines.splice(claims_at..claims_at, claim_lines);
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
    emitter.lines.push(format!(
        "{}if ({PRIVATE_RESULT_ORIGIN_RETURN_SLOT} != NULL) *{PRIVATE_RESULT_ORIGIN_RETURN_SLOT} = {};",
        emitter.indent,
        result_origin_name("__result")
    ));
    emitter.finish_expression_sites()?;
    out.extend(emitter.lines);
    out.push("    return __result;".to_string());
    out.push("}".to_string());

    if authored {
        let entry_uses = authored_entry_uses(ownership_sites, function.params.len())?;
        // A public call admits every aggregate extent in signature order.
        // Internal calls retain their List-only entry boundary.
        let exported_work = entry_work.exported.as_ref().ok_or_else(|| {
            invalid_abi_shape(
                "authored function has no exported entry work".into(),
                "signature entry",
            )
        })?;
        let exported_entry = if exported_work.walks() {
            signature_entry_lines(
                &entry_plan,
                &exported_work.args,
                "    ",
                &delegated_entry_guards,
                Some(exported_work),
                true,
            )?
        } else {
            Vec::new()
        };
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
        out.push(format!("    {}", crate::fp_env::ENTRY));
        out.extend(exported_entry.iter().cloned());
        let wrapper_receipt = if entry_work.extent_at_body && delegated_entry_guards.is_empty() {
            let group = entry_groups
                .get(&function.name)
                .copied()
                .expect("every emitted function has an entry group");
            let (states, bytes) = if exported_work.named_list_binders.is_empty() {
                ("NULL", "0")
            } else {
                (
                    "__chelis_entry_named_states",
                    "sizeof __chelis_entry_named_states",
                )
            };
            out.push(format!(
                "    const __chelis_entry_receipt __chelis_exported_entry_receipt = \
                 {{ &__chelis_entry_contract_token_{group}, {states}, {bytes} }};"
            ));
            "&__chelis_exported_entry_receipt"
        } else {
            "NULL"
        };
        append_invocation_origin_context(out);
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
        append_private_host_context_args(&mut args, wrapper_receipt);
        args.push("NULL".to_string());
        args.push("NULL".to_string());
        out.push(format!(
            "    {} __result = {}({});",
            c_type(&function.ret_ty)?,
            body_name,
            args.join(", ")
        ));
        out.push(
            "    __chelis_host_result_origin_arena_destroy(__chelis_origin_arena);".to_string(),
        );
        out.push(format!("    {}", crate::fp_env::EXIT));
        out.push("    return __result;".to_string());
        out.push("}".to_string());
        out.push(crate::generated_header::render_authored_export_end(
            &function.name,
        ));
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
    helper_output_types: &[Vec<TensorType>],
    helper_result_origins: Vec<Option<String>>,
    external_helpers: &UnordSet<String>,
) -> Result<(), Unsupported> {
    out.push("int main(void) {".to_string());
    // The process starts in the IEEE default environment; entering keeps a
    // custom C runtime or preloaded library from changing that. A trap
    // aborts, and returning from main ends the process, so there is no
    // caller state to restore.
    out.push(format!("    {}", crate::fp_env::ENTRY));
    append_invocation_origin_context(out);
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
            output_types: helper_output_types,
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
        // A top-level read and a same-spelled lexical binder have distinct
        // identities. The alias is a second C name for this binding's value;
        // only the authored binding owns storage and participates in roots.
        let global_label = chelis_ir::LoadStoreName::top_level(&binding.name);
        if hoisted.contains(global_label.as_str()) {
            emitter.lines.push(format!(
                "    {} = {};",
                c_ident(global_label.as_str()),
                c_ident(&binding.name)
            ));
        } else {
            emitter.lines.push(format!(
                "    {} = {};",
                c_decl(&binding.ty, global_label.as_str())?,
                c_ident(&binding.name)
            ));
        }
        let binding_origin = result_origin_name(&binding.name);
        let value_origin = result_origin_name(&binding_var);
        emitter.lines.push(format!(
            "    const __chelis_host_result_origin *{binding_origin} = {value_origin};"
        ));
        emitter.lines.push(format!(
            "    const __chelis_host_result_origin *{} = {binding_origin};",
            result_origin_name(global_label.as_str())
        ));
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
    out.push("    __chelis_host_result_origin_arena_destroy(__chelis_origin_arena);".to_string());
    out.push("    return 0;".to_string());
    out.push("}".to_string());
    Ok(())
}

/// Function names referenced from `expr` - `Call`/`Named`-callback
/// targets plus bare `Var` references (a def passed as a value). The
/// over-approximation direction is the safe one for the reachability
/// gate below: an over-counted reference makes a failing wrapper a hard
/// build error rather than a loud stub.
fn collect_referenced_fn_names(expr: &HostExpr, out: &mut UnordSet<String>) {
    chelis_ir::host::collect_host_var_names(expr, out);
    fn walk(expr: &HostExpr, out: &mut UnordSet<String>) {
        match &expr.kind {
            HostExprKind::ResultClaimScope { body, .. } => walk(body, out),
            HostExprKind::FormalIngress { value, .. } | HostExprKind::ExtentSites { value, .. } => {
                walk(value, out)
            }
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
            HostExprKind::SignatureEntry { args, lists, .. } => {
                for arg in args.iter().chain(lists.iter().map(|entry| &entry.value)) {
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
            HostExprKind::Let { bindings, body, .. }
            | HostExprKind::RetainedInvocation { bindings, body, .. } => {
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
    /// [04-NUM-2]: the NaN finalization of the tensor builtin whose
    /// elementwise loop is being emitted.
    tensor_nan_finalization: Option<crate::fp_env::NanFinalization>,
    indent: String,
    helper_prefix: String,
    emitted_names: UnordMap<String, String>,
    function_specializations: UnordMap<String, HostFunctionSpecialization>,
    tensor_helpers: &'a [HostTensorHelper],
    tensor_helper_output_types: &'a [Vec<TensorType>],
    tensor_helper_result_origins: Vec<Option<String>>,
    expression_sites: Vec<ProjectedHostSite<'a>>,
    expression_site_index: usize,
    entry_projection: entry::Projection,
    external_helpers: UnordSet<String>,
    /// Values stored outside this invocation have no arena-owned provenance.
    /// Rebuild their aggregate load tree at each reference instead of reading
    /// an origin pointer retained by the cached/global value.
    interface_reload_names: UnordSet<String>,
    pre_emitted_clone_sites: UnordSet<HostSiteId>,
    pre_emitted_terminals: UnordSet<(HostSiteId, VerifiedOperationId)>,
    owner_vars: UnordMap<VerifiedOwnerId, String>,
    entry_group: Option<usize>,
    entry_groups: UnordMap<String, usize>,
    entry_proof_owners: Vec<(usize, VerifiedOwnerId)>,
    temp_counter: usize,
    /// Immutable invocation context. Only the expression on the returned-value
    /// spine receives it; nested arguments and sibling bindings get no context.
    result_claims: Option<String>,
    /// Taken at each expression entry and forwarded to its returned-value child.
    claim_on_spine: bool,
    /// The mutable claim frames of the output-inferred binders the function's
    /// sites name ([`FirstSiteFrames`]); `None` outside a function body.
    first_site_frames: Option<FirstSiteFrames>,
}

/// The claim frame of one output-inferred binder of the function being
/// emitted (spec/04-type-system.md section 4.4.1): a binder in the
/// declaration's [`chelis_ir::lower::DimBinderRoles`] `return_only` set,
/// which a site marks [`chelis_ir::lower::LocalAscriptionNamedSite::output_inferred`].
/// The frame starts empty and the first site the body executes fills it
/// ([`HostExprKind::ExtentSites`]). Every later site, and the declared result
/// where the result is a tensor, is then a claim against that site. The frame
/// lives for the whole invocation, so a site inside a block binds the binder
/// after the block, and a site inside an `if` arm binds it on that arm's path
/// only.
struct FirstSiteFrame {
    binder: String,
    axes_name: String,
    frame_name: String,
    /// The declared tensor result's axes that name the binder; empty when
    /// the result is not a tensor (a tuple, `Option` or `List` result is
    /// not claimed here, chelis#2644). The frame then only holds the bound
    /// extent for later sites.
    result_axes: Vec<usize>,
}

impl FirstSiteFrame {
    /// The axes array always has a slot: the first one holds the bound
    /// extent even when no result axis names the binder.
    fn slots(&self) -> usize {
        self.result_axes.len().max(1)
    }
}

/// One function's first-site frames, one per output-inferred binder a body
/// site names, created as the body's sites are emitted.
struct FirstSiteFrames {
    /// The declared result's dimensions when the result is a tensor.
    result_dims: Option<Vec<DimInfo>>,
    frames: Vec<FirstSiteFrame>,
    /// The binders a `List` parameter's elements name, in the order of the
    /// body's `__chelis_entry_named_states`, which recorded each one from
    /// the elements when the invocation started; empty when the body holds
    /// no such states.
    list_states: Vec<String>,
}

impl FirstSiteFrames {
    fn of(function: &HostFunction, list_states: Vec<String>) -> Self {
        Self {
            result_dims: match &function.ret_ty {
                HostAbiType::Tensor(ty) => Some(ty.dims.clone()),
                _ => None,
            },
            frames: Vec::new(),
            list_states,
        }
    }

    /// The frame of output-inferred `binder`, created on its first site.
    fn frame(&mut self, binder: &str) -> &mut FirstSiteFrame {
        let index = match self.frames.iter().position(|frame| frame.binder == binder) {
            Some(index) => index,
            None => {
                let index = self.frames.len();
                let result_axes = self
                    .result_dims
                    .iter()
                    .flatten()
                    .enumerate()
                    .filter(|(_, dim)| {
                        matches!(dim, DimInfo::Named(name, _)
                            if chelis_ir::lower::extent_binder_label(name) == binder)
                    })
                    .map(|(axis, _)| axis)
                    .collect();
                self.frames.push(FirstSiteFrame {
                    binder: binder.to_string(),
                    axes_name: format!("__chelis_first_site_axes_{index}"),
                    frame_name: format!("__chelis_first_site_frame_{index}"),
                    result_axes,
                });
                index
            }
        };
        &mut self.frames[index]
    }

    /// The frames a tensor result is claimed against, in declared order.
    fn result_claims(&self) -> Vec<&FirstSiteFrame> {
        let mut frames = self
            .frames
            .iter()
            .filter(|frame| !frame.result_axes.is_empty())
            .collect::<Vec<_>>();
        frames.sort_by_key(|frame| frame.result_axes.first().copied());
        frames
    }

    /// Declare every frame. The ones the result is claimed against are
    /// chained in declared order in front of `parent`; return the chain's
    /// head.
    fn declare(&self, indent: &str, parent: &str) -> (Vec<String>, String) {
        let mut lines = Vec::new();
        for frame in self
            .frames
            .iter()
            .filter(|frame| frame.result_axes.is_empty())
        {
            lines.push(format!(
                "{indent}__chelis_host_result_axis {}[] = {{ {{ 0, 0, {}, NULL, 0 }} }};",
                frame.axes_name,
                c_string_literal(&frame.binder)
            ));
            lines.push(format!(
                "{indent}__chelis_host_result_claim {} = {{ NULL, 0, 0, {}, 0 }};",
                frame.frame_name, frame.axes_name
            ));
        }
        let rank = match &self.result_dims {
            Some(dims) => dims.len(),
            None => 0,
        };
        let mut next = parent.to_string();
        for frame in self.result_claims().into_iter().rev() {
            lines.push(format!(
                "{indent}__chelis_host_result_axis {}[] = {{",
                frame.axes_name
            ));
            for axis in &frame.result_axes {
                lines.push(format!(
                    "{indent}    {{ {axis}, 0, {}, NULL, 0 }},",
                    c_string_literal(&frame.binder)
                ));
            }
            lines.push(format!("{indent}}};"));
            lines.push(format!(
                "{indent}__chelis_host_result_claim {} = {{ {next}, {rank}, 0, {}, 0 }};",
                frame.frame_name, frame.axes_name
            ));
            next = format!("&{}", frame.frame_name);
        }
        (lines, next)
    }
}

struct HostTensorHelpers<'a> {
    helpers: &'a [HostTensorHelper],
    output_types: &'a [Vec<TensorType>],
    result_origins: Vec<Option<String>>,
}

/// The verified direct-call operands identify logical owners, independent of
/// source variable spellings or emitted C temporaries. A receipt crosses an
/// edge only when every value observed by the entry contract is the same
/// positional owner that the caller admitted.
fn call_forwards_entry_owners(
    site: &ProjectedHostSite<'_>,
    owners: &[(usize, VerifiedOwnerId)],
) -> bool {
    let clones = site
        .directives
        .iter()
        .filter_map(|action| match action {
            VerifiedHostAction::Operation(VerifiedHostOperation::Clone {
                dest, source, ..
            }) => Some((dest.id(), source.owner().id())),
            _ => None,
        })
        .collect::<Vec<_>>();
    let original_owner = |mut owner| {
        for _ in 0..clones.len() {
            let Some((_, source)) = clones.iter().find(|(dest, _)| *dest == owner) else {
                break;
            };
            owner = *source;
        }
        owner
    };
    let mut calls = site.directives.iter().filter_map(|action| match action {
        VerifiedHostAction::Operation(VerifiedHostOperation::Apply {
            kind: VerifiedApplyKind::DirectCall { .. },
            args,
            ..
        }) => Some(args),
        _ => None,
    });
    let Some(args) = calls.next() else {
        return false;
    };
    calls.next().is_none()
        && owners.iter().all(|(position, owner)| {
            args.get(*position)
                .is_some_and(|arg| original_owner(arg.owner().id()) == *owner)
        })
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
/// a *direct* call. Missing or duplicated structural direct-call identity
/// fails closed here; the general pre-call hook below separately admits a
/// verifier-authorized closed key callback application.
#[cfg(test)]
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

fn verified_call_action_index(site: &ProjectedHostSite<'_>) -> Result<usize, Unsupported> {
    let mut call = None;
    for (index, action) in site.directives.iter().enumerate() {
        if matches!(
            action,
            VerifiedHostAction::Operation(VerifiedHostOperation::Apply {
                kind: VerifiedApplyKind::DirectCall { .. } | VerifiedApplyKind::KeyBuiltinCall(_),
                ..
            })
        ) && call.replace(index).is_some()
        {
            return Err(invalid_abi_shape(
                "verified call site contains multiple call authorities".to_string(),
                "verified C host ownership emission",
            ));
        }
    }
    call.ok_or_else(|| {
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
            tensor_nan_finalization: None,
            indent,
            helper_prefix: helper_prefix.to_string(),
            emitted_names,
            function_specializations,
            tensor_helpers: tensor_helpers.helpers,
            tensor_helper_output_types: tensor_helpers.output_types,
            tensor_helper_result_origins: tensor_helpers.result_origins,
            expression_sites: ownership_sites
                .iter()
                .filter(|site| site.kind == chelis_ir::ownership::HostSiteKind::Expression)
                .cloned()
                .collect(),
            expression_site_index: 0,
            entry_projection: entry::Projection::default(),
            external_helpers: UnordSet::new(),
            interface_reload_names: UnordSet::new(),
            pre_emitted_clone_sites: UnordSet::new(),
            pre_emitted_terminals: UnordSet::new(),
            owner_vars: UnordMap::new(),
            entry_group: None,
            entry_groups: UnordMap::new(),
            entry_proof_owners: Vec::new(),
            temp_counter: 0,
            result_claims: None,
            claim_on_spine: false,
            first_site_frames: None,
        }
    }

    /// chelis#2120: fill a freshly allocated tensor with a `[05-OP-8]`
    /// uniform draw in the C HOST lane.
    ///
    /// The host lane draws with the key its first argument computed, so the
    /// two rules below are what keep it in step with `chelis eval`
    /// (`chelis-compiler-api` `runtime/eval.rs` `"uniform_like"`):
    ///
    /// 1. **The key is read after the arguments.** `arg_vars` are already
    ///    emitted when this runs, matching the evaluator's left-to-right
    ///    argument evaluation. The key is read once, into a temporary, and
    ///    never inside the element loop.
    /// 2. **A foldable bound is re-folded from the structural `args`, not
    ///    read from `arg_vars`.** This stamps the same exact bit pattern the
    ///    evaluator computes, so a template that folds and one that does not
    ///    sample identically; any other bound is the host scalar the
    ///    arguments computed.
    ///
    ///    This fold honours every rounding in a bound's cast chain, so a
    ///    bound spelled `cast(cast(x, f16), f32)` bakes the value `eval`
    ///    computes. Both lanes stage a bound the same way — an integer leaf
    ///    exact through i64, a float leaf at its source dtype until a cast
    ///    finalizes it — and both reach `finalize_scalar` through the shared
    ///    `chelis_types` cast primitives, so they take the same roundings in
    ///    the same order (chelis#2316). The two stagings are separate
    ///    readers of differently-shaped trees; they are NOT one function.
    ///
    ///    Keeping the lanes identical here is not stylistic. Correcting one
    ///    lane alone makes template foldability observable again, which is
    ///    the defect chelis#2120 exists to remove — a first revision of the
    ///    chelis#2316 fix did exactly that, and its regression witnesses are
    ///    `a_suffixed_literal_inside_a_narrowing_cast_agrees_across_lanes`
    ///    and its siblings. Change both folds together or neither.
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
        // [05-OP-8]: `uniform_like(k, template, low, high)`; the key is the
        // first operand, and the rest keep their positions relative to it.
        let key_var = match arg_vars.first() {
            Some((var, HostType::Key)) => var.clone(),
            _ => {
                return Err(Unsupported::new(
                    UnsupportedKind::Builtin("uniform_like".to_string()),
                    "`chelis build` host emission",
                    Stage::Codegen("c"),
                    chelis_types::deliberate_rejection!(
                        "[04-TOT-2]",
                        "uniform_like's first operand must be a key; the checker types it \
                         as one, so a non-key here is an internal desync"
                    ),
                ));
            }
        };
        // The key check above proves both slices non-empty.
        let args = &args[1..];
        let arg_vars = &arg_vars[1..];
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
        // [05-OP-8]: each bound is an f32 scalar operand. A bound the emitter
        // can fold is stamped as its exact f32 image, as the DAG lane folds
        // it; any other bound is the host scalar the arguments computed.
        let low_f32_expr = uniform_bound_f32_expr(args.get(1), arg_vars.get(1), "low")?;
        let high_f32_expr = uniform_bound_f32_expr(args.get(2), arg_vars.get(2), "high")?;
        let low_f64_expr = format!("((double)({low_f32_expr}))");
        let high_f64_expr = format!("((double)({high_f32_expr}))");

        let key = format!("{key_var}.bits");
        // [05-OP-8] validates finite bounds, `low <= high` and a finite
        // difference at the draw's arithmetic width before any element is
        // drawn: f64 subtracts the widened bounds, every other float dtype
        // subtracts in f32.
        let low = self.next_temp("uniform_low");
        let high = self.next_temp("uniform_high");
        let dtype = self.next_temp("uniform_dtype");
        let ind = self.indent.clone();
        self.lines.push(format!(
            "{ind}float {low} = {low_f32_expr}, {high} = {high_f32_expr};"
        ));
        self.lines.push(format!(
            "{ind}int {dtype} = (int)chelis_host_tensor_dtype({template});"
        ));
        self.lines.push(format!(
            "{ind}if (!(isfinite({low}) && isfinite({high}) && {low} <= {high} && ({dtype} == {f64} ? isfinite((double){high} - (double){low}) : isfinite({high} - {low})))) {{",
            f64 = chelis_vocab::RuntimeDType::F64.c_macro(),
        ));
        self.lines.push(format!("{ind}    switch ({dtype}) {{"));
        for (runtime, prim) in [
            (chelis_vocab::RuntimeDType::F32, Prim::F32),
            (chelis_vocab::RuntimeDType::F64, Prim::F64),
            (chelis_vocab::RuntimeDType::F16, Prim::F16),
            (chelis_vocab::RuntimeDType::Bf16, Prim::Bf16),
        ] {
            let trap = chelis_types::NumericTrap::Domain {
                op: "uniform_like",
                prim,
            }
            .to_string();
            self.lines.push(format!(
                "{ind}    case {}: chelis_numeric_trap({trap:?}); break;",
                runtime.c_macro()
            ));
        }
        self.lines.push(format!(
            "{ind}    default: fprintf(stderr, \"uniform_like unsupported dtype %d\\n\", {dtype}); abort();"
        ));
        self.lines.push(format!("{ind}    }}"));
        self.lines.push(format!("{ind}}}"));
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
            "chelis_uniform_sample_f32({key}, (uint64_t)i, {low_f32_expr}, {high_f32_expr})"
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
                    "chelis_uniform_sample_f64({key}, (uint64_t)i, {low_f64_expr}, {high_f64_expr})"
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
        self.declare_local(target, ty)?;
        self.assign_expr(target, expr, ty)?;
        Ok(())
    }

    /// Declare a local that an expression will be assigned to, with the
    /// result origin every assignment writes alongside the value.
    fn declare_local(&mut self, target: &str, ty: &HostType) -> Result<(), Unsupported> {
        self.lines
            .push(format!("{}{};", self.indent, c_decl(ty, target)?));
        self.declare_result_origin(target, ty, None);
        Ok(())
    }

    fn declare_result_origin(&mut self, value: &str, ty: &HostType, producer: Option<&str>) {
        let origin = result_origin_name(value);
        let initializer = match producer {
            Some("load") => Self::interface_result_origin_expr(ty),
            Some(op) if matches!(ty, HostType::Tensor(_)) => {
                let op = chelis_ir::span_sanitize::sanitize_for_format_string(op);
                format!(
                    "__chelis_host_result_origin_leaf(__chelis_origin_arena, \"{op}\", \"numeric trap: domain in {op} at i64\")"
                )
            }
            _ => "NULL".to_string(),
        };
        self.lines.push(format!(
            "{}const __chelis_host_result_origin *{origin} = {initializer};",
            self.indent
        ));
    }

    /// The origin of a value that entered from outside the expression: a
    /// parameter, a loop item, or a callee without private provenance. It is
    /// one uniform `load` leaf, so no ingress walks the value, and a
    /// recursive call over a large carried value stays linear (chelis#2522).
    fn interface_result_origin_expr(ty: &HostType) -> String {
        if host_type_may_carry_result_origin(ty) {
            "__chelis_host_result_origin_load(__chelis_origin_arena)".to_string()
        } else {
            "NULL".to_string()
        }
    }

    fn assign_interface_result_origin(&mut self, target: &str, ty: &HostType) {
        let origin = result_origin_name(target);
        let initializer = Self::interface_result_origin_expr(ty);
        self.lines
            .push(format!("{}{origin} = {initializer};", self.indent));
    }

    fn assign_aggregate_result_origin(&mut self, target: &str, children: &[String]) {
        let origin = result_origin_name(target);
        if children.is_empty() {
            self.lines.push(format!("{}{origin} = NULL;", self.indent));
            return;
        }
        let child_array = self.next_temp("result_origin_children");
        self.lines.push(format!(
            "{}const __chelis_host_result_origin *{child_array}[{}] = {{ {} }};",
            self.indent,
            children.len(),
            children
                .iter()
                .map(|child| result_origin_name(child))
                .collect::<Vec<_>>()
                .join(", ")
        ));
        self.lines.push(format!(
            "{}{origin} = __chelis_host_result_origin_aggregate(__chelis_origin_arena, {}, {child_array});",
            self.indent,
            children.len()
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
                    label: "builtin:copy" | "builtin:debug" | "extent_sites",
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
        let call_index = verified_call_action_index(site)?;
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
                }) if index < call_index => {
                    self.emit_owner_drop(owner.owner())?;
                    self.pre_emitted_terminals.insert((site.id, *operation));
                }
                VerifiedHostAction::Operation(VerifiedHostOperation::Discard {
                    operation, ..
                }) if index < call_index => {
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
            self.emit_expression_block_action(site, block, target, action)?;
        }
        Ok(())
    }

    /// Emit a list-building loop body's block actions in verified order, with
    /// the loop's consuming step emitted at the step's own position.
    ///
    /// The step (`list_push`, `list_extend`, `filter_step`,
    /// `partition_step`) moves an operand the body produced, and the actions
    /// the schedule places before it (the copy that pays for a captured,
    /// parameter or global owner the body returns, and every earlier release)
    /// must run first: a consuming entry point may release its operand before
    /// returning. Emitting the step from the verified action sequence, rather
    /// than ahead of every block action, makes that order the schedule's
    /// (chelis#2508).
    ///
    /// The step's result is the accumulator in `target`. The callback's
    /// result is the owner `callback_result` locates in the verified block,
    /// bound to the C variable the loop wrote it into: a named callback's
    /// call is an action of this block, and binding its result to `target`
    /// made the copy `scan` pushes retain the output list (chelis#2578). Any
    /// other value the block defines has no C variable here, so it is
    /// refused rather than bound by elimination.
    fn emit_loop_step_block_actions(
        &mut self,
        site: &ProjectedHostSite<'a>,
        block: VerifiedBlockId,
        target: &str,
        callback_result: LoopCallbackResult<'_>,
        step: &str,
        emit_step: impl FnOnce(&mut Self) -> Result<(), Unsupported>,
    ) -> Result<(), Unsupported> {
        let result_owner = site
            .directives
            .iter()
            .find_map(|action| match (action, callback_result) {
                (
                    VerifiedHostAction::Operation(VerifiedHostOperation::Apply {
                        block: owner_block,
                        label,
                        args,
                        ..
                    }),
                    LoopCallbackResult::StepArgument(_, index),
                ) if *owner_block == block && *label == step => {
                    args.get(index).map(|operand| operand.owner().id())
                }
                (
                    VerifiedHostAction::Terminator(VerifiedHostTerminator::Jump {
                        block: owner_block,
                        edge,
                    }),
                    LoopCallbackResult::BackEdgeArgument(_, index),
                ) if *owner_block == block => {
                    edge.args().get(index).map(|operand| operand.owner().id())
                }
                _ => None,
            })
            .ok_or_else(|| {
                invalid_abi_shape(
                    format!("verified `{step}` loop body has no {callback_result:?} operand"),
                    "verified C host ownership emission",
                )
            })?;
        let mut emit_step = Some(emit_step);
        for action in &site.directives {
            let binding = match action {
                VerifiedHostAction::Operation(VerifiedHostOperation::Apply {
                    block: owner_block,
                    label,
                    ..
                }) if *owner_block == block && *label == step => {
                    let emit = emit_step.take().ok_or_else(|| {
                        invalid_abi_shape(
                            format!("verified loop body repeats its `{step}` step"),
                            "verified C host ownership emission",
                        )
                    })?;
                    emit(self)?;
                    target
                }
                VerifiedHostAction::Operation(VerifiedHostOperation::Apply {
                    block: owner_block,
                    dest: Some(dest),
                    ..
                })
                | VerifiedHostAction::Operation(VerifiedHostOperation::Define {
                    block: owner_block,
                    dest,
                    ..
                }) if *owner_block == block => {
                    if dest.id() != result_owner {
                        return Err(invalid_abi_shape(
                            format!(
                                "verified `{step}` loop body defines {:?}, which is neither its step nor its callback's result",
                                dest.id()
                            ),
                            "verified C host ownership emission",
                        ));
                    }
                    callback_result.variable()
                }
                _ => target,
            };
            self.emit_expression_block_action(site, block, binding, action)?;
        }
        if emit_step.is_some() {
            return Err(invalid_abi_shape(
                format!("verified loop body has no `{step}` step"),
                "verified C host ownership emission",
            ));
        }
        Ok(())
    }

    fn emit_expression_block_action(
        &mut self,
        site: &ProjectedHostSite<'a>,
        block: VerifiedBlockId,
        target: &str,
        action: &VerifiedHostAction<'a>,
    ) -> Result<(), Unsupported> {
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
        Ok(())
    }

    /// Emit the terminals a site's verified schedule places in the entry block
    /// of one of its control regions (an `if` or match arm, or a loop body),
    /// at that entry, right after the region binds its payloads or loop item.
    ///
    /// The schedule puts a terminal right after its owner's last use, so a
    /// payload nothing reads, or a scrutinee whose last use is the payload
    /// extraction, is released in the region's entry block. When the region's
    /// body has control flow of its own, the region completes in a later block,
    /// and the completion pass sees only that block: such a terminal was never
    /// emitted in a match arm (chelis#2485, chelis#2458), and a loop emitted it
    /// after the loop, outside the item's C scope. A region that completes in
    /// its entry block is left to the completion pass, which emits the same
    /// terminals after the body. Every region entry goes through here, including
    /// those where lowering places no site action today (`if` arms, `None`
    /// arms, the default ADT arm), so no entry can drop a terminal silently.
    fn emit_region_entry_terminals(
        &mut self,
        site: &ProjectedHostSite<'a>,
        entry: VerifiedBlockId,
        completion: VerifiedBlockId,
    ) -> Result<(), Unsupported> {
        if entry == completion {
            return Ok(());
        }
        for action in &site.directives {
            if !Self::action_is_in_block(action, entry) {
                continue;
            }
            match action {
                // The region's own binders, bound by `bind_match_payload` and
                // `bind_loop_item` before this runs.
                VerifiedHostAction::Operation(VerifiedHostOperation::Apply {
                    dest: Some(_),
                    label,
                    ..
                }) if label.starts_with("option_payload") || label.starts_with("adt_payload") => {}
                VerifiedHostAction::Operation(VerifiedHostOperation::LoopItem { .. }) => {}
                VerifiedHostAction::Operation(VerifiedHostOperation::Drop {
                    operation,
                    owner,
                    ..
                }) => {
                    if self.pre_emitted_terminals.insert((site.id, *operation)) {
                        self.emit_owner_drop(owner.owner())?;
                    }
                }
                VerifiedHostAction::Operation(VerifiedHostOperation::Discard {
                    operation, ..
                }) => {
                    self.pre_emitted_terminals.insert((site.id, *operation));
                }
                other => {
                    return Err(invalid_abi_shape(
                        format!(
                            "verified control-region entry block {entry:?}, which completes in {completion:?}, carries an action the C emitter cannot place at the region entry: {other:?}"
                        ),
                        "verified C host ownership emission",
                    ));
                }
            }
        }
        Ok(())
    }

    /// Bind the loop item and emit the terminals scheduled in the loop body's
    /// entry block ([`Self::emit_region_entry_terminals`]).
    fn bind_loop_item(
        &mut self,
        site: &ProjectedHostSite<'a>,
        emitted_var: &str,
    ) -> Result<(), Unsupported> {
        let mut items = site.directives.iter().filter_map(|action| match action {
            VerifiedHostAction::Operation(VerifiedHostOperation::LoopItem {
                dest, block, ..
            }) => Some((*dest, *block)),
            _ => None,
        });
        let Some((owner, entry)) = items.next() else {
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
        let (_, completion) = Self::loop_blocks(site)?;
        self.emit_region_entry_terminals(site, entry, completion)
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
        self.assign_expr_at_site(target, expr, ty, &site, on_result_spine)?;
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
                let axes = self.next_temp("result_claim_axes");
                let frame = self.next_temp("result_claim_frame");
                let parent = result_claims.as_deref().unwrap_or("NULL");
                self.lines
                    .extend(HostResultClaim::from_plan(plan).frame_lines(
                        &self.indent,
                        &axes,
                        &frame,
                        parent,
                        None,
                        plan.outer_claims_first(),
                    ));
                let previous_claims = self.result_claims.replace(format!("&{frame}"));
                self.claim_on_spine = true;
                self.assign_expr(target, body, ty)?;
                self.result_claims = previous_claims;
                self.emit_expression_site(site, target)?;
                return Ok(());
            }
            HostExprKind::ExtentSites {
                value,
                sites,
                ty: sites_ty,
            } => {
                require_same_abi_type(ty, sites_ty, "local ascription extent sites")?;
                self.claim_on_spine = on_result_spine;
                self.assign_expr(target, value, ty)?;
                self.emit_extent_sites(target, ty, sites)?;
                self.emit_expression_site(site, target)?;
                return Ok(());
            }
            HostExprKind::FormalIngress {
                value,
                ty: ingress_ty,
            } => {
                require_same_abi_type(ty, ingress_ty, "formal ingress")?;
                self.claim_on_spine = false;
                self.assign_expr(target, value, ty)?;
                self.assign_interface_result_origin(target, ty);
                self.emit_result_claim_guard(target, ty, result_claims.as_deref());
                self.claim_on_spine = on_result_spine;
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
                    let target_origin = result_origin_name(target);
                    if self.interface_reload_names.contains(name) {
                        let load = Self::interface_result_origin_expr(ty);
                        self.lines
                            .push(format!("{}{target_origin} = {load};", self.indent));
                    } else {
                        self.lines.push(format!(
                            "{}{target_origin} = {};",
                            self.indent,
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
                self.assign_builtin(target, name, args, ty, site, result_claims.as_deref())?;
                if chelis_ir::host::produces_its_result(name) {
                    self.stamp_combinator_result_origin(target, ty, name);
                } else if !matches!(name.as_str(), "tuple-get" | "index") {
                    self.stamp_result_origin(target, ty, name);
                }
                if name != "pad_sequences_to" {
                    self.emit_result_claim_guard(target, ty, result_claims.as_deref());
                }
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
                self.emit_result_claim_guard(target, ty, result_claims.as_deref());
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
                self.emit_region_entry_terminals(site, then_edge.target(), then_block)?;
                self.claim_on_spine = on_result_spine;
                self.assign_expr(target, then_expr, ty)?;
                self.emit_expression_block_actions(site, then_block, target)?;
                self.indent = previous.clone();
                self.lines.push(format!("{}}} else {{", self.indent));
                let nested_indent = format!("{}    ", self.indent);
                let previous = std::mem::replace(&mut self.indent, nested_indent);
                self.emit_edge_terminals(site.id, &else_edge)?;
                self.emit_region_entry_terminals(site, else_edge.target(), else_block)?;
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
                let (some_edge, none_edge) = site
                    .directives
                    .iter()
                    .find_map(|action| match action {
                        VerifiedHostAction::Terminator(VerifiedHostTerminator::Match {
                            arms,
                            ..
                        }) if arms.len() == 2 => Some((arms[0].clone(), arms[1].clone())),
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
                    }
                    other => {
                        return Err(invalid_abi_shape(
                            format!("option match scrutinee has non-option ABI type `{other:?}`"),
                            "option match",
                        ));
                    }
                }
                self.declare_result_origin(bind_name, &inner_ty, None);
                self.lines.push(format!(
                    "{}{} = __chelis_host_result_origin_child({}, 0);",
                    self.indent,
                    result_origin_name(bind_name),
                    result_origin_name(&option_var)
                ));
                let shadowed_interface_global = self.interface_reload_names.remove(bind_name);
                self.bind_match_payload(site, some_edge.target(), "option_payload", bind_name)?;
                // chelis#1222: the binder shadows any enclosing name it
                // reuses. Its key carries no outgoing edge, because the
                // value is freshly extracted here rather than copied from
                // something this scope already owns -- which is also what
                // keeps emitted C unchanged for every program that does not
                // shadow: a reference to it dead-ends exactly as it does
                // today.
                self.emit_region_entry_terminals(site, some_edge.target(), arm_blocks.0)?;
                self.claim_on_spine = on_result_spine;
                self.assign_expr(target, some_expr, ty)?;
                self.emit_expression_block_actions(site, arm_blocks.0, target)?;
                if shadowed_interface_global {
                    self.interface_reload_names.insert(bind_name.clone());
                }
                self.indent = previous.clone();
                self.lines.push(format!("{}}} else {{", self.indent));
                let nested_indent = format!("{}    ", self.indent);
                let previous = std::mem::replace(&mut self.indent, nested_indent);
                self.emit_edge_terminals(site.id, &none_edge)?;
                self.emit_region_entry_terminals(site, none_edge.target(), arm_blocks.1)?;
                self.claim_on_spine = on_result_spine;
                self.assign_expr(target, none_expr, ty)?;
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
                let mut shadowed_interface_globals = Vec::new();
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
                    self.declare_result_origin(&binding.name, &binding.ty, None);
                    // #379: assign to the same mangled identifier the
                    // declaration used (both route through `c_ident`).
                    self.lines.push(format!(
                        "{}{} = {};",
                        self.indent,
                        c_ident(&binding.name),
                        temp
                    ));
                    self.lines.push(format!(
                        "{}{} = {};",
                        self.indent,
                        result_origin_name(&binding.name),
                        result_origin_name(&temp)
                    ));
                    if self.interface_reload_names.remove(&binding.name) {
                        shadowed_interface_globals.push(binding.name.clone());
                    }
                }
                self.claim_on_spine = on_result_spine && spine_binding.is_none();
                self.assign_expr(target, body, ty)?;
                self.emit_expression_site(site, target)?;
                self.interface_reload_names
                    .extend(shadowed_interface_globals);
                self.indent = previous;
                self.lines.push(format!("{}}}", self.indent));
                return Ok(());
            }
            HostExprKind::RetainedInvocation {
                bindings,
                body,
                ty: expr_ty,
            } => {
                require_same_abi_type(ty, expr_ty, "retained invocation")?;
                self.lines.push(format!("{}{{", self.indent));
                let nested_indent = format!("{}    ", self.indent);
                let previous = std::mem::replace(&mut self.indent, nested_indent);
                let mut shadowed_interface_globals = Vec::new();
                for binding in bindings {
                    let temp = self.next_temp("retained_actual");
                    // A callee result contract never constrains actual
                    // preparation, formal ingress, or signature entry.
                    self.claim_on_spine = false;
                    self.emit_expr_to_var(&binding.value, &temp, &binding.ty)?;
                    self.lines.push(format!(
                        "{}{};",
                        self.indent,
                        c_decl(&binding.ty, &binding.name)?
                    ));
                    self.declare_result_origin(&binding.name, &binding.ty, None);
                    self.lines.push(format!(
                        "{}{} = {};",
                        self.indent,
                        c_ident(&binding.name),
                        temp
                    ));
                    self.lines.push(format!(
                        "{}{} = {};",
                        self.indent,
                        result_origin_name(&binding.name),
                        result_origin_name(&temp)
                    ));
                    if self.interface_reload_names.remove(&binding.name) {
                        shadowed_interface_globals.push(binding.name.clone());
                    }
                }
                self.claim_on_spine = on_result_spine;
                self.assign_expr(target, body, ty)?;
                self.emit_expression_site(site, target)?;
                self.interface_reload_names
                    .extend(shadowed_interface_globals);
                self.indent = previous;
                self.lines.push(format!("{}}}", self.indent));
                return Ok(());
            }
            HostExprKind::Map { callback, list, ty } => {
                let (_, body_block) = Self::loop_blocks(site)?;
                self.assign_map(target, callback, list, ty, site, body_block)?;
                self.stamp_combinator_result_origin(target, ty, "map");
                self.emit_expression_site_excluding_block(site, target, Some(body_block))?;
                return Ok(());
            }
            HostExprKind::Filter { callback, list, ty } => {
                let (_, body_block) = Self::loop_blocks(site)?;
                self.assign_filter(target, callback, list, ty, site, body_block)?;
                self.stamp_combinator_result_origin(target, ty, "filter");
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
                // chelis#2581: `fold` itself produces a returned tensor,
                // whichever iteration or seed supplied it, so it stamps and
                // guards that value exactly as a builtin producer does. A
                // tensor nested in an aggregate result is its too.
                self.stamp_combinator_result_origin(target, ty, "fold");
                self.emit_result_claim_guard(target, ty, result_claims.as_deref());
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
                let (preheader_block, body_block) = Self::loop_blocks(site)?;
                self.assign_scan(
                    target,
                    callback,
                    init,
                    list,
                    ty,
                    site,
                    preheader_block,
                    body_block,
                )?;
                self.stamp_combinator_result_origin(target, ty, "scan");
                self.emit_expression_site_excluding_blocks(
                    site,
                    target,
                    &[preheader_block, body_block],
                )?;
                return Ok(());
            }
            HostExprKind::Partition { callback, list, ty } => {
                let (_, body_block) = Self::loop_blocks(site)?;
                self.assign_partition(target, callback, list, ty, site, body_block)?;
                self.stamp_combinator_result_origin(target, ty, "partition");
                self.emit_expression_site_excluding_block(site, target, Some(body_block))?;
                return Ok(());
            }
            HostExprKind::FlatMap { callback, list, ty } => {
                let (_, body_block) = Self::loop_blocks(site)?;
                self.assign_flat_map(target, callback, list, ty, site, body_block)?;
                self.stamp_combinator_result_origin(target, ty, "flat_map");
                self.emit_expression_site_excluding_block(site, target, Some(body_block))?;
                return Ok(());
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
            HostExprKind::SignatureEntry {
                contract,
                plan,
                args,
                positions,
                lists,
            } => {
                require_same_abi_type(ty, &HostType::Unit, "signature entry")?;
                validate_retained_entry_contract(contract, plan, positions, lists)?;
                let mut actuals = Vec::with_capacity(args.len());
                for arg in args {
                    let temp = self.next_temp("entry_arg");
                    self.emit_expr_to_var(arg, &temp, &host_type(arg))?;
                    actuals.push(temp);
                }
                if lists.is_empty() {
                    self.lines.extend(signature_entry_lines(
                        plan,
                        &actuals,
                        &self.indent,
                        &[],
                        None,
                        true,
                    )?);
                } else {
                    let count = contract.formals().len();
                    let mut work = entry_walk::FunctionEntryWork {
                        args: actuals.clone(),
                        owners: positions.clone(),
                        params: (0..count)
                            .map(|_| entry_walk::ParamEntryWork::default())
                            .collect(),
                        named_list_binders: contract.named_list_binders().to_vec(),
                        release: Vec::new(),
                    };
                    if positions.len() != args.len() {
                        return Err(invalid_abi_shape(
                            "retained List entry lost its fixed tensor observations".into(),
                            "signature entry",
                        ));
                    }
                    for (index, position) in positions.iter().enumerate() {
                        work.params[*position]
                            .ordered_extents
                            .push(entry_walk::ExtentStep::Fixed(index));
                    }
                    for entry in lists {
                        let temp = self.next_temp("entry_list");
                        self.emit_expr_to_var(&entry.value, &temp, &host_type(&entry.value))?;
                        work.params[entry.position].metadata.extend(
                            entry_walk::retained_list_pass(
                                &entry.ty,
                                &temp,
                                &entry.name,
                                false,
                                work.named_list_binders.len(),
                            ),
                        );
                        let extent_lines = entry_walk::retained_list_pass(
                            &entry.ty,
                            &temp,
                            &entry.name,
                            true,
                            work.named_list_binders.len(),
                        );
                        let extent_call = extent_lines.join("\n");
                        work.params[entry.position]
                            .extents
                            .push(extent_call.clone());
                        work.params[entry.position]
                            .ordered_extents
                            .push(entry_walk::ExtentStep::Walk(extent_call));
                    }
                    self.lines.extend(signature_entry_lines(
                        plan,
                        &actuals,
                        &self.indent,
                        &[],
                        Some(&work),
                        true,
                    )?);
                }
                self.lines.push(format!("{}{target} = 0;", self.indent));
            }
            HostExprKind::Unit => {
                require_same_abi_type(ty, &HostType::Unit, "unit expression")?;
                self.lines.push(format!("{}{target} = 0;", self.indent));
            }
        }
        self.emit_expression_site(site, target)
    }

    /// spec/04 section 4.7: a list combinator produces every tensor it
    /// returns, directly or nested in its aggregate result.
    fn stamp_combinator_result_origin(&mut self, target: &str, ty: &HostType, op: &str) {
        if matches!(ty, HostType::Tensor(_)) {
            self.stamp_result_origin(target, ty, op);
            return;
        }
        if !matches!(
            ty,
            HostType::List(_)
                | HostType::Tuple(_)
                | HostType::Option(_)
                | HostType::Dict(..)
                | HostType::Adt(..)
        ) {
            return;
        }
        let origin = result_origin_name(target);
        self.lines.push(format!(
            "{}{origin} = __chelis_host_result_origin_uniform(__chelis_origin_arena, \"{op}\", \"numeric trap: domain in {op} at i64\");",
            self.indent
        ));
    }

    fn stamp_result_origin(&mut self, target: &str, ty: &HostType, op: &str) {
        if !matches!(ty, HostType::Tensor(_)) {
            return;
        }
        let op = chelis_ir::span_sanitize::sanitize_for_format_string(op);
        let origin = result_origin_name(target);
        self.lines.push(format!(
            "{}{origin} = __chelis_host_result_origin_leaf(__chelis_origin_arena, \"{op}\", \"numeric trap: domain in {op} at i64\");",
            self.indent
        ));
    }

    /// Relate the named sites of the local tensor ascription whose value is
    /// `target` to the invocation's other sites of the same output-inferred
    /// binder: an empty frame takes this site's extent, and a filled one is a
    /// section 4.7 claim that traps `Domain` at the value's producer. A
    /// binder a parameter declares has no frame here; the ascription's region
    /// claims it against that parameter.
    fn emit_extent_sites(
        &mut self,
        target: &str,
        ty: &HostType,
        sites: &[chelis_ir::lower::LocalAscriptionNamedSite],
    ) -> Result<(), Unsupported> {
        let origin = result_origin_name(target);
        self.emit_list_element_sites(target, ty, sites)?;
        // Only an output-inferred binder's site binds or claims a frame, and
        // each such binder gets its frame at its first site, whatever the
        // declared result's shape.
        for site in sites.iter().filter(|site| site.output_inferred) {
            if !matches!(ty, HostType::Tensor(_)) {
                return Err(invalid_abi_shape(
                    format!(
                        "local ascription `{}` names output-inferred `{}` on a non-tensor value",
                        site.binding, site.binder
                    ),
                    "verified C host first-site emission",
                ));
            }
            let Some(frames) = self.first_site_frames.as_mut() else {
                return Err(invalid_abi_shape(
                    format!(
                        "local ascription `{}` names output-inferred `{}` outside a function body",
                        site.binding, site.binder
                    ),
                    "verified C host first-site emission",
                ));
            };
            let frame = frames.frame(&site.binder);
            let (axes, frame_name) = (frame.axes_name.clone(), frame.frame_name.clone());
            let count = frame.slots();
            let indent = self.indent.clone();
            let axis = site.axis;
            // The site's extent is read where it is used rather than held in
            // a local, so this emitter spells no element type of its own.
            let extent = format!("chelis_tensor_shape({target}, {axis})");
            self.lines.push(format!("{indent}{{"));
            self.lines
                .push(format!("{indent}    if ({frame_name}.count == 0) {{"));
            for index in 0..count {
                self.lines.push(format!(
                    "{indent}        {axes}[{index}].required = {extent}; {axes}[{index}].source = {}; {axes}[{index}].source_axis = {axis};",
                    c_string_literal(&site.binding)
                ));
            }
            self.lines
                .push(format!("{indent}        {frame_name}.count = {count};"));
            self.lines.push(format!(
                "{indent}    }} else if ({axes}[0].required != {extent}) {{"
            ));
            self.lines.push(format!(
                "{indent}        if ({origin} == NULL || {origin}->child_count != -1 || {origin}->op == NULL || {origin}->trap == NULL) {{ fprintf(stderr, \"host runtime: an extent claim reached a tensor without producer provenance\\n\"); abort(); }}"
            ));
            self.lines.push(format!(
                "{indent}        fprintf(stderr, \"extent `%s`: claimed = %lld, %s axis %lld = %lld\\n\", {}, (long long){axes}[0].required, {origin}->op, (long long){axis}, (long long){extent});",
                c_string_literal(&site.binder)
            ));
            self.lines.push(format!(
                "{indent}        chelis_numeric_trap({origin}->trap);"
            ));
            self.lines.push(format!("{indent}    }}"));
            self.lines.push(format!("{indent}}}"));
        }
        Ok(())
    }

    /// Claim each site naming a binder a `List` parameter's elements name
    /// against the extent the invocation recorded from those elements
    /// (spec/04-type-system.md section 4.7), as `chelis eval` claims it
    /// against its activation. A List that held no tensor recorded none, so
    /// the site cannot resolve its extent and the program stops with the
    /// evaluator's refusal (chelis#3039).
    fn emit_list_element_sites(
        &mut self,
        target: &str,
        ty: &HostType,
        sites: &[chelis_ir::lower::LocalAscriptionNamedSite],
    ) -> Result<(), Unsupported> {
        let origin = result_origin_name(target);
        for site in sites
            .iter()
            .filter(|site| site.list_element && !site.output_inferred)
        {
            if !matches!(ty, HostType::Tensor(_)) {
                return Err(invalid_abi_shape(
                    format!(
                        "local ascription `{}` names List element binder `{}` on a non-tensor value",
                        site.binding, site.binder
                    ),
                    "verified C host List element site emission",
                ));
            }
            let binder = chelis_ir::lower::extent_binder_label(&site.binder);
            let state = self
                .first_site_frames
                .as_ref()
                .and_then(|frames| {
                    frames
                        .list_states
                        .iter()
                        .position(|name| chelis_ir::lower::extent_binder_label(name) == binder)
                })
                .ok_or_else(|| {
                    invalid_abi_shape(
                        format!(
                            "local ascription `{}` names List element binder `{}`, which the \
                             invocation records no extent for",
                            site.binding, site.binder
                        ),
                        "verified C host List element site emission",
                    )
                })?;
            let indent = self.indent.clone();
            let axis = site.axis;
            let recorded = format!("__chelis_entry_named_states[{state}]");
            let extent = format!("chelis_tensor_shape({target}, {axis})");
            self.lines.push(format!("{indent}if (!{recorded}.seen) {{"));
            self.lines.push(format!(
                "{indent}    fprintf(stderr, \"local tensor ascription `%s` cannot resolve authored extent `%s` in this activation\\n\", {}, {});",
                c_string_literal(&site.binding),
                c_string_literal(&binder)
            ));
            self.lines
                .push(format!("{indent}    chelis_flush_and_abort();"));
            self.lines.push(format!(
                "{indent}}} else if ({recorded}.value != {extent}) {{"
            ));
            self.lines.push(format!(
                "{indent}    if ({origin} == NULL || {origin}->child_count != -1 || {origin}->op == NULL || {origin}->trap == NULL) {{ fprintf(stderr, \"host runtime: an extent claim reached a tensor without producer provenance\\n\"); abort(); }}"
            ));
            self.lines.push(format!(
                "{indent}    fprintf(stderr, \"extent `%s`: claimed = %lld, %s axis %lld = %lld\\n\", {}, (long long){recorded}.value, {origin}->op, (long long){axis}, (long long){extent});",
                c_string_literal(&binder)
            ));
            self.lines
                .push(format!("{indent}    chelis_numeric_trap({origin}->trap);"));
            self.lines.push(format!("{indent}}}"));
        }
        Ok(())
    }

    /// Check each output-inferred binder's frame alone against the returned
    /// `target`, at the return, where the result is a tensor that names it.
    fn emit_late_first_site_checks(&mut self, target: &str) {
        let frame_names = match &self.first_site_frames {
            Some(frames) => frames
                .result_claims()
                .into_iter()
                .map(|frame| frame.frame_name.clone())
                .collect::<Vec<_>>(),
            None => Vec::new(),
        };
        if frame_names.is_empty() {
            return;
        }
        let origin = result_origin_name(target);
        let indent = self.indent.clone();
        self.lines.push(format!("{indent}{{"));
        self.lines.push(format!(
            "{indent}    const int __chelis_late_known = {origin} != NULL && {origin}->child_count == -1 && {origin}->op != NULL && {origin}->trap != NULL;"
        ));
        self.lines.push(format!(
            "{indent}    const char *__chelis_late_op = __chelis_late_known ? {origin}->op : \"return\";"
        ));
        self.lines.push(format!(
            "{indent}    const char *__chelis_late_trap = __chelis_late_known ? {origin}->trap : \"numeric trap: domain in return at i64\";"
        ));
        for frame_name in frame_names {
            self.lines.push(format!(
                "{indent}    {{ __chelis_host_result_claim __chelis_late = {frame_name}; __chelis_late.next = NULL; __chelis_check_host_result_claims(&__chelis_late, {target}, __chelis_late_op, __chelis_late_trap); }}"
            ));
        }
        self.lines.push(format!("{indent}}}"));
    }

    fn emit_result_claim_guard(&mut self, target: &str, ty: &HostType, claims: Option<&str>) {
        if let Some(claims) = claims
            && matches!(ty, HostType::Tensor(_))
        {
            let origin = result_origin_name(target);
            self.lines
                .push(format!("{}if ({claims} != NULL) {{", self.indent));
            self.lines.push(format!(
                "{}    if ({origin} == NULL || {origin}->child_count != -1 || {origin}->op == NULL || {origin}->trap == NULL) {{ fprintf(stderr, \"host runtime: pending result claim reached a tensor without producer provenance\\n\"); abort(); }}",
                self.indent
            ));
            self.lines.push(format!(
                "{}    __chelis_check_host_result_claims({claims}, {target}, {origin}->op, {origin}->trap);",
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
        result_claims: Option<&str>,
    ) -> Result<(), Unsupported> {
        if args.is_empty()
            && let HostType::KeyBuiltinCallable(op) = ty
        {
            if op.symbol() != name {
                return Err(invalid_abi_shape(
                    format!("closed key callable identity {op:?} disagrees with `{name}`"),
                    "C host key callable selection",
                ));
            }
            self.lines.push(format!(
                "{}{target} = (chelis_key_callable){{ NULL }};",
                self.indent
            ));
            return Ok(());
        }
        if args.is_empty() && matches!(ty, HostType::Callback(_, _)) {
            let symbol = key_callable_symbol(name, ty).ok_or_else(|| {
                invalid_abi_shape(
                    format!("checked key callable `{name}` has incompatible C signature {ty:?}"),
                    "C host key callable selection",
                )
            })?;
            self.lines
                .push(format!("{}{target} = {symbol};", self.indent));
            return Ok(());
        }
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

        // Tensor key forms preserve the checked rank; split_keys appends one
        // axis. Read every extent before allocation and every element through
        // the dtype-tagged views, as the tensor kernel emitter does.
        if matches!(
            name,
            "key_from_seed" | "split_key" | "split_keys" | "fold_in"
        ) && let HostType::Tensor(input_ty) = &arg_vars[0].1
        {
            let rank = input_ty.dims.len();
            let out_rank = rank + usize::from(name == "split_keys");
            let prefix = self.next_temp("key_tensor");
            let ind = self.indent.clone();
            let input = &arg_vars[0].0;
            self.lines.push(format!(
                "{ind}chelis_read_view {prefix}_in = chelis_tensor_read_view({input});"
            ));
            let dtype = if name == "key_from_seed" {
                "CHELIS_DTYPE_I64"
            } else {
                "CHELIS_DTYPE_KEY"
            };
            self.lines.push(format!("{ind}if ({prefix}_in.dtype != {dtype} || chelis_tensor_rank({input}) != {rank}) abort();"));
            self.lines
                .push(format!("{ind}int64_t {prefix}_shape[{}];", out_rank.max(1)));
            for axis in 0..rank {
                self.lines.push(format!(
                    "{ind}{prefix}_shape[{axis}] = chelis_tensor_shape({input}, {axis});"
                ));
            }
            if name == "fold_in" {
                let index = &arg_vars[1].0;
                self.lines.push(format!(
                    "{ind}chelis_read_view {prefix}_index = chelis_tensor_read_view({index});"
                ));
                self.lines.push(format!("{ind}if ({prefix}_index.dtype != CHELIS_DTYPE_I64 || chelis_tensor_rank({index}) != {rank}) abort();"));
                for axis in 0..rank {
                    self.lines.push(format!("{ind}if (chelis_tensor_shape({index}, {axis}) != {prefix}_shape[{axis}]) {{ fprintf(stderr, \"fold_in requires exactly equal key and index shapes ([05-OP-72])\\n\"); chelis_numeric_trap(\"numeric trap: domain in fold_in at i64\"); }}"));
                }
            }
            if name == "split_keys" {
                let count = &arg_vars[1].0;
                self.lines.push(format!("{ind}if ({count} < 0) chelis_numeric_trap(\"numeric trap: domain in split_keys at i64\");"));
                self.lines
                    .push(format!("{ind}{prefix}_shape[{rank}] = {count};"));
            }
            let halves = if name == "split_key" { 2 } else { 1 };
            for half in 0..halves {
                self.lines.push(format!("{ind}chelis_tensor *{prefix}_{half} = chelis_alloc({out_rank}, {prefix}_shape, CHELIS_DTYPE_KEY);"));
                self.lines.push(format!("{ind}chelis_tensor_write *{prefix}_guard_{half} = chelis_tensor_begin_write({prefix}_{half});"));
                self.lines.push(format!("{ind}chelis_write_view {prefix}_out_{half} = chelis_tensor_write_view({prefix}_guard_{half});"));
                let source = format!("((const uint64_t *){prefix}_in.data)[i]");
                let value = match name {
                    "key_from_seed" => format!("(uint64_t)((const int64_t *){prefix}_in.data)[i]"),
                    "split_key" => format!("chelis_key_derive({source}, {half}ULL)"),
                    "fold_in" => format!(
                        "chelis_key_derive(chelis_key_derive({source}, 2ULL), (uint64_t)((const int64_t *){prefix}_index.data)[i])"
                    ),
                    "split_keys" => {
                        format!("chelis_key_derive(chelis_key_derive({source}, 2ULL), (uint64_t)j)")
                    }
                    _ => unreachable!(),
                };
                if name == "split_keys" {
                    self.lines.push(format!("{ind}for (int64_t i = 0; i < {prefix}_in.count; ++i) for (int64_t j = 0; j < {prefix}_shape[{rank}]; ++j) ((uint64_t *){prefix}_out_{half}.data)[i * {prefix}_shape[{rank}] + j] = {value};"));
                } else {
                    self.lines.push(format!("{ind}for (int64_t i = 0; i < {prefix}_in.count; ++i) ((uint64_t *){prefix}_out_{half}.data)[i] = {value};"));
                }
                self.lines.push(format!(
                    "{ind}chelis_tensor_end_write({prefix}_guard_{half});"
                ));
            }
            if name == "split_key" {
                self.lines.push(format!("{ind}chelis_value {prefix}_halves[2] = {{ chelis_value_take_tensor({prefix}_0), chelis_value_take_tensor({prefix}_1) }};"));
                self.lines.push(format!(
                    "{ind}{target} = chelis_tuple_from_values({prefix}_halves, 2);"
                ));
                for half in 0..2 {
                    self.lines.push(format!(
                        "{ind}chelis_value_release({prefix}_halves[{half}]);"
                    ));
                }
            } else {
                self.lines.push(format!("{ind}{target} = {prefix}_0;"));
            }
            return Ok(());
        }

        // [05-OP-69]..[05-OP-72]: the key operations over host scalar keys,
        // with the same derivation the kernels use (`chelis_key_derive`).
        match name {
            "key_from_seed" => {
                self.lines.push(format!(
                    "{}{target} = chelis_key_from_seed_bits((long long){});",
                    self.indent, arg_vars[0].0
                ));
                return Ok(());
            }
            "fold_in" => {
                self.lines.push(format!(
                    "{}{target} = chelis_key_fold_in({}, (long long){});",
                    self.indent, arg_vars[0].0, arg_vars[1].0
                ));
                return Ok(());
            }
            "split_key" => {
                let halves = self.next_temp("split_key_values");
                let ind = self.indent.clone();
                let key = &arg_vars[0].0;
                self.lines.push(format!("{ind}chelis_value {halves}[2];"));
                for index in 0..2 {
                    self.lines.push(format!(
                        "{ind}{halves}[{index}] = {};",
                        self.box_value_expr(
                            &format!("chelis_key_derive_value({key}, {index}ULL)"),
                            &HostType::Key
                        )?
                    ));
                }
                // The tuple retains its items, so the boxes are released
                // here, as every other tuple construction releases them.
                self.lines.push(format!(
                    "{ind}{target} = chelis_tuple_from_values({halves}, 2);"
                ));
                for index in 0..2 {
                    self.lines
                        .push(format!("{ind}chelis_value_release({halves}[{index}]);"));
                }
                return Ok(());
            }
            "split_keys" => {
                self.lines.push(format!(
                    "{}{target} = chelis_split_keys_tensor({}, (long long){});",
                    self.indent, arg_vars[0].0, arg_vars[1].0
                ));
                return Ok(());
            }
            _ => {}
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
            // [04-NUM-2]: the elementwise loops below store each element
            // through the same classification as the scalar lane.
            self.tensor_nan_finalization = CExpressionBuiltin::decode(name)
                .ok()
                .and_then(CExpressionBuiltin::nan_finalization);
            // The loops below cover the dtypes their arms name. Host lowering
            // runs every other checked tensor operation in the tensor lane, so
            // a dtype outside the arms is refused here, before emission, and
            // never reaches a loop that would abort at run time (chelis#2734).
            if let Some(arms) = host_elementwise_arms(name, &arg_vars) {
                admit_host_elementwise(name, &arg_vars, arms)?;
            }
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
                // A logical operation whose operands are computed on the host,
                // such as two inline comparisons of run-time `to_tensor`
                // results, combines its bool operands element by element after
                // the same operand agreement check as arithmetic.
                "and" | "or"
                    if matches!(
                        (&arg_vars[0].1, &arg_vars[1].1),
                        (HostType::Tensor(_), HostType::Tensor(_))
                    ) =>
                {
                    self.assign_tensor_binary_elementwise(
                        target,
                        &arg_vars[0].0,
                        &arg_vars[1].0,
                        if name == "and" { "&&" } else { "||" },
                    );
                    return Ok(());
                }
                // A comparison whose operands are computed on the host, such as
                // two calls that each return a tensor, compares element by
                // element after the same operand agreement check, into a bool
                // tensor ([05-OP-36]; chelis#3000).
                "lt" | "lte" | "gt" | "gte" | "eq" | "neq"
                    if matches!(
                        (&arg_vars[0].1, &arg_vars[1].1),
                        (HostType::Tensor(_), HostType::Tensor(_))
                    ) =>
                {
                    self.assign_tensor_comparison_elementwise(
                        target,
                        &arg_vars[0].0,
                        &arg_vars[1].0,
                        ElementwiseComparison::from_builtin(name),
                    );
                    return Ok(());
                }
                // chelis#2076: `mod` and the bitwise and shift operations
                // over two tensors computed on the host apply the scalar rule
                // element by element after the same operand agreement check
                // ([05-OP-47], [05-OP-64]).
                "mod" | "bitand" | "bitor" | "bitxor" | "shl" | "shr"
                    if matches!(
                        (&arg_vars[0].1, &arg_vars[1].1),
                        (HostType::Tensor(_), HostType::Tensor(_))
                    ) =>
                {
                    self.assign_tensor_integer_elementwise(
                        target,
                        &arg_vars[0].0,
                        &arg_vars[1].0,
                        name,
                    );
                    return Ok(());
                }
                "exp" if matches!(&arg_vars[0].1, HostType::Tensor(_)) => {
                    self.assign_tensor_unary_func_elementwise(
                        target,
                        &arg_vars[0].0,
                        "chelis_cr_expf",
                    );
                    return Ok(());
                }
                "log" if matches!(&arg_vars[0].1, HostType::Tensor(_)) => {
                    self.assign_tensor_unary_func_elementwise(
                        target,
                        &arg_vars[0].0,
                        "chelis_cr_logf",
                    );
                    return Ok(());
                }
                "sin" if matches!(&arg_vars[0].1, HostType::Tensor(_)) => {
                    self.assign_tensor_unary_func_elementwise(
                        target,
                        &arg_vars[0].0,
                        "chelis_cr_sinf",
                    );
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
                self.assign_aggregate_result_origin(target, &[arg_vars[0].0.clone()]);
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
                self.lines.push(format!(
                    "{}{} = __chelis_host_result_origin_child({}, {});",
                    self.indent,
                    result_origin_name(target),
                    result_origin_name(&arg_vars[0].0),
                    arg_vars[1].0
                ));
                return Ok(());
            }
            "index" => {
                let value_var = self.next_temp("list_value");
                self.lines.push(format!(
                    "{}chelis_value {} = chelis_list_index({}, {});",
                    self.indent, value_var, arg_vars[0].0, arg_vars[1].0
                ));
                self.assign_unboxed_value(target, ty, &value_var)?;
                self.lines.push(format!(
                    "{}{} = __chelis_host_result_origin_child({}, {});",
                    self.indent,
                    result_origin_name(target),
                    result_origin_name(&arg_vars[0].0),
                    arg_vars[1].0
                ));
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
                // A selection: the prefix keeps each element's producer.
                self.lines.push(format!(
                    "{}{} = __chelis_host_result_origin_list_prefix(__chelis_origin_arena, {}, {});",
                    self.indent,
                    result_origin_name(target),
                    result_origin_name(&arg_vars[0].0),
                    arg_vars[1].0
                ));
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
                // capacity-census row without removing any ambiguity. The
                // consuming counterpart keeps the same stem.
                //
                // chelis#2334: a list the verifier moved at its last use is
                // skipped in place by advancing the runtime's private
                // offset; a borrowed one still clones the retained suffix.
                let entry = if container_operand_is_moved(site, "builtin:skip")? {
                    "chelis_list_drop_owned"
                } else {
                    "chelis_list_drop"
                };
                // Build the immutable metadata view before the consuming
                // entry point can release or advance the source payload.
                self.lines.push(format!(
                    "{}{} = __chelis_host_result_origin_list_suffix(__chelis_origin_arena, {}, {});",
                    self.indent,
                    result_origin_name(target),
                    result_origin_name(&arg_vars[0].0),
                    arg_vars[1].0
                ));
                self.lines.push(format!(
                    "{}{target} = {entry}({}, {});",
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
                if let Some(claims) = result_claims {
                    let width = &arg_vars[1].0;
                    let sequences = &arg_vars[0].0;
                    self.lines.push(format!(
                        "{}if ({width} < 0) {{ fprintf(stderr, \"Domain: pad_sequences_to requires non-negative width\\n\"); exit(1); }}",
                        self.indent));
                    self.lines.push(format!(
                        "{}__chelis_check_host_result_extent_claims({claims}, 2, (const int64_t[][3]){{ {{0, 0, chelis_list_len({sequences})}}, {{1, 1, {width}}} }}, 2, \"pad_sequences_to\", \"numeric trap: domain in pad_sequences_to at i64\");",
                        self.indent));
                }
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
            "abs",
            "min",
            "max",
            "min_elem",
            "max_elem",
            // C's `&&`, `||` and `!` combine scalars; over `chelis_tensor *`
            // they combine the pointers. Their tensor arms above take only
            // tensor operands.
            "and",
            "or",
            "not",
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
                CExpressionBuiltin::Add => {
                    binary(BinaryOperator::Add, numeric_arg(0), numeric_arg(1))
                }
                CExpressionBuiltin::Sub if is_integer_abi(ty) => integer_checked_binary_expr(
                    "chelis_int_checked_sub",
                    "sub",
                    arg(0),
                    arg(1),
                    ty,
                )?,
                CExpressionBuiltin::Sub => {
                    binary(BinaryOperator::Subtract, numeric_arg(0), numeric_arg(1))
                }
                CExpressionBuiltin::Mul if is_integer_abi(ty) => integer_checked_binary_expr(
                    "chelis_int_checked_mul",
                    "mul",
                    arg(0),
                    arg(1),
                    ty,
                )?,
                CExpressionBuiltin::Mul => {
                    binary(BinaryOperator::Multiply, numeric_arg(0), numeric_arg(1))
                }
                // chelis#178: integer `div` is a type error; this arm is dead
                // (the checker rejects it before host-emit) but kept as a
                // defensive guard. It checks the divisor like `trunc_div`, so
                // neither a zero divisor nor MIN / -1 reaches C undefined
                // behavior. Float `div` is IEEE-754 and never guarded.
                CExpressionBuiltin::Div if is_integer_abi(&arg_vars[0].1) => binary(
                    BinaryOperator::Divide,
                    arg(0),
                    checked_integer_divisor_expr("div", arg(0), arg(1), ty)?,
                ),
                CExpressionBuiltin::Div => {
                    binary(BinaryOperator::Divide, numeric_arg(0), numeric_arg(1))
                }
                // chelis#178: `trunc_div` is integer-only — the guarded C `/`
                // quotient (round toward zero).
                CExpressionBuiltin::TruncDiv => binary(
                    BinaryOperator::Divide,
                    arg(0),
                    checked_integer_divisor_expr("trunc_div", arg(0), arg(1), ty)?,
                ),
                // chelis#178: `floor_div` rounds toward -inf. Integer (host
                // scalar) operands use the runtime's checked floor division,
                // the same helper the tensor lane emits; float operands use
                // `floor(a / b)`.
                CExpressionBuiltin::FloorDiv if is_integer_abi(&arg_vars[0].1) => {
                    EmittedExpr::call(
                        "chelis_int_checked_floor_div",
                        [
                            arg(0),
                            arg(1),
                            EmittedExpr::integer(integer_abi_width(ty)?),
                            EmittedExpr::string_literal(integer_trap_message(
                                ty,
                                "floor_div",
                                false,
                            )?),
                            EmittedExpr::string_literal(integer_trap_message(
                                ty,
                                "floor_div",
                                true,
                            )?),
                        ],
                    )
                }
                CExpressionBuiltin::FloorDiv => EmittedExpr::call(
                    float_math_function(ty, "floor", "floorf"),
                    [binary(
                        BinaryOperator::Divide,
                        numeric_arg(0),
                        numeric_arg(1),
                    )],
                ),
                // [05-OP-64] float `mod` is C `fmod`, which is exact; the
                // result is finalized below like every float arm (chelis#626).
                CExpressionBuiltin::Mod if !is_integer_abi(&arg_vars[0].1) => EmittedExpr::call(
                    float_math_function(ty, "fmod", "fmodf"),
                    [numeric_arg(0), numeric_arg(1)],
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
                CExpressionBuiltin::Equal | CExpressionBuiltin::NotEqual => {
                    let equal = expression_builtin == CExpressionBuiltin::Equal;
                    match equality_entry(name, &arg_vars[0].1, &arg_vars[1].1)? {
                        EqualityEntry::Scalar => binary(
                            if equal {
                                BinaryOperator::Equal
                            } else {
                                BinaryOperator::NotEqual
                            },
                            numeric_arg(0),
                            numeric_arg(1),
                        ),
                        EqualityEntry::Unit => {
                            EmittedExpr::identifier(if equal { "true" } else { "false" })
                        }
                        EqualityEntry::Runtime(entry) => {
                            let call = EmittedExpr::call(entry, [arg(0), arg(1)]);
                            if equal {
                                call
                            } else {
                                unary(UnaryOperator::LogicalNot, call)
                            }
                        }
                    }
                }
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
                CExpressionBuiltin::Neg => unary(UnaryOperator::Negate, numeric_arg(0)),
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
                    // to_string renders the f32 scalar at its own width
                    // through the tagged formatter ([05-OBS-2]).
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
                // [05-OP-59]'s parsers, not [05-OP-31]'s scalar-carrier
                // `chelis_parse_scalar`, whose trimming, special spellings and
                // overflow refusal belong to a different contract.
                CExpressionBuiltin::ToInt => EmittedExpr::call("chelis_to_int", [arg(0)]),
                CExpressionBuiltin::ToFloat => EmittedExpr::call("chelis_to_float", [arg(0)]),
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
                // Scalar math in a Surf `def` body routed through the host
                // lane. The RISC DAG variants of these ops are handled in
                // `emit.rs`; both name the correctly rounded `chelis_cr_*`
                // kernels ([05-OP-46]) the unit carries, and `sqrt` is the
                // correctly rounded IEEE square root.
                CExpressionBuiltin::Sqrt => {
                    EmittedExpr::call(float_math_function(ty, "sqrt", "sqrtf"), [numeric_arg(0)])
                }
                CExpressionBuiltin::Exp => EmittedExpr::call(
                    float_math_function(ty, "chelis_cr_exp", "chelis_cr_expf"),
                    [numeric_arg(0)],
                ),
                CExpressionBuiltin::Log => EmittedExpr::call(
                    float_math_function(ty, "chelis_cr_log", "chelis_cr_logf"),
                    [numeric_arg(0)],
                ),
                CExpressionBuiltin::Sin => EmittedExpr::call(
                    float_math_function(ty, "chelis_cr_sin", "chelis_cr_sinf"),
                    [numeric_arg(0)],
                ),
                CExpressionBuiltin::Cos => EmittedExpr::call(
                    float_math_function(ty, "chelis_cr_cos", "chelis_cr_cosf"),
                    [numeric_arg(0)],
                ),
                CExpressionBuiltin::Tan => EmittedExpr::call(
                    float_math_function(ty, "chelis_cr_tan", "chelis_cr_tanf"),
                    [numeric_arg(0)],
                ),
                CExpressionBuiltin::Atan => EmittedExpr::call(
                    float_math_function(ty, "chelis_cr_atan", "chelis_cr_atanf"),
                    [numeric_arg(0)],
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
                CExpressionBuiltin::Sigmoid => EmittedExpr::call(
                    activation_math_function(
                        ty,
                        "chelis_host_sigmoid_f16",
                        "chelis_host_sigmoid_bf16",
                        "chelis_host_sigmoid_f32",
                        "chelis_host_sigmoid_f64",
                    ),
                    [numeric_arg(0)],
                ),
                CExpressionBuiltin::Tanh => EmittedExpr::call(
                    activation_math_function(
                        ty,
                        "chelis_host_tanh_f16",
                        "chelis_host_tanh_bf16",
                        "chelis_host_tanh_f32",
                        "chelis_host_tanh_f64",
                    ),
                    [numeric_arg(0)],
                ),
                CExpressionBuiltin::Silu => EmittedExpr::call(
                    activation_math_function(
                        ty,
                        "chelis_host_silu_f16",
                        "chelis_host_silu_bf16",
                        "chelis_host_silu_f32",
                        "chelis_host_silu_f64",
                    ),
                    [numeric_arg(0)],
                ),
                CExpressionBuiltin::Gelu => EmittedExpr::call(
                    activation_math_function(
                        ty,
                        "chelis_host_gelu_f16",
                        "chelis_host_gelu_bf16",
                        "chelis_host_gelu_f32",
                        "chelis_host_gelu_f64",
                    ),
                    [numeric_arg(0)],
                ),
                CExpressionBuiltin::Floor
                | CExpressionBuiltin::Ceil
                | CExpressionBuiltin::Round
                    if is_integer_abi(ty) =>
                {
                    arg(0)
                }
                CExpressionBuiltin::Floor => {
                    EmittedExpr::call(float_math_function(ty, "floor", "floorf"), [numeric_arg(0)])
                }
                CExpressionBuiltin::Ceil => {
                    EmittedExpr::call(float_math_function(ty, "ceil", "ceilf"), [numeric_arg(0)])
                }
                CExpressionBuiltin::Round =>
                // spec/05 §2.2: `round` is IEEE roundTiesToEven. The C
                // `round{,f}` family resolves half ties away from zero;
                // `rint{,f}` under the default rounding mode matches the
                // evaluator and the typed-DAG C emitter.
                {
                    EmittedExpr::call(float_math_function(ty, "rint", "rintf"), [numeric_arg(0)])
                }
                CExpressionBuiltin::Recip => binary(
                    BinaryOperator::Divide,
                    EmittedExpr::integer(1),
                    numeric_arg(0),
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
                    | HostType::Float64 => EmittedExpr::call(
                        float_math_function(ty, "fabs", "fabsf"),
                        [numeric_arg(0)],
                    ),
                    ref other => {
                        return Err(invalid_abi_shape(
                            format!("abs carries non-numeric argument type `{other:?}`"),
                            "abs builtin",
                        ));
                    }
                },
                CExpressionBuiltin::Min => EmittedExpr::call(
                    float_math_function(ty, "fmin", "fminf"),
                    [numeric_arg(0), numeric_arg(1)],
                ),
                CExpressionBuiltin::Max => EmittedExpr::call(
                    float_math_function(ty, "fmax", "fmaxf"),
                    [numeric_arg(0), numeric_arg(1)],
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
        // [04-NUM-1]/[04-NUM-2]: every arm above yields its value at the
        // computation width; the result is finalized here, once, by the
        // builtin's classification rather than by each arm.
        let expr = finalize_scalar_expr(
            build_expression()?,
            ty,
            CExpressionBuiltin::decode(name)?.nan_finalization(),
        );
        self.lines
            .push(format!("{}{target} = {};", self.indent, expr.as_c()));
        if matches!(ty, HostType::Unit) {
            self.lines.push(format!("{}{target} = 0;", self.indent));
        }
        Ok(())
    }

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

    /// Compare two agreeing tensors element by element into a bool tensor.
    /// The C operators give [05-OP-36]'s float rule directly: a NaN operand
    /// makes every comparison but `neq` false, and signed zeros are equal.
    /// The default arm is reached only by a dtype the checker refuses.
    fn assign_tensor_comparison_elementwise(
        &mut self,
        target: &str,
        lhs: &str,
        rhs: &str,
        comparison: ElementwiseComparison,
    ) {
        self.emit_elementwise_operand_guard(target, lhs, rhs);
        self.emit_elementwise_index_step(target, "lhs", lhs, lhs);
        self.emit_elementwise_index_step(target, "rhs", rhs, lhs);
        self.lines.push(format!(
            "{}{target} = chelis_host_alloc_like({lhs}, {});",
            self.indent,
            DtypeArm::Bool.dtype_macro()
        ));
        let (guard, view) = self.begin_tensor_write(target);
        self.lines.push(format!(
            "{}switch (chelis_host_tensor_dtype({lhs})) {{",
            self.indent
        ));
        // Every dtype [05-OP-36] admits has an arm: the C integer and float
        // types compare directly, and f16 and bf16 compare after the exact
        // widening to binary32, which keeps their order, NaNs and signed
        // zeros. Bool operands reach only equality.
        let mut arms = vec![
            (chelis_vocab::RuntimeDType::F32, Prim::F32, None),
            (chelis_vocab::RuntimeDType::F64, Prim::F64, None),
            (
                chelis_vocab::RuntimeDType::F16,
                Prim::F16,
                Some("chelis_f16_to_f32"),
            ),
            (
                chelis_vocab::RuntimeDType::Bf16,
                Prim::Bf16,
                Some("chelis_bf16_to_f32"),
            ),
            (chelis_vocab::RuntimeDType::I8, Prim::Int8, None),
            (chelis_vocab::RuntimeDType::I16, Prim::Int16, None),
            (chelis_vocab::RuntimeDType::I32, Prim::Int32, None),
            (chelis_vocab::RuntimeDType::I64, Prim::Int64, None),
        ];
        if comparison.admits_bool() {
            arms.push((chelis_vocab::RuntimeDType::Bool, Prim::Bool, None));
        }
        // Element types come from the existing spelling authorities: the
        // bool output from `DtypeArm`, each operand and the index from
        // `cast_prim_c_type`.
        let target_t = DtypeArm::Bool.elem_t();
        // The loop index is an i64 element count.
        let index_t = cast_prim_c_type(Prim::Int64);
        let ind = self.indent.clone();
        for (dtype, prim, widen) in arms {
            let elem_t = cast_prim_c_type(prim);
            let read = |side: &str| match widen {
                Some(widen) => format!("{widen}(__{side}_data[i * {target}_{side}_step])"),
                None => format!("__{side}_data[i * {target}_{side}_step]"),
            };
            self.lines
                .push(format!("{ind}    case {}: {{", dtype.c_macro()));
            self.lines.push(format!(
                "{ind}        {target_t} *__target_data = ({target_t}*){view}.data;"
            ));
            self.lines.push(format!(
                "{ind}        const {elem_t} *__lhs_data = (const {elem_t}*)chelis_host_tensor_data({lhs});"
            ));
            self.lines.push(format!(
                "{ind}        const {elem_t} *__rhs_data = (const {elem_t}*)chelis_host_tensor_data({rhs});"
            ));
            self.lines.push(format!(
                "{ind}        for ({index_t} i = 0; i < {view}.count; i++) {{"
            ));
            self.lines.push(format!(
                "{ind}            __target_data[i] = ({target_t})({} {} {});",
                read("lhs"),
                comparison.c_operator(),
                read("rhs")
            ));
            self.lines.push(format!("{ind}        }}"));
            self.lines.push(format!("{ind}        break;"));
            self.lines.push(format!("{ind}    }}"));
        }
        self.emit_default_runtime_fail_arm_for(
            &format!("chelis_host_tensor_dtype({lhs})"),
            &format!("elementwise comparison ({})", comparison.c_operator()),
        );
        self.lines.push(format!("{}}}", self.indent));
        self.end_tensor_write(&guard);
    }

    /// Apply an integer `mod`, bitwise or shift operation to two agreeing
    /// signed-integer tensors element by element, with the scalar lane's
    /// rule: `mod` traps a zero divisor and gives 0 for a -1 divisor, and a
    /// shift goes through the declared-width runtime helper, which traps a
    /// negative count. The default arm is reached only by a dtype the checker
    /// refuses.
    fn assign_tensor_integer_elementwise(
        &mut self,
        target: &str,
        lhs: &str,
        rhs: &str,
        builtin: &str,
    ) {
        self.emit_elementwise_operand_guard(target, lhs, rhs);
        self.emit_elementwise_index_step(target, "lhs", lhs, lhs);
        self.emit_elementwise_index_step(target, "rhs", rhs, lhs);
        self.lines.push(format!(
            "{}{target} = chelis_host_alloc_like({lhs}, chelis_host_tensor_dtype({lhs}));",
            self.indent
        ));
        let (guard, view) = self.begin_tensor_write(target);
        self.lines.push(format!(
            "{}switch (chelis_host_tensor_dtype({lhs})) {{",
            self.indent
        ));
        let index_t = cast_prim_c_type(Prim::Int64);
        let ind = self.indent.clone();
        for (dtype, prim, width) in [
            (chelis_vocab::RuntimeDType::I8, Prim::Int8, 8),
            (chelis_vocab::RuntimeDType::I16, Prim::Int16, 16),
            (chelis_vocab::RuntimeDType::I32, Prim::Int32, 32),
            (chelis_vocab::RuntimeDType::I64, Prim::Int64, 64),
        ] {
            let elem_t = cast_prim_c_type(prim);
            let l = format!("__lhs_data[i * {target}_lhs_step]");
            let r = format!("__rhs_data[i * {target}_rhs_step]");
            let value = match builtin {
                "mod" => {
                    let zero = NumericTrap::DivZero { op: "mod", prim }.to_string();
                    let overflow = NumericTrap::Overflow { op: "mod", prim }.to_string();
                    format!(
                        "({r} == -1) ? 0 : ({l} % ({elem_t})chelis_int_checked_divisor({l}, {r}, {width}, {zero:?}, {overflow:?}))"
                    )
                }
                "bitand" => format!("{l} & {r}"),
                "bitor" => format!("{l} | {r}"),
                "bitxor" => format!("{l} ^ {r}"),
                "shl" => format!("chelis_int_shl({l}, {r}, {width})"),
                "shr" => format!("chelis_int_shr({l}, {r}, {width})"),
                other => unreachable!("not an integer elementwise builtin: {other}"),
            };
            self.lines
                .push(format!("{ind}    case {}: {{", dtype.c_macro()));
            self.lines.push(format!(
                "{ind}        {elem_t} *__target_data = ({elem_t}*){view}.data;"
            ));
            self.lines.push(format!(
                "{ind}        const {elem_t} *__lhs_data = (const {elem_t}*)chelis_host_tensor_data({lhs});"
            ));
            self.lines.push(format!(
                "{ind}        const {elem_t} *__rhs_data = (const {elem_t}*)chelis_host_tensor_data({rhs});"
            ));
            self.lines.push(format!(
                "{ind}        for ({index_t} i = 0; i < {view}.count; i++) {{"
            ));
            self.lines.push(format!(
                "{ind}            __target_data[i] = ({elem_t})({value});"
            ));
            self.lines.push(format!("{ind}        }}"));
            self.lines.push(format!("{ind}        break;"));
            self.lines.push(format!("{ind}    }}"));
        }
        if builtin == "mod" {
            self.emit_float_mod_arms(target, lhs, rhs, &view, ind.as_str());
        }
        self.emit_default_runtime_fail_arm_for(
            &format!("chelis_host_tensor_dtype({lhs})"),
            &format!("elementwise {builtin}"),
        );
        self.lines.push(format!("{}}}", self.indent));
        self.end_tensor_write(&guard);
    }

    /// [05-OP-64] float `mod` arms for [`Self::assign_tensor_integer_elementwise`]:
    /// C `fmod` at f64 and `fmodf` at f32, each finalized through
    /// `fp_env::finalize_float`, and `fmodf` over the exact f32 widening of
    /// f16 and bf16 with one narrowing store, which is exact because `fmod`
    /// is (chelis#626).
    fn emit_float_mod_arms(&mut self, target: &str, lhs: &str, rhs: &str, view: &str, ind: &str) {
        let index_t = cast_prim_c_type(Prim::Int64);
        for (dtype, prim) in [
            (chelis_vocab::RuntimeDType::F32, Prim::F32),
            (chelis_vocab::RuntimeDType::F64, Prim::F64),
            (chelis_vocab::RuntimeDType::F16, Prim::F16),
            (chelis_vocab::RuntimeDType::Bf16, Prim::Bf16),
        ] {
            let elem_t = cast_prim_c_type(prim);
            let l = format!("__lhs_data[i * {target}_lhs_step]");
            let r = format!("__rhs_data[i * {target}_rhs_step]");
            let value = match prim {
                Prim::F32 => crate::fp_env::finalize_float(
                    &format!("fmodf({l}, {r})"),
                    false,
                    crate::fp_env::NanFinalization::Canonical,
                ),
                Prim::F64 => crate::fp_env::finalize_float(
                    &format!("fmod({l}, {r})"),
                    true,
                    crate::fp_env::NanFinalization::Canonical,
                ),
                Prim::F16 => format!(
                    "chelis_f32_to_f16(fmodf(chelis_f16_to_f32({l}), chelis_f16_to_f32({r})))"
                ),
                _ => format!(
                    "chelis_f32_to_bf16(fmodf(chelis_bf16_to_f32({l}), chelis_bf16_to_f32({r})))"
                ),
            };
            self.lines
                .push(format!("{ind}    case {}: {{", dtype.c_macro()));
            self.lines.push(format!(
                "{ind}        {elem_t} *__target_data = ({elem_t}*){view}.data;"
            ));
            self.lines.push(format!(
                "{ind}        const {elem_t} *__lhs_data = (const {elem_t}*)chelis_host_tensor_data({lhs});"
            ));
            self.lines.push(format!(
                "{ind}        const {elem_t} *__rhs_data = (const {elem_t}*)chelis_host_tensor_data({rhs});"
            ));
            self.lines.push(format!(
                "{ind}        for ({index_t} i = 0; i < {view}.count; i++) {{"
            ));
            self.lines
                .push(format!("{ind}            __target_data[i] = {value};"));
            self.lines.push(format!("{ind}        }}"));
            self.lines.push(format!("{ind}        break;"));
            self.lines.push(format!("{ind}    }}"));
        }
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
        self.emit_default_runtime_fail_arm_for(
            &format!("{view}.dtype"),
            &format!("unary func elementwise ({func})"),
        );
        self.lines.push(format!("{}}}", self.indent));
        self.end_tensor_write(&guard);
    }

    /// Finalize one f32 or f64 element of the host tensor elementwise loop
    /// being emitted through `fp_env::finalize_float`.
    fn finalize_tensor_elem(&self, arm: DtypeArm, expr: String) -> String {
        match (arm, self.tensor_nan_finalization) {
            (DtypeArm::F32, Some(finalization)) => {
                crate::fp_env::finalize_float(&expr, false, finalization)
            }
            (DtypeArm::F64, Some(finalization)) => {
                crate::fp_env::finalize_float(&expr, true, finalization)
            }
            _ => expr,
        }
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
        let element = match (arm, checked_integer_op(op)) {
            // [04-NUM-3]: integer arithmetic traps on overflow at its width,
            // as the tensor lane's kernels do, rather than wrapping.
            (DtypeArm::I32 | DtypeArm::I64, Some((helper, name))) => {
                let overflow = NumericTrap::Overflow {
                    op: name,
                    prim: arm.prim(),
                }
                .to_string();
                // The checked helpers compute at the i64 width.
                let wide = cast_prim_c_type(Prim::Int64);
                format!(
                    "({elem_t}){helper}(({wide})__lhs_data[idx_lhs], ({wide})__rhs_data[idx_rhs], {}, {overflow:?})",
                    arm.integer_bits()
                )
            }
            _ => self
                .finalize_tensor_elem(arm, format!("__lhs_data[idx_lhs] {op} __rhs_data[idx_rhs]")),
        };
        self.lines
            .push(format!("{ind}            __target_data[i] = {element};"));
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
        let element = match arm {
            DtypeArm::I32 | DtypeArm::I64 if op == "-" => {
                let overflow = NumericTrap::Overflow {
                    op: "neg",
                    prim: arm.prim(),
                }
                .to_string();
                let wide = cast_prim_c_type(Prim::Int64);
                format!(
                    "({elem_t})chelis_int_checked_neg(({wide})__input_data[idx], {}, {overflow:?})",
                    arm.integer_bits()
                )
            }
            _ => self.finalize_tensor_elem(arm, format!("{op}__input_data[idx]")),
        };
        self.lines
            .push(format!("{ind}            __target_data[i] = {element};"));
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
            "{ind}            __target_data[i] = {};",
            self.finalize_tensor_elem(arm, format!("{func}(__input_data[idx])"))
        ));
        self.lines.push(format!("{ind}        }}"));
        self.lines.push(format!("{ind}        break;"));
        self.lines.push(format!("{ind}    }}"));
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
        let helper_name = private_helper_name(&entry::variant_name(&base, entry_variant));
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
                // A key has no scalar carrier: it enters the kernel as the
                // rank-0 key tensor its key `Load` reads.
                let tensor = if inferred_ty == HostType::Key {
                    format!("chelis_key_tensor({value_name})")
                } else {
                    format!(
                        "chelis_scalar_tensor({})",
                        scalar_carrier_expr(&value_name, &inferred_ty)?
                    )
                };
                self.lines
                    .push(format!("{}{tensor_name} = {tensor};", self.indent));
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
        // The verified emitter's output order includes stores as well as roots.
        // A result slot is always a tensor; its destination remains a logical
        // scalar, tensor, or tuple. Validate that boundary before emitting it.
        let output_types = self
            .tensor_helper_output_types
            .get(helper)
            .ok_or_else(|| {
                invalid_abi_shape(
                    "tensor helper has no verified outputs".into(),
                    "tensor helper result",
                )
            })?
            .clone();
        let root_count = output_types.len();
        validate_tensor_result(ty, &mut output_types.iter())?;
        if root_count == 0 || tensor_result_leaf_count(ty) != root_count {
            return Err(invalid_abi_shape(
                format!("tensor helper has {root_count} outputs for result type {ty:?}"),
                "tensor helper result",
            ));
        }
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
        if !self.external_helpers.contains(&base) {
            helper_args.push(result_claims.unwrap_or("NULL").to_string());
        }
        self.lines.push(format!(
            "{}{}({});",
            self.indent,
            helper_name,
            helper_args.join(", ")
        ));
        self.materialize_tensor_result(target, ty, &outputs_name, &mut 0)?;
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

    /// Consume each owned helper output exactly once. Tensor leaves transfer
    /// their owner; scalar leaves extract before releasing it; tuples retain
    /// their boxed fields before the temporary field owners are released.
    fn materialize_tensor_result(
        &mut self,
        target: &str,
        ty: &HostType,
        outputs: &str,
        next: &mut usize,
    ) -> Result<(), Unsupported> {
        if let HostType::Tuple(parts) = ty {
            let values = self.next_temp("tuple_values");
            if parts.is_empty() {
                self.lines.push(format!(
                    "{}{target} = chelis_tuple_from_values(NULL, 0);",
                    self.indent
                ));
                return Ok(());
            }
            self.lines.push(format!(
                "{}chelis_value {values}[{}];",
                self.indent,
                parts.len()
            ));
            for (index, part) in parts.iter().enumerate() {
                let value = self.next_temp("helper_result");
                self.lines
                    .push(format!("{}{} {value};", self.indent, c_type(part)?));
                self.materialize_tensor_result(&value, part, outputs, next)?;
                self.lines.push(format!(
                    "{}{values}[{index}] = {};",
                    self.indent,
                    self.box_value_expr(&value, part)?
                ));
            }
            self.lines.push(format!(
                "{}{target} = chelis_tuple_from_values({values}, {});",
                self.indent,
                parts.len()
            ));
            for index in 0..parts.len() {
                self.lines.push(format!(
                    "{}chelis_value_release({values}[{index}]);",
                    self.indent
                ));
            }
            return Ok(());
        }
        let slot = format!("{outputs}[{}]", *next);
        *next += 1;
        if matches!(ty, HostType::Tensor(_)) {
            self.lines
                .push(format!("{}{target} = {slot};", self.indent));
        } else {
            let value = if *ty == HostType::Key {
                format!("chelis_key_of_tensor({slot})")
            } else {
                scalar_carrier_value_expr(&format!("chelis_tensor_to_scalar({slot})"), ty)?
            };
            self.lines
                .push(format!("{}{target} = {value};", self.indent));
            self.lines
                .push(format!("{}chelis_tensor_release({slot});", self.indent));
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
        let tree = matches!(kind, SparseSummaryKind::ScatterAdd).then(|| {
            let prefix = self.next_temp("sparse_add");
            let lines = CEmitter::scatter_add_tree_open_lines(
                &prefix,
                count,
                &format!("chelis_tensor_numel({target})"),
                summary.output.precision,
            );
            self.push_relative_lines(lines);
            prefix
        });
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
            SparseSummaryKind::ScatterAdd => {
                let tree = tree.as_deref().expect("scatter-add opened its tree");
                let record = CEmitter::scatter_add_tree_record_line(tree, &linear, &offset);
                self.lines.push(format!("{}    {record}", self.indent));
            }
            SparseSummaryKind::ScatterReplace => {
                let updates = &args[summary.input_indices[2]];
                self.lines.push(format!("{}    (({element}*){view}.data)[{offset}] = ((const {element}*)chelis_host_tensor_data({updates}))[{linear}];", self.indent));
            }
        }
        self.lines.push(format!("{}}}", self.indent));
        if let Some(tree) = tree {
            let updates = &args[summary.input_indices[2]];
            let lines = CEmitter::scatter_add_tree_close_lines(
                &tree,
                count,
                &format!("{view}.data"),
                &format!("chelis_tensor_numel({target})"),
                &format!("chelis_host_tensor_data({updates})"),
                summary.output.precision,
                crate::fp_env::risc_nan_finalization(&chelis_ir::dag::RiscOp::ScatterAdd {
                    axis: summary.axis,
                    batch_rank: 0,
                }),
            );
            self.push_relative_lines(lines);
        }
        self.end_tensor_write(&guard);
        self.lines.push(format!(
            "{}chelis_sparse_plan_release({plan});",
            self.indent
        ));
    }

    /// Push lines whose indentation is relative to the current depth.
    fn push_relative_lines(&mut self, lines: Vec<(usize, String)>) {
        for (depth, line) in lines {
            self.lines
                .push(format!("{}{}{line}", self.indent, "    ".repeat(depth)));
        }
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
            let forwards_receipt = self
                .entry_group
                .is_some_and(|group| self.entry_groups.get(function) == Some(&group))
                && !self.entry_proof_owners.is_empty()
                && call_forwards_entry_owners(site, &self.entry_proof_owners);
            append_private_host_context_args(
                &mut arg_vars,
                if forwards_receipt {
                    "&__chelis_entry_receipt_current"
                } else {
                    "NULL"
                },
            );
            arg_vars.push(result_claims.unwrap_or("NULL").to_string());
            arg_vars.push(format!("&{}", result_origin_name(target)));
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
            self.assign_interface_result_origin(target, ty);
            self.emit_result_claim_guard(target, ty, result_claims);
        }
        Ok(())
    }

    fn assign_adt_construct(
        &mut self,
        target: &str,
        ctor: &str,
        fields: &[HostExpr],
        ty: &HostType,
    ) -> Result<(), Unsupported> {
        // A nullary variant (e.g. `Nothing`, `True`) has no payload fields.
        // ISO C forbids a zero-length array (`chelis_value adt_fields[0];`),
        // so pass a NULL fields pointer with count 0 instead; the runtime
        // helper's `len <= 0` guard never dereferences it (issue #310).
        let mut field_vars = Vec::with_capacity(fields.len());
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
                field_vars.push(field_var);
            }
            (values_name.clone(), Some(values_name))
        };
        let ctor_value = self.next_temp("adt_ctor");
        self.lines.push(format!(
            "{}chelis_string {ctor_value} = {};",
            self.indent,
            runtime_string_literal(&stored_constructor_name(ty, ctor))
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
        self.assign_aggregate_result_origin(target, &field_vars);
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
        self.lines.push(format!(
            "{}{} = __chelis_host_result_origin_child({}, {});",
            self.indent,
            result_origin_name(target),
            result_origin_name(&base_var),
            field_index
        ));
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
        let scrutinee_ty = host_type(scrutinee);
        self.emit_expr_to_var(scrutinee, &scrutinee_var, &scrutinee_ty)?;
        // Pattern bindings may shadow the authored name that originally held
        // the scrutinee. Keep the verified owner attached to this
        // compiler-generated temporary so an arm-completion drop cannot be
        // redirected to a same-spelled scalar or another payload binding.
        // The temporary is declared in the enclosing C block, so the alias
        // ends with the match: a sibling branch that releases the same owner
        // must name the variable its own block can see.
        let aliased_owner = self
            .owner_vars
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
                self.indent,
                tag_var,
                stored_constructor_name(&scrutinee_ty, &arm.ctor)
            ));
            let nested_indent = format!("{}    ", self.indent);
            let previous = std::mem::replace(&mut self.indent, nested_indent);
            self.emit_edge_terminals(site.id, &arm_edges[index])?;
            // chelis#1222: one binder scope per arm. Each arm's pattern
            // bindings shadow any enclosing name they reuse, and their keys
            // carry no outgoing edge -- `chelis_adt_field` hands back an
            // independently retained handle, so the arm binding is not a
            // copy of anything this scope already owns.
            let mut shadowed_interface_globals = Vec::new();
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
                self.declare_result_origin(&binding.name, &binding.ty, None);
                self.lines.push(format!(
                    "{}{} = __chelis_host_result_origin_child({}, {});",
                    self.indent,
                    result_origin_name(&binding.name),
                    result_origin_name(&scrutinee_var),
                    binding.field_index
                ));
                self.bind_match_payload(
                    site,
                    arm_edges[index].target(),
                    "adt_payload:",
                    &binding.name,
                )?;
                if self.interface_reload_names.remove(&binding.name) {
                    shadowed_interface_globals.push(binding.name.clone());
                }
            }
            self.emit_region_entry_terminals(site, arm_edges[index].target(), arm_blocks[index])?;
            self.claim_on_spine = on_result_spine;
            self.assign_expr(target, &arm.expr, expr_ty)?;
            self.emit_expression_block_actions(site, arm_blocks[index], target)?;
            self.interface_reload_names
                .extend(shadowed_interface_globals);
            self.indent = previous;
            self.lines.push(format!("{}}}", self.indent));
        }
        self.lines.push(format!("{}else {{", self.indent));
        let nested_indent = format!("{}    ", self.indent);
        let previous = std::mem::replace(&mut self.indent, nested_indent);
        if let Some(default_expr) = default_expr {
            self.emit_edge_terminals(site.id, &arm_edges[arms.len()])?;
            self.emit_region_entry_terminals(
                site,
                arm_edges[arms.len()].target(),
                arm_blocks[arms.len()],
            )?;
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
        match aliased_owner {
            Some(name) => self.owner_vars.insert(scrutinee_owner, name),
            None => self.owner_vars.remove(&scrutinee_owner),
        };
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
        let mut item_vars = Vec::with_capacity(items.len());
        for (index, item) in items.iter().enumerate() {
            let item_var = self.next_temp(&format!("list_item{index}"));
            let actual_ty = host_type(item);
            require_list_element_abi_type(item_ty, &actual_ty)?;
            self.emit_expr_to_var(item, &item_var, &actual_ty)?;
            self.lines.push(format!(
                "{}{}[{index}] = {};",
                self.indent,
                values_name,
                self.box_aggregate_value_expr(&item_var, &actual_ty, item)?
            ));
            item_vars.push(item_var);
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
        self.assign_aggregate_result_origin(target, &item_vars);
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
        let mut item_vars = Vec::with_capacity(items.len());
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
                item_vars.push(item_var);
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
        self.assign_aggregate_result_origin(target, &item_vars);
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
        self.declare_local(&result_var, &callback.ret_ty)?;
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
        let pushed = self.box_value_expr(&result_var, &callback.ret_ty)?;
        self.emit_loop_step_block_actions(
            site,
            body_block,
            target,
            LoopCallbackResult::StepArgument(&result_var, 1),
            "list_push",
            |this| {
                this.lines.push(format!(
                    "{}chelis_list_push_moved({target}, {pushed});",
                    this.indent
                ));
                Ok(())
            },
        )?;
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
        self.declare_local(&keep_var, &callback.ret_ty)?;
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
        // `filter_step` moves the item: a kept item moves into the result
        // and a rejected one is released here.
        self.emit_loop_step_block_actions(
            site,
            body_block,
            target,
            LoopCallbackResult::StepArgument(&keep_var, 2),
            "filter_step",
            |this| {
                let indent = &this.indent;
                this.lines.extend([
                    format!("{indent}if ({keep_var}) {{"),
                    format!("{indent}    chelis_list_push_moved({target}, {item_value});"),
                    format!("{indent}}} else {{"),
                    format!("{indent}    chelis_value_release({item_value});"),
                    format!("{indent}}}"),
                ]);
                Ok(())
            },
        )?;
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
        self.lines.push(format!(
            "{}const __chelis_host_result_origin *{} = {};",
            self.indent,
            result_origin_name(&acc_arg),
            result_origin_name(target)
        ));
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
        if body_acc_dropped && let Some(released) = params[0].ty.c_released_value() {
            // An inline callback still has a physical parameter even when the
            // verified program proves that parameter dead on entry.  Do not
            // propagate a released handle into that non-semantic C alias.
            self.lines
                .push(format!("{}{} = {released};", self.indent, acc_arg));
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

    /// Emit `scan` with the same accumulator discipline as
    /// [`Self::assign_fold`]: the preheader's actions run before the loop,
    /// and the state the body receives is a per-iteration copy of the C
    /// variable the callback writes the next state into.
    ///
    /// The preheader holds the copy that pays for a seed the caller still
    /// owns, so emitting it after the loop let the first consuming operation
    /// on `acc` mutate the caller's value in place (chelis#2579). The body's
    /// state owner names `acc_arg` rather than `acc_var`, so a release or
    /// retain the schedule places after the callback names the old state,
    /// not the result that has already overwritten it (chelis#2580).
    #[allow(clippy::too_many_arguments)]
    fn assign_scan(
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
        let HostType::List(inner_ty) = ty else {
            self.lines
                .push(format!("{}{target} = chelis_list_empty();", self.indent));
            return Ok(());
        };
        let acc_ty = inner_ty.as_ref().clone();
        let (body_edge, exit_edge) = Self::loop_edges(site)?;
        let state_and_output = |edge: &VerifiedEdgeView<'a>, name: &str| match edge.params() {
            [state, output] => Ok((*state, *output)),
            params => Err(invalid_abi_shape(
                format!(
                    "verified scan {name} edge carries {} parameters, expected state and output",
                    params.len()
                ),
                "verified C host ownership emission",
            )),
        };
        let (body_state, body_output) = state_and_output(&body_edge, "body")?;
        let (exit_state, exit_output) = state_and_output(&exit_edge, "exit")?;
        let acc_var = self.next_temp("scan_acc");
        self.emit_expr_to_var(init, &acc_var, &acc_ty)?;
        self.emit_expression_block_actions(site, preheader_block, target)?;
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
        let params = callback_params(callback);
        let acc_arg = self.next_temp("scan_acc_arg");
        self.lines.push(format!(
            "{}{} {} = {};",
            self.indent,
            c_type(&params[0].ty)?,
            acc_arg,
            acc_var
        ));
        self.lines.push(format!(
            "{}const __chelis_host_result_origin *{} = {};",
            self.indent,
            result_origin_name(&acc_arg),
            result_origin_name(&acc_var)
        ));
        self.owner_vars.insert(body_state.id(), acc_arg.clone());
        self.owner_vars.insert(body_output.id(), target.to_string());
        self.owner_vars.insert(exit_state.id(), acc_var.clone());
        self.owner_vars.insert(exit_output.id(), target.to_string());
        let body_state_dropped = body_edge.terminals().iter().any(|terminal| {
            matches!(
                terminal,
                VerifiedTerminalView::Drop { owner, .. } if owner.id() == body_state.id()
            )
        });
        self.emit_edge_terminals(site.id, &body_edge)?;
        if body_state_dropped && let Some(released) = params[0].ty.c_released_value() {
            // As in `assign_fold`: an inline callback still binds a C alias
            // for a state the schedule proves dead on entry.
            self.lines
                .push(format!("{}{} = {released};", self.indent, acc_arg));
        }
        let item_value = self.next_temp("scan_item_value");
        self.lines.push(format!(
            "{}chelis_value {} = chelis_list_index({}, __i);",
            self.indent, item_value, list_var
        ));
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
        let pushed = self.box_value_expr(&acc_var, &acc_ty)?;
        self.emit_loop_step_block_actions(
            site,
            body_block,
            target,
            LoopCallbackResult::BackEdgeArgument(&acc_var, 0),
            "list_push",
            |this| {
                this.lines.push(format!(
                    "{}chelis_list_push_moved({target}, {pushed});",
                    this.indent
                ));
                Ok(())
            },
        )?;
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
        self.declare_local(&keep_var, &callback.ret_ty)?;
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
        self.emit_loop_step_block_actions(
            site,
            body_block,
            target,
            LoopCallbackResult::StepArgument(&keep_var, 2),
            "partition_step",
            |this| {
                let indent = &this.indent;
                this.lines.extend([
                    format!("{indent}if ({keep_var}) {{"),
                    format!("{indent}    chelis_list_push_moved({pass_var}, {item_value});"),
                    format!("{indent}}} else {{"),
                    format!("{indent}    chelis_list_push_moved({fail_var}, {item_value});"),
                    format!("{indent}}}"),
                ]);
                Ok(())
            },
        )?;
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
        // `chelis_tuple_from_values` retains its items, as for a tuple
        // literal, so the two moved lists are released here.
        for index in 0..2 {
            self.lines.push(format!(
                "{}chelis_value_release({tuple_values}[{index}]);",
                self.indent
            ));
        }
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
        self.declare_local(&result_var, &callback.ret_ty)?;
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
        self.emit_loop_step_block_actions(
            site,
            body_block,
            target,
            LoopCallbackResult::StepArgument(&result_var, 1),
            "list_extend",
            |this| {
                this.lines.push(format!(
                    "{}chelis_list_extend_moved({target}, {result_var});",
                    this.indent
                ));
                Ok(())
            },
        )?;
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
                    append_private_host_context_args(&mut arg_vars, "NULL");
                    arg_vars.push("NULL".to_string());
                    arg_vars.push(format!("&{}", result_origin_name(target)));
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
                if !self.emitted_names.contains_key(function) {
                    self.assign_interface_result_origin(target, &callback.ret_ty);
                }
            }
            HostCallbackKind::Inline { params, body } => {
                let mut shadowed_interface_globals = Vec::new();
                for (param, arg_var) in params.iter().zip(arg_vars.iter()) {
                    self.lines.push(format!(
                        "{}{} {} = {};",
                        self.indent,
                        c_type(&param.ty)?,
                        param.name,
                        arg_var
                    ));
                    // The parameter aliases the argument, so it carries the
                    // argument's origin. Rescanning would walk the value on
                    // every iteration, and would dereference a dead
                    // accumulator the fold has already cleared.
                    self.lines.push(format!(
                        "{}const __chelis_host_result_origin *{} = {};",
                        self.indent,
                        result_origin_name(&param.name),
                        result_origin_name(arg_var)
                    ));
                    if self.interface_reload_names.remove(&param.name) {
                        shadowed_interface_globals.push(param.name.clone());
                    }
                }
                self.assign_expr(target, body, &callback.ret_ty)?;
                self.interface_reload_names
                    .extend(shadowed_interface_globals);
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
            // A key has no scalar carrier ([05-OP-31] governs numeric and
            // bool scalars); a boxed key is its rank-0 key tensor.
            HostType::Key => format!("chelis_value_take_tensor(chelis_key_tensor({value}))"),
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
            // The callable's operation identity is retained by its checked
            // host type. An aggregate stores only a zero-payload witness.
            HostType::KeyBuiltinCallable(_) => {
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
        if matches!(ty, HostType::KeyBuiltinCallable(_)) {
            self.lines.push(format!(
                "{}chelis_tuple_release(chelis_tuple_take_value({value_expr}));",
                self.indent
            ));
            self.lines.push(format!(
                "{}{target} = (chelis_key_callable){{ NULL }};",
                self.indent
            ));
            return Ok(());
        }
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
            HostType::Key => format!("chelis_key_take_value({value_expr})"),
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
            HostType::KeyBuiltinCallable(_) => unreachable!("handled before scalar unboxing"),
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
            HostType::Option(_) => self.lines.push(format!(
                "{}chelis_host_print_option({}); printf(\"\\n\");",
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
        // C-compatible `\xNN`/`\\`/`\"` escapes).
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
            // [05-OBS-2]: a key root renders as the runtime's key text.
            HostType::Key => self.lines.push(format!(
                "{}{{ chelis_string text = chelis_string_from_key({value}); \
                 chelis_print_string(text); chelis_string_release(text); }}",
                self.indent
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
            HostType::Option(_) => self.lines.push(format!(
                "{}chelis_host_print_option({});",
                self.indent, value
            )),
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
            (
                "OPTION",
                "chelis_host_print_option",
                "chelis_option_borrow_value",
            ),
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
    // Ownership sites carry the already-emitted C variable. Mangling again
    // can turn an escaped source binding into a different, undeclared name.
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
    // The caller supplies the already-emitted C variable (as above).
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

/// How `eq` and `neq` compare one pair of host operands.
enum EqualityEntry {
    /// The C comparison operators over two scalar values.
    Scalar,
    /// Unit equals unit ([05-OP-36]).
    Unit,
    /// A runtime entry that compares two borrowed values of one type.
    Runtime(&'static str),
}

/// Select `eq` and `neq`'s comparison for an operand pair. A heap value is a
/// pointer in C, so every handle type names its runtime entry here and the C
/// operators apply only to scalars: an unlisted handle type is refused rather
/// than compared by address.
fn equality_entry(
    name: &str,
    lhs: &HostType,
    rhs: &HostType,
) -> Result<EqualityEntry, Unsupported> {
    let scalar = |ty: &HostType| {
        matches!(
            ty,
            HostType::Int8
                | HostType::Int16
                | HostType::Int32
                | HostType::Int64
                | HostType::Float16
                | HostType::BFloat16
                | HostType::Float32
                | HostType::Float64
                | HostType::Bool
        )
    };
    Ok(match (lhs, rhs) {
        (lhs, rhs) if scalar(lhs) && scalar(rhs) => EqualityEntry::Scalar,
        (HostType::Unit, HostType::Unit) => EqualityEntry::Unit,
        (HostType::String, HostType::String) => EqualityEntry::Runtime("chelis_string_eq"),
        // [05-OP-36]'s recursive equality.
        (HostType::List(_), HostType::List(_)) => EqualityEntry::Runtime("chelis_list_eq"),
        (HostType::Tuple(_), HostType::Tuple(_)) => EqualityEntry::Runtime("chelis_tuple_eq"),
        (HostType::Dict(..), HostType::Dict(..)) => EqualityEntry::Runtime("chelis_dict_eq"),
        (HostType::Option(_), HostType::Option(_)) => EqualityEntry::Runtime("chelis_option_eq"),
        (HostType::Adt(..), HostType::Adt(..)) => EqualityEntry::Runtime("chelis_adt_eq"),
        (lhs, rhs) => {
            return Err(Unsupported::new(
                UnsupportedKind::Builtin(name.to_string()),
                format!("`{lhs:?}` and `{rhs:?}` operands in `chelis build` host emission"),
                Stage::Codegen("c"),
                chelis_types::deliberate_rejection!(
                    "[04-TOT-2]",
                    "the host scalar lane has no comparison for this operand pair; a heap \
                     value is never compared by address"
                ),
            ));
        }
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

/// Namespace for escaped authored C names. Escaping occupants of the
/// namespace as well keeps it disjoint from private global aliases.
const C_USER_IDENT_PREFIX: &str = "chelis_user__";

fn result_origin_name(value: &str) -> String {
    format!("__chelis_result_origin_{}", c_ident(value))
}

/// Map a Chelis identifier or resolved global label to a legal C identifier.
///
/// Most names pass through byte-identical so the existing C/HIP corpus is
/// unchanged. A name is rewritten only when emitting it verbatim would
/// break compilation:
///   * it is a C/C++ reserved word (`register`, `static`, `main`, ...), or
///   * it collides with the compiler's emitted-helper or resolved-global
///     namespace, or occupies the escaped-user namespace itself.
///
/// The emitter's OWN temporaries (`__binding_N_value`, `__arg...`,
/// `__result`, `__call_...`, `__let_...`) are generated internally, are
/// already legal C, and are NOT user-controlled, so they must pass through
/// untouched — `c_decl` is called with both user names and these temps.
/// They do not occupy either escaped namespace, so the rules below leave
/// them alone. Existing #379 temporary-name limits are outside this fix.
///
/// The same mapping must be applied at every site that turns a user name
/// into a C identifier (declaration AND reference) so the two stay
/// consistent; `c_decl` and the `Var`/binding/hoist emit paths all route
/// through here.
fn c_ident(name: &str) -> std::borrow::Cow<'_, str> {
    // The `@` label cannot be authored in Surf. In emitted C it occupies a
    // namespace from which authored lookalikes are escaped below.
    if let Some(source) = chelis_ir::LoadStoreName::top_level_source_for_label(name)
        .expect("compiler-created global labels are canonical")
    {
        return std::borrow::Cow::Owned(format!(
            "__chelis_global_{}",
            source
                .as_bytes()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        ));
    }
    let collides_with_helper_scheme = name.contains("__tensor_") || name.contains("__global__");
    if name.starts_with("__chelis_global_")
        || name.starts_with(C_USER_IDENT_PREFIX)
        || C_RESERVED_WORDS.contains(&name)
        || collides_with_helper_scheme && !name.starts_with("__")
    {
        // Escaping this namespace's occupants makes the mapping injective
        // across ordinary names and compiler-created global aliases.
        return std::borrow::Cow::Owned(format!(
            "{C_USER_IDENT_PREFIX}{}",
            name.as_bytes()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        ));
    }
    // The other `__` names are the emitter's existing temporaries. Keep
    // their spellings so helper declarations and references still agree.
    std::borrow::Cow::Borrowed(name)
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
        | HostExprKind::RetainedInvocation { ty, .. }
        | HostExprKind::Map { ty, .. }
        | HostExprKind::Filter { ty, .. }
        | HostExprKind::Fold { ty, .. }
        | HostExprKind::Scan { ty, .. }
        | HostExprKind::Partition { ty, .. }
        | HostExprKind::FlatMap { ty, .. }
        | HostExprKind::TensorCall { ty, .. }
        | HostExprKind::ResultClaimScope { ty, .. }
        | HostExprKind::FormalIngress { ty, .. }
        | HostExprKind::ExtentSites { ty, .. } => ty.clone(),
        HostExprKind::Unit | HostExprKind::SignatureEntry { .. } => HostType::Unit,
    }
}

/// Whether the evaluator's `ResultProducer` model can attach provenance to
/// a value of this resolved ABI type. A value that cannot carries a null
/// origin rather than the uniform `load` leaf. ADT arguments are type
/// parameters rather than a field-layout description, so every ADT is
/// conservatively treated as able to.
fn host_type_may_carry_result_origin(ty: &HostType) -> bool {
    match ty {
        HostType::Tensor(_) | HostType::Adt(_, _) => true,
        HostType::List(inner) => host_type_may_carry_result_origin(inner),
        HostType::Tuple(items) => items.iter().any(host_type_may_carry_result_origin),
        HostType::Option(inner) => host_type_may_carry_result_origin(inner),
        // Dict is a distinct evaluator value and is outside this tree model.
        HostType::Int8
        | HostType::Int16
        | HostType::Int32
        | HostType::Int64
        | HostType::Float16
        | HostType::BFloat16
        | HostType::Float32
        | HostType::Float64
        | HostType::Bool
        | HostType::String
        | HostType::Key
        | HostType::KeyBuiltinCallable(_)
        | HostType::Callback(_, _)
        | HostType::Dict(_, _)
        | HostType::MappedFile
        | HostType::Unit => false,
    }
}

/// The checked runtime helper and trap operation name of an integer
/// binary operator that can overflow.
fn checked_integer_op(op: &str) -> Option<(&'static str, &'static str)> {
    match op {
        "+" => Some(("chelis_int_checked_add", "add")),
        "-" => Some(("chelis_int_checked_sub", "sub")),
        "*" => Some(("chelis_int_checked_mul", "mul")),
        _ => None,
    }
}

/// The arms the host elementwise loop for `name` emits, when `name` over
/// these operands is one of those loops.
fn host_elementwise_arms(
    name: &str,
    arg_vars: &[(String, HostType)],
) -> Option<&'static [DtypeArm]> {
    let tensor = |index: usize| matches!(arg_vars.get(index), Some((_, HostType::Tensor(_))));
    match name {
        "add" | "sub" | "mul" | "div" | "max_elem" | "min_elem" | "and" | "or"
            if tensor(0) && tensor(1) =>
        {
            Some(DtypeArm::all_operator_arms())
        }
        "neg" | "not" if tensor(0) => Some(DtypeArm::all_operator_arms()),
        "exp" | "log" | "sin" | "sqrt" | "relu" | "sigmoid" | "tanh" | "silu" | "gelu"
            if tensor(0) =>
        {
            Some(DtypeArm::f32_payload_func_arms())
        }
        _ => None,
    }
}

/// Refuse a host elementwise loop over a dtype the shared host-loop table
/// ([`chelis_ir::host::host_elementwise_loop_admits`]) does not admit. The
/// loop's `arms` are that table's admitted dtypes, which
/// `host_elementwise_arms_match_the_shared_table` checks.
fn admit_host_elementwise(
    name: &str,
    arg_vars: &[(String, HostType)],
    arms: &[DtypeArm],
) -> Result<(), Unsupported> {
    let Some((_, HostType::Tensor(operand))) = arg_vars.first() else {
        return Ok(());
    };
    let admitted =
        chelis_ir::host::host_elementwise_loop_admits(name, operand.precision) == Some(true);
    if admitted {
        if !arms.iter().any(|arm| arm.prim() == operand.precision) {
            return Err(invalid_abi_shape(
                format!(
                    "the host loop for `{name}` has no arm for admitted `{}`",
                    operand.precision.name()
                ),
                "C host elementwise emission",
            ));
        }
        return Ok(());
    }
    Err(Unsupported::new(
        UnsupportedKind::Builtin(name.to_string()),
        format!(
            "`{name}` over a `{}` tensor in `chelis build` host emission (the host \
             elementwise loop has no arm for this dtype)",
            operand.precision.name()
        ),
        Stage::Codegen("c"),
        chelis_types::deliberate_rejection!(
            "[04-TOT-2]",
            "a checked tensor operation must route through the typed DAG lane; the C \
             host lane's elementwise loops are no fallback for a dtype they do not cover"
        ),
    ))
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

fn require_list_element_abi_type(
    expected: &HostType,
    actual: &HostType,
) -> Result<(), Unsupported> {
    if let (HostType::Tensor(expected), HostType::Tensor(actual)) = (expected, actual) {
        // Symbolic axis names are checker identities, not different C tensor
        // representations. A literal can likewise inhabit a joined list
        // element type; the consumer checks concrete extent relationships.
        if expected.precision == actual.precision && expected.dims.len() == actual.dims.len() {
            return Ok(());
        }
    }
    require_same_abi_type(expected, actual, "list element")
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
        | HostType::Key
        | HostType::KeyBuiltinCallable(_)
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

/// Where a list loop's verified body hands on its callback's result, with
/// the C variable the loop wrote that result into.
#[derive(Clone, Copy, Debug)]
enum LoopCallbackResult<'v> {
    /// The step consumes it as its argument at this index.
    StepArgument(&'v str, usize),
    /// The back-edge carries it into the loop state at this index (`scan`).
    BackEdgeArgument(&'v str, usize),
}

impl LoopCallbackResult<'_> {
    fn variable(&self) -> &str {
        match self {
            Self::StepArgument(variable, _) | Self::BackEdgeArgument(variable, _) => variable,
        }
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

fn tensor_result_leaf_count(ty: &HostType) -> usize {
    match ty {
        HostType::Tuple(parts) => parts.iter().map(tensor_result_leaf_count).sum(),
        _ => 1,
    }
}

fn validate_tensor_result<'a>(
    ty: &HostType,
    outputs: &mut impl Iterator<Item = &'a TensorType>,
) -> Result<(), Unsupported> {
    if let HostType::Tuple(parts) = ty {
        for part in parts {
            validate_tensor_result(part, outputs)?;
        }
        return Ok(());
    }
    let output = outputs.next().ok_or_else(|| {
        invalid_abi_shape(
            format!("missing tensor helper output for {ty:?}"),
            "tensor helper result",
        )
    })?;
    let valid = match ty {
        HostType::Tensor(expected) => {
            expected.precision == output.precision && expected.dims.len() == output.dims.len()
        }
        HostType::Key => output.dims.is_empty() && output.precision == Prim::Key,
        _ => output.dims.is_empty() && checked_cast_abi_scalar_prim(ty)? == output.precision,
    };
    if !valid {
        return Err(invalid_abi_shape(
            format!("tensor helper output {output:?} cannot materialize as {ty:?}"),
            "tensor helper result",
        ));
    }
    Ok(())
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

/// The host scalar lane's one finalization point ([04-NUM-1], [04-NUM-2]).
/// Arithmetic f16 and bf16 values computed at f32 narrow through storage
/// helpers that canonicalize NaN; f32 and f64 values pass through
/// `fp_env::finalize_float`. A bit-preserving selection already holds the
/// selected operand's stored bits, and a non-float result has no NaN.
fn finalize_scalar_expr(
    value: EmittedExpr,
    ty: &HostType,
    finalization: Option<crate::fp_env::NanFinalization>,
) -> EmittedExpr {
    use crate::fp_env::NanFinalization;
    let Some(finalization) = finalization else {
        return value;
    };
    match (ty, finalization) {
        (HostType::Float16, NanFinalization::Canonical) => {
            EmittedExpr::call("chelis_f32_to_f16", [value])
        }
        (HostType::BFloat16, NanFinalization::Canonical) => {
            EmittedExpr::call("chelis_f32_to_bf16", [value])
        }
        (HostType::Float32, NanFinalization::Canonical) => {
            EmittedExpr::call(crate::fp_env::canonical_nan_helper(false), [value])
        }
        (HostType::Float64, NanFinalization::Canonical) => {
            EmittedExpr::call(crate::fp_env::canonical_nan_helper(true), [value])
        }
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

/// The element-wise tensor comparisons the host lane emits ([05-OP-36]).
#[derive(Debug, Clone, Copy)]
enum ElementwiseComparison {
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Equal,
    NotEqual,
}

impl ElementwiseComparison {
    fn from_builtin(name: &str) -> Self {
        match name {
            "lt" => Self::Less,
            "lte" => Self::LessEqual,
            "gt" => Self::Greater,
            "gte" => Self::GreaterEqual,
            "eq" => Self::Equal,
            "neq" => Self::NotEqual,
            other => unreachable!("`{other}` is not an element-wise comparison"),
        }
    }

    fn c_operator(self) -> &'static str {
        match self {
            Self::Less => "<",
            Self::LessEqual => "<=",
            Self::Greater => ">",
            Self::GreaterEqual => ">=",
            Self::Equal => "==",
            Self::NotEqual => "!=",
        }
    }

    /// Only equality admits bool operands; ordering them is a type error.
    fn admits_bool(self) -> bool {
        matches!(self, Self::Equal | Self::NotEqual)
    }
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

/// Fold a `uniform_like` bound to the exact compile-time value the emitter
/// bakes into the generated call (chelis#2120).
///
/// The checker already restricts these bounds to static literals
/// (`chelis-types` `infer::app_operand_dtype`, whose message is the
/// chelis#776 gate), and this mirrors the shapes `is_static_numeric_bound`
/// admits there: a bare float or integer literal, a float-target `cast`
/// around one, and `neg` of either. Anything else is an internal desync
/// between that gate and this emitter, and is rejected loudly rather than
/// defaulted — silently substituting `[0, 1)` for an unreadable bound is the
/// exact chelis#776 failure this must not reintroduce.
///
/// chelis#2316: the value is staged exactly as the IR lanes evaluate the
/// bound's operand graph — an integer leaf stays EXACT through
/// i64 and a float leaf stays at its source dtype until a cast finalizes it —
/// and every transition goes through the shared `chelis_types` cast
/// primitives. Both lanes therefore apply the same roundings in the same
/// order, which is the property chelis#2120 needs: a template that folds and
/// one that does not must sample identically.
///
/// Two earlier revisions got this wrong in opposite directions. The first
/// ignored a cast's target entirely, so a narrowing intermediate was dropped.
/// The second finalized eagerly at every float cast and returned `None` for a
/// non-float target — which broke a bound like `cast(3i32, f32)`, because host
/// lowering expresses a literal's declared dtype by SYNTHESIZING a cast around
/// the raw lexical value ([04-LIT-1], `chelis-ir/src/host.rs`). That
/// synthesized `cast[i32]` is not a user-written narrowing cast; it is the
/// literal's own type, and its value is exact.
#[derive(Clone, Copy)]
enum StagedBound {
    /// An untyped lexical leaf: an integer that is still exact, or a float
    /// literal that has not yet been finalized at any declared width.
    Raw(chelis_types::RawScalar),
    /// A value already finalized at a concrete dtype.
    Typed(chelis_types::ScalarValue),
}

impl StagedBound {
    /// Finalize at `target`, going through the same `chelis_types` primitives
    /// `chelis-ir`'s typed fold uses.
    fn finalize(self, target: Prim) -> Option<chelis_types::ScalarValue> {
        match self {
            StagedBound::Raw(raw) => chelis_types::cast_raw("uniform_like", raw, target).ok(),
            StagedBound::Typed(value) => {
                chelis_types::cast_scalar("uniform_like", value, target).ok()
            }
        }
    }

    fn negate(self) -> Option<Self> {
        match self {
            StagedBound::Raw(chelis_types::RawScalar::Int(v)) => Some(StagedBound::Raw(
                chelis_types::RawScalar::Int(v.checked_neg()?),
            )),
            StagedBound::Raw(chelis_types::RawScalar::Float(v)) => {
                Some(StagedBound::Raw(chelis_types::RawScalar::Float(-v)))
            }
            // A finalized value negates at its own width, not through f64.
            StagedBound::Typed(value) => {
                let prim = value.prim();
                chelis_types::cast_raw(
                    "uniform_like",
                    chelis_types::RawScalar::Float(-value.as_f64_lossy()),
                    prim,
                )
                .ok()
                .map(StagedBound::Typed)
            }
        }
    }
}

fn staged_bound(expr: Option<&HostExpr>) -> Option<StagedBound> {
    match &expr?.kind {
        HostExprKind::Float(value) => {
            Some(StagedBound::Raw(chelis_types::RawScalar::Float(*value)))
        }
        HostExprKind::Int(value) => Some(StagedBound::Raw(chelis_types::RawScalar::Int(*value))),
        HostExprKind::Builtin { name, args, ty } if name == "cast" => {
            let inner = staged_bound(args.first())?;
            let (prim, surface) = checked_cast_abi_axis(ty).ok()?;
            if surface != CheckedCastSurface::Scalar {
                return None;
            }
            // Finalize at THIS cast's target, whatever it is. An integer
            // target is a declared-dtype marker from host lowering, not a
            // user-written truncation: the checker rejects an integer-target
            // bound before emission (`PrecisionMismatch: expected f32`), so a
            // non-float target reaching here is always the literal's own type.
            Some(StagedBound::Typed(inner.finalize(prim)?))
        }
        HostExprKind::Builtin { name, args, .. } if name == "neg" => {
            staged_bound(args.first())?.negate()
        }
        _ => None,
    }
}

fn static_float_bound(expr: Option<&HostExpr>) -> Option<f64> {
    // f32 is the emitted bound width; see the matching note at the
    // `uniform_like` site in `chelis-ir`'s lowering.
    staged_bound(expr)
        .and_then(|staged| staged.finalize(Prim::F32))
        .map(|value| value.as_f64_lossy())
}

/// The C `float` expression of one `[05-OP-8]` bound: the exact f32 image of
/// a bound the emitter folds, else the f32 host scalar the arguments computed.
fn uniform_bound_f32_expr(
    expr: Option<&HostExpr>,
    var: Option<&(String, HostType)>,
    which: &str,
) -> Result<String, Unsupported> {
    if let Some(bound) = static_float_bound(expr) {
        let bits = (bound as f32).to_bits();
        return Ok(format!("chelis_f32_from_bits(UINT32_C(0x{bits:08x}))"));
    }
    match var {
        Some((name, HostType::Float32)) => {
            Ok(format!("(({})({name}))", cast_prim_c_type(Prim::F32)))
        }
        _ => Err(Unsupported::new(
            UnsupportedKind::Builtin("uniform_like".to_string()),
            "`chelis build` host emission",
            Stage::Codegen("c"),
            chelis_types::deliberate_rejection!(
                "[04-TOT-2]",
                "uniform_like's bounds are f32 scalars; the checker types them so, so a \
                 bound that is neither a foldable literal nor an f32 host scalar here is \
                 an internal desync"
            ),
        )
        .with_supported_alternative(match which {
            "low" => "give `uniform_like` an f32 low bound",
            _ => "give `uniform_like` an f32 high bound",
        })),
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

    fn prim(self) -> Prim {
        match self {
            DtypeArm::F32 => Prim::F32,
            DtypeArm::F64 => Prim::F64,
            DtypeArm::I32 => Prim::Int32,
            DtypeArm::I64 => Prim::Int64,
            DtypeArm::Bool => Prim::Bool,
        }
    }

    /// The width an integer arm's checked arithmetic traps at.
    fn integer_bits(self) -> u32 {
        match self {
            DtypeArm::I32 => 32,
            DtypeArm::I64 => 64,
            DtypeArm::F32 | DtypeArm::F64 | DtypeArm::Bool => {
                unreachable!("only an integer arm has checked arithmetic")
            }
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

    /// chelis#2734: host lowering keeps an operation on the host loop exactly
    /// when the shared table admits its dtype, so each loop's arms must be
    /// that table's admitted dtypes, no more and no fewer.
    #[test]
    fn host_elementwise_arms_match_the_shared_table() {
        let prims = [
            Prim::F16,
            Prim::Bf16,
            Prim::F32,
            Prim::F64,
            Prim::Int8,
            Prim::Int16,
            Prim::Int32,
            Prim::Int64,
            Prim::Bool,
        ];
        let tensor = |prim| {
            (
                "x".to_string(),
                HostType::Tensor(TensorType {
                    dims: vec![DimInfo::Lit(2)],
                    precision: prim,
                }),
            )
        };
        for name in [
            "add", "sub", "mul", "div", "max_elem", "min_elem", "and", "or", "neg", "not", "exp",
            "log", "sin", "sqrt", "relu", "sigmoid", "tanh", "silu", "gelu",
        ] {
            for prim in prims {
                let arms = host_elementwise_arms(name, &[tensor(prim), tensor(prim)])
                    .unwrap_or_else(|| panic!("`{name}` has a host loop"));
                assert_eq!(
                    chelis_ir::host::host_elementwise_loop_admits(name, prim),
                    Some(arms.iter().any(|arm| arm.prim() == prim)),
                    "{name} at {prim:?}"
                );
            }
        }
    }

    #[test]
    fn resolved_global_c_symbol_cannot_alias_authored_lookalikes() {
        let global = chelis_ir::LoadStoreName::top_level("x");
        let alias = c_ident(global.as_str());
        assert_eq!(alias, "__chelis_global_78");
        assert_ne!(alias, c_ident("__chelis_global_78"));
        assert_ne!(
            c_ident("__chelis_global_78"),
            c_ident("chelis_user__5f5fchelis_global_78")
        );
        assert_eq!(c_ident("ordinary"), "ordinary");
    }

    // chelis#2408: the standalone kernel prelude and the host translation unit
    // each carry the C port of the Random stream. They must be one port.
    #[test]
    fn host_random_stream_helpers_are_the_standalone_kernel_prelude() {
        let mut dag = chelis_ir::dag::Dag::new();
        let decl = dag.declare("test");
        let ty = TensorType {
            dims: vec![chelis_ir::dag::DimInfo::Lit(2)],
            precision: Prim::F32,
        };
        let template = dag.add_node(
            decl,
            chelis_ir::dag::RiscOp::Load {
                name: "template".into(),
            },
            vec![],
            ty.clone(),
            None,
        );
        let rank0 = |precision| TensorType {
            dims: vec![],
            precision,
        };
        let bound = |dag: &mut chelis_ir::dag::Dag, value| {
            dag.add_node(
                decl,
                chelis_ir::dag::RiscOp::synth_const(Prim::F32, value),
                vec![],
                rank0(Prim::F32),
                None,
            )
        };
        let low = bound(&mut dag, 0.0);
        let high = bound(&mut dag, 1.0);
        let seed = dag.add_node(
            decl,
            chelis_ir::dag::RiscOp::synth_const(Prim::Int64, 7.0),
            vec![],
            rank0(Prim::Int64),
            None,
        );
        let key = dag.add_node(
            decl,
            chelis_ir::dag::RiscOp::KeyFromSeed,
            vec![seed],
            rank0(Prim::Key),
            None,
        );
        let draw = dag.add_node(
            decl,
            chelis_ir::dag::RiscOp::UniformLike,
            vec![template, low, high, key],
            ty,
            None,
        );
        dag.add_root(draw);
        let verified = crate::testing::verified_dag(&dag, crate::CodegenOptions::default())
            .expect("uniform DAG verifies");
        let kernel = CEmitter::emit_dag(verified, "draw").expect("uniform DAG emits");
        let prelude = kernel
            .split("/* CHELIS_UNIFORM_HELPERS_BEGIN */\n")
            .nth(1)
            .and_then(|rest| rest.split("/* CHELIS_UNIFORM_HELPERS_END */").next())
            .expect("the kernel carries the Random prelude");
        let mut host = Vec::new();
        append_uniform_sample_helper(&mut host);
        let host = host.join("\n");
        assert!(
            prelude.contains("chelis_random_unit") && host.starts_with(prelude.trim_end()),
            "kernel prelude:\n{prelude}\nhost helpers:\n{host}"
        );
    }

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
                output_types: &[],
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
        append_tensor_math_helpers(&mut helpers, &|_| true);
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

    /// The host activation helpers compute with the carried kernels: `tanh`
    /// is [05-OP-46]'s correctly rounded primitive at every float width, and
    /// the composite helpers call `chelis_cr_*` kernels and no libm name.
    /// Compiled and run against `chelis-crmath`'s bits (chelis#2957).
    #[test]
    fn activation_helpers_call_the_correctly_rounded_kernels() {
        let mut helpers = Vec::new();
        append_tensor_math_helpers(&mut helpers, &|_| true);
        let emitted = helpers.join("\n");
        let identifiers: Vec<&str> = emitted
            .split(|c: char| !(c == '_' || c.is_ascii_alphanumeric()))
            .collect();
        for libm in ["expf", "tanhf", "exp", "tanh"] {
            assert!(
                !identifiers.contains(&libm),
                "helpers must not call `{libm}`:\n{emitted}"
            );
        }
        assert!(emitted.contains("chelis_cr_expf(") && emitted.contains("chelis_cr_tanh("));
        let witnesses: [f32; 4] = [-0.055_804_74, 1e-3, 1e-5, -3.0];
        let mut main = String::from(
            "#include \"chelis_runtime.h\"\n#include <stdio.h>\n#include <string.h>\n",
        );
        main.push_str(&emitted);
        main.push_str("\nint main(void) {\n");
        for w in witnesses {
            main.push_str(&format!(
                "    {{ float y = chelis_host_tanh_f32(chelis_f32_from_bits(UINT32_C(0x{:08x}))); uint32_t b; memcpy(&b, &y, 4); printf(\"%08x\\n\", (unsigned)b); }}\n",
                w.to_bits()
            ));
            main.push_str(&format!(
                "    {{ double y = chelis_host_tanh_f64(chelis_f64_from_bits(UINT64_C(0x{:016x}))); uint64_t b; memcpy(&b, &y, 8); printf(\"%016llx\\n\", (unsigned long long)b); }}\n",
                f64::from(w).to_bits()
            ));
        }
        main.push_str("    return 0;\n}\n");
        let main = crate::crmath_kernels::link_called_kernels(main);

        let dir = tempfile::tempdir().unwrap();
        let staged = chelis_runtime_bundle::stage(dir.path()).unwrap();
        std::fs::write(dir.path().join("main.c"), &main).unwrap();
        let toolchain = crate::toolchain::strict_reference_toolchain(
            crate::toolchain::c_compiler(),
            crate::toolchain::CodegenRequirements::default(),
        );
        let bin = dir.path().join("helpers");
        let compiled = std::process::Command::new(&toolchain.compiler)
            .args(&toolchain.compile_flags)
            .arg("-std=c11")
            .arg("-I")
            .arg(dir.path())
            .arg(dir.path().join("main.c"))
            .arg(&staged.archive)
            .args(&toolchain.link_flags)
            .arg("-o")
            .arg(&bin)
            .output()
            .unwrap();
        assert!(
            compiled.status.success(),
            "{}",
            String::from_utf8_lossy(&compiled.stderr)
        );
        let run = std::process::Command::new(&bin).output().unwrap();
        assert!(run.status.success());
        let stdout = String::from_utf8(run.stdout).unwrap();
        let got: Vec<&str> = stdout.lines().collect();
        let mut want = Vec::new();
        for w in witnesses {
            want.push(format!("{:08x}", chelis_crmath::tanh_f32(w).to_bits()));
            want.push(format!(
                "{:016x}",
                chelis_crmath::tanh_f64(f64::from(w)).to_bits()
            ));
        }
        assert_eq!(
            got, want,
            "tanh helpers must equal the correctly rounded kernel"
        );
    }

    /// spec/05 §3.3's gelu graph at operand width, through `chelis-crmath`'s
    /// exp: each primitive computes at f32 and `finalize` rounds it to the
    /// operand dtype before the next primitive reads it.
    fn gelu_reference_f32(
        x: f32,
        finalize: &dyn Fn(f32) -> f32,
        constant: &dyn Fn(f64) -> f32,
    ) -> f32 {
        let c = constant(0.7978845608028654);
        let k = constant(0.044715);
        let two = constant(2.0);
        let one = constant(1.0);
        let x_squared = finalize(x * x);
        let x_cubed = finalize(x_squared * x);
        let u = finalize(c * finalize(x + finalize(k * x_cubed)));
        let two_u = finalize(two * u);
        let exp_neg = finalize(chelis_crmath::exp_f32(finalize(-two_u)));
        let sigmoid = finalize(1.0 / finalize(one + exp_neg));
        finalize(x * sigmoid)
    }

    fn gelu_reference_f64(x: f64) -> f64 {
        let u = 0.7978845608028654 * (x + 0.044715 * ((x * x) * x));
        x * (1.0 / (1.0 + chelis_crmath::exp_f64(-(2.0 * u))))
    }

    /// chelis#2997: the host gelu helpers, derived from `tier2::lower_gelu`,
    /// follow spec/05 §3.3's
    /// `x*sigmoid(2u)` graph bit for bit, so gelu of a large finite input is
    /// that input (the old `0.5*x*(1+tanh(u))` helper overflowed to `inf`).
    /// Checked over every finite f16, at bf16/f32/f64 max finite, and at the
    /// negative inputs where the tanh spelling cancels.
    #[test]
    fn gelu_helpers_follow_the_pinned_sigmoid_graph() {
        use half::{bf16, f16};
        let mut helpers = Vec::new();
        append_tensor_math_helpers(&mut helpers, &|_| true);
        let mut main = String::from(
            "#include \"chelis_runtime.h\"\n#include <stdio.h>\n#include <string.h>\n",
        );
        main.push_str(&helpers.join("\n"));
        main.push_str(concat!(
            "\nint main(void) {\n",
            "    for (uint32_t b = 0; b < 65536u; b++) {\n",
            "        if ((b & 0x7c00u) == 0x7c00u) continue;\n",
            "        float y = chelis_host_gelu_f16(chelis_f16_to_f32((uint16_t)b));\n",
            "        printf(\"f16 %04x %04x\\n\", (unsigned)b, (unsigned)chelis_f32_to_f16(y));\n",
            "    }\n",
        ));
        let bf16_max = bf16::MAX.to_f32();
        main.push_str(&format!(
            "    {{ float y = chelis_host_gelu_bf16(chelis_f32_from_bits(UINT32_C(0x{:08x}))); uint32_t o; memcpy(&o, &y, 4); printf(\"bf16 %08x\\n\", (unsigned)o); }}\n",
            bf16_max.to_bits()
        ));
        let f32_inputs = [f32::MAX, -3.0, -4.0, -5.0, -6.0, -9.336, 2.5, -0.5];
        for x in f32_inputs {
            main.push_str(&format!(
                "    {{ float y = chelis_host_gelu_f32(chelis_f32_from_bits(UINT32_C(0x{:08x}))); uint32_t o; memcpy(&o, &y, 4); printf(\"f32 %08x\\n\", (unsigned)o); }}\n",
                x.to_bits()
            ));
        }
        let f64_inputs = [f64::MAX, -3.0, -4.0, -5.0, -6.0, -9.336, 2.5, -0.5];
        for x in f64_inputs {
            main.push_str(&format!(
                "    {{ double y = chelis_host_gelu_f64(chelis_f64_from_bits(UINT64_C(0x{:016x}))); uint64_t o; memcpy(&o, &y, 8); printf(\"f64 %016llx\\n\", (unsigned long long)o); }}\n",
                x.to_bits()
            ));
        }
        main.push_str("    return 0;\n}\n");
        let main = crate::crmath_kernels::link_called_kernels(main);

        let dir = tempfile::tempdir().unwrap();
        let staged = chelis_runtime_bundle::stage(dir.path()).unwrap();
        std::fs::write(dir.path().join("main.c"), &main).unwrap();
        let toolchain = crate::toolchain::strict_reference_toolchain(
            crate::toolchain::c_compiler(),
            crate::toolchain::CodegenRequirements::default(),
        );
        let bin = dir.path().join("gelu");
        let compiled = std::process::Command::new(&toolchain.compiler)
            .args(&toolchain.compile_flags)
            .arg("-std=c11")
            .arg("-I")
            .arg(dir.path())
            .arg(dir.path().join("main.c"))
            .arg(&staged.archive)
            .args(&toolchain.link_flags)
            .arg("-o")
            .arg(&bin)
            .output()
            .unwrap();
        assert!(
            compiled.status.success(),
            "{}",
            String::from_utf8_lossy(&compiled.stderr)
        );
        let run = std::process::Command::new(&bin).output().unwrap();
        assert!(run.status.success());
        let stdout = String::from_utf8(run.stdout).unwrap();

        let f16_finalize = |v: f32| f16::from_f32(v).to_f32();
        let f16_constant = |v: f64| chelis_types::f16_from_f64_rne(v).to_f32();
        let mut f16_rows = 0;
        let mut mismatches = Vec::new();
        for line in stdout.lines().filter(|line| line.starts_with("f16 ")) {
            let mut fields = line.split(' ').skip(1);
            let input = u16::from_str_radix(fields.next().unwrap(), 16).unwrap();
            let got = u16::from_str_radix(fields.next().unwrap(), 16).unwrap();
            let x = f16::from_bits(input).to_f32();
            let want = f16::from_f32(gelu_reference_f32(x, &f16_finalize, &f16_constant)).to_bits();
            f16_rows += 1;
            if got != want {
                mismatches.push(format!(
                    "f16 gelu({input:#06x}): got {got:#06x}, graph {want:#06x}"
                ));
            }
        }
        assert_eq!(f16_rows, 63488, "every finite f16 input");
        assert!(
            stdout.lines().any(|line| line == "f16 7bff 7bff"),
            "gelu(65504) must be 65504 (chelis#2997)"
        );

        let bf16_line = stdout
            .lines()
            .find(|line| line.starts_with("bf16 "))
            .unwrap();
        let got = u32::from_str_radix(&bf16_line[5..], 16).unwrap();
        assert_eq!(
            got,
            bf16_max.to_bits(),
            "gelu(bf16 max) must be bf16 max (chelis#2997)"
        );

        let f32_rows: Vec<u32> = stdout
            .lines()
            .filter_map(|line| line.strip_prefix("f32 "))
            .map(|bits| u32::from_str_radix(bits, 16).unwrap())
            .collect();
        for (x, got) in f32_inputs.iter().zip(&f32_rows) {
            let want = gelu_reference_f32(*x, &|v| v, &|v| v as f32).to_bits();
            if *got != want {
                mismatches.push(format!(
                    "f32 gelu({x}): got {got:#010x}, graph {want:#010x}"
                ));
            }
        }
        assert_eq!(
            f32_rows[0],
            f32::MAX.to_bits(),
            "gelu(f32 max) must be f32 max (chelis#2997)"
        );
        let f64_rows: Vec<u64> = stdout
            .lines()
            .filter_map(|line| line.strip_prefix("f64 "))
            .map(|bits| u64::from_str_radix(bits, 16).unwrap())
            .collect();
        for (x, got) in f64_inputs.iter().zip(&f64_rows) {
            let want = gelu_reference_f64(*x).to_bits();
            if *got != want {
                mismatches.push(format!(
                    "f64 gelu({x}): got {got:#018x}, graph {want:#018x}"
                ));
            }
        }
        assert_eq!(
            f64_rows[0],
            f64::MAX.to_bits(),
            "gelu(f64 max) must be f64 max (chelis#2997)"
        );
        assert!(
            mismatches.is_empty(),
            "{} mismatches:\n{}",
            mismatches.len(),
            mismatches.join("\n")
        );
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

/// Generated C's f64 storage conversions against the integer reference
/// (`chelis_crmath::profile::storage_reference`, [04-NUM-2], [04-NUM-14]): at every
/// f16 and bf16 rounding boundary per pull request, and over every f32's exact f64
/// widening as a manual gate (`docs/manual_gates.md`).
#[cfg(test)]
mod storage_narrowing_tests {
    use super::append_checked_cast_conversion_helpers;
    use crate::toolchain::{CodegenRequirements, strict_reference_toolchain, test_toolchain};
    use chelis_crmath::profile::{Output, rows, storage_midpoint_inputs, storage_reference};
    use std::io::Write as _;
    use std::path::{Path, PathBuf};
    use std::process::{Command, Stdio};
    use std::sync::atomic::{AtomicU64, Ordering};

    /// The driver, compiled with the product profile.
    fn driver(dir: &Path) -> PathBuf {
        let mut helpers = Vec::new();
        append_checked_cast_conversion_helpers(&mut helpers);
        let mut source = String::from("#include <stdint.h>\n#include <string.h>\n");
        for line in helpers {
            source.push_str(&line);
            source.push('\n');
        }
        source.push_str(include_str!("../tests/fixtures/f64_storage_narrowing.c"));
        let path = dir.join("f64_storage_narrowing.c");
        let program = dir.join("f64_storage_narrowing");
        std::fs::write(&path, source).unwrap();
        let compiler = test_toolchain(CodegenRequirements::default()).compiler;
        let flags =
            strict_reference_toolchain(String::new(), CodegenRequirements::default()).compile_flags;
        let output = Command::new(compiler)
            .args(flags)
            .arg(&path)
            .arg("-o")
            .arg(&program)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        program
    }

    #[test]
    fn f64_storage_conversion_matches_the_reference_at_every_boundary() {
        let dir = tempfile::tempdir().unwrap();
        let program = driver(dir.path());
        let mut inputs: Vec<u64> = rows()
            .iter()
            .filter(|row| {
                row.primitive.width == 64
                    && matches!(row.primitive.result, Output::F16 | Output::Bf16)
            })
            .map(|row| row.operands[0])
            .collect();
        inputs.extend(storage_midpoint_inputs(Output::F16));
        inputs.extend(storage_midpoint_inputs(Output::Bf16));
        let mut child = Command::new(&program)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        let text: String = inputs.iter().map(|bits| format!("{bits:x}\n")).collect();
        let mut stdin = child.stdin.take().unwrap();
        let writer = std::thread::spawn(move || stdin.write_all(text.as_bytes()));
        let output = child.wait_with_output().unwrap();
        writer.join().unwrap().unwrap();
        assert!(output.status.success());
        let printed = String::from_utf8(output.stdout).unwrap();
        assert_eq!(printed.lines().count(), inputs.len());
        let mut bad = Vec::new();
        for (bits, line) in inputs.iter().zip(printed.lines()) {
            let (f16, bf16) = line.split_once(' ').unwrap();
            for (output, got) in [(Output::F16, f16), (Output::Bf16, bf16)] {
                let got = u16::from_str_radix(got, 16).unwrap();
                let expected = storage_reference(*bits, 64, output);
                if got != expected {
                    bad.push(format!(
                        "{output:?} {bits:x}: {got:04x}, reference {expected:04x}"
                    ));
                }
            }
        }
        assert!(
            bad.is_empty(),
            "{} mismatches: {:?}",
            bad.len(),
            &bad[..bad.len().min(8)]
        );
    }

    #[test]
    #[ignore = "manual gate: 2^32 inputs (docs/manual_gates.md)"]
    fn every_f32_widening_narrows_once_to_f16_and_bf16_storage() {
        const CHUNK: u64 = 1 << 24;
        let dir = tempfile::tempdir().unwrap();
        let program = driver(dir.path());
        let threads = std::thread::available_parallelism()
            .expect("the host reports its parallelism")
            .get()
            .div_ceil(2);
        let next = AtomicU64::new(0);
        let total: u64 = std::thread::scope(|scope| {
            let workers: Vec<_> = (0..threads)
                .map(|_| {
                    scope.spawn(|| {
                        let mut count = 0_u64;
                        loop {
                            let low = next.fetch_add(CHUNK, Ordering::Relaxed);
                            if low >= 1 << 32 {
                                break count;
                            }
                            let output = Command::new(&program)
                                .args([format!("{low:x}"), format!("{:x}", low + CHUNK)])
                                .output()
                                .unwrap();
                            assert!(output.status.success());
                            for (offset, pair) in
                                output.stdout.as_chunks::<4>().0.iter().enumerate()
                            {
                                let bits = low + offset as u64;
                                let value = f32::from_bits(u32::try_from(bits).unwrap());
                                let wide = f64::from(value).to_bits();
                                let f16 = u16::from_ne_bytes([pair[0], pair[1]]);
                                let bf16 = u16::from_ne_bytes([pair[2], pair[3]]);
                                count += u64::from(f16 != storage_reference(wide, 64, Output::F16));
                                count +=
                                    u64::from(bf16 != storage_reference(wide, 64, Output::Bf16));
                            }
                        }
                    })
                })
                .collect();
            workers
                .into_iter()
                .map(|worker| worker.join().unwrap())
                .sum()
        });
        assert_eq!(total, 0, "{total} misrounded conversions over 2^32 inputs");
    }
}
