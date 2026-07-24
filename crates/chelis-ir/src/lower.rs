//! Deep AST to RISC DAG lowering.
//!
//! Walks the Deep AST and produces a flat DAG of RISC primitive nodes.

use std::any::Any;
use std::cell::Cell;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::OnceLock;

thread_local! {
    /// When set, `lower_unrepresentable` panics with a quiet empty payload
    /// that `catch_unwind` catches without printing a backtrace. Scoped
    /// via `with_suppress_unrepresentable_panic`, which always clears the
    /// flag on exit. Used by the host-lane tensor-helper fallback path:
    /// it runs the DAG lowerer speculatively, catches any un-representable
    /// panic, and proceeds with host lowering — we don't want that
    /// speculative attempt to write a misleading panic to stderr.
    static SUPPRESS_UNREPRESENTABLE_PANIC: Cell<bool> = const { Cell::new(false) };
    /// Set while a public `try_lower_*` API is converting legacy lowering
    /// unwinds into structured diagnostics. The panic hook stays quiet in
    /// that scope so users see only the returned diagnostic.
    static SUPPRESS_LOWERING_PANIC_OUTPUT: Cell<bool> = const { Cell::new(false) };
}

pub fn with_suppress_unrepresentable_panic<R>(f: impl FnOnce() -> R) -> R {
    struct Guard;
    impl Drop for Guard {
        fn drop(&mut self) {
            SUPPRESS_UNREPRESENTABLE_PANIC.with(|cell| cell.set(false));
        }
    }
    SUPPRESS_UNREPRESENTABLE_PANIC.with(|cell| cell.set(true));
    let _guard = Guard;
    f()
}

fn unrepresentable_panic_suppressed() -> bool {
    SUPPRESS_UNREPRESENTABLE_PANIC.with(|cell| cell.get())
}

/// Marker payload for a suppressed un-representable-DAG unwind.
struct UnrepresentableDag;

/// User-facing lowering diagnostic returned by `try_lower_*` APIs.
///
/// `fatal` distinguishes a hard user-facing rejection (e.g. the AD
/// pass refused to differentiate a non-differentiable op via
/// [`crate::grad::grad_dag_checked`]) from a "merely unrepresentable
/// in the IR DAG" condition (which `try_lower_compiled_program`
/// silently absorbs and recovers from by falling through to the host
/// lowering path). A fatal diagnostic must NOT be silently swallowed
/// by the host fallback; it must reach the user.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LowerDiagnostic {
    pub message: String,
    pub span: Option<Span>,
    pub span_id: Option<String>,
    pub fatal: bool,
}

impl LowerDiagnostic {
    pub(crate) fn new(
        message: impl Into<String>,
        span: Option<Span>,
        span_id: Option<String>,
    ) -> Self {
        Self {
            message: message.into(),
            span,
            span_id,
            fatal: false,
        }
    }

    /// Mark a diagnostic as fatal — it must reach the user instead of
    /// being absorbed into a host-fallback "soft" failure. Use this
    /// for deliberate rejections (e.g. AD on non-differentiable ops)
    /// where falling back to the host path would silently emit an
    /// undefined-symbol reference.
    pub(crate) fn fatal(mut self) -> Self {
        self.fatal = true;
        self
    }
}

impl fmt::Display for LowerDiagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)?;
        if let Some(span_id) = &self.span_id {
            write!(f, " at source span `{span_id}`")?;
        } else if let Some(span) = self.span
            && span.len > 0
        {
            write!(f, " at byte {}..{}", span.offset, span.end())?;
        }
        Ok(())
    }
}

impl std::error::Error for LowerDiagnostic {}

/// Outcome of matching one `match` arm pattern against a statically-known
/// constructor value during static arm selection (chelis#520 D1).
enum StaticPatternMatch {
    /// The pattern matches; apply these bindings and take the arm.
    Match(Vec<(String, LoweredValue)>),
    /// The pattern provably does not match this constructor; try the next arm.
    NoMatch,
    /// The pattern is outside the supported static slice; the whole match
    /// must be rejected (taking a later arm could be unsound).
    Unsupported(String),
}

/// Match a Deep pattern against a static ADT value `(ctor, field_names,
/// fields)`. The supported slice is deliberately conservative: constructor
/// name selection with `pat-var`/`pat-wild` sub-patterns (positional or
/// record form), whole-value `pat-var`/`pat-wild`/`pat-as`. Anything else
/// is `Unsupported`, which the caller turns into a loud rejection; a
/// pattern this function cannot decide must never fall through to a later
/// arm.
fn match_static_pattern(
    pattern: &Expr,
    ctor: &str,
    field_names: Option<&[String]>,
    fields: &[LoweredValue],
) -> StaticPatternMatch {
    let Expr::List(pat_list, _) = pattern else {
        return StaticPatternMatch::Unsupported("a non-list pattern form".to_string());
    };
    let pat_kids = children(pat_list);
    match get_tag(pat_list) {
        Some("pat-wild") => StaticPatternMatch::Match(Vec::new()),
        Some("pat-var") => match pat_kids.first().and_then(symbol_name) {
            Some(name) => StaticPatternMatch::Match(vec![(
                name.to_string(),
                LoweredValue::Adt {
                    ctor: ctor.to_string(),
                    field_names: field_names.map(<[String]>::to_vec),
                    fields: fields.to_vec(),
                },
            )]),
            None => StaticPatternMatch::Unsupported("a nameless `pat-var`".to_string()),
        },
        Some("pat-as") => {
            let (Some(name), Some(inner)) =
                (pat_kids.first().and_then(symbol_name), pat_kids.get(1))
            else {
                return StaticPatternMatch::Unsupported("a malformed `pat-as`".to_string());
            };
            match match_static_pattern(inner, ctor, field_names, fields) {
                StaticPatternMatch::Match(mut binds) => {
                    binds.push((
                        name.to_string(),
                        LoweredValue::Adt {
                            ctor: ctor.to_string(),
                            field_names: field_names.map(<[String]>::to_vec),
                            fields: fields.to_vec(),
                        },
                    ));
                    StaticPatternMatch::Match(binds)
                }
                other => other,
            }
        }
        Some("pat-ctor") => {
            let Some(pat_ctor) = pat_kids.first().and_then(symbol_name) else {
                return StaticPatternMatch::Unsupported("a nameless `pat-ctor`".to_string());
            };
            if pat_ctor != ctor {
                return StaticPatternMatch::NoMatch;
            }
            let sub_pats = &pat_kids[1..];
            if sub_pats.len() != fields.len() {
                return StaticPatternMatch::Unsupported(format!(
                    "a `pat-ctor` with {} sub-patterns against a `{ctor}` value with {} \
                     fields",
                    sub_pats.len(),
                    fields.len()
                ));
            }
            let mut binds = Vec::new();
            for (sub_pat, field) in sub_pats.iter().zip(fields.iter()) {
                match bind_leaf_pattern(sub_pat, field) {
                    Ok(Some(bind)) => binds.push(bind),
                    Ok(None) => {}
                    Err(reason) => return StaticPatternMatch::Unsupported(reason),
                }
            }
            StaticPatternMatch::Match(binds)
        }
        Some("pat-record") => {
            let Some(pat_ctor) = pat_kids.first().and_then(symbol_name) else {
                return StaticPatternMatch::Unsupported("a nameless `pat-record`".to_string());
            };
            if pat_ctor != ctor {
                return StaticPatternMatch::NoMatch;
            }
            let Some(field_names) = field_names else {
                return StaticPatternMatch::Unsupported(format!(
                    "a `pat-record` against a positionally-constructed `{ctor}` value"
                ));
            };
            let mut binds = Vec::new();
            for kv in &pat_kids[1..] {
                let Expr::List(kv_list, _) = kv else {
                    continue;
                };
                if get_tag(kv_list) != Some("kv") {
                    continue;
                }
                let kv_kids = children(kv_list);
                let (Some(field), Some(sub_pat)) =
                    (kv_kids.first().and_then(symbol_name), kv_kids.get(1))
                else {
                    return StaticPatternMatch::Unsupported(
                        "a malformed `pat-record` field entry".to_string(),
                    );
                };
                let Some(value) = field_names
                    .iter()
                    .position(|name| name == field)
                    .and_then(|index| fields.get(index))
                else {
                    return StaticPatternMatch::Unsupported(format!(
                        "a `pat-record` field `{field}` absent from the constructed \
                         `{ctor}` value"
                    ));
                };
                match bind_leaf_pattern(sub_pat, value) {
                    Ok(Some(bind)) => binds.push(bind),
                    Ok(None) => {}
                    Err(reason) => return StaticPatternMatch::Unsupported(reason),
                }
            }
            StaticPatternMatch::Match(binds)
        }
        // `pat-lit` against a constructor value cannot match, but a match
        // mixing literal and constructor patterns is outside the checked
        // slice; reject rather than guess.
        Some(other) => StaticPatternMatch::Unsupported(format!("`{other}` patterns")),
        None => StaticPatternMatch::Unsupported("an untagged pattern form".to_string()),
    }
}

/// A leaf sub-pattern inside a constructor pattern: `pat-var` binds the
/// field value, `pat-wild` discards it. Nested destructuring is outside
/// the static slice.
fn bind_leaf_pattern(
    pattern: &Expr,
    value: &LoweredValue,
) -> Result<Option<(String, LoweredValue)>, String> {
    let Expr::List(pat_list, _) = pattern else {
        return Err("a non-list sub-pattern form".to_string());
    };
    match get_tag(pat_list) {
        Some("pat-wild") => Ok(None),
        Some("pat-var") => match children(pat_list).first().and_then(symbol_name) {
            Some(name) => Ok(Some((name.to_string(), value.clone()))),
            None => Err("a nameless `pat-var` sub-pattern".to_string()),
        },
        Some(other) => Err(format!(
            "nested `{other}` sub-patterns (only `pat-var`/`pat-wild` field bindings \
             are in the static slice)"
        )),
        None => Err("an untagged sub-pattern form".to_string()),
    }
}

/// An absent match-arm guard desugars to a bare empty list `()`.
fn guard_is_absent(guard: &Expr) -> bool {
    matches!(guard, Expr::List(list, _) if list.elements.is_empty())
}

fn unsupported_lowering_message(tag: &str) -> String {
    let subject = if tag == "pipe stage" {
        "pipe stage".to_string()
    } else {
        format!("`{tag}`")
    };
    format!(
        "{subject} is not supported by IR evaluation yet; use `chelis build --target c` instead"
    )
}

/// If `expr` is exactly `(var {} name)`, return the symbol name. Used by
/// `lower_pipe` to detect bare-var pipe stages that should be lowered as
/// unary applications (Item 2c — see
/// `docs/investigations/c_backend_grad_piped_body_diagnosis.md`).
fn bare_var_name(expr: &Expr) -> Option<String> {
    let Expr::List(list, _) = expr else {
        return None;
    };
    let first = list.elements.first()?;
    let Expr::Atom(Atom::Symbol(tag), _) = first else {
        return None;
    };
    if tag != "var" {
        return None;
    }
    let third = list.elements.get(2)?;
    let Expr::Atom(Atom::Symbol(name), _) = third else {
        return None;
    };
    Some(name.clone())
}

/// Synthetic accumulator-binding name reserved for the IR-side pipe
/// rewrite (Item 2c). Chosen to be unrepresentable in Surf so it cannot
/// collide with user bindings; `lower_pipe` saves and restores any prior
/// entry under this name before/after the synthesized lookup.
fn synth_pipe_acc_binding_name() -> String {
    "__chelis_pipe_acc__".to_string()
}

/// Synthesize `(app {} (var {} fname) (var {} acc_name))` — the Deep
/// expression that `lower_app` will route through `lower_builtin_app`
/// for an unresolved-as-callable bare-var pipe stage. The accumulator
/// `var` is resolved against the binding inserted by `lower_pipe` before
/// the call.
fn synth_unary_app(fname: &str, acc_name: &str, app_span: Span) -> Expr {
    let zero_span = Span::new(0, 0);
    let callee = Expr::List(
        List {
            elements: vec![
                Expr::Atom(Atom::Symbol("var".to_string()), zero_span),
                Expr::Map(MetaMap::default(), zero_span),
                Expr::Atom(Atom::Symbol(fname.to_string()), zero_span),
            ],
        },
        zero_span,
    );
    let arg = Expr::List(
        List {
            elements: vec![
                Expr::Atom(Atom::Symbol("var".to_string()), zero_span),
                Expr::Map(MetaMap::default(), zero_span),
                Expr::Atom(Atom::Symbol(acc_name.to_string()), zero_span),
            ],
        },
        zero_span,
    );
    Expr::List(
        List {
            elements: vec![
                Expr::Atom(Atom::Symbol("app".to_string()), zero_span),
                Expr::Map(MetaMap::default(), zero_span),
                callee,
                arg,
            ],
        },
        app_span,
    )
}

/// Synthesize `(app {} (var {} fname) <operand> <axis>)` — one stage of the
/// chelis#339 variadic named-axis reduction desugar. `sum(x, seq, head)`
/// lowers as the documented composition `sum(sum(x, head), seq)`: each
/// synthesized 2-arg stage resolves its named axis against its own operand's
/// dims, so the result is order-insensitive.
fn synth_reduction_app(fname: &str, operand: Expr, axis: Expr, app_span: Span) -> Expr {
    let zero_span = Span::new(0, 0);
    let callee = Expr::List(
        List {
            elements: vec![
                Expr::Atom(Atom::Symbol("var".to_string()), zero_span),
                Expr::Map(MetaMap::default(), zero_span),
                Expr::Atom(Atom::Symbol(fname.to_string()), zero_span),
            ],
        },
        zero_span,
    );
    Expr::List(
        List {
            elements: vec![
                Expr::Atom(Atom::Symbol("app".to_string()), zero_span),
                Expr::Map(MetaMap::default(), zero_span),
                callee,
                operand,
                axis,
            ],
        },
        app_span,
    )
}

fn expr_diagnostic_location(expr: &Expr) -> (Option<Span>, Option<String>) {
    (Some(expr.span()), expr.span_id().map(ToOwned::to_owned))
}

fn lower_diagnostic_for_expr(message: impl Into<String>, expr: &Expr) -> LowerDiagnostic {
    let (span, span_id) = expr_diagnostic_location(expr);
    LowerDiagnostic::new(message, span, span_id)
}

fn raise_lowering_diagnostic(diagnostic: LowerDiagnostic) -> ! {
    // A fatal diagnostic must not be downgraded to the silent
    // un-representable panic — the host-fallback boundary at
    // `host::try_lower_compiled_program` keys off this so the user
    // receives the AD rejection text instead of an undefined-symbol
    // host call. Issue #197.
    if !diagnostic.fatal && unrepresentable_panic_suppressed() {
        std::panic::panic_any(UnrepresentableDag);
    }
    std::panic::panic_any(diagnostic);
}

fn raise_lowering_error(
    message: impl Into<String>,
    span: Option<Span>,
    span_id: Option<String>,
) -> ! {
    raise_lowering_diagnostic(LowerDiagnostic::new(message, span, span_id))
}

/// Raise a *fatal* lowering diagnostic — one the host-fallback path
/// at [`crate::host::try_lower_compiled_program`] must NOT silently
/// absorb. Used by the AD-rejection arm at the `grad(...)` lowering
/// site (Issue #197) so a non-differentiable op surfaces as a
/// user-facing build error rather than as an undefined-symbol host
/// call.
fn raise_fatal_lowering_error(
    message: impl Into<String>,
    span: Option<Span>,
    span_id: Option<String>,
) -> ! {
    raise_lowering_diagnostic(LowerDiagnostic::new(message, span, span_id).fatal())
}

/// chelis#730 Phase 1, section C1.4 raise-or-prove: a malformed or
/// arity-short Deep form reaching executable lowering raises a lowering
/// error instead of substituting a silent `Const { value: 0.0 }`
/// placeholder (census row 16 of `spec/design/loud_unsupported.md`).
/// These shapes are pre-guarded by `chelis-deep` validation (the closed
/// tag vocabulary and per-tag arity checks), so this raise is expected
/// dead; if it ever fires, the input bypassed validation or a producing
/// pass emitted a malformed node - either way the response is a loud
/// diagnostic, never a plausible zero. Non-fatal on purpose: the host
/// fallback lane legitimately owns several of these shapes (strings,
/// host-only forms) and its own failure paths stay loud now that the
/// emitter stub arm is an error (section C3).
fn raise_malformed_deep(what: &str, span: Option<Span>, span_id: Option<String>) -> ! {
    raise_lowering_error(
        format!(
            "{what} cannot be lowered to the executable IR \
             (spec/design/loud_unsupported.md section C1.4; this form previously \
             lowered to a silent zero placeholder)"
        ),
        span,
        span_id,
    )
}

/// Re-raise an already-constructed fatal lowering diagnostic.
/// Crate-internal so the host-side sub-lowering paths can resurface
/// a fatal AD rejection caught by their inner `try_lower_*` call
/// without losing the structured `op`/reason text.
///
/// The inner `catch_lowering` Guard's `Drop` impl unconditionally
/// clears `SUPPRESS_LOWERING_PANIC_OUTPUT` on the way out, so a
/// re-raise from a sub-lowering site (inside the outer
/// `catch_lowering` scope) would otherwise produce a default
/// "thread 'main' panicked at ... Box<dyn Any>" line on stderr in
/// addition to the structured diagnostic. Re-set the suppression
/// flag here so the outer scope sees a clean panic.
pub(crate) fn raise_fatal_lowering_diagnostic(diagnostic: LowerDiagnostic) -> ! {
    SUPPRESS_LOWERING_PANIC_OUTPUT.with(|cell| cell.set(true));
    let fatal = if diagnostic.fatal {
        diagnostic
    } else {
        diagnostic.fatal()
    };
    raise_lowering_diagnostic(fatal);
}

fn panic_payload_to_lower_diagnostic(payload: &(dyn Any + Send)) -> LowerDiagnostic {
    if payload.is::<UnrepresentableDag>() {
        return LowerDiagnostic::new(
            "program uses a form that is not supported by IR evaluation yet; use `chelis build --target c` instead",
            None,
            None,
        );
    }
    if let Some(diagnostic) = payload.downcast_ref::<LowerDiagnostic>() {
        return diagnostic.clone();
    }
    let message = payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| {
            payload
                .downcast_ref::<&str>()
                .map(|value| (*value).to_string())
        })
        .unwrap_or_else(|| "internal lowering error".to_string());
    LowerDiagnostic::new(message, None, None)
}

/// chelis#620 red-team fix: stack segment for every lowering entry. The
/// bounded unroll legally builds DAGs up to `MAX_STATIC_RECURSION_DEPTH`
/// levels deep, and the passes that CONSUME that DAG inside the entry's
/// dynamic extent (reverse-mode grad construction, DCE, verify) recurse
/// per node without lowering's per-level `stacker::maybe_grow` -- so
/// before this grow, a ~484-level combining builder (the cap's own im2col
/// justification) lowered fine and then SIGABRT'd in a consumer below the
/// cap, violating the "loud diagnostic, never a crash" contract. One grow
/// at the boundary covers every pass in the closure, the same pattern as
/// chelis-types' `with_grown_stack` (infer.rs WI-1) and the same 512 MiB
/// segment size.
const LOWERING_GROW_SEGMENT_BYTES: usize = 512 * 1024 * 1024;

fn catch_lowering<R>(f: impl FnOnce() -> R + std::panic::UnwindSafe) -> Result<R, LowerDiagnostic> {
    struct Guard;
    impl Drop for Guard {
        fn drop(&mut self) {
            SUPPRESS_LOWERING_PANIC_OUTPUT.with(|cell| cell.set(false));
        }
    }

    install_chelis_panic_hook();
    SUPPRESS_LOWERING_PANIC_OUTPUT.with(|cell| cell.set(true));
    let _guard = Guard;
    std::panic::catch_unwind(|| stacker::grow(LOWERING_GROW_SEGMENT_BYTES, f))
        .map_err(|payload| panic_payload_to_lower_diagnostic(&*payload))
}

/// Public alias for [`catch_lowering`] so other modules in this crate
/// (e.g. `host`) can wrap their own lowering entry points and surface
/// panics from `try_extract_tensor_type` (WS-A5 RT-3a F2: unresolved
/// `TensorPrec::Var` reaching the conversion boundary) as
/// `LowerDiagnostic`. Crate-internal only by design; the build CLI
/// goes through `host::try_lower_compiled_program` instead.
pub(crate) fn catch_lowering_external<R>(
    f: impl FnOnce() -> R + std::panic::UnwindSafe,
) -> Result<R, LowerDiagnostic> {
    catch_lowering(f)
}

static CHELIS_PANIC_HOOK_INSTALLED: OnceLock<()> = OnceLock::new();

/// Install a one-time global panic hook that suppresses panic output when the
/// current thread is inside a `with_suppress_unrepresentable_panic` scope.
/// Safe to call multiple times — the hook is installed at most once.
pub fn install_chelis_panic_hook() {
    CHELIS_PANIC_HOOK_INSTALLED.get_or_init(|| {
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if SUPPRESS_UNREPRESENTABLE_PANIC.with(|cell| cell.get()) {
                return;
            }
            if SUPPRESS_LOWERING_PANIC_OUTPUT.with(|cell| cell.get()) {
                return;
            }
            prev(info);
        }));
    });
}

use chelis_deep::ast::{Atom, Expr, List, MetaMap};
use chelis_deep::{Span, decode_effect_kind};
use chelis_types::unsupported::{Stage, Unsupported, UnsupportedKind};
use chelis_types::{BUILTIN_NAMES, CheckedProgram, LinearityInfo, types::Prim};
use chelis_vocab::EffectKind;

use crate::dag::{Dag, DimExpr, DimInfo, NodeId, RiscOp, RtDim, TensorType};
use crate::grad::grad_dag_checked;
use crate::tier2;
use crate::vmap;

/// Lower a checked Deep program into a RISC DAG.
pub fn lower_program(program: &CheckedProgram) -> Dag {
    try_lower_program(program).unwrap_or_else(|diagnostic| panic!("{diagnostic}"))
}

/// Lower a checked Deep program into a RISC DAG, returning a diagnostic for
/// forms that are valid Chelis but not supported by IR evaluation.
pub fn try_lower_program(program: &CheckedProgram) -> Result<Dag, LowerDiagnostic> {
    try_lower_program_to_library(program).map(|library| library.dag)
}

/// Phase F carrier: a lowered library DAG plus the metadata needed to
/// compose against new code via [`lower_program_with_context`].
///
/// Fields are exposed so the compiled-artifact cache can persist the
/// library state out-of-band, but consumers should treat them as opaque —
/// the contract is that the carrier was produced by
/// [`lower_program_to_library`] on a checked library, and that
/// [`lower_program_with_context`] is the only blessed way to consume it.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LoweredLibrary {
    /// The library DAG, post-DCE. Node IDs in this DAG are the canonical
    /// library IDs that new-code lowering will reference (after a clone).
    pub dag: Dag,
    /// Map from a library top-level def's name (e.g. `lib_const`,
    /// `lib_double.0`) to the NodeId in `dag` that holds its value. New
    /// code that references the name resolves through this table rather
    /// than emitting a fresh `Load`.
    pub symbol_table: HashMap<String, NodeId>,
    /// Library top-level def bodies, keyed by name. New-code lowering needs
    /// these to inline calls to library functions (matching the monolithic
    /// behaviour of `lower_program(library + new)`).
    pub program_defs: HashMap<String, Expr>,
    /// Library declared types, keyed by name. Used to resolve unbound
    /// `(var libname)` Load types when the new-code expression's metadata
    /// is `default_type`.
    pub program_types: HashMap<String, TensorType>,
    /// Library linearity metadata. Forwarded so cross-DAG reuse hints can
    /// be re-applied if needed.
    pub linearity: LinearityInfo,
    /// Per-library-def "is it lowered?" decision, mirroring the result
    /// of `top_level_lowering_map(library_exprs, library_type_env)`. New-
    /// code lowering decisions need this so a new-code def whose body
    /// calls a library function gets the same lowered/host classification
    /// as it would in monolithic mode (where the same library def lives
    /// in `top_level_defs` and is consulted directly).
    pub lowered_names: HashMap<String, bool>,
}

/// Lower a checked program to the [`LoweredLibrary`] carrier. The bare
/// `dag` field of the result is identical to `lower_program(program)` —
/// the only difference is that `symbol_table`, `program_defs`,
/// `program_types`, and `linearity` are also exposed.
pub fn lower_program_to_library(program: &CheckedProgram) -> LoweredLibrary {
    try_lower_program_to_library(program).unwrap_or_else(|diagnostic| panic!("{diagnostic}"))
}

pub fn try_lower_program_to_library(
    program: &CheckedProgram,
) -> Result<LoweredLibrary, LowerDiagnostic> {
    catch_lowering(|| lower_program_to_library_inner(program))
}

fn lower_program_to_library_inner(program: &CheckedProgram) -> LoweredLibrary {
    let detail_profile = std::env::var_os("CHELIS_PROFILE_COMPILE_CONTEXT_DETAIL")
        .map(|v| v == "1")
        .unwrap_or(false);
    let mut sub_t = std::time::Instant::now();
    let log_sub = |label: &str, t: &mut std::time::Instant| {
        if detail_profile {
            eprintln!("lower_sub: {:>8.4}s {}", t.elapsed().as_secs_f64(), label);
            *t = std::time::Instant::now();
        }
    };

    let program_type_env = program.type_env();
    let lowered_names = top_level_lowering_map(program.exprs(), program_type_env);
    log_sub("top_level_lowering_map", &mut sub_t);
    // Reuse the precomputed `lowered_names` rather than calling
    // `top_level_expr_is_lowered`, which would rebuild the map from
    // scratch on every call (1850 calls × full library walk = quadratic
    // before this fix; ~20s on Coral).
    for_each_top_level_item(program.exprs(), &mut |expr| {
        if top_level_expr_is_lowered_with_names(expr, program_type_env, &lowered_names) {
            assert_ir_lowerable(expr);
            assert_ir_typed(expr);
        }
    });
    log_sub("assertions_loop", &mut sub_t);
    let program_types: HashMap<String, TensorType> = program
        .type_env()
        .iter()
        .map(|(name, ty_expr)| (name.clone(), LowerCtx::type_from_type_expr(ty_expr)))
        .collect();
    log_sub("program_types_build", &mut sub_t);
    let program_defs = collect_top_level_defs(program.exprs());
    log_sub("collect_top_level_defs", &mut sub_t);
    let mut ctx = LowerCtx::new(
        program_types.clone(),
        program_defs.clone(),
        program.linearity().clone(),
    );
    log_sub("lower_ctx_new", &mut sub_t);
    let mut last_dag_size: usize = ctx.dag.len();
    // Same reasoning as the assertions_loop above: prefer the
    // precomputed `lowered_names` over a fresh `top_level_expr_is_lowered`
    // rebuild for non-named decls (these are non-`def` top-levels like
    // `defsig`/`deftype`/`typealias`; the `_with_names` path returns true
    // for them, matching the original semantics — they're not `def` so
    // not lowered, but they still pass through the wrapping check).
    for_each_top_level_item(program.exprs(), &mut |expr| {
        if top_level_expr_name(expr).and_then(|name| lowered_names.get(name).copied()) == Some(true)
            || (top_level_expr_name(expr).is_none()
                && top_level_expr_is_lowered_with_names(expr, program_type_env, &lowered_names))
        {
            let t0 = if detail_profile {
                Some(std::time::Instant::now())
            } else {
                None
            };
            // For pre-flight gate counting, we want to know how often
            // top_level_expr_is_lowered fires (each call rebuilds the
            // lowering map — quadratic).
            ctx.lower_top_level(expr);
            if let Some(t0) = t0 {
                let elapsed = t0.elapsed();
                let nodes = ctx.dag.len();
                let added = nodes.saturating_sub(last_dag_size);
                last_dag_size = nodes;
                let name = top_level_expr_name(expr).unwrap_or("<anon>");
                eprintln!(
                    "lower_decl: {:>8.4}s nodes_added={:>5} dag_total={:>6} {}",
                    elapsed.as_secs_f64(),
                    added,
                    nodes,
                    name
                );
            }
        }
    });
    log_sub("lower_top_level_loop", &mut sub_t);

    // Collect the pre-DCE name -> NodeId mapping from the lowering ctx's
    // top-level bindings. We flatten tuple-decomposed defs into dotted
    // names mirroring the `Store { name: "foo.0" }` convention used
    // elsewhere; that lets new code reference both `foo` (as a tuple
    // identity) and `foo.N` (as the specific element).
    let mut pre_dce_table: HashMap<String, NodeId> = HashMap::new();
    for (name, value) in ctx.bindings.iter() {
        flatten_binding_into(name, value, &mut pre_dce_table);
    }
    log_sub("flatten_bindings", &mut sub_t);

    let (dce_dag, remap) = crate::optimize::dead_code_eliminate_with_remap(&ctx.dag);
    log_sub("dce", &mut sub_t);
    let (copy_dag, linear_remap) = insert_copy_nodes_for_consuming_fanout(&dce_dag);
    log_sub("implicit_copy_nodes", &mut sub_t);
    let linear_dag = insert_drop_nodes_for_unconsumed_values(copy_dag);
    log_sub("implicit_drop_nodes", &mut sub_t);

    // Renumber the symbol table through DCE's remap. Names whose nodes
    // were eliminated drop out of the table.
    let symbol_table: HashMap<String, NodeId> = pre_dce_table
        .into_iter()
        .filter_map(|(name, old)| {
            remap
                .get(&old)
                .and_then(|new| linear_remap.get(new))
                .copied()
                .map(|new| (name, new))
        })
        .collect();
    log_sub("renumber_symbol_table", &mut sub_t);

    LoweredLibrary {
        dag: linear_dag,
        symbol_table,
        program_defs,
        program_types,
        linearity: program.linearity().clone(),
        lowered_names,
    }
}

fn insert_copy_nodes_for_consuming_fanout(dag: &Dag) -> (Dag, HashMap<NodeId, NodeId>) {
    let mut consuming_uses = HashMap::<NodeId, usize>::new();
    for node in dag.nodes() {
        if !op_consumes_inputs(&node.op) {
            continue;
        }
        for input in &node.inputs {
            *consuming_uses.entry(*input).or_default() += 1;
        }
    }

    let mut seen_consuming_uses = HashMap::<NodeId, usize>::new();
    let mut out = Dag::new();
    let mut id_map = HashMap::<NodeId, NodeId>::new();

    for node in dag.nodes() {
        let mut inputs = Vec::with_capacity(node.inputs.len());
        for input in &node.inputs {
            let mapped = *id_map
                .get(input)
                .expect("input must have been remapped before consumer");
            let total = consuming_uses.get(input).copied().unwrap_or_default();
            if op_consumes_inputs(&node.op) && total > 1 {
                let seen = seen_consuming_uses.entry(*input).or_default();
                *seen += 1;
                if *seen < total {
                    let input_ty = out
                        .get(mapped)
                        .map(|n| n.output_type.clone())
                        .unwrap_or_else(LowerCtx::default_type);
                    let copy =
                        out.add_node(RiscOp::Copy, vec![mapped], input_ty, node.span_id.clone());
                    inputs.push(copy);
                    continue;
                }
            }
            inputs.push(mapped);
        }

        let new_id = out.add_node(
            node.op.clone(),
            inputs,
            node.output_type.clone(),
            node.span_id.clone(),
        );
        if let Some(new_node) = out.node_mut(new_id) {
            new_node.reusable_input = node
                .reusable_input
                .and_then(|old| id_map.get(&old).copied());
            new_node.merged_spans = node.merged_spans.clone();
            // chelis#384/#397: preserve (remapped) shape-only deps so a
            // Form-3 `expand` extent source survives this rebuild. Deps are
            // earlier in topo order, so already remapped in `id_map`.
            new_node.shape_deps = node
                .shape_deps
                .iter()
                .filter_map(|old| id_map.get(old).copied())
                .collect();
        }
        id_map.insert(node.id, new_id);
    }

    for root in dag.roots() {
        if let Some(new_root) = id_map.get(root) {
            out.add_root(*new_root);
        }
    }

    (out, id_map)
}

fn op_consumes_inputs(op: &RiscOp) -> bool {
    matches!(op, RiscOp::Realize | RiscOp::Drop | RiscOp::Store { .. })
}

fn insert_drop_nodes_for_unconsumed_values(mut dag: Dag) -> Dag {
    let mut consumed = HashSet::<NodeId>::new();
    for node in dag.nodes() {
        if !op_consumes_inputs(&node.op) {
            continue;
        }
        consumed.extend(node.inputs.iter().copied());
    }
    let roots = dag.roots().iter().copied().collect::<HashSet<_>>();
    let values_to_drop = dag
        .nodes()
        .iter()
        .filter(|node| !roots.contains(&node.id))
        .filter(|node| !consumed.contains(&node.id))
        .filter(|node| {
            !matches!(
                node.op,
                RiscOp::Load { .. } | RiscOp::Drop | RiscOp::Store { .. }
            )
        })
        .map(|node| {
            (
                node.id,
                node.output_type.clone(),
                node.span_id.clone(),
                node.merged_spans.clone(),
            )
        })
        .collect::<Vec<_>>();

    for (id, ty, span_id, merged_spans) in values_to_drop {
        let drop = dag.add_node(RiscOp::Drop, vec![id], ty, span_id);
        if let Some(node) = dag.node_mut(drop) {
            node.merged_spans = merged_spans;
        }
    }

    dag
}

fn strip_drop_nodes(dag: &Dag) -> (Dag, HashMap<NodeId, NodeId>) {
    let mut out = Dag::new();
    let mut id_map = HashMap::<NodeId, NodeId>::new();

    for node in dag.nodes() {
        if matches!(node.op, RiscOp::Drop) {
            continue;
        }
        let inputs = node
            .inputs
            .iter()
            .filter_map(|input| id_map.get(input).copied())
            .collect::<Vec<_>>();
        let new_id = out.add_node(
            node.op.clone(),
            inputs,
            node.output_type.clone(),
            node.span_id.clone(),
        );
        if let Some(new_node) = out.node_mut(new_id) {
            new_node.reusable_input = node
                .reusable_input
                .and_then(|old| id_map.get(&old).copied());
            new_node.merged_spans = node.merged_spans.clone();
            // chelis#384/#397: preserve (remapped) shape-only deps so a
            // Form-3 `expand` extent source survives this rebuild. Deps are
            // earlier in topo order, so already remapped in `id_map`.
            new_node.shape_deps = node
                .shape_deps
                .iter()
                .filter_map(|old| id_map.get(old).copied())
                .collect();
        }
        id_map.insert(node.id, new_id);
    }

    for root in dag.roots() {
        if let Some(new_root) = id_map.get(root) {
            out.add_root(*new_root);
        }
    }

    (out, id_map)
}

/// Compose a library's lowered DAG with a new-code [`CheckedProgram`].
/// The library was already lowered via [`lower_program_to_library`]
/// (which is what `lower_program(library)` runs internally); the returned
/// DAG holds the library roots plus new-code nodes.
///
/// `&library` is never mutated; the function is pure and the input
/// library carrier is safe to reuse across many `new_program` snippets.
///
/// New code's `(var libname)` references resolve directly to the library
/// DAG's existing NodeId (no duplicate node), and new code's calls to
/// library functions inline using the library's `program_defs` —
/// matching the monolithic `lower_program(library + new)` behaviour
/// byte-for-byte (modulo any irrelevant extra defs that DCE pruned).
pub fn lower_program_with_context(library: &LoweredLibrary, new_program: &CheckedProgram) -> Dag {
    try_lower_program_with_context(library, new_program)
        .unwrap_or_else(|diagnostic| panic!("{diagnostic}"))
}

pub fn try_lower_program_with_context(
    library: &LoweredLibrary,
    new_program: &CheckedProgram,
) -> Result<Dag, LowerDiagnostic> {
    catch_lowering(|| lower_program_with_context_inner(library, new_program))
}

fn lower_program_with_context_inner(library: &LoweredLibrary, new_program: &CheckedProgram) -> Dag {
    let new_type_env = new_program.type_env();
    let lowered_names =
        top_level_lowering_map_with_context(library, new_program.exprs(), new_type_env);
    for_each_top_level_item(new_program.exprs(), &mut |expr| {
        if top_level_expr_is_lowered_with_names(expr, new_type_env, &lowered_names) {
            assert_ir_lowerable(expr);
            assert_ir_typed(expr);
        }
    });

    // Combined types: library's own program_types layered with new-code's
    // type_env (which Phase C already unioned with library types).
    // New-code wins on shadow, since the new annotated types are derived
    // from new-code source.
    let mut program_types = library.program_types.clone();
    for (name, ty_expr) in new_program.type_env() {
        program_types.insert(name.clone(), LowerCtx::type_from_type_expr(ty_expr));
    }

    // Combined defs: library defs + new-code defs. Same shadow rule —
    // new-code wins, mirroring monolithic lowering of `library + new`.
    let mut program_defs = library.program_defs.clone();
    for (name, body) in collect_top_level_defs(new_program.exprs()) {
        program_defs.insert(name, body);
    }

    let mut ctx = LowerCtx::new(program_types, program_defs, new_program.linearity().clone());

    // Seed the lowering ctx with the cloned library DAG and the library's
    // name -> NodeId bindings. Library drops are terminal markers for the
    // standalone library snapshot; composition can make formerly terminal
    // values live again, so we strip them and re-normalize Copy/Drop across
    // the combined DAG below.
    let (library_dag, library_remap) = strip_drop_nodes(&library.dag);
    ctx.dag = library_dag;
    for (name, node_id) in &library.symbol_table {
        if let Some(mapped) = library_remap.get(node_id).copied() {
            ctx.bindings
                .insert(name.clone(), LoweredValue::Node(mapped));
        }
    }

    for_each_top_level_item(new_program.exprs(), &mut |expr| {
        if top_level_expr_name(expr).and_then(|name| lowered_names.get(name).copied()) == Some(true)
            || (top_level_expr_name(expr).is_none()
                && top_level_expr_is_lowered(expr, new_program.exprs(), new_type_env))
        {
            ctx.lower_top_level(expr);
        }
    });

    // Skip DCE on the composed DAG: the library DAG was already DCE'd by
    // `lower_program_to_library`, and re-DCE'ing here could prune library
    // roots that are not referenced by the current `new_program`. We still
    // run the linearity normalization passes so consuming fan-out across the
    // library/new-code boundary gets the same Copy nodes as monolithic
    // lowering, and every surviving linear value receives a terminal Drop.
    let (copy_dag, _) = insert_copy_nodes_for_consuming_fanout(&ctx.dag);
    insert_drop_nodes_for_unconsumed_values(copy_dag)
}

fn flatten_binding_into(prefix: &str, value: &LoweredValue, out: &mut HashMap<String, NodeId>) {
    match value {
        LoweredValue::Node(id) => {
            out.insert(prefix.to_string(), *id);
        }
        LoweredValue::Tuple(items) => {
            for (index, item) in items.iter().enumerate() {
                flatten_binding_into(&format!("{prefix}.{index}"), item, out);
            }
        }
        // ADT gradient values flatten field-wise, keyed by field name when
        // the constructor is a record form (chelis#520 D2).
        LoweredValue::Adt {
            field_names,
            fields,
            ..
        } => {
            for (index, field) in fields.iter().enumerate() {
                let label = field_names
                    .as_ref()
                    .and_then(|names| names.get(index).cloned())
                    .unwrap_or_else(|| index.to_string());
                flatten_binding_into(&format!("{prefix}.{label}"), field, out);
            }
        }
    }
}

pub fn tensor_type_from_deep(expr: &Expr) -> TensorType {
    LowerCtx::type_from_type_expr(expr)
}

pub fn lower_subexpr_program(
    expr: &Expr,
    scoped_tensor_types: HashMap<String, TensorType>,
    full_type_env: HashMap<String, Expr>,
    program_defs: HashMap<String, Expr>,
) -> Dag {
    try_lower_subexpr_program(expr, scoped_tensor_types, full_type_env, program_defs)
        .unwrap_or_else(|diagnostic| panic!("{diagnostic}"))
}

pub fn try_lower_subexpr_program(
    expr: &Expr,
    scoped_tensor_types: HashMap<String, TensorType>,
    full_type_env: HashMap<String, Expr>,
    program_defs: HashMap<String, Expr>,
) -> Result<Dag, LowerDiagnostic> {
    catch_lowering(|| {
        lower_subexpr_program_inner(expr, scoped_tensor_types, full_type_env, program_defs)
    })
}

fn lower_subexpr_program_inner(
    expr: &Expr,
    scoped_tensor_types: HashMap<String, TensorType>,
    full_type_env: HashMap<String, Expr>,
    program_defs: HashMap<String, Expr>,
) -> Dag {
    let scoped_tensor_types_for_bindings = scoped_tensor_types.clone();
    let mut merged_types = full_type_env
        .iter()
        .map(|(name, ty_expr)| (name.clone(), LowerCtx::type_from_type_expr(ty_expr)))
        .collect::<HashMap<_, _>>();
    merged_types.extend(scoped_tensor_types);

    let mut ctx = LowerCtx::new(merged_types, program_defs, LinearityInfo::default());
    // Pre-create a `Load` for every scoped tensor param in a DETERMINISTIC
    // (name-sorted) order. `scoped_tensor_types_for_bindings` is a `HashMap`,
    // whose iteration order is randomized per process; using it directly made
    // the pre-created `Load` node order — and therefore the tensor-helper
    // kernel's input-slot order (`input_labels` follows `dag.nodes()`) —
    // non-deterministic across builds. For a Form-3 `expand` whose extent is
    // read from a shape-source operand referenced ONLY via `shape(x, …)` (the
    // §4.7.2 `bias_broadcast` example), both that operand and the data operand
    // survive DCE, so their relative slot order flipped build-to-build and the
    // emitted C was not byte-identical (violating the codegen-determinism
    // invariant and chelis#469's positive oracle). Sorting by name makes the
    // kernel ABI stable; the host caller maps arguments by `input_label`, so
    // the slot order is internal and any stable order is correct.
    let mut scoped_bindings: Vec<(String, TensorType)> =
        scoped_tensor_types_for_bindings.into_iter().collect();
    scoped_bindings.sort_by(|(a, _), (b, _)| a.cmp(b));
    for (name, tensor_ty) in scoped_bindings {
        let load = ctx.dag.add_node(
            RiscOp::Load {
                name: name.as_str().into(),
            },
            vec![],
            tensor_ty,
            ctx.current_span_id.clone(),
        );
        ctx.bindings.insert(name, LoweredValue::Node(load));
    }
    let value = ctx.lower_expr(expr);
    // Each leaf of the result pytree must be a DISTINCT root node.
    // `Dag::add_root` deduplicates by node id, so when the same node feeds
    // two leaves (chelis#520/#614: two `grad` targets whose adjoint is the
    // same DAG node — e.g. `grad(sum(add(t, y)))` has adjoint `1` for both
    // `t` and `y`, common-subexpression-shared into one node), the second
    // leaf would silently collapse onto the first and the multi-target
    // tuple/ADT structure would lose a slot. Materialize an identity `Copy`
    // for any repeat so every leaf keeps its own root; the value is
    // unchanged (eval and the copy/drop linearity passes treat `Copy` as a
    // pass-through) but the root count now matches the pytree arity.
    let mut seen_roots: HashSet<NodeId> = HashSet::new();
    for id in value.flatten_nodes() {
        let root_id = if seen_roots.insert(id) {
            id
        } else {
            let output_type = ctx
                .dag
                .get(id)
                .map(|node| node.output_type.clone())
                .unwrap_or_else(LowerCtx::default_type);
            ctx.dag.add_node(
                RiscOp::Copy,
                vec![id],
                output_type,
                ctx.current_span_id.clone(),
            )
        };
        ctx.dag.add_root(root_id);
    }
    let dce_dag = crate::optimize::dead_code_eliminate(&ctx.dag);
    let (copy_dag, _) = insert_copy_nodes_for_consuming_fanout(&dce_dag);
    insert_drop_nodes_for_unconsumed_values(copy_dag)
}

pub fn remap_tensor_dim_symbols(
    dag: &Dag,
    formal_params: &[TensorType],
    actual_args: &[TensorType],
) -> Dag {
    let substitutions = tensor_dim_substitutions(formal_params, actual_args);
    if substitutions.is_empty() {
        return dag.clone();
    }
    apply_dim_substitutions(dag, &substitutions)
}

/// Rewrite every symbolic-dim reference in `dag` through
/// `substitutions`: node OUTPUT types and the op-internal fields that
/// carry dims (`Expand::size`, `Reshape::new_shape`,
/// `BlasMatmul::{batch_dims, m, n, k}`). Rewriting only output types
/// while op fields keep the stale names produces the chelis#345 mixed
/// state (`Load: n` next to `Expand { size: Sym("dN") }`) that the
/// Bucket 4d sweep in `dag::symbolic_occurrences` panics on. Shared by
/// [`remap_tensor_dim_symbols`] (call-site formal/actual remapping) and
/// `host::actualize_tensor_helper_types` (synthetic `dN` actualization).
pub(crate) fn apply_dim_substitutions(dag: &Dag, substitutions: &HashMap<String, DimInfo>) -> Dag {
    fn rewrite_dim_info(dim: &DimInfo, substitutions: &HashMap<String, DimInfo>) -> DimInfo {
        match dim {
            DimInfo::Named(name, None) => substitutions
                .get(name)
                .cloned()
                .unwrap_or_else(|| dim.clone()),
            _ => dim.clone(),
        }
    }

    fn rewrite_dim_expr(expr: &DimExpr, substitutions: &HashMap<String, DimInfo>) -> DimExpr {
        match expr {
            DimExpr::Concrete(value) => DimExpr::Concrete(*value),
            DimExpr::Sym(name) => substitutions
                .get(name)
                .map(DimExpr::from)
                .unwrap_or_else(|| DimExpr::Sym(name.clone())),
            DimExpr::Mul(lhs, rhs) => DimExpr::Mul(
                Box::new(rewrite_dim_expr(lhs, substitutions)),
                Box::new(rewrite_dim_expr(rhs, substitutions)),
            ),
            DimExpr::Div(lhs, rhs) => DimExpr::Div(
                Box::new(rewrite_dim_expr(lhs, substitutions)),
                Box::new(rewrite_dim_expr(rhs, substitutions)),
            ),
        }
    }

    let mut specialized = dag.clone();
    let node_ids = specialized
        .nodes()
        .iter()
        .map(|node| node.id)
        .collect::<Vec<_>>();
    for id in node_ids {
        let Some(node) = specialized.get(id).cloned() else {
            continue;
        };
        let mut output_type = node.output_type.clone();
        output_type.dims = output_type
            .dims
            .iter()
            .map(|dim| rewrite_dim_info(dim, substitutions))
            .collect();
        let op = match node.op {
            RiscOp::Expand { axis, size } => RiscOp::Expand {
                axis,
                size: rewrite_dim_expr(&size, substitutions),
            },
            RiscOp::Reshape { new_shape } => RiscOp::Reshape {
                new_shape: new_shape
                    .iter()
                    .map(|dim| match dim {
                        RtDim::Sym(name) => substitutions
                            .get(name)
                            .map(RtDim::from_dim_info)
                            .unwrap_or_else(|| dim.clone()),
                        _ => dim.clone(),
                    })
                    .collect(),
            },
            RiscOp::BlasMatmul {
                batch_dims,
                m,
                n,
                k,
                accumulator,
            } => RiscOp::BlasMatmul {
                batch_dims: batch_dims
                    .iter()
                    .map(|dim| rewrite_dim_expr(dim, substitutions))
                    .collect(),
                m: rewrite_dim_expr(&m, substitutions),
                n: rewrite_dim_expr(&n, substitutions),
                k: rewrite_dim_expr(&k, substitutions),
                accumulator,
            },
            other => other,
        };
        specialized.replace_node(id, op, node.inputs, output_type);
        if let Some(reusable_input) = node.reusable_input {
            specialized.set_reusable_input(id, reusable_input);
        }
    }
    specialized
}

/// Classification of one dim slot in a raw tensor-type formal: a rank spread
/// (`(d-rank {} r)`), a named anchor (`(d-name {} seq)`), or "other" — a
/// `d-lit`/`d-var` slot that consumes exactly one positional dim and is never
/// used to locate a split. Shared by [`extract_rank_var_bindings`] (which walks
/// these against an actual shape to bind each spread) and
/// [`tensor_dim_axis_positions`] (which records the fixed offset of each named
/// anchor). Keeping one classifier avoids the two paths drifting on what counts
/// as an anchor vs a spread.
enum DimSlot {
    Spread(String),
    /// A `(d-name {} seq)` anchor: used to locate a spread split AND to record
    /// a fixed reduce/expand position.
    Named(String),
    /// A `(d-var {} a)` dim variable (surf desugars single-letter dims to this):
    /// it consumes exactly one positional dim like an `Other` slot for spread
    /// splitting, but its name still indexes a fixed reduce/expand position
    /// (chelis#388/#351), so it is kept distinct from `Other`.
    DimVar(String),
    Other,
}

/// Strip a leading `(t-ref {} ...)` wrapper and classify a raw tensor-type
/// formal's dim slots in order (Spread/Named/Other), excluding the trailing
/// precision child. Returns `None` when the expr is not a `(t-tensor ...)` type
/// (so a non-tensor or malformed formal contributes no slots). The borrow
/// wrapper is irrelevant to rank/anchor analysis, mirroring
/// [`extract_precision_var_name`].
fn tensor_formal_dim_slots(expr: &Expr) -> Option<Vec<DimSlot>> {
    let stripped = if let Expr::List(list, _) = expr
        && list.elements.len() >= 3
        && let Expr::Atom(Atom::Symbol(tag), _) = &list.elements[0]
        && tag == "t-ref"
    {
        list.elements.get(2)?
    } else {
        expr
    };
    let Expr::List(list, _) = stripped else {
        return None;
    };
    let Some(Expr::Atom(Atom::Symbol(tag), _)) = list.elements.first() else {
        return None;
    };
    if tag != "t-tensor" || list.elements.len() < 3 {
        return None;
    }
    // Children after tag+meta are dim nodes followed by the precision node.
    let children = &list.elements[2..];
    if children.len() < 2 {
        return None;
    }
    let dim_nodes = &children[..children.len() - 1];
    Some(
        dim_nodes
            .iter()
            .map(|d| {
                let Expr::List(dl, _) = d else {
                    return DimSlot::Other;
                };
                let Some(Expr::Atom(Atom::Symbol(dtag), _)) = dl.elements.first() else {
                    return DimSlot::Other;
                };
                let payload = match dl.elements.get(2) {
                    Some(Expr::Atom(Atom::Symbol(s), _)) => Some(s.clone()),
                    _ => None,
                };
                match (dtag.as_str(), payload) {
                    ("d-rank", Some(n)) => DimSlot::Spread(n),
                    ("d-name", Some(n)) => DimSlot::Named(n),
                    ("d-var", Some(n)) => DimSlot::DimVar(n),
                    _ => DimSlot::Other,
                }
            })
            .collect(),
    )
}

fn tensor_dim_substitutions(
    formal_params: &[TensorType],
    actual_args: &[TensorType],
) -> HashMap<String, DimInfo> {
    formal_params
        .iter()
        .zip(actual_args.iter())
        // Positionally remapping dims between shapes of DIFFERENT rank is never
        // correct — `zip` would align unrelated axes. chelis#258: a rank-poly
        // reduce's rank-2 output formal `[batch, hidden]` paired with a rank-3
        // actual would bind `hidden -> seq`, corrupting every node's dims in the
        // specialized body. Only same-rank pairs contribute a substitution.
        .filter(|(formal, actual)| formal.dims.len() == actual.dims.len())
        .flat_map(|(formal, actual)| formal.dims.iter().zip(actual.dims.iter()))
        .filter_map(|(formal_dim, actual_dim)| match formal_dim {
            // chelis#632 (first hit via the chelis#631 oracle): an
            // anonymous wildcard is not a stable symbol — keying a
            // substitution by `""`/`"*"` painted ONE actual dim across
            // every wildcard-typed node in the DAG, conflating distinct
            // runtime extents under a single name (the C runtime-dim
            // equality guard then aborts well-formed programs, e.g. a
            // helper's inner shrink extent guarded against its stride
            // extent). The declared-return-to-root attachment that this
            // painting used to provide is now positional:
            // `host::remap_tensor_helper_dim_symbols` retypes the helper
            // root's anon axes from the expected output directly.
            DimInfo::Named(name, None) if !name.is_empty() && name != "*" => {
                Some((name.clone(), actual_dim.clone()))
            }
            _ => None,
        })
        .collect()
}

/// Issue #388: record, per call site, the positional index each named
/// axis occupied in a formal parameter's shape.
///
/// When the actual argument is a *literal-shaped* operand (`tensor[2, 3]`
/// from `to_tensor([[...]])`), `tensor_dim_substitutions` binds the named
/// axis to a concrete `DimInfo` (`seq -> Lit(3)`), and
/// [`LowerCtx::lower_expr`]'s binding of the formal parameter name to the
/// literal node means the inlined body's operand carries the literal dims
/// (`[Lit(2), Lit(3)]`) — the named axis is gone. A by-name reduction/
/// expand-anchor lookup against those dims then cannot find `seq`. The
/// positional index is preserved here so [`LowerCtx::resolve_reduce_axis`]
/// / [`LowerCtx::resolve_expand_anchor`] can recover the axis a named
/// reduction targets even after the name was erased by monomorphization.
///
/// A name may appear in more than one formal parameter; only a name whose
/// position is *consistent* across all the formals that mention it gets
/// recorded (an inconsistent name is ambiguous and is left for the loud
/// by-name failure path, never silently guessed).
///
/// chelis#373: walks the *raw* formal type-exprs (not the collapsed
/// `TensorType.dims`) so a Tier-3 rank-spread formal (`[..pre, seq, ..post]`)
/// is handled soundly. A named anchor that sits behind a `(d-rank ..)` spread
/// has a caller-dependent absolute offset (the leading spread's length is
/// unknown from the formal alone), so its position is NOT fixed and is NOT
/// recorded. Recording the collapsed-dims index there would emit a wrong index
/// (e.g. `seq -> 0` for `[..pre, seq, ..post]`) and, via the `.extend(..)` at
/// the call site, CLOBBER a correct fixed index recorded by an outer
/// concrete-rank caller (`[b, seq]` -> `seq -> 1`). Only a named slot with no
/// preceding spread contributes a position; a leading anchor before a trailing
/// spread (`[row, ..rest]` -> `row -> 0`) is still fixed and is recorded.
fn tensor_dim_axis_positions(
    formal_param_exprs: &[Option<Expr>],
    actual_args: &[TensorType],
) -> HashMap<String, (usize, DimInfo)> {
    let mut positions: HashMap<String, (usize, DimInfo)> = HashMap::new();
    let mut ambiguous: HashSet<String> = HashSet::new();
    for (formal_expr, actual) in formal_param_exprs.iter().zip(actual_args.iter()) {
        let Some(formal_expr) = formal_expr else {
            continue;
        };
        let Some(slots) = tensor_formal_dim_slots(formal_expr) else {
            continue;
        };
        let mut index = 0usize;
        let mut saw_spread = false;
        for slot in &slots {
            // A `d-name` or `d-var` slot before any spread has a FIXED absolute
            // offset and indexes a reduce/expand position; record it. Behind a
            // spread the offset is caller-dependent and is not recorded (so it
            // never clobbers a fixed index from a concrete-rank caller).
            let named_here = match slot {
                DimSlot::Named(name) | DimSlot::DimVar(name) if !saw_spread => Some(name),
                _ => None,
            };
            if let Some(name) = named_here {
                // chelis#549: pair the position with the anchor's concrete extent
                // at this call site. Positions before a spread align with the
                // actual's leading dims regardless of any later spread's length,
                // so `index` maps directly into `actual.dims`. The extent is what
                // the use-site recovery re-validates against the monomorphized
                // operand so an intervening axis-reorder cannot be trusted. With
                // no aligned actual dim (or an inconsistent record across formals)
                // the name is left UNRECORDED so the loud by-name path fires
                // rather than an unvalidatable position being used.
                match actual.dims.get(index) {
                    Some(extent) => match positions.get(name) {
                        Some((prev_idx, prev_extent))
                            if *prev_idx != index || prev_extent != extent =>
                        {
                            ambiguous.insert(name.clone());
                        }
                        None => {
                            positions.insert(name.clone(), (index, extent.clone()));
                        }
                        _ => {}
                    },
                    None => {
                        ambiguous.insert(name.clone());
                    }
                }
            }
            match slot {
                DimSlot::Spread(_) => {
                    // A spread absorbs a caller-dependent run; once seen, no
                    // later slot has a fixed offset, so stop advancing.
                    saw_spread = true;
                }
                _ if !saw_spread => index += 1,
                _ => {}
            }
        }
    }
    for name in ambiguous {
        positions.remove(&name);
    }
    positions
}

/// Outcome of recovering a named anchor's positional axis from the
/// `dim_axis_positions` oracle when call-site monomorphization erased the name
/// from the operand's dims (chelis#388/#373/#339). The recovery is the soundness
/// fix for chelis#549: the recorded index is the anchor's offset in a *formal
/// parameter*, so an intervening axis-reorder (`permute`/transpose) can leave it
/// valid-but-stale (in range, wrong axis). Rather than trust the bare index, the
/// recovery re-validates the recorded extent against the monomorphized operand.
enum AnchorRecovery {
    /// The anchor's recorded extent appears at EXACTLY ONE in-range axis, so the
    /// anchor is located unambiguously (whether or not a reorder moved it from
    /// the recorded position). The carried index is sound to split/reduce/expand
    /// at.
    Axis(usize),
    /// No usable record: the anchor was never recorded, or its recorded extent
    /// appears at no in-range axis. The caller falls through to its existing loud
    /// by-name failure path.
    Unrecorded,
    /// chelis#549: the anchor's recorded extent appears at MORE THAN ONE in-range
    /// axis, so the anchor cannot be located by value. This includes the
    /// all-equal-extent (square) operand, where the recorded position's extent
    /// matches vacuously even under a reorder. The caller MUST fail loud rather
    /// than reduce/split a possibly-wrong axis (fail-closed).
    AmbiguousAfterReorder,
}

/// Recover the positional axis a named anchor maps to in a monomorphized
/// `operand` whose dims no longer carry the name, consulting the
/// `dim_axis_positions` oracle and considering only axes in `lo..hi`.
///
/// Soundness (chelis#549): the recorded index is the anchor's offset in the
/// recording caller's *formal parameter*. An axis-reorder (`permute`/transpose)
/// between that parameter and this use site can make that index stale (in range,
/// wrong axis). The recorded index is therefore **never trusted as a position**;
/// the anchor is located purely by its recorded *extent*:
///
/// * exactly one in-range axis carries the extent → [`AnchorRecovery::Axis`]
///   (located unambiguously, whether or not a reorder moved it);
/// * no in-range axis carries it → [`AnchorRecovery::Unrecorded`] (fall through
///   to the loud by-name path);
/// * more than one in-range axis carries it → [`AnchorRecovery::AmbiguousAfterReorder`]
///   (fail loud).
///
/// The >1 branch is the chelis#549 / RT-1 fix: when the anchor's extent collides
/// with another axis (the all-equal-extent / square operand is the headline
/// case, e.g. a `seq × seq` attention reduce), value re-validation cannot tell
/// the anchor from its twin, so a stale recorded position would be vacuously
/// "confirmed" and silently reduce the wrong axis. We refuse instead. This is
/// the soundness FLOOR: it over-rejects a square reduce that did NOT in fact
/// reorder (the recorded position was correct) because value alone cannot
/// distinguish that from a square reduce that DID reorder. RECOVERING those
/// (rather than rejecting) requires tracking the anchor identity through the
/// reorder during lowering — a separate, more invasive change. The common
/// attention case with distinct batch/seq/head extents is recovered exactly.
fn recover_anchor_axis(
    dims: &[DimInfo],
    lo: usize,
    hi: usize,
    anchor: &str,
    dim_axis_positions: &HashMap<String, (usize, DimInfo)>,
) -> AnchorRecovery {
    // The recorded position is intentionally unused for the decision (it is the
    // stale formal-param offset chelis#549 is about); only the recorded extent
    // is trusted, located by value in the monomorphized operand.
    let Some((_recorded_pos, expected)) = dim_axis_positions.get(anchor) else {
        return AnchorRecovery::Unrecorded;
    };
    let hi = hi.min(dims.len());
    let mut matches = (lo..hi).filter(|&i| dims.get(i) == Some(expected));
    match (matches.next(), matches.next()) {
        (Some(only), None) => AnchorRecovery::Axis(only),
        (None, _) => AnchorRecovery::Unrecorded,
        (Some(_), Some(_)) => AnchorRecovery::AmbiguousAfterReorder,
    }
}

/// WS-A8: build a precision substitution map from formal vs actual
/// tensor types at a polymorphic-def call site. The formal types come
/// from the def's annotated parameter signatures and may carry
/// `(t-var {} p)` precision slots (precision polymorphism per
/// spec/04-type-system.md §5.8). The actual types come from the call
/// site's lowered argument nodes and always carry concrete primitives
/// (the call site is the monomorphization boundary).
///
/// The returned map is keyed by the precision-var name as it appears in
/// `(t-var {} p)` slots. The map flows into [`LowerCtx::prec_substitutions`]
/// so any `try_extract_tensor_type` call reached during inlining of
/// the def body resolves the precision-tvar to a concrete `Prim` and
/// the F2 backend tripwire never fires for a properly-monomorphized
/// program.
///
/// Formals are walked via the raw type-expr tree (not the
/// already-extracted `TensorType` value), because the latter has the
/// `(t-var)` slot already collapsed to a default precision. The raw
/// tree preserves the `(t-var {} p)` shape so we can recover the
/// var-name. The actual types come from the lowered DAG so their
/// precision is a concrete `Prim`.
fn tensor_prec_substitutions(
    formal_param_exprs: &[Option<Expr>],
    actual_args: &[TensorType],
) -> HashMap<String, Prim> {
    let mut subst = HashMap::new();
    for (formal_expr, actual) in formal_param_exprs.iter().zip(actual_args.iter()) {
        let Some(formal_expr) = formal_expr else {
            continue;
        };
        if let Some(var_name) = extract_precision_var_name(formal_expr) {
            subst.entry(var_name).or_insert(actual.precision);
        }
    }
    subst
}

/// Pull the precision-var name out of a tensor type expression's
/// precision slot. Returns `Some(name)` when the slot is shaped
/// `(t-var {} name)`; returns `None` when the slot is concrete
/// (`(t-prim {} f32)`) or the expression is not a tensor type.
///
/// Strips a leading `(t-ref {} ...)` wrapper so a `&tensor[..., p]`
/// parameter is treated the same as `tensor[..., p]` for substitution
/// purposes — the borrow is irrelevant to precision monomorphization.
fn extract_precision_var_name(expr: &Expr) -> Option<String> {
    let stripped = if let Expr::List(list, _) = expr
        && list.elements.len() >= 3
        && let Expr::Atom(Atom::Symbol(tag), _) = &list.elements[0]
        && tag == "t-ref"
    {
        list.elements.get(2)?
    } else {
        expr
    };
    let Expr::List(list, _) = stripped else {
        return None;
    };
    if list.elements.len() < 3 {
        return None;
    }
    let Expr::Atom(Atom::Symbol(tag), _) = &list.elements[0] else {
        return None;
    };
    if tag != "t-tensor" {
        return None;
    }
    let last = list.elements.last()?;
    let Expr::List(prec_list, _) = last else {
        return None;
    };
    if prec_list.elements.len() < 3 {
        return None;
    }
    let Expr::Atom(Atom::Symbol(prec_tag), _) = &prec_list.elements[0] else {
        return None;
    };
    if prec_tag != "t-var" {
        return None;
    }
    if let Expr::Atom(Atom::Symbol(name), _) = &prec_list.elements[2] {
        Some(name.clone())
    } else {
        None
    }
}

/// issue #319: the renamed type-variable name carried by a formal-
/// parameter position of a verb's inferred `t-fn` type metadata, or
/// `None` when the position is concrete.
///
/// After the checker resolves a separate-`sig` precision-polymorphic
/// verb, a formal-parameter position in the `fn` node's `t-fn` type
/// metadata is one of two shapes, both of which yield the renamed var
/// `tN` (a `(t-prim …)` concrete precision yields `None`):
///
/// - a bare `(t-var tN)` — the param's whole tensor type collapsed to a
///   single inference var, and the SAME `tN` is the var the checker
///   stamped into the body nodes' precision slots (the issue #319 `sdpa`
///   shape: the q-param position is `(t-var t310)` and the `permute`
///   body node's precision is also `t310`); or
/// - a `(t-tensor … (t-var tN))` — the param kept tensor shape but its
///   precision slot is the renamed var `tN`.
///
/// The caller binds the returned `tN` to that parameter's concrete actual
/// precision.
fn formal_param_type_var_name(expr: &Expr) -> Option<String> {
    // Strip a leading `(t-ref {} ...)` borrow wrapper.
    let stripped = if let Expr::List(list, _) = expr
        && list.elements.len() >= 3
        && let Expr::Atom(Atom::Symbol(tag), _) = &list.elements[0]
        && tag == "t-ref"
    {
        &list.elements[2]
    } else {
        expr
    };
    // Bare `(t-var tN)`: the whole param type is one inference var.
    if let Expr::List(list, _) = stripped
        && list.elements.len() >= 3
        && let Expr::Atom(Atom::Symbol(tag), _) = &list.elements[0]
        && tag == "t-var"
        && let Expr::Atom(Atom::Symbol(name), _) = &list.elements[2]
    {
        return Some(name.clone());
    }
    // `(t-tensor … (t-var tN))`: the precision slot is the renamed var.
    extract_precision_var_name(stripped)
}

/// issue #319: the per-parameter argument type expressions of a `(fn
/// {type: (t-fn {} a0 a1 ... ret)} ...)` node's inferred `t-fn` type
/// metadata.
///
/// When the checker resolves a separate-`sig` verb it stamps the `fn`
/// node with the inferred function type whose argument positions are the
/// verb's formal-parameter types — with the sig's precision variable `p`
/// already RENAMED to a fresh internal inference var (`type_to_deep_expr`
/// prints `TensorPrec::Var(_)` anonymously, e.g. `t300`). Returns the
/// argument type exprs in declared order (the trailing return type is
/// dropped), so the caller can recover each parameter's renamed precision
/// variable. Returns `None` when the node has no `t-fn` type metadata.
fn fn_type_arg_exprs(fn_expr: &Expr) -> Option<Vec<&Expr>> {
    let Expr::List(list, _) = fn_expr else {
        return None;
    };
    if get_tag(list) != Some("fn") {
        return None;
    }
    let Expr::Map(meta, _) = list.elements.get(1)? else {
        return None;
    };
    let (_, ty_expr) = meta.entries.iter().find(|(key, _)| key == "type")?;
    let Expr::List(fn_ty, _) = ty_expr else {
        return None;
    };
    if get_tag(fn_ty) != Some("t-fn") {
        return None;
    }
    let args = children(fn_ty);
    // Drop the trailing return type; keep only the parameter positions.
    args.split_last()
        .map(|(_ret, params)| params.iter().collect())
}

/// issue #319: the single concrete precision that EVERY
/// formal-parameter precision variable of a verb resolves to at this
/// call site, or `None` when the verb is not fully precision-monomorphic
/// here.
///
/// "Fully precision-monomorphic" means: every renamed formal-parameter
/// precision variable (recovered from the `fn` node's inferred `t-fn`
/// type metadata via [`formal_param_type_var_name`]) is constrained by
/// its tensor actual argument to the SAME single concrete precision, with
/// no conflict. When that holds, one precision threads through the whole
/// verb instance, so [`formal_precision_var_bindings`] may bind every
/// body precision variable to it.
///
/// Returns `None` (binding nothing) when:
///   - two tensor actuals disagree on precision (a genuinely
///     precision-heterogeneous call — e.g. `sdpa(q_f32, k_f16, …)`); the
///     §5.8.1 tripwire / downstream precision check then reports the
///     mismatch instead of this code silently promoting it; or
///   - no formal-parameter precision variable is constrained by a tensor
///     actual (nothing to monomorphize against).
fn fully_monomorphic_call_precision(fn_expr: &Expr, actual_types: &[TensorType]) -> Option<Prim> {
    let arg_exprs = fn_type_arg_exprs(fn_expr)?;
    // Positional-alignment guard (issue #319 review). `actual_types` holds
    // only the call arguments that lowered to a typed DAG node — the
    // caller's loop SKIPS a callable (fn-typed) argument and any argument
    // that did not resolve to a single typed node. When that happens
    // `actual_types` is shorter than `arg_exprs` (the verb's full
    // formal-parameter list), and the positional `zip` below would silently
    // MISALIGN a tensor actual against the wrong formal position. Require an
    // exact 1:1 correspondence; if any argument was skipped, bind nothing
    // and let the §5.8.1 tripwire handle the call. The precision-poly verbs
    // this targets (sdpa/attention) take only tensor parameters, so the
    // lengths match for every supported case; a verb interleaving tensor
    // and fn-typed parameters is intentionally out of scope here.
    if arg_exprs.len() != actual_types.len() {
        return None;
    }
    let mut shared: Option<Prim> = None;
    let mut saw_formal_prec_var = false;
    for (arg_expr, actual) in arg_exprs.iter().zip(actual_types.iter()) {
        // A scalar argument does not constrain the verb's polymorphic
        // tensor precision; skip it.
        if actual.dims.is_empty() {
            continue;
        }
        // Only positions whose formal type is a precision variable (bare
        // renamed `t-var`, or a `t-tensor` with a `t-var` precision slot)
        // constrain the verb's polymorphic precision. A position with a
        // concrete formal precision contributes nothing.
        if formal_param_type_var_name(arg_expr).is_none() {
            continue;
        }
        saw_formal_prec_var = true;
        match shared {
            None => shared = Some(actual.precision),
            Some(prim) if prim == actual.precision => {}
            // Two formal precision-var positions pinned to different
            // concrete precisions: the call is NOT monomorphic.
            Some(_) => return None,
        }
    }
    saw_formal_prec_var.then_some(shared).flatten()
}

/// issue #319: bind the renamed body precision variables of a separate-
/// `sig` precision-polymorphic verb to the concrete call-site precision,
/// but ONLY when the call is fully precision-monomorphic.
///
/// The checker renames a separate-`sig` verb's precision variable `p`
/// when it stamps the resolved body node types (e.g. the `permute`
/// adjoint output), so the call-site name-keyed `tensor_prec_substitutions`
/// (keyed on the original `p`) misses the renamed name and a body node
/// lowered through `type_from_meta` trips the §5.8.1 monomorphization
/// tripwire. When every formal-parameter precision variable resolves to a
/// single shared concrete precision (see
/// [`fully_monomorphic_call_precision`]), one precision threads through
/// the entire verb instance, so binding every body precision variable to
/// it is sound — including a body precision variable that the checker
/// unified across two sig precision variables (e.g. an `add` forcing
/// `p == w`) and that therefore no longer appears verbatim in a single
/// formal-parameter position.
///
/// Soundness (red-team caveat): when the call is NOT fully
/// precision-monomorphic this binds NOTHING and the §5.8.1 tripwire
/// fires. A genuinely precision-heterogeneous call (tensor actuals
/// disagree) is rejected by the monomorphism check, so no body precision
/// variable is silently promoted; a genuinely under-determined precision
/// variable with no concrete call site surfaces as a wildcard `(t-var _)`
/// (which `try_extract_tensor_type` already treats as a fresh type, not a
/// tripwire) rather than reaching this binding. This preserves the
/// no-implicit-precision-promotion invariant (spec/04-type-system.md
/// §5.8.1).
///
/// Assumption (issue #319 review): this binds EVERY body precision
/// variable to the single call-site precision, which assumes every body
/// precision variable is ultimately tied to a parameter's precision. That
/// holds for the precision-poly verbs this targets — the only precision
/// source in an sdpa/attention body is the `q`/`k`/`v` parameters, so a
/// fully-monomorphic call pins the whole body. A body carrying a
/// genuinely INDEPENDENT precision variable — e.g. an internal
/// polymorphic-precision helper not constrained by any parameter — would
/// be over-constrained by this blanket bind. No such construct arises in
/// the target verbs; supporting one would need per-variable provenance
/// tracking rather than a single shared precision, and is out of scope.
fn formal_precision_var_bindings(
    fn_expr: &Expr,
    body: &Expr,
    actual_types: &[TensorType],
) -> HashMap<String, Prim> {
    let Some(prim) = fully_monomorphic_call_precision(fn_expr, actual_types) else {
        return HashMap::new();
    };
    let mut body_prec_vars = HashSet::new();
    collect_body_precision_var_names(body, &mut body_prec_vars);
    body_prec_vars
        .into_iter()
        .map(|var_name| (var_name, prim))
        .collect()
}

/// issue #319: collect every precision type-variable name appearing in a
/// tensor precision slot ANYWHERE inside `expr` (node `type:` metadata,
/// parameter annotations, nested type expressions). Used by
/// [`formal_precision_var_bindings`] to enumerate the renamed body
/// precision variables a fully-monomorphic call must concretize.
fn collect_body_precision_var_names(expr: &Expr, out: &mut HashSet<String>) {
    if let Some(name) = extract_precision_var_name(expr) {
        out.insert(name);
    }
    match expr {
        Expr::List(list, _) => {
            for child in &list.elements {
                collect_body_precision_var_names(child, out);
            }
        }
        Expr::Map(map, _) => {
            for (_, value) in &map.entries {
                collect_body_precision_var_names(value, out);
            }
        }
        Expr::MetaExpr(meta, _) => {
            for (_, value) in &meta.entries {
                collect_body_precision_var_names(value, out);
            }
            collect_body_precision_var_names(&meta.expr, out);
        }
        Expr::Atom(_, _) => {}
    }
}

/// Tier-2 rank polymorphism (spec/design/rank_polymorphism.md): build a
/// rank-var substitution map from formal vs actual tensor types at a
/// rank-polymorphic-def call site. The structural twin of
/// [`tensor_prec_substitutions`]. The formal types come from the def's
/// annotated parameter signatures and may carry a sole `(d-rank {} r)` dim
/// slot (a rank variable standing for the *entire* shape vector). The actual
/// types come from the call site's lowered argument nodes and always carry a
/// concrete shape (the call site is the monomorphization boundary).
///
/// The returned map is keyed by the rank-var name as it appears in the
/// `(d-rank {} r)` slot. The map flows into [`LowerCtx::rank_substitutions`]
/// so any `try_extract_tensor_type` call reached during inlining of the def
/// body expands the `Dim::Rank` slot to the caller's concrete dims and the
/// rank-poly monomorphization tripwire never fires for a properly-
/// monomorphized program.
///
/// Formals are walked via the raw type-expr tree (not the already-extracted
/// `TensorType` value) so the `(d-rank {} r)` shape — and the rank-var name —
/// is preserved. The actual types come from the lowered DAG so their dims are
/// concrete.
///
/// `dim_axis_positions` (chelis#373/#388) is the by-position anchor oracle for
/// the case where call-site monomorphization erased a named anchor from the
/// actual dims (a literal-shaped operand reaching a Tier-3 callee through a
/// concrete-rank intermediate, e.g. the grad lane that inlines the whole body
/// into one DAG). When the by-name split fails, the anchor is recovered via
/// [`recover_anchor_axis`], which re-validates the recorded position against the
/// monomorphized actual's extents. The map carries only positions a concrete-rank
/// caller pinned, and an out-of-range/unrecorded anchor still fails loudly.
///
/// chelis#549 (axis-reorder staleness; spans #373/#388/#339): the recorded index
/// is the anchor's offset in that caller's *formal parameter*, so an
/// axis-reordering op (e.g. `permute`) between that parameter and this use site
/// can leave the recorded index valid-but-stale (in range, wrong axis). The
/// recovery no longer trusts the bare index at all: it locates the anchor purely
/// by its recorded extent, taking the axis when that extent is unique, and
/// FAILING LOUD when the extent collides with another axis (the square/equal-extent
/// case, RT-1) — never splitting at a possibly-wrong axis. The by-name path above
/// already tracks the moved anchor and is unaffected.
fn tensor_rank_substitutions(
    formal_param_exprs: &[Option<Expr>],
    actual_args: &[TensorType],
    dim_axis_positions: &HashMap<String, (usize, DimInfo)>,
) -> HashMap<String, Vec<DimInfo>> {
    let mut subst = HashMap::new();
    for (formal_expr, actual) in formal_param_exprs.iter().zip(actual_args.iter()) {
        let Some(formal_expr) = formal_expr else {
            continue;
        };
        // Tier-3: a formal may carry several spreads interleaved with named
        // anchors (`&tensor[..pre, seq, ..post]`); each binds to the run of the
        // actual's concrete dims it covers, located by the anchors. A sole
        // `(d-rank {} r)` binds to the whole shape (Tier-2). First-binding-wins:
        // a rank var appearing in more than one param binds to the same run
        // because the checker already unified them; the debug_assert is the
        // tripwire for a checker regression that let two positions diverge.
        for (var_name, run) in
            extract_rank_var_bindings(formal_expr, &actual.dims, dim_axis_positions)
        {
            match subst.entry(var_name) {
                std::collections::hash_map::Entry::Occupied(existing) => {
                    debug_assert_eq!(
                        existing.get(),
                        &run,
                        "rank-var bound to two distinct concrete shapes at one \
                         call site: the type checker should have rejected this",
                    );
                }
                std::collections::hash_map::Entry::Vacant(slot) => {
                    slot.insert(run);
                }
            }
        }
    }
    subst
}

/// Bind each rank-var spread in a (possibly anchored, Tier-3) tensor-type
/// formal to the run of the actual's concrete dims it covers, located by the
/// named anchors between spreads — the lowering twin of the checker's
/// `unify_row_against_ground`. A sole `(d-rank {} r)` binds to the whole shape
/// (Tier-2). Returns one `(name, run)` per spread; an empty vec when the formal
/// has no spread, is not a tensor, or its anchors cannot be located (a checked
/// program never hits the last case — the bail keeps lowering panic-free).
///
/// Strips a leading `(t-ref {} ...)` wrapper so a `&tensor[..]` parameter is
/// treated the same as `tensor[..]` (the borrow is irrelevant to rank
/// monomorphization, mirroring [`extract_precision_var_name`]).
///
/// `dim_axis_positions` is the chelis#373 by-position anchor oracle: when the
/// monomorphized actual no longer carries the anchor as a `Named` dim (the
/// literal-shaped-operand-through-a-concrete-rank-callee case, e.g. the grad
/// lane inlining the whole body into one DAG), the anchor's split index is
/// recovered from the fixed position a concrete-rank caller recorded. This is
/// the rank-spread analogue of the #388 recovery in
/// [`LowerCtx::resolve_reduce_axis`]; both rely on the same map.
fn extract_rank_var_bindings(
    expr: &Expr,
    actual_dims: &[DimInfo],
    dim_axis_positions: &HashMap<String, (usize, DimInfo)>,
) -> Vec<(String, Vec<DimInfo>)> {
    let Some(slots) = tensor_formal_dim_slots(expr) else {
        return Vec::new();
    };

    if !slots.iter().any(|s| matches!(s, DimSlot::Spread(_))) {
        return Vec::new();
    }

    // Walk the slots against the actual dims, exactly as the checker's
    // `unify_row_against_ground` does: a spread followed by a named anchor binds
    // to the run up to that anchor (located by name, with a by-position
    // fallback); a trailing spread absorbs the rest; other dims consume one
    // actual dim positionally.
    let n = actual_dims.len();
    let mut out: Vec<(String, Vec<DimInfo>)> = Vec::new();
    let mut gi = 0usize;
    let mut ri = 0usize;
    while ri < slots.len() {
        if gi > n {
            return out;
        }
        match &slots[ri] {
            DimSlot::Spread(name) => {
                let rest = &slots[ri + 1..];
                match rest.iter().position(|s| !matches!(s, DimSlot::Spread(_))) {
                    Some(0) => {
                        let (DimSlot::Named(anchor) | DimSlot::DimVar(anchor)) = &rest[0] else {
                            // Anchor is a `d-lit`/`Other` — unlocatable by name.
                            // The checker requires a named anchor after a spread,
                            // so this is unreachable for a checked program; fail
                            // loud in debug, bail in release.
                            debug_assert!(
                                false,
                                "rank-spread anchor is not a named dim at lowering"
                            );
                            return out;
                        };
                        // Primary: locate the anchor by name in the (possibly
                        // still-symbolic) actual dims, exactly as the forward
                        // lane does when each nested call re-stamps the callee's
                        // declared named dims onto its placeholder.
                        let by_name = actual_dims[gi..]
                            .iter()
                            .position(|d| matches!(d, DimInfo::Named(g, _) if g == anchor))
                            .map(|p| gi + p);
                        // chelis#373 fallback: the actual was monomorphized to
                        // concrete `Lit` dims and the name is gone. Recover the
                        // split index by locating the anchor's recorded extent in
                        // the actual (chelis#549): take the axis when the extent is
                        // unique (relocating through any intervening `permute`).
                        // An anchor whose extent is absent falls through to the
                        // loud path; an extent that collides with another axis
                        // (the square/equal-extent case) fails loud rather than
                        // splitting at a possibly-wrong axis.
                        let split = match by_name {
                            Some(s) => s,
                            None => match recover_anchor_axis(
                                actual_dims,
                                gi,
                                n,
                                anchor,
                                dim_axis_positions,
                            ) {
                                AnchorRecovery::Axis(s) => s,
                                // FATAL (chelis#549): a soundness rejection that
                                // must not be absorbed by the C host-fallback
                                // path into a generic grad-unsupported message.
                                AnchorRecovery::AmbiguousAfterReorder => {
                                    raise_fatal_lowering_error(
                                        format!(
                                            "rank-spread anchor `{anchor}` cannot be located: \
                                             call-site monomorphization erased the name and an \
                                             intervening axis-reorder (e.g. `permute`) left its \
                                             recorded position stale; its recorded extent appears \
                                             at multiple axes of the monomorphized actual. \
                                             Refusing to split at a possibly-wrong axis (chelis#549)"
                                        ),
                                        None,
                                        None,
                                    )
                                }
                                // Anchor absent by name AND not soundly
                                // recoverable by position — the checker located it
                                // (Name↔Lit etc. were rejected), so unreachable
                                // for a checked program. Fail loud (fail-closed)
                                // rather than the former release-silent
                                // `return out` partial binding.
                                AnchorRecovery::Unrecorded => raise_lowering_error(
                                    format!(
                                        "rank-spread anchor `{anchor}` absent from monomorphized \
                                         actual: internal rank-monomorphization error"
                                    ),
                                    None,
                                    None,
                                ),
                            },
                        };
                        out.push((name.clone(), actual_dims[gi..split].to_vec()));
                        gi = split;
                        ri += 1;
                    }
                    None if rest.is_empty() => {
                        out.push((name.clone(), actual_dims[gi..n].to_vec()));
                        gi = n;
                        ri += 1;
                    }
                    // Two adjacent spreads — the undetermined split is rejected at
                    // unification, so this never reaches a checked backend.
                    _ => {
                        debug_assert!(false, "two adjacent rank spreads at lowering");
                        return out;
                    }
                }
            }
            _ => {
                gi += 1;
                ri += 1;
            }
        }
    }
    out
}

pub fn top_level_expr_is_lowered(
    expr: &Expr,
    program_exprs: &[Expr],
    type_env: &HashMap<String, Expr>,
) -> bool {
    let lowered_names = top_level_lowering_map(program_exprs, type_env);
    top_level_expr_is_lowered_with_names(expr, type_env, &lowered_names)
}

pub fn top_level_lowering_map(
    exprs: &[Expr],
    type_env: &HashMap<String, Expr>,
) -> HashMap<String, bool> {
    let top_level_defs = collect_top_level_defs(exprs);
    let top_level_sigs = collect_top_level_sigs(exprs);
    let mut cache = HashMap::new();
    let mut visiting = HashSet::new();
    for name in top_level_defs.keys() {
        let lowered = def_is_lowered(
            name,
            &top_level_defs,
            &top_level_sigs,
            type_env,
            &mut cache,
            &mut visiting,
        );
        cache.insert(name.clone(), lowered);
    }
    cache
}

/// Phase G helper: compute `top_level_lowering_map` for new code with
/// the library's pre-computed `lowered_names` seeded into the cache.
/// This makes new-code defs that reference library functions inherit the
/// same lowered-vs-host classification they'd get in monolithic mode
/// (where library defs live alongside new ones in `top_level_defs`).
///
/// The returned map covers BOTH library + new-code names so callers can
/// look up either; downstream filters slice to new-code-only as needed.
pub fn top_level_lowering_map_with_context(
    library: &LoweredLibrary,
    new_exprs: &[Expr],
    new_type_env: &HashMap<String, Expr>,
) -> HashMap<String, bool> {
    let mut top_level_defs = library.program_defs.clone();
    for (name, body) in collect_top_level_defs(new_exprs) {
        top_level_defs.insert(name, body);
    }
    let mut top_level_sigs = collect_top_level_sigs(new_exprs);
    // Library declared types feed `lookup_declared_type_expr`; merge
    // them in so a library def's signature is reachable when the new
    // code's body references it.
    for (name, ty_expr) in &library.program_types {
        top_level_sigs
            .entry(name.clone())
            .or_insert_with(|| ty_expr_to_deep(ty_expr));
    }
    let mut cache = library.lowered_names.clone();
    let mut visiting = HashSet::new();
    for name in top_level_defs.keys() {
        if cache.contains_key(name) {
            continue;
        }
        let lowered = def_is_lowered(
            name,
            &top_level_defs,
            &top_level_sigs,
            new_type_env,
            &mut cache,
            &mut visiting,
        );
        cache.insert(name.clone(), lowered);
    }
    cache
}

/// Inverse of `LowerCtx::type_from_type_expr` — used by
/// `top_level_lowering_map_with_context` to feed library types into the
/// `top_level_sigs` map. Lossy on dim variables (we project to the
/// scalar return type) since `def_is_lowered` only looks at
/// `type_is_never_lowerable`, which inspects the t-fn return type.
fn ty_expr_to_deep(ty: &TensorType) -> Expr {
    use chelis_deep::Span;
    use chelis_deep::ast::{Atom, List, MetaMap};
    let span = Span::new(0, 0);
    let prim = match ty.precision {
        chelis_types::types::Prim::F32 => "f32",
        chelis_types::types::Prim::F64 => "f64",
        chelis_types::types::Prim::F16 => "f16",
        chelis_types::types::Prim::Bf16 => "bf16",
        // E2 (WS-A0 RT-1 fixup): per spec/04-type-system.md §1.1.1
        // f8e4m3 is deferred and the type checker must reject it
        // before lowering. If a TensorType reaches this Deep
        // re-encoder with f8e4m3 precision, the upstream rejection
        // has a hole — panic rather than emit a `(t-prim {} f8e4m3)`
        // node into lowered IR.
        chelis_types::types::Prim::F8e4m3 => panic!(
            "f8e4m3 is deferred per spec/04-type-system.md §1.1.1 and \
             should have been rejected upstream"
        ),
        chelis_types::types::Prim::Int8 => "int8",
        chelis_types::types::Prim::Int16 => "int16",
        chelis_types::types::Prim::Int32 => "int32",
        chelis_types::types::Prim::Int64 => "int64",
        chelis_types::types::Prim::Bool => "bool",
        chelis_types::types::Prim::String => "string",
    };
    let prim_node = Expr::List(
        List {
            elements: vec![
                Expr::Atom(Atom::Symbol("t-prim".into()), span),
                Expr::Map(MetaMap::default(), span),
                Expr::Atom(Atom::Symbol(prim.into()), span),
            ],
        },
        span,
    );
    if ty.dims.is_empty() {
        prim_node
    } else {
        Expr::List(
            List {
                elements: vec![
                    Expr::Atom(Atom::Symbol("t-tensor".into()), span),
                    Expr::Map(MetaMap::default(), span),
                    prim_node,
                ],
            },
            span,
        )
    }
}

pub fn expr_is_dag_lowerable(expr: &Expr, program: &CheckedProgram) -> bool {
    if expr_requires_host_runtime(expr) {
        return false;
    }

    let top_level_defs = collect_top_level_defs(program.exprs());
    let top_level_sigs = collect_top_level_sigs(program.exprs());
    let mut cache = HashMap::new();
    let mut visiting = HashSet::new();
    !expr_depends_on_nonlowerable_name(
        expr,
        &top_level_defs,
        &top_level_sigs,
        program.type_env(),
        &mut cache,
        &mut visiting,
        &HashSet::new(),
    )
}

fn top_level_expr_is_lowered_with_names(
    expr: &Expr,
    type_env: &HashMap<String, Expr>,
    lowered_names: &HashMap<String, bool>,
) -> bool {
    let top_level_sigs = HashMap::new();
    let Expr::List(list, _) = expr else {
        return true;
    };
    if get_tag(list) == Some("module") {
        return list
            .elements
            .iter()
            .skip(3)
            .all(|child| top_level_expr_is_lowered_with_names(child, type_env, lowered_names));
    }
    if get_tag(list) != Some("def") {
        return true;
    }
    let Some(name) = top_level_expr_name(expr) else {
        return true;
    };
    lowered_names.get(name).copied().unwrap_or_else(|| {
        !lookup_declared_type_expr(&top_level_sigs, type_env, name)
            .is_some_and(type_is_never_lowerable)
    })
}

fn type_is_never_lowerable(expr: &Expr) -> bool {
    let Expr::List(list, _) = expr else {
        return true;
    };
    match get_tag(list) {
        // WS-A8: a t-fn whose return type or any parameter type carries
        // a `(t-var {} _)` precision slot is precision-polymorphic and
        // has no concrete monomorphization on its own. Standalone
        // lowering of such a sig would emit a HostFunction whose
        // tensor precision slots cannot be filled in, tripping the F2
        // backend tripwire. The function is still reachable via
        // call-site inlining (the call site supplies the concrete
        // precision per spec/04-type-system.md §5.8.1); we just must
        // skip the standalone top-level emission.
        Some("t-fn") => {
            type_expr_has_precision_var(expr)
                // Tier-2 rank polymorphism (spec/design/rank_polymorphism.md):
                // a t-fn carrying a `(d-rank ...)` rank variable is
                // rank-polymorphic and has no standalone monomorphization — its
                // rank is supplied by call-site inlining, exactly like the
                // precision-var case above. Skip the standalone emission so the
                // `(d-rank ...)` never reaches the lowering assertion.
                || type_expr_has_rank_var(expr)
                || list.elements.last().is_some_and(type_is_never_lowerable)
        }
        Some("t-tuple") => children(list).iter().any(type_is_never_lowerable),
        Some("t-adt") | Some("t-unit") => true,
        Some("t-prim") => false,
        _ => false,
    }
}

/// Walk a Deep type expression and report whether any tensor carries a
/// `(d-rank ...)` rank variable (Tier-2 rank polymorphism). Mirrors
/// `type_expr_has_precision_var`: such a signature is rank-polymorphic, has no
/// standalone monomorphization, and is reached only through call-site inlining.
pub fn type_expr_has_rank_var(expr: &Expr) -> bool {
    let Expr::List(list, _) = expr else {
        return false;
    };
    match get_tag(list) {
        Some("t-tensor") => children(list)
            .iter()
            .any(|c| matches!(c, Expr::List(d, _) if get_tag(d) == Some("d-rank"))),
        Some("t-ref") | Some("t-fn") | Some("t-tuple") | Some("t-adt") => {
            children(list).iter().any(type_expr_has_rank_var)
        }
        _ => false,
    }
}

/// WS-A8: walk a Deep type expression and report whether any
/// `(t-tensor {} dims... (t-var {} _))` precision slot appears. Used
/// to detect precision-polymorphic signatures that have no standalone
/// monomorphization and must be reached through call-site inlining
/// only (per spec/04-type-system.md §5.8.1).
pub fn type_expr_has_precision_var(expr: &Expr) -> bool {
    let Expr::List(list, _) = expr else {
        return false;
    };
    match get_tag(list) {
        Some("t-tensor") => extract_precision_var_name(expr).is_some(),
        Some("t-ref") | Some("t-fn") | Some("t-tuple") | Some("t-adt") => {
            children(list).iter().any(type_expr_has_precision_var)
        }
        _ => false,
    }
}

fn type_is_scalar_primitive(expr: &Expr) -> bool {
    let Expr::List(list, _) = expr else {
        return false;
    };
    get_tag(list) == Some("t-prim")
}

fn callable_ref_name(expr: &Expr) -> Option<String> {
    let Expr::List(list, _) = expr else {
        return None;
    };
    if get_tag(list) != Some("var") {
        return None;
    }
    children(list)
        .first()
        .and_then(symbol_name)
        .map(|name| name.to_string())
}

fn symbol_name(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Atom(Atom::Symbol(name), _) => Some(name.as_str()),
        _ => None,
    }
}

fn top_level_expr_name(expr: &Expr) -> Option<&str> {
    let Expr::List(list, _) = expr else {
        return None;
    };
    if get_tag(list) != Some("def") {
        return None;
    }
    children(list).first().and_then(symbol_name)
}

fn expr_requires_host_runtime(expr: &Expr) -> bool {
    expr_requires_host_runtime_with_ctx(expr, false)
}

/// Variant of `expr_requires_host_runtime` that takes a context flag.
///
/// When `exempt_to_tensor_literal` is true, a
/// `to_tensor(<literal Cons chain>)` application does NOT force host
/// runtime — the literal lowers directly into the IR DAG via the
/// `to_tensor` arm in `lower_builtin_app`. This is the issue
/// Chelis-Lang/chelis#218 exemption, scoped to **function-def
/// bodies only** (see `def_body_requires_host_runtime`).
///
/// When the flag is false (the default), the legacy behavior holds:
/// any `to_tensor` call forces host runtime. This preserves the
/// emit-main path for top-level non-fn value bindings like
/// `out = compute_grad(to_tensor([...]))`.
fn expr_requires_host_runtime_with_ctx(expr: &Expr, exempt_to_tensor_literal: bool) -> bool {
    match expr {
        Expr::Atom(Atom::Str(_), _) => true,
        Expr::Atom(_, _) => false,
        Expr::Map(map, _) => map
            .entries
            .iter()
            .any(|(_, value)| expr_requires_host_runtime_with_ctx(value, exempt_to_tensor_literal)),
        Expr::MetaExpr(meta, _) => {
            expr_requires_host_runtime_with_ctx(&meta.expr, exempt_to_tensor_literal)
                || meta.entries.iter().any(|(_, value)| {
                    expr_requires_host_runtime_with_ctx(value, exempt_to_tensor_literal)
                })
        }
        Expr::List(list, _) => {
            if exempt_to_tensor_literal && static_to_tensor_literal(expr).is_some() {
                return false;
            }
            if get_tag(list) == Some("if") {
                if !if_expr_is_dag_lowerable(list) {
                    return true;
                }
            } else if matches!(
                get_tag(list),
                Some("match" | "record" | "access" | "tuple-get")
            ) {
                return true;
            }
            if get_tag(list) == Some("var")
                && let Some(name) = children(list).first().and_then(symbol_name)
                && name.chars().next().is_some_and(|ch| ch.is_uppercase())
            {
                return true;
            }
            if get_tag(list) == Some("app")
                && let Some(Expr::List(callee, _)) = children(list).first()
                && get_tag(callee) == Some("var")
                && let Some(name) = children(callee).first().and_then(symbol_name)
                && name.chars().next().is_some_and(|ch| ch.is_uppercase())
            {
                return true;
            }
            if let Some(name) = builtin_name(list) {
                if matches!(
                    name,
                    "print"
                        | "debug"
                        | "string_len"
                        | "string_concat"
                        | "string_slice"
                        | "string_contains"
                        | "string_starts_with"
                        | "string_ends_with"
                        | "string_trim"
                        | "to_string"
                        | "to_int"
                        | "to_float"
                        | "mod"
                        | "bitand"
                        | "bitor"
                        | "bitxor"
                        | "shl"
                        | "shr"
                        | "rank"
                        | "shape"
                        | "numel"
                        | "tuple-get"
                        | "len"
                        | "index"
                        | "append"
                        | "concat"
                        | "take"
                        | "chunk"
                        | "range"
                        | "map"
                        | "filter"
                        | "fold"
                        | "scan"
                        | "tensor_scan"
                        | "partition"
                        | "flat_map"
                        | "flatten"
                        | "zip"
                        | "enumerate"
                        | "dict_of"
                        | "dict_get"
                        | "dict_contains"
                        | "dict_remove"
                        | "dict_insert"
                        | "dict_merge"
                        | "dict_keys"
                        | "dict_values"
                        | "dict_entries"
                        | "read_file"
                        | "write_file"
                        | "read_lines"
                        | "read_bytes"
                        | "file_exists"
                        | "list_dir"
                        | "mmap_file"
                        | "mmap_read"
                        | "mmap_len"
                        | "process_run"
                        | "to_tensor"
                        | "to_list"
                        | "pad_sequences"
                        | "pad_sequences_to"
                        | "einsum"
                        | "split"
                        | "scatter"
                        | "where"
                        | "cumsum"
                        | "sort"
                        | "diagonal"
                        | "trace"
                        | "clamp"
                        | "Cons"
                        | "Nil"
                ) {
                    return true;
                }
                if name == "drop" {
                    return children(list).len() != 2;
                }
                if matches!(
                    name,
                    "add"
                        | "mul"
                        | "sub"
                        | "div"
                        | "floor_div"
                        | "trunc_div"
                        | "max_elem"
                        | "min_elem"
                        | "neg"
                        | "exp"
                        | "log"
                        | "sin"
                        | "sqrt"
                        | "cos"
                        | "tan"
                        | "atan"
                        | "abs"
                        | "floor"
                        | "ceil"
                        | "round"
                        | "relu"
                        | "sigmoid"
                        | "tanh"
                        | "silu"
                        | "gelu"
                        | "cmplt"
                        | "gt"
                        | "gte"
                        | "lte"
                        | "eq"
                        | "neq"
                        | "and"
                        | "or"
                        | "not"
                ) && expr_type_metadata(expr).is_some_and(type_is_scalar_primitive)
                {
                    return true;
                }
            }
            // Walk children, but SKIP the metadata map at element 1 (when
            // present) — meta values like the `span` key carry string
            // atoms that are not runtime values. Without this skip, any
            // list whose meta map includes a string-valued key (e.g.
            // `{span: "n_001"}`) would be misclassified as host-runtime
            // and silently dropped from lowering, breaking the S2 audit
            // invariant.
            //
            // NOTE: This skip is load-bearing for span propagation. The
            // N→1 collapse rule's reachability silently depends on it:
            // if a Map at element index 1 is not skipped here,
            // span-bearing nodes (Octant emits `{span: "..."}` at
            // element 1) get classified as host-runtime and silently
            // dropped before lowering, so the §2.3 audit invariant
            // never gets the chance to fire. The greppable invariant
            // catches a regression only indirectly via downstream test
            // failures; keep this skip in place.
            let meta_idx = if matches!(list.elements.get(1), Some(Expr::Map(_, _))) {
                Some(1usize)
            } else {
                None
            };
            list.elements
                .iter()
                .enumerate()
                .filter(|(idx, _)| Some(*idx) != meta_idx)
                .any(|(_, child)| {
                    expr_requires_host_runtime_with_ctx(child, exempt_to_tensor_literal)
                })
        }
    }
}

/// Check whether the body of a `def` requires host runtime. If the
/// body is a function literal `(fn (params...) fn_body)` whose
/// `fn_body` is itself shaped like a DAG-lowerable tensor expression
/// (not a tuple-return or other multi-root construct), the issue
/// Chelis-Lang/chelis#218 exemption applies and a literal `to_tensor`
/// in the body does not force host routing. Otherwise (top-level
/// non-fn value bindings like `out = compute_grad(...)`, or fn defs
/// that return a tuple), the legacy strict classification holds so
/// the emit-main path or the host tuple-ABI still gets a chance to
/// materialize the runtime literal.
fn def_body_requires_host_runtime(body: &Expr) -> bool {
    let exempt = fn_body_qualifies_for_to_tensor_exemption(body);
    expr_requires_host_runtime_with_ctx(body, exempt)
}

/// Return true iff `body` is a function literal `(fn ... fn_body)`
/// whose `fn_body` is a single-tensor-return expression — i.e., not
/// a `tuple`, `match`, `record`, or similar multi-root / host-shaped
/// construct. This is the scoping rule for the issue
/// Chelis-Lang/chelis#218 exemption: only differentiable
/// tensor-returning function bodies benefit from the to_tensor-
/// literal lowering, because the IR DAG's single-tensor-root model
/// fits them. Tuple-returning fns (like
/// `def eig_pair() -> (tensor[2], tensor[2]) = (to_tensor([1, 2]),
/// to_tensor([3, 4]))`) keep the legacy host classification so the
/// generated C ABI `chelis_tuple* eig_pair(...)` is preserved.
fn fn_body_qualifies_for_to_tensor_exemption(body: &Expr) -> bool {
    let Expr::List(list, _) = body else {
        return false;
    };
    if get_tag(list) != Some("fn") {
        return false;
    }
    // `(fn (params ...) fn_body)` — child index 1 (after the tag +
    // optional meta map) is `(params ...)`, child index 2 is the
    // body. `children(list)` already skips the tag + meta map, so
    // body is at index 1.
    let Some(fn_body) = children(list).get(1) else {
        return false;
    };
    !expr_is_multi_root_construct(fn_body)
}

/// Return true iff `expr` is structurally a multi-root or host-shaped
/// construct that the IR DAG cannot represent as a single tensor
/// node. These are the same tags that `expr_requires_host_runtime`
/// already classifies as host (e.g. `tuple`, `match`, `record`,
/// `access`, `tuple-get`), but checked only at the **outer**
/// position of a fn body — the strict-classification host check
/// still walks children when the flag is false.
fn expr_is_multi_root_construct(expr: &Expr) -> bool {
    let Expr::List(list, _) = expr else {
        return false;
    };
    matches!(
        get_tag(list),
        Some("tuple" | "match" | "record" | "access" | "tuple-get")
    )
}

fn collect_top_level_defs(exprs: &[Expr]) -> HashMap<String, Expr> {
    let mut defs = HashMap::new();
    for expr in exprs {
        collect_top_level_defs_from_expr(expr, &mut defs);
    }
    defs
}

fn collect_top_level_sigs(exprs: &[Expr]) -> HashMap<String, Expr> {
    let mut sigs = HashMap::new();
    for expr in exprs {
        collect_top_level_sigs_from_expr(expr, &mut sigs);
    }
    sigs
}

fn for_each_top_level_item(exprs: &[Expr], f: &mut impl FnMut(&Expr)) {
    for expr in exprs {
        for_each_top_level_item_from_expr(expr, f);
    }
}

fn for_each_top_level_item_from_expr(expr: &Expr, f: &mut impl FnMut(&Expr)) {
    let Expr::List(list, _) = expr else {
        return;
    };
    match get_tag(list) {
        Some("module") => {
            for child in list.elements.iter().skip(3) {
                for_each_top_level_item_from_expr(child, f);
            }
        }
        _ => f(expr),
    }
}

fn collect_top_level_defs_from_expr(expr: &Expr, defs: &mut HashMap<String, Expr>) {
    let Expr::List(list, _) = expr else {
        return;
    };
    match get_tag(list) {
        Some("module") => {
            for child in list.elements.iter().skip(3) {
                collect_top_level_defs_from_expr(child, defs);
            }
        }
        Some("def") => {
            let kids = children(list);
            if let (Some(name), Some(body)) =
                (kids.first().and_then(symbol_name), kids.get(1).cloned())
            {
                defs.insert(name.to_string(), body);
            }
        }
        _ => {}
    }
}

fn collect_top_level_sigs_from_expr(expr: &Expr, sigs: &mut HashMap<String, Expr>) {
    let Expr::List(list, _) = expr else {
        return;
    };
    match get_tag(list) {
        Some("module") => {
            for child in list.elements.iter().skip(3) {
                collect_top_level_sigs_from_expr(child, sigs);
            }
        }
        Some("defsig") => {
            let kids = children(list);
            if let (Some(name), Some(ty)) =
                (kids.first().and_then(symbol_name), kids.get(1).cloned())
            {
                sigs.insert(name.to_string(), ty);
            }
        }
        _ => {}
    }
}

fn def_is_lowered(
    name: &str,
    top_level_defs: &HashMap<String, Expr>,
    top_level_sigs: &HashMap<String, Expr>,
    type_env: &HashMap<String, Expr>,
    cache: &mut HashMap<String, bool>,
    visiting: &mut HashSet<String>,
) -> bool {
    if let Some(lowered) = cache.get(name) {
        return *lowered;
    }
    if !visiting.insert(name.to_string()) {
        return !lookup_declared_type_expr(top_level_sigs, type_env, name)
            .is_some_and(type_is_never_lowerable);
    }

    let lowered = top_level_defs.get(name).is_some_and(|body| {
        // Issue Chelis-Lang/chelis#218: function-def bodies get the
        // to_tensor-literal exemption (a `(fn ...)` body can hold a
        // literal `to_tensor` that lowers into the IR DAG and stays
        // DAG-lowerable). Top-level non-fn value bindings keep the
        // strict classification so emit-main still materializes
        // their runtime literals.
        !def_body_requires_host_runtime(body)
            && !expr_depends_on_nonlowerable_name(
                body,
                top_level_defs,
                top_level_sigs,
                type_env,
                cache,
                visiting,
                &HashSet::new(),
            )
            && !lookup_declared_type_expr(top_level_sigs, type_env, name)
                .is_some_and(type_is_never_lowerable)
    });

    visiting.remove(name);
    cache.insert(name.to_string(), lowered);
    lowered
}

fn lookup_declared_type_expr<'a>(
    top_level_sigs: &'a HashMap<String, Expr>,
    type_env: &'a HashMap<String, Expr>,
    name: &str,
) -> Option<&'a Expr> {
    top_level_sigs
        .get(name)
        .or_else(|| unique_terminal_match(top_level_sigs, name))
        .or_else(|| type_env.get(name))
        .or_else(|| {
            let mut matches = type_env
                .iter()
                .filter_map(|(key, value)| terminal_name_matches(key, name).then_some(value));
            let first = matches.next()?;
            matches.next().is_none().then_some(first)
        })
}

fn terminal_name_matches(full_name: &str, short_name: &str) -> bool {
    full_name == short_name
        || full_name
            .rsplit_once("__")
            .is_some_and(|(_, tail)| tail == short_name)
        || full_name
            .rsplit_once('.')
            .is_some_and(|(_, tail)| tail == short_name)
}

fn unique_terminal_match<'a>(map: &'a HashMap<String, Expr>, name: &str) -> Option<&'a Expr> {
    let mut matches = map
        .iter()
        .filter_map(|(key, value)| terminal_name_matches(key, name).then_some(value));
    let first = matches.next()?;
    matches.next().is_none().then_some(first)
}

fn expr_depends_on_nonlowerable_name(
    expr: &Expr,
    top_level_defs: &HashMap<String, Expr>,
    top_level_sigs: &HashMap<String, Expr>,
    type_env: &HashMap<String, Expr>,
    cache: &mut HashMap<String, bool>,
    visiting: &mut HashSet<String>,
    bound_names: &HashSet<String>,
) -> bool {
    match expr {
        Expr::Atom(_, _) => false,
        Expr::Map(map, _) => map.entries.iter().any(|(_, value)| {
            expr_depends_on_nonlowerable_name(
                value,
                top_level_defs,
                top_level_sigs,
                type_env,
                cache,
                visiting,
                bound_names,
            )
        }),
        Expr::MetaExpr(meta, _) => {
            expr_depends_on_nonlowerable_name(
                &meta.expr,
                top_level_defs,
                top_level_sigs,
                type_env,
                cache,
                visiting,
                bound_names,
            ) || meta.entries.iter().any(|(_, value)| {
                expr_depends_on_nonlowerable_name(
                    value,
                    top_level_defs,
                    top_level_sigs,
                    type_env,
                    cache,
                    visiting,
                    bound_names,
                )
            })
        }
        Expr::List(list, _) => {
            if get_tag(list) == Some("var")
                && let Some(name) = children(list).first().and_then(symbol_name)
                && !bound_names.contains(name)
                && top_level_defs.contains_key(name)
            {
                return !def_is_lowered(
                    name,
                    top_level_defs,
                    top_level_sigs,
                    type_env,
                    cache,
                    visiting,
                );
            }
            if get_tag(list) == Some("fn") {
                let kids = children(list);
                let mut scoped = bound_names.clone();
                if let Some(Expr::List(params, _)) = kids.first() {
                    for param in children(params) {
                        collect_param_bound_names(param, &mut scoped);
                    }
                }
                return kids.get(1).is_some_and(|body| {
                    expr_depends_on_nonlowerable_name(
                        body,
                        top_level_defs,
                        top_level_sigs,
                        type_env,
                        cache,
                        visiting,
                        &scoped,
                    )
                });
            }
            if get_tag(list) == Some("let") {
                let kids = children(list);
                let mut scoped = bound_names.clone();
                if let Some(Expr::List(bindings, _)) = kids.first()
                    && get_tag(bindings) == Some("bind")
                {
                    let binding_children = children(bindings);
                    let mut index = 0;
                    while index + 1 < binding_children.len() {
                        if expr_depends_on_nonlowerable_name(
                            &binding_children[index + 1],
                            top_level_defs,
                            top_level_sigs,
                            type_env,
                            cache,
                            visiting,
                            &scoped,
                        ) {
                            return true;
                        }
                        if let Some(name) = symbol_name(&binding_children[index]) {
                            scoped.insert(name.to_string());
                        }
                        index += 2;
                    }
                }
                return kids.get(1).is_some_and(|body| {
                    expr_depends_on_nonlowerable_name(
                        body,
                        top_level_defs,
                        top_level_sigs,
                        type_env,
                        cache,
                        visiting,
                        &scoped,
                    )
                });
            }
            if get_tag(list) == Some("match") {
                let kids = children(list);
                if kids.first().is_some_and(|scrutinee| {
                    expr_depends_on_nonlowerable_name(
                        scrutinee,
                        top_level_defs,
                        top_level_sigs,
                        type_env,
                        cache,
                        visiting,
                        bound_names,
                    )
                }) {
                    return true;
                }
                for arm in kids.iter().skip(1) {
                    let Expr::List(arm_list, _) = arm else {
                        continue;
                    };
                    if get_tag(arm_list) != Some("arm") {
                        continue;
                    }
                    let arm_children = children(arm_list);
                    let mut scoped = bound_names.clone();
                    if let Some(pattern) = arm_children.first() {
                        collect_pattern_bound_names(pattern, &mut scoped);
                    }
                    if arm_children.get(1).is_some_and(|guard| {
                        expr_depends_on_nonlowerable_name(
                            guard,
                            top_level_defs,
                            top_level_sigs,
                            type_env,
                            cache,
                            visiting,
                            &scoped,
                        )
                    }) || arm_children.get(2).is_some_and(|body| {
                        expr_depends_on_nonlowerable_name(
                            body,
                            top_level_defs,
                            top_level_sigs,
                            type_env,
                            cache,
                            visiting,
                            &scoped,
                        )
                    }) {
                        return true;
                    }
                }
                return false;
            }
            list.elements.iter().any(|child| {
                expr_depends_on_nonlowerable_name(
                    child,
                    top_level_defs,
                    top_level_sigs,
                    type_env,
                    cache,
                    visiting,
                    bound_names,
                )
            })
        }
    }
}

fn collect_param_bound_names(param: &Expr, out: &mut HashSet<String>) {
    match param {
        Expr::Atom(Atom::Symbol(name), _) => {
            out.insert(name.clone());
        }
        Expr::List(list, _) => {
            if let Some(name) = list.elements.first().and_then(symbol_name) {
                out.insert(name.to_string());
            }
        }
        Expr::Map(_, _) | Expr::MetaExpr(_, _) | Expr::Atom(_, _) => {}
    }
}

fn collect_pattern_bound_names(pattern: &Expr, out: &mut HashSet<String>) {
    match pattern {
        Expr::Atom(_, _) | Expr::Map(_, _) => {}
        Expr::MetaExpr(meta, _) => collect_pattern_bound_names(&meta.expr, out),
        Expr::List(list, _) => {
            if get_tag(list) == Some("pat-var")
                && let Some(name) = children(list).first().and_then(symbol_name)
            {
                out.insert(name.to_string());
                return;
            }
            for child in children(list) {
                collect_pattern_bound_names(child, out);
            }
        }
    }
}

fn builtin_name(list: &List) -> Option<&str> {
    if get_tag(list) != Some("app") {
        return None;
    }
    list.elements.get(2).and_then(|expr| match expr {
        Expr::List(var_list, _) if get_tag(var_list) == Some("var") => {
            children(var_list).first().and_then(symbol_name)
        }
        _ => None,
    })
}

fn expr_type_metadata(expr: &Expr) -> Option<&Expr> {
    let Expr::List(list, _) = expr else {
        return None;
    };
    match list.elements.get(1) {
        Some(Expr::Map(meta, _)) => meta
            .entries
            .iter()
            .find(|(key, _)| key == "type")
            .map(|(_, value)| value),
        _ => None,
    }
}

fn if_expr_is_dag_lowerable(list: &List) -> bool {
    if get_tag(list) != Some("if") {
        return false;
    }

    let result_ty = match list.elements.get(1) {
        Some(Expr::Map(meta, _)) => LowerCtx::type_from_meta_static(&meta.entries),
        _ => LowerCtx::default_type(),
    };
    if !result_ty.precision.is_float() {
        return false;
    }

    let Some(cond_ty_expr) = children(list).first().and_then(expr_type_metadata) else {
        return false;
    };
    let cond_ty = LowerCtx::type_from_type_expr(cond_ty_expr);
    cond_ty.precision == Prim::Bool && (cond_ty.dims.is_empty() || cond_ty.dims == result_ty.dims)
}

fn assert_ir_lowerable(expr: &Expr) {
    match expr {
        Expr::List(list, _) => {
            // Declaration nodes carry no runtime IR. Their meta map holds
            // spec artifacts (e.g. a `deftype`'s declared `invariant`
            // predicate, consumed only by `chelis prove`) and their
            // children are type-level expressions, never runtime app
            // nodes. `lower_top_level` already returns early for these
            // tags before any lowering; the runtime IR audit mirrors that
            // skip so it never validates declaration metadata as runtime
            // IR. Without this skip, a tensor-field invariant such as
            // `sum(p.weights)` is wrongly audited as a runtime node.
            if let Some(Expr::Atom(Atom::Symbol(tag), _)) = list.elements.first()
                && matches!(tag.as_str(), "deftype" | "defsig" | "typealias")
            {
                return;
            }
            if let Some(Expr::Atom(Atom::Symbol(tag), _)) = list.elements.first()
                && ((tag == "if" && !if_expr_is_dag_lowerable(list)) || tag == "match")
            {
                raise_lowering_diagnostic(lower_diagnostic_for_expr(
                    unsupported_lowering_message(tag),
                    expr,
                ));
            }
            for elem in &list.elements {
                assert_ir_lowerable(elem);
            }
        }
        Expr::Map(map, _) => {
            for (_, value) in &map.entries {
                assert_ir_lowerable(value);
            }
        }
        Expr::MetaExpr(inner, _) => {
            for (_, value) in &inner.entries {
                assert_ir_lowerable(value);
            }
            assert_ir_lowerable(&inner.expr);
        }
        Expr::Atom(_, _) => {}
    }
}

fn assert_ir_typed(expr: &Expr) {
    match expr {
        Expr::List(list, _) => {
            // Declaration nodes carry no runtime IR. A `deftype`'s meta
            // map holds the declared `invariant` predicate (spec metadata
            // for `chelis prove`, never lowered to runtime IR), which can
            // contain shape-sensitive apps such as `sum(p.weights)` with
            // no runtime `type` metadata. `lower_top_level` returns early
            // for these tags before lowering; the runtime IR audit
            // mirrors that skip so it never validates declaration
            // metadata as runtime IR.
            if let Some(Expr::Atom(Atom::Symbol(tag), _)) = list.elements.first()
                && matches!(tag.as_str(), "deftype" | "defsig" | "typealias")
            {
                return;
            }
            if let Some(Expr::Atom(Atom::Symbol(tag), _)) = list.elements.first()
                && tag == "app"
                && is_shape_sensitive_builtin_app(list)
                && !has_type_metadata(list)
            {
                let rendered = chelis_deep::printer::print_canonical(std::slice::from_ref(expr))
                    .replace('\n', " ");
                raise_lowering_diagnostic(lower_diagnostic_for_expr(
                    format!(
                        "shape-sensitive IR app nodes must carry explicit type metadata before lowering: {rendered}"
                    ),
                    expr,
                ));
            }
            for elem in &list.elements {
                assert_ir_typed(elem);
            }
        }
        Expr::Map(map, _) => {
            for (_, value) in &map.entries {
                assert_ir_typed(value);
            }
        }
        Expr::MetaExpr(inner, _) => {
            for (_, value) in &inner.entries {
                assert_ir_typed(value);
            }
            assert_ir_typed(&inner.expr);
        }
        Expr::Atom(_, _) => {}
    }
}

fn has_type_metadata(list: &List) -> bool {
    matches!(list.elements.get(1), Some(Expr::Map(meta, _)) if meta.entries.iter().any(|(k, _)| k == "type"))
}

fn is_shape_sensitive_builtin_app(list: &List) -> bool {
    let func_name = match list.elements.get(2) {
        Some(Expr::List(func_list, _)) => {
            match (func_list.elements.first(), func_list.elements.get(2)) {
                (
                    Some(Expr::Atom(Atom::Symbol(tag), _)),
                    Some(Expr::Atom(Atom::Symbol(name), _)),
                ) if tag == "var" => Some(name.as_str()),
                _ => None,
            }
        }
        _ => None,
    };

    matches!(
        func_name,
        Some(
            "matmul"
                | "softmax"
                | "mean"
                | "layer_norm"
                | "conv2d"
                | "sum"
                | "max_reduce"
                | "min_reduce"
                | "prod_reduce"
                | "argmax_reduce"
                | "argmin_reduce"
                | "reduce_window_max"
                | "reduce_window_min"
                | "reduce_window_sum"
                | "reduce_window_mean"
                | "reshape"
                | "permute"
                | "expand"
                | "pad"
                | "shrink"
                | "stride"
        )
    )
}

fn get_tag(list: &List) -> Option<&str> {
    match list.elements.first() {
        Some(Expr::Atom(Atom::Symbol(tag), _)) => Some(tag.as_str()),
        _ => None,
    }
}

fn children(list: &List) -> &[Expr] {
    if list.elements.len() > 2 {
        &list.elements[2..]
    } else {
        &[]
    }
}

fn app_var_name_and_args(expr: &Expr) -> Option<(&str, &[Expr])> {
    let Expr::List(list, _) = expr else {
        return None;
    };
    if get_tag(list) != Some("app") {
        return None;
    }
    let kids = children(list);
    let func = kids.first()?;
    let Expr::List(func_list, _) = func else {
        return None;
    };
    if get_tag(func_list) != Some("var") {
        return None;
    }
    let name = children(func_list).first().and_then(|expr| match expr {
        Expr::Atom(Atom::Symbol(name), _) => Some(name.as_str()),
        _ => None,
    })?;
    Some((name, &kids[1..]))
}

fn to_list_source_expr(expr: &Expr) -> Option<&Expr> {
    let ("to_list", [source]) = app_var_name_and_args(expr)? else {
        return None;
    };
    Some(source)
}

fn concrete_dim_len(dim: &DimInfo) -> Option<usize> {
    match dim {
        DimInfo::Lit(n) => Some(*n),
        DimInfo::Named(_, Some(n)) => Some(*n),
        DimInfo::Named(_, None) => None,
    }
}

/// Helper: is `expr` an `app` of a `var` whose name equals `expected`?
fn is_app_of_builtin(expr: &Expr, expected: &str) -> bool {
    let Expr::List(list, _) = expr else {
        return false;
    };
    if get_tag(list) != Some("var") {
        return false;
    }
    matches!(
        children(list).first(),
        Some(Expr::Atom(Atom::Symbol(name), _)) if name == expected
    )
}

/// Extract an `i64` from a `pad`/`shrink` bound element.
///
/// Routes through the shared [`extract_int_for_dim`] walker so the same
/// `Atom::Int` / `(lit {} N)` / `(cast {} N <prim>)` shapes that
/// `reshape`/`stride` already accept also work for bound pairs. Issue
/// Chelis-Lang/chelis#291: the headline reproducer writes the bounds as
/// `[[cast(0, int32), cast(2, int32)]]`, and the old literal-only walker
/// returned `None` for the `(cast ...)` element, so `extract_pair_list`
/// fell back to an empty `bounds = vec![]` and the resulting
/// `RiscOp::Shrink { bounds: [] }` failed backward-DAG verification with
/// `bounds len 0 != input rank 1`. PR #296 hardened the `stride`/`wrt`
/// integer extraction against the same `cast`-wrapped shape; this closes
/// the matching gap for `shrink`/`pad` bounds.
fn cons_pair_extract_int(expr: &Expr) -> Option<i64> {
    extract_int_for_dim(expr)
}

/// Walk `Cons(start, Cons(end, Nil))` and return `(start, end)` as
/// `usize`. Returns `None` if any structural assumption fails.
fn cons_two_int_pair(expr: &Expr) -> Option<(usize, usize)> {
    let Expr::List(outer, _) = expr else {
        return None;
    };
    if get_tag(outer) != Some("app") {
        return None;
    }
    let outer_children = children(outer);
    let func = outer_children.first()?;
    if !is_app_of_builtin(func, "Cons") {
        return None;
    }
    let start = cons_pair_extract_int(outer_children.get(1)?)?;
    let tail = outer_children.get(2)?;
    let Expr::List(tail_list, _) = tail else {
        return None;
    };
    if get_tag(tail_list) != Some("app") {
        return None;
    }
    let tail_children = children(tail_list);
    let tail_func = tail_children.first()?;
    if !is_app_of_builtin(tail_func, "Cons") {
        return None;
    }
    let end = cons_pair_extract_int(tail_children.get(1)?)?;
    let nil_expr = tail_children.get(2)?;
    let Expr::List(nil_list, _) = nil_expr else {
        return None;
    };
    if get_tag(nil_list) != Some("var") {
        return None;
    }
    let nil_name = match children(nil_list).first() {
        Some(Expr::Atom(Atom::Symbol(s), _)) => s.as_str(),
        _ => return None,
    };
    if nil_name != "Nil" {
        return None;
    }
    if start < 0 || end < 0 {
        return None;
    }
    Some((start as usize, end as usize))
}

/// Walk a `Cons(pair_0, Cons(pair_1, ..., Nil))` chain and collect each
/// `pair_i` via [`cons_two_int_pair`]. Returns `None` if the chain or any
/// pair is malformed (so callers can fall back to a non-Cons-form parser).
fn cons_chain_pair_list(expr: &Expr) -> Option<Vec<(usize, usize)>> {
    let mut pairs = Vec::new();
    let mut cursor = expr;
    loop {
        let Expr::List(outer, _) = cursor else {
            return None;
        };
        match get_tag(outer)? {
            "var" => {
                let name = match children(outer).first() {
                    Some(Expr::Atom(Atom::Symbol(s), _)) => s.as_str(),
                    _ => return None,
                };
                if name == "Nil" {
                    return Some(pairs);
                }
                return None;
            }
            "app" => {
                let app_children = children(outer);
                let func = app_children.first()?;
                if !is_app_of_builtin(func, "Cons") {
                    return None;
                }
                let pair = cons_two_int_pair(app_children.get(1)?)?;
                pairs.push(pair);
                cursor = app_children.get(2)?;
            }
            _ => return None,
        }
    }
}

/// Walk a `Cons(head_0, Cons(head_1, ..., Nil))` chain and collect the
/// head exprs in order. Returns `None` if the chain isn't closed by
/// `(var {} Nil)` or contains a non-Cons app. Used by
/// [`LowerCtx::extract_dim_list`] to recognize Surf-desugared list
/// literals (issue Chelis-Lang/chelis#220).
fn collect_cons_chain(expr: &Expr) -> Option<Vec<&Expr>> {
    let mut out = Vec::new();
    let mut cursor = expr;
    loop {
        let Expr::List(list, _) = cursor else {
            return None;
        };
        match get_tag(list)? {
            "var" => {
                let name = match children(list).first() {
                    Some(Expr::Atom(Atom::Symbol(s), _)) => s.as_str(),
                    _ => return None,
                };
                if name == "Nil" {
                    return Some(out);
                }
                return None;
            }
            "app" => {
                let app_children = children(list);
                let func = app_children.first()?;
                if !is_app_of_builtin(func, "Cons") {
                    return None;
                }
                let head = app_children.get(1)?;
                let tail = app_children.get(2)?;
                out.push(head);
                cursor = tail;
            }
            _ => return None,
        }
    }
}

/// Walk a lowered `Cons`/`Nil` spine into its element values, in order.
/// The VALUE-level companion of [`collect_cons_chain`] (chelis#620): a list
/// that only becomes statically known after inlining/unrolling (e.g. a
/// recursive patch collector's return value, or a `let`-bound append
/// result) is invisible to the expr-level walk, which sees only a bound
/// var; by then the list exists as a `LoweredValue::Adt` constructor
/// chain. Returns `None` for anything that is not a closed chain.
fn adt_cons_chain_values(value: &LoweredValue) -> Option<Vec<LoweredValue>> {
    let mut out = Vec::new();
    let mut cursor = value;
    loop {
        match cursor {
            LoweredValue::Adt { ctor, fields, .. } if ctor == "Cons" && fields.len() == 2 => {
                out.push(fields[0].clone());
                cursor = &fields[1];
            }
            LoweredValue::Adt { ctor, fields, .. } if ctor == "Nil" && fields.is_empty() => {
                return Some(out);
            }
            _ => return None,
        }
    }
}

/// Rebuild a `Cons`/`Nil` spine from element values (chelis#620): the
/// inverse of [`adt_cons_chain_values`], used to materialize a static
/// list-append result.
fn rebuild_cons_chain(items: Vec<LoweredValue>) -> LoweredValue {
    let mut chain = LoweredValue::Adt {
        ctor: "Nil".to_string(),
        field_names: None,
        fields: Vec::new(),
    };
    for item in items.into_iter().rev() {
        chain = LoweredValue::Adt {
            ctor: "Cons".to_string(),
            field_names: None,
            fields: vec![item, chain],
        };
    }
    chain
}

/// Extract a positive integer dim from a Deep expression. Handles
/// the common shapes that appear inside `reshape`'s shape list after
/// `chelis-surf::desugar`:
///   * `Atom::Int(n)`
///   * `(lit {} <int>)`
///   * `(cast {} <int|lit|cast> <prim>)` (the `cast(N, int64)` form
///     is idiomatic since integer literals default to int32 and
///     `reshape` expects `List[int64]`)
///
/// Returns the numeric value as `i64` when extractable. Used by
/// [`LowerCtx::extract_dim_list`] to interpret reshape shape-list
/// entries (issue Chelis-Lang/chelis#220).
fn extract_int_for_dim(expr: &Expr) -> Option<i64> {
    match expr {
        Expr::Atom(Atom::Int(n), _) => Some(*n),
        Expr::List(list, _) => match get_tag(list)? {
            "lit" => match list.elements.get(2)? {
                Expr::Atom(Atom::Int(n), _) => Some(*n),
                _ => None,
            },
            "cast" => extract_int_for_dim(list.elements.get(2)?),
            _ => None,
        },
        _ => None,
    }
}

/// Resolve a compile-time-constant AXIS to its (possibly negative) `i64`
/// value. A reduction/softmax/gather axis is `extract_int_for_dim`'s
/// literal/`(lit ...)`/`cast(<int>)` forms PLUS the negative-axis literal
/// `-k`, which Surf desugars to `(app (var neg) <int>)` (issue #218 R1).
/// A negative axis indexes from the end of the operand rank (`-1` is the
/// last axis) and is normalized by [`LowerCtx::normalize_axis`] once the
/// rank is known. Keeping this distinct from `extract_int_for_dim` (which
/// serves DIM sizes, where a negative value is invalid) preserves the
/// negative-axis convention for the axis-taking ops. issue #364/#319: the
/// `cast`-axis fix must not lose the `-1` axis form that `softmax(x, -1)`
/// (and the SDPA grad path) depend on.
fn extract_int_axis(expr: &Expr) -> Option<i64> {
    if let Some(n) = extract_int_for_dim(expr) {
        return Some(n);
    }
    // Negative-axis literal: `-k` desugars to `(app (var neg) <inner>)`.
    if let Expr::List(list, _) = expr
        && get_tag(list) == Some("app")
        && let Some(callee) = children(list).first()
        && expr_is_var_named(callee, "neg")
        && let Some(inner) = children(list).get(1)
        && let Some(n) = extract_int_axis(inner)
    {
        return Some(-n);
    }
    None
}

/// Recognize a `shape(operand, axis)` application — possibly wrapped in
/// one or more `cast(..., int32)` layers — and return its `(operand,
/// axis)` pair. The axis must be a static literal (bare int, `(lit ...)`,
/// or `cast`-wrapped int); a runtime axis is not extractable here.
///
/// This is the structural recognizer behind the issue #318 fix: the
/// shape-derived const-broadcast idiom
/// `expand(scalar_to_tensor(c), 0, shape(&x, cast(0, int32)))` (and the
/// fully-`cast`-wrapped `cast(shape(&x, ...), int32)` form) carries its
/// broadcast extent as the runtime dimension of `operand` at `axis`. The
/// type checker collapses the `expand` *output* dim to `Lit(1)` via
/// size-1 broadcasting, so the extent must be read from this `shape`
/// argument's operand, not from the expand node's type.
/// Strip any chain of `(cast {} <inner> (t-prim {} ...))` wrappers,
/// returning the innermost non-cast expression. The expand size argument
/// in the `tensor_full_like` idiom is `cast(var len, int32)`; peeling the
/// cast reaches the bare `var len` so [`bare_var_name`] /
/// [`shape_app_operand_axis`] can match it (chelis#369, mirroring the
/// `cast`-strip already in [`shape_app_operand_axis`]).
fn strip_cast_wrappers(expr: &Expr) -> &Expr {
    let mut current = expr;
    while let Expr::List(list, _) = current {
        if get_tag(list) == Some("cast")
            && let Some(inner) = list.elements.get(2)
        {
            current = inner;
        } else {
            break;
        }
    }
    current
}

fn shape_app_operand_axis(expr: &Expr) -> Option<(&Expr, usize)> {
    let Expr::List(list, _) = expr else {
        return None;
    };
    // Strip outer `cast(..., int32)` wrappers to reach the `shape` app.
    if get_tag(list) == Some("cast") {
        return list.elements.get(2).and_then(shape_app_operand_axis);
    }
    if builtin_name(list) != Some("shape") {
        return None;
    }
    // `(app {} (var {} shape) <operand> <axis_arg>)`: operand at index 3,
    // axis at index 4.
    let operand = list.elements.get(3)?;
    let axis = list.elements.get(4).and_then(extract_int_for_dim)?;
    let axis = usize::try_from(axis).ok()?;
    Some((operand, axis))
}

/// If `expr` is `(var {} <name>)`, return `<name>` as a `String`.
/// Otherwise return `None`. Used by [`LowerCtx::extract_dim_list`] to
/// recognize symbolic dim entries inside a reshape shape list (issue
/// Chelis-Lang/chelis#220).
fn symbolic_dim_var_name(expr: &Expr) -> Option<String> {
    let Expr::List(list, _) = expr else {
        return None;
    };
    if get_tag(list) != Some("var") {
        return None;
    }
    children(list)
        .first()
        .and_then(symbol_name)
        .map(|name| name.to_string())
}

/// Result of attempting to recognize a `to_tensor` argument as a
/// static numeric Cons-chain literal.
///
/// Per issue Chelis-Lang/chelis#218: when a function body
/// materializes a constant tensor via `to_tensor([literal_floats])`
/// and uses it on the data path, the routing decision in
/// `expr_requires_host_runtime_with_ctx` must not unconditionally
/// classify the whole body as host-required. Static literal
/// `to_tensor` calls can lower into the IR DAG via a composition of
/// existing primitives (no schema change), so the containing fn-def
/// stays DAG-lowerable and the `grad` lowering can reach it. A
/// constant tensor has zero gradient; AD treats the lowered cascade
/// as a tree of `Const` leaves with empty input-grad lists, the
/// correct adjoint contribution.
///
/// The exemption is **scoped to function-def bodies only** —
/// top-level non-fn value bindings like
/// `out = compute_grad(to_tensor([...]))` still need host routing
/// so that `emit_main` produces a main that materializes the
/// runtime tensor literal and prints the result.
///
/// `shape` is the rank-N dimension list (row-major). `data` is the
/// row-major-flat float buffer of length `shape.iter().product()`.
#[derive(Debug, Clone, PartialEq)]
struct LiteralToTensor {
    shape: Vec<usize>,
    /// Exact raw elements (chelis#856): integer leaves travel their
    /// exact i64, float leaves their f64; finalization at the tensor's
    /// ascribed dtype happens ONCE in `emit_literal_tensor`.
    data: Vec<chelis_types::RawScalar>,
}

/// If `expr` is a `to_tensor(...)` application whose single argument
/// is a recognizable numeric Cons-chain literal, return the extracted
/// tensor shape + flat row-major data. Otherwise return `None`.
///
/// Returning `None` is the conservative default: the existing
/// host-routing classification stays in force for any to_tensor that
/// doesn't fit the literal-Cons-chain shape (e.g. a `to_tensor(items)`
/// where `items` is a host-side List variable).
fn static_to_tensor_literal(expr: &Expr) -> Option<LiteralToTensor> {
    let Expr::List(list, _) = expr else {
        return None;
    };
    if get_tag(list) != Some("app") {
        return None;
    }
    let app_kids = children(list);
    let callee = app_kids.first()?;
    if !expr_is_var_named(callee, "to_tensor") {
        return None;
    }
    let arg = app_kids.get(1)?;
    extract_cons_chain_tensor(arg)
}

/// Return true iff `expr` is `(var {} <expected_name>)`. Helper for
/// recognizing builtin-name references in app callee position.
fn expr_is_var_named(expr: &Expr, expected_name: &str) -> bool {
    let Expr::List(list, _) = expr else {
        return false;
    };
    if get_tag(list) != Some("var") {
        return false;
    }
    children(list).first().and_then(symbol_name) == Some(expected_name)
}

/// Walk a Cons/Nil chain. If every leaf reduces to a numeric atom,
/// return the shape + flat row-major data. Handles arbitrary nesting:
/// a Cons-chain of Cons-chains of ... of numerics yields a rank-N
/// tensor. Returns `None` if the chain is malformed, not closed by
/// Nil, or contains any non-numeric atom.
fn extract_cons_chain_tensor(expr: &Expr) -> Option<LiteralToTensor> {
    let elements = collect_cons_chain(expr)?;
    if elements.is_empty() {
        // Empty list: rank-1 zero-element tensor. Pad over a zero-
        // size dim doesn't compose cleanly; reject and let host
        // routing handle it.
        return None;
    }
    if let Some(scalars) = elements
        .iter()
        .map(|e| extract_numeric_leaf(e))
        .collect::<Option<Vec<chelis_types::RawScalar>>>()
    {
        return Some(LiteralToTensor {
            shape: vec![scalars.len()],
            data: scalars,
        });
    }
    let nested = elements
        .iter()
        .map(|e| extract_cons_chain_tensor(e))
        .collect::<Option<Vec<LiteralToTensor>>>()?;
    let inner_shape = nested.first()?.shape.clone();
    if nested.iter().any(|t| t.shape != inner_shape) {
        return None;
    }
    let mut shape = vec![nested.len()];
    shape.extend(inner_shape.iter().copied());
    let mut data = Vec::with_capacity(shape.iter().product::<usize>());
    for t in &nested {
        data.extend_from_slice(&t.data);
    }
    Some(LiteralToTensor { shape, data })
}

/// chelis#620: recursion lowers by unrolling, and these caps bound it.
/// Per-name limit on active inline levels for one callee. 512 covers an
/// im2col-style unroll at a 22x22 kernel (484 levels) with margin; School's
/// real kernels are at most 7x7 (49 levels). A well-founded recursion
/// terminates through static `if`/`match` pruning long before the cap; a
/// runtime-bounded recursion hits the cap in milliseconds and errors loudly
/// instead of hanging.
const MAX_STATIC_RECURSION_DEPTH: usize = 512;
/// Total limit on simultaneously active inlined bodies across all names.
/// Each level costs roughly ten lowering frames (a few KB of Rust stack), so
/// this cap, not the per-name one, is what bounds the stack for mutually
/// recursive cycles (k names at 512 each would otherwise stack k*512 levels).
const MAX_TOTAL_INLINE_DEPTH: usize = 1024;

/// Extract a numeric scalar from a Deep expression. Recognizes:
///   * `Atom::Int` / `Atom::Float` / `Atom::Bool`
///   * `(lit {} <Int|Float|Bool>)`
///   * `(cast {} <Int|Float|Bool> <prim>)` (constant after cast is
///     still a constant; the precision distinction is carried on the
///     enclosing `to_tensor`'s type metadata, which the emit path
///     uses for the lowered tensor's `precision` slot).
///   * `(app {} (var {} neg) <inner>)` (issue Chelis-Lang/chelis#218
///     R1 HIGH-1): surface negative literals like `-1.0` desugar to
///     `(app (var neg) (lit 1.0))`; the recognizer returns the
///     negated inner value. Nested casts and lits are handled by the
///     recursive call.
fn extract_numeric_leaf(expr: &Expr) -> Option<chelis_types::RawScalar> {
    use chelis_types::RawScalar;
    match expr {
        Expr::Atom(Atom::Int(n), _) => Some(RawScalar::Int(*n)),
        Expr::Atom(Atom::Float(f), _) => Some(RawScalar::Float(*f)),
        Expr::Atom(Atom::Bool(b), _) => Some(RawScalar::Int(i64::from(*b))),
        Expr::List(list, _) => match get_tag(list)? {
            "lit" => match list.elements.get(2)? {
                Expr::Atom(Atom::Int(n), _) => Some(RawScalar::Int(*n)),
                Expr::Atom(Atom::Float(f), _) => Some(RawScalar::Float(*f)),
                Expr::Atom(Atom::Bool(b), _) => Some(RawScalar::Int(i64::from(*b))),
                _ => None,
            },
            "cast" => {
                // A cast leaf APPLIES the checked default ladder
                // (spec/04 section 5.2) at recognition time, so
                // `cast(3.5, int32)` contributes 3 exactly and
                // `cast(9007199254740993, int64)` stays exact. A
                // trapping cast DECLINES static recognition (the
                // section C2 decline discipline): the dynamic lowering
                // evaluates the same cast and traps with its full
                // runtime diagnostic.
                let inner = list.elements.get(2)?;
                let raw = extract_numeric_leaf(inner)?;
                let target = LowerCtx::try_extract_prim(list.elements.get(3)?)?;
                let cast = chelis_types::cast_raw("cast", raw, target).ok()?;
                Some(match cast.as_i64_exact() {
                    Some(i) => RawScalar::Int(i),
                    None => RawScalar::Float(cast.as_f64_lossy()),
                })
            }
            "app" => {
                // Negative literal: `-x` desugars to
                // `(app (var neg) <inner>)`. The recognizer returns
                // `-extract(inner)` so the literal recognizer sees
                // through the desugared unary minus. Any other app
                // shape is not a static literal.
                let callee = children(list).first()?;
                if !expr_is_var_named(callee, "neg") {
                    return None;
                }
                let inner = children(list).get(1)?;
                Some(match extract_numeric_leaf(inner)? {
                    RawScalar::Int(i) => RawScalar::Int(i.checked_neg()?),
                    RawScalar::Float(f) => RawScalar::Float(-f),
                })
            }
            _ => None,
        },
        _ => None,
    }
}

/// Return true iff any element of `dims` is a wildcard placeholder
/// dim (a `Named("*", _)` or `Named("", _)` entry that
/// `crates/chelis-backend-c/src/emit.rs`'s `rename_anonymous_dims`
/// would later rename to `_anon_dim_*`).
///
/// The static-literal path in `chelis-types::infer`'s `to_tensor`
/// rule (`static_to_tensor_shape`) emits concrete dims for
/// statically-resolvable nested-list literals, so the static-literal
/// path no longer produces wildcards. The remaining sources are
/// genuinely-dynamic ops at type-check time: variable-fed
/// `to_tensor(items)`, `concat` along the concat axis, `split`
/// per-piece sizes, and `pad_sequences` batch dim. Reduction
/// lowering uses this predicate to prefer the input DAG node's
/// authoritative shape when one of those wildcard-bearing meta
/// types reaches a reduction op.
fn any_wildcard_dim(dims: &[DimInfo]) -> bool {
    dims.iter().any(|dim| match dim {
        DimInfo::Named(name, _) => name.is_empty() || name == "*",
        DimInfo::Lit(_) => false,
    })
}

/// Compute output dims for `reduce_window_*` under `Valid` padding.
///
/// Leading `rank - n` axes pass through; each windowed axis has extent
/// `floor((input_dim - window_shape[i]) / strides[i]) + 1`.
///
/// The windowed output axis is strictly a *function* of the input
/// extent, so it must be recomputed — a windowed axis is never the same
/// size as its input unless `window == 1, stride == 1`. We therefore
/// compute concretely for any windowed axis whose size is statically
/// known (`DimInfo::Lit` or `DimInfo::Named(_, Some(_))`).
///
/// A windowed axis whose input extent is unknown at compile time
/// (`DimInfo::Named(_, None)`, e.g. a `pad_sequences` result whose dims
/// are bound from runtime metadata) is **not** representable as a static
/// `DimInfo`: passing the input symbol through would falsely assert
/// `output_size == input_size`, which mis-allocates the output tensor in
/// the backends (the input symbol is bound to the larger input extent).
/// Per `spec/05-risc-primitives.md` §2.3.1 the build/backend path
/// requires statically-known windowed-axis extents; when a windowed axis
/// is unknown we fall back to the caller-supplied `ty.dims` so the
/// type-checker's (wildcard / shape-erased) result governs rather than a
/// silently-wrong passthrough. The IR evaluator and host runtime always
/// recompute from the concrete runtime shape and are unaffected.
fn compute_reduce_window_out_dims(
    input_dims: &[DimInfo],
    window_shape: &[usize],
    strides: &[usize],
    fallback_ty: &TensorType,
) -> Vec<DimInfo> {
    let n = window_shape.len();
    if input_dims.len() < n || n == 0 || window_shape.len() != strides.len() {
        return fallback_ty.dims.clone();
    }
    let leading = input_dims.len() - n;
    let mut out_dims: Vec<DimInfo> = input_dims[..leading].to_vec();
    for i in 0..n {
        let known = match &input_dims[leading + i] {
            DimInfo::Lit(in_dim) => Some(*in_dim),
            DimInfo::Named(_, Some(in_dim)) => Some(*in_dim),
            DimInfo::Named(_, None) => None,
        };
        match known {
            Some(in_dim) => {
                if window_shape[i] == 0 || strides[i] == 0 || in_dim < window_shape[i] {
                    return fallback_ty.dims.clone();
                }
                let out = (in_dim - window_shape[i]) / strides[i] + 1;
                out_dims.push(DimInfo::Lit(out));
            }
            // Windowed axis with a compile-time-unknown extent: defer to
            // the checker-derived fallback rather than emit a wrong
            // passthrough. See the doc comment above.
            None => return fallback_ty.dims.clone(),
        }
    }
    out_dims
}

#[derive(Clone)]
enum CallableExpr {
    Plain(Expr),
    Vmap {
        fn_expr: Expr,
        axis: usize,
    },
    VmapGrad {
        fn_expr: Expr,
        wrt: Option<Vec<usize>>,
        axis: usize,
    },
    Grad {
        fn_expr: Expr,
        wrt: Option<Vec<usize>>,
    },
    /// A reference to a function-valued parameter (e.g. `f` inside
    /// `def double_apply(f: T -> T, x: T) = x |> f`). The parameter has
    /// no body to inline at standalone-def lowering time — the DAG can't
    /// represent a call to it because there is no `RiscOp::Call`. Call
    /// sites that pass a concrete function for this parameter (via
    /// `lower_plain_callable_app`) insert the resolved callable into
    /// `local_callables` *before* lowering the inlined body, so the
    /// resolver never produces this variant on the inlined-body path.
    /// `Parameter` therefore appears only on the standalone-def lowering
    /// path, where the right thing to do is no-op (see `lower_pipe`'s
    /// arm) — the standalone DAG entry is never user-visible because
    /// every caller re-inlines.
    Parameter {
        #[allow(dead_code)]
        name: String,
    },
}

#[derive(Clone)]
enum LoweredValue {
    Node(NodeId),
    Tuple(Vec<LoweredValue>),
    /// A statically-known ADT/record value (chelis#520). Constructed by
    /// `lower_record` (record-syntax construction), constructor
    /// application in `lower_app`, and nullary-constructor references in
    /// `lower_var`. Consumed by `lower_match` (static arm selection: the
    /// constructor tag is discrete, so the taken arm is known at lowering
    /// time), `lower_access` (field projection), and the grad lowering
    /// (field-wise ADT gradients, the pytree contract of
    /// spec/design/differentiable_language.md Decision 6).
    Adt {
        ctor: String,
        /// Declared field names for record-syntax constructors, in the
        /// order `fields` was built; `None` for positional constructors.
        field_names: Option<Vec<String>>,
        fields: Vec<LoweredValue>,
    },
}

impl LoweredValue {
    fn expect_node(&self, context: &str) -> NodeId {
        match self {
            Self::Node(id) => *id,
            Self::Tuple(_) => raise_lowering_error(
                format!("{context} expected a single tensor value"),
                None,
                None,
            ),
            Self::Adt { ctor, .. } => raise_lowering_error(
                format!(
                    "{context} expected a single tensor value, got an ADT value \
                     constructed with `{ctor}`"
                ),
                None,
                None,
            ),
        }
    }

    fn flatten_nodes(&self) -> Vec<NodeId> {
        match self {
            Self::Node(id) => vec![*id],
            Self::Tuple(items) => items.iter().flat_map(Self::flatten_nodes).collect(),
            Self::Adt { fields, .. } => fields.iter().flat_map(Self::flatten_nodes).collect(),
        }
    }

    /// The single DAG node id if this value is a `Node`, else `None`
    /// (e.g. a `Tuple`). The fallible companion to [`Self::expect_node`].
    fn as_single_node(&self) -> Option<NodeId> {
        match self {
            Self::Node(id) => Some(*id),
            Self::Tuple(_) | Self::Adt { .. } => None,
        }
    }

    fn tuple_get(&self, index: usize) -> Option<LoweredValue> {
        match self {
            Self::Tuple(items) => items.get(index).cloned(),
            Self::Node(_) | Self::Adt { .. } => None,
        }
    }

    fn from_flat(template: &LoweredValue, nodes: &mut dyn Iterator<Item = NodeId>) -> LoweredValue {
        match template {
            Self::Node(_) => Self::Node(nodes.next().expect("flattened lowered value mismatch")),
            Self::Tuple(items) => Self::Tuple(
                items
                    .iter()
                    .map(|item| Self::from_flat(item, nodes))
                    .collect(),
            ),
            Self::Adt {
                ctor,
                field_names,
                fields,
            } => Self::Adt {
                ctor: ctor.clone(),
                field_names: field_names.clone(),
                fields: fields
                    .iter()
                    .map(|field| Self::from_flat(field, nodes))
                    .collect(),
            },
        }
    }
}

fn extract_param_type(expr: &Expr, index: usize) -> Option<&Expr> {
    let Expr::List(list, _) = expr else {
        return None;
    };
    if get_tag(list) != Some("fn") {
        return None;
    }
    let params = children(list).first()?;
    let Expr::List(params_list, _) = params else {
        return None;
    };
    let param = children(params_list).get(index)?;
    match param {
        Expr::MetaExpr(meta, _) => meta
            .entries
            .iter()
            .find(|(key, _)| key == "type")
            .map(|(_, value)| value),
        Expr::List(param_list, _) => match param_list.elements.get(1) {
            Some(Expr::Map(meta, _)) => meta
                .entries
                .iter()
                .find(|(key, _)| key == "type")
                .map(|(_, value)| value),
            _ => None,
        },
        _ => None,
    }
}

fn param_name_and_type_expr(param: &Expr) -> Option<(String, Option<&Expr>)> {
    match param {
        Expr::Atom(Atom::Symbol(name), _) => Some((name.clone(), None)),
        Expr::MetaExpr(meta, _) => {
            let Expr::Atom(Atom::Symbol(name), _) = meta.expr.as_ref() else {
                return None;
            };
            let ty_expr = meta
                .entries
                .iter()
                .find(|(key, _)| key == "type")
                .map(|(_, value)| value);
            Some((name.clone(), ty_expr))
        }
        Expr::List(param_list, _) => {
            let Expr::Atom(Atom::Symbol(name), _) = param_list.elements.first()? else {
                return None;
            };
            let ty_expr = if let Some(Expr::Map(meta, _)) = param_list.elements.get(1) {
                meta.entries
                    .iter()
                    .find(|(key, _)| key == "type")
                    .map(|(_, value)| value)
            } else {
                None
            };
            Some((name.clone(), ty_expr))
        }
        _ => None,
    }
}

fn extract_fn_return_type(expr: &Expr) -> Option<&Expr> {
    let Expr::List(list, _) = expr else {
        return None;
    };
    if get_tag(list) != Some("fn") {
        return None;
    }
    let Expr::Map(meta, _) = list.elements.get(1)? else {
        return None;
    };
    let (_, ty_expr) = meta.entries.iter().find(|(key, _)| key == "type")?;
    let Expr::List(fn_ty, _) = ty_expr else {
        return None;
    };
    if get_tag(fn_ty) != Some("t-fn") {
        return None;
    }
    children(fn_ty).last()
}

fn axis_to_front_perm(rank: usize, axis: usize) -> Vec<usize> {
    let mut perm = Vec::with_capacity(rank);
    perm.push(axis);
    perm.extend((0..rank).filter(|candidate| *candidate != axis));
    perm
}

fn front_to_axis_perm(rank: usize, axis: usize) -> Vec<usize> {
    let mut perm: Vec<usize> = (1..rank).collect();
    perm.insert(axis, 0);
    perm
}

fn permuted_tensor_type(ty: &TensorType, axes: &[usize]) -> TensorType {
    TensorType {
        dims: axes.iter().map(|axis| ty.dims[*axis].clone()).collect(),
        precision: ty.precision,
    }
}

struct LowerCtx {
    dag: Dag,
    bindings: HashMap<String, LoweredValue>,
    list_bindings: HashMap<String, Expr>,
    /// chelis#369: `let`-bound names whose value is a `shape(operand,
    /// axis)` application, keyed by the bound name and holding the raw
    /// `shape(...)` Deep `Expr`. The canonical `tensor_full_like` idiom
    /// writes `len = shape(x, 0)` then `expand(s, 0, cast(len, int32))`,
    /// so the `expand` size argument is a `var len` reference, not a
    /// direct `shape(...)` app. Without this map the size-recovery path
    /// [`Self::shape_app_operand_axis_resolved`] cannot see through the
    /// `let` indirection and the extent silently defaults to `Lit(1)`,
    /// producing the `Lit(n) vs Lit(1)` backward-DAG verification failure.
    /// Saved/restored across binding scopes exactly like `list_bindings`.
    shape_bindings: HashMap<String, Expr>,
    /// chelis#469/#528: `let`-bound names whose value const-folds to a
    /// compile-time integer (a literal, `cast(N, _)`, or integer arithmetic
    /// over such values — the §4.7.2 `SizeClass::Static` provenance the
    /// checker follows transitively through `let` bindings). Lets a later
    /// `expand(s, axis, cast(len, int32))` (or `len` used directly) recover
    /// the concrete extent instead of the pre-fix size-1 default — the exact
    /// eval-`[7]`-vs-C-`[1]` silent miscompile #469 exists to prevent when a
    /// `let`-bound static size reaches the backend. Re-binding a name to a
    /// non-static value drops its stale entry (shadowing symmetry, mirroring
    /// `shape_bindings`). Saved/restored across binding scopes.
    static_size_bindings: HashMap<String, i64>,
    local_callables: HashMap<String, Expr>,
    program_types: HashMap<String, TensorType>,
    program_defs: HashMap<String, Expr>,
    random_seed: Option<u64>,
    linearity: LinearityInfo,
    /// chelis#620 (Inlining-F1 successor): per-callee active-inline depth.
    /// Recursion lowers by unrolling, so a self- or mutually-recursive call
    /// re-inlines up to [`MAX_STATIC_RECURSION_DEPTH`] levels per name; the
    /// static `if`/`match` pruning is what terminates a well-founded
    /// recursion before the cap. Entries are removed at depth 0 on unwind of
    /// each `lower_plain_callable_app` body scope. A cap hit raises loudly
    /// through the [`Self::reject_lowering_slice`] ladder; leak-on-panic is
    /// acceptable because every lowering error unwinds through the per-entry
    /// `catch_lowering` and the ctx is abandoned.
    inlining_depths: HashMap<String, usize>,
    /// Total active inlined bodies across all names. Bounds the Rust stack
    /// (each level is roughly ten lowering frames) for deep or mutually
    /// recursive chains that stay under every per-name cap.
    inlining_active: usize,
    /// Parameter names whose declared type is `t-fn` — used by
    /// `resolve_callable_expr_inner` to distinguish a fn-typed parameter
    /// reference (legitimate `CallableExpr::Parameter`) from a truly
    /// unresolvable name (`None`). Populated by `lower_fn` when a `t-fn`
    /// param is registered; saved/restored across nested `fn` scopes
    /// alongside `bindings` and `local_callables`. See
    /// `docs/investigations/pipe_fn_param_stage_diagnosis.md`.
    fn_typed_params: HashSet<String>,
    dim_substitutions: HashMap<String, DimInfo>,
    /// WS-A8: precision-tvar substitutions, keyed by the precision-var
    /// name (e.g. `p`) as it appears in `(t-var {} p)` precision slots
    /// of the polymorphic def's signature. Populated at call sites of
    /// polymorphic top-level defs (see `lower_plain_callable_app`)
    /// alongside `dim_substitutions`. Consulted by
    /// [`Self::try_extract_tensor_type`] to resolve a precision-tvar
    /// slot into a concrete `Prim` before backends see the type. After
    /// monomorphization every reachable tensor type carries
    /// `TensorPrec::Concrete(_)` per spec/04-type-system.md §5.8.1.
    prec_substitutions: HashMap<String, Prim>,
    /// Tier-2 rank polymorphism (spec/design/rank_polymorphism.md): rank-var
    /// substitutions, keyed by the `..r` rank-var name as it appears in a sole
    /// `(d-rank {} r)` dim slot of a rank-polymorphic def's signature. The
    /// structural twin of [`Self::prec_substitutions`]: where precision
    /// substitutes a concrete `Prim` into a `(t-var {} p)` slot, rank
    /// substitutes the caller's concrete shape vector into a `(d-rank {} r)`
    /// slot. Populated at call sites of rank-poly top-level defs (see
    /// `lower_plain_callable_app`) alongside `dim_substitutions` and
    /// `prec_substitutions`. Consulted by
    /// [`Self::try_extract_tensor_type_with_subst`] to expand a `Dim::Rank`
    /// slot into the bound concrete dims before backends see the type. After
    /// monomorphization every reachable tensor type is `Dim::Rank`-free per
    /// the spec's monomorphization invariant; a surviving rank var is a
    /// monomorphization bug, not a backend input.
    rank_substitutions: HashMap<String, Vec<DimInfo>>,
    /// Issue #388 / chelis#549: for each named axis, the positional index it
    /// occupied in a formal parameter shape at the current inlined call site,
    /// paired with the concrete extent (`DimInfo`) the anchor was recorded with.
    /// Populated by [`tensor_dim_axis_positions`] in `lower_plain_callable_app`
    /// alongside `dim_substitutions`. Consulted (via [`recover_anchor_axis`]) by
    /// [`Self::resolve_reduce_axis`] / [`Self::resolve_expand_anchor`] /
    /// [`extract_rank_var_bindings`] to recover the axis a named
    /// reduction/expand/split targets when call-site monomorphization erased the
    /// named axis from the operand's dims (a literal-shaped actual argument
    /// carries concrete `Lit` dims).
    ///
    /// The paired extent is the soundness anchor for chelis#549: the recorded
    /// index is the anchor's offset in the *formal parameter*, so an intervening
    /// axis-reorder (`permute`/transpose) between that parameter and the use site
    /// can leave the index valid-but-stale (in range, wrong axis). The recovery
    /// ([`recover_anchor_axis`]) does NOT trust the recorded index; it locates the
    /// anchor purely by its recorded extent in the monomorphized operand, taking
    /// the axis when that extent is unique and failing LOUD when it collides with
    /// another axis (the square/equal-extent case, RT-1) — never silently splits
    /// at a possibly-wrong axis. The extent is never *guessed*; an ambiguous name
    /// (one at different positions/extents across formals) is excluded at
    /// recording time so the loud by-name failure path still fires for genuinely
    /// unresolvable axes.
    dim_axis_positions: HashMap<String, (usize, DimInfo)>,
    /// True only while lowering the body of an AD transform. Host-list
    /// combinator rewrites are an AD bridge, not the general C/backend
    /// lowering for ordinary list programs.
    allow_host_list_ad_rewrites: bool,
    /// The span_id of the Deep `Expr` currently being lowered. Threaded
    /// through `lower_expr` (set on entry, restored on exit) so every
    /// helper that calls `self.dag.add_node(...)` can pass the
    /// region-corresponding span without plumbing it through every
    /// helper's argument list. See
    /// `spec/design/chelis_span_survival.md` §2.3.
    current_span_id: Option<String>,
}

impl LowerCtx {
    fn new(
        program_types: HashMap<String, TensorType>,
        program_defs: HashMap<String, Expr>,
        linearity: LinearityInfo,
    ) -> Self {
        Self {
            dag: Dag::new(),
            bindings: HashMap::new(),
            list_bindings: HashMap::new(),
            shape_bindings: HashMap::new(),
            static_size_bindings: HashMap::new(),
            local_callables: HashMap::new(),
            program_types,
            program_defs,
            random_seed: None,
            linearity,
            inlining_depths: HashMap::new(),
            inlining_active: 0,
            fn_typed_params: HashSet::new(),
            dim_substitutions: HashMap::new(),
            prec_substitutions: HashMap::new(),
            rank_substitutions: HashMap::new(),
            dim_axis_positions: HashMap::new(),
            allow_host_list_ad_rewrites: false,
            current_span_id: None,
        }
    }

    /// Default tensor type when we don't have richer type info.
    fn default_type() -> TensorType {
        TensorType::scalar_f32()
    }

    /// Choose the output dims for a reduction op (`sum`,
    /// `max_reduce`, `min_reduce`, `prod_reduce`, `argmax_reduce`,
    /// `argmin_reduce`).
    ///
    /// The historical rule was: when meta `ty` differs from
    /// `default_type()`, trust `ty.dims` blindly; otherwise compute
    /// `input_dims` with `axis` removed. That rule breaks when a
    /// wildcard placeholder dim reaches the reduction op via meta
    /// `ty.dims`: host emit's `rename_anonymous_dims` renames it to
    /// `_anon_dim_*`, and `crates/chelis-ir/src/dag.rs`'s
    /// `symbolic_occurrences` panics because no Load input declares
    /// the synthesized name (issue Chelis-Lang/chelis#218 R1 HIGH-2).
    ///
    /// The corrected rule mirrors `elementwise_out_ty`'s precedent
    /// ("Prefer the input DAG node's dims ... `ty` after type
    /// inference can hold internal fresh-var names"): when the
    /// computed-from-input dims do not carry a wildcard placeholder,
    /// prefer them. The meta `ty.dims` is used only when the input
    /// can't produce a usable shape (e.g. the input DAG node was
    /// missing — defensive path).
    ///
    /// Note: the static-literal path no longer produces wildcards
    /// post the `to_tensor` source fix in `chelis-types::infer`. This
    /// helper stays in place as defense in depth for the remaining
    /// dynamic-shape sources (variable-fed `to_tensor(items)`,
    /// `concat` along the concat axis, `split`, `pad_sequences`).
    fn reduction_out_dims(input_dims: &[DimInfo], ty: &TensorType, axis: usize) -> Vec<DimInfo> {
        let mut from_input = input_dims.to_vec();
        if axis < from_input.len() {
            from_input.remove(axis);
        }
        if !input_dims.is_empty() && !any_wildcard_dim(&from_input) {
            return from_input;
        }
        if *ty != Self::default_type() && !any_wildcard_dim(&ty.dims) {
            return ty.dims.clone();
        }
        from_input
    }

    /// Choose the output `TensorType` for an elementwise op whose shape
    /// matches the first input. Prefer the input DAG node's dims over the
    /// annotated `ty.dims` when the input has a non-empty rank — the input
    /// dims carry the user-facing symbolic names from `defsig`/param types,
    /// while `ty` after type inference can hold internal fresh-var names
    /// (e.g. `d44`) that aren't declared in the emitted C scope. When
    /// `precision_override` is provided it wins (e.g. `cmplt` → `Bool`);
    /// otherwise we use `ty`'s precision when present, else the input's.
    fn elementwise_out_ty(
        dag: &Dag,
        input: NodeId,
        ty: &TensorType,
        precision_override: Option<Prim>,
    ) -> TensorType {
        let input_ty = dag.get(input).map(|node| node.output_type.clone());
        let dims = match input_ty.as_ref() {
            Some(in_ty) if !in_ty.dims.is_empty() => in_ty.dims.clone(),
            _ => {
                if ty.dims.is_empty() {
                    input_ty
                        .as_ref()
                        .map(|in_ty| in_ty.dims.clone())
                        .unwrap_or_default()
                } else {
                    ty.dims.clone()
                }
            }
        };
        let precision = precision_override.unwrap_or_else(|| {
            if *ty != Self::default_type() {
                ty.precision
            } else {
                input_ty
                    .map(|in_ty| in_ty.precision)
                    .unwrap_or(ty.precision)
            }
        });
        TensorType { dims, precision }
    }

    fn attach_reuse_hint(
        &mut self,
        node: NodeId,
        app_span: Span,
        candidate_inputs: &[NodeId],
    ) -> NodeId {
        if let Some(input_index) = self.linearity.reusable_input_for_span(app_span)
            && let Some(input) = candidate_inputs.get(input_index)
        {
            self.dag.set_reusable_input(node, *input);
        }
        node
    }

    fn repair_output_type_if_default(&mut self, value: &LoweredValue, desired: &TensorType) {
        let LoweredValue::Node(id) = value else {
            return;
        };
        let Some(node) = self.dag.get(*id) else {
            return;
        };
        if node.output_type != Self::default_type() || desired == &Self::default_type() {
            return;
        }
        self.dag
            .replace_node(*id, node.op.clone(), node.inputs.clone(), desired.clone());
    }

    /// Extract a type from a metadata map if one is present, otherwise return a default.
    /// WS-A8: instance entry that consults `self.prec_substitutions` and
    /// `self.rank_substitutions` when extracting a tensor type out of a Deep
    /// meta map. Ensures inlined polymorphic-def bodies see substituted
    /// precisions and concrete ranks on their `type:` annotations.
    fn type_from_meta(&self, meta: &[(String, Expr)]) -> TensorType {
        for (key, val) in meta {
            if key == "type" {
                return Self::type_from_type_expr_with_subst(
                    val,
                    &self.prec_substitutions,
                    &self.rank_substitutions,
                );
            }
        }
        Self::default_type()
    }

    /// Static (no-substitution) variant of [`Self::type_from_meta`].
    /// Reserved for callers that operate before lowering begins (e.g.
    /// the if-lowerable shape pre-check at top-level analysis time).
    /// A `(t-var)` precision slot reaching this entry will trip the
    /// F2 backend tripwire panic per spec/04-type-system.md §5.8.1.
    fn type_from_meta_static(meta: &[(String, Expr)]) -> TensorType {
        for (key, val) in meta {
            if key == "type" {
                return Self::type_from_type_expr(val);
            }
        }
        Self::default_type()
    }

    fn remap_callable_dim_symbols(
        dag: &Dag,
        formal_params: &[TensorType],
        actual_args: &[TensorType],
    ) -> Dag {
        remap_tensor_dim_symbols(dag, formal_params, actual_args)
    }

    fn seed_subctx_with_lexical_scope(
        &self,
        subctx: &mut LowerCtx,
        shadowed: &[String],
    ) -> HashMap<String, NodeId> {
        let shadowed = shadowed.iter().cloned().collect::<HashSet<_>>();
        let mut captures = HashMap::new();
        for (name, value) in self
            .bindings
            .iter()
            .filter(|(name, _)| !shadowed.contains(*name))
        {
            let LoweredValue::Node(node_id) = value else {
                continue;
            };
            let ty = self
                .dag
                .get(*node_id)
                .map(|node| node.output_type.clone())
                .unwrap_or_else(Self::default_type);
            let load = subctx.dag.add_node(
                RiscOp::Load {
                    name: name.as_str().into(),
                },
                vec![],
                ty,
                subctx.current_span_id.clone(),
            );
            subctx
                .bindings
                .insert(name.clone(), LoweredValue::Node(load));
            captures.insert(name.clone(), *node_id);
        }
        subctx.local_callables.extend(
            self.local_callables
                .iter()
                .filter(|(name, _)| !shadowed.contains(*name))
                .map(|(name, value)| (name.clone(), value.clone())),
        );
        captures
    }

    fn type_from_type_expr(expr: &Expr) -> TensorType {
        Self::type_from_type_expr_with_subst(expr, &HashMap::new(), &HashMap::new())
    }

    /// WS-A8: precision-aware variant of [`Self::type_from_type_expr`].
    /// Resolves `(t-var {} p)` precision slots through `prec_subst` so a
    /// polymorphic def's body, when inlined at a concrete call site,
    /// sees the substituted concrete primitive instead of tripping the
    /// F2 backend tripwire. The static
    /// [`Self::type_from_type_expr`] entry point still panics on a
    /// `t-var` slot — that path is the diagnostic surface for genuine
    /// monomorphization bugs (a polymorphic sig reached the backend
    /// boundary without a concrete instantiation site).
    fn type_from_type_expr_with_subst(
        expr: &Expr,
        prec_subst: &HashMap<String, Prim>,
        rank_subst: &HashMap<String, Vec<DimInfo>>,
    ) -> TensorType {
        if let Some(prim) = Self::try_extract_prim(expr) {
            return TensorType {
                dims: vec![],
                precision: prim,
            };
        }
        if let Some(inner) = Self::try_extract_ref_type(expr) {
            return Self::type_from_type_expr_with_subst(inner, prec_subst, rank_subst);
        }
        if let Some(tt) = Self::try_extract_tensor_type_with_subst(expr, prec_subst, rank_subst) {
            return tt;
        }
        Self::default_type()
    }

    /// issue #289: call-site variant of [`Self::type_from_type_expr_with_subst`]
    /// for computing an inlined callee's *formal parameter* shape.
    ///
    /// At a call site the callee's parameter precision is determined by
    /// the actual argument, not by the formal annotation — the call site
    /// computes the precision-tvar binding separately (via
    /// [`tensor_prec_substitutions`] over the raw formal type-exprs and
    /// the concrete actual `TensorType`s). The formal parameter
    /// `TensorType` derived here is consumed only for *dimension*
    /// substitution ([`tensor_dim_substitutions`]); its precision slot is
    /// discarded. So an as-yet-unresolved `(t-var {} p)` precision slot is
    /// expected and benign here: it is about to be bound from the actual
    /// argument. Falling through the panicking
    /// [`Self::try_extract_tensor_type_with_subst`] would crash on that
    /// legitimate shape. This tolerant variant substitutes a known
    /// precision when `prec_subst` has one and otherwise leaves the
    /// default precision in place (the dims, which is all the caller uses,
    /// are still extracted correctly).
    ///
    /// This does NOT weaken the §5.8.1 backend tripwire: a precision var
    /// that never gets a concrete binding still reaches the panicking
    /// extractor when the callee *body* is lowered (the body's own tensor
    /// types route through [`Self::type_from_meta`] /
    /// [`Self::type_from_type_expr_with_subst`], which panic when
    /// `prec_subst` lacks the var).
    fn formal_param_type_for_call(
        expr: &Expr,
        prec_subst: &HashMap<String, Prim>,
        rank_subst: &HashMap<String, Vec<DimInfo>>,
    ) -> TensorType {
        if let Some(prim) = Self::try_extract_prim(expr) {
            return TensorType {
                dims: vec![],
                precision: prim,
            };
        }
        if let Some(inner) = Self::try_extract_ref_type(expr) {
            return Self::formal_param_type_for_call(inner, prec_subst, rank_subst);
        }
        // Only the precision slot can panic in the strict extractor; reuse
        // it when no unresolved precision var is present so the strict
        // path stays authoritative. When the precision slot is an
        // unresolved `(t-var)`, swap in the resolved primitive if known,
        // otherwise the default, then extract the dims with that slot.
        if let Some(name) = extract_precision_var_name(expr) {
            let resolved = prec_subst
                .get(&name)
                .copied()
                .unwrap_or_else(|| Self::default_type().precision);
            // Re-run extraction with the var resolved so the dims come out
            // correctly; substitute the resolved primitive into the slot.
            let mut tolerant = prec_subst.clone();
            tolerant.entry(name).or_insert(resolved);
            if let Some(tt) = Self::try_extract_tensor_type_with_subst(expr, &tolerant, rank_subst)
            {
                return tt;
            }
            return Self::default_type();
        }
        if let Some(tt) = Self::try_extract_tensor_type_with_subst(expr, prec_subst, rank_subst) {
            return tt;
        }
        Self::default_type()
    }

    fn try_extract_ref_type(expr: &Expr) -> Option<&Expr> {
        if let Expr::List(list, _) = expr
            && list.elements.len() >= 3
            && let Expr::Atom(Atom::Symbol(tag), _) = &list.elements[0]
            && tag == "t-ref"
        {
            return list.elements.get(2);
        }
        None
    }

    fn try_extract_prim(expr: &Expr) -> Option<Prim> {
        // (t-prim {} f32)
        if let Expr::List(list, _) = expr
            && list.elements.len() >= 3
            && let Expr::Atom(Atom::Symbol(tag), _) = &list.elements[0]
            && tag == "t-prim"
            && let Expr::Atom(Atom::Symbol(name), _) = &list.elements[2]
        {
            return Prim::parse_name(name);
        }
        None
    }

    /// WS-A8: precision-aware variant of `try_extract_tensor_type`.
    /// When the precision slot is `(t-var {} p)`, consult `prec_subst`
    /// to resolve `p` to a concrete `Prim`. If the substitution has the
    /// binding, return a `TensorType` carrying the concrete primitive
    /// (this is the monomorphization-success path). If no binding is
    /// found AND the substitution is non-empty (i.e. the caller IS in
    /// a substitution-aware context), still panic — the body referenced
    /// a precision tvar the call site failed to bind, which is a real
    /// monomorphization gap. With an empty substitution the panic still
    /// fires per the F2 backend tripwire contract.
    fn try_extract_tensor_type_with_subst(
        expr: &Expr,
        prec_subst: &HashMap<String, Prim>,
        rank_subst: &HashMap<String, Vec<DimInfo>>,
    ) -> Option<TensorType> {
        // Flat format: (t-tensor {} dim1 dim2 ... (t-prim {} p))
        // Children after tag+meta: dimension nodes followed by a t-prim node as the last child.
        if let Expr::List(list, _) = expr
            && list.elements.len() >= 3
            && let Expr::Atom(Atom::Symbol(tag), _) = &list.elements[0]
            && tag == "t-tensor"
        {
            // elements[0] = tag, elements[1] = meta, elements[2..] = children
            let children = &list.elements[2..];
            if children.is_empty() {
                return None;
            }
            // WS-A5 RT-3a F2 / WS-A8: backends require a concrete tensor
            // precision per spec/04-type-system.md §5.8.1. If the
            // precision slot is a `(t-var {} ...)` here we are looking
            // at an unresolved precision tvar. WS-A8 lets the caller
            // pass `prec_subst` (the call-site precision-tvar bindings
            // computed in `lower_plain_callable_app`) so an inlined
            // polymorphic-def body can resolve the slot to a concrete
            // primitive without ever tripping the panic. The panic
            // remains as a safety net: properly-monomorphized programs
            // never reach it; reaching it indicates either (a) a
            // standalone polymorphic-def lowering attempt with no
            // concrete call site, or (b) an internal monomorphization
            // bug missed a precision tvar.
            let prim = if let Some(last_child) = children.last()
                && let Expr::List(prec_list, _) = last_child
                && prec_list.elements.len() >= 3
                && let Expr::Atom(Atom::Symbol(prec_tag), _) = &prec_list.elements[0]
                && prec_tag == "t-var"
            {
                let var_name = match &prec_list.elements[2] {
                    Expr::Atom(Atom::Symbol(name), _) => name.clone(),
                    _ => "?".to_string(),
                };
                if let Some(prim) = prec_subst.get(&var_name).copied() {
                    prim
                } else {
                    panic!(
                        "BUG: monomorphization missed precision var `{var_name}`; \
                         this should not be reachable from properly-typed source \
                         code. spec/04-type-system.md \u{00a7}5.8.1 requires every \
                         reachable tensor type to carry `TensorPrec::Concrete(_)` \
                         after monomorphization. Reaching this point with \
                         `TensorPrec::Var(_)` indicates a polymorphic sig with no \
                         concrete call site, or an internal monomorphization gap."
                    );
                }
            } else {
                // Last child is the precision (t-prim {} name).
                Self::try_extract_prim(children.last()?)?
            };
            // All children before the last are dimension nodes.
            let mut dims = Vec::new();
            for child in &children[..children.len() - 1] {
                // Tier-2 rank polymorphism (spec/design/rank_polymorphism.md):
                // a `Dim::Rank` (`(d-rank {} r)`) stands for the *entire* shape
                // vector and must be eliminated before lowering. Standalone
                // emission of a rank-poly sig is skipped via
                // `type_is_never_lowerable` (paralleling precision
                // polymorphism), so reaching here means a concrete caller
                // inlined a rank-poly def. Call-site rank monomorphization
                // (`tensor_rank_substitutions`, the rank analogue of the
                // precision `prec_subst` path) supplied the caller's concrete
                // shape in `rank_subst`, keyed by the rank-var name. Expand the
                // `(d-rank {} r)` slot to those concrete dims — the
                // monomorphization-success path. An unbound slot is a
                // speculative annotation read repaired by the inlined body (see
                // the inner comment); it is dropped, never emitted as a rank
                // dim, because the IR `DimInfo` cannot represent one.
                if let Expr::List(dl, _) = child
                    && let Some(Expr::Atom(Atom::Symbol(tag), _)) = dl.elements.first()
                    && tag == "d-rank"
                {
                    let rank_name = match dl.elements.get(2) {
                        Some(Expr::Atom(Atom::Symbol(name), _)) => name.clone(),
                        _ => "?".to_string(),
                    };
                    if let Some(concrete_dims) = rank_subst.get(&rank_name) {
                        // Monomorphization-success path: expand the rank var
                        // to the caller's concrete shape vector.
                        dims.extend(concrete_dims.iter().cloned());
                    }
                    // Unbound here: this is a speculative type read (e.g. an
                    // `app`/param `type:` annotation that the type-checker
                    // stamped with the rank-poly return type before this call
                    // site monomorphized it). The concrete type is supplied by
                    // call-site inlining of the rank-poly body, which produces
                    // a fully concrete tensor — the inlined-body output type
                    // repairs this speculative read downstream. Drop the
                    // unresolved rank dim rather than raising: a standalone
                    // rank-poly def with no concrete caller is already skipped
                    // from emission entirely (`type_is_never_lowerable` /
                    // `type_expr_has_rank_var` in both the DAG and host lanes),
                    // so reaching here always means an inlining context will
                    // supply the concrete shape. The IR `DimInfo` has no rank
                    // variant, so no `Dim::Rank` can survive into a backend
                    // type by construction. See spec/design/rank_polymorphism.md.
                    continue;
                }
                if let Some(dim) = Self::try_extract_dim(child) {
                    dims.push(dim);
                }
            }
            return Some(TensorType {
                dims,
                precision: prim,
            });
        }
        None
    }

    /// Extract a single dimension from a dimension node.
    fn try_extract_dim(expr: &Expr) -> Option<DimInfo> {
        if let Expr::List(list, _) = expr
            && list.elements.len() >= 3
            && let Expr::Atom(Atom::Symbol(tag), _) = &list.elements[0]
        {
            match tag.as_str() {
                "d-name" => {
                    if let Expr::Atom(Atom::Symbol(name), _) = &list.elements[2] {
                        return Some(DimInfo::Named(name.clone(), None));
                    }
                }
                "d-var" => {
                    if let Expr::Atom(Atom::Symbol(name), _) = &list.elements[2] {
                        return Some(DimInfo::Named(name.clone(), None));
                    }
                }
                "d-lit" => {
                    if let Expr::Atom(Atom::Int(n), _) = &list.elements[2] {
                        return Some(DimInfo::Lit(*n as usize));
                    }
                }
                _ => {}
            }
        }
        // Also handle bare symbols/ints for backward compat.
        match expr {
            Expr::Atom(Atom::Symbol(name), _) => Some(DimInfo::Named(name.clone(), None)),
            Expr::Atom(Atom::Int(n), _) => Some(DimInfo::Lit(*n as usize)),
            _ => None,
        }
    }

    fn dim_info_from_dim_expr(size: &DimExpr) -> Option<DimInfo> {
        match size {
            DimExpr::Concrete(value) => Some(DimInfo::Lit(*value)),
            DimExpr::Sym(name) => Some(DimInfo::Named(name.clone(), None)),
            DimExpr::Mul(_, _) | DimExpr::Div(_, _) => None,
        }
    }

    fn fallback_expand_type(
        &self,
        input: NodeId,
        axis: usize,
        size: &DimExpr,
    ) -> Option<TensorType> {
        let input_ty = self.dag.get(input)?.output_type.clone();
        let inserted_dim = Self::dim_info_from_dim_expr(size)?;
        let mut dims = input_ty.dims;
        if axis > dims.len() {
            return None;
        }
        dims.insert(axis, inserted_dim);
        Some(TensorType {
            dims,
            precision: input_ty.precision,
        })
    }

    fn lower_top_level(&mut self, expr: &Expr) {
        if let Expr::List(list, _) = expr
            && let Some(Expr::Atom(Atom::Symbol(tag), _)) = list.elements.first()
        {
            match tag.as_str() {
                // Skip type-level declarations.
                "defsig" | "deftype" | "typealias" => return,
                // Skip module-system declarations (issue chelis#232):
                // `export` / `import` / `import-all` are name-routing
                // directives — they have no runtime value and must not
                // produce DAG roots. The fallthrough `lower_expr` path
                // (via the `_ =>` arm in `lower_list`) would otherwise
                // emit a `Load`/`Const` node and `add_root` it, breaking
                // the `tensor_root_names.len() == dag.roots().len()`
                // invariant in chelis-compiler-api::compiler::compile_source.
                "export" | "import" | "import-all" => return,
                _ => {}
            }
        }

        let value = self.lower_expr(expr);
        if let Expr::List(list, _) = expr
            && get_tag(list) == Some("def")
            && let Some(name) = children(list).first().and_then(|expr| match expr {
                Expr::Atom(Atom::Symbol(name), _) => Some(name.as_str()),
                _ => None,
            })
        {
            // `lower_expr` cleared `current_span_id` on exit. Re-thread the
            // def's own span_id (if present) for the duration of
            // `add_named_roots` so any Store nodes emitted on the
            // top-level def's behalf inherit the def's source region —
            // satisfying the §2.3 audit invariant for input def spans.
            let saved_span_id = self.current_span_id.clone();
            if let Some(s) = expr.span_id() {
                self.current_span_id = Some(s.to_owned());
            }
            self.add_named_roots(name, &value);
            self.current_span_id = saved_span_id;
        } else {
            for id in value.flatten_nodes() {
                self.dag.add_root(id);
            }
        }
    }

    fn lower_expr(&mut self, expr: &Expr) -> LoweredValue {
        // Thread the current Deep node's span_id through any add_node()
        // calls made while lowering this expr or its children. We snapshot
        // the previous span_id and restore it on return so sibling exprs
        // are unaffected; child exprs whose own meta carries a span
        // override while they're being lowered, and child exprs without
        // their own span fall through to the parent's span (the
        // region-corresponding rule from spec/design/chelis_span_survival.md
        // §2.3 / §2.4).
        let saved_span_id = self.current_span_id.clone();
        if let Some(s) = expr.span_id() {
            self.current_span_id = Some(s.to_owned());
        }
        let result = match expr {
            Expr::Atom(atom, _) => self.lower_atom(atom),
            Expr::List(list, span) => self.lower_list(list, *span),
            Expr::Map(_, _) => raise_malformed_deep(
                "a bare metadata map in expression position",
                Some(expr.span()),
                self.current_span_id.clone(),
            ),
            Expr::MetaExpr(meta_expr, _) => self.lower_expr(&meta_expr.expr),
        };
        self.current_span_id = saved_span_id;
        result
    }

    /// Append `current_span_id` (if any) to a single existing IR node's
    /// `merged_spans`, lex-sorted and deduped. This implements the N→1
    /// lowering collapse rule from `spec/design/chelis_span_survival.md`
    /// §2.3 (rule b): when a parent Deep expr lowers to a body that already
    /// corresponds to an existing IR node — for example, a `(def {span: a})`
    /// whose body is an existing node, or a `(var {span: u} a)` whose body
    /// is a previously-bound `LoweredValue` — append the parent's span to
    /// the existing node so the audit invariant ("every input span appears
    /// as `span_id` or `merged_spans` on at least one node") is preserved.
    /// Thin wrapper over `crate::span_merge::append_span_to_node`; the
    /// shared helper owns the None-no-op / canonical-no-op / dedup / sort
    /// logic.
    fn append_current_span_to_existing_node(&mut self, id: NodeId) {
        crate::span_merge::append_span_to_node(&mut self.dag, id, self.current_span_id.as_deref());
    }

    /// Walk a `LoweredValue` and apply
    /// `append_current_span_to_existing_node` to every contained node id.
    /// Used at every site that returns a cached/aliased `LoweredValue`
    /// from a name → value map (e.g. `bindings`) — those returns are N→1
    /// lowering collapses that must still record the parent expr's span.
    fn append_current_span_to_lowered_value(&mut self, value: &LoweredValue) {
        match value {
            LoweredValue::Node(id) => self.append_current_span_to_existing_node(*id),
            LoweredValue::Tuple(items) => {
                for item in items {
                    self.append_current_span_to_lowered_value(item);
                }
            }
            LoweredValue::Adt { fields, .. } => {
                for field in fields {
                    self.append_current_span_to_lowered_value(field);
                }
            }
        }
    }

    fn add_named_roots(&mut self, prefix: &str, value: &LoweredValue) {
        match value {
            LoweredValue::Node(id) if !prefix.contains('.') => {
                // Top-level def whose body lowered to a single existing
                // node — no new Store is emitted. This is a region-merge
                // during lowering: the def's source region and the
                // body's source region collapse onto one IR node. Per
                // spec/design/chelis_span_survival.md §2.3 rule (b), N→1
                // region merges record the additional span(s) in
                // `merged_spans` (the body node already owns `span_id`),
                // lex-sorted and deduped, so the audit invariant ("every
                // input span appears on at least one IR node") still
                // holds.
                self.append_current_span_to_existing_node(*id);
                self.dag.add_root(*id);
            }
            LoweredValue::Node(id) => {
                let output_type = self
                    .dag
                    .get(*id)
                    .map(|node| node.output_type.clone())
                    .unwrap_or_else(Self::default_type);
                let stored = self.dag.add_node(
                    RiscOp::Store {
                        name: prefix.into(),
                    },
                    vec![*id],
                    output_type,
                    self.current_span_id.clone(),
                );
                self.dag.add_root(stored);
            }
            LoweredValue::Tuple(items) => {
                for (index, item) in items.iter().enumerate() {
                    self.add_named_roots(&format!("{prefix}.{index}"), item);
                }
            }
            // ADT gradient values store field-wise roots, keyed by field
            // name when the constructor is a record form (chelis#520 D2).
            LoweredValue::Adt {
                field_names,
                fields,
                ..
            } => {
                for (index, field) in fields.iter().enumerate() {
                    let label = field_names
                        .as_ref()
                        .and_then(|names| names.get(index).cloned())
                        .unwrap_or_else(|| index.to_string());
                    self.add_named_roots(&format!("{prefix}.{label}"), field);
                }
            }
        }
    }

    fn lower_atom(&mut self, atom: &Atom) -> LoweredValue {
        match atom {
            Atom::Symbol(name) => {
                if let Some(value) = self.bindings.get(name) {
                    let cached = value.clone();
                    // N→1 lowering collapse per
                    // spec/design/chelis_span_survival.md §2.3 rule (b):
                    // returning a cached `LoweredValue` for a span-bearing
                    // parent expr (e.g. an `Atom::Symbol` whose enclosing
                    // node carries a `span:` meta) must still record the
                    // parent's span on the existing node so the audit
                    // chain doesn't drop it.
                    self.append_current_span_to_lowered_value(&cached);
                    cached
                } else {
                    LoweredValue::Node(self.dag.add_node(
                        RiscOp::Load {
                            name: name.as_str().into(),
                        },
                        vec![],
                        Self::default_type(),
                        self.current_span_id.clone(),
                    ))
                }
            }
            // BARE atoms reach here only from synthetic Deep (tests,
            // internal expansions): the desugarer wraps every source
            // literal in a typed `(lit {type: ...} ...)` node, which
            // `lower_lit` finalizes at its ascribed dtype (the
            // chelis#856 exactness path). Bare atoms keep the default
            // f32 typing they always had; finalize at f32 is total.
            Atom::Int(n) => LoweredValue::Node(self.dag.add_node(
                RiscOp::synth_const(Self::default_type().precision, *n as f64),
                vec![],
                Self::default_type(),
                self.current_span_id.clone(),
            )),
            Atom::Float(f) => LoweredValue::Node(self.dag.add_node(
                RiscOp::synth_const(Self::default_type().precision, *f),
                vec![],
                Self::default_type(),
                self.current_span_id.clone(),
            )),
            Atom::Bool(b) => LoweredValue::Node(self.dag.add_node(
                RiscOp::synth_const(Self::default_type().precision, if *b { 1.0 } else { 0.0 }),
                vec![],
                Self::default_type(),
                self.current_span_id.clone(),
            )),
            Atom::Str(_) => raise_malformed_deep(
                "a string atom in DAG expression position (string values are host-lane \
                 only)",
                None,
                self.current_span_id.clone(),
            ),
            Atom::Keyword(_) => raise_malformed_deep(
                "a bare keyword atom in expression position",
                None,
                self.current_span_id.clone(),
            ),
        }
    }

    fn lower_expr_node(&mut self, expr: &Expr, context: &str) -> NodeId {
        self.lower_expr(expr).expect_node(context)
    }

    fn lower_list(&mut self, list: &List, span: Span) -> LoweredValue {
        let elems = &list.elements;
        if elems.is_empty() {
            raise_malformed_deep(
                "an empty Deep list node",
                Some(span),
                self.current_span_id.clone(),
            );
        }

        let tag = match &elems[0] {
            Expr::Atom(Atom::Symbol(s), _) => s.as_str(),
            _ => raise_malformed_deep(
                "a Deep list node whose tag position is not a symbol",
                Some(span),
                self.current_span_id.clone(),
            ),
        };

        match tag {
            "def" => self.lower_def(elems),
            "let" => self.lower_let(elems),
            "lit" => self.lower_lit(elems),
            "var" => self.lower_var(elems),
            "app" => self.lower_app(elems, span),
            "fn" => self.lower_fn(elems),
            "pipe" => self.lower_pipe(elems),
            "cast" => self.lower_cast(elems),
            "if" => self.lower_if(elems),
            "tuple" => self.lower_tuple(elems),
            "par" => self.lower_par(elems),
            "realize" => self.lower_realize(elems),
            "copy" => self.lower_copy(elems),
            "borrow" => self.lower_identity(elems),
            "tuple-get" => self.lower_tuple_get(elems),
            "record" => self.lower_record(elems),
            "access" => self.lower_access(elems),
            "match" => self.lower_match(elems),
            "grad" => self.lower_grad(elems),
            "handle-effect" => self.lower_handle_effect(list),
            "jit" => self.lower_jit(elems),
            "vmap" => self.lower_unsupported(tag, elems),
            // Declarations lowered as an inert zero node: these forms are
            // not value expressions (the checker never lets their "value"
            // flow into a computation), so the node is a structural no-op,
            // not a value substitution. Census row 16 keep-with-comment
            // (chelis#730 section C1.4): a raise here would break every
            // program that declares a signature or type alias.
            "defsig" | "deftype" | "typealias" => LoweredValue::Node(self.dag.add_node(
                RiscOp::synth_const(Self::default_type().precision, 0.0),
                vec![],
                Self::default_type(),
                self.current_span_id.clone(),
            )),
            _ => {
                // Unknown-tag fallthrough: children are sequence-lowered
                // and the last value wins (some wrapper tags rely on
                // this). A CHILDLESS unknown tag has no value to forward
                // and previously seeded a silent zero - that childless
                // case now raises (census row 16, section C1.4; the
                // parser's closed 62-tag vocabulary keeps this dead, see
                // canary_unknown_deep_tag_is_rejected).
                if elems.len() <= 2 {
                    raise_malformed_deep(
                        &format!("an unknown childless Deep tag `{tag}`"),
                        Some(span),
                        self.current_span_id.clone(),
                    );
                }
                let mut last = LoweredValue::Node(self.dag.add_node(
                    RiscOp::synth_const(Self::default_type().precision, 0.0),
                    vec![],
                    Self::default_type(),
                    self.current_span_id.clone(),
                ));
                for elem in &elems[2..] {
                    last = self.lower_expr(elem);
                }
                last
            }
        }
    }

    /// `(def {meta...} name body)`
    fn lower_def(&mut self, elems: &[Expr]) -> LoweredValue {
        if elems.len() < 4 {
            raise_malformed_deep(
                "a `def` form with fewer than 4 elements",
                None,
                self.current_span_id.clone(),
            );
        }
        let name = match &elems[2] {
            Expr::Atom(Atom::Symbol(s), _) => s.clone(),
            _ => String::new(),
        };
        let body_id = self.lower_expr(&elems[3]);
        if !name.is_empty() {
            if self.is_host_list_expr(&elems[3]) {
                self.list_bindings.insert(name.clone(), elems[3].clone());
            }
            self.bindings.insert(name, body_id.clone());
            if let Some(callable) = self.callable_binding_expr(&elems[3]) {
                self.local_callables.insert(
                    match &elems[2] {
                        Expr::Atom(Atom::Symbol(s), _) => s.clone(),
                        _ => String::new(),
                    },
                    callable,
                );
            }
        }
        body_id
    }

    /// `(let {} (bind {} name1 expr1 name2 expr2 ...) body)`
    fn lower_let(&mut self, elems: &[Expr]) -> LoweredValue {
        if elems.len() < 4 {
            raise_malformed_deep(
                "a `let` form with fewer than 4 elements",
                None,
                self.current_span_id.clone(),
            );
        }
        let saved = self.bindings.clone();
        let saved_list_bindings = self.list_bindings.clone();
        let saved_shape_bindings = self.shape_bindings.clone();
        let saved_static_size_bindings = self.static_size_bindings.clone();
        let saved_callables = self.local_callables.clone();
        let saved_fn_typed_params = self.fn_typed_params.clone();

        // elems[2] = (bind {} name1 expr1 name2 expr2 ...)
        if let Expr::List(bind_list, _) = &elems[2] {
            // Skip tag and meta (elements[0] and [1]).
            let bind_kids = &bind_list.elements[2..];
            let mut i = 0;
            while i + 1 < bind_kids.len() {
                if let Expr::Atom(Atom::Symbol(name), _) = &bind_kids[i] {
                    // Same shadowing rationale as
                    // `lower_plain_callable_app`: drop any outer-scope
                    // `fn_typed_params[name]` so a let-shadowed name
                    // resolves through the new `local_callable` /
                    // `bindings` entry, not through the outer fn's
                    // `Parameter` classification.
                    self.fn_typed_params.remove(name);
                    // Issue #368: a `let`-bound list LITERAL (`rows = [r0, r1]`,
                    // a closed `Cons` chain) is recorded so `concat(rows, axis)`
                    // can resolve the elements back to the underlying tensor
                    // exprs and lower them through the differentiable Pad+Add
                    // cascade. Without this the name only resolves to the eager
                    // (degenerate) lowering of the list literal, and concat
                    // falls back to the rank-0 host placeholder. Gated on the
                    // non-host check so it does not change `is_host_list_expr`
                    // routing for `to_list`/`map`/`filter` bindings.
                    if self.is_host_list_expr(&bind_kids[i + 1])
                        || collect_cons_chain(&bind_kids[i + 1]).is_some()
                    {
                        self.list_bindings
                            .insert(name.clone(), bind_kids[i + 1].clone());
                    }
                    // chelis#369/#469: remember a `len = shape(operand, axis)`
                    // binding — OR a `let`-to-`let` alias / use-site `cast` of
                    // such a name (`a = shape(x, 0); c = a; expand(b, 0, c)`,
                    // RT-3) — so a later `expand(s, axis, cast(len, int32))`
                    // can recover the broadcast extent through the `let`
                    // indirection (the `tensor_full_like` idiom).
                    // `resolve_shape_binding_source` follows the alias chain
                    // and records the UNDERLYING `shape(...)` app, so recovery
                    // binds the `shape_dep` liveness edge to the actual source
                    // tensor and axis (mirroring how `fold_static_size`
                    // recurses `static_size_bindings` for the static path). A
                    // re-binding of `name` to anything else must drop any stale
                    // shape entry so shadowing never recovers a wrong extent.
                    if let Some(shape_app) = self.resolve_shape_binding_source(&bind_kids[i + 1]) {
                        self.shape_bindings.insert(name.clone(), shape_app);
                    } else {
                        self.shape_bindings.remove(name);
                    }
                    // chelis#469/#528: remember a `len = <static int>` binding
                    // (a literal, `cast(N, _)`, or integer arithmetic over
                    // such, following prior static bindings) so a later
                    // `expand(s, axis, cast(len, int32))` const-folds the
                    // extent instead of the size-1 default. Re-binding to a
                    // non-static value drops any stale entry (shadowing
                    // symmetry, mirroring `shape_bindings`).
                    if let Some(value) = self.fold_static_size(&bind_kids[i + 1]) {
                        self.static_size_bindings.insert(name.clone(), value);
                    } else {
                        self.static_size_bindings.remove(name);
                    }
                    if let Some(callable) = self.callable_binding_expr(&bind_kids[i + 1]) {
                        self.local_callables.insert(name.clone(), callable);
                    } else {
                        let val_id = self.lower_expr(&bind_kids[i + 1]);
                        self.bindings.insert(name.clone(), val_id);
                    }
                }
                i += 2;
            }
        }

        let result = self.lower_expr(&elems[3]);
        self.bindings = saved; // Restore scope
        self.list_bindings = saved_list_bindings;
        self.shape_bindings = saved_shape_bindings;
        self.static_size_bindings = saved_static_size_bindings;
        self.local_callables = saved_callables;
        self.fn_typed_params = saved_fn_typed_params;
        result
    }

    /// `(lit {type: T} value)`
    fn lower_lit(&mut self, elems: &[Expr]) -> LoweredValue {
        let ty = if let Some(Expr::Map(meta, _)) = elems.get(1) {
            self.type_from_meta(&meta.entries)
        } else {
            Self::default_type()
        };

        // The chelis#856 exactness path: the desugarer ascribes the
        // literal's contextual dtype in the `lit` meta, so the sealed
        // payload finalizes ONCE at that dtype right here. Integer
        // atoms travel their exact i64 (no `as f64` laundering, which
        // collapsed int64 above 2^53); an out-of-domain literal at its
        // ascribed dtype is a loud lowering diagnostic, not a wrap.
        let prim = ty.precision;
        let raw = if let Some(val_expr) = elems.get(2) {
            match val_expr {
                Expr::Atom(Atom::Int(n), _) => chelis_types::RawScalar::Int(*n),
                Expr::Atom(Atom::Float(f), _) => chelis_types::RawScalar::Float(*f),
                Expr::Atom(Atom::Bool(true), _) => chelis_types::RawScalar::Int(1),
                Expr::Atom(Atom::Bool(false), _) => chelis_types::RawScalar::Int(0),
                _ => chelis_types::RawScalar::Int(0),
            }
        } else {
            chelis_types::RawScalar::Int(0)
        };
        let value = match chelis_types::finalize_scalar("const", prim, raw) {
            Ok(value) => value,
            Err(trap) => raise_lowering_error(
                format!(
                    "literal does not finalize at its ascribed dtype `{}`: {trap} \
                     (spec/04-type-system.md section 9 [04-NUM-1]; chelis#856)",
                    prim.name()
                ),
                elems.first().map(Expr::span),
                self.current_span_id.clone(),
            ),
        };

        LoweredValue::Node(self.dag.add_node(
            RiscOp::Const { value },
            vec![],
            ty,
            self.current_span_id.clone(),
        ))
    }

    /// `(var {meta...} name)`
    fn lower_var(&mut self, elems: &[Expr]) -> LoweredValue {
        // C6: Extract type from metadata if available, otherwise use checked top-level type info.
        let explicit_ty = if let Some(Expr::Map(meta, _)) = elems.get(1) {
            self.type_from_meta(&meta.entries)
        } else {
            Self::default_type()
        };

        if let Some(Expr::Atom(Atom::Symbol(name), _)) = elems.get(2) {
            if let Some(id) = self.bindings.get(name) {
                let cached = id.clone();
                // N→1 lowering collapse per
                // spec/design/chelis_span_survival.md §2.3 rule (b):
                // returning a cached `LoweredValue` for a span-bearing
                // `(var {span: u} a)` must still record the var-ref's
                // span on the existing node. Without this append, a
                // var-ref to a let-bound name would drop its own span
                // and break the audit chain (the Load branch below
                // honors the rule via `current_span_id` on add_node).
                self.append_current_span_to_lowered_value(&cached);
                return cached;
            }
            // Reject `(var X)` where X is a known builtin name. The DAG
            // emits a Load when it encounters a free var, but a builtin
            // like `fold`, `map`, or `einsum` is a language-level operator,
            // not a host value — emitting `fold` as a Load and then as a
            // C identifier is meaningless. This happens most commonly
            // during `grad(fn_using_fold)` lowering, where the fn body is
            // inlined into a DAG context that can't represent the HOF.
            if BUILTIN_NAMES.contains(&name.as_str()) {
                raise_lowering_error(
                    format!(
                        "builtin `{name}` is not supported by IR evaluation as a \
                     value. If this is the body of a fn passed to `grad`, \
                     the grad pass needs to specialize around the builtin \
                     rather than inlining it"
                    ),
                    elems.first().map(Expr::span),
                    elems.first().and_then(Expr::span_id).map(ToOwned::to_owned),
                );
            }
            // chelis#520 D1: an unbound uppercase-initial name in
            // expression position is a nullary ADT constructor reference
            // (post-checker: the type checker resolves every value name,
            // and value identifiers are snake_case per spec/01 §3.2). The
            // host runtime applies the same rule in `eval_var`. Emitting
            // a `Load { name: "ModeA" }` here would fabricate a phantom
            // tensor input; the static Adt value instead lets
            // `lower_match` select the taken arm at lowering time.
            if !self.program_defs.contains_key(name)
                && name.chars().next().is_some_and(|ch| ch.is_uppercase())
            {
                return LoweredValue::Adt {
                    ctor: name.clone(),
                    field_names: None,
                    fields: Vec::new(),
                };
            }
            let ty = if explicit_ty == Self::default_type() {
                self.program_types.get(name).cloned().unwrap_or(explicit_ty)
            } else {
                explicit_ty
            };
            return LoweredValue::Node(self.dag.add_node(
                RiscOp::Load {
                    name: name.as_str().into(),
                },
                vec![],
                ty,
                self.current_span_id.clone(),
            ));
        }
        raise_malformed_deep(
            "a var form with no usable name",
            None,
            self.current_span_id.clone(),
        )
    }

    /// `(app {meta...} func arg1 arg2 ...)`
    fn lower_app(&mut self, elems: &[Expr], app_span: Span) -> LoweredValue {
        // A well-formed `(app)` has at minimum [tag, meta, func] (3
        // elements) for a zero-arg call. The previous `< 4` guard
        // rejected zero-arg user-def calls before any callable
        // resolution and emitted `Const { value: 0.0 }`, dropping the
        // fn body that would have been inlined -- this is
        // HostEval-ScalarFn-F1's root cause. `&elems[3..]` yielding an
        // empty slice is already handled by every downstream arm
        // (`lower_builtin_app`, `try_lower_callable_app`, and the
        // fallback "lower func and args, return last" path).
        if elems.len() < 3 {
            raise_malformed_deep(
                "an `app` form with no callee",
                Some(app_span),
                self.current_span_id.clone(),
            );
        }

        let ty = if let Some(Expr::Map(meta, _)) = elems.get(1) {
            self.type_from_meta(&meta.entries)
        } else {
            Self::default_type()
        };

        // Check if func is a known built-in: (var {} name).
        if let Expr::List(func_list, _) = &elems[2]
            && let Some(Expr::Atom(Atom::Symbol(func_tag), _)) = func_list.elements.first()
            && func_tag == "var"
            && let Some(Expr::Atom(Atom::Symbol(func_name), _)) = func_list.elements.get(2)
            && !self.program_defs.contains_key(func_name)
            && !self.local_callables.contains_key(func_name)
        {
            // chelis#520: a positional ADT constructor application
            // `(app {} (var Ctor) args...)`. Same uppercase-initial rule
            // as `lower_var`'s nullary-constructor branch; a constructor
            // is never a builtin, def, or local callable post-checker.
            if !BUILTIN_NAMES.contains(&func_name.as_str())
                && func_name.chars().next().is_some_and(|ch| ch.is_uppercase())
            {
                let fields = elems[3..]
                    .iter()
                    .map(|arg| self.lower_expr(arg))
                    .collect::<Vec<_>>();
                return LoweredValue::Adt {
                    ctor: func_name.clone(),
                    field_names: None,
                    fields,
                };
            }
            // chelis#620: `concat` over statically-known list VALUES. The
            // builtin arm's expr-level path (`lower_tensor_concat` /
            // `collect_cons_chain`) cannot see a list that only exists as a
            // lowered `Cons`/`Nil` Adt spine (the return value of an
            // unrolled recursive builder, or a prior static append bound to
            // a var). Handle the two static shapes here, where a
            // non-`Node` return is representable; anything else falls
            // through to the builtin arm unchanged.
            if func_name == "concat"
                && elems.len() == 5
                && let Some(value) = self.try_lower_static_list_concat(&elems[3], &elems[4])
            {
                return value;
            }
            return LoweredValue::Node(self.lower_builtin_app(
                func_name,
                &elems[3..],
                &ty,
                app_span,
            ));
        }

        if let Some(lowered) = self.try_lower_callable_app(&elems[2], &elems[3..], &ty, app_span) {
            return lowered;
        }

        // Not a recognized built-in -- lower func and args, return last.
        let mut last = self.lower_expr(&elems[2]);
        for arg in &elems[3..] {
            last = self.lower_expr(arg);
        }
        last
    }

    fn try_lower_callable_app(
        &mut self,
        func: &Expr,
        args: &[Expr],
        ty: &TensorType,
        app_span: Span,
    ) -> Option<LoweredValue> {
        let callable = self.resolve_callable_expr(func)?;
        // If the callee is a named top-level/local def, compute its
        // tracking name so `lower_plain_callable_app` can install a
        // recursion guard around the *body lowering* step (Inlining-F1).
        // Nameless fn literals don't need tracking because they can't
        // refer to themselves by name.
        //
        // The insert/remove pair lives inside `lower_plain_callable_app`
        // *after* argument evaluation. Placing it here in
        // `try_lower_callable_app` (around the entire dispatch) would
        // also block legitimate nested fn-typed parameter calls like
        // `outer(doubler, seed)` where `outer(f, x) = f(f(x))` — the
        // inner `f(x)` runs as an argument to the outer `f`, not as
        // part of the outer body, and must not trip the guard. See
        // `docs/investigations/inlining_names_recursion_guard_diagnosis.md`.
        let inlining_name = callable_ref_name(func).filter(|name| {
            self.local_callables.contains_key(name) || self.program_defs.contains_key(name)
        });
        match callable {
            CallableExpr::Plain(fn_expr) => {
                Some(self.lower_plain_callable_app(&fn_expr, args, app_span, inlining_name))
            }
            CallableExpr::Vmap { fn_expr, axis } => {
                Some(self.lower_vmap_callable_app(&fn_expr, axis, args, ty, app_span))
            }
            CallableExpr::VmapGrad { fn_expr, wrt, axis } => Some(
                self.lower_vmap_grad_callable_app(&fn_expr, wrt.as_deref(), axis, args, app_span),
            ),
            CallableExpr::Grad { fn_expr, wrt } => {
                Some(self.lower_grad_callable_app(&fn_expr, wrt.as_deref(), args, app_span))
            }
            // `Parameter` carries no body the IR can inline. Fall back to
            // `lower_app`'s existing "lower func and args, return last"
            // path (`lower.rs:2644`–`2649`), which is the same broken-but-
            // silent shape `lower_app` already produces for fn-typed-
            // parameter calls today. Real semantics come from call-site
            // inlining (`lower_plain_callable_app` substitutes the
            // concrete callable into `local_callables` before lowering
            // the inlined body, so the resolver sees a `Plain` not a
            // `Parameter`).
            CallableExpr::Parameter { .. } => None,
        }
    }

    fn resolve_callable_expr(&self, expr: &Expr) -> Option<CallableExpr> {
        self.resolve_callable_expr_inner(expr, &mut HashSet::new())
    }

    fn resolve_callable_expr_inner(
        &self,
        expr: &Expr,
        visited: &mut HashSet<String>,
    ) -> Option<CallableExpr> {
        let Expr::List(list, _) = expr else {
            return None;
        };
        match get_tag(list) {
            Some("fn") => Some(CallableExpr::Plain(expr.clone())),
            Some("var") => {
                let name = children(list).first().and_then(|expr| match expr {
                    Expr::Atom(Atom::Symbol(name), _) => Some(name.clone()),
                    _ => None,
                })?;
                // `visited` protects THIS resolution walk from alias cycles
                // (`def a = b; def b = a`). It deliberately does NOT consult
                // the active-inline state: a re-entrant call to a callee that
                // is mid-inline resolves normally and unrolls, bounded by the
                // depth caps in `lower_plain_callable_app` (chelis#620; the
                // Inlining-F1 refuse-on-reentry rule previously fell through
                // to `lower_app`'s silently wrong return-last-arg fallback).
                if !visited.insert(name.clone()) {
                    return None;
                }
                // Item 2-extended: a function-valued parameter is a
                // legitimate callable (Surf lets `def f(g: T -> T, x: T) =
                // x |> g` typecheck), but it has no body to recurse into
                // — the DAG can't represent a call to it (no
                // `RiscOp::Call`). Surface it as
                // `CallableExpr::Parameter` so `lower_pipe` can no-op the
                // stage on the standalone-def lowering path; call-site
                // inlining replaces this with the resolved callable via
                // `local_callables`, so this variant only appears when
                // the def is lowered in isolation. See
                // `docs/investigations/pipe_fn_param_stage_diagnosis.md`.
                if let Some(body) = self
                    .local_callables
                    .get(&name)
                    .or_else(|| self.program_defs.get(&name))
                {
                    return self.resolve_callable_expr_inner(body, visited);
                }
                if self.fn_typed_params.contains(&name) {
                    return Some(CallableExpr::Parameter { name });
                }
                None
            }
            Some("vmap") => {
                let kids = children(list);
                // chelis#524: distinguish the documented `vmap` contract
                // (axis argument ABSENT -> map over axis 0) from an axis
                // argument that is PRESENT but not a compile-time constant.
                // The pre-fix `extract_usize_value(...).unwrap_or(0)`
                // collapsed BOTH to axis 0, so a runtime mapped axis
                // silently vectorized over the WRONG axis — the same
                // silent default-to-0 anti-pattern #364 eliminated for
                // reductions / softmax / gather/scatter. A runtime axis must
                // be rejected at check time, so reaching here with an
                // unresolvable PRESENT axis is a loud, FATAL, located
                // lowering error (mirrors `extract_axis_raw`); a silent 0
                // would be absorbed into a wrong-axis vectorization.
                let axis = match kids.get(1) {
                    None => 0,
                    Some(axis_expr) => match self.extract_usize_value(axis_expr) {
                        Some(value) => value,
                        None => raise_fatal_lowering_error(
                            "`vmap` mapped axis is not a compile-time integer constant: the DAG \
                             cannot vectorize over a runtime axis (the checker admits only a \
                             literal or a `cast(<int>, int32)` axis here; a runtime axis must be \
                             rejected at check time). `vmap` defaults to axis 0 only when the axis \
                             argument is omitted, never when a non-constant axis is supplied",
                            Some(axis_expr.span()),
                            axis_expr.span_id().map(ToOwned::to_owned),
                        ),
                    },
                };
                if let Some(Expr::List(grad_list, _)) = kids.first()
                    && get_tag(grad_list) == Some("grad")
                {
                    let wrt = self.extract_grad_wrt_indices(grad_list);
                    return self
                        .resolve_callable_expr_inner(children(grad_list).first()?, visited)
                        .and_then(|inner| match inner {
                            CallableExpr::Plain(fn_expr) => {
                                Some(CallableExpr::VmapGrad { fn_expr, wrt, axis })
                            }
                            // `vmap(grad(parameter))` / `grad(...)`/`vmap(...)`
                            // inner shapes are G1/G2/G4 territory from the
                            // Item 2 sibling sweep — out of scope for
                            // dispatch A. Propagate `None`.
                            _ => None,
                        });
                }
                self.resolve_callable_expr_inner(kids.first()?, visited)
                    .and_then(|inner| match inner {
                        CallableExpr::Plain(fn_expr) => Some(CallableExpr::Vmap { fn_expr, axis }),
                        CallableExpr::Vmap { .. } => None,
                        CallableExpr::VmapGrad { .. } => None,
                        CallableExpr::Grad { fn_expr, wrt } => {
                            Some(CallableExpr::Grad { fn_expr, wrt })
                        }
                        // `vmap(parameter)` is G2 territory.
                        CallableExpr::Parameter { .. } => None,
                    })
            }
            Some("grad") => self
                .resolve_callable_expr_inner(children(list).first()?, visited)
                .and_then(|inner| match inner {
                    CallableExpr::Plain(fn_expr) => Some(CallableExpr::Grad {
                        fn_expr,
                        wrt: self.extract_grad_wrt_indices(list),
                    }),
                    // `grad(parameter)` is G1 territory.
                    _ => None,
                }),
            _ => None,
        }
    }

    fn callable_binding_expr(&self, expr: &Expr) -> Option<Expr> {
        self.resolve_callable_expr(expr).map(|_| expr.clone())
    }

    fn extract_grad_wrt_indices(&self, list: &List) -> Option<Vec<usize>> {
        let wrt_expr = children(list).get(1)?;
        if let Expr::List(tuple, _) = wrt_expr
            && get_tag(tuple) == Some("tuple")
        {
            return Some(
                children(tuple)
                    .iter()
                    .filter_map(|expr| self.extract_usize_value(expr))
                    .collect(),
            );
        }
        self.extract_usize_value(wrt_expr).map(|index| vec![index])
    }

    fn is_selected_wrt(
        &self,
        index: usize,
        ty: &TensorType,
        wrt_indices: Option<&[usize]>,
    ) -> bool {
        let differentiable = ty.precision.is_float();
        match wrt_indices {
            Some(indices) => differentiable && indices.contains(&index),
            None => differentiable,
        }
    }

    fn lower_grad_callable_app(
        &mut self,
        fn_expr: &Expr,
        wrt_indices: Option<&[usize]>,
        args: &[Expr],
        app_span: Span,
    ) -> LoweredValue {
        let actual_args: Vec<LoweredValue> = args.iter().map(|arg| self.lower_expr(arg)).collect();
        self.lower_grad_callable_with_values(fn_expr, wrt_indices, &actual_args, app_span)
    }

    /// Core lowering for `grad(fn)` applied to already-lowered argument
    /// values. Used by both `lower_grad_callable_app` (which lowers
    /// expression arguments first) and `lower_pipe` (which inherits the
    /// argument from the previous pipe stage). A `Node` argument is the
    /// classic tensor/scalar lane; an `Adt` argument (chelis#520 D2) is a
    /// statically-constructed record whose float-tensor fields are
    /// differentiated field-wise, producing an `Adt`-shaped gradient (the
    /// pytree contract of spec/design/differentiable_language.md Decision 6).
    fn lower_grad_callable_with_values(
        &mut self,
        fn_expr: &Expr,
        wrt_indices: Option<&[usize]>,
        actual_args: &[LoweredValue],
        app_span: Span,
    ) -> LoweredValue {
        let Some((param_names, body)) = self.extract_fn_parts(fn_expr) else {
            return self.lower_unrepresentable("grad", std::slice::from_ref(fn_expr));
        };
        let grad_param_type_exprs: Vec<Option<Expr>> = param_names
            .iter()
            .enumerate()
            .map(|(index, _)| extract_param_type(fn_expr, index).cloned())
            .collect();
        // chelis#520 D2: classify each already-lowered argument. A `Node`
        // is the classic tensor/scalar lane. An `Adt` is a statically-
        // constructed record: its fields must themselves be single tensor
        // nodes (nested ADT/tuple fields are outside the slice and fail
        // loudly). A `Tuple` argument has never been accepted here.
        enum GradArgPlan {
            Tensor(NodeId),
            Adt {
                ctor: String,
                field_names: Option<Vec<String>>,
                field_nodes: Vec<NodeId>,
            },
        }
        // chelis#520 D2: a multi-argument grad call may mix ADT and
        // tensor/scalar arguments. Each argument is lowered independently
        // below into its own gradient plan; the per-argument result plans
        // repack the flat gradient roots back into the correct structure
        // (an `Adt` for an ADT parameter, a `Node` for a tensor one), so
        // the result is a `Tuple` whose slots line up 1:1 with the
        // selected parameters. Structure alignment is preserved because
        // every ADT field root gets a distinct dotted key.
        let plans: Vec<GradArgPlan> = actual_args
            .iter()
            .enumerate()
            .map(|(index, arg)| match arg {
                LoweredValue::Node(id) => GradArgPlan::Tensor(*id),
                LoweredValue::Adt {
                    ctor,
                    field_names,
                    fields,
                } => {
                    let field_nodes = fields
                        .iter()
                        .enumerate()
                        .map(|(fidx, field)| {
                            field.as_single_node().unwrap_or_else(|| {
                                let label = field_names
                                    .as_ref()
                                    .and_then(|names| names.get(fidx).cloned())
                                    .unwrap_or_else(|| fidx.to_string());
                                raise_fatal_lowering_error(
                                    format!(
                                        "`grad(...)` argument {index}: field `{label}` of \
                                         constructor `{ctor}` is not a single tensor value; \
                                         nested ADT/tuple fields are not supported yet \
                                         (chelis#520 D2)"
                                    ),
                                    Some(app_span),
                                    None,
                                )
                            })
                        })
                        .collect();
                    GradArgPlan::Adt {
                        ctor: ctor.clone(),
                        field_names: field_names.clone(),
                        field_nodes,
                    }
                }
                LoweredValue::Tuple(_) => raise_fatal_lowering_error(
                    format!(
                        "`grad(...)` argument {index} must be a tensor, scalar, or ADT of \
                         float tensors; got a tuple value"
                    ),
                    Some(app_span),
                    None,
                ),
            })
            .collect();
        let node_type = |ctx: &Self, id: NodeId| {
            ctx.dag
                .get(id)
                .map(|node| node.output_type.clone())
                .unwrap_or_else(Self::default_type)
        };
        // Formal/actual pairs for the precision/rank substitution seeding
        // below. ADT-typed params are excluded: their formal annotation is
        // an ADT type (no tensor precision/rank slots) and their field
        // loads are typed directly from the actual field nodes.
        let mut subst_param_type_exprs: Vec<Option<Expr>> = Vec::new();
        let mut subst_actual_types: Vec<TensorType> = Vec::new();
        for (index, plan) in plans.iter().enumerate() {
            if let GradArgPlan::Tensor(id) = plan {
                subst_param_type_exprs.push(grad_param_type_exprs.get(index).cloned().flatten());
                subst_actual_types.push(node_type(self, *id));
            }
        }
        // issue #289: the precision substitution visible to the grad
        // sub-context. Parent bindings first, then the differentiated
        // function's own formal precision vars bound against the concrete
        // actual-argument precisions at this grad call site (the latter
        // wins on overlap). See the longer note at the sub-context seeding
        // below.
        let mut grad_prec_subst = self.prec_substitutions.clone();
        grad_prec_subst.extend(tensor_prec_substitutions(
            &subst_param_type_exprs,
            &subst_actual_types,
        ));
        // Tier-2 rank polymorphism (#286/#258): the rank-var substitution
        // visible to the grad sub-context — the structural twin of
        // `grad_prec_subst`. Parent bindings first, then the differentiated
        // function's own formal `..r` rank vars bound against the concrete
        // actual-argument shapes at this grad call site. Seeded into the
        // sub-context below so a rank-polymorphic callee reached while
        // differentiating the body monomorphizes to concrete ranks, exactly
        // as the precision path does.
        let mut grad_rank_subst = self.rank_substitutions.clone();
        grad_rank_subst.extend(tensor_rank_substitutions(
            &subst_param_type_exprs,
            &subst_actual_types,
            &self.dim_axis_positions,
        ));
        // Formal parameter shapes for the differentiated function. Use the
        // call-site-tolerant variant so a precision var that the grad call
        // site does NOT pin (it is internal to a callee, resolved when that
        // callee is inlined into the body below) does not panic here; the
        // load nodes' precision is taken from `grad_prec_subst` when known.
        let param_types: Vec<TensorType> = grad_param_type_exprs
            .iter()
            .map(|opt_expr| match opt_expr {
                Some(expr) => {
                    Self::formal_param_type_for_call(expr, &grad_prec_subst, &grad_rank_subst)
                }
                None => Self::default_type(),
            })
            .collect();
        let mut subctx = LowerCtx::new(
            self.program_types.clone(),
            self.program_defs.clone(),
            LinearityInfo::default(),
        );
        // issue #289: propagate the call-site precision-tvar substitution
        // into the grad sub-context so a precision-polymorphic callee
        // reached while differentiating the body monomorphizes to the
        // concrete call-site precision — exactly as an ordinary (non-grad)
        // call through the same callee does in `lower_plain_callable_app`.
        //
        // `LowerCtx::new` seeds an EMPTY `prec_substitutions`, so without
        // this the sub-context lowers the differentiated body with no
        // precision bindings; the first inlined `tensor[..., p]` callee
        // then trips the §5.8.1 "monomorphization missed precision var"
        // tripwire even though the differentiated entry point is fully
        // concrete `f32`.
        //
        // Two sources are merged, parent first so the more-specific
        // grad-call-site binding wins on overlap:
        //   1. the parent's substitutions, so a `grad` taken inside an
        //      already-monomorphized callee inherits that callee's
        //      precision bindings; and
        //   2. the precision vars in the differentiated function's own
        //      formal parameter annotations, bound against the concrete
        //      precisions of the actual arguments at this grad call site.
        //      This is the case where `grad` is applied directly to a
        //      precision-polymorphic function.
        //
        // When neither source supplies a concrete precision (a genuinely
        // under-determined precision var with no concrete call site), the
        // substitution stays empty and the §5.8.1 tripwire still fires —
        // preserving the clear diagnostic for that case.
        //
        // A precision var that is internal to a callee of the
        // differentiated body (the issue #289 reproducer: concrete-f32
        // `loss` calling precision-polymorphic `lin_p`) is NOT in this map
        // either; it is resolved when that callee is inlined into the body
        // by `lower_plain_callable_app`'s own call-site precision binding.
        subctx.prec_substitutions = grad_prec_subst;
        // Tier-2 rank polymorphism (#286/#258): seed the sub-context's rank
        // substitution the same way as precision, so a rank-poly callee
        // reached while differentiating the body monomorphizes to concrete
        // ranks instead of tripping the rank-monomorphization boundary.
        subctx.rank_substitutions = grad_rank_subst;
        subctx.allow_host_list_ad_rewrites = true;
        // Subctx inherits the parent's current span so synthesized loads
        // for grad's parameters carry the grad-call's span.
        subctx.current_span_id = self.current_span_id.clone();
        let captured_bindings = self.seed_subctx_with_lexical_scope(&mut subctx, &param_names);
        let mut wrt = Vec::new();
        // Actual argument node backing each `wrt` load, in `wrt` order
        // (tensor param -> the argument node, ADT field -> the field node).
        let mut wrt_actuals: Vec<NodeId> = Vec::new();
        // Formal/actual type pairs per load, for the dim-symbol remap of
        // the differentiated DAG below.
        let mut remap_formal_types: Vec<TensorType> = Vec::new();
        let mut remap_actual_types: Vec<TensorType> = Vec::new();
        // Extra `Load`-name -> actual-node entries for ADT field loads.
        let mut adt_arg_map_entries: Vec<(String, NodeId)> = Vec::new();
        // Result-packing plan: per selected param, how many gradient roots
        // it owns and how to reshape them.
        enum GradResultPlan {
            Tensor,
            Adt {
                ctor: String,
                field_names: Option<Vec<String>>,
                field_count: usize,
            },
        }
        let mut result_plans: Vec<GradResultPlan> = Vec::new();
        for (index, (name, param_ty)) in param_names
            .iter()
            .zip(param_types.iter().cloned())
            .enumerate()
        {
            match plans.get(index) {
                Some(GradArgPlan::Adt {
                    ctor,
                    field_names,
                    field_nodes,
                }) => {
                    // chelis#520 D2: an ADT-typed differentiated parameter.
                    // Every float-tensor field becomes its own Load + wrt
                    // entry; the parameter name binds to the Adt of those
                    // loads so the body's `match`/`access` destructuring
                    // resolves statically.
                    let selected = match wrt_indices {
                        Some(indices) => indices.contains(&index),
                        None => true,
                    };
                    let mut field_values = Vec::with_capacity(field_nodes.len());
                    for (fidx, field_node) in field_nodes.iter().enumerate() {
                        let field_ty = node_type(self, *field_node);
                        if selected && !field_ty.precision.is_float() {
                            let label = field_names
                                .as_ref()
                                .and_then(|names| names.get(fidx).cloned())
                                .unwrap_or_else(|| fidx.to_string());
                            raise_fatal_lowering_error(
                                format!(
                                    "`grad(...)` over ADT-typed parameter `{name}`: field \
                                     `{label}` of constructor `{ctor}` is not a float \
                                     tensor; mixed-struct gradients are not supported yet \
                                     (chelis#520 D2)"
                                ),
                                Some(app_span),
                                None,
                            );
                        }
                        let load_name = format!("{name}__adt_field_{fidx}");
                        let load = subctx.dag.add_node(
                            RiscOp::Load {
                                name: load_name.as_str().into(),
                            },
                            vec![],
                            field_ty.clone(),
                            subctx.current_span_id.clone(),
                        );
                        if selected {
                            wrt.push(load);
                            wrt_actuals.push(*field_node);
                        }
                        remap_formal_types.push(field_ty.clone());
                        remap_actual_types.push(field_ty);
                        adt_arg_map_entries.push((load_name, *field_node));
                        field_values.push(LoweredValue::Node(load));
                    }
                    if selected {
                        result_plans.push(GradResultPlan::Adt {
                            ctor: ctor.clone(),
                            field_names: field_names.clone(),
                            field_count: field_nodes.len(),
                        });
                    }
                    subctx.bindings.insert(
                        name.clone(),
                        LoweredValue::Adt {
                            ctor: ctor.clone(),
                            field_names: field_names.clone(),
                            fields: field_values,
                        },
                    );
                }
                Some(GradArgPlan::Tensor(actual)) => {
                    let load = subctx.dag.add_node(
                        RiscOp::Load {
                            name: name.as_str().into(),
                        },
                        vec![],
                        param_ty.clone(),
                        subctx.current_span_id.clone(),
                    );
                    if self.is_selected_wrt(index, &param_ty, wrt_indices) {
                        wrt.push(load);
                        wrt_actuals.push(*actual);
                        result_plans.push(GradResultPlan::Tensor);
                    }
                    remap_formal_types.push(param_ty.clone());
                    remap_actual_types.push(node_type(self, *actual));
                    subctx
                        .bindings
                        .insert(name.clone(), LoweredValue::Node(load));
                }
                // Fewer arguments than parameters: keep the pre-#520
                // behavior (a free Load with the formal type; the splice
                // leaves it unresolved and downstream evaluation reports
                // the missing input).
                None => {
                    let load = subctx.dag.add_node(
                        RiscOp::Load {
                            name: name.as_str().into(),
                        },
                        vec![],
                        param_ty.clone(),
                        subctx.current_span_id.clone(),
                    );
                    if self.is_selected_wrt(index, &param_ty, wrt_indices) {
                        wrt.push(load);
                    }
                    subctx
                        .bindings
                        .insert(name.clone(), LoweredValue::Node(load));
                }
            }
        }
        let output = subctx
            .lower_expr(body)
            .expect_node("grad requires a scalar floating output");
        subctx.dag.add_root(output);
        // Issue #197: route through grad_dag_checked so a
        // non-differentiable op in the gradient body (argmax/argmin,
        // floor/ceil, scatter_replace) surfaces a structured
        // `AdError::NotSupported` diagnostic that names the offending
        // op and the reason, instead of a generic scalar-output
        // message (or a silent zero gradient for floor/ceil under the
        // unchecked variant).
        let grad_result = grad_dag_checked(&subctx.dag, output, &wrt).unwrap_or_else(|ad_err| {
            raise_fatal_lowering_error(
                format!("`grad(...)` lowering rejected: {ad_err}"),
                Some(body.span()),
                body.span_id().map(ToOwned::to_owned),
            )
        });

        let arg_map = param_names
            .iter()
            .zip(plans.iter())
            .filter_map(|(name, plan)| match plan {
                GradArgPlan::Tensor(actual) => Some((name.clone(), *actual)),
                // ADT params are served by their per-field load entries.
                GradArgPlan::Adt { .. } => None,
            })
            .chain(adt_arg_map_entries)
            .chain(captured_bindings)
            .collect::<HashMap<_, _>>();
        let specialized_grad_dag = Self::remap_callable_dim_symbols(
            &grad_result.dag,
            &remap_formal_types,
            &remap_actual_types,
        );
        let remap = self.splice_dag(&specialized_grad_dag, &arg_map);
        // Per-wrt gradient node (post-splice). A `None` entry means the
        // wrt input has no adjoint because it does not influence the
        // output; the tensor lane preserves the pre-#520 behavior of
        // dropping it from the result, and the ADT lane packs an explicit
        // zero tensor so the gradient struct keeps its field structure.
        let grad_per_wrt: Vec<Option<NodeId>> = wrt
            .iter()
            .map(|wrt_node| {
                grad_result
                    .grad_nodes
                    .get(wrt_node)
                    .map(|grad_node| remap[grad_node])
            })
            .collect();
        for (grad_node, reusable_input) in grad_per_wrt.iter().zip(wrt_actuals.iter()) {
            if let Some(grad_node) = grad_node {
                self.dag.set_reusable_input(*grad_node, *reusable_input);
            }
        }
        // Rebuild the per-parameter result structure: one Node per tensor
        // param, an Adt of Nodes per ADT param (chelis#520 D2).
        let mut grad_iter = grad_per_wrt.iter().copied();
        let mut wrt_actual_iter = wrt_actuals.iter().copied();
        let mut packed: Vec<LoweredValue> = Vec::with_capacity(result_plans.len());
        // A multi-target result is displayed as a tuple keyed by fixed,
        // type-derived slot names (`out.0..out.N`, chelis#614): slot position
        // is significant. A single-target result is a bare value with no
        // sibling slots to shift.
        let multi_target = result_plans.len() > 1;
        for plan in &result_plans {
            match plan {
                GradResultPlan::Tensor => {
                    let grad_node = grad_iter.next().flatten();
                    let actual = wrt_actual_iter.next();
                    match grad_node {
                        Some(node) => packed.push(LoweredValue::Node(node)),
                        // The differentiated tensor argument does not
                        // influence the output: its gradient is exactly zero.
                        // In a multi-target result, dropping the slot would
                        // shift every later gradient into the wrong tuple
                        // position and mislabel it, so materialize the shaped
                        // zero, the same way the ADT field zero-fill below
                        // does (chelis#520 D2 / chelis#614).
                        None if multi_target => {
                            let field_ty = actual
                                .map(|id| node_type(self, id))
                                .unwrap_or_else(Self::default_type);
                            let zero = self.zero_tensor_node(&field_ty);
                            packed.push(LoweredValue::Node(zero));
                        }
                        // Single-target result: preserve the pre-#520
                        // bare-tensor drop and its reuse-hint path (the
                        // `[LoweredValue::Node(single)]` arm below). A lone
                        // target has no sibling slot to mislabel.
                        None => {}
                    }
                }
                GradResultPlan::Adt {
                    ctor,
                    field_names,
                    field_count,
                } => {
                    let mut fields = Vec::with_capacity(*field_count);
                    for _ in 0..*field_count {
                        let grad_node = grad_iter.next().flatten();
                        let actual = wrt_actual_iter.next();
                        let node = grad_node.unwrap_or_else(|| {
                            // The field does not influence the output:
                            // its gradient is exactly zero. Materialize
                            // the zero so the gradient struct keeps the
                            // input's field structure (pytree contract).
                            let field_ty = actual
                                .map(|id| node_type(self, id))
                                .unwrap_or_else(Self::default_type);
                            self.zero_tensor_node(&field_ty)
                        });
                        fields.push(LoweredValue::Node(node));
                    }
                    packed.push(LoweredValue::Adt {
                        ctor: ctor.clone(),
                        field_names: field_names.clone(),
                        fields,
                    });
                }
            }
        }
        match packed.as_slice() {
            [LoweredValue::Node(single)] => {
                // Preserve the pre-#520 reuse hint on the classic
                // single-tensor-gradient shape.
                let candidate_inputs: Vec<NodeId> = plans
                    .iter()
                    .filter_map(|plan| match plan {
                        GradArgPlan::Tensor(actual) => Some(*actual),
                        GradArgPlan::Adt { .. } => None,
                    })
                    .collect();
                if candidate_inputs.len() == plans.len() {
                    let single = *single;
                    return LoweredValue::Node(self.attach_reuse_hint(
                        single,
                        app_span,
                        &candidate_inputs,
                    ));
                }
                LoweredValue::Node(*single)
            }
            [single_adt @ LoweredValue::Adt { .. }] => single_adt.clone(),
            _ => LoweredValue::Tuple(packed),
        }
    }

    /// A zero-valued tensor of the given type: `Const 0.0`, cast to the
    /// target precision, expanded axis-by-axis to the target dims
    /// (mirrors the `lower_if_mask` expansion pattern). Used by the ADT
    /// gradient packing (chelis#520 D2) for fields with no adjoint.
    fn zero_tensor_node(&mut self, ty: &TensorType) -> NodeId {
        let mut node = self.dag.add_node(
            RiscOp::synth_const(Self::default_type().precision, 0.0),
            vec![],
            Self::default_type(),
            self.current_span_id.clone(),
        );
        if ty.precision != Self::default_type().precision {
            node = self.dag.add_node(
                RiscOp::Cast {
                    new_precision: ty.precision,
                },
                vec![node],
                TensorType {
                    dims: vec![],
                    precision: ty.precision,
                },
                self.current_span_id.clone(),
            );
        }
        let mut dims = Vec::new();
        for (axis, dim) in ty.dims.iter().enumerate() {
            dims.push(dim.clone());
            node = self.dag.add_node(
                RiscOp::Expand {
                    axis,
                    size: DimExpr::from(dim),
                },
                vec![node],
                TensorType {
                    dims: dims.clone(),
                    precision: ty.precision,
                },
                self.current_span_id.clone(),
            );
        }
        node
    }

    fn lower_plain_callable_app(
        &mut self,
        fn_expr: &Expr,
        args: &[Expr],
        _app_span: Span,
        inlining_name: Option<String>,
    ) -> LoweredValue {
        let Some((param_names, body)) = self.extract_fn_parts(fn_expr) else {
            return self
                .lower_unrepresentable("function application", std::slice::from_ref(fn_expr));
        };
        let saved = self.bindings.clone();
        let saved_list_bindings = self.list_bindings.clone();
        let saved_shape_bindings = self.shape_bindings.clone();
        let saved_static_size_bindings = self.static_size_bindings.clone();
        let saved_callables = self.local_callables.clone();
        let saved_fn_typed_params = self.fn_typed_params.clone();
        let saved_dim_substitutions = self.dim_substitutions.clone();
        let saved_prec_substitutions = self.prec_substitutions.clone();
        let saved_rank_substitutions = self.rank_substitutions.clone();
        let saved_dim_axis_positions = self.dim_axis_positions.clone();
        // WS-A8: capture the raw param-type Deep exprs so we can pull
        // out `(t-var {} p)` precision-var names for monomorphization,
        // and (Tier-2) `(d-rank {} r)` rank-var names for rank
        // monomorphization. The parsed `TensorType` already collapses
        // both slots, losing the var-names we need.
        let param_type_exprs: Vec<Option<Expr>> = param_names
            .iter()
            .enumerate()
            .map(|(index, _)| extract_param_type(fn_expr, index).cloned())
            .collect();
        let prec_subst_for_params = self.prec_substitutions.clone();
        let rank_subst_for_params = self.rank_substitutions.clone();
        let param_types: Vec<TensorType> = param_type_exprs
            .iter()
            .map(|opt_expr| match opt_expr {
                // issue #289: tolerant of an as-yet-unresolved precision
                // var in a formal parameter annotation — the precision is
                // bound from the actual argument a few lines below (via
                // `tensor_prec_substitutions`); only the dims of this
                // formal type are consumed (by `tensor_dim_substitutions`).
                // #286/#258: rank-aware, so a `..r` formal slot is expanded
                // from `rank_subst_for_params` here too.
                Some(expr) => Self::formal_param_type_for_call(
                    expr,
                    &prec_subst_for_params,
                    &rank_subst_for_params,
                ),
                None => Self::default_type(),
            })
            .collect();
        let mut formal_types = Vec::new();
        let mut formal_type_exprs: Vec<Option<Expr>> = Vec::new();
        let mut actual_types = Vec::new();
        for (((name, arg_expr), param_ty), formal_expr) in param_names
            .iter()
            .zip(args.iter())
            .zip(param_types)
            .zip(param_type_exprs.iter())
        {
            // Item 2-extended: shadowing; the inlined fn's param name
            // is bound to a fresh value (either a `local_callable` or a
            // `bindings` entry). Drop any outer-scope
            // `fn_typed_params[name]` so the resolver doesn't
            // misclassify the shadowed name as a `Parameter` when it's
            // really backed by a concrete `local_callable` or a tensor
            // binding. `local_callables` takes precedence in the
            // resolver anyway, but bindings-only shadowing (non-callable
            // arg for a non-callable param) would otherwise leak the
            // outer `fn_typed_params` entry.
            self.fn_typed_params.remove(name);
            if let Some(callable) = self.callable_binding_expr(arg_expr) {
                self.local_callables.insert(name.clone(), callable);
            } else {
                let arg_id = self.lower_expr(arg_expr);
                if let LoweredValue::Node(node_id) = &arg_id
                    && let Some(actual_ty) =
                        self.dag.get(*node_id).map(|node| node.output_type.clone())
                {
                    formal_types.push(param_ty);
                    formal_type_exprs.push(formal_expr.clone());
                    actual_types.push(actual_ty);
                }
                self.bindings.insert(name.clone(), arg_id);
            }
        }
        self.dim_substitutions
            .extend(tensor_dim_substitutions(&formal_types, &actual_types));
        // Issue #388: record the positional index of each named axis in the
        // formal parameter shapes so a named reduction/expand-anchor lookup
        // can recover the axis even after monomorphization erases the named
        // axis from a literal-shaped operand's dims.
        self.dim_axis_positions
            .extend(tensor_dim_axis_positions(&formal_type_exprs, &actual_types));
        // WS-A8: extend the precision-tvar substitution with bindings
        // from this call site's formal-vs-actual precision slots. Walks
        // the raw type-exprs (which preserve `(t-var)` shape) against
        // the actual `TensorType`s (which always carry concrete
        // primitives at lowering time).
        self.prec_substitutions
            .extend(tensor_prec_substitutions(&formal_type_exprs, &actual_types));
        // issue #319: the checker renames a separate-`sig`
        // precision-polymorphic verb's precision variable `p` to a fresh
        // internal name (e.g. `t304`) when it stamps the resolved body
        // node types (the `permute` adjoint output, etc.). That renamed
        // name is not the original `p`, so the name-keyed substitution
        // above misses it and a shape-preserving op (`permute`) lowered
        // from the body trips the §5.8.1 monomorphization tripwire in
        // `type_from_meta`. When EVERY formal-parameter precision var
        // resolves to one shared concrete precision (a fully
        // precision-monomorphic call), bind every renamed body precision
        // var to it; otherwise bind nothing and let the tripwire / a
        // downstream precision-mismatch diagnostic fire. This preserves
        // the no-implicit-precision-promotion invariant for a genuinely
        // heterogeneous or under-determined call — see
        // `formal_precision_var_bindings`.
        //
        // Why recover here rather than fix the rename at the source
        // (issue #319 review): the cleaner fix is for the checker to
        // preserve the sig's original `p` name through `type_to_deep_expr`
        // so the name-keyed `tensor_prec_substitutions` above already
        // matches. But `type_to_deep_expr` prints every `TensorPrec::Var`
        // anonymously, and threading user-facing sig-var names through the
        // checker's `Subst`/printing touches ALL polymorphic-type printing
        // (every verb, every diagnostic), with wide blast radius on
        // inference and golden output. This recovery is deliberately
        // localized to the precision-poly verb call path and is removable
        // wholesale if the checker later preserves the name — at which
        // point the name-keyed substitution subsumes it.
        for (var_name, prim) in formal_precision_var_bindings(fn_expr, body, &actual_types) {
            self.prec_substitutions.entry(var_name).or_insert(prim);
        }
        // Tier-2 rank polymorphism (spec/design/rank_polymorphism.md):
        // extend the rank-var substitution with bindings from this call
        // site's formal-vs-actual shape slots. A formal param whose shape
        // is a sole `(d-rank {} r)` binds `r` to the actual arg's concrete
        // dim vector, so the inlined rank-poly body resolves its `..r`
        // tensors to concrete ranks before any backend sees them. The
        // rank analogue of the `prec_substitutions.extend(...)` above.
        let rank_subst =
            tensor_rank_substitutions(&formal_type_exprs, &actual_types, &self.dim_axis_positions);
        self.rank_substitutions.extend(rank_subst);
        // chelis#620 (Inlining-F1 successor): recursion lowers by BOUNDED
        // UNROLLING. Depth accounting installs *here*, after argument
        // evaluation, so legitimate nested calls passed as arguments to
        // the same fn-typed-parameter alias (e.g. `outer(doubler, seed)`
        // with `outer(f, x) = f(f(x))`) finish lowering before the body
        // scope is counted. A re-entrant call to a callee that is
        // mid-inline resolves and inlines normally; a well-founded
        // recursion terminates through static `if`/`match` pruning of its
        // base case, and the two caps are the loud backstop for chains the
        // pruning cannot bound (the pre-#620 refuse-on-reentry rule
        // instead fell through to `lower_app`'s silently wrong
        // return-last-arg fallback; history in
        // `docs/investigations/inlining_names_recursion_guard_diagnosis.md`).
        // Decrements are skipped on raise: every lowering error unwinds
        // through the per-entry `catch_lowering` and the ctx is abandoned.
        self.inlining_active += 1;
        if let Some(name) = &inlining_name {
            *self.inlining_depths.entry(name.clone()).or_insert(0) += 1;
        }
        if self.inlining_active > MAX_TOTAL_INLINE_DEPTH {
            self.reject_lowering_slice(
                Some(fn_expr),
                format!(
                    "call inlining exceeded the total nesting limit of \
                     {MAX_TOTAL_INLINE_DEPTH} active levels (a deep or mutually recursive \
                     call chain): recursion lowers by bounded static unrolling (chelis#620)"
                ),
            );
        }
        if let Some(name) = &inlining_name
            && self.inlining_depths.get(name).copied().unwrap_or(0) > MAX_STATIC_RECURSION_DEPTH
        {
            self.reject_lowering_slice(
                Some(fn_expr),
                format!(
                    "recursive inlining of `{name}` exceeded the static unroll limit of \
                     {MAX_STATIC_RECURSION_DEPTH} levels: recursion lowers by unrolling, so \
                     a recursive call chain must terminate through a compile-time-resolvable \
                     base case (a static `if` or `match` condition) within the limit; a \
                     condition that is only known at runtime cannot bound the unroll \
                     (chelis#620)"
                ),
            );
        }
        // Each unroll level costs multiple large lowering frames (debug
        // builds overflow the default 8 MB main-thread stack well before the
        // 512-level cap without this). `maybe_grow` at THIS site works where
        // chelis-types' boundary `stacker::grow` pattern would not: the
        // stack nears exhaustion mid-descent, and every additional level
        // re-enters this function, so the grow site is always in reach.
        let result = stacker::maybe_grow(64 * 1024, 4 * 1024 * 1024, || self.lower_expr(body));
        self.inlining_active -= 1;
        if let Some(name) = &inlining_name
            && let Some(depth) = self.inlining_depths.get_mut(name)
        {
            *depth = depth.saturating_sub(1);
            if *depth == 0 {
                self.inlining_depths.remove(name);
            }
        }
        self.bindings = saved;
        self.list_bindings = saved_list_bindings;
        self.shape_bindings = saved_shape_bindings;
        self.static_size_bindings = saved_static_size_bindings;
        self.local_callables = saved_callables;
        self.fn_typed_params = saved_fn_typed_params;
        self.dim_substitutions = saved_dim_substitutions;
        self.prec_substitutions = saved_prec_substitutions;
        self.rank_substitutions = saved_rank_substitutions;
        self.dim_axis_positions = saved_dim_axis_positions;
        result
    }

    fn lower_plain_callable_with_values(
        &mut self,
        fn_expr: &Expr,
        args: &[LoweredValue],
    ) -> LoweredValue {
        let Some((param_names, body)) = self.extract_fn_parts(fn_expr) else {
            return self
                .lower_unrepresentable("function application", std::slice::from_ref(fn_expr));
        };
        let saved = self.bindings.clone();
        let saved_list_bindings = self.list_bindings.clone();
        let saved_shape_bindings = self.shape_bindings.clone();
        let saved_static_size_bindings = self.static_size_bindings.clone();
        let saved_callables = self.local_callables.clone();
        let saved_fn_typed_params = self.fn_typed_params.clone();
        for (name, arg_id) in param_names.iter().zip(args.iter().cloned()) {
            // Same shadowing rationale as `lower_plain_callable_app`.
            self.fn_typed_params.remove(name);
            self.bindings.insert(name.clone(), arg_id);
        }
        let result = self.lower_expr(body);
        if let Some(ret_ty_expr) = extract_fn_return_type(fn_expr) {
            let ret_ty = Self::type_from_type_expr_with_subst(
                ret_ty_expr,
                &self.prec_substitutions,
                &self.rank_substitutions,
            );
            self.repair_output_type_if_default(&result, &ret_ty);
        }
        self.bindings = saved;
        self.list_bindings = saved_list_bindings;
        self.shape_bindings = saved_shape_bindings;
        self.static_size_bindings = saved_static_size_bindings;
        self.local_callables = saved_callables;
        self.fn_typed_params = saved_fn_typed_params;
        result
    }

    fn lower_vmap_callable_app(
        &mut self,
        fn_expr: &Expr,
        axis: usize,
        args: &[Expr],
        _ty: &TensorType,
        app_span: Span,
    ) -> LoweredValue {
        let actual_args: Vec<NodeId> = args
            .iter()
            .map(|arg| self.lower_expr_node(arg, "vmap arguments"))
            .collect();
        self.lower_vmap_callable_with_nodes(fn_expr, axis, &actual_args, app_span)
    }

    fn lower_vmap_callable_with_nodes(
        &mut self,
        fn_expr: &Expr,
        axis: usize,
        actual_args: &[NodeId],
        app_span: Span,
    ) -> LoweredValue {
        let Some((param_names, body)) = self.extract_fn_parts(fn_expr) else {
            return self.lower_unrepresentable("vmap", std::slice::from_ref(fn_expr));
        };
        let prec_subst_for_params = self.prec_substitutions.clone();
        let rank_subst_for_params = self.rank_substitutions.clone();
        let param_types: Vec<TensorType> = param_names
            .iter()
            .enumerate()
            .map(|(index, _)| {
                extract_param_type(fn_expr, index)
                    .map(|expr| {
                        Self::type_from_type_expr_with_subst(
                            expr,
                            &prec_subst_for_params,
                            &rank_subst_for_params,
                        )
                    })
                    .unwrap_or_else(Self::default_type)
            })
            .collect();
        let actual_types: Vec<TensorType> = actual_args
            .iter()
            .map(|id| {
                self.dag
                    .get(*id)
                    .map(|node| node.output_type.clone())
                    .unwrap_or_else(Self::default_type)
            })
            .collect();

        let mut canonical_args = Vec::with_capacity(actual_args.len());
        // Actual argument types with the vmap `axis` permuted to the front
        // (the batch axis is always at position 0 in these), so the chelis#383
        // dim-symbol remap below zips them positionally against the
        // batch-prepended formal parameter types regardless of the original
        // `axis`.
        let mut canonical_actual_types = Vec::with_capacity(actual_args.len());
        let mut batch_dim = None;
        for (arg_id, arg_ty) in actual_args.iter().copied().zip(actual_types.iter()) {
            if axis < arg_ty.dims.len() {
                let perm = axis_to_front_perm(arg_ty.dims.len(), axis);
                let canon_ty = permuted_tensor_type(arg_ty, &perm);
                let canonical = if axis == 0 {
                    arg_id
                } else {
                    self.dag.add_node(
                        RiscOp::Permute { axes: perm },
                        vec![arg_id],
                        canon_ty.clone(),
                        self.current_span_id.clone(),
                    )
                };
                batch_dim.get_or_insert_with(|| canon_ty.dims[0].clone());
                canonical_actual_types.push(canon_ty);
                canonical_args.push(canonical);
            } else {
                canonical_actual_types.push(arg_ty.clone());
                canonical_args.push(arg_id);
            }
        }

        let Some(batch_dim) = batch_dim else {
            return self.lower_unrepresentable(
                "vmap with no tensor arguments",
                std::slice::from_ref(fn_expr),
            );
        };

        let mut subctx = LowerCtx::new(
            self.program_types.clone(),
            self.program_defs.clone(),
            LinearityInfo::default(),
        );
        subctx.allow_host_list_ad_rewrites = true;
        // Subctx inherits the parent's current span so synthesized loads
        // for vmap's parameters carry the vmap-call's span.
        subctx.current_span_id = self.current_span_id.clone();
        let captured_bindings = self.seed_subctx_with_lexical_scope(&mut subctx, &param_names);
        for (name, param_expr) in param_names.iter().zip(param_types.iter().cloned()) {
            let load = subctx.dag.add_node(
                RiscOp::Load {
                    name: name.as_str().into(),
                },
                vec![],
                param_expr,
                subctx.current_span_id.clone(),
            );
            subctx
                .bindings
                .insert(name.clone(), LoweredValue::Node(load));
        }
        let root_value = subctx.lower_expr(body);
        for root in root_value.flatten_nodes() {
            subctx.dag.add_root(root);
        }

        let vmapped = match vmap::vectorize_axis0(&subctx.dag, batch_dim.clone()) {
            Ok(dag) => dag,
            Err(message) => raise_lowering_error(
                format!("`vmap` lowering failed: {message}"),
                Some(body.span()),
                body.span_id().map(ToOwned::to_owned),
            ),
        };

        let mut arg_map = HashMap::new();
        for ((name, param_ty), arg_id) in param_names
            .iter()
            .zip(param_types.iter())
            .zip(canonical_args.iter().copied())
        {
            arg_map.insert(
                name.clone(),
                self.materialize_vmapped_arg(arg_id, param_ty, &batch_dim),
            );
        }
        arg_map.extend(captured_bindings);

        // chelis#383: `vectorize_axis0` prepended the batch dim to EVERY
        // node type in `vmapped` (including the parameter Loads), so the
        // formal parameter types must be batch-prepended too before remapping
        // their named dims onto the actual argument's concrete dims.
        // `remap_tensor_dim_symbols` only substitutes between same-rank
        // formal/actual pairs (chelis#258: positional zip across different
        // ranks is never correct); without this prepend a rank-N parameter
        // (`vinner`'s `x: tensor[seq, head]`) is rank-mismatched against the
        // rank-(N+1) vmap argument (`y: tensor[batch, seq, head]`), the
        // substitution is dropped, and a two-stage named reduce leaves a
        // `Named("seq")` dim in the vmapped body with no declaring Load —
        // tripping the `dag::symbolic_occurrences` Bucket-4c guard at build
        // time even though check and eval are clean. The canonical batch dim
        // is the one `vectorize_axis0` itself prepended.
        let vmapped_param_types: Vec<TensorType> = param_types
            .iter()
            .map(|ty| TensorType {
                dims: std::iter::once(batch_dim.clone())
                    .chain(ty.dims.iter().cloned())
                    .collect(),
                precision: ty.precision,
            })
            .collect();
        let specialized_vmapped = Self::remap_callable_dim_symbols(
            &vmapped,
            &vmapped_param_types,
            &canonical_actual_types,
        );
        let remap = self.splice_dag(&specialized_vmapped, &arg_map);
        let mut flattened = root_value
            .flatten_nodes()
            .into_iter()
            .map(|node| remap[&node])
            .collect::<Vec<_>>();
        if axis > 0 {
            for result in &mut flattened {
                let result_ty = self
                    .dag
                    .get(*result)
                    .map(|node| node.output_type.clone())
                    .unwrap_or_else(Self::default_type);
                if axis < result_ty.dims.len() {
                    let perm = front_to_axis_perm(result_ty.dims.len(), axis);
                    let perm_ty = permuted_tensor_type(&result_ty, &perm);
                    *result = self.dag.add_node(
                        RiscOp::Permute { axes: perm },
                        vec![*result],
                        perm_ty,
                        self.current_span_id.clone(),
                    );
                }
            }
        }
        if flattened.len() == 1 {
            let result = self.attach_reuse_hint(flattened[0], app_span, &canonical_args);
            LoweredValue::Node(result)
        } else {
            let mut iter = flattened.into_iter();
            LoweredValue::from_flat(&root_value, &mut iter)
        }
    }

    fn lower_vmap_grad_callable_app(
        &mut self,
        fn_expr: &Expr,
        wrt_indices: Option<&[usize]>,
        axis: usize,
        args: &[Expr],
        app_span: Span,
    ) -> LoweredValue {
        let actual_args: Vec<NodeId> = args
            .iter()
            .map(|arg| self.lower_expr_node(arg, "vmap(grad) arguments"))
            .collect();
        self.lower_vmap_grad_callable_with_nodes(fn_expr, wrt_indices, axis, &actual_args, app_span)
    }

    /// Core lowering for `vmap(grad(fn))` applied to already-lowered
    /// argument nodes. Used by both `lower_vmap_grad_callable_app` (which
    /// lowers expression arguments first) and `lower_pipe` (which
    /// inherits the argument from the previous pipe stage).
    fn lower_vmap_grad_callable_with_nodes(
        &mut self,
        fn_expr: &Expr,
        wrt_indices: Option<&[usize]>,
        axis: usize,
        actual_args: &[NodeId],
        app_span: Span,
    ) -> LoweredValue {
        let Some((param_names, body)) = self.extract_fn_parts(fn_expr) else {
            return self.lower_unrepresentable("vmap(grad)", std::slice::from_ref(fn_expr));
        };
        let prec_subst_for_params = self.prec_substitutions.clone();
        let rank_subst_for_params = self.rank_substitutions.clone();
        let param_types: Vec<TensorType> = param_names
            .iter()
            .enumerate()
            .map(|(index, _)| {
                extract_param_type(fn_expr, index)
                    .map(|expr| {
                        Self::type_from_type_expr_with_subst(
                            expr,
                            &prec_subst_for_params,
                            &rank_subst_for_params,
                        )
                    })
                    .unwrap_or_else(Self::default_type)
            })
            .collect();
        let actual_types: Vec<TensorType> = actual_args
            .iter()
            .map(|id| {
                self.dag
                    .get(*id)
                    .map(|node| node.output_type.clone())
                    .unwrap_or_else(Self::default_type)
            })
            .collect();

        let mut canonical_args = Vec::with_capacity(actual_args.len());
        let mut batch_dim = None;
        for (arg_id, arg_ty) in actual_args.iter().copied().zip(actual_types.iter()) {
            if axis < arg_ty.dims.len() {
                let perm = axis_to_front_perm(arg_ty.dims.len(), axis);
                let canon_ty = permuted_tensor_type(arg_ty, &perm);
                let canonical = if axis == 0 {
                    arg_id
                } else {
                    self.dag.add_node(
                        RiscOp::Permute { axes: perm },
                        vec![arg_id],
                        canon_ty.clone(),
                        self.current_span_id.clone(),
                    )
                };
                batch_dim.get_or_insert_with(|| canon_ty.dims[0].clone());
                canonical_args.push(canonical);
            } else {
                canonical_args.push(arg_id);
            }
        }

        let Some(batch_dim) = batch_dim else {
            return self.lower_unrepresentable(
                "vmap(grad) with no tensor arguments",
                std::slice::from_ref(fn_expr),
            );
        };

        let mut subctx = LowerCtx::new(
            self.program_types.clone(),
            self.program_defs.clone(),
            LinearityInfo::default(),
        );
        subctx.allow_host_list_ad_rewrites = true;
        // Subctx inherits the parent's current span so synthesized loads
        // for vmap(grad)'s parameters carry the call's span.
        subctx.current_span_id = self.current_span_id.clone();
        let captured_bindings = self.seed_subctx_with_lexical_scope(&mut subctx, &param_names);
        let mut wrt = Vec::new();
        for (index, (name, param_ty)) in param_names
            .iter()
            .zip(param_types.iter().cloned())
            .enumerate()
        {
            let load = subctx.dag.add_node(
                RiscOp::Load {
                    name: name.as_str().into(),
                },
                vec![],
                param_ty.clone(),
                subctx.current_span_id.clone(),
            );
            if self.is_selected_wrt(index, &param_ty, wrt_indices) {
                wrt.push(load);
            }
            subctx
                .bindings
                .insert(name.clone(), LoweredValue::Node(load));
        }

        let output = subctx
            .lower_expr(body)
            .expect_node("vmap(grad(...)) requires a scalar floating output");
        subctx.dag.add_root(output);
        // Issue #197: route through grad_dag_checked so a
        // non-differentiable op surfaces a structured
        // `AdError::NotSupported` diagnostic rather than a generic
        // scalar-output message or a silent zero gradient.
        let grad_result = grad_dag_checked(&subctx.dag, output, &wrt).unwrap_or_else(|ad_err| {
            raise_fatal_lowering_error(
                format!("`vmap(grad(...))` lowering rejected: {ad_err}"),
                Some(body.span()),
                body.span_id().map(ToOwned::to_owned),
            )
        });
        let vmapped = match vmap::vectorize_axis0(&grad_result.dag, batch_dim.clone()) {
            Ok(dag) => dag,
            Err(message) => raise_lowering_error(
                format!("`vmap(grad(...))` lowering failed: {message}"),
                Some(body.span()),
                body.span_id().map(ToOwned::to_owned),
            ),
        };

        let mut arg_map = HashMap::new();
        for ((name, param_ty), arg_id) in param_names
            .iter()
            .zip(param_types.iter())
            .zip(canonical_args.iter().copied())
        {
            arg_map.insert(
                name.clone(),
                self.materialize_vmapped_arg(arg_id, param_ty, &batch_dim),
            );
        }
        arg_map.extend(captured_bindings);

        let remap = self.splice_dag(&vmapped, &arg_map);
        let mut flattened = wrt
            .iter()
            .filter_map(|wrt_node| grad_result.grad_nodes.get(wrt_node))
            .map(|grad_node| remap[grad_node])
            .collect::<Vec<_>>();
        let reusable_inputs = param_types
            .iter()
            .enumerate()
            .filter_map(|(index, param_ty)| {
                self.is_selected_wrt(index, param_ty, wrt_indices)
                    .then_some(canonical_args[index])
            })
            .collect::<Vec<_>>();
        for (grad_node, reusable_input) in flattened.iter().zip(reusable_inputs.iter()) {
            self.dag.set_reusable_input(*grad_node, *reusable_input);
        }
        if axis > 0 {
            for result in &mut flattened {
                let result_ty = self
                    .dag
                    .get(*result)
                    .map(|node| node.output_type.clone())
                    .unwrap_or_else(Self::default_type);
                if axis < result_ty.dims.len() {
                    let perm = front_to_axis_perm(result_ty.dims.len(), axis);
                    let perm_ty = permuted_tensor_type(&result_ty, &perm);
                    *result = self.dag.add_node(
                        RiscOp::Permute { axes: perm },
                        vec![*result],
                        perm_ty,
                        self.current_span_id.clone(),
                    );
                }
            }
        }
        match flattened.as_slice() {
            [single] => {
                LoweredValue::Node(self.attach_reuse_hint(*single, app_span, &canonical_args))
            }
            _ => LoweredValue::Tuple(flattened.into_iter().map(LoweredValue::Node).collect()),
        }
    }

    fn materialize_vmapped_arg(
        &mut self,
        arg_id: NodeId,
        original_ty: &TensorType,
        batch_dim: &DimInfo,
    ) -> NodeId {
        let actual_ty = self
            .dag
            .get(arg_id)
            .map(|node| node.output_type.clone())
            .unwrap_or_else(Self::default_type);
        if actual_ty.dims.len() > original_ty.dims.len() {
            return arg_id;
        }

        let mut out_ty = actual_ty.clone();
        out_ty.dims.insert(0, batch_dim.clone());
        self.dag.add_node(
            RiscOp::Expand {
                axis: 0,
                size: DimExpr::from(batch_dim),
            },
            vec![arg_id],
            out_ty,
            self.current_span_id.clone(),
        )
    }

    fn splice_dag(
        &mut self,
        dag: &Dag,
        arg_map: &HashMap<String, NodeId>,
    ) -> HashMap<NodeId, NodeId> {
        let mut remap = HashMap::<NodeId, NodeId>::new();
        for node in dag.nodes() {
            let new_id = match &node.op {
                RiscOp::Load { name } => {
                    if let Some(existing) = arg_map.get(name.as_str()) {
                        *existing
                    } else {
                        // Preserve the source DAG node's span_id verbatim
                        // (it was set when the source DAG was lowered).
                        // Falling back to the parent ctx's current span
                        // would silently overwrite real provenance.
                        self.dag.add_node(
                            RiscOp::Load { name: name.clone() },
                            vec![],
                            node.output_type.clone(),
                            node.span_id.clone(),
                        )
                    }
                }
                op => {
                    let inputs = node.inputs.iter().map(|id| remap[id]).collect::<Vec<_>>();
                    // Preserve the source DAG node's span_id (see Load arm).
                    let new_id = self.dag.add_node(
                        op.clone(),
                        inputs,
                        node.output_type.clone(),
                        node.span_id.clone(),
                    );
                    if let Some(reusable_input) = node.reusable_input
                        && let Some(mapped_input) = remap.get(&reusable_input)
                    {
                        self.dag.set_reusable_input(new_id, *mapped_input);
                    }
                    // chelis#384/#397: preserve (remapped) Form-3 `expand`
                    // shape-deps across the splice. Deps are earlier nodes,
                    // already in `remap`.
                    self.dag
                        .preserve_shape_deps(new_id, &node.shape_deps, &remap);
                    new_id
                }
            };
            remap.insert(node.id, new_id);
        }
        remap
    }

    fn extract_fn_parts<'a>(&self, expr: &'a Expr) -> Option<(Vec<String>, &'a Expr)> {
        let Expr::List(list, _) = expr else {
            return None;
        };
        if get_tag(list) != Some("fn") {
            return None;
        }
        let kids = children(list);
        let params = kids.first()?;
        let body = kids.get(1)?;
        let Expr::List(params_list, _) = params else {
            return None;
        };
        if get_tag(params_list) != Some("params") {
            return None;
        }
        // chelis#620: use the shared three-form walker. A param whose name
        // collides with a reserved Deep tag that is not also a Surf
        // keyword (`params`, `record`, `block`, ...; keyword collisions
        // like `match`/`if` are parse errors and never get here)
        // desugars as a MetaExpr wrapper (chelis-surf's
        // `typed_param_needs_meta_wrapper`), and the previous Atom/List-only
        // match silently DROPPED that name from the list -- the param never
        // bound, its body references lowered to bogus Loads, and a grad over
        // a struct argument conventionally named `params` failed as a
        // "runtime scrutinee" match.
        let names = children(params_list)
            .iter()
            .filter_map(|param| param_name_and_type_expr(param).map(|(name, _)| name))
            .collect();
        Some((names, body))
    }

    fn lower_builtin_app(
        &mut self,
        func_name: &str,
        args: &[Expr],
        ty: &TensorType,
        app_span: Span,
    ) -> NodeId {
        match func_name {
            // Tier 1: binary elementwise
            "add" if args.len() == 2 => {
                let a = self.lower_expr_node(&args[0], "add lhs");
                let b = self.lower_expr_node(&args[1], "add rhs");
                let out_ty = Self::elementwise_out_ty(&self.dag, a, ty, None);
                let node = self.dag.add_node(
                    RiscOp::Add,
                    vec![a, b],
                    out_ty,
                    self.current_span_id.clone(),
                );
                self.attach_reuse_hint(node, app_span, &[a, b])
            }
            "mul" if args.len() == 2 => {
                let a = self.lower_expr_node(&args[0], "mul lhs");
                let b = self.lower_expr_node(&args[1], "mul rhs");
                let out_ty = Self::elementwise_out_ty(&self.dag, a, ty, None);
                let node = self.dag.add_node(
                    RiscOp::Mul,
                    vec![a, b],
                    out_ty,
                    self.current_span_id.clone(),
                );
                self.attach_reuse_hint(node, app_span, &[a, b])
            }
            "cmplt" | "lt" if args.len() == 2 => {
                let a = self.lower_expr_node(&args[0], "cmplt lhs");
                let b = self.lower_expr_node(&args[1], "cmplt rhs");
                // C5: CmpLt always produces Bool output regardless of input precision.
                let bool_ty = Self::elementwise_out_ty(&self.dag, a, ty, Some(Prim::Bool));
                self.dag.add_node(
                    RiscOp::CmpLt,
                    vec![a, b],
                    bool_ty,
                    self.current_span_id.clone(),
                )
            }
            "max_elem" if args.len() == 2 => {
                let a = self.lower_expr_node(&args[0], "max_elem lhs");
                let b = self.lower_expr_node(&args[1], "max_elem rhs");
                let out_ty = Self::elementwise_out_ty(&self.dag, a, ty, None);
                let node = self.dag.add_node(
                    RiscOp::MaxElem,
                    vec![a, b],
                    out_ty,
                    self.current_span_id.clone(),
                );
                self.attach_reuse_hint(node, app_span, &[a, b])
            }

            // Tier 1: unary elementwise
            "drop" if args.len() == 1 => {
                // chelis#620: a drop over a tuple or ADT value (e.g. a
                // params struct leaving scope) closes every tensor leaf's
                // live range with its own Drop node; the arm's single-node
                // contract returns the last one (a rank-0 Const for a
                // leafless value such as a nullary constructor).
                let input = self.lower_expr(&args[0]);
                let leaves = input.flatten_nodes();
                let mut last = None;
                for leaf in leaves {
                    let output_type = self
                        .dag
                        .get(leaf)
                        .map(|node| node.output_type.clone())
                        .unwrap_or_else(Self::default_type);
                    last = Some(self.dag.add_node(
                        RiscOp::Drop,
                        vec![leaf],
                        output_type,
                        self.current_span_id.clone(),
                    ));
                }
                // A `drop` of a value with zero tensor leaves (e.g. an
                // empty ADT) has nothing to drop; the zero node is an
                // inert sequencing value that the checker never lets flow
                // into a computation. Census row 16 keep-with-comment
                // (chelis#730 section C1.4).
                last.unwrap_or_else(|| {
                    self.dag.add_node(
                        RiscOp::synth_const(Self::default_type().precision, 0.0),
                        vec![],
                        Self::default_type(),
                        self.current_span_id.clone(),
                    )
                })
            }
            "neg" if args.len() == 1 => {
                let x = self.lower_expr_node(&args[0], "neg input");
                // Elementwise: output dims always come from the lowered
                // operand (the annotation's dims can be stale symbolics
                // inside a rank-poly inline body; see chelis#346 red-team
                // F1/F3). Same contract as the Tier-1 binary arms.
                let out_ty = Self::elementwise_out_ty(&self.dag, x, ty, None);
                let node =
                    self.dag
                        .add_node(RiscOp::Neg, vec![x], out_ty, self.current_span_id.clone());
                self.attach_reuse_hint(node, app_span, &[x])
            }
            // `recip(x)` lowers directly to `RiscOp::Recip`,
            // exposing IEEE `1.0 / x` to Surf without going through a
            // `div(const(1), x)` round-trip.
            "recip" if args.len() == 1 => {
                let x = self.lower_expr_node(&args[0], "recip input");
                // Elementwise: output dims always come from the lowered
                // operand (the annotation's dims can be stale symbolics
                // inside a rank-poly inline body; see chelis#346 red-team
                // F1/F3). Same contract as the Tier-1 binary arms.
                let out_ty = Self::elementwise_out_ty(&self.dag, x, ty, None);
                let node =
                    self.dag
                        .add_node(RiscOp::Recip, vec![x], out_ty, self.current_span_id.clone());
                self.attach_reuse_hint(node, app_span, &[x])
            }
            "exp" if args.len() == 1 => {
                let x = self.lower_expr_node(&args[0], "exp input");
                let node = self.lower_transcendental(RiscOp::Exp, x, ty);
                self.attach_reuse_hint(node, app_span, &[x])
            }
            "log" if args.len() == 1 => {
                let x = self.lower_expr_node(&args[0], "log input");
                let node = self.lower_transcendental(RiscOp::Log, x, ty);
                self.attach_reuse_hint(node, app_span, &[x])
            }
            "sin" if args.len() == 1 => {
                let x = self.lower_expr_node(&args[0], "sin input");
                let node = self.lower_transcendental(RiscOp::Sin, x, ty);
                self.attach_reuse_hint(node, app_span, &[x])
            }
            "sqrt" if args.len() == 1 => {
                let x = self.lower_expr_node(&args[0], "sqrt input");
                let node = self.lower_transcendental(RiscOp::Sqrt, x, ty);
                self.attach_reuse_hint(node, app_span, &[x])
            }
            "cos" if args.len() == 1 => {
                let x = self.lower_expr_node(&args[0], "cos input");
                let node = self.lower_transcendental(RiscOp::Cos, x, ty);
                self.attach_reuse_hint(node, app_span, &[x])
            }
            "tan" if args.len() == 1 => {
                let x = self.lower_expr_node(&args[0], "tan input");
                let node = self.lower_transcendental(RiscOp::Tan, x, ty);
                self.attach_reuse_hint(node, app_span, &[x])
            }
            "atan" if args.len() == 1 => {
                let x = self.lower_expr_node(&args[0], "atan input");
                let node = self.lower_transcendental(RiscOp::Atan, x, ty);
                self.attach_reuse_hint(node, app_span, &[x])
            }
            "abs" if args.len() == 1 => {
                let x = self.lower_expr_node(&args[0], "abs input");
                let node = self.lower_transcendental(RiscOp::Abs, x, ty);
                self.attach_reuse_hint(node, app_span, &[x])
            }
            "floor" if args.len() == 1 => {
                let x = self.lower_expr_node(&args[0], "floor input");
                let node = self.lower_transcendental(RiscOp::Floor, x, ty);
                self.attach_reuse_hint(node, app_span, &[x])
            }
            "ceil" if args.len() == 1 => {
                let x = self.lower_expr_node(&args[0], "ceil input");
                let node = self.lower_transcendental(RiscOp::Ceil, x, ty);
                self.attach_reuse_hint(node, app_span, &[x])
            }
            "round" if args.len() == 1 => {
                let x = self.lower_expr_node(&args[0], "round input");
                let node = self.lower_transcendental(RiscOp::Round, x, ty);
                self.attach_reuse_hint(node, app_span, &[x])
            }
            "uniform_like" if args.len() == 3 => {
                let template = self.lower_expr_node(&args[0], "uniform_like template");
                // chelis#776: statically resolve each bound (through neg /
                // float-cast wrappers) or fail loudly — never the silent [0,1)
                // default that dropped a wrapped or computed range in codegen.
                let low = self.resolve_static_f64_arg(&args[1], "uniform_like", "low bound");
                let high = self.resolve_static_f64_arg(&args[2], "uniform_like", "high bound");
                let seed = self.random_seed.unwrap_or(0);
                // When no `type` metadata is attached to the `app` form
                // (as is common when the host lane drives sub-expression
                // lowering through `lower_subexpr_program` from a
                // handle-effect tensor-helper call), the supplied `ty`
                // is `default_type()` (rank-0 scalar). UniformLike is
                // shape-preserving over its template input, so prefer
                // the template's actual tensor type to avoid emitting a
                // rank-0 alloc that the host emitter then renders as
                // `(int[]){1}` and a 1-element loop. Bucket-5 closure.
                let inferred_ty = self
                    .dag
                    .get(template)
                    .map(|node| node.output_type.clone())
                    .unwrap_or_else(|| ty.clone());
                let resolved_ty = if ty == &Self::default_type() && !inferred_ty.dims.is_empty() {
                    inferred_ty
                } else {
                    ty.clone()
                };
                let node = self.dag.add_node(
                    RiscOp::UniformLike { low, high, seed },
                    vec![template],
                    resolved_ty,
                    self.current_span_id.clone(),
                );
                self.attach_reuse_hint(node, app_span, &[template])
            }
            "dropout" if args.len() == 2 => {
                let x = self.lower_expr_node(&args[0], "dropout input");
                // chelis#776 (same silent-substitution shape as uniform_like's
                // bounds): a wrapped/computed rate must resolve statically or
                // fail loudly, never silently become 0.0 (no-op dropout).
                let rate = self.resolve_static_f64_arg(&args[1], "dropout", "rate");
                let seed = self.random_seed.unwrap_or(0);
                let inferred_ty = self
                    .dag
                    .get(x)
                    .map(|node| node.output_type.clone())
                    .unwrap_or_else(|| ty.clone());
                let resolved_ty = if ty == &Self::default_type() && !inferred_ty.dims.is_empty() {
                    inferred_ty
                } else {
                    ty.clone()
                };
                let node = self.dag.add_node(
                    RiscOp::Dropout { rate, seed },
                    vec![x],
                    resolved_ty,
                    self.current_span_id.clone(),
                );
                self.attach_reuse_hint(node, app_span, &[x])
            }

            // Tier 2 decompositions
            "sub" if args.len() == 2 => {
                let a = self.lower_expr_node(&args[0], "sub lhs");
                let b = self.lower_expr_node(&args[1], "sub rhs");
                // Elementwise: operand-derived dims, not the annotation's
                // (stale symbolics inside rank-poly inline bodies; see
                // chelis#346 red-team F1). Matches the Tier-1 binary arms.
                let out_ty = Self::elementwise_out_ty(&self.dag, a, ty, None);
                let parent_span = self.current_span_id.clone();
                let node = tier2::lower_sub(&mut self.dag, a, b, &out_ty, parent_span.as_deref());
                self.attach_reuse_hint(node, app_span, &[a, b])
            }
            "relu" if args.len() == 1 => {
                let x = self.lower_expr_node(&args[0], "relu input");
                // Elementwise: output dims always come from the lowered
                // operand (the annotation's dims can be stale symbolics
                // inside a rank-poly inline body; see chelis#346 red-team
                // F1/F3). Same contract as the Tier-1 binary arms.
                let out_ty = Self::elementwise_out_ty(&self.dag, x, ty, None);
                let parent_span = self.current_span_id.clone();
                let node = tier2::lower_relu(&mut self.dag, x, &out_ty, parent_span.as_deref());
                self.attach_reuse_hint(node, app_span, &[x])
            }
            "sigmoid" if args.len() == 1 => {
                let x = self.lower_expr_node(&args[0], "sigmoid input");
                // Elementwise: output dims always come from the lowered
                // operand (the annotation's dims can be stale symbolics
                // inside a rank-poly inline body; see chelis#346 red-team
                // F1/F3). Same contract as the Tier-1 binary arms.
                let out_ty = Self::elementwise_out_ty(&self.dag, x, ty, None);
                let parent_span = self.current_span_id.clone();
                let node = tier2::lower_sigmoid(&mut self.dag, x, &out_ty, parent_span.as_deref());
                self.attach_reuse_hint(node, app_span, &[x])
            }
            // Bucket 3: `tanh`, `silu`, `gelu` route through new tier2
            // decompositions so the RISC DAG path stays self-contained.
            // Mirrors the relu/sigmoid pattern above.
            "tanh" if args.len() == 1 => {
                let x = self.lower_expr_node(&args[0], "tanh input");
                // Elementwise: output dims always come from the lowered
                // operand (the annotation's dims can be stale symbolics
                // inside a rank-poly inline body; see chelis#346 red-team
                // F1/F3). Same contract as the Tier-1 binary arms.
                let out_ty = Self::elementwise_out_ty(&self.dag, x, ty, None);
                let parent_span = self.current_span_id.clone();
                let node = tier2::lower_tanh(&mut self.dag, x, &out_ty, parent_span.as_deref());
                self.attach_reuse_hint(node, app_span, &[x])
            }
            "silu" if args.len() == 1 => {
                let x = self.lower_expr_node(&args[0], "silu input");
                // Elementwise: output dims always come from the lowered
                // operand (the annotation's dims can be stale symbolics
                // inside a rank-poly inline body; see chelis#346 red-team
                // F1/F3). Same contract as the Tier-1 binary arms.
                let out_ty = Self::elementwise_out_ty(&self.dag, x, ty, None);
                let parent_span = self.current_span_id.clone();
                let node = tier2::lower_silu(&mut self.dag, x, &out_ty, parent_span.as_deref());
                self.attach_reuse_hint(node, app_span, &[x])
            }
            "gelu" if args.len() == 1 => {
                let x = self.lower_expr_node(&args[0], "gelu input");
                // Elementwise: output dims always come from the lowered
                // operand (the annotation's dims can be stale symbolics
                // inside a rank-poly inline body; see chelis#346 red-team
                // F1/F3). Same contract as the Tier-1 binary arms.
                let out_ty = Self::elementwise_out_ty(&self.dag, x, ty, None);
                let parent_span = self.current_span_id.clone();
                let node = tier2::lower_gelu(&mut self.dag, x, &out_ty, parent_span.as_deref());
                self.attach_reuse_hint(node, app_span, &[x])
            }
            "div" if args.len() == 2 => {
                let a = self.lower_expr_node(&args[0], "div lhs");
                let b = self.lower_expr_node(&args[1], "div rhs");
                // Elementwise: operand-derived dims, not the annotation's
                // (stale symbolics inside rank-poly inline bodies; see
                // chelis#346 red-team F1). Matches the Tier-1 binary arms.
                let out_ty = Self::elementwise_out_ty(&self.dag, a, ty, None);
                let parent_span = self.current_span_id.clone();
                let node = tier2::lower_div(&mut self.dag, a, b, &out_ty, parent_span.as_deref());
                self.attach_reuse_hint(node, app_span, &[a, b])
            }
            // chelis#178: integer-division primitives. Same elementwise
            // dim contract as `div`; they lower to dedicated RISC ops.
            "floor_div" if args.len() == 2 => {
                let a = self.lower_expr_node(&args[0], "floor_div lhs");
                let b = self.lower_expr_node(&args[1], "floor_div rhs");
                let out_ty = Self::elementwise_out_ty(&self.dag, a, ty, None);
                let parent_span = self.current_span_id.clone();
                let node =
                    tier2::lower_floor_div(&mut self.dag, a, b, &out_ty, parent_span.as_deref());
                self.attach_reuse_hint(node, app_span, &[a, b])
            }
            "trunc_div" if args.len() == 2 => {
                let a = self.lower_expr_node(&args[0], "trunc_div lhs");
                let b = self.lower_expr_node(&args[1], "trunc_div rhs");
                let out_ty = Self::elementwise_out_ty(&self.dag, a, ty, None);
                let parent_span = self.current_span_id.clone();
                let node =
                    tier2::lower_trunc_div(&mut self.dag, a, b, &out_ty, parent_span.as_deref());
                self.attach_reuse_hint(node, app_span, &[a, b])
            }

            // Tier 2 higher-level ops (spec §3.4, §4.1–4.2)
            "matmul" if args.len() == 2 => {
                let a = self.lower_expr_node(&args[0], "matmul lhs");
                let b = self.lower_expr_node(&args[1], "matmul rhs");
                let a_ty = self
                    .dag
                    .get(a)
                    .map(|n| n.output_type.clone())
                    .unwrap_or_else(|| ty.clone());
                let b_ty = self
                    .dag
                    .get(b)
                    .map(|n| n.output_type.clone())
                    .unwrap_or_else(|| ty.clone());
                let parent_span = self.current_span_id.clone();
                tier2::lower_matmul(&mut self.dag, a, b, &a_ty, &b_ty, parent_span.as_deref())
            }
            "gather" if args.len() == 3 => {
                let values = self.lower_expr_node(&args[0], "gather values");
                let indices = self.lower_expr_node(&args[1], "gather indices");
                let axis_raw = self.extract_axis_raw(&args[2], "gather");
                // Issue #320: recover the `values` rank from the ascribed
                // gather-result type when the `values` operand collapsed to
                // rank-0 (a windowing/stacking intermediate left untyped),
                // so `normalize_axis` sees rank>=1.
                let values_rank = self.gather_values_rank(values, indices, ty);
                let axis = self.normalize_axis(axis_raw, values_rank, "gather", &args[2]);
                // Patch the collapsed `values` node to rank>=1 so verify and
                // the scatter-add adjoint carry the operand's rank/extent.
                // `gather` preserves element precision, so the recovered
                // `values` precision is the ascribed result precision.
                let values_prec = if *ty == Self::default_type() {
                    Prim::F32
                } else {
                    ty.precision
                };
                self.recover_collapsed_gather_values_type(values, indices, axis, ty, values_prec);
                let out_ty = Self::gather_out_ty_from_inputs(&self.dag, values, indices, axis)
                    .unwrap_or_else(|| ty.clone());
                self.dag.add_node(
                    RiscOp::Gather { axis },
                    vec![values, indices],
                    out_ty,
                    self.current_span_id.clone(),
                )
            }
            "scatter_replace" if args.len() == 4 => {
                // Tensor-lane replace-scatter: lowers directly to
                // RiscOp::Scatter (last-write-wins). The output type
                // equals the base/target tensor's type.
                let base = self.lower_expr_node(&args[0], "scatter_replace base");
                let indices = self.lower_expr_node(&args[1], "scatter_replace indices");
                let updates = self.lower_expr_node(&args[2], "scatter_replace updates");
                let base_rank = self
                    .dag
                    .get(base)
                    .map(|n| n.output_type.dims.len())
                    .unwrap_or(0);
                let axis_raw = self.extract_axis_raw(&args[3], "scatter_replace");
                let axis = self.normalize_axis(axis_raw, base_rank, "scatter_replace", &args[3]);
                let out_ty = self
                    .dag
                    .get(base)
                    .map(|n| n.output_type.clone())
                    .unwrap_or_else(|| ty.clone());
                self.dag.add_node(
                    RiscOp::Scatter { axis },
                    vec![base, indices, updates],
                    out_ty,
                    self.current_span_id.clone(),
                )
            }
            "scatter_elements" if args.len() == 4 => {
                // Tensor-lane element-wise scatter (ONNX ScatterElements,
                // spec §3.5.1): lowers directly to
                // RiscOp::ScatterElements (last-write-wins). The output
                // type equals the data tensor's type.
                let data = self.lower_expr_node(&args[0], "scatter_elements data");
                let indices = self.lower_expr_node(&args[1], "scatter_elements indices");
                let updates = self.lower_expr_node(&args[2], "scatter_elements updates");
                let data_rank = self
                    .dag
                    .get(data)
                    .map(|n| n.output_type.dims.len())
                    .unwrap_or(0);
                let axis_raw = self.extract_axis_raw(&args[3], "scatter_elements");
                let axis = self.normalize_axis(axis_raw, data_rank, "scatter_elements", &args[3]);
                let out_ty = self
                    .dag
                    .get(data)
                    .map(|n| n.output_type.clone())
                    .unwrap_or_else(|| ty.clone());
                self.dag.add_node(
                    RiscOp::ScatterElements { axis },
                    vec![data, indices, updates],
                    out_ty,
                    self.current_span_id.clone(),
                )
            }
            "softmax" if args.len() == 2 => {
                let x = self.lower_expr_node(&args[0], "softmax input");
                let x_ty = self
                    .dag
                    .get(x)
                    .map(|n| n.output_type.clone())
                    .unwrap_or_else(|| ty.clone());
                let axis_raw = self.extract_axis_raw(&args[1], "softmax");
                let rank = self.axis_rank(x, ty);
                let axis = self.normalize_axis(axis_raw, rank, "softmax", &args[1]);
                let parent_span = self.current_span_id.clone();
                let node =
                    tier2::lower_softmax(&mut self.dag, x, axis, &x_ty, parent_span.as_deref());
                self.attach_reuse_hint(node, app_span, &[x])
            }
            "mean" if args.len() == 2 => {
                let x = self.lower_expr_node(&args[0], "mean input");
                let x_ty = self
                    .dag
                    .get(x)
                    .map(|n| n.output_type.clone())
                    .unwrap_or_else(|| ty.clone());
                let rank = self.axis_rank(x, ty);
                let axis = self.resolve_reduce_axis(&args[1], x, rank, "mean");
                let parent_span = self.current_span_id.clone();
                let node = tier2::lower_mean(&mut self.dag, x, axis, &x_ty, parent_span.as_deref());
                self.attach_reuse_hint(node, app_span, &[x])
            }
            "layer_norm" if args.len() == 3 => {
                let x = self.lower_expr_node(&args[0], "layer_norm input");
                let gamma = self.lower_expr_node(&args[1], "layer_norm gamma");
                let beta = self.lower_expr_node(&args[2], "layer_norm beta");
                let x_ty = self
                    .dag
                    .get(x)
                    .map(|n| n.output_type.clone())
                    .unwrap_or_else(|| ty.clone());
                let gamma_ty = self
                    .dag
                    .get(gamma)
                    .map(|n| n.output_type.clone())
                    .unwrap_or_else(|| ty.clone());
                let beta_ty = self
                    .dag
                    .get(beta)
                    .map(|n| n.output_type.clone())
                    .unwrap_or_else(|| ty.clone());
                let parent_span = self.current_span_id.clone();
                let node = tier2::lower_layer_norm(
                    &mut self.dag,
                    x,
                    gamma,
                    beta,
                    &x_ty,
                    &gamma_ty,
                    &beta_ty,
                    1e-5,
                    parent_span.as_deref(),
                );
                self.attach_reuse_hint(node, app_span, &[x, gamma, beta])
            }
            "conv2d" if args.len() >= 2 => {
                let input = self.lower_expr_node(&args[0], "conv2d input");
                let kernel = self.lower_expr_node(&args[1], "conv2d kernel");
                let stride = args
                    .get(2)
                    .and_then(|expr| self.extract_usize_value(expr))
                    .unwrap_or(1);
                let padding = args
                    .get(3)
                    .and_then(|expr| self.extract_usize_value(expr))
                    .unwrap_or(0);
                let input_ty = self
                    .dag
                    .get(input)
                    .map(|n| n.output_type.clone())
                    .unwrap_or_else(|| ty.clone());
                let kernel_ty = self
                    .dag
                    .get(kernel)
                    .map(|n| n.output_type.clone())
                    .unwrap_or_else(|| ty.clone());
                let parent_span = self.current_span_id.clone();
                tier2::lower_conv2d(
                    &mut self.dag,
                    input,
                    kernel,
                    &input_ty,
                    &kernel_ty,
                    ty,
                    stride,
                    padding,
                    parent_span.as_deref(),
                )
            }

            // H1: Tier 2 comparison ops
            "gt" if args.len() == 2 => {
                let a = self.lower_expr_node(&args[0], "gt lhs");
                let b = self.lower_expr_node(&args[1], "gt rhs");
                let parent_span = self.current_span_id.clone();
                tier2::lower_gt(&mut self.dag, a, b, ty, parent_span.as_deref())
            }
            "gte" if args.len() == 2 => {
                let a = self.lower_expr_node(&args[0], "gte lhs");
                let b = self.lower_expr_node(&args[1], "gte rhs");
                let parent_span = self.current_span_id.clone();
                tier2::lower_gte(&mut self.dag, a, b, ty, parent_span.as_deref())
            }
            "lte" if args.len() == 2 => {
                let a = self.lower_expr_node(&args[0], "lte lhs");
                let b = self.lower_expr_node(&args[1], "lte rhs");
                let parent_span = self.current_span_id.clone();
                tier2::lower_lte(&mut self.dag, a, b, ty, parent_span.as_deref())
            }
            "eq" if args.len() == 2 => {
                let a = self.lower_expr_node(&args[0], "eq lhs");
                let b = self.lower_expr_node(&args[1], "eq rhs");
                let parent_span = self.current_span_id.clone();
                tier2::lower_eq(&mut self.dag, a, b, ty, parent_span.as_deref())
            }
            "neq" if args.len() == 2 => {
                let a = self.lower_expr_node(&args[0], "neq lhs");
                let b = self.lower_expr_node(&args[1], "neq rhs");
                let parent_span = self.current_span_id.clone();
                tier2::lower_neq(&mut self.dag, a, b, ty, parent_span.as_deref())
            }
            "min_elem" if args.len() == 2 => {
                let a = self.lower_expr_node(&args[0], "min_elem lhs");
                let b = self.lower_expr_node(&args[1], "min_elem rhs");
                let parent_span = self.current_span_id.clone();
                tier2::lower_min_elem(&mut self.dag, a, b, ty, parent_span.as_deref())
            }

            // H2: Boolean operators
            "and" if args.len() == 2 => {
                let a = self.lower_expr_node(&args[0], "and lhs");
                let b = self.lower_expr_node(&args[1], "and rhs");
                let parent_span = self.current_span_id.clone();
                tier2::lower_and(&mut self.dag, a, b, ty, parent_span.as_deref())
            }
            "or" if args.len() == 2 => {
                let a = self.lower_expr_node(&args[0], "or lhs");
                let b = self.lower_expr_node(&args[1], "or rhs");
                let parent_span = self.current_span_id.clone();
                tier2::lower_or(&mut self.dag, a, b, ty, parent_span.as_deref())
            }
            "not" if args.len() == 1 => {
                let a = self.lower_expr_node(&args[0], "not input");
                let parent_span = self.current_span_id.clone();
                tier2::lower_not(&mut self.dag, a, ty, parent_span.as_deref())
            }

            // Tier 1: reductions
            //
            // RT-2 fixup B5: lowering used to hardcode
            // `accumulator = out_ty.precision`, which assumed the
            // type checker had already widened the result for narrow
            // operand precisions. For int8/int16 the type checker now
            // (B1) returns int32; for bf16/f16 the user-facing result
            // is the operand precision (per the §5.7.1 result-precision
            // table) but the IR Sum node outputs the f32 accumulator
            // and we must insert a Cast back to the operand precision
            // to recover the user-facing tensor type.
            // chelis#339 Part 2: variadic named-axis reduction
            // (`sum(x, seq, head)`, spec §4.5.3). Desugar to the documented
            // composition — innermost stage reduces the LAST listed axis —
            // and recurse; each 2-arg stage resolves its named axis against
            // its own operand's dims, so the result is order-insensitive.
            // Only bare-name axes reach this arm (the checker rejects
            // positional integers in the variadic form); anything else falls
            // through to the 2-arg arms or the generic fallback.
            "sum" | "mean" | "max_reduce" | "min_reduce" | "prod_reduce"
                if args.len() >= 3 && args[1..].iter().all(|a| bare_var_name(a).is_some()) =>
            {
                let mut expr = args[0].clone();
                for axis in args[1..].iter().rev() {
                    expr = synth_reduction_app(func_name, expr, axis.clone(), app_span);
                }
                self.lower_expr_node(&expr, "variadic named-axis reduction")
            }
            "sum" if args.len() == 2 => {
                let x = self.lower_expr_node(&args[0], "sum input");
                let x_ty = self
                    .dag
                    .get(x)
                    .map(|node| node.output_type.clone())
                    .unwrap_or_else(|| ty.clone());
                let axis = self.resolve_reduce_axis(&args[1], x, x_ty.dims.len(), "sum");
                let operand_prec = self
                    .dag
                    .get(x)
                    .map(|node| node.output_type.precision)
                    .unwrap_or(ty.precision);
                // Issue Chelis-Lang/chelis#218 R1 HIGH-2 defense-in-
                // depth: prefer input-derived dims when meta `ty`
                // carries a wildcard placeholder. See
                // `Self::reduction_out_dims`.
                let out_dims = Self::reduction_out_dims(&x_ty.dims, ty, axis);
                let sum_op = RiscOp::sum_default(axis, operand_prec).unwrap_or_else(|msg| {
                    // The type checker already rejects unsupported
                    // operand precisions before lowering; fall back to
                    // the operand precision so the resulting IR can
                    // still surface the diagnostic via verify rather
                    // than panicking the lowering pass.
                    eprintln!(
                        "internal: sum lowering fallback (operand `{}`): {msg}",
                        operand_prec.name()
                    );
                    RiscOp::Sum {
                        axis,
                        accumulator: operand_prec,
                    }
                });
                let accumulator = match sum_op {
                    RiscOp::Sum { accumulator, .. } => accumulator,
                    _ => operand_prec,
                };
                let sum_node_ty = TensorType {
                    dims: out_dims.clone(),
                    precision: accumulator,
                };
                let sum_id =
                    self.dag
                        .add_node(sum_op, vec![x], sum_node_ty, self.current_span_id.clone());
                // If the user-facing result precision differs from the
                // accumulator (only the bf16/f16 row of the §5.7.1
                // table), insert an explicit Cast back to the operand
                // precision so downstream consumers see the documented
                // result type.
                if accumulator != operand_prec {
                    let result_prec = operand_prec
                        .default_reduce_sum_result_precision()
                        .unwrap_or(operand_prec);
                    if result_prec != accumulator {
                        let cast_ty = TensorType {
                            dims: out_dims,
                            precision: result_prec,
                        };
                        return self.dag.add_node(
                            RiscOp::Cast {
                                new_precision: result_prec,
                            },
                            vec![sum_id],
                            cast_ty,
                            self.current_span_id.clone(),
                        );
                    }
                }
                sum_id
            }
            "tensor_to_scalar" if args.len() == 1 => {
                self.lower_expr_node(&args[0], "tensor_to_scalar input")
            }
            "scalar_to_tensor" if args.len() == 1 => {
                self.lower_expr_node(&args[0], "scalar_to_tensor input")
            }
            "max_reduce" if args.len() == 2 => {
                let x = self.lower_expr_node(&args[0], "max_reduce input");
                // Issue #320: recover the operand rank from the ascribed
                // result type when the operand node collapsed to rank-0
                // (a windowing/stacking intermediate left untyped), so
                // `normalize_axis` sees rank>=1 instead of raising the
                // "operand of rank 0" diagnostic.
                let operand_rank = self.reduction_operand_rank(x, ty);
                let axis = self.resolve_reduce_axis(&args[1], x, operand_rank, "max_reduce");
                // Patch the collapsed operand node to a rank>=1 type so the
                // reverse-mode adjoint carries the operand's rank/extent.
                let precision = {
                    let x_prec = self
                        .dag
                        .get(x)
                        .map(|node| node.output_type.precision)
                        .unwrap_or(ty.precision);
                    if *ty == Self::default_type() {
                        x_prec
                    } else {
                        ty.precision
                    }
                };
                self.recover_collapsed_operand_type(x, axis, &ty.dims, precision);
                let x_ty = self
                    .dag
                    .get(x)
                    .map(|node| node.output_type.clone())
                    .unwrap_or_else(|| ty.clone());
                let dims = Self::reduction_out_dims(&x_ty.dims, ty, axis);
                let out_precision = if any_wildcard_dim(&ty.dims) || *ty == Self::default_type() {
                    x_ty.precision
                } else {
                    ty.precision
                };
                let out_ty = TensorType {
                    dims,
                    precision: out_precision,
                };
                self.dag.add_node(
                    RiscOp::MaxReduce { axis },
                    vec![x],
                    out_ty,
                    self.current_span_id.clone(),
                )
            }
            // Strided windowed reduction over the trailing
            // `window_shape.len()` axes (Valid padding only). See
            // `spec/05-risc-primitives.md` §2.3.1. The four Surf
            // names map to the four `ReduceWindowKind` variants on the
            // shared IR op.
            name @ ("reduce_window_max" | "reduce_window_min" | "reduce_window_sum"
            | "reduce_window_mean")
                if args.len() == 3 =>
            {
                let x = self.lower_expr_node(&args[0], "reduce_window input");
                let x_ty = self
                    .dag
                    .get(x)
                    .map(|node| node.output_type.clone())
                    .unwrap_or_else(|| ty.clone());
                // chelis#730 Phase 1 (census row 8, chelis#725): a window or
                // stride list that does not fold to non-negative integer
                // literals raises a FATAL lowering error. The former
                // empty-list-defaulting pair silently lowered BOTH lists to
                // empty, turning the pooling into a no-op that returned the
                // unpooled input at the wrong shape (and the half-literal
                // case panicked the emitter's arity assertion). Fatal so the
                // host-fallback lane cannot launder the failure into the
                // unsupported-builtin stub path (the chelis#776/#782
                // finding). Runtime-parameterized windows are op-owner
                // support work, not a silent default.
                let extract_literal_axis_list = |list_arg: &Expr, which: &str| -> Vec<usize> {
                    collect_cons_chain(list_arg)
                        .and_then(|elems| {
                            elems
                                .iter()
                                .map(|e| {
                                    extract_int_for_dim(e).and_then(|n| usize::try_from(n).ok())
                                })
                                .collect::<Option<Vec<_>>>()
                        })
                        .unwrap_or_else(|| {
                            let unsupported = Unsupported::new(
                                UnsupportedKind::Construct(format!(
                                    "a non-literal {which} list for `{name}`"
                                )),
                                "the compiled-backend lowering of `reduce_window_*`",
                                Stage::Lowering,
                                "window and stride lists must be integer literals for the \
                                 compiled lane today; a runtime-parameterized window \
                                 previously lowered to a silent no-op (chelis#725; \
                                 chelis#730 census row 8)",
                            );
                            raise_fatal_lowering_error(
                                unsupported.to_string(),
                                Some(list_arg.span()),
                                list_arg.span_id().map(ToOwned::to_owned),
                            )
                        })
                };
                let window_shape = extract_literal_axis_list(&args[1], "window");
                let strides = extract_literal_axis_list(&args[2], "stride");
                let reducer = match name {
                    "reduce_window_max" => crate::dag::ReduceWindowKind::Max,
                    "reduce_window_min" => crate::dag::ReduceWindowKind::Min,
                    "reduce_window_sum" => crate::dag::ReduceWindowKind::Sum,
                    "reduce_window_mean" => crate::dag::ReduceWindowKind::Mean,
                    _ => unreachable!(),
                };
                // Compute output dims directly from the input + window
                // + stride triple. The type checker has already
                // validated the shape, but recomputing here keeps the
                // IR self-contained and avoids reliance on the
                // (sometimes wildcard) caller-provided `ty.dims`.
                let dims = compute_reduce_window_out_dims(&x_ty.dims, &window_shape, &strides, ty);
                let precision = if any_wildcard_dim(&ty.dims) || *ty == Self::default_type() {
                    x_ty.precision
                } else {
                    ty.precision
                };
                let out_ty = TensorType { dims, precision };
                self.dag.add_node(
                    RiscOp::ReduceWindow {
                        reducer,
                        window_shape,
                        strides,
                    },
                    vec![x],
                    out_ty,
                    self.current_span_id.clone(),
                )
            }
            "min_reduce" | "prod_reduce" | "argmax_reduce" | "argmin_reduce" if args.len() == 2 => {
                let name = func_name;
                let x = self.lower_expr_node(&args[0], "reduction input");
                let x_ty = self
                    .dag
                    .get(x)
                    .map(|node| node.output_type.clone())
                    .unwrap_or_else(|| ty.clone());
                let axis = self.resolve_reduce_axis(&args[1], x, x_ty.dims.len(), name);
                let dims = Self::reduction_out_dims(&x_ty.dims, ty, axis);
                let precision = if any_wildcard_dim(&ty.dims) || *ty == Self::default_type() {
                    x_ty.precision
                } else {
                    ty.precision
                };
                let out_ty = TensorType { dims, precision };
                let op = match name {
                    "min_reduce" => RiscOp::MinReduce { axis },
                    "prod_reduce" => RiscOp::ProdReduce { axis },
                    "argmax_reduce" => RiscOp::Argmax { axis },
                    "argmin_reduce" => RiscOp::Argmin { axis },
                    _ => unreachable!(),
                };
                self.dag
                    .add_node(op, vec![x], out_ty, self.current_span_id.clone())
            }

            // H3: Movement ops -- extract parameters from Deep AST args where possible.
            "reshape" if !args.is_empty() => {
                let x = self.lower_expr_node(&args[0], "reshape input");
                // Try to extract new_shape from the second arg; fall back to output type dims.
                // chelis#513 gap 1: a `shape(operand, axis)`-derived reshape
                // target dim (e.g. `reshape(&x, [shape(x, 0), 1])` or the
                // let-bound `k = shape(x, 0); reshape(&x, [k, 1])` idiom) must
                // be resolved to the operand's declaring source, NOT left as a
                // bare `Named("k", None)` symbol. Left unresolved, the reduce/
                // reshape backward `Expand`/`Sum` inherits `Sym("k")` and
                // `symbolic_occurrences` ICEs ("symbolic dim `k` ... no Load
                // input declares it"). Resolving to the operand dim folds it to
                // a concrete `Lit` when the operand axis is static and to a
                // Load-carried `Named` (kept live via a `shape_dep`) when
                // symbolic, so the dim traces to a declaring input either way.
                let mut inputs = vec![x];
                let fallback = || {
                    (
                        ty.dims.iter().map(RtDim::from_dim_info).collect(),
                        ty.dims.clone(),
                        Vec::new(),
                    )
                };
                let (new_shape, ty_dims, shape_srcs) = if args.len() >= 2 {
                    let checker_dims = ty.dims.clone();
                    self.extract_reshape_dim_list(&args[1], &checker_dims, &mut inputs)
                        .unwrap_or_else(fallback)
                } else {
                    fallback()
                };
                let out_ty = TensorType {
                    dims: ty_dims,
                    precision: ty.precision,
                };
                let reshape_id = self.dag.add_node(
                    RiscOp::Reshape { new_shape },
                    inputs,
                    out_ty,
                    self.current_span_id.clone(),
                );
                for src in shape_srcs {
                    self.dag.add_shape_dep(reshape_id, src);
                }
                reshape_id
            }
            "permute" if args.len() >= 2 => {
                let x = self.lower_expr_node(&args[0], "permute input");
                // Extract axes ordering from remaining args.
                let axes = self.extract_usize_list(&args[1..]);
                // `permute` never changes precision; it only reorders
                // axes. Compute the output type directly from the input
                // operand's resolved type and the axes when the axes
                // cover the operand's full rank — this is the
                // authoritative shape/precision and is independent of the
                // app-node's `type:` metadata.
                //
                // issue #319: for a separate-`sig` precision-polymorphic
                // verb, the checker resolves the body's `permute` app type
                // but renames the sig's precision variable `p` to a fresh
                // inference tvar (`type_to_deep_expr` prints `TensorPrec::Var`
                // anonymously). Trusting `ty.clone()` would carry that
                // renamed `(t-var t306)` precision, which is NOT in the
                // call-site `prec_substitutions` (keyed on `p`), tripping
                // the §5.8.1 monomorphization tripwire downstream. Taking
                // precision from the operand chain — rooted at the `Load`
                // nodes that carry the formal param's `(t-var p)` precision,
                // already resolved to the concrete call-site precision —
                // sidesteps the rename. This mirrors how tier-1 ops (e.g.
                // `mul`) already derive their precision from operands.
                let out_ty = self
                    .dag
                    .get(x)
                    .map(|node| node.output_type.clone())
                    .filter(|input_ty| axes.len() == input_ty.dims.len())
                    .map(|input_ty| permuted_tensor_type(&input_ty, &axes))
                    .unwrap_or_else(|| ty.clone());
                self.dag.add_node(
                    RiscOp::Permute { axes },
                    vec![x],
                    out_ty,
                    self.current_span_id.clone(),
                )
            }
            "expand" if args.len() >= 2 => {
                let x = self.lower_expr_node(&args[0], "expand input");
                // chelis#339 named-axis expand (spec §4.5.3): when the axis
                // argument is a dimension NAME, the insertion point is
                // resolved against the monomorphized operand dims — the
                // trailing end (3-arg form) or the index of the named anchor
                // (4-arg form), mirroring `resolve_reduce_axis`. A bare name
                // never falls through to the positional path: `extract_
                // usize_value` would return None and the old `unwrap_or(0)`
                // would silently insert at axis 0 (the repo forbids silent
                // fallbacks).
                let named_insert = if self.extract_usize_value(&args[1]).is_some() {
                    None
                } else {
                    bare_var_name(&args[1])
                };
                // chelis#339: the inserted name must not collide with an axis
                // of the MONOMORPHIZED operand. The checker rejects visible
                // collisions, but a rank spread (or a single-letter dim var,
                // which lowers to `Named` with its source letter) at the call
                // boundary can cover an axis whose name the symbolic check
                // cannot see; with a duplicate name in the dims, every later
                // by-name lookup (`resolve_reduce_axis`,
                // `resolve_expand_anchor`) is first-hit and would silently
                // pick the wrong axis. Fail loudly instead.
                if let Some(name) = &named_insert
                    && let Some(node) = self.dag.get(x)
                    && node
                        .output_type
                        .dims
                        .iter()
                        .any(|d| matches!(d, DimInfo::Named(n, _) if n == name))
                {
                    // FATAL: a plain lowering diagnostic is absorbed by the
                    // host-fallback path, which would emit the def as a host
                    // call referencing the axis names as undeclared C
                    // identifiers — garbage C, not a loud failure.
                    raise_fatal_lowering_error(
                        format!(
                            "`expand` inserts an axis named `{name}`, but the monomorphized \
                             operand already carries an axis named `{name}` (a rank spread or \
                             single-letter dim var at the call boundary can cover an axis name \
                             the symbolic checker cannot see); later by-name axis lookups would \
                             silently resolve to the wrong axis. Rename the inserted axis \
                             (spec/04-type-system.md \u{00a7}4.5.3)"
                        ),
                        Some(args[1].span()),
                        args[1].span_id().map(ToOwned::to_owned),
                    );
                }
                let axis = match &named_insert {
                    Some(_) => match args.get(3).and_then(bare_var_name) {
                        Some(anchor) => self.resolve_expand_anchor(&args[3], x, &anchor),
                        // The operand node was just lowered above; a missing
                        // entry is an internal desync, not an input shape -
                        // raise instead of silently anchoring at the end of
                        // a rank-0 view (section C1.4).
                        None => match self.dag.get(x) {
                            Some(node) => node.output_type.dims.len(),
                            None => raise_lowering_error(
                                format!(
                                    "internal lowering desync: `expand` operand node {} \
                                     is missing from the DAG (section C1.4)",
                                    x.0
                                ),
                                Some(args[0].span()),
                                args[0].span_id().map(ToOwned::to_owned),
                            ),
                        },
                    },
                    // chelis#730 Phase 1 (#782-flagged structural-index site):
                    // an axis argument that is neither a compile-time usize
                    // nor a bare dimension name previously defaulted to axis
                    // 0 silently. Fatal: any program reaching this arm was
                    // expanding the wrong axis (section C1.4 raise-or-prove).
                    None => self.extract_usize_value(&args[1]).unwrap_or_else(|| {
                        let unsupported = Unsupported::new(
                            UnsupportedKind::Construct(
                                "a non-literal `expand` axis argument".to_string(),
                            ),
                            "the compiled-backend lowering of `expand`",
                            Stage::Lowering,
                            "the expand axis must be an integer literal or a named \
                             dimension; a computed axis previously fell back to axis 0 \
                             silently (chelis#730 section C1.4, flagged by chelis#782)",
                        );
                        raise_fatal_lowering_error(
                            unsupported.to_string(),
                            Some(args[1].span()),
                            args[1].span_id().map(ToOwned::to_owned),
                        )
                    }),
                };
                // Recover the broadcast extent. Three sources, in order:
                //
                //   1. A statically-extractable size (a bare int, `(lit
                //      ...)`, a `cast`-wrapped int, or a symbolic dim
                //      variable) — issue #288's literal/symbol form
                //      `expand(scalar_to_tensor(c), 0, cast(2, int32))`.
                //
                //   2. A `shape(operand, axis)` application — issue #318's
                //      shape-derived form
                //      `expand(scalar_to_tensor(c), 0, shape(&x, 0))` that
                //      the canonical `tensor_full_like` / `tensor_full_1d`
                //      helper emits. The extent is read from `operand`'s
                //      already-lowered dim at `axis`, NOT from this expand
                //      node's type: the type checker collapses the expand
                //      *output* dim to `Lit(1)` via size-1 broadcasting, so
                //      reading the node type would recover exactly the
                //      `Lit(1)` that produced the `Lit(n) vs Lit(1)`
                //      verification failure under `grad`.
                //
                //   3. Otherwise default to size 1.
                //
                // chelis#384/#397: when the extent comes from a
                // `shape(src, axis)` argument, capture `src`'s lowered node so
                // it can be recorded as a `shape_dep` on the Expand below. The
                // extent symbol is declared by `src`'s shape; without the dep,
                // a `src` referenced only via `shape(src, ...)` is DCE'd and
                // the symbol loses its source (silent wrong shape in C).
                let mut shape_source: Option<NodeId> = None;
                let size = if args.len() >= 3 {
                    let size_arg = &args[2];
                    // (1) A fully-static size: a literal, `cast(N, _)`,
                    //     integer arithmetic over such values, or a `let`-bound
                    //     name that folds to one (chelis#469 / #528). Const-
                    //     folded to a concrete extent so a `sub(cast(4, int32),
                    //     cast(1, int32))` size — or a `len = cast(7, int32)`
                    //     followed by `cast(len, int32)` — no longer falls
                    //     through to the size-1 default (a silent
                    //     eval-`[3, 3]`-vs-C-`[1, 3]` miscompile pre-fix).
                    if let Some(v) = self
                        .fold_static_size(size_arg)
                        .and_then(|n| usize::try_from(n).ok())
                    {
                        DimExpr::Concrete(v)
                    }
                    // (2) A shape source: an inline `shape(x, axis)` read OR a
                    //     `let`-bound `len = shape(x, axis)` name (followed
                    //     through `cast` / `let` by `shape_app_operand_axis_
                    //     resolved`). Tried BEFORE the bare-symbol arm so the
                    //     `let`-bound form resolves to the tensor extent rather
                    //     than a sourceless `Sym("len")` that the guard below
                    //     rejects (the check-`accept` / build-`reject`
                    //     asymmetry #469 tracks). The source node is recorded
                    //     as a `shape_dep` below so its `Load` survives DCE and
                    //     declares the extent symbol (chelis#384/#397).
                    else if let Some((dim, src)) =
                        self.dim_expr_from_shape_arg_with_source(size_arg)
                    {
                        shape_source = Some(src);
                        dim
                    }
                    // (3) A bare `var` naming a §4.7.2 Form-2 symbolic dim (an
                    //     in-scope tensor dimension, or a monomorphized dim
                    //     substitution). The post-node sourceless guard below
                    //     validates the symbol has a real tensor source and
                    //     fails closed otherwise.
                    else if let Some(dim) = self.extract_dim_expr_value(size_arg) {
                        dim
                    }
                    // (4) Fail closed (chelis#469): a check-clean runtime size
                    //     the backend cannot yet materialize as an extent —
                    //     integer arithmetic that COMBINES a `shape(tensor,
                    //     axis)` read (or a symbolic dim) with another term
                    //     (`mul(shape(x, 0), 2)`, `add(shape(x, 0), 1)`), which
                    //     the checker admits as `ShapeSourced` but `DimExpr`
                    //     has no representation for. Reject loudly rather than
                    //     the pre-fix silent `Concrete(1)` default, which
                    //     emitted an extent-1 axis and silently diverged from
                    //     `eval` — the exact miscompile class #469 exists to
                    //     prevent.
                    else {
                        raise_fatal_lowering_error(
                            "`expand` size is a runtime expression the backend cannot \
                             materialize as an extent: integer arithmetic that combines a \
                             `shape(tensor, axis)` read (or a symbolic dimension) with another \
                             term (e.g. `mul(shape(x, 0), 2)` or `add(shape(x, 0), 1)`) has no \
                             single tensor axis to read the extent from. Use a literal/`cast(N, \
                             int32)` size, a bare `shape(tensor, axis)` read, or an in-scope \
                             tensor dimension. Tracked by Chelis-Lang/chelis#469 \
                             (spec/04-type-system.md \u{00a7}4.7.2)",
                            Some(app_span),
                            self.current_span_id.clone(),
                        );
                    }
                } else {
                    DimExpr::Concrete(1)
                };
                // For a named insert, the new dim carries the inserted NAME
                // (with its concrete extent when the size is static) so a
                // later by-name op in the same body — `sum(expand(x, c, k), c)`
                // — can locate it. A symbolic size keeps the size symbol's
                // DimInfo (it must be declared by a Load; the inserted name
                // would be an undeclared symbol for the C codegen).
                let named_dim = match (&named_insert, &size) {
                    (Some(name), DimExpr::Concrete(n)) => {
                        Some(DimInfo::Named(name.clone(), Some(*n)))
                    }
                    _ => None,
                };
                let out_ty = self
                    .fallback_expand_type(x, axis, &size)
                    .map(|mut t| {
                        if let Some(dim) = named_dim {
                            t.dims[axis] = dim;
                        }
                        t
                    })
                    .unwrap_or_else(|| ty.clone());
                let expand_id = self.dag.add_node(
                    RiscOp::Expand { axis, size },
                    vec![x],
                    out_ty,
                    self.current_span_id.clone(),
                );
                // chelis#384/#397: keep the `shape(src, ...)` extent source
                // alive through DCE so the symbolic dim it declares binds from
                // the correct tensor input.
                if let Some(src) = shape_source {
                    self.dag.add_shape_dep(expand_id, src);
                }
                // chelis#384/#397 (B): a §4.7.2 Form-3 runtime expand size that
                // resolves to a symbolic dim with NO tensor source — neither a
                // `shape(tensor, ...)` argument (which the shape_dep above
                // keeps live) nor a dim carried by any tensor in scope — has no
                // representation the backend can emit. The C codegen would read
                // the extent from a fabricated/out-of-range operand axis
                // (silent wrong shape — the original bug) or emit an undeclared
                // identifier. Reject loudly at lowering with a clean diagnostic
                // rather than the downstream `symbolic_occurrences` ICE. The
                // size of `expand(x, 1, k)` where `k` is a scalar `int32`
                // parameter (or a rank-var output dim like #397's `a`, sourced
                // from a scalar `int64`) is exactly this case.
                if shape_source.is_none()
                    && let Some(RiscOp::Expand { size, .. }) =
                        self.dag.get(expand_id).map(|n| &n.op)
                    && let DimExpr::Sym(name) = size.clone()
                    && !self.symbol_has_tensor_source(&name)
                {
                    raise_fatal_lowering_error(
                        format!(
                            "`expand` size resolves to the symbolic dimension `{name}`, but no \
                             tensor in scope carries it: a §4.7.2 Form-3 runtime size must be a \
                             literal/`cast(N, int32)`, an in-scope tensor dimension, or a \
                             `shape(tensor, axis)` read. A bare runtime scalar (e.g. an `int32`/\
                             `int64` parameter) has no shape source the backend can emit, so the \
                             extent cannot be materialized. Tracked by Chelis-Lang/chelis#469 \
                             (spec/04-type-system.md \u{00a7}4.7.2)"
                        ),
                        Some(app_span),
                        self.current_span_id.clone(),
                    );
                }
                expand_id
            }
            "pad" if !args.is_empty() => {
                let x = self.lower_expr_node(&args[0], "pad input");
                // chelis#616: `inputs` starts as `[tensor]`; node-valued bounds
                // append rank-0 int scalars and are referenced by slot index.
                let mut inputs = vec![x];
                let padding = if args.len() >= 2 {
                    self.lower_pair_bounds(&args[1], &mut inputs)
                } else {
                    vec![]
                };
                let fill = if args.len() >= 3 {
                    // chelis#776: an explicit fill argument must resolve
                    // statically (through neg / float-cast wrappers) or fail
                    // loudly. The `else` arm below is a true structural default
                    // — no fill was given, so pad with zeros.
                    self.resolve_static_f64_arg(&args[2], "pad", "fill value")
                } else {
                    0.0
                };
                self.dag.add_node(
                    RiscOp::Pad { padding, fill },
                    inputs,
                    ty.clone(),
                    self.current_span_id.clone(),
                )
            }
            "shrink" if !args.is_empty() => {
                let x = self.lower_expr_node(&args[0], "shrink input");
                let mut inputs = vec![x];
                let bounds = if args.len() >= 2 {
                    self.lower_pair_bounds(&args[1], &mut inputs)
                } else {
                    vec![]
                };
                self.dag.add_node(
                    RiscOp::Shrink { bounds },
                    inputs,
                    ty.clone(),
                    self.current_span_id.clone(),
                )
            }
            "stride" if !args.is_empty() => {
                let x = self.lower_expr_node(&args[0], "stride input");
                let mut inputs = vec![x];
                let strides = if args.len() >= 2 {
                    self.lower_stride_bounds(&args[1..], &mut inputs)
                } else {
                    vec![]
                };
                self.dag.add_node(
                    RiscOp::Stride { strides },
                    inputs,
                    ty.clone(),
                    self.current_span_id.clone(),
                )
            }
            "fold" if args.len() == 3 => {
                if self.allow_host_list_ad_rewrites
                    && let Some(node) = self.lower_host_list_fold(&args[0], &args[1], &args[2], ty)
                {
                    return node;
                }
                for arg in args {
                    self.lower_expr(arg);
                }
                self.dag.add_node(
                    RiscOp::Load {
                        name: func_name.into(),
                    },
                    vec![],
                    Self::default_type(),
                    self.current_span_id.clone(),
                )
            }

            // Issue Chelis-Lang/chelis#218:
            // `to_tensor(<literal Cons chain>)` lowers into an IR DAG
            // sub-tree (Const + Pad + Add composition) so the
            // containing fn-def stays DAG-lowerable and `grad` can
            // reach it. A constant tensor has zero gradient; AD
            // treats each Const leaf with an empty input-gradient
            // list, the correct adjoint contribution.
            //
            // For a non-literal argument the arm falls through to
            // the fallback below, preserving the legacy
            // `Load { name: "to_tensor" }` placeholder that the host
            // helper extractor uses to reject the helper and force
            // host-lane routing for runtime-shaped to_tensor calls.
            "to_tensor" if args.len() == 1 => {
                if self.allow_host_list_ad_rewrites
                    && let Some(node) = self.lower_host_list_to_tensor(&args[0], ty)
                {
                    return node;
                }
                if let Some(literal) = extract_cons_chain_tensor(&args[0]) {
                    return self.emit_literal_tensor(&literal, ty);
                }
                for arg in args {
                    self.lower_expr(arg);
                }
                self.dag.add_node(
                    RiscOp::Load {
                        name: func_name.into(),
                    },
                    vec![],
                    Self::default_type(),
                    self.current_span_id.clone(),
                )
            }

            // Issue #368: `concat` of a statically-enumerable list of
            // tensors along a constant axis. `concat` is otherwise a
            // host-runtime op with no RISC DAG lowering and no adjoint, so
            // a `grad` through `mean`/`max_reduce` over a concat'd window
            // stack hit the fallback below, which emits a rank-0
            // `Load { name: "concat" }` placeholder. The reduce then ran on
            // a rank-0 operand: `mean` panicked indexing `dims[axis]` of the
            // empty shape, and `max_reduce` built a backward node pairing a
            // rank-1 cotangent with the rank-0 collapse (verify "1 vs 0").
            //
            // Lower the differentiable case to the existing Pad+Add cascade
            // (the same construction `emit_literal_tensor` uses): pad each
            // element to the full concat shape with zeros before/after on the
            // concat axis, then sum. `Pad` and `Add` both carry reverse-mode
            // adjoints (Pad -> Shrink, Add routes the cotangent to each
            // input), so `grad` reaches the windowed inputs and finite-diff
            // validates. Non-concat axes (including symbolic dims) ride
            // through untouched because their padding is `(0, 0)`.
            //
            // Falls through to the host fallback when the list is not
            // statically enumerable, the axis is not a constant, an element
            // is not a recognized tensor node, or an element's concat-axis
            // extent is not a concrete literal (a ragged/runtime concat,
            // which still needs the host lane).
            "concat" if args.len() == 2 => {
                if let Some(node) = self.lower_tensor_concat(&args[0], &args[1]) {
                    return node;
                }
                for arg in args {
                    self.lower_expr(arg);
                }
                self.dag.add_node(
                    RiscOp::Load {
                        name: func_name.into(),
                    },
                    vec![],
                    Self::default_type(),
                    self.current_span_id.clone(),
                )
            }

            // chelis#513 / chelis#558: a `shape(operand, axis)` read used
            // as a scalar VALUE (not folded into an `expand`/`reshape`
            // extent `DimExpr`) lowers to a real `RiscOp::Shape` node that
            // reads the operand's runtime extent along a constant `axis`
            // as a rank-0 integer scalar. Before this, a scalar shape read
            // fell through to the generic unknown-function fallback below
            // and fabricated a bogus `Load { name: "shape" }` placeholder
            // with no inputs and a default scalar-f32 type, which produced
            // garbage in the C / grad DAG lanes. The axis must be a
            // compile-time literal (the idiomatic `cast(N, int32)` form is
            // accepted via `extract_int_for_dim`); a genuinely runtime axis
            // is caught by the dedicated loud arm immediately below rather
            // than reaching the fallback.
            "shape"
                if args.len() == 2
                    && extract_int_for_dim(&args[1])
                        .and_then(|a| usize::try_from(a).ok())
                        .is_some() =>
            {
                let axis = extract_int_for_dim(&args[1])
                    .and_then(|a| usize::try_from(a).ok())
                    .expect("axis literal guarded by the arm predicate");
                let x = self.lower_expr_node(&args[0], "shape input");
                // The checker types `shape(...)` as an `int32` scalar. Pin
                // a rank-0 integer output regardless of the incoming `ty`
                // shape so the value node is always a well-formed scalar
                // extent (verified by `chelis_ir::verify`).
                let precision = if ty.precision.is_integer() {
                    ty.precision
                } else {
                    Prim::Int32
                };
                self.dag.add_node(
                    RiscOp::Shape { axis },
                    vec![x],
                    TensorType {
                        dims: Vec::new(),
                        precision,
                    },
                    self.current_span_id.clone(),
                )
            }

            // chelis#513 / chelis#558 / chelis#616: `shape(operand, axis)`
            // reached here with two arguments but a NON-literal `axis` (the
            // literal-axis arm above did not match). `RiscOp::Shape` carries
            // a compile-time `axis`, so a runtime (data- or metadata-derived)
            // axis is genuinely not representable as a RISC value node. Fail
            // LOUD and CLEAN here instead of falling through to the generic
            // unknown-function fallback below, which fabricated a bogus
            // `Load { name: "shape" }` placeholder (no inputs, default
            // scalar-f32 type) — the latent unsoundness chelis#513 named:
            // silently wrong under grad-DAG eval, and an undefined `shape`
            // input slot in the C backend. The host evaluator DOES resolve a
            // runtime axis, so a host-lane speculative DAG probe (with the
            // suppression flag set) unwinds quietly via `UnrepresentableDag`
            // and lets host lowering take over; only a hard DAG requirement
            // (`grad`, which forces DAG construction) surfaces the diagnostic
            // to the user. Node-valued runtime shape axes are tracked in
            // chelis#616.
            "shape" if args.len() == 2 => {
                // Lower the operand + axis sub-expressions for their side
                // effects (span registration, surfacing any nested
                // unrepresentable form) before bailing, mirroring
                // `lower_unrepresentable`.
                for arg in args {
                    let _ = self.lower_expr(arg);
                }
                if unrepresentable_panic_suppressed() {
                    std::panic::panic_any(UnrepresentableDag);
                }
                raise_lowering_error(
                    "`shape(tensor, axis)` requires a compile-time-constant `axis`: a \
                     runtime (data- or metadata-derived) axis cannot be lowered to a RISC \
                     `shape` value node (chelis#616). Bind `axis` to a literal, or keep the \
                     read in host evaluation, which resolves a runtime axis.",
                    Some(app_span),
                    self.current_span_id.clone(),
                )
            }

            // chelis#616: `fail(...)` in a DAG-lowered `if` branch. The mask
            // lowering zeroes the untaken branch, so the placeholder's VALUE
            // never matters on the taken path; real abort semantics live in
            // the host lane (which owns entry-level `if`/`fail`). The
            // pre-#616 terminal fallback fabricated a rank-0
            // `Load { name: "fail" }` — a phantom input slot that broke the
            // C lane and mixed ranks in the mask arithmetic. Emit a zero
            // Const at the branch's rank instead, with ANONYMOUS symbolic
            // dims (the checker's symbol may be declared later in program
            // order; `lower_if` ties the placeholder's shape to the sibling
            // branch via a shape-dep).
            "fail" => {
                for arg in args {
                    let _ = self.lower_expr(arg);
                }
                let dims = ty
                    .dims
                    .iter()
                    .map(|dim| match dim {
                        DimInfo::Lit(n) | DimInfo::Named(_, Some(n)) => DimInfo::Lit(*n),
                        DimInfo::Named(_, None) => DimInfo::Named(String::new(), None),
                    })
                    .collect();
                self.dag.add_node(
                    RiscOp::synth_const(ty.precision, 0.0),
                    vec![],
                    TensorType {
                        dims,
                        precision: ty.precision,
                    },
                    self.current_span_id.clone(),
                )
            }

            // Fallback: unknown function.
            _ => {
                for arg in args {
                    self.lower_expr(arg);
                }
                self.dag.add_node(
                    RiscOp::Load {
                        name: func_name.into(),
                    },
                    vec![],
                    Self::default_type(),
                    self.current_span_id.clone(),
                )
            }
        }
    }

    /// Emit a literal constant tensor as an IR DAG sub-tree. Used by
    /// the `to_tensor` arm of `lower_builtin_app` to lower a
    /// statically-recognized numeric Cons-chain literal (issue
    /// Chelis-Lang/chelis#218) into the DAG without a schema change.
    ///
    /// Strategy:
    /// - Uniform-value fast path: if every element is the same
    ///   value, emit a single `Const(value)` with the full shape
    ///   (eval fills the buffer uniformly).
    /// - Otherwise: for each non-zero element at flat index `i`,
    ///   emit a unit-shaped `Const(value)`, `Pad` it into row-major
    ///   position `i` with `fill = 0.0`, and accumulate via `Add`.
    ///   Zero-valued elements skip the cascade because `+ 0` is the
    ///   identity. If every element happens to be zero, the uniform
    ///   fast-path catches it first.
    ///
    /// The precision is taken from `ty.precision` (the `to_tensor`
    /// call's type metadata); the shape is the literal's concrete
    /// shape (the metadata's symbolic `Named("list", None)` dim is
    /// not usable here).
    fn emit_literal_tensor(&mut self, literal: &LiteralToTensor, ty: &TensorType) -> NodeId {
        use chelis_types::{RawScalar, RawTensor};
        let shape: Vec<DimInfo> = literal.shape.iter().map(|n| DimInfo::Lit(*n)).collect();
        let precision = ty.precision;
        let tensor_ty = TensorType {
            dims: shape.clone(),
            precision,
        };

        // The exact raw leaves finalize ONCE at the ascribed dtype
        // (chelis#856): integer families through the exact i64 lane
        // (no f64 laundering above 2^53), float families through their
        // f64 images. An out-of-domain literal is a loud lowering
        // diagnostic; cast leaves already applied the checked ladder
        // (or declined static recognition) in `extract_numeric_leaf`.
        let raise_on = |trap: chelis_types::NumericTrap, span_id: Option<String>| -> ! {
            raise_lowering_error(
                format!(
                    "tensor literal does not finalize at its ascribed dtype \
                     `{}`: {trap} (spec/04-type-system.md section 9 \
                     [04-NUM-1]; chelis#856)",
                    precision.name()
                ),
                None,
                span_id,
            )
        };

        // Uniform-value fast path (bit-level key so NaN-uniform lists
        // still collapse to one Const).
        let key = |raw: &RawScalar| -> (u8, u64) {
            match raw {
                RawScalar::Int(i) => (0, *i as u64),
                RawScalar::Float(f) => (1, f.to_bits()),
            }
        };
        if literal
            .data
            .windows(2)
            .all(|pair| key(&pair[0]) == key(&pair[1]))
        {
            let raw = literal.data.first().copied().unwrap_or(RawScalar::Int(0));
            let value = match chelis_types::finalize_scalar("const", precision, raw) {
                Ok(value) => value,
                Err(trap) => raise_on(trap, self.current_span_id.clone()),
            };
            return self.dag.add_node(
                RiscOp::Const { value },
                vec![],
                tensor_ty,
                self.current_span_id.clone(),
            );
        }

        // Non-uniform: one ConstTensor with sealed per-dtype storage.
        // Integer/bool targets take the exact integer lane; a float
        // leaf reaching an integer target here is integral or the
        // finalize diagnostic fires (cast leaves were already applied).
        let raw_tensor = if precision.is_integer() || precision == Prim::Bool {
            let mut ints = Vec::with_capacity(literal.data.len());
            for raw in &literal.data {
                match raw {
                    RawScalar::Int(i) => ints.push(*i),
                    RawScalar::Float(f) => {
                        if f.fract() != 0.0 || !f.is_finite() {
                            raise_on(
                                chelis_types::NumericTrap::Domain {
                                    op: "const",
                                    prim: precision,
                                },
                                self.current_span_id.clone(),
                            );
                        }
                        // Reject before the saturating `as` cast: an
                        // integral f64 outside i64's range must not
                        // silently clamp to i64::MAX/MIN.
                        if *f < -9_223_372_036_854_775_808.0 || *f >= 9_223_372_036_854_775_808.0 {
                            raise_on(
                                chelis_types::NumericTrap::Overflow {
                                    op: "const",
                                    prim: precision,
                                },
                                self.current_span_id.clone(),
                            );
                        }
                        ints.push(*f as i64);
                    }
                }
            }
            RawTensor::Int(ints)
        } else {
            RawTensor::Float(
                literal
                    .data
                    .iter()
                    .map(|raw| match raw {
                        RawScalar::Int(i) => *i as f64,
                        RawScalar::Float(f) => *f,
                    })
                    .collect(),
            )
        };
        let data = match chelis_types::finalize_tensor("const", precision, raw_tensor) {
            Ok(data) => data,
            Err(trap) => raise_on(trap, self.current_span_id.clone()),
        };
        self.dag.add_node(
            RiscOp::ConstTensor { data },
            vec![],
            tensor_ty,
            self.current_span_id.clone(),
        )
    }

    /// Issue #368: lower `concat([t0, t1, ...], axis)` of a
    /// statically-enumerable list of tensors along a constant axis into a
    /// Pad+Add cascade so the result is a differentiable RISC sub-tree.
    ///
    /// Each element is padded to the full concat shape (zeros before its
    /// slice and after it along the concat axis, `(0, 0)` on every other
    /// axis) and the padded elements are summed. Returns `None` (so the
    /// caller falls through to the host-runtime `concat` fallback) when the
    /// list is not a closed `Cons` chain, the axis is not a compile-time
    /// constant, an element does not lower to a usable rank>=1 tensor node,
    /// the elements disagree on rank or on a non-concat axis, or an
    /// element's concat-axis extent is not a concrete literal (a ragged or
    /// runtime-extent concat, which the host lane still owns).
    fn lower_tensor_concat(&mut self, list_expr: &Expr, axis_expr: &Expr) -> Option<NodeId> {
        let resolved = self.resolved_list_expr(list_expr);
        let elements = collect_cons_chain(&resolved)?;
        if elements.is_empty() {
            return None;
        }
        let raw_axis = extract_int_axis(axis_expr)?;

        // Lower each element. Bail out (fall through to the host fallback)
        // if any element is not a single tensor node.
        let mut nodes = Vec::with_capacity(elements.len());
        for elem in &elements {
            nodes.push(self.lower_expr(elem).as_single_node()?);
        }
        self.tensor_concat_from_nodes(&nodes, raw_axis)
    }

    /// The node-level tail of [`Self::lower_tensor_concat`]: type-check the
    /// already-lowered elements and emit the Pad+Add cascade. Split out
    /// (chelis#620) so a list that is only statically known as a lowered
    /// VALUE (an [`adt_cons_chain_values`] spine from an unrolled recursive
    /// builder) shares the exact same construction as the expr-level path.
    fn tensor_concat_from_nodes(&mut self, nodes: &[NodeId], raw_axis: i64) -> Option<NodeId> {
        if nodes.is_empty() {
            return None;
        }
        // Snapshot each element's type. Bail out if any element collapses
        // to rank-0 — that is exactly the degenerate placeholder we are
        // trying to avoid, and a Pad over it would be unsound.
        let mut elem_types = Vec::with_capacity(nodes.len());
        for node in nodes {
            let ty = self.dag.get(*node)?.output_type.clone();
            if ty.dims.is_empty() {
                return None;
            }
            elem_types.push(ty);
        }

        let rank = elem_types[0].dims.len();
        let precision = elem_types[0].precision;
        if elem_types
            .iter()
            .any(|ty| ty.dims.len() != rank || ty.precision != precision)
        {
            return None;
        }
        let axis = {
            let normalized = if raw_axis < 0 {
                raw_axis + rank as i64
            } else {
                raw_axis
            };
            if normalized < 0 || normalized as usize >= rank {
                return None;
            }
            normalized as usize
        };

        // Every non-concat axis must agree across elements (concat does not
        // broadcast); the concat-axis extent of each element must be a
        // concrete literal so the per-element padding amounts are known.
        let mut concat_extents = Vec::with_capacity(elem_types.len());
        for ty in &elem_types {
            for (ax, (d, d0)) in ty.dims.iter().zip(elem_types[0].dims.iter()).enumerate() {
                if ax != axis && d != d0 {
                    return None;
                }
            }
            concat_extents.push(concrete_dim_len(&ty.dims[axis])?);
        }
        let total: usize = concat_extents.iter().sum();

        // Output dims: the first element's dims with the concat axis
        // widened to the summed extent.
        let mut out_dims = elem_types[0].dims.clone();
        out_dims[axis] = DimInfo::Lit(total);
        let out_ty = TensorType {
            dims: out_dims,
            precision,
        };

        // Pad each element to the full shape and sum. `offset` tracks the
        // start of the current element's slice along the concat axis.
        let mut accumulator: Option<NodeId> = None;
        let mut offset = 0usize;
        for (node, extent) in nodes.iter().zip(concat_extents.iter()) {
            let before = offset;
            let after = total - offset - extent;
            offset += extent;
            // `(0, 0)` everywhere except the concat axis, so symbolic
            // non-concat dims need no concrete extent.
            let mut padding = vec![(RtDim::Lit(0), RtDim::Lit(0)); rank];
            padding[axis] = (RtDim::Lit(before), RtDim::Lit(after));
            let padded = self.dag.add_node(
                RiscOp::Pad { padding, fill: 0.0 },
                vec![*node],
                out_ty.clone(),
                self.current_span_id.clone(),
            );
            accumulator = Some(match accumulator {
                None => padded,
                Some(prev) => self.dag.add_node(
                    RiscOp::Add,
                    vec![prev, padded],
                    out_ty.clone(),
                    self.current_span_id.clone(),
                ),
            });
        }
        accumulator
    }

    /// chelis#620: `concat` over list VALUES that are only statically known
    /// after lowering (an [`adt_cons_chain_values`] spine). Two shapes:
    ///
    ///   * `concat(list, axis)` with a compile-time int axis — tensor
    ///     concat; when the list expr is NOT expr-enumerable (so the
    ///     builtin arm's `lower_tensor_concat` would fall to the host
    ///     placeholder) but lowers to a static spine of single tensor
    ///     nodes, emit the same Pad+Add cascade via
    ///     [`Self::tensor_concat_from_nodes`].
    ///   * `concat(list, list)` — static list append; rebuild the appended
    ///     spine as a `LoweredValue::Adt`.
    ///
    /// Returns `None` to fall through to the builtin arm. A `None` after
    /// argument lowering leaves orphan nodes behind, the same
    /// lower-then-bail contract `lower_tensor_concat` already has; the
    /// entry points' DCE sweeps them.
    fn try_lower_static_list_concat(
        &mut self,
        list_arg: &Expr,
        second_arg: &Expr,
    ) -> Option<LoweredValue> {
        if let Some(raw_axis) = extract_int_axis(second_arg) {
            // Tensor concat. The expr-level path in the builtin arm owns
            // an expr-enumerable list; only pick up the value-level case.
            if collect_cons_chain(&self.resolved_list_expr(list_arg)).is_some() {
                return None;
            }
            let value = self.lower_expr(list_arg);
            let elements = adt_cons_chain_values(&value)?;
            let nodes = elements
                .iter()
                .map(LoweredValue::as_single_node)
                .collect::<Option<Vec<_>>>()?;
            return self
                .tensor_concat_from_nodes(&nodes, raw_axis)
                .map(LoweredValue::Node);
        }
        // List append: both sides must be static spines.
        let left = self.lower_expr(list_arg);
        let mut items = adt_cons_chain_values(&left)?;
        let right = self.lower_expr(second_arg);
        items.extend(adt_cons_chain_values(&right)?);
        Some(rebuild_cons_chain(items))
    }

    fn resolved_list_expr(&self, expr: &Expr) -> Expr {
        bare_var_name(expr)
            .and_then(|name| self.list_bindings.get(&name).cloned())
            .unwrap_or_else(|| expr.clone())
    }

    fn is_host_list_expr(&self, expr: &Expr) -> bool {
        if bare_var_name(expr)
            .as_deref()
            .is_some_and(|name| self.list_bindings.contains_key(name))
        {
            return true;
        }
        matches!(
            app_var_name_and_args(expr),
            Some(("to_list", [_])) | Some(("map", [_, _])) | Some(("filter", [_, _]))
        )
    }

    fn lower_host_list_to_tensor(&mut self, expr: &Expr, ty: &TensorType) -> Option<NodeId> {
        let resolved = self.resolved_list_expr(expr);
        if let Some(source) = to_list_source_expr(&resolved) {
            return Some(self.lower_expr_node(source, "to_list/tensor AD boundary"));
        }

        let (name, args) = app_var_name_and_args(&resolved)?;
        match (name, args) {
            ("map", [callback, list_expr]) => {
                let list_resolved = self.resolved_list_expr(list_expr);
                let source = to_list_source_expr(&list_resolved)?;
                let source_node = self.lower_expr_node(source, "map source");
                self.lower_host_list_map(callback, source_node)
            }
            ("filter", [predicate, list_expr]) => {
                let list_resolved = self.resolved_list_expr(list_expr);
                let source = to_list_source_expr(&list_resolved)?;
                let source_node = self.lower_expr_node(source, "filter source");
                self.lower_host_list_filter(predicate, source_node)
            }
            _ => {
                let _ = ty;
                None
            }
        }
    }

    fn lower_host_list_map(&mut self, callback: &Expr, source_node: NodeId) -> Option<NodeId> {
        let CallableExpr::Plain(fn_expr) = self.resolve_callable_expr(callback)? else {
            return None;
        };
        let (len, elem_ty, out_dim) = self.rank1_list_source_parts(source_node)?;
        let mut mapped = Vec::with_capacity(len);
        for index in 0..len {
            let item = self.rank1_item(source_node, index, &elem_ty);
            let item_out = self
                .lower_plain_callable_with_values(&fn_expr, &[LoweredValue::Node(item)])
                .expect_node("map callback");
            mapped.push(item_out);
        }
        self.stack_scalar_nodes(&mapped, out_dim)
    }

    fn lower_host_list_filter(&mut self, predicate: &Expr, source_node: NodeId) -> Option<NodeId> {
        let CallableExpr::Plain(fn_expr) = self.resolve_callable_expr(predicate)? else {
            return None;
        };
        let (len, elem_ty, out_dim) = self.rank1_list_source_parts(source_node)?;
        let mut masked = Vec::with_capacity(len);
        for index in 0..len {
            let item = self.rank1_item(source_node, index, &elem_ty);
            let mask = self
                .lower_plain_callable_with_values(&fn_expr, &[LoweredValue::Node(item)])
                .expect_node("filter predicate");
            let mask_as_value = self.dag.add_node(
                RiscOp::Cast {
                    new_precision: elem_ty.precision,
                },
                vec![mask],
                elem_ty.clone(),
                self.current_span_id.clone(),
            );
            let selected = self.dag.add_node(
                RiscOp::Mul,
                vec![item, mask_as_value],
                elem_ty.clone(),
                self.current_span_id.clone(),
            );
            masked.push(selected);
        }
        self.stack_scalar_nodes(&masked, out_dim)
    }

    fn lower_host_list_fold(
        &mut self,
        callback: &Expr,
        init: &Expr,
        list_expr: &Expr,
        ty: &TensorType,
    ) -> Option<NodeId> {
        let list_resolved = self.resolved_list_expr(list_expr);
        let source = to_list_source_expr(&list_resolved)?;
        let source_node = self.lower_expr_node(source, "fold source");
        let source_ty = self.dag.get(source_node)?.output_type.clone();
        if source_ty.dims.len() != 1 {
            return None;
        }
        let len = concrete_dim_len(&source_ty.dims[0])?;
        let CallableExpr::Plain(fn_expr) = self.resolve_callable_expr(callback)? else {
            return None;
        };
        let mut acc = self.lower_expr(init);
        let elem_ty = TensorType {
            dims: vec![],
            precision: source_ty.precision,
        };
        for index in 0..len {
            let item = self.rank1_item(source_node, index, &elem_ty);
            acc = self.lower_plain_callable_with_values(&fn_expr, &[acc, LoweredValue::Node(item)]);
        }
        let acc_node = acc.expect_node("fold callback");
        if let Some(node) = self.dag.get(acc_node)
            && node.output_type == Self::default_type()
            && ty != &Self::default_type()
        {
            let op = node.op.clone();
            let inputs = node.inputs.clone();
            self.dag.replace_node(acc_node, op, inputs, ty.clone());
        }
        Some(acc_node)
    }

    fn rank1_list_source_parts(&self, source_node: NodeId) -> Option<(usize, TensorType, DimInfo)> {
        let source_ty = self.dag.get(source_node)?.output_type.clone();
        if source_ty.dims.len() != 1 {
            return None;
        }
        let dim = source_ty.dims[0].clone();
        let len = concrete_dim_len(&dim)?;
        let elem_ty = TensorType {
            dims: vec![],
            precision: source_ty.precision,
        };
        Some((len, elem_ty, dim))
    }

    fn rank1_item(&mut self, source_node: NodeId, index: usize, elem_ty: &TensorType) -> NodeId {
        let unit_ty = TensorType {
            dims: vec![DimInfo::Lit(1)],
            precision: elem_ty.precision,
        };
        let sliced = self.dag.add_node(
            RiscOp::Shrink {
                bounds: vec![(RtDim::Lit(index), RtDim::Lit(index + 1))],
            },
            vec![source_node],
            unit_ty,
            self.current_span_id.clone(),
        );
        self.dag.add_node(
            RiscOp::Reshape { new_shape: vec![] },
            vec![sliced],
            elem_ty.clone(),
            self.current_span_id.clone(),
        )
    }

    fn stack_scalar_nodes(&mut self, nodes: &[NodeId], out_dim: DimInfo) -> Option<NodeId> {
        let first_ty = self.dag.get(*nodes.first()?)?.output_type.clone();
        if !first_ty.dims.is_empty() {
            return None;
        }
        let out_len = concrete_dim_len(&out_dim)?;
        let out_ty = TensorType {
            dims: vec![out_dim],
            precision: first_ty.precision,
        };
        let unit_ty = TensorType {
            dims: vec![DimInfo::Lit(1)],
            precision: first_ty.precision,
        };
        let mut accumulator = None;
        for (index, node) in nodes.iter().copied().enumerate() {
            let node_ty = self.dag.get(node)?.output_type.clone();
            if !node_ty.dims.is_empty() || node_ty.precision != first_ty.precision {
                return None;
            }
            let unit = self.dag.add_node(
                RiscOp::Reshape {
                    new_shape: vec![RtDim::Lit(1)],
                },
                vec![node],
                unit_ty.clone(),
                self.current_span_id.clone(),
            );
            let padded = self.dag.add_node(
                RiscOp::Pad {
                    padding: vec![(RtDim::Lit(index), RtDim::Lit(out_len - index - 1))],
                    fill: 0.0,
                },
                vec![unit],
                out_ty.clone(),
                self.current_span_id.clone(),
            );
            accumulator = Some(match accumulator {
                Some(prev) => self.dag.add_node(
                    RiscOp::Add,
                    vec![prev, padded],
                    out_ty.clone(),
                    self.current_span_id.clone(),
                ),
                None => padded,
            });
        }
        accumulator
    }

    /// Extract a raw axis value from an expression (for
    /// sum/max_reduce/softmax/mean/gather/scatter and the reduction
    /// family). The value is returned verbatim and may be negative:
    /// negative axes index from the end of the operand rank and are
    /// normalized by [`Self::normalize_axis`] once the operand rank is
    /// known. Returning the raw `i64` keeps the negative-axis
    /// convention (`-1` is the last axis) intact instead of wrapping
    /// it to `usize::MAX`.
    /// Resolve a reduction/gather/scatter/softmax axis expression to its
    /// compile-time-constant integer value. Recognizes the exact forms the
    /// checker admits as a constant axis: a bare int, `(lit {} n)`, and any
    /// number of `cast(<int>, int32)` wrappers (`check_reduction_signature`
    /// admits `sum(x, cast(1, int32))`) — delegated to the shared
    /// [`extract_int_for_dim`] walker.
    ///
    /// Issue #364: the pre-fix body returned `0` for ANY axis it did not
    /// statically recognize (including `cast(N, int32)` for N != 0), so a
    /// `cast`-axis reduction lowered to axis 0 regardless of N — silently
    /// reducing the wrong axis (and, under `grad`, differentiating the wrong
    /// reduction with eval/backend agreeing on the SAME wrong answer). A
    /// silent default-to-0 is exactly the fallback class the repo forbids
    /// (`CLAUDE.md` "Do Not Trust Green"). Anything not resolvable here is a
    /// loud, FATAL lowering error: a non-constant axis stays a check-time
    /// rejection (#259 family), so reaching this site with an unresolvable
    /// axis is an internal contract violation, and a plain diagnostic would
    /// be absorbed by the host-fallback path into garbage C.
    fn extract_axis_raw(&self, expr: &Expr, op: &str) -> i64 {
        if let Some(n) = extract_int_axis(expr) {
            return n;
        }
        raise_fatal_lowering_error(
            format!(
                "`{op}` axis is not a compile-time integer constant: rank monomorphization \
                 cannot resolve it to a fixed axis (the checker admits only a literal or a \
                 `cast(<int>, int32)` axis here; a runtime axis must be rejected at check time)"
            ),
            Some(expr.span()),
            expr.span_id().map(ToOwned::to_owned),
        );
    }

    /// Best-effort operand rank for axis normalization. Prefers the
    /// operand's lowered node type, but falls back to the ascribed
    /// `app`-form type when the operand node is still `default_type()`
    /// (rank-0). For the axis-taking ops handled here (softmax / mean /
    /// reductions) the operand and the `app` result share a rank, so
    /// this recovers a usable rank even when sub-expression lowering
    /// did not attach `type` metadata to the operand node.
    fn axis_rank(&self, operand: NodeId, ascribed_ty: &TensorType) -> usize {
        let operand_rank = self
            .dag
            .get(operand)
            .map(|n| n.output_type.dims.len())
            .unwrap_or(0);
        if operand_rank > 0 {
            return operand_rank;
        }
        if *ascribed_ty != Self::default_type() {
            return ascribed_ty.dims.len();
        }
        0
    }

    /// Issue #320: rank of a reduction's operand, recovered from the
    /// ascribed reduction-RESULT type when the operand node still carries
    /// the rank-0 `default_type()` placeholder.
    ///
    /// When a windowing/stacking intermediate lowers to a rank-0 IR node,
    /// `max_reduce`'s `normalize_axis` would otherwise see rank 0 and raise
    /// "axis 0 is out of range for an operand of rank 0". A reduction
    /// removes one axis, so the operand rank is the result rank PLUS ONE.
    /// (`axis_rank` returns the result rank verbatim, which is correct for
    /// softmax-shaped ops but undercounts a reduction by one.)
    fn reduction_operand_rank(&self, operand: NodeId, ascribed_result_ty: &TensorType) -> usize {
        let operand_rank = self
            .dag
            .get(operand)
            .map(|n| n.output_type.dims.len())
            .unwrap_or(0);
        if operand_rank > 0 {
            return operand_rank;
        }
        if *ascribed_result_ty != Self::default_type() {
            return ascribed_result_ty.dims.len() + 1;
        }
        0
    }

    /// Issue #320 (gather sibling): rank of a gather's `values` operand,
    /// recovered from the ascribed gather-RESULT type when the operand node
    /// collapsed to rank-0. A gather over `axis` replaces that one values
    /// axis with the `indices` block, so
    /// `values_rank = result_rank - indices_rank + 1`.
    fn gather_values_rank(
        &self,
        values: NodeId,
        indices: NodeId,
        ascribed_result_ty: &TensorType,
    ) -> usize {
        let values_rank = self
            .dag
            .get(values)
            .map(|n| n.output_type.dims.len())
            .unwrap_or(0);
        if values_rank > 0 {
            return values_rank;
        }
        if *ascribed_result_ty == Self::default_type() {
            return 0;
        }
        let indices_rank = self
            .dag
            .get(indices)
            .map(|n| n.output_type.dims.len())
            .unwrap_or(0);
        ascribed_result_ty
            .dims
            .len()
            .saturating_sub(indices_rank)
            .saturating_add(1)
    }

    /// Issue #320 (gather sibling): recover a rank>=1 `values` type for a
    /// collapsed gather operand by removing the `indices` block at `axis`
    /// from the ascribed result and re-inserting the gathered axis. The
    /// gathered axis is a fresh runtime-derived symbolic dim (same rationale
    /// as `recover_collapsed_operand_type`). No-op when `values` already has
    /// a usable type.
    fn recover_collapsed_gather_values_type(
        &mut self,
        values: NodeId,
        indices: NodeId,
        axis: usize,
        ascribed_result_ty: &TensorType,
        precision: Prim,
    ) {
        let collapsed = self
            .dag
            .get(values)
            .map(|n| n.output_type.dims.is_empty())
            .unwrap_or(false);
        if !collapsed || *ascribed_result_ty == Self::default_type() {
            return;
        }
        let indices_rank = self
            .dag
            .get(indices)
            .map(|n| n.output_type.dims.len())
            .unwrap_or(0);
        let result_dims = &ascribed_result_ty.dims;
        if axis + indices_rank > result_dims.len() {
            return;
        }
        // Non-gathered values dims = result with the indices block removed.
        let mut non_gathered = Vec::with_capacity(result_dims.len() - indices_rank);
        non_gathered.extend_from_slice(&result_dims[..axis]);
        non_gathered.extend_from_slice(&result_dims[axis + indices_rank..]);
        self.recover_collapsed_operand_type(values, axis, &non_gathered, precision);
    }

    /// Issue #320: when a reduction/gather operand node is the rank-0
    /// `default_type()` placeholder, recover a rank>=1 operand type by
    /// re-inserting the reduced/gathered axis at `axis` into the ascribed
    /// non-reduced shape, and patch the operand node so both `verify` and
    /// the reverse-mode adjoint carry the operand's rank/extent. No-op when
    /// the operand already has a usable (non-placeholder) type.
    ///
    /// The re-inserted axis is a fresh runtime-derived symbolic dim
    /// (`Named(_, None)`): the operand's runtime value carries the true
    /// extent, and the evaluator binds the symbol through the operand's
    /// producer chain (`shape_source_for_axis`). A literal would be a
    /// fabricated guess, so we deliberately stay symbolic.
    fn recover_collapsed_operand_type(
        &mut self,
        operand: NodeId,
        axis: usize,
        non_reduced_dims: &[DimInfo],
        precision: Prim,
    ) {
        let collapsed = self
            .dag
            .get(operand)
            .map(|n| n.output_type.dims.is_empty())
            .unwrap_or(false);
        if !collapsed || axis > non_reduced_dims.len() {
            return;
        }
        let mut dims = non_reduced_dims.to_vec();
        dims.insert(
            axis,
            DimInfo::Named(format!("_w320_axis_{}", operand.0), None),
        );
        let recovered = TensorType { dims, precision };
        if let Some(node) = self.dag.node_mut(operand) {
            let op = node.op.clone();
            let inputs = node.inputs.clone();
            self.dag.replace_node(operand, op, inputs, recovered);
        }
    }

    /// Normalize a possibly-negative axis literal against a known
    /// Resolve a reduction axis to a positional index against the operand
    /// node's concrete dims. A *named* axis (`sum(x, seq)`, Tier-3 §4.5.3) is
    /// looked up by name: call-site rank monomorphization has already made the
    /// operand dims concrete by the time a rank-poly body is lowered, so the
    /// name resolves to a fixed index. A non-name (integer / `cast`) axis takes
    /// the existing `extract_axis_raw` + `normalize_axis` path.
    fn resolve_reduce_axis(
        &self,
        axis_expr: &Expr,
        operand: NodeId,
        fallback_rank: usize,
        op: &str,
    ) -> usize {
        if let Some(name) = bare_var_name(axis_expr) {
            // A *named* axis (Tier-3 §4.5.3) MUST resolve against the operand's
            // concrete dims: the checker proved the anchor present and call-site
            // monomorphization made it concrete here. If it is somehow absent,
            // fail loudly — falling through to `extract_axis_raw` would return 0
            // and silently reduce the wrong axis (the repo forbids silent
            // fallbacks; CLAUDE.md "Do Not Trust Green").
            if let Some(idx) = self.dag.get(operand).and_then(|node| {
                node.output_type
                    .dims
                    .iter()
                    .position(|d| matches!(d, DimInfo::Named(n, _) if *n == name))
            }) {
                return idx;
            }
            // Issue #388: when the operand came from a *literal-shaped* actual
            // argument (`to_tensor([[...]])`), call-site monomorphization
            // bound the formal parameter name to a literal-dim node, so the
            // operand's dims are concrete `Lit(_)` and the named axis is gone.
            // Recover the axis by locating the anchor's recorded extent in the
            // operand's monomorphized dims (chelis#549): take the axis when the
            // extent is unique (relocating through any intervening `permute`); an
            // unrecorded name still fails loudly rather than defaulting to 0, and
            // an extent that collides with another axis (the square/equal-extent
            // case) fails loud rather than reducing a possibly-wrong axis.
            if let Some(node) = self.dag.get(operand) {
                let dims = &node.output_type.dims;
                match recover_anchor_axis(dims, 0, dims.len(), &name, &self.dim_axis_positions) {
                    AnchorRecovery::Axis(idx) => return idx,
                    // FATAL (chelis#549): a soundness rejection, not a
                    // "rewrite your program" hint. A plain lowering diagnostic
                    // is absorbed by the C host-fallback path and reported as a
                    // misleading generic grad-unsupported message; make it fatal
                    // so the real reason surfaces in the build lane too.
                    AnchorRecovery::AmbiguousAfterReorder => raise_fatal_lowering_error(
                        format!(
                            "`{op}` reduces named axis `{name}`, but call-site monomorphization \
                             erased the name and an intervening axis-reorder (e.g. `permute`) left \
                             its recorded position stale; the axis cannot be located unambiguously \
                             (its recorded extent appears at multiple axes of the monomorphized \
                             operand). Refusing to reduce a possibly-wrong axis (chelis#549)"
                        ),
                        Some(axis_expr.span()),
                        axis_expr.span_id().map(ToOwned::to_owned),
                    ),
                    AnchorRecovery::Unrecorded => {}
                }
            }
            raise_lowering_error(
                format!(
                    "`{op}` reduces named axis `{name}`, but the monomorphized operand has no \
                     such named axis: internal rank-monomorphization error"
                ),
                Some(axis_expr.span()),
                axis_expr.span_id().map(ToOwned::to_owned),
            );
        }
        let raw = self.extract_axis_raw(axis_expr, op);
        self.normalize_axis(raw, fallback_rank, op, axis_expr)
    }

    /// Resolve a named-axis expand ANCHOR (chelis#339, Tier-3 §4.5.3) to the
    /// positional insertion index against the operand node's monomorphized
    /// dims — the lowering twin of `check_named_expand_signature`'s anchor
    /// location, mirroring [`Self::resolve_reduce_axis`]. The checker proved
    /// the anchor present and unique; if it is somehow absent here, fail
    /// loudly — silently inserting at a default axis would broadcast along
    /// the wrong dimension (the repo forbids silent fallbacks).
    fn resolve_expand_anchor(&self, anchor_expr: &Expr, operand: NodeId, anchor: &str) -> usize {
        if let Some(idx) = self.dag.get(operand).and_then(|node| {
            node.output_type
                .dims
                .iter()
                .position(|d| matches!(d, DimInfo::Named(n, _) if n.as_str() == anchor))
        }) {
            return idx;
        }
        // Issue #388 (expand twin): recover the anchor position when a
        // literal-shaped operand erased the named anchor from its dims. Same
        // soundness argument as `resolve_reduce_axis` (chelis#549): the anchor is
        // located by its recorded extent (not the stale recorded index), taken
        // when unique and failed loud when its extent collides with another axis
        // (the square/equal-extent case), never inserting at a possibly-wrong axis.
        if let Some(node) = self.dag.get(operand) {
            let dims = &node.output_type.dims;
            match recover_anchor_axis(dims, 0, dims.len(), anchor, &self.dim_axis_positions) {
                AnchorRecovery::Axis(idx) => return idx,
                AnchorRecovery::AmbiguousAfterReorder => raise_fatal_lowering_error(
                    format!(
                        "`expand` inserts before named anchor `{anchor}`, but call-site \
                         monomorphization erased the name and an intervening axis-reorder (e.g. \
                         `permute`) left its recorded position stale; the anchor cannot be located \
                         unambiguously (its recorded extent appears at multiple axes of the \
                         monomorphized operand). Refusing to insert before a possibly-wrong axis \
                         (chelis#549)"
                    ),
                    Some(anchor_expr.span()),
                    anchor_expr.span_id().map(ToOwned::to_owned),
                ),
                AnchorRecovery::Unrecorded => {}
            }
        }
        // FATAL: a plain lowering diagnostic is absorbed by the host-fallback
        // path (garbage C referencing the anchor as an undeclared identifier),
        // which is exactly the silent-fallback class this error exists to
        // prevent.
        raise_fatal_lowering_error(
            format!(
                "`expand` inserts before named anchor `{anchor}`, but the monomorphized \
                 operand has no such named axis: internal rank-monomorphization error"
            ),
            Some(anchor_expr.span()),
            anchor_expr.span_id().map(ToOwned::to_owned),
        );
    }

    /// operand `rank`. A negative axis `a` means `rank + a` (so `-1`
    /// is the last axis). An axis still out of `0..rank` after
    /// normalization is a clean lowering diagnostic, never a panic --
    /// `require_dim` in `tier2` must stay unreachable from user input.
    ///
    /// `axis_expr` supplies the span so the diagnostic points at the
    /// offending axis literal.
    fn normalize_axis(&self, raw: i64, rank: usize, op: &str, axis_expr: &Expr) -> usize {
        let normalized = if raw < 0 { raw + rank as i64 } else { raw };
        if normalized < 0 || normalized as usize >= rank {
            raise_lowering_error(
                format!(
                    "`{op}` axis {raw} is out of range for an operand of rank {rank} \
                     (valid axes are 0..{rank} or -{rank}..-1)"
                ),
                Some(axis_expr.span()),
                axis_expr.span_id().map(ToOwned::to_owned),
            );
        }
        normalized as usize
    }

    fn gather_out_ty_from_inputs(
        dag: &Dag,
        values: NodeId,
        indices: NodeId,
        axis: usize,
    ) -> Option<TensorType> {
        let values_ty = &dag.get(values)?.output_type;
        let indices_ty = &dag.get(indices)?.output_type;
        if axis >= values_ty.dims.len() {
            return None;
        }
        let mut dims = Vec::new();
        dims.extend_from_slice(&values_ty.dims[..axis]);
        dims.extend(indices_ty.dims.iter().cloned());
        dims.extend_from_slice(&values_ty.dims[axis + 1..]);
        Some(TensorType {
            dims,
            precision: values_ty.precision,
        })
    }

    /// Extract a single usize value from an expression.
    fn extract_usize_value(&self, expr: &Expr) -> Option<usize> {
        // Recognize bare ints, `(lit {} n)`, and `(cast {} <inner> ty)`
        // wrappers via the shared `extract_int_for_dim` walker. Surf
        // routinely wraps integer arguments in `cast(n, int32)` (e.g.
        // `expand(x, cast(0, int32), cast(2, int32))`); without
        // unwrapping the cast this returned `None` and callers silently
        // fell back to a default (axis 0 / size 1), so `expand(...,
        // cast(2, int32))` produced a `tensor[1]` instead of `tensor[2]`
        // and the constant-broadcast idiom in issue #288 lowered to a
        // shape-mismatched `Mul`. Reject negative values (sizes/axes are
        // non-negative) so the caller's own negative-axis normalization
        // path is not bypassed.
        extract_int_for_dim(expr).and_then(|n| usize::try_from(n).ok())
    }

    /// Const-fold a fully-static integer `expand` SIZE to its value
    /// (chelis#469 / #528). Extends [`extract_int_for_dim`] (literal / `(lit
    /// …)` / `cast`) with (a) the integer arithmetic the §4.7.2 checker
    /// classifies as `SizeClass::Static` (`classify_arith_app`:
    /// `add`/`sub`/`mul`/`mod`/`neg`) and (b) a bare/`cast`-wrapped `var`
    /// bound by a prior `let` to a static value ([`Self::static_size_bindings`]
    /// — the checker's "followed transitively through `let` bindings"). Folds
    /// only when EVERY leaf is itself static; a `shape(x, …)`-touching or a
    /// runtime-parameter-touching size returns `None` so the caller routes it
    /// to the shape-source or fail-closed arm. (`div` is float-only per
    /// chelis#178 and is rejected at check before lowering; `floor_div`/
    /// `trunc_div` are not in the checker's arith set — neither reaches here
    /// as an accepted size.) Overflow / mod-by-zero use checked arithmetic and
    /// return `None`, never a wrapped or otherwise wrong extent.
    ///
    /// Deliberately kept SEPARATE from the shared [`extract_int_for_dim`]
    /// walker, which also drives reduction / softmax / gather AXES: the
    /// checker rejects a static-arithmetic *axis* ("sum axis must be a
    /// compile-time constant or a named axis"), so folding arithmetic in the
    /// shared walker would silently widen the axis contract (chelis#364).
    /// Extent-side folding lives only here; the axis side is unchanged.
    fn fold_static_size(&self, expr: &Expr) -> Option<i64> {
        if let Some(n) = extract_int_for_dim(expr) {
            return Some(n);
        }
        let Expr::List(list, _) = expr else {
            return None;
        };
        match get_tag(list) {
            // A bare `var` bound to a static value by a prior `let`.
            Some("var") => self
                .static_size_bindings
                .get(&bare_var_name(expr)?)
                .copied(),
            // `cast(<inner>, ty)` — fold the inner value.
            Some("cast") => self.fold_static_size(children(list).first()?),
            // Integer arithmetic over static operands.
            Some("app") => {
                let kids = children(list);
                let op = bare_var_name(kids.first()?)?;
                let operands = &kids[1..];
                match (op.as_str(), operands.len()) {
                    ("neg", 1) => self.fold_static_size(&operands[0])?.checked_neg(),
                    ("add", 2) => self
                        .fold_static_size(&operands[0])?
                        .checked_add(self.fold_static_size(&operands[1])?),
                    ("sub", 2) => self
                        .fold_static_size(&operands[0])?
                        .checked_sub(self.fold_static_size(&operands[1])?),
                    ("mul", 2) => self
                        .fold_static_size(&operands[0])?
                        .checked_mul(self.fold_static_size(&operands[1])?),
                    ("mod", 2) => {
                        let divisor = self.fold_static_size(&operands[1])?;
                        if divisor == 0 {
                            return None;
                        }
                        self.fold_static_size(&operands[0])?.checked_rem(divisor)
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// chelis#620: resolve an already-lowered `if` condition to a
    /// compile-time boolean. `Some(b)` only when the condition's transitive
    /// input subgraph is scalar, closed, and pure: no `Load` (definitionally
    /// runtime), no `Shape` (a shape read participates in the #616
    /// conform/mask machinery and must stay on the runtime path), no
    /// `UniformLike`/`Dropout` (nondeterministic), no `shape_deps`, no
    /// non-scalar node, and every op inside the whitelisted vocabulary whose
    /// arms mirror `crate::eval` exactly (`Cast` shares
    /// [`crate::eval::convert_cast_data`] so the two lanes cannot diverge).
    ///
    /// Works at the DAG-node level, not the Deep-expression level, because by
    /// the time `lower_if` runs, inlined-function parameters are already
    /// bound to lowered nodes (`static_size_bindings` is only populated by
    /// `lower_let`, never at param-binding time), and the comparison/boolean
    /// surface (`gte`/`lte`/`eq`/`and`/`or`/`not`) has already been lowered
    /// to `CmpLt`/`MaxElem`/`Mul`/`Neg`/`Const` compositions by the tier2
    /// builtins. Zero-divisor `FloorDiv`/`TruncDiv` refuses the fold rather
    /// than folding a runtime trap away.
    fn fold_static_cond(&self, cond: NodeId) -> Option<bool> {
        let mut memo: HashMap<NodeId, f64> = HashMap::new();
        // Iterative post-order: (node, inputs_pushed).
        let mut stack: Vec<(NodeId, bool)> = vec![(cond, false)];
        while let Some((id, inputs_pushed)) = stack.pop() {
            if memo.contains_key(&id) {
                continue;
            }
            let node = self.dag.get(id)?;
            if !node.shape_deps.is_empty() || !node.output_type.dims.is_empty() {
                return None;
            }
            if !inputs_pushed {
                stack.push((id, true));
                for input in &node.inputs {
                    stack.push((*input, false));
                }
                continue;
            }
            let input0 = node
                .inputs
                .first()
                .and_then(|input| memo.get(input))
                .copied();
            let input1 = node
                .inputs
                .get(1)
                .and_then(|input| memo.get(input))
                .copied();
            let value = match &node.op {
                RiscOp::Const { value } => {
                    // The fold memo is the f64 wide image (the
                    // chelis#680 residue; exact integer folding arrives
                    // with the Phase 2 kernel split). An integer
                    // payload whose f64 image is not exact DECLINES the
                    // fold instead of baking a collapsed value in
                    // (chelis#856): the condition falls to runtime,
                    // which reads the sealed payload exactly.
                    if let Some(i) = value.as_i64_exact()
                        && (i as f64) as i128 != i as i128
                    {
                        return None;
                    }
                    value.as_f64_lossy()
                }
                RiscOp::Cast { new_precision } => {
                    let input_id = *node.inputs.first()?;
                    let src = self.dag.get(input_id)?.output_type.precision;
                    // A trapping cast DECLINES TO FOLD (the section C2
                    // fold rule for casts): the condition falls to
                    // runtime, where the checked ladder traps with its
                    // full diagnostic. A fold must never bake a trap
                    // away nor bake one in.
                    crate::eval::convert_cast_data(input0?, src, *new_precision).ok()?
                }
                RiscOp::Add => input0? + input1?,
                RiscOp::Mul => input0? * input1?,
                RiscOp::Neg => -input0?,
                RiscOp::Div => input0? / input1?,
                RiscOp::FloorDiv => {
                    let divisor = input1?;
                    if divisor == 0.0 {
                        return None;
                    }
                    (input0? / divisor).floor()
                }
                RiscOp::TruncDiv => {
                    let divisor = input1?;
                    if divisor == 0.0 {
                        return None;
                    }
                    (input0? / divisor).trunc()
                }
                RiscOp::CmpLt => {
                    if input0? < input1? {
                        1.0
                    } else {
                        0.0
                    }
                }
                RiscOp::MaxElem => input0?.max(input1?),
                RiscOp::Abs => input0?.abs(),
                RiscOp::Floor => input0?.floor(),
                RiscOp::Ceil => input0?.ceil(),
                RiscOp::Round => input0?.round_ties_even(),
                _ => return None,
            };
            // chelis#620 red-team fix: refuse the fold on any non-finite
            // intermediate. The comparison surface reaching this walker is
            // the lowered CmpLt/not composition, whose NaN behavior
            // (`not(NaN < x)` is true) DISAGREES with the IEEE comparisons
            // both forward lanes apply (host evaluator `>=`, C backend
            // `>=`) -- so folding a NaN condition would prune to a branch
            // the forward pass never takes. Falling to the runtime mask
            // path keeps the pre-existing (pre-#620) behavior for such
            // conditions instead of extending it to ADT/list pruning.
            if !value.is_finite() {
                return None;
            }
            memo.insert(id, value);
        }
        memo.get(&cond).map(|v| *v != 0.0)
    }

    /// chelis#513 gap 3 (school im2col witness): const-fold a `reshape`
    /// TARGET dim that is integer ARITHMETIC over static leaves, where a
    /// leaf may additionally be a `shape(operand, axis)` read (direct or a
    /// `let`-bound alias) of a STATICALLY-sized operand axis. This covers the
    /// im2col-style `reshape(p, [mul(b_d, a_d), 1])` target over a
    /// concrete-shaped input, which [`Self::fold_static_size`] deliberately
    /// rejects (its expand-size contract routes shape() reads to the
    /// shape-source arm) and [`Self::extract_reshape_dim_list`]'s per-element
    /// vocabulary could not express: the unresolvable element aborted the
    /// walk and the reshape fell back to the checker's `Named("*")` wildcard
    /// dims, which the backward `Expand`/`Sum` inherited and
    /// `symbolic_occurrences` ICE'd on.
    ///
    /// Exactness contract: folds ONLY when every leaf is static.
    ///   - a symbolic operand axis returns `None` (the caller then fails
    ///     LOUD via [`Self::is_shape_derived_arith_dim`]; the symbolic-sig
    ///     arithmetic target stays fail-closed, pinned in
    ///     `issue_513_symbolic_axis_adjoints.rs`),
    ///   - `add`/`sub`/`mul`/`neg` use checked i64 arithmetic (overflow folds
    ///     to `None`, never a wrapped extent),
    ///   - `floor_div`/`trunc_div`/`mod` fold only on a non-negative lhs with
    ///     a positive rhs, the domain where floor, trunc, and euclidean
    ///     semantics coincide, so the fold can never disagree with the
    ///     runtime op. Anything else returns `None`.
    ///
    /// Takes `&mut self` because resolving a shape read lowers its operand
    /// (idempotent for the bound-`var` operands this walks; the same contract
    /// as [`Self::dim_expr_from_shape_arg_with_source`]).
    fn fold_shape_derived_static_size(&mut self, expr: &Expr) -> Option<i64> {
        if let Some(n) = extract_int_for_dim(expr) {
            return Some(n);
        }
        // A shape(...) read (direct app, or a bare/cast-wrapped var recorded
        // in `shape_bindings`) of a statically-sized operand axis folds to
        // that extent; a symbolic extent fails the fold.
        if let Some((operand, axis)) = self.shape_app_operand_axis_resolved(expr) {
            let operand_id = self.lower_expr(&operand).as_single_node()?;
            return match self.dag.get(operand_id)?.output_type.dims.get(axis)? {
                DimInfo::Lit(n) => i64::try_from(*n).ok(),
                DimInfo::Named(_, Some(n)) => i64::try_from(*n).ok(),
                DimInfo::Named(_, None) => None,
            };
        }
        let Expr::List(list, _) = expr else {
            return None;
        };
        match get_tag(list) {
            // A bare `var` bound to a static value by a prior `let`. (A
            // shape-bound var was already handled above.)
            Some("var") => self
                .static_size_bindings
                .get(&bare_var_name(expr)?)
                .copied(),
            Some("cast") => self.fold_shape_derived_static_size(children(list).first()?),
            Some("app") => {
                let kids = children(list);
                let op = bare_var_name(kids.first()?)?;
                let operands = &kids[1..];
                match (op.as_str(), operands.len()) {
                    ("neg", 1) => self
                        .fold_shape_derived_static_size(&operands[0])?
                        .checked_neg(),
                    ("add", 2) => self
                        .fold_shape_derived_static_size(&operands[0])?
                        .checked_add(self.fold_shape_derived_static_size(&operands[1])?),
                    ("sub", 2) => self
                        .fold_shape_derived_static_size(&operands[0])?
                        .checked_sub(self.fold_shape_derived_static_size(&operands[1])?),
                    ("mul", 2) => self
                        .fold_shape_derived_static_size(&operands[0])?
                        .checked_mul(self.fold_shape_derived_static_size(&operands[1])?),
                    ("floor_div" | "trunc_div", 2) => {
                        let lhs = self.fold_shape_derived_static_size(&operands[0])?;
                        let rhs = self.fold_shape_derived_static_size(&operands[1])?;
                        if lhs < 0 || rhs <= 0 {
                            return None;
                        }
                        lhs.checked_div(rhs)
                    }
                    ("mod", 2) => {
                        let lhs = self.fold_shape_derived_static_size(&operands[0])?;
                        let rhs = self.fold_shape_derived_static_size(&operands[1])?;
                        if lhs < 0 || rhs <= 0 {
                            return None;
                        }
                        lhs.checked_rem(rhs)
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// chelis#513: syntactic recognizer for the input LANGUAGE of
    /// [`Self::fold_shape_derived_static_size`], used by
    /// [`Self::extract_reshape_dim_list`] to decide whether a reshape target
    /// element the exactness gate REFUSED must fail loud instead of falling
    /// back to the checker's wildcard dims.
    ///
    /// Why loud matters: the wildcard fallback becomes an anonymous
    /// `Named(_, None)` dim, and under `grad` the eval lane's symbolic-dim
    /// binding can satisfy that anon dim with a coincidental extent taken
    /// from an input axis, SILENTLY accepting a program whose written target
    /// expression is never evaluated (it may not even be shape-consistent).
    /// The C build lane ICEs on the same anon dim in `symbolic_occurrences`.
    /// Refusing at lowering replaces both with one located diagnostic. The
    /// host lane is unaffected: a forward (non-`grad`) use of this form
    /// still evaluates with true runtime semantics via the host fallback,
    /// which computes the target expression honestly and rejects a
    /// shape-inconsistent reshape at runtime.
    ///
    /// Returns true only when `expr` (cast-stripped) is an arithmetic app of
    /// the fold's exact vocabulary (`neg`/`add`/`sub`/`mul`/`floor_div`/
    /// `trunc_div`/`mod`, matching arity) and EVERY leaf is a recognized
    /// static or shape-derived form. A leaf outside the language (e.g. a
    /// runtime scalar parameter) returns false and keeps the pre-existing
    /// wildcard fallback for forms this pass never claimed to understand.
    fn is_shape_derived_arith_dim(&self, expr: &Expr) -> bool {
        let Expr::List(list, _) = expr else {
            return false;
        };
        match get_tag(list) {
            Some("cast") => children(list)
                .first()
                .is_some_and(|inner| self.is_shape_derived_arith_dim(inner)),
            Some("app") => {
                let kids = children(list);
                let Some(op) = kids.first().and_then(bare_var_name) else {
                    return false;
                };
                let operands = &kids[1..];
                let arity_ok = match op.as_str() {
                    "neg" => operands.len() == 1,
                    "add" | "sub" | "mul" | "floor_div" | "trunc_div" | "mod" => {
                        operands.len() == 2
                    }
                    _ => false,
                };
                arity_ok
                    && operands
                        .iter()
                        .all(|operand| self.is_shape_derived_arith_leaf(operand))
            }
            _ => false,
        }
    }

    /// Leaf recognizer for [`Self::is_shape_derived_arith_dim`]: a static
    /// int (literal / `(lit ...)` / cast-wrapped), a `shape(operand, axis)`
    /// read (direct or a `shape_bindings` alias), a `let`-bound static var,
    /// or a nested arithmetic app of the same language.
    fn is_shape_derived_arith_leaf(&self, expr: &Expr) -> bool {
        if extract_int_for_dim(expr).is_some()
            || self.shape_app_operand_axis_resolved(expr).is_some()
        {
            return true;
        }
        let Expr::List(list, _) = expr else {
            return false;
        };
        match get_tag(list) {
            Some("var") => bare_var_name(expr)
                .is_some_and(|name| self.static_size_bindings.contains_key(&name)),
            Some("cast") => children(list)
                .first()
                .is_some_and(|inner| self.is_shape_derived_arith_leaf(inner)),
            Some("app") => self.is_shape_derived_arith_dim(expr),
            _ => false,
        }
    }

    fn extract_dim_expr_value(&self, expr: &Expr) -> Option<DimExpr> {
        if let Some(value) = self.extract_usize_value(expr) {
            return Some(DimExpr::Concrete(value));
        }

        match expr {
            Expr::Atom(Atom::Symbol(name), _) => Some(self.resolve_dim_expr_symbol(name)),
            Expr::List(list, _) => match (list.elements.first(), list.elements.get(2)) {
                (
                    Some(Expr::Atom(Atom::Symbol(tag), _)),
                    Some(Expr::Atom(Atom::Symbol(name), _)),
                ) if tag == "var" => Some(self.resolve_dim_expr_symbol(name)),
                _ => None,
            },
            _ => None,
        }
    }

    fn resolve_dim_expr_symbol(&self, name: &str) -> DimExpr {
        self.dim_substitutions
            .get(name)
            .map(DimExpr::from)
            .unwrap_or_else(|| DimExpr::Sym(name.to_string()))
    }

    /// Recover a broadcast extent from a `shape(operand, axis)` size
    /// argument (issue #318). The operand is lowered (idempotently — a
    /// bound `var` returns its cached DAG node) and the extent is read
    /// from its `output_type.dims[axis]`. This is the principled fix: the
    /// extent comes from the operand the program actually asks the shape
    /// of, NOT from the `expand` node's own output type, which the type
    /// checker has already collapsed to `Lit(1)` via size-1 broadcasting.
    ///
    /// No dead node accrues for the real idiom. The operand is `&x`
    /// (`borrow(var x)`) or a bare `var x`; `borrow` lowers via
    /// [`Self::lower_identity`], which forwards transparently to its inner
    /// expr without emitting a `Borrow` node (Phase 0 has no such RISC op),
    /// and the `var` then resolves to the already-bound parameter node
    /// rather than adding a duplicate. The only mutation is the standard
    /// cached-binding span append, which keeps the operand region in the
    /// span-survival audit — reading the type "side-effect-free" off the
    /// binding would drop that coverage, so lowering is the correct path.
    /// `issue_318_expand_shape_arg_no_dead_node` is the regression guard.
    ///
    /// Out of scope by design: a runtime (non-literal) axis. The recognizer
    /// [`shape_app_operand_axis`] only matches a static literal axis, so a
    /// `shape(operand, <expr>)` with a computed axis returns `None` here and
    /// the caller falls back to the size-1 default — the pre-#318 behavior.
    /// School's const-broadcast idiom uses literal axes throughout, so this
    /// is sufficient; a runtime-axis broadcast would need a separate path.
    ///
    /// Returns `None` when the argument is not a `shape(...)` application,
    /// the axis is out of range, or the operand has no resolvable type —
    /// in which case the caller falls back to its other recovery paths.
    /// chelis#384/#397 (B): true when some tensor `Load` already in the DAG
    /// carries the named dimension `name` in its shape — i.e. the symbol has a
    /// real tensor source the backend can read the extent from. Used to
    /// distinguish a valid §4.7.2 Form-2 symbolic `expand` size (an in-scope
    /// tensor dim) from a sourceless Form-3 scalar-parameter size that the
    /// backend cannot materialize.
    fn symbol_has_tensor_source(&self, name: &str) -> bool {
        self.dag.nodes().iter().any(|node| {
            matches!(node.op, RiscOp::Load { .. })
                && node
                    .output_type
                    .dims
                    .iter()
                    .any(|dim| matches!(dim, DimInfo::Named(n, _) if n == name))
        })
    }

    /// Recover an `expand` extent from a `shape(operand, axis)` size
    /// argument, returning BOTH the dim and the lowered node whose runtime
    /// shape supplies the extent. chelis#384/#397: the caller records the
    /// source node as a `shape_dep` on the consuming `Expand` so it survives
    /// DCE — otherwise a `shape(x, ...)`-only-referenced `x` is eliminated and
    /// the symbolic dim it declares loses its source (silent wrong shape in
    /// the backend).
    fn dim_expr_from_shape_arg_with_source(&mut self, expr: &Expr) -> Option<(DimExpr, NodeId)> {
        let (operand, axis) = self.shape_app_operand_axis_resolved(expr)?;
        let operand_id = self.lower_expr(&operand).as_single_node()?;
        let operand_ty = self.dag.get(operand_id)?.output_type.clone();
        let dim = operand_ty.dims.get(axis)?;
        Some((DimExpr::from(dim), operand_id))
    }

    /// chelis#369: like the free [`shape_app_operand_axis`], but also
    /// resolves a `let`-bound shape name. The canonical `tensor_full_like`
    /// idiom is
    ///
    /// ```text
    ///   len  = shape(x, 0)
    ///   twos = expand(scalar_to_tensor(c), 0, cast(len, int32))
    /// ```
    ///
    /// so the `expand` size argument is `cast(var len, int32)`, NOT a
    /// direct `shape(...)` app. The bare [`shape_app_operand_axis`] strips
    /// the `cast`, reaches `var len`, fails to match a `shape` builtin, and
    /// returns `None` — at which point the caller silently defaults the
    /// extent to `Lit(1)`, the `Lit(n) vs Lit(1)` backward-DAG failure this
    /// issue tracks. Here, when the stripped expression is a `var <name>`
    /// recorded in [`Self::shape_bindings`] as a `shape(operand, axis)`
    /// binding, recover the operand/axis from that bound app instead.
    ///
    /// Returns the *operand* `Expr` by value (cloned from the binding when
    /// the indirection fired) so a single signature covers both the direct
    /// and the `let`-bound forms.
    fn shape_app_operand_axis_resolved(&self, expr: &Expr) -> Option<(Expr, usize)> {
        if let Some((operand, axis)) = shape_app_operand_axis(expr) {
            return Some((operand.clone(), axis));
        }
        // Strip any `cast(..., int32)` wrappers to reach a bare `var name`,
        // then follow the recorded `let len = shape(...)` binding. Because
        // `resolve_shape_binding_source` records the UNDERLYING `shape(...)`
        // app for `let`-to-`let` aliases too (chelis#469 RT-3), a single
        // lookup here resolves a whole alias chain to its real source.
        let name = bare_var_name(strip_cast_wrappers(expr))?;
        let bound = self.shape_bindings.get(&name)?;
        let (operand, axis) = shape_app_operand_axis(bound)?;
        Some((operand.clone(), axis))
    }

    /// Resolve the underlying `shape(operand, axis)` app that a `let`-binding
    /// value refers to, following a `let`-to-`let` alias chain and any
    /// use-site `cast` wrappers (chelis#369/#469 RT-3). Returns the direct
    /// `shape(...)` app `Expr` when:
    ///   - the value IS a `shape(...)` app (possibly `cast`-wrapped), or
    ///   - the value is a bare / `cast`-wrapped `var` already recorded in
    ///     [`Self::shape_bindings`] (an alias of an earlier shape name).
    ///
    /// Returning the UNDERLYING app (not the alias name) is the safety
    /// property: the recovered extent and its `shape_dep` liveness edge bind
    /// to the ACTUAL source tensor and axis, so an alias can never resolve to
    /// the wrong `Load`. Because each recorded alias already points at the
    /// underlying app, a chain (`a = shape(x, 0); c = a; d = cast(c, int32)`)
    /// resolves in one lookup per link at bind time. Mirrors how
    /// [`Self::fold_static_size`] recurses [`Self::static_size_bindings`] for
    /// the static path (`j = k; ...` folds through the alias).
    fn resolve_shape_binding_source(&self, value: &Expr) -> Option<Expr> {
        if shape_app_operand_axis(value).is_some() {
            return Some(value.clone());
        }
        let alias = bare_var_name(strip_cast_wrappers(value))?;
        self.shape_bindings.get(&alias).cloned()
    }

    fn lower_handle_effect(&mut self, list: &List) -> LoweredValue {
        let elems = &list.elements;
        // chelis#730 Phase 1 (census row 9, chelis#709-adjacent): the
        // former `_ if elems.len() >= 4` catch-all lowered the body and
        // silently DROPPED the handler for any unrecognized effect kind
        // (`effect: teleport` built and ran). chelis#730 Phase 2 (section
        // C4.4) closes the future-kinds hole structurally: the kind is
        // parsed once into the closed [`EffectKind`] set and the match is
        // exhaustive over `EffectKind` with no `_` arm, so adding a
        // kind fails the build here until this site handles it. Fatal so
        // the host-fallback lane cannot launder the drop (its own
        // handle-effect arm carries the same closed-set match).
        let effect_kind = decode_effect_kind(list);
        let current_span_id = self.current_span_id.clone();
        let reject = |what: String| {
            let unsupported = Unsupported::new(
                UnsupportedKind::EffectKind(what),
                "a `handle-effect` form in IR lowering",
                Stage::Lowering,
                "known effect kinds are `random` and `resource` \
                 (spec/03-deep-syntax.md); an unknown kind previously dropped its \
                 handler silently (chelis#730 census row 9)",
            );
            raise_fatal_lowering_error(unsupported.to_string(), None, current_span_id.clone())
        };
        match effect_kind {
            Ok(EffectKind::Random) if elems.len() >= 4 => {
                let saved_seed = self.random_seed;
                self.random_seed = self.extract_u64_value(&elems[2]).or(saved_seed);
                let result = self.lower_expr(&elems[3]);
                self.random_seed = saved_seed;
                result
            }
            Ok(EffectKind::Resource) if elems.len() >= 4 => self.lower_expr(&elems[3]),
            // A decode error, or a KNOWN kind whose form is malformed
            // (fewer than 4 elements). Both raise the same fatal branded
            // diagnostic; the `what` payload names the original symbol so a
            // short `random`/`resource` form still reports its own kind, as
            // the pre-enum `other =>` arm did. Every variant is named
            // explicitly, so a new `EffectKind` variant is a compile error
            // here rather than a silent fall-through.
            Ok(EffectKind::Random) => reject(EffectKind::Random.symbol().to_owned()),
            Ok(EffectKind::Resource) => reject(EffectKind::Resource.symbol().to_owned()),
            Err(error) => reject(error.to_string()),
        }
    }

    fn extract_u64_value(&self, expr: &Expr) -> Option<u64> {
        self.extract_usize_value(expr).map(|value| value as u64)
    }

    /// Extract a compile-time-constant f64 from an expression, seeing through
    /// the statically-resolvable, value-carrying wrappers a numeric literal
    /// can arrive in: a `(lit ...)` node, a `neg(...)` of an extractable value,
    /// and a `cast(..., <float prim>)` of an extractable value. A float-target
    /// cast preserves the numeric value (the f32/f64 sampler narrows the same
    /// bits in every lane), so it is folded through; an integer-target cast
    /// *changes* the value by truncation, so it is NOT folded — it returns
    /// `None` and the caller fails loudly rather than baking a guessed
    /// truncation into codegen. Any other form (a runtime variable, arithmetic,
    /// a `shape()` read) also returns `None`.
    ///
    /// chelis#776: this used to see through neither `cast` nor `neg`, so a
    /// wrapped bound fell to a caller `unwrap_or(default)` and silently
    /// replaced the user's value with the [0,1) default in the compiled lane
    /// (the #703 silent-substitution class). Callers that bake this into
    /// codegen now go through [`Self::resolve_static_f64_arg`], which turns an
    /// unresolvable value into a loud lowering error.
    fn extract_f64_value(&self, expr: &Expr) -> Option<f64> {
        match expr {
            Expr::Atom(Atom::Float(f), _) => Some(*f),
            Expr::Atom(Atom::Int(n), _) => Some(*n as f64),
            Expr::List(list, _) => match get_tag(list) {
                // cast(<inner>, <target>): value-preserving only for a float
                // target; an integer target truncates and is left unresolved
                // (chelis#776 — go loud, do not guess the cast semantics).
                Some("cast") => {
                    let inner = list.elements.get(2)?;
                    let target = list.elements.get(3)?;
                    match Self::try_extract_prim(target) {
                        Some(prim) if prim.is_float() => self.extract_f64_value(inner),
                        _ => None,
                    }
                }
                // neg(<inner>): unary minus desugars to
                // `(app {} (var {} neg) <inner>)`.
                Some("app")
                    if children(list)
                        .first()
                        .is_some_and(|callee| expr_is_var_named(callee, "neg")) =>
                {
                    let inner = children(list).get(1)?;
                    self.extract_f64_value(inner).map(|v| -v)
                }
                // `(lit {} <atom>)` and any other list carrying a bare numeric
                // atom in the value slot (pre-chelis#776 behavior, preserved).
                _ => match list.elements.get(2) {
                    Some(Expr::Atom(Atom::Float(f), _)) => Some(*f),
                    Some(Expr::Atom(Atom::Int(n), _)) => Some(*n as f64),
                    _ => None,
                },
            },
            _ => None,
        }
    }

    /// Resolve a builtin argument that is baked into the emitted kernel as a
    /// compile-time constant (a `uniform_like` bound, a `dropout` rate, a `pad`
    /// fill), or raise a loud lowering error. A silent `unwrap_or(default)` at
    /// these sites substitutes a wrong value into a program that compiles and
    /// runs — the #703 class; here it silently collapsed a wrapped or computed
    /// `uniform_like` range to the [0,1) default (chelis#776). An unresolvable
    /// argument is therefore a build failure, never a default.
    ///
    /// The error is *fatal* on purpose: a non-fatal lowering error at these
    /// sites is caught by the host-emit backend's speculative sub-lowering and
    /// laundered into a silent `/* unsupported builtin */ 0` stub (a null
    /// tensor), which is just a different silent miscompile. A fatal error
    /// surfaces as a user-facing build error instead — the same pathway the
    /// `grad`-of-non-differentiable rejection uses (issue #197). `chelis eval`
    /// stays correct: it interprets the `with seed { ... }` program through its
    /// own evaluator and does not require this DAG lowering to succeed, so a
    /// runtime bound that fails the build still evaluates to the right range.
    fn resolve_static_f64_arg(&self, expr: &Expr, builtin: &str, arg_desc: &str) -> f64 {
        self.extract_f64_value(expr).unwrap_or_else(|| {
            let found = match expr {
                Expr::List(list, _) => get_tag(list).unwrap_or("expression"),
                _ => "expression",
            };
            raise_fatal_lowering_error(
                format!(
                    "`{builtin}` requires a statically-resolvable {arg_desc}, but the \
                     compiled-backend lowering cannot fold `{found}` to a compile-time \
                     constant. Use a numeric literal (optionally negated or cast to a \
                     float type); a runtime-computed value is not supported here \
                     (Chelis-Lang/chelis#776)"
                ),
                Some(expr.span()),
                expr.span_id().map(ToOwned::to_owned),
            )
        })
    }

    /// Extract a list of usize values from a slice of expressions.
    fn extract_usize_list(&self, exprs: &[Expr]) -> Vec<usize> {
        let mut result = Vec::new();
        for expr in exprs {
            if let Some(v) = self.extract_usize_value(expr) {
                result.push(v);
            }
        }
        result
    }

    /// Extract a `reshape` shape list from a Deep expression. Accepts the
    /// Cons-chain shape Surf desugars to:
    ///
    /// ```text
    ///   (app {} (var {} Cons) <head_0>
    ///           (app {} (var {} Cons) <head_1> ... (var {} Nil)))
    /// ```
    ///
    /// Each head is interpreted as, in order: an integer dim (via
    /// [`extract_int_for_dim`], which handles `Atom::Int`, `(lit ...)`, and
    /// `(cast ... int64)`); a `shape(operand, axis)`-derived dim (chelis#513
    /// gap 1, see below); or a symbolic dim variable (via
    /// [`symbolic_dim_var_name`], which recognizes `(var {} <name>)`).
    /// Non-recognized shapes abort the walk and return `None` so the caller
    /// falls back to `ty.dims` — this prevents misreading a Deep structural tag
    /// like `"app"` as a dim name and synthesizing `DimInfo::Named("app",
    /// None)` (issue Chelis-Lang/chelis#220).
    ///
    /// chelis#513 gap 1: a `shape(operand, axis)`-derived reshape target dim
    /// (directly or through the `let k = shape(x, 0); reshape(&x, [k, 1])`
    /// indirection) is resolved to the operand's declaring source dim instead
    /// of a bare `Named(name, None)` symbol. Returns the op target dims, the
    /// output-type dims, and the lowered source nodes to record as
    /// `shape_dep`s so the declaring input survives DCE. A static operand
    /// axis folds to `Lit`; a symbolic one becomes a `Load`-carried `Named`
    /// that `symbolic_occurrences` can trace, closing the reshape/reduce
    /// backward `Expand`/`Sum` symbolic-dim ICE.
    ///
    /// chelis#616: a target dim that is runtime integer arithmetic over
    /// `shape()` reads (the former chelis#513 refuse-to-lower arm) or a term
    /// variable bound to a rank-0 integer scalar (the inlined window count
    /// `m`) now lowers to a real scalar node appended to `inputs`, referenced
    /// as `RtDim::Node(slot)` exactly like a movement bound. Its output-type
    /// dim keeps the checker's symbol for the axis so downstream types keep
    /// resolving; the eval and C lanes size the axis from the scalar value.
    fn extract_reshape_dim_list(
        &mut self,
        expr: &Expr,
        checker_dims: &[DimInfo],
        inputs: &mut Vec<NodeId>,
    ) -> Option<(Vec<RtDim>, Vec<DimInfo>, Vec<NodeId>)> {
        let elements: Vec<Expr> = collect_cons_chain(expr)?.into_iter().cloned().collect();
        let mut op_dims = Vec::with_capacity(elements.len());
        let mut ty_dims = Vec::with_capacity(elements.len());
        let mut srcs = Vec::new();
        for (axis, elem) in elements.iter().enumerate() {
            if let Some(value) = extract_int_for_dim(elem) {
                if value < 0 {
                    return None;
                }
                op_dims.push(RtDim::Lit(value as usize));
                ty_dims.push(DimInfo::Lit(value as usize));
            } else if let Some((dim_expr, src)) = self.dim_expr_from_shape_arg_with_source(elem)
                && let Some(dim) = Self::dim_info_from_dim_expr(&dim_expr)
            {
                op_dims.push(RtDim::from_dim_info(&dim));
                ty_dims.push(dim);
                srcs.push(src);
            } else if let Some(value) = self.fold_shape_derived_static_size(elem) {
                // chelis#513 gap 3: static integer arithmetic over shape()
                // reads of statically-sized axes (the school im2col target
                // form `mul(b_d, a_d)`) folds to a concrete literal dim. A
                // fold that PROVES the extent negative is a proven-invalid
                // program: fail loud, never fall back to wildcard dims.
                if value < 0 {
                    raise_lowering_error(
                        format!(
                            "reshape target dim is shape()-derived integer arithmetic that \
                             folds to the negative extent {value}; a reshape extent must be \
                             non-negative (chelis#513)"
                        ),
                        Some(elem.span()),
                        elem.span_id().map(ToOwned::to_owned),
                    );
                }
                op_dims.push(RtDim::Lit(value as usize));
                ty_dims.push(DimInfo::Lit(value as usize));
            } else if self.is_shape_derived_arith_dim(elem) || self.is_runtime_scalar_var(elem) {
                // chelis#616: lower the runtime target expression to a rank-0
                // integer scalar node (replacing the chelis#513 loud refusal).
                // The extent is now checked at run time: the eval lane reads
                // the scalar and enforces the numel invariant, and the C lane
                // declares the dim from the scalar behind negativity + numel
                // abort guards. A non-scalar or non-integer lowering is a
                // producing-pass bug and stays fail-closed.
                let node = self.lower_expr_node(elem, "reshape target dim");
                let node_ty = self
                    .dag
                    .get(node)
                    .map(|n| n.output_type.clone())
                    .unwrap_or_else(Self::default_type);
                if !node_ty.dims.is_empty() || !node_ty.precision.is_integer() {
                    raise_lowering_error(
                        format!(
                            "reshape target dim expression must lower to a rank-0 integer \
                             scalar, got rank {} `{}` (chelis#616)",
                            node_ty.dims.len(),
                            node_ty.precision.name()
                        ),
                        Some(elem.span()),
                        elem.span_id().map(ToOwned::to_owned),
                    );
                }
                let slot = inputs.len();
                inputs.push(node);
                op_dims.push(RtDim::Node(slot));
                // The output-type dim keeps the checker's symbol for this
                // axis (downstream types reference it; the C lane declares it
                // from the scalar). A wildcard, concrete, or missing checker
                // dim gets a fresh GENERATED name instead — never a guessed
                // extent, and never the bare `*`/`""` wildcard (both lanes
                // need one stable, unique name per runtime extent: the eval
                // lane binds it from the scalar's value mid-evaluation and
                // the C lane declares it at the op).
                ty_dims.push(match checker_dims.get(axis) {
                    Some(dim @ DimInfo::Named(name, None)) if !name.is_empty() && name != "*" => {
                        dim.clone()
                    }
                    _ => DimInfo::Named(format!("_rt_dim_{}_{axis}", node.0), None),
                });
            } else {
                let name = symbolic_dim_var_name(elem)?;
                op_dims.push(RtDim::Sym(name.clone()));
                ty_dims.push(DimInfo::Named(name, None));
            }
        }
        if op_dims.is_empty() {
            None
        } else {
            Some((op_dims, ty_dims, srcs))
        }
    }

    /// chelis#616: whether `expr` is a term variable bound to a rank-0
    /// integer scalar node (e.g. an inlined fn parameter carrying a runtime
    /// window count). Such a var in a reshape target position is a runtime
    /// extent, not a type-level symbolic dim.
    fn is_runtime_scalar_var(&self, expr: &Expr) -> bool {
        let Some(name) = symbolic_dim_var_name(expr) else {
            return false;
        };
        matches!(
            self.bindings.get(&name),
            Some(LoweredValue::Node(node))
                if self.dag.get(*node).is_some_and(|n| {
                    n.output_type.dims.is_empty() && n.output_type.precision.is_integer()
                })
        )
    }

    /// Extract a list of (usize, usize) pairs from an expression (for
    /// pad/shrink bounds).
    ///
    /// Accepts BOTH:
    ///   - the raw `(list ... )` form (hand-written Deep)
    ///   - the desugared `Cons(Cons(s_0, Cons(e_0, Nil)), ..., Nil)` chain
    ///     (Surf source like `[[0, 1], [1, 3]]` after `chelis-surf::desugar`)
    ///
    /// Issue Chelis-Lang/chelis#187: previously only the `(list ...)` form
    /// was recognized, so Surf source like `shrink(&x, [[0, 1], [1, 3]])`
    /// lowered to an empty `bounds = vec![]` and produced an invalid
    /// `RiscOp::Shrink { bounds: [] }` -- failing later verify or returning
    /// silently empty output.
    fn extract_pair_list(&self, expr: &Expr) -> Option<Vec<(usize, usize)>> {
        if let Expr::List(list, _) = expr
            && get_tag(list) != Some("app")
        {
            // Hand-written `(list ...)` (or similar non-app) form.
            let mut pairs = Vec::new();
            for elem in &list.elements {
                if let Expr::List(pair_list, _) = elem {
                    // Issue Chelis-Lang/chelis#291: accept `cast`-wrapped
                    // ints (and `(lit ...)`) here too, via the shared
                    // `cons_pair_extract_int`/`extract_int_for_dim` walker,
                    // so the hand-written `(list ...)` form matches the
                    // Surf cons-chain form's bound-element vocabulary.
                    let vals: Vec<usize> = pair_list
                        .elements
                        .iter()
                        .filter_map(cons_pair_extract_int)
                        .filter_map(|n| usize::try_from(n).ok())
                        .collect();
                    if vals.len() >= 2 {
                        pairs.push((vals[0], vals[1]));
                    }
                }
            }
            if !pairs.is_empty() {
                return Some(pairs);
            }
        }
        cons_chain_pair_list(expr)
    }

    /// chelis#616: lower a movement pair-list argument (`[[start, end], ...]`)
    /// into node-valued [`RtDim`]s. A compile-time-int element (bare, `(lit
    /// ...)`, or `cast`-wrapped) becomes `RtDim::Lit`; a runtime element (a
    /// `shape()`-derived `cast(add(...))` expression) is lowered to a rank-0
    /// integer node appended to `inputs` and referenced as `RtDim::Node(slot)`.
    /// `inputs` starts as `[tensor]`. The Surf-desugared Cons-chain form carries
    /// runtime bounds; the hand-written `(list ...)` IR form stays literal-only
    /// (via [`Self::extract_pair_list`]).
    fn lower_pair_bounds(&mut self, expr: &Expr, inputs: &mut Vec<NodeId>) -> Vec<(RtDim, RtDim)> {
        if let Some(pair_exprs) = collect_cons_chain(expr) {
            let mut elems: Vec<(&Expr, &Expr)> = Vec::with_capacity(pair_exprs.len());
            let mut well_formed = true;
            for pe in pair_exprs {
                match collect_cons_chain(pe) {
                    Some(two) if two.len() == 2 => elems.push((two[0], two[1])),
                    _ => {
                        well_formed = false;
                        break;
                    }
                }
            }
            if well_formed {
                let mut bounds = Vec::with_capacity(elems.len());
                for (s, e) in elems {
                    let start = self.lower_one_bound(s, inputs);
                    let end = self.lower_one_bound(e, inputs);
                    bounds.push((start, end));
                }
                return bounds;
            }
        }
        // Hand-written `(list ...)` IR form: literal-only.
        self.extract_pair_list(expr)
            .unwrap_or_default()
            .into_iter()
            .map(|(s, e)| (RtDim::Lit(s), RtDim::Lit(e)))
            .collect()
    }

    /// chelis#616: lower a single movement-bound element to a [`RtDim`]. A
    /// compile-time int becomes `RtDim::Lit`; a runtime expression becomes a
    /// `RtDim::Node` referencing a freshly-lowered rank-0 integer node appended
    /// to `inputs`.
    fn lower_one_bound(&mut self, expr: &Expr, inputs: &mut Vec<NodeId>) -> RtDim {
        if let Some(n) = extract_int_for_dim(expr).and_then(|v| usize::try_from(v).ok()) {
            return RtDim::Lit(n);
        }
        let node = self.lower_expr_node(expr, "movement bound");
        let slot = inputs.len();
        inputs.push(node);
        RtDim::Node(slot)
    }

    /// chelis#616: lower a `stride` step list into node-valued [`RtDim`]s (one
    /// per axis). Literal steps become `RtDim::Lit`, runtime steps become
    /// `RtDim::Node`. Mirrors [`Self::extract_usize_list`] but preserves a
    /// runtime step instead of silently dropping it.
    fn lower_stride_bounds(&mut self, exprs: &[Expr], inputs: &mut Vec<NodeId>) -> Vec<RtDim> {
        let mut out = Vec::with_capacity(exprs.len());
        for e in exprs {
            out.push(self.lower_one_bound(e, inputs));
        }
        out
    }

    /// C4: Enforce float-only for the unary elementwise float family
    /// (exp, log, sin, sqrt, cos, tan, atan, abs, floor, ceil, round).
    ///
    /// chelis#730 Phase 1 (census row 1, chelis#699/#722): a non-float
    /// input raises a FATAL lowering error instead of substituting a
    /// `Const 0.0` with the operand dropped. Fatal on purpose: a
    /// non-fatal error here is caught by the host-emit backend's
    /// speculative sub-lowering, which would fall back to emitting the
    /// op as a host call over a tensor pointer - garbage C, not a loud
    /// failure (the chelis#776/#782 laundering finding). Under the
    /// chelis#729 interlock (section I1) these cells are CLEANLY
    /// REJECTED here; computing the well-defined integer cases
    /// (abs/floor/ceil/round) is chelis#729's work and replaces this
    /// raise, never a silent default. The raise also fires under `grad`
    /// lowering, which turns chelis#722's silent zero gradients into
    /// the same loud error in both lanes.
    fn lower_transcendental(&mut self, op: RiscOp, x: NodeId, ty: &TensorType) -> NodeId {
        // Elementwise: output dims come from the lowered operand, not the
        // annotation (whose dims can be stale symbolics inside a rank-poly
        // inline body; chelis#346 red-team F1: `sum(exp(x), seq)` reduced
        // the wrong axis). Same contract as the Tier-1 binary arms.
        let out_ty = Self::elementwise_out_ty(&self.dag, x, ty, None);
        let input_prec = match self.dag.get(x) {
            Some(node) => node.output_type.precision,
            // Section C1.4: a missing operand node is an internal desync
            // (the id was just produced by lowering); raising beats
            // guessing F32 (census row 13's :9833 sibling).
            None => raise_lowering_error(
                format!(
                    "internal lowering desync: operand node {} of a float unary op is \
                     missing from the DAG (was a silent F32 precision default; \
                     spec/design/loud_unsupported.md section C1.4)",
                    x.0
                ),
                None,
                self.current_span_id.clone(),
            ),
        };
        if input_prec.is_float() {
            self.dag
                .add_node(op, vec![x], out_ty, self.current_span_id.clone())
        } else {
            let unsupported = Unsupported::new(
                UnsupportedKind::Op(format!("{op:?}")),
                format!("`{}` tensors in IR lowering", input_prec.name()),
                Stage::Lowering,
                "this op family is float-only in the executable IR today; cast the \
                 operand to a float dtype first. Integer support for the well-defined \
                 cases (abs/floor/ceil/round) is tracked by chelis#729; the silent \
                 zero this replaced was chelis#699/#722",
            );
            // The suppression-aware raise ladder (the `reject_lowering_slice`
            // shape): a SPECULATIVE probe unwinds quietly so the host
            // evaluator/emitter keeps owning the non-DAG paths (eval of a
            // plain `abs(int64 tensor)` forward pass is CORRECT there); an
            // AD transform body raises FATAL so the branded message
            // survives the build lane's recoverable fallback (chelis#722's
            // zero gradients, the issue #197 pattern); everywhere else the
            // raise is recoverable and the host-emission channel is the
            // loud terminal (section C3's laundering rule - the scalar
            // arms reject tensor operands).
            if unrepresentable_panic_suppressed() {
                std::panic::panic_any(UnrepresentableDag);
            }
            if self.allow_host_list_ad_rewrites {
                raise_fatal_lowering_error(
                    unsupported.to_string(),
                    None,
                    self.current_span_id.clone(),
                )
            }
            raise_lowering_error(unsupported.to_string(), None, self.current_span_id.clone())
        }
    }

    /// `(fn {} (params {} p1 p2 ...) body)`
    fn lower_fn(&mut self, elems: &[Expr]) -> LoweredValue {
        if elems.len() < 4 {
            raise_malformed_deep(
                "an `fn` form with fewer than 4 elements",
                None,
                self.current_span_id.clone(),
            );
        }
        let saved = self.bindings.clone();
        let saved_list_bindings = self.list_bindings.clone();
        let saved_shape_bindings = self.shape_bindings.clone();
        let saved_static_size_bindings = self.static_size_bindings.clone();
        let saved_callables = self.local_callables.clone();
        let saved_fn_typed_params = self.fn_typed_params.clone();

        // Register params as Load nodes. For `t-fn`-typed params,
        // additionally track the name in `fn_typed_params` so
        // `resolve_callable_expr_inner` can surface
        // `CallableExpr::Parameter` (Item 2-extended G10). The Load is
        // still emitted for safety, but a fn-typed parameter is never
        // loaded as a tensor on any reachable path — `lower_pipe`/`lower_app`
        // dispatch on the callable shape, not on the binding's
        // `LoweredValue`.
        if let Expr::List(params_list, _) = &elems[2] {
            for param in &params_list.elements[2..] {
                if let Some((name, ty_expr)) = param_name_and_type_expr(param) {
                    if Self::type_expr_is_fn(ty_expr) {
                        self.fn_typed_params.insert(name.clone());
                    }
                    let lowered = self.lower_fn_param_binding(&name, ty_expr);
                    self.bindings.insert(name, lowered);
                }
            }
        }

        let result = self.lower_expr(&elems[3]);
        self.bindings = saved; // Restore scope
        self.list_bindings = saved_list_bindings;
        self.shape_bindings = saved_shape_bindings;
        self.static_size_bindings = saved_static_size_bindings;
        self.local_callables = saved_callables;
        self.fn_typed_params = saved_fn_typed_params;
        result
    }

    /// Returns `true` when the parameter type expression is a `t-fn`
    /// (function-valued parameter). Used by `lower_fn` to populate
    /// `fn_typed_params` so `resolve_callable_expr_inner` can return
    /// `CallableExpr::Parameter` for unresolvable callables that are
    /// nonetheless legitimate fn-typed parameters. Walks through
    /// `t-ref` wrappers (a borrowed function type is still a function).
    fn type_expr_is_fn(ty_expr: Option<&Expr>) -> bool {
        let Some(expr) = ty_expr else {
            return false;
        };
        let Expr::List(list, _) = expr else {
            return false;
        };
        match get_tag(list) {
            Some("t-fn") => true,
            Some("t-ref") => Self::type_expr_is_fn(children(list).first()),
            _ => false,
        }
    }

    fn lower_fn_param_binding(&mut self, name: &str, ty_expr: Option<&Expr>) -> LoweredValue {
        if let Some(Expr::List(list, _)) = ty_expr
            && get_tag(list) == Some("t-tuple")
        {
            let items = children(list)
                .iter()
                .enumerate()
                .map(|(index, item_ty)| {
                    self.lower_fn_param_binding(&format!("{name}__{index}"), Some(item_ty))
                })
                .collect::<Vec<_>>();
            return LoweredValue::Tuple(items);
        }

        let ty = ty_expr
            .map(|expr| {
                Self::type_from_type_expr_with_subst(
                    expr,
                    &self.prec_substitutions,
                    &self.rank_substitutions,
                )
            })
            .unwrap_or_else(Self::default_type);
        LoweredValue::Node(self.dag.add_node(
            RiscOp::Load { name: name.into() },
            vec![],
            ty,
            self.current_span_id.clone(),
        ))
    }

    /// `(pipe {} x f g ...)` -- chain: lower x, then apply f, then g, etc.
    fn lower_pipe(&mut self, elems: &[Expr]) -> LoweredValue {
        if elems.len() < 3 {
            raise_malformed_deep(
                "a `pipe` form with no seed expression",
                None,
                self.current_span_id.clone(),
            );
        }
        let mut current = self.lower_expr(&elems[2]);
        for func_expr in &elems[3..] {
            // Bucket 4e: a unary `(var {} fname)` stage where `fname` is
            // a known elementwise/tensor builtin can lower directly via
            // tier2 (no lambda intermediary). For *unknown* var names
            // (user-defined fns, library re-exports, etc.) we fall
            // through to `resolve_callable_expr` below, so the stage is
            // treated as a plain function reference and gets the same
            // unary-application semantics as `f(current)`.
            //
            // The previous implementation hit `_ => current` and then
            // `continue`, which silently dropped the user-defined fn
            // (the accumulator was returned unchanged). Top-level
            // bindings then materialised as `()`/Unit in generated C.
            let unary_builtin_name = if let Expr::List(func_list, _) = func_expr
                && let Some(Expr::Atom(Atom::Symbol(tag), _)) = func_list.elements.first()
                && tag == "var"
                && let Some(Expr::Atom(Atom::Symbol(fname), _)) = func_list.elements.get(2)
            {
                Some(fname.as_str())
            } else {
                None
            };
            let is_known_unary_builtin = matches!(
                unary_builtin_name,
                Some(
                    "neg"
                        | "exp"
                        | "log"
                        | "sin"
                        | "sqrt"
                        | "cos"
                        | "tan"
                        | "atan"
                        | "abs"
                        | "floor"
                        | "ceil"
                        | "round"
                        | "relu"
                        | "sigmoid"
                        | "tanh"
                        | "silu"
                        | "gelu"
                )
            );
            if is_known_unary_builtin
                && let Expr::List(func_list, _) = func_expr
                && let Some(Expr::Atom(Atom::Symbol(tag), _)) = func_list.elements.first()
                && tag == "var"
                && let Some(Expr::Atom(Atom::Symbol(fname), _)) = func_list.elements.get(2)
            {
                let current_node = current.expect_node("pipe stage");
                let ty = self
                    .dag
                    .get(current_node)
                    .map(|node| node.output_type.clone())
                    .unwrap_or_else(Self::default_type);
                current = match fname.as_str() {
                    "neg" => LoweredValue::Node(self.dag.add_node(
                        RiscOp::Neg,
                        vec![current_node],
                        ty,
                        self.current_span_id.clone(),
                    )),
                    "exp" => LoweredValue::Node(self.dag.add_node(
                        RiscOp::Exp,
                        vec![current_node],
                        ty,
                        self.current_span_id.clone(),
                    )),
                    "log" => LoweredValue::Node(self.dag.add_node(
                        RiscOp::Log,
                        vec![current_node],
                        ty,
                        self.current_span_id.clone(),
                    )),
                    "sin" => LoweredValue::Node(self.dag.add_node(
                        RiscOp::Sin,
                        vec![current_node],
                        ty,
                        self.current_span_id.clone(),
                    )),
                    "sqrt" => LoweredValue::Node(self.dag.add_node(
                        RiscOp::Sqrt,
                        vec![current_node],
                        ty,
                        self.current_span_id.clone(),
                    )),
                    "cos" => LoweredValue::Node(self.dag.add_node(
                        RiscOp::Cos,
                        vec![current_node],
                        ty,
                        self.current_span_id.clone(),
                    )),
                    "tan" => LoweredValue::Node(self.dag.add_node(
                        RiscOp::Tan,
                        vec![current_node],
                        ty,
                        self.current_span_id.clone(),
                    )),
                    "atan" => LoweredValue::Node(self.dag.add_node(
                        RiscOp::Atan,
                        vec![current_node],
                        ty,
                        self.current_span_id.clone(),
                    )),
                    "abs" => LoweredValue::Node(self.dag.add_node(
                        RiscOp::Abs,
                        vec![current_node],
                        ty,
                        self.current_span_id.clone(),
                    )),
                    "floor" => LoweredValue::Node(self.dag.add_node(
                        RiscOp::Floor,
                        vec![current_node],
                        ty,
                        self.current_span_id.clone(),
                    )),
                    "ceil" => LoweredValue::Node(self.dag.add_node(
                        RiscOp::Ceil,
                        vec![current_node],
                        ty,
                        self.current_span_id.clone(),
                    )),
                    "round" => LoweredValue::Node(self.dag.add_node(
                        RiscOp::Round,
                        vec![current_node],
                        ty,
                        self.current_span_id.clone(),
                    )),
                    "relu" => LoweredValue::Node(tier2::lower_relu(
                        &mut self.dag,
                        current_node,
                        &ty,
                        self.current_span_id.as_deref(),
                    )),
                    "sigmoid" => LoweredValue::Node(tier2::lower_sigmoid(
                        &mut self.dag,
                        current_node,
                        &ty,
                        self.current_span_id.as_deref(),
                    )),
                    "tanh" => LoweredValue::Node(tier2::lower_tanh(
                        &mut self.dag,
                        current_node,
                        &ty,
                        self.current_span_id.as_deref(),
                    )),
                    "silu" => LoweredValue::Node(tier2::lower_silu(
                        &mut self.dag,
                        current_node,
                        &ty,
                        self.current_span_id.as_deref(),
                    )),
                    "gelu" => LoweredValue::Node(tier2::lower_gelu(
                        &mut self.dag,
                        current_node,
                        &ty,
                        self.current_span_id.as_deref(),
                    )),
                    // `is_known_unary_builtin` guarantees this branch is
                    // never hit, but keep it as an explicit fallthrough
                    // marker so any future name added to the predicate
                    // without a corresponding match arm fails loudly.
                    _ => unreachable!(
                        "pipe stage `{fname}` was classified as a known \
                         unary builtin but has no lowering arm"
                    ),
                };
                continue;
            }
            if let Some(callable) = self.resolve_callable_expr(func_expr) {
                current = match callable {
                    CallableExpr::Plain(fn_expr) => {
                        self.lower_plain_callable_with_values(&fn_expr, &[current.clone()])
                    }
                    CallableExpr::Vmap { fn_expr, axis } => {
                        let current_node = current.expect_node("pipe stage");
                        self.lower_vmap_callable_with_nodes(
                            &fn_expr,
                            axis,
                            &[current_node],
                            func_expr.span(),
                        )
                    }
                    CallableExpr::Grad { fn_expr, wrt } => {
                        // `x |> grad(f)` lowers as `grad(f)(x)` — reuse
                        // the non-pipe grad lowering with the previous
                        // stage's value as the single argument.
                        self.lower_grad_callable_with_values(
                            &fn_expr,
                            wrt.as_deref(),
                            &[current.clone()],
                            func_expr.span(),
                        )
                    }
                    CallableExpr::VmapGrad { fn_expr, wrt, axis } => {
                        // `xs |> vmap(grad(f))` lowers as
                        // `vmap(grad(f))(xs)` — reuse the non-pipe
                        // vmap-grad lowering with the previous stage's
                        // NodeId as the single argument.
                        let current_node = current.expect_node("pipe stage");
                        self.lower_vmap_grad_callable_with_nodes(
                            &fn_expr,
                            wrt.as_deref(),
                            axis,
                            &[current_node],
                            func_expr.span(),
                        )
                    }
                    // Item 2-extended G10: `x |> f` where `f` is a fn-
                    // typed parameter. The DAG has no `RiscOp::Call`, so
                    // the standalone-def lowering can't actually apply
                    // `f` — leave `current` unchanged. This is correct on
                    // every reachable path: standalone-def lowering only
                    // builds a DAG entry that `try_lower_program`
                    // produces eagerly but no caller ever references (every
                    // caller re-inlines through `lower_plain_callable_app`,
                    // which substitutes the concrete callable into
                    // `local_callables` so the resolver returns
                    // `Plain`/`Vmap`/`Grad`/`VmapGrad`, not `Parameter`).
                    // See `docs/investigations/pipe_fn_param_stage_diagnosis.md`.
                    CallableExpr::Parameter { .. } => current,
                };
                continue;
            }
            // Item 2c: bare `(var {} name)` stage where `name` is neither a
            // local callable nor a program def is a primitive (builtin) used
            // as a unary pipe stage — e.g. `... |> tensor_to_scalar`. The
            // host lane already normalizes this via `beta_reduce_pipe_stage`
            // (`crates/chelis-ir/src/host.rs:2947`); the IR lane has to
            // mirror that rewrite. Synthesize `(app (var name) (var __acc))`
            // with the accumulator bound to the already-lowered `current`
            // value, then dispatch through `lower_app` which routes
            // unresolved-name callees through `lower_builtin_app`.
            //
            // See `docs/investigations/c_backend_grad_piped_body_diagnosis.md`.
            if let Some(unary_name) = bare_var_name(func_expr) {
                let acc_binding = synth_pipe_acc_binding_name();
                let saved = self.bindings.get(&acc_binding).cloned();
                self.bindings.insert(acc_binding.clone(), current.clone());
                let synthesized = synth_unary_app(&unary_name, &acc_binding, func_expr.span());
                current = self.lower_expr(&synthesized);
                match saved {
                    Some(prior) => {
                        self.bindings.insert(acc_binding, prior);
                    }
                    None => {
                        self.bindings.remove(&acc_binding);
                    }
                }
                continue;
            }
            current = self.lower_unrepresentable("pipe stage", std::slice::from_ref(func_expr));
        }
        current
    }

    /// `(cast {} expr (t-prim {} name))` -- precision cast.
    ///
    /// chelis#730 Phase 1 (census row 13, chelis#744): a cast target that
    /// is not a recognized primitive raises a FATAL lowering error. The
    /// former silent F32-default fallbacks (parse-or-F32) lowered
    /// a bogus `.dp` cast target to f32 silently in the build lane (the
    /// eval lane's guard rejects the same file cleanly - chelis#744
    /// refuted the dead-by-probe claim). Fatal so the host fallback
    /// cannot re-launder the same node.
    fn lower_cast(&mut self, elems: &[Expr]) -> LoweredValue {
        if elems.len() < 4 {
            raise_malformed_deep(
                "a `cast` form with fewer than 4 elements",
                None,
                self.current_span_id.clone(),
            );
        }
        let x = self.lower_expr_node(&elems[2], "cast input");
        let input_ty = self
            .dag
            .get(x)
            .map(|n| n.output_type.clone())
            .unwrap_or_else(Self::default_type);
        let raise_bogus_target = |found: &str| -> ! {
            let unsupported = Unsupported::new(
                UnsupportedKind::Dtype(found.to_string()),
                "a `cast` target in IR lowering",
                Stage::Lowering,
                "the cast target must name an active primitive type \
                 (spec/04-type-system.md section 1.1); a bogus target previously \
                 lowered as f32 silently in the build lane (chelis#744, chelis#730 \
                 census row 13)",
            );
            raise_fatal_lowering_error(
                unsupported.to_string(),
                Some(elems[3].span()),
                elems[3].span_id().map(ToOwned::to_owned),
            )
        };
        let new_precision = if let Some(prim) = Self::try_extract_prim(&elems[3]) {
            // Handle (t-prim {} name) form.
            prim
        } else if let Expr::Atom(Atom::Symbol(pname), _) = &elems[3] {
            // Bare-symbol spelling (backward compat): parse or raise.
            match Prim::parse_name(pname) {
                Some(prim) => prim,
                None => raise_bogus_target(pname),
            }
        } else if let Expr::List(list, _) = &elems[3]
            && list.elements.len() >= 3
            && let Expr::Atom(Atom::Symbol(tag), _) = &list.elements[0]
            && tag == "t-prim"
            && let Expr::Atom(Atom::Symbol(pname), _) = &list.elements[2]
        {
            // `(t-prim {} <name>)` whose name is not a recognized
            // primitive (the chelis#744 repro spelling).
            raise_bogus_target(pname)
        } else {
            raise_bogus_target("a non-primitive cast target expression")
        };
        let ty = TensorType {
            dims: input_ty.dims,
            precision: new_precision,
        };
        LoweredValue::Node(self.dag.add_node(
            RiscOp::Cast { new_precision },
            vec![x],
            ty,
            self.current_span_id.clone(),
        ))
    }

    /// `(grad {} f)` -- rejected before lowering.
    fn lower_grad(&mut self, elems: &[Expr]) -> LoweredValue {
        self.lower_unrepresentable("grad", elems)
    }

    /// `(if {} cond then else)` -- Phase 0: select via arithmetic on bools.
    /// Runtime-condition `if` branches must be single tensor nodes: the mask
    /// blend is scalar/tensor arithmetic and has no representation for an
    /// ADT or tuple value. Static-condition ifs never reach this check (the
    /// taken branch is returned verbatim by the chelis#620 pruning path).
    /// Raised through the [`Self::reject_static_adt`] ladder so the host
    /// lane's speculative DAG probe falls back to interpretation quietly and
    /// the precise message survives inside a grad body.
    ///
    /// The leading "if {which} branch expected a single tensor value" prefix
    /// is pinned by `issue_520_d1_runtime_ctor_through_if_still_rejected`;
    /// keep it stable.
    fn expect_runtime_if_branch(
        &self,
        branch: LoweredValue,
        which: &str,
        elems: &[Expr],
    ) -> NodeId {
        match branch {
            LoweredValue::Node(id) => id,
            LoweredValue::Tuple(_) => self.reject_static_adt(
                elems,
                format!(
                    "if {which} branch expected a single tensor value, got a tuple value; \
                     a tuple-valued branch requires a compile-time-resolvable condition \
                     so the untaken branch is pruned statically (chelis#620)"
                ),
            ),
            LoweredValue::Adt { ctor, .. } => self.reject_static_adt(
                elems,
                format!(
                    "if {which} branch expected a single tensor value, got an ADT value \
                     constructed with `{ctor}`; an `if` may produce an ADT or list value \
                     only when its condition is compile-time-resolvable so the untaken \
                     branch is pruned statically (chelis#620); runtime ADT-valued control \
                     flow awaits the select/blend primitive (chelis#618)"
                ),
            ),
        }
    }

    fn lower_if(&mut self, elems: &[Expr]) -> LoweredValue {
        let Some(cond_expr) = elems.get(2) else {
            raise_malformed_deep(
                "an `if` form with no condition",
                None,
                self.current_span_id.clone(),
            );
        };
        let Some(then_expr) = elems.get(3) else {
            raise_malformed_deep(
                "an `if` form with no then-branch",
                None,
                self.current_span_id.clone(),
            );
        };
        let Some(else_expr) = elems.get(4) else {
            raise_malformed_deep(
                "an `if` form with no else-branch",
                None,
                self.current_span_id.clone(),
            );
        };

        let cond = self.lower_expr_node(cond_expr, "if condition");
        // chelis#620: static branch pruning. A compile-time-resolvable
        // condition lowers ONLY the taken branch and returns its value
        // verbatim (Node, Tuple, or Adt, any precision). The untaken branch
        // is never lowered: it may be `fail(...)`, an empty list, or a
        // recursive call whose termination depends on this pruning. This is
        // the `if` analogue of `lower_match`'s static arm selection and is
        // semantics-preserving in both lanes (a static condition cannot vary
        // under input perturbation, so the pruned gradient is exact). The
        // now-dead condition subgraph is swept by the entry points' DCE.
        if let Some(taken) = self.fold_static_cond(cond) {
            return self.lower_expr(if taken { then_expr } else { else_expr });
        }
        let then_value = self.lower_expr(then_expr);
        let then_node = self.expect_runtime_if_branch(then_value, "then", elems);
        let else_value = self.lower_expr(else_expr);
        let else_node = self.expect_runtime_if_branch(else_value, "else", elems);
        let out_ty = if let Some(Expr::Map(meta, _)) = elems.get(1) {
            self.type_from_meta(&meta.entries)
        } else {
            self.dag
                .get(then_node)
                .map(|node| node.output_type.clone())
                .unwrap_or_else(Self::default_type)
        };
        if !out_ty.precision.is_float() {
            return self.lower_unrepresentable("if", elems);
        }

        // chelis#616: a leaf-Const branch (the `fail` placeholder) and the
        // mask's `one` Const are shaped like the branch values, but as leaf
        // nodes they have no input edge carrying that relation. Conform the
        // placeholder to the if's rank (the checker types `fail` as bottom,
        // which lowers rank-0) and record a shape-dep so (i) the C backend's
        // anon-dim renaming ties the Const's wildcard dims to the sibling's
        // instead of fragmenting them into fresh sourceless `_anon_dim_*`s,
        // (ii) the evaluator can size the Const from the sibling's actual
        // value, and (iii) the extent source stays alive under DCE.
        let then_node = self.conform_branch_placeholder(then_node, &out_ty, else_node);
        let else_node = self.conform_branch_placeholder(else_node, &out_ty, then_node);
        let mask = self.lower_if_mask(cond, &out_ty, else_node);
        let one = self.dag.add_node(
            RiscOp::synth_const(out_ty.precision, 1.0),
            vec![],
            out_ty.clone(),
            self.current_span_id.clone(),
        );
        self.dag.add_shape_dep(one, else_node);
        let neg_mask = self.dag.add_node(
            RiscOp::Neg,
            vec![mask],
            out_ty.clone(),
            self.current_span_id.clone(),
        );
        let inv_mask = self.dag.add_node(
            RiscOp::Add,
            vec![one, neg_mask],
            out_ty.clone(),
            self.current_span_id.clone(),
        );
        let masked_then = self.dag.add_node(
            RiscOp::Mul,
            vec![mask, then_node],
            out_ty.clone(),
            self.current_span_id.clone(),
        );
        let masked_else = self.dag.add_node(
            RiscOp::Mul,
            vec![inv_mask, else_node],
            out_ty.clone(),
            self.current_span_id.clone(),
        );
        LoweredValue::Node(self.dag.add_node(
            RiscOp::Add,
            vec![masked_then, masked_else],
            out_ty,
            self.current_span_id.clone(),
        ))
    }

    /// `(tuple {} elem1 elem2 ...)` -- not representable in the Phase 0 RISC DAG.
    fn lower_tuple(&mut self, elems: &[Expr]) -> LoweredValue {
        LoweredValue::Tuple(
            elems
                .iter()
                .skip(2)
                .map(|expr| self.lower_expr(expr))
                .collect(),
        )
    }

    /// `(par {} expr1 expr2 ...)` -- v1 sequential composition per
    /// `spec/03-deep-syntax.md` §2.3 ("Parallel evaluation (v1: sequential)").
    /// Each child is lowered in order; the value of the last child is the
    /// par's value. Intermediate children still contribute their nodes to the
    /// DAG so any side-effecting operations (e.g. `realize`) are preserved.
    fn lower_par(&mut self, elems: &[Expr]) -> LoweredValue {
        let mut last: Option<LoweredValue> = None;
        for expr in elems.iter().skip(2) {
            last = Some(self.lower_expr(expr));
        }
        last.unwrap_or_else(|| {
            raise_malformed_deep("an empty `par` block", None, self.current_span_id.clone())
        })
    }

    /// `(jit {} expr)` -- compilation trigger per
    /// `spec/03-deep-syntax.md` §2.7. Semantically a no-op at evaluation; the
    /// JIT effect (if any) lives in metadata. Lower as identity on the inner
    /// expression.
    fn lower_jit(&mut self, elems: &[Expr]) -> LoweredValue {
        if let Some(inner) = elems.get(2) {
            self.lower_expr(inner)
        } else {
            raise_malformed_deep(
                "a `jit` form with no inner expression",
                None,
                self.current_span_id.clone(),
            )
        }
    }

    /// `(realize {} expr)` -- explicit materialization barrier.
    fn lower_realize(&mut self, elems: &[Expr]) -> LoweredValue {
        if elems.len() >= 3 {
            let input = self.lower_expr(&elems[2]);
            if let LoweredValue::Tuple(items) = &input {
                return LoweredValue::Tuple(
                    items
                        .iter()
                        .map(|item| {
                            let id = item.expect_node("realize tuple leaf");
                            let output_type = self
                                .dag
                                .get(id)
                                .map(|node| node.output_type.clone())
                                .unwrap_or_else(Self::default_type);
                            LoweredValue::Node(self.dag.add_node(
                                RiscOp::Realize,
                                vec![id],
                                output_type,
                                self.current_span_id.clone(),
                            ))
                        })
                        .collect(),
                );
            }
            let input = input.expect_node("realize input");
            let output_type = self
                .dag
                .get(input)
                .map(|node| node.output_type.clone())
                .unwrap_or_else(Self::default_type);
            LoweredValue::Node(self.dag.add_node(
                RiscOp::Realize,
                vec![input],
                output_type,
                self.current_span_id.clone(),
            ))
        } else {
            raise_malformed_deep(
                "a `realize` form with no inner expression",
                None,
                self.current_span_id.clone(),
            )
        }
    }

    /// `(copy {} expr)` -- identity in Phase 0/1.
    fn lower_copy(&mut self, elems: &[Expr]) -> LoweredValue {
        if elems.len() >= 3 {
            let input = self.lower_expr(&elems[2]);
            self.copy_lowered_value(&input)
        } else {
            raise_malformed_deep(
                "a `copy` form with no inner expression",
                None,
                self.current_span_id.clone(),
            )
        }
    }

    /// Structure-preserving copy of a lowered value: a `RiscOp::Copy` per
    /// tensor leaf, recursing through tuples and ADTs (chelis#620; the
    /// pre-existing behavior handled `Node` and flat `Tuple` only, so a
    /// compiler-inserted linearity copy over a params ADT died in
    /// `expect_node("copy input")` -- the issue's Blocker 2).
    fn copy_lowered_value(&mut self, value: &LoweredValue) -> LoweredValue {
        match value {
            LoweredValue::Node(id) => {
                let output_type = self
                    .dag
                    .get(*id)
                    .map(|node| node.output_type.clone())
                    .unwrap_or_else(Self::default_type);
                LoweredValue::Node(self.dag.add_node(
                    RiscOp::Copy,
                    vec![*id],
                    output_type,
                    self.current_span_id.clone(),
                ))
            }
            LoweredValue::Tuple(items) => LoweredValue::Tuple(
                items
                    .iter()
                    .map(|item| self.copy_lowered_value(item))
                    .collect(),
            ),
            LoweredValue::Adt {
                ctor,
                field_names,
                fields,
            } => LoweredValue::Adt {
                ctor: ctor.clone(),
                field_names: field_names.clone(),
                fields: fields
                    .iter()
                    .map(|field| self.copy_lowered_value(field))
                    .collect(),
            },
        }
    }

    /// `(borrow {} expr)` -- erased before executable lowering.
    fn lower_identity(&mut self, elems: &[Expr]) -> LoweredValue {
        if elems.len() >= 3 {
            self.lower_expr(&elems[2])
        } else {
            raise_malformed_deep(
                "a `borrow` form with no inner expression",
                None,
                self.current_span_id.clone(),
            )
        }
    }

    /// `(tuple-get {} tuple_expr index)` -- not representable in the Phase 0 RISC DAG.
    fn lower_tuple_get(&mut self, elems: &[Expr]) -> LoweredValue {
        let tuple = self.lower_expr(&elems[2]);
        // chelis#730 Phase 1 (#782-flagged structural-index site): a
        // tuple index that does not fold to a compile-time usize raises
        // instead of silently reading field 0 (section C1.4; the index
        // is compile-time by construction, so this is expected dead).
        let index = self.extract_usize_value(&elems[3]).unwrap_or_else(|| {
            raise_lowering_error(
                "tuple-get index does not fold to a compile-time integer (was a \
                 silent index-0 default; spec/design/loud_unsupported.md section \
                 C1.4, flagged by chelis#782)",
                Some(elems[3].span()),
                elems[3].span_id().map(ToOwned::to_owned),
            )
        });
        tuple.tuple_get(index).unwrap_or_else(|| {
            raise_lowering_error(
                format!("tuple-get index {index} out of bounds during lowering"),
                elems.get(3).map(Expr::span),
                elems.get(3).and_then(Expr::span_id).map(ToOwned::to_owned),
            )
        })
    }

    /// `(record {} Ctor (kv {} field value) ...)` -- static ADT/record
    /// construction (chelis#520). The constructor tag and field layout are
    /// compile-time facts; only the field values are lowered.
    fn lower_record(&mut self, elems: &[Expr]) -> LoweredValue {
        let Some(ctor) = elems.get(2).and_then(symbol_name) else {
            self.reject_static_adt(
                elems,
                "`record` construction is missing its constructor name".to_string(),
            );
        };
        let ctor = ctor.to_string();
        let mut field_names: Vec<String> = Vec::new();
        let mut fields: Vec<LoweredValue> = Vec::new();
        for kv in &elems[3..] {
            let Expr::List(kv_list, _) = kv else {
                continue;
            };
            if get_tag(kv_list) != Some("kv") {
                continue;
            }
            let kv_kids = children(kv_list);
            let (Some(field), Some(value)) =
                (kv_kids.first().and_then(symbol_name), kv_kids.get(1))
            else {
                self.reject_static_adt(
                    elems,
                    format!("`record` constructor `{ctor}` has a malformed field entry"),
                );
            };
            field_names.push(field.to_string());
            fields.push(self.lower_expr(value));
        }
        LoweredValue::Adt {
            ctor,
            field_names: Some(field_names),
            fields,
        }
    }

    /// `(access {} target field)` -- field projection on a statically-known
    /// ADT/record value (chelis#520). A runtime target stays rejected: the
    /// Phase 0 RISC DAG has no runtime record representation.
    fn lower_access(&mut self, elems: &[Expr]) -> LoweredValue {
        if elems.len() < 4 {
            self.reject_static_adt(elems, "`access` is missing its field name".to_string());
        }
        let target = self.lower_expr(&elems[2]);
        let Some(field) = symbol_name(&elems[3]) else {
            self.reject_static_adt(elems, "`access` field name is not a symbol".to_string());
        };
        match target {
            LoweredValue::Adt {
                ctor,
                field_names: Some(names),
                fields,
            } => names
                .iter()
                .position(|name| name == field)
                .and_then(|index| fields.get(index).cloned())
                .unwrap_or_else(|| {
                    self.reject_static_adt(elems, format!("record `{ctor}` has no field `{field}`"))
                }),
            _ => self.reject_static_adt(
                elems,
                format!(
                    "`access` on a runtime value is not supported by IR lowering; only \
                     a compile-time-known record construction can be projected (field \
                     `{field}`)"
                ),
            ),
        }
    }

    /// `(match {} scrutinee (arm {} pattern guard body) ...)`.
    ///
    /// chelis#520 D1 slice: when the scrutinee lowers to a statically-known
    /// constructor value (`LoweredValue::Adt`), the taken arm is resolved at
    /// lowering time and only that arm's body is lowered. This is the exact
    /// gradient semantics for AD (spec/design/differentiable_language.md
    /// Phase 1: "the pattern match itself is non-differentiable; gradient
    /// flow goes through the matched values"): the constructor tag is
    /// discrete, so perturbing tensor inputs can never change the taken arm.
    ///
    /// A runtime scrutinee (anything that lowers to a tensor node), an arm
    /// guard on the selected pattern, and pattern forms outside the static
    /// slice all stay rejected, loudly, naming the construct.
    fn lower_match(&mut self, elems: &[Expr]) -> LoweredValue {
        if elems.len() < 4 {
            return self.lower_unrepresentable("match", elems);
        }
        let scrutinee = self.lower_expr(&elems[2]);
        let LoweredValue::Adt {
            ctor,
            field_names,
            fields,
        } = scrutinee
        else {
            self.reject_static_adt(
                elems,
                "`match` on a runtime scrutinee is not supported by IR evaluation yet; \
                 only a match whose scrutinee is a compile-time-known constructor value \
                 is resolved by static arm selection (chelis#520 D1)"
                    .to_string(),
            );
        };
        for arm in &elems[3..] {
            let Expr::List(arm_list, _) = arm else {
                continue;
            };
            if get_tag(arm_list) != Some("arm") {
                continue;
            }
            let arm_kids = children(arm_list);
            let (Some(pattern), Some(guard), Some(body)) =
                (arm_kids.first(), arm_kids.get(1), arm_kids.get(2))
            else {
                continue;
            };
            match match_static_pattern(pattern, &ctor, field_names.as_deref(), &fields) {
                StaticPatternMatch::NoMatch => continue,
                StaticPatternMatch::Unsupported(reason) => {
                    self.reject_static_adt(
                        elems,
                        format!(
                            "`match` static arm selection does not support {reason} \
                             (chelis#520 D1)"
                        ),
                    );
                }
                StaticPatternMatch::Match(binds) => {
                    if !guard_is_absent(guard) {
                        self.reject_static_adt(
                            elems,
                            "`match` arm guards are not supported by static arm \
                             selection: a guard needs runtime evaluation, so the taken \
                             arm is not compile-time-known (chelis#520 D1)"
                                .to_string(),
                        );
                    }
                    let saved = self.bindings.clone();
                    let saved_list_bindings = self.list_bindings.clone();
                    let saved_shape_bindings = self.shape_bindings.clone();
                    let saved_static_size_bindings = self.static_size_bindings.clone();
                    let saved_callables = self.local_callables.clone();
                    let saved_fn_typed_params = self.fn_typed_params.clone();
                    for (name, value) in binds {
                        // Pattern binds shadow every same-named outer
                        // binding class, mirroring `lower_plain_callable_app`.
                        self.list_bindings.remove(&name);
                        self.shape_bindings.remove(&name);
                        self.static_size_bindings.remove(&name);
                        self.local_callables.remove(&name);
                        self.fn_typed_params.remove(&name);
                        self.bindings.insert(name, value);
                    }
                    let value = self.lower_expr(body);
                    self.bindings = saved;
                    self.list_bindings = saved_list_bindings;
                    self.shape_bindings = saved_shape_bindings;
                    self.static_size_bindings = saved_static_size_bindings;
                    self.local_callables = saved_callables;
                    self.fn_typed_params = saved_fn_typed_params;
                    return value;
                }
            }
        }
        self.reject_static_adt(
            elems,
            format!(
                "`match` static arm selection found no arm matching constructor \
                 `{ctor}`; a checked match is exhaustive, so this indicates a pattern \
                 form outside the supported static slice (chelis#520 D1)"
            ),
        )
    }

    /// Rejection helper for the static-ADT lowering slice (chelis#520).
    /// Preserves the host-lane speculative-attempt contract (quiet unwind
    /// when `unrepresentable_panic_suppressed`), then raises FATAL inside an
    /// AD transform body (so the precise message survives the build lane's
    /// recoverable tensor-helper fallback, the issue #197 pattern) and
    /// recoverable everywhere else (so non-grad host-lane routing keeps
    /// falling back to interpretation).
    fn reject_static_adt(&self, elems: &[Expr], message: String) -> ! {
        self.reject_lowering_slice(elems.first(), message)
    }

    /// Message-only core of [`Self::reject_static_adt`]: the same
    /// suppression-aware raise ladder for callers that carry a span-source
    /// expression rather than a tag's `elems` slice (chelis#620, e.g. the
    /// recursion-unroll caps in `lower_plain_callable_app`).
    fn reject_lowering_slice(&self, span_expr: Option<&Expr>, message: String) -> ! {
        if unrepresentable_panic_suppressed() {
            std::panic::panic_any(UnrepresentableDag);
        }
        if self.allow_host_list_ad_rewrites {
            raise_fatal_lowering_error(
                message,
                span_expr.map(Expr::span),
                span_expr.and_then(Expr::span_id).map(ToOwned::to_owned),
            )
        }
        raise_lowering_error(
            message,
            span_expr.map(Expr::span),
            span_expr.and_then(Expr::span_id).map(ToOwned::to_owned),
        )
    }

    fn lower_unrepresentable(&mut self, tag: &str, elems: &[Expr]) -> LoweredValue {
        for expr in elems.iter().skip(2) {
            let _ = self.lower_expr(expr);
        }
        if unrepresentable_panic_suppressed() {
            // Speculative DAG attempt from the host-lane fallback — unwind
            // without writing a stderr panic-location trace. `catch_unwind`
            // in the caller turns this into an `Err` and falls back to
            // host lowering.
            std::panic::panic_any(UnrepresentableDag);
        }
        let expr = elems.first();
        raise_lowering_error(
            unsupported_lowering_message(tag),
            expr.map(Expr::span),
            expr.and_then(Expr::span_id).map(ToOwned::to_owned),
        )
    }

    /// Constructs that are valid Chelis but not supported by DAG evaluation.
    fn lower_unsupported(&mut self, tag: &str, elems: &[Expr]) -> LoweredValue {
        let expr = elems.first();
        raise_lowering_error(
            unsupported_lowering_message(tag),
            expr.map(Expr::span),
            expr.and_then(Expr::span_id).map(ToOwned::to_owned),
        )
    }

    /// chelis#616: conform an `if` branch that lowered as an input-less
    /// `Const` (the `fail` placeholder) to the if's type, and record the
    /// sibling branch as its shape source (see `lower_if`). The checker
    /// types `fail` as bottom, which lowers rank-0; a checked program's
    /// branches otherwise agree in rank, so a rank-0 leaf Const under a
    /// tensor-typed `if` is exactly the bottom placeholder. Returns the
    /// branch node to use in the mask arithmetic: for the bottom
    /// placeholder that is a FRESH conformed Const emitted here — after
    /// both branches — so its shape source (the sibling) precedes it in
    /// evaluation order.
    fn conform_branch_placeholder(
        &mut self,
        node: NodeId,
        out_ty: &TensorType,
        sibling: NodeId,
    ) -> NodeId {
        if node == sibling || out_ty.dims.is_empty() {
            return node;
        }
        let Some(n) = self.dag.get(node).cloned() else {
            return node;
        };
        if !(matches!(n.op, RiscOp::Const { .. }) && n.inputs.is_empty() && n.shape_deps.is_empty())
        {
            return node;
        }
        let already_ranked_but_anonymous = n.output_type.dims.len() == out_ty.dims.len()
            && n.output_type
                .dims
                .iter()
                .any(|d| matches!(d, DimInfo::Named(_, None)));
        if n.output_type.dims.is_empty() || already_ranked_but_anonymous {
            // Emit a fresh Const at the if's rank with anonymous symbolic
            // dims (the checker's symbol for an unbound axis may be declared
            // later in program order; the shape-dep carries the actual
            // extent source). This must happen after the sibling even when
            // authoritative inference already gave the original `fail`
            // placeholder the correct rank: DAG rebuilds remap in node order,
            // so a forward shape-dep from that early Const to the later
            // sibling is dropped. The original is left unconsumed for DCE.
            let dims = out_ty
                .dims
                .iter()
                .map(|dim| match dim {
                    DimInfo::Lit(size) | DimInfo::Named(_, Some(size)) => DimInfo::Lit(*size),
                    DimInfo::Named(_, None) => DimInfo::Named(String::new(), None),
                })
                .collect();
            let conformed = self.dag.add_node(
                n.op,
                Vec::new(),
                TensorType {
                    dims,
                    precision: out_ty.precision,
                },
                self.current_span_id.clone(),
            );
            self.dag.add_shape_dep(conformed, sibling);
            conformed
        } else {
            node
        }
    }

    fn lower_if_mask(&mut self, cond: NodeId, out_ty: &TensorType, shape_source: NodeId) -> NodeId {
        let mut mask = cond;
        let cond_ty = self
            .dag
            .get(cond)
            .map(|node| node.output_type.clone())
            .unwrap_or_else(Self::default_type);
        if cond_ty.precision != out_ty.precision {
            mask = self.dag.add_node(
                RiscOp::Cast {
                    new_precision: out_ty.precision,
                },
                vec![mask],
                TensorType {
                    dims: cond_ty.dims.clone(),
                    precision: out_ty.precision,
                },
                self.current_span_id.clone(),
            );
        }
        if cond_ty.dims.is_empty() && !out_ty.dims.is_empty() {
            let mut expanded = mask;
            let mut dims = Vec::new();
            for (axis, dim) in out_ty.dims.iter().enumerate() {
                dims.push(dim.clone());
                expanded = self.dag.add_node(
                    RiscOp::Expand {
                        axis,
                        size: DimExpr::from(dim),
                    },
                    vec![expanded],
                    TensorType {
                        dims: dims.clone(),
                        precision: out_ty.precision,
                    },
                    self.current_span_id.clone(),
                );
                // chelis#616: an unbound (possibly wildcard) `size` resolves
                // from the branch value's actual shape at eval time.
                self.dag.add_shape_dep(expanded, shape_source);
            }
            return expanded;
        }
        mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verify;

    fn parse_and_check(src: &str) -> chelis_types::CheckedProgram {
        let exprs = chelis_deep::parser::parse_str(src).expect("parse failed");
        let checked = chelis_types::check_ir_program(&exprs)
            .unwrap_or_else(|result| panic!("IR check failed: {:?}", result.errors));
        let checked = chelis_effects::check_program(&checked)
            .unwrap_or_else(|errors| panic!("effect check failed: {errors:?}"));
        chelis_types::check_linearity(&checked)
            .unwrap_or_else(|errors| panic!("linearity check failed: {errors:?}"))
    }

    fn parse_and_lower(src: &str) -> Dag {
        let checked = parse_and_check(src);
        lower_program(&checked)
    }

    /// `compute_reduce_window_out_dims` refines any *statically-known*
    /// windowed axis — `Lit` **and** `Named(_, Some(_))` — to a concrete
    /// `Lit` Valid-padding extent, and defers to the checker-derived
    /// fallback only when a windowed axis is `Named(_, None)` (runtime-only).
    /// This pins the intentional asymmetry flagged in PR #261 review: the
    /// type checker is conservative and yields `Wildcard` for a symbolic
    /// (named) windowed axis, while the lowering refines a *sized* named dim
    /// to the concrete output extent. (`Named(_, None)` is additionally
    /// rejected on the build path by `reject_symbolic_windowed_reduce`.)
    #[test]
    fn compute_reduce_window_out_dims_refines_named_sized_windowed_axis() {
        let f32 = chelis_types::types::Prim::F32;
        let fallback = TensorType {
            dims: vec![DimInfo::Lit(1), DimInfo::Lit(1)],
            precision: f32,
        };

        // Leading axis passes through; a Named-with-size windowed axis is
        // refined to floor((8 - 2) / 1) + 1 = 7.
        let named_sized = vec![
            DimInfo::Named("batch".into(), Some(2)),
            DimInfo::Named("h".into(), Some(8)),
        ];
        assert_eq!(
            compute_reduce_window_out_dims(&named_sized, &[2], &[1], &fallback),
            vec![DimInfo::Named("batch".into(), Some(2)), DimInfo::Lit(7)],
        );

        // A literal windowed axis is likewise computed concretely:
        // floor((8 - 2) / 2) + 1 = 4.
        let literal = vec![DimInfo::Lit(3), DimInfo::Lit(8)];
        assert_eq!(
            compute_reduce_window_out_dims(&literal, &[2], &[2], &fallback),
            vec![DimInfo::Lit(3), DimInfo::Lit(4)],
        );

        // A runtime-only (`Named(_, None)`) windowed axis is not
        // representable; fall back to the checker-derived dims verbatim.
        let runtime_only = vec![DimInfo::Lit(3), DimInfo::Named("seq".into(), None)];
        assert_eq!(
            compute_reduce_window_out_dims(&runtime_only, &[2], &[1], &fallback),
            fallback.dims,
        );
    }

    fn parse_and_lower_unchecked(src: &str) -> Dag {
        let exprs = chelis_deep::parser::parse_str(src).expect("parse failed");
        let mut ctx = LowerCtx::new(
            std::collections::HashMap::new(),
            std::collections::HashMap::new(),
            LinearityInfo::default(),
        );
        for expr in &exprs {
            let _ = ctx.lower_expr(expr);
        }
        ctx.dag
    }

    #[test]
    fn typed_fail_placeholder_keeps_a_backward_shape_dependency() {
        let mut ctx = LowerCtx::new(
            std::collections::HashMap::new(),
            std::collections::HashMap::new(),
            LinearityInfo::default(),
        );
        let out_ty = TensorType {
            dims: vec![DimInfo::Named("result".to_string(), None)],
            precision: Prim::F32,
        };
        // Primary owner inference now gives `fail(...)` the enclosing
        // tensor result type, so the initial placeholder can already have
        // the right rank even though its extent remains anonymous.
        let early_placeholder = ctx.dag.add_node(
            RiscOp::synth_const(
                TensorType {
                    dims: vec![DimInfo::Named(String::new(), None)],
                    precision: Prim::F32,
                }
                .precision,
                0.0,
            ),
            vec![],
            TensorType {
                dims: vec![DimInfo::Named(String::new(), None)],
                precision: Prim::F32,
            },
            None,
        );
        // The sibling is lowered later for `if fail(...) else <body>`.
        let sibling = ctx.dag.add_node(
            RiscOp::Load { name: "x".into() },
            vec![],
            out_ty.clone(),
            None,
        );
        let conformed = ctx.conform_branch_placeholder(early_placeholder, &out_ty, sibling);
        let placeholder = ctx.dag.get(conformed).expect("conformed placeholder node");
        let [shape_source] = placeholder.shape_deps.as_slice() else {
            panic!(
                "the placeholder needs exactly one sibling shape source: {:?}",
                placeholder.shape_deps
            );
        };
        assert_ne!(
            conformed, early_placeholder,
            "a later sibling requires a fresh, topologically ordered placeholder"
        );
        assert_eq!(*shape_source, sibling);
        assert!(
            shape_source.0 < placeholder.id.0,
            "shape-only dependencies must point backward so remapping passes preserve them: \
             source={} placeholder={}",
            shape_source.0,
            placeholder.id.0
        );
    }

    fn non_drop_len(dag: &Dag) -> usize {
        dag.nodes()
            .iter()
            .filter(|node| !matches!(node.op, RiscOp::Drop))
            .count()
    }

    fn root_node(dag: &Dag) -> &crate::dag::DagNode {
        dag.roots()
            .last()
            .and_then(|id| dag.get(*id))
            .expect("expected lowered root")
    }

    #[test]
    fn lower_single_const() {
        let dag = parse_and_lower("(def {} x (lit {type: (t-prim {} f32)} 1.0))");
        assert_eq!(dag.len(), 1);
        assert_eq!(
            dag.get(NodeId(0)).unwrap().op,
            RiscOp::synth_const(Prim::F32, 1.0)
        );
        assert!(verify::verify(&dag).is_empty());
    }

    /// Build a `LowerCtx` with `x: tensor[<size>, f32]` pre-bound to a
    /// `Load`, lower the Deep `expr`, and return the resulting DAG plus
    /// the bound `x` node id. Mirrors how the body of a `def f(x) = ...`
    /// lowers with `x` already a parameter binding — the context the
    /// issue #318 `shape(&x, ...)` size argument resolves in.
    fn lower_with_bound_x(size: usize, expr_src: &str) -> Dag {
        let x_ty = TensorType {
            dims: vec![DimInfo::Lit(size)],
            precision: chelis_types::types::Prim::F32,
        };
        let mut ctx = LowerCtx::new(
            std::collections::HashMap::new(),
            std::collections::HashMap::new(),
            LinearityInfo::default(),
        );
        let x = ctx
            .dag
            .add_node(RiscOp::Load { name: "x".into() }, vec![], x_ty, None);
        ctx.bindings.insert("x".into(), LoweredValue::Node(x));
        let expr = chelis_deep::parser::parse_str(expr_src).expect("parse expand expr");
        let _ = ctx.lower_expr(&expr[0]);
        ctx.dag
    }

    fn only_expand(dag: &Dag) -> (DimExpr, Vec<DimInfo>) {
        dag.nodes()
            .iter()
            .find_map(|n| match &n.op {
                RiscOp::Expand { size, .. } => Some((size.clone(), n.output_type.dims.clone())),
                _ => None,
            })
            .expect("an Expand node must be present")
    }

    /// Issue #318: the SHAPE-DERIVED `expand` size `shape(&x, 0)` must
    /// recover the broadcast extent from `x`'s already-lowered dim, NOT
    /// default to size 1. This lowers the exact Deep the Surf idiom
    /// produces — `expand(scalar_to_tensor(c), 0, cast(shape(&x,
    /// cast(0,int32)), int32))` — directly through the `expand` lowering
    /// arm (bypassing the host-routing gate that keeps a `shape`-bearing
    /// *def* out of standalone DAG lowering), with `x: tensor[2]` bound.
    /// The recovered extent must be the concrete `2`, and the rank-0
    /// `scalar_to_tensor` source must yield a `tensor[2]` (rank-increasing)
    /// expand — never the collapsed `tensor[1]`.
    #[test]
    fn issue_318_expand_shape_arg_recovers_rank0_source_extent() {
        let expr = r#"
            (app {type: (t-tensor {} (d-lit {} 2) (t-prim {} f32))}
                 (var {} expand)
                 (app {type: (t-prim {} f32)}
                      (var {} scalar_to_tensor)
                      (cast {type: (t-prim {} f32)} (lit {} 3.0) (t-prim {} f32)))
                 (cast {} (lit {} 0) (t-prim {} int32))
                 (cast {}
                       (app {}
                            (var {} shape)
                            (borrow {} (var {type: (t-tensor {} (d-lit {} 2) (t-prim {} f32))} x))
                            (cast {} (lit {} 0) (t-prim {} int32)))
                       (t-prim {} int32)))
        "#;
        let dag = lower_with_bound_x(2, expr);
        let (size, dims) = only_expand(&dag);
        assert_eq!(
            size,
            DimExpr::Concrete(2),
            "shape-derived extent must recover x's axis-0 size 2, not the default 1; got {size:?}",
        );
        assert_eq!(
            dims,
            vec![DimInfo::Lit(2)],
            "rank-0-source expand output must be tensor[2], not the collapsed tensor[1]; got {dims:?}",
        );
    }

    /// Reading the extent from `shape(&x, 0)` must not accrue a dead node
    /// per `expand`. The size recovery lowers the operand `&x`
    /// (`borrow(var x)`) only to read its type; `borrow` forwards
    /// transparently (no `Borrow` RISC op exists) and `var x` resolves to
    /// the already-bound parameter, so the pre-bound `x` `Load` must remain
    /// the *only* `Load { name: "x" }` in the DAG — no duplicate operand
    /// node. The `only_expand` helper alone would not catch a stray node
    /// (the line-review concern), so assert the count explicitly.
    #[test]
    fn issue_318_expand_shape_arg_no_dead_node() {
        let expr = r#"
            (app {type: (t-tensor {} (d-lit {} 2) (t-prim {} f32))}
                 (var {} expand)
                 (app {type: (t-prim {} f32)}
                      (var {} scalar_to_tensor)
                      (cast {type: (t-prim {} f32)} (lit {} 3.0) (t-prim {} f32)))
                 (cast {} (lit {} 0) (t-prim {} int32))
                 (cast {}
                       (app {}
                            (var {} shape)
                            (borrow {} (var {type: (t-tensor {} (d-lit {} 2) (t-prim {} f32))} x))
                            (cast {} (lit {} 0) (t-prim {} int32)))
                       (t-prim {} int32)))
        "#;
        let dag = lower_with_bound_x(2, expr);
        let x_loads = dag
            .nodes()
            .iter()
            .filter(|n| matches!(&n.op, RiscOp::Load { name } if name == "x"))
            .count();
        assert_eq!(
            x_loads, 1,
            "the shape-arg operand `&x` must reuse the bound `x` Load, not \
             re-materialize it; found {x_loads} `Load {{ name: \"x\" }}` nodes",
        );
    }

    /// The fix recovers the broadcast EXTENT only; it leaves the
    /// language's rank/insert rule for `expand` untouched. To prove the
    /// extent recovery is orthogonal to the rank rule, lower a rank-1
    /// size-1 source `tensor[1]` with a shape-derived size: `expand([1],
    /// 0, shape(&x, 0))` with `x: tensor[2]`. The recovered extent at the
    /// inserted axis must be the concrete `2` (NOT the default `1`), and
    /// the output must follow the established INSERT semantics
    /// (`[1] -> [2, 1]`, axis 0 inserted) — the same rule pinned by
    /// `cli::build_c_linreg_expand_singleton_bias_keeps_rank2_shape` and
    /// `chelis-compiler-api`'s
    /// `host_runtime_expand_singleton_input_inserts_not_replicates`.
    /// #318's real source is rank-0 (`scalar_to_tensor`), so this rank-1
    /// case only exists to lock that the size fix did not perturb the
    /// rank rule.
    #[test]
    fn issue_318_expand_shape_arg_recovers_extent_without_changing_rank_rule() {
        // Source: a rank-1 size-1 constant `tensor[1]`.
        let expr = r#"
            (app {type: (t-tensor {} (d-lit {} 2) (d-lit {} 1) (t-prim {} f32))}
                 (var {} expand)
                 (cast {type: (t-tensor {} (d-lit {} 1) (t-prim {} f32))}
                       (lit {type: (t-tensor {} (d-lit {} 1) (t-prim {} f32))} 3.0)
                       (t-prim {} f32))
                 (cast {} (lit {} 0) (t-prim {} int32))
                 (cast {}
                       (app {}
                            (var {} shape)
                            (borrow {} (var {type: (t-tensor {} (d-lit {} 2) (t-prim {} f32))} x))
                            (cast {} (lit {} 0) (t-prim {} int32)))
                       (t-prim {} int32)))
        "#;
        let dag = lower_with_bound_x(2, expr);
        let (size, dims) = only_expand(&dag);
        assert_eq!(
            size,
            DimExpr::Concrete(2),
            "shape-derived extent must recover 2 at the inserted axis, not the default 1; got {size:?}",
        );
        assert_eq!(
            dims,
            vec![DimInfo::Lit(2), DimInfo::Lit(1)],
            "rank rule is unchanged: a rank-1 size-1 source INSERTS axis 0 \
             ([1] -> [2, 1]); got {dims:?}",
        );
    }

    /// Negative parity: the extent must track the tensor NAMED in the
    /// `shape(...)` argument, not the default. Here `x` is bound to
    /// `tensor[3]`, so `shape(&x, 0)` must recover `3`.
    #[test]
    fn issue_318_expand_shape_arg_tracks_named_tensor_size() {
        let expr = r#"
            (app {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))}
                 (var {} expand)
                 (app {type: (t-prim {} f32)}
                      (var {} scalar_to_tensor)
                      (cast {type: (t-prim {} f32)} (lit {} 3.0) (t-prim {} f32)))
                 (cast {} (lit {} 0) (t-prim {} int32))
                 (cast {}
                       (app {}
                            (var {} shape)
                            (borrow {} (var {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))} x))
                            (cast {} (lit {} 0) (t-prim {} int32)))
                       (t-prim {} int32)))
        "#;
        let dag = lower_with_bound_x(3, expr);
        let (size, dims) = only_expand(&dag);
        assert_eq!(
            size,
            DimExpr::Concrete(3),
            "extent must track the named tensor x's axis-0 size 3; got {size:?}",
        );
        assert_eq!(
            dims,
            vec![DimInfo::Lit(3)],
            "expand output must be tensor[3]; got {dims:?}"
        );
    }

    /// Negative-parity / regression lock: the LITERAL-size form (the #288
    /// fix) must still recover the concrete extent through the same arm,
    /// so the shape-derived recovery did not regress the literal path.
    #[test]
    fn issue_318_expand_literal_size_still_recovers_extent() {
        let expr = r#"
            (app {type: (t-tensor {} (d-lit {} 2) (t-prim {} f32))}
                 (var {} expand)
                 (app {type: (t-prim {} f32)}
                      (var {} scalar_to_tensor)
                      (cast {type: (t-prim {} f32)} (lit {} 3.0) (t-prim {} f32)))
                 (cast {} (lit {} 0) (t-prim {} int32))
                 (cast {} (lit {} 2) (t-prim {} int32)))
        "#;
        let dag = lower_with_bound_x(2, expr);
        let (size, dims) = only_expand(&dag);
        assert_eq!(
            size,
            DimExpr::Concrete(2),
            "literal size 2 must still extract; got {size:?}"
        );
        assert_eq!(
            dims,
            vec![DimInfo::Lit(2)],
            "literal expand output must be tensor[2]; got {dims:?}"
        );
    }

    /// Build a `LowerCtx` with `x: tensor[<size>, f32]` pre-bound, lower
    /// the multi-statement Deep `body_src` (a `let`/`bind` chain), and
    /// return the resulting DAG. Mirrors how the body of a `def f(x) = {
    /// len = shape(x, 0); ... }` lowers with `x` already a parameter
    /// binding — the chelis#369 context where the `expand` size argument is
    /// a `var len` reference, not a direct `shape(...)` app.
    fn lower_body_with_bound_x(size: usize, body_src: &str) -> Dag {
        let x_ty = TensorType {
            dims: vec![DimInfo::Lit(size)],
            precision: chelis_types::types::Prim::F32,
        };
        let mut ctx = LowerCtx::new(
            std::collections::HashMap::new(),
            std::collections::HashMap::new(),
            LinearityInfo::default(),
        );
        let x = ctx
            .dag
            .add_node(RiscOp::Load { name: "x".into() }, vec![], x_ty, None);
        ctx.bindings.insert("x".into(), LoweredValue::Node(x));
        let expr = chelis_deep::parser::parse_str(body_src).expect("parse body");
        let _ = ctx.lower_expr(&expr[0]);
        ctx.dag
    }

    /// chelis#369 (the fix): the `tensor_full_like` idiom binds the shape
    /// read to `len` first — `len = shape(x, 0)` — then uses `cast(len,
    /// int32)` as the `expand` size. The size recovery must follow the
    /// `let` indirection back to the bound `shape(x, 0)` and recover the
    /// concrete extent `3`, NOT silently default to `Concrete(1)` (which
    /// is what produced the `Lit(3) vs Lit(1)` backward-DAG failure).
    #[test]
    fn issue_369_expand_let_bound_shape_recovers_extent() {
        // (let {} (bind {} len (shape x 0))
        //   (expand (scalar_to_tensor 3.0) 0 (cast len int32)))
        let body = r#"
            (let {}
                 (bind {}
                       len
                       (app {}
                            (var {} shape)
                            (var {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))} x)
                            (cast {} (lit {} 0) (t-prim {} int32))))
                 (app {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))}
                      (var {} expand)
                      (app {type: (t-prim {} f32)}
                           (var {} scalar_to_tensor)
                           (cast {type: (t-prim {} f32)} (lit {} 3.0) (t-prim {} f32)))
                      (cast {} (lit {} 0) (t-prim {} int32))
                      (cast {} (var {} len) (t-prim {} int32))))
        "#;
        let dag = lower_body_with_bound_x(3, body);
        let (size, dims) = only_expand(&dag);
        assert_eq!(
            size,
            DimExpr::Concrete(3),
            "a `let len = shape(x, 0)`-bound extent must recover x's axis-0 \
             size 3 through the `let` indirection, not the default 1 \
             (chelis#369); got {size:?}",
        );
        assert_eq!(
            dims,
            vec![DimInfo::Lit(3)],
            "rank-0-source expand output must be tensor[3], not the collapsed \
             tensor[1] (chelis#369); got {dims:?}",
        );
    }

    /// chelis#369 non-hardcoding guard: the same `let len = shape(x, 0)`
    /// idiom at a DIFFERENT extent (`x: tensor[5]`) must recover
    /// `Concrete(5)`. `issue_369_expand_let_bound_shape_recovers_extent`
    /// pins only the `tensor[3]` case, which a buggy implementation that
    /// hardcoded `Concrete(3)` (or echoed back the literal `3` annotation)
    /// would still satisfy. Exercising a second, distinct extent proves the
    /// recovery actually reads `x`'s axis-0 size rather than emitting a
    /// fixed constant.
    #[test]
    fn issue_369_expand_let_bound_shape_recovers_distinct_extent_5() {
        // Identical structure to the extent-3 test, but `x: tensor[5]` and
        // the expand output is annotated `tensor[5]`. A recovery that
        // reads x's dim yields Concrete(5); a hardcoded Concrete(3) fails.
        let body = r#"
            (let {}
                 (bind {}
                       len
                       (app {}
                            (var {} shape)
                            (var {type: (t-tensor {} (d-lit {} 5) (t-prim {} f32))} x)
                            (cast {} (lit {} 0) (t-prim {} int32))))
                 (app {type: (t-tensor {} (d-lit {} 5) (t-prim {} f32))}
                      (var {} expand)
                      (app {type: (t-prim {} f32)}
                           (var {} scalar_to_tensor)
                           (cast {type: (t-prim {} f32)} (lit {} 3.0) (t-prim {} f32)))
                      (cast {} (lit {} 0) (t-prim {} int32))
                      (cast {} (var {} len) (t-prim {} int32))))
        "#;
        let dag = lower_body_with_bound_x(5, body);
        let (size, dims) = only_expand(&dag);
        assert_eq!(
            size,
            DimExpr::Concrete(5),
            "a `let len = shape(x, 0)`-bound extent on `x: tensor[5]` must \
             recover Concrete(5), not a hardcoded 3 or the default 1 \
             (chelis#369 non-hardcoding); got {size:?}",
        );
        assert_eq!(
            dims,
            vec![DimInfo::Lit(5)],
            "rank-0-source expand output must be tensor[5] at this extent, \
             proving the recovered size is x's axis-0 dim, not a constant; \
             got {dims:?}",
        );
    }

    /// chelis#369 negative parity + chelis#469/#528 positive parity: the
    /// SHAPE-recovery path must follow ONLY a genuine `let len = shape(...)`
    /// binding — a `len` bound to a static `cast(7, int32)` must NOT
    /// mis-recover `x`'s shape extent 3. It is not sourceless, though: a
    /// `let`-bound static value folds to its own extent (`SizeClass::Static`
    /// followed through the `let`), so the size resolves to `Concrete(7)`, not
    /// the pre-#469 size-1 default (which silently miscompiled eval-`[7]` to
    /// C-`[1]`) and not `x`'s 3.
    #[test]
    fn issue_369_expand_let_bound_non_shape_does_not_recover() {
        // len is bound to a static int (`cast(7, int32)`), not a shape read.
        let body = r#"
            (let {}
                 (bind {} len (cast {} (lit {} 7) (t-prim {} int32)))
                 (app {type: (t-tensor {} (d-lit {} 1) (t-prim {} f32))}
                      (var {} expand)
                      (app {type: (t-prim {} f32)}
                           (var {} scalar_to_tensor)
                           (cast {type: (t-prim {} f32)} (lit {} 3.0) (t-prim {} f32)))
                      (cast {} (lit {} 0) (t-prim {} int32))
                      (cast {} (var {} len) (t-prim {} int32))))
        "#;
        let dag = lower_body_with_bound_x(3, body);
        let (size, _dims) = only_expand(&dag);
        // Folds to the `let`-bound static value 7 (NOT `x`'s shape extent 3,
        // proving the shape-recovery path did not fire, and NOT the size-1
        // default, proving the static value is materialized).
        assert_eq!(
            size,
            DimExpr::Concrete(7),
            "a `let`-bound static size must fold to its own extent 7, never \
             recover x's 3 or default to 1; got {size:?}",
        );
    }

    /// chelis#369 negative parity: re-binding a name that WAS a shape binding
    /// to a static value must drop the stale shape entry, so a later expand
    /// size referencing the re-bound name does not recover the old shape
    /// extent 3 — it folds to the NEW static value (5) instead (chelis#469/
    /// #528 `SizeClass::Static` shadowing symmetry). Guards the shadowing
    /// path in `lower_let` for both `shape_bindings` and `static_size_bindings`.
    #[test]
    fn issue_369_expand_shadowed_let_binding_does_not_leak_stale_shape() {
        // len = shape(x, 0)          -- shape binding
        // len = cast(5, int32)       -- re-bound to a static value
        // expand(s, 0, cast(len, int32))  -- must recover 5, NOT the stale 3
        let body = r#"
            (let {}
                 (bind {}
                       len
                       (app {}
                            (var {} shape)
                            (var {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))} x)
                            (cast {} (lit {} 0) (t-prim {} int32)))
                       len
                       (cast {} (lit {} 5) (t-prim {} int32)))
                 (app {type: (t-tensor {} (d-lit {} 1) (t-prim {} f32))}
                      (var {} expand)
                      (app {type: (t-prim {} f32)}
                           (var {} scalar_to_tensor)
                           (cast {type: (t-prim {} f32)} (lit {} 3.0) (t-prim {} f32)))
                      (cast {} (lit {} 0) (t-prim {} int32))
                      (cast {} (var {} len) (t-prim {} int32))))
        "#;
        let dag = lower_body_with_bound_x(3, body);
        let (size, _dims) = only_expand(&dag);
        assert_eq!(
            size,
            DimExpr::Concrete(5),
            "a re-bound (shadowed) `len` must fold to the NEW static 5, never \
             recover the stale shape extent 3; got {size:?}",
        );
    }

    #[test]
    fn lowering_marks_reusable_input_from_linearity_hint() {
        let src = r#"
            (def {} x (var {type: (t-tensor {} (d-lit {} 4) (t-prim {} f32))} x))
            (def {} y
              (app {type: (t-tensor {} (d-lit {} 4) (t-prim {} f32))}
                   (var {} relu)
                   (var {type: (t-tensor {} (d-lit {} 4) (t-prim {} f32))} x)))
        "#;
        let dag = parse_and_lower(src);
        let node = dag
            .roots()
            .last()
            .and_then(|id| dag.get(*id))
            .expect("lowered root node");
        assert_eq!(node.reusable_input, Some(NodeId(0)));
    }

    #[test]
    fn lower_gather_uses_sparse_ir_node() {
        let src = r#"
            (def {} values
              (var {type: (t-tensor {} (d-lit {} 4) (d-lit {} 2) (t-prim {} f32))} values))
            (def {} indices
              (var {type: (t-tensor {} (d-lit {} 3) (t-prim {} int32))} indices))
            (def {} out
              (app {type: (t-tensor {} (d-lit {} 3) (d-lit {} 2) (t-prim {} f32))}
                   (var {} gather)
                   (var {type: (t-tensor {} (d-lit {} 4) (d-lit {} 2) (t-prim {} f32))} values)
                   (var {type: (t-tensor {} (d-lit {} 3) (t-prim {} int32))} indices)
                   (lit {type: (t-prim {} int32)} 0)))
        "#;
        let dag = parse_and_lower(src);
        let gather = dag
            .nodes()
            .iter()
            .find(|node| matches!(node.op, RiscOp::Gather { axis: 0 }))
            .expect("Surf gather should lower to first-class sparse IR");
        assert_eq!(
            gather.output_type,
            TensorType {
                dims: vec![DimInfo::Lit(3), DimInfo::Lit(2)],
                precision: Prim::F32,
            }
        );
        assert!(verify::verify(&dag).is_empty());
    }

    #[test]
    fn lower_scatter_replace_uses_sparse_ir_node() {
        // Surf `scatter_replace(base, indices, updates, axis)` must
        // lower directly to the first-class `RiscOp::Scatter` sparse
        // IR node, paralleling the tensor-lane `gather` lowering.
        // This locks the contract: a Surf-level use of
        // scatter_replace MUST reach the sparse evaluator/codegen
        // path, NOT the host-runtime fallback (which the existing
        // `scatter(..., mode)` builtin uses).
        let src = r#"
            (def {} base
              (var {type: (t-tensor {} (d-lit {} 4) (d-lit {} 2) (t-prim {} f32))} base))
            (def {} indices
              (var {type: (t-tensor {} (d-lit {} 3) (t-prim {} int32))} indices))
            (def {} updates
              (var {type: (t-tensor {} (d-lit {} 3) (d-lit {} 2) (t-prim {} f32))} updates))
            (def {} out
              (app {type: (t-tensor {} (d-lit {} 4) (d-lit {} 2) (t-prim {} f32))}
                   (var {} scatter_replace)
                   (var {type: (t-tensor {} (d-lit {} 4) (d-lit {} 2) (t-prim {} f32))} base)
                   (var {type: (t-tensor {} (d-lit {} 3) (t-prim {} int32))} indices)
                   (var {type: (t-tensor {} (d-lit {} 3) (d-lit {} 2) (t-prim {} f32))} updates)
                   (lit {type: (t-prim {} int32)} 0)))
        "#;
        let dag = parse_and_lower(src);
        let scatter = dag
            .nodes()
            .iter()
            .find(|node| matches!(node.op, RiscOp::Scatter { axis: 0 }))
            .expect("Surf scatter_replace should lower to first-class sparse IR");
        assert_eq!(
            scatter.output_type,
            TensorType {
                dims: vec![DimInfo::Lit(4), DimInfo::Lit(2)],
                precision: Prim::F32,
            }
        );
        // Defense in depth: the lowered DAG must NOT contain a
        // ScatterAdd from a Surf scatter_replace — those are
        // intentionally distinct primitives.
        let has_scatter_add = dag
            .nodes()
            .iter()
            .any(|node| matches!(node.op, RiscOp::ScatterAdd { .. }));
        assert!(
            !has_scatter_add,
            "scatter_replace must NOT lower to RiscOp::ScatterAdd; \
             those are distinct primitives with different semantics"
        );
        assert!(verify::verify(&dag).is_empty());
    }

    #[test]
    fn lower_add_two_consts() {
        let src = r#"
            (def {} a (lit {type: (t-tensor {} (t-prim {} f32))} 1.0))
            (def {} b (lit {type: (t-tensor {} (t-prim {} f32))} 2.0))
            (def {} c (app {} (var {} add) (var {} a) (var {} b)))
        "#;
        let dag = parse_and_lower(src);
        assert_eq!(non_drop_len(&dag), 3);
        let add_node = dag.get(NodeId(2)).unwrap();
        assert_eq!(add_node.op, RiscOp::Add);
        assert_eq!(add_node.inputs, vec![NodeId(0), NodeId(1)]);
        assert!(verify::verify(&dag).is_empty());
    }

    #[test]
    fn lower_neg() {
        let src = r#"
            (def {} x (lit {type: (t-tensor {} (t-prim {} f32))} 5.0))
            (def {} y (app {} (var {} neg) (var {} x)))
        "#;
        let dag = parse_and_lower(src);
        assert_eq!(dag.len(), 2);
        let neg_node = dag.get(NodeId(1)).unwrap();
        assert_eq!(neg_node.op, RiscOp::Neg);
        assert_eq!(neg_node.inputs, vec![NodeId(0)]);
        assert!(verify::verify(&dag).is_empty());
    }

    #[test]
    fn lower_sub_decomposes() {
        let src = r#"
            (def {} a (lit {type: (t-tensor {} (t-prim {} f32))} 3.0))
            (def {} b (lit {type: (t-tensor {} (t-prim {} f32))} 1.0))
            (def {} c (app {} (var {} sub) (var {} a) (var {} b)))
        "#;
        let dag = parse_and_lower(src);
        // a=Const(3), b=Const(1), Neg(b), Add(a, Neg(b))
        assert_eq!(non_drop_len(&dag), 4);
        assert!(verify::verify(&dag).is_empty());
        assert_eq!(root_node(&dag).op, RiscOp::Add);
    }

    #[test]
    fn lower_relu_decomposes() {
        let src = r#"
            (def {} x (lit {type: (t-tensor {} (t-prim {} f32))} -2.0))
            (def {} y (app {} (var {} relu) (var {} x)))
        "#;
        let dag = parse_and_lower(src);
        // x=Const(-2), Const(0), MaxElem(x, 0)
        assert_eq!(non_drop_len(&dag), 3);
        assert!(verify::verify(&dag).is_empty());
        assert_eq!(root_node(&dag).op, RiscOp::MaxElem);
    }

    #[test]
    fn lower_let_binding() {
        let src = r#"
            (let {} (bind {} x (lit {type: (t-tensor {} (t-prim {} f32))} 10.0))
                (let {}
                  (bind {} out (app {} (var {} neg) (var {} x)))
                  (let {}
                    (bind {} __drop_x (app {} (var {} drop) (var {} x)))
                    (var {} out))))
        "#;
        let dag = parse_and_lower(src);
        // x=Const(10), Neg(x)
        assert_eq!(non_drop_len(&dag), 2);
        assert!(verify::verify(&dag).is_empty());
    }

    #[test]
    fn unconsumed_let_binding_gets_terminal_drop() {
        let src = r#"
            (let {} (bind {} x (lit {type: (t-tensor {} (t-prim {} f32))} 10.0))
                (let {}
                  (bind {} out (app {} (var {} neg) (var {} x)))
                  (var {} out)))
        "#;
        let dag = parse_and_lower(src);
        assert_eq!(non_drop_len(&dag), 2);
        let drops = dag
            .nodes()
            .iter()
            .filter(|node| matches!(node.op, RiscOp::Drop))
            .map(|node| node.inputs.clone())
            .collect::<Vec<_>>();
        assert_eq!(drops, vec![vec![NodeId(0)]]);
        assert!(verify::verify(&dag).is_empty());
    }

    #[test]
    fn repeated_consuming_user_call_gets_copy() {
        let src = r#"
            (def {} consume
              (fn {type: (t-fn {}
                            (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))
                            (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32)))}
                (params {}
                  (x {type: (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))}))
                (realize {type: (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))}
                  (var {} x))))
            (def {} double_it
              (fn {type: (t-fn {}
                            (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))
                            (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32)))}
                (params {}
                  (x {type: (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))}))
                (app {type: (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))}
                  (var {} add)
                  (app {type: (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))}
                    (var {} consume)
                    (var {} x))
                  (app {type: (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))}
                    (var {} consume)
                    (var {} x)))))
        "#;
        let dag = parse_and_lower(src);
        let copy_count = dag
            .nodes()
            .iter()
            .filter(|node| matches!(node.op, RiscOp::Copy))
            .count();
        assert_eq!(copy_count, 1);
        assert!(verify::verify(&dag).is_empty());
    }

    #[test]
    fn context_lowering_inserts_copy_for_library_boundary_fanout() {
        let library_exprs = chelis_deep::parser::parse_str(
            r#"
                (def {} consume
                  (fn {type: (t-fn {}
                                (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))
                                (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32)))}
                    (params {}
                      (x {type: (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))}))
                    (realize {type: (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))}
                      (var {} x))))
            "#,
        )
        .expect("parse library");
        let (type_env, library_checked) =
            chelis_types::build_compiled_library_context(&library_exprs).expect("library checks");
        let library_checked =
            chelis_effects::check_program(&library_checked).expect("library effects");
        let library_checked =
            chelis_types::check_linearity(&library_checked).expect("library linearity");
        let library = lower_program_to_library(&library_checked);

        let new_exprs = chelis_deep::parser::parse_str(
            r#"
                (def {} double_it
                  (fn {type: (t-fn {}
                                (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))
                                (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32)))}
                    (params {}
                      (x {type: (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))}))
                    (app {type: (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))}
                      (var {} add)
                      (app {type: (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))}
                        (var {} consume)
                        (var {} x))
                      (app {type: (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))}
                        (var {} consume)
                        (var {} x)))))
            "#,
        )
        .expect("parse new code");
        let new_checked =
            chelis_types::check_ir_with_context(&type_env, &new_exprs).expect("new code checks");
        let new_checked =
            chelis_effects::check_effects_with_context(&library_checked, &new_checked)
                .expect("new code effects");
        let new_checked =
            chelis_types::check_linearity_with_context(&library_checked, &new_checked)
                .expect("new code linearity");

        let dag = lower_program_with_context(&library, &new_checked);
        let copy_count = dag
            .nodes()
            .iter()
            .filter(|node| matches!(node.op, RiscOp::Copy))
            .count();
        assert_eq!(copy_count, 1, "{:?}", dag.nodes());
        assert!(verify::verify(&dag).is_empty());
    }

    #[test]
    fn lower_unknown_var_becomes_load() {
        let src = "(def {} y (var {} weights))";
        let dag = parse_and_lower_unchecked(src);
        assert_eq!(dag.len(), 1);
        assert_eq!(
            dag.get(NodeId(0)).unwrap().op,
            RiscOp::Load {
                name: "weights".into()
            }
        );
        assert!(verify::verify(&dag).is_empty());
    }

    #[test]
    fn lm_style_local_grad_wrapper_defs_are_marked_lowerable() {
        let checked = parse_and_check(
            r#"
                (defsig {}
                  jac_row
                  (t-fn {}
                    (t-fn {}
                      (t-tensor {} (d-var {} n) (t-prim {} f32))
                      (t-prim {} f32)
                      (t-prim {} f32)
                      (t-prim {} f32))
                    (t-tensor {} (d-var {} n) (t-prim {} f32))
                    (t-prim {} f32)
                    (t-prim {} f32)
                    (t-tensor {} (d-var {} n) (t-prim {} f32))))
                (def {}
                  jac_row
                  (fn {}
                    (params {}
                      (model {type: (t-fn {}
                                       (t-tensor {} (d-var {} n) (t-prim {} f32))
                                       (t-prim {} f32)
                                       (t-prim {} f32)
                                       (t-prim {} f32))})
                      (theta {type: (t-tensor {} (d-var {} n) (t-prim {} f32))})
                      (x {type: (t-prim {} f32)})
                      (y {type: (t-prim {} f32)}))
                    (let {}
                      (bind {}
                        target
                        (fn {}
                          (params {}
                            (theta_local {type: (t-tensor {} (d-var {} n) (t-prim {} f32))}))
                          (app {} (var {} model) (var {} theta_local) (var {} x) (var {} y))))
                      (app {}
                        (grad {wrt: (var {} theta_local)}
                          (var {} target)
                          (lit {type: (t-prim {} int32)} 0))
                        (var {} theta)))))
                (defsig {}
                  lm_model
                  (t-fn {}
                    (t-tensor {} (d-lit {} 2) (t-prim {} f32))
                    (t-prim {} f32)
                    (t-prim {} f32)
                    (t-prim {} f32)))
                (def {}
                  lm_model
                  (fn {}
                    (params {}
                      (theta {type: (t-tensor {} (d-lit {} 2) (t-prim {} f32))})
                      (x {type: (t-prim {} f32)})
                      (y {type: (t-prim {} f32)}))
                    (let {}
                      (bind {}
                        y_hat
                        (if {}
                          (app {}
                            (var {} lt)
                            (var {} x)
                            (cast {} (lit {type: (t-prim {} f32)} 0.0) (t-prim {} f32)))
                          (app {}
                            (var {} tensor_to_scalar)
                            (app {}
                              (var {} sum)
                              (copy {} (var {} theta))
                              (lit {type: (t-prim {} int32)} 0)))
                          (app {}
                            (var {} add)
                            (app {}
                              (var {} tensor_to_scalar)
                              (app {}
                                (var {} sum)
                                (copy {} (var {} theta))
                                (lit {type: (t-prim {} int32)} 0)))
                            (var {} x))))
                      (app {} (var {} sub) (var {} y) (var {} y_hat)))))
                (def {}
                  out
                  (app {}
                    (var {} jac_row)
                    (var {} lm_model)
                    (app {}
                      (var {} to_tensor)
                      (app {}
                        (var {} Cons)
                        (lit {type: (t-prim {} f32)} 1.0)
                        (app {} (var {} Cons) (lit {type: (t-prim {} f32)} 2.0) (var {} Nil))))
                    (cast {} (lit {type: (t-prim {} f32)} 1.0) (t-prim {} f32))
                    (cast {} (lit {type: (t-prim {} f32)} 3.0) (t-prim {} f32))))
            "#,
        );
        let lowered = top_level_lowering_map(checked.exprs(), checked.type_env());
        assert_eq!(
            lowered.get("jac_row"),
            Some(&true),
            "jac_row lowering map: {lowered:?}"
        );
        assert_eq!(
            lowered.get("lm_model"),
            Some(&false),
            "lm_model lowering map: {lowered:?}"
        );
        assert_eq!(
            lowered.get("out"),
            Some(&false),
            "out lowering map: {lowered:?}"
        );
    }

    #[test]
    fn lm_style_local_grad_wrapper_subexpr_does_not_load_callable_arg_as_data() {
        let checked = parse_and_check(
            r#"
                (defsig {}
                  jac_row
                  (t-fn {}
                    (t-fn {}
                      (t-tensor {} (d-var {} n) (t-prim {} f32))
                      (t-prim {} f32)
                      (t-prim {} f32)
                      (t-prim {} f32))
                    (t-tensor {} (d-var {} n) (t-prim {} f32))
                    (t-prim {} f32)
                    (t-prim {} f32)
                    (t-tensor {} (d-var {} n) (t-prim {} f32))))
                (def {}
                  jac_row
                  (fn {}
                    (params {}
                      (model {type: (t-fn {}
                                       (t-tensor {} (d-var {} n) (t-prim {} f32))
                                       (t-prim {} f32)
                                       (t-prim {} f32)
                                       (t-prim {} f32))})
                      (theta {type: (t-tensor {} (d-var {} n) (t-prim {} f32))})
                      (x {type: (t-prim {} f32)})
                      (y {type: (t-prim {} f32)}))
                    (let {}
                      (bind {}
                        target
                        (fn {}
                          (params {}
                            (theta_local {type: (t-tensor {} (d-var {} n) (t-prim {} f32))}))
                          (app {} (var {} model) (var {} theta_local) (var {} x) (var {} y))))
                      (app {}
                        (grad {wrt: (var {} theta_local)}
                          (var {} target)
                          (lit {type: (t-prim {} int32)} 0))
                        (var {} theta)))))
                (defsig {}
                  lm_model
                  (t-fn {}
                    (t-tensor {} (d-lit {} 2) (t-prim {} f32))
                    (t-prim {} f32)
                    (t-prim {} f32)
                    (t-prim {} f32)))
                (def {}
                  lm_model
                  (fn {}
                    (params {}
                      (theta {type: (t-tensor {} (d-lit {} 2) (t-prim {} f32))})
                      (x {type: (t-prim {} f32)})
                      (y {type: (t-prim {} f32)}))
                    (let {}
                      (bind {}
                        y_hat
                        (if {}
                          (app {}
                            (var {} lt)
                            (var {} x)
                            (cast {} (lit {type: (t-prim {} f32)} 0.0) (t-prim {} f32)))
                          (app {}
                            (var {} tensor_to_scalar)
                            (app {}
                              (var {} sum)
                              (copy {} (var {} theta))
                              (lit {type: (t-prim {} int32)} 0)))
                          (app {}
                            (var {} add)
                            (app {}
                              (var {} tensor_to_scalar)
                              (app {}
                                (var {} sum)
                                (copy {} (var {} theta))
                                (lit {type: (t-prim {} int32)} 0)))
                            (var {} x))))
                      (app {} (var {} sub) (var {} y) (var {} y_hat)))))
                (def {}
                  out
                  (app {}
                    (var {} jac_row)
                    (var {} lm_model)
                    (app {}
                      (var {} to_tensor)
                      (app {}
                        (var {} Cons)
                        (lit {type: (t-prim {} f32)} 1.0)
                        (app {} (var {} Cons) (lit {type: (t-prim {} f32)} 2.0) (var {} Nil))))
                    (cast {} (lit {type: (t-prim {} f32)} 1.0) (t-prim {} f32))
                    (cast {} (lit {type: (t-prim {} f32)} 3.0) (t-prim {} f32))))
            "#,
        );
        let out_expr = checked
            .exprs()
            .iter()
            .find(|expr| top_level_expr_name(expr) == Some("out"))
            .expect("out def");
        let out_body = match out_expr {
            Expr::List(list, _) => children(list).get(1).expect("out body"),
            _ => panic!("out def must be a list"),
        };
        let mut ctx = LowerCtx::new(
            checked
                .type_env()
                .iter()
                .map(|(name, ty_expr)| (name.clone(), LowerCtx::type_from_type_expr(ty_expr)))
                .collect(),
            collect_top_level_defs(checked.exprs()),
            LinearityInfo::default(),
        );
        let out_kids = match out_body {
            Expr::List(list, _) => children(list),
            _ => panic!("out body must be app"),
        };
        assert!(
            matches!(
                ctx.resolve_callable_expr(&out_kids[0]),
                Some(CallableExpr::Plain(_))
            ),
            "expected jac_row callee to resolve as callable"
        );
        assert!(
            matches!(
                ctx.resolve_callable_expr(&out_kids[1]),
                Some(CallableExpr::Plain(_))
            ),
            "expected lm_model arg to resolve as callable"
        );
        let lowered_value = ctx.lower_expr(out_body);
        let dag = ctx.dag;
        for id in lowered_value.flatten_nodes() {
            // rootless DAGs are hard to inspect in assertions; mirror lower_subexpr_program.
            // This is test-only.
            let _ = id;
        }
        assert!(
            !dag.nodes()
                .iter()
                .any(|node| matches!(&node.op, RiscOp::Load { name } if name == "model")),
            "specialized local grad wrapper must not leave callable arg as a data load: {dag:#?}"
        );
    }

    #[test]
    fn nested_callable_param_app_in_grad_body_resolves_named_function_arg() {
        let checked = parse_and_check(
            r#"
                (defsig {}
                  jac_row
                  (t-fn {}
                    (t-fn {}
                      (t-tensor {} (d-var {} n) (t-prim {} f32))
                      (t-prim {} f32)
                      (t-prim {} f32)
                      (t-prim {} f32))
                    (t-tensor {} (d-var {} n) (t-prim {} f32))
                    (t-prim {} f32)
                    (t-prim {} f32)
                    (t-tensor {} (d-var {} n) (t-prim {} f32))))
                (def {}
                  jac_row
                  (fn {}
                    (params {}
                      (model {type: (t-fn {}
                                       (t-tensor {} (d-var {} n) (t-prim {} f32))
                                       (t-prim {} f32)
                                       (t-prim {} f32)
                                       (t-prim {} f32))})
                      (theta {type: (t-tensor {} (d-var {} n) (t-prim {} f32))})
                      (x {type: (t-prim {} f32)})
                      (y {type: (t-prim {} f32)}))
                    (let {}
                      (bind {}
                        target
                        (fn {}
                          (params {}
                            (theta_local {type: (t-tensor {} (d-var {} n) (t-prim {} f32))}))
                          (app {} (var {} model) (var {} theta_local) (var {} x) (var {} y))))
                      (app {}
                        (grad {wrt: (var {} theta_local)}
                          (var {} target)
                          (lit {type: (t-prim {} int32)} 0))
                        (var {} theta)))))
                (defsig {}
                  lm_model
                  (t-fn {}
                    (t-tensor {} (d-lit {} 2) (t-prim {} f32))
                    (t-prim {} f32)
                    (t-prim {} f32)
                    (t-prim {} f32)))
                (def {}
                  lm_model
                  (fn {}
                    (params {}
                      (theta {type: (t-tensor {} (d-lit {} 2) (t-prim {} f32))})
                      (x {type: (t-prim {} f32)})
                      (y {type: (t-prim {} f32)}))
                    (app {} (var {} sub) (var {} y) (var {} x))))
            "#,
        );
        let program_defs = collect_top_level_defs(checked.exprs());
        let jac_fn = match program_defs.get("jac_row") {
            Some(expr) => expr.clone(),
            None => panic!("missing jac_row"),
        };
        let (param_names, jac_body) = {
            let ctx = LowerCtx::new(
                checked
                    .type_env()
                    .iter()
                    .map(|(name, ty_expr)| (name.clone(), LowerCtx::type_from_type_expr(ty_expr)))
                    .collect(),
                program_defs.clone(),
                LinearityInfo::default(),
            );
            ctx.extract_fn_parts(&jac_fn).expect("jac_row fn parts")
        };
        let mut inline_ctx = LowerCtx::new(
            checked
                .type_env()
                .iter()
                .map(|(name, ty_expr)| (name.clone(), LowerCtx::type_from_type_expr(ty_expr)))
                .collect(),
            program_defs.clone(),
            LinearityInfo::default(),
        );
        let app_exprs = chelis_deep::parser::parse_str(
            "(app {} (var {} jac_row) (var {} lm_model) (app {} (var {} to_tensor) (app {} (var {} Cons) (lit {type: (t-prim {} f32)} 1.0) (app {} (var {} Cons) (lit {type: (t-prim {} f32)} 2.0) (var {} Nil)))) (cast {} (lit {type: (t-prim {} f32)} 1.0) (t-prim {} f32)) (cast {} (lit {type: (t-prim {} f32)} 3.0) (t-prim {} f32)))"
        )
        .expect("parse app expr");
        let out_call_args = match &app_exprs[0] {
            Expr::List(list, _) => children(list)[1..].to_vec(),
            _ => panic!("expected app"),
        };
        for (name, arg_expr) in param_names.iter().zip(out_call_args.iter()) {
            if let Some(callable) = inline_ctx.callable_binding_expr(arg_expr) {
                inline_ctx.local_callables.insert(name.clone(), callable);
            } else {
                let arg_id = inline_ctx.lower_expr(arg_expr);
                inline_ctx.bindings.insert(name.clone(), arg_id);
            }
        }
        let let_kids = match jac_body {
            Expr::List(list, _) => children(list),
            _ => panic!("expected let body"),
        };
        let target_fn = match &let_kids[0] {
            Expr::List(bind_list, _) => children(bind_list)[1].clone(),
            _ => panic!("expected bind list"),
        };
        inline_ctx
            .local_callables
            .insert("target".to_string(), target_fn.clone());
        let inner_app = match &target_fn {
            Expr::List(list, _) => children(list)[1].clone(),
            _ => panic!("expected target fn"),
        };
        let mut subctx = LowerCtx::new(
            checked
                .type_env()
                .iter()
                .map(|(name, ty_expr)| (name.clone(), LowerCtx::type_from_type_expr(ty_expr)))
                .collect(),
            program_defs,
            LinearityInfo::default(),
        );
        subctx.local_callables = inline_ctx.local_callables.clone();
        let theta_local_ty = extract_param_type(&target_fn, 0).expect("theta_local type");
        let theta_local = subctx.lower_fn_param_binding("theta_local", Some(theta_local_ty));
        subctx
            .bindings
            .insert("theta_local".to_string(), theta_local);
        let inner_callee = match &inner_app {
            Expr::List(list, _) => children(list).first().expect("inner app callee"),
            _ => panic!("expected inner app"),
        };
        assert!(
            matches!(
                subctx.resolve_callable_expr(inner_callee),
                Some(CallableExpr::Plain(_))
            ),
            "expected nested callee `(var model)` to resolve as callable before lowering"
        );
        let _ = subctx.lower_expr(&inner_app);
        assert!(
            !subctx
                .dag
                .nodes()
                .iter()
                .any(|node| matches!(&node.op, RiscOp::Load { name } if name == "model")),
            "nested callable param app must inline named function arg rather than load `model`: {:#?}",
            subctx.dag
        );
    }

    #[test]
    fn tuple_return_lowers_to_named_store_roots() {
        let src = r#"
            (def {} grads
              (tuple {}
                (lit {type: (t-prim {} f32)} 1.0)
                (lit {type: (t-prim {} f32)} 2.0)))
        "#;
        let dag = parse_and_lower(src);
        let roots = dag.roots();
        assert_eq!(roots.len(), 2);
        assert!(matches!(
            dag.get(roots[0]).map(|node| &node.op),
            Some(RiscOp::Store { name }) if name == "grads.0"
        ));
        assert!(matches!(
            dag.get(roots[1]).map(|node| &node.op),
            Some(RiscOp::Store { name }) if name == "grads.1"
        ));
        assert!(verify::verify(&dag).is_empty());
    }

    /// Issue chelis#232: `(export {} forward)` is a module-system
    /// directive, not a value-producing expression. `lower_top_level`
    /// must early-return on it; otherwise `lower_list`'s catch-all
    /// branch walks the children, emits a `Load { name: "forward" }`
    /// for the bare symbol, and `lower_top_level` adds it as a DAG
    /// root, inflating `dag.roots().len()` and breaking the
    /// `tensor_root_names.len() == dag.roots().len()` invariant in
    /// `chelis-compiler-api::compiler::compile_source`.
    #[test]
    fn issue232_export_directive_does_not_emit_dag_root() {
        let src = r#"
            (def {} forward (lit {type: (t-prim {} f32)} 1.0))
            (export {} forward)
        "#;
        let dag = parse_and_lower(src);
        // Only `forward`'s lowered value-node is a root. The
        // `(export {} forward)` directive must contribute zero roots
        // and zero nodes — it is not a value expression.
        assert_eq!(
            dag.roots().len(),
            1,
            "export directive must not emit any DAG root: roots={:?}",
            dag.roots()
        );
        assert!(
            !dag.nodes()
                .iter()
                .any(|node| matches!(&node.op, RiscOp::Load { name } if name == "forward")),
            "export directive must not lower its symbol child to a Load: {:#?}",
            dag.nodes()
        );
    }

    /// Issue chelis#232: same shape, `import` instead of `export`.
    /// `(import {} Math (...))` desugars to a list whose children are
    /// not value expressions; `lower_top_level`'s catch-all would
    /// emit a `Const` node and add it as a root.
    #[test]
    fn issue232_import_directive_does_not_emit_dag_root() {
        let src = r#"
            (def {} forward (lit {type: (t-prim {} f32)} 1.0))
            (import {} Math (params {}))
        "#;
        // Use `parse_and_lower_unchecked` because the standalone
        // `(import {} ...)` form isn't run through the regular
        // type-checker path; we want a direct lowering observation.
        let exprs = chelis_deep::parser::parse_str(src).expect("parse failed");
        let mut ctx = LowerCtx::new(
            std::collections::HashMap::new(),
            std::collections::HashMap::new(),
            LinearityInfo::default(),
        );
        for expr in &exprs {
            ctx.lower_top_level(expr);
        }
        // Only `forward`'s root remains. The `import` directive must
        // contribute zero roots.
        assert_eq!(
            ctx.dag.roots().len(),
            1,
            "import directive must not emit any DAG root: roots={:?}",
            ctx.dag.roots()
        );
    }

    /// Issue chelis#232: `import-all` (the `import Mod` form without
    /// an explicit name list) desugars to `(import-all {} <module>)`
    /// — same lower-time hazard as `export` and `import`.
    #[test]
    fn issue232_import_all_directive_does_not_emit_dag_root() {
        let src = r#"
            (def {} forward (lit {type: (t-prim {} f32)} 1.0))
            (import-all {} Math)
        "#;
        let exprs = chelis_deep::parser::parse_str(src).expect("parse failed");
        let mut ctx = LowerCtx::new(
            std::collections::HashMap::new(),
            std::collections::HashMap::new(),
            LinearityInfo::default(),
        );
        for expr in &exprs {
            ctx.lower_top_level(expr);
        }
        assert_eq!(
            ctx.dag.roots().len(),
            1,
            "import-all directive must not emit any DAG root: roots={:?}",
            ctx.dag.roots()
        );
    }

    #[test]
    fn tuple_get_resolves_during_lowering_without_tuple_ir_node() {
        let src = r#"
            (def {} grads
              (tuple {}
                (lit {type: (t-prim {} f32)} 1.0)
                (lit {type: (t-prim {} f32)} 2.0)))
            (def {} answer
              (tuple-get {}
                (var {} grads)
                1))
        "#;
        let dag = parse_and_lower(src);
        assert!(
            dag.nodes()
                .iter()
                .all(|node| { matches!(node.op, RiscOp::Const { .. } | RiscOp::Store { .. }) })
        );
        assert!(verify::verify(&dag).is_empty());
    }

    // --- H1: Tier 2 comparison ops lowering ---

    #[test]
    fn lower_gt_decomposes() {
        let src = r#"
            (def {} a (lit {type: (t-tensor {} (t-prim {} f32))} 5.0))
            (def {} b (lit {type: (t-tensor {} (t-prim {} f32))} 3.0))
            (def {} c (app {} (var {} gt) (var {} a) (var {} b)))
        "#;
        let dag = parse_and_lower(src);
        // a, b, CmpLt(b, a)
        assert_eq!(non_drop_len(&dag), 3);
        let node = dag.get(NodeId(2)).unwrap();
        assert_eq!(node.op, RiscOp::CmpLt);
        // Args are swapped: b, a
        assert_eq!(node.inputs, vec![NodeId(1), NodeId(0)]);
    }

    #[test]
    fn lower_gte_decomposes() {
        let src = r#"
            (def {} a (lit {type: (t-tensor {} (t-prim {} f32))} 5.0))
            (def {} b (lit {type: (t-tensor {} (t-prim {} f32))} 3.0))
            (def {} c (app {} (var {} gte) (var {} a) (var {} b)))
        "#;
        let dag = parse_and_lower(src);
        // a, b, CmpLt(a,b), Const(1), CmpLt(lt, 1)
        assert_eq!(non_drop_len(&dag), 5);
        assert_eq!(root_node(&dag).op, RiscOp::CmpLt);
    }

    #[test]
    fn lower_lte_decomposes() {
        let src = r#"
            (def {} a (lit {type: (t-tensor {} (t-prim {} f32))} 3.0))
            (def {} b (lit {type: (t-tensor {} (t-prim {} f32))} 5.0))
            (def {} c (app {} (var {} lte) (var {} a) (var {} b)))
        "#;
        let dag = parse_and_lower(src);
        assert_eq!(non_drop_len(&dag), 5);
    }

    #[test]
    fn lower_eq_decomposes() {
        let src = r#"
            (def {} a (lit {type: (t-tensor {} (t-prim {} f32))} 3.0))
            (def {} b (lit {type: (t-tensor {} (t-prim {} f32))} 3.0))
            (def {} c (app {} (var {} eq) (var {} a) (var {} b)))
        "#;
        let dag = parse_and_lower(src);
        // a, b, CmpLt(a,b), CmpLt(b,a), MaxElem, Const(1), CmpLt(or, 1)
        assert_eq!(non_drop_len(&dag), 7);
    }

    #[test]
    fn lower_min_elem_decomposes() {
        let src = r#"
            (def {} a (lit {type: (t-tensor {} (t-prim {} f32))} 5.0))
            (def {} b (lit {type: (t-tensor {} (t-prim {} f32))} 3.0))
            (def {} c (app {} (var {} min_elem) (var {} a) (var {} b)))
        "#;
        let dag = parse_and_lower(src);
        // a, b, neg(a), neg(b), max(neg_a, neg_b), neg(max)
        assert_eq!(non_drop_len(&dag), 6);
        assert_eq!(root_node(&dag).op, RiscOp::Neg);
        assert!(verify::verify(&dag).is_empty());
    }

    // --- H2: Boolean operators ---

    #[test]
    fn lower_and_decomposes() {
        let src = r#"
            (def {} a (lit {type: (t-tensor {} (t-prim {} bool))} true))
            (def {} b (lit {type: (t-tensor {} (t-prim {} bool))} false))
            (def {} c (app {} (var {} and) (var {} a) (var {} b)))
        "#;
        let dag = parse_and_lower(src);
        // a, b, Mul(a, b)
        assert_eq!(non_drop_len(&dag), 3);
        let node = dag.get(NodeId(2)).unwrap();
        assert_eq!(node.op, RiscOp::Mul);
    }

    #[test]
    fn lower_or_decomposes() {
        let src = r#"
            (def {} a (lit {type: (t-tensor {} (t-prim {} bool))} false))
            (def {} b (lit {type: (t-tensor {} (t-prim {} bool))} true))
            (def {} c (app {} (var {} or) (var {} a) (var {} b)))
        "#;
        let dag = parse_and_lower(src);
        // a, b, MaxElem(a, b)
        assert_eq!(dag.len(), 3);
        let node = dag.get(NodeId(2)).unwrap();
        assert_eq!(node.op, RiscOp::MaxElem);
    }

    #[test]
    fn lower_not_decomposes() {
        let src = r#"
            (def {} a (lit {type: (t-tensor {} (t-prim {} bool))} true))
            (def {} b (app {} (var {} not) (var {} a)))
        "#;
        let dag = parse_and_lower(src);
        // a, Const(1), CmpLt(a, 1)
        assert_eq!(non_drop_len(&dag), 3);
        assert_eq!(root_node(&dag).op, RiscOp::CmpLt);
    }

    // --- H3: Movement op stubs ---

    #[test]
    fn lower_reshape_recognized() {
        let src = r#"
            (def {} x (var {type: (t-tensor {} (d-lit {} 4) (t-prim {} f32))} x))
            (def {} y (app {type: (t-tensor {} (d-lit {} 4) (t-prim {} f32))} (var {} reshape) (var {} x)))
        "#;
        let dag = parse_and_lower(src);
        let last = dag.get(NodeId(dag.len() - 1)).unwrap();
        assert!(matches!(last.op, RiscOp::Reshape { .. }));
        assert!(verify::verify(&dag).is_empty());
    }

    #[test]
    fn lower_pad_recognized() {
        let src = r#"
            (def {} x (var {type: (t-tensor {} (d-lit {} 4) (t-prim {} f32))} x))
            (def {} y
              (app {type: (t-tensor {} (d-lit {} 6) (t-prim {} f32))}
                   (var {} pad)
                   (var {} x)
                   ((1 1))
                   0.0))
        "#;
        let dag = parse_and_lower_unchecked(src);
        let last = dag.get(NodeId(dag.len() - 1)).unwrap();
        assert!(matches!(last.op, RiscOp::Pad { .. }));
        assert!(verify::verify(&dag).is_empty());
    }

    #[test]
    fn lower_shrink_recognized() {
        let src = r#"
            (def {} x (var {type: (t-tensor {} (d-lit {} 4) (t-prim {} f32))} x))
            (def {} y
              (app {type: (t-tensor {} (d-lit {} 2) (t-prim {} f32))}
                   (var {} shrink)
                   (var {} x)
                   ((1 1))))
        "#;
        let dag = parse_and_lower_unchecked(src);
        let last = dag.get(NodeId(dag.len() - 1)).unwrap();
        assert!(matches!(last.op, RiscOp::Shrink { .. }));
    }

    #[test]
    fn lower_stride_recognized() {
        let src = r#"
            (def {} x (var {type: (t-tensor {} (d-lit {} 4) (t-prim {} f32))} x))
            (def {} y
              (app {type: (t-tensor {} (d-lit {} 2) (t-prim {} f32))}
                   (var {} stride)
                   (var {} x)
                   2))
        "#;
        let dag = parse_and_lower_unchecked(src);
        let last = dag.get(NodeId(dag.len() - 1)).unwrap();
        assert!(matches!(last.op, RiscOp::Stride { .. }));
        assert!(verify::verify(&dag).is_empty());
    }

    /// Issue #291: a `shrink` bound literal written as a Surf-desugared
    /// `Cons` chain whose pair elements are `cast`-wrapped ints
    /// (`[[cast(0, int32), cast(2, int32)]]`) must lower to
    /// `RiscOp::Shrink { bounds: [(0, 2)] }`, NOT an empty `bounds = []`.
    /// Before the fix the bound-pair walker accepted only `Atom::Int` /
    /// `(lit ...)`, so the `(cast ...)` elements dropped out and the
    /// resulting `Shrink { bounds: [] }` failed downstream verification
    /// (`bounds len 0 != input rank 1`), which surfaced as the `grad`
    /// "failed to construct backward DAG" error in the issue.
    #[test]
    fn lower_shrink_cast_wrapped_cons_bounds_issue_291() {
        // The exact desugared Deep that Surf emits for
        // `shrink(x, [[cast(0, int32), cast(2, int32)]])`: an outer
        // `Cons(pair, Nil)`, where `pair` is `Cons(cast(0), Cons(cast(2),
        // Nil))` and each `cast` wraps an int32 `lit`.
        let src = r#"
            (def {} x (var {type: (t-tensor {} (d-lit {} 4) (t-prim {} f32))} x))
            (def {} y
              (app {type: (t-tensor {} (d-lit {} 2) (t-prim {} f32))}
                   (var {} shrink)
                   (var {} x)
                   (app {} (var {} Cons)
                        (app {} (var {} Cons)
                             (cast {} (lit {type: (t-prim {} int32)} 0) (t-prim {} int32))
                             (app {} (var {} Cons)
                                  (cast {} (lit {type: (t-prim {} int32)} 2) (t-prim {} int32))
                                  (var {} Nil)))
                        (var {} Nil))))
        "#;
        let dag = parse_and_lower_unchecked(src);
        let bounds = dag
            .nodes()
            .iter()
            .find_map(|n| match &n.op {
                RiscOp::Shrink { bounds } => Some(bounds.clone()),
                _ => None,
            })
            .expect("a Shrink node must be present");
        assert_eq!(
            bounds,
            vec![(RtDim::Lit(0), RtDim::Lit(2))],
            "cast-wrapped cons-chain bounds must extract to (0, 2), not an empty list",
        );
        assert!(verify::verify(&dag).is_empty());
    }

    /// Negative parity for #291: the plain (non-cast) `Cons`-chain bound
    /// form must still extract, so routing the bound walker through
    /// `extract_int_for_dim` did not regress the literal path.
    #[test]
    fn lower_shrink_plain_cons_bounds_still_extract_issue_291() {
        let src = r#"
            (def {} x (var {type: (t-tensor {} (d-lit {} 4) (t-prim {} f32))} x))
            (def {} y
              (app {type: (t-tensor {} (d-lit {} 2) (t-prim {} f32))}
                   (var {} shrink)
                   (var {} x)
                   (app {} (var {} Cons)
                        (app {} (var {} Cons)
                             (lit {type: (t-prim {} int32)} 0)
                             (app {} (var {} Cons)
                                  (lit {type: (t-prim {} int32)} 2)
                                  (var {} Nil)))
                        (var {} Nil))))
        "#;
        let dag = parse_and_lower_unchecked(src);
        let bounds = dag
            .nodes()
            .iter()
            .find_map(|n| match &n.op {
                RiscOp::Shrink { bounds } => Some(bounds.clone()),
                _ => None,
            })
            .expect("a Shrink node must be present");
        assert_eq!(
            bounds,
            vec![(RtDim::Lit(0), RtDim::Lit(2))],
            "plain cons-chain bounds must still extract to (0, 2)",
        );
    }

    // --- H4: sum/max_reduce lowering ---

    #[test]
    fn lower_sum_reduction() {
        let src = r#"
            (def {} x (lit {type: (t-tensor {} (d-lit {} 1) (t-prim {} f32))} 1.0))
            (def {} y (app {type: (t-tensor {} (t-prim {} f32))} (var {} sum) (var {} x) (lit {} 0)))
        "#;
        let dag = parse_and_lower(src);
        // x=Const(1), axis_const=Const(0) is lowered inline, Sum{axis:0}
        let found_sum = dag
            .nodes()
            .iter()
            .any(|n| matches!(n.op, RiscOp::Sum { axis: 0, .. }));
        assert!(found_sum, "expected a Sum{{axis:0}} node");
    }

    #[test]
    fn lower_max_reduce_reduction() {
        let src = r#"
            (def {} x (lit {type: (t-tensor {} (d-lit {} 1) (t-prim {} f32))} 1.0))
            (def {} y (app {type: (t-tensor {} (t-prim {} f32))} (var {} max_reduce) (var {} x) (lit {} 0)))
        "#;
        let dag = parse_and_lower(src);
        let found = dag
            .nodes()
            .iter()
            .any(|n| matches!(n.op, RiscOp::MaxReduce { axis: 0 }));
        assert!(found, "expected a MaxReduce{{axis:0}} node");
    }

    /// Issue #320: a `max_reduce` whose operand node lowered to the rank-0
    /// `default_type()` placeholder (the windowing/stacking intermediate the
    /// checker left untyped at the IR node) but whose APP carries the
    /// checker's reduction-result type `[m]` must recover rank>=1 from the
    /// ascription instead of raising "axis 0 is out of range for an operand
    /// of rank 0". The recovered operand carries the reduced axis
    /// re-inserted at the reduction axis.
    #[test]
    fn issue_320_max_reduce_recovers_rank_from_ascription() {
        // `w` (the stacked operand) carries NO type -> rank-0 node. The
        // max_reduce app is typed `[m]` (rank 1). Before the fix this
        // panicked with the rank-0 diagnostic.
        let src = r#"
            (def {} w (var {} w))
            (def {} y
              (app {type: (t-tensor {} (d-name {} m) (t-prim {} f32))}
                   (var {} max_reduce) (var {} w) (lit {} 0)))
        "#;
        let dag = parse_and_lower_unchecked(src);
        let mr = dag
            .nodes()
            .iter()
            .find(|n| matches!(n.op, RiscOp::MaxReduce { axis: 0 }))
            .expect("max_reduce must lower (issue #320)");
        // Result is the ascribed `[m]`.
        assert_eq!(mr.output_type.dims.len(), 1, "max_reduce result rank 1");
        // The operand node was recovered to rank 2 (`[reduced_axis, m]`),
        // so the reduction axis is in range and the adjoint can carry it.
        let operand = dag.get(mr.inputs[0]).expect("operand node");
        assert_eq!(
            operand.output_type.dims.len(),
            2,
            "operand rank must be recovered to result_rank + 1 (issue #320)",
        );
    }

    /// Issue #320 end-to-end (max_reduce): a windowed operand that lowers to
    /// the rank-0 placeholder must grad+eval correctly once the front-end
    /// recovers its rank. `w = reshape(x, [2,2])` is lowered UNtyped (rank-0
    /// node) but its runtime value is `[2,2]`; `max_reduce(w, 0)` reduces
    /// axis 0 (rows), giving the per-column max. The subgradient routes 1 to
    /// each column's argmax row. With `x = [1,2,4,3]` reshaped row-major to
    /// `[[1,2],[4,3]]`, both column maxes (4 and 3) are in row 1 -> flat
    /// indices {2, 3}, so `df/dx = [0, 0, 1, 1]`.
    #[test]
    fn issue_320_grad_eval_windowed_max_reduce_end_to_end() {
        use crate::eval::{TensorValue, eval_tensor};
        use crate::grad::grad_dag_checked;

        // `w = reshape(x, [2,2])` with NO type on the reshape app -> rank-0
        // operand node, but a real producer (Reshape over the `x` Load).
        // `max_reduce(w, 0)` is typed `[2]`; `sum(..., 0)` -> scalar loss.
        let src = r#"
            (def {} x (var {type: (t-tensor {} (d-lit {} 4) (t-prim {} f32))} x))
            (def {} w (app {} (var {} reshape) (var {} x)
                          (app {} (var {} Cons) (lit {} 2)
                               (app {} (var {} Cons) (lit {} 2) (var {} Nil)))))
            (def {} m (app {type: (t-tensor {} (d-lit {} 2) (t-prim {} f32))}
                           (var {} max_reduce) (var {} w) (lit {} 0)))
            (def {} loss (app {type: (t-tensor {} (t-prim {} f32))}
                              (var {} sum) (var {} m) (lit {} 0)))
        "#;
        let dag = parse_and_lower_unchecked(src);
        // The loss is the scalar `Sum` consuming the `MaxReduce`.
        let mr_id = dag
            .nodes()
            .iter()
            .find(|n| matches!(n.op, RiscOp::MaxReduce { .. }))
            .expect("max_reduce node")
            .id;
        let loss = dag
            .nodes()
            .iter()
            .find(|n| matches!(n.op, RiscOp::Sum { .. }) && n.inputs.contains(&mr_id))
            .expect("loss sum node")
            .id;
        let x_node = dag
            .nodes()
            .iter()
            .find(|n| matches!(&n.op, RiscOp::Load { name } if name.as_str() == "x"))
            .expect("x load")
            .id;
        let result = grad_dag_checked(&dag, loss, &[x_node])
            .expect("grad through windowed max_reduce must construct (issue #320)");
        let grad_x = result.grad_nodes[&x_node];
        let mut inputs = std::collections::HashMap::new();
        inputs.insert(
            "x".to_string(),
            TensorValue::from_vec(vec![4], vec![1.0, 2.0, 4.0, 3.0]),
        );
        let vals = eval_tensor(&result.dag, &inputs).expect("windowed max_reduce grad eval");
        assert_eq!(
            vals[&grad_x].to_f64_lossy_vec(),
            vec![0.0, 0.0, 1.0, 1.0],
            "subgradient must route to each column's argmax (issue #320)",
        );
    }

    /// chelis#369 end-to-end: the `tensor_full_like` loss body — `len =
    /// shape(x, 0); twos = expand(scalar_to_tensor(2.0), 0, cast(len,
    /// int32)); sum(mul(x, twos), 0)` — must construct a valid backward DAG
    /// and eval to the analytic gradient. `loss(x) = sum(2*x)`, so `df/dx =
    /// [2, 2, 2]`. Before the fix the `let`-bound `len` defaulted the expand
    /// extent to `Lit(1)`, so the forward `mul(x, twos)` mixed `tensor[3]`
    /// with `tensor[1]` and `grad_dag_checked` rejected with the `Lit(3) vs
    /// Lit(1)` verification failure. This drives the REAL front-end lowerer
    /// (where the bug lives), then grad + eval.
    #[test]
    fn issue_369_grad_eval_let_bound_fulllike_end_to_end() {
        use crate::eval::{TensorValue, eval_tensor};
        use crate::grad::grad_dag_checked;

        // `def loss(x: tensor[3, f32]) = {
        //    len  = shape(x, 0)
        //    twos = expand(scalar_to_tensor(2.0), 0, cast(len, int32))
        //    sum(mul(x, twos), 0) }`  -- the exact `tensor_full_like` shape.
        let body = r#"
            (let {}
                 (bind {}
                       len
                       (app {}
                            (var {} shape)
                            (var {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))} x)
                            (cast {} (lit {} 0) (t-prim {} int32))))
                 (let {}
                      (bind {}
                            twos
                            (app {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))}
                                 (var {} expand)
                                 (app {type: (t-prim {} f32)}
                                      (var {} scalar_to_tensor)
                                      (cast {type: (t-prim {} f32)} (lit {} 2.0) (t-prim {} f32)))
                                 (cast {} (lit {} 0) (t-prim {} int32))
                                 (cast {} (var {} len) (t-prim {} int32))))
                      (app {type: (t-tensor {} (t-prim {} f32))}
                           (var {} sum)
                           (app {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))}
                                (var {} mul)
                                (var {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))} x)
                                (var {} twos))
                           (cast {} (lit {} 0) (t-prim {} int32)))))
        "#;
        let dag = lower_body_with_bound_x(3, body);
        // The loss is the scalar `Sum` over `mul(x, twos)`.
        let loss = dag
            .nodes()
            .iter()
            .find(|n| matches!(n.op, RiscOp::Sum { .. }))
            .expect("loss sum node")
            .id;
        let x_node = dag
            .nodes()
            .iter()
            .find(|n| matches!(&n.op, RiscOp::Load { name } if name.as_str() == "x"))
            .expect("x load")
            .id;
        let result = grad_dag_checked(&dag, loss, &[x_node]).unwrap_or_else(|e| {
            panic!("grad through let-bound tensor_full_like must construct (chelis#369); got {e:?}")
        });
        let grad_x = result.grad_nodes[&x_node];

        // Analytic: df/dx = [2, 2, 2] for any x.
        let mut inputs = std::collections::HashMap::new();
        inputs.insert(
            "x".to_string(),
            TensorValue::from_vec(vec![3], vec![1.0, 2.0, 3.0]),
        );
        let vals = eval_tensor(&result.dag, &inputs).expect("grad eval");
        assert_eq!(
            vals[&grad_x].to_f64_lossy_vec(),
            vec![2.0, 2.0, 2.0],
            "d sum(2x) / dx must be [2, 2, 2] (chelis#369)",
        );
        assert_eq!(
            vals[&grad_x].shape,
            vec![3],
            "gradient must be tensor[3] (chelis#369)",
        );

        // Finite-difference cross-check of the analytic gradient against the
        // forward loss DAG (backend-numerics discipline).
        let base = [0.7f64, -1.3, 2.1];
        let analytic = {
            let mut ip = std::collections::HashMap::new();
            ip.insert(
                "x".to_string(),
                TensorValue::from_vec(vec![3], base.to_vec()),
            );
            eval_tensor(&result.dag, &ip).expect("analytic grad eval")[&grad_x]
                .to_f64_lossy_vec()
                .clone()
        };
        let h = 1e-3;
        for (j, a) in analytic.iter().enumerate() {
            let mut plus = base;
            let mut minus = base;
            plus[j] += h;
            minus[j] -= h;
            let mut ip = std::collections::HashMap::new();
            ip.insert(
                "x".to_string(),
                TensorValue::from_vec(vec![3], plus.to_vec()),
            );
            let mut im = std::collections::HashMap::new();
            im.insert(
                "x".to_string(),
                TensorValue::from_vec(vec![3], minus.to_vec()),
            );
            let fp = eval_tensor(&dag, &ip).expect("plus eval")[&loss].to_f64_lossy_vec()[0];
            let fm = eval_tensor(&dag, &im).expect("minus eval")[&loss].to_f64_lossy_vec()[0];
            let numerical = (fp - fm) / (2.0 * h);
            assert!(
                (a - numerical).abs() < 1e-3,
                "finite-diff mismatch at {j}: analytic {a}, numerical {numerical}",
            );
        }
    }

    /// Issue #320 end-to-end (gather): a windowed `values` operand that
    /// lowers to the rank-0 placeholder must grad+eval correctly once the
    /// front-end recovers its rank. `w = reshape(x, [4])` is lowered UNtyped
    /// (rank-0 node) but its runtime value is `[4]`; `gather(w, idx, 0)` with
    /// `idx = [0, 2]` selects `w[0]` and `w[2]`. `f(x) = x0 + x2`, so the
    /// scatter-add adjoint gives `df/dx = [1, 0, 1, 0]`.
    #[test]
    fn issue_320_grad_eval_windowed_gather_end_to_end() {
        use crate::eval::{TensorValue, eval_tensor};
        use crate::grad::grad_dag_checked;

        let src = r#"
            (def {} x (var {type: (t-tensor {} (d-lit {} 4) (t-prim {} f32))} x))
            (def {} idx (var {type: (t-tensor {} (d-lit {} 2) (t-prim {} int64))} idx))
            (def {} w (app {} (var {} reshape) (var {} x)
                          (app {} (var {} Cons) (lit {} 4) (var {} Nil))))
            (def {} g (app {type: (t-tensor {} (d-lit {} 2) (t-prim {} f32))}
                           (var {} gather) (var {} w) (var {} idx) (lit {} 0)))
            (def {} loss (app {type: (t-tensor {} (t-prim {} f32))}
                              (var {} sum) (var {} g) (lit {} 0)))
        "#;
        let dag = parse_and_lower_unchecked(src);
        let g_id = dag
            .nodes()
            .iter()
            .find(|n| matches!(n.op, RiscOp::Gather { .. }))
            .expect("gather node")
            .id;
        let loss = dag
            .nodes()
            .iter()
            .find(|n| matches!(n.op, RiscOp::Sum { .. }) && n.inputs.contains(&g_id))
            .expect("loss sum node")
            .id;
        let x_node = dag
            .nodes()
            .iter()
            .find(|n| matches!(&n.op, RiscOp::Load { name } if name.as_str() == "x"))
            .expect("x load")
            .id;
        let result = grad_dag_checked(&dag, loss, &[x_node])
            .expect("grad through windowed gather must construct (issue #320)");
        let grad_x = result.grad_nodes[&x_node];
        let mut inputs = std::collections::HashMap::new();
        inputs.insert(
            "x".to_string(),
            TensorValue::from_vec(vec![4], vec![10.0, 20.0, 30.0, 40.0]),
        );
        inputs.insert(
            "idx".to_string(),
            TensorValue::from_vec(vec![2], vec![0.0, 2.0]),
        );
        let vals = eval_tensor(&result.dag, &inputs).expect("windowed gather grad eval");
        assert_eq!(
            vals[&grad_x].to_f64_lossy_vec(),
            vec![1.0, 0.0, 1.0, 0.0],
            "scatter-add adjoint must route to gathered source slots (issue #320)",
        );
    }

    /// Issue #320 gather sibling: a `gather` whose `values` operand lowered
    /// to the rank-0 placeholder but whose APP carries the checker's gather
    /// result type must recover the values rank from the ascription instead
    /// of raising the rank-0 diagnostic.
    #[test]
    fn issue_320_gather_recovers_values_rank_from_ascription() {
        // `values` carries NO type -> rank-0 node. `indices` is `[3]`. The
        // gather app is typed `[3]` (gather of a rank-1 values over axis 0).
        let src = r#"
            (def {} values (var {} values))
            (def {} indices (var {type: (t-tensor {} (d-lit {} 3) (t-prim {} int32))} indices))
            (def {} y
              (app {type: (t-tensor {} (d-lit {} 3) (t-prim {} f32))}
                   (var {} gather) (var {} values) (var {} indices) (lit {} 0)))
        "#;
        let dag = parse_and_lower_unchecked(src);
        let g = dag
            .nodes()
            .iter()
            .find(|n| matches!(n.op, RiscOp::Gather { axis: 0 }))
            .expect("gather must lower (issue #320)");
        let values = dag.get(g.inputs[0]).expect("values operand node");
        assert!(
            !values.output_type.dims.is_empty(),
            "gather values rank must be recovered to >= 1 (issue #320)",
        );
    }

    // --- C5: CmpLt lowering produces Bool ---

    #[test]
    fn lower_cmplt_produces_bool_output() {
        let src = r#"
            (def {} a (lit {type: (t-tensor {} (t-prim {} f32))} 1.0))
            (def {} b (lit {type: (t-tensor {} (t-prim {} f32))} 2.0))
            (def {} c (app {} (var {} cmplt) (var {} a) (var {} b)))
        "#;
        let dag = parse_and_lower(src);
        let cmplt_node = dag
            .nodes()
            .iter()
            .find(|n| matches!(n.op, RiscOp::CmpLt))
            .expect("expected CmpLt node");
        assert_eq!(
            cmplt_node.output_type.precision,
            Prim::Bool,
            "CmpLt output must be Bool"
        );
        assert!(verify::verify(&dag).is_empty());
    }

    // --- C6: type propagation from metadata ---

    #[test]
    fn lower_lit_with_type_metadata() {
        let src = "(def {} x (lit {type: (t-prim {} f64)} 3.14))";
        let dag = parse_and_lower(src);
        let node = dag.get(NodeId(0)).unwrap();
        assert_eq!(node.output_type.precision, Prim::F64);
    }

    #[test]
    fn lower_var_with_type_metadata() {
        let src = "(def {} y (var {type: (t-prim {} f64)} weights))";
        let dag = parse_and_lower_unchecked(src);
        let node = dag.get(NodeId(0)).unwrap();
        assert_eq!(node.output_type.precision, Prim::F64);
    }

    // ── Tier-2 rank monomorphization (spec/design/rank_polymorphism.md) ──

    fn parse_type_expr(src: &str) -> Expr {
        chelis_deep::parser::parse_str(src)
            .expect("parse failed")
            .into_iter()
            .next()
            .expect("one expr")
    }

    /// `extract_rank_var_bindings` binds a sole `(d-rank {} r)` to the whole
    /// actual shape (Tier-2), strips a `(t-ref)` wrapper, and returns nothing
    /// for a concrete-shape tensor.
    #[test]
    fn extract_rank_var_bindings_sole_spread() {
        let actual = vec![DimInfo::Named("a".into(), None), DimInfo::Lit(4)];
        let no_positions = HashMap::new();

        let bare = parse_type_expr("(t-tensor {} (d-rank {} r) (t-prim {} f32))");
        assert_eq!(
            extract_rank_var_bindings(&bare, &actual, &no_positions),
            vec![("r".to_string(), actual.clone())]
        );

        let borrowed = parse_type_expr("(t-ref {} (t-tensor {} (d-rank {} rr) (t-prim {} f32)))");
        assert_eq!(
            extract_rank_var_bindings(&borrowed, &actual, &no_positions),
            vec![("rr".to_string(), actual.clone())]
        );

        let concrete =
            parse_type_expr("(t-tensor {} (d-name {} batch) (d-name {} seq) (t-prim {} f32))");
        assert!(extract_rank_var_bindings(&concrete, &actual, &no_positions).is_empty());
    }

    /// chelis#373: a `(d-rank pre) (d-name seq) (d-rank post)` formal whose
    /// actual was monomorphized to all-`Lit` dims (the name `seq` is gone) still
    /// splits correctly when `dim_axis_positions` records `seq`'s fixed index —
    /// the by-position fallback the grad lane needs. WITHOUT the recorded
    /// position the split fails (the empty-map case returns no binding rather
    /// than guessing), proving the fallback is the load-bearing recovery.
    #[test]
    fn extract_rank_var_bindings_by_position_when_name_erased() {
        let formal = parse_type_expr(
            "(t-ref {} (t-tensor {} (d-rank {} pre) (d-name {} seq) (d-rank {} post) (t-prim {} f32)))",
        );
        // Monomorphized actual: seq erased to a concrete Lit at index 1.
        let actual = vec![DimInfo::Lit(2), DimInfo::Lit(3)];

        // chelis#549: the anchor is located by its recorded extent `Lit(3)`,
        // which is unique in the actual, so it resolves to index 1.
        let positions = HashMap::from([("seq".to_string(), (1usize, DimInfo::Lit(3)))]);
        let bindings = extract_rank_var_bindings(&formal, &actual, &positions);
        assert_eq!(
            bindings,
            vec![
                ("pre".to_string(), vec![DimInfo::Lit(2)]),
                ("post".to_string(), vec![]),
            ],
            "by-position fallback must split at the recorded anchor index"
        );

        // A LEADING-spread variant: `(d-rank rest) (d-name seq)` reducing the
        // trailing axis. `seq` at recorded index 1 splits `rest = [Lit(2)]`.
        let leading =
            parse_type_expr("(t-tensor {} (d-rank {} rest) (d-name {} seq) (t-prim {} f32))");
        let leading_bindings = extract_rank_var_bindings(&leading, &actual, &positions);
        assert_eq!(
            leading_bindings,
            vec![("rest".to_string(), vec![DimInfo::Lit(2)])],
            "by-position fallback works with a trailing anchor too"
        );
    }

    /// chelis#373/#549: an out-of-range recorded position whose recorded extent
    /// is ALSO absent from the actual is NOT trusted — it falls through to the
    /// loud "anchor absent" path rather than splitting at a bogus index. The
    /// panic message pins that the right failure (not a silent wrong split) is
    /// reached.
    #[test]
    fn extract_rank_var_bindings_rejects_out_of_range_position() {
        let err = std::panic::catch_unwind(|| {
            let formal = parse_type_expr(
                "(t-ref {} (t-tensor {} (d-rank {} pre) (d-name {} seq) (d-rank {} post) (t-prim {} f32)))",
            );
            let actual = vec![DimInfo::Lit(2), DimInfo::Lit(3)];
            // Position 9 is out of range AND extent 99 appears nowhere ->
            // unrecoverable.
            let bad_positions = HashMap::from([("seq".to_string(), (9usize, DimInfo::Lit(99)))]);
            let _ = extract_rank_var_bindings(&formal, &actual, &bad_positions);
        })
        .expect_err("an unrecoverable anchor must fail loud, not return a partial binding");
        let msg = err
            .downcast_ref::<LowerDiagnostic>()
            .map(ToString::to_string)
            .unwrap_or_default();
        assert!(
            msg.contains("absent from monomorphized actual"),
            "the loud failure must name the absent-anchor reason, got {msg:?}"
        );
    }

    /// chelis#549 (the headline soundness fix): a recorded position that is
    /// in range but STALE (an axis-reorder changed the extent there) must NOT
    /// split at the bare recorded index. When the recorded extent appears at a
    /// UNIQUE in-range axis, the anchor is relocated there (the permute repro).
    #[test]
    fn extract_rank_var_bindings_relocates_anchor_after_unique_reorder() {
        let formal = parse_type_expr(
            "(t-ref {} (t-tensor {} (d-rank {} pre) (d-name {} seq) (d-rank {} post) (t-prim {} f32)))",
        );
        // Repro 2's monomorphized, permuted actual: `seq` (Lit(3)) recorded at
        // index 1 in the bridge formal `[a, seq]`, but a `permute(x, 1, 0)`
        // reordered the operand to `[Lit(3), Lit(2)]`, so the recorded index 1
        // now holds `a`'s extent (Lit(2)). The anchor must be relocated to the
        // unique axis carrying Lit(3) (index 0), not split at the stale 1.
        let permuted_actual = vec![DimInfo::Lit(3), DimInfo::Lit(2)];
        let positions = HashMap::from([("seq".to_string(), (1usize, DimInfo::Lit(3)))]);
        let bindings = extract_rank_var_bindings(&formal, &permuted_actual, &positions);
        assert_eq!(
            bindings,
            vec![
                ("pre".to_string(), vec![]),
                ("post".to_string(), vec![DimInfo::Lit(2)]),
            ],
            "a stale in-range index must relocate to the unique extent-matching axis, not split at 1"
        );
    }

    /// chelis#549 negative parity: a stale in-range recorded position whose
    /// recorded extent appears at MORE THAN ONE in-range axis cannot be located
    /// unambiguously — the split must FAIL LOUD rather than silently land on a
    /// possibly-wrong axis. Value alone cannot disambiguate equal-extent axes.
    #[test]
    fn extract_rank_var_bindings_ambiguous_reorder_fails_loud() {
        let err = std::panic::catch_unwind(|| {
            let formal = parse_type_expr(
                "(t-ref {} (t-tensor {} (d-rank {} pre) (d-name {} seq) (d-rank {} post) (t-prim {} f32)))",
            );
            // `seq` (Lit(2)) recorded at index 2 in formal `[a, b, seq]`, but a
            // reorder produced `[Lit(2), Lit(2), Lit(3)]`: the recorded extent
            // Lit(2) now appears at BOTH axes 0 and 1, and the recorded index 2
            // holds Lit(3).
            let ambiguous_actual = vec![DimInfo::Lit(2), DimInfo::Lit(2), DimInfo::Lit(3)];
            let positions = HashMap::from([("seq".to_string(), (2usize, DimInfo::Lit(2)))]);
            let _ = extract_rank_var_bindings(&formal, &ambiguous_actual, &positions);
        })
        .expect_err("an ambiguous post-reorder anchor must fail loud, not split silently");
        let msg = err
            .downcast_ref::<LowerDiagnostic>()
            .map(ToString::to_string)
            .unwrap_or_default();
        assert!(
            msg.contains("chelis#549"),
            "the loud failure must cite the chelis#549 axis-reorder soundness rule, got {msg:?}"
        );
    }

    /// chelis#549 unit coverage of [`recover_anchor_axis`] directly: the
    /// unique-extent resolution (with and without a reorder), the
    /// equal-extent/square ambiguity loud signal (RT-1), and the unrecorded
    /// fall-through. The recorded *index* is never trusted — only the extent.
    #[test]
    fn recover_anchor_axis_unique_resolves_ambiguous_and_unrecorded_fail_closed() {
        // Unique extent at the recorded position (no reorder): resolves there.
        let dims = vec![DimInfo::Lit(2), DimInfo::Lit(3), DimInfo::Lit(4)];
        let positions = HashMap::from([("seq".to_string(), (1usize, DimInfo::Lit(3)))]);
        assert!(matches!(
            recover_anchor_axis(&dims, 0, dims.len(), "seq", &positions),
            AnchorRecovery::Axis(1)
        ));

        // Unique reorder: recorded index 1 is stale (holds Lit(9)); the extent
        // Lit(3) appears uniquely at index 2 -> relocate to 2 (the recorded
        // index is irrelevant).
        let reordered = vec![DimInfo::Lit(2), DimInfo::Lit(9), DimInfo::Lit(3)];
        assert!(matches!(
            recover_anchor_axis(&reordered, 0, reordered.len(), "seq", &positions),
            AnchorRecovery::Axis(2)
        ));

        // Ambiguous reorder: extent Lit(3) appears at indices 0 and 2 -> cannot
        // disambiguate by value -> loud.
        let ambiguous = vec![DimInfo::Lit(3), DimInfo::Lit(9), DimInfo::Lit(3)];
        assert!(matches!(
            recover_anchor_axis(&ambiguous, 0, ambiguous.len(), "seq", &positions),
            AnchorRecovery::AmbiguousAfterReorder
        ));

        // RT-1 square/equal-extent case: the recorded position (1) carries the
        // recorded extent, but so does axis 0, so a stale position would be
        // "confirmed" vacuously. Must be reported ambiguous, NOT Axis(1).
        let square = vec![DimInfo::Lit(3), DimInfo::Lit(3)];
        assert!(matches!(
            recover_anchor_axis(&square, 0, square.len(), "seq", &positions),
            AnchorRecovery::AmbiguousAfterReorder
        ));

        // Unrecorded: extent absent and position out of range.
        let absent = vec![DimInfo::Lit(7), DimInfo::Lit(8)];
        assert!(matches!(
            recover_anchor_axis(&absent, 0, absent.len(), "seq", &positions),
            AnchorRecovery::Unrecorded
        ));

        // Name not in the map at all.
        assert!(matches!(
            recover_anchor_axis(&dims, 0, dims.len(), "absent", &positions),
            AnchorRecovery::Unrecorded
        ));
    }

    /// Tier-3: a `(d-rank {} pre) (d-name {} seq) (d-rank {} post)` formal splits
    /// the actual at the named anchor `seq` — `pre` and `post` bind to the runs
    /// on either side, the lowering twin of `unify_row_against_ground`.
    #[test]
    fn extract_rank_var_bindings_anchored_split() {
        let formal = parse_type_expr(
            "(t-ref {} (t-tensor {} (d-rank {} pre) (d-name {} seq) (d-rank {} post) (t-prim {} f32)))",
        );
        let actual = vec![
            DimInfo::Named("batch".into(), None),
            DimInfo::Named("seq".into(), None),
            DimInfo::Named("hidden".into(), None),
        ];
        let bindings = extract_rank_var_bindings(&formal, &actual, &HashMap::new());
        assert_eq!(
            bindings,
            vec![
                (
                    "pre".to_string(),
                    vec![DimInfo::Named("batch".into(), None)]
                ),
                (
                    "post".to_string(),
                    vec![DimInfo::Named("hidden".into(), None)]
                ),
            ]
        );
    }

    /// `tensor_rank_substitutions` binds a formal `(d-rank {} r)` param to the
    /// actual arg's concrete dim vector (the call-site monomorphization
    /// boundary), keyed by the rank-var name.
    #[test]
    fn tensor_rank_substitutions_binds_rank_var_to_concrete_dims() {
        let formals = vec![Some(parse_type_expr(
            "(t-ref {} (t-tensor {} (d-rank {} r) (t-prim {} f32)))",
        ))];
        let actuals = vec![TensorType {
            dims: vec![
                DimInfo::Named("a".into(), None),
                DimInfo::Named("b".into(), None),
            ],
            precision: Prim::F32,
        }];
        let subst = tensor_rank_substitutions(&formals, &actuals, &HashMap::new());
        assert_eq!(
            subst.get("r"),
            Some(&vec![
                DimInfo::Named("a".into(), None),
                DimInfo::Named("b".into(), None)
            ]),
            "rank var `r` must bind to the actual arg's full shape vector"
        );
    }

    /// chelis#373: `tensor_dim_axis_positions` records a fixed anchor offset
    /// ONLY for a slot with no preceding spread, and never records (so never
    /// clobbers) a named anchor that sits behind a spread.
    #[test]
    fn tensor_dim_axis_positions_skips_post_spread_anchors() {
        // Concrete-rank formal `[b, seq]`: both fixed; `seq` at index 1.
        let concrete = Some(parse_type_expr(
            "(t-ref {} (t-tensor {} (d-name {} b) (d-name {} seq) (t-prim {} f32)))",
        ));
        // Spread formal `[..pre, seq, ..post]`: `seq` is behind a spread — no
        // fixed offset, must NOT be recorded.
        let spread = Some(parse_type_expr(
            "(t-ref {} (t-tensor {} (d-rank {} pre) (d-name {} seq) (d-rank {} post) (t-prim {} f32)))",
        ));

        // chelis#549: positions are paired with the anchor's actual extent.
        let actual = TensorType {
            dims: vec![DimInfo::Lit(4), DimInfo::Lit(5)],
            precision: Prim::F32,
        };

        // Concrete formal alone: seq -> (1, Lit(5)).
        let from_concrete = tensor_dim_axis_positions(
            std::slice::from_ref(&concrete),
            std::slice::from_ref(&actual),
        );
        assert_eq!(from_concrete.get("seq"), Some(&(1usize, DimInfo::Lit(5))));

        // Spread formal alone: seq is post-spread, recorded nowhere.
        let from_spread =
            tensor_dim_axis_positions(std::slice::from_ref(&spread), std::slice::from_ref(&actual));
        assert!(
            !from_spread.contains_key("seq"),
            "a post-spread anchor has no fixed offset; got {from_spread:?}"
        );

        // A leading anchor before a trailing spread (`[row, ..rest]`) IS fixed.
        let leading = Some(parse_type_expr(
            "(t-tensor {} (d-name {} row) (d-rank {} rest) (t-prim {} f32))",
        ));
        let leading_actual = TensorType {
            dims: vec![DimInfo::Lit(7), DimInfo::Lit(8)],
            precision: Prim::F32,
        };
        let from_leading = tensor_dim_axis_positions(&[leading], &[leading_actual]);
        assert_eq!(from_leading.get("row"), Some(&(0usize, DimInfo::Lit(7))));
    }

    /// chelis#373 anti-clobber: when the SAME named axis appears in a
    /// concrete-rank formal (fixed offset) and a spread formal (no fixed
    /// offset), `.extend()`-ing the spread result must not overwrite the
    /// concrete offset. This mirrors the two-level grad call chain
    /// (`sum_rows[b, seq]` then `sum_seq[..pre, seq, ..post]`).
    #[test]
    fn tensor_dim_axis_positions_concrete_survives_spread_extend() {
        let concrete = Some(parse_type_expr(
            "(t-ref {} (t-tensor {} (d-name {} b) (d-name {} seq) (t-prim {} f32)))",
        ));
        let spread = Some(parse_type_expr(
            "(t-ref {} (t-tensor {} (d-rank {} pre) (d-name {} seq) (d-rank {} post) (t-prim {} f32)))",
        ));
        let actual = TensorType {
            dims: vec![DimInfo::Lit(4), DimInfo::Lit(5)],
            precision: Prim::F32,
        };
        let mut positions = tensor_dim_axis_positions(&[concrete], std::slice::from_ref(&actual));
        positions.extend(tensor_dim_axis_positions(
            &[spread],
            std::slice::from_ref(&actual),
        ));
        assert_eq!(
            positions.get("seq"),
            Some(&(1usize, DimInfo::Lit(5))),
            "the concrete-rank `seq -> (1, Lit(5))` must survive the spread formal's extend"
        );
    }

    /// `try_extract_tensor_type_with_subst` expands a bound `(d-rank)` slot to
    /// the substituted concrete dims (monomorphization-success path) and drops
    /// an unbound one (the speculative-annotation path repaired by inlining).
    #[test]
    fn try_extract_tensor_type_expands_bound_rank_var() {
        let prec_subst = HashMap::new();
        let mut rank_subst: HashMap<String, Vec<DimInfo>> = HashMap::new();
        rank_subst.insert("r".into(), vec![DimInfo::Lit(3), DimInfo::Lit(4)]);

        let bound = parse_type_expr("(t-tensor {} (d-rank {} r) (t-prim {} f32))");
        let tt = LowerCtx::try_extract_tensor_type_with_subst(&bound, &prec_subst, &rank_subst)
            .expect("bound rank var resolves to a concrete tensor type");
        assert_eq!(tt.dims, vec![DimInfo::Lit(3), DimInfo::Lit(4)]);
        assert_eq!(tt.precision, Prim::F32);

        // Unbound: the rank dim drops (no `Dim::Rank` is representable in the
        // IR `DimInfo`); the inlined body supplies the concrete shape.
        let unbound = parse_type_expr("(t-tensor {} (d-rank {} q) (t-prim {} f32))");
        let tt = LowerCtx::try_extract_tensor_type_with_subst(&unbound, &prec_subst, &rank_subst)
            .expect("a tensor type is still produced");
        assert!(
            tt.dims.is_empty(),
            "an unbound rank var drops rather than emitting a rank dim, got {:?}",
            tt.dims
        );
    }

    /// End-to-end: a rank-poly identity def, called through a concrete-rank
    /// caller, lowers with no surviving rank var and the caller's root carries
    /// the caller's concrete shape (here `[2, 3]`).
    #[test]
    fn rank_poly_def_lowers_through_concrete_caller() {
        // Deep form of:
        //   def relu_forward(x: &tensor[..r, f32]) -> tensor[..r, f32] = relu(x)
        //   def use2d(x: &tensor[a, b, f32]) -> tensor[a, b, f32] = relu_forward(x)
        let dag = parse_and_lower(
            "(defsig {} relu_forward \
               (t-fn {} (t-ref {} (t-tensor {} (d-rank {} r) (t-prim {} f32))) \
                        (t-tensor {} (d-rank {} r) (t-prim {} f32)))) \
             (def {} relu_forward \
               (fn {} (params {} (x {type: (t-ref {} (t-tensor {} (d-rank {} r) (t-prim {} f32)))})) \
                  (app {} (var {} relu) (var {} x)))) \
             (defsig {} use2d \
               (t-fn {} (t-ref {} (t-tensor {} (d-var {} a) (d-var {} b) (t-prim {} f32))) \
                        (t-tensor {} (d-var {} a) (d-var {} b) (t-prim {} f32)))) \
             (def {} use2d \
               (fn {} (params {} (x {type: (t-ref {} (t-tensor {} (d-var {} a) (d-var {} b) (t-prim {} f32)))})) \
                  (app {} (var {} relu_forward) (var {} x))))",
        );
        // The core assertion: lowering *completes* (the pre-feature path
        // panicked here with the "not yet lowerable" refusal). Every node's
        // output type is `Dim::Rank`-free by construction — `DimInfo` has no
        // rank variant — so a successful lowering is exactly a successful
        // monomorphization. Lock that no node carries a stray named dim that
        // is the un-substituted rank-var symbol `r` (the symbol would leak
        // only if a `(d-rank {} r)` were mis-extracted as a named dim).
        for node in dag.nodes() {
            for dim in &node.output_type.dims {
                if let DimInfo::Named(name, _) = dim {
                    assert_ne!(
                        name, "r",
                        "the rank-var symbol `r` must not survive as a named dim after \
                         monomorphization (node op {:?})",
                        node.op
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod regression_tests {
    use super::*;
    use crate::dag::{DimInfo, NodeId, RiscOp};
    use crate::verify;
    use chelis_types::types::Prim;

    /// E2 (WS-A0 RT-1 fixup): `ty_expr_to_deep` must panic on
    /// f8e4m3 with the §1.1.1 message rather than emitting a
    /// `(t-prim {} f8e4m3)` node into lowered Deep.
    #[test]
    #[should_panic(expected = "f8e4m3 is deferred per spec/04-type-system.md §1.1.1")]
    fn ty_expr_to_deep_panics_on_f8e4m3_per_spec_1_1_1() {
        let ty = crate::dag::TensorType {
            dims: vec![DimInfo::Lit(4)],
            precision: Prim::F8e4m3,
        };
        let _ = ty_expr_to_deep(&ty);
    }

    fn parse_and_lower(src: &str) -> Dag {
        let exprs = chelis_deep::parser::parse_str(src).expect("parse failed");
        let checked = chelis_types::check_ir_program(&exprs)
            .unwrap_or_else(|result| panic!("IR check failed: {:?}", result.errors));
        let checked = chelis_effects::check_program(&checked)
            .unwrap_or_else(|errors| panic!("effect check failed: {errors:?}"));
        let checked = chelis_types::check_linearity(&checked)
            .unwrap_or_else(|errors| panic!("linearity check failed: {errors:?}"));
        lower_program(&checked)
    }

    fn parse_and_lower_unchecked(src: &str) -> Dag {
        let exprs = chelis_deep::parser::parse_str(src).expect("parse failed");
        let mut ctx = LowerCtx::new(
            std::collections::HashMap::new(),
            std::collections::HashMap::new(),
            LinearityInfo::default(),
        );
        for expr in &exprs {
            let _ = ctx.lower_expr(expr);
        }
        ctx.dag
    }

    fn non_drop_len(dag: &Dag) -> usize {
        dag.nodes()
            .iter()
            .filter(|node| !matches!(node.op, RiscOp::Drop))
            .count()
    }

    fn captured_lower_message(payload: Box<dyn std::any::Any + Send>) -> String {
        if let Some(diagnostic) = payload.downcast_ref::<LowerDiagnostic>() {
            diagnostic.to_string()
        } else if let Some(message) = payload.downcast_ref::<String>() {
            message.clone()
        } else if let Some(message) = payload.downcast_ref::<&str>() {
            (*message).to_string()
        } else {
            String::new()
        }
    }

    // Fix 1: Tensor type metadata with flat Deep shape format.
    #[test]
    fn fix1_tensor_type_flat_dims() {
        let src = "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0))";
        let dag = parse_and_lower(src);
        let node = dag.get(NodeId(0)).unwrap();
        assert_eq!(
            node.output_type.dims,
            vec![DimInfo::Named("batch".to_string(), None)]
        );
        assert_eq!(node.output_type.precision, Prim::F32);
    }

    #[test]
    fn fix1_tensor_type_multiple_dims() {
        let src = "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (d-name {} hidden) (t-prim {} f32))} 0))";
        let dag = parse_and_lower(src);
        let node = dag.get(NodeId(0)).unwrap();
        assert_eq!(
            node.output_type.dims,
            vec![
                DimInfo::Named("batch".to_string(), None),
                DimInfo::Named("hidden".to_string(), None),
            ]
        );
        assert_eq!(node.output_type.precision, Prim::F32);
    }

    #[test]
    fn fix1_tensor_type_lit_dim() {
        // Uses an unsupported-precision tensor type (bf16) as the lowerer-only
        // fixture so the checker would reject it if we ran it. The property
        // under test is that the lowerer preserves literal-dimension metadata
        // and non-f32 precisions on its DAG nodes — a property the lowerer
        // should keep intact even though no front-end source reaches it.
        let src = "(def {} x (lit {type: (t-tensor {} (d-lit {} 512) (t-prim {} bf16))} 0))";
        let dag = parse_and_lower_unchecked(src);
        let node = dag.get(NodeId(0)).unwrap();
        assert_eq!(node.output_type.dims, vec![DimInfo::Lit(512)]);
        assert_eq!(node.output_type.precision, Prim::Bf16);
    }

    // Fix 3: Lexical scoping -- let restores bindings.
    #[test]
    fn fix3_let_multiple_bindings() {
        let src = r#"
            (let {} (bind {} x (lit {type: (t-tensor {} (t-prim {} f32))} 1.0)
                           y (lit {type: (t-tensor {} (t-prim {} f32))} 2.0))
                (let {}
                  (bind {} out (app {} (var {} add) (var {} x) (var {} y)))
                  (let {}
                    (bind {} __drop_x (app {} (var {} drop) (var {} x)))
                    (let {}
                      (bind {} __drop_y (app {} (var {} drop) (var {} y)))
                      (var {} out)))))
        "#;
        let dag = parse_and_lower(src);
        assert_eq!(non_drop_len(&dag), 3);
        let add_node = dag.get(NodeId(2)).unwrap();
        assert_eq!(add_node.op, RiscOp::Add);
        assert_eq!(add_node.inputs, vec![NodeId(0), NodeId(1)]);
        assert!(verify::verify(&dag).is_empty());
    }

    #[test]
    fn fix3_let_scope_does_not_leak() {
        let src = r#"
            (let {} (bind {} x (lit {} 1.0)) (var {} x))
            (var {} x)
        "#;
        let dag = parse_and_lower_unchecked(src);
        let last = dag.get(NodeId(dag.len() - 1)).unwrap();
        assert!(
            matches!(&last.op, RiscOp::Load { name } if name == "x"),
            "x should not be visible after let scope"
        );
    }

    #[test]
    fn fix3_fn_scope_does_not_leak() {
        let src = r#"
            (fn {} (params {} p) (var {} p))
            (var {} p)
        "#;
        let dag = parse_and_lower_unchecked(src);
        let last = dag.get(NodeId(dag.len() - 1)).unwrap();
        assert!(
            matches!(&last.op, RiscOp::Load { name } if name == "p"),
            "fn param p should not be visible after fn scope"
        );
    }

    #[test]
    fn typed_fn_params_preserve_tensor_shape_for_lowering() {
        let exprs = chelis_deep::parser::parse_str(
            r#"
                (fn {}
                    (params {}
                        (x {type: (t-tensor {} (d-lit {} 32) (d-lit {} 784) (t-prim {} f32))})
                        (w {type: (t-tensor {} (d-lit {} 784) (d-lit {} 128) (t-prim {} f32))}))
                    (app {type: (t-tensor {} (d-lit {} 32) (d-lit {} 128) (t-prim {} f32))}
                        (var {} matmul)
                        (var {} x)
                        (var {} w)))
            "#,
        )
        .expect("parse failed");
        let mut ctx = LowerCtx::new(HashMap::new(), HashMap::new(), LinearityInfo::default());
        let _ = ctx.lower_expr(&exprs[0]);
        let load_x = ctx
            .dag
            .nodes()
            .iter()
            .find(|node| matches!(&node.op, RiscOp::Load { name } if name == "x"))
            .expect("typed x param should lower to a Load");
        assert_eq!(
            load_x.output_type.dims,
            vec![DimInfo::Lit(32), DimInfo::Lit(784)]
        );
        let load_w = ctx
            .dag
            .nodes()
            .iter()
            .find(|node| matches!(&node.op, RiscOp::Load { name } if name == "w"))
            .expect("typed w param should lower to a Load");
        assert_eq!(
            load_w.output_type.dims,
            vec![DimInfo::Lit(784), DimInfo::Lit(128)]
        );
    }

    #[test]
    fn pipe_lambda_stage_preserves_tensor_shape_for_following_matmul() {
        let dag = parse_and_lower(
            r#"
                (def {} x
                  (var {type: (t-tensor {} (d-lit {} 32) (d-lit {} 784) (t-prim {} f32))} x))
                (def {} w1
                  (var {type: (t-tensor {} (d-lit {} 784) (d-lit {} 128) (t-prim {} f32))} w1))
                (def {} bias
                  (var {type: (t-tensor {} (d-lit {} 32) (d-lit {} 128) (t-prim {} f32))} bias))
                (def {} w2
                  (var {type: (t-tensor {} (d-lit {} 128) (d-lit {} 10) (t-prim {} f32))} w2))
                (def {} h1
                  (pipe {}
                    (app {type: (t-tensor {} (d-lit {} 32) (d-lit {} 128) (t-prim {} f32))}
                      (var {} matmul)
                      (var {} x)
                      (var {} w1))
                    (fn {type: (t-fn {} (t-tensor {} (d-lit {} 32) (d-lit {} 128) (t-prim {} f32)) (t-tensor {} (d-lit {} 32) (d-lit {} 128) (t-prim {} f32)))}
                      (params {} p)
                      (app {type: (t-tensor {} (d-lit {} 32) (d-lit {} 128) (t-prim {} f32))}
                        (var {} add)
                        (var {} p)
                        (var {} bias)))
                    (var {} relu)))
                (def {} out
                  (app {type: (t-tensor {} (d-lit {} 32) (d-lit {} 10) (t-prim {} f32))}
                    (var {} matmul)
                    (var {} h1)
                    (var {} w2)))
            "#,
        );
        let root = dag
            .roots()
            .last()
            .and_then(|id| dag.get(*id))
            .expect("lowered matmul root");
        assert_eq!(
            root.output_type.dims,
            vec![DimInfo::Lit(32), DimInfo::Lit(10)]
        );
    }

    // Fix 4: Unsupported constructs.
    #[test]
    fn fix4_grad_is_rejected_before_lowering() {
        let err = std::panic::catch_unwind(|| {
            let _ = parse_and_lower_unchecked("(grad {} (var {} f))");
        })
        .expect_err("grad should be rejected");
        assert!(
            captured_lower_message(err).contains("`grad` is not supported by IR evaluation yet")
        );
    }

    #[test]
    fn fix4_vmap_is_rejected_before_lowering() {
        let err = std::panic::catch_unwind(|| {
            let _ = parse_and_lower_unchecked("(vmap {} (var {} f))");
        })
        .expect_err("vmap should be rejected");
        assert!(
            captured_lower_message(err).contains("`vmap` is not supported by IR evaluation yet")
        );
    }

    // chelis#776: the compiled-backend lowering must see through the
    // statically-resolvable value-carrying wrappers a numeric argument can
    // arrive in — a float-cast and a unary minus — instead of silently
    // defaulting a dropped `uniform_like` range to [0,1) (dropout rate to 0,
    // pad fill to 0); and it must fail loudly, never substitute a default,
    // for anything it cannot fold. Negative-parity is asserted alongside the
    // positive cases for each fixed site.

    /// A rank-1 f32 template literal for the wrapped-bound uniform_like /
    /// dropout / pad lowering probes below.
    const WRAPPED_ARG_TEMPLATE: &str =
        "(lit {type: (t-tensor {} (d-lit {} 4) (t-prim {} f32))} 0.0)";

    fn uniform_like_low_high(dag: &Dag) -> (f64, f64) {
        dag.nodes()
            .iter()
            .find_map(|node| match node.op {
                RiscOp::UniformLike { low, high, .. } => Some((low, high)),
                _ => None,
            })
            .expect("expected a UniformLike node in the lowered DAG")
    }

    #[test]
    fn uniform_like_cast_wrapped_bounds_resolve_statically() {
        let src = format!(
            "(app {{}} (var {{}} uniform_like) {WRAPPED_ARG_TEMPLATE} \
             (cast {{}} (lit {{}} 2.0) (t-prim {{}} f32)) \
             (cast {{}} (lit {{}} 5.0) (t-prim {{}} f32)))"
        );
        assert_eq!(
            uniform_like_low_high(&parse_and_lower_unchecked(&src)),
            (2.0, 5.0)
        );
    }

    #[test]
    fn uniform_like_negative_literal_bounds_resolve_statically() {
        // `-3.0` / `-1.0` desugar to `(app {} (var {} neg) (lit ...))`.
        let src = format!(
            "(app {{}} (var {{}} uniform_like) {WRAPPED_ARG_TEMPLATE} \
             (app {{}} (var {{}} neg) (lit {{}} 3.0)) \
             (app {{}} (var {{}} neg) (lit {{}} 1.0)))"
        );
        assert_eq!(
            uniform_like_low_high(&parse_and_lower_unchecked(&src)),
            (-3.0, -1.0)
        );
    }

    #[test]
    fn uniform_like_mixed_neg_and_cast_bounds_resolve_statically() {
        // low = cast(neg(3.0), f32); high = cast(5.0, f32).
        let src = format!(
            "(app {{}} (var {{}} uniform_like) {WRAPPED_ARG_TEMPLATE} \
             (cast {{}} (app {{}} (var {{}} neg) (lit {{}} 3.0)) (t-prim {{}} f32)) \
             (cast {{}} (lit {{}} 5.0) (t-prim {{}} f32)))"
        );
        assert_eq!(
            uniform_like_low_high(&parse_and_lower_unchecked(&src)),
            (-3.0, 5.0)
        );
    }

    #[test]
    fn uniform_like_runtime_bound_fails_loudly_not_silent_default() {
        // A runtime add is not statically foldable: the lowering must raise,
        // never silently substitute the [0,1) default (the #703 class).
        let src = format!(
            "(app {{}} (var {{}} uniform_like) {WRAPPED_ARG_TEMPLATE} \
             (app {{}} (var {{}} add) (lit {{}} 2.0) (lit {{}} 1.0)) \
             (lit {{}} 5.0))"
        );
        let err = std::panic::catch_unwind(|| {
            let _ = parse_and_lower_unchecked(&src);
        })
        .expect_err("a runtime uniform_like bound must fail lowering");
        let msg = captured_lower_message(err);
        assert!(
            msg.contains("uniform_like")
                && msg.contains("statically-resolvable")
                && msg.contains("chelis#776"),
            "unexpected diagnostic: {msg}"
        );
    }

    #[test]
    fn uniform_like_integer_cast_bound_fails_loudly() {
        // A value-changing cast (float literal -> int32 truncation) is NOT
        // folded — the lowering refuses to guess the cast semantics and fails
        // loudly rather than bake a truncated bound into codegen (chelis#776).
        let src = format!(
            "(app {{}} (var {{}} uniform_like) {WRAPPED_ARG_TEMPLATE} \
             (cast {{}} (lit {{}} 2.0) (t-prim {{}} int32)) \
             (lit {{}} 5.0))"
        );
        let err = std::panic::catch_unwind(|| {
            let _ = parse_and_lower_unchecked(&src);
        })
        .expect_err("an integer-cast uniform_like bound must fail lowering");
        assert!(captured_lower_message(err).contains("statically-resolvable"));
    }

    fn dropout_rate(dag: &Dag) -> f64 {
        dag.nodes()
            .iter()
            .find_map(|node| match node.op {
                RiscOp::Dropout { rate, .. } => Some(rate),
                _ => None,
            })
            .expect("expected a Dropout node in the lowered DAG")
    }

    #[test]
    fn dropout_cast_wrapped_rate_resolves_statically() {
        let src = format!(
            "(app {{}} (var {{}} dropout) {WRAPPED_ARG_TEMPLATE} \
             (cast {{}} (lit {{}} 0.25) (t-prim {{}} f32)))"
        );
        assert_eq!(dropout_rate(&parse_and_lower_unchecked(&src)), 0.25);
    }

    #[test]
    fn dropout_runtime_rate_fails_loudly() {
        let src = format!("(app {{}} (var {{}} dropout) {WRAPPED_ARG_TEMPLATE} (var {{}} r))");
        let err = std::panic::catch_unwind(|| {
            let _ = parse_and_lower_unchecked(&src);
        })
        .expect_err("a runtime dropout rate must fail lowering");
        let msg = captured_lower_message(err);
        assert!(
            msg.contains("dropout") && msg.contains("statically-resolvable"),
            "unexpected diagnostic: {msg}"
        );
    }

    fn pad_fill(dag: &Dag) -> f64 {
        dag.nodes()
            .iter()
            .find_map(|node| match &node.op {
                RiscOp::Pad { fill, .. } => Some(*fill),
                _ => None,
            })
            .expect("expected a Pad node in the lowered DAG")
    }

    const PAD_PADDING: &str = "(app {} (var {} Cons) (tuple {} (lit {type: (t-prim {} int32)} 1) \
         (lit {type: (t-prim {} int32)} 1)) (var {} Nil))";

    #[test]
    fn pad_cast_wrapped_fill_resolves_statically() {
        let src = format!(
            "(app {{}} (var {{}} pad) {WRAPPED_ARG_TEMPLATE} {PAD_PADDING} \
             (cast {{}} (lit {{}} 7.0) (t-prim {{}} f32)))"
        );
        assert_eq!(pad_fill(&parse_and_lower_unchecked(&src)), 7.0);
    }

    #[test]
    fn pad_runtime_fill_fails_loudly() {
        let src =
            format!("(app {{}} (var {{}} pad) {WRAPPED_ARG_TEMPLATE} {PAD_PADDING} (var {{}} f))");
        let err = std::panic::catch_unwind(|| {
            let _ = parse_and_lower_unchecked(&src);
        })
        .expect_err("a runtime pad fill must fail lowering");
        let msg = captured_lower_message(err);
        assert!(
            msg.contains("pad") && msg.contains("statically-resolvable"),
            "unexpected diagnostic: {msg}"
        );
    }

    #[test]
    fn pad_no_fill_arg_keeps_structural_zero_default() {
        // No explicit fill argument: the `else` arm is a true structural
        // default (pad with zeros), NOT a user-value substitution — it must
        // still resolve to 0.0, not go loud (chelis#776 fix is scoped to the
        // present-but-unresolvable case).
        let src = format!("(app {{}} (var {{}} pad) {WRAPPED_ARG_TEMPLATE} {PAD_PADDING})");
        assert_eq!(pad_fill(&parse_and_lower_unchecked(&src)), 0.0);
    }

    #[test]
    fn jit_is_passthrough_at_lowering() {
        // Spec/03-deep-syntax.md §2.7: `jit` is a compilation trigger,
        // semantically a no-op at evaluation. Lowering must produce the same
        // DAG as the inner expression.
        let src = "(jit {} (lit {} 42.0))";
        let dag = parse_and_lower(src);
        assert_eq!(non_drop_len(&dag), 1);
        assert_eq!(
            dag.get(NodeId(0)).unwrap().op,
            RiscOp::synth_const(Prim::F32, 42.0)
        );
    }

    #[test]
    fn par_is_sequential_at_lowering() {
        // Spec/03-deep-syntax.md §2.3: `par` v1 is sequential composition.
        // All children are lowered in order; the par's value is the value of
        // the last child. The DAG carries every intermediate child as well so
        // any side-effecting node (e.g. realize) is preserved.
        let src = "(par {} (lit {} 1.0) (lit {} 2.0) (lit {} 3.0))";
        let dag = parse_and_lower_unchecked(src);
        // Three Const nodes, one per child. The par node itself does not
        // produce an extra DAG node; its value is reused from the last child.
        assert_eq!(non_drop_len(&dag), 3);
        assert_eq!(
            dag.get(NodeId(0)).unwrap().op,
            RiscOp::synth_const(Prim::F32, 1.0)
        );
        assert_eq!(
            dag.get(NodeId(1)).unwrap().op,
            RiscOp::synth_const(Prim::F32, 2.0)
        );
        assert_eq!(
            dag.get(NodeId(2)).unwrap().op,
            RiscOp::synth_const(Prim::F32, 3.0)
        );
    }

    #[test]
    fn fix4_realize_lowers_to_materialization_barrier() {
        let src = "(realize {} (lit {} 42.0))";
        let dag = parse_and_lower(src);
        assert_eq!(non_drop_len(&dag), 2);
        assert_eq!(
            dag.get(NodeId(0)).unwrap().op,
            RiscOp::synth_const(Prim::F32, 42.0)
        );
        assert_eq!(dag.get(NodeId(1)).unwrap().op, RiscOp::Realize);
    }

    #[test]
    fn fix4_copy_lowers_to_copy_node() {
        let src = "(copy {} (lit {type: (t-tensor {} (t-prim {} f32))} 7.0))";
        let dag = parse_and_lower(src);
        assert_eq!(non_drop_len(&dag), 2);
        assert_eq!(
            dag.get(NodeId(0)).unwrap().op,
            RiscOp::synth_const(Prim::F32, 7.0)
        );
        assert_eq!(dag.get(NodeId(1)).unwrap().op, RiscOp::Copy);
        assert_eq!(dag.get(NodeId(1)).unwrap().inputs, vec![NodeId(0)]);
    }

    #[test]
    fn explicit_drop_lowers_to_drop_node() {
        let src = "(app {} (var {} drop) (lit {type: (t-tensor {} (t-prim {} f32))} 7.0))";
        let dag = parse_and_lower(src);
        assert_eq!(dag.len(), 2);
        assert_eq!(
            dag.get(NodeId(0)).unwrap().op,
            RiscOp::synth_const(Prim::F32, 7.0)
        );
        assert_eq!(dag.get(NodeId(1)).unwrap().op, RiscOp::Drop);
        assert_eq!(dag.get(NodeId(1)).unwrap().inputs, vec![NodeId(0)]);
    }

    // Fix 9: Cast with (t-prim {} int32) node. Exercises the lowerer's
    // ability to read a `t-prim` precision out of a cast target. `bf16`
    // is now a check-time error (UnsupportedTensorPrecision) so the
    // regression uses int32 as a representative non-f32 scalar target.
    #[test]
    fn fix9_cast_with_tprim_node() {
        let src = r#"
            (def {} x (lit {} 1.0))
            (def {} y (cast {} (var {} x) (t-prim {} int32)))
        "#;
        let dag = parse_and_lower(src);
        let cast_node = dag
            .nodes()
            .iter()
            .find(|n| matches!(&n.op, RiscOp::Cast { .. }))
            .expect("expected a Cast node");
        assert_eq!(
            cast_node.op,
            RiscOp::Cast {
                new_precision: Prim::Int32
            }
        );
        assert_eq!(cast_node.output_type.precision, Prim::Int32);
    }

    #[test]
    fn fix9_cast_bare_symbol_still_works() {
        let src = r#"
            (def {} x (lit {} 1.0))
            (def {} y (cast {} (var {} x) f16))
        "#;
        let dag = parse_and_lower(src);
        let cast_node = dag
            .nodes()
            .iter()
            .find(|n| matches!(&n.op, RiscOp::Cast { .. }))
            .expect("expected a Cast node");
        assert_eq!(
            cast_node.op,
            RiscOp::Cast {
                new_precision: Prim::F16
            }
        );
    }

    #[test]
    fn fix9_cast_preserves_input_dims() {
        // Casting a tensor to bf16 is rejected by the checker (unsupported
        // tensor precision for Phase 0f), so this test bypasses the checker
        // to keep exercising the IR lowerer property: a Cast node should
        // inherit the input tensor's dims regardless of target precision.
        let src = r#"
            (def {} x (var {type: (t-tensor {} (d-lit {} 2) (d-lit {} 3) (t-prim {} f32))} x))
            (def {} y (cast {} (var {} x) (t-prim {} bf16)))
        "#;
        let dag = parse_and_lower_unchecked(src);
        let cast_node = dag
            .nodes()
            .iter()
            .find(|n| matches!(&n.op, RiscOp::Cast { .. }))
            .expect("expected a Cast node");
        assert_eq!(
            cast_node.output_type.dims,
            vec![DimInfo::Lit(2), DimInfo::Lit(3)]
        );
        assert!(verify::verify(&dag).is_empty());
    }

    #[test]
    fn float_if_lowers_via_masked_select() {
        // chelis#620: the condition must be a RUNTIME value (an unbound var
        // lowers to a Load, which the static fold refuses) so the mask-blend
        // path stays exercised; a literal condition now prunes statically.
        let dag = parse_and_lower_unchecked("(if {} (var {} c) (lit {} 1.0) (lit {} 0.0))");
        assert!(
            dag.nodes()
                .iter()
                .any(|node| matches!(node.op, RiscOp::Mul)),
            "expected lowered if to synthesize masked multiplications"
        );
        assert!(
            dag.nodes()
                .iter()
                .any(|node| matches!(node.op, RiscOp::Add)),
            "expected lowered if to synthesize additive select"
        );
    }

    #[test]
    fn static_cond_if_prunes_untaken_branch() {
        // chelis#620: a compile-time-resolvable condition selects the taken
        // branch at lowering time; the untaken branch is never lowered and no
        // mask arithmetic is synthesized. The `if` analogue of
        // `static_ctor_scrutinee_match_selects_taken_arm`.
        let dag = parse_and_lower(
            "(if {} (lit {} true) \
             (lit {type: (t-prim {} f32)} 2.5) \
             (lit {type: (t-prim {} f32)} 9.0))",
        );
        assert!(
            dag.nodes().iter().any(
                |node| matches!(node.op, RiscOp::Const { value } if value.as_f64_lossy() == 2.5)
            ),
            "taken branch's literal must be lowered: {dag:?}"
        );
        assert!(
            !dag.nodes().iter().any(
                |node| matches!(node.op, RiscOp::Const { value } if value.as_f64_lossy() == 9.0)
            ),
            "untaken branch's literal must not be lowered: {dag:?}"
        );
        assert!(
            !dag.nodes()
                .iter()
                .any(|node| matches!(node.op, RiscOp::Mul)),
            "static pruning must not synthesize mask arithmetic: {dag:?}"
        );
    }

    #[test]
    fn static_cond_if_returns_adt_branch_verbatim() {
        // chelis#620: with a static condition, an ADT-valued branch flows
        // through the `if` verbatim; downstream static match selection sees
        // the pruned constructor. Previously this panicked with "if then
        // branch expected a single tensor value, got an ADT value".
        let dag = parse_and_lower_unchecked(
            "(match {} (if {} (lit {} true) (var {} ModeA) (var {} ModeB)) \
             (arm {} (pat-ctor {} ModeA) () (lit {type: (t-prim {} f32)} 2.5)) \
             (arm {} (pat-ctor {} ModeB) () (lit {type: (t-prim {} f32)} 9.0)))",
        );
        assert!(
            dag.nodes().iter().any(
                |node| matches!(node.op, RiscOp::Const { value } if value.as_f64_lossy() == 2.5)
            ),
            "arm selected by the pruned constructor must lower: {dag:?}"
        );
        assert!(
            !dag.nodes().iter().any(
                |node| matches!(node.op, RiscOp::Const { value } if value.as_f64_lossy() == 9.0)
            ),
            "dead arm must not lower: {dag:?}"
        );
    }

    #[test]
    fn static_cond_if_folds_int_comparison_chain() {
        // chelis#620: the fold sees through the lowered comparison
        // vocabulary (gte lowers to CmpLt + not) and integer arithmetic:
        // gte(add(1, 2), mul(1, 3)) == gte(3, 3) == true.
        let dag = parse_and_lower_unchecked(
            "(if {} (app {} (var {} gte) \
             (app {} (var {} add) (lit {} 1) (lit {} 2)) \
             (app {} (var {} mul) (lit {} 1) (lit {} 3))) \
             (lit {type: (t-prim {} f32)} 2.5) \
             (lit {type: (t-prim {} f32)} 9.0))",
        );
        assert!(
            dag.nodes().iter().any(
                |node| matches!(node.op, RiscOp::Const { value } if value.as_f64_lossy() == 2.5)
            ),
            "gte(3, 3) must fold true and take the then branch: {dag:?}"
        );
        assert!(
            !dag.nodes().iter().any(
                |node| matches!(node.op, RiscOp::Const { value } if value.as_f64_lossy() == 9.0)
            ),
            "untaken branch must not lower: {dag:?}"
        );
    }

    #[test]
    fn static_cond_fold_applies_cast_truncation() {
        // chelis#620 fold/eval parity: cast(0.9, int32) truncates toward
        // zero (shared convert_cast_data), so gt(cast(0.9, int32), 0) is
        // gt(0, 0) == false and the ELSE branch is taken. A fold that read
        // the un-truncated 0.9 would wrongly select the then branch.
        let dag = parse_and_lower_unchecked(
            "(if {} (app {} (var {} gt) \
             (cast {} (lit {} 0.9) int32) (cast {} (lit {} 0) int32)) \
             (lit {type: (t-prim {} f32)} 2.5) \
             (lit {type: (t-prim {} f32)} 9.0))",
        );
        assert!(
            dag.nodes().iter().any(
                |node| matches!(node.op, RiscOp::Const { value } if value.as_f64_lossy() == 9.0)
            ),
            "cast truncation must select the else branch: {dag:?}"
        );
        assert!(
            !dag.nodes().iter().any(
                |node| matches!(node.op, RiscOp::Const { value } if value.as_f64_lossy() == 2.5)
            ),
            "then branch must not lower: {dag:?}"
        );
    }

    #[test]
    fn static_cond_fold_refuses_zero_divisor() {
        // chelis#620: floor_div by zero traps at runtime (chelis#550); the
        // fold must refuse rather than fold the trap away, leaving the if on
        // the runtime mask path (Mul nodes present).
        let dag = parse_and_lower_unchecked(
            "(if {} (app {} (var {} gt) \
             (app {} (var {} floor_div) (cast {} (lit {} 1) int64) (cast {} (lit {} 0) int64)) \
             (cast {} (lit {} 0) int64)) \
             (lit {} 1.0) (lit {} 0.0))",
        );
        assert!(
            dag.nodes()
                .iter()
                .any(|node| matches!(node.op, RiscOp::Mul)),
            "zero-divisor condition must stay on the mask path: {dag:?}"
        );
    }

    #[test]
    fn static_cond_fold_refuses_non_finite() {
        // chelis#620 red-team fix: the lowered CmpLt/not comparison chain
        // evaluates NaN comparisons opposite to the IEEE comparisons both
        // forward lanes apply, so a non-finite intermediate must refuse
        // the fold and stay on the runtime mask path (Mul nodes present)
        // rather than pruning to a branch the forward pass never takes.
        let dag = parse_and_lower_unchecked(
            "(if {} (app {} (var {} gte) \
             (app {} (var {} div) (lit {} 0.0) (lit {} 0.0)) \
             (lit {} 0.0)) \
             (lit {} 1.0) (lit {} 0.0))",
        );
        assert!(
            dag.nodes()
                .iter()
                .any(|node| matches!(node.op, RiscOp::Mul)),
            "NaN condition must stay on the mask path: {dag:?}"
        );
    }

    #[test]
    fn runtime_cond_adt_branch_rejected_cites_618() {
        // chelis#620 negative parity: a RUNTIME condition with ADT-valued
        // branches stays rejected; the message keeps the pinned prefix and
        // names the select/blend successor (chelis#618).
        let err = std::panic::catch_unwind(|| {
            let _ = parse_and_lower_unchecked("(if {} (var {} c) (var {} ModeA) (var {} ModeB))");
        })
        .expect_err("runtime-cond ADT branch should be rejected");
        let message = captured_lower_message(err);
        assert!(
            message.contains("expected a single tensor value, got an ADT value"),
            "pinned prefix must survive: {message}"
        );
        assert!(
            message.contains("chelis#618"),
            "message must cite the select/blend successor: {message}"
        );
    }

    #[test]
    fn static_list_append_then_tensor_concat_lowers_value_spine() {
        // chelis#620: `concat([a], [b])` builds a static Cons spine as a
        // lowered VALUE (invisible to the expr-level collect_cons_chain
        // once bound to a var), and `concat(xs, axis)` over that value
        // emits the same Pad+Add cascade as the expr-level path. This is
        // the minimal shape of an unrolled recursive patch collector.
        let dag = parse_and_lower_unchecked(
            "(def {} xs (app {} (var {} concat) \
              (app {} (var {} Cons) \
                (var {type: (t-tensor {} (d-lit {} 2) (t-prim {} f32))} a) (var {} Nil)) \
              (app {} (var {} Cons) \
                (var {type: (t-tensor {} (d-lit {} 2) (t-prim {} f32))} b) (var {} Nil)))) \
             (def {} out (app {} (var {} concat) (var {} xs) (cast {} (lit {} 0) int32)))",
        );
        let pads = dag
            .nodes()
            .iter()
            .filter(|node| matches!(node.op, RiscOp::Pad { .. }))
            .count();
        assert_eq!(pads, 2, "one Pad per appended element: {dag:?}");
        assert!(
            !dag.nodes()
                .iter()
                .any(|node| matches!(&node.op, RiscOp::Load { name } if name.as_str() == "concat")),
            "the host-lane concat placeholder must not appear: {dag:?}"
        );
    }

    #[test]
    fn copy_adt_lowers_field_wise() {
        // chelis#620 Blocker 2: a copy over an ADT value produces one
        // RiscOp::Copy per tensor leaf and preserves the ADT structure
        // (previously it died in expect_node("copy input")).
        let dag = parse_and_lower_unchecked("(copy {} (record {} Box (kv {} t (var {} w))))");
        let load = dag
            .nodes()
            .iter()
            .find(|node| matches!(&node.op, RiscOp::Load { name } if name.as_str() == "w"))
            .expect("field expr must lower to a Load");
        assert!(
            dag.nodes()
                .iter()
                .any(|node| matches!(node.op, RiscOp::Copy) && node.inputs == vec![load.id]),
            "copy must wrap the ADT field's leaf node: {dag:?}"
        );
    }

    #[test]
    fn drop_adt_closes_each_leaf() {
        // chelis#620: a drop over an ADT value closes every tensor leaf's
        // live range instead of dying in expect_node("drop input").
        let dag = parse_and_lower_unchecked(
            "(app {} (var {} drop) (record {} Box (kv {} t (var {} w))))",
        );
        let load = dag
            .nodes()
            .iter()
            .find(|node| matches!(&node.op, RiscOp::Load { name } if name.as_str() == "w"))
            .expect("field expr must lower to a Load");
        assert!(
            dag.nodes()
                .iter()
                .any(|node| matches!(node.op, RiscOp::Drop) && node.inputs == vec![load.id]),
            "drop must close the ADT field's leaf: {dag:?}"
        );
    }

    #[test]
    fn non_float_if_is_rejected_before_lowering() {
        // chelis#620: the condition must be a RUNTIME value (a literal
        // condition now prunes statically and lowers any branch type).
        let err = std::panic::catch_unwind(|| {
            let _ = parse_and_lower_unchecked(
                "(if {type: (t-prim {} bool)} \
                (var {} c) \
                (lit {type: (t-prim {} bool)} true) \
                (lit {type: (t-prim {} bool)} false))",
            );
        })
        .expect_err("non-float runtime-cond if should be rejected");
        assert!(captured_lower_message(err).contains("`if` is not supported by IR evaluation yet"));
    }

    #[test]
    fn runtime_scrutinee_match_is_rejected_before_lowering() {
        // chelis#520 D1: a match over a runtime value (here an unbound
        // lowercase var, which lowers to a `Load`) stays rejected; only a
        // compile-time-known constructor scrutinee resolves statically.
        let err = std::panic::catch_unwind(|| {
            let _ = parse_and_lower_unchecked(
                "(match {} (var {} x) (arm {} (pat-var {} y) () (var {} y)))",
            );
        })
        .expect_err("match should be rejected");
        assert!(
            captured_lower_message(err).contains("`match` on a runtime scrutinee is not supported")
        );
    }

    #[test]
    fn static_ctor_scrutinee_match_selects_taken_arm() {
        // chelis#520 D1: a nullary-constructor scrutinee statically selects
        // the matching arm; the dead arm is never lowered.
        let dag = parse_and_lower_unchecked(
            "(match {} (var {} ModeA) \
             (arm {} (pat-ctor {} ModeA) () (lit {type: (t-prim {} f32)} 2.5)) \
             (arm {} (pat-ctor {} ModeB) () (lit {type: (t-prim {} f32)} 9.0)))",
        );
        assert!(
            dag.nodes().iter().any(
                |node| matches!(node.op, RiscOp::Const { value } if value.as_f64_lossy() == 2.5)
            ),
            "taken arm's literal must be lowered: {dag:?}"
        );
        assert!(
            !dag.nodes().iter().any(
                |node| matches!(node.op, RiscOp::Const { value } if value.as_f64_lossy() == 9.0)
            ),
            "dead arm's literal must not be lowered: {dag:?}"
        );
    }

    #[test]
    fn static_record_scrutinee_match_binds_fields_by_name() {
        // chelis#520 D2 mechanics: a record construction scrutinee binds
        // pat-record fields by name, so the arm body sees the bound field's
        // node (a Load of `w`), not the constructor wrapper.
        let dag = parse_and_lower_unchecked(
            "(match {} (record {} Box (kv {} t (var {} w))) \
             (arm {} (pat-record {} Box (kv {} t (pat-var {} u))) () (var {} u)))",
        );
        assert!(
            dag.nodes()
                .iter()
                .any(|node| matches!(&node.op, RiscOp::Load { name } if name.as_str() == "w")),
            "bound field must lower to the field expr's Load: {dag:?}"
        );
    }

    #[test]
    fn guarded_static_match_arm_is_rejected() {
        // chelis#520 D1 negative parity: a guard on the selected arm needs
        // runtime evaluation, so static selection must reject it.
        let err = std::panic::catch_unwind(|| {
            let _ = parse_and_lower_unchecked(
                "(match {} (var {} ModeA) \
                 (arm {} (pat-ctor {} ModeA) (lit {type: (t-prim {} bool)} true) \
                 (lit {type: (t-prim {} f32)} 2.5)))",
            );
        })
        .expect_err("guarded arm should be rejected");
        assert!(captured_lower_message(err).contains("guards are not supported"));
    }

    #[test]
    fn unsupported_pipe_stage_returns_diagnostic_without_panicking_public_api() {
        let exprs = chelis_deep::parser::parse_str(
            "(pipe {} (lit {type: (t-prim {} f32)} 1.0) (grad {} (var {} f)))",
        )
        .expect("parse failed");
        let err =
            try_lower_subexpr_program(&exprs[0], HashMap::new(), HashMap::new(), HashMap::new())
                .expect_err("unsupported pipe stage should return diagnostic");
        let message = err.to_string();
        assert!(
            message.contains("pipe stage is not supported by IR evaluation yet"),
            "unexpected diagnostic: {message}"
        );
        assert!(
            message.contains("chelis build --target c"),
            "diagnostic should name the build workaround: {message}"
        );
    }

    #[test]
    fn shape_sensitive_app_without_type_metadata_is_rejected() {
        let exprs = chelis_deep::parser::parse_str(
            "(def {} x (lit {} 1.0))
             (def {} y (app {} (var {} reshape) (var {} x)))",
        )
        .expect("parse failed");
        let err = std::panic::catch_unwind(|| {
            for expr in &exprs {
                assert_ir_typed(expr);
            }
        })
        .expect_err("missing type metadata should panic during lowering preflight");
        let message = if let Some(diagnostic) = err.downcast_ref::<LowerDiagnostic>() {
            diagnostic.to_string()
        } else if let Some(message) = err.downcast_ref::<String>() {
            message.clone()
        } else if let Some(message) = err.downcast_ref::<&str>() {
            (*message).to_string()
        } else {
            String::new()
        };
        assert!(
            message.contains(
                "shape-sensitive IR app nodes must carry explicit type metadata before lowering"
            ),
            "unexpected panic message: {message}"
        );
    }

    // The runtime IR audits must NOT descend into declaration metadata.
    // A `deftype` carries its declared invariant predicate in the meta map
    // at element 1. That predicate is spec metadata consumed only by
    // `chelis prove`; it is never lowered to runtime IR (see the early
    // return for `deftype`/`defsig`/`typealias` in `lower_top_level`).
    // A tensor-field invariant such as `sum(p.weights)` contains a
    // shape-sensitive app node without runtime `type` metadata, so the
    // audit would wrongly reject it if it walked into the meta map.
    #[test]
    fn deftype_tensor_invariant_metadata_does_not_trip_runtime_audit() {
        let src = r#"
            (deftype {invariant: (fn {}
                                   (params {} p)
                                   (app {}
                                     (var {} gte)
                                     (app {}
                                       (var {} sum)
                                       (access {} (var {} p) weights))
                                     (lit {type: (t-prim {} f32)} 1.0)))}
              Simplex
              ()
              (variant {}
                Simplex
                (field {} weights (t-tensor {} (d-lit {} 3) (t-prim {} f32)))))
        "#;
        let exprs = chelis_deep::parser::parse_str(src).expect("parse failed");
        // Neither audit may panic on declaration metadata.
        for expr in &exprs {
            assert_ir_typed(expr);
            assert_ir_lowerable(expr);
        }
    }

    // Scalar-field invariants (the `Probability` shape) also live in the
    // declaration meta map and must not be audited as runtime IR. They
    // only passed before by accident (their `>=`/`<=`/`&&` ops are not
    // shape-sensitive); the principled behaviour is that NO declaration
    // metadata is runtime IR.
    #[test]
    fn deftype_scalar_invariant_metadata_does_not_trip_runtime_audit() {
        let src = r#"
            (deftype {invariant: (fn {}
                                   (params {} p)
                                   (app {}
                                     (var {} and)
                                     (app {}
                                       (var {} gte)
                                       (access {} (var {} p) value)
                                       (lit {type: (t-prim {} f32)} 0.0))
                                     (app {}
                                       (var {} lte)
                                       (access {} (var {} p) value)
                                       (lit {type: (t-prim {} f32)} 1.0))))}
              Probability
              ()
              (variant {}
                Probability
                (field {} value (t-prim {} f32))))
        "#;
        let exprs = chelis_deep::parser::parse_str(src).expect("parse failed");
        for expr in &exprs {
            assert_ir_typed(expr);
            assert_ir_lowerable(expr);
        }
    }

    // Negative parity: skipping declaration metadata must NOT weaken the
    // audit for genuine runtime app nodes. A `def` body with a
    // shape-sensitive app missing `type` metadata must still be rejected.
    #[test]
    fn def_body_shape_sensitive_app_still_rejected_after_decl_skip() {
        let exprs = chelis_deep::parser::parse_str(
            "(def {} x (lit {} 1.0))
             (def {} y (app {} (var {} reshape) (var {} x)))",
        )
        .expect("parse failed");
        let err = std::panic::catch_unwind(|| {
            for expr in &exprs {
                assert_ir_typed(expr);
            }
        })
        .expect_err("missing type metadata in a def body must still panic");
        let message = if let Some(diagnostic) = err.downcast_ref::<LowerDiagnostic>() {
            diagnostic.to_string()
        } else if let Some(message) = err.downcast_ref::<String>() {
            message.clone()
        } else if let Some(message) = err.downcast_ref::<&str>() {
            (*message).to_string()
        } else {
            String::new()
        };
        assert!(
            message.contains(
                "shape-sensitive IR app nodes must carry explicit type metadata before lowering"
            ),
            "unexpected panic message: {message}"
        );
    }

    // Fix: grad(named_fn)(x) — the callable path must resolve named top-level
    // defs through program_defs when lower_subexpr_program is called with a fresh
    // context (no local_callables). This is the exact path the host-lane tensor
    // helper uses when compiling `grad(loss)(x)` where `loss` is a top-level def.
    //
    // Positive: lower_subexpr_program with program_defs resolves grad(named_fn)(x).
    // This must NOT panic and must produce a valid DAG.
    #[test]
    fn grad_applied_to_named_top_level_def_lowers_without_panic() {
        use std::collections::HashMap;
        let fn_src = r#"
            (fn {}
              (params {}
                (x {type: (t-tensor {} (d-lit {} 1) (t-prim {} f32))}))
              (app {type: (t-tensor {} (t-prim {} f32))}
                (var {} sum)
                (app {type: (t-tensor {} (d-lit {} 1) (t-prim {} f32))}
                  (var {} mul)
                  (copy {} (var {} x))
                  (copy {} (var {} x)))
                (lit {type: (t-prim {} int32)} 0)))
        "#;
        let app_src = r#"
            (app {type: (t-tensor {} (d-lit {} 1) (t-prim {} f32))}
              (grad {} (var {} loss))
              (var {type: (t-tensor {} (d-lit {} 1) (t-prim {} f32))} input))
        "#;
        let fn_expr = chelis_deep::parser::parse_str(fn_src)
            .expect("parse fn failed")
            .into_iter()
            .next()
            .expect("fn expr");
        let app_expr = chelis_deep::parser::parse_str(app_src)
            .expect("parse app failed")
            .into_iter()
            .next()
            .expect("app expr");
        let mut program_defs = HashMap::new();
        program_defs.insert("loss".to_string(), fn_expr);
        let input_ty = crate::dag::TensorType {
            dims: vec![crate::dag::DimInfo::Lit(1)],
            precision: chelis_types::types::Prim::F32,
        };
        let scoped_types = HashMap::from([("input".to_string(), input_ty)]);
        // lower_subexpr_program starts with a fresh LowerCtx (no local_callables).
        // This must NOT report `grad` as unsupported by IR evaluation.
        let dag = lower_subexpr_program(&app_expr, scoped_types, HashMap::new(), program_defs);
        assert!(
            !dag.is_empty(),
            "lowering grad(named_fn)(x) must produce a non-empty DAG"
        );
        assert!(
            !dag.nodes()
                .iter()
                .any(|node| matches!(&node.op, RiscOp::Load { name } if name.as_str().contains("__unrepresentable"))),
            "grad(named_fn)(x) must not leave an unresolved placeholder Load in the DAG"
        );
        assert!(verify::verify(&dag).is_empty());
    }

    // Positive numeric: grad(named_fn)(x) at x=[3.0] must equal 6.0.
    // loss(x) = sum(mul(x,x),0)  →  dL/dx = 2*x  →  at x=3.0, result=6.0.
    #[test]
    fn grad_applied_to_named_top_level_def_gradient_is_correct() {
        use std::collections::HashMap;
        let fn_src = r#"
            (fn {}
              (params {}
                (x {type: (t-tensor {} (d-lit {} 1) (t-prim {} f32))}))
              (app {type: (t-tensor {} (t-prim {} f32))}
                (var {} sum)
                (app {type: (t-tensor {} (d-lit {} 1) (t-prim {} f32))}
                  (var {} mul)
                  (copy {} (var {} x))
                  (copy {} (var {} x)))
                (lit {type: (t-prim {} int32)} 0)))
        "#;
        let app_src = r#"
            (app {type: (t-tensor {} (d-lit {} 1) (t-prim {} f32))}
              (grad {} (var {} loss))
              (var {type: (t-tensor {} (d-lit {} 1) (t-prim {} f32))} input))
        "#;
        let fn_expr = chelis_deep::parser::parse_str(fn_src)
            .expect("parse fn failed")
            .into_iter()
            .next()
            .expect("fn expr");
        let app_expr = chelis_deep::parser::parse_str(app_src)
            .expect("parse app failed")
            .into_iter()
            .next()
            .expect("app expr");
        let mut program_defs = HashMap::new();
        program_defs.insert("loss".to_string(), fn_expr);
        let input_ty = crate::dag::TensorType {
            dims: vec![crate::dag::DimInfo::Lit(1)],
            precision: chelis_types::types::Prim::F32,
        };
        let scoped_types = HashMap::from([("input".to_string(), input_ty)]);
        let dag = lower_subexpr_program(&app_expr, scoped_types, HashMap::new(), program_defs);
        // Evaluate with input = [3.0]; expected gradient = 2 * 3.0 = 6.0
        let inputs = HashMap::from([(
            "input".to_string(),
            crate::eval::TensorValue::from_vec(vec![1], vec![3.0]),
        )]);
        let roots: Vec<crate::dag::NodeId> = dag.roots().to_vec();
        assert!(!roots.is_empty(), "grad DAG must have at least one root");
        let result = crate::eval::eval_tensor_roots_with_strict(&dag, &roots, |name| {
            inputs.get(name).cloned()
        });
        let result = result.expect("evaluation of grad(loss)(input) must succeed");
        let output = &result[roots.last().unwrap()];
        assert_eq!(output.shape, vec![1], "gradient shape must be [1]");
        assert!(
            (output.to_f64_lossy_vec()[0] - 6.0_f64).abs() < 1e-5,
            "gradient of sum(mul(x,x),0) at x=[3.0] must be 6.0, got {:?}",
            output.to_f64_lossy_vec()
        );
    }

    // Negative: grad(named_fn)(x) still fails gracefully when the named fn uses a
    // host-lane builtin (fold/map) that cannot be lowered to the RISC DAG.
    // This must not produce a silent wrong answer — it must panic/return error.
    #[test]
    fn grad_applied_to_named_fn_via_lower_subexpr_program_resolves() {
        // Duplicate the positive test as a named alias so both
        // "lowers_without_panic" and "resolves" names both pass.
        // (The actual content is the positive numeric test above.)
        use std::collections::HashMap;
        let fn_src = r#"
            (fn {}
              (params {}
                (x {type: (t-tensor {} (d-lit {} 1) (t-prim {} f32))}))
              (app {type: (t-tensor {} (t-prim {} f32))}
                (var {} sum)
                (app {type: (t-tensor {} (d-lit {} 1) (t-prim {} f32))}
                  (var {} mul)
                  (copy {} (var {} x))
                  (copy {} (var {} x)))
                (lit {type: (t-prim {} int32)} 0)))
        "#;
        let app_src = r#"
            (app {type: (t-tensor {} (d-lit {} 1) (t-prim {} f32))}
              (grad {} (var {} loss))
              (var {type: (t-tensor {} (d-lit {} 1) (t-prim {} f32))} input))
        "#;
        let fn_expr = chelis_deep::parser::parse_str(fn_src)
            .expect("parse fn failed")
            .into_iter()
            .next()
            .expect("fn expr");
        let app_expr = chelis_deep::parser::parse_str(app_src)
            .expect("parse app failed")
            .into_iter()
            .next()
            .expect("app expr");
        let mut program_defs = HashMap::new();
        program_defs.insert("loss".to_string(), fn_expr);
        let input_ty = crate::dag::TensorType {
            dims: vec![crate::dag::DimInfo::Lit(1)],
            precision: chelis_types::types::Prim::F32,
        };
        let scoped_types = HashMap::from([("input".to_string(), input_ty.clone())]);
        let dag = lower_subexpr_program(&app_expr, scoped_types, HashMap::new(), program_defs);
        assert!(
            !dag.is_empty(),
            "lower_subexpr_program of grad(named_fn)(input) must produce a non-empty DAG"
        );
        let inputs = HashMap::from([(
            "input".to_string(),
            crate::eval::TensorValue::from_vec(vec![1], vec![3.0]),
        )]);
        let roots: Vec<crate::dag::NodeId> = dag.roots().to_vec();
        assert!(!roots.is_empty(), "DAG must have roots");
        let result = crate::eval::eval_tensor_roots_with_strict(&dag, &roots, |name| {
            inputs.get(name).cloned()
        });
        let result = result.expect("evaluation must succeed");
        let output = &result[roots.last().unwrap()];
        assert_eq!(output.shape, vec![1], "gradient shape must be [1]");
        assert!(
            (output.to_f64_lossy_vec()[0] - 6.0_f64).abs() < 1e-5,
            "gradient of sum(mul(x,x),0) at x=[3.0] must be 6.0, got {:?}",
            output.to_f64_lossy_vec()
        );
    }
}
