//! Error types for the Chelis type checker.

use serde::{Deserialize, Serialize};

#[doc(hidden)]
pub use crate::session::DiagnosticSink;
use crate::types::{Prim, Type};
use crate::unify::{TypeError, TypeErrorKind};
use crate::unsupported::Unsupported;

/// Zero-sized witness that a `Type::Error` was minted HONESTLY: either a
/// diagnostic reached the error vector (via [`report`]) or an existing
/// witness was propagated for cascade suppression (via [`propagate`]).
///
/// The single field is private, so no code outside this module can call the
/// `ErrorWitness(())` constructor. Because the only way to obtain a witness
/// is [`report`] (which pushes a `CheckError` in the same expression) or
/// [`propagate`] (which requires an *existing* witness, tracing back to a
/// real report), a `Type::Error` can never be conjured from nothing anywhere
/// else in the tree. That makes a silent `Type::Error` -- the chelis#709 /
/// chelis#710 class defect where an unrecognized or malformed construct
/// disabled checking for its whole subtree without a diagnostic --
/// unconstructible by construction (spec/design/checker_totality.md §C3,
/// frozen at Phase 2).
///
/// The mutation oracle: a planted bare `Type::Error` fails to compile (the
/// variant now takes a field), and `Type::Error(ErrorWitness(()))` fails to
/// compile outside this module (the field is private). The checker's own
/// `infer.rs` must therefore route every error through [`report`] /
/// [`propagate`]; it cannot mint one directly.
///
/// Both halves of that sentence are planted below rather than asserted in
/// prose. The bare-variant half (chelis#875) had no executable form until
/// this block existed: it is true by the type system today, but nothing
/// would have noticed if the variant regained a `Default`, gained a second
/// zero-argument constructor path, or had its field widened to something
/// publicly inhabitable. Note the type ascription -- without it,
/// `Type::Error` names the tuple-variant *constructor function* and compiles
/// happily; the oracle has to demand a `Type`.
///
/// ```compile_fail
/// use chelis_types::types::Type;
/// let _silent: Type = Type::Error;
/// ```
///
/// What this witness does NOT cover, stated here because the acceptance
/// surface should not read stronger than the mechanism (chelis#875): the
/// guard is specific to `Type::Error`. It says nothing about an ordinary
/// type standing in as a verdict for "the checker did not recognize this" --
/// the chelis#873 shape, where `infer_atom` returned `Type::Unit` for a form
/// it could not type. No constructor gate can close that one: `Type::Unit`
/// has legitimate uses, so there is nothing to make unconstructible. That
/// shape is held behaviorally instead, by the score-surface corpus in
/// `crates/chelis-cli/tests/issue_731_fitness_honesty_corpus.rs`, not by any
/// oracle in this module.
///
/// Serialization note (chelis#731 open question 2, resolved at Phase 2):
/// `Serialize`/`Deserialize` are derived because `Type` is cached, so serde is
/// the ONE sanctioned non-constructor mint. Production cache writers receive
/// only successful [`crate::CheckedProgram`] / [`crate::TypeEnv`] values; a
/// non-empty checker error vector prevents construction, and the totality
/// invariant forbids `Type::Error` in a successful result. The cache decoder
/// is an internal-artifact boundary, not a semantic re-checker: its envelope
/// verifies format/build identity and byte integrity, then trusts the decoded
/// successful payload. See `context` and the compiler-api cache module docs.
///
/// The constructor privacy is a compile-time boundary:
///
/// ```compile_fail
/// use chelis_types::errors::ErrorWitness;
/// use chelis_types::types::Type;
/// let _silent = Type::Error(ErrorWitness(()));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorWitness(());

/// The ONLY honest way to turn a *fresh* problem into `Type::Error`: push the
/// diagnostic onto the authoritative checker-session sink and mint the witness in the same
/// expression, so the two can never be separated by a later refactor, a
/// review miss, or a new contributor (spec/design/checker_totality.md §C3).
/// Returns the `Type::Error` carrying the freshly-minted witness.
///
/// This replaces the historical `errors.push(e); return Type::Error;` idiom:
/// `return report(errors, e);` is exactly that, with the push and the mint
/// welded into one expression.
///
/// An arbitrary vector cannot be substituted for the session capability;
/// this is the retained hidden-diagnostic mutation oracle:
///
/// ```compile_fail
/// use chelis_types::errors::{CheckError, CheckErrorKind, report};
/// let mut hidden_errors = Vec::new();
/// let _ = report(
///     &mut hidden_errors,
///     CheckError::new(CheckErrorKind::Other, "hidden".to_string(), vec![]),
/// );
/// ```
///
/// The capability cannot be directly constructed:
///
/// ```compile_fail
/// use chelis_types::errors::{CheckError, DiagnosticSink};
/// let mut errors: Vec<CheckError> = Vec::new();
/// let _ = DiagnosticSink { errors: &mut errors };
/// ```
///
/// Nor can code manufacture or duplicate one through standard conversion
/// and ownership traits:
///
/// ```compile_fail
/// use chelis_types::errors::DiagnosticSink;
/// let _ = DiagnosticSink::default();
/// ```
///
/// ```compile_fail
/// use chelis_types::errors::DiagnosticSink;
/// fn require_clone<T: Clone>() {}
/// require_clone::<DiagnosticSink<'static>>();
/// ```
///
/// ```compile_fail
/// use chelis_types::errors::{CheckError, DiagnosticSink};
/// let _ = DiagnosticSink::from(Vec::<CheckError>::new());
/// ```
///
/// ```compile_fail
/// use std::ops::DerefMut;
/// use chelis_types::errors::{CheckError, DiagnosticSink};
/// fn require_vec_deref_mut<T: DerefMut<Target = Vec<CheckError>>>() {}
/// require_vec_deref_mut::<DiagnosticSink<'static>>();
/// ```
pub fn report(errors: &mut DiagnosticSink<'_>, error: CheckError) -> Type {
    Type::Error(report_witness(errors, error))
}

/// The located form of `report`: source identity and coordinate are attached
/// before the same append-and-witness expression, never in a later pass.
pub(crate) fn report_at(
    errors: &mut DiagnosticSink<'_>,
    error: CheckError,
    span: Option<&crate::deep_type::TypeDiagnosticLocation>,
) -> Type {
    report(
        errors,
        match span {
            Some(location) => location.attach(error),
            None => error,
        },
    )
}

/// Crate-private result-boundary form of [`report`]. Deep type resolution
/// cannot manufacture a usable [`Type`] after malformed input, so it returns
/// this witness through `Result` and requires its caller to propagate the
/// failure explicitly. The constructor remains private to this module.
pub(crate) fn report_witness(errors: &mut DiagnosticSink<'_>, error: CheckError) -> ErrorWitness {
    errors.push(error);
    ErrorWitness(())
}

/// Cascade suppression: a node whose child already typed as `Type::Error(w)`
/// types itself `Type::Error` WITHOUT re-reporting, by propagating the
/// existing witness `w`. This preserves the pre-token behavior where an
/// error's descendants unify freely so one mistake does not spray dozens of
/// secondary diagnostics (spec/design/checker_totality.md §C3). Because
/// `propagate` requires an existing witness, the cascade is provably
/// downstream of a real reported error.
pub fn propagate(witness: &ErrorWitness) -> Type {
    Type::Error(*witness)
}

/// A `Type::Error` sentinel for in-crate UNIT tests that need to feed an
/// error-typed value into unification/equality directly (e.g. verifying the
/// permissive `(Error, _) => Ok(())` unify arm). Gated on `#[cfg(test)]`, so
/// it does not exist in a production build and cannot be used to mint a
/// silent `Type::Error` in the shipped checker -- the §C3 oracle is
/// unaffected. Integration tests are separate crates and never see this.
#[cfg(test)]
pub(crate) fn error_sentinel_for_test() -> Type {
    Type::Error(ErrorWitness(()))
}

/// A check-time diagnostic.
///
/// This is checker-internal state, NOT a wire type (chelis#886).
///
/// It briefly derived `Serialize` so the CLI could emit it directly. That
/// made a `chelis-types` struct a numeric wire root: its public
/// `severity: f64` became published surface while the §C6 wire census is
/// rooted in `chelis-compiler-api`'s schema, so the field was exposed with
/// no census row and no final authority class. The wire projection now lives
/// at `chelis_compiler_api::schema::Diagnostic::from_check_error`, on a
/// carrier the census already enumerates.
#[derive(Debug, Clone)]
pub struct CheckError {
    pub kind: CheckErrorKind,
    pub message: String,
    /// Repair hints. The hand-assembled document never carried them, so
    /// emitting them through the typed carrier is a deliberate field-set
    /// change under [04-FIT-15], not a side effect of typing the producer.
    /// This struct is not itself serialized; the change is visible only
    /// where `Diagnostic::from_check_error` copies the field across.
    pub suggestions: Vec<String>,
    pub severity: f64,
    /// Expected type (for structured error reports).
    pub expected: Option<String>,
    /// Actual type found (for structured error reports).
    pub got: Option<String>,
    /// Byte offset into the source where the error occurred.
    pub span_offset: Option<usize>,
    /// External span identifier (e.g. from Octant's LaTeX-to-Deep translator).
    pub span_id: Option<String>,
}

/// A check diagnostic's kind.
///
/// Its published spelling is the governed `chelis_vocab::DiagnosticKind`
/// identity that `Diagnostic::from_check_error` maps it to, not this Rust
/// identifier, so renaming a variant here cannot move the wire.
#[derive(Clone)]
pub enum CheckErrorKind {
    TypeMismatch,
    PrecisionMismatch,
    DimensionMismatch,
    ArityMismatch,
    UnboundVariable {
        identifier: String,
    },
    /// A constructor reference (uppercase-leading name in expression or
    /// pattern position) names an ADT variant that is not in scope: it is
    /// neither declared in the current module nor brought into scope by an
    /// `import` that names it (chelis#317). After reef's module-scoped
    /// constructor mangling (chelis#157, #316), an in-scope constructor is
    /// always referenced by its exact (mangled or bare-builtin) name, so a
    /// bare name that only resolves through the registry's fuzzy
    /// terminal-segment match is a constructor from another module that the
    /// importing module never pulled in. Binding it silently bound the
    /// reference to a foreign tag and deferred the failure to a runtime
    /// `non-exhaustive match`; this rejects it at `check` instead, the same
    /// way an `UnboundVariable` rejects an unknown value name.
    UnknownConstructor {
        identifier: String,
    },
    NotAFunction,
    NonExhaustiveMatch,
    OccursCheck,
    CastNonTensor,
    TupleIndexOutOfBounds,
    UseAfterConsume,
    UnconsumedLinear,
    InvalidBorrow,
    /// [04-LIN-9]: a random key, or a value that carries one, is used a
    /// second time, borrowed, copied, captured by a closure, or read by an
    /// operation that leaves it live. Keys are affine; the repair derives
    /// fresh keys with `split_key` or `split_keys`, never `copy`.
    ///
    /// [04-LIN-10]: also a function's type parameter instantiated at a
    /// key-carrying type, which would let a generic body use the key more than
    /// once; the repair passes the key through a concrete parameter.
    KeyReuse,
    CycleDetected,
    /// A tensor type uses a precision the Phase 0f backend cannot represent
    /// (currently f16, bf16, f64, f8e4m3). Host scalar precisions are unaffected.
    UnsupportedTensorPrecision,
    /// A `deftype` or `typealias` reuses a name already bound by an
    /// earlier `deftype`, `typealias`, or prelude ADT in the same
    /// program. Type names share one flat namespace in `AdtRegistry`
    /// (keyed on bare strings), so silent last-write-wins would let a
    /// later declaration overwrite the original — propagating wrong
    /// constructor types, wrong variant sets, and (per
    /// `compute_tensor_carrying_adts` in `linearity.rs`) order-
    /// dependent borrow semantics. Reject the collision at declaration
    /// time instead.
    DuplicateDefinition,
    /// RFC v4b (RT-1 F2): a named module is opened by more than one
    /// `(module ...)` wrapper in the same check unit. Module identity
    /// is otherwise a forgeable string -- a second wrapper of an
    /// opaque type's defining module would construct and inspect the
    /// type as if it were inside. A named module may be opened at most
    /// once per check unit (the same ambiguity rationale as the
    /// named-module requirement for `@opaque`).
    DuplicateModule,
    /// RFC D-CHECK (spec/design/opaque_invariants_rfc.md): a type
    /// declared `@opaque` was constructed, inspected, forged, or
    /// reached through an unexported binding outside its defining
    /// module, or `@opaque` was declared outside a named module. The
    /// message names the type, the defining module, and the exported
    /// producers with signatures; location context is the enclosing
    /// def name embedded in the message.
    OpaqueTypeViolation,
    /// RFC v5 (RT-1 F2 bypass): a top-level declaration's binding name
    /// matches the reef package-linker's reserved internal-name format
    /// (`Pkg__<pkg>__<Module>__<Name>` / lowercase twin) in a program
    /// NOT produced by the linker. That format is the linker's private
    /// output; hand-authoring it forges module identity through the
    /// reef-stem channel (a flat program of mangled names self-keys to
    /// one module and constructs/inspects opaque types as in-module).
    ReservedLinkerName,
    /// A top-level `def` or `sig` (Deep `defsig`) reuses a name from the
    /// closed builtin vocabulary (`BUILTIN_NAMES`). Top-level builtin names
    /// identify the intrinsic call surface and cannot be rebound with a user
    /// signature, so they are rejected at declaration time
    /// (spec/04-type-system.md §8.6). Ordinary lexical bindings are
    /// different: function parameters, block locals, and pattern bindings
    /// take precedence over builtin dispatch in every lane
    /// (chelis#1076). Reef package modules are also unaffected: their decls
    /// are internal-name-rewritten (`pkg__...`) before the checker runs, and
    /// their call sites are rewritten with them.
    BuiltinShadowing,
    /// chelis#731 / spec/04-type-system.md §10 [04-TOT-1]: a Deep tag
    /// reached `infer_expr`'s dispatch with no checker disposition. The
    /// parser already screens the 62-tag closed vocabulary
    /// (spec/03-deep-syntax.md), so this is a version skew or a bug, not
    /// ordinary user input. The checker rejects it loudly instead of
    /// returning a silent `Type::Error` that would exempt the whole
    /// subtree from checking (chelis#709's class defect).
    UnknownForm,
    /// chelis#731 / spec/04-type-system.md §10 [04-TOT-3]: a structurally
    /// malformed Deep form reached the checker (an arity or shape guard
    /// that used to return a silent `Type::Error`, chelis#710's half).
    /// The message names the tag and the expected shape; deferring the
    /// failure to a later stage (the runtime catching it) is not a
    /// disposition.
    MalformedForm,
    /// A recognized language construct whose implementation is not complete
    /// enough to admit. The typed payload retains the rejection's stage,
    /// issue authority, location, and supported alternative.
    UnsupportedFeature {
        unsupported: Box<Unsupported>,
    },
    Other,
}

impl std::fmt::Debug for CheckErrorKind {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.diagnostic_name())
    }
}

impl CheckErrorKind {
    /// Stable machine-facing spelling used by `chelis check`. This is
    /// intentionally independent of `Debug`, because name diagnostics carry
    /// structured data that must not leak into or invalidate the JSON kind.
    pub fn diagnostic_name(&self) -> &'static str {
        match self {
            CheckErrorKind::TypeMismatch => "TypeMismatch",
            CheckErrorKind::PrecisionMismatch => "PrecisionMismatch",
            CheckErrorKind::DimensionMismatch => "DimensionMismatch",
            CheckErrorKind::ArityMismatch => "ArityMismatch",
            CheckErrorKind::UnboundVariable { .. } => "UnboundVariable",
            CheckErrorKind::UnknownConstructor { .. } => "UnknownConstructor",
            CheckErrorKind::NotAFunction => "NotAFunction",
            CheckErrorKind::NonExhaustiveMatch => "NonExhaustiveMatch",
            CheckErrorKind::OccursCheck => "OccursCheck",
            CheckErrorKind::CastNonTensor => "CastNonTensor",
            CheckErrorKind::TupleIndexOutOfBounds => "TupleIndexOutOfBounds",
            CheckErrorKind::UseAfterConsume => "UseAfterConsume",
            CheckErrorKind::UnconsumedLinear => "UnconsumedLinear",
            CheckErrorKind::InvalidBorrow => "InvalidBorrow",
            CheckErrorKind::KeyReuse => "KeyReuse",
            CheckErrorKind::CycleDetected => "CycleDetected",
            CheckErrorKind::UnsupportedTensorPrecision => "UnsupportedTensorPrecision",
            CheckErrorKind::DuplicateDefinition => "DuplicateDefinition",
            CheckErrorKind::DuplicateModule => "DuplicateModule",
            CheckErrorKind::OpaqueTypeViolation => "OpaqueTypeViolation",
            CheckErrorKind::ReservedLinkerName => "ReservedLinkerName",
            CheckErrorKind::BuiltinShadowing => "BuiltinShadowing",
            CheckErrorKind::UnknownForm => "UnknownForm",
            CheckErrorKind::MalformedForm => "MalformedForm",
            CheckErrorKind::UnsupportedFeature { .. } => "unsupported_feature",
            CheckErrorKind::Other => "Other",
        }
    }

    /// The exact unresolved source identifier carried by a name-resolution
    /// diagnostic. Fitness accounting consumes this structured value instead
    /// of attempting to recover it from the rendered diagnostic message.
    pub fn unresolved_identifier(&self) -> Option<&str> {
        match self {
            CheckErrorKind::UnboundVariable { identifier }
            | CheckErrorKind::UnknownConstructor { identifier } => Some(identifier),
            _ => None,
        }
    }

    pub fn default_severity(&self) -> f64 {
        match self {
            CheckErrorKind::PrecisionMismatch | CheckErrorKind::DimensionMismatch => 0.8,
            CheckErrorKind::ArityMismatch => 0.7,
            CheckErrorKind::UnboundVariable { .. } => 0.6,
            CheckErrorKind::UnknownConstructor { .. } => 0.6,
            CheckErrorKind::NotAFunction => 0.5,
            CheckErrorKind::TypeMismatch => 0.5,
            CheckErrorKind::NonExhaustiveMatch => 0.7,
            CheckErrorKind::OccursCheck => 0.5,
            CheckErrorKind::CastNonTensor => 0.6,
            CheckErrorKind::TupleIndexOutOfBounds => 0.7,
            CheckErrorKind::UseAfterConsume => 0.9,
            CheckErrorKind::UnconsumedLinear => 0.9,
            CheckErrorKind::InvalidBorrow => 0.8,
            CheckErrorKind::KeyReuse => 0.9,
            CheckErrorKind::CycleDetected => 0.9,
            CheckErrorKind::UnsupportedTensorPrecision => 0.8,
            CheckErrorKind::DuplicateDefinition => 0.9,
            CheckErrorKind::DuplicateModule => 0.9,
            CheckErrorKind::OpaqueTypeViolation => 0.8,
            CheckErrorKind::ReservedLinkerName => 0.9,
            CheckErrorKind::BuiltinShadowing => 0.9,
            // chelis#731 open question 4 (decided 2026-07-17): severity
            // parity with `TypeMismatch` (the 0.5 class); no new weight
            // class. The invariant that governs is that any pushed error
            // forces score < 1.0, which §C4.4's corpus locks independently.
            CheckErrorKind::UnknownForm | CheckErrorKind::MalformedForm => 0.5,
            CheckErrorKind::UnsupportedFeature { .. } => 1.0,
            CheckErrorKind::Other => 0.5,
        }
    }
}

impl CheckError {
    /// Create a new CheckError with default severity for its kind.
    pub fn new(kind: CheckErrorKind, message: String, suggestions: Vec<String>) -> Self {
        let severity = kind.default_severity();
        CheckError {
            kind,
            message,
            suggestions,
            severity,
            expected: None,
            got: None,
            span_offset: None,
            span_id: None,
        }
    }

    /// Preserve one typed unsupported rejection through the checker error
    /// channel. Human and machine fields derive from the same payload.
    pub fn from_unsupported(unsupported: Unsupported) -> Self {
        let message = unsupported.to_string();
        let suggestions = unsupported
            .supported_alternative
            .as_deref()
            .map(str::to_owned)
            .into_iter()
            .collect();
        let span_offset = unsupported.span.as_deref().and_then(|span| span.offset);
        let span_id = unsupported
            .span
            .as_deref()
            .and_then(|span| span.span_id.clone());
        Self {
            kind: CheckErrorKind::UnsupportedFeature {
                unsupported: Box::new(unsupported),
            },
            message,
            suggestions,
            severity: 1.0,
            expected: None,
            got: None,
            span_offset,
            span_id,
        }
    }

    /// Create with expected/got for structured error reports.
    pub fn with_types(
        kind: CheckErrorKind,
        message: String,
        expected: String,
        got: String,
        suggestions: Vec<String>,
    ) -> Self {
        let severity = kind.default_severity();
        CheckError {
            kind,
            message,
            suggestions,
            severity,
            expected: Some(expected),
            got: Some(got),
            span_offset: None,
            span_id: None,
        }
    }

    /// Set the byte offset into the source where this error occurred.
    pub fn at_offset(mut self, offset: usize) -> Self {
        self.span_offset = Some(offset);
        self
    }

    /// Set the external span identifier for this error.
    pub fn with_span_id(mut self, id: String) -> Self {
        self.span_id = Some(id);
        self
    }
}

/// [04-LIN-10]: the repair every generic-instantiation key diagnostic carries.
/// It names a concrete key parameter, never `copy`, because a key is never
/// copied.
pub(crate) const KEY_PARAMETER_SUGGESTION: &str = "Pass keys through a concrete `key` or \
     `tensor[n, key]` parameter rather than a type parameter; derive fresh keys with \
     `split_key(k)` or `split_keys(k, n)` where more than one is needed";

/// [04-LIN-10]: the repair for a generic that is a value binding rather than a
/// function, such as a generalized `let` binding: an ascription gives the
/// binding a type with no type parameter, and a value written where it is
/// used is never generalized.
fn key_value_binding_suggestion(binding: &str) -> String {
    format!(
        "Ascribe `{binding}` a type with no type parameter where it is bound, such as \
         `{binding}: List[key] = Nil`, or write its value where it is used; a binding without an \
         ascription is generalized, and a generic never stands for a key"
    )
}

impl From<TypeError> for CheckError {
    fn from(te: TypeError) -> Self {
        let value_binding = match &te.kind {
            TypeErrorKind::KeyInstantiation { value_binding } => value_binding.clone(),
            _ => None,
        };
        let kind = match te.kind {
            TypeErrorKind::TypeMismatch => CheckErrorKind::TypeMismatch,
            TypeErrorKind::PrecisionMismatch | TypeErrorKind::DtypeFamilyMismatch => {
                CheckErrorKind::PrecisionMismatch
            }
            TypeErrorKind::KeyInstantiation { .. } => CheckErrorKind::KeyReuse,
            TypeErrorKind::DimensionMismatch => CheckErrorKind::DimensionMismatch,
            TypeErrorKind::ArityMismatch => CheckErrorKind::ArityMismatch,
            TypeErrorKind::OccursCheck => CheckErrorKind::OccursCheck,
            TypeErrorKind::NotAFunction => CheckErrorKind::NotAFunction,
        };
        let severity = match &kind {
            CheckErrorKind::KeyReuse => 0.9,
            CheckErrorKind::PrecisionMismatch | CheckErrorKind::DimensionMismatch => 0.8,
            CheckErrorKind::ArityMismatch => 0.7,
            CheckErrorKind::UnboundVariable { .. } => 0.6,
            _ => 0.5,
        };
        let mut suggestions = match &kind {
            CheckErrorKind::PrecisionMismatch => vec!["Insert explicit cast".to_string()],
            CheckErrorKind::KeyReuse => vec![match &value_binding {
                Some(binding) => key_value_binding_suggestion(binding),
                None => KEY_PARAMETER_SUGGESTION.to_string(),
            }],
            _ => vec![],
        };
        // Enrich TypeMismatch with opaque/option hints.
        if matches!(kind, CheckErrorKind::TypeMismatch) {
            enrich_type_mismatch_suggestions(&te.message, &mut suggestions);
        }
        CheckError {
            kind,
            message: te.message,
            suggestions,
            severity,
            expected: None,
            got: None,
            span_offset: None,
            span_id: None,
        }
    }
}

/// Enrich TypeMismatch suggestions by detecting common patterns:
/// - Opaque type used where a primitive is expected
/// - Option[T] used where T is expected
pub fn enrich_type_mismatch_suggestions(message: &str, suggestions: &mut Vec<String>) {
    // Detect Option[T] vs T pattern: "type mismatch: Option[X] vs X" or "X vs Option[X]"
    if let Some(hint) = option_unwrap_hint(message) {
        suggestions.push(hint);
    }
    // Detect opaque type vs primitive pattern
    if let Some(hint) = opaque_accessor_hint(message) {
        suggestions.push(hint);
    }
    // Detect a List where a tensor is expected
    if let Some(hint) = list_for_tensor_hint(message) {
        suggestions.push(hint);
    }
}

/// If the mismatch is a tensor against a `List`, name a conversion that keeps
/// each element's value: a bracket literal is a `List` unless its own
/// declaration states a tensor type (spec/02-surf-syntax.md §P10b).
fn list_for_tensor_hint(message: &str) -> Option<String> {
    let (_, type_part) = message.split_once(": ")?;
    let (left, right) = type_part.split_once(" vs ")?;
    let (left, right) = (left.trim(), right.trim());
    let tensor = match (left.starts_with("List"), right.starts_with("List")) {
        (false, true) => left,
        (true, false) => right,
        _ => return None,
    };
    let element = tensor
        .strip_prefix("tensor[")?
        .strip_suffix(']')?
        .rsplit(", ")
        .next()
        .map(str::trim);
    Some(list_to_tensor_hint(element))
}

/// The value-preserving ways to write a tensor whose elements have dtype
/// `element`. `to_tensor` alone keeps each literal's own suffix or §5.3
/// default, so for a numeric element dtype the elements carry its suffix
/// (spec/04-type-system.md §5.6). Any other element type, a `bool` or a
/// dtype binder, has no literal suffix, so the declared binding is the fix.
pub fn list_to_tensor_hint(element: Option<&str>) -> String {
    let numeric = element.filter(|name| {
        Prim::parse_name(name).is_some_and(|prim| prim.is_float() || prim.is_integer())
    });
    let Some(element) = numeric else {
        return "a bracket literal is a List, and only its own declared tensor type converts \
                it; bind the literal under a declared tensor type"
            .to_string();
    };
    let (first, second) = if element.starts_with('i') {
        ("1", "2")
    } else {
        ("1.1", "2.2")
    };
    format!(
        "a bracket literal is a List, and only its own declared tensor type converts it; \
         write `to_tensor([{first}{element}, {second}{element}])`, with each element suffixed at \
         the dtype it keeps, or bind the literal under a declared tensor type such as \
         `xs: tensor[2, {element}] = [{first}, {second}]`"
    )
}

/// If the mismatch is `Option[T]` vs `T`, suggest pattern matching.
fn option_unwrap_hint(message: &str) -> Option<String> {
    // The unify message has form "type mismatch: A vs B"
    let parts: Vec<&str> = message.splitn(2, ": ").collect();
    if parts.len() < 2 {
        return None;
    }
    let type_part = parts[1];
    let sides: Vec<&str> = type_part.splitn(2, " vs ").collect();
    if sides.len() < 2 {
        return None;
    }
    let (left, right) = (sides[0].trim(), sides[1].trim());
    // Check if left is Option[T] and right is T, or vice versa
    if let Some(inner) = strip_option_wrapper(left)
        && inner == right
    {
        return Some(format!(
            "this expression returns Option[{inner}]; use `match ... with {{ | Some(value) => ... | None => ... }}` to unwrap"
        ));
    }
    if let Some(inner) = strip_option_wrapper(right)
        && inner == left
    {
        return Some(format!(
            "this expression returns Option[{inner}]; use `match ... with {{ | Some(value) => ... | None => ... }}` to unwrap"
        ));
    }
    None
}

fn strip_option_wrapper(s: &str) -> Option<&str> {
    let s = s.strip_prefix("Option[")?;
    let s = s.strip_suffix(']')?;
    Some(s)
}

/// If the mismatch involves an opaque type vs a primitive, suggest the accessor.
fn opaque_accessor_hint(message: &str) -> Option<String> {
    let parts: Vec<&str> = message.splitn(2, ": ").collect();
    if parts.len() < 2 {
        return None;
    }
    let type_part = parts[1];
    let sides: Vec<&str> = type_part.splitn(2, " vs ").collect();
    if sides.len() < 2 {
        return None;
    }
    let (left, right) = (sides[0].trim(), sides[1].trim());
    let primitives = ["f32", "f64", "i32", "i64", "bool"];
    // Opaque type (PascalCase, no brackets) vs primitive
    if is_opaque_candidate(left) && primitives.contains(&right) {
        let accessor = format!("{}_value", to_snake_case(left));
        return Some(format!(
            "use `{accessor}(...)` to extract the inner {right} before numeric operations"
        ));
    }
    if is_opaque_candidate(right) && primitives.contains(&left) {
        let accessor = format!("{}_value", to_snake_case(right));
        return Some(format!(
            "use `{accessor}(...)` to extract the inner {left} before numeric operations"
        ));
    }
    None
}

/// Heuristic: a type name that starts with uppercase, has no brackets/parens,
/// and isn't a known non-opaque ADT like Option/List is likely opaque.
fn is_opaque_candidate(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    let first = s.chars().next().unwrap();
    if !first.is_ascii_uppercase() {
        return false;
    }
    // Must not contain brackets (that would be a parameterized ADT display)
    if s.contains('[') || s.contains('(') {
        return false;
    }
    // Exclude well-known non-opaque ADTs
    !matches!(s, "Option" | "List" | "String")
}

fn to_snake_case(s: &str) -> String {
    let mut result = String::new();
    for (i, c) in s.chars().enumerate() {
        if c.is_ascii_uppercase() {
            if i > 0 {
                result.push('_');
            }
            result.push(c.to_ascii_lowercase());
        } else {
            result.push(c);
        }
    }
    result
}
