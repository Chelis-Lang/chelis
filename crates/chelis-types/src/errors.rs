//! Error types for the Chelis type checker.

use crate::unify::{TypeError, TypeErrorKind};

#[derive(Debug, Clone)]
pub struct CheckError {
    pub kind: CheckErrorKind,
    pub message: String,
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

#[derive(Debug, Clone)]
pub enum CheckErrorKind {
    TypeMismatch,
    PrecisionMismatch,
    DimensionMismatch,
    ArityMismatch,
    UnboundVariable,
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
    UnknownConstructor,
    NotAFunction,
    NonExhaustiveMatch,
    OccursCheck,
    CastNonTensor,
    TupleIndexOutOfBounds,
    UseAfterConsume,
    UnconsumedLinear,
    InvalidBorrow,
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
    /// RFC D-CHECK (`spec/design/opaque_invariants_rfc.md)`: a type
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
    /// closed builtin vocabulary (`BUILTIN_NAMES`). Call sites are
    /// dispatched builtin-first by name in both the host evaluator
    /// (`runtime/host_ops.rs::builtin_name`) and IR lowering
    /// (`lower.rs`), so a user definition with a builtin name can never
    /// be reached by name: pre-fix, the chelis#353 reproducer
    /// (`def sum`) checked clean, hit the builtin's arity error under
    /// eval, and segfaulted on the C backend. Rejected at declaration
    /// time instead (spec/04-type-system.md §8.6). Reef package modules
    /// are unaffected: their decls are internal-name-rewritten
    /// (`pkg__...`) before the checker runs, and their call sites are
    /// rewritten with them.
    BuiltinShadowing,
    Other,
}

impl CheckErrorKind {
    #[must_use]
    pub fn default_severity(&self) -> f64 {
        match self {
            CheckErrorKind::PrecisionMismatch | CheckErrorKind::DimensionMismatch => 0.8,
            CheckErrorKind::ArityMismatch => 0.7,
            CheckErrorKind::UnboundVariable => 0.6,
            CheckErrorKind::UnknownConstructor => 0.6,
            CheckErrorKind::NotAFunction => 0.5,
            CheckErrorKind::TypeMismatch => 0.5,
            CheckErrorKind::NonExhaustiveMatch => 0.7,
            CheckErrorKind::OccursCheck => 0.5,
            CheckErrorKind::CastNonTensor => 0.6,
            CheckErrorKind::TupleIndexOutOfBounds => 0.7,
            CheckErrorKind::UseAfterConsume => 0.9,
            CheckErrorKind::UnconsumedLinear => 0.9,
            CheckErrorKind::InvalidBorrow => 0.8,
            CheckErrorKind::CycleDetected => 0.9,
            CheckErrorKind::UnsupportedTensorPrecision => 0.8,
            CheckErrorKind::DuplicateDefinition => 0.9,
            CheckErrorKind::DuplicateModule => 0.9,
            CheckErrorKind::OpaqueTypeViolation => 0.8,
            CheckErrorKind::ReservedLinkerName => 0.9,
            CheckErrorKind::BuiltinShadowing => 0.9,
            CheckErrorKind::Other => 0.5,
        }
    }
}

impl CheckError {
    /// Create a new `CheckError` with default severity for its kind.
    #[must_use]
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

    /// Create with expected/got for structured error reports.
    #[must_use]
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
    #[must_use]
    pub fn at_offset(mut self, offset: usize) -> Self {
        self.span_offset = Some(offset);
        self
    }

    /// Set the external span identifier for this error.
    #[must_use]
    pub fn with_span_id(mut self, id: String) -> Self {
        self.span_id = Some(id);
        self
    }
}

impl From<TypeError> for CheckError {
    fn from(te: TypeError) -> Self {
        let kind = match te.kind {
            TypeErrorKind::TypeMismatch => CheckErrorKind::TypeMismatch,
            TypeErrorKind::PrecisionMismatch => CheckErrorKind::PrecisionMismatch,
            TypeErrorKind::DimensionMismatch => CheckErrorKind::DimensionMismatch,
            TypeErrorKind::ArityMismatch => CheckErrorKind::ArityMismatch,
            TypeErrorKind::OccursCheck => CheckErrorKind::OccursCheck,
            TypeErrorKind::NotAFunction => CheckErrorKind::NotAFunction,
        };
        let severity = match &kind {
            CheckErrorKind::PrecisionMismatch | CheckErrorKind::DimensionMismatch => 0.8,
            CheckErrorKind::ArityMismatch => 0.7,
            CheckErrorKind::UnboundVariable => 0.6,
            _ => 0.5,
        };
        let mut suggestions = match &kind {
            CheckErrorKind::PrecisionMismatch => vec!["Insert explicit cast".to_string()],
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

/// Enrich `TypeMismatch` suggestions by detecting common patterns:
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
    let primitives = ["f32", "f64", "int32", "int64", "bool"];
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
/// and isn't a known non-opaque ADT like Option/List/Result is likely opaque.
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
    !matches!(s, "Option" | "List" | "Result" | "String")
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
