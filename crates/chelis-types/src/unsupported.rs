//! The shared unsupported-case diagnostic (chelis#730 Phase 1).
//!
//! One error kind, one shape, every stage - the section C2 contract of
//! `spec/design/loud_unsupported.md`. When any stage encounters a case it
//! does not support, its response is an [`Unsupported`] through that
//! stage's failure channel; substituting a value, type, dtype, kernel, or
//! emission is forbidden ([05-UNS-1], `spec/05-risc-primitives.md` section
//! 7).
//!
//! Defined here at the workspace bottom (open question 1 of the plan,
//! decided 2026-07-17: ONE shared type in `chelis-types`) so every
//! producer - chelis-ir lowering, the C/HIP/Metal backends, the
//! compiler-api - renders the same shape. Lowering's `LowerDiagnostic`
//! absorbs it by rendering [`Unsupported::to_string`] into its message,
//! not by replacement.
//!
//! **Message format:**
//! `unsupported: <what> on <context> (<stage>); <authority-kind> <citation>: <hint>` - branded with the
//! literal prefix `unsupported:` so tests and shells can match it. The
//! branded string is the RENDERING of the contract; machine surfaces carry
//! the structured kind and payload, and agents match those, never regex
//! over prose.
//!
//! **Status: the complete serialized surface is still the target.**
//! Nothing on this path derives `Serialize`, and nothing constructs
//! [`Stage::Checker`]. Chelis#1870 adds an in-process identity projection and
//! retains the original [`Unsupported`] in compiler-api and selected lowering
//! diagnostics, while serialized clients still receive
//! `kind: "unsupported_feature"` plus the rendering. Unimplemented rows also
//! still lack the future exact capability-table key: their issue is tracking
//! metadata, not semantic authority. Existing exact pins therefore remain.
//! Tracked by the section C2 status note in `spec/design/loud_unsupported.md`.
//!
//! **Executable form of that status (chelis#871).** A permission scoped
//! "until the prerequisites land" needs something that fires when they do,
//! or it outlives its premise silently. The day any of these four types
//! gains `Serialize`, its block below starts compiling, the
//! `cargo test -p chelis-types --doc` gate stage goes red, and whoever
//! made the structured surface real is told to come back here and to
//! section C2. Four separate blocks rather than four bounds in one: a
//! `compile_fail` block passes when ANYTHING in it fails to compile, so a
//! single combined block would keep passing after three of the four had
//! gained `Serialize`.
//!
//! Positive control first. A `compile_fail` block that fails because the
//! doctest harness cannot resolve `serde` at all is indistinguishable from
//! one that fails on the trait bound, so this block proves the bound is
//! what the four below are actually testing:
//!
//! ```
//! fn require_serialize<T: serde::Serialize>() {}
//! require_serialize::<i32>();
//! ```
//!
//! ```compile_fail
//! fn require_serialize<T: serde::Serialize>() {}
//! require_serialize::<chelis_types::unsupported::Unsupported>();
//! ```
//!
//! ```compile_fail
//! fn require_serialize<T: serde::Serialize>() {}
//! require_serialize::<chelis_types::unsupported::UnsupportedKind>();
//! ```
//!
//! ```compile_fail
//! fn require_serialize<T: serde::Serialize>() {}
//! require_serialize::<chelis_types::unsupported::Stage>();
//! ```
//!
//! ```compile_fail
//! fn require_serialize<T: serde::Serialize>() {}
//! require_serialize::<chelis_types::unsupported::SpanRef>();
//! ```

use std::fmt;
use std::num::NonZeroU32;

use crate::rejection_registry_generated::{REGISTERED_OPEN_ISSUES, REGISTERED_SPEC_ATOMS};

/// Stable rendered prefix carried by every structured unsupported identity.
pub const UNSUPPORTED_BRAND: &str = "unsupported:";

/// A registered numbered-spec atom.
///
/// Construction validates both the `[NN-GROUP-N]` grammar and membership in
/// the registry generated from normative blockquote atoms in `spec/00-12`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpecAtomRef(&'static str);

impl SpecAtomRef {
    const fn new(atom: &'static str) -> Result<Self, AuthorityConstructionError> {
        if !valid_atom_grammar(atom) {
            return Err(AuthorityConstructionError::MalformedAtom);
        }
        if response_contract_atom(atom) {
            return Err(AuthorityConstructionError::ResponseContractAtom);
        }
        if !registered_atom(atom) {
            return Err(AuthorityConstructionError::UnknownAtom);
        }
        Ok(Self(atom))
    }

    pub const fn as_str(self) -> &'static str {
        self.0
    }

    /// The generated registry, exposed read-only for byte-agreement tooling.
    pub const fn registry() -> &'static [&'static str] {
        REGISTERED_SPEC_ATOMS
    }
}

/// A nonzero member of the checked-in, live-validated issue manifest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IssueRef(NonZeroU32);

impl IssueRef {
    const fn new(number: u32) -> Result<Self, AuthorityConstructionError> {
        let Some(number) = NonZeroU32::new(number) else {
            return Err(AuthorityConstructionError::ZeroIssue);
        };
        if !registered_issue(number.get()) {
            return Err(AuthorityConstructionError::UnknownIssue);
        }
        Ok(Self(number))
    }

    pub const fn number(self) -> u32 {
        self.0.get()
    }

    /// The generated membership set, exposed read-only for agreement tooling.
    pub const fn registry() -> &'static [u32] {
        REGISTERED_OPEN_ISSUES
    }
}

/// The machine-readable distinction required by [05-UNS-5].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RejectionAuthorityKind {
    Deliberate,
    Unimplemented,
}

impl fmt::Display for RejectionAuthorityKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Deliberate => f.write_str("deliberate"),
            Self::Unimplemented => f.write_str("unimplemented"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum RejectionCitation {
    Atom(SpecAtomRef),
    Issue(IssueRef),
}

/// Opaque, validated authority for an unsupported diagnostic.
///
/// The fields are intentionally private. External code cannot manufacture a
/// citation-bearing value with an empty or unregistered identity:
///
/// ```compile_fail
/// use chelis_types::unsupported::RejectionAuthority;
/// let valid = chelis_types::deliberate_rejection!(
///     "[04-TOT-3]",
///     "malformed typed forms are rejected",
/// );
/// let _ = RejectionAuthority { hint: "forged", ..valid };
/// ```
///
/// The validated scalar constructors are private too. Downstream crates use
/// the literal macros, so adding a second public `IssueRef` constructor cannot
/// be composed with a public authority constructor to bypass the registry:
///
/// ```compile_fail
/// use chelis_types::unsupported::{IssueRef, RejectionAuthority};
/// let issue = IssueRef::new(879).unwrap();
/// let _ = RejectionAuthority::unimplemented(issue, "forged").unwrap();
/// ```
///
/// ```compile_fail
/// use chelis_types::unsupported::{RejectionAuthority, SpecAtomRef};
/// let atom = SpecAtomRef::new("[04-TOT-3]").unwrap();
/// let _ = RejectionAuthority::deliberate(atom, "forged").unwrap();
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RejectionAuthority {
    citation: RejectionCitation,
    hint: &'static str,
}

impl RejectionAuthority {
    const fn deliberate(
        atom: SpecAtomRef,
        hint: &'static str,
    ) -> Result<Self, AuthorityConstructionError> {
        if hint.is_empty() {
            return Err(AuthorityConstructionError::EmptyHint);
        }
        Ok(Self {
            citation: RejectionCitation::Atom(atom),
            hint,
        })
    }

    const fn unimplemented(
        issue: IssueRef,
        hint: &'static str,
    ) -> Result<Self, AuthorityConstructionError> {
        if hint.is_empty() {
            return Err(AuthorityConstructionError::EmptyHint);
        }
        Ok(Self {
            citation: RejectionCitation::Issue(issue),
            hint,
        })
    }

    pub const fn kind(self) -> RejectionAuthorityKind {
        match self.citation {
            RejectionCitation::Atom(_) => RejectionAuthorityKind::Deliberate,
            RejectionCitation::Issue(_) => RejectionAuthorityKind::Unimplemented,
        }
    }

    pub const fn atom(self) -> Option<SpecAtomRef> {
        match self.citation {
            RejectionCitation::Atom(atom) => Some(atom),
            RejectionCitation::Issue(_) => None,
        }
    }

    pub const fn issue(self) -> Option<IssueRef> {
        match self.citation {
            RejectionCitation::Issue(issue) => Some(issue),
            RejectionCitation::Atom(_) => None,
        }
    }

    pub const fn hint(self) -> &'static str {
        self.hint
    }

    pub fn citation(self) -> String {
        match self.citation {
            RejectionCitation::Atom(atom) => atom.as_str().to_owned(),
            RejectionCitation::Issue(issue) => format!("chelis#{}", issue.number()),
        }
    }
}

/// Sole downstream construction edge for a deliberate rejection.
///
/// This function is public only because exported macros expand in downstream
/// crates. It accepts the raw literal, performs both grammar and generated
/// registry validation, and never accepts a preconstructed `SpecAtomRef`.
#[doc(hidden)]
pub const fn __build_deliberate_rejection(
    atom: &'static str,
    hint: &'static str,
) -> Result<RejectionAuthority, AuthorityConstructionError> {
    let atom = match SpecAtomRef::new(atom) {
        Ok(atom) => atom,
        Err(error) => return Err(error),
    };
    RejectionAuthority::deliberate(atom, hint)
}

/// Sole downstream construction edge for an unimplemented rejection.
///
/// Like [`__build_deliberate_rejection`], this validates the raw literal
/// itself so no public `IssueRef` constructor is needed or exposed.
#[doc(hidden)]
pub const fn __build_unimplemented_rejection(
    issue: u32,
    hint: &'static str,
) -> Result<RejectionAuthority, AuthorityConstructionError> {
    let issue = match IssueRef::new(issue) {
        Ok(issue) => issue,
        Err(error) => return Err(error),
    };
    RejectionAuthority::unimplemented(issue, hint)
}

/// Failure to construct a typed rejection authority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthorityConstructionError {
    MalformedAtom,
    ResponseContractAtom,
    UnknownAtom,
    ZeroIssue,
    UnknownIssue,
    EmptyHint,
}

impl fmt::Display for AuthorityConstructionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::MalformedAtom => "malformed spec atom reference",
            Self::ResponseContractAtom => {
                "the unsupported-response contract does not decide a semantic case"
            }
            Self::UnknownAtom => "spec atom is absent from the generated registry",
            Self::ZeroIssue => "issue reference must be nonzero",
            Self::UnknownIssue => "issue is absent from the live-validated manifest",
            Self::EmptyHint => "rejection hint must not be empty",
        };
        f.write_str(message)
    }
}

impl std::error::Error for AuthorityConstructionError {}

const fn response_contract_atom(atom: &str) -> bool {
    let bytes = atom.as_bytes();
    bytes.len() == 10
        && bytes[0] == b'['
        && bytes[1] == b'0'
        && bytes[2] == b'5'
        && bytes[3] == b'-'
        && bytes[4] == b'U'
        && bytes[5] == b'N'
        && bytes[6] == b'S'
        && bytes[7] == b'-'
        && bytes[8] >= b'1'
        && bytes[8] <= b'6'
        && bytes[9] == b']'
}

const fn valid_atom_grammar(atom: &str) -> bool {
    let bytes = atom.as_bytes();
    if bytes.len() < 9
        || bytes[0] != b'['
        || bytes[1] < b'0'
        || bytes[1] > b'9'
        || bytes[2] < b'0'
        || bytes[2] > b'9'
        || bytes[3] != b'-'
        || bytes[bytes.len() - 1] != b']'
    {
        return false;
    }
    let mut index = 4;
    let group_start = index;
    while index < bytes.len() - 1 && bytes[index] >= b'A' && bytes[index] <= b'Z' {
        index += 1;
    }
    if index == group_start || index >= bytes.len() - 2 || bytes[index] != b'-' {
        return false;
    }
    index += 1;
    if bytes[index] < b'1' || bytes[index] > b'9' {
        return false;
    }
    index += 1;
    while index < bytes.len() - 1 {
        if bytes[index] < b'0' || bytes[index] > b'9' {
            return false;
        }
        index += 1;
    }
    true
}

const fn registered_atom(atom: &str) -> bool {
    let mut index = 0;
    while index < REGISTERED_SPEC_ATOMS.len() {
        if const_str_eq(atom, REGISTERED_SPEC_ATOMS[index]) {
            return true;
        }
        index += 1;
    }
    false
}

const fn const_str_eq(left: &str, right: &str) -> bool {
    let left = left.as_bytes();
    let right = right.as_bytes();
    if left.len() != right.len() {
        return false;
    }
    let mut index = 0;
    while index < left.len() {
        if left[index] != right[index] {
            return false;
        }
        index += 1;
    }
    true
}

const fn registered_issue(number: u32) -> bool {
    let mut index = 0;
    while index < REGISTERED_OPEN_ISSUES.len() {
        if number == REGISTERED_OPEN_ISSUES[index] {
            return true;
        }
        index += 1;
    }
    false
}

/// Build a deliberate authority in a const context. Invalid or unregistered
/// literals are compile errors instead of runtime fallbacks.
#[macro_export]
macro_rules! deliberate_rejection {
    ($atom:literal, $hint:expr $(,)?) => {{
        const AUTHORITY: $crate::unsupported::RejectionAuthority =
            match $crate::unsupported::__build_deliberate_rejection($atom, $hint) {
                Ok(authority) => authority,
                Err(_) => panic!("invalid deliberate rejection authority"),
            };
        AUTHORITY
    }};
}

/// Build an unimplemented authority in a const context. Invalid or
/// unregistered issue literals are compile errors instead of runtime fallbacks.
#[macro_export]
macro_rules! unimplemented_rejection {
    ($issue:literal, $hint:expr $(,)?) => {{
        const AUTHORITY: $crate::unsupported::RejectionAuthority =
            match $crate::unsupported::__build_unimplemented_rejection($issue, $hint) {
                Ok(authority) => authority,
                Err(_) => panic!("invalid unimplemented rejection authority"),
            };
        AUTHORITY
    }};
}

/// What was encountered: a closed enum plus payload, not a bare string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnsupportedKind {
    /// A builtin with no emission arm for the requesting target/context.
    Builtin(String),
    /// An IR op with no kernel/emission for the requesting target.
    Op(String),
    /// A dtype with no representation in the requesting context.
    Dtype(String),
    /// An effect kind the lowering does not recognize.
    EffectKind(String),
    /// A construct or argument shape the stage cannot lower/emit.
    Construct(String),
    /// A host-lane type that never resolved to a concrete representation.
    HostType(String),
    /// A host-lane value class the requesting target cannot represent:
    /// either a fully resolved type with no target ABI (the section C6.3
    /// "known logical type without target representation" state) or a
    /// callable use the host lowerer could not resolve to any
    /// representable value (the internal unresolved-callee markers).
    /// Distinct from [`HostType`], which is reserved for a type TERM
    /// that never resolved.
    ///
    /// [`HostType`]: UnsupportedKind::HostType
    HostAbi(String),
}

impl fmt::Display for UnsupportedKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnsupportedKind::Builtin(name) => write!(f, "builtin `{name}`"),
            UnsupportedKind::Op(name) => write!(f, "op `{name}`"),
            UnsupportedKind::Dtype(name) => write!(f, "dtype `{name}`"),
            UnsupportedKind::EffectKind(name) => write!(f, "effect kind `{name}`"),
            UnsupportedKind::Construct(what) => write!(f, "{what}"),
            UnsupportedKind::HostType(name) => write!(f, "unresolved host type `{name}`"),
            UnsupportedKind::HostAbi(name) => {
                write!(f, "{name} with no target ABI representation")
            }
        }
    }
}

/// Which stage refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Checker,
    Lowering,
    /// Code generation for a named target (`"c"`, `"hip"`, `"metal"`).
    Codegen(&'static str),
    Runtime,
}

impl fmt::Display for Stage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Stage::Checker => write!(f, "checker"),
            Stage::Lowering => write!(f, "lowering"),
            Stage::Codegen(target) => write!(f, "codegen:{target}"),
            Stage::Runtime => write!(f, "runtime"),
        }
    }
}

/// Source location reference, self-contained so the workspace-bottom crate
/// needs no AST dependency. Producers with a richer span thread it here;
/// `None` fields are legal when the producing stage genuinely has none.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SpanRef {
    /// Byte offset into the source, when known.
    pub offset: Option<usize>,
    /// Byte length, when known.
    pub len: Option<usize>,
    /// Producer-assigned span id (the `span_id` metadata), when known.
    pub span_id: Option<String>,
}

/// Wording-independent identity available from a production [`Unsupported`].
///
/// The free-form authority hint is deliberately absent. Unimplemented rows
/// retain their typed disposition and tracking issue, but the future exact
/// capability-table key is not available yet, so existing exact pins remain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnsupportedIdentity {
    pub brand: &'static str,
    pub what: UnsupportedKind,
    pub context: String,
    pub stage: Stage,
    pub span: Option<Box<SpanRef>>,
    pub disposition: RejectionAuthorityKind,
    pub atom: Option<SpecAtomRef>,
    pub tracking_issue: Option<IssueRef>,
    pub supported_alternative: Option<String>,
}

/// The section C2 unsupported diagnostic. Construct with
/// [`Unsupported::new`] and render with `to_string()`; the rendering exposes
/// the validated authority kind and citation after the frozen brand/context
/// clauses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unsupported {
    /// What was encountered. The closed typed subject is boxed so dynamic
    /// payload growth cannot push this common Result error over Clippy's
    /// `result_large_err` threshold.
    pub what: Box<UnsupportedKind>,
    /// The context the encounter happened in (an op family, a target
    /// lane, a call position). Rendered as the `on <context>` clause.
    /// Boxed string storage keeps this Result error below Clippy's
    /// `result_large_err` threshold without changing its value semantics.
    pub context: Box<str>,
    /// Which stage refused.
    pub stage: Stage,
    /// Source span when one exists (lowering/codegen thread it). Boxed
    /// so the Err variant stays small on the hot Result-typed emission
    /// paths (clippy::result_large_err); the section C2 shape is
    /// unchanged - the box is a representation detail.
    pub span: Option<Box<SpanRef>>,
    /// Typed atom-or-issue authority.
    pub authority: RejectionAuthority,
    /// A supported route the caller can select, when one exists.
    pub supported_alternative: Option<Box<str>>,
}

impl Unsupported {
    pub fn new(
        what: impl Into<Box<UnsupportedKind>>,
        context: impl Into<Box<str>>,
        stage: Stage,
        authority: RejectionAuthority,
    ) -> Self {
        Self {
            what: what.into(),
            context: context.into(),
            stage,
            span: None,
            authority,
            supported_alternative: None,
        }
    }

    /// The single target-build diagnostic for a host-runtime-only builtin.
    /// Both the compiler API and CLI call this before host expression
    /// lowering, then retain their concrete-HostProgram scans as a second
    /// boundary. Keeping construction here prevents the two public build
    /// entry points from drifting in kind, stage, or remediation text.
    pub fn compiled_host_only_builtin(name: impl Into<String>, target: &'static str) -> Self {
        Self::new(
            UnsupportedKind::Builtin(name.into()),
            format!("`chelis build --target {target}` host emission"),
            Stage::Codegen(target),
            crate::deliberate_rejection!(
                "[05-HOST-1]",
                "host-runtime builders are intentionally excluded from compiled targets; run under `chelis eval` or `chelis test`, or rewrite the caller to use tensor-lane primitives"
            ),
        )
        .with_supported_alternative(
            "run under `chelis eval` or `chelis test`, or rewrite the caller to use tensor-lane primitives",
        )
    }

    /// Attach a span reference.
    #[must_use]
    pub fn with_span(mut self, span: SpanRef) -> Self {
        self.span = Some(Box::new(span));
        self
    }

    /// Attach the supported route for structured consumers.
    #[must_use]
    pub fn with_supported_alternative(mut self, alternative: impl Into<Box<str>>) -> Self {
        self.supported_alternative = Some(alternative.into());
        self
    }

    /// Project blocking fields without depending on rendered hint wording.
    pub fn identity(&self) -> UnsupportedIdentity {
        UnsupportedIdentity {
            brand: UNSUPPORTED_BRAND,
            what: (*self.what).clone(),
            context: self.context.to_string(),
            stage: self.stage,
            span: self.span.clone(),
            disposition: self.authority.kind(),
            atom: self.authority.atom(),
            tracking_issue: self.authority.issue(),
            supported_alternative: self.supported_alternative.as_deref().map(str::to_owned),
        }
    }
}

impl fmt::Display for Unsupported {
    /// The section C2 rendering exposes the validated authority rather than
    /// trusting a citation embedded in free-form hint prose.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "unsupported: {} on {} ({}); {} {}: {}",
            self.what,
            self.context,
            self.stage,
            self.authority.kind(),
            self.authority.citation(),
            self.authority.hint()
        )
    }
}

impl std::error::Error for Unsupported {}

#[cfg(test)]
mod tests {
    use super::*;

    /// The authority-bearing message format, byte-checked.
    #[test]
    fn rendering_exposes_the_authority_bearing_format() {
        let err = Unsupported::new(
            UnsupportedKind::Builtin("tensor_scan".to_string()),
            "`chelis build --target c` host emission",
            Stage::Codegen("c"),
            crate::deliberate_rejection!(
                "[05-HOST-1]",
                "host-runtime builders are intentionally excluded from compiled targets; run under `chelis eval`"
            ),
        );
        assert_eq!(
            err.to_string(),
            "unsupported: builtin `tensor_scan` on `chelis build --target c` host \
             emission (codegen:c); deliberate [05-HOST-1]: host-runtime builders are \
             intentionally excluded from compiled targets; run under `chelis eval`"
        );
    }

    /// Every kind renders with its payload; the brand prefix is invariant.
    #[test]
    fn every_kind_renders_branded() {
        let kinds = [
            UnsupportedKind::Builtin("floor".into()),
            UnsupportedKind::Op("MaxReduce".into()),
            UnsupportedKind::Dtype("int64".into()),
            UnsupportedKind::EffectKind("teleport".into()),
            UnsupportedKind::Construct("a non-literal window list".into()),
            UnsupportedKind::HostType("f16".into()),
            UnsupportedKind::HostAbi("function value `increment`".into()),
        ];
        for kind in kinds {
            let rendered = Unsupported::new(
                kind.clone(),
                "test context",
                Stage::Lowering,
                crate::deliberate_rejection!("[04-TOT-3]", "hint text"),
            )
            .to_string();
            assert!(
                rendered.starts_with("unsupported: "),
                "kind {kind:?} must render with the literal brand; got {rendered}"
            );
            assert!(
                rendered.contains("(lowering); deliberate [04-TOT-3]: hint text"),
                "kind {kind:?} must carry stage, authority, and hint; got {rendered}"
            );
        }
    }
}
