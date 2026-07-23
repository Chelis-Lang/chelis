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
//! **Message format (frozen at Phase 1 exit):**
//! `unsupported: <what> on <context> (<stage>); <hint>` - branded with the
//! literal prefix `unsupported:` so tests and shells can match it. The
//! branded string is the RENDERING of the contract; machine surfaces carry
//! the structured kind and payload, and agents match those, never regex
//! over prose.

use std::fmt;

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
    /// A fully resolved host-lane value class with no ABI representation
    /// on the requesting target (the section C6.3 "known logical type
    /// without target representation" state). Distinct from [`HostType`],
    /// whose term never resolved at all.
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

/// The section C2 unsupported diagnostic. Construct with
/// [`Unsupported::new`] and render with `to_string()`; the rendering is
/// the frozen branded format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unsupported {
    /// What was encountered.
    pub what: UnsupportedKind,
    /// The context the encounter happened in (an op family, a target
    /// lane, a call position). Rendered as the `on <context>` clause.
    pub context: String,
    /// Which stage refused.
    pub stage: Stage,
    /// Source span when one exists (lowering/codegen thread it). Boxed
    /// so the Err variant stays small on the hot Result-typed emission
    /// paths (clippy::result_large_err); the section C2 shape is
    /// unchanged - the box is a representation detail.
    pub span: Option<Box<SpanRef>>,
    /// The supported alternative, when one exists. Not optional prose:
    /// sites without an alternative say why (deferred per a cited spec
    /// atom, or the tracking issue for the unbuilt support).
    pub hint: &'static str,
}

impl Unsupported {
    pub fn new(
        what: UnsupportedKind,
        context: impl Into<String>,
        stage: Stage,
        hint: &'static str,
    ) -> Self {
        Self {
            what,
            context: context.into(),
            stage,
            span: None,
            hint,
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
            "host-only builtin; run it under `chelis eval` or `chelis test`, or rewrite the caller to use tensor-lane primitives (spec/05-risc-primitives.md §3.6; chelis#705)",
        )
    }

    /// Attach a span reference.
    #[must_use]
    pub fn with_span(mut self, span: SpanRef) -> Self {
        self.span = Some(Box::new(span));
        self
    }
}

impl fmt::Display for Unsupported {
    /// The frozen section C2 rendering:
    /// `unsupported: <what> on <context> (<stage>); <hint>`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "unsupported: {} on {} ({}); {}",
            self.what, self.context, self.stage, self.hint
        )
    }
}

impl std::error::Error for Unsupported {}

#[cfg(test)]
mod tests {
    use super::*;

    /// The frozen message format, byte-checked: this string shape is what
    /// downstream shells and the chelis#687 rejected-cells corpus match.
    #[test]
    fn rendering_follows_the_frozen_branded_format() {
        let err = Unsupported::new(
            UnsupportedKind::Builtin("tensor_scan".to_string()),
            "`chelis build --target c` host emission",
            Stage::Codegen("c"),
            "host-only builtin; run it under `chelis eval` (chelis#705)",
        );
        assert_eq!(
            err.to_string(),
            "unsupported: builtin `tensor_scan` on `chelis build --target c` host \
             emission (codegen:c); host-only builtin; run it under `chelis eval` \
             (chelis#705)"
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
            let rendered =
                Unsupported::new(kind.clone(), "test context", Stage::Lowering, "hint text")
                    .to_string();
            assert!(
                rendered.starts_with("unsupported: "),
                "kind {kind:?} must render with the literal brand; got {rendered}"
            );
            assert!(
                rendered.contains("(lowering); hint text"),
                "kind {kind:?} must carry stage and hint; got {rendered}"
            );
        }
    }
}
