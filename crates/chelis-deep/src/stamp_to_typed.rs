//! Role-directed stamp pass: `Vec<RawExpr>` → `Result<Vec<Expr>, StampError>`.
//!
//! Walks top-down, consulting `child_stamp_role` at each child to decide
//! whether a list decodes as a Node, BareList, or UnknownForm.

use crate::Metadata;
use crate::ast::{Atom, Expr};
use crate::node::Node;
use crate::raw::{RawAtom, RawExpr};
use crate::role::{
    BypassExpectation, ChildStampRole, TypeSyntaxRole, bypass_child_expectation, child_stamp_role,
    is_declaration_tag, is_pattern_tag, type_syntax_child_role, type_syntax_role_accepts_tag,
};
use crate::span::Span;
use crate::tag::DeepTag;

/// An error from the stamp pass — structurally invalid input that cannot
/// produce a well-typed AST.
#[derive(Debug, Clone, PartialEq)]
pub struct StampError {
    pub kind: StampErrorKind,
    pub span: Span,
}

/// The syntactic class of a rejected form that has no head to name
/// (`spec/03-deep-syntax.md` [03-PROG-2]).
///
/// The set is closed by that rule: it partitions what the grammar can put in
/// a position where a tagged node was required. A rejection identifies the
/// offending form by its head symbol when it has one and by its class when it
/// does not; [03-PROG-2] forbids substituting a placeholder for either.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FormClass {
    BareIdentifier,
    BareIntegerLiteral,
    BareFloatLiteral,
    BareStringLiteral,
    BareBooleanLiteral,
    EmptyList,
    ListWithoutTagSymbol,
    MetadataMap,
    MetadataAnnotatedForm,
    ExtensionData,
}

impl FormClass {
    /// The exact spelling [03-PROG-2] fixes for this class.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BareIdentifier => "a bare identifier",
            Self::BareIntegerLiteral => "a bare integer literal",
            Self::BareFloatLiteral => "a bare float literal",
            Self::BareStringLiteral => "a bare string literal",
            Self::BareBooleanLiteral => "a bare boolean literal",
            Self::EmptyList => "an empty list",
            Self::ListWithoutTagSymbol => "a list without a tag symbol",
            Self::MetadataMap => "a metadata map",
            Self::ExtensionData => "opaque extension data",
            Self::MetadataAnnotatedForm => "a metadata-annotated form",
        }
    }

    /// Classify a raw form that reached a slot requiring a tagged node.
    pub fn of(raw: &RawExpr) -> Self {
        match raw {
            RawExpr::Atom(RawAtom::Symbol(_), _) => Self::BareIdentifier,
            RawExpr::Atom(RawAtom::Int(_), _) => Self::BareIntegerLiteral,
            RawExpr::Atom(RawAtom::Float(_), _) => Self::BareFloatLiteral,
            RawExpr::Atom(RawAtom::Str(_), _) => Self::BareStringLiteral,
            RawExpr::Atom(RawAtom::Bool(_), _) => Self::BareBooleanLiteral,
            RawExpr::List(elements, _) if elements.is_empty() => Self::EmptyList,
            RawExpr::List(..) => Self::ListWithoutTagSymbol,
            RawExpr::Map(..) => Self::MetadataMap,
            RawExpr::ExtensionData(_) => Self::ExtensionData,
            RawExpr::MetaExpr { .. } => Self::MetadataAnnotatedForm,
        }
    }
}

impl std::fmt::Display for FormClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// How a rejected form is identified in a diagnostic
/// (`spec/03-deep-syntax.md` [03-PROG-2]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FormIdentity {
    /// A form headed by a symbol, identified by that symbol.
    Head(String),
    /// A form with no head, identified by its syntactic class.
    Class(FormClass),
}

impl FormIdentity {
    /// Identify a raw form that reached a slot requiring a tagged node.
    pub fn of(raw: &RawExpr) -> Self {
        if let RawExpr::List(elements, _) = raw
            && let Some(RawExpr::Atom(RawAtom::Symbol(head), _)) = elements.first()
        {
            return Self::Head(head.clone());
        }
        Self::Class(FormClass::of(raw))
    }
}

impl std::fmt::Display for FormIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Head(symbol) => write!(f, "`{symbol}`"),
            Self::Class(class) => write!(f, "{class}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum StampErrorKind {
    /// A bare name appeared at a RuntimeExpr position.
    NameAtExprSlot { name: String },
    /// A list at a Type position had an undecodable head.
    UndecodableTypeHead { form: FormIdentity },
    /// A serialized type fragment did not satisfy its recursive grammar role.
    RequiresTypeSyntaxRole {
        expected: TypeSyntaxRole,
        got: FormIdentity,
    },
    /// A bypass slot required a declaration but the form was not one.
    RequiresDeclaration { form: FormIdentity },
    /// A bypass slot required a specific tag but got something else.
    RequiresTag {
        expected: DeepTag,
        got: FormIdentity,
    },
    /// A bypass slot required a pattern but the form was not one.
    RequiresPattern { form: FormIdentity },
    /// A list was missing the metadata map at element 1.
    MissingMetaMap,
    /// The program text yielded no top-level form ([03-PROG-3]).
    EmptyProgram,
    /// Node construction failed (arity or role violation).
    NodeError(crate::node::NodeError),
}

impl std::fmt::Display for StampError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.kind {
            StampErrorKind::NameAtExprSlot { name } => {
                write!(
                    f,
                    "bare name `{name}` at expression slot; use `(var {{}} {name})`"
                )
            }
            StampErrorKind::UndecodableTypeHead { form } => {
                write!(f, "undecodable type head {form}")
            }
            StampErrorKind::RequiresTypeSyntaxRole { expected, got } => {
                write!(f, "expected {expected:?} type syntax, got {got}")
            }
            StampErrorKind::RequiresDeclaration { form } => {
                write!(f, "expected declaration, got {form}")
            }
            StampErrorKind::RequiresTag { expected, got } => {
                write!(f, "expected `{}`, got {got}", expected.as_str())
            }
            StampErrorKind::RequiresPattern { form } => {
                write!(f, "expected pattern, got {form}")
            }
            StampErrorKind::MissingMetaMap => write!(f, "list missing metadata map at index 1"),
            // [03-PROG-3] fixes this rejection's self-identification. The
            // `empty program` prefix is also the canonical CLI spelling for
            // the same verdict, so `check` and `build` keep agreeing.
            StampErrorKind::EmptyProgram => write!(
                f,
                "empty program: a Deep program requires at least one top-level form"
            ),
            StampErrorKind::NodeError(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for StampError {}

fn extension_at_program_slot(data: crate::ExtensionData) -> StampError {
    StampError {
        span: data.span(),
        kind: StampErrorKind::NodeError(crate::node::NodeError::Metadata(
            crate::annotations::invalid(
                "extension",
                data.span(),
                "program syntax, not opaque extension data",
            ),
        )),
    }
}

/// Convert raw parser output to typed AST using role-directed stamping.
///
/// Top-level expressions are treated as Module children (bypass expecting
/// declarations).
pub fn stamp_to_typed(raw_exprs: Vec<RawExpr>) -> Result<Vec<Expr>, StampError> {
    validate_metadata_raw(&raw_exprs)?;
    let mut out = Vec::with_capacity(raw_exprs.len());
    for raw in raw_exprs {
        out.push(stamp_as_bypass_declaration(raw)?);
    }
    Ok(out)
}

/// Stamp raw expressions leniently — treats each top-level expression as a
/// bare/syntax position (no declaration requirement). Symbols are kept as
/// `Atom::Name`, known tags in list-head position are decoded to Nodes.
/// This is the entry point for `parse_str` which handles arbitrary Deep
/// fragments, not just programs.
pub fn stamp_exprs_lenient(raw_exprs: Vec<RawExpr>) -> Result<Vec<Expr>, StampError> {
    validate_metadata_raw(&raw_exprs)?;
    let mut out = Vec::with_capacity(raw_exprs.len());
    for raw in raw_exprs {
        out.push(stamp_bare(raw)?);
    }
    Ok(out)
}

/// Stamp raw expressions where every top-level form occupies a RuntimeExpr
/// slot (chelis#1088).
///
/// A public text boundary that takes an *expression* fragment (a replacement
/// function body, for instance) names that role here rather than falling
/// through to the lenient bare/syntax stamp. A bare name is then the same
/// ingress rejection it is inside a `(def ...)`, not an `Atom::Name` the
/// consumer has to re-diagnose.
pub fn stamp_runtime_exprs(raw_exprs: Vec<RawExpr>) -> Result<Vec<Expr>, StampError> {
    validate_metadata_raw(&raw_exprs)?;
    let mut out = Vec::with_capacity(raw_exprs.len());
    for raw in raw_exprs {
        out.push(stamp_runtime_expr(raw)?);
    }
    Ok(out)
}

/// Stamp raw expressions where every top-level form must carry `expected`
/// as its head tag (chelis#1088).
///
/// The role-directed counterpart for a text boundary whose contract names one
/// exact tag — a `(params {} ...)` replacement, for instance. Anything else
/// is a [`StampErrorKind::RequiresTag`] rejection at ingress.
pub fn stamp_as_tagged(
    raw_exprs: Vec<RawExpr>,
    expected: DeepTag,
) -> Result<Vec<Expr>, StampError> {
    validate_metadata_raw(&raw_exprs)?;
    let mut out = Vec::with_capacity(raw_exprs.len());
    for raw in raw_exprs {
        out.push(stamp_as_bypass_tag(raw, expected)?);
    }
    Ok(out)
}

/// Stamp a `.dp` file's raw expressions into typed AST.
///
/// `.dp` files may contain:
/// - A single or multiple top-level `(module ...)` wrappers
/// - Bare declarations at top level (hand-written `.dp`)
/// - A mix of modules and bare declarations
///
/// Each top-level form is stamped as either a module Node (if headed by
/// `module`) or a declaration Node (via `stamp_as_bypass_declaration`).
pub fn stamp_deep_file(raw_exprs: Vec<RawExpr>) -> Result<Vec<Expr>, StampError> {
    // [03-PROG-1] requires at least one top-level form, and [03-PROG-3]
    // fixes how the zero-form rejection identifies itself. Enforcing the
    // cardinality here rather than in each consumer is what makes every
    // door that reads a `.dp` program inherit it: the generic
    // `compiler::parse`/`decompile` ingress, the check/compile/eval
    // pipeline, `validate --deep`, the authoring surfaces, the lint rule,
    // the snippet checker, and the editor.
    if raw_exprs.is_empty() {
        return Err(StampError {
            kind: StampErrorKind::EmptyProgram,
            // The position at which a top-level form was required: the end
            // of the input, which for empty text is offset 0.
            span: Span::new(0, 0),
        });
    }
    // Root-role rejection owns its diagnostic before inspecting annotations.
    for raw in &raw_exprs {
        if !top_level_tag(raw)
            .is_some_and(|tag| tag == DeepTag::Module || crate::role::is_declaration_tag(tag))
        {
            return Err(StampError {
                kind: StampErrorKind::RequiresDeclaration {
                    form: FormIdentity::of(raw),
                },
                span: raw.span(),
            });
        }
    }
    validate_metadata_raw(&raw_exprs)?;
    let mut out = Vec::with_capacity(raw_exprs.len());
    for raw in raw_exprs {
        if top_level_tag(&raw) == Some(DeepTag::Module) {
            let span = raw.span();
            let RawExpr::List(elements, _) = raw else {
                unreachable!()
            };
            out.push(build_node(DeepTag::Module, elements, span)?);
        } else {
            out.push(stamp_as_bypass_declaration(raw)?);
        }
    }
    Ok(out)
}

/// Peek at the tag of a top-level raw list expression.
fn top_level_tag(raw: &RawExpr) -> Option<DeepTag> {
    if let RawExpr::List(elements, _) = raw
        && let Some(RawExpr::Atom(RawAtom::Symbol(name), _)) = elements.first()
    {
        return DeepTag::parse(name);
    }
    None
}

/// Stamp a raw expression in a specific role context.
fn stamp_in_role(
    raw: RawExpr,
    role: ChildStampRole,
    parent_tag: DeepTag,
    index: usize,
) -> Result<Expr, StampError> {
    match role {
        ChildStampRole::RuntimeExpr => stamp_runtime_expr(raw),
        ChildStampRole::Type => stamp_type(raw),
        ChildStampRole::Syntax | ChildStampRole::Binder | ChildStampRole::Selector => {
            stamp_bare(raw)
        }
        ChildStampRole::EffectHandler => stamp_effect_handler(raw),
        ChildStampRole::ExplicitInferenceBypass => {
            let expectation = bypass_child_expectation(parent_tag, index);
            stamp_bypass(raw, expectation)
        }
    }
}

// ── RuntimeExpr ──────────────────────────────────────────────────────

/// Stamp a child at an expression position.
///
/// A bare identifier here is the `spec/03-deep-syntax.md` [03-ROLE-2]
/// ingress rejection: a name is not an expression, and the diagnostic
/// names the `(var {} ...)` spelling that is one.
pub(crate) fn stamp_runtime_expr(raw: RawExpr) -> Result<Expr, StampError> {
    match raw {
        RawExpr::Atom(RawAtom::Symbol(name), span) => Err(StampError {
            kind: StampErrorKind::NameAtExprSlot { name },
            span,
        }),
        RawExpr::ExtensionData(data) => Err(extension_at_program_slot(data)),
        RawExpr::Atom(atom, span) => Ok(Expr::Atom(convert_atom(atom), span)),
        RawExpr::List(elements, span) => {
            if elements.is_empty() {
                // Empty list `()` at expression position (e.g., no-guard in arm).
                return Ok(Expr::BareList(vec![], span));
            }
            stamp_list_as_node_or_unknown(elements, span)
        }
        RawExpr::Map(entries, span) => stamp_map(entries, span),
        RawExpr::MetaExpr {
            entries,
            expr,
            span,
        } => stamp_meta_expr(entries, *expr, span),
    }
}

// ── Type ─────────────────────────────────────────────────────────────

/// Stamp a child at a type position.
///
/// A bare identifier here is a name per [03-ROLE-1] (a dtype spelling, a
/// type parameter), so atoms pass through; a non-empty list must decode
/// to a vocabulary node, because type syntax is closed and an undecodable
/// head has no type reading to fall back to. Nominal applications use the
/// dedicated recursive type grammar so their argument slots cannot inherit
/// the broader `Type` role and admit rank spreads ([04-ADT-4]).
pub(crate) fn stamp_type(raw: RawExpr) -> Result<Expr, StampError> {
    match raw {
        RawExpr::ExtensionData(data) => Err(extension_at_program_slot(data)),
        RawExpr::Atom(atom, span) => Ok(Expr::Atom(convert_atom(atom), span)),
        RawExpr::List(elements, span) => {
            if elements.is_empty() {
                return Ok(Expr::BareList(vec![], span));
            }
            let (form, tag_opt) = decode_list_head(&elements);
            match tag_opt {
                Some(tag) => build_node(tag, elements, span),
                None => Err(StampError {
                    kind: StampErrorKind::UndecodableTypeHead { form },
                    span,
                }),
            }
        }
        RawExpr::Map(entries, span) => stamp_map(entries, span),
        RawExpr::MetaExpr {
            entries,
            expr,
            span,
        } => stamp_meta_expr(entries, *expr, span),
    }
}

/// Stamp one complete serialized type representation with the recursive
/// type/dimension/rank grammar. Unlike the broader in-program `Type` child
/// role, this public-boundary form has no bare cast-target or record-name
/// alternatives.
pub(crate) fn stamp_serialized_type(raw: RawExpr) -> Result<Expr, StampError> {
    stamp_type_in_role(raw, TypeSyntaxRole::Type)
}

fn stamp_type_in_role(raw: RawExpr, expected: TypeSyntaxRole) -> Result<Expr, StampError> {
    if expected == TypeSyntaxRole::Name {
        return match raw {
            RawExpr::Atom(RawAtom::Symbol(name), span) => Ok(Expr::Atom(Atom::Name(name), span)),
            other => Err(StampError {
                kind: StampErrorKind::RequiresTypeSyntaxRole {
                    expected,
                    got: FormIdentity::of(&other),
                },
                span: other.span(),
            }),
        };
    }
    if expected == TypeSyntaxRole::Integer {
        return match raw {
            RawExpr::Atom(RawAtom::Int(value), span) => Ok(Expr::Atom(Atom::Int(value), span)),
            other => Err(StampError {
                kind: StampErrorKind::RequiresTypeSyntaxRole {
                    expected,
                    got: FormIdentity::of(&other),
                },
                span: other.span(),
            }),
        };
    }

    let span = raw.span();
    match raw {
        RawExpr::List(elements, span) => {
            let (form, tag_opt) = decode_list_head(&elements);
            match tag_opt {
                Some(tag) if type_syntax_role_accepts_tag(expected, tag) => {
                    build_type_node(tag, elements, span)
                }
                Some(_) => Err(StampError {
                    kind: StampErrorKind::RequiresTypeSyntaxRole {
                        expected,
                        got: form,
                    },
                    span,
                }),
                None => Err(StampError {
                    kind: StampErrorKind::UndecodableTypeHead { form },
                    span,
                }),
            }
        }
        other => Err(StampError {
            kind: StampErrorKind::RequiresTypeSyntaxRole {
                expected,
                got: FormIdentity::of(&other),
            },
            span,
        }),
    }
}

// ── Syntax/Binder/Selector → Node (if vocabulary head) or BareList ───

pub(crate) fn stamp_bare(raw: RawExpr) -> Result<Expr, StampError> {
    match raw {
        RawExpr::ExtensionData(data) => Err(extension_at_program_slot(data)),
        RawExpr::Atom(atom, span) => Ok(Expr::Atom(convert_atom(atom), span)),
        RawExpr::List(elements, span) => stamp_bare_list(elements, span),
        RawExpr::Map(entries, span) => stamp_map(entries, span),
        RawExpr::MetaExpr {
            entries,
            expr,
            span,
        } => stamp_meta_expr(entries, *expr, span),
    }
}

/// A list at a Binder/Syntax/Selector position: attempt vocabulary decode
/// first. If the head is a known tag AND element 1 is a metadata map, stamp
/// as a Node (so `(params {} x)` becomes Node(Params)). Otherwise fall
/// through to BareList (so `(x y z)` becomes BareList).
///
/// The conjunction is `spec/03-deep-syntax.md` [03-ROLE-3]'s disambiguator,
/// and both fall-through rows are deliberate acceptance, not missed decode:
/// a tag-word head without a metadata map is a real structural list (an
/// import name list `(copy fill)`), and a metadata map behind an ordinary
/// name head is a real annotated parameter (`(x {type: ...})`). Neither
/// half of the conjunction may reinterpret the list on its own.
fn stamp_bare_list(elements: Vec<RawExpr>, span: Span) -> Result<Expr, StampError> {
    // Try vocabulary decode: head must be a known DeepTag symbol AND the
    // list must have at least 2 elements with a map at index 1.
    if elements.len() >= 2
        && let Some(RawExpr::Atom(RawAtom::Symbol(head_str), _)) = elements.first()
        && let Some(tag) = DeepTag::parse(head_str)
        && matches!(elements.get(1), Some(RawExpr::Map(..)))
    {
        return build_node(tag, elements, span);
    }
    // Not a vocabulary-headed node — produce BareList.
    let mut out = Vec::with_capacity(elements.len());
    for elem in elements {
        out.push(stamp_bare(elem)?);
    }
    Ok(Expr::BareList(out, span))
}

// ── EffectHandler ────────────────────────────────────────────────────

fn stamp_effect_handler(raw: RawExpr) -> Result<Expr, StampError> {
    match raw {
        RawExpr::ExtensionData(data) => Err(extension_at_program_slot(data)),
        RawExpr::Atom(atom, span) => Ok(Expr::Atom(convert_atom(atom), span)),
        RawExpr::List(elements, span) => {
            if elements.is_empty() {
                return Ok(Expr::BareList(vec![], span));
            }
            stamp_list_as_node_or_unknown(elements, span)
        }
        RawExpr::Map(entries, span) => stamp_map(entries, span),
        RawExpr::MetaExpr {
            entries,
            expr,
            span,
        } => stamp_meta_expr(entries, *expr, span),
    }
}

// ── Bypass ───────────────────────────────────────────────────────────

fn stamp_bypass(raw: RawExpr, expectation: BypassExpectation) -> Result<Expr, StampError> {
    match expectation {
        BypassExpectation::RequiresDeclaration => stamp_as_bypass_declaration(raw),
        BypassExpectation::RequiresTag(expected_tag) => stamp_as_bypass_tag(raw, expected_tag),
        BypassExpectation::RequiresPattern => stamp_as_bypass_pattern(raw),
        BypassExpectation::FormExpecting => stamp_form_expecting(raw),
    }
}

fn stamp_as_bypass_declaration(raw: RawExpr) -> Result<Expr, StampError> {
    let span = raw.span();
    match raw {
        RawExpr::List(elements, span) => {
            let (form, tag_opt) = decode_list_head(&elements);
            match tag_opt {
                Some(tag) if is_declaration_tag(tag) => build_node(tag, elements, span),
                _ => Err(StampError {
                    kind: StampErrorKind::RequiresDeclaration { form },
                    span,
                }),
            }
        }
        other => Err(StampError {
            kind: StampErrorKind::RequiresDeclaration {
                form: FormIdentity::of(&other),
            },
            span,
        }),
    }
}

fn stamp_as_bypass_tag(raw: RawExpr, expected_tag: DeepTag) -> Result<Expr, StampError> {
    let span = raw.span();
    match raw {
        RawExpr::List(elements, span) => {
            let (form, tag_opt) = decode_list_head(&elements);
            match tag_opt {
                Some(tag) if tag == expected_tag => build_node(tag, elements, span),
                _ => Err(StampError {
                    kind: StampErrorKind::RequiresTag {
                        expected: expected_tag,
                        got: form,
                    },
                    span,
                }),
            }
        }
        other => Err(StampError {
            kind: StampErrorKind::RequiresTag {
                expected: expected_tag,
                got: FormIdentity::of(&other),
            },
            span,
        }),
    }
}

fn stamp_as_bypass_pattern(raw: RawExpr) -> Result<Expr, StampError> {
    let span = raw.span();
    match raw {
        RawExpr::List(elements, span) => {
            let (form, tag_opt) = decode_list_head(&elements);
            match tag_opt {
                Some(tag) if is_pattern_tag(tag) => build_node(tag, elements, span),
                _ => Err(StampError {
                    kind: StampErrorKind::RequiresPattern { form },
                    span,
                }),
            }
        }
        other => Err(StampError {
            kind: StampErrorKind::RequiresPattern {
                form: FormIdentity::of(&other),
            },
            span,
        }),
    }
}

fn stamp_form_expecting(raw: RawExpr) -> Result<Expr, StampError> {
    match raw {
        RawExpr::Atom(RawAtom::Symbol(name), span) => Err(StampError {
            kind: StampErrorKind::NameAtExprSlot { name },
            span,
        }),
        RawExpr::ExtensionData(data) => Err(extension_at_program_slot(data)),
        RawExpr::Atom(atom, span) => Ok(Expr::Atom(convert_atom(atom), span)),
        RawExpr::List(elements, span) => {
            if elements.is_empty() {
                return Ok(Expr::BareList(vec![], span));
            }
            stamp_list_as_node_or_unknown(elements, span)
        }
        RawExpr::Map(entries, span) => stamp_map(entries, span),
        RawExpr::MetaExpr {
            entries,
            expr,
            span,
        } => stamp_meta_expr(entries, *expr, span),
    }
}

// ── Helpers ──────────────────────────────────────────────────────────

/// Identify a list's head for a slot that required a tagged node, and decode
/// it when it names one ([03-PROG-2]).
///
/// A list with no first element and a list whose first element is not a
/// symbol both have no head to name, so each is identified by its syntactic
/// class rather than by a placeholder string.
fn decode_list_head(elements: &[RawExpr]) -> (FormIdentity, Option<DeepTag>) {
    match elements.first() {
        None => (FormIdentity::Class(FormClass::EmptyList), None),
        Some(RawExpr::Atom(RawAtom::Symbol(symbol), _)) => {
            (FormIdentity::Head(symbol.clone()), DeepTag::parse(symbol))
        }
        Some(_) => (FormIdentity::Class(FormClass::ListWithoutTagSymbol), None),
    }
}

/// Build a Node from a raw list whose head decoded as `tag`.
fn build_node(tag: DeepTag, elements: Vec<RawExpr>, span: Span) -> Result<Expr, StampError> {
    // A nominal application carries its own recursive child grammar wherever
    // it is decoded, including inside `type:` metadata that crosses a Syntax
    // role. Centralizing the dispatch here prevents those alternate carriers
    // from reopening rank spreads in nominal argument slots (chelis#1125).
    if tag == DeepTag::TAdt {
        return build_type_node(tag, elements, span);
    }

    if elements.len() < 2 {
        return Err(StampError {
            kind: StampErrorKind::MissingMetaMap,
            span,
        });
    }
    let mut iter = elements.into_iter();
    let _head = iter.next(); // skip the tag symbol
    let meta_raw = iter.next().unwrap();
    let meta = match meta_raw {
        RawExpr::Map(entries, _) => convert_meta_map(entries)?,
        _ => {
            return Err(StampError {
                kind: StampErrorKind::MissingMetaMap,
                span,
            });
        }
    };
    let raw_children: Vec<RawExpr> = iter.collect();
    let arity = raw_children.len();
    let mut children = Vec::with_capacity(arity);
    for (index, child) in raw_children.into_iter().enumerate() {
        let role = child_stamp_role(tag, index, arity);
        children.push(stamp_in_role(child, role, tag, index)?);
    }
    let node = Node::try_new(tag, meta, children).map_err(|e| StampError {
        kind: StampErrorKind::NodeError(e),
        span,
    })?;
    Ok(Expr::Node(Box::new(node), span))
}

/// Build a node inside one serialized type tree using the dedicated
/// type/dimension/rank grammar rather than the broader Deep child-role table.
fn build_type_node(tag: DeepTag, elements: Vec<RawExpr>, span: Span) -> Result<Expr, StampError> {
    if elements.len() < 2 {
        return Err(StampError {
            kind: StampErrorKind::MissingMetaMap,
            span,
        });
    }
    let mut iter = elements.into_iter();
    let _head = iter.next();
    let meta_raw = iter.next().expect("length checked above");
    let meta = match meta_raw {
        RawExpr::Map(entries, _) => convert_meta_map(entries)?,
        _ => {
            return Err(StampError {
                kind: StampErrorKind::MissingMetaMap,
                span,
            });
        }
    };
    let raw_children = iter.collect::<Vec<_>>();
    let arity = raw_children.len();
    let mut children = Vec::with_capacity(arity);
    for (index, child) in raw_children.into_iter().enumerate() {
        let expected = type_syntax_child_role(tag, index, arity)
            .expect("type node classification must have child roles");
        children.push(stamp_type_in_role(child, expected)?);
    }
    let node = Node::try_new(tag, meta, children).map_err(|error| StampError {
        kind: StampErrorKind::NodeError(error),
        span,
    })?;
    Ok(Expr::Node(Box::new(node), span))
}

fn convert_atom(raw: RawAtom) -> Atom {
    match raw {
        RawAtom::Symbol(s) => Atom::Name(s),
        RawAtom::Int(n) => Atom::Int(n),
        RawAtom::Float(f) => Atom::Float(f),
        RawAtom::Str(s) => Atom::Str(s),
        RawAtom::Bool(b) => Atom::Bool(b),
    }
}

fn convert_meta_map(entries: Vec<(String, RawExpr)>) -> Result<Metadata, StampError> {
    crate::annotations_codec::decode_entries(entries).map_err(|error| StampError {
        span: error.span,
        kind: StampErrorKind::NodeError(crate::node::NodeError::Metadata(error)),
    })
}

fn stamp_map(entries: Vec<(String, RawExpr)>, span: Span) -> Result<Expr, StampError> {
    Ok(Expr::Map(convert_meta_map(entries)?, span))
}

fn stamp_meta_expr(
    entries: Vec<(String, RawExpr)>,
    expr: RawExpr,
    span: Span,
) -> Result<Expr, StampError> {
    let converted_entries = convert_meta_map(entries)?;
    let converted_expr = stamp_bare(expr)?;
    Ok(Expr::MetaExpr(
        crate::ast::MetaExpr {
            metadata: converted_entries,
            expr: Box::new(converted_expr),
        },
        span,
    ))
}

fn stamp_list_as_node_or_unknown(elements: Vec<RawExpr>, span: Span) -> Result<Expr, StampError> {
    // This is an *acceptance* path, not a [03-PROG-2] rejection: an unknown
    // head is preserved verbatim in `UnknownFormData::head` so the wire AST
    // keeps the producer's spelling and a downstream consumer can name it.
    // The identity vocabulary belongs to rejections and is not used here.
    let (head, tag_opt) = match elements.first() {
        Some(RawExpr::Atom(RawAtom::Symbol(symbol), _)) => (symbol.clone(), DeepTag::parse(symbol)),
        _ => (UNKNOWN_FORM_NON_SYMBOL_HEAD.to_string(), None),
    };
    match tag_opt {
        Some(tag) => build_node(tag, elements, span),
        None => build_unknown_form(head, elements, span),
    }
}

/// The placeholder recorded for an `UnknownForm` built from a list whose head
/// is not a symbol. It is wire data on an acceptance path, not a [03-PROG-2]
/// rejection identification (chelis#1088 review residue).
const UNKNOWN_FORM_NON_SYMBOL_HEAD: &str = "<non-symbol>";

fn build_unknown_form(
    head: String,
    elements: Vec<RawExpr>,
    span: Span,
) -> Result<Expr, StampError> {
    if elements.len() < 2 {
        return Err(StampError {
            kind: StampErrorKind::MissingMetaMap,
            span,
        });
    }
    let mut iter = elements.into_iter();
    let _head_elem = iter.next(); // skip head symbol
    let meta_raw = iter.next().unwrap();
    let meta = match meta_raw {
        RawExpr::Map(entries, _) => convert_meta_map(entries)?,
        _ => {
            return Err(StampError {
                kind: StampErrorKind::MissingMetaMap,
                span,
            });
        }
    };
    let mut children = Vec::new();
    for child in iter {
        children.push(stamp_bare(child)?);
    }
    Ok(Expr::UnknownForm(Box::new(crate::ast::UnknownFormData {
        head,
        meta,
        children,
        span,
    })))
}

fn validate_metadata_raw(raw: &[RawExpr]) -> Result<(), StampError> {
    crate::metadata::validate_raw(raw).map_err(|error| StampError {
        span: error.span,
        kind: StampErrorKind::NodeError(crate::node::NodeError::Metadata(error)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sp() -> Span {
        Span::new(0, 0)
    }

    fn raw_sym(s: &str) -> RawExpr {
        RawExpr::Atom(RawAtom::Symbol(s.to_string()), sp())
    }

    fn raw_int(n: i64) -> RawExpr {
        RawExpr::Atom(RawAtom::Int(n), sp())
    }

    fn raw_map(entries: Vec<(String, RawExpr)>) -> RawExpr {
        RawExpr::Map(entries, sp())
    }

    fn empty_map() -> RawExpr {
        raw_map(vec![])
    }

    /// A well-formed def: `(def {} name body)`
    fn raw_def(name: &str, body: RawExpr) -> RawExpr {
        RawExpr::List(vec![raw_sym("def"), empty_map(), raw_sym(name), body], sp())
    }

    #[test]
    fn stamps_simple_def_with_literal_body() {
        // (def {} f (lit {} 42))
        let lit = RawExpr::List(vec![raw_sym("lit"), empty_map(), raw_int(42)], sp());
        let input = vec![raw_def("f", lit)];
        let result = stamp_to_typed(input);
        assert!(result.is_ok(), "expected Ok, got {:?}", result.err());
        let exprs = result.unwrap();
        assert_eq!(exprs.len(), 1);
        assert!(matches!(&exprs[0], Expr::Node(node, _) if node.tag() == DeepTag::Def));
    }

    #[test]
    fn rejects_bare_name_at_runtime_expr_slot() {
        // (def {} f x) — x is a bare name at RuntimeExpr position
        let input = vec![raw_def("f", raw_sym("x"))];
        let result = stamp_to_typed(input);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err.kind, StampErrorKind::NameAtExprSlot { .. }));
    }

    #[test]
    fn unknown_tag_becomes_unknown_form_at_runtime_expr() {
        // (def {} f (bogus {} 1))
        let bogus = RawExpr::List(vec![raw_sym("bogus"), empty_map(), raw_int(1)], sp());
        let input = vec![raw_def("f", bogus)];
        let result = stamp_to_typed(input);
        assert!(result.is_ok());
        let exprs = result.unwrap();
        // The def should be a Node, its body child should be UnknownForm
        if let Expr::Node(node, _) = &exprs[0] {
            assert_eq!(node.tag(), DeepTag::Def);
        } else {
            panic!("expected Node");
        }
    }

    #[test]
    fn type_slot_rejects_unknown_head() {
        // (defsig {} f (bogus {} x)) — defsig child 1 is Type
        let bogus_type = RawExpr::List(vec![raw_sym("bogus"), empty_map(), raw_sym("x")], sp());
        let input = vec![RawExpr::List(
            vec![raw_sym("defsig"), empty_map(), raw_sym("f"), bogus_type],
            sp(),
        )];
        let result = stamp_to_typed(input);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(
            err.kind,
            StampErrorKind::UndecodableTypeHead { .. }
        ));
    }

    #[test]
    fn type_slot_accepts_known_type_tag() {
        // (defsig {} f (t-prim {} f32))
        let tprim = RawExpr::List(vec![raw_sym("t-prim"), empty_map(), raw_sym("f32")], sp());
        let input = vec![RawExpr::List(
            vec![raw_sym("defsig"), empty_map(), raw_sym("f"), tprim],
            sp(),
        )];
        let result = stamp_to_typed(input);
        assert!(result.is_ok());
    }

    #[test]
    fn declaration_type_parameter_lists_are_structural_binders() {
        // (deftype {} Option (a) (variant {} None))
        let input = vec![RawExpr::List(
            vec![
                raw_sym("deftype"),
                empty_map(),
                raw_sym("Option"),
                RawExpr::List(vec![raw_sym("a")], sp()),
                RawExpr::List(vec![raw_sym("variant"), empty_map(), raw_sym("None")], sp()),
            ],
            sp(),
        )];

        let exprs = stamp_to_typed(input).expect("type parameter list must stamp structurally");
        let Expr::Node(deftype, _) = &exprs[0] else {
            panic!("expected stamped deftype Node");
        };
        assert!(
            matches!(deftype.children_slice().get(1), Some(Expr::BareList(params, _)) if
            matches!(params.as_slice(), [Expr::Atom(Atom::Name(name), _)] if name == "a"))
        );
    }

    #[test]
    fn top_level_non_declaration_is_error() {
        // (app {} (lit {} 1)) — app is not a declaration
        let input = vec![RawExpr::List(
            vec![
                raw_sym("app"),
                empty_map(),
                RawExpr::List(vec![raw_sym("lit"), empty_map(), raw_int(1)], sp()),
            ],
            sp(),
        )];
        let result = stamp_to_typed(input);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(
            err.kind,
            StampErrorKind::RequiresDeclaration { .. }
        ));
    }

    #[test]
    fn binder_slot_permits_name() {
        // (def {} my_name (lit {} 0))
        let lit = RawExpr::List(vec![raw_sym("lit"), empty_map(), raw_int(0)], sp());
        let input = vec![raw_def("my_name", lit)];
        let result = stamp_to_typed(input);
        assert!(result.is_ok());
    }

    #[test]
    fn missing_meta_map_is_error() {
        // (def x (lit {} 0)) — missing {} after tag
        let input = vec![RawExpr::List(
            vec![
                raw_sym("def"),
                raw_sym("x"),
                RawExpr::List(vec![raw_sym("lit"), empty_map(), raw_int(0)], sp()),
            ],
            sp(),
        )];
        let result = stamp_to_typed(input);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err().kind,
            StampErrorKind::MissingMetaMap
        ));
    }
}
