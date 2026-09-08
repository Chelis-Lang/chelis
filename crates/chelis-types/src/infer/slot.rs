//! The typed role-slot read (chelis#874 / chelis#887, `spec/04-type-system.md`
//! §10 [04-TOT-4], `spec/design/checker_totality.md` §PP8).
//!
//! §C4.1's coverage invariant quantifies over the nodes inference VISITS, so a
//! child slot the walk declines to enter satisfies [04-TOT-2] vacuously: the
//! error vector is empty, no `Type::Error` exists, and the program scores a
//! perfect 1.0 while a node the author submitted was discarded unread. Three
//! spellings produced it -- `unwrap_or(default)`, `if let Some(..)` with no
//! `else`, and `else { continue }` -- and each converts "I could not read this
//! child" into "there was no child".
//!
//! This module is the seam that makes those three spellings unwritable at a
//! slot the form's semantics reads:
//!
//! * The result is a `Result`, never an `Option`, so there is nothing at the
//!   call site to `unwrap_or`, fall off, or `continue` past.
//! * Absence and unreadability are different values.
//!   [`read_optional_slot`] returns `Ok(None)` for the omitted optional child
//!   that legitimately takes the form's declared default, and `Err` for a
//!   present child the form cannot read. [04-TOT-4] names exactly this
//!   distinction, and `vmap`'s axis is the instance it was written for: the
//!   zero axis is spelled as bare `vmap(f)`, so the omission must keep
//!   defaulting while the unreadable child must not.
//! * The error arm is an [`ErrorWitness`], which only
//!   `crate::errors::report_witness` can mint. A caller therefore cannot
//!   produce a value from the unreadable branch without visibly laundering a
//!   witness of a diagnostic that has already been pushed.
//! * The diagnostic is composed here from the parent [`DeepTag`] and a
//!   [`SlotShape`], so it always names the form and the expected shape and is
//!   always a `MalformedForm`. A caller may append a richer description of
//!   what it found; it cannot remove either name or change the kind.
//!
//! What the seam does NOT decide is what the caller does next. Reporting a
//! malformed slot must not stop the walk from visiting the form's other
//! children: that is how R5 (`spec/design/checker_totality.md` §PP8) produced
//! an `internal:` owner-stamp violation naming the collateral `lit` instead of
//! the `kv` key that was actually wrong. Callers here report the slot and keep
//! walking their remaining children.

use chelis_deep::ast as deep;
use chelis_deep::role::SlotShape;
use chelis_deep::{Atom, DeepTag};

use crate::errors::{CheckError, CheckErrorKind, ErrorWitness, report_witness};
use crate::session::DiagnosticSink;

use super::common::stamped_parts;

/// Read the child at `index` of a form that reads that child's content, where
/// the form declares a default for the child being absent.
///
/// `Ok(None)` is "the form omitted this child": the caller applies its declared
/// default. `Ok(Some(value))` is "present and readable". `Err(witness)` is
/// "present and unreadable", and a `MalformedForm` naming `tag` and `shape` has
/// already been pushed.
pub(super) fn read_optional_slot<'a, T>(
    kids: &'a [deep::Expr],
    tag: DeepTag,
    index: usize,
    shape: SlotShape,
    extract: impl FnOnce(&'a deep::Expr) -> Option<T>,
    errors: &mut DiagnosticSink<'_>,
) -> Result<Option<T>, ErrorWitness> {
    let Some(child) = kids.get(index) else {
        return Ok(None);
    };
    match extract(child) {
        Some(value) => Ok(Some(value)),
        None => Err(malformed_slot(tag, index, shape, Some(child), None, errors)),
    }
}

/// [`read_optional_slot`] for a slot the form requires. An absent child is
/// malformed too, because a form with no declared default has no honest reading
/// of the omission.
pub(super) fn read_required_slot<'a, T>(
    kids: &'a [deep::Expr],
    tag: DeepTag,
    index: usize,
    shape: SlotShape,
    extract: impl FnOnce(&'a deep::Expr) -> Option<T>,
    errors: &mut DiagnosticSink<'_>,
) -> Result<T, ErrorWitness> {
    let Some(child) = kids.get(index) else {
        return Err(malformed_slot(tag, index, shape, None, None, errors));
    };
    match extract(child) {
        Some(value) => Ok(value),
        None => Err(malformed_slot(tag, index, shape, Some(child), None, errors)),
    }
}

/// Push the slot-read rejection and return its witness.
///
/// The message always names the form, the expected shape, and the child index,
/// per [04-TOT-4]'s "a diagnostic naming the form and the shape it expected".
/// `detail` is the caller's optional richer description.
pub(super) fn malformed_slot(
    tag: DeepTag,
    index: usize,
    shape: SlotShape,
    child: Option<&deep::Expr>,
    detail: Option<String>,
    errors: &mut DiagnosticSink<'_>,
) -> ErrorWitness {
    let found = match child {
        Some(child) => describe_slot_child(child),
        None => "no child at that position".to_string(),
    };
    let detail = detail.map(|d| format!(" ({d})")).unwrap_or_default();
    report_witness(
        errors,
        CheckError::new(
            CheckErrorKind::MalformedForm,
            format!(
                "malformed `{}`: expected {shape} as child {index}, found {found}{detail} \
                 (spec/03-deep-syntax.md; chelis#731 [04-TOT-3], [04-TOT-4])",
                tag.as_str(),
            ),
            vec![],
        ),
    )
}

/// A short, author-facing description of the child a slot read could not use.
///
/// It names what is there rather than restating what was wanted, so the two
/// halves of the diagnostic carry different information.
pub(super) fn describe_slot_child(expr: &deep::Expr) -> String {
    match expr {
        deep::Expr::Atom(Atom::Name(name), _) => format!("symbol `{name}`"),
        deep::Expr::Atom(Atom::Tag(tag), _) => format!("the tag word `{}`", tag.as_str()),
        deep::Expr::Atom(Atom::Int(value), _) => format!("integer literal {value}"),
        deep::Expr::Atom(Atom::Float(value), _) => format!("float literal {value}"),
        deep::Expr::Atom(Atom::Bool(value), _) => format!("bool literal {value}"),
        deep::Expr::Atom(Atom::Str(_), _) => "a string literal".to_string(),
        deep::Expr::Map(..) => "a metadata map".to_string(),
        deep::Expr::MetaExpr(..) => "a metadata-annotated expression".to_string(),
        deep::Expr::BareList(items, _) if items.is_empty() => "an empty list".to_string(),
        deep::Expr::BareList(..) => "a headless list".to_string(),
        deep::Expr::UnknownForm(data) => format!("an unrecognized form `{}`", data.head),
        deep::Expr::Node(..) | deep::Expr::List(..) => match stamped_parts(expr) {
            Some((tag, _, _)) => format!("a `{}` form", tag.as_str()),
            None => "an untagged list".to_string(),
        },
    }
}
