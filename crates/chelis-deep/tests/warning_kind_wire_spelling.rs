//! chelis#886: the validation-warning wire spellings are a published
//! identity, restated here independently of the projection that emits them.
//!
//! `check_snippet` spelled these with `{:?}`, which made the wire a function
//! of the Rust variant name: a rename moved it with nothing objecting. That is
//! the hazard chelis#886 names. This test is the objection -- it fails by name
//! on a rename, and it is written as a literal table rather than derived from
//! the enum.
//!
//! One measured fact, and no inference from it: a naive `sed` across the whole
//! of `crates/chelis-deep` rewrites this file's literals too, so the wire moves
//! and both tests still pass. This file living inside the crate is why.
//!
//! Two earlier revisions of this comment described what the pin "catches" and
//! both descriptions were wrong -- the first claimed a crate-wide `sed` was
//! caught, the second claimed an IDE rename was. An IDE rename does not move
//! the wire at all, because `wire_name`'s table is what holds the spelling, so
//! there is nothing for a test to catch. Rather than write a third
//! description, this comment states the one thing that was executed.
//!
//! Moving this file out of the crate would make the `sed` case genuinely
//! caught. That is a different change from the one this commit makes.

use chelis_deep::validate::WarningKind;

#[test]
fn every_warning_kind_projects_to_its_pinned_spelling() {
    let pinned: [(WarningKind, &str); 4] = [
        (WarningKind::UnknownTag, "UnknownTag"),
        (WarningKind::MissingMetadata, "MissingMetadata"),
        (WarningKind::Structural, "Structural"),
        (WarningKind::Arity, "Arity"),
    ];
    for (kind, expected) in pinned {
        assert_eq!(
            kind.wire_name(),
            expected,
            "the wire spelling of {kind:?} is published; changing it is a wire \
             change and needs a deliberate migration, not a rename"
        );
    }
}

/// The projection must not collapse two kinds onto one identity, which would
/// make them indistinguishable to a consumer matching on `kind`.
///
/// Scope: this is injectivity over the four spellings named above, not over
/// the variant set. `wire_name`'s exhaustive `match` forces a NEW variant to
/// be given some spelling -- that part is compile-enforced -- but nothing here
/// forces that spelling to be unique or to be added to the table, so a fifth
/// variant colliding with an existing spelling passes. Verified.
#[test]
fn no_two_warning_kinds_share_a_spelling() {
    let all = [
        WarningKind::UnknownTag,
        WarningKind::MissingMetadata,
        WarningKind::Structural,
        WarningKind::Arity,
    ];
    let mut seen: Vec<&str> = Vec::new();
    for kind in all {
        let name = kind.wire_name();
        assert!(
            !seen.contains(&name),
            "two warning kinds both spell themselves `{name}`"
        );
        seen.push(name);
    }
}
