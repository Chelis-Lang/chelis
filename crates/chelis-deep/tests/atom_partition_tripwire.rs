//! The `Atom` partition tripwire (chelis#885, part of chelis#908).
//!
//! `spec/03-deep-syntax.md` §7.2 [03-ROLE-1] partitions what an atom can be:
//! a value literal, a structural name, or a decoded vocabulary tag. The enum
//! `chelis_deep::Atom` carries that partition, and this tripwire keeps it
//! total the same way `capacity_census_tripwire` locks `Prim`: the
//! classifier below is an EXHAUSTIVE match with no wildcard arm, and
//! `ALL_ATOMS` is a closed exemplar list paired with it. Adding an `Atom`
//! variant stops the classifier compiling until the new variant is
//! classified; removing one stops `ALL_ATOMS` compiling. The agreement test
//! then checks the classifier's verdicts against the expected partition in
//! both directions, so neither side can drift without the other noticing.
//!
//! Why this matters: chelis#885's history is a structural token slipping
//! into a value position because the old `Atom` union let `Symbol`/`Keyword`
//! sit beside the literals with no forced disposition. A future variant
//! must land with a stated class, not inherit one by wildcard.

use chelis_deep::{Atom, DeepTag};

/// The [03-ROLE-1] partition of the atom vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AtomPartition {
    /// Denotes a runtime value: `Int`, `Float`, `Str`, `Bool`.
    ValueLiteral,
    /// A structural name: never denotes a value, admissible at the
    /// structural/type/effect-handler positions §7.2 assigns.
    StructuralName,
    /// A decoded vocabulary tag: element-0 identity of a node, never a
    /// child value.
    DecodedTag,
}

/// EXHAUSTIVE classifier — no wildcard arm. A new `Atom` variant is a
/// compile error here until it is classified.
fn atom_partition(atom: &Atom) -> AtomPartition {
    match atom {
        Atom::Int(_) | Atom::Float(_) | Atom::Str(_) | Atom::Bool(_) => AtomPartition::ValueLiteral,
        Atom::Name(_) => AtomPartition::StructuralName,
        Atom::Tag(_) => AtomPartition::DecodedTag,
    }
}

/// One exemplar per `Atom` variant, paired with the exhaustive match above:
/// a variant added to the enum breaks the classifier, a variant removed
/// from it breaks this list.
fn all_atoms() -> Vec<(Atom, AtomPartition)> {
    vec![
        (Atom::Int(1), AtomPartition::ValueLiteral),
        (Atom::Float(1.5), AtomPartition::ValueLiteral),
        (Atom::Str("s".to_string()), AtomPartition::ValueLiteral),
        (Atom::Bool(true), AtomPartition::ValueLiteral),
        (Atom::Name("x".to_string()), AtomPartition::StructuralName),
        (Atom::Tag(DeepTag::App), AtomPartition::DecodedTag),
    ]
}

#[test]
fn atom_partition_agrees_with_exemplar_list_in_both_directions() {
    // Forward: every exemplar classifies as its recorded partition.
    for (atom, expected) in all_atoms() {
        assert_eq!(
            atom_partition(&atom),
            expected,
            "partition drift for {atom:?}"
        );
    }
    // Backward: every partition class has at least one exemplar, so a class
    // cannot silently become uninhabited (which would mean a variant was
    // deleted without this list noticing).
    for class in [
        AtomPartition::ValueLiteral,
        AtomPartition::StructuralName,
        AtomPartition::DecodedTag,
    ] {
        assert!(
            all_atoms().iter().any(|(_, c)| *c == class),
            "no exemplar covers {class:?}"
        );
    }
}

#[test]
fn structural_variants_are_exactly_name_and_tag() {
    // The chelis#885 partition: exactly two structural variants remain
    // (`Symbol` and `Keyword` are gone; `Keyword` is metadata-key-only
    // syntax, rejected at parse elsewhere). Four value literals carry the
    // whole expressible value-atom surface.
    let atoms = all_atoms();
    let structural = atoms
        .iter()
        .filter(|(_, c)| *c != AtomPartition::ValueLiteral)
        .count();
    let literals = atoms.len() - structural;
    assert_eq!(structural, 2, "structural atom variants: Name and Tag");
    assert_eq!(
        literals, 4,
        "value-literal atom variants: Int/Float/Str/Bool"
    );
}
