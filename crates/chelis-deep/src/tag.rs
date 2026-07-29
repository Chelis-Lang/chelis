//! The closed Deep tag vocabulary as a typed enum (chelis#731 Phase 3).
//!
//! `DeepTag` is the in-memory, typed form of the 62-tag closed vocabulary
//! normatively owned by `spec/03-deep-syntax.md` §2.
//!
//! The SERIALIZED Deep form stays frozen: a node's tag is still written as
//! its leading symbol string. The IN-MEMORY form is decode-once
//! (`spec/design/checker_totality.md` §C4 item 2, executing [#730]'s
//! §C4.2 doctrine that raw strings exist only at serialization
//! boundaries). The parser stamps the tag as `Atom::Tag(DeepTag)` at
//! element 0, so after parsing the tag string does not exist in the tree
//! and a consumer cannot dispatch on it. [`List::tag`] is the only
//! dispatch accessor; printers and serializers regenerate the string
//! through [`DeepTag::as_str`] at the boundary only.
//!
//! [#730]: https://github.com/Chelis-Lang/chelis/issues/730
//!
//! The point of the enum is compile-time totality
//! (`spec/design/checker_totality.md` §C4): each chokepoint matches it
//! exhaustively with no `_` arm, so adding a 63rd variant turns every
//! consumer that has not chosen a disposition into a compile error. The
//! variant set is frozen to the vocabulary and may change only together
//! with a `spec/03-deep-syntax.md` revision and every consumer, in one
//! change set (checker_totality.md B1).
//!
//! [`DeepTag::parse`] is the decode side of that boundary: the parser's
//! stamping pass and the `find_raw_vocabulary_tag` invariant use it, and
//! raw-string entry points call it directly. It is NOT the dispatch path -
//! decode-once means dispatch reads the already-stamped [`List::tag`].
//! `parse` returning `None` is the raw-string entry verdict: the string is
//! outside the closed vocabulary and the caller owns the loud rejection
//! (§C1.2). Compiler-internal pre-expansion tags such as `defmacro` and
//! `macro-invoke` are deliberately outside this vocabulary (spec/03
//! §1.1.2 macro boundary rule) and do not parse.
//!
//! [`List::tag`]: crate::ast::List::tag

/// One tag of the closed Deep vocabulary (`spec/03-deep-syntax.md` §2.10).
///
/// Serde note: the enum appears inside `Atom::Tag`, which the typecheck
/// cache serializes as part of `Expr`; the cache envelope's format/build
/// identity check invalidates old entries across representation changes,
/// so no cross-version decode path exists or is wanted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum DeepTag {
    // Module structure (spec/03 §2.1).
    Module,
    Import,
    ImportAll,
    Export,
    // Declarations (§2.2).
    Def,
    Defsig,
    Deftype,
    Typealias,
    Variant,
    Field,
    Defdim,
    // Expressions (§2.3; §2.10 counts `borrow` here).
    Fn,
    App,
    Let,
    Match,
    Arm,
    If,
    Var,
    Lit,
    Record,
    Access,
    Pipe,
    Block,
    Tuple,
    TupleGet,
    RecordUpdate,
    Par,
    HandleEffect,
    Borrow,
    // Patterns (§2.4).
    PatVar,
    PatLit,
    PatCtor,
    PatTuple,
    PatRecord,
    PatWild,
    PatAs,
    // Type expressions (§2.5).
    TPrim,
    TFn,
    TTensor,
    TRef,
    TAdt,
    TVar,
    TUnit,
    TTuple,
    // Dimension expressions (§2.6).
    DName,
    DVar,
    DLit,
    DRank,
    // Transforms (§2.7).
    Grad,
    Vmap,
    Jit,
    Realize,
    Cast,
    Copy,
    // Metaprogramming (§2.8).
    Quote,
    Unquote,
    Splice,
    // Helpers (§2.9).
    Params,
    Bind,
    Kv,
    Effects,
    Resource,
}

impl DeepTag {
    /// The vocabulary size (`spec/03-deep-syntax.md` §2.10's total row).
    pub const COUNT: usize = 62;

    /// Every tag once, in §2.10 category order.
    pub const ALL: [Self; Self::COUNT] = [
        Self::Module,
        Self::Import,
        Self::ImportAll,
        Self::Export,
        Self::Def,
        Self::Defsig,
        Self::Deftype,
        Self::Typealias,
        Self::Variant,
        Self::Field,
        Self::Defdim,
        Self::Fn,
        Self::App,
        Self::Let,
        Self::Match,
        Self::Arm,
        Self::If,
        Self::Var,
        Self::Lit,
        Self::Record,
        Self::Access,
        Self::Pipe,
        Self::Block,
        Self::Tuple,
        Self::TupleGet,
        Self::RecordUpdate,
        Self::Par,
        Self::HandleEffect,
        Self::Borrow,
        Self::PatVar,
        Self::PatLit,
        Self::PatCtor,
        Self::PatTuple,
        Self::PatRecord,
        Self::PatWild,
        Self::PatAs,
        Self::TPrim,
        Self::TFn,
        Self::TTensor,
        Self::TRef,
        Self::TAdt,
        Self::TVar,
        Self::TUnit,
        Self::TTuple,
        Self::DName,
        Self::DVar,
        Self::DLit,
        Self::DRank,
        Self::Grad,
        Self::Vmap,
        Self::Jit,
        Self::Realize,
        Self::Cast,
        Self::Copy,
        Self::Quote,
        Self::Unquote,
        Self::Splice,
        Self::Params,
        Self::Bind,
        Self::Kv,
        Self::Effects,
        Self::Resource,
    ];

    /// The canonical serialized spellings, index-aligned with [`Self::ALL`].
    pub const ALL_STRS: [&'static str; Self::COUNT] = {
        let mut strs = [""; Self::COUNT];
        let mut i = 0;
        while i < Self::COUNT {
            strs[i] = Self::ALL[i].as_str();
            i += 1;
        }
        strs
    };

    /// The canonical serialized spelling (`spec/03-deep-syntax.md` §2).
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Module => "module",
            Self::Import => "import",
            Self::ImportAll => "import-all",
            Self::Export => "export",
            Self::Def => "def",
            Self::Defsig => "defsig",
            Self::Deftype => "deftype",
            Self::Typealias => "typealias",
            Self::Variant => "variant",
            Self::Field => "field",
            Self::Defdim => "defdim",
            Self::Fn => "fn",
            Self::App => "app",
            Self::Let => "let",
            Self::Match => "match",
            Self::Arm => "arm",
            Self::If => "if",
            Self::Var => "var",
            Self::Lit => "lit",
            Self::Record => "record",
            Self::Access => "access",
            Self::Pipe => "pipe",
            Self::Block => "block",
            Self::Tuple => "tuple",
            Self::TupleGet => "tuple-get",
            Self::RecordUpdate => "record-update",
            Self::Par => "par",
            Self::HandleEffect => "handle-effect",
            Self::Borrow => "borrow",
            Self::PatVar => "pat-var",
            Self::PatLit => "pat-lit",
            Self::PatCtor => "pat-ctor",
            Self::PatTuple => "pat-tuple",
            Self::PatRecord => "pat-record",
            Self::PatWild => "pat-wild",
            Self::PatAs => "pat-as",
            Self::TPrim => "t-prim",
            Self::TFn => "t-fn",
            Self::TTensor => "t-tensor",
            Self::TRef => "t-ref",
            Self::TAdt => "t-adt",
            Self::TVar => "t-var",
            Self::TUnit => "t-unit",
            Self::TTuple => "t-tuple",
            Self::DName => "d-name",
            Self::DVar => "d-var",
            Self::DLit => "d-lit",
            Self::DRank => "d-rank",
            Self::Grad => "grad",
            Self::Vmap => "vmap",
            Self::Jit => "jit",
            Self::Realize => "realize",
            Self::Cast => "cast",
            Self::Copy => "copy",
            Self::Quote => "quote",
            Self::Unquote => "unquote",
            Self::Splice => "splice",
            Self::Params => "params",
            Self::Bind => "bind",
            Self::Kv => "kv",
            Self::Effects => "effects",
            Self::Resource => "resource",
        }
    }

    /// Decode a serialized tag string. `None` means the string is outside
    /// the closed vocabulary; the caller owns the loud rejection
    /// (checker_totality.md §C1.2 raw-string boundary).
    pub fn parse(tag: &str) -> Option<Self> {
        Some(match tag {
            "module" => Self::Module,
            "import" => Self::Import,
            "import-all" => Self::ImportAll,
            "export" => Self::Export,
            "def" => Self::Def,
            "defsig" => Self::Defsig,
            "deftype" => Self::Deftype,
            "typealias" => Self::Typealias,
            "variant" => Self::Variant,
            "field" => Self::Field,
            "defdim" => Self::Defdim,
            "fn" => Self::Fn,
            "app" => Self::App,
            "let" => Self::Let,
            "match" => Self::Match,
            "arm" => Self::Arm,
            "if" => Self::If,
            "var" => Self::Var,
            "lit" => Self::Lit,
            "record" => Self::Record,
            "access" => Self::Access,
            "pipe" => Self::Pipe,
            "block" => Self::Block,
            "tuple" => Self::Tuple,
            "tuple-get" => Self::TupleGet,
            "record-update" => Self::RecordUpdate,
            "par" => Self::Par,
            "handle-effect" => Self::HandleEffect,
            "borrow" => Self::Borrow,
            "pat-var" => Self::PatVar,
            "pat-lit" => Self::PatLit,
            "pat-ctor" => Self::PatCtor,
            "pat-tuple" => Self::PatTuple,
            "pat-record" => Self::PatRecord,
            "pat-wild" => Self::PatWild,
            "pat-as" => Self::PatAs,
            "t-prim" => Self::TPrim,
            "t-fn" => Self::TFn,
            "t-tensor" => Self::TTensor,
            "t-ref" => Self::TRef,
            "t-adt" => Self::TAdt,
            "t-var" => Self::TVar,
            "t-unit" => Self::TUnit,
            "t-tuple" => Self::TTuple,
            "d-name" => Self::DName,
            "d-var" => Self::DVar,
            "d-lit" => Self::DLit,
            "d-rank" => Self::DRank,
            "grad" => Self::Grad,
            "vmap" => Self::Vmap,
            "jit" => Self::Jit,
            "realize" => Self::Realize,
            "cast" => Self::Cast,
            "copy" => Self::Copy,
            "quote" => Self::Quote,
            "unquote" => Self::Unquote,
            "splice" => Self::Splice,
            "params" => Self::Params,
            "bind" => Self::Bind,
            "kv" => Self::Kv,
            "effects" => Self::Effects,
            "resource" => Self::Resource,
            _ => return None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// The §2.10 tag-count-summary rows, spelled out independently of the
    /// enum so the enum cannot drift from spec/03 without this failing
    /// (the same job the retired duplicated `VALID_TAGS` copies did).
    const SPEC_03_SECTION_2_10: [&str; 62] = [
        "module",
        "import",
        "import-all",
        "export",
        "def",
        "defsig",
        "deftype",
        "typealias",
        "variant",
        "field",
        "defdim",
        "fn",
        "app",
        "let",
        "match",
        "arm",
        "if",
        "var",
        "lit",
        "record",
        "access",
        "pipe",
        "block",
        "tuple",
        "tuple-get",
        "record-update",
        "par",
        "handle-effect",
        "borrow",
        "pat-var",
        "pat-lit",
        "pat-ctor",
        "pat-tuple",
        "pat-record",
        "pat-wild",
        "pat-as",
        "t-prim",
        "t-fn",
        "t-tensor",
        "t-ref",
        "t-adt",
        "t-var",
        "t-unit",
        "t-tuple",
        "d-name",
        "d-var",
        "d-lit",
        "d-rank",
        "grad",
        "vmap",
        "jit",
        "realize",
        "cast",
        "copy",
        "quote",
        "unquote",
        "splice",
        "params",
        "bind",
        "kv",
        "effects",
        "resource",
    ];

    #[test]
    fn vocabulary_matches_spec_03_section_2_10_exactly() {
        let expected: HashSet<&str> = SPEC_03_SECTION_2_10.into_iter().collect();
        let actual: HashSet<&str> = DeepTag::ALL_STRS.into_iter().collect();
        let missing: Vec<&&str> = expected.difference(&actual).collect();
        let extra: Vec<&&str> = actual.difference(&expected).collect();
        assert!(
            missing.is_empty() && extra.is_empty(),
            "DeepTag drifted from spec/03 §2.10.\n  missing (in spec, not in \
             enum): {missing:?}\n  extra (in enum, not in spec): {extra:?}"
        );
    }

    #[test]
    fn vocabulary_count_and_uniqueness() {
        assert_eq!(DeepTag::COUNT, 62);
        let unique_tags: HashSet<DeepTag> = DeepTag::ALL.into_iter().collect();
        assert_eq!(unique_tags.len(), 62, "ALL must list every variant once");
        let unique_strs: HashSet<&str> = DeepTag::ALL_STRS.into_iter().collect();
        assert_eq!(unique_strs.len(), 62, "spellings must be distinct");
    }

    #[test]
    fn every_tag_roundtrips_through_parse_and_as_str() {
        for tag in DeepTag::ALL {
            assert_eq!(
                DeepTag::parse(tag.as_str()),
                Some(tag),
                "`{}` must round-trip",
                tag.as_str()
            );
        }
    }

    #[test]
    fn parse_rejects_strings_outside_the_closed_vocabulary() {
        // Negative parity: near-misses, internal pre-expansion tags
        // (spec/03 macro boundary rule), case variants, and whitespace.
        for bogus in [
            "",
            "bogus",
            "apply",
            "sig",
            "defmacro",
            "macro-invoke",
            "expand",
            "Var",
            "VAR",
            "t_prim",
            "tprim",
            " var",
            "var ",
            "import_all",
        ] {
            assert_eq!(
                DeepTag::parse(bogus),
                None,
                "`{bogus}` must not decode as a Deep tag"
            );
        }
    }

    #[test]
    fn compound_spellings_are_exact() {
        assert_eq!(DeepTag::ImportAll.as_str(), "import-all");
        assert_eq!(DeepTag::TupleGet.as_str(), "tuple-get");
        assert_eq!(DeepTag::RecordUpdate.as_str(), "record-update");
        assert_eq!(DeepTag::HandleEffect.as_str(), "handle-effect");
        assert_eq!(DeepTag::TPrim.as_str(), "t-prim");
        assert_eq!(DeepTag::DRank.as_str(), "d-rank");
        assert_eq!(DeepTag::PatAs.as_str(), "pat-as");
    }
}
