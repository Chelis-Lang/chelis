//! Dtype-family bounds carried as `defsig` / `def` metadata.
//!
//! `spec/04-type-system.md` §5.9 [04-DTYPE-2] lets a declaration's type
//! binder carry a dtype bound, written either as one dtype family or as an
//! explicit dtype set; `spec/03-deep-syntax.md` §1.1 and §2.2
//! carry that bound in the `dtype_bounds` metadata key rather than in a
//! child node: the bound qualifies a name in the `defsig`'s explicit binder
//! list rather than forming part of the signature type. This module owns the
//! encoding so the desugarer, the resugarer, and the type resolver cannot
//! drift on the spelling.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::Metadata;
use crate::annotations::{DtypeBounds, Spanned};
use crate::span::Span;

/// The metadata key holding a declaration's dtype-family bounds.
pub const DTYPE_BOUNDS_KEY: &str = "dtype_bounds";

/// One of the three dtype families of `spec/04-type-system.md` §5.9.
///
/// The Deep spelling is lowercase, matching the effect-name convention
/// (`(effects {} diff accum)` for Surf's `Diff`/`Accum`); the Surf
/// spelling is PascalCase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DtypeFamily {
    /// Every active float dtype of `spec/04-type-system.md` §1.1.
    Float,
    /// Every active signed integer dtype of §1.1.
    Int,
    /// The union of [`DtypeFamily::Float`] and [`DtypeFamily::Int`].
    Numeric,
}

impl DtypeFamily {
    /// Every family, in declaration order. The list is closed: a new family
    /// is a numbered-spec change to §5.9, not a local addition.
    pub const ALL: [DtypeFamily; 3] = [DtypeFamily::Float, DtypeFamily::Int, DtypeFamily::Numeric];

    /// The lowercase Deep spelling.
    pub fn deep_name(self) -> &'static str {
        match self {
            DtypeFamily::Float => "float",
            DtypeFamily::Int => "int",
            DtypeFamily::Numeric => "numeric",
        }
    }

    /// The PascalCase Surf spelling.
    pub fn surf_name(self) -> &'static str {
        match self {
            DtypeFamily::Float => "Float",
            DtypeFamily::Int => "Int",
            DtypeFamily::Numeric => "Numeric",
        }
    }

    /// Parse the lowercase Deep spelling.
    pub fn from_deep_name(name: &str) -> Option<DtypeFamily> {
        DtypeFamily::ALL
            .into_iter()
            .find(|family| family.deep_name() == name)
    }

    /// Parse the PascalCase Surf spelling.
    pub fn from_surf_name(name: &str) -> Option<DtypeFamily> {
        DtypeFamily::ALL
            .into_iter()
            .find(|family| family.surf_name() == name)
    }
}

/// One active numeric dtype of `spec/04-type-system.md` §1.1, as a member of
/// §5.9's explicit dtype set.
///
/// Declaration order is §1.1's, which `spec/03-deep-syntax.md` §6.2 makes the
/// canonical member order, so the derived `Ord` sorts a set canonically. The
/// list is closed: a new member is a numbered-spec change to §1.1.
///
/// This is deliberately a separate vocabulary from [`crate::LiteralSuffix`],
/// which spells the same eight dtypes for the *lexer* under §5.5 / §P10a /
/// §6.4.1. `bound_dtype_spellings_match_literal_suffixes` locks the spellings
/// together so neither drifts, without letting a lexer change reach a bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BoundDtype {
    F32,
    F64,
    Bf16,
    F16,
    I8,
    I16,
    I32,
    I64,
}

impl BoundDtype {
    /// Every member, in `spec/04-type-system.md` §1.1 declaration order.
    pub const ALL: [BoundDtype; 8] = [
        BoundDtype::F32,
        BoundDtype::F64,
        BoundDtype::Bf16,
        BoundDtype::F16,
        BoundDtype::I8,
        BoundDtype::I16,
        BoundDtype::I32,
        BoundDtype::I64,
    ];

    /// The lowercase spelling, identical in Deep and Surf.
    pub fn name(self) -> &'static str {
        match self {
            BoundDtype::F32 => "f32",
            BoundDtype::F64 => "f64",
            BoundDtype::Bf16 => "bf16",
            BoundDtype::F16 => "f16",
            BoundDtype::I8 => "i8",
            BoundDtype::I16 => "i16",
            BoundDtype::I32 => "i32",
            BoundDtype::I64 => "i64",
        }
    }

    /// Parse the lowercase spelling.
    pub fn from_name(name: &str) -> Option<BoundDtype> {
        BoundDtype::ALL
            .into_iter()
            .find(|dtype| dtype.name() == name)
    }

    /// Whether this member belongs to [`DtypeFamily::Float`].
    pub fn is_float(self) -> bool {
        matches!(
            self,
            BoundDtype::F32 | BoundDtype::F64 | BoundDtype::Bf16 | BoundDtype::F16
        )
    }

    /// Whether this member belongs to [`DtypeFamily::Int`].
    pub fn is_integer(self) -> bool {
        !self.is_float()
    }
}

/// A declaration's dtype bound: `spec/04-type-system.md` §5.9's family form or
/// its explicit-set form.
///
/// The two forms are not interchangeable, and §5.9 says so: a family denotes
/// whatever §1.1 admits into it, so activating a dtype widens it, while a set
/// denotes its listed members and nothing else. A one-member set is therefore
/// never equal to a family, which the derived `PartialEq` already gives.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum DtypeBound {
    /// `float`, `int` or `numeric`.
    Family(DtypeFamily),
    /// An explicit set, held in §6.2's canonical order with no duplicates.
    Set(BTreeSet<BoundDtype>),
}

impl DtypeBound {
    /// Whether `dtype` is admitted.
    ///
    /// A family defers to §1.1 membership through [`BoundDtype::is_float`] and
    /// [`BoundDtype::is_integer`] rather than an enumeration, so activating a
    /// dtype widens every family that admits it. A set tests membership.
    pub fn admits(&self, dtype: BoundDtype) -> bool {
        match self {
            DtypeBound::Family(DtypeFamily::Float) => dtype.is_float(),
            DtypeBound::Family(DtypeFamily::Int) => dtype.is_integer(),
            DtypeBound::Family(DtypeFamily::Numeric) => true,
            DtypeBound::Set(members) => members.contains(&dtype),
        }
    }

    /// The §5.9 members this bound admits, in §6.2's canonical order.
    pub fn members(&self) -> BTreeSet<BoundDtype> {
        BoundDtype::ALL
            .into_iter()
            .filter(|dtype| self.admits(*dtype))
            .collect()
    }

    /// [04-DTYPE-2]'s intersection: two families intersect as families, a
    /// family and a set as the set's members the family admits, and two sets
    /// as their common members. An intersection that no longer denotes a
    /// family is a set. `None` is an empty intersection.
    pub fn intersect(&self, other: &DtypeBound) -> Option<DtypeBound> {
        if let (DtypeBound::Family(lhs), DtypeBound::Family(rhs)) = (self, other) {
            return match (lhs, rhs) {
                _ if lhs == rhs => Some(DtypeBound::Family(*lhs)),
                (DtypeFamily::Numeric, narrower) | (narrower, DtypeFamily::Numeric) => {
                    Some(DtypeBound::Family(*narrower))
                }
                // Float and Int are disjoint.
                _ => None,
            };
        }
        let members: BTreeSet<BoundDtype> = self
            .members()
            .into_iter()
            .filter(|dtype| other.admits(*dtype))
            .collect();
        (!members.is_empty()).then_some(DtypeBound::Set(members))
    }

    /// The Surf spelling: a family name, or `{d1, d2}` in canonical order.
    pub fn surf_spelling(&self) -> String {
        match self {
            DtypeBound::Family(family) => family.surf_name().to_string(),
            DtypeBound::Set(members) => format!(
                "{{{}}}",
                members
                    .iter()
                    .map(|dtype| dtype.name())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }
}

/// Decode a node's `dtype_bounds` metadata.
///
/// An absent key decodes to an empty list; that is the ordinary case for
/// every declaration with no bound.
pub fn decode_dtype_bounds(meta: &Metadata) -> Vec<(String, DtypeBound)> {
    meta.dtype_bounds()
        .into_iter()
        .flat_map(|v| v.bounds())
        .map(|(name, bound)| (name.to_string(), bound.value().clone()))
        .collect()
}

/// Build a bound set; duplicate binder names are rejected before construction.
pub fn encode_dtype_bounds(
    bounds: &[(String, DtypeBound)],
    span: Span,
) -> Result<DtypeBounds, crate::metadata::MetadataError> {
    DtypeBounds::try_new(
        bounds
            .iter()
            .map(|(name, bound)| (name.clone(), Spanned::new(bound.clone(), span))),
        span,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Expr, annotations::MetadataValue};

    /// `BoundDtype` and `LiteralSuffix` spell the same eight dtypes for
    /// different readers (§5.9 bounds versus the §5.5 lexer). They are
    /// separate types on purpose, so this locks the spellings together
    /// rather than letting one drift. Adding a member to either stops this
    /// compiling or fails here.
    #[test]
    fn bound_dtype_spellings_match_literal_suffixes() {
        use crate::LiteralSuffix as L;
        let suffixes = [
            L::F32,
            L::F64,
            L::Bf16,
            L::F16,
            L::I8,
            L::I16,
            L::I32,
            L::I64,
        ];
        assert_eq!(BoundDtype::ALL.len(), suffixes.len());
        for (bound, suffix) in BoundDtype::ALL.into_iter().zip(suffixes) {
            assert_eq!(
                bound.name(),
                suffix.t_prim_name(),
                "bound member and literal suffix must spell the same dtype",
            );
            assert_eq!(
                bound.is_float(),
                suffix.is_float(),
                "{} float membership must agree with the lexer",
                bound.name(),
            );
        }
    }

    /// §6.2 makes §1.1 declaration order canonical, and the derived `Ord`
    /// is what delivers it, so a set built in any order prints canonically.
    #[test]
    fn a_set_prints_in_canonical_order_whatever_the_build_order() {
        let authored: BTreeSet<BoundDtype> = [BoundDtype::F64, BoundDtype::Bf16, BoundDtype::F32]
            .into_iter()
            .collect();
        assert_eq!(
            DtypeBound::Set(authored).surf_spelling(),
            "{f32, f64, bf16}",
        );
    }

    /// §5.9: a family tracks §1.1, a set does not, so a one-member set is
    /// never the family it happens to enumerate.
    #[test]
    fn a_family_never_equals_a_set() {
        let every_float: BTreeSet<BoundDtype> = BoundDtype::ALL
            .into_iter()
            .filter(|dtype| dtype.is_float())
            .collect();
        assert_ne!(
            DtypeBound::Family(DtypeFamily::Float),
            DtypeBound::Set(every_float),
            "a set enumerating Float's current members is not Float",
        );
    }

    /// [04-DTYPE-2]'s intersection table, including the empty cases.
    #[test]
    fn intersection_follows_the_dtype_2_table() {
        let set = |members: &[BoundDtype]| DtypeBound::Set(members.iter().copied().collect());
        let float = DtypeBound::Family(DtypeFamily::Float);
        let int = DtypeBound::Family(DtypeFamily::Int);
        let numeric = DtypeBound::Family(DtypeFamily::Numeric);

        // family with family stays a family
        assert_eq!(float.intersect(&numeric), Some(float.clone()));
        assert_eq!(numeric.intersect(&int), Some(int.clone()));
        assert_eq!(float.intersect(&float), Some(float.clone()));
        // Float and Int are disjoint
        assert_eq!(float.intersect(&int), None);

        // family with set keeps the members the family admits
        let wide = set(&[BoundDtype::F32, BoundDtype::F64]);
        assert_eq!(float.intersect(&wide), Some(wide.clone()));
        assert_eq!(wide.intersect(&float), Some(wide.clone()));
        assert_eq!(int.intersect(&wide), None);

        // set with set keeps common members
        let narrow = set(&[BoundDtype::F64]);
        assert_eq!(wide.intersect(&narrow), Some(narrow.clone()));
        assert_eq!(
            narrow.intersect(&set(&[BoundDtype::F32])),
            None,
            "disjoint sets intersect to empty",
        );
    }

    /// A family defers to §1.1 membership rather than an enumeration, which
    /// is the property a frozen member list would have destroyed.
    #[test]
    fn family_admission_is_a_predicate_not_an_enumeration() {
        for dtype in BoundDtype::ALL {
            assert_eq!(
                DtypeBound::Family(DtypeFamily::Float).admits(dtype),
                dtype.is_float(),
            );
            assert_eq!(
                DtypeBound::Family(DtypeFamily::Int).admits(dtype),
                dtype.is_integer(),
            );
            assert!(DtypeBound::Family(DtypeFamily::Numeric).admits(dtype));
        }
    }
    fn metadata(source: &str) -> Metadata {
        let Expr::Node(node, _) = crate::parser::parse_str(source).unwrap().remove(0) else {
            panic!("node")
        };
        node.meta().clone()
    }
    #[test]
    fn absent_and_present_bounds_decode_without_a_fallback() {
        assert!(decode_dtype_bounds(&metadata("(defsig {} f (p) (t-var {} p))")).is_empty());
        assert_eq!(
            decode_dtype_bounds(&metadata(
                "(defsig {dtype_bounds: {a: float, b: int, c: numeric}} f (a b c) (t-var {} a))"
            )),
            vec![
                ("a".into(), DtypeBound::Family(DtypeFamily::Float)),
                ("b".into(), DtypeBound::Family(DtypeFamily::Int)),
                ("c".into(), DtypeBound::Family(DtypeFamily::Numeric))
            ]
        );
        for value in [
            "float",
            "{p: signed}",
            "{p: Float}",
            "{p: \"float\"}",
            "{p: float, p: int}",
        ] {
            let source = format!("(defsig {{dtype_bounds: {value}}} f (p) (t-var {{}} p))");
            assert!(
                crate::parser::parse_str(&source)
                    .unwrap_err()
                    .to_string()
                    .contains("dtype_bounds")
            );
        }
    }
    #[test]
    fn encoding_is_binder_ordered_and_rejects_duplicates() {
        let bounds = vec![
            ("q".into(), DtypeBound::Family(DtypeFamily::Int)),
            ("p".into(), DtypeBound::Family(DtypeFamily::Float)),
        ];
        let payload = encode_dtype_bounds(&bounds, Span::new(0, 0)).unwrap();
        assert_eq!(
            payload.bounds().map(|(name, _)| name).collect::<Vec<_>>(),
            ["p", "q"]
        );
        let metadata = Metadata::from(MetadataValue::DtypeBounds(payload));
        assert_eq!(
            decode_dtype_bounds(&metadata),
            vec![
                ("p".into(), DtypeBound::Family(DtypeFamily::Float)),
                ("q".into(), DtypeBound::Family(DtypeFamily::Int))
            ]
        );
        assert!(
            encode_dtype_bounds(
                &[
                    ("p".into(), DtypeBound::Family(DtypeFamily::Float)),
                    ("p".into(), DtypeBound::Family(DtypeFamily::Int))
                ],
                Span::new(0, 0)
            )
            .is_err()
        );
    }
    #[test]
    fn surf_and_deep_spellings_are_distinct_and_total() {
        for family in DtypeFamily::ALL {
            assert_eq!(
                DtypeFamily::from_deep_name(family.deep_name()),
                Some(family)
            );
            assert_eq!(
                DtypeFamily::from_surf_name(family.surf_name()),
                Some(family)
            );
            assert_ne!(family.deep_name(), family.surf_name());
        }
        for name in ["signed", "float32", ""] {
            assert!(DtypeFamily::from_deep_name(name).is_none());
        }
    }
}
