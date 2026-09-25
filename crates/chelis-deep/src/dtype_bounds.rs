//! Dtype-family bounds carried as `defsig` / `def` metadata.
//!
//! `spec/04-type-system.md` §5.9 [04-DTYPE-2] lets a declaration's type
//! binder name one dtype family; `spec/03-deep-syntax.md` §1.1 and §2.2
//! carry that bound in the `dtype_bounds` metadata key rather than in a
//! child node: the bound qualifies a name in the `defsig`'s explicit binder
//! list rather than forming part of the signature type. This module owns the
//! encoding so the desugarer, the resugarer, and the type resolver cannot
//! drift on the spelling.

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

/// Decode a node's `dtype_bounds` metadata.
///
/// An absent key decodes to an empty list; that is the ordinary case for
/// every declaration with no bound.
pub fn decode_dtype_bounds(meta: &Metadata) -> Vec<(String, DtypeFamily)> {
    meta.dtype_bounds()
        .into_iter()
        .flat_map(|v| v.bounds())
        .map(|(name, family)| (name.to_string(), *family.value()))
        .collect()
}

/// Build a bound set; duplicate binder names are rejected before construction.
pub fn encode_dtype_bounds(
    bounds: &[(String, DtypeFamily)],
    span: Span,
) -> Result<DtypeBounds, crate::metadata::MetadataError> {
    DtypeBounds::try_new(
        bounds
            .iter()
            .map(|(name, family)| (name.clone(), Spanned::new(*family, span))),
        span,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Expr, annotations::MetadataValue};
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
                ("a".into(), DtypeFamily::Float),
                ("b".into(), DtypeFamily::Int),
                ("c".into(), DtypeFamily::Numeric)
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
            ("q".into(), DtypeFamily::Int),
            ("p".into(), DtypeFamily::Float),
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
                ("p".into(), DtypeFamily::Float),
                ("q".into(), DtypeFamily::Int)
            ]
        );
        assert!(
            encode_dtype_bounds(
                &[
                    ("p".into(), DtypeFamily::Float),
                    ("p".into(), DtypeFamily::Int)
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
