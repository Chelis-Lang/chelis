//! Dtype-family bounds carried as `defsig` / `def` metadata.
//!
//! `spec/04-type-system.md` §5.9 [04-DTYPE-2] lets a declaration's type
//! binder name one dtype family; `spec/03-deep-syntax.md` §1.1 and §2.2
//! carry that bound in the `dtype_bounds` metadata key rather than in a
//! child node, because a `defsig` child would change its fixed two-child
//! shape and grow the closed 62-tag vocabulary. This module owns the
//! encoding so the desugarer, the resugarer, and the type resolver cannot
//! drift on the spelling.

use serde::{Deserialize, Serialize};

use crate::ast::{Atom, Expr, MetaMap};
use crate::span::Span;

/// The metadata key holding a declaration's dtype-family bounds.
pub const DTYPE_BOUNDS_KEY: &str = "dtype_bounds";

/// One of the three dtype families of `spec/04-type-system.md` §5.9.
///
/// The Deep spelling is lowercase, matching the effect-name convention
/// (`(effects {} diff random)` for Surf's `Diff`/`Random`); the Surf
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
    pub const ALL: [DtypeFamily; 3] = [
        DtypeFamily::Float,
        DtypeFamily::Int,
        DtypeFamily::Numeric,
    ];

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

/// Why a `dtype_bounds` metadata value is not a well-formed bound set.
///
/// Decoding is fail-closed: a malformed bound is a rejection, never an
/// ignored key that would silently drop a declared restriction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DtypeBoundsError {
    /// The `dtype_bounds` value is not a metadata map.
    NotAMap,
    /// The same binder name carries two bounds.
    DuplicateBinder(String),
    /// The value for `binder` is not a bare family symbol.
    MalformedFamily(String),
    /// The value for `binder` names no family of §5.9.
    UnknownFamily {
        /// The bounded binder name.
        binder: String,
        /// The spelling that named no family.
        spelling: String,
    },
    /// The key is present more than once in one metadata map.
    DuplicateKey,
}

impl std::fmt::Display for DtypeBoundsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DtypeBoundsError::NotAMap => write!(
                f,
                "`{DTYPE_BOUNDS_KEY}` metadata must be a map from binder name to dtype family"
            ),
            DtypeBoundsError::DuplicateBinder(binder) => {
                write!(f, "binder `{binder}` declares more than one dtype family")
            }
            DtypeBoundsError::MalformedFamily(binder) => write!(
                f,
                "the `{DTYPE_BOUNDS_KEY}` value for binder `{binder}` must be a dtype family name"
            ),
            DtypeBoundsError::UnknownFamily { binder, spelling } => write!(
                f,
                "binder `{binder}` names unknown dtype family `{spelling}`; the families are {}",
                family_list()
            ),
            DtypeBoundsError::DuplicateKey => {
                write!(f, "`{DTYPE_BOUNDS_KEY}` metadata appears more than once")
            }
        }
    }
}

/// The Deep family spellings, comma separated, for a diagnostic.
fn family_list() -> String {
    DtypeFamily::ALL
        .into_iter()
        .map(|family| format!("`{}`", family.deep_name()))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Decode a node's `dtype_bounds` metadata.
///
/// An absent key decodes to an empty list; that is the ordinary case for
/// every declaration with no bound.
pub fn decode_dtype_bounds(meta: &MetaMap) -> Result<Vec<(String, DtypeFamily)>, DtypeBoundsError> {
    let mut values = meta
        .entries
        .iter()
        .filter_map(|(key, value)| (key == DTYPE_BOUNDS_KEY).then_some(value));
    let Some(value) = values.next() else {
        return Ok(Vec::new());
    };
    if values.next().is_some() {
        return Err(DtypeBoundsError::DuplicateKey);
    }
    let Expr::Map(bounds, _) = value else {
        return Err(DtypeBoundsError::NotAMap);
    };

    let mut decoded: Vec<(String, DtypeFamily)> = Vec::with_capacity(bounds.entries.len());
    for (binder, spelling) in &bounds.entries {
        if decoded.iter().any(|(seen, _)| seen == binder) {
            return Err(DtypeBoundsError::DuplicateBinder(binder.clone()));
        }
        let Expr::Atom(Atom::Name(spelling), _) = spelling else {
            return Err(DtypeBoundsError::MalformedFamily(binder.clone()));
        };
        let family = DtypeFamily::from_deep_name(spelling).ok_or_else(|| {
            DtypeBoundsError::UnknownFamily {
                binder: binder.clone(),
                spelling: spelling.clone(),
            }
        })?;
        decoded.push((binder.clone(), family));
    }
    Ok(decoded)
}

/// Build the `dtype_bounds` metadata value for `bounds`.
///
/// Entries are emitted in binder-name order so one bound set has exactly one
/// canonical encoding.
pub fn encode_dtype_bounds(bounds: &[(String, DtypeFamily)], span: Span) -> Expr {
    let mut entries: Vec<(String, Expr)> = bounds
        .iter()
        .map(|(binder, family)| {
            (
                binder.clone(),
                Expr::Atom(Atom::Name(family.deep_name().to_string()), span),
            )
        })
        .collect();
    entries.sort_by(|(left, _), (right, _)| left.cmp(right));
    Expr::Map(MetaMap { entries }, span)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_str;

    fn meta_of(source: &str) -> MetaMap {
        let expr = parse_str(source)
            .expect("parse Deep")
            .into_iter()
            .next()
            .expect("one expression");
        match expr {
            Expr::Node(node, _) => node.meta().clone(),
            other => panic!("expected a stamped node, got {other:?}"),
        }
    }

    #[test]
    fn absent_key_decodes_to_no_bounds() {
        assert_eq!(
            decode_dtype_bounds(&meta_of("(defsig {} f (t-var {} p))")),
            Ok(Vec::new())
        );
    }

    #[test]
    fn decodes_each_family() {
        assert_eq!(
            decode_dtype_bounds(&meta_of(
                "(defsig {dtype_bounds: {a: float, b: int, c: numeric}} f (t-var {} a))"
            )),
            Ok(vec![
                ("a".to_string(), DtypeFamily::Float),
                ("b".to_string(), DtypeFamily::Int),
                ("c".to_string(), DtypeFamily::Numeric),
            ])
        );
    }

    #[test]
    fn rejects_a_non_map_value() {
        assert_eq!(
            decode_dtype_bounds(&meta_of("(defsig {dtype_bounds: float} f (t-var {} p))")),
            Err(DtypeBoundsError::NotAMap)
        );
    }

    #[test]
    fn rejects_an_unknown_family() {
        assert_eq!(
            decode_dtype_bounds(&meta_of(
                "(defsig {dtype_bounds: {p: signed}} f (t-var {} p))"
            )),
            Err(DtypeBoundsError::UnknownFamily {
                binder: "p".to_string(),
                spelling: "signed".to_string(),
            })
        );
    }

    #[test]
    fn rejects_a_pascal_case_surf_spelling() {
        // The Surf spelling is `Float`; Deep spells families lowercase, as it
        // spells effect names. Accepting both would give one bound set two
        // canonical encodings.
        assert_eq!(
            decode_dtype_bounds(&meta_of(
                "(defsig {dtype_bounds: {p: Float}} f (t-var {} p))"
            )),
            Err(DtypeBoundsError::UnknownFamily {
                binder: "p".to_string(),
                spelling: "Float".to_string(),
            })
        );
    }

    #[test]
    fn rejects_a_non_symbol_family_value() {
        assert_eq!(
            decode_dtype_bounds(&meta_of(
                "(defsig {dtype_bounds: {p: \"float\"}} f (t-var {} p))"
            )),
            Err(DtypeBoundsError::MalformedFamily("p".to_string()))
        );
    }

    #[test]
    fn encoding_is_binder_name_ordered() {
        let bounds = vec![
            ("q".to_string(), DtypeFamily::Int),
            ("p".to_string(), DtypeFamily::Float),
        ];
        let Expr::Map(map, _) = encode_dtype_bounds(&bounds, Span::new(0, 0)) else {
            panic!("expected a metadata map");
        };
        assert_eq!(
            map.entries
                .iter()
                .map(|(key, _)| key.as_str())
                .collect::<Vec<_>>(),
            vec!["p", "q"]
        );
    }

    #[test]
    fn encode_decode_round_trips() {
        let bounds = vec![
            ("p".to_string(), DtypeFamily::Float),
            ("q".to_string(), DtypeFamily::Numeric),
        ];
        let meta = MetaMap {
            entries: vec![(
                DTYPE_BOUNDS_KEY.to_string(),
                encode_dtype_bounds(&bounds, Span::new(0, 0)),
            )],
        };
        assert_eq!(decode_dtype_bounds(&meta), Ok(bounds));
    }

    #[test]
    fn surf_and_deep_spellings_are_distinct_and_total() {
        for family in DtypeFamily::ALL {
            assert_eq!(DtypeFamily::from_deep_name(family.deep_name()), Some(family));
            assert_eq!(DtypeFamily::from_surf_name(family.surf_name()), Some(family));
            assert_ne!(family.deep_name(), family.surf_name());
        }
        assert_eq!(DtypeFamily::from_surf_name("float"), None);
        assert_eq!(DtypeFamily::from_deep_name("Float"), None);
    }
}
