//! Test-only constructors for adversarial checker-boundary mutations.

use chelis_deep::ast::{Atom, Expr, List, MetaExpr, Metadata};
use chelis_deep::{RawAtom, RawExpr};

/// Parse Deep syntax without running the stamped `Node` constructor.
///
/// Production parsing must reject malformed arity and role shapes before the
/// checker sees them. A few totality tests also need to prove the checker
/// fails closed when a programmatic/legacy caller bypasses that boundary.
/// This test-only converter recreates that mutation explicitly and then
/// decodes known list heads with the deprecated legacy tag stamper.
///
/// This replaces an earlier seam that parsed a valid program and then widened
/// one node through `Node`'s mutable child API. That API no longer exists:
/// the successor carrier revalidates every candidate before commit, so a
/// wrong-arity `Node` is unconstructible. The legacy `List` carrier is the
/// remaining way to hand the checker a malformed tree.
#[path = "../../../../tests/support/legacy_metadata.rs"]
mod legacy_metadata;

pub(crate) fn parse_unchecked_legacy(source: &str) -> Vec<Expr> {
    let raw = legacy_metadata::raw_metadata_fixture(source);
    let mut exprs: Vec<_> = raw.into_iter().map(raw_to_legacy).collect();
    chelis_deep::parser::stamp_tags(&mut exprs);
    exprs
}

// Private legacy-serde fixture codec. Metadata is always admitted through its
// public decoder; only ordinary List child arity/roles remain intentionally raw.
fn metadata(entries: Vec<(String, RawExpr)>) -> Metadata {
    fn wire(raw: RawExpr) -> serde_json::Value {
        use serde_json::json;
        match raw {
            RawExpr::ExtensionData(data) => json!({"ExtensionData": data}),
            RawExpr::Atom(atom, span) => {
                let atom = match atom {
                    RawAtom::Symbol(v) => Atom::Name(v),
                    RawAtom::Int(v) => Atom::Int(v),
                    RawAtom::Float(v) => Atom::Float(v),
                    RawAtom::Str(v) => Atom::Str(v),
                    RawAtom::Bool(v) => Atom::Bool(v),
                };
                json!({"Atom": [atom, span]})
            }
            RawExpr::List(elements, span) => {
                let tag = match elements.first() {
                    Some(RawExpr::Atom(RawAtom::Symbol(name), _)) => {
                        chelis_deep::DeepTag::parse(name)
                    }
                    _ => None,
                };
                let mut elements: Vec<_> = elements.into_iter().map(wire).collect();
                if let Some(tag) = tag {
                    elements[0]["Atom"][0] = json!(Atom::Tag(tag));
                }
                json!({"List": [{"elements": elements}, span]})
            }
            RawExpr::Map(entries, span) => {
                json!({"Map": [{"entries": entries.into_iter().map(|(key, value)| (key, wire(value))).collect::<Vec<_>>()}, span]})
            }
            RawExpr::MetaExpr {
                entries,
                expr,
                span,
            } => {
                json!({"MetaExpr": [{"entries": entries.into_iter().map(|(key, value)| (key, wire(value))).collect::<Vec<_>>(), "expr": wire(*expr)}, span]})
            }
        }
    }
    let encoded = serde_json::json!({"entries": entries.into_iter().map(|(key, value)| {
        let encoded = if chelis_deep::metadata::REGISTERED_METADATA_KEYS.contains(&key.as_str()) { wire(value) } else { serde_json::json!({"ExtensionData": chelis_deep::ExtensionData::from_raw(&value).unwrap()}) };
        (key, encoded)
    }).collect::<Vec<_>>()});
    serde_json::from_value(encoded).expect("legacy role fixture must have valid metadata payloads")
}

fn raw_to_legacy(raw: RawExpr) -> Expr {
    match raw {
        RawExpr::ExtensionData(_) => panic!("opaque data cannot be a legacy program expression"),
        RawExpr::Atom(atom, span) => Expr::Atom(
            match atom {
                RawAtom::Symbol(value) => Atom::Name(value),
                RawAtom::Int(value) => Atom::Int(value),
                RawAtom::Float(value) => Atom::Float(value),
                RawAtom::Str(value) => Atom::Str(value),
                RawAtom::Bool(value) => Atom::Bool(value),
            },
            span,
        ),
        RawExpr::List(elements, span) => Expr::List(
            List {
                elements: elements.into_iter().map(raw_to_legacy).collect(),
            },
            span,
        ),
        RawExpr::Map(entries, span) => Expr::Map(metadata(entries), span),
        RawExpr::MetaExpr {
            entries,
            expr,
            span,
        } => Expr::MetaExpr(
            MetaExpr {
                metadata: metadata(entries),
                expr: Box::new(raw_to_legacy(*expr)),
            },
            span,
        ),
    }
}
