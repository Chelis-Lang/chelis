//! Structural adapter from canonical Deep metadata to the closed effect-kind
//! vocabulary.

use chelis_vocab::{EffectKind, EffectKindDecodeError};

use crate::Metadata;

/// Decode the `effect:` metadata of a canonical `handle-effect` node.
///
/// Payload construction rules out malformed and unknown kinds. Absence is
/// still an explicit error; consumers cannot invent a default effect.
pub fn decode_effect_kind(metadata: &Metadata) -> Result<EffectKind, EffectKindDecodeError<'_>> {
    metadata
        .effect()
        .map(|v| *v.value())
        .ok_or(EffectKindDecodeError::Missing)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Expr;
    use crate::parser::parse_str;

    fn metadata(source: &str) -> Metadata {
        let expr = parse_str(source)
            .expect("parse Deep")
            .into_iter()
            .next()
            .expect("one expression");
        match expr {
            Expr::Node(node, _) => node.meta().clone(),
            other => panic!("expected Node, got {:?}", other),
        }
    }

    #[test]
    fn decodes_each_known_kind() {
        assert_eq!(
            decode_effect_kind(&metadata(
                "(handle-effect {effect: resource} (lit {} 1) (lit {} 2))"
            )),
            Ok(EffectKind::Resource)
        );
    }

    #[test]
    fn missing_malformed_duplicate_and_unknown_metadata_are_loud() {
        assert_eq!(
            decode_effect_kind(&metadata("(handle-effect {} (lit {} 1) (lit {} 2))")),
            Err(EffectKindDecodeError::Missing)
        );
        for metadata in [
            "effect: 1",
            "effect: resource, effect: resource",
            "effect: teleport",
            // The `random` handler kind was retired with the counter stream
            // (#2413); Deep naming it is rejected like any unknown kind.
            "effect: random",
        ] {
            let source = format!("(handle-effect {{{metadata}}} (lit {{}} 1) (lit {{}} 2))");
            assert!(
                parse_str(&source)
                    .unwrap_err()
                    .to_string()
                    .contains("effect")
            );
        }
    }
}
