//! Structural adapter from canonical Deep metadata to the closed effect-kind
//! vocabulary.

use chelis_vocab::{EffectKind, EffectKindDecodeError, EffectKindInput};

use crate::{Expr, List};

/// Decode the `effect:` metadata of a canonical `handle-effect` list.
///
/// Payload construction rules out malformed and unknown kinds. Absence is
/// still an explicit error; consumers cannot invent a default effect.
pub fn decode_effect_kind(list: &List) -> Result<EffectKind, EffectKindDecodeError<'_>> {
    let Some(Expr::Map(metadata, _)) = list.elements.get(1) else {
        return EffectKind::decode(EffectKindInput::Missing);
    };

    metadata
        .effect()
        .map(|v| *v.value())
        .ok_or(EffectKindDecodeError::Missing)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_str;

    fn list(source: &str) -> List {
        let expr = parse_str(source)
            .expect("parse Deep")
            .into_iter()
            .next()
            .expect("one expression");
        match expr {
            Expr::Node(node, span) => node.to_list(span),
            Expr::List(list, _) => list,
            other => panic!("expected Node or List, got {:?}", other),
        }
    }

    #[test]
    fn decodes_each_known_kind() {
        assert_eq!(
            decode_effect_kind(&list(
                "(handle-effect {effect: random} (lit {} 1) (lit {} 2))"
            )),
            Ok(EffectKind::Random)
        );
        assert_eq!(
            decode_effect_kind(&list(
                "(handle-effect {effect: resource} (lit {} 1) (lit {} 2))"
            )),
            Ok(EffectKind::Resource)
        );
    }

    #[test]
    fn missing_malformed_duplicate_and_unknown_metadata_are_loud() {
        assert_eq!(
            decode_effect_kind(&list("(handle-effect {} (lit {} 1) (lit {} 2))")),
            Err(EffectKindDecodeError::Missing)
        );
        for metadata in [
            "effect: 1",
            "effect: random, effect: resource",
            "effect: teleport",
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
