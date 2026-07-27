//! Structural adapter from canonical Deep metadata to the closed effect-kind
//! vocabulary.

use chelis_vocab::{EffectKind, EffectKindDecodeError, EffectKindInput};

use crate::{Atom, Expr, List};

/// Decode the `effect:` metadata of a canonical `handle-effect` list.
///
/// Shape errors are deliberately distinct from unknown symbols.  Every
/// semantic consumer calls this adapter before dispatch, so no stage can
/// invent a missing or malformed effect kind.
pub fn decode_effect_kind(list: &List) -> Result<EffectKind, EffectKindDecodeError<'_>> {
    let Some(Expr::Map(metadata, _)) = list.elements.get(1) else {
        return EffectKind::decode(EffectKindInput::Missing);
    };

    let mut effect_values = metadata
        .entries
        .iter()
        .filter_map(|(key, value)| (key == "effect").then_some(value));
    let Some(value) = effect_values.next() else {
        return EffectKind::decode(EffectKindInput::Missing);
    };
    if effect_values.next().is_some() {
        return EffectKind::decode(EffectKindInput::Malformed);
    }
    match value {
        Expr::Atom(Atom::Symbol(symbol), _) => EffectKind::decode(EffectKindInput::Symbol(symbol)),
        _ => EffectKind::decode(EffectKindInput::Malformed),
    }
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
        let Expr::List(list, _) = expr else {
            panic!("expected list")
        };
        list
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
        assert_eq!(
            decode_effect_kind(&list("(handle-effect {effect: 1} (lit {} 1) (lit {} 2))")),
            Err(EffectKindDecodeError::Malformed)
        );
        assert_eq!(
            decode_effect_kind(&list(
                "(handle-effect {effect: random, effect: resource} (lit {} 1) (lit {} 2))"
            )),
            Err(EffectKindDecodeError::Malformed)
        );
        assert_eq!(
            decode_effect_kind(&list(
                "(handle-effect {effect: teleport} (lit {} 1) (lit {} 2))"
            )),
            Err(EffectKindDecodeError::Unknown { symbol: "teleport" })
        );
    }
}
