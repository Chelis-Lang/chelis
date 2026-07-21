//! The closed set of `handle-effect` effect kinds (chelis#730 Phase 2,
//! section C4.4 of `spec/design/loud_unsupported.md`).
//!
//! Before this enum, every stage that dispatched on the `effect:` metadata
//! symbol string-matched `"random"` / `"resource"` and fell through to a
//! catch-all for anything else. In lowering that catch-all silently
//! dropped the handler and lowered the body (`effect: teleport` built and
//! ran - census rows 9/20); in the checker it was a loud `MalformedForm`
//! but still a string-match with no compile-time coupling to the lowering
//! set.
//!
//! [`EffectKind`] is the ratchet: a CLOSED enum parsed once at each
//! dispatch site. Adding a kind forces a compile-error work-list at every
//! `match` over it (the section C4 ratchet #1), and every site's unknown
//! case now routes through the closed-set `from_symbol` returning `None`,
//! which the caller MUST answer with a branded diagnostic
//! (`Unsupported`/`MalformedForm`), never a silent default.

/// The effect kinds the desugarer produces and every stage recognizes.
///
/// The two known kinds are the ones `spec/03-deep-syntax.md` gives
/// `handle-effect`: `random` (the `with seed(...)` RNG scope) and
/// `resource` (the `with device(...)` passthrough scope). This set is
/// closed by construction; a new kind is a spec change that must add a
/// variant here, which then fails the build at every un-updated dispatch
/// site.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectKind {
    /// `with seed(...)` - a seeded RNG scope (`Effect::Random`).
    Random,
    /// `with device(...)` - a resource/device scope; pure passthrough from
    /// the value stages' perspective.
    Resource,
}

impl EffectKind {
    /// Parse the `effect:` metadata symbol into the closed set.
    ///
    /// Returns `None` for any symbol outside the set. The caller MUST
    /// respond to `None` with a branded section C2 diagnostic - an
    /// `Unsupported` in the value stages, a `MalformedForm` in the checker -
    /// never a silent default. That is the whole point of the ratchet: the
    /// `_ =>`-into-a-substituted-value arm (census rows 9/20) is gone, and
    /// the only way past an unknown kind is a loud rejection.
    pub fn from_symbol(name: &str) -> Option<Self> {
        match name {
            "random" => Some(EffectKind::Random),
            "resource" => Some(EffectKind::Resource),
            _ => None,
        }
    }

    /// The canonical `effect:` symbol for this kind - the inverse of
    /// [`EffectKind::from_symbol`] on the closed set.
    pub fn as_symbol(self) -> &'static str {
        match self {
            EffectKind::Random => "random",
            EffectKind::Resource => "resource",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_two_known_kinds() {
        assert_eq!(EffectKind::from_symbol("random"), Some(EffectKind::Random));
        assert_eq!(
            EffectKind::from_symbol("resource"),
            Some(EffectKind::Resource)
        );
    }

    #[test]
    fn unknown_kinds_are_none_not_a_default() {
        // The negative parity: an unrecognized kind is `None`, so every
        // consumer is forced to reject rather than silently pick a kind.
        assert_eq!(EffectKind::from_symbol("teleport"), None);
        assert_eq!(EffectKind::from_symbol(""), None);
        assert_eq!(EffectKind::from_symbol("Random"), None);
        assert_eq!(EffectKind::from_symbol("seed"), None);
    }

    #[test]
    fn symbol_round_trips() {
        for kind in [EffectKind::Random, EffectKind::Resource] {
            assert_eq!(EffectKind::from_symbol(kind.as_symbol()), Some(kind));
        }
    }
}
