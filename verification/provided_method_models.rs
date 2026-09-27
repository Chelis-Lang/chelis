//! Provided methods of modeled traits that nothing calls (Aeneas
//! PrePasses.resolve_modeled_defaults): Charon translates the copy of a provided method
//! for each implementation when the copy is a root (the whole-core chunks root every
//! item of a crate), but its trait declaration lists only the methods the crate calls.
//! Such a dead copy is dropped when the trait's model provides the method, so the
//! implementation inherits it: `Clone::clone_from` (the default field of the Clone
//! model, Aeneas.Std Core.lean) and `Error::source` (its provided-method model,
//! Core/Error.lean). Before, each copy was an unknown external (an axiom).
//! `ProvidedMethodModelsChecks.lean` evaluates the inherited methods through the
//! implementations' dictionaries on the inputs of the native test below.

use std::error::Error;
use std::fmt;

#[derive(Clone, Debug, PartialEq)]
pub struct Span {
    pub start: u32,
    pub end: u32,
}

#[derive(Debug)]
pub struct ParseError {
    pub span: Span,
    pub message: String,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

/// As chelis-deep `metadata::MetadataError`: an implementation that overrides nothing.
impl Error for ParseError {}

/// `Clone::clone` only: the derived implementation's `clone_from` is a dead copy.
pub fn widen(span: &Span, by: u32) -> Span {
    let mut wide = span.clone();
    wide.end += by;
    wide
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inherited_provided_methods() {
        assert_eq!(widen(&Span { start: 3, end: 5 }, 4), Span { start: 3, end: 9 });
        let mut target = Span { start: 1, end: 2 };
        target.clone_from(&Span { start: 7, end: 8 });
        assert_eq!(target, Span { start: 7, end: 8 });
        let error = ParseError { span: Span { start: 0, end: 1 }, message: "bad".to_string() };
        assert!(error.source().is_none());
    }
}
