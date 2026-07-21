//! `EmittedExpr` - the C-expression payload newtype (chelis#730 Phase 2,
//! section C3 / C4.2 of `spec/design/loud_unsupported.md`).
//!
//! The C host emitter builds expression fragments as text. Before this
//! newtype, an expression builder returned a bare `String`, so ANY string
//! was a legal emission - including the audit's
//! `format!("/* unsupported builtin {other} */ 0")` stub (census row 2).
//! Phase 1 converted that terminal to `Err(Unsupported)`; Phase 2 removes
//! the *ability* to reintroduce it.
//!
//! [`EmittedExpr`] is a newtype over `String` whose only constructor,
//! [`EmittedExpr::raw`], is `pub(crate)`. Outside `chelis-backend-c` an
//! `EmittedExpr` cannot be built from an arbitrary string at all (there is
//! no public constructor, no `From<String>`, no `Default`), so an
//! expression payload can only originate inside the emitter. Inside the
//! crate, `raw` exists for the legitimate template snippets the arms of
//! `assign_builtin` build, and the response to an UNMATCHED builtin is not
//! an `EmittedExpr` at all - it is `Err(Unsupported)`. The stub is
//! unwritable, not merely unfashionable (section C3, "The `EmittedExpr`
//! rule").
//!
//! # The unsupported stub is unwritable outside the crate
//!
//! ```compile_fail
//! // `EmittedExpr::raw` is pub(crate): an out-of-crate caller cannot
//! // fabricate an emission payload from a raw stub string. This block is
//! // a `compile_fail` doctest - it is a PASS iff it does NOT compile.
//! use chelis_backend_c::emitted_expr::EmittedExpr;
//! let _stub = EmittedExpr::raw("/* unsupported builtin */ 0".to_string());
//! ```
//!
//! And there is no public constructor of any other name either:
//!
//! ```compile_fail
//! use chelis_backend_c::emitted_expr::EmittedExpr;
//! // No `From<String>`, no `new`, no `Default` - the type cannot be
//! // constructed from text outside `chelis-backend-c`.
//! let _stub: EmittedExpr = "/* unsupported builtin */ 0".to_string().into();
//! ```

use std::fmt;

/// A single C-expression payload produced by the host emitter.
///
/// Wraps the emitted C text. Constructed only inside `chelis-backend-c`
/// via [`EmittedExpr::raw`]; read back with [`EmittedExpr::as_c`] or
/// `Display`. The unsupported path never produces one - it returns
/// `Err(chelis_types::unsupported::Unsupported)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmittedExpr(String);

impl EmittedExpr {
    /// Wrap a raw C-expression template snippet.
    ///
    /// `pub(crate)` by design (section C3): the legitimate `assign_builtin`
    /// arms build their fragments through here, but no code outside the
    /// emitter can turn an arbitrary string - least of all an `unsupported`
    /// stub - into an emission. Section C4.2's lint and the Phase 0
    /// tripwire (`*/ 0"`) patrol the in-crate use sites; the type system
    /// closes the out-of-crate door.
    pub(crate) fn raw(text: String) -> Self {
        EmittedExpr(text)
    }

    /// Borrow the emitted C text for interpolation into a statement.
    pub(crate) fn as_c(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for EmittedExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_round_trips_the_text() {
        let expr = EmittedExpr::raw("a + b".to_string());
        assert_eq!(expr.as_c(), "a + b");
        assert_eq!(expr.to_string(), "a + b");
    }

    #[test]
    fn is_a_thin_newtype_over_string() {
        // The wrapper carries exactly its String and nothing else, so it is
        // a zero-overhead emission boundary, not a behavior change.
        assert_eq!(
            std::mem::size_of::<EmittedExpr>(),
            std::mem::size_of::<String>()
        );
    }
}
