//! Rule `surf-parses` — a `.ch` file the lint cannot parse is itself a
//! blocking violation (§12.5).
//!
//! Every rule in this crate that needs an abstract syntax tree obtains it the
//! same way, and declines the same way when it cannot:
//!
//! ```ignore
//! let Ok(decls) = chelis_surf::parser::parse_str(source) else {
//!     return Vec::new();
//! };
//! ```
//!
//! Per rule that is right. A rule must not invent a diagnosis out of a broken
//! parse, and several of them say so at the deferral. In aggregate it was
//! wrong: silence from every rule reads exactly like a clean file, so
//! `chelis lint --check` reported success over input no rule could read.
//!
//! The measured consequence (chelis#2765, found while closing chelis#2116) is
//! that a reef package whose source declared two modules got `reef build`
//! exit 1 and `chelis lint --check .` exit 0. `chelis fmt --check` does
//! reject that file, but it has no directory form — `chelis fmt --check
//! <dir>` exits with "Is a directory" — so it cannot stand in for the lint
//! when the lint is run over a tree, which is the form this repository's own
//! gate runs.
//!
//! This rule owns the parse verdict so the other rules' deferral is safe
//! rather than silent. It deliberately reports nothing else: a file that
//! parses is this rule's business finished, whatever else is wrong with it.

use crate::{Context, Rule, Severity, Surface, Violation};

pub struct SurfParses;

impl Rule for SurfParses {
    fn id(&self) -> &str {
        "surf-parses"
    }

    fn spec_ref(&self) -> &str {
        "§12.5"
    }

    fn applies_to(&self) -> &[Surface] {
        &[Surface::SurfSource]
    }

    fn summary(&self) -> &str {
        "a .ch file the lint cannot parse is a blocking violation, so no rule's deferral is mistaken for a clean file (§12.5)"
    }

    fn severity(&self) -> Severity {
        Severity::Error
    }

    fn check(&self, ctx: &Context<'_>) -> Vec<Violation> {
        let Some(source) = ctx.source else {
            return Vec::new();
        };
        let Err(error) = chelis_surf::parser::parse_str(source) else {
            return Vec::new();
        };
        vec![Violation {
            rule_id: self.id().to_string(),
            spec_ref: self.spec_ref().to_string(),
            path: ctx.path.to_path_buf(),
            // The parser reports byte offsets against a source the lint could
            // not build a tree for; rather than translate an offset whose
            // meaning depends on how far the parse got, anchor the diagnostic
            // at the file and let the carried message locate it. `chelis fmt`
            // on the same file prints the parser's own positioned error.
            line: None,
            col: None,
            message: format!("cannot be parsed as Surf: {error}"),
        }]
    }
}
