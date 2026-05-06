//! Rule `phase-identifier-case` — Phase identifiers are
//! `phase` + digit/letter in lowercase form when embedded in filenames
//! (§9.1). The historic `phaseA_*.rs` style (camelCase chunk inside an
//! otherwise snake_case filename) is the violation; the corrected form is
//! `phase_a_*.rs`.

use crate::{Context, Rule, Surface, Violation};
use regex::Regex;
use std::sync::OnceLock;

static PHASE_RE: OnceLock<Regex> = OnceLock::new();

fn phase_re() -> &'static Regex {
    PHASE_RE.get_or_init(|| {
        // Match the bare token `phaseA`, `phaseB`, … — `phase` followed
        // immediately by an uppercase letter, with a token boundary on
        // either side. Allows the new `phase_a` form to pass.
        Regex::new(r"(^|[^a-zA-Z])phase[A-Z]").unwrap()
    })
}

pub struct PhaseIdentifierCase;

impl Rule for PhaseIdentifierCase {
    fn id(&self) -> &str {
        "phase-identifier-case"
    }

    fn spec_ref(&self) -> &str {
        "§9.1"
    }

    fn applies_to(&self) -> &[Surface] {
        // Filename-based check; we surface it on Rust source (the historic
        // `phaseA_item*.rs` violations) and on Surf source for parity. Other
        // surfaces have their own filename-shape rules and would surface
        // case violations through different rules.
        &[Surface::RustSource, Surface::SurfSource]
    }

    fn summary(&self) -> &str {
        "phase identifiers in filenames use `phase_a` (snake), not `phaseA` (camelCase chunk)"
    }

    fn check(&self, ctx: &Context<'_>) -> Vec<Violation> {
        let Some(name) = ctx.path.file_name().and_then(|n| n.to_str()) else {
            return Vec::new();
        };
        if phase_re().is_match(name) {
            vec![Violation {
                rule_id: self.id().to_string(),
                spec_ref: self.spec_ref().to_string(),
                path: ctx.path.to_path_buf(),
                line: None,
                col: None,
                message: format!(
                    "filename `{name}` uses `phaseA` form; rename to `phase_a` per §9.1"
                ),
            }]
        } else {
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn run(filename: &str) -> Vec<Violation> {
        let path = Path::new(filename);
        let ctx = Context {
            root: Path::new("/"),
            path,
            source: None,
            surface: Surface::RustSource,
        };
        PhaseIdentifierCase.check(&ctx)
    }

    #[test]
    fn flags_camel_phase_letter_in_item_filename() {
        let v = run("phaseA_item6_from_github.rs");
        assert_eq!(v.len(), 1);
        assert!(v[0].message.contains("phase_a"));
    }

    #[test]
    fn flags_camel_phase_letter_in_bundled_loader_filename() {
        let v = run("phaseA_bundled_loader.rs");
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn accepts_phase_a_form() {
        let v = run("phase_a_item6_from_github.rs");
        assert!(v.is_empty(), "phase_a is the corrected form, must not fire");
    }

    #[test]
    fn accepts_phase3j_form() {
        // Historical phase3j (digit+lowercase-letter) is the established
        // convention per §9.1 — must not fire.
        let v = run("phase3j_pre_release.md");
        assert!(v.is_empty());
    }

    #[test]
    fn does_not_flag_unrelated_caps() {
        // A capital letter that isn't immediately after `phase` shouldn't
        // fire. `myPhaseAtest` for example.
        let v = run("not_a_phase_marker.rs");
        assert!(v.is_empty());
    }

    #[test]
    fn requires_word_boundary_before_phase() {
        // `unphaseA` shouldn't fire — `phase` must start a token.
        let v = run("unphaseAfoo.rs");
        assert!(v.is_empty());
    }
}
