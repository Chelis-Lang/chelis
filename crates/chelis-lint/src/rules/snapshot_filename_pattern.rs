//! Rule `snapshot-filename-pattern` — Insta-style snapshot tests follow
//! `{context}__{section}__{test_name}.snap` with `snake_case` components
//! (§10.3). Octant's existing convention is the project standard.

use crate::{Context, Rule, Surface, Violation};
use regex::Regex;
use std::sync::OnceLock;

static SNAP_RE: OnceLock<Regex> = OnceLock::new();

fn snap_re() -> &'static Regex {
    // Three snake_case components separated by double underscores, ending in
    // `.snap`. Each component starts with [a-z], continues with [a-z0-9_].
    SNAP_RE.get_or_init(|| {
        Regex::new(r"^[a-z][a-z0-9_]*__[a-z][a-z0-9_]*__[a-z][a-z0-9_]*\.snap$").unwrap()
    })
}

pub struct SnapshotFilenamePattern;

impl Rule for SnapshotFilenamePattern {
    fn id(&self) -> &'static str {
        "snapshot-filename-pattern"
    }

    fn spec_ref(&self) -> &'static str {
        "§10.3"
    }

    fn applies_to(&self) -> &[Surface] {
        &[Surface::SnapshotFile]
    }

    fn summary(&self) -> &'static str {
        "snapshot filenames follow {context}__{section}__{test_name}.snap with snake_case components"
    }

    fn check(&self, ctx: &Context<'_>) -> Vec<Violation> {
        let Some(name) = ctx.path.file_name().and_then(|n| n.to_str()) else {
            return Vec::new();
        };
        // Insta's `.snap.new` and `.pending-snap` files are transient and
        // not part of the convention surface.
        if name.ends_with(".snap.new") || name.ends_with(".pending-snap") {
            return Vec::new();
        }
        if snap_re().is_match(name) {
            Vec::new()
        } else {
            vec![Violation {
                rule_id: self.id().to_string(),
                spec_ref: self.spec_ref().to_string(),
                path: ctx.path.to_path_buf(),
                line: None,
                col: None,
                message: format!(
                    "snapshot filename `{name}` does not match `{{context}}__{{section}}__{{test_name}}.snap` (snake_case components, double-underscore separators) per §10.3"
                ),
            }]
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
            surface: Surface::SnapshotFile,
        };
        SnapshotFilenamePattern.check(&ctx)
    }

    #[test]
    fn accepts_canonical_octant_snapshot() {
        assert!(run("cli__error_format__parse_error_format.snap").is_empty());
        assert!(run("cli__error_format__type_error_format.snap").is_empty());
        assert!(run("cli__error_format__unsupported_error_format.snap").is_empty());
    }

    #[test]
    fn accepts_other_three_part_snake_snapshots() {
        assert!(run("parser__roundtrip__simple_def.snap").is_empty());
        assert!(run("validate__deep__hello_tensor.snap").is_empty());
    }

    #[test]
    fn flags_two_part_snapshot() {
        // Missing one of the three components.
        let v = run("error_format__parse_error_format.snap");
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn flags_pascal_components() {
        let v = run("Cli__ErrorFormat__ParseError.snap");
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn flags_single_underscore_separator() {
        let v = run("cli_error_format_parse_error.snap");
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn ignores_pending_snapshots() {
        // Insta's transient files; not part of the convention surface.
        assert!(run("cli__error_format__parse_error_format.snap.new").is_empty());
        assert!(run("cli__error_format__parse_error_format.pending-snap").is_empty());
    }
}
