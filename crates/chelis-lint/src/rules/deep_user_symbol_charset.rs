//! Rule `deep-user-symbol-charset` — user-defined Deep symbols (anything
//! outside the closed tag vocabulary) must satisfy the Surf identifier
//! charset `[A-Za-z_][A-Za-z0-9_]*` (§1.5 / §11.1).
//!
//! Background: Deep's `Symbol` lexer accepts hyphens to support the
//! closed compound-tag vocabulary (`t-fn`, `pat-ctor`, `d-name`, etc.).
//! That's intentional structure (§11.1 resolved decision). User-defined
//! symbols (variable names, function names, ADT constructor names)
//! originate from Surf desugaring and should inherit Surf's no-hyphen
//! rule by construction. This rule is the guard: if a future Deep
//! emitter accidentally produced a hyphenated user symbol, the lint
//! would catch it without rejecting the legitimate compound-tag use.

use crate::{Context, Rule, Surface, Violation};
use regex::Regex;
use std::collections::HashSet;
use std::sync::OnceLock;

static SYMBOL_RE: OnceLock<Regex> = OnceLock::new();

fn symbol_re() -> &'static Regex {
    // Match Deep `Symbol` per crates/chelis-deep/src/lexer.rs:179-181:
    // `[A-Za-z_][A-Za-z0-9_-]*`. The hyphen is what we're auditing.
    SYMBOL_RE.get_or_init(|| Regex::new(r"[A-Za-z_][A-Za-z0-9_-]*").unwrap())
}

/// Closed Deep tag vocabulary. Derived from
/// `crates/chelis-deep/src/validate.rs`. Hyphens here are intentional
/// per §1.4. Any hyphenated symbol that's NOT in this list is a
/// user-defined symbol and a §1.5 / §11.1 violation.
const CLOSED_TAGS: &[&str] = &[
    "module",
    "import",
    "import-all",
    "export",
    "def",
    "defsig",
    "deftype",
    "typealias",
    "variant",
    "field",
    "defdim",
    "fn",
    "app",
    "let",
    "match",
    "arm",
    "if",
    "var",
    "lit",
    "record",
    "access",
    "pipe",
    "block",
    "tuple",
    "tuple-get",
    "par",
    "borrow",
    "pat-var",
    "pat-lit",
    "pat-ctor",
    "pat-tuple",
    "pat-record",
    "pat-wild",
    "pat-as",
    "record-update",
    "t-prim",
    "t-fn",
    "t-tensor",
    "t-ref",
    "t-adt",
    "t-var",
    "t-unit",
    "t-tuple",
    "d-name",
    "d-var",
    "d-lit",
    "grad",
    "vmap",
    "jit",
    "realize",
    "cast",
    "copy",
    "handle-effect",
    "quote",
    "unquote",
    "splice",
    "params",
    "bind",
    "kv",
    "effects",
    "resource",
];

pub struct DeepUserSymbolCharset;

impl Rule for DeepUserSymbolCharset {
    fn id(&self) -> &str {
        "deep-user-symbol-charset"
    }

    fn spec_ref(&self) -> &str {
        "§11.1"
    }

    fn applies_to(&self) -> &[Surface] {
        &[Surface::DeepSource]
    }

    fn summary(&self) -> &str {
        "user-defined Deep symbols must satisfy Surf's identifier charset (no hyphens); only closed-vocabulary tags may contain hyphens"
    }

    fn check(&self, ctx: &Context<'_>) -> Vec<Violation> {
        let Some(source) = ctx.source else {
            return Vec::new();
        };
        let closed: HashSet<&str> = CLOSED_TAGS.iter().copied().collect();
        let mut out = Vec::new();
        // Deep is line-oriented enough for our purposes; tokenize per-line
        // to get accurate line numbers.
        for (line_idx, line) in source.lines().enumerate() {
            // Skip text inside string literals — Deep allows arbitrary
            // characters in `"..."` and our naive scan would otherwise
            // false-positive on hyphens in string content.
            let stripped = strip_string_literals(line);
            for cap in symbol_re().find_iter(&stripped) {
                let token = cap.as_str();
                if !token.contains('-') {
                    continue;
                }
                if closed.contains(token) {
                    continue;
                }
                out.push(Violation {
                    rule_id: self.id().to_string(),
                    spec_ref: self.spec_ref().to_string(),
                    path: ctx.path.to_path_buf(),
                    line: Some(line_idx + 1),
                    col: Some(cap.start() + 1),
                    message: format!(
                        "Deep symbol `{token}` contains a hyphen but is not in the closed tag vocabulary; user-defined symbols must satisfy the Surf identifier charset `[A-Za-z_][A-Za-z0-9_]*` per §11.1 / §1.5"
                    ),
                });
            }
        }
        out
    }
}

/// Replace every `"..."` string literal and every `;` line comment in
/// `line` with spaces of equal length so the symbol scanner doesn't see
/// hyphens inside string content or human-readable prose.
/// Backslash-escapes inside the string are honored so embedded `"`
/// doesn't terminate prematurely. `;` outside a string starts a line
/// comment that runs to end of line.
fn strip_string_literals(line: &str) -> String {
    let bytes = line.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut in_string = false;
    let mut in_comment = false;
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if in_comment {
            // Comments run to end of line; this entire line is dropped
            // from symbol scanning.
            out.push(b' ');
            i += 1;
            continue;
        }
        if in_string {
            if b == b'\\' && i + 1 < bytes.len() {
                out.push(b' ');
                out.push(b' ');
                i += 2;
                continue;
            }
            if b == b'"' {
                in_string = false;
                out.push(b' ');
                i += 1;
                continue;
            }
            out.push(b' ');
            i += 1;
        } else {
            if b == b'"' {
                in_string = true;
                out.push(b' ');
                i += 1;
                continue;
            }
            if b == b';' {
                in_comment = true;
                out.push(b' ');
                i += 1;
                continue;
            }
            out.push(b);
            i += 1;
        }
    }
    String::from_utf8(out).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn run(src: &str) -> Vec<Violation> {
        let path = Path::new("test.dp");
        let ctx = Context {
            root: Path::new("/"),
            path,
            source: Some(src),
            surface: Surface::DeepSource,
        };
        DeepUserSymbolCharset.check(&ctx)
    }

    #[test]
    fn accepts_canonical_compound_tags() {
        // The closed compound-tag vocabulary is allowed.
        let src = r#"(def {type: (t-fn {} (t-prim {} f32) (t-prim {} f32))}
  square (params {} x)
  (app {} (var {} mul) (var {} x) (var {} x)))
"#;
        let v = run(src);
        assert!(v.is_empty(), "got: {v:?}");
    }

    #[test]
    fn accepts_pattern_tags() {
        let src = r#"(match {} (var {} m)
  (arm {} (pat-ctor {} Some (pat-var {} x)) () (var {} x))
  (arm {} (pat-ctor {} None) () (var {} default)))
"#;
        let v = run(src);
        assert!(v.is_empty(), "got: {v:?}");
    }

    #[test]
    fn flags_user_defined_hyphenated_symbol() {
        // A user-defined function name with a hyphen. Would result from
        // a buggy Deep emitter, since Surf can't produce one.
        let src = "(def {} my-func (params {} x) (var {} x))\n";
        let v = run(src);
        assert_eq!(v.len(), 1);
        assert!(v[0].message.contains("my-func"));
        assert_eq!(v[0].line, Some(1));
    }

    #[test]
    fn flags_hyphenated_var_reference() {
        // A user-defined symbol referenced via (var {} my-var).
        let src = "(app {} (var {} func) (var {} my-var))\n";
        let v = run(src);
        assert_eq!(v.len(), 1);
        assert!(v[0].message.contains("my-var"));
    }

    #[test]
    fn ignores_hyphens_inside_string_literals() {
        // String literals can contain hyphens without being symbols.
        let src = "(def {} f (params {}) (lit {} \"hello-world\"))\n";
        let v = run(src);
        assert!(v.is_empty(), "got: {v:?}");
    }

    #[test]
    fn ignores_hyphens_inside_line_comments() {
        // `;` starts a line comment in Deep. Hyphens in comment prose
        // should be skipped — the comment isn't a symbol.
        let src = "; Wrapped Black-Scholes call-price function (scalar-f32)\n(def {} f (params {}) (lit {} 1))\n";
        let v = run(src);
        assert!(v.is_empty(), "got: {v:?}");
    }

    #[test]
    fn flags_one_per_occurrence() {
        // Two distinct user-defined hyphenated symbols on different lines.
        let src =
            "(def {} foo-bar (params {}) (lit {} 1))\n(def {} baz-qux (params {}) (lit {} 2))\n";
        let v = run(src);
        assert_eq!(v.len(), 2);
        assert_eq!(v[0].line, Some(1));
        assert_eq!(v[1].line, Some(2));
    }

    #[test]
    fn flags_underscore_separated_symbols_pass() {
        // Surf-style snake_case is fine.
        let src = "(def {} my_func (params {}) (var {} my_var))\n";
        let v = run(src);
        assert!(v.is_empty(), "got: {v:?}");
    }

    #[test]
    fn closed_tag_check_is_exact_match() {
        // A token that's a substring of a closed tag but isn't a closed
        // tag itself fires. Example: `pat-var-extra` is not `pat-var`.
        let src = "(def {} pat-var-extra (params {}) (lit {} 1))\n";
        let v = run(src);
        assert_eq!(v.len(), 1);
    }
}
