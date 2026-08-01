//! Rule `type-suffix-policy` — type/shape suffixes describe the element
//! type or shape of the principal argument, never the container or
//! dispatch form (§7.2).
//!
//! Element-type suffixes (`_int`, `_f32`, `_bool`, `_string`) appear on
//! functions whose principal argument or return value involves the matching
//! element type. They do not appear on functions that work generically OR
//! on functions whose name uses the suffix to mean "different container
//! form" (the Coral `*_int` family violation).
//!
//! Heuristic: a function whose name ends in an element-type suffix where
//! NEITHER the first argument's type string NOR the return type string
//! contains the corresponding type token is a violation. The Coral
//! `is_nan_int(f: Frame, name: string) -> tensor[n, bool]` triggers because
//! neither `Frame, string` nor `tensor[n, bool]` mentions `int` — the
//! suffix means dispatch form, not element type.

use crate::{Context, Rule, Surface, Violation};
use regex::Regex;
use std::sync::OnceLock;

static DEF_RE: OnceLock<Regex> = OnceLock::new();

fn def_re() -> &'static Regex {
    // Match `def funcname(arg: TypeExpr, ...)` or
    // `def funcname[generic_params](arg: TypeExpr, ...)`. Captures:
    //   1: function name
    //   2: first argument type expression (greedy across nested [..])
    //
    // The TypeExpr for the first arg is greedy across nested
    // brackets [..] but stops at the next top-level comma or close-paren.
    DEF_RE.get_or_init(|| {
        Regex::new(
            r"(?m)^[ \t]*def[ \t]+([a-z_][a-z0-9_]*)(?:\[[^\]]*\])?[ \t]*\([ \t]*[a-z_][a-z0-9_]*[ \t]*:[ \t]*([^,\)]+(?:\[[^\]]*\][^,\)]*)*)",
        )
        .unwrap()
    })
}

static RETURN_RE: OnceLock<Regex> = OnceLock::new();

fn return_re() -> &'static Regex {
    // After the close-paren of the parameter list, look for `-> RetType =`
    // or `-> RetType\n`. We invoke this on the slice after the function
    // signature starts, so anchoring isn't needed.
    RETURN_RE.get_or_init(|| Regex::new(r"\)[ \t]*->[ \t]*([^=\n]+)").unwrap())
}

pub struct TypeSuffixPolicy;

#[derive(Debug, Clone, Copy)]
struct Suffix {
    name: &'static str,
    /// Tokens that must appear in either the first arg or return type for
    /// the suffix to match.
    matches: &'static [&'static str],
}

const SUFFIXES: &[Suffix] = &[
    Suffix {
        name: "_int",
        matches: &["int", "i32", "i64", "int32", "int64"],
    },
    Suffix {
        name: "_f32",
        matches: &["f32"],
    },
    Suffix {
        name: "_f64",
        matches: &["f64"],
    },
    Suffix {
        name: "_bool",
        matches: &["bool"],
    },
    Suffix {
        name: "_string",
        matches: &["string"],
    },
];

impl Rule for TypeSuffixPolicy {
    fn id(&self) -> &str {
        "type-suffix-policy"
    }

    fn spec_ref(&self) -> &str {
        "§7.2"
    }

    fn applies_to(&self) -> &[Surface] {
        &[Surface::SurfSource]
    }

    fn summary(&self) -> &str {
        "type/shape suffixes (_int/_f32/_bool/_string) describe element type, not container or dispatch form"
    }

    fn check(&self, ctx: &Context<'_>) -> Vec<Violation> {
        let Some(source) = ctx.source else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for caps in def_re().captures_iter(source) {
            let func_name = caps.get(1).unwrap().as_str();
            let first_arg_type = caps.get(2).unwrap().as_str();
            // Find the return type, if any. Search starting at the end of
            // the signature match.
            let sig_end = caps.get(0).unwrap().end();
            let after = &source[sig_end..];
            // Stop at the first newline that isn't escaped — keep the search
            // on the same line as the def.
            let line_after = after.split('\n').next().unwrap_or("");
            let return_type = return_re()
                .captures(line_after)
                .map(|c| c.get(1).unwrap().as_str().trim().to_string());
            for suffix in SUFFIXES {
                if func_name.ends_with(suffix.name) {
                    // §7.2 "Parser/converter idiom" carve-out: when the
                    // function name starts with a recognized parser-verb
                    // prefix (parse_, unwrap_, is_some_, try_, as_,
                    // from_, to_), the type-suffix names the type the
                    // function tests-for or produces, not the principal
                    // argument's element type. Skip the rule for these.
                    if has_parser_verb_prefix(func_name) {
                        continue;
                    }
                    // Two violation modes per §7.2:
                    //
                    // (1) The suffix doesn't match anywhere in the
                    //     signature — neither first arg nor return type
                    //     mentions the suffix's element type.
                    //
                    // (2) The first argument is a container type (Frame,
                    //     Series, etc.) — the suffix is being used for
                    //     dispatch form rather than element type. This is
                    //     the Coral *_int family pattern: five Frame-taking
                    //     functions where _int means "Frame variant", not
                    //     "operates on int data".
                    let matched_in_arg = type_contains_any(first_arg_type, suffix.matches);
                    let matched_in_ret = return_type
                        .as_deref()
                        .is_some_and(|r| type_contains_any(r, suffix.matches));
                    let first_arg_is_container = type_is_container(first_arg_type);
                    let no_match = !matched_in_arg && !matched_in_ret;
                    if no_match || first_arg_is_container {
                        let match_start = caps.get(0).unwrap().start();
                        let line = source[..match_start]
                            .bytes()
                            .filter(|&b| b == b'\n')
                            .count()
                            + 1;
                        let reason = if first_arg_is_container {
                            format!(
                                "first argument type (`{}`) is a container. The `{}` suffix is being used for dispatch form, which §7.2 prohibits (use `_col` for column-form variants or move to a container-specific module)",
                                first_arg_type.trim(),
                                suffix.name
                            )
                        } else {
                            format!(
                                "neither first argument type (`{}`) nor return type ({}) mentions the matching element type: `{}` suffix doesn't describe the signature",
                                first_arg_type.trim(),
                                return_type
                                    .as_deref()
                                    .map_or("not declared".to_string(), |r| format!("`{r}`")),
                                suffix.name
                            )
                        };
                        out.push(Violation {
                            rule_id: self.id().to_string(),
                            spec_ref: self.spec_ref().to_string(),
                            path: ctx.path.to_path_buf(),
                            line: Some(line),
                            col: None,
                            message: format!(
                                "function `{func_name}` has element-type suffix `{}` but {reason}",
                                suffix.name
                            ),
                        });
                        break; // only fire once per function
                    }
                }
            }
        }
        out
    }
}

/// Recognized parser/converter verb prefixes per §7.2 "Parser/converter
/// idiom". A function whose name starts with one of these may carry a
/// type-suffix that names the type the function tests-for or produces
/// rather than the principal argument's element type.
const PARSER_VERB_PREFIXES: &[&str] = &[
    "parse_", "unwrap_", "is_some_", "try_", "as_", "from_", "to_",
];

fn has_parser_verb_prefix(func_name: &str) -> bool {
    PARSER_VERB_PREFIXES
        .iter()
        .any(|p| func_name.starts_with(p))
}

/// Word-boundary check: does `haystack` contain any of `needles` as a
/// distinct token (not as a substring of a longer identifier)? This
/// distinguishes `int` in `int64` (matches) from `int` in `print` (no match).
fn type_contains_any(haystack: &str, needles: &[&str]) -> bool {
    for needle in needles {
        if has_word(haystack, needle) {
            return true;
        }
    }
    false
}

/// Container types whose presence as the first-argument type indicates
/// that an element-type suffix is being used for dispatch form rather
/// than describing element type (§7.2). Currently:
/// `Frame` (Coral dataframe), `Series` (column-shaped). Add more here
/// as new container types appear in the ecosystem.
const CONTAINER_TYPES: &[&str] = &["Frame", "Series"];

fn type_is_container(type_expr: &str) -> bool {
    for c in CONTAINER_TYPES {
        if has_word(type_expr, c) {
            return true;
        }
    }
    false
}

fn has_word(haystack: &str, needle: &str) -> bool {
    let bytes = haystack.as_bytes();
    let n = needle.as_bytes();
    if n.is_empty() {
        return false;
    }
    let mut i = 0;
    while i + n.len() <= bytes.len() {
        if &bytes[i..i + n.len()] == n {
            let before_ok = i == 0 || !is_ident_byte(bytes[i - 1]);
            let after_ok = i + n.len() == bytes.len() || !is_ident_byte(bytes[i + n.len()]);
            if before_ok && after_ok {
                return true;
            }
        }
        i += 1;
    }
    false
}

fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn run(src: &str) -> Vec<Violation> {
        let path = Path::new("test.ch");
        let ctx = Context {
            root: Path::new("/"),
            path,
            source: Some(src),
            surface: Surface::SurfSource,
        };
        TypeSuffixPolicy.check(&ctx)
    }

    #[test]
    fn flags_coral_is_nan_int_dispatch_form() {
        // The exact snapshot §8 #8 case: Frame + _int suffix.
        let src = "def is_nan_int(f: Frame, name: string) -> tensor[n, bool] = todo\n";
        let v = run(src);
        assert_eq!(v.len(), 1);
        assert!(v[0].message.contains("is_nan_int"));
        assert!(v[0].message.contains("dispatch form"));
    }

    #[test]
    fn flags_coral_drop_nan_int() {
        let src = "def drop_nan_int(f: Frame, name: string) -> Frame = todo\n";
        let v = run(src);
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn accepts_parse_int_string_to_int() {
        // parse_int is a canonical string→int parser. The return type
        // `Option[int64]` contains `int64`, which matches `_int`.
        let src = "def parse_int(s: string) -> Option[int64] = todo\n";
        let v = run(src);
        assert!(v.is_empty(), "got: {v:?}");
    }

    #[test]
    fn accepts_is_some_int_parser_idiom() {
        // is_some_int(s: string) -> bool: neither arg nor return is int,
        // but the parser-verb prefix `is_some_` triggers the §7.2
        // parser/converter idiom carve-out.
        let src = "def is_some_int(value: string) -> bool = todo\n";
        let v = run(src);
        assert!(v.is_empty(), "parser-verb idiom must accept; got: {v:?}");
    }

    #[test]
    fn accepts_unwrap_int_parser_idiom() {
        let src = "def unwrap_int(s: string) -> int64 = todo\n";
        let v = run(src);
        assert!(v.is_empty(), "got: {v:?}");
    }

    #[test]
    fn accepts_to_int_converter_idiom() {
        let src = "def to_int(s: string) -> Option[int64] = todo\n";
        let v = run(src);
        assert!(v.is_empty(), "got: {v:?}");
    }

    #[test]
    fn accepts_from_string_converter_idiom() {
        // from_string_to_bool follows the converter idiom even though
        // _bool isn't in the signature directly — the verb prefix carves
        // out the rule.
        let src = "def from_string_to_bool(s: string) -> bool = todo\n";
        let v = run(src);
        assert!(v.is_empty(), "got: {v:?}");
    }

    #[test]
    fn still_flags_dispatch_form_even_without_parser_prefix() {
        // is_nan_int has no parser-verb prefix (`is_` alone is not in
        // the list — only `is_some_`); first arg is Frame, so still fires.
        let src = "def is_nan_int(f: Frame, name: string) -> tensor[n, bool] = todo\n";
        let v = run(src);
        assert_eq!(v.len(), 1);
        assert!(v[0].message.contains("dispatch form"));
    }

    #[test]
    fn accepts_int_consuming_function() {
        // First arg is int → suffix matches.
        let src = "def double_int(x: int64) -> int64 = todo\n";
        let v = run(src);
        assert!(v.is_empty());
    }

    #[test]
    fn accepts_renamed_col_form() {
        // The corrected form drops _int, uses _col. Rule does not fire on
        // `_col` because `_col` is not on the SUFFIXES list.
        let src = "def is_nan_col(f: Frame, name: string) -> tensor[n, bool] = todo\n";
        let v = run(src);
        assert!(v.is_empty());
    }

    #[test]
    fn flags_count_nan_int_first_arg_frame() {
        let src = "def count_nan_int(f: Frame, name: string) -> int64 = todo\n";
        // count_nan_int has first arg `Frame` — the `_int` suffix is being
        // used for dispatch form even though the return type happens to be
        // int64. This is the C1 violation pattern from the brief: five
        // Coral Frame-taking functions all using _int as dispatch form.
        let v = run(src);
        assert_eq!(v.len(), 1);
        assert!(v[0].message.contains("dispatch form"));
    }

    #[test]
    fn flags_generic_is_nan_int() {
        // The actual coral/src/frame.ch shape: generic dim param after name.
        let src = "def is_nan_int[n](df: Frame[n], col_name: string) -> tensor[n, bool] = todo\n";
        let v = run(src);
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn flags_generic_drop_nan_int() {
        let src = "def drop_nan_int[n, k](df: Frame[n], col_name: string) -> Frame[k] = todo\n";
        let v = run(src);
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn flags_generic_fill_nan_int() {
        let src = "def fill_nan_int[n](df: Frame[n], col_name: string, fill_val: int64) -> Frame[n] = todo\n";
        let v = run(src);
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn flags_fill_nan_int_dispatch_form() {
        // fill_nan_int doesn't have int in arg or return; it's a Frame→Frame
        // operation. _int suffix is dispatch form here.
        let src = "def fill_nan_int(f: Frame, name: string, value: f32) -> Frame = todo\n";
        let v = run(src);
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn accepts_generic_function_no_suffix() {
        // Unsuffixed function name: no rule fires.
        let src = "def is_nan(t: tensor[n, f32]) -> tensor[n, bool] = todo\n";
        let v = run(src);
        assert!(v.is_empty());
    }

    #[test]
    fn ignores_function_without_int_substring_match() {
        // print does NOT end in _int (no underscore boundary).
        let src = "def print(s: string) -> () = todo\n";
        let v = run(src);
        assert!(v.is_empty());
    }

    #[test]
    fn flags_bool_suffix_on_non_bool() {
        let src = "def is_set_bool(f: Frame, name: string) -> tensor[n, f32] = todo\n";
        let v = run(src);
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn accepts_f32_consuming_function() {
        let src = "def relu_f32(x: f32) -> f32 = todo\n";
        let v = run(src);
        assert!(v.is_empty());
    }

    #[test]
    fn word_boundary_int_in_int64() {
        // `int` appears as a sub-substring of `int64`; the word-boundary
        // check should still match because `int64` is the surrounding token.
        assert!(has_word("Option[int64]", "int64"));
        assert!(!has_word("Option[print]", "int"));
    }

    #[test]
    fn word_boundary_int_word() {
        // Bare `int` token matches.
        assert!(has_word("int", "int"));
        // `int` inside `tint` doesn't match (word boundary fails).
        assert!(!has_word("tint", "int"));
    }
}
