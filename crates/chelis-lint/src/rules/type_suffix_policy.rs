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
//! Read signatures from the Surf syntax tree: commas in tensor, tuple, and
//! generic types belong to those types, not to the enclosing parameter list.
//! Match type names and tensor precision tokens, excluding comments and bodies.

use crate::{Context, Rule, Surface, Violation};
use chelis_surf::ast::{Decl, TypeExpr};

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
        matches: &["int", "i32", "i64"],
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
        // Most files have no element-type suffix at all. Avoid another full
        // Surf parse when no declaration could possibly use one. This is only
        // a prefilter; the syntax tree decides every violation.
        if !SUFFIXES.iter().any(|suffix| source.contains(suffix.name)) {
            return Vec::new();
        }
        let Ok(decls) = chelis_surf::parser::parse_str(source, &chelis_syntax_host::NativeStack)
        else {
            // `surf-parses` owns malformed-source diagnostics (§12.5).
            return Vec::new();
        };
        let mut out = Vec::new();
        let mut pending: Vec<_> = decls.iter().collect();
        while let Some(decl) = pending.pop() {
            if let Decl::Module { decls, .. } = decl {
                pending.extend(decls.iter());
            }
            let Decl::FunDef {
                name,
                params,
                ret_ty,
                span,
                ..
            } = decl
            else {
                continue;
            };
            let Some(first_arg_type) = params.first().and_then(|param| param.ty.as_ref()) else {
                // This rule compares declared first-argument and return types;
                // inference, unannotated parameters and nullary defs stay outside it.
                continue;
            };
            let Some(suffix) = SUFFIXES.iter().find(|suffix| name.ends_with(suffix.name)) else {
                continue;
            };
            if has_parser_verb_prefix(name) {
                continue;
            }
            let matched_in_arg = type_contains_any(first_arg_type, suffix.matches);
            let matched_in_ret = ret_ty
                .as_ref()
                .is_some_and(|ty| type_contains_any(ty, suffix.matches));
            let first_arg_is_container = type_contains_any(first_arg_type, CONTAINER_TYPES);
            if (matched_in_arg || matched_in_ret) && !first_arg_is_container {
                continue;
            }
            let first_arg_text = type_source(first_arg_type, source);
            let reason = if first_arg_is_container {
                format!(
                    "first argument type (`{first_arg_text}`) is a container. The `{}` suffix is being used for dispatch form, which §7.2 prohibits (use `_col` for column-form variants or move to a container-specific module)",
                    suffix.name
                )
            } else {
                format!(
                    "neither first argument type (`{first_arg_text}`) nor return type ({}) mentions the matching element type: `{}` suffix doesn't describe the signature",
                    ret_ty.as_ref().map_or("not declared".to_string(), |ty| {
                        format!("`{}`", type_source(ty, source))
                    }),
                    suffix.name
                )
            };
            let line = source[..span.offset]
                .bytes()
                .filter(|&byte| byte == b'\n')
                .count()
                + 1;
            out.push(Violation {
                rule_id: self.id().to_string(),
                spec_ref: self.spec_ref().to_string(),
                path: ctx.path.to_path_buf(),
                line: Some(line),
                col: None,
                message: format!(
                    "function `{name}` has element-type suffix `{}` but {reason}",
                    suffix.name
                ),
            });
        }
        out.sort_by_key(|violation| violation.line);
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

/// Inspect type syntax only, with exact type-name and precision matches.
/// An explicit worklist also handles deeply nested applications and references.
fn type_contains_any(ty: &TypeExpr, names: &[&str]) -> bool {
    let mut pending = vec![ty];
    while let Some(ty) = pending.pop() {
        match ty {
            TypeExpr::Named(name, _) | TypeExpr::RankSpread(name, _) => {
                if matches_name(name, names) {
                    return true;
                }
            }
            TypeExpr::Tensor(dims, precision, _) => {
                if names.iter().any(|name| precision == *name) {
                    return true;
                }
                pending.extend(dims);
            }
            TypeExpr::App(name, args, _) => {
                if matches_name(name, names) {
                    return true;
                }
                pending.extend(args);
            }
            TypeExpr::Arrow(args, ret, _) => {
                pending.extend(args);
                pending.push(ret);
            }
            TypeExpr::Ref(inner, _) => pending.push(inner),
            TypeExpr::Tuple(items, _) => pending.extend(items),
            TypeExpr::DimensionLiteral(_, _) | TypeExpr::Infer(_) => {}
        }
    }
    false
}

fn matches_name(name: &str, names: &[&str]) -> bool {
    names.contains(
        &name
            .rsplit('.')
            .next()
            .expect("a type name has a final component"),
    )
}

fn type_source<'a>(ty: &TypeExpr, source: &'a str) -> &'a str {
    let span = match ty {
        TypeExpr::Named(_, span)
        | TypeExpr::DimensionLiteral(_, span)
        | TypeExpr::Tensor(_, _, span)
        | TypeExpr::Arrow(_, _, span)
        | TypeExpr::Ref(_, span)
        | TypeExpr::App(_, _, span)
        | TypeExpr::Tuple(_, span)
        | TypeExpr::Infer(span)
        | TypeExpr::RankSpread(_, span) => span,
    };
    &source[span.offset..span.end()]
}

/// Container types whose presence as the first-argument type indicates
/// that an element-type suffix is being used for dispatch form rather
/// than describing element type (§7.2). Currently:
/// `Frame` (Coral dataframe), `Series` (column-shaped). Add more here
/// as new container types appear in the ecosystem.
const CONTAINER_TYPES: &[&str] = &["Frame", "Series"];

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
        chelis_surf::parser::parse_str(src, &chelis_syntax_host::NativeStack)
            .expect("rule fixture must parse");
        TypeSuffixPolicy.check(&ctx)
    }

    #[test]
    fn accepts_tensor_input_suffix_when_cast_changes_output_dtype() {
        let src = "module Probe.Suffix\nexport (g_f32)\ndef g_f32[d0](a: tensor[d0, f32]) -> tensor[d0, i64] = cast(a, i64)\n";
        assert!(run(src).is_empty(), "the suffix describes the tensor input");
    }

    #[test]
    fn rejects_tensor_input_suffix_with_no_matching_dtype() {
        let src = "def g_f32[d0](a: tensor[d0, f64]) -> tensor[d0, i64] = cast(a, i64)\n";
        let violations = run(src);
        assert_eq!(violations.len(), 1);
        assert!(violations[0].message.contains("`tensor[d0, f64]`"));
    }

    #[test]
    fn accepts_dtype_in_nested_first_parameter_types() {
        for ty in [
            "Option[tensor[d0, f32]]",
            "(i64, tensor[d0, f32])",
            "&tensor[d0, f32]",
            "(i64 -> tensor[d0, f32])",
        ] {
            let src = format!("def g_f32[d0](a: {ty}, b: i64) -> i64 = b\n");
            assert!(run(&src).is_empty(), "complete first type {ty}");
        }
    }

    #[test]
    fn rejects_mismatched_dtype_in_nested_first_parameter_types() {
        for ty in [
            "Option[tensor[d0, f64]]",
            "(i64, tensor[d0, f64])",
            "&tensor[d0, f64]",
            "(i64 -> tensor[d0, f64])",
        ] {
            let src = format!("def g_f32[d0](a: {ty}, b: i64) -> i64 = b\n");
            assert_eq!(run(&src).len(), 1, "mismatched first type {ty}");
        }
    }

    #[test]
    fn rejects_qualified_and_nested_container_dispatch() {
        for ty in ["Coral.Frame[n]", "&Series[n, f32]", "Option[Frame[n]]"] {
            let src = format!("def mean_f32[n](a: {ty}) -> f32 = 1.0f32\n");
            let violations = run(&src);
            assert_eq!(violations.len(), 1);
            assert!(violations[0].message.contains("dispatch form"));
        }
    }

    #[test]
    fn rejects_matching_dtype_only_in_later_parameters() {
        let src = "def g_f32[d0](a: tensor[d0, f64], b: tensor[d0, f32]) -> i64 = 1i64\n";
        assert_eq!(run(src).len(), 1, "the first argument is principal");
    }

    #[test]
    fn accepts_nested_return_type_after_function_parameter() {
        let src = "def g_f32(a: (i64 -> i64), b: i64) -> Option[tensor[2, f32]] = todo\n";
        assert!(
            run(src).is_empty(),
            "read the declared return, not an inner arrow"
        );
    }

    #[test]
    fn accepts_multiline_tensor_signature() {
        let src = "def g_f32[d0](\n  a: tensor[d0, f32],\n  b: i64\n) -> tensor[d0, i64] = cast(a, i64)\n";
        assert!(run(src).is_empty());
    }

    #[test]
    fn rejects_multiline_mismatched_signature_at_definition_line() {
        let src = "-- leading comment\ndef g_f32[d0](\n  a: tensor[d0, f64]\n) -> tensor[d0, i64] = cast(a, i64)\n";
        let violations = run(src);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].line, Some(2));
    }

    #[test]
    fn ignores_signature_text_in_comments_and_strings() {
        let src = "{-\ndef g_f32(a: i64) -> i64 = 1i64\n-}\ndef text() -> string = \"def h_f32(a: i64) -> i64 = 1i64\"\n";
        assert!(run(src).is_empty());
    }

    #[test]
    fn rejects_dtype_mentioned_only_in_type_comment() {
        let src = "def g_f32[d0](a: tensor[d0, {- f32 -} f64]) -> i64 = 1i64\n";
        assert_eq!(run(src).len(), 1);
    }

    #[test]
    fn rejects_dtype_identifier_substrings() {
        let src = "def g_f32(a: Wrapped_f32) -> i64 = 1i64\n";
        assert_eq!(run(src).len(), 1);
    }

    #[test]
    fn flags_coral_is_nan_int_dispatch_form() {
        // The exact snapshot §8 #8 case: Frame + _int suffix.
        let src = "def is_nan_int[n](f: Frame, name: string) -> tensor[n, bool] = todo\n";
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
        // `Option[i64]` contains `i64`, which matches `_int`.
        let src = "def parse_int(s: string) -> Option[i64] = todo\n";
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
        let src = "def unwrap_int(s: string) -> i64 = todo\n";
        let v = run(src);
        assert!(v.is_empty(), "got: {v:?}");
    }

    #[test]
    fn accepts_to_int_converter_idiom() {
        let src = "def to_int(s: string) -> Option[i64] = todo\n";
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
        let src = "def is_nan_int[n](f: Frame, name: string) -> tensor[n, bool] = todo\n";
        let v = run(src);
        assert_eq!(v.len(), 1);
        assert!(v[0].message.contains("dispatch form"));
    }

    #[test]
    fn accepts_int_consuming_function() {
        // First arg is int → suffix matches.
        let src = "def double_int(x: i64) -> i64 = todo\n";
        let v = run(src);
        assert!(v.is_empty());
    }

    #[test]
    fn accepts_renamed_col_form() {
        // The corrected form drops _int, uses _col. Rule does not fire on
        // `_col` because `_col` is not on the SUFFIXES list.
        let src = "def is_nan_col[n](f: Frame, name: string) -> tensor[n, bool] = todo\n";
        let v = run(src);
        assert!(v.is_empty());
    }

    #[test]
    fn flags_count_nan_int_first_arg_frame() {
        let src = "def count_nan_int(f: Frame, name: string) -> i64 = todo\n";
        // count_nan_int has first arg `Frame` — the `_int` suffix is being
        // used for dispatch form even though the return type happens to be
        // i64. This is the C1 violation pattern from the brief: five
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
        let src = "def fill_nan_int[n](df: Frame[n], col_name: string, fill_val: i64) -> Frame[n] = todo\n";
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
        let src = "def is_nan[n](t: tensor[n, f32]) -> tensor[n, bool] = todo\n";
        let v = run(src);
        assert!(v.is_empty());
    }

    #[test]
    fn ignores_function_without_int_substring_match() {
        // print does NOT end in _int (no underscore boundary).
        let src = "def print(s: string) -> unit = todo\n";
        let v = run(src);
        assert!(v.is_empty());
    }

    #[test]
    fn flags_bool_suffix_on_non_bool() {
        let src = "def is_set_bool[n](f: Frame, name: string) -> tensor[n, f32] = todo\n";
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
        assert!(matches_name("i64", &["i64"]));
        assert!(!matches_name("print", &["int"]));
    }

    #[test]
    fn word_boundary_int_word() {
        assert!(matches_name("int", &["int"]));
        assert!(!matches_name("tint", &["int"]));
    }
}
