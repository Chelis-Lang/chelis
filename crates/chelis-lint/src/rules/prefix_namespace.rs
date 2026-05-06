//! Rule `prefix-namespace` — domain-shorthand prefixes on function names
//! within a module follow §7.1 (private/helper marker) or §7.1.1
//! (model/algorithm sub-namespace). Either way, the prefix must be
//! consistent with the module's name.
//!
//! The lint detects function-name prefixes of the form `xx_*` (2–4
//! lowercase letters + underscore) that appear on multiple `def`s within
//! a single Surf source file. For each detected prefix, it checks whether
//! the prefix matches the module's expected shorthand:
//!
//! - The lowercase initials of the last PascalCase compound component
//!   (e.g., `Nautilus.LinAlg` → `la`)
//! - The full lowercased component name (e.g., `Frame` → `frame`)
//! - The first 2–4 characters of the lowercased component name
//!   (e.g., `Pricing` → `pri`)
//!
//! Prefixes that match are accepted (they're the documented private-helper
//! marker). Prefixes that don't match are surfaced for the orchestrator's
//! closer-read step (B1 outcome A vs B): either drop the prefix per §7.1
//! or amend §7.1.1 to record it as a model/algorithm sub-namespace.
//!
//! Snapshot §8 #6 (`la_*` in Nautilus.LinAlg): accepted — `la` matches
//! LinAlg's initials.
//! Snapshot §8 #7 (`bs_*` in Shoals.Pricing): flagged — `bs` doesn't
//! match Pricing's shorthand. Orchestrator's closer-read will determine
//! whether to drop or escalate to §7.1.1 amendment.

use super::module_decl::find_module_decls;
use crate::{Context, Rule, Surface, Violation};
use regex::Regex;
use std::collections::BTreeMap;
use std::sync::OnceLock;

static DEF_NAME_RE: OnceLock<Regex> = OnceLock::new();

fn def_name_re() -> &'static Regex {
    DEF_NAME_RE.get_or_init(|| {
        // Matches `def funcname` at line start, optionally followed by
        // generic-param brackets or open-paren. Captures the function name.
        Regex::new(r"(?m)^[ \t]*def[ \t]+([a-z_][a-z0-9_]*)").unwrap()
    })
}

pub struct PrefixNamespace;

impl Rule for PrefixNamespace {
    fn id(&self) -> &str {
        "prefix-namespace"
    }

    fn spec_ref(&self) -> &str {
        "§7.1"
    }

    fn applies_to(&self) -> &[Surface] {
        &[Surface::SurfSource]
    }

    fn summary(&self) -> &str {
        "function-name prefixes within a module are the module's domain shorthand (§7.1 helper-marker) or a documented model/algorithm sub-namespace (§7.1.1)"
    }

    fn check(&self, ctx: &Context<'_>) -> Vec<Violation> {
        let Some(source) = ctx.source else {
            return Vec::new();
        };
        // The module declaration determines what shorthand is expected.
        // A file with no module declaration can't be checked against
        // module-name shorthand, so we skip.
        let decls = find_module_decls(source);
        let Some(decl) = decls.first() else {
            return Vec::new();
        };
        let Some(last_component) = decl.components.last() else {
            return Vec::new();
        };
        let allowed_prefixes = expected_shorthand_set(last_component);
        // Group function names by their leading 2–4 char prefix.
        let mut prefix_groups: BTreeMap<String, Vec<(String, usize)>> = BTreeMap::new();
        for caps in def_name_re().captures_iter(source) {
            let func_name = caps.get(1).unwrap().as_str();
            let match_start = caps.get(0).unwrap().start();
            let line = source[..match_start]
                .bytes()
                .filter(|&b| b == b'\n')
                .count()
                + 1;
            if let Some(prefix) = extract_short_prefix(func_name) {
                prefix_groups
                    .entry(prefix)
                    .or_default()
                    .push((func_name.to_string(), line));
            }
        }
        let mut out = Vec::new();
        for (prefix, occurrences) in prefix_groups {
            // A "module prefix candidate" is a prefix used by 2+ functions.
            // Single-occurrence prefixes are noise (e.g., `to_int`,
            // `is_nan`); they don't constitute a namespace.
            if occurrences.len() < 2 {
                continue;
            }
            if allowed_prefixes.contains(&prefix) {
                continue;
            }
            // Fire one violation per function, naming the prefix and the
            // module's expected shorthand for closer-read context.
            let module_path = decl.components.join(".");
            let allowed_list: Vec<String> = allowed_prefixes.to_vec();
            let allowed_hint = if allowed_list.is_empty() {
                "no shorthand could be derived".to_string()
            } else {
                format!("expected one of: {}", allowed_list.join(", "))
            };
            let group_size = occurrences.len();
            for (func_name, line) in &occurrences {
                out.push(Violation {
                    rule_id: self.id().to_string(),
                    spec_ref: self.spec_ref().to_string(),
                    path: ctx.path.to_path_buf(),
                    line: Some(*line),
                    col: None,
                    message: format!(
                        "function `{func_name}` uses prefix `{prefix}_` shared by {} other def(s) in `{module_path}`, but `{prefix}` does not match the module's domain shorthand ({allowed_hint}) — closer-read needed: either drop the prefix per §7.1 or document the prefix as a model/algorithm sub-namespace per §7.1.1",
                        group_size - 1
                    ),
                });
            }
        }
        out
    }
}

/// Common-verb prefixes that should never be treated as domain-shorthand
/// namespacing under §7.1. These are universal patterns (`test_*` is the
/// §10.1 test convention; `from_*`/`to_*` are constructor/converter
/// idioms; etc.) — not module-specific shorthand. Any function name with
/// one of these prefixes is excluded from the rule's prefix-grouping.
const COMMON_VERB_PREFIXES: &[&str] = &[
    "test", "from", "to", "is", "has", "get", "set", "with", "add", "new", "make", "for", "as",
    "into", "try", "do", "run", "assert", "build", "load", "save", "read", "write", "open",
    "close", "fmt", "emit", "parse", "lex", "render", "format", "decode", "encode", "find",
    "lookup", "drop", "fill", "count", "any", "all", "map", "fold", "reduce", "scan", "iter",
    "zip", "flat", "chain", "zero", "one", "init", "create", "alloc", "free", "log", "trace",
    "debug", "warn", "info", "error", "panic",
    // Generic accessor-verb idioms (§7.1 doesn't apply — same shape as
    // top_*/head_*/first_*).
    "best", "top", "head", "tail", "last", "first",
    // Test/construction helpers — `mk_*` is the test-fixture-builder
    // idiom paralleling `new_*`/`make_*`.
    "mk",
    // Constant / utility prefixes — `nan_*` is NaN guards, `sqrt_*` is
    // precomputed-sqrt constants, `two_*` is "two" math constants. None
    // are domain-shorthand sub-namespaces.
    "nan", "sqrt", "two", // Domain verbs (shared meaning, not shorthand)
    "put", "call",
    // Time/date accessor verbs in Std.Time — `date_*`/`day_*`/`days_*`
    // are coherent date-handling helpers parallel to `hour_*`/`year_*`
    // patterns elsewhere.
    "date", "day", "days", "year", "month", "hour", "min", "sec",
    // Aggregation verbs in dataframe code (Coral.Frame, Coral.GroupBy):
    // `sum_*`, `mean_*`, `max_*`, `min_*`, `count_*` (already), `mask_*`.
    "mean", "max", "mask", "sum",
    // Data-structure operation verbs (Coral.Frame internals):
    // `list_*`, `key_*`, `hash_*`, `enum_*`, `char_*`, `ints_*`, `agg_*`,
    // `melt_*`, `csv_*`, `json_*`. Each names a coherent op-on-X family.
    "list", "key", "hash", "enum", "char", "ints", "agg", "melt", "csv", "json",
    // Join / set-relation idioms (Coral.GroupBy / Coral.Join):
    // `left_*`, `right_*`, `inner_*`, `outer_*`, `full_*`, `join_*`.
    "left", "right", "inner", "outer", "full", "join",
    // RNG seed accessor verbs in Std.Tokenizer.
    "seed",
    // Example-module-only fixture-helper prefixes (Nautilus.ExampleODEDemo,
    // Nautilus.ExampleOptim): `eo_*` and `eop_*` build per-example
    // problem fixtures.
    "eo", "eop", // Type prefixes (orthogonal to domain shorthand)
    "int", "f32", "f64", "i32", "i64", "u32", "u64", "u8", "str", "vec", "ref", "ptr",
];

/// Recognized model/algorithm sub-namespace prefixes per §7.1.1. These
/// name distinct models, algorithms, mathematical objects, or numerical
/// methods within a module that hosts multiple coexisting variants. The
/// prefix is uniformly applied to every member of its family. See §7.1.1
/// of `chelis/spec/01-nomenclature.md` for the canonical list and the
/// criteria for adding new entries.
const MODEL_NAMESPACE_PREFIXES: &[&str] = &[
    // Pricing models (Shoals.Pricing)
    "bs", "mc", // Stochastic processes (Shoals.Stochastic)
    "gbm",
    // Numerical methods (Shoals.Properties.Greeks; Nautilus.CurveFit; Nautilus.LinAlg)
    "fd", "lm", "cg",
    // Mathematical-object families (Nautilus.Special; Nautilus.Distributions; Nautilus.LinAlg)
    "airy", "beta", "chi", "det", "eig", "inv", "erf", // Math function families
    "lamb",
];

/// Extract a 2–4 lowercase-letter prefix followed by `_` from a function
/// name. Returns the prefix without the underscore. Filters out both the
/// common-verb prefixes in [`COMMON_VERB_PREFIXES`] and the recognized
/// §7.1.1 model/algorithm prefixes in [`MODEL_NAMESPACE_PREFIXES`].
fn extract_short_prefix(name: &str) -> Option<String> {
    let bytes = name.as_bytes();
    for end in 2..=4 {
        if end >= bytes.len() {
            break;
        }
        if bytes[end] == b'_' && bytes[..end].iter().all(|b| b.is_ascii_lowercase()) {
            let prefix = &name[..end];
            if COMMON_VERB_PREFIXES.contains(&prefix) {
                return None;
            }
            if MODEL_NAMESPACE_PREFIXES.contains(&prefix) {
                return None;
            }
            return Some(prefix.to_string());
        }
    }
    None
}

/// Compute the set of acceptable shorthands derived from a module-component
/// name. Any of these is a valid private-helper prefix per §7.1.
///
/// - Initials of compound (LinAlg → "la", OrderBook → "ob", CurveFit → "cf")
/// - Full lowercased name (Frame → "frame", LinAlg → "linalg")
/// - First 2–4 lowercase letters (Pricing → "pr", "pri", "pric")
fn expected_shorthand_set(component: &str) -> Vec<String> {
    let mut out = Vec::new();
    // Initials of PascalCase compound.
    let initials: String = component
        .chars()
        .filter(|c| c.is_ascii_uppercase())
        .map(|c| c.to_ascii_lowercase())
        .collect();
    if (2..=4).contains(&initials.len()) {
        out.push(initials);
    }
    // Full lowercased name (only useful if it's 2–4 chars; otherwise it's
    // the wrong shape for a prefix).
    let full = component.to_ascii_lowercase();
    if (2..=4).contains(&full.len()) {
        out.push(full.clone());
    }
    // First 2–4 chars of the lowercased name.
    for n in 2..=4 {
        if full.len() >= n {
            out.push(full[..n].to_string());
        }
    }
    out.sort();
    out.dedup();
    out
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
        PrefixNamespace.check(&ctx)
    }

    #[test]
    fn extract_short_prefix_examples() {
        // la_ is a domain shorthand (matches LinAlg) — extracted, then
        // accepted by the rule when the module is LinAlg.
        assert_eq!(extract_short_prefix("la_vec_sub"), Some("la".to_string()));
        // bs_/mc_/gbm_ are §7.1.1 model namespaces — filtered, return None.
        assert_eq!(extract_short_prefix("bs_call_scalar"), None);
        assert_eq!(extract_short_prefix("mc_solve"), None);
        assert_eq!(extract_short_prefix("gbm_terminal"), None);
        assert_eq!(
            extract_short_prefix("pric_helper"),
            Some("pric".to_string())
        );
        // No underscore at position 2-4: not a prefix candidate.
        assert_eq!(extract_short_prefix("transpose"), None);
        assert_eq!(extract_short_prefix("matmul_wrap"), None); // matmul is 6 chars, not 2-4
    }

    #[test]
    fn common_verb_prefixes_skipped() {
        // §10.1 test convention; not a domain prefix.
        assert_eq!(extract_short_prefix("test_softmax"), None);
        // Constructor/converter idioms.
        assert_eq!(extract_short_prefix("from_pairs"), None);
        assert_eq!(extract_short_prefix("to_string"), None);
        // Predicate/accessor verbs.
        assert_eq!(extract_short_prefix("is_nan"), None);
        assert_eq!(extract_short_prefix("get_column"), None);
        // Type prefixes.
        assert_eq!(extract_short_prefix("int_col_of_list"), None);
        // Domain verbs that have shared meaning across modules.
        assert_eq!(extract_short_prefix("put_call_parity"), None);
        assert_eq!(extract_short_prefix("call_price"), None);
        // Generic accessor-verb idioms (best_bid, best_ask).
        assert_eq!(extract_short_prefix("best_bid"), None);
        assert_eq!(extract_short_prefix("best_ask"), None);
    }

    #[test]
    fn model_namespace_prefixes_skipped() {
        // §7.1.1 recognized model/algorithm sub-namespaces.
        assert_eq!(extract_short_prefix("bs_call_scalar"), None);
        assert_eq!(extract_short_prefix("mc_call_price"), None);
        assert_eq!(extract_short_prefix("gbm_path"), None);
        assert_eq!(
            extract_short_prefix("fd_delta_in_unit_range_for_call"),
            None
        );
        assert_eq!(extract_short_prefix("lm_jcol"), None);
        assert_eq!(extract_short_prefix("cg_solve"), None);
        assert_eq!(extract_short_prefix("airy_ai"), None);
        assert_eq!(extract_short_prefix("beta_pdf"), None);
        assert_eq!(extract_short_prefix("chi_squared_cdf"), None);
        assert_eq!(extract_short_prefix("det_2x2"), None);
        assert_eq!(extract_short_prefix("eig_2x2"), None);
        assert_eq!(extract_short_prefix("inv_2x2"), None);
    }

    #[test]
    fn shorthand_for_linalg() {
        let s = expected_shorthand_set("LinAlg");
        assert!(s.contains(&"la".to_string()), "got: {s:?}");
    }

    #[test]
    fn shorthand_for_frame() {
        let s = expected_shorthand_set("Frame");
        assert!(
            s.contains(&"fra".to_string()) || s.contains(&"frame".to_string()),
            "got: {s:?}"
        );
    }

    #[test]
    fn shorthand_for_pricing() {
        let s = expected_shorthand_set("Pricing");
        assert!(s.contains(&"pr".to_string()), "got: {s:?}");
        assert!(s.contains(&"pri".to_string()), "got: {s:?}");
    }

    #[test]
    fn shorthand_for_curvefit() {
        let s = expected_shorthand_set("CurveFit");
        assert!(s.contains(&"cf".to_string()), "got: {s:?}");
    }

    // End-to-end rule tests: §7.1 violations.

    #[test]
    fn accepts_la_prefix_in_linalg() {
        let src = "module Nautilus.LinAlg\ndef la_vec_sub(a: f32) = todo\ndef la_vec_add(a: f32) = todo\ndef la_qr_step(a: f32) = todo\n";
        let v = run(src);
        assert!(v.is_empty(), "la_ matches LinAlg shorthand; got: {v:?}");
    }

    #[test]
    fn accepts_bs_prefix_in_pricing_per_section_7_1_1() {
        // Snapshot §8 #7 case, post-§7.1.1 amendment: bs_ in
        // Shoals.Pricing is a recognized model sub-namespace. The lint
        // does NOT fire — Outcome A from the brief.
        let src = "module Shoals.Pricing\ndef bs_call_scalar(s: f32) = todo\ndef bs_put_scalar(s: f32) = todo\ndef call_prices(s: f32) = todo\ndef put_prices(s: f32) = todo\n";
        let v = run(src);
        assert!(v.is_empty(), "got: {v:?}");
    }

    #[test]
    fn accepts_mc_and_gbm_prefixes_per_section_7_1_1() {
        // Both mc_ (Monte-Carlo pricing) and gbm_ (geometric Brownian
        // motion) are §7.1.1 recognized prefixes. The rule does not fire.
        let src = "module Shoals.Stochastic\ndef mc_step(s: f32) = todo\ndef mc_solve(s: f32) = todo\ndef gbm_terminal(s: f32) = todo\ndef gbm_path(s: f32) = todo\n";
        let v = run(src);
        assert!(v.is_empty(), "got: {v:?}");
    }

    #[test]
    fn still_flags_unrecognized_prefix_with_2_plus_uses() {
        // A prefix not in either filter list, used by 2+ functions, with
        // no module-shorthand match → fires (closer-read trigger).
        // `xyz_` is fictional; not in any allowlist.
        let src = "module Foo.Bar\ndef xyz_alpha(s: f32) = todo\ndef xyz_beta(s: f32) = todo\n";
        let v = run(src);
        assert_eq!(v.len(), 2, "got: {v:?}");
    }

    #[test]
    fn no_fire_when_only_one_function_uses_prefix() {
        // Single occurrence of a prefix doesn't constitute a namespace.
        let src = "module Test\ndef bs_solo(x: f32) = todo\ndef other(x: f32) = todo\n";
        let v = run(src);
        assert!(v.is_empty());
    }

    #[test]
    fn no_fire_on_short_words_without_underscore() {
        // `def is_nan` has 'is' as the first 2 chars but no underscore-2,
        // so extract_short_prefix returns Some("is"). If 'is' is also used
        // by another function, this could fire. Test that it doesn't fire
        // on common verb prefixes that match the module shorthand.
        let src = "module Is\ndef is_nan(x: f32) = todo\ndef is_inf(x: f32) = todo\n";
        let v = run(src);
        // Module is named `Is`, lowercased = "is", which is in shorthand
        // set. So `is_*` is allowed. Empty.
        assert!(v.is_empty(), "got: {v:?}");
    }

    #[test]
    fn no_fire_on_skip_when_no_module() {
        let src = "def la_helper(x: f32) = todo\ndef la_other(x: f32) = todo\n";
        let v = run(src);
        // No module declaration → rule skips.
        assert!(v.is_empty());
    }
}
