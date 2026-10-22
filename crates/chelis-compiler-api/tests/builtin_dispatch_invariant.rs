//! Lock invariant: every name in `BUILTIN_NAMES` must either have a
//! dispatch arm in `eval_builtin` (in `crates/chelis-compiler-api/src/
//! runtime/eval.rs`) OR appear in the documented allowlist below.
//!
//! The allowlist is for builtins that exist only at type-check / IR-emit
//! time and never reach the host-runtime evaluator. Each entry MUST have
//! a `//` comment immediately above it stating the reason, drawn from
//! the closed vocabulary:
//!
//!   - `type-only` exists only for typing or signature lookups
//!   - `target=<backend>-only` only emitted on a specific backend target
//!
//! "Lowered before eval" is not a reason: `chelis eval --file` and
//! `chelis test` run this host evaluator on the checked program without
//! IR lowering, so a builtin the checker admits reaches `eval_builtin`
//! (chelis#2892, where `scatter_replace` was allowlisted on that ground).
//!
//! Issue Chelis-Lang/chelis#185 closed the original 12-builtin gap plus a
//! sibling-sweep handful; this invariant test catches future regressions
//! where someone adds a name to `BUILTIN_NAMES` without wiring the
//! host-runtime arm.

use chelis_unord::UnordSet;
use std::fs;
use std::path::PathBuf;

use chelis_numeric_core::dtype_semantics::FloatClassify;
use chelis_types::BUILTIN_NAMES;

/// Builtins that legitimately do not need a host-runtime arm. Document
/// the reason per the closed vocabulary above.
const HOST_RUNTIME_ALLOWLIST: &[&str] = &[];

fn runtime_source_path() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest.join("src").join("runtime").join("eval.rs")
}

/// Read the final `match name` owned by `eval_builtin`, rather than text in
/// comments, nested helpers, or unrelated functions. Literal arms and the closed
/// typed classifier family are the two dispatch forms admitted by this oracle.
fn dispatched_names_in_source(source: &str) -> UnordSet<String> {
    let file = syn::parse_file(source).expect("host-runtime source must parse");
    let functions: Vec<_> = file
        .items
        .iter()
        .filter_map(|item| {
            let syn::Item::Impl(implementation) = item else {
                return None;
            };
            Some(implementation.items.iter().filter_map(|item| {
                let syn::ImplItem::Fn(function) = item else {
                    return None;
                };
                (function.sig.ident == "eval_builtin").then_some(&function.block)
            }))
        })
        .flatten()
        .collect();
    assert_eq!(
        functions.len(),
        1,
        "exactly one eval_builtin owner is required"
    );
    let Some(syn::Stmt::Expr(syn::Expr::Match(dispatch), None)) = functions[0].stmts.last() else {
        panic!("eval_builtin must return its dispatch match");
    };
    assert_eq!(
        *dispatch.expr,
        syn::parse_quote!(name),
        "dispatch must select the builtin name"
    );

    let classifier: syn::Arm = syn::parse_str(CLASSIFIER_ARM).expect("reviewed classifier arm");
    let mut out = UnordSet::new();
    for arm in &dispatch.arms {
        if arm == &classifier {
            // Membership comes from the same canonical resolver as production,
            // not a second spelling inventory or a runtime allowlist.
            for name in BUILTIN_NAMES {
                if let Some(operation) = FloatClassify::from_name(name) {
                    assert_eq!(
                        operation.name(),
                        *name,
                        "classifier identity must round-trip"
                    );
                    out.insert((*name).to_string());
                }
            }
        } else if arm.guard.is_none() {
            literal_pattern_names(&arm.pat, &mut out);
        }
    }
    out
}

fn literal_pattern_names(pattern: &syn::Pat, out: &mut UnordSet<String>) {
    match pattern {
        syn::Pat::Lit(literal) => {
            if let syn::Lit::Str(name) = &literal.lit {
                out.insert(name.value());
            }
        }
        syn::Pat::Or(alternatives) => {
            for pattern in &alternatives.cases {
                literal_pattern_names(pattern, out);
            }
        }
        _ => {}
    }
}

fn extracted_dispatched_names() -> UnordSet<String> {
    let source = fs::read_to_string(runtime_source_path())
        .expect("runtime/eval.rs must be readable from CARGO_MANIFEST_DIR/src");
    dispatched_names_in_source(&source)
}

#[test]
fn every_builtin_name_has_a_host_runtime_arm_or_an_allowlist_entry() {
    let dispatched = extracted_dispatched_names();
    let allowlisted: UnordSet<&str> = HOST_RUNTIME_ALLOWLIST.iter().copied().collect();

    let mut missing: Vec<&str> = BUILTIN_NAMES
        .iter()
        .copied()
        .filter(|name| !dispatched.contains(*name) && !allowlisted.contains(name))
        .collect();
    missing.sort_unstable();

    assert!(
        missing.is_empty(),
        "the following BUILTIN_NAMES entries have no eval_builtin arm and \
         no allowlist entry: {missing:?}. Add a dispatch arm in \
         crates/chelis-compiler-api/src/runtime/eval.rs::eval_builtin OR add an \
         allowlist entry to HOST_RUNTIME_ALLOWLIST in this test with a \
         documented reason."
    );
}

#[test]
fn allowlist_does_not_shadow_any_dispatched_name() {
    // A name should not appear in BOTH the dispatch arms AND the
    // allowlist. If it does, the allowlist comment is stale and the
    // reason no longer applies. Catch the drift early.
    let dispatched = extracted_dispatched_names();
    let mut overlap: Vec<&str> = HOST_RUNTIME_ALLOWLIST
        .iter()
        .copied()
        .filter(|name| dispatched.contains(*name))
        .collect();
    overlap.sort_unstable();

    assert!(
        overlap.is_empty(),
        "the following names appear in BOTH the dispatch arms AND the \
         allowlist: {overlap:?}. Remove them from HOST_RUNTIME_ALLOWLIST \
         since they now have a real host-runtime arm."
    );
}

#[test]
fn allowlist_entries_are_in_builtin_names() {
    // Sanity: an allowlist entry that isn't in BUILTIN_NAMES is dead
    // configuration. The invariant the allowlist documents is "the
    // following BUILTIN_NAMES entries are intentionally not dispatched";
    // an entry that no longer appears in BUILTIN_NAMES has no
    // referent.
    let builtin_set: UnordSet<&str> = BUILTIN_NAMES.iter().copied().collect();
    let mut stale: Vec<&str> = HOST_RUNTIME_ALLOWLIST
        .iter()
        .copied()
        .filter(|name| !builtin_set.contains(name))
        .collect();
    stale.sort_unstable();

    assert!(
        stale.is_empty(),
        "the following allowlist entries are not in BUILTIN_NAMES (likely \
         the name was removed without updating the allowlist): {stale:?}"
    );
}

fn classifier_fixture(arm: &str) -> String {
    format!(
        "impl Runtime {{ fn eval_builtin() {{ match name {{ {arm} _ => unreachable!(), }} }} }}"
    )
}

const CLASSIFIER_ARM: &str = r#"
name if chelis_numeric_core::dtype_semantics::FloatClassify::from_name(name).is_some() => {
    eval_float_classification(
        chelis_numeric_core::dtype_semantics::FloatClassify::from_name(name).expect("classification identity"),
        args,
    )
}
"#;

fn assert_classifiers_present(source: &str, expected: bool) {
    let names = dispatched_names_in_source(source);
    for operation in [
        FloatClassify::Nan,
        FloatClassify::Finite,
        FloatClassify::Infinite,
    ] {
        assert_eq!(
            names.contains(operation.name()),
            expected,
            "{}",
            operation.name()
        );
    }
}

#[test]
fn closed_classifier_arm_covers_its_canonical_names() {
    assert_classifiers_present(&classifier_fixture(CLASSIFIER_ARM), true);
}

#[test]
fn absent_or_erased_classifier_handler_does_not_count() {
    assert_classifiers_present(&classifier_fixture(""), false);
    let erased = CLASSIFIER_ARM.replace(
        "eval_float_classification(\n        chelis_numeric_core::dtype_semantics::FloatClassify::from_name(name).expect(\"classification identity\"),\n        args,\n    )",
        "unreachable!()",
    );
    assert_ne!(erased, CLASSIFIER_ARM);
    assert_classifiers_present(&classifier_fixture(&erased), false);
}

#[test]
fn wrong_classifier_guard_identity_or_arguments_do_not_count() {
    for (old, new) in [
        (".is_some()", ".is_none()"),
        ("name if", "other_name if"),
        ("from_name(name).expect", "from_name(\"is_nan\").expect"),
        (
            ".expect(\"classification identity\")",
            ".unwrap_or(chelis_numeric_core::dtype_semantics::FloatClassify::Nan)",
        ),
        ("        args,", "        other_args,"),
        ("eval_float_classification(", "different_handler("),
    ] {
        let changed = CLASSIFIER_ARM.replace(old, new);
        assert_ne!(changed, CLASSIFIER_ARM);
        assert_classifiers_present(&classifier_fixture(&changed), false);
    }
}

#[test]
fn classifier_evidence_outside_the_dispatch_arm_does_not_count() {
    let outside = format!(
        "{} impl Other {{ fn another() {{ match name {{ {CLASSIFIER_ARM} _ => unreachable!(), }} }} }}",
        classifier_fixture("")
    );
    assert_classifiers_present(&outside, false);
    let comment = format!("/* {CLASSIFIER_ARM} */");
    assert_classifiers_present(&classifier_fixture(&comment), false);
    let closure = CLASSIFIER_ARM
        .replace(
            "    eval_float_classification(",
            "    let unused = || eval_float_classification(",
        )
        .replace("\n    )\n}", "\n    ); unreachable!()\n}");
    assert_ne!(closure, CLASSIFIER_ARM);
    assert_classifiers_present(&classifier_fixture(&closure), false);
}

#[test]
fn literal_dispatch_reads_only_actual_unguarded_match_patterns() {
    let source = classifier_fixture(
        r#""expand" | "insert" => handler(args), "wrong" if false => unreachable!(), /* "comment" => unused, */ "elsewhere" => { let _s = "\"literal\" => unused"; other(args) },"#,
    );
    let names = dispatched_names_in_source(&source);
    assert!(names.contains("expand"));
    assert!(names.contains("insert"));
    assert!(names.contains("elsewhere"));
    assert!(!names.contains("wrong"));
    assert!(!names.contains("comment"));
    assert!(!names.contains("literal"));
}

#[test]
fn removing_or_erasing_the_production_classifier_arm_is_detected() {
    let source = fs::read_to_string(runtime_source_path()).expect("host-runtime source");
    let actual_arm = CLASSIFIER_ARM
        .trim()
        .lines()
        .map(|line| format!("            {line}"))
        .collect::<Vec<_>>()
        .join("\n");
    assert_eq!(
        source.matches(&actual_arm).count(),
        1,
        "exact production arm required"
    );
    assert_classifiers_present(&source, true);
    let removed = source.replacen(&actual_arm, "", 1);
    assert_classifiers_present(&removed, false);
    let erased_arm = actual_arm.replace("eval_float_classification(", "different_handler(");
    assert_ne!(erased_arm, actual_arm);
    let erased = source.replacen(&actual_arm, &erased_arm, 1);
    assert_classifiers_present(&erased, false);
}
