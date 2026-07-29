//! chelis#731 Phase 3 -- `DeepTag` exhaustive dispatch at `infer_expr`.
//!
//! Spec authority: spec/design/checker_totality.md §C1.1/§C1.2/§C4.2 and
//! spec/04-type-system.md §10 [04-TOT-1]. The compile-time half of the
//! contract is the exhaustive no-`_` match over `chelis_deep::DeepTag` in
//! `infer_expr` (mutation oracle: a 63rd variant fails the build there).
//! This file locks the behavioral half:
//!
//! * every in-vocabulary tag with no expression-position inference case is
//!   an explicit loud `UnknownForm` rejection whose message names the tag
//!   and its real disposition (it is IN the vocabulary; it has no
//!   expression-position case because its checking belongs to an owning
//!   enclosing form (`block` has a real sequencing case per chelis#859),
//!   never the pre-Phase-3 wildcard text that wrongly claimed the tag was
//!   outside the 62-tag vocabulary;
//! * the §C1.2 raw-string arm survives for input that never crossed the
//!   parser's closed-vocabulary screen (a lenient-parsed unknown tag), with
//!   its original outside-the-vocabulary message intact;
//! * both polarities: the dispatched tags still check cleanly on a
//!   well-typed control program.

use chelis_deep::DeepTag;
use chelis_deep::parser::parse_str;
use chelis_types::errors::CheckErrorKind;
use chelis_types::infer_program;

/// The 30 in-vocabulary tags with no expression-position inference case as
/// of Phase 3: declaration internals, patterns, type/dimension syntax,
/// metaprogramming forms, and structural helpers, each checked by its
/// owning enclosing form instead. (`block` graduated to a real sequencing
/// case per chelis#859, closing the one spec/03 §2.3 expression form that
/// lacked one.) This list is the executable record of the dispatch split:
/// moving a tag out of it requires giving the tag a real inference case in
/// `infer_expr`, which is exactly the [04-TOT-1] ratchet.
const NO_EXPRESSION_DISPOSITION: [DeepTag; 30] = [
    DeepTag::Variant,
    DeepTag::Field,
    DeepTag::Arm,
    DeepTag::PatVar,
    DeepTag::PatLit,
    DeepTag::PatCtor,
    DeepTag::PatTuple,
    DeepTag::PatRecord,
    DeepTag::PatWild,
    DeepTag::PatAs,
    DeepTag::TPrim,
    DeepTag::TFn,
    DeepTag::TTensor,
    DeepTag::TRef,
    DeepTag::TAdt,
    DeepTag::TVar,
    DeepTag::TUnit,
    DeepTag::TTuple,
    DeepTag::DName,
    DeepTag::DVar,
    DeepTag::DLit,
    DeepTag::DRank,
    DeepTag::Quote,
    DeepTag::Unquote,
    DeepTag::Splice,
    DeepTag::Params,
    DeepTag::Bind,
    DeepTag::Kv,
    DeepTag::Effects,
    DeepTag::Resource,
];

/// Infer `(def {} f (<tag> {}))` and return the error list.
fn errors_for_body_tag(tag: &str) -> Vec<chelis_types::errors::CheckError> {
    let source = format!("(def {{}} f ({tag} {{}}))");
    let exprs = parse_str(&source).expect("lenient Deep parse");
    infer_program(&exprs).errors
}

// ── Negative polarity: no-disposition tags are rejected loudly ──────────────

#[test]
fn in_vocabulary_tags_without_expression_disposition_are_rejected_loudly() {
    for tag in NO_EXPRESSION_DISPOSITION {
        let errors = errors_for_body_tag(tag.as_str());
        assert!(
            !errors.is_empty(),
            "`{}` in expression position must push a diagnostic, not be \
             silently exempted (chelis#709 class)",
            tag.as_str()
        );
        let unknown_form = errors
            .iter()
            .find(|e| matches!(e.kind, CheckErrorKind::UnknownForm))
            .unwrap_or_else(|| {
                panic!(
                    "`{}` must be rejected as UnknownForm; got {:?}",
                    tag.as_str(),
                    errors
                )
            });
        assert!(
            unknown_form.message.contains(tag.as_str()),
            "the diagnostic must name the tag; got: {}",
            unknown_form.message
        );
        assert!(
            unknown_form
                .message
                .contains("no expression-position checker disposition"),
            "the diagnostic must state the real disposition; got: {}",
            unknown_form.message
        );
        assert!(
            !unknown_form
                .message
                .contains("not in the 62-tag closed vocabulary"),
            "`{}` IS in the vocabulary; the pre-Phase-3 wildcard text is \
             dishonest for it; got: {}",
            tag.as_str(),
            unknown_form.message
        );
    }
}

/// The dispatch split is total: every tag is either in the no-disposition
/// list above or has a real `infer_expr` case; nothing is counted twice.
#[test]
fn no_disposition_list_is_disjoint_and_in_vocabulary() {
    let mut seen = std::collections::HashSet::new();
    for tag in NO_EXPRESSION_DISPOSITION {
        assert!(
            seen.insert(tag),
            "duplicate entry in NO_EXPRESSION_DISPOSITION: {}",
            tag.as_str()
        );
    }
    assert_eq!(
        DeepTag::ALL.len() - NO_EXPRESSION_DISPOSITION.len(),
        32,
        "62 tags split into 32 dispatched and 30 rejected (chelis#859 moved \
         `block` to the dispatched side); a vocabulary change must revisit \
         this file in the same change set (B1)"
    );
}

// ── The §C1.2 raw-string boundary: unknown strings keep the loud arm ────────

#[test]
fn unknown_tag_outside_the_vocabulary_keeps_the_raw_string_loud_arm() {
    let errors = errors_for_body_tag("bogus_wrapper");
    let unknown_form = errors
        .iter()
        .find(|e| matches!(e.kind, CheckErrorKind::UnknownForm))
        .unwrap_or_else(|| panic!("an unknown tag must be UnknownForm; got {errors:?}"));
    assert!(
        unknown_form.message.contains("bogus_wrapper")
            && unknown_form
                .message
                .contains("not in the 62-tag closed vocabulary"),
        "the raw-string arm keeps the outside-the-vocabulary message; got: {}",
        unknown_form.message
    );
}

/// An untagged bare list in expression position also lands in the
/// raw-string arm (there is no tag string to decode).
#[test]
fn untagged_list_in_expression_position_is_rejected_loudly() {
    let exprs = parse_str("(def {} f ((var {} g) (var {} x)))").expect("lenient Deep parse");
    let errors = infer_program(&exprs).errors;
    assert!(
        errors
            .iter()
            .any(|e| matches!(e.kind, CheckErrorKind::UnknownForm)),
        "an untagged list must hit the loud arm; got {errors:?}"
    );
}

// ── chelis#859: `block` has a real sequencing case, both polarities ─────────

#[test]
fn block_types_as_its_last_child() {
    let exprs = parse_str(
        "(def {} f (block {} (lit {type: (t-prim {} int32)} 1) \
         (lit {type: (t-prim {} f32)} 2.0)))",
    )
    .expect("lenient Deep parse");
    let result = infer_program(&exprs);
    assert!(
        result.errors.is_empty(),
        "a well-typed block must check cleanly; got {:?}",
        result.errors
    );
}

#[test]
fn block_checks_every_child_not_only_the_last() {
    // The ill-typed NON-LAST child must be caught: sequencing does not
    // exempt discarded children from checking.
    let exprs = parse_str(
        "(def {} f (block {} (app {} (var {} add) (lit {type: (t-prim {} f32)} 1.0) \
         (lit {type: (t-prim {} int64)} 2)) (lit {type: (t-prim {} f32)} 3.0)))",
    )
    .expect("lenient Deep parse");
    let result = infer_program(&exprs);
    assert!(
        !result.errors.is_empty(),
        "an ill-typed discarded block child must be reported"
    );
}

#[test]
fn childless_block_is_malformed() {
    let errors = errors_for_body_tag("block");
    assert!(
        errors
            .iter()
            .any(|e| matches!(e.kind, CheckErrorKind::MalformedForm)),
        "a childless block has no value and must be MalformedForm; got {errors:?}"
    );
}

// ── Positive polarity: dispatched tags still check cleanly ──────────────────

#[test]
fn dispatched_tags_still_check_cleanly() {
    let source = "(def {} f (lit {type: (t-prim {} f32)} 1.0))\n\
                  (def {} g (if {} (lit {type: (t-prim {} bool)} true) \
                  (var {} f) (lit {type: (t-prim {} f32)} 2.0)))";
    let exprs = parse_str(source).expect("lenient Deep parse");
    let result = infer_program(&exprs);
    assert!(
        result.errors.is_empty(),
        "the well-typed control must stay clean; got {:?}",
        result.errors
    );
}
