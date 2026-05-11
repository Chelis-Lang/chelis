//! Integration test for Item 3 of the 0.7.6 toolchain hygiene workstream:
//! `deep-user-symbol-charset` must accept every emitted compound tag from
//! the canonical Deep tag vocabulary (`spec/01-nomenclature.md` §1.4,
//! `spec/03-deep-syntax.md` §2.5/§2.6, `chelis-deep/src/validate.rs`
//! `VALID_TAGS`).
//!
//! The lint and the emitter currently conflict: `chelis deep` emits
//! canonical compound tags like `t-ref` that the lint's `CLOSED_TAGS`
//! allowlist rejects on the same file. These fixtures pin that gap.
//!
//! References:
//! - plan: `/home/<user>/.claude/plans/build-up-a-plan-mossy-meteor.md`,
//!   Item 3 §3.1
//! - rule: `crates/chelis-lint/src/rules/deep_user_symbol_charset.rs`
//! - canonical vocabulary: `crates/chelis-deep/src/validate.rs`

use chelis_lint::rules::deep_user_symbol_charset::DeepUserSymbolCharset;
use chelis_lint::{Context, Rule, Surface, Violation};
use std::path::Path;

fn run(src: &str) -> Vec<Violation> {
    let path = Path::new("fixture.dp");
    let ctx = Context {
        root: Path::new("/"),
        path,
        source: Some(src),
        surface: Surface::DeepSource,
    };
    DeepUserSymbolCharset.check(&ctx)
}

/// The user-reported failure shape: a `t-ref` node appears in emitted
/// Deep (read-only borrow type, `spec/03-deep-syntax.md` §2.5) and the
/// lint rejects the program. After the allowlist fix, this passes.
#[test]
fn accepts_t_ref_compound_tag() {
    // A typed function parameter `x: &Tensor[f32, [n]]` desugars to a
    // `(t-ref {} (t-tensor ...))` annotation in Deep.
    let src = r#"(def {type: (t-fn {} (t-ref {} (t-tensor {} (d-var {} n) (t-prim {} f32))) (t-prim {} f32))}
  read_only (params {} x)
  (app {} (var {} sum) (var {} x)))
"#;
    let v = run(src);
    assert!(
        v.is_empty(),
        "expected zero violations for canonical `t-ref` tag, got: {v:?}"
    );
}

/// Concatenated fixture covering every canonical hyphenated compound
/// tag emitted by the toolchain (per `chelis-deep/src/validate.rs`
/// `VALID_TAGS` and `spec/03-deep-syntax.md` §2). The lint must accept
/// the whole program because every hyphenated symbol is in the closed
/// vocabulary.
///
/// The tags exercised below correspond to plan §3.1's enumeration
/// table, restricted to tags that actually appear in the canonical
/// closed vocabulary. `t-dims` is intentionally excluded: it appears in
/// the plan's enumeration table but is not in `VALID_TAGS` and is not
/// emitted by any current path — see the diagnosis note for
/// `docs/investigations/deep_compound_tag_allowlist_diagnosis.md`.
#[test]
fn accepts_all_emitted_compound_tags() {
    // Each line exercises a distinct hyphenated compound tag from the
    // canonical 61-tag vocabulary. Stringing them into one Deep
    // program keeps the fixture compact while asserting every tag is
    // on the allowlist.
    let src = r#"(module {} demo
  (import-all {} other_mod)
  (defsig {} f
    (t-fn {}
      (t-ref {} (t-tensor {} (d-name {} batch) (d-var {} k) (d-lit {} 16) (t-prim {} f32)))
      (t-adt {} Box (t-var {} a))
      (t-tuple {} (t-unit {}) (t-prim {} f32))))
  (def {} f (params {} p r) (var {} p))
  (def {} use_match (params {} m)
    (match {} (var {} m)
      (arm {} (pat-ctor {} Some (pat-var {} x)) () (var {} x))
      (arm {} (pat-tuple {} (pat-lit {} 0) (pat-wild {})) () (var {} m))
      (arm {} (pat-record {} Rec (kv {} a (pat-as {} y (pat-wild {})))) () (var {} y))))
  (def {} use_tuple (params {} t) (tuple-get {} (var {} t) 0))
  (def {} use_update (params {} r) (record-update {} (var {} r) (kv {} a (lit {} 1))))
  (def {} use_effect (params {} k) (handle-effect {effect: Log} (var {} k) (var {} body))))
"#;
    let v = run(src);
    assert!(
        v.is_empty(),
        "expected zero violations for canonical compound-tag corpus, got: {v:?}"
    );
}

/// Sanity-check sibling: a deliberately user-defined hyphenated
/// identifier still triggers the lint. Guards against a future allowlist
/// expansion that accidentally widens to arbitrary hyphenated tokens.
#[test]
fn still_flags_user_defined_hyphenated_symbol_alongside_canonical_tags() {
    let src = r#"(def {type: (t-ref {} (t-prim {} f32))}
  my-bad-name (params {} x) (var {} x))
"#;
    let v = run(src);
    assert!(
        v.iter().any(|viol| viol.message.contains("my-bad-name")),
        "expected `my-bad-name` to be flagged, got: {v:?}"
    );
}
