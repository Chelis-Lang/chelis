//! chelis#486: prefix-namespace lint now explains WHY it flagged the name
//! and suggests a concrete rename.

use chelis_lint::rules::prefix_namespace::PrefixNamespace;
use chelis_lint::{Context, Rule, Surface};
use std::path::Path;

#[test]
fn prefix_namespace_suggests_concrete_rename() {
    let src = "module Whale.Probability\ndef wp_compute(x: f32) = todo\ndef wp_normalize(x: f32) = todo\n";
    let path = Path::new("test.ch");
    let ctx = Context {
        root: Path::new("/"),
        path,
        source: Some(src),
        surface: Surface::SurfSource,
    };
    let violations = PrefixNamespace.check(&ctx);
    assert_eq!(violations.len(), 2, "got: {violations:?}");
    // Check that the message suggests a concrete rename
    let msg = &violations[0].message;
    assert!(
        msg.contains("Consider renaming to"),
        "message should suggest a rename; got: {msg}"
    );
    assert!(
        msg.contains("module") && msg.contains("already provides namespace"),
        "message should explain the module provides namespace; got: {msg}"
    );
}

#[test]
fn prefix_namespace_rename_strips_prefix() {
    let src = "module Demo.Widget\ndef zz_alpha(x: f32) = todo\ndef zz_beta(x: f32) = todo\n";
    let path = Path::new("test.ch");
    let ctx = Context {
        root: Path::new("/"),
        path,
        source: Some(src),
        surface: Surface::SurfSource,
    };
    let violations = PrefixNamespace.check(&ctx);
    assert!(!violations.is_empty());
    // The first violation for zz_alpha should suggest renaming to 'alpha'
    let msg = &violations[0].message;
    assert!(
        msg.contains("`alpha`"),
        "should suggest stripped name 'alpha'; got: {msg}"
    );
}
