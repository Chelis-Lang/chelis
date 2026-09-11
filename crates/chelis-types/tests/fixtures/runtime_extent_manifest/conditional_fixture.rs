// Fixture source for `runtime_extent_target_manifest.rs`'s two
// conditional-compilation rejection cases. Never compiled, for the reason
// `target_fixture.rs` gives.
//
// Both shapes below are readable only by evaluating a condition this parser
// does not have: whether the test exists in the built binary, and whether it
// carries `#[ignore]` when it does.

#[test]
fn always_present() {}

#[cfg(feature = "a-feature-this-crate-does-not-declare")]
#[test]
fn present_only_under_a_feature() {}

#[test]
#[cfg_attr(target_os = "macos", ignore)]
fn ignored_only_on_one_platform() {}
