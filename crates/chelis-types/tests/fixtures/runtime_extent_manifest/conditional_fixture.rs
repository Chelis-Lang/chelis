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

// The module-level arm: the gate is on the module, not on the test, so the
// refusal has to be inherited down the traversal. `#[cfg(test)] mod tests`
// must NOT be refused this way, which the `included_fixture.rs` case covers.
#[cfg(feature = "a-feature-this-crate-does-not-declare")]
mod gated_module {
    #[test]
    fn present_only_under_its_module_feature() {}
}
