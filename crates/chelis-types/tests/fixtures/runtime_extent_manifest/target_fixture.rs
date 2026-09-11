// Fixture source for `runtime_extent_target_manifest.rs`. It is never
// compiled: nothing declares it as a module and it is not a direct child of
// `tests/`, so Cargo does not pick it up as a target. It exists so the
// tripwire's four rejection cases run against a real parse of a real file
// rather than against a mocked inventory.

#[test]
fn alpha_is_selected() {}

#[test]
fn beta_is_selected() {}

#[test]
#[ignore]
fn gamma_is_a_manual_gate() {}

#[test]
fn selector_prefix_delta() {}

#[path = "included_fixture.rs"]
mod included;
