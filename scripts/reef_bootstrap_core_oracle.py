#!/usr/bin/env python3
"""Check portable graph policy and real host bootstrap parity.

Run from the repository root:
.venv/bin/python scripts/reef_bootstrap_core_oracle.py
Target compilation is not device execution. Host execution is not firmware acceptance.
"""

from pathlib import Path
import sys
import tempfile

from no_std_core_support import (
    ROOT, TARGET, OracleFailure, compile_rejected, consumer, portable, run,
)

CRATE = "chelis-reef-bootstrap-core"
FORBIDDEN = ("chelis-reef", "chelis-types", "chelis-shell", "chelis-pipeline-core")
CORE_TESTS = (
    'stable_depth_first_orders',
    'occurrences_and_last_target_remain_distinct',
    'conflicts_sort_unique_versions_before_other_checks',
    'exact_missing_edges_use_node_then_edge_order_before_cycles',
    'cycle_paths_close_and_preserve_identity_suffixes',
    'long_chain_uses_explicit_traversal_state',
)

HOST_TESTS = (
    'exact_graph_diagnostics_and_no_install',
    'repeated_installs_select_original_releases_and_preserve_fetch_counts',
    'host_projection_keeps_runtime_path_and_manifest_rules',
    'artifact_failure_preserves_prior_install_and_stops_later_installs',
)

CLI_TESTS = (
    'bootstrap_core_preserves_conflict_and_requirement_precedence',
    'bootstrap_runtime_requirement_forms_match_bundled',
    'bootstrap_runtime_requirement_mismatches_rejected_before_install',
    'bootstrap_invalid_runtime_requirements_use_manifest_validation',
    'bootstrap_peer_requirement_forms_match_selected_version',
    'bootstrap_incompatible_peer_requirement_rejected_before_install',
    'bootstrap_invalid_peer_requirements_use_manifest_validation',
    'bootstrap_typed_peer_edges_preserve_cycle_rejection',
    'bootstrap_defaults_resolve_joint_graph_without_project_writes',
    'bootstrap_defaults_same_binary_accepts_later_published_cohort',
    'bootstrap_defaults_permutation_and_repeat_preserve_registry_and_output',
    'bootstrap_defaults_use_canonical_remote_candidates_not_cached_fork',
    'bootstrap_defaults_public_without_token_or_gh',
    'bootstrap_defaults_private_errors_remain_actionable',
    'bootstrap_defaults_reject_invalid_runtime_missing_and_cyclic_graphs',
    'bootstrap_defaults_verify_selected_shells_before_registry_publication',
    'bootstrap_defaults_rebind_identical_cached_bytes_to_observed_canonical_origin',
    'bootstrap_defaults_existing_identity_conflicts_preserve_registry_metadata',
    'bootstrap_defaults_keep_the_shared_release_page_budget',
    'bootstrap_defaults_charge_tags_only_for_window_releases',
    'bootstrap_explicit_selectors_do_not_discover_a_missing_peer',
    'phaseA_item7_bootstrap_oracle',
    'phaseA_item7_three_shell_cycle_named',
    'phaseA_item7_self_loop_is_one_cycle',
    'phaseA_item7_manifest_read_failure_is_typed',
    'phaseA_item7_duplicate_package_versions_rejected',
    'phaseA_item7_empty_input_and_empty_default_errors_nothing_to_install',
    'phaseA_item7_auth_failure_aborts_before_first_install',
    'phaseA_item7_public_bootstrap_without_token',
    'phaseA_item7_idempotence_byte_identical_state',
    'phaseA_item7_order_independence_of_input_list',
    'phaseA_item7_help_lists_bootstrap_flag',
)

TARGET_SOURCE = r'''
#![no_std]
extern crate alloc;
use alloc::vec::Vec;
use chelis_reef_bootstrap_core::{validate_and_order, BootstrapGraphError, BootstrapNode};
pub fn checked(nodes: &[BootstrapNode]) -> Result<Vec<usize>, BootstrapGraphError> {
    validate_and_order(nodes)
}
'''
HOST_SOURCE = r'''
use chelis_reef_bootstrap_core::{validate_and_order, BootstrapGraphError, BootstrapIdentity, BootstrapNode};
#[global_allocator]
static ALLOCATOR: std::alloc::System = std::alloc::System;
fn id(name: &str) -> BootstrapIdentity {
    BootstrapIdentity { name: name.into(), version: "1".into() }
}
fn node(name: &str, dependencies: &[&str]) -> BootstrapNode {
    BootstrapNode { identity: id(name), dependencies: dependencies.iter().map(|name| id(name)).collect() }
}
fn main() {
    let nodes = [node("A", &["B", "C"]), node("B", &["D"]), node("C", &["D"]), node("D", &[])];
    assert_eq!(validate_and_order(&nodes).unwrap(), vec![3, 1, 2, 0]);
    assert_eq!(validate_and_order(&[node("A", &[]), node("C", &["A"]), node("A", &[])]).unwrap(), vec![0, 2, 1]);
    assert_eq!(validate_and_order(&[node("A", &["missing"])]), Err(BootstrapGraphError::MissingDependency { dependent: id("A"), missing: id("missing") }));
    assert_eq!(validate_and_order(&[node("A", &["A"])]), Err(BootstrapGraphError::Cycle { cycle: vec![id("A"), id("A")] }));
    let mut conflict = node("A", &[]);
    conflict.identity.version = "2".into();
    assert_eq!(validate_and_order(&[node("A", &[]), conflict]), Err(BootstrapGraphError::DuplicateVersion { package: "A".into(), versions: vec!["1".into(), "2".into()] }));
}
'''


def host_and_core_suites():
    run(["python3", "--version"])
    run([sys.executable, "scripts/test_no_std_core_support.py"])
    run(["cargo", "nextest", "run", "-p", CRATE, "--no-fail-fast"], required_tests=CORE_TESTS)
    run(["cargo", "test", "-p", CRATE, "--doc"])
    run(["cargo", "nextest", "run", "-p", "chelis-reef", "--test", "bootstrap_core", "--no-fail-fast"], required_tests=HOST_TESTS)
    run(["cargo", "nextest", "run", "-p", "chelis-reef", "--lib",
         "-E", "test(=tests::default_bootstrap_list_names_only_public_shells)", "--no-fail-fast"],
        required_tests=("tests::default_bootstrap_list_names_only_public_shells",))
    run(["cargo", "nextest", "run", "-p", "chelis-cli", "--test", "reef_install_bootstrap",
         "-E", "not test(=phaseA_item7_real_bootstrap_manual_gate)", "--no-fail-fast"], required_tests=CLI_TESTS)


def target_consumers():
    run(["cargo", "build", "-p", CRATE, "--lib", "--no-default-features", "--target", TARGET])
    for defaults in (False, True):
        packages = portable(CRATE, TARGET_SOURCE, default_features=defaults, forbidden=FORBIDDEN)
        if packages != {CRATE, "chelis-no-std-consumer"}:
            raise OracleFailure(f"unexpected portable graph: {sorted(packages)}")


def host_consumer():
    for defaults in (False, True):
        consumer(CRATE, HOST_SOURCE, default_features=defaults)


def rejected_api():
    compile_rejected(CRATE, "pub fn bypass() { chelis_reef_bootstrap_core::topo_sort_bootstrap(&[]); }", expected="cannot find function `topo_sort_bootstrap`")
    compile_rejected(CRATE, "pub fn bypass() { let _ = chelis_reef_bootstrap_core::validate_and_order(&[], true); }", expected="takes 1 argument")


def negative_dependency():
    with tempfile.TemporaryDirectory(prefix="reef-bootstrap-negative-") as temporary:
        root = Path(temporary)
        for name, features, expected in (
            ("reef-bootstrap-host-only", "", "forbidden target dependency: reef-bootstrap-host-only"),
            ("reef-bootstrap-std-leak", '[features]\ndefault = ["std"]\nstd = []\n',
             "target std feature: reef-bootstrap-std-leak"),
        ):
            fixture = root / name
            (fixture / "src").mkdir(parents=True)
            (fixture / "Cargo.toml").write_text(
                f'[package]\nname = "{name}"\nversion = "0.0.0"\nedition = "2021"\n' + features,
                encoding="utf-8",
            )
            (fixture / "src/lib.rs").write_text(
                "pub fn host() { std::thread::yield_now(); }\n", encoding="utf-8"
            )
            try:
                portable(CRATE, TARGET_SOURCE,
                         forbidden=(name,) if not features else (),
                         dependencies=f'{name} = {{ path = "{fixture.as_posix()}" }}\n')
            except OracleFailure as error:
                if str(error) != expected:
                    raise
            else:
                raise OracleFailure(f"negative dependency accepted: {name}")


def run_oracle():
    host_and_core_suites()
    target_consumers()
    host_consumer()
    rejected_api()
    negative_dependency()


def main():
    try:
        run_oracle()
    except (OracleFailure, OSError) as error:
        print(f"reef bootstrap core oracle: FAIL: {error}", file=sys.stderr)
        return 1
    print("reef bootstrap core oracle: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
