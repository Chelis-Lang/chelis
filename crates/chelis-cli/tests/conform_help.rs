//! Public help text for conformance-owned agent surfaces.

use assert_cmd::Command;
use predicates::prelude::*;

#[test]
fn conform_entry_points_name_both_managed_command_trees() {
    for args in [
        &["reef", "conform", "init", "--help"][..],
        &["reef", "conform", "sync", "--help"][..],
        &["reef", "conform", "bump", "--help"][..],
        &["reef", "setup", "--help"][..],
    ] {
        Command::cargo_bin("chelis")
            .expect("binary")
            .args(args)
            .assert()
            .success()
            .stdout(predicate::str::contains(".claude/commands/"))
            .stdout(predicate::str::contains(".codex/commands/"));
    }
}

#[test]
fn sync_help_no_longer_claims_only_agent_skills_are_touched() {
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["reef", "conform", "sync", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Touches only").not())
        .stdout(predicate::str::contains("managed regions and `agent-skills/`").not());
}
