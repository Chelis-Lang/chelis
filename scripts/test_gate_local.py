"""Unit tests for `gate.py --local` and the `--list` annotations (chelis#360).

Run via:
`uv run --managed-python --python 3.11 --no-project python -m unittest
scripts.test_gate_local` from the repo root.

What is locked here:

  (a) changed-crate derivation: canned `git diff --name-only` plus
      `git status --porcelain` output maps to owning workspace package
      names via each member's `Cargo.toml` `[package].name` -- the
      directory name is never assumed to be the package name;
  (b) paths outside every workspace member map to no crate, and an
      empty diff yields the explicit "no crate changes detected"
      message instead of silently running nothing;
  (c) the `--local` command list is exactly the static pre-push subset
      (workspace clippy, fmt --check, chelis lint --check .) plus one
      `cargo nextest run -p <crate>` per changed crate -- no workspace
      build, no workspace nextest;
  (d) `--list` annotates every canonical command as either in the
      `--local` subset or CI-owned, without changing the command list
      itself (the command-list lock stays in `scripts/test_gate.py`).

No test here runs cargo, nextest, git, or any real gate stage: git
output and the member->package mapping are injected as canned inputs,
mirroring how `scripts/test_gate.py` locks gate behavior without
executing it. The only filesystem reads are the repo's own manifests
and a synthesized temp workspace.
"""

import importlib.util
import io
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock


def _load_module():
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "gate_under_local_tests", here / "gate.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


gate = _load_module()
REPO_ROOT = Path(__file__).resolve().parent.parent

# Canned member-directory -> package-name mapping. `crates/legacy-dir`
# deliberately has a package name that differs from its directory name,
# and `tree-sitter-chelis` is a workspace member outside `crates/`.
CANNED_MEMBERS = {
    "crates/chelis-surf": "chelis-surf",
    "crates/chelis-cli": "chelis-cli",
    "crates/legacy-dir": "chelis-renamed",
    "tree-sitter-chelis": "tree-sitter-chelis",
}


class ChangedPathsFromGitTests(unittest.TestCase):
    def test_diff_and_status_paths_combine(self):
        diff = "crates/chelis-surf/src/lib.rs\nscripts/gate.py\n"
        status = " M crates/chelis-cli/src/main.rs\n?? notes.md\n"
        self.assertEqual(
            gate.changed_paths_from_git(diff, status),
            [
                "crates/chelis-surf/src/lib.rs",
                "scripts/gate.py",
                "crates/chelis-cli/src/main.rs",
                "notes.md",
            ],
        )

    def test_porcelain_rename_contributes_both_sides(self):
        status = "R  crates/chelis-surf/src/a.rs -> crates/chelis-cli/src/b.rs\n"
        self.assertEqual(
            gate.changed_paths_from_git("", status),
            [
                "crates/chelis-surf/src/a.rs",
                "crates/chelis-cli/src/b.rs",
            ],
        )

    def test_empty_outputs_yield_no_paths(self):
        self.assertEqual(gate.changed_paths_from_git("", ""), [])
        self.assertEqual(gate.changed_paths_from_git("\n", "\n"), [])


class ChangedCratesTests(unittest.TestCase):
    def test_paths_outside_any_member_map_to_no_crate(self):
        paths = [
            "scripts/gate.py",
            "docs/local_macos_environment.md",
            "AGENTS.md",
            "spec/01-nomenclature.md",
            "examples/mnist.ch",
        ]
        self.assertEqual(gate.changed_crates(paths, CANNED_MEMBERS), [])

    def test_multiple_crates_sorted_and_deduplicated(self):
        paths = [
            "crates/chelis-surf/src/lib.rs",
            "crates/chelis-cli/tests/integration.rs",
            "crates/chelis-surf/Cargo.toml",
            "scripts/gate.py",
        ]
        self.assertEqual(
            gate.changed_crates(paths, CANNED_MEMBERS),
            ["chelis-cli", "chelis-surf"],
        )

    def test_directory_name_is_not_assumed_to_be_package_name(self):
        # The mapping is keyed by directory; the result must be the
        # `[package].name`, which here differs from the directory name.
        paths = ["crates/legacy-dir/src/lib.rs"]
        self.assertEqual(
            gate.changed_crates(paths, CANNED_MEMBERS), ["chelis-renamed"]
        )

    def test_member_outside_crates_directory_is_mapped(self):
        paths = ["tree-sitter-chelis/grammar.js"]
        self.assertEqual(
            gate.changed_crates(paths, CANNED_MEMBERS),
            ["tree-sitter-chelis"],
        )

    def test_sibling_directory_prefix_does_not_match(self):
        # `crates/chelis-surf-extra/...` must not map to the
        # `crates/chelis-surf` member: the prefix match requires a `/`
        # boundary.
        paths = ["crates/chelis-surf-extra/src/lib.rs"]
        self.assertEqual(gate.changed_crates(paths, CANNED_MEMBERS), [])

    def test_empty_paths_yield_no_crates(self):
        self.assertEqual(gate.changed_crates([], CANNED_MEMBERS), [])


class WorkspaceMemberPackagesTests(unittest.TestCase):
    def test_package_name_read_from_member_manifest(self):
        # Synthesized workspace: one member whose directory name differs
        # from its package name, one member outside crates/, and one
        # member declared via a glob.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Cargo.toml").write_text(
                '[workspace]\n'
                'members = ["crates/dir-a", "weird-root", "globbed/*"]\n'
            )
            for member_dir, package in (
                ("crates/dir-a", "package-a-renamed"),
                ("weird-root", "weird-package"),
                ("globbed/g1", "g-one"),
            ):
                d = root / member_dir
                d.mkdir(parents=True)
                (d / "Cargo.toml").write_text(
                    f'[package]\nname = "{package}"\n'
                )
            self.assertEqual(
                gate.workspace_member_packages(root),
                {
                    "crates/dir-a": "package-a-renamed",
                    "weird-root": "weird-package",
                    "globbed/g1": "g-one",
                },
            )

    def test_real_repo_mapping_covers_known_members(self):
        # Disk-read only (no subprocess): the real workspace manifest
        # must resolve, and known members must map to their package
        # names, including the member that lives outside crates/.
        packages = gate.workspace_member_packages()
        self.assertEqual(packages.get("crates/chelis-cli"), "chelis-cli")
        self.assertEqual(
            packages.get("tree-sitter-chelis"), "tree-sitter-chelis"
        )
        self.assertNotIn("", packages)


class LocalCommandListTests(unittest.TestCase):
    def test_static_subset_has_the_exact_compile_time_contracts(self):
        # `cargo nextest` does not execute doctests. The static subset
        # drives the chelis#731 `ErrorWitness` contracts, the compiler
        # pipeline artifact contracts, the raw-checkpoint fixture, the
        # two cheap pipeline-core boundary guards (dependency + no_std doc),
        # the canonical chelis-std generated-artifact currency check, and the
        # chelis#908 unrepresentable-domain oracle.
        # Assert the exact list so no pre-push stage disappears silently.
        rendered = [gate.render(c) for c in gate.local_command_list([])]
        self.assertEqual(
            rendered,
            [
                "cargo clippy --workspace --all-targets -- -D warnings",
                "cargo clippy --workspace --all-targets --features "
                "chelis-backend-c/sleef,"
                "chelis-e2e/hip-local-gpu,"
                "chelis-prove/clarabel,"
                "chelis-python/extension-module,"
                "chelis-runtime/ownership-ledger,"
                "chelis-types/checkpoint-compile-probe,"
                "chelis-types/generalize-sweep-oracle,"
                "chelis-types/hash-order-compile-probe -- -D warnings",
                "cargo clippy --workspace --all-targets --no-default-features "
                "-- -D warnings",
                "cargo fmt --all -- --check",
                "cargo run -p chelis-cli --bin chelis --quiet -- "
                "lint --check .",
                "<managed-python> scripts/regenerate_chelis_std_bundle.py "
                "--debug --check",
                "cargo test -p chelis-types --doc",
                "cargo test -p chelis-compiler-api --doc",
                "cargo test -p chelis-pipeline-core --doc",
                "<managed-python> scripts/check_checkpoint_compile_fail.py",
                "<managed-python> scripts/check_hash_order_compile_fail.py",
                "<managed-python> scripts/check_configuration_closure.py",
                "<managed-python> scripts/pipeline_core_dependency_guard.py",
                "<managed-python> scripts/pipeline_core_documentation_guard.py",
                "<managed-python> scripts/unrepresentable_domain_oracle.py",
            ],
        )

    def test_appends_one_nextest_run_per_changed_crate(self):
        commands = gate.local_command_list(["chelis-cli", "chelis-surf"])
        self.assertEqual(commands[: len(gate.LOCAL_STATIC_COMMANDS)],
                         gate.LOCAL_STATIC_COMMANDS)
        self.assertEqual(
            commands[len(gate.LOCAL_STATIC_COMMANDS):],
            [
                [
                    "cargo",
                    "nextest",
                    "run",
                    "-p",
                    "chelis-cli",
                    "--no-fail-fast",
                ],
                [
                    "cargo",
                    "nextest",
                    "run",
                    "-p",
                    "chelis-surf",
                    "--no-fail-fast",
                ],
            ],
        )

    def test_excludes_workspace_build_and_workspace_nextest(self):
        rendered = [
            gate.render(c)
            for c in gate.local_command_list(["chelis-surf"])
        ]
        self.assertNotIn(gate.render(gate.BUILD_WORKSPACE), rendered)
        self.assertNotIn(gate.render(gate.NEXTEST_WORKSPACE), rendered)
        self.assertTrue(rendered[-1].endswith("--no-fail-fast"))


class ListAnnotationTests(unittest.TestCase):
    def _list_lines(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = gate.main(["--list"])
        self.assertEqual(rc, 0)
        return buf.getvalue().strip().splitlines()

    def test_every_command_line_is_annotated(self):
        lines = self._list_lines()
        command_lines = [ln for ln in lines if not ln.startswith("#")]
        self.assertEqual(len(command_lines), len(gate.full_command_list()))
        annotations = {}
        for line in command_lines:
            self.assertIn("  # ", line, f"unannotated line: {line!r}")
            command, annotation = line.split("  # ", 1)
            self.assertIn(
                annotation,
                (
                    gate.LOCAL_ANNOTATION,
                    gate.CI_OWNED_ANNOTATION,
                    gate.FULL_GATE_SPLIT_ANNOTATION,
                ),
            )
            annotations[command] = annotation
        # The local subset: clippy, fmt, chelis lint. The standalone workspace
        # build is intentionally absent: clippy already compiles all targets,
        # and the full gate's default-profile workspace suite is covered in CI
        # by the split workspace and dtype-oracle jobs.
        self.assertEqual(
            annotations[gate.render(gate.CLIPPY_WORKSPACE)],
            gate.LOCAL_ANNOTATION,
        )
        self.assertEqual(
            annotations[gate.render(gate.FMT_CHECK)],
            gate.LOCAL_ANNOTATION,
        )
        self.assertEqual(
            annotations[gate.render(gate.CHELIS_LINT_CHECK)],
            gate.LOCAL_ANNOTATION,
        )
        self.assertNotIn(gate.render(gate.BUILD_WORKSPACE), annotations)
        self.assertEqual(
            annotations[gate.render(gate.NEXTEST_WORKSPACE)],
            gate.FULL_GATE_SPLIT_ANNOTATION,
        )

    def test_dynamic_local_stage_is_documented_in_list_output(self):
        lines = self._list_lines()
        self.assertEqual(lines[-1], gate.LOCAL_DYNAMIC_NOTE)
        self.assertIn("cargo nextest run -p <crate>", lines[-1])


    def test_quoted_diff_paths_still_derive_their_crate(self):
        # core.quotePath wraps non-ASCII filenames in quotes in
        # `git diff --name-only` output; the prefix match must still
        # see the crate dir (PR #362 review finding 1: this was the one
        # path that erred toward silent exclusion).
        diff_output = '"crates/legacy-dir/tests/caf\\303\\251_corpus.rs"\n'
        paths = gate.changed_paths_from_git(diff_output, "")
        self.assertEqual(paths, ["crates/legacy-dir/tests/caf\\303\\251_corpus.rs"])
        crates = gate.changed_crates(paths, {"crates/legacy-dir": "chelis-renamed"})
        self.assertEqual(crates, ["chelis-renamed"])


class LocalMainTests(unittest.TestCase):
    """Drive `gate.main(["--local"])` end to end with canned git output
    and a recorded `run_commands`, so no subprocess ever runs."""

    def _run_local(self, diff_output, status_output, run_rc=0):
        recorded = []

        def fake_git_output(args):
            if args[0] == "diff":
                return diff_output
            if args[0] == "status":
                return status_output
            raise AssertionError(f"unexpected git invocation: {args}")

        def fake_run_commands(commands, **_kwargs):
            recorded.extend(commands)
            return run_rc

        buf = io.StringIO()
        with mock.patch.object(gate, "_git_output", fake_git_output), \
                mock.patch.object(
                    gate,
                    "workspace_member_packages",
                    lambda: dict(CANNED_MEMBERS),
                ), \
                mock.patch.object(gate, "run_commands", fake_run_commands), \
                redirect_stdout(buf):
            rc = gate.main(["--local"])
        return rc, recorded, buf.getvalue()

    def test_runs_static_subset_plus_changed_crate_suites(self):
        rc, recorded, out = self._run_local(
            "crates/chelis-surf/src/lib.rs\nscripts/gate.py\n",
            " M crates/chelis-cli/src/main.rs\n?? notes.md\n",
        )
        self.assertEqual(rc, 0)
        self.assertIn(
            "gate --local: changed crates vs origin/main: "
            "chelis-cli, chelis-surf",
            out,
        )
        self.assertEqual(
            recorded,
            gate.local_command_list(["chelis-cli", "chelis-surf"]),
        )

    def test_no_crate_changes_is_reported_not_silent(self):
        rc, recorded, out = self._run_local("", "?? todo_humans.md\n")
        self.assertEqual(rc, 0)
        self.assertIn("no crate changes detected", out)
        self.assertIn("CI-owned", out)
        self.assertEqual(recorded, gate.local_command_list([]))

    def test_uncommitted_work_alone_triggers_a_crate_suite(self):
        # `git status --porcelain` is consulted in addition to the
        # committed diff, so dirty-tree-only changes still count.
        rc, recorded, out = self._run_local(
            "", " M crates/chelis-surf/src/lib.rs\n"
        )
        self.assertEqual(rc, 0)
        self.assertIn("chelis-surf", out)
        self.assertEqual(recorded, gate.local_command_list(["chelis-surf"]))

    def test_command_failure_exit_code_propagates(self):
        rc, _, _ = self._run_local(
            "crates/chelis-surf/src/lib.rs\n", "", run_rc=7
        )
        self.assertEqual(rc, 7)

    def test_git_failure_is_reported_and_nonzero(self):
        def failing_git_output(args):
            raise subprocess.CalledProcessError(
                128, ["git", *args], stderr="fatal: bad revision"
            )

        err = io.StringIO()
        with mock.patch.object(gate, "_git_output", failing_git_output), \
                mock.patch.object(
                    gate,
                    "run_commands",
                    lambda commands: self.fail(
                        "run_commands must not run when git fails"
                    ),
                ), \
                redirect_stderr(err):
            rc = gate.main(["--local"])
        self.assertEqual(rc, 128)
        self.assertIn("git failed", err.getvalue())
        self.assertIn("fatal: bad revision", err.getvalue())

    def test_local_cannot_be_combined_with_a_stage(self):
        err = io.StringIO()
        with redirect_stderr(err), self.assertRaises(SystemExit):
            gate.main(["lint-and-unit", "--local"])

    def test_local_cannot_be_combined_with_list(self):
        err = io.StringIO()
        with redirect_stderr(err), self.assertRaises(SystemExit):
            gate.main(["--list", "--local"])


if __name__ == "__main__":
    unittest.main()
