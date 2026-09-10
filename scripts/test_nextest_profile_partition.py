"""Adversarial coverage for the heavy-e2e nextest profile split.

Run via:
`uv run --managed-python --python 3.11 --no-project python -m unittest
scripts.test_nextest_profile_partition` from the repo root.

PR #126 (refined by #127) split the heavyweight end-to-end suite off the
per-PR integration gate. `.config/nextest.toml` carries three profiles:

  - `default` excludes an explicitly-named heavy-e2e set;
  - `ci` excludes that same set plus every complete test binary named by a
    Phase 0-3 oracle `--test` argument;
  - `nightly` carries the EXACT SAME set as a positive filter, and the
    profile remains a manual heavy selection. The hosted Linux Extended
    Validation workflow runs the full workspace with `--ignore-default-filter`.

This file locks the original workspace/nightly split plus the delegation
contract for binaries named by oracle `--test` arguments. Selector-based
oracle legs still overlap the workspace lane, including `-E` expressions with
whole-binary arms, so the dtype oracle is not a third disjoint profile.

Three tiers of check:

  - `FilterTextTests` is a fast, no-compile lock on the *text* of the
    three filter blocks: `ci` may add only binaries named by oracle `--test`
    arguments to the default exclusion, and the `nightly` positive filter must
    equal the negated inner set of the default exclusion block.
  - `ProfilePartitionTests` is the real set-math oracle: it runs
    `cargo nextest list` for the `ci` and `nightly` profiles plus the
    full unfiltered list and asserts the partition. It is skipped when
    `cargo`/`cargo nextest` is unavailable, and is `slow`-tolerant
    (listing compiles test binaries on a cold tree).
  - `GeneralizationPartitionTests` lists the explicit generalization lane,
    `--features chelis-types/generalize-sweep-oracle`, and asserts that the
    two nightly-owned contention cases stay out of it.

The last two classes list different compiled configurations, and that is
why nightly CI runs them in different jobs. `ProfilePartitionTests` runs in
`full-workspace`, whose default-feature build is warm; `GeneralizationPartitionTests`
runs on generalization shard 1, whose feature-enabled build is warm. Listing
the generalization lane on the workspace shard recompiled the workspace under
a second feature set and cost 4.4 hosted minutes per run. Both classes also
run wherever the whole module is invoked, so a developer machine still sees
the complete oracle.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
NEXTEST_TOML = REPO_ROOT / ".config" / "nextest.toml"
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import dtype_oracle_manifest  # noqa: E402


def _filter_blocks() -> list[str]:
    """Return the raw bodies of every `default-filter = '''...'''` block
    in `.config/nextest.toml`, in file order (default, ci, nightly)."""
    text = NEXTEST_TOML.read_text()
    return re.findall(r"default-filter = '''(.*?)'''", text, re.DOTALL)


def _norm(s: str) -> str:
    """Collapse all whitespace runs to single spaces and strip."""
    return re.sub(r"\s+", " ", s).strip()


def _negative_filter_inner(block: str) -> str:
    match = re.search(r"not\s*\((.*)\)\s*$", block, re.DOTALL)
    if match is None:
        raise AssertionError("filter is not in the expected `not ( <set> )` shape")
    return match.group(1)


def _required_phase3_commands() -> tuple[tuple[str, ...], ...]:
    """Resolve every selection flattened into the required Phase 3 union."""
    return tuple(
        leg.argv
        for leg in dtype_oracle_manifest.owned_nextest_legs(sys.executable)
    )


def _oracle_selected_test_binaries() -> set[str]:
    """Return complete binaries explicitly named by oracle `--test` arguments."""
    selected: set[str] = set()
    for argv in _required_phase3_commands():
        if "-p" not in argv:
            continue
        package = argv[argv.index("-p") + 1]
        for index, argument in enumerate(argv[:-1]):
            if argument == "--test":
                selected.add(f"{package}::{argv[index + 1]}")
    return selected


ORACLE_OWNED_BINARY_IDS = frozenset(_oracle_selected_test_binaries())
ORACLE_OWNED_FILTERS = tuple(
    f"binary_id(/^{binary_id}$/)"
    for binary_id in sorted(ORACLE_OWNED_BINARY_IDS)
)
NIGHTLY_RECURSIVE_TEST = (
    "chelis-cli::issue_1293_redteam_round4::"
    "recursive_list_tuple_and_adt_cotangents_match_in_eval_and_c"
)
NIGHTLY_CACHE_CONCURRENCY_TEST = (
    "chelis-cli::stdlib_typecheck_cache_concurrency::"
    "parallel_cold_cache_invocations_all_succeed_identically"
)
NIGHTLY_RECURSIVE_SELECTOR = (
    "binary_id(/^chelis-cli::issue_1293_redteam_round4$/) & "
    "test(/^recursive_list_tuple_and_adt_cotangents_match_in_eval_and_c$/)"
)
NIGHTLY_CACHE_CONCURRENCY_SELECTOR = (
    "binary_id(/^chelis-cli::stdlib_typecheck_cache_concurrency$/)"
)
GENERALIZATION_PR_FILTER = (
    f"not ({NIGHTLY_CACHE_CONCURRENCY_SELECTOR} | "
    f"({NIGHTLY_RECURSIVE_SELECTOR}))"
)
CONTENDED_DEADLINE_RETRY_SELECTOR = (
    "binary_id(/^chelis-cli::test_suite_timeout$/) & "
    "test(/^normal_output_forwarding_is_part_of_whole_command_deadline$/)"
)
SHARED_RUSTDOC_GROUP = "capacity-rustdoc-verifiers"
SHARED_RUSTDOC_OWNER_TESTS = (
    "chelis-compiler-api::capacity_census_wire::"
    "wire_schema_numeric_fields_match_the_reviewed_baseline",
    "chelis-python::capacity_census_bindings::"
    "registered_pyfunctions_match_the_reviewed_rustdoc_signatures",
)


def _cargo_environment(
    environ: dict[str, str] | None = None,
) -> dict[str, str]:
    environment = dict(os.environ if environ is None else environ)
    environment.setdefault("PYO3_PYTHON", sys.executable)
    environment.setdefault("CARGO_TARGET_DIR", str(REPO_ROOT / "target"))
    environment["CARGO_HUSKY_DONT_INSTALL_HOOKS"] = "1"
    return environment


class FilterTextTests(unittest.TestCase):
    """No-compile lock on the three filter blocks' text."""

    def test_nextest_toml_exists(self):
        self.assertTrue(NEXTEST_TOML.is_file(), f"missing {NEXTEST_TOML}")

    def test_exactly_three_filter_blocks(self):
        blocks = _filter_blocks()
        self.assertEqual(
            len(blocks),
            3,
            "expected exactly three default-filter blocks "
            "(default, ci, nightly)",
        )

    def test_every_profile_runs_to_completion_after_failures(self):
        config = tomllib.loads(NEXTEST_TOML.read_text())
        for profile in ("default", "ci", "ci-full", "nightly"):
            with self.subTest(profile=profile):
                self.assertIs(
                    config["profile"][profile].get("fail-fast"),
                    False,
                    f"nextest profile {profile!r} hides later failures",
                )

    def test_full_ci_telemetry_profile_inherits_local_scope_and_writes_junit(self):
        config = tomllib.loads(NEXTEST_TOML.read_text())
        profile = config["profile"]["ci-full"]
        self.assertEqual(profile.get("inherits"), "default")
        self.assertEqual(profile.get("junit", {}).get("path"), "junit.xml")
        self.assertNotIn(
            "default-filter",
            profile,
            "ci-full must inherit the ordinary profile unless a command "
            "explicitly passes --ignore-default-filter",
        )

    def test_deadline_probe_has_no_retry_cost(self):
        config = tomllib.loads(NEXTEST_TOML.read_text())
        overrides = config["profile"]["default"].get("overrides", [])
        matching = [
            override
            for override in overrides
            if override.get("filter") == CONTENDED_DEADLINE_RETRY_SELECTOR
        ]
        self.assertEqual(
            matching,
            [],
            "the deadline test accepts both fail-closed timeout diagnostics; "
            "retrying it only repeats a deterministic contract probe",
        )

    def test_contention_sensitive_recursive_parity_case_is_nightly_owned(self):
        default_block, _ci, nightly_block = _filter_blocks()
        self.assertIn(
            NIGHTLY_RECURSIVE_SELECTOR,
            _norm(_negative_filter_inner(default_block)),
        )
        self.assertIn(NIGHTLY_RECURSIVE_SELECTOR, _norm(nightly_block))

    def test_ci_adds_only_oracle_owned_binaries_to_default_exclusion(self):
        default_block, ci_block, _nightly = _filter_blocks()
        expected = _norm(_negative_filter_inner(default_block))
        expected += " + " + " + ".join(ORACLE_OWNED_FILTERS)
        self.assertEqual(
            _norm(_negative_filter_inner(ci_block)),
            expected,
            "the `ci` filter must differ from `default` only by the exact "
            "complete binaries named by dtype oracle `--test` arguments",
        )

    def test_ci_only_exclusions_are_executed_by_the_dtype_oracle(self):
        self.assertEqual(
            ORACLE_OWNED_BINARY_IDS,
            _oracle_selected_test_binaries(),
            "an excluded binary is not selected by the dtype oracle",
        )

    def test_inherited_phase0_and_observation_binaries_are_delegated(self):
        self.assertTrue(
            {
                "chelis-cli::domain_checker",
                "chelis-cli::parity",
                "chelis-e2e::eval_agreement",
                "chelis-cli::issue_687_rejected_cells_corpus",
                "chelis-types::agreement_tolerance",
            }
            <= ORACLE_OWNED_BINARY_IDS
        )

    def test_nightly_filter_is_the_negated_exclusion_set(self):
        # `default`/`ci` exclude `not ( <set> )`; `nightly` must select
        # exactly `<set>`. If these drift, a test lands in neither
        # profile (dropped coverage) or both (double-run).
        default_block, _ci, nightly_block = _filter_blocks()
        self.assertEqual(
            _norm(_negative_filter_inner(default_block)),
            _norm(nightly_block),
            "the `nightly` positive filter does not equal the negated "
            "inner set of the `default` exclusion filter; a test "
            "now falls into neither profile or both",
        )

    def test_cargo_uses_the_current_managed_python(self):
        environment = _cargo_environment({"PATH": "/usr/bin"})
        self.assertEqual(environment["PYO3_PYTHON"], sys.executable)
        self.assertEqual(
            environment["CARGO_TARGET_DIR"], str(REPO_ROOT / "target")
        )
        self.assertEqual(
            environment["CARGO_HUSKY_DONT_INSTALL_HOOKS"], "1"
        )

    def test_explicit_cargo_python_remains_authoritative(self):
        environment = _cargo_environment(
            {"PYO3_PYTHON": "/configured/python"}
        )
        self.assertEqual(
            environment["PYO3_PYTHON"], "/configured/python"
        )


def _have_nextest() -> bool:
    if shutil.which("cargo") is None:
        return False
    try:
        out = subprocess.run(
            ["cargo", "nextest", "--version"],
            cwd=REPO_ROOT,
            env=_cargo_environment(),
            capture_output=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return out.returncode == 0


def _list_profile(profile: str | None) -> dict[str, tuple[str, bool]]:
    """Run `cargo nextest list` for `profile` (or the implicit default
    when `None`) and return `binary_id::test -> (filter_status,
    ignored)`.

    nextest's `list --message-format json` annotates each testcase with
    `filter-match.status` ("matches" / "mismatch") relative to the
    profile's `default-filter`, and an `ignored` flag.
    """
    cmd = ["cargo", "nextest", "list", "--workspace", "--message-format", "json"]
    if profile is not None:
        cmd += ["--profile", profile]
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=_cargo_environment(),
        capture_output=True,
        text=True,
        timeout=900,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"`{' '.join(cmd)}` failed (exit {result.returncode}): "
            f"{result.stderr[-2000:]}"
        )
    data = json.loads(result.stdout)
    out: dict[str, tuple[str, bool]] = {}
    for binary_id, suite in data.get("rust-suites", {}).items():
        for test_name, info in suite.get("testcases", {}).items():
            key = f"{binary_id}::{test_name}"
            status = info.get("filter-match", {}).get("status")
            out[key] = (status, bool(info.get("ignored")))
    return out


def _list_filterset(filterset: str) -> dict[str, tuple[str, bool]]:
    """List one unfiltered dtype-owner selection through nextest itself."""
    cmd = [
        "cargo",
        "nextest",
        "list",
        "--workspace",
        "--ignore-default-filter",
        "--message-format",
        "json",
        "-E",
        filterset,
    ]
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=_cargo_environment(),
        capture_output=True,
        text=True,
        timeout=900,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"dtype filterset list failed (exit {result.returncode}): "
            f"{result.stderr[-2000:]}"
        )
    data = json.loads(result.stdout)
    out: dict[str, tuple[str, bool]] = {}
    for binary_id, suite in data.get("rust-suites", {}).items():
        for test_name, info in suite.get("testcases", {}).items():
            key = f"{binary_id}::{test_name}"
            status = info.get("filter-match", {}).get("status")
            out[key] = (status, bool(info.get("ignored")))
    return out


def _list_generalization_pr() -> dict[str, tuple[str, bool]]:
    """List the exact explicit-filter scope used by the PR generalization lane."""
    cmd = [
        "cargo",
        "nextest",
        "list",
        "--workspace",
        "--profile",
        "ci-full",
        "--ignore-default-filter",
        "--features",
        "chelis-types/generalize-sweep-oracle",
        "--message-format",
        "json",
        "-E",
        GENERALIZATION_PR_FILTER,
    ]
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=_cargo_environment(),
        capture_output=True,
        text=True,
        timeout=900,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"generalization PR list failed (exit {result.returncode}): "
            f"{result.stderr[-2000:]}"
        )
    data = json.loads(result.stdout)
    out: dict[str, tuple[str, bool]] = {}
    for binary_id, suite in data.get("rust-suites", {}).items():
        for test_name, info in suite.get("testcases", {}).items():
            key = f"{binary_id}::{test_name}"
            status = info.get("filter-match", {}).get("status")
            out[key] = (status, bool(info.get("ignored")))
    return out


def _show_shared_rustdoc_group(
    profile: str, *, audited_consumers_only: bool = False
) -> str:
    """Ask nextest which tests receive the inherited group, without a test filter."""
    cmd = [
        "cargo",
        "nextest",
        "show-config",
        "test-groups",
        "--color",
        "never",
    ]
    if audited_consumers_only:
        cmd.extend(
            [
                "-p",
                "chelis-compiler-api",
                "-p",
                "chelis-python",
                "--test",
                "capacity_census_wire",
                "--test",
                "capacity_census_bindings",
            ]
        )
    else:
        cmd.append("--workspace")
    cmd.extend(
        [
            "--profile",
            profile,
            "--ignore-default-filter",
            "--groups",
            SHARED_RUSTDOC_GROUP,
        ]
    )
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=_cargo_environment(),
        capture_output=True,
        text=True,
        timeout=900,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"nextest group resolution failed for {profile!r} "
            f"(exit {result.returncode}): {result.stderr[-2000:]}"
        )
    return result.stdout


def _resolved_shared_rustdoc_members(output: str) -> set[str]:
    """Parse nextest's resolved binary/test membership, excluding filter prose."""
    members: set[str] = set()
    binary = None
    for line in output.splitlines():
        binary_match = re.fullmatch(r"      ([^ ]+):", line)
        if binary_match is not None:
            binary = binary_match.group(1)
            continue
        test_match = re.fullmatch(r"          ([^ ]+)", line)
        if test_match is not None and binary is not None:
            members.add(f"{binary}::{test_match.group(1)}")
    return members


@unittest.skipUnless(
    _have_nextest(), "cargo nextest unavailable; skipping set-math oracle"
)
class ProfilePartitionTests(unittest.TestCase):
    """The real oracle: `cargo nextest list` set math across profiles.

    The invariant: every non-ignored test is in the workspace `ci` profile,
    the `nightly` profile, or an explicitly selected required-oracle binary.
    `#[ignore]`-d tests are intentionally outside this partition.
    """

    @classmethod
    def setUpClass(cls):
        cls.ci = _list_profile("ci")
        cls.nightly = _list_profile("nightly")
        cls.full = _list_profile(None)
        cls.dtype_flat = _list_filterset(
            dtype_oracle_manifest.flattened_filter(sys.executable)
        )
        cls.dtype_owners = {
            owner: _list_filterset(
                dtype_oracle_manifest.owner_filter(owner, sys.executable)
            )
            for owner in dtype_oracle_manifest.OWNERS
        }

    def test_fast_selection_is_a_subset_of_unfiltered_nightly(self):
        from scripts import ci_test_targets
        metadata = json.loads(subprocess.run(
            ["cargo", "metadata", "--no-deps", "--format-version", "1", "--locked"],
            cwd=REPO_ROOT, check=True, capture_output=True, text=True,
        ).stdout)
        selected = ci_test_targets.read_targets(REPO_ROOT / ".config/ci-test-targets.toml")
        args = ci_test_targets.cargo_args(metadata, selected)
        def listing(selection):
            result = subprocess.run(
                ["cargo", "nextest", "list", *selection, "--profile", "ci-full",
                 "--ignore-default-filter", "--message-format", "json"],
                cwd=REPO_ROOT, env=_cargo_environment(), check=True,
                capture_output=True, text=True, timeout=900,
            )
            return json.loads(result.stdout)
        fast = listing(args)
        ci_test_targets.validate_listing(fast, metadata, selected)
        full = listing(["--workspace"])
        def active(data):
            return {(binary, name) for binary, suite in data["rust-suites"].items()
                    for name, info in suite["testcases"].items() if not info["ignored"]}
        self.assertTrue(active(fast))
        self.assertLess(active(fast), active(full))
        self.assertTrue(all(info["filter-match"]["status"] == "matches"
                            for suite in full["rust-suites"].values()
                            for info in suite["testcases"].values() if not info["ignored"]))

    def _sets(self):
        ci_matches = {k for k, (s, _) in self.ci.items() if s == "matches"}
        nightly_matches = {
            k for k, (s, _) in self.nightly.items() if s == "matches"
        }
        # The true universe: every test that appears under any profile's
        # listing. `nightly`'s json only enumerates binaries its filter
        # can match, so union all three to get the full picture.
        universe = set(self.ci) | set(self.nightly) | set(self.full)
        ignored = {
            k
            for src in (self.ci, self.nightly, self.full)
            for k, (_s, ign) in src.items()
            if ign
        }
        non_ignored = universe - ignored
        oracle_matches = {
            key
            for key in non_ignored
            if any(
                key.startswith(f"{binary_id}::")
                for binary_id in ORACLE_OWNED_BINARY_IDS
            )
        }
        return ci_matches, nightly_matches, oracle_matches, non_ignored

    def test_ci_and_nightly_are_disjoint(self):
        ci_matches, nightly_matches, _oracle_matches, _ = self._sets()
        overlap = ci_matches & nightly_matches
        self.assertEqual(
            overlap,
            set(),
            f"{len(overlap)} test(s) are in BOTH the manual `ci` profile and "
            f"the manual `nightly` profile: "
            f"{sorted(overlap)[:20]}",
        )

    def test_no_non_ignored_test_falls_into_neither_profile(self):
        ci_matches, nightly_matches, oracle_matches, non_ignored = self._sets()
        gap = non_ignored - ci_matches - nightly_matches - oracle_matches
        self.assertEqual(
            gap,
            set(),
            f"{len(gap)} non-ignored test(s) are on neither the workspace, "
            f"nightly, nor manual oracle selections -- they silently stopped running "
            f"(dropped coverage): {sorted(gap)[:20]}",
        )

    def test_profiles_plus_delegated_binaries_cover_every_non_ignored_test(self):
        # Belt-and-braces coverage statement; this is not a disjoint
        # three-lane partition because the dtype oracle overlaps `ci` elsewhere.
        ci_matches, nightly_matches, oracle_matches, non_ignored = self._sets()
        covered = (ci_matches | nightly_matches | oracle_matches) & non_ignored
        self.assertEqual(
            covered,
            non_ignored,
            "the union of workspace, nightly, and required-oracle tests does "
            "not cover every non-ignored test",
        )

    def test_nightly_set_is_nonempty(self):
        # A `nightly` profile that matched nothing would mean the split
        # silently dropped the entire heavy-e2e suite.
        _ci, nightly_matches, _oracle_matches, _ = self._sets()
        self.assertGreater(
            len(nightly_matches),
            0,
            "the `nightly` profile matched zero tests; the heavy-e2e "
            "suite is not running anywhere",
        )

    def test_no_nightly_test_is_ignored(self):
        # A heavy-e2e test that is also `#[ignore]`-d would never run --
        # not on the per-PR gate (excluded) and not nightly (ignored).
        nightly_ignored = {
            k
            for k, (s, ign) in self.nightly.items()
            if s == "matches" and ign
        }
        self.assertEqual(
            nightly_ignored,
            set(),
            f"{len(nightly_ignored)} `nightly`-selected test(s) are also "
            f"`#[ignore]`-d, so they never run: {sorted(nightly_ignored)}",
        )

    def test_nightly_contention_cases_are_nightly_owned(self):
        # The generalization-lane half of this contract lives in
        # `GeneralizationPartitionTests`, which lists the feature-enabled
        # configuration where it is already built.
        for test_id in (NIGHTLY_RECURSIVE_TEST, NIGHTLY_CACHE_CONCURRENCY_TEST):
            with self.subTest(test_id=test_id):
                self.assertIn(test_id, self.nightly)
                nightly_status, ignored = self.nightly[test_id]
                self.assertEqual(nightly_status, "matches")
                self.assertFalse(ignored)

    def test_flattened_dtype_filter_is_the_exact_union_of_phase_owners(self):
        flattened = {
            key
            for key, (status, ignored) in self.dtype_flat.items()
            if status == "matches" and not ignored
        }
        owner_sets = {
            owner: {
                key
                for key, (status, ignored) in listing.items()
                if status == "matches" and not ignored
            }
            for owner, listing in self.dtype_owners.items()
        }
        inherited_union = set().union(*owner_sets.values())
        self.assertEqual(flattened, inherited_union)
        self.assertGreater(len(flattened), 0, "flattened dtype oracle is empty")
        ownership_count = sum(len(selected) for selected in owner_sets.values())
        self.assertGreater(
            ownership_count,
            len(flattened),
            "the control corpus no longer contains any inherited duplicate "
            "selection, so flattening has no executable duplication to remove",
        )

    def test_every_profile_serializes_exactly_the_shared_target_owners(self):
        profiles = tuple(tomllib.loads(NEXTEST_TOML.read_text())["profile"])
        self.assertIn("builtin-atom-closure", profiles)
        with mock.patch.dict(os.environ, {"CARGO_TERM_COLOR": "always"}):
            for profile in profiles:
                with self.subTest(profile=profile):
                    output = _show_shared_rustdoc_group(profile)
                    self.assertRegex(
                        output,
                        rf"(?m)^group: {SHARED_RUSTDOC_GROUP} "
                        r"\(max threads = 1\)$",
                    )
                    self.assertEqual(
                        _resolved_shared_rustdoc_members(output),
                        set(SHARED_RUSTDOC_OWNER_TESTS),
                    )

@unittest.skipUnless(
    _have_nextest(), "cargo nextest unavailable; skipping generalization census"
)
class GeneralizationPartitionTests(unittest.TestCase):
    """Set math for the explicit generalization lane's selection.

    This class lists `--features chelis-types/generalize-sweep-oracle`, a
    different compiled configuration from every `ProfilePartitionTests`
    listing, so CI runs it on generalization shard 1 where that build is
    warm. The nightly-side half of the contention contract stays in
    `ProfilePartitionTests.test_nightly_contention_cases_are_nightly_owned`.
    """

    @classmethod
    def setUpClass(cls):
        cls.generalization_pr = _list_generalization_pr()

    def test_generalization_lane_selects_a_nonempty_corpus(self):
        selected = {
            key
            for key, (status, ignored) in self.generalization_pr.items()
            if status == "matches" and not ignored
        }
        self.assertGreater(len(selected), 0, "generalization lane is empty")

    def test_nightly_contention_cases_are_excluded_from_generalization_pr(self):
        recursive_status, recursive_ignored = self.generalization_pr[
            NIGHTLY_RECURSIVE_TEST
        ]
        self.assertEqual(recursive_status, "mismatch")
        self.assertFalse(recursive_ignored)
        self.assertNotIn(
            NIGHTLY_CACHE_CONCURRENCY_TEST,
            self.generalization_pr,
            "the generalization PR selector must exclude the entire "
            "nightly-owned cache-concurrency binary",
        )


if __name__ == "__main__":
    unittest.main()
